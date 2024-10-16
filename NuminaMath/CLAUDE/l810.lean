import Mathlib

namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l810_81099

theorem cubic_sum_theorem (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_eq : (p^3 - 12)/p = (q^3 - 12)/q ∧ (q^3 - 12)/q = (r^3 - 12)/r) : 
  p^3 + q^3 + r^3 = -36 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l810_81099


namespace NUMINAMATH_CALUDE_martha_juice_bottles_l810_81050

theorem martha_juice_bottles (initial_pantry : ℕ) (bought : ℕ) (consumed : ℕ) (final_total : ℕ) 
  (h1 : initial_pantry = 4)
  (h2 : bought = 5)
  (h3 : consumed = 3)
  (h4 : final_total = 10) :
  ∃ (initial_fridge : ℕ), 
    initial_fridge + initial_pantry + bought - consumed = final_total ∧ 
    initial_fridge = 4 := by
  sorry

end NUMINAMATH_CALUDE_martha_juice_bottles_l810_81050


namespace NUMINAMATH_CALUDE_special_function_is_identity_l810_81069

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧ ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

/-- Theorem stating that any function satisfying the conditions is the identity function -/
theorem special_function_is_identity (f : ℝ → ℝ) (h : special_function f) : 
  ∀ x : ℝ, f x = x := by sorry

end NUMINAMATH_CALUDE_special_function_is_identity_l810_81069


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_pairs_of_roots_l810_81034

/-- Given a quintic polynomial x^5 + 10x^4 + 20x^3 + 15x^2 + 6x + 3, 
    this theorem states that the sum of reciprocals of products of pairs of its roots is 20/3 -/
theorem sum_of_reciprocal_pairs_of_roots (p q r s t : ℂ) : 
  p^5 + 10*p^4 + 20*p^3 + 15*p^2 + 6*p + 3 = 0 →
  q^5 + 10*q^4 + 20*q^3 + 15*q^2 + 6*q + 3 = 0 →
  r^5 + 10*r^4 + 20*r^3 + 15*r^2 + 6*r + 3 = 0 →
  s^5 + 10*s^4 + 20*s^3 + 15*s^2 + 6*s + 3 = 0 →
  t^5 + 10*t^4 + 20*t^3 + 15*t^2 + 6*t + 3 = 0 →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_pairs_of_roots_l810_81034


namespace NUMINAMATH_CALUDE_investment_rate_calculation_l810_81046

/-- Prove that if Rs. 1600 is divided into two parts, where one part (P1) is Rs. 1100
    invested at 6% and the other part (P2) is the remainder, and the total annual
    interest from both parts is Rs. 85, then P2 must be invested at 3.8%. -/
theorem investment_rate_calculation (total : ℝ) (p1 : ℝ) (p2 : ℝ) (r1 : ℝ) (total_interest : ℝ) :
  total = 1600 →
  p1 = 1100 →
  p2 = total - p1 →
  r1 = 6 →
  total_interest = 85 →
  p1 * r1 / 100 + p2 * (total_interest - p1 * r1 / 100) / p2 = total_interest →
  (total_interest - p1 * r1 / 100) / p2 * 100 = 3.8 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_calculation_l810_81046


namespace NUMINAMATH_CALUDE_cube_diagonal_pairs_60_degrees_l810_81015

/-- A regular hexahedron (cube) -/
structure Cube where
  /-- Number of faces in a cube -/
  faces : ℕ
  /-- Number of diagonals per face -/
  diagonals_per_face : ℕ
  /-- Total number of face diagonals -/
  total_diagonals : ℕ
  /-- Total number of possible diagonal pairs -/
  total_pairs : ℕ
  /-- Number of diagonal pairs that don't form a 60° angle -/
  non_60_pairs : ℕ

/-- The number of pairs of face diagonals in a cube that form a 60° angle -/
def pairs_forming_60_degrees (c : Cube) : ℕ :=
  c.total_pairs - c.non_60_pairs

/-- Theorem stating that in a regular hexahedron (cube), 
    the number of pairs of face diagonals that form a 60° angle is 48 -/
theorem cube_diagonal_pairs_60_degrees (c : Cube) 
  (h1 : c.faces = 6)
  (h2 : c.diagonals_per_face = 2)
  (h3 : c.total_diagonals = 12)
  (h4 : c.total_pairs = 66)
  (h5 : c.non_60_pairs = 18) :
  pairs_forming_60_degrees c = 48 := by
  sorry

end NUMINAMATH_CALUDE_cube_diagonal_pairs_60_degrees_l810_81015


namespace NUMINAMATH_CALUDE_find_subtracted_number_l810_81095

theorem find_subtracted_number (x N : ℝ) (h1 : 3 * x = (N - x) + 26) (h2 : x = 22) : N = 62 := by
  sorry

end NUMINAMATH_CALUDE_find_subtracted_number_l810_81095


namespace NUMINAMATH_CALUDE_curve_intersects_median_unique_point_l810_81018

/-- Given non-collinear points A, B, C with complex coordinates, 
    prove that the curve intersects the median of triangle ABC at a unique point. -/
theorem curve_intersects_median_unique_point 
  (a b c : ℝ) 
  (h_non_collinear : a + c ≠ 2*b) : 
  ∃! p : ℂ, 
    (∃ t : ℝ, p = Complex.I * a * (Real.cos t)^4 + 
               (1/2 + Complex.I * b) * 2 * (Real.cos t)^2 * (Real.sin t)^2 + 
               (1 + Complex.I * c) * (Real.sin t)^4) ∧ 
    (p.re = 1/2 ∧ p.im = (a + 2*b + c) / 4) := by
  sorry


end NUMINAMATH_CALUDE_curve_intersects_median_unique_point_l810_81018


namespace NUMINAMATH_CALUDE_projectile_meeting_time_l810_81074

theorem projectile_meeting_time : 
  let initial_distance : ℝ := 2520
  let speed1 : ℝ := 432
  let speed2 : ℝ := 576
  let combined_speed : ℝ := speed1 + speed2
  let time_hours : ℝ := initial_distance / combined_speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 150 := by sorry

end NUMINAMATH_CALUDE_projectile_meeting_time_l810_81074


namespace NUMINAMATH_CALUDE_log_equality_implies_n_fifth_power_l810_81052

theorem log_equality_implies_n_fifth_power (n : ℝ) :
  n > 0 →
  (Real.log (675 * Real.sqrt 3)) / (Real.log (3 * n)) = (Real.log 75) / (Real.log n) →
  n^5 = 5625 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_n_fifth_power_l810_81052


namespace NUMINAMATH_CALUDE_min_value_of_max_expression_l810_81085

theorem min_value_of_max_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (M : ℝ), M = max (1 / (a * c) + b) (max (1 / a + b * c) (a / b + c)) ∧ M ≥ 2 ∧ 
  (∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    max (1 / (a' * c') + b') (max (1 / a' + b' * c') (a' / b' + c')) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_max_expression_l810_81085


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l810_81023

/-- A cubic polynomial with real coefficients. -/
def CubicPolynomial := ℝ → ℝ

/-- The property that a cubic polynomial satisfies the given conditions. -/
def SatisfiesConditions (g : CubicPolynomial) : Prop :=
  ∃ (a b c d : ℝ), 
    (∀ x, g x = a * x^3 + b * x^2 + c * x + d) ∧
    (|g (-2)| = 6) ∧ (|g 0| = 6) ∧ (|g 1| = 6) ∧ (|g 4| = 6)

/-- The theorem stating that if a cubic polynomial satisfies the conditions, then |g(-1)| = 27/2. -/
theorem cubic_polynomial_property (g : CubicPolynomial) 
  (h : SatisfiesConditions g) : |g (-1)| = 27/2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l810_81023


namespace NUMINAMATH_CALUDE_tank_fill_time_is_30_l810_81073

/-- Represents the time it takes to fill a tank given two pipes with different fill/empty rates and a specific operating scenario. -/
def tank_fill_time (fill_rate_A : ℚ) (empty_rate_B : ℚ) (both_open_time : ℚ) : ℚ :=
  let net_fill_rate := fill_rate_A - empty_rate_B
  let filled_portion := net_fill_rate * both_open_time
  let remaining_portion := 1 - filled_portion
  both_open_time + remaining_portion / fill_rate_A

/-- Theorem stating that under the given conditions, the tank will be filled in 30 minutes. -/
theorem tank_fill_time_is_30 :
  tank_fill_time (1/16) (1/24) 21 = 30 :=
by sorry

end NUMINAMATH_CALUDE_tank_fill_time_is_30_l810_81073


namespace NUMINAMATH_CALUDE_paper_area_problem_l810_81019

theorem paper_area_problem (x : ℕ) : 
  (2 * 11 * 11 = 2 * x * 11 + 100) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_paper_area_problem_l810_81019


namespace NUMINAMATH_CALUDE_fraction_equals_decimal_l810_81055

theorem fraction_equals_decimal : (8 : ℚ) / (4 * 25) = 0.08 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_decimal_l810_81055


namespace NUMINAMATH_CALUDE_least_number_of_cans_l810_81080

def maaza_liters : ℕ := 40
def pepsi_liters : ℕ := 144
def sprite_liters : ℕ := 368

theorem least_number_of_cans : 
  ∃ (can_size : ℕ), 
    can_size > 0 ∧
    maaza_liters % can_size = 0 ∧
    pepsi_liters % can_size = 0 ∧
    sprite_liters % can_size = 0 ∧
    (maaza_liters / can_size + pepsi_liters / can_size + sprite_liters / can_size = 69) ∧
    ∀ (other_size : ℕ), 
      other_size > 0 →
      maaza_liters % other_size = 0 →
      pepsi_liters % other_size = 0 →
      sprite_liters % other_size = 0 →
      (maaza_liters / other_size + pepsi_liters / other_size + sprite_liters / other_size ≥ 69) :=
by
  sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l810_81080


namespace NUMINAMATH_CALUDE_trihedral_angle_range_a_trihedral_angle_range_b_l810_81025

-- Define a trihedral angle
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real
  sum_less_than_360 : α + β + γ < 360
  each_less_than_sum_of_others : α < β + γ ∧ β < α + γ ∧ γ < α + β

-- Theorem for part (a)
theorem trihedral_angle_range_a (t : TrihedralAngle) (h1 : t.β = 70) (h2 : t.γ = 100) :
  30 < t.α ∧ t.α < 170 := by sorry

-- Theorem for part (b)
theorem trihedral_angle_range_b (t : TrihedralAngle) (h1 : t.β = 130) (h2 : t.γ = 150) :
  20 < t.α ∧ t.α < 80 := by sorry

end NUMINAMATH_CALUDE_trihedral_angle_range_a_trihedral_angle_range_b_l810_81025


namespace NUMINAMATH_CALUDE_trajectory_of_P_l810_81014

-- Define the line l
def line_l (θ : ℝ) (x y : ℝ) : Prop := x * Real.cos θ + y * Real.sin θ = 1

-- Define the perpendicularity condition
def perpendicular_to_l (x y : ℝ) : Prop := ∃ θ, line_l θ x y

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem trajectory_of_P : ∀ x y : ℝ, perpendicular_to_l x y → x^2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l810_81014


namespace NUMINAMATH_CALUDE_average_speed_calculation_l810_81033

theorem average_speed_calculation (local_distance : ℝ) (local_speed : ℝ) 
  (highway_distance : ℝ) (highway_speed : ℝ) : 
  local_distance = 40 ∧ local_speed = 20 ∧ highway_distance = 180 ∧ highway_speed = 60 →
  (local_distance + highway_distance) / ((local_distance / local_speed) + (highway_distance / highway_speed)) = 44 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l810_81033


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l810_81049

def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l810_81049


namespace NUMINAMATH_CALUDE_complex_power_equivalence_l810_81003

theorem complex_power_equivalence :
  (Complex.exp (Complex.I * Real.pi * (125 / 180)))^28 =
  Complex.exp (Complex.I * Real.pi * (140 / 180)) :=
by sorry

end NUMINAMATH_CALUDE_complex_power_equivalence_l810_81003


namespace NUMINAMATH_CALUDE_scientific_notation_conversion_l810_81066

theorem scientific_notation_conversion :
  (4.6 : ℝ) * (10 ^ 8) = 460000000 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_conversion_l810_81066


namespace NUMINAMATH_CALUDE_books_bound_calculation_remaining_paper_condition_l810_81010

/-- Represents the number of books bound in a bookbinding workshop. -/
def books_bound (initial_white : ℕ) (initial_colored : ℕ) : ℕ :=
  initial_white - (initial_colored - initial_white)

/-- Theorem stating the number of books bound given the initial quantities and conditions. -/
theorem books_bound_calculation :
  let initial_white := 92
  let initial_colored := 135
  books_bound initial_white initial_colored = 178 :=
by
  sorry

/-- Theorem verifying the remaining paper condition after binding. -/
theorem remaining_paper_condition (initial_white initial_colored : ℕ) :
  let bound := books_bound initial_white initial_colored
  initial_white - bound = (initial_colored - bound) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_books_bound_calculation_remaining_paper_condition_l810_81010


namespace NUMINAMATH_CALUDE_hemisphere_volume_l810_81002

theorem hemisphere_volume (diameter : ℝ) (volume : ℝ) : 
  diameter = 8 → volume = (128 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_volume_l810_81002


namespace NUMINAMATH_CALUDE_lillys_fish_l810_81090

theorem lillys_fish (rosys_fish : ℕ) (total_fish : ℕ) (h1 : rosys_fish = 14) (h2 : total_fish = 24) :
  total_fish - rosys_fish = 10 := by
sorry

end NUMINAMATH_CALUDE_lillys_fish_l810_81090


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l810_81036

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  ((2 + i) * (3 - 4*i)) / (2 - i) = 5 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l810_81036


namespace NUMINAMATH_CALUDE_evaluate_expression_l810_81072

theorem evaluate_expression : (1500^2 : ℚ) / (306^2 - 294^2) = 312.5 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l810_81072


namespace NUMINAMATH_CALUDE_every_real_has_cube_root_l810_81006

theorem every_real_has_cube_root : 
  ∀ y : ℝ, ∃ x : ℝ, x^3 = y := by sorry

end NUMINAMATH_CALUDE_every_real_has_cube_root_l810_81006


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l810_81089

theorem largest_multiple_of_15_under_500 :
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l810_81089


namespace NUMINAMATH_CALUDE_benny_baseball_gear_spending_l810_81013

/-- The amount Benny spent on baseball gear --/
def amount_spent (initial_amount left_over : ℕ) : ℕ :=
  initial_amount - left_over

/-- Theorem: Benny spent $47 on baseball gear --/
theorem benny_baseball_gear_spending :
  amount_spent 79 32 = 47 := by
  sorry

end NUMINAMATH_CALUDE_benny_baseball_gear_spending_l810_81013


namespace NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l810_81057

theorem sum_of_special_primes_is_prime (P Q : ℕ) : 
  P > 0 ∧ Q > 0 ∧ 
  Nat.Prime P ∧ Nat.Prime Q ∧ Nat.Prime (P - Q) ∧ Nat.Prime (P + Q) →
  Nat.Prime (P + Q + (P - Q) + P + Q) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l810_81057


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l810_81029

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- complementary angles
  a = 3 * b →   -- ratio of 3:1
  |a - b| = 45  -- positive difference
  := by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l810_81029


namespace NUMINAMATH_CALUDE_power_of_81_l810_81007

theorem power_of_81 : (81 : ℝ) ^ (5/2) = 59049 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_l810_81007


namespace NUMINAMATH_CALUDE_smallest_difference_l810_81030

def digits : List Nat := [2, 4, 5, 6, 9]

def is_valid_arrangement (a b : Nat) : Prop :=
  ∃ (x y z u v : Nat),
    x ∈ digits ∧ y ∈ digits ∧ z ∈ digits ∧ u ∈ digits ∧ v ∈ digits ∧
    x ≠ y ∧ x ≠ z ∧ x ≠ u ∧ x ≠ v ∧
    y ≠ z ∧ y ≠ u ∧ y ≠ v ∧
    z ≠ u ∧ z ≠ v ∧
    u ≠ v ∧
    a = 100 * x + 10 * y + z ∧
    b = 10 * u + v

theorem smallest_difference :
  ∀ a b : Nat,
    is_valid_arrangement a b →
    a > b →
    a - b ≥ 149 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_l810_81030


namespace NUMINAMATH_CALUDE_pie_eating_contest_l810_81020

theorem pie_eating_contest (first_student second_student : ℚ) : 
  first_student = 8/9 → second_student = 5/6 → first_student - second_student = 1/18 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l810_81020


namespace NUMINAMATH_CALUDE_reeya_average_score_l810_81093

def reeya_scores : List ℝ := [65, 67, 76, 80, 95]

theorem reeya_average_score :
  (reeya_scores.sum / reeya_scores.length : ℝ) = 76.6 := by
  sorry

end NUMINAMATH_CALUDE_reeya_average_score_l810_81093


namespace NUMINAMATH_CALUDE_youtube_video_length_l810_81012

theorem youtube_video_length (x : ℝ) 
  (h1 : 6 * x + 6 * (x / 2) = 900) : x = 100 := by
  sorry

end NUMINAMATH_CALUDE_youtube_video_length_l810_81012


namespace NUMINAMATH_CALUDE_x_plus_y_equals_483_l810_81041

theorem x_plus_y_equals_483 (x y : ℝ) : 
  x = 300 * (1 - 0.3) → 
  y = x * (1 + 0.3) → 
  x + y = 483 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_483_l810_81041


namespace NUMINAMATH_CALUDE_darren_boxes_correct_l810_81004

/-- The number of crackers in each box -/
def crackers_per_box : ℕ := 24

/-- The total number of crackers bought by both Darren and Calvin -/
def total_crackers : ℕ := 264

/-- The number of boxes Darren bought -/
def darren_boxes : ℕ := 4

theorem darren_boxes_correct :
  ∃ (calvin_boxes : ℕ),
    calvin_boxes = 2 * darren_boxes - 1 ∧
    crackers_per_box * (darren_boxes + calvin_boxes) = total_crackers :=
by sorry

end NUMINAMATH_CALUDE_darren_boxes_correct_l810_81004


namespace NUMINAMATH_CALUDE_box_properties_l810_81086

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  white : ℕ
  red : ℕ
  yellow : ℕ

/-- The given ball counts in the box -/
def box : BallCounts := { white := 1, red := 2, yellow := 3 }

/-- The total number of balls in the box -/
def totalBalls (b : BallCounts) : ℕ := b.white + b.red + b.yellow

/-- The number of possible outcomes when drawing 1 ball -/
def possibleOutcomes (b : BallCounts) : ℕ := 
  (if b.white > 0 then 1 else 0) + 
  (if b.red > 0 then 1 else 0) + 
  (if b.yellow > 0 then 1 else 0)

/-- The probability of drawing a ball of a specific color -/
def probability (b : BallCounts) (color : ℕ) : ℚ :=
  color / (totalBalls b : ℚ)

theorem box_properties : 
  (possibleOutcomes box = 3) ∧ 
  (probability box box.yellow > probability box box.red ∧ 
   probability box box.yellow > probability box box.white) ∧
  (probability box box.white + probability box box.yellow = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_box_properties_l810_81086


namespace NUMINAMATH_CALUDE_tan_value_for_given_sum_l810_81031

theorem tan_value_for_given_sum (x : ℝ) 
  (h1 : Real.sin x + Real.cos x = 1/5)
  (h2 : 0 ≤ x ∧ x < π) : 
  Real.tan x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_for_given_sum_l810_81031


namespace NUMINAMATH_CALUDE_banana_arrangements_l810_81011

def word := "BANANA"

def letter_count : Nat := word.length

def b_count : Nat := (word.toList.filter (· == 'B')).length
def n_count : Nat := (word.toList.filter (· == 'N')).length
def a_count : Nat := (word.toList.filter (· == 'A')).length

def distinct_arrangements : Nat := letter_count.factorial / (b_count.factorial * n_count.factorial * a_count.factorial)

theorem banana_arrangements :
  letter_count = 6 ∧ b_count = 1 ∧ n_count = 2 ∧ a_count = 3 →
  distinct_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l810_81011


namespace NUMINAMATH_CALUDE_difference_of_squares_l810_81024

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l810_81024


namespace NUMINAMATH_CALUDE_ball_distribution_with_constraint_l810_81040

theorem ball_distribution_with_constraint (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 5 → k = 3 → m = 2 →
  (n.pow k : ℕ) - (k - 1).pow n - n * (k - 1).pow (n - 1) = 131 :=
sorry

end NUMINAMATH_CALUDE_ball_distribution_with_constraint_l810_81040


namespace NUMINAMATH_CALUDE_solve_bus_problem_l810_81059

def bus_problem (first_stop : ℕ) (second_stop_off : ℕ) (second_stop_on : ℕ) (third_stop_off : ℕ) (final_count : ℕ) : Prop :=
  let after_first := first_stop
  let after_second := after_first - second_stop_off + second_stop_on
  let before_third_on := after_second - third_stop_off
  ∃ (third_stop_on : ℕ), before_third_on + third_stop_on = final_count ∧ third_stop_on = 4

theorem solve_bus_problem :
  bus_problem 7 3 5 2 11 :=
by
  sorry

#check solve_bus_problem

end NUMINAMATH_CALUDE_solve_bus_problem_l810_81059


namespace NUMINAMATH_CALUDE_union_of_sets_l810_81027

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 3, 4}
  A ∪ B = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l810_81027


namespace NUMINAMATH_CALUDE_ship_grain_problem_l810_81062

theorem ship_grain_problem (spilled_grain : ℕ) (remaining_grain : ℕ) 
  (h1 : spilled_grain = 49952) (h2 : remaining_grain = 918) : 
  spilled_grain + remaining_grain = 50870 := by
  sorry

end NUMINAMATH_CALUDE_ship_grain_problem_l810_81062


namespace NUMINAMATH_CALUDE_rational_square_fractional_parts_l810_81082

def fractional_part (x : ℚ) : ℚ :=
  x - ↑(⌊x⌋)

theorem rational_square_fractional_parts (S : Set ℚ) :
  (∀ x ∈ S, fractional_part x ∈ {y | ∃ z ∈ S, fractional_part (z^2) = y}) →
  (∀ x ∈ S, fractional_part (x^2) ∈ {y | ∃ z ∈ S, fractional_part z = y}) →
  ∀ x ∈ S, ∃ n : ℤ, x = n := by
  sorry

end NUMINAMATH_CALUDE_rational_square_fractional_parts_l810_81082


namespace NUMINAMATH_CALUDE_sqrt_inequality_l810_81054

theorem sqrt_inequality (a : ℝ) (h : a > 1) : Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l810_81054


namespace NUMINAMATH_CALUDE_no_prime_multiple_of_four_in_range_l810_81043

theorem no_prime_multiple_of_four_in_range : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 50 → ¬(4 ∣ n ∧ Nat.Prime n ∧ n > 10) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_multiple_of_four_in_range_l810_81043


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l810_81058

theorem consecutive_integers_problem (x y z : ℤ) : 
  (y = z + 1) →
  (x = z + 2) →
  (x > y) →
  (y > z) →
  (2 * x + 3 * y + 3 * z = 5 * y + 8) →
  (z = 2) →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l810_81058


namespace NUMINAMATH_CALUDE_property_tax_increase_l810_81079

/-- Represents the property tax increase in Township K --/
theorem property_tax_increase 
  (tax_rate : ℝ) 
  (initial_value : ℝ) 
  (new_value : ℝ) 
  (h1 : tax_rate = 0.1)
  (h2 : initial_value = 20000)
  (h3 : new_value = 28000) : 
  new_value * tax_rate - initial_value * tax_rate = 800 := by
  sorry

#check property_tax_increase

end NUMINAMATH_CALUDE_property_tax_increase_l810_81079


namespace NUMINAMATH_CALUDE_fourth_machine_works_twelve_hours_l810_81016

/-- Represents a factory with machines producing material. -/
structure Factory where
  num_original_machines : ℕ
  hours_per_day_original : ℕ
  production_rate : ℕ
  price_per_kg : ℕ
  total_revenue : ℕ

/-- Calculates the hours worked by the fourth machine. -/
def fourth_machine_hours (f : Factory) : ℕ :=
  let original_production := f.num_original_machines * f.hours_per_day_original * f.production_rate
  let original_revenue := original_production * f.price_per_kg
  let fourth_machine_revenue := f.total_revenue - original_revenue
  let fourth_machine_production := fourth_machine_revenue / f.price_per_kg
  fourth_machine_production / f.production_rate

/-- Theorem stating the fourth machine works 12 hours a day. -/
theorem fourth_machine_works_twelve_hours (f : Factory) 
  (h1 : f.num_original_machines = 3)
  (h2 : f.hours_per_day_original = 23)
  (h3 : f.production_rate = 2)
  (h4 : f.price_per_kg = 50)
  (h5 : f.total_revenue = 8100) :
  fourth_machine_hours f = 12 := by
  sorry

end NUMINAMATH_CALUDE_fourth_machine_works_twelve_hours_l810_81016


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l810_81026

theorem smallest_integer_satisfying_inequality :
  ∃ x : ℤ, (∀ y : ℤ, 8 - 7 * y ≥ 4 * y - 3 → x ≤ y) ∧ (8 - 7 * x ≥ 4 * x - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l810_81026


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l810_81039

theorem polynomial_multiplication (x : ℝ) :
  (3 * x^2 - 2 * x + 4) * (-4 * x^2 + 3 * x - 6) =
  -12 * x^4 + 17 * x^3 - 40 * x^2 + 24 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l810_81039


namespace NUMINAMATH_CALUDE_min_d_value_l810_81005

theorem min_d_value (a b c d : ℕ+) (h_order : a < b ∧ b < c ∧ c < d) 
  (h_unique : ∃! (x y : ℝ), x + 2*y = 2023 ∧ y = |x - a| + |x - b| + |x - c| + |x - d|) :
  d ≥ 1010 ∧ ∃ (a' b' c' : ℕ+), a' < b' ∧ b' < c' ∧ c' < 1010 ∧
    ∃! (x y : ℝ), x + 2*y = 2023 ∧ y = |x - a'| + |x - b'| + |x - c'| + |x - 1010| :=
by sorry

end NUMINAMATH_CALUDE_min_d_value_l810_81005


namespace NUMINAMATH_CALUDE_percentage_of_b_l810_81037

theorem percentage_of_b (a b c : ℝ) (h1 : 8 = 0.02 * a) (h2 : c = b / a) : 
  ∃ p : ℝ, p * b = 2 ∧ p = 0.005 := by sorry

end NUMINAMATH_CALUDE_percentage_of_b_l810_81037


namespace NUMINAMATH_CALUDE_uncorrelated_variables_l810_81070

/-- Represents a variable in our correlation problem -/
structure Variable where
  name : String

/-- Represents a pair of variables -/
structure VariablePair where
  var1 : Variable
  var2 : Variable

/-- Defines what it means for two variables to be correlated -/
def are_correlated (pair : VariablePair) : Prop :=
  sorry  -- The actual definition would go here

/-- The list of variable pairs we're considering -/
def variable_pairs : List VariablePair :=
  [ { var1 := { name := "Grain yield" }, var2 := { name := "Amount of fertilizer used" } },
    { var1 := { name := "College entrance examination scores" }, var2 := { name := "Time spent on review" } },
    { var1 := { name := "Sales of goods" }, var2 := { name := "Advertising expenses" } },
    { var1 := { name := "Number of books sold at fixed price" }, var2 := { name := "Sales revenue" } } ]

/-- The theorem we want to prove -/
theorem uncorrelated_variables : 
  ∃ (pair : VariablePair), pair ∈ variable_pairs ∧ ¬(are_correlated pair) :=
sorry


end NUMINAMATH_CALUDE_uncorrelated_variables_l810_81070


namespace NUMINAMATH_CALUDE_rectangle_divided_by_line_l810_81096

/-- 
Given a rectangle with vertices (1, 0), (x, 0), (1, 2), and (x, 2),
if a line passing through the origin (0, 0) divides the rectangle into two identical quadrilaterals
and has a slope of 1/3, then x = 5.
-/
theorem rectangle_divided_by_line (x : ℝ) : 
  (∃ l : Set (ℝ × ℝ), 
    -- l is a line passing through the origin
    (0, 0) ∈ l ∧
    -- l divides the rectangle into two identical quadrilaterals
    (∃ m : ℝ × ℝ, m ∈ l ∧ m.1 = (1 + x) / 2 ∧ m.2 = 1) ∧
    -- The slope of l is 1/3
    (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q → (q.2 - p.2) / (q.1 - p.1) = 1/3)) →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_divided_by_line_l810_81096


namespace NUMINAMATH_CALUDE_track_circumference_l810_81097

/-- The circumference of a circular track given specific meeting conditions of two travelers -/
theorem track_circumference : 
  ∀ (circumference : ℝ) 
    (speed_A speed_B : ℝ) 
    (first_meeting second_meeting : ℝ),
  speed_A > 0 →
  speed_B > 0 →
  first_meeting = 150 →
  second_meeting = circumference - 90 →
  first_meeting / (circumference / 2 - first_meeting) = 
    (circumference / 2 + 90) / (circumference - 90) →
  circumference = 720 := by
sorry

end NUMINAMATH_CALUDE_track_circumference_l810_81097


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l810_81098

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → a = Real.sqrt 3 → b = 1 →
  (0 < A ∧ A < π) → (0 < B ∧ B < π) → (0 < C ∧ C < π) →
  (A + B + C = π) →
  (Real.sin A / a = Real.sin B / b) →
  B = π/6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l810_81098


namespace NUMINAMATH_CALUDE_base_7_representation_and_properties_l810_81009

def base_10_to_base_7 (n : ℕ) : List ℕ :=
  sorry

def count_even_digits (digits : List ℕ) : ℕ :=
  sorry

def sum_even_digits (digits : List ℕ) : ℕ :=
  sorry

theorem base_7_representation_and_properties :
  let base_7_repr := base_10_to_base_7 1250
  base_7_repr = [3, 4, 3, 4] ∧
  count_even_digits base_7_repr = 2 ∧
  ¬(sum_even_digits base_7_repr % 3 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_base_7_representation_and_properties_l810_81009


namespace NUMINAMATH_CALUDE_regular_tile_area_theorem_l810_81047

/-- Represents the properties of a tiled wall -/
structure TiledWall where
  total_area : ℝ
  regular_tile_length : ℝ
  regular_tile_width : ℝ
  jumbo_tile_length : ℝ
  jumbo_tile_width : ℝ
  jumbo_tile_ratio : ℝ
  regular_tile_count_ratio : ℝ

/-- The area covered by regular tiles in a tiled wall -/
def regular_tile_area (wall : TiledWall) : ℝ :=
  wall.total_area * wall.regular_tile_count_ratio

/-- Theorem stating the area covered by regular tiles in a specific wall configuration -/
theorem regular_tile_area_theorem (wall : TiledWall) 
  (h1 : wall.total_area = 220)
  (h2 : wall.jumbo_tile_ratio = 1/3)
  (h3 : wall.regular_tile_count_ratio = 2/3)
  (h4 : wall.jumbo_tile_length = 3 * wall.regular_tile_length)
  (h5 : wall.jumbo_tile_width = wall.regular_tile_width)
  : regular_tile_area wall = 146.67 := by
  sorry

#check regular_tile_area_theorem

end NUMINAMATH_CALUDE_regular_tile_area_theorem_l810_81047


namespace NUMINAMATH_CALUDE_power_equation_solution_l810_81028

theorem power_equation_solution : ∃ x : ℤ, 5^3 - 7 = 6^2 + x ∧ x = 82 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l810_81028


namespace NUMINAMATH_CALUDE_divisibility_property_l810_81075

theorem divisibility_property (n : ℕ) (a b : ℤ) :
  (a ≠ b) →
  (∀ m : ℕ, (n^m : ℤ) ∣ (a^m - b^m)) →
  (n : ℤ) ∣ a ∧ (n : ℤ) ∣ b :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_l810_81075


namespace NUMINAMATH_CALUDE_biggest_number_is_five_l810_81067

def yoongi_number : ℕ := 4
def jungkook_number : ℕ := 6 - 3
def yuna_number : ℕ := 5

theorem biggest_number_is_five :
  max yoongi_number (max jungkook_number yuna_number) = yuna_number :=
by sorry

end NUMINAMATH_CALUDE_biggest_number_is_five_l810_81067


namespace NUMINAMATH_CALUDE_roller_coaster_theorem_l810_81063

/-- The number of different combinations for two rides with 7 people,
    where each ride accommodates 4 people and no person rides more than once. -/
def roller_coaster_combinations : ℕ := 525

/-- The total number of people in the group. -/
def total_people : ℕ := 7

/-- The number of people that can fit in a car for each ride. -/
def people_per_ride : ℕ := 4

/-- The number of rides. -/
def number_of_rides : ℕ := 2

theorem roller_coaster_theorem :
  roller_coaster_combinations =
    (Nat.choose total_people people_per_ride) *
    (Nat.choose (total_people - 1) people_per_ride) :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_theorem_l810_81063


namespace NUMINAMATH_CALUDE_simplify_quadratic_expression_l810_81091

/-- Simplification of a quadratic expression -/
theorem simplify_quadratic_expression (y : ℝ) :
  4 * y + 8 * y^2 + 6 - (3 - 4 * y - 8 * y^2) = 16 * y^2 + 8 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_quadratic_expression_l810_81091


namespace NUMINAMATH_CALUDE_some_number_exists_l810_81060

theorem some_number_exists : ∃ N : ℝ, 
  (2 * ((3.6 * 0.48 * N) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002) ∧ 
  (abs (N - 2.5) < 0.0000000000000005) := by
  sorry

end NUMINAMATH_CALUDE_some_number_exists_l810_81060


namespace NUMINAMATH_CALUDE_complex_subtraction_multiplication_l810_81017

theorem complex_subtraction_multiplication (i : ℂ) :
  (7 - 3 * i) - 3 * (2 + 4 * i) = 1 - 15 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_multiplication_l810_81017


namespace NUMINAMATH_CALUDE_concert_ticket_sales_l810_81064

/-- Proves that given the conditions of the concert ticket sales, the number of back seat tickets sold is 14,500 --/
theorem concert_ticket_sales 
  (total_seats : ℕ) 
  (main_seat_price back_seat_price : ℚ) 
  (total_revenue : ℚ) 
  (h1 : total_seats = 20000)
  (h2 : main_seat_price = 55)
  (h3 : back_seat_price = 45)
  (h4 : total_revenue = 955000) :
  ∃ (main_seats back_seats : ℕ),
    main_seats + back_seats = total_seats ∧
    main_seat_price * main_seats + back_seat_price * back_seats = total_revenue ∧
    back_seats = 14500 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_sales_l810_81064


namespace NUMINAMATH_CALUDE_triangle_area_is_two_l810_81035

/-- The area of the triangle bounded by the y-axis and two lines -/
def triangle_area : ℝ := 2

/-- The first line equation: y - 2x = 1 -/
def line1 (x y : ℝ) : Prop := y - 2 * x = 1

/-- The second line equation: 4y + x = 16 -/
def line2 (x y : ℝ) : Prop := 4 * y + x = 16

/-- The theorem stating that the area of the triangle is 2 -/
theorem triangle_area_is_two :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ = 0 ∧ line1 x₁ y₁ ∧
    x₂ = 0 ∧ line2 x₂ y₂ ∧
    triangle_area = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_two_l810_81035


namespace NUMINAMATH_CALUDE_promotional_price_equiv_correct_method_l810_81032

/-- Represents the promotional price calculation for books -/
def promotional_price (x : ℝ) : ℝ := 0.8 * (x - 15)

/-- Represents the correct method of calculation as described in option C -/
def correct_method (x : ℝ) : ℝ := 0.8 * (x - 15)

/-- Theorem stating that the promotional price calculation is equivalent to the correct method -/
theorem promotional_price_equiv_correct_method :
  ∀ x : ℝ, promotional_price x = correct_method x := by
  sorry

end NUMINAMATH_CALUDE_promotional_price_equiv_correct_method_l810_81032


namespace NUMINAMATH_CALUDE_proportion_inconsistency_l810_81065

theorem proportion_inconsistency : ¬ ∃ (x : ℚ), (x / 2 = 2 / 6) ∧ (x = 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_proportion_inconsistency_l810_81065


namespace NUMINAMATH_CALUDE_newOp_seven_three_l810_81084

-- Define the new operation ⊗
def newOp (p q : ℝ) : ℝ := p^2 - 2*q

-- Theorem to prove
theorem newOp_seven_three : newOp 7 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_newOp_seven_three_l810_81084


namespace NUMINAMATH_CALUDE_marble_difference_is_negative_21_l810_81061

/-- The number of marbles Jonny has minus the number of marbles Marissa has -/
def marbleDifference : ℤ :=
  let mara_marbles := 12 * 2
  let markus_marbles := 2 * 13
  let jonny_marbles := 18
  let marissa_marbles := 3 * 5 + 3 * 8
  jonny_marbles - marissa_marbles

/-- Theorem stating the difference in marbles between Jonny and Marissa -/
theorem marble_difference_is_negative_21 : marbleDifference = -21 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_is_negative_21_l810_81061


namespace NUMINAMATH_CALUDE_circle_x_axis_intersection_l810_81001

/-- A circle with diameter endpoints (3,2) and (11,8) intersects the x-axis at x = 7 -/
theorem circle_x_axis_intersection :
  let p1 : ℝ × ℝ := (3, 2)
  let p2 : ℝ × ℝ := (11, 8)
  let center : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let radius : ℝ := ((p1.1 - center.1)^2 + (p1.2 - center.2)^2).sqrt
  ∃ x : ℝ, x ≠ p1.1 ∧ (x - center.1)^2 + center.2^2 = radius^2 ∧ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_x_axis_intersection_l810_81001


namespace NUMINAMATH_CALUDE_pigeonhole_birthday_birthday_problem_l810_81042

theorem pigeonhole_birthday (n : ℕ) (m : ℕ) (h : n > m) :
  ∀ f : Fin n → Fin m, ∃ i j : Fin n, i ≠ j ∧ f i = f j := by
  sorry

theorem birthday_problem :
  ∀ f : Fin 367 → Fin 366, ∃ i j : Fin 367, i ≠ j ∧ f i = f j := by
  exact pigeonhole_birthday 367 366 (by norm_num)

end NUMINAMATH_CALUDE_pigeonhole_birthday_birthday_problem_l810_81042


namespace NUMINAMATH_CALUDE_three_numbers_average_l810_81088

theorem three_numbers_average (x y z : ℝ) : 
  x = 18 ∧ y = 4 * x ∧ z = 2 * y → (x + y + z) / 3 = 78 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_average_l810_81088


namespace NUMINAMATH_CALUDE_jade_transactions_l810_81077

/-- The number of transactions handled by each person -/
structure Transactions where
  mabel : ℕ
  anthony : ℕ
  cal : ℕ
  dan : ℕ
  jade : ℕ

/-- The conditions of the problem -/
def problem_conditions (t : Transactions) : Prop :=
  t.mabel = 120 ∧
  t.anthony = t.mabel + t.mabel * 15 / 100 ∧
  t.cal = t.anthony * 3 / 4 ∧
  t.dan = t.mabel + t.mabel * 50 / 100 ∧
  t.jade = t.cal + 20

/-- The theorem stating that Jade handled 123 transactions -/
theorem jade_transactions (t : Transactions) (h : problem_conditions t) : t.jade = 123 := by
  sorry


end NUMINAMATH_CALUDE_jade_transactions_l810_81077


namespace NUMINAMATH_CALUDE_speed_of_M_constant_l810_81008

/-- Represents a crank-slider mechanism -/
structure CrankSlider where
  ω : ℝ  -- Angular velocity of the crank
  OA : ℝ  -- Length of OA
  AB : ℝ  -- Length of AB
  AM : ℝ  -- Length of AM

/-- The speed of point M in a crank-slider mechanism -/
def speed_of_M (cs : CrankSlider) : ℝ := cs.OA * cs.ω

/-- Theorem: The speed of point M is constant and equal to OA * ω -/
theorem speed_of_M_constant (cs : CrankSlider) 
  (h1 : cs.ω = 10)
  (h2 : cs.OA = 90)
  (h3 : cs.AB = 90)
  (h4 : cs.AM = cs.AB / 2) :
  speed_of_M cs = 900 := by
  sorry

end NUMINAMATH_CALUDE_speed_of_M_constant_l810_81008


namespace NUMINAMATH_CALUDE_test_questions_count_l810_81045

theorem test_questions_count (S I C : ℕ) : 
  S = C - 2 * I →
  S = 73 →
  C = 91 →
  C + I = 100 := by
sorry

end NUMINAMATH_CALUDE_test_questions_count_l810_81045


namespace NUMINAMATH_CALUDE_arithmetic_operations_l810_81094

theorem arithmetic_operations :
  (12 - (-5) + (-4) - 8 = 5) ∧
  (-1 - (1 + 1/2) * (1/3) / (-4)^2 = -33/32) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l810_81094


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l810_81044

/-- 
Given a point P with polar coordinates (r, θ), 
this theorem states that its Cartesian coordinates are (r cos(θ), r sin(θ)).
-/
theorem polar_to_cartesian (r θ : ℝ) : 
  let p : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)
  ∃ (x y : ℝ), p = (x, y) ∧ 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l810_81044


namespace NUMINAMATH_CALUDE_algebraic_manipulation_l810_81092

theorem algebraic_manipulation (a b : ℝ) :
  (-2 * a^2 * b)^2 * (3 * a * b^2 - 5 * a^2 * b) / (-a * b)^3 = -12 * a^2 * b + 20 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_manipulation_l810_81092


namespace NUMINAMATH_CALUDE_function_above_identity_l810_81000

theorem function_above_identity (f : ℝ → ℝ) (hf : Continuous f) :
  (∀ a₁ ∈ Set.Ioo 0 1, ∀ n : ℕ, f^[n+1] a₁ > f^[n] a₁) →
  ∀ x ∈ Set.Ioo 0 1, f x > x :=
sorry

end NUMINAMATH_CALUDE_function_above_identity_l810_81000


namespace NUMINAMATH_CALUDE_sequence_general_term_l810_81021

/-- Given a sequence {a_n} with n ∈ ℕ, if S_n = 2a_n - 2^n + 1 represents
    the sum of the first n terms, then a_n = n × 2^(n-1) for all n ∈ ℕ. -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = 2 * a n - 2^n + 1) →
  ∀ n : ℕ, a n = n * 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l810_81021


namespace NUMINAMATH_CALUDE_total_flower_cost_l810_81053

-- Define the promenade perimeter in meters
def promenade_perimeter : ℕ := 1500

-- Define the planting interval in meters
def planting_interval : ℕ := 30

-- Define the cost per flower in won
def cost_per_flower : ℕ := 5000

-- Theorem to prove
theorem total_flower_cost : 
  (promenade_perimeter / planting_interval) * cost_per_flower = 250000 := by
sorry

end NUMINAMATH_CALUDE_total_flower_cost_l810_81053


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l810_81051

theorem unique_solution_to_equation :
  ∃! (x y z : ℝ), 2*x^4 + 2*y^4 - 4*x^3*y + 6*x^2*y^2 - 4*x*y^3 + 7*y^2 + 7*z^2 - 14*y*z - 70*y + 70*z + 175 = 0 ∧
                   x = 0 ∧ y = 0 ∧ z = -5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l810_81051


namespace NUMINAMATH_CALUDE_intersection_inequality_solution_set_l810_81076

/-- Given a line and a hyperbola intersecting at two points, 
    prove the solution set of a related inequality. -/
theorem intersection_inequality_solution_set 
  (k₀ k b m n : ℝ) : 
  (∃ (x : ℝ), k₀ * x + b = k^2 / x ∧ 
              (x = m ∧ k₀ * m + b = -1 ∧ k^2 / m = -1) ∨
              (x = n ∧ k₀ * n + b = 2 ∧ k^2 / n = 2)) →
  {x : ℝ | x^2 > k₀ * k^2 + b * x} = {x : ℝ | x < -1 ∨ x > 2} :=
by sorry

end NUMINAMATH_CALUDE_intersection_inequality_solution_set_l810_81076


namespace NUMINAMATH_CALUDE_magic_square_solution_l810_81038

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  i : ℕ
  sum : ℕ
  row_sums : a + b + c = sum ∧ d + e + f = sum ∧ g + h + i = sum
  col_sums : a + d + g = sum ∧ b + e + h = sum ∧ c + f + i = sum
  diag_sums : a + e + i = sum ∧ c + e + g = sum

/-- The theorem to be proved -/
theorem magic_square_solution (ms : MagicSquare) 
  (h1 : ms.b = 25)
  (h2 : ms.c = 103)
  (h3 : ms.d = 3) :
  ms.a = 214 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_solution_l810_81038


namespace NUMINAMATH_CALUDE_paint_cost_per_quart_paint_cost_example_l810_81048

/-- The cost of paint per quart for a cube with given dimensions and coverage -/
theorem paint_cost_per_quart (cube_side : ℝ) (coverage_per_quart : ℝ) (total_cost : ℝ) : ℝ :=
  let surface_area := 6 * cube_side^2
  let quarts_needed := surface_area / coverage_per_quart
  total_cost / quarts_needed

/-- The cost of paint per quart is $3.20 for the given conditions -/
theorem paint_cost_example : paint_cost_per_quart 10 120 16 = 3.20 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_quart_paint_cost_example_l810_81048


namespace NUMINAMATH_CALUDE_units_digit_of_product_l810_81078

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_product : units_digit (2 * (factorial 1 + factorial 2 + factorial 3 + factorial 4)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l810_81078


namespace NUMINAMATH_CALUDE_inequality_proof_l810_81083

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  b * c / a + c * a / b + a * b / c ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l810_81083


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l810_81071

/-- Represents a 4x4x4 cube composed of unit cubes -/
structure Cube :=
  (size : Nat)
  (total_units : Nat)
  (painted_per_face : Nat)

/-- The number of unpainted unit cubes in the cube -/
def unpainted_cubes (c : Cube) : Nat :=
  c.total_units - (c.painted_per_face * 6)

/-- Theorem stating the number of unpainted cubes in the specific cube configuration -/
theorem unpainted_cubes_count (c : Cube) 
  (h1 : c.size = 4)
  (h2 : c.total_units = 64)
  (h3 : c.painted_per_face = 4) :
  unpainted_cubes c = 40 := by
  sorry


end NUMINAMATH_CALUDE_unpainted_cubes_count_l810_81071


namespace NUMINAMATH_CALUDE_students_liking_both_subjects_l810_81087

theorem students_liking_both_subjects 
  (total_students : ℕ) 
  (art_students : ℕ) 
  (science_students : ℕ) 
  (h1 : total_students = 45)
  (h2 : art_students = 42)
  (h3 : science_students = 40)
  (h4 : art_students ≤ total_students)
  (h5 : science_students ≤ total_students) :
  art_students + science_students - total_students = 37 := by
sorry

end NUMINAMATH_CALUDE_students_liking_both_subjects_l810_81087


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l810_81068

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define symmetry with respect to the origin
def symmetricToOrigin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

-- Theorem statement
theorem symmetric_point_coordinates :
  let A : Point3D := { x := 2, y := 1, z := 0 }
  let B : Point3D := symmetricToOrigin A
  B.x = -2 ∧ B.y = -1 ∧ B.z = 0 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l810_81068


namespace NUMINAMATH_CALUDE_jimmy_pizza_cost_per_slice_l810_81056

/-- Calculates the cost per slice of a pizza given the following parameters:
    * base_cost: The cost of a large pizza
    * num_slices: The number of slices in a large pizza
    * first_topping_cost: The cost of the first topping
    * next_two_toppings_cost: The cost of each of the next two toppings
    * remaining_toppings_cost: The cost of each remaining topping
    * num_toppings: The total number of toppings ordered
-/
def cost_per_slice (base_cost : ℚ) (num_slices : ℕ) (first_topping_cost : ℚ) 
                   (next_two_toppings_cost : ℚ) (remaining_toppings_cost : ℚ) 
                   (num_toppings : ℕ) : ℚ :=
  let total_cost := base_cost + first_topping_cost +
                    2 * next_two_toppings_cost +
                    (num_toppings - 3) * remaining_toppings_cost
  total_cost / num_slices

theorem jimmy_pizza_cost_per_slice :
  cost_per_slice 10 8 2 1 (1/2) 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_pizza_cost_per_slice_l810_81056


namespace NUMINAMATH_CALUDE_monotonicity_interval_min_value_on_interval_max_value_on_interval_l810_81022

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem for the interval of monotonicity
theorem monotonicity_interval :
  ∃ (a b : ℝ), a < b ∧ (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∨ (∀ x y, a < x ∧ x < y ∧ y < b → f x > f y) :=
sorry

-- Theorem for the minimum value on the interval [-3, 2]
theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-3) 2 ∧ f x = -18 ∧ ∀ y ∈ Set.Icc (-3) 2, f y ≥ f x :=
sorry

-- Theorem for the maximum value on the interval [-3, 2]
theorem max_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-3) 2 ∧ f x = 2 ∧ ∀ y ∈ Set.Icc (-3) 2, f y ≤ f x :=
sorry

end NUMINAMATH_CALUDE_monotonicity_interval_min_value_on_interval_max_value_on_interval_l810_81022


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l810_81081

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ≤ 1 / (a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l810_81081
