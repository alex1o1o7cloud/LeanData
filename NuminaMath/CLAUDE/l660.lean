import Mathlib

namespace NUMINAMATH_CALUDE_greg_situps_l660_66052

/-- 
Given:
- For every sit-up Peter does, Greg does 4.
- Peter did 24 sit-ups.

Prove that Greg did 96 sit-ups.
-/
theorem greg_situps (peter_situps : ℕ) (greg_ratio : ℕ) : 
  peter_situps = 24 → greg_ratio = 4 → peter_situps * greg_ratio = 96 := by
  sorry

end NUMINAMATH_CALUDE_greg_situps_l660_66052


namespace NUMINAMATH_CALUDE_negative_max_inverse_is_max_of_negative_inverses_l660_66037

/-- Given a non-empty set A of real numbers not containing zero,
    with a negative maximum value a, -a⁻¹ is the maximum value
    of the set {-x⁻¹ | x ∈ A}. -/
theorem negative_max_inverse_is_max_of_negative_inverses
  (A : Set ℝ)
  (hA_nonempty : A.Nonempty)
  (hA_no_zero : 0 ∉ A)
  (a : ℝ)
  (ha_max : ∀ x ∈ A, x ≤ a)
  (ha_neg : a < 0) :
  ∀ y ∈ {-x⁻¹ | x ∈ A}, y ≤ -a⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_negative_max_inverse_is_max_of_negative_inverses_l660_66037


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l660_66001

theorem least_positive_integer_multiple_of_53 :
  ∃ x : ℕ+, (∀ y : ℕ+, y < x → ¬(53 ∣ (3*y)^2 + 2*41*3*y + 41^2)) ∧
             (53 ∣ (3*x)^2 + 2*41*3*x + 41^2) ∧
             x = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l660_66001


namespace NUMINAMATH_CALUDE_common_roots_product_l660_66024

theorem common_roots_product (A B : ℝ) : 
  (∃ p q r s : ℂ, 
    (p^3 + A*p + 10 = 0) ∧ 
    (q^3 + A*q + 10 = 0) ∧ 
    (r^3 + A*r + 10 = 0) ∧
    (p^3 + B*p^2 + 50 = 0) ∧ 
    (q^3 + B*q^2 + 50 = 0) ∧ 
    (s^3 + B*s^2 + 50 = 0) ∧
    (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧ (p ≠ s) ∧ (q ≠ s)) →
  (∃ p q : ℂ, 
    (p^3 + A*p + 10 = 0) ∧ 
    (q^3 + A*q + 10 = 0) ∧
    (p^3 + B*p^2 + 50 = 0) ∧ 
    (q^3 + B*q^2 + 50 = 0) ∧
    (p*q = 5 * (4^(1/3)))) := by
sorry

end NUMINAMATH_CALUDE_common_roots_product_l660_66024


namespace NUMINAMATH_CALUDE_polynomial_sum_squares_l660_66087

theorem polynomial_sum_squares (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (x - 2)^8 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
                        a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8) →
  (a₂ + a₄ + a₆ + a₈)^2 - (a₁ + a₃ + a₅ + a₇)^2 = -255 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_squares_l660_66087


namespace NUMINAMATH_CALUDE_train_travel_time_l660_66070

/-- Proves that a train traveling at 120 kmph for 80 km takes 40 minutes -/
theorem train_travel_time (speed : ℝ) (distance : ℝ) (time : ℝ) :
  speed = 120 →
  distance = 80 →
  time = distance / speed * 60 →
  time = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_time_l660_66070


namespace NUMINAMATH_CALUDE_lineup_combinations_l660_66078

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of players in a triplet
def triplet_size : ℕ := 3

-- Define the number of triplet sets
def triplet_sets : ℕ := 2

-- Define the number of players to choose for the lineup
def lineup_size : ℕ := 7

-- Define the maximum number of players that can be chosen from a triplet set
def max_from_triplet : ℕ := 2

-- Define the function to calculate the number of ways to choose the lineup
def choose_lineup : ℕ := sorry

-- Theorem stating that the number of ways to choose the lineup is 21582
theorem lineup_combinations : choose_lineup = 21582 := by sorry

end NUMINAMATH_CALUDE_lineup_combinations_l660_66078


namespace NUMINAMATH_CALUDE_window_area_ratio_l660_66028

theorem window_area_ratio :
  let AB : ℝ := 36
  let AD : ℝ := AB * (5/3)
  let circle_area : ℝ := Real.pi * (AB/2)^2
  let rectangle_area : ℝ := AD * AB
  let square_area : ℝ := AB^2
  rectangle_area / (circle_area + square_area) = 2160 / (324 * Real.pi + 1296) :=
by sorry

end NUMINAMATH_CALUDE_window_area_ratio_l660_66028


namespace NUMINAMATH_CALUDE_composite_n4_plus_4_l660_66007

theorem composite_n4_plus_4 (n : ℕ) (h : n ≥ 2) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 4 = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_n4_plus_4_l660_66007


namespace NUMINAMATH_CALUDE_uncle_dave_nieces_l660_66062

theorem uncle_dave_nieces (ice_cream_per_niece : ℝ) (total_ice_cream : ℕ) 
  (h1 : ice_cream_per_niece = 143.0)
  (h2 : total_ice_cream = 1573) :
  (total_ice_cream : ℝ) / ice_cream_per_niece = 11 := by
  sorry

end NUMINAMATH_CALUDE_uncle_dave_nieces_l660_66062


namespace NUMINAMATH_CALUDE_rope_length_satisfies_conditions_l660_66083

/-- The length of the rope in feet -/
def rope_length : ℝ := 10

/-- The length of the rope hanging down from the top of the pillar to the ground in feet -/
def hanging_length : ℝ := 4

/-- The distance from the base of the pillar to where the rope reaches the ground when pulled taut in feet -/
def ground_distance : ℝ := 8

/-- Theorem stating that the rope length satisfies the given conditions -/
theorem rope_length_satisfies_conditions :
  rope_length ^ 2 = (rope_length - hanging_length) ^ 2 + ground_distance ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_rope_length_satisfies_conditions_l660_66083


namespace NUMINAMATH_CALUDE_product_without_x_cube_term_l660_66064

theorem product_without_x_cube_term (m : ℚ) : 
  (∀ a b c d : ℚ, (m * X^4 + a * X^3 + b * X^2 + c * X + d) = 
    (m * X^2 - 3 * X) * (X^2 - 2 * X - 1) → a = 0) → 
  m = -3/2 := by sorry

end NUMINAMATH_CALUDE_product_without_x_cube_term_l660_66064


namespace NUMINAMATH_CALUDE_intersection_distance_l660_66045

def C₁ (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 7

def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

theorem intersection_distance :
  ∃ (ρ₁ ρ₂ : ℝ),
    C₁ (ρ₁ * Real.cos (π/6)) (ρ₁ * Real.sin (π/6)) ∧
    C₂ ρ₂ (π/6) ∧
    ρ₁ > 0 ∧ ρ₂ > 0 ∧
    ρ₁ - ρ₂ = 3 - Real.sqrt 3 :=
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l660_66045


namespace NUMINAMATH_CALUDE_trivia_team_selection_l660_66025

/-- The number of students not picked for a trivia team --/
def students_not_picked (total : ℕ) (groups : ℕ) (per_group : ℕ) : ℕ :=
  total - (groups * per_group)

/-- Theorem: Given 65 total students, 8 groups, and 6 students per group,
    17 students were not picked for the trivia team --/
theorem trivia_team_selection :
  students_not_picked 65 8 6 = 17 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_selection_l660_66025


namespace NUMINAMATH_CALUDE_positive_correlation_implies_positive_slope_negative_correlation_implies_negative_slope_positive_slope_implies_positive_correlation_negative_slope_implies_negative_correlation_l660_66079

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Determines if two variables are positively correlated based on the slope of their linear regression equation -/
def positively_correlated (eq : LinearRegression) : Prop :=
  eq.slope > 0

/-- Determines if two variables are negatively correlated based on the slope of their linear regression equation -/
def negatively_correlated (eq : LinearRegression) : Prop :=
  eq.slope < 0

/-- States that a linear regression equation with positive slope implies positive correlation -/
theorem positive_correlation_implies_positive_slope (eq : LinearRegression) :
  positively_correlated eq → eq.slope > 0 := by sorry

/-- States that a linear regression equation with negative slope implies negative correlation -/
theorem negative_correlation_implies_negative_slope (eq : LinearRegression) :
  negatively_correlated eq → eq.slope < 0 := by sorry

/-- States that positive slope implies positive correlation -/
theorem positive_slope_implies_positive_correlation (eq : LinearRegression) :
  eq.slope > 0 → positively_correlated eq := by sorry

/-- States that negative slope implies negative correlation -/
theorem negative_slope_implies_negative_correlation (eq : LinearRegression) :
  eq.slope < 0 → negatively_correlated eq := by sorry

end NUMINAMATH_CALUDE_positive_correlation_implies_positive_slope_negative_correlation_implies_negative_slope_positive_slope_implies_positive_correlation_negative_slope_implies_negative_correlation_l660_66079


namespace NUMINAMATH_CALUDE_roberts_reading_l660_66032

/-- Given Robert's reading speed, book size, and available time, 
    prove the maximum number of complete books he can read. -/
theorem roberts_reading (
  reading_speed : ℕ) 
  (book_size : ℕ) 
  (available_time : ℕ) 
  (h1 : reading_speed = 120) 
  (h2 : book_size = 360) 
  (h3 : available_time = 8) : 
  (available_time * reading_speed) / book_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_roberts_reading_l660_66032


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l660_66023

/-- Given a line segment CD where C(6,-1) is one endpoint and M(4,3) is the midpoint,
    the product of the coordinates of point D is 14. -/
theorem midpoint_coordinate_product : 
  let C : ℝ × ℝ := (6, -1)
  let M : ℝ × ℝ := (4, 3)
  let D : ℝ × ℝ := (2 * M.1 - C.1, 2 * M.2 - C.2)  -- Midpoint formula solved for D
  (D.1 * D.2 = 14) := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l660_66023


namespace NUMINAMATH_CALUDE_congruence_problem_l660_66093

theorem congruence_problem (x : ℤ) : 
  (5 * x + 9) % 19 = 3 → (3 * x + 14) % 19 = 18 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l660_66093


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l660_66069

theorem smallest_angle_solution (x : Real) : 
  (8 * Real.sin x * (Real.cos x)^6 - 8 * (Real.sin x)^6 * Real.cos x = 2) →
  (x ≥ 0) →
  (∀ y : Real, y > 0 ∧ y < x → 
    8 * Real.sin y * (Real.cos y)^6 - 8 * (Real.sin y)^6 * Real.cos y ≠ 2) →
  x = 11.25 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l660_66069


namespace NUMINAMATH_CALUDE_matrix_tripler_uniqueness_l660_66040

theorem matrix_tripler_uniqueness (A M : Matrix (Fin 2) (Fin 2) ℝ) :
  (∀ (i j : Fin 2), (M • A) i j = 3 * A i j) ↔ M = ![![3, 0], ![0, 3]] := by
sorry

end NUMINAMATH_CALUDE_matrix_tripler_uniqueness_l660_66040


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l660_66022

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (h1 : a 4 + a 8 = -2) : 
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l660_66022


namespace NUMINAMATH_CALUDE_bryans_bookshelves_l660_66063

/-- Given that Bryan has 42 books in total and each bookshelf contains 2 books,
    prove that the number of bookshelves he has is 21. -/
theorem bryans_bookshelves :
  let total_books : ℕ := 42
  let books_per_shelf : ℕ := 2
  let num_shelves : ℕ := total_books / books_per_shelf
  num_shelves = 21 := by
  sorry

end NUMINAMATH_CALUDE_bryans_bookshelves_l660_66063


namespace NUMINAMATH_CALUDE_expected_democrat_votes_l660_66004

/-- Represents the percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 0.60

/-- Represents the percentage of Republican voters expected to vote for candidate A -/
def republican_vote_percentage : ℝ := 0.20

/-- Represents the total percentage of votes candidate A is expected to receive -/
def total_vote_percentage : ℝ := 0.53

/-- Represents the percentage of Democrat voters expected to vote for candidate A -/
def democrat_vote_percentage : ℝ := 0.75

theorem expected_democrat_votes :
  democrat_vote_percentage * democrat_percentage + 
  republican_vote_percentage * (1 - democrat_percentage) = 
  total_vote_percentage :=
sorry

end NUMINAMATH_CALUDE_expected_democrat_votes_l660_66004


namespace NUMINAMATH_CALUDE_point_transformation_l660_66012

/-- Rotation of 90° counterclockwise around a point -/
def rotate90 (x y cx cy : ℝ) : ℝ × ℝ :=
  (cx - (y - cy), cy + (x - cx))

/-- Reflection about y = x line -/
def reflectYeqX (x y : ℝ) : ℝ × ℝ := (y, x)

/-- The main theorem -/
theorem point_transformation (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate90 a b 2 3
  let final := reflectYeqX rotated.1 rotated.2
  final = (5, 1) → b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l660_66012


namespace NUMINAMATH_CALUDE_product_remainder_l660_66043

theorem product_remainder (a b c : ℕ) (ha : a = 1234) (hb : b = 1567) (hc : c = 1912) :
  (a * b * c) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l660_66043


namespace NUMINAMATH_CALUDE_expected_digits_is_1_55_l660_66068

/-- A fair 20-sided die with numbers from 1 to 20 -/
def icosahedral_die : Finset ℕ := Finset.range 20

/-- The probability of rolling any specific number on the die -/
def prob_roll (n : ℕ) : ℚ := if n ∈ icosahedral_die then 1 / 20 else 0

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := 
  if n < 10 then 1 else 2

/-- The expected number of digits when rolling the icosahedral die -/
def expected_digits : ℚ := 
  (icosahedral_die.sum (λ n => prob_roll n * num_digits n))

/-- Theorem: The expected number of digits when rolling a fair 20-sided die 
    with numbers from 1 to 20 is 1.55 -/
theorem expected_digits_is_1_55 : expected_digits = 31 / 20 := by
  sorry

end NUMINAMATH_CALUDE_expected_digits_is_1_55_l660_66068


namespace NUMINAMATH_CALUDE_empty_truck_weight_l660_66053

-- Define the constants
def bridge_limit : ℕ := 20000
def soda_crates : ℕ := 20
def soda_crate_weight : ℕ := 50
def dryers : ℕ := 3
def dryer_weight : ℕ := 3000
def loaded_truck_weight : ℕ := 24000

-- Define the theorem
theorem empty_truck_weight :
  let soda_weight := soda_crates * soda_crate_weight
  let produce_weight := 2 * soda_weight
  let dryers_weight := dryers * dryer_weight
  let cargo_weight := soda_weight + produce_weight + dryers_weight
  loaded_truck_weight - cargo_weight = 12000 := by
  sorry


end NUMINAMATH_CALUDE_empty_truck_weight_l660_66053


namespace NUMINAMATH_CALUDE_hank_total_donation_l660_66021

def carwash_earnings : ℝ := 100
def carwash_donation_percentage : ℝ := 0.90
def bake_sale_earnings : ℝ := 80
def bake_sale_donation_percentage : ℝ := 0.75
def lawn_mowing_earnings : ℝ := 50
def lawn_mowing_donation_percentage : ℝ := 1.00

def total_donation : ℝ := 
  carwash_earnings * carwash_donation_percentage +
  bake_sale_earnings * bake_sale_donation_percentage +
  lawn_mowing_earnings * lawn_mowing_donation_percentage

theorem hank_total_donation : total_donation = 200 := by
  sorry

end NUMINAMATH_CALUDE_hank_total_donation_l660_66021


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l660_66013

/-- Given two parabolas that intersect the coordinate axes in four points forming a kite -/
structure ParabolaKite where
  a' : ℝ
  b' : ℝ
  intersection_points : Fin 4 → ℝ × ℝ
  is_kite : Bool
  kite_area : ℝ

/-- The theorem stating the sum of a' and b' -/
theorem parabola_kite_sum (pk : ParabolaKite)
  (h1 : pk.is_kite = true)
  (h2 : pk.kite_area = 18)
  (h3 : ∀ (i : Fin 4), (pk.intersection_points i).1 = 0 ∨ (pk.intersection_points i).2 = 0)
  (h4 : ∀ (x y : ℝ), y = pk.a' * x^2 + 3 ∨ y = 6 - pk.b' * x^2) :
  pk.a' + pk.b' = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l660_66013


namespace NUMINAMATH_CALUDE_no_two_roots_in_interval_l660_66016

theorem no_two_roots_in_interval (a b c : ℝ) (ha : a > 0) (hcond : 12 * a + 5 * b + 2 * c > 0) :
  ¬∃ (x y : ℝ), 2 < x ∧ x < 3 ∧ 2 < y ∧ y < 3 ∧
  x ≠ y ∧
  a * x^2 + b * x + c = 0 ∧
  a * y^2 + b * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_two_roots_in_interval_l660_66016


namespace NUMINAMATH_CALUDE_no_simultaneous_cubes_l660_66099

theorem no_simultaneous_cubes (n : ℕ) : 
  ¬(∃ (a b : ℤ), (2^(n+1) - 1 = a^3) ∧ (2^(n-1) * (2^n - 1) = b^3)) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_cubes_l660_66099


namespace NUMINAMATH_CALUDE_complement_A_in_U_l660_66010

open Set

-- Define the universal set U
def U : Set ℝ := {x | x^2 > 1}

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Theorem statement
theorem complement_A_in_U :
  (U \ A) = {x : ℝ | x ≥ 3 ∨ x < -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l660_66010


namespace NUMINAMATH_CALUDE_flower_planting_area_l660_66077

/-- Represents a square lawn with flowers -/
structure FlowerLawn where
  side_length : ℝ
  flower_area : ℝ

/-- Theorem: A square lawn with side length 16 meters can have a flower planting area of 144 square meters -/
theorem flower_planting_area (lawn : FlowerLawn) (h1 : lawn.side_length = 16) 
  (h2 : lawn.flower_area = 144) : 
  lawn.flower_area ≤ lawn.side_length ^ 2 ∧ lawn.flower_area > 0 := by
  sorry

#check flower_planting_area

end NUMINAMATH_CALUDE_flower_planting_area_l660_66077


namespace NUMINAMATH_CALUDE_pencils_per_child_l660_66044

theorem pencils_per_child (num_children : ℕ) (total_pencils : ℕ) (h1 : num_children = 2) (h2 : total_pencils = 12) :
  total_pencils / num_children = 6 := by
sorry

end NUMINAMATH_CALUDE_pencils_per_child_l660_66044


namespace NUMINAMATH_CALUDE_prize_distribution_l660_66026

theorem prize_distribution (total_prize : ℕ) (num_prizes : ℕ) (first_prize : ℕ) (second_prize : ℕ) (third_prize : ℕ)
  (h_total : total_prize = 4200)
  (h_num : num_prizes = 7)
  (h_first : first_prize = 800)
  (h_second : second_prize = 700)
  (h_third : third_prize = 300) :
  ∃ (x y z : ℕ),
    x + y + z = num_prizes ∧
    x * first_prize + y * second_prize + z * third_prize = total_prize ∧
    x = 1 ∧ y = 4 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_prize_distribution_l660_66026


namespace NUMINAMATH_CALUDE_evaluate_x_squared_minus_y_squared_l660_66081

theorem evaluate_x_squared_minus_y_squared : 
  ∀ x y : ℝ, x + y = 10 → 2 * x + y = 13 → x^2 - y^2 = -40 := by
sorry

end NUMINAMATH_CALUDE_evaluate_x_squared_minus_y_squared_l660_66081


namespace NUMINAMATH_CALUDE_birds_on_fence_l660_66094

/-- Given an initial number of birds on a fence and an additional number of birds that land on the fence,
    calculate the total number of birds on the fence. -/
def total_birds (initial : Nat) (additional : Nat) : Nat :=
  initial + additional

/-- Theorem stating that with 12 initial birds and 8 additional birds, the total is 20 -/
theorem birds_on_fence : total_birds 12 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l660_66094


namespace NUMINAMATH_CALUDE_gcd_of_four_numbers_l660_66071

theorem gcd_of_four_numbers : Nat.gcd 84 (Nat.gcd 108 (Nat.gcd 132 156)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_four_numbers_l660_66071


namespace NUMINAMATH_CALUDE_fraction_equality_l660_66074

theorem fraction_equality (a b : ℝ) (x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + b) / (a - b) = (x + 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l660_66074


namespace NUMINAMATH_CALUDE_union_of_sets_l660_66058

open Set

theorem union_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | -1 < x ∧ x < 3} → 
  B = {x : ℝ | x ≥ 1} → 
  A ∪ B = {x : ℝ | x > -1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l660_66058


namespace NUMINAMATH_CALUDE_three_number_sum_l660_66076

theorem three_number_sum (a b c : ℝ) (h1 : a < b) (h2 : b < c) 
  (h3 : ((a + b)/2 + (b + c)/2) / 2 = (a + b + c) / 3)
  (h4 : (a + c) / 2 = 2022) : 
  a + b + c = 6066 := by
  sorry

end NUMINAMATH_CALUDE_three_number_sum_l660_66076


namespace NUMINAMATH_CALUDE_non_monotonic_interval_implies_k_range_l660_66057

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the property of non-monotonicity in an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a < x ∧ x < y ∧ y < b ∧ (f x < f y ∧ ∃ z, x < z ∧ z < y ∧ f z < f x)

-- State the theorem
theorem non_monotonic_interval_implies_k_range (k : ℝ) :
  not_monotonic f (k - 1) (k + 1) → (-3 < k ∧ k < -1) ∨ (1 < k ∧ k < 3) :=
sorry

end NUMINAMATH_CALUDE_non_monotonic_interval_implies_k_range_l660_66057


namespace NUMINAMATH_CALUDE_max_volume_right_prism_l660_66030

/-- 
Given a right prism with triangular bases where:
- Base triangle sides are a, b, b
- a = 2b
- Angle between sides a and b is π/2
- Sum of areas of two lateral faces and one base is 30
The maximum volume of the prism is 2.5√5.
-/
theorem max_volume_right_prism (a b h : ℝ) :
  a = 2 * b →
  4 * b * h + b^2 = 30 →
  (∀ h' : ℝ, 4 * b * h' + b^2 = 30 → b^2 * h / 2 ≤ b^2 * h' / 2) →
  b^2 * h / 2 = 2.5 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_right_prism_l660_66030


namespace NUMINAMATH_CALUDE_true_discount_calculation_l660_66056

/-- Calculates the true discount given the banker's gain, interest rate, and time period. -/
def true_discount (bankers_gain : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  (bankers_gain * 100) / (interest_rate * time)

/-- Theorem stating that under the given conditions, the true discount is 55. -/
theorem true_discount_calculation :
  let bankers_gain : ℚ := 6.6
  let interest_rate : ℚ := 12
  let time : ℚ := 1
  true_discount bankers_gain interest_rate time = 55 := by
sorry

end NUMINAMATH_CALUDE_true_discount_calculation_l660_66056


namespace NUMINAMATH_CALUDE_student_goldfish_difference_l660_66090

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 20

/-- The number of goldfish in each fourth-grade classroom -/
def goldfish_per_classroom : ℕ := 3

/-- The theorem stating the difference between the total number of students and goldfish -/
theorem student_goldfish_difference :
  num_classrooms * students_per_classroom - num_classrooms * goldfish_per_classroom = 85 := by
  sorry

end NUMINAMATH_CALUDE_student_goldfish_difference_l660_66090


namespace NUMINAMATH_CALUDE_lost_ship_depth_l660_66033

/-- The depth of a lost ship given the descent rate and time taken to reach it. -/
def ship_depth (descent_rate : ℝ) (time_taken : ℝ) : ℝ := descent_rate * time_taken

/-- Theorem stating the depth of the lost ship -/
theorem lost_ship_depth :
  let descent_rate : ℝ := 80
  let time_taken : ℝ := 50
  ship_depth descent_rate time_taken = 4000 := by
  sorry

end NUMINAMATH_CALUDE_lost_ship_depth_l660_66033


namespace NUMINAMATH_CALUDE_round_trip_distance_approx_l660_66049

/-- Represents the total distance traveled in John's round trip --/
def total_distance (city_speed outbound_highway_speed return_highway_speed : ℝ)
  (outbound_city_time outbound_highway_time return_highway_time1 return_highway_time2 return_city_time : ℝ) : ℝ :=
  let outbound_city_distance := city_speed * outbound_city_time
  let outbound_highway_distance := outbound_highway_speed * outbound_highway_time
  let return_highway_distance := return_highway_speed * (return_highway_time1 + return_highway_time2)
  let return_city_distance := city_speed * return_city_time
  outbound_city_distance + outbound_highway_distance + return_highway_distance + return_city_distance

/-- Theorem stating that the total round trip distance is approximately 166.67 km --/
theorem round_trip_distance_approx : 
  ∀ (ε : ℝ), ε > 0 → 
  ∃ (city_speed outbound_highway_speed return_highway_speed : ℝ)
    (outbound_city_time outbound_highway_time return_highway_time1 return_highway_time2 return_city_time : ℝ),
  city_speed = 40 ∧ 
  outbound_highway_speed = 80 ∧
  return_highway_speed = 100 ∧
  outbound_city_time = 1/3 ∧
  outbound_highway_time = 2/3 ∧
  return_highway_time1 = 1/2 ∧
  return_highway_time2 = 1/6 ∧
  return_city_time = 1/3 ∧
  |total_distance city_speed outbound_highway_speed return_highway_speed
    outbound_city_time outbound_highway_time return_highway_time1 return_highway_time2 return_city_time - 166.67| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_round_trip_distance_approx_l660_66049


namespace NUMINAMATH_CALUDE_fraction_equality_l660_66029

theorem fraction_equality : (5 * 7) / 8 = 4 + 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l660_66029


namespace NUMINAMATH_CALUDE_hobby_store_sales_l660_66008

/-- The combined sales of trading cards in June and July -/
def combined_sales (normal_sales : ℕ) (june_extra : ℕ) : ℕ :=
  (normal_sales + june_extra) + normal_sales

/-- Theorem stating the combined sales of trading cards in June and July -/
theorem hobby_store_sales : combined_sales 21122 3922 = 46166 := by
  sorry

end NUMINAMATH_CALUDE_hobby_store_sales_l660_66008


namespace NUMINAMATH_CALUDE_circles_tangent_internally_l660_66034

theorem circles_tangent_internally (r₁ r₂ d : ℝ) :
  r₁ = 4 → r₂ = 7 → d = 3 → d = r₂ - r₁ := by
  sorry

end NUMINAMATH_CALUDE_circles_tangent_internally_l660_66034


namespace NUMINAMATH_CALUDE_three_circles_collinearity_l660_66072

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the line structure
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem three_circles_collinearity 
  (circleA circleB circleC : Circle)
  (A B C : Point)
  (B₁ C₁ C₂ A₂ A₃ B₃ : Point)
  (X Y Z : Point) :
  -- Conditions
  circleA.center = (A.x, A.y) →
  circleB.center = (B.x, B.y) →
  circleC.center = (C.x, C.y) →
  circleA.radius = circleB.radius →
  circleB.radius = circleC.radius →
  -- B₁ and C₁ are on circleA
  (B₁.x - A.x)^2 + (B₁.y - A.y)^2 = circleA.radius^2 →
  (C₁.x - A.x)^2 + (C₁.y - A.y)^2 = circleA.radius^2 →
  -- C₂ and A₂ are on circleB
  (C₂.x - B.x)^2 + (C₂.y - B.y)^2 = circleB.radius^2 →
  (A₂.x - B.x)^2 + (A₂.y - B.y)^2 = circleB.radius^2 →
  -- A₃ and B₃ are on circleC
  (A₃.x - C.x)^2 + (A₃.y - C.y)^2 = circleC.radius^2 →
  (B₃.x - C.x)^2 + (B₃.y - C.y)^2 = circleC.radius^2 →
  -- X is the intersection of B₁C₁ and BC
  (∃ (l₁ : Line), l₁.a * B₁.x + l₁.b * B₁.y + l₁.c = 0 ∧
                  l₁.a * C₁.x + l₁.b * C₁.y + l₁.c = 0 ∧
                  l₁.a * X.x + l₁.b * X.y + l₁.c = 0) →
  (∃ (l₂ : Line), l₂.a * B.x + l₂.b * B.y + l₂.c = 0 ∧
                  l₂.a * C.x + l₂.b * C.y + l₂.c = 0 ∧
                  l₂.a * X.x + l₂.b * X.y + l₂.c = 0) →
  -- Y is the intersection of C₂A₂ and CA
  (∃ (l₃ : Line), l₃.a * C₂.x + l₃.b * C₂.y + l₃.c = 0 ∧
                  l₃.a * A₂.x + l₃.b * A₂.y + l₃.c = 0 ∧
                  l₃.a * Y.x + l₃.b * Y.y + l₃.c = 0) →
  (∃ (l₄ : Line), l₄.a * C.x + l₄.b * C.y + l₄.c = 0 ∧
                  l₄.a * A.x + l₄.b * A.y + l₄.c = 0 ∧
                  l₄.a * Y.x + l₄.b * Y.y + l₄.c = 0) →
  -- Z is the intersection of A₃B₃ and AB
  (∃ (l₅ : Line), l₅.a * A₃.x + l₅.b * A₃.y + l₅.c = 0 ∧
                  l₅.a * B₃.x + l₅.b * B₃.y + l₅.c = 0 ∧
                  l₅.a * Z.x + l₅.b * Z.y + l₅.c = 0) →
  (∃ (l₆ : Line), l₆.a * A.x + l₆.b * A.y + l₆.c = 0 ∧
                  l₆.a * B.x + l₆.b * B.y + l₆.c = 0 ∧
                  l₆.a * Z.x + l₆.b * Z.y + l₆.c = 0) →
  -- Conclusion: X, Y, and Z are collinear
  ∃ (l : Line), l.a * X.x + l.b * X.y + l.c = 0 ∧
                l.a * Y.x + l.b * Y.y + l.c = 0 ∧
                l.a * Z.x + l.b * Z.y + l.c = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_three_circles_collinearity_l660_66072


namespace NUMINAMATH_CALUDE_iggy_wednesday_miles_l660_66000

/-- Represents the days of the week Iggy runs --/
inductive RunDay
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Represents Iggy's running schedule --/
def IggySchedule : RunDay → ℕ
  | RunDay.Monday => 3
  | RunDay.Tuesday => 4
  | RunDay.Thursday => 8
  | RunDay.Friday => 3
  | RunDay.Wednesday => 0  -- We'll prove this should be 6

/-- Iggy's pace in minutes per mile --/
def IggyPace : ℕ := 10

/-- Total running time in hours --/
def TotalRunningTime : ℕ := 4

/-- Converts hours to minutes --/
def HoursToMinutes (hours : ℕ) : ℕ := hours * 60

theorem iggy_wednesday_miles :
  ∃ (wednesday_miles : ℕ),
    wednesday_miles = 6 ∧
    HoursToMinutes TotalRunningTime =
      (IggySchedule RunDay.Monday +
       IggySchedule RunDay.Tuesday +
       wednesday_miles +
       IggySchedule RunDay.Thursday +
       IggySchedule RunDay.Friday) * IggyPace :=
by sorry

end NUMINAMATH_CALUDE_iggy_wednesday_miles_l660_66000


namespace NUMINAMATH_CALUDE_book_cost_l660_66059

/-- Given that three identical books cost $36, prove that seven of these books cost $84. -/
theorem book_cost (cost_of_three : ℝ) (h : cost_of_three = 36) : 
  (7 / 3) * cost_of_three = 84 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l660_66059


namespace NUMINAMATH_CALUDE_square_minimizes_diagonal_l660_66006

/-- A parallelogram with side lengths and angles -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  angle : ℝ
  area : ℝ

/-- The length of the larger diagonal of a parallelogram -/
def largerDiagonal (p : Parallelogram) : ℝ :=
  sorry

/-- Theorem: Among all parallelograms with a given area, the square has the smallest larger diagonal -/
theorem square_minimizes_diagonal {A : ℝ} (h : A > 0) :
  ∀ p : Parallelogram, p.area = A →
    largerDiagonal p ≥ largerDiagonal { side1 := Real.sqrt A, side2 := Real.sqrt A, angle := π/2, area := A } :=
  sorry

end NUMINAMATH_CALUDE_square_minimizes_diagonal_l660_66006


namespace NUMINAMATH_CALUDE_adjacent_triangle_number_l660_66080

/-- Given a triangular arrangement of natural numbers where the k-th row 
    contains numbers from (k-1)^2 + 1 to k^2, if 267 is in one triangle, 
    then 301 is in the adjacent triangle that shares a horizontal side. -/
theorem adjacent_triangle_number : ∀ (k : ℕ),
  (k - 1)^2 + 1 ≤ 267 ∧ 267 ≤ k^2 →
  ∃ (n : ℕ), n ≤ k^2 - ((k - 1)^2 + 1) + 1 ∧
  301 = (k + 1)^2 - (n + k - 1) :=
by sorry

end NUMINAMATH_CALUDE_adjacent_triangle_number_l660_66080


namespace NUMINAMATH_CALUDE_max_value_abc_l660_66015

theorem max_value_abc (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 5)
  (hb : 0 ≤ b ∧ b ≤ 5)
  (hc : 0 ≤ c ∧ c ≤ 5)
  (h_sum : 2 * a + b + c = 10) :
  a + 2 * b + 3 * c ≤ 25 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l660_66015


namespace NUMINAMATH_CALUDE_first_term_exceeding_2020_l660_66014

theorem first_term_exceeding_2020 : 
  (∃ n : ℕ, 2 * n^2 ≥ 2020 ∧ ∀ m : ℕ, m < n → 2 * m^2 < 2020) → 
  (∃ n : ℕ, 2 * n^2 ≥ 2020 ∧ ∀ m : ℕ, m < n → 2 * m^2 < 2020) ∧ 
  (∀ n : ℕ, (2 * n^2 ≥ 2020 ∧ ∀ m : ℕ, m < n → 2 * m^2 < 2020) → n = 32) :=
by sorry

end NUMINAMATH_CALUDE_first_term_exceeding_2020_l660_66014


namespace NUMINAMATH_CALUDE_first_digit_of_87_base_5_l660_66055

def base_5_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

def first_digit_base_5 (n : ℕ) : ℕ :=
  match base_5_representation n with
  | [] => 0  -- This case should never occur for a valid input
  | d::_ => d

theorem first_digit_of_87_base_5 :
  first_digit_base_5 87 = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_of_87_base_5_l660_66055


namespace NUMINAMATH_CALUDE_complex_number_properties_l660_66060

theorem complex_number_properties (z₁ z₂ : ℂ) (h : Complex.abs z₁ * Complex.abs z₂ ≠ 0) :
  (Complex.abs (z₁ + z₂) ≤ Complex.abs z₁ + Complex.abs z₂) ∧
  (Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l660_66060


namespace NUMINAMATH_CALUDE_stamp_redistribution_l660_66036

/-- Represents the stamp redistribution problem -/
theorem stamp_redistribution (
  initial_albums : ℕ)
  (initial_pages_per_album : ℕ)
  (initial_stamps_per_page : ℕ)
  (new_stamps_per_page : ℕ)
  (filled_albums : ℕ)
  (h1 : initial_albums = 10)
  (h2 : initial_pages_per_album = 50)
  (h3 : initial_stamps_per_page = 7)
  (h4 : new_stamps_per_page = 12)
  (h5 : filled_albums = 6) :
  (initial_albums * initial_pages_per_album * initial_stamps_per_page) % new_stamps_per_page = 8 := by
  sorry

#check stamp_redistribution

end NUMINAMATH_CALUDE_stamp_redistribution_l660_66036


namespace NUMINAMATH_CALUDE_students_playing_neither_l660_66091

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) 
  (h1 : total = 36)
  (h2 : football = 26)
  (h3 : tennis = 20)
  (h4 : both = 17) :
  total - (football + tennis - both) = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_l660_66091


namespace NUMINAMATH_CALUDE_prime_sum_square_cube_l660_66061

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def validSolution (p q r : ℕ) : Prop :=
  isPrime p ∧ isPrime q ∧ isPrime r ∧ p + q^2 + r^3 = 200

theorem prime_sum_square_cube :
  {(p, q, r) : ℕ × ℕ × ℕ | validSolution p q r} =
  {(167, 5, 2), (71, 11, 2), (23, 13, 2), (71, 2, 5)} :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_square_cube_l660_66061


namespace NUMINAMATH_CALUDE_month_days_l660_66046

theorem month_days (days_taken : ℕ) (days_forgotten : ℕ) : 
  days_taken = 27 → days_forgotten = 4 → days_taken + days_forgotten = 31 := by
sorry

end NUMINAMATH_CALUDE_month_days_l660_66046


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l660_66041

theorem fraction_equals_zero (x : ℝ) : 
  (x - 5) / (6 * x + 12) = 0 ↔ x = 5 ∧ 6 * x + 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l660_66041


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l660_66065

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the closed interval [-3, 0]
def interval : Set ℝ := { x | -3 ≤ x ∧ x ≤ 0 }

-- Theorem statement
theorem max_min_values_of_f :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∃ x ∈ interval, f x = 3) ∧
  (∃ x ∈ interval, f x = -17) :=
sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l660_66065


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l660_66051

/-- A sequence with common difference -/
def ArithmeticSequence (a₁ d : ℝ) (n : ℕ) := fun i : ℕ => a₁ + d * (i : ℝ)

/-- Condition for a sequence to be geometric -/
def IsGeometric (s : ℕ → ℝ) := ∀ i j k, s i * s k = s j * s j

/-- Removing one term from a sequence -/
def RemoveTerm (s : ℕ → ℝ) (k : ℕ) := fun i : ℕ => if i < k then s i else s (i + 1)

theorem arithmetic_to_geometric_sequence 
  (n : ℕ) (a₁ d : ℝ) (hn : n ≥ 4) (hd : d ≠ 0) :
  (∃ k, IsGeometric (RemoveTerm (ArithmeticSequence a₁ d n) k)) ↔ 
  (n = 4 ∧ (a₁ / d = -4 ∨ a₁ / d = 1)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l660_66051


namespace NUMINAMATH_CALUDE_smallest_odd_four_primes_l660_66085

def is_prime_factor (p n : ℕ) : Prop := Nat.Prime p ∧ p ∣ n

theorem smallest_odd_four_primes : 
  ∀ n : ℕ, 
    n % 2 = 1 → 
    (∃ p₁ p₂ p₃ p₄ : ℕ, 
      p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧
      3 < p₁ ∧
      is_prime_factor p₁ n ∧
      is_prime_factor p₂ n ∧
      is_prime_factor p₃ n ∧
      is_prime_factor p₄ n ∧
      n = p₁ * p₂ * p₃ * p₄) →
    5005 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_four_primes_l660_66085


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l660_66054

theorem sqrt_meaningful_range (a : ℝ) : (∃ x : ℝ, x^2 = a - 1) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l660_66054


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l660_66018

theorem quadratic_inequality_solution_sets (p q : ℝ) :
  (∀ x : ℝ, x^2 + p*x + q < 0 ↔ -1/2 < x ∧ x < 1/3) →
  (∀ x : ℝ, q*x^2 + p*x + 1 > 0 ↔ -2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l660_66018


namespace NUMINAMATH_CALUDE_range_of_product_l660_66084

theorem range_of_product (a b : ℝ) (ha : |a| ≤ 1) (hab : |a + b| ≤ 1) :
  -2 ≤ (a + 1) * (b + 1) ∧ (a + 1) * (b + 1) ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_product_l660_66084


namespace NUMINAMATH_CALUDE_power_fraction_equality_l660_66003

theorem power_fraction_equality : (2^4 * 3^2 * 5^3 * 7^2) / 11 = 80182 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l660_66003


namespace NUMINAMATH_CALUDE_abs_h_eq_half_l660_66092

/-- Given a quadratic equation x^2 - 4hx = 8, if the sum of squares of its roots is 20,
    then the absolute value of h is 1/2 -/
theorem abs_h_eq_half (h : ℝ) : 
  (∃ x y : ℝ, x^2 - 4*h*x = 8 ∧ y^2 - 4*h*y = 8 ∧ x^2 + y^2 = 20) → |h| = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_h_eq_half_l660_66092


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l660_66017

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (36 - 18*x - x^2 = 0) → 
  (∃ r s : ℝ, (36 - 18*r - r^2 = 0) ∧ (36 - 18*s - s^2 = 0) ∧ (r + s = -18)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l660_66017


namespace NUMINAMATH_CALUDE_jelly_beans_problem_l660_66027

theorem jelly_beans_problem (b c : ℕ) : 
  b = 3 * c →                   -- Initially, blueberry count is 3 times cherry count
  b - 20 = 4 * (c - 20) →       -- After eating 20 of each, blueberry count is 4 times cherry count
  b = 180                       -- Prove that initial blueberry count was 180
  := by sorry

end NUMINAMATH_CALUDE_jelly_beans_problem_l660_66027


namespace NUMINAMATH_CALUDE_range_of_m_l660_66082

theorem range_of_m (x y m : ℝ) : 
  (2 * x + y = -4 * m + 5) →
  (x + 2 * y = m + 4) →
  (x - y > -6) →
  (x + y < 8) →
  (-5 < m ∧ m < 7/5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l660_66082


namespace NUMINAMATH_CALUDE_rectangleAreaStage4_l660_66047

/-- The area of a rectangle formed by four squares with side lengths
    starting from 2 inches and increasing by 1 inch per stage. -/
def rectangleArea : ℕ → ℕ
| 1 => 2^2
| 2 => 2^2 + 3^2
| 3 => 2^2 + 3^2 + 4^2
| 4 => 2^2 + 3^2 + 4^2 + 5^2
| _ => 0

/-- The area of the rectangle at Stage 4 is 54 square inches. -/
theorem rectangleAreaStage4 : rectangleArea 4 = 54 := by
  sorry

end NUMINAMATH_CALUDE_rectangleAreaStage4_l660_66047


namespace NUMINAMATH_CALUDE_sum_of_continuity_points_l660_66050

/-- Piecewise function f(x) defined by n -/
noncomputable def f (n : ℝ) (x : ℝ) : ℝ :=
  if x < n then x^2 + 2*x + 3 else 3*x + 6

/-- Theorem stating that the sum of all values of n that make f(x) continuous is 2 -/
theorem sum_of_continuity_points (n : ℝ) :
  (∀ x : ℝ, ContinuousAt (f n) x) →
  (∃ n₁ n₂ : ℝ, n₁ ≠ n₂ ∧ 
    (∀ x : ℝ, ContinuousAt (f n₁) x) ∧ 
    (∀ x : ℝ, ContinuousAt (f n₂) x) ∧
    n₁ + n₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_continuity_points_l660_66050


namespace NUMINAMATH_CALUDE_unique_circle_through_three_points_l660_66096

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a function to check if three points are collinear
def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Define a function to determine the number of circles through three points
def circles_through_points (p1 p2 p3 : Point) : ℕ := sorry

-- Theorem statement
theorem unique_circle_through_three_points (p1 p2 p3 : Point) :
  ¬collinear p1 p2 p3 → circles_through_points p1 p2 p3 = 1 := by sorry

end NUMINAMATH_CALUDE_unique_circle_through_three_points_l660_66096


namespace NUMINAMATH_CALUDE_opposite_values_imply_a_half_l660_66019

theorem opposite_values_imply_a_half (a : ℚ) : (2 * a) + (1 - 4 * a) = 0 → a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_values_imply_a_half_l660_66019


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l660_66086

theorem inscribed_circle_radius (r : ℝ) :
  r > 0 →
  ∃ (R : ℝ), R > 0 ∧ R = 4 →
  r + r * Real.sqrt 2 = R →
  r = 4 * Real.sqrt 2 - 4 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l660_66086


namespace NUMINAMATH_CALUDE_base_ten_to_base_seven_l660_66031

theorem base_ten_to_base_seven : 
  ∃ (a b c d : ℕ), 
    1357 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
    a = 3 ∧ b = 6 ∧ c = 4 ∧ d = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_ten_to_base_seven_l660_66031


namespace NUMINAMATH_CALUDE_right_triangle_m_values_l660_66073

-- Define points A, B, and P
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (4, 0)
def P (m : ℝ) : ℝ × ℝ := (m, 0.5 * m + 2)

-- Define the condition for a right-angled triangle
def isRightAngled (a b c : ℝ × ℝ) : Prop :=
  (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0 ∨
  (a.1 - b.1) * (c.1 - b.1) + (a.2 - b.2) * (c.2 - b.2) = 0 ∨
  (a.1 - c.1) * (b.1 - c.1) + (a.2 - c.2) * (b.2 - c.2) = 0

-- State the theorem
theorem right_triangle_m_values :
  ∀ m : ℝ, isRightAngled A B (P m) →
    m = -2 ∨ m = 4 ∨ m = (4 * Real.sqrt 5) / 5 ∨ m = -(4 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_m_values_l660_66073


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l660_66002

theorem similar_triangle_perimeter (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a = 24) (h4 : b = 12) 
  (h5 : a = b * 2) (c : ℝ) (h6 : c = 30) :
  let scale := c / b
  let new_a := a * scale
  let new_b := b * scale
  2 * new_a + new_b = 150 := by sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l660_66002


namespace NUMINAMATH_CALUDE_frank_reading_rate_l660_66095

/-- Given a book with a certain number of chapters read over a certain number of days,
    calculate the number of chapters read per day. -/
def chapters_per_day (total_chapters : ℕ) (days : ℕ) : ℚ :=
  (total_chapters : ℚ) / (days : ℚ)

/-- Theorem: For a book with 2 chapters read over 664 days,
    the number of chapters read per day is 2/664. -/
theorem frank_reading_rate : chapters_per_day 2 664 = 2 / 664 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_rate_l660_66095


namespace NUMINAMATH_CALUDE_x_value_proof_l660_66098

theorem x_value_proof (x y z a b d : ℝ) 
  (h1 : x * y / (x + y) = a) 
  (h2 : x * z / (x + z) = b) 
  (h3 : y * z / (y - z) = d) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hd : d ≠ 0) : 
  x = a * b / (a + b) := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l660_66098


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l660_66039

/-- Calculates the number of samples to be drawn from a subgroup in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (total_sample_size : ℕ) (subgroup_size : ℕ) : ℕ :=
  (total_sample_size * subgroup_size) / total_population

/-- Theorem: In a stratified sampling scenario with a total population of 1000 and a sample size of 50,
    the number of samples drawn from a subgroup of 200 is equal to 10 -/
theorem stratified_sample_theorem :
  stratified_sample_size 1000 50 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l660_66039


namespace NUMINAMATH_CALUDE_constant_term_expansion_l660_66009

theorem constant_term_expansion (a : ℝ) : 
  a > 0 → (∃ c : ℝ, c = 80 ∧ c = (5 : ℕ).choose 4 * a^4) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l660_66009


namespace NUMINAMATH_CALUDE_minutes_in_three_and_half_hours_l660_66042

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours -/
def hours : ℚ := 3.5

/-- Theorem: The number of minutes in 3.5 hours is 210 -/
theorem minutes_in_three_and_half_hours : 
  (hours * minutes_per_hour : ℚ) = 210 := by sorry

end NUMINAMATH_CALUDE_minutes_in_three_and_half_hours_l660_66042


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l660_66011

def letter_count : ℕ := 26
def digit_count : ℕ := 10
def plate_length : ℕ := 4

def is_palindrome (s : List α) : Prop :=
  s = s.reverse

def prob_palindrome_letters : ℚ :=
  (letter_count ^ 2 : ℚ) / (letter_count ^ plate_length)

def prob_palindrome_digits : ℚ :=
  (digit_count ^ 2 : ℚ) / (digit_count ^ plate_length)

theorem license_plate_palindrome_probability :
  let prob := prob_palindrome_letters + prob_palindrome_digits - 
              prob_palindrome_letters * prob_palindrome_digits
  prob = 775 / 67600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l660_66011


namespace NUMINAMATH_CALUDE_sum_of_roots_bounds_l660_66035

theorem sum_of_roots_bounds (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : x^2 + y^2 + z^2 = 10) : 
  Real.sqrt 6 + Real.sqrt 2 ≤ Real.sqrt (6 - x^2) + Real.sqrt (6 - y^2) + Real.sqrt (6 - z^2) 
  ∧ Real.sqrt (6 - x^2) + Real.sqrt (6 - y^2) + Real.sqrt (6 - z^2) ≤ 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_bounds_l660_66035


namespace NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l660_66066

theorem exponential_function_sum_of_extrema (a : ℝ) : 
  (a > 0) → 
  (∀ x ∈ Set.Icc 0 1, ∃ y, y = a^x) →
  (a^1 + a^0 = 3) →
  (a = 2) := by
sorry

end NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l660_66066


namespace NUMINAMATH_CALUDE_obtuse_triangle_x_range_l660_66088

/-- Represents the side lengths of a triangle --/
structure TriangleSides where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is obtuse --/
def isObtuse (t : TriangleSides) : Prop :=
  (t.a ^ 2 + t.b ^ 2 < t.c ^ 2) ∨ (t.a ^ 2 + t.c ^ 2 < t.b ^ 2) ∨ (t.b ^ 2 + t.c ^ 2 < t.a ^ 2)

/-- The theorem stating the range of x for the given obtuse triangle --/
theorem obtuse_triangle_x_range :
  ∀ x : ℝ,
  let t := TriangleSides.mk 3 4 x
  isObtuse t →
  (1 < x ∧ x < Real.sqrt 7) ∨ (5 < x ∧ x < 7) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_x_range_l660_66088


namespace NUMINAMATH_CALUDE_bowling_tournament_orders_l660_66038

/-- A tournament structure with players and games. -/
structure Tournament :=
  (num_players : ℕ)
  (num_games : ℕ)
  (outcomes_per_game : ℕ)

/-- The number of possible prize distribution orders in a tournament. -/
def prize_distribution_orders (t : Tournament) : ℕ := t.outcomes_per_game ^ t.num_games

/-- The specific tournament described in the problem. -/
def bowling_tournament : Tournament :=
  { num_players := 6,
    num_games := 5,
    outcomes_per_game := 2 }

/-- Theorem stating that the number of possible prize distribution orders
    in the bowling tournament is 32. -/
theorem bowling_tournament_orders :
  prize_distribution_orders bowling_tournament = 32 := by
  sorry


end NUMINAMATH_CALUDE_bowling_tournament_orders_l660_66038


namespace NUMINAMATH_CALUDE_brian_stones_l660_66048

theorem brian_stones (total : ℕ) (white black : ℕ) (h1 : total = 100) 
  (h2 : white + black = total) 
  (h3 : white * 60 = black * 40) 
  (h4 : white > black) : white = 40 := by
  sorry

end NUMINAMATH_CALUDE_brian_stones_l660_66048


namespace NUMINAMATH_CALUDE_paint_for_similar_statues_l660_66020

-- Define the height and paint amount for the original statue
def original_height : ℝ := 8
def original_paint : ℝ := 2

-- Define the height and number of new statues
def new_height : ℝ := 2
def num_new_statues : ℕ := 360

-- Theorem statement
theorem paint_for_similar_statues :
  let surface_area_ratio := (new_height / original_height) ^ 2
  let paint_per_new_statue := original_paint * surface_area_ratio
  let total_paint := num_new_statues * paint_per_new_statue
  total_paint = 45 := by sorry

end NUMINAMATH_CALUDE_paint_for_similar_statues_l660_66020


namespace NUMINAMATH_CALUDE_closest_point_l660_66075

/-- The vector that depends on the scalar parameter s -/
def u (s : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3 + 4*s
  | 1 => -2 - 6*s
  | 2 => 1 + 2*s

/-- The constant vector b -/
def b : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 1
  | 1 => 5
  | 2 => -3

/-- The direction vector of the line -/
def v : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 4
  | 1 => -6
  | 2 => 2

/-- Theorem stating that s = 13/8 minimizes the distance between u and b -/
theorem closest_point (s : ℝ) :
  (∀ i, (u s i - b i) * v i = 0) ↔ s = 13/8 := by
  sorry

end NUMINAMATH_CALUDE_closest_point_l660_66075


namespace NUMINAMATH_CALUDE_washing_machine_cycle_time_l660_66005

theorem washing_machine_cycle_time 
  (total_items : ℕ) 
  (machine_capacity : ℕ) 
  (total_wash_time_minutes : ℕ) 
  (h1 : total_items = 60)
  (h2 : machine_capacity = 15)
  (h3 : total_wash_time_minutes = 180) :
  total_wash_time_minutes / (total_items / machine_capacity) = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_washing_machine_cycle_time_l660_66005


namespace NUMINAMATH_CALUDE_max_value_f_when_a_2_range_of_a_for_F_unique_solution_implies_m_1_l660_66097

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Theorem 1
theorem max_value_f_when_a_2 :
  ∃ (x : ℝ), x > 0 ∧ f 2 x = 0 ∧ ∀ (y : ℝ), y > 0 → f 2 y ≤ f 2 x :=
sorry

-- Theorem 2
theorem range_of_a_for_F :
  ∀ (a : ℝ), (∀ (x : ℝ), 0 < x ∧ x ≤ 3 → (deriv (F a)) x ≤ 1/2) → a ≥ 1/2 :=
sorry

-- Theorem 3
theorem unique_solution_implies_m_1 :
  ∃! (m : ℝ), m > 0 ∧ ∃! (x : ℝ), x > 0 ∧ m * (f 0 x) = x^2 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_when_a_2_range_of_a_for_F_unique_solution_implies_m_1_l660_66097


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l660_66067

-- Problem 1
theorem problem_1 : -23 + 58 - (-5) = 40 := by sorry

-- Problem 2
theorem problem_2 : (5/8 + 1/6 - 3/4) * 24 = 1 := by sorry

-- Problem 3
theorem problem_3 : -3^2 - (-5 - 0.2 / (4/5) * (-2)^2) = -3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l660_66067


namespace NUMINAMATH_CALUDE_sum_of_coefficients_plus_a_l660_66089

theorem sum_of_coefficients_plus_a (a : ℝ) (as : Fin 2007 → ℝ) :
  (∀ x : ℝ, (1 - 2 * x)^2006 = a + (Finset.sum (Finset.range 2007) (λ i => as i * x^i))) →
  Finset.sum (Finset.range 2007) (λ i => a + as i) = 2006 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_plus_a_l660_66089
