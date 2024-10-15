import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_negative_four_satisfies_l1830_183028

theorem quadratic_two_distinct_roots (c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + c = 0 ∧ y^2 - 4*y + c = 0) → c < 4 :=
by sorry

theorem negative_four_satisfies (c : ℝ) : 
  c = -4 → (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + c = 0 ∧ y^2 - 4*y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_negative_four_satisfies_l1830_183028


namespace NUMINAMATH_CALUDE_f_is_even_l1830_183022

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1830_183022


namespace NUMINAMATH_CALUDE_QED_product_l1830_183070

theorem QED_product (Q E D : ℂ) : 
  Q = 5 + 2*I ∧ E = I ∧ D = 5 - 2*I → Q * E * D = 29 * I :=
by sorry

end NUMINAMATH_CALUDE_QED_product_l1830_183070


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1830_183092

theorem gcd_of_specific_numbers (p : Nat) (h : Prime p) :
  Nat.gcd (p^10 + 1) (p^10 + p^3 + 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1830_183092


namespace NUMINAMATH_CALUDE_player_positions_satisfy_distances_l1830_183086

/-- Represents the positions of four soccer players on a number line -/
def PlayerPositions : Fin 4 → ℝ
  | 0 => 0
  | 1 => 1
  | 2 => 4
  | 3 => 6

/-- Calculates the distance between two players -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required pairwise distances -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem player_positions_satisfy_distances :
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances :=
sorry

end NUMINAMATH_CALUDE_player_positions_satisfy_distances_l1830_183086


namespace NUMINAMATH_CALUDE_quadratic_equation_root_difference_l1830_183005

theorem quadratic_equation_root_difference (k : ℝ) : 
  (∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ 2 * x₁^2 + k * x₁ + 26 = 0 ∧ 2 * x₂^2 + k * x₂ + 26 = 0) →
  Complex.abs (x₁ - x₂) = 6 →
  k = 4 * Real.sqrt 22 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_difference_l1830_183005


namespace NUMINAMATH_CALUDE_initially_tagged_fish_l1830_183016

/-- The number of fish initially caught and tagged -/
def T : ℕ := 70

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 50

/-- The number of tagged fish in the second catch -/
def tagged_in_second_catch : ℕ := 2

/-- The total number of fish in the pond -/
def total_fish : ℕ := 1750

/-- Theorem stating that T is the correct number of initially tagged fish -/
theorem initially_tagged_fish :
  (T : ℚ) / total_fish = tagged_in_second_catch / second_catch :=
by sorry

end NUMINAMATH_CALUDE_initially_tagged_fish_l1830_183016


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1830_183039

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + k = 0 → y = x) → 
  k = 4 := by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1830_183039


namespace NUMINAMATH_CALUDE_jakes_birdhouse_height_l1830_183000

/-- Represents the dimensions of a birdhouse in inches -/
structure BirdhouseDimensions where
  width : ℕ
  height : ℕ
  depth : ℕ

/-- Calculates the volume of a birdhouse given its dimensions -/
def birdhouse_volume (d : BirdhouseDimensions) : ℕ :=
  d.width * d.height * d.depth

theorem jakes_birdhouse_height :
  let sara_birdhouse : BirdhouseDimensions := {
    width := 12,  -- 1 foot = 12 inches
    height := 24, -- 2 feet = 24 inches
    depth := 24   -- 2 feet = 24 inches
  }
  let jake_birdhouse : BirdhouseDimensions := {
    width := 16,
    height := 20, -- We'll prove this is correct
    depth := 18
  }
  birdhouse_volume sara_birdhouse - birdhouse_volume jake_birdhouse = 1152 :=
by sorry


end NUMINAMATH_CALUDE_jakes_birdhouse_height_l1830_183000


namespace NUMINAMATH_CALUDE_hancho_height_calculation_l1830_183051

/-- Hancho's height in centimeters, given Hansol's height and the ratio between their heights -/
def hanchos_height (hansols_height : ℝ) (height_ratio : ℝ) : ℝ :=
  hansols_height * height_ratio

/-- Theorem stating that Hancho's height is 142.57 cm -/
theorem hancho_height_calculation :
  let hansols_height : ℝ := 134.5
  let height_ratio : ℝ := 1.06
  hanchos_height hansols_height height_ratio = 142.57 := by sorry

end NUMINAMATH_CALUDE_hancho_height_calculation_l1830_183051


namespace NUMINAMATH_CALUDE_x_minus_y_values_l1830_183042

theorem x_minus_y_values (x y : ℝ) (hx : |x| = 4) (hy : |y| = 7) (hsum : x + y > 0) :
  x - y = -3 ∨ x - y = -11 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l1830_183042


namespace NUMINAMATH_CALUDE_sphere_radius_calculation_l1830_183017

-- Define the radius of the hemisphere
def hemisphere_radius : ℝ := 2

-- Define the number of smaller spheres
def num_spheres : ℕ := 8

-- State the theorem
theorem sphere_radius_calculation :
  ∃ (r : ℝ), 
    (2 / 3 * Real.pi * hemisphere_radius ^ 3 = num_spheres * (4 / 3 * Real.pi * r ^ 3)) ∧
    (r = (Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_calculation_l1830_183017


namespace NUMINAMATH_CALUDE_patio_layout_change_l1830_183075

theorem patio_layout_change (total_tiles : ℕ) (original_rows : ℕ) (added_rows : ℕ) :
  total_tiles = 96 →
  original_rows = 8 →
  added_rows = 4 →
  (total_tiles / original_rows) - (total_tiles / (original_rows + added_rows)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_patio_layout_change_l1830_183075


namespace NUMINAMATH_CALUDE_only_translation_preserves_pattern_l1830_183048

/-- Represents a shape in the pattern -/
inductive Shape
| Triangle
| Circle

/-- Represents the infinite alternating pattern -/
def Pattern := ℕ → Shape

/-- The alternating pattern of triangles and circles -/
def alternatingPattern : Pattern :=
  fun n => if n % 2 = 0 then Shape.Triangle else Shape.Circle

/-- Represents a transformation on the pattern -/
structure Transformation :=
  (apply : Pattern → Pattern)

/-- Rotation around a point on line ℓ under a triangle apex -/
def rotationTransformation : Transformation :=
  { apply := fun _ => alternatingPattern }

/-- Translation parallel to line ℓ -/
def translationTransformation : Transformation :=
  { apply := fun p n => p (n + 2) }

/-- Reflection across a line perpendicular to line ℓ -/
def reflectionTransformation : Transformation :=
  { apply := fun p n => p (n + 1) }

/-- Checks if a transformation preserves the pattern -/
def preservesPattern (t : Transformation) : Prop :=
  ∀ n, t.apply alternatingPattern n = alternatingPattern n

theorem only_translation_preserves_pattern :
  preservesPattern translationTransformation ∧
  ¬preservesPattern rotationTransformation ∧
  ¬preservesPattern reflectionTransformation :=
sorry

end NUMINAMATH_CALUDE_only_translation_preserves_pattern_l1830_183048


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l1830_183079

theorem quadratic_root_k_value (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 3 * x - k = 0 ∧ x = 7) → k = 119 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l1830_183079


namespace NUMINAMATH_CALUDE_inequality_properties_l1830_183078

theorem inequality_properties (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  (a + b < a * b) ∧
  (abs a ≤ abs b) ∧
  (a ≥ b) ∧
  (b/a + a/b > 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_properties_l1830_183078


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1830_183004

theorem cubic_root_sum (p q s : ℝ) : 
  10 * p^3 - 25 * p^2 + 8 * p - 1 = 0 →
  10 * q^3 - 25 * q^2 + 8 * q - 1 = 0 →
  10 * s^3 - 25 * s^2 + 8 * s - 1 = 0 →
  0 < p → p < 1 →
  0 < q → q < 1 →
  0 < s → s < 1 →
  1 / (1 - p) + 1 / (1 - q) + 1 / (1 - s) = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1830_183004


namespace NUMINAMATH_CALUDE_log_inequalities_l1830_183019

theorem log_inequalities (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x < x - 1) ∧ (Real.log x > (x - 1) / x) := by
  sorry

end NUMINAMATH_CALUDE_log_inequalities_l1830_183019


namespace NUMINAMATH_CALUDE_amy_video_files_amy_video_files_proof_l1830_183054

theorem amy_video_files : ℕ → Prop :=
  fun initial_video_files =>
    let initial_music_files : ℕ := 4
    let deleted_files : ℕ := 23
    let remaining_files : ℕ := 2
    initial_music_files + initial_video_files - deleted_files = remaining_files →
    initial_video_files = 21

-- Proof
theorem amy_video_files_proof : amy_video_files 21 := by
  sorry

end NUMINAMATH_CALUDE_amy_video_files_amy_video_files_proof_l1830_183054


namespace NUMINAMATH_CALUDE_sum_of_factorials_perfect_square_l1830_183038

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem sum_of_factorials_perfect_square :
  ∀ n : ℕ, is_perfect_square (sum_of_factorials n) ↔ n = 1 ∨ n = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_factorials_perfect_square_l1830_183038


namespace NUMINAMATH_CALUDE_sqrt_fifth_power_l1830_183035

theorem sqrt_fifth_power : (Real.sqrt ((Real.sqrt 5)^4))^5 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifth_power_l1830_183035


namespace NUMINAMATH_CALUDE_defective_smartphones_l1830_183057

theorem defective_smartphones (total : ℕ) (prob : ℝ) (defective : ℕ) : 
  total = 220 → 
  prob = 0.14470734744707348 →
  (defective : ℝ) / total * ((defective : ℝ) - 1) / (total - 1) = prob →
  defective = 84 :=
by sorry

end NUMINAMATH_CALUDE_defective_smartphones_l1830_183057


namespace NUMINAMATH_CALUDE_prep_time_score_relation_student_score_for_six_hours_l1830_183012

/-- Represents the direct variation between score and preparation time -/
def score_variation (prep_time : ℝ) (score : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ score = k * prep_time

/-- Theorem stating the relationship between preparation time and test score -/
theorem prep_time_score_relation (initial_prep_time initial_score new_prep_time : ℝ) :
  initial_prep_time > 0 →
  initial_score > 0 →
  new_prep_time > 0 →
  score_variation initial_prep_time initial_score →
  score_variation new_prep_time (new_prep_time * initial_score / initial_prep_time) :=
by sorry

/-- Main theorem proving the specific case from the problem -/
theorem student_score_for_six_hours :
  let initial_prep_time : ℝ := 4
  let initial_score : ℝ := 80
  let new_prep_time : ℝ := 6
  score_variation initial_prep_time initial_score →
  score_variation new_prep_time 120 :=
by sorry

end NUMINAMATH_CALUDE_prep_time_score_relation_student_score_for_six_hours_l1830_183012


namespace NUMINAMATH_CALUDE_expected_coffee_days_expected_tea_days_expected_more_coffee_days_l1830_183056

/-- Represents the outcome of rolling a die -/
inductive DieOutcome
| Prime
| Composite
| RollAgain

/-- Represents a fair eight-sided die with the given rules -/
def fairDie : Fin 8 → DieOutcome
| 1 => DieOutcome.RollAgain
| 2 => DieOutcome.Prime
| 3 => DieOutcome.Prime
| 4 => DieOutcome.Composite
| 5 => DieOutcome.Prime
| 6 => DieOutcome.Composite
| 7 => DieOutcome.Prime
| 8 => DieOutcome.Composite

/-- The probability of getting a prime number -/
def primeProbability : ℚ := 4 / 7

/-- The probability of getting a composite number -/
def compositeProbability : ℚ := 3 / 7

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

theorem expected_coffee_days (p : ℚ) (d : ℕ) (h : p = primeProbability) : 
  ⌊p * d⌋ = 209 :=
sorry

theorem expected_tea_days (p : ℚ) (d : ℕ) (h : p = compositeProbability) : 
  ⌊p * d⌋ = 156 :=
sorry

theorem expected_more_coffee_days : 
  ⌊primeProbability * daysInYear⌋ - ⌊compositeProbability * daysInYear⌋ = 53 :=
sorry

end NUMINAMATH_CALUDE_expected_coffee_days_expected_tea_days_expected_more_coffee_days_l1830_183056


namespace NUMINAMATH_CALUDE_remainder_problem_l1830_183006

theorem remainder_problem (k : ℤ) : ∃ (x : ℤ), x = 8 * k + 1 ∧ 71 * x % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1830_183006


namespace NUMINAMATH_CALUDE_min_a_for_four_integer_solutions_l1830_183007

theorem min_a_for_four_integer_solutions : 
  let has_four_solutions (a : ℤ) := 
    (∃ x₁ x₂ x₃ x₄ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
      (x₁ - a < 0) ∧ (2 * x₁ + 3 > 0) ∧
      (x₂ - a < 0) ∧ (2 * x₂ + 3 > 0) ∧
      (x₃ - a < 0) ∧ (2 * x₃ + 3 > 0) ∧
      (x₄ - a < 0) ∧ (2 * x₄ + 3 > 0))
  ∀ a : ℤ, has_four_solutions a → a ≥ 3 ∧ has_four_solutions 3 :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_four_integer_solutions_l1830_183007


namespace NUMINAMATH_CALUDE_subtraction_problem_l1830_183025

theorem subtraction_problem (x N V : ℝ) : 
  x = 10 → 3 * x = (N - x) + V → V = 0 → N = 40 := by
sorry

end NUMINAMATH_CALUDE_subtraction_problem_l1830_183025


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l1830_183074

theorem increase_by_percentage (x : ℝ) (p : ℝ) :
  x * (1 + p / 100) = x + x * (p / 100) :=
by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 :=
by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l1830_183074


namespace NUMINAMATH_CALUDE_lewis_money_at_end_of_harvest_l1830_183064

/-- Calculates the money Lewis will have at the end of the harvest season -/
def money_at_end_of_harvest (weekly_earnings : ℕ) (weekly_rent : ℕ) (num_weeks : ℕ) : ℕ :=
  (weekly_earnings - weekly_rent) * num_weeks

/-- Proves that Lewis will have $325175 at the end of the harvest season -/
theorem lewis_money_at_end_of_harvest :
  money_at_end_of_harvest 491 216 1181 = 325175 := by
  sorry

end NUMINAMATH_CALUDE_lewis_money_at_end_of_harvest_l1830_183064


namespace NUMINAMATH_CALUDE_cuboid_missing_edge_l1830_183069

/-- Proves that for a cuboid with given dimensions and volume, the unknown edge length is 5 cm -/
theorem cuboid_missing_edge :
  let edge1 : ℝ := 2
  let edge3 : ℝ := 8
  let volume : ℝ := 80
  ∃ edge2 : ℝ, edge1 * edge2 * edge3 = volume ∧ edge2 = 5
  := by sorry

end NUMINAMATH_CALUDE_cuboid_missing_edge_l1830_183069


namespace NUMINAMATH_CALUDE_two_apples_per_slice_l1830_183020

/-- The number of apples in each slice of pie -/
def apples_per_slice (total_apples : ℕ) (num_pies : ℕ) (slices_per_pie : ℕ) : ℚ :=
  (total_apples : ℚ) / (num_pies * slices_per_pie)

/-- Theorem stating that there are 2 apples per slice given the problem conditions -/
theorem two_apples_per_slice :
  apples_per_slice (4 * 12) 4 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_apples_per_slice_l1830_183020


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1830_183041

theorem complex_number_in_second_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := 1 + 2 * i + 3 * i^2
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1830_183041


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l1830_183032

theorem z_in_third_quadrant (z : ℂ) (h : Complex.I * z = (4 + 3 * Complex.I) / (1 + 2 * Complex.I)) : 
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l1830_183032


namespace NUMINAMATH_CALUDE_range_of_a_l1830_183024

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x

-- Define the proposition P
def P (a : ℝ) : Prop := ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂

-- Define the function inside the logarithm
def g (a : ℝ) (x : ℝ) : ℝ := a*x^2 - x + a

-- Define the proposition Q
def Q (a : ℝ) : Prop := ∀ x, g a x > 0

-- Main theorem
theorem range_of_a (a : ℝ) : (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a ≤ 1/2 ∨ a > 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1830_183024


namespace NUMINAMATH_CALUDE_johns_allowance_l1830_183010

/-- John's weekly allowance problem -/
theorem johns_allowance :
  ∀ (A : ℚ),
  (A > 0) →
  (3/5 * A + 1/3 * (A - 3/5 * A) + 88/100 = A) →
  A = 33/10 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l1830_183010


namespace NUMINAMATH_CALUDE_point_transformation_l1830_183085

/-- Rotation of a point (x, y) by 180° around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ := (2*h - x, 2*k - y)

/-- Reflection of a point (x, y) about y = x -/
def reflectYEqualX (x y : ℝ) : ℝ × ℝ := (y, x)

/-- The main theorem -/
theorem point_transformation (a b : ℝ) :
  let Q : ℝ × ℝ := (a, b)
  let rotated := rotate180 a b 2 3
  let final := reflectYEqualX rotated.1 rotated.2
  final = (3, -7) → a - b = 8 := by
sorry

end NUMINAMATH_CALUDE_point_transformation_l1830_183085


namespace NUMINAMATH_CALUDE_inverse_proportion_relationship_l1830_183030

/-- Given points A(-1, y₁), B(2, y₂), and C(3, y₃) on the graph of y = -6/x,
    prove that y₁ > y₃ > y₂ -/
theorem inverse_proportion_relationship (y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = 6 / (-1))
  (h₂ : y₂ = -6 / 2)
  (h₃ : y₃ = -6 / 3) :
  y₁ > y₃ ∧ y₃ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_relationship_l1830_183030


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1830_183047

theorem quadratic_minimum (k : ℝ) : 
  (∃ x₀ ∈ Set.Icc 0 2, ∀ x ∈ Set.Icc 0 2, 
    (x^2 - 4*k*x + 4*k^2 + 2*k - 1) ≥ (x₀^2 - 4*k*x₀ + 4*k^2 + 2*k - 1)) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1830_183047


namespace NUMINAMATH_CALUDE_troy_computer_savings_l1830_183093

/-- The amount Troy needs to save to buy a new computer -/
theorem troy_computer_savings (new_computer_cost initial_savings old_computer_value : ℕ) 
  (h1 : new_computer_cost = 1800)
  (h2 : initial_savings = 350)
  (h3 : old_computer_value = 100) :
  new_computer_cost - (initial_savings + old_computer_value) = 1350 := by
  sorry

end NUMINAMATH_CALUDE_troy_computer_savings_l1830_183093


namespace NUMINAMATH_CALUDE_sum_of_roots_l1830_183037

theorem sum_of_roots (a b c d k : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  k > 0 →
  a + b = k →
  c + d = k^2 →
  c^2 - 4*a*c - 5*b = 0 →
  d^2 - 4*a*d - 5*b = 0 →
  a^2 - 4*c*a - 5*d = 0 →
  b^2 - 4*c*b - 5*d = 0 →
  a + b + c + d = k + k^2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1830_183037


namespace NUMINAMATH_CALUDE_client_cost_is_3400_l1830_183015

def ladders_cost (num_ladders1 : ℕ) (rungs_per_ladder1 : ℕ) 
                 (num_ladders2 : ℕ) (rungs_per_ladder2 : ℕ) 
                 (cost_per_rung : ℕ) : ℕ :=
  (num_ladders1 * rungs_per_ladder1 + num_ladders2 * rungs_per_ladder2) * cost_per_rung

theorem client_cost_is_3400 :
  ladders_cost 10 50 20 60 2 = 3400 := by
  sorry

end NUMINAMATH_CALUDE_client_cost_is_3400_l1830_183015


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l1830_183066

def a (n : ℕ) : ℕ := n.factorial + n

theorem max_gcd_consecutive_terms :
  ∃ (k : ℕ), k ≥ 2 ∧ 
  (∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ k) ∧
  (∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = k) :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l1830_183066


namespace NUMINAMATH_CALUDE_daniel_initial_noodles_l1830_183008

/-- The number of noodles Daniel had initially -/
def initial_noodles : ℕ := sorry

/-- The number of noodles Daniel gave to William -/
def noodles_to_william : ℕ := 15

/-- The number of noodles Daniel gave to Emily -/
def noodles_to_emily : ℕ := 20

/-- The number of noodles Daniel has left -/
def noodles_left : ℕ := 40

/-- Theorem stating that Daniel started with 75 noodles -/
theorem daniel_initial_noodles : initial_noodles = 75 := by sorry

end NUMINAMATH_CALUDE_daniel_initial_noodles_l1830_183008


namespace NUMINAMATH_CALUDE_jacoby_trip_cost_l1830_183026

def trip_cost (hourly_rate job_hours cookie_price cookies_sold 
               lottery_ticket_cost lottery_winnings sister_gift sister_count
               additional_needed : ℕ) : ℕ :=
  let job_earnings := hourly_rate * job_hours
  let cookie_earnings := cookie_price * cookies_sold
  let sister_gifts := sister_gift * sister_count
  let total_earned := job_earnings + cookie_earnings + lottery_winnings + sister_gifts
  let total_after_ticket := total_earned - lottery_ticket_cost
  total_after_ticket + additional_needed

theorem jacoby_trip_cost : 
  trip_cost 20 10 4 24 10 500 500 2 3214 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_jacoby_trip_cost_l1830_183026


namespace NUMINAMATH_CALUDE_range_of_m_l1830_183052

/-- The proposition p: "The equation x^2 + 2mx + 1 = 0 has two distinct positive roots" -/
def p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 + 2*m*x₁ + 1 = 0 ∧ x₂^2 + 2*m*x₂ + 1 = 0

/-- The proposition q: "The equation x^2 + 2(m-2)x - 3m + 10 = 0 has no real roots" -/
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2*(m-2)*x - 3*m + 10 ≠ 0

/-- The set representing the range of m -/
def S : Set ℝ := {m | m ≤ -2 ∨ (-1 ≤ m ∧ m < 3)}

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ S := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1830_183052


namespace NUMINAMATH_CALUDE_minimum_apples_l1830_183073

theorem minimum_apples (n : ℕ) (total_apples : ℕ) : 
  (∃ k : ℕ, total_apples = 25 * k + 24) →   -- Condition 1 and 2
  total_apples > 300 →                      -- Condition 3
  total_apples ≥ 324 :=                     -- Minimum number of apples
by
  sorry

#check minimum_apples

end NUMINAMATH_CALUDE_minimum_apples_l1830_183073


namespace NUMINAMATH_CALUDE_division_sum_equality_l1830_183098

theorem division_sum_equality : 3752 / (39 * 2) + 5030 / (39 * 10) = 61 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_equality_l1830_183098


namespace NUMINAMATH_CALUDE_c_monthly_income_l1830_183044

/-- Proves that C's monthly income is 17000, given the conditions from the problem -/
theorem c_monthly_income (a_annual_income : ℕ) (a_b_ratio : ℚ) (b_c_percentage : ℚ) :
  a_annual_income = 571200 →
  a_b_ratio = 5 / 2 →
  b_c_percentage = 112 / 100 →
  (a_annual_income / 12 : ℚ) * (2 / 5) / b_c_percentage = 17000 :=
by sorry

end NUMINAMATH_CALUDE_c_monthly_income_l1830_183044


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1830_183013

theorem trigonometric_identity (α : Real) :
  (Real.tan (π/4 - α/2) * (1 - Real.cos (3*π/2 - α)) / Real.cos α - 2 * Real.cos (2*α)) /
  (Real.tan (π/4 - α/2) * (1 + Real.sin (4*π + α)) / Real.cos α + 2 * Real.cos (2*α)) =
  Real.tan (π/6 + α) * Real.tan (α - π/6) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1830_183013


namespace NUMINAMATH_CALUDE_largest_x_and_multiples_l1830_183003

theorem largest_x_and_multiples :
  let x := Int.floor ((23 - 7) / -3)
  x = -6 ∧
  (x^2 * 1 = 36 ∧ x^2 * 2 = 72 ∧ x^2 * 3 = 108) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_and_multiples_l1830_183003


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l1830_183046

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l1830_183046


namespace NUMINAMATH_CALUDE_bookcase_sum_l1830_183033

theorem bookcase_sum (a₁ : ℕ) (d : ℤ) (n : ℕ) (aₙ : ℕ) : 
  a₁ = 32 → 
  d = -3 → 
  aₙ > 0 → 
  aₙ = a₁ + (n - 1) * d → 
  n * (a₁ + aₙ) = 374 → 
  (n : ℤ) * (2 * a₁ + (n - 1) * d) = 374 :=
by sorry

end NUMINAMATH_CALUDE_bookcase_sum_l1830_183033


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1830_183077

/-- A rectangle inscribed in a semicircle -/
structure InscribedRectangle where
  /-- Length of side MO of the rectangle -/
  mo : ℝ
  /-- Length of MG (equal to KO) -/
  mg : ℝ
  /-- The rectangle is inscribed in a semicircle -/
  inscribed : mo > 0 ∧ mg > 0

/-- The area of the inscribed rectangle is 240 -/
theorem inscribed_rectangle_area
  (rect : InscribedRectangle)
  (h1 : rect.mo = 20)
  (h2 : rect.mg = 12) :
  rect.mo * (rect.mg * rect.mg / rect.mo) = 240 :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1830_183077


namespace NUMINAMATH_CALUDE_trader_gain_l1830_183050

theorem trader_gain (cost selling_price : ℝ) (h1 : selling_price = 1.25 * cost) : 
  (80 * selling_price - 80 * cost) / cost = 20 := by
  sorry

#check trader_gain

end NUMINAMATH_CALUDE_trader_gain_l1830_183050


namespace NUMINAMATH_CALUDE_erasers_problem_l1830_183053

theorem erasers_problem (initial_erasers bought_erasers final_erasers : ℕ) : 
  bought_erasers = 42 ∧ final_erasers = 137 → initial_erasers = 95 :=
by sorry

end NUMINAMATH_CALUDE_erasers_problem_l1830_183053


namespace NUMINAMATH_CALUDE_equation_solutions_sum_of_squares_complex_equation_l1830_183084

theorem equation_solutions (x : ℝ) :
  (x^2 + 2) / x = 5 + 2/5 → x = 5 ∨ x = 2/5 := by sorry

theorem sum_of_squares (a b : ℝ) :
  a + 3/a = 7 ∧ b + 3/b = 7 → a^2 + b^2 = 43 := by sorry

theorem complex_equation (t k : ℝ) :
  (∃ x₁ x₂ : ℝ, 6/(x₁ - 1) = k - x₁ ∧ 6/(x₂ - 1) = k - x₂ ∧ x₁ = t + 1 ∧ x₂ = t^2 + 2) →
  k^2 - 4*k + 4*t^3 = 32 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_sum_of_squares_complex_equation_l1830_183084


namespace NUMINAMATH_CALUDE_negation_equivalence_l1830_183034

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 2 < 0) ↔ (∀ x : ℝ, x^2 + x - 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1830_183034


namespace NUMINAMATH_CALUDE_xyz_product_l1830_183091

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 3 * y = -9)
  (eq2 : y * z + 3 * z = -9)
  (eq3 : z * x + 3 * x = -9) : 
  x * y * z = -27 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l1830_183091


namespace NUMINAMATH_CALUDE_simplest_common_denominator_l1830_183083

-- Define the fractions
def fraction1 (x y : ℚ) : ℚ := 1 / (2 * x^2 * y)
def fraction2 (x y : ℚ) : ℚ := 1 / (6 * x * y^3)

-- Define the common denominator
def common_denominator (x y : ℚ) : ℚ := 6 * x^2 * y^3

-- Theorem statement
theorem simplest_common_denominator (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (a b : ℚ), 
    fraction1 x y = a / common_denominator x y ∧
    fraction2 x y = b / common_denominator x y ∧
    (∀ (c : ℚ), c > 0 → 
      (∃ (d e : ℚ), fraction1 x y = d / c ∧ fraction2 x y = e / c) →
      c ≥ common_denominator x y) :=
sorry

end NUMINAMATH_CALUDE_simplest_common_denominator_l1830_183083


namespace NUMINAMATH_CALUDE_natural_number_squares_l1830_183011

theorem natural_number_squares (x y : ℕ) : 
  1 + x + x^2 + x^3 + x^4 = y^2 ↔ (x = 0 ∧ y = 1) ∨ (x = 3 ∧ y = 11) := by
  sorry

end NUMINAMATH_CALUDE_natural_number_squares_l1830_183011


namespace NUMINAMATH_CALUDE_inequality_proof_l1830_183036

theorem inequality_proof (a b c : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0)
  (sum_one : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1830_183036


namespace NUMINAMATH_CALUDE_square_area_not_possible_l1830_183063

-- Define the points
def P : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (4, 0)
def S : ℝ × ℝ := (8, 0)

-- Define a predicate for four lines forming a square
def forms_square (l₁ l₂ l₃ l₄ : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (side : ℝ),
    side > 0 ∧
    (∀ p ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄, ∃ i j : ℤ, (p.1 - center.1)^2 + (p.2 - center.2)^2 = 2 * side^2 * (i^2 + j^2)) ∧
    (P ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄) ∧
    (Q ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄) ∧
    (R ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄) ∧
    (S ∈ l₁ ∪ l₂ ∪ l₃ ∪ l₄)

-- The theorem to prove
theorem square_area_not_possible :
  ∀ l₁ l₂ l₃ l₄ : Set (ℝ × ℝ),
  forms_square l₁ l₂ l₃ l₄ →
  ∀ side : ℝ, side^2 ≠ 26/5 :=
by sorry

end NUMINAMATH_CALUDE_square_area_not_possible_l1830_183063


namespace NUMINAMATH_CALUDE_simplify_fraction_l1830_183087

theorem simplify_fraction : (84 : ℚ) / 144 = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1830_183087


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1830_183021

/-- The line equation is satisfied by the point (2, 3) for all values of k -/
theorem fixed_point_on_line (k : ℝ) : (2*k - 1) * 2 - (k - 2) * 3 - (k + 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1830_183021


namespace NUMINAMATH_CALUDE_equation_solution_l1830_183072

theorem equation_solution : ∃ x : ℝ, 9 - 3 / (1/3) + x = 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1830_183072


namespace NUMINAMATH_CALUDE_fraction_power_four_l1830_183029

theorem fraction_power_four : (5 / 3 : ℚ) ^ 4 = 625 / 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_four_l1830_183029


namespace NUMINAMATH_CALUDE_expression_simplification_l1830_183058

theorem expression_simplification (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  (p / q + q / r + r / p - 1) * (p + q + r) +
  (p / q + q / r - r / p + 1) * (p + q - r) +
  (p / q - q / r + r / p + 1) * (p - q + r) +
  (-p / q + q / r + r / p + 1) * (-p + q + r) =
  4 * (p^2 / q + q^2 / r + r^2 / p) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1830_183058


namespace NUMINAMATH_CALUDE_class_election_votes_l1830_183067

theorem class_election_votes (total_votes : ℕ) (fiona_votes : ℕ) : 
  fiona_votes = 48 → 
  (fiona_votes : ℚ) / total_votes = 2 / 5 → 
  total_votes = 120 := by
sorry

end NUMINAMATH_CALUDE_class_election_votes_l1830_183067


namespace NUMINAMATH_CALUDE_certain_number_proof_l1830_183071

theorem certain_number_proof (x : ℝ) : 
  (x / 10) - (x / 2000) = 796 → x = 8000 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1830_183071


namespace NUMINAMATH_CALUDE_pizza_delivery_problem_l1830_183082

theorem pizza_delivery_problem (total_time : ℕ) (avg_time_per_stop : ℕ) 
  (two_pizza_stops : ℕ) (pizzas_per_two_pizza_stop : ℕ) :
  total_time = 40 →
  avg_time_per_stop = 4 →
  two_pizza_stops = 2 →
  pizzas_per_two_pizza_stop = 2 →
  ∃ (single_pizza_stops : ℕ),
    (single_pizza_stops + two_pizza_stops) * avg_time_per_stop = total_time ∧
    single_pizza_stops + two_pizza_stops * pizzas_per_two_pizza_stop = 12 :=
by sorry

end NUMINAMATH_CALUDE_pizza_delivery_problem_l1830_183082


namespace NUMINAMATH_CALUDE_additional_nails_l1830_183089

/-- Calculates the number of additional nails used in a house wall construction. -/
theorem additional_nails (total_nails : ℕ) (nails_per_plank : ℕ) (planks_needed : ℕ) :
  total_nails = 11 →
  nails_per_plank = 3 →
  planks_needed = 1 →
  total_nails - (nails_per_plank * planks_needed) = 8 := by
  sorry

#check additional_nails

end NUMINAMATH_CALUDE_additional_nails_l1830_183089


namespace NUMINAMATH_CALUDE_solve_lollipops_problem_l1830_183096

def lollipops_problem (alison_lollipops henry_lollipops diane_lollipops days : ℕ) : Prop :=
  alison_lollipops = 60 ∧
  henry_lollipops = alison_lollipops + 30 ∧
  diane_lollipops = 2 * alison_lollipops ∧
  days = 6 ∧
  (alison_lollipops + henry_lollipops + diane_lollipops) / days = 45

theorem solve_lollipops_problem :
  ∃ (alison_lollipops henry_lollipops diane_lollipops days : ℕ),
    lollipops_problem alison_lollipops henry_lollipops diane_lollipops days :=
by
  sorry

end NUMINAMATH_CALUDE_solve_lollipops_problem_l1830_183096


namespace NUMINAMATH_CALUDE_age_difference_l1830_183065

theorem age_difference (a b c : ℕ) : 
  b = 20 →
  c = b / 2 →
  a + b + c = 52 →
  a = b + 2 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1830_183065


namespace NUMINAMATH_CALUDE_remainder_of_sum_l1830_183076

def start_num : ℕ := 11085

theorem remainder_of_sum (start : ℕ) (h : start = start_num) : 
  (2 * (List.sum (List.map (λ i => start + 2 * i) (List.range 8)))) % 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_l1830_183076


namespace NUMINAMATH_CALUDE_pyramid_rows_equal_ten_l1830_183045

/-- The number of spheres in a square-based pyramid with n rows -/
def square_pyramid (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The number of spheres in a triangle-based pyramid with n rows -/
def triangle_pyramid (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The total number of spheres -/
def total_spheres : ℕ := 605

theorem pyramid_rows_equal_ten :
  ∃ (n : ℕ), n > 0 ∧ square_pyramid n + triangle_pyramid n = total_spheres := by
  sorry

end NUMINAMATH_CALUDE_pyramid_rows_equal_ten_l1830_183045


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_1_range_of_a_given_condition_l1830_183055

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |3*x + a|

-- Part 1
theorem solution_set_when_a_is_1 :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} := by sorry

-- Part 2
theorem range_of_a_given_condition (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ + 2*|x₀ - 2| < 3) → -9 < a ∧ a < -3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_1_range_of_a_given_condition_l1830_183055


namespace NUMINAMATH_CALUDE_complex_square_simplification_l1830_183001

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (5 - 3 * i)^2 = 16 - 30 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l1830_183001


namespace NUMINAMATH_CALUDE_figure_can_form_square_l1830_183049

/-- Represents a figure drawn on a grid -/
structure GridFigure where
  cells : Set (ℤ × ℤ)

/-- Represents a cut of the figure -/
structure Cut where
  piece1 : Set (ℤ × ℤ)
  piece2 : Set (ℤ × ℤ)
  piece3 : Set (ℤ × ℤ)

/-- Checks if a set of cells forms a square -/
def isSquare (s : Set (ℤ × ℤ)) : Prop :=
  ∃ (x y w : ℤ), ∀ (i j : ℤ), (i, j) ∈ s ↔ x ≤ i ∧ i < x + w ∧ y ≤ j ∧ j < y + w

/-- Theorem stating that the figure can be cut into three parts and reassembled into a square -/
theorem figure_can_form_square (f : GridFigure) :
  ∃ (c : Cut), c.piece1 ∪ c.piece2 ∪ c.piece3 = f.cells ∧
               isSquare (c.piece1 ∪ c.piece2 ∪ c.piece3) :=
sorry

end NUMINAMATH_CALUDE_figure_can_form_square_l1830_183049


namespace NUMINAMATH_CALUDE_intersection_points_form_parallelogram_l1830_183088

-- Define the circle type
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the intersection points
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry

-- Define the four circles
def circle1 : Circle := sorry
def circle2 : Circle := sorry
def circle3 : Circle := sorry
def circle4 : Circle := sorry

-- Define the properties of the circles' intersections
def three_circles_intersect (c1 c2 c3 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, (p = M ∨ p = N) ∧ 
    (‖p - c1.center‖ = c1.radius) ∧
    (‖p - c2.center‖ = c2.radius) ∧
    (‖p - c3.center‖ = c3.radius)

def two_circles_intersect (c1 c2 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, (p = A ∨ p = B ∨ p = C ∨ p = D) ∧
    (‖p - c1.center‖ = c1.radius) ∧
    (‖p - c2.center‖ = c2.radius)

-- Theorem statement
theorem intersection_points_form_parallelogram
  (h1 : circle1.radius = circle2.radius ∧ circle2.radius = circle3.radius ∧ circle3.radius = circle4.radius)
  (h2 : three_circles_intersect circle1 circle2 circle3 ∧
        three_circles_intersect circle1 circle2 circle4 ∧
        three_circles_intersect circle1 circle3 circle4 ∧
        three_circles_intersect circle2 circle3 circle4)
  (h3 : two_circles_intersect circle1 circle2 ∧
        two_circles_intersect circle1 circle3 ∧
        two_circles_intersect circle1 circle4 ∧
        two_circles_intersect circle2 circle3 ∧
        two_circles_intersect circle2 circle4 ∧
        two_circles_intersect circle3 circle4) :
  C - D = B - A :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_form_parallelogram_l1830_183088


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l1830_183099

/-- The minimum distance between a point on the line y = (5/12)x - 11 and a point on the parabola y = x² is 6311/624 -/
theorem min_distance_line_parabola :
  let line := λ x : ℝ => (5/12) * x - 11
  let parabola := λ x : ℝ => x^2
  ∃ (d : ℝ), d = 6311/624 ∧
    ∀ (x₁ x₂ : ℝ),
      d ≤ Real.sqrt ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l1830_183099


namespace NUMINAMATH_CALUDE_lcm_18_24_l1830_183040

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l1830_183040


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l1830_183009

theorem line_circle_intersection_range (a : ℝ) :
  (∃ x y : ℝ, x - y - a = 0 ∧ (x - 1)^2 + y^2 = 2) →
  -1 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l1830_183009


namespace NUMINAMATH_CALUDE_celestia_badges_l1830_183068

theorem celestia_badges (total : ℕ) (hermione : ℕ) (luna : ℕ) (celestia : ℕ)
  (h_total : total = 83)
  (h_hermione : hermione = 14)
  (h_luna : luna = 17)
  (h_sum : total = hermione + luna + celestia) :
  celestia = 52 := by
sorry

end NUMINAMATH_CALUDE_celestia_badges_l1830_183068


namespace NUMINAMATH_CALUDE_unique_balance_l1830_183095

def weights : List ℕ := [1, 2, 4, 8, 16, 32]
def candy_weight : ℕ := 25

def is_valid_partition (partition : List ℕ × List ℕ) : Prop :=
  partition.1.length = 3 ∧ 
  partition.2.length = 3 ∧
  (partition.1 ++ partition.2).toFinset = weights.toFinset

def is_balanced (partition : List ℕ × List ℕ) : Prop :=
  (partition.1.sum + candy_weight = partition.2.sum) ∧
  is_valid_partition partition

theorem unique_balance :
  ∃! partition : List ℕ × List ℕ, 
    is_balanced partition ∧ 
    partition.2.toFinset = {4, 8, 32} := by sorry

end NUMINAMATH_CALUDE_unique_balance_l1830_183095


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_l1830_183002

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  equation 0 = 0 → sum_of_roots = (-b) / a :=
by
  sorry

-- Specific instance for the given problem
theorem sum_of_solutions_specific :
  let equation := fun x => -48 * x^2 + 96 * x + 180
  let sum_of_roots := 2
  (∀ x, equation x = 0 → x = sum_of_roots ∨ x = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_l1830_183002


namespace NUMINAMATH_CALUDE_bowling_team_weight_problem_l1830_183081

theorem bowling_team_weight_problem (original_players : ℕ) 
                                    (original_avg_weight : ℝ) 
                                    (new_players : ℕ) 
                                    (known_new_player_weight : ℝ) 
                                    (new_avg_weight : ℝ) :
  original_players = 7 →
  original_avg_weight = 76 →
  new_players = 2 →
  known_new_player_weight = 60 →
  new_avg_weight = 78 →
  ∃ (unknown_new_player_weight : ℝ),
    (original_players * original_avg_weight + known_new_player_weight + unknown_new_player_weight) / 
    (original_players + new_players) = new_avg_weight ∧
    unknown_new_player_weight = 110 :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_weight_problem_l1830_183081


namespace NUMINAMATH_CALUDE_tan_120_degrees_l1830_183061

theorem tan_120_degrees : Real.tan (2 * Real.pi / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_120_degrees_l1830_183061


namespace NUMINAMATH_CALUDE_optimal_k_value_l1830_183031

theorem optimal_k_value : ∃! k : ℝ, 
  (∀ a b c d : ℝ, a ≥ -1 ∧ b ≥ -1 ∧ c ≥ -1 ∧ d ≥ -1 → 
    a^3 + b^3 + c^3 + d^3 + 1 ≥ k * (a + b + c + d)) ∧ 
  k = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_optimal_k_value_l1830_183031


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l1830_183060

def complex_mul (a b c d : ℝ) : ℂ := Complex.mk (a*c - b*d) (a*d + b*c)

theorem real_part_of_complex_product : 
  (complex_mul 1 1 1 (-2)).re = 3 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l1830_183060


namespace NUMINAMATH_CALUDE_hotel_cost_calculation_l1830_183023

theorem hotel_cost_calculation 
  (cost_per_night_per_person : ℕ) 
  (number_of_people : ℕ) 
  (number_of_nights : ℕ) 
  (h1 : cost_per_night_per_person = 40)
  (h2 : number_of_people = 3)
  (h3 : number_of_nights = 3) :
  cost_per_night_per_person * number_of_people * number_of_nights = 360 :=
by sorry

end NUMINAMATH_CALUDE_hotel_cost_calculation_l1830_183023


namespace NUMINAMATH_CALUDE_circle_ratio_after_radius_increase_l1830_183043

/-- 
For any circle with radius r, if the radius is increased by 2 units, 
the ratio of the new circumference to the new diameter is equal to π.
-/
theorem circle_ratio_after_radius_increase (r : ℝ) : 
  let new_radius : ℝ := r + 2
  let new_circumference : ℝ := 2 * Real.pi * new_radius
  let new_diameter : ℝ := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_radius_increase_l1830_183043


namespace NUMINAMATH_CALUDE_tenth_prime_is_29_l1830_183080

/-- Definition of natural numbers -/
def NaturalNumber (n : ℕ) : Prop := n ≥ 0

/-- Definition of prime numbers -/
def PrimeNumber (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 0 ∧ m < p → (p % m = 0 → m = 1)

/-- Function to get the nth prime number -/
def nthPrime (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The 10th prime number is 29 -/
theorem tenth_prime_is_29 : nthPrime 10 = 29 := by
  sorry

end NUMINAMATH_CALUDE_tenth_prime_is_29_l1830_183080


namespace NUMINAMATH_CALUDE_even_function_m_value_l1830_183062

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7*m + 12)

theorem even_function_m_value (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_m_value_l1830_183062


namespace NUMINAMATH_CALUDE_total_concert_attendance_l1830_183018

def first_concert_attendance : ℕ := 65899
def second_concert_difference : ℕ := 119

theorem total_concert_attendance : 
  let second_concert_attendance := first_concert_attendance + second_concert_difference
  let third_concert_attendance := 2 * second_concert_attendance
  first_concert_attendance + second_concert_attendance + third_concert_attendance = 263953 := by
sorry

end NUMINAMATH_CALUDE_total_concert_attendance_l1830_183018


namespace NUMINAMATH_CALUDE_fermats_little_theorem_l1830_183090

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (h : Prime p) :
  a^p ≡ a [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_fermats_little_theorem_l1830_183090


namespace NUMINAMATH_CALUDE_soccer_team_age_mode_l1830_183059

def player_ages : List ℕ := [17, 17, 18, 18, 16, 18, 17, 15, 18, 18, 17, 16, 18, 17, 18, 14]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem soccer_team_age_mode :
  mode player_ages = 18 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_age_mode_l1830_183059


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_l1830_183097

theorem min_sum_of_reciprocal_sum (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 8) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 8 ∧ (a : ℕ) + b = 36 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 8 → (c : ℕ) + d ≥ 36 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_l1830_183097


namespace NUMINAMATH_CALUDE_star_3_5_l1830_183014

-- Define the star operation
def star (a b : ℝ) : ℝ := (a + b)^2 + (a - b)^2

-- Theorem statement
theorem star_3_5 : star 3 5 = 68 := by sorry

end NUMINAMATH_CALUDE_star_3_5_l1830_183014


namespace NUMINAMATH_CALUDE_highlighter_difference_l1830_183027

/-- Proves that the difference between blue and pink highlighters is 5 --/
theorem highlighter_difference (yellow pink blue : ℕ) : 
  yellow = 7 →
  pink = yellow + 7 →
  yellow + pink + blue = 40 →
  blue - pink = 5 := by
sorry


end NUMINAMATH_CALUDE_highlighter_difference_l1830_183027


namespace NUMINAMATH_CALUDE_calculator_trick_l1830_183094

theorem calculator_trick (a b c : ℕ) (h1 : 100 ≤ a * 100 + b * 10 + c) (h2 : a * 100 + b * 10 + c < 1000) :
  let abc := a * 100 + b * 10 + c
  let abcabc := abc * 1000 + abc
  (((abcabc / 7) / 11) / 13) = abc :=
sorry

end NUMINAMATH_CALUDE_calculator_trick_l1830_183094
