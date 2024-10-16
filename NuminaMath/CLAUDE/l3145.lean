import Mathlib

namespace NUMINAMATH_CALUDE_jordans_garden_area_l3145_314574

/-- Represents a rectangular garden with fence posts --/
structure Garden where
  total_posts : ℕ
  post_spacing : ℕ
  short_side_posts : ℕ
  long_side_posts : ℕ

/-- Calculates the area of the garden in square yards --/
def garden_area (g : Garden) : ℕ :=
  (g.short_side_posts - 1) * g.post_spacing * ((g.long_side_posts - 1) * g.post_spacing)

/-- Theorem stating the area of Jordan's garden --/
theorem jordans_garden_area :
  ∀ g : Garden,
    g.total_posts = 28 →
    g.post_spacing = 3 →
    g.long_side_posts = 2 * g.short_side_posts + 3 →
    garden_area g = 630 := by
  sorry

end NUMINAMATH_CALUDE_jordans_garden_area_l3145_314574


namespace NUMINAMATH_CALUDE_coin_sum_problem_l3145_314514

def coin_values : List ℕ := [1, 5, 10, 25, 50]

def is_valid_sum (n : ℕ) : Prop :=
  ∃ (coins : List ℕ),
    coins.all (λ c => c ∈ coin_values) ∧
    coins.length = 6 ∧
    coins.sum = n

theorem coin_sum_problem :
  ¬ is_valid_sum 55 ∧
  is_valid_sum 40 ∧
  is_valid_sum 85 ∧
  is_valid_sum 105 ∧
  is_valid_sum 130 := by
  sorry

end NUMINAMATH_CALUDE_coin_sum_problem_l3145_314514


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3145_314586

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 5) :
  (a - 1)^2 - 2*a*(a - 1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3145_314586


namespace NUMINAMATH_CALUDE_average_difference_theorem_l3145_314595

theorem average_difference_theorem (x : ℤ) : 
  x ≤ 250 →
  (100 + 400) / 2 = (x + 250) / 2 + 100 →
  x = 50 := by
sorry

end NUMINAMATH_CALUDE_average_difference_theorem_l3145_314595


namespace NUMINAMATH_CALUDE_rectangle_perimeter_ratio_l3145_314554

theorem rectangle_perimeter_ratio :
  let original_width : ℚ := 6
  let original_height : ℚ := 8
  let folded_height : ℚ := original_height / 2
  let small_width : ℚ := original_width / 2
  let small_height : ℚ := folded_height
  let original_perimeter : ℚ := 2 * (original_width + original_height)
  let small_perimeter : ℚ := 2 * (small_width + small_height)
  small_perimeter / original_perimeter = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_ratio_l3145_314554


namespace NUMINAMATH_CALUDE_A_profit_share_l3145_314505

-- Define the capital shares of partners A, B, C, and D
def share_A : ℚ := 1/3
def share_B : ℚ := 1/4
def share_C : ℚ := 1/5
def share_D : ℚ := 1 - (share_A + share_B + share_C)

-- Define the total profit
def total_profit : ℕ := 2445

-- Theorem statement
theorem A_profit_share :
  (share_A * total_profit : ℚ) = 815 := by
  sorry

end NUMINAMATH_CALUDE_A_profit_share_l3145_314505


namespace NUMINAMATH_CALUDE_weight_of_seven_moles_l3145_314580

/-- The weight of a given number of moles of a compound -/
def weight_of_moles (molecular_weight : ℕ) (moles : ℕ) : ℕ :=
  molecular_weight * moles

/-- Theorem: The weight of 7 moles of a compound with molecular weight 2856 is 19992 -/
theorem weight_of_seven_moles :
  weight_of_moles 2856 7 = 19992 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_seven_moles_l3145_314580


namespace NUMINAMATH_CALUDE_sqrt_less_than_linear_l3145_314557

theorem sqrt_less_than_linear (x : ℝ) (hx : x > 0) : 
  Real.sqrt (1 + x) < 1 + x / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_less_than_linear_l3145_314557


namespace NUMINAMATH_CALUDE_range_of_a_l3145_314562

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ a then Real.cos x else 1 / x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, f a x ∈ Set.Icc (-1) 1) ↔ a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3145_314562


namespace NUMINAMATH_CALUDE_covered_area_is_56_l3145_314516

/-- Represents a rectangular strip of paper -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℝ := s.length * s.width

/-- Calculates the area of intersection between two perpendicular strips -/
def intersectionArea (s1 s2 : Strip) : ℝ := s1.width * s2.width

/-- Represents the arrangement of strips on the table -/
structure StripArrangement where
  horizontalStrips : Fin 3 → Strip
  verticalStrips : Fin 2 → Strip
  all_strips_same : ∀ (i : Fin 3) (j : Fin 2), 
    (horizontalStrips i).length = 8 ∧ (horizontalStrips i).width = 2 ∧
    (verticalStrips j).length = 8 ∧ (verticalStrips j).width = 2

/-- Calculates the total area covered by the strips -/
def coveredArea (arr : StripArrangement) : ℝ :=
  let totalStripArea := (3 * stripArea (arr.horizontalStrips 0)) + (2 * stripArea (arr.verticalStrips 0))
  let totalOverlapArea := 6 * intersectionArea (arr.horizontalStrips 0) (arr.verticalStrips 0)
  totalStripArea - totalOverlapArea

/-- Theorem stating that the covered area is 56 square units -/
theorem covered_area_is_56 (arr : StripArrangement) : coveredArea arr = 56 := by
  sorry

end NUMINAMATH_CALUDE_covered_area_is_56_l3145_314516


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_product_l3145_314546

theorem quadratic_rational_solutions_product : ∃ (c₁ c₂ : ℕ+), 
  (∀ (c : ℕ+), (∃ (x : ℚ), 5 * x^2 + 11 * x + c.val = 0) ↔ (c = c₁ ∨ c = c₂)) ∧
  c₁.val * c₂.val = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_product_l3145_314546


namespace NUMINAMATH_CALUDE_unique_y_exists_l3145_314538

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y - 3

-- Theorem statement
theorem unique_y_exists : ∃! y : ℝ, star 4 y = 17 := by
  sorry

end NUMINAMATH_CALUDE_unique_y_exists_l3145_314538


namespace NUMINAMATH_CALUDE_min_value_g_not_neg_half_l3145_314513

open Real

theorem min_value_g_not_neg_half :
  let g (x : ℝ) := -Real.sqrt 3 * Real.sin (2 * x) + 1
  ∃ x ∈ Set.Icc (π / 6) (π / 2), g x < -1/2 ∨ ∀ y ∈ Set.Icc (π / 6) (π / 2), g y > -1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_g_not_neg_half_l3145_314513


namespace NUMINAMATH_CALUDE_max_airlines_with_both_amenities_is_zero_l3145_314585

/-- Represents a type of plane -/
inductive PlaneType
| A
| B

/-- Represents whether a plane has both amenities -/
def has_both_amenities : PlaneType → Bool
| PlaneType.A => true
| PlaneType.B => false

/-- Represents a fleet composition -/
structure FleetComposition :=
  (type_a_percent : ℚ)
  (type_b_percent : ℚ)
  (sum_to_one : type_a_percent + type_b_percent = 1)
  (valid_range : 0.1 ≤ type_a_percent ∧ type_a_percent ≤ 0.9)

/-- Minimum number of planes in a fleet -/
def min_fleet_size : ℕ := 5

/-- Theorem: The maximum percentage of airlines offering both amenities on all planes is 0% -/
theorem max_airlines_with_both_amenities_is_zero :
  ∀ (fc : FleetComposition),
    ¬(∀ (plane : PlaneType), has_both_amenities plane = true) :=
by sorry

end NUMINAMATH_CALUDE_max_airlines_with_both_amenities_is_zero_l3145_314585


namespace NUMINAMATH_CALUDE_fermat_like_equation_exponent_l3145_314583

theorem fermat_like_equation_exponent (x y p n k : ℕ) : 
  x^n + y^n = p^k →
  n > 1 →
  Odd n →
  Nat.Prime p →
  Odd p →
  ∃ l : ℕ, n = p^l := by
sorry

end NUMINAMATH_CALUDE_fermat_like_equation_exponent_l3145_314583


namespace NUMINAMATH_CALUDE_queen_mary_legs_l3145_314535

/-- The total number of legs on a ship with cats and humans -/
def total_legs (total_heads : ℕ) (cat_count : ℕ) (one_legged_human_count : ℕ) : ℕ :=
  let human_count := total_heads - cat_count
  let cat_legs := cat_count * 4
  let human_legs := (human_count - one_legged_human_count) * 2 + one_legged_human_count
  cat_legs + human_legs

/-- Theorem stating the total number of legs on the Queen Mary II -/
theorem queen_mary_legs : total_legs 16 5 1 = 41 := by
  sorry

end NUMINAMATH_CALUDE_queen_mary_legs_l3145_314535


namespace NUMINAMATH_CALUDE_simplified_expression_constant_expression_l3145_314529

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2 * x^2 + 4 * x * y - 2 * x - 3
def B (x y : ℝ) : ℝ := -x^2 + x * y + 2

-- Theorem 1: Prove the simplified expression for 3A - 2(A + 2B)
theorem simplified_expression (x y : ℝ) :
  3 * A x y - 2 * (A x y + 2 * B x y) = 6 * x^2 - 2 * x - 11 := by sorry

-- Theorem 2: Prove the value of y when B + (1/2)A is constant for any x
theorem constant_expression (y : ℝ) :
  (∀ x : ℝ, ∃ c : ℝ, B x y + (1/2) * A x y = c) ↔ y = 1/3 := by sorry

end NUMINAMATH_CALUDE_simplified_expression_constant_expression_l3145_314529


namespace NUMINAMATH_CALUDE_sarah_stamp_collection_value_l3145_314527

/-- Calculates the total value of a stamp collection given the following conditions:
    - The total number of stamps in the collection
    - The number of stamps in a subset
    - The total value of the subset
    Assuming the price per stamp is constant. -/
def stamp_collection_value (total_stamps : ℕ) (subset_stamps : ℕ) (subset_value : ℚ) : ℚ :=
  (total_stamps : ℚ) * (subset_value / subset_stamps)

/-- Theorem stating that a collection of 20 stamps, where 4 stamps are worth $10,
    has a total value of $50. -/
theorem sarah_stamp_collection_value :
  stamp_collection_value 20 4 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sarah_stamp_collection_value_l3145_314527


namespace NUMINAMATH_CALUDE_problem_statement_l3145_314559

theorem problem_statement (a b : ℕ) (h_a : a ≠ 0) (h_b : b ≠ 0) 
  (h : ∀ n : ℕ, n ≥ 1 → (2^n * b + 1) ∣ (a^(2^n) - 1)) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3145_314559


namespace NUMINAMATH_CALUDE_digit_sum_proof_l3145_314540

/-- Represents the number of '1's in the original number -/
def num_ones : ℕ := 2018

/-- Represents the number of '5's in the original number -/
def num_fives : ℕ := 2017

/-- Represents the original number under the square root -/
def original_number : ℕ :=
  (10^num_ones - 1) / 9 * 10^(num_fives + 1) + 
  5 * (10^num_fives - 1) / 9 * 10^num_ones + 
  6

/-- The sum of digits in the decimal representation of the integer part 
    of the square root of the original number -/
def digit_sum : ℕ := num_ones * 3 + 4

theorem digit_sum_proof : digit_sum = 6055 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_proof_l3145_314540


namespace NUMINAMATH_CALUDE_blue_lipstick_count_l3145_314579

theorem blue_lipstick_count (total_students : ℕ) 
  (h1 : total_students = 200)
  (h2 : ∃ colored_lipstick : ℕ, colored_lipstick = total_students / 2)
  (h3 : ∃ red_lipstick : ℕ, red_lipstick = colored_lipstick / 4)
  (h4 : ∃ blue_lipstick : ℕ, blue_lipstick = red_lipstick / 5) :
  blue_lipstick = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_lipstick_count_l3145_314579


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l3145_314510

theorem fraction_sum_simplification (a b : ℝ) (h : a ≠ b) :
  a^2 / (a - b) + (2*a*b - b^2) / (b - a) = a - b := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l3145_314510


namespace NUMINAMATH_CALUDE_distinct_arrangements_count_l3145_314577

/-- A regular six-pointed star -/
structure SixPointedStar :=
  (points : Fin 12)

/-- The symmetry group of a regular six-pointed star -/
def star_symmetry_group_order : ℕ := 12

/-- The number of distinct arrangements of 12 unique objects on a regular six-pointed star,
    considering reflections and rotations as equivalent -/
def distinct_arrangements (star : SixPointedStar) : ℕ :=
  Nat.factorial 12 / star_symmetry_group_order

theorem distinct_arrangements_count :
  ∀ (star : SixPointedStar), distinct_arrangements star = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_count_l3145_314577


namespace NUMINAMATH_CALUDE_choose_3_from_10_l3145_314542

theorem choose_3_from_10 : (Nat.choose 10 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_3_from_10_l3145_314542


namespace NUMINAMATH_CALUDE_mountain_loop_trail_length_l3145_314526

/-- Represents the Mountain Loop Trail hike --/
structure MountainLoopTrail where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the hike --/
def validHike (hike : MountainLoopTrail) : Prop :=
  hike.day1 + hike.day2 = 28 ∧
  (hike.day2 + hike.day3) / 2 = 14 ∧
  hike.day4 + hike.day5 = 36 ∧
  hike.day1 + hike.day3 = 30

/-- The theorem stating the total length of the trail --/
theorem mountain_loop_trail_length (hike : MountainLoopTrail) 
  (h : validHike hike) : 
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 94 := by
  sorry


end NUMINAMATH_CALUDE_mountain_loop_trail_length_l3145_314526


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3145_314594

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 5 * z) = 7 :=
by
  -- The unique solution is z = -44/5
  use -44/5
  constructor
  -- Prove that -44/5 satisfies the equation
  · sorry
  -- Prove uniqueness
  · sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3145_314594


namespace NUMINAMATH_CALUDE_range_of_a_l3145_314500

theorem range_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 * Real.exp (y/x) - a * y^3 = 0) : a ≥ Real.exp 3 / 27 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3145_314500


namespace NUMINAMATH_CALUDE_sqrt_square_eq_x_for_nonnegative_l3145_314572

theorem sqrt_square_eq_x_for_nonnegative (x : ℝ) (h : x ≥ 0) : (Real.sqrt x)^2 = x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_x_for_nonnegative_l3145_314572


namespace NUMINAMATH_CALUDE_factorization_equality_l3145_314512

theorem factorization_equality (x y : ℝ) : x + x^2 - y - y^2 = (x + y + 1) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3145_314512


namespace NUMINAMATH_CALUDE_calen_lost_pencils_l3145_314531

theorem calen_lost_pencils (p_candy p_caleb p_calen_original p_calen_after_loss : ℕ) :
  p_candy = 9 →
  p_caleb = 2 * p_candy - 3 →
  p_calen_original = p_caleb + 5 →
  p_calen_after_loss = 10 →
  p_calen_original - p_calen_after_loss = 10 := by
sorry

end NUMINAMATH_CALUDE_calen_lost_pencils_l3145_314531


namespace NUMINAMATH_CALUDE_diagonals_not_bisect_equiv_not_p_l3145_314536

-- Define the proposition "The diagonals of a trapezoid bisect each other"
def diagonals_bisect_each_other : Prop := sorry

-- Define the proposition "The diagonals of a trapezoid do not bisect each other"
def diagonals_do_not_bisect_each_other : Prop := ¬diagonals_bisect_each_other

-- Theorem stating that the given proposition is equivalent to "not p"
theorem diagonals_not_bisect_equiv_not_p : 
  diagonals_do_not_bisect_each_other ↔ ¬diagonals_bisect_each_other :=
sorry

end NUMINAMATH_CALUDE_diagonals_not_bisect_equiv_not_p_l3145_314536


namespace NUMINAMATH_CALUDE_john_pen_difference_l3145_314506

theorem john_pen_difference (total_pens : ℕ) (blue_pens : ℕ) (black_pens : ℕ) (red_pens : ℕ) : 
  total_pens = 31 →
  blue_pens + black_pens + red_pens = total_pens →
  blue_pens = 18 →
  blue_pens = 2 * black_pens →
  black_pens > red_pens →
  black_pens - red_pens = 5 := by
  sorry

end NUMINAMATH_CALUDE_john_pen_difference_l3145_314506


namespace NUMINAMATH_CALUDE_cartesian_to_polar_equivalence_curve_transformation_l3145_314545

-- Part I
theorem cartesian_to_polar_equivalence :
  let x : ℝ := -Real.sqrt 2
  let y : ℝ := Real.sqrt 2
  let r : ℝ := 2
  let θ : ℝ := 3 * Real.pi / 4
  x = r * Real.cos θ ∧ y = r * Real.sin θ := by sorry

-- Part II
theorem curve_transformation (x y x' y' : ℝ) :
  x' = 5 * x →
  y' = 3 * y →
  (2 * x' ^ 2 + 8 * y' ^ 2 = 1) →
  (25 * x ^ 2 + 36 * y ^ 2 = 1) := by sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_equivalence_curve_transformation_l3145_314545


namespace NUMINAMATH_CALUDE_problem_solution_l3145_314548

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 31) :
  z + 1 / y = 5 / 27 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3145_314548


namespace NUMINAMATH_CALUDE_point_distance_theorem_l3145_314502

/-- A point in the first quadrant with coordinates (3-a, 2a+2) -/
structure Point (a : ℝ) where
  x : ℝ := 3 - a
  y : ℝ := 2*a + 2
  first_quadrant : 0 < x ∧ 0 < y

/-- The theorem stating that if the distance from the point to the x-axis is twice
    the distance to the y-axis, then a = 1 -/
theorem point_distance_theorem (a : ℝ) (P : Point a) :
  P.y = 2 * P.x → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_theorem_l3145_314502


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3145_314568

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_difference 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 81)
  (h_sum2 : a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 171) :
  ∃ d : ℝ, d = 10 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3145_314568


namespace NUMINAMATH_CALUDE_sequence_sum_l3145_314508

/-- Given a sequence {a_n} where the sum of its first n terms S_n = n^2,
    and a sequence {b_n} defined as b_n = 2^(a_n),
    prove that the sum of the first n terms of {b_n}, T_n, is (2/3) * (4^n - 1) -/
theorem sequence_sum (n : ℕ) (a b : ℕ → ℕ) (S T : ℕ → ℚ)
  (h_S : ∀ k, S k = k^2)
  (h_b : ∀ k, b k = 2^(a k)) :
  T n = 2/3 * (4^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3145_314508


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3145_314581

/-- An isosceles triangle with side lengths that are roots of x^2 - 4x + 3 = 0 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  is_root : base^2 - 4*base + 3 = 0 ∧ leg^2 - 4*leg + 3 = 0
  is_isosceles : base ≠ leg
  triangle_inequality : base < 2*leg

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.base + 2*t.leg

/-- Theorem: The perimeter of the isosceles triangle is 7 -/
theorem isosceles_triangle_perimeter : 
  ∀ t : IsoscelesTriangle, perimeter t = 7 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3145_314581


namespace NUMINAMATH_CALUDE_nabla_computation_l3145_314576

def nabla (a b : ℕ) : ℕ := 3 + a^b

theorem nabla_computation : nabla (nabla 3 2) 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_nabla_computation_l3145_314576


namespace NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_l3145_314565

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Statement for the correct option (A)
theorem correct_calculation : -cubeRoot 8 = -2 := by sorry

-- Statements for the incorrect options (B, C, D)
theorem incorrect_calculation_B : -abs (-3) ≠ 3 := by sorry

theorem incorrect_calculation_C : Real.sqrt 16 ≠ 4 ∧ Real.sqrt 16 ≠ -4 := by sorry

theorem incorrect_calculation_D : -(2^2) ≠ 4 := by sorry

end NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_l3145_314565


namespace NUMINAMATH_CALUDE_allocation_ways_l3145_314564

-- Define the number of doctors and nurses
def num_doctors : ℕ := 2
def num_nurses : ℕ := 4

-- Define the number of schools
def num_schools : ℕ := 2

-- Define the number of doctors and nurses needed per school
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

-- Theorem statement
theorem allocation_ways :
  (Nat.choose num_doctors doctors_per_school) * (Nat.choose num_nurses nurses_per_school) = 12 :=
sorry

end NUMINAMATH_CALUDE_allocation_ways_l3145_314564


namespace NUMINAMATH_CALUDE_revenue_calculation_l3145_314552

/-- Calculates the total revenue for a tax center given the prices and number of returns sold for each type of tax service. -/
def total_revenue (federal_price state_price quarterly_price : ℕ) 
                  (federal_sold state_sold quarterly_sold : ℕ) : ℕ :=
  federal_price * federal_sold + state_price * state_sold + quarterly_price * quarterly_sold

/-- Theorem stating that the total revenue for the day is 4400, given the specific prices and number of returns sold. -/
theorem revenue_calculation :
  total_revenue 50 30 80 60 20 10 = 4400 := by
  sorry

end NUMINAMATH_CALUDE_revenue_calculation_l3145_314552


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3145_314501

theorem sin_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3145_314501


namespace NUMINAMATH_CALUDE_sequence_square_l3145_314522

theorem sequence_square (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n - 1) →
  ∀ n : ℕ, n ≥ 1 → a n = n^2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_square_l3145_314522


namespace NUMINAMATH_CALUDE_gloria_leftover_money_l3145_314549

/-- Calculates the amount of money Gloria has left after selling her trees and buying a cabin -/
def gloria_money_left (initial_cash : ℕ) (cypress_count : ℕ) (pine_count : ℕ) (maple_count : ℕ)
  (cypress_price : ℕ) (pine_price : ℕ) (maple_price : ℕ) (cabin_price : ℕ) : ℕ :=
  let total_earned := initial_cash + cypress_count * cypress_price + pine_count * pine_price + maple_count * maple_price
  total_earned - cabin_price

/-- Theorem stating that Gloria will have $350 left after buying the cabin -/
theorem gloria_leftover_money :
  gloria_money_left 150 20 600 24 100 200 300 129000 = 350 := by
  sorry

end NUMINAMATH_CALUDE_gloria_leftover_money_l3145_314549


namespace NUMINAMATH_CALUDE_kamal_average_marks_l3145_314563

/-- Calculates the average marks given a list of obtained marks and total marks -/
def averageMarks (obtained : List ℕ) (total : List ℕ) : ℚ :=
  (obtained.sum : ℚ) / (total.sum : ℚ) * 100

theorem kamal_average_marks : 
  let obtained := [76, 60, 82, 67, 85, 78]
  let total := [120, 110, 100, 90, 100, 95]
  averageMarks obtained total = 448 / 615 * 100 := by
  sorry

#eval (448 : ℚ) / 615 * 100

end NUMINAMATH_CALUDE_kamal_average_marks_l3145_314563


namespace NUMINAMATH_CALUDE_min_value_fraction_l3145_314524

theorem min_value_fraction (x : ℝ) (h : x ≥ 0) :
  (5 * x^2 + 20 * x + 25) / (8 * (1 + x)) ≥ 65 / 16 ∧
  ∃ y : ℝ, y ≥ 0 ∧ (5 * y^2 + 20 * y + 25) / (8 * (1 + y)) = 65 / 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3145_314524


namespace NUMINAMATH_CALUDE_factorial_ratio_l3145_314570

theorem factorial_ratio : (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial 9)) = 5005 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3145_314570


namespace NUMINAMATH_CALUDE_exponent_equation_solution_l3145_314507

theorem exponent_equation_solution (a : ℝ) (m : ℝ) (h : a ≠ 0) :
  a^(m + 1) * a^(2*m - 1) = a^9 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_solution_l3145_314507


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3145_314591

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line with 60° inclination passing through the focus
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

-- Define a point in the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the intersection point A
def point_A (x y : ℝ) : Prop :=
  parabola x y ∧ line x y ∧ first_quadrant x y

-- The main theorem
theorem parabola_line_intersection :
  ∀ x y : ℝ, point_A x y → |((x, y) : ℝ × ℝ) - focus| = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3145_314591


namespace NUMINAMATH_CALUDE_exact_two_out_of_three_germinate_l3145_314573

/-- The probability of a single seed germinating -/
def p : ℚ := 4/5

/-- The total number of seeds -/
def n : ℕ := 3

/-- The number of seeds we want to germinate -/
def k : ℕ := 2

/-- The probability of exactly k out of n seeds germinating -/
def prob_k_out_of_n (p : ℚ) (n k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem exact_two_out_of_three_germinate :
  prob_k_out_of_n p n k = 48/125 := by sorry

end NUMINAMATH_CALUDE_exact_two_out_of_three_germinate_l3145_314573


namespace NUMINAMATH_CALUDE_office_age_problem_l3145_314567

theorem office_age_problem (total_persons : Nat) (avg_age_all : Nat) (num_group1 : Nat) 
  (avg_age_group1 : Nat) (num_group2 : Nat) (age_person15 : Nat) :
  total_persons = 19 →
  avg_age_all = 15 →
  num_group1 = 9 →
  avg_age_group1 = 16 →
  num_group2 = 5 →
  age_person15 = 71 →
  (((total_persons * avg_age_all) - (num_group1 * avg_age_group1) - age_person15) / num_group2) = 14 := by
  sorry

#check office_age_problem

end NUMINAMATH_CALUDE_office_age_problem_l3145_314567


namespace NUMINAMATH_CALUDE_probability_same_group_four_people_prove_probability_same_group_four_people_l3145_314517

/-- The probability that two specific people are in the same group when four people are divided into two groups. -/
theorem probability_same_group_four_people : ℚ :=
  5 / 6

/-- Proof that the probability of two specific people being in the same group when four people are divided into two groups is 5/6. -/
theorem prove_probability_same_group_four_people :
  probability_same_group_four_people = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_group_four_people_prove_probability_same_group_four_people_l3145_314517


namespace NUMINAMATH_CALUDE_frog_jump_distance_l3145_314525

theorem frog_jump_distance (grasshopper_distance : ℕ) (difference : ℕ) (frog_distance : ℕ) :
  grasshopper_distance = 13 →
  difference = 2 →
  grasshopper_distance = frog_distance + difference →
  frog_distance = 11 := by
sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l3145_314525


namespace NUMINAMATH_CALUDE_eggs_scrambled_l3145_314504

-- Define the parameters
def total_time : ℕ := 39
def time_per_sausage : ℕ := 5
def num_sausages : ℕ := 3
def time_per_egg : ℕ := 4

-- Define the theorem
theorem eggs_scrambled :
  ∃ (num_eggs : ℕ),
    num_eggs * time_per_egg = total_time - (num_sausages * time_per_sausage) ∧
    num_eggs = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_eggs_scrambled_l3145_314504


namespace NUMINAMATH_CALUDE_b_is_ten_l3145_314511

/-- The base of the number system that satisfies the given equation -/
def b : ℕ := sorry

/-- The equation that b must satisfy -/
axiom eq_condition : (3 * b + 5)^2 = 1 * b^3 + 2 * b^2 + 2 * b + 5

/-- Proof that b is the only positive integer solution -/
theorem b_is_ten : b = 10 := by sorry

end NUMINAMATH_CALUDE_b_is_ten_l3145_314511


namespace NUMINAMATH_CALUDE_suzanne_reading_difference_l3145_314596

/-- Represents the number of pages Suzanne read on Tuesday -/
def pages_tuesday (total_pages monday_pages remaining_pages : ℕ) : ℕ :=
  total_pages - monday_pages - remaining_pages

/-- The difference in pages read between Tuesday and Monday -/
def pages_difference (total_pages monday_pages remaining_pages : ℕ) : ℕ :=
  pages_tuesday total_pages monday_pages remaining_pages - monday_pages

theorem suzanne_reading_difference :
  pages_difference 64 15 18 = 16 := by sorry

end NUMINAMATH_CALUDE_suzanne_reading_difference_l3145_314596


namespace NUMINAMATH_CALUDE_derivative_of_x_exp_x_l3145_314532

noncomputable def f (x : ℝ) := x * Real.exp x

theorem derivative_of_x_exp_x :
  deriv f = fun x ↦ (1 + x) * Real.exp x := by sorry

end NUMINAMATH_CALUDE_derivative_of_x_exp_x_l3145_314532


namespace NUMINAMATH_CALUDE_lowest_cost_scheme_l3145_314530

-- Define excavator types
inductive ExcavatorType
| A
| B

-- Define the excavation capacity for each type
def excavation_capacity (t : ExcavatorType) : ℝ :=
  match t with
  | ExcavatorType.A => 30
  | ExcavatorType.B => 15

-- Define the hourly cost for each type
def hourly_cost (t : ExcavatorType) : ℝ :=
  match t with
  | ExcavatorType.A => 300
  | ExcavatorType.B => 180

-- Define the total excavation function
def total_excavation (a b : ℕ) : ℝ :=
  4 * (a * excavation_capacity ExcavatorType.A + b * excavation_capacity ExcavatorType.B)

-- Define the total cost function
def total_cost (a b : ℕ) : ℝ :=
  4 * (a * hourly_cost ExcavatorType.A + b * hourly_cost ExcavatorType.B)

-- Theorem statement
theorem lowest_cost_scheme :
  ∀ a b : ℕ,
    a + b = 12 →
    total_excavation a b ≥ 1080 →
    total_cost a b ≤ 12960 →
    total_cost 7 5 ≤ total_cost a b ∧
    total_cost 7 5 = 12000 :=
sorry

end NUMINAMATH_CALUDE_lowest_cost_scheme_l3145_314530


namespace NUMINAMATH_CALUDE_second_month_sale_l3145_314582

/-- Calculates the sale in the second month given sales for other months and the average --/
def calculate_second_month_sale (first_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) 
  (fifth_month : ℕ) (sixth_month : ℕ) (average : ℕ) : ℕ :=
  6 * average - (first_month + third_month + fourth_month + fifth_month + sixth_month)

/-- Theorem stating that the sale in the second month is 5660 --/
theorem second_month_sale : 
  calculate_second_month_sale 5420 6200 6350 6500 6470 6100 = 5660 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l3145_314582


namespace NUMINAMATH_CALUDE_no_real_roots_l3145_314599

theorem no_real_roots (a b c d : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + a*x + b ≠ 0)
  (h2 : ∀ x : ℝ, x^2 + c*x + d ≠ 0) :
  ∀ x : ℝ, x^2 + ((a+c)/2)*x + ((b+d)/2) ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_l3145_314599


namespace NUMINAMATH_CALUDE_work_completion_time_l3145_314571

theorem work_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1 / a = 1 / 6) → (1 / a + 1 / b = 1 / 4) → b = 12 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3145_314571


namespace NUMINAMATH_CALUDE_function_range_l3145_314555

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem function_range :
  ∀ y ∈ Set.Icc 2 6, ∃ x ∈ Set.Icc (-1) 2, f x = y ∧
  ∀ x ∈ Set.Icc (-1) 2, f x ∈ Set.Icc 2 6 :=
by sorry

end NUMINAMATH_CALUDE_function_range_l3145_314555


namespace NUMINAMATH_CALUDE_m_range_theorem_l3145_314558

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x y : ℝ, y = x + m - 2 → ¬(x < 0 ∧ y > 0)

def q (m : ℝ) : Prop := 0 < 1 - m ∧ m < 1

-- Define the range of m
def m_range (m : ℝ) : Prop := m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2)

-- State the theorem
theorem m_range_theorem :
  (∀ m : ℝ, ¬(p m ∧ q m)) →
  (∀ m : ℝ, p m ∨ q m) →
  ∀ m : ℝ, m_range m ↔ (p m ∨ q m) :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l3145_314558


namespace NUMINAMATH_CALUDE_floor_expression_equals_eight_l3145_314544

theorem floor_expression_equals_eight :
  ⌊(2023^3 : ℝ) / (2021 * 2022) - (2021^3 : ℝ) / (2022 * 2023)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_eight_l3145_314544


namespace NUMINAMATH_CALUDE_remainder_of_1279_divided_by_89_l3145_314534

theorem remainder_of_1279_divided_by_89 : Nat.mod 1279 89 = 33 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1279_divided_by_89_l3145_314534


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3145_314528

theorem quadratic_roots_property : ∀ x₁ x₂ : ℝ, 
  (x₁^2 + 2019*x₁ + 1 = 0) → 
  (x₂^2 + 2019*x₂ + 1 = 0) → 
  (x₁ ≠ x₂) →
  (x₁*x₂ - x₁ - x₂ = 2020) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3145_314528


namespace NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l3145_314550

theorem sin_negative_thirty_degrees : Real.sin (-(30 * π / 180)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l3145_314550


namespace NUMINAMATH_CALUDE_contrapositive_example_l3145_314518

theorem contrapositive_example : 
  (∀ x : ℝ, x > 2 → x^2 > 4) ↔ (∀ x : ℝ, x^2 ≤ 4 → x ≤ 2) := by
sorry

end NUMINAMATH_CALUDE_contrapositive_example_l3145_314518


namespace NUMINAMATH_CALUDE_friends_weekly_biking_distance_l3145_314539

/-- The total distance biked by two friends in a week -/
def total_weekly_distance (onur_daily_distance : ℕ) (hanil_extra_distance : ℕ) (days_per_week : ℕ) : ℕ :=
  (onur_daily_distance * days_per_week) + ((onur_daily_distance + hanil_extra_distance) * days_per_week)

/-- Theorem: The total distance biked by Onur and Hanil in a week is 2700 kilometers -/
theorem friends_weekly_biking_distance :
  total_weekly_distance 250 40 5 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_friends_weekly_biking_distance_l3145_314539


namespace NUMINAMATH_CALUDE_ben_lighter_than_carl_l3145_314597

/-- Given the weights of several people and their relationships, prove that Ben is 16 pounds lighter than Carl. -/
theorem ben_lighter_than_carl (al ben carl ed : ℕ) : 
  al = ben + 25 →  -- Al is 25 pounds heavier than Ben
  ed = 146 →       -- Ed weighs 146 pounds
  al = ed + 38 →   -- Ed is 38 pounds lighter than Al
  carl = 175 →     -- Carl weighs 175 pounds
  carl - ben = 16  -- Ben is 16 pounds lighter than Carl
:= by sorry

end NUMINAMATH_CALUDE_ben_lighter_than_carl_l3145_314597


namespace NUMINAMATH_CALUDE_investment_ratio_equals_return_ratio_l3145_314556

/-- Given three investors with investments in some ratio, prove that their investment ratio
    is the same as their return ratio under certain conditions. -/
theorem investment_ratio_equals_return_ratio
  (a b c : ℕ) -- investments of A, B, and C
  (ra rb rc : ℕ) -- returns of A, B, and C
  (h1 : ra = 6 * k ∧ rb = 5 * k ∧ rc = 4 * k) -- return ratio condition
  (h2 : rb = ra + 250) -- B earns 250 more than A
  (h3 : ra + rb + rc = 7250) -- total earnings
  : ∃ (m : ℕ), a = 6 * m ∧ b = 5 * m ∧ c = 4 * m := by
  sorry


end NUMINAMATH_CALUDE_investment_ratio_equals_return_ratio_l3145_314556


namespace NUMINAMATH_CALUDE_simplified_inverse_sum_l3145_314547

theorem simplified_inverse_sum (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  ((1 / x^3) + (1 / y^3) + (1 / z^3))⁻¹ = (x^3 * y^3 * z^3) / (y^3 * z^3 + x^3 * z^3 + x^3 * y^3) := by
  sorry

end NUMINAMATH_CALUDE_simplified_inverse_sum_l3145_314547


namespace NUMINAMATH_CALUDE_function_composition_ratio_l3145_314521

/-- Given two functions f and g, prove that f(g(f(3))) / g(f(g(3))) = 59/19 -/
theorem function_composition_ratio
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = 3 * x + 2)
  (hg : ∀ x, g x = 2 * x - 3) :
  f (g (f 3)) / g (f (g 3)) = 59 / 19 :=
by sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l3145_314521


namespace NUMINAMATH_CALUDE_growth_comparison_l3145_314566

theorem growth_comparison (x : ℝ) (h : x > 0) :
  (0 < x ∧ x < 1/2 → (fun y => y) x > (fun y => y^2) x) ∧
  (x > 1/2 → (fun y => y^2) x > (fun y => y) x) := by
sorry

end NUMINAMATH_CALUDE_growth_comparison_l3145_314566


namespace NUMINAMATH_CALUDE_applicant_overall_score_l3145_314541

/-- Calculates the overall score given written test and interview scores and their weights -/
def overall_score (written_score interview_score : ℝ) (written_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

/-- Theorem stating that the overall score is 72 points given the specific scores and weights -/
theorem applicant_overall_score :
  let written_score : ℝ := 80
  let interview_score : ℝ := 60
  let written_weight : ℝ := 0.6
  let interview_weight : ℝ := 0.4
  overall_score written_score interview_score written_weight interview_weight = 72 := by
  sorry

#eval overall_score 80 60 0.6 0.4

end NUMINAMATH_CALUDE_applicant_overall_score_l3145_314541


namespace NUMINAMATH_CALUDE_second_tap_empty_time_l3145_314503

/-- The time it takes for the second tap to empty the cistern -/
def T : ℝ := 8

/-- The time it takes for the first tap to fill the cistern -/
def fill_time : ℝ := 3

/-- The time it takes to fill the cistern when both taps are open -/
def both_open_time : ℝ := 4.8

/-- Theorem stating that T is the correct time for the second tap to empty the cistern -/
theorem second_tap_empty_time : 
  (1 / fill_time) - (1 / T) = (1 / both_open_time) :=
sorry

end NUMINAMATH_CALUDE_second_tap_empty_time_l3145_314503


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3145_314590

theorem linear_equation_solution :
  ∃ x : ℚ, 3 * x + 5 = 500 - (4 * x + 6 * x) ∧ x = 495 / 13 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3145_314590


namespace NUMINAMATH_CALUDE_parallel_line_l3145_314592

/-- A linear function in two variables -/
def LinearFunction (α : Type) [Ring α] := α → α → α

/-- A point in 2D space -/
structure Point (α : Type) [Ring α] where
  x : α
  y : α

/-- Theorem stating that the given equation represents a line parallel to l -/
theorem parallel_line
  {α : Type} [Field α]
  (f : LinearFunction α)
  (M N : Point α)
  (h1 : f M.x M.y = 0)
  (h2 : f N.x N.y ≠ 0) :
  ∃ (k : α), ∀ (P : Point α),
    f P.x P.y - f M.x M.y - f N.x N.y = 0 ↔ f P.x P.y = k :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_l3145_314592


namespace NUMINAMATH_CALUDE_positions_after_631_moves_l3145_314587

/-- Represents the possible positions of the dog on the hexagon -/
inductive DogPosition
  | Top
  | TopRight
  | BottomRight
  | Bottom
  | BottomLeft
  | TopLeft

/-- Represents the possible positions of the rabbit on the hexagon -/
inductive RabbitPosition
  | TopCenter
  | TopRight
  | RightUpper
  | RightLower
  | BottomRight
  | BottomCenter
  | BottomLeft
  | LeftLower
  | LeftUpper
  | TopLeft
  | LeftCenter
  | RightCenter

/-- Calculates the position of the dog after a given number of moves -/
def dogPositionAfterMoves (moves : Nat) : DogPosition :=
  match moves % 6 with
  | 0 => DogPosition.TopLeft
  | 1 => DogPosition.Top
  | 2 => DogPosition.TopRight
  | 3 => DogPosition.BottomRight
  | 4 => DogPosition.Bottom
  | 5 => DogPosition.BottomLeft
  | _ => DogPosition.Top  -- This case is unreachable, but needed for exhaustiveness

/-- Calculates the position of the rabbit after a given number of moves -/
def rabbitPositionAfterMoves (moves : Nat) : RabbitPosition :=
  match moves % 12 with
  | 0 => RabbitPosition.RightCenter
  | 1 => RabbitPosition.TopCenter
  | 2 => RabbitPosition.TopRight
  | 3 => RabbitPosition.RightUpper
  | 4 => RabbitPosition.RightLower
  | 5 => RabbitPosition.BottomRight
  | 6 => RabbitPosition.BottomCenter
  | 7 => RabbitPosition.BottomLeft
  | 8 => RabbitPosition.LeftLower
  | 9 => RabbitPosition.LeftUpper
  | 10 => RabbitPosition.TopLeft
  | 11 => RabbitPosition.LeftCenter
  | _ => RabbitPosition.TopCenter  -- This case is unreachable, but needed for exhaustiveness

theorem positions_after_631_moves :
  dogPositionAfterMoves 631 = DogPosition.Top ∧
  rabbitPositionAfterMoves 631 = RabbitPosition.BottomLeft :=
by sorry

end NUMINAMATH_CALUDE_positions_after_631_moves_l3145_314587


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3145_314509

theorem units_digit_of_product (n : ℕ) : 
  (2^2010 * 5^2011 * 11^2012) % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3145_314509


namespace NUMINAMATH_CALUDE_incorrect_average_calculation_l3145_314537

theorem incorrect_average_calculation (n : ℕ) (correct_avg : ℝ) (error : ℝ) :
  n = 10 ∧ correct_avg = 18 ∧ error = 20 →
  (n * correct_avg - error) / n = 16 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_average_calculation_l3145_314537


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3145_314551

theorem fraction_equation_solution (x : ℚ) :
  (x + 10) / (x - 4) = (x + 3) / (x - 6) → x = 48 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3145_314551


namespace NUMINAMATH_CALUDE_pipeline_construction_equation_l3145_314543

theorem pipeline_construction_equation 
  (total_length : ℝ) 
  (efficiency_increase : ℝ) 
  (days_ahead : ℝ) 
  (x : ℝ) 
  (h1 : total_length = 3000)
  (h2 : efficiency_increase = 0.2)
  (h3 : days_ahead = 10)
  (h4 : x > 0) :
  total_length / x - total_length / ((1 + efficiency_increase) * x) = days_ahead :=
sorry

end NUMINAMATH_CALUDE_pipeline_construction_equation_l3145_314543


namespace NUMINAMATH_CALUDE_advertising_time_l3145_314523

def newscast_duration : ℕ := 30
def national_news_duration : ℕ := 12
def international_news_duration : ℕ := 5
def sports_duration : ℕ := 5
def weather_forecast_duration : ℕ := 2

theorem advertising_time :
  newscast_duration - (national_news_duration + international_news_duration + sports_duration + weather_forecast_duration) = 6 := by
  sorry

end NUMINAMATH_CALUDE_advertising_time_l3145_314523


namespace NUMINAMATH_CALUDE_airway_graph_diameter_at_most_two_l3145_314589

/-- A simple graph with 20 vertices and 172 edges -/
structure AirwayGraph where
  V : Finset (Fin 20)
  E : Finset (Fin 20 × Fin 20)
  edge_count : E.card = 172
  simple : ∀ (e : Fin 20 × Fin 20), e ∈ E → e.1 ≠ e.2
  undirected : ∀ (e : Fin 20 × Fin 20), e ∈ E → (e.2, e.1) ∈ E
  at_most_one : ∀ (u v : Fin 20), u ≠ v → ({(u, v), (v, u)} ∩ E).card ≤ 1

/-- The diameter of an AirwayGraph is at most 2 -/
theorem airway_graph_diameter_at_most_two (G : AirwayGraph) :
  ∀ (u v : Fin 20), u ≠ v → ∃ (w : Fin 20), (u = w ∨ (u, w) ∈ G.E) ∧ (w = v ∨ (w, v) ∈ G.E) :=
sorry

end NUMINAMATH_CALUDE_airway_graph_diameter_at_most_two_l3145_314589


namespace NUMINAMATH_CALUDE_icosahedron_edge_probability_l3145_314561

-- Define the properties of a regular icosahedron
structure RegularIcosahedron where
  vertices : ℕ
  edges : ℕ
  connections_per_vertex : ℕ
  vertices_eq : vertices = 12
  edges_eq : edges = 30
  connections_eq : connections_per_vertex = 5

-- Define the probability function
def probability_edge_endpoints (i : RegularIcosahedron) : ℚ :=
  i.edges / (i.vertices.choose 2)

-- Theorem statement
theorem icosahedron_edge_probability :
  ∀ i : RegularIcosahedron, probability_edge_endpoints i = 5 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_icosahedron_edge_probability_l3145_314561


namespace NUMINAMATH_CALUDE_jesses_rooms_l3145_314598

theorem jesses_rooms (room_length : ℝ) (room_width : ℝ) (total_carpet : ℝ) :
  room_length = 19 →
  room_width = 18 →
  total_carpet = 6840 →
  (total_carpet / (room_length * room_width) : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_jesses_rooms_l3145_314598


namespace NUMINAMATH_CALUDE_one_fourth_of_8_4_l3145_314593

theorem one_fourth_of_8_4 : (8.4 : ℚ) / 4 = 21 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_4_l3145_314593


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l3145_314533

theorem max_value_of_sum_products (a b c d : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0) (nonneg_d : d ≥ 0)
  (sum_constraint : a + b + c + d = 120) :
  ab + bc + cd ≤ 3600 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l3145_314533


namespace NUMINAMATH_CALUDE_total_distance_12_hours_l3145_314519

def car_distance (initial_speed : ℕ) (speed_increase : ℕ) (hours : ℕ) : ℕ :=
  let speeds := List.range hours |>.map (fun h => initial_speed + h * speed_increase)
  speeds.sum

theorem total_distance_12_hours :
  car_distance 50 2 12 = 782 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_12_hours_l3145_314519


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3145_314578

/-- Given a boat that travels 11 km along a stream and 5 km against the stream in one hour,
    prove that its speed in still water is 8 km/hr. -/
theorem boat_speed_in_still_water : 
  ∀ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = 11 →
    boat_speed - stream_speed = 5 →
    boat_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3145_314578


namespace NUMINAMATH_CALUDE_weight_of_new_person_l3145_314520

theorem weight_of_new_person (initial_count : ℕ) (weight_increase : ℝ) (leaving_weight : ℝ) :
  initial_count = 12 →
  weight_increase = 4 →
  leaving_weight = 58 →
  (initial_count : ℝ) * weight_increase + leaving_weight = 106 :=
by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l3145_314520


namespace NUMINAMATH_CALUDE_b_upper_bound_l3145_314575

theorem b_upper_bound (b : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) (1/2), Real.sqrt (1 - x^2) > x + b) → 
  b < 0 := by
sorry

end NUMINAMATH_CALUDE_b_upper_bound_l3145_314575


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3145_314515

theorem complex_equation_solution (a b : ℝ) (h : b ≠ 0) :
  (Complex.I : ℂ)^2 = -1 →
  (a + b * Complex.I)^2 = -b * Complex.I →
  (a = -1/2 ∧ (b = 1/2 ∨ b = -1/2)) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3145_314515


namespace NUMINAMATH_CALUDE_same_color_probability_problem_die_l3145_314588

/-- Represents a 30-sided die with colored sides. -/
structure ColoredDie :=
  (maroon : ℕ)
  (teal : ℕ)
  (cyan : ℕ)
  (sparkly : ℕ)
  (total_sides : ℕ)
  (sum_equals_total : maroon + teal + cyan + sparkly = total_sides)

/-- Calculates the probability of rolling the same color on two identical dice. -/
def same_color_probability (die : ColoredDie) : ℚ :=
  let maroon_prob := (die.maroon : ℚ) / die.total_sides
  let teal_prob := (die.teal : ℚ) / die.total_sides
  let cyan_prob := (die.cyan : ℚ) / die.total_sides
  let sparkly_prob := (die.sparkly : ℚ) / die.total_sides
  maroon_prob ^ 2 + teal_prob ^ 2 + cyan_prob ^ 2 + sparkly_prob ^ 2

/-- The specific 30-sided die described in the problem. -/
def problem_die : ColoredDie :=
  { maroon := 5
    teal := 10
    cyan := 12
    sparkly := 3
    total_sides := 30
    sum_equals_total := by simp }

/-- Theorem stating that the probability of rolling the same color
    on two problem_die is 139/450. -/
theorem same_color_probability_problem_die :
  same_color_probability problem_die = 139 / 450 := by
  sorry

#eval same_color_probability problem_die

end NUMINAMATH_CALUDE_same_color_probability_problem_die_l3145_314588


namespace NUMINAMATH_CALUDE_relay_race_arrangements_eq_12_l3145_314569

/-- The number of ways to arrange 5 runners in a relay race with specific constraints -/
def relay_race_arrangements : ℕ :=
  let total_runners : ℕ := 5
  let specific_runners : ℕ := 2
  let other_runners : ℕ := total_runners - specific_runners
  let ways_to_arrange_specific_runners : ℕ := 2
  let ways_to_arrange_other_runners : ℕ := Nat.factorial other_runners
  ways_to_arrange_specific_runners * ways_to_arrange_other_runners

/-- Theorem stating that the number of arrangements is 12 -/
theorem relay_race_arrangements_eq_12 : relay_race_arrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_eq_12_l3145_314569


namespace NUMINAMATH_CALUDE_triangle_longest_side_l3145_314553

theorem triangle_longest_side (x : ℚ) :
  9 + (x + 5) + (2 * x + 2) = 42 →
  max 9 (max (x + 5) (2 * x + 2)) = 58 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l3145_314553


namespace NUMINAMATH_CALUDE_exponential_inverse_sum_l3145_314584

-- Define the exponential function f
def f (x : ℝ) : ℝ := sorry

-- Define the inverse function g
def g (x : ℝ) : ℝ := sorry

-- Theorem statement
theorem exponential_inverse_sum :
  (∃ (a : ℝ), ∀ (x : ℝ), f x = a^x) →  -- f is an exponential function
  (f (1 + Real.sqrt 3) * f (1 - Real.sqrt 3) = 9) →  -- Given condition
  (∀ (x : ℝ), g (f x) = x ∧ f (g x) = x) →  -- g is the inverse of f
  (g (Real.sqrt 10 + 1) + g (Real.sqrt 10 - 1) = 2) :=
by sorry

end NUMINAMATH_CALUDE_exponential_inverse_sum_l3145_314584


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3145_314560

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3145_314560
