import Mathlib

namespace NUMINAMATH_CALUDE_concentric_circles_area_l736_73637

theorem concentric_circles_area (R r : ℝ) (h1 : R > r) (h2 : r > 0) 
  (h3 : R^2 - r^2 = 2500) : 
  π * (R^2 - r^2) = 2500 * π := by
  sorry

#check concentric_circles_area

end NUMINAMATH_CALUDE_concentric_circles_area_l736_73637


namespace NUMINAMATH_CALUDE_pascal_triangle_sum_rows_7_8_l736_73677

theorem pascal_triangle_sum_rows_7_8 : ℕ := by
  -- Define the sum of numbers in row n of Pascal's Triangle
  let sum_row (n : ℕ) := 2^n
  
  -- Sum of Row 7
  let sum_row_7 := sum_row 7
  
  -- Sum of Row 8
  let sum_row_8 := sum_row 8
  
  -- Total sum of Rows 7 and 8
  let total_sum := sum_row_7 + sum_row_8
  
  -- Prove that the total sum equals 384
  have h : total_sum = 384 := by sorry
  
  exact 384


end NUMINAMATH_CALUDE_pascal_triangle_sum_rows_7_8_l736_73677


namespace NUMINAMATH_CALUDE_ellipse1_focal_points_ellipse2_focal_points_ellipses_different_focal_points_l736_73688

-- Define the ellipse equations
def ellipse1 (x y : ℝ) : Prop := x^2 / 144 + y^2 / 169 = 1
def ellipse2 (x y m : ℝ) : Prop := x^2 / m^2 + y^2 / (m^2 + 1) = 1
def ellipse3 (x y : ℝ) : Prop := x^2 / 16 + y^2 / 7 = 1
def ellipse4 (x y m : ℝ) : Prop := x^2 / (m - 5) + y^2 / (m + 4) = 1

-- Define focal points
def focal_points (a b : ℝ) : Set (ℝ × ℝ) := {(-a, 0), (a, 0)} ∪ {(0, -b), (0, b)}

-- Theorem statements
theorem ellipse1_focal_points :
  ∃ (f : Set (ℝ × ℝ)), f = focal_points 0 5 ∧ 
  ∀ (x y : ℝ), ellipse1 x y → (x, y) ∈ f := sorry

theorem ellipse2_focal_points :
  ∀ (m : ℝ), ∃ (f : Set (ℝ × ℝ)), f = focal_points 0 1 ∧ 
  ∀ (x y : ℝ), ellipse2 x y m → (x, y) ∈ f := sorry

theorem ellipses_different_focal_points :
  ∀ (m : ℝ), m > 0 →
  ¬∃ (f : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), ellipse3 x y → (x, y) ∈ f) ∧
    (∀ (x y : ℝ), ellipse4 x y m → (x, y) ∈ f) := sorry

end NUMINAMATH_CALUDE_ellipse1_focal_points_ellipse2_focal_points_ellipses_different_focal_points_l736_73688


namespace NUMINAMATH_CALUDE_microphotonics_budget_percentage_l736_73617

theorem microphotonics_budget_percentage 
  (total_degrees : ℝ)
  (home_electronics : ℝ)
  (food_additives : ℝ)
  (genetically_modified_microorganisms : ℝ)
  (industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ)
  (h1 : total_degrees = 360)
  (h2 : home_electronics = 24)
  (h3 : food_additives = 15)
  (h4 : genetically_modified_microorganisms = 29)
  (h5 : industrial_lubricants = 8)
  (h6 : basic_astrophysics_degrees = 43.2) : 
  (100 - (home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants + (basic_astrophysics_degrees / total_degrees * 100))) = 12 := by
  sorry

end NUMINAMATH_CALUDE_microphotonics_budget_percentage_l736_73617


namespace NUMINAMATH_CALUDE_john_weekly_production_l736_73659

/-- The number of widgets John makes per week -/
def widgets_per_week (widgets_per_hour : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  widgets_per_hour * hours_per_day * days_per_week

/-- Theorem stating that John makes 800 widgets per week -/
theorem john_weekly_production : 
  widgets_per_week 20 8 5 = 800 := by
  sorry

end NUMINAMATH_CALUDE_john_weekly_production_l736_73659


namespace NUMINAMATH_CALUDE_symmetry_axis_implies_ratio_l736_73694

/-- Given a function f(x) = 2m*sin(x) - n*cos(x), if x = π/3 is an axis of symmetry
    for the graph of f(x), then n/m = -2√3/3 -/
theorem symmetry_axis_implies_ratio (m n : ℝ) (h : m ≠ 0) :
  (∀ x : ℝ, 2 * m * Real.sin x - n * Real.cos x =
    2 * m * Real.sin (2 * π / 3 - x) - n * Real.cos (2 * π / 3 - x)) →
  n / m = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_implies_ratio_l736_73694


namespace NUMINAMATH_CALUDE_smallest_solutions_l736_73687

/-- The function that checks if a given positive integer k satisfies the equation cos²(k² + 6²)° = 1 --/
def satisfies_equation (k : ℕ+) : Prop :=
  (Real.cos ((k.val ^ 2 + 6 ^ 2 : ℕ) : ℝ) * Real.pi / 180) ^ 2 = 1

/-- Theorem stating that 12 and 18 are the two smallest positive integers satisfying the equation --/
theorem smallest_solutions : 
  (satisfies_equation 12) ∧ 
  (satisfies_equation 18) ∧ 
  (∀ k : ℕ+, k < 12 → ¬(satisfies_equation k)) ∧
  (∀ k : ℕ+, 12 < k → k < 18 → ¬(satisfies_equation k)) :=
sorry

end NUMINAMATH_CALUDE_smallest_solutions_l736_73687


namespace NUMINAMATH_CALUDE_sector_arc_length_l736_73692

/-- Given a circular sector with area 10 cm² and central angle 2 radians,
    the arc length of the sector is 2√10 cm. -/
theorem sector_arc_length (S : ℝ) (α : ℝ) (l : ℝ) :
  S = 10 →  -- Area of the sector
  α = 2 →   -- Central angle in radians
  l = 2 * Real.sqrt 10 -- Arc length
  := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l736_73692


namespace NUMINAMATH_CALUDE_kats_strength_training_time_l736_73678

/-- Given Kat's training schedule, prove that she spends 1 hour on strength training each session -/
theorem kats_strength_training_time (
  strength_sessions : ℕ) 
  (boxing_sessions : ℕ) 
  (boxing_hours_per_session : ℚ)
  (total_weekly_hours : ℕ) 
  (h1 : strength_sessions = 3)
  (h2 : boxing_sessions = 4)
  (h3 : boxing_hours_per_session = 3/2)
  (h4 : total_weekly_hours = 9) :
  (total_weekly_hours - boxing_sessions * boxing_hours_per_session) / strength_sessions = 1 := by
  sorry

end NUMINAMATH_CALUDE_kats_strength_training_time_l736_73678


namespace NUMINAMATH_CALUDE_expression_always_positive_l736_73680

theorem expression_always_positive (x : ℝ) : (x - 3) * (x - 5) + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_always_positive_l736_73680


namespace NUMINAMATH_CALUDE_picture_distribution_l736_73638

theorem picture_distribution (total_pictures : ℕ) 
  (first_albums : ℕ) (first_album_capacity : ℕ) (remaining_albums : ℕ) :
  total_pictures = 100 →
  first_albums = 2 →
  first_album_capacity = 15 →
  remaining_albums = 3 →
  ∃ (pictures_per_remaining_album : ℕ) (leftover : ℕ),
    pictures_per_remaining_album = 23 ∧
    leftover = 1 ∧
    total_pictures = 
      (first_albums * first_album_capacity) + 
      (remaining_albums * pictures_per_remaining_album) + 
      leftover :=
by sorry

end NUMINAMATH_CALUDE_picture_distribution_l736_73638


namespace NUMINAMATH_CALUDE_unique_solution_l736_73681

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x * y * z) = f x * f y * f z - 6 * x * y * z

/-- The main theorem stating that the only function satisfying the equation is f(x) = 2x -/
theorem unique_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) : 
  ∀ x : ℝ, f x = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l736_73681


namespace NUMINAMATH_CALUDE_sets_equality_implies_x_minus_y_l736_73676

-- Define the sets A and B
def A (x y : ℝ) : Set ℝ := {1, x, y}
def B (x y : ℝ) : Set ℝ := {1, x^2, 2*y}

-- State the theorem
theorem sets_equality_implies_x_minus_y (x y : ℝ) : 
  A x y = B x y → x - y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sets_equality_implies_x_minus_y_l736_73676


namespace NUMINAMATH_CALUDE_average_age_combined_l736_73662

theorem average_age_combined (num_students : ℕ) (avg_age_students : ℝ)
                              (num_teachers : ℕ) (avg_age_teachers : ℝ)
                              (num_parents : ℕ) (avg_age_parents : ℝ) :
  num_students = 40 →
  avg_age_students = 10 →
  num_teachers = 4 →
  avg_age_teachers = 40 →
  num_parents = 60 →
  avg_age_parents = 34 →
  (num_students * avg_age_students + num_teachers * avg_age_teachers + num_parents * avg_age_parents) /
  (num_students + num_teachers + num_parents : ℝ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l736_73662


namespace NUMINAMATH_CALUDE_multiple_of_six_squared_gt_200_lt_30_l736_73695

theorem multiple_of_six_squared_gt_200_lt_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 6 * k)
  (h2 : x^2 > 200)
  (h3 : x < 30) :
  x = 18 ∨ x = 24 :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_six_squared_gt_200_lt_30_l736_73695


namespace NUMINAMATH_CALUDE_absolute_value_and_exponents_l736_73616

theorem absolute_value_and_exponents : 
  |(-3 : ℝ)| + (Real.pi + 1)^(0 : ℝ) - (1/3 : ℝ)^(-1 : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponents_l736_73616


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l736_73674

/-- 
Given two varieties of rice mixed in a specific ratio to create a mixture with a known cost,
this theorem proves the cost of the first variety of rice.
-/
theorem rice_mixture_cost 
  (cost_second : ℝ) 
  (cost_mixture : ℝ) 
  (mix_ratio : ℝ) 
  (h1 : cost_second = 8.75)
  (h2 : cost_mixture = 7.50)
  (h3 : mix_ratio = 0.625)
  : ∃ (cost_first : ℝ), cost_first = 8.28125 := by
  sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_l736_73674


namespace NUMINAMATH_CALUDE_meat_calculation_l736_73656

/-- Given an initial amount of meat, calculate the remaining amount after using some for meatballs and spring rolls. -/
def remaining_meat (initial : ℝ) (meatball_fraction : ℝ) (spring_roll_amount : ℝ) : ℝ :=
  initial - (initial * meatball_fraction) - spring_roll_amount

/-- Theorem stating that given 20 kg of meat, using 1/4 for meatballs and 3 kg for spring rolls leaves 12 kg. -/
theorem meat_calculation :
  remaining_meat 20 (1/4) 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_meat_calculation_l736_73656


namespace NUMINAMATH_CALUDE_equal_angles_45_degrees_l736_73648

theorem equal_angles_45_degrees (α₁ α₂ α₃ : Real) : 
  0 < α₁ ∧ α₁ < π / 2 →
  0 < α₂ ∧ α₂ < π / 2 →
  0 < α₃ ∧ α₃ < π / 2 →
  Real.sin α₁ = Real.cos α₂ →
  Real.sin α₂ = Real.cos α₃ →
  Real.sin α₃ = Real.cos α₁ →
  α₁ = π / 4 ∧ α₂ = π / 4 ∧ α₃ = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_45_degrees_l736_73648


namespace NUMINAMATH_CALUDE_price_decrease_proof_l736_73661

/-- The original price of an article before a price decrease -/
def original_price : ℝ := 1300

/-- The percentage of the original price after the decrease -/
def price_decrease_percentage : ℝ := 24

/-- The price of the article after the decrease -/
def decreased_price : ℝ := 988

theorem price_decrease_proof : 
  (1 - price_decrease_percentage / 100) * original_price = decreased_price := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_proof_l736_73661


namespace NUMINAMATH_CALUDE_nancy_spend_l736_73664

/-- The cost of a set of crystal beads in dollars -/
def crystal_cost : ℕ := 9

/-- The cost of a set of metal beads in dollars -/
def metal_cost : ℕ := 10

/-- The number of crystal bead sets Nancy buys -/
def crystal_sets : ℕ := 1

/-- The number of metal bead sets Nancy buys -/
def metal_sets : ℕ := 2

/-- The total amount Nancy spends in dollars -/
def total_cost : ℕ := crystal_cost * crystal_sets + metal_cost * metal_sets

theorem nancy_spend : total_cost = 29 := by
  sorry

end NUMINAMATH_CALUDE_nancy_spend_l736_73664


namespace NUMINAMATH_CALUDE_abs_p_minus_q_equals_five_l736_73625

theorem abs_p_minus_q_equals_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) :
  |p - q| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_p_minus_q_equals_five_l736_73625


namespace NUMINAMATH_CALUDE_average_not_two_l736_73679

def data : List ℝ := [1, 1, 0, 2, 4]

theorem average_not_two : 
  (data.sum / data.length) ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_average_not_two_l736_73679


namespace NUMINAMATH_CALUDE_volume_from_vessel_b_l736_73605

def vessel_a_concentration : ℝ := 0.45
def vessel_b_concentration : ℝ := 0.30
def vessel_c_concentration : ℝ := 0.10
def vessel_a_volume : ℝ := 4
def vessel_c_volume : ℝ := 6
def resultant_concentration : ℝ := 0.26

theorem volume_from_vessel_b (x : ℝ) : 
  vessel_a_concentration * vessel_a_volume + 
  vessel_b_concentration * x + 
  vessel_c_concentration * vessel_c_volume = 
  resultant_concentration * (vessel_a_volume + x + vessel_c_volume) → 
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_volume_from_vessel_b_l736_73605


namespace NUMINAMATH_CALUDE_income_distribution_l736_73684

theorem income_distribution (total_income : ℝ) (wife_percentage : ℝ) (orphan_percentage : ℝ) 
  (final_amount : ℝ) (num_children : ℕ) :
  total_income = 1000 →
  wife_percentage = 0.2 →
  orphan_percentage = 0.1 →
  final_amount = 500 →
  num_children = 2 →
  let remaining_after_wife := total_income * (1 - wife_percentage)
  let remaining_after_orphan := remaining_after_wife * (1 - orphan_percentage)
  let amount_to_children := remaining_after_orphan - final_amount
  let amount_per_child := amount_to_children / num_children
  amount_per_child / total_income = 0.11 := by
sorry

end NUMINAMATH_CALUDE_income_distribution_l736_73684


namespace NUMINAMATH_CALUDE_shirt_trouser_combinations_l736_73665

theorem shirt_trouser_combinations (shirt_styles : ℕ) (trouser_colors : ℕ) 
  (h1 : shirt_styles = 4) (h2 : trouser_colors = 3) : 
  shirt_styles * trouser_colors = 12 := by
  sorry

end NUMINAMATH_CALUDE_shirt_trouser_combinations_l736_73665


namespace NUMINAMATH_CALUDE_probability_of_pair_after_removal_l736_73615

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (numbers : List ℕ)
  (counts : List ℕ)

/-- The probability of selecting a pair from the remaining deck -/
def probability_of_pair (d : Deck) : ℚ :=
  83 / 1035

theorem probability_of_pair_after_removal (d : Deck) : 
  d.total = 50 ∧ 
  d.numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ 
  d.counts = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5] →
  let remaining_deck := {
    total := d.total - 4,
    numbers := d.numbers,
    counts := d.counts.map (fun c => if c = 5 then 3 else 5)
  }
  probability_of_pair remaining_deck = 83 / 1035 := by
sorry

#eval 83 + 1035  -- Should output 1118

end NUMINAMATH_CALUDE_probability_of_pair_after_removal_l736_73615


namespace NUMINAMATH_CALUDE_value_of_expression_l736_73620

theorem value_of_expression (s t : ℝ) 
  (hs : 19 * s^2 + 99 * s + 1 = 0)
  (ht : t^2 + 99 * t + 19 = 0)
  (hst : s * t ≠ 1) :
  (s * t + 4 * s + 1) / t = -5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l736_73620


namespace NUMINAMATH_CALUDE_apple_sorting_probability_l736_73641

/-- Ratio of large apples to small apples -/
def largeToSmallRatio : ℚ := 9/1

/-- Probability of sorting a large apple as a small apple -/
def largeSortedAsSmall : ℚ := 5/100

/-- Probability of sorting a small apple as a large apple -/
def smallSortedAsLarge : ℚ := 2/100

/-- The probability that a "large apple" selected after sorting is indeed a large apple -/
def probLargeGivenSortedLarge : ℚ := 855/857

theorem apple_sorting_probability :
  let totalApples : ℚ := 10
  let largeApples : ℚ := (largeToSmallRatio * totalApples) / (largeToSmallRatio + 1)
  let smallApples : ℚ := totalApples - largeApples
  let probLarge : ℚ := largeApples / totalApples
  let probSmall : ℚ := smallApples / totalApples
  let probLargeSortedLarge : ℚ := 1 - largeSortedAsSmall
  let probLargeAndSortedLarge : ℚ := probLarge * probLargeSortedLarge
  let probSmallAndSortedLarge : ℚ := probSmall * smallSortedAsLarge
  let probSortedLarge : ℚ := probLargeAndSortedLarge + probSmallAndSortedLarge
  probLargeGivenSortedLarge = probLargeAndSortedLarge / probSortedLarge :=
by sorry

end NUMINAMATH_CALUDE_apple_sorting_probability_l736_73641


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_l736_73600

/-- The number of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_elements (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- Theorem: The number of elements in the first 30 rows of Pascal's Triangle is 465 -/
theorem pascal_triangle_30_rows : pascal_triangle_elements 29 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_l736_73600


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l736_73673

theorem quadratic_root_difference : 
  let a : ℝ := 5 + 3 * Real.sqrt 5
  let b : ℝ := 5 + Real.sqrt 5
  let c : ℝ := -3
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  abs (root1 - root2) = 1/2 + 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l736_73673


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l736_73647

/-- Represents a repeating decimal with a single repeating digit -/
def repeatingDecimal (wholePart : ℚ) (repeatingDigit : ℕ) : ℚ :=
  wholePart + (repeatingDigit : ℚ) / 99

theorem product_of_repeating_decimals :
  (repeatingDecimal 0 3) * (repeatingDecimal 0 81) = 9 / 363 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l736_73647


namespace NUMINAMATH_CALUDE_value_of_expression_l736_73606

theorem value_of_expression (m n : ℤ) (h1 : |m| = 5) (h2 : |n| = 4) (h3 : m * n < 0) :
  m^2 - m*n + n = 41 ∨ m^2 - m*n + n = 49 :=
sorry

end NUMINAMATH_CALUDE_value_of_expression_l736_73606


namespace NUMINAMATH_CALUDE_square_area_calculation_l736_73642

theorem square_area_calculation (s : ℝ) (r : ℝ) (l : ℝ) (b : ℝ) : 
  r = s →                -- radius of circle equals side of square
  l = (1 / 6) * r →      -- length of rectangle is one-sixth of circle radius
  l * b = 360 →          -- area of rectangle is 360 sq. units
  b = 10 →               -- breadth of rectangle is 10 units
  s^2 = 46656 :=         -- area of square is 46656 sq. units
by
  sorry

end NUMINAMATH_CALUDE_square_area_calculation_l736_73642


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l736_73604

/-- Two points in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of parallel to x-axis for a line segment -/
def parallelToXAxis (p1 p2 : Point2D) : Prop :=
  p1.y = p2.y

theorem parallel_line_k_value (A B : Point2D) (k : ℝ) 
    (hA : A = ⟨2, 3⟩) 
    (hB : B = ⟨4, k⟩) 
    (hParallel : parallelToXAxis A B) : 
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l736_73604


namespace NUMINAMATH_CALUDE_decimal_expansion_18_37_l736_73630

/-- The decimal expansion of 18/37 has a repeating pattern of length 3 -/
def decimal_expansion_period (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧
  (18 : ℚ) / 37 = (a * 100 + b * 10 + c : ℚ) / 999

/-- The 123rd digit after the decimal point in the expansion of 18/37 -/
def digit_123 : ℕ := 6

theorem decimal_expansion_18_37 :
  decimal_expansion_period 3 ∧ digit_123 = 6 :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_18_37_l736_73630


namespace NUMINAMATH_CALUDE_like_terms_exponent_product_l736_73612

theorem like_terms_exponent_product (a b : ℝ) (m n : ℤ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ 3 * a^m * b^2 = k * (-a^2 * b^(n+3))) → m * n = -2 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_product_l736_73612


namespace NUMINAMATH_CALUDE_min_value_of_expression_l736_73696

theorem min_value_of_expression (x y z : ℝ) (h : x + y + 3 * z = 6) :
  ∃ (m : ℝ), m = 0 ∧ ∀ (x' y' z' : ℝ), x' + y' + 3 * z' = 6 → x' * y' + 2 * x' * z' + 3 * y' * z' ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l736_73696


namespace NUMINAMATH_CALUDE_candy_distribution_l736_73607

/-- Given a total number of candy pieces and the number of pieces per student,
    calculate the number of students. -/
def number_of_students (total_candy : ℕ) (candy_per_student : ℕ) : ℕ :=
  total_candy / candy_per_student

theorem candy_distribution (total_candy : ℕ) (candy_per_student : ℕ) 
    (h1 : total_candy = 344) 
    (h2 : candy_per_student = 8) :
    number_of_students total_candy candy_per_student = 43 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l736_73607


namespace NUMINAMATH_CALUDE_danielles_rooms_l736_73689

theorem danielles_rooms (heidi danielle grant : ℕ) : 
  heidi = 3 * danielle →
  grant = heidi / 9 →
  grant = 2 →
  danielle = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_danielles_rooms_l736_73689


namespace NUMINAMATH_CALUDE_c_share_l736_73618

def total_amount : ℕ := 880

def share_ratio (a b c : ℕ) : Prop :=
  4 * a = 5 * b ∧ 5 * b = 10 * c

theorem c_share (a b c : ℕ) (h1 : share_ratio a b c) (h2 : a + b + c = total_amount) :
  c = 160 := by
  sorry

end NUMINAMATH_CALUDE_c_share_l736_73618


namespace NUMINAMATH_CALUDE_all_error_types_cause_random_errors_at_least_three_random_error_causes_l736_73654

-- Define the types of errors
inductive ErrorType
  | ApproximationError
  | OmittedVariableError
  | ObservationError

-- Define a predicate for causes of random errors
def is_random_error_cause (error_type : ErrorType) : Prop :=
  match error_type with
  | ErrorType.ApproximationError => true
  | ErrorType.OmittedVariableError => true
  | ErrorType.ObservationError => true

-- Theorem stating that all three error types are causes of random errors
theorem all_error_types_cause_random_errors :
  (∀ (error_type : ErrorType), is_random_error_cause error_type) :=
by
  sorry

-- Theorem stating that there are at least three distinct causes of random errors
theorem at_least_three_random_error_causes :
  ∃ (e1 e2 e3 : ErrorType),
    e1 ≠ e2 ∧ e1 ≠ e3 ∧ e2 ≠ e3 ∧
    is_random_error_cause e1 ∧
    is_random_error_cause e2 ∧
    is_random_error_cause e3 :=
by
  sorry

end NUMINAMATH_CALUDE_all_error_types_cause_random_errors_at_least_three_random_error_causes_l736_73654


namespace NUMINAMATH_CALUDE_two_year_inflation_rate_real_yield_bank_deposit_l736_73658

-- Define the annual inflation rate
def annual_inflation_rate : ℝ := 0.015

-- Define the nominal annual yield of the bank deposit
def nominal_annual_yield : ℝ := 0.07

-- Theorem for two-year inflation rate
theorem two_year_inflation_rate :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 3.0225 := by sorry

-- Theorem for real yield of bank deposit
theorem real_yield_bank_deposit :
  ((1 + nominal_annual_yield)^2 / (1 + ((1 + annual_inflation_rate)^2 - 1)) - 1) * 100 = 11.13 := by sorry

end NUMINAMATH_CALUDE_two_year_inflation_rate_real_yield_bank_deposit_l736_73658


namespace NUMINAMATH_CALUDE_distance_proof_l736_73693

def point : ℝ × ℝ × ℝ := (2, 1, -5)

def line_point : ℝ × ℝ × ℝ := (4, -3, 2)
def line_direction : ℝ × ℝ × ℝ := (-1, 4, 3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_proof : 
  distance_to_line point line_point line_direction = Real.sqrt (34489 / 676) := by
  sorry

end NUMINAMATH_CALUDE_distance_proof_l736_73693


namespace NUMINAMATH_CALUDE_trader_profit_l736_73624

theorem trader_profit (original_price : ℝ) (original_price_positive : original_price > 0) : 
  let discount_rate : ℝ := 0.4
  let increase_rate : ℝ := 0.8
  let purchase_price : ℝ := original_price * (1 - discount_rate)
  let selling_price : ℝ := purchase_price * (1 + increase_rate)
  let profit : ℝ := selling_price - original_price
  let profit_percentage : ℝ := (profit / original_price) * 100
  profit_percentage = 8 := by sorry

end NUMINAMATH_CALUDE_trader_profit_l736_73624


namespace NUMINAMATH_CALUDE_min_value_on_interval_l736_73670

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = -15 ∧ ∀ y ∈ Set.Icc 0 3, f y ≥ f x := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l736_73670


namespace NUMINAMATH_CALUDE_common_point_and_tangent_l736_73632

theorem common_point_and_tangent (t : ℝ) (h : t ≠ 0) :
  let f := fun x : ℝ => x^3 + a*x
  let g := fun x : ℝ => b*x^2 + c
  let f' := fun x : ℝ => 3*x^2 + a
  let g' := fun x : ℝ => 2*b*x
  f t = 0 ∧ g t = 0 ∧ f' t = g' t →
  a = -t^2 ∧ b = t ∧ c = -t^3 :=
by sorry

end NUMINAMATH_CALUDE_common_point_and_tangent_l736_73632


namespace NUMINAMATH_CALUDE_class_size_is_40_l736_73634

/-- Represents the heights of rectangles in a histogram --/
structure HistogramHeights where
  ratios : List Nat
  first_frequency : Nat

/-- Calculates the total number of students represented by a histogram --/
def totalStudents (h : HistogramHeights) : Nat :=
  let unit_frequency := h.first_frequency / h.ratios.head!
  unit_frequency * h.ratios.sum

/-- Theorem stating that for the given histogram, the total number of students is 40 --/
theorem class_size_is_40 (h : HistogramHeights) 
    (height_ratio : h.ratios = [4, 3, 7, 6]) 
    (first_freq : h.first_frequency = 8) : 
  totalStudents h = 40 := by
  sorry

#eval totalStudents { ratios := [4, 3, 7, 6], first_frequency := 8 }

end NUMINAMATH_CALUDE_class_size_is_40_l736_73634


namespace NUMINAMATH_CALUDE_sunflower_contest_total_l736_73672

/-- Represents the total number of seeds eaten in a sunflower eating contest -/
def total_seeds_eaten (player1 player2 player3 : ℕ) : ℕ :=
  player1 + player2 + player3

/-- Theorem stating the total number of seeds eaten in the contest -/
theorem sunflower_contest_total :
  let player1 := 78
  let player2 := 53
  let player3 := player2 + 30
  total_seeds_eaten player1 player2 player3 = 214 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_contest_total_l736_73672


namespace NUMINAMATH_CALUDE_inequality_proof_l736_73613

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (a^4 + b^4) ≥ (a^2 + b^2) * (a^3 + b^3) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l736_73613


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l736_73663

-- Define the circles O₁ and O₂
def O₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def O₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Define the trajectory of the center M
def trajectory (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- State the theorem
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x₁ y₁ : ℝ), O₁ x₁ y₁ → (x - x₁)^2 + (y - y₁)^2 = (1 + r)^2) ∧
    (∀ (x₂ y₂ : ℝ), O₂ x₂ y₂ → (x - x₂)^2 + (y - y₂)^2 = (9 - r)^2)) →
  trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l736_73663


namespace NUMINAMATH_CALUDE_binary_is_largest_l736_73601

/-- Convert a number from base b to decimal --/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

/-- The given numbers in their respective bases --/
def binary : List Nat := [1, 1, 0, 1, 1]
def base_4 : List Nat := [3, 0, 1]
def base_5 : List Nat := [4, 4]
def decimal : Nat := 25

/-- Theorem stating that the binary number is the largest --/
theorem binary_is_largest :
  let a := to_decimal binary 2
  let b := to_decimal base_4 4
  let c := to_decimal base_5 5
  let d := decimal
  a > b ∧ a > c ∧ a > d :=
by sorry


end NUMINAMATH_CALUDE_binary_is_largest_l736_73601


namespace NUMINAMATH_CALUDE_meaningful_expression_l736_73683

theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x = (Real.sqrt (a + 1)) / (a - 2)) ↔ (a ≥ -1 ∧ a ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l736_73683


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_length_l736_73645

theorem right_triangle_shorter_leg_length 
  (a : ℝ)  -- length of the shorter leg
  (h1 : a > 0)  -- ensure positive length
  (h2 : (a^2 + (2*a)^2)^(1/2) = a * 5^(1/2))  -- Pythagorean theorem
  (h3 : 12 = (1/2) * a * 5^(1/2))  -- median to hypotenuse formula
  : a = 24 * 5^(1/2) / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_length_l736_73645


namespace NUMINAMATH_CALUDE_hamburger_sales_l736_73621

theorem hamburger_sales (total_target : ℕ) (price_per_hamburger : ℕ) (remaining_hamburgers : ℕ) : 
  total_target = 50 →
  price_per_hamburger = 5 →
  remaining_hamburgers = 4 →
  (total_target - remaining_hamburgers * price_per_hamburger) / price_per_hamburger = 6 :=
by sorry

end NUMINAMATH_CALUDE_hamburger_sales_l736_73621


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_l736_73609

theorem log_equality_implies_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.log a / Real.log 9 = Real.log b / Real.log 12 ∧ 
       Real.log a / Real.log 9 = Real.log (a + b) / Real.log 16) : 
  b / a = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_l736_73609


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l736_73660

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

-- Define the scenarios
structure Scenario where
  totalPopulation : ℕ
  sampleSize : ℕ
  hasSubgroups : Bool
  isLargeScale : Bool

-- Define the function to determine the appropriate sampling method
def appropriateSamplingMethod (scenario : Scenario) : SamplingMethod :=
  if scenario.hasSubgroups then
    SamplingMethod.Stratified
  else if scenario.isLargeScale then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom

-- Define the three scenarios
def scenario1 : Scenario := ⟨60, 8, false, false⟩
def scenario2 : Scenario := ⟨0, 0, false, true⟩  -- We don't know exact numbers, but it's large scale
def scenario3 : Scenario := ⟨130, 13, true, false⟩

-- State the theorem
theorem appropriate_sampling_methods :
  (appropriateSamplingMethod scenario1 = SamplingMethod.SimpleRandom) ∧
  (appropriateSamplingMethod scenario2 = SamplingMethod.Systematic) ∧
  (appropriateSamplingMethod scenario3 = SamplingMethod.Stratified) :=
by sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l736_73660


namespace NUMINAMATH_CALUDE_sugar_in_recipe_l736_73686

theorem sugar_in_recipe (sugar_already_in : ℕ) (sugar_to_add : ℕ) : 
  sugar_already_in = 2 → sugar_to_add = 11 → sugar_already_in + sugar_to_add = 13 := by
  sorry

end NUMINAMATH_CALUDE_sugar_in_recipe_l736_73686


namespace NUMINAMATH_CALUDE_min_value_and_range_l736_73635

-- Define the function f(x, y, a) = 2xy - x - y - a(x^2 + y^2)
def f (x y a : ℝ) : ℝ := 2 * x * y - x - y - a * (x^2 + y^2)

theorem min_value_and_range {x y a : ℝ} (hx : x > 0) (hy : y > 0) (hf : f x y a = 0) :
  -- Part 1: When a = 0, minimum value of 2x + 4y and corresponding x, y
  (a = 0 → 2 * x + 4 * y ≥ 3 + 2 * Real.sqrt 2 ∧
    (2 * x + 4 * y = 3 + 2 * Real.sqrt 2 ↔ x = (1 + Real.sqrt 2) / 2 ∧ y = (2 + Real.sqrt 2) / 4)) ∧
  -- Part 2: When a = 1/2, range of x + y
  (a = 1/2 → x + y ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_range_l736_73635


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l736_73640

theorem quadratic_root_implies_coefficient (a : ℝ) : 
  (3 : ℝ)^2 + a * 3 + 9 = 0 → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l736_73640


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l736_73603

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

-- Define the line equation
def line_eq (m : ℝ) (x y : ℝ) : Prop := y = m * x + 4

-- Theorem statement
theorem line_ellipse_intersection_slopes :
  ∀ m : ℝ, (∃ x y : ℝ, ellipse_eq x y ∧ line_eq m x y) →
  m^2 ≥ 0.48 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l736_73603


namespace NUMINAMATH_CALUDE_smallest_integer_solution_system_of_inequalities_solution_l736_73639

-- Part 1
theorem smallest_integer_solution (x : ℤ) :
  (5 * x + 15 > x - 1) ∧ (∀ y : ℤ, y < x → ¬(5 * y + 15 > y - 1)) ↔ x = -3 :=
sorry

-- Part 2
theorem system_of_inequalities_solution (x : ℝ) :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 4 * x) / 3 > x - 1) ↔ -4 < x ∧ x ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_system_of_inequalities_solution_l736_73639


namespace NUMINAMATH_CALUDE_work_completion_time_l736_73610

theorem work_completion_time (a b t : ℝ) (ha : a > 0) (hb : b > 0) (ht : t > 0) :
  1 / a + 1 / b = 1 / t → b = 18 → t = 7.2 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l736_73610


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l736_73653

theorem geometric_sequence_property (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_product : a 1 * a 7 = 36) : 
  a 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l736_73653


namespace NUMINAMATH_CALUDE_chipped_marbles_bag_l736_73671

def marbleBags : List Nat := [15, 20, 22, 31, 33, 37, 40]

def isValidDistribution (jane : List Nat) (george : List Nat) : Prop :=
  jane.length = 4 ∧ 
  george.length = 2 ∧ 
  (jane.sum : ℚ) = 1.5 * george.sum ∧ 
  (jane.sum + george.sum) % 5 = 0

theorem chipped_marbles_bag (h : ∃ (jane george : List Nat),
  (∀ x ∈ jane ++ george, x ∈ marbleBags) ∧
  isValidDistribution jane george) :
  33 ∈ marbleBags \ (jane ++ george) :=
sorry

end NUMINAMATH_CALUDE_chipped_marbles_bag_l736_73671


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l736_73602

theorem sqrt_product_equality : Real.sqrt (49 + 121) * Real.sqrt (64 - 49) = Real.sqrt 2550 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l736_73602


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l736_73611

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_345 : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l736_73611


namespace NUMINAMATH_CALUDE_sixth_root_of_68968845601_l736_73675

theorem sixth_root_of_68968845601 :
  51^6 = 68968845601 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_68968845601_l736_73675


namespace NUMINAMATH_CALUDE_product_sum_theorem_l736_73666

theorem product_sum_theorem (W F : ℕ) (c : ℕ) : 
  W > 20 → F > 20 → W * F = 770 → W + F = c → c = 57 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l736_73666


namespace NUMINAMATH_CALUDE_complex_power_2018_l736_73643

theorem complex_power_2018 : (((1 - Complex.I) / (1 + Complex.I)) ^ 2018 : ℂ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_2018_l736_73643


namespace NUMINAMATH_CALUDE_cruise_liner_passengers_l736_73631

theorem cruise_liner_passengers : ∃ n : ℕ, 
  (250 ≤ n ∧ n ≤ 400) ∧ 
  (∃ r : ℕ, n = 15 * r + 7) ∧
  (∃ s : ℕ, n = 25 * s - 8) ∧
  (n = 292 ∨ n = 367) := by
sorry

end NUMINAMATH_CALUDE_cruise_liner_passengers_l736_73631


namespace NUMINAMATH_CALUDE_factorization_equality_l736_73619

theorem factorization_equality (a m n : ℝ) :
  -3 * a * m^2 + 12 * a * n^2 = -3 * a * (m + 2*n) * (m - 2*n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l736_73619


namespace NUMINAMATH_CALUDE_alcohol_dilution_l736_73668

/-- Proves that adding 16 liters of water to 24 liters of a 90% alcohol solution
    results in a new mixture with 54% alcohol. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 24 →
  initial_concentration = 0.90 →
  added_water = 16 →
  final_concentration = 0.54 →
  initial_volume * initial_concentration = 
    (initial_volume + added_water) * final_concentration :=
by
  sorry

#check alcohol_dilution

end NUMINAMATH_CALUDE_alcohol_dilution_l736_73668


namespace NUMINAMATH_CALUDE_garrison_provisions_l736_73652

theorem garrison_provisions (initial_men : ℕ) (initial_days : ℕ) (reinforcement : ℕ) (days_before_reinforcement : ℕ) : 
  initial_men = 2000 →
  initial_days = 54 →
  reinforcement = 1600 →
  days_before_reinforcement = 18 →
  let total_provisions := initial_men * initial_days
  let used_provisions := initial_men * days_before_reinforcement
  let remaining_provisions := total_provisions - used_provisions
  let total_men_after_reinforcement := initial_men + reinforcement
  (remaining_provisions / total_men_after_reinforcement : ℚ) = 20 := by
sorry

end NUMINAMATH_CALUDE_garrison_provisions_l736_73652


namespace NUMINAMATH_CALUDE_complement_of_intersection_main_theorem_l736_73629

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {2, 3, 4}

-- Theorem statement
theorem complement_of_intersection (x : Nat) : 
  x ∈ (A ∩ B)ᶜ ↔ (x ∈ U ∧ x ∉ (A ∩ B)) :=
by
  sorry

-- Main theorem to prove
theorem main_theorem : (A ∩ B)ᶜ = {1, 4} :=
by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_main_theorem_l736_73629


namespace NUMINAMATH_CALUDE_least_N_for_probability_condition_l736_73690

def P (N : ℕ) : ℚ :=
  (⌊(2 * N : ℚ) / 5⌋ + (N - ⌈(3 * N : ℚ) / 5⌉)) / (N + 1 : ℚ)

theorem least_N_for_probability_condition :
  (∀ k : ℕ, k % 5 = 0 ∧ 0 < k ∧ k < 480 → P k ≥ 321/400) ∧
  P 480 < 321/400 := by
  sorry

end NUMINAMATH_CALUDE_least_N_for_probability_condition_l736_73690


namespace NUMINAMATH_CALUDE_product_sequence_sum_l736_73649

theorem product_sequence_sum (a b : ℕ) : 
  (a : ℚ) / 3 = 16 → b = a - 1 → a + b = 95 := by sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l736_73649


namespace NUMINAMATH_CALUDE_line_intersection_with_y_axis_l736_73682

/-- Given a line passing through points (3, 10) and (-7, -6), 
    prove that its intersection with the y-axis is the point (0, 5.2) -/
theorem line_intersection_with_y_axis :
  let p₁ : ℝ × ℝ := (3, 10)
  let p₂ : ℝ × ℝ := (-7, -6)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  let b : ℝ := p₁.2 - m * p₁.1
  let line (x : ℝ) : ℝ := m * x + b
  let y_intercept : ℝ := line 0
  (0, y_intercept) = (0, 5.2) := by sorry

end NUMINAMATH_CALUDE_line_intersection_with_y_axis_l736_73682


namespace NUMINAMATH_CALUDE_sisters_contribution_l736_73628

/-- The amount of money Miranda's sister gave her to buy heels -/
theorem sisters_contribution (months_saved : ℕ) (monthly_savings : ℕ) (total_cost : ℕ) : 
  months_saved = 3 → monthly_savings = 70 → total_cost = 260 →
  total_cost - (months_saved * monthly_savings) = 50 := by
  sorry

end NUMINAMATH_CALUDE_sisters_contribution_l736_73628


namespace NUMINAMATH_CALUDE_fraction_simplification_l736_73608

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l736_73608


namespace NUMINAMATH_CALUDE_problem_statement_l736_73623

def p (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 2, m ≤ x^2 - 2*x

def q (m : ℝ) : Prop := ∃ x ≥ 0, 2^x + 3 = m

theorem problem_statement :
  (∀ m : ℝ, p m ↔ m ∈ Set.Iic (-1)) ∧
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Iic (-1) ∪ Set.Ici 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l736_73623


namespace NUMINAMATH_CALUDE_original_amount_l736_73697

theorem original_amount (x : ℚ) : x > 0 → (4/9 : ℚ) * x + 80/3 = x ↔ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_original_amount_l736_73697


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l736_73657

theorem solution_implies_a_value (a : ℝ) : (2 * 2 - a = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l736_73657


namespace NUMINAMATH_CALUDE_equation_solution_l736_73650

theorem equation_solution : ∃! x : ℝ, (x + 1)^63 + (x + 1)^62*(x - 1) + (x + 1)^61*(x - 1)^2 + (x - 1)^63 = 0 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l736_73650


namespace NUMINAMATH_CALUDE_pentagon_perimeter_eq_sum_of_coefficients_l736_73633

/-- The perimeter of a pentagon with specified vertices -/
def pentagon_perimeter : ℝ := sorry

/-- Theorem stating that the perimeter of the specified pentagon equals 2 + 2√10 -/
theorem pentagon_perimeter_eq : pentagon_perimeter = 2 + 2 * Real.sqrt 10 := by sorry

/-- Corollary showing that when expressed as p + q√10 + r√13, p + q + r = 4 -/
theorem sum_of_coefficients : ∃ (p q r : ℤ), 
  pentagon_perimeter = ↑p + ↑q * Real.sqrt 10 + ↑r * Real.sqrt 13 ∧ p + q + r = 4 := by sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_eq_sum_of_coefficients_l736_73633


namespace NUMINAMATH_CALUDE_inverse_sum_mod_25_l736_73614

theorem inverse_sum_mod_25 :
  ∃ (a b c : ℤ), (7 * a) % 25 = 1 ∧ 
                 (7 * b) % 25 = a % 25 ∧ 
                 (7 * c) % 25 = b % 25 ∧ 
                 (a + b + c) % 25 = 9 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_25_l736_73614


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l736_73636

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3}

-- Define set A
def A : Set Nat := {0, 1}

-- Define set B
def B : Set Nat := {1, 2, 3}

-- Theorem statement
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l736_73636


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_112_l736_73622

theorem smallest_four_digit_multiple_of_112 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 112 ∣ n → 1008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_112_l736_73622


namespace NUMINAMATH_CALUDE_train_speed_problem_train_speed_solution_l736_73627

theorem train_speed_problem (train1_length train2_length : ℝ) 
  (train1_speed time_to_clear : ℝ) : ℝ :=
  let total_length := train1_length + train2_length
  let total_length_km := total_length / 1000
  let time_to_clear_hours := time_to_clear / 3600
  let relative_speed := total_length_km / time_to_clear_hours
  relative_speed - train1_speed

theorem train_speed_solution :
  train_speed_problem 140 280 42 20.99832013438925 = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_train_speed_solution_l736_73627


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l736_73626

theorem movie_ticket_cost (ticket_price : ℕ) (num_students : ℕ) (budget : ℕ) : 
  ticket_price = 29 → num_students = 498 → budget = 1500 → 
  ticket_price * num_students > budget :=
by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l736_73626


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l736_73669

def old_salary : ℝ := 10000
def new_salary : ℝ := 10200

theorem salary_increase_percentage :
  (new_salary - old_salary) / old_salary * 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l736_73669


namespace NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l736_73667

/-- The area of a circle circumscribing an equilateral triangle with side length 12 units is 48π square units. -/
theorem circle_area_equilateral_triangle : 
  ∀ (s : ℝ) (A : ℝ),
  s = 12 →  -- Side length of the equilateral triangle
  A = π * (s / Real.sqrt 3)^2 →  -- Area of the circle (using the circumradius formula)
  A = 48 * π := by
sorry

end NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l736_73667


namespace NUMINAMATH_CALUDE_tory_has_six_games_l736_73699

/-- The number of video games Theresa, Julia, and Tory have. -/
structure VideoGames where
  theresa : ℕ
  julia : ℕ
  tory : ℕ

/-- The conditions given in the problem. -/
def problem_conditions (vg : VideoGames) : Prop :=
  vg.theresa = 3 * vg.julia + 5 ∧
  vg.julia = vg.tory / 3 ∧
  vg.theresa = 11

/-- The theorem stating that Tory has 6 video games. -/
theorem tory_has_six_games (vg : VideoGames) (h : problem_conditions vg) : vg.tory = 6 := by
  sorry

end NUMINAMATH_CALUDE_tory_has_six_games_l736_73699


namespace NUMINAMATH_CALUDE_sale_price_ratio_l736_73685

theorem sale_price_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_ratio_l736_73685


namespace NUMINAMATH_CALUDE_quadratic_root_proof_l736_73691

theorem quadratic_root_proof : ∃ x : ℝ, x^2 - 4*x*Real.sqrt 2 + 8 = 0 ∧ x = 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_proof_l736_73691


namespace NUMINAMATH_CALUDE_frog_eggs_eaten_percentage_l736_73698

theorem frog_eggs_eaten_percentage (total_eggs : ℕ) (dry_up_rate : ℚ) (hatch_rate : ℚ) (hatched_frogs : ℕ) : 
  total_eggs = 800 →
  dry_up_rate = 1/10 →
  hatch_rate = 1/4 →
  hatched_frogs = 40 →
  (total_eggs - (dry_up_rate * total_eggs).floor - (hatch_rate * (total_eggs - (dry_up_rate * total_eggs).floor)).floor) / total_eggs = 7/10 := by
sorry

end NUMINAMATH_CALUDE_frog_eggs_eaten_percentage_l736_73698


namespace NUMINAMATH_CALUDE_tangent_intersection_y_coord_l736_73646

/-- Given two points on the parabola y = x^2 + 1 with perpendicular tangents,
    the y-coordinate of their intersection is 3/4 -/
theorem tangent_intersection_y_coord (a b : ℝ) : 
  (2 * a) * (2 * b) = -1 →  -- Perpendicular tangents condition
  (∃ (x : ℝ), (2 * a) * (x - a) + a^2 + 1 = (2 * b) * (x - b) + b^2 + 1) →  -- Intersection exists
  (2 * a) * ((a + b) / 2 - a) + a^2 + 1 = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_intersection_y_coord_l736_73646


namespace NUMINAMATH_CALUDE_waiter_tables_l736_73655

/-- Given a waiter's customer and table information, prove the number of tables. -/
theorem waiter_tables
  (initial_customers : ℕ)
  (departed_customers : ℕ)
  (people_per_table : ℕ)
  (h1 : initial_customers = 44)
  (h2 : departed_customers = 12)
  (h3 : people_per_table = 8)
  : (initial_customers - departed_customers) / people_per_table = 4 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_l736_73655


namespace NUMINAMATH_CALUDE_at_least_one_prime_between_nfact_minus_n_and_nfact_l736_73651

theorem at_least_one_prime_between_nfact_minus_n_and_nfact (n : ℕ) (h : n > 2) :
  ∃ p : ℕ, Prime p ∧ n! - n < p ∧ p < n! :=
sorry

end NUMINAMATH_CALUDE_at_least_one_prime_between_nfact_minus_n_and_nfact_l736_73651


namespace NUMINAMATH_CALUDE_jerry_age_l736_73644

theorem jerry_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 16 → 
  mickey_age = 2 * jerry_age - 6 → 
  jerry_age = 11 := by
sorry

end NUMINAMATH_CALUDE_jerry_age_l736_73644
