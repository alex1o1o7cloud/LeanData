import Mathlib

namespace NUMINAMATH_CALUDE_total_fans_l3509_350997

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- Defines the conditions of the problem -/
def fan_conditions (f : FanCounts) : Prop :=
  f.yankees * 2 = f.mets * 3 ∧  -- Ratio of Yankees to Mets fans is 3:2
  f.mets * 5 = f.red_sox * 4 ∧  -- Ratio of Mets to Red Sox fans is 4:5
  f.mets = 88                   -- There are 88 Mets fans

/-- The theorem to be proved -/
theorem total_fans (f : FanCounts) (h : fan_conditions f) : 
  f.yankees + f.mets + f.red_sox = 330 := by
  sorry

#check total_fans

end NUMINAMATH_CALUDE_total_fans_l3509_350997


namespace NUMINAMATH_CALUDE_total_problems_practiced_l3509_350924

def marvin_yesterday : ℕ := 40
def marvin_today : ℕ := 3 * marvin_yesterday
def arvin_yesterday : ℕ := 2 * marvin_yesterday
def arvin_today : ℕ := 2 * marvin_today
def kevin_yesterday : ℕ := 30
def kevin_today : ℕ := kevin_yesterday ^ 2

theorem total_problems_practiced :
  marvin_yesterday + marvin_today + arvin_yesterday + arvin_today + kevin_yesterday + kevin_today = 1410 :=
by sorry

end NUMINAMATH_CALUDE_total_problems_practiced_l3509_350924


namespace NUMINAMATH_CALUDE_base8_perfect_square_b_zero_l3509_350914

/-- Represents a number in base 8 of the form a1b4 -/
structure Base8Number where
  a : ℕ
  b : ℕ
  h_a_nonzero : a ≠ 0

/-- Converts a Base8Number to its decimal representation -/
def toDecimal (n : Base8Number) : ℕ :=
  512 * n.a + 64 + 8 * n.b + 4

/-- Predicate to check if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem base8_perfect_square_b_zero (n : Base8Number) :
  isPerfectSquare (toDecimal n) → n.b = 0 := by
  sorry

end NUMINAMATH_CALUDE_base8_perfect_square_b_zero_l3509_350914


namespace NUMINAMATH_CALUDE_larger_number_four_times_smaller_l3509_350991

theorem larger_number_four_times_smaller
  (a b : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_distinct : a ≠ b)
  (h_equation : a^3 - b^3 = 3*(2*a^2*b - 3*a*b^2 + b^3)) :
  a = 4*b :=
sorry

end NUMINAMATH_CALUDE_larger_number_four_times_smaller_l3509_350991


namespace NUMINAMATH_CALUDE_supermarket_spending_l3509_350982

theorem supermarket_spending (F : ℚ) : 
  F + (1 : ℚ)/3 + (1 : ℚ)/10 + 8/120 = 1 → F = (1 : ℚ)/2 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_l3509_350982


namespace NUMINAMATH_CALUDE_smallest_integer_lower_bound_l3509_350961

theorem smallest_integer_lower_bound 
  (a b c d : ℤ) 
  (different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (average : (a + b + c + d) / 4 = 76) 
  (largest : d = 90) 
  (ordered : a ≤ b ∧ b ≤ c ∧ c ≤ d) : 
  a ≥ 37 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_lower_bound_l3509_350961


namespace NUMINAMATH_CALUDE_aron_vacuum_time_l3509_350965

/-- Represents the cleaning schedule and total cleaning time for Aron. -/
structure CleaningSchedule where
  vacuum_frequency : Nat  -- Number of days Aron vacuums per week
  dust_time : Nat         -- Minutes Aron spends dusting per day
  dust_frequency : Nat    -- Number of days Aron dusts per week
  total_cleaning_time : Nat  -- Total minutes Aron spends cleaning per week

/-- Calculates the number of minutes Aron spends vacuuming each day. -/
def vacuum_time_per_day (schedule : CleaningSchedule) : Nat :=
  (schedule.total_cleaning_time - schedule.dust_time * schedule.dust_frequency) / schedule.vacuum_frequency

/-- Theorem stating that Aron spends 30 minutes vacuuming each day. -/
theorem aron_vacuum_time (schedule : CleaningSchedule) 
    (h1 : schedule.vacuum_frequency = 3)
    (h2 : schedule.dust_time = 20)
    (h3 : schedule.dust_frequency = 2)
    (h4 : schedule.total_cleaning_time = 130) :
    vacuum_time_per_day schedule = 30 := by
  sorry


end NUMINAMATH_CALUDE_aron_vacuum_time_l3509_350965


namespace NUMINAMATH_CALUDE_cubic_inequality_l3509_350906

theorem cubic_inequality (x : ℝ) : x^3 - 16*x^2 + 73*x > 84 ↔ x > 13 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3509_350906


namespace NUMINAMATH_CALUDE_coefficient_x6_in_x_plus_2_to_8_l3509_350964

theorem coefficient_x6_in_x_plus_2_to_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k * 2^(8 - k)) * (if k = 6 then 1 else 0)) = 112 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x6_in_x_plus_2_to_8_l3509_350964


namespace NUMINAMATH_CALUDE_uki_earnings_l3509_350989

/-- Represents Uki's bakery business -/
structure BakeryBusiness where
  cupcake_price : ℝ
  cookie_price : ℝ
  biscuit_price : ℝ
  daily_cupcakes : ℕ
  daily_cookies : ℕ
  daily_biscuits : ℕ

/-- Calculates the total earnings for a given number of days -/
def total_earnings (b : BakeryBusiness) (days : ℕ) : ℝ :=
  (b.cupcake_price * b.daily_cupcakes + 
   b.cookie_price * b.daily_cookies + 
   b.biscuit_price * b.daily_biscuits) * days

/-- Theorem stating that Uki's total earnings for five days is $350 -/
theorem uki_earnings : ∃ (b : BakeryBusiness), 
  b.cupcake_price = 1.5 ∧ 
  b.cookie_price = 2 ∧ 
  b.biscuit_price = 1 ∧ 
  b.daily_cupcakes = 20 ∧ 
  b.daily_cookies = 10 ∧ 
  b.daily_biscuits = 20 ∧ 
  total_earnings b 5 = 350 := by
  sorry

end NUMINAMATH_CALUDE_uki_earnings_l3509_350989


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_points_product_l3509_350910

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    prove that the product of the distances from the origin to any two 
    perpendicular points on the ellipse is at least 2a²b²/(a² + b²) -/
theorem ellipse_perpendicular_points_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∀ (P Q : ℝ × ℝ), 
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) →
    (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) →
    (P.1 * Q.1 + P.2 * Q.2 = 0) →
    (P.1^2 + P.2^2) * (Q.1^2 + Q.2^2) ≥ (2 * a^2 * b^2) / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_points_product_l3509_350910


namespace NUMINAMATH_CALUDE_percentage_of_B_grades_l3509_350990

def scores : List ℕ := [88, 73, 55, 95, 76, 91, 86, 73, 76, 64, 85, 79, 72, 81, 89, 77]

def is_B_grade (score : ℕ) : Bool :=
  87 ≤ score ∧ score ≤ 94

def count_B_grades (scores : List ℕ) : ℕ :=
  scores.filter is_B_grade |>.length

theorem percentage_of_B_grades :
  (count_B_grades scores : ℚ) / scores.length * 100 = 12.5 := by sorry

end NUMINAMATH_CALUDE_percentage_of_B_grades_l3509_350990


namespace NUMINAMATH_CALUDE_smallest_N_satisfying_condition_l3509_350986

def P (N : ℕ) : ℚ := (4 * N + 2) / (5 * N + 1)

theorem smallest_N_satisfying_condition :
  ∃ (N : ℕ), N > 0 ∧ N % 5 = 0 ∧ P N < 321 / 400 ∧
  ∀ (M : ℕ), M > 0 → M % 5 = 0 → P M < 321 / 400 → N ≤ M ∧
  N = 480 := by
  sorry

end NUMINAMATH_CALUDE_smallest_N_satisfying_condition_l3509_350986


namespace NUMINAMATH_CALUDE_scientific_notation_of_448000_l3509_350984

/-- Proves that 448,000 is equal to 4.48 × 10^5 in scientific notation -/
theorem scientific_notation_of_448000 : 
  ∃ (a : ℝ) (n : ℤ), 448000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.48 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_448000_l3509_350984


namespace NUMINAMATH_CALUDE_place_value_comparison_l3509_350918

theorem place_value_comparison (n : Real) (h : n = 85376.4201) : 
  (10 : Real) / (1 / 10 : Real) = 100 := by
  sorry

end NUMINAMATH_CALUDE_place_value_comparison_l3509_350918


namespace NUMINAMATH_CALUDE_sum_representation_l3509_350970

def sum_of_complex_exponentials (z₁ z₂ : ℂ) : ℂ := z₁ + z₂

theorem sum_representation (z₁ z₂ : ℂ) :
  let sum := sum_of_complex_exponentials z₁ z₂
  let r := 30 * Real.cos (π / 10)
  let θ := 9 * π / 20
  z₁ = 15 * Complex.exp (Complex.I * π / 5) ∧
  z₂ = 15 * Complex.exp (Complex.I * 7 * π / 10) →
  sum = r * Complex.exp (Complex.I * θ) :=
by sorry

end NUMINAMATH_CALUDE_sum_representation_l3509_350970


namespace NUMINAMATH_CALUDE_curve_length_of_right_square_prism_l3509_350916

/-- Represents a right square prism -/
structure RightSquarePrism where
  sideEdge : ℝ
  baseEdge : ℝ

/-- Calculates the total length of curves on the surface of a right square prism
    formed by points at a given distance from a vertex -/
def totalCurveLength (prism : RightSquarePrism) (distance : ℝ) : ℝ :=
  sorry

/-- The theorem statement -/
theorem curve_length_of_right_square_prism :
  let prism : RightSquarePrism := ⟨4, 4⟩
  totalCurveLength prism 3 = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_curve_length_of_right_square_prism_l3509_350916


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3509_350923

/-- A square inscribed in a right triangle with one vertex at the right angle -/
def square_in_triangle_vertex (a b c : ℝ) (x : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ (c - x) / x = a / b

/-- A square inscribed in a right triangle with one side on the hypotenuse -/
def square_in_triangle_hypotenuse (a b c : ℝ) (y : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ (a - y) / y = (b - y) / y

theorem inscribed_squares_ratio :
  ∀ x y : ℝ,
    square_in_triangle_vertex 5 12 13 x →
    square_in_triangle_hypotenuse 6 8 10 y →
    x / y = 37 / 35 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3509_350923


namespace NUMINAMATH_CALUDE_exists_same_answer_question_l3509_350904

/-- A person who either always tells the truth or always lies -/
inductive Person
| TruthTeller
| Liar

/-- The answer a person gives to a question -/
inductive Answer
| Yes
| No

/-- A question that can be asked to a person -/
def Question := Person → Answer

/-- The actual answer to a question about a person -/
def actualAnswer (p : Person) (q : Question) : Answer :=
  match p with
  | Person.TruthTeller => q Person.TruthTeller
  | Person.Liar => match q Person.Liar with
    | Answer.Yes => Answer.No
    | Answer.No => Answer.Yes

/-- There exists a question that makes both a truth-teller and a liar give the same answer -/
theorem exists_same_answer_question : ∃ (q : Question),
  actualAnswer Person.TruthTeller q = actualAnswer Person.Liar q :=
sorry

end NUMINAMATH_CALUDE_exists_same_answer_question_l3509_350904


namespace NUMINAMATH_CALUDE_yellow_block_weight_l3509_350919

theorem yellow_block_weight (green_weight : ℝ) (weight_difference : ℝ) 
  (h1 : green_weight = 0.4)
  (h2 : weight_difference = 0.2) :
  green_weight + weight_difference = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_yellow_block_weight_l3509_350919


namespace NUMINAMATH_CALUDE_smallest_4_9_divisible_by_4_and_9_l3509_350948

def is_composed_of_4_and_9 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 9

theorem smallest_4_9_divisible_by_4_and_9 :
  ∃! n : ℕ,
    n > 0 ∧
    n % 4 = 0 ∧
    n % 9 = 0 ∧
    is_composed_of_4_and_9 n ∧
    ∀ m : ℕ, m > 0 ∧ m % 4 = 0 ∧ m % 9 = 0 ∧ is_composed_of_4_and_9 m → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_4_9_divisible_by_4_and_9_l3509_350948


namespace NUMINAMATH_CALUDE_perspective_square_area_l3509_350980

/-- A square whose perspective drawing is a parallelogram with one side of length 4 -/
structure PerspectiveSquare where
  /-- The side length of the original square -/
  side : ℝ
  /-- The side length of the parallelogram in the perspective drawing -/
  perspective_side : ℝ
  /-- The perspective drawing is a parallelogram -/
  is_parallelogram : Bool
  /-- One side of the parallelogram has length 4 -/
  perspective_side_eq_four : perspective_side = 4

/-- The possible areas of the original square -/
def possible_areas (s : PerspectiveSquare) : Set ℝ :=
  {16, 64}

/-- Theorem: The area of the original square is either 16 or 64 -/
theorem perspective_square_area (s : PerspectiveSquare) :
  (s.side ^ 2) ∈ possible_areas s :=
sorry

end NUMINAMATH_CALUDE_perspective_square_area_l3509_350980


namespace NUMINAMATH_CALUDE_not_complete_residue_sum_l3509_350903

/-- For an even number n, if a and b are complete residue systems modulo n,
    then their pairwise sum is not a complete residue system modulo n. -/
theorem not_complete_residue_sum
  (n : ℕ) (hn : Even n) 
  (a b : Fin n → ℕ)
  (ha : Function.Surjective (λ i => a i % n))
  (hb : Function.Surjective (λ i => b i % n)) :
  ¬ Function.Surjective (λ i => (a i + b i) % n) :=
sorry

end NUMINAMATH_CALUDE_not_complete_residue_sum_l3509_350903


namespace NUMINAMATH_CALUDE_cat_food_sale_revenue_l3509_350933

/-- Calculates the total revenue from cat food sales during a promotion --/
theorem cat_food_sale_revenue : 
  let original_price : ℚ := 25
  let first_group_size : ℕ := 8
  let first_group_cases : ℕ := 3
  let first_group_discount : ℚ := 15/100
  let second_group_size : ℕ := 4
  let second_group_cases : ℕ := 2
  let second_group_discount : ℚ := 10/100
  let third_group_size : ℕ := 8
  let third_group_cases : ℕ := 1
  let third_group_discount : ℚ := 0

  let first_group_revenue := (first_group_size * first_group_cases : ℚ) * 
    (original_price * (1 - first_group_discount))
  let second_group_revenue := (second_group_size * second_group_cases : ℚ) * 
    (original_price * (1 - second_group_discount))
  let third_group_revenue := (third_group_size * third_group_cases : ℚ) * 
    (original_price * (1 - third_group_discount))

  let total_revenue := first_group_revenue + second_group_revenue + third_group_revenue

  total_revenue = 890 := by
  sorry

end NUMINAMATH_CALUDE_cat_food_sale_revenue_l3509_350933


namespace NUMINAMATH_CALUDE_triangle_min_value_l3509_350998

theorem triangle_min_value (a b c : ℝ) (A : ℝ) (area : ℝ) : 
  A = Real.pi / 3 →
  area = Real.sqrt 3 →
  area = (1 / 2) * b * c * Real.sin A →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  (∀ x y : ℝ, (4 * x ^ 2 + 4 * y ^ 2 - 3 * a ^ 2) / (x + y) ≥ 5) ∧
  (∃ x y : ℝ, (4 * x ^ 2 + 4 * y ^ 2 - 3 * a ^ 2) / (x + y) = 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_min_value_l3509_350998


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l3509_350942

def numbers : List ℤ := [1871, 2011, 2059, 2084, 2113, 2167, 2198, 2210]

theorem mean_of_remaining_numbers :
  ∀ (subset : List ℤ),
    subset ⊆ numbers →
    subset.length = 6 →
    (subset.sum : ℚ) / 6 = 2100 →
    let remaining := numbers.filter (λ x => x ∉ subset)
    (remaining.sum : ℚ) / 2 = 2056.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l3509_350942


namespace NUMINAMATH_CALUDE_point_on_line_l3509_350905

/-- The value of m for which the point (m + 1, 3) lies on the line x + y + 1 = 0 -/
theorem point_on_line (m : ℝ) : (m + 1) + 3 + 1 = 0 ↔ m = -5 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l3509_350905


namespace NUMINAMATH_CALUDE_compare_roots_l3509_350935

theorem compare_roots : 3^(1/3) > 2^(1/2) ∧ 2^(1/2) > 8^(1/8) ∧ 8^(1/8) > 9^(1/9) := by
  sorry

end NUMINAMATH_CALUDE_compare_roots_l3509_350935


namespace NUMINAMATH_CALUDE_expression_evaluation_l3509_350947

theorem expression_evaluation :
  let x := Real.sqrt 2 * Real.sin (π / 4) + Real.tan (π / 3)
  (x / (x^2 - 1)) / (1 - 1 / (x + 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3509_350947


namespace NUMINAMATH_CALUDE_special_triangle_area_l3509_350920

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- The angle between the two longest sides (in degrees)
  x : ℝ
  -- The perimeter of the triangle (in cm)
  perimeter : ℝ
  -- The inradius of the triangle (in cm)
  inradius : ℝ
  -- Constraint on the angle x
  angle_constraint : 60 < x ∧ x < 120
  -- Given perimeter value
  perimeter_value : perimeter = 48
  -- Given inradius value
  inradius_value : inradius = 2.5

/-- The area of a triangle given its perimeter and inradius -/
def triangleArea (t : SpecialTriangle) : ℝ := t.perimeter * t.inradius

/-- Theorem stating that the area of the special triangle is 120 cm² -/
theorem special_triangle_area (t : SpecialTriangle) : triangleArea t = 120 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_area_l3509_350920


namespace NUMINAMATH_CALUDE_not_always_possible_to_make_all_white_l3509_350978

/-- Represents a smaller equilateral triangle within the larger triangle -/
structure SmallTriangle where
  color : Bool  -- true for white, false for black

/-- Represents the entire configuration of the divided equilateral triangle -/
structure TriangleConfiguration where
  smallTriangles : List SmallTriangle
  numRows : Nat  -- number of rows in the triangle

/-- Represents a repainting operation -/
def repaint (config : TriangleConfiguration) (lineIndex : Nat) : TriangleConfiguration :=
  sorry

/-- Checks if all small triangles in the configuration are white -/
def allWhite (config : TriangleConfiguration) : Bool :=
  sorry

/-- Theorem stating that there exists a configuration where it's impossible to make all triangles white -/
theorem not_always_possible_to_make_all_white :
  ∃ (initialConfig : TriangleConfiguration),
    ∀ (repaintSequence : List Nat),
      let finalConfig := repaintSequence.foldl repaint initialConfig
      ¬(allWhite finalConfig) :=
sorry

end NUMINAMATH_CALUDE_not_always_possible_to_make_all_white_l3509_350978


namespace NUMINAMATH_CALUDE_product_expansion_l3509_350959

theorem product_expansion (x : ℝ) : (2*x + 3) * (3*x^2 + 4*x + 1) = 6*x^3 + 17*x^2 + 14*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3509_350959


namespace NUMINAMATH_CALUDE_emily_walk_distance_l3509_350936

/-- Calculates the total distance walked given the number of blocks and block length -/
def total_distance (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Proves that walking 8 blocks west and 10 blocks south, with each block being 1/4 mile, results in a total distance of 4.5 miles -/
theorem emily_walk_distance : total_distance 8 10 (1/4) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_emily_walk_distance_l3509_350936


namespace NUMINAMATH_CALUDE_specific_sculpture_surface_area_l3509_350960

/-- Represents a cube sculpture with a 3x3 bottom layer and a cross-shaped top layer --/
structure CubeSculpture where
  cubeEdgeLength : ℝ
  bottomLayerSize : ℕ
  topLayerSize : ℕ

/-- Calculates the exposed surface area of the cube sculpture --/
def exposedSurfaceArea (sculpture : CubeSculpture) : ℝ :=
  sorry

/-- Theorem stating that the exposed surface area of the specific sculpture is 46 square meters --/
theorem specific_sculpture_surface_area :
  let sculpture : CubeSculpture := {
    cubeEdgeLength := 1,
    bottomLayerSize := 3,
    topLayerSize := 5
  }
  exposedSurfaceArea sculpture = 46 := by sorry

end NUMINAMATH_CALUDE_specific_sculpture_surface_area_l3509_350960


namespace NUMINAMATH_CALUDE_ducks_in_marsh_l3509_350908

/-- The number of ducks in a marsh, given the total number of birds and the number of geese. -/
def number_of_ducks (total_birds geese : ℕ) : ℕ := total_birds - geese

/-- Theorem stating that there are 37 ducks in the marsh. -/
theorem ducks_in_marsh : number_of_ducks 95 58 = 37 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_marsh_l3509_350908


namespace NUMINAMATH_CALUDE_sum_of_456_l3509_350949

/-- A geometric sequence with first term 3 and sum of first three terms 9 -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 2) * a n = (a (n + 1))^2
  first_term : a 1 = 3
  sum_first_three : a 1 + a 2 + a 3 = 9

/-- The sum of the 4th, 5th, and 6th terms is either 9 or -72 -/
theorem sum_of_456 (seq : GeometricSequence) :
  seq.a 4 + seq.a 5 + seq.a 6 = 9 ∨ seq.a 4 + seq.a 5 + seq.a 6 = -72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_456_l3509_350949


namespace NUMINAMATH_CALUDE_brick_length_proof_l3509_350928

/-- The length of a brick in centimeters. -/
def brick_length : ℝ := 25

/-- The width of a brick in centimeters. -/
def brick_width : ℝ := 11.25

/-- The height of a brick in centimeters. -/
def brick_height : ℝ := 6

/-- The length of the wall in centimeters. -/
def wall_length : ℝ := 800

/-- The width of the wall in centimeters. -/
def wall_width : ℝ := 600

/-- The height of the wall in centimeters. -/
def wall_height : ℝ := 22.5

/-- The number of bricks needed to build the wall. -/
def num_bricks : ℕ := 6400

/-- The volume of the wall in cubic centimeters. -/
def wall_volume : ℝ := wall_length * wall_width * wall_height

/-- The volume of a single brick in cubic centimeters. -/
def brick_volume : ℝ := brick_length * brick_width * brick_height

theorem brick_length_proof : 
  brick_length * brick_width * brick_height * num_bricks = wall_volume :=
by sorry

end NUMINAMATH_CALUDE_brick_length_proof_l3509_350928


namespace NUMINAMATH_CALUDE_lowest_number_of_students_twenty_four_is_lowest_l3509_350943

theorem lowest_number_of_students (n : ℕ) : n > 0 ∧ 12 ∣ n ∧ 24 ∣ n → n ≥ 24 := by
  sorry

theorem twenty_four_is_lowest : ∃ (n : ℕ), n > 0 ∧ 12 ∣ n ∧ 24 ∣ n ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_lowest_number_of_students_twenty_four_is_lowest_l3509_350943


namespace NUMINAMATH_CALUDE_no_perfect_square_with_only_six_and_zero_l3509_350999

theorem no_perfect_square_with_only_six_and_zero : 
  ¬ ∃ (n : ℕ), (∃ (m : ℕ), n = m^2) ∧ 
  (∀ (d : ℕ), d ∈ n.digits 10 → d = 6 ∨ d = 0) :=
sorry

end NUMINAMATH_CALUDE_no_perfect_square_with_only_six_and_zero_l3509_350999


namespace NUMINAMATH_CALUDE_integer_root_count_theorem_l3509_350926

/-- A polynomial of degree 5 with integer coefficients -/
structure IntPolynomial5 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- The number of integer roots of a polynomial, counting multiplicity -/
def numIntegerRoots (p : IntPolynomial5) : ℕ := sorry

/-- The set of possible values for the number of integer roots -/
def possibleRootCounts : Set ℕ := {0, 1, 2, 3, 5}

/-- Theorem: The number of integer roots of a degree 5 polynomial 
    with integer coefficients is always in the set {0, 1, 2, 3, 5} -/
theorem integer_root_count_theorem (p : IntPolynomial5) : 
  numIntegerRoots p ∈ possibleRootCounts := by sorry

end NUMINAMATH_CALUDE_integer_root_count_theorem_l3509_350926


namespace NUMINAMATH_CALUDE_largest_power_of_18_dividing_30_factorial_l3509_350977

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_power_of_18_dividing_30_factorial :
  (∃ n : ℕ, 18^n ∣ factorial 30 ∧ ∀ m : ℕ, m > n → ¬(18^m ∣ factorial 30)) →
  (∃ n : ℕ, n = 7 ∧ 18^n ∣ factorial 30 ∧ ∀ m : ℕ, m > n → ¬(18^m ∣ factorial 30)) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_18_dividing_30_factorial_l3509_350977


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_45_l3509_350922

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem no_primes_divisible_by_45 : ¬∃ p : ℕ, is_prime p ∧ 45 ∣ p :=
sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_45_l3509_350922


namespace NUMINAMATH_CALUDE_smallest_q_is_6_l3509_350911

/-- Three consecutive terms of an arithmetic sequence -/
structure ArithmeticTriple where
  p : ℝ
  q : ℝ
  r : ℝ
  positive : 0 < p ∧ 0 < q ∧ 0 < r
  consecutive : ∃ d : ℝ, p + d = q ∧ q + d = r

/-- The product of the three terms equals 216 -/
def productIs216 (t : ArithmeticTriple) : Prop :=
  t.p * t.q * t.r = 216

theorem smallest_q_is_6 (t : ArithmeticTriple) (h : productIs216 t) :
    t.q ≥ 6 ∧ ∃ t' : ArithmeticTriple, productIs216 t' ∧ t'.q = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_q_is_6_l3509_350911


namespace NUMINAMATH_CALUDE_balls_in_bins_probability_ratio_l3509_350909

def number_of_balls : ℕ := 20
def number_of_bins : ℕ := 5

def p' : ℚ := (number_of_bins * (number_of_bins - 1) * (Nat.factorial 11) / 
  (Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 3)) / 
  (Nat.choose (number_of_balls + number_of_bins - 1) (number_of_bins - 1))

def q : ℚ := (Nat.factorial number_of_balls) / 
  ((Nat.factorial 4)^number_of_bins * Nat.factorial number_of_bins) / 
  (Nat.choose (number_of_balls + number_of_bins - 1) (number_of_bins - 1))

theorem balls_in_bins_probability_ratio : 
  p' / q = 8 / 57 := by sorry

end NUMINAMATH_CALUDE_balls_in_bins_probability_ratio_l3509_350909


namespace NUMINAMATH_CALUDE_james_writing_hours_l3509_350975

/-- Calculates the number of hours James spends writing per week -/
def writing_hours_per_week (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ) : ℕ :=
  (pages_per_person_per_day * num_people * days_per_week) / pages_per_hour

theorem james_writing_hours (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ)
  (h1 : pages_per_hour = 10)
  (h2 : pages_per_person_per_day = 5)
  (h3 : num_people = 2)
  (h4 : days_per_week = 7) :
  writing_hours_per_week pages_per_hour pages_per_person_per_day num_people days_per_week = 7 := by
  sorry

end NUMINAMATH_CALUDE_james_writing_hours_l3509_350975


namespace NUMINAMATH_CALUDE_arthur_sword_problem_l3509_350900

theorem arthur_sword_problem (A B : ℕ) : 
  5 * A + 7 * B = 49 → A - B = 5 := by
sorry

end NUMINAMATH_CALUDE_arthur_sword_problem_l3509_350900


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l3509_350940

theorem least_positive_integer_to_multiple_of_five : ∃ (n : ℕ), n > 0 ∧ (567 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (567 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l3509_350940


namespace NUMINAMATH_CALUDE_area_bounded_by_curve_l3509_350937

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.sqrt (4 - x^2)

theorem area_bounded_by_curve : ∫ x in (0)..(2), f x = π := by sorry

end NUMINAMATH_CALUDE_area_bounded_by_curve_l3509_350937


namespace NUMINAMATH_CALUDE_side_b_value_l3509_350954

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Add conditions to ensure it's a valid triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

-- State the theorem
theorem side_b_value (a b c : ℝ) (A B C : ℝ) :
  triangle_ABC a b c A B C →
  c = Real.sqrt 3 →
  B = Real.pi / 4 →
  C = Real.pi / 3 →
  b = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_side_b_value_l3509_350954


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l3509_350945

theorem square_minus_product_plus_square : 5^2 - 3*4 + 3^2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l3509_350945


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l3509_350934

theorem mod_equivalence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -1774 [ZMOD 7] ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l3509_350934


namespace NUMINAMATH_CALUDE_l_shape_area_is_58_l3509_350925

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the "L" shaped figure -/
structure LShape where
  outerRectangle : Rectangle
  innerRectangle : Rectangle

/-- Calculates the area of the "L" shaped figure -/
def lShapeArea (l : LShape) : ℝ :=
  rectangleArea l.outerRectangle - rectangleArea l.innerRectangle

/-- Theorem: The area of the "L" shaped figure is 58 square units -/
theorem l_shape_area_is_58 :
  let outer := Rectangle.mk 10 7
  let inner := Rectangle.mk 4 3
  let l := LShape.mk outer inner
  lShapeArea l = 58 := by sorry

end NUMINAMATH_CALUDE_l_shape_area_is_58_l3509_350925


namespace NUMINAMATH_CALUDE_coin_move_termination_uniqueness_l3509_350913

-- Define the coin configuration as a function from integers to natural numbers
def CoinConfiguration := ℤ → ℕ

-- Define a legal move
def is_legal_move (c₁ c₂ : CoinConfiguration) : Prop :=
  ∃ i : ℤ, c₁ i ≥ 2 ∧
    c₂ i = c₁ i - 2 ∧
    c₂ (i - 1) = c₁ (i - 1) + 1 ∧
    c₂ (i + 1) = c₁ (i + 1) + 1 ∧
    ∀ j : ℤ, j ≠ i ∧ j ≠ (i - 1) ∧ j ≠ (i + 1) → c₂ j = c₁ j

-- Define a legal sequence of moves
def legal_sequence (c₀ : CoinConfiguration) (n : ℕ) (c : ℕ → CoinConfiguration) : Prop :=
  c 0 = c₀ ∧
  ∀ i : ℕ, i < n → is_legal_move (c i) (c (i + 1))

-- Define a terminal configuration
def is_terminal (c : CoinConfiguration) : Prop :=
  ∀ i : ℤ, c i ≤ 1

-- The main theorem
theorem coin_move_termination_uniqueness
  (c₀ : CoinConfiguration)
  (n₁ n₂ : ℕ)
  (c₁ : ℕ → CoinConfiguration)
  (c₂ : ℕ → CoinConfiguration)
  (h₁ : legal_sequence c₀ n₁ c₁)
  (h₂ : legal_sequence c₀ n₂ c₂)
  (t₁ : is_terminal (c₁ n₁))
  (t₂ : is_terminal (c₂ n₂)) :
  n₁ = n₂ ∧ c₁ n₁ = c₂ n₂ :=
sorry

end NUMINAMATH_CALUDE_coin_move_termination_uniqueness_l3509_350913


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3509_350901

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + 4 * r^2 + 5 * r - 3) - (r^3 + 5 * r^2 + 9 * r - 6) = r^3 - r^2 - 4 * r + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3509_350901


namespace NUMINAMATH_CALUDE_elf_circle_arrangement_exists_l3509_350952

/-- Represents the height of an elf -/
inductive ElfHeight
| Short
| Tall

/-- Represents an elf in the circle -/
structure Elf :=
  (position : Nat)
  (height : ElfHeight)

/-- Checks if an elf is taller than both neighbors -/
def isTallerThanNeighbors (elves : List Elf) (position : Nat) : Bool :=
  sorry

/-- Checks if an elf is shorter than both neighbors -/
def isShorterThanNeighbors (elves : List Elf) (position : Nat) : Bool :=
  sorry

/-- Checks if all elves in the circle satisfy the eye-closing condition -/
def allElvesSatisfyCondition (elves : List Elf) : Bool :=
  sorry

/-- Theorem: There exists an arrangement of 100 elves that satisfies all conditions -/
theorem elf_circle_arrangement_exists : 
  ∃ (elves : List Elf), 
    elves.length = 100 ∧ 
    (∀ e ∈ elves, e.position ≤ 100) ∧
    allElvesSatisfyCondition elves :=
  sorry

end NUMINAMATH_CALUDE_elf_circle_arrangement_exists_l3509_350952


namespace NUMINAMATH_CALUDE_number_reduced_by_six_times_l3509_350929

/-- 
Given a natural number N that does not end in zero, and a digit a (1 ≤ a ≤ 9) in N,
if replacing a with 0 reduces N by 6 times, then N = 12a.
-/
theorem number_reduced_by_six_times (N : ℕ) (a : ℕ) : 
  (1 ≤ a ∧ a ≤ 9) →  -- a is a single digit
  (∃ k : ℕ, N = 12 * 10^k + 2 * a * 10^k) →  -- N has the form 12a in base 10
  (N % 10 ≠ 0) →  -- N does not end in zero
  (∃ N' : ℕ, N' = N / 10^k ∧ N = 6 * (N' - a * 10^k + 0)) →  -- replacing a with 0 reduces N by 6 times
  N = 12 * a := by
sorry

end NUMINAMATH_CALUDE_number_reduced_by_six_times_l3509_350929


namespace NUMINAMATH_CALUDE_monomial_exponents_sum_l3509_350966

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (a b : ℕ) (c d : ℤ) : Prop :=
  a = 3 ∧ b = 1 ∧ c = 3 ∧ d = 1

theorem monomial_exponents_sum (m n : ℕ) :
  like_terms m 1 3 n → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponents_sum_l3509_350966


namespace NUMINAMATH_CALUDE_x_equals_nine_l3509_350953

theorem x_equals_nine (u : ℤ) (x : ℚ) 
  (h1 : u = -6) 
  (h2 : x = (1 : ℚ) / 3 * (3 - 4 * u)) : 
  x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_nine_l3509_350953


namespace NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l3509_350917

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ
  pentagonal_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - (2 * Q.quadrilateral_faces + 5 * Q.pentagonal_faces)

/-- Theorem: A specific convex polyhedron Q has 323 space diagonals -/
theorem specific_polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 30,
    quadrilateral_faces := 10,
    pentagonal_faces := 4
  }
  space_diagonals Q = 323 := by
  sorry


end NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l3509_350917


namespace NUMINAMATH_CALUDE_intersection_points_l3509_350979

/-- The number of intersection points for k lines in a plane -/
def f (k : ℕ) : ℕ := sorry

/-- No two lines are parallel and no three lines intersect at the same point -/
axiom line_properties (k : ℕ) : True

theorem intersection_points (k : ℕ) : f (k + 1) = f k + k :=
  sorry

end NUMINAMATH_CALUDE_intersection_points_l3509_350979


namespace NUMINAMATH_CALUDE_evelyn_marbles_count_l3509_350972

def initial_marbles : ℕ := 95
def marbles_from_henry : ℕ := 9
def cards_bought : ℕ := 6

theorem evelyn_marbles_count :
  initial_marbles + marbles_from_henry = 104 :=
by sorry

end NUMINAMATH_CALUDE_evelyn_marbles_count_l3509_350972


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l3509_350958

/-- Represents the rates of biking, jogging, and swimming in km/h -/
structure Rates where
  biking : ℕ
  jogging : ℕ
  swimming : ℕ

/-- The sum of the squares of the rates -/
def sumOfSquares (r : Rates) : ℕ :=
  r.biking ^ 2 + r.jogging ^ 2 + r.swimming ^ 2

theorem rates_sum_of_squares : ∃ r : Rates,
  (3 * r.biking + 2 * r.jogging + 2 * r.swimming = 112) ∧
  (2 * r.biking + 3 * r.jogging + 4 * r.swimming = 129) ∧
  (sumOfSquares r = 1218) := by
  sorry

end NUMINAMATH_CALUDE_rates_sum_of_squares_l3509_350958


namespace NUMINAMATH_CALUDE_trihedral_angle_sum_bounds_l3509_350967

-- Define a trihedral angle
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real

-- State the theorem
theorem trihedral_angle_sum_bounds (t : TrihedralAngle) :
  180 < t.α + t.β + t.γ ∧ t.α + t.β + t.γ < 540 := by
  sorry

end NUMINAMATH_CALUDE_trihedral_angle_sum_bounds_l3509_350967


namespace NUMINAMATH_CALUDE_cost_price_correct_l3509_350930

/-- The cost price of an eye-protection lamp -/
def cost_price : ℝ := 150

/-- The original selling price of the lamp -/
def original_price : ℝ := 200

/-- The discount rate during the special period -/
def discount_rate : ℝ := 0.1

/-- The profit rate after the discount -/
def profit_rate : ℝ := 0.2

/-- Theorem stating that the cost price is correct given the conditions -/
theorem cost_price_correct : 
  original_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
sorry

end NUMINAMATH_CALUDE_cost_price_correct_l3509_350930


namespace NUMINAMATH_CALUDE_handshake_problem_l3509_350994

theorem handshake_problem :
  let n : ℕ := 6  -- number of people
  let handshakes := n * (n - 1) / 2  -- formula for total handshakes
  handshakes = 15
  := by sorry

end NUMINAMATH_CALUDE_handshake_problem_l3509_350994


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_determination_l3509_350939

/-- A geometric sequence is defined by its first term and common ratio -/
structure GeometricSequence where
  first_term : ℝ
  common_ratio : ℝ

/-- The nth term of a geometric sequence -/
def nth_term (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem geometric_sequence_first_term_determination 
  (seq : GeometricSequence) 
  (h5 : nth_term seq 5 = 72)
  (h8 : nth_term seq 8 = 576) : 
  seq.first_term = 4.5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_determination_l3509_350939


namespace NUMINAMATH_CALUDE_a_equals_five_l3509_350971

/-- Given the equation 632 - A9B = 41, where A and B are single digits, prove that A must equal 5. -/
theorem a_equals_five (A B : ℕ) (h1 : A ≤ 9) (h2 : B ≤ 9) (h3 : 632 - (100 * A + 10 * B) = 41) : A = 5 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_five_l3509_350971


namespace NUMINAMATH_CALUDE_derivative_of_f_l3509_350927

noncomputable def f (x : ℝ) : ℝ := x^3 / 3 + 1 / x

theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : 
  deriv f x = x^2 - 1 / x^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3509_350927


namespace NUMINAMATH_CALUDE_solve_equation_l3509_350956

theorem solve_equation (C D : ℚ) 
  (eq1 : 2 * C + 3 * D + 4 = 31)
  (eq2 : D = C + 2) :
  C = 21 / 5 ∧ D = 31 / 5 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l3509_350956


namespace NUMINAMATH_CALUDE_distance_to_origin_l3509_350985

theorem distance_to_origin (x y : ℝ) (h1 : y = 15) (h2 : x = 2 + Real.sqrt 105)
  (h3 : Real.sqrt ((x - 2)^2 + (y - 7)^2) = 13) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (334 + 4 * Real.sqrt 105) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3509_350985


namespace NUMINAMATH_CALUDE_pie_crust_flour_calculation_l3509_350931

theorem pie_crust_flour_calculation :
  let original_crusts : ℕ := 30
  let new_crusts : ℕ := 40
  let flour_per_original_crust : ℚ := 1 / 5
  let total_flour : ℚ := original_crusts * flour_per_original_crust
  let flour_per_new_crust : ℚ := total_flour / new_crusts
  flour_per_new_crust = 3 / 20 := by sorry

end NUMINAMATH_CALUDE_pie_crust_flour_calculation_l3509_350931


namespace NUMINAMATH_CALUDE_field_trip_difference_l3509_350987

/-- Given the number of vans, buses, people per van, and people per bus,
    prove that the difference between the number of people traveling by bus
    and the number of people traveling by van is 108.0 --/
theorem field_trip_difference (num_vans : ℝ) (num_buses : ℝ) 
                               (people_per_van : ℝ) (people_per_bus : ℝ) :
  num_vans = 6.0 →
  num_buses = 8.0 →
  people_per_van = 6.0 →
  people_per_bus = 18.0 →
  num_buses * people_per_bus - num_vans * people_per_van = 108.0 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_difference_l3509_350987


namespace NUMINAMATH_CALUDE_expansion_coefficient_a_l3509_350976

theorem expansion_coefficient_a (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^5 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a = 32 := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_a_l3509_350976


namespace NUMINAMATH_CALUDE_opposite_of_eight_l3509_350907

theorem opposite_of_eight :
  ∀ x : ℤ, x + 8 = 0 ↔ x = -8 := by sorry

end NUMINAMATH_CALUDE_opposite_of_eight_l3509_350907


namespace NUMINAMATH_CALUDE_logarithm_inequality_l3509_350996

theorem logarithm_inequality (m : ℝ) (a b c : ℝ) 
  (h1 : 1/10 < m ∧ m < 1) 
  (h2 : a = Real.log m) 
  (h3 : b = Real.log (m^2)) 
  (h4 : c = Real.log (m^3)) : 
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l3509_350996


namespace NUMINAMATH_CALUDE_expression_simplification_l3509_350921

theorem expression_simplification (x : ℝ) (hx : x ≠ 0) :
  ((2 * x^2)^3 - 6 * x^3 * (x^3 - 2 * x^2)) / (2 * x^4) = x^2 + 6 * x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3509_350921


namespace NUMINAMATH_CALUDE_dropped_students_scores_sum_l3509_350995

theorem dropped_students_scores_sum 
  (initial_students : ℕ) 
  (initial_average : ℝ) 
  (remaining_students : ℕ) 
  (new_average : ℝ) 
  (h1 : initial_students = 25) 
  (h2 : initial_average = 60.5) 
  (h3 : remaining_students = 23) 
  (h4 : new_average = 64.0) : 
  (initial_students : ℝ) * initial_average - (remaining_students : ℝ) * new_average = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_dropped_students_scores_sum_l3509_350995


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l3509_350944

theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 52) :
  (perimeter / 4) ^ 2 = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l3509_350944


namespace NUMINAMATH_CALUDE_ab_length_in_specific_triangle_l3509_350969

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def isAcute (t : Triangle) : Prop := sorry

def sideLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

def triangleArea (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem ab_length_in_specific_triangle :
  ∀ (t : Triangle),
    isAcute t →
    sideLength t.A t.C = 4 →
    sideLength t.B t.C = 3 →
    triangleArea t = 3 * Real.sqrt 3 →
    sideLength t.A t.B = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ab_length_in_specific_triangle_l3509_350969


namespace NUMINAMATH_CALUDE_inscribed_sphere_sum_l3509_350902

/-- A right cone with a sphere inscribed in it -/
structure InscribedSphere where
  baseRadius : ℝ
  height : ℝ
  sphereRadius : ℝ
  b : ℝ
  d : ℝ
  base_radius_positive : 0 < baseRadius
  height_positive : 0 < height
  sphere_radius_formula : sphereRadius = b * Real.sqrt d - b

/-- The theorem stating that b + d = 20 for the given conditions -/
theorem inscribed_sphere_sum (cone : InscribedSphere)
  (h1 : cone.baseRadius = 15)
  (h2 : cone.height = 30) :
  cone.b + cone.d = 20 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_sum_l3509_350902


namespace NUMINAMATH_CALUDE_debt_installment_problem_l3509_350974

/-- Proves that given 52 installments where the first 12 are x and the remaining 40 are (x + 65),
    if the average payment is $460, then x = $410. -/
theorem debt_installment_problem (x : ℝ) : 
  (12 * x + 40 * (x + 65)) / 52 = 460 → x = 410 := by
  sorry

end NUMINAMATH_CALUDE_debt_installment_problem_l3509_350974


namespace NUMINAMATH_CALUDE_arithmetic_progression_pairs_l3509_350912

/-- A pair of real numbers (a, b) forms an arithmetic progression with 10 and ab if
    the differences between consecutive terms are equal. -/
def is_arithmetic_progression (a b : ℝ) : Prop :=
  (a - 10 = b - a) ∧ (b - a = a * b - b)

/-- The only pairs (a, b) of real numbers such that 10, a, b, ab form an arithmetic progression
    are (4, -2) and (2.5, -5). -/
theorem arithmetic_progression_pairs :
  ∀ a b : ℝ, is_arithmetic_progression a b ↔ (a = 4 ∧ b = -2) ∨ (a = 2.5 ∧ b = -5) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_pairs_l3509_350912


namespace NUMINAMATH_CALUDE_movies_to_watch_l3509_350993

/-- Given a series with 17 movies, if 7 movies have been watched,
    then the number of movies still to watch is 10. -/
theorem movies_to_watch (total_movies : ℕ) (watched_movies : ℕ) : 
  total_movies = 17 → watched_movies = 7 → total_movies - watched_movies = 10 := by
  sorry

end NUMINAMATH_CALUDE_movies_to_watch_l3509_350993


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3509_350955

theorem inequality_equivalence (x : ℝ) : 
  |((7-x)/4)| < 3 ↔ 2 < x ∧ x < 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3509_350955


namespace NUMINAMATH_CALUDE_average_of_five_quantities_l3509_350915

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 19) :
  (q1 + q2 + q3 + q4 + q5) / 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_quantities_l3509_350915


namespace NUMINAMATH_CALUDE_smaller_cubes_count_l3509_350983

theorem smaller_cubes_count (larger_cube_volume : ℝ) (smaller_cube_volume : ℝ) (surface_area_difference : ℝ) :
  larger_cube_volume = 216 →
  smaller_cube_volume = 1 →
  surface_area_difference = 1080 →
  (smaller_cube_volume^(1/3) * 6 * (larger_cube_volume / smaller_cube_volume) - larger_cube_volume^(2/3) * 6 = surface_area_difference) →
  (larger_cube_volume / smaller_cube_volume) = 216 :=
by
  sorry

#check smaller_cubes_count

end NUMINAMATH_CALUDE_smaller_cubes_count_l3509_350983


namespace NUMINAMATH_CALUDE_sum_of_unique_areas_l3509_350938

-- Define a structure for right triangles with integer leg lengths
structure SuperCoolTriangle where
  a : ℕ
  b : ℕ
  h : (a * b) / 2 = 3 * (a + b)

-- Define a function to calculate the area of a triangle
def triangleArea (t : SuperCoolTriangle) : ℕ := (t.a * t.b) / 2

-- Define a function to get all unique areas of super cool triangles
def uniqueAreas : List ℕ := sorry

-- Theorem statement
theorem sum_of_unique_areas : (uniqueAreas.sum) = 471 := by sorry

end NUMINAMATH_CALUDE_sum_of_unique_areas_l3509_350938


namespace NUMINAMATH_CALUDE_seven_power_minus_three_times_two_power_l3509_350957

theorem seven_power_minus_three_times_two_power (x y : ℕ+) : 
  7^(x.val) - 3 * 2^(y.val) = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) := by
sorry

end NUMINAMATH_CALUDE_seven_power_minus_three_times_two_power_l3509_350957


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_solve_unknown_blanket_rate_l3509_350950

/-- Proves that the unknown rate of two blankets is 275, given the conditions of the problem --/
theorem unknown_blanket_rate : ℕ → Prop := fun x =>
  let total_blankets : ℕ := 12
  let average_price : ℕ := 150
  let total_cost : ℕ := total_blankets * average_price
  let known_cost : ℕ := 5 * 100 + 5 * 150
  2 * x = total_cost - known_cost → x = 275

/-- Solution to the unknown_blanket_rate theorem --/
theorem solve_unknown_blanket_rate : unknown_blanket_rate 275 := by
  sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_solve_unknown_blanket_rate_l3509_350950


namespace NUMINAMATH_CALUDE_same_point_on_bisector_l3509_350932

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the angle bisector of the first and third quadrants
def firstThirdQuadrantBisector : Set Point2D :=
  { p : Point2D | p.x = p.y }

theorem same_point_on_bisector (a b : ℝ) :
  (Point2D.mk a b = Point2D.mk b a) →
  Point2D.mk a b ∈ firstThirdQuadrantBisector := by
  sorry

end NUMINAMATH_CALUDE_same_point_on_bisector_l3509_350932


namespace NUMINAMATH_CALUDE_thursday_beef_sales_l3509_350951

/-- Given a store's beef sales over three days, prove that Thursday's sales were 210 pounds -/
theorem thursday_beef_sales (x : ℝ) : 
  (x + 2*x + 150) / 3 = 260 → x = 210 := by sorry

end NUMINAMATH_CALUDE_thursday_beef_sales_l3509_350951


namespace NUMINAMATH_CALUDE_lacrosse_football_difference_l3509_350968

/-- Represents the number of bottles filled for each team and the total --/
structure BottleCounts where
  total : ℕ
  football : ℕ
  soccer : ℕ
  rugby : ℕ
  lacrosse : ℕ

/-- The difference in bottles between lacrosse and football teams --/
def bottleDifference (counts : BottleCounts) : ℕ :=
  counts.lacrosse - counts.football

/-- Theorem stating the difference in bottles between lacrosse and football teams --/
theorem lacrosse_football_difference (counts : BottleCounts) 
  (h1 : counts.total = 254)
  (h2 : counts.football = 11 * 6)
  (h3 : counts.soccer = 53)
  (h4 : counts.rugby = 49)
  (h5 : counts.total = counts.football + counts.soccer + counts.rugby + counts.lacrosse) :
  bottleDifference counts = 20 := by
  sorry

#check lacrosse_football_difference

end NUMINAMATH_CALUDE_lacrosse_football_difference_l3509_350968


namespace NUMINAMATH_CALUDE_sum_of_digits_less_than_1000_is_13500_l3509_350946

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def sum_of_digits_less_than_1000 : ℕ :=
  (List.range 1000).map digit_sum |> List.sum

theorem sum_of_digits_less_than_1000_is_13500 :
  sum_of_digits_less_than_1000 = 13500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_less_than_1000_is_13500_l3509_350946


namespace NUMINAMATH_CALUDE_equation_solution_l3509_350992

theorem equation_solution : ∃! x : ℝ, (x ≠ 1 ∧ x ≠ -1) ∧ (x / (x - 1) = 2 / (x^2 - 1)) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3509_350992


namespace NUMINAMATH_CALUDE_average_stickers_per_album_l3509_350962

def album_stickers : List ℕ := [5, 7, 9, 14, 19, 12, 26, 18, 11, 15]

theorem average_stickers_per_album :
  (album_stickers.sum : ℚ) / album_stickers.length = 13.6 := by
  sorry

end NUMINAMATH_CALUDE_average_stickers_per_album_l3509_350962


namespace NUMINAMATH_CALUDE_average_minutes_run_per_day_l3509_350973

/-- The average number of minutes run per day by sixth graders -/
def sixth_grade_avg : ℚ := 20

/-- The average number of minutes run per day by seventh graders -/
def seventh_grade_avg : ℚ := 12

/-- The average number of minutes run per day by eighth graders -/
def eighth_grade_avg : ℚ := 18

/-- The ratio of sixth graders to eighth graders -/
def sixth_to_eighth_ratio : ℚ := 3

/-- The ratio of seventh graders to eighth graders -/
def seventh_to_eighth_ratio : ℚ := 3

/-- The theorem stating the average number of minutes run per day by all students -/
theorem average_minutes_run_per_day :
  let total_students := sixth_to_eighth_ratio + seventh_to_eighth_ratio + 1
  let total_minutes := sixth_grade_avg * sixth_to_eighth_ratio + 
                       seventh_grade_avg * seventh_to_eighth_ratio + 
                       eighth_grade_avg
  total_minutes / total_students = 114 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_minutes_run_per_day_l3509_350973


namespace NUMINAMATH_CALUDE_reflection_y_axis_l3509_350988

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

theorem reflection_y_axis : 
  let A : ℝ × ℝ := (-3, 4)
  reflect_y A = (3, 4) := by sorry

end NUMINAMATH_CALUDE_reflection_y_axis_l3509_350988


namespace NUMINAMATH_CALUDE_unique_positive_integer_l3509_350941

theorem unique_positive_integer : ∃! (x : ℕ), x > 0 ∧ 15 * x = x^2 + 56 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_l3509_350941


namespace NUMINAMATH_CALUDE_donut_selections_l3509_350963

theorem donut_selections :
  (Nat.choose 9 3) = 84 := by
  sorry

end NUMINAMATH_CALUDE_donut_selections_l3509_350963


namespace NUMINAMATH_CALUDE_feeding_theorem_l3509_350981

/-- Represents the number of animal pairs in the sanctuary -/
def num_pairs : ℕ := 6

/-- Represents the feeding order constraint for tigers -/
def tiger_constraint : Prop := true

/-- Represents the constraint that no two same-gender animals can be fed consecutively -/
def alternating_gender_constraint : Prop := true

/-- Represents that the first animal fed is the male lion -/
def starts_with_male_lion : Prop := true

/-- Calculates the number of ways to feed the animals given the constraints -/
def feeding_ways : ℕ := 14400

/-- Theorem stating the number of ways to feed the animals -/
theorem feeding_theorem :
  num_pairs = 6 ∧
  tiger_constraint ∧
  alternating_gender_constraint ∧
  starts_with_male_lion →
  feeding_ways = 14400 := by
  sorry

end NUMINAMATH_CALUDE_feeding_theorem_l3509_350981
