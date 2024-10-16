import Mathlib

namespace NUMINAMATH_CALUDE_range_of_m_l3171_317172

def f (x : ℝ) : ℝ := sorry

theorem range_of_m (h1 : ∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≠ 0)
                   (h2 : ∀ x, f (-x) = -f x)
                   (h3 : ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ 2 → f x > f y)
                   (h4 : ∀ m, f (1 + m) + f m < 0) :
  ∀ m, (-1/2 < m ∧ m ≤ 1) ↔ (∃ x, -2 ≤ x ∧ x ≤ 2 ∧ f (1 + x) + f x < 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3171_317172


namespace NUMINAMATH_CALUDE_regression_slope_effect_l3171_317113

/-- Represents a simple linear regression model -/
structure LinearRegression where
  intercept : ℝ
  slope : ℝ

/-- The predicted value of y given x in a linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.intercept + model.slope * x

/-- The change in y when x increases by one unit -/
def change_in_y (model : LinearRegression) : ℝ :=
  predict model 1 - predict model 0

theorem regression_slope_effect (model : LinearRegression) 
  (h : model = {intercept := 3, slope := -5}) : 
  change_in_y model = -5 := by
  sorry

end NUMINAMATH_CALUDE_regression_slope_effect_l3171_317113


namespace NUMINAMATH_CALUDE_f_bounded_by_four_l3171_317122

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 3|

-- State the theorem
theorem f_bounded_by_four : ∀ x : ℝ, |f x| ≤ 4 := by sorry

end NUMINAMATH_CALUDE_f_bounded_by_four_l3171_317122


namespace NUMINAMATH_CALUDE_correct_logarithms_l3171_317160

-- Define the logarithm function
noncomputable def log (x : ℝ) : ℝ := Real.log x

-- Define the variables a, b, and c
variable (a b c : ℝ)

-- Define the given logarithmic relationships
axiom log_3 : log 3 = 2*a - b
axiom log_5 : log 5 = a + c
axiom log_2 : log 2 = 1 - a - c
axiom log_9 : log 9 = 4*a - 2*b
axiom log_14 : log 14 = 1 - c + 2*b

-- State the theorem to be proved
theorem correct_logarithms :
  log 1.5 = 3*a - b + c - 1 ∧ log 7 = 2*b + c :=
by sorry

end NUMINAMATH_CALUDE_correct_logarithms_l3171_317160


namespace NUMINAMATH_CALUDE_xenia_june_earnings_l3171_317112

/-- Xenia's earnings during the first two weeks of June -/
def xenia_earnings (hours_week1 hours_week2 : ℕ) (wage_difference : ℚ) : ℚ :=
  let hourly_wage := wage_difference / (hours_week2 - hours_week1 : ℚ)
  hourly_wage * (hours_week1 + hours_week2 : ℚ)

/-- Theorem stating Xenia's earnings during the first two weeks of June -/
theorem xenia_june_earnings :
  xenia_earnings 15 22 (47.60 : ℚ) = (251.60 : ℚ) := by
  sorry

#eval xenia_earnings 15 22 (47.60 : ℚ)

end NUMINAMATH_CALUDE_xenia_june_earnings_l3171_317112


namespace NUMINAMATH_CALUDE_seashells_given_away_l3171_317145

/-- Represents the number of seashells Maura collected and gave away -/
structure SeashellCollection where
  total : ℕ
  left : ℕ
  given : ℕ

/-- Theorem stating that the number of seashells given away is the difference between total and left -/
theorem seashells_given_away (collection : SeashellCollection) 
  (h1 : collection.total = 75)
  (h2 : collection.left = 57)
  (h3 : collection.given = collection.total - collection.left) :
  collection.given = 18 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_away_l3171_317145


namespace NUMINAMATH_CALUDE_power_two_mod_seven_l3171_317135

theorem power_two_mod_seven : (2^200 - 3) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_mod_seven_l3171_317135


namespace NUMINAMATH_CALUDE_parallel_line_m_value_l3171_317105

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Get the line passing through two points -/
def line_through_points (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y
    b := p1.x - p2.x
    c := p2.x * p1.y - p1.x * p2.y }

/-- The main theorem -/
theorem parallel_line_m_value :
  ∀ m : ℝ,
  let A : Point := ⟨-2, m⟩
  let B : Point := ⟨m, 4⟩
  let L1 : Line := line_through_points A B
  let L2 : Line := ⟨2, 1, -1⟩
  are_parallel L1 L2 → m = -8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_m_value_l3171_317105


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l3171_317131

theorem triangle_inequality_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  Real.sqrt (a / (b + c - a)) + Real.sqrt (b / (c + a - b)) + Real.sqrt (c / (a + b - c)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l3171_317131


namespace NUMINAMATH_CALUDE_four_distinct_concentric_circles_touch_ellipse_l3171_317100

/-- An ellipse with center O, major axis length 2a, and minor axis length 2b. -/
structure Ellipse where
  O : ℝ × ℝ
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_noncircular : a ≠ b

/-- A circle with center P and radius r. -/
structure Circle where
  P : ℝ × ℝ
  r : ℝ
  h_positive : 0 < r

/-- A point Q on an ellipse. -/
def PointOnEllipse (E : Ellipse) : Type := { Q : ℝ × ℝ // ∃ (t : ℝ), Q.1 = E.O.1 + E.a * Real.cos t ∧ Q.2 = E.O.2 + E.b * Real.sin t }

/-- Predicate to check if a circle touches an ellipse. -/
def TouchesEllipse (E : Ellipse) (C : Circle) : Prop :=
  ∃ (Q : PointOnEllipse E), Real.sqrt ((Q.val.1 - C.P.1)^2 + (Q.val.2 - C.P.2)^2) = C.r

/-- Theorem: There exist four distinct concentric circles touching a non-circular ellipse. -/
theorem four_distinct_concentric_circles_touch_ellipse (E : Ellipse) :
  ∃ (P : ℝ × ℝ) (r₁ r₂ r₃ r₄ : ℝ),
    r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄ ∧
    TouchesEllipse E ⟨P, r₁, sorry⟩ ∧
    TouchesEllipse E ⟨P, r₂, sorry⟩ ∧
    TouchesEllipse E ⟨P, r₃, sorry⟩ ∧
    TouchesEllipse E ⟨P, r₄, sorry⟩ :=
  sorry

end NUMINAMATH_CALUDE_four_distinct_concentric_circles_touch_ellipse_l3171_317100


namespace NUMINAMATH_CALUDE_second_part_multiplier_l3171_317117

theorem second_part_multiplier (total : ℕ) (first_part : ℕ) (k : ℕ) : 
  total = 36 →
  first_part = 19 →
  8 * first_part + k * (total - first_part) = 203 →
  k = 3 := by sorry

end NUMINAMATH_CALUDE_second_part_multiplier_l3171_317117


namespace NUMINAMATH_CALUDE_unique_prime_pair_l3171_317102

theorem unique_prime_pair : ∃! (p q : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime (p^2 + 2*p*q^2 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l3171_317102


namespace NUMINAMATH_CALUDE_equation_solution_l3171_317161

theorem equation_solution :
  ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 57 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3171_317161


namespace NUMINAMATH_CALUDE_not_all_exp_increasing_l3171_317116

-- Define the exponential function
noncomputable def exp (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem not_all_exp_increasing :
  ¬ (∀ (a : ℝ), a > 0 → (∀ (x y : ℝ), x < y → exp a x < exp a y)) :=
by sorry

end NUMINAMATH_CALUDE_not_all_exp_increasing_l3171_317116


namespace NUMINAMATH_CALUDE_optimal_arrangement_maximizes_sum_l3171_317118

/-- The type of arrangements of numbers from 1 to 1999 in a circle -/
def Arrangement := Fin 1999 → Fin 1999

/-- The sum of products of all sets of 10 consecutive numbers in an arrangement -/
def sumOfProducts (a : Arrangement) : ℕ :=
  sorry

/-- The optimal arrangement of numbers -/
def optimalArrangement : Arrangement :=
  fun i => if i.val % 2 = 0 then (1999 - i.val + 1) else (i.val + 1)

/-- Theorem stating that the optimal arrangement maximizes the sum of products -/
theorem optimal_arrangement_maximizes_sum :
  ∀ a : Arrangement, sumOfProducts a ≤ sumOfProducts optimalArrangement :=
sorry

end NUMINAMATH_CALUDE_optimal_arrangement_maximizes_sum_l3171_317118


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3171_317169

/-- Given a rhombus with side length 60 units and shorter diagonal 56 units,
    the longer diagonal has a length of 32√11 units. -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diag : ℝ) (longer_diag : ℝ) 
    (h1 : side = 60) 
    (h2 : shorter_diag = 56) 
    (h3 : side^2 = (shorter_diag/2)^2 + (longer_diag/2)^2) : 
  longer_diag = 32 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3171_317169


namespace NUMINAMATH_CALUDE_parallel_transitive_l3171_317193

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def IsParallel (v w : V) : Prop := ∃ (k : ℝ), v = k • w

theorem parallel_transitive (a b c : V) :
  IsParallel a b → IsParallel b c → b ≠ 0 → IsParallel a c :=
sorry

end NUMINAMATH_CALUDE_parallel_transitive_l3171_317193


namespace NUMINAMATH_CALUDE_fraction_simplification_l3171_317133

theorem fraction_simplification : (-150 + 50) / (-50) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3171_317133


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l3171_317178

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2 * Real.sqrt (a*b*c)) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l3171_317178


namespace NUMINAMATH_CALUDE_farmer_land_problem_l3171_317197

theorem farmer_land_problem (original_land : ℚ) : 
  (9 / 10 : ℚ) * original_land = 10 → original_land = 11 + 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_problem_l3171_317197


namespace NUMINAMATH_CALUDE_average_problem_l3171_317150

theorem average_problem (y : ℝ) : 
  (15 + 25 + 35 + y) / 4 = 30 → y = 45 := by
sorry

end NUMINAMATH_CALUDE_average_problem_l3171_317150


namespace NUMINAMATH_CALUDE_eric_containers_l3171_317191

/-- The number of containers Eric has for his colored pencils. -/
def number_of_containers (initial_pencils : ℕ) (additional_pencils : ℕ) (pencils_per_container : ℕ) : ℕ :=
  (initial_pencils + additional_pencils) / pencils_per_container

theorem eric_containers :
  number_of_containers 150 30 36 = 5 := by
  sorry

end NUMINAMATH_CALUDE_eric_containers_l3171_317191


namespace NUMINAMATH_CALUDE_sum_range_l3171_317148

theorem sum_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y + 4*y^2 = 1) :
  1/2 < x + y ∧ x + y < 1 := by
sorry

end NUMINAMATH_CALUDE_sum_range_l3171_317148


namespace NUMINAMATH_CALUDE_internally_tangent_circles_distance_l3171_317166

theorem internally_tangent_circles_distance (r₁ r₂ d : ℝ) : 
  r₁ = 3 → r₂ = 6 → d = r₂ - r₁ → d = 3 := by sorry

end NUMINAMATH_CALUDE_internally_tangent_circles_distance_l3171_317166


namespace NUMINAMATH_CALUDE_power_five_plus_five_mod_eight_l3171_317162

theorem power_five_plus_five_mod_eight : (5^123 + 5) % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_five_plus_five_mod_eight_l3171_317162


namespace NUMINAMATH_CALUDE_square_of_difference_positive_l3171_317138

theorem square_of_difference_positive {a b : ℝ} (h : a ≠ b) : (a - b)^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_positive_l3171_317138


namespace NUMINAMATH_CALUDE_divisible_by_27_l3171_317114

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, 2^(2*n - 1) - 9*n^2 + 21*n - 14 = 27*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_27_l3171_317114


namespace NUMINAMATH_CALUDE_sum_of_two_smallest_prime_factors_of_294_l3171_317183

theorem sum_of_two_smallest_prime_factors_of_294 :
  ∃ (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p < q ∧
    p ∣ 294 ∧ 
    q ∣ 294 ∧
    (∀ (r : ℕ), Nat.Prime r → r ∣ 294 → r = p ∨ r ≥ q) ∧
    p + q = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_two_smallest_prime_factors_of_294_l3171_317183


namespace NUMINAMATH_CALUDE_painting_width_l3171_317141

/-- Given a wall and a painting with specific dimensions, prove the width of the painting -/
theorem painting_width
  (wall_height : ℝ)
  (wall_width : ℝ)
  (painting_height : ℝ)
  (painting_area_percentage : ℝ)
  (h1 : wall_height = 5)
  (h2 : wall_width = 10)
  (h3 : painting_height = 2)
  (h4 : painting_area_percentage = 0.16)
  : (wall_height * wall_width * painting_area_percentage) / painting_height = 4 := by
  sorry

end NUMINAMATH_CALUDE_painting_width_l3171_317141


namespace NUMINAMATH_CALUDE_bamboo_pole_sections_l3171_317187

/-- Represents the properties of a bamboo pole with n sections -/
structure BambooPole (n : ℕ) where
  -- The common difference of the arithmetic sequence
  d : ℝ
  -- The number of sections is at least 6
  h_n_ge_6 : n ≥ 6
  -- The length of the top section is 10 cm
  h_top_length : 10 = 10
  -- The total length of the last three sections is 114 cm
  h_last_three : (10 + (n - 3) * d) + (10 + (n - 2) * d) + (10 + (n - 1) * d) = 114
  -- The length of the 6th section is the geometric mean of the lengths of the first and last sections
  h_geometric_mean : (10 + 5 * d)^2 = 10 * (10 + (n - 1) * d)

/-- The number of sections in the bamboo pole is 16 -/
theorem bamboo_pole_sections : ∃ (n : ℕ), ∃ (p : BambooPole n), n = 16 :=
sorry

end NUMINAMATH_CALUDE_bamboo_pole_sections_l3171_317187


namespace NUMINAMATH_CALUDE_fraction_power_equality_l3171_317184

theorem fraction_power_equality : (72000 ^ 5) / (18000 ^ 5) = 1024 := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l3171_317184


namespace NUMINAMATH_CALUDE_non_adjacent_book_arrangements_l3171_317198

/-- Represents the number of books of each subject -/
structure BookCounts where
  chinese : Nat
  math : Nat
  physics : Nat

/-- Calculates the total number of books -/
def totalBooks (counts : BookCounts) : Nat :=
  counts.chinese + counts.math + counts.physics

/-- Calculates the number of permutations of n items -/
def permutations (n : Nat) : Nat :=
  Nat.factorial n

/-- Calculates the number of arrangements where books of the same subject are not adjacent -/
def nonAdjacentArrangements (counts : BookCounts) : Nat :=
  let total := totalBooks counts
  let allArrangements := permutations total
  let chineseAdjacent := (permutations (total - counts.chinese + 1)) * (permutations counts.chinese)
  let mathAdjacent := (permutations (total - counts.math + 1)) * (permutations counts.math)
  let bothAdjacent := (permutations (total - counts.chinese - counts.math + 2)) * 
                      (permutations counts.chinese) * (permutations counts.math)
  allArrangements - chineseAdjacent - mathAdjacent + bothAdjacent

theorem non_adjacent_book_arrangements :
  let counts : BookCounts := { chinese := 2, math := 2, physics := 1 }
  nonAdjacentArrangements counts = 48 := by
  sorry

end NUMINAMATH_CALUDE_non_adjacent_book_arrangements_l3171_317198


namespace NUMINAMATH_CALUDE_only_sphere_all_circular_l3171_317170

-- Define the geometric shapes
inductive Shape
  | Cuboid
  | Cylinder
  | Cone
  | Sphere

-- Define the views
inductive View
  | Front
  | Left
  | Top

-- Define a function to determine if a view is circular
def isCircularView (s : Shape) (v : View) : Prop :=
  match s, v with
  | Shape.Sphere, _ => True
  | Shape.Cylinder, View.Top => True
  | _, _ => False

-- Define a function to check if all views are circular
def allViewsCircular (s : Shape) : Prop :=
  isCircularView s View.Front ∧ isCircularView s View.Left ∧ isCircularView s View.Top

-- Theorem: Only the Sphere has circular views from all perspectives
theorem only_sphere_all_circular :
  ∀ s : Shape, allViewsCircular s ↔ s = Shape.Sphere :=
sorry

end NUMINAMATH_CALUDE_only_sphere_all_circular_l3171_317170


namespace NUMINAMATH_CALUDE_right_triangle_cos_z_l3171_317120

theorem right_triangle_cos_z (X Y Z : Real) (h1 : X + Y + Z = π) (h2 : X = π/2) (h3 : Real.sin Y = 3/5) :
  Real.cos Z = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cos_z_l3171_317120


namespace NUMINAMATH_CALUDE_equivalence_of_statements_l3171_317190

theorem equivalence_of_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1/a + 1/b = Real.sqrt (a*b)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_equivalence_of_statements_l3171_317190


namespace NUMINAMATH_CALUDE_cos_difference_angle_l3171_317195

theorem cos_difference_angle (α β : ℝ) : 
  Real.cos (α - β) = Real.cos α * Real.cos β + Real.sin α * Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_angle_l3171_317195


namespace NUMINAMATH_CALUDE_quiz_correct_answers_l3171_317110

theorem quiz_correct_answers (cherry kim nicole : ℕ) 
  (h1 : nicole + 3 = kim)
  (h2 : kim = cherry + 8)
  (h3 : cherry = 17) : 
  nicole = 22 := by
sorry

end NUMINAMATH_CALUDE_quiz_correct_answers_l3171_317110


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l3171_317111

theorem right_triangle_third_side (a b c : ℝ) : 
  a = 6 ∧ b = 8 ∧ c > 0 ∧ a^2 + b^2 = c^2 → c = 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l3171_317111


namespace NUMINAMATH_CALUDE_zeros_when_a_1_b_neg_2_range_of_a_for_two_distinct_zeros_l3171_317182

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + b - 1

-- Theorem for part (1)
theorem zeros_when_a_1_b_neg_2 :
  let f := f 1 (-2)
  ∀ x, f x = 0 ↔ x = 3 ∨ x = -1 := by sorry

-- Theorem for part (2)
theorem range_of_a_for_two_distinct_zeros :
  ∀ a : ℝ, (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) ↔ 0 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_zeros_when_a_1_b_neg_2_range_of_a_for_two_distinct_zeros_l3171_317182


namespace NUMINAMATH_CALUDE_B_subset_A_l3171_317165

def A : Set ℕ := {2, 0, 3}
def B : Set ℕ := {2, 3}

theorem B_subset_A : B ⊆ A := by sorry

end NUMINAMATH_CALUDE_B_subset_A_l3171_317165


namespace NUMINAMATH_CALUDE_tv_price_calculation_l3171_317151

/-- Calculates the final price of an item given the original price, discount rate, tax rate, and rebate amount. -/
def finalPrice (originalPrice : ℝ) (discountRate : ℝ) (taxRate : ℝ) (rebate : ℝ) : ℝ :=
  let salePrice := originalPrice * (1 - discountRate)
  let priceWithTax := salePrice * (1 + taxRate)
  priceWithTax - rebate

/-- Theorem stating that the final price of a $1200 item with 30% discount, 8% tax, and $50 rebate is $857.2. -/
theorem tv_price_calculation :
  finalPrice 1200 0.30 0.08 50 = 857.2 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_calculation_l3171_317151


namespace NUMINAMATH_CALUDE_remainder_of_n_l3171_317154

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 7 = 4) (h2 : n^3 % 7 = 6) : n % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_l3171_317154


namespace NUMINAMATH_CALUDE_min_value_f_prime_2_l3171_317123

theorem min_value_f_prime_2 (a : ℝ) (h : a > 0) :
  let f := fun x : ℝ => x^3 + 2*a*x^2 + (1/a)*x
  let f_prime := fun x : ℝ => 3*x^2 + 4*a*x + 1/a
  ∀ x : ℝ, f_prime 2 ≥ 12 + 4*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_f_prime_2_l3171_317123


namespace NUMINAMATH_CALUDE_square_sum_equals_product_l3171_317164

theorem square_sum_equals_product (x y z t : ℤ) :
  x^2 + y^2 + z^2 + t^2 = 2*x*y*z*t → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_product_l3171_317164


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3171_317108

theorem diophantine_equation_solutions :
  ∀ x y : ℕ, x^2 + (x + y)^2 = (x + 9)^2 ↔ (x = 0 ∧ y = 9) ∨ (x = 8 ∧ y = 7) ∨ (x = 20 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3171_317108


namespace NUMINAMATH_CALUDE_dice_sum_probability_l3171_317146

theorem dice_sum_probability (n : ℕ) : n = 36 →
  ∃ (d1 d2 : Finset ℕ),
    d1.card = 6 ∧ d2.card = 6 ∧
    (∀ k : ℕ, k ∈ Finset.range (n + 1) →
      (∃! (x y : ℕ), x ∈ d1 ∧ y ∈ d2 ∧ x + y = k)) :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_probability_l3171_317146


namespace NUMINAMATH_CALUDE_oranges_picked_total_l3171_317144

theorem oranges_picked_total (mary_oranges jason_oranges : ℕ) 
  (h1 : mary_oranges = 122) 
  (h2 : jason_oranges = 105) : 
  mary_oranges + jason_oranges = 227 := by
  sorry

end NUMINAMATH_CALUDE_oranges_picked_total_l3171_317144


namespace NUMINAMATH_CALUDE_parabola_c_value_l3171_317149

/-- A parabola with equation y = 2x^2 + bx + c passes through the points (1,5) and (3,17).
    This theorem proves that the value of c is 5. -/
theorem parabola_c_value :
  ∀ b c : ℝ,
  (5 : ℝ) = 2 * (1 : ℝ)^2 + b * (1 : ℝ) + c →
  (17 : ℝ) = 2 * (3 : ℝ)^2 + b * (3 : ℝ) + c →
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3171_317149


namespace NUMINAMATH_CALUDE_speed_ratio_l3171_317126

-- Define the speeds of A and B
def v_A : ℝ := sorry
def v_B : ℝ := sorry

-- Define the initial position of B
def initial_B_position : ℝ := -800

-- Define the equidistant condition at 3 minutes
def equidistant_3min : Prop :=
  3 * v_A = abs (initial_B_position + 3 * v_B)

-- Define the equidistant condition at 9 minutes
def equidistant_9min : Prop :=
  9 * v_A = abs (initial_B_position + 9 * v_B)

-- Theorem statement
theorem speed_ratio :
  equidistant_3min →
  equidistant_9min →
  v_A / v_B = 9 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_l3171_317126


namespace NUMINAMATH_CALUDE_marc_journey_fraction_l3171_317171

/-- Represents the time in minutes for a round trip journey -/
def roundTripTime (cyclingTime walkingTime : ℝ) : ℝ := cyclingTime + walkingTime

/-- Represents the time for Marc's modified journey -/
def modifiedJourneyTime (cyclingFraction : ℝ) : ℝ :=
  20 * cyclingFraction + 60 * (1 - cyclingFraction)

theorem marc_journey_fraction :
  ∃ (cyclingFraction : ℝ),
    roundTripTime 20 60 = 80 ∧
    modifiedJourneyTime cyclingFraction = 52 ∧
    cyclingFraction = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_marc_journey_fraction_l3171_317171


namespace NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l3171_317130

/-- The average speed of a car traveling different distances in two hours -/
theorem average_speed_two_hours (d1 d2 : ℝ) : 
  d1 > 0 → d2 > 0 → (d1 + d2) / 2 = (d1 + d2) / 2 := by
  sorry

/-- The average speed of a car traveling 50 km in the first hour and 60 km in the second hour is 55 km/h -/
theorem car_average_speed : (50 + 60) / 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l3171_317130


namespace NUMINAMATH_CALUDE_parallelogram_angle_measure_l3171_317107

/-- In a parallelogram, if one angle exceeds the other by 50 degrees,
    and the smaller angle is 65 degrees, then the larger angle is 115 degrees. -/
theorem parallelogram_angle_measure (smaller_angle larger_angle : ℝ) : 
  smaller_angle = 65 →
  larger_angle = smaller_angle + 50 →
  larger_angle = 115 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_angle_measure_l3171_317107


namespace NUMINAMATH_CALUDE_impossible_arrangement_l3171_317134

theorem impossible_arrangement :
  ¬ ∃ (grid : Matrix (Fin 6) (Fin 7) ℕ),
    (∀ i j, grid i j ∈ Set.range (fun n => n + 1) ∩ Set.Icc 1 42) ∧
    (∀ i j, ∃! p, grid p j = grid i j) ∧
    (∀ i j, Even (grid i j + grid (i + 1) j)) :=
by sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l3171_317134


namespace NUMINAMATH_CALUDE_investment_percentage_l3171_317168

/-- Given an investment scenario, prove that the unknown percentage is 7% -/
theorem investment_percentage (total_investment : ℝ) (known_rate : ℝ) (total_interest : ℝ) (amount_at_unknown_rate : ℝ) (unknown_rate : ℝ) :
  total_investment = 12000 ∧
  known_rate = 0.09 ∧
  total_interest = 970 ∧
  amount_at_unknown_rate = 5500 ∧
  amount_at_unknown_rate * unknown_rate + (total_investment - amount_at_unknown_rate) * known_rate = total_interest →
  unknown_rate = 0.07 := by
  sorry

end NUMINAMATH_CALUDE_investment_percentage_l3171_317168


namespace NUMINAMATH_CALUDE_integer_sum_and_square_is_twelve_l3171_317104

theorem integer_sum_and_square_is_twelve : ∃ N : ℕ+, (N : ℤ)^2 + (N : ℤ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_and_square_is_twelve_l3171_317104


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3171_317153

theorem decimal_to_fraction : 
  (3.56 : ℚ) = 89 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3171_317153


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l3171_317174

theorem r_value_when_n_is_3 : 
  ∀ (n s r : ℕ), 
    n = 3 → 
    s = 2^n + 2 → 
    r = 4^s + 3*s → 
    r = 1048606 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l3171_317174


namespace NUMINAMATH_CALUDE_f_of_2_eq_6_l3171_317109

/-- The function f(x) = x^3 - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- Theorem: f(2) = 6 -/
theorem f_of_2_eq_6 : f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_eq_6_l3171_317109


namespace NUMINAMATH_CALUDE_fraction_power_equality_l3171_317158

theorem fraction_power_equality : (125000 : ℝ)^5 / (25000 : ℝ)^5 = 3125 := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l3171_317158


namespace NUMINAMATH_CALUDE_melanie_dimes_proof_l3171_317136

def final_dimes (initial : ℕ) (received : ℕ) (given_away : ℕ) : ℕ :=
  initial + received - given_away

theorem melanie_dimes_proof :
  final_dimes 7 8 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_proof_l3171_317136


namespace NUMINAMATH_CALUDE_line_passes_through_center_line_is_diameter_l3171_317163

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- Theorem: The line passes through the center of the circle
theorem line_passes_through_center :
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq x y :=
sorry

-- Theorem: The line is a diameter of the circle
theorem line_is_diameter :
  ∀ (x y : ℝ), circle_eq x y → line_eq x y → 
  ∃ (x' y' : ℝ), circle_eq x' y' ∧ line_eq x' y' ∧ 
  (x - x')^2 + (y - y')^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_center_line_is_diameter_l3171_317163


namespace NUMINAMATH_CALUDE_purple_book_pages_purple_book_pages_proof_l3171_317188

theorem purple_book_pages : ℕ → Prop :=
  fun p =>
    let orange_pages : ℕ := 510
    let purple_books_read : ℕ := 5
    let orange_books_read : ℕ := 4
    let page_difference : ℕ := 890
    orange_books_read * orange_pages - purple_books_read * p = page_difference →
    p = 230

-- The proof goes here
theorem purple_book_pages_proof : purple_book_pages 230 := by
  sorry

end NUMINAMATH_CALUDE_purple_book_pages_purple_book_pages_proof_l3171_317188


namespace NUMINAMATH_CALUDE_trip_duration_l3171_317147

theorem trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) :
  initial_speed = 70 ∧
  initial_time = 4 ∧
  additional_speed = 60 ∧
  average_speed = 65 →
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) / total_time = average_speed ∧
    total_time = 8 :=
by sorry

end NUMINAMATH_CALUDE_trip_duration_l3171_317147


namespace NUMINAMATH_CALUDE_min_value_at_seven_l3171_317129

/-- The quadratic function f(x) = x^2 - 14x + 45 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 45

theorem min_value_at_seven :
  ∀ x : ℝ, f 7 ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_min_value_at_seven_l3171_317129


namespace NUMINAMATH_CALUDE_speed_ratio_l3171_317175

-- Define the speeds of A and B
def speed_A : ℝ := sorry
def speed_B : ℝ := sorry

-- Define the initial position of B
def initial_B_position : ℝ := 600

-- Define the time when they are first equidistant
def first_equidistant_time : ℝ := 3

-- Define the time when they are second equidistant
def second_equidistant_time : ℝ := 12

-- Define the condition for being equidistant at the first time
def first_equidistant_condition : Prop :=
  (first_equidistant_time * speed_A) = abs (-initial_B_position + first_equidistant_time * speed_B)

-- Define the condition for being equidistant at the second time
def second_equidistant_condition : Prop :=
  (second_equidistant_time * speed_A) = abs (-initial_B_position + second_equidistant_time * speed_B)

-- Theorem stating that the ratio of speeds is 1:5
theorem speed_ratio : 
  first_equidistant_condition → second_equidistant_condition → speed_A / speed_B = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_speed_ratio_l3171_317175


namespace NUMINAMATH_CALUDE_intersection_M_N_l3171_317180

open Set

def M : Set ℝ := {x : ℝ | 3 * x - x^2 > 0}
def N : Set ℝ := {x : ℝ | x^2 - 4 * x + 3 > 0}

theorem intersection_M_N : M ∩ N = Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3171_317180


namespace NUMINAMATH_CALUDE_blocks_added_to_tower_l3171_317140

/-- The number of blocks added to a tower -/
def blocks_added (initial final : ℝ) : ℝ := final - initial

/-- Proof that 65.0 blocks were added to the tower -/
theorem blocks_added_to_tower : blocks_added 35.0 100 = 65.0 := by
  sorry

end NUMINAMATH_CALUDE_blocks_added_to_tower_l3171_317140


namespace NUMINAMATH_CALUDE_bennetts_brothers_l3171_317176

theorem bennetts_brothers (aaron_brothers : ℕ) (bennett_brothers : ℕ) : 
  aaron_brothers = 4 → 
  bennett_brothers = 2 * aaron_brothers - 2 → 
  bennett_brothers = 6 := by
sorry

end NUMINAMATH_CALUDE_bennetts_brothers_l3171_317176


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_2024_l3171_317192

theorem units_digit_of_7_to_2024 : ∃ n : ℕ, 7^2024 ≡ 1 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_2024_l3171_317192


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3171_317139

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| - |x - 3| ≥ 0} = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3171_317139


namespace NUMINAMATH_CALUDE_quadratic_value_l3171_317186

/-- A quadratic function with vertex (2,7) passing through (0,-7) -/
def f (x : ℝ) : ℝ :=
  let a : ℝ := -3.5
  a * (x - 2)^2 + 7

theorem quadratic_value : f 5 = -24.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_l3171_317186


namespace NUMINAMATH_CALUDE_line_contains_point_l3171_317199

/-- Given a line equation 2 - kx = -4y that contains the point (2, -1), prove that k = -1 -/
theorem line_contains_point (k : ℝ) : 
  (2 - k * 2 = -4 * (-1)) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l3171_317199


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l3171_317106

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x * y = 36) : 
  x + y ≤ Real.sqrt 202 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l3171_317106


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l3171_317173

-- Define the property of having at most finitely many zeros
def HasFinitelyManyZeros (f : ℝ → ℝ) : Prop :=
  ∃ (S : Finset ℝ), ∀ x, f x = 0 → x ∈ S

-- Define the functional equation
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x^4 + y) = x^3 * f x + f (f y)

-- Theorem statement
theorem unique_function_satisfying_conditions :
  ∃! f : ℝ → ℝ, HasFinitelyManyZeros f ∧ SatisfiesFunctionalEquation f ∧ (∀ x, f x = x) :=
sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l3171_317173


namespace NUMINAMATH_CALUDE_completing_square_result_l3171_317119

theorem completing_square_result (x : ℝ) : 
  x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_completing_square_result_l3171_317119


namespace NUMINAMATH_CALUDE_board_crossing_area_l3171_317196

/-- The area of the parallelogram formed by two boards crossed at a 45-degree angle -/
theorem board_crossing_area (width1 width2 : ℝ) (angle : ℝ) : 
  width1 = 5 → width2 = 6 → angle = π/4 → 
  width2 * width1 = 30 := by sorry

end NUMINAMATH_CALUDE_board_crossing_area_l3171_317196


namespace NUMINAMATH_CALUDE_solve_parking_ticket_problem_l3171_317121

def parking_ticket_problem (first_two_ticket_cost : ℚ) (third_ticket_fraction : ℚ) (james_remaining_money : ℚ) : Prop :=
  let total_cost := 2 * first_two_ticket_cost + third_ticket_fraction * first_two_ticket_cost
  let james_paid := total_cost - james_remaining_money
  let roommate_paid := total_cost - james_paid
  (roommate_paid / total_cost) = 13 / 14

theorem solve_parking_ticket_problem :
  parking_ticket_problem 150 (1/3) 325 := by
  sorry

end NUMINAMATH_CALUDE_solve_parking_ticket_problem_l3171_317121


namespace NUMINAMATH_CALUDE_sqrt_88200_simplification_l3171_317167

theorem sqrt_88200_simplification : Real.sqrt 88200 = 70 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_88200_simplification_l3171_317167


namespace NUMINAMATH_CALUDE_f_symmetry_l3171_317137

/-- A function f(x) = x^5 + ax^3 + bx - 8 for some real a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

/-- Theorem: If f(-2) = 10, then f(2) = -26 -/
theorem f_symmetry (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l3171_317137


namespace NUMINAMATH_CALUDE_sin_cos_derivative_l3171_317155

theorem sin_cos_derivative (x : ℝ) : 
  deriv (λ x => Real.sin x * Real.cos x) x = Real.cos x ^ 2 - Real.sin x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_derivative_l3171_317155


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3171_317125

theorem polynomial_simplification (x : ℝ) : 
  (12 * x^10 - 3 * x^9 + 8 * x^8 - 5 * x^7) - 
  (2 * x^10 + 2 * x^9 - x^8 + x^7 + 4 * x^4 + 6 * x^2 + 9) = 
  10 * x^10 - 5 * x^9 + 9 * x^8 - 6 * x^7 - 4 * x^4 - 6 * x^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3171_317125


namespace NUMINAMATH_CALUDE_sara_letters_problem_l3171_317115

theorem sara_letters_problem (january february march total : ℕ) :
  february = 9 →
  march = 3 * january →
  total = january + february + march →
  total = 33 →
  january = 6 :=
by sorry

end NUMINAMATH_CALUDE_sara_letters_problem_l3171_317115


namespace NUMINAMATH_CALUDE_solve_for_a_l3171_317177

theorem solve_for_a (x y a : ℝ) : 
  x + y = 1 → 
  2 * x + y = 0 → 
  a * x - 3 * y = 0 → 
  a = -6 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l3171_317177


namespace NUMINAMATH_CALUDE_matrix_op_example_l3171_317143

/-- Definition of the operation for 2x2 matrices -/
def matrix_op (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating that the operation on the given matrix equals -9 -/
theorem matrix_op_example : matrix_op (-2) 0.5 2 4 = -9 := by
  sorry

end NUMINAMATH_CALUDE_matrix_op_example_l3171_317143


namespace NUMINAMATH_CALUDE_square_of_product_plus_one_l3171_317103

theorem square_of_product_plus_one :
  24 * 25 * 26 * 27 + 1 = (24^2 + 3 * 24 + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_plus_one_l3171_317103


namespace NUMINAMATH_CALUDE_solution_set_min_value_min_value_ab_equality_condition_l3171_317128

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Part 1: Solution set of f(x) ≤ 4
theorem solution_set (x : ℝ) : f x ≤ 4 ↔ 0 ≤ x ∧ x ≤ 4/3 := by sorry

-- Part 2: Minimum value of f(x)
theorem min_value : ∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y ∧ f x = 3 := by sorry

-- Part 3: Minimum value of 1/(a-1) + 2/b given conditions
theorem min_value_ab (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + 2*b = 3) :
  1/(a-1) + 2/b ≥ 9/2 := by sorry

-- Part 4: Equality condition for the minimum value
theorem equality_condition (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + 2*b = 3) :
  1/(a-1) + 2/b = 9/2 ↔ a = 5/3 ∧ b = 2/3 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_min_value_ab_equality_condition_l3171_317128


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3171_317189

/-- Given a sale price including tax, sales tax rate, and profit rate,
    calculate the approximate cost price of an article. -/
theorem cost_price_calculation (sale_price_with_tax : ℝ)
                                (sales_tax_rate : ℝ)
                                (profit_rate : ℝ)
                                (h1 : sale_price_with_tax = 616)
                                (h2 : sales_tax_rate = 0.1)
                                (h3 : profit_rate = 0.17) :
  ∃ (cost_price : ℝ), 
    (cost_price * (1 + profit_rate) * (1 + sales_tax_rate) = sale_price_with_tax) ∧
    (abs (cost_price - 478.77) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3171_317189


namespace NUMINAMATH_CALUDE_leadership_selection_count_l3171_317159

/-- The number of ways to choose a president, vice president, and a 3-person committee from a group of people. -/
def choose_leadership (total : ℕ) (males : ℕ) (females : ℕ) : ℕ :=
  let remaining := total - 2  -- After choosing president and vice president
  let committee_choices := 
    (males.choose 1 * females.choose 2) +  -- 1 male and 2 females
    (males.choose 2 * females.choose 1)    -- 2 males and 1 female
  (total * (total - 1)) * committee_choices

/-- The theorem stating the number of ways to choose leadership positions from a specific group. -/
theorem leadership_selection_count : 
  choose_leadership 10 6 4 = 8640 := by
  sorry


end NUMINAMATH_CALUDE_leadership_selection_count_l3171_317159


namespace NUMINAMATH_CALUDE_arithmetic_sequence_before_one_l3171_317181

theorem arithmetic_sequence_before_one (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 100 → d = -7 → n = 15 →
  a₁ + (n - 1) * d = 1 ∧ n - 1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_before_one_l3171_317181


namespace NUMINAMATH_CALUDE_roots_nature_l3171_317152

/-- The quadratic equation x^2 + 2x + m = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 + 2*x + m = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  4 - 4*m

/-- The nature of the roots is determined by the value of m -/
theorem roots_nature (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x m ∧ quadratic_equation y m) ∨
  (∃ x : ℝ, quadratic_equation x m ∧ ∀ y : ℝ, quadratic_equation y m → y = x) ∨
  (∀ x : ℝ, ¬quadratic_equation x m) :=
sorry

end NUMINAMATH_CALUDE_roots_nature_l3171_317152


namespace NUMINAMATH_CALUDE_tens_digit_of_9_to_1503_l3171_317157

theorem tens_digit_of_9_to_1503 : ∃ n : ℕ, n ≥ 0 ∧ n < 10 ∧ 9^1503 ≡ 20 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_9_to_1503_l3171_317157


namespace NUMINAMATH_CALUDE_count_polygons_l3171_317179

/-- The number of distinct convex polygons with 3 or more sides
    that can be drawn from 12 points on a circle -/
def num_polygons : ℕ := 4017

/-- The number of points marked on the circle -/
def num_points : ℕ := 12

/-- Theorem stating that the number of distinct convex polygons
    with 3 or more sides drawn from 12 points on a circle is 4017 -/
theorem count_polygons :
  (2^num_points : ℕ) - (Nat.choose num_points 0) - (Nat.choose num_points 1) - (Nat.choose num_points 2) = num_polygons :=
by sorry

end NUMINAMATH_CALUDE_count_polygons_l3171_317179


namespace NUMINAMATH_CALUDE_octahedron_sphere_probability_l3171_317124

/-- Represents a regular octahedron with inscribed and circumscribed spheres -/
structure OctahedronWithSpheres where
  /-- Radius of the circumscribed sphere -/
  R : ℝ
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- Assumption that the inscribed sphere radius is one-third of the circumscribed sphere radius -/
  h_r_eq : r = R / 3

/-- The probability that a randomly chosen point in the circumscribed sphere
    lies inside one of the nine smaller spheres (one inscribed and eight tangent to faces) -/
theorem octahedron_sphere_probability (o : OctahedronWithSpheres) :
  (volume_ratio : ℝ) = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_octahedron_sphere_probability_l3171_317124


namespace NUMINAMATH_CALUDE_hilt_family_fitness_l3171_317142

-- Define conversion rates
def yards_per_mile : ℝ := 1760
def miles_per_km : ℝ := 0.621371

-- Define Mrs. Hilt's activities
def mrs_hilt_running : List ℝ := [3, 2, 7]
def mrs_hilt_swimming : List ℝ := [1760, 0, 1000]
def mrs_hilt_biking : List ℝ := [0, 6, 3, 10]

-- Define Mr. Hilt's activities
def mr_hilt_biking : List ℝ := [5, 8]
def mr_hilt_running : List ℝ := [4]
def mr_hilt_swimming : List ℝ := [2000]

-- Theorem statement
theorem hilt_family_fitness :
  (mrs_hilt_running.sum = 12) ∧
  (mrs_hilt_swimming.sum / yards_per_mile + 1000 / yards_per_mile * miles_per_km = 2854 / yards_per_mile) ∧
  (mrs_hilt_biking.sum = 19) ∧
  (mr_hilt_biking.sum = 13) ∧
  (mr_hilt_running.sum = 4) ∧
  (mr_hilt_swimming.sum = 2000) :=
by sorry

end NUMINAMATH_CALUDE_hilt_family_fitness_l3171_317142


namespace NUMINAMATH_CALUDE_egg_distribution_l3171_317194

theorem egg_distribution (total_eggs : Nat) (num_students : Nat) 
  (h1 : total_eggs = 73) (h2 : num_students = 9) :
  ∃ (eggs_per_student : Nat) (leftover : Nat),
    total_eggs = num_students * eggs_per_student + leftover ∧
    eggs_per_student = 8 ∧
    leftover = 1 := by
  sorry

end NUMINAMATH_CALUDE_egg_distribution_l3171_317194


namespace NUMINAMATH_CALUDE_intersection_perpendicular_implies_m_l3171_317101

-- Define the curve C
def curve (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1*x2 + y1*y2 = 0

theorem intersection_perpendicular_implies_m (m : ℝ) :
  ∃ x1 y1 x2 y2 : ℝ,
    curve x1 y1 m ∧ curve x2 y2 m ∧
    line x1 y1 ∧ line x2 y2 ∧
    perpendicular x1 y1 x2 y2 →
  m = 8/5 := by sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_implies_m_l3171_317101


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3171_317132

def f (x : ℝ) : ℝ := x^2 - 3*x + 4

def g (x m : ℝ) : ℝ := 2*x + m

def h (x t : ℝ) : ℝ := f x - (2*t - 3)*x

def F (x m : ℝ) : ℝ := f x - g x m

theorem quadratic_function_properties :
  (f 0 = 4) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 4 ∧ f x₁ = 2*x₁ ∧ f x₂ = 2*x₂) ∧
  (∃ t : ℝ, t = Real.sqrt 2 / 2 ∧ 
    (∀ x : ℝ, x ∈ Set.Icc 0 1 → h x t ≥ 7/2) ∧
    (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ h x t = 7/2)) ∧
  (∀ m : ℝ, m ∈ Set.Ioo (-9/4) (-2) →
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 3 ∧ x₂ ∈ Set.Icc 0 3 ∧ 
      F x₁ m = 0 ∧ F x₂ m = 0)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3171_317132


namespace NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l3171_317127

/-- Represents the number of male athletes to be drawn in a stratified sampling -/
def male_athletes_drawn (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ) : ℚ :=
  (male_athletes : ℚ) * (sample_size : ℚ) / (total_athletes : ℚ)

/-- Theorem stating that in the given scenario, 4 male athletes should be drawn -/
theorem stratified_sampling_male_athletes :
  male_athletes_drawn 30 20 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l3171_317127


namespace NUMINAMATH_CALUDE_sasha_train_journey_l3171_317185

/-- Represents a day of the week -/
inductive DayOfWeek
  | Saturday
  | Sunday
  | Monday

/-- Represents the train journey -/
structure TrainJourney where
  departureDay : DayOfWeek
  arrivalDay : DayOfWeek
  journeyDuration : Nat
  departureDateNumber : Nat
  arrivalDateNumber : Nat
  trainCarNumber : Nat
  seatNumber : Nat

/-- The conditions of Sasha's train journey -/
def sashasJourney : TrainJourney :=
  { departureDay := DayOfWeek.Saturday
  , arrivalDay := DayOfWeek.Monday
  , journeyDuration := 50
  , departureDateNumber := 31  -- Assuming end of month
  , arrivalDateNumber := 2     -- Assuming start of next month
  , trainCarNumber := 2
  , seatNumber := 1
  }

theorem sasha_train_journey :
  ∀ (journey : TrainJourney),
    journey.departureDay = DayOfWeek.Saturday →
    journey.arrivalDay = DayOfWeek.Monday →
    journey.journeyDuration = 50 →
    journey.arrivalDateNumber = journey.trainCarNumber →
    journey.seatNumber < journey.trainCarNumber →
    journey.departureDateNumber > journey.trainCarNumber →
    journey.trainCarNumber = 2 ∧ journey.seatNumber = 1 := by
  sorry

#check sasha_train_journey

end NUMINAMATH_CALUDE_sasha_train_journey_l3171_317185


namespace NUMINAMATH_CALUDE_two_face_painted_count_l3171_317156

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  side_length : Nat
  painted_faces : Nat

/-- Counts the number of smaller cubes painted on exactly two faces -/
def count_two_face_painted (c : CutCube) : Nat :=
  if c.side_length = 3 ∧ c.painted_faces = 6 then
    24
  else
    0

theorem two_face_painted_count (c : CutCube) :
  c.side_length = 3 ∧ c.painted_faces = 6 → count_two_face_painted c = 24 := by
  sorry

end NUMINAMATH_CALUDE_two_face_painted_count_l3171_317156
