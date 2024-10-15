import Mathlib

namespace NUMINAMATH_CALUDE_ray_reflection_l3028_302809

/-- Given a point A, a line l, and a point B, prove the equations of the incident and reflected rays --/
theorem ray_reflection (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) : 
  A = (2, 3) → 
  B = (1, 1) → 
  (∀ x y, l x y ↔ x + y + 1 = 0) →
  ∃ (incident reflected : ℝ → ℝ → Prop),
    (∀ x y, incident x y ↔ 9*x - 7*y + 3 = 0) ∧
    (∀ x y, reflected x y ↔ 7*x - 9*y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ray_reflection_l3028_302809


namespace NUMINAMATH_CALUDE_min_value_xyz_l3028_302891

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : (x + y) / (x * y * z) ≥ 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xyz_l3028_302891


namespace NUMINAMATH_CALUDE_student_presentations_periods_class_presentation_periods_l3028_302844

/-- Calculates the number of periods needed for all student presentations --/
theorem student_presentations_periods (total_students : ℕ) (period_length : ℕ) 
  (individual_presentation_time : ℕ) (individual_qa_time : ℕ) 
  (group_presentations : ℕ) (group_presentation_time : ℕ) : ℕ :=
  let individual_students := total_students - group_presentations
  let individual_time := individual_students * (individual_presentation_time + individual_qa_time)
  let group_time := group_presentations * group_presentation_time
  let total_time := individual_time + group_time
  (total_time + period_length - 1) / period_length

/-- The number of periods needed for the given class presentation scenario is 7 --/
theorem class_presentation_periods : 
  student_presentations_periods 32 40 5 3 4 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_student_presentations_periods_class_presentation_periods_l3028_302844


namespace NUMINAMATH_CALUDE_even_and_mono_decreasing_implies_ordering_l3028_302870

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being monotonically decreasing on an interval
def IsMonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- State the theorem
theorem even_and_mono_decreasing_implies_ordering (heven : IsEven f) 
    (hmono : IsMonoDecreasing (fun x ↦ f (x - 2)) 0 2) :
  f 2 < f (-1) ∧ f (-1) < f 0 :=
by sorry

end NUMINAMATH_CALUDE_even_and_mono_decreasing_implies_ordering_l3028_302870


namespace NUMINAMATH_CALUDE_martha_latte_days_l3028_302873

/-- The number of days Martha buys a latte per week -/
def latte_days : ℕ := sorry

/-- The cost of a latte in dollars -/
def latte_cost : ℚ := 4

/-- The cost of an iced coffee in dollars -/
def iced_coffee_cost : ℚ := 2

/-- The number of days Martha buys an iced coffee per week -/
def iced_coffee_days : ℕ := 3

/-- The percentage reduction in annual coffee spending -/
def spending_reduction_percentage : ℚ := 25 / 100

/-- The amount saved in dollars due to spending reduction -/
def amount_saved : ℚ := 338

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

theorem martha_latte_days : 
  latte_days = 5 :=
by sorry

end NUMINAMATH_CALUDE_martha_latte_days_l3028_302873


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l3028_302896

theorem min_value_of_fraction :
  ∀ x : ℝ, (1/2 * x^2 + x + 1 ≠ 0) →
  ((3 * x^2 + 6 * x + 5) / (1/2 * x^2 + x + 1) ≥ 4) ∧
  (∃ y : ℝ, (1/2 * y^2 + y + 1 ≠ 0) ∧ ((3 * y^2 + 6 * y + 5) / (1/2 * y^2 + y + 1) = 4)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l3028_302896


namespace NUMINAMATH_CALUDE_constant_product_rule_l3028_302851

theorem constant_product_rule (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0) :
  a * b = (k * a) * (b / k) :=
by sorry

end NUMINAMATH_CALUDE_constant_product_rule_l3028_302851


namespace NUMINAMATH_CALUDE_infinite_integers_satisfying_inequality_l3028_302850

theorem infinite_integers_satisfying_inequality :
  ∃ (S : Set ℤ), (Set.Infinite S) ∧ 
  (∀ n ∈ S, (Real.sqrt (n + 1 : ℝ) ≤ Real.sqrt (3 * n + 2 : ℝ)) ∧ 
             (Real.sqrt (3 * n + 2 : ℝ) < Real.sqrt (4 * n - 1 : ℝ))) :=
sorry

end NUMINAMATH_CALUDE_infinite_integers_satisfying_inequality_l3028_302850


namespace NUMINAMATH_CALUDE_apple_cost_price_l3028_302802

theorem apple_cost_price (selling_price : ℚ) (loss_fraction : ℚ) : 
  selling_price = 16 → loss_fraction = 1/6 → 
  ∃ cost_price : ℚ, 
    selling_price = cost_price - loss_fraction * cost_price ∧ 
    cost_price = 19.2 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_price_l3028_302802


namespace NUMINAMATH_CALUDE_monotonicity_nonpositive_a_monotonicity_positive_a_f_geq_f_neg_l3028_302876

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

-- Theorem for monotonicity when a ≤ 0
theorem monotonicity_nonpositive_a (a : ℝ) (h : a ≤ 0) :
  StrictMono (f a) := by sorry

-- Theorem for monotonicity when a > 0
theorem monotonicity_positive_a (a : ℝ) (h : a > 0) :
  ∀ x y, x < y → (
    (x < Real.log a ∧ y < Real.log a → f a y < f a x) ∧
    (Real.log a < x ∧ Real.log a < y → f a x < f a y)
  ) := by sorry

-- Theorem for f(x) ≥ f(-x) when a = 1 and x ≥ 0
theorem f_geq_f_neg (x : ℝ) (h : x ≥ 0) :
  f 1 x ≥ f 1 (-x) := by sorry

end

end NUMINAMATH_CALUDE_monotonicity_nonpositive_a_monotonicity_positive_a_f_geq_f_neg_l3028_302876


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficients_l3028_302862

theorem quadratic_root_implies_coefficients
  (a b : ℚ)
  (h : (1 + Real.sqrt 3)^2 + a * (1 + Real.sqrt 3) + b = 0) :
  a = -2 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficients_l3028_302862


namespace NUMINAMATH_CALUDE_distinct_laptop_choices_l3028_302894

/-- The number of ways to choose 3 distinct items from a set of 15 items -/
def choose_distinct (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else (n - k + 1).factorial / (n - k).factorial

theorem distinct_laptop_choices :
  choose_distinct 15 3 = 2730 := by
sorry

end NUMINAMATH_CALUDE_distinct_laptop_choices_l3028_302894


namespace NUMINAMATH_CALUDE_only_3_4_5_is_right_triangle_l3028_302843

/-- A function that checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers --/
def sets : List (ℕ × ℕ × ℕ) :=
  [(1, 2, 2), (3, 4, 5), (3, 4, 9), (4, 5, 7)]

/-- Theorem stating that only (3, 4, 5) forms a right-angled triangle --/
theorem only_3_4_5_is_right_triangle :
  ∃! (a b c : ℕ), (a, b, c) ∈ sets ∧ is_right_triangle a b c :=
by sorry

end NUMINAMATH_CALUDE_only_3_4_5_is_right_triangle_l3028_302843


namespace NUMINAMATH_CALUDE_expression_evaluation_l3028_302884

theorem expression_evaluation : 
  |5 - 8 * (3 - 12)^2| - |5 - 11| + Real.sqrt 16 + Real.sin (π / 2) = 642 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3028_302884


namespace NUMINAMATH_CALUDE_triangle_BC_proof_l3028_302853

def triangle_BC (A B C : ℝ) (tanA : ℝ) (AB : ℝ) : Prop :=
  let angleB := Real.pi / 2
  let BC := ((AB ^ 2) + (tanA * AB) ^ 2).sqrt
  angleB = Real.pi / 2 ∧ 
  tanA = 3 / 7 ∧ 
  AB = 14 → 
  BC = 2 * Real.sqrt 58

theorem triangle_BC_proof : triangle_BC Real.pi Real.pi Real.pi (3/7) 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_BC_proof_l3028_302853


namespace NUMINAMATH_CALUDE_summer_break_difference_l3028_302838

theorem summer_break_difference (camp_kids : ℕ) (home_kids : ℕ) 
  (h1 : camp_kids = 819058) (h2 : home_kids = 668278) : 
  camp_kids - home_kids = 150780 := by
  sorry

end NUMINAMATH_CALUDE_summer_break_difference_l3028_302838


namespace NUMINAMATH_CALUDE_system_solution_existence_and_values_l3028_302822

/-- Given a system of equations with parameters α₁, α₂, α₃, α₄, prove that a solution exists
    if and only if α₁ = α₂ = α₃ = α or α₄ = α, and find the solution in this case. -/
theorem system_solution_existence_and_values (α₁ α₂ α₃ α₄ : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    x₁ + x₂ = α₁ * α₂ ∧
    x₁ + x₃ = α₁ * α₃ ∧
    x₁ + x₄ = α₁ * α₄ ∧
    x₂ + x₃ = α₂ * α₃ ∧
    x₂ + x₄ = α₂ * α₄ ∧
    x₃ + x₄ = α₃ * α₄) ↔
  ((α₁ = α₂ ∧ α₂ = α₃) ∨ α₄ = α₂) ∧
  (∃ α β : ℝ,
    (α = α₁ ∧ β = α₄) ∨ (α = α₂ ∧ β = α₁) ∧
    x₁ = α^2 / 2 ∧
    x₂ = α^2 / 2 ∧
    x₃ = α^2 / 2 ∧
    x₄ = α * (β - α / 2)) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_existence_and_values_l3028_302822


namespace NUMINAMATH_CALUDE_triangle_is_obtuse_l3028_302888

theorem triangle_is_obtuse (a b c : ℝ) (ha : a = 4) (hb : b = 6) (hc : c = 8) :
  a^2 + b^2 < c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_obtuse_l3028_302888


namespace NUMINAMATH_CALUDE_min_value_product_quotient_min_value_achieved_l3028_302836

theorem min_value_product_quotient (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x*y*z) ≥ 343 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a*b*c) = 343 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_min_value_achieved_l3028_302836


namespace NUMINAMATH_CALUDE_line_circle_separation_l3028_302849

/-- If a point (a,b) is inside the unit circle, then the line ax + by = 1 is separated from the circle -/
theorem line_circle_separation (a b : ℝ) (h : a^2 + b^2 < 1) :
  ∃ (d : ℝ), d > 1 ∧ d = 1 / Real.sqrt (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_separation_l3028_302849


namespace NUMINAMATH_CALUDE_at_most_one_acute_forming_point_l3028_302829

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A function to check if a triangle is acute-angled -/
def isAcuteTriangle (p q r : Point) : Prop :=
  sorry -- Definition of acute triangle

/-- The theorem stating that at most one point can form acute triangles with any other two points -/
theorem at_most_one_acute_forming_point (points : Finset Point) (h : points.card = 2006) :
  ∃ (p : Point), p ∈ points ∧
    (∀ (q r : Point), q ∈ points → r ∈ points → q ≠ r → q ≠ p → r ≠ p → isAcuteTriangle p q r) →
    ∀ (p' : Point), p' ∈ points → p' ≠ p →
      ∃ (q r : Point), q ∈ points ∧ r ∈ points ∧ q ≠ r ∧ q ≠ p' ∧ r ≠ p' ∧ ¬isAcuteTriangle p' q r :=
by
  sorry

end NUMINAMATH_CALUDE_at_most_one_acute_forming_point_l3028_302829


namespace NUMINAMATH_CALUDE_inequality_proof_l3028_302889

theorem inequality_proof (a : ℝ) (h : a > 1) : (1/2 : ℝ) + (1 / Real.log a) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3028_302889


namespace NUMINAMATH_CALUDE_min_points_all_but_one_hemisphere_l3028_302842

/-- A point on the surface of a sphere -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ
  on_sphere : x^2 + y^2 + z^2 = 1

/-- A hemisphere of a sphere -/
def Hemisphere := Set Point3D

/-- The set of all possible hemispheres of a sphere -/
def AllHemispheres : Set Hemisphere := sorry

/-- A set of points is in all hemispheres except one if it intersects with all but one hemisphere -/
def InAllButOneHemisphere (points : Set Point3D) : Prop :=
  ∃ h : Hemisphere, h ∈ AllHemispheres ∧ 
    ∀ h' : Hemisphere, h' ∈ AllHemispheres → h' ≠ h → (points ∩ h').Nonempty

theorem min_points_all_but_one_hemisphere :
  ∃ (points : Set Point3D), points.ncard = 4 ∧ InAllButOneHemisphere points ∧
    ∀ (points' : Set Point3D), points'.ncard < 4 → ¬InAllButOneHemisphere points' :=
  sorry

end NUMINAMATH_CALUDE_min_points_all_but_one_hemisphere_l3028_302842


namespace NUMINAMATH_CALUDE_a_range_l3028_302892

theorem a_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0) →
  (∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0) →
  ¬((∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0) ∧ 
    (∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0)) →
  a ∈ Set.union (Set.Ioo (-1) 0) (Set.Ioo 0 1) :=
by sorry

end NUMINAMATH_CALUDE_a_range_l3028_302892


namespace NUMINAMATH_CALUDE_sqrt_900_squared_times_6_l3028_302813

theorem sqrt_900_squared_times_6 : (Real.sqrt 900)^2 * 6 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_900_squared_times_6_l3028_302813


namespace NUMINAMATH_CALUDE_table_rearrangement_l3028_302834

/-- Represents a table with n rows and n columns -/
def Table (α : Type) (n : ℕ) := Fin n → Fin n → α

/-- Predicate to check if a row has no repeated elements -/
def NoRepeatsInRow {α : Type} [DecidableEq α] (row : Fin n → α) : Prop :=
  ∀ i j : Fin n, i ≠ j → row i ≠ row j

/-- Predicate to check if a table has no repeated elements in any row -/
def NoRepeatsInRows {α : Type} [DecidableEq α] (T : Table α n) : Prop :=
  ∀ i : Fin n, NoRepeatsInRow (T i)

/-- Predicate to check if two rows are permutations of each other -/
def RowsArePermutations {α : Type} [DecidableEq α] (row1 row2 : Fin n → α) : Prop :=
  ∀ x : α, (∃ i : Fin n, row1 i = x) ↔ (∃ j : Fin n, row2 j = x)

/-- Predicate to check if a column has no repeated elements -/
def NoRepeatsInColumn {α : Type} [DecidableEq α] (T : Table α n) (j : Fin n) : Prop :=
  ∀ i k : Fin n, i ≠ k → T i j ≠ T k j

/-- The main theorem statement -/
theorem table_rearrangement {α : Type} [DecidableEq α] (n : ℕ) (T : Table α n) 
  (h : NoRepeatsInRows T) :
  ∃ T_star : Table α n,
    (∀ i : Fin n, RowsArePermutations (T i) (T_star i)) ∧
    (∀ j : Fin n, NoRepeatsInColumn T_star j) :=
  sorry

end NUMINAMATH_CALUDE_table_rearrangement_l3028_302834


namespace NUMINAMATH_CALUDE_string_folding_theorem_l3028_302811

/-- The number of layers after folding a string n times -/
def layers (n : ℕ) : ℕ := 2^n

/-- The number of longer strings after folding and cutting -/
def longer_strings (total_layers : ℕ) : ℕ := total_layers - 1

/-- The number of shorter strings after folding and cutting -/
def shorter_strings (total_layers : ℕ) (num_cuts : ℕ) : ℕ :=
  (num_cuts - 2) * total_layers + 2

theorem string_folding_theorem (num_folds num_cuts : ℕ) 
  (h1 : num_folds = 10) (h2 : num_cuts = 10) :
  longer_strings (layers num_folds) = 1023 ∧
  shorter_strings (layers num_folds) num_cuts = 8194 := by
  sorry

#eval longer_strings (layers 10)
#eval shorter_strings (layers 10) 10

end NUMINAMATH_CALUDE_string_folding_theorem_l3028_302811


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_a_range_l3028_302880

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The interval (-∞, 4] -/
def interval : Set ℝ := Set.Iic 4

theorem quadratic_decreasing_implies_a_range (a : ℝ) :
  (∀ x ∈ interval, StrictMonoOn (f a) interval) → a < -5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_a_range_l3028_302880


namespace NUMINAMATH_CALUDE_cylindrical_to_rectangular_conversion_l3028_302824

/-- Conversion from cylindrical coordinates to rectangular coordinates -/
theorem cylindrical_to_rectangular_conversion 
  (r θ z : ℝ) 
  (hr : r = 7) 
  (hθ : θ = π / 3) 
  (hz : z = -3) :
  ∃ (x y : ℝ), 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ ∧ 
    x = 3.5 ∧ 
    y = 7 * Real.sqrt 3 / 2 ∧ 
    z = -3 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_to_rectangular_conversion_l3028_302824


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l3028_302823

theorem quadratic_form_h_value (x : ℝ) :
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 15 = a * (x + 3/2)^2 + k :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l3028_302823


namespace NUMINAMATH_CALUDE_number_of_chords_l3028_302858

/-- The number of points on the circle -/
def n : ℕ := 10

/-- The number of points needed to form a chord -/
def k : ℕ := 2

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Theorem: The number of unique chords formed by selecting any 2 points 
    from 10 equally spaced points on a circle is equal to 45 -/
theorem number_of_chords : binomial n k = 45 := by sorry

end NUMINAMATH_CALUDE_number_of_chords_l3028_302858


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3028_302863

theorem solution_set_of_inequality (x : ℝ) :
  (-x^2 + 2*x + 15 ≥ 0) ↔ (-3 ≤ x ∧ x ≤ 5) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3028_302863


namespace NUMINAMATH_CALUDE_staircase_shape_perimeter_l3028_302800

/-- Represents the shape described in the problem -/
structure StaircaseShape where
  width : ℝ
  height : ℝ
  staircase_sides : ℕ
  area : ℝ

/-- Calculates the perimeter of the StaircaseShape -/
def perimeter (shape : StaircaseShape) : ℝ :=
  shape.width + shape.height + 4 + 5 + (shape.staircase_sides : ℝ)

/-- Theorem stating the perimeter of the specific shape described in the problem -/
theorem staircase_shape_perimeter : 
  ∀ (shape : StaircaseShape), 
    shape.width = 12 ∧ 
    shape.staircase_sides = 10 ∧ 
    shape.area = 72 → 
    perimeter shape = 42.25 := by
  sorry


end NUMINAMATH_CALUDE_staircase_shape_perimeter_l3028_302800


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l3028_302808

theorem quadratic_root_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 2023*x₁ + 1 = 0 → 
  x₂^2 - 2023*x₂ + 1 = 0 → 
  x₁ ≠ x₂ →
  (1/x₁) + (1/x₂) = 2023 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l3028_302808


namespace NUMINAMATH_CALUDE_min_product_with_constraint_min_product_achievable_l3028_302835

theorem min_product_with_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 20 * a * b = 13 * a + 14 * b) : a * b ≥ 1.82 := by
  sorry

theorem min_product_achievable : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  20 * a * b = 13 * a + 14 * b ∧ a * b = 1.82 := by
  sorry

end NUMINAMATH_CALUDE_min_product_with_constraint_min_product_achievable_l3028_302835


namespace NUMINAMATH_CALUDE_not_divisible_by_169_l3028_302803

theorem not_divisible_by_169 (n : ℕ) : ¬(169 ∣ (n^2 + 5*n + 16)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_169_l3028_302803


namespace NUMINAMATH_CALUDE_fencing_calculation_l3028_302867

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_calculation (area : ℝ) (uncovered_side : ℝ) : 
  area = 600 → uncovered_side = 30 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    2 * width + uncovered_side = 70 :=
by sorry

end NUMINAMATH_CALUDE_fencing_calculation_l3028_302867


namespace NUMINAMATH_CALUDE_ipod_original_price_l3028_302841

theorem ipod_original_price (discount_percent : ℝ) (final_price : ℝ) (original_price : ℝ) : 
  discount_percent = 35 →
  final_price = 83.2 →
  final_price = original_price * (1 - discount_percent / 100) →
  original_price = 128 := by
sorry

end NUMINAMATH_CALUDE_ipod_original_price_l3028_302841


namespace NUMINAMATH_CALUDE_eraser_cost_proof_l3028_302866

/-- The cost of an eraser in dollars -/
def eraser_cost : ℚ := 2

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := 4

/-- The number of pencils sold -/
def pencils_sold : ℕ := 20

/-- The total revenue in dollars -/
def total_revenue : ℚ := 80

/-- The ratio of erasers to pencils sold -/
def eraser_pencil_ratio : ℕ := 2

theorem eraser_cost_proof :
  eraser_cost = 2 ∧
  eraser_cost = pencil_cost / 2 ∧
  pencils_sold * pencil_cost = total_revenue ∧
  pencils_sold * eraser_pencil_ratio * eraser_cost = total_revenue / 2 :=
by sorry

end NUMINAMATH_CALUDE_eraser_cost_proof_l3028_302866


namespace NUMINAMATH_CALUDE_eliza_ironed_17_pieces_l3028_302865

/-- Calculates the total number of clothes Eliza ironed given the time spent on blouses and dresses --/
def total_clothes_ironed (blouse_time : ℕ) (dress_time : ℕ) (blouse_hours : ℕ) (dress_hours : ℕ) : ℕ :=
  let blouses := (blouse_hours * 60) / blouse_time
  let dresses := (dress_hours * 60) / dress_time
  blouses + dresses

/-- Theorem stating that Eliza ironed 17 pieces of clothes --/
theorem eliza_ironed_17_pieces :
  total_clothes_ironed 15 20 2 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_eliza_ironed_17_pieces_l3028_302865


namespace NUMINAMATH_CALUDE_farmers_extra_days_l3028_302804

/-- A farmer's ploughing problem -/
theorem farmers_extra_days
  (total_area : ℕ)
  (planned_daily_area : ℕ)
  (actual_daily_area : ℕ)
  (area_left : ℕ)
  (h1 : total_area = 720)
  (h2 : planned_daily_area = 120)
  (h3 : actual_daily_area = 85)
  (h4 : area_left = 40) :
  ∃ extra_days : ℕ,
    actual_daily_area * (total_area / planned_daily_area + extra_days) = total_area - area_left ∧
    extra_days = 2 := by
  sorry

#check farmers_extra_days

end NUMINAMATH_CALUDE_farmers_extra_days_l3028_302804


namespace NUMINAMATH_CALUDE_equal_volume_equal_capacity_container2_capacity_l3028_302857

/-- Represents a rectangular container -/
structure Container where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℝ

/-- Calculates the volume of a container -/
def volume (c : Container) : ℝ := c.height * c.width * c.length

/-- Theorem: Two containers with the same volume have the same capacity -/
theorem equal_volume_equal_capacity (c1 c2 : Container) 
  (h_volume : volume c1 = volume c2) 
  (h_capacity : c1.capacity = 80) : 
  c2.capacity = 80 := by
  sorry

/-- The first container -/
def container1 : Container := {
  height := 2,
  width := 3,
  length := 10,
  capacity := 80
}

/-- The second container -/
def container2 : Container := {
  height := 1,
  width := 3,
  length := 20,
  capacity := 80  -- We'll prove this
}

/-- Proof that container2 can hold 80 grams -/
theorem container2_capacity : container2.capacity = 80 := by
  apply equal_volume_equal_capacity container1 container2
  · -- Prove that volumes are equal
    simp [volume, container1, container2]
    -- 2 * 3 * 10 = 1 * 3 * 20
    ring
  · -- Show that container1's capacity is 80
    rfl

#check container2_capacity

end NUMINAMATH_CALUDE_equal_volume_equal_capacity_container2_capacity_l3028_302857


namespace NUMINAMATH_CALUDE_range_of_m_l3028_302825

def f (x : ℝ) : ℝ := x^2 - 4*x - 2

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-6) (-2)) ∧
  (∀ y ∈ Set.Icc (-6) (-2), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3028_302825


namespace NUMINAMATH_CALUDE_number_division_problem_l3028_302885

theorem number_division_problem : ∃ x : ℚ, x / 5 = 75 + x / 6 ∧ x = 2250 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l3028_302885


namespace NUMINAMATH_CALUDE_unique_prime_squared_plus_fourteen_prime_l3028_302879

theorem unique_prime_squared_plus_fourteen_prime :
  ∀ p : ℕ, Prime p → Prime (p^2 + 14) → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_squared_plus_fourteen_prime_l3028_302879


namespace NUMINAMATH_CALUDE_conditional_inequality_1_conditional_inequality_2_conditional_log_inequality_conditional_reciprocal_inequality_l3028_302806

-- Statement 1
theorem conditional_inequality_1 (a b c : ℝ) (h1 : a > b) (h2 : c ≤ 0) :
  a * c ≤ b * c := by sorry

-- Statement 2
theorem conditional_inequality_2 (a b c : ℝ) (h1 : a * c^2 > b * c^2) (h2 : b ≥ 0) :
  a^2 > b^2 := by sorry

-- Statement 3
theorem conditional_log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > -1) :
  Real.log (a + 1) > Real.log (b + 1) := by sorry

-- Statement 4
theorem conditional_reciprocal_inequality (a b : ℝ) (h1 : a > b) (h2 : a * b > 0) :
  1 / a < 1 / b := by sorry

end NUMINAMATH_CALUDE_conditional_inequality_1_conditional_inequality_2_conditional_log_inequality_conditional_reciprocal_inequality_l3028_302806


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3028_302860

theorem quadratic_root_in_unit_interval (a b c : ℝ) 
  (h : 2*a + 3*b + 6*c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3028_302860


namespace NUMINAMATH_CALUDE_f_deriv_positive_at_midpoint_l3028_302821

noncomputable section

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + Real.log x

-- Define the derivative of f
def f_deriv (x : ℝ) : ℝ := 2*x + 1/x

-- Theorem statement
theorem f_deriv_positive_at_midpoint 
  (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) 
  (h₂ : x₁ < x₂) 
  (h₃ : f x₁ = 0) 
  (h₄ : f x₂ = 0) :
  let x₀ := (x₁ + x₂) / 2
  f_deriv x₀ > 0 := by
sorry

end

end NUMINAMATH_CALUDE_f_deriv_positive_at_midpoint_l3028_302821


namespace NUMINAMATH_CALUDE_women_picnic_attendance_l3028_302871

/-- Represents the percentage of employees in a company -/
structure CompanyPercentage where
  total : Real
  men : Real
  women : Real
  menAttended : Real
  womenAttended : Real
  totalAttended : Real

/-- Conditions for the company picnic attendance problem -/
def picnicConditions (c : CompanyPercentage) : Prop :=
  c.total = 100 ∧
  c.men = 50 ∧
  c.women = 50 ∧
  c.menAttended = 20 * c.men / 100 ∧
  c.totalAttended = 30.000000000000004 ∧
  c.womenAttended = c.totalAttended - c.menAttended

/-- Theorem stating that 40% of women attended the picnic -/
theorem women_picnic_attendance (c : CompanyPercentage) 
  (h : picnicConditions c) : c.womenAttended / c.women * 100 = 40 := by
  sorry


end NUMINAMATH_CALUDE_women_picnic_attendance_l3028_302871


namespace NUMINAMATH_CALUDE_prob_same_fee_prob_sum_fee_4_prob_sum_fee_6_l3028_302868

/-- Represents the rental time bracket for a bike rental -/
inductive RentalTime
  | WithinTwo
  | TwoToThree
  | ThreeToFour

/-- Calculates the rental fee based on the rental time -/
def rentalFee (time : RentalTime) : ℕ :=
  match time with
  | RentalTime.WithinTwo => 0
  | RentalTime.TwoToThree => 2
  | RentalTime.ThreeToFour => 4

/-- Represents the probability distribution for a person's rental time -/
structure RentalDistribution where
  withinTwo : ℚ
  twoToThree : ℚ
  threeToFour : ℚ
  sum_to_one : withinTwo + twoToThree + threeToFour = 1

/-- The rental distribution for person A -/
def distA : RentalDistribution :=
  { withinTwo := 1/4
  , twoToThree := 1/2
  , threeToFour := 1/4
  , sum_to_one := by norm_num }

/-- The rental distribution for person B -/
def distB : RentalDistribution :=
  { withinTwo := 1/2
  , twoToThree := 1/4
  , threeToFour := 1/4
  , sum_to_one := by norm_num }

/-- Theorem stating the probability that A and B pay the same fee -/
theorem prob_same_fee : 
  distA.withinTwo * distB.withinTwo + 
  distA.twoToThree * distB.twoToThree + 
  distA.threeToFour * distB.threeToFour = 5/16 := by sorry

/-- Theorem stating the probability that the sum of fees is 4 -/
theorem prob_sum_fee_4 :
  distA.withinTwo * distB.threeToFour + 
  distB.withinTwo * distA.threeToFour + 
  distA.twoToThree * distB.twoToThree = 5/16 := by sorry

/-- Theorem stating the probability that the sum of fees is 6 -/
theorem prob_sum_fee_6 :
  distA.twoToThree * distB.threeToFour + 
  distB.twoToThree * distA.threeToFour = 3/16 := by sorry

end NUMINAMATH_CALUDE_prob_same_fee_prob_sum_fee_4_prob_sum_fee_6_l3028_302868


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3028_302801

theorem arithmetic_mean_of_fractions : 
  (3/8 + 5/9) / 2 = 67/144 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3028_302801


namespace NUMINAMATH_CALUDE_branch_A_more_profitable_l3028_302872

/-- Represents a branch of the factory -/
inductive Branch
| A
| B

/-- Represents a grade of the product -/
inductive Grade
| A
| B
| C
| D

/-- Returns the processing fee for a given grade -/
def processingFee (g : Grade) : Int :=
  match g with
  | Grade.A => 90
  | Grade.B => 50
  | Grade.C => 20
  | Grade.D => -50

/-- Returns the processing cost for a given branch -/
def processingCost (b : Branch) : Int :=
  match b with
  | Branch.A => 25
  | Branch.B => 20

/-- Returns the frequency of a grade for a given branch -/
def frequency (b : Branch) (g : Grade) : Rat :=
  match b, g with
  | Branch.A, Grade.A => 40 / 100
  | Branch.A, Grade.B => 20 / 100
  | Branch.A, Grade.C => 20 / 100
  | Branch.A, Grade.D => 20 / 100
  | Branch.B, Grade.A => 28 / 100
  | Branch.B, Grade.B => 17 / 100
  | Branch.B, Grade.C => 34 / 100
  | Branch.B, Grade.D => 21 / 100

/-- Calculates the average profit for a given branch -/
def averageProfit (b : Branch) : Rat :=
  (processingFee Grade.A - processingCost b) * frequency b Grade.A +
  (processingFee Grade.B - processingCost b) * frequency b Grade.B +
  (processingFee Grade.C - processingCost b) * frequency b Grade.C +
  (processingFee Grade.D - processingCost b) * frequency b Grade.D

/-- Theorem stating that Branch A has higher average profit than Branch B -/
theorem branch_A_more_profitable : averageProfit Branch.A > averageProfit Branch.B := by
  sorry


end NUMINAMATH_CALUDE_branch_A_more_profitable_l3028_302872


namespace NUMINAMATH_CALUDE_perfect_square_difference_l3028_302832

theorem perfect_square_difference (a b c : ℕ) 
  (h1 : Nat.gcd a (Nat.gcd b c) = 1)
  (h2 : a * b = c * (a - b)) : 
  ∃ (k : ℕ), a - b = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_difference_l3028_302832


namespace NUMINAMATH_CALUDE_birds_and_storks_on_fence_l3028_302852

theorem birds_and_storks_on_fence (initial_birds : ℕ) (initial_storks : ℕ) (additional_storks : ℕ) : 
  initial_birds = 3 → initial_storks = 4 → additional_storks = 6 →
  initial_birds + initial_storks + additional_storks = 13 := by
sorry

end NUMINAMATH_CALUDE_birds_and_storks_on_fence_l3028_302852


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3028_302846

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (x^2 + 5*x - 6) / (x^3 - x) = 6 / x + (-5*x + 5) / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3028_302846


namespace NUMINAMATH_CALUDE_mean_of_six_numbers_with_sum_one_third_l3028_302845

theorem mean_of_six_numbers_with_sum_one_third :
  ∀ (a b c d e f : ℚ),
  a + b + c + d + e + f = 1/3 →
  (a + b + c + d + e + f) / 6 = 1/18 := by
sorry

end NUMINAMATH_CALUDE_mean_of_six_numbers_with_sum_one_third_l3028_302845


namespace NUMINAMATH_CALUDE_a₃_eq_10_l3028_302883

/-- The coefficient a₃ in the expansion of x^5 as a polynomial in (1+x) -/
def a₃ : ℝ := 10

/-- The function f(x) = x^5 -/
def f (x : ℝ) : ℝ := x^5

/-- The expansion of f(x) in terms of (1+x) -/
def f_expansion (x a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : ℝ :=
  a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5

/-- Theorem stating that a₃ = 10 in the expansion of x^5 -/
theorem a₃_eq_10 :
  ∀ x a₀ a₁ a₂ a₄ a₅ : ℝ, f x = f_expansion x a₀ a₁ a₂ a₃ a₄ a₅ → a₃ = 10 :=
by sorry

end NUMINAMATH_CALUDE_a₃_eq_10_l3028_302883


namespace NUMINAMATH_CALUDE_negation_equivalence_l3028_302877

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3028_302877


namespace NUMINAMATH_CALUDE_max_value_sqrt_x_over_x_plus_one_l3028_302897

theorem max_value_sqrt_x_over_x_plus_one :
  (∃ x : ℝ, x ≥ 0 ∧ Real.sqrt x / (x + 1) = 1/2) ∧
  (∀ x : ℝ, x ≥ 0 → Real.sqrt x / (x + 1) ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_x_over_x_plus_one_l3028_302897


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3028_302839

theorem election_votes_theorem (total_votes : ℕ) 
  (h1 : ∃ (candidate_votes : ℕ), candidate_votes = (30 * total_votes) / 100)
  (h2 : ∃ (rival_votes : ℕ), rival_votes = (70 * total_votes) / 100)
  (h3 : ∃ (candidate_votes rival_votes : ℕ), rival_votes = candidate_votes + 4000) :
  total_votes = 10000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3028_302839


namespace NUMINAMATH_CALUDE_complement_of_28_45_l3028_302882

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the complement of an angle
def complement (α : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (α.degrees * 60 + α.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- State the theorem
theorem complement_of_28_45 :
  let α : Angle := ⟨28, 45⟩
  complement α = ⟨61, 15⟩ := by
  sorry


end NUMINAMATH_CALUDE_complement_of_28_45_l3028_302882


namespace NUMINAMATH_CALUDE_evaluate_expression_l3028_302886

theorem evaluate_expression : 3 * (-5) ^ (2 ^ (3/4)) = -15 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3028_302886


namespace NUMINAMATH_CALUDE_chris_pennies_l3028_302856

theorem chris_pennies (a c : ℕ) : 
  (c + 2 = 4 * (a - 2)) → 
  (c - 2 = 3 * (a + 2)) → 
  c = 62 := by
sorry

end NUMINAMATH_CALUDE_chris_pennies_l3028_302856


namespace NUMINAMATH_CALUDE_complete_square_constant_l3028_302805

theorem complete_square_constant (a h k : ℚ) : 
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_constant_l3028_302805


namespace NUMINAMATH_CALUDE_greatest_power_of_three_l3028_302893

def p : ℕ := (List.range 34).foldl (· * ·) 1

theorem greatest_power_of_three (k : ℕ) : k ≤ 16 ↔ (3^k : ℕ) ∣ p := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_l3028_302893


namespace NUMINAMATH_CALUDE_min_S_and_max_m_l3028_302887

-- Define the function S
def S (x : ℝ) : ℝ := |x - 2| + |x - 4|

-- State the theorem
theorem min_S_and_max_m :
  (∃ (min_S : ℝ), ∀ x : ℝ, S x ≥ min_S ∧ ∃ x₀ : ℝ, S x₀ = min_S) ∧
  (∃ (max_m : ℝ), (∀ x y : ℝ, S x ≥ max_m * (-y^2 + 2*y)) ∧
    ∀ m : ℝ, (∀ x y : ℝ, S x ≥ m * (-y^2 + 2*y)) → m ≤ max_m) ∧
  (∀ x : ℝ, S x ≥ 2) ∧
  (∀ x y : ℝ, S x ≥ 2 * (-y^2 + 2*y)) :=
by sorry

end NUMINAMATH_CALUDE_min_S_and_max_m_l3028_302887


namespace NUMINAMATH_CALUDE_ellipse_line_slope_product_l3028_302899

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/2
  h_point : 4/a^2 + 2/b^2 = 1

/-- A line not passing through origin and not parallel to axes -/
structure Line where
  k : ℝ
  b : ℝ
  h_k_nonzero : k ≠ 0
  h_b_nonzero : b ≠ 0

/-- The theorem statement -/
theorem ellipse_line_slope_product (C : Ellipse) (l : Line) : 
  ∃ (A B M : ℝ × ℝ), 
    (A.1^2 / C.a^2 + A.2^2 / C.b^2 = 1) ∧ 
    (B.1^2 / C.a^2 + B.2^2 / C.b^2 = 1) ∧
    (A.2 = l.k * A.1 + l.b) ∧ 
    (B.2 = l.k * B.1 + l.b) ∧
    (M = ((A.1 + B.1)/2, (A.2 + B.2)/2)) →
    (M.2 / M.1) * l.k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_product_l3028_302899


namespace NUMINAMATH_CALUDE_eccentricity_range_lower_bound_l3028_302855

/-- The common foci of an ellipse and a hyperbola -/
structure CommonFoci :=
  (F₁ F₂ : ℝ × ℝ)

/-- An ellipse with equation x²/a² + y²/b² = 1 -/
structure Ellipse :=
  (a b : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_gt_b : a > b)

/-- A hyperbola with equation x²/m² - y²/n² = 1 -/
structure Hyperbola :=
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)

/-- A point in the first quadrant -/
structure FirstQuadrantPoint :=
  (P : ℝ × ℝ)
  (h_x_pos : P.1 > 0)
  (h_y_pos : P.2 > 0)

/-- The main theorem -/
theorem eccentricity_range_lower_bound
  (cf : CommonFoci)
  (e : Ellipse)
  (h : Hyperbola)
  (P : FirstQuadrantPoint)
  (h_common_point : P.P ∈ {x : ℝ × ℝ | x.1^2 / e.a^2 + x.2^2 / e.b^2 = 1} ∩
                            {x : ℝ × ℝ | x.1^2 / h.m^2 - x.2^2 / h.n^2 = 1})
  (h_orthogonal : (cf.F₂.1 - P.P.1, cf.F₂.2 - P.P.2) • (P.P.1 - cf.F₁.1, P.P.2 - cf.F₁.2) +
                  (cf.F₂.1 - cf.F₁.1, cf.F₂.2 - cf.F₁.2) • (P.P.1 - cf.F₁.1, P.P.2 - cf.F₁.2) = 0)
  (e₁ : ℝ)
  (h_e₁ : e₁ = Real.sqrt (1 - e.b^2 / e.a^2))
  (e₂ : ℝ)
  (h_e₂ : e₂ = Real.sqrt (1 + h.n^2 / h.m^2)) :
  (4 + e₁ * e₂) / (2 * e₁) ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_eccentricity_range_lower_bound_l3028_302855


namespace NUMINAMATH_CALUDE_tan_value_from_trig_equation_l3028_302816

theorem tan_value_from_trig_equation (x : Real) 
  (h1 : 0 < x ∧ x < π/2) 
  (h2 : (Real.sin x)^4 / 9 + (Real.cos x)^4 / 4 = 1/13) : 
  Real.tan x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_trig_equation_l3028_302816


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l3028_302830

theorem arithmetic_sequence_squares (k : ℤ) : 
  (∃ a d : ℚ, (36 + k : ℚ) = (a - d)^2 ∧ 
               (300 + k : ℚ) = a^2 ∧ 
               (596 + k : ℚ) = (a + d)^2) ↔ 
  k = 925 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l3028_302830


namespace NUMINAMATH_CALUDE_river_width_l3028_302828

/-- A configuration of points for measuring river width -/
structure RiverMeasurement where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  AC_eq_40 : dist A C = 40
  CD_eq_12 : dist C D = 12
  AE_eq_24 : dist A E = 24
  EC_eq_16 : dist E C = 16
  AB_perp_CD : ((B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) : ℝ) = 0
  E_on_AB : ∃ t : ℝ, E = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

/-- The width of the river is 18 meters -/
theorem river_width (m : RiverMeasurement) : dist m.A m.B = 18 := by
  sorry

end NUMINAMATH_CALUDE_river_width_l3028_302828


namespace NUMINAMATH_CALUDE_only_set2_forms_triangle_l3028_302815

-- Define a structure for a set of three line segments
structure TripleSegment where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle inequality theorem
def satisfiesTriangleInequality (t : TripleSegment) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

-- Define the given sets of line segments
def set1 : TripleSegment := ⟨1, 2, 3⟩
def set2 : TripleSegment := ⟨3, 4, 5⟩
def set3 : TripleSegment := ⟨4, 5, 10⟩
def set4 : TripleSegment := ⟨6, 9, 2⟩

-- State the theorem
theorem only_set2_forms_triangle :
  satisfiesTriangleInequality set2 ∧
  ¬satisfiesTriangleInequality set1 ∧
  ¬satisfiesTriangleInequality set3 ∧
  ¬satisfiesTriangleInequality set4 :=
sorry

end NUMINAMATH_CALUDE_only_set2_forms_triangle_l3028_302815


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3028_302854

theorem polynomial_remainder (x : ℝ) : 
  (x^4 - 4*x^2 + 7*x - 8) % (x - 3) = 58 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3028_302854


namespace NUMINAMATH_CALUDE_variance_of_transformed_data_l3028_302831

variable {n : ℕ}
variable (x : Fin n → ℝ)

def variance (data : Fin n → ℝ) : ℝ := sorry

def transform (data : Fin n → ℝ) : Fin n → ℝ := 
  fun i => 3 * data i + 1

theorem variance_of_transformed_data 
  (h : variance x = 2) : 
  variance (transform x) = 18 := by sorry

end NUMINAMATH_CALUDE_variance_of_transformed_data_l3028_302831


namespace NUMINAMATH_CALUDE_min_value_expression_l3028_302890

theorem min_value_expression (x y : ℝ) : (x^2*y - 1)^2 + (x - y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3028_302890


namespace NUMINAMATH_CALUDE_bisection_exact_solution_possible_l3028_302807

/-- The bisection method can potentially find an exact solution -/
theorem bisection_exact_solution_possible
  {f : ℝ → ℝ} {a b : ℝ} (hab : a < b)
  (hf : Continuous f) (hfab : f a * f b < 0) :
  ∃ x ∈ Set.Icc a b, f x = 0 ∧ ∃ n : ℕ, x = (a + b) / 2^(n + 1) :=
sorry

end NUMINAMATH_CALUDE_bisection_exact_solution_possible_l3028_302807


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3028_302826

/-- Quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-value of a quadratic function at a given x -/
def evaluate (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex_x (f : QuadraticFunction) : ℚ :=
  -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
def vertex_y (f : QuadraticFunction) : ℚ :=
  evaluate f (vertex_x f)

theorem quadratic_coefficient (f : QuadraticFunction) 
  (h1 : vertex_x f = 2)
  (h2 : vertex_y f = 5)
  (h3 : evaluate f 3 = 4) :
  f.a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3028_302826


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_t1_l3028_302861

-- Define the motion distance function
def S (t : ℝ) : ℝ := t^3 - 2

-- Define the derivative of S
def S_derivative (t : ℝ) : ℝ := 3 * t^2

-- Theorem statement
theorem instantaneous_velocity_at_t1 :
  S_derivative 1 = 3 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_t1_l3028_302861


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3028_302814

universe u

def U : Set ℤ := {-1, 0, 2}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3028_302814


namespace NUMINAMATH_CALUDE_triangle_side_length_l3028_302878

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  b = Real.sqrt 7 →
  B = π / 3 →  -- 60° in radians
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3028_302878


namespace NUMINAMATH_CALUDE_sara_movie_day_total_expense_l3028_302898

def movie_day_expenses (ticket_price : ℚ) (num_tickets : ℕ) (rented_movie : ℚ) (snacks : ℚ) (parking : ℚ) (movie_poster : ℚ) (bought_movie : ℚ) : ℚ :=
  ticket_price * num_tickets + rented_movie + snacks + parking + movie_poster + bought_movie

theorem sara_movie_day_total_expense :
  movie_day_expenses 10.62 2 1.59 8.75 5.50 12.50 13.95 = 63.53 := by
  sorry

end NUMINAMATH_CALUDE_sara_movie_day_total_expense_l3028_302898


namespace NUMINAMATH_CALUDE_wilson_class_blue_eyes_l3028_302817

/-- Represents the class composition -/
structure ClassComposition where
  total : ℕ
  blond_to_blue_ratio : Rat
  both_traits : ℕ
  neither_trait : ℕ

/-- Calculates the number of blue-eyed students -/
def blue_eyed_count (c : ClassComposition) : ℕ :=
  sorry

/-- Theorem stating the number of blue-eyed students in Mrs. Wilson's class -/
theorem wilson_class_blue_eyes :
  let c : ClassComposition := {
    total := 40,
    blond_to_blue_ratio := 3 / 2,
    both_traits := 8,
    neither_trait := 5
  }
  blue_eyed_count c = 18 := by sorry

end NUMINAMATH_CALUDE_wilson_class_blue_eyes_l3028_302817


namespace NUMINAMATH_CALUDE_football_tournament_semifinal_probability_l3028_302847

theorem football_tournament_semifinal_probability :
  let num_teams : ℕ := 8
  let num_semifinal_pairs : ℕ := 2
  let prob_win_match : ℚ := 1 / 2
  
  -- Probability of team B being in the correct subgroup
  let prob_correct_subgroup : ℚ := num_semifinal_pairs / (num_teams - 1)
  
  -- Probability of both teams winning their matches to reach semifinals
  let prob_both_win : ℚ := prob_win_match * prob_win_match
  
  -- Total probability
  prob_correct_subgroup * prob_both_win = 1 / 14 :=
by sorry

end NUMINAMATH_CALUDE_football_tournament_semifinal_probability_l3028_302847


namespace NUMINAMATH_CALUDE_three_times_work_time_l3028_302818

/-- Given a person can complete a piece of work in a certain number of days,
    this function calculates how many days it will take to complete a multiple of that work. -/
def time_for_multiple_work (days_for_single_work : ℕ) (work_multiple : ℕ) : ℕ :=
  days_for_single_work * work_multiple

/-- Theorem stating that if a person can complete a piece of work in 8 days,
    then they will complete three times the work in 24 days. -/
theorem three_times_work_time :
  time_for_multiple_work 8 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_three_times_work_time_l3028_302818


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3028_302820

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a + 20 → 
  (a + b + c) / 3 = c - 25 → 
  a + b + c = 45 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3028_302820


namespace NUMINAMATH_CALUDE_parallelogram_area_theorem_l3028_302810

/-- Represents the area of a parallelogram with a square removed -/
def parallelogram_area_with_square_removed (base : ℝ) (height : ℝ) (square_side : ℝ) : ℝ :=
  base * height - square_side * square_side

/-- Theorem stating that a parallelogram with base 20 and height 4, 
    after removing a 2x2 square, has an area of 76 square feet -/
theorem parallelogram_area_theorem :
  parallelogram_area_with_square_removed 20 4 2 = 76 := by
  sorry

#eval parallelogram_area_with_square_removed 20 4 2

end NUMINAMATH_CALUDE_parallelogram_area_theorem_l3028_302810


namespace NUMINAMATH_CALUDE_defect_rate_calculation_l3028_302859

/-- Calculates the overall defect rate given three suppliers' defect rates and their supply ratios -/
def overall_defect_rate (rate1 rate2 rate3 : ℚ) (ratio1 ratio2 ratio3 : ℕ) : ℚ :=
  (rate1 * ratio1 + rate2 * ratio2 + rate3 * ratio3) / (ratio1 + ratio2 + ratio3)

/-- Theorem stating that the overall defect rate for the given problem is 14/15 -/
theorem defect_rate_calculation :
  overall_defect_rate (92/100) (95/100) (94/100) 3 2 1 = 14/15 := by
  sorry

end NUMINAMATH_CALUDE_defect_rate_calculation_l3028_302859


namespace NUMINAMATH_CALUDE_trains_at_initial_positions_l3028_302881

/-- Represents a metro line with a given cycle time -/
structure MetroLine where
  cycletime : ℕ

/-- Represents a metro system with multiple lines -/
structure MetroSystem where
  lines : List MetroLine

/-- Checks if all trains return to their initial positions after a given time -/
def allTrainsAtInitialPositions (system : MetroSystem) (time : ℕ) : Prop :=
  ∀ line ∈ system.lines, time % line.cycletime = 0

/-- The metro system of city N -/
def cityNMetro : MetroSystem :=
  { lines := [
      { cycletime := 14 },  -- Red line
      { cycletime := 16 },  -- Blue line
      { cycletime := 18 }   -- Green line
    ]
  }

/-- Theorem: After 2016 minutes, all trains in city N's metro system will be at their initial positions -/
theorem trains_at_initial_positions :
  allTrainsAtInitialPositions cityNMetro 2016 :=
by
  sorry


end NUMINAMATH_CALUDE_trains_at_initial_positions_l3028_302881


namespace NUMINAMATH_CALUDE_sum_reciprocals_simplification_l3028_302895

theorem sum_reciprocals_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^3 + b^3 = 3*(a + b)) : 
  a/b + b/a + 1/(a*b) = 4/(a*b) + 1 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_simplification_l3028_302895


namespace NUMINAMATH_CALUDE_divisibility_of_product_difference_l3028_302840

theorem divisibility_of_product_difference (a₁ a₂ b₁ b₂ c₁ c₂ d : ℤ) 
  (h1 : d ∣ (a₁ - a₂)) 
  (h2 : d ∣ (b₁ - b₂)) 
  (h3 : d ∣ (c₁ - c₂)) : 
  d ∣ (a₁ * b₁ * c₁ - a₂ * b₂ * c₂) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_product_difference_l3028_302840


namespace NUMINAMATH_CALUDE_problem_statement_l3028_302819

theorem problem_statement (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 17 → 
  (40/100 : ℝ) * N = 204 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3028_302819


namespace NUMINAMATH_CALUDE_speed_conversion_l3028_302827

/-- Converts meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

theorem speed_conversion :
  let speed_mps : ℝ := 5.0004
  mps_to_kmph speed_mps = 18.00144 := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l3028_302827


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3028_302837

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence {a_n} satisfying a_1 + a_6 = 11 and a_3 * a_4 = 32/9,
    prove that a_1 = 32/3 or a_1 = 1/3 -/
theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h_sum : a 1 + a 6 = 11)
  (h_prod : a 3 * a 4 = 32/9) :
  a 1 = 32/3 ∨ a 1 = 1/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3028_302837


namespace NUMINAMATH_CALUDE_olympiad_problem_l3028_302864

theorem olympiad_problem (total_students : ℕ) 
  (solved_at_least_1 solved_at_least_2 solved_at_least_3 solved_at_least_4 solved_at_least_5 solved_all_6 : ℕ) : 
  total_students = 2006 →
  solved_at_least_1 = 4 * solved_at_least_2 →
  solved_at_least_2 = 4 * solved_at_least_3 →
  solved_at_least_3 = 4 * solved_at_least_4 →
  solved_at_least_4 = 4 * solved_at_least_5 →
  solved_at_least_5 = 4 * solved_all_6 →
  total_students - solved_at_least_1 = 982 :=
by sorry

end NUMINAMATH_CALUDE_olympiad_problem_l3028_302864


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3028_302833

theorem complex_equation_solution (z : ℂ) : (3 - z) * Complex.I = 2 → z = 3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3028_302833


namespace NUMINAMATH_CALUDE_square_root_of_four_l3028_302874

theorem square_root_of_four (a : ℝ) : a^2 = 4 → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_four_l3028_302874


namespace NUMINAMATH_CALUDE_eight_pointed_star_tip_sum_l3028_302869

/-- An 8-pointed star formed by connecting 8 evenly spaced points on a circle -/
structure EightPointedStar where
  /-- The number of points on the circle -/
  num_points : ℕ
  /-- The points are evenly spaced -/
  evenly_spaced : num_points = 8
  /-- The measure of each small arc between adjacent points -/
  small_arc_measure : ℝ
  /-- Each small arc is 1/8 of the full circle -/
  small_arc_def : small_arc_measure = 360 / 8

/-- The sum of angle measurements of the eight tips of the star -/
def sum_of_tip_angles (star : EightPointedStar) : ℝ :=
  8 * (360 - 4 * star.small_arc_measure)

theorem eight_pointed_star_tip_sum :
  ∀ (star : EightPointedStar), sum_of_tip_angles star = 1440 := by
  sorry

end NUMINAMATH_CALUDE_eight_pointed_star_tip_sum_l3028_302869


namespace NUMINAMATH_CALUDE_additional_teddies_calculation_l3028_302875

/-- The number of additional teddies Jina gets for each bunny -/
def additional_teddies_per_bunny : ℕ :=
  let initial_teddies : ℕ := 5
  let bunnies : ℕ := 3 * initial_teddies
  let koalas : ℕ := 1
  let total_mascots : ℕ := 51
  let initial_mascots : ℕ := initial_teddies + bunnies + koalas
  let additional_teddies : ℕ := total_mascots - initial_mascots
  additional_teddies / bunnies

theorem additional_teddies_calculation : additional_teddies_per_bunny = 2 := by
  sorry

end NUMINAMATH_CALUDE_additional_teddies_calculation_l3028_302875


namespace NUMINAMATH_CALUDE_fraction_simplification_l3028_302812

theorem fraction_simplification (x : ℝ) (h : x = 3) :
  (x^8 - 32*x^4 + 256) / (x^4 - 8) = 65 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3028_302812


namespace NUMINAMATH_CALUDE_mr_martin_coffee_cups_verify_mrs_martin_purchase_verify_mr_martin_purchase_l3028_302848

-- Define the cost of items
def bagel_cost : ℝ := 1.5
def coffee_cost : ℝ := 3.25

-- Define Mrs. Martin's purchase
def mrs_martin_coffee : ℕ := 3
def mrs_martin_bagels : ℕ := 2
def mrs_martin_total : ℝ := 12.75

-- Define Mr. Martin's purchase
def mr_martin_bagels : ℕ := 5
def mr_martin_total : ℝ := 14.00

-- Theorem to prove
theorem mr_martin_coffee_cups : ℕ := by
  -- The number of coffee cups Mr. Martin bought
  sorry

-- Verify Mrs. Martin's purchase
theorem verify_mrs_martin_purchase :
  mrs_martin_coffee * coffee_cost + mrs_martin_bagels * bagel_cost = mrs_martin_total := by
  sorry

-- Verify Mr. Martin's purchase
theorem verify_mr_martin_purchase :
  mr_martin_coffee_cups * coffee_cost + mr_martin_bagels * bagel_cost = mr_martin_total := by
  sorry

end NUMINAMATH_CALUDE_mr_martin_coffee_cups_verify_mrs_martin_purchase_verify_mr_martin_purchase_l3028_302848
