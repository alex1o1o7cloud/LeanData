import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l951_95113

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculate the area of a rectangle -/
noncomputable def rectangleArea (r : Rectangle) : ℝ := r.length * r.width

/-- Calculate the area of a semicircle given its radius -/
noncomputable def semicircleArea (radius : ℝ) : ℝ := (1/2) * Real.pi * radius^2

/-- Calculate the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

theorem enclosed_area_theorem (acde : Rectangle) (a b c d e f : Point) :
  acde.length = 40 →
  acde.width = 24 →
  b.x = (1/3) * acde.length →
  f.y = (1/2) * acde.width →
  let rectangleArea := rectangleArea acde
  let semicircleArea := semicircleArea (acde.length / 2)
  let quadrilateralArea := 2 * triangleArea (acde.length / 3) (acde.width / 2)
  rectangleArea + semicircleArea - quadrilateralArea = 800 + 200 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l951_95113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_earnings_value_l951_95191

/-- Represents the expected quarterly earnings per share in dollars -/
noncomputable def expected_earnings : ℝ := 0.80

/-- Represents the actual quarterly earnings per share in dollars -/
def actual_earnings : ℝ := 1.10

/-- Represents the number of shares owned by a person -/
def shares_owned : ℕ := 500

/-- Represents the total dividend paid to the person in dollars -/
def total_dividend : ℝ := 260

/-- Calculates the dividend per share based on expected and actual earnings -/
noncomputable def dividend_per_share (e : ℝ) : ℝ :=
  e / 2 + 0.4 * (actual_earnings - e)

/-- Theorem stating that the expected quarterly earnings per share is $0.80 -/
theorem expected_earnings_value :
  expected_earnings = 0.80 ∧
  total_dividend = shares_owned * dividend_per_share expected_earnings :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_earnings_value_l951_95191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_properties_l951_95131

-- Define the piecewise function
noncomputable def y (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 2 * x
  else if 1 < x ∧ x ≤ 2 then 2
  else if x > 3 then 3
  else 0  -- This else case is added to make the function total

-- Define the range of the function
def range_y : Set ℝ := {y | 0 ≤ y ∧ y ≤ 2 ∨ y = 3}

-- Theorem statement
theorem y_properties :
  (∃ (x : ℝ), y x = 0) ∧  -- Minimum value
  (∀ (x : ℝ), y x ≤ 3) ∧  -- Maximum value
  (∀ (z : ℝ), z ∈ range_y ↔ ∃ (x : ℝ), y x = z) -- Range
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_properties_l951_95131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_pairs_order_preserving_isomorphic_l951_95112

-- Define the property of being order-preserving isomorphic
def OrderPreservingIsomorphic {α β : Type*} [LinearOrder α] [LinearOrder β] (S : Set α) (T : Set β) : Prop :=
  ∃ f : α → β,
    (∀ x, x ∈ S → f x ∈ T) ∧
    (∀ x₁ x₂, x₁ ∈ S → x₂ ∈ S → x₁ < x₂ → f x₁ < f x₂) ∧
    (∀ y, y ∈ T → ∃ x, x ∈ S ∧ f x = y)

-- Define the sets
def N : Set ℕ := Set.univ
def Nstar : Set ℕ := {n | n > 0}
def A2 : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B2 : Set ℝ := {x | -8 ≤ x ∧ x ≤ 10}
def A3 : Set ℝ := {x | 0 < x ∧ x < 1}
def B3 : Set ℝ := Set.univ

-- State the theorem
theorem all_pairs_order_preserving_isomorphic :
  OrderPreservingIsomorphic N Nstar ∧
  OrderPreservingIsomorphic A2 B2 ∧
  OrderPreservingIsomorphic A3 B3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_pairs_order_preserving_isomorphic_l951_95112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_energy_vehicles_problem_l951_95163

/-- Represents the cost of A-type and B-type cars and the minimum number of A-type cars that can be purchased under given conditions. -/
theorem new_energy_vehicles_problem :
  let cost_equation1 : ℝ → ℝ → Prop := λ x y => 3 * x + y = 55
  let cost_equation2 : ℝ → ℝ → Prop := λ x y => 2 * x + 4 * y = 120
  let total_cars : ℕ := 15
  let max_budget : ℝ := 220
  let min_a_cars : ℕ → Prop := λ m => 
    ∃ (x y : ℝ), cost_equation1 x y ∧ cost_equation2 x y ∧ 
    (∀ k : ℕ, k < m → (x * (k : ℝ) + y * ((total_cars - k) : ℝ) > max_budget)) ∧
    (x * (m : ℝ) + y * ((total_cars - m) : ℝ) ≤ max_budget)
  ∃ (x y : ℝ) (m : ℕ), 
    cost_equation1 x y ∧ 
    cost_equation2 x y ∧ 
    x = 10 ∧ 
    y = 25 ∧ 
    min_a_cars m ∧ 
    m = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_energy_vehicles_problem_l951_95163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l951_95170

/-- The radius of a circle surrounded by six congruent parabolas -/
noncomputable def circle_radius : ℝ := 1/4

/-- A parabola in the arrangement -/
def parabola (r : ℝ) : ℝ → ℝ := λ x ↦ x^2 + r

/-- Tangent line to the parabola at 45° angle -/
def tangent_line : ℝ → ℝ := λ x ↦ x

theorem circle_radius_proof (r : ℝ) :
  (∀ x, (parabola r x - tangent_line x = 0) → (deriv (parabola r)) x = (deriv tangent_line) x) →
  (∃! x, parabola r x = tangent_line x) →
  r = circle_radius := by
  sorry

#check circle_radius_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l951_95170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l951_95184

/-- The time taken for two people to complete a job together, given their individual completion times -/
noncomputable def time_together (time_a time_b : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b)

/-- Theorem: If A can complete a job in 15 days and B in 20 days, they will complete it together in 60/7 days -/
theorem job_completion_time :
  time_together 15 20 = 60 / 7 := by
  -- Unfold the definition of time_together
  unfold time_together
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l951_95184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_conditions_l951_95187

def is_divisible (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

def has_two_consecutive_nondivisors (n : ℕ) : Prop :=
  ∃ k, k ∈ Finset.range 24 ∧ ¬(is_divisible n (k + 1)) ∧ ¬(is_divisible n (k + 2))

def divisible_by_rest (n : ℕ) : Prop :=
  ∀ m, m ∈ Finset.range 25 → 
    (¬∃ k, k ∈ Finset.range 24 ∧ m = k + 1 ∧ ¬(is_divisible n (k + 1)) ∧ ¬(is_divisible n (k + 2))) →
    is_divisible n (m + 1)

theorem least_integer_with_conditions :
  ∃ n, n = 787386600 ∧ 
    n > 0 ∧
    has_two_consecutive_nondivisors n ∧
    divisible_by_rest n ∧
    ∀ m, m < n →
      ¬(m > 0 ∧ has_two_consecutive_nondivisors m ∧ divisible_by_rest m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_conditions_l951_95187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distances_l951_95141

/-- The circle equation --/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 18*x + y^2 + 8*y + 137 = 0

/-- The shortest distance from the origin to the circle --/
noncomputable def shortest_distance : ℝ := Real.sqrt 97 - 2 * Real.sqrt 10

/-- The longest distance from the origin to the circle --/
noncomputable def longest_distance : ℝ := Real.sqrt 97 + 2 * Real.sqrt 10

/-- Theorem stating the shortest and longest distances from the origin to the circle --/
theorem circle_distances :
  (∃ (x y : ℝ), circle_equation x y) →
  (∀ (x y : ℝ), circle_equation x y → 
    (shortest_distance ≤ Real.sqrt (x^2 + y^2)) ∧
    (Real.sqrt (x^2 + y^2) ≤ longest_distance)) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    Real.sqrt (x₁^2 + y₁^2) = shortest_distance ∧
    Real.sqrt (x₂^2 + y₂^2) = longest_distance) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distances_l951_95141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_is_five_l951_95123

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

theorem sum_of_first_five_terms_is_five
  (seq : ArithmeticSequence)
  (h : seq.a 1 + seq.a 3 + seq.a 5 = 3) :
  sumFirstN seq 5 = 5 := by
  sorry

#check sum_of_first_five_terms_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_is_five_l951_95123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_time_difference_is_five_l951_95160

/-- Lina's cycling data --/
structure CyclingData where
  teen_distance : ℚ  -- Distance cycled as a teenager in miles
  teen_time : ℚ      -- Time taken as a teenager in hours
  adult_uphill_distance : ℚ  -- Uphill distance cycled as an adult in miles
  adult_uphill_time : ℚ      -- Uphill time taken as an adult in hours

/-- Calculate the difference in minutes per mile between adult uphill cycling and teenage cycling --/
def cyclingTimeDifference (data : CyclingData) : ℚ :=
  (data.adult_uphill_time * 60 / data.adult_uphill_distance) - 
  (data.teen_time * 60 / data.teen_distance)

/-- Theorem stating the difference in cycling time --/
theorem cycling_time_difference_is_five (data : CyclingData) 
  (h1 : data.teen_distance = 30)
  (h2 : data.teen_time = 2)
  (h3 : data.adult_uphill_distance = 20)
  (h4 : data.adult_uphill_time = 3) :
  cyclingTimeDifference data = 5 := by
  sorry

def exampleData : CyclingData := { 
  teen_distance := 30, 
  teen_time := 2, 
  adult_uphill_distance := 20, 
  adult_uphill_time := 3 
}

#eval cyclingTimeDifference exampleData

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_time_difference_is_five_l951_95160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_angle_theorem_l951_95175

/-- Represents a truncated cone with an inscribed sphere -/
structure TruncatedCone where
  R : ℝ  -- radius of the lower base
  r : ℝ  -- radius of the upper base
  x : ℝ  -- radius of the inscribed sphere
  h : 0 < r ∧ r < R ∧ 0 < x

/-- The angle between the slant height and the base of a truncated cone -/
noncomputable def slant_angle (tc : TruncatedCone) : ℝ := Real.arcsin (2 / Real.sqrt 5)

/-- The total surface area of a truncated cone -/
noncomputable def total_surface_area (tc : TruncatedCone) : ℝ :=
  Real.pi * (2 * tc.x * (tc.R + tc.r) / Real.sqrt (1 - (tc.R - tc.r)^2 / (tc.R + tc.r)^2) + tc.R^2 + tc.r^2)

/-- The surface area of a sphere -/
noncomputable def sphere_surface_area (radius : ℝ) : ℝ := 4 * Real.pi * radius^2

theorem truncated_cone_angle_theorem (tc : TruncatedCone) :
  total_surface_area tc = 2 * sphere_surface_area tc.x →
  slant_angle tc = Real.arcsin (2 / Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_angle_theorem_l951_95175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_times_fifth_root_thirtytwo_equals_four_l951_95197

theorem cube_root_eight_times_fifth_root_thirtytwo_equals_four :
  (8 : Real)^(1/3) * (32 : Real)^(1/5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_times_fifth_root_thirtytwo_equals_four_l951_95197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l951_95130

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle
def circle_C (x y : ℝ) : Prop := (x+3)^2 + (y+3)^2 = 4

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance from a point to the y-axis
def dist_to_y_axis (x : ℝ) : ℝ := |x|

-- Define the distance between two points
noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (m : ℝ), 
    (∀ (x_a y_a x_b y_b : ℝ), 
      parabola x_a y_a → circle_C x_b y_b → 
      m + dist x_a y_a x_b y_b ≥ 2) ∧
    (∃ (x_a y_a x_b y_b : ℝ), 
      parabola x_a y_a ∧ circle_C x_b y_b ∧ 
      m = dist_to_y_axis x_a ∧
      m + dist x_a y_a x_b y_b = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l951_95130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inequality_l951_95194

theorem angle_inequality : 
  (Real.sqrt 3 / 2 : ℝ) > Real.cos (17 * π / 180) * Real.cos (23 * π / 180) - 
  Real.sin (17 * π / 180) * Real.sin (23 * π / 180) ∧
  Real.cos (17 * π / 180) * Real.cos (23 * π / 180) - 
  Real.sin (17 * π / 180) * Real.sin (23 * π / 180) > 
  2 * (Real.cos (25 * π / 180))^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inequality_l951_95194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adults_attending_show_l951_95110

/-- Represents the ticket prices and attendance information for a show. -/
structure ShowInfo where
  adultPrice : ℚ
  childPrice : ℚ
  seniorPrice : ℚ
  totalRevenue : ℚ
  adultRatio : ℕ
  childRatio : ℕ
  seniorRatio : ℕ

/-- Calculates the number of adults attending the show based on the given information. -/
def calculateAdults (info : ShowInfo) : ℕ :=
  let totalRatio := info.adultRatio + info.childRatio + info.seniorRatio
  let unitRevenue := info.adultPrice * info.adultRatio + info.childPrice * info.childRatio + info.seniorPrice * info.seniorRatio
  (info.totalRevenue / unitRevenue * info.adultRatio : ℚ).floor.toNat

/-- Theorem stating that given the specific show information, the number of adults attending is 207. -/
theorem adults_attending_show :
  let info : ShowInfo := {
    adultPrice := 13/2,
    childPrice := 7/2,
    seniorPrice := 9/2,
    totalRevenue := 2124,
    adultRatio := 3,
    childRatio := 2,
    seniorRatio := 1
  }
  calculateAdults info = 207 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adults_attending_show_l951_95110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_ten_l951_95118

-- Define the lengths
noncomputable def AH : ℝ := 12
noncomputable def GF : ℝ := 4
noncomputable def HF : ℝ := 16

-- Define the similarity of triangles
def similar_triangles (DG : ℝ) : Prop := DG / GF = AH / HF

-- Define the area of the shaded region
noncomputable def shaded_area (DG : ℝ) : ℝ := 16 - (1/2 * DG * GF)

-- Theorem statement
theorem shaded_area_is_ten :
  ∃ DG : ℝ, similar_triangles DG ∧ shaded_area DG = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_ten_l951_95118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l951_95174

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 - Real.cos x ^ 4 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1

noncomputable def g (a b x : ℝ) : ℝ := a * f x + b

theorem function_values (a b : ℝ) :
  (∀ x ∈ Set.Icc (-π/6) (2*π/3), g a b x ≤ 11) ∧
  (∀ x ∈ Set.Icc (-π/6) (2*π/3), g a b x ≥ 3) ∧
  (∃ x ∈ Set.Icc (-π/6) (2*π/3), g a b x = 11) ∧
  (∃ x ∈ Set.Icc (-π/6) (2*π/3), g a b x = 3) →
  ((a = 2 ∧ b = 9) ∨ (a = -2 ∧ b = 5)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l951_95174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l951_95196

theorem book_price_change (initial_price decrease_percent increase_percent : ℝ) :
  initial_price = 400 ∧ 
  decrease_percent = 15 ∧ 
  increase_percent = 40 → 
  (initial_price * (1 - decrease_percent / 100) * (1 + increase_percent / 100)) = 476 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l951_95196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_m_is_two_superset_condition_l951_95117

-- Define sets A and B
def A : Set ℝ := {x | (1/32 : ℝ) ≤ Real.exp (-x * Real.log 2) ∧ Real.exp (-x * Real.log 2) ≤ 4}
def B (m : ℝ) : Set ℝ := {x | x^2 + 2*m*x - 3*m^2 < 0}

-- Theorem 1: When m = 2, A ∩ B = {x | -2 ≤ x < 2}
theorem intersection_when_m_is_two :
  A ∩ B 2 = {x : ℝ | -2 ≤ x ∧ x < 2} := by sorry

-- Theorem 2: A ⊇ B if and only if 0 < m ≤ 2/3
theorem superset_condition (m : ℝ) :
  (A ⊇ B m) ↔ (0 < m ∧ m ≤ 2/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_m_is_two_superset_condition_l951_95117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_circle_max_min_distance_l951_95145

theorem complex_circle_max_min_distance (z : ℂ) (h : Complex.abs (z - 2) = 1) :
  (∀ w : ℂ, Complex.abs (w - 2) = 1 → Complex.abs (w + 2 + 5*I) ≤ Complex.abs (z + 2 + 5*I)) →
  Complex.abs (z + 2 + 5*I) = Real.sqrt 41 + 1 ∧
  (∀ w : ℂ, Complex.abs (w - 2) = 1 → Complex.abs (z + 2 + 5*I) ≤ Complex.abs (w + 2 + 5*I)) →
  Complex.abs (z + 2 + 5*I) = Real.sqrt 41 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_circle_max_min_distance_l951_95145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circle_centers_distance_l951_95199

/-- The distance between the centers of inscribed and circumscribed circles of an isosceles triangle -/
noncomputable def circle_centers_distance (h : ℝ) (α : ℝ) : ℝ :=
  (2 * h * Real.sin (Real.pi / 12 - α / 2) * Real.cos (Real.pi / 12 + α / 2)) / (Real.cos α) ^ 2

/-- Theorem: For an isosceles triangle with height h and angle α between the height and the lateral side,
    where α ≤ π/6, the distance between the centers of the inscribed and circumscribed circles is given
    by the circle_centers_distance function. -/
theorem isosceles_triangle_circle_centers_distance (h : ℝ) (α : ℝ) 
    (h_pos : h > 0) (α_pos : α > 0) (α_bound : α ≤ Real.pi / 6) :
  ∃ (d : ℝ), d = circle_centers_distance h α ∧ 
    d = (2 * h * Real.sin (Real.pi / 12 - α / 2) * Real.cos (Real.pi / 12 + α / 2)) / (Real.cos α) ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circle_centers_distance_l951_95199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_circle_radius_l951_95159

theorem middle_circle_radius 
  (r : Fin 5 → ℝ) 
  (geometric_sequence : ∀ i j : Fin 5, i.val < j.val → r i * (r j / r i) = r (Fin.add i (j.val - i.val - 1)))
  (smallest_radius : r 0 = 10)
  (largest_radius : r 4 = 40) :
  r 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_circle_radius_l951_95159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l951_95155

/-- Predicate indicating that a, b, c form a triangle -/
def IsTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate indicating that R is the circumradius of a triangle with sides a, b, c -/
def IsCircumradius (R a b c : ℝ) : Prop :=
  4 * R^2 * (a + b + c) = (a + b + c) * (a + b - c) * (b + c - a) * (c + a - b)

/-- Given a triangle with side lengths a, b, c and circumradius R, 
    the sum of reciprocals of products of pairs of sides is greater than 
    or equal to the reciprocal of the square of the circumradius. -/
theorem triangle_inequality (a b c R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0)
  (h_triangle : IsTriangle a b c)
  (h_circumradius : IsCircumradius R a b c) :
  1 / (a * b) + 1 / (b * c) + 1 / (c * a) ≥ 1 / (R^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l951_95155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_l951_95173

/-- A nonreal cube root of unity -/
noncomputable def ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

/-- ω is a nonreal cube root of unity -/
axiom ω_cube_root : ω^3 = 1 ∧ ω ≠ 1

/-- The set of ordered pairs (a,b) of integers such that |aω + b| = 1 -/
def S : Set (ℤ × ℤ) :=
  {p : ℤ × ℤ | Complex.abs (p.1 • ω + p.2) = 1}

/-- There are exactly 6 ordered pairs (a,b) of integers such that |aω + b| = 1 -/
theorem count_pairs : Finset.card (Finset.filter (fun p => Complex.abs (p.1 • ω + p.2) = 1) (Finset.product (Finset.range 3) (Finset.range 3))) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_l951_95173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unicorn_tether_sum_l951_95135

/-- Represents the configuration of a unicorn tethered to a cylindrical tower. -/
structure UnicornTether where
  towerRadius : ℝ
  ropeLength : ℝ
  unicornHeight : ℝ
  ropeTowerDistance : ℝ
  d : ℕ
  e : ℕ
  f : ℕ

/-- Checks if the given configuration satisfies the problem conditions. -/
def isValidConfiguration (config : UnicornTether) : Prop :=
  config.towerRadius = 10 ∧
  config.ropeLength = 24 ∧
  config.unicornHeight = 6 ∧
  config.ropeTowerDistance = 6 ∧
  config.d > 0 ∧
  config.e > 0 ∧
  config.f > 0 ∧
  Nat.Prime config.f

/-- Calculates the length of rope touching the tower. -/
noncomputable def ropeTouchingTower (config : UnicornTether) : ℝ :=
  (config.d - Real.sqrt (config.e : ℝ)) / config.f

/-- The main theorem to be proved. -/
theorem unicorn_tether_sum (config : UnicornTether) :
  isValidConfiguration config →
  ropeTouchingTower config = (96 - Real.sqrt 36) / 6 →
  config.d + config.e + config.f = 138 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unicorn_tether_sum_l951_95135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_equals_function_l951_95183

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (3 * x - 5) / (m * x + 3)

theorem inverse_equals_function (m : ℝ) :
  (∀ x, g m (g m x) = x) ↔ m = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_equals_function_l951_95183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l951_95114

theorem vector_difference_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 2) 
  (h2 : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 3) 
  (h3 : Real.arccos ((a.1 * b.1 + a.2 * b.2) / (2 * 3)) = π / 3) :
  Real.sqrt (((a.1 - b.1) ^ 2) + ((a.2 - b.2) ^ 2)) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l951_95114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_with_perpendicular_l951_95132

/-- A plane in 3D space -/
structure Plane3D where
  -- Define the plane (this is a placeholder definition)
  dummy : Unit

/-- A line in 3D space -/
structure Line3D where
  -- Define the line (this is a placeholder definition)
  dummy : Unit

/-- Parallel relation between two lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  -- Define parallel relation (this is a placeholder definition)
  True

/-- Perpendicular relation between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  -- Define perpendicular relation (this is a placeholder definition)
  True

/-- Theorem: If l is parallel to m, m is perpendicular to α, and n is perpendicular to α, then l is parallel to n -/
theorem parallel_transitive_with_perpendicular 
  (α : Plane3D) (l m n : Line3D) :
  parallel l m → perpendicular m α → perpendicular n α → parallel l n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_with_perpendicular_l951_95132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_expression_l951_95128

theorem limit_expression : 
  Filter.Tendsto (fun h => ((3 + h)^2 - 3^2) / h) (Filter.atTop.comap Real.toNNReal) (nhds 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_expression_l951_95128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_travel_time_l951_95179

/-- Represents the speed of the pedestrian in km/h -/
def v : ℝ → ℝ := fun x => x

/-- The speed of the cyclist is 4 times the speed of the pedestrian -/
def cyclist_speed (v : ℝ) : ℝ := 4 * v

/-- The time (in hours) between the cyclist overtaking the pedestrian and meeting again -/
def time_between_encounters : ℝ := 2

/-- The distance (in km) traveled by the pedestrian between encounters -/
def pedestrian_distance_between_encounters (v : ℝ) : ℝ := v * time_between_encounters

/-- The distance (in km) traveled by the cyclist between encounters -/
def cyclist_distance_between_encounters (v : ℝ) : ℝ := cyclist_speed v * time_between_encounters

/-- The distance (in km) from point B where the first encounter occurs -/
def encounter_distance_from_B (v : ℝ) : ℝ := 4 * v

/-- The time (in hours) the cyclist stays at point B -/
def cyclist_stop_time : ℝ := 0.5

/-- The total distance (in km) traveled by the cyclist in one direction -/
def total_distance (v : ℝ) : ℝ := 6 * v

/-- Theorem stating that the time taken by the pedestrian to travel from A to B is 10 hours -/
theorem pedestrian_travel_time (v : ℝ) (h : v ≠ 0) : (total_distance v / v) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_travel_time_l951_95179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_cosine_equality_l951_95147

theorem acute_angle_cosine_equality (α : ℝ) :
  0 < α ∧ α < π / 2 →  -- α is an acute angle
  Real.cos (5 * α) = Real.cos (3 * α) →
  α = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_cosine_equality_l951_95147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_internal_tangent_length_l951_95164

theorem common_internal_tangent_length 
  (distance_between_centers : ℝ)
  (radius_smaller : ℝ)
  (radius_larger : ℝ)
  (h1 : distance_between_centers = 50)
  (h2 : radius_smaller = 7)
  (h3 : radius_larger = 12) :
  Real.sqrt (distance_between_centers^2 - (radius_larger + radius_smaller)^2) = Real.sqrt 2139 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_internal_tangent_length_l951_95164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_values_l951_95146

theorem x_values (x : ℤ) : 
  (¬(abs (x - 1) ≥ 2) ∨ False) → 
  (x = 0 ∨ x = 1 ∨ x = 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_values_l951_95146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l951_95115

-- Define the line x-4y+3=0
def reference_line (x y : ℝ) : Prop := x - 4*y + 3 = 0

-- Define the point P(3,2)
def point_P : ℝ × ℝ := (3, 2)

-- Define a line passing through a point with a given slope
def line_through_point (p : ℝ × ℝ) (m : ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

-- Define a line with equal intercepts on the coordinate axes
def equal_intercepts (x y : ℝ) : Prop :=
  ∃ (a : ℝ), x/a + y/a = 1 ∧ a ≠ 0

-- Main theorem
theorem line_equation (l : ℝ → ℝ → Prop) :
  (∃ (x y : ℝ), l x y ∧ x = point_P.1 ∧ y = point_P.2) →  -- l passes through P(3,2)
  (∃ (m : ℝ), ∀ (x y : ℝ), l x y ↔ line_through_point point_P m x y) →  -- l is a line
  (∃ (α : ℝ), Real.tan α = 1/4 ∧ 
    ∀ (x y : ℝ), l x y ↔ line_through_point point_P (Real.tan (2*α)) x y) →  -- Angle condition
  (∀ (x y : ℝ), l x y → equal_intercepts x y) →  -- Equal intercepts condition
  (∀ (x y : ℝ), l x y ↔ (2*x - 3*y = 0 ∨ x + y - 5 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l951_95115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pebbles_cannot_switch_positions_l951_95165

/-- A pebble on a 2D lattice -/
structure Pebble where
  x : ℤ
  y : ℤ

/-- The squared distance between two pebbles -/
def squaredDistance (a b : Pebble) : ℤ :=
  (a.x - b.x)^2 + (a.y - b.y)^2

/-- A valid move preserves the squared distance between pebbles -/
def isValidMove (a b a' b' : Pebble) : Prop :=
  squaredDistance a b = squaredDistance a' b'

/-- A sequence of valid moves -/
def validMoveSequence : List (Pebble × Pebble) → Prop
  | [] => True
  | [_] => True
  | (a, b) :: (a', b') :: rest => isValidMove a b a' b' ∧ validMoveSequence ((a', b') :: rest)

theorem pebbles_cannot_switch_positions (a b : Pebble) :
  ¬∃ (moves : List (Pebble × Pebble)), 
    validMoveSequence ((a, b) :: moves) ∧ 
    moves.getLast? = some (b, a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pebbles_cannot_switch_positions_l951_95165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_profit_percentage_l951_95169

/-- Proves that given A's cost price, A's profit percentage, and the final selling price to C, 
    B's profit percentage is 25%. -/
theorem bicycle_profit_percentage 
  (a_cost_price : ℝ) 
  (a_profit_percentage : ℝ) 
  (c_selling_price : ℝ) 
  (h1 : a_cost_price = 112.5)
  (h2 : a_profit_percentage = 60)
  (h3 : c_selling_price = 225) : 
  (let a_selling_price := a_cost_price * (1 + a_profit_percentage / 100)
   let b_profit := c_selling_price - a_selling_price
   let b_profit_percentage := (b_profit / a_selling_price) * 100
   b_profit_percentage) = 25 := by
  sorry

#check bicycle_profit_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_profit_percentage_l951_95169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l951_95122

-- Define the logarithms as noncomputable
noncomputable def a : Real := Real.log 3 / Real.log 2
noncomputable def b : Real := Real.log 2 / Real.log 3
noncomputable def c : Real := (1/2) * Real.log 5 / Real.log 2

-- State the theorem
theorem log_inequality : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l951_95122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_parabola_standard_equation_l951_95138

-- Ellipse
def ellipse_equation (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x^2 / a^2 + y^2 / b^2 = 1

-- Parabola
def parabola_equation_x (p : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ y^2 = -2 * p * x

def parabola_equation_y (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x^2 = -2 * m * y

theorem ellipse_standard_equation
  (major_axis : ℝ) (focal_length : ℝ)
  (h1 : major_axis = 10)
  (h2 : focal_length = 4) :
  (∃ a b : ℝ, (a = 5 ∧ b^2 = 21) ∧
    (ellipse_equation a b = ellipse_equation 5 (Real.sqrt 21) ∨
     ellipse_equation b a = ellipse_equation (Real.sqrt 21) 5)) := by
  sorry

theorem parabola_standard_equation
  (x0 y0 : ℝ)
  (h1 : x0 = -2)
  (h2 : y0 = -4) :
  (∃ p m : ℝ, (p = 4 ∧ m = 1/2) ∧
    (parabola_equation_x p x0 y0 ∨ parabola_equation_y m x0 y0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_parabola_standard_equation_l951_95138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_revenue_calculation_l951_95133

/-- Calculates the total revenue for a theater performance series -/
def theater_revenue (seats : ℕ) (capacity : ℚ) (ticket_price : ℕ) (performances : ℕ) : ℕ :=
  let tickets_sold := (seats : ℚ) * capacity
  let revenue_per_performance := (tickets_sold * (ticket_price : ℚ)).floor
  (revenue_per_performance * performances).toNat

/-- Theorem: The theater company's total revenue is $28,800 -/
theorem theater_revenue_calculation :
  theater_revenue 400 (4/5) 30 3 = 28800 := by
  -- Unfold the definition of theater_revenue
  unfold theater_revenue
  -- Simplify the arithmetic expressions
  simp [Nat.cast_mul, Nat.cast_ofNat]
  -- Evaluate the numerical expressions
  norm_num
  -- QED
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_revenue_calculation_l951_95133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_line_l951_95190

-- Define the curve and line functions
def curve (x : ℝ) : ℝ := x^2
def line (x : ℝ) : ℝ := 3*x

-- Define the enclosed area
noncomputable def enclosed_area : ℝ := ∫ x in (0)..(3), (line x - curve x)

-- Theorem statement
theorem area_enclosed_by_curve_and_line : enclosed_area = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_line_l951_95190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expr_value_tan_x_value_l951_95140

noncomputable section

-- Define the angle α based on the given point P(-4,3)
def α : Real := Real.arctan (3 / -4)

-- Define the trigonometric expression
def trig_expr (α : Real) : Real :=
  (Real.cos (Real.pi/2 + α) * Real.sin (-Real.pi - α)) / 
  (Real.cos (11*Real.pi/2 - α) * Real.sin (9*Real.pi/2 + α))

-- Theorem for the first part
theorem trig_expr_value : trig_expr α = -3/4 := by sorry

-- Define x and m
variable (x m : Real)

-- Hypotheses for the second part
axiom h1 : Real.sin x = (m - 3) / (m + 5)
axiom h2 : Real.cos x = (4 - 2*m) / (m + 5)
axiom h3 : Real.pi/2 < x ∧ x < Real.pi

-- Theorem for the second part
theorem tan_x_value : Real.tan x = -5/12 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expr_value_tan_x_value_l951_95140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l951_95124

/-- The trajectory of a point P satisfying |PM| - |PN| = 2, where M(2,0) and N(-2,0) are fixed points -/
def trajectory (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  let M : ℝ × ℝ := (2, 0)
  let N : ℝ × ℝ := (-2, 0)
  (Real.sqrt ((x - 2)^2 + y^2) - Real.sqrt ((x + 2)^2 + y^2) = 2) ∧
  (x^2 - y^2/3 = 1) ∧
  (x ≤ -1)

/-- Theorem stating that the trajectory satisfies the given equation -/
theorem trajectory_equation :
  ∀ P : ℝ × ℝ, trajectory P ↔ 
    P.1^2 - P.2^2/3 = 1 ∧ P.1 ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l951_95124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_green_final_state_final_state_is_green_l951_95119

/-- Represents the number of chameleons of each color -/
structure ChameleonCount where
  yellow : ℕ
  red : ℕ
  green : ℕ

/-- The color-changing rule for chameleons -/
def color_change (c : ChameleonCount) : ChameleonCount :=
  { yellow := c.yellow,
    red := c.red,
    green := c.green }

/-- The initial state of chameleons on the island -/
def initial_state : ChameleonCount :=
  { yellow := 7,
    red := 10,
    green := 17 }

/-- The total number of chameleons remains constant -/
axiom total_constant {c : ChameleonCount} : 
  c.yellow + c.red + c.green = 34

/-- The modular differences between colors remain invariant -/
axiom modular_invariant {c : ChameleonCount} :
  (c.red - c.yellow) % 3 = 0 ∧
  (c.yellow - c.green) % 3 = 2 ∧
  (c.green - c.red) % 3 = 1

/-- The final state where all chameleons have the same color -/
def final_state : ChameleonCount :=
  { yellow := 0,
    red := 0,
    green := 34 }

/-- Theorem: The only possible final state is all green chameleons -/
theorem all_green_final_state :
  ∀ c : ChameleonCount, 
    (c.yellow = 0 ∧ c.red = 0 ∧ c.green = 34) ∨
    (c.yellow = 34 ∧ c.red = 0 ∧ c.green = 0) ∨
    (c.yellow = 0 ∧ c.red = 34 ∧ c.green = 0) :=
by sorry

/-- Corollary: The final state is all green chameleons -/
theorem final_state_is_green :
  final_state.yellow = 0 ∧ 
  final_state.red = 0 ∧ 
  final_state.green = 34 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_green_final_state_final_state_is_green_l951_95119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_l951_95177

theorem min_value_sin_cos (x : ℝ) : Real.sin x ^ 6 + (5 / 3) * Real.cos x ^ 6 ≥ 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_l951_95177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_range_l951_95167

open Real

noncomputable section

/-- A plane is represented as ℝ² -/
def Plane := ℝ × ℝ

/-- The dot product of two vectors in the plane -/
def dot_product (v w : Plane) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The squared magnitude of a vector in the plane -/
def magnitude_squared (v : Plane) : ℝ := v.1 * v.1 + v.2 * v.2

/-- The magnitude of a vector in the plane -/
noncomputable def magnitude (v : Plane) : ℝ := Real.sqrt (magnitude_squared v)

/-- A vector is a unit vector if its magnitude is 1 -/
def is_unit_vector (v : Plane) : Prop := magnitude v = 1

theorem vector_magnitude_range 
  (e a : Plane) 
  (h1 : is_unit_vector e) 
  (h2 : dot_product a e = 2) 
  (h3 : ∀ t : ℝ, magnitude_squared a ≤ 5 * magnitude (⟨a.1 + t * e.1, a.2 + t * e.2⟩)) :
  sqrt 5 ≤ magnitude a ∧ magnitude a ≤ 2 * sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_range_l951_95167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_solutions_is_zero_l951_95106

theorem sum_of_x_solutions_is_zero (y : ℝ) : 
  y = 8 → 
  (∃ a b : ℝ, a^2 + y^2 = 169 ∧ b^2 + y^2 = 169 ∧ ∀ z, z^2 + y^2 = 169 → (z = a ∨ z = b)) →
  (∃ a b : ℝ, a^2 + y^2 = 169 ∧ b^2 + y^2 = 169 ∧ a + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_solutions_is_zero_l951_95106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l951_95104

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt 2 * Real.sin (x/4) * Real.cos (x/4) + Real.sqrt 6 * (Real.cos (x/4))^2 - Real.sqrt 6 / 2

-- State the theorem
theorem min_value_of_f :
  ∀ x ∈ Set.Icc (-π/3) (π/3), 
    f x ≥ Real.sqrt 2 / 2 ∧ 
    ∃ y ∈ Set.Icc (-π/3) (π/3), f y = Real.sqrt 2 / 2 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l951_95104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l951_95188

/-- The sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 2 => 2 * a (n + 1) / (a (n + 1) + 2)

/-- Theorem stating the general term formula for the sequence a_n -/
theorem a_general_term : ∀ n : ℕ, n > 0 → a n = 2 / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l951_95188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_side_for_table_movement_l951_95103

/-- The side length of a square room that can accommodate a 9' × 12' table -/
def room_side_length : ℝ := 15

/-- The length of the table -/
def table_length : ℝ := 12

/-- The width of the table -/
def table_width : ℝ := 9

/-- Theorem stating that the room_side_length is the smallest integer that allows
    a table with dimensions table_length × table_width to be moved to a different corner
    without tilting or disassembling -/
theorem min_room_side_for_table_movement :
  room_side_length = ⌈Real.sqrt (table_length^2 + table_width^2)⌉ ∧
  ∀ s : ℝ, s < room_side_length → s < Real.sqrt (table_length^2 + table_width^2) := by
  sorry

#eval room_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_side_for_table_movement_l951_95103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l951_95105

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : Real.cos α ^ 2 + Real.sin (π + 2*α) = 3/10) : 
  Real.tan α = -7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l951_95105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_value_l951_95171

open Real

theorem minimum_a_value (a : ℝ) (h_a : a > 0) :
  (∀ (x₁ : ℝ) (x₂ : ℝ), x₂ ∈ Set.Icc 1 (Real.exp 1) → 
    x₁ + a^2 / x₁ ≥ x₂ - Real.log x₂) →
  a ≥ (Real.exp 1 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_value_l951_95171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kelseys_sister_age_l951_95150

/-- Calculates the age of Kelsey's older sister in 2021 given the provided conditions. -/
theorem kelseys_sister_age 
  (kelsey_age_1999 : ℕ)
  (sister_age_diff : ℕ)
  (current_year : ℕ)
  (kelsey_birth_year : ℕ)
  (h1 : kelsey_age_1999 = 25)
  (h2 : sister_age_diff = 3)
  (h3 : current_year = 2021)
  (h4 : kelsey_birth_year = 1999 - kelsey_age_1999) :
  current_year - (kelsey_birth_year - sister_age_diff) = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kelseys_sister_age_l951_95150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_plus_two_l951_95156

open Real MeasureTheory

theorem integral_sin_plus_two (f : ℝ → ℝ) :
  (∀ a, f a = ∫ x in (0:ℝ)..a, (2 + Real.sin x)) →
  f (π / 2) = π + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_plus_two_l951_95156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_range_l951_95182

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := Real.log (2 / (1 - x) + a)

-- State the theorem
theorem f_negative_range (a : ℝ) :
  ∀ x : ℝ, f x a < 0 ↔ -1 < x ∧ x < 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_range_l951_95182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheilas_family_contribution_l951_95168

/-- Calculates the amount Sheila's family added to her piggy bank --/
theorem sheilas_family_contribution
  (initial_savings : ℕ)
  (monthly_savings : ℕ)
  (saving_period_months : ℕ)
  (final_amount : ℕ)
  (h1 : initial_savings = 3000)
  (h2 : monthly_savings = 276)
  (h3 : saving_period_months = 4 * 12)
  (h4 : final_amount = 23248) :
  final_amount - (initial_savings + monthly_savings * saving_period_months) = 7000 := by
  -- Proof steps would go here
  sorry

#check sheilas_family_contribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheilas_family_contribution_l951_95168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AF₁F₂_is_15_degrees_l951_95189

-- Define the hyperbola parameters
variable (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)

-- Define the eccentricity
noncomputable def e : ℝ := 2 + Real.sqrt 6 - Real.sqrt 3 - Real.sqrt 2

-- Define the angle AFₑF₂
noncomputable def angle_AF₁F₂ (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) : ℝ := 
  Real.arctan (2 - Real.sqrt 3)

-- State the theorem
theorem angle_AF₁F₂_is_15_degrees (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  angle_AF₁F₂ a b ha hb hab = 15 * Real.pi / 180 := by
  sorry

#check angle_AF₁F₂_is_15_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AF₁F₂_is_15_degrees_l951_95189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_a_l951_95152

theorem irrational_a (a : ℝ) (h : (1 : ℝ) / a = a - ⌊a⌋) : Irrational a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_a_l951_95152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_D_l951_95153

theorem smallest_constant_D : 
  (∃ (D : ℝ), ∀ (θ : ℝ), Real.sin θ ^ 2 + Real.cos θ ^ 2 + 1 ≥ D * (Real.sin θ + Real.cos θ)) ∧ 
  (∀ (D : ℝ), (∀ (θ : ℝ), Real.sin θ ^ 2 + Real.cos θ ^ 2 + 1 ≥ D * (Real.sin θ + Real.cos θ)) → D ≤ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_D_l951_95153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_angle_sine_value_l951_95143

/-- A rhombus with diagonals and side lengths in geometric progression --/
structure GeometricRhombus where
  side : ℝ
  diag1 : ℝ
  diag2 : ℝ
  side_pos : 0 < side
  diag1_pos : 0 < diag1
  diag2_pos : 0 < diag2
  geom_prog : diag1 * diag2 = side^2
  diag_order : diag1 ≤ diag2

/-- The sine of the angle between the side and the longer diagonal --/
noncomputable def angle_sine (r : GeometricRhombus) : ℝ :=
  r.diag1 / (2 * r.side)

theorem rhombus_angle_sine_value (r : GeometricRhombus) 
    (h : 1/2 < angle_sine r) : 
    angle_sine r = Real.sqrt ((Real.sqrt 17 - 1) / 8) := by
  sorry

#check rhombus_angle_sine_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_angle_sine_value_l951_95143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_bounds_l951_95172

noncomputable def g : ℕ → ℝ
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => Real.log 3
  | n + 4 => Real.log (n + 4 + g (n + 3))

noncomputable def B : ℝ := g 3016

theorem B_bounds : Real.log 3019 < B ∧ B < Real.log 3020 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_bounds_l951_95172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_range_l951_95157

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * log x - exp 1

-- State the theorem
theorem f_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ 0) → a ≤ exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_range_l951_95157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_nonzero_digit_of_1_over_347_l951_95116

theorem first_nonzero_digit_of_1_over_347 :
  ∃ (n : ℕ) (d : ℕ), d ∈ Finset.range 10 \ {0} ∧
  (1 : ℚ) / 347 = (n : ℚ) / 10^(n.succ) + d / 10^(n.succ + 1) + ((1 : ℚ) / 347 - ((n : ℚ) / 10^(n.succ) + d / 10^(n.succ + 1))) ∧
  d = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_nonzero_digit_of_1_over_347_l951_95116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_operation_satisfies_equation_l951_95136

theorem unique_operation_satisfies_equation :
  ∃! op : ℝ → ℝ → ℝ, 
    ((op = (λ x y => x + y)) ∨ (op = (λ x y => x - y)) ∨ (op = (λ x y => x * y)) ∨ (op = (λ x y => x / y))) ∧
    (op 8 2) + 5 - (3 - 2) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_operation_satisfies_equation_l951_95136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_part_two_range_of_a_part_three_l951_95192

noncomputable def e : ℝ := Real.exp 1

def f (x : ℝ) : ℝ := -x^3 + x^2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

noncomputable def h (x : ℝ) : ℝ := Real.exp (1 - x) * f x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then f x else g a x

theorem tangent_line_at_one (x : ℝ) :
  (h x - h 1) = -(x - 1) * (deriv h) 1 := by sorry

theorem range_of_a_part_two (a : ℝ) :
  (∃ x ∈ Set.Icc 1 e, g a x ≥ -x^2 + (a + 2) * x) →
  a ≤ (e^2 - 2*e) / (e - 1) := by sorry

theorem range_of_a_part_three (a : ℝ) :
  (∀ t ≤ -1, ∃ q : ℝ,
    q = -t ∧
    (t * F a t * q * F a q < 0) ∧
    ((t + q) / 2 = 0)) →
  a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_part_two_range_of_a_part_three_l951_95192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_special_sequence_length_l951_95144

/-- A sequence of positive integers satisfying the given conditions -/
def SpecialSequence (n : ℕ) (a : ℕ → ℕ) : Prop :=
  (∀ i, a i > 0) ∧
  (∃ v₁ v₂ v₃ v₄ v₅, v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₁ ≠ v₄ ∧ v₁ ≠ v₅ ∧ 
                      v₂ ≠ v₃ ∧ v₂ ≠ v₄ ∧ v₂ ≠ v₅ ∧
                      v₃ ≠ v₄ ∧ v₃ ≠ v₅ ∧
                      v₄ ≠ v₅ ∧
                      ∃ i₁ i₂ i₃ i₄ i₅, i₁ ≤ n ∧ i₂ ≤ n ∧ i₃ ≤ n ∧ i₄ ≤ n ∧ i₅ ≤ n ∧
                      a i₁ = v₁ ∧ a i₂ = v₂ ∧ a i₃ = v₃ ∧ a i₄ = v₄ ∧ a i₅ = v₅) ∧
  (∀ i j, 1 ≤ i → i < j → j ≤ n → ∃ k l, k ≠ l ∧ k ≠ i ∧ k ≠ j ∧ l ≠ i ∧ l ≠ j ∧ a i + a j = a k + a l)

/-- The main theorem stating that the minimum value of n for a SpecialSequence is 13 -/
theorem min_special_sequence_length :
  (∃ n a, SpecialSequence n a) ∧ (∀ n a, SpecialSequence n a → n ≥ 13) ∧ (∃ a, SpecialSequence 13 a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_special_sequence_length_l951_95144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l951_95100

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
noncomputable def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

/-- Calculates the volume of a cube given its side length -/
noncomputable def cubeVolume (side : ℝ) : ℝ := side^3

/-- Calculates the percentage of volume removed from a box -/
noncomputable def percentageVolumeRemoved (boxDim : BoxDimensions) (cubeSide : ℝ) : ℝ :=
  let originalVolume := boxVolume boxDim
  let removedVolume := 8 * cubeVolume cubeSide
  (removedVolume / originalVolume) * 100

/-- Theorem stating that the percentage of volume removed from the given box is approximately 21.33% -/
theorem volume_removed_percentage :
  let boxDim : BoxDimensions := ⟨20, 12, 10⟩
  let cubeSide : ℝ := 4
  abs (percentageVolumeRemoved boxDim cubeSide - 21.33) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l951_95100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l951_95180

theorem congruent_integers_count : 
  (Finset.filter (λ n : ℕ => n < 500 ∧ n % 7 = 2) (Finset.range 500)).card = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l951_95180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limits_theorem_l951_95108

-- Define the limits
noncomputable def limit_x_sin_x : ℝ := 1
noncomputable def limit_sin_4x_sin_5x : ℝ := 4/5
noncomputable def limit_one_minus_cos_x_squared : ℝ := 1/2

-- State the theorem
theorem limits_theorem :
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 0 → |x| < δ → |x / Real.sin x - limit_x_sin_x| < ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 0 → |x| < δ → |Real.sin (4*x) / Real.sin (5*x) - limit_sin_4x_sin_5x| < ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 0 → |x| < δ → |(1 - Real.cos x) / x^2 - limit_one_minus_cos_x_squared| < ε) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limits_theorem_l951_95108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_s_plus_t_l951_95142

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  p : ℕ
  q : ℕ
  r : ℕ
  s : ℕ
  t : ℕ
  sum : ℕ
  top_left : ℕ := 27
  top_right : ℕ := 19
  middle_left : ℕ := 17
  middle_top : ℕ := 26

/-- The magic square satisfies the properties of having consistent sums in rows, columns, and diagonals -/
def is_valid_magic_square (m : MagicSquare) : Prop :=
  m.top_left + m.s + m.top_right = m.sum ∧
  m.top_left + m.middle_left + m.p = m.sum ∧
  m.top_right + m.t + m.q = m.sum ∧
  m.s + m.r + m.middle_top = m.sum ∧
  m.p + m.middle_top + m.q = m.sum ∧
  m.top_left + m.r + m.q = m.sum ∧
  m.p + m.r + m.top_right = m.sum

theorem magic_square_s_plus_t (m : MagicSquare) 
  (h : is_valid_magic_square m) : m.s + m.t = 46 := by
  sorry

#check magic_square_s_plus_t

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_s_plus_t_l951_95142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_property_l951_95193

/-- Given an invertible function f and real numbers a and b such that f(a) = b and f(b) = 3, prove that a - b = -2 -/
theorem inverse_function_property (f : ℝ → ℝ) (a b : ℝ) 
  (h_inv : Function.Bijective f) (h_fa : f a = b) (h_fb : f b = 3) : 
  a - b = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_property_l951_95193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constructible_polygons_l951_95102

/-- A right triangle with one leg half the length of the hypotenuse -/
structure SpecialRightTriangle where
  hypotenuse : ℝ
  shortLeg : ℝ
  longLeg : ℝ
  hypotenuse_positive : hypotenuse > 0
  right_angle : shortLeg^2 + longLeg^2 = hypotenuse^2
  short_leg_half_hypotenuse : shortLeg = hypotenuse / 2

/-- Regular polygons that can be constructed with the special right triangle -/
def ConstructibleRegularPolygon : Set ℕ := {3, 4, 6, 12}

/-- A placeholder for the concept of a regular polygon -/
def RegularPolygon (n : ℕ) := Unit

/-- A placeholder for the concept of constructing a regular polygon -/
def ConstructPolygon (t : SpecialRightTriangle) (n : ℕ) : Prop := True

/-- Theorem stating which regular polygons can be constructed -/
theorem constructible_polygons (t : SpecialRightTriangle) :
  ∀ n : ℕ, n ∈ ConstructibleRegularPolygon ↔ ConstructPolygon t n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constructible_polygons_l951_95102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_I_is_positive_l951_95127

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the expression I
noncomputable def I (n : ℕ+) : ℝ := (n + 1)^2 + n - (floor (Real.sqrt ((n + 1)^2 + n + 1)))^2

-- Theorem statement
theorem I_is_positive (n : ℕ+) : I n > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_I_is_positive_l951_95127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_for_two_maxima_l951_95126

open Real Set

theorem min_interval_for_two_maxima (t : ℕ+) : 
  (∀ x, x ∈ Icc (0 : ℝ) (t : ℝ) → sin (π * x / 3) ≤ 1) ∧
  (∃ x₁ x₂, x₁ ∈ Icc (0 : ℝ) (t : ℝ) ∧ x₂ ∈ Icc (0 : ℝ) (t : ℝ) ∧ x₁ < x₂ ∧ 
    sin (π * x₁ / 3) = 1 ∧ sin (π * x₂ / 3) = 1) →
  8 ≤ t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_for_two_maxima_l951_95126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_correct_l951_95125

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x < 0 then x^2 + x else -x^2 + x

-- State the theorem
theorem f_is_odd_and_correct : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, x < 0 → f x = x^2 + x) ∧
  (∀ x, x > 0 → f x = -x^2 + x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_correct_l951_95125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_selection_is_five_l951_95149

def valid_number (n : ℕ) : Bool := 1 ≤ n ∧ n ≤ 20

def random_table : List ℕ :=
  [65, 72, 08, 02, 63, 14, 07, 02, 43, 69, 97, 28, 01, 98]

def valid_selections (table : List ℕ) : List ℕ :=
  table.filter valid_number

theorem fifth_selection_is_five (table : List ℕ) 
  (h : table = random_table) :
  (valid_selections table).get? 4 = some 5 := by
  sorry

#eval valid_selections random_table

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_selection_is_five_l951_95149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_l951_95195

/-- The function f(x) = kx - ln x is monotonically increasing on (1/2, +∞) iff k ≥ 2 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > (1/2 : ℝ), Monotone (λ x => k * x - Real.log x)) ↔ k ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_l951_95195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l951_95120

theorem evaluate_expression : (81 : ℝ) ^ (1/2 : ℝ) * (125 : ℝ) ^ (-(1/3) : ℝ) * (32 : ℝ) ^ (1/5 : ℝ) = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l951_95120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_specific_tetrahedron_l951_95134

/-- Represents a tetrahedron with equilateral triangle faces -/
structure Tetrahedron where
  side_length : ℝ
  dihedral_angle : ℝ

/-- Calculates the maximum projection area of a rotating tetrahedron -/
noncomputable def max_projection_area (t : Tetrahedron) : ℝ :=
  (3 * t.side_length^2 * Real.sqrt 3) / 4

/-- Theorem stating the maximum projection area for a specific tetrahedron -/
theorem max_projection_area_specific_tetrahedron :
  let t : Tetrahedron := { side_length := 3, dihedral_angle := π/6 }
  max_projection_area t = (9 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_specific_tetrahedron_l951_95134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_savings_l951_95154

/-- Calculates the savings amount given total expenses and savings percentage -/
noncomputable def calculate_savings (total_expenses : ℝ) (savings_percentage : ℝ) : ℝ :=
  let monthly_salary := total_expenses / (1 - savings_percentage)
  savings_percentage * monthly_salary

/-- Theorem stating that given the specified conditions, the savings amount is approximately 1937.78 -/
theorem kishore_savings :
  let total_expenses : ℝ := 17440
  let savings_percentage : ℝ := 0.1
  let calculated_savings := calculate_savings total_expenses savings_percentage
  abs (calculated_savings - 1937.78) < 0.01 := by
  sorry

-- Use #eval only for computable functions
def approx_savings : ℚ := 1937.78

#check approx_savings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_savings_l951_95154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_not_covered_by_overlapping_circles_l951_95109

/-- The area of the region within a circumscribing circle not covered by two overlapping circles -/
theorem area_not_covered_by_overlapping_circles 
  (r₁ r₂ d R : ℝ) 
  (h₁ : r₁ = 4) 
  (h₂ : r₂ = 5) 
  (h₃ : d = 6) 
  (h₄ : R = (d * r₁ * r₂) / (4 * Real.sqrt (7.5 * 1.5 * 3.5 * 2.5))) : 
  |π * R^2 - π * r₁^2 - π * r₂^2 - 110.57 * π| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_not_covered_by_overlapping_circles_l951_95109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_birds_caught_l951_95101

/-- Represents a hunting session with the number of birds hunted and the success rate -/
structure HuntingSession where
  birds_hunted : ℕ
  success_rate : ℚ
  hunted_nonneg : 0 ≤ birds_hunted
  rate_between_zero_and_one : 0 ≤ success_rate ∧ success_rate ≤ 1

/-- Calculates the number of birds caught in a session -/
def birds_caught (session : HuntingSession) : ℕ :=
  Int.toNat ⌊(session.birds_hunted : ℚ) * session.success_rate⌋

/-- Theorem stating the total number of birds caught by the cat -/
theorem total_birds_caught
  (morning : HuntingSession)
  (afternoon : HuntingSession)
  (night : HuntingSession)
  (h_morning : morning.birds_hunted = 15 ∧ morning.success_rate = 3/5)
  (h_afternoon : afternoon.birds_hunted = 25 ∧ afternoon.success_rate = 4/5)
  (h_night : night.birds_hunted = 20 ∧ night.success_rate = 9/10) :
  birds_caught morning + birds_caught afternoon + birds_caught night = 47 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_birds_caught_l951_95101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sum_properties_l951_95166

theorem special_sum_properties (a b c : ℤ) (h : a + b + c = 0) :
  let d : ℤ := a^1999 + b^1999 + c^1999
  (d ≠ 2) ∧ (Nat.Prime (d.natAbs) ↔ d = 1999) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sum_properties_l951_95166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eighth_value_l951_95185

/-- Sequence b defined recursively -/
def b : ℕ → ℕ
  | 0 => 2  -- Adding the base case for 0
  | 1 => 3
  | n + 2 => b (n + 1) + b n

/-- Sequence a defined in terms of b -/
def a (n : ℕ) : ℚ := (b n : ℚ) / (b (n + 1) : ℚ)

theorem a_eighth_value :
  a 1 = 2/3 →
  a 2 = 3/5 →
  a 3 = 5/8 →
  a 4 = 8/13 →
  a 8 = 55/89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eighth_value_l951_95185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_arc_length_l951_95151

noncomputable section

-- Define the radius and central angle
def radius : ℝ := 4
def central_angle : ℝ := 60 * (Real.pi / 180)

-- Define the arc length formula
def arc_length (r : ℝ) (θ : ℝ) : ℝ := r * θ

-- Theorem statement
theorem sector_arc_length : 
  arc_length radius central_angle = (4 * Real.pi) / 3 := by
  -- Expand the definitions
  unfold arc_length radius central_angle
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_arc_length_l951_95151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_less_than_product_l951_95129

def valid_numbers : Finset Nat := {1, 3, 5, 7, 8, 9, 10}

def is_valid_pair (a b : Nat) : Bool := (a - 1) * (b - 1) > 1

def total_pairs : Nat := valid_numbers.card ^ 2

def valid_pairs : Finset (Nat × Nat) :=
  Finset.filter (λ p => is_valid_pair p.1 p.2) (Finset.product valid_numbers valid_numbers)

theorem probability_sum_less_than_product :
  (valid_pairs.card : Rat) / total_pairs = 38 / 49 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_less_than_product_l951_95129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farey_property_l951_95121

/-- Farey sequence of order n -/
def farey_sequence (n : ℕ) : Set ℚ :=
  {q : ℚ | 0 ≤ q ∧ q < 1 ∧ q.den ≤ n}

/-- Two fractions are consecutive in a set if there's no fraction between them -/
def consecutive_in (a b c d : ℤ) (S : Set ℚ) : Prop :=
  (a : ℚ) / b ∈ S ∧ (c : ℚ) / d ∈ S ∧
  (a : ℚ) / b < (c : ℚ) / d ∧
  ∀ (x y : ℤ), (x : ℚ) / y ∈ S → (a : ℚ) / b < (x : ℚ) / y → (x : ℚ) / y < (c : ℚ) / d → False

theorem farey_property (n : ℕ) (a b c d : ℤ) :
  consecutive_in a b c d (farey_sequence n) →
  |b * c - a * d| = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farey_property_l951_95121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percent_when_cost_is_75_percent_of_selling_price_l951_95139

theorem profit_percent_when_cost_is_75_percent_of_selling_price :
  ∀ (selling_price : ℝ), selling_price > 0 →
    (((selling_price - 0.75 * selling_price) / (0.75 * selling_price)) * 100 = 100 / 3) := by
  intro selling_price h_positive
  -- The proof steps would go here
  sorry

#check profit_percent_when_cost_is_75_percent_of_selling_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percent_when_cost_is_75_percent_of_selling_price_l951_95139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_meaningful_g_equals_zero_l951_95178

noncomputable section

-- Define the fractions
noncomputable def f (x : ℝ) := 2 * x / (x + 3)
noncomputable def g (x : ℝ) := (x + 1) / (2 * x - 3)

-- Theorem for the first fraction
theorem f_meaningful (x : ℝ) : 
  (∃ y : ℝ, f x = y) ↔ x ≠ -3 :=
by sorry

-- Theorem for the second fraction
theorem g_equals_zero (x : ℝ) :
  g x = 0 ↔ x = -1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_meaningful_g_equals_zero_l951_95178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_condition_upper_bound_when_a_is_one_l951_95137

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 2 / Real.sqrt x

-- Statement 1: Non-monotonicity condition
theorem non_monotonic_condition (a : ℝ) :
  (∃ x y, x > 1 ∧ y > 1 ∧ x < y ∧ f a x > f a y) ↔ 0 < a ∧ a < 1 := by
  sorry

-- Statement 2: Upper bound when a = 1
theorem upper_bound_when_a_is_one (x : ℝ) (h : x > 1) :
  f 1 x < x^2/2 - x + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_condition_upper_bound_when_a_is_one_l951_95137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_true_l951_95162

-- Define the propositions
def proposition_A : Prop := 
  (¬(∀ x : ℝ, x^2 = 1 → x = 1)) ↔ (∀ x : ℝ, x^2 = 1 → x ≠ 1)

def proposition_B : Prop := 
  (¬(∃ x : ℝ, x^2 + x - 1 < 0)) ↔ (∀ x : ℝ, x^2 + x - 1 > 0)

def proposition_C : Prop := 
  (∀ x y : ℝ, Real.sin x ≠ Real.sin y → x ≠ y) → true

def proposition_D : Prop := 
  (∀ x : ℝ, x^2 - 5*x - 6 = 0 → x = -1) ∧ (∃ x : ℝ, x ≠ -1 ∧ x^2 - 5*x - 6 = 0)

-- Theorem stating that only proposition C is true
theorem only_C_is_true : 
  ¬proposition_A ∧ ¬proposition_B ∧ proposition_C ∧ ¬proposition_D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_true_l951_95162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_area_implies_a_equals_one_l951_95107

/-- The curve function f(x) = √x -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

/-- The tangent line at point (a, f(a)) -/
noncomputable def tangentLine (a : ℝ) (x : ℝ) : ℝ :=
  f a + (deriv f a) * (x - a)

/-- The area of the triangle formed by the tangent line and the coordinate axes -/
noncomputable def triangleArea (a : ℝ) : ℝ :=
  (1 / 2) * abs (tangentLine a 0) * abs a

theorem tangent_area_implies_a_equals_one (a : ℝ) (h : a > 0) :
  triangleArea a = 1 / 4 → a = 1 := by
  sorry

#check tangent_area_implies_a_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_area_implies_a_equals_one_l951_95107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_geometric_remainders_l951_95176

theorem smallest_m_for_geometric_remainders : ∃ (M : ℕ),
  (∀ (M' : ℕ), M' < M →
    ¬∃ (N : ℕ) (r : ℚ),
      0 < N % 6 ∧
      N % 6 < N % 36 ∧
      N % 36 < N % 216 ∧
      N % 216 < N % M' ∧
      (N % 36 : ℚ) / (N % 6 : ℚ) = r ∧
      (N % 216 : ℚ) / (N % 36 : ℚ) = r ∧
      (N % M' : ℚ) / (N % 216 : ℚ) = r) ∧
  (∃ (N : ℕ) (r : ℚ),
    0 < N % 6 ∧
    N % 6 < N % 36 ∧
    N % 36 < N % 216 ∧
    N % 216 < N % M ∧
    (N % 36 : ℚ) / (N % 6 : ℚ) = r ∧
    (N % 216 : ℚ) / (N % 36 : ℚ) = r ∧
    (N % M : ℚ) / (N % 216 : ℚ) = r) ∧
  M = 2001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_geometric_remainders_l951_95176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circle_area_ratio_l951_95198

/-- The radius of the circle -/
def radius : ℝ := 3

/-- The number of sectors the circle is divided into -/
def num_sectors : ℕ := 6

/-- The area of the original circle -/
noncomputable def circle_area : ℝ := Real.pi * radius^2

/-- The area of one triangular sector -/
noncomputable def sector_area : ℝ := (1/2) * radius^2 * (2*Real.pi/num_sectors)

/-- The area of the hexagonal figure formed by recombining the sectors -/
noncomputable def hexagon_area : ℝ := num_sectors * sector_area

/-- The ratio of the hexagonal area to the circle area -/
noncomputable def area_ratio : ℝ := hexagon_area / circle_area

theorem hexagon_circle_area_ratio : area_ratio = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circle_area_ratio_l951_95198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l951_95186

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_theorem (h1 : ∀ x > 0, (x * f' x - 1) < 0) 
                             (h2 : f (Real.exp 1) = 2) :
  ∀ x : ℝ, f (Real.exp x) < x + 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l951_95186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_periodic_f_exists_odd_f_l951_95161

-- Define the function f as noncomputable due to Real.sqrt
noncomputable def f (m n x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (m * x + n)

-- Theorem for the periodic property
theorem exists_periodic_f : ∃ m : ℝ, ∀ x n : ℝ, f m n (x + 1) = f m n x := by
  -- Provide m = 2π as the witness
  use 2 * Real.pi
  -- The rest of the proof
  sorry

-- Theorem for the odd function property
theorem exists_odd_f : ∃ m : ℝ, ∀ x n : ℝ, f m n (-x) = -f m n x := by
  -- Provide m = 1 as the witness (any non-zero real number would work)
  use 1
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_periodic_f_exists_odd_f_l951_95161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hyperdeficient_numbers_l951_95181

/-- Sum of all divisors of a positive integer n -/
def f (n : ℕ) : ℕ := sorry

/-- A number n is hyperdeficient if f(f(n)) = n + 3 -/
def is_hyperdeficient (n : ℕ) : Prop := f (f n) = n + 3

/-- There are no hyperdeficient numbers -/
theorem no_hyperdeficient_numbers : ∀ n : ℕ, n > 0 → ¬ is_hyperdeficient n := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hyperdeficient_numbers_l951_95181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l951_95148

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  4 * a^(2/3 : ℝ) * b^(-(1/3) : ℝ) / (-(2/3 : ℝ) * a^(-(1/3) : ℝ) * b^(2/3 : ℝ)) = -6 * a / b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l951_95148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_example_monomial_properties_l951_95158

/-- Represents a monomial with coefficient and variables --/
structure Monomial where
  coefficient : ℤ
  vars : List (Char × ℕ)

/-- Calculates the degree of a monomial --/
def degree (m : Monomial) : ℕ := m.vars.foldl (fun acc (_, exp) => acc + exp) 0

/-- The monomial -2xy^3 --/
def example_monomial : Monomial := {
  coefficient := -2,
  vars := [('x', 1), ('y', 3)]
}

theorem example_monomial_properties :
  example_monomial.coefficient = -2 ∧ degree example_monomial = 4 := by
  apply And.intro
  · rfl
  · rfl

#eval example_monomial.coefficient
#eval degree example_monomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_example_monomial_properties_l951_95158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_orders_eq_18_l951_95111

/-- Represents the number of participants in the race -/
def num_participants : ℕ := 4

/-- Represents the number of participants who can finish last -/
def num_can_finish_last : ℕ := 3

/-- Calculates the number of possible race orders given the number of participants
    and the number of participants who can finish last -/
def race_orders (n : ℕ) (k : ℕ) : ℕ := 
  k * Nat.factorial (n - 1)

/-- Theorem stating that the number of possible race orders is 18 -/
theorem race_orders_eq_18 : 
  race_orders num_participants num_can_finish_last = 18 := by
  sorry

#eval race_orders num_participants num_can_finish_last

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_orders_eq_18_l951_95111
