import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_min_value_and_ratio_l537_53777

/-- Given a positive integer m ≥ 3, an increasing arithmetic sequence {a_n},
    and an increasing geometric sequence {b_n} satisfying certain conditions,
    this theorem proves the minimum value of a_m and the ratio of a_1 to b_1 when a_m is minimum. -/
theorem sequence_min_value_and_ratio (m : ℕ) (a b : ℕ → ℝ) (d q : ℝ)
  (hm : m ≥ 3)
  (ha : ∀ n, a (n + 1) = a n + d)
  (hb : ∀ n, b (n + 1) = b n * q)
  (ha_inc : ∀ n, a (n + 1) > a n)
  (hb_inc : ∀ n, b (n + 1) > b n)
  (ha1 : a 1 = q)
  (hb1 : b 1 = d)
  (heq : a m = b m) :
  ∃ (lambda : ℝ),
    lambda ≥ (m^m / (m - 1)^(m - 2))^(1 / (m - 1 : ℝ)) ∧
    (lambda = (m^m / (m - 1)^(m - 2))^(1 / (m - 1 : ℝ)) →
      a 1 / b 1 = (m - 1)^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_min_value_and_ratio_l537_53777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_equals_one_l537_53724

theorem sum_reciprocals_equals_one (m n : ℝ) (h1 : (2 : ℝ)^m = 10) (h2 : (5 : ℝ)^n = 10) : 
  1/m + 1/n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_equals_one_l537_53724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_sufficient_not_necessary_l537_53708

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line (m+1)x+3my+2=0 -/
noncomputable def slope1 (m : ℝ) : ℝ := -(m + 1) / (3 * m)

/-- The slope of the second line (m-2)x+(m+1)y-1=0 -/
noncomputable def slope2 (m : ℝ) : ℝ := -(m - 2) / (m + 1)

/-- The condition that m = 1/2 is sufficient but not necessary for perpendicularity -/
theorem half_sufficient_not_necessary :
  (∀ m : ℝ, m = 1/2 → are_perpendicular (slope1 m) (slope2 m)) ∧
  (∃ m : ℝ, m ≠ 1/2 ∧ are_perpendicular (slope1 m) (slope2 m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_sufficient_not_necessary_l537_53708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_constant_l537_53750

/-- Prove that projecting any vector on y = 3x + 4 onto w results in (-6/5, 2/5) --/
theorem projection_constant (w : ℝ × ℝ) (h : w.1 + 3 * w.2 = 0) :
  ∀ (v : ℝ × ℝ), v.2 = 3 * v.1 + 4 →
    ((v.1 * w.1 + v.2 * w.2) / (w.1^2 + w.2^2)) • w = (-6/5, 2/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_constant_l537_53750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spending_difference_theorem_l537_53754

/-- The difference in spending between Rayden and Lily on birds -/
def spending_difference (lily_ducks lily_geese lily_chickens lily_pigeons 
                         duck_price goose_price chicken_price pigeon_price : ℕ) : ℕ :=
  let rayden_ducks := 3 * lily_ducks
  let rayden_geese := 4 * lily_geese
  let rayden_chickens := 5 * lily_chickens
  let rayden_pigeons := lily_pigeons / 2
  let lily_total := lily_ducks * duck_price + lily_geese * goose_price + 
                    lily_chickens * chicken_price + lily_pigeons * pigeon_price
  let rayden_total := rayden_ducks * duck_price + rayden_geese * goose_price + 
                      rayden_chickens * chicken_price + rayden_pigeons * pigeon_price
  rayden_total - lily_total

/-- Theorem stating the difference in spending between Rayden and Lily -/
theorem spending_difference_theorem :
  spending_difference 20 10 5 30 15 20 10 5 = 1325 := by
  -- Unfold the definition of spending_difference
  unfold spending_difference
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spending_difference_theorem_l537_53754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edward_hourly_rate_l537_53781

/-- Calculates the hourly rate given total earnings and hours worked -/
noncomputable def calculate_hourly_rate (total_earnings : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ) : ℝ :=
  total_earnings / (regular_hours + 2 * overtime_hours)

theorem edward_hourly_rate :
  let regular_hours : ℝ := 40
  let overtime_hours : ℝ := 5
  let total_earnings : ℝ := 210
  let hourly_rate := calculate_hourly_rate total_earnings regular_hours overtime_hours
  hourly_rate = 4.2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edward_hourly_rate_l537_53781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_swept_equals_circle_area_l537_53729

/-- The area swept by a side of a rectangle when rotated 90 degrees around a corner -/
def AreaSwept90Deg (A B C D : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- Predicate to check if four points form a rectangle -/
def IsRectangle (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Given a rectangle ABCD, prove that the area swept by side CD when rotated 90° clockwise around point A is equal to the area of a circle with radius 1/2 AB. -/
theorem area_swept_equals_circle_area 
  (A B C D : EuclideanSpace ℝ (Fin 2)) 
  (h_rectangle : IsRectangle A B C D) :
  AreaSwept90Deg A B C D = π * (1/2 * ‖B - A‖) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_swept_equals_circle_area_l537_53729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_denominators_is_seven_l537_53707

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def repeating_decimal_to_fraction (a b : ℕ) : ℚ := (100 * a + b : ℚ) / 999

def count_distinct_denominators (S : Set ℕ) : ℕ := 
  Finset.card (Finset.image 
    (λ n => (repeating_decimal_to_fraction n.1 n.2).den) 
    (Finset.filter (λ p => p.1 ≠ 0 ∨ p.2 ≠ 0) 
      (Finset.product (Finset.range 10) (Finset.range 10))))

theorem count_denominators_is_seven :
  ∀ a b : ℕ, is_digit a → is_digit b → (a ≠ 0 ∨ b ≠ 0) →
  count_distinct_denominators {3, 9, 27, 37, 111, 333, 999} = 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_denominators_is_seven_l537_53707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_passing_students_l537_53766

theorem percentage_of_passing_students (total : ℕ) (failed : ℕ) : 
  total = 804 → failed = 201 → ((total - failed : ℚ) / total * 100).floor = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_passing_students_l537_53766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mom_food_per_meal_is_correct_l537_53763

/-- Represents the food consumption for dogs in Joy's foster care --/
structure DogFood where
  totalFood : ℚ
  days : ℕ
  puppyCount : ℕ
  puppyMealsPerDay : ℕ
  puppyFoodPerMeal : ℚ
  momMealsPerDay : ℕ

/-- Calculates the amount of food the mom foster dog eats in one meal --/
def momFoodPerMeal (df : DogFood) : ℚ :=
  let totalPuppyFood := df.puppyCount * df.puppyMealsPerDay * df.puppyFoodPerMeal * df.days
  let totalMomFood := df.totalFood - totalPuppyFood
  let totalMomMeals := df.momMealsPerDay * df.days
  totalMomFood / totalMomMeals

/-- Theorem stating that the mom foster dog eats 1.5 cups of food in one meal --/
theorem mom_food_per_meal_is_correct (df : DogFood) 
    (h1 : df.totalFood = 57)
    (h2 : df.days = 6)
    (h3 : df.puppyCount = 5)
    (h4 : df.puppyMealsPerDay = 2)
    (h5 : df.puppyFoodPerMeal = 1/2)
    (h6 : df.momMealsPerDay = 3) :
    momFoodPerMeal df = 3/2 := by
  sorry

#eval momFoodPerMeal { totalFood := 57, days := 6, puppyCount := 5, 
                       puppyMealsPerDay := 2, puppyFoodPerMeal := 1/2, momMealsPerDay := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mom_food_per_meal_is_correct_l537_53763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payroll_from_tax_l537_53759

/-- Represents the special municipal payroll tax system -/
structure TaxSystem where
  threshold : ℝ
  rate : ℝ

/-- Calculates the tax for a given payroll under the special municipal payroll tax system -/
noncomputable def calculate_tax (system : TaxSystem) (payroll : ℝ) : ℝ :=
  if payroll ≤ system.threshold then 0
  else (payroll - system.threshold) * system.rate

/-- The theorem to be proved -/
theorem payroll_from_tax (tax_paid : ℝ) : 
  let system : TaxSystem := { threshold := 200000, rate := 0.002 }
  tax_paid = 400 → 
  ∃ (payroll : ℝ), calculate_tax system payroll = tax_paid ∧ payroll = 400000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_payroll_from_tax_l537_53759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_2_pow_minus_x_l537_53774

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x-1)
def domain_f_x_minus_1 : Set ℝ := Set.Ioo 1 3

-- State the theorem
theorem domain_of_f_2_pow_minus_x 
  (h : ∀ x ∈ domain_f_x_minus_1, f (x - 1) = f (x - 1)) :
  {x : ℝ | f (2^(-x)) = f (2^(-x))} = Set.Ioi (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_2_pow_minus_x_l537_53774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_slope_range_l537_53780

-- Define the ellipse C
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define the maximum distance from any point on C to F
def max_distance : ℝ := 3

-- Define a line passing through F with slope k
noncomputable def line (k x : ℝ) : ℝ := k * (x - F.1)

-- Define the product of distances FA and FB
noncomputable def distance_product (k : ℝ) : ℝ := 9 * (1 + k^2) / (3 + 4 * k^2)

theorem ellipse_and_slope_range :
  (∀ x y, ellipse x y → ((x - F.1)^2 + (y - F.2)^2).sqrt ≤ max_distance) →
  (∀ m : ℝ, ∃ x y, ellipse x y ∧ (1 + 3*m) * x - (3 - 2*m) * y - (1 + 3*m) = 0) →
  (∀ k : ℝ, (12/5 ≤ distance_product k ∧ distance_product k ≤ 18/7) → 
    (k ∈ Set.Icc (-Real.sqrt 3) (-1) ∪ Set.Icc 1 (Real.sqrt 3))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_slope_range_l537_53780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_2_implies_reciprocal_sin_2alpha_l537_53736

theorem tan_alpha_2_implies_reciprocal_sin_2alpha (α : Real) : 
  Real.tan α = 2 → 1 / Real.sin (2 * α) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_2_implies_reciprocal_sin_2alpha_l537_53736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l537_53734

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a hyperbola with equation x²/9 - y²/b² = 1 (b > 0),
    if a point P on the hyperbola satisfies |PF₁| = 5, then |PF₂| = 11 -/
theorem hyperbola_focal_distance
  (h : Hyperbola)
  (p f1 f2 : Point)
  (h_eq : h.a = 3)
  (h_on_hyp : on_hyperbola h p)
  (h_dist_f1 : distance p f1 = 5) :
  distance p f2 = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l537_53734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_tangent_lines_equation_trajectory_equation_l537_53758

-- Define the circle C
def CircleC (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define the center and radius
def Center : ℝ × ℝ := (-1, 2)
noncomputable def Radius : ℝ := Real.sqrt 2

-- Define the tangent line equations
def TangentLine1 (x y : ℝ) : Prop := x + y + 1 = 0
def TangentLine2 (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the trajectory equation
def Trajectory (x y : ℝ) : Prop := 2*x - 4*y + 3 = 0

-- Theorem statements
theorem circle_center_and_radius :
  (∀ x y, CircleC x y ↔ (x - Center.1)^2 + (y - Center.2)^2 = Radius^2) := by sorry

theorem tangent_lines_equation :
  (∀ x y, (TangentLine1 x y ∨ TangentLine2 x y) →
    (∃ t, x = t ∧ y = t) ∧
    (x ≠ 0 ∨ y ≠ 0) ∧
    ((x - Center.1)^2 + (y - Center.2)^2 = (Radius + 1)^2)) := by sorry

theorem trajectory_equation :
  (∀ x y, Trajectory x y →
    ∃ m, (x - Center.1)^2 + (y - Center.2)^2 = (Radius^2 + m^2) ∧
         x^2 + y^2 = m^2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_radius_tangent_lines_equation_trajectory_equation_l537_53758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_alpha_l537_53794

theorem sin_double_alpha (α : ℝ) (h1 : Real.tan α = -1/2) (h2 : α ∈ Set.Ioo 0 π) :
  Real.sin (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_alpha_l537_53794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloringSchemes_correct_l537_53756

/-- The number of different coloring schemes for n connected regions with m colors,
    where any two regions separated by one region have different colors. -/
def coloringSchemes (m n : ℕ) : ℤ :=
  if n % 2 = 0 then
    ((m - 1)^(n / 2) + (-1 : ℤ)^(n / 2) * (m - 1))^2
  else
    (m - 1)^n + (-1 : ℤ)^n * (m - 1)

/-- Theorem stating the number of different coloring schemes for n connected regions with m colors. -/
theorem coloringSchemes_correct (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 4) :
  coloringSchemes m n = 
    if n % 2 = 0 then
      ((m - 1)^(n / 2) + (-1 : ℤ)^(n / 2) * (m - 1))^2
    else
      (m - 1)^n + (-1 : ℤ)^n * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloringSchemes_correct_l537_53756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l537_53772

open Real

noncomputable def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = π ∧ a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem triangle_properties 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : triangle A B C a b c) 
  (h_eq : cos A / (1 + sin A) = sin (2 * B) / (1 + cos (2 * B))) :
  (C = 2 * π / 3 → B = π / 6) ∧
  (∃ (min : ℝ), min = 4 * sqrt 2 - 5 ∧ 
    ∀ (a' b' c' : ℝ), triangle A B C a' b' c' → 
      (a' ^ 2 + b' ^ 2) / c' ^ 2 ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l537_53772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_artworks_class_l537_53749

/-- The variance of artworks per student in a class -/
noncomputable def variance_artworks (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (avg_boy : ℝ) (var_boy : ℝ) (avg_girl : ℝ) (var_girl : ℝ) : ℝ :=
  ((boys : ℝ) / total_students) * (var_boy + (avg_boy - ((boys : ℝ) * avg_boy + (girls : ℝ) * avg_girl) / total_students)^2) +
  ((girls : ℝ) / total_students) * (var_girl + (avg_girl - ((boys : ℝ) * avg_boy + (girls : ℝ) * avg_girl) / total_students)^2)

/-- Theorem: The variance of artworks per student is 38/5 given the specified conditions -/
theorem variance_artworks_class : 
  variance_artworks 25 10 15 25 1 30 2 = 38/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_artworks_class_l537_53749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_product_l537_53753

theorem polynomial_root_product (d e f : ℝ) : 
  (∃ Q : ℝ → ℝ, Q = λ x ↦ x^3 + d*x^2 + e*x + f) →
  (∀ x, (x^3 + d*x^2 + e*x + f = 0) ↔ (x = Real.cos (π/9) ∨ x = Real.cos (4*π/9) ∨ x = Real.cos (7*π/9))) →
  d * e * f = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_product_l537_53753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l537_53722

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x^(1/x)

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := (x^(1/x)) * ((1 - Real.log x) / (x^2))

-- State the theorem
theorem monotonically_decreasing_interval :
  ∀ x, x > Real.exp 1 → HasDerivAt f (f' x) x ∧ f' x < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l537_53722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_f_is_increasing_k_range_l537_53765

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Part 1: If f is odd, then a = 1
theorem odd_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a x = -f a (-x)) → a = 1 := by sorry

-- Part 2: For any a, f is increasing on ℝ
theorem f_is_increasing (a : ℝ) :
  ∀ x y, x < y → f a x < f a y := by sorry

-- Part 3: If f is odd and satisfies the inequality, then k < -1 + 2√2
theorem k_range (a k : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (∀ x, f a (k * 3^x) + f a (3^x - 9^x - 2) < 0) →
  k < -1 + 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_f_is_increasing_k_range_l537_53765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_tangent_line_through_Q_l537_53727

noncomputable def curve (x : ℝ) : ℝ := 1 / x

noncomputable def curve_derivative (x : ℝ) : ℝ := -1 / (x^2)

def P : ℝ × ℝ := (1, 1)

def Q : ℝ × ℝ := (1, 0)

noncomputable def tangent_line (x₀ y₀ k : ℝ) (x : ℝ) : ℝ := k * (x - x₀) + y₀

theorem tangent_line_at_P :
  ∃ (k : ℝ), (λ x ↦ tangent_line P.1 P.2 k x) = λ x ↦ -x + 2 :=
sorry

theorem tangent_line_through_Q :
  ∃ (x₀ : ℝ), x₀ ≠ 0 ∧
  curve x₀ = 1 / x₀ ∧
  (λ x ↦ tangent_line x₀ (curve x₀) (curve_derivative x₀) x) = λ x ↦ -4 * x + 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_tangent_line_through_Q_l537_53727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_credit_percentage_approx_36_percent_l537_53746

/-- The percentage of consumer installment credit accounted for by automobile installment credit -/
noncomputable def auto_credit_percentage (finance_company_credit : ℝ) (total_consumer_credit : ℝ) : ℝ :=
  (3 * finance_company_credit / total_consumer_credit) * 100

/-- Theorem stating that the automobile installment credit percentage is approximately 36% -/
theorem auto_credit_percentage_approx_36_percent 
  (finance_company_credit : ℝ) 
  (total_consumer_credit : ℝ) 
  (h1 : finance_company_credit = 57)
  (h2 : total_consumer_credit = 475) : 
  ∃ ε > 0, |auto_credit_percentage finance_company_credit total_consumer_credit - 36| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_credit_percentage_approx_36_percent_l537_53746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_9_or_10_l537_53761

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  first_term_negative : a 1 < 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem min_sum_at_9_or_10 (seq : ArithmeticSequence) 
  (h : sum_n seq 3 = sum_n seq 16) :
  ∃ n : ℕ, (n = 9 ∨ n = 10) ∧
    ∀ k : ℕ, sum_n seq n ≤ sum_n seq k :=
  sorry

#check min_sum_at_9_or_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_9_or_10_l537_53761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_duty_sequences_l537_53714

/-- Represents the number of days in the week for scheduling --/
def num_days : ℕ := 5

/-- Represents the number of students to be scheduled --/
def num_students : ℕ := 5

/-- Represents the number of days the restricted student can be scheduled --/
def restricted_student_days : ℕ := 3

/-- Calculates the number of duty sequences --/
def duty_sequences : ℕ :=
  restricted_student_days * Nat.factorial (num_students - 1)

theorem correct_duty_sequences :
  duty_sequences = 72 := by
  -- Expand the definition of duty_sequences
  unfold duty_sequences
  -- Simplify the expression
  simp [num_students, restricted_student_days]
  -- The proof is completed
  rfl

#eval duty_sequences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_duty_sequences_l537_53714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_t_squared_l537_53716

/-- A hyperbola centered at the origin, opening vertically, and passing through specific points -/
structure VerticalHyperbola where
  toSet : Set (ℝ × ℝ)
  /-- The hyperbola passes through (4, -3) -/
  point1 : (4, -3) ∈ toSet
  /-- The hyperbola passes through (0, -2) -/
  point2 : (0, -2) ∈ toSet
  /-- The hyperbola passes through (2, t) for some real t -/
  point3 : ∃ t : ℝ, (2, t) ∈ toSet
  /-- The hyperbola opens vertically -/
  opens_vertically : True  -- This is a placeholder; we'd need to define this properly
  /-- The hyperbola is centered at the origin -/
  center : (0, 0) ∈ toSet

/-- The theorem stating that t^2 = 8 for the given hyperbola -/
theorem hyperbola_t_squared (h : VerticalHyperbola) : 
  ∃ t : ℝ, (2, t) ∈ h.toSet ∧ t^2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_t_squared_l537_53716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_same_calendar_year_l537_53720

/-- Represents a year in the Gregorian calendar. -/
structure Year where
  value : Nat

/-- Returns true if the given year is a leap year, false otherwise. -/
def isLeapYear (y : Year) : Bool :=
  let yv := y.value
  (yv % 4 == 0 && yv % 100 != 0) || (yv % 400 == 0)

/-- Returns the day of the week (0-6) for January 1st of the given year. -/
noncomputable def dayOfWeek (y : Year) : Nat :=
  sorry -- Implementation details omitted for simplicity

/-- The theorem stating that 2088 is the next year after 2060 with the same calendar. -/
theorem next_same_calendar_year : 
  (isLeapYear ⟨2060⟩ = true) → 
  (dayOfWeek ⟨2060⟩ = 5) → -- 5 represents Friday
  (∀ y : Year, 2060 < y.value → y.value < 2088 → 
    (isLeapYear y ≠ isLeapYear ⟨2060⟩ ∨ dayOfWeek y ≠ dayOfWeek ⟨2060⟩)) →
  (isLeapYear ⟨2088⟩ = isLeapYear ⟨2060⟩) ∧ 
  (dayOfWeek ⟨2088⟩ = dayOfWeek ⟨2060⟩) :=
by
  sorry

#check next_same_calendar_year

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_same_calendar_year_l537_53720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_value_l537_53762

noncomputable def f (x : ℝ) : ℝ := 5 * x^3 - 1/x + 3
noncomputable def g (k c x : ℝ) : ℝ := x^2 - k*x + c

theorem find_c_value (k : ℝ) (h1 : k = 1) :
  ∃ c : ℝ, f 2 - g k c 2 = 2 ∧ c = 38.5 :=
by
  -- We use 'sorry' to skip the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_value_l537_53762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_mean_and_std_dev_l537_53792

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ := 
  (xs.map (λ x => (x - mean xs) ^ 2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ := 
  Real.sqrt (variance xs)

theorem transform_mean_and_std_dev 
  (xs : List ℝ) 
  (h_mean : mean xs = 10) 
  (h_std_dev : standardDeviation xs = 2) : 
  let ys := xs.map (λ x => 2 * x - 1)
  mean ys = 19 ∧ standardDeviation ys = 4 := by
  sorry

#check transform_mean_and_std_dev

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_mean_and_std_dev_l537_53792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_cycling_speed_l537_53730

/-- Proves that B's cycling speed is 20 kmph given the problem conditions -/
theorem b_cycling_speed (a_speed : ℝ) (b_start_delay : ℝ) (catch_up_distance : ℝ)
  (h1 : a_speed = 10)
  (h2 : b_start_delay = 10)
  (h3 : catch_up_distance = 200) :
  let a_initial_distance := a_speed * b_start_delay
  let remaining_distance := catch_up_distance - a_initial_distance
  let catch_up_time := remaining_distance / a_speed
  let b_speed := catch_up_distance / catch_up_time
  b_speed = 20 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_cycling_speed_l537_53730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_share_is_6000_l537_53755

/-- Represents the capital of a partner -/
structure Capital where
  amount : ℚ
  deriving Repr

/-- Represents a business partnership -/
structure Partnership where
  a : Capital
  b : Capital
  c : Capital
  total_profit : ℚ
  deriving Repr

/-- Conditions for the partnership -/
def valid_partnership (p : Partnership) : Prop :=
  p.a.amount = 3 * p.b.amount / 2 ∧
  p.b.amount = 4 * p.c.amount ∧
  p.total_profit = 16500

/-- Calculate partner b's share of the profit -/
def b_share (p : Partnership) : ℚ :=
  (p.b.amount / (p.a.amount + p.b.amount + p.c.amount)) * p.total_profit

/-- Theorem stating that b's share is 6000 given the conditions -/
theorem b_share_is_6000 (p : Partnership) (h : valid_partnership p) :
  b_share p = 6000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_share_is_6000_l537_53755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_lower_bound_equality_infinite_strict_inequality_infinite_l537_53702

/-- Operation that replaces two numbers with their geometric mean plus 1/2 -/
noncomputable def operation (a b : ℝ) : ℝ := Real.sqrt ((a * b + 1) / 2)

/-- The final number after n-1 operations on n initial 2's -/
noncomputable def finalNumber (n : ℕ) : ℝ :=
  sorry  -- Definition would go here, but we skip it as per instructions

/-- Theorem stating the lower bound of the final number -/
theorem final_number_lower_bound (n : ℕ) (h : n > 0) : 
  finalNumber n ≥ Real.sqrt ((n + 3) / n) := by
  sorry

/-- There are infinitely many n for which equality holds -/
theorem equality_infinite : 
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, finalNumber n = Real.sqrt ((n + 3) / n) := by
  sorry

/-- There are infinitely many n for which the inequality is strict -/
theorem strict_inequality_infinite : 
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, finalNumber n > Real.sqrt ((n + 3) / n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_lower_bound_equality_infinite_strict_inequality_infinite_l537_53702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l537_53743

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 1 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (2, -1)
def radius1 : ℝ := 2
def center2 : ℝ × ℝ := (-2, 2)
def radius2 : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ :=
  Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)

-- Theorem stating that the circles are externally tangent
theorem circles_externally_tangent :
  distance_between_centers = radius1 + radius2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l537_53743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_proof_l537_53740

/-- The eccentricity of an ellipse described by parametric equations x = 3cos(φ) and y = √5sin(φ) -/
noncomputable def ellipse_eccentricity : ℝ := 2/3

/-- Parametric equations of the ellipse -/
noncomputable def ellipse_param (φ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos φ, Real.sqrt 5 * Real.sin φ)

theorem ellipse_eccentricity_proof :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (∀ φ, (ellipse_param φ).1^2 / a^2 + (ellipse_param φ).2^2 / b^2 = 1) ∧
  c^2 = a^2 - b^2 ∧
  ellipse_eccentricity = c / a := by
  sorry

#check ellipse_eccentricity_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_proof_l537_53740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_exists_unique_l537_53748

/-- A triangle with given altitude, angle bisector, and incircle radius -/
structure SpecialTriangle where
  m_a : ℝ  -- altitude from vertex A
  f_a : ℝ  -- length of angle bisector from vertex A
  ρ   : ℝ  -- radius of inscribed circle
  h_positive : 0 < m_a ∧ 0 < f_a ∧ 0 < ρ

/-- The existence and uniqueness condition for the special triangle -/
def SpecialTriangle.existsUnique (t : SpecialTriangle) : Prop :=
  t.f_a > t.m_a ∧ t.ρ < t.m_a / 2

/-- Theorem stating the existence and uniqueness of the special triangle -/
theorem special_triangle_exists_unique (t : SpecialTriangle) :
  ∃! (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧  -- positive side lengths
    ∃ (h_a h_b h_c : ℝ),     -- altitudes
      0 < h_a ∧ 0 < h_b ∧ 0 < h_c ∧
      h_a = t.m_a ∧          -- given altitude
      a * h_a = b * h_b ∧ b * h_b = c * h_c ∧  -- area equality
      ∃ (α β γ : ℝ),         -- angles
        0 < α ∧ 0 < β ∧ 0 < γ ∧
        α + β + γ = Real.pi ∧      -- angle sum
        (a / Real.sin α = b / Real.sin β) ∧ (b / Real.sin β = c / Real.sin γ) ∧  -- sine law
        ∃ (f_α : ℝ),         -- angle bisector
          0 < f_α ∧
          f_α = t.f_a ∧      -- given angle bisector
          2 * f_α * b * c = a * (b + c) ∧  -- angle bisector theorem
        ∃ (s : ℝ),           -- semiperimeter
          s = (a + b + c) / 2 ∧
          t.ρ = (a + b - c) * (b + c - a) * (c + a - b) / (4 * s)  -- inradius formula
  ↔ t.existsUnique :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_exists_unique_l537_53748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_tan_l537_53752

theorem triangle_angle_tan (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- Angles are positive
  A + B + C = Real.pi →  -- Sum of angles in a triangle
  2 * B = A + C →  -- Arithmetic sequence condition
  Real.tan (A + C) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_tan_l537_53752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l537_53711

open Complex

theorem complex_problem (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := ((1 + i)^2 + 3*(1 - i)) / (2 + i)
  (abs z = Real.sqrt 2) ∧
  (∃ (a b : ℝ), z^2 + a*z + b = 1 + i ∧ a = -3 ∧ b = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l537_53711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l537_53767

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | |x - 1| ≤ 2}

theorem intersection_M_N : M ∩ N = Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l537_53767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l537_53732

theorem quadratic_root_form (a b c m n p : ℤ) : 
  a > 0 ∧ b < 0 ∧ c < 0 →
  (3 * a * x^2 + b * x + c = 0) →
  (∃ x : ℚ, x = (m + Int.sqrt n) / p ∨ x = (m - Int.sqrt n) / p) →
  m > 0 ∧ n > 0 ∧ p > 0 →
  Int.gcd m (Int.gcd n p) = 1 →
  n = 100 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l537_53732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tulip_field_length_l537_53728

/-- The length of a rectangle-shaped tulip flower field in centimeters, given its width and a relationship to its length. -/
theorem tulip_field_length : ℝ := by
  let tape_measure : ℝ := 250 -- Width in centimeters
  let field_length : ℝ := 5 * tape_measure + 80 -- Length in centimeters
  have h : field_length = 1330 := by
    -- Proof goes here
    sorry
  exact field_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tulip_field_length_l537_53728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orphanage_donation_percentage_l537_53731

/-- Proves that the percentage donated to the orphan house is 50% of the remaining amount after distributing to children and wife. -/
theorem orphanage_donation_percentage 
  (total_income : ℝ)
  (children_percentage : ℝ)
  (wife_percentage : ℝ)
  (num_children : ℕ)
  (remaining_amount : ℝ)
  (h1 : total_income = 800000)
  (h2 : children_percentage = 0.2)
  (h3 : wife_percentage = 0.3)
  (h4 : num_children = 3)
  (h5 : remaining_amount = 40000)
  : (total_income - (children_percentage * ↑num_children + wife_percentage) * total_income) / 2 = remaining_amount := by
  sorry

-- Remove the #eval line as it's not necessary for building and might cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orphanage_donation_percentage_l537_53731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_coprimes_to_2001_l537_53738

theorem count_coprimes_to_2001 : 
  (Finset.filter (fun n => Nat.gcd n 2001 = 1) (Finset.range 2000)).card = 1232 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_coprimes_to_2001_l537_53738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_area_calculation_l537_53733

theorem gray_area_calculation 
  (small_diameter : ℝ) 
  (radius_ratio : ℝ) 
  (h1 : small_diameter = 4)
  (h2 : radius_ratio = 3) :
  let gray_area := 32 * Real.pi
  gray_area = 32 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_area_calculation_l537_53733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_overlapping_sectors_l537_53737

/-- The area of the shaded region in a circle with overlapping sectors -/
theorem shaded_area_overlapping_sectors (r : ℝ) (h_r : r = 15) : 
  (2 * (π * r^2 / 4) + (π * r^2 / 6) - (π * r^2 / 8)) = 121.875 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_overlapping_sectors_l537_53737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_always_wins_l537_53786

/-- Represents a player in the game -/
inductive Player
| Bela
| Jenn

/-- Represents the game state -/
structure GameState where
  n : ℕ
  choices : List ℝ

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : ℝ) : Prop :=
  0 ≤ move ∧ move ≤ state.n ∧ ∀ c ∈ state.choices, |move - c| > 1

/-- Defines the optimal strategy for Bela -/
noncomputable def optimalStrategy (state : GameState) : ℝ :=
  state.n / 2

/-- Theorem stating that Bela always wins using the optimal strategy -/
theorem bela_always_wins (n : ℕ) (h : n > 4) :
  ∃ (strategy : GameState → ℝ),
    ∀ (game : List ℝ),
      (∀ i : ℕ, i < game.length → isValidMove ⟨n, game.take i⟩ (game.get? i).get!) →
      (game.length % 2 = 0 → 
        ¬isValidMove ⟨n, game⟩ (optimalStrategy ⟨n, game⟩)) ∧
      (game.length % 2 = 1 → 
        ∃ m, isValidMove ⟨n, game⟩ m ∧ 
             ¬isValidMove ⟨n, game ++ [m]⟩ (optimalStrategy ⟨n, game ++ [m]⟩)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_always_wins_l537_53786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l537_53719

noncomputable def f (x : ℝ) := Real.sqrt (2 * Real.cos x + 1)

theorem domain_of_f :
  ∀ x : ℝ, f x ∈ Set.range f ↔ ∃ k : ℤ, 2 * π * ↑k - 2 * π / 3 ≤ x ∧ x ≤ 2 * π * ↑k + 2 * π / 3 :=
by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l537_53719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_circumcenter_locus_l537_53790

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a circle in a 2D plane -/
structure Circle2D where
  center : Point2D
  radius : ℝ

/-- Represents a Thales circle constructed on a diameter -/
noncomputable def thalesCircle (a b : Point2D) : Circle2D :=
  { center := { x := (a.x + b.x) / 2, y := (a.y + b.y) / 2 },
    radius := Real.sqrt ((b.x - a.x)^2 + (b.y - a.y)^2) / 2 }

/-- The locus of centers of circumcircles -/
def circumcenterLocus (k l m : Point2D) : Set Circle2D :=
  {c | ∃ (v v2 : Point2D), c = thalesCircle v m ∨ c = thalesCircle v2 m}

theorem isosceles_right_triangle_circumcenter_locus 
  (k l m : Point2D) (hDistinct : k ≠ l ∧ l ≠ m ∧ m ≠ k) :
  ∃ (v v2 : Point2D),
    circumcenterLocus k l m = 
      {c | c = thalesCircle v m ∨ c = thalesCircle v2 m} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_circumcenter_locus_l537_53790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_circle_l537_53798

/-- Given a circle with center (2,-1) and a point (7,3) on the circle,
    the slope of the tangent line at (7,3) is -5/4 -/
theorem tangent_slope_circle (center point : ℝ × ℝ) : 
  center = (2, -1) →
  point = (7, 3) →
  (point.2 - center.2) / (point.1 - center.1) = 4/5 →
  -1 / ((point.2 - center.2) / (point.1 - center.1)) = -5/4 := by
  sorry

#check tangent_slope_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_circle_l537_53798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_throws_for_repeated_sum_l537_53747

/-- Represents a fair six-sided die -/
def Die : Type := Fin 6

/-- The sum of four dice rolls -/
def DiceSum := Nat

/-- The minimum possible sum when rolling four dice -/
def min_sum : Nat := 4

/-- The maximum possible sum when rolling four dice -/
def max_sum : Nat := 24

/-- The number of distinct possible sums when rolling four dice -/
def distinct_sums : Nat := max_sum - min_sum + 1

theorem min_throws_for_repeated_sum :
  ∃ (n : Nat), n = distinct_sums + 1 ∧
  (∀ (m : Nat), m < n → ∃ (rolls : Fin m → Fin n),
    Function.Injective rolls) ∧
  (∀ (rolls : Fin (n + 1) → Fin n),
    ¬Function.Injective rolls) := by
  sorry

#eval distinct_sums

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_throws_for_repeated_sum_l537_53747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_day_matchup_l537_53768

structure Tournament where
  players : Finset Char
  days : Fin 3 → Finset (Char × Char)
  all_play : ∀ p q : Char, p ∈ players → q ∈ players → p ≠ q → ∃ d, (p, q) ∈ days d ∨ (q, p) ∈ days d
  one_game_per_day : ∀ d p, (∃ q, (p, q) ∈ days d ∨ (q, p) ∈ days d) → 
    ∀ r, (p, r) ∈ days d ∨ (r, p) ∈ days d → r = q
  two_games_per_day : ∀ d, (days d).card = 2

def go_tournament : Tournament where
  players := {'A', 'B', 'C', 'D'}
  days := fun d => match d with
    | 0 => {('A', 'C'), ('B', 'D')}
    | 1 => {('C', 'D'), ('A', 'B')}
    | 2 => {('B', 'C'), ('A', 'D')}
  all_play := by sorry
  one_game_per_day := by sorry
  two_games_per_day := by sorry

theorem third_day_matchup (t : Tournament) 
  (h1 : ('A', 'C') ∈ t.days 0 ∨ ('C', 'A') ∈ t.days 0)
  (h2 : ('C', 'D') ∈ t.days 1 ∨ ('D', 'C') ∈ t.days 1)
  (h3 : ∃ p, ('B', p) ∈ t.days 2 ∨ (p, 'B') ∈ t.days 2) :
  ('B', 'C') ∈ t.days 2 ∨ ('C', 'B') ∈ t.days 2 := by
  sorry

#check go_tournament
#check third_day_matchup

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_day_matchup_l537_53768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l537_53789

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (-1 + 3*t, 2 - 4*t)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (y - 2)^2 - x^2 = 1

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    A = line_l t₁ ∧ B = line_l t₂ ∧
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    t₁ ≠ t₂

-- Define point P
def point_P : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem intersection_properties 
  (A B : ℝ × ℝ) 
  (h : intersection_points A B) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ‖A - B‖ = 10 * Real.sqrt 23 / 7 ∧
  ‖point_P - C‖ = 15 / 7 := by
  sorry

#check intersection_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l537_53789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_order_l537_53791

-- Define the inverse proportion function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (-m^2 - 2) / x

-- Define the theorem
theorem inverse_proportion_point_order (m a b c : ℝ) :
  f m a = -1 →
  f m b = 2 →
  f m c = 3 →
  a > c ∧ c > b :=
by
  -- The proof is skipped for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_order_l537_53791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_curvature_l537_53775

/-- A polyhedron with four triangular faces and one square face -/
structure Polyhedron :=
  (triangular_faces : Fin 4 → Face)
  (square_face : Face)

/-- A face of a polyhedron -/
structure Face :=
  (angles : List ℝ)

/-- The curvature of a vertex is 2π minus the sum of its face angles -/
noncomputable def vertex_curvature (angles : List ℝ) : ℝ :=
  2 * Real.pi - angles.sum

/-- The total curvature of a polyhedron is the sum of its vertex curvatures -/
noncomputable def total_curvature (p : Polyhedron) : ℝ :=
  let base_vertices := 4
  let apex_vertex := 1
  let base_curvature := base_vertices * (vertex_curvature [Real.pi / 2, Real.pi / 2, Real.pi / 2])
  let apex_curvature := apex_vertex * (vertex_curvature [Real.pi, Real.pi, Real.pi, Real.pi])
  base_curvature + apex_curvature

/-- The total curvature of a polyhedron with four triangular faces and one square face is 4π -/
theorem polyhedron_curvature (p : Polyhedron) : total_curvature p = 4 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_curvature_l537_53775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_of_skew_lines_l537_53710

-- Define a 3D space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [Fact (finrank ℝ V = 3)]

-- Define a line in 3D space
def Line (p q : V) : Set V := {r : V | ∃ t : ℝ, r = p + t • (q - p)}

-- Define skew lines
def Skew (l1 l2 : Set V) : Prop :=
  ∀ p1 p2 q1 q2 : V, l1 = Line V p1 q1 → l2 = Line V p2 q2 →
    ¬∃ r : V, r ∈ l1 ∧ r ∈ l2

-- Define intersecting lines
def Intersect (l1 l2 : Set V) : Prop :=
  ∃ r : V, r ∈ l1 ∧ r ∈ l2

-- Theorem statement
theorem intersecting_lines_of_skew_lines (l1 l2 l3 l4 : Set V)
  (h_skew : Skew V l1 l2)
  (h_int1 : Intersect V l1 l3)
  (h_int2 : Intersect V l2 l4) :
  ¬ (¬ ∃ l : Set V, l = l3 ∧ l = l4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_of_skew_lines_l537_53710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_ellipse_l537_53773

/-- The ellipse C defined by x²/5 + y² = 1 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 5 + p.2^2 = 1}

/-- The upper vertex B of the ellipse C -/
def B : ℝ × ℝ := (0, 1)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The maximum distance between B and any point P on the ellipse C -/
theorem max_distance_on_ellipse :
  ∃ (max_dist : ℝ), max_dist = 5/2 ∧
  ∀ P ∈ C, distance P B ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_ellipse_l537_53773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l537_53704

-- Define the lines that form the quadrilateral
def line1 : ℝ → ℝ := λ _ => 8
def line2 : ℝ → ℝ := λ x => x + 3
def line3 : ℝ → ℝ := λ x => -x + 3
def line4 : ℝ := 5

-- Define the quadrilateral
def quadrilateral : Set (ℝ × ℝ) :=
  {p | (p.1 ≤ line4 ∧ p.2 ≤ line1 p.1) ∧
       (p.2 ≥ line2 p.1 ∨ p.2 ≥ line3 p.1)}

-- State the theorem
theorem quadrilateral_area : MeasureTheory.volume quadrilateral = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l537_53704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_counterexamples_l537_53757

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ := sorry

/-- A function that checks if a positive integer contains the digit 0 -/
def contains_zero (n : ℕ+) : Prop := sorry

/-- The set of positive integers whose digits sum to 5 and don't contain 0 -/
def S : Set ℕ+ := {n | digit_sum n = 5 ∧ ¬contains_zero n}

/-- The set of composite numbers in S -/
def counterexamples : Finset ℕ+ := sorry

/-- Theorem stating that there are 10 counterexamples -/
theorem count_counterexamples : counterexamples.card = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_counterexamples_l537_53757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polyhedron_to_cube_exists_l537_53764

/-- A face of a polyhedron -/
structure Face where
  sides : ℕ

/-- A convex polyhedron with only triangular and hexagonal faces -/
structure SpecialPolyhedron where
  convex : Bool
  faces : Set Face
  only_tri_hex : ∀ f ∈ faces, f.sides = 3 ∨ f.sides = 6

/-- Represents a cube -/
structure Cube where
  side_length : ℝ

/-- Represents a cut of a polyhedron into two parts -/
structure Cut (P : SpecialPolyhedron) where
  part1 : Set Face
  part2 : Set Face
  valid_cut : part1 ∪ part2 = P.faces ∧ part1 ∩ part2 = ∅

/-- Function to reassemble cut parts into a new shape -/
noncomputable def reassemble (P : SpecialPolyhedron) (c : Cut P) : Cube :=
  sorry

/-- Theorem stating the existence of a special polyhedron that can be cut and reassembled into a cube -/
theorem special_polyhedron_to_cube_exists : 
  ∃ (P : SpecialPolyhedron) (c : Cut P), reassemble P c = Cube.mk 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polyhedron_to_cube_exists_l537_53764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_problem_l537_53715

theorem fraction_sum_problem (a b : ℕ) (h1 : Nat.Coprime a b) 
  (h2 : (3*a)/(5*b) + (2*a)/(9*b) + (4*a)/(15*b) = 28/45) : 
  5*b + 9*b + 15*b = 203 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_problem_l537_53715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_and_set_l537_53735

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (x + Real.pi/3)

theorem f_minimum_value_and_set :
  ∃ (min : ℝ) (S : Set ℝ),
    (∀ x, f x ≥ min) ∧
    (S = {x | f x = min}) ∧
    (min = -Real.sqrt 3) ∧
    (S = {x : ℝ | ∃ k : ℤ, x = 2 * ↑k * Real.pi - 2 * Real.pi / 3}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_and_set_l537_53735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l537_53771

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Iic (-2) ∪ Set.Ioi (-1/2) = {x : ℝ | a * x^2 + b * x + c < 0}) :
  {x : ℝ | c * x^2 - b * x + a > 0} = Set.Ioo (1/2) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l537_53771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_sale_price_is_ten_percent_l537_53760

/-- Represents the discount percentage as a real number between 0 and 1 -/
def Discount := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- The list price of the jersey -/
def list_price : ℝ := 120

/-- The range of initial discounts -/
def initial_discount_range : Set ℝ := { d | 0.2 ≤ d ∧ d ≤ 0.8 }

/-- The additional summer sale discount for items initially discounted 40% or less -/
def additional_discount_low : ℝ := 0.25

/-- The additional summer sale discount for items initially discounted between 41% and 80% -/
def additional_discount_high : ℝ := 0.1

/-- Calculate the sale price after applying both initial and additional discounts -/
noncomputable def sale_price (initial_discount : ℝ) : ℝ :=
  if initial_discount ≤ 0.4
  then list_price * (1 - initial_discount) * (1 - additional_discount_low)
  else list_price * (1 - initial_discount - additional_discount_high)

/-- The theorem stating that the lowest possible total sale price is 10% of the list price -/
theorem lowest_sale_price_is_ten_percent :
  ∃ (d : ℝ), d ∈ initial_discount_range ∧
  ∀ (d' : ℝ), d' ∈ initial_discount_range →
  sale_price d ≤ sale_price d' ∧
  sale_price d = list_price * 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_sale_price_is_ten_percent_l537_53760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_common_point_condition_l537_53751

noncomputable section

variable (a b c d : ℝ)

/-- The function f(x) = 2a + 1/(x-b) -/
noncomputable def f (a b x : ℝ) : ℝ := 2*a + 1/(x-b)

/-- The function g(x) = 2c + 1/(x-d) -/
noncomputable def g (c d x : ℝ) : ℝ := 2*c + 1/(x-d)

/-- The center of symmetry -/
noncomputable def center (a b c d : ℝ) : ℝ × ℝ := ((b+d)/2, a+c)

/-- Condition for central symmetry -/
def centrally_symmetric (f g : ℝ → ℝ) (center : ℝ × ℝ) : Prop :=
  ∀ x y, f x = y ↔ g (2*center.1 - x) = 2*center.2 - y

/-- Theorem: The condition for f and g to have exactly one common point -/
theorem one_common_point_condition :
  (centrally_symmetric (f a b) (g c d) (center a b c d)) →
  (∃! x, f a b x = g c d x) ↔ (a-c)*(b-d) = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_common_point_condition_l537_53751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2020_2187_l537_53718

noncomputable def sequence_term (n : ℕ) : ℚ :=
  let group := (n - 1) / (3 ^ (Nat.log 3 (n - 1) + 1) - 1) + 1
  let position := n - (3 ^ group - 1) / 2
  2 * (position : ℚ) / 3 ^ group

theorem sequence_2020_2187 : sequence_term 1553 = 2020 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2020_2187_l537_53718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_star_sum_l537_53703

noncomputable def star (a b : ℝ) : ℝ := Real.sin a * Real.cos b

theorem max_value_of_star_sum (x y : ℝ) (h : star x y - star y x = 1) :
  ∃ (max : ℝ), max = 1 ∧ star x y + star y x ≤ max := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_star_sum_l537_53703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_one_eighth_l537_53784

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The large rectangle in the problem -/
def largeRectangle : Rectangle := ⟨15, 20⟩

/-- Theorem stating that the shaded fraction of the large rectangle is 1/8 -/
theorem shaded_fraction_is_one_eighth :
  let quarterRectangle : Rectangle := ⟨largeRectangle.width / 2, largeRectangle.height / 2⟩
  let shadedArea : ℝ := (quarterRectangle.area) / 2
  shadedArea / largeRectangle.area = 1 / 8 := by
  -- Expand definitions
  unfold Rectangle.area
  -- Perform algebraic manipulations
  simp [largeRectangle]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_one_eighth_l537_53784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l537_53797

-- Define the number of arcs
def num_arcs : ℕ := 12

-- Define the length of each arc
noncomputable def arc_length : ℝ := Real.pi / 2

-- Define the side length of the octagon
def octagon_side : ℝ := 3

-- Define the area of the enclosed figure
noncomputable def enclosed_area : ℝ := 18 * (1 + Real.sqrt 2) + 6 * Real.pi

theorem enclosed_area_theorem :
  let radius : ℝ := arc_length / (Real.pi / 2)
  let octagon_area : ℝ := 2 * (1 + Real.sqrt 2) * octagon_side ^ 2
  let sector_area : ℝ := ↑num_arcs * (Real.pi * radius ^ 2 / 4)
  octagon_area + sector_area = enclosed_area := by sorry

#check enclosed_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l537_53797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_fixed_point_l537_53745

/-- The fixed point through which the line of tangency passes -/
noncomputable def fixed_point : ℝ × ℝ := (1, 1/2)

/-- The equation of the line on which P moves -/
def line_eq (x y : ℝ) : Prop := x + 2*y = 4

/-- The equation of the ellipse -/
def ellipse_eq (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

/-- Theorem stating that the line of tangency always passes through the fixed point -/
theorem tangent_line_passes_through_fixed_point :
  ∀ (x₀ y₀ : ℝ), line_eq x₀ y₀ →
  ∃ (A B : ℝ × ℝ),
    (∀ (x y : ℝ), ellipse_eq x y → (x - x₀)*(A.1 - x₀) + 4*(y - y₀)*(A.2 - y₀) = 0) ∧
    (∀ (x y : ℝ), ellipse_eq x y → (x - x₀)*(B.1 - x₀) + 4*(y - y₀)*(B.2 - y₀) = 0) ∧
    (∃ (t : ℝ), fixed_point = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_fixed_point_l537_53745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_x_l537_53741

noncomputable def f (x : ℝ) : ℝ := 1 + x - x^3/3 + x^5/5 - x^7/7 + x^9/9 - x^11/11 + x^13/13

theorem smallest_positive_x (x : ℤ) : 
  (∀ y : ℤ, y < x → f (↑y - 1) ≤ 0) ∧ f (↑x - 1) > 0 ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_x_l537_53741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l537_53744

noncomputable section

-- Define the parabola y = -x^2
def parabola (x : ℝ) : ℝ := -x^2

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola x

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define an isosceles right triangle
def isosceles_right_triangle (a b o : ℝ × ℝ) : Prop :=
  distance a o = distance b o ∧ distance a b = Real.sqrt 2 * distance a o

-- Theorem statement
theorem triangle_side_length
  (a b : PointOnParabola)
  (h : isosceles_right_triangle (a.x, a.y) (b.x, b.y) origin) :
  distance (a.x, a.y) (b.x, b.y) = 4 * Real.sqrt 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l537_53744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l537_53788

noncomputable section

/-- Ellipse C: x^2/25 + y^2/9 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

/-- Moving circle Γ: x^2 + y^2 = r^2, where 3 < r < 5 -/
def circle_Γ (x y r : ℝ) : Prop := x^2 + y^2 = r^2 ∧ 3 < r ∧ r < 5

/-- Point M on ellipse C -/
def point_M (x y : ℝ) : Prop := ellipse_C x y

/-- Point N on circle Γ -/
def point_N (x y r : ℝ) : Prop := circle_Γ x y r

/-- Line segment MN is tangent to both C and Γ -/
def tangent_MN (x1 y1 x2 y2 r : ℝ) : Prop :=
  point_M x1 y1 ∧ point_N x2 y2 r ∧
  ∃ k m : ℝ, (y2 - y1) = k * (x2 - x1) ∧
             (25 * k^2 + 9) * x1^2 + 50 * k * m * x1 + 25 * (m^2 - 9) = 0 ∧
             (1 + k^2) * x2^2 + 2 * k * m * x2 + (m^2 - r^2) = 0

/-- Distance between points M and N -/
noncomputable def distance_MN (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem max_distance_MN :
  ∀ x1 y1 x2 y2 r : ℝ,
    tangent_MN x1 y1 x2 y2 r →
    distance_MN x1 y1 x2 y2 ≤ 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l537_53788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_fifth_term_is_seven_l537_53701

/-- The sequence where each number n appears n times consecutively -/
def our_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => if n + 1 ≤ our_sequence n * (our_sequence n + 1) / 2 then our_sequence n else our_sequence n + 1

/-- The 25th term of the sequence is 7 -/
theorem twenty_fifth_term_is_seven : our_sequence 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_fifth_term_is_seven_l537_53701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_incircle_contact_point_l537_53799

-- Define the hyperbola
structure Hyperbola where
  F₁ : EuclideanSpace ℝ (Fin 2)
  F₂ : EuclideanSpace ℝ (Fin 2)
  M : EuclideanSpace ℝ (Fin 2)
  N : EuclideanSpace ℝ (Fin 2)

-- Define a point on the hyperbola
def PointOnHyperbola (h : Hyperbola) (P : EuclideanSpace ℝ (Fin 2)) : Prop :=
  abs (dist P h.F₁ - dist P h.F₂) = dist h.M h.N

-- Define the incircle of a triangle
def Incircle (A B C G : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 
    dist G A = r ∧ 
    dist G B = r ∧ 
    dist G C = r

-- Main theorem
theorem hyperbola_incircle_contact_point 
  (h : Hyperbola) (P G : EuclideanSpace ℝ (Fin 2)) :
  PointOnHyperbola h P →
  Incircle P h.F₁ h.F₂ G →
  G = h.M ∨ G = h.N :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_incircle_contact_point_l537_53799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_inverse_relation_l537_53795

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x - 3) / (x + 4)

-- State the theorem
theorem incorrect_inverse_relation (x y : ℝ) (h : y = g x) : 
  ¬(x = (y - 3) / (y + 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_inverse_relation_l537_53795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_willam_land_percentage_l537_53742

/-- Represents the farm tax calculation for a village and an individual farmer. -/
structure FarmTax where
  totalTax : ℝ  -- Total tax collected from the village
  individualTax : ℝ  -- Tax paid by the individual farmer
  taxRate : ℝ  -- Percentage of cultivated land that is taxed

/-- Calculates the percentage of an individual's taxable land relative to the village's total taxable land. -/
noncomputable def landPercentage (ft : FarmTax) : ℝ :=
  (ft.individualTax / ft.totalTax) * 100 / ft.taxRate

/-- Theorem stating that given the specific tax values, the land percentage is 5.625%. -/
theorem willam_land_percentage :
  let ft : FarmTax := ⟨3840, 480, 0.45⟩
  landPercentage ft = 5.625 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_willam_land_percentage_l537_53742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_prism_surface_area_l537_53776

/-- The total surface area of a regular triangular prism -/
noncomputable def total_surface_area (a Q : ℝ) : ℝ :=
  Real.sqrt 3 * (0.5 * a^2 + 2 * Q)

/-- 
  Theorem: The total surface area of a regular triangular prism is √3 * (0.5 * a² + 2Q),
  where 'a' is the side length of the prism's base, and 'Q' is the area of the cross-section
  passing through the lateral edge and perpendicular to the opposite lateral face.
-/
theorem regular_triangular_prism_surface_area (a Q : ℝ) 
  (h_a : a > 0) (h_Q : Q > 0) : 
  total_surface_area a Q = 
    2 * (Real.sqrt 3 / 4 * a^2) + 3 * (2 * Q / Real.sqrt 3) := by
  -- Unfold the definition of total_surface_area
  unfold total_surface_area
  -- Algebraic manipulation
  ring
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_prism_surface_area_l537_53776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_equation_l537_53705

/-- A line that passes through (-1, 1) and intersects the circle x^2 + 4x + y^2 = 0 with a chord length of 2√3 -/
def IntersectingLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (k : ℝ), (p.1 = -1 ∧ p.2 = k) ∨ (p.2 - 1 = k * (p.1 + 1))}

/-- The circle x^2 + 4x + y^2 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + 4*p.1 + p.2^2 = 0}

/-- The chord length of the intersection between the line and the circle -/
noncomputable def ChordLength (l : Set (ℝ × ℝ)) : ℝ :=
  2 * Real.sqrt 3

/-- Theorem: The line passing through (-1, 1) and intersecting the given circle
    with a chord length of 2√3 has the equation x = -1 or y = 1 -/
theorem intersecting_line_equation :
  IntersectingLine = {p : ℝ × ℝ | p.1 = -1 ∨ p.2 = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_equation_l537_53705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_base_for_87_l537_53721

/-- Given a natural number n and a base b, returns the number of digits required to represent n in base b -/
def num_digits (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1
  else Nat.log b n + 1

/-- Checks if a number n can be represented in base b using at most k digits -/
def representable_in_k_digits (n : ℕ) (b : ℕ) (k : ℕ) : Prop :=
  num_digits n b ≤ k

theorem smallest_base_for_87 :
  ∃ (b : ℕ), b > 0 ∧ representable_in_k_digits 87 b 3 ∧
  ∀ (c : ℕ), c > 0 → representable_in_k_digits 87 c 3 → b ≤ c :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_base_for_87_l537_53721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_l537_53796

/-- A function f is increasing on an open interval (a, b) if for all x, y in (a, b),
    x < y implies f(x) < f(y) -/
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem power_function_increasing (m : ℝ) :
  IncreasingOn (fun x ↦ (m^2 - m - 5) * x^(m-1)) 0 (Real.pi/2) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_l537_53796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equal_angles_parallel_transitive_l537_53787

-- Define the concept of lines in space
variable (Line : Type)

-- Define the concept of parallelism between lines
variable (parallel : Line → Line → Prop)

-- Define the concept of intersection between lines
variable (intersect : Line → Line → Prop)

-- Define the concept of angle between two intersecting lines
variable (angle : Line → Line → ℝ)

-- Define the concept of acute angle
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Theorem 1: If two intersecting lines are parallel to another two intersecting lines, 
-- then the acute angles (or right angles) formed by these two sets of lines are equal
theorem parallel_lines_equal_angles 
  (l1 l2 l3 l4 : Line) 
  (h1 : intersect l1 l2) 
  (h2 : intersect l3 l4) 
  (h3 : parallel l1 l3) 
  (h4 : parallel l2 l4) 
  (h5 : is_acute_angle (angle l1 l2) ∨ angle l1 l2 = Real.pi / 2) 
  (h6 : is_acute_angle (angle l3 l4) ∨ angle l3 l4 = Real.pi / 2) : 
  angle l1 l2 = angle l3 l4 := by
  sorry

-- Theorem 2: If two lines are parallel to a third line at the same time, 
-- then these two lines are parallel to each other
theorem parallel_transitive 
  (l1 l2 l3 : Line) 
  (h1 : parallel l1 l3) 
  (h2 : parallel l2 l3) : 
  parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equal_angles_parallel_transitive_l537_53787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_range_l537_53700

-- Define the line l: y = 2x - 4
def line_l (x : ℝ) : ℝ := 2 * x - 4

-- Define the circle C with center (a, line_l a) and radius 1
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - line_l a)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (0, 3)

-- Define the origin O
def point_O : ℝ × ℝ := (0, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem circle_center_range :
  ∀ a : ℝ,
  (∃ x y : ℝ, circle_C a x y ∧ 
    distance (x, y) point_A = 2 * distance (x, y) point_O) →
  0 ≤ a ∧ a ≤ 12/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_range_l537_53700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l537_53778

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  pos : 0 < a ∧ a < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := Real.sqrt (1 + (b/a)^2)

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents the condition that the symmetric point of the left focus
    about one asymptote lies on the other asymptote -/
def symmetric_focus_condition (h : Hyperbola a b) : Prop :=
  ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ 
  (∃ (f : Point), ∃ (l : Line), True)  -- Placeholder for focus and asymptote conditions

theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (hc : symmetric_focus_condition h) : eccentricity h = 2 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l537_53778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l537_53725

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x + φ)

theorem monotonic_increasing_interval 
  (φ : ℝ) 
  (h : f (π / 12) φ - f (-5 * π / 12) φ = 2) :
  ∀ k : ℤ, StrictMonoOn (f · φ) (Set.Icc (k * π - 5 * π / 12) (k * π + π / 12)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l537_53725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_most_3_approx_l537_53793

/-- The probability of rain on a single day in April in Garden Town -/
noncomputable def p : ℝ := 1/5

/-- The number of days in April -/
def n : ℕ := 30

/-- The binomial coefficient -/
def binom (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of exactly k rainy days in April -/
noncomputable def prob_k_days (k : ℕ) : ℝ :=
  (binom n k : ℝ) * p^k * (1-p)^(n-k)

/-- The probability of at most 3 rainy days in April -/
noncomputable def prob_at_most_3 : ℝ :=
  prob_k_days 0 + prob_k_days 1 + prob_k_days 2 + prob_k_days 3

/-- Theorem stating the probability of at most 3 rainy days in April is approximately 0.616 -/
theorem prob_at_most_3_approx :
  abs (prob_at_most_3 - 0.616) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_most_3_approx_l537_53793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l537_53706

theorem cos_minus_sin_value (α : Real) 
  (h1 : π / 2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin (2 * α) = -24 / 25) : 
  Real.cos α - Real.sin α = -7 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l537_53706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_l537_53770

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2*a*x + 2
  else x + 16/x - 3*a

-- State the theorem
theorem min_value_condition (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a 1) ↔ a ∈ Set.Icc 1 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_l537_53770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l537_53717

/-- A function f(x) = 2^(|x+a|) that satisfies certain conditions -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(abs (x + a))

/-- The theorem stating the maximum value of m -/
theorem max_m_value (a : ℝ) (m : ℝ) : 
  (∀ x, f a (3 + x) = f a (3 - x)) →  -- Condition 1
  (∀ x y, x ≤ y → y ≤ m → f a x ≥ f a y) →  -- Condition 2 (monotonically decreasing)
  m ≤ 3 :=  -- Conclusion: maximum value of m is 3
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l537_53717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_l537_53726

def f : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => f (n + 1) + 2^(f (n + 1))

theorem distinct_remainders :
  ∀ x y, x ∈ Finset.range (3^2013) → y ∈ Finset.range (3^2013) →
  x ≠ y → f (x + 1) % (3^2013) ≠ f (y + 1) % (3^2013) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_l537_53726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_analysis_l537_53783

open Real

noncomputable def f (t : ℝ) : ℝ := 10 - Real.sqrt 3 * cos (π * t / 12) - sin (π * t / 12)

theorem temperature_analysis (t : ℝ) (h : t ∈ Set.Icc 0 24) :
  (∃ t₁ t₂, f t₁ - f t₂ = 4 ∧ ∀ t₃, f t₃ ≤ f t₁ ∧ f t₃ ≥ f t₂) ∧
  (∀ t', t' ∈ Set.Ioo 10 18 → f t' > 11) ∧
  (∀ t', (t' ∉ Set.Ioo 10 18 ∧ t' ∈ Set.Icc 0 24) → f t' ≤ 11) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_analysis_l537_53783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l537_53785

def A : Set ℝ := {x : ℝ | x ≥ 0}
def B : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n ∧ -1 ≤ n ∧ n ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l537_53785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_and_exponential_l537_53782

theorem order_of_logarithms_and_exponential : 
  let a := 2 * Real.log (3/2)
  let b := Real.log 3 / Real.log 2
  let c := (1/2)^(-0.3 : ℝ)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_and_exponential_l537_53782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ethane_combustion_enthalpy_change_l537_53709

/-- The enthalpy change for the complete combustion of 1 mole of ethane -/
noncomputable def enthalpy_change_ethane_combustion (
  ΔHf_C2H6 : ℝ)  -- Standard enthalpy of formation of C2H6
  (ΔHf_O2 : ℝ)    -- Standard enthalpy of formation of O2
  (ΔHf_CO2 : ℝ)   -- Standard enthalpy of formation of CO2
  (ΔHf_H2O : ℝ)   -- Standard enthalpy of formation of H2O
  : ℝ :=
  2 * ΔHf_CO2 + 3 * ΔHf_H2O - ΔHf_C2H6 - (7/2) * ΔHf_O2

/-- Theorem stating the enthalpy change for the complete combustion of 1 mole of ethane -/
theorem ethane_combustion_enthalpy_change :
  enthalpy_change_ethane_combustion (-84) 0 (-394) (-286) = -1562 :=
by
  -- Unfold the definition of enthalpy_change_ethane_combustion
  unfold enthalpy_change_ethane_combustion
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ethane_combustion_enthalpy_change_l537_53709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_difference_l537_53739

theorem square_side_length_difference : ∃ (area_A area_B side_A side_B : ℝ),
  area_A = 25 ∧
  area_B = 81 ∧
  side_A = Real.sqrt area_A ∧
  side_B = Real.sqrt area_B ∧
  side_B - side_A = 4 := by
  -- Define the areas of the squares
  let area_A : ℝ := 25
  let area_B : ℝ := 81

  -- Define the side lengths of the squares
  let side_A : ℝ := Real.sqrt area_A
  let side_B : ℝ := Real.sqrt area_B

  -- Prove the theorem
  exists area_A, area_B, side_A, side_B
  constructor
  · exact rfl
  constructor
  · exact rfl
  constructor
  · exact rfl
  constructor
  · exact rfl
  -- The final step of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_difference_l537_53739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_round_trip_motorist_journey_average_speed_l537_53723

/-- Calculates the average speed for a round trip journey -/
theorem average_speed_round_trip 
  (distance_one_way : ℝ) 
  (speed_to : ℝ) 
  (speed_from : ℝ) 
  (distance_one_way_positive : 0 < distance_one_way)
  (speed_to_positive : 0 < speed_to)
  (speed_from_positive : 0 < speed_from) :
  (2 * distance_one_way) / (distance_one_way / speed_to + distance_one_way / speed_from) = 
    2 * speed_to * speed_from / (speed_to + speed_from) :=
by sorry

/-- Proves that the average speed for the given journey is 37.5 km/hr -/
theorem motorist_journey_average_speed :
  (2 * 150) / (150 / 50 + 150 / 30) = 37.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_round_trip_motorist_journey_average_speed_l537_53723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_l537_53712

/-- Definition of a line in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Definition of the three lines -/
def l₁ : Line := { A := 1, B := 2, C := 1 }
def l₂ : Line := { A := 3, B := -4, C := 5 }
def l₃ (a : ℝ) : Line := { A := a, B := 2, C := -6 }

/-- Function to check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.A * l2.B = l2.A * l1.B

/-- Function to check if a point (x, y) lies on a line -/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  l.A * x + l.B * y + l.C = 0

/-- Function to find the intersection point of two lines -/
noncomputable def intersection (l1 l2 : Line) : ℝ × ℝ :=
  let x := (l1.B * l2.C - l2.B * l1.C) / (l1.A * l2.B - l2.A * l1.B)
  let y := (l2.A * l1.C - l1.A * l2.C) / (l1.A * l2.B - l2.A * l1.B)
  (x, y)

/-- Theorem stating the conditions for l₃ not forming a triangle with l₁ and l₂ -/
theorem no_triangle (a : ℝ) : 
  (parallel l₁ (l₃ a) ∨ parallel l₂ (l₃ a) ∨ 
   (let (x, y) := intersection l₁ l₂; point_on_line (l₃ a) x y)) ↔ 
  (a = 1 ∨ a = -3/2 ∨ a = -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_l537_53712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_theorem_l537_53713

-- Define the line l: x - y + 3 = 0
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the circle C: (x - a)^2 + (y - 2)^2 = 4
def circle_C (x y a : ℝ) : Prop := (x - a)^2 + (y - 2)^2 = 4

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 2

-- Theorem statement
theorem chord_intercept_theorem (a : ℝ) :
  (∃ x y : ℝ, line_l x y ∧ circle_C x y a) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
    circle_C x₁ y₁ a ∧ circle_C x₂ y₂ a ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2) →
  a = 1 ∨ a = -3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_theorem_l537_53713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_bound_l537_53769

/-- Given a cubic polynomial with real coefficients and roots satisfying certain conditions,
    the expression (2a³ + 27c - 9ab)/λ³ is bounded above by 3√3/2. -/
theorem cubic_polynomial_bound (a b c lambda : ℝ) (x₁ x₂ x₃ : ℝ) 
    (hlambda_pos : lambda > 0)
    (hroots : x₁^3 + a*x₁^2 + b*x₁ + c = 0 ∧ 
              x₂^3 + a*x₂^2 + b*x₂ + c = 0 ∧ 
              x₃^3 + a*x₃^2 + b*x₃ + c = 0)
    (hdiff : x₂ - x₁ = lambda)
    (hx₃ : x₃ > (x₁ + x₂) / 2) :
  (2 * a^3 + 27 * c - 9 * a * b) / lambda^3 ≤ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_bound_l537_53769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_distance_calculation_l537_53779

/-- Calculates the initial distance between a policeman and a criminal given their speeds and the distance after a certain time. -/
theorem initial_distance_calculation 
  (criminal_speed : ℝ) 
  (policeman_speed : ℝ) 
  (time : ℝ) 
  (distance_after : ℝ) 
  (h1 : criminal_speed = 8) 
  (h2 : policeman_speed = 9) 
  (h3 : time = 3 / 60) 
  (h4 : distance_after = 130) : 
  ∃ initial_distance : ℝ, 
    initial_distance = distance_after + (policeman_speed - criminal_speed) * time ∧ 
    abs (initial_distance - 130.05) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_distance_calculation_l537_53779
