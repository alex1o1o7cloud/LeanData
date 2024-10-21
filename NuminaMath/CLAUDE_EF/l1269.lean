import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_1005_eq_193_l1269_126989

/-- Define the sequence v_n -/
def v : ℕ → ℕ
| 0 => 1  -- First term
| n + 1 => 
  let k := (Nat.sqrt ((8 * (n + 1) + 1)) - 1) / 2  -- Group number
  let m := (n + 1) - k * (k + 1) / 2  -- Position within group
  (k + 1) * 4 - (k + 1) + m * 4

/-- The 1005th term of the sequence is 193 -/
theorem v_1005_eq_193 : v 1004 = 193 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_1005_eq_193_l1269_126989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_alpha_gamma_sum_l1269_126967

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the function f
noncomputable def f (α γ : ℂ) (z : ℂ) : ℂ := (5 + 2*i)*z^2 + α*z + γ

-- State the theorem
theorem min_alpha_gamma_sum :
  ∃ (min : ℝ), min = 2 * Real.sqrt 2 ∧
  ∀ (α γ : ℂ), (f α γ 1).im = 0 ∧ (f α γ (-i)).im = 0 →
  Complex.abs α + Complex.abs γ ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_alpha_gamma_sum_l1269_126967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1269_126977

/-- A function satisfying the given functional equation -/
class FunctionalEquation (f : ℝ → ℝ) :=
  (eq : ∀ x y : ℝ, f (x + y) * f (x - y) = f x ^ 2 + f y ^ 2 - 1)

/-- The main theorem -/
theorem functional_equation_solution {f : ℝ → ℝ} (hf : FunctionalEquation f)
  (h_cont : ContDiff ℝ 2 f) :
  (∃ k : ℝ, ∀ x : ℝ, deriv (deriv f) x = k^2 * f x ∨ deriv (deriv f) x = -k^2 * f x) ∧
  (∃ k : ℝ, (∀ x : ℝ, f x = Real.cos (k * x) ∨ f x = -Real.cos (k * x)) ∨
            (∀ x : ℝ, f x = Real.cosh (k * x) ∨ f x = -Real.cosh (k * x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1269_126977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_triangle_theorem_l1269_126932

/-- Given a triangle ABC and points P, Q, R satisfying certain angle conditions,
    prove that ∠PRQ is a right angle and QR = PR. -/
theorem external_triangle_theorem (A B C P Q R : ℂ) : 
  (Complex.arg ((P - B) / (C - B)) = π / 4) →
  (Complex.arg ((P - C) / (B - C)) = π / 6) →
  (Complex.arg ((Q - A) / (C - A)) = π / 4) →
  (Complex.arg ((Q - C) / (A - C)) = π / 6) →
  (Complex.arg ((R - B) / (A - B)) = π / 12) →
  (Complex.arg ((R - A) / (B - A)) = π / 12) →
  (Complex.arg ((Q - R) / (P - R)) = π / 2) ∧ (Complex.abs (Q - R) = Complex.abs (P - R)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_triangle_theorem_l1269_126932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1269_126997

theorem sum_of_coefficients : 
  (fun x y : ℝ => (x^3 - 3*x*y^2 + y^3)^5) 1 1 = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l1269_126997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l1269_126926

open Real

theorem function_satisfies_equation (x : ℝ) (hx : cos x ≠ 0) :
  let y : ℝ → ℝ := λ x => x / cos x
  let y' : ℝ → ℝ := deriv y
  y' x - y x * tan x = 1 / cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l1269_126926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_third_l1269_126982

theorem sin_alpha_plus_pi_third (α : ℝ) 
  (h1 : Real.cos (Real.pi / 2 + α) = Real.sqrt 3 / 3) 
  (h2 : -Real.pi / 2 < α ∧ α < Real.pi / 2) : 
  Real.sin (α + Real.pi / 3) = (3 * Real.sqrt 2 - Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_third_l1269_126982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_24_l1269_126962

/-- A function that checks if a positive integer is composed only of 0s and 1s -/
def isComposedOf0sAnd1s (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The theorem statement -/
theorem smallest_divisible_by_24 :
  ∀ T : ℕ, T > 0 → isComposedOf0sAnd1s T → T % 24 = 0 →
  ∀ X : ℕ, X * 24 = T → X ≥ 4625 :=
by
  sorry

#check smallest_divisible_by_24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_24_l1269_126962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_price_l1269_126942

/-- Represents the pricing structure of an article -/
structure ArticlePricing where
  cost_price : ℚ
  selling_price : ℚ
  markup_percentage : ℚ
  discount_percentage : ℚ
  profit_percentage : ℚ

/-- Calculates the cost price of an article given its pricing structure -/
def calculate_cost_price (ap : ArticlePricing) : ℚ :=
  ap.selling_price / (1 + ap.profit_percentage)

/-- Theorem stating the cost price of the article given the conditions -/
theorem article_cost_price :
  let ap : ArticlePricing := {
    cost_price := 0,  -- We don't know this yet
    selling_price := 69.44,
    markup_percentage := 0.10,
    discount_percentage := 0.10,
    profit_percentage := 0.25
  }
  ∃ (cost_price : ℚ), 
    cost_price = calculate_cost_price ap ∧ 
    (cost_price * 100).floor / 100 = 5555 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_price_l1269_126942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_proof_l1269_126937

theorem vector_operation_proof :
  let v₁ : Fin 3 → ℝ := ![(-3), 2, (-1)]
  let v₂ : Fin 3 → ℝ := ![1, 5, (-3)]
  (3 : ℝ) • (v₁ + v₂) = ![(-6), 21, (-12)] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_proof_l1269_126937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_to_parallelogram_l1269_126907

open EuclideanGeometry

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the equality of angles
def angle_eq (A B C D E F : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the circumcenter of a triangle
noncomputable def circumcenter (A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

-- Define the orthocenter of a triangle
noncomputable def orthocenter (A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

-- Define collinearity of three points
def collinear (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define a parallelogram
def is_parallelogram (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- The main theorem
theorem quadrilateral_to_parallelogram 
  (A B C D : EuclideanSpace ℝ (Fin 2))
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : angle_eq B A C B C D)
  (h3 : collinear B (circumcenter A B C) (orthocenter A D C)) :
  is_parallelogram A B C D :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_to_parallelogram_l1269_126907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1269_126903

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x / Real.log x

theorem function_properties :
  ∃ (m : ℝ),
    (∀ x : ℝ, x > 0 → x ≠ 1 → HasDerivAt (f m) ((1/2) / (Real.exp 2)) (Real.exp 2)) ∧
    (∀ x : ℝ, 0 < x → x < 1 → (deriv (f m)) x < 0) ∧
    (∀ x : ℝ, 1 < x → x < Real.exp 1 → (deriv (f m)) x < 0) ∧
    (∃ k : ℝ, ∀ x : ℝ, x > 0 → x ≠ 1 → f m x > k / Real.log x + 2 * Real.sqrt x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1269_126903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_sinusoidal_function_l1269_126917

/-- Given a sinusoidal function f(x) with specific properties, prove that (-2π/3, 0) is a symmetry center coordinate. -/
theorem symmetry_center_of_sinusoidal_function 
  (f : ℝ → ℝ) 
  (w φ : ℝ) 
  (h_f : ∀ x, f x = Real.sin (w * x + φ))
  (h_w : w > 0)
  (h_φ : |φ| < π / 2)
  (h_period : (2 * π) / w = 4 * π)
  (h_max : ∀ x, f x ≤ f (π / 3)) :
  ∃ (k : ℤ), ∀ x, f (x - 2 * π / 3) = f (-x - 2 * π / 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_sinusoidal_function_l1269_126917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l1269_126908

/-- The function f(x) = cos x - 4x² -/
noncomputable def f (x : ℝ) : ℝ := Real.cos x - 4 * x^2

/-- The domain of f is [-π, π] -/
def f_domain : Set ℝ := Set.Icc (-Real.pi) Real.pi

theorem solution_set_characterization (x : ℝ) (hx : x > 0) :
  f (Real.log x) + Real.pi^2 > 0 ↔ 
    x ∈ Set.union (Set.Ioo 0 (Real.exp (-Real.pi/2))) (Set.Ioi (Real.exp (Real.pi/2))) :=
by
  sorry

#check solution_set_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l1269_126908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1269_126929

/-- Given points A and B, and a point P on a line, if there are two points P
    satisfying a certain equation, then λ is less than 2. -/
theorem lambda_range (A B P : ℝ × ℝ) (lambda : ℝ) : 
  A = (2, 3) →
  B = (6, -3) →
  (3 * P.1 - 4 * P.2 + 3 = 0) →
  (∃ P₁ P₂ : ℝ × ℝ, P₁ ≠ P₂ ∧ 
    (3 * P₁.1 - 4 * P₁.2 + 3 = 0) ∧
    (3 * P₂.1 - 4 * P₂.2 + 3 = 0) ∧
    ((P₁.1 - A.1) * (P₁.1 - B.1) + (P₁.2 - A.2) * (P₁.2 - B.2) + 2 * lambda = 0) ∧
    ((P₂.1 - A.1) * (P₂.1 - B.1) + (P₂.2 - A.2) * (P₂.2 - B.2) + 2 * lambda = 0)) →
  lambda < 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1269_126929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shovel_time_eight_hours_l1269_126910

/-- Calculates the amount of snow removed in a given hour -/
def snowRemovedInHour (hour : ℕ) : ℕ :=
  max (30 - 2 * (hour - 1)) 0

/-- Calculates the total amount of snow removed up to a given hour -/
def totalSnowRemoved (hours : ℕ) : ℕ :=
  Finset.sum (Finset.range hours) (fun i => snowRemovedInHour (i + 1))

/-- The theorem stating that it takes 8 hours to shovel the driveway clean -/
theorem shovel_time_eight_hours :
  ∃ (h : ℕ), h = 8 ∧ totalSnowRemoved h ≥ 180 ∧ totalSnowRemoved (h - 1) < 180 := by
  sorry

#eval totalSnowRemoved 8  -- This will evaluate the total snow removed after 8 hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shovel_time_eight_hours_l1269_126910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_50pi_l1269_126986

-- Define the lower and upper functions
noncomputable def lower_function (x : ℝ) : ℝ := 2 * Real.cos (4 * x)
noncomputable def upper_function (x : ℝ) : ℝ := Real.sin (2 * x) + 10

-- Define the left and right boundaries
noncomputable def left_boundary : ℝ := 0
noncomputable def right_boundary : ℝ := 5 * Real.pi

-- Define the area function
noncomputable def area : ℝ := ∫ x in left_boundary..right_boundary, upper_function x - lower_function x

-- Theorem statement
theorem area_equals_50pi : area = 50 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_50pi_l1269_126986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increased_productivity_l1269_126920

theorem increased_productivity (base_production : ℝ) (base_hours : ℝ) 
  (production_increase_rate : ℝ) (hours_decrease_rate : ℝ) 
  (h1 : production_increase_rate = 0.8) 
  (h2 : hours_decrease_rate = 0.1) 
  (h3 : base_production > 0) 
  (h4 : base_hours > 0) : 
  (base_production * (1 + production_increase_rate)) / (base_hours * (1 - hours_decrease_rate)) = 
  2 * (base_production / base_hours) := by
  sorry

#check increased_productivity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increased_productivity_l1269_126920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_survey_l1269_126979

-- Define an approximation relation for natural numbers
def approx (a b : ℕ) : Prop := (a : ℤ) - b ≤ 10 ∧ b - a ≤ 10

-- Use infix notation for the approximation relation
infix:50 " ≈ " => approx

theorem bookstore_survey (total_surveyed : ℕ) (a m : ℕ) (total_school : ℕ) (inclined_surveyed : ℕ) :
  total_surveyed = 50 →
  a = 20 →
  m = 8 →
  total_school = 1000 →
  inclined_surveyed = 8 →
  ∃ (estimated : ℕ), estimated ≈ 600 ∧ 
    estimated = (inclined_surveyed * total_school) / total_surveyed :=
by
  sorry

#check bookstore_survey

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_survey_l1269_126979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_portion_weight_l1269_126913

-- Define the given parameters
noncomputable def bag_length : ℝ := 5
noncomputable def bag_weight : ℝ := 29/8
noncomputable def chair_length : ℝ := 4
noncomputable def chair_weight : ℝ := 2.8
noncomputable def portion_length : ℝ := 2

-- Theorem statement
theorem combined_portion_weight :
  let bag_weight_per_meter := bag_weight / bag_length
  let chair_weight_per_meter := chair_weight / chair_length
  let bag_portion_weight := bag_weight_per_meter * portion_length
  let chair_portion_weight := chair_weight_per_meter * portion_length
  bag_portion_weight + chair_portion_weight = 2.85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_portion_weight_l1269_126913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buckingham_palace_visitors_l1269_126964

/-- The number of visitors to Buckingham Palace on a given day -/
def visitors_on_day (n : ℕ) : ℕ := sorry

/-- The total number of visitors over a period of days -/
def total_visitors (n : ℕ) : ℕ := sorry

theorem buckingham_palace_visitors 
  (h1 : total_visitors 327 = 406)
  (h2 : visitors_on_day 326 = 274) :
  visitors_on_day 327 = 132 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_buckingham_palace_visitors_l1269_126964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petri_dish_ten_percent_day_petri_dish_first_day_over_ten_percent_l1269_126980

/-- Represents the growth rate of the bacterial colony -/
noncomputable def growthRate : ℝ := 1.5

/-- Represents the number of days it takes to fill the entire petri dish -/
def totalDays : ℕ := 40

/-- Calculates the percentage of the petri dish filled on a given day -/
noncomputable def percentageFilled (day : ℕ) : ℝ :=
  100 / (growthRate ^ (totalDays - day))

/-- The day we're looking for when the dish is approximately 10% filled -/
def targetDay : ℕ := 35

/-- Theorem stating that the targetDay is when the petri dish is approximately 10% filled -/
theorem petri_dish_ten_percent_day :
  10 < percentageFilled targetDay ∧ percentageFilled targetDay < 15 := by
  sorry

/-- Theorem stating that the targetDay is the first day when the percentage exceeds 10% -/
theorem petri_dish_first_day_over_ten_percent :
  percentageFilled (targetDay - 1) < 10 ∧
  10 < percentageFilled targetDay := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petri_dish_ten_percent_day_petri_dish_first_day_over_ten_percent_l1269_126980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l1269_126934

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles :
  let circle1 : ℝ → ℝ → Prop := fun x y ↦ x^2 - 6*x + y^2 - 8*y + 9 = 0
  let circle2 : ℝ → ℝ → Prop := fun x y ↦ x^2 + 8*x + y^2 + 2*y + 16 = 0
  let shortest_distance := Real.sqrt 74 - 8
  ∃ (d : ℝ), d = shortest_distance ∧ d ≥ 0 := by
  sorry

#check shortest_distance_between_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l1269_126934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_mixture_percentage_l1269_126963

/-- Proves that adding 20 liters of pure chemical x to 80 liters of a mixture
    containing 20% chemical x results in a new mixture with 36% chemical x -/
theorem chemical_mixture_percentage : 
  ∀ (initial_volume : ℝ) (initial_x_percentage : ℝ) (added_x_volume : ℝ),
  initial_volume = 80 →
  initial_x_percentage = 20 →
  added_x_volume = 20 →
  let initial_x_volume := initial_volume * (initial_x_percentage / 100)
  let total_x_volume := initial_x_volume + added_x_volume
  let final_volume := initial_volume + added_x_volume
  let final_x_percentage := (total_x_volume / final_volume) * 100
  final_x_percentage = 36 := by
  intros initial_volume initial_x_percentage added_x_volume h1 h2 h3
  -- The proof steps would go here
  sorry

#check chemical_mixture_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_mixture_percentage_l1269_126963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1269_126971

theorem constant_term_expansion (a : ℝ) (h1 : a > 0) (h2 : (1 + a)^4 = 81) :
  let expansion := (fun x : ℝ => (1 + x)^(2*a) * (2 - 1/x))
  let constant_term := expansion 1
  constant_term = -2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1269_126971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l1269_126906

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-axis intersection point of a line -/
noncomputable def yAxisIntersection (l : Line) : ℝ × ℝ :=
  (0, l.y₁ - (l.y₂ - l.y₁) / (l.x₂ - l.x₁) * l.x₁)

/-- Theorem: The line passing through (0, 8) and (6, -4) intersects the y-axis at (0, 8) -/
theorem line_intersection_y_axis :
  let l := Line.mk 0 8 6 (-4)
  yAxisIntersection l = (0, 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l1269_126906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1269_126965

/-- The curve y = x^2 + x - 2 -/
def f (x : ℝ) : ℝ := x^2 + x - 2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 2*x + 1

/-- Line l₁ is tangent to f at (1, 0) -/
def l₁_tangent (x : ℝ) : Prop := 
  f 1 = 0 ∧ (λ x ↦ f' 1 * (x - 1)) = (λ x ↦ f x - f 1)

/-- Line l₂ is tangent to f at some point b -/
def l₂_tangent (b : ℝ) : Prop := 
  ∃ (y : ℝ), f b = y ∧ (λ x ↦ f' b * (x - b) + y) = (λ x ↦ f x - f b + y)

/-- l₁ is perpendicular to l₂ -/
def l₁_perp_l₂ (b : ℝ) : Prop := f' 1 * f' b = -1

/-- The equation of line l₂ -/
def l₂_equation : (ℝ → ℝ → Prop) := λ x y ↦ 3*x + 9*y + 22 = 0

theorem tangent_line_equation :
  ∀ b : ℝ, l₁_tangent 1 → l₂_tangent b → l₁_perp_l₂ b → l₂_equation = (λ x y ↦ 3*x + 9*y + 22 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1269_126965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_neg_one_l1269_126918

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 - x

theorem f_derivative_neg_one (a b : ℝ) :
  (deriv (f a b)) 1 = 3 → (deriv (f a b)) (-1) = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_neg_one_l1269_126918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_values_l1269_126940

def S (x : ℝ) : Set ℝ := {1, 3, x^3 - x^2 - 2*x}

def A (x : ℝ) : Set ℝ := {1, |2*x - 1|}

theorem x_values (x : ℝ) :
  (S x) \ (A x) = {0} → x = -1 ∨ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_values_l1269_126940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_argument_problem_l1269_126998

open Complex

theorem complex_argument_problem (z : ℂ) 
  (h1 : arg (z^2 - 4) = 5*π/6)
  (h2 : arg (z^2 + 4) = π/3) :
  z = 1 + I * Real.sqrt 3 ∨ z = -(1 + I * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_argument_problem_l1269_126998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_shift_l1269_126966

/-- The function f(x) = a^(x-1) - 1 has a fixed point at (1, 0) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_shift (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f := fun x : ℝ => a^(x - 1) - 1
  f 1 = 0 ∧ ∀ x : ℝ, f x = x → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_shift_l1269_126966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abs_x_minus_y_plus_one_l1269_126983

theorem max_abs_x_minus_y_plus_one (x y : ℝ) 
  (hx : |x - 1| ≤ 1) (hy : |y - 2| ≤ 1) : 
  ∃ (M : ℝ), M = 2 ∧ (|x - y + 1| ≤ M) ∧ ∃ x₀ y₀, |x₀ - y₀ + 1| = M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_abs_x_minus_y_plus_one_l1269_126983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_2_proposition_3_proposition_4_l1269_126972

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the given lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Define that m and n are different lines
variable (m_neq_n : m ≠ n)

-- Define that α and β are non-coincident planes
variable (α_neq_β : α ≠ β)

-- Theorem for proposition ②
theorem proposition_2 : 
  (perpendicular m α ∧ perpendicular n α) → parallel m n :=
sorry

-- Theorem for proposition ③
theorem proposition_3 : 
  (parallel_line_plane m α ∧ perpendicular m β) → perpendicular_plane α β :=
sorry

-- Theorem for proposition ④
theorem proposition_4 : 
  (perpendicular m α ∧ perpendicular n β ∧ parallel m n) → parallel_plane α β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_2_proposition_3_proposition_4_l1269_126972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_l1269_126927

theorem min_value_exponential_sum (a b : ℝ) (h : a - 3 * b + 6 = 0) :
  (∀ x y : ℝ, (2:ℝ)^x + 1/((8:ℝ)^y) ≥ (2:ℝ)^a + 1/((8:ℝ)^b)) → (2:ℝ)^a + 1/((8:ℝ)^b) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_l1269_126927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_f_equal_at_one_and_minus_one_l1269_126909

theorem exists_f_equal_at_one_and_minus_one :
  ∃ (a b c : ℝ), (fun x => a * Real.cos x + b * x^2 + c) 1 = 1 ∧
                  (fun x => a * Real.cos x + b * x^2 + c) (-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_f_equal_at_one_and_minus_one_l1269_126909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_c_l1269_126976

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, -6)
def c : ℝ × ℝ := sorry

axiom c_magnitude : Real.sqrt ((c.fst ^ 2) + (c.snd ^ 2)) = Real.sqrt 10
axiom dot_product : (a.fst + b.fst) * c.fst + (a.snd + b.snd) * c.snd = 5

theorem angle_between_a_and_c :
  Real.arccos ((a.fst * c.fst + a.snd * c.snd) / (Real.sqrt ((a.fst ^ 2) + (a.snd ^ 2)) * Real.sqrt ((c.fst ^ 2) + (c.snd ^ 2)))) = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_c_l1269_126976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_population_theorem_l1269_126957

/-- Represents the growth rate of the student population -/
noncomputable def G : ℝ → ℝ := sorry

/-- Represents the decline rate of the girl population -/
noncomputable def D : ℝ → ℝ := sorry

/-- Represents the total school population at time t -/
noncomputable def x : ℝ → ℝ := sorry

/-- Represents the percentage of boys that 90 students represent -/
noncomputable def boys_percent : ℝ := sorry

theorem student_population_theorem (t : ℝ) :
  (90 / (boys_percent / 100) = 0.60 * x t) →
  (x t = 15000 / boys_percent) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_population_theorem_l1269_126957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_75_meters_l1269_126990

/-- The length of a train given its speed, the time to cross a man walking in the opposite direction, and the man's speed. -/
noncomputable def train_length (train_speed : ℝ) (crossing_time : ℝ) (man_speed : ℝ) : ℝ :=
  (train_speed + man_speed) * crossing_time * (1000 / 3600)

/-- Theorem stating that the length of the train is 75 meters under the given conditions. -/
theorem train_length_is_75_meters :
  let train_speed := (40 : ℝ) -- km/h
  let crossing_time := (6 : ℝ) -- seconds
  let man_speed := (5 : ℝ) -- km/h
  train_length train_speed crossing_time man_speed = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_75_meters_l1269_126990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_equals_one_l1269_126955

/-- A function of the form a*sin(πx + α) + b*cos(πx + β) -/
noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

/-- Theorem stating that if f(2009) = -1, then f(2010) = 1 -/
theorem f_2010_equals_one
  (a b α β : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hα : α ≠ 0)
  (hβ : β ≠ 0)
  (h_2009 : f a b α β 2009 = -1) :
  f a b α β 2010 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_equals_one_l1269_126955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l1269_126973

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

def digits_used (n m : ℕ) : Prop :=
  let digits := [2, 3, 4, 6, 7, 8]
  ∀ d ∈ digits, (d ∈ (Nat.digits 10 n)) ∨ (d ∈ (Nat.digits 10 m))

theorem smallest_difference (n m : ℕ) :
  is_three_digit n ∧ is_three_digit m ∧ digits_used n m →
  ∃ (a b : ℕ), is_three_digit a ∧ is_three_digit b ∧ digits_used a b ∧
               (a : ℤ) - (b : ℤ) = 112 ∧
               ∀ (x y : ℕ), is_three_digit x ∧ is_three_digit y ∧ digits_used x y →
                             (x : ℤ) - (y : ℤ) ≥ 112 ∨ (y : ℤ) - (x : ℤ) ≥ 112 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_l1269_126973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_scheme_difference_l1269_126904

/-- Calculates the compound interest amount -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * time)

/-- Calculates the simple interest amount -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem loan_scheme_difference :
  let principal := (15000 : ℝ)
  let compound_rate := (0.08 : ℝ)
  let simple_rate := (0.10 : ℝ)
  let total_time := (12 : ℝ)
  let partial_time := (3 : ℝ)
  let compounds_per_year := (2 : ℝ)

  let compound_amount_3years := compound_interest principal compound_rate compounds_per_year partial_time
  let payment_3years := compound_amount_3years / 3
  let remaining_balance := compound_amount_3years - payment_3years
  let final_compound_amount := payment_3years + compound_interest remaining_balance compound_rate compounds_per_year (total_time - partial_time)

  let simple_amount := simple_interest principal simple_rate total_time

  round_to_nearest (simple_amount - final_compound_amount) = 2834 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_scheme_difference_l1269_126904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_first_18_even_numbers_l1269_126991

/-- The nth even number -/
def evenNumber (n : ℕ) : ℕ := 2 * n

/-- The sum of the first n even numbers -/
def sumFirstEvenNumbers (n : ℕ) : ℕ := 
  (List.range n).map (λ i => evenNumber (i + 1)) |>.sum

/-- The average of the first n even numbers -/
noncomputable def avgFirstEvenNumbers (n : ℕ) : ℚ := 
  (sumFirstEvenNumbers n : ℚ) / n

theorem avg_first_18_even_numbers : 
  avgFirstEvenNumbers 18 = 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_first_18_even_numbers_l1269_126991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1269_126954

/-- Triangle ABC with internal angles A, B, C and opposite sides a, b, c -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Vector in 2D space -/
structure Vector2D where
  x : Real
  y : Real

/-- Angle between two vectors -/
noncomputable def angle_between (v w : Vector2D) : Real := sorry

/-- Area of a triangle given two sides and the included angle -/
noncomputable def triangle_area (a b : Real) (C : Real) : Real := sorry

theorem triangle_theorem (t : Triangle) 
  (m : Vector2D) 
  (n : Vector2D) 
  (h1 : m.x = Real.cos (t.C / 2)) 
  (h2 : m.y = Real.sin (t.C / 2))
  (h3 : n.x = Real.cos (t.C / 2)) 
  (h4 : n.y = -Real.sin (t.C / 2))
  (h5 : angle_between m n = π / 3)
  (h6 : t.c = 7 / 2)
  (h7 : triangle_area t.a t.b t.C = 3 * Real.sqrt 3 / 2) :
  t.C = π / 3 ∧ t.a + t.b = 11 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1269_126954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l1269_126996

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define our function
noncomputable def f (x : ℝ) : ℝ := lg (x^2 - 1)

-- State the theorem
theorem f_increasing :
  (∀ x y, x < y ∧ x < -1 ∧ y < -1 → f x < f y) ∧
  (∀ x y, x < y ∧ x > 1 ∧ y > 1 → f x < f y) := by
  sorry

#check f_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l1269_126996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l1269_126988

/-- Vector a in ℝ³ -/
def a : Fin 3 → ℝ := ![1, -1, 1]

/-- Vector b in ℝ³ -/
def b : Fin 3 → ℝ := ![-2, 2, 1]

/-- Theorem stating that the projection of a onto b is (2/3, -2/3, -1/3) -/
theorem projection_a_onto_b :
  let proj := ((a • b) / (b • b)) • b
  proj 0 = 2/3 ∧ proj 1 = -2/3 ∧ proj 2 = -1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l1269_126988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_prime_ball_l1269_126953

def ball_numbers : List Nat := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def is_prime (n : Nat) : Bool :=
  n > 1 && (Nat.factorial (n - 1) + 1) % n = 1

theorem probability_of_prime_ball :
  (ball_numbers.filter is_prime).length / ball_numbers.length = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_prime_ball_l1269_126953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larry_win_probability_l1269_126952

/-- Represents the probability of winning for the first player in a three-player game -/
noncomputable def first_player_win_prob (p : ℝ) : ℝ :=
  p / (1 - (1 - p)^3)

/-- The main theorem stating the probability of Larry winning the game -/
theorem larry_win_probability :
  first_player_win_prob (1/3) = 9/19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larry_win_probability_l1269_126952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_6374_to_hundredth_l1269_126916

-- Define a function to round a number to the nearest hundredth
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

-- State the theorem
theorem round_24_6374_to_hundredth :
  roundToHundredth 24.6374 = 24.64 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_6374_to_hundredth_l1269_126916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frances_towels_weight_approx_l1269_126902

/-- The weight of Frances's towels in kilograms -/
noncomputable def frances_towels_weight (mary_towels : ℕ) (total_weight : ℝ) : ℝ :=
  let frances_towels := mary_towels / 8
  let total_towels := mary_towels + frances_towels
  let weight_per_towel := total_weight / total_towels
  let frances_weight_pounds := weight_per_towel * frances_towels
  frances_weight_pounds * 0.453592

/-- Theorem stating the weight of Frances's towels -/
theorem frances_towels_weight_approx :
  ∃ ε > 0, |frances_towels_weight 48 85 - 4.283| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frances_towels_weight_approx_l1269_126902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l1269_126944

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 7 else a/x

-- Define what it means for f to be increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem f_increasing_implies_a_range (a : ℝ) :
  is_increasing (f a) → -4 ≤ a ∧ a ≤ -2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_range_l1269_126944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_10_l1269_126959

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem min_sum_at_10 (seq : ArithmeticSequence) 
  (h_negative_first : seq.a 1 < 0)
  (h_equal_sums : sum_n seq 7 = sum_n seq 13) :
  ∃ (n : ℕ), (∀ (m : ℕ), sum_n seq n ≤ sum_n seq m) ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_10_l1269_126959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sine_ratio_iff_rational_ratio_l1269_126939

/-- The function f(x) = |sin(βx) / sin(αx)| is periodic if and only if β/α is rational -/
theorem periodic_sine_ratio_iff_rational_ratio (α β : ℝ) (hα : α > 0) (hβ : β > 0) :
  (∃ t : ℝ, t > 0 ∧ ∀ x, |Real.sin (β * x) / Real.sin (α * x)| = |Real.sin (β * (x + t)) / Real.sin (α * (x + t))|) ↔
  ∃ (p q : ℤ), q ≠ 0 ∧ β / α = (p : ℝ) / q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sine_ratio_iff_rational_ratio_l1269_126939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_property_l1269_126905

/-- Parabola with equation y² = 4x -/
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Line with slope k -/
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * x + k

/-- Vertical bisector of a line segment -/
def VerticalBisector (x₀ y₀ k : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = -(1/k) * (x - x₀)

/-- Distance between two points -/
noncomputable def Distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Main theorem -/
theorem parabola_intersection_property
  (k : ℝ)
  (xM yM xN yN : ℝ)
  (hM : Parabola xM yM)
  (hN : Parabola xN yN)
  (hLine : Line k xM yM ∧ Line k xN yN)
  (x₀ y₀ : ℝ)
  (hMidpoint : x₀ = (xM + xN) / 2 ∧ y₀ = (yM + yN) / 2)
  (a : ℝ)
  (ha : a > 0)
  (hVerticalBisector : VerticalBisector x₀ y₀ k a 0)
  (n : ℝ)
  (hn : n = Distance xM yM (Focus.1) (Focus.2) + Distance xN yN (Focus.1) (Focus.2)) :
  2 * a - n = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_property_l1269_126905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_FQ_length_l1269_126900

-- Define the triangle DEF
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  right_angle_at_E : (E.1 - D.1) * (F.1 - E.1) + (E.2 - D.2) * (F.2 - E.2) = 0
  DE_length : Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = 3
  DF_length : Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2) = Real.sqrt 34

-- Define the circle
structure Circle (tri : Triangle) where
  center : ℝ × ℝ
  radius : ℝ
  center_on_DE : ∃ t : ℝ, center = (tri.D.1 + t * (tri.E.1 - tri.D.1), tri.D.2 + t * (tri.E.2 - tri.D.2))
  tangent_to_DF : ∃ Q : ℝ × ℝ, (Q.1 - center.1)^2 + (Q.2 - center.2)^2 = radius^2 ∧
                               (Q.1 - tri.D.1) * (tri.F.1 - tri.D.1) + (Q.2 - tri.D.2) * (tri.F.2 - tri.D.2) = 
                               Real.sqrt ((Q.1 - tri.D.1)^2 + (Q.2 - tri.D.2)^2) * Real.sqrt ((tri.F.1 - tri.D.1)^2 + (tri.F.2 - tri.D.2)^2)
  tangent_to_EF : (tri.E.1 - center.1)^2 + (tri.E.2 - center.2)^2 = radius^2

-- Theorem statement
theorem FQ_length (tri : Triangle) (circle : Circle tri) : 
  ∃ Q : ℝ × ℝ, Real.sqrt ((tri.F.1 - Q.1)^2 + (tri.F.2 - Q.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_FQ_length_l1269_126900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1269_126968

noncomputable section

def f (a b x : ℝ) : ℝ := (a * Real.log x - b * Real.exp x) / x

theorem problem_statement (a b : ℝ) (ha : a ≠ 0) :
  -- Part I
  (∀ x, x > 0 → (deriv (f a b)) x = 0 ↔ x = Real.exp 1) →
  (∃ x₀, IsLocalMin (f a b) x₀) →
  a < 0 ∧
  -- Part II(i)
  (a = 1 ∧ b = 1 → ∀ x > 0, x * f 1 1 x + 2 < 0) ∧
  -- Part II(ii)
  (a = 1 ∧ b = -1 →
    IsGreatest {m : ℝ | ∀ x > 1, x * f 1 (-1) x > Real.exp 1 + m * (x - 1)} (1 + Real.exp 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1269_126968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l1269_126987

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 2*y + 3 = 0

-- Define the line equation
def line_eq (x y a : ℝ) : Prop :=
  x + a*y - 1 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ :=
  (3, 1)

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x y a : ℝ) : ℝ :=
  |x + a*y - 1| / Real.sqrt (1 + a^2)

-- Theorem statement
theorem circle_line_distance (a : ℝ) :
  (∃ x y : ℝ, circle_eq x y) →
  (distance_point_to_line (circle_center.1) (circle_center.2) a = 1) →
  a = -3/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_l1269_126987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_implies_a_range_l1269_126961

-- Define the function f(x) = log_a(2x - a)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x - a) / Real.log a

-- Define the interval [1/2, 2/3]
def interval : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2/3}

-- State the theorem
theorem f_positive_implies_a_range (a : ℝ) :
  (∀ x ∈ interval, f a x > 0) ↔ (1/3 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_implies_a_range_l1269_126961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_above_line_l1269_126936

noncomputable def points : List (ℝ × ℝ) := [(4, 15), (8, 25), (14, 42), (19, 48), (22, 60)]

noncomputable def isAboveLine (p : ℝ × ℝ) : Bool :=
  p.2 > 3 * p.1 + 4

noncomputable def sumXCoordinatesAboveLine (points : List (ℝ × ℝ)) : ℝ :=
  (points.filter isAboveLine).map (·.1) |>.sum

theorem sum_x_coordinates_above_line :
  sumXCoordinatesAboveLine points = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_above_line_l1269_126936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1269_126946

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω (x + Real.pi / ω) = f ω x) 
  (h_smallest_period : ∀ T, T > 0 → (∀ x, f ω (x + T) = f ω x) → T ≥ Real.pi / ω) :
  ω = 2 ∧ 
  ∀ α : ℝ, α ∈ Set.Ioo 0 (Real.pi / 8) → f ω α = 2/3 → Real.cos (2 * α) = (1 + 2 * Real.sqrt 6) / 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1269_126946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_complement_union_l1269_126947

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}

def A : Finset ℕ := U.filter (fun x => x^2 - 3*x + 2 = 0)

def B : Finset ℕ := U.filter (fun x => ∃ a ∈ A, x = 2*a)

theorem number_of_subsets_of_complement_union :
  Fintype.card (Finset.powerset (U \ (A ∪ B))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_complement_union_l1269_126947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_diff_degree_l1269_126951

/-- The polynomial representing the difference of areas of two circles -/
noncomputable def circle_area_diff (R r : ℝ) : Polynomial ℝ :=
  Polynomial.monomial 2 Real.pi * Polynomial.X - Polynomial.monomial 2 Real.pi * (Polynomial.X.comp (Polynomial.C r))

/-- The degree of the polynomial representing the difference of areas of two circles is 2 -/
theorem circle_area_diff_degree (R r : ℝ) :
  (circle_area_diff R r).degree = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_diff_degree_l1269_126951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_preserves_order_l1269_126924

theorem cube_root_preserves_order (a b : ℝ) (h : a > b) : a^(1/3) > b^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_preserves_order_l1269_126924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_pens_gain_percentage_l1269_126949

/-- A trader sells pens. The gain from selling 100 pens equals the cost of 40 pens. -/
def TraderPens (cost_per_pen : ℝ) : Prop :=
  40 * cost_per_pen = 40 * cost_per_pen

/-- The gain percentage for the trader selling pens -/
noncomputable def GainPercentage (cost_per_pen : ℝ) : ℝ :=
  let gain := 40 * cost_per_pen
  let cost := 100 * cost_per_pen
  (gain / cost) * 100

theorem trader_pens_gain_percentage (cost_per_pen : ℝ) (h : cost_per_pen > 0) :
  TraderPens cost_per_pen → GainPercentage cost_per_pen = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_pens_gain_percentage_l1269_126949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_l1269_126948

/-- Given complex numbers a, b, c forming an equilateral triangle with side length 24
    in the complex plane, if |a + b + c| = 48, then |ab + ac + bc| = 768 -/
theorem equilateral_triangle_complex (a b c : ℂ) : 
  (∀ (x y : ℂ), x ∈ ({a, b, c} : Set ℂ) ∧ y ∈ ({a, b, c} : Set ℂ) ∧ x ≠ y → Complex.abs (x - y) = 24) →
  Complex.abs (a + b + c) = 48 →
  Complex.abs (a * b + a * c + b * c) = 768 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_l1269_126948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_from_2_to_infinity_l1269_126943

/-- Definition of the function f(n) -/
noncomputable def f (n : ℕ) : ℝ := ∑' k : ℕ, (1 : ℝ) / ((k + 1) ^ n)

/-- The main theorem to prove -/
theorem sum_of_f_from_2_to_infinity : ∑' n : ℕ, f (n + 2) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_from_2_to_infinity_l1269_126943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l1269_126921

/-- Helper function to represent the area of a quadrilateral -/
noncomputable def area_quadrilateral (a b c d : ℝ) : ℝ := 
  sorry

/-- Given a quadrilateral with consecutive side lengths a, b, c, and d, and area S,
    prove that S ≤ (1/4)(a+b)(c+d). -/
theorem quadrilateral_area_inequality (a b c d S : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (hS : S > 0) (hquad : S = area_quadrilateral a b c d) : 
  S ≤ (1/4) * (a + b) * (c + d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l1269_126921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_photos_theorem_l1269_126941

/-- Represents a tourist's photo-taking behavior -/
structure Tourist where
  photos : Fin 3 → Bool

/-- The problem setup -/
def TouristGroup (n : ℕ) : Type :=
  { group : Fin n → Tourist // 
    ∀ i j : Fin n, i ≠ j → 
      ∃ m : Fin 3, (group i).photos m = true ∨ (group j).photos m = true }

/-- The total number of photos taken by a group -/
def totalPhotos {n : ℕ} (group : TouristGroup n) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin n)) fun i => 
    Finset.sum (Finset.univ : Finset (Fin 3)) fun m => 
      if (group.val i).photos m then 1 else 0

/-- The main theorem -/
theorem min_photos_theorem :
  ∀ (group : TouristGroup 42),
    totalPhotos group ≥ 123 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_photos_theorem_l1269_126941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1269_126994

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - Real.cos (ω * x)

-- State the theorem
theorem omega_range (ω : ℝ) : 
  ω > 0 ∧ 
  (∀ x y, -2*π/5 ≤ x ∧ x < y ∧ y ≤ 3*π/4 → f ω x < f ω y) ∧ 
  (∃! x₀, 0 ≤ x₀ ∧ x₀ ≤ π ∧ f ω x₀ = 2) →
  2/3 ≤ ω ∧ ω ≤ 5/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1269_126994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_unit_vectors_at_120_degrees_l1269_126984

theorem sum_of_three_unit_vectors_at_120_degrees (e₁ e₂ e₃ : ℝ × ℝ × ℝ) :
  ‖e₁‖ = 1 →
  ‖e₂‖ = 1 →
  ‖e₃‖ = 1 →
  e₁ • e₂ = -1/2 →
  e₁ • e₃ = -1/2 →
  e₂ • e₃ = -1/2 →
  ‖e₁ + e₂ + e₃‖ = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_unit_vectors_at_120_degrees_l1269_126984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_propositions_truth_l1269_126999

-- Define the basic propositions
variable (p₁ p₂ p₃ p₄ : Prop)

-- Define the conditions
axiom h₁ : p₁
axiom h₂ : ¬p₂
axiom h₃ : ¬p₃
axiom h₄ : p₄

-- Define the compound propositions
def prop₁ (p₁ p₄ : Prop) : Prop := p₁ ∧ p₄
def prop₂ (p₁ p₂ : Prop) : Prop := p₁ ∧ p₂
def prop₃ (p₂ p₃ : Prop) : Prop := ¬p₂ ∨ p₃
def prop₄ (p₃ p₄ : Prop) : Prop := ¬p₃ ∨ ¬p₄

-- State the theorem
theorem compound_propositions_truth (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) :
  prop₁ p₁ p₄ ∧ ¬(prop₂ p₁ p₂) ∧ prop₃ p₂ p₃ ∧ prop₄ p₃ p₄ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_propositions_truth_l1269_126999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_intersection_l1269_126912

/-- Given two lines in the xy-plane -/
def line1 (x y : ℚ) : Prop := 4 * y - 3 * x = 16
def line2 (x y : ℚ) : Prop := 3 * y + 4 * x = 15

/-- The slope of a line given its coefficients -/
def line_slope (a b : ℚ) : ℚ := -a / b

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

/-- The intersection point of the two lines -/
def intersection : ℚ × ℚ := (12/25, 109/25)

theorem perpendicular_lines_intersection :
  perpendicular (line_slope 3 4) (line_slope (-3) 4) ∧
  line1 intersection.1 intersection.2 ∧
  line2 intersection.1 intersection.2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_intersection_l1269_126912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1269_126981

-- Define the set of positive real numbers
def PositiveReals : Type := {x : ℝ // x > 0}

-- Define the property that f must satisfy
def SatisfiesFunctionalEquation (f : PositiveReals → PositiveReals) : Prop :=
  ∀ x y : PositiveReals, (f x).val * (f (⟨y.val * (f x).val, sorry⟩)).val = (f ⟨x.val + y.val, sorry⟩).val

-- Theorem statement
theorem functional_equation_solution :
  ∀ f : PositiveReals → PositiveReals, 
  SatisfiesFunctionalEquation f → 
  (∀ x : PositiveReals, f x = ⟨1, sorry⟩) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1269_126981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharks_daily_feed_l1269_126969

-- Define the number of buckets fed to sharks each day
noncomputable def sharks_buckets : ℝ := 4

-- Define the conditions
noncomputable def dolphins_buckets : ℝ := sharks_buckets / 2
noncomputable def other_animals_buckets : ℝ := 5 * sharks_buckets
noncomputable def total_buckets_per_day : ℝ := sharks_buckets + dolphins_buckets + other_animals_buckets
def total_days : ℕ := 21
def total_buckets : ℕ := 546

-- Theorem statement
theorem sharks_daily_feed :
  sharks_buckets = 4 ∧
  dolphins_buckets = sharks_buckets / 2 ∧
  other_animals_buckets = 5 * sharks_buckets ∧
  total_buckets_per_day = sharks_buckets + dolphins_buckets + other_animals_buckets ∧
  (total_buckets : ℝ) = total_buckets_per_day * total_days :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharks_daily_feed_l1269_126969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l1269_126922

/-- The equation of a circle with radius √5 centered at the origin -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- The point P on the circle -/
def P : ℝ × ℝ := (1, 2)

/-- The proposed equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := x + 2*y - 5 = 0

/-- Theorem stating that the proposed equation is indeed the tangent line to the circle at point P -/
theorem tangent_line_at_P :
  (circle_equation P.1 P.2) →
  (∀ x y : ℝ, circle_equation x y → (x - P.1) * (x - P.1) + (y - P.2) * (y - P.2) ≥ 0) →
  (∀ x y : ℝ, tangent_line x y ↔ 
    (x - P.1) * (x - P.1) + (y - P.2) * (y - P.2) = 0 ∨
    ¬(circle_equation x y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l1269_126922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_2_fourth_power_l1269_126938

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom fg_condition : ∀ x, x ≥ 1 → f (g x) = x^3
axiom gf_condition : ∀ x, x ≥ 1 → g (f x) = x^4
axiom g_16 : g 16 = 8

-- State the theorem to be proved
theorem g_2_fourth_power : (g 2)^4 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_2_fourth_power_l1269_126938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisects_circle_sufficient_not_necessary_another_line_bisects_circle_l1269_126928

-- Define the line
def line (x y : ℝ) : Prop := x - y = 0

-- Define what it means for a line to bisect the circle
def bisects_circle (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₀ y₀ : ℝ), l x₀ y₀ ∧ x₀^2 + y₀^2 = 0

-- Theorem statement
theorem line_bisects_circle_sufficient_not_necessary :
  (bisects_circle line) ∧ 
  (∃ (l : ℝ → ℝ → Prop), l ≠ line ∧ bisects_circle l) := by
  sorry

-- Example of another line that bisects the circle
def another_line (x y : ℝ) : Prop := x + y = 0

-- Theorem to show that another_line also bisects the circle
theorem another_line_bisects_circle :
  bisects_circle another_line := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisects_circle_sufficient_not_necessary_another_line_bisects_circle_l1269_126928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1269_126995

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Conditions
  a > 0 → b > 0 → c > 0 →  -- Positive side lengths
  b^2 = a * c →  -- Geometric sequence condition
  B = π / 3 →  -- 60 degrees in radians
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos B →  -- Cosine rule
  b / Real.sin B = a / Real.sin A ∧ b / Real.sin B = c / Real.sin C →  -- Sine rule
  -- Conclusions
  (a = b ∧ b = c) ∧  -- Triangle is equilateral
  Real.sin B = Real.sqrt 3 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1269_126995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_price_change_l1269_126950

/-- Calculates the price after a percentage change --/
noncomputable def price_after_change (initial_price : ℝ) (percentage_change : ℝ) : ℝ :=
  initial_price * (1 + percentage_change / 100)

/-- Rounds a real number to the nearest integer --/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem gasoline_price_change :
  ∀ (initial_price : ℝ) (y : ℝ),
    initial_price > 0 →
    let price1 := price_after_change initial_price 25
    let price2 := price_after_change price1 (-25)
    let price3 := price_after_change price2 30
    let price4 := price_after_change price3 (-y)
    price4 = initial_price →
    round_to_nearest y = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_price_change_l1269_126950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_xy_length_l1269_126925

theorem right_triangle_xy_length (X Y Z : ℝ × ℝ) 
  (h_right_angle : (Y.1 - X.1) * (Z.2 - X.2) = (Y.2 - X.2) * (Z.1 - X.1))
  (h_angle_x : Real.arccos ((Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2)) / 
    (((Y.1 - X.1)^2 + (Y.2 - X.2)^2).sqrt * ((Z.1 - X.1)^2 + (Z.2 - X.2)^2).sqrt) = π / 3)
  (h_hypotenuse : Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2) = 12) :
  Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_xy_length_l1269_126925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l1269_126930

noncomputable def complex_number : ℂ := (2 + 3 * Complex.I) / Complex.I

theorem complex_number_in_fourth_quadrant :
  0 < complex_number.re ∧ complex_number.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l1269_126930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_equals_naturals_l1269_126945

def is_valid_subset (S : Set ℕ) : Prop :=
  (∀ n : ℕ, ∃ k ∈ S, n ≤ k ∧ k < n + 2003) ∧
  (∀ n ∈ S, n > 1 → (n / 2 : ℕ) ∈ S)

theorem subset_equals_naturals (S : Set ℕ) (h : is_valid_subset S) : S = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_equals_naturals_l1269_126945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_number_puzzle_solution_l1269_126975

/-- Represents a digit in the cross-number puzzle -/
def Digit := Fin 9

/-- Represents a two-digit number in the cross-number puzzle -/
def TwoDigitNumber := Fin 99

/-- Represents a three-digit number in the cross-number puzzle -/
def ThreeDigitNumber := Fin 999

/-- Check if a number is a perfect square -/
def is_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

/-- Check if a number is one less than a cube -/
def is_one_less_than_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k - 1 = n

/-- The highest common factor of two numbers -/
def hcf (a b : ℕ) : ℕ := Nat.gcd a b

/-- The cross-number puzzle configuration -/
structure CrossNumberPuzzle where
  across_1 : TwoDigitNumber
  down_1 : TwoDigitNumber
  across_3 : ThreeDigitNumber
  down_2 : ThreeDigitNumber
  down_4 : TwoDigitNumber
  across_5 : TwoDigitNumber

/-- The conditions of the cross-number puzzle -/
def valid_configuration (p : CrossNumberPuzzle) : Prop :=
  is_square p.across_1.val ∧
  p.down_1.val = p.down_4.val - 11 ∧
  p.across_3.val = 829 ∧
  is_one_less_than_cube p.down_2.val ∧
  p.down_4.val = hcf p.down_1.val p.across_5.val ∧
  is_square p.across_5.val ∧
  (p.down_1.val * p.down_4.val) % 10 ≠ 0 ∧
  p.down_1.val * p.down_1.val > 1

theorem cross_number_puzzle_solution :
  ∃! p : CrossNumberPuzzle, valid_configuration p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_number_puzzle_solution_l1269_126975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1269_126935

noncomputable def a (ω x : ℝ) : ℝ × ℝ := (Real.sin (ω * x) + Real.cos (ω * x), Real.sqrt 3 * Real.cos (ω * x))

noncomputable def b (ω x : ℝ) : ℝ × ℝ := (Real.cos (ω * x) - Real.sin (ω * x), 2 * Real.sin (ω * x))

noncomputable def f (ω x : ℝ) : ℝ := (a ω x).1 * (b ω x).1 + (a ω x).2 * (b ω x).2

noncomputable def area_triangle (A B C : ℝ) : ℝ := sorry

theorem problem_solution (ω : ℝ) (A B C : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω (x + π / (2 * ω)) = f ω x) →
  f ω C = 1 →
  Real.sin C + Real.sin (B - A) = 3 * Real.sin (2 * A) →
  ω = 1 ∧
  ((Real.sqrt 3 / 3 * 2 : ℝ) = area_triangle A B C ∨ (3 * Real.sqrt 3 / 7 : ℝ) = area_triangle A B C) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1269_126935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l1269_126993

/-- Parabola struct representing y² = ax -/
structure Parabola where
  a : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line with slope m passing through point p -/
structure Line where
  m : ℝ
  p : Point

noncomputable def focus (p : Parabola) : Point :=
  { x := p.a / 4, y := 0 }

noncomputable def intersect_parabola_line (p : Parabola) (l : Line) : Point × Point := sorry

noncomputable def distance (p1 p2 : Point) : ℝ := sorry

theorem parabola_chord_length 
  (p : Parabola)
  (l : Line)
  (h1 : p.a = 5)
  (h2 : l.m = Real.sqrt 3)
  (h3 : l.p = focus p) :
  let (A, B) := intersect_parabola_line p l
  distance A B = 20 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l1269_126993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1269_126915

open Real

theorem trigonometric_equation_solution (x : ℝ) :
  (3 * sin x ^ 2 - 4 * cos x ^ 2 = (sin (2 * x)) / 2) ↔
  (∃ k : ℤ, x = π * (3 / 4 + k : ℝ)) ∨
  (∃ k : ℤ, x = π * (0.2952 + k : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1269_126915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_difference_not_22_l1269_126933

def is_product_of_four_distinct_primes (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ : ℕ, 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n = p₁ * p₂ * p₃ * p₄

noncomputable def divisors (n : ℕ) : List ℕ :=
  (Finset.range (n + 1)).filter (λ d => n % d = 0) |>.toList

theorem divisor_difference_not_22 (n : ℕ) 
  (h1 : is_product_of_four_distinct_primes n) 
  (h2 : n < 1995) : 
  let d := divisors n
  (d.length = 16) → 
  (d[8]! - d[7]! ≠ 22) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_difference_not_22_l1269_126933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_percentage_l1269_126901

/-- Represents the profit percentage calculation for a shopkeeper selling two products --/
theorem shopkeeper_profit_percentage :
  let weight_A : ℚ := 15 / 16
  let weight_B : ℚ := 47 / 50
  let cost_price_A : ℚ := 12
  let cost_price_B : ℚ := 18
  let selling_price_A : ℚ := cost_price_A
  let selling_price_B : ℚ := cost_price_B
  let profit_A : ℚ := selling_price_A - (weight_A * cost_price_A)
  let profit_B : ℚ := selling_price_B - (weight_B * cost_price_B)
  let total_profit : ℚ := profit_A + profit_B
  let total_cost : ℚ := (weight_A * cost_price_A) + (weight_B * cost_price_B)
  let profit_percentage : ℚ := (total_profit / total_cost) * 100
  ∃ (ε : ℚ), abs (profit_percentage - 6.5) < ε ∧ ε > 0 := by
  sorry

#eval (((12 - (15/16 * 12)) + (18 - (47/50 * 18))) / ((15/16 * 12) + (47/50 * 18))) * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_percentage_l1269_126901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_reciprocal_squares_l1269_126970

noncomputable section

open Real

-- Define the curve E in polar coordinates
def E (ρ θ : ℝ) : Prop := ρ^2 * (1/3 * cos θ^2 + 1/2 * sin θ^2) = 1

-- Define a point on the curve
def point_on_E (ρ θ : ℝ) : Prop := E ρ θ

-- Define perpendicularity in polar coordinates
def perpendicular (θ₁ θ₂ : ℝ) : Prop := θ₂ = θ₁ + Real.pi/2

-- Main theorem
theorem constant_sum_of_reciprocal_squares :
  ∀ (ρ₁ ρ₂ θ₁ θ₂ : ℝ),
    point_on_E ρ₁ θ₁ →
    point_on_E ρ₂ θ₂ →
    perpendicular θ₁ θ₂ →
    1 / ρ₁^2 + 1 / ρ₂^2 = 5/6 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_reciprocal_squares_l1269_126970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l1269_126985

-- Define the function f as noncomputable
noncomputable def f : ℝ → ℝ := λ x => (1/4) * x^2 + 2*x + 19/4

-- State the theorem
theorem function_equivalence : 
  (∀ x : ℝ, f (2*x - 3) = x^2 + x + 1) ↔ 
  (∀ x : ℝ, f x = (1/4) * x^2 + 2*x + 19/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l1269_126985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_empty_proper_subsets_count_l1269_126931

def A : Set ℤ := {x | x ≠ 0 ∧ (2 * x - 5) / x ≤ 1}

theorem non_empty_proper_subsets_count : Finset.card (Finset.powerset {1, 2, 3, 4, 5} \ {∅, {1, 2, 3, 4, 5}}) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_empty_proper_subsets_count_l1269_126931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_includes_32_cents_l1269_126923

/-- Represents the annual interest rate as a decimal -/
noncomputable def annual_rate : ℚ := 8 / 100

/-- Represents the time period in years -/
noncomputable def time : ℚ := 1 / 4

/-- Represents the fee deducted -/
def fee : ℚ := 5

/-- Represents the final amount in the bank -/
def final_amount : ℚ := 31744 / 100

/-- Calculates the interest amount given the principal -/
noncomputable def interest (principal : ℚ) : ℚ :=
  principal * annual_rate * time

/-- Theorem stating that the interest credited includes 32 cents -/
theorem interest_includes_32_cents (P : ℚ) :
  P * (1 + annual_rate * time) - fee = final_amount →
  ∃ n : ℕ, interest P = n + 32 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_includes_32_cents_l1269_126923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_configuration_l1269_126974

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  lines : Finset (Set (ℝ × ℝ))
  unique_intersections : ∀ l₁ l₂, l₁ ∈ lines → l₂ ∈ lines → l₁ ≠ l₂ → ∃! p, p ∈ l₁ ∩ l₂

/-- The set of all intersection points in a configuration -/
def intersection_points (config : LineConfiguration) : Set (ℝ × ℝ) :=
  {p | ∃ l₁ l₂, l₁ ∈ config.lines ∧ l₂ ∈ config.lines ∧ l₁ ≠ l₂ ∧ p ∈ l₁ ∩ l₂}

/-- The property that any 8 lines leave at least one intersection point uncovered -/
def property_8 (config : LineConfiguration) : Prop :=
  ∀ subset : Finset (Set (ℝ × ℝ)), subset ⊆ config.lines → subset.card = 8 →
    ∃ p ∈ intersection_points config, ∀ l ∈ subset, p ∉ l

/-- The property that any 9 lines cover all intersection points -/
def property_9 (config : LineConfiguration) : Prop :=
  ∀ subset : Finset (Set (ℝ × ℝ)), subset ⊆ config.lines → subset.card = 9 →
    ∀ p ∈ intersection_points config, ∃ l ∈ subset, p ∈ l

/-- The main theorem stating the existence of a configuration with the desired properties -/
theorem exists_special_configuration :
  ∃ config : LineConfiguration,
    config.lines.card = 10 ∧
    property_8 config ∧
    property_9 config :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_configuration_l1269_126974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integer_power_coefficients_l1269_126911

theorem sum_of_integer_power_coefficients (n : ℕ) (hn : 0 < n) :
  let expansion := (fun x => (x + Real.sqrt x + 1) ^ (2 * n + 1))
  let sum_of_coefficients := (expansion 1 + expansion (-1)) / 2
  sum_of_coefficients = (3 ^ (2 * n + 1) + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integer_power_coefficients_l1269_126911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_position_after_five_rotations_l1269_126958

/-- Represents the four sides of a square -/
inductive SquareSide
  | Top
  | Right
  | Bottom
  | Left
deriving Repr

/-- Calculates the new side after a given number of rotations -/
def new_side_after_rotations (initial_side : SquareSide) (num_rotations : ℕ) : SquareSide :=
  match (initial_side, num_rotations % 4) with
  | (SquareSide.Bottom, 0) => SquareSide.Bottom
  | (SquareSide.Bottom, 1) => SquareSide.Right
  | (SquareSide.Bottom, 2) => SquareSide.Top
  | (SquareSide.Bottom, 3) => SquareSide.Left
  | _ => SquareSide.Bottom  -- Default case, should not occur in this problem

theorem triangle_position_after_five_rotations :
  new_side_after_rotations SquareSide.Bottom 5 = SquareSide.Right :=
by sorry

#eval new_side_after_rotations SquareSide.Bottom 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_position_after_five_rotations_l1269_126958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l1269_126919

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_relation (O A B C : V) (a b c : V)
  (h1 : A - O = a) (h2 : B - O = b) (h3 : C - O = c) 
  (h4 : C - A = (3 : ℝ) • (C - B)) : 
  c = -(1/2 : ℝ) • a + (3/2 : ℝ) • b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l1269_126919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_11001_equals_25_l1269_126960

/-- Converts a binary digit (0 or 1) to its decimal value -/
def binaryToDecimal (digit : Nat) : Nat :=
  if digit = 0 then 0 else if digit = 1 then 1 else 0

/-- Calculates the decimal value of a binary number represented as a list of digits -/
def binaryListToDecimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + binaryToDecimal b * 2^i) 0

theorem binary_11001_equals_25 :
  binaryListToDecimal [1, 0, 0, 1, 1] = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_11001_equals_25_l1269_126960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_find_A_l1269_126978

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def is_acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def satisfies_conditions (t : Triangle) : Prop :=
  is_acute_triangle t ∧
  t.a = t.b * Real.cos t.C + (Real.sqrt 3 / 3) * t.c * Real.sin t.B

-- Theorem 1
theorem find_c (t : Triangle) 
  (h1 : satisfies_conditions t)
  (h2 : t.a = 2)
  (h3 : t.b = Real.sqrt 7) :
  t.c = 3 := by sorry

-- Theorem 2
theorem find_A (t : Triangle)
  (h1 : satisfies_conditions t)
  (h2 : Real.sqrt 3 * Real.sin (2 * t.A - Real.pi/6) - 2 * (Real.sin (t.C - Real.pi/12))^2 = 0) :
  t.A = Real.pi/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_find_A_l1269_126978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_side_length_ratio_l1269_126914

theorem cube_side_length_ratio (volume_ratio : ℚ) 
  (h_volume_ratio : volume_ratio = 1232 / 405) : 
  ∃ (a b c d : ℕ), 
    (a : ℝ) * Real.sqrt (b : ℝ) / ((c : ℝ) * Real.sqrt (d : ℝ)) = (volume_ratio : ℝ) ^ (1/3 : ℝ) ∧ 
    a + b + c + d = 467 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_side_length_ratio_l1269_126914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_equation_l1269_126992

/-- Represents the monthly production growth rate. -/
def x : ℝ := sorry

/-- The total production in the first quarter. -/
def totalProduction : ℝ := 364

/-- The production in January. -/
def januaryProduction : ℝ := 100

/-- Theorem stating the equation for the total production in the first quarter. -/
theorem production_equation : 
  januaryProduction + januaryProduction * (1 + x) + januaryProduction * (1 + x)^2 = totalProduction :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_equation_l1269_126992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_as_fraction_sum_of_a_b_l1269_126956

/-- The probability of encountering 4 heads before 3 tails in continuous fair coin flips -/
def q : ℚ :=
  1 / 4

/-- q can be written as a/b where a and b are relatively prime positive integers -/
theorem q_as_fraction : ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ Nat.Coprime a b ∧ q = a / b := by
  sorry

/-- The sum of a and b is 5 -/
theorem sum_of_a_b : ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ Nat.Coprime a b ∧ q = a / b ∧ a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_as_fraction_sum_of_a_b_l1269_126956
