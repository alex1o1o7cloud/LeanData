import Mathlib

namespace NUMINAMATH_CALUDE_equivalent_division_l3811_381102

theorem equivalent_division (x : ℝ) : (x / (3/9)) * (2/15) = x / 2.5 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_division_l3811_381102


namespace NUMINAMATH_CALUDE_store_earnings_is_400_l3811_381167

/-- Calculates the total earnings of a clothing store selling shirts and jeans -/
def store_earnings (num_shirts : ℕ) (num_jeans : ℕ) (shirt_price : ℕ) : ℕ :=
  let jeans_price := 2 * shirt_price
  num_shirts * shirt_price + num_jeans * jeans_price

/-- Theorem: The clothing store will earn $400 if all shirts and jeans are sold -/
theorem store_earnings_is_400 :
  store_earnings 20 10 10 = 400 := by
sorry

end NUMINAMATH_CALUDE_store_earnings_is_400_l3811_381167


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l3811_381199

theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = !![1, 4; -2, -7] →
  (A^3)⁻¹ = !![41, 144; -72, -247] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l3811_381199


namespace NUMINAMATH_CALUDE_april_flower_sale_earnings_l3811_381126

/-- April's flower sale earnings calculation --/
theorem april_flower_sale_earnings : 
  ∀ (initial_roses final_roses price_per_rose : ℕ),
  initial_roses = 9 →
  final_roses = 4 →
  price_per_rose = 7 →
  (initial_roses - final_roses) * price_per_rose = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_april_flower_sale_earnings_l3811_381126


namespace NUMINAMATH_CALUDE_area_of_three_presentable_set_l3811_381132

/-- A complex number is three-presentable if there exists a complex number w
    with |w| = 3 such that z = w - 1/w -/
def ThreePresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 3 ∧ z = w - 1 / w

/-- T is the set of all three-presentable complex numbers -/
def T : Set ℂ :=
  {z : ℂ | ThreePresentable z}

/-- The area of a set in the complex plane -/
noncomputable def AreaInside (S : Set ℂ) : ℝ := sorry

theorem area_of_three_presentable_set :
  AreaInside T = (80 / 9) * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_three_presentable_set_l3811_381132


namespace NUMINAMATH_CALUDE_fractional_equation_solutions_l3811_381173

/-- The fractional equation in terms of x and m -/
def fractional_equation (x m : ℝ) : Prop :=
  3 * x / (x - 1) = m / (x - 1) + 2

theorem fractional_equation_solutions :
  (∃! x : ℝ, fractional_equation x 4) ∧
  (∀ x : ℝ, ¬fractional_equation x 3) ∧
  (∀ m : ℝ, m ≠ 3 → ∃ x : ℝ, fractional_equation x m) :=
sorry

end NUMINAMATH_CALUDE_fractional_equation_solutions_l3811_381173


namespace NUMINAMATH_CALUDE_function_property_l3811_381114

/-- Piecewise function f(x) as described in the problem -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then a * x + b else 7 - 2 * x

/-- The main theorem to prove -/
theorem function_property (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3811_381114


namespace NUMINAMATH_CALUDE_droid_coffee_usage_l3811_381181

/-- The number of bags of coffee beans Droid uses in a week -/
def weekly_coffee_usage : ℕ :=
  let morning_usage := 3
  let afternoon_usage := 3 * morning_usage
  let evening_usage := 2 * morning_usage
  let daily_usage := morning_usage + afternoon_usage + evening_usage
  7 * daily_usage

/-- Theorem stating that Droid uses 126 bags of coffee beans in a week -/
theorem droid_coffee_usage : weekly_coffee_usage = 126 := by
  sorry

end NUMINAMATH_CALUDE_droid_coffee_usage_l3811_381181


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3811_381142

/-- Given a hyperbola and a parabola, prove that if the left focus of the hyperbola
    lies on the directrix of the parabola, then p = 4 -/
theorem hyperbola_parabola_intersection (p : ℝ) (hp : p > 0) : 
  (∃ x y : ℝ, x^2 / 3 - 16 * y^2 / p^2 = 1) →  -- hyperbola equation
  (∃ x y : ℝ, y^2 = 2 * p * x) →              -- parabola equation
  (- Real.sqrt (3 + p^2 / 16) = - p / 2) →    -- left focus on directrix condition
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3811_381142


namespace NUMINAMATH_CALUDE_extreme_value_implies_f_2_l3811_381190

/-- A function f with an extreme value at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_value_implies_f_2 (a b : ℝ) :
  (f' a b 1 = 0) →  -- f has an extreme value at x = 1
  (f a b 1 = 10) →  -- The extreme value is 10
  (f a b 2 = 11 ∨ f a b 2 = 18) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_f_2_l3811_381190


namespace NUMINAMATH_CALUDE_locus_of_center_C_l3811_381179

/-- Circle C₁ with equation x² + y² + 4y + 3 = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 4*p.2 + 3 = 0}

/-- Circle C₂ with equation x² + y² - 4y - 77 = 0 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.2 - 77 = 0}

/-- The locus of the center of circle C -/
def locus_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 / 25 + p.1^2 / 21 = 1}

/-- Theorem stating that the locus of the center of circle C forms an ellipse
    given the tangency conditions with C₁ and C₂ -/
theorem locus_of_center_C (C : Set (ℝ × ℝ)) :
  (∃ r : ℝ, ∀ p ∈ C, ∃ q ∈ C₁, ‖p - q‖ = r) →  -- C is externally tangent to C₁
  (∃ R : ℝ, ∀ p ∈ C, ∃ q ∈ C₂, ‖p - q‖ = R) →  -- C is internally tangent to C₂
  C = locus_C :=
sorry

end NUMINAMATH_CALUDE_locus_of_center_C_l3811_381179


namespace NUMINAMATH_CALUDE_slope_angle_range_l3811_381131

-- Define the slope k and the angle θ
variable (k : ℝ) (θ : ℝ)

-- Define the condition that the lines intersect in the first quadrant
def intersect_in_first_quadrant (k : ℝ) : Prop :=
  (3 + Real.sqrt 3) / (1 + k) > 0 ∧ (3 * k - Real.sqrt 3) / (1 + k) > 0

-- Define the relationship between k and θ
def slope_angle_relation (k θ : ℝ) : Prop :=
  k = Real.tan θ

-- State the theorem
theorem slope_angle_range (h1 : intersect_in_first_quadrant k) 
  (h2 : slope_angle_relation k θ) : 
  θ > Real.pi / 6 ∧ θ < Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_slope_angle_range_l3811_381131


namespace NUMINAMATH_CALUDE_range_of_m_plus_n_l3811_381119

/-- Given a function f(x) = me^x + x^2 + nx where the set of roots of f and f∘f are equal and non-empty,
    prove that the range of m + n is [0, 4). -/
theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x, m * Real.exp x + x^2 + n * x = 0) →
  {x | m * Real.exp x + x^2 + n * x = 0} = {x | m * Real.exp (m * Real.exp x + x^2 + n * x) + 
    (m * Real.exp x + x^2 + n * x)^2 + n * (m * Real.exp x + x^2 + n * x) = 0} →
  m + n ∈ Set.Icc 0 4 ∧ ¬(m + n = 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_plus_n_l3811_381119


namespace NUMINAMATH_CALUDE_max_subjects_per_teacher_l3811_381168

theorem max_subjects_per_teacher 
  (total_subjects : Nat) 
  (min_teachers : Nat) 
  (h1 : total_subjects = 18) 
  (h2 : min_teachers = 6) : 
  Nat.ceil (total_subjects / min_teachers) = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_subjects_per_teacher_l3811_381168


namespace NUMINAMATH_CALUDE_linear_function_passes_through_point_l3811_381171

/-- A linear function of the form y = kx + k passes through the point (-1, 0) for any non-zero k. -/
theorem linear_function_passes_through_point (k : ℝ) (hk : k ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ k * x + k
  f (-1) = 0 := by sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_point_l3811_381171


namespace NUMINAMATH_CALUDE_function_range_l3811_381100

theorem function_range (a : ℝ) (h_a : a ≠ 0) :
  (∀ x : ℝ, a * Real.sin x - 1/2 * Real.cos (2*x) + a - 3/a + 1/2 ≤ 0) →
  (0 < a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l3811_381100


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l3811_381191

theorem line_ellipse_intersection
  (m n : ℝ)
  (h1 : m^2 + n^2 < 3)
  (h2 : 0 < m^2 + n^2) :
  ∀ (a b : ℝ), ∃! (x y : ℝ),
    x^2 / 7 + y^2 / 3 = 1 ∧
    y = a*x + b ∧
    a*m + b = n :=
by sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l3811_381191


namespace NUMINAMATH_CALUDE_horizontal_line_slope_l3811_381103

/-- The slope of a horizontal line y + 3 = 0 is 0 -/
theorem horizontal_line_slope (x y : ℝ) : y + 3 = 0 → (∀ x₁ x₂, x₁ ≠ x₂ → (y - y) / (x₁ - x₂) = 0) := by
  sorry

end NUMINAMATH_CALUDE_horizontal_line_slope_l3811_381103


namespace NUMINAMATH_CALUDE_first_degree_function_determination_l3811_381124

-- Define a first-degree function
def FirstDegreeFunction (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b

-- State the theorem
theorem first_degree_function_determination
  (f : ℝ → ℝ)
  (h1 : FirstDegreeFunction f)
  (h2 : 2 * f 2 - 3 * f 1 = 5)
  (h3 : 2 * f 0 - f (-1) = 1) :
  ∀ x, f x = 3 * x - 2 :=
sorry

end NUMINAMATH_CALUDE_first_degree_function_determination_l3811_381124


namespace NUMINAMATH_CALUDE_cone_height_from_circular_sector_l3811_381128

theorem cone_height_from_circular_sector (r : ℝ) (h : r = 10) :
  let circumference := 2 * Real.pi * r
  let sector_arc_length := circumference / 3
  let base_radius := sector_arc_length / (2 * Real.pi)
  let height := Real.sqrt (r^2 - base_radius^2)
  height = 20 * Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_cone_height_from_circular_sector_l3811_381128


namespace NUMINAMATH_CALUDE_product_mod_five_l3811_381115

theorem product_mod_five : (2023 * 2024 * 2025 * 2026) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_five_l3811_381115


namespace NUMINAMATH_CALUDE_rectangle_problem_l3811_381172

theorem rectangle_problem (A B C D E F G H I : ℕ) : 
  (A * B = D * E) →  -- Areas of ABCD and DEFG are equal
  (A * B = C * H) →  -- Areas of ABCD and CEIH are equal
  (B = 43) →         -- BC = 43
  (D > E) →          -- Assume DG > DE
  (D = 1892) →       -- DG = 1892
  True               -- Conclusion (to be proved)
  := by sorry

end NUMINAMATH_CALUDE_rectangle_problem_l3811_381172


namespace NUMINAMATH_CALUDE_maisy_earns_fifteen_more_l3811_381152

/-- Represents Maisy's job options and calculates the earnings difference --/
def maisys_job_earnings_difference : ℝ :=
  let current_job_hours : ℝ := 8
  let current_job_wage : ℝ := 10
  let new_job_hours : ℝ := 4
  let new_job_wage : ℝ := 15
  let new_job_bonus : ℝ := 35
  let current_job_earnings := current_job_hours * current_job_wage
  let new_job_earnings := new_job_hours * new_job_wage + new_job_bonus
  new_job_earnings - current_job_earnings

/-- Theorem stating that Maisy will earn $15 more per week at her new job --/
theorem maisy_earns_fifteen_more : maisys_job_earnings_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_maisy_earns_fifteen_more_l3811_381152


namespace NUMINAMATH_CALUDE_area_ratio_squares_l3811_381194

/-- Given squares A, B, and C with the following properties:
  - The perimeter of square A is 16 units
  - The perimeter of square B is 32 units
  - The side length of square C is 4 times the side length of square B
  Prove that the ratio of the area of square B to the area of square C is 1/16 -/
theorem area_ratio_squares (a b c : ℝ) 
  (ha : 4 * a = 16) 
  (hb : 4 * b = 32) 
  (hc : c = 4 * b) : 
  (b ^ 2) / (c ^ 2) = 1 / 16 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_squares_l3811_381194


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3811_381166

theorem sqrt_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) ≤ 2 * Real.sqrt 3 ∧
  (Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) = 2 * Real.sqrt 3 ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3811_381166


namespace NUMINAMATH_CALUDE_zhang_san_not_losing_probability_l3811_381182

theorem zhang_san_not_losing_probability
  (p_win : ℚ) (p_draw : ℚ)
  (h_win : p_win = 1 / 3)
  (h_draw : p_draw = 1 / 4) :
  p_win + p_draw = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_zhang_san_not_losing_probability_l3811_381182


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3811_381149

def is_geometric_sequence (a : Fin 4 → ℝ) : Prop :=
  ∃ q : ℝ, ∀ i : Fin 3, a (i + 1) = a i * q

theorem geometric_sequence_property
  (a : Fin 4 → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_sum : a 0 + a 1 + a 2 + a 3 = Real.log (a 0 + a 1 + a 2))
  (h_a1 : a 0 > 1) :
  a 0 > a 2 ∧ a 1 < a 3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3811_381149


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3811_381165

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, b = (t * a.1, t * a.2)

/-- The vectors a and b as defined in the problem -/
def a (k : ℝ) : ℝ × ℝ := (1, k)
def b (k : ℝ) : ℝ × ℝ := (k, 4)

/-- k=-2 is sufficient for collinearity -/
theorem sufficient_condition (k : ℝ) : 
  k = -2 → collinear (a k) (b k) :=
sorry

/-- k=-2 is not necessary for collinearity -/
theorem not_necessary_condition : 
  ∃ k : ℝ, k ≠ -2 ∧ collinear (a k) (b k) :=
sorry

/-- The main theorem stating that k=-2 is sufficient but not necessary -/
theorem sufficient_but_not_necessary : 
  (∀ k : ℝ, k = -2 → collinear (a k) (b k)) ∧
  (∃ k : ℝ, k ≠ -2 ∧ collinear (a k) (b k)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3811_381165


namespace NUMINAMATH_CALUDE_stock_price_theorem_l3811_381109

/-- The stock price after three years of changes -/
def stock_price_after_three_years (initial_price : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + 0.8)
  let price_after_second_year := price_after_first_year * (1 - 0.3)
  let price_after_third_year := price_after_second_year * (1 + 0.5)
  price_after_third_year

/-- Theorem stating that the stock price after three years is $226.8 -/
theorem stock_price_theorem :
  stock_price_after_three_years 120 = 226.8 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_theorem_l3811_381109


namespace NUMINAMATH_CALUDE_smartphone_cost_decrease_l3811_381151

theorem smartphone_cost_decrease (original_cost new_cost : ℝ) 
  (h1 : original_cost = 600)
  (h2 : new_cost = 450) :
  (original_cost - new_cost) / original_cost * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_smartphone_cost_decrease_l3811_381151


namespace NUMINAMATH_CALUDE_base_9_to_base_10_conversion_l3811_381105

-- Define the base-9 number
def base_9_number : ℕ := 5126

-- Define the conversion function from base 9 to base 10
def base_9_to_base_10 (n : ℕ) : ℕ :=
  (n % 10) +
  ((n / 10) % 10) * 9 +
  ((n / 100) % 10) * 9^2 +
  ((n / 1000) % 10) * 9^3

-- Theorem statement
theorem base_9_to_base_10_conversion :
  base_9_to_base_10 base_9_number = 3750 := by
  sorry

end NUMINAMATH_CALUDE_base_9_to_base_10_conversion_l3811_381105


namespace NUMINAMATH_CALUDE_max_a_for_three_solutions_l3811_381197

/-- The equation function that we're analyzing -/
def f (x a : ℝ) : ℝ := (|x - 2| + 2*a)^2 - 3*(|x - 2| + 2*a) + 4*a*(3 - 4*a)

/-- Predicate to check if the equation has three solutions for a given 'a' -/
def has_three_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0

/-- The theorem stating that 0.5 is the maximum value of 'a' for which the equation has three solutions -/
theorem max_a_for_three_solutions :
  ∀ a : ℝ, has_three_solutions a → a ≤ 0.5 ∧
  has_three_solutions 0.5 :=
sorry

end NUMINAMATH_CALUDE_max_a_for_three_solutions_l3811_381197


namespace NUMINAMATH_CALUDE_triangle_side_length_l3811_381106

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, angle B is 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : Real) : 
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →
  B = π/3 →
  a^2 + c^2 = 3*a*c →
  b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3811_381106


namespace NUMINAMATH_CALUDE_perpendicular_equivalence_l3811_381141

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_equivalence
  (α β : Plane) (m n : Line)
  (h_distinct : α ≠ β)
  (h_non_coincident : m ≠ n)
  (h_m_perp_α : perp m α)
  (h_m_perp_β : perp m β) :
  perp n α ↔ perp n β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_equivalence_l3811_381141


namespace NUMINAMATH_CALUDE_bart_tuesday_surveys_l3811_381139

/-- Represents the number of surveys Bart completed on Tuesday -/
def tuesday_surveys : ℕ := sorry

/-- The amount earned per question in dollars -/
def earnings_per_question : ℚ := 1/5

/-- The number of questions in each survey -/
def questions_per_survey : ℕ := 10

/-- The number of surveys completed on Monday -/
def monday_surveys : ℕ := 3

/-- The total amount earned over two days in dollars -/
def total_earnings : ℚ := 14

theorem bart_tuesday_surveys :
  tuesday_surveys = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_bart_tuesday_surveys_l3811_381139


namespace NUMINAMATH_CALUDE_right_triangle_external_angles_ratio_l3811_381183

theorem right_triangle_external_angles_ratio (α β : Real) : 
  α + β = 90 →  -- The triangle is right-angled
  (180 - α) / (90 + α) = 9 / 11 →  -- External angles ratio
  α = 58.5 ∧ β = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_external_angles_ratio_l3811_381183


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3811_381118

-- Define the set M
def M (k : ℝ) : Set ℝ := {x : ℝ | |x| > k}

-- Define the statement
theorem sufficient_not_necessary (k : ℝ) :
  (k = 2 → 2 ∈ (M k)ᶜ) ∧ (∃ k', k' ≠ 2 ∧ 2 ∈ (M k')ᶜ) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3811_381118


namespace NUMINAMATH_CALUDE_salary_before_raise_l3811_381112

theorem salary_before_raise (new_salary : ℝ) (increase_percentage : ℝ) (old_salary : ℝ) :
  new_salary = 70 →
  increase_percentage = 16.666666666666664 →
  old_salary * (1 + increase_percentage / 100) = new_salary →
  old_salary = 60 := by
sorry

end NUMINAMATH_CALUDE_salary_before_raise_l3811_381112


namespace NUMINAMATH_CALUDE_division_sum_theorem_l3811_381108

theorem division_sum_theorem (quotient divisor remainder : ℕ) : 
  quotient = 120 → divisor = 456 → remainder = 333 → 
  (divisor * quotient + remainder = 55053) := by sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l3811_381108


namespace NUMINAMATH_CALUDE_parabola_point_slope_l3811_381111

/-- A point on a parabola with given properties -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  distance_to_focus : Real.sqrt ((x - 1)^2 + y^2) = 5

/-- The theorem stating the absolute value of the slope -/
theorem parabola_point_slope (P : ParabolaPoint) : 
  |((P.y - 0) / (P.x - 1))| = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_slope_l3811_381111


namespace NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l3811_381123

theorem smallest_divisible_by_one_to_ten : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ (∀ m : ℕ, m > 0 → (∀ k : ℕ, k ≤ 10 → k > 0 → m % k = 0) → m ≥ 2520) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l3811_381123


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3811_381137

/-- Given that x and y are inversely proportional, prove that y = -16.875 when x = -10 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 30 ∧ x₀ = 3 * y₀ ∧ x₀ * y₀ = k) : 
  x = -10 → y = -16.875 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3811_381137


namespace NUMINAMATH_CALUDE_max_cables_theorem_l3811_381140

/-- Represents the maximum number of cables that can be used to connect computers
    in an organization with specific constraints. -/
def max_cables (total_employees : ℕ) (brand_a_computers : ℕ) (brand_b_computers : ℕ) : ℕ :=
  30

/-- Theorem stating that the maximum number of cables is 30 under given conditions. -/
theorem max_cables_theorem (total_employees : ℕ) (brand_a_computers : ℕ) (brand_b_computers : ℕ) :
  total_employees = 40 →
  brand_a_computers = 25 →
  brand_b_computers = 15 →
  total_employees = brand_a_computers + brand_b_computers →
  max_cables total_employees brand_a_computers brand_b_computers = 30 :=
by
  sorry

#check max_cables_theorem

end NUMINAMATH_CALUDE_max_cables_theorem_l3811_381140


namespace NUMINAMATH_CALUDE_function_value_proof_l3811_381145

theorem function_value_proof (f : ℝ → ℝ) :
  (3 : ℝ) + 17 = 60 * f 3 → f 3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_proof_l3811_381145


namespace NUMINAMATH_CALUDE_optimal_candy_combination_l3811_381174

/-- Represents a candy set with its cost and number of candies -/
structure CandySet where
  cost : ℕ
  candies : ℕ

/-- Finds the optimal combination of candy sets to maximize total candies within a budget -/
def findOptimalCombination (set1 set2 set3 : CandySet) (budget : ℕ) : 
  Option (ℕ × ℕ × ℕ) :=
  sorry

/-- Theorem stating that the given combination maximizes candies within the budget -/
theorem optimal_candy_combination :
  let set1 : CandySet := { cost := 50, candies := 25 }
  let set2 : CandySet := { cost := 180, candies := 95 }
  let set3 : CandySet := { cost := 150, candies := 80 }
  let budget : ℕ := 2200
  let optimal := findOptimalCombination set1 set2 set3 budget
  optimal = some (2, 5, 8) ∧
  (∀ x y z : ℕ, 
    x * set1.cost + y * set2.cost + z * set3.cost ≤ budget →
    x * set1.candies + y * set2.candies + z * set3.candies ≤ 
    2 * set1.candies + 5 * set2.candies + 8 * set3.candies) :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_candy_combination_l3811_381174


namespace NUMINAMATH_CALUDE_population_growth_problem_l3811_381189

theorem population_growth_problem (initial_population : ℝ) (final_population : ℝ) (second_year_decrease : ℝ) :
  initial_population = 10000 →
  final_population = 9600 →
  second_year_decrease = 20 →
  ∃ first_year_increase : ℝ,
    first_year_increase = 20 ∧
    final_population = initial_population * (1 + first_year_increase / 100) * (1 - second_year_decrease / 100) :=
by sorry

end NUMINAMATH_CALUDE_population_growth_problem_l3811_381189


namespace NUMINAMATH_CALUDE_limit_at_one_equals_five_l3811_381159

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 3*x

-- State the theorem
theorem limit_at_one_equals_five :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |(f (1 + Δx) - f 1) / Δx - 5| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_at_one_equals_five_l3811_381159


namespace NUMINAMATH_CALUDE_line_relationships_l3811_381110

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines when two lines are parallel -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Defines when two lines coincide -/
def coincide (l1 l2 : Line2D) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ l1.a = k * l2.a ∧ l1.b = k * l2.b ∧ l1.c = k * l2.c

/-- Defines when two lines are perpendicular -/
def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem about the relationships between two specific lines -/
theorem line_relationships (m : ℝ) :
  let l1 : Line2D := ⟨m + 3, 4, 3*m - 5⟩
  let l2 : Line2D := ⟨2, m + 5, -8⟩
  (parallel l1 l2 ↔ m = -7) ∧
  (coincide l1 l2 ↔ m = -1) ∧
  (perpendicular l1 l2 ↔ m = -13/3) := by
  sorry

end NUMINAMATH_CALUDE_line_relationships_l3811_381110


namespace NUMINAMATH_CALUDE_last_page_cards_l3811_381162

/-- Calculates the number of cards on the last page after reorganization --/
def cards_on_last_page (initial_albums : ℕ) (initial_pages_per_album : ℕ) 
  (initial_cards_per_page : ℕ) (new_cards_per_page : ℕ) (full_albums : ℕ) 
  (extra_full_pages : ℕ) : ℕ :=
  let total_cards := initial_albums * initial_pages_per_album * initial_cards_per_page
  let cards_in_full_albums := full_albums * initial_pages_per_album * new_cards_per_page
  let cards_in_extra_pages := extra_full_pages * new_cards_per_page
  let remaining_cards := total_cards - (cards_in_full_albums + cards_in_extra_pages)
  remaining_cards - (extra_full_pages * new_cards_per_page)

/-- Theorem stating that given the problem conditions, the last page contains 40 cards --/
theorem last_page_cards : 
  cards_on_last_page 10 50 8 12 5 40 = 40 := by
  sorry

end NUMINAMATH_CALUDE_last_page_cards_l3811_381162


namespace NUMINAMATH_CALUDE_cost_price_of_article_l3811_381154

/-- 
Proves that the cost price of an article is 44, given that the profit obtained 
by selling it for 66 is the same as the loss obtained by selling it for 22.
-/
theorem cost_price_of_article : ∃ (x : ℝ), 
  (66 - x = x - 22) → x = 44 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_article_l3811_381154


namespace NUMINAMATH_CALUDE_car_waiting_time_l3811_381150

/-- Proves that a car waiting for a cyclist to catch up after 18 minutes must have initially waited 4.5 minutes -/
theorem car_waiting_time 
  (cyclist_speed : ℝ) 
  (car_speed : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : cyclist_speed = 15) 
  (h2 : car_speed = 60) 
  (h3 : catch_up_time = 18 / 60) : 
  let relative_speed := car_speed - cyclist_speed
  let distance := cyclist_speed * catch_up_time
  let initial_wait_time := distance / car_speed
  initial_wait_time * 60 = 4.5 := by sorry

end NUMINAMATH_CALUDE_car_waiting_time_l3811_381150


namespace NUMINAMATH_CALUDE_extreme_values_imply_a_range_extreme_values_imply_a_in_range_l3811_381116

/-- A function f with two extreme values in R -/
structure TwoExtremeFunction (f : ℝ → ℝ) : Prop where
  has_two_extremes : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (∀ (x : ℝ), f x ≤ f x₁) ∧ 
    (∀ (x : ℝ), f x ≤ f x₂)

/-- The main theorem -/
theorem extreme_values_imply_a_range 
  (a : ℝ) 
  (ha : a ≠ 0) 
  (f : ℝ → ℝ)
  (hf : f = λ x => (1 + a * x^2) * Real.exp x)
  (h_two_extremes : TwoExtremeFunction f) :
  a < 0 ∨ a > 1 := by
  sorry

/-- The range of a as a set -/
def a_range : Set ℝ := {a | a < 0 ∨ a > 1}

/-- An equivalent formulation of the main theorem using sets -/
theorem extreme_values_imply_a_in_range 
  (a : ℝ) 
  (ha : a ≠ 0) 
  (f : ℝ → ℝ)
  (hf : f = λ x => (1 + a * x^2) * Real.exp x)
  (h_two_extremes : TwoExtremeFunction f) :
  a ∈ a_range := by
  sorry

end NUMINAMATH_CALUDE_extreme_values_imply_a_range_extreme_values_imply_a_in_range_l3811_381116


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3811_381143

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x > 0}

theorem union_of_M_and_N : M ∪ N = {x | x = 0 ∨ x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3811_381143


namespace NUMINAMATH_CALUDE_mary_screw_sections_l3811_381135

def number_of_sections (initial_screws : ℕ) (multiplier : ℕ) (screws_per_section : ℕ) : ℕ :=
  (initial_screws + initial_screws * multiplier) / screws_per_section

theorem mary_screw_sections :
  number_of_sections 8 2 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mary_screw_sections_l3811_381135


namespace NUMINAMATH_CALUDE_fixed_point_of_function_l3811_381133

theorem fixed_point_of_function (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 4 + a^(x - 1)
  f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_function_l3811_381133


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_two_l3811_381176

theorem fraction_zero_implies_x_two (x : ℝ) : 
  (x^2 - 4) / (x + 2) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_two_l3811_381176


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_l3811_381184

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℤ := 2*n - 12

-- Define the sum of the geometric sequence b_n
def S (n : ℕ) : ℤ := 4*(1 - 3^n)

theorem arithmetic_and_geometric_sequences :
  -- Conditions for a_n
  (a 3 = -6) ∧ (a 6 = 0) ∧
  -- Arithmetic sequence property
  (∀ n : ℕ, a (n+1) - a n = a (n+2) - a (n+1)) ∧
  -- Conditions for b_n
  (∃ b : ℕ → ℤ, b 1 = -8 ∧ b 2 = a 1 + a 2 + a 3 ∧
  -- Geometric sequence property
  (∀ n : ℕ, n ≥ 1 → b (n+1) / b n = b 2 / b 1) ∧
  -- S_n is the sum of the first n terms of b_n
  (∀ n : ℕ, n ≥ 1 → S n = (b 1) * (1 - (b 2 / b 1)^n) / (1 - b 2 / b 1))) := by
  sorry

#check arithmetic_and_geometric_sequences

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_l3811_381184


namespace NUMINAMATH_CALUDE_least_marbles_count_l3811_381177

theorem least_marbles_count (n : ℕ) : n ≥ 402 →
  (n % 7 = 3 ∧ n % 4 = 2 ∧ n % 6 = 1) →
  n = 402 :=
by sorry

end NUMINAMATH_CALUDE_least_marbles_count_l3811_381177


namespace NUMINAMATH_CALUDE_min_cost_theorem_l3811_381175

/-- Represents the ticket prices and family information for a subway system -/
structure TicketSystem :=
  (adult_single : ℕ)
  (child_single : ℕ)
  (day_pass_single : ℕ)
  (day_pass_group : ℕ)
  (three_day_pass_single : ℕ)
  (three_day_pass_group : ℕ)
  (num_adults : ℕ)
  (num_children : ℕ)
  (num_days : ℕ)
  (trips_per_day : ℕ)

/-- Calculates the minimum cost for the given ticket system -/
def min_cost (ts : TicketSystem) : ℕ :=
  let adult_cost := 2 * ts.three_day_pass_single + 2 * ts.day_pass_single
  let child_cost := ts.num_children * ts.num_days * ts.trips_per_day * ts.child_single
  adult_cost + child_cost

/-- Theorem stating the minimum cost for the given scenario -/
theorem min_cost_theorem (ts : TicketSystem) :
  ts.adult_single = 40 →
  ts.child_single = 20 →
  ts.day_pass_single = 350 →
  ts.day_pass_group = 1500 →
  ts.three_day_pass_single = 900 →
  ts.three_day_pass_group = 3500 →
  ts.num_adults = 2 →
  ts.num_children = 2 →
  ts.num_days = 5 →
  ts.trips_per_day = 10 →
  min_cost ts = 5200 :=
sorry

end NUMINAMATH_CALUDE_min_cost_theorem_l3811_381175


namespace NUMINAMATH_CALUDE_calories_burned_jogging_l3811_381130

/-- Calculate calories burned by jogging -/
theorem calories_burned_jogging (laps_per_night : ℕ) (feet_per_lap : ℕ) (feet_per_calorie : ℕ) (days : ℕ) : 
  laps_per_night = 5 →
  feet_per_lap = 100 →
  feet_per_calorie = 25 →
  days = 5 →
  (laps_per_night * feet_per_lap * days) / feet_per_calorie = 100 := by
  sorry

end NUMINAMATH_CALUDE_calories_burned_jogging_l3811_381130


namespace NUMINAMATH_CALUDE_mean_median_difference_l3811_381186

theorem mean_median_difference (x : ℕ) (h : x > 0) : 
  let sequence := [x, x + 2, x + 4, x + 7, x + 37]
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 37)) / 5
  let median := x + 4
  mean - median = 6 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l3811_381186


namespace NUMINAMATH_CALUDE_coin_distribution_proof_l3811_381117

/-- Represents the coin distribution scheme between Charlie and Fred -/
def coin_distribution (x : ℕ) : Prop :=
  -- Charlie's coins are the sum of 1 to x
  let charlie_coins := x * (x + 1) / 2
  -- Fred's coins are x at the end
  let fred_coins := x
  -- Charlie has 5 times as many coins as Fred
  charlie_coins = 5 * fred_coins

/-- The total number of coins after distribution -/
def total_coins (x : ℕ) : ℕ := x * 6

theorem coin_distribution_proof :
  ∃ x : ℕ, coin_distribution x ∧ total_coins x = 54 :=
sorry

end NUMINAMATH_CALUDE_coin_distribution_proof_l3811_381117


namespace NUMINAMATH_CALUDE_pete_backward_speed_l3811_381148

/-- Represents the speeds of various activities in miles per hour -/
structure Speeds where
  susan_forward : ℝ
  pete_backward : ℝ
  tracy_cartwheel : ℝ
  pete_hands : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : Speeds) : Prop :=
  s.pete_backward = 3 * s.susan_forward ∧
  s.tracy_cartwheel = 2 * s.susan_forward ∧
  s.pete_hands = 1/4 * s.tracy_cartwheel ∧
  s.pete_hands = 2

/-- The theorem stating Pete's backward walking speed -/
theorem pete_backward_speed (s : Speeds) 
  (h : problem_conditions s) : s.pete_backward = 12 := by
  sorry


end NUMINAMATH_CALUDE_pete_backward_speed_l3811_381148


namespace NUMINAMATH_CALUDE_rain_stop_time_l3811_381146

def rain_duration (start_time : ℕ) (day1_duration : ℕ) : ℕ → ℕ
  | 1 => day1_duration
  | 2 => day1_duration + 2
  | 3 => 2 * (day1_duration + 2)
  | _ => 0

theorem rain_stop_time (start_time : ℕ) (day1_duration : ℕ) :
  start_time = 7 ∧ 
  (rain_duration start_time day1_duration 1 + 
   rain_duration start_time day1_duration 2 + 
   rain_duration start_time day1_duration 3 = 46) →
  start_time + day1_duration = 17 := by
  sorry

end NUMINAMATH_CALUDE_rain_stop_time_l3811_381146


namespace NUMINAMATH_CALUDE_max_tickets_for_hockey_l3811_381121

def max_tickets (ticket_price : ℕ) (budget : ℕ) : ℕ :=
  (budget / ticket_price : ℕ)

theorem max_tickets_for_hockey (ticket_price : ℕ) (budget : ℕ) 
  (h1 : ticket_price = 20) (h2 : budget = 150) : 
  max_tickets ticket_price budget = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_tickets_for_hockey_l3811_381121


namespace NUMINAMATH_CALUDE_cost_calculation_l3811_381187

/-- Given the cost relationships between mangos, rice, and flour, 
    prove the total cost of a specific quantity of each. -/
theorem cost_calculation 
  (mango_rice_relation : ∀ (mango_cost rice_cost : ℝ), 10 * mango_cost = 24 * rice_cost)
  (flour_rice_relation : ∀ (flour_cost rice_cost : ℝ), 6 * flour_cost = 2 * rice_cost)
  (flour_cost : ℝ) (h_flour_cost : flour_cost = 23)
  : ∃ (mango_cost rice_cost : ℝ),
    4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 984.4 :=
by sorry

end NUMINAMATH_CALUDE_cost_calculation_l3811_381187


namespace NUMINAMATH_CALUDE_chocolate_price_proof_l3811_381129

/-- Proves that if a chocolate's price is reduced by 57 cents and the resulting price is $1.43, then the original price was $2.00. -/
theorem chocolate_price_proof (original_price : ℝ) : 
  (original_price - 0.57 = 1.43) → original_price = 2.00 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_price_proof_l3811_381129


namespace NUMINAMATH_CALUDE_sheila_weekly_earnings_l3811_381122

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  hourly_rate : ℕ

/-- Calculates the total weekly hours worked --/
def total_weekly_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu

/-- Calculates the weekly earnings --/
def weekly_earnings (schedule : WorkSchedule) : ℕ :=
  (total_weekly_hours schedule) * schedule.hourly_rate

/-- Theorem stating Sheila's weekly earnings --/
theorem sheila_weekly_earnings :
  ∀ (schedule : WorkSchedule),
  schedule.hours_mon_wed_fri = 8 →
  schedule.hours_tue_thu = 6 →
  schedule.hourly_rate = 10 →
  weekly_earnings schedule = 360 := by
  sorry


end NUMINAMATH_CALUDE_sheila_weekly_earnings_l3811_381122


namespace NUMINAMATH_CALUDE_line_and_circle_equations_l3811_381195

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Theorem statement
theorem line_and_circle_equations :
  ∀ (x y : ℝ),
  (∃ (t : ℝ), x = 2 + 4*t ∧ y = 1 + 2*t) →  -- Line l passes through (2, 1) and (6, 3)
  (∃ (a : ℝ), line_l (2*a) a ∧ circle_C (2*a) a) →  -- Circle C's center lies on line l
  circle_C 2 0 →  -- Circle C is tangent to x-axis at (2, 0)
  (line_l x y ↔ x - 2*y = 0) ∧  -- Equation of line l
  (circle_C x y ↔ (x - 2)^2 + (y - 1)^2 = 1)  -- Equation of circle C
  := by sorry

end NUMINAMATH_CALUDE_line_and_circle_equations_l3811_381195


namespace NUMINAMATH_CALUDE_line_through_points_l3811_381113

-- Define the line equation
def line_equation (a b x : ℝ) : ℝ := a * x + b

-- Define the condition that the line passes through two points
def passes_through (a b : ℝ) : Prop :=
  line_equation a b 3 = 4 ∧ line_equation a b 9 = 22

-- Theorem statement
theorem line_through_points :
  ∀ a b : ℝ, passes_through a b → a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l3811_381113


namespace NUMINAMATH_CALUDE_fourth_term_is_two_l3811_381136

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  a_1_eq : a 1 = 16
  a_6_eq : a 6 = 2 * a 5 * a 7

/-- The fourth term of the geometric sequence is 2 -/
theorem fourth_term_is_two (seq : GeometricSequence) : seq.a 4 = 2 := by
  sorry


end NUMINAMATH_CALUDE_fourth_term_is_two_l3811_381136


namespace NUMINAMATH_CALUDE_problem_solution_l3811_381185

theorem problem_solution (x y : ℤ) (h1 : x + y = 250) (h2 : x - y = 200) : y = 25 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3811_381185


namespace NUMINAMATH_CALUDE_quadratic_real_root_and_inequality_l3811_381193

theorem quadratic_real_root_and_inequality (a b c : ℝ) :
  (∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0) ∧
  (a + b + c)^2 ≥ 3 * (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_and_inequality_l3811_381193


namespace NUMINAMATH_CALUDE_work_completion_time_l3811_381120

theorem work_completion_time 
  (days_a : ℝ) 
  (days_b : ℝ) 
  (h1 : days_a = 12) 
  (h2 : days_b = 24) : 
  1 / (1 / days_a + 1 / days_b) = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3811_381120


namespace NUMINAMATH_CALUDE_ap_terms_count_l3811_381125

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  n : ℕ
  a : ℝ
  d : ℝ
  even_n : Even n
  odd_sum : (n / 2) * (a + (a + (n - 2) * d)) = 30
  even_sum : (n / 2) * ((a + d) + (a + (n - 1) * d)) = 45
  last_first_diff : (a + (n - 1) * d) - a = 7.5

/-- The theorem stating that the number of terms in the arithmetic progression is 12 -/
theorem ap_terms_count (ap : ArithmeticProgression) : ap.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ap_terms_count_l3811_381125


namespace NUMINAMATH_CALUDE_min_selections_for_multiple_of_five_l3811_381158

theorem min_selections_for_multiple_of_five (n : ℕ) (h : n = 30) : 
  (∀ S : Finset ℕ, S ⊆ Finset.range n → S.card ≥ 25 → ∃ x ∈ S, x % 5 = 0) ∧
  (∃ S : Finset ℕ, S ⊆ Finset.range n ∧ S.card = 24 ∧ ∀ x ∈ S, x % 5 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_min_selections_for_multiple_of_five_l3811_381158


namespace NUMINAMATH_CALUDE_apple_tree_problem_l3811_381157

/-- The number of apples initially on the tree -/
def initial_apples : ℕ := 11

/-- The number of apples picked from the tree -/
def apples_picked : ℕ := 7

/-- The number of new apples that grew on the tree -/
def new_apples : ℕ := 2

/-- The number of apples currently on the tree -/
def current_apples : ℕ := 6

theorem apple_tree_problem :
  initial_apples - apples_picked + new_apples = current_apples :=
by sorry

end NUMINAMATH_CALUDE_apple_tree_problem_l3811_381157


namespace NUMINAMATH_CALUDE_cubic_equation_properties_l3811_381104

theorem cubic_equation_properties :
  (∀ x y : ℕ, x^3 + 5*y = y^3 + 5*x → x = y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + 5*y = y^3 + 5*x) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_properties_l3811_381104


namespace NUMINAMATH_CALUDE_equilibrium_force_l3811_381161

/-- A 2D vector representation --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Addition of two 2D vectors --/
def Vector2D.add (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Negation of a 2D vector --/
def Vector2D.neg (v : Vector2D) : Vector2D :=
  ⟨-v.x, -v.y⟩

/-- Zero 2D vector --/
def Vector2D.zero : Vector2D :=
  ⟨0, 0⟩

theorem equilibrium_force (f₁ f₂ f₃ f₄ : Vector2D) 
    (h₁ : f₁ = ⟨-2, -1⟩) 
    (h₂ : f₂ = ⟨-3, 2⟩)
    (h₃ : f₃ = ⟨4, -3⟩)
    (h₄ : f₄ = ⟨1, 2⟩) :
    Vector2D.add (Vector2D.add (Vector2D.add f₁ f₂) f₃) f₄ = Vector2D.zero := by
  sorry

#check equilibrium_force

end NUMINAMATH_CALUDE_equilibrium_force_l3811_381161


namespace NUMINAMATH_CALUDE_euler_totient_multiple_l3811_381155

theorem euler_totient_multiple (m n : ℕ+) : ∃ a : ℕ+, ∀ i : ℕ, i ≤ n → (m : ℕ) ∣ Nat.totient (a + i) := by
  sorry

end NUMINAMATH_CALUDE_euler_totient_multiple_l3811_381155


namespace NUMINAMATH_CALUDE_solve_for_m_l3811_381101

theorem solve_for_m : ∃ m : ℝ, 
  (∀ x : ℝ, x > 2 ↔ x - 3*m + 1 > 0) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3811_381101


namespace NUMINAMATH_CALUDE_product_abcd_l3811_381107

theorem product_abcd : 
  ∀ (a b c d : ℚ),
  (3 * a + 2 * b + 4 * c + 6 * d = 36) →
  (4 * (d + c) = b) →
  (4 * b + 2 * c = a) →
  (c - 2 = d) →
  (a * b * c * d = -315/32) :=
by
  sorry

end NUMINAMATH_CALUDE_product_abcd_l3811_381107


namespace NUMINAMATH_CALUDE_jemma_grasshopper_count_l3811_381164

/-- The number of grasshoppers Jemma saw on her African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The number of dozens of baby grasshoppers Jemma found on the grass -/
def dozens_of_baby_grasshoppers : ℕ := 2

/-- The number of grasshoppers in a dozen -/
def grasshoppers_per_dozen : ℕ := 12

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := grasshoppers_on_plant + dozens_of_baby_grasshoppers * grasshoppers_per_dozen

theorem jemma_grasshopper_count : total_grasshoppers = 31 := by
  sorry

end NUMINAMATH_CALUDE_jemma_grasshopper_count_l3811_381164


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l3811_381169

/-- A dishonest dealer's profit percentage when using underweight measurements --/
theorem dishonest_dealer_profit_percentage 
  (actual_weight : ℝ) 
  (claimed_weight : ℝ) 
  (actual_weight_positive : 0 < actual_weight)
  (claimed_weight_greater : actual_weight < claimed_weight) : 
  (claimed_weight - actual_weight) / actual_weight * 100 = 
  (1000 - 920) / 920 * 100 := by
sorry

#eval (1000 - 920) / 920 * 100  -- To show the approximate result

end NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l3811_381169


namespace NUMINAMATH_CALUDE_hexagon_semicircles_area_l3811_381147

/-- The area of the region inside a regular hexagon with side length 4, 
    but outside eight semicircles (where each semicircle's diameter 
    coincides with each side of the hexagon) -/
theorem hexagon_semicircles_area : 
  let s : ℝ := 4 -- side length of the hexagon
  let r : ℝ := s / 2 -- radius of each semicircle
  let hexagon_area : ℝ := 3 * Real.sqrt 3 / 2 * s^2
  let semicircle_area : ℝ := 8 * (Real.pi * r^2 / 2)
  hexagon_area - semicircle_area = 24 * Real.sqrt 3 - 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_hexagon_semicircles_area_l3811_381147


namespace NUMINAMATH_CALUDE_essay_section_ratio_l3811_381163

theorem essay_section_ratio (total_words introduction_words body_section_words : ℕ)
  (h1 : total_words = 5000)
  (h2 : introduction_words = 450)
  (h3 : body_section_words = 800)
  (h4 : ∃ (k : ℕ), total_words = introduction_words + 4 * body_section_words + k * introduction_words) :
  ∃ (conclusion_words : ℕ), conclusion_words = 3 * introduction_words :=
by sorry

end NUMINAMATH_CALUDE_essay_section_ratio_l3811_381163


namespace NUMINAMATH_CALUDE_emma_yield_calculation_l3811_381153

/-- The annual yield percentage of Emma's investment -/
def emma_yield : ℝ := 18.33

/-- Emma's investment amount -/
def emma_investment : ℝ := 300

/-- Briana's investment amount -/
def briana_investment : ℝ := 500

/-- Briana's annual yield percentage -/
def briana_yield : ℝ := 10

/-- The difference in return-on-investment after 2 years -/
def roi_difference : ℝ := 10

/-- The number of years for the investment -/
def years : ℝ := 2

theorem emma_yield_calculation :
  emma_investment * (emma_yield / 100) * years - 
  briana_investment * (briana_yield / 100) * years = roi_difference :=
sorry

end NUMINAMATH_CALUDE_emma_yield_calculation_l3811_381153


namespace NUMINAMATH_CALUDE_kath_group_cost_l3811_381160

/-- Calculates the total cost of movie admission for a group, given a regular price, 
    discount amount, and number of people in the group. -/
def total_cost (regular_price discount : ℕ) (group_size : ℕ) : ℕ :=
  (regular_price - discount) * group_size

/-- Proves that the total cost for Kath's group is $30 -/
theorem kath_group_cost : 
  let regular_price : ℕ := 8
  let discount : ℕ := 3
  let kath_siblings : ℕ := 2
  let kath_friends : ℕ := 3
  let group_size : ℕ := 1 + kath_siblings + kath_friends
  total_cost regular_price discount group_size = 30 := by
  sorry

#eval total_cost 8 3 6

end NUMINAMATH_CALUDE_kath_group_cost_l3811_381160


namespace NUMINAMATH_CALUDE_set_operations_and_conditions_l3811_381198

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 8}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Define the theorem
theorem set_operations_and_conditions :
  -- Part 1
  (A 0 ∩ B = {x | 5 < x ∧ x ≤ 8}) ∧
  (A 0 ∪ Bᶜ = {x | -1 ≤ x ∧ x ≤ 8}) ∧
  -- Part 2
  (∀ a : ℝ, A a ∪ B = B ↔ a < -9 ∨ a > 5) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_conditions_l3811_381198


namespace NUMINAMATH_CALUDE_value_of_a_l3811_381156

/-- A sequence where each term is the sum of the two terms to its left -/
def Sequence : Type := ℤ → ℤ

/-- Property that each term is the sum of the two terms to its left -/
def is_sum_of_previous_two (s : Sequence) : Prop :=
  ∀ n : ℤ, s (n + 2) = s (n + 1) + s n

/-- The specific sequence we're interested in -/
def our_sequence : Sequence := sorry

/-- The properties of our specific sequence -/
axiom our_sequence_property : is_sum_of_previous_two our_sequence
axiom our_sequence_known_values :
  ∃ k : ℤ,
    our_sequence (k + 3) = 0 ∧
    our_sequence (k + 4) = 1 ∧
    our_sequence (k + 5) = 1 ∧
    our_sequence (k + 6) = 2 ∧
    our_sequence (k + 7) = 3 ∧
    our_sequence (k + 8) = 5 ∧
    our_sequence (k + 9) = 8

/-- The theorem to prove -/
theorem value_of_a :
  ∃ k : ℤ, our_sequence k = -3 ∧
    our_sequence (k + 3) = 0 ∧
    our_sequence (k + 4) = 1 :=
sorry

end NUMINAMATH_CALUDE_value_of_a_l3811_381156


namespace NUMINAMATH_CALUDE_doll_difference_l3811_381192

/-- The number of dolls Lindsay has with blonde hair -/
def blonde_dolls : ℕ := 4

/-- The number of dolls Lindsay has with brown hair -/
def brown_dolls : ℕ := 4 * blonde_dolls

/-- The number of dolls Lindsay has with black hair -/
def black_dolls : ℕ := brown_dolls - 2

/-- The theorem stating the difference between the combined number of black and brown-haired dolls
    and the number of blonde-haired dolls -/
theorem doll_difference : brown_dolls + black_dolls - blonde_dolls = 26 := by
  sorry

end NUMINAMATH_CALUDE_doll_difference_l3811_381192


namespace NUMINAMATH_CALUDE_product_remainder_by_10_l3811_381144

theorem product_remainder_by_10 : (4219 * 2675 * 394082 * 5001) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_10_l3811_381144


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l3811_381178

theorem closest_integer_to_cube_root_250 : 
  ∀ n : ℤ, |n - (250 : ℝ)^(1/3)| ≥ |(6 : ℝ) - (250 : ℝ)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l3811_381178


namespace NUMINAMATH_CALUDE_program_production_cost_l3811_381127

/-- The cost to produce a program for a college football game. -/
def cost_to_produce : ℝ :=
  sorry

/-- Theorem: Given the conditions, the cost to produce a program is 5500 rupees. -/
theorem program_production_cost :
  let advertisement_revenue : ℝ := 15000
  let copies_sold : ℝ := 35000
  let price_per_copy : ℝ := 0.50
  let desired_profit : ℝ := 8000
  cost_to_produce = advertisement_revenue + (copies_sold * price_per_copy) - (advertisement_revenue + desired_profit) :=
by
  sorry

end NUMINAMATH_CALUDE_program_production_cost_l3811_381127


namespace NUMINAMATH_CALUDE_algebraic_equality_l3811_381138

theorem algebraic_equality (a b c : ℝ) : a + b * c = (a + b) * (a + c) ↔ a + b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_equality_l3811_381138


namespace NUMINAMATH_CALUDE_circle_sum_of_center_and_radius_l3811_381170

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 16*y + 81 = -y^2 + 14*x

-- Define the center and radius of the circle
def circle_center_radius (a b r : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_sum_of_center_and_radius :
  ∃ a b r, circle_center_radius a b r ∧ a + b + r = 15 + 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_circle_sum_of_center_and_radius_l3811_381170


namespace NUMINAMATH_CALUDE_break_time_calculation_l3811_381188

-- Define the speeds of A and B
def speed_A : ℝ := 60
def speed_B : ℝ := 40

-- Define the distances from midpoint for the two meeting points
def distance_first_meeting : ℝ := 300
def distance_second_meeting : ℝ := 150

-- Define the total distance between A and B
def total_distance : ℝ := 2 * distance_first_meeting

-- Define the theorem
theorem break_time_calculation :
  ∃ (t : ℝ), (t = 6.25 ∨ t = 18.75) ∧
  ((speed_A * (total_distance / (speed_A + speed_B) - t) = distance_first_meeting + distance_second_meeting) ∨
   (speed_A * (total_distance / (speed_A + speed_B) - t) = total_distance - (distance_first_meeting + distance_second_meeting))) :=
by
  sorry


end NUMINAMATH_CALUDE_break_time_calculation_l3811_381188


namespace NUMINAMATH_CALUDE_locus_is_circle_l3811_381196

/-- The locus of points satisfying the given equation is a circle -/
theorem locus_is_circle (x y : ℝ) : 
  (10 * Real.sqrt ((x - 1)^2 + (y - 2)^2) = |3*x - 4*y|) → 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_locus_is_circle_l3811_381196


namespace NUMINAMATH_CALUDE_author_writing_speed_l3811_381134

/-- Calculates the words written per hour, given the total words, total hours, and break hours -/
def wordsPerHour (totalWords : ℕ) (totalHours : ℕ) (breakHours : ℕ) : ℕ :=
  totalWords / (totalHours - breakHours)

/-- Theorem stating that under the given conditions, the author wrote at least 705 words per hour -/
theorem author_writing_speed :
  let totalWords : ℕ := 60000
  let totalHours : ℕ := 100
  let breakHours : ℕ := 15
  wordsPerHour totalWords totalHours breakHours ≥ 705 := by
  sorry

#eval wordsPerHour 60000 100 15

end NUMINAMATH_CALUDE_author_writing_speed_l3811_381134


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3811_381180

theorem largest_prime_factor_of_expression : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (12^3 + 15^4 - 6^5) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (12^3 + 15^4 - 6^5) → q ≤ p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3811_381180
