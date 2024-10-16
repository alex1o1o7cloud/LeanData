import Mathlib

namespace NUMINAMATH_CALUDE_boys_in_class_l2191_219179

theorem boys_in_class (total : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) : 
  total = 49 → ratio_boys = 4 → ratio_girls = 3 → 
  (ratio_boys * total) / (ratio_boys + ratio_girls) = 28 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l2191_219179


namespace NUMINAMATH_CALUDE_equation_solution_complex_l2191_219184

theorem equation_solution_complex (a b : ℂ) : 
  a ≠ 0 → 
  a + b ≠ 0 → 
  (a + b) / a = 3 * b / (a + b) → 
  (¬(a.im = 0) ∧ b.im = 0) ∨ (a.im = 0 ∧ ¬(b.im = 0)) ∨ (¬(a.im = 0) ∧ ¬(b.im = 0)) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_complex_l2191_219184


namespace NUMINAMATH_CALUDE_sequence_property_l2191_219197

theorem sequence_property (a : ℕ → ℝ) 
  (h_pos : ∀ n : ℕ, n ≥ 1 → a n > 0)
  (h_ineq : ∀ n : ℕ, n ≥ 1 → (a (n + 1))^2 + a n * a (n + 2) ≤ a n + a (n + 2)) :
  a 21022 ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l2191_219197


namespace NUMINAMATH_CALUDE_range_of_a_when_proposition_false_l2191_219127

theorem range_of_a_when_proposition_false :
  (¬ ∃ x₀ : ℝ, ∃ a : ℝ, a * x₀^2 - 2 * a * x₀ - 3 > 0) →
  (∀ a : ℝ, a ∈ Set.Icc (-3 : ℝ) 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_when_proposition_false_l2191_219127


namespace NUMINAMATH_CALUDE_strip_overlap_area_l2191_219173

/-- The area of overlap for three strips of width 2 intersecting at angle θ -/
theorem strip_overlap_area (θ : Real) (h1 : θ ≠ 0) (h2 : θ ≠ π / 2) : Real :=
  let strip_width : Real := 2
  let overlap_area := 8 * Real.sin θ
  overlap_area

#check strip_overlap_area

end NUMINAMATH_CALUDE_strip_overlap_area_l2191_219173


namespace NUMINAMATH_CALUDE_satisfying_function_is_constant_l2191_219181

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℕ → ℤ) : Prop :=
  (∀ a b : ℕ, a > 0 ∧ b > 0 → a ∣ b → f a ≥ f b) ∧
  (∀ a b : ℕ, a > 0 ∧ b > 0 → f (a * b) + f (a^2 + b^2) = f a + f b)

/-- The main theorem stating that any satisfying function is constant -/
theorem satisfying_function_is_constant (f : ℕ → ℤ) (hf : SatisfyingFunction f) :
  ∃ C : ℤ, ∀ n : ℕ, f n = C :=
sorry

end NUMINAMATH_CALUDE_satisfying_function_is_constant_l2191_219181


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l2191_219105

theorem intersection_point_of_lines (x y : ℚ) :
  (8 * x - 5 * y = 40) ∧ (10 * x + 2 * y = 14) ↔ x = 25/11 ∧ y = 48/11 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l2191_219105


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2191_219109

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2191_219109


namespace NUMINAMATH_CALUDE_equation_solution_l2191_219129

theorem equation_solution : 
  ∃! x : ℚ, (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 2)) ∧ x = -13/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2191_219129


namespace NUMINAMATH_CALUDE_percent_commutation_l2191_219146

theorem percent_commutation (x : ℝ) (h : 0.3 * (0.4 * x) = 36) :
  0.4 * (0.3 * x) = 0.3 * (0.4 * x) := by
  sorry

end NUMINAMATH_CALUDE_percent_commutation_l2191_219146


namespace NUMINAMATH_CALUDE_quadratic_root_k_l2191_219111

theorem quadratic_root_k (k : ℝ) : (1 : ℝ)^2 + k * 1 - 3 = 0 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_l2191_219111


namespace NUMINAMATH_CALUDE_train_length_l2191_219145

/-- The length of a train that crosses a platform of equal length in one minute at 126 km/hr -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (train_length platform_length : ℝ) :
  train_speed = 126 * 1000 / 3600 →
  crossing_time = 60 →
  train_length = platform_length →
  train_length * 2 = train_speed * crossing_time →
  train_length = 1050 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2191_219145


namespace NUMINAMATH_CALUDE_base7_subtraction_l2191_219190

/-- Represents a number in base 7 --/
def Base7 : Type := List Nat

/-- Converts a base 7 number to a natural number --/
def to_nat (b : Base7) : Nat :=
  b.foldr (fun digit acc => acc * 7 + digit) 0

/-- Subtracts two base 7 numbers --/
def subtract_base7 (a b : Base7) : Base7 :=
  sorry

theorem base7_subtraction :
  let a : Base7 := [1, 2, 1, 0, 0]
  let b : Base7 := [3, 6, 6, 6]
  subtract_base7 a b = [1, 1, 1, 1] := by sorry

end NUMINAMATH_CALUDE_base7_subtraction_l2191_219190


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_f_positive_range_l2191_219156

noncomputable section

variables (a : ℝ)

-- Define the function f
def f (x : ℝ) : ℝ := (a * x + 1) * Real.exp x - (a + 1) * x - 1

-- Theorem 1: The tangent line at (0, f(0)) is y = 0
theorem tangent_line_at_zero (a : ℝ) : 
  ∃ (m b : ℝ), ∀ x, m * x + b = 0 ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs (f a (0 + h) - f a 0 - m * h) ≤ ε * abs h) :=
sorry

-- Theorem 2: For f(x) > 0 to always hold when x > 0, a must be in [0,+∞)
theorem f_positive_range (a : ℝ) :
  (∀ x > 0, f a x > 0) ↔ a ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_f_positive_range_l2191_219156


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2191_219108

theorem imaginary_part_of_complex_number (z : ℂ) : z = (3 - 2 * Complex.I^2) / (1 + Complex.I) → z.im = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2191_219108


namespace NUMINAMATH_CALUDE_bake_sale_fundraiser_l2191_219139

/-- 
Given a bake sale that earned $400 total, prove that the amount kept for ingredients
is $100, when half of the remaining amount plus $10 equals $160.
-/
theorem bake_sale_fundraiser (total_earnings : ℝ) (donation_to_shelter : ℝ) :
  total_earnings = 400 ∧ 
  donation_to_shelter = 160 ∧
  donation_to_shelter = (total_earnings - (total_earnings - donation_to_shelter + 10)) / 2 + 10 →
  total_earnings - donation_to_shelter + 10 = 100 := by
sorry

end NUMINAMATH_CALUDE_bake_sale_fundraiser_l2191_219139


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l2191_219103

theorem geometric_series_first_term
  (r : ℝ)
  (hr : |r| < 1)
  (h_sum : (∑' n, r^n) * a = 15)
  (h_sum_squares : (∑' n, (r^n)^2) * a^2 = 45) :
  a = 5 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l2191_219103


namespace NUMINAMATH_CALUDE_f_properties_l2191_219199

noncomputable section

def f (x : ℝ) : ℝ := x^2 * Real.log x - x + 1

theorem f_properties :
  (∀ x > 0, f x = x^2 * Real.log x - x + 1) →
  f (Real.exp 1) = Real.exp 2 - Real.exp 1 + 1 ∧
  (deriv f) 1 = 0 ∧
  (∀ x ≥ 1, f x ≥ (x - 1)^2) ∧
  (∀ m > 3/2, ∃ x ≥ 1, f x < m * (x - 1)^2) ∧
  (∀ m ≤ 3/2, ∀ x ≥ 1, f x ≥ m * (x - 1)^2) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l2191_219199


namespace NUMINAMATH_CALUDE_unique_c_for_degree_two_l2191_219137

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 3 - 6*x + 4*x^2 - 5*x^3

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 4 - 3*x + 7*x^3

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The theorem stating that 5/7 is the unique value of c such that h(x) has degree 2 -/
theorem unique_c_for_degree_two :
  ∃! c : ℝ, (∀ x : ℝ, h c x = 3 + (-6 - 3*c)*x + 4*x^2) ∧ 
            (∀ x : ℝ, h c x ≠ 3 + (-6 - 3*c)*x + 4*x^2 + 0*x^3) :=
sorry

end NUMINAMATH_CALUDE_unique_c_for_degree_two_l2191_219137


namespace NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l2191_219149

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 9) 
  (h2 : x * y = 2) : 
  x^2 + y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l2191_219149


namespace NUMINAMATH_CALUDE_intersection_line_canonical_equations_l2191_219154

/-- The canonical equations of the intersection line of two planes -/
theorem intersection_line_canonical_equations 
  (plane1 : ℝ → ℝ → ℝ → ℝ) 
  (plane2 : ℝ → ℝ → ℝ → ℝ) 
  (h1 : ∀ x y z, plane1 x y z = 3*x + 4*y - 2*z + 1)
  (h2 : ∀ x y z, plane2 x y z = 2*x - 4*y + 3*z + 4) :
  ∃ (t : ℝ), ∀ x y z, 
    plane1 x y z = 0 ∧ plane2 x y z = 0 ↔ 
    (x + 1) / 4 = (y - 1/2) / (-13) ∧ (y - 1/2) / (-13) = z / (-20) ∧ 
    x = -1 + 4*t ∧ y = 1/2 - 13*t ∧ z = -20*t :=
sorry

end NUMINAMATH_CALUDE_intersection_line_canonical_equations_l2191_219154


namespace NUMINAMATH_CALUDE_quadratic_min_diff_l2191_219183

/-- The quadratic function f(x) = ax² - 2020x + 2021 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2020 * x + 2021

/-- The theorem stating that if the minimum difference between max and min values
    of f on any 2-unit interval is 2, then a must be 2 -/
theorem quadratic_min_diff (a : ℝ) (h_pos : a > 0) :
  (∀ t : ℝ, ∃ M N : ℝ,
    (∀ x ∈ Set.Icc (t - 1) (t + 1), f a x ≤ M) ∧
    (∀ x ∈ Set.Icc (t - 1) (t + 1), N ≤ f a x) ∧
    (∀ K L : ℝ,
      (∀ x ∈ Set.Icc (t - 1) (t + 1), f a x ≤ K) →
      (∀ x ∈ Set.Icc (t - 1) (t + 1), L ≤ f a x) →
      2 ≤ K - L)) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_min_diff_l2191_219183


namespace NUMINAMATH_CALUDE_equation_solutions_l2191_219157

theorem equation_solutions :
  ∀ x : ℝ, x * (x + 1) = 12 ↔ x = -4 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2191_219157


namespace NUMINAMATH_CALUDE_stations_visited_l2191_219138

/-- The number of stations visited by Joline and the junior ranger -/
def num_stations (total_nails : ℕ) (nails_per_station : ℕ) : ℕ :=
  total_nails / nails_per_station

/-- Theorem stating that the number of stations visited is 40 -/
theorem stations_visited :
  num_stations 560 14 = 40 := by
  sorry

end NUMINAMATH_CALUDE_stations_visited_l2191_219138


namespace NUMINAMATH_CALUDE_solution_range_l2191_219133

-- Define the equation
def equation (x a : ℝ) : Prop :=
  1 / (x - 2) + (a - 2) / (2 - x) = 1

-- Define the solution function
def solution (a : ℝ) : ℝ := 5 - a

-- Theorem statement
theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ equation x a) ↔ (a < 5 ∧ a ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_solution_range_l2191_219133


namespace NUMINAMATH_CALUDE_geometric_series_relation_l2191_219155

/-- Given two infinite geometric series with specific conditions, prove that m = 7 -/
theorem geometric_series_relation (m : ℤ) : 
  let a₁ : ℚ := 15  -- first term of both series
  let b₁ : ℚ := 5   -- second term of first series
  let b₂ : ℚ := 5 + m  -- second term of second series
  let r₁ : ℚ := b₁ / a₁  -- common ratio of first series
  let r₂ : ℚ := b₂ / a₁  -- common ratio of second series
  let S₁ : ℚ := a₁ / (1 - r₁)  -- sum of first series
  let S₂ : ℚ := a₁ / (1 - r₂)  -- sum of second series
  S₂ = 3 * S₁ → m = 7 := by
  sorry


end NUMINAMATH_CALUDE_geometric_series_relation_l2191_219155


namespace NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l2191_219152

theorem integer_solutions_of_inequalities :
  {x : ℤ | 2 * x + 4 > 0 ∧ 1 + x ≥ 2 * x - 1} = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l2191_219152


namespace NUMINAMATH_CALUDE_min_a_for_inequality_l2191_219166

theorem min_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + a*x + 1 ≥ 0) ↔ a ≥ -5/2 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_inequality_l2191_219166


namespace NUMINAMATH_CALUDE_gmat_question_percentage_l2191_219112

/-- The percentage of test takers who answered the second question correctly -/
def second_correct : ℝ := 75

/-- The percentage of test takers who answered neither question correctly -/
def neither_correct : ℝ := 5

/-- The percentage of test takers who answered both questions correctly -/
def both_correct : ℝ := 60

/-- The percentage of test takers who answered the first question correctly -/
def first_correct : ℝ := 80

theorem gmat_question_percentage :
  first_correct = 80 :=
sorry

end NUMINAMATH_CALUDE_gmat_question_percentage_l2191_219112


namespace NUMINAMATH_CALUDE_distinct_cube_constructions_proof_l2191_219175

/-- The number of distinct ways to construct a 2 × 2 × 2 cube 
    using 6 white unit cubes and 2 black unit cubes, 
    where constructions are considered the same if one can be rotated to match the other -/
def distinct_cube_constructions : ℕ := 3

/-- The total number of unit cubes used -/
def total_cubes : ℕ := 8

/-- The number of white unit cubes -/
def white_cubes : ℕ := 6

/-- The number of black unit cubes -/
def black_cubes : ℕ := 2

/-- The dimensions of the cube -/
def cube_dimensions : Fin 3 → ℕ := λ _ => 2

/-- The order of the rotational symmetry group of a cube -/
def cube_symmetry_order : ℕ := 24

theorem distinct_cube_constructions_proof :
  distinct_cube_constructions = 3 ∧
  total_cubes = white_cubes + black_cubes ∧
  (∀ i, cube_dimensions i = 2) ∧
  cube_symmetry_order = 24 := by
  sorry

end NUMINAMATH_CALUDE_distinct_cube_constructions_proof_l2191_219175


namespace NUMINAMATH_CALUDE_pool_water_increase_l2191_219174

theorem pool_water_increase (total_capacity : ℝ) (additional_water : ℝ) 
  (h1 : total_capacity = 1312.5)
  (h2 : additional_water = 300)
  (h3 : (0.8 : ℝ) * total_capacity = additional_water + (total_capacity - additional_water)) :
  let current_water := total_capacity - additional_water
  let new_water := current_water + additional_water
  (new_water - current_water) / current_water * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_pool_water_increase_l2191_219174


namespace NUMINAMATH_CALUDE_cubic_equation_value_l2191_219136

theorem cubic_equation_value (x : ℝ) (h : x^2 + x - 2 = 0) :
  x^3 + 2*x^2 - x + 2007 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l2191_219136


namespace NUMINAMATH_CALUDE_prob_even_sum_spinners_l2191_219168

/-- Represents a spinner with three sections -/
structure Spinner :=
  (sections : Fin 3 → ℕ)

/-- Calculates the probability of getting an even number on a spinner -/
def probEven (s : Spinner) : ℚ :=
  (Finset.filter (λ i => s.sections i % 2 = 0) Finset.univ).card / 3

/-- Calculates the probability of getting an odd number on a spinner -/
def probOdd (s : Spinner) : ℚ :=
  1 - probEven s

/-- The first spinner with sections 2, 3, and 7 -/
def spinner1 : Spinner :=
  ⟨λ i => [2, 3, 7].get i⟩

/-- The second spinner with sections 5, 3, and 6 -/
def spinner2 : Spinner :=
  ⟨λ i => [5, 3, 6].get i⟩

/-- The probability of getting an even sum when spinning both spinners -/
def probEvenSum (s1 s2 : Spinner) : ℚ :=
  probEven s1 * probEven s2 + probOdd s1 * probOdd s2

theorem prob_even_sum_spinners :
  probEvenSum spinner1 spinner2 = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_spinners_l2191_219168


namespace NUMINAMATH_CALUDE_cat_food_sale_l2191_219132

theorem cat_food_sale (total_customers : Nat) (first_group : Nat) (middle_group : Nat) (last_group : Nat)
  (first_group_cases : Nat) (last_group_cases : Nat) (total_cases : Nat)
  (h1 : total_customers = first_group + middle_group + last_group)
  (h2 : total_customers = 20)
  (h3 : first_group = 8)
  (h4 : middle_group = 4)
  (h5 : last_group = 8)
  (h6 : first_group_cases = 3)
  (h7 : last_group_cases = 1)
  (h8 : total_cases = 40)
  (h9 : total_cases = first_group * first_group_cases + middle_group * x + last_group * last_group_cases)
  : x = 2 := by
  sorry

#check cat_food_sale

end NUMINAMATH_CALUDE_cat_food_sale_l2191_219132


namespace NUMINAMATH_CALUDE_count_hyperbola_integer_tangent_points_l2191_219102

/-- The number of points on the hyperbola y = 2013/x where the tangent line
    intersects both coordinate axes at integer points -/
def hyperbola_integer_tangent_points : ℕ := 48

/-- The hyperbola equation y = 2013/x -/
def hyperbola (x y : ℝ) : Prop := y = 2013 / x

/-- Predicate for a point (x, y) on the hyperbola having a tangent line
    that intersects both axes at integer coordinates -/
def has_integer_intercepts (x y : ℝ) : Prop :=
  hyperbola x y ∧
  ∃ (x_int y_int : ℤ),
    (x_int ≠ 0 ∧ y_int ≠ 0) ∧
    (y - 2013 / x = -(2013 / x^2) * (x_int - x)) ∧
    (0 = -(2013 / x^2) * x_int + 2 * 2013 / x)

theorem count_hyperbola_integer_tangent_points :
  (∑' p : {p : ℝ × ℝ // has_integer_intercepts p.1 p.2}, 1) =
    hyperbola_integer_tangent_points :=
sorry

end NUMINAMATH_CALUDE_count_hyperbola_integer_tangent_points_l2191_219102


namespace NUMINAMATH_CALUDE_triangle_dot_product_l2191_219193

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 3^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 3^2 ∧
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define point M
def point_M (A B M : ℝ × ℝ) : Prop :=
  (M.1 - B.1) = 2 * (A.1 - M.1) ∧
  (M.2 - B.2) = 2 * (A.2 - M.2)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem triangle_dot_product 
  (A B C M : ℝ × ℝ) 
  (h1 : triangle_ABC A B C) 
  (h2 : point_M A B M) : 
  dot_product (C.1 - M.1, C.2 - M.2) (C.1 - B.1, C.2 - B.2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_dot_product_l2191_219193


namespace NUMINAMATH_CALUDE_intersection_point_l2191_219101

/-- The slope of the first line -/
def m₁ : ℚ := 3

/-- The y-intercept of the first line -/
def b₁ : ℚ := -4

/-- The x-coordinate of the given point -/
def x₀ : ℚ := 3

/-- The y-coordinate of the given point -/
def y₀ : ℚ := 2

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -1 / m₁

/-- The y-intercept of the perpendicular line -/
def b₂ : ℚ := y₀ - m₂ * x₀

/-- The x-coordinate of the intersection point -/
def x_intersect : ℚ := (b₂ - b₁) / (m₁ - m₂)

/-- The y-coordinate of the intersection point -/
def y_intersect : ℚ := m₁ * x_intersect + b₁

theorem intersection_point :
  (x_intersect = 27 / 10) ∧ (y_intersect = 41 / 10) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2191_219101


namespace NUMINAMATH_CALUDE_compare_P_Q_l2191_219187

def a : ℕ := 10^2010 - 1

def P : ℕ := (8 * a) * (3 * a)

def Q : ℕ := (4 * a) * (6 * a + 1)

theorem compare_P_Q : Q > P := by
  sorry

end NUMINAMATH_CALUDE_compare_P_Q_l2191_219187


namespace NUMINAMATH_CALUDE_grant_total_sales_l2191_219191

def baseball_cards_price : ℝ := 25
def baseball_bat_price : ℝ := 10
def baseball_glove_original_price : ℝ := 30
def baseball_glove_discount : ℝ := 0.2
def baseball_cleats_price : ℝ := 10
def baseball_cleats_count : ℕ := 2

def total_sales : ℝ :=
  baseball_cards_price +
  baseball_bat_price +
  (baseball_glove_original_price * (1 - baseball_glove_discount)) +
  (baseball_cleats_price * baseball_cleats_count)

theorem grant_total_sales :
  total_sales = 79 := by sorry

end NUMINAMATH_CALUDE_grant_total_sales_l2191_219191


namespace NUMINAMATH_CALUDE_person_2019_chooses_left_l2191_219192

def chocolate_distribution (L M R : ℕ+) (n : ℕ) : ℕ :=
  let total := L + M + R
  let full_rounds := n / total
  let remainder := n % total
  let left_count := full_rounds * L.val + min remainder L.val
  let middle_count := full_rounds * M.val + min (remainder - left_count) M.val
  let right_count := full_rounds * R.val + (remainder - left_count - middle_count)
  if (L.val : ℚ) / (left_count + 1) ≥ max ((M.val : ℚ) / (middle_count + 1)) ((R.val : ℚ) / (right_count + 1))
  then 0  -- Left table
  else if (M.val : ℚ) / (middle_count + 1) > (R.val : ℚ) / (right_count + 1)
  then 1  -- Middle table
  else 2  -- Right table

theorem person_2019_chooses_left (L M R : ℕ+) (h1 : L = 9) (h2 : M = 19) (h3 : R = 25) :
  chocolate_distribution L M R 2019 = 0 :=
sorry

end NUMINAMATH_CALUDE_person_2019_chooses_left_l2191_219192


namespace NUMINAMATH_CALUDE_michaels_crayons_value_l2191_219188

/-- The value of crayons Michael will have after the purchase -/
def total_value (initial_packs : ℕ) (additional_packs : ℕ) (price_per_pack : ℚ) : ℚ :=
  (initial_packs + additional_packs : ℚ) * price_per_pack

/-- Proof that Michael's crayons will be worth $15 after the purchase -/
theorem michaels_crayons_value :
  total_value 4 2 (5/2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_michaels_crayons_value_l2191_219188


namespace NUMINAMATH_CALUDE_smallest_x_value_l2191_219119

theorem smallest_x_value (x : ℝ) : 
  ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20 → x ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2191_219119


namespace NUMINAMATH_CALUDE_initial_flea_distance_l2191_219116

/-- Represents a flea's position on a 2D plane -/
structure FleaPosition where
  x : ℝ
  y : ℝ

/-- Represents the jump pattern of a flea -/
inductive JumpDirection
  | Right
  | Up
  | Left
  | Down

/-- Calculates the position of a flea after n jumps -/
def flea_position_after_jumps (initial_pos : FleaPosition) (direction : JumpDirection) (n : ℕ) : FleaPosition :=
  sorry

/-- Calculates the distance between two points on a 2D plane -/
def distance (p1 p2 : FleaPosition) : ℝ :=
  sorry

/-- Theorem stating the initial distance between the fleas -/
theorem initial_flea_distance (flea1_start flea2_start : FleaPosition)
  (h1 : flea_position_after_jumps flea1_start JumpDirection.Right 100 = 
        FleaPosition.mk (flea1_start.x - 50) (flea1_start.y - 50))
  (h2 : flea_position_after_jumps flea2_start JumpDirection.Left 100 = 
        FleaPosition.mk (flea2_start.x + 50) (flea2_start.y - 50))
  (h3 : distance (flea_position_after_jumps flea1_start JumpDirection.Right 100)
                 (flea_position_after_jumps flea2_start JumpDirection.Left 100) = 300) :
  distance flea1_start flea2_start = 2 :=
sorry

end NUMINAMATH_CALUDE_initial_flea_distance_l2191_219116


namespace NUMINAMATH_CALUDE_sticker_pages_l2191_219159

theorem sticker_pages (stickers_per_page : ℕ) (total_stickers : ℕ) (h1 : stickers_per_page = 10) (h2 : total_stickers = 220) :
  total_stickers / stickers_per_page = 22 := by
  sorry

end NUMINAMATH_CALUDE_sticker_pages_l2191_219159


namespace NUMINAMATH_CALUDE_dog_burrs_problem_l2191_219161

theorem dog_burrs_problem (burrs ticks : ℕ) : 
  ticks = 6 * burrs → 
  burrs + ticks = 84 → 
  burrs = 12 := by sorry

end NUMINAMATH_CALUDE_dog_burrs_problem_l2191_219161


namespace NUMINAMATH_CALUDE_equal_payment_payment_difference_l2191_219110

/-- Represents the pizza scenario with given conditions -/
structure PizzaScenario where
  total_slices : ℕ
  meat_slices : ℕ
  plain_cost : ℚ
  meat_cost : ℚ
  joe_meat_slices : ℕ
  joe_veg_slices : ℕ

/-- Calculate the total cost of the pizza -/
def total_cost (p : PizzaScenario) : ℚ :=
  p.plain_cost + p.meat_cost

/-- Calculate the cost per slice -/
def cost_per_slice (p : PizzaScenario) : ℚ :=
  total_cost p / p.total_slices

/-- Calculate Joe's payment -/
def joe_payment (p : PizzaScenario) : ℚ :=
  cost_per_slice p * (p.joe_meat_slices + p.joe_veg_slices)

/-- Calculate Karen's payment -/
def karen_payment (p : PizzaScenario) : ℚ :=
  cost_per_slice p * (p.total_slices - p.joe_meat_slices - p.joe_veg_slices)

/-- The main theorem stating that Joe and Karen paid the same amount -/
theorem equal_payment (p : PizzaScenario) 
  (h1 : p.total_slices = 12)
  (h2 : p.meat_slices = 4)
  (h3 : p.plain_cost = 12)
  (h4 : p.meat_cost = 4)
  (h5 : p.joe_meat_slices = 4)
  (h6 : p.joe_veg_slices = 2) :
  joe_payment p = karen_payment p :=
by sorry

/-- The difference in payment is zero -/
theorem payment_difference (p : PizzaScenario) 
  (h1 : p.total_slices = 12)
  (h2 : p.meat_slices = 4)
  (h3 : p.plain_cost = 12)
  (h4 : p.meat_cost = 4)
  (h5 : p.joe_meat_slices = 4)
  (h6 : p.joe_veg_slices = 2) :
  joe_payment p - karen_payment p = 0 :=
by sorry

end NUMINAMATH_CALUDE_equal_payment_payment_difference_l2191_219110


namespace NUMINAMATH_CALUDE_cylinder_from_equation_l2191_219195

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying r = d -/
def CylindricalSet (d : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = d}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ d : ℝ, d > 0 ∧ S = CylindricalSet d

/-- Theorem: The set of points satisfying r = d forms a cylinder -/
theorem cylinder_from_equation (d : ℝ) (h : d > 0) : 
  IsCylinder (CylindricalSet d) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_from_equation_l2191_219195


namespace NUMINAMATH_CALUDE_unique_perfect_square_l2191_219182

def f (k : ℕ) : ℕ := 2^k + 8*k + 5

theorem unique_perfect_square : ∃! k : ℕ, ∃ n : ℕ, f k = n^2 ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_l2191_219182


namespace NUMINAMATH_CALUDE_same_roots_imply_a_equals_five_l2191_219121

theorem same_roots_imply_a_equals_five (a : ℝ) : 
  (∀ x : ℝ, (|x|^2 - 3*|x| + 2 = 0) ↔ (x^4 - a*x^2 + 4 = 0)) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_same_roots_imply_a_equals_five_l2191_219121


namespace NUMINAMATH_CALUDE_smallest_n_for_congruence_l2191_219114

/-- Concatenation of powers of 2 -/
def A (n : ℕ) : ℕ :=
  -- We define A as a placeholder function, as the actual implementation is complex
  sorry

/-- The main theorem -/
theorem smallest_n_for_congruence : 
  (∀ k : ℕ, 3 ≤ k → k < 14 → ¬(A k ≡ 2^(10*k) [MOD 2^170])) ∧ 
  (A 14 ≡ 2^(10*14) [MOD 2^170]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_congruence_l2191_219114


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l2191_219100

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n : ℝ) * exterior_angle = 360 → exterior_angle = 45 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l2191_219100


namespace NUMINAMATH_CALUDE_problem_solution_l2191_219178

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4*a*x + 1

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := 6*a^2 * log x + 2*b + 1

noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x + g a b x

theorem problem_solution (a : ℝ) (ha : a > 0) :
  ∃ b : ℝ,
    (∃ x : ℝ, x > 0 ∧ f a x = g a b x ∧ (deriv (f a)) x = (deriv (g a b)) x) ∧
    b = (5/2)*a^2 - 3*a^2 * log a ∧
    ∀ b' : ℝ, b' ≤ (3/2) * Real.exp ((2:ℝ)/3) ∧
    (a ≥ Real.sqrt 3 - 1 →
      ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
        (h a b x₂ - h a b x₁) / (x₂ - x₁) > 8) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2191_219178


namespace NUMINAMATH_CALUDE_prob_odd_after_removal_on_die_l2191_219165

/-- Represents a standard die face with its number of dots -/
inductive DieFace
| one
| two
| three
| four
| five
| six

/-- Calculates the number of ways to choose 2 dots from n dots -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the probability of choosing 2 dots from a face with n dots -/
def prob_choose_from_face (n : ℕ) : ℚ := (choose_two n : ℚ) / (choose_two 21 : ℚ)

/-- Determines if a face will have an odd number of dots after removing 2 dots -/
def odd_after_removal (face : DieFace) : Bool :=
  match face with
  | DieFace.one => false
  | DieFace.two => false
  | DieFace.three => false
  | DieFace.four => true
  | DieFace.five => false
  | DieFace.six => true

/-- Calculates the probability of getting an odd number of dots on a specific face after removal -/
def prob_odd_after_removal (face : DieFace) : ℚ :=
  if odd_after_removal face then
    match face with
    | DieFace.four => prob_choose_from_face 4
    | DieFace.six => prob_choose_from_face 6
    | _ => 0
  else 0

/-- The main theorem to prove -/
theorem prob_odd_after_removal_on_die : 
  (1 / 6 : ℚ) * (prob_odd_after_removal DieFace.one + 
                 prob_odd_after_removal DieFace.two + 
                 prob_odd_after_removal DieFace.three + 
                 prob_odd_after_removal DieFace.four + 
                 prob_odd_after_removal DieFace.five + 
                 prob_odd_after_removal DieFace.six) = 1 / 60 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_after_removal_on_die_l2191_219165


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2191_219177

theorem square_plus_reciprocal_square (x : ℝ) (h : x + (1/x) = 3) : x^2 + (1/x^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2191_219177


namespace NUMINAMATH_CALUDE_gala_trees_count_l2191_219163

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- Determines if an orchard satisfies the given conditions -/
def satisfies_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.pure_fuji + o.cross_pollinated = 153 ∧
  o.pure_fuji = 3 * o.total / 4 ∧
  o.total = o.pure_fuji + o.pure_gala + o.cross_pollinated

theorem gala_trees_count (o : Orchard) :
  satisfies_conditions o → o.pure_gala = 45 := by
  sorry

end NUMINAMATH_CALUDE_gala_trees_count_l2191_219163


namespace NUMINAMATH_CALUDE_square_plus_minus_one_divisible_by_five_l2191_219126

theorem square_plus_minus_one_divisible_by_five (n : ℤ) (h : ¬ 5 ∣ n) : 
  5 ∣ (n^2 + 1) ∨ 5 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_square_plus_minus_one_divisible_by_five_l2191_219126


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2191_219135

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | -3 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2191_219135


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l2191_219144

theorem mean_of_remaining_numbers : 
  let numbers : List ℕ := [1867, 1993, 2019, 2025, 2109, 2121]
  let total_sum : ℕ := numbers.sum
  let mean_of_four : ℕ := 2008
  let sum_of_four : ℕ := 4 * mean_of_four
  let sum_of_two : ℕ := total_sum - sum_of_four
  sum_of_two / 2 = 2051 := by sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l2191_219144


namespace NUMINAMATH_CALUDE_cube_edge_is_nine_l2191_219128

-- Define the dimensions of the cuboid
def cuboid_base : Real := 10
def cuboid_height : Real := 73

-- Define the volume difference between the cuboid and the cube
def volume_difference : Real := 1

-- Define the function to calculate the edge length of the cube
def cube_edge_length : Real :=
  (cuboid_base * cuboid_height - volume_difference) ^ (1/3)

-- Theorem statement
theorem cube_edge_is_nine :
  cube_edge_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_is_nine_l2191_219128


namespace NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l2191_219171

/-- The number of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1's in the first n rows of Pascal's Triangle -/
def pascal_triangle_ones (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probability_of_one (n : ℕ) : ℚ :=
  (pascal_triangle_ones n : ℚ) / (pascal_triangle_elements n : ℚ)

theorem probability_of_one_in_20_rows :
  probability_of_one 20 = 13 / 70 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l2191_219171


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l2191_219122

theorem gcd_count_for_product_360 : 
  ∃! (s : Finset ℕ), 
    (∀ d ∈ s, d > 0 ∧ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ Nat.gcd a b = d ∧ Nat.gcd a b * Nat.lcm a b = 360) ∧
    s.card = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l2191_219122


namespace NUMINAMATH_CALUDE_binomial_probabilities_l2191_219106

/-- The probability of success in a single trial -/
def p : ℝ := 0.7

/-- The number of trials -/
def n : ℕ := 5

/-- The probability of failure in a single trial -/
def q : ℝ := 1 - p

/-- Binomial probability mass function -/
def binomialPMF (k : ℕ) : ℝ :=
  (n.choose k) * p^k * q^(n-k)

/-- The probability of at most 3 successes in 5 trials -/
def probAtMost3 : ℝ :=
  binomialPMF 0 + binomialPMF 1 + binomialPMF 2 + binomialPMF 3

/-- The probability of at least 4 successes in 5 trials -/
def probAtLeast4 : ℝ :=
  binomialPMF 4 + binomialPMF 5

theorem binomial_probabilities :
  probAtMost3 = 0.4718 ∧ probAtLeast4 = 0.5282 := by
  sorry

#eval probAtMost3
#eval probAtLeast4

end NUMINAMATH_CALUDE_binomial_probabilities_l2191_219106


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l2191_219170

def total_balls : ℕ := 4 + 6
def white_balls : ℕ := 4
def yellow_balls : ℕ := 6

theorem probability_of_white_ball :
  (white_balls : ℚ) / total_balls = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l2191_219170


namespace NUMINAMATH_CALUDE_min_rectangles_correct_l2191_219125

/-- The minimum number of rectangles needed to cover a board -/
def min_rectangles (n : ℕ) : ℕ := 2 * n

/-- A rectangle with integer side lengths and area equal to a power of 2 -/
structure PowerRect where
  width : ℕ
  height : ℕ
  is_power_of_two : ∃ k : ℕ, width * height = 2^k

/-- A covering of the board with rectangles -/
structure BoardCovering (n : ℕ) where
  rectangles : List PowerRect
  covers_board : (List.sum (rectangles.map (λ r => r.width * r.height))) = (2^n - 1) * (2^n + 1)

theorem min_rectangles_correct (n : ℕ) :
  ∀ (cover : BoardCovering n), cover.rectangles.length ≥ min_rectangles n ∧
  ∃ (optimal_cover : BoardCovering n), optimal_cover.rectangles.length = min_rectangles n :=
sorry

end NUMINAMATH_CALUDE_min_rectangles_correct_l2191_219125


namespace NUMINAMATH_CALUDE_least_value_x_minus_y_minus_z_l2191_219134

theorem least_value_x_minus_y_minus_z (x y z : ℕ+) 
  (h1 : x = 4 * y) (h2 : y = 7 * z) : 
  (x - y - z : ℤ) ≥ 19 ∧ ∃ (x₀ y₀ z₀ : ℕ+), 
    x₀ = 4 * y₀ ∧ y₀ = 7 * z₀ ∧ (x₀ - y₀ - z₀ : ℤ) = 19 := by
  sorry

end NUMINAMATH_CALUDE_least_value_x_minus_y_minus_z_l2191_219134


namespace NUMINAMATH_CALUDE_division_relation_l2191_219194

theorem division_relation (h : 2994 / 14.5 = 171) : 29.94 / 1.75 = 17.1 := by
  sorry

end NUMINAMATH_CALUDE_division_relation_l2191_219194


namespace NUMINAMATH_CALUDE_audience_fraction_girls_l2191_219120

theorem audience_fraction_girls (total : ℝ) (h1 : total > 0) : 
  let adults : ℝ := total / 6
  let children : ℝ := total - adults
  let boys : ℝ := (2 / 5) * children
  let girls : ℝ := children - boys
  girls / total = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_audience_fraction_girls_l2191_219120


namespace NUMINAMATH_CALUDE_triangle_side_length_l2191_219131

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively

-- State the theorem
theorem triangle_side_length 
  (ABC : Triangle) 
  (h1 : 2 * (ABC.b * Real.cos ABC.A + ABC.a * Real.cos ABC.B) = ABC.c ^ 2)
  (h2 : ABC.b = 3)
  (h3 : 3 * Real.cos ABC.A = 1) :
  ABC.a = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l2191_219131


namespace NUMINAMATH_CALUDE_saree_stripes_l2191_219123

theorem saree_stripes (brown gold blue : ℕ) : 
  gold = 3 * brown → 
  blue = 5 * gold → 
  brown = 4 → 
  blue = 60 := by
  sorry

end NUMINAMATH_CALUDE_saree_stripes_l2191_219123


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2191_219189

theorem inequality_system_solution :
  {x : ℝ | x + 2 > 3 * (2 - x) ∧ x < (x + 3) / 2} = {x : ℝ | 1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2191_219189


namespace NUMINAMATH_CALUDE_largest_gold_coins_max_gold_coins_l2191_219140

theorem largest_gold_coins (n : ℕ) : n < 120 → n % 15 = 3 → n ≤ 105 := by
  sorry

theorem max_gold_coins : ∃ n : ℕ, n = 105 ∧ n < 120 ∧ n % 15 = 3 ∧ ∀ m : ℕ, m < 120 → m % 15 = 3 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_largest_gold_coins_max_gold_coins_l2191_219140


namespace NUMINAMATH_CALUDE_tank_leak_emptying_time_l2191_219150

/-- Given a tank that can be filled in 7 hours without a leak and 8 hours with a leak,
    prove that it takes 56 hours for the tank to become empty due to the leak. -/
theorem tank_leak_emptying_time :
  ∀ (fill_rate_no_leak fill_rate_with_leak leak_rate : ℚ),
    fill_rate_no_leak = 1 / 7 →
    fill_rate_with_leak = 1 / 8 →
    fill_rate_with_leak = fill_rate_no_leak - leak_rate →
    (1 : ℚ) / leak_rate = 56 := by
  sorry

end NUMINAMATH_CALUDE_tank_leak_emptying_time_l2191_219150


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2191_219124

theorem arithmetic_mean_problem (x : ℝ) : 
  (10 + 20 + 60) / 3 = (10 + 40 + x) / 3 + 5 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2191_219124


namespace NUMINAMATH_CALUDE_canoe_weight_proof_l2191_219117

def canoe_capacity : ℕ := 6
def person_weight : ℕ := 140

def total_weight_with_dog : ℕ :=
  let people_with_dog := (2 * canoe_capacity) / 3
  let total_people_weight := people_with_dog * person_weight
  let dog_weight := person_weight / 4
  total_people_weight + dog_weight

theorem canoe_weight_proof :
  total_weight_with_dog = 595 := by
  sorry

end NUMINAMATH_CALUDE_canoe_weight_proof_l2191_219117


namespace NUMINAMATH_CALUDE_team_size_l2191_219130

/-- A soccer team with goalies, defenders, midfielders, and strikers -/
structure SoccerTeam where
  goalies : ℕ
  defenders : ℕ
  midfielders : ℕ
  strikers : ℕ

/-- The total number of players in a soccer team -/
def totalPlayers (team : SoccerTeam) : ℕ :=
  team.goalies + team.defenders + team.midfielders + team.strikers

/-- Theorem stating the total number of players in the given team -/
theorem team_size (team : SoccerTeam) 
  (h1 : team.goalies = 3)
  (h2 : team.defenders = 10)
  (h3 : team.midfielders = 2 * team.defenders)
  (h4 : team.strikers = 7) :
  totalPlayers team = 40 := by
  sorry

#eval totalPlayers { goalies := 3, defenders := 10, midfielders := 20, strikers := 7 }

end NUMINAMATH_CALUDE_team_size_l2191_219130


namespace NUMINAMATH_CALUDE_derivative_y_wrt_x_at_zero_l2191_219153

noncomputable def x (t : ℝ) : ℝ := Real.exp t * Real.cos t

noncomputable def y (t : ℝ) : ℝ := Real.exp t * Real.sin t

theorem derivative_y_wrt_x_at_zero :
  deriv (fun t => y t) 0 / deriv (fun t => x t) 0 = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_y_wrt_x_at_zero_l2191_219153


namespace NUMINAMATH_CALUDE_point_on_y_axis_l2191_219186

/-- A point P with coordinates (1-x, 2x+1) that lies on the y-axis has coordinates (0, 3) -/
theorem point_on_y_axis (x : ℝ) :
  (1 - x = 0) → (1 - x, 2*x + 1) = (0, 3) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l2191_219186


namespace NUMINAMATH_CALUDE_median_and_mode_of_scores_l2191_219185

def student_scores : List Nat := [3, 6, 4, 6, 4, 3, 6, 5, 7]

def median (l : List Nat) : Nat := sorry

def mode (l : List Nat) : Nat := sorry

theorem median_and_mode_of_scores : 
  median student_scores = 5 ∧ mode student_scores = 6 := by sorry

end NUMINAMATH_CALUDE_median_and_mode_of_scores_l2191_219185


namespace NUMINAMATH_CALUDE_vector_sum_coords_l2191_219167

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, -2)

theorem vector_sum_coords : 
  (2 : ℝ) • a + b = (-3, 4) := by sorry

end NUMINAMATH_CALUDE_vector_sum_coords_l2191_219167


namespace NUMINAMATH_CALUDE_sarah_time_hours_l2191_219169

-- Define the time Samuel took in minutes
def samuel_time : ℕ := 30

-- Define the time difference between Sarah and Samuel in minutes
def time_difference : ℕ := 48

-- Define Sarah's time in minutes
def sarah_time_minutes : ℕ := samuel_time + time_difference

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem sarah_time_hours : 
  (sarah_time_minutes : ℚ) / minutes_per_hour = 1.3 := by
  sorry

end NUMINAMATH_CALUDE_sarah_time_hours_l2191_219169


namespace NUMINAMATH_CALUDE_solution_system_equations_l2191_219158

theorem solution_system_equations (x y z : ℝ) : 
  x^2 + y^2 = 6*z ∧ 
  y^2 + z^2 = 6*x ∧ 
  z^2 + x^2 = 6*y → 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 3 ∧ y = 3 ∧ z = 3) := by
sorry

end NUMINAMATH_CALUDE_solution_system_equations_l2191_219158


namespace NUMINAMATH_CALUDE_power_simplification_l2191_219141

theorem power_simplification : 16^10 * 8^5 / 4^15 = 2^25 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l2191_219141


namespace NUMINAMATH_CALUDE_special_hyperbola_equation_l2191_219104

/-- A hyperbola with center at the origin, foci on the x-axis, and specific properties. -/
structure SpecialHyperbola where
  -- The equation of the hyperbola in the form x²/a² - y²/b² = 1
  a : ℝ
  b : ℝ
  -- The right focus is at (c, 0) where c² = a² + b²
  c : ℝ
  h_c : c^2 = a^2 + b^2
  -- A line through the right focus with slope √(3/5)
  line_slope : ℝ
  h_slope : line_slope^2 = 3/5
  -- The line intersects the hyperbola at P and Q
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_P_on_hyperbola : (P.1/a)^2 - (P.2/b)^2 = 1
  h_Q_on_hyperbola : (Q.1/a)^2 - (Q.2/b)^2 = 1
  h_P_on_line : P.2 = line_slope * (P.1 - c)
  h_Q_on_line : Q.2 = line_slope * (Q.1 - c)
  -- PO ⊥ OQ
  h_perpendicular : P.1 * Q.1 + P.2 * Q.2 = 0
  -- |PQ| = 4
  h_distance : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 16

/-- The theorem stating that the special hyperbola has the equation x² - y²/3 = 1 -/
theorem special_hyperbola_equation (h : SpecialHyperbola) : h.a^2 = 1 ∧ h.b^2 = 3 := by
  sorry

#check special_hyperbola_equation

end NUMINAMATH_CALUDE_special_hyperbola_equation_l2191_219104


namespace NUMINAMATH_CALUDE_a_2_times_a_3_l2191_219162

def a : ℕ → ℤ
  | n => if n % 2 = 1 then 3 * n + 1 else 2 * n - 2

theorem a_2_times_a_3 : a 2 * a 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_a_2_times_a_3_l2191_219162


namespace NUMINAMATH_CALUDE_trig_expression_equality_l2191_219160

theorem trig_expression_equality : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 1 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l2191_219160


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l2191_219143

theorem product_mod_seventeen : (2024 * 2025 * 2026 * 2027 * 2028) % 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l2191_219143


namespace NUMINAMATH_CALUDE_f_neither_odd_nor_even_l2191_219176

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2

-- Define the domain of f
def domain : Set ℝ := Set.Ioc (-5) 5

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statement
theorem f_neither_odd_nor_even :
  ¬(is_odd f) ∧ ¬(is_even f) :=
sorry

end NUMINAMATH_CALUDE_f_neither_odd_nor_even_l2191_219176


namespace NUMINAMATH_CALUDE_roque_bike_time_l2191_219113

/-- Represents the time in hours for Roque's commute -/
structure CommuteTime where
  walk_one_way : ℝ
  bike_one_way : ℝ
  walk_trips_per_week : ℕ
  bike_trips_per_week : ℕ
  total_time_per_week : ℝ

/-- Theorem stating that given the conditions, Roque's bike ride to work takes 1 hour -/
theorem roque_bike_time (c : CommuteTime)
  (h1 : c.walk_one_way = 2)
  (h2 : c.walk_trips_per_week = 3)
  (h3 : c.bike_trips_per_week = 2)
  (h4 : c.total_time_per_week = 16)
  (h5 : c.total_time_per_week = 2 * c.walk_one_way * c.walk_trips_per_week + 2 * c.bike_one_way * c.bike_trips_per_week) :
  c.bike_one_way = 1 := by
  sorry

end NUMINAMATH_CALUDE_roque_bike_time_l2191_219113


namespace NUMINAMATH_CALUDE_g_diverges_from_negative_two_l2191_219180

def g (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem g_diverges_from_negative_two :
  ∀ n : ℕ, ∃ M : ℝ, M > 0 ∧ 
  (n.iterate g (-2) > M ∧ n.iterate g (-2) > n.pred.iterate g (-2)) :=
sorry

end NUMINAMATH_CALUDE_g_diverges_from_negative_two_l2191_219180


namespace NUMINAMATH_CALUDE_least_positive_tan_value_l2191_219147

theorem least_positive_tan_value (x a b : ℝ) (h1 : Real.tan x = a / b) 
  (h2 : Real.tan (2 * x) = 2 * b / (a + 2 * b)) :
  ∃ k, k > 0 ∧ x = k ∧ Real.arctan 1 = k := by sorry

end NUMINAMATH_CALUDE_least_positive_tan_value_l2191_219147


namespace NUMINAMATH_CALUDE_sandy_savings_l2191_219172

theorem sandy_savings (last_year_salary : ℝ) : 
  let last_year_savings := 0.1 * last_year_salary
  let this_year_salary := 1.1 * last_year_salary
  let this_year_savings := 0.6599999999999999 * last_year_savings
  (this_year_savings / this_year_salary) * 100 = 6 := by
sorry

end NUMINAMATH_CALUDE_sandy_savings_l2191_219172


namespace NUMINAMATH_CALUDE_parabola_properties_given_parabola_properties_l2191_219115

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  D : ℝ × ℝ

/-- Given conditions for the parabola -/
def given_parabola : Parabola where
  p := 2  -- This is derived from the solution, not given directly
  equation := λ x y => y^2 = 2 * 2 * x
  focus := (1, 0)
  D := (2, 0)

/-- Theorem stating the main results -/
theorem parabola_properties (C : Parabola) 
  (h1 : C.p > 0)
  (h2 : C.D = (C.p, 0))
  (h3 : ∃ (M : ℝ × ℝ), C.equation M.1 M.2 ∧ 
        (M.2 - C.D.2) / (M.1 - C.D.1) = 0 ∧ 
        Real.sqrt ((M.1 - C.focus.1)^2 + (M.2 - C.focus.2)^2) = 3) :
  (C.equation = λ x y => y^2 = 4*x) ∧
  (∃ (A B : ℝ × ℝ), 
    C.equation A.1 A.2 ∧ 
    C.equation B.1 B.2 ∧
    (B.2 - A.2) / (B.1 - A.1) = -1/Real.sqrt 2 ∧
    A.1 - Real.sqrt 2 * A.2 - 4 = 0) :=
by sorry

/-- Applying the theorem to the given parabola -/
theorem given_parabola_properties : 
  (given_parabola.equation = λ x y => y^2 = 4*x) ∧
  (∃ (A B : ℝ × ℝ), 
    given_parabola.equation A.1 A.2 ∧ 
    given_parabola.equation B.1 B.2 ∧
    (B.2 - A.2) / (B.1 - A.1) = -1/Real.sqrt 2 ∧
    A.1 - Real.sqrt 2 * A.2 - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_given_parabola_properties_l2191_219115


namespace NUMINAMATH_CALUDE_squirrel_pine_cones_theorem_l2191_219151

/-- Represents the number of pine cones each squirrel has -/
structure SquirrelPineCones where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Redistributes pine cones according to the problem description -/
def redistribute (initial : SquirrelPineCones) : SquirrelPineCones :=
  let step1 := SquirrelPineCones.mk (initial.a - 10) (initial.b + 5) (initial.c + 5)
  let step2 := SquirrelPineCones.mk (step1.a + 9) (step1.b - 18) (step1.c + 9)
  let final_c := step2.c / 2
  SquirrelPineCones.mk (step2.a + final_c) (step2.b + final_c) final_c

/-- The theorem to be proved -/
theorem squirrel_pine_cones_theorem (initial : SquirrelPineCones) 
  (h1 : initial.a = 26)
  (h2 : initial.c = 86) :
  let final := redistribute initial
  final.a = final.b ∧ final.b = final.c := by
  sorry

end NUMINAMATH_CALUDE_squirrel_pine_cones_theorem_l2191_219151


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2191_219148

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2191_219148


namespace NUMINAMATH_CALUDE_pascal_triangle_53_l2191_219196

theorem pascal_triangle_53 (p : ℕ) (h_prime : Prime p) (h_p : p = 53) :
  (∃! n : ℕ, ∃ k : ℕ, Nat.choose n k = p) :=
sorry

end NUMINAMATH_CALUDE_pascal_triangle_53_l2191_219196


namespace NUMINAMATH_CALUDE_cats_remaining_l2191_219198

theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) 
  (h1 : siamese = 13) 
  (h2 : house = 5) 
  (h3 : sold = 10) : 
  siamese + house - sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_l2191_219198


namespace NUMINAMATH_CALUDE_square_difference_division_equals_318_l2191_219142

theorem square_difference_division_equals_318 : (165^2 - 153^2) / 12 = 318 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_division_equals_318_l2191_219142


namespace NUMINAMATH_CALUDE_original_vocabulary_l2191_219164

/-- The number of words learned per day -/
def words_per_day : ℕ := 10

/-- The number of days in 2 years -/
def days_in_two_years : ℕ := 365 * 2

/-- The percentage increase in vocabulary -/
def percentage_increase : ℚ := 1 / 2

theorem original_vocabulary (original : ℕ) : 
  (original : ℚ) + (original : ℚ) * percentage_increase = 
    (words_per_day * days_in_two_years : ℚ) → 
  original = 14600 := by sorry

end NUMINAMATH_CALUDE_original_vocabulary_l2191_219164


namespace NUMINAMATH_CALUDE_min_teachers_is_16_l2191_219107

/-- Represents the number of teachers in each subject --/
structure TeacherCounts where
  maths : Nat
  physics : Nat
  chemistry : Nat

/-- Calculates the minimum number of teachers required --/
def minTeachersRequired (counts : TeacherCounts) : Nat :=
  counts.maths + counts.physics + counts.chemistry

/-- Theorem stating the minimum number of teachers required --/
theorem min_teachers_is_16 (counts : TeacherCounts) 
  (h_maths : counts.maths = 6)
  (h_physics : counts.physics = 5)
  (h_chemistry : counts.chemistry = 5) :
  minTeachersRequired counts = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_teachers_is_16_l2191_219107


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l2191_219118

/-- Converts a list of bits (0s and 1s) to a natural number -/
def binaryToNat (bits : List Nat) : Nat :=
  bits.foldl (fun acc bit => 2 * acc + bit) 0

/-- The theorem to be proved -/
theorem binary_arithmetic_equality : 
  let a := binaryToNat [1, 1, 0, 1, 1]
  let b := binaryToNat [1, 0, 1, 0]
  let c := binaryToNat [1, 0, 0, 0, 1]
  let d := binaryToNat [1, 0, 1, 1]
  let e := binaryToNat [1, 1, 1, 0]
  let result := binaryToNat [0, 0, 1, 0, 0, 1]
  a + b - c + d - e = result := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l2191_219118
