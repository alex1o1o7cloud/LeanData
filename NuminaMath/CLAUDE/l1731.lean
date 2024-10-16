import Mathlib

namespace NUMINAMATH_CALUDE_no_integer_tangent_l1731_173141

/-- A circle with a point P outside it, from which a tangent and a secant are drawn -/
structure CircleWithExternalPoint where
  /-- The circumference of the circle -/
  circumference : ℝ
  /-- The length of the first arc created by the secant -/
  m : ℝ
  /-- The length of the second arc created by the secant -/
  n : ℝ
  /-- The length of the tangent -/
  t : ℝ
  /-- The circumference is 12π -/
  circ_eq : circumference = 12 * Real.pi
  /-- The sum of m and n equals the circumference -/
  arc_sum : m + n = circumference
  /-- m is thrice n -/
  m_thrice_n : m = 3 * n
  /-- The tangent is the mean proportional between m and n -/
  tangent_mean_prop : t ^ 2 = m * n

/-- Theorem stating that there are no integer values for the tangent length -/
theorem no_integer_tangent (c : CircleWithExternalPoint) : ¬ ∃ (k : ℤ), c.t = k := by
  sorry

end NUMINAMATH_CALUDE_no_integer_tangent_l1731_173141


namespace NUMINAMATH_CALUDE_disjoint_sets_imply_a_values_l1731_173149

-- Define the sets A and B
def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = a + 1}

def B (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (a^2 - 1) * p.1 + (a - 1) * p.2 = 15}

-- State the theorem
theorem disjoint_sets_imply_a_values (a : ℝ) :
  A a ∩ B a = ∅ → a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -4 :=
by sorry

end NUMINAMATH_CALUDE_disjoint_sets_imply_a_values_l1731_173149


namespace NUMINAMATH_CALUDE_nates_scallop_cost_l1731_173115

/-- Calculates the cost of scallops for a dinner party. -/
def scallop_cost (scallops_per_pound : ℕ) (cost_per_pound : ℚ) 
                 (scallops_per_person : ℕ) (num_people : ℕ) : ℚ :=
  let total_scallops := num_people * scallops_per_person
  let pounds_needed := total_scallops / scallops_per_pound
  pounds_needed * cost_per_pound

/-- The cost of scallops for Nate's dinner party is $48.00. -/
theorem nates_scallop_cost :
  scallop_cost 8 24 2 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_nates_scallop_cost_l1731_173115


namespace NUMINAMATH_CALUDE_fraction_integer_iff_q_values_l1731_173187

theorem fraction_integer_iff_q_values (q : ℕ+) :
  (∃ (k : ℕ+), (4 * q + 28 : ℚ) / (3 * q - 7 : ℚ) = k) ↔ q ∈ ({7, 15, 25} : Set ℕ+) := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_q_values_l1731_173187


namespace NUMINAMATH_CALUDE_quadrilateral_front_view_solids_l1731_173120

-- Define the possible solid figures
inductive SolidFigure
  | Cone
  | Cylinder
  | TriangularPyramid
  | RectangularPrism

-- Define a predicate for having a quadrilateral front view
def has_quadrilateral_front_view (s : SolidFigure) : Prop :=
  match s with
  | SolidFigure.Cylinder => True
  | SolidFigure.RectangularPrism => True
  | _ => False

-- Theorem statement
theorem quadrilateral_front_view_solids (s : SolidFigure) :
  has_quadrilateral_front_view s ↔ (s = SolidFigure.Cylinder ∨ s = SolidFigure.RectangularPrism) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_front_view_solids_l1731_173120


namespace NUMINAMATH_CALUDE_right_triangle_sin_sum_l1731_173181

theorem right_triangle_sin_sum (A B C : Real) (a b c : Real) : 
  -- ABC is a right triangle with ∠C = 90°
  A + B + C = Real.pi →
  C = Real.pi / 2 →
  -- Side lengths
  a = 15 →
  b = 8 →
  -- c is the hypotenuse (though we don't need to specify its value)
  c^2 = a^2 + b^2 →
  -- Sine definitions
  Real.sin A = a / c →
  Real.sin B = b / c →
  -- Theorem to prove
  Real.sin A + Real.sin B = 23 / 17 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sin_sum_l1731_173181


namespace NUMINAMATH_CALUDE_odd_function_and_monotonicity_l1731_173144

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 1) / (x + a)

theorem odd_function_and_monotonicity (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 0 ∧ ∀ x y, 0 < x → x < y → f a x < f a y) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_and_monotonicity_l1731_173144


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l1731_173167

theorem sum_of_cubes_of_roots (r s t : ℝ) : 
  (r - (27 : ℝ)^(1/3 : ℝ)) * (r - (64 : ℝ)^(1/3 : ℝ)) * (r - (125 : ℝ)^(1/3 : ℝ)) = 1/2 →
  (s - (27 : ℝ)^(1/3 : ℝ)) * (s - (64 : ℝ)^(1/3 : ℝ)) * (s - (125 : ℝ)^(1/3 : ℝ)) = 1/2 →
  (t - (27 : ℝ)^(1/3 : ℝ)) * (t - (64 : ℝ)^(1/3 : ℝ)) * (t - (125 : ℝ)^(1/3 : ℝ)) = 1/2 →
  r ≠ s → r ≠ t → s ≠ t →
  r^3 + s^3 + t^3 = 214.5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l1731_173167


namespace NUMINAMATH_CALUDE_race_distance_l1731_173111

/-- The race problem -/
theorem race_distance (a_time b_time : ℕ) (beat_distance : ℕ) (total_distance : ℕ) : 
  a_time = 20 →
  b_time = 25 →
  beat_distance = 26 →
  (total_distance : ℚ) / a_time * b_time = total_distance + beat_distance →
  total_distance = 104 := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l1731_173111


namespace NUMINAMATH_CALUDE_equation_solution_l1731_173182

theorem equation_solution : ∃! x : ℝ, (6 * x) / (x + 2) - 4 / (x + 2) = 2 / (x + 2) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1731_173182


namespace NUMINAMATH_CALUDE_triangle_properties_l1731_173123

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  2 * t.c * Real.cos t.A = 2 * t.b - Real.sqrt 3 * t.a

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t) : 
  t.C = Real.pi / 6 ∧ 
  (t.b = 2 → 2 * Real.sqrt 3 = 1/2 * t.a * t.b * Real.sin t.C → 
   Real.sin t.A = Real.sqrt 7 / 7) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1731_173123


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l1731_173142

theorem roots_quadratic_equation (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) →
  q = -2*p →
  p = 1 ∧ q = -2 := by
sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l1731_173142


namespace NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l1731_173180

/-- Calculates the toll for a truck given the number of axles -/
def toll (axles : ℕ) : ℚ :=
  1.5 + 0.5 * (axles - 2)

/-- Calculates the number of axles for a truck given the total number of wheels -/
def axles_count (total_wheels : ℕ) : ℕ :=
  1 + (total_wheels - 2) / 4

theorem eighteen_wheel_truck_toll :
  toll (axles_count 18) = 3 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l1731_173180


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_sum_180_l1731_173125

theorem largest_of_five_consecutive_sum_180 (a : ℕ) :
  (∃ (x : ℕ), x = a ∧ 
    x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 180) →
  a + 4 = 38 :=
by sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_sum_180_l1731_173125


namespace NUMINAMATH_CALUDE_work_multiple_l1731_173196

/-- Given that P people can complete a job in 8 days, 
    this theorem proves that 2P people can complete half the job in 2 days -/
theorem work_multiple (P : ℕ) : 
  (P * 8 : ℚ)⁻¹ * 2 * P * 2 = 1/2 := by sorry

end NUMINAMATH_CALUDE_work_multiple_l1731_173196


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_50_and_294_l1731_173165

theorem smallest_n_divisible_by_50_and_294 :
  ∃ (n : ℕ), n > 0 ∧ 50 ∣ n^2 ∧ 294 ∣ n^3 ∧
  ∀ (m : ℕ), m > 0 ∧ 50 ∣ m^2 ∧ 294 ∣ m^3 → n ≤ m :=
by
  use 210
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_50_and_294_l1731_173165


namespace NUMINAMATH_CALUDE_f_properties_l1731_173102

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_properties (a : ℝ) (h : a > 0) :
  ∃ (x_min : ℝ), 
    (∀ x, f a x ≥ f a x_min) ∧ 
    (x_min = Real.log (1 / a)) ∧
    (f a x_min > 2 * Real.log a + 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1731_173102


namespace NUMINAMATH_CALUDE_exponent_calculation_l1731_173186

theorem exponent_calculation : ((15^15 / 15^14)^3 * 3^3) / 3^3 = 3375 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l1731_173186


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l1731_173131

theorem polynomial_division_quotient : 
  ∀ (x : ℝ), (10 * x^3 + 20 * x^2 - 9 * x + 6) = (2 * x + 6) * (5 * x^2 - 5 * x + 3) + (-57) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l1731_173131


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1731_173116

/-- A quadratic function f(x) = ax^2 + bx + c with roots at -1/2 and 2 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_inequality_properties (a b c : ℝ) :
  (∀ x ∈ Set.Ioo (-1/2 : ℝ) 2, QuadraticFunction a b c x > 0) →
  (QuadraticFunction a b c (-1/2) = 0) →
  (QuadraticFunction a b c 2 = 0) →
  (b > 0 ∧ c > 0 ∧ a + b + c > 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1731_173116


namespace NUMINAMATH_CALUDE_xyz_product_magnitude_l1731_173104

theorem xyz_product_magnitude (x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 → 
  x ≠ y → y ≠ z → x ≠ z →
  x + 1/y = y + 1/z → y + 1/z = z + 1/x + 1 →
  |x*y*z| = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_magnitude_l1731_173104


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1731_173197

theorem quadratic_inequality_condition (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2)*x - 2*k + 4 < 0) ↔ -6 < k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1731_173197


namespace NUMINAMATH_CALUDE_distance_time_relationship_l1731_173136

/-- Represents the speed of a car in km/h -/
def speed : ℝ := 70

/-- Represents the distance traveled by the car in km -/
def distance (t : ℝ) : ℝ := speed * t

/-- Theorem stating the relationship between distance and time for the car -/
theorem distance_time_relationship (t : ℝ) : 
  distance t = speed * t ∧ 
  (∃ (S : ℝ → ℝ), S = distance ∧ (∀ x, S x = speed * x)) := by
  sorry

/-- The independent variable is time -/
def independent_variable : Type := ℝ

/-- The dependent variable is distance -/
def dependent_variable : ℝ → ℝ := distance

end NUMINAMATH_CALUDE_distance_time_relationship_l1731_173136


namespace NUMINAMATH_CALUDE_chef_cooked_seven_potatoes_l1731_173193

/-- Represents the cooking scenario of a chef with potatoes -/
structure PotatoCookingScenario where
  total_potatoes : ℕ
  cooking_time_per_potato : ℕ
  remaining_cooking_time : ℕ

/-- Calculates the number of potatoes already cooked -/
def potatoes_already_cooked (scenario : PotatoCookingScenario) : ℕ :=
  scenario.total_potatoes - (scenario.remaining_cooking_time / scenario.cooking_time_per_potato)

/-- Theorem stating that the chef has already cooked 7 potatoes -/
theorem chef_cooked_seven_potatoes (scenario : PotatoCookingScenario)
  (h1 : scenario.total_potatoes = 16)
  (h2 : scenario.cooking_time_per_potato = 5)
  (h3 : scenario.remaining_cooking_time = 45) :
  potatoes_already_cooked scenario = 7 := by
  sorry

#eval potatoes_already_cooked { total_potatoes := 16, cooking_time_per_potato := 5, remaining_cooking_time := 45 }

end NUMINAMATH_CALUDE_chef_cooked_seven_potatoes_l1731_173193


namespace NUMINAMATH_CALUDE_inequality_theorem_l1731_173155

open Real

theorem inequality_theorem (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x > 0, f x > 0)
  (h2 : ∀ x, deriv f x < f x)
  (h3 : 0 < a ∧ a < 1) :
  3 * f 0 > f a ∧ f a > a * f 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1731_173155


namespace NUMINAMATH_CALUDE_cost_per_serving_soup_l1731_173143

/-- Calculates the cost per serving of soup given ingredient quantities and prices -/
theorem cost_per_serving_soup (beef_quantity beef_price chicken_quantity chicken_price
                               carrot_quantity carrot_price potato_quantity potato_price
                               onion_quantity onion_price servings : ℚ) :
  beef_quantity = 4 →
  beef_price = 6 →
  chicken_quantity = 3 →
  chicken_price = 4 →
  carrot_quantity = 2 →
  carrot_price = (3/2) →
  potato_quantity = 3 →
  potato_price = 2 →
  onion_quantity = 1 →
  onion_price = 3 →
  servings = 12 →
  (beef_quantity * beef_price +
   chicken_quantity * chicken_price +
   carrot_quantity * carrot_price +
   potato_quantity * potato_price +
   onion_quantity * onion_price) / servings = 4 :=
by sorry

end NUMINAMATH_CALUDE_cost_per_serving_soup_l1731_173143


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l1731_173124

theorem arccos_equation_solution :
  ∃ x : ℝ, x = -1/3 ∧ Real.arccos (3*x) - Real.arccos (2*x) = π/6 :=
by sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l1731_173124


namespace NUMINAMATH_CALUDE_rhombus_height_is_half_side_l1731_173160

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  side : ℝ
  diag1 : ℝ
  diag2 : ℝ
  side_positive : 0 < side
  diag1_positive : 0 < diag1
  diag2_positive : 0 < diag2
  geometric_mean : side ^ 2 = diag1 * diag2

/-- The height of a rhombus with side s that is the geometric mean of its diagonals is s/2 -/
theorem rhombus_height_is_half_side (r : Rhombus) : 
  r.side / 2 = (r.diag1 * r.diag2) / (4 * r.side) := by
  sorry

#check rhombus_height_is_half_side

end NUMINAMATH_CALUDE_rhombus_height_is_half_side_l1731_173160


namespace NUMINAMATH_CALUDE_student_position_l1731_173129

theorem student_position (total_students : ℕ) (position_from_back : ℕ) (position_from_front : ℕ) :
  total_students = 27 →
  position_from_back = 13 →
  position_from_front = total_students - position_from_back + 1 →
  position_from_front = 15 :=
by sorry

end NUMINAMATH_CALUDE_student_position_l1731_173129


namespace NUMINAMATH_CALUDE_inequality_range_l1731_173190

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, |2*x - a| > x - 1) ↔ (a < 3 ∨ a > 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1731_173190


namespace NUMINAMATH_CALUDE_triangle_diameter_quadrilateral_diameter_pentagon_diameter_hexagon_diameter_l1731_173103

-- Define a convex n-gon with all sides equal to 1 and diameter d
def ConvexNGon (n : ℕ) (d : ℝ) : Prop :=
  n ≥ 3 ∧ d > 0

-- Theorem for n = 3
theorem triangle_diameter (d : ℝ) (h : ConvexNGon 3 d) : d = 1 := by
  sorry

-- Theorem for n = 4
theorem quadrilateral_diameter (d : ℝ) (h : ConvexNGon 4 d) : Real.sqrt 2 ≤ d ∧ d < 2 := by
  sorry

-- Theorem for n = 5
theorem pentagon_diameter (d : ℝ) (h : ConvexNGon 5 d) : (1 + Real.sqrt 5) / 2 ≤ d ∧ d < 2 := by
  sorry

-- Theorem for n = 6
theorem hexagon_diameter (d : ℝ) (h : ConvexNGon 6 d) : Real.sqrt (2 + Real.sqrt 3) ≤ d ∧ d < 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_diameter_quadrilateral_diameter_pentagon_diameter_hexagon_diameter_l1731_173103


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_for_parallel_l1731_173183

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∨ l1.b ≠ 0

/-- The lines l₁ and l₂ as functions of a --/
def l1 (a : ℝ) : Line := ⟨a, a + 2, 1⟩
def l2 (a : ℝ) : Line := ⟨a, -1, 2⟩

/-- Statement: a = -3 is a sufficient but not necessary condition for l₁ // l₂ --/
theorem sufficient_but_not_necessary_for_parallel :
  (∀ a : ℝ, a = -3 → are_parallel (l1 a) (l2 a)) ∧
  ¬(∀ a : ℝ, are_parallel (l1 a) (l2 a) → a = -3) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_for_parallel_l1731_173183


namespace NUMINAMATH_CALUDE_greatest_constant_inequality_l1731_173192

theorem greatest_constant_inequality (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^2 + b^2 + c^2 + d^2) := by
  sorry

#check greatest_constant_inequality

end NUMINAMATH_CALUDE_greatest_constant_inequality_l1731_173192


namespace NUMINAMATH_CALUDE_second_derivative_sin_plus_cos_l1731_173152

open Real

theorem second_derivative_sin_plus_cos :
  let f : ℝ → ℝ := fun x ↦ sin x + cos x
  ∀ x : ℝ, (deriv^[2] f) x = -(cos x) - sin x := by
  sorry

end NUMINAMATH_CALUDE_second_derivative_sin_plus_cos_l1731_173152


namespace NUMINAMATH_CALUDE_parabola_vertex_l1731_173188

/-- The vertex of the parabola y = 3x^2 - 6x + 2 is (1, -1) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 3 * x^2 - 6 * x + 2 → (1, -1) = (x, y) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1731_173188


namespace NUMINAMATH_CALUDE_sphere_diameter_triple_volume_l1731_173127

theorem sphere_diameter_triple_volume (π : ℝ) (h_π : π > 0) : 
  let r₁ : ℝ := 6
  let V₁ : ℝ := (4/3) * π * r₁^3
  let V₂ : ℝ := 3 * V₁
  let r₂ : ℝ := (V₂ / ((4/3) * π))^(1/3)
  2 * r₂ = 12 * (2 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_diameter_triple_volume_l1731_173127


namespace NUMINAMATH_CALUDE_numerator_increase_percentage_l1731_173128

theorem numerator_increase_percentage (P : ℝ) : 
  (5 * (1 + P / 100)) / (7 * (1 - 10 / 100)) = 20 / 21 → P = 20 := by
  sorry

end NUMINAMATH_CALUDE_numerator_increase_percentage_l1731_173128


namespace NUMINAMATH_CALUDE_math_books_count_l1731_173146

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℕ) :
  total_books = 90 →
  math_cost = 4 →
  history_cost = 5 →
  total_price = 396 →
  ∃ (math_books : ℕ),
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧
    math_books = 54 :=
by sorry

end NUMINAMATH_CALUDE_math_books_count_l1731_173146


namespace NUMINAMATH_CALUDE_xyz_sum_value_l1731_173178

theorem xyz_sum_value (x y z : ℝ) 
  (h1 : x^2 - y*z = 2) 
  (h2 : y^2 - z*x = 2) 
  (h3 : z^2 - x*y = 2) : 
  x*y + y*z + z*x = -2 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_value_l1731_173178


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1731_173100

theorem quadratic_inequality_range (c : ℝ) : 
  (¬ ∀ x : ℝ, c ≤ -1/2 → x^2 + 4*c*x + 1 > 0) → c ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1731_173100


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1731_173101

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2)  -- Pythagorean theorem
  (h2 : c = 10)           -- Hypotenuse is 10
  (h3 : a = 6)            -- One side is 6
  : b = 8 :=              -- Prove the other side is 8
by sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1731_173101


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1731_173145

theorem quadratic_inequality_solution : 
  {x : ℝ | x^2 + 2*x ≤ -1} = {-1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1731_173145


namespace NUMINAMATH_CALUDE_train_length_proof_l1731_173112

/-- The length of a train that passes a pole in 15 seconds and a 100-meter platform in 40 seconds -/
def train_length : ℝ := 60

theorem train_length_proof (t : ℝ) (h1 : t > 0) :
  (t / 15 = (t + 100) / 40) → t = train_length :=
by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l1731_173112


namespace NUMINAMATH_CALUDE_potato_cost_proof_l1731_173135

/-- The original cost of one bag of potatoes from the farmer -/
def original_cost : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase factor -/
def andrey_increase : ℝ := 2

/-- Boris's first price increase factor -/
def boris_first_increase : ℝ := 1.6

/-- Boris's second price increase factor -/
def boris_second_increase : ℝ := 1.4

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The extra profit Boris made compared to Andrey -/
def extra_profit : ℝ := 1200

theorem potato_cost_proof :
  bags_bought * original_cost * andrey_increase +
  extra_profit =
  boris_first_sale * original_cost * boris_first_increase +
  boris_second_sale * original_cost * boris_first_increase * boris_second_increase :=
by sorry

end NUMINAMATH_CALUDE_potato_cost_proof_l1731_173135


namespace NUMINAMATH_CALUDE_abs_neg_three_times_two_l1731_173176

theorem abs_neg_three_times_two : |(-3 : ℤ)| * 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_times_two_l1731_173176


namespace NUMINAMATH_CALUDE_negative_three_is_monomial_l1731_173198

/-- A monomial is a constant term or a variable raised to a non-negative integer power -/
def IsMonomial (x : ℝ) : Prop :=
  x ≠ 0 ∨ ∃ (n : ℕ), x = 1 ∨ x = -1

/-- Prove that -3 is a monomial -/
theorem negative_three_is_monomial : IsMonomial (-3) := by
  sorry

end NUMINAMATH_CALUDE_negative_three_is_monomial_l1731_173198


namespace NUMINAMATH_CALUDE_orange_cost_l1731_173163

/-- Given that 4 dozen oranges cost $28.80, prove that 5 dozen oranges at the same rate cost $36.00 -/
theorem orange_cost (cost_four_dozen : ℝ) (h1 : cost_four_dozen = 28.80) :
  let cost_per_dozen : ℝ := cost_four_dozen / 4
  let cost_five_dozen : ℝ := 5 * cost_per_dozen
  cost_five_dozen = 36 :=
by sorry

end NUMINAMATH_CALUDE_orange_cost_l1731_173163


namespace NUMINAMATH_CALUDE_regular_polygon_60_properties_l1731_173119

/-- A regular polygon with 60 sides -/
structure RegularPolygon60 where
  -- No additional fields needed as the number of sides is fixed

/-- The number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The measure of an exterior angle in a regular polygon -/
def exterior_angle (n : ℕ) : ℚ := 360 / n

theorem regular_polygon_60_properties (p : RegularPolygon60) :
  (num_diagonals 60 = 1710) ∧ (exterior_angle 60 = 6) := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_60_properties_l1731_173119


namespace NUMINAMATH_CALUDE_cube_parabola_locus_l1731_173118

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  origin : Point3D
  sideLength : ℝ

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ :=
  sorry

/-- Calculate the distance from a point to a plane -/
def distanceToPlane (p : Point3D) (plane : Plane3D) : ℝ :=
  sorry

/-- Check if a point is on a face of the cube -/
def isOnFace (p : Point3D) (cube : Cube) (face : Plane3D) : Prop :=
  sorry

/-- Define a parabola as a set of points -/
def isParabola (points : Set Point3D) : Prop :=
  sorry

theorem cube_parabola_locus (cube : Cube) (B : Point3D) (faceBCC₁B₁ planeCDD₁C₁ : Plane3D) :
  let locus := {M : Point3D | isOnFace M cube faceBCC₁B₁ ∧ 
                               distance M B = distanceToPlane M planeCDD₁C₁}
  isParabola locus := by
  sorry

end NUMINAMATH_CALUDE_cube_parabola_locus_l1731_173118


namespace NUMINAMATH_CALUDE_min_dot_product_l1731_173122

-- Define the rectangle ABCD
def Rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ A.2 = B.2 ∧ B.1 = C.1 ∧ C.2 = D.2

-- Define the points P and Q
def P (x : ℝ) : ℝ × ℝ := (2 - x, 0)
def Q (x : ℝ) : ℝ × ℝ := (2, 1 + x)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem min_dot_product :
  ∀ (A B C D : ℝ × ℝ) (x : ℝ),
    Rectangle A B C D →
    A = (0, 1) →
    B = (2, 1) →
    C = (2, 0) →
    D = (0, 0) →
    0 ≤ x →
    x ≤ 2 →
    (∀ y : ℝ, 0 ≤ y → y ≤ 2 →
      dot_product ((-2 + y, 1)) (y, 1 + y) ≥ 3/4) :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_l1731_173122


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1731_173157

-- Define the quadratic function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the function F
def F (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the function g
def g (k : ℝ) (x : ℝ) : ℝ := F x - k * x

theorem quadratic_function_theorem (a b : ℝ) (h1 : a > 0) (h2 : f a b (-1) = 0) 
  (h3 : ∀ x : ℝ, f a b x ≥ 0) :
  (∀ x : ℝ, F x = f a b x) ∧ 
  (∀ k : ℝ, (∀ x ∈ Set.Icc (-2) 2, Monotone (g k)) ↔ (k ≤ -2 ∨ k ≥ 6)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1731_173157


namespace NUMINAMATH_CALUDE_male_female_ratio_l1731_173156

theorem male_female_ratio (M F : ℝ) (h1 : M > 0) (h2 : F > 0) : 
  (1/4 * M + 3/4 * F) / (M + F) = 198 / 360 → M / F = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_male_female_ratio_l1731_173156


namespace NUMINAMATH_CALUDE_calculate_expression_l1731_173138

theorem calculate_expression : 125^2 - 50 * 125 + 25^2 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1731_173138


namespace NUMINAMATH_CALUDE_reading_difference_l1731_173172

/-- Calculates the number of pages read given a reading rate (pages per hour) and time (in minutes) -/
def pages_read (rate : ℚ) (time : ℚ) : ℚ :=
  rate * time / 60

/-- The difference in pages read between two people given their reading rates and a specific time -/
def pages_difference (rate1 : ℚ) (rate2 : ℚ) (time : ℚ) : ℚ :=
  pages_read rate1 time - pages_read rate2 time

theorem reading_difference :
  pages_difference 75 24 40 = 34 := by
  sorry

end NUMINAMATH_CALUDE_reading_difference_l1731_173172


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1731_173164

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 5 + a 7 = ∫ x in (0 : ℝ)..2, |1 - x^2|) →
  a 4 + a 6 + a 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1731_173164


namespace NUMINAMATH_CALUDE_reservoir_shortage_l1731_173184

/-- Represents a water reservoir with its capacity and current amount --/
structure Reservoir where
  capacity : ℝ
  current_amount : ℝ
  normal_level : ℝ
  h1 : current_amount = 14
  h2 : current_amount = 2 * normal_level
  h3 : current_amount = 0.7 * capacity

/-- The difference between the total capacity and the normal level is 13 million gallons --/
theorem reservoir_shortage (r : Reservoir) : r.capacity - r.normal_level = 13 := by
  sorry

#check reservoir_shortage

end NUMINAMATH_CALUDE_reservoir_shortage_l1731_173184


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l1731_173151

/-- Given a point M with polar coordinates (ρ, θ), 
    prove that its Cartesian coordinates are (x, y) --/
theorem polar_to_cartesian (ρ θ x y : Real) : 
  ρ = 2 ∧ θ = 5 * Real.pi / 6 →
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ →
  x = -Real.sqrt 3 ∧ y = 1 := by
  sorry

#check polar_to_cartesian

end NUMINAMATH_CALUDE_polar_to_cartesian_l1731_173151


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1731_173147

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | 2*x - 3 > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 3/2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1731_173147


namespace NUMINAMATH_CALUDE_round_trip_ticket_percentage_l1731_173159

/-- Given a ship's passenger statistics, calculate the percentage of round-trip ticket holders. -/
theorem round_trip_ticket_percentage
  (total_passengers : ℝ)
  (h1 : total_passengers > 0)
  (h2 : (20 : ℝ) / 100 * total_passengers = (60 : ℝ) / 100 * (round_trip_tickets : ℝ)) :
  (round_trip_tickets : ℝ) / total_passengers = (100 : ℝ) / 3 :=
by sorry

#check round_trip_ticket_percentage

end NUMINAMATH_CALUDE_round_trip_ticket_percentage_l1731_173159


namespace NUMINAMATH_CALUDE_never_equal_implies_m_range_l1731_173199

theorem never_equal_implies_m_range (m : ℝ) :
  (∀ x : ℝ, 2 * x^2 + 4 * x + m ≠ 3 * x^2 - 2 * x + 6) →
  m < -3 := by
sorry

end NUMINAMATH_CALUDE_never_equal_implies_m_range_l1731_173199


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1731_173133

theorem solve_exponential_equation :
  ∃! x : ℤ, (3 : ℝ) ^ 8 * (3 : ℝ) ^ x = 81 :=
by
  use -4
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1731_173133


namespace NUMINAMATH_CALUDE_comparison_of_powers_l1731_173191

theorem comparison_of_powers : (2^40 : ℕ) < 3^28 ∧ (31^11 : ℕ) < 17^14 := by sorry

end NUMINAMATH_CALUDE_comparison_of_powers_l1731_173191


namespace NUMINAMATH_CALUDE_s_5_value_l1731_173162

/-- s(n) is a number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ :=
  sorry

/-- The theorem states that s(5) equals 1491625 -/
theorem s_5_value : s 5 = 1491625 :=
  sorry

end NUMINAMATH_CALUDE_s_5_value_l1731_173162


namespace NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l1731_173105

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, eccentricity e, and length of real axis 2a,
    prove that the distance from the focus to the asymptote line is √3 when e = 2 and 2a = 2. -/
theorem hyperbola_focus_asymptote_distance
  (a b c : ℝ)
  (h_hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
  (h_eccentricity : c / a = 2)
  (h_real_axis : 2 * a = 2) :
  (b * c) / Real.sqrt (a^2 + b^2) = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l1731_173105


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1731_173148

theorem complex_equation_sum (x y : ℝ) : 
  (x : ℂ) + (y - 2) * Complex.I = 2 / (1 + Complex.I) → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1731_173148


namespace NUMINAMATH_CALUDE_longest_working_secretary_time_l1731_173170

/-- Proves that given three secretaries whose working times are in the ratio of 2:3:5 
    and who worked a combined total of 110 hours, the secretary who worked the longest 
    spent 55 hours on the project. -/
theorem longest_working_secretary_time (a b c : ℕ) : 
  a + b + c = 110 →
  2 * a = 3 * b →
  2 * a = 5 * c →
  c = 55 := by
  sorry

end NUMINAMATH_CALUDE_longest_working_secretary_time_l1731_173170


namespace NUMINAMATH_CALUDE_total_days_2001_to_2004_l1731_173132

def regularYearDays : ℕ := 365
def leapYearDays : ℕ := 366
def regularYearsCount : ℕ := 3
def leapYearsCount : ℕ := 1

theorem total_days_2001_to_2004 :
  regularYearDays * regularYearsCount + leapYearDays * leapYearsCount = 1461 := by
  sorry

end NUMINAMATH_CALUDE_total_days_2001_to_2004_l1731_173132


namespace NUMINAMATH_CALUDE_three_integers_exist_l1731_173168

theorem three_integers_exist : ∃ x y z : ℤ,
  (x + y) / 2 + z = 42 ∧
  (y + z) / 2 + x = 13 ∧
  (x + z) / 2 + y = 37 :=
by sorry

end NUMINAMATH_CALUDE_three_integers_exist_l1731_173168


namespace NUMINAMATH_CALUDE_marble_draw_probability_l1731_173137

/-- The probability of drawing a red marble first, a white marble second, and a blue marble third
    from a bag containing 5 red, 4 white, and 3 blue marbles, without replacement. -/
def drawProbability (redMarbles whiteMarbles blueMarbles : ℕ) : ℚ :=
  let totalMarbles := redMarbles + whiteMarbles + blueMarbles
  let firstDraw := redMarbles / totalMarbles
  let secondDraw := whiteMarbles / (totalMarbles - 1)
  let thirdDraw := blueMarbles / (totalMarbles - 2)
  firstDraw * secondDraw * thirdDraw

/-- Theorem stating that the probability of drawing red, white, then blue
    from a bag with 5 red, 4 white, and 3 blue marbles is 1/22. -/
theorem marble_draw_probability :
  drawProbability 5 4 3 = 1 / 22 := by
  sorry

end NUMINAMATH_CALUDE_marble_draw_probability_l1731_173137


namespace NUMINAMATH_CALUDE_intersection_product_range_l1731_173134

open Real

-- Define the curves and ray
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := ∃ φ, x = 2 * cos φ ∧ y = sin φ
def l (θ ρ : ℝ) (α : ℝ) : Prop := θ = α ∧ ρ > 0

-- Define the range of α
def α_range (α : ℝ) : Prop := 0 ≤ α ∧ α ≤ π/4

-- Define the polar equations
def C₁_polar (ρ θ : ℝ) : Prop := ρ = 4 * cos θ
def C₂_polar (ρ θ : ℝ) : Prop := ρ^2 = 4 / (1 + 3 * sin θ^2)

-- Define the intersection points
def M (ρ_M : ℝ) (α : ℝ) : Prop := C₁_polar ρ_M α ∧ l α ρ_M α
def N (ρ_N : ℝ) (α : ℝ) : Prop := C₂_polar ρ_N α ∧ l α ρ_N α

-- State the theorem
theorem intersection_product_range :
  ∀ α ρ_M ρ_N, α_range α → M ρ_M α → N ρ_N α → ρ_M ≠ 0 → ρ_N ≠ 0 →
  (8 * sqrt 5 / 5) ≤ ρ_M * ρ_N ∧ ρ_M * ρ_N ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_intersection_product_range_l1731_173134


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1731_173130

/-- A line mx-y+2=0 is tangent to the circle x^2+y^2=1 if and only if m = ± √3 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∃ (x y : ℝ), m*x - y + 2 = 0 ∧ x^2 + y^2 = 1 ∧ 
   ∀ (x' y' : ℝ), m*x' - y' + 2 = 0 → x'^2 + y'^2 ≥ 1) ↔ 
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1731_173130


namespace NUMINAMATH_CALUDE_point_b_representation_l1731_173177

theorem point_b_representation (a b : ℝ) : 
  a = -2 → (b - a = 3 ∨ a - b = 3) → (b = 1 ∨ b = -5) := by sorry

end NUMINAMATH_CALUDE_point_b_representation_l1731_173177


namespace NUMINAMATH_CALUDE_root_order_quadratic_equations_l1731_173154

theorem root_order_quadratic_equations (m : ℝ) (a b c d : ℝ) 
  (hm : m > 0)
  (h1 : a^2 - m*a - 1 = 0)
  (h2 : b^2 - m*b - 1 = 0)
  (h3 : c^2 + m*c - 1 = 0)
  (h4 : d^2 + m*d - 1 = 0)
  (ha : a > 0)
  (hb : b < 0)
  (hc : c > 0)
  (hd : d < 0) :
  abs a > abs c ∧ abs c > abs b ∧ abs b > abs d :=
sorry

end NUMINAMATH_CALUDE_root_order_quadratic_equations_l1731_173154


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1731_173179

theorem solution_set_of_inequality (x : ℝ) :
  (x - 50) * (60 - x) > 0 ↔ x ∈ Set.Ioo 50 60 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1731_173179


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l1731_173153

theorem least_integer_satisfying_inequality :
  ∀ y : ℤ, (3 * |y| + 6 < 24) → y ≥ -5 ∧ 
  ∃ x : ℤ, x = -5 ∧ (3 * |x| + 6 < 24) := by
  sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l1731_173153


namespace NUMINAMATH_CALUDE_sector_area_l1731_173107

/-- The area of a sector with radius 2 and central angle 2π/3 is 4π/3 -/
theorem sector_area (r : ℝ) (θ : ℝ) (area : ℝ) : 
  r = 2 → θ = 2 * π / 3 → area = (1 / 2) * r^2 * θ → area = 4 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1731_173107


namespace NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l1731_173140

/-- A random variable following a binomial distribution -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  expectation : ℝ
  variance : ℝ
  h1 : 0 < p ∧ p < 1
  h2 : expectation = n * p
  h3 : variance = n * p * (1 - p)

/-- Theorem stating that a binomial distribution with given expectation and variance has specific n and p values -/
theorem binomial_distribution_unique_parameters
  (ξ : BinomialDistribution)
  (h_expectation : ξ.expectation = 2.4)
  (h_variance : ξ.variance = 1.44) :
  ξ.n = 6 ∧ ξ.p = 0.4 :=
sorry

end NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l1731_173140


namespace NUMINAMATH_CALUDE_prob_select_green_is_101_180_l1731_173174

def container_I : ℕ × ℕ := (12, 6)
def container_II : ℕ × ℕ := (4, 6)
def container_III : ℕ × ℕ := (3, 9)

def total_balls (c : ℕ × ℕ) : ℕ := c.1 + c.2

def prob_green (c : ℕ × ℕ) : ℚ :=
  c.2 / (total_balls c)

def prob_select_green : ℚ :=
  (1/3) * (prob_green container_I) +
  (1/3) * (prob_green container_II) +
  (1/3) * (prob_green container_III)

theorem prob_select_green_is_101_180 :
  prob_select_green = 101/180 := by sorry

end NUMINAMATH_CALUDE_prob_select_green_is_101_180_l1731_173174


namespace NUMINAMATH_CALUDE_sister_glue_sticks_l1731_173109

theorem sister_glue_sticks (total : ℕ) (emily : ℕ) (sister : ℕ) : 
  total = 13 → emily = 6 → sister = total - emily → sister = 7 := by
  sorry

end NUMINAMATH_CALUDE_sister_glue_sticks_l1731_173109


namespace NUMINAMATH_CALUDE_line_slope_point_sum_l1731_173126

/-- Theorem: For a line with slope 8 passing through (-2, 4), m + b = 28 -/
theorem line_slope_point_sum (m b : ℝ) : 
  m = 8 → -- The slope is 8
  4 = 8 * (-2) + b → -- The line passes through (-2, 4)
  m + b = 28 := by sorry

end NUMINAMATH_CALUDE_line_slope_point_sum_l1731_173126


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l1731_173169

/-- Represents the lengths of external diagonals of a right regular prism -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given lengths satisfy the triangle inequality for external diagonals -/
def satisfies_triangle_inequality (d : ExternalDiagonals) : Prop :=
  d.a^2 + d.b^2 > d.c^2 ∧ 
  d.b^2 + d.c^2 > d.a^2 ∧ 
  d.a^2 + d.c^2 > d.b^2

/-- Theorem stating that {5, 6, 8} cannot be the lengths of external diagonals of a right regular prism -/
theorem invalid_external_diagonals : 
  ¬(satisfies_triangle_inequality ⟨5, 6, 8⟩) :=
by
  sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l1731_173169


namespace NUMINAMATH_CALUDE_abs_neg_2023_l1731_173108

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l1731_173108


namespace NUMINAMATH_CALUDE_smallest_palindrome_l1731_173173

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- The decimal representation of the number we're proving about. -/
def target : ℕ := 17

theorem smallest_palindrome :
  (isPalindrome target 2) ∧
  (∃ (digits : List ℕ), digits = toBase target 2 ∧ digits.length = 5) ∧
  (isPalindrome target 3) ∧
  (∃ (digits : List ℕ), digits = toBase target 3 ∧ digits.length = 3) ∧
  (∀ m : ℕ, m < target →
    ¬(isPalindrome m 2 ∧
      (∃ (digits : List ℕ), digits = toBase m 2 ∧ digits.length = 5) ∧
      isPalindrome m 3 ∧
      (∃ (digits : List ℕ), digits = toBase m 3 ∧ digits.length = 3))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_palindrome_l1731_173173


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1731_173139

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1731_173139


namespace NUMINAMATH_CALUDE_binary_representation_of_89_l1731_173158

def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem binary_representation_of_89 :
  decimal_to_binary 89 = [1, 0, 1, 1, 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_89_l1731_173158


namespace NUMINAMATH_CALUDE_collinear_sufficient_not_necessary_for_coplanar_l1731_173189

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Check if four points are coplanar -/
def coplanar (p q r s : Point3D) : Prop := sorry

/-- Theorem stating that collinearity of three out of four points is sufficient but not necessary for coplanarity -/
theorem collinear_sufficient_not_necessary_for_coplanar :
  (∀ p q r s : Point3D, (collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) → coplanar p q r s) ∧
  (∃ p q r s : Point3D, coplanar p q r s ∧ ¬collinear p q r ∧ ¬collinear p q s ∧ ¬collinear p r s ∧ ¬collinear q r s) :=
sorry

end NUMINAMATH_CALUDE_collinear_sufficient_not_necessary_for_coplanar_l1731_173189


namespace NUMINAMATH_CALUDE_triangle_height_l1731_173150

/-- Proves that a triangle with area 48 square decimeters and base 6 meters has a height of 1.6 decimeters -/
theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 48 → 
  base = 6 →
  area = (base * 10) * height / 2 →
  height = 1.6 := by sorry

end NUMINAMATH_CALUDE_triangle_height_l1731_173150


namespace NUMINAMATH_CALUDE_certain_number_power_l1731_173113

theorem certain_number_power (m : ℤ) (a : ℝ) : 
  (-2 : ℝ)^(2*m) = a^(21-m) → m = 7 → a = -2 := by sorry

end NUMINAMATH_CALUDE_certain_number_power_l1731_173113


namespace NUMINAMATH_CALUDE_remaining_statue_weight_l1731_173114

/-- Represents the weights of Hammond's statues and marble block -/
structure HammondStatues where
  initial_weight : ℝ
  first_statue : ℝ
  second_statue : ℝ
  discarded_marble : ℝ

/-- Theorem stating the weight of each remaining statue -/
theorem remaining_statue_weight (h : HammondStatues)
  (h_initial : h.initial_weight = 80)
  (h_first : h.first_statue = 10)
  (h_second : h.second_statue = 18)
  (h_discarded : h.discarded_marble = 22)
  (h_equal_remaining : ∃ x : ℝ, 
    h.initial_weight - h.discarded_marble - h.first_statue - h.second_statue = 2 * x) :
  ∃ x : ℝ, x = 15 ∧ 
    h.initial_weight - h.discarded_marble - h.first_statue - h.second_statue = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_remaining_statue_weight_l1731_173114


namespace NUMINAMATH_CALUDE_shirt_price_change_l1731_173194

theorem shirt_price_change (P : ℝ) (P_pos : P > 0) :
  P * (1 + 0.15) * (1 - 0.15) = P * 0.9775 := by
  sorry

#check shirt_price_change

end NUMINAMATH_CALUDE_shirt_price_change_l1731_173194


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l1731_173117

/-- Given a rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0),
    if the area of the rectangle is 35 square units and y > 0, then y = 7. -/
theorem rectangle_area_theorem (y : ℝ) : y > 0 → 5 * y = 35 → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l1731_173117


namespace NUMINAMATH_CALUDE_smallest_lucky_number_l1731_173195

theorem smallest_lucky_number : 
  ∃ (a b c d : ℕ+), 
    (545 = a^2 + b^2 ∧ 545 = c^2 + d^2) ∧
    (a - c = 7 ∧ d - b = 13) ∧
    (∀ (N : ℕ) (a' b' c' d' : ℕ+), 
      (N < 545 → ¬(N = a'^2 + b'^2 ∧ N = c'^2 + d'^2 ∧ a' - c' = 7 ∧ d' - b' = 13))) := by
  sorry

#check smallest_lucky_number

end NUMINAMATH_CALUDE_smallest_lucky_number_l1731_173195


namespace NUMINAMATH_CALUDE_percent_greater_than_average_l1731_173175

theorem percent_greater_than_average (M N : ℝ) (h : M > N) :
  (M - (M + N) / 2) / ((M + N) / 2) * 100 = 200 * (M - N) / (M + N) := by
  sorry

end NUMINAMATH_CALUDE_percent_greater_than_average_l1731_173175


namespace NUMINAMATH_CALUDE_difference_of_squares_l1731_173110

theorem difference_of_squares : 72^2 - 54^2 = 2268 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1731_173110


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1731_173161

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  ((a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4) ∨ (a = 3 ∧ c = 5) ∨ (b = 3 ∧ c = 5)) →
  c = Real.sqrt 41 ∨ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1731_173161


namespace NUMINAMATH_CALUDE_pentagon_square_side_ratio_l1731_173106

theorem pentagon_square_side_ratio :
  ∀ (p s : ℝ),
  p > 0 → s > 0 →
  5 * p = 20 →
  4 * s = 20 →
  p / s = 4 / 5 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_square_side_ratio_l1731_173106


namespace NUMINAMATH_CALUDE_unique_three_digit_divisible_by_seven_l1731_173171

theorem unique_three_digit_divisible_by_seven : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit number
    n % 10 = 4 ∧          -- units digit is 4
    n / 100 = 6 ∧         -- hundreds digit is 6
    n % 7 = 0 ∧           -- divisible by 7
    n = 658               -- the number is 658
  := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_divisible_by_seven_l1731_173171


namespace NUMINAMATH_CALUDE_fourth_term_is_2016_l1731_173121

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  second_term : a 2 = 606
  sum_first_four : a 1 + a 2 + a 3 + a 4 = 3834

/-- The fourth term of the arithmetic sequence is 2016 -/
theorem fourth_term_is_2016 (seq : ArithmeticSequence) : seq.a 4 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_2016_l1731_173121


namespace NUMINAMATH_CALUDE_shaded_area_concentric_circles_l1731_173185

theorem shaded_area_concentric_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 6) (h₂ : r₂ = 3) :
  let area_triangles := 4 * (1/2 * r₂ * r₂)
  let area_small_sectors := 4 * (1/4 * Real.pi * r₂^2)
  area_triangles + area_small_sectors = 18 + 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_concentric_circles_l1731_173185


namespace NUMINAMATH_CALUDE_area_inequality_l1731_173166

/-- A convex n-gon with circumscribed and inscribed circles -/
class ConvexNGon (n : ℕ) where
  /-- The area of the n-gon -/
  area : ℝ
  /-- The area of the circumscribed circle -/
  circumArea : ℝ
  /-- The area of the inscribed circle -/
  inscribedArea : ℝ
  /-- The n-gon is convex -/
  convex : Prop
  /-- The n-gon has a circumscribed circle -/
  hasCircumscribed : Prop
  /-- The n-gon has an inscribed circle -/
  hasInscribed : Prop

/-- Theorem: For a convex n-gon with circumscribed and inscribed circles,
    twice the area of the n-gon is less than the sum of the areas of the circumscribed and inscribed circles -/
theorem area_inequality {n : ℕ} (ngon : ConvexNGon n) :
  2 * ngon.area < ngon.circumArea + ngon.inscribedArea :=
sorry

end NUMINAMATH_CALUDE_area_inequality_l1731_173166
