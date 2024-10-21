import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l596_59650

noncomputable def F (n : ℕ) : ℝ :=
  (1 / Real.sqrt 5) * ((((1 + Real.sqrt 5) / 2) ^ (n - 2)) - (((1 - Real.sqrt 5) / 2) ^ (n - 2)))

noncomputable def a : ℕ → ℝ
| 0 => 1  -- Adding case for 0
| 1 => 1
| 2 => 5
| (n + 3) => (a (n + 2) * a (n + 1)) / Real.sqrt ((a (n + 2))^2 + (a (n + 1))^2 + 1)

theorem a_general_term : ∀ n : ℕ, n ≥ 1 → 
  a n = Real.sqrt (2^(F (n + 2)) * 13^(F (n + 1)) * 5^(-2 * F (n + 1)) - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l596_59650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_similarity_l596_59605

/-- A triangle represented by its three vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A part of a triangle -/
structure TrianglePart where
  vertices : List (ℝ × ℝ)

/-- Predicate to check if a list of parts forms a partition of a triangle -/
def is_partition (parts : List TrianglePart) (T : Triangle) : Prop :=
  sorry

/-- Predicate to check if two triangles are similar -/
def is_similar (T1 T2 : Triangle) : Prop :=
  sorry

/-- Predicate to check if a triangle can be formed from a list of parts -/
def can_form_from_parts (T : Triangle) (parts : List TrianglePart) : Prop :=
  sorry

/-- Theorem: Any triangle can be divided into four parts that can be reassembled into two similar triangles -/
theorem triangle_division_similarity (T : Triangle) : 
  ∃ (parts : List TrianglePart) (T1 T2 : Triangle),
    (parts.length = 4) ∧ 
    (is_partition parts T) ∧
    (is_similar T T1) ∧
    (is_similar T T2) ∧
    (can_form_from_parts T1 parts) ∧
    (can_form_from_parts T2 parts) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_similarity_l596_59605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_second_quadrant_l596_59656

theorem sin_double_angle_second_quadrant (α : Real) :
  (π / 2 < α) ∧ (α < π) →  -- α is in the second quadrant
  Real.sin (π - α) = 3 / 5 →
  Real.sin (2 * α) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_second_quadrant_l596_59656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_fold_area_ratio_l596_59685

/-- Represents a rectangular piece of paper -/
structure Paper where
  width : ℝ
  length : ℝ
  area : ℝ

/-- The ratio of the new area to the original area after folding -/
noncomputable def foldedAreaRatio (p : Paper) : ℝ :=
  1 - Real.sqrt 2 / 4

theorem paper_fold_area_ratio (p : Paper) 
  (h1 : p.length = 2 * p.width) 
  (h2 : p.area = p.width * p.length) : 
  let newArea := p.area - 2 * ((Real.sqrt 2 / 2) * (p.width / 2) / 2)
  newArea / p.area = foldedAreaRatio p := by
  sorry

#check paper_fold_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_fold_area_ratio_l596_59685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l596_59698

noncomputable def g (x : ℝ) := (3*x - 8)*(x - 4)/(x + 1)

theorem inequality_solution (x : ℝ) : 
  g x > 0 ↔ x < -1 ∨ x > 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l596_59698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l596_59636

theorem min_value_expression (s t : ℝ) :
  (s + 5 - 3 * abs (Real.cos t))^2 + (s - 2 * abs (Real.sin t))^2 ≥ 2 ∧
  ∃ s₀ t₀ : ℝ, (s₀ + 5 - 3 * abs (Real.cos t₀))^2 + (s₀ - 2 * abs (Real.sin t₀))^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l596_59636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_for_346_l596_59662

/-- Represents a number in a given base -/
def toBase (n : ℕ) (b : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (fuel : ℕ) : List ℕ :=
    if fuel = 0 then []
    else if m = 0 then []
    else (m % b) :: aux (m / b) (fuel - 1)
  aux n n

/-- Checks if a number is odd -/
def isOdd (n : ℕ) : Bool := n % 2 = 1

theorem unique_base_for_346 :
  ∃! b : ℕ, b > 1 ∧ 
    (let repr := toBase 346 b
     repr.length = 4 ∧ isOdd (repr.head!)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_for_346_l596_59662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrilateral_collinearity_l596_59632

-- Define the basic geometric elements
structure Point where
  x : ℝ
  y : ℝ

-- Define the circle
structure Circle where
  center : Point
  radius : ℝ

-- Define the quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define a line
structure Line where
  p1 : Point
  p2 : Point

-- Define the tangent points
structure TangentPoints where
  P : Point
  Q : Point
  R : Point
  S : Point

-- Helper function to determine if three points are collinear
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

-- Helper function to find the intersection of two lines
noncomputable def intersection (l1 l2 : Line) : Point :=
  { x := 0, y := 0 } -- Placeholder implementation

-- Define the main theorem
theorem tangent_quadrilateral_collinearity 
  (ABCD : Quadrilateral) 
  (O : Circle) 
  (tangents : TangentPoints) 
  (E : Point) :
  -- Conditions
  (∃ (t : ℝ), E = { x := t * ABCD.A.x + (1 - t) * ABCD.B.x, y := t * ABCD.A.y + (1 - t) * ABCD.B.y }) →
  (Line.mk ABCD.A ABCD.B).p1 = tangents.P →
  (Line.mk ABCD.B ABCD.C).p1 = tangents.Q →
  (Line.mk ABCD.C ABCD.D).p1 = tangents.R →
  (Line.mk ABCD.D ABCD.A).p1 = tangents.S →
  -- Define intersection points
  let M := intersection (Line.mk E ABCD.C) (Line.mk ABCD.A ABCD.D)
  let N := intersection (Line.mk E ABCD.D) (Line.mk ABCD.B ABCD.C)
  let K := intersection (Line.mk E ABCD.A) (Line.mk ABCD.C ABCD.D)
  -- Conclusion
  collinear M N K :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrilateral_collinearity_l596_59632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sqrt_two_third_quadrant_l596_59640

theorem tan_sqrt_two_third_quadrant (α : ℝ) : 
  Real.tan α = Real.sqrt 2 → 
  α ∈ Set.Ioo π (3 * π / 2) → 
  Real.sqrt 2 * Real.sin α + Real.cos α = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sqrt_two_third_quadrant_l596_59640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modifiedLucas_50th_term_mod_5_l596_59664

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 5
  | n + 2 => modifiedLucas (n + 1) + modifiedLucas n

theorem modifiedLucas_50th_term_mod_5 :
  modifiedLucas 49 % 5 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modifiedLucas_50th_term_mod_5_l596_59664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petr_receives_one_million_l596_59690

/-- Represents the investment and share distribution of a company --/
structure CompanyShares where
  vasiliy_investment : ℕ
  petr_investment : ℕ
  anastasia_payment : ℕ

/-- Calculates the amount received by Petr Gennadievich from the sale of shares --/
def petr_received (shares : CompanyShares) : ℕ :=
  shares.anastasia_payment - (shares.anastasia_payment * 3 * shares.vasiliy_investment / (shares.vasiliy_investment + shares.petr_investment) - shares.anastasia_payment)

/-- Theorem stating that Petr Gennadievich receives 1,000,000 rubles --/
theorem petr_receives_one_million (shares : CompanyShares) 
  (h1 : shares.vasiliy_investment = 200000)
  (h2 : shares.petr_investment = 350000)
  (h3 : shares.anastasia_payment = 1100000) :
  petr_received shares = 1000000 := by
  sorry

#eval petr_received { vasiliy_investment := 200000, petr_investment := 350000, anastasia_payment := 1100000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petr_receives_one_million_l596_59690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l596_59692

/-- The slope of a line intersecting an ellipse, given the midpoint of the intersection points -/
noncomputable def slope_of_intersecting_line (m : ℝ) : ℝ := -3 / (4 * m)

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

theorem slope_range (m : ℝ) (h_m_pos : m > 0) :
  let k := slope_of_intersecting_line m
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_on_ellipse x₁ y₁ ∧
    is_on_ellipse x₂ y₂ ∧
    (x₁ + x₂) / 2 = 1 ∧
    (y₁ + y₂) / 2 = m ∧
    k = (y₂ - y₁) / (x₂ - x₁) ∧
    k < -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l596_59692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_imply_a_value_intervals_of_increase_decrease_l596_59631

/-- The function f(x) given in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (2*a + 1) * x + 2 * Real.log x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a * x - (2*a + 1) + 2 / x

theorem parallel_tangents_imply_a_value :
  ∀ a : ℝ, (f' a 1 = f' a 3) → a = 2/3 := by
  sorry

-- Part II: Intervals of increase and decrease
theorem intervals_of_increase_decrease (a : ℝ) :
  (a ≤ 0 →
    (∀ x : ℝ, 0 < x → x < 2 → (f' a x > 0)) ∧
    (∀ x : ℝ, x > 2 → (f' a x < 0))) ∧
  (0 < a → a < 1/2 →
    (∀ x : ℝ, 0 < x → x < 2 → (f' a x > 0)) ∧
    (∀ x : ℝ, 2 < x → x < 1/a → (f' a x < 0)) ∧
    (∀ x : ℝ, x > 1/a → (f' a x > 0))) ∧
  (a = 1/2 →
    (∀ x : ℝ, x > 0 → (f' a x > 0))) ∧
  (a > 1/2 →
    (∀ x : ℝ, 0 < x → x < 1/a → (f' a x > 0)) ∧
    (∀ x : ℝ, 1/a < x → x < 2 → (f' a x < 0)) ∧
    (∀ x : ℝ, x > 2 → (f' a x > 0))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_imply_a_value_intervals_of_increase_decrease_l596_59631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_bounds_l596_59689

-- Define the variables and conditions
axiom p : ℝ
axiom q : ℝ
axiom p_positive : 0 < p
axiom q_positive : 0 < q
axiom p_prime : Nat.Prime (⌊p⌋₊)
axiom q_prime : Nat.Prime (⌊q⌋₊)
axiom q_minus_p : q - p = 29

-- Define A
noncomputable def A : ℝ := 6 * Real.log p + Real.log q

-- Theorem statement
theorem A_bounds : 3 < A ∧ A < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_bounds_l596_59689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_zero_max_a_for_positive_f_l596_59675

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - (a * x) / (x + 1)

-- Theorem for part (I)
theorem min_value_at_zero (a : ℝ) :
  (∀ x > -1, f a 0 ≤ f a x) → a = 1 := by sorry

-- Theorem for part (II)
theorem max_a_for_positive_f :
  (∀ a : ℝ, (∀ x > 0, f a x > 0) → a ≤ 1) ∧
  (∀ x > 0, f 1 x > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_zero_max_a_for_positive_f_l596_59675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_budget_theorem_l596_59668

/-- The fraction of the budget spent on rent in a restaurant -/
noncomputable def rent_fraction (total_budget : ℝ) (rent : ℝ) (food_beverage : ℝ) : ℝ :=
  rent / total_budget

/-- The fraction of the remaining budget spent on food and beverages -/
noncomputable def food_beverage_fraction (total_budget : ℝ) (rent : ℝ) (food_beverage : ℝ) : ℝ :=
  food_beverage / (total_budget - rent)

theorem restaurant_budget_theorem (total_budget : ℝ) (rent : ℝ) (food_beverage : ℝ)
  (h1 : food_beverage_fraction total_budget rent food_beverage = 1/4)
  (h2 : food_beverage / total_budget = 0.1875)
  (h3 : total_budget > 0) :
  rent_fraction total_budget rent food_beverage = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_budget_theorem_l596_59668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_as_sum_of_three_cubes_l596_59679

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def is_sum_of_three_cubes (n : ℕ) : Prop :=
  ∃ a b c : ℕ, is_cube a ∧ is_cube b ∧ is_cube c ∧ n = a + b + c

theorem smallest_cube_as_sum_of_three_cubes :
  ∀ n, n ∈ ({27, 64, 125, 216, 512} : Finset ℕ) →
    (is_cube n ∧ is_sum_of_three_cubes n) →
    (∀ m, m ∈ ({27, 64, 125, 216, 512} : Finset ℕ) → m < n → ¬(is_cube m ∧ is_sum_of_three_cubes m)) →
    n = 216 :=
by sorry

#check smallest_cube_as_sum_of_three_cubes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_as_sum_of_three_cubes_l596_59679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_g_value_l596_59697

noncomputable def f (a x : ℝ) : ℝ := x^2 - a*x + a/2

noncomputable def g (a : ℝ) : ℝ := 
  if a ≤ 2 then a/2 - a^2/4 else 1 - a/2

theorem max_g_value (a : ℝ) (ha : a > 0) :
  ∃ (m : ℝ), m = 1/4 ∧ ∀ x ∈ Set.Icc 0 1, g a ≤ m ∧ ∃ a₀ > 0, g a₀ = m := by
  sorry

#check max_g_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_g_value_l596_59697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l596_59651

theorem equation_solutions_count :
  ∃! (s : Finset (ℤ × ℤ × ℤ × ℤ)), 
    (∀ (x y z t : ℤ), (x, y, z, t) ∈ s ↔ 
      x^2 + y^2 + z^2 + t^2 = 2^2004 ∧ 
      0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ t) ∧
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l596_59651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_permutation_integer_root_l596_59606

/-- A quadratic trinomial with real coefficients -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic trinomial at a given point -/
def QuadraticTrinomial.eval (f : QuadraticTrinomial) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Check if a real number is an integer -/
def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- The set of all permutations of a quadratic trinomial's coefficients -/
def permutations (f : QuadraticTrinomial) : Set QuadraticTrinomial :=
  { g | ∃ σ : Equiv.Perm (Fin 3), 
    g.a = [f.a, f.b, f.c].get (σ 0) ∧
    g.b = [f.a, f.b, f.c].get (σ 1) ∧
    g.c = [f.a, f.b, f.c].get (σ 2) }

/-- The main theorem -/
theorem quadratic_permutation_integer_root (f : QuadraticTrinomial) : 
  (∀ g ∈ permutations f, ∃ x : ℝ, is_integer x ∧ g.eval x = 0) → 
  f.eval 1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_permutation_integer_root_l596_59606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andy_carlos_tie_first_l596_59686

/-- Represents the area of a lawn -/
structure LawnArea where
  value : ℝ
  positive : value > 0

/-- Represents the mowing speed in square meters per hour -/
structure MowingSpeed where
  value : ℝ
  positive : value > 0

/-- Represents a person's lawn and mowing speed -/
structure Person where
  lawn_area : LawnArea
  mowing_speed : MowingSpeed

noncomputable def mowing_time (p : Person) : ℝ :=
  p.lawn_area.value / p.mowing_speed.value

theorem andy_carlos_tie_first (beth : Person)
  (andy_lawn_area : LawnArea)
  (carlos_lawn_area : LawnArea)
  (andy_mowing_speed : MowingSpeed)
  (carlos_mowing_speed : MowingSpeed)
  (h1 : andy_lawn_area.value = 3 * beth.lawn_area.value)
  (h2 : andy_lawn_area.value = 4 * carlos_lawn_area.value)
  (h3 : carlos_mowing_speed.value = 1/3 * beth.mowing_speed.value)
  (h4 : carlos_mowing_speed.value = 1/4 * andy_mowing_speed.value)
  (h5 : beth.mowing_speed.value = 90) :
  let andy : Person := ⟨andy_lawn_area, andy_mowing_speed⟩
  let carlos : Person := ⟨carlos_lawn_area, carlos_mowing_speed⟩
  mowing_time andy = mowing_time carlos ∧ mowing_time andy < mowing_time beth :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andy_carlos_tie_first_l596_59686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l596_59608

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a + 2

/-- The function g(x) defined in the problem -/
def g (a : ℝ) (x : ℝ) : ℝ := f a x + |x^2 - 1|

/-- The maximum value of f(sin x) for x ∈ ℝ -/
noncomputable def M (a : ℝ) : ℝ := 
  if a ≥ 0 then 2*a^2 + 3*a + 3 else 2*a^2 - a + 3

theorem problem_solution :
  (∀ a : ℝ, ∀ x : ℝ, f a (Real.sin x) ≤ M a) ∧
  (∀ a : ℝ, (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 ∧
    g a x₁ = 0 ∧ g a x₂ = 0 ∧
    (∀ x : ℝ, 0 < x ∧ x < 3 ∧ g a x = 0 → x = x₁ ∨ x = x₂)) →
    1 + Real.sqrt 3 < a ∧ a < 19/5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l596_59608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_B_l596_59678

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The specific triangle in the problem -/
def problemTriangle : Triangle :=
  { A := { x := 0, y := 0 }
  , B := { x := 12, y := 0 }
  , C := { x := 8, y := 10 }
  }

/-- Calculates the area of a triangle given its vertices -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  (1/2) * abs ((t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y))

/-- Calculates the area of the region closer to B than to A or C -/
noncomputable def areaCloserToB (t : Triangle) : ℝ := sorry

/-- The theorem to be proved -/
theorem probability_closer_to_B (t : Triangle) : 
  t = problemTriangle → areaCloserToB t / triangleArea t = 109 / 300 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_B_l596_59678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lion_consumption_l596_59681

/-- Represents the daily meat consumption scenario at San Diego Zoo -/
structure ZooMeatConsumption where
  lionDaily : ℝ
  tigerDaily : ℝ
  totalMeat : ℝ
  durationDays : ℝ

/-- The specific scenario at San Diego Zoo -/
def sanDiegoZoo : ZooMeatConsumption where
  lionDaily := 25  -- We set this to 25 as it's what we want to prove
  tigerDaily := 20
  totalMeat := 90
  durationDays := 2

theorem lion_consumption (zoo : ZooMeatConsumption) (h1 : zoo = sanDiegoZoo) :
  zoo.lionDaily = 25 := by
  rw [h1]
  rfl

#check lion_consumption

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lion_consumption_l596_59681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l596_59602

/-- Hyperbola with given parameters -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b

/-- Circle intersecting hyperbola's asymptote -/
structure IntersectingCircle (h : Hyperbola) where
  c : ℝ  -- x-coordinate of the right focus
  pq_length : ℝ
  h_pq_ge_2b : pq_length ≥ 2 * h.b

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) (ic : IntersectingCircle h) : ℝ :=
  ic.c / h.a

/-- Main theorem statement -/
theorem eccentricity_range (h : Hyperbola) (ic : IntersectingCircle h) :
  1 < eccentricity h ic ∧ eccentricity h ic ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l596_59602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_automobile_distance_theorem_l596_59655

/-- Calculates the distance traveled by an automobile with uniform acceleration -/
noncomputable def distanceTraveled (initialVelocity : ℝ) (acceleration : ℝ) (time : ℝ) : ℝ :=
  initialVelocity * time + (1/2) * acceleration * time^2

theorem automobile_distance_theorem (a b : ℝ) :
  distanceTraveled (a/4) b 300 = 75*a + 45000*b := by
  -- Unfold the definition of distanceTraveled
  unfold distanceTraveled
  -- Simplify the expression
  simp [mul_add, add_mul, mul_assoc]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_automobile_distance_theorem_l596_59655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l596_59693

/-- Focal length of a hyperbola -/
noncomputable def focal_length (C : Set (ℝ × ℝ)) : ℝ := sorry

/-- A hyperbola C defined by x²/m - y² = 1 with m > 0 and an asymptote √3x + my = 0 has focal length 4 -/
theorem hyperbola_focal_length (m : ℝ) (h1 : m > 0) : 
  let C := {(x, y) : ℝ × ℝ | x^2 / m - y^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | Real.sqrt 3 * x + m * y = 0}
  focal_length C = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l596_59693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folding_area_specific_l596_59683

open Real

/-- The area of the region where point P can be located on a rectangle ABCD 
    such that the folding creases don't intersect within the rectangle. -/
noncomputable def folding_area (ab bc : ℝ) : ℝ :=
  (π / 3 * ab^2 - 1 / 2 * ab^2 * sin (2 * π / 3)) +
  (π / 6 * bc^2 - 1 / 2 * bc^2 * sin (π / 3))

/-- The theorem stating the area of the region for a specific rectangle. -/
theorem folding_area_specific : 
  folding_area 48 96 = 1152 * π - 1152 * sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folding_area_specific_l596_59683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_isomorphism_properties_l596_59614

/-- The degree of a vertex in a simple graph. -/
def degree (G : SimpleGraph V) (v : V) : ℕ := sorry

/-- Two graphs are isomorphic. -/
def isomorphic (G₁ : SimpleGraph V) (G₂ : SimpleGraph W) : Prop := sorry

/-- A graph is connected. -/
def connected (G : SimpleGraph V) : Prop := sorry

/-- A graph is acyclic. -/
def acyclic (G : SimpleGraph V) : Prop := sorry

/-- The number of edges in a graph. -/
def edgeCount (G : SimpleGraph V) : ℕ := sorry

theorem graph_isomorphism_properties :
  (∀ (G₁ G₂ : SimpleGraph (Fin 10)), 
    (∀ v, degree G₁ v = 9) → 
    (∀ v, degree G₂ v = 9) → 
    isomorphic G₁ G₂) ∧
  (∃ (G₁ G₂ : SimpleGraph (Fin 8)), 
    (∀ v, degree G₁ v = 3) ∧ 
    (∀ v, degree G₂ v = 3) ∧ 
    ¬isomorphic G₁ G₂) ∧
  (∃ (G₁ G₂ : SimpleGraph V), 
    connected G₁ ∧ 
    connected G₂ ∧ 
    acyclic G₁ ∧ 
    acyclic G₂ ∧ 
    edgeCount G₁ = 6 ∧ 
    edgeCount G₂ = 6 ∧ 
    ¬isomorphic G₁ G₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_isomorphism_properties_l596_59614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l596_59652

/-- Given that log₂ₓ 216 = x where x is a real number, prove that x = 3 and is a non-square, non-cube integer -/
theorem log_equation_solution (x : ℝ) (h : Real.log 216 / Real.log (2 * x) = x) :
  x = 3 ∧ (∃ n : ℤ, x = n) ∧ ¬ ∃ (n : ℤ), x = n^2 ∧ ¬ ∃ (n : ℤ), x = n^3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l596_59652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l596_59616

theorem tan_double_angle (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.cos x = 4/5) : 
  Real.tan (2 * x) = -24/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l596_59616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l596_59603

/-- The function f(x) = 2sin²(x) - √3sin(2x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x ^ 2 - Real.sqrt 3 * Real.sin (2 * x)

/-- The minimum value of f(x) is -1 -/
theorem min_value_of_f : 
  ∀ x : ℝ, f x ≥ -1 ∧ ∃ y : ℝ, f y = -1 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l596_59603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_calculation_l596_59620

/-- Calculates the speed of the slower train given the lengths of two trains,
    the time they take to cross each other, and the speed of the faster train. -/
noncomputable def slower_train_speed (length1 length2 : ℝ) (crossing_time : ℝ) (faster_speed : ℝ) : ℝ :=
  let total_length := length1 + length2
  let relative_speed := total_length / crossing_time
  let relative_speed_kmh := relative_speed * 3.6
  relative_speed_kmh - faster_speed

/-- Theorem stating that under the given conditions, the speed of the slower train
    is approximately 39.9788 km/hr. -/
theorem slower_train_speed_calculation :
  let length1 : ℝ := 140  -- meters
  let length2 : ℝ := 210  -- meters
  let crossing_time : ℝ := 12.59899208063355  -- seconds
  let faster_speed : ℝ := 60  -- km/hr
  let result := slower_train_speed length1 length2 crossing_time faster_speed
  ∀ ε > 0, |result - 39.9788| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_calculation_l596_59620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l596_59699

-- Define new operations
noncomputable def newAdd (a b : ℝ) : ℝ := a * b
noncomputable def newSub (a b : ℝ) : ℝ := a + b
noncomputable def newMul (a b : ℝ) : ℝ := a / b
noncomputable def newDiv (a b : ℝ) : ℝ := a - b

-- Define the equation
def equation (x : ℝ) : Prop :=
  newSub (newSub (newAdd 6 9) (newMul 8 3)) x = 5

-- Theorem statement
theorem solution_exists : ∃ x : ℝ, equation x ∧ x = 25 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l596_59699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_bound_l596_59611

noncomputable def sequence_x (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => Real.sqrt (sequence_x x₀ n + 1)

theorem convergence_bound (x₀ : ℝ) (h₀ : x₀ > 0) :
  ∃ (A C : ℝ), A > 1 ∧ C > 0 ∧
    ∀ n : ℕ, |sequence_x x₀ n - A| < C / A ^ n :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#check convergence_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_bound_l596_59611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l596_59670

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (2 - x) + Real.sqrt (x + 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-1) 2 ∩ Set.Iio 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l596_59670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_outing_equation_l596_59672

theorem spring_outing_equation (x : ℝ) (h1 : x > 0) :
  15 / (1.2 * x) = 15 / x - 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_outing_equation_l596_59672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_difference_l596_59694

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the right focus F
def right_focus (F : ℝ × ℝ) : Prop := F.1 = 1 ∧ F.2 = 0

-- Define point A
def point_A : ℝ × ℝ := (2, 4)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem min_distance_difference (P : ℝ × ℝ) (F : ℝ × ℝ) :
  ellipse_C P.1 P.2 →
  right_focus F →
  ∃ (min_val : ℝ), min_val = 1 ∧
    ∀ (Q : ℝ × ℝ), ellipse_C Q.1 Q.2 →
      distance Q point_A - distance Q F ≥ min_val := by
  sorry

#check min_distance_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_difference_l596_59694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_l596_59663

-- Define the table radius
def table_radius : ℝ := 5

-- Define the number of place mats
def num_mats : ℕ := 8

-- Define the width of each place mat
def mat_width : ℝ := 1

-- Define the function to calculate the length of each place mat
noncomputable def mat_length (r : ℝ) (n : ℕ) (w : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / (2 * n : ℝ))
  (r + (s^2 - (w/2)^2).sqrt) - r

-- Theorem stating the length of each place mat
theorem place_mat_length :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |mat_length table_radius num_mats mat_width - 3.82| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_l596_59663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_property_l596_59639

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 4

-- State the theorem
theorem zero_property (x x₁ x₂ : ℝ) : 
  f x = 0 → 
  -1 < x₁ → x₁ < x → 
  x < x₂ → x₂ < 2 → 
  f x₁ < 0 ∧ f x₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_property_l596_59639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ab_equation_l596_59619

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define a point on the circle
def point_on_my_circle (p : ℝ × ℝ) : Prop := my_circle p.1 p.2

-- Define the midpoint of a line segment
def my_midpoint (p q m : ℝ × ℝ) : Prop :=
  m.1 = (p.1 + q.1) / 2 ∧ m.2 = (p.2 + q.2) / 2

-- Define a line equation
def line_equation (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 + c = 0

-- Theorem statement
theorem line_ab_equation (A B : ℝ × ℝ) :
  point_on_my_circle A ∧ point_on_my_circle B ∧ my_midpoint A B (1, 1) →
  line_equation 1 1 (-2) A ∧ line_equation 1 1 (-2) B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ab_equation_l596_59619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_as_power_of_x_specific_case_l596_59648

-- Define variables
variable (x t m n p q r : ℝ)
variable (a b c : ℝ)

-- Define conditions
axiom def_a : a = x^t
axiom def_b : b = x^m
axiom def_c : c = x^n

-- Define the theorem
theorem fraction_as_power_of_x :
  (a^p * b^q) / c^r = x^(t*p + m*q - n*r) :=
by sorry

-- Define the specific case
theorem specific_case (h : x = 3) (h2 : (a^p * b^q) / c^r = Real.sqrt 243) :
  t*p + m*q - n*r = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_as_power_of_x_specific_case_l596_59648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_at_zero_range_of_a_l596_59621

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

-- Part 1: Tangent line at x = 0 when a = 1
theorem tangent_line_slope_at_zero :
  (deriv (f 1)) 0 = 2 := by sorry

-- Part 2: Condition for exactly one zero in each interval
def has_unique_zero_in_intervals (a : ℝ) : Prop :=
  (∃! x₁, x₁ ∈ Set.Ioo (-1 : ℝ) 0 ∧ f a x₁ = 0) ∧
  (∃! x₂, x₂ ∈ Set.Ioi 0 ∧ f a x₂ = 0)

-- Theorem stating the range of a
theorem range_of_a :
  ∀ a : ℝ, has_unique_zero_in_intervals a ↔ a < -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_at_zero_range_of_a_l596_59621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l596_59646

/-- Given an ellipse x^2 + 2y^2 = 3 and a line with slope k that passes through a focus of the ellipse
    and intersects the ellipse at points A and B such that AB = 2, prove that |k| = √(1 + √3) -/
theorem ellipse_line_intersection (k : ℝ) : 
  let ellipse := λ (x y : ℝ) => x^2 + 2*y^2 = 3
  ∃ (A B : ℝ × ℝ), 
    (ellipse A.1 A.2) ∧ 
    (ellipse B.1 B.2) ∧ 
    (∃ (focus : ℝ × ℝ), ellipse focus.1 focus.2 ∧ 
      (B.2 - A.2) = k * (B.1 - A.1) ∧ 
      (focus.2 - A.2) = k * (focus.1 - A.1)) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 4) →
  |k| = Real.sqrt (1 + Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l596_59646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_squared_identity_l596_59665

theorem cosine_squared_identity (c d : ℝ) :
  (∀ θ : ℝ, (Real.cos θ)^2 = c * Real.cos (2 * θ) + d * Real.cos θ) →
  c = 1/2 ∧ d = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_squared_identity_l596_59665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_loss_percentage_l596_59626

noncomputable def loss_percentage (cost_price selling_price : ℝ) : ℝ :=
  (cost_price - selling_price) / cost_price * 100

noncomputable def selling_price_with_loss (cost_price : ℝ) (loss_percent : ℝ) : ℝ :=
  cost_price * (1 - loss_percent / 100)

noncomputable def selling_price_with_gain (cost_price : ℝ) (gain_percent : ℝ) : ℝ :=
  cost_price * (1 + gain_percent / 100)

theorem watch_loss_percentage (cost_price : ℝ) (loss_percent : ℝ) 
  (h1 : cost_price = 350)
  (h2 : selling_price_with_gain cost_price 4 = selling_price_with_loss cost_price loss_percent + 140) :
  loss_percent = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_loss_percentage_l596_59626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_investment_amount_l596_59613

/-- The amount invested in a stock given its rate, income, brokerage, and market value --/
noncomputable def amount_invested (rate : ℝ) (income : ℝ) (brokerage : ℝ) (market_value : ℝ) : ℝ :=
  let face_value := (income * 100) / rate
  let actual_market_value := market_value + (market_value * brokerage) / 100
  (face_value / 100) * actual_market_value

/-- Theorem stating the amount invested in the stock --/
theorem stock_investment_amount :
  let rate := 10.5
  let income := 756
  let brokerage := 0.25
  let market_value := 90.02777777777779
  ∃ (x : ℝ), abs (amount_invested rate income brokerage market_value - x) < 0.001 ∧ x = 6498.205 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_investment_amount_l596_59613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_packages_required_l596_59624

/-- Represents a range of apartment numbers -/
structure ApartmentRange where
  start : Nat
  finish : Nat

/-- Counts the occurrences of a specific digit in a range of numbers -/
def countDigitOccurrences (digit : Nat) (range : ApartmentRange) : Nat :=
  sorry

/-- Calculates the maximum digit count for a given range -/
def maxDigitCount (range : ApartmentRange) : Nat :=
  sorry

/-- The main theorem proving the minimum number of packages required -/
theorem min_packages_required (floor1 floor2 : ApartmentRange) 
    (h1 : floor1.start = 107 ∧ floor1.finish = 132)
    (h2 : floor2.start = 207 ∧ floor2.finish = 232) :
    max (maxDigitCount floor1) (maxDigitCount floor2) = 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_packages_required_l596_59624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_towels_is_975_l596_59696

/-- Represents a building in the hotel -/
structure Building where
  rooms : Nat
  familySize : Nat
  towelsPerPerson : Nat

/-- Calculates the total number of towels for a building -/
def towelsForBuilding (b : Building) : Nat :=
  b.rooms * b.familySize * b.towelsPerPerson

/-- The hotel with three buildings -/
def hotel : List Building :=
  [{ rooms := 25, familySize := 5, towelsPerPerson := 3 },
   { rooms := 30, familySize := 6, towelsPerPerson := 2 },
   { rooms := 15, familySize := 4, towelsPerPerson := 4 }]

/-- Theorem: The total number of towels handed out across all three buildings is 975 -/
theorem total_towels_is_975 : (hotel.map towelsForBuilding).sum = 975 := by
  sorry

#eval (hotel.map towelsForBuilding).sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_towels_is_975_l596_59696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l596_59691

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin (2 * x) - Real.cos x ^ 2 - 1/2

noncomputable def α : ℝ := Real.arctan (2 * Real.sqrt 3)

theorem function_properties (m : ℝ) :
  (f m α = -3/26) →
  (m = Real.sqrt 3 / 2) ∧
  (∀ x : ℝ, f m x = f m (x + Real.pi)) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi/3) ∨ x ∈ Set.Icc (5*Real.pi/6) Real.pi →
    ∀ y : ℝ, y ∈ Set.Icc 0 Real.pi → x < y → f m x < f m y) :=
by sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l596_59691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_problem_correct_system_l596_59623

/-- Represents the capacity of a bucket in dou -/
noncomputable def bucket_capacity : ℝ := 10

/-- Represents the total amount of rice obtained after threshing -/
noncomputable def total_rice : ℝ := 7

/-- Represents the rice production rate -/
noncomputable def rice_production_rate : ℝ := 3/5

/-- Represents the system of equations describing the rice problem -/
def rice_system (x y : ℝ) : Prop :=
  x + y = bucket_capacity ∧ x + rice_production_rate * y = total_rice

/-- Theorem stating that the given system of equations correctly describes the rice problem -/
theorem rice_problem_correct_system : 
  ∃ x y : ℝ, rice_system x y := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_problem_correct_system_l596_59623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_sb_is_rotation_axis_l596_59676

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

instance : Add Point3D where
  add a b := ⟨a.x + b.x, a.y + b.y, a.z + b.z⟩

instance : Sub Point3D where
  sub a b := ⟨a.x - b.x, a.y - b.y, a.z - b.z⟩

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a plane in 3D space -/
structure Plane3D where
  normal : Point3D
  point : Point3D

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  S : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Represents an axial symmetry -/
noncomputable def AxialSymmetry (axis : Line3D) : Point3D → Point3D := sorry

/-- Represents the composition of three axial symmetries -/
noncomputable def CompositionOfThreeAxialSymmetries (s1 s2 s3 : Point3D → Point3D) : Point3D → Point3D := sorry

/-- Theorem: The edge SB is the rotation axis of the composition of three axial symmetries -/
theorem edge_sb_is_rotation_axis (T : TrihedralAngle) 
  (SA' SB' SC' : Line3D) 
  (h1 : SA' = Line3D.mk T.S (T.A - T.S)) 
  (h2 : SB' = Line3D.mk T.S (T.B - T.S)) 
  (h3 : SC' = Line3D.mk T.S (T.C - T.S)) 
  (h4 : SA' = Line3D.mk T.S ((T.B - T.S) + (T.C - T.S)))
  (h5 : SB' = Line3D.mk T.S ((T.C - T.S) + (T.A - T.S)))
  (h6 : SC' = Line3D.mk T.S ((T.A - T.S) + (T.B - T.S)))
  : ∃ (rotation : Point3D → Point3D), 
    rotation = CompositionOfThreeAxialSymmetries (AxialSymmetry SA') (AxialSymmetry SB') (AxialSymmetry SC') ∧
    ∀ p, rotation p = AxialSymmetry (Line3D.mk T.S (T.B - T.S)) p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_sb_is_rotation_axis_l596_59676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ap_eight_terms_l596_59647

/-- An arithmetic progression with an even number of terms -/
structure EvenAP where
  n : ℕ
  a : ℚ
  d : ℚ
  even_terms : Even n

/-- The sum of the first k terms of an arithmetic progression -/
def sum_k_terms (ap : EvenAP) (k : ℕ) : ℚ :=
  k * (2 * ap.a + (k - 1) * ap.d) / 2

theorem ap_eight_terms (ap : EvenAP) 
  (sum_odd : sum_k_terms ap (ap.n / 2) = 30)
  (sum_even : sum_k_terms { ap with a := ap.a + ap.d } (ap.n / 2) = 36)
  (last_first_diff : ap.a + (ap.n - 1) * ap.d - ap.a = 12) :
  ap.n = 8 := by
  sorry

#check ap_eight_terms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ap_eight_terms_l596_59647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_range_of_f_when_odd_l596_59644

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Theorem 1: If f is odd, then a = 1
theorem odd_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) → a = 1 := by sorry

-- Theorem 2: If a = 1, then the range of f is (-1, 1)
theorem range_of_f_when_odd :
  Set.range (f 1) = Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_range_of_f_when_odd_l596_59644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l596_59671

-- Define the interval (1/e, 1)
def interval (x : ℝ) : Prop := 1 / Real.exp 1 < x ∧ x < 1

-- Define a, b, and c
noncomputable def a (x : ℝ) : ℝ := Real.log x
noncomputable def b (x : ℝ) : ℝ := Real.exp (Real.log x)
noncomputable def c (x : ℝ) : ℝ := Real.exp (Real.log (1 / x))

-- Theorem statement
theorem relationship_abc {x : ℝ} (h : interval x) : a x < b x ∧ b x < c x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l596_59671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_discount_theorem_l596_59633

/-- Represents a bill with its amount and due time --/
structure Bill where
  amount : ℝ
  dueTime : ℝ

/-- Calculates the banker's discount for a given bill and discount rate --/
noncomputable def bankerDiscount (bill : Bill) (discountRate : ℝ) : ℝ :=
  (bill.amount * discountRate * bill.dueTime) / 12

/-- Calculates the effective discount rate for a given bill and banker's discount --/
noncomputable def effectiveDiscountRate (bill : Bill) (bankersDiscount : ℝ) : ℝ :=
  (bankersDiscount / bill.amount) * 100

theorem bill_discount_theorem (bill : Bill) (trueDiscountRate : ℝ) (bankDiscountRate : ℝ) :
  bill.amount = 12800 ∧
  bill.dueTime = 0.5 ∧
  trueDiscountRate = 5/8 * 4.5/100 ∧
  bankDiscountRate = 7.5/100 →
  bankerDiscount bill bankDiscountRate = 480 ∧
  effectiveDiscountRate bill (bankerDiscount bill bankDiscountRate) = 3.75 := by
  sorry

#eval "Bill discount theorem defined"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_discount_theorem_l596_59633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l596_59629

-- Define the base-2 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Statement 1
theorem simplify_expression_1 :
  (2 + 4/5 : ℝ)^(0 : ℝ) + 2^(-2 : ℝ) * (2 + 1/4 : ℝ)^(-1/2 : ℝ) - (8/27 : ℝ)^(1/3 : ℝ) = 1/2 := by
  sorry

-- Statement 2
theorem simplify_expression_2 :
  2 * (lg (Real.sqrt 2))^2 + lg (Real.sqrt 2) * lg 5 + Real.sqrt ((lg (Real.sqrt 2))^2 - lg 2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l596_59629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l596_59645

theorem relationship_abc : 
  let a : ℝ := (0.6 : ℝ)^2
  let b : ℝ := Real.log 0.6 / Real.log 2
  let c : ℝ := (2 : ℝ)^(0.6 : ℝ)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l596_59645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_radius_l596_59643

-- Define the circles and their properties
noncomputable def circle_small (r₁ : Real) : Real := Real.pi * r₁^2
noncomputable def circle_large (r₂ : Real) : Real := Real.pi * r₂^2

-- State the theorem
theorem smaller_circle_radius (A₁ A₂ : Real) : 
  -- The larger circle has radius 3
  circle_large 3 = A₁ + A₂ →
  -- A₁, A₂, A₁ + A₂ form an arithmetic progression
  A₂ = (A₁ + (A₁ + A₂)) / 2 →
  -- The radius of the smaller circle is √3
  ∃ r₁, circle_small r₁ = A₁ ∧ r₁ = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_radius_l596_59643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinates_l596_59673

/-- Triangle area formula -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Theorem: The third vertex of the triangle is at (-24, 0) -/
theorem third_vertex_coordinates (x : ℝ) (h1 : x < 0) 
  (h2 : triangleArea 3 2 0 0 x 0 = 24) : x = -24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinates_l596_59673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l596_59601

/-- Represents a parabola with equation y² = 5x -/
structure Parabola where
  equation : ∀ x y : ℝ, y^2 = 5*x

/-- The distance from the focus to the directrix of the parabola -/
noncomputable def focus_directrix_distance (p : Parabola) : ℝ := 5/2

/-- Theorem stating that the distance from the focus to the directrix of the parabola y² = 5x is 5/2 -/
theorem parabola_focus_directrix_distance (p : Parabola) : 
  focus_directrix_distance p = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l596_59601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printing_presses_count_is_40_printing_press_rate_equality_l596_59667

def printing_presses_count : ℕ :=
  let papers : ℕ := 500000
  let hours_scenario1 : ℝ := 12
  let presses_scenario2 : ℕ := 30
  let hours_scenario2 : ℝ := 15.999999999999998
  let presses_scenario1 : ℕ := 40

  presses_scenario1

theorem printing_presses_count_is_40 : printing_presses_count = 40 := by
  unfold printing_presses_count
  rfl

theorem printing_press_rate_equality : 
  let papers : ℕ := 500000
  let hours_scenario1 : ℝ := 12
  let presses_scenario2 : ℕ := 30
  let hours_scenario2 : ℝ := 15.999999999999998
  let presses_scenario1 : ℕ := 40
  (presses_scenario1 : ℝ) * (papers : ℝ) / hours_scenario1 = 
  (presses_scenario2 : ℝ) * (papers : ℝ) / hours_scenario2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_printing_presses_count_is_40_printing_press_rate_equality_l596_59667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_large_number_l596_59695

/-- Helper function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The sum of the digits of 10^2008 - 2008 is 18063 -/
theorem sum_of_digits_of_large_number : ∃ (n : ℕ), n = 10^2008 - 2008 ∧ (sum_of_digits n = 18063) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_large_number_l596_59695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l596_59661

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -Real.exp x else x + 3/x - 5

-- Theorem statement
theorem min_value_of_f :
  ∃ (m : ℝ), m = -Real.exp 1 ∧ ∀ (x : ℝ), f x ≥ m :=
by
  -- We'll use -e as our minimum value
  let m := -Real.exp 1
  
  -- Prove that m satisfies the conditions
  use m
  constructor
  
  -- First condition: m = -e
  · rfl
  
  -- Second condition: ∀ (x : ℝ), f x ≥ m
  · intro x
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l596_59661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_third_minus_alpha_l596_59612

theorem sin_pi_third_minus_alpha (α : Real) :
  0 < α → α < π / 2 → Real.cos (α + π / 6) = 1 / 3 → Real.sin (π / 3 - α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_third_minus_alpha_l596_59612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l596_59666

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

theorem arithmetic_sequence_common_difference 
  (a₁ : ℝ) (h₁ : a₁ = 4) 
  (S₃ : ℝ) (h₂ : S₃ = 6) :
  ∃ d : ℝ, d = -2 ∧ arithmetic_sum a₁ d 3 = S₃ := by
  sorry

#check arithmetic_sequence_common_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l596_59666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_5_5_l596_59618

/-- A right triangle with sides 5, 12, and 13 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_eq : a = 5
  b_eq : b = 12
  c_eq : c = 13
  right_angle : a^2 + b^2 = c^2

/-- The length of the crease when the triangle is folded so A aligns with C -/
noncomputable def crease_length (t : RightTriangle) : ℝ :=
  (t.a + t.b) / 2 - t.c / 2

theorem crease_length_is_5_5 (t : RightTriangle) : 
  crease_length t = 5.5 := by
  sorry

#eval show String from "The theorem has been stated and the proof is left as an exercise."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_5_5_l596_59618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_star_30_equals_5_l596_59657

noncomputable def star (a b : ℝ) : ℝ := (Real.sqrt (a^2 + b)) / (Real.sqrt (a - b))

theorem x_star_30_equals_5 (x : ℝ) (h1 : x > 30) (h2 : star x 30 = 5) :
  x = (25 + Real.sqrt 2495) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_star_30_equals_5_l596_59657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l596_59669

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition that a*cos(C) + c/2 = b -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a * Real.cos t.C + t.c / 2 = t.b

/-- Predicate to check if a real number is the inradius of a triangle -/
def is_inradius (r : ℝ) (t : Triangle) : Prop :=
  2 * r * (t.a + t.b + t.c) = t.a * t.b * Real.sin t.C

/-- The theorem stating the measure of angle A and the maximum inradius -/
theorem triangle_properties (t : Triangle) (h : satisfiesCondition t) :
  t.A = π / 3 ∧ 
  (∀ (r : ℝ), t.a = 1 → is_inradius r t → r ≤ Real.sqrt 3 / 6) ∧
  (∃ (t' : Triangle), t'.a = 1 ∧ satisfiesCondition t' ∧ 
    (∃ (r : ℝ), is_inradius r t' ∧ r = Real.sqrt 3 / 6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l596_59669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_multiplication_l596_59682

theorem fraction_power_multiplication :
  (3 / 5 : ℚ) ^ 10 * (2 / 3 : ℚ) ^ (-4 : ℤ) = 4782969 / 156250000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_multiplication_l596_59682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_zero_l596_59635

open Real

theorem cosine_sum_zero (x y z : ℝ) 
  (h1 : Real.cos (x + π/4) + Real.cos (y + π/4) + Real.cos (z + π/4) = 0)
  (h2 : Real.sin (x + π/4) + Real.sin (y + π/4) + Real.sin (z + π/4) = 0) :
  Real.cos (2*x) + Real.cos (2*y) + Real.cos (2*z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_zero_l596_59635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_structures_count_l596_59688

-- Define the teams
inductive Team : Type
| A | B | C | D | E | F

-- Define a match result
structure MatchResult :=
  (winner : Team)
  (loser : Team)

-- Define the tournament structure
structure TournamentStructure :=
  (friday_match : MatchResult)
  (saturday_match1 : MatchResult)
  (saturday_match2 : MatchResult)
  (sunday_match1 : MatchResult)
  (sunday_match2 : MatchResult)

-- Define a valid tournament structure
def is_valid_tournament (t : TournamentStructure) : Prop :=
  (t.friday_match.winner = Team.E ∨ t.friday_match.winner = Team.F) ∧
  (t.friday_match.loser = Team.E ∨ t.friday_match.loser = Team.F) ∧
  (t.friday_match.winner ≠ t.friday_match.loser) ∧
  (t.saturday_match1.winner = Team.A ∨ t.saturday_match1.winner = Team.B) ∧
  (t.saturday_match1.loser = Team.A ∨ t.saturday_match1.loser = Team.B) ∧
  (t.saturday_match1.winner ≠ t.saturday_match1.loser) ∧
  (t.saturday_match2.winner = Team.C ∨ t.saturday_match2.winner = t.friday_match.winner) ∧
  (t.saturday_match2.loser = Team.C ∨ t.saturday_match2.loser = t.friday_match.winner) ∧
  (t.saturday_match2.winner ≠ t.saturday_match2.loser) ∧
  (t.sunday_match1.winner = t.saturday_match1.winner ∨ t.sunday_match1.winner = t.saturday_match2.winner) ∧
  (t.sunday_match1.loser = t.saturday_match1.winner ∨ t.sunday_match1.loser = t.saturday_match2.winner) ∧
  (t.sunday_match1.winner ≠ t.sunday_match1.loser) ∧
  (t.sunday_match2.winner = t.saturday_match1.loser ∨ t.sunday_match2.winner = t.saturday_match2.loser) ∧
  (t.sunday_match2.loser = t.saturday_match1.loser ∨ t.sunday_match2.loser = t.saturday_match2.loser) ∧
  (t.sunday_match2.winner ≠ t.sunday_match2.loser)

-- Theorem: The number of valid tournament structures is 16
theorem tournament_structures_count :
  ∃ (s : Finset TournamentStructure), s.card = 16 ∧ ∀ t ∈ s, is_valid_tournament t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_structures_count_l596_59688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l596_59680

theorem calculation_proof : Real.sqrt 4 - 2 * Real.sin (45 * π / 180) + (1/3)⁻¹ + abs (-Real.sqrt 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l596_59680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_pow_2018_l596_59627

noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![Real.sqrt 3 / 2, 0, -1 / 2],
    ![0, -1, 0],
    ![1 / 2, 0, Real.sqrt 3 / 2]]

theorem A_pow_2018 :
  A ^ 2018 = ![![1 / 2, 0, -Real.sqrt 3 / 2],
               ![0, 1, 0],
               ![Real.sqrt 3 / 2, 0, 1 / 2]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_pow_2018_l596_59627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_point_of_f_l596_59625

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2 - 2*x + 3

theorem max_point_of_f :
  ∃ (c : ℝ), c = -2 ∧ 
  (∀ x : ℝ, f x ≤ f c) ∧
  (∀ ε > 0, ∃ x : ℝ, |x - c| < ε ∧ f x < f c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_point_of_f_l596_59625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_surjective_l596_59600

-- Define the function s(x)
noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^3

-- State the theorem
theorem s_surjective : ∀ y : ℝ, ∃ x : ℝ, x ≠ 2 ∧ s x = y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_surjective_l596_59600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_puzzle_l596_59658

structure Person where
  name : String
  items : ℕ

structure Couple where
  husband : Person
  wife : Person

def is_valid_couple (c : Couple) : Prop :=
  c.husband.items^2 - c.wife.items^2 = 63

theorem market_puzzle :
  ∀ (john peter alexis : Person) (mary kitty jenny : Person),
    john.items^2 - jenny.items^2 = 63 →
    peter.items^2 - kitty.items^2 = 63 →
    alexis.items^2 - mary.items^2 = 63 →
    john.items = kitty.items + 23 →
    peter.items = mary.items + 11 →
    (∀ (h w : Person), h.items^2 - w.items^2 = 63 → h.items > w.items) →
    is_valid_couple (Couple.mk john jenny) ∧
    is_valid_couple (Couple.mk peter kitty) ∧
    is_valid_couple (Couple.mk alexis mary) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_puzzle_l596_59658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l596_59607

noncomputable section

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

/-- Definition of eccentricity for an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b ^ 2 / a ^ 2))

/-- Theorem stating the eccentricity of the ellipse under given conditions -/
theorem ellipse_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (C : Set (ℝ × ℝ)) (hC : C = Ellipse a b)
  (F₂ : ℝ × ℝ) (hF₂ : F₂ ∈ C) (hF₂_focus : F₂.1 > 0 ∧ F₂.2 = 0)
  (M : ℝ × ℝ) (hM : M.1 = 0)
  (A : ℝ × ℝ) (hA : A ∈ C)
  (h_line : ∃ (t : ℝ), A = M + t • (F₂ - M))
  (h_dist : ‖A‖ = ‖F₂‖ ∧ ‖A‖ = 3 * ‖M‖) :
  eccentricity a b = Real.sqrt 10 / 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l596_59607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_to_exp_form_l596_59687

noncomputable def complex_number : ℂ := 2 - 2 * Complex.I * Real.sqrt 2

theorem complex_to_exp_form (z : ℂ) (r : ℝ) (θ : ℝ) 
  (h : z = r * Complex.exp (Complex.I * θ)) :
  z = complex_number → θ = -5 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_to_exp_form_l596_59687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_height_in_specific_tank_l596_59628

/-- The height of oil in a cylindrical tank -/
noncomputable def oil_height (tank_diameter : ℝ) (oil_volume : ℝ) : ℝ :=
  (4 * oil_volume) / (Real.pi * tank_diameter^2)

/-- Theorem stating the height of oil in a specific cylindrical tank -/
theorem oil_height_in_specific_tank :
  oil_height 4 48 = 12 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_height_in_specific_tank_l596_59628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l596_59638

/-- The repeating decimal 0.363636... -/
noncomputable def repeating_decimal : ℚ :=
  ∑' n, (36 : ℚ) / (100 ^ (n + 1))

/-- The fraction 4/11 -/
def target_fraction : ℚ := 4/11

theorem repeating_decimal_equals_fraction :
  repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l596_59638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_angle_l596_59610

/-- A truncated cone circumscribed around a sphere -/
structure TruncatedCone :=
  (r₁ : ℝ) -- radius of the smaller base
  (r₂ : ℝ) -- radius of the larger base
  (h : ℝ)  -- height of the truncated cone

/-- The angle between the generatrix and the base of a truncated cone -/
noncomputable def generatrix_angle (tc : TruncatedCone) : ℝ :=
  Real.arccos (tc.r₁ / Real.sqrt (tc.r₁^2 + tc.h^2))

/-- Theorem: For a truncated cone circumscribed around a sphere, 
    if the area of one base is four times larger than the area of the other base, 
    then the angle between the generatrix of the cone and the plane of its base is arccos(1/√5) -/
theorem truncated_cone_angle (tc : TruncatedCone) 
  (h_sphere : tc.h = 2 * tc.r₁) -- The height is twice the smaller radius (due to being circumscribed around a sphere)
  (h_area : tc.r₂^2 = 4 * tc.r₁^2) -- The area of one base is four times larger than the other
  : generatrix_angle tc = Real.arccos (1 / Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_angle_l596_59610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_y_value_l596_59630

-- Define the equation that (x,y) satisfies
def satisfies_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 20*x + 36*y

-- Define the minimum value of y
noncomputable def min_y : ℝ := 18 - 2 * Real.sqrt 106

-- Theorem statement
theorem minimum_y_value :
  ∀ x y : ℝ, satisfies_equation x y → ∀ z : ℝ, satisfies_equation x z → z ≥ min_y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_y_value_l596_59630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_property_characterization_l596_59684

def is_digit_of (n d : ℕ) : Prop :=
  ∃ (k m : ℕ), n = k * 10 + d ∧ d < 10 ∧ m = k * 10

def has_digit_property (x : ℕ) : Prop :=
  ∃ (a : ℕ), a ≠ 0 ∧
  (∀ (d : ℕ), is_digit_of x d → d ≥ a) ∧
  (∀ (d : ℕ), is_digit_of x d → 
    ∃ (d' : ℕ), is_digit_of ((x - a)^2) d' ∧ d' = d - a)

theorem digit_property_characterization :
  ∀ (x : ℕ), has_digit_property x ↔ (1 ≤ x ∧ x ≤ 9) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_property_characterization_l596_59684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_magnitude_l596_59660

noncomputable def z : ℂ := Complex.exp (4 * Real.pi * Complex.I / 7)

theorem complex_sum_magnitude : Complex.abs (z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_magnitude_l596_59660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_on_board_l596_59677

theorem max_numbers_on_board : 
  ∃ (S : Finset ℕ), 
    (∀ n, n ∈ S → n ≤ 235) ∧ 
    (∀ a b c, a ∈ S → b ∈ S → c ∈ S → a ≠ b → a ≠ c → b ≠ c → ¬(a ∣ (b - c))) ∧
    S.card = 118 ∧
    (∀ T : Finset ℕ, 
      (∀ n, n ∈ T → n ≤ 235) → 
      (∀ a b c, a ∈ T → b ∈ T → c ∈ T → a ≠ b → a ≠ c → b ≠ c → ¬(a ∣ (b - c))) → 
      T.card ≤ 118) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_on_board_l596_59677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_equation_l596_59641

/-- Represents the daily price change of a stock -/
def DailyPriceChange : Type := ℝ

/-- Represents the average growth rate over two days -/
def AverageGrowthRate : Type := ℝ

/-- The maximum allowed daily price change (as a decimal) -/
def MaxDailyChange : ℝ := 0.1

/-- The price change when the stock hits the lower limit -/
def LowerLimitChange : ℝ := -MaxDailyChange

/-- 
Given:
- A stock's price can change by at most 10% each day
- The stock hit the lower limit (10% decrease) on one day
- The stock returned to its original price over the next two days
- x is the average growth rate over these two days

Prove that x satisfies the equation (1+x)^2 = 10/9
-/
theorem stock_price_equation (x : ℝ) :
  (LowerLimitChange = -MaxDailyChange) →
  ((1 + LowerLimitChange) * (1 + x)^2 = 1) →
  (1 + x)^2 = 10/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_equation_l596_59641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_range_l596_59642

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |Real.exp x + a / Real.exp x|

theorem f_monotone_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, x ≤ y → f a x ≤ f a y) →
  a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_range_l596_59642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_digit_is_one_l596_59622

/-- The decimal representation of 7/19 repeats every 18 digits -/
def period : ℕ := 18

/-- The repeating sequence of digits in the decimal representation of 7/19 -/
def digit_sequence : List ℕ := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]

/-- The position we're interested in -/
def target_position : ℕ := 421

theorem seventh_digit_is_one (h : target_position % period = 7) :
  digit_sequence.get! ((target_position - 1) % period) = 1 := by
  sorry

#eval digit_sequence.get! ((target_position - 1) % period)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_digit_is_one_l596_59622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_value_l596_59615

theorem smallest_c_value (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (∃ r : ℚ, r > 0 ∧ b = a * r ∧ c = b * r) →  -- geometric progression
  (2 * c = a + b) →                           -- arithmetic progression
  c ≥ 2 :=
by
  intro ⟨r, hr, hgeo1, hgeo2⟩ harith
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_value_l596_59615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_A_in_triangle_l596_59604

theorem max_angle_A_in_triangle (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < Real.pi) ∧
  (0 < B) ∧ (B < Real.pi) ∧
  (0 < C) ∧ (C < Real.pi) ∧
  (A + B + C = Real.pi) ∧
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  (a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)) ∧
  (b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)) ∧
  (c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) ∧
  ((Real.cos B) / b = -3 * (Real.cos C) / c) →
  A ≤ Real.pi/6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_A_in_triangle_l596_59604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_3d_space_l596_59653

/-- The area of a triangle given two sides and the angle between them -/
noncomputable def triangleArea (a b : ℝ) (θ : ℝ) : ℝ := (1/2) * a * b * Real.sin θ

/-- The theorem to be proved -/
theorem triangle_area_in_3d_space (O A B C : Fin 3 → ℝ) (θ : ℝ) : 
  O = ![0, 0, 0] →
  A = ![3, 0, 0] →
  B = ![0, 4, 0] →
  C = ![0, 0, 5] →
  θ = π/4 →
  triangleArea (‖A - B‖) (‖A - C‖) θ = (5 * Real.sqrt 68) / 4 := by
  sorry

#check triangle_area_in_3d_space

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_3d_space_l596_59653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_exp_inequality_l596_59609

theorem log_exp_inequality : ∃ (a b c : ℝ),
  a = Real.log 0.99 ∧
  b = Real.exp 0.1 ∧
  c = (0.99 : ℝ) ^ Real.exp 1 ∧
  a < c ∧ c < b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_exp_inequality_l596_59609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_longest_paths_l596_59659

/-- A circle with two points on its diameter equidistant from the center -/
structure CircleWithDiameterPoints where
  /-- The radius of the circle -/
  r : ℝ
  /-- The center of the circle -/
  O : ℝ × ℝ
  /-- Point A on the diameter -/
  A : ℝ × ℝ
  /-- Point B on the diameter -/
  B : ℝ × ℝ
  /-- A and B are on the diameter -/
  diam : A.1 + B.1 = 2 * O.1 ∧ A.2 = B.2
  /-- A and B are equidistant from the center -/
  equidistant : dist A O = dist B O
  /-- The distance between A and O (and B and O) is the radius -/
  radius : dist A O = r

/-- A point is on the circle if its distance from the center is equal to the radius -/
def on_circle (P O : ℝ × ℝ) (r : ℝ) : Prop :=
  dist P O = r

/-- The shortest and longest paths from A to B via the circumference -/
theorem shortest_longest_paths (c : CircleWithDiameterPoints) :
  (∃ path : ℝ, path = 2 * c.r ∧ 
    ∀ other_path : ℝ, (∃ P : ℝ × ℝ, on_circle P c.O c.r ∧ 
      other_path = dist c.A P + dist P c.B) → path ≤ other_path) ∧
  (∀ M : ℝ, ∃ path : ℝ, (∃ P : ℝ × ℝ, on_circle P c.O c.r ∧ 
    path = dist c.A P + dist P c.B) ∧ path > M) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_longest_paths_l596_59659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l596_59674

/-- Sequence defined by a₁ = 1 and aₙ₊₁ = 2aₙ + 1 for n ≥ 1 -/
def a : ℕ → ℤ
  | 0 => 0  -- Add this case to handle n = 0
  | 1 => 1
  | n + 1 => 2 * a n + 1

/-- Theorem: The general term of the sequence is aₙ = 2ⁿ - 1 for n > 0 -/
theorem a_general_term (n : ℕ) (h : n > 0) : a n = 2^n - 1 := by
  sorry

#eval a 5  -- Add this line to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l596_59674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_four_white_rooks_l596_59637

-- Define the chessboard size
def boardSize : Nat := 8

-- Define a rook's position
structure RookPosition where
  row : Fin boardSize
  col : Fin boardSize

-- Define the color of a square
def isWhiteSquare (pos : RookPosition) : Prop :=
  (pos.row.val + pos.col.val) % 2 = 1

-- Define non-attacking rooks
def nonAttacking (rooks : List RookPosition) : Prop :=
  ∀ r1 r2, r1 ∈ rooks → r2 ∈ rooks → r1 ≠ r2 → r1.row ≠ r2.row ∧ r1.col ≠ r2.col

theorem at_least_four_white_rooks 
  (rooks : List RookPosition) 
  (h1 : rooks.length = boardSize)
  (h2 : nonAttacking rooks)
  (h3 : ∃ r1 r2 r3, r1 ∈ rooks ∧ r2 ∈ rooks ∧ r3 ∈ rooks ∧ 
        isWhiteSquare r1 ∧ isWhiteSquare r2 ∧ isWhiteSquare r3) :
  ∃ r1 r2 r3 r4, r1 ∈ rooks ∧ r2 ∈ rooks ∧ r3 ∈ rooks ∧ r4 ∈ rooks ∧
        isWhiteSquare r1 ∧ isWhiteSquare r2 ∧ isWhiteSquare r3 ∧ isWhiteSquare r4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_four_white_rooks_l596_59637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_theorem_l596_59654

-- Define the triangle ABC and points P and Q
variable (A B C P Q : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : (A -ᵥ P) + (C -ᵥ P) = (0 : EuclideanSpace ℝ (Fin 2)))
variable (h2 : 2 • (A -ᵥ Q) + (B -ᵥ Q) + (C -ᵥ Q) = B -ᵥ C)

-- State the theorem
theorem midpoint_theorem :
  ‖P -ᵥ Q‖ = (1/2 : ℝ) * ‖B -ᵥ C‖ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_theorem_l596_59654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_when_b_is_2_cos_half_angle_difference_l596_59634

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B ∧ t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧ t.A + t.B + t.C = Real.pi

-- Theorem for part 1
theorem max_area_when_b_is_2 (t : Triangle) (h : triangle_conditions t) (h_b : t.b = 2) :
  ∃ (max_area : Real), max_area = 1 ∧ ∀ (area : Real), area ≤ max_area := by sorry

-- Theorem for part 2
theorem cos_half_angle_difference (t : Triangle) (h : triangle_conditions t)
  (h_cos : 1 / Real.cos t.A + 1 / Real.cos t.C = -Real.sqrt 2 / Real.cos t.B) :
  Real.cos ((t.A - t.C) / 2) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_when_b_is_2_cos_half_angle_difference_l596_59634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_distance_l596_59617

-- Define the ellipse and circle
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4
def circle' (x y : ℝ) : Prop := x^2 + (y-2)^2 = 1/3

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem max_min_distance :
  ∃ (max_dist min_dist : ℝ),
    (∀ (x1 y1 x2 y2 : ℝ), 
      ellipse x1 y1 → circle' x2 y2 → 
      distance x1 y1 x2 y2 ≤ max_dist) ∧
    (∃ (x1 y1 x2 y2 : ℝ), 
      ellipse x1 y1 ∧ circle' x2 y2 ∧ 
      distance x1 y1 x2 y2 = max_dist) ∧
    (∀ (x1 y1 x2 y2 : ℝ), 
      ellipse x1 y1 → circle' x2 y2 → 
      distance x1 y1 x2 y2 ≥ min_dist) ∧
    (∃ (x1 y1 x2 y2 : ℝ), 
      ellipse x1 y1 ∧ circle' x2 y2 ∧ 
      distance x1 y1 x2 y2 = min_dist) ∧
    max_dist = (2 * Real.sqrt 21) / 3 + Real.sqrt 3 / 3 ∧
    min_dist = 1 - Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_distance_l596_59617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_B_statement_C_statement_D_l596_59649

-- Define a and b
noncomputable def a : ℝ := 2 * Real.sin (1/2)
noncomputable def b : ℝ := Real.cos (1/2)

-- Statement B
theorem statement_B : 2 * b^2 - 1 > 1/2 := by sorry

-- Statement C
theorem statement_C : a > b := by sorry

-- Statement D
theorem statement_D : a + b < Real.sqrt 15 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_B_statement_C_statement_D_l596_59649
