import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_implies_m_leq_2_l3618_361865

theorem no_solution_implies_m_leq_2 (m : ℝ) :
  (∀ x : ℝ, ¬(x - 2 < 3*x - 6 ∧ x < m)) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_leq_2_l3618_361865


namespace NUMINAMATH_CALUDE_heartsuit_property_false_l3618_361801

-- Define the ♥ operation for real numbers
def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem stating that 3(x ♥ y) ≠ (3x) ♥ y for all real x and y
theorem heartsuit_property_false :
  ∀ x y : ℝ, 3 * (heartsuit x y) ≠ heartsuit (3 * x) y := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_property_false_l3618_361801


namespace NUMINAMATH_CALUDE_two_consistent_faces_l3618_361888

/-- A graph representing a convex polyhedron -/
structure ConvexPolyhedronGraph where
  V : Type* -- Vertices
  E : Type* -- Edges
  F : Type* -- Faces
  adj : V → List E -- Adjacent edges for each vertex
  face_edges : F → List E -- Edges for each face
  orientation : E → Bool -- Edge orientation (True for outgoing, False for incoming)

/-- Number of changes in edge orientation around a vertex -/
def vertex_orientation_changes (G : ConvexPolyhedronGraph) (v : G.V) : Nat :=
  sorry

/-- Number of changes in edge orientation around a face -/
def face_orientation_changes (G : ConvexPolyhedronGraph) (f : G.F) : Nat :=
  sorry

/-- Main theorem -/
theorem two_consistent_faces (G : ConvexPolyhedronGraph)
  (h1 : ∀ v : G.V, ∃ e1 e2 : G.E, e1 ∈ G.adj v ∧ e2 ∈ G.adj v ∧ G.orientation e1 ≠ G.orientation e2) :
  ∃ f1 f2 : G.F, f1 ≠ f2 ∧ face_orientation_changes G f1 = 0 ∧ face_orientation_changes G f2 = 0 :=
sorry

end NUMINAMATH_CALUDE_two_consistent_faces_l3618_361888


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3618_361898

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_5 + a_8 = 24, then a_6 + a_7 = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 5 + a 8 = 24) : 
  a 6 + a 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3618_361898


namespace NUMINAMATH_CALUDE_louis_age_proof_l3618_361818

/-- Carla's age in 6 years -/
def carla_future_age : ℕ := 30

/-- Number of years until Carla reaches her future age -/
def years_until_future : ℕ := 6

/-- Sum of Carla's and Louis's current ages -/
def sum_of_ages : ℕ := 55

/-- Louis's current age -/
def louis_age : ℕ := 31

theorem louis_age_proof :
  louis_age = sum_of_ages - (carla_future_age - years_until_future) :=
by sorry

end NUMINAMATH_CALUDE_louis_age_proof_l3618_361818


namespace NUMINAMATH_CALUDE_dragon_cannot_be_killed_l3618_361847

/-- Represents the possible number of heads Arthur can cut off in a single swipe --/
inductive CutOff
  | fifteen
  | seventeen
  | twenty
  | five

/-- Represents the number of heads that grow back after a cut --/
def regrow (c : CutOff) : ℕ :=
  match c with
  | CutOff.fifteen => 24
  | CutOff.seventeen => 2
  | CutOff.twenty => 14
  | CutOff.five => 17

/-- Represents a single action of cutting off heads and regrowing --/
def action (c : CutOff) : ℤ :=
  match c with
  | CutOff.fifteen => 24 - 15
  | CutOff.seventeen => 2 - 17
  | CutOff.twenty => 14 - 20
  | CutOff.five => 17 - 5

/-- The main theorem stating that it's impossible to kill the dragon --/
theorem dragon_cannot_be_killed :
  ∀ (n : ℕ) (actions : List CutOff),
    (100 + (actions.map action).sum : ℤ) % 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_dragon_cannot_be_killed_l3618_361847


namespace NUMINAMATH_CALUDE_domino_pigeonhole_l3618_361896

/-- Represents a domino with two halves -/
structure Domino :=
  (half1 : Fin 7)
  (half2 : Fin 7)

/-- Represents the state of dominoes after cutting -/
structure DominoState :=
  (row : List Domino)
  (cut_halves : List (Fin 7))

/-- The theorem statement -/
theorem domino_pigeonhole 
  (dominoes : List Domino)
  (h1 : dominoes.length = 28)
  (h2 : ∀ i : Fin 7, (dominoes.map Domino.half1 ++ dominoes.map Domino.half2).count i = 7)
  (state : DominoState)
  (h3 : state.row.length = 26)
  (h4 : state.cut_halves.length = 4)
  (h5 : ∀ d ∈ dominoes, d ∈ state.row ∨ (d.half1 ∈ state.cut_halves ∧ d.half2 ∈ state.cut_halves)) :
  ∃ i j : Fin 4, i ≠ j ∧ state.cut_halves[i] = state.cut_halves[j] :=
sorry

end NUMINAMATH_CALUDE_domino_pigeonhole_l3618_361896


namespace NUMINAMATH_CALUDE_product_25_sum_0_l3618_361879

theorem product_25_sum_0 (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a * b * c * d = 25 → 
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_product_25_sum_0_l3618_361879


namespace NUMINAMATH_CALUDE_power_function_property_l3618_361852

/-- Given a power function f(x) = x^α where α ∈ ℝ, 
    if f(2) = √2, then f(4) = 2 -/
theorem power_function_property (α : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, f x = x ^ α) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l3618_361852


namespace NUMINAMATH_CALUDE_lucky_larry_coincidence_l3618_361820

theorem lucky_larry_coincidence (a b c d e : ℤ) : 
  a = 2 → b = 4 → c = 3 → d = 5 → e = -15 →
  a - (b - (c * (d + e))) = a - b - c * d + e := by sorry

end NUMINAMATH_CALUDE_lucky_larry_coincidence_l3618_361820


namespace NUMINAMATH_CALUDE_smallest_n_for_multiples_l3618_361844

theorem smallest_n_for_multiples : ∃ (a : Fin 15 → ℕ), 
  (∀ i : Fin 15, 16 ≤ a i ∧ a i ≤ 34) ∧ 
  (∀ i : Fin 15, a i % (i.val + 1) = 0) ∧
  (∀ i j : Fin 15, i ≠ j → a i ≠ a j) ∧
  (∀ n : ℕ, n < 34 → ¬∃ (b : Fin 15 → ℕ), 
    (∀ i : Fin 15, 16 ≤ b i ∧ b i ≤ n) ∧ 
    (∀ i : Fin 15, b i % (i.val + 1) = 0) ∧
    (∀ i j : Fin 15, i ≠ j → b i ≠ b j)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_multiples_l3618_361844


namespace NUMINAMATH_CALUDE_softball_team_size_l3618_361897

theorem softball_team_size (men women : ℕ) : 
  women = men + 6 →
  (men : ℝ) / (women : ℝ) = 0.45454545454545453 →
  men + women = 16 := by
sorry

end NUMINAMATH_CALUDE_softball_team_size_l3618_361897


namespace NUMINAMATH_CALUDE_coin_problem_l3618_361831

/-- Represents the number of different coin values that can be obtained -/
def different_values (five_cent : ℕ) (ten_cent : ℕ) : ℕ :=
  23 - five_cent

/-- Represents the total number of coins -/
def total_coins (five_cent : ℕ) (ten_cent : ℕ) : ℕ :=
  five_cent + ten_cent

theorem coin_problem (five_cent ten_cent : ℕ) :
  total_coins five_cent ten_cent = 12 →
  different_values five_cent ten_cent = 19 →
  ten_cent = 8 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l3618_361831


namespace NUMINAMATH_CALUDE_total_students_in_class_l3618_361855

/-- 
Given a class where 45 students are present when 10% are absent,
prove that the total number of students in the class is 50.
-/
theorem total_students_in_class : 
  ∀ (total : ℕ), 
  (↑total * (1 - 0.1) : ℝ) = 45 → 
  total = 50 := by
sorry

end NUMINAMATH_CALUDE_total_students_in_class_l3618_361855


namespace NUMINAMATH_CALUDE_ellipse_and_rhombus_problem_l3618_361819

-- Define the ellipse C₁
def C₁ (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line BD
def BD (x y : ℝ) : Prop := 7 * x - 7 * y + 1 = 0

-- Define the rhombus ABCD
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Main theorem
theorem ellipse_and_rhombus_problem 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (F₁ F₂ M : ℝ × ℝ) 
  (hF₂ : F₂ = (1, 0)) 
  (hM : C₁ a b M.1 M.2 ∧ C₂ M.1 M.2) 
  (hMF₂ : Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = 2) 
  (ABCD : Rhombus) 
  (hAC : C₁ a b ABCD.A.1 ABCD.A.2 ∧ C₁ a b ABCD.C.1 ABCD.C.2) 
  (hBD : BD ABCD.B.1 ABCD.B.2 ∧ BD ABCD.D.1 ABCD.D.2) :
  (∀ x y, C₁ a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (ABCD.A.2 = -ABCD.A.1 - 1/14 ∧ ABCD.C.2 = -ABCD.C.1 - 1/14) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_rhombus_problem_l3618_361819


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l3618_361815

-- Define the set of real polynomials
def RealPolynomial := Polynomial ℝ

-- Define the condition for a, b, c
def SumProductZero (a b c : ℝ) : Prop := a * b + b * c + c * a = 0

-- Define the equation that P must satisfy
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ a b c : ℝ, SumProductZero a b c →
    P.eval (a - b) + P.eval (b - c) + P.eval (c - a) = 2 * P.eval (a + b + c)

-- Define the form of the solution polynomial
def IsSolutionForm (P : RealPolynomial) : Prop :=
  ∃ u v : ℝ, P = Polynomial.monomial 4 u + Polynomial.monomial 2 v

-- State the theorem
theorem polynomial_equation_solution :
  ∀ P : RealPolynomial, SatisfiesEquation P → IsSolutionForm P :=
sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l3618_361815


namespace NUMINAMATH_CALUDE_expression_value_l3618_361841

theorem expression_value (a : ℚ) (h : a = 1/3) : (2 * a⁻¹ + a⁻¹ / 3) / a^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3618_361841


namespace NUMINAMATH_CALUDE_ellen_legos_l3618_361869

theorem ellen_legos (initial_legos : ℕ) (lost_legos : ℕ) 
  (h1 : initial_legos = 2080) 
  (h2 : lost_legos = 17) : 
  initial_legos - lost_legos = 2063 := by
sorry

end NUMINAMATH_CALUDE_ellen_legos_l3618_361869


namespace NUMINAMATH_CALUDE_function_properties_imply_cosine_and_value_l3618_361806

/-- The function f(x) = sin(ωx + φ) with given properties -/
noncomputable def f (ω φ : ℝ) : ℝ → ℝ := fun x ↦ Real.sin (ω * x + φ)

/-- The theorem statement -/
theorem function_properties_imply_cosine_and_value
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : 0 ≤ φ ∧ φ ≤ π)
  (h_even : ∀ x, f ω φ x = f ω φ (-x))
  (h_distance : ∃ (x₁ x₂ : ℝ), abs (x₁ - x₂) = π ∧ abs (f ω φ x₁ - f ω φ x₂) = 2)
  (α : ℝ)
  (h_sum : Real.sin α + f ω φ α = 2/3) :
  (∀ x, f ω φ x = Real.cos x) ∧
  ((Real.sqrt 2 * Real.sin (2*α - π/4) + 1) / (1 + Real.tan α) = -5/9) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_imply_cosine_and_value_l3618_361806


namespace NUMINAMATH_CALUDE_expression_equality_l3618_361840

theorem expression_equality : 201 * 5 + 1220 - 2 * 3 * 5 * 7 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3618_361840


namespace NUMINAMATH_CALUDE_penguins_fed_correct_l3618_361871

/-- The number of penguins that have already gotten a fish -/
def penguins_fed (total_penguins : ℕ) (penguins_to_feed : ℕ) : ℕ :=
  total_penguins - penguins_to_feed

theorem penguins_fed_correct (total_fish : ℕ) (total_penguins : ℕ) (penguins_to_feed : ℕ) :
  total_fish = 68 →
  total_penguins = 36 →
  penguins_to_feed = 17 →
  penguins_fed total_penguins penguins_to_feed = 19 :=
by
  sorry

#eval penguins_fed 36 17

end NUMINAMATH_CALUDE_penguins_fed_correct_l3618_361871


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3618_361878

/-- Given a rectangular field with one uncovered side of 30 feet and three sides
    requiring 70 feet of fencing, the area of the field is 600 square feet. -/
theorem rectangular_field_area (L W : ℝ) : 
  L = 30 →
  L + 2 * W = 70 →
  L * W = 600 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3618_361878


namespace NUMINAMATH_CALUDE_cubic_equation_root_c_value_l3618_361830

theorem cubic_equation_root_c_value : ∃ (c d : ℚ),
  ((-2 : ℝ) - 3 * Real.sqrt 5) ^ 3 + c * ((-2 : ℝ) - 3 * Real.sqrt 5) ^ 2 + 
  d * ((-2 : ℝ) - 3 * Real.sqrt 5) + 50 = 0 → c = 114 / 41 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_c_value_l3618_361830


namespace NUMINAMATH_CALUDE_geometric_solid_sum_of_edges_l3618_361848

/-- Represents a rectangular solid with sides in geometric progression -/
structure GeometricSolid where
  a : ℝ  -- shortest side
  r : ℝ  -- common ratio
  h : r > 0  -- ensure positive ratio

/-- Volume of a GeometricSolid -/
def volume (s : GeometricSolid) : ℝ := s.a * (s.a * s.r) * (s.a * s.r * s.r)

/-- Surface area of a GeometricSolid -/
def surfaceArea (s : GeometricSolid) : ℝ :=
  2 * (s.a * (s.a * s.r) + s.a * (s.a * s.r * s.r) + (s.a * s.r) * (s.a * s.r * s.r))

/-- Sum of lengths of all edges of a GeometricSolid -/
def sumOfEdges (s : GeometricSolid) : ℝ := 4 * (s.a + (s.a * s.r) + (s.a * s.r * s.r))

/-- Theorem statement -/
theorem geometric_solid_sum_of_edges :
  ∀ s : GeometricSolid,
    volume s = 125 →
    surfaceArea s = 150 →
    sumOfEdges s = 60 := by
  sorry

end NUMINAMATH_CALUDE_geometric_solid_sum_of_edges_l3618_361848


namespace NUMINAMATH_CALUDE_circle_triangle_problem_l3618_361876

/-- Given that a triangle equals three circles and a triangle plus a circle equals 40,
    prove that the circle equals 10 and the triangle equals 30. -/
theorem circle_triangle_problem (circle triangle : ℕ) 
    (h1 : triangle = 3 * circle)
    (h2 : triangle + circle = 40) :
    circle = 10 ∧ triangle = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_problem_l3618_361876


namespace NUMINAMATH_CALUDE_a_eq_neg_one_sufficient_not_necessary_l3618_361821

-- Define the complex number z
def z (a : ℝ) : ℂ := (a - 2*Complex.I)*Complex.I

-- Define the point M in the complex plane
def M (a : ℝ) : ℂ := z a

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (c : ℂ) : Prop := 0 < c.re ∧ c.im < 0

-- State the theorem
theorem a_eq_neg_one_sufficient_not_necessary :
  (∀ a : ℝ, a = -1 → in_fourth_quadrant (M a)) ∧
  (∃ a : ℝ, a ≠ -1 ∧ in_fourth_quadrant (M a)) :=
sorry

end NUMINAMATH_CALUDE_a_eq_neg_one_sufficient_not_necessary_l3618_361821


namespace NUMINAMATH_CALUDE_hyperbola_focus_l3618_361839

/-- Given a hyperbola with equation x²/a² - y²/2 = 1, where one of its asymptotes
    passes through the point (√2, 1), prove that one of its foci has coordinates (√6, 0) -/
theorem hyperbola_focus (a : ℝ) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 2 = 1) →  -- Hyperbola equation
  (∃ (m : ℝ), m * Real.sqrt 2 = 1 ∧ m = Real.sqrt 2 / a) →  -- Asymptote through (√2, 1)
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 2 = 1 ∧ x = Real.sqrt 6 ∧ y = 0) :=  -- Focus at (√6, 0)
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l3618_361839


namespace NUMINAMATH_CALUDE_toy_shop_spending_l3618_361875

def total_spent (trevor_spending : ℕ) (reed_spending : ℕ) (quinn_spending : ℕ) (years : ℕ) : ℕ :=
  (trevor_spending + reed_spending + quinn_spending) * years

theorem toy_shop_spending (trevor_spending reed_spending quinn_spending : ℕ) :
  trevor_spending = reed_spending + 20 →
  reed_spending = 2 * quinn_spending →
  trevor_spending = 80 →
  total_spent trevor_spending reed_spending quinn_spending 4 = 680 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_shop_spending_l3618_361875


namespace NUMINAMATH_CALUDE_product_units_digit_in_base_6_l3618_361861

/-- The units digit of the product of two numbers in a given base -/
def unitsDigitInBase (a b : ℕ) (base : ℕ) : ℕ :=
  (a * b) % base

/-- 314 in base 10 -/
def num1 : ℕ := 314

/-- 59 in base 10 -/
def num2 : ℕ := 59

/-- The base we're converting to -/
def targetBase : ℕ := 6

theorem product_units_digit_in_base_6 :
  unitsDigitInBase num1 num2 targetBase = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_units_digit_in_base_6_l3618_361861


namespace NUMINAMATH_CALUDE_five_fridays_in_august_l3618_361887

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Given a month and a day of the week, count how many times that day appears -/
def countDayInMonth (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- The next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  sorry

theorem five_fridays_in_august 
  (july : Month)
  (august : Month)
  (h1 : july.days = 31)
  (h2 : august.days = 31)
  (h3 : countDayInMonth july DayOfWeek.Tuesday = 5) :
  countDayInMonth august DayOfWeek.Friday = 5 :=
sorry

end NUMINAMATH_CALUDE_five_fridays_in_august_l3618_361887


namespace NUMINAMATH_CALUDE_new_person_weight_l3618_361867

def group_size : ℕ := 8
def average_weight_increase : ℝ := 2.5
def replaced_person_weight : ℝ := 65

theorem new_person_weight (new_weight : ℝ) :
  (group_size : ℝ) * average_weight_increase = new_weight - replaced_person_weight →
  new_weight = 85 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l3618_361867


namespace NUMINAMATH_CALUDE_inequality_proof_l3618_361803

theorem inequality_proof (f : ℝ → ℝ) (a m n : ℝ) :
  (∀ x, f x = |x - a| + 1) →
  (Set.Icc 0 2 = {x | f x ≤ 2}) →
  m > 0 →
  n > 0 →
  1/m + 1/n = a →
  m + 2*n ≥ 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3618_361803


namespace NUMINAMATH_CALUDE_circle_theorem_l3618_361866

-- Define the set of complex numbers satisfying the condition
def S : Set ℂ := {z : ℂ | Complex.abs (z - 3 * Complex.I) = 10}

-- State the theorem
theorem circle_theorem : 
  S = {z : ℂ | Complex.abs (z - Complex.ofReal 0 - Complex.I * 3) = 10} := by
sorry

end NUMINAMATH_CALUDE_circle_theorem_l3618_361866


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_half_l3618_361873

theorem angle_sum_is_pi_half (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β)) : 
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_half_l3618_361873


namespace NUMINAMATH_CALUDE_senior_titles_in_sample_l3618_361827

/-- Represents the number of staff members with senior titles in a stratified sample -/
def seniorTitlesInSample (totalStaff : ℕ) (seniorStaff : ℕ) (sampleSize : ℕ) : ℕ :=
  (seniorStaff * sampleSize) / totalStaff

/-- Theorem: In a company with 150 staff members, including 15 with senior titles,
    a stratified sample of size 30 will contain 3 staff members with senior titles. -/
theorem senior_titles_in_sample :
  seniorTitlesInSample 150 15 30 = 3 := by
  sorry

end NUMINAMATH_CALUDE_senior_titles_in_sample_l3618_361827


namespace NUMINAMATH_CALUDE_chicken_difference_l3618_361893

theorem chicken_difference (mary john ray : ℕ) 
  (h1 : john = mary + 5)
  (h2 : ray + 6 = mary)
  (h3 : ray = 10) : 
  john - ray = 11 :=
by sorry

end NUMINAMATH_CALUDE_chicken_difference_l3618_361893


namespace NUMINAMATH_CALUDE_larger_number_proof_l3618_361817

theorem larger_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 25) (h2 : Nat.lcm a b = 25 * 14 * 16) :
  max a b = 400 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3618_361817


namespace NUMINAMATH_CALUDE_inscribed_rectangle_height_l3618_361805

/-- 
Given a triangle with base b and height h, and a rectangle inscribed in it such that:
1. The base of the rectangle coincides with the base of the triangle
2. The height of the rectangle is half its base
Prove that the height of the rectangle x is equal to bh / (2h + b)
-/
theorem inscribed_rectangle_height (b h : ℝ) (h1 : 0 < b) (h2 : 0 < h) : 
  ∃ x : ℝ, x > 0 ∧ x = b * h / (2 * h + b) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_height_l3618_361805


namespace NUMINAMATH_CALUDE_f_composition_value_l3618_361845

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else 2 * Real.sqrt 2 * Real.cos x

theorem f_composition_value : f (f (-Real.pi/4)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3618_361845


namespace NUMINAMATH_CALUDE_problem_solution_l3618_361835

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 3)
  (h_eq2 : y + 1 / x = 31) : 
  z + 1 / y = 9 / 23 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3618_361835


namespace NUMINAMATH_CALUDE_inequality_range_l3618_361822

theorem inequality_range (a : ℝ) : 
  (∀ x > a, 2 * x + 1 / (x - a) ≥ 2 * Real.sqrt 2) ↔ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l3618_361822


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l3618_361828

noncomputable def s (t : ℝ) : ℝ := (1/4) * t^4 - 3

theorem instantaneous_velocity_at_5 :
  (deriv s) 5 = 125 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l3618_361828


namespace NUMINAMATH_CALUDE_unique_prime_solution_l3618_361851

theorem unique_prime_solution : ∃! (p q r : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p + q^2 = r^4 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l3618_361851


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_6_l3618_361832

theorem greatest_integer_with_gcd_6 :
  ∃ n : ℕ, n < 150 ∧ n.gcd 12 = 6 ∧ ∀ m : ℕ, m < 150 → m.gcd 12 = 6 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_6_l3618_361832


namespace NUMINAMATH_CALUDE_prob_three_red_large_deck_l3618_361884

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (hsum : total_cards = red_cards + black_cards)

/-- Probability of drawing three red cards in a row -/
def prob_three_red (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards *
  ((d.red_cards - 1) : ℚ) / (d.total_cards - 1) *
  ((d.red_cards - 2) : ℚ) / (d.total_cards - 2)

/-- The main theorem -/
theorem prob_three_red_large_deck :
  let d : Deck := ⟨104, 52, 52, rfl⟩
  prob_three_red d = 425 / 3502 := by sorry

end NUMINAMATH_CALUDE_prob_three_red_large_deck_l3618_361884


namespace NUMINAMATH_CALUDE_square_roots_and_cube_root_problem_l3618_361850

theorem square_roots_and_cube_root_problem (x y a : ℝ) :
  x > 0 ∧
  (a + 3)^2 = x ∧
  (2*a - 15)^2 = x ∧
  (x + y - 2)^(1/3) = 4 →
  x - 2*y + 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_and_cube_root_problem_l3618_361850


namespace NUMINAMATH_CALUDE_tank_emptying_time_l3618_361809

/-- Proves the time to empty a tank with given conditions -/
theorem tank_emptying_time 
  (tank_capacity : ℝ) 
  (leak_empty_time : ℝ) 
  (inlet_rate_per_minute : ℝ) 
  (h1 : tank_capacity = 4320)
  (h2 : leak_empty_time = 6)
  (h3 : inlet_rate_per_minute = 3) : 
  (tank_capacity / (tank_capacity / leak_empty_time - inlet_rate_per_minute * 60)) = 8 := by
  sorry

#check tank_emptying_time

end NUMINAMATH_CALUDE_tank_emptying_time_l3618_361809


namespace NUMINAMATH_CALUDE_unique_solution_cube_root_system_l3618_361826

theorem unique_solution_cube_root_system :
  ∃! (x y z : ℝ),
    Real.sqrt (x^3 - y) = z - 1 ∧
    Real.sqrt (y^3 - z) = x - 1 ∧
    Real.sqrt (z^3 - x) = y - 1 ∧
    x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_root_system_l3618_361826


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3618_361802

/-- The repeating decimal 0.215215215... -/
def repeating_decimal : ℚ := 0.215215215

/-- The fraction 215/999 -/
def fraction : ℚ := 215 / 999

/-- Theorem stating that the repeating decimal 0.215215215... is equal to the fraction 215/999 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3618_361802


namespace NUMINAMATH_CALUDE_savings_from_discount_l3618_361899

def initial_price : ℚ := 475
def discounted_price : ℚ := 199

theorem savings_from_discount :
  initial_price - discounted_price = 276 := by
  sorry

end NUMINAMATH_CALUDE_savings_from_discount_l3618_361899


namespace NUMINAMATH_CALUDE_min_side_difference_in_triangle_l3618_361885

theorem min_side_difference_in_triangle (xy xz yz : ℕ) : 
  xy + xz + yz = 3021 →
  xy < xz →
  xz < yz →
  2 ≤ yz - xy :=
by sorry

end NUMINAMATH_CALUDE_min_side_difference_in_triangle_l3618_361885


namespace NUMINAMATH_CALUDE_probability_of_two_defective_in_two_tests_l3618_361854

/-- The number of electronic components -/
def total_components : ℕ := 6

/-- The number of defective components -/
def defective_components : ℕ := 2

/-- The number of qualified components -/
def qualified_components : ℕ := 4

/-- The probability of finding exactly 2 defective components after 2 tests -/
def probability_two_defective_in_two_tests : ℚ := 1 / 15

/-- Theorem stating the probability of finding exactly 2 defective components after 2 tests -/
theorem probability_of_two_defective_in_two_tests :
  probability_two_defective_in_two_tests = 1 / 15 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_two_defective_in_two_tests_l3618_361854


namespace NUMINAMATH_CALUDE_sum_of_digits_plus_two_l3618_361857

/-- T(n) represents the sum of the digits of a positive integer n -/
def T (n : ℕ+) : ℕ := sorry

/-- For a certain positive integer n, T(n) = 1598 implies T(n+2) = 1600 -/
theorem sum_of_digits_plus_two (n : ℕ+) (h : T n = 1598) : T (n + 2) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_plus_two_l3618_361857


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3618_361837

/-- Represents a geometric sequence with common ratio greater than 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : q > 1
  h_geometric : ∀ n, a (n + 1) = q * a n

/-- Given conditions for the geometric sequence -/
def SequenceConditions (seq : GeometricSequence) : Prop :=
  seq.a 3 * seq.a 7 = 72 ∧ seq.a 2 + seq.a 8 = 27

theorem geometric_sequence_problem (seq : GeometricSequence) 
  (h : SequenceConditions seq) : seq.a 12 = 96 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3618_361837


namespace NUMINAMATH_CALUDE_gwen_total_books_l3618_361807

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 4

/-- The number of shelves for mystery books -/
def mystery_shelves : ℕ := 5

/-- The number of shelves for picture books -/
def picture_shelves : ℕ := 3

/-- The total number of books Gwen has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem gwen_total_books : total_books = 32 := by sorry

end NUMINAMATH_CALUDE_gwen_total_books_l3618_361807


namespace NUMINAMATH_CALUDE_arrangement_count_l3618_361834

def number_of_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose m k * Nat.factorial (n + m - 2 * k)

theorem arrangement_count :
  let total_people : ℕ := 6
  let people_per_row : ℕ := 3
  number_of_arrangements total_people people_per_row 2 = 216 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l3618_361834


namespace NUMINAMATH_CALUDE_incorrect_equation_l3618_361892

/-- Represents a decimal number with a non-repeating segment followed by a repeating segment -/
structure DecimalNumber where
  X : ℕ  -- non-repeating segment
  Y : ℕ  -- repeating segment
  u : ℕ  -- number of digits in X
  v : ℕ  -- number of digits in Y

/-- Converts a DecimalNumber to its real value -/
def toReal (z : DecimalNumber) : ℚ :=
  (z.X : ℚ) / 10^z.u + (z.Y : ℚ) / (10^z.u * (10^z.v - 1))

/-- The main theorem stating that the given equation does not hold for all DecimalNumbers -/
theorem incorrect_equation (z : DecimalNumber) : 
  ¬(10^(2*z.u) * (10^z.v - 1) * toReal z = (z.Y : ℚ) * ((z.X : ℚ)^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_equation_l3618_361892


namespace NUMINAMATH_CALUDE_zero_last_to_appear_l3618_361890

/-- Modified Fibonacci sequence -/
def modFib : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => modFib (n + 1) + modFib n

/-- The set of digits that have appeared in the units position up to the nth term -/
def digitsAppeared (n : ℕ) : Finset ℕ :=
  Finset.filter (fun d => ∃ k ≤ n, modFib k % 10 = d) (Finset.range 10)

/-- The proposition that 0 is the last digit to appear in the units position -/
theorem zero_last_to_appear : ∃ N : ℕ, 
  (∀ n ≥ N, 0 ∈ digitsAppeared n) ∧ 
  (∀ d : ℕ, d < 10 → d ≠ 0 → ∃ n < N, d ∈ digitsAppeared n) :=
sorry

end NUMINAMATH_CALUDE_zero_last_to_appear_l3618_361890


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3618_361883

theorem inequality_solution_set (x : ℝ) : 
  (|x + 1| - 2 > 0) ↔ (x < -3 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3618_361883


namespace NUMINAMATH_CALUDE_car_resale_gain_l3618_361881

/-- Calculates the percentage gain when reselling a car -/
theorem car_resale_gain (original_price selling_price_2 : ℝ) (loss_percent : ℝ) : 
  original_price = 50561.80 →
  loss_percent = 11 →
  selling_price_2 = 54000 →
  (selling_price_2 - (original_price * (1 - loss_percent / 100))) / (original_price * (1 - loss_percent / 100)) * 100 = 20 :=
by sorry

end NUMINAMATH_CALUDE_car_resale_gain_l3618_361881


namespace NUMINAMATH_CALUDE_weight_difference_l3618_361862

def bridget_weight : ℕ := 39
def martha_weight : ℕ := 2

theorem weight_difference : bridget_weight - martha_weight = 37 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3618_361862


namespace NUMINAMATH_CALUDE_amanda_peaches_difference_l3618_361870

/-- The number of peaches each person has -/
structure Peaches where
  jill : ℕ
  steven : ℕ
  jake : ℕ
  amanda : ℕ

/-- The conditions of the problem -/
def peach_conditions (p : Peaches) : Prop :=
  p.jill = 12 ∧
  p.steven = p.jill + 15 ∧
  p.jake = p.steven - 16 ∧
  p.amanda = 2 * p.jill

/-- The average number of peaches Jake, Steven, and Jill have -/
def average (p : Peaches) : ℚ :=
  (p.jake + p.steven + p.jill : ℚ) / 3

/-- The theorem to be proved -/
theorem amanda_peaches_difference (p : Peaches) (h : peach_conditions p) :
  p.amanda - average p = 7.33 := by
  sorry

end NUMINAMATH_CALUDE_amanda_peaches_difference_l3618_361870


namespace NUMINAMATH_CALUDE_average_side_length_of_squares_l3618_361811

theorem average_side_length_of_squares (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_side_length_of_squares_l3618_361811


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3618_361877

/-- A quadratic equation (k-1)x^2 + 4x + 1 = 0 has two distinct real roots -/
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  (k - 1 ≠ 0) ∧ (16 - 4 * k + 4 > 0)

/-- The range of k for which the quadratic equation has two distinct real roots -/
theorem quadratic_roots_range :
  ∀ k : ℝ, has_two_distinct_real_roots k ↔ (k < 5 ∧ k ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3618_361877


namespace NUMINAMATH_CALUDE_fewer_female_students_l3618_361856

theorem fewer_female_students (total_students : ℕ) (female_students : ℕ) 
  (h1 : total_students = 280) (h2 : female_students = 127) :
  total_students - female_students - female_students = 26 := by
  sorry

end NUMINAMATH_CALUDE_fewer_female_students_l3618_361856


namespace NUMINAMATH_CALUDE_inequality_and_bound_l3618_361858

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- State the theorem
theorem inequality_and_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = Real.sqrt 2) :
  (∀ x, f x > 3 - |x + 2| ↔ x < -3 ∨ x > 0) ∧
  (∀ x, f x - |x| ≤ Real.sqrt (a^2 + 4*b^2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_bound_l3618_361858


namespace NUMINAMATH_CALUDE_harry_stamps_l3618_361863

theorem harry_stamps (total : ℕ) (harry_ratio : ℕ) (harry_stamps : ℕ) : 
  total = 240 →
  harry_ratio = 3 →
  harry_stamps = total * harry_ratio / (harry_ratio + 1) →
  harry_stamps = 180 := by
sorry

end NUMINAMATH_CALUDE_harry_stamps_l3618_361863


namespace NUMINAMATH_CALUDE_max_value_expression_l3618_361889

theorem max_value_expression (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 10 + 10 * y * z ≤ 5 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3618_361889


namespace NUMINAMATH_CALUDE_swordfish_difference_l3618_361823

/-- The number of times Shelly and Sam go fishing -/
def fishing_trips : ℕ := 5

/-- The total number of swordfish caught in all trips -/
def total_swordfish : ℕ := 25

/-- The number of swordfish Shelly catches each time -/
def shelly_catch : ℕ := 5 - 2

/-- The number of swordfish Sam catches each time -/
def sam_catch : ℕ := (total_swordfish - fishing_trips * shelly_catch) / fishing_trips

theorem swordfish_difference : shelly_catch - sam_catch = 1 := by
  sorry

end NUMINAMATH_CALUDE_swordfish_difference_l3618_361823


namespace NUMINAMATH_CALUDE_truth_teller_liar_arrangement_l3618_361882

def is_valid_arrangement (n k : ℕ) : Prop :=
  n > 0 ∧ k > 0 ∧ k < n ∧ 
  ∃ (m : ℕ), 2^m * k < n ∧ n ≤ 2^(m+1) * k

theorem truth_teller_liar_arrangement (n k : ℕ) :
  is_valid_arrangement n k →
  ∃ (m : ℕ), n = 2^m * (n.gcd k) ∧ 2^m > (k / (n.gcd k)) :=
sorry

end NUMINAMATH_CALUDE_truth_teller_liar_arrangement_l3618_361882


namespace NUMINAMATH_CALUDE_number_difference_proof_l3618_361810

theorem number_difference_proof (s l : ℕ) : 
  (∃ x : ℕ, l = 2 * s - x) →  -- One number is some less than twice another
  s + l = 39 →               -- Their sum is 39
  s = 14 →                   -- The smaller number is 14
  2 * s - l = 3 :=           -- The difference between twice the smaller number and the larger number is 3
by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l3618_361810


namespace NUMINAMATH_CALUDE_divisibility_implies_sum_representation_l3618_361812

theorem divisibility_implies_sum_representation (n k : ℕ) 
  (h1 : n > 20) 
  (h2 : k > 1) 
  (h3 : k^2 ∣ n) : 
  ∃ a b c : ℕ, n = a * b + b * c + c * a := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_sum_representation_l3618_361812


namespace NUMINAMATH_CALUDE_three_lines_intersection_l3618_361825

-- Define the lines
def line1 (x y : ℝ) := x - y + 1 = 0
def line2 (x y : ℝ) := 2*x + y - 4 = 0
def line3 (a x y : ℝ) := a*x - y + 2 = 0

-- Define the condition of exactly two intersection points
def has_two_intersections (a : ℝ) :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    (line1 x1 y1 ∧ line2 x1 y1 ∧ line3 a x1 y1) ∧
    (line1 x2 y2 ∧ line2 x2 y2 ∧ line3 a x2 y2) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (∀ x y, line1 x y ∧ line2 x y ∧ line3 a x y → (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2))

-- Theorem statement
theorem three_lines_intersection (a : ℝ) :
  has_two_intersections a → a = 1 ∨ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l3618_361825


namespace NUMINAMATH_CALUDE_cos_negative_nineteen_pi_sixths_l3618_361833

theorem cos_negative_nineteen_pi_sixths :
  Real.cos (-19 * π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_nineteen_pi_sixths_l3618_361833


namespace NUMINAMATH_CALUDE_machinery_cost_proof_l3618_361813

def total_amount : ℝ := 250
def raw_materials_cost : ℝ := 100
def cash_percentage : ℝ := 0.1

def machinery_cost : ℝ := total_amount - raw_materials_cost - (cash_percentage * total_amount)

theorem machinery_cost_proof : machinery_cost = 125 := by
  sorry

end NUMINAMATH_CALUDE_machinery_cost_proof_l3618_361813


namespace NUMINAMATH_CALUDE_arctan_sum_three_seven_l3618_361860

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_seven_l3618_361860


namespace NUMINAMATH_CALUDE_repeating_ones_not_square_l3618_361814

/-- Defines a function that returns a number consisting of n repeating 1's -/
def repeatingOnes (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Theorem stating that for any positive natural number n, 
    the number consisting of n repeating 1's is not a perfect square -/
theorem repeating_ones_not_square (n : ℕ) (h : n > 0) : 
  ¬ ∃ m : ℕ, (repeatingOnes n) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_repeating_ones_not_square_l3618_361814


namespace NUMINAMATH_CALUDE_pamphlet_cost_l3618_361846

theorem pamphlet_cost : ∃ p : ℝ, p = 1.10 ∧ 8 * p < 9 ∧ 11 * p > 12 := by
  sorry

end NUMINAMATH_CALUDE_pamphlet_cost_l3618_361846


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l3618_361880

/-- Probability of getting heads on a biased coin -/
def prob_heads : ℚ := 2/3

/-- Probability of getting an even number on a regular six-sided die -/
def prob_even_die : ℚ := 1/2

/-- Theorem: The probability of getting heads on a biased coin with 2/3 probability for heads
    and an even number on a regular six-sided die is 1/3 -/
theorem coin_and_die_probability :
  prob_heads * prob_even_die = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_coin_and_die_probability_l3618_361880


namespace NUMINAMATH_CALUDE_parrot_count_l3618_361824

/-- Represents the number of animals in a zoo --/
structure ZooCount where
  parrots : ℕ
  snakes : ℕ
  monkeys : ℕ
  elephants : ℕ
  zebras : ℕ

/-- Checks if the zoo count satisfies the given conditions --/
def isValidZooCount (z : ZooCount) : Prop :=
  z.snakes = 3 * z.parrots ∧
  z.monkeys = 2 * z.snakes ∧
  z.elephants = (z.parrots + z.snakes) / 2 ∧
  z.zebras = z.elephants - 3 ∧
  z.monkeys - z.zebras = 35

/-- Theorem stating that there are 8 parrots in the zoo --/
theorem parrot_count : ∃ z : ZooCount, isValidZooCount z ∧ z.parrots = 8 := by
  sorry

end NUMINAMATH_CALUDE_parrot_count_l3618_361824


namespace NUMINAMATH_CALUDE_smallest_angle_in_quadrilateral_l3618_361849

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_angle_in_quadrilateral (p q r s : ℕ) : 
  is_prime p → is_prime q → is_prime r → is_prime s →
  p > q → q > r → r > s →
  p + q + r = 270 →
  p + q + r + s = 360 →
  s ≥ 97 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_quadrilateral_l3618_361849


namespace NUMINAMATH_CALUDE_line_intersection_y_coordinate_l3618_361829

/-- Given a line with slope 3/4 passing through (400, 0), 
    prove that the y-coordinate at x = -12 is -309 -/
theorem line_intersection_y_coordinate 
  (slope : ℚ) 
  (x_intercept : ℝ) 
  (x_coord : ℝ) :
  slope = 3/4 →
  x_intercept = 400 →
  x_coord = -12 →
  let y_intercept := -(slope * x_intercept)
  let y_coord := slope * x_coord + y_intercept
  y_coord = -309 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_y_coordinate_l3618_361829


namespace NUMINAMATH_CALUDE_circle_intersection_angle_l3618_361874

-- Define the circle equation
def circle_equation (x y c : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + c = 0

-- Define the center of the circle
def center : ℝ × ℝ := (2, -1)

-- Define the angle APB
def angle_APB : ℝ := 120

-- Theorem statement
theorem circle_intersection_angle (c : ℝ) :
  ∃ (A B : ℝ × ℝ),
    (A.1 = 0 ∧ circle_equation A.1 A.2 c) ∧
    (B.1 = 0 ∧ circle_equation B.1 B.2 c) ∧
    (angle_APB = 120) →
    c = -11 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_angle_l3618_361874


namespace NUMINAMATH_CALUDE_total_haircut_time_l3618_361859

/-- The time it takes to cut a woman's hair in minutes -/
def womanHairCutTime : ℕ := 50

/-- The time it takes to cut a man's hair in minutes -/
def manHairCutTime : ℕ := 15

/-- The time it takes to cut a kid's hair in minutes -/
def kidHairCutTime : ℕ := 25

/-- The number of women's haircuts Joe performed -/
def numWomenHaircuts : ℕ := 3

/-- The number of men's haircuts Joe performed -/
def numMenHaircuts : ℕ := 2

/-- The number of kids' haircuts Joe performed -/
def numKidsHaircuts : ℕ := 3

/-- Theorem stating the total time Joe spent cutting hair -/
theorem total_haircut_time :
  numWomenHaircuts * womanHairCutTime +
  numMenHaircuts * manHairCutTime +
  numKidsHaircuts * kidHairCutTime = 255 := by
  sorry

end NUMINAMATH_CALUDE_total_haircut_time_l3618_361859


namespace NUMINAMATH_CALUDE_gcd_3150_9800_l3618_361872

theorem gcd_3150_9800 : Nat.gcd 3150 9800 = 350 := by sorry

end NUMINAMATH_CALUDE_gcd_3150_9800_l3618_361872


namespace NUMINAMATH_CALUDE_division_problem_l3618_361842

theorem division_problem (A : ℕ) : A / 3 = 8 ∧ A % 3 = 2 → A = 26 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3618_361842


namespace NUMINAMATH_CALUDE_increased_value_l3618_361816

theorem increased_value (x : ℝ) (p : ℝ) (h1 : x = 1200) (h2 : p = 40) :
  x * (1 + p / 100) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_increased_value_l3618_361816


namespace NUMINAMATH_CALUDE_residue_of_power_mod_13_l3618_361843

theorem residue_of_power_mod_13 : (5 ^ 1234 : ℕ) % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_power_mod_13_l3618_361843


namespace NUMINAMATH_CALUDE_point_relationship_l3618_361895

/-- Given two points A(-1/2, m) and B(2, n) on the line y = 3x + b, prove that m < n -/
theorem point_relationship (m n b : ℝ) : 
  ((-1/2 : ℝ), m) ∈ {(x, y) | y = 3*x + b} →
  ((2 : ℝ), n) ∈ {(x, y) | y = 3*x + b} →
  m < n :=
by sorry

end NUMINAMATH_CALUDE_point_relationship_l3618_361895


namespace NUMINAMATH_CALUDE_probability_cousins_names_l3618_361838

/-- Represents the number of letters in each cousin's name -/
structure NameLengths where
  amelia : ℕ
  bethany : ℕ
  claire : ℕ

/-- The probability of selecting two cards from different cousins' names -/
def probability_different_names (nl : NameLengths) : ℚ :=
  let total := nl.amelia + nl.bethany + nl.claire
  2 * (nl.amelia * nl.bethany + nl.amelia * nl.claire + nl.bethany * nl.claire) / (total * (total - 1))

/-- Theorem stating the probability of selecting two cards from different cousins' names -/
theorem probability_cousins_names :
  let nl : NameLengths := { amelia := 6, bethany := 7, claire := 6 }
  probability_different_names nl = 40 / 57 := by
  sorry


end NUMINAMATH_CALUDE_probability_cousins_names_l3618_361838


namespace NUMINAMATH_CALUDE_mystic_aquarium_fish_duration_l3618_361804

/-- The number of weeks that a given number of fish buckets will last at the Mystic Aquarium -/
def weeks_of_fish (total_buckets : ℕ) : ℕ :=
  let sharks_daily := 4
  let dolphins_daily := sharks_daily / 2
  let others_daily := sharks_daily * 5
  let daily_consumption := sharks_daily + dolphins_daily + others_daily
  let weekly_consumption := daily_consumption * 7
  total_buckets / weekly_consumption

/-- Theorem stating that 546 buckets of fish will last for 3 weeks -/
theorem mystic_aquarium_fish_duration : weeks_of_fish 546 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mystic_aquarium_fish_duration_l3618_361804


namespace NUMINAMATH_CALUDE_complement_A_U_equality_l3618_361886

-- Define the universal set U
def U : Set ℝ := {x | -3 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

-- Define the complement of A with respect to U
def complement_A_U : Set ℝ := U \ A

-- Theorem statement
theorem complement_A_U_equality :
  complement_A_U = {x | (-3 < x ∧ x ≤ -2) ∨ (1 < x ∧ x < 3)} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_U_equality_l3618_361886


namespace NUMINAMATH_CALUDE_expand_expression_l3618_361868

theorem expand_expression (x : ℝ) : (x - 3) * (x + 6) = x^2 + 3*x - 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3618_361868


namespace NUMINAMATH_CALUDE_nara_height_l3618_361891

/-- Given the heights of Sangheon, Chiho, and Nara, prove Nara's height -/
theorem nara_height (sangheon_height : Real) (chiho_diff : Real) (nara_diff : Real)
  (h1 : sangheon_height = 1.56)
  (h2 : chiho_diff = 0.14)
  (h3 : nara_diff = 0.27) :
  sangheon_height - chiho_diff + nara_diff = 1.69 := by
  sorry


end NUMINAMATH_CALUDE_nara_height_l3618_361891


namespace NUMINAMATH_CALUDE_five_attraction_permutations_l3618_361800

theorem five_attraction_permutations : Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_five_attraction_permutations_l3618_361800


namespace NUMINAMATH_CALUDE_line_points_k_value_l3618_361853

theorem line_points_k_value (m n k : ℝ) : 
  (∀ x y, x - 5/2 * y + 1 = 0 ↔ y = 2/5 * x + 2/5) →
  (m - 5/2 * n + 1 = 0) →
  ((m + 1/2) - 5/2 * (n + 1/k) + 1 = 0) →
  (n + 1/k = n + 1) →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_points_k_value_l3618_361853


namespace NUMINAMATH_CALUDE_female_students_count_l3618_361836

theorem female_students_count (total_average : ℚ) (male_count : ℕ) (male_average : ℚ) (female_average : ℚ) :
  total_average = 90 →
  male_count = 8 →
  male_average = 87 →
  female_average = 92 →
  ∃ (female_count : ℕ), 
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 12 := by
  sorry

end NUMINAMATH_CALUDE_female_students_count_l3618_361836


namespace NUMINAMATH_CALUDE_max_value_expression_l3618_361864

theorem max_value_expression (a b c d : ℝ) 
  (ha : -10.5 ≤ a ∧ a ≤ 10.5)
  (hb : -10.5 ≤ b ∧ b ≤ 10.5)
  (hc : -10.5 ≤ c ∧ c ≤ 10.5)
  (hd : -10.5 ≤ d ∧ d ≤ 10.5) :
  (∀ x y z w, -10.5 ≤ x ∧ x ≤ 10.5 → -10.5 ≤ y ∧ y ≤ 10.5 → 
              -10.5 ≤ z ∧ z ≤ 10.5 → -10.5 ≤ w ∧ w ≤ 10.5 →
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 462) ∧
  (∃ x y z w, -10.5 ≤ x ∧ x ≤ 10.5 ∧ -10.5 ≤ y ∧ y ≤ 10.5 ∧
              -10.5 ≤ z ∧ z ≤ 10.5 ∧ -10.5 ≤ w ∧ w ≤ 10.5 ∧
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 462) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3618_361864


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l3618_361894

-- Equation 1
theorem solve_equation_one : 
  ∃ x : ℚ, (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 ∧ x = -20 := by sorry

-- Equation 2
theorem solve_equation_two : 
  ∃ x : ℚ, (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 3 ∧ x = 67 / 23 := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l3618_361894


namespace NUMINAMATH_CALUDE_impossible_equal_sums_l3618_361808

/-- A configuration of numbers on a triangle with medians -/
structure TriangleConfig where
  vertices : Fin 3 → ℕ
  midpoints : Fin 3 → ℕ
  center : ℕ

/-- The sum of numbers on a side of the triangle -/
def side_sum (config : TriangleConfig) (i : Fin 3) : ℕ :=
  config.vertices i + config.midpoints i + config.vertices ((i + 1) % 3)

/-- The sum of numbers on a median of the triangle -/
def median_sum (config : TriangleConfig) (i : Fin 3) : ℕ :=
  config.vertices i + config.midpoints ((i + 1) % 3) + config.center

/-- Predicate to check if a configuration is valid -/
def is_valid_config (config : TriangleConfig) : Prop :=
  (∀ i : Fin 3, config.vertices i ≤ 7) ∧
  (∀ i : Fin 3, config.midpoints i ≤ 7) ∧
  (config.center ≤ 7) ∧
  (config.vertices 0 + config.vertices 1 + config.vertices 2 +
   config.midpoints 0 + config.midpoints 1 + config.midpoints 2 +
   config.center = 28)

/-- Predicate to check if a configuration has equal sums -/
def has_equal_sums (config : TriangleConfig) : Prop :=
  ∃ x : ℕ, (∀ i : Fin 3, side_sum config i = x) ∧
            (∀ i : Fin 3, median_sum config i = x)

theorem impossible_equal_sums : ¬∃ config : TriangleConfig, 
  is_valid_config config ∧ has_equal_sums config := by
  sorry

end NUMINAMATH_CALUDE_impossible_equal_sums_l3618_361808
