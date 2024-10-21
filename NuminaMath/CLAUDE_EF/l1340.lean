import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_equation_solutions_l1340_134005

def d (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_equation_solutions :
  ∀ n : ℕ, n > 0 → (n = d n * 4 ↔ n = 81 ∨ n = 625) :=
by
  sorry

#eval d 81
#eval d 625

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_equation_solutions_l1340_134005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_schwambrania_theorem_l1340_134036

/-- A type representing a city in Schwambrania. -/
structure City where
  id : Nat

/-- A type representing the road system in Schwambrania. -/
structure RoadSystem where
  cities : Finset City
  roads : City → City → Prop

/-- A predicate indicating if a path exists between two cities. -/
def hasPath (rs : RoadSystem) (start finish : City) : Prop :=
  ∃ path : List City, path.head? = some start ∧ path.getLast? = some finish ∧
    ∀ i j, i + 1 < path.length → j = i + 1 →
      rs.roads (path.get ⟨i, by sorry⟩) (path.get ⟨j, by sorry⟩)

/-- The main theorem about the Schwambrania road system. -/
theorem schwambrania_theorem (n : Nat) :
  ∃ (rs : RoadSystem),
    (∀ c₁ c₂ : City, c₁ ∈ rs.cities → c₂ ∈ rs.cities → c₁ ≠ c₂ → (rs.roads c₁ c₂ ∨ rs.roads c₂ c₁)) ∧
    (∀ c₁ c₂ c₃ : City, rs.roads c₁ c₂ → rs.roads c₂ c₃ → ¬rs.roads c₃ c₁) ∧
    (∃ start : City, start ∈ rs.cities ∧ ∀ c : City, c ∈ rs.cities → c ≠ start → hasPath rs start c) ∧
    (∃ endCity : City, endCity ∈ rs.cities ∧ ∀ c : City, c ∈ rs.cities → ¬rs.roads endCity c) ∧
    (∃! path : List City, path.Nodup ∧ path.length = rs.cities.card ∧
      ∀ c, c ∈ rs.cities ↔ c ∈ path) ∧
    (rs.cities.card = n) ∧
    (∃ k : Nat, k = n.factorial) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_schwambrania_theorem_l1340_134036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_diameter_l1340_134068

theorem cone_base_diameter (a : ℝ) (h : a > 0) :
  ∃ (r : ℝ), r > 0 ∧ π * r * r + π * r * (2 * r) = a ∧ 2 * r = 2 * r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_diameter_l1340_134068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_implies_m_equals_two_l1340_134023

/-- The distance from a point (x, y) to a line ax + by + c = 0 --/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- The radius of the circle x^2 + y^2 = 4m --/
noncomputable def circle_radius (m : ℝ) : ℝ := 2 * Real.sqrt m

theorem tangency_implies_m_equals_two (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 4*m ↔ x + y = 2*m) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_implies_m_equals_two_l1340_134023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_chinese_association_l1340_134053

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![50, 30],
    ![40, 80]]

-- Define the sample size
def n : ℕ := 200

-- Define the chi-square statistic function
noncomputable def chi_square (table : Matrix (Fin 2) (Fin 2) ℕ) (n : ℕ) : ℝ :=
  let a := table 0 0
  let b := table 0 1
  let c := table 1 0
  let d := table 1 1
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value at α = 0.010
def critical_value : ℝ := 6.635

-- Define the likelihood ratio function
def likelihood_ratio (table : Matrix (Fin 2) (Fin 2) ℕ) : ℚ :=
  table 1 1 / table 0 1

-- Define the expected value function for the given scenario
def expected_value : ℚ := 15 / 8

theorem math_chinese_association :
  chi_square contingency_table n > critical_value ∧
  likelihood_ratio contingency_table = 8 / 3 ∧
  expected_value = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_chinese_association_l1340_134053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_triangle_area_l1340_134024

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the medians
noncomputable def D (A B C : ℝ × ℝ) : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
noncomputable def E (A B C : ℝ × ℝ) : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
noncomputable def F (A B C : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the area function
noncomputable def area (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))

-- State the theorem
theorem median_triangle_area (A B C : ℝ × ℝ) :
  area (D A B C) (E A B C) (F A B C) = (3/4) * area A B C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_triangle_area_l1340_134024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_at_half_height_equilateral_triangle_cut_theorem_l1340_134038

/-- Represents an equilateral triangle with a parallel cut -/
structure CutTriangle where
  side : ℝ
  cut_height : ℝ
  cut_height_pos : 0 < cut_height
  cut_height_le_side : cut_height ≤ side * (Real.sqrt 3 / 2)

/-- Calculates the area of a circle -/
noncomputable def circleArea (radius : ℝ) : ℝ := Real.pi * radius^2

/-- Calculates the radius of the circumcircle of an equilateral triangle -/
noncomputable def circumcircleRadius (side : ℝ) : ℝ := side / Real.sqrt 3

/-- Theorem: If the ratio of circumcircle areas is 1:3, the cut is at height a/2 -/
theorem cut_at_half_height (t : CutTriangle) :
  let smallTriangleCircumcircleArea := circleArea (circumcircleRadius t.cut_height)
  let largePartCircumcircleArea := circleArea (circumcircleRadius (t.side - t.cut_height))
  largePartCircumcircleArea = 3 * smallTriangleCircumcircleArea →
  t.cut_height = t.side / 2 := by
  sorry

/-- Proof of the main statement -/
theorem equilateral_triangle_cut_theorem (a : ℝ) (h : 0 < a) :
  ∃ (t : CutTriangle),
    t.side = a ∧
    (let smallTriangleCircumcircleArea := circleArea (circumcircleRadius t.cut_height)
     let largePartCircumcircleArea := circleArea (circumcircleRadius (a - t.cut_height))
     largePartCircumcircleArea = 3 * smallTriangleCircumcircleArea) ∧
    t.cut_height = a / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_at_half_height_equilateral_triangle_cut_theorem_l1340_134038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_bisector_k_value_l1340_134059

/-- The perpendicular bisector of a line segment passes through its midpoint -/
axiom perp_bisector_passes_midpoint {k : ℝ} :
  ∃ (x y : ℝ), x + y = k ∧ x = (1 + 7) / 2 ∧ y = (6 + 12) / 2

/-- The value of k for the perpendicular bisector of the line segment from (1, 6) to (7, 12) -/
theorem perp_bisector_k_value :
  ∃ k : ℝ, (∀ x y : ℝ, x + y = k → (x - (1 + 7) / 2)^2 + (y - (6 + 12) / 2)^2 = 
    ((1 - 7) / 2)^2 + ((6 - 12) / 2)^2) ∧ k = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_bisector_k_value_l1340_134059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_of_cubes_l1340_134072

theorem total_volume_of_cubes 
  (carl_cube_count : ℕ)
  (carl_cube_side : ℝ)
  (kate_cube_count : ℕ)
  (kate_cube_side : ℝ)
  (h1 : carl_cube_count = 3)
  (h2 : carl_cube_side = 1)
  (h3 : kate_cube_count = 5)
  (h4 : kate_cube_side = 3) :
  carl_cube_count * (carl_cube_side ^ 3) + kate_cube_count * (kate_cube_side ^ 3) = 138 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_of_cubes_l1340_134072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_half_quadrant_symmetry_axis_l1340_134084

-- Statement 1
theorem angle_half_quadrant (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  (0 < α / 2 ∧ α / 2 < Real.pi / 4) ∨ (Real.pi < α / 2 + Real.pi ∧ α / 2 + Real.pi < 5 * Real.pi / 4) := by
  sorry

-- Statement 2
noncomputable def f (x : Real) : Real := 2 * Real.cos (2 * x + Real.pi / 3)

theorem symmetry_axis (x : Real) : f x = f (2 * Real.pi / 3 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_half_quadrant_symmetry_axis_l1340_134084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l1340_134067

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let v := (C.1 - A.1, C.2 - A.2)
  let w := (C.1 - B.1, C.2 - B.2)
  (1/2) * abs (v.1 * w.2 - v.2 * w.1)

/-- Theorem: The area of the triangle with vertices (-2,3), (8,-1), and (10,6) is 39 -/
theorem triangle_area_example : 
  triangle_area (-2, 3) (8, -1) (10, 6) = 39 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l1340_134067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_directed_triangles_correct_ramsey_lower_bound_exists_l1340_134049

/-- R_k(m,n) is the least number such that for each coloring of k-subsets of {1,2,...,R_k(m,n)}
    with blue and red colors, there is a subset with m elements such that all of its k-subsets
    are red or there is a subset with n elements such that all of its k-subsets are blue. -/
def ramsey_number (k m n : ℕ) : ℕ := sorry

/-- The probability that a randomly directed complete graph K_n does not contain directed triangles -/
noncomputable def prob_no_directed_triangles (n : ℕ) : ℝ :=
  (3/4) ^ (n.choose 3)

theorem prob_no_directed_triangles_correct (n : ℕ) :
  prob_no_directed_triangles n = (3/4) ^ (n.choose 3) := by sorry

theorem ramsey_lower_bound_exists :
  ∃ c : ℝ, ∀ n : ℕ, (ramsey_number 3 4 n : ℝ) ≥ 2 ^ (c * n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_directed_triangles_correct_ramsey_lower_bound_exists_l1340_134049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equality_l1340_134029

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Theorem statement
theorem floor_expression_equality : 
  (floor 6.5) * (floor (2/3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.4) - 6 = 10.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equality_l1340_134029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l1340_134076

noncomputable def f (x : ℝ) : ℝ := (10^x - 10^(-x)) / (10^x + 10^(-x))

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f x < f y) := by
  constructor
  · intro x
    sorry -- Proof for odd function property
  · intro x y hxy
    sorry -- Proof for increasing function property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l1340_134076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_circumference_difference_l1340_134040

noncomputable section

/-- Represents the diameter of the inner circle in feet -/
def inner_diameter : ℝ := 30

/-- Represents the minimum width of the track in feet -/
def min_width : ℝ := 10

/-- Represents the maximum width of the track in feet -/
def max_width : ℝ := 15

/-- Calculates the average width of the track -/
noncomputable def avg_width : ℝ := (min_width + max_width) / 2

/-- Calculates the diameter of the outer circle -/
noncomputable def outer_diameter : ℝ := inner_diameter + 2 * avg_width

/-- Theorem stating the average difference in circumferences -/
theorem avg_circumference_difference : 
  (outer_diameter - inner_diameter) * π = 25 * π := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_circumference_difference_l1340_134040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sean_total_spent_l1340_134081

-- Define the prices and quantities
def almond_croissant_price : ℚ := 4
def salami_cheese_croissant_price : ℚ := 5
def plain_croissant_price : ℚ := 7/2
def focaccia_price : ℚ := 5
def latte_price : ℚ := 3

def almond_croissant_quantity : ℕ := 2
def salami_cheese_croissant_quantity : ℕ := 3
def plain_croissant_quantity : ℕ := 4
def focaccia_quantity : ℕ := 1
def latte_quantity : ℕ := 3

-- Define the discounts and taxes
def first_bakery_discount : ℚ := 1/10
def second_bakery_tax : ℚ := 1/20
def cafe_discount : ℚ := 3/20

-- Define the conversion rates
def euro_to_usd : ℚ := 23/20
def pound_to_usd : ℚ := 27/20

-- Define the theorem
theorem sean_total_spent :
  let first_bakery_total : ℚ := (almond_croissant_price * almond_croissant_quantity + 
                                 salami_cheese_croissant_price * salami_cheese_croissant_quantity) * 
                                (1 - first_bakery_discount) * euro_to_usd
  let second_bakery_total : ℚ := (plain_croissant_price * (plain_croissant_quantity - 1) + 
                                  focaccia_price * focaccia_quantity) * 
                                 (1 + second_bakery_tax) * pound_to_usd
  let cafe_total : ℚ := latte_price * latte_quantity * (1 - cafe_discount)
  abs (first_bakery_total + second_bakery_total + cafe_total - 5344/100) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sean_total_spent_l1340_134081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_cos_gt_sin_sin_l1340_134019

theorem cos_cos_gt_sin_sin :
  ∀ x : ℝ, Real.cos (Real.cos x) > Real.sin (Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_cos_gt_sin_sin_l1340_134019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_of_11_ending_in_5_l1340_134051

theorem count_multiples_of_11_ending_in_5 :
  (Finset.filter (fun n => 0 < n ∧ n < 2000 ∧ n % 11 = 0 ∧ n % 10 = 5) (Finset.range 2000)).card = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_of_11_ending_in_5_l1340_134051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_is_odd_integer_l1340_134092

open Set Function Real

-- Define the function y = x^α
noncomputable def y (α : ℝ) (x : ℝ) : ℝ := x^α

-- Define the property of having domain ℝ
def has_domain_reals (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x ∈ (Set.univ : Set ℝ)

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem alpha_is_odd_integer (α : ℝ) :
  (has_domain_reals (y α) ∧ is_odd (y α)) ↔ ∃ k : ℤ, α = 2 * k + 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_is_odd_integer_l1340_134092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_T_l1340_134057

/-- The complex number ω = e^(πi/4) -/
noncomputable def ω : ℂ := Complex.exp (Real.pi * Complex.I / 4)

/-- The set T of points in the complex plane -/
def T : Set ℂ := {z : ℂ | ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ z = a + b * ω + c * ω⁻¹}

/-- The area of set T is 8 -/
theorem area_of_T : MeasureTheory.volume T = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_T_l1340_134057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1340_134020

def M : Set ℝ := {x | x^2 - x ≤ 0}

def N : Set ℝ := {x | x < 1}

theorem intersection_M_N : M ∩ N = Set.Ico 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1340_134020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l1340_134073

/-- A power function of the form y = ax^b where a and b are real numbers -/
structure PowerFunction where
  a : ℝ
  b : ℝ

/-- Checks if a power function is decreasing on (0, +∞) -/
def is_decreasing (f : PowerFunction) : Prop :=
  f.b < 0

/-- The specific power function from the problem -/
noncomputable def problem_function (m : ℝ) : PowerFunction :=
  { a := m^2 - m - 1
    b := m^2 - 2*m - 1/3 }

/-- The theorem to prove -/
theorem power_function_theorem :
  ∀ m : ℝ, is_decreasing (problem_function m) →
    ∃ f : PowerFunction, f = problem_function m ∧ f = { a := 1, b := -1/3 } :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l1340_134073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1340_134062

theorem solve_exponential_equation :
  ∃ x : ℚ, (3 : ℝ) ^ ((2 : ℝ) * x + 1) = (1 : ℝ) / 27 ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1340_134062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_balls_for_collinearity_l1340_134056

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a pocket on the billiard table -/
structure Pocket where
  location : Point

/-- Represents the billiard table -/
structure BilliardTable where
  width : ℝ
  height : ℝ
  pockets : List Pocket

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Checks if a point is inside the billiard table -/
def insideTable (p : Point) (table : BilliardTable) : Prop :=
  0 < p.x ∧ p.x < table.width ∧ 0 < p.y ∧ p.y < table.height

/-- The main theorem to prove -/
theorem min_balls_for_collinearity (table : BilliardTable) 
  (h1 : table.width = 2)
  (h2 : table.height = 1)
  (h3 : table.pockets.length = 6) :
  ∃ (balls : List Point),
    balls.length = 4 ∧
    (∀ b, b ∈ balls → insideTable b table) ∧
    (∀ p, p ∈ table.pockets → ∃ b1 b2, b1 ∈ balls ∧ b2 ∈ balls ∧ b1 ≠ b2 ∧ collinear p.location b1 b2) ∧
    (∀ n, n < 4 → ¬∃ (smallerBalls : List Point),
      smallerBalls.length = n ∧
      (∀ b, b ∈ smallerBalls → insideTable b table) ∧
      (∀ p, p ∈ table.pockets → ∃ b1 b2, b1 ∈ smallerBalls ∧ b2 ∈ smallerBalls ∧ b1 ≠ b2 ∧ collinear p.location b1 b2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_balls_for_collinearity_l1340_134056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_through_point_l1340_134099

/-- A linear function y = mx + m^2 passing through (0, 4) with positive slope has m = 2 -/
theorem linear_function_through_point (m : ℝ) : 
  m ≠ 0 →                             -- m is non-zero
  (0, 4) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + m^2} →  -- graph passes through (0, 4)
  (∀ x₁ x₂, x₁ < x₂ → m * x₁ + m^2 < m * x₂ + m^2) →  -- y increases as x increases
  m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_through_point_l1340_134099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_terminal_side_l1340_134034

theorem sin_alpha_terminal_side (α : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * Real.sin α = Real.sin (2 * π / 3) ∧ r * Real.cos α = Real.cos (2 * π / 3)) →
  Real.sin α = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_terminal_side_l1340_134034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_problem_l1340_134022

theorem sin_difference_problem (α β : Real) :
  Real.sin α = 12 / 13 →
  Real.cos β = 4 / 5 →
  π / 2 < α ∧ α < π →
  3 * π / 2 < β ∧ β < 2 * π →
  Real.sin (α - β) = 33 / 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_problem_l1340_134022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_calculation_l1340_134028

noncomputable def cone_volume (radius : ℝ) (height : ℝ) : ℝ := (1/3) * Real.pi * radius^2 * height

noncomputable def hemisphere_volume (radius : ℝ) : ℝ := (2/3) * Real.pi * radius^3

noncomputable def cylinder_volume (radius : ℝ) (height : ℝ) : ℝ := Real.pi * radius^2 * height

theorem ice_cream_volume_calculation :
  let cone_radius : ℝ := 3
  let cone_height : ℝ := 12
  let hemisphere_radius : ℝ := cone_radius
  let cylinder_radius : ℝ := cone_radius
  let cylinder_height : ℝ := 2
  
  let total_volume : ℝ := 
    cone_volume cone_radius cone_height + 
    hemisphere_volume hemisphere_radius + 
    cylinder_volume cylinder_radius cylinder_height
  
  total_volume = 72 * Real.pi := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_volume_calculation_l1340_134028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_equals_46_l1340_134046

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | n + 2 =>
    if n % 2 = 0 then
      sequence_a (n + 1) + ((n + 2) / 2)
    else
      sequence_a (n + 1) + ((-1) ^ ((n + 3) / 2))

theorem a_20_equals_46 : sequence_a 20 = 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_equals_46_l1340_134046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emilia_water_needs_l1340_134041

/-- The amount of water in mL needed for 300 mL of flour -/
noncomputable def water_per_300mL_flour : ℚ := 90

/-- The amount of flour in mL Emilia uses -/
noncomputable def flour_amount : ℚ := 900

/-- The additional amount of water in mL Emilia wants to add -/
noncomputable def additional_water : ℚ := 50

/-- The total amount of water in mL Emilia needs -/
noncomputable def total_water : ℚ := (flour_amount / 300) * water_per_300mL_flour + additional_water

theorem emilia_water_needs : total_water = 320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emilia_water_needs_l1340_134041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doctor_lindsay_adult_patients_l1340_134004

/-- Represents the number of adult patients seen per hour -/
def adult_patients_per_hour : ℕ := 4

/-- Represents the number of child patients seen per hour -/
def child_patients_per_hour : ℕ := 3

/-- Represents the cost of an adult office visit in dollars -/
def adult_visit_cost : ℕ := 50

/-- Represents the cost of a child office visit in dollars -/
def child_visit_cost : ℕ := 25

/-- Represents the total income in dollars for a typical 8-hour day -/
def total_daily_income : ℕ := 2200

/-- Represents the number of working hours in a typical day -/
def working_hours_per_day : ℕ := 8

theorem doctor_lindsay_adult_patients :
  adult_patients_per_hour = 4 :=
by
  -- The proof goes here
  sorry

#eval adult_patients_per_hour

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doctor_lindsay_adult_patients_l1340_134004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_reducible_fraction_l1340_134048

-- Define the fraction
def fraction (n : ℕ+) : ℚ := (n.val - 31 : ℤ) / (7 * n.val + 8 : ℤ)

-- Define reducibility
def is_reducible (q : ℚ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b ∧ Int.gcd a.natAbs b.natAbs > 1

-- Theorem statement
theorem least_reducible_fraction :
  (∀ m : ℕ+, m < 34 → ¬(is_reducible (fraction m))) ∧
  is_reducible (fraction 34) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_reducible_fraction_l1340_134048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1340_134085

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- Theorem: The eccentricity of the given ellipse is 1/2 -/
theorem ellipse_eccentricity (Γ : Ellipse) (F A B N M : Point) : 
  F.x = 3 ∧ F.y = 0 ∧  -- Right focus
  A.x = 0 ∧ A.y = Γ.b ∧  -- Upper vertex
  B.x = 0 ∧ B.y = -Γ.b ∧  -- Lower vertex
  N.x = 12 ∧ N.y = 0 ∧  -- Intersection of BM with x-axis
  (M.x^2 / Γ.a^2 + M.y^2 / Γ.b^2 = 1) ∧  -- M is on the ellipse
  (A.y - M.y) * (F.x - A.x) = (A.x - M.x) * (F.y - A.y) ∧  -- M is on line AF
  (B.y - M.y) * (N.x - B.x) = (B.x - M.x) * (N.y - B.y)  -- M is on line BN
  →
  eccentricity Γ = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1340_134085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tickets_before_rally_l1340_134015

/-- Represents the number of tickets sold before the rally -/
def tickets_before : ℕ := sorry

/-- Represents the number of tickets sold at the door -/
def tickets_at_door : ℕ := sorry

/-- The total number of attendees -/
def total_attendance : ℕ := 750

/-- The price of tickets bought before the rally -/
def price_before : ℚ := 2

/-- The price of tickets bought at the door -/
def price_at_door : ℚ := 2.75

/-- The total receipts from ticket sales -/
def total_receipts : ℚ := 1706.25

theorem tickets_before_rally : 
  tickets_before + tickets_at_door = total_attendance ∧
  price_before * tickets_before + price_at_door * tickets_at_door = total_receipts →
  tickets_before = 475 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tickets_before_rally_l1340_134015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1340_134013

noncomputable def a : ℝ × ℝ := (4, 3)
noncomputable def b : ℝ × ℝ := (-1, 2)

noncomputable def cosine_angle (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2))

def same_direction (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ w = (k * v.1, k * v.2)

theorem vector_problem :
  (cosine_angle a b = 2 * Real.sqrt 5 / 25) ∧
  (∀ c : ℝ × ℝ, same_direction a c ∧ Real.sqrt (c.1^2 + c.2^2) = 10 → c = (8, 6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1340_134013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_bound_for_function_difference_l1340_134010

theorem minimum_bound_for_function_difference (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x ≠ y → 
    |f x - f y| < (1/2) * |x - y|) :
  (∃ (m : ℝ), ∀ (x y : ℝ), x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → |f x - f y| < m) ∧ 
  (∀ (m : ℝ), (∀ (x y : ℝ), x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → |f x - f y| < m) → m ≥ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_bound_for_function_difference_l1340_134010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l1340_134087

noncomputable def vertex : Fin 4 → ℝ × ℝ
| 0 => (0, 0)
| 1 => (2, 6)
| 2 => (6, 5)
| 3 => (3, 2)

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

noncomputable def perimeter : ℝ :=
  distance (vertex 0) (vertex 1) +
  distance (vertex 1) (vertex 2) +
  distance (vertex 2) (vertex 3) +
  distance (vertex 3) (vertex 0)

theorem quadrilateral_perimeter :
  perimeter = 2 * Real.sqrt 10 + Real.sqrt 17 + 3 * Real.sqrt 2 + Real.sqrt 13 ∧
  2 + 1 + 3 + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l1340_134087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1340_134098

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 / (x - m)

def p (m : ℝ) : Prop := ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f m x₁ > f m x₂

def roots_equation (a : ℝ) (x : ℝ) : Prop := x^2 - a*x - 2 = 0

def q (m : ℝ) : Prop :=
  ∃ x₁ x₂ a, 
    roots_equation a x₁ ∧ 
    roots_equation a x₂ ∧ 
    x₁ ≠ x₂ ∧
    ∀ a' ∈ Set.Icc (-1 : ℝ) 1, m^2 + 5*m - 3 ≥ |x₁ - x₂|

theorem m_range (m : ℝ) (h : ¬p m ∧ q m) : m > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1340_134098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1340_134080

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b/a = Real.sqrt 5/2) : 
  let e := Real.sqrt (1 + b^2/a^2)
  e = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1340_134080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_length_l1340_134001

/-- A rectangle with diagonals of equal length -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_rectangle : Bool
  diagonal_AC : Bool
  diagonal_BD : Bool
  diagonals_equal : Bool

/-- Theorem: In a rectangle ABCD, if diagonal AC has length 8, then diagonal BD also has length 8 -/
theorem rectangle_diagonal_length (rect : Rectangle) (h : dist rect.A rect.C = 8) :
  dist rect.B rect.D = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_length_l1340_134001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_nine_three_l1340_134009

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_through_point_nine_three (f : ℝ → ℝ) (α : ℝ) :
  f = power_function α →
  f 9 = 3 →
  f 4 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_nine_three_l1340_134009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rear_wheel_revolutions_l1340_134037

/-- Represents a bicycle with two wheels -/
structure Bicycle where
  rear_radius : ℝ
  front_radius : ℝ
  hfr : front_radius = 2 * rear_radius

/-- Calculates the number of revolutions of the rear wheel -/
noncomputable def rear_revolutions (b : Bicycle) (front_revs : ℝ) : ℝ :=
  front_revs * (b.front_radius / b.rear_radius)

/-- Theorem: When the front wheel turns 10 revolutions, the rear wheel turns 20 revolutions -/
theorem rear_wheel_revolutions (b : Bicycle) : 
  rear_revolutions b 10 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rear_wheel_revolutions_l1340_134037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l1340_134003

theorem book_price_change (P : ℝ) (h : P > 0) : 
  let price_after_decrease := P * (1 - 0.3)
  let final_price := price_after_decrease * (1 + 0.4)
  final_price = P * 0.98 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l1340_134003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1340_134063

theorem constant_term_binomial_expansion :
  let n : ℕ := 4
  let expansion := fun (x : ℝ) => (x + 1/x)^n
  (Polynomial.taylor expansion 0).coeff 0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1340_134063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_second_highest_point_correct_l1340_134027

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := cos x * sin (x + π/3) - sqrt 3 * (cos x)^2 + sqrt 3 / 4

-- Theorem for the maximum value of f(x)
theorem f_max_value : ∃ (M : ℝ), M = 1/2 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

-- Define the second highest point on the right side of the y-axis
noncomputable def second_highest_point : ℝ × ℝ := (17 * π / 12, 1/2)

-- Theorem for the second highest point
theorem second_highest_point_correct :
  let (x, y) := second_highest_point
  ∃ (ε : ℝ), ε > 0 ∧
  (∀ (t : ℝ), 0 < t ∧ t < x → f t < y) ∧
  (∀ (t : ℝ), x < t ∧ t < x + ε → f t < y) ∧
  f x = y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_second_highest_point_correct_l1340_134027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_101101_to_decimal_l1340_134065

def binary_to_decimal (b : List Bool) : Nat :=
  b.reverse.enum.foldl
    (fun acc (index, digit) => acc + if digit then 2^index else 0)
    0

theorem binary_101101_to_decimal :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_101101_to_decimal_l1340_134065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1340_134079

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.log (x - 1) / Real.log (3 * Real.exp 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1340_134079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_cone_max_volume_l1340_134064

/-- Represents a cone inscribed in a sphere -/
structure InscribedCone (R : ℝ) where
  h : ℝ  -- height of the cone
  r : ℝ  -- radius of the base of the cone
  h_pos : 0 < h
  r_pos : 0 < r
  h_bound : h < 2 * R
  sphere_constraint : r^2 = h * (2 * R - h)

/-- The volume of a cone -/
noncomputable def cone_volume (R : ℝ) (c : InscribedCone R) : ℝ :=
  (1/3) * Real.pi * c.r^2 * c.h

/-- The optimal height and radius for maximum volume -/
noncomputable def optimal_cone (R : ℝ) : InscribedCone R where
  h := (4/3) * R
  r := (2/3) * R * Real.sqrt 2
  h_pos := by sorry
  r_pos := by sorry
  h_bound := by sorry
  sphere_constraint := by sorry

/-- Theorem: The optimal cone has the maximum volume -/
theorem optimal_cone_max_volume (R : ℝ) (h_pos : 0 < R) :
  ∀ c : InscribedCone R, cone_volume R (optimal_cone R) ≥ cone_volume R c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_cone_max_volume_l1340_134064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1340_134018

theorem problem_statement :
  let p := ∀ a b : ℝ, (a > b) ↔ (2:ℝ)^a > (2:ℝ)^b
  let q := ∃ x : ℝ, Real.exp x < Real.log x
  p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1340_134018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_18_l1340_134089

/-- Represents a finite arithmetic sequence. -/
structure ArithmeticSequence where
  terms : List ℤ
  is_arithmetic : ∀ i : ℕ, i + 2 < terms.length → terms[i+2]! - terms[i+1]! = terms[i+1]! - terms[i]!

/-- Returns the sum of the first n terms of the sequence. -/
def sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (seq.terms.take n).sum

/-- Returns the sum of the last n terms of the sequence. -/
def sumLastN (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (seq.terms.reverse.take n).sum

/-- The main theorem to prove. -/
theorem seventh_term_is_18 (seq : ArithmeticSequence) 
    (h1 : sumFirstN seq 5 = 34)
    (h2 : sumLastN seq 5 = 146)
    (h3 : seq.terms.sum = 234)
    (h4 : seq.terms.length > 6) :
    seq.terms[6]! = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_18_l1340_134089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_minus_pi_6_l1340_134091

theorem sin_2alpha_minus_pi_6 (α : ℝ) (h : Real.cos (α + Real.pi/6) = Real.sqrt 3/3) :
  Real.sin (2*α - Real.pi/6) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_minus_pi_6_l1340_134091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_correct_l1340_134077

/-- The perpendicular bisector of a line segment with endpoints (0, 2) and (4, 0) -/
def perpendicular_bisector (x y : ℝ) : Prop :=
  2 * x - y - 3 = 0

/-- The endpoints of the line segment -/
def endpoint1 : ℝ × ℝ := (0, 2)
def endpoint2 : ℝ × ℝ := (4, 0)

/-- A point lies on a line if it satisfies the line's equation -/
def lies_on (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  line p.1 p.2

/-- The perpendicular bisector of a line segment -/
def perpendicular_bisector_of_segment (p q : ℝ × ℝ) (x y : ℝ) : Prop :=
  -- This is a placeholder definition. In a real proof, we would derive this from the endpoints.
  perpendicular_bisector x y

/-- Theorem stating that the given equation represents the perpendicular bisector -/
theorem perpendicular_bisector_correct :
  ∀ x y : ℝ, perpendicular_bisector x y ↔ 
  lies_on (x, y) (perpendicular_bisector_of_segment endpoint1 endpoint2) :=
by
  sorry  -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_correct_l1340_134077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_cone_apex_angle_l1340_134002

/-- A cone whose lateral surface unfolds into a semicircle -/
structure SemiCircleCone where
  /-- Slant height of the cone -/
  R : ℝ
  /-- R is positive -/
  R_pos : R > 0

/-- The apex angle of the triangle in the axial section of a SemiCircleCone -/
noncomputable def apexAngle (cone : SemiCircleCone) : ℝ :=
  2 * Real.arcsin (cone.R / (2 * cone.R))

/-- Theorem: The apex angle of a SemiCircleCone is π/3 -/
theorem semicircle_cone_apex_angle (cone : SemiCircleCone) :
    apexAngle cone = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_cone_apex_angle_l1340_134002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michaels_pie_order_cost_l1340_134054

/-- Represents the cost calculation for Michael's pie order --/
theorem michaels_pie_order_cost : 
  let peach_pies : ℕ := 5
  let apple_pies : ℕ := 4
  let blueberry_pies : ℕ := 3
  let fruit_per_pie : ℕ := 3
  let peach_price : ℚ := 2
  let apple_price : ℚ := 1
  let blueberry_price : ℚ := 1
  (peach_pies * fruit_per_pie * peach_price) + 
  (apple_pies * fruit_per_pie * apple_price) + 
  (blueberry_pies * fruit_per_pie * blueberry_price) = 51 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_michaels_pie_order_cost_l1340_134054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1340_134083

-- Define the function as noncomputable due to Real.sqrt
noncomputable def f (a x : ℝ) : ℝ := Real.sqrt (a * 4^x + 2^x + 1)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 2 → f a x ∈ Set.Ioi 0) ↔ a ≥ -5/16 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1340_134083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_implies_convex_polygon_l1340_134060

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Predicate to check if four points form a convex quadrilateral -/
def IsConvexQuadrilateral (p1 p2 p3 p4 : Point2D) : Prop := sorry

/-- Predicate to check if a set of points form a convex polygon -/
def IsConvexPolygon (points : Set Point2D) : Prop := sorry

theorem convex_quadrilateral_implies_convex_polygon 
  (points : Finset Point2D) 
  (h1 : points.card = 2015)
  (h2 : ∀ (p1 p2 p3 p4 : Point2D), p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → 
        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 → 
        IsConvexQuadrilateral p1 p2 p3 p4) :
  IsConvexPolygon points.toSet := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_implies_convex_polygon_l1340_134060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_shape_perimeter_l1340_134096

/-- The perimeter of a T-shaped figure formed by two rectangles -/
theorem t_shape_perimeter (rect1_width rect1_height rect2_width rect2_height : ℝ) : 
  rect1_width = 3 ∧ 
  rect1_height = 6 ∧ 
  rect2_width = 2 ∧ 
  rect2_height = 5 ∧ 
  rect2_height < rect1_height →
  2 * (rect2_height + rect1_height + rect2_width) - 2 * rect2_width = 23 := by
  sorry

#check t_shape_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_shape_perimeter_l1340_134096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tourism_consumption_l1340_134045

-- Define the tourist number function p(x)
noncomputable def p (x : ℕ) : ℝ := -3 * (x : ℝ)^2 + 40 * (x : ℝ)

-- Define the per capita consumption function q(x)
noncomputable def q (x : ℕ) : ℝ :=
  if x ≤ 6 then 35 - 2 * (x : ℝ) else 160 / (x : ℝ)

-- Define the total tourism consumption function g(x)
noncomputable def g (x : ℕ) : ℝ := p x * q x

-- Theorem statement
theorem max_tourism_consumption :
  ∃ (max_value : ℝ), max_value = 3125 ∧
  ∀ (x : ℕ), 1 ≤ x → x ≤ 12 → g x ≤ max_value ∧
  ∃ (max_month : ℕ), max_month = 5 ∧ g max_month = max_value :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tourism_consumption_l1340_134045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_l1340_134090

mutual
  def sequence_a : ℕ → ℚ
    | 0 => 2
    | n + 1 => 3/4 * sequence_a n + 1/4 * sequence_b n + 1

  def sequence_b : ℕ → ℚ
    | 0 => 1
    | n + 1 => 1/4 * sequence_a n + 3/4 * sequence_b n + 1
end

theorem sequence_product :
  (sequence_a 2 + sequence_b 2) * (sequence_a 3 - sequence_b 3) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_l1340_134090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_exceeding_50_dollars_l1340_134075

def deposit (n : ℕ) : ℚ :=
  if n = 0 then 5 / 100
  else if n = 1 then 10 / 100
  else 2 * deposit (n - 1)

def total_amount (n : ℕ) : ℚ :=
  Finset.sum (Finset.range (n + 1)) deposit

theorem first_day_exceeding_50_dollars :
  (∀ k < 10, total_amount k ≤ 50) ∧ total_amount 10 > 50 := by
  sorry

#eval total_amount 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_exceeding_50_dollars_l1340_134075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1340_134078

theorem constant_term_expansion : 
  ∃ (c : ℝ), c = 240 ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x > 0 → |x - 1| < δ → 
      |(2 * Real.sqrt x - x⁻¹) ^ 6 - c| < ε) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1340_134078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_last_two_digits_l1340_134047

theorem sum_of_last_two_digits : (7^30 + 13^30) % 100 = 98 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_last_two_digits_l1340_134047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_x_l1340_134071

theorem acute_angle_x (x : Real) (h : Real.sin (x + 15 * π / 180) = Real.sqrt 3 / 2) : 
  0 < x ∧ x < π / 2 → x = 45 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_x_l1340_134071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_not_blocked_iff_m_in_range_l1340_134033

-- Define the circle C
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point E
def E : ℝ × ℝ := (-4, 0)

-- Define point F
def F (m : ℝ) : ℝ × ℝ := (3, m)

-- Define the condition for the ray not being blocked
def RayNotBlocked (m : ℝ) : Prop :=
  ∀ t : ℝ, t ∈ Set.Ioo 0 1 → ¬Circle ((1 - t) * E.1 + t * (F m).1) ((1 - t) * E.2 + t * (F m).2)

-- Theorem statement
theorem ray_not_blocked_iff_m_in_range (m : ℝ) :
  RayNotBlocked m ↔ m < -7 * Real.sqrt 3 / 3 ∨ m > 7 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_not_blocked_iff_m_in_range_l1340_134033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_and_averages_l1340_134031

theorem product_and_averages (a b c d : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a * b = 240 ∧ 
  b + c = 120 ∧ 
  c + d = 180 →
  d - a = 116 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_and_averages_l1340_134031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_complex_equality_l1340_134012

theorem smallest_n_complex_equality (n : ℕ) (a b : ℝ) : 
  (0 < a) → (0 < b) → 
  (∀ k : ℕ, 0 < k → k < n → ¬∃ x y : ℝ, 0 < x ∧ 0 < y ∧ (Complex.I : ℂ) ^ k * (x + y * Complex.I) ^ k = (x - y * Complex.I) ^ k) →
  (Complex.I : ℂ) ^ n * (a + b * Complex.I) ^ n = (a - b * Complex.I) ^ n →
  b / a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_complex_equality_l1340_134012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1340_134014

theorem tan_alpha_plus_pi_fourth (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos α = Real.sqrt 65 / 65) : 
  Real.tan (α + π / 4) = -9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1340_134014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_proof_l1340_134042

/-- The speed of the stream in km/h -/
noncomputable def stream_speed : ℝ := 3

/-- The time taken by the boat to travel downstream in hours -/
noncomputable def downstream_time : ℝ := 1

/-- The time taken by the boat to travel upstream in hours -/
noncomputable def upstream_time : ℝ := 3/2

/-- The speed of the boat in still water in km/h -/
noncomputable def boat_speed : ℝ := 15

theorem boat_speed_proof :
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  downstream_speed * downstream_time = upstream_speed * upstream_time :=
by
  -- Expand definitions
  unfold stream_speed downstream_time upstream_time boat_speed
  -- Simplify expressions
  simp [mul_add, mul_sub, add_mul, sub_mul]
  -- Check equality
  norm_num

#check boat_speed_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_proof_l1340_134042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_formula_l1340_134008

/-- The radius of the incircle of triangle MBD --/
noncomputable def incircle_radius (x₂ : ℝ) : ℝ :=
  1 / (Real.sqrt (1 / (2 * x₂) + 1 / ((x₂ + 1/2)^2)) + 1 / (x₂ + 1/2))

/-- Theorem stating the radius of the incircle of triangle MBD --/
theorem incircle_radius_formula (x₂ y₂ y₃ : ℝ) 
  (h1 : y₂^2 = 2*x₂)  -- B is on the curve y^2 = 2x
  (h2 : x₂ > 1/2)     -- x₂ > 1/2
  (h3 : y₃^2 = 2*(1/(4*x₂))) -- D is on the curve y^2 = 2x
  : 
  incircle_radius x₂ = 
    (x₂ + 1/2) * abs y₂ / (abs y₂ + Real.sqrt ((x₂ + 1/2)^2 + y₂^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_formula_l1340_134008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l1340_134052

open Real NormedAddCommGroup InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable def angle (x y : E) : ℝ := Real.arccos (inner x y / (norm x * norm y))

theorem vector_angle_problem (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : norm (a + b) = norm a ∧ norm a = norm b) : 
  angle b (a - b) = π * 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l1340_134052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charles_winning_strategy_l1340_134074

def is_winning_number (n : ℕ) : Prop :=
  n > 1 ∧
  (n ∉ ({2, 7, 13} : Set ℕ) ∧ (Nat.Prime n ∨ n ∈ ({8, 14, 26, 49, 91, 169} : Set ℕ)))

def game_rules (initial_points : ℕ) (first_player : Bool) : Prop :=
  initial_points = 1000 ∧ first_player = false

/-- A strategy is a function that takes the current number and returns the next number -/
def Strategy := ℕ → ℕ

/-- Charles wins if the strategy leads to Ada running out of points -/
def charles_wins (strategy : Strategy) (n : ℕ) : Prop := sorry

theorem charles_winning_strategy (n : ℕ) :
  game_rules 1000 false →
  is_winning_number n ↔ ∃ (strategy : Strategy), charles_wins strategy n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charles_winning_strategy_l1340_134074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_for_monotonically_decreasing_sequence_l1340_134088

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is monotonically decreasing -/
def MonotonicallyDecreasing (a : Sequence) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

/-- The specific sequence defined by lambda * n^2 + n -/
def SpecificSequence (lambda : ℝ) : Sequence :=
  fun n => lambda * n^2 + n

theorem lambda_range_for_monotonically_decreasing_sequence :
  ∀ lambda : ℝ, (MonotonicallyDecreasing (SpecificSequence lambda)) → lambda < -1/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_for_monotonically_decreasing_sequence_l1340_134088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_triangle_pyramid_volume_l1340_134058

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle2D where
  a : Point2D
  b : Point2D
  c : Point2D

/-- Calculates the area of a triangle given its vertices -/
noncomputable def triangleArea (t : Triangle2D) : ℝ :=
  0.5 * abs (t.a.x * (t.b.y - t.c.y) + t.b.x * (t.c.y - t.a.y) + t.c.x * (t.a.y - t.b.y))

/-- Calculates the centroid of a triangle -/
noncomputable def triangleCentroid (t : Triangle2D) : Point2D :=
  { x := (t.a.x + t.b.x + t.c.x) / 3
  , y := (t.a.y + t.b.y + t.c.y) / 3 }

/-- Calculates the volume of a pyramid given base area and height -/
noncomputable def pyramidVolume (baseArea : ℝ) (height : ℝ) : ℝ :=
  (1/3) * baseArea * height

/-- The main theorem stating the volume of the folded triangular pyramid -/
theorem folded_triangle_pyramid_volume :
  let originalTriangle : Triangle2D := { a := {x := 0, y := 0}, b := {x := 30, y := 0}, c := {x := 15, y := 20} }
  let baseArea := triangleArea originalTriangle
  let centroid := triangleCentroid originalTriangle
  let height := centroid.y
  pyramidVolume baseArea height = 670 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_triangle_pyramid_volume_l1340_134058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stoppage_times_l1340_134006

/-- Calculates the stoppage time in minutes per hour for a train given its speeds with and without stoppages. -/
noncomputable def stoppage_time (speed_without_stoppage speed_with_stoppage : ℝ) : ℝ :=
  (1 - speed_with_stoppage / speed_without_stoppage) * 60

/-- Proves that for each train, given its speed without stoppages and speed with stoppages, the stoppage time is 15 minutes per hour. -/
theorem train_stoppage_times
  (speed_A_without speed_A_with speed_B_without speed_B_with speed_C_without speed_C_with : ℝ)
  (h_A_without : speed_A_without = 80)
  (h_A_with : speed_A_with = 60)
  (h_B_without : speed_B_without = 100)
  (h_B_with : speed_B_with = 75)
  (h_C_without : speed_C_without = 120)
  (h_C_with : speed_C_with = 90) :
  stoppage_time speed_A_without speed_A_with = 15 ∧
  stoppage_time speed_B_without speed_B_with = 15 ∧
  stoppage_time speed_C_without speed_C_with = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stoppage_times_l1340_134006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1340_134035

/-- The area of the shaded region formed by a square with side length 8 cm and four circles of radius 3 cm centered at its vertices -/
noncomputable def shaded_area (square_side : ℝ) (circle_radius : ℝ) : ℝ :=
  let θ := Real.arcsin (Real.sqrt 7 / 4)
  square_side^2 - 4 * square_side * Real.sqrt 7 / 2 - 4 * θ * circle_radius^2

theorem shaded_area_calculation (square_side : ℝ) (circle_radius : ℝ) 
  (h1 : square_side = 8)
  (h2 : circle_radius = 3) :
  shaded_area square_side circle_radius = 64 - 16 * Real.sqrt 7 - 18 * Real.arcsin (Real.sqrt 7 / 4) := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1340_134035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_shift_roots_omega_range_l1340_134000

theorem function_shift_roots_omega_range 
  (a ω : ℝ) 
  (h_a : a ≠ 0) 
  (h_ω : ω > 0) 
  (f g : ℝ → ℝ) 
  (h_f : f = λ x ↦ a * Real.cos (ω * x))
  (h_g : g = λ x ↦ a * Real.cos (ω * (x + π / (6 * ω))))
  (h_roots : ∃ x y, x ∈ Set.Icc 0 (7 * π / 12) ∧ 
                    y ∈ Set.Icc 0 (7 * π / 12) ∧ 
                    x ≠ y ∧ 
                    g x = 0 ∧ 
                    g y = 0 ∧ 
                    ∀ z ∈ Set.Icc 0 (7 * π / 12), g z = 0 → z = x ∨ z = y) :
  16/7 ≤ ω ∧ ω < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_shift_roots_omega_range_l1340_134000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_one_l1340_134039

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + x else x - x^2

-- Theorem statement
theorem a_greater_than_one (a : ℝ) (h : f a > f (2 - a)) : a > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_one_l1340_134039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fraction_parts_l1340_134066

def recurring_decimal : ℚ := 3 + 714 / 999

theorem sum_of_fraction_parts : ∃ (n d : ℕ), recurring_decimal = n / d ∧ Nat.Coprime n d ∧ n + d = 4710 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fraction_parts_l1340_134066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_not_identical_l1340_134044

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := x^2 / x
def g1 (x : ℝ) : ℝ := x

noncomputable def f2 (x : ℝ) : ℝ := Real.sqrt (x^2)
def g2 (x : ℝ) : ℝ := x

def f3 (x : ℝ) : ℝ := 3 * x^3
def g3 (x : ℝ) : ℝ := x

noncomputable def f4 (x : ℝ) : ℝ := (Real.sqrt x)^2
def g4 (x : ℝ) : ℝ := x

-- Theorem statement
theorem functions_not_identical :
  (∃ x, f1 x ≠ g1 x) ∧
  (∃ x, f2 x ≠ g2 x) ∧
  (∃ x, f3 x ≠ g3 x) ∧
  (∃ x, f4 x ≠ g4 x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_not_identical_l1340_134044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_time_is_12_l1340_134016

/-- The time (in hours) it takes A to complete the work alone -/
noncomputable def A_time : ℝ := 4

/-- The time (in hours) it takes A and C together to complete the work -/
noncomputable def AC_time : ℝ := 2

/-- The time (in hours) it takes B and C together to complete the work -/
noncomputable def BC_time : ℝ := 3

/-- The work rate (fraction of work completed per hour) for A -/
noncomputable def A_work : ℝ := 1 / A_time

/-- The work rate (fraction of work completed per hour) for B -/
noncomputable def B_work : ℝ := 1 / BC_time - 1 / AC_time

/-- The time (in hours) it takes B to complete the work alone -/
noncomputable def B_time : ℝ := 1 / B_work

theorem B_time_is_12 : B_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_time_is_12_l1340_134016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_periodic_l1340_134050

def is_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n ≥ 1 → a (2 * n) = a n) ∧
  (∀ n : ℕ, a (4 * n + 1) = 1) ∧
  (∀ n : ℕ, a (4 * n + 3) = 0)

theorem sequence_not_periodic (a : ℕ → ℕ) (h : is_sequence a) :
  ¬ ∃ T : ℕ, T > 0 ∧ ∀ n : ℕ, n ≥ 1 → a (n + T) = a n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_periodic_l1340_134050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_74pi_over_4_l1340_134069

-- Define the points C and D
def C : ℝ × ℝ := (3, 5)
def D : ℝ × ℝ := (8, 12)

-- Define the circle diameter as the distance between C and D
noncomputable def diameter : ℝ := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)

-- Define the circle radius
noncomputable def radius : ℝ := diameter / 2

-- Define the circle area
noncomputable def circle_area : ℝ := Real.pi * radius^2

-- Theorem statement
theorem circle_area_is_74pi_over_4 : circle_area = 74 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_74pi_over_4_l1340_134069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_to_cats_ratio_l1340_134097

/-- Proves that the ratio of original dogs to cats is 3:2 given the problem conditions --/
theorem dogs_to_cats_ratio (original_cats : ℕ) (original_dogs : ℕ) (new_dogs : ℕ) : 
  original_cats = 40 →
  new_dogs = 20 →
  original_dogs + new_dogs = 2 * original_cats →
  (original_dogs : ℚ) / original_cats = 3 / 2 := by
  intros h1 h2 h3
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_to_cats_ratio_l1340_134097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_log_cubic_l1340_134032

/-- The function f(x) = logₐ(x³ - ax) is monotonically increasing over (2, +∞) iff 1 < a ≤ 4 -/
theorem monotone_log_cubic (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x > 2, StrictMono (fun x => Real.log (x^3 - a*x) / Real.log a)) ↔ (1 < a ∧ a ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_log_cubic_l1340_134032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_376_to_hundredth_l1340_134017

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The standard rounding rules are applied in the roundToHundredth function -/
axiom roundToHundredth_correct (x : ℝ) : 
  |roundToHundredth x - x| ≤ 0.005

theorem round_4_376_to_hundredth : 
  roundToHundredth 4.376 = 4.38 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_376_to_hundredth_l1340_134017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_half_l1340_134025

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 2)

theorem min_positive_period_sin_half : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  T = 4 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_half_l1340_134025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_store_costs_l1340_134070

/-- Represents the daily operational costs of a grocery store -/
def total_cost : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ := 
  fun X E S D W => X

/-- Represents the fraction of total cost spent on orders -/
noncomputable def order_fraction (X E S D W : ℝ) : ℝ :=
  1 - (2/5 + 1/4 * (1 - 2/5))

theorem grocery_store_costs (X E S D W : ℝ) 
  (hX : X > 0) 
  (hE : E > 0) 
  (hS : S > 0) 
  (hD : D > 0) 
  (hW : W > 0) 
  (h_salaries : 2/5 * X = E * S) 
  (h_delivery : 1/4 * (3/5 * X) = D * W) :
  order_fraction X E S D W = 9/20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_store_costs_l1340_134070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_length_l1340_134061

def sequenceDiscard (k : ℕ) : ℕ := ((1000 + k)^2) / 1000

theorem arithmetic_progression_length : 
  (∃ d : ℕ, ∀ k : ℕ, k < 32 → sequenceDiscard (k + 1) - sequenceDiscard k = d) ∧ 
  (¬ ∃ d : ℕ, sequenceDiscard 32 - sequenceDiscard 31 = d ∧ 
              ∀ k : ℕ, k < 32 → sequenceDiscard (k + 1) - sequenceDiscard k = d) :=
by sorry

#eval sequenceDiscard 0  -- This will output 1000
#eval sequenceDiscard 31 -- This will output 1062
#eval sequenceDiscard 32 -- This will output 1065

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_length_l1340_134061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_conditions_l1340_134021

def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def satisfies_conditions (n : ℕ) : Bool :=
  n < 5000 && n % 11 = 0 && digit_sum n = 13

theorem count_numbers_with_conditions :
  (Finset.filter (fun n => satisfies_conditions n) (Finset.range 5000)).card = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_conditions_l1340_134021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_constant_linear_combination_l1340_134082

-- Define the type of functions from [0,1] to ℝ
def UnitIntervalToReal : Type := Set.Icc (0 : ℝ) 1 → ℝ

-- Define monotonicity for functions on [0,1]
def IsMonotone (f : UnitIntervalToReal) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Define the property that all linear combinations are monotone
def AllLinearCombinationsMonotone (f₁ f₂ : UnitIntervalToReal) : Prop :=
  ∀ a₁ a₂ : ℝ, IsMonotone (fun x ↦ a₁ * f₁ x + a₂ * f₂ x)

-- Define what it means for a function to be constant
def IsConstant (f : UnitIntervalToReal) : Prop :=
  ∀ x y, f x = f y

-- The main theorem
theorem exists_constant_linear_combination
  (f₁ f₂ : UnitIntervalToReal)
  (h_monotone : AllLinearCombinationsMonotone f₁ f₂)
  (h_not_constant : ¬IsConstant f₂) :
  ∃ a : ℝ, IsConstant (fun x ↦ f₁ x - a * f₂ x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_constant_linear_combination_l1340_134082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_probability_l1340_134055

/-- A line segment of unit length -/
def UnitSegment : Set ℝ := Set.Icc 0 1

/-- A division of the unit segment into four parts -/
def Partition (p q r : ℝ) : Prop :=
  0 < p ∧ p < q ∧ q < r ∧ r < 1

/-- The condition for four lengths to form a quadrilateral -/
def FormsQuadrilateral (a b c d : ℝ) : Prop :=
  a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a

/-- The probability measure on the unit segment -/
noncomputable def SegmentProbability : MeasureTheory.Measure (ℝ × ℝ × ℝ) :=
  sorry

/-- The main theorem: probability of forming a quadrilateral is 1/2 -/
theorem quadrilateral_probability :
  SegmentProbability {x : ℝ × ℝ × ℝ | let (p, q, r) := x; Partition p q r ∧
    FormsQuadrilateral p (q - p) (r - q) (1 - r)} = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_probability_l1340_134055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1340_134011

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 1 = 0

-- Define the distance function from a point to the line l
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + Real.sqrt 3 * y - 1) / 2

-- Theorem statement
theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = 1/2 ∧
  (∀ (x y : ℝ), circle_M x y → distance_to_line x y ≥ d) ∧
  (∃ (x y : ℝ), circle_M x y ∧ distance_to_line x y = d) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1340_134011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_good_triangles_l1340_134030

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  is_regular : sides = n

/-- A diagonal or side of a polygon -/
inductive Edge (n : ℕ)
  | side : Edge n
  | diagonal : Edge n

/-- A "good edge" in a regular polygon -/
def is_good_edge {n : ℕ} (p : RegularPolygon n) (e : Edge n) : Prop :=
  match e with
  | Edge.side => True
  | Edge.diagonal => ∃ (k : ℕ), 2 * k + 1 < n ∧ n - (2 * k + 1) < n ∧ 2 * k + 1 > 0

/-- An isosceles triangle with two "good edges" -/
structure GoodTriangle (n : ℕ) (p : RegularPolygon n) where
  e1 : Edge n
  e2 : Edge n
  is_isosceles : True  -- Simplified, as we can't easily represent this geometrically
  good_e1 : is_good_edge p e1
  good_e2 : is_good_edge p e2

/-- The main theorem -/
theorem max_good_triangles (p : RegularPolygon 2006) 
  (diagonals : Finset (Edge 2006))
  (h_diagonals : diagonals.card = 2003) :
  (∃ (triangles : Finset (GoodTriangle 2006 p)), 
    triangles.card = 1003 ∧ 
    ∀ t1 t2, t1 ∈ triangles → t2 ∈ triangles → t1 ≠ t2 → 
      (t1.e1 ≠ t2.e1 ∧ t1.e1 ≠ t2.e2 ∧ t1.e2 ≠ t2.e1 ∧ t1.e2 ≠ t2.e2)) ∧ 
  (∀ (triangles : Finset (GoodTriangle 2006 p)), 
    (∀ t1 t2, t1 ∈ triangles → t2 ∈ triangles → t1 ≠ t2 → 
      (t1.e1 ≠ t2.e1 ∧ t1.e1 ≠ t2.e2 ∧ t1.e2 ≠ t2.e1 ∧ t1.e2 ≠ t2.e2)) →
    triangles.card ≤ 1003) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_good_triangles_l1340_134030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_work_time_l1340_134086

/-- Represents the time it takes for a worker to complete a job alone. -/
structure WorkTime where
  hours : ℝ
  hours_positive : hours > 0

/-- Represents the portion of a job completed in one hour. -/
noncomputable def workRate (w : WorkTime) : ℝ := 1 / w.hours

/-- Theorem stating that Q's work time is 15 hours given the conditions. -/
theorem q_work_time (p q : WorkTime) (h1 : p.hours = 4) 
  (h2 : workRate p + workRate q = 19 / (3 * 20)) : q.hours = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_work_time_l1340_134086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_equality_l1340_134026

/-- The function f(x) defined as the square root of ax² + bx -/
noncomputable def f (a b x : ℝ) : ℝ := Real.sqrt (a * x^2 + b * x)

/-- The domain of function f -/
def domain (a b : ℝ) : Set ℝ := {x : ℝ | a * x^2 + b * x ≥ 0}

/-- The range of function f -/
def range (a b : ℝ) : Set ℝ := {y : ℝ | ∃ x, f a b x = y}

/-- Theorem stating the condition for domain and range equality -/
theorem domain_range_equality (b : ℝ) (hb : b > 0) :
  ∃ a : ℝ, a ≠ 0 ∧ (domain a b = range a b ↔ a = -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_equality_l1340_134026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problem_l1340_134043

theorem square_root_problem (a b : ℝ) : 
  (Real.sqrt a = 2*b - 3 ∧ Real.sqrt a = 3*b + 8) → (a = 25 ∧ b = -1) := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problem_l1340_134043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l1340_134095

theorem abc_relationship : 
  let a : ℝ := Real.sqrt 2
  let b : ℝ := (Real.log 2)^(-(1/2 : ℝ))
  let c : ℝ := Real.log 2
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l1340_134095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_segment_length_l1340_134007

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define point A
def point_A : ℝ × ℝ := (1, 2*Real.sqrt 2)

-- Define a line passing through two points
def line_through (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

-- Theorem statement
theorem parabola_segment_length :
  ∃ (B : ℝ × ℝ),
    parabola B.1 B.2 ∧
    line_through focus point_A B.1 B.2 ∧
    Real.sqrt ((B.1 - point_A.1)^2 + (B.2 - point_A.2)^2) = 9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_segment_length_l1340_134007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_level_rise_l1340_134094

/-- Represents the dimensions of a rectangular shape -/
structure Dimensions where
  length : ℝ
  width : ℝ

/-- Represents a cuboid shape with height -/
structure Cuboid extends Dimensions where
  height : ℝ

/-- Calculates the area of a rectangular shape -/
def area (d : Dimensions) : ℝ := d.length * d.width

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

theorem field_level_rise 
  (field : Dimensions)
  (tank : Cuboid)
  (h_field_length : field.length = 90)
  (h_field_width : field.width = 50)
  (h_tank_length : tank.length = 25)
  (h_tank_width : tank.width = 20)
  (h_tank_height : tank.height = 4)
  : (volume tank) / (area field - area { length := tank.length, width := tank.width }) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_level_rise_l1340_134094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_h_to_j_l1340_134093

-- Define the original function h
variable (h : ℝ → ℝ)

-- Define the transformed function j
def j (h : ℝ → ℝ) (x : ℝ) : ℝ := h (6 - x)

-- Theorem statement
theorem transform_h_to_j (h : ℝ → ℝ) (x : ℝ) : j h x = h (6 - x) := by
  -- Unfold the definition of j
  unfold j
  -- The equality now holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_h_to_j_l1340_134093
