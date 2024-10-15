import Mathlib

namespace NUMINAMATH_CALUDE_milk_expense_l2036_203623

/-- Proves that the amount spent on milk is 1500, given the total expenses
    excluding milk and savings, the savings amount, and the savings rate. -/
theorem milk_expense (total_expenses_excl_milk : ℕ) (savings : ℕ) (savings_rate : ℚ) :
  total_expenses_excl_milk = 16500 →
  savings = 2000 →
  savings_rate = 1/10 →
  (total_expenses_excl_milk + savings) / (1 - savings_rate) - (total_expenses_excl_milk + savings) = 1500 := by
  sorry

end NUMINAMATH_CALUDE_milk_expense_l2036_203623


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2036_203606

theorem inscribed_circle_radius 
  (a b c : ℝ) 
  (ha : a = 8) 
  (hb : b = 5) 
  (hc : c = 7) 
  (h_area : (a + b + c) / 2 - 2 = (a + b + c) / 2 * r) : 
  r = 1.8 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2036_203606


namespace NUMINAMATH_CALUDE_pyramid_lateral_angle_l2036_203691

/-- Given a pyramid with an isosceles triangular base of area S, angle α between the equal sides,
    and volume V, the angle θ between the lateral edges and the base plane is:
    θ = arctan((3V * cos(α/2) / S) * sqrt(2 * sin(α) / S)) -/
theorem pyramid_lateral_angle (S V : ℝ) (α : ℝ) (hS : S > 0) (hV : V > 0) (hα : 0 < α ∧ α < π) :
  ∃ θ : ℝ, θ = Real.arctan ((3 * V * Real.cos (α / 2) / S) * Real.sqrt (2 * Real.sin α / S)) :=
sorry

end NUMINAMATH_CALUDE_pyramid_lateral_angle_l2036_203691


namespace NUMINAMATH_CALUDE_work_hours_first_scenario_l2036_203635

/-- Represents the work rate of a person -/
structure WorkRate where
  rate : ℝ

/-- Represents a work scenario -/
structure WorkScenario where
  men : ℕ
  women : ℕ
  hours : ℝ
  days : ℝ

/-- The theorem to prove -/
theorem work_hours_first_scenario 
  (man_rate : WorkRate)
  (woman_rate : WorkRate)
  (scenario1 : WorkScenario)
  (scenario2 : WorkScenario)
  (scenario3 : WorkScenario)
  (h1 : scenario1.men = 2 ∧ scenario1.women = 3 ∧ scenario1.days = 5)
  (h2 : scenario2.men = 4 ∧ scenario2.women = 4 ∧ scenario2.hours = 3 ∧ scenario2.days = 7)
  (h3 : scenario3.men = 7 ∧ scenario3.hours = 4 ∧ scenario3.days = 5.000000000000001)
  (h4 : (scenario1.men : ℝ) * man_rate.rate * scenario1.hours * scenario1.days + 
        (scenario1.women : ℝ) * woman_rate.rate * scenario1.hours * scenario1.days = 1)
  (h5 : (scenario2.men : ℝ) * man_rate.rate * scenario2.hours * scenario2.days + 
        (scenario2.women : ℝ) * woman_rate.rate * scenario2.hours * scenario2.days = 1)
  (h6 : (scenario3.men : ℝ) * man_rate.rate * scenario3.hours * scenario3.days = 1) :
  scenario1.hours = 7 := by
  sorry


end NUMINAMATH_CALUDE_work_hours_first_scenario_l2036_203635


namespace NUMINAMATH_CALUDE_lines_parallel_in_intersecting_planes_l2036_203638

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation between a line and a plane
variable (lineParallelPlane : Line → Plane → Prop)

-- State the theorem
theorem lines_parallel_in_intersecting_planes
  (l m n : Line) (α β γ : Plane)
  (distinct_lines : l ≠ m ∧ m ≠ n ∧ n ≠ l)
  (distinct_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α)
  (h1 : intersect α β = l)
  (h2 : intersect β γ = m)
  (h3 : intersect γ α = n)
  (h4 : lineParallelPlane l γ) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_in_intersecting_planes_l2036_203638


namespace NUMINAMATH_CALUDE_adjacent_same_face_exists_l2036_203663

/-- Represents the face of a coin -/
inductive CoinFace
| Heads
| Tails

/-- Represents a circular arrangement of coins -/
def CoinCircle := List CoinFace

/-- Checks if two adjacent coins have the same face -/
def hasAdjacentSameFace (circle : CoinCircle) : Prop :=
  ∃ i, (circle.get? i = circle.get? ((i + 1) % circle.length))

/-- Theorem: Any arrangement of 11 coins in a circle always has at least one pair of adjacent coins with the same face -/
theorem adjacent_same_face_exists (circle : CoinCircle) (h : circle.length = 11) :
  hasAdjacentSameFace circle :=
sorry

end NUMINAMATH_CALUDE_adjacent_same_face_exists_l2036_203663


namespace NUMINAMATH_CALUDE_possible_y_values_l2036_203621

-- Define the relationship between x and y
def relation (x y : ℝ) : Prop := x^2 = y - 5

-- Theorem statement
theorem possible_y_values :
  (∃ y : ℝ, relation (-7) y ∧ y = 54) ∧
  (∃ y : ℝ, relation 2 y ∧ y = 9) := by
  sorry

end NUMINAMATH_CALUDE_possible_y_values_l2036_203621


namespace NUMINAMATH_CALUDE_min_lines_8x8_grid_l2036_203646

/-- Represents a grid with points at the center of each square -/
structure Grid :=
  (size : ℕ)
  (points : ℕ)

/-- Calculates the minimum number of lines needed to separate all points in a grid -/
def min_lines (g : Grid) : ℕ :=
  2 * (g.size - 1)

/-- Theorem: For an 8x8 grid with 64 points, the minimum number of lines to separate all points is 14 -/
theorem min_lines_8x8_grid :
  let g : Grid := ⟨8, 64⟩
  min_lines g = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_lines_8x8_grid_l2036_203646


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2036_203618

theorem sum_of_fractions : (2 / 12 : ℚ) + (4 / 24 : ℚ) + (6 / 36 : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2036_203618


namespace NUMINAMATH_CALUDE_leja_theorem_l2036_203697

/-- A set of points in a plane where any three points lie on a circle of radius r -/
def SpecialPointSet (P : Set (ℝ × ℝ)) (r : ℝ) : Prop :=
  ∀ p q s : ℝ × ℝ, p ∈ P → q ∈ P → s ∈ P → p ≠ q → q ≠ s → p ≠ s →
    ∃ c : ℝ × ℝ, dist c p = r ∧ dist c q = r ∧ dist c s = r

/-- Leja's theorem -/
theorem leja_theorem (P : Set (ℝ × ℝ)) (r : ℝ) (h : SpecialPointSet P r) :
  ∃ A : ℝ × ℝ, ∀ p ∈ P, dist A p ≤ r := by
  sorry

end NUMINAMATH_CALUDE_leja_theorem_l2036_203697


namespace NUMINAMATH_CALUDE_power_function_solution_l2036_203698

-- Define a power function type
def PowerFunction := ℝ → ℝ

-- Define the properties of our specific power function
def isPowerFunctionThroughPoint (f : PowerFunction) : Prop :=
  ∃ α : ℝ, (∀ x : ℝ, f x = x ^ α) ∧ f (-2) = -1/8

-- State the theorem
theorem power_function_solution 
  (f : PowerFunction) 
  (h : isPowerFunctionThroughPoint f) : 
  ∃ x : ℝ, f x = 27 ∧ x = 1/3 := by
sorry

end NUMINAMATH_CALUDE_power_function_solution_l2036_203698


namespace NUMINAMATH_CALUDE_base5_product_132_23_l2036_203633

/-- Converts a base 5 number to base 10 -/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base 10 number to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Multiplies two base 5 numbers -/
def multiplyBase5 (a b : List Nat) : List Nat :=
  base10ToBase5 (base5ToBase10 a * base5ToBase10 b)

theorem base5_product_132_23 :
  multiplyBase5 [2, 3, 1] [3, 2] = [1, 4, 1, 4] :=
sorry

end NUMINAMATH_CALUDE_base5_product_132_23_l2036_203633


namespace NUMINAMATH_CALUDE_parabola_intercepts_l2036_203641

-- Define the parabola equation
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- Theorem statement
theorem parabola_intercepts :
  -- There is exactly one x-intercept at x = 3
  (∃! x : ℝ, x = 3 ∧ ∃ y : ℝ, parabola y = x) ∧
  -- There are exactly two y-intercepts
  (∃! y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ 
    parabola y₁ = 0 ∧ parabola y₂ = 0 ∧
    y₁ = (1 + Real.sqrt 10) / 3 ∧
    y₂ = (1 - Real.sqrt 10) / 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intercepts_l2036_203641


namespace NUMINAMATH_CALUDE_number_comparison_l2036_203662

theorem number_comparison (A B : ℝ) (h : A = B + B / 4) : 
  B = A - A / 5 ∧ B ≠ A - A / 4 := by
  sorry

end NUMINAMATH_CALUDE_number_comparison_l2036_203662


namespace NUMINAMATH_CALUDE_three_planes_solutions_two_planes_one_point_solutions_one_plane_two_points_solutions_three_points_solutions_l2036_203620

-- Define the basic types
structure Point
structure Plane

-- Define the distance function
def distance (p : Point) (x : Point ⊕ Plane) : ℝ := sorry

-- Define the function to count solutions
def countSolutions (objects : List (Point ⊕ Plane)) (d : ℝ) : ℕ := sorry

-- Theorem for case (a)
theorem three_planes_solutions (p1 p2 p3 : Plane) (d : ℝ) :
  countSolutions [Sum.inr p1, Sum.inr p2, Sum.inr p3] d = 8 := sorry

-- Theorem for case (b)
theorem two_planes_one_point_solutions (p1 p2 : Plane) (pt : Point) (d : ℝ) :
  countSolutions [Sum.inr p1, Sum.inr p2, Sum.inl pt] d = 8 := sorry

-- Theorem for case (c)
theorem one_plane_two_points_solutions (p : Plane) (pt1 pt2 : Point) (d : ℝ) :
  countSolutions [Sum.inr p, Sum.inl pt1, Sum.inl pt2] d = 4 := sorry

-- Theorem for case (d)
theorem three_points_solutions (pt1 pt2 pt3 : Point) (d : ℝ) :
  let n := countSolutions [Sum.inl pt1, Sum.inl pt2, Sum.inl pt3] d
  n = 0 ∨ n = 1 ∨ n = 2 := sorry

end NUMINAMATH_CALUDE_three_planes_solutions_two_planes_one_point_solutions_one_plane_two_points_solutions_three_points_solutions_l2036_203620


namespace NUMINAMATH_CALUDE_function_equation_solution_l2036_203690

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (x - y) * (f x - f y) = f (x - f y) * f (f x - y)) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) :=
sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2036_203690


namespace NUMINAMATH_CALUDE_linear_function_properties_l2036_203645

def f (x : ℝ) : ℝ := -2 * x + 3

theorem linear_function_properties :
  (f 1 = 1) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ f x = y) ∧
  (f⁻¹ 0 ≠ 0) ∧
  (∀ (x1 x2 : ℝ), x1 < x2 → f x1 > f x2) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2036_203645


namespace NUMINAMATH_CALUDE_unique_positive_integers_sum_l2036_203642

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 37) / 2 + 5 / 2)

theorem unique_positive_integers_sum (d e f : ℕ+) :
  y^50 = 2*y^48 + 6*y^46 + 5*y^44 - y^25 + (d:ℝ)*y^21 + (e:ℝ)*y^19 + (f:ℝ)*y^15 →
  d + e + f = 98 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_integers_sum_l2036_203642


namespace NUMINAMATH_CALUDE_sara_received_four_onions_l2036_203688

/-- The number of onions given to Sara -/
def onions_given_to_sara (sally_onions fred_onions remaining_onions : ℕ) : ℕ :=
  sally_onions + fred_onions - remaining_onions

/-- Theorem stating that Sara received 4 onions -/
theorem sara_received_four_onions :
  onions_given_to_sara 5 9 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sara_received_four_onions_l2036_203688


namespace NUMINAMATH_CALUDE_no_quaint_two_digit_integers_l2036_203668

def is_quaint (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ 
  ∃ (a b : ℕ), n = 10 * a + b ∧ a > 0 ∧ b < 10 ∧ n = a + b^3

theorem no_quaint_two_digit_integers : ¬∃ (n : ℕ), is_quaint n := by
  sorry

end NUMINAMATH_CALUDE_no_quaint_two_digit_integers_l2036_203668


namespace NUMINAMATH_CALUDE_pascal_triangle_12th_row_4th_number_l2036_203686

theorem pascal_triangle_12th_row_4th_number : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_12th_row_4th_number_l2036_203686


namespace NUMINAMATH_CALUDE_fraction_value_l2036_203680

theorem fraction_value (N : ℝ) (h : 0.4 * N = 168) : (1/4) * (1/3) * (2/5) * N = 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2036_203680


namespace NUMINAMATH_CALUDE_inequality_proof_l2036_203637

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2036_203637


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l2036_203665

/-- Represents the number of ways to arrange frogs with given constraints -/
def frog_arrangements (total : ℕ) (green : ℕ) (red : ℕ) (blue : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  frog_arrangements 8 3 4 1 = 576 :=
by
  sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l2036_203665


namespace NUMINAMATH_CALUDE_crocodile_coloring_l2036_203626

theorem crocodile_coloring (m n : ℕ) (h_m : m > 0) (h_n : n > 0) :
  ∃ f : ℤ × ℤ → Bool,
    ∀ x y : ℤ, f (x, y) ≠ f (x + m, y + n) ∧ f (x, y) ≠ f (x + n, y + m) := by
  sorry

end NUMINAMATH_CALUDE_crocodile_coloring_l2036_203626


namespace NUMINAMATH_CALUDE_divisibility_sequence_l2036_203640

theorem divisibility_sequence (t : ℤ) (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ n : ℕ, n > 0 ∧ (p : ℤ) ∣ ((3 - 7*t) * 2^n + (18*t - 9) * 3^n + (6 - 10*t) * 4^n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_sequence_l2036_203640


namespace NUMINAMATH_CALUDE_H_range_l2036_203612

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem H_range : Set.range H = Set.Icc (-5) 5 := by sorry

end NUMINAMATH_CALUDE_H_range_l2036_203612


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l2036_203684

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def N (m : ℝ) : Set ℝ := {x : ℝ | x - m ≥ 0}

-- Theorem for the first part
theorem union_condition (m : ℝ) : M ∪ N m = N m ↔ m ≤ -2 := by sorry

-- Theorem for the second part
theorem intersection_condition (m : ℝ) : M ∩ N m = ∅ ↔ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l2036_203684


namespace NUMINAMATH_CALUDE_bird_count_l2036_203609

theorem bird_count (total_heads : ℕ) (total_legs : ℕ) (bird_legs : ℕ) (mammal_legs : ℕ) (insect_legs : ℕ) :
  total_heads = 300 →
  total_legs = 1112 →
  bird_legs = 2 →
  mammal_legs = 4 →
  insect_legs = 6 →
  ∃ (birds mammals insects : ℕ),
    birds + mammals + insects = total_heads ∧
    birds * bird_legs + mammals * mammal_legs + insects * insect_legs = total_legs ∧
    birds = 122 :=
by sorry

end NUMINAMATH_CALUDE_bird_count_l2036_203609


namespace NUMINAMATH_CALUDE_square_side_increase_l2036_203671

theorem square_side_increase (s : ℝ) (h : s > 0) :
  let new_area := s^2 * (1 + 0.3225)
  let new_side := s * (1 + 0.15)
  new_side^2 = new_area := by sorry

end NUMINAMATH_CALUDE_square_side_increase_l2036_203671


namespace NUMINAMATH_CALUDE_impossible_table_fill_l2036_203673

/-- Represents a table filled with natural numbers -/
def Table (n : ℕ) := Fin n → Fin n → ℕ

/-- Checks if a row in the table satisfies the product condition -/
def RowSatisfiesCondition (row : Fin n → ℕ) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ row i * row j = row k

/-- Checks if all elements in the table are distinct and within the range 1 to n^2 -/
def ValidTable (t : Table n) : Prop :=
  (∀ i j, 1 ≤ t i j ∧ t i j ≤ n^2) ∧
  (∀ i₁ j₁ i₂ j₂, (i₁, j₁) ≠ (i₂, j₂) → t i₁ j₁ ≠ t i₂ j₂)

/-- The main theorem stating the impossibility of filling the table -/
theorem impossible_table_fill (n : ℕ) (h : n ≥ 3) :
  ¬∃ (t : Table n), ValidTable t ∧ (∀ i : Fin n, RowSatisfiesCondition (t i)) :=
sorry

end NUMINAMATH_CALUDE_impossible_table_fill_l2036_203673


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l2036_203656

theorem tangent_line_to_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d ∧ y^2 = 12 * x ∧ 
   ∀ x' y' : ℝ, y' = 3 * x' + d → y'^2 ≥ 12 * x') → 
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l2036_203656


namespace NUMINAMATH_CALUDE_sqrt_fifty_minus_sqrt_thirtytwo_equals_sqrt_two_l2036_203628

theorem sqrt_fifty_minus_sqrt_thirtytwo_equals_sqrt_two :
  Real.sqrt 50 - Real.sqrt 32 = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_fifty_minus_sqrt_thirtytwo_equals_sqrt_two_l2036_203628


namespace NUMINAMATH_CALUDE_greater_than_negative_two_by_one_l2036_203685

theorem greater_than_negative_two_by_one : 
  ∃ x : ℝ, x = -2 + 1 ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_greater_than_negative_two_by_one_l2036_203685


namespace NUMINAMATH_CALUDE_parabola_adjoint_tangent_locus_l2036_203689

/-- Given a parabola y = 2px, prove that the locus of points (x, y) where the tangents 
    to the parabola are its own adjoint lines is described by the equation y² = -p/2 * x -/
theorem parabola_adjoint_tangent_locus (p : ℝ) (x y x₁ y₁ : ℝ) 
  (h1 : y₁ = 2 * p * x₁)  -- Original parabola equation
  (h2 : x = -x₁)          -- Relation between x and x₁
  (h3 : y = y₁ / 2)       -- Relation between y and y₁
  : y^2 = -p/2 * x := by sorry

end NUMINAMATH_CALUDE_parabola_adjoint_tangent_locus_l2036_203689


namespace NUMINAMATH_CALUDE_triangle_side_length_l2036_203625

-- Define the triangle
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = c^2

-- Define right angle
def RightAngle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define 60 degree angle
def SixtyDegreeAngle (A B C : ℝ × ℝ) : Prop :=
  ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2))^2 =
  3 * ((B.1 - A.1)^2 + (B.2 - A.2)^2) * ((C.1 - A.1)^2 + (C.2 - A.2)^2) / 4

-- Define inscribed circle radius
def InscribedCircleRadius (A B C : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ (O : ℝ × ℝ), 
    (O.1 - A.1)^2 + (O.2 - A.2)^2 = r^2 ∧
    (O.1 - B.1)^2 + (O.2 - B.2)^2 = r^2 ∧
    (O.1 - C.1)^2 + (O.2 - C.2)^2 = r^2

-- Theorem statement
theorem triangle_side_length 
  (A B C : ℝ × ℝ) 
  (h1 : Triangle A B C)
  (h2 : RightAngle B A C)
  (h3 : SixtyDegreeAngle A B C)
  (h4 : InscribedCircleRadius A B C 8) :
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (24 * Real.sqrt 3 + 24)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2036_203625


namespace NUMINAMATH_CALUDE_circle_bisecting_two_circles_l2036_203600

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 8)^2 = 1
def C2 (x y : ℝ) : Prop := (x - 6)^2 + (y + 6)^2 = 9

-- Define a circle C with center (a, 0) and radius r
def C (x y a r : ℝ) : Prop := (x - a)^2 + y^2 = r^2

-- Define the property of C bisecting C1 and C2
def bisects (a r : ℝ) : Prop :=
  ∀ x y : ℝ, C1 x y → (C x y a r ∨ C x y a r)
  ∧ ∀ x y : ℝ, C2 x y → (C x y a r ∨ C x y a r)

-- Theorem statement
theorem circle_bisecting_two_circles :
  ∀ a r : ℝ, bisects a r → C x y 0 9 :=
sorry

end NUMINAMATH_CALUDE_circle_bisecting_two_circles_l2036_203600


namespace NUMINAMATH_CALUDE_det_2x2_matrix_l2036_203679

theorem det_2x2_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 3]
  Matrix.det A = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_2x2_matrix_l2036_203679


namespace NUMINAMATH_CALUDE_all_yarns_are_xants_and_wooks_l2036_203643

-- Define the sets
variable (Zelm Xant Yarn Wook : Type)

-- Define the conditions
variable (zelm_xant : Zelm → Xant)
variable (yarn_zelm : Yarn → Zelm)
variable (xant_wook : Xant → Wook)

-- Theorem to prove
theorem all_yarns_are_xants_and_wooks :
  (∀ y : Yarn, ∃ x : Xant, zelm_xant (yarn_zelm y) = x) ∧
  (∀ y : Yarn, ∃ w : Wook, xant_wook (zelm_xant (yarn_zelm y)) = w) :=
sorry

end NUMINAMATH_CALUDE_all_yarns_are_xants_and_wooks_l2036_203643


namespace NUMINAMATH_CALUDE_J_specific_value_l2036_203615

/-- Definition of J function -/
def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

/-- Theorem: J(3, 3/4, 4) equals 259/48 -/
theorem J_specific_value : J 3 (3/4) 4 = 259/48 := by
  sorry

/-- Lemma: Relationship between a, b, and c -/
lemma abc_relationship (a b c k : ℚ) (hk : k ≠ 0) : 
  b = a / k ∧ c = k * b → J a b c = J a (a / k) (k * (a / k)) := by
  sorry

end NUMINAMATH_CALUDE_J_specific_value_l2036_203615


namespace NUMINAMATH_CALUDE_count_numbers_theorem_l2036_203610

/-- The count of positive integers less than 50000 with at most three different digits -/
def count_numbers_with_at_most_three_digits : ℕ :=
  let one_digit := 45  -- 5 * 9
  let two_digits_without_zero := 1872  -- 36 * 52
  let two_digits_with_zero := 234  -- 9 * 26
  let three_digits_with_zero := 900  -- 36 * 25
  let three_digits_without_zero := 4452  -- 84 * 53
  one_digit + two_digits_without_zero + two_digits_with_zero + three_digits_with_zero + three_digits_without_zero

/-- The theorem stating that the count of positive integers less than 50000 
    with at most three different digits is 7503 -/
theorem count_numbers_theorem : count_numbers_with_at_most_three_digits = 7503 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_theorem_l2036_203610


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2036_203683

/-- An arithmetic sequence {a_n} where a_1 = 1/3, a_2 + a_5 = 4, and a_n = 33 has n = 50 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) (n : ℕ) :
  (∀ k : ℕ, a (k + 1) - a k = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 1 / 3 →
  a 2 + a 5 = 4 →
  a n = 33 →
  n = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2036_203683


namespace NUMINAMATH_CALUDE_least_positive_integer_satisfying_conditions_l2036_203614

theorem least_positive_integer_satisfying_conditions : ∃ (N : ℕ), 
  (N > 1) ∧ 
  (∃ (a : ℕ), a > 0 ∧ N = a * (2 * a - 1)) ∧ 
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 10 → (List.sum (List.range (N - 1))) % k = 0) ∧
  (∀ (M : ℕ), M > 1 ∧ M < N → 
    (∃ (b : ℕ), b > 0 ∧ M = b * (2 * b - 1)) → 
    (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 10 → (List.sum (List.range (M - 1))) % k = 0) → False) ∧
  N = 2016 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_satisfying_conditions_l2036_203614


namespace NUMINAMATH_CALUDE_homework_problem_l2036_203624

theorem homework_problem (p t : ℕ) : 
  p ≥ 10 ∧ 
  p * t = (2 * p + 2) * (t + 1) →
  p * t = 60 :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_l2036_203624


namespace NUMINAMATH_CALUDE_parabola_translation_correct_l2036_203672

/-- Represents a parabola in the form y = a(x - h)² + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The original parabola y = 2x² --/
def original_parabola : Parabola := { a := 2, h := 0, k := 0 }

/-- The transformed parabola y = 2(x+4)² + 1 --/
def transformed_parabola : Parabola := { a := 2, h := -4, k := 1 }

/-- Represents a translation in 2D space --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The translation that should transform the original parabola to the transformed parabola --/
def correct_translation : Translation := { dx := -4, dy := 1 }

/-- Applies a translation to a parabola --/
def apply_translation (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a, h := p.h - t.dx, k := p.k + t.dy }

theorem parabola_translation_correct :
  apply_translation original_parabola correct_translation = transformed_parabola :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_correct_l2036_203672


namespace NUMINAMATH_CALUDE_dividend_calculation_l2036_203630

theorem dividend_calculation (D d Q R : ℕ) 
  (eq_condition : D = d * Q + R)
  (d_value : d = 17)
  (Q_value : Q = 9)
  (R_value : R = 9) :
  D = 162 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2036_203630


namespace NUMINAMATH_CALUDE_pencils_per_package_l2036_203601

theorem pencils_per_package (pens_per_package : ℕ) (total_pens : ℕ) (pencil_packages : ℕ) :
  pens_per_package = 12 →
  total_pens = 60 →
  total_pens / pens_per_package = pencil_packages →
  total_pens / pencil_packages = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_pencils_per_package_l2036_203601


namespace NUMINAMATH_CALUDE_sandy_shirt_cost_l2036_203681

/-- The amount Sandy spent on clothes -/
def total_spent : ℝ := 33.56

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℝ := 13.99

/-- The amount Sandy spent on a jacket -/
def jacket_cost : ℝ := 7.43

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℝ := total_spent - shorts_cost - jacket_cost

theorem sandy_shirt_cost : shirt_cost = 12.14 := by
  sorry

end NUMINAMATH_CALUDE_sandy_shirt_cost_l2036_203681


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l2036_203650

theorem two_digit_number_puzzle :
  ∃! n : ℕ,
    n ≥ 10 ∧ n < 100 ∧
    (n / 10 + n % 10 = 8) ∧
    (n - 36 = (n % 10) * 10 + (n / 10)) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l2036_203650


namespace NUMINAMATH_CALUDE_cab_driver_income_l2036_203653

theorem cab_driver_income (income2 income3 income4 income5 avg_income : ℕ)
  (h1 : income2 = 150)
  (h2 : income3 = 750)
  (h3 : income4 = 200)
  (h4 : income5 = 600)
  (h5 : avg_income = 400)
  (h6 : ∃ income1 : ℕ, (income1 + income2 + income3 + income4 + income5) / 5 = avg_income) :
  ∃ income1 : ℕ, income1 = 300 ∧ (income1 + income2 + income3 + income4 + income5) / 5 = avg_income :=
by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l2036_203653


namespace NUMINAMATH_CALUDE_opposite_points_on_number_line_l2036_203602

theorem opposite_points_on_number_line (A B : ℝ) :
  A < B →  -- A is to the left of B
  A = -B →  -- A and B are opposite numbers
  B - A = 6.4 →  -- The distance between A and B is 6.4
  A = -3.2 ∧ B = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_points_on_number_line_l2036_203602


namespace NUMINAMATH_CALUDE_baseball_cards_cost_l2036_203654

theorem baseball_cards_cost (football_pack_cost : ℝ) (pokemon_pack_cost : ℝ) (total_spent : ℝ)
  (h1 : football_pack_cost = 2.73)
  (h2 : pokemon_pack_cost = 4.01)
  (h3 : total_spent = 18.42) :
  total_spent - (2 * football_pack_cost + pokemon_pack_cost) = 8.95 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_cost_l2036_203654


namespace NUMINAMATH_CALUDE_amp_eight_five_plus_ten_l2036_203634

def amp (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem amp_eight_five_plus_ten : (amp 8 5) + 10 = 49 := by
  sorry

end NUMINAMATH_CALUDE_amp_eight_five_plus_ten_l2036_203634


namespace NUMINAMATH_CALUDE_f_derivative_at_neg_one_l2036_203632

noncomputable def f' (x : ℝ) : ℝ := 2 * Real.exp x / (Real.exp x - 1)

theorem f_derivative_at_neg_one :
  let f (x : ℝ) := (f' (-1)) * Real.exp x - x^2
  (deriv f) (-1) = 2 * Real.exp 1 / (Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_f_derivative_at_neg_one_l2036_203632


namespace NUMINAMATH_CALUDE_sunlovers_happy_days_l2036_203674

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2*D*(R^2 + 4) - 2*R*(D^2 + 4) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sunlovers_happy_days_l2036_203674


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2036_203631

theorem complex_magnitude_problem (z : ℂ) (h : (z + Complex.I) * (1 + Complex.I) = 1 - Complex.I) : 
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2036_203631


namespace NUMINAMATH_CALUDE_bolt_nut_balance_l2036_203639

theorem bolt_nut_balance (total_workers : ℕ) (bolts_per_worker : ℕ) (nuts_per_worker : ℕ) 
  (nuts_per_bolt : ℕ) (bolt_workers : ℕ) : 
  total_workers = 56 →
  bolts_per_worker = 16 →
  nuts_per_worker = 24 →
  nuts_per_bolt = 2 →
  bolt_workers ≤ total_workers →
  (2 * bolts_per_worker * bolt_workers = nuts_per_worker * (total_workers - bolt_workers)) ↔
  (bolts_per_worker * bolt_workers * nuts_per_bolt = nuts_per_worker * (total_workers - bolt_workers)) :=
by sorry

end NUMINAMATH_CALUDE_bolt_nut_balance_l2036_203639


namespace NUMINAMATH_CALUDE_luna_budget_l2036_203622

/-- Luna's monthly budget calculation -/
theorem luna_budget (house_rental food phone : ℝ) : 
  food = 0.6 * house_rental →
  phone = 0.1 * food →
  house_rental + food = 240 →
  house_rental + food + phone = 249 := by
  sorry

end NUMINAMATH_CALUDE_luna_budget_l2036_203622


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l2036_203661

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 15th term of the sequence is 15 -/
def Term15Is15 (a : ℕ → ℝ) : Prop := a 15 = 15

/-- The 16th term of the sequence is 21 -/
def Term16Is21 (a : ℕ → ℝ) : Prop := a 16 = 21

/-- The 3rd term of the sequence is -57 -/
def Term3IsNeg57 (a : ℕ → ℝ) : Prop := a 3 = -57

theorem arithmetic_sequence_theorem (a : ℕ → ℝ) :
  ArithmeticSequence a → Term15Is15 a → Term16Is21 a → Term3IsNeg57 a := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l2036_203661


namespace NUMINAMATH_CALUDE_zeros_of_continuous_function_l2036_203676

theorem zeros_of_continuous_function (f : ℝ → ℝ) (a b c : ℝ) 
  (h_cont : Continuous f) 
  (h_order : a < b ∧ b < c) 
  (h_sign1 : f a * f b < 0) 
  (h_sign2 : f b * f c < 0) : 
  ∃ (n : ℕ), n ≥ 2 ∧ Even n ∧ 
  (∃ (S : Finset ℝ), S.card = n ∧ (∀ x ∈ S, a < x ∧ x < c ∧ f x = 0)) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_continuous_function_l2036_203676


namespace NUMINAMATH_CALUDE_council_vote_difference_l2036_203611

theorem council_vote_difference (total_members : ℕ) 
  (initial_for initial_against : ℕ) 
  (revote_for revote_against : ℕ) : 
  total_members = 500 →
  initial_for + initial_against = total_members →
  initial_against > initial_for →
  revote_for + revote_against = total_members →
  revote_for - revote_against = 3 * (initial_against - initial_for) →
  revote_for = (13 * initial_against) / 12 →
  revote_for - initial_for = 40 := by
sorry

end NUMINAMATH_CALUDE_council_vote_difference_l2036_203611


namespace NUMINAMATH_CALUDE_cubic_identity_l2036_203608

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l2036_203608


namespace NUMINAMATH_CALUDE_equation_solution_l2036_203664

theorem equation_solution :
  let f (x : ℝ) := (7 * x + 3) / (3 * x^2 + 7 * x - 6)
  let g (x : ℝ) := (3 * x) / (3 * x - 2)
  let sol₁ := (-1 + Real.sqrt 10) / 3
  let sol₂ := (-1 - Real.sqrt 10) / 3
  ∀ x : ℝ, x ≠ 2/3 →
    (f x = g x ↔ x = sol₁ ∨ x = sol₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2036_203664


namespace NUMINAMATH_CALUDE_intersection_sum_l2036_203652

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 5*x + 2
def g (x y : ℝ) : Prop := x + 5*y = 5

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {p | f p.1 = p.2 ∧ g p.1 p.2}

-- State the theorem
theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    x₁ + x₂ + x₃ = 0 ∧
    y₁ + y₂ + y₃ = 3 :=
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2036_203652


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l2036_203616

/-- Given a quadratic equation with parameter k, prove that k = 1 under specific conditions -/
theorem quadratic_equation_k_value (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ 
    x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0 ∧
    x1 + x2 + 2*x1*x2 = 1) →
  k = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l2036_203616


namespace NUMINAMATH_CALUDE_root_in_interval_implies_k_range_l2036_203666

/-- Given a quadratic function f(x) = x^2 + (1-k)x - k, if f has a root in the interval (2, 3), 
    then k is in the open interval (2, 3) -/
theorem root_in_interval_implies_k_range (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + (1-k)*x - k
  (∃ x ∈ Set.Ioo 2 3, f x = 0) → k ∈ Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_k_range_l2036_203666


namespace NUMINAMATH_CALUDE_prime_squared_plus_17_mod_12_l2036_203659

theorem prime_squared_plus_17_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  (p^2 + 17) % 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_plus_17_mod_12_l2036_203659


namespace NUMINAMATH_CALUDE_frequency_calculation_l2036_203607

/-- Given a sample capacity and a frequency rate, calculate the frequency of a group of samples. -/
def calculate_frequency (sample_capacity : ℕ) (frequency_rate : ℚ) : ℚ :=
  frequency_rate * sample_capacity

/-- Theorem: Given a sample capacity of 32 and a frequency rate of 0.125, the frequency is 4. -/
theorem frequency_calculation :
  let sample_capacity : ℕ := 32
  let frequency_rate : ℚ := 1/8
  calculate_frequency sample_capacity frequency_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_frequency_calculation_l2036_203607


namespace NUMINAMATH_CALUDE_smallest_coloring_number_l2036_203644

/-- A coloring of positive integers -/
def Coloring (k : ℕ) := ℕ+ → Fin k

/-- A function from positive integers to positive integers -/
def IntegerFunction := ℕ+ → ℕ+

/-- Condition 1: For all n, m of the same color, f(n+m) = f(n) + f(m) -/
def SameColorAdditive (c : Coloring k) (f : IntegerFunction) : Prop :=
  ∀ n m : ℕ+, c n = c m → f (n + m) = f n + f m

/-- Condition 2: There exist n, m such that f(n+m) ≠ f(n) + f(m) -/
def ExistsNonAdditive (f : IntegerFunction) : Prop :=
  ∃ n m : ℕ+, f (n + m) ≠ f n + f m

/-- The main theorem statement -/
theorem smallest_coloring_number :
  (∃ k : ℕ+, ∃ c : Coloring k, ∃ f : IntegerFunction,
    SameColorAdditive c f ∧ ExistsNonAdditive f) ∧
  (∀ k : ℕ+, k < 3 →
    ¬∃ c : Coloring k, ∃ f : IntegerFunction,
      SameColorAdditive c f ∧ ExistsNonAdditive f) :=
sorry

end NUMINAMATH_CALUDE_smallest_coloring_number_l2036_203644


namespace NUMINAMATH_CALUDE_congruence_solution_l2036_203694

theorem congruence_solution : ∃ x : ℤ, x ≡ 1 [ZMOD 7] ∧ x ≡ 2 [ZMOD 11] :=
by
  use 57
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2036_203694


namespace NUMINAMATH_CALUDE_six_digit_multiple_of_three_l2036_203617

theorem six_digit_multiple_of_three : ∃ (n : ℕ), 325473 = 3 * n := by
  sorry

end NUMINAMATH_CALUDE_six_digit_multiple_of_three_l2036_203617


namespace NUMINAMATH_CALUDE_percentage_calculation_l2036_203636

theorem percentage_calculation : 
  (0.47 * 1442 - 0.36 * 1412) + 65 = 234.42 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2036_203636


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_47_l2036_203677

def is_multiple (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

theorem least_positive_integer_multiple_47 :
  ∃! x : ℕ+, (x : ℤ) = 5 ∧ 
  (∀ y : ℕ+, y < x → ¬ is_multiple ((2 * y : ℤ)^2 + 2 * 37 * (2 * y) + 37^2) 47) ∧
  is_multiple ((2 * x : ℤ)^2 + 2 * 37 * (2 * x) + 37^2) 47 :=
sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_47_l2036_203677


namespace NUMINAMATH_CALUDE_sqrt_two_value_l2036_203660

def f_property (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂

theorem sqrt_two_value (f : ℝ → ℝ) (h1 : f_property f) (h2 : f 8 = 3) :
  f (Real.sqrt 2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_value_l2036_203660


namespace NUMINAMATH_CALUDE_pencil_theorem_l2036_203613

def pencil_problem (anna_pencils : ℕ) (harry_pencils : ℕ) (lost_pencils : ℕ) : Prop :=
  anna_pencils = 50 ∧
  harry_pencils = 2 * anna_pencils ∧
  harry_pencils - lost_pencils = 81

theorem pencil_theorem : 
  ∃ (anna_pencils harry_pencils lost_pencils : ℕ),
    pencil_problem anna_pencils harry_pencils lost_pencils ∧ lost_pencils = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_theorem_l2036_203613


namespace NUMINAMATH_CALUDE_number_of_laborers_l2036_203651

/-- Proves that the number of laborers is 24 given the salary information --/
theorem number_of_laborers (total_avg : ℝ) (num_supervisors : ℕ) (supervisor_avg : ℝ) (laborer_avg : ℝ) :
  total_avg = 1250 →
  num_supervisors = 6 →
  supervisor_avg = 2450 →
  laborer_avg = 950 →
  ∃ (num_laborers : ℕ), 
    (num_laborers : ℝ) * laborer_avg + (num_supervisors : ℝ) * supervisor_avg = 
    (num_laborers + num_supervisors : ℝ) * total_avg ∧
    num_laborers = 24 :=
by sorry

end NUMINAMATH_CALUDE_number_of_laborers_l2036_203651


namespace NUMINAMATH_CALUDE_pet_food_difference_l2036_203647

theorem pet_food_difference (dog_food cat_food : ℕ) 
  (h1 : dog_food = 600) 
  (h2 : cat_food = 327) : 
  dog_food - cat_food = 273 := by
sorry

end NUMINAMATH_CALUDE_pet_food_difference_l2036_203647


namespace NUMINAMATH_CALUDE_number_of_pupils_in_class_l2036_203619

/-- 
Given a class where:
1. A pupil's marks were wrongly entered as 67 instead of 45.
2. The wrong entry caused the average marks for the class to increase by half a mark.

Prove that the number of pupils in the class is 44.
-/
theorem number_of_pupils_in_class : ℕ := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_in_class_l2036_203619


namespace NUMINAMATH_CALUDE_expression_simplification_l2036_203657

theorem expression_simplification (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h : a + b + c = d) :
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2) = 0 := by
sorry


end NUMINAMATH_CALUDE_expression_simplification_l2036_203657


namespace NUMINAMATH_CALUDE_september_1_2017_is_friday_l2036_203627

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

def march_19_2017 : Date :=
  { year := 2017, month := 3, day := 19 }

def september_1_2017 : Date :=
  { year := 2017, month := 9, day := 1 }

/-- Returns the day of the week for a given date -/
def dayOfWeek (d : Date) : DayOfWeek :=
  sorry

/-- Calculates the number of days between two dates -/
def daysBetween (d1 d2 : Date) : Nat :=
  sorry

theorem september_1_2017_is_friday :
  dayOfWeek march_19_2017 = DayOfWeek.Sunday →
  dayOfWeek september_1_2017 = DayOfWeek.Friday :=
by
  sorry

#check september_1_2017_is_friday

end NUMINAMATH_CALUDE_september_1_2017_is_friday_l2036_203627


namespace NUMINAMATH_CALUDE_tan_inequality_l2036_203693

theorem tan_inequality (x : ℝ) (h : 0 ≤ x ∧ x < 1) :
  (2 / Real.pi) * (x / (1 - x)) ≤ Real.tan (Real.pi * x / 2) ∧
  Real.tan (Real.pi * x / 2) ≤ (Real.pi / 2) * (x / (1 - x)) := by
  sorry

end NUMINAMATH_CALUDE_tan_inequality_l2036_203693


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l2036_203604

theorem arithmetic_progression_sum (a d : ℝ) (n : ℕ) : 
  (a + 10 * d = 5.25) →
  (a + 6 * d = 3.25) →
  (n : ℝ) * (2 * a + (n - 1) * d) / 2 = 56.25 →
  n = 15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l2036_203604


namespace NUMINAMATH_CALUDE_rectangle_area_relation_l2036_203692

/-- Given a rectangle with area 10 and adjacent sides x and y, 
    prove that the relationship between x and y is y = 10/x -/
theorem rectangle_area_relation (x y : ℝ) (h : x * y = 10) : y = 10 / x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_relation_l2036_203692


namespace NUMINAMATH_CALUDE_log_product_equals_24_l2036_203667

theorem log_product_equals_24 :
  Real.log 9 / Real.log 2 * (Real.log 16 / Real.log 3) * (Real.log 27 / Real.log 7) = 24 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_24_l2036_203667


namespace NUMINAMATH_CALUDE_pipe_filling_time_l2036_203603

theorem pipe_filling_time (fill_time_A : ℝ) (fill_time_B : ℝ) (combined_time : ℝ) :
  (fill_time_B = fill_time_A / 6) →
  (combined_time = 3.75) →
  (1 / fill_time_A + 1 / fill_time_B = 1 / combined_time) →
  fill_time_A = 26.25 := by
  sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l2036_203603


namespace NUMINAMATH_CALUDE_total_grain_calculation_l2036_203670

/-- The amount of grain in kilograms transported from the first warehouse. -/
def transported : ℕ := 2500

/-- The amount of grain in kilograms in the second warehouse. -/
def second_warehouse : ℕ := 50200

/-- The total amount of grain in kilograms in both warehouses. -/
def total_grain : ℕ := second_warehouse + (second_warehouse + transported)

theorem total_grain_calculation :
  total_grain = 102900 :=
by sorry

end NUMINAMATH_CALUDE_total_grain_calculation_l2036_203670


namespace NUMINAMATH_CALUDE_solve_exponential_system_l2036_203605

theorem solve_exponential_system (x y : ℝ) 
  (h1 : (6 : ℝ) ^ (x + y) = 36)
  (h2 : (6 : ℝ) ^ (x + 5 * y) = 216) :
  x = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_system_l2036_203605


namespace NUMINAMATH_CALUDE_complex_on_line_l2036_203678

theorem complex_on_line (a : ℝ) : 
  let z : ℂ := (2 + a * Complex.I) / (1 + Complex.I)
  (z.re = -z.im) → a = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_on_line_l2036_203678


namespace NUMINAMATH_CALUDE_monogram_count_l2036_203675

/-- The number of letters in the alphabet excluding 'A' -/
def n : ℕ := 25

/-- The number of letters we need to choose for middle and last initials -/
def k : ℕ := 2

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The number of ways to choose two distinct letters from 25 letters in alphabetical order is 300 -/
theorem monogram_count : choose n k = 300 := by sorry

end NUMINAMATH_CALUDE_monogram_count_l2036_203675


namespace NUMINAMATH_CALUDE_min_value_expression_l2036_203687

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = 1) :
  1/a + 1/(2*b) ≥ 9/2 ∧ (1/a + 1/(2*b) = 9/2 ↔ a = 1/3 ∧ b = 1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2036_203687


namespace NUMINAMATH_CALUDE_p_sufficient_for_q_l2036_203658

theorem p_sufficient_for_q : ∀ (x y : ℝ),
  (x - 1)^2 + (y - 1)^2 ≤ 2 →
  y ≥ x - 1 ∧ y ≥ 1 - x ∧ y ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_for_q_l2036_203658


namespace NUMINAMATH_CALUDE_brian_bought_22_pencils_l2036_203629

/-- The number of pencils Brian bought -/
def pencils_bought (initial : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_away)

/-- Theorem stating that Brian bought 22 pencils -/
theorem brian_bought_22_pencils :
  pencils_bought 39 18 43 = 22 := by
  sorry

end NUMINAMATH_CALUDE_brian_bought_22_pencils_l2036_203629


namespace NUMINAMATH_CALUDE_inequality_proof_l2036_203699

theorem inequality_proof (a : ℝ) (h : -1 < a ∧ a < 0) : -a > a^2 ∧ a^2 > -a^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2036_203699


namespace NUMINAMATH_CALUDE_rectangular_floor_shorter_side_l2036_203655

theorem rectangular_floor_shorter_side (floor_length : ℝ) (floor_width : ℝ) 
  (carpet_side : ℝ) (carpet_cost : ℝ) (total_cost : ℝ) :
  floor_length = 10 →
  carpet_side = 2 →
  carpet_cost = 15 →
  total_cost = 225 →
  floor_width * floor_length = (total_cost / carpet_cost) * carpet_side^2 →
  floor_width = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangular_floor_shorter_side_l2036_203655


namespace NUMINAMATH_CALUDE_license_plate_count_l2036_203648

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of even digits -/
def num_even_digits : ℕ := 5

/-- The number of odd digits -/
def num_odd_digits : ℕ := 5

/-- The total number of license plates -/
def total_plates : ℕ := num_letters ^ 3 * num_even_digits * num_odd_digits * num_even_digits

theorem license_plate_count :
  total_plates = 2197000 := by sorry

end NUMINAMATH_CALUDE_license_plate_count_l2036_203648


namespace NUMINAMATH_CALUDE_cow_ratio_theorem_l2036_203696

theorem cow_ratio_theorem (big_cows small_cows : ℕ) 
  (h : big_cows * 7 = small_cows * 6) : 
  (small_cows - big_cows : ℚ) / small_cows = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cow_ratio_theorem_l2036_203696


namespace NUMINAMATH_CALUDE_gas_refill_proof_l2036_203649

def gas_problem (initial_gas tank_capacity gas_to_store gas_to_doctor : ℕ) : Prop :=
  let remaining_gas := initial_gas - gas_to_store - gas_to_doctor
  tank_capacity - remaining_gas = tank_capacity - (initial_gas - gas_to_store - gas_to_doctor)

theorem gas_refill_proof (initial_gas tank_capacity gas_to_store gas_to_doctor : ℕ) 
  (h1 : initial_gas ≥ gas_to_store + gas_to_doctor)
  (h2 : tank_capacity ≥ initial_gas) :
  gas_problem initial_gas tank_capacity gas_to_store gas_to_doctor :=
by
  sorry

#check gas_refill_proof 10 12 6 2

end NUMINAMATH_CALUDE_gas_refill_proof_l2036_203649


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2036_203682

/-- Given a > 0 and a ≠ 1, prove that f(x) = a^(x-1) + 3 passes through (1, 4) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 3
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2036_203682


namespace NUMINAMATH_CALUDE_chinese_characters_equation_l2036_203695

theorem chinese_characters_equation (x : ℝ) 
  (h1 : x > 100) -- Ensure x - 100 is positive
  (h2 : x ≠ 0) -- Ensure division by x is valid
  : (8000 / x = 6000 / (x - 100)) ↔ 
    (∃ (days : ℝ), 
      days > 0 ∧ 
      days * x = 8000 ∧ 
      days * (x - 100) = 6000) := by
sorry

end NUMINAMATH_CALUDE_chinese_characters_equation_l2036_203695


namespace NUMINAMATH_CALUDE_largest_nice_sequence_l2036_203669

/-- A sequence is nice if it satisfies the given conditions -/
def IsNice (a : ℕ → ℝ) (n : ℕ) : Prop :=
  n ≥ 1 ∧ 
  a 0 + a 1 = -1 / n ∧ 
  ∀ k : ℕ, k ≥ 1 → (a (k-1) + a k) * (a k + a (k+1)) = a (k-1) + a (k+1)

/-- The largest N for which a nice sequence of length N+1 exists is equal to n -/
theorem largest_nice_sequence (n : ℕ) : 
  n ≥ 1 → 
  (∃ (N : ℕ) (a : ℕ → ℝ), IsNice a n ∧ N = n) ∧ 
  (∀ (M : ℕ) (a : ℕ → ℝ), M > n → ¬ IsNice a n) :=
sorry

end NUMINAMATH_CALUDE_largest_nice_sequence_l2036_203669
