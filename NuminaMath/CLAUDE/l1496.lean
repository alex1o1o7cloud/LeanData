import Mathlib

namespace NUMINAMATH_CALUDE_circle_center_distance_l1496_149662

/-- The distance between the center of the circle defined by x^2 + y^2 = 8x - 2y + 16 and the point (-3, 4) is √74. -/
theorem circle_center_distance : 
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 - 8*x + 2*y - 16 = 0
  let center := (fun (x y : ℝ) => circle_eq x y ∧ 
                 ∀ (x' y' : ℝ), circle_eq x' y' → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2)
  let distance := fun (x₁ y₁ x₂ y₂ : ℝ) => Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  ∃ (cx cy : ℝ), center cx cy ∧ distance cx cy (-3) 4 = Real.sqrt 74 := by sorry

end NUMINAMATH_CALUDE_circle_center_distance_l1496_149662


namespace NUMINAMATH_CALUDE_triangle_tangent_sum_properties_l1496_149614

noncomputable def tanHalfAngle (θ : Real) : Real := Real.tan (θ / 2)

def isAcute (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

def isObtuse (θ : Real) : Prop := Real.pi / 2 < θ ∧ θ < Real.pi

theorem triangle_tangent_sum_properties
  (A B C : Real)
  (triangle_angles : A + B + C = Real.pi)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C) :
  let S := (tanHalfAngle A)^2 + (tanHalfAngle B)^2 + (tanHalfAngle C)^2
  let T := tanHalfAngle A + tanHalfAngle B + tanHalfAngle C
  -- Relationship between S and T
  (T^2 = S + 2) →
  -- 1. For acute triangles
  ((isAcute A ∧ isAcute B ∧ isAcute C) → S < 2) ∧
  -- 2. For obtuse triangles with obtuse angle ≥ 2arctan(4/3)
  ((isObtuse A ∨ isObtuse B ∨ isObtuse C) ∧
   (max A (max B C) ≥ 2 * Real.arctan (4/3)) → S ≥ 2) ∧
  -- 3. For obtuse triangles with obtuse angle < 2arctan(4/3)
  ((isObtuse A ∨ isObtuse B ∨ isObtuse C) ∧
   (max A (max B C) < 2 * Real.arctan (4/3)) →
   ∃ (A' B' C' : Real),
     A' + B' + C' = Real.pi ∧
     (isObtuse A' ∨ isObtuse B' ∨ isObtuse C') ∧
     max A' (max B' C') < 2 * Real.arctan (4/3) ∧
     (tanHalfAngle A')^2 + (tanHalfAngle B')^2 + (tanHalfAngle C')^2 > 2 ∧
     ∃ (A'' B'' C'' : Real),
       A'' + B'' + C'' = Real.pi ∧
       (isObtuse A'' ∨ isObtuse B'' ∨ isObtuse C'') ∧
       max A'' (max B'' C'') < 2 * Real.arctan (4/3) ∧
       (tanHalfAngle A'')^2 + (tanHalfAngle B'')^2 + (tanHalfAngle C'')^2 < 2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_sum_properties_l1496_149614


namespace NUMINAMATH_CALUDE_problem_solution_l1496_149671

theorem problem_solution (p_xavier p_yvonne p_zelda : ℚ) 
  (hx : p_xavier = 1/4) 
  (hy : p_yvonne = 2/3) 
  (hz : p_zelda = 5/8) : 
  p_xavier * p_yvonne * (1 - p_zelda) = 1/16 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1496_149671


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l1496_149635

/-- The positive difference between solutions of the quadratic equation x^2 - 5x + m = 13 + (x+5) -/
theorem quadratic_solution_difference (m : ℝ) (h : 27 - m > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (x₁^2 - 5*x₁ + m = 13 + (x₁ + 5)) ∧
  (x₂^2 - 5*x₂ + m = 13 + (x₂ + 5)) ∧
  |x₁ - x₂| = 2 * Real.sqrt (27 - m) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l1496_149635


namespace NUMINAMATH_CALUDE_delta_curve_circumscribed_triangle_height_l1496_149601

/-- A Δ-curve is a curve with the property that all equilateral triangles circumscribing it have the same height -/
class DeltaCurve (α : Type*) [MetricSpace α] where
  is_delta_curve : α → Prop

variable {α : Type*} [MetricSpace α]

/-- An equilateral triangle -/
structure EquilateralTriangle (α : Type*) [MetricSpace α] where
  points : Fin 3 → α
  is_equilateral : ∀ i j : Fin 3, dist (points i) (points j) = dist (points 0) (points 1)

/-- A point lies on a line -/
def PointOnLine (p : α) (l : Set α) : Prop := p ∈ l

/-- A triangle circumscribes a curve if each side of the triangle touches the curve at exactly one point -/
def Circumscribes (t : EquilateralTriangle α) (k : Set α) : Prop :=
  ∃ a b c : α, a ∈ k ∧ b ∈ k ∧ c ∈ k ∧
    PointOnLine a {x | dist x (t.points 0) = dist x (t.points 1)} ∧
    PointOnLine b {x | dist x (t.points 1) = dist x (t.points 2)} ∧
    PointOnLine c {x | dist x (t.points 2) = dist x (t.points 0)}

/-- The height of an equilateral triangle -/
def Height (t : EquilateralTriangle α) : ℝ := sorry

/-- The main theorem -/
theorem delta_curve_circumscribed_triangle_height 
  (k : Set α) [DeltaCurve α] (t : EquilateralTriangle α) 
  (h_circumscribes : Circumscribes t k) :
  ∀ (t₁ : EquilateralTriangle α),
    (∃ a b c : α, a ∈ k ∧ b ∈ k ∧ c ∈ k ∧
      PointOnLine a {x | dist x (t₁.points 0) = dist x (t₁.points 1)} ∧
      PointOnLine b {x | dist x (t₁.points 1) = dist x (t₁.points 2)} ∧
      PointOnLine c {x | dist x (t₁.points 2) = dist x (t₁.points 0)}) →
    Height t₁ ≤ Height t :=
sorry

end NUMINAMATH_CALUDE_delta_curve_circumscribed_triangle_height_l1496_149601


namespace NUMINAMATH_CALUDE_blue_parrots_l1496_149675

theorem blue_parrots (total : ℕ) (red_fraction : ℚ) (green_fraction : ℚ) :
  total = 120 →
  red_fraction = 2/3 →
  green_fraction = 1/6 →
  (total : ℚ) * (1 - (red_fraction + green_fraction)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_blue_parrots_l1496_149675


namespace NUMINAMATH_CALUDE_positive_sum_y_z_l1496_149680

theorem positive_sum_y_z (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 1 < z ∧ z < 2) : 
  y + z > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_y_z_l1496_149680


namespace NUMINAMATH_CALUDE_mancino_garden_width_is_5_l1496_149684

/-- The width of Mancino's gardens -/
def mancino_garden_width : ℝ := 5

/-- The number of Mancino's gardens -/
def mancino_garden_count : ℕ := 3

/-- The length of Mancino's gardens -/
def mancino_garden_length : ℝ := 16

/-- The number of Marquita's gardens -/
def marquita_garden_count : ℕ := 2

/-- The length of Marquita's gardens -/
def marquita_garden_length : ℝ := 8

/-- The width of Marquita's gardens -/
def marquita_garden_width : ℝ := 4

/-- The total area of all gardens -/
def total_garden_area : ℝ := 304

theorem mancino_garden_width_is_5 :
  mancino_garden_width = 5 ∧
  mancino_garden_count * mancino_garden_length * mancino_garden_width +
  marquita_garden_count * marquita_garden_length * marquita_garden_width =
  total_garden_area :=
by sorry

end NUMINAMATH_CALUDE_mancino_garden_width_is_5_l1496_149684


namespace NUMINAMATH_CALUDE_fluffy_spotted_cats_ratio_l1496_149618

theorem fluffy_spotted_cats_ratio (total_cats : ℕ) (fluffy_spotted_cats : ℕ) :
  total_cats = 120 →
  fluffy_spotted_cats = 10 →
  (total_cats / 3 : ℚ) = (total_cats / 3 : ℕ) →
  (fluffy_spotted_cats : ℚ) / (total_cats / 3 : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fluffy_spotted_cats_ratio_l1496_149618


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1496_149679

theorem fractional_equation_solution (x : ℝ) (h : x * (x - 2) ≠ 0) :
  (4 / (x^2 - 2*x) + 1 / x = 3 / (x - 2)) ↔ (x = 1) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1496_149679


namespace NUMINAMATH_CALUDE_box_volume_formula_l1496_149616

/-- The volume of a box formed by cutting rectangles from a sheet's corners. -/
def box_volume (x y : ℝ) : ℝ :=
  (16 - 2*x) * (12 - 2*y) * y

/-- Theorem stating the volume of the box in terms of x and y -/
theorem box_volume_formula (x y : ℝ) :
  box_volume x y = 4*x*y^2 - 24*x*y + 192*y - 32*y^2 := by
  sorry

#check box_volume_formula

end NUMINAMATH_CALUDE_box_volume_formula_l1496_149616


namespace NUMINAMATH_CALUDE_symmetric_trapezoid_construction_l1496_149641

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a trapezoid
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define symmetry for a trapezoid
def isSymmetricTrapezoid (t : Trapezoid) : Prop :=
  -- Add conditions for symmetry here
  sorry

-- Define the construction function
def constructSymmetricTrapezoid (c : Circle) (sideLength : ℝ) : Trapezoid :=
  sorry

-- Theorem statement
theorem symmetric_trapezoid_construction
  (c : Circle) (sideLength : ℝ) :
  isSymmetricTrapezoid (constructSymmetricTrapezoid c sideLength) :=
sorry

end NUMINAMATH_CALUDE_symmetric_trapezoid_construction_l1496_149641


namespace NUMINAMATH_CALUDE_right_triangle_circumscribed_circle_perimeter_l1496_149636

theorem right_triangle_circumscribed_circle_perimeter 
  (r : ℝ) (h : ℝ) (a b : ℝ) :
  r = 4 →
  h = 26 →
  a^2 + b^2 = h^2 →
  a * b = 4 * (a + b + h) →
  a + b + h = 60 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_circumscribed_circle_perimeter_l1496_149636


namespace NUMINAMATH_CALUDE_part1_part2_l1496_149625

-- Define the polynomials A and B
def A (x : ℝ) : ℝ := x^2 - x
def B (m : ℝ) (x : ℝ) : ℝ := m * x + 1

-- Part 1: Prove that when ■ = 2, 2A - B = 2x^2 - 4x - 1
theorem part1 (x : ℝ) : 2 * A x - B 2 x = 2 * x^2 - 4 * x - 1 := by
  sorry

-- Part 2: Prove that when A - B does not contain x terms, ■ = -1
theorem part2 (x : ℝ) : (∀ m : ℝ, A x - B m x = (-1 : ℝ)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_part1_part2_l1496_149625


namespace NUMINAMATH_CALUDE_inheritance_solution_l1496_149656

def inheritance_problem (x : ℝ) : Prop :=
  let federal_tax_rate := 0.25
  let state_tax_rate := 0.15
  let total_tax := 12000
  (federal_tax_rate * x) + (state_tax_rate * (1 - federal_tax_rate) * x) = total_tax

theorem inheritance_solution :
  ∃ x : ℝ, inheritance_problem x ∧ (round x = 33103) :=
sorry

end NUMINAMATH_CALUDE_inheritance_solution_l1496_149656


namespace NUMINAMATH_CALUDE_smallest_divisible_by_hundred_million_l1496_149633

/-- The smallest positive integer n such that the nth term of a geometric sequence
    with first term 5/6 and second term 25 is divisible by 100 million. -/
theorem smallest_divisible_by_hundred_million : ℕ :=
  let a₁ : ℚ := 5 / 6  -- First term
  let a₂ : ℚ := 25     -- Second term
  let r : ℚ := a₂ / a₁ -- Common ratio
  let aₙ : ℕ → ℚ := λ n => r ^ (n - 1) * a₁  -- nth term
  9  -- The smallest n (to be proved)

#check smallest_divisible_by_hundred_million

end NUMINAMATH_CALUDE_smallest_divisible_by_hundred_million_l1496_149633


namespace NUMINAMATH_CALUDE_largest_d_for_two_in_range_l1496_149657

/-- The function g(x) defined as x^2 - 6x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + d

/-- Theorem stating that the largest value of d for which 2 is in the range of g(x) is 11 -/
theorem largest_d_for_two_in_range :
  (∃ (d : ℝ), ∀ (e : ℝ), (∃ (x : ℝ), g d x = 2) → (e ≤ d)) ∧
  (∃ (x : ℝ), g 11 x = 2) :=
sorry

end NUMINAMATH_CALUDE_largest_d_for_two_in_range_l1496_149657


namespace NUMINAMATH_CALUDE_dodecagon_diagonal_intersection_probability_l1496_149608

/-- A regular dodecagon is a 12-sided polygon with all sides equal and all angles equal. -/
def RegularDodecagon : Type := Unit

/-- A diagonal of a regular dodecagon is a line segment connecting two non-adjacent vertices. -/
def Diagonal (d : RegularDodecagon) : Type := Unit

/-- The probability that two randomly chosen diagonals of a regular dodecagon intersect inside the polygon. -/
def intersectionProbability (d : RegularDodecagon) : ℚ :=
  165 / 287

/-- Theorem: The probability that the intersection of two randomly chosen diagonals 
    of a regular dodecagon lies inside the polygon is 165/287. -/
theorem dodecagon_diagonal_intersection_probability (d : RegularDodecagon) :
  intersectionProbability d = 165 / 287 :=
by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonal_intersection_probability_l1496_149608


namespace NUMINAMATH_CALUDE_parking_garage_weekly_rate_l1496_149666

theorem parking_garage_weekly_rate :
  let monthly_rate : ℕ := 35
  let months_per_year : ℕ := 12
  let weeks_per_year : ℕ := 52
  let yearly_savings : ℕ := 100
  let weekly_rate : ℚ := (monthly_rate * months_per_year + yearly_savings) / weeks_per_year
  weekly_rate = 10 := by
sorry

end NUMINAMATH_CALUDE_parking_garage_weekly_rate_l1496_149666


namespace NUMINAMATH_CALUDE_fraction_sum_integer_implies_fractions_integer_l1496_149651

theorem fraction_sum_integer_implies_fractions_integer 
  (x y : ℕ+) 
  (h : ∃ (k : ℤ), (x.val^2 - 1 : ℤ) / (y.val + 1) + (y.val^2 - 1 : ℤ) / (x.val + 1) = k) :
  (∃ (m : ℤ), (x.val^2 - 1 : ℤ) / (y.val + 1) = m) ∧ 
  (∃ (n : ℤ), (y.val^2 - 1 : ℤ) / (x.val + 1) = n) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_implies_fractions_integer_l1496_149651


namespace NUMINAMATH_CALUDE_nonstudent_ticket_price_l1496_149693

theorem nonstudent_ticket_price
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (student_price : ℕ)
  (student_tickets : ℕ)
  (h1 : total_tickets = 821)
  (h2 : total_revenue = 1933)
  (h3 : student_price = 2)
  (h4 : student_tickets = 530)
  (h5 : student_tickets < total_tickets) :
  let nonstudent_tickets : ℕ := total_tickets - student_tickets
  let nonstudent_price : ℕ := (total_revenue - student_price * student_tickets) / nonstudent_tickets
  nonstudent_price = 3 := by
sorry

end NUMINAMATH_CALUDE_nonstudent_ticket_price_l1496_149693


namespace NUMINAMATH_CALUDE_triangle_perimeter_proof_l1496_149659

theorem triangle_perimeter_proof (a b c : ℝ) (h1 : a = 7) (h2 : b = 10) (h3 : c = 15) :
  a + b + c = 32 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_proof_l1496_149659


namespace NUMINAMATH_CALUDE_multiple_indecomposable_factorizations_l1496_149690

/-- The set V_n for a given n -/
def V_n (n : ℕ) : Set ℕ := {m : ℕ | ∃ k : ℕ+, m = 1 + k * n}

/-- A number is indecomposable in V_n if it cannot be expressed as the product of two members of V_n -/
def Indecomposable (m : ℕ) (n : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ a b : ℕ, a ∈ V_n n → b ∈ V_n n → a * b ≠ m

/-- There exists a number in V_n with multiple indecomposable factorizations -/
theorem multiple_indecomposable_factorizations (n : ℕ) (h : n > 2) :
  ∃ m : ℕ, m ∈ V_n n ∧
    ∃ (a b c d : ℕ),
      Indecomposable a n ∧ Indecomposable b n ∧ Indecomposable c n ∧ Indecomposable d n ∧
      a * b = m ∧ c * d = m ∧ (a ≠ c ∨ b ≠ d) :=
  sorry

end NUMINAMATH_CALUDE_multiple_indecomposable_factorizations_l1496_149690


namespace NUMINAMATH_CALUDE_skating_time_for_seventh_day_l1496_149660

def skating_minutes_first_four_days : ℕ := 80
def skating_minutes_next_two_days : ℕ := 100
def total_days : ℕ := 7
def target_average : ℕ := 100

theorem skating_time_for_seventh_day :
  let total_minutes_six_days := 4 * skating_minutes_first_four_days + 2 * skating_minutes_next_two_days
  let required_total_minutes := total_days * target_average
  required_total_minutes - total_minutes_six_days = 180 := by
  sorry

end NUMINAMATH_CALUDE_skating_time_for_seventh_day_l1496_149660


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1496_149612

theorem arithmetic_mean_problem (a b c : ℝ) :
  (a + b + c + 97) / 4 = 85 →
  (a + b + c) / 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1496_149612


namespace NUMINAMATH_CALUDE_least_number_to_add_l1496_149672

def problem (x : ℕ) : Prop :=
  let lcm := 7 * 11 * 13 * 17 * 19
  (∃ k : ℕ, (625573 + x) = k * lcm) ∧
  (∀ y : ℕ, y < x → ¬∃ k : ℕ, (625573 + y) = k * lcm)

theorem least_number_to_add : problem 21073 := by
  sorry

end NUMINAMATH_CALUDE_least_number_to_add_l1496_149672


namespace NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l1496_149669

/-- A linear function y = mx + b, where m is the slope and b is the y-intercept -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- The first quadrant of the Cartesian plane -/
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 > 0 ∧ p.2 > 0}

/-- Theorem: The linear function y = -2x - 3 does not pass through the first quadrant -/
theorem linear_function_not_in_first_quadrant :
  let f : LinearFunction := ⟨-2, -3⟩
  ∀ x y : ℝ, y = f.m * x + f.b → (x, y) ∉ FirstQuadrant :=
by
  sorry


end NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l1496_149669


namespace NUMINAMATH_CALUDE_range_of_p_l1496_149682

noncomputable def p (x : ℝ) : ℝ := x^6 + 6*x^3 + 9

theorem range_of_p :
  Set.range (fun (x : ℝ) ↦ p x) = Set.Ici 9 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l1496_149682


namespace NUMINAMATH_CALUDE_fraction_sign_l1496_149698

theorem fraction_sign (a b : ℝ) (ha : a > 0) (hb : b < 0) : a / b < 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sign_l1496_149698


namespace NUMINAMATH_CALUDE_smallest_m_for_inequality_l1496_149650

theorem smallest_m_for_inequality : 
  ∃ (m : ℝ), (∀ (a b c : ℕ+), a + b + c = 1 → 
    m * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1) ∧ 
  (∀ (m' : ℝ), m' < m → 
    ∃ (a b c : ℕ+), a + b + c = 1 ∧ 
    m' * (a^3 + b^3 + c^3) < 6 * (a^2 + b^2 + c^2) + 1) ∧
  m = 27 := by
sorry

end NUMINAMATH_CALUDE_smallest_m_for_inequality_l1496_149650


namespace NUMINAMATH_CALUDE_cubic_increasing_minor_premise_l1496_149619

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define what it means for a function to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the concept of a minor premise in a deduction
def IsMinorPremise (statement : Prop) (conclusion : Prop) : Prop :=
  statement → conclusion

-- Theorem statement
theorem cubic_increasing_minor_premise :
  IsMinorPremise (IsIncreasing f) (IsIncreasing f) :=
sorry

end NUMINAMATH_CALUDE_cubic_increasing_minor_premise_l1496_149619


namespace NUMINAMATH_CALUDE_solve_pocket_money_problem_l1496_149664

def pocket_money_problem (initial_money : ℕ) : Prop :=
  let remaining_money := initial_money / 2
  let total_money := remaining_money + 550
  total_money = 1000 ∧ initial_money = 900

theorem solve_pocket_money_problem :
  ∃ (initial_money : ℕ), pocket_money_problem initial_money :=
sorry

end NUMINAMATH_CALUDE_solve_pocket_money_problem_l1496_149664


namespace NUMINAMATH_CALUDE_mary_pizza_order_l1496_149688

def large_pizza_slices : ℕ := 8
def slices_eaten : ℕ := 7
def slices_remaining : ℕ := 9

theorem mary_pizza_order : 
  ∃ (pizzas_ordered : ℕ), 
    pizzas_ordered * large_pizza_slices = slices_eaten + slices_remaining ∧ 
    pizzas_ordered = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_pizza_order_l1496_149688


namespace NUMINAMATH_CALUDE_lower_right_is_three_l1496_149695

/-- Represents a 5x5 grid with digits from 1 to 5 -/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Checks if a number is unique in its row -/
def unique_in_row (g : Grid) (row col : Fin 5) : Prop :=
  ∀ c : Fin 5, c ≠ col → g row c ≠ g row col

/-- Checks if a number is unique in its column -/
def unique_in_col (g : Grid) (row col : Fin 5) : Prop :=
  ∀ r : Fin 5, r ≠ row → g r col ≠ g row col

/-- Checks if the grid satisfies the uniqueness conditions -/
def valid_grid (g : Grid) : Prop :=
  ∀ r c : Fin 5, unique_in_row g r c ∧ unique_in_col g r c

/-- The theorem to be proved -/
theorem lower_right_is_three (g : Grid) 
  (h1 : valid_grid g)
  (h2 : g 0 0 = 1)
  (h3 : g 0 4 = 2)
  (h4 : g 1 1 = 4)
  (h5 : g 2 3 = 3)
  (h6 : g 3 2 = 5) :
  g 4 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lower_right_is_three_l1496_149695


namespace NUMINAMATH_CALUDE_majorB_higher_admission_rate_male_higher_admission_rate_l1496_149673

/-- Represents the gender of applicants -/
inductive Gender
| Male
| Female

/-- Represents the major of applicants -/
inductive Major
| A
| B

/-- Data structure for application and admission information -/
structure MajorData where
  applicants : Gender → ℕ
  admissionRate : Gender → ℚ

/-- Calculate the weighted average admission rate for a major -/
def weightedAverageAdmissionRate (data : MajorData) : ℚ :=
  let totalApplicants := data.applicants Gender.Male + data.applicants Gender.Female
  let weightedSum := (data.applicants Gender.Male * data.admissionRate Gender.Male) +
                     (data.applicants Gender.Female * data.admissionRate Gender.Female)
  weightedSum / totalApplicants

/-- Calculate the overall admission rate for a gender across both majors -/
def overallAdmissionRate (majorA : MajorData) (majorB : MajorData) (gender : Gender) : ℚ :=
  let totalApplicants := majorA.applicants gender + majorB.applicants gender
  let admittedA := majorA.applicants gender * majorA.admissionRate gender
  let admittedB := majorB.applicants gender * majorB.admissionRate gender
  (admittedA + admittedB) / totalApplicants

/-- Theorem: The weighted average admission rate of Major B is higher than that of Major A -/
theorem majorB_higher_admission_rate (majorA : MajorData) (majorB : MajorData) :
  weightedAverageAdmissionRate majorB > weightedAverageAdmissionRate majorA := by
  sorry

/-- Theorem: The overall admission rate of males is higher than that of females -/
theorem male_higher_admission_rate (majorA : MajorData) (majorB : MajorData) :
  overallAdmissionRate majorA majorB Gender.Male > overallAdmissionRate majorA majorB Gender.Female := by
  sorry

end NUMINAMATH_CALUDE_majorB_higher_admission_rate_male_higher_admission_rate_l1496_149673


namespace NUMINAMATH_CALUDE_remaining_quantities_l1496_149631

theorem remaining_quantities (total : ℕ) (total_avg : ℚ) (subset : ℕ) (subset_avg : ℚ) (remaining_avg : ℚ) :
  total = 5 ∧ 
  total_avg = 10 ∧ 
  subset = 3 ∧ 
  subset_avg = 4 ∧ 
  remaining_avg = 19 →
  total - subset = 2 :=
by sorry

end NUMINAMATH_CALUDE_remaining_quantities_l1496_149631


namespace NUMINAMATH_CALUDE_michaels_currency_problem_l1496_149622

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Represents the problem of Michael's currency exchange and spending -/
theorem michaels_currency_problem :
  ∃ (d : ℕ),
    (5 / 8 : ℚ) * d - 75 = d ∧
    d = 200 ∧
    sum_of_digits d = 2 := by sorry

end NUMINAMATH_CALUDE_michaels_currency_problem_l1496_149622


namespace NUMINAMATH_CALUDE_class_size_l1496_149647

theorem class_size (boys girls : ℕ) : 
  boys = 3 * (boys / 3) ∧ 
  girls = 2 * (boys / 3) ∧ 
  boys = girls + 20 → 
  boys + girls = 100 := by
sorry

end NUMINAMATH_CALUDE_class_size_l1496_149647


namespace NUMINAMATH_CALUDE_chemistry_lab_workstations_l1496_149638

theorem chemistry_lab_workstations (total_capacity : ℕ) (total_workstations : ℕ) 
  (three_student_stations : ℕ) (remaining_stations : ℕ) 
  (h1 : total_capacity = 38)
  (h2 : total_workstations = 16)
  (h3 : three_student_stations = 6)
  (h4 : remaining_stations = 10)
  (h5 : total_workstations = three_student_stations + remaining_stations) :
  ∃ (students_per_remaining : ℕ),
    students_per_remaining * remaining_stations + 3 * three_student_stations = total_capacity ∧
    students_per_remaining * remaining_stations = 20 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_lab_workstations_l1496_149638


namespace NUMINAMATH_CALUDE_second_girl_speed_l1496_149627

/-- Given two girls walking in opposite directions, prove that the second girl's speed is 3 km/hr -/
theorem second_girl_speed (girl1_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  girl1_speed = 7 ∧ time = 12 ∧ distance = 120 →
  ∃ girl2_speed : ℝ, girl2_speed = 3 ∧ distance = (girl1_speed + girl2_speed) * time :=
by
  sorry

end NUMINAMATH_CALUDE_second_girl_speed_l1496_149627


namespace NUMINAMATH_CALUDE_monomial_difference_implies_m_pow_n_l1496_149697

/-- If the difference between 2ab^(2m+n) and a^(m-n)b^8 is a monomial, then m^n = 9 -/
theorem monomial_difference_implies_m_pow_n (a b : ℝ) (m n : ℕ) :
  (∃ k : ℝ, ∃ p q : ℕ, 2 * a * b^(2*m+n) - a^(m-n) * b^8 = k * a^p * b^q) →
  m^n = 9 := by
  sorry

end NUMINAMATH_CALUDE_monomial_difference_implies_m_pow_n_l1496_149697


namespace NUMINAMATH_CALUDE_sara_minus_lucas_sum_l1496_149692

def sara_list := List.range 50

def replace_three_with_two (n : ℕ) : ℕ :=
  let s := toString n
  (s.replace "3" "2").toNat!

def lucas_list := sara_list.map replace_three_with_two

theorem sara_minus_lucas_sum : 
  sara_list.sum - lucas_list.sum = 105 := by sorry

end NUMINAMATH_CALUDE_sara_minus_lucas_sum_l1496_149692


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l1496_149640

theorem range_of_2a_minus_b (a b : ℝ) (h1 : a > b) (h2 : 2*a^2 - a*b - b^2 - 4 = 0) :
  ∃ (k : ℝ), k ≥ 8/3 ∧ 2*a - b = k :=
sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l1496_149640


namespace NUMINAMATH_CALUDE_sci_fi_readers_l1496_149668

theorem sci_fi_readers (total : ℕ) (literary : ℕ) (both : ℕ) (sci_fi : ℕ) : 
  total = 400 → literary = 230 → both = 80 → 
  total = sci_fi + literary - both →
  sci_fi = 250 := by
sorry

end NUMINAMATH_CALUDE_sci_fi_readers_l1496_149668


namespace NUMINAMATH_CALUDE_cos_sum_diff_product_leq_cos_sq_l1496_149607

theorem cos_sum_diff_product_leq_cos_sq (x y : ℝ) :
  Real.cos (x + y) * Real.cos (x - y) ≤ Real.cos x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_diff_product_leq_cos_sq_l1496_149607


namespace NUMINAMATH_CALUDE_quadrilateral_angle_inequality_l1496_149674

variable (A B C D A₁ B₁ C₁ D₁ : Point)

-- Define the quadrilaterals
def is_convex_quadrilateral (P Q R S : Point) : Prop := sorry

-- Define the equality of corresponding sides
def equal_corresponding_sides (P Q R S P₁ Q₁ R₁ S₁ : Point) : Prop := sorry

-- Define the angle measure
def angle_measure (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_angle_inequality
  (h_convex_ABCD : is_convex_quadrilateral A B C D)
  (h_convex_A₁B₁C₁D₁ : is_convex_quadrilateral A₁ B₁ C₁ D₁)
  (h_equal_sides : equal_corresponding_sides A B C D A₁ B₁ C₁ D₁)
  (h_angle_A : angle_measure B A D > angle_measure B₁ A₁ D₁) :
  angle_measure A B C < angle_measure A₁ B₁ C₁ ∧
  angle_measure B C D > angle_measure B₁ C₁ D₁ ∧
  angle_measure C D A < angle_measure C₁ D₁ A₁ :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_inequality_l1496_149674


namespace NUMINAMATH_CALUDE_number_reciprocal_problem_l1496_149685

theorem number_reciprocal_problem (x y : ℝ) : 
  x > 0 → x = 3 → x + y = 60 * (1 / x) → y = 17 := by
sorry

end NUMINAMATH_CALUDE_number_reciprocal_problem_l1496_149685


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_8_l1496_149677

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * a 1 + (n * (n - 1) : ℤ) / 2 * (a 2 - a 1)

theorem arithmetic_sequence_sum_8 (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 1 = -40 →
  a 6 + a 10 = -10 →
  sum_of_arithmetic_sequence a 8 = -180 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_8_l1496_149677


namespace NUMINAMATH_CALUDE_intersection_points_with_constraints_l1496_149652

/-- The number of intersection points of n lines -/
def intersectionPoints (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of lines -/
def numLines : ℕ := 10

/-- The number of parallel line pairs -/
def numParallelPairs : ℕ := 1

/-- The number of lines intersecting at a single point -/
def numConcurrentLines : ℕ := 3

theorem intersection_points_with_constraints :
  intersectionPoints numLines - numParallelPairs - (numConcurrentLines.choose 2 - 1) = 42 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_with_constraints_l1496_149652


namespace NUMINAMATH_CALUDE_two_digit_primes_with_digit_sum_10_l1496_149653

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 10

theorem two_digit_primes_with_digit_sum_10 :
  ∃! (s : Finset ℕ), ∀ n, n ∈ s ↔ satisfies_condition n ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_two_digit_primes_with_digit_sum_10_l1496_149653


namespace NUMINAMATH_CALUDE_function_form_l1496_149637

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem function_form (f : ℕ → ℕ) 
  (h1 : StrictlyIncreasing f)
  (h2 : ∀ x y : ℕ, ∃ k : ℕ+, (f x + f y) / (1 + f (x + y)) = k) :
  ∃ a : ℕ+, ∀ x : ℕ, f x = a * x + 1 :=
sorry

end NUMINAMATH_CALUDE_function_form_l1496_149637


namespace NUMINAMATH_CALUDE_company_results_l1496_149600

structure Company where
  team_a_success_prob : ℚ
  team_b_success_prob : ℚ
  profit_a_success : ℤ
  loss_a_failure : ℤ
  profit_b_success : ℤ
  loss_b_failure : ℤ

def company : Company := {
  team_a_success_prob := 3/4,
  team_b_success_prob := 3/5,
  profit_a_success := 120,
  loss_a_failure := 50,
  profit_b_success := 100,
  loss_b_failure := 40
}

def exactly_one_success_prob (c : Company) : ℚ :=
  (1 - c.team_a_success_prob) * c.team_b_success_prob +
  c.team_a_success_prob * (1 - c.team_b_success_prob)

def profit_distribution (c : Company) : List (ℤ × ℚ) :=
  [(-90, (1 - c.team_a_success_prob) * (1 - c.team_b_success_prob)),
   (50, (1 - c.team_a_success_prob) * c.team_b_success_prob),
   (80, c.team_a_success_prob * (1 - c.team_b_success_prob)),
   (220, c.team_a_success_prob * c.team_b_success_prob)]

theorem company_results :
  exactly_one_success_prob company = 9/20 ∧
  profit_distribution company = [(-90, 1/10), (50, 3/20), (80, 3/10), (220, 9/20)] := by
  sorry

end NUMINAMATH_CALUDE_company_results_l1496_149600


namespace NUMINAMATH_CALUDE_original_number_proof_l1496_149628

theorem original_number_proof : ∃! n : ℤ, n * 74 = 19732 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1496_149628


namespace NUMINAMATH_CALUDE_school_vote_problem_l1496_149696

theorem school_vote_problem (U A B : Finset Nat) : 
  Finset.card U = 250 →
  Finset.card A = 175 →
  Finset.card B = 140 →
  Finset.card (U \ (A ∪ B)) = 45 →
  Finset.card (A ∩ B) = 110 := by
sorry

end NUMINAMATH_CALUDE_school_vote_problem_l1496_149696


namespace NUMINAMATH_CALUDE_distance_sum_bounds_l1496_149630

/-- Given points A, B, and D in a coordinate plane, prove that the sum of distances AD and BD is between 17 and 18 -/
theorem distance_sum_bounds (A B D : ℝ × ℝ) : 
  A = (15, 0) → B = (0, 0) → D = (3, 4) → 
  17 < Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) ∧
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) < 18 :=
by sorry

end NUMINAMATH_CALUDE_distance_sum_bounds_l1496_149630


namespace NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l1496_149644

/-- The volume of a cone whose lateral surface unfolds into a semicircle with radius 2 -/
theorem cone_volume_from_lateral_surface (r : Real) (h : Real) : 
  (r = 1) → (h = Real.sqrt 3) → (2 * π * r = 2 * π) → 
  (1 / 3 : Real) * π * r^2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l1496_149644


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l1496_149661

theorem factorization_of_difference_of_squares (a b : ℝ) :
  3 * a^2 - 3 * b^2 = 3 * (a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l1496_149661


namespace NUMINAMATH_CALUDE_xyz_value_l1496_149689

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 27)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : 
  x * y * z = 6 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1496_149689


namespace NUMINAMATH_CALUDE_xiao_ming_score_l1496_149642

/-- Calculates the weighted score for a given component -/
def weightedScore (score : ℝ) (weight : ℝ) : ℝ := score * weight

/-- Calculates the total score based on individual scores and weights -/
def totalScore (regularScore midtermScore finalScore : ℝ) 
               (regularWeight midtermWeight finalWeight : ℝ) : ℝ :=
  weightedScore regularScore regularWeight + 
  weightedScore midtermScore midtermWeight + 
  weightedScore finalScore finalWeight

theorem xiao_ming_score : 
  let regularScore : ℝ := 70
  let midtermScore : ℝ := 80
  let finalScore : ℝ := 85
  let totalWeight : ℝ := 3 + 3 + 4
  let regularWeight : ℝ := 3 / totalWeight
  let midtermWeight : ℝ := 3 / totalWeight
  let finalWeight : ℝ := 4 / totalWeight
  totalScore regularScore midtermScore finalScore 
             regularWeight midtermWeight finalWeight = 79 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_score_l1496_149642


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1496_149632

theorem min_value_of_expression (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 ∧ ∃ y > 1, y + 1 / (y - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1496_149632


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1496_149649

theorem quadratic_inequality (y : ℝ) : y^2 - 9*y + 14 ≤ 0 ↔ 2 ≤ y ∧ y ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1496_149649


namespace NUMINAMATH_CALUDE_zoo_supplies_total_l1496_149615

/-- Represents the number of treats and bread pieces brought by Jane and Wanda to the zoo. -/
structure ZooSupplies where
  jane_treats : ℕ
  wanda_bread : ℕ := 90

/-- Calculates the total number of bread pieces and treats brought to the zoo. -/
def total_supplies (s : ZooSupplies) : ℕ :=
  let jane_bread := (s.jane_treats * 3) / 4
  let wanda_treats := s.jane_treats / 2
  s.jane_treats + jane_bread + s.wanda_bread + wanda_treats

/-- Theorem stating that the total number of supplies brought to the zoo is 225. -/
theorem zoo_supplies_total (s : ZooSupplies) :
  s.wanda_bread = 90 →
  s.wanda_bread = 3 * (s.jane_treats / 2) →
  total_supplies s = 225 := by
  sorry

end NUMINAMATH_CALUDE_zoo_supplies_total_l1496_149615


namespace NUMINAMATH_CALUDE_total_visits_equals_39_l1496_149654

/-- Calculates the total number of doctor visits in a year -/
def total_visits (visits_per_month_doc1 : ℕ) 
                 (months_between_visits_doc2 : ℕ) 
                 (visits_per_period_doc3 : ℕ) 
                 (months_per_period_doc3 : ℕ) : ℕ :=
  let months_in_year := 12
  let visits_doc1 := visits_per_month_doc1 * months_in_year
  let visits_doc2 := months_in_year / months_between_visits_doc2
  let periods_in_year := months_in_year / months_per_period_doc3
  let visits_doc3 := visits_per_period_doc3 * periods_in_year
  visits_doc1 + visits_doc2 + visits_doc3

/-- Theorem stating that the total visits in a year is 39 -/
theorem total_visits_equals_39 : 
  total_visits 2 2 3 4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_total_visits_equals_39_l1496_149654


namespace NUMINAMATH_CALUDE_intersection_max_value_l1496_149643

def P (a b : ℝ) (x : ℝ) : ℝ := x^6 - 8*x^5 + 24*x^4 - 37*x^3 + a*x^2 + b*x - 6

def L (d : ℝ) (x : ℝ) : ℝ := d*x + 2

theorem intersection_max_value (a b d : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ 
    P a b x = L d x ∧
    P a b y = L d y ∧
    P a b z = L d z ∧
    (∀ t : ℝ, t ≠ z → (P a b t - L d t) / (t - z) ≠ 0)) →
  (∃ w : ℝ, P a b w = L d w ∧ ∀ v : ℝ, P a b v = L d v → v ≤ w ∧ w = 5) :=
by sorry

end NUMINAMATH_CALUDE_intersection_max_value_l1496_149643


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_1728_l1496_149602

theorem closest_integer_to_cube_root_1728 : 
  ∃ n : ℤ, ∀ m : ℤ, |n - (1728 : ℝ)^(1/3)| ≤ |m - (1728 : ℝ)^(1/3)| ∧ n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_1728_l1496_149602


namespace NUMINAMATH_CALUDE_jimmy_card_distribution_l1496_149606

theorem jimmy_card_distribution (initial_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 18)
  (h2 : remaining_cards = 9) :
  ∃ (cards_to_bob : ℕ), 
    cards_to_bob = 3 ∧ 
    initial_cards = remaining_cards + cards_to_bob + 2 * cards_to_bob :=
by
  sorry

end NUMINAMATH_CALUDE_jimmy_card_distribution_l1496_149606


namespace NUMINAMATH_CALUDE_x_value_when_y_is_5_l1496_149687

-- Define the constant ratio
def k : ℚ := (5 * 3 - 6) / (2 * 2 + 10)

-- Define the relationship between x and y
def relation (x y : ℚ) : Prop := (5 * x - 6) / (2 * y + 10) = k

-- State the theorem
theorem x_value_when_y_is_5 :
  ∀ x : ℚ, relation x 2 → relation 3 2 → relation x 5 → x = 53 / 14 :=
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_5_l1496_149687


namespace NUMINAMATH_CALUDE_series_evaluation_l1496_149611

open Real

noncomputable def series_sum : ℝ := ∑' k, (k : ℝ)^2 / 3^k

theorem series_evaluation : series_sum = 7 := by sorry

end NUMINAMATH_CALUDE_series_evaluation_l1496_149611


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l1496_149605

theorem root_equation_implies_expression_value (m : ℝ) : 
  2 * m^2 + 3 * m - 1 = 0 → 4 * m^2 + 6 * m - 2019 = -2017 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l1496_149605


namespace NUMINAMATH_CALUDE_ryan_fundraising_goal_l1496_149620

/-- The total amount Ryan wants to raise for his business -/
def total_amount (avg_funding : ℕ) (num_people : ℕ) (existing_funds : ℕ) : ℕ :=
  avg_funding * num_people + existing_funds

/-- Proof that Ryan wants to raise $1000 for his business -/
theorem ryan_fundraising_goal :
  let avg_funding : ℕ := 10
  let num_people : ℕ := 80
  let existing_funds : ℕ := 200
  total_amount avg_funding num_people existing_funds = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ryan_fundraising_goal_l1496_149620


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l1496_149613

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |a * x + 1|

-- Theorem 1: Solution set for f(x) ≥ 3 when a = 1
theorem solution_set_f_geq_3 :
  {x : ℝ | f 1 x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem 2: Range of a when solution set for f(x) ≤ 3-x contains [-1, 1]
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 3 - x) → a ∈ Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l1496_149613


namespace NUMINAMATH_CALUDE_triangle_configuration_l1496_149694

/-- Given a configuration of similar right-angled triangles, prove the values of v, w, x, y, and z -/
theorem triangle_configuration (v w x y z : ℝ) : 
  v / 8 = 9 / x ∧ 
  9 / x = y / 20 ∧ 
  y^2 = x^2 + 9^2 ∧ 
  z^2 = 20^2 - x^2 ∧ 
  w^2 = 8^2 + v^2 →
  v = 6 ∧ w = 10 ∧ x = 12 ∧ y = 15 ∧ z = 16 :=
by sorry

end NUMINAMATH_CALUDE_triangle_configuration_l1496_149694


namespace NUMINAMATH_CALUDE_three_planes_six_parts_l1496_149678

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane (this is a simplified representation)
  dummy : Unit

/-- The number of parts that a set of planes divides the space into -/
def num_parts (planes : List Plane3D) : Nat :=
  sorry

/-- Defines if three planes are collinear -/
def are_collinear (p1 p2 p3 : Plane3D) : Prop :=
  sorry

/-- Defines if two planes are parallel -/
def are_parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Defines if two planes intersect -/
def intersect (p1 p2 : Plane3D) : Prop :=
  sorry

theorem three_planes_six_parts 
  (p1 p2 p3 : Plane3D) 
  (h : num_parts [p1, p2, p3] = 6) :
  (are_collinear p1 p2 p3) ∨ 
  ((are_parallel p1 p2 ∧ intersect p1 p3 ∧ intersect p2 p3) ∨
   (are_parallel p1 p3 ∧ intersect p1 p2 ∧ intersect p3 p2) ∨
   (are_parallel p2 p3 ∧ intersect p2 p1 ∧ intersect p3 p1)) :=
by
  sorry

end NUMINAMATH_CALUDE_three_planes_six_parts_l1496_149678


namespace NUMINAMATH_CALUDE_trace_equality_for_cubed_matrices_l1496_149681

open Matrix

theorem trace_equality_for_cubed_matrices
  (A B : Matrix (Fin 2) (Fin 2) ℝ)
  (h_not_commute : A * B ≠ B * A)
  (h_cubed_equal : A^3 = B^3) :
  ∀ n : ℕ, Matrix.trace (A^n) = Matrix.trace (B^n) :=
by sorry

end NUMINAMATH_CALUDE_trace_equality_for_cubed_matrices_l1496_149681


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l1496_149683

theorem triangle_abc_proof (c : ℝ) (A C : ℝ) 
  (h_c : c = 10)
  (h_A : A = 45 * π / 180)
  (h_C : C = 30 * π / 180) :
  ∃ (a b B : ℝ),
    a = 10 * Real.sqrt 2 ∧
    b = 5 * (Real.sqrt 2 + Real.sqrt 6) ∧
    B = 105 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l1496_149683


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_6_l1496_149699

def is_divisible_by_2 (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_6 (n : ℕ) : Prop := is_divisible_by_2 n ∧ is_divisible_by_3 n

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999

theorem largest_four_digit_divisible_by_6 :
  ∀ n : ℕ, is_four_digit n → is_divisible_by_6 n → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_6_l1496_149699


namespace NUMINAMATH_CALUDE_businessmen_beverages_l1496_149626

theorem businessmen_beverages (total : ℕ) (coffee tea juice : ℕ) 
  (coffee_tea coffee_juice tea_juice : ℕ) (all_three : ℕ) : 
  total = 30 → 
  coffee = 15 → 
  tea = 12 → 
  juice = 8 → 
  coffee_tea = 6 → 
  coffee_juice = 4 → 
  tea_juice = 2 → 
  all_three = 1 → 
  total - (coffee + tea + juice - coffee_tea - coffee_juice - tea_juice + all_three) = 6 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_beverages_l1496_149626


namespace NUMINAMATH_CALUDE_rectangles_equal_perimeter_different_shape_l1496_149648

/-- Two rectangles with equal perimeters can have different shapes -/
theorem rectangles_equal_perimeter_different_shape :
  ∃ (l₁ w₁ l₂ w₂ : ℝ), 
    l₁ > 0 ∧ w₁ > 0 ∧ l₂ > 0 ∧ w₂ > 0 ∧
    2 * (l₁ + w₁) = 2 * (l₂ + w₂) ∧
    l₁ / w₁ ≠ l₂ / w₂ :=
by sorry

end NUMINAMATH_CALUDE_rectangles_equal_perimeter_different_shape_l1496_149648


namespace NUMINAMATH_CALUDE_factorization_equality_l1496_149603

theorem factorization_equality (m x : ℝ) : m * x^2 - 4 * m = m * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1496_149603


namespace NUMINAMATH_CALUDE_square_of_difference_l1496_149645

theorem square_of_difference (x : ℝ) : (x - 3)^2 = x^2 - 6*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l1496_149645


namespace NUMINAMATH_CALUDE_xiaoming_home_most_precise_l1496_149655

-- Define the possible descriptions of location
inductive LocationDescription
  | RightSide
  | Distance (d : ℝ)
  | WestSide
  | WestSideWithDistance (d : ℝ)

-- Define a function to check if a description is complete (has both direction and distance)
def isCompleteDescription (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.WestSideWithDistance _ => True
  | _ => False

-- Define Xiao Ming's home location
def xiaomingHome : LocationDescription := LocationDescription.WestSideWithDistance 900

-- Theorem: Xiao Ming's home location is the most precise description
theorem xiaoming_home_most_precise :
  isCompleteDescription xiaomingHome ∧
  ∀ (desc : LocationDescription), isCompleteDescription desc → desc = xiaomingHome :=
sorry

end NUMINAMATH_CALUDE_xiaoming_home_most_precise_l1496_149655


namespace NUMINAMATH_CALUDE_johns_house_paintable_area_l1496_149691

/-- Calculates the total paintable wall area in John's house -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (non_paintable_area : ℝ) : ℝ :=
  num_bedrooms * (2 * (length * height + width * height) - non_paintable_area)

/-- Proves that the total paintable wall area in John's house is 1820 square feet -/
theorem johns_house_paintable_area :
  total_paintable_area 4 15 12 10 85 = 1820 := by
  sorry

#eval total_paintable_area 4 15 12 10 85

end NUMINAMATH_CALUDE_johns_house_paintable_area_l1496_149691


namespace NUMINAMATH_CALUDE_rice_cost_l1496_149629

/-- Proves that the cost of each kilogram of rice is $2 given the conditions of Vicente's purchase --/
theorem rice_cost (rice_kg : ℕ) (meat_lb : ℕ) (meat_cost_per_lb : ℕ) (total_spent : ℕ) : 
  rice_kg = 5 → meat_lb = 3 → meat_cost_per_lb = 5 → total_spent = 25 →
  ∃ (rice_cost_per_kg : ℕ), rice_cost_per_kg = 2 ∧ rice_kg * rice_cost_per_kg + meat_lb * meat_cost_per_lb = total_spent :=
by sorry

end NUMINAMATH_CALUDE_rice_cost_l1496_149629


namespace NUMINAMATH_CALUDE_one_way_fare_calculation_l1496_149676

/-- The one-way fare from home to office -/
def one_way_fare : ℚ := 16

/-- The total amount spent on travel for 9 working days -/
def total_spent : ℚ := 288

/-- The number of working days -/
def working_days : ℕ := 9

/-- The number of trips per day -/
def trips_per_day : ℕ := 2

theorem one_way_fare_calculation :
  one_way_fare * (working_days * trips_per_day : ℚ) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_one_way_fare_calculation_l1496_149676


namespace NUMINAMATH_CALUDE_angle_B_is_30_degrees_l1496_149610

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.c * Real.cos t.B + t.b * Real.cos t.C = t.a * Real.sin t.A ∧
  (Real.sqrt 3 / 4) * (t.b^2 + t.a^2 - t.c^2) = (1/2) * t.a * t.b * Real.sin t.C

-- Theorem statement
theorem angle_B_is_30_degrees (t : Triangle) 
  (h : satisfies_conditions t) : t.B = 30 * (Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_30_degrees_l1496_149610


namespace NUMINAMATH_CALUDE_john_change_theorem_l1496_149604

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- Calculates the total payment in cents given the number of each coin type -/
def total_payment (quarters dimes nickels : ℕ) : ℕ :=
  quarters * coin_value "quarter" + dimes * coin_value "dime" + nickels * coin_value "nickel"

/-- Calculates the change received given the total payment and the cost of the item -/
def change_received (payment cost : ℕ) : ℕ :=
  payment - cost

theorem john_change_theorem (candy_cost : ℕ) (h1 : candy_cost = 131) :
  change_received (total_payment 4 3 1) candy_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_john_change_theorem_l1496_149604


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1496_149663

/-- Given a line passing through points (-3, 8) and (0, -1), prove that the sum of its slope and y-intercept is -4 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (∀ x y : ℝ, (x = -3 ∧ y = 8) ∨ (x = 0 ∧ y = -1) → y = m * x + b) → 
  m + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1496_149663


namespace NUMINAMATH_CALUDE_herd_size_l1496_149621

theorem herd_size (bulls : ℕ) (h : bulls = 70) : 
  (2 / 3 : ℚ) * (1 / 3 : ℚ) * (total_herd : ℚ) = bulls → total_herd = 315 := by
  sorry

end NUMINAMATH_CALUDE_herd_size_l1496_149621


namespace NUMINAMATH_CALUDE_regression_lines_intersection_l1496_149658

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a regression line -/
def lies_on (l : RegressionLine) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : ∀ (x y : ℝ), x = s ∧ y = t → lies_on l₁ x y)
  (h₂ : ∀ (x y : ℝ), x = s ∧ y = t → lies_on l₂ x y) :
  lies_on l₁ s t ∧ lies_on l₂ s t := by
  sorry


end NUMINAMATH_CALUDE_regression_lines_intersection_l1496_149658


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l1496_149617

theorem min_value_of_expression (x : ℝ) : 
  (14 - x) * (9 - x) * (14 + x) * (9 + x) ≥ -1156.25 :=
by sorry

theorem min_value_attained : 
  ∃ x : ℝ, (14 - x) * (9 - x) * (14 + x) * (9 + x) = -1156.25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l1496_149617


namespace NUMINAMATH_CALUDE_lattice_point_theorem_four_point_counterexample_l1496_149667

-- Define a lattice point as a pair of integers
def LatticePoint := ℤ × ℤ

-- Define the set of all lattice points
def S : Set LatticePoint := Set.univ

-- Define a function to check if a point is between two other points
def isBetween (p q r : LatticePoint) : Prop :=
  ∃ t : ℚ, 0 < t ∧ t < 1 ∧ 
    (t * p.1 + (1 - t) * q.1 = r.1) ∧
    (t * p.2 + (1 - t) * q.2 = r.2)

-- State the theorem
theorem lattice_point_theorem (A B C : LatticePoint) 
  (hA : A ∈ S) (hB : B ∈ S) (hC : C ∈ S) 
  (hAB : A ≠ B) (hBC : B ≠ C) (hAC : A ≠ C) :
  ∃ D : LatticePoint, D ∈ S ∧ D ≠ A ∧ D ≠ B ∧ D ≠ C ∧
    (∀ P : LatticePoint, P ∈ S → 
      ¬(isBetween A D P) ∧ ¬(isBetween B D P) ∧ ¬(isBetween C D P)) :=
sorry

-- Counter-example for 4 points
theorem four_point_counterexample :
  ∃ A B C D : LatticePoint, 
    A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    ∀ E : LatticePoint, E ∈ S →
      (isBetween A B E) ∨ (isBetween A C E) ∨ (isBetween A D E) ∨
      (isBetween B C E) ∨ (isBetween B D E) ∨ (isBetween C D E) :=
sorry

end NUMINAMATH_CALUDE_lattice_point_theorem_four_point_counterexample_l1496_149667


namespace NUMINAMATH_CALUDE_hannahs_dogs_food_l1496_149646

/-- The amount of food eaten by Hannah's first dog -/
def first_dog_food : ℝ := 1.5

/-- The amount of food eaten by Hannah's second dog -/
def second_dog_food : ℝ := 2 * first_dog_food

/-- The amount of food eaten by Hannah's third dog -/
def third_dog_food : ℝ := second_dog_food + 2.5

/-- The total amount of food prepared by Hannah for her three dogs -/
def total_food : ℝ := 10

theorem hannahs_dogs_food :
  first_dog_food + second_dog_food + third_dog_food = total_food :=
by sorry

end NUMINAMATH_CALUDE_hannahs_dogs_food_l1496_149646


namespace NUMINAMATH_CALUDE_num_lions_seen_l1496_149686

/-- The number of legs Borgnine wants to see at the zoo -/
def total_legs : ℕ := 1100

/-- The number of chimps Borgnine has seen -/
def num_chimps : ℕ := 12

/-- The number of lizards Borgnine has seen -/
def num_lizards : ℕ := 5

/-- The number of tarantulas Borgnine will see -/
def num_tarantulas : ℕ := 125

/-- The number of legs a chimp has -/
def chimp_legs : ℕ := 4

/-- The number of legs a lion has -/
def lion_legs : ℕ := 4

/-- The number of legs a lizard has -/
def lizard_legs : ℕ := 4

/-- The number of legs a tarantula has -/
def tarantula_legs : ℕ := 8

theorem num_lions_seen : ℕ := by
  sorry

end NUMINAMATH_CALUDE_num_lions_seen_l1496_149686


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1496_149639

/-- Three concentric circles with radii 1, 2, and 3 units -/
def circles := {r : ℝ | r = 1 ∨ r = 2 ∨ r = 3}

/-- A point on one of the circles -/
structure CirclePoint where
  x : ℝ
  y : ℝ
  r : ℝ
  on_circle : x^2 + y^2 = r^2
  radius_valid : r ∈ circles

/-- An equilateral triangle formed by points on the circles -/
structure EquilateralTriangle where
  A : CirclePoint
  B : CirclePoint
  C : CirclePoint
  equilateral : (A.x - B.x)^2 + (A.y - B.y)^2 = (B.x - C.x)^2 + (B.y - C.y)^2 ∧
                (B.x - C.x)^2 + (B.y - C.y)^2 = (C.x - A.x)^2 + (C.y - A.y)^2
  on_different_circles : A.r ≠ B.r ∧ B.r ≠ C.r ∧ C.r ≠ A.r

/-- The theorem stating that the side length of the equilateral triangle is √7 -/
theorem equilateral_triangle_side_length 
  (triangle : EquilateralTriangle) : 
  (triangle.A.x - triangle.B.x)^2 + (triangle.A.y - triangle.B.y)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1496_149639


namespace NUMINAMATH_CALUDE_sin_cos_power_sum_l1496_149665

theorem sin_cos_power_sum (x : ℝ) (h : 3 * Real.sin x ^ 3 + Real.cos x ^ 3 = 3) :
  Real.sin x ^ 2018 + Real.cos x ^ 2018 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_power_sum_l1496_149665


namespace NUMINAMATH_CALUDE_zeros_order_l1496_149623

noncomputable def f (x : ℝ) := x + 2^x
noncomputable def g (x : ℝ) := x + Real.log x
def h (x : ℝ) := x^3 + x - 2

theorem zeros_order (x₁ x₂ x₃ : ℝ) 
  (hf : f x₁ = 0)
  (hg : g x₂ = 0)
  (hh : h x₃ = 0) :
  x₁ < x₂ ∧ x₂ < x₃ :=
sorry

end NUMINAMATH_CALUDE_zeros_order_l1496_149623


namespace NUMINAMATH_CALUDE_binary_addition_theorem_l1496_149624

/-- Converts a binary number (represented as a list of bits) to decimal -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to binary (represented as a list of bits) -/
def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec to_binary_aux (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
    to_binary_aux n

/-- The main theorem: 1010₂ + 10₂ = 1100₂ -/
theorem binary_addition_theorem : 
  decimal_to_binary (binary_to_decimal [false, true, false, true] + 
                     binary_to_decimal [false, true]) =
  [false, false, true, true] := by sorry

end NUMINAMATH_CALUDE_binary_addition_theorem_l1496_149624


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1496_149634

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ)) ∧
  (x : ℕ) + (y : ℕ) = 64 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1496_149634


namespace NUMINAMATH_CALUDE_f_is_even_l1496_149609

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1496_149609


namespace NUMINAMATH_CALUDE_divisibility_condition_l1496_149670

theorem divisibility_condition (m n : ℕ+) :
  (∃ k : ℤ, 4 * (m.val * n.val + 1) = k * (m.val + n.val)^2) ↔ m = n :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1496_149670
