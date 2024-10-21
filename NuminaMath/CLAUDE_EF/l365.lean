import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_distribution_convergence_l365_36599

def f : ℕ → ℤ → ℕ
  | 0, 0 => 5^2003
  | 0, n => if n ≠ 0 then 0 else 5^2003
  | m+1, n => f m n - 2 * (f m n / 2) + (f m (n-1) / 2) + (f m (n+1) / 2)

theorem coin_distribution_convergence :
  ∃ M : ℕ+, ∀ n : ℤ,
    (abs n ≤ (5^2003 - 1) / 2 → f M.val n = 1) ∧
    (abs n > (5^2003 - 1) / 2 → f M.val n = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_distribution_convergence_l365_36599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a_2015_eq_6_l365_36571

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x * (1 - x) else x * (1 + x)

noncomputable def a : ℕ → ℝ
  | 0 => 1/2
  | n + 1 => 1 / (1 - a n)

theorem f_a_2015_eq_6 (h_odd : ∀ x, f (-x) = -f x) : f (a 2015) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a_2015_eq_6_l365_36571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l365_36586

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt ((Real.log x / Real.log 2)^2 - 1)

-- Define the domain of f
def domain_f : Set ℝ := Set.Ioo 0 (1/2) ∪ Set.Ioi 2

-- State the theorem
theorem domain_of_f : 
  {x : ℝ | x > 0 ∧ (Real.log x / Real.log 2)^2 - 1 > 0} = domain_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l365_36586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_modulus_sqrt2_l365_36531

def A : Set ℂ := {z | ∃ n : ℕ+, z = (Finset.range (n + 1)).sum (fun k => (Complex.I : ℂ) ^ k)}

def B : Set ℂ := {ω | ∃ z₁ z₂, z₁ ∈ A ∧ z₂ ∈ A ∧ ω = z₁ * z₂}

theorem probability_modulus_sqrt2 : 
  let B_finset : Finset ℂ := {1, 1 + Complex.I, Complex.I, 2 * Complex.I, -1 + Complex.I, -1, 0}
  (B_finset.filter (fun z => Complex.abs z = Real.sqrt 2)).card / B_finset.card = 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_modulus_sqrt2_l365_36531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_range_of_x_l365_36554

variable (a b x : ℝ)

-- Part I
theorem min_value_expression :
  (∀ a b : ℝ, a + b = 1 → a > 0 → b > 0 → 1/a + 4/b ≥ 9) ∧
  (∃ a b : ℝ, a + b = 1 ∧ a > 0 ∧ b > 0 ∧ 1/a + 4/b = 9) :=
sorry

-- Part II
theorem range_of_x :
  (∀ a b x : ℝ, a + b = 1 → a > 0 → b > 0 → 1/a + 4/b ≥ |2*x - 1| - |x + 1|) ↔
  ∀ x : ℝ, -7 ≤ x ∧ x ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_range_of_x_l365_36554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_divisibility_l365_36579

/-- A type representing points on a circle --/
structure CirclePoint where
  position : ℚ
  length : ℚ
  h : 0 ≤ position ∧ position < length

/-- The distance between two points on a circle --/
noncomputable def circleDistance (p q : CirclePoint) : ℚ :=
  min (abs (p.position - q.position)) (p.length - abs (p.position - q.position))

/-- A predicate stating that a set of points satisfies the distance conditions --/
def satisfiesDistanceConditions (points : Set CirclePoint) : Prop :=
  ∀ p ∈ points,
    (∃! q, q ∈ points ∧ circleDistance p q = 1) ∧
    (∃! q, q ∈ points ∧ circleDistance p q = 2)

theorem circle_points_divisibility
  (points : Finset CirclePoint)
  (h_length : ∀ p ∈ points, p.length = 15)
  (h_conditions : satisfiesDistanceConditions (↑points : Set CirclePoint)) :
  points.card % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_divisibility_l365_36579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_properties_l365_36569

/-- Given a quadratic equation x² + ax + 2a = 0 with two distinct real roots,
    prove properties about the roots and a function of a. -/
theorem quadratic_roots_properties (a : ℝ) (x₁ x₂ : ℝ) 
    (h₁ : x₁^2 + a*x₁ + 2*a = 0)
    (h₂ : x₂^2 + a*x₂ + 2*a = 0)
    (h₃ : x₁ ≠ x₂) :
  (∃ min_value : ℝ, min_value = 34 + 2 * Real.sqrt 2 ∧
    ∀ (a : ℝ), |a| > 17 → |x₁ * x₂| + 1 / (|a| - 17) ≥ min_value) ∧
  (1 / x₁ + 1 / x₂ = -1 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_properties_l365_36569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_is_four_l365_36513

/-- The circle with center (-1, 1) and radius 2 -/
def Circle (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 1)^2 = 4

/-- The point P -/
def P : ℝ × ℝ := (2, 5)

/-- The distance between two points -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

/-- The theorem stating that the difference between max and min distances is 4 -/
theorem distance_difference_is_four :
  ∃ (M N : ℝ),
    (∀ (A : ℝ × ℝ), Circle A.1 A.2 → distance P A ≤ M) ∧
    (∀ (A : ℝ × ℝ), Circle A.1 A.2 → distance P A ≥ N) ∧
    (∃ (A₁ A₂ : ℝ × ℝ), Circle A₁.1 A₁.2 ∧ Circle A₂.1 A₂.2 ∧ 
      distance P A₁ = M ∧ distance P A₂ = N) ∧
    M - N = 4 := by
  sorry

#check distance_difference_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_is_four_l365_36513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_10_l365_36592

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x^2 + 1 else 2*x

-- State the theorem
theorem f_equals_10 (a : ℝ) :
  f a = 10 ↔ a = -3 ∨ a = 5 := by
  sorry

#check f_equals_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_10_l365_36592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l365_36562

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y - 3 = 0

-- Define the area of the region
noncomputable def region_area : ℝ := 28 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof will be added here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l365_36562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pink_crayons_is_fourteen_l365_36525

/-- The total number of pink crayons Mara and Luna have -/
def total_pink_crayons : ℕ :=
  let mara_crayons : ℕ := 40
  let mara_pink_percent : ℚ := 10 / 100
  let luna_crayons : ℕ := 50
  let luna_pink_percent : ℚ := 20 / 100
  let mara_pink : ℕ := (mara_crayons : ℚ) * mara_pink_percent |>.floor.toNat
  let luna_pink : ℕ := (luna_crayons : ℚ) * luna_pink_percent |>.floor.toNat
  mara_pink + luna_pink

#eval total_pink_crayons

theorem total_pink_crayons_is_fourteen : total_pink_crayons = 14 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pink_crayons_is_fourteen_l365_36525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_at_2_iff_b_eq_5_l365_36506

/-- A piecewise function f(x) defined as:
    f(x) = 5x^2 - 3 for x ≤ 2
    f(x) = bx + 7 for x > 2 --/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then 5 * x^2 - 3 else b * x + 7

/-- The function f is continuous at x = 2 --/
def is_continuous_at_2 (b : ℝ) : Prop :=
  ContinuousAt (f b) 2

/-- Theorem: For the function f to be continuous at x = 2, b must equal 5 --/
theorem continuous_at_2_iff_b_eq_5 :
  ∀ b : ℝ, is_continuous_at_2 b ↔ b = 5 := by
  sorry

#check continuous_at_2_iff_b_eq_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_at_2_iff_b_eq_5_l365_36506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_red_3x3_is_340_341_l365_36585

/-- Represents a 4-by-4 grid where each cell can be colored red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- The probability of a cell being red -/
noncomputable def p_red : ℝ := 1 / 2

/-- Checks if a 3x3 subgrid starting at (i, j) is all red -/
def is_red_3x3 (g : Grid) (i j : Fin 2) : Prop :=
  ∀ (x y : Fin 3), g (i + x) (j + y) = true

/-- The probability of obtaining a grid without a 3-by-3 red square -/
noncomputable def p_no_red_3x3 : ℝ :=
  1 - 4 * (1 - p_red^9) * p_red^7 + 2 * (1 - p_red^12) * p_red^4

theorem prob_no_red_3x3_is_340_341 : p_no_red_3x3 = 340 / 341 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_red_3x3_is_340_341_l365_36585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_price_ratio_l365_36589

/-- The ratio of unit prices given a volume and price difference -/
theorem unit_price_ratio (v : ℝ) (p : ℝ) (hv : v > 0) (hp : p > 0) :
  (0.85 * p) / (1.25 * v) / (p / v) = 17 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_price_ratio_l365_36589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l365_36565

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then -x + 5 else 2 + (Real.log x) / (Real.log a)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∀ x, f a x ≥ 3) →
  (∀ y ≥ 3, ∃ x, f a x = y) →
  1 < a ∧ a ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l365_36565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_set_l365_36518

noncomputable def integer_part (x : ℝ) : ℤ := Int.floor x

noncomputable def fractional_part (x : ℝ) : ℝ := x - (Int.floor x : ℝ)

theorem equation_solution_set (x : ℝ) :
  (integer_part x)^5 + (fractional_part x)^5 = x^5 ↔ x ∈ (Set.Icc 0 1) ∪ (Set.range (Int.cast : ℤ → ℝ)) := by
  sorry

#check equation_solution_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_set_l365_36518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l365_36539

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the circle equation
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define a point C on the circle
def C : {p : ℝ × ℝ // circleEq p.1 p.2} := sorry

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

-- Theorem statement
theorem max_triangle_area :
  (⨆ (c : {p : ℝ × ℝ // circleEq p.1 p.2}), triangleArea A B c.val) = 3 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l365_36539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equation_solution_l365_36521

/-- The kernel function K(x) = e^(-|x|) -/
noncomputable def K (x : ℝ) : ℝ := Real.exp (-abs x)

/-- The integral equation -/
def integral_equation (φ f : ℝ → ℝ) (l : ℝ) : Prop :=
  ∀ x, φ x = f x + l * ∫ (t : ℝ), K (x - t) * φ t

/-- The proposed solution function -/
noncomputable def φ_solution (l : ℝ) (x : ℝ) : ℝ :=
  Real.exp (-Real.sqrt (1 - 2*l) * abs x) / Real.sqrt (1 - 2*l)

/-- The theorem statement -/
theorem integral_equation_solution (l : ℝ) (h : l < 1/2) :
  ∃ f, integral_equation (φ_solution l) f l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equation_solution_l365_36521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l365_36551

noncomputable def data : List ℝ := [4, 6, 5, 8, 7, 6]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

theorem variance_of_data : variance data = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l365_36551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_assignment_count_l365_36578

theorem factory_assignment_count : ℕ := by
  -- Define the number of men and women
  let num_men : ℕ := 5
  let num_women : ℕ := 4
  
  -- Define the number of representatives to be selected
  let num_representatives : ℕ := 4
  
  -- Define the number of factories
  let num_factories : ℕ := 4
  
  -- Define the constraints on the selection
  let min_men : ℕ := 2
  let min_women : ℕ := 1
  
  -- Calculate the number of ways to select representatives
  let ways_to_select : ℕ := 
    (Nat.choose num_men 3 * Nat.choose num_women 1) + 
    (Nat.choose num_men 2 * Nat.choose num_women 2)
  
  -- Calculate the number of ways to assign representatives to factories
  let total_assignments : ℕ := ways_to_select * Nat.factorial num_factories
  
  -- The theorem states that the number of different assignment methods is 2400
  have h : total_assignments = 2400 := by
    -- Proof goes here
    sorry
  
  exact 2400


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_assignment_count_l365_36578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l365_36563

/-- The curve C -/
def C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8

/-- The line l -/
def l (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 2 * Real.sqrt 3 - 3 = 0

/-- Point P -/
def P : ℝ × ℝ := (-2, -3)

/-- Intersection points of C and l -/
def intersectionPoints : Set (ℝ × ℝ) :=
  {p | C p.1 p.2 ∧ l p.1 p.2}

theorem intersection_product :
  ∃ A B : ℝ × ℝ, A ∈ intersectionPoints ∧ B ∈ intersectionPoints ∧
    (A.1 - P.1)^2 + (A.2 - P.2)^2 * ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 33^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l365_36563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_sqrt_five_is_21st_term_l365_36535

/-- The sequence term for a given index -/
noncomputable def a (n : ℕ) : ℝ := Real.sqrt (6 * n - 1)

/-- The statement that 5√5 is the 21st term in the sequence -/
theorem five_sqrt_five_is_21st_term : a 21 = 5 * Real.sqrt 5 := by
  -- Unfold the definition of a
  unfold a
  -- Simplify the left-hand side
  simp [Real.sqrt_mul]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_sqrt_five_is_21st_term_l365_36535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheapest_route_l365_36561

-- Define the triangle
def triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 4500^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 4000^2 ∧
  (C.1 - B.1) * (B.1 - A.1) + (C.2 - B.2) * (B.2 - A.2) = 0

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Define the cost functions
def bus_cost (d : ℝ) : ℝ := 0.20 * d

def plane_cost (d : ℝ) : ℝ := 120 + 0.15 * d

def plane_cost_BC (d : ℝ) : ℝ := 170 + 0.15 * d

-- Theorem statement
theorem cheapest_route (A B C : ℝ × ℝ) :
  triangle A B C →
  let AB := distance A B
  let BC := distance B C
  let CA := distance C A
  let total_cost := plane_cost AB + bus_cost BC + plane_cost CA
  ∀ route_cost,
    (route_cost = bus_cost AB + bus_cost BC + bus_cost CA ∨
     route_cost = bus_cost AB + bus_cost BC + plane_cost CA ∨
     route_cost = bus_cost AB + plane_cost_BC BC + bus_cost CA ∨
     route_cost = bus_cost AB + plane_cost_BC BC + plane_cost CA ∨
     route_cost = plane_cost AB + bus_cost BC + bus_cost CA ∨
     route_cost = plane_cost AB + bus_cost BC + plane_cost CA ∨
     route_cost = plane_cost AB + plane_cost_BC BC + bus_cost CA ∨
     route_cost = plane_cost AB + plane_cost_BC BC + plane_cost CA) →
    total_cost ≤ route_cost ∧ total_cost = 2115 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheapest_route_l365_36561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_properties_l365_36570

def A : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2^k}

theorem set_A_properties :
  (∀ a ∈ A, ∀ b : ℕ, b < 2*a - 1 → ¬(2*a ∣ b*(b+1))) ∧
  (∀ a ∈ (Set.univ : Set ℕ) \ A, a ≠ 1 → ∃ b : ℕ, b < 2*a - 1 ∧ (2*a ∣ b*(b+1))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_properties_l365_36570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l365_36548

-- Define the function f(x) = e^x - x^2
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x^2

-- State the theorem
theorem f_properties :
  -- 1. The tangent line at (1, f(1)) is y = (e-2)x + 1
  (∀ x, f 1 + (Real.exp 1 - 2) * (x - 1) = (Real.exp 1 - 2) * x + 1) ∧
  -- 2. The maximum value of f(x) on [0, 1] is e - 1
  (∀ x ∈ Set.Icc 0 1, f x ≤ Real.exp 1 - 1) ∧
  (∃ x ∈ Set.Icc 0 1, f x = Real.exp 1 - 1) ∧
  -- 3. For all x > 0, e^x + (1-e)x - x ln x - 1 ≥ 0
  (∀ x > 0, Real.exp x + (1 - Real.exp 1) * x - x * Real.log x - 1 ≥ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l365_36548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_46_gt_tan_44_l365_36555

-- Define the degree-to-radian conversion factor
noncomputable def deg_to_rad : ℝ := Real.pi / 180

-- Define the tangent function for degrees
noncomputable def tan_deg (x : ℝ) : ℝ := Real.tan (x * deg_to_rad)

-- Statement to prove
theorem tan_46_gt_tan_44 : tan_deg 46 > tan_deg 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_46_gt_tan_44_l365_36555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_areas_theorem_l365_36508

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define a point on the ellipse
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

-- Define the area of a triangle
noncomputable def triangleArea (p q : PointOnEllipse) : ℝ :=
  abs (p.x * q.y - q.x * p.y) / 2

-- Theorem statement
theorem triangle_areas_theorem (P Q R : PointOnEllipse) 
  (hP : P ≠ Q ∧ P ≠ R ∧ Q ≠ R)
  (S₁ S₂ S₃ : ℝ)
  (hS₁ : S₁ = triangleArea P Q)
  (hS₂ : S₂ = triangleArea P R)
  (hS₃ : S₃ = triangleArea R Q)
  (hS_pos : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0)
  (hS_eq : S₁^2 + S₂^2 = S₃^2) :
  S₃ = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_areas_theorem_l365_36508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_draws_l365_36549

/-- Represents a chess tournament. -/
structure ChessTournament where
  /-- The number of players in the tournament. -/
  num_players : ℕ
  /-- The number of lists each player creates. -/
  num_lists : ℕ
  /-- The function that determines if player i can reach player j in k or fewer steps. -/
  can_reach : ℕ → ℕ → ℕ → Prop
  /-- The number of games played in the tournament. -/
  num_games : ℕ
  /-- The number of non-draw games in the tournament. -/
  num_non_draws : ℕ

/-- The conditions of our specific tournament. -/
def tournament_conditions (t : ChessTournament) : Prop :=
  t.num_players = 12 ∧
  t.num_lists = 12 ∧
  (∀ i, t.can_reach i i 1) ∧
  (∀ i j k, t.can_reach i j k → t.can_reach i j (k+1)) ∧
  (∀ i j, t.can_reach i j t.num_lists ↔ t.can_reach i j (t.num_lists - 1) ∨ 
    ∃ k, t.can_reach i k (t.num_lists - 1) ∧ t.can_reach k j 1) ∧
  (∀ i, ∃ j, t.can_reach i j t.num_lists ∧ ¬t.can_reach i j (t.num_lists - 1)) ∧
  t.num_games = (t.num_players * (t.num_players - 1)) / 2 ∧
  t.num_non_draws = t.num_players

/-- The theorem to be proved. -/
theorem chess_tournament_draws (t : ChessTournament) :
  tournament_conditions t →
  t.num_games - t.num_non_draws = 54 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_draws_l365_36549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_from_origin_l365_36597

noncomputable def distance_from_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

theorem farthest_point_from_origin :
  let points : List (ℝ × ℝ) := [(2, 3), (-5, 1), (4, -3), (-2, -6), (7, 0)]
  (∀ p ∈ points, distance_from_origin (-2) (-6) ≥ distance_from_origin p.1 p.2) ∧
  (∃ p ∈ points, distance_from_origin (-2) (-6) > distance_from_origin p.1 p.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_from_origin_l365_36597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sum_inequality_l365_36553

def solution_set : Set ℝ := 
  {x | (-9/2 < x ∧ x < -4) ∨ (-4 ≤ x ∧ x < 3) ∨ (3 ≤ x ∧ x < 7/2)}

theorem abs_sum_inequality (x : ℝ) :
  |x - 3| + |x + 4| < 8 ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sum_inequality_l365_36553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_cube_root_l365_36505

theorem xyz_cube_root (x y z : ℝ) :
  (|x - 5| + (y + 1/5)^2 + Real.sqrt (z - 1) = 0) →
  (x * y * z)^(1/3 : ℝ) = -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_cube_root_l365_36505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_different_colors_l365_36567

/-- A coloring of the vertices of a regular n-gon using three colors -/
def Coloring (n : ℕ) := Fin n → Fin 3

/-- The number of vertices colored with a specific color -/
def colorCount (n : ℕ) (c : Coloring n) (color : Fin 3) : ℕ :=
  (Finset.univ.filter (λ v : Fin n => c v = color)).card

/-- Predicate to check if three vertices form an isosceles triangle -/
def IsIsosceles (n : ℕ) (v1 v2 v3 : Fin n) : Prop :=
  sorry -- Definition of isosceles triangle in a regular n-gon

theorem isosceles_triangle_different_colors
  (n : ℕ) (hn : n.Coprime 6) (c : Coloring n)
  (h_odd : ∀ color, Odd (colorCount n c color)) :
  ∃ (v1 v2 v3 : Fin n), IsIsosceles n v1 v2 v3 ∧ c v1 ≠ c v2 ∧ c v2 ≠ c v3 ∧ c v3 ≠ c v1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_different_colors_l365_36567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_special_case_l365_36515

open Real

theorem tangent_sum_special_case (α β : ℝ) 
  (h1 : tan α + tan β = 2) 
  (h2 : (tan α)⁻¹ + (tan β)⁻¹ = 5) : 
  tan (α + β) = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_special_case_l365_36515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_extreme_points_imply_a_range_l365_36580

open Real

/-- The function f(x) defined on (1/2, 2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x / x + a * (x - log x)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (exp x + a * x) * (x - 1) / x^2

/-- The auxiliary function g(x) -/
noncomputable def g (x : ℝ) : ℝ := -exp x / x

theorem three_extreme_points_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 1/2 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < 2 ∧
    f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0 ∧ f_deriv a x₃ = 0) →
  a ∈ Set.Ioo (-2 * Real.sqrt e) (-e) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_extreme_points_imply_a_range_l365_36580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l365_36573

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₁ - C₂| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance between two specific parallel lines -/
theorem distance_between_specific_lines :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 2*x + y - 1 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ 2*x + y + 1 = 0
  distance_between_parallel_lines 2 1 (-1) 1 = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l365_36573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l365_36556

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 8}
def N : Set ℝ := {x : ℝ | x > 4}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Ici (-1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l365_36556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_pure_imaginary_m_zero_l365_36547

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define the property that i is the imaginary unit
theorem i_squared : i * i = -1 := Complex.I_mul_I

-- Define z as a function of m
noncomputable def z (m : ℝ) : ℂ := (1 + i) / (1 - i) + m * (1 - i)

-- Define what it means for a complex number to be pure imaginary
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem pure_imaginary_m_zero :
  ∀ m : ℝ, is_pure_imaginary (z m) → m = 0 := by
  sorry

#check pure_imaginary_m_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_pure_imaginary_m_zero_l365_36547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flooring_rate_is_600_l365_36544

/-- The rate per square meter for flooring a room -/
noncomputable def flooringRate (length width : ℝ) (totalCost : ℝ) : ℝ :=
  totalCost / (length * width)

/-- Theorem: The flooring rate is $600 per square meter -/
theorem flooring_rate_is_600 :
  flooringRate 5.5 3.75 12375 = 600 := by
  -- Unfold the definition of flooringRate
  unfold flooringRate
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flooring_rate_is_600_l365_36544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_not_in_first_quadrant_l365_36564

-- Define the function f(x) = a^x + b
noncomputable def f (a b x : ℝ) : ℝ := a^x + b

-- Theorem statement
theorem function_not_in_first_quadrant 
  (a b : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : b < -1) :
  ∀ x y : ℝ, f a b x = y → ¬(x > 0 ∧ y > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_not_in_first_quadrant_l365_36564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l365_36517

theorem min_value_expression (s t : ℝ) : 
  (s + 5 - 3 * abs (Real.cos t))^2 + (s - 2 * abs (Real.sin t))^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l365_36517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l365_36583

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) : ℝ := 5 * x - 7
noncomputable def g (x : ℝ) : ℝ := 3 * x + 2
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) : ℝ := (x - 3) / 15

-- Theorem statement
theorem h_inverse_correct : 
  ∀ x : ℝ, h (h_inv x) = x ∧ h_inv (h x) = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l365_36583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_l365_36591

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- State the theorem
theorem function_properties_and_inequality :
  -- Given f'(0) = -1
  (∃ a : ℝ, (deriv (f a)) 0 = -1) →
  -- There exists an 'a' such that:
  ∃ a : ℝ, 
    -- a = 2
    a = 2 ∧ 
    -- The minimum value of f(x) is 2 - ln(4)
    (∃ x : ℝ, f a x = 2 - Real.log 4 ∧ ∀ y : ℝ, f a y ≥ f a x) ∧
    -- For all positive x, x^2 < e^x
    (∀ x : ℝ, x > 0 → x^2 < Real.exp x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_inequality_l365_36591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_arithmetic_mean_of_nine_consecutive_numbers_l365_36594

theorem smallest_arithmetic_mean_of_nine_consecutive_numbers (n : ℕ) : 
  (∀ k : ℕ, k ∈ Finset.range 9 → ((n + k).factorial % 1111 = 0)) →
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ∈ Finset.range 9 ∧ ((m + k).factorial % 1111 ≠ 0)) →
  (Finset.range 9).sum (λ k => n + k) / 9 = 97 := by
  sorry

#check smallest_arithmetic_mean_of_nine_consecutive_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_arithmetic_mean_of_nine_consecutive_numbers_l365_36594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l365_36516

def Point := ℝ × ℝ

def is_parallel_to_x_axis (A B : Point) : Prop :=
  A.2 = B.2

noncomputable def distance (A B : Point) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem point_B_coordinates (A B : Point) :
  is_parallel_to_x_axis A B →
  distance A B = 1 →
  A = (-2, 3) →
  (B = (-1, 3) ∨ B = (-3, 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l365_36516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_prism_volume_l365_36519

/-- A triangular prism with specified dimensions -/
structure TriangularPrism where
  base_length : ℝ
  base_height : ℝ
  depth : ℝ

/-- Calculate the volume of a triangular prism -/
noncomputable def volume (prism : TriangularPrism) : ℝ :=
  (1/2) * prism.base_length * prism.base_height * prism.depth

/-- The volume of the specific triangular prism is 80 -/
theorem specific_prism_volume :
  let prism : TriangularPrism := {
    base_length := 8,
    base_height := 5,
    depth := 6
  }
  volume prism = 80 := by
  -- Unfold the definition of volume and perform the calculation
  unfold volume
  -- Simplify the arithmetic
  simp [TriangularPrism.base_length, TriangularPrism.base_height, TriangularPrism.depth]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_prism_volume_l365_36519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_triangle_area_l365_36545

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  b = 2 ∧  -- Given condition
  A + C = 2*π/3 ∧  -- 120° in radians
  a = 2*c →  -- Given condition
  c = 2*Real.sqrt 3/3 := by
  sorry

-- Part 2
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  b = 2 ∧  -- Given condition
  A - C = π/12 ∧  -- 15° in radians
  a = Real.sqrt 2 * c * Real.sin A →  -- Given condition
  1/2 * a * b * Real.sin C = 3 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_triangle_area_l365_36545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_remainder_is_53_l365_36502

/-- M is the 96-digit number formed by concatenating integers from 1 to 53 -/
def M : ℕ := sorry

/-- The remainder when M is divided by 55 -/
def remainder : ℕ := M % 55

theorem M_remainder_is_53 : remainder = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_remainder_is_53_l365_36502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l365_36504

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.B ^ 2 = Real.sin t.A ^ 2 + Real.sin t.C ^ 2 - Real.sin t.A * Real.sin t.C)
  (h2 : t.b = Real.sqrt 3)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 3 / 2) : 
  t.B = π/3 ∧ t.a + t.c = 3 ∧ (-t.a * t.c * Real.cos t.B) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l365_36504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_in_intersecting_cones_l365_36588

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- The maximum squared radius of a sphere fitting in two intersecting cones -/
noncomputable def maxSphereRadiusSquared (cone1 cone2 : Cone) (intersectionDistance : ℝ) : ℝ :=
  let sphereRadius := cone1.baseRadius * (cone1.height - intersectionDistance) / 
    Real.sqrt (cone1.height^2 + cone1.baseRadius^2)
  sphereRadius ^ 2

/-- The theorem stating the maximum squared radius of the sphere -/
theorem max_sphere_radius_squared_in_intersecting_cones 
  (cone1 cone2 : Cone) (intersectionDistance : ℝ) :
  cone1 = cone2 ∧ 
  cone1.baseRadius = 5 ∧ 
  cone1.height = 12 ∧ 
  intersectionDistance = 4 →
  maxSphereRadiusSquared cone1 cone2 intersectionDistance = 1600 / 169 := by
  sorry

#eval (1600 : Nat) + 169

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_in_intersecting_cones_l365_36588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_150_by_40_percent_l365_36566

/-- Calculates the result of increasing a number by a given percentage. -/
noncomputable def increase_by_percent (initial : ℝ) (percent : ℝ) : ℝ :=
  initial * (1 + percent / 100)

/-- Theorem stating that increasing 150 by 40% results in 210. -/
theorem increase_150_by_40_percent :
  increase_by_percent 150 40 = 210 := by
  -- Unfold the definition of increase_by_percent
  unfold increase_by_percent
  -- Simplify the arithmetic
  simp [mul_add, mul_div_right_comm]
  -- Check that the result is equal to 210
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_150_by_40_percent_l365_36566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2α_negative_in_fourth_quadrant_l365_36568

/-- An angle in the fourth quadrant -/
def fourth_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, -Real.pi / 2 + 2 * Real.pi * (k : ℝ) < α ∧ α < 2 * Real.pi * (k : ℝ)

/-- Theorem: If α is in the fourth quadrant, then sin(2α) < 0 -/
theorem sin_2α_negative_in_fourth_quadrant (α : ℝ) (h : fourth_quadrant α) :
  Real.sin (2 * α) < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2α_negative_in_fourth_quadrant_l365_36568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_equality_l365_36522

theorem complex_number_equality (a : ℝ) : 
  (Complex.re ((a - Complex.I) * (1 - Complex.I)) = Complex.im ((a - Complex.I) * (1 - Complex.I))) → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_equality_l365_36522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_prove_inequality_l365_36527

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x + 4|

-- Define the solution set condition
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | f m (x - 2) ≥ 0}

-- Theorem 1: Prove m = 1
theorem find_m (m : ℝ) (h1 : m > 0) (h2 : solution_set m = Set.Icc (-3) (-1)) : m = 1 := by
  sorry

-- Theorem 2: Prove inequality
theorem prove_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/(2*b) + 1/(3*c) = 1) : a + 2*b + 3*c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_prove_inequality_l365_36527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catriona_fish_count_l365_36530

def total_fish (goldfish angelfish guppies tetras bettas : ℕ) : ℕ :=
  goldfish + angelfish + guppies + tetras + bettas

theorem catriona_fish_count :
  ∀ (goldfish angelfish guppies tetras bettas : ℕ),
    goldfish = 8 →
    angelfish = 4 + goldfish / 2 →
    guppies = 2 * (if angelfish ≥ goldfish then angelfish - goldfish else goldfish - angelfish) →
    tetras = max 0 (Int.floor (Real.sqrt (goldfish : ℝ)) - 3) →
    bettas = 5 + tetras ^ 2 →
    total_fish goldfish angelfish guppies tetras bettas = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_catriona_fish_count_l365_36530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_travel_times_l365_36576

/-- Represents the time for a plane's return trip given wind conditions -/
noncomputable def return_trip_time (d : ℝ) (p : ℝ) (w : ℝ) : ℝ := d / (p + w)

/-- Theorem stating the possible return trip times for the given conditions -/
theorem plane_travel_times 
  (d : ℝ) -- distance between towns
  (p : ℝ) -- plane speed in still air
  (w : ℝ) -- wind speed
  (h1 : d / (p - w) = 90) -- trip against wind takes 90 minutes
  (h2 : return_trip_time d p w = d / p - 12) -- return trip is 12 minutes less than still air
  : return_trip_time d p w = 72 ∨ return_trip_time d p w = 15 := by
  sorry

#check plane_travel_times

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_travel_times_l365_36576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biology_collections_l365_36582

/-- Represents the number of each letter in BIOLOGY -/
def biology : Finset Char := {'B', 'I', 'O', 'L', 'O', 'G', 'Y'}

/-- Represents the vowels in BIOLOGY -/
def vowels : Finset Char := {'I', 'O'}

/-- Represents the consonants in BIOLOGY -/
def consonants : Finset Char := {'B', 'L', 'G', 'Y'}

/-- Represents the indistinguishable letters -/
def indistinguishable : Finset Char := {'O', 'G', 'I'}

/-- The number of distinct collections of 3 vowels and 2 consonants from BIOLOGY -/
def distinct_collections : ℕ := 6

theorem biology_collections :
  (Finset.card vowels = 2) →
  (Finset.card consonants = 4) →
  (Finset.card indistinguishable = 3) →
  (∀ c ∈ indistinguishable, Multiset.count c (Finset.val biology) ≤ 2) →
  distinct_collections = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biology_collections_l365_36582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_term_is_59_l365_36577

def joBlairSequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => joBlairSequence n + 2

theorem thirtieth_term_is_59 : joBlairSequence 29 = 59 := by
  -- Proof goes here
  sorry

#eval joBlairSequence 29  -- This will evaluate the 30th term (index 29)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_term_is_59_l365_36577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equivalence_l365_36501

theorem cos_shift_equivalence (x : ℝ) :
  Real.cos (3 * x - π / 3) = Real.sin (3 * (x + π / 18)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_equivalence_l365_36501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_plan_is_80_days_l365_36590

/-- Represents the job completion scenario --/
structure JobCompletion where
  initialWorkers : ℕ
  daysBeforeFiring : ℕ
  workDoneBeforeFiring : ℚ
  firedWorkers : ℕ
  remainingDays : ℕ

/-- The given job completion scenario --/
def givenScenario : JobCompletion where
  initialWorkers := 10
  daysBeforeFiring := 20
  workDoneBeforeFiring := 1/4
  firedWorkers := 2
  remainingDays := 75

/-- Calculates the initial plan in days based on the given scenario --/
def calculateInitialPlan (scenario : JobCompletion) : ℕ :=
  (scenario.daysBeforeFiring : ℕ) * (((1 : ℚ) / scenario.workDoneBeforeFiring).num.toNat)

/-- Theorem stating that the initial plan for the given scenario is 80 days --/
theorem initial_plan_is_80_days :
  calculateInitialPlan givenScenario = 80 := by
  -- Expand the definition of calculateInitialPlan
  unfold calculateInitialPlan
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl

#eval calculateInitialPlan givenScenario

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_plan_is_80_days_l365_36590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l365_36537

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point -/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculate the distance from a line to a point -/
noncomputable def Line.distanceTo (l : Line) (p : Point) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Calculate the area of a triangle formed by a line with x and y axes -/
noncomputable def Line.triangleArea (l : Line) : ℝ :=
  abs (l.c / l.a * l.c / l.b) / 2

theorem line_equations (l : Line) (P B C : Point) :
  P.x = -2 ∧ P.y = 1 ∧
  B.x = -5 ∧ B.y = 4 ∧
  C.x = 3 ∧ C.y = 2 ∧
  l.passesThrough P →
  ((l.distanceTo B = l.distanceTo C) →
    ((l.a = 1 ∧ l.b = 4 ∧ l.c = -2) ∨ (l.a = 2 ∧ l.b = -1 ∧ l.c = -5))) ∧
  (l.triangleArea = 1/2 →
    ((l.a = 1 ∧ l.b = 1 ∧ l.c = -1) ∨ (l.a = 1 ∧ l.b = 4 ∧ l.c = -2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l365_36537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_equation_1_shifted_equation_2_shifted_equation_3_l365_36560

-- Definition of k-shifted equation
def is_k_shifted (f g : ℝ → ℝ) (k : ℕ) : Prop :=
  ∃ (x y : ℝ), f x = 0 ∧ g y = 0 ∧ x - y = k

-- Statement 1
theorem shifted_equation_1 : ∃ k : ℕ, is_k_shifted (λ x ↦ 2*x - 3) (λ x ↦ 2*x - 1) k := by
  sorry

-- Statement 2
theorem shifted_equation_2 (m : ℝ) : 
  is_k_shifted (λ x ↦ 2*x + m + (-4)) (λ x ↦ 2*x + m) 2 := by
  sorry

-- Statement 3
theorem shifted_equation_3 (b c : ℝ) :
  is_k_shifted (λ x ↦ 5*x + b - 1) (λ x ↦ 5*x + c - 1) 3 → 2*b - 2*(c + 3) = -36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_equation_1_shifted_equation_2_shifted_equation_3_l365_36560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l365_36533

-- Define the function f as noncomputable due to Real.sqrt
noncomputable def f (ω : ℚ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)

-- State the theorem
theorem omega_value (ω : ℚ) (h_ω_pos : ω > 0) :
  (∃ α β : ℝ, f ω α = -2 ∧ f ω β = 0 ∧ 
    ∀ γ δ : ℝ, f ω γ = -2 ∧ f ω δ = 0 → |α - β| ≤ |γ - δ|) ∧
  |α - β| = (3 * Real.pi) / 4 →
  ω = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l365_36533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equation_solution_l365_36572

theorem exponent_equation_solution : 
  ∃ y : ℚ, (1000 : ℝ) ^ (y : ℝ) * 10 ^ (3 * (y : ℝ)) = 10000 ^ 4 ∧ y = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equation_solution_l365_36572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_pension_formula_l365_36542

/-- Represents the annual pension calculation for an employee based on years of service -/
structure PensionCalculation where
  k : ℝ  -- Proportionality constant
  x : ℝ  -- Original years of service
  c : ℝ  -- First additional years scenario
  d : ℝ  -- Second additional years scenario
  r : ℝ  -- Pension increase for c additional years
  s : ℝ  -- Pension increase for d additional years

/-- Calculates the pension based on years of service -/
def pension (p : PensionCalculation) (years : ℝ) : ℝ :=
  p.k * years^(3/4)

theorem original_pension_formula (p : PensionCalculation) 
  (hc : pension p (p.x + p.c) = pension p p.x + p.r)
  (hd : pension p (p.x + p.d) = pension p p.x + p.s)
  (hcd : p.c ≠ p.d) :
  pension p p.x = (p.r - p.s) / (3/4 * (p.d - p.c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_pension_formula_l365_36542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_paper_corner_distance_l365_36512

/-- Represents a square sheet of paper with front and back sides --/
structure Paper where
  side_length : ℝ
  area : ℝ
  has_front_and_back : Bool

/-- Represents the state of the paper after folding --/
structure FoldedPaper where
  paper : Paper
  fold_length : ℝ
  visible_black_area : ℝ
  visible_white_area : ℝ

/-- Calculates the distance of a corner point from its original position after folding --/
noncomputable def corner_distance (fp : FoldedPaper) : ℝ :=
  Real.sqrt (2 * fp.fold_length ^ 2)

theorem folded_paper_corner_distance 
  (p : Paper)
  (fp : FoldedPaper)
  (h1 : p.area = 18)
  (h2 : p.has_front_and_back = true)
  (h3 : fp.paper = p)
  (h4 : fp.visible_black_area = fp.visible_white_area) :
  corner_distance fp = 2 * Real.sqrt 6 := by
  sorry

#check folded_paper_corner_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_paper_corner_distance_l365_36512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_sum_terms_l365_36581

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem max_positive_sum_terms (a₁ d : ℝ) :
  a₁ > 0 →
  d < 0 →
  arithmetic_sequence a₁ d 2013 * (arithmetic_sequence a₁ d 2012 + arithmetic_sequence a₁ d 2013) < 0 →
  (∀ n > 4024, arithmetic_sum a₁ d n ≤ 0) ∧
  arithmetic_sum a₁ d 4024 > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_sum_terms_l365_36581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l365_36543

/-- Tetrahedron with given vertices -/
structure Tetrahedron where
  A₁ : ℝ × ℝ × ℝ := (1, 2, 0)
  A₂ : ℝ × ℝ × ℝ := (1, -1, 2)
  A₃ : ℝ × ℝ × ℝ := (0, 1, -1)
  A₄ : ℝ × ℝ × ℝ := (-3, 0, 1)

/-- Volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Height from A₄ to face A₁A₂A₃ -/
def tetraHeight (t : Tetrahedron) : ℝ := sorry

theorem tetrahedron_properties (t : Tetrahedron) :
  volume t = 19 / 6 ∧ tetraHeight t = Real.sqrt (19 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_properties_l365_36543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l365_36587

-- Define the power function f(x) = x^α
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

-- State the theorem
theorem power_function_through_point (α : ℝ) :
  f α (Real.sqrt 2) = 2 → ∀ x : ℝ, f α x = x^2 := by
  intro h
  intro x
  have : α = 2 := by
    -- Proof that α = 2
    sorry
  -- Show that f α x = x^2 for all x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l365_36587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_A_l365_36546

/-- Given a triangle ABC with area 100 square units and geometric mean of sides AB and AC equal to 15 inches, prove that sin A = 8/9 -/
theorem triangle_sin_A (A B C : ℝ) (area geometric_mean : ℝ) :
  area = 100 →
  geometric_mean = 15 →
  geometric_mean^2 = (B - A) * (C - A) →
  area = (1/2) * (B - A) * (C - A) * Real.sin A →
  Real.sin A = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_A_l365_36546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_representation_of_8_11_l365_36524

theorem decimal_representation_of_8_11 : ∃ (d : ℕ → ℕ), 
  (∀ n, d n < 10) ∧ 
  (∀ n, d (n + 2) = d n) ∧
  (8 : ℚ) / 11 = (∑' n, (d n : ℚ) / (10 : ℚ) ^ (n + 1)) ∧
  d 99 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_representation_of_8_11_l365_36524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximation_l365_36507

/-- The area of the region inside a rectangle but outside three quarter circles --/
noncomputable def areaOutsideCircles (rectangleLength : ℝ) (rectangleWidth : ℝ) 
  (radius1 : ℝ) (radius2 : ℝ) (radius3 : ℝ) : ℝ :=
  rectangleLength * rectangleWidth - (Real.pi / 4) * (radius1^2 + radius2^2 + radius3^2)

/-- Theorem stating that the area of the region inside the specific rectangle 
    but outside the three quarter circles is approximately 1.215 --/
theorem area_approximation :
  let rectangleLength : ℝ := 4
  let rectangleWidth : ℝ := 6
  let radius1 : ℝ := 2
  let radius2 : ℝ := 3
  let radius3 : ℝ := 4
  ∃ ε > 0, |areaOutsideCircles rectangleLength rectangleWidth radius1 radius2 radius3 - 1.215| < ε := by
  sorry

-- We can't use #eval for noncomputable functions, so we'll use the following instead:
#check areaOutsideCircles 4 6 2 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximation_l365_36507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_minus_y_zero_l365_36514

noncomputable def average (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := average xs
  (xs.map (fun x => (x - μ)^2)).sum / xs.length

theorem x_minus_y_zero (x y : ℝ) :
  average [x, y, 30, 29, 31] = 30 →
  variance [x, y, 30, 29, 31] = 2 →
  |x - y| = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_minus_y_zero_l365_36514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_distance_l365_36557

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- State the theorem
theorem two_color_distance (coloring : Coloring) (x : ℝ) (h : x > 0) :
  ∃ (c : Color) (p1 p2 : Point), coloring p1 = c ∧ coloring p2 = c ∧ distance p1 p2 = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_distance_l365_36557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l365_36523

/-- A polynomial f(x) is divisible by (x-1)^2 if and only if f(1) = 0 and f'(1) = 0 -/
axiom divisibility_condition {f : ℝ → ℝ} : 
  (∃ g : ℝ → ℝ, f = λ x ↦ (x - 1)^2 * g x) ↔ (f 1 = 0 ∧ (deriv f) 1 = 0)

/-- The polynomial we're considering -/
def f (A B : ℝ) (n : ℕ) : ℝ → ℝ := λ x ↦ A * x^(n+2) + B * x^(n+1) + x - 1

theorem polynomial_divisibility (n : ℕ) :
  (∃ g : ℝ → ℝ, f (-1) 1 n = λ x ↦ (x - 1)^2 * g x) ∧
  ∀ A B : ℝ, (∃ g : ℝ → ℝ, f A B n = λ x ↦ (x - 1)^2 * g x) → A = -1 ∧ B = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l365_36523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l365_36558

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - (1/2) * log x + 1

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2*x - 1/(2*x)

theorem non_monotonic_interval (k : ℝ) :
  (∃ x y, x ∈ Set.Ioo (k - 1) (k + 1) ∧ y ∈ Set.Ioo (k - 1) (k + 1) ∧ f x < f y ∧ x > y) →
  k ∈ Set.Icc 1 (3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l365_36558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_price_equation_l365_36536

theorem flour_price_equation (x : ℝ) (h : x > 0) :
  (9600 / (1.5 * x)) - (6000 / x) = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_price_equation_l365_36536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_expr4_combines_with_sqrt2_l365_36529

-- Define a function to represent the ability to combine with √2
noncomputable def can_combine_with_sqrt2 (x : ℝ) : Prop :=
  ∃ (a : ℝ), x = a * Real.sqrt 2

-- Define the given expressions
noncomputable def expr1 : ℝ := 2 * Real.sqrt 6
noncomputable def expr2 : ℝ := 2 * Real.sqrt 3
def expr3 : ℝ := 2
noncomputable def expr4 : ℝ := 3 * Real.sqrt 2

-- Theorem stating that only expr4 can be combined with √2
theorem only_expr4_combines_with_sqrt2 :
  ¬ can_combine_with_sqrt2 expr1 ∧
  ¬ can_combine_with_sqrt2 expr2 ∧
  ¬ can_combine_with_sqrt2 expr3 ∧
  can_combine_with_sqrt2 expr4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_expr4_combines_with_sqrt2_l365_36529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l365_36520

/-- The area of a triangle with vertices (2,3), (8,6), and (14,2) is 21. -/
theorem triangle_area : ∃ (area : ℝ), area = 21 := by
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (8, 6)
  let C : ℝ × ℝ := (14, 2)
  let area := abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2)) / 2
  use area
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l365_36520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l365_36534

-- Define the vector type
def Vector3D := Fin 3 → ℝ

-- Define the dot product for Vector3D
def dot_product (v w : Vector3D) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

-- Define parallel vectors
def parallel (v w : Vector3D) : Prop :=
  ∃ (k : ℝ), ∀ i, v i = k * w i

-- Define perpendicular vectors
def perpendicular (v w : Vector3D) : Prop :=
  dot_product v w = 0

-- Define the problem statement
theorem problem_statement :
  let a : Vector3D := λ i => [2, 0, -1].get i
  let b : Vector3D := λ i => [-4, 0, 2].get i
  let u : Vector3D := λ i => [2, 2, -1].get i
  let v : Vector3D := λ i => [-3, 4, 2].get i
  (parallel a b) ∧ (perpendicular u v) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l365_36534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_teams_for_scenario_l365_36509

/-- Represents a team in the tournament -/
structure Team where
  wins : Nat
  draws : Nat
  losses : Nat

/-- Calculate the total points for a team -/
def totalPoints (t : Team) : Nat :=
  2 * t.wins + t.draws

/-- Represents the tournament -/
structure Tournament where
  teams : Finset Team
  roundRobin : ∀ t1 t2, t1 ∈ teams → t2 ∈ teams → t1 ≠ t2 → 
    (t1.wins + t1.draws + t1.losses = teams.card - 1)
  uniqueHighestScore : ∃! t, t ∈ teams ∧ ∀ t' ∈ teams, t ≠ t' → totalPoints t > totalPoints t'
  highestScoreFewestWins : ∀ t t', t ∈ teams → t' ∈ teams → 
    (∀ t'' ∈ teams, totalPoints t ≥ totalPoints t'') → 
    (∀ t'' ∈ teams, t.wins ≤ t''.wins)

theorem min_teams_for_scenario (n : Nat) : 
  (∃ t : Tournament, t.teams.card = n) → n ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_teams_for_scenario_l365_36509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_water_volume_ratio_l365_36538

theorem cone_water_volume_ratio (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  (1 / 3) * Real.pi * ((3 / 4) * r)^2 * ((3 / 4) * h) / ((1 / 3) * Real.pi * r^2 * h) = 27 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_water_volume_ratio_l365_36538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_increasing_implies_a_range_l365_36500

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 - 3 * a * x) / Real.log a

-- State the theorem
theorem log_function_increasing_implies_a_range 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : StrictMono (f a) ∧ Set.MapsTo (f a) (Set.Ioo 0 2) (Set.range (f a))) : 
  0 < a ∧ a ≤ 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_increasing_implies_a_range_l365_36500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_and_monotonicity_l365_36550

noncomputable def f (x θ : ℝ) : ℝ := x^2 + 2*x*(Real.tan θ) - 1

theorem function_extrema_and_monotonicity :
  (∀ x ∈ Set.Icc (-1 : ℝ) (Real.sqrt 3),
    (f x (-π/6) ≥ -4/3 ∧ f x (-π/6) ≤ 2*Real.sqrt 3/3)) ∧
  (∀ θ ∈ Set.Ioo (-π/2 : ℝ) (π/2),
    (∀ x ∈ Set.Icc (-1 : ℝ) (Real.sqrt 3),
      Monotone (fun x => f x θ) ∨ StrictMono (fun x => f x θ))
    ↔ θ ∈ Set.Ioc (-π/2) (-π/3) ∪ Set.Ico (π/4) (π/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_and_monotonicity_l365_36550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_sqrt2_plus_minus_1_l365_36598

theorem geometric_mean_sqrt2_plus_minus_1 :
  ∃ a : ℝ, a^2 = (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) ↔ a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_sqrt2_plus_minus_1_l365_36598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_and_U_l365_36510

def U (a : ℝ) : Set ℝ := {2, 3, a^2 + 2*a - 1}
def A (a : ℝ) : Set ℝ := {|1 - 2*a|, 2}

theorem value_of_a_and_U : 
  ∃ (a : ℝ), (U a) \ (A a) = {7} → a = 2 ∧ U a = {2, 3, 7} := by
  sorry

#check value_of_a_and_U

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_and_U_l365_36510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_two_digit_l365_36596

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divisor_count (n : ℕ) : ℕ := (Finset.filter (λ d ↦ n % d = 0) (Finset.range (n + 1))).card

theorem max_divisors_two_digit :
  ∀ n : ℕ, is_two_digit n →
    divisor_count n ≤ 12 ∧
    (n = 60 ∨ n = 72 ∨ n = 90 ∨ n = 96 → divisor_count n = 12) :=
by sorry

#eval divisor_count 60
#eval divisor_count 72
#eval divisor_count 90
#eval divisor_count 96

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_two_digit_l365_36596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_angle_bisector_l365_36528

-- Define the circle A
def circle_A (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 15 = 0

-- Define point B
def point_B : ℝ × ℝ := (1, 0)

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line that intersects C
def intersecting_line (k x : ℝ) : ℝ := k*(x - 1)

-- Main theorem
theorem trajectory_and_angle_bisector :
  ∃ (N : Set (ℝ × ℝ)),
    (∀ M : ℝ × ℝ, circle_A M.1 M.2 → ∃ N' : ℝ × ℝ, N' ∈ N ∧ 
      (∀ x y, (x, y) ∈ N ↔ ellipse_C x y)) ∧
    (∃ R : ℝ × ℝ, R.2 = 0 ∧ R.1 = 4 ∧
      ∀ k : ℝ, ∀ P Q : ℝ × ℝ,
        ellipse_C P.1 P.2 ∧ 
        ellipse_C Q.1 Q.2 ∧
        P.2 = intersecting_line k P.1 ∧ 
        Q.2 = intersecting_line k Q.1 →
        (P.2 - R.2)/(P.1 - R.1) + (Q.2 - R.2)/(Q.1 - R.1) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_angle_bisector_l365_36528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryosuke_trip_cost_l365_36559

/-- Calculates the cost of gas for a trip given odometer readings and fuel efficiency -/
def gas_cost (start_odometer end_odometer : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  let distance := end_odometer - start_odometer
  let gallons_used := (distance : ℚ) / fuel_efficiency
  gallons_used * gas_price

/-- Rounds a rational number to the nearest cent -/
def round_to_cent (x : ℚ) : ℚ :=
  ⌊x * 100 + 1/2⌋ / 100

theorem ryosuke_trip_cost :
  let start_odometer : ℕ := 74568
  let grocery_odometer : ℕ := 74580
  let end_odometer : ℕ := 74608
  let fuel_efficiency : ℚ := 32
  let gas_price : ℚ := 43/10

  round_to_cent (gas_cost start_odometer end_odometer fuel_efficiency gas_price) = 538/100 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryosuke_trip_cost_l365_36559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l365_36503

noncomputable def g (x : ℝ) : ℝ := 1/2 - Real.sqrt (1 - x^2)

noncomputable def g_reflected (x : ℝ) : ℝ := Real.sqrt (1 - x^2) + 1/2

theorem area_between_curves : 
  ∃ (area : ℝ), area = ∫ x in (-1)..(1), (g_reflected x - g x) ∧ area = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l365_36503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_eq_fib_l365_36575

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of valid coin flip sequences of length n -/
def validSequences : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | n + 3 => validSequences (n + 2) + validSequences (n + 1)

/-- Theorem: The number of valid coin flip sequences of length n
    is equal to the (n-1)th Fibonacci number -/
theorem valid_sequences_eq_fib (n : ℕ) (h : n > 0) :
  validSequences n = fib (n - 1) := by
  sorry

#check valid_sequences_eq_fib

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_eq_fib_l365_36575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l365_36552

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  Real.sin t.A = Real.sqrt 5 / 5 ∧
  t.b = 2 * t.a * Real.cos t.A ∧
  t.a * t.c = 5 ∧
  0 < t.B ∧ t.B < Real.pi/2

-- Define the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_properties t) :
  (1/2 * t.a * t.c * Real.sin t.B = 2) ∧
  (Real.sin t.C = 11 * Real.sqrt 5 / 25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l365_36552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l365_36595

-- Define floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define ceiling function
noncomputable def ceil (x : ℝ) : ℤ := -Int.floor (-x)

-- Define nearest integer function
noncomputable def nearest (x : ℝ) : ℤ :=
  if x - Int.floor x < 0.5 then Int.floor x else Int.ceil x

-- Theorem statement
theorem equation_solution (x : ℝ) :
  (3 * (floor x) + 2 * (ceil x) + (nearest x) = 8) ↔ (1 < x ∧ x < 1.5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l365_36595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_when_m_is_neg_two_z_is_pure_imaginary_when_m_is_three_l365_36540

/-- Definition of the complex number z as a function of m -/
noncomputable def z (m : ℂ) : ℂ := (m^2 - m - 6) / (m + 3) + (m^2 + 5*m + 6)*Complex.I

/-- Theorem stating that z is real when m = -2 -/
theorem z_is_real_when_m_is_neg_two :
  z (-2) ∈ Set.range (Complex.ofReal : ℝ → ℂ) :=
sorry

/-- Theorem stating that z is pure imaginary when m = 3 -/
theorem z_is_pure_imaginary_when_m_is_three :
  ∃ (y : ℝ), z 3 = Complex.I * y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_when_m_is_neg_two_z_is_pure_imaginary_when_m_is_three_l365_36540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l365_36532

theorem trigonometric_identity (α β γ : ℝ) 
  (h1 : Real.sin α + Real.sin β + Real.sin γ = 0)
  (h2 : Real.cos α + Real.cos β + Real.cos γ = 0) :
  Real.tan (3 * α) = Real.tan (3 * β) ∧ Real.tan (3 * β) = Real.tan (3 * γ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l365_36532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_quadratic_trinomial_l365_36584

/-- Definition of a quadratic trinomial -/
def is_quadratic_trinomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing π x^2 + 4x - 3 -/
noncomputable def f (x : ℝ) : ℝ := Real.pi * x^2 + 4 * x - 3

/-- Theorem stating that f is a quadratic trinomial -/
theorem f_is_quadratic_trinomial : is_quadratic_trinomial f := by
  use Real.pi, 4, -3
  constructor
  · exact Real.pi_ne_zero
  · intro x
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_quadratic_trinomial_l365_36584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trigonometric_expression_max_value_achievable_l365_36526

open Real

theorem max_value_trigonometric_expression :
  ∀ x y z : ℝ,
  (Real.sin (3 * x) + Real.sin (2 * y) + Real.sin z) * (Real.cos (3 * x) + Real.cos (2 * y) + Real.cos z) ≤ 9 / 2 :=
by sorry

theorem max_value_achievable :
  ∃ x y z : ℝ,
  (Real.sin (3 * x) + Real.sin (2 * y) + Real.sin z) * (Real.cos (3 * x) + Real.cos (2 * y) + Real.cos z) = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trigonometric_expression_max_value_achievable_l365_36526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationery_boxes_solution_l365_36511

/-- Represents the purchase and sales data for two batches of stationery boxes -/
structure StationeryBoxes where
  first_batch_total : ℝ
  second_batch_total : ℝ
  price_ratio : ℝ
  box_difference : ℕ
  selling_price : ℝ
  min_profit : ℝ

/-- Calculates the purchase price per box for the first batch -/
noncomputable def purchase_price_first_batch (data : StationeryBoxes) : ℝ :=
  data.first_batch_total / (data.second_batch_total / (data.price_ratio * data.first_batch_total) - data.box_difference)

/-- Calculates the lowest possible discount for the second batch -/
noncomputable def lowest_discount (data : StationeryBoxes) (first_price : ℝ) : ℝ :=
  10 * (1 - (2 * data.min_profit / (data.second_batch_total / (data.price_ratio * first_price)) - 
    (data.selling_price - data.price_ratio * first_price)) / data.selling_price)

/-- Theorem stating the correct purchase price and lowest discount -/
theorem stationery_boxes_solution (data : StationeryBoxes) 
  (h1 : data.first_batch_total = 1050)
  (h2 : data.second_batch_total = 1440)
  (h3 : data.price_ratio = 1.2)
  (h4 : data.box_difference = 10)
  (h5 : data.selling_price = 24)
  (h6 : data.min_profit = 288) :
  purchase_price_first_batch data = 15 ∧ 
  lowest_discount data (purchase_price_first_batch data) = 8 := by
  sorry

-- Remove the #eval statements as they won't work with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationery_boxes_solution_l365_36511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_monochromatic_triangle_l365_36593

/-- A complete graph with n vertices -/
structure CompleteGraph (n : ℕ) where
  n_ge_6 : n ≥ 6

/-- A 2-coloring of the edges of a complete graph -/
def TwoColoring (G : CompleteGraph n) := Unit

/-- A triangle in a graph -/
structure Triangle (G : CompleteGraph n) where

/-- A monochromatic triangle in a 2-colored complete graph -/
def MonochromaticTriangle (G : CompleteGraph n) (c : TwoColoring G) (t : Triangle G) : Prop := 
  True  -- We use True as a placeholder for the actual condition

/-- In any 2-coloring of a complete graph with at least 6 vertices, 
    there exists a monochromatic triangle -/
theorem exists_monochromatic_triangle 
  (n : ℕ) (G : CompleteGraph n) (c : TwoColoring G) : 
  ∃ (t : Triangle G), MonochromaticTriangle G c t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_monochromatic_triangle_l365_36593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_37_5_l365_36541

-- Define the distances, speeds, and delays
noncomputable def local_distance : ℝ := 60
noncomputable def gravel_distance : ℝ := 10
noncomputable def highway_distance : ℝ := 105
noncomputable def local_speed : ℝ := 30
noncomputable def gravel_speed : ℝ := 20
noncomputable def highway_speed : ℝ := 60
noncomputable def traffic_delay : ℝ := 15 / 60  -- Convert to hours
noncomputable def obstruction_delay : ℝ := 10 / 60  -- Convert to hours

-- Define the total distance
noncomputable def total_distance : ℝ := local_distance + gravel_distance + highway_distance

-- Define the total time
noncomputable def total_time : ℝ :=
  local_distance / local_speed +
  gravel_distance / gravel_speed +
  highway_distance / highway_speed +
  traffic_delay + obstruction_delay

-- State the theorem
theorem average_speed_approx_37_5 :
  abs ((total_distance / total_time) - 37.5) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approx_37_5_l365_36541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l365_36574

theorem calculation_proof : (2023 : ℝ)^0 - (-27) * 3^(-3 : ℤ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l365_36574
