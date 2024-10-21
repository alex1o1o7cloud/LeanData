import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_proof_l1173_117374

/-- The perimeter of a triangle with sides of lengths 15, 8, and 10 is 33. -/
def triangle_perimeter (a b c : ℝ) : ℝ := a + b + c

/-- Proof of the theorem -/
theorem triangle_perimeter_proof : triangle_perimeter 15 8 10 = 33 := by
  unfold triangle_perimeter
  norm_num

#eval triangle_perimeter 15 8 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_proof_l1173_117374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_m_with_sigma_inequality_l1173_117301

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := (Finset.sum (Nat.divisors n) id)

/-- Main theorem -/
theorem infinitely_many_m_with_sigma_inequality (a : ℕ) (ha : a > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ m ∈ S, sigma (a * m) < sigma (a * m + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_m_with_sigma_inequality_l1173_117301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_triangles_l1173_117315

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add necessary fields
  mk :: -- This allows creating a RegularPolygon without specifying fields

/-- Represents a triangle formed by connecting three vertices of a regular polygon -/
structure PolygonTriangle (n : ℕ) where
  polygon : RegularPolygon n
  vertex1 : Fin n
  vertex2 : Fin n
  vertex3 : Fin n

/-- Two triangles are similar if they have the same set of angles -/
def similar (t1 t2 : PolygonTriangle n) : Prop :=
  sorry

/-- The set of all triangles formed by connecting three vertices of a regular polygon -/
def allTriangles (p : RegularPolygon n) : Set (PolygonTriangle n) :=
  sorry

/-- The set of all non-similar triangles in a regular polygon -/
noncomputable def nonSimilarTriangles (p : RegularPolygon n) : Set (PolygonTriangle n) :=
  sorry

/-- A partition of n into three positive integers -/
def Partition3 (n : ℕ) : Type :=
  { p : Fin 3 → ℕ | (p 0) + (p 1) + (p 2) = n ∧ ∀ i, p i > 0 }

/-- The main theorem -/
theorem decagon_triangles :
  ∀ (d : RegularPolygon 10),
  ∃ (f : nonSimilarTriangles d → Partition3 10), Function.Bijective f :=
by
  sorry

#check decagon_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decagon_triangles_l1173_117315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_average_change_l1173_117396

noncomputable def lisa_scores : List ℝ := [92, 89, 93, 95]

noncomputable def average (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

theorem lisa_average_change :
  average lisa_scores - average (lisa_scores.take 3) = 0.92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_average_change_l1173_117396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_range_l1173_117385

open Real

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := log x + 2 * sin α

theorem alpha_range (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : ∃ x₀ : ℝ, x₀ < 1 ∧ deriv (f α) x₀ = f α x₀) : 
  α ∈ Set.Ioo (π/6) (π/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_range_l1173_117385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bertha_family_problem_l1173_117320

/-- Represents the family structure described in the problem -/
structure Family where
  num_daughters : ℕ
  num_total : ℕ
  h_daughters : num_daughters = 6
  h_total : num_total = 30
  h_no_great_grand : ∀ (d : ℕ), d = 0 ∨ d = 6

/-- The number of women in the family who have no daughters -/
def num_no_daughters (f : Family) : ℕ :=
  f.num_total - (f.num_daughters - (f.num_total - f.num_daughters) / 6)

/-- The main theorem to be proved -/
theorem bertha_family_problem (f : Family) : num_no_daughters f = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bertha_family_problem_l1173_117320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_six_terms_equals_63_l1173_117307

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometricSum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

/-- Geometric sequence with first term a₁ and common ratio q -/
noncomputable def geometricSequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  fun n => a₁ * q^(n - 1)

theorem sum_of_first_six_terms_equals_63 :
  let a₁ : ℝ := -1
  let q : ℝ := 2
  let a : ℕ → ℝ := geometricSequence a₁ q
  geometricSum a₁ q 6 = 63 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_six_terms_equals_63_l1173_117307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1173_117339

/-- The line l is defined by the equation x - y + 1 = 0 -/
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

/-- The circle C is defined by the equation (x-2)² + (y-1)² = 4 -/
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

/-- The center of the circle C is at (2, 1) -/
def center : ℝ × ℝ := (2, 1)

/-- The radius of the circle C is 2 -/
def radius : ℝ := 2

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - y + 1| / Real.sqrt 2

theorem line_circle_intersection :
  (∃ x y : ℝ, line_l x y ∧ circle_C x y) ∧
  ¬(line_l center.1 center.2) ∧
  distance_to_line center.1 center.2 < radius := by
  sorry

#check line_circle_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1173_117339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_ball_costs_price_reduction_amount_l1173_117326

-- Define the variables for soccer ball costs
noncomputable def cost_A : ℝ := sorry
noncomputable def cost_B : ℝ := sorry

-- Define the price reduction variable
noncomputable def price_reduction : ℝ := sorry

-- Define the conditions from the problem
axiom purchase_scenario_1 : 5 * cost_A + 3 * cost_B = 450
axiom purchase_scenario_2 : 10 * cost_A + 8 * cost_B = 1000

-- Define the conditions for the price reduction scenario
def original_price_A : ℝ := 80
axiom sales_increase : ∀ m : ℝ, 100 + 20 * m > 200 → m > 5
axiom profit_equation : (original_price_A - price_reduction) * (100 + 20 * price_reduction) + 7000 = 10000

-- Theorem for the cost of soccer balls
theorem soccer_ball_costs : cost_A = 60 ∧ cost_B = 50 := by sorry

-- Theorem for the price reduction
theorem price_reduction_amount : price_reduction = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_ball_costs_price_reduction_amount_l1173_117326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seq_not_arithmetic_nor_geometric_l1173_117373

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - floor x

noncomputable def seq (n : ℕ) : ℝ := frac (1 - (10 : ℝ)^(-n : ℤ))

def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) / a n = r

theorem seq_not_arithmetic_nor_geometric :
  ¬(is_arithmetic seq) ∧ ¬(is_geometric seq) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seq_not_arithmetic_nor_geometric_l1173_117373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_relation_l1173_117311

def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 7, d]

theorem inverse_relation (d k : ℝ) :
  (A d)⁻¹ = k • !![d, 4; 7, 3] → d = 0 ∧ k = 1/28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_relation_l1173_117311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1173_117300

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A : ℝ) : Prop :=
  a = 14 ∧ A = 60 * Real.pi / 180 ∧ b / c = 8 / 5

-- Theorem statement
theorem area_of_triangle_ABC :
  ∀ a b c A,
  triangle_ABC a b c A →
  (1/2) * b * c * Real.sin A = 40 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1173_117300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_state_repetition_reachability_l1173_117377

-- Define the type for the state of the system
def State (n : ℕ) := Fin n → ℕ

-- Define the type for the move operation
def Move (n : ℕ) := Fin n → State n → State n

-- Axiom: The move operation is well-defined and deterministic
axiom move_deterministic {n : ℕ} (m : Move n) (i : Fin n) :
  ∀ s : State n, ∃! s' : State n, m i s = s'

-- Theorem 1: The system will eventually return to its initial state
theorem state_repetition {n : ℕ} (m : Move n) (s : State n) :
  ∃ k : ℕ+, ∃ i : Fin n, (Nat.iterate (m i) k s = s) :=
sorry

-- Theorem 2: Any arrangement can be reached from any other arrangement
theorem reachability {n : ℕ} (m : Move n) :
  ∀ s₁ s₂ : State n, ∃ k : ℕ, ∃ i : Fin n, (Nat.iterate (m i) k s₁ = s₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_state_repetition_reachability_l1173_117377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_median_l1173_117347

def number_set (x : ℕ) : Finset ℕ := {x, 2*x, 3*x, 1, 7}

def is_median (m : ℕ) (s : Finset ℕ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧
  2 * (s.filter (· ≥ m)).card ≥ s.card

theorem smallest_median :
  ∃ (m : ℕ), is_median m (number_set 1) ∧
  ∀ (x : ℕ) (n : ℕ), x > 0 → is_median n (number_set x) → m ≤ n :=
by
  sorry

#check smallest_median

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_median_l1173_117347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1173_117364

theorem power_equation_solution (n : ℝ) : 
  (1/2 : ℝ)^n * (1/81 : ℝ)^(25/2) = 1/(18^25 : ℝ) → n = 25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1173_117364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l1173_117358

theorem complex_fraction_simplification :
  ((((4 : ℚ) + 2)⁻¹ + 2)⁻¹ + 2)⁻¹ + 2 = 77 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l1173_117358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_x_eq_pi_minus_two_over_four_l1173_117381

open MeasureTheory

theorem integral_sqrt_minus_x_eq_pi_minus_two_over_four : 
  ∫ x in Set.Icc 0 1, (Real.sqrt (1 - x^2) - x) = (π - 2) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_x_eq_pi_minus_two_over_four_l1173_117381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAB_is_one_l1173_117392

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + Real.sqrt 2 * t, Real.sqrt 2 * t)

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2*x

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    A = line_l t₁ ∧ circle_C A.1 A.2 ∧
    B = line_l t₂ ∧ circle_C B.1 B.2 ∧
    t₁ ≠ t₂

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the area of a triangle given three points
noncomputable def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (x₃, y₃) := p₃
  abs ((x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂)) / 2)

-- The main theorem
theorem area_of_triangle_OAB_is_one 
  (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  triangle_area origin A B = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAB_is_one_l1173_117392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1173_117322

/-- The number of days it takes for x and y to complete the work together -/
noncomputable def days_together : ℝ := 20

/-- The number of days it takes for y to complete the work alone -/
noncomputable def days_y_alone : ℝ := 80

/-- The work rate of y (work completed per day) -/
noncomputable def work_rate_y : ℝ := 1 / days_y_alone

/-- The work rate of x (work completed per day) -/
noncomputable def work_rate_x : ℝ := 3 * work_rate_y

/-- The combined work rate of x and y -/
noncomputable def work_rate_combined : ℝ := work_rate_x + work_rate_y

theorem work_completion_time :
  days_together = 1 / work_rate_combined := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1173_117322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_earnings_l1173_117314

/-- Represents the earnings for each hour in the repeating pattern -/
def hourly_earnings : Fin 6 → ℕ
| 0 => 2
| 1 => 4
| 2 => 6
| 3 => 2
| 4 => 4
| 5 => 6

/-- Calculates the total earnings for a given number of hours -/
def total_earnings (hours : ℕ) : ℕ :=
  (List.sum (List.map hourly_earnings (List.range 6))) * (hours / 6) +
  (List.sum (List.map hourly_earnings (List.range (hours % 6))))

/-- Theorem stating that Jason's earnings for 48 hours of work is $192 -/
theorem jason_earnings :
  total_earnings 48 = 192 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_earnings_l1173_117314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_center_l1173_117316

noncomputable section

-- Define the circle equation
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + k*x + 2*y + k^2 = 0

-- Define the center of the circle
noncomputable def circle_center (k : ℝ) : ℝ × ℝ := (-k/2, -1)

-- Define the radius squared of the circle
noncomputable def radius_squared (k : ℝ) : ℝ := 1 - (3/4) * k^2

-- Theorem stating the coordinates of the center when area is maximum
theorem max_area_center : 
  ∃ (k : ℝ), (∀ k' : ℝ, radius_squared k ≥ radius_squared k') → 
  circle_center k = (0, -1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_center_l1173_117316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_of_dilated_square_l1173_117351

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  center : Point
  area : ℝ
  vertices : List Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Dilates a point with respect to the origin -/
def dilate (p : Point) (factor : ℝ) : Point :=
  { x := p.x * factor, y := p.y * factor }

/-- Theorem: The farthest vertex of the dilated square -/
theorem farthest_vertex_of_dilated_square (s : Square)
  (h1 : s.center = { x := -3, y := 4 })
  (h2 : s.area = 16)
  (h3 : ∀ p : Point, p ∈ s.vertices → p.y = s.center.y + 2 ∨ p.y = s.center.y - 2)
  (h4 : ∀ p : Point, p ∈ s.vertices → p.x = s.center.x + 2 ∨ p.x = s.center.x - 2) :
  ∃ v : Point, v ∈ (s.vertices.map (dilate · 3)) ∧
    (∀ u : Point, u ∈ (s.vertices.map (dilate · 3)) → distance v { x := 0, y := 0 } ≥ distance u { x := 0, y := 0 }) ∧
    v = { x := -15, y := 18 } :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_of_dilated_square_l1173_117351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l1173_117305

theorem existence_of_special_set : ∃ (S : Finset ℕ), 
  S.card = 2011 ∧ 
  ∀ m n, m ∈ S → n ∈ S → m ≠ n → |Int.ofNat m - Int.ofNat n| = Nat.gcd m n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l1173_117305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_extreme_point_l1173_117303

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - Real.log x - x

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem f_has_one_extreme_point :
  ∃! x : ℝ, x ∈ domain ∧ ∀ y ∈ domain, y ≠ x → (f y - f x) * (y - x) < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_extreme_point_l1173_117303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_15_l1173_117335

-- Define the triangle vertices
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (3, 7)

-- Define the function to calculate the area of a triangle given its vertices
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Theorem statement
theorem triangle_area_is_15 : triangleArea A B C = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_15_l1173_117335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cab_is_right_angle_l1173_117323

-- Define the structure for a circle
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the structure for a line
structure Line where
  point1 : EuclideanSpace ℝ (Fin 2)
  point2 : EuclideanSpace ℝ (Fin 2)

-- Define necessary functions (without implementation)
def circles_touch_at (p : EuclideanSpace ℝ (Fin 2)) (c1 c2 : Circle) : Prop :=
  sorry

def is_external_tangent (l : Line) (c1 c2 : Circle) : Prop :=
  sorry

def tangent_point_on_circle (p : EuclideanSpace ℝ (Fin 2)) (c : Circle) (l : Line) : Prop :=
  sorry

noncomputable def angle_measure (p1 p2 p3 : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry

-- Define the theorem
theorem angle_cab_is_right_angle 
  (circle1 circle2 : Circle)
  (A B C : EuclideanSpace ℝ (Fin 2))
  (tangent : Line) :
  circles_touch_at A circle1 circle2 →
  is_external_tangent tangent circle1 circle2 →
  tangent_point_on_circle B circle1 tangent →
  tangent_point_on_circle C circle2 tangent →
  angle_measure C A B = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cab_is_right_angle_l1173_117323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1173_117395

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (1 - Real.sqrt (1 - x))

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | IsRegular (f x)} = Set.Ioi 0 ∪ Set.Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1173_117395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paco_cookie_consumption_l1173_117386

/-- Represents the number of cookies Paco ate -/
structure CookiesEaten where
  sweet : ℕ
  salty : ℕ

/-- Represents the initial number of cookies -/
structure InitialCookies where
  sweet : ℕ
  salty : ℕ

/-- Calculates the difference between salty and sweet cookies eaten -/
def cookieDifference (eaten : CookiesEaten) : Int :=
  (eaten.salty : Int) - (eaten.sweet : Int)

theorem paco_cookie_consumption (initial : InitialCookies) (eaten : CookiesEaten) 
    (h1 : initial.sweet = 40)
    (h2 : initial.salty = 25)
    (h3 : eaten.sweet * 2 = initial.sweet)
    (h4 : eaten.salty * 5 = initial.salty * 3) :
    cookieDifference eaten = -38 := by
  sorry

#eval cookieDifference { sweet := 80, salty := 42 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paco_cookie_consumption_l1173_117386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_root_condition_l1173_117329

-- Define the inequality
noncomputable def inequality (x : ℝ) : Prop := |x + 3| - 2 * x - 1 < 0

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := |x - m| + |x + 1 / m| - 2

-- Statement for part (I)
theorem solution_set : 
  ∀ x : ℝ, inequality x ↔ x > 2 :=
by sorry

-- Statement for part (II)
theorem root_condition : 
  ∀ m : ℝ, m > 0 → (∃ x : ℝ, f x m = 0) → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_root_condition_l1173_117329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l1173_117359

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := x^3 - 6*x + 4
noncomputable def g (x : ℝ) : ℝ := (3 - x) / 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | f p.1 = g p.1 ∧ p.2 = f p.1}

-- State the theorem
theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    x₁ + x₂ + x₃ = 0 ∧
    y₁ + y₂ + y₃ = 3 := by
  sorry

#check intersection_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l1173_117359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l1173_117310

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*a*x)*Real.log x + 2*a*x - (1/2)*x^2

def has_two_extreme_points (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
  (∀ x : ℝ, x ≠ x₁ ∧ x ≠ x₂ → (deriv f x = 0 → (deriv (deriv f) x ≠ 0))) ∧
  deriv f x₁ = 0 ∧ deriv f x₂ = 0

theorem extreme_points_inequality (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  has_two_extreme_points (f a) a →
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f a x₁ + f a x₂ < (1/2)*a^2 + 3*a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l1173_117310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l1173_117331

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (4, 3)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v w : ℝ × ℝ) : ℝ :=
  (dot_product v w) / (magnitude w)

theorem projection_a_onto_b :
  projection a b = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l1173_117331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_diagonals_regular_heptagon_l1173_117360

/-- RegularPolygon P n s means P is a regular n-gon with side length s -/
def RegularPolygon (P : Set (ℝ × ℝ)) (n : ℕ) (s : ℝ) : Prop := sorry

/-- IsDiagonal P d means d is the length of a diagonal in polygon P -/
def IsDiagonal (P : Set (ℝ × ℝ)) (d : ℝ) : Prop := sorry

/-- DiagonalCount P l returns the number of diagonals in polygon P with length l -/
def DiagonalCount (P : Set (ℝ × ℝ)) (l : ℝ) : ℕ := sorry

/-- The harmonic mean of the diagonals of a regular heptagon with unit side length is 2. -/
theorem harmonic_mean_diagonals_regular_heptagon :
  ∀ (x y : ℝ),
  (∃ (heptagon : Set (ℝ × ℝ)),
    -- The heptagon is regular with unit side length
    RegularPolygon heptagon 7 1 ∧
    -- x and y are the lengths of the two types of diagonals
    (∀ d : ℝ, IsDiagonal heptagon d → d = x ∨ d = y) ∧
    -- There are 7 diagonals of each length
    (DiagonalCount heptagon x = 7 ∧ DiagonalCount heptagon y = 7)) →
  14 / (7/x + 7/y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_diagonals_regular_heptagon_l1173_117360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_six_l1173_117324

/-- Helper function to calculate the area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

/-- Given two lines intersecting at (3,3) with slopes 1/3 and 3 respectively,
    and a third line with equation x + y = 12, prove that the area of the
    triangle formed by their intersection is 6. -/
theorem triangle_area_is_six (line1 line2 line3 : ℝ → ℝ → Prop) :
  (∀ x y, line1 x y ↔ y = (1/3) * x + 2) →
  (∀ x y, line2 x y ↔ y = 3 * x - 6) →
  (∀ x y, line3 x y ↔ x + y = 12) →
  line1 3 3 →
  line2 3 3 →
  (∃ A B C : ℝ × ℝ,
    (line1 A.1 A.2 ∧ line2 A.1 A.2) ∧
    (line1 B.1 B.2 ∧ line3 B.1 B.2) ∧
    (line2 C.1 C.2 ∧ line3 C.1 C.2) ∧
    triangle_area A B C = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_six_l1173_117324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_dwarves_l1173_117355

/-- The number of elves in the company -/
def n : ℕ := sorry

/-- The number of fairies in the company -/
def m : ℕ := sorry

/-- The number of dwarves in the company -/
def k : ℕ := sorry

/-- Each elf is friends with all fairies except for three -/
axiom elf_fairy_friendship : n * (m - 3) = m * (2 * (m - 3))

/-- Each elf is friends with exactly three dwarves -/
axiom elf_dwarf_friendship : n * 3 = n * k

/-- Each fairy is friends with all dwarves -/
axiom fairy_dwarf_friendship : m * k = m * k

/-- Each dwarf is friends with half of the total number of elves and fairies -/
axiom dwarf_friendship : k * ((n + m) / 2) = 3 * n + m * k

theorem company_dwarves : k = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_dwarves_l1173_117355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_f_is_odd_l1173_117309

-- Define the sets A and B
noncomputable def A (a : ℝ) : Set ℝ := {x : ℝ | (x - 1)^2 ≤ a^2 ∧ a > 0}
def B : Set ℝ := {x : ℝ | x < -2 ∨ x > 2}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((x - 2) / (x + 2))

-- Part I
theorem range_of_a (a : ℝ) : A a ∩ B = ∅ → 0 < a ∧ a ≤ 1 := by sorry

-- Part II
theorem f_is_odd : ∀ x ∈ B, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_f_is_odd_l1173_117309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_multiplication_theorem_l1173_117349

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n |>.reverse

def a : List Bool := [true, false, true, true]  -- 1101₂
def b : List Bool := [true, true, false, true]  -- 1011₂
def c : List Bool := [true, true, true]         -- 111₂

theorem binary_multiplication_theorem :
  nat_to_binary ((binary_to_nat a + binary_to_nat b) * binary_to_nat c) =
  [false, false, true, true, true, true, true] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_multiplication_theorem_l1173_117349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_6km_l1173_117308

/-- Represents the travel mode -/
inductive TravelMode
| Walking
| Driving

/-- Represents the travel data for a given distance -/
structure TravelData where
  distance : ℚ
  time : ℚ

/-- Calculates the travel time for a given distance and mode -/
noncomputable def travelTime (v : ℚ) (V : ℚ) (T : ℚ) (d : ℚ) (mode : TravelMode) : ℚ :=
  match mode with
  | TravelMode.Walking => d / v
  | TravelMode.Driving => T + d / V

/-- Theorem: Given the conditions, the time to travel 6 km is 25 minutes -/
theorem travel_time_6km
  (v : ℚ)
  (V : ℚ)
  (T : ℚ)
  (data : List TravelData)
  (h1 : v > 0)
  (h2 : V > 0)
  (h3 : T ≥ 0)
  (h4 : data = [
    ⟨1, 1/6⟩,
    ⟨2, 1/4⟩,
    ⟨3, 7/24⟩
  ])
  (h5 : ∀ d ∈ data, d.time = min (travelTime v V T d.distance TravelMode.Walking) (travelTime v V T d.distance TravelMode.Driving))
  : min (travelTime v V T 6 TravelMode.Walking) (travelTime v V T 6 TravelMode.Driving) = 25/60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_6km_l1173_117308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_specific_parabola_vertex_l1173_117370

/-- The vertex of a parabola defined by y = ax^2 + bx + c is at (-b/(2a), f(-b/(2a))) --/
theorem parabola_vertex (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let x_vertex : ℝ := -b / (2 * a)
  (x_vertex, f x_vertex) = (-b / (2 * a), f (-b / (2 * a))) :=
by sorry

/-- The vertex of the parabola y = -3x^2 - 6x + 2 is at (1, -7) --/
theorem specific_parabola_vertex :
  let f : ℝ → ℝ := λ x => -3 * x^2 - 6 * x + 2
  (1, f 1) = (1, -7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_specific_parabola_vertex_l1173_117370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1173_117343

theorem lambda_range (lambda : ℝ) :
  (∀ m n, n > 0 → (m - n)^2 + (m - Real.log n + lambda)^2 ≥ 2) →
  lambda ≥ 2 ∨ lambda ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1173_117343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1173_117365

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_property (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : b * Real.sin A = a * Real.cos (B - Real.pi/6))
  (h_b : b = 2) :
  B = Real.pi/3 ∧ 
  (∀ (a' c' : ℝ), Triangle a' 2 c' A (Real.pi/3) C → 
    (1/2 * a' * 2 * Real.sin (Real.pi/3) ≤ Real.sqrt 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1173_117365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_calculations_l1173_117321

theorem expression_calculations :
  (6 * Real.sqrt (1 / 9) - (27 : Real)^(1/3) + (Real.sqrt 2)^2 = 1) ∧
  (2 * Real.sqrt 2 + Real.sqrt 9 + (-8 : Real)^(1/3) + abs (Real.sqrt 2 - 2) = Real.sqrt 2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_calculations_l1173_117321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_difference_l1173_117317

/-- Two trains traveling towards each other -/
structure TrainProblem where
  speed1 : ℚ  -- Speed of train 1 in km/hr
  speed2 : ℚ  -- Speed of train 2 in km/hr
  total_distance : ℚ  -- Total distance between stations in km

/-- Calculate the difference in distance traveled by the two trains -/
def distance_difference (p : TrainProblem) : ℚ :=
  let time := p.total_distance / (p.speed1 + p.speed2)
  (p.speed2 * time) - (p.speed1 * time)

/-- The theorem stating the difference in distance traveled -/
theorem train_distance_difference :
  let p := TrainProblem.mk 20 25 585
  distance_difference p = 65 := by
  -- Unfold the definition and simplify
  unfold distance_difference
  simp
  -- The rest of the proof would go here
  sorry

#eval distance_difference (TrainProblem.mk 20 25 585)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_difference_l1173_117317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jelly_bean_distribution_l1173_117342

theorem jelly_bean_distribution (total_beans : ℕ) (people_group1 : ℕ) (people_group2 : ℕ) 
  (beans_per_person_group2 : ℕ) (remaining_beans : ℕ)
  (h1 : total_beans = 8000)
  (h2 : people_group1 = 6)
  (h3 : people_group2 = 4)
  (h4 : beans_per_person_group2 = 400)
  (h5 : remaining_beans = 1600) :
  (total_beans - remaining_beans - people_group2 * beans_per_person_group2) / 
  (people_group2 * beans_per_person_group2) = 3 := by
  sorry

#check jelly_bean_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jelly_bean_distribution_l1173_117342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_to_AK_l1173_117341

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 1) ∧ B = (1, 1) ∧ C = (1, 0) ∧ D = (0, 0)

-- Define point K on side CD
noncomputable def K : ℝ × ℝ := (2/3, 0)

-- Define the line AK
def LineAK (x y : ℝ) : Prop :=
  y = -3/2 * x + 1

-- Define the distance function from a point to a line
noncomputable def DistancePointToLine (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

theorem distance_C_to_AK (A B C D : ℝ × ℝ) (h : Square A B C D) :
  DistancePointToLine 1 0 3 (-2) (-2) = 1 / Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_to_AK_l1173_117341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1173_117376

/-- An ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop

/-- A chord of an ellipse -/
structure Chord (e : Ellipse) where
  p : ℝ × ℝ  -- midpoint of the chord
  on_ellipse : e.equation p.fst p.snd
  is_midpoint : True  -- abstract property that p is the midpoint

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The main theorem -/
theorem chord_equation (e : Ellipse) (ch : Chord e) : 
  e.a = 4 → e.b = 9 → 
  e.equation = (fun x y => 4 * x^2 + 9 * y^2 = 144) →
  ch.p = (3, 2) → 
  ∃ (l : Line), l.a = 2 ∧ l.b = 3 ∧ l.c = -12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1173_117376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1173_117394

noncomputable def z : ℂ := ((-3 : ℂ) + Complex.I) / ((2 : ℂ) + Complex.I)

theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by
  -- Simplify the complex number
  have h1 : z = -1 + Complex.I := by
    -- Proof steps for simplification
    sorry
  
  -- Calculate the modulus
  calc
    Complex.abs z = Complex.abs (-1 + Complex.I) := by rw [h1]
    _ = Real.sqrt ((-1)^2 + 1^2) := by sorry
    _ = Real.sqrt 2 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1173_117394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_true_l1173_117375

-- Define the propositions
def proposition1 : Prop := ∀ x y : ℝ, x * y ≠ 1 → ¬(x = 1 / y ∧ y = 1 / x)

-- For proposition2, we'll use a more generic formulation without specific geometric concepts
def proposition2 : Prop := ∃ A B : Set ℝ, (A ≠ B) ∧ (∃ f : Set ℝ → ℝ, f A = f B)

def proposition3 : Prop := ∀ a b : ℝ, (∃ x : ℝ, x^2 + a*x + b = 0) → b ≥ 0

-- Define the theorem
theorem propositions_true : proposition1 ∧ proposition2 ∧ proposition3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_true_l1173_117375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_generator_angle_l1173_117387

/-- The angle between the generator of a cone and the plane of its base, given that the lateral surface area of the cone is equal to the sum of the areas of its base and axial section. -/
theorem cone_generator_angle (R : ℝ) (h : R > 0) : 
  let α := Real.arccos ((π - 1) / π)
  let l := R / Real.cos α
  let H := R * Real.tan α
  π * R * l = π * R^2 + R * H → α = 2 * Real.arctan (1 / π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_generator_angle_l1173_117387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_grid_ratio_l1173_117379

/-- Represents a grid with shaded triangles --/
structure Grid where
  size : ℕ
  shaded_area : ℚ

/-- The ratio of shaded to unshaded area in the grid --/
noncomputable def shaded_to_unshaded_ratio (g : Grid) : ℚ :=
  g.shaded_area / (g.size^2 - g.shaded_area)

/-- Theorem stating that for a specific 5x5 grid, the ratio is 1/4 --/
theorem specific_grid_ratio :
  ∃ g : Grid, g.size = 5 ∧ shaded_to_unshaded_ratio g = 1/4 := by
  -- Construct the specific grid
  let g : Grid := ⟨5, 5⟩
  -- Show that this grid satisfies the conditions
  have h1 : g.size = 5 := rfl
  have h2 : shaded_to_unshaded_ratio g = 1/4 := by
    -- Proof steps would go here
    sorry
  -- Provide the witness and prove the conjunction
  exact ⟨g, ⟨h1, h2⟩⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_grid_ratio_l1173_117379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_difference_l1173_117350

-- Define the curve
noncomputable def C : ℝ → ℝ := λ x => x / (x - 1)

-- Define the point Q
def Q : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem intersection_dot_product_difference (M N : ℝ × ℝ) :
  M.2 = C M.1 →  -- M is on the curve
  N.2 = C N.1 →  -- N is on the curve
  Q = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) →  -- Q is the midpoint of MN
  (N.1 * Q.1 + N.2 * Q.2) - (M.1 * Q.1 + M.2 * Q.2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_dot_product_difference_l1173_117350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_a_score_ge_135_l1173_117354

/-- Represents the competition structure and scoring system -/
structure Competition :=
  (total_questions : Nat)
  (options_per_question : Nat)
  (correct_score : Nat)
  (not_answered_score : Nat)
  (wrong_score : Nat)
  (total_score : Nat)

/-- Represents a participant's performance -/
structure Participant :=
  (correct_answers : Nat)
  (eliminated_options : Nat)
  (questions_attempted : Nat)
  (options_per_question : Nat)

/-- Calculates the probability of getting a specific number of correct answers -/
def probability_correct (p : Participant) (k : Nat) : Rat :=
  (Nat.choose p.questions_attempted k) * 
  ((1 : Rat) / (p.options_per_question - p.eliminated_options)) ^ k * 
  ((p.options_per_question - p.eliminated_options - 1 : Rat) / (p.options_per_question - p.eliminated_options)) ^ (p.questions_attempted - k)

/-- Theorem: The probability of A's total score being not less than 135 points is 7/27 -/
theorem probability_a_score_ge_135 (comp : Competition) (a : Participant) : 
  comp.total_questions = 25 ∧ 
  comp.options_per_question = 4 ∧ 
  comp.correct_score = 6 ∧ 
  comp.not_answered_score = 2 ∧ 
  comp.wrong_score = 0 ∧ 
  comp.total_score = 150 ∧ 
  a.correct_answers = 20 ∧ 
  a.eliminated_options = 1 ∧ 
  a.questions_attempted = 3 ∧
  a.options_per_question = 4 →
  (probability_correct a 2 + probability_correct a 3 : Rat) = 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_a_score_ge_135_l1173_117354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_solution_set_l1173_117380

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x - 4) + Real.sqrt (5 - x)

-- State the theorem
theorem max_value_and_solution_set :
  (∃ M : ℝ, ∀ x : ℝ, f x ≤ M ∧ ∃ x₀ : ℝ, f x₀ = M) ∧
  (∀ x : ℝ, |x - 1| + |x + 2| ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_solution_set_l1173_117380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_faucet_drains_42_hours_l1173_117388

/-- Represents the draining scenario of a pool with multiple faucets -/
structure PoolDraining where
  /-- The time it takes to drain the pool when all faucets are on simultaneously -/
  all_faucets_time : ℝ
  /-- The ratio of the first faucet's draining time to the last faucet's draining time -/
  first_to_last_ratio : ℝ

/-- Calculates the draining time of the first faucet -/
noncomputable def first_faucet_time (pd : PoolDraining) : ℝ :=
  (2 * pd.all_faucets_time * (pd.first_to_last_ratio + 1)) / (pd.first_to_last_ratio + 1)

/-- Theorem stating that under the given conditions, the first faucet drains for 42 hours -/
theorem first_faucet_drains_42_hours (pd : PoolDraining) 
    (h1 : pd.all_faucets_time = 24)
    (h2 : pd.first_to_last_ratio = 7) : 
  first_faucet_time pd = 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_faucet_drains_42_hours_l1173_117388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1173_117337

def a : ℕ → ℚ
  | 0 => 5  -- Add this case to cover all natural numbers
  | 1 => 5
  | 2 => 5/11
  | n+3 => (a (n+1) * a (n+2)) / (3 * a (n+1) - a (n+2))

theorem sequence_general_term (n : ℕ) (h : n ≥ 1) : a n = 5 / (5 * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1173_117337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_projection_l1173_117356

/-- A vector in R² --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The dot product of two 2D vectors --/
def dot (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

/-- The squared norm of a 2D vector --/
def norm_squared (v : Vector2D) : ℝ := dot v v

/-- Projection of one vector onto another --/
noncomputable def proj (v w : Vector2D) : Vector2D :=
  let scalar := (dot v w) / (norm_squared w)
  { x := scalar * w.x, y := scalar * w.y }

/-- Theorem: The projection of any vector on the line y = 3/2 * x + 3 onto a specific vector w
    is always equal to the vector (-18/13, 12/13) --/
theorem constant_projection :
  ∃ w : Vector2D,
    ∀ a : ℝ,
      let v := { x := a, y := 3/2 * a + 3 : Vector2D }
      proj v w = { x := -18/13, y := 12/13 : Vector2D } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_projection_l1173_117356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1173_117346

def sequence_a : ℕ → ℤ
  | 0 => 1
  | n + 1 => 3 * sequence_a n + 2^(n + 2)

theorem sequence_a_formula (n : ℕ) : 
  sequence_a n = 5 * 3^n - 2^(n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1173_117346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calendar_diagonal_difference_l1173_117389

open BigOperators

def calendar_matrix : Matrix (Fin 5) (Fin 5) ℕ := 
  λ i j => i.val * 5 + j.val + 1

def reverse_row (m : Matrix (Fin 5) (Fin 5) ℕ) (row : Fin 5) : Matrix (Fin 5) (Fin 5) ℕ :=
  λ i j => if i = row then m i (4 - j) else m i j

def main_diagonal_sum (m : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  ∑ i, m i i

def secondary_diagonal_sum (m : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  ∑ i, m i (4 - i)

theorem calendar_diagonal_difference : 
  let m := reverse_row (reverse_row calendar_matrix 2) 4
  |Int.ofNat (main_diagonal_sum m) - Int.ofNat (secondary_diagonal_sum m)| = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calendar_diagonal_difference_l1173_117389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1173_117378

noncomputable def IsFocus (p : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c^2 = a^2 - b^2 ∧
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ 
    dist (x, y) p + dist (x, y) (p.1, -p.2) = 2 * a)

theorem ellipse_foci_distance (x y : ℝ) :
  (x^2 / 36 + y^2 / 9 = 16) →
  (∃ (f₁ f₂ : ℝ × ℝ), IsFocus f₁ ∧ IsFocus f₂ ∧ dist f₁ f₂ = 24 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1173_117378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_radius_sum_l1173_117384

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 14*y + 73 = -y^2 + 8*x

def is_center (a b r : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

def is_radius (r : ℝ) : Prop :=
  ∃ (a b : ℝ), is_center a b r ∧ r > 0

theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), is_center a b r ∧ is_radius r ∧ a + b + r = 11 + 2 * Real.sqrt 2 := by
  sorry

#check circle_center_radius_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_radius_sum_l1173_117384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_expense_popcorn_boxes_l1173_117304

/-- Prove that Jam bought 2 boxes of popcorn given the movie expenses scenario --/
theorem movie_expense_popcorn_boxes :
  let ticket_price : ℚ := 7
  let milk_tea_price : ℚ := 3
  let popcorn_price : ℚ := 3/2
  let num_friends : ℕ := 3
  let individual_contribution : ℚ := 11
  let total_contribution : ℚ := individual_contribution * num_friends
  let ticket_expense : ℚ := ticket_price * num_friends
  let milk_tea_expense : ℚ := milk_tea_price * num_friends
  ∀ popcorn_boxes : ℕ,
  (ticket_expense + milk_tea_expense + popcorn_price * popcorn_boxes = total_contribution) →
  popcorn_boxes = 2 := by
  intro popcorn_boxes hyp
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_expense_popcorn_boxes_l1173_117304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l1173_117344

def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y - 2 = 0

def line2 (m : ℝ) (x y : ℝ) : Prop := 5 * x + (m + 3) * y - 5 = 0

def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄, 
    f x₁ y₁ → f x₂ y₂ → x₁ ≠ x₂ → 
    g x₃ y₃ → g x₄ y₄ → x₃ ≠ x₄ →
    (y₁ - y₂) / (x₁ - x₂) = (y₃ - y₄) / (x₃ - x₄)

theorem parallel_lines_m_value :
  ∀ m : ℝ, parallel (line1 m) (line2 m) → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l1173_117344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_f_neg_one_iff_x_gt_neg_one_l1173_117371

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 4*x + 6 else -x + 6

-- State the theorem
theorem f_less_than_f_neg_one_iff_x_gt_neg_one :
  ∀ x : ℝ, f x < f (-1) ↔ x > -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_f_neg_one_iff_x_gt_neg_one_l1173_117371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_ratio_proof_l1173_117398

/-- Represents the work efficiency of a worker relative to a standard worker -/
@[reducible] def WorkEfficiency := ℝ

/-- Represents the time taken to complete a job in days -/
@[reducible] def Days := ℝ

/-- Given two workers A and B, proves that their efficiency ratio is 1/2
    when B takes 30 days alone and they take 20 days together -/
theorem efficiency_ratio_proof 
  (efficiency_A efficiency_B : WorkEfficiency)
  (time_B_alone : Days)
  (time_together : Days)
  (h1 : time_B_alone = 30)
  (h2 : time_together = 20)
  (h3 : efficiency_B * (1 / time_B_alone) + efficiency_A * (1 / time_B_alone) = 1 / time_together)
  (h4 : efficiency_B = 1) :
  efficiency_A / efficiency_B = 1 / 2 := by
  sorry

#check efficiency_ratio_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_ratio_proof_l1173_117398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_determinant_zero_l1173_117362

/-- Given a cubic polynomial x³ - sx² + px + q = 0 with roots a, b, and c,
    the determinant of the matrix [[a, c, b], [c, b, a], [b, a, c]] is zero. -/
theorem cubic_roots_determinant_zero (s p q : ℝ) (a b c : ℝ) : 
  a^3 - s*a^2 + p*a + q = 0 →
  b^3 - s*b^2 + p*b + q = 0 →
  c^3 - s*c^2 + p*c + q = 0 →
  Matrix.det ![![a, c, b], ![c, b, a], ![b, a, c]] = 0 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_determinant_zero_l1173_117362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_from_distinct_set_not_isosceles_l1173_117367

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define what it means for a triangle to be isosceles
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Define a set of distinct real numbers
def DistinctSet (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≠ y

-- Theorem statement
theorem triangle_from_distinct_set_not_isosceles 
  (s : Set ℝ) (hs : DistinctSet s) (t : Triangle) 
  (ht : {t.a, t.b, t.c} ⊆ s) : 
  ¬ t.isIsosceles :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_from_distinct_set_not_isosceles_l1173_117367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_length_ellipse_equation_l1173_117319

/-- Ellipse C with foci F1 and F2, and a line l passing through F2 intersecting C at A and B -/
structure EllipseWithLine where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  l : ℝ → ℝ
  h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) = A ∨ (x, y) = B
  h_line : l F2.1 = F2.2 ∧ l A.1 = A.2 ∧ l B.1 = B.2
  h_slope : Real.tan (60 * π / 180) = (l 1 - l 0)
  h_distance : |F1.2 - l F1.1| / Real.sqrt (1 + (l 1 - l 0)^2) = 2
  h_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3

/-- The focal length of ellipse C is 4√3/3 -/
theorem focal_length (e : EllipseWithLine) : |e.F2.1 - e.F1.1| = 4 * Real.sqrt 3 / 3 := by
  sorry

/-- The equation of ellipse C is x²/9 + y²/5 = 1 -/
theorem ellipse_equation (e : EllipseWithLine) : e.a^2 = 9 ∧ e.b^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_length_ellipse_equation_l1173_117319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l1173_117363

def is_valid_number (n : ℕ) : Bool :=
  5000 ≤ n ∧ n < 7000 ∧ 
  n % 5 = 0 ∧
  let digits := n.digits 10
  digits.length ≥ 4 ∧
  2 ≤ digits[2]! ∧ digits[2]! < digits[1]! ∧ digits[1]! ≤ 7

def count_valid_numbers : ℕ := (List.range 10000).filter is_valid_number |>.length

theorem valid_numbers_count : count_valid_numbers = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l1173_117363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_implies_b_over_a_geq_neg_one_l1173_117333

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.log x - a * x - b

-- Theorem statement
theorem f_nonpositive_implies_b_over_a_geq_neg_one
  (a b : ℝ)
  (h_a_pos : a > 0)
  (h_f_nonpos : ∀ x > 0, f a b x ≤ 0) :
  b / a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_implies_b_over_a_geq_neg_one_l1173_117333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1173_117393

/-- Given a sequence {a_n} and S_n = 3^n + m - 5, prove that m = 4 if {a_n} is geometric. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (m : ℝ) 
  (h_sum : ∀ n, (Finset.range n).sum (λ i => a i) = 3^n + m - 5)
  (h_geometric : ∀ n, a (n+2) * a n = (a (n+1))^2) :
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1173_117393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_circle_sector_l1173_117366

/-- Represents a sector of a circle divided into an arithmetic sequence of angles -/
structure Sector where
  num_subdivisions : ℕ
  smallest_angle : ℚ
  common_difference : ℚ

/-- The sum of angles in a sector -/
def sector_sum (s : Sector) : ℚ :=
  s.num_subdivisions * (2 * s.smallest_angle + (s.num_subdivisions - 1) * s.common_difference) / 2

theorem smallest_angle_in_circle_sector :
  ∃ (s : Sector), 
    s.num_subdivisions > 0 ∧
    s.smallest_angle > 0 ∧
    s.common_difference > 0 ∧
    sector_sum s = 180 ∧
    (∀ (t : Sector), 
      t.num_subdivisions > 0 → 
      t.smallest_angle > 0 → 
      t.common_difference > 0 → 
      sector_sum t = 180 → 
      s.smallest_angle ≤ t.smallest_angle) ∧
    s.smallest_angle = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_circle_sector_l1173_117366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_variance_set_with_zero_and_one_l1173_117391

/-- A set of n real numbers containing 0 and 1 -/
structure NumberSet (n : ℕ) where
  s : Finset ℝ
  card_eq : s.card = n
  zero_mem : (0 : ℝ) ∈ s
  one_mem : (1 : ℝ) ∈ s

/-- The variance of a set of numbers -/
noncomputable def variance (s : Finset ℝ) : ℝ :=
  let mean := s.sum id / s.card
  s.sum (fun x => (x - mean) ^ 2) / s.card

/-- The theorem stating the minimum variance and the set achieving it -/
theorem min_variance_set_with_zero_and_one (n : ℕ) (h : 2 ≤ n) :
  (∃ (s : NumberSet n), ∀ (t : NumberSet n), variance s.s ≤ variance t.s) ∧
  (∃ (s : NumberSet n), variance s.s = 1 / (2 * n) ∧
    ∀ x ∈ s.s, x = 0 ∨ x = 1 ∨ x = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_variance_set_with_zero_and_one_l1173_117391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_3_range_of_a_l1173_117338

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x + 4|

-- Part I
theorem solution_set_for_a_3 :
  {x : ℝ | f 3 x ≥ 8} = Set.Iic (-3) ∪ Set.Ici 1 := by sorry

-- Part II
theorem range_of_a :
  {a : ℝ | ∃ x, f a x - |x + 2| ≤ 4} = Set.Icc (-6) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_3_range_of_a_l1173_117338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1173_117399

/-- A hyperbola with foci on the x-axis and asymptotic lines y = ± 1/2 x has eccentricity √(5/4) -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : b / a = 1 / 2) -- Condition from asymptotic lines
  (h5 : c^2 = a^2 + b^2) -- Standard hyperbola equation
  : c / a = Real.sqrt (5/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1173_117399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l1173_117348

noncomputable def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  IsOdd (f a b) → a = -1/2 ∧ b = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l1173_117348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_values_l1173_117330

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log (1/4))^2 - (Real.log x / Real.log (1/4)) + 5

-- State the theorem
theorem f_max_min_values :
  ∀ x ∈ Set.Icc 2 4,
    (∀ y ∈ Set.Icc 2 4, f y ≤ 7) ∧
    (∃ z ∈ Set.Icc 2 4, f z = 7) ∧
    (∀ y ∈ Set.Icc 2 4, f y ≥ 5.75) ∧
    (∃ z ∈ Set.Icc 2 4, f z = 5.75) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_values_l1173_117330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_efgh_l1173_117368

/-- The area of a rectangle given the coordinates of three of its vertices -/
noncomputable def rectangleArea (e f h : ℝ × ℝ) : ℝ :=
  let ef := Real.sqrt ((f.1 - e.1)^2 + (f.2 - e.2)^2)
  let eh := Real.sqrt ((h.1 - e.1)^2 + (h.2 - e.2)^2)
  ef * eh

/-- Theorem stating that the area of rectangle EFGH is 202020 -/
theorem rectangle_area_efgh :
  ∃ y : ℤ, rectangleArea (-2, -7) (998, 93) (0, ↑y) = 202020 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_efgh_l1173_117368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hair_color_assignment_l1173_117306

-- Define the set of friends
inductive Friend
| Belov
| Chernov
| Ryzhov

-- Define the set of hair colors
inductive HairColor
| Blonde
| Brunette
| RedHaired

-- Function to assign hair color to a friend
def hairColorAssignment : Friend → HairColor := sorry

-- Function to check if a hair color matches the friend's name
def matchesName (f : Friend) (c : HairColor) : Prop :=
  (f = Friend.Belov ∧ c = HairColor.Blonde) ∨
  (f = Friend.Chernov ∧ c = HairColor.Brunette) ∨
  (f = Friend.Ryzhov ∧ c = HairColor.RedHaired)

-- Theorem statement
theorem hair_color_assignment :
  (∀ f : Friend, ¬matchesName f (hairColorAssignment f)) →
  (∀ f₁ f₂ : Friend, f₁ ≠ f₂ → hairColorAssignment f₁ ≠ hairColorAssignment f₂) →
  hairColorAssignment Friend.Belov ≠ HairColor.Brunette →
  (hairColorAssignment Friend.Belov = HairColor.RedHaired ∧
   hairColorAssignment Friend.Chernov = HairColor.Blonde ∧
   hairColorAssignment Friend.Ryzhov = HairColor.Brunette) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hair_color_assignment_l1173_117306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l1173_117361

/-- Given two lines intersecting at P(2,5) with slopes 3 and -1 respectively,
    and Q and R being the x-intercepts of these lines,
    prove that the area of triangle PQR is 50/3 -/
theorem area_of_triangle_PQR (P Q R : ℝ × ℝ) : 
  P = (2, 5) →
  (∃ (m₁ m₂ : ℝ), m₁ = 3 ∧ m₂ = -1 ∧
    (∀ x y, y - P.2 = m₁ * (x - P.1) ∨ y - P.2 = m₂ * (x - P.1))) →
  Q.2 = 0 ∧ R.2 = 0 →
  (∃ x₁ x₂, Q = (x₁, 0) ∧ R = (x₂, 0) ∧ 
    (x₁ - P.1) * 3 + P.2 = 0 ∧ (x₂ - P.1) * (-1) + P.2 = 0) →
  (1/2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) = 50 / 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l1173_117361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Al_mass_percentage_in_AlI3_l1173_117353

-- Define the atomic masses and composition
noncomputable def atomic_mass_Al : ℝ := 26.98
noncomputable def atomic_mass_I : ℝ := 126.90
def Al_atoms : ℕ := 1
def I_atoms : ℕ := 3

-- Define the molar mass of AlI3
noncomputable def molar_mass_AlI3 : ℝ := atomic_mass_Al * Al_atoms + atomic_mass_I * I_atoms

-- Define the mass percentage calculation
noncomputable def mass_percentage (element_mass : ℝ) (total_mass : ℝ) : ℝ :=
  (element_mass / total_mass) * 100

-- Theorem statement
theorem Al_mass_percentage_in_AlI3 :
  abs (mass_percentage (atomic_mass_Al * Al_atoms) molar_mass_AlI3 - 6.62) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Al_mass_percentage_in_AlI3_l1173_117353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1173_117327

noncomputable section

-- Define the hyperbola M
def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (x y c : ℝ) : Prop := y^2 = 4/3 * c * x

-- Define the right focus
def right_focus (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the intersection points A and B
def intersection_points (a c : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ y, p.1 = a ∧ p.2 = y ∧ parabola a y c}

-- Define the right triangle ABF
def right_triangle (A B F : ℝ × ℝ) : Prop :=
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - F.1)^2 + (B.2 - F.2)^2 ∧
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 0

-- Define the eccentricity
def eccentricity (c a : ℝ) : ℝ := c / a

-- State the theorem
theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hM : ∀ x y, hyperbola x y a b → x^2 / a^2 - y^2 / b^2 = 1)
  (hF : right_focus c = (c, 0))
  (hAB : ∃ A B, A ∈ intersection_points a c ∧ B ∈ intersection_points a c ∧ right_triangle A B (c, 0)) :
  eccentricity c a = 3 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1173_117327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_on_ellipse_l1173_117383

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

/-- The fixed point A -/
def A : ℝ × ℝ := (2, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The range of distances between points on the ellipse and point A -/
theorem distance_range_on_ellipse :
  ∀ (x y : ℝ), is_on_ellipse x y →
  (Real.sqrt 2 / 2 : ℝ) ≤ distance (x, y) A ∧ distance (x, y) A ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_on_ellipse_l1173_117383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_is_distance_l1173_117328

noncomputable def d (x y : ℝ) : ℝ := |x - y| / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2))

theorem d_is_distance : 
  (∀ x y, d x y ≥ 0) ∧ 
  (∀ x y, d x y = d y x) ∧ 
  (∀ x y z, d x y + d y z ≥ d x z) ∧ 
  (∀ x, d x x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_is_distance_l1173_117328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_opposites_l1173_117336

theorem complex_opposites (b : ℝ) : 
  (Complex.re ((2 - b * Complex.I) * Complex.I) = 
   -Complex.im ((2 - b * Complex.I) * Complex.I)) → 
  b = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_opposites_l1173_117336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_number_l1173_117332

theorem existence_of_special_number : ∃ N : ℕ+, 
  (∃ (S : Finset ℕ), S.card = 2000 ∧ ∀ p ∈ S, Nat.Prime p ∧ p ∣ N) ∧ 
  (N ∣ 2 * N + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_number_l1173_117332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kayak_production_l1173_117334

noncomputable section

/-- The sum of a geometric sequence with first term a, common ratio r, and n terms -/
def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

/-- The number of kayaks built in January -/
def initialKayaks : ℝ := 9

/-- The ratio of kayaks built each month compared to the previous month -/
def monthlyRatio : ℝ := 3

/-- The number of months considered -/
def monthCount : ℕ := 4

/-- The total number of kayaks built from January to April -/
def totalKayaks : ℝ := 360

theorem kayak_production :
  geometricSum initialKayaks monthlyRatio monthCount = totalKayaks := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kayak_production_l1173_117334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_ratio_l1173_117340

-- Define the basic geometric objects
def Point : Type := ℝ × ℝ
def Line : Type := Point → Prop
def Circle : Type := Point → ℝ → Prop

-- Define the segment AB and point C
noncomputable def AB : Set Point := sorry
noncomputable def C : Point := sorry

-- Define the semicircles
noncomputable def α : Circle := sorry
noncomputable def β : Circle := sorry
noncomputable def γ : Circle := sorry

-- Define the inscribed circles δ_n
noncomputable def δ : ℕ → Circle := sorry

-- Define the radius of δ_n
noncomputable def r (n : ℕ) : ℝ := sorry

-- Define the distance from the center of δ_n to line AB
noncomputable def d (n : ℕ) : ℝ := sorry

-- State the theorem
theorem inscribed_circle_ratio : 
  ∀ n : ℕ, d n / r n = 2 * (n : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_ratio_l1173_117340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_line_eq_l1173_117397

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + (y-1)^2 = 4

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define symmetry about a point
def symmetric_about (A B P : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define a line equation
def line_eq (x y : ℝ) : Prop := x + y - 3 = 0

-- State the theorem
theorem circle_symmetry_line_eq :
  ∀ (A B : ℝ × ℝ),
  my_circle A.1 A.2 → my_circle B.1 B.2 →
  symmetric_about A B P →
  line_eq A.1 A.2 ∧ line_eq B.1 B.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_line_eq_l1173_117397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_height_at_ten_l1173_117382

/-- Represents a parabolic bridge function -/
def bridge_function (a k : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + k

theorem bridge_height_at_ten (a k : ℝ) :
  bridge_function a k 0 = 25 →
  bridge_function a k 25 = 0 →
  bridge_function a k (-25) = 0 →
  bridge_function a k 10 = 21 := by
  sorry

#check bridge_height_at_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_height_at_ten_l1173_117382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_marks_l1173_117352

theorem exam_average_marks (total_boys : ℕ) (passed_boys : ℕ) (total_avg : ℚ) (passed_avg : ℚ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 105)
  (h3 : total_avg = 36)
  (h4 : passed_avg = 39) :
  (total_boys * total_avg - passed_boys * passed_avg) / (total_boys - passed_boys) = 15 := by
  -- Start of the proof
  sorry

#check exam_average_marks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_marks_l1173_117352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1173_117313

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x + 1) / Real.log (a^2 - 1)

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo (-1/2 : ℝ) 0, f a x > 0) → 
  a ∈ Set.union (Set.Ioo (-Real.sqrt 2) (-1)) (Set.Ioo 1 (Real.sqrt 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1173_117313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_block_distance_increase_l1173_117345

/-- The increase in distance between two blocks after a compressed spring expands -/
theorem block_distance_increase 
  (m : ℝ) -- mass of each block
  (μ : ℝ) -- coefficient of friction
  (g : ℝ) -- gravitational acceleration
  (PE : ℝ) -- initial potential energy of the compressed spring
  (h₁ : m > 0)
  (h₂ : μ > 0)
  (h₃ : g > 0)
  (h₄ : PE > 0) :
  ∃ L : ℝ, L = PE / (μ * m * g) ∧ L > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_block_distance_increase_l1173_117345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_iff_lambda_gt_neg_three_l1173_117325

/-- A sequence defined by a_n = n^2 + lambda*n for n ∈ ℕ+ -/
def a (lambda : ℝ) (n : ℕ+) : ℝ := n.val^2 + lambda * n.val

/-- The condition for the sequence to be increasing -/
def is_increasing (lambda : ℝ) : Prop :=
  ∀ n : ℕ+, a lambda n < a lambda (n + 1)

theorem sequence_increasing_iff_lambda_gt_neg_three :
  ∀ lambda : ℝ, is_increasing lambda ↔ lambda > -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_iff_lambda_gt_neg_three_l1173_117325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_m_n_is_135_degrees_l1173_117369

noncomputable section

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (9, 12)
def c : ℝ × ℝ := (4, -3)

def m : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)
def n : ℝ × ℝ := (a.1 + c.1, a.2 + c.2)

def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem angle_between_m_n_is_135_degrees :
  angle_between m n = 135 * π / 180 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_m_n_is_135_degrees_l1173_117369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_5_distance_l1173_117312

def z : ℕ → ℂ
  | 0 => 1
  | n + 1 => (z n)^2 - 1 + Complex.I

theorem z_5_distance : Complex.abs (z 5) = Real.sqrt 157 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_5_distance_l1173_117312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_bisector_concurrency_l1173_117302

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Define a function to check if a line divides the perimeter in half
def divides_perimeter_in_half (t : Triangle) (l : Line) : Prop :=
  sorry

-- Define a function to check if three lines intersect at a single point
def intersect_at_single_point (l1 l2 l3 : Line) : Prop :=
  sorry

-- State the theorem
theorem perimeter_bisector_concurrency (t : Triangle) 
  (l1 l2 l3 : Line) : 
  divides_perimeter_in_half t l1 ∧ 
  divides_perimeter_in_half t l2 ∧ 
  divides_perimeter_in_half t l3 →
  intersect_at_single_point l1 l2 l3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_bisector_concurrency_l1173_117302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_supplement_calculation_l1173_117357

/-- Calculates the monthly pension supplement given the following parameters:
    * monthly_contribution: The amount contributed each month
    * annual_interest_rate: The annual interest rate as a decimal
    * contribution_years: The number of years contributions are made
    * payout_years: The number of years the pension is paid out
-/
noncomputable def calculate_monthly_pension_supplement (
  monthly_contribution : ℝ
) (annual_interest_rate : ℝ
) (contribution_years : ℕ
) (payout_years : ℕ
) : ℝ :=
  let annual_contribution := monthly_contribution * 12
  let accumulated_amount := annual_contribution * 
    ((1 + annual_interest_rate) ^ contribution_years - 1) / 
    annual_interest_rate * (1 + annual_interest_rate)
  accumulated_amount / (payout_years * 12)

/-- Theorem stating that given the specified conditions, 
    the monthly pension supplement is approximately 26023.45 rubles -/
theorem pension_supplement_calculation :
  let monthly_contribution : ℝ := 7000
  let annual_interest_rate : ℝ := 0.09
  let contribution_years : ℕ := 20
  let payout_years : ℕ := 15
  abs (calculate_monthly_pension_supplement 
    monthly_contribution 
    annual_interest_rate 
    contribution_years 
    payout_years - 26023.45) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_supplement_calculation_l1173_117357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mappings_count_l1173_117318

def A : Finset Nat := {1, 2, 3}
def B : Finset Nat := {1, 2, 3, 4}

def num_one_to_one_mappings (A B : Finset Nat) : Nat :=
  Nat.factorial B.card / Nat.factorial (B.card - A.card)

def num_all_mappings (A B : Finset Nat) : Nat :=
  Finset.sum (Finset.range (B.card + 1)) (fun k => k ^ A.card * Nat.choose B.card k)

theorem mappings_count (A B : Finset Nat) 
  (hA : A.card = 3) (hB : B.card = 4) : 
  num_one_to_one_mappings A B = 24 ∧ num_all_mappings A B = 224 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mappings_count_l1173_117318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposites_imply_neg_two_l1173_117372

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the real part function
def re (z : ℂ) : ℝ := Complex.re z

-- Define the imaginary part function
def im (z : ℂ) : ℝ := Complex.im z

-- State the theorem
theorem opposites_imply_neg_two (a : ℝ) :
  re ((2 - a * i) * i) = -im ((2 - a * i) * i) → a = -2 :=
by
  intro h
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposites_imply_neg_two_l1173_117372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_norm_range_l1173_117390

theorem vector_norm_range (a b c : ℝ × ℝ) : 
  ‖a‖ = 1 → 
  ‖b‖ = 1 → 
  a • b = 0 → 
  ‖c - 2 • a - b‖ = 1 → 
  ∃ (x : ℝ), x ∈ Set.Icc (6 - 2 * Real.sqrt 5) (6 + 2 * Real.sqrt 5) ∧ ‖c‖^2 = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_norm_range_l1173_117390
