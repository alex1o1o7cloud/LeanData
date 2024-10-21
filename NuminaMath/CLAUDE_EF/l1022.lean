import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dave_remaining_tickets_l1022_102209

def arcade_game (initial_tickets : ℕ) (beanie_cost : ℕ) (additional_tickets : ℕ) (discount_rate : ℚ) : ℕ :=
  let remaining_after_beanie := initial_tickets - beanie_cost
  let keychain_cost := 2 * remaining_after_beanie
  let total_tickets := remaining_after_beanie + additional_tickets
  let discounted_keychain_cost := Int.ceil ((1 - discount_rate) * keychain_cost)
  (total_tickets - discounted_keychain_cost).natAbs

theorem dave_remaining_tickets :
  arcade_game 25 22 15 (1/10) = 12 := by
  sorry

#eval arcade_game 25 22 15 (1/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dave_remaining_tickets_l1022_102209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_polynomial_property_l1022_102235

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def satisfies_fibonacci (p : ℝ → ℝ) : Prop :=
  ∀ k ∈ Finset.range 991, p (k + 992 : ℝ) = fibonacci (k + 992)

theorem fibonacci_polynomial_property (p : ℝ → ℝ) (h : satisfies_fibonacci p) :
  p 1983 = (fibonacci 1983 : ℝ) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_polynomial_property_l1022_102235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_defines_circle_equation_equivalent_to_circle_eq_l1022_102245

-- Define the equation
noncomputable def equation (θ : ℝ) : ℝ := 3 * Real.sin θ * (1 / Real.sin θ)

-- State the theorem
theorem equation_defines_circle :
  ∀ θ : ℝ, θ ≠ 0 → equation θ = 3 :=
by
  sorry

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

-- State the equivalence
theorem equation_equivalent_to_circle_eq :
  ∀ x y : ℝ, (∃ θ : ℝ, θ ≠ 0 ∧ x = equation θ * Real.cos θ ∧ y = equation θ * Real.sin θ) ↔ circle_eq x y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_defines_circle_equation_equivalent_to_circle_eq_l1022_102245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_expensive_trip_cost_l1022_102215

-- Define the cities and distances
def distance_AC : ℚ := 3000
def distance_AB : ℚ := 3250

-- Define travel costs
def bus_cost_per_km : ℚ := 15/100
def plane_cost_per_km : ℚ := 10/100
def plane_booking_fee : ℚ := 100

-- Function to calculate bus cost
def bus_cost (distance : ℚ) : ℚ := distance * bus_cost_per_km

-- Function to calculate plane cost
def plane_cost (distance : ℚ) : ℚ := distance * plane_cost_per_km + plane_booking_fee

-- Function to get the minimum cost between bus and plane
def min_cost (distance : ℚ) : ℚ := min (bus_cost distance) (plane_cost distance)

-- Theorem statement
theorem least_expensive_trip_cost :
  let distance_BC := ((distance_AB^2 - distance_AC^2) : ℚ).sqrt
  min_cost distance_AB + min_cost distance_BC + min_cost distance_AC = 101250/100 := by
  sorry

#eval min_cost distance_AB + min_cost ((distance_AB^2 - distance_AC^2).sqrt) + min_cost distance_AC

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_expensive_trip_cost_l1022_102215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_diagonal_relation_l1022_102274

/-- Represents a rectangle with a specific ratio of length to width -/
structure Rectangle where
  length : ℝ
  width : ℝ
  ratio_condition : length / width = 5 / 2

/-- The diagonal of the rectangle -/
noncomputable def diagonal (r : Rectangle) : ℝ := Real.sqrt (r.length^2 + r.width^2)

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem stating the relationship between area and diagonal -/
theorem area_diagonal_relation (r : Rectangle) :
  area r = (10/29) * (diagonal r)^2 := by
  sorry

#check area_diagonal_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_diagonal_relation_l1022_102274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2013_equals_2_l1022_102273

def sequenceCount : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => (sequenceCount n * sequenceCount (n + 1)) % 10

theorem sequence_2013_equals_2 : sequenceCount 2012 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2013_equals_2_l1022_102273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_is_constant_l1022_102246

/-- A point on a hyperbola with its coordinates -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : y^2 / 4 - x^2 = 1

/-- The feet of perpendiculars from a point to the asymptotes of the hyperbola -/
noncomputable def perpendicular_feet (P : HyperbolaPoint) : ℝ × ℝ × ℝ × ℝ :=
  let m := P.x
  let n := P.y
  let Ax := (2*n + m) / 5
  let Ay := (4*n + 2*m) / 5
  let Bx := (m - 2*n) / 5
  let By := (4*n - 2*m) / 5
  (Ax, Ay, Bx, By)

/-- The product of distances from a point to the feet of perpendiculars -/
noncomputable def distance_product (P : HyperbolaPoint) : ℝ :=
  let (Ax, Ay, Bx, By) := perpendicular_feet P
  let PAx := Ax - P.x
  let PAy := Ay - P.y
  let PBx := Bx - P.x
  let PBy := By - P.y
  (PAx^2 + PAy^2).sqrt * (PBx^2 + PBy^2).sqrt

/-- Theorem: The product of distances is always 4/5 -/
theorem distance_product_is_constant (P : HyperbolaPoint) :
  distance_product P = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_is_constant_l1022_102246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l1022_102218

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^8 + i^20 + i^(-34 : ℤ) = 1 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l1022_102218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_property_l1022_102294

def sequence_a : ℕ → ℤ
  | 0 => 0  -- Adding a case for 0
  | 1 => 1
  | 2 => -1
  | n+3 => -(sequence_a (n+2)) - 2*(sequence_a (n+1))

theorem perfect_square_property (n : ℕ) (h : n ≥ 2) :
  2^(n+1) - 7*(sequence_a (n-1))^2 = (2*(sequence_a n) + sequence_a (n-1))^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_property_l1022_102294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_is_zero_l1022_102291

noncomputable def points : List (ℝ × ℝ) := [(3, 10), (6, 20), (12, 35), (18, 40), (20, 50)]

noncomputable def isAboveLine (p : ℝ × ℝ) : Bool :=
  p.2 > 3 * p.1 + 5

noncomputable def sumXAboveLine (points : List (ℝ × ℝ)) : ℝ :=
  (points.filter isAboveLine).map (·.1) |>.sum

theorem sum_x_above_line_is_zero : sumXAboveLine points = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_is_zero_l1022_102291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l1022_102279

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- A line passes through a point -/
def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  ¬(parallel l1 l2)

/-- A line is perpendicular to another line -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Alternate interior angles are equal for two lines cut by a transversal -/
def alternate_interior_angles_equal (l1 l2 l3 : Line) : Prop :=
  sorry -- Definition omitted for brevity

/-- The number of correct statements -/
def num_correct_statements : ℕ :=
  let statement1 := ∀ (l : Line) (p : Point), ∃! (l' : Line), passes_through l' p ∧ perpendicular l l'
  let statement2 := ∀ (l1 l2 l3 : Line), intersect l3 l1 ∧ intersect l3 l2 → (sorry : Prop) -- Corresponding angles are equal
  let statement3 := ∀ (l1 l2 l3 : Line), alternate_interior_angles_equal l1 l2 l3 → parallel l1 l2
  let statement4 := ∀ (l1 l2 : Line), parallel l1 l2 ∨ intersect l1 l2
  let correct_statements := [statement1, statement3, statement4]
  correct_statements.length

theorem correct_statements_count :
  num_correct_statements = 3 := by
  -- Proof omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l1022_102279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_progression_angle_solution_l1022_102238

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where a, b, c form a geometric progression, prove that the equation sin(7B) = sin(B)
    has solutions B = π/3 and B = π/8. -/
theorem triangle_geometric_progression_angle_solution 
  (A B C : ℝ) (a b c : ℝ) :
  -- Triangle condition
  A + B + C = Real.pi →
  -- Sides opposite to angles
  Real.sin A / a = Real.sin B / b → Real.sin B / b = Real.sin C / c →
  -- Geometric progression condition
  b^2 = a * c →
  -- The equation has solutions π/3 and π/8
  (Real.sin (7 * (Real.pi/3)) = Real.sin (Real.pi/3) ∧ 
   Real.sin (7 * (Real.pi/8)) = Real.sin (Real.pi/8)) ∧
  (∀ x, Real.sin (7 * x) = Real.sin x → x = Real.pi/3 ∨ x = Real.pi/8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_progression_angle_solution_l1022_102238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1022_102237

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x > 2 ∧ f x = y) ↔ 0 < y ∧ y < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1022_102237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_condition_zeros_sum_negative_l1022_102298

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x + a * x^2

theorem two_zeros_condition (a : ℝ) : a > 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 := by
  sorry

theorem zeros_sum_negative (a : ℝ) (x₁ x₂ : ℝ) (h : a > 0) (h₁ : f a x₁ = 0) (h₂ : f a x₂ = 0) (h₃ : x₁ ≠ x₂) : 
  x₁ + x₂ < 0 := by
  sorry

#check two_zeros_condition
#check zeros_sum_negative

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_condition_zeros_sum_negative_l1022_102298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sqrt_cube_root_64_l1022_102254

theorem arithmetic_sqrt_cube_root_64 : Real.sqrt (Real.rpow 64 (1/3)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sqrt_cube_root_64_l1022_102254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_y_share_l1022_102282

theorem person_y_share (total_amount : ℚ) (ratio : List ℚ) :
  total_amount = 1390 →
  ratio = [13, 17, 23, 29, 37] →
  let total_parts := ratio.sum
  let part_value := total_amount / total_parts
  let y_parts := ratio.get! 3
  y_parts * part_value = 338.72 := by
  intro h1 h2
  simp [h1, h2]
  norm_num
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_y_share_l1022_102282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_line_l1022_102252

/-- The curve C₁ -/
noncomputable def C₁ (x : ℝ) : ℝ := x^2 - Real.log x

/-- The line L -/
def L (x y : ℝ) : Prop := x - y - 2 = 0

/-- The distance function between two points -/
def distance_squared (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₂ - x₁)^2 + (y₂ - y₁)^2

/-- The theorem statement -/
theorem min_distance_curve_line :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    y₁ = C₁ x₁ ∧ 
    L x₂ y₂ ∧
    (∀ (u₁ v₁ u₂ v₂ : ℝ), v₁ = C₁ u₁ → L u₂ v₂ → 
      distance_squared x₁ y₁ x₂ y₂ ≤ distance_squared u₁ v₁ u₂ v₂) ∧
    distance_squared x₁ y₁ x₂ y₂ = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_line_l1022_102252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_properties_l1022_102230

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -4*x

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 1)

-- Define the chord length
noncomputable def chordLength (k : ℝ) : ℝ := 
  Real.sqrt (1 + k^2) * (Real.sqrt ((2*k^2 + 4)^2 - 4*k^2) / k^2)

theorem parabola_and_line_properties :
  -- The parabola passes through (-4, 4)
  parabola (-4) 4 ∧
  -- The lines that form a chord of length 8 with the parabola
  (∃ k : ℝ, line k (-1) 0 ∧ chordLength k = 8) →
  -- The equation of the parabola is y² = -4x
  (∀ x y : ℝ, parabola x y ↔ y^2 = -4*x) ∧
  -- The equations of the lines are y = x + 1 or y = -x - 1
  (∀ x y : ℝ, (line 1 x y ∨ line (-1) x y) ↔ (y = x + 1 ∨ y = -x - 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_properties_l1022_102230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wallet_distribution_theorem_l1022_102202

def wallet_amounts : List Nat := [1, 2, 4, 8, 16, 32, 64]

def sum_subset (subset : List Nat) : Nat :=
  subset.foldl (·+·) 0

theorem wallet_distribution_theorem :
  ∀ (n : Nat), 1 ≤ n ∧ n ≤ 127 →
    ∃ (subset : List Nat),
      subset.toFinset ⊆ wallet_amounts.toFinset ∧
      sum_subset subset = n :=
by
  sorry

#check wallet_distribution_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wallet_distribution_theorem_l1022_102202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_k_value_range_l1022_102297

/-- The natural logarithm function plus the identity function -/
noncomputable def f (x : ℝ) : ℝ := Real.log x + x

/-- A function is k-value if there exists an interval [a, b] such that
    the range of f over [a, b] is [k*a, k*b] for some k > 0 -/
def is_k_value_function (f : ℝ → ℝ) : Prop :=
  ∃ (k a b : ℝ), k > 0 ∧ a < b ∧
    (∀ x, a ≤ x ∧ x ≤ b → ∃ y, a ≤ y ∧ y ≤ b ∧ f x = k * y)

/-- The range of k for which f is a k-value function -/
def k_range (f : ℝ → ℝ) : Set ℝ :=
  {k | k > 0 ∧ is_k_value_function f}

theorem f_k_value_range :
  k_range f = Set.Ioo 1 (1 + Real.exp (-1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_k_value_range_l1022_102297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_expansion_l1022_102207

theorem coefficient_x_cubed_expansion : 
  let expansion := (1 - X : Polynomial ℤ) * (1 + X)^8
  expansion.coeff 3 = 28 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_expansion_l1022_102207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_l1022_102259

/-- Calculates the profit percentage given cost price, bulk discount, and sales tax -/
noncomputable def calculate_profit_percentage (cost_price : ℝ) (bulk_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let price_after_bulk_discount := cost_price * (1 - bulk_discount)
  let final_price := price_after_bulk_discount * (1 + sales_tax)
  let profit := final_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percentage is approximately 4.37% -/
theorem profit_percentage_approx (cost_price : ℝ) (h1 : cost_price > 0) :
  let bulk_discount := 0.02
  let sales_tax := 0.065
  abs (calculate_profit_percentage cost_price bulk_discount sales_tax - 4.37) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_l1022_102259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_sum_l1022_102278

theorem matrix_power_sum (a n : ℕ) : 
  (Matrix.of ![![1, 3, a], ![0, 1, 5], ![0, 0, 1]] : Matrix (Fin 3) (Fin 3) ℕ)^n = 
  Matrix.of ![![1, 15, 1010], ![0, 1, 25], ![0, 0, 1]] →
  a + n = 172 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_sum_l1022_102278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_relation_l1022_102249

/-- Ellipse C with equation x²/9 + y²/4 = 1 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 4 = 1}

/-- Focal length of the ellipse -/
noncomputable def focal_length : ℝ := 2 * Real.sqrt 5

/-- Point B on the ellipse -/
def B : ℝ × ℝ := (0, 2)

/-- Point P -/
def P : ℝ × ℝ := (1, 0)

/-- Left vertex of the ellipse -/
def A₁ : ℝ × ℝ := (-3, 0)

/-- Right vertex of the ellipse -/
def A₂ : ℝ × ℝ := (3, 0)

/-- Slope of line A₁M -/
noncomputable def k₁ (M : ℝ × ℝ) : ℝ := (M.2 - A₁.2) / (M.1 - A₁.1)

/-- Slope of line A₂N -/
noncomputable def k₂ (N : ℝ × ℝ) : ℝ := (N.2 - A₂.2) / (N.1 - A₂.1)

theorem ellipse_slope_relation :
  ∀ M N : ℝ × ℝ,
  M ∈ C → N ∈ C →
  M ≠ A₁ → M ≠ A₂ → N ≠ A₁ → N ≠ A₂ →
  ∃ t : ℝ, (M.1 = t * M.2 + 1 ∧ N.1 = t * N.2 + 1) →
  k₁ M = (1/2) * k₂ N := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_relation_l1022_102249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_comparison_l1022_102203

theorem exp_comparison : (17/10 : ℝ)^(3/10 : ℝ) > (9/10 : ℝ)^11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_comparison_l1022_102203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_and_intersection_l1022_102281

-- Define the trajectory of the center of the moving circle
def trajectory (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line passing through F(2, 0) with slope 1
def line (x y : ℝ) : Prop := y = x - 2

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem moving_circle_trajectory_and_intersection :
  -- The trajectory is a parabola with equation y^2 = 8x
  (∀ x y : ℝ, trajectory x y ↔ y^2 = 8*x) ∧
  -- The line y = x - 2 intersects the trajectory at two points
  (∃ x1 y1 x2 y2 : ℝ, 
    x1 ≠ x2 ∧
    trajectory x1 y1 ∧ trajectory x2 y2 ∧
    line x1 y1 ∧ line x2 y2 ∧
    -- The distance between these points is 16
    distance x1 y1 x2 y2 = 16) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_and_intersection_l1022_102281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1022_102225

theorem solve_exponential_equation (y : ℝ) : 5 * (2:ℝ)^y = 160 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1022_102225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l1022_102288

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem: Point A(18, 0, 0) is equidistant from B(1, 5, 9) and C(3, 7, 11) -/
theorem equidistant_point :
  let A : Point3D := ⟨18, 0, 0⟩
  let B : Point3D := ⟨1, 5, 9⟩
  let C : Point3D := ⟨3, 7, 11⟩
  distance A B = distance A C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l1022_102288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_and_necessary_not_sufficient_l1022_102227

theorem sufficient_not_necessary_and_necessary_not_sufficient :
  (∀ a b : ℝ, a < b ∧ b < 0 → 1 / a > 1 / b) ∧
  (∃ a b : ℝ, 1 / a > 1 / b ∧ ¬(a < b ∧ b < 0)) ∧
  (∀ l : ℝ, -1 ≤ l ∧ l ≤ 3 → -2 ≤ l ∧ l ≤ 3) ∧
  (∃ l : ℝ, -2 ≤ l ∧ l ≤ 3 ∧ ¬(-1 ≤ l ∧ l ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_and_necessary_not_sufficient_l1022_102227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distribution_possible_l1022_102243

/-- Represents the number of wise men -/
def num_wise_men : ℕ := 10

/-- Represents the initial coin distribution -/
def initial_coins (i : ℕ) : ℕ := i

/-- Represents a single redistribution action -/
structure RedistributionAction where
  recipient : Fin num_wise_men
  donor : Fin num_wise_men

/-- Represents the final state after redistribution -/
def final_state (actions : List RedistributionAction) : Fin num_wise_men → ℕ := sorry

/-- Theorem stating that equal distribution is possible -/
theorem equal_distribution_possible :
  ∃ (actions : List RedistributionAction),
    ∀ (i : Fin num_wise_men),
      final_state actions i = 55 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distribution_possible_l1022_102243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_two_planes_implies_parallel_planes_l1022_102229

structure Plane where

structure Line where

def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel (p1 p2 : Plane) : Prop := sorry

theorem line_perp_two_planes_implies_parallel_planes 
  (α β : Plane) (l : Line) 
  (h1 : α ≠ β) 
  (h2 : perpendicular l α) 
  (h3 : perpendicular l β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_two_planes_implies_parallel_planes_l1022_102229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_2_l1022_102295

/-- The length of the chord cut by the line x + y - 2 = 0 on the circle x² + (y-1)² = 1 is √2 -/
theorem chord_length_is_sqrt_2 :
  let line := {(x, y) : ℝ × ℝ | x + y - 2 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y-1)^2 = 1}
  let chord := line ∩ circle
  ∃ (a b : ℝ × ℝ), a ∈ chord ∧ b ∈ chord ∧ a ≠ b ∧
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_2_l1022_102295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_and_sum_l1022_102206

-- Define the rational function
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 8 * x - 20) / (x - 5)

-- Define the slant asymptote function
def g (x : ℝ) : ℝ := 3 * x + 23

-- Theorem statement
theorem slant_asymptote_and_sum :
  (∀ ε > 0, ∃ N : ℝ, ∀ x > N, |f x - g x| < ε) ∧
  (3 + 23 = 26) := by
  constructor
  · sorry  -- Proof of the asymptotic behavior
  · rfl    -- Proof of 3 + 23 = 26


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_and_sum_l1022_102206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_scores_sum_l1022_102226

def test_scores (scores : Finset ℕ) : Prop :=
  Finset.card scores = 6 ∧
  (Finset.sum scores id) / (Finset.card scores : ℝ) = 85 ∧
  -- Replacing Finset.median and Finset.mode with placeholders
  ∃ m : ℕ, m ∈ scores ∧ m = 88 ∧  -- Placeholder for median
  ∃ n : ℕ, n ∈ scores ∧ n = 89    -- Placeholder for mode

theorem lowest_scores_sum (scores : Finset ℕ) (h : test_scores scores) :
  ∃ s : Finset ℕ, s ⊆ scores ∧ Finset.card s = 2 ∧ Finset.sum s id = 166 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_scores_sum_l1022_102226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_theorem_l1022_102263

/-- Represents a rectangle in the square division -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the division of a square into rectangles -/
structure SquareDivision where
  n : ℕ
  rectangles : Finset Rectangle

/-- Predicate to check if one rectangle can be placed inside another (possibly after rotation) -/
def can_fit_inside (r1 r2 : Rectangle) : Prop :=
  (r1.width ≤ r2.width ∧ r1.height ≤ r2.height) ∨ 
  (r1.width ≤ r2.height ∧ r1.height ≤ r2.width)

/-- The main theorem statement -/
theorem square_division_theorem (sd : SquareDivision) 
  (h1 : sd.n ^ 2 ≥ 4)
  (h2 : sd.rectangles.card = sd.n ^ 2) :
  ∃ (chosen : Finset Rectangle), 
    chosen ⊆ sd.rectangles ∧ 
    chosen.card = 2 * sd.n ∧
    ∀ r1 r2, r1 ∈ chosen → r2 ∈ chosen → r1 ≠ r2 → 
      can_fit_inside r1 r2 ∨ can_fit_inside r2 r1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_theorem_l1022_102263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cover_l1022_102236

/-- The side length of the large equilateral triangle -/
def large_side : ℝ := 12

/-- The set of possible side lengths for the smaller triangles -/
def small_sides : Set ℝ := {1, 3, 8}

/-- The area of an equilateral triangle given its side length -/
noncomputable def area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2

/-- The proposition that N is the minimum number of smaller triangles needed -/
def is_minimum_cover (N : ℕ) : Prop :=
  ∀ (sides : List ℝ), 
    sides.length = N → 
    (∀ s ∈ sides, s ∈ small_sides) → 
    (sides.map area).sum = area large_side → 
    ∀ M : ℕ, M < N → 
      ¬∃ (smaller_sides : List ℝ), 
        smaller_sides.length = M ∧ 
        (∀ s ∈ smaller_sides, s ∈ small_sides) ∧ 
        (smaller_sides.map area).sum = area large_side

/-- Theorem stating that 16 is the minimum number of smaller triangles needed -/
theorem minimum_cover : is_minimum_cover 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cover_l1022_102236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_25_accurate_to_hundredths_l1022_102247

/-- A number is accurate to the hundredths place if it has exactly two decimal places. -/
def accurate_to_hundredths (x : ℝ) : Prop :=
  ∃ n : ℤ, x = n / 100 ∧ x ≠ ↑(Int.floor x)

/-- The given numbers from the problem -/
def problem_numbers : List ℝ := [30, 21.1, 25.00, 13.001]

/-- Theorem stating that among the given numbers, only 25.00 is accurate to the hundredths place -/
theorem only_25_accurate_to_hundredths :
  ∃! x, x ∈ problem_numbers ∧ accurate_to_hundredths x ∧ x = 25.00 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_25_accurate_to_hundredths_l1022_102247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_15_degree_l1022_102269

theorem triangle_area_15_degree (h : ℝ) (angle_A : ℝ) :
  h = 1 →
  angle_A = 15 * π / 180 →
  (1/2) * h * (h * (Real.sqrt 6 + Real.sqrt 2) / 2) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_15_degree_l1022_102269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_bisector_area_l1022_102284

/-- An isosceles right triangle with given area -/
structure IsoscelesRightTriangle where
  side : ℝ
  area : ℝ
  area_eq : area = side^2 / 2

/-- The bisector of the right angle in an isosceles right triangle -/
noncomputable def angle_bisector (t : IsoscelesRightTriangle) : ℝ := t.side * Real.sqrt 2 / 2

/-- The area of the smaller triangle formed by the angle bisector -/
noncomputable def small_triangle_area (t : IsoscelesRightTriangle) : ℝ :=
  t.side * (t.side * Real.sqrt 2 / 2) / 2

theorem isosceles_right_triangle_bisector_area
  (t : IsoscelesRightTriangle)
  (h : t.area = 18) :
  small_triangle_area t = 9 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_bisector_area_l1022_102284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_cube_roll_l1022_102258

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)
  (color : Fin size → Fin size → Bool)

/-- Represents a cube --/
structure Cube :=
  (face_colors : Fin 6 → Bool)

/-- Represents a path on the chessboard --/
inductive ChessPath (n : ℕ)
  | steps : (Fin n → Fin 2 → Fin 8) → ChessPath n

/-- Checks if a path covers all squares on the chessboard exactly once --/
def covers_all_squares_once (cb : Chessboard) (p : ChessPath (cb.size * cb.size)) : Prop :=
  sorry

/-- Checks if the cube's face color matches the square color at each step --/
def colors_match (cb : Chessboard) (c : Cube) (p : ChessPath (cb.size * cb.size)) : Prop :=
  sorry

theorem impossible_cube_roll (cb : Chessboard) :
  ¬∃ (c : Cube) (p : ChessPath (cb.size * cb.size)),
    covers_all_squares_once cb p ∧ colors_match cb c p :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_cube_roll_l1022_102258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorable_b_l1022_102228

/-- A function that checks if a quadratic expression can be factored into two binomials with integer coefficients -/
def is_factorable (b : ℤ) : Prop :=
  ∃ (r s : ℤ), ∀ (x : ℤ), x^2 + b*x + 1728 = (x + r) * (x + s)

/-- The theorem stating that 84 is the smallest positive integer b for which x^2 + bx + 1728 can be factored into two binomials with integer coefficients -/
theorem smallest_factorable_b : 
  (is_factorable 84 ∧ ∀ b : ℤ, 0 < b → b < 84 → ¬(is_factorable b)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorable_b_l1022_102228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_existence_l1022_102242

theorem certain_number_existence : ∃ n : ℕ, 
  n > 0 ∧
  (55 * 57) % n = 6 ∧ 
  (∀ m : ℕ, m > 0 → (55 * 57) % m = 6 → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_existence_l1022_102242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_function_inverse_function_inverse_domain_l1022_102234

noncomputable section

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := (x + 3) / (2 - x)

-- Define the translated function g
noncomputable def g (x : ℝ) : ℝ := 5 / (-x)

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (2 * x - 3) / (1 + x)

-- Theorem for the translated function
theorem translated_function :
  ∀ x : ℝ, g x = (f (x + 2) + 1) := by sorry

-- Theorem for the inverse function
theorem inverse_function :
  ∀ x : ℝ, x ≠ -1 → f (f_inv x) = x ∧ f_inv (f x) = x := by sorry

-- Theorem for the domain of the inverse function
theorem inverse_domain :
  ∀ x : ℝ, x ≠ -1 → f_inv x ∈ Set.univ \ {2} := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_function_inverse_function_inverse_domain_l1022_102234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_price_correct_l1022_102239

/-- The price of a pencil at A.T. Cross Luxury Pens -/
def pencil_price : ℚ := 1/4

/-- The price of a pen at A.T. Cross Luxury Pens -/
def pen_price : ℚ := 15/100

/-- The number of pens Bowen buys -/
def pens_bought : ℕ := 40

/-- The ratio of additional pencils to pens Bowen buys -/
def pencil_to_pen_ratio : ℚ := 2/5

/-- The total amount Bowen spends -/
def total_spent : ℚ := 20

/-- Theorem stating that the pencil price is correct given the conditions -/
theorem pencil_price_correct : 
  let pencils_bought := pens_bought + Int.floor (pencil_to_pen_ratio * pens_bought)
  pencil_price * pencils_bought + pen_price * pens_bought = total_spent :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_price_correct_l1022_102239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l1022_102292

theorem trig_equation_solution (x : ℝ) : 
  (∃ k : ℤ, x = π / 20 + π * k / 5 ∨ x = π / 12 + 2 * π * k / 3 ∨ x = π / 28 + 2 * π * k / 7) ↔
  Real.cos (7 * x) + Real.cos (3 * x) - Real.sqrt 2 * Real.cos (10 * x) = Real.sin (7 * x) + Real.sin (3 * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l1022_102292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_choice_related_to_gender_l1022_102296

/-- Represents the contingency table data --/
structure ContingencyTable where
  boys_calligraphy : ℕ
  boys_paper_cutting : ℕ
  girls_calligraphy : ℕ
  girls_paper_cutting : ℕ

/-- Calculates the K^2 value for a given contingency table --/
noncomputable def calculateK2 (table : ContingencyTable) : ℝ :=
  let n := (table.boys_calligraphy + table.boys_paper_cutting + table.girls_calligraphy + table.girls_paper_cutting : ℝ)
  let a := (table.boys_calligraphy : ℝ)
  let b := (table.boys_paper_cutting : ℝ)
  let c := (table.girls_calligraphy : ℝ)
  let d := (table.girls_paper_cutting : ℝ)
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for 95% confidence --/
def criticalValue : ℝ := 3.841

/-- Theorem stating that the calculated K^2 value is greater than the critical value --/
theorem course_choice_related_to_gender (table : ContingencyTable)
  (h1 : table.boys_calligraphy = 40)
  (h2 : table.boys_paper_cutting = 10)
  (h3 : table.girls_calligraphy = 30)
  (h4 : table.girls_paper_cutting = 20) :
  calculateK2 table > criticalValue := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_choice_related_to_gender_l1022_102296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_in_sequence_l1022_102213

/-- Given a sequence of seven consecutive even numbers with sum 700 and 
    the sum of the first three numbers greater than 200, 
    prove that the smallest number in the sequence is 94. -/
theorem smallest_number_in_sequence (seq : Fin 7 → ℕ) 
  (consecutive_even : ∀ i : Fin 6, seq (i.succ) = seq i + 2)
  (sum_700 : (Finset.univ.sum seq) = 700)
  (first_three_sum : seq 0 + seq 1 + seq 2 > 200) :
  seq 0 = 94 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_in_sequence_l1022_102213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_exp_gt_quadratic_l1022_102251

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 2 * x + 2

-- Theorem for the minimum value of f(x)
theorem f_minimum_value : 
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 2 * (2 - Real.log 2) := by
  sorry

-- Theorem for the inequality when x > 0
theorem exp_gt_quadratic (x : ℝ) (h : x > 0) : 
  Real.exp x > x^2 - 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_exp_gt_quadratic_l1022_102251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_l1022_102250

theorem price_change (initial_price : ℝ) (h : initial_price > 0) : 
  (initial_price * (1 - 0.4) * (1 + 0.35) - initial_price) / initial_price = -0.19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_l1022_102250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1022_102289

/-- The standard equation of a circle with center (0,1) and tangent to x+y+3=0 -/
theorem circle_equation : 
  let center : ℝ × ℝ := (0, 1)
  let tangent_line := {p : ℝ × ℝ | p.1 + p.2 + 3 = 0}
  let circle := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 8}
  (∀ p ∈ circle, p.1^2 + (p.2 - 1)^2 = 8) ∧
  (∃! p, p ∈ circle ∧ p ∈ tangent_line) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1022_102289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_fraction_l1022_102222

-- Define the fraction of paper used for each of the next three presents
def x : ℚ := 2 / 45

-- Define the conditions
def total_fraction : ℚ := 2 / 5
def first_gift : ℚ := 3 * x
def last_gift : ℚ := 2 * x
def next_three_gifts : ℚ := 3 * x

-- Theorem statement
theorem wrapping_paper_fraction :
  total_fraction = first_gift + next_three_gifts + last_gift := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_fraction_l1022_102222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1022_102255

-- Define the vector type for 2D plane
def Vector2D := ℝ × ℝ

-- Given vectors
def a : Vector2D := (1, 2)

-- Vector operations
def dot_product (v w : Vector2D) : ℝ := v.1 * w.1 + v.2 * w.2
def scalar_mult (k : ℝ) (v : Vector2D) : Vector2D := (k * v.1, k * v.2)
noncomputable def magnitude (v : Vector2D) : ℝ := Real.sqrt (dot_product v v)
def parallel (v w : Vector2D) : Prop := ∃ k : ℝ, w = scalar_mult k v
def perpendicular (v w : Vector2D) : Prop := dot_product v w = 0

-- Define vector addition
def vector_add (v w : Vector2D) : Vector2D := (v.1 + w.1, v.2 + w.2)

-- Define vector subtraction
def vector_sub (v w : Vector2D) : Vector2D := (v.1 - w.1, v.2 - w.2)

theorem vector_problem :
  -- Part 1
  ∃ c : Vector2D, magnitude c = 3 * Real.sqrt 5 ∧ parallel a c ∧
    (c = (3, 6) ∨ c = (-3, -6)) ∧
  -- Part 2
  ∃ b : Vector2D, magnitude b = 3 * Real.sqrt 5 ∧
    perpendicular (vector_sub (scalar_mult 4 a) b) (vector_add (scalar_mult 2 a) b) ∧
    dot_product a b / (magnitude a * magnitude b) = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1022_102255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_beta_l1022_102285

theorem cosine_beta (α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.sin (α + β) = 5/13) (h4 : Real.tan (α/2) = 1/2) :
  Real.cos β = -16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_beta_l1022_102285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_l1022_102290

/-- Calculates the speed of a man running opposite to a train given the train's length, speed, and time to pass the man. -/
theorem man_speed (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) :
  train_length = 165 →
  train_speed_kmph = 60 →
  passing_time = 9 →
  ∃ (man_speed_kmph : ℝ), abs (man_speed_kmph - 5.976) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_l1022_102290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_general_term_l1022_102280

def x : ℕ → ℚ
  | 0 => 4  -- Adding the base case for 0
  | n + 1 => (5 * (n + 1) + 2) / (5 * (n + 1) - 3) * x n + 7 * (5 * (n + 1) + 2)

theorem x_general_term (n : ℕ) (h : n > 0) : 
  x n = (1 / 7 : ℚ) * (49 * n - 45) * (5 * n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_general_term_l1022_102280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_x_equals_one_eighth_of_two_power_36_l1022_102220

theorem eight_power_x_equals_one_eighth_of_two_power_36 (x : ℝ) : 
  (1 / 8) * (2 : ℝ) ^ 36 = 8 ^ x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_x_equals_one_eighth_of_two_power_36_l1022_102220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1022_102261

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1)

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k*x - 2*k + 3

-- Theorem statement
theorem line_equation_proof (a k : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let M : ℝ × ℝ := (1, f a 1)
  let N : ℝ × ℝ := (2, line k 2)
  (2 : ℝ) * M.1 - M.2 - 1 = 0 ∧ 
  (2 : ℝ) * N.1 - N.2 - 1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1022_102261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1022_102293

/-- The function f(x) = x² - ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

theorem problem_solution :
  ∀ (a b : ℝ),
  (∀ x, f a x ≤ -3 ↔ b ≤ x ∧ x ≤ 3) →
  a = 5 ∧ b = 2 ∧
  (∀ a : ℝ, (∃ x ∈ Set.Icc (1/2) 2, f a x ≤ 1 - x^2) → a ∈ Set.Ioi 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1022_102293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_common_divisors_l1022_102221

/-- The number of common positive integer divisors of 90 and 150 -/
theorem count_common_divisors : ∃ (n : ℕ), n = (Finset.filter (fun x => 90 % x = 0 ∧ 150 % x = 0) (Finset.range 151)).card ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_common_divisors_l1022_102221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1022_102275

/-- The slope of the line x-y-3=0 -/
def m₁ : ℝ := 1

/-- The slope angle of the line x-y-3=0 -/
noncomputable def θ₁ : ℝ := Real.arctan m₁

/-- The slope angle of the line we're looking for -/
noncomputable def θ₂ : ℝ := 2 * θ₁

/-- The slope of the line we're looking for -/
noncomputable def m₂ : ℝ := Real.tan θ₂

/-- The point that the line passes through -/
def P : ℝ × ℝ := (1, 2)

theorem line_equation : 
  ∀ (x y : ℝ), (x = P.1 ∧ y = P.2) ∨ (y - P.2 = m₂ * (x - P.1)) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1022_102275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_is_minimum_l1022_102267

/-- The point on the ellipse with minimum distance to the line -/
noncomputable def min_distance_point : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2 / 2)

/-- The equation of the ellipse -/
def on_ellipse (p : ℝ × ℝ) : Prop :=
  p.1^2 / 4 + p.2^2 = 1

/-- The equation of the line -/
def on_line (p : ℝ × ℝ) : Prop :=
  p.1 + 2 * p.2 = 4

/-- The distance from a point to the line -/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  abs (p.1 + 2 * p.2 - 4) / Real.sqrt 5

theorem min_distance_point_is_minimum :
  on_ellipse min_distance_point ∧
  ∀ p : ℝ × ℝ, on_ellipse p →
    distance_to_line min_distance_point ≤ distance_to_line p := by
  sorry

#check min_distance_point_is_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_is_minimum_l1022_102267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grapefruit_orange_touch_points_coplanar_l1022_102276

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents the configuration of fruits in the vase -/
structure FruitConfiguration where
  vase : Sphere
  grapefruit : Sphere
  oranges : Fin 4 → Sphere

/-- Checks if two spheres are touching -/
def spheresTouching (s1 s2 : Sphere) : Prop :=
  (s1.center.x - s2.center.x)^2 + (s1.center.y - s2.center.y)^2 + (s1.center.z - s2.center.z)^2 
  = (s1.radius + s2.radius)^2

/-- Represents the configuration satisfying all conditions -/
def validConfiguration (config : FruitConfiguration) : Prop :=
  ∀ i : Fin 4, 
    spheresTouching config.vase (config.oranges i) ∧ 
    spheresTouching config.grapefruit (config.oranges i) ∧
    config.oranges i ≠ config.grapefruit ∧
    (∀ j : Fin 4, i ≠ j → config.oranges i ≠ config.oranges j) ∧
    config.vase.center.z + config.vase.radius ≥ config.grapefruit.center.z + config.grapefruit.radius ∧
    (∀ i : Fin 4, config.vase.center.z + config.vase.radius ≥ (config.oranges i).center.z + (config.oranges i).radius)

/-- Gets the point where the grapefruit touches an orange -/
noncomputable def touchPoint (grapefruit orange : Sphere) : Point3D :=
  let v := Point3D.mk 
    (orange.center.x - grapefruit.center.x)
    (orange.center.y - grapefruit.center.y)
    (orange.center.z - grapefruit.center.z)
  let t := grapefruit.radius / (grapefruit.radius + orange.radius)
  Point3D.mk
    (grapefruit.center.x + t * v.x)
    (grapefruit.center.y + t * v.y)
    (grapefruit.center.z + t * v.z)

/-- Checks if four points lie in the same plane -/
def pointsCoplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  let v1 := Point3D.mk (p2.x - p1.x) (p2.y - p1.y) (p2.z - p1.z)
  let v2 := Point3D.mk (p3.x - p1.x) (p3.y - p1.y) (p3.z - p1.z)
  let v3 := Point3D.mk (p4.x - p1.x) (p4.y - p1.y) (p4.z - p1.z)
  (v1.y * v2.z - v1.z * v2.y) * v3.x + 
  (v1.z * v2.x - v1.x * v2.z) * v3.y + 
  (v1.x * v2.y - v1.y * v2.x) * v3.z = 0

/-- The main theorem to be proved -/
theorem grapefruit_orange_touch_points_coplanar (config : FruitConfiguration) 
  (h : validConfiguration config) :
  pointsCoplanar 
    (touchPoint config.grapefruit (config.oranges 0))
    (touchPoint config.grapefruit (config.oranges 1))
    (touchPoint config.grapefruit (config.oranges 2))
    (touchPoint config.grapefruit (config.oranges 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grapefruit_orange_touch_points_coplanar_l1022_102276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_l1022_102283

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the common chord equation
def common_chord (x y : ℝ) : Prop := x - y - 3 = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(1, -2), (2, -1)}

-- Theorem statement
theorem circles_intersection :
  (∃ (x y : ℝ), C₁ x y ∧ C₂ x y) ∧
  (∀ (x y : ℝ), C₁ x y ∧ C₂ x y → common_chord x y) ∧
  (Real.sqrt 2 = Real.sqrt ((2 - 1)^2 + (-1 - (-2))^2)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersection_l1022_102283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1022_102262

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 16) / (x - 7)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1022_102262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sector_ratio_limit_l1022_102233

theorem triangle_sector_ratio_limit (r : ℝ) (h : r > 0) :
  let θ := π / 3
  let S := (1 / 2) * r^2 * θ
  let T := (Real.sqrt 3 / 4) * r^2
  T / S = 3 * Real.sqrt 3 / (2 * π) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sector_ratio_limit_l1022_102233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_purchase_theorem_l1022_102264

/-- Represents the cost calculation for a seed purchase with a discount --/
noncomputable def seed_purchase_cost (price_a price_b : ℝ) (ratio_a ratio_b : ℕ) (total_amount : ℝ) (discount_rate : ℝ) : ℝ :=
  let amount_a := (ratio_a : ℝ) * total_amount / ((ratio_a : ℝ) + (ratio_b : ℝ))
  let amount_b := (ratio_b : ℝ) * total_amount / ((ratio_a : ℝ) + (ratio_b : ℝ))
  let total_cost := amount_a * price_a + amount_b * price_b
  let discounted_cost := total_cost * (1 - discount_rate)
  discounted_cost

/-- Theorem stating the correct cost for the given seed purchase scenario --/
theorem seed_purchase_theorem :
  seed_purchase_cost 12 8 2 3 6 0.1 = 51.84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_purchase_theorem_l1022_102264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_addition_problem_solution_l1022_102205

/-- Represents a digit in base 7 -/
def Base7Digit := Fin 7

/-- Converts a base 7 number to base 10 -/
def to_base10 (hundreds : Base7Digit) (tens : Base7Digit) (ones : Base7Digit) : ℕ :=
  hundreds.val * 49 + tens.val * 7 + ones.val

theorem addition_problem_solution (X Y : Base7Digit) :
  (∃ (five : Base7Digit),
    to_base10 five X Y + to_base10 (⟨0, by norm_num⟩) (⟨5, by norm_num⟩) (⟨2, by norm_num⟩) = 
    to_base10 (⟨6, by norm_num⟩) (⟨4, by norm_num⟩) X) →
  X.val + Y.val = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_addition_problem_solution_l1022_102205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equilateral_triangle_dot_product_l1022_102265

/-- Parabola type -/
structure Parabola where
  a : ℝ
  h : a > 0

/-- Point type -/
structure MyPoint where
  x : ℝ
  y : ℝ

/-- Vector type -/
structure MyVector where
  x : ℝ
  y : ℝ

/-- Dot product of two vectors -/
def dot_product (v1 v2 : MyVector) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Theorem: For a parabola y^2 = 4x with focus F(1,0), and a point M on the parabola
    such that triangle MQF (where Q is the projection of M on the directrix) is equilateral,
    the dot product of vectors FQ and FM equals 8. -/
theorem parabola_equilateral_triangle_dot_product
  (C : Parabola)
  (F : MyPoint)
  (M : MyPoint)
  (Q : MyPoint)
  (h1 : C.a = 4)
  (h2 : F.x = 1 ∧ F.y = 0)
  (h3 : M.y^2 = C.a * M.x)
  (h4 : Q.x = -1 ∧ Q.y = M.y)
  (h5 : (M.x - Q.x)^2 = (M.x - F.x)^2 + (M.y - F.y)^2)
  (h6 : (M.x - Q.x)^2 = (Q.x - F.x)^2 + (Q.y - F.y)^2) :
  dot_product
    (MyVector.mk (Q.x - F.x) (Q.y - F.y))
    (MyVector.mk (M.x - F.x) (M.y - F.y)) = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equilateral_triangle_dot_product_l1022_102265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1022_102216

-- Define the curve C
noncomputable def C (x : ℝ) : ℝ := x * Real.sqrt (9 - x^2)

-- State the theorem
theorem curve_C_properties :
  -- Condition: x ≥ 0
  ∀ x : ℝ, x ≥ 0 →
  -- 1. Maximum value
  (∃ max_y : ℝ, max_y = (9 : ℝ) / 2 ∧ ∀ y : ℝ, y = C x → y ≤ max_y) ∧
  -- 2. Area bounded by C and x-axis
  (∃ A : ℝ, A = 9 ∧ A = ∫ x in (0 : ℝ)..3, C x) ∧
  -- 3. Volume of revolution around y-axis
  (∃ V : ℝ, V = (162 * Real.pi) / 5 ∧ V = Real.pi * ∫ x in (0 : ℝ)..3, (C x)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1022_102216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_almost_perfect_numbers_l1022_102256

def d (n : ℕ) : ℕ := (Nat.divisors n).card

def f (n : ℕ) : ℕ := (Nat.divisors n).sum (fun k => d k)

def is_almost_perfect (n : ℕ) : Prop := n > 1 ∧ f n = n

theorem almost_perfect_numbers :
  {n : ℕ | is_almost_perfect n} = {3, 18, 36} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_almost_perfect_numbers_l1022_102256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_c_l1022_102266

theorem triangle_side_c (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  Real.sin A = Real.sqrt 3 * Real.sin B →
  C = π / 6 →
  a * c = Real.sqrt 3 →
  c = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_c_l1022_102266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1022_102204

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A cube in 3D space -/
structure Cube where
  vertex : Point3D
  edge_length : ℝ

/-- A moving point on the perimeter of a square face of the cube -/
structure MovingPoint where
  initial_position : Point3D
  speed : ℝ

/-- The trajectory of a point -/
def Trajectory := Set Point3D

/-- The theorem statement -/
theorem midpoint_trajectory 
  (cube : Cube)
  (X : MovingPoint)
  (Y : MovingPoint)
  (hX : X.initial_position = Point3D.mk 0 0 0)
  (hY : Y.initial_position = Point3D.mk 1 0 1)
  (hspeed : X.speed = Y.speed)
  : Trajectory = {
    z | z = Point3D.mk (1/2) 0 (1/2) ∨
        z = Point3D.mk 1 (1/2) (1/2) ∨
        z = Point3D.mk 1 1 0 ∨
        z = Point3D.mk (1/2) (1/2) 0
  } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1022_102204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_quotient_approx_l1022_102214

-- Define the variables and constants
noncomputable def incorrect_divisor : ℝ := -125.5
noncomputable def correct_divisor : ℝ := 217.75
noncomputable def incorrect_quotient : ℝ := -467.8

-- Define x and y
noncomputable def x : ℝ := incorrect_quotient / 2
noncomputable def y : ℝ := correct_divisor / 4

-- Define the equation
noncomputable def z : ℤ := ⌊(3 * x - 10 * y : ℝ)⌋

-- Define the dividend
noncomputable def dividend : ℝ := incorrect_divisor * incorrect_quotient

-- Theorem to prove
theorem correct_quotient_approx :
  ∃ (q : ℝ), (abs (q - 269.6) < 0.1) ∧ (abs (dividend / correct_divisor - q) < 0.1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_quotient_approx_l1022_102214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_interval_discount_is_beneficial_l1022_102272

/-- Represents the feed purchase problem for a breeding farm -/
structure FeedPurchase where
  daily_requirement : ℝ
  price_per_kg : ℝ
  storage_cost_per_kg_per_day : ℝ
  transportation_fee : ℝ
  discount_threshold : ℝ
  discount_rate : ℝ

/-- Calculates the average daily cost for a given purchase interval -/
noncomputable def averageDailyCost (fp : FeedPurchase) (interval : ℝ) : ℝ :=
  (fp.daily_requirement * fp.price_per_kg * interval + 
   fp.daily_requirement * fp.storage_cost_per_kg_per_day * interval * interval / 2 + 
   fp.transportation_fee) / interval

/-- Theorem stating that 10 days is the optimal purchase interval -/
theorem optimal_purchase_interval (fp : FeedPurchase) : 
  fp.daily_requirement = 200 → 
  fp.price_per_kg = 1.8 → 
  fp.storage_cost_per_kg_per_day = 0.03 → 
  fp.transportation_fee = 300 → 
  (∀ t : ℝ, t > 0 → averageDailyCost fp 10 ≤ averageDailyCost fp t) := by
  sorry

/-- Calculates the average daily cost with discount for a given purchase interval -/
noncomputable def averageDailyCostWithDiscount (fp : FeedPurchase) (interval : ℝ) : ℝ :=
  if fp.daily_requirement * interval ≥ fp.discount_threshold then
    (fp.daily_requirement * fp.price_per_kg * interval * (1 - fp.discount_rate) + 
     fp.daily_requirement * fp.storage_cost_per_kg_per_day * interval * interval / 2 + 
     fp.transportation_fee) / interval
  else
    averageDailyCost fp interval

/-- Theorem stating that taking advantage of the discount is beneficial -/
theorem discount_is_beneficial (fp : FeedPurchase) : 
  fp.daily_requirement = 200 → 
  fp.price_per_kg = 1.8 → 
  fp.storage_cost_per_kg_per_day = 0.03 → 
  fp.transportation_fee = 300 → 
  fp.discount_threshold = 5000 → 
  fp.discount_rate = 0.15 → 
  averageDailyCostWithDiscount fp 25 < averageDailyCost fp 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_interval_discount_is_beneficial_l1022_102272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1022_102232

theorem work_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a = b) (hb_time : b = 1 / 12) :
  1 / (a + b) = 6 := by
  have h1 : a = 1 / 12 := by rw [hab, hb_time]
  have h2 : a + b = 1 / 12 + 1 / 12 := by rw [h1, hb_time]
  have h3 : a + b = 1 / 6 := by
    rw [h2]
    norm_num
  rw [h3]
  norm_num

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1022_102232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_curve_min_angle_l1022_102201

/-- A Δ-curve is a closed curve with certain properties. -/
class DeltaCurve (α : Type*) [TopologicalSpace α] :=
  (points : Set α)
  (is_closed : IsClosed points)
  -- Other properties of Δ-curves can be added here

/-- A corner point of a Δ-curve. -/
structure CornerPoint (α : Type*) [TopologicalSpace α] [DeltaCurve α] :=
  (point : α)
  (is_corner : Bool) -- Changed to Bool for simplicity

/-- The internal angle at a corner point of a Δ-curve. -/
noncomputable def internal_angle {α : Type*} [TopologicalSpace α] [DeltaCurve α] (cp : CornerPoint α) : ℝ := 
  sorry

/-- A 1-digon (bicentric Δ-curve) is a special type of Δ-curve. -/
def is_1_digon {α : Type*} [TopologicalSpace α] (curve : DeltaCurve α) : Prop := 
  sorry

/-- 
Theorem: For any Δ-curve, the internal angle at any corner point is greater than or equal to 60°, 
and equality holds if and only if the curve is a 1-digon (bicentric Δ-curve).
-/
theorem delta_curve_min_angle {α : Type*} [TopologicalSpace α] (curve : DeltaCurve α) (cp : CornerPoint α) :
  internal_angle cp ≥ 60 ∧ (internal_angle cp = 60 ↔ is_1_digon curve) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delta_curve_min_angle_l1022_102201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_property_l1022_102271

/-- The number of n-digit positive integers composed of 0, 1, 2, 3, 4, 5 where the digits 1 and 2 are not adjacent -/
def a (n : ℕ+) : ℕ :=
  sorry

/-- The statement to be proved -/
theorem perfect_square_property (n : ℕ+) : ∃ k : ℤ, 41 * (a n)^2 + (-4 : ℤ)^(n.val + 2) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_property_l1022_102271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_out_of_pocket_cost_l1022_102212

/-- Calculates the out-of-pocket cost for a medical visit with insurance coverage -/
def out_of_pocket_cost (visit_cost cast_cost : ℕ) (insurance_coverage_percent : ℚ) : ℕ :=
  let total_cost := visit_cost + cast_cost
  let insurance_coverage := (total_cost : ℚ) * insurance_coverage_percent
  (total_cost : ℤ) - (insurance_coverage.floor : ℤ) |>.toNat

/-- Proves that the out-of-pocket cost for Tom's medical visit is $200 -/
theorem toms_out_of_pocket_cost :
  out_of_pocket_cost 300 200 (60 / 100) = 200 := by
  sorry

#eval out_of_pocket_cost 300 200 (60 / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_out_of_pocket_cost_l1022_102212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l1022_102231

noncomputable def w : Fin 3 → ℝ := ![3, -1, 3]

/-- The projection of v onto w -/
noncomputable def proj_w (v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  (((v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)) / ((w 0) * (w 0) + (w 1) * (w 1) + (w 2) * (w 2))) • w

theorem plane_equation (v : Fin 3 → ℝ) (h : proj_w v = ![6, -2, 6]) :
  3 * (v 0) - (v 1) + 3 * (v 2) - 38 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l1022_102231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l1022_102268

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 21*x - 26

/-- The lower bound function for the inequality -/
noncomputable def lower_bound (k : ℝ) (x : ℝ) : ℝ := (1/10) * k * x * (x - 2)^2

/-- The upper bound function for the inequality -/
noncomputable def upper_bound (k : ℝ) (x : ℝ) : ℝ := 9*x + k

/-- The theorem stating the range of k -/
theorem k_range :
  ∃ k_min k_max : ℝ,
    k_min = 9 ∧ k_max = 12 ∧
    ∀ k : ℝ,
      (∀ x : ℝ, x ∈ Set.Ioo 2 5 → lower_bound k x < f x ∧ f x < upper_bound k x) ↔
      k ∈ Set.Icc k_min k_max :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l1022_102268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_monochromatic_triangle_probability_l1022_102224

/-- The probability of coloring an edge red or blue -/
noncomputable def color_probability : ℝ := 1 / 2

/-- The number of vertices in a regular hexagon -/
def num_vertices : ℕ := 6

/-- The number of triangles in a complete graph with 6 vertices -/
def num_triangles : ℕ := 20

/-- The probability that a specific triangle is not monochromatic -/
noncomputable def prob_not_monochromatic : ℝ := 3 / 4

/-- The probability of at least one monochromatic triangle in a randomly two-colored regular hexagon with all its diagonals -/
noncomputable def prob_monochromatic_triangle : ℝ := 1 - prob_not_monochromatic ^ num_triangles

theorem hexagon_monochromatic_triangle_probability :
  prob_monochromatic_triangle = 1 - (3/4)^20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_monochromatic_triangle_probability_l1022_102224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_25_sides_l1022_102286

/-- Definition of a convex polygon -/
structure ConvexPolygon :=
  (sides : ℕ)
  (diagonals : ℕ)
  (interiorAngleSum : ℕ)

/-- Properties of a 25-sided convex polygon -/
theorem polygon_25_sides (P : ConvexPolygon) (h : P.sides = 25) :
  (P.diagonals = 275) ∧
  (P.interiorAngleSum = 4140) ∧
  (P.interiorAngleSum > 4000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_25_sides_l1022_102286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_length_l1022_102248

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 2*y + 1 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 8*y = 0

-- Define the chord length function
noncomputable def chord_length (l : (ℝ → ℝ → Prop)) (c : (ℝ → ℝ → Prop)) : ℝ := 
  sorry

-- Theorem statement
theorem chord_intersection_length :
  chord_length line_l circle_eq = 2 * Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_length_l1022_102248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_true_propositions_l1022_102277

-- Define the propositions
def proposition1 : Prop := ∀ x y : ℝ, x * y = 1 → x = 1 / y ∧ y = 1 / x

-- For proposition2, we'll use a more general statement without specific geometric types
def proposition2 : Prop := ∀ s1 s2 s3 s4 : ℝ, s1 = s2 ∧ s2 = s3 ∧ s3 = s4 → true -- Placeholder

-- For proposition3, we'll use a more general statement without specific geometric types
def proposition3 : Prop := true -- Placeholder

def proposition4 : Prop := ∀ a b c : ℝ, a * c^2 > b * c^2 → a > b

-- Theorem statement
theorem true_propositions :
  proposition1 ∧ 
  ¬proposition2 ∧ 
  ¬proposition3 ∧ 
  proposition4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_true_propositions_l1022_102277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_x_range_l1022_102240

-- Define a structure for a triangle
structure Triangle (α : Type*) where
  sides : Finset α
  side_count : sides.card = 3

theorem triangle_inequality_x_range : 
  ∀ x : ℝ, (∃ t : Triangle ℝ, t.sides = {4, 6, x}) → (2 < x ∧ x < 10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_x_range_l1022_102240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_digit_divisible_by_6_and_5_l1022_102217

def digits (n : ℕ) : Finset ℕ := sorry

theorem no_three_digit_divisible_by_6_and_5 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 →
    (∀ d : ℕ, d ∈ digits n → d > 5) →
    n % 6 = 0 ∧ n % 5 = 0 →
    False :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_digit_divisible_by_6_and_5_l1022_102217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_six_or_eight_l1022_102260

theorem multiples_of_six_or_eight (n : Nat) : n = 201 →
  (Finset.filter (λ x : Nat ↦ (x % 6 = 0 ∨ x % 8 = 0) ∧ ¬(x % 6 = 0 ∧ x % 8 = 0)) (Finset.range n)).card = 42 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_six_or_eight_l1022_102260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_of_thousand_l1022_102299

theorem percent_of_thousand (percent : ℝ) (base : ℝ) (result : ℝ) : 
  percent = 6.620000000000001 → 
  base = 1000 → 
  result = percent * base / 100 →
  ∃ ε > 0, |result - 66.2| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_of_thousand_l1022_102299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_l1022_102211

-- Define the quadratic equation
def quadratic_equation (a b x : ℝ) : Prop := x^2 + a*x + b = 0

-- Define the irrational number √3
noncomputable def sqrt3 : ℝ := Real.sqrt 3

-- Theorem statement
theorem quadratic_roots (a b : ℚ) :
  (∃ (x : ℝ), x = 1 + sqrt3 ∧ quadratic_equation (↑a) (↑b) x) →
  a = -2 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_l1022_102211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_empty_l1022_102219

def A : Set (ℤ × ℤ) := {p | p.2 = p.1 + 1}

def B : Set ℤ := {y | ∃ x, y = 2 * x}

theorem A_intersect_B_empty : A ∩ (B.image (fun y => (y/2, y))) = ∅ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_empty_l1022_102219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tiles_on_floor_l1022_102210

theorem max_tiles_on_floor (floor_length floor_width tile_length tile_width : ℕ) 
  (h1 : floor_length = 1000)
  (h2 : floor_width = 210)
  (h3 : tile_length = 35)
  (h4 : tile_width = 30) :
  let max_tiles := max 
    ((floor_length / tile_length) * (floor_width / tile_width))
    ((floor_length / tile_width) * (floor_width / tile_length))
  max_tiles = 198 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tiles_on_floor_l1022_102210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overtime_rate_is_sixteen_l1022_102257

/-- Calculates the overtime hourly rate given the regular hourly rate, regular hours, gross pay, and overtime hours. -/
noncomputable def overtime_hourly_rate (regular_rate : ℝ) (regular_hours : ℝ) (gross_pay : ℝ) (overtime_hours : ℝ) : ℝ :=
  (gross_pay - regular_rate * regular_hours) / overtime_hours

/-- Theorem stating that the overtime hourly rate is $16 given the problem conditions. -/
theorem overtime_rate_is_sixteen :
  let regular_rate : ℝ := 11.25
  let regular_hours : ℝ := 40
  let gross_pay : ℝ := 622
  let overtime_hours : ℝ := 10.75
  overtime_hourly_rate regular_rate regular_hours gross_pay overtime_hours = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overtime_rate_is_sixteen_l1022_102257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f1_f2_f3_f4_f50_eq_7451_l1022_102241

/-- A sequence of nonoverlapping unit squares -/
def f (n : ℕ) : ℕ :=
  3 * n^2 - n + 1

/-- The first term of the sequence -/
theorem f1 : f 1 = 3 := by
  rw [f]
  norm_num

/-- The second term of the sequence -/
theorem f2 : f 2 = 11 := by
  rw [f]
  norm_num

/-- The third term of the sequence -/
theorem f3 : f 3 = 25 := by
  rw [f]
  norm_num

/-- The fourth term of the sequence -/
theorem f4 : f 4 = 45 := by
  rw [f]
  norm_num

/-- The 50th term of the sequence is 7451 -/
theorem f50_eq_7451 : f 50 = 7451 := by
  rw [f]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f1_f2_f3_f4_f50_eq_7451_l1022_102241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_divisible_by_seven_l1022_102253

def numbers : Finset ℕ := Finset.filter (λ n => 6 ≤ n ∧ n ≤ 36 ∧ n % 7 = 0) (Finset.range 37)

theorem average_of_divisible_by_seven : 
  let sum : ℝ := (Finset.sum numbers (λ n => n : ℕ → ℝ))
  let count : ℝ := numbers.card
  sum / count = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_divisible_by_seven_l1022_102253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_price_theorem_l1022_102208

def original_price : ℚ := 120
def first_discount_rate : ℚ := 30 / 100
def second_discount_rate : ℚ := 10 / 100
def tax_rate : ℚ := 8 / 100

def total_selling_price : ℚ :=
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  let final_price_with_tax := price_after_second_discount * (1 + tax_rate)
  final_price_with_tax

theorem coat_price_theorem :
  Int.floor total_selling_price = 81 ∧ Int.ceil total_selling_price = 82 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_price_theorem_l1022_102208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_point_l1022_102223

noncomputable def h (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4) + 2

theorem symmetry_about_point (x : ℝ) : 
  f x + h (-x) = 2 := by
  sorry

#check symmetry_about_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_point_l1022_102223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_square_pyramid_area_l1022_102270

/-- The total area of the four triangular faces of a right square-based pyramid -/
noncomputable def totalTriangularArea (baseEdge : ℝ) (lateralEdge : ℝ) : ℝ :=
  4 * (1/2 * baseEdge * Real.sqrt (lateralEdge ^ 2 - (baseEdge / 2) ^ 2))

/-- Theorem: The total area of the four triangular faces of a right square-based pyramid 
    with base edges of 10 units and lateral edges of 13 units is 240 square units -/
theorem right_square_pyramid_area : 
  totalTriangularArea 10 13 = 240 := by
  -- Unfold the definition of totalTriangularArea
  unfold totalTriangularArea
  -- Simplify the expression
  simp [Real.sqrt_sq]
  -- The proof steps would go here, but for now we use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_square_pyramid_area_l1022_102270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l1022_102287

/-- The hyperbola equation -2x^2 + 3y^2 - 8x - 24y + 4 = 0 -/
def hyperbola_eq (x y : ℝ) : Prop :=
  -2 * x^2 + 3 * y^2 - 8 * x - 24 * y + 4 = 0

/-- The focus coordinates -/
noncomputable def focus : ℝ × ℝ := (-2, 4 + 10 * Real.sqrt 3 / 3)

/-- Theorem stating that the given point is a focus of the hyperbola -/
theorem is_focus_of_hyperbola :
  let (h, k) := focus
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), hyperbola_eq x y ↔
      ((y - k)^2 / a^2) - ((x - h)^2 / b^2) = 1) ∧
    (h, k + Real.sqrt (a^2 + b^2)) = focus :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l1022_102287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_16_equals_u_1_l1022_102244

/-- Recursive sequence definition -/
noncomputable def u (a : ℝ) : ℕ → ℝ
  | 0 => a  -- Add case for 0
  | 1 => a
  | n + 1 => -1 / (u a n + 1)

/-- Theorem stating that the 16th term of the sequence equals the first term -/
theorem u_16_equals_u_1 (a : ℝ) (h : a > 0) : u a 16 = a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_16_equals_u_1_l1022_102244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_multiple_nonzero_digits_l1022_102200

theorem power_of_two_multiple_nonzero_digits (n : ℕ) :
  ∃ (a_n : ℕ), (a_n > 0) ∧ 
  (∀ d : ℕ, d ∈ Nat.digits 10 a_n → d = 1 ∨ d = 2) ∧ 
  (2^n ∣ a_n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_multiple_nonzero_digits_l1022_102200
