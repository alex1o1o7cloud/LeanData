import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1108_110862

/-- Two points are 10 units apart -/
def distance_between_points : ℝ := 10

/-- The area of the triangle formed by the two points and the point of tangency -/
def triangle_area (r : ℝ) : ℝ := 5 * r

/-- The theorem stating the maximum area and the corresponding radius -/
theorem max_triangle_area :
  (∀ r' : ℝ, 0 < r' ∧ r' ≤ distance_between_points → triangle_area r' ≤ triangle_area 5) ∧
  triangle_area 5 = 25 := by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1108_110862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_approximately_456_98_l1108_110825

/-- Calculates the selling price of an article given the cost price, markup percentage, and discount percentage. -/
noncomputable def selling_price (cost_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let marked_price := cost_price * (1 + markup_percent / 100)
  marked_price * (1 - discount_percent / 100)

/-- Theorem stating that the selling price is approximately 456.98 given the specified conditions. -/
theorem selling_price_approximately_456_98 :
  let cost_price := (540 : ℝ)
  let markup_percent := (15 : ℝ)
  let discount_percent := (26.40901771336554 : ℝ)
  let result := selling_price cost_price markup_percent discount_percent
  ∃ ε > 0, |result - 456.98| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_approximately_456_98_l1108_110825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_sum_odd_impossible_to_reach_zero_l1108_110888

-- Define the type of moves
inductive Move
| split (n a b : ℕ) : Move
| difference (a b : ℕ) : Move

-- Define the state of the blackboard
def Blackboard := List ℕ

-- Define the result of applying a move
def apply_move : Move → Blackboard → Blackboard
  | Move.split n a b, board => 
      if board.contains n then (a :: b :: board.filter (· ≠ n)) else board
  | Move.difference a b, board => 
      if board.contains a ∧ board.contains b ∧ a ≥ b 
      then ((a - b) :: board.filter (λ x => x ≠ a ∧ x ≠ b)) 
      else board

-- Define the sum of the blackboard
def board_sum (board : Blackboard) : ℕ := board.sum

-- Theorem: The sum of the blackboard always remains odd
theorem blackboard_sum_odd (initial_board : Blackboard) (moves : List Move) :
  board_sum initial_board = 2011 →
  Odd (board_sum (moves.foldl (λ b m => apply_move m b) initial_board)) :=
by
  sorry

-- Theorem: It's impossible to reach a single 0 on the blackboard
theorem impossible_to_reach_zero (initial_board : Blackboard) (moves : List Move) :
  board_sum initial_board = 2011 →
  ¬(board_sum (moves.foldl (λ b m => apply_move m b) initial_board) = 0 ∧ 
    (moves.foldl (λ b m => apply_move m b) initial_board).length = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blackboard_sum_odd_impossible_to_reach_zero_l1108_110888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1108_110841

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 2) + (x - 3) / (3 * x)

-- Define the solution set
def S : Set ℝ := Set.union (Set.Icc (-1/4) 0) (Set.Ioc 2 3)

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 4} = S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1108_110841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_surface_area_l1108_110827

/-- The total surface area of a regular tetrahedron with height h -/
noncomputable def tetrahedron_surface_area (h : ℝ) : ℝ := (3 * h^2 * Real.sqrt 3) / 2

/-- Theorem: The total surface area of a regular tetrahedron with height h is (3h² √3) / 2 -/
theorem regular_tetrahedron_surface_area (h : ℝ) (h_pos : h > 0) :
  tetrahedron_surface_area h = (3 * h^2 * Real.sqrt 3) / 2 := by
  -- Unfold the definition of tetrahedron_surface_area
  unfold tetrahedron_surface_area
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_surface_area_l1108_110827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1108_110891

theorem product_remainder (a b c : ℕ) 
  (ha : a % 36 = 16)
  (hb : b % 36 = 8)
  (hc : c % 36 = 24) :
  (a * b * c) % 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1108_110891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_centroid_to_sides_specific_triangle_l1108_110830

/-- Triangle ABC with given side lengths --/
structure Triangle where
  AB : ℝ
  BC : ℝ
  CA : ℝ

/-- The centroid of a triangle --/
def Centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Distance from a point to a line --/
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ := sorry

/-- Sum of distances from centroid to sides of triangle --/
noncomputable def sum_distances_centroid_to_sides (t : Triangle) : ℝ :=
  let G := Centroid t
  let side_AB := sorry
  let side_BC := sorry
  let side_CA := sorry
  distance_point_to_line G side_BC + 
  distance_point_to_line G side_CA + 
  distance_point_to_line G side_AB

/-- Theorem: Sum of distances from centroid to sides for given triangle --/
theorem sum_distances_centroid_to_sides_specific_triangle :
  let t : Triangle := { AB := 13, BC := 14, CA := 15 }
  sum_distances_centroid_to_sides t = 2348 / 195 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_centroid_to_sides_specific_triangle_l1108_110830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_two_heads_five_coins_l1108_110850

theorem probability_at_least_two_heads_five_coins :
  let n : ℕ := 5  -- number of coins
  let p : ℚ := 1/2  -- probability of heads for a fair coin
  let prob_at_least_two_heads := 1 - (Nat.choose n 0 * p^0 * (1-p)^n + Nat.choose n 1 * p^1 * (1-p)^(n-1))
  prob_at_least_two_heads = 13/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_two_heads_five_coins_l1108_110850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circumradius_l1108_110837

/-- The radius of a circle circumscribing an isosceles triangle with two sides of 13 cm and one side of 10 cm. -/
theorem isosceles_triangle_circumradius : 
  ∃ (R : ℝ), 
  (let a : ℝ := 13
   let b : ℝ := 13
   let c : ℝ := 10
   let s : ℝ := (a + b + c) / 2
   let K : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
   R = (a * b * c) / (4 * K)) ∧ R = 169 / 24 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circumradius_l1108_110837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_all_prime_polynomial_l1108_110822

-- Define a polynomial type
def MyPolynomial (α : Type*) := List α

-- Define the evaluation of a polynomial at a point
def eval_poly {α : Type*} [CommRing α] (p : MyPolynomial α) (x : α) : α :=
  match p with
  | [] => 0
  | a::p' => a + x * eval_poly p' x

-- Define primality
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- State the theorem
theorem no_all_prime_polynomial :
  ∀ (p : MyPolynomial ℤ), ∃ k : ℕ, ¬ is_prime (Int.natAbs (eval_poly p k)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_all_prime_polynomial_l1108_110822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_and_tangent_circle_l1108_110844

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- Distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (l1 l2 : Line) : ℝ :=
  abs (l1.c - l2.c) / Real.sqrt (l1.a^2 + l1.b^2)

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x y : ℝ) (l : Line) : ℝ :=
  abs (l.a * x + l.b * y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Distance between a point and a line -/
noncomputable def distance_between_point_and_line (px py : ℝ) (l : Line) : ℝ :=
  abs (l.a * px + l.b * py + l.c) / Real.sqrt (l.a^2 + l.b^2)

theorem parallel_lines_and_tangent_circle 
  (l1 : Line) 
  (l2 : Line) 
  (l : Line) 
  (c : Circle) 
  (h1 : l1.a = 4 ∧ l1.b = -2 ∧ l1.c = 7)
  (h2 : l2.a = 2 ∧ l2.b = -1 ∧ l2.c = 1)
  (h3 : l.a = 1 ∧ l.b = -2 ∧ l.c > 0)
  (h4 : c.center_x = 0 ∧ c.center_y = 2 ∧ c.radius = 1 / Real.sqrt 5)
  (h5 : distance_between_parallel_lines l1 l2 = (1/2) * distance_point_to_line 0 0 l) :
  (l.c = 5) ∧ 
  (distance_between_point_and_line c.center_x c.center_y l = c.radius) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_and_tangent_circle_l1108_110844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_l1108_110803

-- Define the ages of A, B, C, and D
variable (a b c d : ℕ)

-- Define the conditions
def cond1 (a b c : ℕ) : Prop := a + b = b + c + 11
def cond2 (a c d : ℕ) : Prop := a + c = c + d + 15
def cond3 (b d : ℕ) : Prop := b + d = 36
def cond4 (a d : ℕ) : Prop := 2 * a = 3 * d

-- Define the theorem
theorem age_difference :
  cond1 a b c → cond2 a c d → cond3 b d → cond4 a d →
  (max a (max b (max c d))) - (min a (min b (min c d))) = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_l1108_110803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_meeting_time_on_track_l1108_110856

/-- The time (in minutes) it takes for two people walking in opposite directions 
    on a circular track to meet for the first time. -/
noncomputable def meeting_time (track_circumference : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  track_circumference / (speed1 + speed2)

/-- Converts km/hr to m/min -/
noncomputable def km_per_hr_to_m_per_min (speed : ℝ) : ℝ :=
  speed * 1000 / 60

theorem first_meeting_time_on_track : 
  let track_circumference := (1000 : ℝ)
  let speed1 := km_per_hr_to_m_per_min 20
  let speed2 := km_per_hr_to_m_per_min 13
  abs (meeting_time track_circumference speed1 speed2 - 1.82) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_meeting_time_on_track_l1108_110856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_tangent_l1108_110882

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (1 - k) * x + Real.exp (-x)

theorem f_monotonicity_and_tangent (k : ℝ) :
  (∀ x y, k ≥ 1 → x < y → f k x > f k y) ∧
  (k < 1 →
    (∀ x y, x < y ∧ y < -Real.log (1 - k) → f k x > f k y) ∧
    (∀ x y, -Real.log (1 - k) < x ∧ x < y → f k x < f k y)) ∧
  (∀ t : ℝ, (∃ x : ℝ, (f 0 x - t) / x = (deriv (f 0)) x) ↔ t ≤ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_tangent_l1108_110882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_positive_reals_l1108_110840

-- Define the function f(x) = (1/3)^(1-x)
noncomputable def f (x : ℝ) : ℝ := (1/3)^(1-x)

-- Statement to prove
theorem f_range_is_positive_reals :
  (∀ y : ℝ, y > 0 → ∃ x : ℝ, f x = y) ∧
  (∀ z : ℝ, f z > 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_positive_reals_l1108_110840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_properties_l1108_110833

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a truncated pyramid -/
structure TruncatedPyramid where
  A : Point3D
  B : Point3D
  C : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D

/-- Theorem about properties of a specific truncated pyramid -/
theorem truncated_pyramid_properties (ABCA₁B₁C₁ : TruncatedPyramid) (Ω : Sphere) (M : Point3D) :
  -- ABC ∥ A₁B₁C₁
  (ABCA₁B₁C₁.A.z = ABCA₁B₁C₁.B.z) ∧ (ABCA₁B₁C₁.B.z = ABCA₁B₁C₁.C.z) ∧
  (ABCA₁B₁C₁.A₁.z = ABCA₁B₁C₁.B₁.z) ∧ (ABCA₁B₁C₁.B₁.z = ABCA₁B₁C₁.C₁.z) →
  -- Triangle ABA₁ is equilateral
  (ABCA₁B₁C₁.A.x - ABCA₁B₁C₁.B.x)^2 + (ABCA₁B₁C₁.A.y - ABCA₁B₁C₁.B.y)^2 + (ABCA₁B₁C₁.A.z - ABCA₁B₁C₁.B.z)^2 =
  (ABCA₁B₁C₁.A.x - ABCA₁B₁C₁.A₁.x)^2 + (ABCA₁B₁C₁.A.y - ABCA₁B₁C₁.A₁.y)^2 + (ABCA₁B₁C₁.A.z - ABCA₁B₁C₁.A₁.z)^2 →
  -- CC₁ ⟂ ABC
  (ABCA₁B₁C₁.C.x = ABCA₁B₁C₁.C₁.x) ∧ (ABCA₁B₁C₁.C.y = ABCA₁B₁C₁.C₁.y) →
  -- CM : MC₁ = 1:2
  (M.z - ABCA₁B₁C₁.C.z) / (ABCA₁B₁C₁.C₁.z - M.z) = 1/2 →
  -- Sphere Ω passes through vertices of AA₁B and touches CC₁ at M
  Ω.radius = Real.sqrt 7 ∧
  (Ω.center.x - ABCA₁B₁C₁.A.x)^2 + (Ω.center.y - ABCA₁B₁C₁.A.y)^2 + (Ω.center.z - ABCA₁B₁C₁.A.z)^2 = 7 ∧
  (Ω.center.x - ABCA₁B₁C₁.A₁.x)^2 + (Ω.center.y - ABCA₁B₁C₁.A₁.y)^2 + (Ω.center.z - ABCA₁B₁C₁.A₁.z)^2 = 7 ∧
  (Ω.center.x - ABCA₁B₁C₁.B.x)^2 + (Ω.center.y - ABCA₁B₁C₁.B.y)^2 + (Ω.center.z - ABCA₁B₁C₁.B.z)^2 = 7 ∧
  (Ω.center.x - M.x)^2 + (Ω.center.y - M.y)^2 + (Ω.center.z - M.z)^2 = 7 →
  -- ∠BAC = arcsin(√(2/3))
  Real.arcsin (Real.sqrt (2/3)) = Real.arccos ((ABCA₁B₁C₁.B.x - ABCA₁B₁C₁.A.x) * (ABCA₁B₁C₁.C.x - ABCA₁B₁C₁.A.x) +
    (ABCA₁B₁C₁.B.y - ABCA₁B₁C₁.A.y) * (ABCA₁B₁C₁.C.y - ABCA₁B₁C₁.A.y) +
    (ABCA₁B₁C₁.B.z - ABCA₁B₁C₁.A.z) * (ABCA₁B₁C₁.C.z - ABCA₁B₁C₁.A.z)) /
    (Real.sqrt ((ABCA₁B₁C₁.B.x - ABCA₁B₁C₁.A.x)^2 + (ABCA₁B₁C₁.B.y - ABCA₁B₁C₁.A.y)^2 + (ABCA₁B₁C₁.B.z - ABCA₁B₁C₁.A.z)^2) *
    Real.sqrt ((ABCA₁B₁C₁.C.x - ABCA₁B₁C₁.A.x)^2 + (ABCA₁B₁C₁.C.y - ABCA₁B₁C₁.A.y)^2 + (ABCA₁B₁C₁.C.z - ABCA₁B₁C₁.A.z)^2)) →
  -- Conclusions
  ((ABCA₁B₁C₁.A.x - ABCA₁B₁C₁.B.x)^2 + (ABCA₁B₁C₁.A.y - ABCA₁B₁C₁.B.y)^2 + (ABCA₁B₁C₁.A.z - ABCA₁B₁C₁.B.z)^2 = 21) ∧
  (Real.arcsin (Real.sqrt (2/3)) = Real.arccos ((ABCA₁B₁C₁.C₁.x - ABCA₁B₁C₁.C.x) * (ABCA₁B₁C₁.A₁.x - ABCA₁B₁C₁.A.x) +
    (ABCA₁B₁C₁.C₁.y - ABCA₁B₁C₁.C.y) * (ABCA₁B₁C₁.A₁.y - ABCA₁B₁C₁.A.y) +
    (ABCA₁B₁C₁.C₁.z - ABCA₁B₁C₁.C.z) * (ABCA₁B₁C₁.A₁.z - ABCA₁B₁C₁.A.z)) /
    (Real.sqrt ((ABCA₁B₁C₁.C₁.x - ABCA₁B₁C₁.C.x)^2 + (ABCA₁B₁C₁.C₁.y - ABCA₁B₁C₁.C.y)^2 + (ABCA₁B₁C₁.C₁.z - ABCA₁B₁C₁.C.z)^2) *
    Real.sqrt ((ABCA₁B₁C₁.A₁.x - ABCA₁B₁C₁.A.x)^2 + (ABCA₁B₁C₁.A₁.y - ABCA₁B₁C₁.A.y)^2 + (ABCA₁B₁C₁.A₁.z - ABCA₁B₁C₁.A.z)^2))) ∧
  ((ABCA₁B₁C₁.A₁.x - ABCA₁B₁C₁.C₁.x)^2 + (ABCA₁B₁C₁.A₁.y - ABCA₁B₁C₁.C₁.y)^2 + (ABCA₁B₁C₁.A₁.z - ABCA₁B₁C₁.C₁.z)^2 = ((7 - 2 * Real.sqrt 7) / 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_properties_l1108_110833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mars_colony_suitability_l1108_110893

noncomputable def mars_surface_area : ℝ := 1

noncomputable def ice_free_fraction : ℝ := 1/3

noncomputable def suitable_fraction_of_ice_free : ℝ := 2/3

theorem mars_colony_suitability :
  ice_free_fraction * suitable_fraction_of_ice_free * mars_surface_area = 2/9 := by
  -- Unfold definitions
  unfold ice_free_fraction suitable_fraction_of_ice_free mars_surface_area
  -- Simplify the expression
  simp [mul_assoc]
  -- Prove the equality
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mars_colony_suitability_l1108_110893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P₀_parallel_to_intersection_l1108_110883

def P₀ : ℝ × ℝ × ℝ := (4, -3, 2)

def plane1 (x y z : ℝ) : Prop := 2*x - y - z + 1 = 0
def plane2 (x y z : ℝ) : Prop := 3*x + 2*y + z - 8 = 0

def is_parallel_to_intersection (l : ℝ × ℝ × ℝ → Prop) : Prop :=
  ∃ (v : ℝ × ℝ × ℝ), v ≠ (0, 0, 0) ∧
  (∀ (p q : ℝ × ℝ × ℝ), plane1 p.1 p.2.1 p.2.2 → plane2 p.1 p.2.1 p.2.2 →
                         plane1 q.1 q.2.1 q.2.2 → plane2 q.1 q.2.1 q.2.2 →
                         ∃ (t : ℝ), q = (p.1 + t * v.1, p.2.1 + t * v.2.1, p.2.2 + t * v.2.2)) ∧
  (∀ (p q : ℝ × ℝ × ℝ), l p → l q → ∃ (t : ℝ), q = (p.1 + t * v.1, p.2.1 + t * v.2.1, p.2.2 + t * v.2.2))

def line_equation (x y z : ℝ) : Prop :=
  (x - 4) / 1 = (y + 3) / (-5) ∧ (y + 3) / (-5) = (z - 2) / 7

theorem line_through_P₀_parallel_to_intersection :
  ∃ (l : ℝ × ℝ × ℝ → Prop),
    l P₀ ∧
    is_parallel_to_intersection l ∧
    ∀ (x y z : ℝ), l (x, y, z) ↔ line_equation x y z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P₀_parallel_to_intersection_l1108_110883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1108_110826

def my_sequence : List Nat := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

theorem product_remainder (seq : List Nat) (h : seq = my_sequence) :
  (seq.prod) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1108_110826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1108_110838

-- Define the ⋄ operation
noncomputable def diamond (x y : ℝ) : ℝ := (x^2 + y^2) / (x^2 - y^2)

-- Theorem statement
theorem diamond_calculation :
  diamond (diamond 2 3) 4 = -569/231 := by
  -- Expand the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp [pow_two]
  -- Perform numerical calculations
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1108_110838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_e_l1108_110804

/-- The function f(x) = xe^x -/
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := Real.exp x + x * Real.exp x

theorem tangent_line_at_one_e :
  let x₀ : ℝ := 1
  let y₀ : ℝ := Real.exp 1
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 2 * Real.exp 1 * x - Real.exp 1 :=
by
  intros x₀ y₀ m x y
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_e_l1108_110804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_subset_size_l1108_110894

/-- A subset of integers from 1 to 150 where no element is 4 times another -/
def ValidSubset (S : Finset ℕ) : Prop :=
  ∀ x ∈ S, x ≤ 150 ∧ x ≥ 1 ∧ ∀ y ∈ S, x ≠ 4 * y ∧ y ≠ 4 * x

/-- The maximum size of a valid subset is 120 -/
theorem max_valid_subset_size :
  ∃ S : Finset ℕ, ValidSubset S ∧ S.card = 120 ∧
    ∀ T : Finset ℕ, ValidSubset T → T.card ≤ 120 := by
  sorry

#check max_valid_subset_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_subset_size_l1108_110894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_abc_value_l1108_110819

theorem smallest_abc_value (a b c : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : (a * b) % (5 * c) = 0)
  (h2 : (b * c) % (13 * a) = 0)
  (h3 : (a * c) % (31 * b) = 0) :
  4060225 ≤ (a * b * c) :=
by
  sorry

#eval 2015 * 2015  -- This should output 4060225

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_abc_value_l1108_110819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compose_functions_equal_eleven_l1108_110808

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x + 1
noncomputable def g (x : ℝ) : ℝ := x / 4

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := x - 1
noncomputable def g_inv (x : ℝ) : ℝ := 4 * x

-- State the theorem
theorem compose_functions_equal_eleven :
  f (g_inv (f_inv (f_inv (g (f 17))))) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compose_functions_equal_eleven_l1108_110808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integer_a_value_min_integer_a_is_three_l1108_110817

theorem min_integer_a_value (a : ℤ) : 
  (∀ x : ℤ, x - a < 0 ∧ x > -3/2 → x ∈ Set.Icc (-1 : ℤ) 2) →
  (∃ x₁ x₂ x₃ x₄ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    x₁ - a < 0 ∧ x₁ > -3/2 ∧
    x₂ - a < 0 ∧ x₂ > -3/2 ∧
    x₃ - a < 0 ∧ x₃ > -3/2 ∧
    x₄ - a < 0 ∧ x₄ > -3/2) →
  a ≥ 3 :=
by sorry

theorem min_integer_a_is_three : 
  ∃ a : ℤ, a = 3 ∧
  (∀ x : ℤ, x - a < 0 ∧ x > -3/2 → x ∈ Set.Icc (-1 : ℤ) 2) ∧
  (∃ x₁ x₂ x₃ x₄ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    x₁ - a < 0 ∧ x₁ > -3/2 ∧
    x₂ - a < 0 ∧ x₂ > -3/2 ∧
    x₃ - a < 0 ∧ x₃ > -3/2 ∧
    x₄ - a < 0 ∧ x₄ > -3/2) ∧
  (∀ b : ℤ, b < 3 →
    ¬(∃ x₁ x₂ x₃ x₄ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
      x₁ - b < 0 ∧ x₁ > -3/2 ∧
      x₂ - b < 0 ∧ x₂ > -3/2 ∧
      x₃ - b < 0 ∧ x₃ > -3/2 ∧
      x₄ - b < 0 ∧ x₄ > -3/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integer_a_value_min_integer_a_is_three_l1108_110817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_brother_is_trulya_l1108_110897

/-- Represents the two possible states of a card. -/
inductive CardState
| Purple
| NotPurple

/-- Represents the two brothers. -/
inductive Brother
| First
| Second

/-- Represents whether a statement is true or false. -/
inductive Truthfulness
| True
| False

/-- The nature of each brother (always tells the truth or always lies). -/
def brother_nature : Brother → Truthfulness
| Brother.First => Truthfulness.False  -- Trulya (always lies)
| Brother.Second => Truthfulness.True  -- Veritius (always tells the truth)

/-- The statement made by each brother. -/
def brother_statement : Brother → Prop
| Brother.First => ∀ (card : CardState), card = CardState.Purple
| Brother.Second => ¬(∀ (card : CardState), card = CardState.Purple)

/-- Helper function to get the other brother -/
def other_brother : Brother → Brother
| Brother.First => Brother.Second
| Brother.Second => Brother.First

/-- The theorem stating that the first brother must be Trulya. -/
theorem first_brother_is_trulya :
  (∃ (b : Brother), brother_nature b = Truthfulness.False ∧ 
                    brother_nature (other_brother b) = Truthfulness.True) →
  (∀ (b : Brother), (brother_nature b = Truthfulness.True) = brother_statement b) →
  brother_nature Brother.First = Truthfulness.False := by
  sorry

/-- Proof that Brother is decidable -/
instance : DecidableEq Brother :=
  fun a b => match a, b with
  | Brother.First, Brother.First => isTrue rfl
  | Brother.Second, Brother.Second => isTrue rfl
  | Brother.First, Brother.Second => isFalse (fun h => Brother.noConfusion h)
  | Brother.Second, Brother.First => isFalse (fun h => Brother.noConfusion h)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_brother_is_trulya_l1108_110897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1108_110810

/-- Sequence a_n with properties as described in the problem -/
def a : ℕ+ → ℚ := sorry

/-- Sum of first n terms of sequence a_n -/
def S (n : ℕ+) : ℚ := (Finset.range n.val).sum (λ i => a ⟨i+1, Nat.succ_pos i⟩)

/-- Sequence b_n defined in terms of a_n -/
def b (n : ℕ+) : ℚ := 4 / (a n * (a n + 2))

/-- Sum of first n terms of sequence b_n -/
def T (n : ℕ+) : ℚ := (Finset.range n.val).sum (λ i => b ⟨i+1, Nat.succ_pos i⟩)

/-- The main theorem encompassing both parts of the problem -/
theorem sequence_properties :
  (∀ n : ℕ+, 2 * S n = a n ^ 2 + a n) →
  (a 1 ≠ 0) →
  (∀ n : ℕ+, 0 < a n) →
  ((∀ n : ℕ+, ∃ m : ℤ, T n < m) →
   (∃ m_min : ℤ, (∀ n : ℕ+, T n < m_min) ∧ 
    (∀ m' : ℤ, m' < m_min → ∃ n : ℕ+, m' ≤ T n))) →
  ((∀ n : ℕ+, a n = n) ∧
   (∃ m_min : ℤ, m_min = 3 ∧ 
    (∀ n : ℕ+, T n < m_min) ∧
    (∀ m' : ℤ, m' < m_min → ∃ n : ℕ+, m' ≤ T n))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1108_110810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1108_110824

theorem calculate_expression : 2 * Real.sin (π / 4) - Real.sqrt 4 + (-1/3)⁻¹ + abs (Real.sqrt 2 - 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1108_110824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeans_markup_theorem_l1108_110832

/-- Represents the percentage increase in price -/
noncomputable def percentage_increase (initial : ℝ) (final : ℝ) : ℝ :=
  (final - initial) / initial * 100

/-- Calculates the final price after applying a markup percentage -/
noncomputable def apply_markup (price : ℝ) (markup_percent : ℝ) : ℝ :=
  price * (1 + markup_percent / 100)

theorem jeans_markup_theorem (p_A p_B p_C : ℝ) 
  (h_positive : p_A > 0 ∧ p_B > 0 ∧ p_C > 0) : 
  let final_A := apply_markup (apply_markup p_A 30) 20
  let final_B := apply_markup (apply_markup p_B 40) 15
  let final_C := apply_markup (apply_markup p_C 50) 10
  percentage_increase p_A final_A = 56 ∧
  percentage_increase p_B final_B = 61 ∧
  percentage_increase p_C final_C = 65 := by
  sorry

#eval "Jeans markup theorem has been stated."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeans_markup_theorem_l1108_110832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_rank_correct_l1108_110834

def letter_group : List Char := ['c', 'e', 'i', 'i', 'i', 'n', 'o', 's', 't', 't', 'u', 'v']

def target_permutation : List Char := ['u', 't', 't', 'e', 'n', 's', 'i', 'o', 's', 'i', 'c', 'v', 'i', 's']

def permutation_rank : ℕ := 1115600587

theorem permutation_rank_correct : 
  ∃ (rank : List Char → List Char → ℕ), rank letter_group target_permutation = permutation_rank := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_rank_correct_l1108_110834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l1108_110807

theorem unique_solution_condition (a : ℝ) : 
  (a ≥ 0) →
  (∃! x : ℝ, |((x^3 - 10*x^2 + 31*x - 30) / (x^2 - 8*x + 15))| = (Real.sqrt (2*x - a))^2 + 2 - 2*x) ↔ 
  (a = 1 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l1108_110807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_tangent_lines_through_M_l1108_110899

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the line
def my_line (a x y : ℝ) : Prop := a * x - y + 4 = 0

-- Define the point M
def point_M : ℝ × ℝ := (3, 1)

-- Theorem 1: Line is tangent to the circle iff a = 0 or a = 4/3
theorem tangent_line_condition (a : ℝ) :
  (∃ x y, my_circle x y ∧ my_line a x y ∧ 
    ∀ x' y', my_circle x' y' → my_line a x' y' → (x', y') = (x, y)) ↔ 
  (a = 0 ∨ a = 4/3) :=
sorry

-- Theorem 2: Tangent lines through M
theorem tangent_lines_through_M :
  (∃ x y, my_circle x y ∧ (x = 3 ∨ 3*x - 4*y - 5 = 0) ∧ 
    ∀ x' y', my_circle x' y' → (x' = 3 ∨ 3*x' - 4*y' - 5 = 0) → (x', y') = (x, y)) ∧
  (∀ k m : ℝ, (∃ x y, my_circle x y ∧ k*x + m = y ∧ k*(3:ℝ) + m = 1 ∧
    ∀ x' y', my_circle x' y' → k*x' + m = y' → (x', y') = (x, y)) →
    (k = 0 ∧ m = 3) ∨ (3*k = 4 ∧ 4*m = -1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_tangent_lines_through_M_l1108_110899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2022nd_term_l1108_110815

def digit_square_sum (n : ℕ) : ℕ :=
  (n.digits 10).map (· ^ 2) |>.sum

def sequence_term (n : ℕ) : ℕ → ℕ
  | 0 => 2022
  | m + 1 => digit_square_sum (sequence_term n m)

def is_cyclic (s : ℕ → ℕ) (start period : ℕ) : Prop :=
  ∀ n, n ≥ start → s n = s (n + period)

theorem sequence_2022nd_term :
  (is_cyclic (sequence_term 0) 5 8) →
  sequence_term 0 5 = 89 →
  sequence_term 0 2021 = 89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2022nd_term_l1108_110815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_numbers_to_seven_final_result_l1108_110898

theorem sum_odd_numbers_to_seven : 
  List.sum (List.map (fun i => 2 * i + 1) (List.range 4)) = 16 := by
  sorry

theorem final_result :
  List.sum (List.map (fun i => 2 * i + 1) (List.range 4)) + 0 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_numbers_to_seven_final_result_l1108_110898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elegant_number_equality_l1108_110885

def is_elegant (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.Nodup ∧ List.Sorted (· < ·) digits

def four_digit_elegant_count : ℕ := Nat.choose 9 4

def five_digit_elegant_count : ℕ := Nat.choose 9 5

theorem elegant_number_equality : 
  four_digit_elegant_count = five_digit_elegant_count := by
  -- Unfold definitions
  unfold four_digit_elegant_count five_digit_elegant_count
  -- Both are equal to Nat.choose 9 5
  rfl

#eval four_digit_elegant_count
#eval five_digit_elegant_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elegant_number_equality_l1108_110885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_size_l1108_110831

/-- The number of second-year students studying numeric methods -/
def numeric_methods : ℕ := 230

/-- The number of second-year students studying automatic control of airborne vehicles -/
def auto_control : ℕ := 423

/-- The number of second-year students studying both subjects -/
def both_subjects : ℕ := 134

/-- The percentage of second-year students in the faculty -/
def second_year_percentage : ℚ := 4/5

/-- The total number of students in the faculty -/
def total_students : ℕ := 649

/-- A small tolerance for approximate equality -/
def epsilon : ℚ := 1/100

theorem faculty_size :
  abs ((numeric_methods + auto_control - both_subjects : ℚ) / second_year_percentage - total_students) < epsilon :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_size_l1108_110831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_and_even_l1108_110879

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ∈ Set.Icc (-2) 2 then -x^2 + 1 
  else if x ∈ Set.Icc 6 10 then -(x-8)^2 + 1 
  else 0  -- placeholder for other x values

-- State the theorem
theorem f_symmetric_and_even :
  (∀ x, f (-x) = f x) ∧  -- f is even
  (∀ x, f (4 - x) = f x) ∧  -- f is symmetric about x = 2
  (∀ x ∈ Set.Icc (-2) 2, f x = -x^2 + 1) →  -- f(x) = -x^2 + 1 for x ∈ [-2, 2]
  (∀ x ∈ Set.Icc 6 10, f x = -(x-8)^2 + 1) :=  -- f(x) = -(x-8)^2 + 1 for x ∈ [6, 10]
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_and_even_l1108_110879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1108_110842

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the hyperbola -/
def onHyperbola (h : Hyperbola) (p : Point2D) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a point P on the hyperbola x²/64 - y²/36 = 1 with foci F₁ and F₂,
    if |PF₁| = 15, then |PF₂| = 31 -/
theorem hyperbola_focal_distance
  (h : Hyperbola)
  (p f1 f2 : Point2D)
  (h_eq : h.a^2 = 64 ∧ h.b^2 = 36)
  (h_on : onHyperbola h p)
  (h_foci : f1.x = -10 ∧ f1.y = 0 ∧ f2.x = 10 ∧ f2.y = 0)
  (h_dist : distance p f1 = 15) :
  distance p f2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1108_110842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1108_110806

/-- The line equation 4x - 3y = 0 -/
def line (x y : ℝ) : Prop := 4 * x - 3 * y = 0

/-- The circle equation x² + y² - 18x - 45 = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 18*x - 45 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (9, 0)

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := Real.sqrt 126

/-- The distance from the center of the circle to the line -/
noncomputable def distance_center_to_line : ℝ := 36 / 5

theorem line_intersects_circle : 
  distance_center_to_line < circle_radius :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1108_110806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_l1108_110892

/-- The function defining the boundary of the region -/
def g (x y z : ℝ) : ℝ :=
  2 * |2*x + y + z| + 2 * |x + 2*y - z| + |x - y + 2*z| + |y - 2*x + z|

/-- The region defined by the inequality -/
def Region : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | g p.1 p.2.1 p.2.2 ≤ 10}

/-- The volume of the region -/
noncomputable def volume : ℝ := sorry

/-- Theorem stating the volume of the region -/
theorem volume_of_region : volume = 125 / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_l1108_110892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_equation_l1108_110877

/-- Given a hyperbola and a circle, prove the equation of the hyperbola's asymptotes -/
theorem hyperbola_asymptotes_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  (∃ x y : ℝ, x^2 + y^2 - 6*x + 5 = 0) →
  ∃ k : ℝ, k = 2*Real.sqrt 5/5 ∧ 
    ∀ x y : ℝ, (y = k*x ∨ y = -k*x) → 
      ∃ x' y' : ℝ, x'^2 + y'^2 - 6*x' + 5 = 0 ∧ 
        (x' - x)^2 + (y' - y)^2 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_equation_l1108_110877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1108_110836

noncomputable def F (x y : ℝ) : ℝ := y^x

noncomputable def a (n : ℕ) : ℝ := F n 2 / F 2 n

theorem min_value_of_a (n k : ℕ) (h1 : n > 0) (h2 : k > 0) :
  (∀ m : ℕ, m > 0 → a n ≥ a m) →
  (a k = 8/9 ∧ k = 3 ∧ ∀ m : ℕ, m > 0 → a k ≤ a m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1108_110836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1108_110820

namespace TriangleProof

structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

def Triangle.valid (t : Triangle) : Prop :=
  t.b = Real.sqrt 3 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  Real.cos t.A * Real.sin t.B + (t.c - Real.sin t.A) * Real.cos (t.A + t.C) = 0

theorem triangle_properties (t : Triangle) (h : t.valid) :
  t.B = Real.pi / 3 ∧
  (t.area = Real.sqrt 3 / 2 → Real.sin t.A + Real.sin t.C = 3 / 2) := by
  sorry

end TriangleProof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1108_110820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l1108_110868

noncomputable def points : List (ℝ × ℝ) := [(0, 7), (2, 4), (4, -6), (8, 0), (-2, -3)]

noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ := Real.sqrt (p.1^2 + p.2^2)

def below_line (p : ℝ × ℝ) : Prop := p.2 ≤ -1/2 * p.1 + 7

theorem farthest_point :
  (∀ p ∈ points, below_line p) →
  (∀ p ∈ points, distance_from_origin (4, -6) ≥ distance_from_origin p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l1108_110868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1108_110859

/-- The focus of a parabola y = ax^2 -/
noncomputable def focus (a : ℝ) : ℝ × ℝ := (0, 1 / (4 * a))

/-- The parabola equation y = 4x^2 -/
def parabola (x : ℝ) : ℝ := 4 * x^2

theorem parabola_focus :
  focus 4 = (0, 1/16) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l1108_110859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_equation_pattern_l1108_110889

/-- The nth equation in the pattern -/
def nthEquation (n : ℕ) : ℚ → ℚ → Prop :=
  match n with
  | 1 => λ x y => x * 1 = y
  | 2 => λ x y => x * 1 * 3 = y * (y + 1)
  | 3 => λ x y => x * 1 * 3 * 5 = y * (y + 1) * (y + 2)
  | _ => λ x y => x = y  -- Default case to avoid type inference issues

/-- The theorem to be proved -/
theorem fifth_equation_pattern :
  (nthEquation 1 (2^1) 2) →
  (nthEquation 2 (2^2) 3) →
  (nthEquation 3 (2^3) 4) →
  2^5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_equation_pattern_l1108_110889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_theorem_l1108_110846

/-- A line passing through a point with a specific X-axis projection between two other lines -/
structure SpecialLine where
  -- The slope of the line
  m : ℝ
  -- The line passes through the point (5, 3)
  passes_through : m * (5 : ℝ) - 3 = m * 5 - 3
  -- The X-axis projection between two lines is 1 unit in length
  projection_length : |(-15 : ℝ) / (4 * m + 3)| = 1

/-- The two possible equations of the special line -/
def special_line_equations (l : SpecialLine) : Prop :=
  (l.m = 3 ∧ ∀ x y : ℝ, y = 3 * x - 12) ∨
  (l.m = -4.5 ∧ ∀ x y : ℝ, y = -4.5 * x + 25.5)

/-- The main theorem stating that a SpecialLine satisfies one of the two equations -/
theorem special_line_theorem (l : SpecialLine) : special_line_equations l := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_theorem_l1108_110846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nates_matches_l1108_110848

/-- The number of matches Nate started with -/
def initial_matches : ℕ → Prop := λ M => M > 0

/-- The number of matches Nate dropped in the creek -/
def dropped_matches : ℕ := 10

/-- The number of matches Nate's dog ate -/
def dog_eaten_matches : ℕ := 2 * dropped_matches

/-- The number of matches Nate has left -/
def remaining_matches : ℕ := 40

/-- Theorem stating the initial number of matches Nate had -/
theorem nates_matches (M : ℕ) : 
  initial_matches M ↔ M - dropped_matches - dog_eaten_matches = remaining_matches := by
  sorry

#eval dropped_matches + dog_eaten_matches + remaining_matches

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nates_matches_l1108_110848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1108_110853

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  let m := (a + b + c) / 3
  (Real.sqrt (a + Real.sqrt (b + Real.sqrt c)) +
   Real.sqrt (b + Real.sqrt (c + Real.sqrt a)) +
   Real.sqrt (c + Real.sqrt (a + Real.sqrt b))) ≤
  (3 * Real.sqrt (m + Real.sqrt (m + Real.sqrt m))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1108_110853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_B25_to_dec_l1108_110866

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => c.toNat - '0'.toNat

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.data.reverse.enum.foldl (fun acc (i, c) => acc + hex_to_dec c * 16^i) 0

theorem hex_B25_to_dec :
  hex_string_to_dec "B25" = 2853 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_B25_to_dec_l1108_110866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_semicircle_area_l1108_110871

/-- A figure composed of a rectangle and a semicircle -/
structure RectangleSemicircle where
  width : ℝ
  height : ℝ
  semicircle_side : Bool

/-- The y-coordinates of the vertices -/
def y_coordinates : List ℝ := [0, 3, 6, 9]

/-- Calculate the area of a RectangleSemicircle -/
noncomputable def area (fig : RectangleSemicircle) : ℝ :=
  fig.width * fig.height + (if fig.semicircle_side then Real.pi * fig.width^2 / 8 else 0)

/-- The main theorem -/
theorem rectangle_semicircle_area :
  ∃ (fig : RectangleSemicircle),
    (fig.height = 6) ∧
    (fig.width = 4) ∧
    fig.semicircle_side ∧
    (area fig = 24 + 2 * Real.pi) := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_semicircle_area_l1108_110871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coeff_is_ninth_term_n_is_sixteen_l1108_110863

/-- The binomial expansion of (3x + 1/x)^n where the fifth term is constant -/
noncomputable def binomial_expansion (n : ℕ) (x : ℝ) : ℝ := (3*x + 1/x)^n

/-- The rth term of the expansion -/
noncomputable def term (n r : ℕ) (x : ℝ) : ℝ := 
  (Nat.choose n r : ℝ) * (3*x)^(n-r) * (1/x)^r

/-- Condition that the fifth term is constant -/
def fifth_term_constant (n : ℕ) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → term n 4 x = c

/-- The term number with maximum coefficient -/
def max_coeff_term (n : ℕ) : ℕ := n / 2 + 1

/-- Theorem: The term with maximum coefficient is the 9th term when n = 16 -/
theorem max_coeff_is_ninth_term (n : ℕ) (h : fifth_term_constant n) : 
  max_coeff_term n = 9 := by
  sorry

/-- Corollary: When the fifth term is constant, n must be 16 -/
theorem n_is_sixteen (h : fifth_term_constant n) : n = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coeff_is_ninth_term_n_is_sixteen_l1108_110863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_second_red_given_first_white_l1108_110851

/-- Represents the color of a ball -/
inductive Color where
  | Red
  | White

/-- Represents the state of the bag after the first draw -/
structure BagState where
  red_balls : ℕ
  white_balls : ℕ

/-- Calculate the probability of drawing a red ball given the current bag state -/
def prob_red (state : BagState) : ℚ :=
  state.red_balls / (state.red_balls + state.white_balls)

theorem probability_second_red_given_first_white 
  (initial_red : ℕ) 
  (initial_white : ℕ) 
  (h_initial_red : initial_red = 3) 
  (h_initial_white : initial_white = 7) :
  let first_draw_state : BagState := ⟨initial_red, initial_white - 1⟩
  prob_red first_draw_state = 1/3 := by
  sorry

#check probability_second_red_given_first_white

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_second_red_given_first_white_l1108_110851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_combinations_to_open_lock_l1108_110805

/-- Represents a combination of three dial positions -/
structure Combination where
  first : Fin 8
  second : Fin 8
  third : Fin 8
deriving Fintype

/-- Checks if two combinations match in at least two positions -/
def matches_two_positions (c1 c2 : Combination) : Bool :=
  (c1.first = c2.first && c1.second = c2.second) ||
  (c1.first = c2.first && c1.third = c2.third) ||
  (c1.second = c2.second && c1.third = c2.third)

/-- The set of all possible combinations -/
def all_combinations : Finset Combination :=
  Finset.univ

/-- A set of combinations is sufficient if it matches every possible combination in at least two positions -/
def is_sufficient (S : Finset Combination) : Prop :=
  ∀ c ∈ all_combinations, ∃ s ∈ S, matches_two_positions c s

/-- The main theorem: The minimum number of combinations to guarantee opening the lock is 32 -/
theorem min_combinations_to_open_lock :
  ∃ S : Finset Combination, is_sufficient S ∧ S.card = 32 ∧
  ∀ T : Finset Combination, is_sufficient T → T.card ≥ 32 := by
  sorry

#eval Fintype.card Combination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_combinations_to_open_lock_l1108_110805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_z_l1108_110852

-- Define the complex number z
variable (z : ℂ)

-- State the theorem
theorem real_part_of_z (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_z_l1108_110852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l1108_110845

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the perimeter of a triangle -/
noncomputable def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Represents the similarity ratio between two triangles -/
noncomputable def similarity_ratio (t1 t2 : Triangle) : ℝ := t2.c / t1.c

theorem similar_triangle_perimeter (t1 t2 : Triangle) 
  (h1 : t1.a = 12 ∧ t1.b = 12 ∧ t1.c = 16)
  (h2 : t2.c = 40)
  (h3 : similarity_ratio t1 t2 = t2.c / t1.c) :
  t2.perimeter = 100 := by
  sorry

#check similar_triangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l1108_110845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_l1108_110829

-- Define the sample space
inductive Ball : Type
| Red : Ball
| White : Ball
deriving DecidableEq

-- Define the bag contents
def bag : Multiset Ball :=
  2 • {Ball.Red} + 2 • {Ball.White}

-- Define the event of drawing two balls
def draw : Finset (Ball × Ball) :=
  (Finset.product bag.toFinset bag.toFinset).filter (fun (b1, b2) => b1 ≠ b2)

-- Define the events
def atLeastOneWhite : Set (Ball × Ball) :=
  {p | p.1 = Ball.White ∨ p.2 = Ball.White}

def bothRed : Set (Ball × Ball) :=
  {p | p.1 = Ball.Red ∧ p.2 = Ball.Red}

-- Theorem to prove
theorem events_mutually_exclusive :
  atLeastOneWhite ∩ bothRed = ∅ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_l1108_110829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l1108_110860

-- Define the complex number z
def z : ℂ := Complex.I * (1 + 2 * Complex.I)

-- Theorem stating that z is in the second quadrant
theorem z_in_second_quadrant : 
  (Complex.re z < 0) ∧ (Complex.im z > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l1108_110860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_compound_approx_l1108_110823

-- Define the atomic masses
def carbon_mass : ℝ := 12.01
def hydrogen_mass : ℝ := 1.008
def oxygen_mass : ℝ := 16.00

-- Define the number of atoms in the compound C6H8O6
def carbon_atoms : ℕ := 6
def hydrogen_atoms : ℕ := 8
def oxygen_atoms : ℕ := 6

-- Define the number of moles
def moles : ℝ := 10

-- Define the given total weight
def given_total_weight : ℝ := 1760

-- Define the molar mass of the compound
def molar_mass : ℝ :=
  carbon_mass * (carbon_atoms : ℝ) +
  hydrogen_mass * (hydrogen_atoms : ℝ) +
  oxygen_mass * (oxygen_atoms : ℝ)

-- Define the calculated weight
def calculated_weight : ℝ := molar_mass * moles

-- Theorem to prove
theorem weight_of_compound_approx :
  abs (calculated_weight - given_total_weight) < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_compound_approx_l1108_110823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1108_110800

noncomputable def m1 (a : ℝ) : ℝ := -(2 * a - 1) / a
noncomputable def m2 (a : ℝ) : ℝ := a

-- Theorem statement
theorem perpendicular_lines (a : ℝ) : 
  (m1 a * m2 a = -1 ∨ (a = 0 ∧ m2 a = 0)) → a = 0 ∨ a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1108_110800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_is_circle_through_A_and_D_l1108_110812

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given points and circle
variable (A B C D : Point)
variable (ω : Circle)

-- Define the condition that ω passes through B and C
def ω_passes_through_BC (B C : Point) (ω : Circle) : Prop := 
  (B.x - ω.center.x)^2 + (B.y - ω.center.y)^2 = ω.radius^2 ∧
  (C.x - ω.center.x)^2 + (C.y - ω.center.y)^2 = ω.radius^2

-- Define a point P on ω
def P_on_ω (P : Point) (ω : Circle) : Prop :=
  (P.x - ω.center.x)^2 + (P.y - ω.center.y)^2 = ω.radius^2

-- Define circles ABP and PCD
noncomputable def circle_ABP (A B P : Point) : Circle := sorry
noncomputable def circle_PCD (P C D : Point) : Circle := sorry

-- Define Q as the common point of circles ABP and PCD distinct from P
def Q_is_common_point (A B C D P Q : Point) : Prop :=
  Q ≠ P ∧
  ((Q.x - (circle_ABP A B P).center.x)^2 + (Q.y - (circle_ABP A B P).center.y)^2 = (circle_ABP A B P).radius^2) ∧
  ((Q.x - (circle_PCD P C D).center.x)^2 + (Q.y - (circle_PCD P C D).center.y)^2 = (circle_PCD P C D).radius^2)

-- Main theorem
theorem locus_of_Q_is_circle_through_A_and_D :
  ∃ (locus : Circle), ∀ (P Q : Point),
    ω_passes_through_BC B C ω →
    P_on_ω P ω →
    Q_is_common_point A B C D P Q →
    ((Q.x - locus.center.x)^2 + (Q.y - locus.center.y)^2 = locus.radius^2) ∧
    ((A.x - locus.center.x)^2 + (A.y - locus.center.y)^2 = locus.radius^2) ∧
    ((D.x - locus.center.x)^2 + (D.y - locus.center.y)^2 = locus.radius^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_is_circle_through_A_and_D_l1108_110812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_theorem_l1108_110816

/-- The number of ways to arrange 6 people in two rows of 3 seats each, 
    where two specific individuals must sit in the same row and next to each other. -/
def seating_arrangements : ℕ := 192

/-- The number of people choosing seats. -/
def total_people : ℕ := 6

/-- The number of rows. -/
def num_rows : ℕ := 2

/-- The number of seats in each row. -/
def seats_per_row : ℕ := 3

/-- The number of possible positions for the two specific individuals to sit together. -/
def positions_for_pair : ℕ := 4

/-- The number of ways to arrange the two specific individuals next to each other. -/
def arrangements_of_pair : ℕ := 2

/-- The number of remaining people after placing the specific pair. -/
def remaining_people : ℕ := total_people - 2

theorem seating_theorem : 
  seating_arrangements = positions_for_pair * arrangements_of_pair * Nat.factorial remaining_people :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_theorem_l1108_110816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_is_18_l1108_110839

def points : List (ℚ × ℚ) := [(2, 9), (5, 15), (10, 25), (15, 30), (18, 55)]

noncomputable def is_above_line (point : ℚ × ℚ) : Bool :=
  point.2 > 2 * point.1 + 5

noncomputable def sum_x_above_line (points : List (ℚ × ℚ)) : ℚ :=
  (points.filter is_above_line).map (·.1) |>.sum

theorem sum_x_above_line_is_18 : sum_x_above_line points = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_is_18_l1108_110839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_zero_l1108_110849

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 
    Real.tan (2^(x^2 * Real.cos (1 / (8*x))) - 1 + x)
  else 
    0

-- State the theorem
theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_zero_l1108_110849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_abs_l1108_110865

-- Define the function f with domain [-2, 3]
def f : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem domain_of_f_abs (x : ℝ) : 
  x ∈ {y : ℝ | f (|y|)} ↔ x ∈ Set.Ioo (-3) 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_abs_l1108_110865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_ratio_l1108_110828

noncomputable section

-- Define the radius of the sphere
variable (r : ℝ)

-- Define the volume of the sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Define the volume of the cone
noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * (2*r)^2 * h

-- Define the ratio of cone volume to sphere volume
noncomputable def volume_ratio (r : ℝ) (h : ℝ) : ℝ := cone_volume r h / sphere_volume r

-- Define the ratio of cone height to base radius
noncomputable def height_base_ratio (r : ℝ) (h : ℝ) : ℝ := h / (2*r)

-- Theorem statement
theorem cone_sphere_ratio (r : ℝ) (h : ℝ) (hr : r > 0) :
  volume_ratio r h = 1/3 → height_base_ratio r h = 1/6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_ratio_l1108_110828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_minus_one_l1108_110870

theorem gcd_power_minus_one (a b n : ℕ) :
  Nat.gcd (n^a - 1) (n^b - 1) = n^(Nat.gcd a b) - 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_minus_one_l1108_110870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_difference_l1108_110876

theorem average_difference : ℤ := by
  -- Define the sets of integers
  let set1 := Finset.Icc 200 400
  let set2 := Finset.Icc 100 200

  -- Define the averages
  let avg1 : ℚ := (200 + 400) / 2
  let avg2 : ℚ := (100 + 200) / 2

  -- State the theorem
  have h : avg1 - avg2 = 150 := by
    -- Proof goes here
    sorry

  -- Convert the result to an integer
  exact ⌊avg1 - avg2⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_difference_l1108_110876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_three_pi_four_l1108_110843

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x - Real.pi / 2) + 2 * Real.cos x

theorem tangent_slope_at_three_pi_four :
  (deriv f) (3 * Real.pi / 4) = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_three_pi_four_l1108_110843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_rectangles_is_62_l1108_110847

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat
  x_valid : x ≤ 15
  y_valid : y ≤ 15
  x_multiple_of_5 : 5 ∣ x
  y_multiple_of_5 : 5 ∣ y

/-- Represents a rectangle on the grid -/
structure GridRectangle where
  topLeft : GridPoint
  bottomRight : GridPoint
  valid : topLeft.x < bottomRight.x ∧ topLeft.y > bottomRight.y

/-- The set of all valid grid points -/
def gridPoints : Finset GridPoint := sorry

/-- The set of all valid rectangles on the grid -/
def gridRectangles : Finset GridRectangle := sorry

/-- The number of rectangles that can be formed on the grid -/
def numRectangles : Nat := gridRectangles.card

theorem num_rectangles_is_62 : numRectangles = 62 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_rectangles_is_62_l1108_110847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_f_g_inequality_iff_l1108_110818

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def g (a x : ℝ) : ℝ := -x^2 - a*x - 4

-- Statement 1: The minimum value of f(x) for x > 0 is -1/e
theorem f_minimum : ∃ (x : ℝ), x > 0 ∧ f x = -(1 / Real.exp 1) ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x := by
  sorry

-- Statement 2: For all x > 0, f(x) > (1/3)g(x) if and only if a > -5
theorem f_g_inequality_iff (a : ℝ) : (∀ (x : ℝ), x > 0 → f x > (1/3) * g a x) ↔ a > -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_f_g_inequality_iff_l1108_110818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interchangeable_propositions_l1108_110890

-- Define the concept of a geometric object (line or plane)
inductive GeometricObject
| Line
| Plane
deriving DecidableEq

-- Define the concept of a geometric proposition
structure GeometricProposition :=
  (statement : GeometricObject → GeometricObject → GeometricObject → Prop)

-- Define the concept of an interchangeable proposition
def isInterchangeable (p : GeometricProposition) : Prop :=
  ∀ (x y z : GeometricObject), 
    p.statement x y z ↔ 
    p.statement (if x = GeometricObject.Line then GeometricObject.Plane else GeometricObject.Line)
                (if y = GeometricObject.Line then GeometricObject.Plane else GeometricObject.Line)
                (if z = GeometricObject.Line then GeometricObject.Plane else GeometricObject.Line)

-- Define the four propositions
def prop1 : GeometricProposition := 
  ⟨λ x y z ↦ x = GeometricObject.Line ∧ y = GeometricObject.Line ∧ z = GeometricObject.Plane⟩

def prop2 : GeometricProposition := 
  ⟨λ x y z ↦ x = GeometricObject.Plane ∧ y = GeometricObject.Plane ∧ z = GeometricObject.Plane⟩

def prop3 : GeometricProposition := 
  ⟨λ x y z ↦ x = GeometricObject.Line ∧ y = GeometricObject.Line ∧ z = GeometricObject.Line⟩

def prop4 : GeometricProposition := 
  ⟨λ x y z ↦ x = GeometricObject.Line ∧ y = GeometricObject.Line ∧ z = GeometricObject.Plane⟩

-- Theorem statement
theorem interchangeable_propositions :
  isInterchangeable prop1 ∧ 
  ¬isInterchangeable prop2 ∧ 
  isInterchangeable prop3 ∧ 
  ¬isInterchangeable prop4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interchangeable_propositions_l1108_110890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_PQ_l1108_110896

-- Define the point R
def R : ℝ × ℝ := (10, 7)

-- Define the lines
def line1 (x y : ℝ) : Prop := 6 * y = 11 * x
def line2 (x y : ℝ) : Prop := 7 * y = 2 * x

-- Define P and Q
axiom P : ℝ × ℝ
axiom Q : ℝ × ℝ

-- State that P is on line1
axiom P_on_line1 : line1 P.1 P.2

-- State that Q is on line2
axiom Q_on_line2 : line2 Q.1 Q.2

-- Define the midpoint condition
def is_midpoint (m p q : ℝ × ℝ) : Prop :=
  m.1 = (p.1 + q.1) / 2 ∧ m.2 = (p.2 + q.2) / 2

-- State that R is the midpoint of PQ
axiom R_midpoint_PQ : is_midpoint R P Q

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem length_of_PQ : distance P Q = 337824 / 1365 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_PQ_l1108_110896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l1108_110875

open Set

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x - 3 > 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | -1 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l1108_110875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydras_always_odd_hydras_never_equal_l1108_110864

/-- Represents the number of heads a hydra can grow in a week -/
inductive HeadGrowth where
  | five : HeadGrowth
  | seven : HeadGrowth

/-- The state of the hydras' heads after a certain number of weeks -/
structure HydraState where
  weeks : ℕ
  totalHeads : ℕ

/-- The initial state of the hydras -/
def initialState : HydraState :=
  { weeks := 0, totalHeads := 4033 }

/-- The weekly change in the total number of heads -/
def weeklyChange (h1 h2 : HeadGrowth) : ℕ :=
  match h1, h2 with
  | HeadGrowth.five, HeadGrowth.five => 6
  | HeadGrowth.five, HeadGrowth.seven => 8
  | HeadGrowth.seven, HeadGrowth.five => 8
  | HeadGrowth.seven, HeadGrowth.seven => 10

/-- The next state of the hydras after one week -/
def nextState (s : HydraState) (h1 h2 : HeadGrowth) : HydraState :=
  { weeks := s.weeks + 1,
    totalHeads := s.totalHeads + weeklyChange h1 h2 }

theorem hydras_always_odd :
  ∀ (s : HydraState) (h1 h2 : HeadGrowth),
    s.totalHeads % 2 = 1 →
    (nextState s h1 h2).totalHeads % 2 = 1 := by
  sorry

theorem hydras_never_equal :
  ∀ (n : ℕ) (h1 h2 : HeadGrowth),
    (Nat.iterate (λ s => nextState s h1 h2) n initialState).totalHeads % 2 = 1 := by
  sorry

#check hydras_always_odd
#check hydras_never_equal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydras_always_odd_hydras_never_equal_l1108_110864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twentyThree_is_eighth_term_l1108_110886

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n ↦ a + (n - 1) * d

theorem twentyThree_is_eighth_term :
  ∃ (a d : ℕ), 
    a = 2 ∧ 
    d = 3 ∧ 
    arithmeticSequence a d 8 = 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twentyThree_is_eighth_term_l1108_110886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_l1108_110857

/-- The number of small circles -/
def n : ℕ := 8

/-- The radius of each small circle -/
def r : ℝ := 4

/-- The side length of the inner octagon formed by the centers of the small circles -/
def s : ℝ := 2 * r

/-- The radius of the inner octagon -/
noncomputable def R_i : ℝ := s / (2 * Real.sin (Real.pi / n))

/-- The radius of the large circle -/
noncomputable def R : ℝ := R_i + r

/-- The diameter of the large circle -/
noncomputable def D : ℝ := 2 * R

theorem large_circle_diameter :
  ∃ ε > 0, |D - 28.92| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_l1108_110857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_inequalities_and_set_inclusion_l1108_110814

/-- Distance function d_α for two points in R² -/
noncomputable def d_α (α : ℝ) (A B : ℝ × ℝ) : ℝ :=
  ((|A.1 - B.1|^α + |A.2 - B.2|^α)^(1/α))

/-- Set D_α -/
def D_α (α : ℝ) : Set (ℝ × ℝ) :=
  {M | d_α α M (0, 0) ≤ 1}

theorem distance_inequalities_and_set_inclusion 
  (A B : ℝ × ℝ) (α β : ℝ) (h_pos_α : 0 < α) (h_pos_β : 0 < β) (h_α_lt_β : α < β) :
  (d_α 2 A B ≤ d_α 1 A B ∧ d_α 1 A B ≤ Real.sqrt 2 * d_α 2 A B) ∧ 
  (D_α α ⊆ D_α β) := by
  sorry

#check distance_inequalities_and_set_inclusion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_inequalities_and_set_inclusion_l1108_110814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l1108_110855

theorem cos_pi_minus_alpha (α : ℝ) : Real.cos (π - α) = -Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l1108_110855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1108_110873

noncomputable section

-- Define the given numbers
def given_numbers : List ℝ := [-8, -0.275, 22/7, 0, 10, -1.4040040004, -1/3, -2, Real.pi/3, 0.5]

-- Define the sets
def positive_set : Set ℝ := {x | x > 0}
def irrational_set : Set ℝ := {x | ¬ ∃ (p q : ℤ), q ≠ 0 ∧ x = ↑p / ↑q}
def integer_set : Set ℝ := {x | ∃ (n : ℤ), x = ↑n}
def negative_fraction_set : Set ℝ := {x | x < 0 ∧ ∃ (p q : ℤ), q ≠ 0 ∧ x = ↑p / ↑q}

-- Theorem statement
theorem number_categorization :
  (22/7 ∈ positive_set) ∧
  (10 ∈ positive_set) ∧
  (Real.pi/3 ∈ positive_set) ∧
  (0.5 ∈ positive_set) ∧
  (-1.4040040004 ∈ irrational_set) ∧
  (Real.pi/3 ∈ irrational_set) ∧
  (-8 ∈ integer_set) ∧
  (0 ∈ integer_set) ∧
  (10 ∈ integer_set) ∧
  (-2 ∈ integer_set) ∧
  (-0.275 ∈ negative_fraction_set) ∧
  (-1/3 ∈ negative_fraction_set) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1108_110873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transform_l1108_110801

/-- Represents the state of the four dials --/
def DialState := Fin 4 → Fin 3

/-- Calculates the sum of the digits in a DialState --/
def sum_digits (state : DialState) : Nat :=
  (state 0).val + (state 1).val + (state 2).val + (state 3).val

/-- Defines a valid transformation on adjacent dials --/
def valid_transform (before after : DialState) : Prop :=
  ∃ i : Fin 3, 
    (before i ≠ before (i + 1)) ∧ 
    (after i ≠ before i) ∧ (after i ≠ before (i + 1)) ∧
    (after (i + 1) = after i) ∧
    (∀ j : Fin 4, j ≠ i ∧ j ≠ i + 1 → after j = before j)

/-- Theorem: It's impossible to transform a state where the sum of digits
    is a multiple of 3 to a state where it's not, using valid transformations --/
theorem impossible_transform (initial final : DialState) :
  (sum_digits initial) % 3 = 0 →
  (sum_digits final) % 3 ≠ 0 →
  ¬∃ (sequence : List DialState), 
    sequence.head? = some initial ∧
    sequence.getLast? = some final ∧
    ∀ i j, i + 1 = j → valid_transform (sequence.get? i |>.getD initial) (sequence.get? j |>.getD final) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transform_l1108_110801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_difference_perfect_square_product_l1108_110821

theorem prime_difference_perfect_square_product (p : ℕ) (hp : Nat.Prime p) (hp2 : p > 2) :
  ∃ (a b k : ℕ),
    a > 0 ∧ 
    b > 0 ∧ 
    a - b = p ∧ 
    a * b = k ^ 2 ∧
    a = ((p + 1) ^ 2) / 4 ∧
    b = ((p - 1) ^ 2) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_difference_perfect_square_product_l1108_110821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_conversion_l1108_110802

/-- Converts speed from kilometers per hour to meters per second -/
noncomputable def kmph_to_ms (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

/-- Theorem: A speed of 135 kmph is equivalent to 37.5 m/s -/
theorem train_speed_conversion :
  kmph_to_ms 135 = 37.5 := by
  -- Unfold the definition of kmph_to_ms
  unfold kmph_to_ms
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- The rest of the proof
  sorry

-- Use #eval with rational numbers instead of real numbers
#eval (135 * 1000 : ℚ) / 3600

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_conversion_l1108_110802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_l1108_110813

/-- The slope of a line perpendicular to the line containing points (3, 5) and (-2, 4) is -5 -/
theorem perpendicular_slope : ∃ (m : ℝ), m = -5 ∧ m * ((4 - 5) / (-2 - 3)) = -1 := by
  let point1 : ℝ × ℝ := (3, 5)
  let point2 : ℝ × ℝ := (-2, 4)
  let slope : ℝ := (point2.2 - point1.2) / (point2.1 - point1.1)
  let perpendicular_slope : ℝ := -1 / slope
  
  use perpendicular_slope
  apply And.intro
  · -- Prove that perpendicular_slope = -5
    sorry
  · -- Prove that perpendicular_slope * slope = -1
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_l1108_110813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intern_teacher_assignment_l1108_110881

theorem intern_teacher_assignment (n : ℕ) (k : ℕ) :
  n = 4 ∧ k = 3 →
  (Finset.card (Finset.filter (Function.Surjective) (Finset.univ : Finset (Fin n → Fin k)))) = 36 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intern_teacher_assignment_l1108_110881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_tan_one_angle_pi_third_implies_x_value_l1108_110895

-- Define the vectors m and n
noncomputable def m : ℝ × ℝ := (Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

-- Define the condition that x is in the open interval (0, π/2)
def x_in_range (x : ℝ) : Prop := 0 < x ∧ x < Real.pi / 2

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define perpendicularity of two vectors
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Define the angle between two vectors
noncomputable def angle_between (v w : ℝ × ℝ) : ℝ := 
  Real.arccos (dot_product v w / (Real.sqrt (dot_product v v) * Real.sqrt (dot_product w w)))

-- Theorem 1
theorem perpendicular_implies_tan_one (x : ℝ) (h : x_in_range x) :
  perpendicular m (n x) → Real.tan x = 1 := by sorry

-- Theorem 2
theorem angle_pi_third_implies_x_value (x : ℝ) (h : x_in_range x) :
  angle_between m (n x) = Real.pi / 3 → x = 5 * Real.pi / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_tan_one_angle_pi_third_implies_x_value_l1108_110895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_triangle_area_bound_l1108_110858

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

structure Quadrilateral where
  P₁ : ℝ × ℝ
  P₂ : ℝ × ℝ
  P₃ : ℝ × ℝ
  P₄ : ℝ × ℝ

noncomputable def area (t : Triangle) : ℝ := sorry

noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

def onSide (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

theorem quadrilateral_triangle_area_bound 
  (abc : Triangle) (quad : Quadrilateral) 
  (h₁ : onSide quad.P₁ abc) (h₂ : onSide quad.P₂ abc) 
  (h₃ : onSide quad.P₃ abc) (h₄ : onSide quad.P₄ abc) :
  min (min (triangleArea quad.P₁ quad.P₂ quad.P₃) (triangleArea quad.P₁ quad.P₂ quad.P₄))
      (min (triangleArea quad.P₁ quad.P₃ quad.P₄) (triangleArea quad.P₂ quad.P₃ quad.P₄))
  ≤ (1/4 : ℝ) * area abc := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_triangle_area_bound_l1108_110858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_maintenance_cost_l1108_110869

def monthly_pool_cost (cleaning_interval : ℕ) (cleaning_cost : ℕ) (tip_percentage : ℚ)
  (chemical_cost : ℕ) (chemical_frequency : ℕ) (days_in_month : ℕ) : ℕ :=
  let cleanings_per_month := days_in_month / cleaning_interval
  let total_cleaning_cost := cleanings_per_month * (cleaning_cost + ⌊cleaning_cost * tip_percentage⌋)
  let total_chemical_cost := chemical_cost * chemical_frequency
  (total_cleaning_cost + total_chemical_cost).toNat

theorem pool_maintenance_cost :
  monthly_pool_cost 3 150 (1/10) 200 2 30 = 2050 := by
  sorry

#eval monthly_pool_cost 3 150 (1/10) 200 2 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_maintenance_cost_l1108_110869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1108_110874

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def satisfies_conditions (t : Triangle) : Prop :=
  2 * t.b * (Real.cos t.A) = t.a * (Real.cos t.C) + t.c * (Real.cos t.A) ∧
  t.a = 2 ∧
  t.b + t.c = 4

-- Helper function to calculate triangle area
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * (Real.sin t.A)

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.A = Real.pi / 3 ∧ 
  triangle_area t = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1108_110874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_conditions_l1108_110878

/-- Quadratic function f(x) = -x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- Sequence a_n defined recursively -/
def a (b c : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f b c (a b c n)

/-- Theorem stating the conditions for monotonicity of sequence a_n -/
theorem monotonicity_conditions (b c : ℝ) 
  (h1 : ∀ x, f b c x = f b c (1 - x)) : 
  ((∀ n, a b c (n + 1) ≤ a b c n) ↔ c < 0) ∧ 
  ((∀ n, a b c n < a b c (n + 1)) ↔ c > 1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_conditions_l1108_110878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l1108_110854

/-- The time (in seconds) for a train to pass a bridge -/
noncomputable def train_pass_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600)

/-- Theorem: A train of length 360 meters, traveling at 72 km/hour, 
    passes a bridge of length 140 meters in 25 seconds -/
theorem train_bridge_passing_time :
  train_pass_time 360 140 72 = 25 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_pass_time 360 140 72

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l1108_110854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_theorem_l1108_110861

/-- Represents a rectangle with short side 'a' and long side 'b' -/
structure Rectangle where
  a : ℝ
  b : ℝ
  h1 : 0 < a
  h2 : 0 < b

/-- The diagonal of a rectangle -/
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ := Real.sqrt (r.a^2 + r.b^2)

/-- The perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.a + r.b)

/-- The theorem stating that if the ratio of the long side to the diagonal
    equals the ratio of the diagonal to half the perimeter, then b/a = 1 -/
theorem rectangle_ratio_theorem (r : Rectangle) :
  r.b / r.diagonal = r.diagonal / (r.perimeter / 2) → r.b / r.a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_theorem_l1108_110861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_exponent_representation_l1108_110811

theorem fractional_exponent_representation (a : ℝ) (ha : 0 < a) :
  Real.sqrt (a * (a * Real.sqrt a) ^ (1/3)) = a ^ (3/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_exponent_representation_l1108_110811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_value_l1108_110884

noncomputable def g (x : ℝ) : ℝ := x + x / (x^2 + 2) + x * (x + 5) / (x^2 + 3) + 3 * (x + 3) / (x * (x^2 + 3))

theorem g_minimum_value (x : ℝ) (h : x > 0) : g x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_value_l1108_110884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euclidean_algorithm_steps_bound_l1108_110880

/-- 
Given:
- m₁ is a natural number represented in the decimal system using n digits
- m₀ is any natural number
- k is the number of steps in the Euclidean algorithm for m₀ and m₁

Prove that k ≤ 5n
-/
theorem euclidean_algorithm_steps_bound 
  (m₀ m₁ n k : ℕ) 
  (h_m₁_digits : (Nat.log 10 m₁ + 1 : ℕ) = n) 
  (h_k_steps : k = Nat.gcdA m₀ m₁) : 
  k ≤ 5 * n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_euclidean_algorithm_steps_bound_l1108_110880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_sin_cos_inequality_l1108_110872

theorem max_n_sin_cos_inequality : 
  ∀ n : ℕ, n > 0 → 
  (∀ x : ℝ, Real.sin x ^ n + Real.cos x ^ n ≥ 1 / (n : ℝ)) ↔ 
  n ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_sin_cos_inequality_l1108_110872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_a_exist_l1108_110867

-- Define the function g
noncomputable def g (a b : ℝ) : ℝ := a * Real.sqrt b - (1/4) * b

-- Theorem statement
theorem infinitely_many_a_exist :
  ∃ S : Set ℝ, (Set.Infinite S) ∧ 
  (∀ (a : ℝ), a ∈ S → ∀ (b : ℝ), b > 0 → g a 4 ≥ g a b) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_a_exist_l1108_110867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_four_circles_l1108_110887

/-- The area of the shaded region formed by the intersection of four circles --/
noncomputable def shaded_area (r : ℝ) (n : ℕ) : ℝ :=
  n * (Real.pi * r^2 / 4 - r^2 / 2)

/-- Theorem: The area of the shaded region formed by the intersection of four circles
    with radius 6 units, intersecting at the origin and forming 4 checkered regions,
    is equal to 36π - 72 square units. --/
theorem shaded_area_four_circles :
  shaded_area 6 4 = 36 * Real.pi - 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_four_circles_l1108_110887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_point_partition_l1108_110835

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on one side of a line -/
noncomputable def onOneSide (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c > 0

/-- Check if a line separates two sets of points -/
def separates (l : Line) (s1 s2 : Set Point) : Prop :=
  (∀ p ∈ s1, onOneSide p l) ∧ (∀ p ∈ s2, ¬onOneSide p l)

/-- Main theorem: For any four points in a plane, there exists a partition
    into two non-empty subsets such that no line can separate them -/
theorem four_point_partition (p1 p2 p3 p4 : Point) :
  ∃ (s1 s2 : Set Point),
    s1 ∪ s2 = {p1, p2, p3, p4} ∧
    s1 ≠ ∅ ∧ s2 ≠ ∅ ∧
    s1 ∩ s2 = ∅ ∧
    ∀ l : Line, ¬separates l s1 s2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_point_partition_l1108_110835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_85_l1108_110809

def scores : List ℕ := [62, 67, 73, 75, 85, 92]

def is_integer_average (partial_scores : List ℕ) : Prop :=
  ∀ k : ℕ, k ≤ partial_scores.length → (partial_scores.take k).sum % k = 0

theorem last_score_is_85 :
  ∃! x : ℕ, x ∈ scores ∧ is_integer_average (scores.filter (· ≠ x) ++ [x]) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_85_l1108_110809
