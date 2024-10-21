import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_plate_suspension_point_l98_9886

/-- Given a square plate with 20 cm sides, the distance from the optimal suspension point
    on the perimeter to the nearest vertex is 5.47 cm, with a precision of 0.1 mm. -/
theorem square_plate_suspension_point (a : ℝ) (h : a = 10) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ a ∧ 
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ a → 
    a * (1 - Real.tan (Real.arctan (x/a))) * Real.sin (Real.arctan (x/a)) ≥ 
    a * (1 - Real.tan (Real.arctan (y/a))) * Real.sin (Real.arctan (y/a))) ∧
  (a - x) = 5.47 := by
  sorry

#check square_plate_suspension_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_plate_suspension_point_l98_9886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_implies_a_value_l98_9867

noncomputable def curve (x : ℝ) : ℝ := Real.sqrt (x^2 + 1)

def circleC (a : ℝ) (x y : ℝ) : Prop := x^2 + (y - a)^2 = 3/4

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem min_distance_implies_a_value (a : ℝ) : 
  (a > 0) →
  (∀ x ∈ Set.Icc (-1 : ℝ) (Real.sqrt 3), 
    ∀ x' y' : ℝ, circleC a x' y' → 
      distance x (curve x) x' y' ≥ 3/2 * Real.sqrt 3) →
  (∃ x ∈ Set.Icc (-1 : ℝ) (Real.sqrt 3), 
    ∃ x' y' : ℝ, circleC a x' y' ∧ 
      distance x (curve x) x' y' = 3/2 * Real.sqrt 3) →
  a = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_implies_a_value_l98_9867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_root_relation_l98_9860

-- Define the relationship between z, x, and y
noncomputable def z_relation (k : ℝ) (x y : ℝ) : ℝ := k * y / Real.sqrt x

-- Theorem statement
theorem inverse_square_root_relation (k : ℝ) :
  z_relation k 4 3 = 6 → z_relation k 9 6 = 8 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_square_root_relation_l98_9860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_neg_inv_ln2_l98_9874

/-- The function f(x) = x * 2^x -/
noncomputable def f (x : ℝ) : ℝ := x * (2 : ℝ)^x

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := (2 : ℝ)^x + x * (2 : ℝ)^x * Real.log 2

theorem f_min_at_neg_inv_ln2 :
  ∃ (x : ℝ), x = -1 / Real.log 2 ∧ 
  (∀ (y : ℝ), f y ≥ f x) ∧
  (f' x = 0) := by
  sorry

#check f_min_at_neg_inv_ln2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_neg_inv_ln2_l98_9874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_area_theorem_sin_3x_area_l98_9875

/-- The area of a function f on an interval [a, b] -/
noncomputable def area (f : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

/-- The sine function -/
noncomputable def sin_nx (n : ℕ+) (x : ℝ) : ℝ := Real.sin (n * x)

theorem sin_area_theorem (n : ℕ+) :
  area (sin_nx n) 0 (Real.pi / n) = 2 / n := by sorry

theorem sin_3x_area :
  area (sin_nx 3) 0 ((2 : ℝ) * Real.pi / 3) = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_area_theorem_sin_3x_area_l98_9875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l98_9899

open Real

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * log x

-- Theorem statement
theorem f_properties :
  (∀ x, x > 0 → f x ≥ x - 1) ∧
  (∀ t : ℝ, ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → f x₁ = t → f x₂ = t → x₁ + x₂ > 2 / Real.exp 1) ∧
  (∃ a : ℝ, a = -(Real.exp 1)^3 ∧ ∀ x, x > 0 → f x ≥ a * x^2 + 2 / a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l98_9899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_condition_l98_9844

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

noncomputable def angle (x y : n) : ℝ := Real.arccos ((inner x y) / (norm x * norm y))

theorem vector_equality_condition (a b : n) (ha : a ≠ 0) (hb : b ≠ 0) :
  norm a = norm b ∧ angle a b = 0 → a = b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_condition_l98_9844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_union_N_eq_M_l98_9847

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 = 1 ∧ p.1 > 0}

def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | Real.arctan p.1 + Real.arctan (1 / p.2) = Real.pi}

theorem M_union_N_eq_M : M ∪ N = M := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_union_N_eq_M_l98_9847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_prime_divisors_l98_9855

/-- A function f: ℕ → ℕ satisfying the given properties -/
def special_function (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, ∃ k : ℤ, (f b : ℤ) - (f a : ℤ) = k * ((b : ℤ) - (a : ℤ))

/-- The set of prime numbers that divide f(c) for at least one natural number c -/
def prime_divisors (f : ℕ → ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ c : ℕ, p ∣ f c}

theorem infinite_prime_divisors (f : ℕ → ℕ) 
  (h_non_constant : ∃ x y : ℕ, f x ≠ f y) 
  (h_special : special_function f) : 
  Set.Infinite (prime_divisors f) := by
  sorry

#check infinite_prime_divisors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_prime_divisors_l98_9855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_H_upper_bound_l98_9865

open Real

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x + Real.log x

noncomputable def H (x m : ℝ) : ℝ := x + Real.log x - Real.log (Real.exp x - 1)

-- State the theorem
theorem H_upper_bound {m : ℝ} (hm : m > 0) :
  ∀ x, 0 < x ∧ x < m → H x m < m / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_H_upper_bound_l98_9865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_hiking_distance_l98_9853

theorem billy_hiking_distance (eastward_distance : ℝ) (northeast_distance : ℝ) 
  (h1 : eastward_distance = 5)
  (h2 : northeast_distance = 8)
  (h3 : Real.sqrt 2 * (northeast_distance / Real.sqrt 2) = northeast_distance) :
  Real.sqrt (eastward_distance^2 + northeast_distance^2 + 
    2 * eastward_distance * (northeast_distance / Real.sqrt 2)) = 
  Real.sqrt (89 + 40 * Real.sqrt 2) := by
  sorry

#check billy_hiking_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_hiking_distance_l98_9853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DTO_measure_l98_9825

-- Define the triangle DOG
structure Triangle (D G O : Point) : Prop where
  -- Add any necessary conditions for a valid triangle
  valid : true -- Placeholder for valid triangle conditions

-- Define the angle measure in degrees
noncomputable def angle_measure (A B C : Point) : ℝ := sorry

-- State the theorem
theorem angle_DTO_measure
  (D G O T : Point)
  (tri : Triangle D G O)
  (h1 : angle_measure D G O = angle_measure D O G)
  (h2 : angle_measure G O D = 30)
  (h3 : angle_measure D O T = (angle_measure D O G) / 2) :
  angle_measure D T O = 67.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DTO_measure_l98_9825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_value_log_inequality_solution_a_gt_1_log_inequality_solution_a_lt_1_l98_9887

-- Define the logarithmic function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem for part 1
theorem log_base_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 8 = 3 → a = 2 := by sorry

-- Theorem for part 2 when a > 1
theorem log_inequality_solution_a_gt_1 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a > 1) :
  {x : ℝ | f a x ≤ f a (2 - 3*x)} = {x : ℝ | 0 < x ∧ x ≤ 1/2} := by sorry

-- Theorem for part 2 when 0 < a < 1
theorem log_inequality_solution_a_lt_1 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a < 1) :
  {x : ℝ | f a x ≤ f a (2 - 3*x)} = {x : ℝ | 1/2 ≤ x ∧ x < 2/3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_value_log_inequality_solution_a_gt_1_log_inequality_solution_a_lt_1_l98_9887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_constant_max_sum_of_squared_distances_l98_9803

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Define the line L
def line_L (k : ℝ) (x y : ℝ) : Prop := y = k * x ∧ k > 0

-- Define the intersection points
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ circle_C x y ∧ line_L k x y}

-- Theorem 1: Sum of reciprocals of x-coordinates is constant
theorem sum_of_reciprocals_constant (k : ℝ) :
  ∀ p q, p ∈ intersection_points k → q ∈ intersection_points k → p.1 ≠ 0 → q.1 ≠ 0 →
  1 / p.1 + 1 / q.1 = 2 / 3 :=
sorry

-- Theorem 2: Maximum value of |PN|² + |QN|²
theorem max_sum_of_squared_distances :
  ∃ (k : ℝ), ∀ p q, p ∈ intersection_points k → q ∈ intersection_points k →
  (p.1 - 2)^2 + (p.2 - 1)^2 + (q.1 - 2)^2 + (q.2 - 1)^2 ≤ 2 * Real.sqrt 10 + 22 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_constant_max_sum_of_squared_distances_l98_9803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_lengths_l98_9821

/-- The volume of a regular tetrahedron with edge length s -/
noncomputable def tetrahedronVolume (s : ℝ) : ℝ := s^3 / (12 * Real.sqrt 2)

/-- Given two regular tetrahedrons with edge lengths x and y, 
    if their sum of edge lengths is a and sum of volumes is b,
    then x and y can be expressed in terms of a and b -/
theorem tetrahedron_edge_lengths (a b x y : ℝ) 
  (h1 : x + y = a) 
  (h2 : tetrahedronVolume x + tetrahedronVolume y = b) :
  x = (1/2) * (a + Real.sqrt ((16*b*Real.sqrt 2)/a - a^2/3)) ∧ 
  y = (1/2) * (a - Real.sqrt ((16*b*Real.sqrt 2)/a - a^2/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_lengths_l98_9821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_divisor_remainder_l98_9892

theorem dividend_divisor_remainder (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x % y = 8 →
  (x : ℝ) / (y : ℝ) = 96.16 →
  y = 50 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_divisor_remainder_l98_9892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_line_and_circle_l98_9845

/-- The line l in Cartesian coordinates -/
def line (x y : ℝ) : Prop := x + y = 4

/-- The circle C in Cartesian coordinates -/
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

/-- The minimum distance between a point on the line and a point on the circle -/
noncomputable def min_distance : ℝ := Real.sqrt 2 - 1

/-- Theorem stating that the minimum distance between a point on the line and a point on the circle is √2 - 1 -/
theorem min_distance_between_line_and_circle :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), line x₁ y₁ → circle_eq x₂ y₂ →
  ∀ (d : ℝ), d = Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) →
  d ≥ min_distance :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_line_and_circle_l98_9845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_plus_2x_equals_e_l98_9898

/-- The definite integral of e^x + 2x from 0 to 1 equals e -/
theorem integral_exp_plus_2x_equals_e : ∫ x in (0:ℝ)..(1:ℝ), (Real.exp x + 2 * x) = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_plus_2x_equals_e_l98_9898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_fraction_at_event_l98_9883

/-- Represents a school with a given number of students and ratio of boys to girls -/
structure School where
  students : ℕ
  boys_ratio : ℕ
  girls_ratio : ℕ

/-- Calculates the number of girls in a school -/
def girls_count (s : School) : ℕ :=
  s.students * s.girls_ratio / (s.boys_ratio + s.girls_ratio)

/-- The three schools in the problem -/
def maplewood : School := ⟨320, 3, 5⟩
def brookfield : School := ⟨240, 5, 3⟩
def pinehurst : School := ⟨400, 1, 1⟩

/-- The combined event with all students -/
def combined_event : List School := [maplewood, brookfield, pinehurst]

/-- Theorem: The fraction of girls at the combined event is 35/69 -/
theorem girls_fraction_at_event :
  (combined_event.map girls_count).sum * 69 = 
  (combined_event.map (·.students)).sum * 35 := by
  sorry

#eval (combined_event.map girls_count).sum
#eval (combined_event.map (·.students)).sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_fraction_at_event_l98_9883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_value_l98_9824

/-- The line equation forming a triangle with coordinate axes -/
def line_equation (x y : ℝ) : Prop := 9 * x + 7 * y = 63

/-- The triangle formed by the line and coordinate axes -/
structure Triangle where
  x_intercept : ℝ
  y_intercept : ℝ
  hypotenuse : ℝ
  area : ℝ

/-- The triangle satisfies the line equation -/
axiom triangle_satisfies_equation (t : Triangle) :
  line_equation t.x_intercept 0 ∧ line_equation 0 t.y_intercept

/-- The triangle's area is correct -/
axiom triangle_area (t : Triangle) : t.area = (1/2) * t.x_intercept * t.y_intercept

/-- The triangle's hypotenuse is correct -/
axiom triangle_hypotenuse (t : Triangle) : 
  t.hypotenuse^2 = t.x_intercept^2 + t.y_intercept^2

/-- Function to calculate the sum of altitudes -/
noncomputable def sum_of_altitudes (t : Triangle) : ℝ :=
  t.x_intercept + t.y_intercept + (2 * t.area / t.hypotenuse)

/-- Theorem: The sum of altitudes is equal to 2695/130 -/
theorem sum_of_altitudes_value (t : Triangle) :
  sum_of_altitudes t = 2695/130 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_value_l98_9824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ostap_max_position_1002nd_round_ostap_max_position_1001st_round_l98_9826

/-- Represents the state of an election round -/
structure ElectionRound where
  candidates : ℕ
  ostap_position : ℕ
  round : ℕ

/-- Defines the election process -/
def election_process (initial_candidates : ℕ) (initial_ostap_position : ℕ) : ElectionRound → Prop :=
  sorry

/-- Ostap wins the election -/
def ostap_wins (er : ElectionRound) : Prop :=
  sorry

theorem ostap_max_position_1002nd_round :
  ∀ k : ℕ,
    k ≤ 2002 →
    (∃ er : ElectionRound,
      election_process 2002 k er ∧
      er.round = 1002 ∧
      ostap_wins er) →
    k ≤ 2001 :=
by
  sorry

theorem ostap_max_position_1001st_round :
  ∀ k : ℕ,
    k ≤ 2002 →
    (∃ er : ElectionRound,
      election_process 2002 k er ∧
      er.round = 1001 ∧
      ostap_wins er) →
    k = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ostap_max_position_1002nd_round_ostap_max_position_1001st_round_l98_9826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_configuration_exists_l98_9839

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A configuration of 6 points satisfying the problem conditions -/
structure Configuration where
  points : Fin 6 → Point
  distance_condition : ∀ i : Fin 6, ∃! (s : Finset (Fin 6)), s.card = 3 ∧ ∀ j ∈ s, distance (points i) (points j) = 1
  equilateral_triangles : ∃ (i j k l m n : Fin 6), 
    distance (points i) (points j) = 1 ∧
    distance (points j) (points k) = 1 ∧
    distance (points k) (points i) = 1 ∧
    distance (points l) (points m) = 1 ∧
    distance (points m) (points n) = 1 ∧
    distance (points n) (points l) = 1 ∧
    distance (points i) (points l) = 1 ∧
    distance (points j) (points m) = 1 ∧
    distance (points k) (points n) = 1

/-- Theorem stating the existence of a configuration satisfying the problem conditions -/
theorem configuration_exists : ∃ c : Configuration, True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_configuration_exists_l98_9839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l98_9814

/-- Given vectors in ℝ² -/
def OA : ℝ × ℝ := (1, 1)
def OB : ℝ × ℝ := (3, -1)
def OC (m : ℝ) : ℝ × ℝ := (m, 3)
def OD (x y : ℝ) : ℝ × ℝ := (x, y)

/-- Vector subtraction -/
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Magnitude of a vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- Collinearity condition -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  let AB := vector_sub B A
  let AC := vector_sub C A
  AB.1 * AC.2 = AB.2 * AC.1

/-- Rectangle condition -/
def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  let AB := vector_sub B A
  let BC := vector_sub C B
  dot_product AB BC = 0 ∧ vector_sub D A = vector_sub B C

/-- Cosine of angle between two vectors -/
noncomputable def cos_angle (v w : ℝ × ℝ) : ℝ :=
  dot_product v w / (magnitude v * magnitude w)

theorem vector_problem :
  (∀ m : ℝ, collinear OA OB (OC m) → m = -1) ∧
  (∀ x y : ℝ, is_rectangle OA OB (OC 7) (OD x y) →
    cos_angle OB (OD x y) = Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l98_9814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l98_9822

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.cos (ω * x + Real.pi / 3)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * x + φ) + 1

theorem f_range (ω : ℝ) (h_ω : ω > 0) :
  ∃ φ, (∀ x, ∃ y, f ω x = g φ y) →
  (Set.range (f ω) ∩ Set.Icc 0 (Real.pi / 3) = Set.Icc (-3) (3 / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l98_9822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l98_9835

/-- Given two trains approaching each other, calculate the length of the second train -/
theorem second_train_length 
  (length_train1 : ℝ)
  (speed_train1 : ℝ)
  (speed_train2 : ℝ)
  (clear_time : ℝ)
  (h1 : length_train1 = 100)
  (h2 : speed_train1 = 42 * 1000 / 3600)
  (h3 : speed_train2 = 30 * 1000 / 3600)
  (h4 : clear_time = 18.998480121590273)
  : ℝ :=
by
  -- Calculate the relative speed
  let relative_speed := speed_train1 + speed_train2
  
  -- Calculate the total distance traveled
  let total_distance := relative_speed * clear_time
  
  -- Calculate the length of the second train
  let length_train2 := total_distance - length_train1
  
  -- Return the result
  exact length_train2

-- Example usage (commented out to avoid evaluation issues)
-- #eval second_train_length 100 (42 * 1000 / 3600) (30 * 1000 / 3600) 18.998480121590273

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l98_9835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_2_to_100_l98_9896

theorem digits_of_2_to_100 : ⌊Real.log 2^100 / Real.log 10⌋ + 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_2_to_100_l98_9896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_menus_is_4096_l98_9846

/-- Represents the days of the week -/
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

/-- Represents the dessert options -/
inductive Dessert
| Cake | Pie | IceCream | Pudding | Cookies

/-- A dessert menu for a week -/
def WeekMenu := Day → Dessert

/-- The next day of the week -/
def nextDay : Day → Day
| Day.Sunday => Day.Monday
| Day.Monday => Day.Tuesday
| Day.Tuesday => Day.Wednesday
| Day.Wednesday => Day.Thursday
| Day.Thursday => Day.Friday
| Day.Friday => Day.Saturday
| Day.Saturday => Day.Sunday

/-- Checks if a menu is valid according to the rules -/
def is_valid_menu (menu : WeekMenu) : Prop :=
  (∀ d : Day, d ≠ Day.Saturday → menu d ≠ menu (nextDay d)) ∧
  (menu Day.Wednesday = Dessert.Pie)

/-- The number of valid dessert menus for a week -/
def num_valid_menus : ℕ := sorry

/-- The main theorem stating the number of valid dessert menus -/
theorem num_valid_menus_is_4096 : num_valid_menus = 4096 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_menus_is_4096_l98_9846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_chair_subsets_l98_9894

/-- The number of chairs in the circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets of adjacent chairs of a given size -/
def subsetsOfSize (size : ℕ) : ℕ := n

/-- The total number of subsets with at least 3 and at most 7 adjacent chairs -/
def totalSubsets : ℕ := (List.range 5).map (fun i => subsetsOfSize (i + 3)) |>.sum

/-- Theorem stating that the total number of subsets is 60 -/
theorem adjacent_chair_subsets : totalSubsets = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_chair_subsets_l98_9894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_c_prime_concentric_different_radius_l98_9881

-- Define the circle C
def CircleC (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b r : ℝ), ∀ (x y : ℝ), f x y = 0 ↔ (x - a)^2 + (y - b)^2 = r^2

-- Define the point P
def PointP (x : ℝ) : ℝ × ℝ := (x, x)

-- Define that P is not on C and not at its center
def PNotOnCOrCenter (f : ℝ → ℝ → ℝ) (x : ℝ) : Prop :=
  f x x ≠ 0 ∧ ∃ (a b r : ℝ), (∀ (x y : ℝ), f x y = 0 ↔ (x - a)^2 + (y - b)^2 = r^2) ∧ (x ≠ a ∨ x ≠ b)

-- Define the circle C'
def CircleC' (f : ℝ → ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ (a b r : ℝ), ∀ (x y : ℝ), f x y - f x x = 0 ↔ (x - a)^2 + (y - b)^2 = r^2

-- State the theorem
theorem circle_c_prime_concentric_different_radius 
  (f : ℝ → ℝ → ℝ) (x : ℝ) 
  (hC : CircleC f) 
  (hP : PNotOnCOrCenter f x) 
  (hC' : CircleC' f x) : 
  ∃ (a b : ℝ), ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧
  (∀ (x y : ℝ), f x y = 0 ↔ (x - a)^2 + (y - b)^2 = r₁^2) ∧
  (∀ (x y : ℝ), f x y - f x x = 0 ↔ (x - a)^2 + (y - b)^2 = r₂^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_c_prime_concentric_different_radius_l98_9881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_select_AB_3_out_of_5_l98_9869

/-- The number of students -/
def n : ℕ := 5

/-- The number of students to be selected -/
def k : ℕ := 3

/-- The probability of selecting both A and B when choosing k students out of n -/
def prob_select_AB (n k : ℕ) : ℚ :=
  (Nat.choose (n - 2) (k - 2)) / (Nat.choose n k)

/-- Theorem: The probability of selecting both A and B when choosing 3 students out of 5 is 3/10 -/
theorem prob_select_AB_3_out_of_5 : 
  prob_select_AB n k = 3 / 10 := by
  -- Unfold the definition of prob_select_AB
  unfold prob_select_AB
  -- Substitute the values of n and k
  simp [n, k]
  -- Evaluate the Nat.choose expressions
  norm_num
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_select_AB_3_out_of_5_l98_9869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scrap_cookie_radius_l98_9868

/-- The radius of the original cookie dough circle -/
def R : ℝ := 8

/-- The radius of each small cookie -/
def r : ℝ := 2

/-- The number of small cookies cut from the large circle -/
def n : ℕ := 9

/-- The radius of the scrap cookie -/
noncomputable def scrap_radius : ℝ := 2 * Real.sqrt 7

/-- Theorem stating that the radius of the scrap cookie is 2√7 -/
theorem scrap_cookie_radius : 
  π * scrap_radius ^ 2 = π * R ^ 2 - n * π * r ^ 2 := by
  sorry

#eval R
#eval r
#eval n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scrap_cookie_radius_l98_9868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l98_9877

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/3) + Real.sin x

/-- The shifted function g(x) -/
noncomputable def g (a x : ℝ) : ℝ := f (x + a)

/-- Theorem stating the minimum value of a for y-axis symmetry -/
theorem min_shift_for_symmetry :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ x : ℝ, g a x = g a (-x)) ∧
  (∀ b : ℝ, b > 0 → (∀ x : ℝ, g b x = g b (-x)) → a ≤ b) ∧
  a = Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l98_9877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_2001_balloons_l98_9830

/-- The largest prime divisor of a positive integer n -/
def largestPrimeDivisor (n : ℕ) : ℕ :=
  (Nat.factors n).maximum?.getD 1

/-- A function that represents the process of equalizing balloon sizes -/
def canEqualizeAllBalloons (n k : ℕ) : Prop :=
  ∀ (sizes : Fin n → ℚ), ∃ (equalizedSize : ℚ), 
    ∃ (steps : ℕ), ∀ (i : Fin n), 
      ∃ (finalSize : ℚ), finalSize = equalizedSize

theorem smallest_k_for_2001_balloons :
  ∃ (k : ℕ), k > 0 ∧ canEqualizeAllBalloons 2001 k ∧
  (∀ (k' : ℕ), k' > 0 → canEqualizeAllBalloons 2001 k' → k ≤ k') ∧
  k = largestPrimeDivisor 2001 := by
  sorry

#eval largestPrimeDivisor 2001

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_2001_balloons_l98_9830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_mapping_l98_9866

-- Define the payment technologies
inductive PaymentTech
| Chip
| MagneticStripe
| Paypass
| CVC

-- Define the actions
inductive PaymentAction
| Tap
| PayOnline
| Swipe
| InsertIntoTerminal

-- Define the mapping function
def tech_to_action : PaymentTech → PaymentAction
| PaymentTech.Chip => PaymentAction.InsertIntoTerminal
| PaymentTech.MagneticStripe => PaymentAction.Swipe
| PaymentTech.Paypass => PaymentAction.Tap
| PaymentTech.CVC => PaymentAction.PayOnline

-- State the theorem
theorem correct_mapping :
  (tech_to_action PaymentTech.Chip = PaymentAction.InsertIntoTerminal) ∧
  (tech_to_action PaymentTech.MagneticStripe = PaymentAction.Swipe) ∧
  (tech_to_action PaymentTech.Paypass = PaymentAction.Tap) ∧
  (tech_to_action PaymentTech.CVC = PaymentAction.PayOnline) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_mapping_l98_9866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_formula_l98_9890

/-- The perimeter of a semicircle in centimeters -/
def semicircle_perimeter : ℝ := 113

/-- The radius of a semicircle given its perimeter -/
noncomputable def semicircle_radius (p : ℝ) : ℝ := p / (Real.pi + 2)

/-- Theorem: The radius of a semicircle with perimeter 113 cm is equal to 113 / (π + 2) cm -/
theorem semicircle_radius_formula :
  semicircle_radius semicircle_perimeter = 113 / (Real.pi + 2) := by
  -- Unfold the definitions
  unfold semicircle_radius semicircle_perimeter
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_formula_l98_9890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_sphere_with_radius_one_third_l98_9833

/-- The volume of a sphere given its diameter -/
noncomputable def sphere_volume (d : ℝ) : ℝ := (4 / 3) * Real.pi * (d / 2) ^ 3

/-- The diameter of a sphere given its volume according to the ancient Chinese method -/
noncomputable def sphere_diameter (V : ℝ) : ℝ := (16 * V / 9) ^ (1 / 3 : ℝ)

theorem volume_of_sphere_with_radius_one_third :
  let r : ℝ := 1 / 3
  let d : ℝ := 2 * r
  let V : ℝ := sphere_volume d
  V = 1 / 6 ∧ d = sphere_diameter V := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_sphere_with_radius_one_third_l98_9833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_crown_distribution_l98_9837

/-- Represents the amount of bread each person has initially -/
structure BreadDistribution where
  person_a : ℚ
  person_b : ℚ

/-- Represents the final distribution of crowns -/
structure CrownDistribution where
  person_a : ℚ
  person_b : ℚ
deriving Repr

/-- Calculates the fair distribution of crowns based on initial bread amounts -/
def calculate_crown_distribution (initial_bread : BreadDistribution) (total_crowns : ℚ) : CrownDistribution :=
  let total_bread := initial_bread.person_a + initial_bread.person_b
  let person_a_contribution := initial_bread.person_a - total_bread / 3
  let person_b_contribution := initial_bread.person_b - total_bread / 3
  let total_contribution := person_a_contribution + person_b_contribution
  { person_a := (person_a_contribution / total_contribution) * total_crowns,
    person_b := (person_b_contribution / total_contribution) * total_crowns }

theorem fair_crown_distribution 
  (initial_bread : BreadDistribution) 
  (total_crowns : ℚ) :
  initial_bread.person_a = 5 ∧ 
  initial_bread.person_b = 3 ∧ 
  total_crowns = 2 →
  let result := calculate_crown_distribution initial_bread total_crowns
  result.person_a = 7/4 ∧ result.person_b = 1/4 :=
by sorry

#eval calculate_crown_distribution ⟨5, 3⟩ 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_crown_distribution_l98_9837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_exponential_inequality_l98_9816

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_exponential_inequality :
  (¬ ∃ x : ℝ, (2 : ℝ)^x ≥ 1) ↔ (∀ x : ℝ, (2 : ℝ)^x < 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_exponential_inequality_l98_9816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l98_9858

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / (2^x + a) - 1/2

theorem problem_solution (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 1 ∧
   (∀ x y, x < y → f a x > f a y) ∧
   (∀ y, y ∈ Set.Ioo (-1/2 : ℝ) (1/2) ↔ ∃ x, f a x = y) ∧
   (∀ k, (∀ x ∈ Set.Icc 1 4, f a (k - 2/x) + f a (2-x) > 0) → k < 2 * Real.sqrt 2 - 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l98_9858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_five_tosses_l98_9888

/-- The expected number of heads when tossing a fair coin n times -/
noncomputable def expected_heads (n : ℕ) : ℝ := n * (1 / 2 : ℝ)

/-- The number of tosses -/
def num_tosses : ℕ := 5

/-- Theorem: The expected number of heads when tossing a fair coin 5 times is 2.5 -/
theorem expected_heads_five_tosses :
  expected_heads num_tosses = (5 / 2 : ℝ) := by
  -- Unfold the definition of expected_heads
  unfold expected_heads
  -- Simplify the expression
  simp [num_tosses]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_five_tosses_l98_9888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_tangent_to_x_axis_l98_9829

/-- Given functions f and g -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3*x - (a+1)*Real.log x

def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 4

/-- Theorem for part 1 -/
theorem monotonically_increasing_condition (a : ℝ) :
  (∀ x > 0, HasDerivAt (λ x ↦ f a x + g a x) ((λ x ↦ 3 - (a+1)/x + 2*x - a) x) x) →
  (∀ x > 0, (λ x ↦ 3 - (a+1)/x + 2*x - a) x ≥ 0) →
  a ≤ -1 := by sorry

/-- Theorem for part 2 -/
theorem tangent_to_x_axis :
  ∃ a₁ a₂, a₁ = 2 ∧ 1 < a₂ ∧ a₂ < 2*(Real.exp 2) - 1 ∧
  (∃ x₁ > 0, f a₁ x₁ - g a₁ x₁ = 0 ∧ HasDerivAt (λ x ↦ f a₁ x - g a₁ x) 0 x₁) ∧
  (∃ x₂ > 0, f a₂ x₂ - g a₂ x₂ = 0 ∧ HasDerivAt (λ x ↦ f a₂ x - g a₂ x) 0 x₂) ∧
  (∀ a, a ≠ a₁ ∧ a ≠ a₂ → ¬∃ x > 0, f a x - g a x = 0 ∧ HasDerivAt (λ x ↦ f a x - g a x) 0 x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_tangent_to_x_axis_l98_9829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersections_circle_l98_9850

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 - 2*x - 3

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 5

/-- The intersection points of the parabola with x-axis -/
def x_intersections : Set ℝ := {x : ℝ | parabola x 0}

/-- The intersection point of the parabola with y-axis -/
def y_intersection : ℝ := -3

theorem parabola_intersections_circle :
  ∀ (x y : ℝ), (x ∈ x_intersections ∧ y = 0) ∨ (x = 0 ∧ y = y_intersection) → circle_eq x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersections_circle_l98_9850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l98_9876

-- Define p₁
def p₁ : Prop := ∀ x : ℝ, x > 0 → (3 : ℝ)^x > (2 : ℝ)^x

-- Define p₂
def p₂ : Prop := ∃ θ : ℝ, Real.sin θ + Real.cos θ = 3/2

-- Define q₁, q₂, q₃, q₄
def q₁ : Prop := p₁ ∨ p₂
def q₂ : Prop := p₁ ∧ p₂
def q₃ : Prop := (¬p₁) ∨ p₂
def q₄ : Prop := p₁ ∧ (¬p₂)

-- Theorem statement
theorem problem_solution : q₁ ∧ q₄ ∧ ¬q₂ ∧ ¬q₃ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l98_9876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_l98_9802

/-- Represents a triathlon race with given distances and speeds for each segment. -/
structure Triathlon where
  swim_distance : ℝ
  bike_distance : ℝ
  run_distance : ℝ
  swim_speed : ℝ
  bike_speed : ℝ
  run_speed : ℝ

/-- Calculates the average speed for a triathlon race. -/
noncomputable def average_speed (t : Triathlon) : ℝ :=
  let total_distance := t.swim_distance + t.bike_distance + t.run_distance
  let total_time := t.swim_distance / t.swim_speed + t.bike_distance / t.bike_speed + t.run_distance / t.run_speed
  total_distance / total_time

/-- The triathlon race details. -/
def race : Triathlon := {
  swim_distance := 5,
  bike_distance := 30,
  run_distance := 15,
  swim_speed := 3,
  bike_speed := 25,
  run_speed := 8
}

/-- Theorem stating that the average speed of the given triathlon race is approximately 10.5 km/h. -/
theorem triathlon_average_speed :
  ∃ ε > 0, |average_speed race - 10.5| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_l98_9802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_domain_of_g_l98_9808

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- Define the domain of g
def DomainG : Set ℝ := {x : ℝ | ∃ y, g x = y}

-- State the properties of g
axiom g_property (x : ℝ) : x ∈ DomainG → (1 / x^2) ∈ DomainG ∧ g x + g (1 / x^2) = x^2

-- Theorem to prove
theorem largest_domain_of_g : 
  DomainG = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_domain_of_g_l98_9808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minimized_at_40_l98_9817

/-- Definition of the sequence a_n -/
def a : ℕ → ℝ → ℝ
  | 0, p => p  -- Add case for n = 0
  | 1, p => p
  | 2, p => p + 1
  | (n + 3), p => 2 * a (n + 2) p - a (n + 1) p + (n + 3) - 20

/-- The value of n that minimizes a_n -/
def minimizing_n : ℕ := 40

/-- Theorem stating that minimizing_n minimizes a_n -/
theorem a_minimized_at_40 (p : ℝ) :
  ∀ n : ℕ, n ≥ 1 → a minimizing_n p ≤ a n p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minimized_at_40_l98_9817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implication_l98_9856

theorem inequality_implication (x y : ℝ) : (3 : ℝ)^x - (5 : ℝ)^(-x) ≥ (3 : ℝ)^(-y) - (5 : ℝ)^y → x + y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implication_l98_9856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_running_time_l98_9801

/-- Represents the race between Nicky and Cristina --/
structure Race where
  total_distance : ℝ
  cristina_speed : ℝ
  nicky_speed : ℝ
  head_start : ℝ

/-- Calculates the time when Cristina catches up to Nicky --/
noncomputable def catch_up_time (race : Race) : ℝ :=
  (race.nicky_speed * race.head_start) / (race.cristina_speed - race.nicky_speed)

/-- Theorem stating that Nicky runs for 30 seconds before Cristina catches up --/
theorem nicky_running_time (race : Race) 
  (h1 : race.total_distance = 200)
  (h2 : race.cristina_speed = 5)
  (h3 : race.nicky_speed = 3)
  (h4 : race.head_start = 12) :
  catch_up_time race + race.head_start = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_running_time_l98_9801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_interval_l98_9836

open Real

/-- The function for which we want to find the monotonically decreasing interval -/
noncomputable def f (x : ℝ) : ℝ := sqrt (2 * sin (3 * x + π / 4) - 1)

/-- Predicate to check if a function is monotonically decreasing on an interval -/
def IsMonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

/-- The theorem stating the monotonically decreasing interval of the function -/
theorem f_monotonically_decreasing_interval (k : ℤ) :
  IsMonotonicallyDecreasing f (π / 12 + 2 * ↑k * π / 3) (7 * π / 36 + 2 * ↑k * π / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_interval_l98_9836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_70_to_79_approx_l98_9831

/-- Represents the grade ranges in the frequency distribution --/
inductive GradeRange
  | Above90
  | From80To89
  | From70To79
  | From60To69
  | Below60

/-- Represents the frequency distribution of grades --/
def frequency_distribution : GradeRange → ℕ
  | GradeRange.Above90 => 5
  | GradeRange.From80To89 => 8
  | GradeRange.From70To79 => 10
  | GradeRange.From60To69 => 4
  | GradeRange.Below60 => 6

/-- Calculates the total number of students --/
def total_students : ℕ :=
  (frequency_distribution GradeRange.Above90) +
  (frequency_distribution GradeRange.From80To89) +
  (frequency_distribution GradeRange.From70To79) +
  (frequency_distribution GradeRange.From60To69) +
  (frequency_distribution GradeRange.Below60)

/-- Calculates the percentage of students in the 70% - 79% range --/
noncomputable def percentage_70_to_79 : ℝ :=
  (frequency_distribution GradeRange.From70To79 : ℝ) / (total_students : ℝ) * 100

/-- Theorem stating that the percentage of students in the 70% - 79% range is approximately 30.3% --/
theorem percentage_70_to_79_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |percentage_70_to_79 - 30.3| < ε := by
  sorry

#eval total_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_70_to_79_approx_l98_9831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_traffic_flow_l98_9891

/-- The traffic speed function -/
noncomputable def v (x : ℝ) : ℝ :=
  if x < 20 then 60
  else if x ≤ 200 then (200 - x) / 3
  else 0

/-- The traffic flow function -/
noncomputable def f (x : ℝ) : ℝ := x * v x

/-- Theorem: Maximum traffic flow occurs at x = 100 with a value of 3333 vehicles/hour -/
theorem max_traffic_flow :
  ∃ (x : ℝ), x ≥ 0 ∧ x ≤ 200 ∧
  (∀ y, y ≥ 0 → y ≤ 200 → f y ≤ f x) ∧
  x = 100 ∧ 
  (Int.floor (f x) : ℝ) = 3333 := by
  sorry

#check max_traffic_flow

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_traffic_flow_l98_9891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l98_9882

/-- The number of days Matt and Peter worked together -/
def days_worked_together : ℝ := 12

/-- The total amount of work to be done -/
noncomputable def W : ℝ := 1  -- We set this to 1 as a placeholder

/-- Matt and Peter's combined work rate per day -/
noncomputable def combined_rate : ℝ := W / 20

/-- Peter's individual work rate per day -/
noncomputable def peter_rate : ℝ := W / 35

/-- The number of days Peter worked alone after Matt stopped -/
def peter_solo_days : ℝ := 14

theorem work_completion :
  days_worked_together * combined_rate + peter_solo_days * peter_rate = W := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l98_9882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_equality_l98_9838

/-- α(n) is the number of ways n can be expressed as a sum of 1 and 2, considering different orders as distinct ways -/
def α : ℕ+ → ℕ :=
  sorry

/-- β(n) is the number of ways n can be expressed as a sum of integers greater than 1, considering different orders as distinct ways -/
def β : ℕ+ → ℕ :=
  sorry

/-- For every positive integer n, α(n) equals β(n+2) -/
theorem alpha_beta_equality (n : ℕ+) : α n = β (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_equality_l98_9838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l98_9884

/-- Circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 1

/-- Circle D -/
def circle_D (x y : ℝ) : Prop := (x + 4)^2 + (y - 1)^2 = 4

/-- Point P is outside circle C -/
def P_outside_C (x₀ y₀ : ℝ) : Prop := x₀^2 + (y₀ - 4)^2 > 1

/-- Distance from P to the center of circle C -/
noncomputable def distance_PC (x₀ y₀ : ℝ) : ℝ := Real.sqrt (x₀^2 + (y₀ - 4)^2)

/-- Area of quadrilateral PACB -/
noncomputable def f (x₀ y₀ : ℝ) : ℝ := Real.sqrt ((distance_PC x₀ y₀)^2 - 1)

theorem range_of_f :
  ∀ x₀ y₀ : ℝ, circle_D x₀ y₀ → P_outside_C x₀ y₀ →
  ∃ (a b : ℝ), a = 2 * Real.sqrt 2 ∧ b = 4 * Real.sqrt 3 ∧ a ≤ f x₀ y₀ ∧ f x₀ y₀ ≤ b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l98_9884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_window_pane_width_l98_9815

/-- Represents a rectangular glass pane -/
structure GlassPane where
  length : ℝ
  width : ℝ

/-- Represents a window composed of multiple glass panes -/
structure Window where
  panes : List GlassPane
  total_area : ℝ

/-- Calculates the area of a single glass pane -/
def pane_area (pane : GlassPane) : ℝ :=
  pane.length * pane.width

/-- Theorem: Given a window with 8 identical panes, each 12 inches long,
    and a total area of 768 square inches, the width of each pane is 8 inches -/
theorem window_pane_width (w : Window) 
    (h1 : w.panes.length = 8)
    (h2 : ∀ p, p ∈ w.panes → p.length = 12)
    (h3 : w.total_area = 768)
    (h4 : ∀ p q, p ∈ w.panes → q ∈ w.panes → p = q) :
  ∀ p, p ∈ w.panes → p.width = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_window_pane_width_l98_9815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_product_zero_l98_9857

/-- The numerator of the rational function -/
noncomputable def numerator (x : ℝ) : ℝ := x^2 + 5*x - 14

/-- The denominator of the rational function -/
noncomputable def denominator (x : ℝ) : ℝ := x^3 + x^2 - 14*x + 24

/-- The partial fraction decomposition form -/
noncomputable def partialFraction (A B C x : ℝ) : ℝ := A / (x - 2) + B / (x + 3) + C / (x - 4)

/-- The theorem stating that the product ABC in the partial fraction decomposition equals 0 -/
theorem partial_fraction_product_zero :
  ∃ A B C : ℝ, ∀ x : ℝ, x ≠ 2 ∧ x ≠ -3 ∧ x ≠ 4 →
    numerator x / denominator x = partialFraction A B C x ∧ A * B * C = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_fraction_product_zero_l98_9857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_z_l98_9885

-- Define the variables
variable (w z : ℂ)

-- State the theorem
theorem magnitude_z (h1 : w * z = 12 - 8 * Complex.I) (h2 : Complex.abs w = Real.sqrt 13) :
  Complex.abs z = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_z_l98_9885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_parallel_length_is_100_l98_9807

/-- Represents a rectangular garden with a hedge on three sides and a wall on the fourth side. -/
structure Garden where
  wall_length : ℝ
  hedge_cost_per_foot : ℝ
  total_hedge_cost : ℝ

/-- Calculates the area of the garden given the length of the side perpendicular to the wall. -/
noncomputable def garden_area (g : Garden) (x : ℝ) : ℝ :=
  x * (g.total_hedge_cost / g.hedge_cost_per_foot - 2 * x)

/-- Finds the length of the side perpendicular to the wall that maximizes the garden area. -/
noncomputable def optimal_perpendicular_length (g : Garden) : ℝ :=
  g.total_hedge_cost / (2 * g.hedge_cost_per_foot)

/-- Theorem stating that the optimal length of the side parallel to the wall is 100 feet. -/
theorem optimal_parallel_length_is_100 (g : Garden) 
    (h1 : g.wall_length = 500)
    (h2 : g.hedge_cost_per_foot = 10)
    (h3 : g.total_hedge_cost = 2000) : 
    g.total_hedge_cost / g.hedge_cost_per_foot - 2 * optimal_perpendicular_length g = 100 := by
  sorry

#check optimal_parallel_length_is_100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_parallel_length_is_100_l98_9807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_value_l98_9810

/-- The sequence a_n defined by the given recurrence relation -/
def a : ℕ → ℕ
  | 0 => 0  -- Add this case to handle n = 0
  | 1 => 0
  | n + 2 => a (n + 1) + 2 * (n + 1)

/-- Theorem stating that the 2016th term of the sequence equals 2015 * 2016 -/
theorem a_2016_value : a 2016 = 2015 * 2016 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_value_l98_9810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_determines_a_l98_9849

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = (2^x + a) / (2^x - a) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x + a) / (2^x - a)

theorem odd_function_determines_a :
  ∀ a : ℝ, a > 0 → IsOdd (f a) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_determines_a_l98_9849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salmon_oxygen_consumption_ratio_l98_9834

-- Define the relationship between swimming speed and oxygen consumption
noncomputable def swimming_speed (O : ℝ) : ℝ := (1/2) * (Real.log O / Real.log 3 - Real.log 100 / Real.log 3)

-- State the theorem
theorem salmon_oxygen_consumption_ratio :
  ∀ (O₁ O₂ : ℝ), O₁ > 0 → O₂ > 0 →
  swimming_speed O₂ = swimming_speed O₁ + 2 →
  O₂ / O₁ = 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_salmon_oxygen_consumption_ratio_l98_9834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l98_9832

open Set Real

theorem solution_set_inequality (f : ℝ → ℝ) (h1 : f 1 = 1) 
  (h2 : ∀ x, Differentiable ℝ f ∧ Differentiable ℝ (deriv f) ∧ deriv (deriv f) x > 1/3) :
  {x : ℝ | f (log x) < (2 + log x) / 3} = Ioo 0 (exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l98_9832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_slope_l98_9859

/-- Proves that for a parabola y^2 = 2px (p > 0) with focus F and directrix intersecting x-axis at M,
    if a line passing through M intersects the parabola at point A such that |AM| = 5/4 * |AF|,
    then the slope of this line is ± 3/4. -/
theorem parabola_intersection_slope (p : ℝ) (x₀ y₀ : ℝ) (h_p : p > 0) :
  let C := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let M := (-p/2, 0)
  let A := (x₀, y₀)
  let F := (p/2, 0)
  A ∈ C →
  (x₀ + p/2)^2 + y₀^2 = (5/4 * (x₀ + p/2))^2 →
  (abs (y₀ / (x₀ + p/2)) = 3/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_slope_l98_9859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_3_eq_zero_l98_9812

-- Define the function f
def f (x : ℝ) : ℝ := 1 - 2 * x

-- Define the function g
noncomputable def g (y : ℝ) : ℝ :=
  let x := (1 - y) / 2
  if x ≠ 0 then (x^2 - 1) / x^2 else 0

-- Theorem to prove
theorem g_of_3_eq_zero : g 3 = 0 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the let expression
  simp
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_3_eq_zero_l98_9812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equals_f_l98_9828

/-- The function defining the circle -/
def f (a b c : ℝ) (x y : ℝ) : ℝ := x^2 + y^2 + a*x + b*y + c

/-- The power of a point with respect to a circle -/
noncomputable def power_of_point (a b c : ℝ) (x₀ y₀ : ℝ) : ℝ :=
  let α := -a/2
  let β := -b/2
  let R := Real.sqrt (α^2 + β^2 - c)
  (x₀ - α)^2 + (y₀ - β)^2 - R^2

/-- Theorem: The power of a point with respect to a circle is equal to f(x₀, y₀) -/
theorem power_equals_f (a b c : ℝ) (x₀ y₀ : ℝ) :
  power_of_point a b c x₀ y₀ = f a b c x₀ y₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equals_f_l98_9828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sqrt_one_minus_x_squared_l98_9852

open MeasureTheory Interval Real

theorem integral_x_squared_plus_sqrt_one_minus_x_squared :
  ∫ x in (-1)..1, (x^2 + sqrt (1 - x^2)) = 2/3 + π/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sqrt_one_minus_x_squared_l98_9852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_triangle_side_length_l98_9827

/-- Given a sequence of equilateral triangles where each subsequent triangle
    is formed by joining the midpoints of the previous triangle's sides,
    this function returns the side length of the nth triangle in the sequence. -/
noncomputable def triangleSideLength (n : ℕ) (a : ℝ) : ℝ :=
  a / (2 ^ (n - 1))

/-- The sum of the perimeters of all triangles in the infinite sequence. -/
noncomputable def sumOfPerimeters (a : ℝ) : ℝ :=
  3 * a / (1 - 1/2)

theorem third_triangle_side_length :
  ∃ (a : ℝ),
    a > 0 ∧
    triangleSideLength 1 a = 60 ∧
    sumOfPerimeters a = 360 ∧
    triangleSideLength 3 a = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_triangle_side_length_l98_9827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_vector_sum_l98_9811

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 4

-- Define point M
def M : ℝ × ℝ := (3, 2)

-- Define the set of points P on the y-axis
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

-- Define the set of points Q on the circle C
def Q : Set (ℝ × ℝ) := {q : ℝ × ℝ | circleC q.1 q.2}

-- Define the vector MP + MQ
def vectorSum (p q : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - M.1 + q.1 - M.1, p.2 - M.2 + q.2 - M.2)

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- Theorem statement
theorem min_vector_sum :
  ∃ (min : ℝ), min = 3 ∧
  ∀ (p q : ℝ × ℝ), p ∈ P → q ∈ Q → magnitude (vectorSum p q) ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_vector_sum_l98_9811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_trig_ratios_l98_9848

theorem right_triangle_trig_ratios (AC BC : ℝ) (h1 : AC = 5) (h2 : BC = 13) :
  let AB : ℝ := Real.sqrt (BC^2 - AC^2)
  Real.cos (Real.arccos (AC / BC)) = 5/13 ∧ Real.sin (Real.arcsin (AC / BC)) = 5/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_trig_ratios_l98_9848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_value_triangle_area_l98_9809

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
noncomputable def given_triangle : Triangle where
  a := 5  -- Given for part II
  b := Real.sqrt ((25 * 14) / 11)  -- Derived value, not explicitly given
  c := 8  -- Derived value, not explicitly given
  A := Real.arccos (11/14)  -- Derived value, not explicitly given
  B := Real.pi / 3
  C := Real.arccos (1/7)  -- Derived value, not explicitly given

-- Theorem for part I
theorem cos_C_value (t : Triangle) (h : (t.a - t.b + t.c) * (t.a + t.b - t.c) = 3/7 * t.b * t.c) :
  Real.cos t.C = 1/7 := by sorry

-- Theorem for part II
theorem triangle_area (t : Triangle) (h1 : t.a = 5) (h2 : (t.a - t.b + t.c) * (t.a + t.b - t.c) = 3/7 * t.b * t.c) :
  (1/2) * t.a * t.c * Real.sin t.B = 10 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_value_triangle_area_l98_9809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l98_9861

def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 7*x^2 + 6*x - 2

theorem roots_of_equation :
  (∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ - x₂ = 1) →
  ∀ x : ℝ, f x = 0 ↔ x ∈ ({1 + Real.sqrt 3, 2 + Real.sqrt 3, 1 - Real.sqrt 3, 2 - Real.sqrt 3} : Set ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l98_9861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_path_1_to_9_l98_9818

/-- Represents a city by its digit -/
def City := Fin 10

/-- Two cities are connected if their two-digit number is divisible by 3 -/
def connected (a b : City) : Prop :=
  (10 * a.val + b.val) % 3 = 0

/-- A path between cities is a list of connected cities -/
inductive CityPath : City → City → Type where
  | single : (a : City) → CityPath a a
  | cons : (a b c : City) → connected a b → CityPath b c → CityPath a c

/-- The main theorem: there is no path from city 1 to city 9 -/
theorem no_path_1_to_9 : ¬ ∃ (p : CityPath ⟨1, by norm_num⟩ ⟨9, by norm_num⟩), True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_path_1_to_9_l98_9818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_l98_9819

/-- The area of the shaded region in a regular hexagon with side length 2 and six inscribed semicircles --/
noncomputable def shaded_area (side_length : ℝ) : ℝ :=
  let hexagon_area := 3 * Real.sqrt 3 * side_length^2 / 2
  let semicircle_area := Real.pi * (side_length / 2)^2 / 2
  hexagon_area - 6 * semicircle_area

/-- Theorem stating that the shaded area for a hexagon with side length 2 is 6√3 - 3π --/
theorem shaded_area_hexagon : shaded_area 2 = 6 * Real.sqrt 3 - 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_l98_9819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l98_9842

theorem inequality_proof (a b c : ℝ) (l : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hl : l > -2 ∧ l < 2) :
  (Real.sqrt (a^2 + l*a*b + b^2) + Real.sqrt (b^2 + l*b*c + c^2) + Real.sqrt (c^2 + l*c*a + a^2))^2 
  ≥ (2+l)*(a+b+c)^2 + (2-l)*(a-b)^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l98_9842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_raisins_eaten_l98_9873

/-- Represents the number of raisins eaten by each guest -/
def RaisinSequence := List Nat

/-- Checks if a sequence of raisins eaten satisfies the given conditions -/
def IsValidSequence (seq : RaisinSequence) : Prop :=
  seq.length > 0 ∧
  ∀ i, i < seq.length →
    (seq.get! ((i + 1) % seq.length) = 2 * seq.get! i ∨
     seq.get! i = seq.get! ((i + 1) % seq.length) + 6)

theorem not_all_raisins_eaten (total_raisins : Nat) (seq : RaisinSequence)
    (h_total : total_raisins = 2011)
    (h_valid : IsValidSequence seq)
    (h_sum : seq.sum ≤ total_raisins) :
    seq.sum < total_raisins := by
  sorry

#check not_all_raisins_eaten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_raisins_eaten_l98_9873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_example_l98_9820

/-- Represents a number in base 8 -/
structure OctalNumber where
  value : ℕ
  isOctal : value < 8^64 := by sorry

/-- Converts an octal number to its decimal representation -/
def octal_to_decimal (n : OctalNumber) : ℕ := sorry

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : OctalNumber := ⟨n, sorry⟩

/-- Subtracts two octal numbers -/
def octal_subtract (a b : OctalNumber) : OctalNumber :=
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

/-- Helper function to create OctalNumber from literal -/
def mkOctal (n : ℕ) : OctalNumber := ⟨n, sorry⟩

theorem octal_subtraction_example :
  octal_subtract (mkOctal 453) (mkOctal 267) = mkOctal 164 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_example_l98_9820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_20_l98_9841

noncomputable def selling_price : ℝ := 100
noncomputable def cost_price : ℝ := 83.33

noncomputable def profit : ℝ := selling_price - cost_price

noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

theorem profit_percentage_is_20 : 
  ∃ ε > 0, |profit_percentage - 20| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_20_l98_9841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_tents_after_week_l98_9880

def campsite_growth (initial : ℚ) (rate : ℚ) (days : ℕ) : ℚ :=
  initial * (1 + rate) ^ days

theorem total_tents_after_week (
  north_initial east_initial center_initial south_initial : ℚ)
  (north_rate east_rate center_rate south_rate : ℚ)
  (days : ℕ)
  (h1 : north_initial = 100)
  (h2 : east_initial = 2 * north_initial)
  (h3 : center_initial = 4 * north_initial)
  (h4 : south_initial = 200)
  (h5 : north_rate = 1/10)
  (h6 : east_rate = 1/20)
  (h7 : center_rate = 3/20)
  (h8 : south_rate = 7/100)
  (h9 : days = 7) :
  ⌊campsite_growth north_initial north_rate days +
   campsite_growth east_initial east_rate days +
   campsite_growth center_initial center_rate days +
   campsite_growth south_initial south_rate days⌋ = 1901 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_tents_after_week_l98_9880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l98_9893

theorem max_product_sum (a b c d : ℕ) : 
  a ∈ ({6, 7, 8, 9} : Set ℕ) ∧ b ∈ ({6, 7, 8, 9} : Set ℕ) ∧ 
  c ∈ ({6, 7, 8, 9} : Set ℕ) ∧ d ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ p q r s : ℕ, 
    p ∈ ({6, 7, 8, 9} : Set ℕ) ∧ q ∈ ({6, 7, 8, 9} : Set ℕ) ∧ 
    r ∈ ({6, 7, 8, 9} : Set ℕ) ∧ s ∈ ({6, 7, 8, 9} : Set ℕ) ∧
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    a * b + b * c + c * d + a * d ≥ p * q + q * r + r * s + p * s) ∧
  a * b + b * c + c * d + a * d = 225 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l98_9893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l98_9872

noncomputable section

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

-- Define the slope of the line passing through the right vertex and (0,-2)
noncomputable def slope_through_vertex (a : ℝ) : ℝ := 2 / a

-- Define the slope of the asymptote
noncomputable def slope_asymptote (a b : ℝ) : ℝ := b / a

-- The main theorem
theorem hyperbola_properties (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  eccentricity a b = Real.sqrt 3 →
  slope_through_vertex a = slope_asymptote a b →
  a = Real.sqrt 2 ∧ b = 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l98_9872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_book_count_l98_9840

def initial_books : ℕ := 34
def sell_percentage : ℚ := 45 / 100

theorem final_book_count : 
  initial_books - (Int.toNat ⌊(initial_books : ℚ) * sell_percentage⌋) + 
  (Int.toNat ⌊((Int.toNat ⌊(initial_books : ℚ) * sell_percentage⌋) : ℚ) / 3⌋) = 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_book_count_l98_9840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_special_subset_l98_9823

/-- A subset of real numbers satisfying the given condition -/
def SpecialSubset (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≠ y → (x + y - 1)^2 = x * y + 1

/-- The maximum number of elements in the special subset is 3 -/
theorem max_elements_special_subset :
  ∃ (S : Set ℝ), SpecialSubset S ∧ S.Finite ∧ S.ncard = 3 ∧
  ∀ (T : Set ℝ), SpecialSubset T → T.Finite → T.ncard ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_special_subset_l98_9823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_problem_l98_9805

theorem olympiad_problem (b g : ℕ) : 
  b > 0 → g > 0 →
  (b * 4 + g * (13/4 : ℚ)) / (b + g : ℚ) = 18/5 →
  31 ≤ b + g → b + g ≤ 50 →
  b + g = 45 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_problem_l98_9805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_four_necessary_not_sufficient_l98_9800

/-- The hyperbola equation -/
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / 8 = 1

/-- Definition of the distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Theorem stating that "a = 4" is a necessary but not sufficient condition -/
theorem a_four_necessary_not_sufficient 
  (a : ℝ) 
  (h_a_pos : a > 0)
  (P : ℝ × ℝ)
  (F₁ F₂ : ℝ × ℝ)
  (h_P_on_hyperbola : hyperbola a P.1 P.2)
  (h_PF₁_dist : distance P.1 P.2 F₁.1 F₁.2 = 9)
  : (a = 4 → distance P.1 P.2 F₂.1 F₂.2 = 17) ∧ 
    ¬(a = 4 ↔ distance P.1 P.2 F₂.1 F₂.2 = 17) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_four_necessary_not_sufficient_l98_9800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_hygiene_relationship_l98_9854

/-- The probability of event A given not B -/
noncomputable def P_A_given_not_B : ℝ := 3/4

/-- The probability of event B given not A -/
noncomputable def P_B_given_not_A : ℝ := 12/13

/-- The probability of event B -/
noncomputable def P_B : ℝ := 4/5

/-- The probability of event A -/
noncomputable def P_A : ℝ := 7/20

/-- The probability of event A given B -/
noncomputable def P_A_given_B : ℝ := 1/4

theorem disease_hygiene_relationship :
  P_A_given_not_B = 3/4 ∧ P_B_given_not_A = 12/13 ∧ P_B = 4/5 →
  P_A = 7/20 ∧ P_A_given_B = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_hygiene_relationship_l98_9854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l98_9851

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2^(x^2 + x - 3)

-- State the theorem
theorem f_increasing_interval :
  ∀ x y : ℝ, -1/2 < x ∧ x < y → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l98_9851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sin_equality_l98_9864

theorem contrapositive_sin_equality (x y : ℝ) : 
  (¬(Real.sin x = Real.sin y) → ¬(x = y)) ↔ (x = y → Real.sin x = Real.sin y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sin_equality_l98_9864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l98_9878

noncomputable def bridge_length : ℝ := 120
noncomputable def crossing_time : ℝ := 21.998240140788738
noncomputable def train_speed_kmph : ℝ := 36

noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

noncomputable def total_distance : ℝ := train_speed_mps * crossing_time

noncomputable def train_length : ℝ := total_distance - bridge_length

theorem train_length_calculation :
  abs (train_length - 99.98240140788738) < 1e-10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l98_9878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_pi_4_l98_9813

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^3
  else if 0 ≤ x ∧ x < Real.pi/2 then -Real.tan x
  else 0  -- undefined for x ≥ π/2

theorem f_composition_pi_4 :
  f (f (Real.pi/4)) = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_pi_4_l98_9813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l98_9897

/-- Checks if a triangle is a right triangle -/
def is_right_triangle (triangle : Set ℝ × Set ℝ) : Prop :=
  sorry

/-- Calculates the area of a triangle -/
def triangle_area (triangle : Set ℝ × Set ℝ) : ℝ :=
  sorry

/-- The area of a right triangle with hypotenuse 9 inches and smaller angle 30° -/
theorem right_triangle_area : 
  ∀ (triangle : Set ℝ × Set ℝ) (hypotenuse smaller_angle : ℝ),
    is_right_triangle triangle →
    hypotenuse = 9 →
    smaller_angle = 30 * π / 180 →
    triangle_area triangle = 10.125 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l98_9897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_at_specific_time_l98_9804

/-- The radius of a sphere with a time-varying curved surface area -/
theorem sphere_radius_at_specific_time (ω β c : ℝ) (r : ℝ → ℝ) : 
  (∀ (t : ℝ), 4 * Real.pi * (r t)^2 = ω * Real.sin (β * t) + c) →
  ω * Real.sin (β * (Real.pi / (2 * β))) + c = 64 * Real.pi →
  r (Real.pi / (2 * β)) = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_at_specific_time_l98_9804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_additional_time_for_normal_saturation_l98_9871

/-- Blood oxygen saturation model -/
noncomputable def S (S₀ K : ℝ) (t : ℝ) : ℝ := S₀ * Real.exp (K * t)

/-- Rate constant K calculated from 1-hour data point -/
noncomputable def K : ℝ := Real.log (70 / 60)

/-- Initial blood oxygen saturation -/
def S₀ : ℝ := 60

/-- Theorem stating the minimum additional time needed to reach 95% saturation -/
theorem min_additional_time_for_normal_saturation :
  ∃ t : ℝ, t > 0 ∧ (∀ t' : ℝ, t' ≥ 0 → S S₀ K (1 + t') ≥ 95 → t' ≥ t) ∧ t = 1.875 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_additional_time_for_normal_saturation_l98_9871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_second_and_eighth_term_l98_9889

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

noncomputable def geometric_mean (x y : ℝ) : ℝ :=
  Real.sqrt (x * y)

theorem geometric_mean_of_second_and_eighth_term (a : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  a 1 = 1 →
  r = 2 →
  geometric_mean (a 2) (a 8) = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_second_and_eighth_term_l98_9889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sec_sum_l98_9863

theorem min_value_sec_sum (m n s : ℝ) (α β : ℝ) 
  (hm : m > 0) (hn : n > 0) (hs : s > 0)
  (hα : α ∈ Set.Ioo 0 (Real.pi/2)) (hβ : β ∈ Set.Ioo 0 (Real.pi/2))
  (heq : m * Real.tan α + n * Real.tan β = s) :
  ∃ (min_value : ℝ), 
    (∀ α' β' : ℝ, α' ∈ Set.Ioo 0 (Real.pi/2) → β' ∈ Set.Ioo 0 (Real.pi/2) → 
      m * Real.tan α' + n * Real.tan β' = s → 
      m * (1 / Real.cos α') + n * (1 / Real.cos β') ≥ min_value) ∧
    min_value = Real.sqrt ((m + n)^2 + s^2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sec_sum_l98_9863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_nonnegative_abs_g_nonnegative_in_domain_l98_9806

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ -2 then x + 4
  else if x ≥ -4 then -(x+2)^2 + 2
  else if x ≥ -6 then -3*x + 3
  else 0  -- Assuming g(x) is not defined for x < -6

-- Theorem statement
theorem abs_g_nonnegative (x : ℝ) : |g x| ≥ 0 := by
  sorry

-- Theorem stating that |g(x)| is always non-negative in its domain
theorem abs_g_nonnegative_in_domain (x : ℝ) (h : x ≥ -6) : |g x| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_nonnegative_abs_g_nonnegative_in_domain_l98_9806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l98_9879

/-- The function f(x) as defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 1/m|

/-- The main theorem to be proved -/
theorem f_inequality (m : ℝ) (hm : m > 1) :
  ∀ x : ℝ, f m x + 1/(m*(m-1)) ≥ 3 := by
  intro x
  -- The proof steps would go here
  sorry

#check f_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l98_9879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l98_9870

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_c_pos : 0 < c
  h_a_gt_b : a > b
  h_c_sq : c^2 = a^2 - b^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

theorem ellipse_eccentricity_special_case (e : Ellipse) 
  (h : e.b = 2/3 * e.a) : eccentricity e = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l98_9870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_tbone_expected_filet_boxes_l98_9862

-- Define the total number of boxes and the number of each type of steak
def total_boxes : ℕ := 100
def filet_boxes : ℕ := 20
def sirloin_boxes : ℕ := 30
def ribeye_boxes : ℕ := 20
def tbone_boxes : ℕ := 30

-- Define the number of boxes selected using stratified sampling
def stratified_sample : ℕ := 10

-- Define the number of boxes randomly selected from the stratified sample
def random_sample : ℕ := 4

-- Define the probability of selecting a filet steak box
def filet_prob : ℚ := 1/5

-- Theorem 1: Probability of exactly 2 T-bone steak boxes
theorem prob_two_tbone (h : tbone_boxes = 30) :
  (Nat.choose 3 2 * Nat.choose 7 2 : ℚ) / Nat.choose stratified_sample random_sample = 3/10 := by
  sorry

-- Theorem 2: Expected number of filet steak boxes in 3 random selections
theorem expected_filet_boxes :
  3 * filet_prob = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_tbone_expected_filet_boxes_l98_9862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l98_9843

noncomputable def direction_vector : ℝ × ℝ := (-Real.sqrt 3, 3)

theorem angle_of_inclination (α : ℝ) : 
  α = 120 * π / 180 ↔ 
  Real.tan α = -(Real.sqrt 3) ∧ 
  0 ≤ α ∧ α < π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l98_9843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_of_g_l98_9895

noncomputable section

-- Define the function g
noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

-- State the theorem
theorem unique_number_not_in_range_of_g 
  (p q r s : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h17 : g p q r s 17 = 17)
  (h89 : g p q r s 89 = 89)
  (hg : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∃! y, (∀ x, g p q r s x ≠ y) ∧ y = 53 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_of_g_l98_9895
