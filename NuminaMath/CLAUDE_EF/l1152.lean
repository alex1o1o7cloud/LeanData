import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1152_115255

/-- The ellipse C -/
def C (x y : ℝ) : Prop := x^2/3 + y^2/2 = 1

/-- Point P on the ellipse C -/
def P (x y : ℝ) : Prop := C x y

/-- Point H is the intersection of the perpendicular line from P to the directrix and the ellipse -/
def H (x y : ℝ) : Prop := x = 3 ∧ C 3 y

/-- Point Q is on the extension of PH such that HQ = λPH, where λ ≥ 1 -/
def Q (lambda x y : ℝ) : Prop := lambda ≥ 1 ∧ ∃ x₁ y₁, P x₁ y₁ ∧ H 3 y₁ ∧ 
  x₁ = (3*(1 + lambda) - x)/lambda ∧ y = y₁

/-- The eccentricity of the locus of Q -/
noncomputable def eccentricity (lambda : ℝ) : ℝ := Real.sqrt (1 - 2/(3*lambda^2))

/-- Theorem: The eccentricity of the locus of Q is in the range [√3/3, 1) -/
theorem eccentricity_range : 
  ∀ lambda x y, Q lambda x y → Real.sqrt 3 / 3 ≤ eccentricity lambda ∧ eccentricity lambda < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1152_115255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_revenue_l1152_115204

theorem book_sale_revenue (total_books : ℕ) (price_per_book : ℚ) : 
  (2 : ℚ) / 3 * total_books = (total_books - 36 : ℕ) →
  price_per_book = 7/2 →
  ((2 : ℚ) / 3 * total_books * price_per_book : ℚ) = 252 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_revenue_l1152_115204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l1152_115248

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 70 →
  crossing_time = 45 →
  ∃ bridge_length : ℝ, abs (bridge_length - 624.8) < 0.1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l1152_115248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l1152_115262

-- Define the function h
noncomputable def h (t : ℝ) : ℝ := (t^2 + t + 1) / (t^2 + 2)

-- State the theorem
theorem h_range : 
  ∀ y : ℝ, (∃ t : ℝ, h t = y) → 1/2 ≤ y ∧ y ≤ 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l1152_115262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_correct_min_value_on_interval_max_value_on_interval_range_for_inequality_l1152_115217

noncomputable section

variable (a m : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x

def f_derivative (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - 1

theorem f_derivative_correct (a : ℝ) :
  ∀ x, deriv (f a) x = f_derivative a x := by
  sorry

theorem min_value_on_interval (h : m > 0) :
  IsMinOn (f 1) (Set.Icc (-m) m) 1 := by
  sorry

theorem max_value_on_interval (h : m > 0) :
  IsMaxOn (f 1) (Set.Icc (-m) m) (Real.exp m - m) := by
  sorry

theorem range_for_inequality (h : m > 0) :
  (∀ x ∈ Set.Icc (-m) m, f 1 x < Real.exp 2 - 2) ↔ m < 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_correct_min_value_on_interval_max_value_on_interval_range_for_inequality_l1152_115217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_in_folded_isosceles_right_triangle_l1152_115273

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is isosceles and right-angled -/
def IsIsoscelesRight (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 ∧
  (t.B.x - t.C.x) * (t.A.x - t.C.x) + (t.B.y - t.C.y) * (t.A.y - t.C.y) = 0

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Length of crease in folded isosceles right triangle -/
theorem crease_length_in_folded_isosceles_right_triangle
  (t : Triangle) (A' : Point) :
  IsIsoscelesRight t →
  A'.x = t.B.x ∨ A'.x = t.C.x →
  A'.y = t.B.y ∨ A'.y = t.C.y →
  distance t.B A' = 2 →
  distance A' t.C = 3 →
  ∃ (P Q : Point), distance P Q = (10 * Real.sqrt 2 - 6) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_in_folded_isosceles_right_triangle_l1152_115273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_43_l1152_115270

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -5; 3, 7]

theorem det_A_eq_43 : Matrix.det A = 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_43_l1152_115270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_underwear_pairs_is_four_l1152_115233

/-- Represents the washing machine problem with given weight limits and clothes. -/
structure WashingMachineProblem where
  limit : ℕ
  sock_weight : ℕ
  underwear_weight : ℕ
  shirt_weight : ℕ
  shorts_weight : ℕ
  pants_weight : ℕ
  pants_count : ℕ
  shirt_count : ℕ
  shorts_count : ℕ
  sock_pairs_count : ℕ

/-- Calculates the total weight of clothes already in the wash. -/
def total_weight (p : WashingMachineProblem) : ℕ :=
  p.pants_weight * p.pants_count +
  p.shirt_weight * p.shirt_count +
  p.shorts_weight * p.shorts_count +
  p.sock_weight * p.sock_pairs_count

/-- Calculates the remaining weight capacity. -/
def remaining_weight (p : WashingMachineProblem) : ℕ :=
  p.limit - total_weight p

/-- Calculates the maximum number of underwear pairs that can be added. -/
def max_underwear_pairs (p : WashingMachineProblem) : ℕ :=
  remaining_weight p / p.underwear_weight

/-- Theorem stating that given the problem constraints, the maximum number of
    additional underwear pairs is 4. -/
theorem max_underwear_pairs_is_four :
  ∀ (p : WashingMachineProblem),
    p.limit = 50 ∧
    p.sock_weight = 2 ∧
    p.underwear_weight = 4 ∧
    p.shirt_weight = 5 ∧
    p.shorts_weight = 8 ∧
    p.pants_weight = 10 ∧
    p.pants_count = 1 ∧
    p.shirt_count = 2 ∧
    p.shorts_count = 1 ∧
    p.sock_pairs_count = 3 →
    max_underwear_pairs p = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_underwear_pairs_is_four_l1152_115233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_hits_probability_l1152_115241

-- Define the probability of hitting the target in one shot
def hit_probability : ℝ := 0.6

-- Define the number of shots
def total_shots : ℕ := 3

-- Define the number of successful hits we're interested in
def successful_hits : ℕ := 2

-- Theorem statement
theorem exact_hits_probability :
  (Nat.choose total_shots successful_hits : ℝ) * 
  hit_probability ^ successful_hits * 
  (1 - hit_probability) ^ (total_shots - successful_hits) = 54 / 125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_hits_probability_l1152_115241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_return_speed_l1152_115297

/-- Calculates the return speed of a round trip given the total distance, 
    outbound time, outbound speed, and average speed of the entire trip. -/
noncomputable def return_speed (total_distance : ℝ) (outbound_time : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) : ℝ :=
  let total_time := total_distance / average_speed
  let return_time := total_time - outbound_time
  let return_distance := total_distance / 2
  return_distance / return_time

/-- Proves that given the specific conditions of Joey's round trip, 
    his return speed is 20 miles per hour. -/
theorem joey_return_speed : 
  let total_distance : ℝ := 10
  let outbound_time : ℝ := 1
  let outbound_speed : ℝ := 5
  let average_speed : ℝ := 8
  return_speed total_distance outbound_time outbound_speed average_speed = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_return_speed_l1152_115297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_bound_l1152_115299

noncomputable section

open Real

theorem triangle_side_ratio_bound (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- angles are positive
  A + B + C = π ∧          -- sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- sides are positive
  B = 3 * A ∧              -- given condition
  a / sin A = b / sin B    -- sine law
  →
  1 < b / a ∧ b / a < 3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_bound_l1152_115299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_properties_l1152_115245

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x - ω * Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

theorem cosine_function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x : ℝ, f ω (x + Real.pi) = f ω x) :
  ω = 2 ∧ 
  (∀ x : ℝ, f ω x = Real.cos (2 * x - Real.pi / 3)) ∧
  (∀ x : ℝ, f ω x = g (x - Real.pi / 6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_properties_l1152_115245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_terms_l1152_115269

theorem max_positive_terms (x : Fin 2022 → ℝ) 
  (h1 : ∀ k, x k ≠ 0)
  (h2 : ∀ k : Fin 2022, x k + (1 / x (Fin.succ k)) < 0)
  (h3 : x 0 = x (Fin.last 2022)) :
  (Finset.filter (fun i => 0 < x i) (Finset.univ : Finset (Fin 2022))).card ≤ 1011 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_terms_l1152_115269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_correct_l1152_115243

/-- A right circular cone with four congruent spheres inside -/
structure ConeSpheres where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  is_right_circular : Bool
  num_spheres : Nat
  spheres_tangent : Bool
  spheres_tangent_base : Bool
  spheres_tangent_side : Bool

/-- The specific cone with spheres from the problem -/
noncomputable def problem_cone : ConeSpheres :=
  { base_radius := 8
  , height := 15
  , sphere_radius := (4320 * Real.sqrt 6 - 9900) / 431
  , is_right_circular := true
  , num_spheres := 4
  , spheres_tangent := true
  , spheres_tangent_base := true
  , spheres_tangent_side := true
  }

/-- Theorem stating that the sphere radius in the problem_cone is correct -/
theorem sphere_radius_correct (c : ConeSpheres) :
  c = problem_cone →
  c.sphere_radius = (4320 * Real.sqrt 6 - 9900) / 431 :=
by
  sorry

#check sphere_radius_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_correct_l1152_115243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_angle_cosine_l1152_115206

theorem chord_angle_cosine (r : ℝ) (α β : ℝ) : 
  r > 0 ∧ 
  α > 0 ∧ 
  β > 0 ∧ 
  α + β < π ∧ 
  4^2 = 4 * r^2 * (1 - Real.cos α) ∧
  6^2 = 4 * r^2 * (1 - Real.cos β) ∧
  8^2 = 4 * r^2 * (1 - Real.cos (α + β)) ∧
  0 < Real.cos α ∧ 
  ∃ (q : ℚ), Real.cos α = ↑q
  →
  Real.cos α = (1 : ℚ) / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_angle_cosine_l1152_115206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_l1152_115279

def a (n : ℕ) : ℚ :=
  match n with
  | 0 => 1/2
  | n+1 => 2 * a n / (3 * a n + 2)

theorem a_2017 : a 2016 = 1/3026 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_l1152_115279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_inverse_squares_l1152_115278

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)
noncomputable def C₂ (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define points M₁ and M₂
def M₁ : ℝ × ℝ := (0, 1)
def M₂ : ℝ × ℝ := (2, 0)

-- Define the line M₁M₂
def M₁M₂_line (x y : ℝ) : Prop := x + 2 * y = 2

-- Define points P and Q as intersections of M₁M₂ with C₂
noncomputable def P : ℝ × ℝ := sorry
noncomputable def Q : ℝ × ℝ := sorry

-- Define points A and B as intersections of OP and OQ with C₁
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Statement to prove
theorem intersection_sum_inverse_squares :
  1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) = 5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_inverse_squares_l1152_115278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1152_115249

theorem division_problem (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : x % y = 9)
  (h2 : (x : ℝ) / (y : ℝ) = 86.12) :
  y = 75 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1152_115249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l1152_115287

noncomputable section

/-- The function f(x) = -x² + ax - a/4 + 1/2 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - a/4 + 1/2

/-- The maximum value of f(x) in [0, 1] is 2 -/
axiom max_value (a : ℝ) : ∃ x₀ ∈ Set.Icc 0 1, ∀ x ∈ Set.Icc 0 1, f a x ≤ f a x₀ ∧ f a x₀ = 2

theorem min_value_f (a : ℝ) :
  (a < 0 → ∃ x₀ ∈ Set.Icc 0 1, ∀ x ∈ Set.Icc 0 1, f a x ≥ f a x₀ ∧ f a x₀ = -5) ∧
  (a > 2 → ∃ x₀ ∈ Set.Icc 0 1, ∀ x ∈ Set.Icc 0 1, f a x ≥ f a x₀ ∧ f a x₀ = -1/3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l1152_115287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l1152_115240

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (4*a - 1)*x + 4*a else a^x

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/7 : ℝ) (1/4 : ℝ) ∧ a ≠ 1/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l1152_115240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_given_range_l1152_115251

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (3 - 2 * x)

theorem f_domain_given_range :
  (∀ y ∈ Set.Icc 1 2, ∃ x, f x = y) →
  {x : ℝ | f x ∈ Set.Icc 1 2} = Set.Icc (2/3) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_given_range_l1152_115251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_positive_integer_l1152_115274

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => ((sequence_a (n + 1))^2 + 2) / (sequence_a n)

theorem sequence_a_positive_integer (n : ℕ) : 
  sequence_a n > 0 ∧ ∃ (k : ℤ), sequence_a n = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_positive_integer_l1152_115274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kerosene_mixture_percentage_l1152_115259

noncomputable def liquid1_kerosene : ℝ := 0.25
noncomputable def liquid2_kerosene : ℝ := 0.30
noncomputable def liquid3_kerosene : ℝ := 0.45

noncomputable def parts_liquid1 : ℝ := 6
noncomputable def parts_liquid2 : ℝ := 4
noncomputable def parts_liquid3 : ℝ := 3

noncomputable def total_parts : ℝ := parts_liquid1 + parts_liquid2 + parts_liquid3

noncomputable def total_kerosene : ℝ := 
  parts_liquid1 * liquid1_kerosene + 
  parts_liquid2 * liquid2_kerosene + 
  parts_liquid3 * liquid3_kerosene

noncomputable def kerosene_percentage : ℝ := (total_kerosene / total_parts) * 100

theorem kerosene_mixture_percentage :
  |kerosene_percentage - 31.15| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kerosene_mixture_percentage_l1152_115259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_quadratic_analogy_l1152_115250

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Represents the number of intersection points between a line and a circle -/
inductive IntersectionCount
  | Zero
  | One
  | Two

/-- Function to determine the number of intersection points between a line and a circle -/
def intersectionCount (l : Line) (c : Circle) : IntersectionCount :=
  sorry

/-- Function to determine the number of roots of a quadratic equation -/
def quadraticRootCount (a b c : ℝ) : IntersectionCount :=
  sorry

/-- Theorem stating that the analogy between triangle construction and quadratic roots has deeper reasons -/
theorem triangle_quadratic_analogy :
  ∀ (l : Line) (c : Circle) (a b d : ℝ),
  ∃ (f : IntersectionCount → IntersectionCount),
  f (intersectionCount l c) = quadraticRootCount a b d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_quadratic_analogy_l1152_115250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birch_tree_arrangement_probability_l1152_115276

def maple_count : ℕ := 3
def oak_count : ℕ := 4
def birch_count : ℕ := 5

def total_trees : ℕ := maple_count + oak_count + birch_count

def non_birch_count : ℕ := maple_count + oak_count

def birch_spaces : ℕ := non_birch_count + 1

theorem birch_tree_arrangement_probability :
  (Nat.factorial non_birch_count * Nat.choose birch_spaces birch_count) / 
  Nat.factorial total_trees = 7 / 99 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birch_tree_arrangement_probability_l1152_115276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1152_115244

noncomputable def m : ℝ × ℝ := (Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

theorem vector_problem (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (((m.1 * (n x).1 + m.2 * (n x).2 = 0) → Real.tan x = 1) ∧
   ((m.1 * (n x).1 + m.2 * (n x).2 = Real.cos (Real.pi / 3)) → x = 5 * Real.pi / 12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1152_115244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_9_l1152_115257

/-- The area of a quadrilateral given its four vertices -/
noncomputable def quadrilateral_area (a b c d : ℝ × ℝ) : ℝ :=
  let (x1, y1) := a
  let (x2, y2) := b
  let (x3, y3) := c
  let (x4, y4) := d
  (1/2) * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

/-- The theorem stating that the area of the given quadrilateral is 9 -/
theorem quadrilateral_area_is_9 : 
  quadrilateral_area (0, 0) (2, 3) (5, 3) (3, 0) = 9 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval quadrilateral_area (0, 0) (2, 3) (5, 3) (3, 0)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_9_l1152_115257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_iff_a_in_range_l1152_115230

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (4 - a/2) * x + 2 else a^x

-- State the theorem
theorem monotonic_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (4 ≤ a ∧ a < 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_iff_a_in_range_l1152_115230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_cost_is_26_l1152_115266

/-- Represents the cost of a single chair -/
def chair_cost : ℕ := 26

/-- Represents the cost of the table -/
def table_cost : ℕ := 2 * chair_cost

/-- Represents the cost of a single lamp -/
def lamp_cost : ℕ := table_cost / 2

/-- Represents the cost of a set of plates -/
def plate_set_cost : ℕ := 20

/-- Represents the original cost of the painting -/
def painting_original_cost : ℕ := 100

/-- Represents the discount percentage on the painting -/
def painting_discount : ℚ := 1/4

/-- Represents the total amount Jude gave to the cashier -/
def total_given : ℕ := 350

/-- The main theorem stating that each chair costs $26 -/
theorem chair_cost_is_26 :
  chair_cost = 26 ∧
  ∃ (change : ℕ), 
    3 * chair_cost + table_cost + 4 * lamp_cost + 2 * plate_set_cost + 
    (painting_original_cost - (painting_original_cost * painting_discount).floor) + 
    change = total_given := by
  sorry

#eval chair_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_cost_is_26_l1152_115266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_sum_is_300_l1152_115211

/-- The side length of the initial equilateral triangle -/
def initial_side_length : ℝ := 50

/-- The sequence of side lengths for the triangles -/
noncomputable def triangle_side_sequence : ℕ → ℝ
  | 0 => initial_side_length
  | n + 1 => triangle_side_sequence n / 2

/-- The perimeter of the nth triangle -/
noncomputable def triangle_perimeter (n : ℕ) : ℝ := 3 * triangle_side_sequence n

/-- The sum of the perimeters of all triangles -/
noncomputable def perimeter_sum : ℝ := ∑' n, triangle_perimeter n

/-- Theorem: The sum of the perimeters of all triangles is 300 -/
theorem perimeter_sum_is_300 : perimeter_sum = 300 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_sum_is_300_l1152_115211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_point_inside_circle_l1152_115208

/-- A circle in the complex plane. -/
class IsCircle (S : Set ℂ) : Prop

/-- The interior of a set in the complex plane. -/
def interior' (S : Set ℂ) : Set ℂ := sorry

/-- Given complex numbers z₁ and z₂ where z₁z₂ = 1, and a circle Γ passing through -1 and 1
    but not through z₁ and z₂, exactly one of z₁ or z₂ is inside Γ. -/
theorem one_point_inside_circle (z₁ z₂ : ℂ) (Γ : Set ℂ) : 
  z₁ * z₂ = 1 →
  IsCircle Γ →
  (-1 : ℂ) ∈ Γ →
  (1 : ℂ) ∈ Γ →
  z₁ ∉ Γ →
  z₂ ∉ Γ →
  (z₁ ∈ interior' Γ ∧ z₂ ∉ interior' Γ) ∨ (z₁ ∉ interior' Γ ∧ z₂ ∈ interior' Γ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_point_inside_circle_l1152_115208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_chord_perimeter_l1152_115201

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a chord of a hyperbola -/
structure Chord where
  start : Point
  finish : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Check if a chord passes through a point -/
def chordPassesThrough (c : Chord) (p : Point) : Prop :=
  distance c.start p + distance p c.finish = distance c.start c.finish

theorem hyperbola_chord_perimeter 
  (h : Hyperbola) 
  (f1 f2 : Point) 
  (c : Chord) :
  h.a = 4 →
  h.b = 3 →
  isOnHyperbola h c.start →
  isOnHyperbola h c.finish →
  chordPassesThrough c f1 →
  distance c.start c.finish = 6 →
  distance c.start f2 + distance c.finish f2 + distance c.start c.finish = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_chord_perimeter_l1152_115201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_approximation_l1152_115295

/-- Given a square and a rectangle satisfying certain conditions, 
    prove that the length of the rectangle is approximately 7.1 cm. -/
theorem rectangle_length_approximation (s : ℝ) (l : ℝ) :
  π * s / 2 + s = 21.99 →  -- Semicircle circumference condition
  4 * s = 2 * (l + 10) →   -- Perimeter equality condition
  abs (l - 7.1) < 0.05 := by  -- Conclusion (approximate equality)
sorry

/-- Helper function to define approximate equality -/
def approx_equal (x y : ℝ) : Prop :=
  abs (x - y) < 0.05

notation:50 x " ≈ " y => approx_equal x y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_approximation_l1152_115295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_depends_on_a_two_zeros_range_l1152_115212

/-- Function f(x) defined as ln x - (a+2)x + ax^2 --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (a + 2) * x + a * x^2

/-- Derivative of f(x) --/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x - (a + 2) + 2 * a * x

/-- Tangent line equation when a = 0 --/
theorem tangent_line_at_one (y : ℝ) : 
  (f 0 1 = -2) ∧ (∀ x, x + y + 1 = 0 ↔ y = f 0 x + f_deriv 0 1 * (x - 1)) := by sorry

/-- Monotonicity of f(x) depends on a --/
theorem monotonicity_depends_on_a (a : ℝ) :
  ∃ (I₁ I₂ : Set ℝ), StrictMono (fun x => f a x) ∧ StrictAnti (fun x => f a x) := by sorry

/-- Range of a for which f(x) has exactly two zeros --/
theorem two_zeros_range (a : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ ∀ x, f a x = 0 → x = x₁ ∨ x = x₂) ↔ 
  a < -4 * Real.log 2 - 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_depends_on_a_two_zeros_range_l1152_115212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_bound_l1152_115216

/-- A polynomial of degree n with real coefficients -/
def MyPolynomial (n : ℕ) := ℝ → ℝ

/-- The property that |P(y)| ≤ 1 for all y in [0, 1] -/
def BoundedOnUnitInterval (P : MyPolynomial n) : Prop :=
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 1 → |P y| ≤ 1

/-- The main theorem -/
theorem polynomial_bound (n : ℕ) (P : MyPolynomial n) 
    (h : BoundedOnUnitInterval P) : 
    P (-1 / (n : ℝ)) ≤ 2^(n+1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_bound_l1152_115216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1152_115290

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 ∧ x ≤ 2 then 2^x - 1
  else if x < 0 ∧ x ≥ -2 then -(2^(-x) - 1)
  else 0

def g (x m : ℝ) : ℝ := x^2 - 2*x + m

-- State the theorem
theorem m_range (hf : ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ∈ Set.Icc (-3 : ℝ) 3)
                (hg : ∀ x₁ ∈ Set.Icc (-2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (-2 : ℝ) 2, g x₂ m = f x₁) :
  m ∈ Set.Icc (-5 : ℝ) (-2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1152_115290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_polar_curve_l1152_115237

/-- The length of the arc of the curve given by the polar equation ρ = 6 sin φ, 
    where 0 ≤ φ ≤ π/3, is equal to 2π. -/
theorem arc_length_polar_curve (ρ : ℝ → ℝ) :
  (∀ φ, 0 ≤ φ ∧ φ ≤ π/3 → ρ φ = 6 * Real.sin φ) →
  ∫ φ in Set.Icc 0 (π/3), Real.sqrt ((ρ φ)^2 + (deriv ρ φ)^2) = 2 * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_polar_curve_l1152_115237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1152_115296

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - x^2) / x

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -2 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ 2} := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1152_115296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_nonzero_l1152_115226

-- Define a cube vertex assignment as a function from vertex index to {-1, 1}
def CubeAssignment := Fin 8 → Int

-- Define a predicate for valid assignments (only -1 or 1)
def IsValidAssignment (a : CubeAssignment) : Prop :=
  ∀ i, a i = 1 ∨ a i = -1

-- Define the sum of vertex values
def VertexSum (a : CubeAssignment) : Int :=
  (Finset.univ : Finset (Fin 8)).sum a

-- Define the product of values on a face
def FaceProduct (a : CubeAssignment) (face : Fin 6) : Int :=
  match face with
  | ⟨0, _⟩ => a 0 * a 1 * a 2 * a 3
  | ⟨1, _⟩ => a 4 * a 5 * a 6 * a 7
  | ⟨2, _⟩ => a 0 * a 1 * a 4 * a 5
  | ⟨3, _⟩ => a 2 * a 3 * a 6 * a 7
  | ⟨4, _⟩ => a 0 * a 2 * a 4 * a 6
  | ⟨5, _⟩ => a 1 * a 3 * a 5 * a 7
  | _ => 1  -- This case should never occur due to Fin 6

-- Define the sum of face products
def FaceSum (a : CubeAssignment) : Int :=
  (Finset.univ : Finset (Fin 6)).sum (FaceProduct a)

-- Theorem: The sum of vertex values and face products cannot be zero
theorem cube_sum_nonzero (a : CubeAssignment) (h : IsValidAssignment a) :
  VertexSum a + FaceSum a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_nonzero_l1152_115226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_120_l1152_115229

/-- Represents a cone with given base radius and slant height -/
structure Cone where
  baseRadius : ℝ
  slantHeight : ℝ

/-- Calculates the degree of the central angle of a cone's lateral surface development diagram -/
noncomputable def centralAngleDegree (c : Cone) : ℝ :=
  (2 * c.baseRadius * 180) / c.slantHeight

/-- Theorem: For a cone with base radius 1 and slant height 3, 
    the central angle of its lateral surface development diagram is 120 degrees -/
theorem cone_central_angle_120 :
  let c : Cone := { baseRadius := 1, slantHeight := 3 }
  centralAngleDegree c = 120 := by
  -- Expand the definition of centralAngleDegree
  unfold centralAngleDegree
  -- Simplify the expression
  simp [Cone.baseRadius, Cone.slantHeight]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_120_l1152_115229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parabola_and_AB_l1152_115282

-- Define the equilateral triangle
noncomputable def triangle_ABC : Set (ℝ × ℝ) :=
  {(0, 0), (1, 0), (1/2, Real.sqrt 3 / 2)}

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := -Real.sqrt 3 * x^2 + Real.sqrt 3 * x

-- State the theorem
theorem area_between_parabola_and_AB :
  (∫ x in Set.Icc 0 1, parabola x) = Real.sqrt 3 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parabola_and_AB_l1152_115282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_increase_l1152_115256

/-- The ratio of a circle's circumference to its diameter -/
noncomputable def π : ℝ := Real.pi

/-- Theorem: If the circumference of a circle increases by 0.628 cm, 
    then its diameter increases by 0.2 cm -/
theorem circle_increase (r : ℝ) : 
  (2 * π * r + 0.628) - (2 * π * r) = π * (0.2) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_increase_l1152_115256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_base_length_l1152_115239

/-- Represents a parallelepiped with given properties -/
structure Parallelepiped where
  volume : ℝ
  side_length : ℝ
  base_length : ℝ
  height : ℝ
  other_side : ℝ
  volume_eq : volume = base_length * other_side * height
  height_eq : height = 3 * base_length
  other_side_eq : other_side = 2 * base_length

/-- The base length of a parallelepiped with given properties is approximately 4.02 meters -/
theorem parallelepiped_base_length (p : Parallelepiped)
  (h_volume : p.volume = 392)
  (h_side_length : p.side_length = 7) :
  ∃ (ε : ℝ), ε > 0 ∧ |p.base_length - 4.02| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_base_length_l1152_115239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_2_tangent_line_through_2_0_l1152_115221

-- Define the function f(x) = 4/x
noncomputable def f (x : ℝ) : ℝ := 4 / x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := -4 / (x^2)

-- Theorem for the first part
theorem tangent_line_at_2_2 :
  ∀ x y : ℝ, (x = 2 ∧ y = f 2) → (y - f 2 = f' 2 * (x - 2)) ↔ x + y - 4 = 0 :=
by sorry

-- Theorem for the second part
theorem tangent_line_through_2_0 :
  ∃ m : ℝ, m > 0 ∧
  (∀ x y : ℝ, y - f m = f' m * (x - m) → x = 2 ∧ y = 0) ↔
  (∀ x y : ℝ, 4*x + y - 8 = 0 → y - f m = f' m * (x - m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_2_tangent_line_through_2_0_l1152_115221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stand_distance_l1152_115264

/-- The distance to the bus stand in kilometers -/
noncomputable def D : ℝ := 2.2

/-- The time at which the bus arrives in hours -/
noncomputable def T : ℝ := 7/15

/-- Theorem stating that the given walking scenarios imply the distance to the bus stand is 2.2 km -/
theorem bus_stand_distance :
  (D / 3 = T + 12/60) ∧
  (D / 6 = T - 10/60) ∧
  (D / 4 = T + 5/60) →
  D = 2.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stand_distance_l1152_115264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_area_is_24_root_6_l1152_115294

/-- Triangle PQR with side lengths 10, 12, and 14 -/
structure Triangle :=
  (side_a : ℝ)
  (side_b : ℝ)
  (side_c : ℝ)

/-- The area of a triangle given its side lengths -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  let s := (t.side_a + t.side_b + t.side_c) / 2
  Real.sqrt (s * (s - t.side_a) * (s - t.side_b) * (s - t.side_c))

/-- The area of the union of a triangle and its image after 180° rotation about its centroid -/
noncomputable def union_area_after_rotation (t : Triangle) : ℝ :=
  triangle_area t

/-- Theorem: The area of the union of triangle PQR (10, 12, 14) and its image
    after 180° rotation about its centroid is 24√6 -/
theorem union_area_is_24_root_6 :
  let t : Triangle := { side_a := 10, side_b := 12, side_c := 14 }
  union_area_after_rotation t = 24 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_area_is_24_root_6_l1152_115294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_tim_speed_ratio_l1152_115247

/-- Tim's typing speed in pages per hour -/
def tim_speed : ℝ := sorry

/-- Tom's normal typing speed in pages per hour -/
def tom_speed : ℝ := sorry

/-- Tim and Tom can type 12 pages in one hour together -/
axiom combined_speed : tim_speed + tom_speed = 12

/-- If Tom increases his typing speed by 25%, they can type 14 pages in one hour -/
axiom increased_speed : tim_speed + 1.25 * tom_speed = 14

/-- The ratio of Tom's normal typing speed to Tim's typing speed is 2:1 -/
theorem tom_tim_speed_ratio : tom_speed / tim_speed = 2 := by 
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_tim_speed_ratio_l1152_115247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_equals_neg_cos_l1152_115292

open Real

noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => sin
  | n + 1 => deriv (f n)

theorem f_2015_equals_neg_cos : f 2015 = λ x => -cos x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_equals_neg_cos_l1152_115292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_tree_height_l1152_115235

-- Define the structure for a tree
structure TreeData where
  height : ℝ
  branches : ℕ

-- Define the given trees
def tree1 : TreeData := { height := 50, branches := 200 }
def tree2 : TreeData := { height := 0, branches := 180 } -- Height unknown
def tree3 : TreeData := { height := 60, branches := 180 }
def tree4 : TreeData := { height := 34, branches := 153 }

-- Define the average branches per foot
def averageBranchesPerFoot : ℝ := 4

-- Theorem to prove
theorem second_tree_height : 
  tree2.height = (tree2.branches : ℝ) / averageBranchesPerFoot := by
  sorry

#check second_tree_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_tree_height_l1152_115235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_prism_lateral_surface_area_l1152_115252

/-- A right prism with a rhombus base -/
structure RhombusPrism where
  /-- Area of one diagonal section -/
  P : ℝ
  /-- Area of the other diagonal section -/
  Q : ℝ
  /-- Assumption that P and Q are positive -/
  h_positive : P > 0 ∧ Q > 0

/-- The lateral surface area of a rhombus prism -/
noncomputable def lateralSurfaceArea (prism : RhombusPrism) : ℝ :=
  2 * Real.sqrt (prism.P^2 + prism.Q^2)

/-- Theorem stating that the lateral surface area of a rhombus prism
    is equal to 2√(P² + Q²) -/
theorem rhombus_prism_lateral_surface_area (prism : RhombusPrism) :
  lateralSurfaceArea prism = 2 * Real.sqrt (prism.P^2 + prism.Q^2) := by
  -- Unfold the definition of lateralSurfaceArea
  unfold lateralSurfaceArea
  -- The equation is now trivially true
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_prism_lateral_surface_area_l1152_115252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_evaluation_l1152_115222

-- Define the functions P and Q
noncomputable def P (x : ℝ) : ℝ := 3 * Real.sqrt x
def Q (x : ℝ) : ℝ := x^2

-- State the theorem
theorem nested_function_evaluation :
  P (Q (P (Q (P (Q 5))))) = 135 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_evaluation_l1152_115222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_count_l1152_115232

theorem sheep_count (sheep_to_horse_ratio : ℚ) (horse_food_per_day : ℕ) (total_food_per_day : ℕ) :
  sheep_to_horse_ratio = 3 / 7 →
  horse_food_per_day = 230 →
  total_food_per_day = 12880 →
  ∃ (sheep_count horse_count : ℕ),
    (sheep_count : ℚ) / (horse_count : ℚ) = sheep_to_horse_ratio ∧
    horse_count * horse_food_per_day = total_food_per_day ∧
    sheep_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_count_l1152_115232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1152_115205

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  (Real.sin t.A) / t.a = (Real.sqrt 3 * Real.cos t.C) / t.c ∧
  t.a + t.b = 6 ∧
  t.a * t.b * Real.cos t.C = 4

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1152_115205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1152_115219

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The equation of a line with a given point and angle -/
def line_equation (x₀ y₀ θ : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = Real.tan θ * (x - x₀)

theorem intersection_distance :
  let x₀ : ℝ := 1
  let y₀ : ℝ := 5
  let θ : ℝ := π / 3
  ∃ (x y : ℝ),
    line_equation x₀ y₀ θ x y ∧
    x - y - 2 * Real.sqrt 3 = 0 ∧
    distance x₀ y₀ x y = 5 - 3 * Real.sqrt 3 :=
by sorry

#check intersection_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1152_115219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_at_distance_9_from_focus_l1152_115286

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the parabola y^2 = 8x -/
def IsOnParabola (p : Point) : Prop :=
  p.y^2 = 8 * p.x

/-- The focus of the parabola y^2 = 8x is at (2, 0) -/
def FocusOfParabola : Point :=
  ⟨2, 0⟩

/-- Distance between two points -/
noncomputable def Distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating the coordinates of point P -/
theorem point_on_parabola_at_distance_9_from_focus 
  (p : Point) 
  (h1 : IsOnParabola p) 
  (h2 : Distance p FocusOfParabola = 9) : 
  p.x = 7 ∧ (p.y = 2 * Real.sqrt 14 ∨ p.y = -2 * Real.sqrt 14) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_at_distance_9_from_focus_l1152_115286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_existence_l1152_115225

theorem polynomial_root_existence (a b : ℤ) :
  (∀ p : ℕ, Nat.Prime p → ∃ k : ℤ, (p : ℤ) ∣ (k^2 + a*k + b) ∧ (p : ℤ) ∣ ((k+1)^2 + a*(k+1) + b)) →
  ∃ m : ℤ, m^2 + a*m + b = 0 ∧ (m+1)^2 + a*(m+1) + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_existence_l1152_115225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trips_l1152_115220

/-- Represents a cargo item with weight in tenths of a ton -/
structure Cargo where
  weight : Nat
  weight_le_100 : weight ≤ 100
deriving DecidableEq

/-- Represents the problem setup -/
structure CargoSetup where
  cargos : Finset Cargo
  total_weight_400 : (cargos.sum (fun c => c.weight)) = 4000
  distinct_weights : ∀ c1 c2, c1 ∈ cargos → c2 ∈ cargos → c1 ≠ c2 → c1.weight ≠ c2.weight

/-- The main theorem stating the minimum number of trips required -/
theorem min_trips (setup : CargoSetup) : 
  ∃ (trips : Finset (Finset Cargo)), 
    (∀ t, t ∈ trips → (t.sum (fun c => c.weight)) ≤ 100) ∧ 
    (trips.biUnion id = setup.cargos) ∧
    (trips.card = 51) ∧
    (∀ trips' : Finset (Finset Cargo), 
      (∀ t, t ∈ trips' → (t.sum (fun c => c.weight)) ≤ 100) → 
      (trips'.biUnion id = setup.cargos) → 
      trips'.card ≥ 51) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trips_l1152_115220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_digit_numbers_from_1357_l1152_115224

theorem sum_of_four_digit_numbers_from_1357 : 
  let digits : Finset ℕ := {1, 3, 5, 7}
  let four_digit_numbers := Finset.filter (fun n => n ≥ 1000 ∧ n < 10000 ∧ 
    (Finset.card (Finset.filter (fun d => d ∈ digits) (Finset.range 10))) = 4) (Finset.range 10000)
  (Finset.sum four_digit_numbers id) = 106656 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_digit_numbers_from_1357_l1152_115224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leg_length_is_ten_sqrt_two_l1152_115227

/-- An isosceles right triangle with a median to the hypotenuse of length 10 -/
structure IsoscelesRightTriangle where
  /-- The length of the median to the hypotenuse -/
  median_length : ℝ
  /-- The median length is 10 -/
  median_is_ten : median_length = 10

/-- The length of a leg in an isosceles right triangle -/
noncomputable def leg_length (t : IsoscelesRightTriangle) : ℝ :=
  10 * Real.sqrt 2

/-- Theorem: In an isosceles right triangle where the median to the hypotenuse
    has length 10, the length of a leg is 10√2 -/
theorem leg_length_is_ten_sqrt_two (t : IsoscelesRightTriangle) :
  leg_length t = 10 * Real.sqrt 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leg_length_is_ten_sqrt_two_l1152_115227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_agnes_works_eight_hours_l1152_115242

/-- Agnes' weekly work hours given Mila and Agnes' hourly rates and Mila's equivalent monthly hours -/
noncomputable def agnes_weekly_hours (mila_rate : ℚ) (agnes_rate : ℚ) (mila_monthly_hours : ℚ) : ℚ :=
  (mila_rate * mila_monthly_hours) / (agnes_rate * 4)

/-- Theorem stating Agnes works 8 hours per week given the problem conditions -/
theorem agnes_works_eight_hours :
  agnes_weekly_hours 10 15 48 = 8 := by
  -- Unfold the definition of agnes_weekly_hours
  unfold agnes_weekly_hours
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_agnes_works_eight_hours_l1152_115242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_value_l1152_115202

/-- Triangle ABC with incenter I, circumradius R, and inradius r -/
structure TriangleABC where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Incenter I of the triangle -/
  I : ℝ × ℝ
  /-- Circumradius of the triangle -/
  R : ℝ
  /-- Inradius of the triangle -/
  r : ℝ
  /-- I is the incenter of triangle ABC -/
  incenter_condition : True  -- Placeholder for the incenter condition
  /-- Relation between vectors IA, BI, and CI -/
  vector_relation : 5 * (I.1 - A.1, I.2 - A.2) = 4 * ((B.1 - I.1, B.2 - I.2) + (C.1 - I.1, C.2 - I.2))
  /-- Inradius r is equal to 15 -/
  inradius_value : r = 15

/-- The main theorem stating that under given conditions, R = 32 -/
theorem circumradius_value (t : TriangleABC) : t.R = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_value_l1152_115202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_at_fifteen_feet_l1152_115234

/-- Represents a parabolic arch -/
structure ParabolicArch where
  maxHeight : ℝ
  span : ℝ

/-- Calculates the height of a parabolic arch at a given horizontal distance -/
noncomputable def archHeight (arch : ParabolicArch) (x : ℝ) : ℝ :=
  let a := -arch.maxHeight / ((arch.span / 2) ^ 2)
  a * (x - arch.span / 2) ^ 2 + arch.maxHeight

theorem height_at_fifteen_feet (arch : ParabolicArch) 
  (h1 : arch.maxHeight = 25)
  (h2 : arch.span = 60) :
  archHeight arch 15 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_at_fifteen_feet_l1152_115234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_range_theorem_l1152_115203

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- State the theorem
theorem extremum_and_range_theorem (a b : ℝ) :
  a > 1 ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1-ε) (-1+ε), f a b (-1) ≤ f a b x) ∧
  f a b (-1) = 0 →
  a = 2 ∧ b = 9 ∧
  (∀ x ∈ Set.Icc (-4) 0, f a b x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-4) 0, f a b x = 4) ∧
  (∀ x ∈ Set.Icc (-4) 0, f a b x ≥ 0) ∧
  (∃ x ∈ Set.Icc (-4) 0, f a b x = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_range_theorem_l1152_115203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_16_2_l1152_115238

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_expression_equals_16_2 (y : ℝ) (h : y = 7.2) :
  (floor 6.5) * (floor (2 / 3)) + (floor 2) * y + (floor 8.4) - 6.2 = 16.2 := by
  -- Replace the entire proof with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_equals_16_2_l1152_115238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l1152_115285

theorem product_of_four_integers (A B C D : ℕ) :
  A > 0 → B > 0 → C > 0 → D > 0 →
  A + B + C + D = 72 →
  A + 2 = B - 2 →
  A + 2 = C * 2 →
  A + 2 = D / 2 →
  A * B * C * D = 64512 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l1152_115285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taylor_series_correct_l1152_115200

noncomputable def taylor_series_reciprocal (x : ℝ) : ℝ := 
  -1/2 * (∑' n, ((x + 2) / 2) ^ n)

noncomputable def taylor_series_cos (x : ℝ) : ℝ := 
  Real.sqrt 2 / 2 * (∑' n, ((-1)^n / Nat.factorial (2*n)) * (x - Real.pi/4)^(2*n))

theorem taylor_series_correct (x : ℝ) : 
  (x ≠ 0 ∧ -4 < x ∧ x < 0) → 
  taylor_series_reciprocal x = 1/x ∧ 
  taylor_series_cos x = Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taylor_series_correct_l1152_115200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1152_115267

/-- Given an ellipse and a hyperbola sharing the same foci, prove that the hyperbola's asymptotes have a specific equation. -/
theorem hyperbola_asymptotes (a b c : ℝ) (h_ellipse : a^2 = 25 ∧ b^2 = 9)
  (h_foci : c^2 = a^2 - b^2)
  (h_eccentricity_sum : c/a + 2 = 14/5) :
  let e_hyperbola := 2
  let a_hyperbola := c / e_hyperbola
  let b_hyperbola := Real.sqrt (c^2 - a_hyperbola^2)
  b_hyperbola / a_hyperbola = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1152_115267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_to_add_l1152_115215

theorem least_number_to_add : 
  ∃ n : ℕ, 
    (∀ m : ℕ, m < n → ¬((1789 + m) % 5 = 0 ∧ (1789 + m) % 6 = 0 ∧ (1789 + m) % 4 = 0 ∧ (1789 + m) % 3 = 0)) ∧
    ((1789 + n) % 5 = 0 ∧ (1789 + n) % 6 = 0 ∧ (1789 + n) % 4 = 0 ∧ (1789 + n) % 3 = 0) ∧
    n = 11 := by
  sorry

#check least_number_to_add

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_to_add_l1152_115215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_close_in_rectangle_l1152_115272

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Check if a point is inside a rectangle -/
def isInside (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

theorem two_points_close_in_rectangle :
  ∀ (pts : Finset Point),
    pts.card = 6 →
    (∀ p, p ∈ pts → isInside p { width := 4, height := 3 }) →
    ∃ p1 p2, p1 ∈ pts ∧ p2 ∈ pts ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_close_in_rectangle_l1152_115272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimizing_numbers_exist_l1152_115284

/-- A type representing a five-digit number --/
def FiveDigitNumber := Fin 100000

/-- Function to check if a number uses each digit from 0 to 9 exactly once --/
def usesEachDigitOnce (n m : FiveDigitNumber) : Prop := sorry

/-- Function to calculate the sum of digits of a number --/
def digitSum (n : FiveDigitNumber) : ℕ := sorry

/-- Function to calculate the product of two numbers --/
def product (n m : FiveDigitNumber) : ℕ := sorry

/-- Predicate to check if a pair of numbers minimizes both sum of digits and product --/
def isMinimizing (n m : FiveDigitNumber) : Prop := sorry

/-- Helper function to convert a natural number to FiveDigitNumber --/
def natToFiveDigitNumber (n : ℕ) : FiveDigitNumber :=
  ⟨n % 100000, by
    apply Nat.mod_lt
    exact Nat.zero_lt_succ _⟩

theorem minimizing_numbers_exist :
  ∃ (n m : FiveDigitNumber),
    usesEachDigitOnce n m ∧
    isMinimizing n m ∧
    (n = natToFiveDigitNumber 3489 ∨ n = natToFiveDigitNumber 12567 ∨
     m = natToFiveDigitNumber 3489 ∨ m = natToFiveDigitNumber 12567) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimizing_numbers_exist_l1152_115284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l1152_115265

theorem equidistant_point : 
  let p : ℝ × ℝ × ℝ := (11/10, -21/20, 0)
  let a : ℝ × ℝ × ℝ := (2, 0, -1)
  let b : ℝ × ℝ × ℝ := (1, -2, 3)
  let c : ℝ × ℝ × ℝ := (4, 2, 1)
  let dist (x y : ℝ × ℝ × ℝ) := 
    (x.1 - y.1)^2 + (x.2.1 - y.2.1)^2 + (x.2.2 - y.2.2)^2
  dist p a = dist p b ∧ dist p a = dist p c := by
  sorry

#check equidistant_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l1152_115265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_positive_reals_l1152_115213

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt x

-- State the theorem
theorem range_of_f_is_positive_reals :
  ∀ y : ℝ, y > 0 ↔ ∃ x : ℝ, x > 0 ∧ f x = y :=
by
  sorry

#check range_of_f_is_positive_reals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_positive_reals_l1152_115213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_filling_result_l1152_115260

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

def box_filling_process (n : ℕ) : ℕ × ℕ :=
  sorry

theorem box_filling_result :
  box_filling_process (factorial 33) = (36, 31) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_filling_result_l1152_115260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1152_115254

theorem product_remainder (a b c d : ℕ) 
  (ha : a % 7 = 2)
  (hb : b % 7 = 3)
  (hc : c % 7 = 4)
  (hd : d % 7 = 5) : 
  (a * b * c * d) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1152_115254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_range_l1152_115281

/-- The smallest angle in a triangle, in radians -/
noncomputable def x : ℝ := sorry

/-- The function y = √2 * sin(x + 45°) -/
noncomputable def y (θ : ℝ) : ℝ := Real.sqrt 2 * Real.sin (θ + Real.pi/4)

/-- Assumption that x is the smallest angle in a triangle -/
axiom x_is_smallest : 0 < x ∧ x ≤ Real.pi/3

/-- The range of y is (1, √2] -/
theorem y_range : Set.range y = Set.Ioo 1 (Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_range_l1152_115281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_speed_is_pi_third_l1152_115223

/-- Represents a rectangular track with semicircular ends -/
structure Track where
  width : ℝ
  time_difference : ℝ

/-- Calculates the walking speed given a track -/
noncomputable def walking_speed (track : Track) : ℝ :=
  (2 * Real.pi * track.width) / track.time_difference

/-- Theorem stating that for the given track, the walking speed is π/3 -/
theorem walking_speed_is_pi_third (track : Track)
  (h_width : track.width = 8)
  (h_time : track.time_difference = 48) :
  walking_speed track = Real.pi / 3 := by
  sorry

#check walking_speed_is_pi_third

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_speed_is_pi_third_l1152_115223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_analysis_l1152_115209

/-- Represents the daily sales deviations from the planned amount -/
def salesDeviations : List Int := [8, -2, 17, 22, -5, -8, -3]

/-- The planned sales amount per day in kilograms -/
def plannedSales : Nat := 100

/-- The selling price per kilogram in yuan -/
def sellingPrice : Rat := 5/2

/-- The shipping cost per kilogram in yuan -/
def shippingCost : Rat := 1/2

theorem sales_analysis :
  -- Part 1: Total sales for first three days
  (List.take 3 salesDeviations).sum + 3 * plannedSales = 323 ∧
  -- Part 2: Difference between highest and lowest sales
  (∀ x ∈ salesDeviations, x ≤ 22) ∧
  (∀ x ∈ salesDeviations, x ≥ -8) ∧
  22 - (-8) = 30 ∧
  -- Part 3: Total earnings
  let totalSales := salesDeviations.sum + 7 * plannedSales
  totalSales * (sellingPrice - shippingCost) = 1458 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_analysis_l1152_115209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelves_full_percentage_l1152_115214

/-- Calculates the percentage of full shelves given the number of ridges per record,
    number of cases, shelves per case, records per shelf, and total ridges. -/
theorem shelves_full_percentage
  (ridges_per_record : ℕ)
  (num_cases : ℕ)
  (shelves_per_case : ℕ)
  (records_per_shelf : ℕ)
  (total_ridges : ℕ)
  (h1 : ridges_per_record = 60)
  (h2 : num_cases = 4)
  (h3 : shelves_per_case = 3)
  (h4 : records_per_shelf = 20)
  (h5 : total_ridges = 8640) :
  (total_ridges / ridges_per_record) / (num_cases * shelves_per_case * records_per_shelf : ℚ) = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelves_full_percentage_l1152_115214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_75_to_half_l1152_115293

/-- Rounding function to the nearest half -/
noncomputable def roundToHalf (x : ℝ) : ℝ :=
  if x - ⌊x⌋ < 0.25 then ⌊x⌋
  else if x - ⌊x⌋ > 0.75 then ⌈x⌉
  else ⌊x⌋ + 0.5

/-- Theorem stating that 4.75 rounded to the nearest half is 5.0 -/
theorem round_4_75_to_half : roundToHalf 4.75 = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_4_75_to_half_l1152_115293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameter_sum_l1152_115261

-- Define the foci
def F₁ : ℝ × ℝ := (0, 2)
def F₂ : ℝ × ℝ := (6, 2)

-- Define the constant sum of distances
def distance_sum : ℝ := 10

-- Define the ellipse parameters
noncomputable def h : ℝ := (F₁.1 + F₂.1) / 2
noncomputable def k : ℝ := (F₁.2 + F₂.2) / 2
noncomputable def a : ℝ := distance_sum / 2
noncomputable def c : ℝ := Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) / 2
noncomputable def b : ℝ := Real.sqrt (a^2 - c^2)

-- Theorem statement
theorem ellipse_parameter_sum : h + k + a + b = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameter_sum_l1152_115261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_four_l1152_115268

-- Define the fraction
def fraction : ℚ := 5 / 7

-- Define the decimal representation as a sequence of digits
def decimal_rep : ℕ → ℕ
| n => match n % 6 with
  | 0 => 7
  | 1 => 1
  | 2 => 4
  | 3 => 2
  | 4 => 8
  | 5 => 5
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

-- Define the length of the repeating cycle
def cycle_length : ℕ := 6

-- Define the count of 4's in one cycle
def count_fours : ℕ := 1

-- Theorem statement
theorem probability_of_four :
  (count_fours : ℚ) / cycle_length = 1 / 6 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_four_l1152_115268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_l1152_115277

/-- Given that -x^(m+1)*y^3 + (3/2)*y^n*x^2 is a monomial, prove that (3*m^2 - 4*m*n) - 2*(m^2 + 2*m*n) = -23 -/
theorem simplify_and_evaluate (m n : ℤ) 
  (h : ∃ (k : ℚ) (p q : ℤ), -x^(m+1) * y^3 + (3/2) * y^n * x^2 = k * x^p * y^q) :
  (3*m^2 - 4*m*n) - 2*(m^2 + 2*m*n) = -23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_evaluate_l1152_115277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_is_8pi_l1152_115218

-- Define the radius of the original circle
def radius : ℝ := 6

-- Define the angle of the sector in degrees
def sector_angle : ℝ := 240

-- Define pi (using noncomputable as it depends on Real.pi)
noncomputable def π : ℝ := Real.pi

-- Define the circumference of the base of the cone (using noncomputable as it depends on π)
noncomputable def cone_base_circumference : ℝ := (sector_angle / 360) * (2 * π * radius)

-- Theorem statement
theorem cone_base_circumference_is_8pi :
  cone_base_circumference = 8 * π := by
  -- Expand the definition of cone_base_circumference
  unfold cone_base_circumference
  -- Simplify the expression
  simp [π, radius, sector_angle]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_is_8pi_l1152_115218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_type_quadratic_radicals_l1152_115246

theorem same_type_quadratic_radicals :
  ∃ (a b c : ℚ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a * a * b : ℚ) = 24 ∧ (c * c * b : ℚ) = 54 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_type_quadratic_radicals_l1152_115246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_theorem_l1152_115253

/-- The time taken for two workers to complete a job together, given their individual completion times -/
noncomputable def combined_work_time (x_time y_time : ℝ) : ℝ :=
  (x_time * y_time) / (x_time + y_time)

/-- Theorem: Two workers completing a job in 15 and 45 days respectively will together complete it in 11.25 days -/
theorem combined_work_theorem :
  combined_work_time 15 45 = 11.25 := by
  -- Unfold the definition of combined_work_time
  unfold combined_work_time
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_theorem_l1152_115253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_win_probability_correct_expected_flips_correct_l1152_115283

/-- A coin flipping game where the outcome is determined by specific conditions. -/
structure CoinFlipGame where
  /-- The probability of getting heads on a single coin flip. -/
  p_heads : ℝ
  /-- Assumption that the coin is fair. -/
  fair_coin : p_heads = 1/2

/-- The probability of winning for the player who wins on even-numbered heads. -/
noncomputable def win_probability (game : CoinFlipGame) : ℝ := 1/3

/-- The expected number of coin flips until the game ends. -/
noncomputable def expected_flips (game : CoinFlipGame) : ℝ := 2

/-- Theorem stating the win probability for the player who wins on even-numbered heads. -/
theorem win_probability_correct (game : CoinFlipGame) :
  win_probability game = 1/3 := by sorry

/-- Theorem stating the expected number of coin flips until the game ends. -/
theorem expected_flips_correct (game : CoinFlipGame) :
  expected_flips game = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_win_probability_correct_expected_flips_correct_l1152_115283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_equation_correct_l1152_115288

/-- Represents the daily production rate of person A -/
def x : ℝ := sorry

/-- The total daily production of A and B combined -/
def total_production : ℝ := 130

/-- The number of parts A produces in a reference time period -/
def a_reference_production : ℝ := 240

/-- The number of parts B produces in the same reference time period as A -/
def b_reference_production : ℝ := 280

/-- Theorem stating that the equation correctly represents the production scenario -/
theorem production_equation_correct :
  x > 0 ∧ x < total_production →
  a_reference_production / x = b_reference_production / (total_production - x) := by
  sorry

#check production_equation_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_equation_correct_l1152_115288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_specific_selection_l1152_115298

def jar_contents : List ℕ := [6, 4, 3, 2]
def total_balls : ℕ := jar_contents.sum
def selected_balls : ℕ := 5

def blue_balls : ℕ := 4
def green_balls : ℕ := 3
def yellow_balls : ℕ := 2

def specific_selection : ℕ × ℕ × ℕ := (3, 1, 1)

theorem probability_of_specific_selection :
  (Nat.choose blue_balls specific_selection.1 *
   Nat.choose green_balls specific_selection.2.1 *
   Nat.choose yellow_balls specific_selection.2.2) /
  Nat.choose total_balls selected_balls = 8 / 1001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_specific_selection_l1152_115298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l1152_115236

theorem triangle_cosine_inequality (A B C : ℝ) : 
  0 < A → 0 < B → 0 < C → 
  A + B + C = Real.pi → 
  Real.cos A + Real.cos B * Real.cos C ≤ 1 ∧ 
  (Real.cos A + Real.cos B * Real.cos C = 1 ↔ A = 0 ∧ B = Real.pi/2 ∧ C = Real.pi/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l1152_115236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l1152_115263

noncomputable def expansion (a : ℝ) (x : ℝ) : ℝ := (a / x - Real.sqrt x) ^ 6

theorem expansion_properties (a : ℝ) (h1 : a > 0) (h2 : expansion a 1 = 60) :
  a = 2 ∧
  ∃ (t : ℝ), t = -12 ∧ 
  ∃ (s : Set ℝ), s = {120, -2} ∧
  ∃ (l : ℝ), l = -960 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l1152_115263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_min_value_of_m_l1152_115231

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by sorry

-- Theorem for the minimum value of m
theorem min_value_of_m :
  ∃ (m : ℝ), m ≥ -Real.pi/3 ∧
  (∀ (x : ℝ), -Real.pi/3 ≤ x → x ≤ m → f x ≤ 3/2) ∧
  f m = 3/2 ∧
  (∀ (m' : ℝ), m' ≥ -Real.pi/3 →
    (∀ (x : ℝ), -Real.pi/3 ≤ x → x ≤ m' → f x ≤ 3/2) →
    f m' = 3/2 → m ≤ m') ∧
  m = Real.pi/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_min_value_of_m_l1152_115231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_tangents_theorem_l1152_115271

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

structure Circle where
  center : Point
  radius : ℝ

-- Define the necessary functions
def Median (t : Triangle) (v : Point) : Set Point :=
  sorry

def CircleOverMedian (t : Triangle) (v : Point) : Circle :=
  sorry

noncomputable def TangentLength (p : Point) (c : Circle) : ℝ :=
  sorry

def Altitude (t : Triangle) (v : Point) : Set Point :=
  sorry

-- Main theorem
theorem equal_tangents_theorem (t : Triangle) :
  let k_a := CircleOverMedian t t.A
  let k_b := CircleOverMedian t t.B
  ∀ (p : Point),
    TangentLength p k_a = TangentLength p k_b →
    p ∈ Altitude t t.C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_tangents_theorem_l1152_115271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_time_ratio_l1152_115228

noncomputable def printer_x_time : ℝ := 16
noncomputable def printer_y_time : ℝ := 10
noncomputable def printer_z_time : ℝ := 20

noncomputable def printer_x_rate : ℝ := 1 / printer_x_time
noncomputable def printer_y_rate : ℝ := 1 / printer_y_time
noncomputable def printer_z_rate : ℝ := 1 / printer_z_time

noncomputable def printers_yz_rate : ℝ := printer_y_rate + printer_z_rate
noncomputable def printers_yz_time : ℝ := 1 / printers_yz_rate

theorem printer_time_ratio :
  printer_x_time / printers_yz_time = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_printer_time_ratio_l1152_115228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jesse_always_wins_l1152_115280

/-- Represents the state of the game -/
structure GameState where
  n : ℕ
  blackBox : List ℝ
  whiteBox : List ℝ

/-- Represents a move in the game -/
inductive Move
  | Place (value : ℝ) (toBlack : Bool)
  | Move (fromBlack : Bool)

/-- Jesse's strategy function type -/
def Strategy := GameState → Move

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Prop :=
  state.blackBox.length = state.n / 2 ∧ state.whiteBox.length = (state.n + 1) / 2

/-- Checks if Jesse wins given the final state -/
noncomputable def jesseWins (state : GameState) : Prop :=
  state.blackBox.sum > state.whiteBox.sum

/-- Applies the moves to the initial game state -/
def applyMoves (n : ℕ) (strategy : Strategy) (tjeerd_moves : List Move) : GameState :=
  sorry

/-- Theorem: Jesse has a winning strategy for all n ≥ 2 -/
theorem jesse_always_wins (n : ℕ) (h : n ≥ 2) :
  ∃ (strategy : Strategy), ∀ (tjeerd_moves : List Move),
    let final_state := applyMoves n strategy tjeerd_moves
    isGameOver final_state → jesseWins final_state := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jesse_always_wins_l1152_115280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_eq_1176_l1152_115275

/-- The number of distinct ordered triples (x, y, z) of positive integers satisfying x + y + z = 50 -/
def count_solutions : ℕ :=
  Finset.card (Finset.filter
    (fun t : ℕ × ℕ × ℕ => t.1 + t.2.1 + t.2.2 = 50 ∧ t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0)
    (Finset.product (Finset.range 50) (Finset.product (Finset.range 50) (Finset.range 50))))

/-- Theorem stating that the number of solutions is 1176 -/
theorem count_solutions_eq_1176 : count_solutions = 1176 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_eq_1176_l1152_115275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l1152_115289

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b² + c² - a² = bc, AB · BC > 0, and a = √3/2,
    then √3/2 < b + c < 3/2 --/
theorem triangle_side_sum_range (a b c : ℝ) (A B C : ℝ) : 
  b^2 + c^2 - a^2 = b*c →
  (∃ (AB BC : ℝ × ℝ), AB.1 * BC.1 + AB.2 * BC.2 > 0) →
  a = Real.sqrt 3 / 2 →
  Real.sqrt 3 / 2 < b + c ∧ b + c < 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l1152_115289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1152_115210

/-- Determinant of a 2x2 matrix --/
def det (a₁ a₂ a₃ a₄ : ℝ) : ℝ := a₁ * a₄ - a₂ * a₃

/-- The function f after translation --/
noncomputable def f (x : ℝ) : ℝ := 
  det (Real.sin (2 * (x - Real.pi/6))) (Real.sqrt 3) (Real.cos (2 * (x - Real.pi/6))) 1

/-- Theorem stating that x = 7π/12 is an axis of symmetry for f --/
theorem axis_of_symmetry :
  ∀ (t : ℝ), f (7*Real.pi/12 + t) = f (7*Real.pi/12 - t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1152_115210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_renne_monthly_earnings_l1152_115207

noncomputable def vehicle_cost : ℝ := 16000
def saving_months : ℕ := 8

noncomputable def monthly_savings (monthly_earnings : ℝ) : ℝ := monthly_earnings / 2

theorem renne_monthly_earnings :
  ∃ (monthly_earnings : ℝ),
    monthly_savings monthly_earnings * (saving_months : ℝ) = vehicle_cost ∧
    monthly_earnings = 4000 :=
by
  use 4000
  constructor
  · simp [monthly_savings, vehicle_cost, saving_months]
    norm_num
  · rfl

#check renne_monthly_earnings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_renne_monthly_earnings_l1152_115207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1152_115258

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log (2 * x)

-- Theorem statement
theorem unique_intersection :
  ∃! x : ℝ, x > 0 ∧ f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1152_115258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_of_intersection_l1152_115291

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The focal distance of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

theorem hyperbola_eccentricity_of_intersection
  (h : Hyperbola)
  (c : ℝ)
  (h_intersect : c^2 / h.a^2 - (2*c)^2 / h.b^2 = 1)
  (h_focal : c = focal_distance h) :
  eccentricity h = Real.sqrt 2 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_of_intersection_l1152_115291
