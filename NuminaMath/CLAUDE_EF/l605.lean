import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_5_non_standard_approx_l605_60593

/-- The probability of manufacturing a non-standard part -/
def p : ℝ := 0.004

/-- The total number of parts -/
def n : ℕ := 1000

/-- The number of non-standard parts we're interested in -/
def k : ℕ := 5

/-- The mean number of non-standard parts -/
def a : ℝ := n * p

/-- The Poisson probability mass function -/
noncomputable def poisson_pmf (lambda : ℝ) (k : ℕ) : ℝ :=
  (lambda ^ k * Real.exp (-lambda)) / k.factorial

/-- The probability of exactly 5 non-standard parts out of 1000 -/
noncomputable def prob_5_non_standard : ℝ := poisson_pmf a k

theorem prob_5_non_standard_approx :
  abs (prob_5_non_standard - 0.1562) < 0.0001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_5_non_standard_approx_l605_60593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l605_60525

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define point X on AB such that AX/AB = 1/4
def X (A B : ℝ × ℝ) : ℝ × ℝ := (3/4 * A.1 + 1/4 * B.1, 3/4 * A.2 + 1/4 * B.2)

-- Define the centroid G of triangle ABC
def G (A B C : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Define point A' on median AG
def A' (A B C : ℝ × ℝ) : ℝ × ℝ := 
  let g := G A B C
  (2/5 * A.1 + 3/5 * g.1, 2/5 * A.2 + 3/5 * g.2)

-- Define point B'' on median BG
def B'' (A B C : ℝ × ℝ) : ℝ × ℝ := 
  let g := G A B C
  (4/7 * B.1 + 3/7 * g.1, 4/7 * B.2 + 3/7 * g.2)

-- Define other points similarly (B', C', A'', C'')
def B' (A B C : ℝ × ℝ) : ℝ × ℝ := 
  let g := G A B C
  (2/5 * B.1 + 3/5 * g.1, 2/5 * B.2 + 3/5 * g.2)

def C' (A B C : ℝ × ℝ) : ℝ × ℝ := 
  let g := G A B C
  (2/5 * C.1 + 3/5 * g.1, 2/5 * C.2 + 3/5 * g.2)

def A'' (A B C : ℝ × ℝ) : ℝ × ℝ := 
  let g := G A B C
  (4/7 * A.1 + 3/7 * g.1, 4/7 * A.2 + 3/7 * g.2)

def C'' (A B C : ℝ × ℝ) : ℝ × ℝ := 
  let g := G A B C
  (4/7 * C.1 + 3/7 * g.1, 4/7 * C.2 + 3/7 * g.2)

-- Function to calculate the area of a triangle given its vertices
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  1/2 * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Theorem statement
theorem area_ratio_theorem (A B C : ℝ × ℝ) :
  triangleArea (A'' A B C) (B'' A B C) (C'' A B C) / triangleArea (A' A B C) (B' A B C) (C' A B C) = 25/49 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l605_60525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_flight_cost_l605_60565

/-- Represents the cost function for an airplane flight -/
noncomputable def flightCost (a m n p : ℝ) (x : ℝ) : ℝ :=
  p * (m * x^2 / a^3 + n / x)

/-- Theorem stating the minimum cost and optimal speed for an airplane flight -/
theorem optimal_flight_cost (a m n p : ℝ) (ha : a > 0) (hm : m > 0) (hn : n > 0) (hp : p > 0) :
  let optimal_speed := a * (n / (2 * m))^(1/3)
  let min_cost := (3 * p / a) * ((m * n^2) / 4)^(1/3)
  (∀ x > 0, flightCost a m n p x ≥ min_cost) ∧
  flightCost a m n p optimal_speed = min_cost := by
  sorry

#check optimal_flight_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_flight_cost_l605_60565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_share_is_1400_l605_60561

/-- Represents the investment and profit distribution in a partnership --/
structure Partnership where
  a_invest : ℚ
  b_invest : ℚ
  c_invest : ℚ
  total_profit : ℚ

/-- Calculates the share of profit for partner B --/
def share_of_b (p : Partnership) : ℚ :=
  (p.b_invest / (p.a_invest + p.b_invest + p.c_invest)) * p.total_profit

/-- Theorem stating that B's share of the profit is 1400 --/
theorem b_share_is_1400 (p : Partnership) 
  (h1 : p.a_invest = 3 * p.b_invest)
  (h2 : p.b_invest = 2/3 * p.c_invest)
  (h3 : p.total_profit = 7700) :
  share_of_b p = 1400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_share_is_1400_l605_60561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_theorem_l605_60566

def sequenceProperty (a : ℕ → ℚ) : Prop :=
  a 1 = 0 ∧ a 2 = 2 ∧
  ∀ k : ℕ, k ≥ 1 →
    (a (2*k - 1) + a (2*k + 1) = 2 * a (2*k)) ∧
    ∃ q : ℚ, a (2*k + 1) = a (2*k) * q ∧ a (2*k + 2) = a (2*k + 1) * q

def commonRatio (a : ℕ → ℚ) (k : ℕ) : ℚ :=
  (a (2*k + 1)) / (a (2*k))

theorem sequence_ratio_theorem (a : ℕ → ℚ) (h : sequenceProperty a) :
  commonRatio a 10 = 11/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_theorem_l605_60566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_l605_60500

theorem max_value_of_a (a b c : ℚ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a ∈ ({1, 2, 4} : Set ℚ) ∧ b ∈ ({1, 2, 4} : Set ℚ) ∧ c ∈ ({1, 2, 4} : Set ℚ) →
  (a / 2) / (b / c) ≤ 4 →
  (∃ b' c' : ℚ, b' ≠ c' ∧ 
    b' ∈ ({1, 2, 4} : Set ℚ) ∧ c' ∈ ({1, 2, 4} : Set ℚ) ∧ 
    (4 / 2) / (b' / c') = 4) →
  a = 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_l605_60500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equality_l605_60567

-- Define the quadrilateral ABCD
variable (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the point E
variable (E : EuclideanSpace ℝ (Fin 2))

-- Assumption that ABCD is convex
variable (h_convex : ConvexHull {A, B, C, D})

-- Assumption that E is on the line through A parallel to BD
variable (h_E_on_A_parallel_BD : ∃ (t : ℝ), E = A + t • (D - B))

-- Assumption that E is on the line through B parallel to AC
variable (h_E_on_B_parallel_AC : ∃ (s : ℝ), E = B + s • (C - A))

-- Define points G and F
variable (G F : EuclideanSpace ℝ (Fin 2))

-- Assumption that G is on BD and EC
variable (h_G_on_BD : ∃ (u : ℝ), G = B + u • (D - B))
variable (h_G_on_EC : ∃ (v : ℝ), G = E + v • (C - E))

-- Assumption that F is on AC and ED
variable (h_F_on_AC : ∃ (w : ℝ), F = A + w • (C - A))
variable (h_F_on_ED : ∃ (x : ℝ), F = E + x • (D - E))

-- The theorem to be proved
theorem ratio_equality :
  ‖G - B‖ / ‖D - B‖ = ‖F - A‖ / ‖C - A‖ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equality_l605_60567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_ratio_l605_60574

theorem rectangle_triangle_ratio (t x y : ℝ) (h1 : t > 0) (h2 : x > 0) (h3 : y > 0) 
  (h4 : x = t) (h5 : (Real.sqrt 3 / 4) * t^2 = 3 * (x * y)) : x / y = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_ratio_l605_60574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_tank_water_height_l605_60573

/-- Represents the height of water in a conical tank -/
structure WaterHeight where
  c : ℕ
  d : ℕ
  h : ℝ
  h_eq : h = c * (d : ℝ) ^ (1/3 : ℝ)

/-- Properties of the conical water tank problem -/
structure ConicalTank where
  radius : ℝ
  height : ℝ
  fillRatio : ℝ
  waterHeight : WaterHeight
  radius_pos : radius > 0
  height_pos : height > 0
  fill_ratio_valid : 0 < fillRatio ∧ fillRatio < 1
  d_not_perfect_cube : ∀ (n : ℕ), n > 1 → waterHeight.d % (n^3) ≠ 0

/-- The main theorem to be proved -/
theorem conical_tank_water_height (tank : ConicalTank) 
  (h_radius : tank.radius = 10)
  (h_height : tank.height = 60)
  (h_fill_ratio : tank.fillRatio = 0.4) :
  tank.waterHeight.c = 30 ∧ 
  tank.waterHeight.d = 2 ∧ 
  tank.waterHeight.c + tank.waterHeight.d = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_tank_water_height_l605_60573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_triangle_max_area_l605_60592

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

def is_monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

theorem f_monotone_increasing (k : ℤ) : 
  is_monotone_increasing f (k * Real.pi - Real.pi/12) (k * Real.pi + 5*Real.pi/12) := by
  sorry

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi/2 →
  0 < B ∧ B < Real.pi/2 →
  0 < C ∧ C < Real.pi/2 →
  A + B + C = Real.pi →
  f A = Real.sqrt 3 / 4 →
  a = Real.sqrt 3 →
  a = b * Real.sin C →
  a = c * Real.sin B →
  b = c * Real.sin A →
  (1/2 * b * c * Real.sin A) ≤ 3 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_triangle_max_area_l605_60592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l605_60537

/-- Represents the capital shares and profit of a partnership --/
structure Partnership where
  total_capital : ℚ
  a_share : ℚ
  b_share : ℚ
  c_share : ℚ
  d_share : ℚ
  a_profit : ℚ

/-- The partnership satisfies the given conditions --/
def valid_partnership (p : Partnership) : Prop :=
  p.a_share = (1/3) * p.total_capital ∧
  p.b_share = (1/4) * p.total_capital ∧
  p.c_share = (1/5) * p.total_capital ∧
  p.d_share = p.total_capital - (p.a_share + p.b_share + p.c_share) ∧
  p.a_profit = 815

/-- The total profit of the partnership --/
def total_profit (p : Partnership) : ℚ :=
  p.a_profit * (p.total_capital / p.a_share)

/-- Theorem: The total profit is 2445 for a valid partnership --/
theorem partnership_profit (p : Partnership) (h : valid_partnership p) :
  total_profit p = 2445 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l605_60537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enrollment_difference_l605_60516

/-- Represents the enrollment of a school --/
structure School where
  name : String
  enrollment : ℕ

/-- The set of schools given in the problem --/
def schools : List School := [
  ⟨"Varsity", 1250⟩,
  ⟨"Northwest", 1430⟩,
  ⟨"Central", 1900⟩,
  ⟨"Greenbriar", 1720⟩
]

/-- The theorem to be proved --/
theorem enrollment_difference : 
  (List.maximum? (schools.map School.enrollment)).isSome ∧ 
  (List.minimum? (schools.map School.enrollment)).isSome → 
  ((List.maximum? (schools.map School.enrollment)).get! - 
   (List.minimum? (schools.map School.enrollment)).get!) = 650 := by
  intro h
  sorry

#eval (List.maximum? (schools.map School.enrollment)).get! - 
      (List.minimum? (schools.map School.enrollment)).get!

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enrollment_difference_l605_60516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangleArea_correct_l605_60575

/-- The area of a triangle bounded by three lines -/
noncomputable def triangleArea (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℝ) : ℝ :=
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![a₁, b₁, c₁; a₂, b₂, c₂; a₃, b₃, c₃]
  let Δ := Matrix.det M
  abs (Δ^2) / (2 * abs (a₂ * b₃ - a₃ * b₂) * abs (a₃ * b₁ - a₁ * b₃) * abs (a₁ * b₂ - a₂ * b₁))

/-- Theorem stating that the triangleArea function correctly computes the area of a triangle
    bounded by three lines -/
theorem triangleArea_correct (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℝ) :
  let l₁ : ℝ → ℝ → ℝ := λ x y => a₁ * x + b₁ * y + c₁
  let l₂ : ℝ → ℝ → ℝ := λ x y => a₂ * x + b₂ * y + c₂
  let l₃ : ℝ → ℝ → ℝ := λ x y => a₃ * x + b₃ * y + c₃
  ∃ (A : ℝ), A = triangleArea a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ ∧ A ≥ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangleArea_correct_l605_60575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_specific_complex_l605_60529

theorem magnitude_of_specific_complex : Complex.abs (-1 + 3 * Complex.I) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_specific_complex_l605_60529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_with_nines_l605_60527

theorem power_of_two_with_nines (k : ℕ) (hk : k > 1) :
  ∃ n : ℕ, ∃ m : ℕ, (((2^n : ℕ) % (10^k : ℕ)).digits 10).count 9 ≥ k / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_with_nines_l605_60527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_nineteen_twelfths_l605_60594

-- Define the function g as noncomputable
noncomputable def g (x y : ℝ) : ℝ :=
  if x + y ≤ 5 then
    (3 * x * y - x + 3) / (3 * x)
  else
    (x * y - y - 3) / (-3 * y)

-- State the theorem
theorem g_sum_equals_nineteen_twelfths :
  g 3 2 + g 3 4 = 19 / 12 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_nineteen_twelfths_l605_60594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eldest_brother_age_l605_60512

theorem eldest_brother_age (a b c : ℝ) (ha : a > b) (hc : c = max b (min a c)) :
  c ^ 2 = a * b →
  a + b + c = 35 →
  Real.log a + Real.log b + Real.log c = 3 →
  a = 20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eldest_brother_age_l605_60512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_parallel_l605_60548

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (passes_through : Plane → Line → Prop)
variable (intersects : Plane → Plane → Line → Prop)

-- State the theorem
theorem intersection_line_parallel 
  (a : Line) (α β : Plane) (b : Line)
  (h1 : parallel_line_plane a α)
  (h2 : passes_through β a)
  (h3 : intersects β α b) :
  parallel_lines b a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_parallel_l605_60548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_proof_l605_60507

/-- Proves that the total number of students in a class is 50, given specific height averages. -/
theorem class_size_proof (n : ℕ) (r : ℕ) :
  n = 40 + r →
  40 * 169 + r * 167 = n * (1686 / 10 : ℚ) →
  n = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_proof_l605_60507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_arrangement_l605_60517

/-- Represents the number of type A carriages -/
def x : ℕ := 0

/-- Represents the total number of carriages -/
def total_carriages : ℕ := 40

/-- Represents the capacity of goods A for type A carriage -/
def capacity_A_A : ℕ := 35

/-- Represents the capacity of goods B for type A carriage -/
def capacity_A_B : ℕ := 15

/-- Represents the capacity of goods A for type B carriage -/
def capacity_B_A : ℕ := 25

/-- Represents the capacity of goods B for type B carriage -/
def capacity_B_B : ℕ := 35

/-- Represents the total amount of goods A to be transported -/
def total_goods_A : ℕ := 1240

/-- Represents the total amount of goods B to be transported -/
def total_goods_B : ℕ := 880

/-- Represents the cost of using a type A carriage (in ten thousand yuan) -/
def cost_A : ℚ := 6/10

/-- Represents the cost of using a type B carriage (in ten thousand yuan) -/
def cost_B : ℚ := 8/10

/-- Represents the total cost function -/
def total_cost (x : ℕ) : ℚ := cost_A * x + cost_B * (total_carriages - x)

/-- Theorem stating that 26 type A carriages minimizes the total cost -/
theorem min_cost_arrangement : 
  (∀ y : ℕ, y ≤ total_carriages → 
    y * capacity_A_A + (total_carriages - y) * capacity_B_A ≥ total_goods_A ∧
    y * capacity_A_B + (total_carriages - y) * capacity_B_B ≥ total_goods_B →
    total_cost y ≥ total_cost 26) :=
by
  sorry

#eval total_cost 26

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_arrangement_l605_60517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_distance_2_from_origin_line_through_P_max_distance_from_origin_l605_60564

/-- Point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (p : Point2D) (l : Line2D) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Distance from the origin to a line -/
noncomputable def distanceOriginToLine (l : Line2D) : ℝ :=
  abs l.c / Real.sqrt (l.a^2 + l.b^2)

/-- Check if a line passes through a point -/
def linePassesThroughPoint (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Given point P -/
def P : Point2D := { x := 2, y := -1 }

theorem line_through_P_distance_2_from_origin :
  ∃ l₁ l₂ : Line2D,
    (linePassesThroughPoint l₁ P ∧ distanceOriginToLine l₁ = 2 ∧ l₁ = { a := 1, b := 0, c := -2 }) ∧
    (linePassesThroughPoint l₂ P ∧ distanceOriginToLine l₂ = 2 ∧ l₂ = { a := 3, b := -4, c := -10 }) := by
  sorry

theorem line_through_P_max_distance_from_origin :
  ∃ l : Line2D, linePassesThroughPoint l P ∧
    (∀ l' : Line2D, linePassesThroughPoint l' P → distanceOriginToLine l' ≤ distanceOriginToLine l) ∧
    l = { a := 2, b := -1, c := -5 } ∧
    distanceOriginToLine l = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_distance_2_from_origin_line_through_P_max_distance_from_origin_l605_60564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_maps_curve_l605_60587

/-- The original curve equation -/
def original_curve (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The transformed curve equation -/
def transformed_curve (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

/-- The scaling transformation -/
def scaling_transform (x y : ℝ) : ℝ × ℝ := (2*x, 3*y)

/-- Theorem stating that the scaling transformation maps the original curve to the transformed curve -/
theorem scaling_transformation_maps_curve :
  ∀ (x y : ℝ), original_curve x y →
  transformed_curve ((scaling_transform x y).1) ((scaling_transform x y).2) := by
    intro x y h
    simp [original_curve, transformed_curve, scaling_transform] at *
    sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_maps_curve_l605_60587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_identity_l605_60579

theorem cosine_product_identity (α β : ℝ) : 
  Real.cos (α + β) * Real.cos (α - β) = Real.cos α ^ 2 - Real.sin β ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_identity_l605_60579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_properties_l605_60553

/-- The number of vertices in a hexagon -/
def hexagon_vertices : Nat := 6

/-- The number of sides in a hexagon -/
def hexagon_sides : Nat := 6

/-- Calculate the number of diagonals in a hexagon -/
def hexagon_diagonals : Nat :=
  (hexagon_vertices.choose 2) - hexagon_sides

/-- Calculate the total number of line segments (sides + diagonals) -/
def total_line_segments : Nat := hexagon_sides + hexagon_diagonals

/-- Calculate the number of intersections per vertex -/
def intersections_per_vertex : Nat := Nat.choose 5 2

theorem hexagon_properties :
  (hexagon_diagonals = 9) ∧
  (Nat.choose total_line_segments 2 - hexagon_vertices * intersections_per_vertex = 45) := by
  sorry

#eval hexagon_diagonals
#eval Nat.choose total_line_segments 2 - hexagon_vertices * intersections_per_vertex

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_properties_l605_60553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_of_0_0000314_l605_60539

theorem scientific_notation_of_0_0000314 :
  (0.0000314 : ℝ) = 3.14 * (10 : ℝ)^(-5 : ℤ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_of_0_0000314_l605_60539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_midpoints_sum_l605_60508

theorem triangle_midpoints_sum (a b c : ℝ) (h : a + b + c = 25) :
  (1.1 * a + 1.1 * b) / 2 + (1.1 * a + 1.1 * c) / 2 + (1.1 * b + 1.1 * c) / 2 = 27.5 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_midpoints_sum_l605_60508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l605_60547

noncomputable def f (x : Real) : Real := 4 * Real.sin x * Real.cos (x - Real.pi/3) - Real.sqrt 3

def isZeroPoint (x : Real) : Prop :=
  ∃ k : Int, x = Real.pi/6 + k * Real.pi/2

theorem f_properties :
  (∀ x, f (x + Real.pi) = f x) ∧  -- minimum positive period is π
  (∀ x, f x = 0 ↔ isZeroPoint x) ∧  -- zero points
  (∀ x ∈ Set.Icc (Real.pi/24) (3*Real.pi/4), f x ≤ 2) ∧  -- maximum value in interval
  (∀ x ∈ Set.Icc (Real.pi/24) (3*Real.pi/4), f x ≥ -Real.sqrt 2) ∧  -- minimum value in interval
  (∃ x ∈ Set.Icc (Real.pi/24) (3*Real.pi/4), f x = 2) ∧  -- maximum value is achieved
  (∃ x ∈ Set.Icc (Real.pi/24) (3*Real.pi/4), f x = -Real.sqrt 2)  -- minimum value is achieved
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l605_60547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l605_60530

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

noncomputable def g (x φ : ℝ) : ℝ := Real.sin (2 * x + 2 * Real.pi / 3 + φ)

theorem max_value_f (φ : ℝ) (h1 : |φ| < Real.pi / 2) 
  (h2 : ∀ x, g x φ = g (-x) φ) : 
  ∃ x ∈ Set.Icc 0 (Real.pi / 2), ∀ y ∈ Set.Icc 0 (Real.pi / 2), f x φ ≥ f y φ ∧ f x φ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l605_60530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_increase_l605_60505

/-- Represents the volume of a right circular cylinder in liters -/
def CylinderVolume : ℝ → ℝ := sorry

/-- The increase in volume when the radius is doubled -/
def VolumeIncrease (v : ℝ) : ℝ := CylinderVolume (2 * v) - CylinderVolume v

theorem cylinder_volume_increase (v : ℝ) :
  CylinderVolume v = 6 → VolumeIncrease v = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_increase_l605_60505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_distance_formula_l605_60596

/-- Regular triangular pyramid with base side length a and lateral face angle α -/
structure RegularTriangularPyramid where
  a : ℝ
  α : ℝ

/-- Distance between a lateral edge and the opposite side of the base -/
noncomputable def lateral_edge_distance (p : RegularTriangularPyramid) : ℝ :=
  (p.a * Real.sqrt 3 * Real.tan p.α) / (2 * Real.sqrt (4 + Real.tan p.α ^ 2))

/-- Theorem: The distance between a lateral edge and the opposite side of the base
    in a regular triangular pyramid is (a * √3 * tan(α)) / (2 * √(4 + tan²(α))) -/
theorem lateral_edge_distance_formula (p : RegularTriangularPyramid) :
  lateral_edge_distance p = (p.a * Real.sqrt 3 * Real.tan p.α) / (2 * Real.sqrt (4 + Real.tan p.α ^ 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_distance_formula_l605_60596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_is_four_l605_60597

def is_valid_set (s : Finset ℕ) : Prop :=
  s.Nonempty ∧ (∀ x y : ℕ, x ∈ s → y ∈ s → x ≠ y)

noncomputable def sum_product_ratio (s : Finset ℕ) : ℚ :=
  (s.sum (λ x => (x : ℚ))) / (s.prod (λ x => (x : ℚ)))

theorem smallest_number_is_four
  (s : Finset ℕ)
  (h_valid : is_valid_set s)
  (a : ℕ)
  (h_a_min : a = Finset.min' s h_valid.1)
  (h_ratio : sum_product_ratio (s.erase a) = 3 * sum_product_ratio s) :
  a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_is_four_l605_60597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_perimeter_less_l605_60523

/-- Represents a polygon in 2D space -/
structure Polygon where
  -- Define the polygon structure (vertices, edges, etc.)
  -- This is a simplified representation
  vertices : List (ℝ × ℝ)

/-- Represents a line segment connecting two points -/
structure Segment where
  start : ℝ × ℝ
  endPoint : ℝ × ℝ

/-- Calculates the perimeter of a polygon -/
noncomputable def perimeter (p : Polygon) : ℝ :=
  sorry

/-- Folds a polygon along a given segment -/
noncomputable def foldPolygon (p : Polygon) (s : Segment) : Polygon :=
  sorry

/-- Theorem: The perimeter of a folded polygon is less than the original -/
theorem folded_perimeter_less (p : Polygon) (s : Segment) :
  s.start ∈ p.vertices ∧ s.endPoint ∈ p.vertices →
  perimeter (foldPolygon p s) < perimeter p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_perimeter_less_l605_60523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l605_60534

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + Real.sqrt 3 * Real.sin (2 * x) - 1

theorem f_properties :
  (∃ x, f x = 2 ∧ ∀ y, f y ≤ 2) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3),
    ∀ y ∈ Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3),
    x < y → f x > f y) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 3), 
    f x ∈ Set.Icc (-1) 2) ∧
  (∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 3), f x = -1) ∧
  (∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 3), f x = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l605_60534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l605_60595

/-- A linear function f: ℝ → ℝ that is increasing and satisfies f(f(x)) = 4x - 1 -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is a linear function -/
axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

/-- f is an increasing function -/
axiom f_increasing : ∀ x y, x < y → f x < f y

/-- f satisfies the functional equation f(f(x)) = 4x - 1 -/
axiom f_equation : ∀ x, f (f x) = 4 * x - 1

/-- The main theorem: f(x) = 2x - 1/3 -/
theorem f_expression : ∀ x, f x = 2 * x - 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_l605_60595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_l605_60511

theorem points_on_line :
  ∀ (t : ℝ), t ≠ 0 →
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧
    a * ((2 * t + 2) / t) + b * ((2 * t - 2) / t) + c = 0 :=
by
  intro t ht
  use 1, 1, -4  -- Coefficients for the line x + y = 4
  constructor
  · left
    exact one_ne_zero
  · field_simp [ht]
    ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_l605_60511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l605_60522

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio q -/
noncomputable def geometricSum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with common ratio q, if 2S_4 = S_5 + S_6, then q = -2 -/
theorem geometric_sequence_common_ratio 
  (a : ℝ) (q : ℝ) (h : q ≠ 1) :
  2 * (geometricSum a q 4) = (geometricSum a q 5) + (geometricSum a q 6) →
  q = -2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l605_60522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ride_time_difference_l605_60577

/-- Represents a cyclist with their speeds on different terrains -/
structure Cyclist where
  flatSpeed : ℚ
  downhillSpeed : ℚ
  uphillSpeed : ℚ

/-- Represents a segment of the route -/
structure RouteSegment where
  distance : ℚ
  terrain : String

/-- Calculates the time taken for a cyclist to complete a route segment -/
def timeTaken (c : Cyclist) (s : RouteSegment) : ℚ :=
  match s.terrain with
  | "flat" => s.distance / c.flatSpeed
  | "downhill" => s.distance / c.downhillSpeed
  | "uphill" => s.distance / c.uphillSpeed
  | _ => 0  -- Default case, should not occur in our problem

/-- Calculates the total time taken for a cyclist to complete a route -/
def totalTime (c : Cyclist) (route : List RouteSegment) : ℚ :=
  route.foldl (fun acc seg => acc + timeTaken c seg) 0

theorem ride_time_difference :
  let minnie : Cyclist := { flatSpeed := 25, downhillSpeed := 30, uphillSpeed := 5 }
  let penny : Cyclist := { flatSpeed := 35, downhillSpeed := 45, uphillSpeed := 10 }
  let route : List RouteSegment := [
    { distance := 15, terrain := "uphill" },
    { distance := 10, terrain := "flat" },
    { distance := 20, terrain := "downhill" },
    { distance := 25, terrain := "flat" }
  ]
  let minnieTime := totalTime minnie route
  let pennyTime := totalTime penny (route.reverse)
  (minnieTime - pennyTime) * 60 = 104 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ride_time_difference_l605_60577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_pi_thirds_l605_60513

theorem cos_two_pi_thirds : Real.cos (2 * π / 3) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_pi_thirds_l605_60513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_g_monotonicity_sum_condition_l605_60598

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (2 * x) + Real.exp x + x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - (a + 3) * Real.exp x

-- Theorem 1: If f'(0) = 0, then a = -1
theorem extremum_condition (a : ℝ) : 
  (deriv (f a)) 0 = 0 → a = -1 := by
  sorry

-- Theorem 2: The monotonicity of g(x) depends on the value of a
theorem g_monotonicity (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    ((a ≤ 0 → g a x₁ > g a x₂) ∨
     (a > 0 → g a x₁ < g a x₂)) := by
  sorry

-- Theorem 3: When a = 2, if f(x₁) + f(x₂) + 3e^(x₁)e^(x₂) = 0, then e^(x₁) + e^(x₂) > 1/2
theorem sum_condition (x₁ x₂ : ℝ) :
  f 2 x₁ + f 2 x₂ + 3 * Real.exp x₁ * Real.exp x₂ = 0 →
  Real.exp x₁ + Real.exp x₂ > (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_g_monotonicity_sum_condition_l605_60598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_of_specific_cistern_l605_60532

/-- Represents a rectangular cistern partially filled with water. -/
structure Cistern where
  length : ℝ
  width : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the depth of water in a cistern given its dimensions and wet surface area. -/
noncomputable def waterDepth (c : Cistern) : ℝ :=
  (c.wetSurfaceArea - c.length * c.width) / (2 * (c.length + c.width))

/-- Theorem stating that for a cistern with given dimensions and wet surface area,
    the water depth is 1.25 meters. -/
theorem water_depth_of_specific_cistern :
  let c : Cistern := { length := 8, width := 4, wetSurfaceArea := 62 }
  waterDepth c = 1.25 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_of_specific_cistern_l605_60532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_255_l605_60521

def sequenceA (n : ℕ) : ℤ :=
  match n with
  | 0 => 0
  | n + 1 => 4 * sequenceA n + 3

theorem fifth_term_is_255 : sequenceA 4 = 255 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_255_l605_60521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_to_plane_parallel_to_all_lines_l605_60570

-- Define the basic geometric objects
variable (α : Type) -- Plane α
variable (a b : Type) -- Lines a and b

-- Define the geometric relations
def contains (plane : Type) (line : Type) : Prop := sorry
def parallel (line : Type) (plane : Type) : Prop := sorry
def parallel_lines (line1 : Type) (line2 : Type) : Prop := sorry

-- State the theorem to be disproved
theorem parallel_to_plane_parallel_to_all_lines (α : Type) (b : Type) :
  parallel b α → ∀ a, contains α a → parallel_lines b a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_to_plane_parallel_to_all_lines_l605_60570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_coloring_l605_60536

/-- Represents the color of a point -/
inductive Color where
  | Red
  | Blue
deriving BEq, Repr

/-- Represents a point on the circle or the center -/
structure Point where
  id : Nat
  color : Color
deriving BEq, Repr

/-- Represents the circle configuration -/
structure CircleConfig where
  points : List Point
  center : Point

/-- Checks if a point has the correct number of blue connections -/
def hasCorrectBlueConnections (config : CircleConfig) (p : Point) : Prop :=
  let blueConnections := (config.points.filter (fun q => 
    q.color == Color.Blue && 
    (q.id == (p.id + 1) % 2013 || q.id == (p.id + 2012) % 2013 || q == config.center)
  )).length
  match p.color with
  | Color.Red => Odd blueConnections
  | Color.Blue => Even blueConnections

/-- The main theorem stating the impossibility of the required coloring -/
theorem impossible_coloring (config : CircleConfig) :
  config.points.length = 2013 →
  (config.points.filter (fun p => p.color == Color.Red)).length = 1007 →
  (config.points.filter (fun p => p.color == Color.Blue)).length = 1007 →
  (∀ p ∈ config.points, hasCorrectBlueConnections config p) →
  False := by
  sorry

#check impossible_coloring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_coloring_l605_60536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_negative_one_fourth_l605_60580

noncomputable def g (a b : ℝ) : ℝ :=
  if a + b ≤ 4 then
    (a^2 * b - a^2 + 3) / (2 * a)
  else
    (a^2 * b - b^2 - 3) / (-2 * b)

theorem g_sum_equals_negative_one_fourth :
  g 1 3 + g 3 2 = -1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_negative_one_fourth_l605_60580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_points_is_16_l605_60506

/-- A configuration of red points on a 6x6 grid -/
def RedConfig := Fin 7 → Fin 7 → Bool

/-- Checks if a point is on the boundary of a subgrid -/
def onBoundary (i j x y : Fin 7) (k : Fin 6) : Prop :=
  (i = x ∨ i = x + k) ∧ y ≤ j ∧ j ≤ y + k ∨
  (j = y ∨ j = y + k) ∧ x ≤ i ∧ i ≤ x + k

/-- Checks if a configuration satisfies the red point condition for all subgrids -/
def satisfiesCondition (config : RedConfig) : Prop :=
  ∀ k : Fin 6, ∀ x y : Fin 7, 
    ∃ i j : Fin 7, onBoundary i j x y k ∧ config i j = true

/-- Counts the number of red points in a configuration -/
def countRedPoints (config : RedConfig) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 7)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 7)) fun j =>
      if config i j then 1 else 0)

/-- The main theorem: 16 is the minimum number of red points required -/
theorem min_red_points_is_16 :
  (∃ config : RedConfig, satisfiesCondition config ∧ countRedPoints config = 16) ∧
  (∀ config : RedConfig, satisfiesCondition config → countRedPoints config ≥ 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_points_is_16_l605_60506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_sequence_lambda_bound_l605_60514

/-- A sequence {a_n} is defined as a_n = -2n^2 + ln for n ∈ ℕ*, where l is a real number.
    If the sequence is decreasing, then l < 6. -/
theorem decreasing_sequence_lambda_bound (l : ℝ) :
  (∀ n : ℕ+, (n : ℝ) ≤ (n + 1 : ℝ) → 
    -2 * (n : ℝ)^2 + l * (n : ℝ) ≥ -2 * (n + 1 : ℝ)^2 + l * (n + 1 : ℝ)) →
  l < 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_sequence_lambda_bound_l605_60514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_condition_l605_60526

theorem perfect_square_sum_condition (n : ℕ) :
  (∀ x y z : ℤ, x + y + z = 0 → ∃ k : ℤ, (x^n + y^n + z^n) / 2 = k^2) ↔ n = 1 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_condition_l605_60526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_symmetry_l605_60519

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) - x) + 1

-- State the theorem
theorem f_sum_symmetry (x : ℝ) : f x + f (-x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_symmetry_l605_60519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_powers_is_n_minus_one_l605_60569

/-- Given a set of n distinct positive integers where n ≥ 2, 
    this function returns the maximum number of distinct powers of 2 
    that could be written as the largest power of 2 dividing the 
    difference between any two of these integers. -/
def max_distinct_powers (n : ℕ) (integers : Finset ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of distinct powers of 2 
    is n - 1 for n ≥ 2 distinct positive integers. -/
theorem max_distinct_powers_is_n_minus_one 
  (n : ℕ) (integers : Finset ℕ) 
  (h1 : n ≥ 2) 
  (h2 : integers.card = n) 
  (h3 : ∀ i j : ℕ, i ∈ integers → j ∈ integers → i ≠ j → i ≠ j) : 
  max_distinct_powers n integers = n - 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_powers_is_n_minus_one_l605_60569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_120_equals_five_sixths_l605_60563

def b : ℕ → ℚ
  | 0 => 2  -- Adding this case for Nat.zero
  | 1 => 2
  | 2 => 1/3
  | n+3 => (2 - 3 * b (n+2)) / (3 * b (n+1))

theorem b_120_equals_five_sixths : b 120 = 5/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_120_equals_five_sixths_l605_60563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_and_symmetric_l605_60542

noncomputable def f (x : ℝ) := Real.cos x + abs (Real.cos x)

theorem f_periodic_and_symmetric :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x) ∧
  (∃ (S : Set ℝ), Set.Infinite S ∧ ∀ (a : ℝ), a ∈ S → ∀ (x : ℝ), f (a + x) = f (a - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_and_symmetric_l605_60542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_from_cross_section_l605_60558

/-- Given a sphere with a cross-section of area 2π cm² located 1 cm from its center,
    prove that the volume of the sphere is 4√3π cm³. -/
theorem sphere_volume_from_cross_section (r : ℝ) : 
  (2 * Real.pi = Real.pi * r^2) →   -- Cross-sectional area is 2π cm²
  (1^2 + r^2 = (Real.sqrt 3)^2) →   -- Distance from center to cross-section is 1 cm
  ((4/3) * Real.pi * (Real.sqrt 3)^3 = 4 * Real.sqrt 3 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_from_cross_section_l605_60558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l605_60533

/-- The area of a triangle with side lengths 18, 80, and 82 is 720 -/
def triangle_area (a b c area : ℝ) : Prop :=
  a = 18 ∧ b = 80 ∧ c = 82 ∧
  a * a + b * b = c * c ∧
  area = (1 / 2) * a * b ∧
  area = 720

-- Proof
theorem triangle_area_proof : triangle_area 18 80 82 720 := by
  unfold triangle_area
  apply And.intro
  · norm_num
  apply And.intro
  · norm_num
  apply And.intro
  · norm_num
  apply And.intro
  · norm_num
  apply And.intro
  · norm_num
  · norm_num

#check triangle_area_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l605_60533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l605_60554

/-- The function that is directly proportional to x and passes through (4, 3) -/
noncomputable def f (x : ℝ) : ℝ := (3/4) * x

/-- The linear function passing through (-2, 1) and (4, -3) -/
noncomputable def g (x : ℝ) : ℝ := -(2/3) * x - (1/3)

theorem problem_solution :
  (∀ x y : ℝ, f y = f x → y = x) ∧  -- f is directly proportional
  f 4 = 3 ∧                         -- f passes through (4, 3)
  g (-2) = 1 ∧                      -- g passes through (-2, 1)
  g 4 = -3                          -- g passes through (4, -3)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l605_60554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_is_10pi_div_3_l605_60524

noncomputable section

/-- The length of a parametric curve -/
def curve_length (x y : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)

/-- The parametric equations of the curve -/
def x (θ : ℝ) : ℝ := 5 * Real.cos θ
def y (θ : ℝ) : ℝ := 5 * Real.sin θ

theorem curve_length_is_10pi_div_3 :
  curve_length x y (π/3) π = (10*π)/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_is_10pi_div_3_l605_60524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_marbles_even_and_positive_l605_60591

/-- Represents the state of marbles in the jar -/
structure JarState where
  white : Nat
  black : Nat

/-- Represents a single operation on the jar -/
def operation (state : JarState) : JarState :=
  { white := if state.white % 2 = 1 then state.white - 2 else state.white,
    black := state.black }

/-- The theorem stating that after any number of operations, 
    the number of white marbles will always be even and greater than 0 -/
theorem white_marbles_even_and_positive (n : Nat) : 
  ∀ (final : JarState), 
  (Nat.iterate operation n { white := 100, black := 100 } = final) → 
  (final.white % 2 = 0 ∧ final.white > 0) := by
  sorry

#check white_marbles_even_and_positive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_marbles_even_and_positive_l605_60591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_l605_60590

/-- Parabola definition -/
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Theorem statement -/
theorem parabola_distance (A : ℝ × ℝ) 
  (hA : Parabola A.1 A.2) 
  (hMid : (A.1 + Focus.1) / 2 = 2) : 
  ‖A - Focus‖ = 4 := by 
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_l605_60590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l605_60572

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos x) ^ 2 + (1 / 2) * Real.sin (2 * x) - Real.sqrt 3 / 2

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧
    (∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q)) ∧
  (∀ x : ℝ, f x ≤ 1) ∧ (∃ x : ℝ, f x = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l605_60572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_cos_f_at_specific_angle_f_in_second_quadrant_l605_60585

noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi - α) * Real.cos (2*Real.pi - α) * Real.cos (3*Real.pi/2 + α)) /
  (Real.cos (Real.pi/2 + α) * Real.sin (Real.pi + α))

theorem f_equals_cos (α : Real) : f α = Real.cos α := by sorry

theorem f_at_specific_angle : f (-13*Real.pi/3) = 1/2 := by sorry

theorem f_in_second_quadrant (α : Real)
  (h1 : Real.pi/2 < α ∧ α < Real.pi)
  (h2 : Real.cos (α - Real.pi/2) = 3/5) :
  f α = -4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_cos_f_at_specific_angle_f_in_second_quadrant_l605_60585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_200_reduction_max_profit_reduction_l605_60543

/-- Represents the watermelon vendor's business model -/
structure WatermelonVendor where
  buyPrice : ℚ
  initialSellPrice : ℚ
  initialDailySale : ℚ
  additionalSalePerReduction : ℚ
  reductionStep : ℚ
  dailyFixedCosts : ℚ

/-- Calculates the daily profit for a given price reduction -/
def dailyProfit (v : WatermelonVendor) (priceReduction : ℚ) : ℚ :=
  let newSellPrice := v.initialSellPrice - priceReduction
  let newSaleAmount := v.initialDailySale + (priceReduction / v.reductionStep) * v.additionalSalePerReduction
  (newSellPrice - v.buyPrice) * newSaleAmount - v.dailyFixedCosts

/-- The vendor's business parameters -/
def vendor : WatermelonVendor where
  buyPrice := 2
  initialSellPrice := 3
  initialDailySale := 200
  additionalSalePerReduction := 40
  reductionStep := 1/10
  dailyFixedCosts := 24

/-- Theorem: The price reduction to achieve a daily profit of 200 yuan is 0.3 yuan/kg -/
theorem profit_200_reduction : 
  ∃ x : ℚ, x = 3/10 ∧ dailyProfit vendor x = 200 := by
  sorry

/-- Theorem: The price reduction to maximize daily profit is 0.25 yuan/kg -/
theorem max_profit_reduction :
  ∃ x : ℚ, x = 1/4 ∧ ∀ y : ℚ, dailyProfit vendor y ≤ dailyProfit vendor x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_200_reduction_max_profit_reduction_l605_60543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l605_60551

open Real

variable (a b c A B C : ℝ)

theorem triangle_properties 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_sin : a / sin A = b / sin B)
  (h_condition : c = sqrt 3 * a * sin C - c * cos A) :
  A = π / 3 ∧ 
  (cos (2 * A) / a^2 - cos (2 * B) / b^2 = 1 / a^2 - 1 / b^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l605_60551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_percent_l605_60504

/-- Calculate the profit percent from a car sale --/
theorem car_sale_profit_percent 
  (purchase_price repair_cost selling_price : ℚ) 
  (h1 : purchase_price = 45000)
  (h2 : repair_cost = 12000)
  (h3 : selling_price = 80000) :
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = 40.35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_percent_l605_60504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_positive_l605_60560

/-- The function f(x) = (e^x - e^(-x)) / 2 -/
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

/-- Theorem: Given x₁, x₂, x₃ ∈ ℝ satisfying the conditions,
    the sum f(x₁) + f(x₂) + f(x₃) is always greater than zero -/
theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
    (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
    f x₁ + f x₂ + f x₃ > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_positive_l605_60560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_iff_m_gt_4_p_xor_q_iff_m_in_range_l605_60535

/-- Proposition p: the equation x^2 - mx + m = 0 has two distinct real roots in (1, +∞) -/
def p (m : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ x > 1 ∧ y > 1 ∧ x^2 - m*x + m = 0 ∧ y^2 - m*y + m = 0

/-- Function f(x) = 1 / (4x^2 + mx + m) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1 / (4*x^2 + m*x + m)

/-- Proposition q: the function f(x) is defined for all real numbers -/
def q (m : ℝ) : Prop :=
  ∀ x, 4*x^2 + m*x + m ≠ 0

/-- Part 1: p is true if and only if m ∈ (4, +∞) -/
theorem p_iff_m_gt_4 (m : ℝ) : p m ↔ m > 4 := by sorry

/-- Part 2: Exactly one of p and q is true if and only if m ∈ (0, 4] ∪ [16, +∞) -/
theorem p_xor_q_iff_m_in_range (m : ℝ) : (p m ∧ ¬q m) ∨ (¬p m ∧ q m) ↔ (0 < m ∧ m ≤ 4) ∨ m ≥ 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_iff_m_gt_4_p_xor_q_iff_m_in_range_l605_60535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statements_l605_60509

theorem problem_statements :
  (∀ a : ℝ, a > 0 → a * (6 - a) ≤ 9) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a * b = a + b + 3 → a * b ≥ 9) ∧
  (∀ a : ℝ, a > 0 → a^2 + 4 / (a^2 + 3) > 1) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 1/a + 2/b ≥ 3/2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statements_l605_60509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l605_60520

/-- Calculates the length of a platform given train speed, train length, and time to cross -/
theorem platform_length_calculation (train_speed_kmph : ℝ) (train_length_m : ℝ) (time_to_cross_s : ℝ) :
  train_speed_kmph = 72 →
  train_length_m = 350.048 →
  time_to_cross_s = 30 →
  249.952 = (train_speed_kmph * (1000 / 3600) * time_to_cross_s) - train_length_m := by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l605_60520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_properties_l605_60501

noncomputable def cube_edge_length : ℝ := 2 * Real.sqrt 2

theorem cube_properties :
  let volume := cube_edge_length ^ 3
  let surface_area := 6 * cube_edge_length ^ 2
  let inscribed_sphere_surface_area := 4 * Real.pi * (cube_edge_length / 2) ^ 2
  (volume = 16 * Real.sqrt 2) ∧
  (surface_area = 48) ∧
  (inscribed_sphere_surface_area = 8 * Real.pi) := by
  sorry

#check cube_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_properties_l605_60501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l605_60589

-- Define the speeds of the trains in km/hr
noncomputable def speed_train1 : ℝ := 45
noncomputable def speed_train2 : ℝ := 30

-- Define the time taken for the slower train to pass the driver of the faster one in seconds
noncomputable def passing_time : ℝ := 47.99616030717543

-- Define the conversion factor from km/hr to m/s
noncomputable def km_per_hr_to_m_per_s : ℝ := 5 / 18

-- Theorem statement
theorem train_length_proof :
  let relative_speed : ℝ := (speed_train1 + speed_train2) * km_per_hr_to_m_per_s
  let total_distance : ℝ := relative_speed * passing_time
  let train_length : ℝ := total_distance / 2
  train_length = 500 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l605_60589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l605_60510

-- Define the slopes of the two lines
noncomputable def slope1 (a : ℝ) : ℝ := -a / (1 - a)
noncomputable def slope2 (a : ℝ) : ℝ := -(a - 1) / (2*a + 3)

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop := 
  (a ≠ 1 ∧ a ≠ -3/2) → (slope1 a * slope2 a = -1)

-- Define the theorem
theorem perpendicular_lines (a : ℝ) : 
  perpendicular a ↔ (a = 1 ∨ a = -3) :=
sorry

-- Example usage (optional)
example : perpendicular 1 ∨ perpendicular (-3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l605_60510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_four_l605_60552

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + a * x^2) - 2 * x)

theorem odd_function_implies_a_equals_four (a : ℝ) :
  (∀ x, f a x = -f a (-x)) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_four_l605_60552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_proof_l605_60571

/-- The minimum distance between two circles, one centered at (2,0) with radius 2
    and another centered at (0,0) with radius 1 -/
def min_distance_between_circles : ℝ := 1

/-- Rational Woman's circle -/
def rational_woman_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - 2) ^ 2 + p.2 ^ 2 = 4

/-- Rational Man's circle -/
def rational_man_circle (p : ℝ × ℝ) : Prop :=
  p.1 ^ 2 + p.2 ^ 2 = 1

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem min_distance_proof :
  ∀ (p q : ℝ × ℝ), rational_woman_circle p → rational_man_circle q →
  distance p q ≥ min_distance_between_circles := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_proof_l605_60571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadable_cells_even_shadable_cells_odd_no_symmetry_lines_l605_60503

/-- The number of cells that can be shaded in an n × n grid to result in no lines of symmetry -/
def shadable_cells (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 - 2*n
  else
    n^2 - 4*n + 3

/-- Theorem stating the number of shadable cells for even n -/
theorem shadable_cells_even (n : ℕ) (h : n % 2 = 0) :
  shadable_cells n = n^2 - 2*n :=
by sorry

/-- Theorem stating the number of shadable cells for odd n -/
theorem shadable_cells_odd (n : ℕ) (h : n % 2 = 1) :
  shadable_cells n = n^2 - 4*n + 3 :=
by sorry

/-- Represents a grid with a shaded cell -/
def Grid (n : ℕ) := Fin n → Fin n → Bool

/-- Shades a cell in the grid -/
def shade_grid (n : ℕ) (cell : Fin n × Fin n) : Grid n :=
  λ i j => i = cell.1 ∧ j = cell.2

/-- Represents a line of symmetry -/
inductive SymmetryLine
| Horizontal : Fin n → SymmetryLine
| Vertical : Fin n → SymmetryLine
| Diagonal : Bool → SymmetryLine

/-- Checks if a given line is a symmetry line for the grid -/
def is_symmetry_line (n : ℕ) (line : SymmetryLine) (grid : Grid n) : Prop :=
  sorry  -- Definition of symmetry check goes here

/-- Theorem stating that the resulting grid has no lines of symmetry -/
theorem no_symmetry_lines (n : ℕ) (cell : Fin n × Fin n) 
  (h : cell.1.val < shadable_cells n) :
  ∀ line, ¬ is_symmetry_line n line (shade_grid n cell) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadable_cells_even_shadable_cells_odd_no_symmetry_lines_l605_60503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_symmetry_product_l605_60588

theorem complex_symmetry_product (z₁ z₂ : ℂ) :
  z₁ = 2 + I →
  z₂.re = -z₁.re →
  z₂.im = z₁.im →
  z₁ * z₂ = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_symmetry_product_l605_60588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_dot_product_l605_60541

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a vector in 2D space -/
structure Vec where
  x : ℝ
  y : ℝ

/-- The ellipse C: x^2/4 + y^2/3 = 1 -/
def ellipse (p : Point) : Prop :=
  p.x^2 / 4 + p.y^2 / 3 = 1

/-- The right focus of the ellipse -/
def F : Point :=
  ⟨1, 0⟩

/-- The left vertex of the ellipse -/
def P : Point :=
  ⟨-2, 0⟩

/-- A line passing through two points -/
def line (p q : Point) : Prop :=
  (q.y - p.y) * (q.x - p.x) = (q.x - p.x) * (q.y - p.y)

/-- The dot product of two vectors -/
def dot_product (v w : Vec) : ℝ :=
  v.x * w.x + v.y * w.y

/-- The main theorem -/
theorem ellipse_intersection_dot_product :
  ∀ (A B D E : Point),
    ellipse A →
    ellipse B →
    line F A →
    line F B →
    line P D →
    line P E →
    D.x = 4 →
    E.x = 4 →
    line A D →
    line B E →
    dot_product ⟨D.x - F.x, D.y - F.y⟩ ⟨E.x - F.x, E.y - F.y⟩ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_dot_product_l605_60541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l605_60583

/-- The function f(x) = 2 sin x + sin 2x -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sin (2 * x)

/-- The minimum value of f(x) is -3√3/2 -/
theorem f_min_value : 
  ∃ (min : ℝ), min = -3 * Real.sqrt 3 / 2 ∧ ∀ (x : ℝ), f x ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l605_60583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheesy_fries_cost_is_four_l605_60549

/-- The cost of a Taco Grande Plate -/
def taco_grande_cost : ℝ := sorry

/-- The cost of the cheesy fries -/
def cheesy_fries_cost : ℝ := sorry

/-- The cost of Mike's side salad -/
def side_salad_cost : ℝ := 2

/-- The cost of Mike's diet cola -/
def diet_cola_cost : ℝ := 2

/-- Mike's total bill -/
def mike_bill : ℝ := taco_grande_cost + side_salad_cost + cheesy_fries_cost + diet_cola_cost

/-- John's total bill -/
def john_bill : ℝ := taco_grande_cost

theorem cheesy_fries_cost_is_four :
  mike_bill = 2 * john_bill ∧ 
  mike_bill + john_bill = 24 → 
  cheesy_fries_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheesy_fries_cost_is_four_l605_60549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_area_l605_60559

theorem tetrahedron_sphere_area (a : ℝ) (h : a > 0) :
  let section_area := 3 * Real.sqrt 2
  let edge_length := 2 * Real.sqrt 3
  let sphere_radius := (3 * Real.sqrt 2) / 2
  let sphere_surface_area := 4 * Real.pi * sphere_radius^2
  (a = edge_length ∧ section_area = (1/2) * a * ((Real.sqrt 2)/2) * a) →
  sphere_surface_area = 18 * Real.pi :=
by
  intros
  sorry

#check tetrahedron_sphere_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_area_l605_60559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l605_60545

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

-- Define the set A
def A : Set ℝ := {x | -2 < f x ∧ f x < 0}

-- Theorem statement
theorem problem_solution :
  (A = Set.Ioo (-1/2) (1/2)) ∧
  (∀ (m n : ℝ), m ∈ A → n ∈ A → |1 - 4*m*n| > 2 * |m - n|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l605_60545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l605_60578

/-- The coefficient of x^2 in the expansion of (1-x)^6(1+x)^4 is -3 -/
theorem coefficient_x_squared_in_expansion : 
  (Polynomial.coeff ((1 - Polynomial.X) ^ 6 * (1 + Polynomial.X) ^ 4) 2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l605_60578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l605_60540

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) : ℝ := 4 * x - 3
noncomputable def g (x : ℝ) : ℝ := 3 * x + 2
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of h
noncomputable def h_inv (x : ℝ) : ℝ := (x - 5) / 12

-- Theorem statement
theorem h_inverse_correct : ∀ x, h (h_inv x) = x ∧ h_inv (h x) = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l605_60540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_triangle_area_formula_l605_60582

/-- The minimal area of a right triangle with hypotenuse passing through (x, y) and legs on axes --/
noncomputable def minimalTriangleArea (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- Theorem stating the minimal area of the described right triangle --/
theorem minimal_triangle_area_formula {x y : ℝ} (hx : x > 0) (hy : y > 0) :
  ∀ (a b : ℝ), a > 0 → b > 0 → x / a + y / b = 1 →
  minimalTriangleArea x y ≤ (1/2) * a * b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_triangle_area_formula_l605_60582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_circle_radius_l605_60528

/-- Given two externally tangent circles and a third circle tangent to both and their common external tangent, prove the radius of the third circle -/
theorem third_circle_radius 
  (A B D : ℝ × ℝ) -- Centers of the circles
  (r₁ r₂ r₃ : ℝ) -- Radii of the circles
  (h₁ : r₁ = 2) -- Radius of first circle is 2
  (h₂ : r₂ = 3) -- Radius of second circle is 3
  (h₃ : ‖A - B‖ = r₁ + r₂) -- Circles are externally tangent
  (h₄ : ‖A - D‖ = r₁ + r₃) -- Third circle is tangent to first circle
  (h₅ : ‖B - D‖ = r₂ + r₃) -- Third circle is tangent to second circle
  (h₆ : ∃ P : ℝ × ℝ, ‖P - A‖ = r₁ ∧ ‖P - B‖ = r₂ ∧ ‖P - D‖ = r₃) -- Third circle is tangent to common external tangent
  : r₃ = 1/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_circle_radius_l605_60528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l605_60555

/-- The function f(x) = -x³ + ax² - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

/-- The second derivative of f(x) -/
def f'' (a : ℝ) (x : ℝ) : ℝ := -6*x + 2*a

theorem max_value_f''_plus_f (a : ℝ) :
  f' a 2 = 0 →
  ∃ (m n : ℝ), m ∈ Set.Icc 0 1 ∧ n ∈ Set.Icc 0 1 ∧
  (∀ (m' n' : ℝ), m' ∈ Set.Icc 0 1 → n' ∈ Set.Icc 0 1 →
  f'' a n' + f a m' ≤ f'' a n + f a m) ∧
  f'' a n + f a m = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_l605_60555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_C_l605_60550

/-- Represents a station on the transit line -/
inductive Station : Type
| A : Station
| B : Station
| C : Station
| D : Station

/-- Represents the distance between two stations -/
noncomputable def distance : Station → Station → ℝ
| Station.A, Station.B => 3
| Station.B, Station.C => 2
| Station.C, Station.D => 5
| Station.A, Station.D => 6
| Station.B, Station.A => 3
| Station.C, Station.B => 2
| Station.D, Station.C => 5
| Station.D, Station.A => 6
| _, _ => 0  -- Default case for other combinations

/-- The transit line is straight, implying additivity of distances -/
axiom additivity (s1 s2 s3 : Station) : 
  distance s1 s2 + distance s2 s3 = distance s1 s3

theorem distance_A_to_C : distance Station.A Station.C = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_C_l605_60550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinate_sum_l605_60584

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, 3, -4) and (-2, 7, 6) is 10. -/
theorem midpoint_coordinate_sum : 
  let p₁ : ℝ × ℝ × ℝ := (10, 3, -4)
  let p₂ : ℝ × ℝ × ℝ := (-2, 7, 6)
  let midpoint := (
    (p₁.fst + p₂.fst) / 2, 
    (p₁.snd.fst + p₂.snd.fst) / 2, 
    (p₁.snd.snd + p₂.snd.snd) / 2
  )
  midpoint.fst + midpoint.snd.fst + midpoint.snd.snd = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinate_sum_l605_60584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_theorem_l605_60581

theorem class_average_theorem (N : ℝ) (h : N > 0) :
  let total_marks := 80 * N
  let high_scorers_marks := 0.1 * N * 95 + 0.2 * N * 90
  let remaining_students := 0.7 * N
  total_marks = high_scorers_marks + remaining_students * 75 := by
  -- Unfold the let bindings
  simp_all
  -- Algebraic manipulation
  ring
  -- The proof is complete
  done

#check class_average_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_theorem_l605_60581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l605_60544

-- Define the function y = 3^x
noncomputable def y (x : ℝ) : ℝ := Real.exp (x * Real.log 3)

-- Define f as the inverse function of y
noncomputable def f : ℝ → ℝ := Function.invFun y

-- Theorem to prove
theorem inverse_function_value : f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l605_60544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_partition_exists_l605_60515

/-- Represents a cube with integer edge length -/
structure Cube where
  edge : ℕ

/-- Represents a partition of a larger cube into smaller cubes -/
structure CubePartition where
  largeCube : Cube
  smallerCubes : List Cube

def isValidPartition (p : CubePartition) : Prop :=
  p.largeCube.edge = 4 ∧
  p.smallerCubes.length = 31 ∧
  (∃ c1 c2, c1 ∈ p.smallerCubes ∧ c2 ∈ p.smallerCubes ∧ c1.edge ≠ c2.edge) ∧
  (∀ c ∈ p.smallerCubes, c.edge > 0 ∧ c.edge ≤ 4) ∧
  (p.smallerCubes.map (fun c => c.edge ^ 3)).sum = p.largeCube.edge ^ 3

theorem cube_partition_exists : ∃ p : CubePartition, isValidPartition p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_partition_exists_l605_60515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l605_60556

noncomputable section

-- Define the triangle ABC
def A : ℝ × ℝ := (-1, 5)
def B : ℝ × ℝ := (-2, -1)
def C : ℝ × ℝ := (4, 3)

-- Define M as the midpoint of BC
def M : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Theorem for the equation of line AB and length of median AM
theorem triangle_abc_properties :
  (∀ x y : ℝ, 6 * x - y + 11 = 0 ↔ (y - A.2) / (x - A.1) = (B.2 - A.2) / (B.1 - A.1)) ∧
  Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l605_60556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l605_60518

/-- The function f(x) = ln x + b/(x+1) -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.log x + b / (x + 1)

theorem function_inequality (b : ℝ) :
  b > 0 →
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → x₁ ≠ x₂ →
    (f b x₁ - f b x₂) / (x₁ - x₂) < -1) →
  b > 27/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l605_60518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_roots_iff_k_in_range_l605_60599

open Real

/-- The function f(x) for a given k -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (log x) / x - k * x

/-- Theorem stating the condition for f to have exactly two roots in [1/e, e²] -/
theorem f_two_roots_iff_k_in_range :
  ∀ k : ℝ, (∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ r₁ ∈ Set.Icc (1/ℯ) (ℯ^2) ∧ r₂ ∈ Set.Icc (1/ℯ) (ℯ^2) ∧ f k r₁ = 0 ∧ f k r₂ = 0) ↔
  k ∈ Set.Icc (2/ℯ^4) (1/(2*ℯ)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_roots_iff_k_in_range_l605_60599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_number_l605_60562

-- Define the expressions as noncomputable
noncomputable def expr1 : ℝ := 5 - 2 * Real.sqrt 8
noncomputable def expr2 : ℝ := 2 * Real.sqrt 8 - 5
noncomputable def expr3 : ℝ := 12 - 3 * Real.sqrt 9
noncomputable def expr4 : ℝ := 27 - 5 * Real.sqrt 18
noncomputable def expr5 : ℝ := 5 * Real.sqrt 18 - 27

-- Define a list of all expressions
noncomputable def expressions : List ℝ := [expr1, expr2, expr3, expr4, expr5]

-- Theorem statement
theorem smallest_positive_number :
  expr2 > 0 ∧ ∀ x ∈ expressions, x > 0 → expr2 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_number_l605_60562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_B_radius_l605_60557

noncomputable section

-- Define the circles and their properties
def circle_A : ℝ := 2  -- radius of circle A
def circle_D : ℝ := 3  -- radius of circle D

-- Define the function to calculate the radius of circle B
noncomputable def radius_B : ℝ := (5 + Real.sqrt 17) / 2

-- Theorem statement
theorem circle_B_radius :
  -- Given conditions
  ∀ (circle_B : ℝ),
  -- Circle B is externally tangent to A and internally tangent to D
  circle_B > 0 ∧
  circle_B < circle_A ∧
  circle_B < circle_D ∧
  -- Pythagorean theorem application (from step 2 in the solution)
  (circle_D - circle_B)^2 + (circle_A - circle_B)^2 = circle_D^2 →
  -- Conclusion
  circle_B = radius_B := by
  sorry  -- Proof omitted

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_B_radius_l605_60557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_cost_l605_60576

/-- The cost of one pair of jeans -/
def j : ℚ := sorry

/-- The cost of one shirt -/
def s : ℚ := sorry

/-- The cost of one jacket -/
def k : ℚ := sorry

/-- Given condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom h1 : 3 * j + 2 * s = 69

/-- Given condition: 2 pairs of jeans and 3 shirts cost $71 -/
axiom h2 : 2 * j + 3 * s = 71

/-- Given condition: 3 pairs of jeans, 2 shirts, and 1 jacket cost $90 -/
axiom h3 : 3 * j + 2 * s + k = 90

/-- Theorem: The cost of one shirt is $15 -/
theorem shirt_cost : s = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_cost_l605_60576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_ratio_l605_60502

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A hexagon defined by six points -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- Check if a hexagon is regular -/
def isRegularHexagon (h : Hexagon) : Prop := sorry

/-- Calculate the area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- Check if a point is on a side of the hexagon -/
def isOnSide (p : Point) (h : Hexagon) : Prop := sorry

/-- Check if lines are parallel and equally spaced -/
def areParallelAndEquallySpaced (l1 l2 l3 l4 : Point → Point → Prop) : Prop := sorry

/-- Check if lines divide the height of the hexagon into four equal parts -/
def dividesHeightIntoFourParts (h : Hexagon) (l1 l2 l3 l4 : Point → Point → Prop) : Prop := sorry

theorem hexagon_area_ratio 
  (ABCDEF : Hexagon)
  (M N O P Q R : Point)
  (h_regular : isRegularHexagon ABCDEF)
  (h_M : isOnSide M ABCDEF)
  (h_N : isOnSide N ABCDEF)
  (h_O : isOnSide O ABCDEF)
  (h_P : isOnSide P ABCDEF)
  (h_Q : isOnSide Q ABCDEF)
  (h_R : isOnSide R ABCDEF)
  (h_parallel : areParallelAndEquallySpaced 
    (λ p1 p2 => p1 = ABCDEF.A ∧ p2 = M) 
    (λ p1 p2 => p1 = N ∧ p2 = P)
    (λ p1 p2 => p1 = O ∧ p2 = Q)
    (λ p1 p2 => p1 = ABCDEF.F ∧ p2 = R))
  (h_divide : dividesHeightIntoFourParts ABCDEF 
    (λ p1 p2 => p1 = ABCDEF.A ∧ p2 = M) 
    (λ p1 p2 => p1 = N ∧ p2 = P)
    (λ p1 p2 => p1 = O ∧ p2 = Q)
    (λ p1 p2 => p1 = ABCDEF.F ∧ p2 = R)) :
  hexagonArea ⟨M, N, P, Q, R, ABCDEF.F⟩ / hexagonArea ABCDEF = (7 - 4 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_ratio_l605_60502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_saved_is_one_hour_time_saved_in_minutes_l605_60546

/-- The distance between city A and B in miles -/
noncomputable def distance : ℝ := 180

/-- The time taken for the first trip in hours -/
noncomputable def time1 : ℝ := 3

/-- The time taken for the second trip in hours -/
noncomputable def time2 : ℝ := 2.5

/-- The speed of the round trip if time was saved, in miles per hour -/
noncomputable def speedWithTimeSaved : ℝ := 80

/-- The time saved in both trips, in hours -/
noncomputable def timeSaved : ℝ := (time1 + time2) - (2 * distance / speedWithTimeSaved)

theorem time_saved_is_one_hour : timeSaved = 1 := by sorry

theorem time_saved_in_minutes : timeSaved * 60 = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_saved_is_one_hour_time_saved_in_minutes_l605_60546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_one_l605_60586

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x < 3 then 5 * x + 10 else 3 * x - 9

-- Theorem statement
theorem sum_of_roots_is_one : 
  ∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_one_l605_60586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_10000_l605_60531

/-- Calculates simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest (payable half-yearly) -/
noncomputable def compoundInterestHalfYearly (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 200) ^ (2 * time) - 1)

/-- Theorem: Given the conditions, the principal sum is 10000 -/
theorem principal_is_10000 (principal : ℝ) 
    (h1 : compoundInterestHalfYearly principal 10 1 - simpleInterest principal 10 1 = 25) :
    principal = 10000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_10000_l605_60531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_to_square_l605_60538

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smallBase : ℝ
  /-- The length of the larger base -/
  largeBase : ℝ
  /-- The angle at the larger base -/
  angle : ℝ
  /-- The larger base is three times the smaller base -/
  baseRatio : largeBase = 3 * smallBase
  /-- The angle at the larger base is 45 degrees -/
  angleValue : angle = π / 4

/-- A square -/
structure Square where
  /-- The side length of the square -/
  side : ℝ

/-- A function that represents cutting the trapezoid into three parts -/
def cutTrapezoid (t : IsoscelesTrapezoid) : List ℝ := sorry

/-- A function that represents forming a square from three parts -/
def formSquare (parts : List ℝ) : Square := sorry

/-- Theorem stating that the trapezoid can be cut into three parts to form a square -/
theorem trapezoid_to_square (t : IsoscelesTrapezoid) :
  ∃ (s : Square), formSquare (cutTrapezoid t) = s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_to_square_l605_60538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_fifteen_fortyfive_degrees_l605_60568

theorem sin_cos_sum_fifteen_fortyfive_degrees :
  (Real.sin (15 * π / 180))^2 + (Real.cos (45 * π / 180))^2 + 
  Real.sin (15 * π / 180) * Real.cos (45 * π / 180) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_fifteen_fortyfive_degrees_l605_60568
