import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_range_compare_functions_exponential_sum_inequality_l392_39268

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m * Real.log (x + 1)

-- Statement 1
theorem monotonic_range (m : ℝ) :
  (∀ x y, x < y → f m x < f m y) ↔ m ≥ (1/2 : ℝ) := by sorry

-- Statement 2
theorem compare_functions (x : ℝ) (h : x > 0) :
  f (-1) x < x^3 := by sorry

-- Statement 3
theorem exponential_sum_inequality (n : ℕ) (h : n > 0) :
  (Finset.range (n+1)).sum (λ i => Real.exp ((1 - ↑i) * ↑i^2)) < n * (n + 3) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_range_compare_functions_exponential_sum_inequality_l392_39268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_batch_cost_is_16_l392_39283

/-- Represents the production of two batches of handicrafts --/
structure HandicraftProduction where
  first_batch_cost : ℚ
  second_batch_cost : ℚ
  quantity_ratio : ℚ
  cost_increase : ℚ

/-- Calculates the cost per item in the second batch of handicrafts --/
noncomputable def second_batch_item_cost (prod : HandicraftProduction) : ℚ :=
  let first_batch_item_cost := prod.first_batch_cost / (prod.second_batch_cost / (prod.quantity_ratio * (prod.first_batch_cost / (prod.second_batch_cost / prod.quantity_ratio - prod.cost_increase))))
  first_batch_item_cost + prod.cost_increase

/-- Theorem stating that the cost of each item in the second batch is 16 --/
theorem second_batch_cost_is_16 (prod : HandicraftProduction)
  (h1 : prod.first_batch_cost = 3000)
  (h2 : prod.second_batch_cost = 9600)
  (h3 : prod.quantity_ratio = 3)
  (h4 : prod.cost_increase = 1) :
  second_batch_item_cost prod = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_batch_cost_is_16_l392_39283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_m_values_eq_eight_l392_39212

/-- The number of different integer values of m for which the quadratic equation
    x^2 - mx + 40 = 0 has two integer roots -/
def count_m_values : ℕ :=
  let possible_m : List ℤ := [-41, -22, -14, -13, 13, 14, 22, 41]
  possible_m.length

theorem count_m_values_eq_eight : count_m_values = 8 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_m_values_eq_eight_l392_39212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turtleneck_profit_percentage_l392_39237

/-- Proves that the profit percentage on turtleneck sweaters sold in February is 35% of the cost price, given specific markups and discounts. -/
theorem turtleneck_profit_percentage (C : ℝ) : 
  (C * (1 + 0.20) * (1 + 0.25) * (1 - 0.10) - C) / C = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turtleneck_profit_percentage_l392_39237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_value_l392_39256

theorem complex_expression_value : 
  Complex.I^3 * (1 + Complex.I)^2 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_value_l392_39256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_points_l392_39260

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem different_points (r₁ r₂ θ₁ θ₂ : ℝ) 
  (h₁ : r₁ = -5) (h₂ : θ₁ = π/3) 
  (h₃ : r₂ = 5) (h₄ : θ₂ = -π/3) : 
  polar_to_cartesian r₁ θ₁ ≠ polar_to_cartesian r₂ θ₂ := by
  sorry

#check different_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_points_l392_39260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l392_39201

-- Define the functions
noncomputable def f1 (x : ℝ) := |x|
noncomputable def g1 (x : ℝ) := Real.sqrt (x^2)

def f2 (x : ℝ) := x^2 - 2*x - 1
def g2 (t : ℝ) := t^2 - 2*t - 1

-- Theorem statement
theorem function_equality :
  (∀ x : ℝ, f1 x = g1 x) ∧ 
  (∀ x : ℝ, f2 x = g2 x) := by
  constructor
  · intro x
    sorry -- Proof for f1 = g1
  · intro x
    sorry -- Proof for f2 = g2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l392_39201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_weight_arrangement_l392_39213

theorem impossible_weight_arrangement :
  ¬ ∃ (piles : Fin 10 → List Nat),
    let weights := List.range 100
    -- All weights are used
    (weights.sum = (List.join (List.ofFn piles)).sum) ∧
    -- Each pile has a different mass
    (∀ i j, i ≠ j → (piles i).sum ≠ (piles j).sum) ∧
    -- Heavier piles have fewer weights
    (∀ i j, (piles i).sum > (piles j).sum → (piles i).length < (piles j).length) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_weight_arrangement_l392_39213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_or_green_probability_l392_39234

/-- A cube with colored faces -/
structure ColoredCube where
  blue_faces : ℕ
  red_faces : ℕ
  green_faces : ℕ
  total_faces : ℕ
  face_sum : blue_faces + red_faces + green_faces = total_faces

/-- The probability of an event -/
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  ↑favorable_outcomes / ↑total_outcomes

/-- The probability of rolling a blue or green face on a colored cube -/
theorem blue_or_green_probability (cube : ColoredCube) 
    (h : cube.total_faces = 6) 
    (hb : cube.blue_faces = 3) 
    (hr : cube.red_faces = 2) 
    (hg : cube.green_faces = 1) : 
  probability (cube.blue_faces + cube.green_faces) cube.total_faces = 2/3 := by
  sorry

#check blue_or_green_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_or_green_probability_l392_39234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_12_max_value_achievable_l392_39297

/-- A polynomial function with real, nonnegative coefficients -/
def NonNegativePolynomial (f : ℝ → ℝ) : Prop :=
  ∃ (n : ℕ) (a : ℕ → ℝ), (∀ i, 0 ≤ a i) ∧
    ∀ x, f x = (Finset.range (n + 1)).sum (λ i ↦ a i * x ^ i)

/-- The theorem stating the maximum value of f(12) given the conditions -/
theorem max_value_at_12 (f : ℝ → ℝ) 
    (h_nonneg : NonNegativePolynomial f)
    (h_6 : f 6 = 24)
    (h_24 : f 24 = 1536) :
    f 12 ≤ 192 := by
  sorry

/-- The theorem stating that the maximum value is achievable -/
theorem max_value_achievable :
    ∃ f : ℝ → ℝ, NonNegativePolynomial f ∧ f 6 = 24 ∧ f 24 = 1536 ∧ f 12 = 192 := by
  use λ x ↦ x^3 / 9
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_12_max_value_achievable_l392_39297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l392_39275

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (2 - x))

-- Define the domain of f
def f_domain : Set ℝ := Set.Icc 0 2

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (2 * x) / (x - 1)

-- State the theorem
theorem g_domain : 
  {x : ℝ | g x ∈ Set.range g} = Set.Ico 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_l392_39275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_own_cards_eq_one_l392_39266

/-- The number of students in the school -/
def n : ℕ := 2012

/-- The probability that a student receives their own card -/
noncomputable def p : ℝ := 1 / n

/-- The indicator random variable for a student receiving their own card -/
noncomputable def I : Fin n → ℝ := fun _ => p

/-- The expected number of students who receive their own card -/
noncomputable def expected_own_cards : ℝ := (Finset.univ.sum I)

/-- Theorem: The expected number of students who receive their own card is 1 -/
theorem expected_own_cards_eq_one : expected_own_cards = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_own_cards_eq_one_l392_39266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_given_altitudes_is_obtuse_l392_39273

-- Define a triangle with given altitude lengths
structure Triangle where
  alt1 : ℝ
  alt2 : ℝ
  alt3 : ℝ

-- Define the property of being obtuse
def is_obtuse (t : Triangle) : Prop :=
  ∃ (θ : ℝ), θ > Real.pi / 2 ∧ θ < Real.pi ∧
  (∃ (a b c : ℝ), 
    a * t.alt1 = b * t.alt2 ∧
    a * t.alt1 = c * t.alt3 ∧
    b * t.alt2 = c * t.alt3 ∧
    Real.cos θ = (b^2 + c^2 - a^2) / (2 * b * c))

-- Theorem statement
theorem triangle_with_given_altitudes_is_obtuse :
  let t : Triangle := { alt1 := 1/13, alt2 := 1/10, alt3 := 1/5 }
  is_obtuse t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_given_altitudes_is_obtuse_l392_39273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_correct_l392_39242

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x - 3)) / (Real.sqrt (7 - x))

def domain_of_f : Set ℝ := { x | 3 ≤ x ∧ x < 7 }

theorem f_domain_correct :
  ∀ x : ℝ, x ∈ domain_of_f ↔ (∃ y : ℝ, f x = y) := by
  sorry

#check f_domain_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_correct_l392_39242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l392_39289

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y = -x^2 + 5*x

/-- Triangle area function -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1*y2 + x2*y3 + x3*y1 - y1*x2 - y2*x3 - y3*x1)

/-- Theorem: Maximum area of triangle ABC -/
theorem max_triangle_area :
  ∃ (p q : ℝ),
    parabola 0 1 ∧
    parabola 3 4 ∧
    parabola p q ∧
    0 ≤ p ∧ p ≤ 3 ∧
    (∀ (r s : ℝ),
      parabola r s →
      0 ≤ r →
      r ≤ 3 →
      triangleArea 0 1 3 4 p q ≥ triangleArea 0 1 3 4 r s) ∧
    triangleArea 0 1 3 4 p q = 35/9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l392_39289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_extra_chores_l392_39299

/-- Calculates the average number of extra chores per week given the total amount earned,
    fixed weekly allowance, extra earning per chore, and number of weeks. -/
noncomputable def average_extra_chores (total_amount : ℝ) (weekly_allowance : ℝ) (extra_per_chore : ℝ) (weeks : ℕ) : ℝ :=
  ((total_amount - (weekly_allowance * weeks)) / (extra_per_chore * ↑weeks))

/-- Theorem stating that given the specific conditions from the problem,
    the average number of extra chores per week is 15. -/
theorem carol_extra_chores :
  let total_amount : ℝ := 425
  let weekly_allowance : ℝ := 20
  let extra_per_chore : ℝ := 1.5
  let weeks : ℕ := 10
  average_extra_chores total_amount weekly_allowance extra_per_chore weeks = 15 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_extra_chores_l392_39299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_nine_factorial_over_168_l392_39232

-- Define factorial function
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

-- Define the problem
theorem sqrt_nine_factorial_over_168 :
  Real.sqrt (factorial 9 / 168) = 3 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_nine_factorial_over_168_l392_39232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_neg_half_neg_one_unique_solution_otimes_equation_l392_39252

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ :=
  if a ≥ b then b / (4 * a - b) else a / (4 * a + b)

-- Theorem 1: (-1/2) ⊗ (-1) = 1
theorem otimes_neg_half_neg_one : otimes (-1/2) (-1) = 1 := by sorry

-- Theorem 2: The unique solution to (x - 3) ⊗ (x + 3) = 1 is x = 3/2
theorem unique_solution_otimes_equation :
  ∃! x : ℝ, otimes (x - 3) (x + 3) = 1 ∧ x = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_neg_half_neg_one_unique_solution_otimes_equation_l392_39252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l392_39200

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : Real) : Real := 2 * Real.sin θ

-- Define the line l in parametric form
noncomputable def line_l (t : Real) : Real × Real :=
  (-3/5 * t + 2, 4/5 * t)

-- Define point M as the intersection of line l and x-axis
def point_M : Real × Real :=
  (2, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : Real × Real) : Real :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem max_distance_MN :
  ∃ (N : Real × Real), ∀ (θ : Real),
    let ρ := curve_C θ
    let x := ρ * Real.cos θ
    let y := ρ * Real.sin θ
    let N' := (x, y)
    distance point_M N' ≤ distance point_M N ∧
    distance point_M N = Real.sqrt 5 + 1 := by
  sorry

#check max_distance_MN

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l392_39200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_measure_108_l392_39214

-- Define a regular polygon with n sides
def RegularPolygon (n : ℕ) : Prop :=
  n > 2 ∧ n * (n - 3) / 2 = n

-- Define the measure of an interior angle of a regular polygon
noncomputable def InteriorAngleMeasure (n : ℕ) : ℝ :=
  (n - 2 : ℝ) * 180 / n

-- Theorem statement
theorem interior_angle_measure_108 :
  ∀ n : ℕ, RegularPolygon n → InteriorAngleMeasure n = 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_measure_108_l392_39214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C1_intersection_theorem_l392_39254

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the curve C1
def curve_C1 (ρ θ : ℝ) : Prop := ρ = -4 * Real.sqrt 3 * Real.sin θ

-- Define the line C2
def line_C2 (x y t : ℝ) : Prop := x = 2 + (Real.sqrt 3 / 2) * t ∧ y = (1 / 2) * t

-- Define the point C
def point_C : ℝ × ℝ := (2, 0)

-- Define the polar equation of line AB
def line_AB_polar (θ : ℝ) : Prop := θ = -Real.pi / 6

-- Define the ratio |CD| : |CE|
noncomputable def ratio_CD_CE : ℝ := 1 / 2

theorem circle_C1_intersection_theorem :
  ∀ (x y ρ θ t : ℝ),
  circle_C x y →
  curve_C1 ρ θ →
  line_C2 x y t →
  line_AB_polar θ →
  ratio_CD_CE = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C1_intersection_theorem_l392_39254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_united_price_is_1100_l392_39290

/-- Represents the original and discounted prices of airline flights -/
structure FlightPrices where
  delta_original : ℚ
  delta_discount : ℚ
  united_discount : ℚ
  price_difference : ℚ

/-- Calculates the original price of the United Airlines flight -/
def calculate_united_price (prices : FlightPrices) : ℚ :=
  let delta_discounted := prices.delta_original * (1 - prices.delta_discount)
  let united_discounted := delta_discounted + prices.price_difference
  united_discounted / (1 - prices.united_discount)

/-- Theorem stating that given the conditions, the United Airlines flight's original price is $1100 -/
theorem united_price_is_1100 (prices : FlightPrices) 
    (h1 : prices.delta_original = 850)
    (h2 : prices.delta_discount = 1/5)
    (h3 : prices.united_discount = 3/10)
    (h4 : prices.price_difference = 90) :
    calculate_united_price prices = 1100 := by
  sorry

#eval calculate_united_price { 
  delta_original := 850, 
  delta_discount := 1/5, 
  united_discount := 3/10, 
  price_difference := 90 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_united_price_is_1100_l392_39290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_M_to_circle_l392_39219

/-- Circle C with equation x² + y² - 2x = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 = 0}

/-- Point M with coordinates (0, 2) -/
def M : ℝ × ℝ := (0, 2)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The maximum distance between M and any point on the circle is √5 + 1 -/
theorem max_distance_M_to_circle :
  ∃ (max_dist : ℝ), max_dist = Real.sqrt 5 + 1 ∧
    ∀ (N : ℝ × ℝ), N ∈ Circle → distance M N ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_M_to_circle_l392_39219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_equality_condition_l392_39279

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^2 + b^2 + 1 / (a + b)^2 + 1 / (a * b) ≥ Real.sqrt 10 := by
  sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + b^2 + 1 / (a + b)^2 + 1 / (a * b) = Real.sqrt 10 ↔ 
  a = 2^(-(3/4 : ℝ)) ∧ b = 2^(-(3/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_equality_condition_l392_39279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l392_39263

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≥ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l392_39263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_seating_arrangements_l392_39225

/-- The number of chairs in a row -/
def num_chairs : ℕ := 10

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of professors -/
def num_professors : ℕ := 4

/-- The number of effective positions for professors (excluding first and last chairs) -/
def effective_positions : ℕ := num_chairs - 2

/-- Represents the seating arrangement problem -/
def seating_arrangements : Prop :=
  (Nat.choose effective_positions num_professors) * (Nat.factorial num_professors) = 1680

theorem prove_seating_arrangements : seating_arrangements := by
  sorry

#eval Nat.choose effective_positions num_professors * Nat.factorial num_professors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_seating_arrangements_l392_39225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_coplanar_implies_not_collinear_l392_39287

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition: Points are coplanar -/
def coplanar (A B C D : Point3D) : Prop :=
  ∃ (a b c d : ℝ), a * A.x + b * A.y + c * A.z + d = 0 ∧
                   a * B.x + b * B.y + c * B.z + d = 0 ∧
                   a * C.x + b * C.y + c * C.z + d = 0 ∧
                   a * D.x + b * D.y + c * D.z + d = 0 ∧
                   (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

/-- Definition: Three points are collinear -/
def collinear (P Q R : Point3D) : Prop :=
  ∃ (t : ℝ), (Q.x - P.x, Q.y - P.y, Q.z - P.z) = t • (R.x - P.x, R.y - P.y, R.z - P.z)

/-- Theorem: If four points are not coplanar, then any three of them are not collinear -/
theorem not_coplanar_implies_not_collinear {A B C D : Point3D} (h : ¬ coplanar A B C D) :
  ¬ collinear A B C ∧ ¬ collinear A B D ∧ ¬ collinear A C D ∧ ¬ collinear B C D := by
  sorry

#check not_coplanar_implies_not_collinear

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_coplanar_implies_not_collinear_l392_39287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_special_perpendicular_diagonal_quadrilateral_l392_39239

/-- A quadrilateral with perpendicular diagonals -/
structure PerpendicularDiagonalQuadrilateral where
  /-- The four vertices of the quadrilateral -/
  vertices : Fin 4 → ℝ × ℝ
  /-- The diagonals are perpendicular -/
  perpendicular_diagonals : 
    let d1 := vertices 2 - vertices 0
    let d2 := vertices 3 - vertices 1
    d1.1 * d2.1 + d1.2 * d2.2 = 0

/-- Predicate for a rhombus -/
def is_rhombus (q : PerpendicularDiagonalQuadrilateral) : Prop :=
  ∀ i j : Fin 4, (q.vertices i - q.vertices j).1^2 + (q.vertices i - q.vertices j).2^2 = 
                 (q.vertices 0 - q.vertices 1).1^2 + (q.vertices 0 - q.vertices 1).2^2

/-- Predicate for a rectangle -/
def is_rectangle (q : PerpendicularDiagonalQuadrilateral) : Prop :=
  ∀ i : Fin 4, 
    let v1 := q.vertices i - q.vertices ((i + 1) % 4)
    let v2 := q.vertices ((i + 1) % 4) - q.vertices ((i + 2) % 4)
    v1.1 * v2.1 + v1.2 * v2.2 = 0

/-- Predicate for a square -/
def is_square (q : PerpendicularDiagonalQuadrilateral) : Prop :=
  is_rhombus q ∧ is_rectangle q

/-- Predicate for an isosceles trapezoid -/
def is_isosceles_trapezoid (q : PerpendicularDiagonalQuadrilateral) : Prop :=
  (q.vertices 0 - q.vertices 1).1^2 + (q.vertices 0 - q.vertices 1).2^2 = 
  (q.vertices 2 - q.vertices 3).1^2 + (q.vertices 2 - q.vertices 3).2^2 ∧
  (q.vertices 0 - q.vertices 3).1^2 + (q.vertices 0 - q.vertices 3).2^2 = 
  (q.vertices 1 - q.vertices 2).1^2 + (q.vertices 1 - q.vertices 2).2^2

/-- Theorem: There exists a quadrilateral with perpendicular diagonals that is not a rhombus, rectangle, square, or isosceles trapezoid -/
theorem exists_non_special_perpendicular_diagonal_quadrilateral :
  ∃ q : PerpendicularDiagonalQuadrilateral, 
    ¬is_rhombus q ∧ ¬is_rectangle q ∧ ¬is_square q ∧ ¬is_isosceles_trapezoid q :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_special_perpendicular_diagonal_quadrilateral_l392_39239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_positive_reals_l392_39245

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4) / Real.log 0.5

-- State the theorem
theorem f_decreasing_on_positive_reals :
  ∀ x y, 2 < x → x < y → f y < f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_positive_reals_l392_39245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_point_planar_graph_exists_six_point_planar_graph_not_exists_l392_39294

/-- The degree of a vertex in a graph. -/
def Degree {V : Type*} (G : SimpleGraph V) (v : V) : ℕ := sorry

/-- A graph is planar if it can be drawn on the plane without edge crossings. -/
def IsPlanar {V : Type*} (G : SimpleGraph V) : Prop := sorry

theorem four_point_planar_graph_exists :
  ∃ (V : Type*) (G : SimpleGraph V) (v1 v2 v3 v4 : V),
    IsPlanar G ∧
    (∀ v : V, v = v1 ∨ v = v2 ∨ v = v3 ∨ v = v4) ∧
    (∀ v : V, Degree G v = 3) := by
  sorry

theorem six_point_planar_graph_not_exists :
  ¬∃ (V : Type*) (G : SimpleGraph V) (v1 v2 v3 v4 v5 v6 : V),
    IsPlanar G ∧
    (∀ v : V, v = v1 ∨ v = v2 ∨ v = v3 ∨ v = v4 ∨ v = v5 ∨ v = v6) ∧
    (∀ v : V, Degree G v = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_point_planar_graph_exists_six_point_planar_graph_not_exists_l392_39294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_GHCD_is_165_l392_39243

/-- Represents a trapezoid with parallel sides AB and CD -/
structure Trapezoid where
  ab : ℝ  -- Length of side AB
  cd : ℝ  -- Length of side CD
  h : ℝ   -- Altitude from AB to CD

/-- Calculates the area of a quadrilateral GHCD formed by connecting the midpoints 
    of the legs of the trapezoid to its longer base -/
noncomputable def area_GHCD (t : Trapezoid) : ℝ :=
  (t.h / 2) * ((t.ab + t.cd) / 2 + t.cd) / 2

/-- Theorem stating that for the given trapezoid, the area of GHCD is 165 square units -/
theorem area_GHCD_is_165 (t : Trapezoid) 
    (h_ab : t.ab = 10)
    (h_cd : t.cd = 26)
    (h_h : t.h = 15) : 
  area_GHCD t = 165 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_GHCD_is_165_l392_39243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l392_39285

/-- Given an ellipse and a hyperbola sharing the same foci, prove that the eccentricity of the ellipse is 1/2 -/
theorem ellipse_eccentricity (a b m n c : ℝ) 
    (ha : a > 0) (hb : b > 0) (hm : m > 0) (hn : n > 0) (hab : a > b)
    (h_shared_foci : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/m^2 - y^2/n^2 = 1)
    (hc : c^2 = a * m)
    (hn2 : n^2 = (2 * m^2 + c^2) / 2) : 
  let e := c / a
  e = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l392_39285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l392_39220

/-- Given two hyperbolas C₁ and C₂, prove that C₁ has the equation x² - y²/4 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →  -- Equation of C₁
  (∀ x y : ℝ, x^2/4 - y^2/16 = 1) →    -- Equation of C₂
  (∀ k : ℝ, (∃ x y : ℝ, y = k*x ∧ x^2/a^2 - y^2/b^2 = 1) ↔ 
            (∃ x y : ℝ, y = k*x ∧ x^2/4 - y^2/16 = 1)) →  -- Same asymptotes
  (∃ x : ℝ, x^2/a^2 - 0^2/b^2 = 1 ∧ x = Real.sqrt 5) →  -- Right focus at (√5, 0)
  (∀ x y : ℝ, x^2 - y^2/4 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l392_39220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toucan_count_toucan_problem_l392_39281

/-- The number of toucans after one joins an initial group -/
theorem toucan_count (initial joining : ℕ) : initial + joining = initial + joining :=
by rfl

/-- Given 2 initial toucans and 1 joining toucan, prove there are 3 toucans total -/
theorem toucan_problem : 2 + 1 = 3 :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toucan_count_toucan_problem_l392_39281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l392_39209

-- Define points A and B
def A : Fin 3 → ℝ := ![4, -1, 2]
def B : Fin 3 → ℝ := ![2, -3, 0]

-- Define vector operations
def vector_sub (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ i => v i - w i

def vector_mul (k : ℝ) (v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ i => k * v i

-- Theorem statement
theorem point_C_coordinates (C : Fin 3 → ℝ) :
  vector_sub C B = vector_mul 2 (vector_sub A C) →
  C = λ i => ![10/3, -5/3, 4/3] i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l392_39209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l392_39244

-- Define the function g(x) as noncomputable
noncomputable def g (x : ℝ) : ℝ := (Real.log x) / (x^2 + 2*x)

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (∃ (x y : ℕ), x ≠ y ∧ 
    m * x^2 + 2*m*x - Real.log x < 0 ∧
    m * y^2 + 2*m*y - Real.log y < 0 ∧
    (∀ (z : ℕ), z ≠ x ∧ z ≠ y → m * z^2 + 2*m*z - Real.log z ≥ 0)) →
  (Real.log 2 / 12 < m ∧ m < Real.log 3 / 15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l392_39244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_pi_plus_six_l392_39235

noncomputable def f (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 0 then Real.sqrt (4 - x^2)
  else if 0 < x ∧ x ≤ 2 then x + 2
  else 0

theorem integral_f_equals_pi_plus_six :
  ∫ x in (-2)..2, f x = π + 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_pi_plus_six_l392_39235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_angle_l392_39271

/-- Two circles in a plane -/
structure TwoCircles where
  S₁ : Set (ℝ × ℝ)
  S₂ : Set (ℝ × ℝ)
  isCircle₁ : ∃ (c : ℝ × ℝ) (r : ℝ), S₁ = {p | ‖p - c‖ = r}
  isCircle₂ : ∃ (c : ℝ × ℝ) (r : ℝ), S₂ = {p | ‖p - c‖ = r}
  intersect : S₁ ∩ S₂ ≠ ∅

/-- A line through a point intersecting two circles -/
structure IntersectingLine (tc : TwoCircles) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  A_in_intersection : A ∈ tc.S₁ ∩ tc.S₂
  B_in_S₁ : B ∈ tc.S₁
  C_in_S₂ : C ∈ tc.S₂
  collinear : ∃ (t : ℝ), B = A + t • (C - A)

/-- The angle formed by tangents at the intersection points -/
noncomputable def tangentAngle (tc : TwoCircles) (l : IntersectingLine tc) : ℝ :=
  sorry

/-- The main theorem: the angle is constant for any intersecting line -/
theorem constant_angle (tc : TwoCircles) :
  ∀ l₁ l₂ : IntersectingLine tc, tangentAngle tc l₁ = tangentAngle tc l₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_angle_l392_39271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_repeating_block_length_l392_39276

/-- The length of the smallest repeating block in the decimal expansion of 6/7 -/
def repeating_block_length : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 6/7

/-- Function to calculate the smallest repeating block length of a rational number -/
noncomputable def smallest_repeating_block_length (q : ℚ) : ℕ := 
  sorry

theorem fraction_repeating_block_length : 
  smallest_repeating_block_length fraction = repeating_block_length := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_repeating_block_length_l392_39276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l392_39233

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 6)

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  f A = 1 →
  a = 2 →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a = 2 * Real.sin (B / 2) * Real.sin (C / 2) →
  b = 2 * Real.sin (A / 2) * Real.sin (C / 2) →
  c = 2 * Real.sin (A / 2) * Real.sin (B / 2) →
  (1 / 2 : ℝ) * b * c * Real.sin A ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l392_39233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_2_l392_39298

-- Define the function f as noncomputable
noncomputable def f : ℝ → ℝ := fun x => 
  if x < 0 then x / (x - 1)
  else x / (x + 1)

-- State the theorem
theorem tangent_slope_at_2 (h : ∀ x, f (-x) = f x) :
  HasDerivAt f (1/9) 2 := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_2_l392_39298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_scaled_l392_39286

noncomputable def cylindrical_to_rectangular (r θ z : Real) : Real × Real × Real :=
  (r * Real.cos θ, r * Real.sin θ, z)

def scale_coordinates (x y z : Real) (scale : Real) : Real × Real × Real :=
  (scale * x, scale * y, scale * z)

theorem cylindrical_to_rectangular_scaled :
  let (r, θ, z) := (7, Real.pi / 4, -3)
  let (x, y, z) := cylindrical_to_rectangular r θ z
  let (x', y', z') := scale_coordinates x y z 2
  (x', y', z') = (7 * Real.sqrt 2, 7 * Real.sqrt 2, -6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_scaled_l392_39286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_area_floor_l392_39230

/-- The area of the region inside a large circle and outside eight smaller circles arranged in a ring --/
noncomputable def ringArea (R : ℝ) : ℝ :=
  let r := R / (2 * Real.sqrt 2 + 1)
  Real.pi * (R^2 - 8 * r^2)

/-- The problem statement --/
theorem ring_area_floor :
  ⌊ringArea 40⌋ = 1150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_area_floor_l392_39230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_population_approximation_l392_39211

/-- The initial population of a town that grows to 262,500 after 10 years with a 5% annual growth rate --/
noncomputable def initial_population : ℝ :=
  262500 / (1 + 0.05)^10

/-- The expected approximate value of the initial population --/
def expected_initial_population : ℕ := 161182

theorem initial_population_approximation :
  ⌊initial_population⌋ = expected_initial_population := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval ⌊initial_population⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_population_approximation_l392_39211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_five_sevenths_sin_l392_39203

/-- The function f for which we want to prove the equality -/
noncomputable def f : ℝ → ℝ := sorry

/-- The period of the function f -/
def T : ℝ := sorry

/-- The main theorem stating the properties of f and the resulting equality -/
theorem f_equals_five_sevenths_sin (h1 : ∀ x, f (x + T) = f x) 
  (h2 : ∀ x, Real.sin x = f x - 0.4 * f (x - π))
  (h3 : ∀ x, Real.sin x = f (x - T) - 0.4 * f (x - T - π))
  (h4 : ∀ x, Real.sin x = Real.sin (x - T)) :
  ∀ x, f x = (5/7) * Real.sin x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_five_sevenths_sin_l392_39203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_celebration_day_l392_39226

inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  deriving Repr, DecidableEq

def friday : DayOfWeek := DayOfWeek.Friday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def nthDay (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | n + 1 => nthDay (nextDay start) n

theorem celebration_day (birthDay : DayOfWeek) (days : Nat) : 
  birthDay = friday → days = 1200 → nthDay birthDay days = DayOfWeek.Monday := by
  sorry

#eval nthDay friday 1200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_celebration_day_l392_39226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l392_39250

/-- The circle with equation x^2 + y^2 = 4 -/
def myCircle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 4}

/-- The line with equation 4x - 3y + 25 = 0 -/
def myLine : Set (ℝ × ℝ) :=
  {p | 4 * p.1 - 3 * p.2 + 25 = 0}

/-- The distance from a point to the line -/
noncomputable def distToLine (p : ℝ × ℝ) : ℝ :=
  |4 * p.1 - 3 * p.2 + 25| / Real.sqrt (4^2 + (-3)^2)

/-- The maximum distance from a point on the circle to the line is 7 -/
theorem max_distance_circle_to_line :
  ∃ (p : ℝ × ℝ), p ∈ myCircle ∧ ∀ (q : ℝ × ℝ), q ∈ myCircle → distToLine q ≤ distToLine p ∧ distToLine p = 7 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l392_39250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_go_match_theorem_l392_39284

/-- Represents the outcome of a single game -/
inductive GameOutcome
| A_wins
| B_wins

/-- Represents the state of the match after the first two games -/
structure MatchState :=
  (games_won_A : Nat)
  (games_won_B : Nat)

/-- Probability of A winning a single game -/
def p_A_win : ℝ := 0.6

/-- Probability of B winning a single game -/
def p_B_win : ℝ := 0.4

/-- The initial state of the match after two games -/
def initial_state : MatchState := ⟨1, 1⟩

/-- The number of games needed to win the match -/
def games_to_win : Nat := 3

/-- ξ represents the number of games played from the 3rd game until the end of the match -/
def ξ : Nat → Prop := sorry

/-- The probability of A winning the match -/
noncomputable def prob_A_wins_match : ℝ := sorry

/-- The distribution of ξ -/
noncomputable def prob_ξ : Nat → ℝ := sorry

/-- The mathematical expectation of ξ -/
noncomputable def expectation_ξ : ℝ := sorry

theorem go_match_theorem :
  prob_A_wins_match = 0.648 ∧
  prob_ξ 2 = 0.52 ∧
  prob_ξ 3 = 0.48 ∧
  expectation_ξ = 2.48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_go_match_theorem_l392_39284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_journey_l392_39208

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The cowboy's journey -/
theorem cowboy_journey (cowboy stream cabin firewood : Point)
  (h1 : stream.y - cowboy.y = 6)
  (h2 : cabin.x - cowboy.x = 12)
  (h3 : cowboy.y - cabin.y = 10)
  (h4 : firewood.x - cowboy.x = 5)
  (h5 : firewood.y = stream.y) :
  distance cowboy stream + distance stream firewood + distance firewood cabin = 11 + Real.sqrt 305 := by
  sorry

#check cowboy_journey

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_journey_l392_39208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_quadrilateral_area_l392_39265

noncomputable section

/-- Square EFGH with side length 40 and point Q inside -/
structure Square :=
  (E F G H Q : ℝ × ℝ)
  (side_length : ℝ)
  (is_square : side_length = 40)
  (Q_inside : Q.1 ≥ 0 ∧ Q.1 ≤ 40 ∧ Q.2 ≥ 0 ∧ Q.2 ≤ 40)
  (EQ_length : Real.sqrt ((E.1 - Q.1)^2 + (E.2 - Q.2)^2) = 15)
  (FQ_length : Real.sqrt ((F.1 - Q.1)^2 + (F.2 - Q.2)^2) = 34)

/-- Centroid of a triangle -/
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- Area of a quadrilateral given its vertices -/
def quadrilateral_area (A B C D : ℝ × ℝ) : ℝ :=
  let s1 := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let s2 := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let s3 := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let s4 := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let p := (s1 + s2 + s3 + s4) / 2
  Real.sqrt ((p - s1) * (p - s2) * (p - s3) * (p - s4))

theorem centroid_quadrilateral_area (sq : Square) :
  let c1 := centroid sq.E sq.F sq.Q
  let c2 := centroid sq.F sq.G sq.Q
  let c3 := centroid sq.G sq.H sq.Q
  let c4 := centroid sq.H sq.E sq.Q
  quadrilateral_area c1 c2 c3 c4 = 1600 / 9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_quadrilateral_area_l392_39265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_multiple_of_five_divisor_25_factorial_l392_39227

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_divisor (d n : ℕ) : Prop := n % d = 0

def is_odd (n : ℕ) : Prop := n % 2 ≠ 0

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def divisors (n : ℕ) : Finset ℕ := Finset.filter (λ d => d > 0 ∧ n % d = 0) (Finset.range (n + 1))

theorem probability_odd_multiple_of_five_divisor_25_factorial :
  let n := factorial 25
  let all_divisors := divisors n
  let odd_multiple_of_five_divisors := all_divisors.filter (λ d => d % 2 ≠ 0 ∧ d % 5 = 0)
  (odd_multiple_of_five_divisors.card : ℚ) / all_divisors.card = 7 / 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_multiple_of_five_divisor_25_factorial_l392_39227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picture_framing_theorem_l392_39216

/-- The minimum number of linear feet of framing required for a picture with given dimensions and border -/
def min_framing_feet (width_inch : ℕ) (height_inch : ℕ) (border_inch : ℕ) : ℕ :=
  let enlarged_width := 2 * width_inch
  let enlarged_height := 2 * height_inch
  let final_width := enlarged_width + 2 * border_inch
  let final_height := enlarged_height + 2 * border_inch
  let perimeter_inch := 2 * (final_width + final_height)
  (perimeter_inch + 11) / 12  -- Round up to the nearest foot

theorem picture_framing_theorem :
  min_framing_feet 5 7 3 = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_picture_framing_theorem_l392_39216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l392_39210

variable (A B C : ℝ) (r R : ℝ)

-- Define the triangle ABC
def is_triangle (A B C : ℝ) : Prop := A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi

-- Define inradius and circumradius
def is_inradius (r : ℝ) (A B C : ℝ) : Prop := r > 0
def is_circumradius (R : ℝ) (A B C : ℝ) : Prop := R > 0

theorem triangle_inequality (h1 : is_triangle A B C) (h2 : is_inradius r A B C) (h3 : is_circumradius R A B C) :
  Real.sin (A/2) * Real.sin (B/2) + Real.sin (B/2) * Real.sin (C/2) + Real.sin (C/2) * Real.sin (A/2) ≤ 5/8 + r/(4*R) :=
by
  sorry

#check triangle_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l392_39210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_escape_minimum_t_l392_39288

theorem fox_escape_minimum_t : ∃ t : ℕ, 
  (∀ s : ℕ, s < t → (123 : ℚ) / 100 * (1 - s / 100) ≥ 1) ∧ 
  ((123 : ℚ) / 100 * (1 - t / 100) < 1) ∧
  t = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fox_escape_minimum_t_l392_39288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_royal_family_children_l392_39259

/-- Represents the number of years that have passed -/
def n : ℕ → ℕ := sorry

/-- Represents the number of daughters -/
def d : ℕ → ℕ := sorry

/-- Represents the total number of children -/
def total_children (d : ℕ → ℕ) : ℕ → ℕ := λ i => d i + 3

/-- The initial age of the king and queen -/
def initial_parent_age : ℕ := 35

/-- The initial total age of the children -/
def initial_children_age : ℕ := 35

theorem royal_family_children (d : ℕ → ℕ) (n : ℕ → ℕ) :
  (∀ i, d i ≥ 1) →
  (∀ i, total_children d i ≤ 20) →
  (∀ i, initial_parent_age + initial_parent_age + 2 * n i = 
        initial_children_age + (total_children d i) * n i) →
  (∃ i, total_children d i = 7 ∨ total_children d i = 9) :=
by
  sorry

#check royal_family_children

end NUMINAMATH_CALUDE_ERRORFEEDBACK_royal_family_children_l392_39259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_configuration_2012_l392_39257

/-- A configuration of circles on a plane -/
structure CircleConfiguration where
  circles : Finset (ℝ × ℝ)
  radius : ℝ
  touching : (c : ℝ × ℝ) → Finset (ℝ × ℝ)

/-- A valid circle configuration satisfies the touching condition -/
def ValidConfiguration (config : CircleConfiguration) : Prop :=
  ∀ c ∈ config.circles, (config.touching c).card ≥ 3

/-- There exists a valid configuration of 2012 circles -/
theorem exists_valid_configuration_2012 :
  ∃ (config : CircleConfiguration), config.circles.card = 2012 ∧ ValidConfiguration config :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_configuration_2012_l392_39257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_one_l392_39282

def modifiedFibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 2
  | n + 2 => ((modifiedFibonacci (n + 1))^2 + (modifiedFibonacci n)^2) % 8

def appearsInSequence (d : ℕ) : Prop :=
  ∃ n : ℕ, modifiedFibonacci n % 8 = d

theorem last_digit_is_one :
  (∀ d : ℕ, d < 8 → appearsInSequence d) ∧
  (∀ d : ℕ, d < 8 → d ≠ 1 → ∃ n : ℕ, (∀ m : ℕ, m ≤ n → modifiedFibonacci m % 8 ≠ 1) ∧ modifiedFibonacci n % 8 = d) :=
by
  sorry

#eval [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14].map modifiedFibonacci

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_one_l392_39282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_propositions_l392_39278

/-- Propositions about geometry in space -/
def p₁ : Prop := True
def p₂ : Prop := False
def p₃ : Prop := False
def p₄ : Prop := True

/-- The theorem to be proved -/
theorem geometry_propositions :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_propositions_l392_39278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completes_in_twelve_point_five_days_l392_39236

noncomputable section

/-- The number of days A and B together can finish the work -/
def ab_days : ℝ := 12

/-- The number of days A alone can finish the work -/
def a_days : ℝ := 20

/-- The number of days C can finish the work -/
def c_days : ℝ := 30

/-- The number of days A, B, and C work together -/
def initial_days : ℝ := 5

/-- The rate at which A and B work together -/
noncomputable def ab_rate : ℝ := 1 / ab_days

/-- The rate at which A works alone -/
noncomputable def a_rate : ℝ := 1 / a_days

/-- The rate at which B works alone -/
noncomputable def b_rate : ℝ := ab_rate - a_rate

/-- The rate at which C works alone -/
noncomputable def c_rate : ℝ := 1 / c_days

/-- The combined work rate of A, B, and C -/
noncomputable def abc_rate : ℝ := a_rate + b_rate + c_rate

/-- The amount of work done in the first 5 days -/
noncomputable def work_done : ℝ := abc_rate * initial_days

/-- The amount of work remaining -/
noncomputable def work_remaining : ℝ := 1 - work_done

/-- The time B needs to complete the remaining work -/
noncomputable def b_completion_time : ℝ := work_remaining / b_rate

theorem b_completes_in_twelve_point_five_days :
  b_completion_time = 12.5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completes_in_twelve_point_five_days_l392_39236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_sum_l392_39277

theorem polynomial_identity_sum (d₁ d₂ d₃ d₄ e₁ e₂ e₃ e₄ : ℝ) :
  (∀ x : ℝ, x^8 - 2*x^7 + 2*x^6 - 2*x^5 + 2*x^4 - 2*x^3 + 2*x^2 - 2*x + 1 =
    (x^2 + d₁*x + e₁)*(x^2 + d₂*x + e₂)*(x^2 + d₃*x + e₃)*(x^2 + d₄*x + e₄)) →
  d₁*e₁ + d₂*e₂ + d₃*e₃ + d₄*e₄ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_sum_l392_39277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_distribution_properties_l392_39223

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  prob_success : 0 ≤ p ∧ p ≤ 1

/-- The probability mass function for a binomial distribution -/
def binomialPMF (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The expected value of a binomial distribution -/
def binomialExpectation (n : ℕ) (p : ℝ) : ℝ := n * p

/-- Theorem about the properties of a specific binomial distribution -/
theorem binomial_distribution_properties :
  let X : BinomialDistribution 6 (2/3) := ⟨by norm_num⟩
  binomialPMF 6 (2/3) 2 = 20/243 ∧ binomialExpectation 6 (2/3) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_distribution_properties_l392_39223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jackie_free_shipping_amount_l392_39241

/-- The amount Jackie needs to spend to be eligible for free shipping -/
def free_shipping_amount (shampoo_cost conditioner_cost lotion_cost : ℚ)
  (lotion_quantity : ℕ) (additional_amount : ℚ) : ℚ :=
  shampoo_cost + conditioner_cost + lotion_cost * lotion_quantity + additional_amount

/-- Theorem: Jackie needs to spend $50.00 to be eligible for free shipping -/
theorem jackie_free_shipping_amount :
  free_shipping_amount 10 10 6 3 12 = 50 := by
  -- Unfold the definition of free_shipping_amount
  unfold free_shipping_amount
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jackie_free_shipping_amount_l392_39241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_B_l392_39272

theorem max_tan_B (A B : ℝ) (h1 : 0 < A) (h2 : A < π/2) (h3 : 0 < B) (h4 : B < π/2)
  (h5 : Real.tan (A + B) = 2 * Real.tan A) :
  ∃ (C : ℝ), C = Real.sqrt 2 / 4 ∧ Real.tan B ≤ C ∧ (∃ (E : ℝ), Real.tan E = C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_B_l392_39272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_l392_39231

/-- The speed of a boat in still water, given its downstream and upstream times -/
theorem boat_speed (distance : ℝ) (downstream_time upstream_time : ℝ) 
  (h1 : distance > 0)
  (h2 : downstream_time > 0)
  (h3 : upstream_time > 0)
  (h4 : distance = 10)
  (h5 : downstream_time = 3)
  (h6 : upstream_time = 6) :
  (distance / downstream_time + distance / upstream_time) / 2 = 2.5 := by
  sorry

#check boat_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_l392_39231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_values_l392_39206

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def Lines_are_perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- Definition of line l₁ -/
def line_l₁ (k : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ k * x + (1 - k) * y - 3 = 0

/-- Definition of line l₂ -/
def line_l₂ (k : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ (k - 1) * x + (2 * k + 3) * y - 2 = 0

/-- The slope of line l₁ -/
noncomputable def slope_l₁ (k : ℝ) : ℝ := -k / (1 - k)

/-- The slope of line l₂ -/
noncomputable def slope_l₂ (k : ℝ) : ℝ := -(k - 1) / (2 * k + 3)

theorem perpendicular_lines_k_values :
  ∀ k : ℝ, k ≠ 1 ∧ 2 * k + 3 ≠ 0 →
    (Lines_are_perpendicular (slope_l₁ k) (slope_l₂ k) ↔ k = -3 ∨ k = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_k_values_l392_39206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_change_l392_39238

theorem rectangle_perimeter_change (s h : ℝ) (h_pos : h > 0) (s_pos : s > 0) :
  h = 1.5 * s →
  (2 * (0.8 * s + 1.3 * h) - 2 * (s + h)) / (2 * (s + h)) = 0.1 := by
  intro h_eq
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_change_l392_39238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passage_time_l392_39228

/-- The time (in seconds) required for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 360 meters traveling at 90 km/hour 
    will take 20 seconds to pass a bridge of length 140 meters -/
theorem train_bridge_passage_time :
  time_to_pass_bridge 360 90 140 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passage_time_l392_39228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l392_39205

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.cos (ω * x)

theorem omega_range (ω : ℝ) (h1 : ω > 0) :
  (∀ x₁ x₂, π / 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < π → f ω x₁ > f ω x₂) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l392_39205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l392_39207

def negative_fraction_set : Set ℚ := {x | x < 0 ∧ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}
def integer_set : Set ℤ := Set.univ
def rational_number_set : Set ℚ := Set.univ
def non_positive_integer_set : Set ℤ := {x | x ≤ 0}

theorem number_categorization :
  (-5/2 ∈ negative_fraction_set) ∧
  (-2/3 ∈ negative_fraction_set) ∧
  (-5/99 ∈ negative_fraction_set) ∧
  (0 ∈ integer_set) ∧
  (8 ∈ integer_set) ∧
  (-2 ∈ integer_set) ∧
  (-5/2 ∈ rational_number_set) ∧
  (11/2 ∈ rational_number_set) ∧
  (0 ∈ rational_number_set) ∧
  (8 ∈ rational_number_set) ∧
  (-2 ∈ rational_number_set) ∧
  (7/10 ∈ rational_number_set) ∧
  (-2/3 ∈ rational_number_set) ∧
  (3/4 ∈ rational_number_set) ∧
  (-5/99 ∈ rational_number_set) ∧
  (0 ∈ non_positive_integer_set) ∧
  (-2 ∈ non_positive_integer_set) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l392_39207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_max_value_l392_39253

theorem sin_sum_max_value : 
  ∃ x : ℝ, Real.sin x + Real.sin (2 * x) + Real.sin (3 * x) ≥ (3 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_max_value_l392_39253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_perfect_squares_l392_39292

theorem triangle_area_perfect_squares : ∃ (a b c d e f : ℕ),
  -- Triangle inequality for set 1
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
  -- Triangle inequality for set 2
  (d + e > f) ∧ (e + f > d) ∧ (f + d > e) ∧
  -- Heron's formula factors are perfect squares for set 1
  (∃ (k l m n : ℕ), 
    (((a + b + c) / 2) * (((a + b + c) / 2) - a) * (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c) = (k * l * m * n)^2)) ∧
  -- Heron's formula factors are perfect squares for set 2
  (∃ (p q r t : ℕ),
    (((d + e + f) / 2) * (((d + e + f) / 2) - d) * (((d + e + f) / 2) - e) * (((d + e + f) / 2) - f) = (p * q * r * t)^2)) ∧
  -- Set 1 forms an isosceles triangle
  ((a = b ∧ b ≠ c) ∨ (b = c ∧ c ≠ a) ∨ (c = a ∧ a ≠ b)) ∧
  -- Set 2 forms a scalene triangle
  (d ≠ e ∧ e ≠ f ∧ f ≠ d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_perfect_squares_l392_39292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l392_39293

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line
def line_eq (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

theorem distance_circle_center_to_line :
  ∃ (x₀ y₀ : ℝ), circle_eq x₀ y₀ ∧
  (∀ (x y : ℝ), circle_eq x y → (x - x₀)^2 + (y - y₀)^2 ≤ (x - 1)^2 + y^2) ∧
  distance_point_to_line x₀ y₀ (Real.sqrt 3 / 3) (-1) 0 = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l392_39293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_l392_39269

theorem sum_of_roots_quadratic :
  let a : ℝ := 5 + 3 * Real.sqrt 3
  let b : ℝ := -(1 + 2 * Real.sqrt 3)
  let c : ℝ := 1
  let equation := fun x : ℝ => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  sum_of_roots = -13/2 + 7/2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_l392_39269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l392_39258

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The shortest path problem for the cowboy -/
theorem cowboy_shortest_path (river_y cabin_x cabin_y : ℝ) 
  (h1 : river_y = 0)
  (h2 : cabin_x = 6)
  (h3 : cabin_y = -14) : 
  let cowboy : Point := ⟨0, -5⟩
  let cabin : Point := ⟨cabin_x, cabin_y⟩
  let river_point : Point := ⟨0, river_y⟩
  5 + distance ⟨0, 5⟩ cabin = 5 + Real.sqrt 397 := by
  sorry

#check cowboy_shortest_path

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l392_39258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l392_39251

/-- Definition of a triangle -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Definition of an isosceles triangle -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.side1 = t.side2 ∨ t.side2 = t.side3 ∨ t.side3 = t.side1

/-- Definition of triangle perimeter -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.side1 + t.side2 + t.side3

/-- Definition of triangle similarity -/
def Triangle.isSimilarTo (t1 t2 : Triangle) (scale : ℝ) : Prop :=
  t1.side1 = scale * t2.side1 ∧
  t1.side2 = scale * t2.side2 ∧
  t1.side3 = scale * t2.side3

/-- The perimeter of a triangle similar to an isosceles triangle with sides 7, 7, and 12,
    where the longest side of the similar triangle is 30, is equal to 65. -/
theorem similar_triangle_perimeter : 
  ∀ (smaller larger : Triangle) (scale : ℝ),
  smaller.isIsosceles →
  smaller.side1 = 7 →
  smaller.side2 = 7 →
  smaller.side3 = 12 →
  scale * smaller.side3 = 30 →
  larger.isSimilarTo smaller scale →
  larger.perimeter = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_l392_39251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_line_l392_39215

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the line
def line (x y : ℝ) : Prop := x - y = 0

-- Define the left focus of the ellipse
noncomputable def left_focus : ℝ × ℝ := (-2 * Real.sqrt 3, 0)

-- Theorem statement
theorem distance_from_focus_to_line :
  ∃ (d : ℝ), d = Real.sqrt 6 ∧
  d = abs (left_focus.1 - left_focus.2) / Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_line_l392_39215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_of_17_eq_8_l392_39222

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ+) : ℕ := sum_of_digits ((n : ℕ)^2 - 2*n + 2)

/-- Recursive definition of f_k -/
def f_k : ℕ → ℕ+ → ℕ
  | 0, n => f n
  | k+1, n => f_k k ⟨f_k k n, by sorry⟩

theorem f_2010_of_17_eq_8 : f_k 2010 17 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_of_17_eq_8_l392_39222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l392_39264

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 6) * (Real.sqrt 4 / Real.sqrt 7) * ((27 ^ (1/3 : ℝ)) / Real.sqrt 9) = Real.sqrt 14 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_rationalize_l392_39264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2013_l392_39221

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0
  | (n + 1) => (Real.sqrt 3 + sequence_a n) / (1 - Real.sqrt 3 * sequence_a n)

theorem sequence_a_2013 : sequence_a 2013 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2013_l392_39221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_lengths_theorem_l392_39270

/-- An isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  m : ℝ
  n : ℝ
  h_positive_m : 0 < m
  h_positive_n : 0 < n

/-- Lengths of tangent segments in an isosceles triangle with inscribed circle -/
noncomputable def tangent_segment_lengths (triangle : IsoscelesTriangleWithInscribedCircle) : ℝ × ℝ :=
  let m := triangle.m
  let n := triangle.n
  (2 * m * n / (m + 2 * n), n * (m + n) / (m + 2 * n))

/-- Theorem: The lengths of tangent segments MN and KL in an isosceles triangle
    with an inscribed circle are 2mn/(m+2n) and n(m+n)/(m+2n) respectively -/
theorem tangent_segment_lengths_theorem (triangle : IsoscelesTriangleWithInscribedCircle) :
  tangent_segment_lengths triangle = (2 * triangle.m * triangle.n / (triangle.m + 2 * triangle.n),
                                      triangle.n * (triangle.m + triangle.n) / (triangle.m + 2 * triangle.n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_segment_lengths_theorem_l392_39270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_upper_bound_P_l392_39291

/-- The probability of not buying a product at the nth push -/
def P : ℕ → ℚ
| 0 => 1 -- Adding a base case for 0, though it's not used in the problem
| 1 => 9/11
| (n+2) => 1/12 * P (n+1) + 2/3

/-- The theorem stating the minimum upper bound for P(n) when n ≥ 2 -/
theorem min_upper_bound_P : 
  ∃ M : ℚ, (∀ n : ℕ, n ≥ 2 → P n ≤ M) ∧ (∀ M' : ℚ, (∀ n : ℕ, n ≥ 2 → P n ≤ M') → M ≤ M') ∧ M = 97/132 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_upper_bound_P_l392_39291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_labelings_l392_39229

/-- Represents a labeling of a cube's vertices -/
def CubeLabeling := Fin 8 → Fin 8

/-- The sum of labels on a face of the cube -/
def face_sum (l : CubeLabeling) (face : Fin 6) : ℕ :=
  sorry

/-- Predicate for a valid cube labeling -/
def is_valid_labeling (l : CubeLabeling) : Prop :=
  (∀ i j : Fin 8, i ≠ j → l i ≠ l j) ∧
  (∀ face₁ face₂ : Fin 6, face_sum l face₁ = face_sum l face₂)

/-- Equivalence relation for cube labelings under rotation -/
def labeling_equiv : Setoid CubeLabeling :=
  ⟨λ l₁ l₂ => sorry, sorry⟩

/-- The set of all valid labelings -/
def valid_labelings : Set CubeLabeling :=
  {l | is_valid_labeling l}

/-- The quotient of valid labelings by rotation equivalence -/
def distinct_labelings : Set (Quotient labeling_equiv) :=
  Quotient.mk labeling_equiv '' valid_labelings

/-- Assume finiteness of distinct_labelings -/
instance : Fintype (Quotient labeling_equiv) := sorry

theorem count_distinct_labelings :
  Fintype.card (Quotient labeling_equiv) = 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_labelings_l392_39229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l392_39202

noncomputable def f (x : ℝ) : ℝ := Real.log x + (2*x + 1) / x

theorem f_properties :
  let f' := λ x ↦ (x - 2) / (x^2)
  (∀ x > 0, HasDerivAt f (f' x) x) ∧
  (HasDerivAt f (1/4) 2) ∧
  (MonotoneOn f (Set.Icc 0 2)) ∧
  (MonotoneOn f (Set.Ioi 2)) ∧
  (∃ m : ℤ, m = 5 ∧ 
    (∀ n : ℤ, n < m → 
      ∃ x : ℝ, x > 1 ∧ f x ≥ (n * (x - 1) + 2) / x) ∧
    (∃ x : ℝ, x > 1 ∧ f x < (m * (x - 1) + 2) / x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l392_39202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l392_39280

theorem sufficient_not_necessary : 
  ∃ (x : ℝ), (x = 3 → x^2 - x - 6 = 0) ∧ (∀ y : ℝ, y^2 - y - 6 = 0 → y = 3 ∨ y = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l392_39280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_kenny_visibility_l392_39249

noncomputable section

-- Define the constants
def building_side : ℝ := 100
def path_distance : ℝ := 300
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2

-- Define the positions of Jenny and Kenny as functions of time
noncomputable def jenny_position (t : ℝ) : ℝ × ℝ := (-50 + jenny_speed * t, path_distance / 2)
noncomputable def kenny_position (t : ℝ) : ℝ × ℝ := (-50 + kenny_speed * t, -path_distance / 2)

-- Define when they can see each other (when both are past the building)
def can_see_each_other (t : ℝ) : Prop :=
  (jenny_position t).1 > building_side / 2 ∧ (kenny_position t).1 > building_side / 2

-- The theorem to prove
theorem jenny_kenny_visibility : can_see_each_other 50 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_kenny_visibility_l392_39249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l392_39261

theorem parallel_vectors_angle (x : ℝ) : 
  let a : Fin 2 → ℝ := ![Real.sin x, 1]
  let b : Fin 2 → ℝ := ![1/2, Real.cos x]
  (∃ (k : ℝ), a = k • b) →  -- parallel vectors condition
  0 < x → x < π/2 →           -- acute angle condition
  x = π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l392_39261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_l392_39246

/-- The line equation forming a triangle with coordinate axes -/
def line_equation (x y : ℝ) : Prop := 15 * x + 3 * y = 90

/-- The x-intercept of the line -/
noncomputable def x_intercept : ℝ := 90 / 15

/-- The y-intercept of the line -/
noncomputable def y_intercept : ℝ := 90 / 3

/-- The hypotenuse of the triangle -/
noncomputable def hypotenuse : ℝ := Real.sqrt (x_intercept ^ 2 + y_intercept ^ 2)

/-- The area of the triangle -/
noncomputable def triangle_area : ℝ := (1 / 2) * x_intercept * y_intercept

/-- The altitude from the origin to the line -/
noncomputable def altitude_from_origin : ℝ := (2 * triangle_area) / hypotenuse

/-- Theorem: The sum of the altitudes of the triangle formed by the line 15x + 3y = 90 and the coordinate axes is 36 + 90/√234 -/
theorem sum_of_altitudes :
  x_intercept + y_intercept + altitude_from_origin = 36 + 90 / Real.sqrt 234 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_l392_39246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_parameter_l392_39296

/-- 
Given a parabola with equation y^2 = 2px where p > 0,
and its directrix has equation x = -4,
prove that p = 8.
-/
theorem parabola_directrix_parameter (p : ℝ) : 
  p > 0 → 
  (∀ x y : ℝ, y^2 = 2*p*x) → 
  (∀ x : ℝ, x = -4 → x = -p/2) → 
  p = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_parameter_l392_39296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_score_probability_l392_39240

/-- Represents the point value of a region on the dartboard -/
inductive PointValue : Type
| Three : PointValue
| Four : PointValue

/-- Represents a region on the dartboard -/
structure Region where
  area : ℝ
  value : PointValue

/-- Represents the dartboard -/
structure Dartboard where
  regions : List Region
  totalArea : ℝ

/-- Calculates the probability of hitting a region -/
noncomputable def hitProbability (r : Region) (d : Dartboard) : ℝ :=
  r.area / d.totalArea

/-- Determines if a point value is even -/
def isEven : PointValue → Bool
  | PointValue.Three => false
  | PointValue.Four => true

/-- Calculates the probability of getting an even score with two throws -/
noncomputable def evenScoreProbability (d : Dartboard) : ℝ :=
  sorry

/-- The dartboard as described in the problem -/
noncomputable def problemDartboard : Dartboard where
  regions := [
    { area := (16 * Real.pi) / 3, value := PointValue.Three },
    { area := (16 * Real.pi) / 3, value := PointValue.Four },
    { area := (16 * Real.pi) / 3, value := PointValue.Three },
    { area := (48 * Real.pi) / 3, value := PointValue.Four },
    { area := (48 * Real.pi) / 3, value := PointValue.Three },
    { area := (48 * Real.pi) / 3, value := PointValue.Three }
  ]
  totalArea := 64 * Real.pi

/-- Theorem stating the probability of an even score -/
theorem even_score_probability :
  evenScoreProbability problemDartboard = 37 / 72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_score_probability_l392_39240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_3_l392_39204

-- Define the circle in polar coordinates
def polar_circle (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Define the line in polar coordinates
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.sin θ = 3

-- Define the length of the chord
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem chord_length_is_2_sqrt_3 : 
  ∃ (c : ℝ), c = chord_length ∧ c = 2 * Real.sqrt 3 := by
  use chord_length
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_3_l392_39204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_calculation_l392_39262

theorem correct_calculation : -(1/2) - (-(1/3)) = -(1/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_calculation_l392_39262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equation_solutions_l392_39224

open Real Set

-- Define the interval
def I : Set ℝ := Icc 0 (arctan 1000)

-- Define the function T(x) = tan x - x
noncomputable def T (x : ℝ) : ℝ := tan x - x

-- State the theorem
theorem tan_equation_solutions :
  (∀ θ, 0 < θ → θ < π/2 → tan θ > θ) →
  (∃! (s : Finset ℝ), s.card = 318 ∧ ∀ x ∈ s, x ∈ I ∧ tan x = tan (tan x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equation_solutions_l392_39224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_l392_39247

/-- Given three points on a straight line, prove that the third point's y-coordinate is 13 -/
theorem points_on_line (m : ℝ) : 
  let p1 : ℝ × ℝ := (1, -2)
  let p2 : ℝ × ℝ := (3, 4)
  let p3 : ℝ × ℝ := (6, m/3)
  (p1.2 - p2.2) / (p1.1 - p2.1) = (p2.2 - p3.2) / (p2.1 - p3.1) → m = 39 := by
  sorry

#check points_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_l392_39247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_five_l392_39267

-- Define the vertices of the triangle
def A : ℝ × ℝ := (5, -3)
def B : ℝ × ℝ := (0, 2)
def C : ℝ × ℝ := (4, -4)

-- Define the function to calculate the area of a triangle given its vertices
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Theorem statement
theorem triangle_area_is_five :
  triangleArea A B C = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_five_l392_39267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_satisfies_conditions_l392_39248

theorem matrix_satisfies_conditions : ∃ (M : Matrix (Fin 2) (Fin 2) ℝ),
  M.vecMul (![1, 2] : Fin 2 → ℝ) = ![- 4, 4] ∧
  M.vecMul (![- 3, 1] : Fin 2 → ℝ) = ![- 23, 2] :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_satisfies_conditions_l392_39248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_reflection_l392_39217

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Define the point A
def point_A : ℝ × ℝ := (-3, 3)

-- Define the x-axis
def x_axis (y : ℝ) : Prop := y = 0

-- Define a general line equation
def line_equation (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define the reflection property
def is_reflection (incident_line reflected_line : ℝ → ℝ → Prop) : Prop :=
  ∃ (x : ℝ), incident_line x 0 ∧ reflected_line x 0 ∧
  ∀ (y : ℝ), incident_line x y → reflected_line x (-y)

-- Define the tangent property
def is_tangent (line : ℝ → ℝ → Prop) : Prop :=
  ∃! (x y : ℝ), circle_C x y ∧ line x y

-- State the theorem
theorem light_ray_reflection :
  ∃ (reflected_line : ℝ → ℝ → Prop),
    (∃ (a b c : ℝ), reflected_line = line_equation a b c) ∧
    is_reflection (line_equation 3 4 15) reflected_line ∧
    is_tangent reflected_line ∧
    ((reflected_line = line_equation 3 (-4) (-3)) ∨
     (reflected_line = line_equation 4 (-3) (-3))) ∧
    (let (ax, ay) := point_A;
     ∃ (tx ty : ℝ), circle_C tx ty ∧ reflected_line tx ty ∧
     Real.sqrt ((tx - ax)^2 + (ty - ay)^2) = 7) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_reflection_l392_39217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equilateral_triangle_on_grid_l392_39295

/-- A point on the integer grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- The slope between two grid points is rational -/
def gridSlope (p1 p2 : GridPoint) : ℚ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- An equilateral triangle on the grid -/
structure EquilateralTriangle where
  a : GridPoint
  b : GridPoint
  c : GridPoint

/-- The theorem stating that an equilateral triangle cannot be drawn on the integer grid -/
theorem no_equilateral_triangle_on_grid :
  ¬ ∃ (t : EquilateralTriangle), True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equilateral_triangle_on_grid_l392_39295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_phi_value_l392_39218

-- Define the function f as noncomputable
noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ) + Real.sqrt 3 * Real.cos (2 * x + φ)

-- State the theorem
theorem even_function_phi_value (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi) 
  (h3 : ∀ x, f x φ = f (-x) φ) : 
  φ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_phi_value_l392_39218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_value_l392_39255

/-- The length of each segment between adjacent points -/
def segment_length : ℝ := 5

/-- The number of segments -/
def num_segments : ℕ := 5

/-- The diameter of the large semicircle -/
def large_diameter : ℝ := segment_length * num_segments

/-- The area of a semicircle given its diameter -/
noncomputable def semicircle_area (d : ℝ) : ℝ := (Real.pi * d^2) / 8

/-- The shaded area in the diagram -/
noncomputable def shaded_area : ℝ :=
  semicircle_area large_diameter + semicircle_area segment_length

theorem shaded_area_value : shaded_area = (325 / 4) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_value_l392_39255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_from_interest_and_discount_l392_39274

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- True discount calculation -/
noncomputable def true_discount (principal interest : ℝ) : ℝ :=
  (interest * principal) / (principal + interest)

/-- Theorem stating the principal given simple interest and true discount -/
theorem principal_from_interest_and_discount 
  (P : ℝ) 
  (h1 : ∃ (r t : ℝ), simple_interest P r t = 85) 
  (h2 : true_discount P 85 = 80) : 
  P = 1360 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_from_interest_and_discount_l392_39274
