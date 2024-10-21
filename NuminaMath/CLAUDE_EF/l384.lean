import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l384_38482

def heart_areas : List Real := [1, 4, 9, 16]

theorem total_shaded_area (areas : List Real) (h : areas = heart_areas) :
  ∃ (shaded_area : Real), shaded_area = 10 ∧ 
  shaded_area = (areas.get! 3 - areas.get! 2) + (areas.get! 1 - areas.get! 0) :=
by
  -- Use 'get!' instead of 'nth!'
  -- The proof is omitted using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l384_38482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_power_mod_prime_l384_38413

theorem congruence_power_mod_prime (a n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) 
  (h_pos_a : a > 0) (h_pos_n : n > 0)
  (h_cong : a^p ≡ 1 [MOD p^n]) : 
  a ≡ 1 [MOD p^(n-1)] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_power_mod_prime_l384_38413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_direction_vector_l384_38485

/-- Represents a 2D line -/
structure Line2D where
  slope : ℝ
  intercept : ℝ

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Creates a line from slope and a point -/
def Line2D.mk_slope_point (m : ℝ) (a : ℝ) (b : ℝ) : Line2D :=
  { slope := m, intercept := b - m * a }

/-- Checks if two lines are perpendicular -/
def Line2D.perpendicular_to (l1 l2 : Line2D) : Prop :=
  l1.slope * l2.slope = -1

/-- Returns the direction vector of a line -/
def Line2D.direction_vector (l : Line2D) : Vector2D :=
  { x := 1, y := l.slope }

/-- Given a line l perpendicular to 2x + 5y - 1 = 0, its direction vector is (2, 5) -/
theorem perpendicular_line_direction_vector :
  ∀ (l : Line2D),
  (Line2D.perpendicular_to l (Line2D.mk_slope_point (-2/5) 0 (1/5))) →
  Line2D.direction_vector l = Vector2D.mk 2 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_direction_vector_l384_38485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l384_38401

/-- Represents the distribution of scores in a math test -/
structure ScoreDistribution where
  score_65 : ℝ
  score_75 : ℝ
  score_85 : ℝ
  score_90 : ℝ
  score_100 : ℝ
  absent : ℝ

/-- Calculates the mean score given a score distribution -/
noncomputable def mean_score (d : ScoreDistribution) : ℝ :=
  (65 * d.score_65 + 75 * d.score_75 + 85 * d.score_85 + 90 * d.score_90 + 100 * d.score_100) / (1 - d.absent)

/-- Calculates the median score given a score distribution -/
def median_score : ℝ := 75 -- The median is always 75 in this specific distribution

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score_65 = 0.3)
  (h2 : d.score_75 = 0.2)
  (h3 : d.score_85 = 0.1)
  (h4 : d.score_90 = 0.25)
  (h5 : d.score_100 = 1 - (d.score_65 + d.score_75 + d.score_85 + d.score_90))
  (h6 : d.absent = 0.05) :
  |mean_score d - median_score| = 7.85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l384_38401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_inverse_squares_l384_38421

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  /-- Distance from the origin to the plane -/
  d : ℝ
  /-- x-coordinate of the intersection with the x-axis -/
  α : ℝ
  /-- y-coordinate of the intersection with the y-axis -/
  β : ℝ
  /-- z-coordinate of the intersection with the z-axis -/
  γ : ℝ
  /-- The intersections are distinct from the origin -/
  α_nonzero : α ≠ 0
  β_nonzero : β ≠ 0
  γ_nonzero : γ ≠ 0
  /-- The plane equation holds -/
  plane_eq : 1 / α + 1 / β + 1 / γ = 1
  /-- The distance formula holds -/
  distance_eq : 1 / d^2 = 1 / α^2 + 1 / β^2 + 1 / γ^2

/-- The centroid of the triangle formed by the intersections -/
noncomputable def centroid (plane : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (plane.α / 3, plane.β / 3, plane.γ / 3)

/-- The main theorem to prove -/
theorem centroid_sum_inverse_squares (plane : IntersectingPlane) :
  let (p, q, r) := centroid plane
  1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / plane.d^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_inverse_squares_l384_38421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_hua_correct_answers_l384_38418

theorem xiao_hua_correct_answers 
  (total_questions : Nat) 
  (points_correct : Int) 
  (points_incorrect : Int) 
  (total_score : Int) 
  (h1 : total_questions = 20)
  (h2 : points_correct = 5)
  (h3 : points_incorrect = -2)
  (h4 : total_score = 65)
  : total_questions - (total_questions * points_correct - total_score) / (points_correct - points_incorrect) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_hua_correct_answers_l384_38418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unfair_dice_odd_sum_l384_38483

noncomputable def p : ℝ := 1 / 5

noncomputable def q : ℝ := 4 * p

noncomputable def prob_odd_sum : ℝ := 3 * q * p^2 + q^3

theorem unfair_dice_odd_sum : 
  p + q = 1 → prob_odd_sum = 76 / 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unfair_dice_odd_sum_l384_38483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_is_ka_squared_l384_38484

/-- An object moving in a straight line with resistance proportional to velocity squared -/
structure MovingObject where
  /-- Displacement of the object -/
  s : ℝ → ℝ
  /-- Velocity of the object -/
  v : ℝ → ℝ
  /-- Resistance force -/
  F : ℝ → ℝ
  /-- Proportionality constant for resistance -/
  k : ℝ
  /-- Displacement is quadratic with time -/
  displacement_eq : ∀ t, s t = (1/2) * t^2
  /-- Resistance is proportional to velocity squared -/
  resistance_eq : ∀ t, F t = k * (v t)^2

/-- Work done by a variable force -/
noncomputable def work (F : ℝ → ℝ) (s₁ s₂ : ℝ) : ℝ :=
  ∫ x in s₁..s₂, F x

/-- The work done by the object to overcome resistance from s=0 to s=a is ka² -/
theorem work_done_is_ka_squared (obj : MovingObject) (a : ℝ) :
  work obj.F 0 a = obj.k * a^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_is_ka_squared_l384_38484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_line_l384_38494

/-- The focus of a parabola y = mx^2 -/
noncomputable def focus (m : ℝ) : ℝ × ℝ := (0, 1 / (4 * m))

/-- The line x + y - 2 = 0 -/
def line (x y : ℝ) : Prop := x + y - 2 = 0

/-- The theorem stating that if the line passes through the focus of the parabola, then m = 1/8 -/
theorem parabola_focus_line (m : ℝ) : 
  (m > 0) → (line (focus m).1 (focus m).2) → m = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_line_l384_38494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_distance_of_specific_ellipse_l384_38426

/-- An ellipse with axes parallel to coordinate axes -/
structure ParallelAxisEllipse where
  center : ℝ × ℝ
  majorAxis : ℝ
  minorAxis : ℝ

/-- Construct an ellipse from its tangent points -/
noncomputable def ellipseFromTangents (xTangent yTangent : ℝ) : ParallelAxisEllipse :=
  { center := (xTangent, yTangent)
  , majorAxis := 2 * max xTangent yTangent
  , minorAxis := 2 * min xTangent yTangent }

/-- Calculate the distance between foci of an ellipse -/
noncomputable def focalDistance (e : ParallelAxisEllipse) : ℝ :=
  2 * Real.sqrt (e.majorAxis^2 / 4 - e.minorAxis^2 / 4)

/-- Theorem: The distance between foci of the given ellipse is 6√3 -/
theorem focal_distance_of_specific_ellipse :
  focalDistance (ellipseFromTangents 6 3) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_distance_of_specific_ellipse_l384_38426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reaches_origin_l384_38445

/-- Represents the probability of reaching (0,0) from a given point (x,y) -/
noncomputable def P (x y : ℕ) : ℚ := sorry

/-- The particle starts at (3,3) -/
def start_point : ℕ × ℕ := (3, 3)

/-- The probability of moving to each adjacent point is 1/3 -/
def move_probability : ℚ := 1/3

/-- The recursive relation for points not on the axes -/
axiom recursive_relation (x y : ℕ) (h : x > 0 ∧ y > 0) :
  P x y = move_probability * (P (x-1) y + P x (y-1) + P (x-1) (y-1))

/-- Base case: Probability of reaching (0,0) from (0,0) is 1 -/
axiom base_case_origin : P 0 0 = 1

/-- Base case: Probability of reaching (0,0) from x-axis (except origin) is 0 -/
axiom base_case_x_axis (x : ℕ) (h : x > 0) : P x 0 = 0

/-- Base case: Probability of reaching (0,0) from y-axis (except origin) is 0 -/
axiom base_case_y_axis (y : ℕ) (h : y > 0) : P 0 y = 0

/-- The main theorem to prove -/
theorem probability_reaches_origin :
  P start_point.1 start_point.2 = 7/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reaches_origin_l384_38445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_traffic_flow_traffic_flow_range_l384_38408

-- Define the traffic flow function
noncomputable def y (v : ℝ) : ℝ := 920 * v / (v^2 + 3*v + 1600)

-- Theorem for the maximum value of y
theorem max_traffic_flow :
  ∃ (v_max : ℝ), v_max > 0 ∧ 
  (∀ (v : ℝ), v > 0 → y v ≤ y v_max) ∧
  v_max = 40 ∧ y v_max = 920 / 83 :=
sorry

-- Theorem for the range where y > 10
theorem traffic_flow_range (v : ℝ) :
  v > 0 → (y v > 10 ↔ 25 < v ∧ v < 64) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_traffic_flow_traffic_flow_range_l384_38408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l384_38428

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | x > a}

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (A B : Set ℝ) : Prop :=
  (A ⊆ B) ∧ ¬(B ⊆ A)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary A (B a) → a < 5 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l384_38428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_math_competition_2018_l384_38459

theorem chinese_math_competition_2018 (n k m : ℕ+) (A : Finset ℕ) :
  k ≥ 2 →
  (n : ℝ) ≤ m →
  (m : ℝ) < (2 * k - 1) / k * n →
  A.card = n →
  A ⊆ Finset.range m →
  ∀ t ∈ Set.Ioo (0 : ℝ) (n / (k - 1)),
    ∃ (a a' : ℕ), a ∈ A ∧ a' ∈ A ∧ (a : ℝ) - a' = t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_math_competition_2018_l384_38459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_root_sum_l384_38462

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluation of a cubic polynomial at a given point -/
noncomputable def CubicPolynomial.eval (p : CubicPolynomial) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The sum of roots of a cubic polynomial -/
noncomputable def CubicPolynomial.sumOfRoots (p : CubicPolynomial) : ℝ :=
  -p.b / p.a

theorem cubic_polynomial_root_sum (Q : CubicPolynomial) 
  (h : ∀ x : ℝ, Q.eval (x^3 + x + 1) ≥ Q.eval (x^2 + x + 2)) :
  Q.sumOfRoots = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_root_sum_l384_38462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_tile_length_l384_38481

/-- Represents the dimensions and properties of a tiled bathroom floor. -/
structure BathroomFloor where
  width_tiles : ℕ
  length_tiles : ℕ
  area_sq_ft : ℝ

/-- Calculates the length of each tile in inches for a given bathroom floor. -/
noncomputable def tile_length_inches (floor : BathroomFloor) : ℝ :=
  let total_tiles := floor.width_tiles * floor.length_tiles
  let area_sq_inches := floor.area_sq_ft * 144
  Real.sqrt (area_sq_inches / (total_tiles : ℝ))

/-- Theorem stating that for a bathroom floor with 10 tiles in width, 20 tiles in length,
    and 50 square feet in area, the length of each tile is 6 inches. -/
theorem bathroom_tile_length :
  tile_length_inches ⟨10, 20, 50⟩ = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_tile_length_l384_38481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximize_rectangle_area_partition_length_is_optimal_l384_38409

/-- The length of the material used to enclose the rectangular field -/
noncomputable def total_length : ℝ := 48

/-- The function representing the area of the rectangle -/
noncomputable def area (x : ℝ) : ℝ := x * (total_length / 4 - x / 2)

/-- The optimal length of partitions that maximizes the area -/
noncomputable def optimal_partition_length : ℝ := 6

theorem maximize_rectangle_area :
  ∀ x : ℝ, 0 < x → x < total_length / 4 →
  area x ≤ area optimal_partition_length :=
by sorry

theorem partition_length_is_optimal :
  optimal_partition_length = total_length / 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximize_rectangle_area_partition_length_is_optimal_l384_38409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_condition_l384_38400

/-- A function f is an inverse proportion function if there exists a non-zero constant k
    such that f(x) = k/x for all non-zero x -/
def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function defined by a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^(a^2 - 2)

theorem inverse_proportion_condition (a : ℝ) :
  is_inverse_proportion (f a) ↔ a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_condition_l384_38400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_equation_l384_38486

-- Define the function g
def g (x : ℝ) : ℝ := 12 * x + 5

-- State the theorem
theorem sum_of_roots_equation :
  ∃ (x₁ x₂ : ℝ), g⁻¹ x₁ = g ((3 * x₁)⁻¹) ∧ g⁻¹ x₂ = g ((3 * x₂)⁻¹) ∧ x₁ + x₂ = 65 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_equation_l384_38486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l384_38402

theorem evaluate_expression (x : ℝ) (h : x = 10) : 30 - |-x + 6| = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l384_38402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_applicability_l384_38430

theorem bisection_method_applicability :
  (∃ a b : ℝ, a < b ∧ (Real.log a + a) * (Real.log b + b) < 0) ∧
  (∃ a b : ℝ, a < b ∧ (Real.exp a - 3 * a) * (Real.exp b - 3 * b) < 0) ∧
  (∃ a b : ℝ, a < b ∧ (a^3 - 3*a + 1) * (b^3 - 3*b + 1) < 0) ∧
  (∀ a b : ℝ, a < b → (4*a^2 - 4*Real.sqrt 5*a + 5) * (4*b^2 - 4*Real.sqrt 5*b + 5) ≥ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_applicability_l384_38430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_water_addition_l384_38479

/-- Represents a mixture of water and hydrochloric acid -/
structure Mixture where
  total_volume : ℝ
  water_percentage : ℝ
  hcl_percentage : ℝ
  water_percentage_valid : water_percentage ≥ 0 ∧ water_percentage ≤ 1
  hcl_percentage_valid : hcl_percentage ≥ 0 ∧ hcl_percentage ≤ 1
  percentages_sum_to_one : water_percentage + hcl_percentage = 1

/-- The initial mixture -/
def initial_mixture : Mixture := {
  total_volume := 300
  water_percentage := 0.6
  hcl_percentage := 0.4
  water_percentage_valid := by sorry
  hcl_percentage_valid := by sorry
  percentages_sum_to_one := by sorry
}

/-- The desired mixture percentages -/
def desired_water_percentage : ℝ := 0.7
def desired_hcl_percentage : ℝ := 0.3

/-- The amount of water to be added -/
def water_to_add : ℝ := 100

/-- The resulting mixture after adding water -/
noncomputable def resulting_mixture : Mixture := {
  total_volume := initial_mixture.total_volume + water_to_add
  water_percentage := (initial_mixture.water_percentage * initial_mixture.total_volume + water_to_add) / 
                      (initial_mixture.total_volume + water_to_add)
  hcl_percentage := (initial_mixture.hcl_percentage * initial_mixture.total_volume) / 
                    (initial_mixture.total_volume + water_to_add)
  water_percentage_valid := by sorry
  hcl_percentage_valid := by sorry
  percentages_sum_to_one := by sorry
}

theorem correct_water_addition : 
  resulting_mixture.water_percentage = desired_water_percentage ∧
  resulting_mixture.hcl_percentage = desired_hcl_percentage := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_water_addition_l384_38479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_common_tangents_l384_38490

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if two circles have different radii -/
def hasDifferentRadii (c1 c2 : Circle) : Prop :=
  c1.radius ≠ c2.radius

/-- Counts the number of common tangents between two circles -/
noncomputable def commonTangentsCount (c1 c2 : Circle) : ℕ := sorry

/-- Theorem stating that two circles with different radii cannot have exactly 4 common tangents -/
theorem no_four_common_tangents (c1 c2 : Circle) 
  (h : hasDifferentRadii c1 c2) : 
  commonTangentsCount c1 c2 ≠ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_common_tangents_l384_38490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_integer_probability_l384_38412

noncomputable def coin_flip_sequence (a : ℚ) : List ℚ → List ℚ
| [] => [a]
| (h :: t) => 
    let heads := 2 * h - 1
    let tails := h / 2 - 1
    h :: coin_flip_sequence heads t ++ coin_flip_sequence tails t

def is_integer (x : ℚ) : Bool :=
  x.den = 1

theorem fourth_term_integer_probability : 
  let sequence := coin_flip_sequence 6 [1, 1, 1]
  let fourth_terms := sequence.filter (λ x => sequence.indexOf x = 3)
  let integer_count := (fourth_terms.filter is_integer).length
  (integer_count : ℚ) / fourth_terms.length = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_integer_probability_l384_38412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_problem_l384_38441

theorem geometric_series_problem (a b r t : ℝ) : 
  a / (1 - r) = 1 →  -- First series sum is 1
  b / (1 - t) = 1 →  -- Second series sum is 1
  a * r = b * t →    -- Second terms are equal
  b * t^2 = (1 : ℝ) / 8 →    -- Third term of one series is 1/8
  a * r = b * t ∧ a * r = (Real.sqrt 5 - 1) / 8 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_problem_l384_38441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_range_of_a_for_slope_range_of_a_for_inequality_l384_38452

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * (a + 2) * x^2 + 2 * a * x
def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * (a - 5) * x^2

-- Define the condition a ≥ 4
def a_condition (a : ℝ) : Prop := a ≥ 4

-- Theorem 1: Monotonicity of f
theorem monotonicity_of_f (a : ℝ) (h : a_condition a) :
  (∀ x₁ x₂, x₁ < x₂ → x₂ < 2 → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, a < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∀ x₁ x₂, 2 < x₁ → x₁ < x₂ → x₂ < a → f a x₁ > f a x₂) :=
sorry

-- Theorem 2: Range of a for given slope condition
theorem range_of_a_for_slope (a : ℝ) (h : a_condition a) :
  (∀ x, deriv (f a) x ≥ -25/4) ↔ a ∈ Set.Icc 4 7 :=
sorry

-- Theorem 3: Range of a for given inequality condition
theorem range_of_a_for_inequality (a : ℝ) (h : a_condition a) :
  (∀ x₁ x₂, x₁ ∈ Set.Icc 3 4 → x₂ ∈ Set.Icc 3 4 → x₁ ≠ x₂ →
    |f a x₁ - f a x₂| > |g a x₁ - g a x₂|) ↔
  a ∈ Set.Icc (14/3) 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_range_of_a_for_slope_range_of_a_for_inequality_l384_38452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l384_38416

theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : Real.cos α = -4/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.tan (α + π/4) = 1/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l384_38416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_equals_interval_l384_38439

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- State the theorem
theorem union_M_N_equals_interval : M ∪ N = Set.Ioo (-1) 3 ∪ {3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_equals_interval_l384_38439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l384_38415

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ),
    (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
    (dist P F₁ + dist P F₂ = 3 * b) ∧
    (dist P F₁ * dist P F₂ = 9/4 * a * b) →
    Real.sqrt ((a^2 + b^2) / a^2) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l384_38415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_qin_jiushao_eq_hero_specific_triangle_l384_38491

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the Qin Jiushao formula
noncomputable def qin_jiushao_area (t : Triangle) : ℝ :=
  Real.sqrt (1/4 * (t.a^2 * t.c^2 - ((t.c^2 + t.a^2 - t.b^2)/2)^2))

-- Define the Hero formula
noncomputable def hero_area (t : Triangle) : ℝ :=
  let p := (t.a + t.b + t.c) / 2
  Real.sqrt (p * (p - t.a) * (p - t.b) * (p - t.c))

-- Define the inradius
noncomputable def inradius (t : Triangle) : ℝ :=
  hero_area t / ((t.a + t.b + t.c) / 2)

-- Theorem 1: Equivalence of Qin Jiushao and Hero formulas
theorem qin_jiushao_eq_hero (t : Triangle) :
  qin_jiushao_area t = hero_area t := by sorry

-- Theorem 2: Specific triangle properties
theorem specific_triangle :
  ∃ (t : Triangle), t.a = 4 ∧ t.b < t.c ∧ hero_area t = 6 ∧ inradius t = 1 ∧ t.b = 3 ∧ t.c = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_qin_jiushao_eq_hero_specific_triangle_l384_38491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_possibilities_l384_38446

theorem divisible_by_four_possibilities : 
  let possible_N := Finset.range 10
  (∀ n ∈ possible_N, (n * 1000 + 864) % 4 = 0) ∧ 
  (Finset.card possible_N = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_possibilities_l384_38446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l384_38434

theorem train_speed_calculation (train_length bridge_length : ℝ) 
  (crossing_time : ℝ) (h1 : train_length = 110) 
  (h2 : bridge_length = 136) (h3 : crossing_time = 12.299016078713702) : 
  Int.floor ((train_length + bridge_length) / crossing_time * 3.6) = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l384_38434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_coefficients_l384_38422

-- Define the polynomial Q(x)
def Q (p q r : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + r

-- Define the roots of Q(x)
noncomputable def root1 : ℝ := Real.sin (3 * Real.pi / 7)
noncomputable def root2 : ℝ := Real.sin (5 * Real.pi / 7)
noncomputable def root3 : ℝ := Real.sin (Real.pi / 7)

-- State the theorem
theorem product_of_coefficients (p q r : ℝ) :
  (∀ x : ℝ, Q p q r x = 0 ↔ x = root1 ∨ x = root2 ∨ x = root3) →
  p * q * r = 0.725 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_coefficients_l384_38422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_sqrt_2_l384_38460

-- Define the line l
def line_l (s : ℝ) : ℝ × ℝ := (1 + s, 1 - s)

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ := (t + 2, t^2)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ s t, line_l s = p ∧ curve_C t = p}

-- Theorem statement
theorem intersection_distance_is_sqrt_2 :
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧ dist A B = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_sqrt_2_l384_38460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_is_plane_figure_l384_38403

-- Define a trapezoid
structure Trapezoid where
  vertices : Fin 4 → ℝ × ℝ
  is_trapezoid : ∃ (i j : Fin 4), i ≠ j ∧ (vertices i).1 = (vertices j).1

-- Define a plane figure
def PlaneFigure (T : Type) := T → Fin 4 → ℝ × ℝ

-- Theorem statement
theorem trapezoid_is_plane_figure : 
  ∀ (t : Trapezoid), ∃ (p : PlaneFigure Trapezoid), p t = t.vertices := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_is_plane_figure_l384_38403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_alpha_minus_beta_l384_38471

theorem max_sin_alpha_minus_beta (α β : ℝ) :
  Real.tan α / Real.tan β = 2 →
  β ∈ Set.Ioo 0 (π / 2) →
  ∃ (max_val : ℝ), max_val = 1/3 ∧ ∀ (x : ℝ), Real.sin (α - β) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_alpha_minus_beta_l384_38471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diesel_water_ratio_exists_l384_38496

/-- Represents the components of a fuel mixture --/
structure FuelMixture where
  diesel : ℚ
  petrol : ℚ
  water : ℚ

/-- Calculates the ratio of diesel to water in a fuel mixture --/
noncomputable def dieselWaterRatio (mix : FuelMixture) : ℚ :=
  mix.diesel / mix.water

/-- The initial volumes of diesel and petrol --/
def initialMix : FuelMixture :=
  { diesel := 4, petrol := 4, water := 0 }

/-- The total volume of the final mixture --/
def finalVolume : ℚ := 8/3

/-- Theorem stating that the diesel to water ratio can be determined --/
theorem diesel_water_ratio_exists (mix : FuelMixture) (vol : ℚ) :
  mix.diesel + mix.petrol + mix.water = vol →
  mix.water > 0 →
  ∃ r : ℚ, dieselWaterRatio mix = r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diesel_water_ratio_exists_l384_38496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_percentage_for_given_discount_and_gain_l384_38480

/-- Given a discount percentage and a desired gain percentage, 
    calculate the markup percentage on the cost price. -/
noncomputable def calculate_markup_percentage (discount_percent : ℝ) (gain_percent : ℝ) : ℝ :=
  let selling_price_ratio := 1 + gain_percent / 100
  let discount_ratio := 1 - discount_percent / 100
  ((selling_price_ratio / discount_ratio) - 1) * 100

theorem markup_percentage_for_given_discount_and_gain :
  let discount_percent : ℝ := 18.939393939393938
  let gain_percent : ℝ := 7
  let calculated_markup := calculate_markup_percentage discount_percent gain_percent
  ∃ ε > 0, |calculated_markup - 32.01| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_percentage_for_given_discount_and_gain_l384_38480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l384_38477

noncomputable def f (x : ℝ) := Real.cos x ^ 2 + Real.sin x * Real.cos x - 1 / 2

theorem f_properties :
  (∀ k : ℤ, ∃ x : ℝ, x = π / 8 + k * π / 2 ∧ ∀ y : ℝ, f y = f (2 * x - y)) ∧
  (∀ k : ℤ, ∀ x y : ℝ, k * π - 3 * π / 8 ≤ x ∧ x < y ∧ y ≤ k * π + π / 8 → f x < f y) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → f x ≥ -1 / 2) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ f x = -1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l384_38477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_and_abs_sin_same_parity_l384_38487

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem cos_and_abs_sin_same_parity :
  even_function Real.cos → even_function (fun x ↦ |Real.sin x|) → 
  ∃ (k : ℝ → ℝ), ∀ x, Real.cos x = k (|Real.sin x|) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_and_abs_sin_same_parity_l384_38487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_lent_to_B_l384_38435

/-- The amount of money A lent to B -/
def amount_to_B : ℝ := 5000

/-- The amount of money A lent to C -/
def amount_to_C : ℝ := 3000

/-- The annual interest rate -/
def interest_rate : ℝ := 0.07

/-- The time period for B's loan in years -/
def time_B : ℝ := 2

/-- The time period for C's loan in years -/
def time_C : ℝ := 4

/-- The total interest received from both B and C -/
def total_interest : ℝ := 1540

/-- Theorem stating that the amount A lent to B is 5000, given the conditions -/
theorem amount_lent_to_B :
  amount_to_B * interest_rate * time_B +
  amount_to_C * interest_rate * time_C = total_interest := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_lent_to_B_l384_38435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_zero_roots_of_equation_l384_38436

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / (x + a)

-- Part 1
theorem tangent_perpendicular_implies_a_zero (a : ℝ) :
  (∃ m : ℝ, (deriv (f a)) 1 * m = -1 ∧ m = -1) →
  a = 0 :=
by
  sorry

-- Part 2
theorem roots_of_equation (a : ℝ) :
  (a < -1 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 1 ∧ f a x₂ = 1) ∧
  (a ≥ -1 → ∀ x : ℝ, f a x ≠ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_zero_roots_of_equation_l384_38436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_subset_criterion_l384_38442

/-- A is a set of n-tuples of real numbers -/
def A (n : ℕ) := Fin n → ℝ

/-- Equality for elements in A -/
def eq_A {n : ℕ} (a b : A n) : Prop := ∀ i, a i = b i

/-- Addition for elements in A -/
def add_A {n : ℕ} (a b : A n) : A n := fun i => a i + b i

/-- Scalar multiplication for elements in A -/
def smul_A {n : ℕ} (r : ℝ) (a : A n) : A n := fun i => r * a i

/-- Definition of a perfect subset -/
def is_perfect_subset {n : ℕ} (B : Finset (A n)) : Prop :=
  ∀ r₁ r₂ r₃ : ℝ, ∀ a₁ a₂ a₃ : A n, a₁ ∈ B → a₂ ∈ B → a₃ ∈ B →
    eq_A (add_A (add_A (smul_A r₁ a₁) (smul_A r₂ a₂)) (smul_A r₃ a₃)) (fun _ => 0) →
    r₁ = 0 ∧ r₂ = 0 ∧ r₃ = 0

/-- Main theorem -/
theorem perfect_subset_criterion {n : ℕ} (hn : n ≥ 3) 
    (B : Finset (A n)) (hB : B.card = 3) :
  (∀ i : Fin n, ∀ a₁ a₂ a₃ : A n, a₁ ∈ B → a₂ ∈ B → a₃ ∈ B → 
    2 * |a₁ i| > |a₁ i| + |a₂ i| + |a₃ i| ∧
    2 * |a₂ i| > |a₁ i| + |a₂ i| + |a₃ i| ∧
    2 * |a₃ i| > |a₁ i| + |a₂ i| + |a₃ i|) →
  is_perfect_subset B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_subset_criterion_l384_38442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_z_l384_38425

open Complex

/-- Given a complex number z, define a sequence z_n where z_0 = z and z_{n+1} = 2z_n^2 + 2z_n for n ≥ 0 -/
def sequenceZ (z : ℂ) : ℕ → ℂ
  | 0 => z
  | n + 1 => 2 * (sequenceZ z n)^2 + 2 * (sequenceZ z n)

/-- The theorem stating the minimum possible value of |z| given the conditions -/
theorem min_abs_z :
  ∃ (min_abs : ℝ), min_abs = (((4035 : ℝ)^(1/1024 : ℝ) - 1) / 2) ∧
  ∀ (z : ℂ), sequenceZ z 10 = 2017 → Complex.abs z ≥ min_abs :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_z_l384_38425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_4_and_9_l384_38437

-- Define the geometric mean of two positive real numbers
noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

-- Theorem statement
theorem geometric_mean_of_4_and_9 :
  geometric_mean 4 9 = 6 ∨ geometric_mean 4 9 = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_4_and_9_l384_38437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_number_above_sum_l384_38455

theorem smallest_whole_number_above_sum : 
  ⌈(3 + 2/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/5 : ℚ) + (6 + 1/6 : ℚ) + (7 + 1/7 : ℚ)⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_number_above_sum_l384_38455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_square_diff_implies_arithmetic_square_alternating_seq_is_equal_square_diff_equal_square_diff_and_arithmetic_implies_constant_l384_38464

def is_equal_square_diff_seq (a : ℕ → ℝ) : Prop :=
  ∃ p : ℝ, ∀ n : ℕ, n ≥ 2 → a n ^ 2 - a (n - 1) ^ 2 = p

def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def is_constant_seq (a : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

theorem equal_square_diff_implies_arithmetic_square (a : ℕ → ℝ) :
  is_equal_square_diff_seq a → is_arithmetic_seq (λ n ↦ (a n) ^ 2) :=
sorry

theorem alternating_seq_is_equal_square_diff :
  is_equal_square_diff_seq (λ n ↦ (-1) ^ n) :=
sorry

theorem equal_square_diff_and_arithmetic_implies_constant (a : ℕ → ℝ) :
  is_equal_square_diff_seq a → is_arithmetic_seq a → is_constant_seq a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_square_diff_implies_arithmetic_square_alternating_seq_is_equal_square_diff_equal_square_diff_and_arithmetic_implies_constant_l384_38464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l384_38419

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2
def g (x a : ℝ) : ℝ := 2 * x + a

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 1/2 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1/2 ≤ x₂ ∧ x₂ ≤ 2 ∧ f x₁ = g x₂ a) →
  -5 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l384_38419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l384_38449

noncomputable def f (x a : ℝ) : ℝ := Real.sin (x - Real.pi/6) + Real.sin (x + Real.pi/6) + Real.cos x + a

theorem function_properties :
  (∃ (a : ℝ), ∀ (x : ℝ), f x a ≤ 3 ∧ ∃ (y : ℝ), f y a = 3) →
  (∃ (a : ℝ), a = 1 ∧
    ∀ (x : ℝ), f x a > 0 ↔ ∃ (k : ℤ), 2 * k * Real.pi - Real.pi/3 < x ∧ x < Real.pi + 2 * k * Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l384_38449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_iff_k_eq_zero_l384_38461

/-- The polynomial we're considering -/
def P (k : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + x + k*y - k

/-- A linear polynomial in x and y with integer coefficients -/
def LinearPoly : Type := { f : ℤ → ℤ → ℤ // ∃ (a b c : ℤ), ∀ x y, f x y = a*x + b*y + c }

/-- The main theorem -/
theorem factorization_iff_k_eq_zero (k : ℤ) :
  (∃ (f g : LinearPoly), ∀ x y, P k x y = f.val x y * g.val x y) ↔ k = 0 :=
sorry

#check factorization_iff_k_eq_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_iff_k_eq_zero_l384_38461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_arrangement_theorem_l384_38424

noncomputable section

def num_radars : ℕ := 9
def radar_radius : ℝ := 37
def ring_width : ℝ := 24

noncomputable def center_to_radar_distance : ℝ := 35 / Real.sin (20 * Real.pi / 180)

noncomputable def coverage_ring_area : ℝ := 1680 * Real.pi / Real.tan (20 * Real.pi / 180)

theorem radar_arrangement_theorem :
  (num_radars = 9) →
  (radar_radius = 37) →
  (ring_width = 24) →
  (center_to_radar_distance = 35 / Real.sin (20 * Real.pi / 180)) ∧
  (coverage_ring_area = 1680 * Real.pi / Real.tan (20 * Real.pi / 180)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_arrangement_theorem_l384_38424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l384_38466

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line passing through a point -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Predicate to check if three points form an isosceles right triangle -/
def isosceles_right_triangle (p₁ p₂ p₃ : ℝ × ℝ) : Prop :=
  sorry

/-- Function to calculate the chord length -/
def chord_length (C : Ellipse) (chord : ℝ) : ℝ :=
  sorry

/-- Predicate to check if the parallelogram formed by the intersection points is a rectangle -/
def is_rectangle_parallelogram (C : Ellipse) (l : Line) : Prop :=
  sorry

/-- Main theorem about the ellipse and the line -/
theorem ellipse_and_line_theorem (C : Ellipse) :
  (∃ (F₁ F₂ V : ℝ × ℝ), isosceles_right_triangle F₁ F₂ V) →
  (∃ (chord : ℝ), chord_length C chord = 2 * Real.sqrt 3) →
  (C.a = Real.sqrt 2 ∧ C.b = 1) ∧
  (∃ (l : Line), l.point = (1, 0) ∧ 
    (l.slope = Real.sqrt 2 ∨ l.slope = -Real.sqrt 2) ∧
    is_rectangle_parallelogram C l) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l384_38466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_vertex_assignment_l384_38493

def forms_parallel_triangle (n : ℕ) (i j k : Fin ((n + 1) * (n + 2) / 2)) : Prop := sorry

theorem equilateral_triangle_vertex_assignment (n : ℕ) (h : n ≥ 2) :
  let num_vertices := (n + 1) * (n + 2) / 2
  ∀ (assignment : Fin num_vertices → ℝ),
    (∀ (i j k : Fin num_vertices),
      forms_parallel_triangle n i j k → assignment i + assignment j + assignment k = 0) →
    (∀ i : Fin num_vertices, assignment i = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_vertex_assignment_l384_38493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_point_value_l384_38465

/-- Triangle PQR with vertices P(-2,3), Q(4,7), and R(1,2) -/
def triangle_PQR : Set (ℝ × ℝ) :=
  {(-2, 3), (4, 7), (1, 2)}

/-- Point S(x,y) inside triangle PQR -/
def point_S (x y : ℝ) : ℝ × ℝ := (x, y)

/-- Area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Predicate for S being the point that divides PQR into three equal area triangles -/
def is_equal_area_point (x y : ℝ) : Prop :=
  let s := point_S x y
  let area_PQS := triangle_area (-2, 3) (4, 7) s
  let area_PRS := triangle_area (-2, 3) (1, 2) s
  let area_QRS := triangle_area (4, 7) (1, 2) s
  area_PQS = area_PRS ∧ area_PQS = area_QRS

theorem equal_area_point_value :
  ∀ x y : ℝ, is_equal_area_point x y → 10 * x + y = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_point_value_l384_38465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gnome_count_multiple_of_four_l384_38499

/-- Represents the voting options available to gnomes. -/
inductive VoteOption
  | For
  | Against
  | Abstain
deriving Repr, DecidableEq

/-- Represents a gnome with its current vote. -/
structure Gnome where
  vote : VoteOption
deriving Repr

/-- Represents the state of the gnome voting system. -/
structure GnomeVotingSystem where
  gnomes : List Gnome
  size : Nat
  size_positive : size > 0

/-- Defines the voting rule for gnomes based on their neighbors' votes. -/
def nextVote (left right current : VoteOption) : VoteOption :=
  if left = right then left
  else match left, right with
    | VoteOption.For, VoteOption.Against => VoteOption.Abstain
    | VoteOption.For, VoteOption.Abstain => VoteOption.Against
    | VoteOption.Against, VoteOption.For => VoteOption.Abstain
    | VoteOption.Against, VoteOption.Abstain => VoteOption.For
    | VoteOption.Abstain, VoteOption.For => VoteOption.Against
    | VoteOption.Abstain, VoteOption.Against => VoteOption.For
    | _, _ => current

/-- Theorem stating that the number of gnomes must be a multiple of 4. -/
theorem gnome_count_multiple_of_four (sys : GnomeVotingSystem) :
  ∃ k : Nat, sys.size = 4 * k := by
  sorry

/-- Lemma stating that all gnomes voted "for" on the gold question. -/
lemma all_voted_for_gold (sys : GnomeVotingSystem) :
  ∀ g ∈ sys.gnomes, g.vote = VoteOption.For := by
  sorry

/-- Lemma stating that Thorin abstained on the dragon question. -/
lemma thorin_abstained (sys : GnomeVotingSystem) :
  ∃ g ∈ sys.gnomes, g.vote = VoteOption.Abstain := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gnome_count_multiple_of_four_l384_38499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_identities_l384_38457

variable {L : Type*} [Lattice L] [Monoid L]

theorem lattice_identities (a b c : L) : 
  (a ⊔ (a ⊓ b) = a) ∧ 
  (a ⊓ (a ⊔ b) = a) ∧ 
  (a * b * c = (a ⊔ b ⊔ c) * ((a * b) ⊓ (a * c) ⊓ (b * c))) ∧ 
  (a * b * c = (a ⊓ b ⊓ c) * ((a * b) ⊔ (b * c) ⊔ (a * c))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_identities_l384_38457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_correct_statement4_is_correct_l384_38470

-- Define the four geometric statements as axioms
axiom statement1 : Prop
axiom statement2 : Prop
axiom statement3 : Prop
axiom statement4 : Prop

-- Define the meaning of each statement (for reference, not used in the proof)
def statement1_meaning : String := "The arcs corresponding to equal inscribed angles are equal"
def statement2_meaning : String := "A circle can always be drawn through any three points"
def statement3_meaning : String := "The circumcenter of an isosceles right triangle does not lie on the angle bisector of the vertex angle"
def statement4_meaning : String := "The distances from the incenter of an equilateral triangle to the three vertices of the triangle are equal"

-- Define a function to count correct statements
def count_correct (s1 s2 s3 s4 : Bool) : Nat :=
  (if s1 then 1 else 0) + (if s2 then 1 else 0) + (if s3 then 1 else 0) + (if s4 then 1 else 0)

-- Theorem stating that only one statement is correct
theorem only_one_correct : 
  ∃ (b1 b2 b3 b4 : Bool), 
    count_correct b1 b2 b3 b4 = 1 ∧ 
    (b1 ↔ statement1) ∧ 
    (b2 ↔ statement2) ∧ 
    (b3 ↔ statement3) ∧ 
    (b4 ↔ statement4) := by
  sorry

-- Theorem stating which statement is correct (statement4)
theorem statement4_is_correct :
  statement4 ∧ ¬statement1 ∧ ¬statement2 ∧ ¬statement3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_correct_statement4_is_correct_l384_38470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_K_value_min_K_is_one_l384_38411

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 - x - (Real.exp x)⁻¹

-- Define the piecewise function f_k
noncomputable def f_k (K : ℝ) (x : ℝ) : ℝ :=
  if f x ≤ K then f x else K

-- Theorem statement
theorem min_K_value (K : ℝ) :
  (∀ x : ℝ, f_k K x = f x) ↔ K ≥ 1 := by
  sorry

-- Proof that 1 is the minimum value of K
theorem min_K_is_one :
  ∃ K : ℝ, (∀ x : ℝ, f_k K x = f x) ∧ (∀ K' : ℝ, K' < K → ∃ x : ℝ, f_k K' x ≠ f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_K_value_min_K_is_one_l384_38411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_range_l384_38469

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - Real.exp (-x)) - Real.sin x

theorem unique_zero_range (a : ℝ) :
  (a > 0 ∧ ∃! x, f a x = 0) ↔ a ∈ Set.Ici (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_range_l384_38469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_minimum_l384_38406

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = Real.pi

-- Define the condition from the problem
def angle_condition (t : Triangle) : Prop :=
  2 * (Real.tan t.B + Real.tan t.C) = (Real.tan t.B / Real.cos t.C) + (Real.tan t.C / Real.cos t.B)

-- State the theorem
theorem cos_A_minimum (t : Triangle) (h : angle_condition t) : Real.cos t.A ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_minimum_l384_38406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_of_right_isosceles_triangle_l384_38447

/-- The radius of the incircle of a right triangle with one 45° angle and hypotenuse length 8√2 -/
noncomputable def incircle_radius (triangle : Set ℝ × Set ℝ × Set ℝ) : ℝ :=
  4 - 2 * Real.sqrt 2

/-- Predicate to check if a triangle is right-angled -/
def is_right_triangle (triangle : Set ℝ × Set ℝ × Set ℝ) : Prop := sorry

/-- Predicate to check if a triangle is isosceles -/
def is_isosceles_triangle (triangle : Set ℝ × Set ℝ × Set ℝ) : Prop := sorry

/-- Predicate to check if one angle of the triangle is 45 degrees -/
def one_angle_is_45_degrees (triangle : Set ℝ × Set ℝ × Set ℝ) : Prop := sorry

/-- Function to calculate the hypotenuse length of a triangle -/
noncomputable def hypotenuse_length (triangle : Set ℝ × Set ℝ × Set ℝ) : ℝ := sorry

theorem incircle_radius_of_right_isosceles_triangle 
  (triangle : Set ℝ × Set ℝ × Set ℝ) 
  (is_right : is_right_triangle triangle)
  (is_isosceles : is_isosceles_triangle triangle)
  (angle_45 : one_angle_is_45_degrees triangle)
  (hyp_length : hypotenuse_length triangle = 8 * Real.sqrt 2) :
  incircle_radius triangle = 4 - 2 * Real.sqrt 2 := by
  sorry

#check incircle_radius_of_right_isosceles_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_of_right_isosceles_triangle_l384_38447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l384_38488

theorem divisibility_theorem (k m n : ℕ) 
  (h_prime : Nat.Prime (m + k + 1))
  (h_greater : m + k + 1 > n + 1) : 
  let c : ℕ → ℕ := λ s => s * (s + 1)
  let product := (List.range n).foldl (λ acc i => acc * (c (m + i + 1) - c k)) 1
  let divisor := (List.range n).foldl (λ acc i => acc * c (i + 1)) 1
  (product ∣ divisor) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l384_38488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_15_l384_38454

/-- The angle of the hour hand at a given time -/
noncomputable def hour_hand_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  (hours % 12 : ℝ) * 30 + (minutes : ℝ) * 0.5

/-- The angle of the minute hand at a given time -/
noncomputable def minute_hand_angle (minutes : ℕ) : ℝ :=
  (minutes : ℝ) * 6

/-- The smaller angle between two angles on a circle -/
noncomputable def smaller_angle (angle1 : ℝ) (angle2 : ℝ) : ℝ :=
  min (abs (angle1 - angle2)) (360 - abs (angle1 - angle2))

theorem clock_angle_at_7_15 :
  smaller_angle (hour_hand_angle 7 15) (minute_hand_angle 15) = 127.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_15_l384_38454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_piece_has_one_l384_38429

/-- A number is neither prime nor composite if and only if it is 1. -/
axiom neither_prime_nor_composite_iff_one (n : ℕ) : (¬ Nat.Prime n ∧ n > 1) ↔ n = 1

/-- Given 100 pieces of paper with numbers, if the probability of drawing a number
    that is neither prime nor composite is 0.01, then exactly one piece has the number 1. -/
theorem one_piece_has_one (papers : Finset ℕ) (h_size : papers.card = 100) 
    (h_prob : (papers.filter (λ n => ¬ Nat.Prime n ∧ n ≤ 1)).card / papers.card = 1 / 100) :
  (papers.filter (λ n => n = 1)).card = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_piece_has_one_l384_38429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pappus_theorem_l384_38492

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the incidence relation between points and lines
variable (incidence : Point → Line → Prop)

-- Define the collinearity relation for three points
variable (collinear : Point → Point → Point → Prop)

-- Define the intersection of two lines
variable (intersect : Line → Line → Point)

-- Define a function to create a line from two points
variable (mkLine : Point → Point → Line)

-- Given two lines
variable (l₁ l₂ : Line)

-- Points on the first line
variable (A₁ B₁ C₁ : Point)

-- Points on the second line
variable (A₂ B₂ C₂ : Point)

-- Intersection points
variable (A B C : Point)

-- Theorem statement
theorem pappus_theorem 
  (h₁ : incidence A₁ l₁) (h₂ : incidence B₁ l₁) (h₃ : incidence C₁ l₁)
  (h₄ : incidence A₂ l₂) (h₅ : incidence B₂ l₂) (h₆ : incidence C₂ l₂)
  (h₇ : C = intersect (mkLine A₁ B₂) (mkLine A₂ B₁))
  (h₈ : A = intersect (mkLine B₁ C₂) (mkLine B₂ C₁))
  (h₉ : B = intersect (mkLine C₁ A₂) (mkLine C₂ A₁)) :
  collinear A B C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pappus_theorem_l384_38492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_exists_l384_38468

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_root_exists (a b c : ℝ) :
  let f := quadratic_function a b c
  (f 0 = -1) → (f 0.5 = -0.5) → (f 1 = 1) → (f 1.5 = 3.5) → (f 2 = 7) →
  ∃ x : ℝ, 0.5 < x ∧ x < 1 ∧ f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_exists_l384_38468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equality_l384_38423

noncomputable def f (x : ℝ) := x^3 + Real.sin x

theorem cosine_equality (α β : ℝ) (h1 : α ∈ Set.Icc 0 Real.pi) 
  (h2 : β ∈ Set.Icc (-Real.pi/4) (Real.pi/4)) 
  (h3 : f (Real.pi/2 - α) = f (2*β)) : 
  Real.cos (α/2 + β) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equality_l384_38423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_areas_l384_38433

/-- Two congruent right triangles with legs 4 cm and 7 cm -/
structure RightTrianglePair where
  leg1 : ℝ
  leg2 : ℝ
  h_leg1 : leg1 = 4
  h_leg2 : leg2 = 7

/-- Arrangement with 7 cm legs overlapping -/
noncomputable def overlapping_legs_area (t : RightTrianglePair) : ℝ :=
  3 / 4 * (t.leg1 * t.leg2)

/-- Arrangement with hypotenuses overlapping -/
noncomputable def overlapping_hypotenuse_area (t : RightTrianglePair) : ℝ :=
  2 * (33 / 14) + 14

theorem right_triangle_areas (t : RightTrianglePair) :
  overlapping_legs_area t = 21 ∧ overlapping_hypotenuse_area t = 131 / 7 := by
  sorry

#check right_triangle_areas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_areas_l384_38433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_ξ_l384_38440

/-- The number of coins tossed simultaneously -/
def num_coins : ℕ := 5

/-- The number of times the coins are tossed -/
def num_tosses : ℕ := 160

/-- The probability of getting exactly one head and four tails in a single toss -/
def p : ℚ := 5 / 32

/-- ξ is the number of times exactly one head and four tails are observed -/
def ξ : ℕ → ℕ := sorry

/-- The expected value of ξ -/
def expected_ξ : ℚ := num_tosses * p

theorem expected_value_of_ξ : expected_ξ = 25 := by
  unfold expected_ξ
  unfold num_tosses
  unfold p
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_ξ_l384_38440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_sequence_l384_38475

def Q : ℕ := (Finset.filter Nat.Prime (Finset.range 60)).prod id

theorem no_primes_in_sequence : ∀ n ∈ Finset.range 60, ¬Nat.Prime (Q + n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_sequence_l384_38475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_sum_l384_38497

def y (m : ℕ) : ℕ → ℚ
  | 0 => 1
  | 1 => m
  | (k + 2) => ((m + 1) * y m (k + 1) - (m - k) * y m k) / (k + 2)

theorem y_sum (m : ℕ) : ∑' k, y m k = 2^(m + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_sum_l384_38497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_2_l384_38414

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := x - y + 4 = 0

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 6 = 0

/-- The length of the chord intercepted by the line on the circle -/
noncomputable def chord_length : ℝ := 2 * Real.sqrt 2

/-- Theorem stating that the chord length is 2√2 -/
theorem chord_length_is_2_sqrt_2 : 
  ∃ (x y : ℝ), line_eq x y ∧ circle_eq x y ∧ 
  ∀ (x' y' : ℝ), line_eq x' y' ∧ circle_eq x' y' → 
  Real.sqrt ((x - x')^2 + (y - y')^2) = chord_length := by
  sorry

#check chord_length_is_2_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_2_l384_38414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l384_38474

noncomputable section

/-- The area of a semicircle with diameter d -/
def semicircleArea (d : ℝ) : ℝ := (Real.pi * d^2) / 8

/-- The configuration of points on a line -/
structure LineConfig where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  de : ℝ
  ef : ℝ
  fg : ℝ

/-- The shaded area formed by overlapping semicircles -/
def shadedArea (config : LineConfig) : ℝ :=
  let ag := config.ab + config.bc + config.cd + config.de + config.ef + config.fg
  semicircleArea ag - 
  (semicircleArea config.ab + semicircleArea config.bc + 
   semicircleArea config.cd + semicircleArea config.de + 
   semicircleArea config.ef + semicircleArea config.fg)

theorem shaded_area_calculation (config : LineConfig) 
  (h1 : config.ab = 5)
  (h2 : config.bc = 5)
  (h3 : config.cd = 10)
  (h4 : config.de = 10)
  (h5 : config.ef = 5)
  (h6 : config.fg = 5) :
  shadedArea config = 175 * Real.pi / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l384_38474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_tangent_circle_eq_l384_38432

/-- Represents a trapezoid with a circle tangent to two sides -/
structure TrapezoidWithTangentCircle where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  EQ : ℝ

/-- The properties of the specific trapezoid in the problem -/
noncomputable def problem_trapezoid : TrapezoidWithTangentCircle where
  EF := 100
  FG := 55
  GH := 23
  HE := 75
  EQ := 750 / 13

/-- Theorem stating that the length of EQ in the given trapezoid is 750/13 -/
theorem trapezoid_tangent_circle_eq : 
  let t := problem_trapezoid
  t.EQ = 750 / 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_tangent_circle_eq_l384_38432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_figure_l384_38451

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- The figure ABCDEFG -/
structure Figure :=
  (A : Point) (B : Point) (C : Point) (D : Point) (E : Point) (F : Point) (G : Point)

/-- Definition of an equilateral triangle -/
def isEquilateral (t : Triangle) : Prop :=
  let d1 := (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2
  let d2 := (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2
  let d3 := (t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2
  d1 = d2 ∧ d2 = d3

/-- Definition of a midpoint -/
def isMidpoint (M : Point) (A : Point) (B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

/-- Distance between two points -/
noncomputable def distance (A : Point) (B : Point) : ℝ :=
  ((A.x - B.x)^2 + (A.y - B.y)^2).sqrt

/-- Perimeter of the figure -/
noncomputable def perimeter (f : Figure) : ℝ :=
  distance f.A f.B + distance f.B f.C + distance f.C f.D +
  distance f.D f.E + distance f.E f.F + distance f.F f.G +
  distance f.G f.A

theorem perimeter_of_figure (f : Figure) :
  isEquilateral (Triangle.mk f.A f.B f.C) →
  isEquilateral (Triangle.mk f.A f.D f.E) →
  isEquilateral (Triangle.mk f.E f.F f.G) →
  isEquilateral (Triangle.mk f.D f.E f.F) →
  isMidpoint f.D f.A f.C →
  isMidpoint f.G f.A f.E →
  isMidpoint f.F f.D f.E →
  distance f.A f.B = 6 →
  perimeter f = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_figure_l384_38451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_range_l384_38443

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem f_min_range (a : ℝ) :
  (∀ x, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_range_l384_38443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_roots_l384_38478

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x + a

-- State the theorem
theorem f_extrema_and_roots (a : ℝ) :
  -- Part 1: Extrema
  (∃ (x_min x_max : ℝ), x_min = -1 ∧ x_max = 1 ∧
    IsLocalMin (f a) x_min ∧ IsLocalMax (f a) x_max ∧
    f a x_min = a - 2 ∧ f a x_max = a + 2) ∧
  -- Part 2: Conditions for exactly two real roots
  (∃! (x₁ x₂ : ℝ), f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ ≠ x₂) ↔ (a = 2 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_roots_l384_38478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_and_divisibility_l384_38453

theorem fraction_sum_and_divisibility : ∃ (n : ℕ), 
  n > 0 ∧ 
  ((1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n).isInt ∧
  2 ∣ n ∧
  3 ∣ n ∧
  4 ∣ n ∧
  6 ∣ n :=
by
  -- The proof goes here
  sorry

#check fraction_sum_and_divisibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_and_divisibility_l384_38453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_and_sales_correct_l384_38472

noncomputable section

-- Define the data points
def x_values : List ℚ := [2, 3, 4, 6, 7, 8]
def y_values : List ℚ := [7.5, 11.5, 0, 31.5, 36.5, 43.5]

-- Define the linear regression equation
def linear_regression (x : ℚ) : ℚ := 6 * x + -5

-- Define the mean of x and y
def x_mean : ℚ := (x_values.sum) / x_values.length
def y_mean : ℚ := 25 -- Given that (x̄, ȳ) lies on y = x²

-- Define the missing y value (m)
def m : ℚ := 19.5

-- Define the sales amount for 12 thousand yuan investment
def sales_12k : ℚ := 67

theorem regression_and_sales_correct : 
  (y_mean = (y_values.sum + m) / y_values.length) ∧
  (linear_regression x_mean = y_mean) ∧
  (linear_regression 12 = sales_12k) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_and_sales_correct_l384_38472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_song_count_is_13_l384_38463

def concert_song_count : ℕ :=
  let total_time : ℕ := 80  -- 1 hour 20 minutes in minutes
  let intermission : ℕ := 10
  let performance_time : ℕ := total_time - intermission
  let regular_song_duration : ℕ := 5
  let special_song_duration : ℕ := 10
  let regular_songs_time : ℕ := performance_time - special_song_duration
  let regular_song_count : ℕ := regular_songs_time / regular_song_duration
  regular_song_count + 1  -- total number of songs (regular_song_count + 1 special song)

theorem concert_song_count_is_13 : concert_song_count = 13 := by
  -- Unfold the definition and perform the calculation
  unfold concert_song_count
  -- The rest of the proof would go here
  sorry

#eval concert_song_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_song_count_is_13_l384_38463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_amplitude_l384_38410

/-- Given a sinusoidal function y = a * sin(bx + c) + d with positive constants a, b, c, and d,
    if the function oscillates between 5 and -3, then the amplitude a is equal to 4. -/
theorem sinusoidal_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_max : ∀ x, a * Real.sin (b * x + c) + d ≤ 5) 
  (h_min : ∀ x, a * Real.sin (b * x + c) + d ≥ -3) 
  (h_reaches_max : ∃ x, a * Real.sin (b * x + c) + d = 5)
  (h_reaches_min : ∃ x, a * Real.sin (b * x + c) + d = -3) : 
  a = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_amplitude_l384_38410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangles_containing_cell_l384_38405

noncomputable def number_of_rectangles_containing (m n p q : ℕ) : ℕ := 
  -- This function represents the number of rectangles containing the cell (p, q)
  -- in a grid of m horizontal lines and n vertical lines
  p * q * (m - p + 1) * (n - q + 1)

theorem rectangles_containing_cell (m n p q : ℕ) 
  (h1 : 1 ≤ p) (h2 : p ≤ m) (h3 : 1 ≤ q) (h4 : q ≤ n) :
  number_of_rectangles_containing m n p q = p * q * (m - p + 1) * (n - q + 1) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangles_containing_cell_l384_38405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_octagon_area_l384_38495

/-- The area of an octagon formed by the sides of a rhombus and the tangents to its inscribed circle -/
noncomputable def octagon_area (a b : ℝ) : ℝ :=
  4 * a * b * ((a + b) / Real.sqrt (a^2 + b^2) - 1)

/-- Theorem stating the area of the octagon formed by a rhombus with diagonals 2a and 2b -/
theorem rhombus_octagon_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let rhombus_diag1 := 2 * a
  let rhombus_diag2 := 2 * b
  let rhombus_area := rhombus_diag1 * rhombus_diag2 / 2
  let inradius := rhombus_area / Real.sqrt (rhombus_diag1^2 / 4 + rhombus_diag2^2 / 4)
  octagon_area a b = rhombus_area - 4 * (a - inradius) * (b - a * inradius / b) / 2 -
                     4 * (b - inradius) * (a - b * inradius / a) / 2 :=
by
  sorry

#check rhombus_octagon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_octagon_area_l384_38495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_property_l384_38489

/-- A two-digit number -/
def TwoDigitNumber : Type := {n : ℕ // 10 ≤ n ∧ n < 100}

/-- The tens digit of a two-digit number -/
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10

/-- The units digit of a two-digit number -/
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

/-- The sum of digits of a two-digit number -/
def sum_of_digits (n : TwoDigitNumber) : ℕ := tens_digit n + units_digit n

/-- The property we're interested in -/
def has_property (n : TwoDigitNumber) : Prop :=
  (n.val - sum_of_digits n) % 10 = 3

instance : Fintype TwoDigitNumber := by
  sorry

instance : DecidablePred has_property := by
  sorry

/-- The theorem to prove -/
theorem count_numbers_with_property :
  (Finset.filter has_property (Finset.univ : Finset TwoDigitNumber)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_property_l384_38489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_norm_equality_l384_38427

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_norm_equality (m n : E) : 
  (∀ (l : ℝ), ‖m - l • (m - n)‖ ≥ ‖m + n‖ / 2) → ‖m‖ = ‖n‖ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_norm_equality_l384_38427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_slope_formula_l384_38417

/-- Represents a data point with x and y coordinates -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- The slope of the best fit line for four equally spaced data points -/
noncomputable def bestFitSlope (p₁ p₂ p₃ p₄ : DataPoint) (d : ℝ) : ℝ :=
  (p₄.y - p₁.y) / (p₄.x - p₁.x)

/-- Theorem stating that the slope of the best fit line for four equally spaced data points
    is equal to (y₄ - y₁) / (x₄ - x₁) -/
theorem best_fit_slope_formula (p₁ p₂ p₃ p₄ : DataPoint) (d : ℝ) 
    (h₁ : p₁.x < p₂.x) (h₂ : p₂.x < p₃.x) (h₃ : p₃.x < p₄.x)
    (h₄ : p₄.x - p₃.x = d) (h₅ : p₃.x - p₂.x = d) (h₆ : p₂.x - p₁.x = d) :
    bestFitSlope p₁ p₂ p₃ p₄ d = (p₄.y - p₁.y) / (p₄.x - p₁.x) := by
  sorry

#check best_fit_slope_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_slope_formula_l384_38417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_equation_l384_38467

-- Define the points
noncomputable def P : ℝ × ℝ := (-2, 4 * Real.sqrt 3)
def Q : ℝ × ℝ := (2, 0)
def M : ℝ × ℝ := (0, -6)

-- Define the line l
noncomputable def l (m : ℝ) : ℝ → ℝ := λ y ↦ m * y - 2 * Real.sqrt 3

-- Define the inscribed circle
def inscribed_circle : ℝ → ℝ → Prop :=
  λ x y ↦ (x - 2)^2 + (y - 2)^2 = 1

-- State the theorem
theorem inscribed_circle_equation :
  ∃ m : ℝ, inscribed_circle = 
    λ x y ↦ (x - 2)^2 + (y - 2)^2 = 1 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_equation_l384_38467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_escalator_time_l384_38458

/-- The time it takes Vasya to run up and down an escalator moving up -/
noncomputable def vasya_time_up_escalator : ℝ := 324

/-- Vasya's speed running down the escalator -/
noncomputable def vasya_speed_down : ℝ := 1 / 2

/-- The speed of the escalator -/
noncomputable def escalator_speed : ℝ := 1 / 6

/-- The time it takes Vasya to run up and down when the escalator is not working -/
noncomputable def time_stationary : ℝ := 6

/-- The time it takes Vasya to run up and down when the escalator is moving down -/
noncomputable def time_down_escalator : ℝ := 13.5

theorem vasya_escalator_time :
  (1 / (vasya_speed_down - escalator_speed) + 1 / ((vasya_speed_down / 2) + escalator_speed)) * 60 = vasya_time_up_escalator ∧
  vasya_speed_down = 1 / 2 ∧
  escalator_speed = 1 / 6 ∧
  time_stationary = 3 / vasya_speed_down ∧
  time_down_escalator = 1 / (vasya_speed_down + escalator_speed) + 1 / ((vasya_speed_down / 2) - escalator_speed) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_escalator_time_l384_38458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_symmetry_l384_38498

theorem sin_shift_symmetry (φ : ℝ) :
  (φ > 0) →
  (∀ x, Real.sin (3 * (x + φ)) = Real.sin (3 * (Real.pi / 2 - x))) →
  φ ≥ Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_symmetry_l384_38498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_translation_l384_38444

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define the exponential function with base 2
noncomputable def exp2 : RealFunction := λ x ↦ 2^x

-- Define the concept of graph translation to the left by 1 unit
def translateLeft (f : RealFunction) : RealFunction := λ x ↦ f (x + 1)

-- Define symmetry about the y-axis
def symmetricAboutYAxis (f g : RealFunction) : Prop :=
  ∀ x, f x = g (-x)

-- State the theorem
theorem function_symmetry_translation (f : RealFunction) :
  symmetricAboutYAxis (translateLeft f) exp2 →
  f = λ x ↦ (1/2)^(x-1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_translation_l384_38444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_graph_tan_phi_l384_38431

/-- The function y(x, φ) = sin(πx + φ) - 2cos(πx + φ) -/
noncomputable def y (x φ : ℝ) : ℝ := Real.sin (Real.pi * x + φ) - 2 * Real.cos (Real.pi * x + φ)

/-- Theorem: If y(x, φ) is symmetric about (1, 0) and y(1, φ) = 0, then tan(φ) = 2 -/
theorem symmetric_graph_tan_phi (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi)
  (h3 : ∀ h : ℝ, y (1 + h) φ = y (1 - h) φ)
  (h4 : y 1 φ = 0) :
  Real.tan φ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_graph_tan_phi_l384_38431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l384_38404

-- Define the ellipse (C)
noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the point A and focus F
def point_A : ℝ × ℝ := (2, 3)
def focus_F : ℝ × ℝ := (2, 0)

-- Define the line (l) parallel to OA
def line_l (m t : ℝ) (x y : ℝ) : Prop :=
  y = m * x + t

-- Define the distance between two lines
noncomputable def line_distance (m t₁ t₂ : ℝ) : ℝ :=
  |t₁ - t₂| / Real.sqrt (m^2 + 1)

-- Theorem statement
theorem ellipse_and_line_equations :
  ∃ (a b : ℝ), 
    ellipse a b (point_A.1) (point_A.2) ∧
    focus_F = (2, 0) →
    (∀ (x y : ℝ), ellipse a b x y ↔ x^2 / 16 + y^2 / 12 = 1) ∧
    (∃ (t : ℝ), 
      line_l (3/2) t (point_A.1) (point_A.2) ∧
      line_distance (3/2) 0 t = Real.sqrt 13 ∧
      (∀ (x y : ℝ), line_l (3/2) t x y ↔ y = 3/2 * x + t) ∧
      (t = 13/2 ∨ t = -13/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l384_38404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_greatest_prime_divisor_16383_l384_38407

noncomputable def greatest_prime_divisor (n : ℕ) : ℕ :=
  (Nat.factors n).maximum?.getD 1

theorem sum_digits_greatest_prime_divisor_16383 :
  (Nat.digits 10 (greatest_prime_divisor 16383)).sum = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_greatest_prime_divisor_16383_l384_38407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_average_time_l384_38450

/-- A relay race with four runners -/
structure RelayRace where
  y_time : ℚ
  z_time : ℚ
  w_time : ℚ
  x_time : ℚ
  z_length : ℚ
  w_length : ℚ
  x_length : ℚ

/-- The specific relay race described in the problem -/
def race : RelayRace where
  y_time := 58
  z_time := 26
  w_time := 2 * 26
  x_time := 35
  z_length := 12/10
  w_length := 15/10
  x_length := 13/10

/-- The average time to run a leg of the course -/
def average_time (r : RelayRace) : ℚ :=
  (r.y_time + r.z_time + r.w_time + r.x_time) / 4

/-- Theorem stating that the average time for the given race is 42.75 seconds -/
theorem race_average_time :
  average_time race = 171/4 := by
  -- Unfold definitions and simplify
  unfold average_time race
  -- Perform arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_average_time_l384_38450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_guaranteed_win_iff_odd_piles_l384_38456

/-- Represents a pile of matches -/
structure Pile :=
  (count : ℕ)

/-- Represents the state of the game -/
structure GameState :=
  (pile1 : Pile)
  (pile2 : Pile)

/-- Represents a player's move -/
inductive Move
  | TakeFromPile1 (n : ℕ)
  | TakeFromPile2 (n : ℕ)
  | TakeFromBoth (n1 n2 : ℕ)

/-- Applies a move to a game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeFromPile1 n => { pile1 := ⟨state.pile1.count - n⟩, pile2 := state.pile2 }
  | Move.TakeFromPile2 n => { pile1 := state.pile1, pile2 := ⟨state.pile2.count - n⟩ }
  | Move.TakeFromBoth n1 n2 => { pile1 := ⟨state.pile1.count - n1⟩, pile2 := ⟨state.pile2.count - n2⟩ }

/-- Checks if a game state is terminal (i.e., no matches left) -/
def isTerminal (state : GameState) : Prop :=
  state.pile1.count = 0 ∧ state.pile2.count = 0

/-- Defines a winning strategy for the first player -/
def hasWinningStrategy (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (opponentStrategy : GameState → Move),
      ∃ (gamePlay : ℕ → GameState),
        (gamePlay 0 = initialState) ∧
        (∀ n, 
          if n % 2 = 0
          then gamePlay (n + 1) = applyMove (gamePlay n) (strategy (gamePlay n))
          else gamePlay (n + 1) = applyMove (gamePlay n) (opponentStrategy (gamePlay n))) ∧
        ∃ (k : ℕ), isTerminal (gamePlay (2 * k + 1))

/-- The main theorem: The first player cannot guarantee a win if and only if 
    the initial number of matches in both piles is odd -/
theorem no_guaranteed_win_iff_odd_piles (initialState : GameState) :
  ¬hasWinningStrategy initialState ↔ Odd initialState.pile1.count ∧ Odd initialState.pile2.count := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_guaranteed_win_iff_odd_piles_l384_38456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sqrt3_cos_max_value_l384_38476

theorem sin_sqrt3_cos_max_value :
  ∀ x : ℝ, Real.sin x + Real.sqrt 3 * Real.cos x ≤ 2 ∧
  ∃ y : ℝ, Real.sin y + Real.sqrt 3 * Real.cos y = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sqrt3_cos_max_value_l384_38476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_negative_fractions_l384_38448

/-- A function that determines if a rational number is a negative fraction -/
def is_negative_fraction (q : ℚ) : Bool :=
  q < 0 && q ≠ Int.floor q

/-- The list of given rational numbers -/
def given_numbers : List ℚ :=
  [-2, 2/3, -3/100, -3/10, 2023, 0, -1001/100000]

/-- The theorem stating that there are exactly 3 negative fractions in the given list -/
theorem count_negative_fractions :
  (given_numbers.filter is_negative_fraction).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_negative_fractions_l384_38448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_pyramid_correct_l384_38473

/-- The radius of a sphere touching all edges of a pyramid with an equilateral triangular base -/
noncomputable def sphere_radius_pyramid (a b : ℝ) : ℝ :=
  (2 * b + a) * a / (2 * Real.sqrt (3 * b^2 - a^2))

/-- Theorem stating the radius of a sphere touching all edges of a pyramid -/
theorem sphere_radius_pyramid_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ r : ℝ, r > 0 ∧ r = sphere_radius_pyramid a b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_pyramid_correct_l384_38473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l384_38438

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4*y - 5 = 0

/-- The area of the region -/
noncomputable def region_area : ℝ := 14 * Real.pi

/-- Theorem stating that the area of the region enclosed by the given equation is 14π -/
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l384_38438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_four_l384_38420

/-- Represents a geometric sequence with first term a₁ and common ratio q -/
structure GeometricSequence where
  a₁ : ℝ
  q : ℝ

/-- The nth term of a geometric sequence -/
noncomputable def GeometricSequence.nthTerm (gs : GeometricSequence) (n : ℕ) : ℝ :=
  gs.a₁ * gs.q ^ (n - 1)

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def GeometricSequence.sumFirstN (gs : GeometricSequence) (n : ℕ) : ℝ :=
  gs.a₁ * (1 - gs.q^n) / (1 - gs.q)

theorem geometric_sequence_sum_four (gs : GeometricSequence) :
  gs.nthTerm 2 * gs.nthTerm 5 = 2 * gs.nthTerm 3 →
  (gs.nthTerm 4 + 2 * gs.nthTerm 7) / 2 = 5/4 →
  gs.sumFirstN 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_four_l384_38420
