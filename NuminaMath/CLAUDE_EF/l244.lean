import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_sequence_length_l244_24440

def is_valid_sequence (s : List Int) : Prop :=
  (∀ i, i + 2 < s.length → s[i]! + s[i+1]! + s[i+2]! > 0) ∧
  (∀ i, i + 4 < s.length → s[i]! + s[i+1]! + s[i+2]! + s[i+3]! + s[i+4]! < 0)

theorem largest_valid_sequence_length :
  (∃ s : List Int, is_valid_sequence s ∧ s.length = 9) ∧
  (∀ s : List Int, is_valid_sequence s → s.length ≤ 9) := by
  sorry

#check largest_valid_sequence_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_sequence_length_l244_24440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l244_24443

def my_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n : ℕ, n ≥ 2 → 2 * (a n)^2 = (a (n + 1))^2 + (a (n - 1))^2

theorem sixth_term_value (a : ℕ → ℝ) (h : my_sequence a) : a 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l244_24443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_survey_results_l244_24494

structure ReadingSurvey where
  n : ℕ
  frequencies : List ℕ
  frequencyRates : List ℚ

def stratifiedSample (survey : ReadingSurvey) (sampleSize : ℕ) : ℕ → ℕ := sorry

def minReadingTime (survey : ReadingSurvey) (topN : ℕ) : ℚ := sorry

theorem reading_survey_results (survey : ReadingSurvey) 
  (h1 : survey.frequencies = [4, 10, 46, 16, 20, 4])
  (h2 : survey.frequencyRates.get? 1 = some (1/10)) : 
  survey.n = 100 ∧ 
  survey.frequencyRates.get? 3 = some (16/100) ∧
  (stratifiedSample survey 10 80 : ℚ) / 10 = 3/5 ∧
  minReadingTime survey 10 = 94 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_survey_results_l244_24494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quad_pyramid_volume_l244_24424

/-- A quadrilateral pyramid with a rectangular base -/
structure QuadPyramid where
  /-- Height of the pyramid -/
  h : ℝ
  /-- Angle between lateral edges and height -/
  α : ℝ
  /-- Angle between base diagonals -/
  β : ℝ
  /-- Height is positive -/
  h_pos : 0 < h
  /-- Angle α is between 0 and π/2 -/
  α_range : 0 < α ∧ α < π/2
  /-- Angle β is between 0 and π -/
  β_range : 0 < β ∧ β < π

/-- Volume of a quadrilateral pyramid -/
noncomputable def volume (p : QuadPyramid) : ℝ :=
  (2/3) * p.h^3 * Real.tan p.α^2 * Real.sin p.β

/-- Theorem stating the volume of a quadrilateral pyramid -/
theorem quad_pyramid_volume (p : QuadPyramid) : 
  volume p = (2/3) * p.h^3 * Real.tan p.α^2 * Real.sin p.β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quad_pyramid_volume_l244_24424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l244_24409

noncomputable def y (a b x : ℝ) := a * Real.cos x + b

noncomputable def f (a b x : ℝ) := b * Real.sin (a * x + Real.pi / 3)

def increasing_interval (f : ℝ → ℝ) (a b : ℝ) :=
  if a > 0 then
    {x | ∃ k : ℤ, k * Real.pi + Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 7 * Real.pi / 12}
  else
    {x | ∃ k : ℤ, k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12}

theorem increasing_interval_of_f (a b : ℝ) :
  (∀ x, y a b x ≤ 1) ∧ 
  (∃ x, y a b x = 1) ∧
  (∀ x, y a b x ≥ -3) ∧
  (∃ x, y a b x = -3) →
  increasing_interval (f a b) a b = 
    if a > 0 then
      {x | ∃ k : ℤ, k * Real.pi + Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 7 * Real.pi / 12}
    else
      {x | ∃ k : ℤ, k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l244_24409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leontina_anniversary_l244_24465

/-- Represents Leontina's age --/
def leontina_age : ℕ := sorry

/-- Leontina's 40th anniversary year --/
def anniversary_year : ℕ := sorry

/-- The current year --/
def current_year : ℕ := sorry

/-- Function to calculate the sum of digits of a number --/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that Leontina's 40th anniversary was in 1962 --/
theorem leontina_anniversary :
  (leontina_age / 2 = 2 * sum_of_digits leontina_age) →
  (leontina_age > 40) →
  (leontina_age % 12 = 0) →
  (current_year = anniversary_year + leontina_age) →
  anniversary_year = 1962 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leontina_anniversary_l244_24465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_missouri_to_newyork_l244_24459

/-- The distance between two locations by car, given the distance by plane and the percentage increase -/
noncomputable def driving_distance (plane_distance : ℝ) (percentage_increase : ℝ) : ℝ :=
  plane_distance * (1 + percentage_increase / 100)

/-- The theorem stating the distance from Missouri to New York by car -/
theorem distance_missouri_to_newyork :
  let plane_distance : ℝ := 2000
  let percentage_increase : ℝ := 40
  let total_driving_distance := driving_distance plane_distance percentage_increase
  let missouri_to_newyork := total_driving_distance / 2
  missouri_to_newyork = 1400 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_missouri_to_newyork_l244_24459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_percentage_in_mixture_b_l244_24451

/-- Given two mixtures A and B, prove the alcohol percentage in mixture B -/
theorem alcohol_percentage_in_mixture_b 
  (percent_a : ℝ)
  (volume_final : ℝ)
  (percent_final : ℝ)
  (volume_a : ℝ)
  (h1 : percent_a = 20)
  (h2 : volume_final = 15)
  (h3 : percent_final = 30)
  (h4 : volume_a = 10)
  : (volume_final * percent_final / 100 - volume_a * percent_a / 100) / (volume_final - volume_a) * 100 = 50 := by
  sorry

#check alcohol_percentage_in_mixture_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_percentage_in_mixture_b_l244_24451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_triangle_area_unchanged_l244_24475

noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem transformed_triangle_area_unchanged
  (f : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) (A : ℝ) :
  triangle_area (x₁, f x₁) (x₂, f x₂) (x₃, f x₃) = A →
  triangle_area (x₁/2, 2 * f x₁) (x₂/2, 2 * f x₂) (x₃/2, 2 * f x₃) = A :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_triangle_area_unchanged_l244_24475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transition_characteristic_answer_is_correct_l244_24496

/-- Represents the type of population reproduction --/
inductive PopulationReproductionType
  | Traditional
  | Modern

/-- Represents demographic rates --/
structure DemographicRates where
  mortality_rate : ℝ
  birth_rate : ℝ

/-- Defines the transition from traditional to modern population reproduction --/
def transition_sign (initial : DemographicRates) (final : DemographicRates) : Prop :=
  final.mortality_rate < initial.mortality_rate ∧ final.birth_rate < initial.birth_rate

/-- Theorem stating the characteristic of the transition from traditional to modern population reproduction --/
theorem transition_characteristic 
  (initial : DemographicRates) 
  (final : DemographicRates) :
  transition_sign initial final →
  PopulationReproductionType.Traditional = PopulationReproductionType.Traditional ∧
  PopulationReproductionType.Modern = PopulationReproductionType.Modern :=
by
  intro h
  apply And.intro
  · rfl
  · rfl

/-- The correct answer is option B --/
def correct_answer : Prop :=
  ∃ (initial final : DemographicRates),
    transition_sign initial final ∧
    (PopulationReproductionType.Traditional = PopulationReproductionType.Traditional ∧
     PopulationReproductionType.Modern = PopulationReproductionType.Modern)

theorem answer_is_correct : correct_answer :=
by
  use { mortality_rate := 10, birth_rate := 20 }
  use { mortality_rate := 5, birth_rate := 15 }
  apply And.intro
  · simp [transition_sign]
    apply And.intro
    · linarith
    · linarith
  · exact transition_characteristic 
      { mortality_rate := 10, birth_rate := 20 } 
      { mortality_rate := 5, birth_rate := 15 } 
      (by simp [transition_sign]; apply And.intro; linarith; linarith)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transition_characteristic_answer_is_correct_l244_24496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_three_not_five_l244_24404

theorem multiples_of_three_not_five (n : ℕ) : n = 2009 →
  (Finset.filter (λ x : ℕ => x % 3 = 0 ∧ x % 5 ≠ 0) (Finset.range (n + 1))).card = 536 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_three_not_five_l244_24404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_property_diamond_19_98_l244_24412

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := x

-- Main theorem
theorem diamond_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  diamond a b = a :=
by
  -- Assumptions
  have h1 : ∀ (x y : ℝ), 0 < x → 0 < y → diamond (x * y) y = x * diamond y y := by sorry
  have h2 : ∀ (x : ℝ), 0 < x → diamond (diamond x 1) x = diamond x 1 := by sorry
  have h3 : diamond 1 1 = 1 := by sorry

  -- Proof
  rfl

-- Corollary for the specific case
theorem diamond_19_98 : diamond 19 98 = 19 :=
by
  have h19 : 0 < (19 : ℝ) := by norm_num
  have h98 : 0 < (98 : ℝ) := by norm_num
  exact diamond_property 19 98 h19 h98

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_property_diamond_19_98_l244_24412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l244_24403

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 2 - x) * Real.cos x + Real.sqrt 3 * Real.sin x ^ 2

theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  -- Intervals of monotonic decrease
  (∀ (k : ℤ), ∀ (x y : ℝ), 5*Real.pi/12 + k*Real.pi ≤ x ∧ x < y ∧ y ≤ 11*Real.pi/12 + k*Real.pi → f y < f x) ∧
  -- Minimum and maximum values in [π/6, π/2]
  (∀ (x : ℝ), Real.pi/6 ≤ x ∧ x ≤ Real.pi/2 → Real.sqrt 3 / 2 ≤ f x ∧ f x ≤ Real.sqrt 3 / 2 + 1) ∧
  (∃ (x₁ x₂ : ℝ), Real.pi/6 ≤ x₁ ∧ x₁ ≤ Real.pi/2 ∧ Real.pi/6 ≤ x₂ ∧ x₂ ≤ Real.pi/2 ∧ f x₁ = Real.sqrt 3 / 2 ∧ f x₂ = Real.sqrt 3 / 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l244_24403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_5_l244_24486

def A : Matrix (Fin 3) (Fin 3) ℚ := !![2, -1, 1; 3, 2, 2; 1, -2, 1]

theorem det_A_eq_5 : Matrix.det A = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_5_l244_24486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_graph_is_finite_distinct_points_l244_24401

-- Define the cost function for goldfish
def cost (n : ℕ) : ℕ := 20 * n

-- Define the set of valid numbers of goldfish
def valid_n : Set ℕ := {n | 1 ≤ n ∧ n ≤ 15}

-- Define the graph as a set of points
def graph : Set (ℕ × ℕ) := {p | ∃ n, n ∈ valid_n ∧ p = (n, cost n)}

-- Theorem statement
theorem goldfish_graph_is_finite_distinct_points :
  (Finite graph) ∧ 
  (∀ p q, p ∈ graph → q ∈ graph → p ≠ q → p.1 ≠ q.1 ∧ p.2 ≠ q.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_graph_is_finite_distinct_points_l244_24401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_imply_power_l244_24455

theorem like_terms_imply_power (m n : ℕ) : 
  (∃ (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0), ∀ (x y : ℚ), a * x^m * y^3 = b * x^2 * y^n) → 
  (-n : ℤ)^m = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_imply_power_l244_24455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_right_two_g_is_horizontal_shift_of_f_l244_24489

-- Define a generic continuous function f
variable (f : ℝ → ℝ)
variable (h : Continuous f)

-- Define the transformation g(x) = f(x - 2)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - 2)

-- State the theorem
theorem horizontal_shift_right_two (f : ℝ → ℝ) (x : ℝ) : 
  g f x = f (x - 2) :=
by
  -- Unfold the definition of g
  unfold g
  -- The result follows directly from the definition
  rfl

-- State that g is a horizontal shift of f
theorem g_is_horizontal_shift_of_f (f : ℝ → ℝ) : 
  ∃ (k : ℝ), ∀ (x : ℝ), g f x = f (x - k) ∧ k = 2 :=
by
  -- We claim that k = 2 satisfies the condition
  use 2
  intro x
  -- Split the goal into two parts
  constructor
  -- The first part follows from the definition of g
  · rfl
  -- The second part is trivial
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_right_two_g_is_horizontal_shift_of_f_l244_24489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l244_24418

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6) - 1

theorem f_properties :
  -- The period of f is π
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) → p ≥ Real.pi) ∧
  -- If f(α) = 6/5 and α ∈ (0, π/4), then cos(2α) = (3 + 4√3) / 10
  ∀ (α : ℝ), 0 < α ∧ α < Real.pi / 4 ∧ f α = 6 / 5 →
    Real.cos (2 * α) = (3 + 4 * Real.sqrt 3) / 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l244_24418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l244_24430

theorem solution_difference : ∃ x y : ℝ, 
  ((2 - x^2 / 5) ^ (1/3 : ℝ) = -1) ∧ 
  ((2 - y^2 / 5) ^ (1/3 : ℝ) = -1) ∧ 
  x ≠ y ∧ 
  |x - y| = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l244_24430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_reflection_l244_24491

-- Define the point P
def P : ℝ × ℝ := (-6, 7)

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 6*y + 21 = 0

-- Define the possible equations of the ray l
def ray_equation_1 (x y : ℝ) : Prop := 3*x + 4*y - 10 = 0
def ray_equation_2 (x y : ℝ) : Prop := 4*x + 3*y + 3 = 0

-- Define the distance function
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem light_ray_reflection :
  ∃ (x y : ℝ),
    (circle_eq x y) ∧ 
    ((ray_equation_1 x y) ∨ (ray_equation_2 x y)) ∧
    (∃ (x_reflect y_reflect : ℝ),
      y_reflect = 0 ∧ 
      distance P.1 P.2 x_reflect y_reflect + distance x_reflect y_reflect x y = 14) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_reflection_l244_24491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slice_volume_ratio_l244_24493

/-- Represents a right circular cone -/
structure RightCircularCone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a slice of a cone -/
structure ConeSlice where
  bottomRadius : ℝ
  topRadius : ℝ
  height : ℝ

/-- Calculates the volume of a cone slice -/
noncomputable def coneSliceVolume (slice : ConeSlice) : ℝ :=
  (1/3) * Real.pi * slice.height * (slice.bottomRadius^2 + slice.topRadius^2 + slice.bottomRadius * slice.topRadius)

/-- Theorem: The ratio of the volume of the second-largest piece to the largest piece
    in a right circular cone sliced into 6 equal height pieces is 61/91 -/
theorem cone_slice_volume_ratio (cone : RightCircularCone) : 
  let sliceHeight := cone.height / 6
  let largestSlice : ConeSlice := {
    bottomRadius := cone.baseRadius * 5 / 6,
    topRadius := cone.baseRadius,
    height := sliceHeight
  }
  let secondLargestSlice : ConeSlice := {
    bottomRadius := cone.baseRadius * 4 / 6,
    topRadius := cone.baseRadius * 5 / 6,
    height := sliceHeight
  }
  (coneSliceVolume secondLargestSlice) / (coneSliceVolume largestSlice) = 61 / 91 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slice_volume_ratio_l244_24493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_no_small_odd_prime_divisor_l244_24417

def has_no_small_odd_prime_divisor (n : ℤ) : Prop :=
  ∀ p : ℕ, p < 37 → Nat.Prime p → p ≠ 2 → ¬(p ∣ Int.natAbs n)

theorem sum_no_small_odd_prime_divisor 
  (S : Finset ℤ) 
  (h1 : S.card = 2019) 
  (h2 : ∀ n ∈ S, has_no_small_odd_prime_divisor n) : 
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ has_no_small_odd_prime_divisor (a + b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_no_small_odd_prime_divisor_l244_24417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramanujan_picked_l244_24471

-- Define the complex numbers
def hardy_number : ℂ := Complex.mk 6 2
def product : ℂ := Complex.mk 48 (-16)

-- Define Ramanujan's number as a function of Hardy's number and the product
noncomputable def ramanujan_number : ℂ := product / hardy_number

-- Theorem statement
theorem ramanujan_picked : ramanujan_number = Complex.mk 6.4 (-4.8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramanujan_picked_l244_24471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_circle_cat_max_distance_l244_24433

/-- The maximum distance from the origin to a point on a circle -/
theorem max_distance_to_circle (center_x center_y radius : ℝ) :
  let center_distance := Real.sqrt (center_x^2 + center_y^2)
  let max_distance := center_distance + radius
  ∀ point : ℝ × ℝ, 
    ((point.1 - center_x)^2 + (point.2 - center_y)^2 ≤ radius^2) →
    (point.1^2 + point.2^2 ≤ max_distance^2) :=
by sorry

/-- The specific case for the cat problem -/
theorem cat_max_distance : 
  let center : ℝ × ℝ := (6, 8)
  let radius : ℝ := 12
  let max_distance := Real.sqrt (center.1^2 + center.2^2) + radius
  max_distance = 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_circle_cat_max_distance_l244_24433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_range_l244_24428

noncomputable def f (a x : ℝ) : ℝ :=
  if x < a + 1 then (1/2) ^ abs (x - a)
  else -abs (x + 1) - a

theorem f_max_value_range (a : ℝ) :
  (∀ x, f a x ≤ 1) ↔ a ∈ Set.Ici (- 3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_range_l244_24428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_sculpture_weight_l244_24456

theorem marble_sculpture_weight (W : ℝ) : 
  W * 0.75 * 0.85 * 0.90 = 109.0125 → 
  ∃ ε > 0, |W - 190| < ε := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_sculpture_weight_l244_24456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_sides_l244_24431

/-- A convex polygon with n sides where interior angles form an arithmetic sequence --/
structure ConvexPolygon where
  n : ℕ
  smallestAngle : ℚ
  commonDifference : ℚ

/-- The sum of interior angles of a polygon with n sides --/
def interiorAngleSum (n : ℕ) : ℚ := (n - 2) * 180

/-- The sum of an arithmetic sequence --/
def arithmeticSequenceSum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

/-- Theorem: A convex polygon with interior angles in arithmetic sequence,
    smallest angle 120°, and common difference 5°, has 9 sides --/
theorem convex_polygon_sides (p : ConvexPolygon) 
    (h₁ : p.smallestAngle = 120)
    (h₂ : p.commonDifference = 5) :
    p.n = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_sides_l244_24431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l244_24406

open MeasureTheory Interval Set

variable (f g : ℝ → ℝ) (a : ℝ)

theorem integral_inequality
  (hf : ContinuousOn f (Icc 0 1))
  (hg : ContinuousOn g (Icc 0 1))
  (hf_diff : DifferentiableOn ℝ f (Icc 0 1))
  (hg_diff : DifferentiableOn ℝ g (Icc 0 1))
  (hf_zero : f 0 = 0)
  (hf_nonneg : ∀ x ∈ Icc 0 1, deriv f x ≥ 0)
  (hg_nonneg : ∀ x ∈ Icc 0 1, deriv g x ≥ 0)
  (ha : a ∈ Icc 0 1) :
  ∫ x in Ioc 0 a, g x * (deriv f x) + ∫ x in Ioc 0 1, f x * (deriv g x) ≥ f a * g 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l244_24406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_value_f_max_min_value_l244_24439

noncomputable section

open Real

def m (x : ℝ) : ℝ × ℝ := (sin (x - π/3), 1)
def n (x : ℝ) : ℝ × ℝ := (cos x, 1)

def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem parallel_vectors_tan_value (x : ℝ) :
  (∃ (k : ℝ), m x = k • n x) → tan x = sqrt 3 + 2 :=
by sorry

theorem f_max_min_value :
  (∀ x ∈ Set.Icc 0 (π/2), f x ≤ 1 + sqrt 3 / 2) ∧
  (∀ x ∈ Set.Icc 0 (π/2), f x ≥ 1 - sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 (π/2), f x = 1 + sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 (π/2), f x = 1 - sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tan_value_f_max_min_value_l244_24439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_triangle_exists_l244_24488

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the centroid of triangle ABC
noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Define a point M
variable (M : ℝ × ℝ)

-- Define the theorem
theorem inner_triangle_exists :
  ∃ (A₁ B₁ C₁ : ℝ × ℝ),
    -- A₁B₁C₁ is inside ABC
    (∃ (α β γ : ℝ), α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0 ∧ α + β + γ = 1 ∧
      A₁ = (α * A.1 + β * B.1 + γ * C.1, α * A.2 + β * B.2 + γ * C.2)) ∧
    (∃ (α β γ : ℝ), α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0 ∧ α + β + γ = 1 ∧
      B₁ = (α * A.1 + β * B.1 + γ * C.1, α * A.2 + β * B.2 + γ * C.2)) ∧
    (∃ (α β γ : ℝ), α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0 ∧ α + β + γ = 1 ∧
      C₁ = (α * A.1 + β * B.1 + γ * C.1, α * A.2 + β * B.2 + γ * C.2)) ∧
    -- One side of A₁B₁C₁ passes through M
    (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
      (M = (t * A₁.1 + (1 - t) * B₁.1, t * A₁.2 + (1 - t) * B₁.2) ∨
       M = (t * B₁.1 + (1 - t) * C₁.1, t * B₁.2 + (1 - t) * C₁.2) ∨
       M = (t * C₁.1 + (1 - t) * A₁.1, t * C₁.2 + (1 - t) * A₁.2))) ∧
    -- Centroid of A₁B₁C₁ coincides with centroid of ABC
    centroid A₁ B₁ C₁ = centroid A B C := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_triangle_exists_l244_24488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_ten_l244_24446

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2 + Real.log x else -2 - Real.log (-x)

-- State the theorem
theorem f_negative_ten (h_odd : ∀ x, f (-x) = -f x) : f (-10) = -3 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_ten_l244_24446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_of_equation_l244_24469

theorem min_abs_diff_of_equation (a b : ℕ) (h : a * b + 5 * a - 4 * b = 684) :
  ∃ (c d : ℕ), c * d + 5 * c - 4 * d = 684 ∧ |Int.ofNat c - Int.ofNat d| ≤ |Int.ofNat a - Int.ofNat b| ∧ |Int.ofNat c - Int.ofNat d| = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_of_equation_l244_24469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_equals_three_l244_24462

/-- Prove that in an acute triangle ABC, given specific conditions, the side b equals 3. -/
theorem side_b_equals_three (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π/2 →  -- Acute angle A
  0 < B ∧ B < π/2 →  -- Acute angle B
  0 < C ∧ C < π/2 →  -- Acute angle C
  Real.sin A = 2 * Real.sqrt 2 / 3 →
  Real.sin B > Real.sin C →
  a = 3 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 2 →
  b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_equals_three_l244_24462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_cdf_l244_24434

noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if x ≤ 2 then x^2
  else 1

theorem not_cdf : ¬ (∀ (x : ℝ), 0 ≤ F x ∧ F x ≤ 1 ∧
  (∀ (y : ℝ), x ≤ y → F x ≤ F y) ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ ∀ (y : ℝ), x < y ∧ y < x + δ → |F y - F x| < ε) ∧
  (∀ (x : ℝ), F (-abs x) = 0) ∧
  (∃ (M : ℝ), ∀ (x : ℝ), x ≥ M → F x = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_cdf_l244_24434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_acute_triangles_on_circle_l244_24442

open Complex

noncomputable def is_acute_triangle (t : Finset (ℂ)) : Prop :=
  sorry

theorem max_acute_triangles_on_circle (n : ℕ) (h : n = 16) :
  ∃ (points : Finset ℂ),
    (∀ p ∈ points, Complex.abs p = 1) ∧
    Finset.card points = n ∧
    (∃ (acute_triangles : Finset (Finset ℂ)),
      (∀ t ∈ acute_triangles,
        t.card = 3 ∧
        (∀ p ∈ t, p ∈ points) ∧
        is_acute_triangle t) ∧
      acute_triangles.card = 168 ∧
      (∀ other_acute_triangles : Finset (Finset ℂ),
        (∀ t ∈ other_acute_triangles,
          t.card = 3 ∧
          (∀ p ∈ t, p ∈ points) ∧
          is_acute_triangle t) →
        other_acute_triangles.card ≤ 168)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_acute_triangles_on_circle_l244_24442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bears_championship_probability_l244_24492

/-- The probability of the Bears winning a single game -/
noncomputable def p : ℝ := 2/3

/-- The number of games the Bears need to win to secure the championship -/
def games_to_win : ℕ := 5

/-- The probability of the Bears winning the championship -/
noncomputable def bears_win_probability : ℝ :=
  (Finset.range 5).sum (λ i => 
    (Nat.choose (games_to_win + i - 1) i) * p^games_to_win * (1-p)^i)

/-- The theorem stating the probability of the Bears winning the championship -/
theorem bears_championship_probability : 
  ∃ (x : ℝ), abs (bears_win_probability - x) < 0.005 ∧ x = 0.82 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bears_championship_probability_l244_24492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_consecutive_x_l244_24485

def x : ℕ → ℚ
  | 0 => 1/20  -- Added case for 0
  | 1 => 1/20
  | 2 => 1/13
  | (n+3) => (2 * x (n+1) * x (n+2) * (x (n+1) + x (n+2))) / (x (n+1)^2 + x (n+2)^2)

theorem sum_reciprocal_consecutive_x : ∑' n, 1 / (x n + x (n+1)) = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_consecutive_x_l244_24485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_theorem_l244_24402

theorem product_sum_theorem (x₁ x₂ x₃ x₄ : ℝ) :
  (x₁ * x₂ * x₃ + x₁ * x₂ * x₄ + x₁ * x₃ * x₄ + x₂ * x₃ * x₄ = 2) →
  ((x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
   (Finset.card {x₁, x₂, x₃, x₄} = 2 ∧ x₁ * x₂ * x₃ * x₄ = -3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_theorem_l244_24402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_increases_are_100_percent_options_ADE_are_correct_l244_24499

-- Define the prize values for each question
noncomputable def prize_values : List ℚ := [3600, 7200, 14400, 28800, 57600, 115200, 230400, 921600, 1843200]

-- Define the function to calculate percent increase
noncomputable def percent_increase (a b : ℚ) : ℚ := (b - a) / a * 100

-- Theorem statement
theorem all_increases_are_100_percent :
  ∀ i, i + 1 < prize_values.length →
    percent_increase (prize_values.get ⟨i, by sorry⟩) (prize_values.get ⟨i + 1, by sorry⟩) = 100 := by
  sorry

-- Theorem to show that options A, D, and E are all correct answers
theorem options_ADE_are_correct :
  percent_increase 3600 7200 = 100 ∧
  percent_increase 115200 230400 = 100 ∧
  percent_increase 921600 1843200 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_increases_are_100_percent_options_ADE_are_correct_l244_24499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l244_24464

def b : ℕ → ℕ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | (n + 2) => b (n + 1) + 3 * (n + 1)^2 - (n + 1) + 2

theorem b_50_value : b 50 = 122600 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l244_24464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_defective_l244_24441

/-- The probability of selecting at least one defective bulb when randomly choosing two bulbs from a box containing 23 bulbs, of which 4 are defective. -/
theorem probability_at_least_one_defective (total_bulbs : ℕ) (defective_bulbs : ℕ) 
  (h1 : total_bulbs = 23) (h2 : defective_bulbs = 4) : 
  (1 : ℚ) - (total_bulbs - defective_bulbs : ℚ) * (total_bulbs - defective_bulbs - 1) / (total_bulbs * (total_bulbs - 1)) = 164 / 506 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_defective_l244_24441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arrangement_l244_24425

/-- Represents a 5x8 table filled with 1s and 3s -/
def Table := Fin 5 → Fin 8 → Fin 2

/-- The sum of a row in the table -/
def row_sum (t : Table) (i : Fin 5) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 8)) fun j => if t i j = 0 then 1 else 3)

/-- The sum of a column in the table -/
def col_sum (t : Table) (j : Fin 8) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 5)) fun i => if t i j = 0 then 1 else 3)

/-- Statement: It's impossible to fill the table such that all row and column sums are divisible by 7 -/
theorem impossible_arrangement : ¬ ∃ (t : Table), 
  (∀ i : Fin 5, row_sum t i % 7 = 0) ∧ 
  (∀ j : Fin 8, col_sum t j % 7 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arrangement_l244_24425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_dihedral_angle_l244_24420

noncomputable def dihedral_angle : ℝ := 60 * (Real.pi / 180)

noncomputable def spherical_distance : ℝ := 2 * Real.pi

theorem sphere_radius_in_dihedral_angle 
  (angle : ℝ) 
  (dist : ℝ) 
  (h1 : angle = dihedral_angle) 
  (h2 : dist = spherical_distance) : 
  ∃ (r : ℝ), r = 3 ∧ 
  (angle / Real.pi) * r = dist / (2 * Real.pi) := by
  sorry

#check sphere_radius_in_dihedral_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_dihedral_angle_l244_24420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_stations_l244_24483

/-- Represents a train with its speed -/
structure Train where
  speed : ℝ

/-- Represents the problem setup -/
structure TrainProblem where
  t1 : Train
  t2 : Train
  distance_at_3h : ℝ
  distance_at_6h : ℝ

/-- The theorem stating the distance between stations -/
theorem distance_between_stations (p : TrainProblem)
  (h1 : p.distance_at_3h = 70)
  (h2 : p.distance_at_6h = 70) :
  p.t1.speed * 3 + p.t2.speed * 3 = 70 := by
  sorry

#check distance_between_stations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_stations_l244_24483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_classification_l244_24476

/-- Definition of the complex number z in terms of the real number m -/
def z (m : ℝ) : ℂ := m + 1 + (m - 1) * Complex.I

/-- Theorem stating the conditions for z to be real, complex, or purely imaginary -/
theorem z_classification (m : ℝ) : 
  ((z m).im = 0 ↔ m = 1) ∧
  ((z m).im ≠ 0 ↔ m ≠ 1) ∧
  ((z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_classification_l244_24476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pumpkin_slice_price_is_5_l244_24461

/-- Represents the price of a slice of pumpkin pie -/
def pumpkin_slice_price : ℝ := 5

/-- Number of slices in a pumpkin pie -/
def pumpkin_slices_per_pie : ℕ := 8

/-- Number of slices in a custard pie -/
def custard_slices_per_pie : ℕ := 6

/-- Price of a slice of custard pie -/
def custard_slice_price : ℝ := 6

/-- Number of pumpkin pies sold -/
def pumpkin_pies_sold : ℕ := 4

/-- Number of custard pies sold -/
def custard_pies_sold : ℕ := 5

/-- Total revenue from all pie sales -/
def total_revenue : ℝ := 340

theorem pumpkin_slice_price_is_5 :
  pumpkin_slice_price = 5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pumpkin_slice_price_is_5_l244_24461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_integer_in_sequence_l244_24460

def my_sequence (x : ℕ → ℚ) : Prop :=
  x 1 > 1 ∧ ∀ n : ℕ, x (n + 1) = x n + 1 / ↑⌊x n⌋

theorem exists_integer_in_sequence (x : ℕ → ℚ) (h : my_sequence x) :
  ∃ n : ℕ, ∃ k : ℤ, x n = k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_integer_in_sequence_l244_24460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_for_f_zero_l244_24453

def f (x y z : ℕ) : ℕ := (x - y).factorial % (x + z)

theorem max_y_for_f_zero (x z : ℕ) (hx : x = 100) (hz : z = 50) :
  ∃ y_max : ℕ, y_max = 75 ∧
  f x y_max z = 0 ∧
  ∀ y : ℕ, y > y_max → f x y z ≠ 0 := by
  sorry

#eval f 100 75 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_for_f_zero_l244_24453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_70_l244_24419

def scores : List ℕ := [65, 70, 75, 85, 95]

def is_integer_average (subset : List ℕ) : Prop :=
  ∃ n : ℤ, (subset.sum : ℚ) / subset.length = n

theorem last_score_is_70 :
  ∃ (order : List ℕ), 
    order.toFinset = scores.toFinset ∧
    order.length = scores.length ∧
    (∀ k : ℕ, k ≤ order.length → is_integer_average (order.take k)) ∧
    order.getLast? = some 70 :=
by
  sorry

#check last_score_is_70

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_70_l244_24419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l244_24447

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 6

-- Theorem statement
theorem f_has_unique_zero : ∃! x : ℝ, f x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l244_24447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_primes_not_dividing_l244_24481

/-- A function that pairs the positive divisors of a number -/
def divisor_pairing (n : ℕ+) : Set (ℕ × ℕ) := sorry

/-- The property that the sum of each pair in the pairing is prime -/
def sum_of_pairs_prime (pairing : Set (ℕ × ℕ)) : Prop :=
  ∀ p ∈ pairing, Nat.Prime (p.1 + p.2)

/-- The theorem stating that if a positive integer's divisors can be paired
    such that the sum of each pair is prime, then these primes are distinct
    and do not divide the original number -/
theorem distinct_primes_not_dividing (n : ℕ+) 
  (pairing : Set (ℕ × ℕ))
  (h_pairing : pairing = divisor_pairing n)
  (h : sum_of_pairs_prime pairing) :
  (∀ p q : ℕ, p ∈ (λ (x : ℕ × ℕ) => x.1 + x.2) '' pairing → 
              q ∈ (λ (x : ℕ × ℕ) => x.1 + x.2) '' pairing → 
              p ≠ q → p ≠ q) ∧ 
  (∀ p : ℕ, p ∈ (λ (x : ℕ × ℕ) => x.1 + x.2) '' pairing → ¬(p ∣ n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_primes_not_dividing_l244_24481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_cone_l244_24479

/-- The volume of a cone given its slant height and height --/
noncomputable def cone_volume (slant_height height : ℝ) : ℝ :=
  let radius := Real.sqrt (slant_height^2 - height^2)
  (1/3) * Real.pi * radius^2 * height

/-- Theorem: The volume of a cone with slant height 15 cm and height 9 cm is 432π cubic centimeters --/
theorem volume_of_specific_cone :
  cone_volume 15 9 = 432 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_cone_l244_24479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_equals_cos_l244_24480

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => Real.cos
  | n + 1 => λ x => deriv (f n) x

theorem f_2012_equals_cos : f 2012 = Real.cos := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_equals_cos_l244_24480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_gt_f_six_l244_24467

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of f being decreasing on (4, +∞)
def decreasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 4 ∧ y > 4 ∧ x < y → f x > f y

-- Define the property of f(x + 4) being an even function
def even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f (-x + 4)

-- State the theorem
theorem f_three_gt_f_six
  (h1 : decreasing_on_interval f)
  (h2 : even_shifted f) :
  f 3 > f 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_gt_f_six_l244_24467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sculpture_exposed_area_l244_24458

/-- Represents a layer in the sculpture -/
structure Layer where
  cubes : Nat
  exposed_side_faces : Nat

/-- Represents the sculpture -/
structure Sculpture where
  layers : List Layer
  total_cubes : Nat

/-- Calculates the total exposed surface area of the sculpture -/
def total_exposed_area (s : Sculpture) : Nat :=
  let side_area := s.layers.map (fun l => l.exposed_side_faces) |>.sum
  let top_area := s.layers.map (fun l => l.cubes) |>.sum
  side_area + top_area

/-- The theorem to be proved -/
theorem sculpture_exposed_area :
  ∃ (s : Sculpture),
    s.total_cubes = 15 ∧
    s.layers.length = 3 ∧
    (s.layers.map (fun l => l.cubes)) = [3, 5, 7] ∧
    total_exposed_area s = 49 := by
  sorry

#eval "Theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sculpture_exposed_area_l244_24458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_min_area_l244_24490

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the line L
def line_L (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the tangent line equation
def tangent_line (m : ℝ) (x y : ℝ) : Prop :=
  x = 3 ∨ y = ((m^2 - 1) / (2*m)) * x + ((3 - m^2) / (2*m))

-- Define the area of quadrilateral QACB
noncomputable def area_QACB (x y : ℝ) : ℝ := Real.sqrt ((x - 2)^2 + y^2 - 1)

theorem circle_tangent_and_min_area :
  ∀ (m x y : ℝ),
    circle_C x y →
    line_L x y →
    (∀ (x' y' : ℝ), tangent_line m x' y' → 
      ((x' - x)^2 + (y' - y)^2 ≤ (3 - x)^2 + (m - y)^2)) →
    area_QACB x y = Real.sqrt 7 ∧
    x = 4 ∧ y = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_min_area_l244_24490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_shaded_area_l244_24448

noncomputable section

/-- The shaded area of a tiled floor with white quarter circles in the corners -/
def shaded_area (floor_length floor_width tile_size circle_radius : ℝ) : ℝ :=
  let total_tiles := (floor_length / tile_size) * (floor_width / tile_size)
  let white_area_per_tile := 4 * (Real.pi * circle_radius^2 / 4)
  let shaded_area_per_tile := tile_size^2 - white_area_per_tile
  total_tiles * shaded_area_per_tile

/-- The total shaded area of the floor is 72 - 8π square feet -/
theorem floor_shaded_area :
  shaded_area 12 6 1 (1/3) = 72 - 8*Real.pi := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_shaded_area_l244_24448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l244_24411

theorem expression_evaluation : 3 - (-3)^(3-3) = 2 := by
  -- Evaluate the exponent
  have h1 : 3 - 3 = 0 := by ring
  
  -- Simplify (-3)^0
  have h2 : (-3)^0 = 1 := by exact pow_zero (-3)
  
  -- Evaluate the final expression
  calc
    3 - (-3)^(3-3) = 3 - (-3)^0 := by rw [h1]
    _               = 3 - 1     := by rw [h2]
    _               = 2         := by ring

  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l244_24411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l244_24457

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (1 + t, 1 + 2 * t)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ :=
  let ρ := 1 / (1 - Real.sin θ)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point M
def M : ℝ × ℝ := (0, -1)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Statement of the theorem
theorem intersection_distance_product :
  ∃ A B : ℝ × ℝ,
    (∃ t : ℝ, C₁ t = A) ∧
    (∃ θ : ℝ, C₂ θ = A) ∧
    (∃ t : ℝ, C₁ t = B) ∧
    (∃ θ : ℝ, C₂ θ = B) ∧
    A ≠ B ∧
    distance M A * distance M B = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l244_24457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_middle_section_l244_24498

/-- Generic shape structure (for type checking purposes) -/
structure Shape where
  area : ℝ
  perimeter : ℝ

/-- Truncated pyramid structure (for type checking purposes) -/
structure TruncatedPyramid where
  base₁ : Shape
  base₂ : Shape
  middleSection : Shape

/-- Properties of the middle section of a truncated pyramid -/
theorem truncated_pyramid_middle_section
  (k₁ k₂ t₁ t₂ : ℝ)
  (h₁ : 0 < k₁) (h₂ : 0 < k₂) (h₃ : 0 < t₁) (h₄ : 0 < t₂) :
  ∃ (pyramid : TruncatedPyramid),
    pyramid.base₁.perimeter = k₁ ∧
    pyramid.base₂.perimeter = k₂ ∧
    pyramid.base₁.area = t₁ ∧
    pyramid.base₂.area = t₂ ∧
    pyramid.middleSection.area = (t₁ + 2 * Real.sqrt (t₁ * t₂) + t₂) / 4 ∧
    pyramid.middleSection.perimeter = (k₁ + k₂) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_middle_section_l244_24498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_flight_time_l244_24410

/-- The time Lisa flew in hours -/
noncomputable def flight_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Theorem stating that Lisa's flight time is 8 hours -/
theorem lisa_flight_time :
  flight_time 256 32 = 8 := by
  -- Unfold the definition of flight_time
  unfold flight_time
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_flight_time_l244_24410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_exponents_max_value_is_neg_two_max_sum_exponents_equals_neg_two_l244_24436

theorem max_sum_exponents (x y : ℝ) (h : (2 : ℝ)^x + (2 : ℝ)^y = 1) : 
  ∀ a b : ℝ, (2 : ℝ)^a + (2 : ℝ)^b = 1 → x + y ≥ a + b :=
by sorry

theorem max_value_is_neg_two (x y : ℝ) (h : (2 : ℝ)^x + (2 : ℝ)^y = 1) : 
  x + y ≤ -2 :=
by sorry

theorem max_sum_exponents_equals_neg_two : 
  ∃ x y : ℝ, (2 : ℝ)^x + (2 : ℝ)^y = 1 ∧ x + y = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_exponents_max_value_is_neg_two_max_sum_exponents_equals_neg_two_l244_24436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_degrees_is_216_l244_24470

/-- Represents the budget percentages for different categories --/
structure BudgetPercentages where
  transportation : ℚ
  research_and_development : ℚ
  utilities : ℚ
  equipment : ℚ
  supplies : ℚ

/-- The total number of degrees in a circle --/
def circle_degrees : ℚ := 360

/-- Calculates the percentage for salaries given the other budget percentages --/
def salary_percentage (bp : BudgetPercentages) : ℚ :=
  100 - (bp.transportation + bp.research_and_development + bp.utilities + bp.equipment + bp.supplies)

/-- Calculates the number of degrees in a circle graph representing salaries --/
def salary_degrees (bp : BudgetPercentages) : ℚ :=
  (salary_percentage bp / 100) * circle_degrees

/-- Theorem stating that given the specified budget percentages, 
    the number of degrees representing salaries is 216 --/
theorem salary_degrees_is_216 (bp : BudgetPercentages) 
  (h1 : bp.transportation = 20)
  (h2 : bp.research_and_development = 9)
  (h3 : bp.utilities = 5)
  (h4 : bp.equipment = 4)
  (h5 : bp.supplies = 2) :
  salary_degrees bp = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_degrees_is_216_l244_24470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domains_and_ranges_l244_24400

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (9^x + 3^x - a)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if x > 0 then Real.exp (f 0 x) - 9^x
  else if x < 0 then -(Real.exp (f 0 (-x)) - 9^(-x))
  else 0

-- State the theorem
theorem function_domains_and_ranges :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ a ≤ 0 ∧
  (∀ x : ℝ, g (-x) = -g x) ∧
  (∀ x : ℝ, x > 0 → g x = 3^x) ∧
  (∀ t : ℝ, (∀ x : ℝ, ∃! y z : ℝ, y ≠ z ∧ g (y^2 - 2*t*y + 3) / g y = |g y| ∧ g (z^2 - 2*t*z + 3) / g z = |g z|) ↔
    (t < -1 - Real.sqrt 3 ∨ t > Real.sqrt 3 - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domains_and_ranges_l244_24400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_determines_a_l244_24473

/-- Represents an ellipse with equation x²/a² + y²/8 = 1 -/
structure Ellipse where
  a : ℝ
  h_pos : a > 0

/-- The focal length of an ellipse -/
noncomputable def focalLength (e : Ellipse) : ℝ := 2 * Real.sqrt (e.a^2 - 8)

theorem ellipse_focal_length_determines_a (e : Ellipse) (h : focalLength e = 4) : e.a = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_determines_a_l244_24473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_third_quadrant_l244_24474

theorem cos_B_third_quadrant (B : ℝ) (h1 : B ∈ Set.Icc π (3*π/2)) 
  (h2 : Real.sin B = 5/13) : Real.cos B = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_third_quadrant_l244_24474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_theorem_l244_24445

/-- The diameter of a circle encompassing seven tangent circles of radius 2, which are also tangent to each other -/
noncomputable def large_circle_diameter : ℝ :=
  4 / Real.sin (Real.pi / 7) + 4

/-- The configuration of seven small circles within a large circle -/
structure CircleConfiguration where
  /-- The radius of each small circle -/
  small_radius : ℝ
  /-- The number of small circles -/
  num_circles : ℕ
  /-- The small circles are tangent to each other -/
  tangent_small : Prop
  /-- The small circles are tangent to the large circle -/
  tangent_large : Prop

/-- The theorem stating the diameter of the large circle in the given configuration -/
theorem large_circle_diameter_theorem (config : CircleConfiguration)
    (h1 : config.small_radius = 2)
    (h2 : config.num_circles = 7)
    (h3 : config.tangent_small)
    (h4 : config.tangent_large) :
    large_circle_diameter = 4 / Real.sin (Real.pi / 7) + 4 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_theorem_l244_24445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iphone_price_reduction_l244_24405

/-- Given an initial price and two consecutive percentage reductions,
    calculate the final price after both reductions are applied. -/
noncomputable def final_price (initial_price : ℝ) (reduction1 : ℝ) (reduction2 : ℝ) : ℝ :=
  initial_price * (1 - reduction1 / 100) * (1 - reduction2 / 100)

/-- Theorem stating that the price of an iPhone after two consecutive
    reductions of 10% and 20%, starting from $1000, is $720. -/
theorem iphone_price_reduction : final_price 1000 10 20 = 720 := by
  -- Unfold the definition of final_price
  unfold final_price
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iphone_price_reduction_l244_24405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_intersection_l244_24438

-- Define the circles and points as structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define the necessary functions
def Intersect (c1 c2 : Circle) (p1 p2 : Point) : Prop := sorry
def Contains (c1 c2 : Circle) : Prop := sorry
def TangentAt (c1 c2 : Circle) (p : Point) : Prop := sorry
def OnLine (p : Point) (l : Point → Point → Prop) : Prop := sorry
def OnCircle (p : Point) (c : Circle) : Prop := sorry
def Line (p1 p2 : Point) : Point → Point → Prop := sorry
def TangentPoint (c : Circle) (p : Point) : Point := sorry

-- Define the theorem
theorem tangents_intersection
  (Γ₁ Γ₂ Γ : Circle)
  (A B C D E F : Point) :
  Intersect Γ₁ Γ₂ A B →
  Contains Γ Γ₁ →
  Contains Γ Γ₂ →
  TangentAt Γ Γ₁ C →
  TangentAt Γ Γ₂ D →
  OnLine E (Line A B) →
  OnLine F (Line A B) →
  OnCircle E Γ →
  OnCircle F Γ →
  ∃ P : Point, OnLine P (Line C D) ∧ OnLine P (Line (TangentPoint Γ E) (TangentPoint Γ F)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_intersection_l244_24438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_future_value_approx_l244_24454

/-- Calculate the future value of an investment with compound interest -/
def future_value (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The combined future value of the three investments -/
def combined_future_value : ℝ :=
  future_value 3000 0.05 3 + future_value 5000 0.06 4 + future_value 7000 0.07 5

/-- Theorem stating that the combined future value is approximately 19603.119 -/
theorem combined_future_value_approx :
  |combined_future_value - 19603.119| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_future_value_approx_l244_24454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_sum_representation_l244_24477

theorem lcm_sum_representation (n : ℕ) :
  (∃ a b c : ℕ, Nat.lcm a b + Nat.lcm a c + Nat.lcm b c = n) ↔ ¬(∃ k : ℕ, n = 2^k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_sum_representation_l244_24477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_of_f_B_l244_24427

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos (x - Real.pi / 2) * Real.sin (x - Real.pi / 3) - 1

-- Theorem for the smallest positive period
theorem smallest_positive_period : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by sorry

-- Define a triangle with sides forming a geometric sequence
structure GeometricTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  geometric_seq : b^2 = a * c

-- Theorem for the range of f(B)
theorem range_of_f_B (t : GeometricTriangle) : 
  ∃ (B : ℝ), 0 < B ∧ B < Real.pi ∧ 
  Real.cos B = (t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c) ∧
  f B ∈ Set.Icc (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_of_f_B_l244_24427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_l244_24472

/-- The slope of a line perpendicular to the line containing points (1, 3) and (-6, 6) is 7/3 -/
theorem perpendicular_slope :
  let x₁ : ℝ := 1
  let y₁ : ℝ := 3
  let x₂ : ℝ := -6
  let y₂ : ℝ := 6
  let m := (y₂ - y₁) / (x₂ - x₁)
  (-1 / m) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_l244_24472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l244_24421

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def Parallel (v w : V) : Prop := ∃ (c : ℝ), v = c • w

theorem parallel_vectors (a b : V) (lambda : ℝ) 
  (h1 : ¬ Parallel a b) 
  (h2 : Parallel (lambda • a + b) (a + 2 • b)) : 
  lambda = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l244_24421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_l244_24408

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the curve
def my_curve (x y : ℝ) : Prop := y = abs x - 1

-- Theorem statement
theorem no_common_points :
  ¬ ∃ (x y : ℝ), my_circle x y ∧ my_curve x y :=
by
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_l244_24408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_5_2_range_of_a_l244_24463

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 1/2|

-- Part I
theorem solution_set_when_a_is_5_2 :
  {x : ℝ | f (5/2) x ≤ x + 10} = {x : ℝ | -7/3 ≤ x ∧ x ≤ 13} := by sorry

-- Part II
theorem range_of_a :
  {a : ℝ | ∀ x, f a x ≥ a} = Set.Ici 0 ∩ Set.Iic (1/4) := by sorry

-- Set.Ici 0 ∩ Set.Iic (1/4) represents the interval [0, 1/4] in Lean

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_5_2_range_of_a_l244_24463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_cube_root_equation_l244_24407

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.rpow (15 * x + Real.rpow (15 * x + 8) (1/3 : ℝ)) (1/3 : ℝ)

-- State the theorem
theorem unique_solution_cube_root_equation :
  ∃! x : ℝ, f x = 10 ∧ x = 992 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_cube_root_equation_l244_24407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_perpendicular_vector_l244_24416

/-- Given two vectors a and b in R^2, where a is perpendicular to b,
    prove that the magnitude of b is 3√5 -/
theorem magnitude_of_perpendicular_vector 
  (a b : ℝ × ℝ) 
  (h1 : a = (-1, 2)) 
  (h2 : ∃ x, b = (x, 3)) 
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) : 
  ‖b‖ = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_perpendicular_vector_l244_24416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_symmetry_l244_24444

open Real

-- Define the polar curve
noncomputable def polar_curve (θ : ℝ) : ℝ := 4 * sin (θ - π/3)

-- Define the symmetry line
noncomputable def symmetry_line : ℝ := 5 * π / 6

-- Theorem statement
theorem polar_curve_symmetry :
  ∀ θ : ℝ, polar_curve (2 * symmetry_line - θ) = polar_curve θ :=
by
  intro θ
  -- The actual proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_symmetry_l244_24444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_p_l244_24423

theorem existence_of_p (n : ℕ) : ∃ p : ℕ, 4 * p + 5 = (3^n)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_p_l244_24423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_of_eleven_students_l244_24468

theorem average_age_of_eleven_students
  (total_students : Nat)
  (average_age_all : ℝ)
  (group_of_three : Nat)
  (average_age_three : ℝ)
  (age_fifteenth : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : group_of_three = 3)
  (h4 : average_age_three = 14)
  (h5 : age_fifteenth = 7) :
  let remaining_students := total_students - group_of_three - 1
  let total_age_all := average_age_all * total_students
  let total_age_three := average_age_three * group_of_three
  let total_age_remaining := total_age_all - total_age_three - age_fifteenth
  let average_age_remaining := total_age_remaining / remaining_students
  average_age_remaining = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_of_eleven_students_l244_24468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l244_24495

/-- Definition of the ellipse E -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the quadrilateral area -/
def quadrilateral_area (a b : ℝ) : ℝ :=
  4 * a * b

/-- Definition of the minimum distance to foci -/
def min_distance_to_foci (a c : ℝ) : ℝ :=
  a - c

/-- Definition of the line l -/
def line (k m : ℝ) (x : ℝ) : ℝ :=
  k * x + m

/-- Definition of the distance |OP| -/
noncomputable def distance_OP (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

/-- Main theorem -/
theorem ellipse_properties (a b c : ℝ) :
  a > b ∧ b > 0 ∧
  quadrilateral_area a b = 4 * Real.sqrt 3 ∧
  min_distance_to_foci a c = 1 →
  (∀ x y, ellipse a b x y ↔ ellipse 2 (Real.sqrt 3) x y) ∧
  (∀ k m, 0 ≤ k ∧ k ≤ Real.sqrt 3 →
    let P := {p : ℝ × ℝ | ellipse 2 (Real.sqrt 3) p.1 p.2 ∧ p.2 = line k m p.1}
    (∀ p ∈ P, Real.sqrt 3 ≤ distance_OP p.1 p.2) ∧
    (∀ p ∈ P, distance_OP p.1 p.2 ≤ Real.sqrt 95 / 5)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l244_24495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_divides_triple_product_larger_divisor_divides_sixty_sixty_is_largest_divisor_l244_24487

/-- A Pythagorean triple is a tuple of three natural numbers (a, b, c) such that a² + b² = c² -/
def isPythagoreanTriple (a b c : ℕ) : Prop := a * a + b * b = c * c

/-- The product of the side lengths of a Pythagorean triple -/
def tripleProduct (a b c : ℕ) : ℕ := a * b * c

/-- 60 divides the product of the side lengths of any Pythagorean triple -/
theorem sixty_divides_triple_product (a b c : ℕ) (h : isPythagoreanTriple a b c) :
  60 ∣ tripleProduct a b c := by
  sorry

/-- If n > 60 and n divides the product of the side lengths of all Pythagorean triples,
    then n divides 60 -/
theorem larger_divisor_divides_sixty (n : ℕ) (hn : n > 60)
  (h : ∀ a b c : ℕ, isPythagoreanTriple a b c → n ∣ tripleProduct a b c) :
  n ∣ 60 := by
  sorry

/-- 60 is the largest integer that divides the product of the side lengths
    of any Pythagorean triple -/
theorem sixty_is_largest_divisor :
  ∀ n : ℕ, (∀ a b c : ℕ, isPythagoreanTriple a b c → n ∣ tripleProduct a b c) → n ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_divides_triple_product_larger_divisor_divides_sixty_sixty_is_largest_divisor_l244_24487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_simplify_fraction_main_result_l244_24426

theorem decimal_to_fraction : ∃ (n d : ℕ), d ≠ 0 ∧ (n / d : ℚ) = 45 / 99 :=
by sorry

theorem simplify_fraction : (5 : ℚ) / 11 = 45 / 99 :=
by sorry

theorem main_result : ∃ (n d : ℕ), d ≠ 0 ∧ (n / d : ℚ) = (5 : ℚ) / 11 ∧ d = 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_simplify_fraction_main_result_l244_24426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_zero_point_l244_24414

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

-- Define the floor function g
noncomputable def g (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem floor_of_zero_point :
  ∃ (x₀ : ℝ), (x₀ > 0 ∧ f x₀ = 0) ∧ g x₀ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_zero_point_l244_24414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l244_24435

theorem triangle_area (K : Real) (KL : Real) (area : Real) : 
  K = 45 * Real.pi / 180 →  -- Convert 45° to radians
  KL = 20 → 
  area = 100 → 
  area = (KL^2 / 4) * Real.sin K := by
    sorry

#check triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l244_24435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_minimizes_cost_l244_24422

/-- Annual purchase amount in tons -/
noncomputable def annual_purchase : ℝ := 400

/-- Freight cost per trip in yuan -/
noncomputable def freight_cost_per_trip : ℝ := 40000

/-- Storage cost coefficient in yuan per ton -/
noncomputable def storage_cost_coeff : ℝ := 40000

/-- Total annual cost as a function of purchase amount per trip -/
noncomputable def total_annual_cost (x : ℝ) : ℝ :=
  (annual_purchase / x) * freight_cost_per_trip + storage_cost_coeff * x

/-- Optimal purchase amount per trip -/
noncomputable def optimal_purchase : ℝ := 20

theorem optimal_purchase_minimizes_cost :
  ∀ x > 0, total_annual_cost optimal_purchase ≤ total_annual_cost x := by
  sorry

#check optimal_purchase_minimizes_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_minimizes_cost_l244_24422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l244_24415

/-- The function f(x) = ln(x + √(x^2 + 1)) -/
noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1))

/-- Theorem stating the minimum value of 1/a + 1/b given the conditions -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : f (2 * a) + f (b - 1) = 0) :
  (1 / a + 1 / b) ≥ 2 * Real.sqrt 2 + 3 := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l244_24415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_floor_f_l244_24482

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (3^x) / (3^x + 1) - 1/3

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Theorem statement
theorem range_of_floor_f :
  (∀ x : ℝ, floor (f x) ∈ ({-1, 0} : Set ℤ)) ∧
  (∀ y ∈ ({-1, 0} : Set ℤ), ∃ x : ℝ, floor (f x) = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_floor_f_l244_24482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l244_24484

theorem remainder_problem (s t : ℕ) 
  (h1 : s % 6 = 2)
  (h2 : s > t)
  (h3 : (s - t) % 6 = 5) :
  t % 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l244_24484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l244_24437

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (x^2 - 4)

theorem range_of_f : Set.range f = Set.Ici 2 ∪ Set.Iic (-2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l244_24437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_iff_lambda_gt_neg_three_l244_24450

def a (n : ℕ) (l : ℝ) : ℝ := n^2 + l * n

theorem increasing_sequence_iff_lambda_gt_neg_three (l : ℝ) :
  (∀ n : ℕ, a (n + 1) l > a n l) ↔ l > -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_iff_lambda_gt_neg_three_l244_24450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_selling_price_l244_24478

/-- Calculates the selling price of an item given its cost price and loss percentage. -/
def selling_price (cost_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  cost_price - (cost_price * (loss_percentage / 100))

/-- Theorem stating that a radio with a cost price of 1500 and a loss of 16% has a selling price of 1260. -/
theorem radio_selling_price :
  selling_price 1500 16 = 1260 := by
  -- Unfold the definition of selling_price
  unfold selling_price
  -- Simplify the arithmetic
  simp [mul_div_assoc, sub_eq_iff_eq_add]
  -- Check that 1500 - (1500 * (16 / 100)) = 1260
  norm_num

#eval selling_price 1500 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_selling_price_l244_24478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_count_l244_24432

/-- Represents a point in the hexagonal lattice --/
structure LatticePoint where
  x : ℝ
  y : ℝ

/-- Represents the expanded hexagonal lattice --/
structure HexagonalLattice where
  points : List LatticePoint
  center : LatticePoint

/-- Function to count equilateral triangles in the lattice --/
def countEquilateralTriangles (lattice : HexagonalLattice) : ℕ :=
  sorry

/-- Theorem stating the number of equilateral triangles in the specific lattice --/
theorem equilateral_triangles_count (lattice : HexagonalLattice) :
  (lattice.points.length = 12) →
  (∀ p q, p ∈ lattice.points → q ∈ lattice.points → p ≠ q → 
    (p.x - q.x)^2 + (p.y - q.y)^2 = 1 ∨ (p.x - q.x)^2 + (p.y - q.y)^2 = 3) →
  countEquilateralTriangles lattice = 28 :=
by
  sorry

#check equilateral_triangles_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_count_l244_24432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l244_24466

noncomputable section

/-- The curve function f(x) = ln x + x^2 -/
def f (x : ℝ) : ℝ := Real.log x + x^2

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := 1/x + 2*x

/-- The slope of the tangent line at x = 1 -/
noncomputable def k : ℝ := f' 1

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 1)

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 3*x - y - 2 = 0

theorem tangent_line_equation :
  tangent_line point.1 point.2 ∧
  ∀ x y, y = f x → (x = point.1 ∧ y = point.2) ∨ (tangent_line x y ↔ 
    ∃ t, x = point.1 + t ∧ y = point.2 + k * t) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l244_24466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_expression_l244_24497

/-- Sequence a_n -/
def a (n : ℕ) : ℝ := 2 * n - 1

/-- Sum of first n terms of a_n -/
def S (n : ℕ) : ℝ := n^2

/-- Sequence b_n -/
def b (n : ℕ) : ℝ := n^2 - n + 17

/-- The expression to be minimized -/
noncomputable def f (n : ℕ) : ℝ := b n / Real.sqrt (S n)

theorem minimize_expression :
  ∀ n : ℕ, n ≥ 1 → f 4 ≤ f n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_expression_l244_24497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leahs_age_l244_24449

def possible_ages : List Nat := [26, 29, 31, 33, 35, 39, 41, 43, 45, 50]

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def count_lower (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (λ g ↦ g < age)).length

def count_off_by_one (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (λ g ↦ g = age - 1 ∨ g = age + 1)).length

theorem leahs_age :
  ∃ age ∈ possible_ages,
    is_prime age ∧
    count_lower age possible_ages > possible_ages.length / 2 ∧
    count_off_by_one age possible_ages = 3 ∧
    age = 41 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leahs_age_l244_24449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winnie_the_pooh_apples_l244_24429

theorem winnie_the_pooh_apples :
  ∀ (winnie_rate tigger_rate : ℕ) 
    (winnie_time tigger_time total_apples : ℕ),
  winnie_rate * 7 = tigger_rate * 4 →
  winnie_time = 80 →
  tigger_time = 50 →
  winnie_rate * winnie_time + tigger_rate * tigger_time = total_apples →
  total_apples = 2010 →
  winnie_rate * winnie_time = 960 := by
  sorry

#check winnie_the_pooh_apples

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winnie_the_pooh_apples_l244_24429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_areas_sum_eq_16pi_div_7_l244_24413

/-- The sum of the areas of an infinite series of circles, where the radius of each circle
    is 3/4 of the previous one, starting with a radius of 1 inch. -/
noncomputable def circle_areas_sum : ℝ :=
  let r := 3 / 4  -- Common ratio of the geometric sequence of radii
  let a := Real.pi      -- First term of the geometric sequence of areas (π * 1²)
  a / (1 - r^2)   -- Sum of the infinite geometric series

/-- Theorem stating that the sum of the areas of the circles is 16π/7 -/
theorem circle_areas_sum_eq_16pi_div_7 : circle_areas_sum = 16 * Real.pi / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_areas_sum_eq_16pi_div_7_l244_24413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l244_24452

-- Define the complex number z
def z : ℂ := (1 - 2*Complex.I) * (3 + Complex.I)

-- State the theorem
theorem magnitude_of_z : Complex.abs z = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l244_24452
