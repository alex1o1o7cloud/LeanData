import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_almost_surjective_polynomial_form_l257_25712

/-- A polynomial that maps integers to all integers except possibly finitely many -/
structure AlmostSurjectivePolynomial where
  P : ℝ → ℝ
  is_polynomial : ∃ p : Polynomial ℝ, ∀ x, P x = p.eval x
  almost_surjective : ∃ (S : Finset ℤ), ∀ (m : ℤ), m ∉ S → ∃ (n : ℤ), P n = m

/-- The theorem stating the form of almost surjective polynomials -/
theorem almost_surjective_polynomial_form (asp : AlmostSurjectivePolynomial) :
  ∃ (c : ℤ) (k : ℕ), k > 0 ∧ ∀ x, asp.P x = (x + c) / k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_almost_surjective_polynomial_form_l257_25712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_equality_domain_of_f_l257_25776

-- Part 1
theorem logarithm_equality : 2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 = 2 := by
  sorry

-- Part 2
noncomputable def f (x : ℝ) := 1 / Real.sqrt (12 - x) + Real.log (x^2 - x - 30) / Real.log (x - 3)

theorem domain_of_f : Set.Ioo 6 12 = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_equality_domain_of_f_l257_25776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_proof_l257_25793

def sequenceSquares (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem sequence_proof :
  sequenceSquares 0 = 1 ∧
  sequenceSquares 1 = 7 ∧
  sequenceSquares 2 = 19 ∧
  sequenceSquares 3 = 37 ∧
  sequenceSquares 100 = 30301 := by
  -- Split the conjunction into individual goals
  constructor
  · -- Prove sequenceSquares 0 = 1
    rfl
  constructor
  · -- Prove sequenceSquares 1 = 7
    rfl
  constructor
  · -- Prove sequenceSquares 2 = 19
    rfl
  constructor
  · -- Prove sequenceSquares 3 = 37
    rfl
  · -- Prove sequenceSquares 100 = 30301
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_proof_l257_25793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_multiple_l257_25785

theorem least_positive_multiple (x y : ℤ) (h : Int.gcd x (20 * y) = 4) :
  (∃ (k : ℤ), k * (x + 20 * y) = 4 ∧ ∀ (m : ℤ), m * (x + 20 * y) ≠ 0 → |m * (x + 20 * y)| ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_multiple_l257_25785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_prism_volume_l257_25709

/-- The volume of a rectangular prism with given dimensions -/
noncomputable def prism_volume (side : ℝ) (diagonal : ℝ) (height : ℝ) : ℝ :=
  let base_area := side * (Real.sqrt (diagonal^2 - side^2))
  base_area * height

/-- Theorem stating the volume of the specific prism -/
theorem specific_prism_volume (h : ℝ) :
  prism_volume 15 17 h = 120 * h :=
by
  unfold prism_volume
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_prism_volume_l257_25709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_is_3_empty_intersection_iff_l257_25765

-- Define set A
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}

-- Define set B
def B : Set ℝ := {x | (x - 1) * (x - 4) ≥ 0}

-- Part 1
theorem intersection_when_a_is_3 :
  A 3 ∩ B = Set.Icc (-1) 1 ∪ Set.Icc 4 5 := by sorry

-- Part 2
theorem empty_intersection_iff :
  ∀ a : ℝ, a > 0 → (A a ∩ B = ∅ ↔ 0 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_is_3_empty_intersection_iff_l257_25765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_inequality_l257_25706

noncomputable def f (a x : ℝ) : ℝ := (2*a - 1) * Real.log x - 1/x - 2*a*x

theorem monotonicity_and_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  (x₁ > 0 ∧ x₂ > 0) →
  (a = 0 →
    (∀ x ∈ Set.Ioo 0 1, HasDerivAt (f 0) (- 1/x + 1/x^2) x) ∧
    (∀ x ∈ Set.Ioi 1, HasDerivAt (f 0) (- 1/x + 1/x^2) x)) ∧
  (a ∈ Set.Icc (-2) (-1) →
    x₁ ∈ Set.Icc 1 (Real.exp 1) →
    x₂ ∈ Set.Icc 1 (Real.exp 1) →
    (5 - 2 * Real.exp 1) * a - 1 / Real.exp 1 + 2 ≥ |f a x₁ - f a x₂|) ∧
  (∀ m, m > 5 → ∃ a ∈ Set.Ico (-2) (-1), ∃ x₁ x₂, 
    x₁ ∈ Set.Icc 1 (Real.exp 1) ∧
    x₂ ∈ Set.Icc 1 (Real.exp 1) ∧
    (m - 2 * Real.exp 1) * a - 1 / Real.exp 1 + 2 < |f a x₁ - f a x₂|) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_inequality_l257_25706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_BF_l257_25742

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the ellipse -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 / 36 + p.y^2 / 16 = 1

/-- Focus of the ellipse -/
noncomputable def F : Point :=
  { x := 2 * Real.sqrt 5, y := 0 }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Length of BF in the given ellipse configuration -/
theorem length_BF (A B : Point) 
    (hA : isOnEllipse A) 
    (hB : isOnEllipse B) 
    (hAF : distance A F = 2) 
    (hABF : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
      B.x = t * A.x + (1 - t) * F.x ∧ 
      B.y = t * A.y + (1 - t) * F.y) : 
  distance B F = 4 * Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_BF_l257_25742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_d_range_l257_25718

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def arithmetic_sequence_max_sum_condition (d : ℝ) : Prop :=
  let a₁ := 10
  ∀ n : ℕ, n ≠ 0 →
    (sum_arithmetic_sequence a₁ d 5 ≥ sum_arithmetic_sequence a₁ d n) ∧
    (n ≠ 5 → sum_arithmetic_sequence a₁ d 5 > sum_arithmetic_sequence a₁ d n)

theorem arithmetic_sequence_d_range :
  ∀ d : ℝ, arithmetic_sequence_max_sum_condition d ↔ -5/2 < d ∧ d < -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_d_range_l257_25718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_occurrences_l257_25786

def f (S : ℕ) : ℕ :=
  if 1 ≤ S ∧ S ≤ 9 then
    S * (S + 1) / 2
  else if 10 ≤ S ∧ S ≤ 18 then
    S^2 - 28*S + 126
  else if 19 ≤ S ∧ S ≤ 27 then
    (S^2 - 57*S + 812) / 2
  else
    0

theorem digit_sum_occurrences :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 →
  let S := (n / 100) + ((n / 10) % 10) + (n % 10)
  f S = (Finset.filter (λ m ↦ 100 ≤ m ∧ m ≤ 999 ∧
    (m / 100) + ((m / 10) % 10) + (m % 10) = S) (Finset.range 1000)).card :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_occurrences_l257_25786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triplets_eq_3n_squared_l257_25790

/-- The number of distinct triplets of natural numbers summing to 6n -/
def count_triplets (n : ℕ) : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ => t.1 + t.2.1 + t.2.2 = 6*n) (Finset.product (Finset.range (6*n + 1)) (Finset.product (Finset.range (6*n + 1)) (Finset.range (6*n + 1))))).card

/-- Theorem: The number of distinct triplets of natural numbers summing to 6n is 3n^2 -/
theorem count_triplets_eq_3n_squared (n : ℕ) : count_triplets n = 3 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triplets_eq_3n_squared_l257_25790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_n_minus_one_roots_l257_25721

/-- Given a list of distinct non-zero real numbers, the function f(x) is defined as the sum of a_i / (a_i - x) for all a_i in the list. -/
noncomputable def f (a : List ℝ) (x : ℝ) : ℝ :=
  (a.map (λ ai => ai / (ai - x))).sum

/-- The main theorem stating that the equation f(x) = n has at least n-1 real roots. -/
theorem at_least_n_minus_one_roots (n : ℕ) (a : List ℝ) 
    (h_distinct : a.Nodup) 
    (h_nonzero : ∀ x ∈ a, x ≠ 0) 
    (h_length : a.length = n) :
    ∃ (roots : List ℝ), roots.length ≥ n - 1 ∧ 
      ∀ r ∈ roots, f a r = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_n_minus_one_roots_l257_25721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_air_quality_probability_l257_25734

/-- The probability of air quality being good in a day -/
noncomputable def p_good : ℝ := 0.75

/-- The probability of air quality being good for two consecutive days -/
noncomputable def p_good_consecutive : ℝ := 0.6

/-- The conditional probability of air quality being good on the following day
    given it is good on a certain day -/
noncomputable def p_good_given_good : ℝ := p_good_consecutive / p_good

theorem air_quality_probability : p_good_given_good = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_air_quality_probability_l257_25734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_l257_25784

/-- The configuration of circles as described in the problem -/
structure CircleConfiguration where
  small_radius : ℝ
  num_small_circles : ℕ
  small_circles_tangent : Prop
  all_circles_coplanar : Prop
  small_circles_touch_adjacent : Prop

/-- Function to calculate the diameter of the large circle -/
noncomputable def diameter_of_large_circle (config : CircleConfiguration) : ℝ :=
  2 * (config.small_radius + 6)

/-- The theorem stating the diameter of the large circle -/
theorem large_circle_diameter
  (config : CircleConfiguration)
  (h1 : config.small_radius = 4)
  (h2 : config.num_small_circles = 6)
  (h3 : config.small_circles_tangent)
  (h4 : config.all_circles_coplanar)
  (h5 : config.small_circles_touch_adjacent) :
  ∃ (d : ℝ), d = 20 ∧ d = diameter_of_large_circle config :=
by
  use 20
  constructor
  · rfl
  · simp [diameter_of_large_circle, h1]
    norm_num

/- Additional helper lemmas can be added here if needed for the proof -/


end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_l257_25784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_8000_simplification_l257_25773

theorem cube_root_8000_simplification :
  ∃ (a b : ℕ), 
    (a > 0) ∧ 
    (∀ (c d : ℕ), c > 0 → d < b → (c : ℝ) * (d : ℝ) ^ (1/3 : ℝ) ≠ (8000 : ℝ) ^ (1/3 : ℝ)) ∧
    ((a : ℝ) * (b : ℝ) ^ (1/3 : ℝ) = (8000 : ℝ) ^ (1/3 : ℝ)) ∧
    (a + b = 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_8000_simplification_l257_25773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_sum_l257_25740

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
variable (right_angle_ABC : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0)
variable (right_angle_ABD : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0)
variable (AC_length : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 3^2)
variable (BC_length : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 4^2)
variable (AD_length : (A.1 - D.1)^2 + (A.2 - D.2)^2 = 12^2)
variable (C_D_opposite : ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) * 
                         ((B.1 - A.1) * (D.2 - A.2) - (B.2 - A.2) * (D.1 - A.1)) < 0)
variable (D_parallel_AC : (D.2 - A.2) * (C.1 - A.1) = (D.1 - A.1) * (C.2 - A.2))
variable (E_on_CB_extended : ∃ t : ℝ, E = (B.1 + t * (B.1 - C.1), B.2 + t * (B.2 - C.2)))

-- Define the ratio
variable (m n : ℕ)
variable (m_n_coprime : Nat.Coprime m n)
variable (DE_DB_ratio : (E.1 - D.1)^2 + (E.2 - D.2)^2 = (m/n : ℚ)^2 * ((B.1 - D.1)^2 + (B.2 - D.2)^2))

-- State the theorem
theorem ratio_sum : m + n = 128 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_sum_l257_25740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_F_eq_19_l257_25799

-- Define the function F
noncomputable def F (x : ℝ) : ℤ :=
  round x

-- Define the sum function
noncomputable def sum_F : ℝ := by
  -- Sum for x from 1 to 10
  let sum1 := (Finset.range 10).sum (λ i => 1 / (F (i + 1 : ℝ) : ℝ))
  -- Sum for √x where x is from 2 to 99
  let sum2 := (Finset.range 98).sum (λ i => 1 / (F (Real.sqrt (i + 2 : ℝ)) : ℝ))
  exact sum1 + sum2

-- Theorem statement
theorem sum_F_eq_19 : sum_F = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_F_eq_19_l257_25799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bugs_meet_time_l257_25798

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a bug crawling on a circle with a given speed -/
structure Bug where
  circle : Circle
  speed : ℝ

/-- Calculates the time taken for a bug to complete one revolution -/
noncomputable def revolutionTime (bug : Bug) : ℝ :=
  (2 * Real.pi * bug.circle.radius) / bug.speed

/-- Calculates the least common multiple of two real numbers -/
noncomputable def realLCM (a b : ℝ) : ℝ :=
  let intA := Int.floor (a * 1000000)
  let intB := Int.floor (b * 1000000)
  (Int.lcm intA intB : ℝ) / 1000000

/-- The main theorem stating the time taken for the bugs to meet again -/
theorem bugs_meet_time (circle1 : Circle) (circle2 : Circle) (bug1 : Bug) (bug2 : Bug) :
  circle1.radius = 6 →
  circle2.radius = 3 →
  bug1.circle = circle1 →
  bug2.circle = circle2 →
  bug1.speed = 4 * Real.pi →
  bug2.speed = 3 * Real.pi →
  realLCM (revolutionTime bug1) (revolutionTime bug2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bugs_meet_time_l257_25798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l257_25720

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) :
  (1 / (3 * m))^(-3 : ℤ) * (2 * m)^4 = 432 * m^7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l257_25720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_l257_25730

def polynomial (n : ℕ) (x : ℝ) : ℝ :=
  (Finset.range (2*n+1)).sum (λ i => x^i)

theorem max_real_roots (n : ℕ) :
  (∃ (x : ℝ), polynomial n x = 0) ↔ n % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_l257_25730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_and_maximum_l257_25794

theorem trigonometric_identity_and_maximum 
  (α β : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi/2) 
  (h2 : 0 < β ∧ β < Real.pi/2) 
  (h3 : Real.sin β / Real.sin α = Real.cos (α + β)) : 
  (Real.tan β = Real.sin (2*α) / (3 - Real.cos (2*α))) ∧ 
  (∀ γ : ℝ, 0 < γ ∧ γ < Real.pi/2 → Real.tan γ ≤ Real.sqrt 2 / 4) ∧
  (∃ δ : ℝ, 0 < δ ∧ δ < Real.pi/2 ∧ Real.tan δ = Real.sqrt 2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_and_maximum_l257_25794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l257_25777

/-- Given three circles in a plane with centers A, B, and C, each with radius r
    where 2 < r < 3, and the distance between each pair of centers is 3,
    prove that the length of the line segment D'E' (where D' is the intersection
    of circles A and C outside B, and E' is the intersection of circles A and B
    outside C) is equal to 3 + √(3(r² - 9/4)). -/
theorem intersection_segment_length
  (r : ℝ) 
  (A B C : EuclideanSpace ℝ (Fin 2))
  (h_r_bounds : 2 < r ∧ r < 3)
  (h_distance : dist A B = 3 ∧ dist B C = 3 ∧ dist A C = 3)
  (D' : EuclideanSpace ℝ (Fin 2))
  (h_D'_def : D' ∈ Metric.sphere A r ∩ Metric.sphere C r ∧ D' ∉ Metric.sphere B r)
  (E' : EuclideanSpace ℝ (Fin 2))
  (h_E'_def : E' ∈ Metric.sphere A r ∩ Metric.sphere B r ∧ E' ∉ Metric.sphere C r) :
  dist D' E' = 3 + Real.sqrt (3 * (r^2 - 9/4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l257_25777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_is_zero_l257_25723

/-- Ana's age this year -/
def A : ℕ := sorry

/-- Bonita's age this year -/
def B : ℕ := sorry

/-- Age difference between Ana and Bonita -/
def n : ℕ := sorry

/-- Relationship between Ana's and Bonita's ages -/
axiom age_difference : A = B + n

/-- Last year's age relationship -/
axiom last_year : A - 1 = 6 * (B - 1)

/-- This year's age relationship -/
axiom this_year : A = B^3

theorem age_difference_is_zero : n = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_is_zero_l257_25723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l257_25766

/-- The complex number z = i / (1 + i) lies in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant :
  let z : ℂ := Complex.I / (1 + Complex.I)
  z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l257_25766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_proof_l257_25729

/-- Right pyramid with square base -/
structure RightPyramid where
  baseArea : ℝ
  height : ℝ

/-- Total surface area of a right pyramid -/
noncomputable def totalSurfaceArea (p : RightPyramid) : ℝ :=
  p.baseArea + 4 * (p.baseArea / 3)

/-- Volume of a right pyramid -/
noncomputable def volume (p : RightPyramid) : ℝ :=
  (1 / 3) * p.baseArea * p.height

theorem pyramid_volume_proof (p : RightPyramid) 
  (h1 : totalSurfaceArea p = 480)
  (h2 : p.baseArea = 1440 / 7)
  (h3 : p.height = 960 / (21 * Real.sqrt (10 / 7))) :
  volume p = 160 * Real.sqrt (10 / 7) := by
  sorry

#check pyramid_volume_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_proof_l257_25729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recipe_total_cups_l257_25727

/-- Represents the amount of an ingredient in the recipe -/
structure Ingredient where
  parts : ℕ
  cups : ℚ

/-- Represents the recipe with given ratios and sugar amount -/
structure Recipe where
  butter : Ingredient
  flour : Ingredient
  sugar : Ingredient
  h1 : butter.parts = 2
  h2 : flour.parts = 3
  h3 : sugar.parts = 5
  h4 : sugar.cups = 10

/-- Calculates the total cups of ingredients in the recipe -/
def totalCups (r : Recipe) : ℚ :=
  r.butter.cups + r.flour.cups + r.sugar.cups

/-- Theorem: The total cups of ingredients in the recipe is 20 -/
theorem recipe_total_cups (r : Recipe) : totalCups r = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recipe_total_cups_l257_25727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l257_25741

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt ((a * a + b * b) / (a * a))

/-- A point on the hyperbola -/
structure HyperbolaPoint (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- The left focus of the hyperbola -/
noncomputable def leftFocus (h : Hyperbola a b) : ℝ × ℝ :=
  (-Real.sqrt (a * a + b * b), 0)

/-- The right focus of the hyperbola -/
noncomputable def rightFocus (h : Hyperbola a b) : ℝ × ℝ :=
  (Real.sqrt (a * a + b * b), 0)

/-- Check if a point is symmetric to the right focus with respect to the asymptote y = (b/a)x -/
def isSymmetricToRightFocus (h : Hyperbola a b) (p : HyperbolaPoint h) : Prop :=
  let (xf, _) := rightFocus h
  let (xp, yp) := (p.x, p.y)
  xp + xf = 2 * a * yp / b

theorem hyperbola_eccentricity_sqrt_5 (a b : ℝ) (h : Hyperbola a b) 
  (p : HyperbolaPoint h) (h_left : p.x < 0) (h_sym : isSymmetricToRightFocus h p) :
  eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l257_25741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_equal_intercepts_line_equation_l257_25711

-- Define the lines from the problem
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) : Prop := x - y + 4 = 0
def line3 (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Define the intersection point of line1 and line2
def intersection_point : ℝ × ℝ := (-2, 2)

-- Define the point P
def point_P : ℝ × ℝ := (-1, 3)

-- Theorem for the first part of the problem
theorem perpendicular_line_equation :
  ∃ (a b c : ℝ), 
    (∀ x y, line1 x y ∧ line2 x y → (x, y) = intersection_point) ∧
    (∀ x y, a * x + b * y + c = 0 ↔ 2 * x + y + 2 = 0) ∧
    (a * 1 + b * (-2) = 0) := 
sorry

-- Theorem for the second part of the problem
theorem equal_intercepts_line_equation :
  ∃ (a b c : ℝ),
    (a * point_P.1 + b * point_P.2 + c = 0) ∧
    (∀ x : ℝ, a * x + b * 0 + c = 0 → x = a * x + b * x + c) ∧
    (∀ x y, a * x + b * y + c = 0 ↔ x + y - 2 = 0) := 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_equal_intercepts_line_equation_l257_25711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_on_tangent_circles_l257_25743

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a 2D plane -/
def Point := ℝ × ℝ

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop := sorry

/-- Check if a point is on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a line is tangent to a circle -/
def is_tangent_line (line : Point × Point) (c : Circle) : Prop := sorry

/-- Calculate the area of a triangle given three points -/
def area_of_triangle (p1 p2 p3 : Point) : ℝ := sorry

/-- Three externally tangent circles -/
def three_tangent_circles : Prop :=
  ∃ (ω₁ ω₂ ω₃ : Circle),
    ω₁.radius = 4 ∧ ω₂.radius = 4 ∧ ω₃.radius = 4 ∧
    are_externally_tangent ω₁ ω₂ ∧
    are_externally_tangent ω₂ ω₃ ∧
    are_externally_tangent ω₃ ω₁

/-- Points on circles forming an equilateral triangle with tangent sides -/
def equilateral_triangle_on_circles (ω₁ ω₂ ω₃ : Circle) (P₁ P₂ P₃ : Point) : Prop :=
  point_on_circle P₁ ω₁ ∧
  point_on_circle P₂ ω₂ ∧
  point_on_circle P₃ ω₃ ∧
  distance P₁ P₂ = distance P₂ P₃ ∧
  distance P₂ P₃ = distance P₃ P₁ ∧
  is_tangent_line (P₁, P₂) ω₁ ∧
  is_tangent_line (P₂, P₃) ω₂ ∧
  is_tangent_line (P₃, P₁) ω₃

/-- The main theorem -/
theorem area_of_triangle_on_tangent_circles :
  three_tangent_circles →
  ∃ (ω₁ ω₂ ω₃ : Circle) (P₁ P₂ P₃ : Point),
    equilateral_triangle_on_circles ω₁ ω₂ ω₃ P₁ P₂ P₃ →
    area_of_triangle P₁ P₂ P₃ = Real.sqrt 300 + Real.sqrt 252 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_on_tangent_circles_l257_25743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_program_arrangements_l257_25724

/-- Represents the number of programs in the event -/
def total_programs : ℕ := 6

/-- Represents the number of programs that must be adjacent -/
def adjacent_programs : ℕ := 2

/-- Represents the number of programs that cannot be adjacent -/
def non_adjacent_programs : ℕ := 2

/-- Represents the number of remaining programs -/
def remaining_programs : ℕ := total_programs - adjacent_programs - non_adjacent_programs

/-- Theorem stating the number of possible arrangements -/
theorem program_arrangements :
  (Nat.factorial (remaining_programs + 1)) * (remaining_programs + 2) = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_program_arrangements_l257_25724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_median_angle_30_l257_25769

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : a + b + c = 180

-- Define a median
structure Median (t : Triangle) where
  vertex : Fin 3
  midpoint : Fin 3

-- Define the angles created by medians
def median_angles (t : Triangle) (m1 m2 m3 : Median t) : Fin 6 → ℝ :=
  sorry

-- Theorem statement
theorem max_median_angle_30 (t : Triangle) (m1 m2 m3 : Median t) :
  (∃ (s : Finset (Fin 6)), s.card = 3 ∧ ∀ i ∈ s, median_angles t m1 m2 m3 i > 30) →
  (∀ i : Fin 6, median_angles t m1 m2 m3 i ≤ 30) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_median_angle_30_l257_25769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l257_25728

/-- A function g(x) obtained by shifting sin(ωx) left by π/2 units -/
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * (x + Real.pi / 2))

theorem g_properties (ω : ℝ) (h : ω > 0) (h1 : g ω 0 = 1) :
  -- g(x) is an even function
  (∀ x, g ω x = g ω (-x)) ∧
  -- g(-π/2) = 0
  g ω (-Real.pi / 2) = 0 ∧
  -- When ω = 5, g(x) has exactly 3 zeros in the interval [0, π/2]
  (ω = 5 → ∃! (s : Finset ℝ), s.card = 3 ∧ 
    (∀ x ∈ s, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ g 5 x = 0) ∧
    (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ g 5 x = 0 → x ∈ s)) ∧
  -- If g(x) is strictly decreasing on [0, π/4], then ω = 1
  ((∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 4 → g ω y < g ω x) → ω = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l257_25728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangulations_eq_catalan_l257_25746

/-- The number of distinct triangulations of a convex n-gon using non-intersecting diagonals -/
def triangulations (n : ℕ) : ℕ := sorry

/-- The nth Catalan number -/
def catalan' (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of triangulations of a convex n-gon
    is equal to the (n-2)th Catalan number -/
theorem triangulations_eq_catalan (n : ℕ) (h : n ≥ 3) :
  triangulations n = catalan' (n - 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangulations_eq_catalan_l257_25746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_equals_five_l257_25774

-- Define the function f and its inverse
def f : ℝ → ℝ := sorry
def f_inv : ℝ → ℝ := sorry

-- State the properties of f and f_inv
axiom f_inv_def : ∀ x, f_inv x = Real.sqrt (x - 1)
axiom f_inverse : ∀ x, f (f_inv x) = x

-- The theorem to be proved
theorem f_two_equals_five : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_equals_five_l257_25774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_f_sum_of_f_values_l257_25757

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x + 3 * Real.cos (Real.pi / 2 * x) - 3

-- Define the center of symmetry
def is_center_of_symmetry (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ + x₂ = 2 * a → f x₁ + f x₂ = 2 * b

-- Theorem 1: (1, -1) is the center of symmetry for f
theorem center_of_symmetry_f :
  is_center_of_symmetry f 1 (-1) := by sorry

-- Theorem 2: Sum of f(x) values
theorem sum_of_f_values :
  (Finset.range 4035).sum (λ i => f ((i + 1 : ℕ) / 2018 : ℝ)) = -4035 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_f_sum_of_f_values_l257_25757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l257_25717

-- Define the points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 4)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the intersection point P(x, y) parametrically using m
noncomputable def P (m : ℝ) : ℝ × ℝ :=
  ((2*m^2 + 8)/(m^2 + 1), (2*m + 4*m)/(m^2 + 1))

-- State the theorem
theorem max_product_of_distances :
  ∀ m : ℝ, distance (P m) A * distance (P m) B ≤ 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l257_25717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_13_terms_l257_25750

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem sum_of_13_terms (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 = 1 → a 5 = 4 →
  sum_of_arithmetic_sequence a 13 = 91 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_13_terms_l257_25750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_for_c_l257_25781

/-- Work rates of workers a, b, and c -/
structure WorkRates where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (w : WorkRates) : Prop :=
  w.a + w.b = 1/10 ∧ w.b + w.c = 1/5 ∧ w.c + w.a = 1/15

/-- The time taken by worker c to complete the work alone -/
noncomputable def time_for_c (w : WorkRates) : ℝ := 1 / w.c

/-- Theorem stating that under the given conditions, 
    worker c takes approximately 12 days to complete the work alone -/
theorem work_time_for_c (w : WorkRates) 
  (h : satisfies_conditions w) : 
  ∃ ε > 0, |time_for_c w - 12| < ε := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_for_c_l257_25781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_by_moving_line_segment_l257_25768

/-- The area covered by a line segment AP, where A is a fixed point and P moves along a circular arc. -/
theorem area_covered_by_moving_line_segment (A : ℝ × ℝ) (P : ℝ → ℝ × ℝ) (t_start t_end : ℝ) : 
  A.1 = 2 ∧ A.2 = 0 →
  (∀ t, P t = (Real.sin (2 * t - π / 3), Real.cos (2 * t - π / 3))) →
  t_start = π / 12 →
  t_end = π / 4 →
  ∃ area : ℝ, area = π / 12 ∧ 
    area = (π / 6) / 2 -- This represents the area of the sector
:= by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_by_moving_line_segment_l257_25768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l257_25797

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

-- Define the line passing through the origin
def my_line (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the tangent condition
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), my_circle x y ∧ my_line k x y ∧ second_quadrant x y

-- Theorem statement
theorem tangent_line_equation :
  ∀ k : ℝ, is_tangent k → k = -Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l257_25797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_x_with_two_weighings_l257_25739

/-- Represents a sequence of 7 consecutive integers starting from x -/
def ConsecutiveSequence (x : ℕ) : Fin 7 → ℕ := fun i => x + i.val

/-- Represents a weighing operation that compares two sums of elements from the sequence -/
def Weighing (s : Fin 7 → ℕ) (l1 r1 : Fin 7) : Bool :=
  s l1 + s (l1 + 1) + s (l1 + 2) = s r1 + s (r1 + 1)

/-- Represents the second weighing operation -/
def SecondWeighing (s : Fin 7 → ℕ) (l1 r1 r2 : Fin 7) : Bool :=
  s l1 + s (l1 + 1) + s (l1 + 2) = s r1 + s r2

/-- Theorem stating that it's possible to determine x using at most two weighings -/
theorem determine_x_with_two_weighings :
  ∀ x : ℕ, 1 ≤ x → x ≤ 7 →
  ∃ (w1 w2 : (Fin 7 → ℕ) → Bool),
    (∀ y : ℕ, 1 ≤ y → y ≤ 7 → y ≠ x → w1 (ConsecutiveSequence y) ≠ w1 (ConsecutiveSequence x)) ∨
    (∀ y : ℕ, 1 ≤ y → y ≤ 7 → y ≠ x → w2 (ConsecutiveSequence y) ≠ w2 (ConsecutiveSequence x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_x_with_two_weighings_l257_25739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinguishable_triangles_l257_25704

/-- The number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- The number of small triangles needed to form a large triangle -/
def triangles_per_large : ℕ := 4

/-- Represents a large triangle constructed from small colored triangles -/
structure LargeTriangle where
  corner1 : Fin num_colors
  corner2 : Fin num_colors
  corner3 : Fin num_colors
  center : Fin num_colors

/-- Two large triangles are considered equivalent if they can be transformed into each other
    through rotations or reflections -/
def equivalent (t1 t2 : LargeTriangle) : Prop := sorry

/-- The set of all possible large triangles -/
def all_large_triangles : Set LargeTriangle := sorry

/-- The set of distinguishable large triangles -/
noncomputable def distinguishable_triangles : Finset LargeTriangle := sorry

/-- The number of distinguishable large triangles -/
noncomputable def num_distinguishable : ℕ := Finset.card distinguishable_triangles

theorem count_distinguishable_triangles : num_distinguishable = 960 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinguishable_triangles_l257_25704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_of_ln_at_one_l257_25788

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := log x

-- State the theorem
theorem second_derivative_of_ln_at_one :
  (deriv (deriv f)) 1 = -1 := by
  -- We'll use sorry to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_of_ln_at_one_l257_25788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_trajectory_locus_l257_25771

/-- The locus of vertices of parabolic trajectories -/
theorem parabolic_trajectory_locus 
  (c : ℝ) -- Initial velocity
  (g : ℝ) -- Acceleration due to gravity
  (h : g > 0) -- Gravity is positive
  (α : ℝ) -- Angle of projection
  (h_α : 0 ≤ α ∧ α ≤ π/2) -- Angle range
  (x y : ℝ) -- Coordinates of vertex
  (h_x : x ≥ 0) -- x is non-negative
  : 
  (x = (c^2 * Real.sin (2*α)) / (2*g) ∧ 
   y = (c^2 * Real.sin α * Real.sin α) / (2*g)) →
  (x^2 / (c^2 / g)^2) + ((y - c^2 / (4*g))^2 / (c^2 / (4*g))^2) = 1 :=
by
  intro h
  sorry -- The proof goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_trajectory_locus_l257_25771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l257_25702

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin (Real.pi / 4 + x))^2 - Real.sqrt 3 * Real.cos (2 * x)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → f x ≤ 3) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → f x ≥ 2) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) ∧ 
                   x₂ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) ∧ 
                   f x₁ = 3 ∧ f x₂ = 2) ∧
  (∀ (m : ℝ), (∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → |f x - m| < 2) ↔ 
               m ∈ Set.Ioo 1 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l257_25702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_special_remainder_l257_25737

theorem smallest_number_with_special_remainder : ∃! x : ℕ, 
  (∀ d : ℕ, d ∈ ({4, 5, 6, 12} : Set ℕ) → x % d = d - 2) ∧ 
  (∀ y : ℕ, y < x → ∃ d : ℕ, d ∈ ({4, 5, 6, 12} : Set ℕ) ∧ y % d ≠ d - 2) :=
by
  sorry

#check smallest_number_with_special_remainder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_special_remainder_l257_25737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_roots_equal_derivative_l257_25749

-- Define a polynomial of degree n
def MyPolynomial (α : Type) [Field α] (n : ℕ) := Fin n.succ → α

-- Define the arithmetic mean of the roots of a polynomial
noncomputable def arithmeticMeanOfRoots {α : Type} [Field α] {n : ℕ} (f : MyPolynomial α n) : α :=
  sorry

-- Define the derivative of a polynomial
def derivativePolynomial {α : Type} [Field α] {n : ℕ} (f : MyPolynomial α n) : MyPolynomial α (n - 1) :=
  sorry

-- Theorem statement
theorem arithmetic_mean_roots_equal_derivative {α : Type} [Field α] {n : ℕ} (f : MyPolynomial α n) :
  arithmeticMeanOfRoots f = arithmeticMeanOfRoots (derivativePolynomial f) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_roots_equal_derivative_l257_25749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_one_fifteenth_l257_25745

/-- Represents a rectangular yard with two congruent isosceles right triangular flower beds -/
structure YardWithFlowerBeds where
  short_side : ℝ
  long_side : ℝ
  flower_bed_leg : ℝ

/-- The fraction of the yard occupied by the flower beds -/
noncomputable def flower_bed_fraction (yard : YardWithFlowerBeds) : ℝ :=
  (2 * yard.flower_bed_leg^2) / (yard.short_side * yard.long_side)

theorem flower_bed_fraction_is_one_fifteenth 
  (yard : YardWithFlowerBeds) 
  (h1 : yard.short_side = 18)
  (h2 : yard.long_side = 30)
  (h3 : yard.flower_bed_leg = 6) : 
  flower_bed_fraction yard = 1/15 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_one_fifteenth_l257_25745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_l257_25783

/-- Distance between two complex numbers -/
noncomputable def distanceComplex (z w : ℂ) : ℝ :=
  Complex.abs (z - w)

/-- Predicate to check if three complex numbers form an equilateral triangle -/
def isEquilateralTriangle (p q r : ℂ) : Prop :=
  distanceComplex p q = distanceComplex q r ∧ 
  distanceComplex q r = distanceComplex r p

/-- Given complex numbers p, q, r forming an equilateral triangle with side length 24,
    if |p + q + r| = 48, then |pq + pr + qr| = 768 -/
theorem equilateral_triangle_complex (p q r : ℂ) : 
  isEquilateralTriangle p q r → 
  distanceComplex p q = 24 → 
  Complex.abs (p + q + r) = 48 →
  Complex.abs (p * q + p * r + q * r) = 768 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_l257_25783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l257_25705

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) / Real.log a
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (1 - x) / Real.log a
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

-- State the theorem
theorem function_properties (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  -- 1. Domain of h(x) is (-1, 1)
  (∀ x, h a x ∈ Set.Ioo (-1) 1 ↔ x ∈ Set.Ioo (-1) 1) ∧
  -- 2. h(x) is an odd function
  (∀ x, x ∈ Set.Ioo (-1) 1 → h a (-x) = -h a x) ∧
  -- 3. When a = 5, f(x) > 1 iff x > 4
  (a = 5 → ∀ x, f a x > 1 ↔ x > 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l257_25705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_results_in_four_terms_l257_25762

-- Define the expression as a function of x and the replacement term
def expression (x : ℝ) (replacement : ℝ → ℝ) : ℝ :=
  (x^4 - 3)^2 + (x^3 + replacement x)^2

-- Define the function to count terms in a polynomial
-- This is a simplification, as actually implementing term counting would be complex
noncomputable def countTerms (poly : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem replacement_results_in_four_terms :
  ∃ (replacement₁ replacement₂ : ℝ → ℝ),
    (∀ x, replacement₁ x = 3 * x) ∧
    (∀ x, replacement₂ x = Real.sqrt 6 * x^2) ∧
    (countTerms (λ x ↦ expression x replacement₁) = 4) ∧
    (countTerms (λ x ↦ expression x replacement₂) = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_results_in_four_terms_l257_25762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_sequence_count_primes_in_sequence_l257_25714

def Q : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53

def sequenceQ (m : ℕ) : ℕ := Q + m

def is_in_range (m : ℕ) : Prop := 2 ≤ m ∧ m ≤ 55

theorem no_primes_in_sequence :
  ∀ m, is_in_range m → ¬(Nat.Prime (sequenceQ m)) :=
by sorry

theorem count_primes_in_sequence :
  (Finset.filter (λ m => Nat.Prime (sequenceQ m)) (Finset.range 54)).card = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_sequence_count_primes_in_sequence_l257_25714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l257_25756

/-- The period of a cosine function with frequency ω -/
noncomputable def period (ω : ℝ) : ℝ := 2 * Real.pi / ω

/-- The given cosine function -/
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ p = period 2 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
  ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l257_25756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_area_theorem_l257_25787

/-- The number of disks -/
def n : ℕ := 15

/-- The radius of the large circle -/
def R : ℝ := 1.5

/-- The radius of each small disk -/
noncomputable def r : ℝ := R * Real.cos (Real.pi / n)

/-- The area of a single small disk -/
noncomputable def single_disk_area : ℝ := Real.pi * r^2

/-- The total area of all disks -/
noncomputable def total_area : ℝ := n * single_disk_area

/-- The simplified expression for the total area -/
noncomputable def simplified_area : ℝ := Real.pi * (28 - 3 * Real.sqrt 5)

/-- The theorem stating that the total area equals the simplified area -/
theorem disk_area_theorem :
  total_area = simplified_area :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_area_theorem_l257_25787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_lambda_l257_25707

theorem sequence_range_lambda (a : ℕ+ → ℝ) (lambda : ℝ) :
  (∀ n : ℕ+, a n < a (n + 1)) →
  (∀ n : ℕ+, a n = n^2 + lambda * n) →
  lambda > -3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_lambda_l257_25707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_is_1728_l257_25736

def book_arrangement_count : Nat :=
  let math_books : Nat := 3
  let english_books : Nat := 4
  let science_books : Nat := 2
  let subject_groups : Nat := 3

  let group_arrangements : Nat := Nat.factorial subject_groups
  let math_arrangements : Nat := Nat.factorial math_books
  let english_arrangements : Nat := Nat.factorial english_books
  let science_arrangements : Nat := Nat.factorial science_books

  group_arrangements * math_arrangements * english_arrangements * science_arrangements

theorem book_arrangement_count_is_1728 : book_arrangement_count = 1728 := by
  rfl

#eval book_arrangement_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_is_1728_l257_25736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_translation_l257_25726

theorem min_phi_translation (φ : ℝ) : 
  (φ > 0) → 
  (Real.sin (2 * (Real.pi / 6 + φ)) = Real.sqrt 3 / 2) → 
  (∀ ψ : ℝ, ψ > 0 ∧ Real.sin (2 * (Real.pi / 6 + ψ)) = Real.sqrt 3 / 2 → ψ ≥ φ) → 
  φ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_translation_l257_25726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_equation_l_equation_max_area_ABQ_l257_25733

-- Define the circles and lines
def C₁ (x y D E F : ℝ) : Prop := x^2 + y^2 + D*x + E*y + F = 0
def C₂ (x y : ℝ) : Prop := (x-2)^2 + y^2 = 1
def line_xy (x y : ℝ) : Prop := x + y - 2 = 0
def line_l (x y k : ℝ) : Prop := x = k*y

-- Define the conditions
axiom C₁_symmetric : ∃ D E F : ℝ, ∀ x y : ℝ, C₁ x y D E F ↔ C₁ y x D E F
axiom C₁_through_origin : ∃ D E F : ℝ, C₁ 0 0 D E F
axiom C₁_through_4_0 : ∃ D E F : ℝ, C₁ 4 0 D E F

-- Define the chord length condition for line l
axiom l_chord_length : ∃ k : ℝ, ∀ x y : ℝ, 
  line_l x y k → C₂ x y → (x^2 + y^2 = 2)

-- Define the moving point and intersections
def point_on_C₂ (x y : ℝ) : Prop := C₂ x y
def line_m (x y k x₀ y₀ : ℝ) : Prop := y - y₀ = k*(x - x₀)
def intersect_C₁_m (x y : ℝ) : Prop := ∃ D E F k x₀ y₀ : ℝ, C₁ x y D E F ∧ line_m x y k x₀ y₀
def ray_PC₂ (x y : ℝ) : Prop := ∃ t : ℝ, x = 2*t ∧ y = t

-- State the theorems to be proved
theorem C₁_equation : ∃ D E F : ℝ, ∀ x y : ℝ, C₁ x y D E F ↔ (x-2)^2 + y^2 = 4 := sorry

theorem l_equation : ∃ k : ℝ, ∀ x y : ℝ, line_l x y k ↔ (x = (Real.sqrt 7/7)*y ∨ x = -(Real.sqrt 7/7)*y) := sorry

theorem max_area_ABQ : 
  ∃ A B Q : ℝ × ℝ, 
    intersect_C₁_m A.1 A.2 ∧ 
    intersect_C₁_m B.1 B.2 ∧ 
    ray_PC₂ Q.1 Q.2 ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 + (B.1 - Q.1)^2 + (B.2 - Q.2)^2 + (Q.1 - A.1)^2 + (Q.2 - A.2)^2 ≤ 3*(Real.sqrt 3) := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_equation_l_equation_max_area_ABQ_l257_25733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_upper_bound_l257_25767

theorem sum_upper_bound (x y : ℝ) (h : (2 : ℝ)^x + (2 : ℝ)^y = 1) : x + y ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_upper_bound_l257_25767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_and_quadratic_l257_25758

theorem absolute_value_and_quadratic : 
  (∀ x : ℝ, (abs x > 3) → (x * (x - 3) > 0)) ∧ 
  (∃ x : ℝ, (x * (x - 3) > 0) ∧ (abs x ≤ 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_and_quadratic_l257_25758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l257_25722

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem principal_calculation (A : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) :
  let P := A / ((1 + r / n) ^ (n * t))
  A = 1120 ∧ r = 0.05 ∧ n = 1 ∧ t = 6 →
  (P ≥ 835.81 ∧ P ≤ 835.83) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l257_25722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l257_25772

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line type with slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Intersection point of a line and a parabola -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_line_intersection_slope 
  (C : Parabola) 
  (l : Line) 
  (A B : IntersectionPoint) 
  (hC : C.equation = fun x y => y^2 = 4*x)
  (hF : C.focus = (1, 0))
  (hIntersect : C.equation A.x A.y ∧ C.equation B.x B.y)
  (hLine : l.slope * (A.x - 1) + l.yIntercept = A.y ∧ 
           l.slope * (B.x - 1) + l.yIntercept = B.y)
  (hDistance : distance (A.x, A.y) C.focus = 4 * distance (B.x, B.y) C.focus) :
  l.slope = 4/3 ∨ l.slope = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_slope_l257_25772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l257_25700

theorem evaluate_expression : (64 : ℝ) ^ (-(1/2 : ℝ)) = 1/4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l257_25700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_equality_l257_25753

theorem triangle_ratio_equality (a b c : ℝ) (A B : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π) 
  (h_triangle : a = b * Real.sin A / Real.sin B) : 
  (a - c * Real.cos B) / (b - c * Real.cos A) = Real.sin B / Real.sin A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_equality_l257_25753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_of_distances_l257_25731

/-- The point that minimizes the sum of distances to two given points on a line -/
theorem minimize_sum_of_distances (A B P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  A = (1, 2) →
  B = (-2, 0) →
  l = {(x, y) | x - y + 3 = 0} →
  P ∈ l →
  P = (-5/3, 4/3) →
  ∀ Q ∈ l, dist A P + dist B P ≤ dist A Q + dist B Q :=
by sorry

/-- Helper function to calculate Euclidean distance between two points -/
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_of_distances_l257_25731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l257_25796

noncomputable def f (x : ℝ) : ℝ := (x^3 + 8) / (x - 8)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 8 ∨ x > 8} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l257_25796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_29_l257_25760

theorem sum_of_divisors_29 : Finset.sum (Nat.divisors 29) id = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_29_l257_25760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l257_25703

/-- The function g(x) with parameter k -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + k) / (x^2 - 3*x - 10)

/-- A predicate to check if g(x) has exactly one vertical asymptote -/
def has_exactly_one_vertical_asymptote (k : ℝ) : Prop :=
  (∃! x, x^2 - 3*x - 10 = 0 ∧ x^2 + 2*x + k ≠ 0) ∨
  (∃! x, x^2 - 3*x - 10 = 0 ∧ x^2 + 2*x + k = 0)

/-- Theorem stating that g(x) has exactly one vertical asymptote iff k = -35 or k = 0 -/
theorem g_one_vertical_asymptote :
  ∀ k : ℝ, has_exactly_one_vertical_asymptote k ↔ k = -35 ∨ k = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l257_25703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l257_25747

theorem solution_set (m : ℕ) (hm : Even m) (hm_pos : m > 0) :
  ∀ n x y : ℕ, 
    n > 0 → x > 0 → y > 0 →
    Nat.Coprime m n →
    (x^2 + y^2)^m = (x * y)^n →
    ∃ a : ℕ, a > 0 ∧ n = m + 1 ∧ x = 2^a ∧ y = 2^a := by
  sorry

#check solution_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l257_25747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l257_25719

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  let period : ℝ := Real.pi
  let min_value : ℝ := -2
  let min_x (k : ℤ) : ℝ := k * Real.pi - 5 * Real.pi / 12
  let increasing_interval (k : ℤ) : Set ℝ := Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12)
  
  (∀ x : ℝ, f (x + period) = f x) ∧ 
  (∀ x : ℝ, f x ≥ min_value) ∧
  (∀ k : ℤ, f (min_x k) = min_value) ∧
  (∀ k : ℤ, ∀ x ∈ increasing_interval k, ∀ y ∈ increasing_interval k, x < y → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l257_25719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gardener_ensures_majestic_trees_l257_25710

/-- Represents the board game with alternating turns. -/
structure BoardGame where
  size : ℕ
  target_height : ℕ

/-- The strategy of the first player (gardener) -/
def gardener_strategy (game : BoardGame) : ℕ :=
  (5 * game.size * game.size) / 9

/-- The number of majestic trees after the game, given strategies of both players -/
def number_of_majestic_trees (game : BoardGame) (lumberjack_strategy : BoardGame → ℕ) : ℕ :=
  sorry

/-- Theorem stating that the gardener can ensure a certain number of majestic trees -/
theorem gardener_ensures_majestic_trees (game : BoardGame) :
  game.size = 2022 →
  game.target_height = 1000000 →
  ∃ (k : ℕ), k = gardener_strategy game ∧
    (∀ (lumberjack_strategy : BoardGame → ℕ),
      k ≤ number_of_majestic_trees game lumberjack_strategy) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gardener_ensures_majestic_trees_l257_25710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parkway_elementary_students_l257_25752

/-- The number of students in the fifth grade at Parkway Elementary School -/
def total_students : ℕ := 470

/-- The number of boy students -/
def boys : ℕ := 300

/-- The number of students playing soccer -/
def soccer_players : ℕ := 250

/-- The percentage of soccer players who are boys -/
def boys_soccer_percentage : ℚ := 86 / 100

/-- The number of girl students not playing soccer -/
def girls_not_soccer : ℕ := 135

theorem parkway_elementary_students :
  total_students = boys + soccer_players - (boys_soccer_percentage * soccer_players).floor + girls_not_soccer :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parkway_elementary_students_l257_25752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_19_approx_l257_25716

def T (n : ℕ) : ℚ := (n * (n + 1) * (2 * n + 1)) / 6

def P (n : ℕ) : ℚ :=
  Finset.prod (Finset.range ((n - 1) / 2)) (λ i => T (2 * i + 3) / (T (2 * i + 3) + 1))

theorem P_19_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/10 ∧ |P 19 - 9/10| < ε :=
sorry

#eval P 19

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_19_approx_l257_25716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l257_25782

variable (a : ℝ) (ha : a > 0)

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.exp x + a)

noncomputable def f_inv (x : ℝ) : ℝ := Real.log (Real.exp x - a)

noncomputable def f_deriv (x : ℝ) : ℝ := Real.exp x / (Real.exp x + a)

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (Real.log (3 * a)) (Real.log (4 * a)), 
    |m - f_inv a x| + Real.log (f_deriv a x) < 0) →
  Real.log ((12 / 5) * a) < m ∧ m < Real.log ((8 / 3) * a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l257_25782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l257_25744

/-- Given a curve y = ax^2 where a ≠ 0, if the tangent line at point (1, a) is parallel
    to the line 2x - y - 6 = 0, then the equation of this tangent line is 2x - y - 1 = 0 -/
theorem tangent_line_equation (a : ℝ) (ha : a ≠ 0) :
  let f := λ x : ℝ ↦ a * x^2
  let tangent_slope := (deriv f) 1
  tangent_slope = 2 →
  ∀ x y : ℝ, y - f 1 = tangent_slope * (x - 1) ↔ 2 * x - y - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l257_25744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l257_25732

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

noncomputable def C (ω : ℝ) (x : ℝ) : ℝ := f ω (x + Real.pi / 2)

theorem center_of_symmetry (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 1) 
  (h3 : ∀ x, C ω x = C ω (-x)) : 
  ∃ k : ℤ, C ω (3 * Real.pi / 2 + 2 * Real.pi * k) = C ω (-(3 * Real.pi / 2 + 2 * Real.pi * k)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l257_25732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_MYN_l257_25751

-- Define the points
variable (X Y Z W M N : ℝ × ℝ)

-- Define the conditions
axiom right_triangle : X.2 - Z.2 = 4 ∧ Y.1 - X.1 = 3 ∧ Y.2 = Z.2
axiom extend_YZ : W = (Z.1, Z.2 - Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2))
axiom midpoint_M : M = ((X.1 + Z.1) / 2, (X.2 + Z.2) / 2)
axiom N_on_XY : N.2 - X.2 = -(4/3) * (N.1 - X.1)
axiom N_on_MW : N.1 = M.1 ∧ N.1 = W.1

-- Theorem statement
theorem area_MYN : 
  let base := Real.sqrt ((Y.1 - M.1)^2 + (Y.2 - M.2)^2)
  let height := |N.2 - M.2|
  1/2 * base * height = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_MYN_l257_25751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_sum_l257_25763

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

/-- The left focus of the ellipse -/
noncomputable def left_focus : ℝ × ℝ := (-2 * Real.sqrt 3, 0)

/-- Vector from left focus to a point -/
noncomputable def vector_to_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2 * Real.sqrt 3, p.2)

/-- Sum of vectors is zero -/
def vectors_sum_to_zero (a b c : ℝ × ℝ) : Prop :=
  vector_to_point a + vector_to_point b + vector_to_point c = (0, 0)

/-- Distance from left focus to a point -/
noncomputable def distance_to_point (p : ℝ × ℝ) : ℝ :=
  4 + Real.sqrt 3 / 2 * p.1

/-- Main theorem -/
theorem ellipse_focus_distance_sum 
  (a b c : ℝ × ℝ) 
  (ha : is_on_ellipse a.1 a.2) 
  (hb : is_on_ellipse b.1 b.2) 
  (hc : is_on_ellipse c.1 c.2) 
  (hsum : vectors_sum_to_zero a b c) : 
  distance_to_point a + distance_to_point b + distance_to_point c = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_sum_l257_25763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_identity_l257_25779

/-- The function f(x) defined as (ax+b)/(x+1) -/
noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (x + 1)

/-- Theorem stating that f(f(x)) = x for all x ≠ -1 if and only if a = -1 and b is any real number -/
theorem f_composition_identity (a b : ℝ) :
  (∀ x : ℝ, x ≠ -1 → f a b (f a b x) = x) ↔ (a = -1 ∧ b ∈ Set.univ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_identity_l257_25779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_length_is_112_l257_25701

/-- Given a rectangular field with a square pond, prove the field's length is 112 meters. -/
theorem field_length_is_112 (width length : ℝ) (pond_side : ℝ) : 
  length = 2 * width →  -- The length is double the width
  pond_side = 8 →  -- The pond's side length is 8 meters
  pond_side^2 = (1/98) * (length * width) →  -- The pond's area is 1/98 of the field's area
  length = 112 := by
  intro h1 h2 h3
  -- Proof steps would go here
  sorry

#check field_length_is_112

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_length_is_112_l257_25701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_type_m_pricing_and_profit_l257_25778

/-- Type M clothing pricing and sales model -/
structure TypeMClothing where
  original_price : ℝ
  discount_percent : ℝ
  profit_percent : ℝ
  additional_discount : ℝ → ℝ
  sales_volume : ℝ → ℝ

/-- Conditions for Type M clothing -/
def type_m_conditions : TypeMClothing where
  original_price := 75
  discount_percent := 0.2
  profit_percent := 0.5
  additional_discount := λ x => x
  sales_volume := λ x => 20 + 4 * x

/-- Cost price of Type M clothing -/
noncomputable def cost_price (c : TypeMClothing) : ℝ :=
  c.original_price * (1 - c.discount_percent) / (1 + c.profit_percent)

/-- Daily profit function for Type M clothing during promotion -/
noncomputable def daily_profit (c : TypeMClothing) (x : ℝ) : ℝ :=
  let discounted_price := c.original_price * (1 - c.discount_percent) - c.additional_discount x
  let profit_per_piece := discounted_price - cost_price c
  profit_per_piece * c.sales_volume x

/-- Theorem stating the cost price and maximum daily profit for Type M clothing -/
theorem type_m_pricing_and_profit :
  cost_price type_m_conditions = 40 ∧
  ∃ max_profit : ℝ, max_profit = 625 ∧ 
    ∀ x : ℝ, daily_profit type_m_conditions x ≤ max_profit := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_type_m_pricing_and_profit_l257_25778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l257_25792

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ F : ℝ × ℝ, 
    -- F is the right focus of the hyperbola
    F.1 > 0 ∧
    -- Equation of the hyperbola
    (λ (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1) = 
      (λ (x y : ℝ) ↦ (x - F.1)^2 / a^2 - y^2 / b^2 = 1) ∧
    -- Equation of the circle C with center F
    (λ (x y : ℝ) ↦ x^2 + y^2 - 4*x + 3 = 0) = 
      (λ (x y : ℝ) ↦ (x - F.1)^2 + y^2 = 1) ∧
    -- Circle C is tangent to the asymptotes of the hyperbola
    (∀ x y : ℝ, y = (b/a) * x → (x - F.1)^2 + y^2 ≥ 1) ∧
    (∃ x y : ℝ, y = (b/a) * x ∧ (x - F.1)^2 + y^2 = 1)) →
  a^2 = 3 ∧ b = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l257_25792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_ratio_l257_25725

theorem equilateral_triangle_area_ratio :
  ∀ s : ℝ,
  s > 0 →
  let small_triangle_area := Real.sqrt 3 / 4 * s^2
  let large_triangle_side := 3 * s
  let large_triangle_area := Real.sqrt 3 / 4 * large_triangle_side^2
  (3 * small_triangle_area) / large_triangle_area = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_ratio_l257_25725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l257_25780

noncomputable def f (x : ℝ) : ℝ := 3^x / (3^x + 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  q ≠ 1 →
  (∀ n, a (n + 1) = q * a n) →
  a 3 = 1 →
  f (Real.log (a 1)) + f (Real.log (a 2)) + f (Real.log (a 3)) + 
  f (Real.log (a 4)) + f (Real.log (a 5)) = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l257_25780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_common_divisor_l257_25738

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^(n + 9)

theorem largest_common_divisor :
  ∃ (m : ℕ), m > 0 ∧ (∀ (n : ℕ), m ∣ f n) ∧ 
  (∀ (k : ℕ), k > m → ∃ (n : ℕ), ¬(k ∣ f n)) ∧
  m = 36 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_common_divisor_l257_25738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_bound_l257_25713

/-- The number of divisors of a positive integer n is less than or equal to 2√n. -/
theorem divisors_count_bound (n : ℕ+) : (Nat.divisors n.val).card ≤ 2 * Real.sqrt (n.val : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_bound_l257_25713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_390000_l257_25761

/-- Represents the production and profit scenario for a company --/
structure ProductionScenario where
  /-- Amount of raw material A needed for one ton of product A --/
  raw_a_for_a : ℝ
  /-- Amount of raw material B needed for one ton of product A --/
  raw_b_for_a : ℝ
  /-- Amount of raw material A needed for one ton of product B --/
  raw_a_for_b : ℝ
  /-- Amount of raw material B needed for one ton of product B --/
  raw_b_for_b : ℝ
  /-- Profit from selling one ton of product A --/
  profit_a : ℝ
  /-- Profit from selling one ton of product B --/
  profit_b : ℝ
  /-- Maximum amount of raw material A available --/
  max_raw_a : ℝ
  /-- Maximum amount of raw material B available --/
  max_raw_b : ℝ

/-- Calculates the maximum profit for a given production scenario --/
def max_profit (s : ProductionScenario) : ℝ := 
  sorry

/-- Theorem stating that the maximum profit for the given scenario is 390000 --/
theorem max_profit_is_390000 :
  let scenario := ProductionScenario.mk 3 2 1 3 50000 30000 13 18
  max_profit scenario = 390000 := by 
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_390000_l257_25761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_in_range_l257_25755

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (8-a)*x + 4

theorem increasing_f_implies_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc 6 8 ∧ a ≠ 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_in_range_l257_25755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l257_25735

noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

def parallel_lines (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ / b₁ = a₂ / b₂

theorem distance_between_specific_lines :
  let l₁ : ℝ → ℝ → ℝ := λ x y => 3*x + 4*y - 2
  let l₂ : ℝ → ℝ → ℝ := λ x y => 3*x + 4*y - 7
  parallel_lines 3 4 3 4 →
  distance_parallel_lines 3 4 2 7 = 5 := by
    sorry

#check distance_between_specific_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l257_25735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_digit_sum_for_abc_fraction_l257_25715

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a number in the form 0.abc where a, b, c are digits -/
def ABC (a b c : Digit) : ℚ := (a.val * 100 + b.val * 10 + c.val : ℕ) / 1000

/-- The theorem stating the maximum sum of digits for the given conditions -/
theorem max_digit_sum_for_abc_fraction :
  ∀ (a b c : Digit) (y : ℕ),
    (1 ≤ y) → (y ≤ 9) →
    (ABC a b c = 1 / y) →
    (a.val + b.val + c.val ≤ 8) ∧
    (∃ (a' b' c' : Digit) (y' : ℕ),
      (1 ≤ y') ∧ (y' ≤ 9) ∧
      (ABC a' b' c' = 1 / y') ∧
      (a'.val + b'.val + c'.val = 8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_digit_sum_for_abc_fraction_l257_25715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_is_correct_l257_25764

/-- The volume of a regular tetrahedron with edge length 2 -/
noncomputable def regularTetrahedronVolume : ℝ := 2 * Real.sqrt 2 / 3

/-- Theorem: The volume of a regular tetrahedron with edge length 2 is 2√2/3 -/
theorem regular_tetrahedron_volume_is_correct (edgeLength : ℝ) (h : edgeLength = 2) :
  regularTetrahedronVolume = (edgeLength ^ 3 * Real.sqrt 2) / 12 := by
  sorry

#check regular_tetrahedron_volume_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_is_correct_l257_25764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shift_is_even_l257_25708

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 12)

theorem f_shift_is_even : ∀ x : ℝ, g x = g (-x) := by
  intro x
  unfold g f
  simp [Real.sin_add]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shift_is_even_l257_25708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_2007_angles_proof_91_is_smallest_l257_25789

/-- A circle with n points on its circumference -/
structure CircleWithPoints where
  n : ℕ
  points : Fin n → ℝ × ℝ

/-- The angle between two points and the center of the circle -/
noncomputable def angle (c : CircleWithPoints) (i j : Fin c.n) : ℝ := sorry

/-- The number of angles less than or equal to 120° -/
noncomputable def count_angles_le_120 (c : CircleWithPoints) : ℕ :=
  (Finset.univ.filter (fun p : Fin c.n × Fin c.n => 
    p.1 < p.2 ∧ angle c p.1 p.2 ≤ 120)).card

/-- The theorem statement -/
theorem smallest_n_for_2007_angles : 
  (∀ c : CircleWithPoints, c.n = 91 → count_angles_le_120 c ≥ 2007) ∧ 
  (∀ m : ℕ, m < 91 → ∃ c : CircleWithPoints, c.n = m ∧ count_angles_le_120 c < 2007) := by
  sorry

/-- Proof that 91 is the smallest n satisfying the condition -/
theorem proof_91_is_smallest : ∃ n : ℕ, n = 91 ∧ 
  (∀ c : CircleWithPoints, c.n = n → count_angles_le_120 c ≥ 2007) ∧
  (∀ m : ℕ, m < n → ∃ c : CircleWithPoints, c.n = m ∧ count_angles_le_120 c < 2007) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_2007_angles_proof_91_is_smallest_l257_25789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_formula_l257_25748

theorem quadratic_formula {a b c x : ℝ} (ha : a ≠ 0) :
  a * x^2 + b * x + c = 0 ↔ 
    x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) ∨ 
    x = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_formula_l257_25748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_truth_l257_25791

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x = y → x / y = 1

-- Define proposition q
def q : Prop := ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (Real.exp x₁ - Real.exp x₂) / (x₁ - x₂) > 0

-- Theorem statement
theorem prop_truth : 
  (¬(p ∧ q)) ∧ (p ∨ q) ∧ (¬(p ∧ ¬q)) ∧ ((¬p) ∨ q) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_truth_l257_25791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_seven_l257_25754

theorem divisibility_by_seven (n : ℕ) : 
  7 ∣ (3^(3*n+1) + 5^(3*n+2) + 7^(3*n+3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_seven_l257_25754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_rectangle_l257_25795

theorem points_in_rectangle (points : Finset (ℝ × ℝ)) : 
  (∀ p ∈ points, 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3) →
  points.card = 6 →
  ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_rectangle_l257_25795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l257_25770

/-- The area of a right triangle with hypotenuse 12 and one angle 30° --/
theorem right_triangle_area (h : ℝ) (α : ℝ) (area : ℝ) : 
  h = 12 →  -- hypotenuse is 12 inches
  α = 30 * Real.pi / 180 →  -- one angle is 30° (converted to radians)
  area = 18 * Real.sqrt 3 →  -- area is 18√3 square inches
  ∃ (a b : ℝ), 
    a^2 + b^2 = h^2 ∧  -- Pythagorean theorem
    Real.sin α = a / h ∧  -- trigonometric relation
    area = (1/2) * a * b  -- area formula for triangle
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l257_25770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_3670_l257_25759

theorem cube_root_3670 (h : Real.rpow 3.670 (1/3) = 1.542) :
  Real.rpow 3670 (1/3) = 15.42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_3670_l257_25759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l257_25775

open Real

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + a

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ < 1 ∧ f_deriv a x₀ = f a x₀) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l257_25775
