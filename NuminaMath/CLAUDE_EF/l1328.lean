import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_theorem_l1328_132897

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3 * x + 2 else x^2 + a * x

-- State the theorem
theorem piecewise_function_theorem (a : ℝ) :
  f a (f a 0) = 4 * a → a = 2 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_theorem_l1328_132897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_determination_l1328_132860

noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 13/2

theorem interval_determination (a b : ℝ) (h1 : a ≤ b) 
  (h2 : ∀ x ∈ Set.Icc a b, f x ≥ 2*a) 
  (h3 : ∀ x ∈ Set.Icc a b, f x ≤ 2*b) 
  (h4 : ∃ x ∈ Set.Icc a b, f x = 2*a) 
  (h5 : ∃ x ∈ Set.Icc a b, f x = 2*b) : 
  a = 1 ∧ b = 3 := by
  sorry

#check interval_determination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_determination_l1328_132860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_conditions_l1328_132877

noncomputable def isPossibleA (a : ℝ) : Prop :=
  a = -1 ∨ a = 2 ∨ a = 1/2 ∨ a = 3 ∨ a = 1/3

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def isMonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

noncomputable def powerFunction (a : ℝ) : ℝ → ℝ := fun x ↦ x^a

theorem power_function_conditions (a : ℝ) :
  isPossibleA a →
  isOddFunction (powerFunction a) →
  isMonoIncreasing (powerFunction a) →
  (a = 1/3 ∨ a = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_conditions_l1328_132877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_at_three_implies_a_equals_36_l1328_132873

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 * x + a / x

/-- The theorem stating that if f(x) takes its minimum at x = 3, then a = 36 -/
theorem minimum_at_three_implies_a_equals_36 (a : ℝ) (h1 : a > 0) :
  (∀ x > 0, f a x ≥ f a 3) → a = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_at_three_implies_a_equals_36_l1328_132873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1328_132869

theorem trigonometric_equation_solution :
  ∀ x : ℝ, (|Real.sin x| - Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 * Real.sqrt 3 ↔
  (∃ k : ℤ, x = 2 * Real.pi / 3 + 2 * k * Real.pi ∨
            x = -2 * Real.pi / 3 + 2 * k * Real.pi ∨
            x = -Real.pi / 6 + 2 * k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1328_132869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_containers_equivalent_to_1000_original_l1328_132896

/-- Represents a cylindrical container with internal diameter and height --/
structure Container where
  diameter : ℝ
  height : ℝ

/-- Calculates the volume of a cylindrical container --/
noncomputable def containerVolume (c : Container) : ℝ :=
  Real.pi * (c.diameter / 2) ^ 2 * c.height

/-- The original container --/
def originalContainer : Container :=
  { diameter := 18, height := 10 }

/-- The new container --/
def newContainer : Container :=
  { diameter := originalContainer.diameter - 6, height := originalContainer.height + 2 }

/-- Theorem stating the number of new containers equivalent to 1000 original containers --/
theorem new_containers_equivalent_to_1000_original :
  (1000 * containerVolume originalContainer) / containerVolume newContainer = 1875 := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_containers_equivalent_to_1000_original_l1328_132896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fern_ballet_slippers_l1328_132881

noncomputable def price_high_heels : ℝ := 60
noncomputable def price_ballet_slippers : ℝ := (2/3) * price_high_heels
noncomputable def total_cost : ℝ := 260

theorem fern_ballet_slippers :
  ∃ (x : ℕ), (price_high_heels + x * price_ballet_slippers = total_cost) ∧ x = 5 := by
  use 5
  constructor
  · simp [price_high_heels, price_ballet_slippers, total_cost]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fern_ballet_slippers_l1328_132881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1328_132803

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b
  k : b > 0

/-- Represents a point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The theorem statement -/
theorem ellipse_properties (e : Ellipse) 
  (P A B : PointOnEllipse e) 
  (h1 : A ≠ B)
  (h2 : P ≠ A ∧ P ≠ B)
  (h3 : ∃ (m : ℝ), A.x = m * A.y ∧ B.x = m * B.y) -- line through origin
  (h4 : (P.y - A.y) / (P.x - A.x) * (P.y - B.y) / (P.x - B.x) = -1/4) :
  (eccentricity e = Real.sqrt 3 / 2) ∧
  (∀ (k : ℝ), let c := Real.sqrt (e.a^2 - e.b^2)
               let x1 := (4*k^2*Real.sqrt 3*e.b + Real.sqrt ((4*k^2+1)*(12*k^2-4)*e.b^2)) / (2*(4*k^2+1))
               let x2 := (4*k^2*Real.sqrt 3*e.b - Real.sqrt ((4*k^2+1)*(12*k^2-4)*e.b^2)) / (2*(4*k^2+1))
               let y1 := k*(x1 - Real.sqrt 3 * e.b)
               let y2 := k*(x2 - Real.sqrt 3 * e.b)
               ((x1 + c)*(x2 + c) + y1*y2 < 0) → (k^2 < 1/47)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1328_132803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_target_l1328_132828

noncomputable def choices : List ℝ := [100, 1000, 10000, 100000, 1000000]

noncomputable def target : ℝ := 123 / 0.123

theorem closest_to_target : 
  ∀ x ∈ choices, |target - 1000| ≤ |target - x| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_target_l1328_132828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_same_focal_length_l1328_132831

/-- Focal length of an ellipse -/
noncomputable def focal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

/-- Semi-major axis of ellipse C₁ -/
noncomputable def a₁ : ℝ := Real.sqrt 12

/-- Semi-minor axis of ellipse C₁ -/
def b₁ : ℝ := 2

/-- Semi-major axis of ellipse C₂ -/
def a₂ : ℝ := 4

/-- Semi-minor axis of ellipse C₂ -/
noncomputable def b₂ : ℝ := Real.sqrt 8

theorem ellipses_same_focal_length :
  focal_length a₁ b₁ = focal_length a₂ b₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_same_focal_length_l1328_132831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_for_sequence_property_l1328_132866

def a : ℕ → ℤ
  | 0 => -1
  | n + 1 => (n * (n + 1)) / 2 - (n + 2)

theorem largest_k_for_sequence_property : 
  (∀ k > 4, ¬(a k + a (k + 1) = a (k + 2))) ∧ 
  (a 4 + a 5 = a 6) := by
  sorry

#eval a 4 + a 5
#eval a 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_for_sequence_property_l1328_132866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_3_or_5_l1328_132875

theorem count_divisible_by_3_or_5 :
  (Finset.filter (λ n : ℕ => n ≤ 2013 ∧ (n % 3 = 0 ∨ n % 5 = 0)) (Finset.range 2014)).card = 939 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_3_or_5_l1328_132875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_second_half_speed_l1328_132836

/-- A journey with two segments of equal distance but different speeds -/
structure Journey where
  total_distance : ℚ
  total_time : ℚ
  first_half_speed : ℚ

/-- Calculate the speed for the second half of the journey -/
def second_half_speed (j : Journey) : ℚ :=
  let first_half_distance := j.total_distance / 2
  let first_half_time := first_half_distance / j.first_half_speed
  let second_half_time := j.total_time - first_half_time
  first_half_distance / second_half_time

/-- Theorem stating that for the given journey parameters, the second half speed is 30 km/h -/
theorem journey_second_half_speed :
  let j : Journey := { total_distance := 540, total_time := 15, first_half_speed := 45 }
  second_half_speed j = 30 := by
  -- Proof goes here
  sorry

#eval second_half_speed { total_distance := 540, total_time := 15, first_half_speed := 45 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_second_half_speed_l1328_132836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salami_coverage_l1328_132887

/-- Represents a circular plate with salami slices -/
structure SalamiPlate where
  plate_diameter : ℝ
  salami_count_across : ℕ
  total_salami_count : ℕ

/-- Calculates the fraction of the plate covered by salami -/
noncomputable def fraction_covered (sp : SalamiPlate) : ℝ :=
  (sp.total_salami_count : ℝ) * (sp.plate_diameter / (2 * sp.salami_count_across : ℝ))^2 /
  (sp.plate_diameter / 2)^2

/-- Theorem: The fraction of the plate covered by salami is 1/2 -/
theorem salami_coverage (sp : SalamiPlate)
  (h1 : sp.plate_diameter = 16)
  (h2 : sp.salami_count_across = 8)
  (h3 : sp.total_salami_count = 32) :
  fraction_covered sp = 1/2 := by
  sorry

#eval "Salami coverage theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salami_coverage_l1328_132887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_X_largest_area_l1328_132892

/-- Represents the shaded area of Figure W -/
noncomputable def figureW_area : ℝ := 9 - 2.25 * Real.pi

/-- Represents the shaded area of Figure X -/
noncomputable def figureX_area : ℝ := (9 * Real.pi) / 2 - 9

/-- Represents the shaded area of Figure Y -/
noncomputable def figureY_area : ℝ := 9 - Real.pi

/-- Theorem stating that Figure X has the largest shaded area -/
theorem figure_X_largest_area :
  figureX_area > figureW_area ∧ figureX_area > figureY_area := by
  sorry

#eval "Proof completed."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_X_largest_area_l1328_132892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_imply_omega_value_l1328_132898

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem function_properties_imply_omega_value (ω : ℝ) (h_pos : ω > 0) :
  (∀ x y, x ∈ Set.Ioo (-ω) (2*ω) → y ∈ Set.Ioo (-ω) (2*ω) → x < y → f ω x < f ω y) →
  (∀ x : ℝ, f ω (x + ω) = f ω (-x - ω)) →
  ω = Real.sqrt (3 * Real.pi) / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_imply_omega_value_l1328_132898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1328_132865

theorem equation_solution :
  ∃ t : ℝ, 4 * (4 : ℝ)^t + (16 * (16 : ℝ)^t)^(1/2 : ℝ) = 64 ∧ t = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1328_132865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2012_eq_zero_l1328_132835

def sequence_term (n : ℕ) : ℤ :=
  match n % 4 with
  | 0 => (n / 4) * 4 + 1
  | 1 => -((n / 4) * 4 + 2)
  | 2 => -((n / 4) * 4 + 3)
  | 3 => (n / 4) * 4 + 4
  | _ => 0  -- This case is technically unreachable, but Lean requires it

def sequence_sum (n : ℕ) : ℤ :=
  (List.range n).map sequence_term |>.sum

theorem sequence_sum_2012_eq_zero : sequence_sum 2012 = 0 := by
  sorry

#eval sequence_sum 2012  -- This line is optional, but can be useful for checking the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2012_eq_zero_l1328_132835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersects_midline_l1328_132811

-- Define the complex numbers
noncomputable def Z₀ (a : ℝ) : ℂ := Complex.I * a
noncomputable def Z₁ (b : ℝ) : ℂ := (1 : ℝ) / 2 + Complex.I * b
noncomputable def Z₂ (c : ℝ) : ℂ := 1 + Complex.I * c

-- Define the curve
noncomputable def Z (a b c : ℝ) (t : ℝ) : ℂ :=
  Z₀ a * (Real.cos t)^4 + 2 * Z₁ b * (Real.cos t)^2 * (Real.sin t)^2 + Z₂ c * (Real.sin t)^4

-- Define the midline
noncomputable def midline (a b c : ℝ) (x : ℝ) : ℝ := 
  (c - a) * x + (1 : ℝ) / 4 * (3 * a + 2 * b - c)

-- State the theorem
theorem curve_intersects_midline (a b c : ℝ) :
  ∃! z : ℂ, (∃ t : ℝ, Z a b c t = z) ∧
            (z.re = (1 : ℝ) / 2) ∧
            (z.im = midline a b c ((1 : ℝ) / 2)) ∧
            (z = (1 : ℝ) / 2 + Complex.I * ((a + c + 2 * b) / 4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersects_midline_l1328_132811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_gain_percentage_l1328_132833

/-- Represents the dealer's sales strategy -/
structure DealerStrategy where
  weight1 : ℚ  -- Weight used for the first item in grams
  weight2 : ℚ  -- Weight used for the second item in grams
  nominalWeight : ℚ  -- Nominal weight of each item in grams
  nominalPrice : ℚ  -- Nominal price per kg in dollars

/-- Calculates the dealer's gain percentage -/
def gainPercentage (s : DealerStrategy) : ℚ :=
  let actualWeight := s.weight1 + s.weight2
  let nominalTotalWeight := 2 * s.nominalWeight
  let actualCost := (actualWeight / s.nominalWeight) * s.nominalPrice
  let totalGain := 2 * s.nominalPrice - actualCost
  (totalGain / actualCost) * 100

/-- Theorem stating the dealer's gain percentage for the given strategy -/
theorem dealer_gain_percentage (s : DealerStrategy) 
  (h1 : s.weight1 = 900)
  (h2 : s.weight2 = 850)
  (h3 : s.nominalWeight = 1000)
  (h4 : s.nominalPrice = 100) :
  gainPercentage s = 25 / 175 * 100 := by
  sorry

#eval gainPercentage { weight1 := 900, weight2 := 850, nominalWeight := 1000, nominalPrice := 100 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_gain_percentage_l1328_132833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_efficiency_l1328_132838

theorem painting_efficiency (d h e : ℝ) (hd : d > 0) (hh : h > 0) (he : e > 0) :
  let original_rate := h / (d * e)
  let new_days := d^2 * e / (2 * h^2)
  original_rate * (2 * h * new_days) = d := by
  sorry

#check painting_efficiency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_efficiency_l1328_132838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_k_l1328_132853

theorem integral_equals_k (k : ℝ) : (∫ (x : ℝ) in Set.Icc 0 1, k * x + 1) = k → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_k_l1328_132853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_fewer_heads_12_coins_l1328_132817

/-- The number of coins being flipped -/
def n : ℕ := 12

/-- The probability of getting fewer heads than tails when flipping n coins -/
def prob_fewer_heads (n : ℕ) : ℚ :=
  (2^n - Nat.choose n (n/2)) / (2 * 2^n)

/-- Theorem stating that the probability of getting fewer heads than tails 
    when flipping 12 coins is equal to 793/2048 -/
theorem prob_fewer_heads_12_coins : 
  prob_fewer_heads n = 793 / 2048 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_fewer_heads_12_coins_l1328_132817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_count_l1328_132827

/-- A circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the configuration of the circle and tangents -/
structure CircleTangentConfig where
  circle : Circle
  horizontalTangent : Line
  intersectingTangent : Line
  d : ℝ  -- distance from circle center to horizontal tangent and point P
  h : d > circle.radius  -- condition that d > r

/-- Predicate to check if a point is equidistant from a circle and two lines -/
def is_equidistant (p : ℝ × ℝ) (c : Circle) (l1 l2 : Line) : Prop :=
  sorry

/-- The main theorem -/
theorem equidistant_points_count (config : CircleTangentConfig) :
  ∃ (points : Finset (ℝ × ℝ)), points.card = 3 ∧
    ∀ p ∈ points, is_equidistant p config.circle config.horizontalTangent config.intersectingTangent :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_count_l1328_132827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_non_monotonic_l1328_132895

/-- The function f(x) defined on real numbers -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + a*x - 5

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

/-- Theorem stating the condition for f(x) to be non-monotonic in [-1, 2] -/
theorem f_non_monotonic (a : ℝ) :
  (∃ x ∈ Set.Icc (-1) 2, f_deriv a x < 0) ∧
  (∃ y ∈ Set.Icc (-1) 2, f_deriv a y > 0) ↔
  a ∈ Set.Ioo (-3) 1 := by
  sorry

#check f_non_monotonic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_non_monotonic_l1328_132895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_separation_theorem_l1328_132889

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a line intersects a circle -/
def lineIntersectsCircle (line : ℝ × ℝ → Bool) (c : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), line p = true ∧ (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 ≤ c.radius^2

/-- Count the number of circles on one side of a line -/
def circlesOnOneSide (line : ℝ × ℝ → Bool) (circles : List Circle) : ℕ :=
  (circles.filter (λ c => ¬(line c.center))).length

theorem circle_separation_theorem (circles : List Circle) 
  (h1 : circles.length = 6)
  (h2 : ∀ c, c ∈ circles → c.radius = 1)
  (h3 : ∀ c1 c2, c1 ∈ circles → c2 ∈ circles → c1 ≠ c2 → 
    (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 ≥ 4) :
  ∃ (line : ℝ × ℝ → Bool), 
    (∀ c, c ∈ circles → ¬(lineIntersectsCircle line c)) ∧
    (circlesOnOneSide line circles = 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_separation_theorem_l1328_132889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_smallest_prime_divisor_iff_odd_l1328_132858

-- Define the sequence a_n
def a : ℕ → ℕ → ℕ
  | _, 0 => 0  -- Add this case to handle n = 0
  | a₁, 1 => a₁
  | a₁, n + 1 => (a a₁ n)^n - 1

-- Define the smallest prime divisor function
def smallest_prime_divisor (k : ℕ) : ℕ :=
  Nat.minFac k

-- Define boundedness of a sequence
def is_bounded (s : ℕ → ℕ) : Prop :=
  ∃ M, ∀ n, s n ≤ M

-- Main theorem
theorem bounded_smallest_prime_divisor_iff_odd (a₁ : ℕ) (h : a₁ > 2) :
  is_bounded (λ n ↦ smallest_prime_divisor (a a₁ n)) ↔ Odd a₁ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_smallest_prime_divisor_iff_odd_l1328_132858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_parabola_triangle_l1328_132863

/-- Given two points on a parabola, prove the maximum area of the triangle formed with a specific point -/
theorem max_area_parabola_triangle (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁^2 = 4 * y₁ →  -- A is on the parabola
  x₂^2 = 4 * y₂ →  -- B is on the parabola
  y₁ + y₂ = 2 →   -- Sum of y-coordinates
  y₁ ≠ y₂ →       -- y-coordinates are different
  ∃ (max_area : ℝ), ∀ (x₁' x₂' y₁' y₂' : ℝ), 
    x₁'^2 = 4 * y₁' → 
    x₂'^2 = 4 * y₂' → 
    y₁' + y₂' = 2 → 
    y₁' ≠ y₂' → 
    let A' := (x₁', y₁')
    let B' := (x₂', y₂')
    let C := (0, 3)
    let area' := abs (x₁' * (y₂' - 3) + x₂' * (3 - y₁')) / 2
    area' ≤ max_area ∧ max_area = 16 * Real.sqrt 6 / 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_parabola_triangle_l1328_132863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersections_l1328_132874

/-- The line equation 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The circle equation x^2 + y^2 = 4 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- A point (x, y) is an intersection if it satisfies both the line and circle equations -/
def is_intersection (x y : ℝ) : Prop := line x y ∧ circle_eq x y

/-- The number of intersections between the line and the circle is 0 -/
theorem no_intersections : ¬∃ x y : ℝ, is_intersection x y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersections_l1328_132874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_real_l1328_132882

/-- A function f with parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x - 4) / (m * x^2 + 4 * m * x + 3)

/-- The theorem stating the range of m for which f has domain ℝ -/
theorem f_domain_real (m : ℝ) :
  (∀ x, IsRegular (f m x)) ↔ m ∈ Set.Ici 0 ∩ Set.Iio (3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_real_l1328_132882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_construction_blocks_l1328_132870

/-- Calculates the volume of a square pyramid -/
noncomputable def pyramidVolume (baseLength : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * baseLength^2 * height

/-- Calculates the volume of a rectangular prism -/
noncomputable def prismVolume (length width height : ℝ) : ℝ :=
  length * width * height

/-- The minimum number of whole prisms needed to construct the pyramid -/
noncomputable def minPrisms (prismLength prismWidth prismHeight : ℝ) 
              (pyramidBaseLength pyramidHeight : ℝ) : ℕ :=
  (Int.ceil ((pyramidVolume pyramidBaseLength pyramidHeight) / 
   (prismVolume prismLength prismWidth prismHeight))).toNat

theorem pyramid_construction_blocks :
  minPrisms 6 3 2 5 8 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_construction_blocks_l1328_132870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_germination_experiment_l1328_132880

/-- Represents a plot with seeds and germination rate -/
structure Plot where
  seeds : ℕ
  germination_rate : ℚ

/-- Calculates the number of germinated seeds in a plot -/
def germinated_seeds (p : Plot) : ℚ :=
  p.seeds * p.germination_rate

/-- Calculates the total number of seeds across all plots -/
def total_seeds (plots : List Plot) : ℕ :=
  plots.map (·.seeds) |>.sum

/-- Calculates the total number of germinated seeds across all plots -/
def total_germinated (plots : List Plot) : ℚ :=
  plots.map germinated_seeds |>.sum

/-- Calculates the overall germination rate -/
def overall_germination_rate (plots : List Plot) : ℚ :=
  total_germinated plots / (total_seeds plots : ℚ)

/-- The main theorem stating the overall germination rate -/
theorem germination_experiment :
  let plots := [
    { seeds := 300, germination_rate := 25/100 },
    { seeds := 200, germination_rate := 35/100 },
    { seeds := 400, germination_rate := 45/100 },
    { seeds := 350, germination_rate := 15/100 },
    { seeds := 150, germination_rate := 50/100 }
  ]
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000 ∧ 
    |overall_germination_rate plots - 323/1000| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_germination_experiment_l1328_132880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foggy_time_is_40_minutes_l1328_132862

/-- Represents the driving scenario with sunny and foggy conditions -/
structure DrivingScenario where
  sunny_speed : ℝ  -- Speed in sunny conditions (mph)
  foggy_speed : ℝ  -- Speed in foggy conditions (mph)
  total_distance : ℝ  -- Total distance traveled (miles)
  total_time : ℝ  -- Total time traveled (minutes)

/-- Calculates the time spent in foggy conditions given a driving scenario -/
noncomputable def time_in_foggy_conditions (scenario : DrivingScenario) : ℝ :=
  let sunny_speed_per_minute := scenario.sunny_speed / 60
  let foggy_speed_per_minute := scenario.foggy_speed / 60
  ((sunny_speed_per_minute * scenario.total_time - scenario.total_distance) /
   (sunny_speed_per_minute - foggy_speed_per_minute))

/-- Theorem stating that given the specific conditions, the time in foggy conditions is 40 minutes -/
theorem foggy_time_is_40_minutes (scenario : DrivingScenario) 
    (h1 : scenario.sunny_speed = 40)
    (h2 : scenario.foggy_speed = 25)
    (h3 : scenario.total_distance = 30)
    (h4 : scenario.total_time = 60) : 
  time_in_foggy_conditions scenario = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_foggy_time_is_40_minutes_l1328_132862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nickys_pace_l1328_132822

/-- Prove that Nicky's pace is 3 meters per second given the race conditions -/
theorem nickys_pace (head_start : ℝ) (cristina_pace : ℝ) (catch_up_time : ℝ) 
  (h1 : head_start = 54)
  (h2 : cristina_pace = 5)
  (h3 : catch_up_time = 27) :
  let nicky_pace := (head_start + catch_up_time * cristina_pace) / catch_up_time - cristina_pace
  (head_start + catch_up_time * (cristina_pace - nicky_pace)) / catch_up_time = nicky_pace ∧ 
  nicky_pace = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nickys_pace_l1328_132822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_good_students_options_l1328_132861

/-- Represents the number of good students in the class -/
def num_good_students : ℕ := sorry

/-- Represents the number of troublemakers in the class -/
def num_troublemakers : ℕ := sorry

/-- The total number of students in the class -/
def total_students : ℕ := 25

/-- Each student is either a good student or a troublemaker -/
axiom student_classification : num_good_students + num_troublemakers = total_students

/-- Condition for the statement of the first group of 5 students -/
axiom first_group_condition : num_troublemakers > (total_students - 1) / 2

/-- Condition for the statement of the second group of 20 students -/
axiom second_group_condition : num_troublemakers = 3 * (num_good_students - 1)

/-- The theorem stating that the number of good students is either 5 or 7 -/
theorem num_good_students_options : num_good_students = 5 ∨ num_good_students = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_good_students_options_l1328_132861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_square_b_l1328_132884

/-- Given a square A with perimeter 32 cm and a square B with area equal to one-third the area of square A, 
    the perimeter of square B is (32√3)/3 cm. -/
theorem perimeter_of_square_b (square_a square_b : Real) : 
  square_a > 0 ∧
  square_b > 0 ∧
  4 * square_a = 32 ∧ 
  square_b^2 = (1/3) * square_a^2 →
  4 * square_b = (32 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_square_b_l1328_132884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pairs_satisfying_equation_l1328_132867

theorem integer_pairs_satisfying_equation : 
  ∃! (pairs : Finset (ℤ × ℤ)), 
    (∀ (m n : ℤ), (m, n) ∈ pairs ↔ m + n = 2 * m * n) ∧ 
    Finset.card pairs = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pairs_satisfying_equation_l1328_132867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_side_length_l1328_132801

theorem triangle_max_side_length (D E F : ℝ) (a b : ℝ) :
  Real.cos (3 * D) + Real.cos (3 * E) + Real.cos (3 * F) = 1 →
  a = 12 →
  b = 14 →
  ∃ c : ℝ, c ≤ 2 * Real.sqrt 127 ∧
    ∃ (A B C : ℝ), A + B + C = Real.pi ∧
    c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_side_length_l1328_132801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_approximation_l1328_132848

-- Define constants
noncomputable def bus_speed : ℝ := 66  -- km/h
noncomputable def wheel_rpm : ℝ := 250.22747952684256

-- Define function to calculate wheel radius
noncomputable def calculate_wheel_radius (speed : ℝ) (rpm : ℝ) : ℝ :=
  (speed * 100000 / 60) / (rpm * 2 * Real.pi)

-- Theorem statement
theorem wheel_radius_approximation :
  ‖calculate_wheel_radius bus_speed wheel_rpm - 70.007‖ < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_approximation_l1328_132848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1328_132820

noncomputable def f (x : ℝ) : ℝ := (3 * Real.sqrt 2 / 2) * Real.sin (2 * x - Real.pi / 6) + Real.sqrt 2 / 2

def is_monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_properties :
  (∀ x, f x ≤ 2 * Real.sqrt 2) ∧
  (∀ x, f x ≥ -Real.sqrt 2) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  (f 0 = -Real.sqrt 2 / 4) ∧
  (∀ k : ℤ, is_monotone_increasing f (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1328_132820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_c_to_b_approx_l1328_132894

/-- Represents the cost of travel between cities -/
structure TravelCost where
  busCostPerKm : ℚ
  flightBookingFee : ℚ
  flightCostPerKm : ℚ

/-- Calculates the total cost of travel from C to B -/
noncomputable def calculateCostCtoB (ac ab : ℚ) (cost : TravelCost) : ℚ :=
  let bc := (ab^2 - ac^2).sqrt
  let halfDistance := bc / 2
  let flightCost := cost.flightBookingFee + cost.flightCostPerKm * halfDistance
  let busCost := cost.busCostPerKm * halfDistance
  flightCost + busCost

/-- The theorem stating the cost of travel from C to B -/
theorem cost_c_to_b_approx (cost : TravelCost) :
    cost.busCostPerKm = 1/5 →
    cost.flightBookingFee = 150 →
    cost.flightCostPerKm = 3/25 →
    ∃ ε > 0, |calculateCostCtoB 3000 4000 cost - 573.32| < ε := by
  sorry

#check cost_c_to_b_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_c_to_b_approx_l1328_132894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_increase_l1328_132891

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def snowman_volume (r1 r2 r3 : ℝ) : ℝ :=
  sphere_volume r1 + sphere_volume r2 + sphere_volume r3

theorem snowman_volume_increase :
  let initial_volume := snowman_volume 4 6 8
  let increased_volume := snowman_volume 5 7 9
  initial_volume = 1056 * Real.pi ∧ increased_volume = 1596 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_increase_l1328_132891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1328_132824

def sequence_a : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * sequence_a n + 1

theorem sequence_a_properties :
  (sequence_a 3 = 23) ∧
  (∀ n : ℕ, sequence_a n = 3 * 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1328_132824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_two_in_one_seventh_l1328_132878

/-- The probability distribution for choosing a positive integer i -/
noncomputable def prob (i : ℕ+) : ℝ := (2/3)^(i.val - 1) * (1/3)

/-- The decimal expansion of 1/7 -/
def oneSeventhExpansion (i : ℕ+) : ℕ :=
  match i.val % 6 with
  | 1 => 1
  | 2 => 4
  | 3 => 2
  | 4 => 8
  | 5 => 5
  | 0 => 7
  | _ => 0  -- This case should never occur, but Lean requires it for exhaustiveness

/-- The probability that the i-th digit of 1/7's expansion is 2 -/
noncomputable def probOfTwo (i : ℕ+) : ℝ :=
  if oneSeventhExpansion i = 2 then prob i else 0

theorem probability_of_two_in_one_seventh :
  ∑' (i : ℕ+), probOfTwo i = 108/665 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_two_in_one_seventh_l1328_132878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_4_f_max_value_f_achieves_max_l1328_132843

-- Define the function f as noncomputable due to dependency on Real.instPowReal
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else (2 : ℝ)^x

-- Theorem to prove f(f(4)) = 1/2
theorem f_f_4 : f (f 4) = 1/2 := by sorry

-- Theorem to prove the maximum value of f is 1
theorem f_max_value : ∀ x : ℝ, f x ≤ 1 := by sorry

-- Theorem to prove that 1 is actually achieved
theorem f_achieves_max : ∃ x : ℝ, f x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_4_f_max_value_f_achieves_max_l1328_132843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_a_l1328_132844

def t (k : ℕ) : ℕ := Nat.gcd k (2^64 - 1)

theorem no_valid_a : ∀ a : ℕ+, ¬∃ n : ℕ, ∀ i : Fin a, (t (n + a + i) - t (n + i)) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_a_l1328_132844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_log_l1328_132802

theorem inequality_implies_log (x y : ℝ) :
  (2 : ℝ)^x - (2 : ℝ)^y < (3 : ℝ)^(-x) - (3 : ℝ)^(-y) → Real.log (y - x + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_log_l1328_132802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_3sqrt2_l1328_132804

/-- Curve C in Cartesian coordinates -/
def curve_C (x y : ℝ) : Prop := y^2 = x

/-- Line L in parametric form -/
def line_L (t x y : ℝ) : Prop := x = 2 - (Real.sqrt 2 / 2) * t ∧ y = (Real.sqrt 2 / 2) * t

/-- Intersection points of curve C and line L -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ),
    curve_C A.1 A.2 ∧ line_L t₁ A.1 A.2 ∧
    curve_C B.1 B.2 ∧ line_L t₂ B.1 B.2 ∧
    t₁ ≠ t₂

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- Main theorem: The length of AB is 3√2 -/
theorem length_AB_is_3sqrt2 (A B : ℝ × ℝ) :
  intersection_points A B → distance A B = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_3sqrt2_l1328_132804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_parallelogram_point_l1328_132821

/-- Point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle in a plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Construct a circle through two points with a given radius -/
noncomputable def constructCircle (P Q : Point) (R : ℝ) : Circle := sorry

/-- Check if a circle covers a point -/
def Circle.covers (c : Circle) (P : Point) : Prop := sorry

/-- Get the intersection points of two circles -/
noncomputable def Circle.intersect (c1 c2 : Circle) : Set Point := sorry

/-- Check if four points form a parallelogram -/
def isParallelogram (A B C D : Point) : Prop := sorry

/-- Circumradius of a triangle -/
noncomputable def Triangle.circumradius (t : Triangle) : ℝ := sorry

theorem construct_parallelogram_point 
  (t : Triangle) 
  (R : ℝ) 
  (h : R > t.circumradius) :
  ∃ D : Point, isParallelogram t.A t.B t.C D ∧ 
  ∃ (k1 k2 k3 k4 : Circle),
    k1 = constructCircle t.A t.B R ∧
    k2 = constructCircle t.B t.C R ∧
    k1.covers t.C ∧
    k2.covers t.A ∧
    ∃ P ∈ k1.intersect k2,
      k3 = constructCircle P t.A R ∧
      k4 = constructCircle P t.C R ∧
      D ∈ k3.intersect k4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_parallelogram_point_l1328_132821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l1328_132842

/-- Calculates the time (in seconds) it takes for a train to cross a pole -/
noncomputable def train_crossing_time (speed_km_hr : ℝ) (length_m : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * 1000 / 3600
  length_m / speed_m_s

theorem train_crossing_pole_time :
  let speed := (30 : ℝ)
  let length := (75 : ℝ)
  let crossing_time := train_crossing_time speed length
  ∃ ε > 0, |crossing_time - 9| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l1328_132842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_curve_equation_l1328_132851

/-- Rotation of a point (x, y) by π/4 counterclockwise around the origin -/
noncomputable def rotate_point (x y : ℝ) : ℝ × ℝ :=
  (Real.sqrt 2 / 2 * (x - y), Real.sqrt 2 / 2 * (x + y))

/-- The rotated curve equation -/
def rotated_curve (x y : ℝ) : Prop :=
  x^2 - y^2 = 2

/-- The original curve equation -/
def original_curve (x y : ℝ) : Prop :=
  x * y = -1

theorem original_curve_equation :
  (∀ x y : ℝ, rotated_curve (rotate_point x y).1 (rotate_point x y).2) →
  (∀ x y : ℝ, original_curve x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_curve_equation_l1328_132851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1328_132879

/-- A geometric sequence with 10 terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n < 10 → a (n + 1) = r * a n

/-- The product of odd terms in the sequence -/
noncomputable def product_odd_terms (a : ℕ → ℝ) : ℝ :=
  (a 1) * (a 3) * (a 5) * (a 7) * (a 9)

/-- The product of even terms in the sequence -/
noncomputable def product_even_terms (a : ℕ → ℝ) : ℝ :=
  (a 2) * (a 4) * (a 6) * (a 8) * (a 10)

/-- The common ratio of the geometric sequence -/
noncomputable def common_ratio (a : ℕ → ℝ) : ℝ :=
  a 2 / a 1

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  geometric_sequence a →
  product_odd_terms a = 2 →
  product_even_terms a = 64 →
  common_ratio a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1328_132879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_differential_at_one_l1328_132871

-- Define the function y(x)
noncomputable def y (x : ℝ) : ℝ := Real.arctan (1 / x)

-- Define the derivative of y(x)
noncomputable def y_derivative (x : ℝ) : ℝ := -1 / (x^2 + 1)

-- Theorem statement
theorem first_differential_at_one (dx : ℝ) :
  dx = -0.1 →
  y_derivative 1 * dx = 0.05 := by
  intro h
  have h1 : y_derivative 1 = -1/2 := by
    simp [y_derivative]
    ring
  calc
    y_derivative 1 * dx = -1/2 * (-0.1) := by rw [h1, h]
    _ = 0.05 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_differential_at_one_l1328_132871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1328_132857

-- Define the custom operation
noncomputable def circle_slash (a b : ℝ) : ℝ := (Real.sqrt (3 * a - b))^3

-- State the theorem
theorem solve_equation (x : ℝ) : circle_slash 9 x = 64 → x = 11 := by
  intro h
  -- Here we would normally provide the proof steps, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1328_132857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_for_horizontal_asymptote_l1328_132876

/-- The denominator of our rational function -/
noncomputable def p (x : ℝ) : ℝ := 2*x^5 + x^4 - 7*x^2 + 1

/-- The rational function -/
noncomputable def f (q : ℝ → ℝ) (x : ℝ) : ℝ := q x / p x

/-- A function has a horizontal asymptote if it converges to a finite value as x approaches infinity -/
def has_horizontal_asymptote (f : ℝ → ℝ) : Prop :=
  ∃ L : ℝ, ∀ ε > 0, ∃ N : ℝ, ∀ x > N, |f x - L| < ε

/-- The degree of a polynomial -/
noncomputable def degree (q : ℝ → ℝ) : ℕ :=
  sorry

/-- The main theorem: The largest possible degree of q for f to have a horizontal asymptote is 5 -/
theorem largest_degree_for_horizontal_asymptote :
  (∃ q : ℝ → ℝ, has_horizontal_asymptote (f q) ∧ degree q = 5) ∧
  (∀ q : ℝ → ℝ, has_horizontal_asymptote (f q) → degree q ≤ 5) :=
by
  sorry

#check largest_degree_for_horizontal_asymptote

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_for_horizontal_asymptote_l1328_132876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_path_length_l1328_132814

/-- The path length of vertex P in an equilateral triangle rotating in a square -/
theorem triangle_rotation_path_length :
  ∀ (triangle_side : ℝ) (square_side : ℝ),
  triangle_side = 2 →
  square_side = 4 →
  ∃ (path_length : ℝ),
  path_length = (40 * Real.pi) / 3 ∧
  path_length = 12 * triangle_side * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_path_length_l1328_132814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_k_value_l1328_132855

def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem quadratic_function_k_value (a b c k : ℤ) : 
  f a b c 2 = 0 →
  100 < f a b c 7 ∧ f a b c 7 < 110 →
  120 < f a b c 8 ∧ f a b c 8 < 130 →
  6000 * k < f a b c 100 ∧ f a b c 100 < 6000 * (k + 1) →
  k = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_k_value_l1328_132855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_period_22_days_l1328_132868

/-- The number of days required for a given percentage of water to evaporate -/
noncomputable def evaporation_days (initial_amount : ℝ) (evaporation_rate : ℝ) (evaporation_percentage : ℝ) : ℝ :=
  (evaporation_percentage / 100) * initial_amount / evaporation_rate

/-- Theorem stating that the evaporation period is 22 days under given conditions -/
theorem evaporation_period_22_days :
  evaporation_days 12 0.03 5.5 = 22 := by
  -- Proof steps would go here
  sorry

-- Remove the #eval statement as it's not computable
-- #eval evaporation_days 12 0.03 5.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_period_22_days_l1328_132868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1328_132841

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^3 + (Real.sin x)^2 - 4 * (Real.sin x) + 8

theorem f_range :
  (∀ x : ℝ, 6 + 3/4 ≤ f x ∧ f x ≤ 9 + 25/27) ∧
  (∃ x₁ x₂ : ℝ, f x₁ = 6 + 3/4 ∧ f x₂ = 9 + 25/27) := by
  sorry

#check f_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1328_132841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_between_tangents_l1328_132859

-- Define the circle
def circleA (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 2

-- Define the parabola
def parabolaC (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop := parabolaC P.1 P.2

-- Define the tangent lines from a point to the circle
def tangent_lines (P : ℝ × ℝ) (A : ℝ × ℝ) : Prop := 
  circleA A.1 A.2 ∧ point_on_parabola P

-- Define the angle between two lines
noncomputable def angle_between_lines (l1 l2 : ℝ × ℝ → ℝ × ℝ → Prop) : ℝ → Prop := sorry

-- Theorem statement
theorem max_angle_between_tangents :
  ∃ (max_angle : ℝ), 
    (∀ (P : ℝ × ℝ) (l1 l2 : ℝ × ℝ → ℝ × ℝ → Prop),
      tangent_lines P (3, 0) →
      angle_between_lines l1 l2 max_angle) ∧
    max_angle = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_between_tangents_l1328_132859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1328_132807

/-- The function g(t) defined as 1 / ((t-2)^2 + (t+2)^2 + 1) -/
noncomputable def g (t : ℝ) : ℝ := 1 / ((t - 2)^2 + (t + 2)^2 + 1)

/-- The domain of g is all real numbers -/
theorem domain_of_g : ∀ t : ℝ, g t ≠ 0 := by
  intro t
  unfold g
  simp
  apply ne_of_gt
  positivity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1328_132807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olivia_race_time_l1328_132823

-- Define the race participants
structure Racer where
  name : String
  time : ℚ

-- Define the race conditions
def patrick : Racer := ⟨"Patrick", 60⟩
def manu : Racer := ⟨"Manu", patrick.time + 12⟩
def amy : Racer := ⟨"Amy", manu.time / 2⟩
def olivia : Racer := ⟨"Olivia", amy.time * (2/3)⟩

-- Theorem to prove
theorem olivia_race_time : olivia.time = 24 := by
  -- Expand definitions
  simp [olivia, amy, manu, patrick]
  -- Perform arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olivia_race_time_l1328_132823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_repeating_decimal_and_six_l1328_132846

theorem product_of_repeating_decimal_and_six : ∃ x : ℚ, x = 2/3 ∧ x * 6 = 4 := by
  use 2/3
  constructor
  · rfl
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_repeating_decimal_and_six_l1328_132846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1328_132837

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - x^2) / (x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-2) 1 ∪ Set.Ioc 1 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1328_132837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_x₀_equal_x₆_l1328_132856

noncomputable def x (n : ℕ) (x₀ : ℝ) : ℝ :=
  match n with
  | 0 => x₀
  | n + 1 =>
    let xₙ := x n x₀
    if 2 * xₙ < 1 then 2 * xₙ else 2 * xₙ - 1

theorem count_x₀_equal_x₆ :
  ∃! (s : Finset ℝ), 
    (∀ x₀ ∈ s, 0 ≤ x₀ ∧ x₀ < 1 ∧ x 6 x₀ = x₀) ∧
    Finset.card s = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_x₀_equal_x₆_l1328_132856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_space_for_one_more_domino_l1328_132813

/-- Represents a 6x6 chessboard -/
def Chessboard := Fin 6 → Fin 6 → Bool

/-- A domino covers exactly two adjacent squares -/
def isDomino (board : Chessboard) (x1 y1 x2 y2 : Fin 6) : Prop :=
  ((x1 = x2 ∧ y2 = y1 + 1) ∨ (y1 = y2 ∧ x2 = x1 + 1)) ∧
  board x1 y1 = true ∧ board x2 y2 = true

/-- Checks if a given configuration has exactly 11 dominoes -/
def has11Dominoes (board : Chessboard) : Prop :=
  ∃ (dominoes : Fin 11 → Fin 6 × Fin 6 × Fin 6 × Fin 6),
    (∀ i : Fin 11, isDomino board (dominoes i).1 (dominoes i).2.1 (dominoes i).2.2.1 (dominoes i).2.2.2) ∧
    (∀ x y : Fin 6, board x y = true → 
      ∃ i : Fin 11, (x = (dominoes i).1 ∧ y = (dominoes i).2.1) ∨ 
                    (x = (dominoes i).2.2.1 ∧ y = (dominoes i).2.2.2))

/-- There exists space for one more domino -/
def hasSpaceForOneDomino (board : Chessboard) : Prop :=
  ∃ x1 y1 x2 y2 : Fin 6, ((x1 = x2 ∧ y2 = y1 + 1) ∨ (y1 = y2 ∧ x2 = x1 + 1)) ∧ 
    board x1 y1 = false ∧ board x2 y2 = false

/-- Main theorem: If a 6x6 chessboard has 11 dominoes, there's always space for one more -/
theorem always_space_for_one_more_domino (board : Chessboard) 
  (h : has11Dominoes board) : hasSpaceForOneDomino board := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_space_for_one_more_domino_l1328_132813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_sum_equals_original_sum_l1328_132849

theorem modified_sum_equals_original_sum (n : ℕ) : ∃ (half third rest : Finset ℕ),
  half ∪ third ∪ rest = Finset.range (6 * n + 1) \ {0} ∧
  half ∩ third = ∅ ∧ half ∩ rest = ∅ ∧ third ∩ rest = ∅ ∧
  2 * (Finset.sum half (λ x ↦ x / 2)) +
  3 * (Finset.sum third (λ x ↦ x / 3)) +
  (Finset.sum rest (λ x ↦ 6 * x)) =
  Finset.sum (Finset.range (6 * n + 1) \ {0}) id := by
  sorry

#check modified_sum_equals_original_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_sum_equals_original_sum_l1328_132849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_3I_property_l1328_132826

theorem matrix_3I_property (A : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ v : Fin 3 → ℝ, A.mulVec v = (3 : ℝ) • v) ↔ 
  A = ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]] := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_3I_property_l1328_132826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solution_l1328_132888

theorem cosine_equation_solution (x : ℝ) (k : ℤ) : 
  x = 2 * Real.pi * ↑k →
  Real.cos (10 * x) + 2 * (Real.cos (4 * x))^2 + 6 * Real.cos (3 * x) * Real.cos x = 
  Real.cos x + 8 * Real.cos x * (Real.cos (3 * x))^3 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solution_l1328_132888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PMN_MN_passes_fixed_point_l1328_132845

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point P
def P : ℝ × ℝ := (2, 0)

-- Define the slopes of the lines
variable (k1 k2 : ℝ)

-- Define points A, B, C, D
variable (A B C D : ℝ × ℝ)

-- Define M as midpoint of AB
noncomputable def M (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define N as midpoint of CD
noncomputable def N (C D : ℝ × ℝ) : ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Theorem for part I
theorem min_area_PMN 
  (h1 : k1 * k2 = -1)
  (h2 : parabola A.1 A.2)
  (h3 : parabola B.1 B.2)
  (h4 : parabola C.1 C.2)
  (h5 : parabola D.1 D.2)
  : ∃ (area : ℝ), area ≥ 4 ∧ ∀ (other_area : ℝ), other_area ≥ area := by
  sorry

-- Theorem for part II
theorem MN_passes_fixed_point
  (h1 : k1 + k2 = 1)
  (h2 : parabola A.1 A.2)
  (h3 : parabola B.1 B.2)
  (h4 : parabola C.1 C.2)
  (h5 : parabola D.1 D.2)
  : ∃ (fixed_point : ℝ × ℝ), fixed_point = (2, 2) ∧ 
    ((N C D).2 - (M A B).2) / ((N C D).1 - (M A B).1) = 
    (fixed_point.2 - (M A B).2) / (fixed_point.1 - (M A B).1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PMN_MN_passes_fixed_point_l1328_132845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_from_investment_l1328_132847

/-- Calculates the income derived from a stock investment --/
noncomputable def calculate_income (investment_amount : ℝ) (dividend_rate : ℝ) (brokerage_rate : ℝ) (market_value : ℝ) : ℝ :=
  let brokerage_fee := investment_amount * brokerage_rate
  let effective_investment := investment_amount - brokerage_fee
  let face_value := effective_investment / (market_value / 100)
  face_value * dividend_rate

/-- Theorem stating that the income derived from the given investment is 756 --/
theorem income_from_investment :
  calculate_income 6500 0.105 0.0025 90.02777777777779 = 756 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_from_investment_l1328_132847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_distinct_naturals_l1328_132810

theorem no_four_distinct_naturals : ¬ ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (b - c).gcd a = a ∧ (b - d).gcd a = a ∧ (c - d).gcd a = a ∧
  (a - c).gcd b = b ∧ (a - d).gcd b = b ∧ (c - d).gcd b = b ∧
  (a - b).gcd c = c ∧ (a - d).gcd c = c ∧ (b - d).gcd c = c ∧
  (a - b).gcd d = d ∧ (a - c).gcd d = d ∧ (b - c).gcd d = d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_distinct_naturals_l1328_132810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_price_theorem_min_sales_volume_theorem_l1328_132825

/-- Original price in yuan -/
noncomputable def original_price : ℝ := 25

/-- Original annual sales volume in ten thousand units -/
noncomputable def original_volume : ℝ := 8

/-- Price elasticity: units decreased per yuan increase -/
noncomputable def price_elasticity : ℝ := 0.2

/-- Technical reform fee function -/
noncomputable def tech_reform_fee (x : ℝ) : ℝ := (1/6) * (x^2 - 600)

/-- Fixed promotion fee -/
noncomputable def fixed_promo_fee : ℝ := 50

/-- Variable promotion fee function -/
noncomputable def var_promo_fee (x : ℝ) : ℝ := (1/5) * x

/-- New volume function based on price change -/
noncomputable def new_volume (x : ℝ) : ℝ := original_volume - price_elasticity * (x - original_price)

/-- Total revenue function -/
noncomputable def total_revenue (x : ℝ) : ℝ := x * new_volume x

/-- Investment function -/
noncomputable def investment (x : ℝ) : ℝ := tech_reform_fee x + fixed_promo_fee + var_promo_fee x

/-- Theorem for the highest price maintaining or increasing total sales revenue -/
theorem highest_price_theorem :
  ∃ (max_price : ℝ), max_price = 40 ∧
  ∀ (x : ℝ), x ≤ max_price → total_revenue x ≥ total_revenue original_price :=
by sorry

/-- Theorem for minimum sales volume and corresponding price -/
theorem min_sales_volume_theorem :
  ∃ (min_volume price : ℝ), min_volume = 10.2 ∧ price = 30 ∧
  ∀ (x : ℝ), x > original_price →
    (min_volume * x ≥ total_revenue original_price + investment x) ∧
    (∀ (v : ℝ), v < min_volume → v * x < total_revenue original_price + investment x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_price_theorem_min_sales_volume_theorem_l1328_132825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_can_prevent_divisibility_l1328_132850

/-- Represents a digit in the number being constructed -/
inductive Digit
| zero
| one
| two

/-- The game state -/
structure GameState where
  digits : List Digit
  currentPlayer : Bool  -- true for Alice, false for Bob

/-- The strategy function type -/
def Strategy := GameState → Digit

/-- Checks if a number represented by a list of digits is divisible by 3 -/
def isDivisibleBy3 (digits : List Digit) : Bool :=
  sorry

/-- Checks if two digits are in different residue classes modulo 3 -/
def isDifferentMod3 (d1 d2 : Digit) : Bool :=
  sorry

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Digit) : GameState :=
  sorry

/-- Plays the game to completion given strategies for both players -/
def playGame (aliceStrategy bobStrategy : Strategy) : List Digit :=
  sorry

/-- Helper function to safely get a digit from a list at a given index -/
def getDigit (digits : List Digit) (i : Nat) : Option Digit :=
  digits[i]?

theorem alice_can_prevent_divisibility :
  ∃ (aliceStrategy : Strategy),
    ∀ (bobStrategy : Strategy),
      let finalNumber := playGame aliceStrategy bobStrategy
      List.length finalNumber = 2018 ∧
      (∀ i : Nat, i < 2017 → 
        match getDigit finalNumber i, getDigit finalNumber (i+1) with
        | some d1, some d2 => isDifferentMod3 d1 d2
        | _, _ => true
      ) ∧
      ¬isDivisibleBy3 finalNumber :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_can_prevent_divisibility_l1328_132850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1328_132816

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 3}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x : ℝ | 0 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1328_132816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AEC_l1328_132830

-- Define the rectangle
noncomputable def rectangle_length : ℝ := 2
noncomputable def rectangle_width : ℝ := 1

-- Define points A, B, C, D
noncomputable def A : ℝ × ℝ := (0, 1)
noncomputable def B : ℝ × ℝ := (0, 0)
noncomputable def C : ℝ × ℝ := (2, 0)
noncomputable def D : ℝ × ℝ := (2, 1)

-- Define C' on AD
noncomputable def C' : ℝ × ℝ := (2, 1/4)

-- Define E as intersection of BC and AB
noncomputable def E : ℝ × ℝ := (8/3, 0)

-- Define the perimeter function
noncomputable def perimeter (a b c : ℝ × ℝ) : ℝ :=
  let d1 := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  let d2 := Real.sqrt ((b.1 - c.1)^2 + (b.2 - c.2)^2)
  let d3 := Real.sqrt ((c.1 - a.1)^2 + (c.2 - a.2)^2)
  d1 + d2 + d3

-- Theorem statement
theorem triangle_AEC'_perimeter : 
  perimeter A E C' = (41 + Real.sqrt 73) / 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AEC_l1328_132830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_k_value_max_k_value_l1328_132872

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 5 + Real.log x
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := k * x / (x + 1)

-- Statement for the first part of the problem
theorem tangent_line_k_value :
  ∃ k : ℝ, k = 9 ∧
  ∃ x₀ : ℝ, x₀ > 0 ∧
  (deriv (g k)) x₀ = 1 ∧
  g k x₀ = x₀ + 4 :=
sorry

-- Statement for the second part of the problem
theorem max_k_value :
  ∃ k : ℕ+, k = 7 ∧
  ∀ k' : ℕ+, k' > k →
  ∃ x : ℝ, x > 1 ∧ f x ≤ g k' x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_k_value_max_k_value_l1328_132872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1328_132854

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

theorem triangle_problem (t : Triangle)
  (h1 : t.b * Real.sin t.A = 3 * t.c * Real.sin t.B)
  (h2 : t.a = 3)
  (h3 : t.b = Real.sqrt 6) :
  Real.cos t.B = 2/3 ∧
  Real.sin (2 * t.B - π/6) = 2 * Real.sqrt 15 / 9 + 1/18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1328_132854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_pump_emptying_time_three_pump_emptying_time_minutes_l1328_132805

-- Define the pond and pump characteristics
structure Pond where
  initial_water : ℝ
  inflow_rate : ℝ
  pump_rate : ℝ

-- Define the emptying time function
noncomputable def emptying_time (p : Pond) (num_pumps : ℕ) : ℝ :=
  p.initial_water / (num_pumps * p.pump_rate - p.inflow_rate)

-- State the theorem
theorem three_pump_emptying_time (p : Pond) :
  emptying_time p 1 = 1 ∧ emptying_time p 2 = 1/3 →
  emptying_time p 3 = 1/5 := by
  sorry

-- Convert the result to minutes
noncomputable def emptying_time_minutes (p : Pond) (num_pumps : ℕ) : ℝ :=
  60 * emptying_time p num_pumps

-- State the final theorem in minutes
theorem three_pump_emptying_time_minutes (p : Pond) :
  emptying_time p 1 = 1 ∧ emptying_time p 2 = 1/3 →
  emptying_time_minutes p 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_pump_emptying_time_three_pump_emptying_time_minutes_l1328_132805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_squares_properties_l1328_132886

/-- Represents a point in the complex plane -/
structure ComplexPoint where
  re : ℝ
  im : ℝ

/-- Represents an affine regular polygon -/
structure AffineRegularPolygon where
  n : ℕ
  center : ComplexPoint
  vertices : Fin n → ComplexPoint

/-- Represents a square constructed on the side of a polygon -/
structure ExternalSquare where
  vertex1 : ComplexPoint
  vertex2 : ComplexPoint
  corner1 : ComplexPoint
  corner2 : ComplexPoint

/-- Calculates the length of a segment between two complex points -/
noncomputable def segmentLength (p1 p2 : ComplexPoint) : ℝ := sorry

/-- Checks if two segments are perpendicular -/
def isPerpendicular (s1start s1end s2start s2end : ComplexPoint) : Prop := sorry

/-- Main theorem about the properties of external squares on an affine regular polygon -/
theorem external_squares_properties (poly : AffineRegularPolygon) 
  (squares : Fin poly.n → ExternalSquare) : 
  (∀ j : Fin poly.n, 
    isPerpendicular (squares j).corner1 (squares j).corner2 poly.center (poly.vertices j))
  ∧ 
  (∀ j : Fin poly.n, 
    segmentLength (squares j).corner1 (squares j).corner2 / 
    segmentLength poly.center (poly.vertices j) = 
    2 * (1 - Real.cos (2 * Real.pi / (poly.n : ℝ)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_squares_properties_l1328_132886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_arrangement_exists_l1328_132840

/-- Represents a company with employees and teams -/
structure Company where
  employees : Finset Nat
  teams : Set (Finset Nat)
  employee_count : employees.card = 2015
  team_intersection : ∀ t1 t2, t1 ∈ teams → t2 ∈ teams → t1 ≠ t2 → (t1 ∩ t2).Nonempty

/-- Represents an arrangement of employees on a circular perimeter -/
def Arrangement (c : Company) := c.employees → ℝ

/-- The span of a team in an arrangement -/
noncomputable def TeamSpan (c : Company) (arr : Arrangement c) (team : Finset Nat) : ℝ :=
  sorry

/-- The theorem stating that there exists an arrangement where each team spans at least 1/3 of the perimeter -/
theorem company_arrangement_exists (c : Company) :
  ∃ (arr : Arrangement c), ∀ team ∈ c.teams, TeamSpan c arr team ≥ 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_arrangement_exists_l1328_132840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_B_in_A_equals_union_l1328_132864

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x + 2)}
def B : Set ℝ := {y | ∃ x, y = Real.sin x}

-- Define the complement of B in A
def complement_B_in_A : Set ℝ := A \ B

-- Theorem statement
theorem complement_B_in_A_equals_union :
  complement_B_in_A = Set.Ioo (-2) (-1) ∪ Set.Ioi 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_B_in_A_equals_union_l1328_132864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1328_132812

-- Define the curve C₁
def C₁ : ℝ → ℝ × ℝ := λ t ↦ (1 + 2*t, 2 - 2*t)

-- Define the curve C₂
noncomputable def C₂ : ℝ → ℝ × ℝ := λ θ ↦ (2*Real.cos θ + 2, 2*Real.sin θ)

-- Define the domain of θ
def θ_domain : Set ℝ := Set.Icc 0 (2*Real.pi)

-- Theorem statement
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ) (t₁ t₂ : ℝ) (θ₁ θ₂ : ℝ), 
    θ₁ ∈ θ_domain ∧ θ₂ ∈ θ_domain ∧
    C₁ t₁ = C₂ θ₁ ∧ C₁ t₂ = C₂ θ₂ ∧
    A = C₁ t₁ ∧ B = C₁ t₂ ∧
    Real.sqrt 14 = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1328_132812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_final_push_time_l1328_132839

/-- The time (in seconds) it takes for John to catch up to and overtake Steve in a race. -/
noncomputable def catchup_time (john_speed steve_speed initial_distance final_lead : ℝ) : ℝ :=
  (initial_distance + final_lead) / (john_speed - steve_speed)

/-- Theorem stating that John's final push lasts 34 seconds. -/
theorem johns_final_push_time :
  catchup_time 4.2 3.7 15 2 = 34 := by
  -- Unfold the definition of catchup_time
  unfold catchup_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_final_push_time_l1328_132839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_l1328_132819

def a : Fin 3 → ℝ := ![(-1 : ℝ), 2, -1]
def b : Fin 3 → ℝ := ![2, -7, 1]
def c₁ : Fin 3 → ℝ := 6 • a - 2 • b
def c₂ : Fin 3 → ℝ := b - 3 • a

theorem vectors_collinear : ∃ (k : ℝ), c₁ = k • c₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_l1328_132819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_result_l1328_132885

noncomputable def dilation (center : ℂ) (scale : ℝ) (point : ℂ) : ℂ :=
  center + scale • (point - center)

theorem dilation_result : 
  dilation (2 + 3*Complex.I) 3 (-1 - Complex.I) = -7 - 9*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_result_l1328_132885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_separators_l1328_132829

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for circles in a plane
structure Circle where
  center : Point
  radius : ℝ

-- Define a set of five points
def FivePoints : Type := Fin 5 → Point

-- Define the property that no three points are collinear
def NoThreeCollinear (points : FivePoints) : Prop := 
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    (points i).x * ((points j).y - (points k).y) +
    (points j).x * ((points k).y - (points i).y) +
    (points k).x * ((points i).y - (points j).y) ≠ 0

-- Define when a point is on a circle
def PointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define when a point is inside a circle
def PointInsideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

-- Define when a point is outside a circle
def PointOutsideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 > c.radius^2

-- Define the property that no four points are concyclic
def NoFourConcyclic (points : FivePoints) : Prop :=
  ∀ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l →
    ¬∃ (c : Circle), PointOnCircle (points i) c ∧ 
                     PointOnCircle (points j) c ∧ 
                     PointOnCircle (points k) c ∧ 
                     PointOnCircle (points l) c

-- Define a separator
def IsSeparator (c : Circle) (points : FivePoints) : Prop :=
  ∃ i j k l m, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ j ≠ l ∧ j ≠ m ∧ k ≠ m ∧
    PointOnCircle (points i) c ∧
    PointOnCircle (points j) c ∧
    PointOnCircle (points k) c ∧
    PointInsideCircle (points l) c ∧
    PointOutsideCircle (points m) c

-- The main theorem
theorem exactly_four_separators (points : FivePoints) 
  (h1 : NoThreeCollinear points) (h2 : NoFourConcyclic points) :
  ∃! (separators : Finset Circle), (∀ c ∈ separators, IsSeparator c points) ∧ separators.card = 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_separators_l1328_132829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1328_132852

theorem max_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_constraint : a^2 + b^2 + 4*c^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 2 ∧ 
  ∀ x, (x = a*b + 2*a*c + 3*(Real.sqrt 2)*b*c) → x ≤ max :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1328_132852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_monotonic_increasing_range_l1328_132815

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x - (1/2) * Real.sin (2 * x) + a * Real.sin x

-- Part I: Tangent line equation when a = 2
theorem tangent_line_equation :
  let a := 2
  let tangent_line (x y : ℝ) := x + y - 3 * Real.pi = 0
  ∃ k, tangent_line k (f a k) ∧ 
    (∀ x, x ≠ k → (f a x - f a k) / (x - k) < (if tangent_line x (f a x) then 1 else 0)) :=
sorry

-- Part II: Range of a for monotonically increasing f
theorem monotonic_increasing_range :
  {a : ℝ | ∀ x y, x < y → f a x < f a y} = Set.Icc (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_monotonic_increasing_range_l1328_132815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1328_132893

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem max_omega_value (ω φ : ℝ) :
  ω > 0 →
  |φ| ≤ π/2 →
  (∀ x, f ω φ (x - π/4) = -f ω φ (-x + π/4)) →
  (∀ x, f ω φ (x + π/4) = f ω φ (-x + π/4)) →
  (∀ x y, π/14 < x → x < y → y < 13*π/84 → f ω φ x < f ω φ y ∨ f ω φ x > f ω φ y) →
  ω ≤ 11 ∧ ∃ ω₀, ω₀ > 0 ∧ ω₀ ≤ 11 ∧
    (∀ ω' φ', ω' > 0 →
      |φ'| ≤ π/2 →
      (∀ x, f ω' φ' (x - π/4) = -f ω' φ' (-x + π/4)) →
      (∀ x, f ω' φ' (x + π/4) = f ω' φ' (-x + π/4)) →
      (∀ x y, π/14 < x → x < y → y < 13*π/84 → f ω' φ' x < f ω' φ' y ∨ f ω' φ' x > f ω' φ' y) →
      ω' ≤ ω₀) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1328_132893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1328_132834

-- Define the length of each train in meters
noncomputable def train_length : ℝ := 100

-- Define the speed of each train in km/h
noncomputable def train_speed : ℝ := 80

-- Define the relative speed of the trains in m/s
noncomputable def relative_speed : ℝ := 2 * train_speed * 1000 / 3600

-- Define the combined length of both trains
noncomputable def combined_length : ℝ := 2 * train_length

-- Theorem statement
theorem trains_crossing_time :
  combined_length / relative_speed = 2.25 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1328_132834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_l1328_132832

def custom_sequence (t : ℕ → ℕ) : Prop :=
  t 1 = 3 ∧
  t 5 = 18 ∧
  ∀ n : ℕ, n ≥ 2 → 4 * t n = t (n - 1) + t (n + 1)

theorem seventh_term (t : ℕ → ℕ) (h : custom_sequence t) : t 7 = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_l1328_132832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_3pi_characterization_l1328_132808

/-- The set of integers n for which the function f(x) = cos((n+1)x) * sin(8x/(n-2)) has a period of 3π -/
def period_3pi_integers : Set ℤ := {3, 1, 5, -1, 10, -6, 26, -22}

/-- The function f(x) defined in the problem -/
noncomputable def f (n : ℤ) (x : ℝ) : ℝ := Real.cos ((n + 1 : ℝ) * x) * Real.sin (8 * x / (n - 2 : ℝ))

/-- A function has period T if f(x + T) = f(x) for all x -/
def has_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- Theorem stating the characterization of integers n for which f has period 3π -/
theorem period_3pi_characterization (n : ℤ) :
  has_period (f n) (3 * Real.pi) ↔ n ∈ period_3pi_integers := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_3pi_characterization_l1328_132808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l1328_132806

def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-2, -3)

theorem projection_vector :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_squared := a.1^2 + a.2^2
  let scalar := dot_product / magnitude_squared
  let proj := (scalar * a.1, scalar * a.2)
  proj = (-2/5, 1/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l1328_132806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l1328_132890

-- Define an isosceles triangle with side lengths 8 and 9
structure IsoscelesTriangle :=
  (a b : ℝ)
  (isIsosceles : a = b ∨ a = 9 ∨ b = 9)
  (hasLengths : (a = 8 ∧ b = 9) ∨ (a = 9 ∧ b = 8))

-- Define the perimeter of the triangle
noncomputable def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.a + t.b + (if t.a = t.b then 9 else 8)

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  perimeter t = 25 ∨ perimeter t = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l1328_132890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l1328_132800

-- Define the point in rectangular coordinates
noncomputable def x : ℝ := 2 * Real.sqrt 2
def y : ℝ := -2

-- Define the polar coordinates
noncomputable def r : ℝ := 2 * Real.sqrt 3
noncomputable def θ : ℝ := 7 * Real.pi / 4

-- Theorem statement
theorem rectangular_to_polar_conversion :
  (r * Real.cos θ = x) ∧ 
  (r * Real.sin θ = y) ∧ 
  (r > 0) ∧ 
  (0 ≤ θ) ∧ 
  (θ < 2 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_conversion_l1328_132800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chen_trip_distance_l1328_132883

/-- Taxi fare structure -/
structure TaxiFare where
  initial_fare : ℕ
  initial_distance : ℕ
  additional_rate : ℕ
  waiting_time_rate : ℕ

/-- Calculate the fare for a given distance and waiting time -/
def calculate_fare (tf : TaxiFare) (distance : ℕ) (waiting_time : ℕ) : ℕ :=
  tf.initial_fare + max 0 (distance - tf.initial_distance) * tf.additional_rate +
  (waiting_time / tf.waiting_time_rate) * tf.additional_rate

/-- Theorem: Mr. Chen's trip distance was between 5 and 6 km -/
theorem chen_trip_distance (tf : TaxiFare) (x : ℕ)
  (h1 : tf.initial_fare = 6)
  (h2 : tf.initial_distance = 2)
  (h3 : tf.additional_rate = 3)  -- Using 3 instead of 1.5 to work with ℕ
  (h4 : tf.waiting_time_rate = 6)
  (h5 : calculate_fare tf x 2 = 15) :  -- Using 2 instead of 11.5/60 to work with ℕ
  5 < x ∧ x < 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chen_trip_distance_l1328_132883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_inequality_for_positive_integers_l1328_132899

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 0.5 * x^2 - (1 + a) * x

-- Theorem 1: f(x) ≥ 0 for all x > 0 if and only if a ≤ -1/2
theorem f_nonnegative_iff (a : ℝ) : 
  (∀ x > 0, f a x ≥ 0) ↔ a ≤ -1/2 := by sorry

-- Theorem 2: Inequality for positive integers m and n
theorem inequality_for_positive_integers (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (Finset.range n).sum (λ i => 1 / Real.log (m + i + 1 : ℝ)) > n / (m * (m + n)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_inequality_for_positive_integers_l1328_132899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expressions_l1328_132809

noncomputable def K₁ (x : ℝ) : ℝ := 
  (((2*x+1)/(x+1)) - ((2*x-1)/x)) / (((2*x-1)/(x-1)) - ((2*x+1)/x))

noncomputable def K₂ (x : ℝ) : ℝ := 
  (x^2+x)/(x^2+3*x+2) - (x^2-3*x+2)/(x^2-x) + (x^2-7*x+12)/(x^2-5*x+6)

theorem simplify_expressions (x : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 1) (hx3 : x ≠ -1) (hx4 : x ≠ 2) : 
  (K₁ x = (x-1)/(x+1)) ∧ 
  (K₂ x = 1 - 2/(x+2) + 2/x - 4/(x*(x-2))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expressions_l1328_132809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_8_equals_58_l1328_132818

/-- A quadratic polynomial P(x) such that P(P(x)) = x^4 - 2x^3 + 4x^2 - 3x + 4 -/
noncomputable def P : ℝ → ℝ := sorry

/-- The property that P(P(x)) = x^4 - 2x^3 + 4x^2 - 3x + 4 for all x -/
axiom P_property : ∀ x, P (P x) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4

/-- P is a quadratic polynomial -/
axiom P_quadratic : ∃ a b c, ∀ x, P x = a*x^2 + b*x + c

/-- Theorem: P(8) = 58 -/
theorem P_8_equals_58 : P 8 = 58 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_8_equals_58_l1328_132818
