import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_households_surveyed_sum_of_categories_l366_36629

/-- Represents the number of households in a marketing survey --/
structure HouseholdSurvey where
  total : ℕ
  inv : total > 0

/-- Properties of the household survey --/
structure SurveyProperties (s : HouseholdSurvey) where
  total : s.total = 200
  neither : ℕ
  onlyA : ℕ
  both : ℕ
  neitherCount : neither = 80
  onlyACount : onlyA = 60
  bothCount : both = 10
  onlyBCount : ℕ
  onlyBRelation : onlyBCount = 3 * both

/-- Theorem stating that the total number of households surveyed is 200 --/
theorem total_households_surveyed (s : HouseholdSurvey) (props : SurveyProperties s) :
  s.total = 200 := by
  exact props.total

/-- Theorem stating that the sum of all categories equals the total --/
theorem sum_of_categories (s : HouseholdSurvey) (props : SurveyProperties s) :
  props.neither + props.onlyA + props.both + props.onlyBCount = s.total := by
  sorry

#check total_households_surveyed
#check sum_of_categories

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_households_surveyed_sum_of_categories_l366_36629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_problem_l366_36654

theorem congruence_problem (a b : ℤ) (h1 : a ≡ 27 [ZMOD 53]) (h2 : b ≡ 88 [ZMOD 53]) :
  ∃ n : ℤ, n ∈ Set.Icc 120 171 ∧ a - b ≡ n [ZMOD 53] :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_problem_l366_36654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_theorem_l366_36640

/-- Represents the sales and pricing data for Dafu Supermarket --/
structure SupermarketData where
  initial_cost : ℝ
  initial_price : ℝ
  march_sales : ℝ
  april_may_sales : ℝ
  price_reduction_effect : ℝ

/-- Calculates the average monthly growth rate --/
noncomputable def avg_monthly_growth_rate (data : SupermarketData) : ℝ :=
  (data.april_may_sales / data.march_sales) ^ (1/2) - 1

/-- Calculates the monthly profit after price reduction --/
def monthly_profit (data : SupermarketData) (price_reduction : ℝ) : ℝ :=
  (data.initial_price - price_reduction - data.initial_cost) *
  (data.april_may_sales + data.price_reduction_effect * price_reduction)

/-- The main theorem stating the properties of the supermarket data --/
theorem supermarket_theorem (data : SupermarketData) 
  (h1 : data.initial_cost = 25)
  (h2 : data.initial_price = 40)
  (h3 : data.march_sales = 256)
  (h4 : data.april_may_sales = 400)
  (h5 : data.price_reduction_effect = 5) :
  avg_monthly_growth_rate data = 0.25 ∧
  ∃ (price_reduction : ℝ), 
    price_reduction = 5 ∧ 
    monthly_profit data price_reduction = 4250 := by
  sorry

#eval "Supermarket theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_theorem_l366_36640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l366_36659

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

theorem odd_function_properties (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_incr_pos : IncreasingOn f (Set.Ioi 0)) :
    (f 0 = 0) ∧ 
    (IncreasingOn f (Set.Iio 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l366_36659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_reversal_l366_36670

/-- Represents a light ray in a 2D plane -/
structure LightRay where
  direction : ℝ

/-- Represents a mirror in a 2D plane -/
structure Mirror where
  angle : ℝ

/-- Represents a system of two mirrors -/
structure MirrorSystem where
  angle : ℝ

/-- Reflects a light ray off a mirror -/
def reflect (ray : LightRay) (mirror : Mirror) : LightRay :=
  { direction := mirror.angle - ray.direction }

/-- Performs multiple reflections of a light ray in a mirror system -/
def multipleReflections (ray : LightRay) (system : MirrorSystem) (n : ℕ) : LightRay :=
  match n with
  | 0 => ray
  | _ + 1 => 
    let newRay := reflect ray { angle := system.angle }
    reflect newRay { angle := 2 * system.angle }

/-- Theorem: A light ray will reverse its direction after a finite number of reflections
    in a mirror system where the angle between mirrors is 90°/n for some integer n -/
theorem light_ray_reversal (ray : LightRay) (n : ℕ) (hn : n > 0) :
  ∃ (k : ℕ), (multipleReflections ray { angle := 90 / n } (2 * k)).direction = ray.direction + 180 := by
  sorry

-- Example (commented out as it may not be necessary for building)
-- #eval light_ray_reversal { direction := 30 } 3 (by norm_num)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_reversal_l366_36670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_largest_angle_specific_triangle_l366_36636

theorem cosine_largest_angle_specific_triangle : 
  ∀ (a b c : ℝ), 
  a = 4 ∧ b = 5 ∧ c = 6 →
  (a^2 + b^2 - c^2) / (2 * a * b) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_largest_angle_specific_triangle_l366_36636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l366_36604

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x

theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 3 = 1/8) :
  (a = 1/2) ∧ 
  (∀ x ∈ Set.Icc (-1/2 : ℝ) 2, f a x ≤ Real.sqrt 2) ∧
  (∃ c ∈ Set.Ioo (0 : ℝ) 1, g a c = 0) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l366_36604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phase_shift_correct_min_phase_shift_minimal_l366_36680

/-- The minimum positive phase shift that transforms sin x + √3 cos x to sin x - √3 cos x -/
noncomputable def min_phase_shift : ℝ :=
  2 * Real.pi / 3

theorem min_phase_shift_correct (x : ℝ) : 
  2 * Real.sin (x - min_phase_shift + Real.pi / 3) = 2 * Real.sin (x - Real.pi / 3) :=
by sorry

theorem min_phase_shift_minimal (φ : ℝ) :
  (∀ x, 2 * Real.sin (x - φ + Real.pi / 3) = 2 * Real.sin (x - Real.pi / 3)) →
  φ > 0 →
  φ ≥ min_phase_shift :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phase_shift_correct_min_phase_shift_minimal_l366_36680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_zero_complex_square_positive_real_complex_chain_product_positive_real_l366_36669

-- Part (1)
theorem complex_product_zero (α β : ℂ) : α * β = 0 → α = 0 ∨ β = 0 := by sorry

-- Part (2)
theorem complex_square_positive_real (α : ℂ) : 0 < (α * α).re ∧ (α * α).im = 0 → α.im = 0 := by sorry

-- Part (3)
theorem complex_chain_product_positive_real (n : ℕ) (α : Fin (2*n+1) → ℂ) :
  (∀ k : Fin (2*n+1), 0 < (α k * α (k.succ)).re ∧ (α k * α (k.succ)).im = 0) →
  (∀ k : Fin (2*n+1), (α k).im = 0) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_zero_complex_square_positive_real_complex_chain_product_positive_real_l366_36669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l366_36698

-- Define the function f
noncomputable def f (x : ℝ) := Real.log (1 + |x|) - 1 / (1 + x^2)

-- State the theorem
theorem f_inequality_range : 
  ∀ x : ℝ, f x > f (2*x - 1) ↔ x ∈ Set.Ioo (1/3 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l366_36698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l366_36676

/-- The eccentricity of an ellipse given specific conditions -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let C := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}
  let l := {p : ℝ × ℝ | p.2 = -1/2 * (p.1 - 1) + 1}
  let M := (1, 1)
  ∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l ∧ M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  Real.sqrt (1 - b^2 / a^2) = Real.sqrt 2 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l366_36676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_shifted_sine_l366_36696

noncomputable def f (x : ℝ) := -Real.sin (x + Real.pi/6)

theorem symmetry_of_shifted_sine :
  ∀ (x : ℝ), f (Real.pi/3 + (Real.pi/3 - x)) = f x := by
  sorry

#check symmetry_of_shifted_sine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_shifted_sine_l366_36696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_implies_a_in_range_l366_36661

/-- The function f(x) = (1/2)x^2 - 16ln(x) -/
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 16 * Real.log x

/-- f(x) is monotonically decreasing on the interval [a-1, a+2] -/
def is_monotone_decreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc (a - 1) (a + 2) → y ∈ Set.Icc (a - 1) (a + 2) → x ≤ y → f y ≤ f x

/-- Main theorem: If f(x) is monotonically decreasing on [a-1, a+2], then a ∈ (1, 2] -/
theorem f_monotone_decreasing_implies_a_in_range (a : ℝ) :
  is_monotone_decreasing f a → a ∈ Set.Ioo 1 2 ∪ {2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_implies_a_in_range_l366_36661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l366_36693

-- Define the curves
def curve1 (x y : ℝ) : Prop := (x + y)^2 = 16
def curve2 (x y : ℝ) : Prop := (2*x - y)^2 = 4

-- Define the region bounded by the curves
def bounded_region (x y : ℝ) : Prop := curve1 x y ∧ curve2 x y

-- Define the area of the region (this is a placeholder function)
noncomputable def area_of_region : ℝ → ℝ → ℝ := sorry

-- Statement to prove
theorem area_of_bounded_region :
  ∃ (A : ℝ), A = (16 * Real.sqrt 10) / 5 ∧
  (∀ x y : ℝ, bounded_region x y → area_of_region x y = A) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l366_36693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bedbug_population_growth_rate_l366_36639

/-- The daily rate of increase for a bedbug population -/
noncomputable def daily_rate_of_increase (initial : ℕ) (days : ℕ) (final : ℕ) : ℝ :=
  (final / initial : ℝ) ^ (1 / days : ℝ)

/-- Theorem stating the daily rate of increase for the given bedbug problem -/
theorem bedbug_population_growth_rate : 
  daily_rate_of_increase 30 4 810 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bedbug_population_growth_rate_l366_36639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_greater_than_threshold_l366_36628

noncomputable def numbers : List ℝ := [1.4, 9/10, 1.2, 0.5, 13/10]

def threshold : ℝ := 1.1

theorem smallest_greater_than_threshold :
  (numbers.filter (λ x => x ≥ threshold)).minimum? = some 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_greater_than_threshold_l366_36628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshoppers_never_overlap_l366_36626

/-- Represents a position in 2D space -/
structure Position where
  x : ℚ
  y : ℚ

/-- Represents the state of the grasshoppers -/
structure GrasshopperState where
  positions : Fin 4 → Position

/-- Calculates the centroid of three positions -/
def centroid (p1 p2 p3 : Position) : Position :=
  { x := (p1.x + p2.x + p3.x) / 3,
    y := (p1.y + p2.y + p3.y) / 3 }

/-- Calculates the new position after a jump -/
def jump (current : Position) (center : Position) : Position :=
  { x := 2 * center.x - current.x,
    y := 2 * center.y - current.y }

/-- Represents a single jump operation -/
def jumpOperation (state : GrasshopperState) (i : Fin 4) : GrasshopperState :=
  let otherPositions := fun j => if j = i then state.positions j else state.positions j
  let center := centroid (otherPositions 0) (otherPositions 1) (otherPositions 2)
  { positions := fun j => if j = i then jump (state.positions i) center else state.positions j }

/-- Initial state of the grasshoppers -/
def initialState (n : ℕ) : GrasshopperState :=
  { positions := fun i =>
      match i with
      | 0 => { x := 0, y := 0 }
      | 1 => { x := 0, y := (3 : ℚ)^n }
      | 2 => { x := (3 : ℚ)^n, y := (3 : ℚ)^n }
      | 3 => { x := (3 : ℚ)^n, y := 0 } }

/-- Theorem stating that grasshoppers never land on the same position -/
theorem grasshoppers_never_overlap (n : ℕ) :
  ∀ (sequence : List (Fin 4)),
    let finalState := sequence.foldl jumpOperation (initialState n)
    ∀ (i j : Fin 4), i ≠ j → finalState.positions i ≠ finalState.positions j := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshoppers_never_overlap_l366_36626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beibei_distance_theorem_l366_36605

/-- Converts centimeters to kilometers -/
noncomputable def cm_to_km (cm : ℝ) : ℝ := cm / 100000

/-- Calculates the actual distance given the map distance and scale -/
noncomputable def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  cm_to_km (map_distance * scale)

theorem beibei_distance_theorem (map_distance : ℝ) (scale : ℝ) :
  map_distance = 8.5 ∧ scale = 400000 →
  actual_distance map_distance scale = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beibei_distance_theorem_l366_36605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_apex_angle_l366_36600

/-- Function to calculate the cone apex angle given sphere radii -/
noncomputable def cone_apex_angle_for_spheres (r1 r2 r3 : ℝ) : ℝ :=
  let d13 := Real.sqrt ((r1 + r3)^2 - (r1 - r3)^2)
  let h := Real.sqrt (d13^2 - r1^2)
  let angle := Real.arctan (r3 / h)
  2 * angle

/-- The angle at the apex of a cone touching three spheres externally -/
theorem cone_apex_angle (r1 r2 r3 : ℝ) (h1 : r1 = 2) (h2 : r2 = 2) (h3 : r3 = 5) : 
  let angle := Real.arctan (1 / 72)
  2 * angle = cone_apex_angle_for_spheres r1 r2 r3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_apex_angle_l366_36600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concave_function_max_no_min_l366_36684

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/6) * x^3 - (1/2) * a * x^2 + x

-- Define the first derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * x + 1

-- Define the second derivative of f
def f'' (a : ℝ) (x : ℝ) : ℝ := x - a

-- Define concavity
def is_concave (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  ∀ x ∈ Set.Ioo m n, DifferentiableAt ℝ f x ∧ 
  ∀ y ∈ Set.Ioo m n, DifferentiableAt ℝ (deriv f) y ∧ deriv (deriv f) y < 0

-- State the theorem
theorem concave_function_max_no_min (a : ℝ) (h₁ : a ≤ 2) 
  (h₂ : is_concave (f a) (-1) 2) :
  ∃ x₀ ∈ Set.Ioo (-1) 2, IsLocalMax (f a) x₀ ∧
  ∀ x ∈ Set.Ioo (-1) 2, ¬IsLocalMin (f a) x :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concave_function_max_no_min_l366_36684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l366_36660

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 6)^2 + y^2 = 25

-- Define the line l in polar form
def line_l (θ α : ℝ) : Prop := θ = α

-- Define the distance between intersection points A and B
noncomputable def distance_AB : ℝ := Real.sqrt 10

-- Theorem statement
theorem slope_of_line_l (x y θ α : ℝ) :
  circle_C x y →
  line_l θ α →
  ∃ (xA yA xB yB : ℝ),
    circle_C xA yA ∧ circle_C xB yB ∧
    line_l θ α →
    (xA - xB)^2 + (yA - yB)^2 = distance_AB^2 →
    (∃ (m : ℝ), m = Real.sqrt 15 / 3 ∨ m = -Real.sqrt 15 / 3 ∧ 
      ∀ (x y : ℝ), line_l θ α → y = m * x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l366_36660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l366_36667

theorem trigonometric_simplification (x : ℝ) (h1 : Real.sin x ≠ 0) (h2 : 1 + Real.cos x ≠ 0) :
  (Real.sin x) / (1 + Real.cos x) + (1 + Real.cos x) / (Real.sin x) = 2 * (1 / Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l366_36667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_sum_product_reciprocals_power_l366_36681

/-- Represents a geometric progression with 5 terms -/
structure GeometricProgression where
  b : ℝ  -- first term
  q : ℝ  -- common ratio

/-- The product of the 5 terms in the geometric progression -/
noncomputable def T (gp : GeometricProgression) : ℝ :=
  gp.b^5 * gp.q^10

/-- The sum of the 5 terms in the geometric progression -/
noncomputable def U (gp : GeometricProgression) : ℝ :=
  gp.b * (1 - gp.q^5) / (1 - gp.q)

/-- The sum of the reciprocals of the 5 terms in the geometric progression -/
noncomputable def U_prime (gp : GeometricProgression) : ℝ :=
  (gp.q^5 - 1) / (gp.b * (gp.q - 1))

/-- Theorem stating the relationship between T, U, and U' -/
theorem product_equals_sum_product_reciprocals_power (gp : GeometricProgression) :
  T gp = (U gp * U_prime gp)^(5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_sum_product_reciprocals_power_l366_36681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_a_positive_b_positive_c_positive_c_not_divisible_by_prime_square_l366_36658

noncomputable section

-- Define the expression
noncomputable def expression (x : ℝ) : ℝ :=
  (Real.sqrt 3 - 1)^(2 - Real.sqrt 5) / (Real.sqrt 3 + 1)^(2 + Real.sqrt 5)

-- Define the simplified form
noncomputable def simplified_form (x : ℝ) : ℝ :=
  12 * 2^(2 + Real.sqrt 5) - 6 * 2^(2 + Real.sqrt 5) * Real.sqrt 3

-- Theorem statement
theorem expression_simplification :
  ∀ x : ℝ, expression x = simplified_form x :=
by
  sorry

-- Additional conditions
theorem a_positive : ∃ a : ℝ, a > 0 ∧ 12 * 2^(2 + Real.sqrt 5) = a :=
by
  sorry

theorem b_positive : ∃ b : ℝ, b > 0 ∧ 6 * 2^(2 + Real.sqrt 5) = b :=
by
  sorry

theorem c_positive : 3 > 0 :=
by
  sorry

theorem c_not_divisible_by_prime_square :
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ 3) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_a_positive_b_positive_c_positive_c_not_divisible_by_prime_square_l366_36658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l366_36642

theorem problem_statement (a b c : ℝ) :
  (∀ c, c ≠ 0 → (a * c^2 > b * c^2 → a > b)) ∧ (|a| < b → a + b > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l366_36642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_l366_36692

/-- The average speed formula for a round trip -/
noncomputable def average_speed (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  (2 * speed1 * speed2) / (speed1 + speed2)

/-- Theorem stating that given the average speed and return speed, we can determine the outbound speed -/
theorem round_trip_speed
  (avg_speed : ℝ)
  (return_speed : ℝ)
  (h1 : avg_speed = 5.090909090909091)
  (h2 : return_speed = 7)
  : ∃ (outbound_speed : ℝ), 
    average_speed outbound_speed return_speed = avg_speed ∧ 
    outbound_speed = 4 := by
  sorry

#check round_trip_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_l366_36692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l366_36672

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
variable (b : ℝ × ℝ)
variable (c : ℝ × ℝ)

-- Define the angle between a and b
variable (θ : ℝ)

-- Theorem statement
theorem vector_problem :
  (∃ (x y : ℝ), c = (x, y) ∧ (x * -2 = y * 1) ∧ x^2 + y^2 = 20) ∧
  (‖b‖ = 1) ∧
  ((a.1 + b.1, a.2 + b.2) • (a.1 - 2*b.1, a.2 - 2*b.2) = 0) →
  (c = (-2, 4) ∨ c = (2, -4)) ∧
  (Real.cos θ = 3 * Real.sqrt 5 / 5) := by
  sorry

#check vector_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l366_36672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l366_36614

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the midpoint of AB
def midpoint_AB : ℝ × ℝ := (2, 1)

-- Define the length of AB
noncomputable def length_AB : ℝ := (4 / 3) * Real.sqrt 3

-- Theorem statement
theorem ellipse_intersection_theorem :
  ∃ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = midpoint_AB ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = length_AB :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l366_36614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_radius_l366_36619

/-- Two concentric circles with radii in ratio 1:3 -/
structure ConcentricCircles where
  small_radius : ℝ
  large_radius : ℝ
  ratio : large_radius = 3 * small_radius

/-- Point on the larger circle -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line segment between two points -/
noncomputable def LineSegment (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

theorem larger_circle_radius
  (circles : ConcentricCircles)
  (a b c : Point)
  (ac_diameter : LineSegment a c = 2 * circles.large_radius)
  (bc_tangent : ∃ d : Point, LineSegment b d = circles.small_radius ∧ 
                LineSegment c d = circles.large_radius)
  (ab_length : LineSegment a b = 12) :
  circles.large_radius = 18 := by
  sorry

#check larger_circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_radius_l366_36619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_one_l366_36686

/-- The function f(x) = 1/x + 2x -/
noncomputable def f (x : ℝ) : ℝ := 1/x + 2*x

/-- The angle of inclination of the tangent line to f(x) at x = 1 -/
noncomputable def angle_of_inclination : ℝ := Real.pi / 4

/-- Theorem: The angle of inclination of the tangent line to f(x) at x = 1 is π/4 -/
theorem tangent_angle_at_one :
  let tangent_slope := deriv f 1
  Real.arctan tangent_slope = angle_of_inclination := by
  sorry

#check tangent_angle_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_one_l366_36686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_and_range_l366_36694

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 3) + Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x) ^ 2

theorem f_symmetry_and_range :
  (∃ (a : ℝ), a > 0 ∧ a = Real.pi / 12 ∧ ∀ (x : ℝ), f (a + x) = f (a - x)) ∧
  (∃ (x₀ : ℝ), x₀ ∈ Set.Icc 0 (5 * Real.pi / 12) ∧
    (∀ (m : ℝ), m * f x₀ - 2 = 0 → (m ≥ 1 ∨ m ≤ -2)) ∧
    (∀ (m : ℝ), (m ≥ 1 ∨ m ≤ -2) → ∃ (x₀ : ℝ), x₀ ∈ Set.Icc 0 (5 * Real.pi / 12) ∧ m * f x₀ - 2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_and_range_l366_36694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_determine_counterfeit_coin_l366_36634

/-- Represents the weight of a coin -/
def CoinWeight : Type := ℕ

/-- Represents the result of a weighing -/
def WeighingResult : Type := ℤ

/-- The total number of coins -/
def totalCoins : ℕ := 101

/-- The number of counterfeit coins -/
def counterfeitCoins : ℕ := 50

/-- The weight difference between genuine and counterfeit coins -/
def weightDifference : ℕ := 1

/-- A function that performs a single weighing of two groups of coins -/
def weighCoins (group1 : List CoinWeight) (group2 : List CoinWeight) : WeighingResult :=
  sorry

/-- Theorem stating that it's possible to determine if a coin is counterfeit in one weighing -/
theorem can_determine_counterfeit_coin :
  ∃ (strategy : CoinWeight → Bool),
    ∀ (coin : CoinWeight),
      (strategy coin = true → coin = weightDifference) ∧
      (strategy coin = false → coin = weightDifference.succ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_determine_counterfeit_coin_l366_36634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_2_sqrt_11_l366_36699

/-- Two intersecting circles with centers O and Q, and radii in ratio 3:2 -/
structure IntersectingCircles where
  /-- Radius of the smaller circle -/
  R : ℝ
  /-- Center of the smaller circle -/
  O : ℝ × ℝ
  /-- Center of the larger circle -/
  Q : ℝ × ℝ
  /-- The segment OQ is divided by the intersection point N with the common chord -/
  N : ℝ × ℝ
  /-- ON = 4 -/
  h_ON : dist O N = 4
  /-- NQ = 1 -/
  h_NQ : dist N Q = 1
  /-- OQ = 5R -/
  h_OQ : dist O Q = 5 * R

/-- The length of the common chord of two intersecting circles -/
noncomputable def common_chord_length (c : IntersectingCircles) : ℝ :=
  2 * Real.sqrt 11

/-- 
Theorem: The length of the common chord of two intersecting circles,
where the segment connecting their centers is divided by the common chord
into segments of 4 and 1, and the radii of the circles are in the ratio 3:2,
is equal to 2√11.
-/
theorem common_chord_length_is_2_sqrt_11 (c : IntersectingCircles) :
  common_chord_length c = 2 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_2_sqrt_11_l366_36699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l366_36679

-- Define the function f(x) = 1 / (x - 2)
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 2)

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, x ≠ 2 ↔ ∃ y : ℝ, f x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l366_36679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_interior_diagonals_l366_36623

/-- A dodecahedron is a 3-dimensional figure with 12 pentagonal faces and 20 vertices,
    with 3 faces meeting at each vertex. -/
structure Dodecahedron where
  faces : ℕ
  vertices : ℕ
  faces_per_vertex : ℕ
  faces_are_pentagonal : faces = 12
  vertex_count : vertices = 20
  face_meeting : faces_per_vertex = 3

/-- An interior diagonal of a dodecahedron is a segment connecting two vertices
    which do not lie on a common face. -/
def interior_diagonal (d : Dodecahedron) : ℕ → Prop := sorry

/-- The number of interior diagonals in a dodecahedron -/
def num_interior_diagonals (d : Dodecahedron) : ℕ := sorry

/-- Theorem: A dodecahedron has 130 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  num_interior_diagonals d = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_interior_diagonals_l366_36623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_one_range_of_a_for_one_zero_l366_36601

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.sin x - a) * (a - Real.cos x) + Real.sqrt 2 * a

-- Statement for part 1
theorem range_of_f_when_a_is_one :
  Set.range (f 1) = Set.Icc (-3/2 : ℝ) (Real.sqrt 2) := by sorry

-- Statement for part 2
theorem range_of_a_for_one_zero :
  ∀ a : ℝ, a ≥ 1 →
  (∃! x, x ∈ Set.Icc 0 Real.pi ∧ f a x = 0) ↔
  (1 ≤ a ∧ a < Real.sqrt 2 + 1) ∨ a = Real.sqrt 2 + Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_one_range_of_a_for_one_zero_l366_36601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_range_l366_36683

/-- The parabola defined by x^2 = 8y -/
def Parabola (x y : ℝ) : Prop := x^2 = 8*y

/-- The focus of the parabola -/
def FocusY : ℝ := 2

/-- The directrix of the parabola -/
def DirectrixY : ℝ := -2

/-- The distance from a point (x, y) to the focus -/
noncomputable def DistanceToFocus (x y : ℝ) : ℝ := Real.sqrt ((x - 0)^2 + (y - FocusY)^2)

/-- The distance from a point (x, y) to the directrix -/
def DistanceToDirectrix (y : ℝ) : ℝ := y - DirectrixY

theorem parabola_point_range (x₀ y₀ : ℝ) :
  Parabola x₀ y₀ →
  DistanceToFocus x₀ y₀ > DistanceToDirectrix y₀ →
  y₀ > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_range_l366_36683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unequal_gender_distribution_probability_l366_36602

def num_children : ℕ := 8
def prob_male : ℝ := 0.4
def prob_female : ℝ := 0.6

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

noncomputable def probability_unequal_gender_distribution : ℝ :=
  Finset.sum (Finset.range (num_children + 1)) (fun k =>
    if k ≠ num_children / 2 
    then binomial_probability num_children k prob_male
    else 0)

theorem unequal_gender_distribution_probability :
  probability_unequal_gender_distribution = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unequal_gender_distribution_probability_l366_36602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l366_36616

-- Define the curves and ray
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

def curve_C1 : ℝ → ℝ := λ _ => 2

def ray_l (a₀ : ℝ) : ℝ := a₀

-- Define the range of a₀
def a₀_range : Set ℝ := { x | 0 ≤ x ∧ x ≤ Real.pi / 2 }

-- Define the distance function
noncomputable def distance (a₀ : ℝ) : ℝ :=
  |2 - Real.sqrt (3 / (1 + 2 * Real.sin a₀ ^ 2))|

-- State the theorem
theorem distance_range :
  ∀ a₀ ∈ a₀_range, 2 - Real.sqrt 3 ≤ distance a₀ ∧ distance a₀ ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l366_36616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_in_tank_l366_36615

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular tank -/
noncomputable def tankVolume (d : TankDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the volume of a sphere -/
noncomputable def sphereVolume (radius : ℝ) : ℝ :=
  (4 / 3) * Real.pi * radius ^ 3

/-- Theorem: Unoccupied volume in the tank -/
theorem unoccupied_volume_in_tank (tank : TankDimensions) 
    (h1 : tank.length = 12)
    (h2 : tank.width = 8)
    (h3 : tank.height = 10)
    (water_fraction : ℝ)
    (h4 : water_fraction = 1 / 3)
    (num_balls : ℕ)
    (h5 : num_balls = 6)
    (ball_radius : ℝ)
    (h6 : ball_radius = 1) :
    tankVolume tank - (water_fraction * tankVolume tank + ↑num_balls * sphereVolume ball_radius) = 640 - 8 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_in_tank_l366_36615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_expressions_l366_36631

theorem negative_expressions (x : ℝ) (h : x > 0) : -x^2 < 0 ∧ Real.rpow (-x) (1/3) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_expressions_l366_36631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_external_tangent_16_25_l366_36612

/-- The length of the common external tangent between two externally touching circles -/
noncomputable def common_external_tangent (r₁ r₂ : ℝ) : ℝ :=
  Real.sqrt ((r₁ + r₂)^2 - (r₁ - r₂)^2)

/-- Theorem: The length of the common external tangent between two externally touching circles with radii 16 and 25 is 40 -/
theorem common_external_tangent_16_25 :
  common_external_tangent 16 25 = 40 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval common_external_tangent 16 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_external_tangent_16_25_l366_36612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_8_l366_36609

/-- Arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_3, and a_4 form a geometric sequence -/
def geometric_subsequence (a : ℕ → ℚ) : Prop :=
  a 3 ^ 2 = a 1 * a 4

/-- Sum of first n terms of an arithmetic sequence -/
def arithmetic_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a 1 + (n - 1 : ℚ) * 2)

theorem arithmetic_sequence_sum_8 (a : ℕ → ℚ) :
  arithmetic_sequence a → geometric_subsequence a → arithmetic_sum a 8 = -8 := by
  sorry

#eval arithmetic_sum (fun n => -8 + 2 * (n - 1)) 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_8_l366_36609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cotangent_five_expression_zero_l366_36653

theorem cotangent_five_expression_zero (θ : ℝ) (h : Real.tan (Real.pi / 2 - θ) = 5) :
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cotangent_five_expression_zero_l366_36653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_workers_count_l366_36691

/-- Represents the number of workers in the factory -/
def num_workers : ℕ := 8

/-- Represents the initial average salary of workers and old supervisor -/
def initial_average_salary : ℚ := 430

/-- Represents the old supervisor's salary -/
def old_supervisor_salary : ℚ := 870

/-- Represents the new average salary after supervisor change -/
def new_average_salary : ℚ := 420

/-- Represents the new supervisor's salary -/
def new_supervisor_salary : ℚ := 780

/-- Represents the total number of people after the new supervisor joins -/
def total_people : ℕ := 9

theorem factory_workers_count : num_workers = 8 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_workers_count_l366_36691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l366_36611

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  sides_positive : a > 0 ∧ b > 0 ∧ c > 0
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h1 : Real.cos (t.A - t.B) = Real.sqrt 3 * Real.sin t.B - Real.cos t.C)
  (h2 : t.b = 2) :
  (t.A = π/3 ∨ t.A = 2*π/3) ∧ 
  (3 + Real.sqrt 3 < t.a + t.b + t.c) ∧ 
  (t.a + t.b + t.c < 6 + 2*Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l366_36611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PAB_l366_36646

/-- The area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

/-- The maximum area of triangle PAB given specific conditions -/
theorem max_area_triangle_PAB :
  ∀ (k : ℝ) (M N P : ℝ × ℝ),
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (0, 2)
  let circle := λ (p : ℝ × ℝ) => p.1^2 + p.2^2 + k*p.1 = 0
  let symmetry_line := λ (p : ℝ × ℝ) => p.1 - p.2 - 1 = 0
  M ≠ N →
  circle M →
  circle N →
  symmetry_line ((M.1 + N.1)/2, (M.2 + N.2)/2) →
  (∀ (Q : ℝ × ℝ), circle Q →
    area_triangle A B Q ≤ (3 + Real.sqrt 2)) ∧
  (∃ (R : ℝ × ℝ), circle R ∧
    area_triangle A B R = (3 + Real.sqrt 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PAB_l366_36646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_count_l366_36688

theorem inequality_solution_count : 
  (Finset.filter (fun n : ℕ => 
    (150 * n : ℕ)^40 > n^80 ∧ n^80 > 3^240 ∧ n > 0) 
    (Finset.range 150)).card = 122 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_count_l366_36688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_even_numbers_11_to_21_l366_36635

def is_even (n : ℕ) : Bool := n % 2 == 0

def average_of_even_numbers_between (a b : ℕ) : ℚ :=
  let even_numbers := (List.range (b - a + 1)).map (λ x => x + a) |>.filter is_even
  let sum := even_numbers.sum
  let count := even_numbers.length
  (sum : ℚ) / count

theorem average_of_even_numbers_11_to_21 :
  average_of_even_numbers_between 11 21 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_even_numbers_11_to_21_l366_36635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rho_squared_l366_36606

/-- Given a point (ρ,θ) in polar coordinates satisfying the equation
    3ρcos²θ + 2ρsin²θ = 6cosθ, the maximum value of ρ² is 4. -/
theorem max_rho_squared (ρ θ : ℝ) (h : 3 * ρ * Real.cos θ^2 + 2 * ρ * Real.sin θ^2 = 6 * Real.cos θ) :
  ∃ (M : ℝ), M = 4 ∧ ρ^2 ≤ M ∧ ∃ (ρ₀ θ₀ : ℝ), 3 * ρ₀ * Real.cos θ₀^2 + 2 * ρ₀ * Real.sin θ₀^2 = 6 * Real.cos θ₀ ∧ ρ₀^2 = M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rho_squared_l366_36606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_APQ_l366_36608

-- Define the ellipse C
noncomputable def C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the upper vertex A
def A : ℝ × ℝ := (0, 1)

-- Define a line l that intersects the ellipse at P and Q
noncomputable def l (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the condition that l does not pass through A
def l_not_through_A (k b : ℝ) : Prop := l k b A.1 ≠ A.2

-- Define the perpendicularity condition AP ⊥ AQ
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (Q.1 - A.1) + (P.2 - A.2) * (Q.2 - A.2) = 0

-- Define the area of triangle APQ
noncomputable def area_APQ (P Q : ℝ × ℝ) : ℝ :=
  abs ((P.1 - A.1) * (Q.2 - A.2) - (Q.1 - A.1) * (P.2 - A.2)) / 2

-- Theorem statement
theorem max_area_APQ :
  ∀ k b : ℝ,
  ∀ P Q : ℝ × ℝ,
  C P.1 P.2 →
  C Q.1 Q.2 →
  P.2 = l k b P.1 →
  Q.2 = l k b Q.1 →
  l_not_through_A k b →
  perpendicular P Q →
  area_APQ P Q ≤ 9/4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_APQ_l366_36608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_angle_sum_l366_36647

theorem sine_angle_sum (α : ℝ) : 
  Real.cos (α + π/6) - Real.sin α = 4 * Real.sqrt 3 / 5 → 
  Real.sin (α + 11*π/6) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_angle_sum_l366_36647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_difference_sum_l366_36645

theorem mean_difference_sum (m n : ℕ) : 
  m > 0 ∧ n > 0 ∧
  Nat.Coprime m n ∧
  (|((1/2 + 3/4 + 5/6) / 3) - ((7/8 + 9/10) / 2)| : ℚ) = m / n →
  m + n = 859 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_difference_sum_l366_36645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_difference_of_R_l366_36617

/-- Triangle ABC with vertices A(0,6), B(3,0), and C(9,0) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨0, 6⟩, ⟨3, 0⟩, ⟨9, 0⟩}

/-- Point R on line AC -/
def R : ℝ × ℝ := ⟨3, 4⟩

/-- Point S on line BC -/
def S : ℝ × ℝ := ⟨3, 0⟩

/-- Vertical line through R and S -/
def vertical_line (x : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = x}

/-- Area of a triangle given three points -/
noncomputable def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p₂.1 - p₁.1) * (p₃.2 - p₁.2) - (p₃.1 - p₁.1) * (p₂.2 - p₁.2))

/-- Theorem: The positive difference between x and y coordinates of R is 1 -/
theorem coordinate_difference_of_R :
  R ∈ vertical_line 3 ∧
  S ∈ vertical_line 3 ∧
  triangle_area R S ⟨9, 0⟩ = 15 →
  abs (R.1 - R.2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_difference_of_R_l366_36617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_difference_over_log_l366_36662

theorem integral_difference_over_log (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  ∫ (x : ℝ) in Set.Icc 0 1, (x^m - x^n) / Real.log x = Real.log ((m + 1) / (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_difference_over_log_l366_36662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_permutation_and_vector_l366_36652

def is_valid_n (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * k + 2 ∨ n = 4 * k + 3

def hamming_distance {n : ℕ} (x y : Fin n → Fin 2) : ℕ :=
  (Finset.filter (λ i => x i ≠ y i) Finset.univ).card

structure PreservesHammingProperty (n : ℕ) (f : (Fin n → Fin 2) → (Fin n → Fin 2)) : Prop where
  inj : Function.Injective f
  preserves : ∀ x y, hamming_distance x y > n / 2 → hamming_distance (f x) (f y) > n / 2

def permute {n : ℕ} (σ : Equiv.Perm (Fin n)) (x : Fin n → Fin 2) : Fin n → Fin 2 :=
  λ i => x (σ i)

theorem exists_permutation_and_vector
  {n : ℕ} (hn : is_valid_n n) (f : (Fin n → Fin 2) → (Fin n → Fin 2))
  (hf : PreservesHammingProperty n f) :
  ∃ (σ : Equiv.Perm (Fin n)) (v : Fin n → Fin 2),
    ∀ x : Fin n → Fin 2, f x = λ i => (permute σ x i + v i) % 2 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_permutation_and_vector_l366_36652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_polar_axis_l366_36622

/-- A curve in polar coordinates defined by ρ = 2cos(θ) -/
noncomputable def polar_curve (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- Symmetry about the polar axis means that for any angle θ, 
    the radius at θ is equal to the radius at -θ -/
theorem symmetry_about_polar_axis :
  ∀ θ : ℝ, polar_curve θ = polar_curve (-θ) := by
  intro θ
  unfold polar_curve
  simp [Real.cos_neg]

#check symmetry_about_polar_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_polar_axis_l366_36622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_rope_length_satisfies_conditions_l366_36651

/-- The initial length of a rope that, when extended to 23 meters, 
    allows a calf to graze an additional 1408 square meters. -/
noncomputable def initial_rope_length : ℝ :=
  Real.sqrt ((23^2 : ℝ) - 1408 / Real.pi)

/-- Theorem stating that the initial rope length satisfies the problem conditions -/
theorem initial_rope_length_satisfies_conditions :
  Real.pi * (23^2 - initial_rope_length^2) = 1408 := by
  sorry

-- We can't use #eval for noncomputable definitions, so we'll use #check instead
#check initial_rope_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_rope_length_satisfies_conditions_l366_36651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_correct_l366_36666

/-- An isosceles trapezoid with specific measurements and circles --/
structure IsoscelesTrapezoidWithCircles where
  /-- Length of side EF --/
  ef : ℝ
  /-- Length of side FG --/
  fg : ℝ
  /-- Length of side HE --/
  he : ℝ
  /-- Length of side GH --/
  gh : ℝ
  /-- Radius of circles centered at E and F --/
  r_ef : ℝ
  /-- Radius of circles centered at G and H --/
  r_gh : ℝ
  /-- The trapezoid is isosceles --/
  isosceles : fg = he
  /-- Specific measurements --/
  ef_eq : ef = 8
  fg_eq : fg = 7
  gh_eq : gh = 6
  r_ef_eq : r_ef = 4
  r_gh_eq : r_gh = 3

/-- The radius of the inner circle tangent to all four circles --/
noncomputable def inner_circle_radius (t : IsoscelesTrapezoidWithCircles) : ℝ :=
  (-42 + 18 * Real.sqrt 3) / 23

/-- Theorem stating that the inner circle radius is correct --/
theorem inner_circle_radius_correct (t : IsoscelesTrapezoidWithCircles) :
  ∃ (r : ℝ), r = inner_circle_radius t ∧ 
  (∃ (x z : ℝ), 
    x^2 + (t.r_ef + r)^2 = (t.r_ef)^2 ∧
    z^2 + (t.r_gh + r)^2 = (t.r_gh)^2 ∧
    x + z = Real.sqrt (t.fg^2 - ((t.ef - t.gh)/2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_correct_l366_36666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_rotation_volume_formula_l366_36607

/-- The volume of a solid of revolution formed by rotating a rhombus -/
noncomputable def rhombusRotationVolume (a : ℝ) (α : ℝ) : ℝ :=
  2 * Real.pi * a^3 * Real.sin (α/2) * Real.sin α

/-- Theorem: The volume of the solid of revolution formed by rotating a rhombus 
    with side length a and acute angle α around a line passing through its vertex 
    and parallel to the larger diagonal is 2πa³sin(α/2)sin(α) -/
theorem rhombus_rotation_volume_formula (a : ℝ) (α : ℝ) 
    (h1 : a > 0) (h2 : 0 < α ∧ α < Real.pi) : 
  rhombusRotationVolume a α = 2 * Real.pi * a^3 * Real.sin (α/2) * Real.sin α :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_rotation_volume_formula_l366_36607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_coefficients_l366_36618

open BigOperators Finset

/-- Given that (1+x+x^2)^6 = a₀ + a₁x + a₂x² + ... + a₁₂x¹², prove that a₂ + a₄ + ... + a₁₂ = 364 -/
theorem sum_even_coefficients (a : ℕ → ℕ) :
  (∀ x : ℝ, (1 + x + x^2)^6 = ∑ i in range 13, a i * x^i) →
  ∑ i in range 6, a (2 * i + 2) = 364 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_coefficients_l366_36618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_3_seconds_l366_36637

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the velocity function as the derivative of s
noncomputable def v (t : ℝ) : ℝ := deriv s t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_3_seconds_l366_36637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circle_existence_l366_36697

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle structure -/
structure Circle where
  center : Point
  radius : ℝ

/-- Two points are diametrically opposite on a circle -/
def DiametricallyOpposite (c : Circle) (p q : Point) : Prop :=
  ((p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2) ∧
  ((q.x - c.center.x)^2 + (q.y - c.center.y)^2 = c.radius^2) ∧
  ((p.x - q.x)^2 + (p.y - q.y)^2 = (2 * c.radius)^2)

/-- A circle intersects a point -/
def Circle.Intersects (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Given three circles in a plane, there exists a unique circle that intersects
    each of the given circles at two diametrically opposite points. -/
theorem intersecting_circle_existence (A B C : Circle) : 
  ∃! K : Circle, 
    (∃ P Q : Point, P ≠ Q ∧ DiametricallyOpposite A P Q ∧ K.Intersects P ∧ K.Intersects Q) ∧
    (∃ R S : Point, R ≠ S ∧ DiametricallyOpposite B R S ∧ K.Intersects R ∧ K.Intersects S) ∧
    (∃ T U : Point, T ≠ U ∧ DiametricallyOpposite C T U ∧ K.Intersects T ∧ K.Intersects U) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circle_existence_l366_36697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_for_scenario_l366_36649

/-- Represents the scenario of a leaking boat trying to reach the shore --/
structure BoatScenario where
  initial_distance : ℚ  -- Initial distance from shore in miles
  leakage_rate : ℚ      -- Water leakage rate in gallons per minute
  max_capacity : ℚ      -- Boat's maximum water capacity in gallons
  rowing_speed : ℚ      -- Rowing speed in miles per hour

/-- Calculates the minimum bailing rate required to reach the shore without sinking --/
def min_bailing_rate (scenario : BoatScenario) : ℚ :=
  let time_to_shore := scenario.initial_distance / scenario.rowing_speed * 60
  let total_leakage := scenario.leakage_rate * time_to_shore
  (total_leakage - scenario.max_capacity) / time_to_shore

/-- The main theorem stating the minimum bailing rate for the given scenario --/
theorem min_bailing_rate_for_scenario :
  let scenario := BoatScenario.mk 2 15 60 4
  min_bailing_rate scenario = 13 := by
  sorry

#eval min_bailing_rate (BoatScenario.mk 2 15 60 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_for_scenario_l366_36649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_integer_quotient_l366_36624

/-- A function that checks if a number is a valid 14-digit integer with exactly two occurrences of each digit from 1 to 7 -/
def isValidNumber (n : ℕ) : Prop :=
  (n ≥ 10^13 ∧ n < 10^14) ∧
  ∀ d : ℕ, d ∈ Finset.range 7 → (Finset.filter (λ x : ℕ ↦ x = d + 1) (Finset.image (λ i ↦ (n / 10^i) % 10) (Finset.range 14))).card = 2

theorem not_integer_quotient (k m : ℕ) (hk : isValidNumber k) (hm : isValidNumber m) (hkm : k ≠ m) :
  ¬ (∃ (n : ℕ), k = n * m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_integer_quotient_l366_36624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_polar_curve_l366_36603

/-- The polar curve r = 2 cos 6φ -/
noncomputable def r (φ : ℝ) : ℝ := 2 * Real.cos (6 * φ)

/-- The area of the region bounded by the curve r = 2 cos 6φ -/
noncomputable def area : ℝ := 2 * Real.pi

/-- Theorem: The area of the region bounded by the curve r = 2 cos 6φ is 2π -/
theorem area_of_polar_curve : 
  (∫ φ in (0)..(2 * Real.pi), (1/2) * (r φ)^2) = area := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_polar_curve_l366_36603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passage_time_l366_36682

/-- The time (in seconds) for a train to pass a bridge -/
noncomputable def train_pass_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train of length 310 meters, traveling at 45 km/hour, 
    will take 36 seconds to pass a bridge of length 140 meters -/
theorem train_bridge_passage_time :
  train_pass_time 310 45 140 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passage_time_l366_36682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_and_equals_f_f_has_max_iff_a_in_range_l366_36690

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * x - abs (x + 1)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then (a - 1) * x - 1
  else if x = 0 then 0
  else (a - 1) * x + 1

-- State the theorems
theorem g_is_odd_and_equals_f (a : ℝ) :
  (∀ x : ℝ, g a (-x) = -(g a x)) ∧
  (∀ x : ℝ, x > 0 → g a x = f a x) := by
  sorry

theorem f_has_max_iff_a_in_range (a : ℝ) :
  (∃ M : ℝ, ∀ x : ℝ, f a x ≤ M) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_and_equals_f_f_has_max_iff_a_in_range_l366_36690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_trinomial_m_l366_36633

theorem quadratic_trinomial_m (m : ℤ) : 
  (∃ (a b c : ℝ) (h : a ≠ 0), ∀ x, (m - 2) * x^(Int.natAbs m) + 2*x - 2 = a*x^2 + b*x + c) → 
  m = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_trinomial_m_l366_36633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l366_36620

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define f' as the derivative of f
noncomputable def f' : ℝ → ℝ := fun x => deriv f x

-- State the theorem
theorem function_range_theorem 
  (h_even : ∀ x, f x = f (-x))
  (h_continuous : Continuous f)
  (h_decreasing : ∀ x > 0, HasDerivAt f (f' f x) x ∧ f' f x < 0)
  (h_inequality : f (Real.log x / Real.log 10) > f 1) :
  1/10 < x ∧ x < 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l366_36620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_value_l366_36630

/-- The value of the infinite nested radical sqrt(15 + sqrt(15 + sqrt(15 + ...))) -/
noncomputable def nested_radical : ℝ := 
  (1 + Real.sqrt 61) / 2

/-- Theorem stating that the nested radical equals (1 + sqrt(61)) / 2 -/
theorem nested_radical_value : nested_radical = Real.sqrt (15 + nested_radical) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_value_l366_36630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_m_range_is_0_to_12_l366_36671

/-- The range of real numbers m for which 3mx^2 + mx + 1 > 0 holds for all real x -/
def valid_m_range : Set ℝ :=
  {m : ℝ | ∀ x : ℝ, 3 * m * x^2 + m * x + 1 > 0}

/-- The theorem stating that the valid range of m is [0, 12) -/
theorem valid_m_range_is_0_to_12 : valid_m_range = Set.Ici 0 ∩ Set.Iio 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_m_range_is_0_to_12_l366_36671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_inverse_l366_36644

-- Define positive real numbers a and b with a > b
variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hab : a > b)

-- Define the nth term of the series
noncomputable def nthTerm (n : ℕ) : ℝ :=
  1 / (((n - 1) * a^3 - (n - 2) * b^3) * (n * a^3 - (n - 1) * b^3))

-- Define the infinite sum
noncomputable def infiniteSum : ℝ := ∑' n, nthTerm a b n

-- Theorem statement
theorem sum_equals_inverse : infiniteSum a b = 1 / ((a^3 - b^3) * b^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_inverse_l366_36644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_of_i_l366_36663

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the property of i
@[simp] lemma i_squared : i^2 = -1 := Complex.I_sq

-- State the theorem
theorem sum_of_powers_of_i (n : ℤ) :
  i^(2*n - 3) + i^(2*n - 1) + i^(2*n + 1) + i^(2*n + 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_of_i_l366_36663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_about_half_l366_36656

noncomputable def g (x : ℝ) : ℝ := |⌊(2:ℝ)*x⌋| - |⌊(2:ℝ) - (2:ℝ)*x⌋|

theorem g_symmetry_about_half : ∀ x : ℝ, g x = -g (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_about_half_l366_36656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l366_36673

noncomputable section

def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x, x > 0 → f x ≠ 0

axiom f_property : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y

axiom f_positive : ∀ x, x > 1 → f x > 0

axiom f_6 : f 6 = 1

theorem f_properties :
  (f 1 = 0) ∧
  (∀ x y, 0 < x → x < y → f x < f y) ∧
  ({x : ℝ | f (x + 3) - f (1/3) < 2} = Set.Ioo (-3) 9) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l366_36673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_coin_probability_l366_36674

theorem modified_coin_probability (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : (Nat.choose 6 3 : ℝ) * p^3 * (1-p)^3 = 1/20) : 
  p = (1 - Real.sqrt 0.6816) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_coin_probability_l366_36674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l366_36695

theorem right_triangle_area (a c : ℝ) (h1 : a = 24) (h2 : c = 26) : 
  (1/2) * a * Real.sqrt (c^2 - a^2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l366_36695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_l366_36675

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem unused_sector_angle (paper_radius cone_radius : ℝ) (cone_vol : ℝ) : 
  paper_radius = 16 →
  cone_radius = 15 →
  cone_vol = 675 * Real.pi →
  ∃ (h : ℝ), 
    cone_vol = cone_volume cone_radius h ∧
    paper_radius^2 = cone_radius^2 + h^2 ∧
    360 - (cone_radius / paper_radius * 360) = 22.5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_sector_angle_l366_36675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_l366_36678

noncomputable def series (n : ℕ) : ℚ :=
  if n % 2 = 0 then
    (n + 1) / (2 ^ (2 * n + 2))
  else
    (n + 1) / (3 ^ (2 * n + 1))

noncomputable def series_sum : ℚ := ∑' n, series n

theorem fraction_sum (a b : ℕ) (h1 : Nat.Coprime a b) (h2 : (a : ℚ) / b = series_sum) :
  a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_l366_36678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_equation_l366_36643

/-- Theorem: For a hyperbola with the given properties, its asymptote equation is x ± √3y = 0 -/
theorem hyperbola_asymptote_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b / (2 * c) = 1 / 4) : 
  ∃ (k : ℝ), k * x + Real.sqrt 3 * y = 0 ∨ k * x - Real.sqrt 3 * y = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_equation_l366_36643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solution_l366_36665

theorem cosine_equation_solution (k n : ℤ) :
  let f (x : ℝ) := (Real.cos x)^2 + (Real.cos (2*x))^2 - (Real.cos (3*x))^2 - (Real.cos (4*x))^2
  f (π/2 + k*π) = 0 ∧ f (n*π/5) = 0 ∧ f (-n*π/2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solution_l366_36665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cos_A_l366_36677

theorem right_triangle_cos_A (a b : ℝ) (h_right : a = 4 ∧ b = 3) :
  let c := Real.sqrt (a^2 + b^2)
  Real.cos (Real.arcsin (a / c)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cos_A_l366_36677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_monomial_form_l366_36689

-- Define the sequence of monomials
def a : ℕ → (ℚ → ℚ)
  | 0 => λ x => 1  -- Adding a case for 0
  | 1 => λ x => 2 * x
  | 2 => λ x => 4 * x^3
  | 3 => λ x => 8 * x^5
  | n+4 => λ x => 2 * (a n x)

-- State the theorem
theorem nth_monomial_form (n : ℕ) (x : ℚ) :
  a n x = 2^n * x^(2*n - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_monomial_form_l366_36689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l366_36687

theorem problem_solution : 
  ((Real.sqrt 16)^2 - Real.sqrt 25 + Real.sqrt ((-2)^2) = 13) ∧ 
  (Real.sqrt (1/2) * Real.sqrt 48 / Real.sqrt (1/8) = 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l366_36687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_and_min_l366_36668

/-- A function f(x) with parameters a, b, and c -/
noncomputable def f (a b c x : ℝ) : ℝ := a * Real.log x + b / x + c / (x^2)

/-- Theorem stating the conditions for f to have both a maximum and a minimum -/
theorem f_has_max_and_min (a b c : ℝ) (ha : a ≠ 0) 
  (h_max_min : ∃ (x_max x_min : ℝ), x_max ≠ x_min ∧ 
    (∀ x > 0, f a b c x ≤ f a b c x_max) ∧
    (∀ x > 0, f a b c x ≥ f a b c x_min)) :
  (a * b > 0) ∧ (a * c < 0) ∧ (b^2 + 8*a*c > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_and_min_l366_36668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_equation_l366_36664

/-- Given two altitudes of a triangle and one vertex, prove the equation of the opposite side. -/
theorem triangle_side_equation (A B C : ℝ × ℝ) : 
  (∀ x y : ℝ, x + y = 0 → (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1)) →
  (∀ x y : ℝ, 2*x - 3*y + 1 = 0 → (x - A.1) * (C.2 - A.2) = (y - A.2) * (C.1 - A.1)) →
  A = (1, 2) →
  ∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ 
    (∀ x y : ℝ, (x, y) = B ∨ (x, y) = C ↔ a*x + b*y + c = 0) ∧
    (a, b, c) = (2, 3, 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_equation_l366_36664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_implies_m_and_x_l366_36657

/-- A function f(x) with parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (4 : ℝ)^x + m * (2 : ℝ)^x + 1

/-- Theorem stating that if f has exactly one zero, then m = -2 and the zero is at x = 0 -/
theorem unique_zero_implies_m_and_x (m : ℝ) :
  (∃! x, f m x = 0) → (m = -2 ∧ f m 0 = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_implies_m_and_x_l366_36657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nellie_gift_wrap_sales_l366_36627

theorem nellie_gift_wrap_sales
  (grandmother_sales : ℕ)
  (uncle_sales : ℕ)
  (neighbor_sales : ℕ)
  (remaining_sales : ℕ)
  (h1 : grandmother_sales = 1)
  (h2 : uncle_sales = 10)
  (h3 : neighbor_sales = 6)
  (h4 : remaining_sales = 28) :
  grandmother_sales + uncle_sales + neighbor_sales + remaining_sales = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nellie_gift_wrap_sales_l366_36627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_eligible_is_175_l366_36625

def is_eligible (n : ℕ) : Bool := n ≤ 799

def random_table : List ℕ := [
  7447, 6721, 7633, 5025, 8392, 1206, 0076,
  6301, 6378, 5916, 9556, 6719, 9810, 5071, 7512, 8673, 5807, 4439, 5238, 0079,
  3321, 1234, 2978, 6456, 0782, 5242, 0744, 3815, 5100, 1342, 9966, 0279, 0054
]

def find_nth_eligible (n : ℕ) (table : List ℕ) : Option ℕ :=
  let eligible := table.filter is_eligible
  eligible.get? (n - 1)

theorem fifth_eligible_is_175 :
  find_nth_eligible 5 random_table = some 175 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_eligible_is_175_l366_36625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_union_l366_36638

-- Define the quadratic equations
def equation_A (p : ℝ) (x : ℝ) : Prop := 3 * x^2 + p * x - 7 = 0
def equation_B (q : ℝ) (x : ℝ) : Prop := 3 * x^2 - 7 * x + q = 0

-- Define the solution sets
def A (p : ℝ) : Set ℝ := {x | equation_A p x}
def B (q : ℝ) : Set ℝ := {x | equation_B q x}

-- State the theorem
theorem solution_sets_union (p q : ℝ) :
  (A p ∩ B q = {-1/3}) →
  (A p ∪ B q = {7, 8/3, -1/3}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_union_l366_36638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_distances_l366_36655

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- Define point B
def B : ℝ × ℝ := (0, -1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem no_equal_distances :
  ¬ ∃ (m : ℝ) (M N : ℝ × ℝ),
    M ≠ N ∧
    ellipse M.1 M.2 ∧
    ellipse N.1 N.2 ∧
    line M.1 M.2 m ∧
    line N.1 N.2 m ∧
    distance B M = distance B N :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_distances_l366_36655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_fixed_point_l366_36632

noncomputable section

-- Define the hyperbola D
def hyperbola_D (x y : ℝ) : Prop := y^2 / 2 - x^2 = 1 / 3

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the point P
def point_P (t : ℝ) : ℝ × ℝ := (t, 1)

-- Define points A and B on parabola C
def point_A (x : ℝ) : ℝ × ℝ := (x, x^2 / 4)
def point_B (x : ℝ) : ℝ × ℝ := (x, x^2 / 4)

-- Define the perpendicularity condition
def perpendicular (P A B : ℝ × ℝ) : Prop :=
  (A.2 - P.2) * (B.2 - P.2) = -(A.1 - P.1) * (B.1 - P.1)

-- Define the line AB
def line_AB (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - A.2) * (B.1 - A.1) = (x - A.1) * (B.2 - A.2)

theorem parabola_and_fixed_point :
  ∀ (t : ℝ), t > 0 →
  ∀ (x₁ x₂ : ℝ),
  parabola_C (point_P t).1 (point_P t).2 →
  parabola_C (point_A x₁).1 (point_A x₁).2 →
  parabola_C (point_B x₂).1 (point_B x₂).2 →
  perpendicular (point_P t) (point_A x₁) (point_B x₂) →
  line_AB (point_A x₁) (point_B x₂) (-2) 5 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_fixed_point_l366_36632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l366_36610

theorem equation_solution : ∃! x : ℝ, (2 : ℝ) ^ (4 * x + 2) * (8 : ℝ) ^ (2 * x + 7) = (16 : ℝ) ^ (3 * x + 10) ∧ x = -8.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l366_36610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_length_calculation_l366_36621

/-- The length of a ladder given its angle of elevation and distance from the wall -/
noncomputable def ladder_length (angle : ℝ) (distance : ℝ) : ℝ :=
  distance / Real.cos angle

theorem ladder_length_calculation (angle : ℝ) (distance : ℝ) 
  (h1 : angle = Real.pi / 3)  -- 60 degrees in radians
  (h2 : distance = 4.6) :
  ladder_length angle distance = 9.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_length_calculation_l366_36621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_value_l366_36685

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_base_value (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x ∈ Set.Icc 2 8, f a x ≤ f a 2) ∧ 
  (∀ x ∈ Set.Icc 2 8, f a x ≥ f a 8) ∧
  (f a 2 - f a 8 = 2) →
  a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_value_l366_36685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_rate_of_change_f_1_2_l366_36613

/-- The function f(x) = x^2 + x -/
noncomputable def f (x : ℝ) : ℝ := x^2 + x

/-- The average rate of change of f over the interval [a, b] -/
noncomputable def avgRateOfChange (f : ℝ → ℝ) (a b : ℝ) : ℝ := (f b - f a) / (b - a)

theorem avg_rate_of_change_f_1_2 :
  avgRateOfChange f 1 2 = 4 := by
  -- Unfold the definitions
  unfold avgRateOfChange f
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_rate_of_change_f_1_2_l366_36613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_difference_l366_36641

def max_element_in_range (a b : ℤ) : ℤ :=
  if a > b then a else b

def min_element_in_range (a b : ℤ) : ℤ :=
  if a < b then a else b

theorem triangle_side_difference (x : ℤ) 
  (h1 : x > 2 ∧ x < 16)
  (h2 : ∀ y : ℤ, (y > 2 ∧ y < 16) → (y ≤ x ∨ x ≤ y))
  (h3 : ∃ z : ℤ, z > 2 ∧ z < 16 ∧ z ≠ x)
  (maxValue : ℤ) (minValue : ℤ)
  (hmax : maxValue = max x (max_element_in_range 3 15))
  (hmin : minValue = min x (min_element_in_range 3 15)) :
  maxValue - minValue = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_difference_l366_36641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l366_36648

noncomputable section

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-Real.sqrt 2, 0)

-- Define a point on the x-axis
def point_on_x_axis (t : ℝ) : ℝ × ℝ := (t, 0)

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem hyperbola_properties :
  ∃ (min_area : ℝ) (p : ℝ × ℝ),
    -- Part 1: Minimum area of triangle OMN
    (min_area = Real.sqrt 2) ∧
    -- Part 2: Existence of point P
    (p = point_on_x_axis (-Real.sqrt 2 / 2)) ∧
    -- Constant dot product property
    (∀ (m n : ℝ × ℝ),
      hyperbola m.1 m.2 → hyperbola n.1 n.2 →
      ∃ (c : ℝ), dot_product (m.1 - p.1, m.2 - p.2) (n.1 - p.1, n.2 - p.2) = c) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l366_36648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_sufficient_condition_not_necessary_l366_36650

/-- The slope of the first line (a+2)x+3ay+1=0 -/
noncomputable def slope1 (a : ℝ) : ℝ := -(a + 2) / (3 * a)

/-- The slope of the second line (a-2)x+(a+2)y-3=0 -/
noncomputable def slope2 (a : ℝ) : ℝ := -(a - 2) / (a + 2)

/-- Two lines are perpendicular if and only if the product of their slopes equals -1 -/
def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The condition a = -2 is sufficient for the lines to be perpendicular -/
theorem condition_sufficient :
  ∀ a : ℝ, a = -2 → are_perpendicular (slope1 a) (slope2 a) :=
by sorry

/-- The condition a = -2 is not necessary for the lines to be perpendicular -/
theorem condition_not_necessary :
  ∃ a : ℝ, a ≠ -2 ∧ are_perpendicular (slope1 a) (slope2 a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_sufficient_condition_not_necessary_l366_36650
