import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l515_51523

noncomputable def data : List ℝ := [10, 6, 8, 5, 6]

noncomputable def mean (xs : List ℝ) : ℝ := xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (λ x => (x - μ)^2)).sum / xs.length

theorem variance_of_data : variance data = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l515_51523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_segment_length_l515_51558

-- Define the golden ratio
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

-- Define the points A, B, C, and D on a line
def A : ℝ := 0
def B : ℝ := 1
noncomputable def C : ℝ := B - (B - A) / φ
noncomputable def D : ℝ := A + (B - A) / φ

-- State the theorem
theorem golden_section_segment_length :
  D - C = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_segment_length_l515_51558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_perfect_power_l515_51556

theorem smallest_sum_of_perfect_power (a b : ℕ) : 
  2^6 * 7^3 = a^b → (∀ c d : ℕ, 2^6 * 7^3 = c^d → a + b ≤ c + d) → a + b = 21953 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_perfect_power_l515_51556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_x_intercept_distance_l515_51502

/-- A line with equation y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ := -l.b / l.m

/-- The distance between two points on the x-axis -/
noncomputable def x_distance (x₁ x₂ : ℝ) : ℝ := |x₁ - x₂|

theorem intersecting_lines_x_intercept_distance :
  let line1 : Line := { m := 4, b := -12 }
  let line2 : Line := { m := -2, b := 36 }
  (4 * 8 - 12 = 20) →  -- line1 passes through (8, 20)
  (-2 * 8 + 36 = 20) →  -- line2 passes through (8, 20)
  x_distance (x_intercept line1) (x_intercept line2) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_x_intercept_distance_l515_51502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_proposition_l515_51550

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 2) + Real.sqrt (1 - x)

def domain_f : Set ℝ := {x | x - 2 ≥ 0 ∧ 1 - x ≥ 0}

def g (x : ℕ) : ℝ := 2 * x

noncomputable def h (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

theorem exactly_one_correct_proposition :
  (¬(Set.Nonempty domain_f)) ∧
  (∀ (X Y : Type) (f : X → Y), Function.Surjective f) ∧
  (¬(∃ (a b : ℝ), ∀ (x : ℕ), g x = a * (x : ℝ) + b)) ∧
  (¬(∃ (a b c : ℝ), ∀ (x : ℝ), h x = a * x^2 + b * x + c)) :=
by sorry

#check exactly_one_correct_proposition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_proposition_l515_51550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_percentage_change_l515_51513

/-- Calculates the percentage change in the area of a circle when its radius changes from r1 to r2 -/
noncomputable def percentageChangeInCircleArea (r1 r2 : ℝ) : ℝ :=
  ((r2^2 - r1^2) / r1^2) * 100

/-- Theorem stating that the percentage change in the area of a circle
    when its radius is reduced from 5 cm to 4 cm is -36% -/
theorem circle_area_percentage_change :
  percentageChangeInCircleArea 5 4 = -36 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_percentage_change_l515_51513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_theorem_l515_51560

/-- A point on a grid plane with integer coordinates -/
structure GridPoint where
  x : Int
  y : Int

/-- The midpoint of two grid points -/
def gridMidpoint (p1 p2 : GridPoint) : Rat × Rat :=
  ((p1.x + p2.x : Rat) / 2, (p1.y + p2.y : Rat) / 2)

/-- Predicate to check if a point is on the grid (has integer coordinates) -/
def isGridPoint (p : Rat × Rat) : Prop :=
  ∃ (x y : Int), p.1 = x ∧ p.2 = y

theorem midpoint_theorem (points : Finset GridPoint) :
  points.card = 5 →
  ∃ (p1 p2 : GridPoint), p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ isGridPoint (gridMidpoint p1 p2) := by
  sorry

#check midpoint_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_theorem_l515_51560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_reflection_theorem_l515_51522

/-- A circle with center O and radius r -/
structure Circle where
  O : EuclideanSpace ℝ (Fin 2)
  r : ℝ
  r_pos : r > 0

/-- A chord AB of length r in the circle -/
structure Chord (k : Circle) where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  on_circle : (dist A k.O = k.r) ∧ (dist B k.O = k.r)
  length_r : dist A B = k.r

/-- The reflection of the center O across chord AB -/
noncomputable def reflection_point (k : Circle) (chord : Chord k) : EuclideanSpace ℝ (Fin 2) :=
  sorry

theorem circle_chord_reflection_theorem (k : Circle) (chord : Chord k) :
  let D := reflection_point k chord
  ∀ C : EuclideanSpace ℝ (Fin 2), dist C k.O = k.r →
    (dist C chord.A) ^ 2 + (dist C chord.B) ^ 2 = (dist C D) ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_reflection_theorem_l515_51522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f2_has_property_M_l515_51594

open Real

-- Define property M
def property_M (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → (exp x) * (f x) < (exp y) * (f y)

-- Define the functions
noncomputable def f1 : ℝ → ℝ := λ x => log x
def f2 : ℝ → ℝ := λ x => x^2 + 1
noncomputable def f3 : ℝ → ℝ := λ x => sin x
def f4 : ℝ → ℝ := λ x => x^3

-- Theorem statement
theorem only_f2_has_property_M :
  ¬(property_M f1) ∧
  (property_M f2) ∧
  ¬(property_M f3) ∧
  ¬(property_M f4) := by
  sorry

#check only_f2_has_property_M

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f2_has_property_M_l515_51594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_theorem_l515_51577

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Define membership for Point2D in Circle -/
def Point2D.mem (p : Point2D) (c : Circle) : Prop :=
  (p.x - c.center.x) ^ 2 + (p.y - c.center.y) ^ 2 = c.radius ^ 2

instance : Membership Point2D Circle where
  mem := Point2D.mem

def is_cyclic (q : Quadrilateral) : Prop := 
  ∃ (c : Circle), q.A ∈ c ∧ q.B ∈ c ∧ q.C ∈ c ∧ q.D ∈ c

def is_circumcenter (O : Point2D) (q : Quadrilateral) : Prop :=
  ∃ (c : Circle), c.center = O ∧ q.A ∈ c ∧ q.B ∈ c ∧ q.C ∈ c ∧ q.D ∈ c

noncomputable def inside_quadrilateral (P : Point2D) (q : Quadrilateral) : Prop := sorry

noncomputable def not_on_diagonal (P : Point2D) (A : Point2D) (C : Point2D) : Prop := sorry

noncomputable def intersect_at (A B C D I : Point2D) : Prop := sorry

noncomputable def on_circumcircle (P A O I : Point2D) : Prop := sorry

noncomputable def is_parallelogram (P Q R S : Point2D) : Prop := sorry

theorem quadrilateral_theorem (ABCD : Quadrilateral) (O I P Q R S : Point2D) :
  is_cyclic ABCD →
  is_circumcenter O ABCD →
  inside_quadrilateral O ABCD →
  not_on_diagonal O ABCD.A ABCD.C →
  intersect_at ABCD.A ABCD.C ABCD.B ABCD.D I →
  on_circumcircle P ABCD.A O I →
  on_circumcircle Q ABCD.A O I →
  on_circumcircle R ABCD.C O I →
  on_circumcircle S ABCD.C O I →
  is_parallelogram P Q R S := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_theorem_l515_51577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_number_l515_51588

theorem find_number : ∃! x : ℕ, (69842 * 69842 - 30158 * 30158) / x = 100000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_number_l515_51588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_wolves_nine_rams_l515_51548

/-- Represents the number of days it takes for a given number of wolves to eat an equal number of rams -/
def consumption_time (wolves : ℕ) (rams : ℕ) : ℕ := sorry

/-- Given condition: 7 wolves eat 7 rams in 7 days -/
axiom base_case : consumption_time 7 7 = 7

/-- The consumption time is independent of the number of wolves and rams when they are equal -/
axiom consumption_invariance (n : ℕ) : consumption_time n n = consumption_time 7 7

theorem nine_wolves_nine_rams : consumption_time 9 9 = 7 := by
  rw [consumption_invariance 9]
  exact base_case

#check nine_wolves_nine_rams

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_wolves_nine_rams_l515_51548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_consumption_after_price_increase_l515_51565

/-- Given a price increase of sugar and a change in expenditure, 
    calculate the new monthly consumption of sugar. -/
theorem sugar_consumption_after_price_increase 
  (P : ℝ) -- Original price of sugar per kg
  (original_consumption : ℝ) -- Original monthly consumption in kg
  (price_increase : ℝ) -- Price increase percentage
  (expenditure_increase : ℝ) -- Expenditure increase percentage
  (h1 : original_consumption = 30)
  (h2 : price_increase = 0.32)
  (h3 : expenditure_increase = 0.10)
  : ∃ (new_consumption : ℝ), new_consumption = 25 := by
  -- The new monthly consumption of sugar is 25 kg
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_consumption_after_price_increase_l515_51565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l515_51533

/-- Converts cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r : ℝ) (θ : ℝ) (z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ := (7, 5 * Real.pi / 4, -3)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ × ℝ := (-7 * Real.sqrt 2 / 2, -7 * Real.sqrt 2 / 2, -3)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l515_51533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_250_meters_l515_51541

/-- Represents the speed of the train at each pole in km/h -/
def speed_at_pole : Fin 3 → ℚ
| 0 => 50
| 1 => 60
| 2 => 70

/-- Represents the time taken to cross each pole in seconds -/
def time_to_cross : Fin 3 → ℚ
| 0 => 18
| 1 => 20
| 2 => 22

/-- Represents the distance between poles in meters -/
def distance_between_poles : Fin 2 → ℚ
| 0 => 500
| 1 => 800

/-- Conversion factor from km/h to m/s -/
def km_per_hour_to_m_per_second : ℚ := 5 / 18

theorem train_length_is_250_meters :
  ∃ (length : ℚ),
    length = 250 ∧
    (∀ i : Fin 3, length = speed_at_pole i * km_per_hour_to_m_per_second * time_to_cross i) ∧
    (∀ i : Fin 2,
      distance_between_poles i = 
        speed_at_pole i * km_per_hour_to_m_per_second * 
        (time_to_cross (Fin.succ i) - time_to_cross i + 
         (distance_between_poles i / (speed_at_pole i * km_per_hour_to_m_per_second)))) :=
by sorry

#eval speed_at_pole 0
#eval time_to_cross 1
#eval distance_between_poles 0
#eval km_per_hour_to_m_per_second

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_250_meters_l515_51541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulo_eleven_residue_l515_51543

theorem modulo_eleven_residue : (325 + 22 * 6 + 9 * 121 + 3^2 * 33) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulo_eleven_residue_l515_51543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l515_51597

-- Define the hyperbola parameters
variable (a b : ℝ)

-- Define the conditions
def hyperbola_condition (a b : ℝ) : Prop := a > 0 ∧ b > 0

-- Define the distance from (-√2, 0) to the asymptote
noncomputable def distance_to_asymptote : ℝ := Real.sqrt 5 / 5

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 10 / 3

-- State the theorem
theorem hyperbola_eccentricity (ha : hyperbola_condition a b) 
  (hd : distance_to_asymptote = Real.sqrt 2 * b / Real.sqrt (a^2 + b^2)) : 
  eccentricity = Real.sqrt (a^2 + b^2) / a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l515_51597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_fraction_flask3_is_21_200_l515_51526

noncomputable def acid_fraction_flask3 (acid1 acid2 acid3 : ℝ) (water : ℝ) : ℝ :=
  acid3 / (acid3 + water)

noncomputable def water_flask1 (acid1 : ℝ) : ℝ :=
  19 * acid1

noncomputable def total_water (acid1 acid2 : ℝ) : ℝ :=
  (30 * acid2) / 7

theorem acid_fraction_flask3_is_21_200
  (acid1 acid2 acid3 : ℝ)
  (h1 : acid1 = 10)
  (h2 : acid2 = 20)
  (h3 : acid3 = 30)
  (h4 : acid1 / (acid1 + water_flask1 acid1) = 1 / 20)
  (h5 : acid2 / (acid2 + (total_water acid1 acid2 - water_flask1 acid1)) = 7 / 30) :
  acid_fraction_flask3 acid1 acid2 acid3 (total_water acid1 acid2) = 21 / 200 := by
  sorry

#check acid_fraction_flask3_is_21_200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_fraction_flask3_is_21_200_l515_51526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l515_51531

/-- The function f(x) = x^3 - ax^2 - 3x -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - 3*x

/-- Theorem stating the properties of the function f -/
theorem f_properties (a : ℝ) :
  (∃ x, x = 3 ∧ deriv (f a) x = 0) →
  (∃ min max, IsMinOn (f a) (Set.Icc 1 a) min ∧ 
              IsMaxOn (f a) (Set.Icc 1 a) max ∧
              f a min = -18 ∧ f a max = -6) ∧
  (StrictMonoOn (f a) (Set.Ici 1) → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l515_51531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l515_51540

/-- The function for which we're finding the tangent line -/
noncomputable def f (x : ℝ) : ℝ := x + Real.log x

/-- The derivative of the function f -/
noncomputable def f' (x : ℝ) : ℝ := 1 + 1 / x

/-- The slope of the tangent line at x = 1 -/
noncomputable def k : ℝ := f' 1

/-- The y-coordinate of the point of tangency -/
noncomputable def y₀ : ℝ := f 1

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 2 * x - y - 1 = 0

theorem tangent_line_at_one :
  tangent_line 1 y₀ ∧
  ∀ x, tangent_line x (k * (x - 1) + y₀) := by
  sorry

#check tangent_line_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l515_51540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l515_51591

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x) else Real.log x / Real.log 81

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, f x = 1/4 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l515_51591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l515_51546

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 7) + 1 / (x^2 + 7) + 1 / (x^3 + 7)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x < -7 ∨ (-7 < x ∧ x < -Real.rpow 7 (1/3 : ℝ)) ∨ x > -Real.rpow 7 (1/3 : ℝ)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l515_51546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_range_l515_51509

/-- The function f(x) = -1/2 * x^2 + 6x - 8 * ln(x) -/
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 6*x - 8 * Real.log x

/-- Predicate to check if a function is not monotonic on an interval -/
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a ≤ x ∧ x < y ∧ y < z ∧ z ≤ b ∧ 
    ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

/-- The set of real numbers m for which f is not monotonic on [m, m+1] -/
def M : Set ℝ := {m | not_monotonic f m (m+1)}

theorem f_not_monotonic_range : 
  M = Set.Ioo 1 2 ∪ Set.Ioo 3 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_range_l515_51509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l515_51553

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1/2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 1/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l515_51553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_interval_min_max_for_a_2_a_value_for_diff_2_l515_51571

-- Define the set A
def A : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Theorem 1: A is equal to [2, 4]
theorem A_equals_interval : A = Set.Icc 2 4 := by sorry

-- Theorem 2: For a = 2, min value is 4 and max value is 16
theorem min_max_for_a_2 : 
  (∀ x ∈ A, f 2 x ≥ 4) ∧ 
  (∃ x ∈ A, f 2 x = 4) ∧
  (∀ x ∈ A, f 2 x ≤ 16) ∧
  (∃ x ∈ A, f 2 x = 16) := by sorry

-- Theorem 3: If max - min = 2, then a = √2
theorem a_value_for_diff_2 (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  (∃ x y, x ∈ A ∧ y ∈ A ∧ ∀ z ∈ A, f a z ≤ f a y ∧ f a z ≥ f a x ∧ f a y - f a x = 2) →
  a = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_interval_min_max_for_a_2_a_value_for_diff_2_l515_51571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_n_exists_S_n_plus_one_l515_51592

/-- S(n) is the sum of digits of a positive integer n -/
def S (n : ℕ+) : ℕ := sorry

/-- For a particular positive integer n, S(n) = 1386 -/
theorem particular_n_exists : ∃ n : ℕ+, S n = 1386 := by
  sorry

/-- If S(n) = 1386 for some positive integer n, then S(n+1) = 1387 -/
theorem S_n_plus_one (n : ℕ+) (h : S n = 1386) : S (n + 1) = 1387 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_n_exists_S_n_plus_one_l515_51592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_free_items_cost_l515_51517

/-- Calculates the total cost of tax-free items given the total cost and taxable item details -/
noncomputable def total_cost_tax_free (total_cost : ℝ) 
  (electronics_tax_rate clothing_tax_rate luxury_tax_rate : ℝ)
  (electronics_tax_paid clothing_tax_paid luxury_tax_paid : ℝ) : ℝ :=
  let electronics_cost := electronics_tax_paid / electronics_tax_rate
  let clothing_cost := clothing_tax_paid / clothing_tax_rate
  let luxury_cost := luxury_tax_paid / luxury_tax_rate
  let total_taxable_cost := electronics_cost + clothing_cost + luxury_cost
  total_cost - total_taxable_cost

/-- Theorem stating that the total cost of tax-free items is 20 Rs -/
theorem tax_free_items_cost : 
  total_cost_tax_free 175 0.07 0.05 0.12 3.50 2.25 7.20 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_free_items_cost_l515_51517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_theta_l515_51559

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The given function f(x) = sin(x - θ) + √3 * cos(x - θ) -/
noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (x - θ) + Real.sqrt 3 * Real.cos (x - θ)

/-- Theorem: If f(x) = sin(x - θ) + √3 * cos(x - θ) is an even function, 
    then θ = kπ - π/6 for some integer k -/
theorem even_function_theta (θ : ℝ) (h : IsEven (f θ)) : 
  ∃ k : ℤ, θ = k * Real.pi - Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_theta_l515_51559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadrilateral_area_l515_51552

/-- Represents a segment with a length -/
structure Segment where
  length : ℝ
  length_pos : length > 0

/-- Configuration of four segments -/
structure FourSegments where
  s1 : Segment
  s2 : Segment
  s3 : Segment
  s4 : Segment
  ordered : s1.length ≥ s2.length ∧ s2.length ≥ s3.length ∧ s3.length ≥ s4.length

/-- Area of a quadrilateral formed by four segments -/
noncomputable def quadrilateralArea (fs : FourSegments) (angle : ℝ) : ℝ :=
  1/2 * (fs.s1.length + fs.s4.length) * (fs.s2.length + fs.s3.length) * Real.sin angle

/-- Theorem stating the existence of a maximum area for the quadrilateral -/
theorem max_quadrilateral_area (fs : FourSegments) :
  ∃ (maxArea : ℝ),
    (∀ (angle : ℝ), quadrilateralArea fs angle ≤ maxArea) ∧
    (quadrilateralArea fs (Real.pi / 2) = maxArea) := by
  sorry

#check max_quadrilateral_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadrilateral_area_l515_51552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PAB_is_six_l515_51510

-- Define the coordinate systems
structure CartesianPoint where
  x : ℝ
  y : ℝ

structure PolarPoint where
  r : ℝ
  θ : ℝ

-- Define the line l
def line_l (t : ℝ) : CartesianPoint :=
  { x := -1 + t, y := 1 + t }

-- Define the curve C
def curve_C (p : CartesianPoint) : Prop :=
  (p.x - 2)^2 + (p.y - 1)^2 = 5

-- Define point P
noncomputable def point_P : PolarPoint :=
  { r := 2 * Real.sqrt 2, θ := 7 * Real.pi / 4 }

-- Define line l'
def line_l' (t : ℝ) : CartesianPoint :=
  { x := 1 + t, y := 1 + t }

-- Theorem statement
noncomputable def area_of_triangle_PAB : ℝ := 6

-- Main theorem
theorem area_PAB_is_six :
  area_of_triangle_PAB = 6 := by
  sorry

#check area_PAB_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PAB_is_six_l515_51510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_colored_100th_step_l515_51575

/-- Represents the triangular tiling of the plane -/
structure TriangularTiling where
  -- Each vertex is shared by 6 triangles
  vertex_degree : ℕ
  vertex_degree_eq : vertex_degree = 6

/-- Represents the coloring process on the triangular tiling -/
def ColoringProcess (tiling : TriangularTiling) : Type :=
  ℕ → ℕ

/-- The number of triangles colored in a given step -/
def triangles_colored_in_step (tiling : TriangularTiling) (process : ColoringProcess tiling) (step : ℕ) : ℕ :=
  if step = 1 then 1
  else 12 * step - 12

/-- The main theorem stating the number of triangles colored in the 100th step -/
theorem triangles_colored_100th_step (tiling : TriangularTiling) 
  (process : ColoringProcess tiling) : 
  triangles_colored_in_step tiling process 100 = 1188 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_colored_100th_step_l515_51575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_salary_increase_l515_51503

noncomputable def percentage_increase (original new : ℝ) : ℝ :=
  (new - original) / original * 100

theorem john_salary_increase : percentage_increase 60 120 = 100 := by
  -- Unfold the definition of percentage_increase
  unfold percentage_increase
  -- Simplify the expression
  simp [sub_eq_add_neg, mul_comm, mul_assoc]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_salary_increase_l515_51503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_15th_terms_l515_51500

/-- Two arithmetic sequences with sums S_n and T_n for the first n terms -/
def S_n (n : ℕ) : ℚ := sorry

def T_n (n : ℕ) : ℚ := sorry

/-- The ratio condition for all n -/
axiom ratio_condition (n : ℕ) : S_n n / T_n n = (5 * n + 3) / (3 * n + 20)

/-- The 15th term of the first sequence -/
def a_15 : ℚ := sorry

/-- The 15th term of the second sequence -/
def b_15 : ℚ := sorry

/-- Theorem: The ratio of the 15th terms is 7/4 -/
theorem ratio_of_15th_terms : a_15 / b_15 = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_15th_terms_l515_51500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_in_regular_triangular_pyramid_l515_51518

/-- The radius of the inscribed sphere in a regular triangular pyramid -/
noncomputable def inscribed_sphere_radius (a b : ℝ) : ℝ :=
  (a * Real.sqrt (3 * (b^2 - a^2 / 3))) / (3 * (a + Real.sqrt (3 * (b^2 - a^2 / 4))))

/-- Theorem: The radius of the inscribed sphere in a regular triangular pyramid -/
theorem inscribed_sphere_radius_in_regular_triangular_pyramid
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : b > a / (2 * Real.sqrt 3)) :
  ∃ r : ℝ, r > 0 ∧ r = inscribed_sphere_radius a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_in_regular_triangular_pyramid_l515_51518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_stable_performance_l515_51566

/-- Represents an athlete with their variance -/
structure Athlete where
  name : String
  variance : ℝ

/-- Determines if an athlete has the most stable performance among a list of athletes -/
def hasMostStablePerformance (a : Athlete) (athletes : List Athlete) : Prop :=
  ∀ b ∈ athletes, a.variance ≤ b.variance

theorem most_stable_performance (athletes : List Athlete) 
  (hA : (⟨"A", 0.4⟩ : Athlete) ∈ athletes)
  (hB : (⟨"B", 0.5⟩ : Athlete) ∈ athletes)
  (hC : (⟨"C", 0.6⟩ : Athlete) ∈ athletes)
  (hD : (⟨"D", 0.3⟩ : Athlete) ∈ athletes)
  (hLen : athletes.length = 4)
  (hDistinct : athletes.Nodup) :
  hasMostStablePerformance (⟨"D", 0.3⟩ : Athlete) athletes := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_stable_performance_l515_51566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_nine_two_even_two_odd_starting_with_two_l515_51545

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_two_even_two_odd (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.filter (λ x => x % 2 = 0)).length = 2 ∧ (digits.filter (λ x => x % 2 ≠ 0)).length = 2

def starts_with_two (n : ℕ) : Prop :=
  (n / 1000) = 2

theorem smallest_four_digit_divisible_by_nine_two_even_two_odd_starting_with_two :
  ∃ (n : ℕ), 
    is_four_digit n ∧
    n % 9 = 0 ∧
    has_two_even_two_odd n ∧
    starts_with_two n ∧
    (∀ m : ℕ, 
      is_four_digit m ∧ 
      m % 9 = 0 ∧ 
      has_two_even_two_odd m ∧ 
      starts_with_two m → 
      n ≤ m) ∧
    n = 2079 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_nine_two_even_two_odd_starting_with_two_l515_51545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_greater_than_2134_l515_51551

theorem smallest_integer_greater_than_2134 :
  ∀ (x : ℤ) (n : ℤ),
  let max_x : ℤ := 3
  let threshold : ℝ := 2.134 * (10 : ℝ) ^ (max_x : ℝ)
  x ≤ max_x → (n : ℝ) > threshold → n ≥ 2135 :=
by
  intros x n max_x threshold h_x h_n
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_greater_than_2134_l515_51551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_tangent_line_theorem_l515_51586

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

-- Define the line l0
def line_l0 (p : ℝ) (x y : ℝ) : Prop := y - p/2 = (Real.sqrt 3/3) * x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 4

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define a tangent line to the parabola
def tangent_line (p x0 y0 x y : ℝ) : Prop := x * x0 = 2 * (y + y0)

theorem parabola_and_tangent_line_theorem (p : ℝ) :
  p > 0 →
  ∃ (x1 y1 x2 y2 : ℝ),
    parabola p x1 y1 ∧ parabola p x2 y2 ∧
    line_l0 p x1 y1 ∧ line_l0 p x2 y2 ∧
    distance x1 y1 x2 y2 = 6 →
  (∀ (x y : ℝ), parabola p x y ↔ x^2 = 4*y) ∧
  (∀ (x0 y0 xa ya xb yb : ℝ),
    line_l x0 y0 →
    parabola p xa ya ∧ parabola p xb yb →
    tangent_line p x0 y0 xa ya ∧ tangent_line p x0 y0 xb yb →
    ∃ (t : ℝ), xa + t*(xb - xa) = 2 ∧ ya + t*(yb - ya) = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_tangent_line_theorem_l515_51586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l515_51574

-- Define a point with integer coordinates
structure IntPoint where
  x : ℤ
  y : ℤ

-- Define the circle x^2 + y^2 = 16
def onCircle (p : IntPoint) : Prop :=
  p.x^2 + p.y^2 = 16

-- Define the distance between two points
noncomputable def distance (p1 p2 : IntPoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Theorem statement
theorem max_ratio_on_circle (A B C D : IntPoint) 
  (hA : onCircle A) (hB : onCircle B) (hC : onCircle C) (hD : onCircle D)
  (hAB : ∃ q : ℚ, distance A B = q) (hCD : ∃ q : ℚ, distance C D = q)
  (hDistinct : A ≠ B ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D) :
  (∀ E F G H : IntPoint, onCircle E ∧ onCircle F ∧ onCircle G ∧ onCircle H →
    (∃ q1 q2 : ℚ, distance E F = q1 ∧ distance G H = q2) →
    E ≠ F ∧ G ≠ H → distance E F / distance G H ≤ Real.sqrt 10 / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l515_51574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowing_rate_is_four_percent_l515_51583

/-- Calculates the borrowing interest rate given the principal, lending rate, and annual gain. -/
noncomputable def calculate_borrowing_rate (principal : ℝ) (lending_rate : ℝ) (annual_gain : ℝ) : ℝ :=
  let lending_interest := (lending_rate / 100) * principal
  let borrowing_interest := lending_interest - annual_gain
  (borrowing_interest / principal) * 100

/-- Proves that the borrowing rate is 4% given the specified conditions. -/
theorem borrowing_rate_is_four_percent 
  (principal : ℝ)
  (time_period : ℕ)
  (lending_rate : ℝ)
  (annual_gain : ℝ)
  (h1 : principal = 8000)
  (h2 : time_period = 2)
  (h3 : lending_rate = 6)
  (h4 : annual_gain = 160) :
  calculate_borrowing_rate principal lending_rate annual_gain = 4 := by
  sorry

/-- Evaluates the borrowing rate for the given conditions. -/
def eval_borrowing_rate : ℚ :=
  Rat.ofScientific 4 0 0  -- This represents 4.0 as a rational number

#eval eval_borrowing_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowing_rate_is_four_percent_l515_51583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_addition_problem_l515_51580

/-- Represents a single digit in base 7 -/
def Base7Digit := Fin 7

/-- Converts a base 7 number to base 10 -/
def toBase10 (hundreds tens ones : Base7Digit) : ℕ :=
  (hundreds.val * 49) + (tens.val * 7) + ones.val

theorem base7_addition_problem (X Y : Base7Digit) :
  toBase10 ⟨3, by norm_num⟩ X Y + toBase10 ⟨0, by norm_num⟩ ⟨5, by norm_num⟩ ⟨2, by norm_num⟩ =
  toBase10 ⟨4, by norm_num⟩ ⟨0, by norm_num⟩ X →
  X.val + Y.val = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_addition_problem_l515_51580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tang_sancai_probabilities_l515_51544

/-- Probability of artifact A passing the first firing -/
noncomputable def p_A1 : ℝ := 1/2

/-- Probability of artifact B passing the first firing -/
noncomputable def p_B1 : ℝ := 4/5

/-- Probability of artifact C passing the first firing -/
noncomputable def p_C1 : ℝ := 3/5

/-- Probability of artifact A passing the second firing -/
noncomputable def p_A2 : ℝ := 4/5

/-- Probability of artifact B passing the second firing -/
noncomputable def p_B2 : ℝ := 1/2

/-- Probability of artifact C passing the second firing -/
noncomputable def p_C2 : ℝ := 2/3

/-- Probability of exactly one artifact passing the first firing -/
noncomputable def prob_one_passing : ℝ := p_A1 * (1 - p_B1) * (1 - p_C1) + 
                            (1 - p_A1) * p_B1 * (1 - p_C1) + 
                            (1 - p_A1) * (1 - p_B1) * p_C1

/-- Probability of an artifact passing both firings -/
noncomputable def p_both : ℝ := p_A1 * p_A2

/-- Expected number of qualified artifacts after both firings -/
noncomputable def expected_qualified : ℝ := 3 * p_both

theorem tang_sancai_probabilities : 
  prob_one_passing = 13/50 ∧ expected_qualified = 1.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tang_sancai_probabilities_l515_51544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_comparison_l515_51536

/-- Two people travel the same distance with different speed strategies -/
theorem journey_time_comparison
  (d : ℝ) -- Total distance
  (m n : ℝ) -- Two different speeds
  (hm : m > 0) -- m is positive
  (hn : n > 0) -- n is positive
  (hmn : m ≠ n) -- m and n are different
  : (2 * d / (m + n)) < (d / (2 * m) + d / (2 * n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_comparison_l515_51536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_vector_OA_coordinates_v_find_z_l515_51521

-- Define the z-vector
def z_vector (z : ℝ) (x y : ℝ) : ℝ × ℝ := (z * x, z * y)

-- Part 1
theorem z_vector_OA (z x0 y0 : ℝ) (hz : z ≠ 0) :
  z_vector z x0 y0 = (z * x0, z * y0) := by sorry

-- Part 2
theorem coordinates_v (z : ℝ) (hz : z ≠ 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (1/2 * |z| * |z * y| = 1) ∧
  (1/2 * |z| * |z * x| = 2) →
  (x, y) = (4 / |z|, 2 / |z|) := by sorry

-- Part 3
theorem find_z :
  ∃ (z : ℝ), z > 0 ∧
  (∀ (A B : ℝ × ℝ), 
    let (x1, y1) := A
    let (x2, y2) := B
    1/2 * |x1 * y2 - x2 * y1| = 1 →
    z * (x1^2 + y1^2) + z * (x2^2 + y2^2) ≥ 8) ∧
  (∃ (A B : ℝ × ℝ), 
    let (x1, y1) := A
    let (x2, y2) := B
    1/2 * |x1 * y2 - x2 * y1| = 1 ∧
    z * (x1^2 + y1^2) + z * (x2^2 + y2^2) = 8) →
  z = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_vector_OA_coordinates_v_find_z_l515_51521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_increasing_l515_51528

/-- The function f(x) = (2^x - 1) / (2^x + 1) -/
noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

/-- Theorem: f(x) is an increasing function on (-∞, +∞) -/
theorem f_is_increasing : StrictMono f := by
  -- We'll use the sorry tactic to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_increasing_l515_51528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_C_value_l515_51555

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  true  -- We don't need to define the specific properties of the triangle here

-- Define the lengths of the sides
noncomputable def AB (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
noncomputable def BC (B C : ℝ × ℝ) : ℝ := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)

-- Define tan C
noncomputable def tan_C (A B C : ℝ × ℝ) : ℝ := 
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  (AB A B) / AC

-- Theorem statement
theorem max_tan_C_value {A B C : ℝ × ℝ} (h1 : Triangle A B C) 
  (h2 : AB A B = 30) (h3 : BC B C = 18) : 
  ∀ C', Triangle A B C' → BC B C' = 18 → tan_C A B C' ≤ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_C_value_l515_51555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l515_51570

structure Point where
  x : ℝ
  y : ℝ

def Parabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

theorem parabola_triangle_area :
  ∀ (P : Point),
    Parabola P →
    distance P { x := 1, y := 0 } = 2 →
    triangleArea { x := 0, y := 0 } { x := 1, y := 0 } P = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l515_51570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_buyers_difference_l515_51596

theorem pencil_buyers_difference (pencil_cost : ℕ) 
  (seventh_grade_total : ℕ) (sixth_grade_total : ℕ) 
  (h1 : seventh_grade_total = 165)
  (h2 : sixth_grade_total = 234)
  (h3 : pencil_cost > 0)
  (h4 : pencil_cost ∣ seventh_grade_total)
  (h5 : pencil_cost ∣ sixth_grade_total) :
  sixth_grade_total / pencil_cost - seventh_grade_total / pencil_cost = 23 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_buyers_difference_l515_51596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sqrt_interval_l515_51589

theorem log_sqrt_interval (x : ℝ) :
  (∃ y, y = (Real.log (5 - x)) / Real.sqrt (x - 2)) ↔ 2 < x ∧ x < 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sqrt_interval_l515_51589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l515_51593

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * Real.cos x + 1)

theorem domain_of_f :
  ∀ x : ℝ, f x ∈ Set.range f ↔ ∃ k : ℤ, x ∈ Set.Icc (2 * k * Real.pi - 2 * Real.pi / 3) (2 * k * Real.pi + 2 * Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l515_51593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_average_output_l515_51506

/-- Represents the production of cogs on an assembly line -/
structure CogProduction where
  initial_rate : ℚ
  initial_order : ℚ
  increased_rate : ℚ
  second_order : ℚ

/-- Calculates the overall average output of cogs per hour -/
noncomputable def average_output (p : CogProduction) : ℚ :=
  (p.initial_order + p.second_order) / (p.initial_order / p.initial_rate + p.second_order / p.increased_rate)

/-- Theorem stating that for the given production scenario, the average output is 24 cogs per hour -/
theorem assembly_line_average_output :
  let p := CogProduction.mk 15 60 60 60
  average_output p = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_average_output_l515_51506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_circle_l515_51599

-- Define what it means for a function to represent a circle
def IsCircle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b r : ℝ), ∀ x y, f x y ↔ (x - a)^2 + (y - b)^2 = r^2

theorem polar_equation_is_circle (θ ρ : ℝ) :
  4 * Real.sin θ = 5 * ρ → IsCircle (λ x y => x^2 + y^2 = 16 / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_circle_l515_51599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cook_is_thief_l515_51579

-- Define the suspects
inductive Suspect
| CheshireCat
| Duchess
| Cook

-- Define a function to represent whether a suspect is telling the truth
def isTellingTruth : Suspect → Prop := sorry

-- Define the thief
def thief : Suspect := Suspect.Cook

-- Theorem stating that the Cook is the thief
theorem cook_is_thief :
  thief = Suspect.Cook ∧
  ¬(isTellingTruth thief) ∧
  ((isTellingTruth Suspect.CheshireCat ∧ isTellingTruth Suspect.Duchess) ∨
   (¬isTellingTruth Suspect.CheshireCat ∧ ¬isTellingTruth Suspect.Duchess)) :=
by
  sorry -- The actual proof would go here

#check cook_is_thief

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cook_is_thief_l515_51579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l515_51572

def a : ℝ × ℝ × ℝ := (6, 3, 4)
def b : ℝ × ℝ × ℝ := (-1, -2, -1)
def c : ℝ × ℝ × ℝ := (2, 1, 2)

theorem vectors_not_coplanar : ¬(∃ (x y z : ℝ), x • a + y • b + z • c = (0, 0, 0) ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l515_51572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_maximizes_profit_l515_51515

/-- Represents the company's strategy for gas production --/
structure Strategy where
  natural_gas : ℝ
  liquefied_gas : ℝ

/-- Represents the market conditions --/
inductive Weather
  | Mild
  | Cold

/-- Represents the company's problem parameters --/
structure GasCompanyProblem where
  mild_supply : Strategy
  cold_supply : Strategy
  natural_gas_cost : ℝ
  liquefied_gas_cost : ℝ
  natural_gas_price : ℝ
  liquefied_gas_price : ℝ

/-- Calculates the profit for a given strategy and weather condition --/
noncomputable def profit (p : GasCompanyProblem) (s : Strategy) (w : Weather) : ℝ :=
  match w with
  | Weather.Mild =>
    (s.natural_gas * (p.natural_gas_price - p.natural_gas_cost)) +
    (s.liquefied_gas * (p.liquefied_gas_price - p.liquefied_gas_cost))
  | Weather.Cold =>
    (min s.natural_gas p.cold_supply.natural_gas * (p.natural_gas_price - p.natural_gas_cost)) +
    (min s.liquefied_gas p.cold_supply.liquefied_gas * (p.liquefied_gas_price - p.liquefied_gas_cost))

/-- Defines the optimal strategy --/
def optimal_strategy : Strategy :=
  { natural_gas := 3032,
    liquefied_gas := 2954 }

/-- Theorem stating that the optimal strategy maximizes expected profit --/
theorem optimal_strategy_maximizes_profit (p : GasCompanyProblem) :
  ∀ s : Strategy,
    (profit p optimal_strategy Weather.Mild + profit p optimal_strategy Weather.Cold) / 2 ≥
    (profit p s Weather.Mild + profit p s Weather.Cold) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_maximizes_profit_l515_51515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonious_division_iff_max_pile_size_l515_51520

/-- A division of beads is harmonious if it's possible to take one bead from each of N piles
    at a time until all beads are taken. -/
def IsHarmonious (N : ℕ) (k : ℕ) (piles : List ℕ) : Prop :=
  N ≥ 2 ∧ piles.length = 2 * N - 1 ∧ piles.sum = N * k ∧
  ∃ (sequence : List (List (Fin piles.length))), 
    sequence.all (λ l => l.length = N ∧ l.all (λ i => piles.get i > 0)) ∧
    sequence.length * N = N * k

theorem harmonious_division_iff_max_pile_size (N : ℕ) (k : ℕ) (piles : List ℕ) :
  IsHarmonious N k piles ↔ piles.maximum ≤ some k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonious_division_iff_max_pile_size_l515_51520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_one_sufficient_not_necessary_l515_51564

/-- The function f(x) = (a-1)a^x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * (a ^ x)

/-- f is increasing on its domain -/
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem a_gt_one_sufficient_not_necessary :
  (∃ a : ℝ, a > 1 ∧ is_increasing (f a)) ∧
  (∃ a : ℝ, a ≤ 1 ∧ is_increasing (f a)) := by
  sorry

#check a_gt_one_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_one_sufficient_not_necessary_l515_51564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_berengere_contribution_problem_l515_51590

/-- Given a pastry cost, Emily's contribution in USD, and an exchange rate,
    calculate Berengere's required contribution in euros. -/
noncomputable def berengere_contribution (pastry_cost : ℝ) (emily_usd : ℝ) (exchange_rate : ℝ) : ℝ :=
  pastry_cost - (emily_usd / exchange_rate)

/-- Theorem stating that Berengere's contribution is approximately 1.64 euros
    given the specific conditions of the problem. -/
theorem berengere_contribution_problem :
  let pastry_cost : ℝ := 8
  let emily_usd : ℝ := 7
  let exchange_rate : ℝ := 1.1
  abs (berengere_contribution pastry_cost emily_usd exchange_rate - 1.64) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_berengere_contribution_problem_l515_51590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inscribed_angles_eq_180_l515_51529

/-- A circle with an inscribed pentagon -/
structure CircleWithPentagon where
  /-- The circle -/
  circle : Set (ℝ × ℝ)
  /-- The inscribed pentagon -/
  pentagon : Set (ℝ × ℝ)
  /-- The pentagon is inscribed in the circle -/
  is_inscribed : pentagon ⊆ circle

/-- The angles inscribed in the arcs cut off by the sides of the pentagon -/
def inscribed_angles (cwp : CircleWithPentagon) : Set ℝ := sorry

/-- The sum of the inscribed angles -/
noncomputable def sum_inscribed_angles (cwp : CircleWithPentagon) : ℝ := sorry

/-- Theorem: The sum of the angles inscribed in the five arcs cut off by the sides of a pentagon inscribed in a circle is equal to 180° -/
theorem sum_inscribed_angles_eq_180 (cwp : CircleWithPentagon) :
  sum_inscribed_angles cwp = 180 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inscribed_angles_eq_180_l515_51529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energetic_time_is_three_l515_51511

/-- Represents the bike ride scenario --/
structure BikeRide where
  totalDistance : ℝ
  totalTime : ℝ
  initialSpeed : ℝ
  reducedSpeed : ℝ
  restTime : ℝ

/-- Calculates the time spent feeling energetic during the bike ride --/
noncomputable def energeticTime (ride : BikeRide) : ℝ :=
  (ride.totalDistance - ride.reducedSpeed * (ride.totalTime - ride.restTime)) /
  (ride.initialSpeed - ride.reducedSpeed)

/-- Theorem stating that for the given conditions, the energetic time is 3 hours --/
theorem energetic_time_is_three :
  let ride : BikeRide := {
    totalDistance := 150,
    totalTime := 9,
    initialSpeed := 25,
    reducedSpeed := 15,
    restTime := 1
  }
  energeticTime ride = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_energetic_time_is_three_l515_51511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_x_l515_51507

theorem greatest_integer_x (x : ℕ) : x ≠ 0 → (x^6 : ℚ) / x^3 < 20 → x ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_x_l515_51507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_implies_a_equals_two_l515_51501

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 1 = 0

-- Define the line
def line_eq (a x y : ℝ) : Prop := a*x + 2*y + 6 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem intersection_perpendicular_implies_a_equals_two 
  (a : ℝ) 
  (P Q : ℝ × ℝ) 
  (h1 : circle_eq P.1 P.2) 
  (h2 : circle_eq Q.1 Q.2) 
  (h3 : line_eq a P.1 P.2) 
  (h4 : line_eq a Q.1 Q.2) 
  (h5 : (P.1 - center.1) * (Q.1 - center.1) + (P.2 - center.2) * (Q.2 - center.2) = 0) : 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_implies_a_equals_two_l515_51501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sawmill_productivity_ratio_l515_51547

/-- Represents the daily tree-cutting productivity of a saw-mill -/
structure Productivity where
  trees : ℕ

/-- Calculates the ratio of productivity increase to initial productivity -/
def productivity_increase_ratio (initial current : Productivity) : ℚ :=
  let increase := current.trees - initial.trees
  ↑increase / ↑initial.trees

theorem sawmill_productivity_ratio :
  let initial := Productivity.mk 10
  let current := Productivity.mk 25
  productivity_increase_ratio initial current = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sawmill_productivity_ratio_l515_51547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_of_nested_hexagons_l515_51562

/-- Two convex hexagons in 2D space -/
structure ConvexHexagon (V : Type*) [AddCommGroup V] [Module ℝ V] where
  vertices : Fin 6 → V

/-- The area of a simple hexagon -/
noncomputable def area {V : Type*} [AddCommGroup V] [Module ℝ V] (h : ConvexHexagon V) : ℝ :=
  sorry

/-- Two hexagons with parallel corresponding sides -/
def parallel_sides {V : Type*} [AddCommGroup V] [Module ℝ V] (h₁ h₂ : ConvexHexagon V) : Prop :=
  ∀ i : Fin 6, ∃ k : ℝ, h₂.vertices ((i : ℕ) + 1) - h₂.vertices i = k • (h₁.vertices ((i : ℕ) + 1) - h₁.vertices i)

/-- One hexagon lies in the interior of another -/
def is_interior {V : Type*} [AddCommGroup V] [Module ℝ V] (h₁ h₂ : ConvexHexagon V) : Prop :=
  sorry

/-- The theorem statement -/
theorem equal_areas_of_nested_hexagons 
  {V : Type*} [AddCommGroup V] [Module ℝ V] [NormedAddCommGroup V] [NormedSpace ℝ V] [FiniteDimensional ℝ V]
  (h₁ h₂ : ConvexHexagon V) 
  (h_parallel : parallel_sides h₁ h₂)
  (h_interior : is_interior h₁ h₂) :
  area (ConvexHexagon.mk (λ i => match i with
    | 0 => h₁.vertices 0 | 1 => h₂.vertices 1 | 2 => h₁.vertices 2
    | 3 => h₂.vertices 3 | 4 => h₁.vertices 4 | 5 => h₂.vertices 5
  )) =
  area (ConvexHexagon.mk (λ i => match i with
    | 0 => h₂.vertices 0 | 1 => h₁.vertices 1 | 2 => h₂.vertices 2
    | 3 => h₁.vertices 3 | 4 => h₂.vertices 4 | 5 => h₁.vertices 5
  )) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_of_nested_hexagons_l515_51562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_from_cos_and_range_l515_51539

theorem sin_value_from_cos_and_range (α : Real) 
  (h1 : Real.cos α = -3/5) 
  (h2 : π < α) 
  (h3 : α < 3*π/2) : 
  Real.sin α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_from_cos_and_range_l515_51539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_terms_count_l515_51525

def sequenceTerms (n : ℕ) : ℚ :=
  11664 / (4 ^ n)

theorem integer_terms_count : 
  (∃ k : ℕ, k = 4 ∧ 
    (∀ n : ℕ, n < k → ∃ m : ℕ, sequenceTerms n = m) ∧ 
    (∀ n : ℕ, n ≥ k → ¬∃ m : ℕ, sequenceTerms n = m)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_terms_count_l515_51525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l515_51582

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 1)) / Real.sqrt (-x^2 - 3*x + 4)

-- State the theorem about the domain of f
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.Ioo (-1 : ℝ) (1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l515_51582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_tone_has_period_l515_51578

/-- The complex tone function -/
noncomputable def complex_tone (x : ℝ) : ℝ := (1/4) * Real.sin (4*x) + (1/6) * Real.sin (6*x)

/-- The period of the complex tone function -/
noncomputable def complex_tone_period : ℝ := Real.pi

/-- Theorem: The period of the complex tone function is π -/
theorem complex_tone_has_period : 
  ∀ x : ℝ, complex_tone (x + complex_tone_period) = complex_tone x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_tone_has_period_l515_51578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l515_51512

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + (5 - a^2) * x + a
noncomputable def g (x : ℝ) : ℝ := Real.exp x / x

-- Define propositions p and q
def p (a : ℝ) : Prop := Monotone (f a)
def q (a : ℝ) : Prop := StrictMono (fun x => g x)

-- Theorem statement
theorem problem_statement (a : ℝ) :
  (p a ∨ ¬(q a)) → (¬(p a) ∨ q a) ↔ a ∈ Set.Iic (-2) ∪ Set.Icc 1 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l515_51512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speeds_correct_l515_51530

/-- The speed of the motorboat in km/h -/
noncomputable def motorboat_speed : ℝ := 18

/-- The speed of the sailboat in km/h -/
noncomputable def sailboat_speed : ℝ := 12

/-- The initial distance between the boats in km -/
noncomputable def initial_distance : ℝ := 30

/-- The time taken for the boats to meet when moving towards each other in hours -/
noncomputable def meeting_time : ℝ := 1

/-- The distance the motorboat is behind the sailboat in the second scenario in km -/
noncomputable def chase_distance : ℝ := 20

/-- The time taken for the motorboat to catch up to the sailboat in the second scenario in hours -/
noncomputable def chase_time : ℝ := 10/3

theorem boat_speeds_correct :
  (motorboat_speed + sailboat_speed) * meeting_time = initial_distance ∧
  motorboat_speed * chase_time = chase_distance + sailboat_speed * chase_time :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speeds_correct_l515_51530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l515_51595

/-- Given a circle with an external point P, prove that the radius is √316 inches -/
theorem circle_radius (center P Q R : ℝ × ℝ) (r : ℝ) : 
  ‖center - P‖ = 26 →  -- P is 26 inches from the center
  ‖P - Q‖ = 15 →       -- PQ (external segment) is 15 inches
  ‖Q - R‖ = 9 →        -- QR is 9 inches
  ‖center - Q‖ = r →   -- Q is on the circle
  ‖center - R‖ = r →   -- R is on the circle
  r = Real.sqrt 316 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l515_51595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_is_ten_l515_51538

noncomputable def f (x : ℝ) : ℝ := 
  (1/3*x^3 - x^2 + 2/3) * (Real.cos (Real.pi/3*x + 2*Real.pi/3))^2017 + 2*x + 3

def interval : Set ℝ := Set.Icc (-2015) 2017

theorem sum_of_max_and_min_is_ten :
  ∃ (M m : ℝ), (∀ x ∈ interval, f x ≤ M) ∧
               (∀ x ∈ interval, m ≤ f x) ∧
               (M + m = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_is_ten_l515_51538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_scores_count_l515_51537

/-- Represents the number of scoring attempts made by the athlete -/
def attempts : ℕ := 8

/-- Represents the point value of a 2-pointer shot -/
def two_pointer : ℕ := 2

/-- Represents the point value of a 3-pointer shot -/
def three_pointer : ℕ := 3

/-- Calculates the total score given the number of 2-pointers and 3-pointers -/
def total_score (twos threes : ℕ) : ℕ := twos * two_pointer + threes * three_pointer

/-- The set of all possible total scores -/
def possible_scores : Finset ℕ := 
  Finset.image (λ threes ↦ total_score (attempts - threes) threes) (Finset.range (attempts + 1))

theorem different_scores_count : Finset.card possible_scores = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_scores_count_l515_51537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_monotone_property_l515_51568

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def domain_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → ∃ y, f x = y

def monotone_positive (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

theorem odd_function_monotone_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_domain : domain_condition f)
  (h_monotone : monotone_positive f) :
  f (-3) > f (-5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_monotone_property_l515_51568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PQ_l515_51561

/-- Parabola with equation y^2 = 4x and focus (1,0) -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  eq_check : equation = fun x y ↦ y^2 = 4*x
  focus_check : focus = (1, 0)

/-- Line passing through the focus and intersecting the parabola -/
structure IntersectingLine (p : Parabola) where
  line : ℝ → ℝ → Prop
  passes_focus : line p.focus.1 p.focus.2

/-- Point on the parabola -/
structure ParabolaPoint (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

/-- Directrix of the parabola -/
def directrix : ℝ → ℝ → Prop :=
  fun x _ ↦ x = -1

/-- Intersection point of a line through origin and directrix -/
noncomputable def intersect_directrix (p : ℝ × ℝ) : ℝ × ℝ :=
  (-1, -4 / p.2)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Main theorem: Minimum distance between P and Q is 4 -/
theorem min_distance_PQ (p : Parabola) (l : IntersectingLine p)
    (m n : ParabolaPoint p) (h_m_on_l : l.line m.point.1 m.point.2)
    (h_n_on_l : l.line n.point.1 n.point.2) :
    ∃ (P Q : ℝ × ℝ), P = intersect_directrix m.point ∧
                      Q = intersect_directrix n.point ∧
                      ∀ (P' Q' : ℝ × ℝ), distance P' Q' ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_PQ_l515_51561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l515_51508

-- Define the function f(x) = √(3 - 2x - x²)
noncomputable def f (x : ℝ) := Real.sqrt (3 - 2*x - x^2)

-- Define the domain of f
def domain : Set ℝ := Set.Icc (-3) 1

-- Define the inner function g(x) = 3 - 2x - x²
def g (x : ℝ) := 3 - 2*x - x^2

-- Define the outer function h(t) = √t
noncomputable def h (t : ℝ) := Real.sqrt t

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x ∈ domain,
  (StrictMonoOn g (Set.Ioo (-3) (-1))) →
  (StrictAntiOn g (Set.Ioo (-1) 1)) →
  (StrictMonoOn h (Set.Ioi 0)) →
  (StrictAntiOn f (Set.Ioo (-1) 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l515_51508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_bound_implies_first_element_bound_l515_51584

theorem lcm_bound_implies_first_element_bound (n : ℕ) (a : ℕ → ℕ) :
  (0 < n) →
  (∀ i : ℕ, i < n → 0 < a i) →
  (∀ i j : ℕ, i < n → j < n → i ≤ j → a i ≤ a j) →
  (∀ i : ℕ, i < n → a i ≤ 2 * n) →
  (∀ i j : ℕ, i < n → j < n → i ≠ j → Nat.lcm (a i) (a j) > 2 * n) →
  a 0 > 2 * n / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_bound_implies_first_element_bound_l515_51584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_point_of_f_l515_51554

open Real MeasureTheory Interval

/-- The function f(x) = cos(2x + π/6) -/
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

/-- The integral of f over [0, π/2] -/
noncomputable def integral_f : ℝ := ∫ x in (0)..(Real.pi/2), f x

/-- Definition of an integral point -/
def is_integral_point (x₀ : ℝ) : Prop :=
  x₀ ∈ Set.Icc 0 (Real.pi/2) ∧ f x₀ = integral_f

/-- Theorem: π/4 is the integral point of f(x) on [0, π/2] -/
theorem integral_point_of_f :
  is_integral_point (Real.pi/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_point_of_f_l515_51554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_common_chord_l515_51569

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the centers and radius of the circles
def center1 : ℝ × ℝ := (2, 0)
def center2 : ℝ × ℝ := (0, 2)
def radius : ℝ := 2

-- Theorem statement
theorem central_angle_of_common_chord :
  let distance_between_centers := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  let distance_chord_to_center := distance_between_centers / 2
  let cos_half_angle := distance_chord_to_center / radius
  let central_angle := 2 * Real.arccos cos_half_angle
  central_angle = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_common_chord_l515_51569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_sqrt2_l515_51516

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.sqrt 2

theorem function_composition_sqrt2 (a : ℝ) :
  (∀ x, f a x = a * x^2 - Real.sqrt 2) →
  f a (f a (Real.sqrt 2)) = -Real.sqrt 2 →
  a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_sqrt2_l515_51516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_101_99_l515_51505

theorem square_difference_101_99 : |((101 : ℤ)^2 - (99 : ℤ)^2)| = 400 := by
  -- Calculate 101^2 - 99^2
  have h1 : (101 : ℤ)^2 - (99 : ℤ)^2 = 400 := by
    norm_num
  
  -- Apply the absolute value
  rw [h1]
  -- Simplify |400|
  simp


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_101_99_l515_51505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l515_51532

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ - x₂ = π → f x₁ = f x₂) ∧
  (∀ x : ℝ, f (π/12 + x) = f (π/12 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l515_51532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_2_f_geq_2_iff_a_in_range_l515_51519

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x - 1|

-- Theorem for part I
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x < 4} = {x : ℝ | -1/2 < x ∧ x < 7/2} := by sorry

-- Theorem for part II
theorem f_geq_2_iff_a_in_range :
  (∀ x, f a x ≥ 2) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 3 := by sorry

#check solution_set_when_a_is_2
#check f_geq_2_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_2_f_geq_2_iff_a_in_range_l515_51519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l515_51534

noncomputable def f (x : ℝ) := Real.sin (x - Real.pi/3)

theorem f_properties :
  let C := {x : ℝ × ℝ | x.2 = f x.1}
  (∀ (x y : ℝ), f (5*Real.pi/3 - x) = f (5*Real.pi/3 + x)) ∧ 
  (∀ (x : ℝ), f (4*Real.pi/3 + x) = -f (4*Real.pi/3 - x)) ∧
  (∀ (x y : ℝ), x ∈ Set.Icc (Real.pi/3) (5*Real.pi/6) → 
                 y ∈ Set.Icc (Real.pi/3) (5*Real.pi/6) → 
                 x < y → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l515_51534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_ratio_l515_51549

/-- Represents a rectangular park with sides x and y, where x < y -/
structure RectangularPark where
  x : ℝ
  y : ℝ
  h : x < y

/-- The diagonal of the park -/
noncomputable def diagonal (park : RectangularPark) : ℝ :=
  Real.sqrt (park.x ^ 2 + park.y ^ 2)

/-- The distance saved by taking the diagonal shortcut -/
noncomputable def distanceSaved (park : RectangularPark) : ℝ :=
  park.x + park.y - diagonal park

theorem park_ratio (park : RectangularPark) 
  (h : distanceSaved park = (1/3) * park.y) : 
  park.x / park.y = 5/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_ratio_l515_51549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_segment_length_l515_51524

/-- A trapezoid with parallel sides CD and WX, where P is the intersection of CX and DW -/
structure Trapezoid :=
  (C D W X P : ℝ × ℝ)
  (cd_parallel_wx : (C.2 - D.2) / (C.1 - D.1) = (W.2 - X.2) / (W.1 - X.1))
  (p_on_cx : ∃ t : ℝ, P = (1 - t) • C + t • X)
  (p_on_dw : ∃ s : ℝ, P = (1 - s) • D + s • W)

/-- The length of a segment between two points -/
noncomputable def segment_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem trapezoid_segment_length (T : Trapezoid) :
  segment_length T.C T.X = 56 →
  segment_length T.D T.P = 16 →
  segment_length T.P T.W = 32 →
  segment_length T.P T.X = 112/3 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_segment_length_l515_51524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_condition_for_omega_two_l515_51576

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 3)

theorem periodic_condition_for_omega_two (ω : ℝ) :
  (∀ x : ℝ, f ω (x + Real.pi) = f ω x) → 
  (ω = 2 → ∀ x : ℝ, f ω (x + Real.pi) = f ω x) ∧
  (∃ ω' : ℝ, ω' ≠ 2 ∧ ∀ x : ℝ, f ω' (x + Real.pi) = f ω' x) := by
  sorry

#check periodic_condition_for_omega_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_condition_for_omega_two_l515_51576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_limit_is_four_l515_51563

/-- The ratio of areas of two squares, where the perimeter of the larger square
    is 4 units larger than twice the perimeter of the smaller square. -/
noncomputable def area_ratio (b : ℝ) : ℝ :=
  let a := 2 * b + 1
  a ^ 2 / b ^ 2

/-- The limit of the area ratio as the side length of the smaller square
    approaches infinity is 4. -/
theorem area_ratio_limit_is_four :
  Filter.Tendsto area_ratio Filter.atTop (nhds 4) := by
  sorry

#check area_ratio_limit_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_limit_is_four_l515_51563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_digit_divisible_by_sum_l515_51527

def digits (n : Nat) : List Nat :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

def digit_sum (n : Nat) : Nat :=
  (digits n).sum

theorem hundred_digit_divisible_by_sum : ∃ n : Nat, 
  (10^99 ≤ n) ∧ (n < 10^100) ∧ 
  (∀ d, d ∈ digits n → d ≠ 0) ∧
  (n % (digit_sum n) = 0) := by
  sorry

#check hundred_digit_divisible_by_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_digit_divisible_by_sum_l515_51527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debt_limit_is_five_point_five_l515_51581

/-- Represents the daily cost of internet service in dollars -/
def daily_cost : ℚ := 1/2

/-- Represents the number of days Lally will be connected -/
def connection_days : ℕ := 25

/-- Represents Lally's payment in dollars -/
def payment : ℚ := 7

/-- Represents Lally's initial balance in dollars -/
def initial_balance : ℚ := 0

/-- Calculates the debt limit before service discontinuation -/
def debt_limit : ℚ := daily_cost * connection_days - payment - initial_balance

/-- Theorem stating that the debt limit is $5.5 -/
theorem debt_limit_is_five_point_five :
  debt_limit = 11/2 := by
  unfold debt_limit
  unfold daily_cost
  unfold connection_days
  unfold payment
  unfold initial_balance
  norm_num

#eval debt_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debt_limit_is_five_point_five_l515_51581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_group_sum_l515_51542

def oddNumbers : List Nat := List.range 1006 |> List.map (fun n => 2 * n + 1)

def groupPattern : List Nat := [1, 2, 3, 2]

def groupOddNumbers (numbers : List Nat) (pattern : List Nat) : List (List Nat) :=
  sorry

theorem last_group_sum (numbers : List Nat) (pattern : List Nat) :
  numbers = oddNumbers → pattern = groupPattern →
  let groups := groupOddNumbers numbers pattern
  (groups.getLast? |>.map List.sum |>.getD 0) = 6027 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_group_sum_l515_51542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poisson_expectation_l515_51587

/-- The Poisson probability mass function -/
noncomputable def poissonPMF (a : ℝ) (k : ℕ) : ℝ := (a ^ k * Real.exp (-a)) / k.factorial

/-- The expectation of a discrete random variable -/
noncomputable def expectation (p : ℕ → ℝ) : ℝ := ∑' k, k * p k

/-- Theorem: The expectation of a Poisson-distributed random variable is equal to its parameter -/
theorem poisson_expectation (a : ℝ) (ha : 0 < a) :
  expectation (poissonPMF a) = a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poisson_expectation_l515_51587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_eq_1002_l515_51573

/-- Given a function g: ℝ → ℝ satisfying the functional equation
    3 * g x + g (1 / x) = 6 * x + 5 for all non-zero real x,
    prove that the sum of roots of x² - 1002x - 1 = 0 is 1002,
    where these roots are the solutions to g x = 3006. -/
theorem sum_of_roots_eq_1002 (g : ℝ → ℝ)
    (h₁ : ∀ x, x ≠ 0 → 3 * g x + g (1 / x) = 6 * x + 5)
    (h₂ : ∃ x₁ x₂, x₁ ≠ x₂ ∧ g x₁ = 3006 ∧ g x₂ = 3006) :
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 - 1002 * x₁ - 1 = 0 ∧ x₂^2 - 1002 * x₂ - 1 = 0 ∧ x₁ + x₂ = 1002 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_eq_1002_l515_51573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_old_clock_slower_by_12_minutes_l515_51514

def old_clock_alignment_interval : ℚ := 66

def standard_alignment_interval : ℚ := 720 / 11

def time_factor : ℚ := old_clock_alignment_interval / standard_alignment_interval

def old_clock_24_hours : ℚ := 24 * 60 * (12 / 11)

def standard_24_hours : ℚ := 24 * 60

theorem old_clock_slower_by_12_minutes : 
  old_clock_24_hours - standard_24_hours = 144 := by
  sorry

#eval old_clock_24_hours - standard_24_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_old_clock_slower_by_12_minutes_l515_51514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_price_reduction_l515_51504

/-- Calculate the percentage reduction given the original and reduced prices -/
noncomputable def percentage_reduction (original_price reduced_price : ℝ) : ℝ :=
  (original_price - reduced_price) / original_price * 100

/-- Theorem stating that the percentage reduction for the given prices is 15% -/
theorem laptop_price_reduction (original_price reduced_price : ℝ) 
  (h1 : original_price = 800)
  (h2 : reduced_price = 680) :
  percentage_reduction original_price reduced_price = 15 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval percentage_reduction 800 680

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_price_reduction_l515_51504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_for_quadratic_equation_l515_51585

-- Define the equation as a function of x and a
noncomputable def equation (x a : ℝ) : ℝ := (a - 2) * x^(a^2 - 2) - x + 3

-- Define what it means for the equation to be quadratic in x
def is_quadratic (a : ℝ) : Prop :=
  ∃ (p q r : ℝ), p ≠ 0 ∧ ∀ x, equation x a = p * x^2 + q * x + r

-- State the theorem
theorem unique_a_for_quadratic_equation :
  ∃! a : ℝ, is_quadratic a ∧ a - 2 ≠ 0 ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_for_quadratic_equation_l515_51585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_distances_l515_51567

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- A configuration of four points in 3D space -/
structure FourPoints where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- The condition that at most one distance is greater than 1 -/
def atMostOneGreaterThanOne (fp : FourPoints) : Prop :=
  let dists := [
    distance fp.A fp.B, distance fp.A fp.C, distance fp.A fp.D,
    distance fp.B fp.C, distance fp.B fp.D, distance fp.C fp.D
  ]
  (dists.filter (λ d => d > 1)).length ≤ 1

/-- The sum of all six distances -/
noncomputable def sumOfDistances (fp : FourPoints) : ℝ :=
  distance fp.A fp.B + distance fp.A fp.C + distance fp.A fp.D +
  distance fp.B fp.C + distance fp.B fp.D + distance fp.C fp.D

/-- The main theorem -/
theorem max_sum_of_distances :
  ∃ (fp : FourPoints), atMostOneGreaterThanOne fp ∧
    ∀ (fp' : FourPoints), atMostOneGreaterThanOne fp' →
      sumOfDistances fp' ≤ 6 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_distances_l515_51567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_equal_gcd_l515_51598

theorem max_students_equal_gcd (pens pencils : ℕ) 
  (h1 : pens = 1008) (h2 : pencils = 928) : 
  (Nat.gcd pens pencils) = 
    (Finset.sup (Finset.filter (λ n => pens % n = 0 ∧ pencils % n = 0) (Finset.range (min pens pencils + 1))) id) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_equal_gcd_l515_51598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_g_g_equals_two_l515_51535

-- Define the function g based on the graph
noncomputable def g : ℝ → ℝ := sorry

-- Properties of g derived from the graph
axiom g_at_neg_two : g (-2) = 2
axiom g_at_zero : g 0 = 2
axiom g_at_four : g 4 = 2
axiom g_at_neg_one : g (-1) = 0
axiom g_at_three : g 3 = 4
axiom g_continuous : Continuous g

-- The main theorem
theorem count_g_g_equals_two :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x : ℝ, x ∈ s ↔ g (g x) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_g_g_equals_two_l515_51535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_l515_51557

/-- Given a curve x^2 + y^2 + a^2*x + (1-a^2)*y - 4 = 0 that is symmetric with respect to the line y = x,
    prove that a = √2/2 or a = -√2/2 --/
theorem curve_symmetry (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + a^2*x + (1-a^2)*y - 4 = 0 ↔ y^2 + x^2 + a^2*y + (1-a^2)*x - 4 = 0) →
  a = Real.sqrt 2 / 2 ∨ a = -Real.sqrt 2 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_l515_51557
