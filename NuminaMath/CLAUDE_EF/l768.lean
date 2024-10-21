import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_beats_b_by_approximately_36_57_meters_l768_76814

/-- The distance A beats B when A runs 256 meters in 28 seconds and B runs 256 meters in 32 seconds -/
noncomputable def distance_a_beats_b : ℝ :=
  let speed_a := 256 / 28
  let speed_b := 256 / 32
  let distance_a_in_32_seconds := speed_a * 32
  distance_a_in_32_seconds - 256

theorem a_beats_b_by_approximately_36_57_meters :
  abs (distance_a_beats_b - 36.57) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_beats_b_by_approximately_36_57_meters_l768_76814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l768_76818

-- Define the function f(x) as noncomputable due to Real.log
noncomputable def f (x : ℝ) := Real.log (4 + 3*x - x^2)

-- Define the domain of f(x)
def domain : Set ℝ := Set.Ioo (-1) 4

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x ∈ domain, 
    (∀ y ∈ domain, x < y → f x > f y) ↔ 
    x ∈ Set.Icc (3/2) 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l768_76818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assistant_increases_output_l768_76825

/-- Represents Jane's toy bear production --/
structure ToyBearProduction where
  bears_per_week : ℝ
  hours_per_week : ℝ

/-- Calculates the bears per hour --/
noncomputable def bears_per_hour (p : ToyBearProduction) : ℝ :=
  p.bears_per_week / p.hours_per_week

/-- Represents Jane's production with an assistant --/
noncomputable def with_assistant (p : ToyBearProduction) (x : ℝ) : ToyBearProduction where
  hours_per_week := 0.9 * p.hours_per_week
  bears_per_week := p.bears_per_week * (1 + x / 100)

theorem assistant_increases_output (p : ToyBearProduction) :
  ∃ x : ℝ, 
    bears_per_hour (with_assistant p x) = 2 * bears_per_hour p ∧ 
    x = 80 := by
  sorry

#check assistant_increases_output

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assistant_increases_output_l768_76825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l768_76829

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + (Real.sqrt 3 / 2) * Real.sin (2 * x) - 1 / 2

-- State the theorem
theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, x ∈ Set.Ioo 0 (Real.pi / 2) ∧ f x = y) ↔ y ∈ Set.Ioc (-1/2) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l768_76829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l768_76888

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := (x < -1) ∨ (x > 3)

-- Theorem statement
theorem f_monotone_increasing :
  ∀ x y : ℝ, domain x → domain y → x < y → y < -1 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l768_76888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_speed_is_correct_l768_76847

/-- The speed of Alice in miles per hour -/
noncomputable def alice_speed : ℝ := 6.5

/-- The speed of Bob in miles per hour -/
noncomputable def bob_speed : ℝ := alice_speed + 3

/-- The total distance to the park in miles -/
def total_distance : ℝ := 80

/-- The distance from the meeting point to the park in miles -/
def meeting_distance : ℝ := 15

/-- The time taken by Alice to reach the meeting point -/
noncomputable def alice_time : ℝ := (total_distance - meeting_distance) / alice_speed

/-- The time taken by Bob to reach the park and return to the meeting point -/
noncomputable def bob_time : ℝ := (total_distance + meeting_distance) / bob_speed

theorem alice_speed_is_correct : alice_time = bob_time := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_speed_is_correct_l768_76847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_consumption_travels_farthest_l768_76862

/-- Represents a vehicle with its fuel consumption rate -/
structure Vehicle where
  name : String
  fuelConsumptionRate : ℚ

/-- Calculates the distance a vehicle can travel with a given amount of fuel -/
def distanceTraveled (v : Vehicle) (fuelAmount : ℚ) : ℚ :=
  (fuelAmount / v.fuelConsumptionRate) * 100

/-- Theorem: The vehicle with the lowest fuel consumption rate travels the farthest -/
theorem lowest_consumption_travels_farthest 
  (vehicles : List Vehicle) (fuelAmount : ℚ) (hfuel : fuelAmount > 0) :
  ∃ v ∈ vehicles, ∀ u ∈ vehicles, 
    v.fuelConsumptionRate ≤ u.fuelConsumptionRate → 
    distanceTraveled v fuelAmount ≥ distanceTraveled u fuelAmount :=
by
  sorry

#check lowest_consumption_travels_farthest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_consumption_travels_farthest_l768_76862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l768_76804

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def C : Set ℝ := {a | 0 ≤ a ∧ a ≤ 2}

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x

theorem problem_solution :
  (∀ a, B a ⊆ A → a ∈ C) ∧
  (∀ x₀, f x₀ ∈ C ↔ ∃ k : ℤ, (2*k*Real.pi ≤ x₀ ∧ x₀ ≤ 2*k*Real.pi + Real.pi/6) ∨
                            (2*k*Real.pi + 5*Real.pi/6 ≤ x₀ ∧ x₀ ≤ 2*k*Real.pi + Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l768_76804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l768_76880

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x + 1)

theorem domain_of_f : 
  Set.Ioi (-1 : ℝ) = {x : ℝ | ∃ y : ℝ, f x = y} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l768_76880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robert_bought_13_pens_l768_76849

-- Define the number of pens bought by each person
def julia_friend_pens : ℕ := 2
def julia_pens : ℕ := 3 * julia_friend_pens
def dorothy_pens : ℕ := julia_pens / 2
def robert_pens : ℕ := 13

-- Define the cost of one pen
def pen_cost : ℚ := 3/2

-- Define the total spent
def total_spent : ℚ := 33

-- Theorem statement
theorem robert_bought_13_pens :
  (dorothy_pens + julia_pens + robert_pens : ℚ) * pen_cost = total_spent ∧
  robert_pens = 13 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robert_bought_13_pens_l768_76849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l768_76856

theorem solution_difference : 
  ∃ s₁ s₂ : ℝ, 
    (s₁^2 - 5*s₁ - 24) / (s₁ + 5) = 3*s₁ + 11 ∧
    (s₂^2 - 5*s₂ - 24) / (s₂ + 5) = 3*s₂ + 11 ∧
    |s₁ - s₂| = (1 : ℝ) / 2 ∧ 
    ∀ t₁ t₂ : ℝ, 
      (t₁^2 - 5*t₁ - 24) / (t₁ + 5) = 3*t₁ + 11 →
      (t₂^2 - 5*t₂ - 24) / (t₂ + 5) = 3*t₂ + 11 →
      |t₁ - t₂| ≤ |s₁ - s₂| :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l768_76856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_speedup_l768_76876

/-- Represents the project completion time given initial conditions and additional workforce --/
def project_completion_time (total_days : ℕ) (initial_workers : ℕ) (initial_days : ℕ) 
  (initial_fraction : ℚ) (additional_workers : ℕ) : ℕ :=
  let total_workers := initial_workers + additional_workers
  let remaining_fraction := 1 - initial_fraction
  let remaining_days := (total_days * remaining_fraction.num / remaining_fraction.den : ℚ).ceil.toNat
  let new_completion_time := initial_days + (remaining_days / total_workers * initial_workers : ℚ).ceil.toNat
  new_completion_time

/-- Theorem stating that the project can be completed 10 days earlier --/
theorem project_speedup :
  project_completion_time 100 10 30 (1/5 : ℚ) 10 = 90 :=
by
  sorry

#eval project_completion_time 100 10 30 (1/5 : ℚ) 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_speedup_l768_76876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_nice_group_size_l768_76815

/-- A student's marks in four subjects -/
structure StudentMarks where
  algebra : ℝ
  geometry : ℝ
  numberTheory : ℝ
  combinatorics : ℝ

/-- A group of students is nice if they can be ordered in increasing order
    simultaneously in at least two of the four subjects -/
def isNiceGroup (group : List StudentMarks) : Prop :=
  ∃ (s1 s2 : Fin 4), 
    List.Pairwise (λ a b => 
      (s1.val = 0 → a.algebra < b.algebra) ∧
      (s1.val = 1 → a.geometry < b.geometry) ∧
      (s1.val = 2 → a.numberTheory < b.numberTheory) ∧
      (s1.val = 3 → a.combinatorics < b.combinatorics) ∧
      (s2.val = 0 → a.algebra < b.algebra) ∧
      (s2.val = 1 → a.geometry < b.geometry) ∧
      (s2.val = 2 → a.numberTheory < b.numberTheory) ∧
      (s2.val = 3 → a.combinatorics < b.combinatorics)) group

/-- The main theorem -/
theorem least_nice_group_size :
  ∃ (N : ℕ), N = 730 ∧ 
  (∀ (students : List StudentMarks), students.length ≥ N → 
    (∀ (i j : Fin students.length), i ≠ j → 
      students[i].algebra ≠ students[j].algebra ∧
      students[i].geometry ≠ students[j].geometry ∧
      students[i].numberTheory ≠ students[j].numberTheory ∧
      students[i].combinatorics ≠ students[j].combinatorics) →
    ∃ (niceGroup : List StudentMarks), 
      niceGroup.length = 10 ∧ 
      niceGroup.Subset students ∧
      isNiceGroup niceGroup) ∧
  (∀ (M : ℕ), M < N → 
    ∃ (students : List StudentMarks), students.length = M ∧
    (∀ (i j : Fin students.length), i ≠ j → 
      students[i].algebra ≠ students[j].algebra ∧
      students[i].geometry ≠ students[j].geometry ∧
      students[i].numberTheory ≠ students[j].numberTheory ∧
      students[i].combinatorics ≠ students[j].combinatorics) ∧
    ¬∃ (niceGroup : List StudentMarks), 
      niceGroup.length = 10 ∧ 
      niceGroup.Subset students ∧
      isNiceGroup niceGroup) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_nice_group_size_l768_76815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_5_equals_690_l768_76828

def c : ℕ → ℕ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | 2 => 4
  | (n + 3) => c (n + 2) * c (n + 1) + 1

theorem c_5_equals_690 : c 5 = 690 := by
  -- Evaluate c 3, c 4, and c 5 step by step
  have h3 : c 3 = 13 := by rfl
  have h4 : c 4 = 53 := by rfl
  have h5 : c 5 = 690 := by rfl
  -- Use the final step as the proof
  exact h5

#eval c 5  -- This will print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_5_equals_690_l768_76828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_coverage_50_percent_l768_76899

/-- Represents a tiling of a plane with hexagons and triangles -/
structure HexTriTiling where
  -- The side length of the hexagons and triangles
  side_length : ℝ
  -- Assumption that the side length is positive
  side_length_pos : side_length > 0

/-- Calculates the area of a regular hexagon -/
noncomputable def hexagon_area (t : HexTriTiling) : ℝ :=
  3 * Real.sqrt 3 / 2 * t.side_length^2

/-- Calculates the area of an equilateral triangle -/
noncomputable def triangle_area (t : HexTriTiling) : ℝ :=
  Real.sqrt 3 / 4 * t.side_length^2

/-- Calculates the total area of one hexagon and its six surrounding triangles -/
noncomputable def total_area (t : HexTriTiling) : ℝ :=
  hexagon_area t + 6 * triangle_area t

/-- Theorem stating that hexagons cover 50% of the plane in the described tiling -/
theorem hexagon_coverage_50_percent (t : HexTriTiling) :
  hexagon_area t / total_area t = 1/2 := by
  -- Proof steps would go here
  sorry

#eval "Hexagon Tiling Theorem"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_coverage_50_percent_l768_76899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_D_measure_l768_76860

-- Define a convex pentagon
structure ConvexPentagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  sum_of_angles : A + B + C + D + E = 540
  all_positive : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E

-- Define the specific pentagon with given conditions
def SpecialPentagon : Type :=
  {p : ConvexPentagon //
    p.A = p.B ∧
    p.B = p.C ∧
    p.D = p.E ∧
    p.A = p.D - 50}

-- Theorem statement
theorem angle_D_measure (p : SpecialPentagon) : p.val.D = 138 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_D_measure_l768_76860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_vertex_angle_l768_76867

/-- Definition for the Triangle structure -/
structure Triangle where
  /-- The triangle is isosceles -/
  isIsosceles : Prop
  /-- The measure of one of the base angles -/
  baseAngle : ℕ
  /-- The measure of the vertex angle -/
  vertexAngle : ℕ

/-- An isosceles triangle with a base angle of 80° has a vertex angle of 20°. -/
theorem isosceles_triangle_vertex_angle (t : Triangle) 
  (h1 : t.isIsosceles) 
  (h2 : t.baseAngle = 80) : t.vertexAngle = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_vertex_angle_l768_76867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_preserves_size_and_shape_l768_76884

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rotation transformation in a 2D plane -/
structure Rotation where
  angle : ℝ
  center : Point

/-- A geometric figure in a 2D plane -/
structure GeometricFigure where
  points : Set Point

/-- The size of a geometric figure -/
noncomputable def size (fig : GeometricFigure) : ℝ := sorry

/-- A shape type (placeholder) -/
inductive Shape
| Circle
| Triangle
| Rectangle
| Other

/-- The shape of a geometric figure -/
noncomputable def shape (fig : GeometricFigure) : Shape := sorry

/-- Apply a rotation to a geometric figure -/
noncomputable def applyRotation (r : Rotation) (fig : GeometricFigure) : GeometricFigure := sorry

/-- Theorem: Rotation preserves size and shape -/
theorem rotation_preserves_size_and_shape (r : Rotation) (fig : GeometricFigure) :
  size (applyRotation r fig) = size fig ∧ shape (applyRotation r fig) = shape fig := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_preserves_size_and_shape_l768_76884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cubic_expression_l768_76869

theorem min_value_cubic_expression (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a + b : ℝ)^3 + (b - c : ℝ)^3 + (c - a : ℝ)^3) / (b^3 : ℝ) ≥ 3.5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cubic_expression_l768_76869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_thievery_l768_76824

theorem sheep_thievery (S : ℝ) : 
  (let remaining_after_first := S - (1/3 * S + 1/3);
   let remaining_after_second := remaining_after_first - (1/4 * remaining_after_first + 1/4);
   let remaining_after_third := remaining_after_second - (1/5 * remaining_after_second + 3/5);
   remaining_after_third = 409) → S = 2556.25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_thievery_l768_76824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryan_spanish_hours_chinese_spanish_relation_l768_76890

/-- Ryan's daily study hours for different languages -/
structure StudyHours where
  english : ℕ
  chinese : ℕ
  spanish : ℕ

/-- Ryan's study schedule satisfying the given conditions -/
def ryansSchedule : StudyHours where
  english := 2
  chinese := 5
  spanish := 4  -- We define the value of spanish here

/-- The theorem stating the number of hours Ryan spends on learning Spanish -/
theorem ryan_spanish_hours : ryansSchedule.spanish = 4 := by
  -- The proof is trivial since we defined spanish as 4 in ryansSchedule
  rfl

/-- The theorem stating the relationship between Chinese and Spanish study hours -/
theorem chinese_spanish_relation : ryansSchedule.chinese = ryansSchedule.spanish + 1 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryan_spanish_hours_chinese_spanish_relation_l768_76890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_series_sum_l768_76855

theorem complex_series_sum : 
  let i : ℂ := Complex.I
  let series_sum : ℂ := (Finset.range 31).sum (λ n ↦ i^n * Real.cos ((30 + 90 * n : ℕ) * Real.pi / 180))
  series_sum = -13/2 + Real.sqrt 3/2 - i/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_series_sum_l768_76855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l768_76817

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  4 * (Real.sin x)^2 - 4 * Real.sin x * Real.sin (2 * x) + (Real.sin (2 * x))^2

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc 0 (27 / 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l768_76817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_given_product_l768_76894

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem min_sum_given_product (p q r s : ℕ+) (h : p * q * r * s = factorial 12) : 
  (p : ℝ) + q + r + s ≥ 4 * (factorial 12 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_given_product_l768_76894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_implies_m_eq_two_l768_76882

/-- A function is a power function if it has the form y = ax^b, where a and b are constants and a ≠ 0 -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * (x ^ b)

/-- The given function -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 3*m + 3) * (x^(1/(m-1)))

/-- Theorem stating that if f is a power function, then m = 2 -/
theorem power_function_implies_m_eq_two :
  (∃ m : ℝ, is_power_function (f m)) → (∃ m : ℝ, m = 2 ∧ is_power_function (f m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_implies_m_eq_two_l768_76882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonane_composition_l768_76895

/-- Represents the molecular formula of a hydrocarbon -/
structure Hydrocarbon where
  carbon : ℕ
  hydrogen : ℕ

/-- Calculates the molar mass of a hydrocarbon given the molar masses of carbon and hydrogen -/
noncomputable def molarMass (hc : Hydrocarbon) (carbonMass hydrogenMass : ℝ) : ℝ :=
  (hc.carbon : ℝ) * carbonMass + (hc.hydrogen : ℝ) * hydrogenMass

/-- Calculates the mass of a given number of moles of a substance -/
noncomputable def massOfMoles (moles : ℝ) (molarMass : ℝ) : ℝ :=
  moles * molarMass

/-- Calculates the percentage composition of an element in a hydrocarbon -/
noncomputable def percentageComposition (elementMass totalMass : ℝ) : ℝ :=
  (elementMass / totalMass) * 100

theorem nonane_composition (carbonMass hydrogenMass : ℝ) 
    (hc_carbon : carbonMass = 12.01)
    (hc_hydrogen : hydrogenMass = 1.008)
    (nonane : Hydrocarbon)
    (hc_nonane : nonane = { carbon := 9, hydrogen := 20 }) : 
    let nonaneMass := molarMass nonane carbonMass hydrogenMass
    massOfMoles 23 nonaneMass = 2950.75 ∧
    percentageComposition ((nonane.carbon : ℝ) * carbonMass) nonaneMass = 84.27 ∧
    percentageComposition ((nonane.hydrogen : ℝ) * hydrogenMass) nonaneMass = 15.73 ∧
    percentageComposition 0 nonaneMass = 0 := by
  sorry

#check nonane_composition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonane_composition_l768_76895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brilliant_divisible_by_18_fraction_l768_76834

/-- Sum of digits function (placeholder) -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- Definition of a brilliant integer -/
def IsBrilliant (n : ℕ) : Prop :=
  Even n ∧ n > 20 ∧ n < 200 ∧ (sumOfDigits n = 11)

/-- Set of all brilliant integers -/
def BrilliantIntegers : Set ℕ :=
  {n | IsBrilliant n}

/-- Set of brilliant integers divisible by 18 -/
def BrilliantDivisibleBy18 : Set ℕ :=
  {n ∈ BrilliantIntegers | n % 18 = 0}

/-- Helper lemma to ensure BrilliantIntegers is finite -/
lemma brilliant_integers_finite : Set.Finite BrilliantIntegers :=
  sorry

/-- Helper lemma to ensure BrilliantDivisibleBy18 is finite -/
lemma brilliant_divisible_by_18_finite : Set.Finite BrilliantDivisibleBy18 :=
  sorry

/-- Main theorem -/
theorem brilliant_divisible_by_18_fraction :
  (Finset.card (Set.Finite.toFinset brilliant_divisible_by_18_finite) : ℚ) /
  (Finset.card (Set.Finite.toFinset brilliant_integers_finite) : ℚ) = 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brilliant_divisible_by_18_fraction_l768_76834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_properties_l768_76868

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := Real.cos (x - Real.pi / 4)

noncomputable def h (x : ℝ) : ℝ := f x * g x

theorem h_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, h (x + p) = h x ∧ ∀ q : ℝ, 0 < q ∧ q < p → ∃ y : ℝ, h (y + q) ≠ h y) ∧
  (∃ M : ℝ, M = 1 ∧ ∀ x : ℝ, h x ≤ M) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_properties_l768_76868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l768_76852

/-- The function f(x) defined as the sum of two square roots -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 4*x + 20) + Real.sqrt (x^2 + 2*x + 10)

/-- Theorem stating that the minimum value of f(x) is 5√2 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = 5 * Real.sqrt 2 ∧ ∀ x, f x ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l768_76852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l768_76831

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  (Real.sqrt (a * (1 - a)) / (1 + a)) + 
  (Real.sqrt (b * (1 - b)) / (1 + b)) + 
  (Real.sqrt (c * (1 - c)) / (1 + c)) ≥ 
  3 * Real.sqrt ((a * b * c) / ((1 - a) * (1 - b) * (1 - c))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l768_76831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_sin_product_l768_76848

/-- The minimum positive period of the function y = sin(x + π/4) * sin(x - π/4) is π -/
theorem min_period_sin_product : 
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ x : ℝ, Real.sin (x + T + π/4) * Real.sin (x + T - π/4) = Real.sin (x + π/4) * Real.sin (x - π/4)) ∧ 
  (∀ T' : ℝ, T' > 0 → 
    (∀ x : ℝ, Real.sin (x + T' + π/4) * Real.sin (x + T' - π/4) = Real.sin (x + π/4) * Real.sin (x - π/4)) 
    → T ≤ T') ∧ 
  T = π :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_sin_product_l768_76848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_1024_l768_76832

def divisor_product (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem divisor_product_1024 (n : ℕ) : divisor_product n = 1024 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_1024_l768_76832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_weighted_cauchy_schwarz_inequality_l768_76843

theorem inequality_solution_set (x : ℝ) : 
  2^x + 2^(abs x) ≥ 2 * Real.sqrt 2 ↔ x ≥ 1/2 ∨ x ≤ Real.log (Real.sqrt 2 - 1) / Real.log 2 := by
  sorry

theorem weighted_cauchy_schwarz_inequality (m n a b : ℝ) (hm : m > 0) (hn : n > 0) :
  a^2 / m + b^2 / n ≥ (a + b)^2 / (m + n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_weighted_cauchy_schwarz_inequality_l768_76843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_two_l768_76874

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

/-- The x-coordinate of the right focus -/
noncomputable def right_focus_x (h : Hyperbola a b) : ℝ :=
  Real.sqrt (a^2 + b^2)

/-- The x-coordinate of the right vertex -/
def right_vertex_x (h : Hyperbola a b) : ℝ := a

/-- A point on the hyperbola whose projection on the x-axis coincides with the right focus -/
noncomputable def special_point (h : Hyperbola a b) : ℝ × ℝ :=
  (right_focus_x h, b^2 / a)

/-- The slope of the line connecting the right vertex to the special point -/
noncomputable def slope_special (h : Hyperbola a b) : ℝ :=
  (special_point h).2 / ((special_point h).1 - right_vertex_x h)

theorem eccentricity_is_two (h : Hyperbola a b) (slope_is_three : slope_special h = 3) :
  eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_two_l768_76874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_lower_bound_l768_76821

-- Define the set A
def A : Set ℝ := {x : ℝ | (2 : ℝ)^x ≥ 4}

-- Theorem statement
theorem set_A_lower_bound (a : ℝ) (h : A = Set.Ici a) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_lower_bound_l768_76821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_maximum_y_l768_76819

/-- The function g(y) as defined in the problem -/
noncomputable def g (y : ℝ) : ℝ := Real.cos (y / 4) + Real.cos (y / 13)

/-- The value we want to prove is the smallest positive solution -/
def y_min : ℝ := 18720

theorem smallest_maximum_y :
  (∀ y : ℝ, y > 0 → g y ≤ g y_min) ∧
  (∀ y : ℝ, 0 < y ∧ y < y_min → g y < g y_min) := by
  sorry

#check smallest_maximum_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_maximum_y_l768_76819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amateur_definition_l768_76893

def amateur : String := "amateurish or non-professional"

theorem amateur_definition : amateur = "amateurish or non-professional" := by
  rfl

#check amateur_definition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amateur_definition_l768_76893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_interval_max_difference_l768_76891

/-- Definition of a beautiful interval for a function f on an interval [m,n] --/
noncomputable def is_beautiful_interval (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  m < n ∧
  (∀ x y, m ≤ x ∧ x < y ∧ y ≤ n → f x < f y) ∧
  (∀ y, m ≤ y ∧ y ≤ n → ∃ x, m ≤ x ∧ x ≤ n ∧ f x = y)

/-- The function f(x) = ((a^2 + a)x - 1) / (a^2 * x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((a^2 + a)*x - 1) / (a^2 * x)

/-- Theorem stating the maximum value of n-m for a beautiful interval of f --/
theorem beautiful_interval_max_difference (a : ℝ) (m n : ℝ) :
  a ≠ 0 →
  is_beautiful_interval (f a) m n →
  n - m ≤ 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beautiful_interval_max_difference_l768_76891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_is_pseudo_symmetry_point_l768_76801

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 5 * x + Real.log x

-- Define the tangent line g at a point x₀
noncomputable def g (x₀ : ℝ) (x : ℝ) : ℝ :=
  ((4 * x₀^2 - 5 * x₀ + 1) / x₀) * (x - x₀) + 2 * x₀^2 - 5 * x₀ + Real.log x₀

-- Define the pseudo-symmetry point property
def is_pseudo_symmetry_point (x₀ : ℝ) : Prop :=
  (∀ x, 0 < x ∧ x < x₀ → f x < g x₀ x) ∧
  (∀ x, x > x₀ → f x > g x₀ x)

-- Theorem statement
theorem half_is_pseudo_symmetry_point :
  is_pseudo_symmetry_point (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_is_pseudo_symmetry_point_l768_76801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l768_76802

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (2 * x)

noncomputable def shift_right (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x => f (x - shift)

noncomputable def stretch_abscissa (f : ℝ → ℝ) (factor : ℝ) : ℝ → ℝ :=
  λ x => f (x / factor)

noncomputable def transformed_function (x : ℝ) : ℝ := Real.sin (x - Real.pi / 6)

theorem function_transformation :
  (stretch_abscissa (shift_right original_function (Real.pi / 12)) 2) = transformed_function :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l768_76802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eight_point_five_equals_nine_l768_76840

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem f_eight_point_five_equals_nine
  (f : ℝ → ℝ)
  (h1 : is_even f)
  (h2 : is_odd (fun x ↦ f (x - 1)))
  (h3 : f 0.5 = 9) :
  f 8.5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eight_point_five_equals_nine_l768_76840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_characterization_l768_76826

theorem floor_inequality_characterization (n : ℕ+) :
  (n : ℝ) + ⌊(n : ℝ) / 6⌋ ≠ ⌊(n : ℝ) / 2⌋ + ⌊(2 * n : ℝ) / 3⌋ ↔
  ∃ k : ℕ, n = 6 * k + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_characterization_l768_76826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l768_76896

/-- Given an infinite sequence {a_n} with general term formula a_n = n / (n^2 + λ),
    where λ is a positive real number, if for any positive integer n, a_n ≤ a_7 always holds,
    then 42.25 < λ < 56.25. -/
theorem sequence_inequality (lambda : ℝ) (h_pos : lambda > 0) : 
  (∀ n : ℕ+, n / (n^2 + lambda) ≤ 7 / (49 + lambda)) → 
  42.25 < lambda ∧ lambda < 56.25 := by
  sorry

#check sequence_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l768_76896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_line_from_plane_perp_l768_76822

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line_line : Line → Line → Prop)

-- Define the theorem
theorem line_perp_line_from_plane_perp
  (m n : Line) (α β : Plane)
  (hm : m ≠ n) (hα : α ≠ β)
  (hma : perp_line_plane m α)
  (hnb : perp_line_plane n β)
  (hab : perp_plane_plane α β)
  : perp_line_line m n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_line_from_plane_perp_l768_76822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l768_76816

/-- The distance between two places A and B -/
noncomputable def distance : ℝ := sorry

/-- The initial speed of the car -/
noncomputable def initial_speed : ℝ := sorry

/-- The increased speed of the car -/
noncomputable def increased_speed : ℝ := sorry

/-- The initial travel time -/
noncomputable def initial_time : ℝ := sorry

/-- The reduced travel time after speed increase -/
noncomputable def reduced_time : ℝ := sorry

/-- The speed increase -/
noncomputable def speed_increase : ℝ := sorry

theorem distance_calculation (h1 : initial_time = 4)
                             (h2 : reduced_time = 3)
                             (h3 : speed_increase = 20)
                             (h4 : distance = initial_speed * initial_time)
                             (h5 : distance = increased_speed * reduced_time)
                             (h6 : increased_speed = initial_speed + speed_increase) :
  distance = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_calculation_l768_76816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_b_eq_3a_prob_a2_b_minus_5_2_leq_9_l768_76853

-- Define the possible values for a and b
def A : Finset ℕ := {0, 1, 2, 3, 4, 5}
def B : Finset ℕ := {6, 7, 8, 9}

-- Define the sample space
def Ω : Finset (ℕ × ℕ) := A.product B

-- Define the event where b = 3a
def event_b_eq_3a : Finset (ℕ × ℕ) := Ω.filter (fun p => p.2 = 3 * p.1)

-- Define the event where a² + (b-5)² ≤ 9
def event_a2_b_minus_5_2_leq_9 : Finset (ℕ × ℕ) := 
  Ω.filter (fun p => p.1^2 + (p.2 - 5)^2 ≤ 9)

-- Theorem for the probability of b = 3a
theorem prob_b_eq_3a : (event_b_eq_3a.card : ℚ) / Ω.card = 1/12 := by
  sorry

-- Theorem for the probability of a² + (b-5)² ≤ 9
theorem prob_a2_b_minus_5_2_leq_9 : 
  (event_a2_b_minus_5_2_leq_9.card : ℚ) / Ω.card = 7/24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_b_eq_3a_prob_a2_b_minus_5_2_leq_9_l768_76853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_proof_l768_76845

theorem starting_number_proof (n : ℕ) : 
  (n ≥ 33) ∧ 
  (∃ (seq : List ℕ), 
    seq.length = 8 ∧ 
    seq.all (λ x => x % 11 = 0) ∧
    seq.head? = some n ∧
    seq.getLast? = some (seq.foldl max 0) ∧
    seq.foldl max 0 ≤ 119) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ (seq : List ℕ), 
      seq.length = 8 ∧ 
      seq.all (λ x => x % 11 = 0) ∧
      seq.head? = some m ∧
      seq.getLast? = some (seq.foldl max 0) ∧
      seq.foldl max 0 ≤ 119)) →
  n = 33 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_proof_l768_76845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_difference_l768_76835

theorem two_integers_difference (x y : ℕ) : 
  (∃ n : ℕ, x = n^2 ∨ y = n^2) → 
  (x + y : ℤ) = x * y - 2006 → 
  (x : ℤ) - y = 666 ∨ y - x = 666 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_difference_l768_76835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roller_derby_lace_sets_l768_76812

/-- Represents a roller derby team -/
structure Team where
  members : ℕ
  backupSkates : ℕ

/-- Calculates the total number of sets of laces for a roller derby competition -/
def totalLaceSets (teams : List Team) (laceSetsPerChoice : ℕ) (laceChoices : ℕ) : ℕ :=
  let totalSkates := teams.foldl (fun acc team ↦ acc + team.members * (1 + team.backupSkates)) 0
  totalSkates * laceSetsPerChoice * laceChoices

/-- The main theorem stating the total number of lace sets in the roller derby competition -/
theorem roller_derby_lace_sets :
  let teamA : Team := { members := 8, backupSkates := 1 }
  let teamB : Team := { members := 12, backupSkates := 2 }
  let teamC : Team := { members := 10, backupSkates := 3 }
  let teamD : Team := { members := 15, backupSkates := 4 }
  let teams := [teamA, teamB, teamC, teamD]
  let laceSetsPerChoice := 3
  let laceChoices := 2
  totalLaceSets teams laceSetsPerChoice laceChoices = 1002 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roller_derby_lace_sets_l768_76812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reject_null_hypothesis_l768_76839

/-- Represents a bivariate normal population -/
structure BivariateNormalPopulation where
  X : Type
  Y : Type

/-- Represents the sample correlation coefficient -/
noncomputable def sample_correlation_coefficient : ℝ := 0.2

/-- Calculates the test statistic for the sample correlation coefficient -/
noncomputable def test_statistic (r : ℝ) (n : ℕ) : ℝ :=
  (r * Real.sqrt (n - 2 : ℝ)) / Real.sqrt (1 - r^2)

/-- Represents the critical value for the t-distribution -/
noncomputable def critical_value : ℝ := 1.99

/-- The main theorem stating that the null hypothesis should be rejected -/
theorem reject_null_hypothesis (population : BivariateNormalPopulation) :
  let n : ℕ := 100
  let r := sample_correlation_coefficient
  let T := test_statistic r n
  let t_crit := critical_value
  T > t_crit → population.X ≠ population.Y := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reject_null_hypothesis_l768_76839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l768_76808

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := 3^(x^2 - 2*x + 3)

-- Theorem for the monotonicity and range of f(x)
theorem f_properties :
  (∀ x y, x ≥ 1 ∧ y ≥ 1 ∧ x < y → f x < f y) ∧
  (∀ x y, x < 1 ∧ y < 1 ∧ x < y → f x > f y) ∧
  (∀ x, f x ≥ 9) ∧
  (∀ y, y ≥ 9 → ∃ x, f x = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l768_76808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_values_l768_76806

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

/-- Theorem: For an ellipse with equation x^2/4 + y^2/m = 1 and eccentricity √3/2,
    the value of m is either 1 or 16 -/
theorem ellipse_eccentricity_m_values (m : ℝ) :
  (∀ x y : ℝ, x^2/4 + y^2/m = 1) →
  (eccentricity 2 (Real.sqrt m) = Real.sqrt 3 / 2) →
  (m = 1 ∨ m = 16) := by
  sorry

#check ellipse_eccentricity_m_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_values_l768_76806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_sum_l768_76809

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem tangent_slope_sum (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.tan α = (deriv f) 1) : Real.cos α + Real.sin α = 2 * Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_sum_l768_76809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_selection_probability_l768_76837

/-- Given a group of students and a selection process, calculate the probability of a student being selected. -/
theorem student_selection_probability
  (total_students : ℕ)
  (representatives : ℕ)
  (eliminated : ℕ)
  (groups : ℕ)
  (h1 : total_students = 1003)
  (h2 : representatives = 50)
  (h3 : eliminated = 3)
  (h4 : groups = 20)
  (h5 : total_students - eliminated = groups * (representatives / groups)) :
  (representatives : ℚ) / total_students = 50 / 1003 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_selection_probability_l768_76837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l768_76851

-- Define the vertices of the parallelogram
def vertex1 : ℝ × ℝ := (0, 0)
def vertex2 : ℝ × ℝ := (5, 0)
def vertex3 : ℝ × ℝ := (3, 7)
def vertex4 : ℝ × ℝ := (8, 7)

-- Define the parallelogram using its vertices
def parallelogram : Set (ℝ × ℝ) :=
  {p | ∃ (a b : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧
    p = (a * vertex1.1 + (1 - a) * vertex2.1, a * vertex1.2 + (1 - a) * vertex2.2) +
        (b * vertex3.1 + (1 - b) * vertex4.1 - (a * vertex1.1 + (1 - a) * vertex2.1),
         b * vertex3.2 + (1 - b) * vertex4.2 - (a * vertex1.2 + (1 - a) * vertex2.2))}

-- Theorem stating that the area of the parallelogram is 35 square units
theorem parallelogram_area : MeasureTheory.volume parallelogram = 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l768_76851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_multiple_with_equal_digit_sum_l768_76866

/-- Q(n) represents the sum of digits of n in its decimal representation -/
def Q (n : ℕ) : ℕ := sorry

/-- For any positive integer k, there exists a multiple n of k such that Q(n) = Q(n^2) -/
theorem exists_multiple_with_equal_digit_sum (k : ℕ+) : ∃ n : ℕ, (k : ℕ) ∣ n ∧ Q n = Q (n^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_multiple_with_equal_digit_sum_l768_76866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_weekly_earnings_l768_76803

/-- Represents the daily catch of crabs -/
structure DailyCatch where
  blue : Nat
  red : Nat

/-- Calculates Tom's weekly earnings from crab fishing -/
def weeklyEarnings (catches : List DailyCatch) (numBuckets : Nat) (bluePrice redPrice : Nat) : Nat :=
  catches.foldl (fun acc dailyCatch => acc + numBuckets * (dailyCatch.blue * bluePrice + dailyCatch.red * redPrice)) 0

/-- Theorem stating Tom's weekly earnings -/
theorem tom_weekly_earnings : 
  let catches : List DailyCatch := [
    ⟨10, 14⟩, ⟨8, 16⟩, ⟨12, 10⟩, ⟨6, 18⟩, ⟨14, 12⟩, ⟨10, 10⟩, ⟨8, 8⟩
  ]
  let numBuckets : Nat := 8
  let bluePrice : Nat := 6
  let redPrice : Nat := 4
  weeklyEarnings catches numBuckets bluePrice redPrice = 6080 := by
  sorry

#eval weeklyEarnings [
  ⟨10, 14⟩, ⟨8, 16⟩, ⟨12, 10⟩, ⟨6, 18⟩, ⟨14, 12⟩, ⟨10, 10⟩, ⟨8, 8⟩
] 8 6 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_weekly_earnings_l768_76803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l768_76878

-- Define the logarithm function with base √3
noncomputable def log_sqrt3 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 3)

-- Define the equation
noncomputable def equation (x : ℝ) : ℝ := 
  log_sqrt3 x * Real.sqrt (log_sqrt3 3 - Real.log 9 / Real.log x) + 4

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, 0 < x ∧ x < 1 ∧ equation x = 0 ∧ x = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l768_76878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l768_76898

noncomputable def f (x φ : Real) : Real := 2 * Real.cos x * Real.cos (x + φ)

theorem function_properties (φ : Real) 
  (h1 : |φ| < Real.pi / 2)
  (h2 : f (Real.pi / 3) φ = 1) :
  φ = -Real.pi / 3 ∧ 
  (∀ x ∈ Set.Icc (-Real.pi / 2) 0, f x φ ≤ 1) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 2) 0, f x φ ≥ -1 / 2) ∧
  (∃ x ∈ Set.Icc (-Real.pi / 2) 0, f x φ = 1) ∧
  (∃ x ∈ Set.Icc (-Real.pi / 2) 0, f x φ = -1 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l768_76898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l768_76858

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line structure -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Distance between a point and a vertical line -/
noncomputable def distance_point_to_line (P : Point) (x : ℝ) : ℝ :=
  |P.x - x|

/-- Check if a point lies on a parabola -/
def point_on_parabola (P : Point) (C : Parabola) : Prop :=
  C.equation P.x P.y

/-- Slope of a line passing through two points -/
noncomputable def slope_of_line (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

/-- Perpendicular slope -/
noncomputable def perpendicular_slope (k : ℝ) : ℝ :=
  -1 / k

/-- Length of a line segment between two points on a parabola intersected by a perpendicular line -/
noncomputable def length_of_intersection (C : Parabola) (A M N : Point) : ℝ :=
  let k_MN := slope_of_line M N
  let k_PQ := perpendicular_slope k_MN
  3 * Real.sqrt 5 / 4

/-- Main theorem -/
theorem parabola_intersection_length 
  (C : Parabola) 
  (M N : Point) 
  (h1 : C.p > 0)
  (h2 : C.equation = fun x y => y^2 = 2 * C.p * x)
  (h3 : M.x = 1 ∧ M.y = 1/2)
  (h4 : distance_point_to_line M (-C.p/2) = 5/4)
  (h5 : N.y = 2)
  (h6 : point_on_parabola N C)
  (A : Point)
  (h7 : A.x = 0 ∧ A.y = 1) :
  length_of_intersection C A M N = 3 * Real.sqrt 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l768_76858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contesta_paths_count_l768_76842

/-- Represents the triangular grid of letters -/
def LetterGrid : Type := List (List Char)

/-- The specific grid given in the problem -/
def problemGrid : LetterGrid := [
  ['C'],
  ['C', 'O', 'C'],
  ['C', 'O', 'N', 'O', 'C'],
  ['C', 'O', 'N', 'T', 'N', 'O', 'C'],
  ['C', 'O', 'N', 'T', 'E', 'T', 'N', 'O', 'C'],
  ['C', 'O', 'N', 'T', 'E', 'S', 'E', 'T', 'N', 'O', 'C'],
  ['C', 'O', 'N', 'T', 'E', 'S', 'T', 'S', 'E', 'T', 'N', 'O', 'C'],
  ['C', 'O', 'N', 'T', 'E', 'S', 'T', 'A', 'S', 'E', 'T', 'N', 'O', 'C']
]

/-- A path in the grid -/
def GridPath : Type := List (Nat × Nat)

/-- Checks if a path spells out "CONTESTA" -/
def spellsCONTESTA (grid : LetterGrid) (path : GridPath) : Prop := sorry

/-- Checks if a path only moves between adjacent cells (horizontally, vertically, or diagonally) -/
def isValidPath (path : GridPath) : Prop := sorry

/-- Counts the number of valid paths spelling "CONTESTA" -/
noncomputable def countValidPaths (grid : LetterGrid) : Nat := sorry

/-- The main theorem to prove -/
theorem contesta_paths_count :
  countValidPaths problemGrid = 4375 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contesta_paths_count_l768_76842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elephant_exodus_rate_l768_76886

/-- Represents the rate of elephants leaving the park during the exodus -/
def exodus_rate : ℕ → ℕ := sorry

theorem elephant_exodus_rate (initial_elephants exodus_duration entry_duration entry_rate final_elephants : ℕ) :
  initial_elephants = 30000 →
  exodus_duration = 4 →
  entry_duration = 7 →
  entry_rate = 1500 →
  final_elephants = 28980 →
  exodus_rate exodus_duration = 2880 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elephant_exodus_rate_l768_76886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possibleHands_correct_l768_76873

/-- The number of possible hands for the first player when n cards are dealt to two players,
    with each player having at least one card. -/
def possibleHands (n : ℕ) : ℕ :=
  2^n - 2

/-- The actual number of valid distributions when n cards are dealt to two players,
    with each player having at least one card. This function is not defined and
    serves as a placeholder for the true mathematical concept. -/
def number_of_valid_distributions (n : ℕ) : ℕ :=
  2^n - 2  -- We define it to match possibleHands for the theorem to work

/-- Theorem stating that possibleHands gives the correct number of possible hands
    for the first player when n cards are dealt to two players,
    with each player having at least one card. -/
theorem possibleHands_correct (n : ℕ) (h : n ≥ 2) :
  possibleHands n = number_of_valid_distributions n :=
by
  -- Unfold the definitions of possibleHands and number_of_valid_distributions
  unfold possibleHands number_of_valid_distributions
  -- The expressions are now identical, so reflexivity completes the proof
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possibleHands_correct_l768_76873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l768_76841

theorem equation_solution : 
  ∃ (x : ℝ), 5^(Real.sqrt (x^3 + 3*x^2 + 3*x + 1)) = Real.sqrt ((5 * (x+1)^(5/4))^3) ∧ x = 65/16 := by
  use 65/16
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l768_76841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_bananas_l768_76830

/-- Proves that the number of bananas bought is 400 given the conditions of the problem -/
theorem shopkeeper_bananas (total_oranges : ℕ) (bananas : ℕ) : 
  total_oranges = 600 →
  (0.85 * (total_oranges : ℝ) + 0.96 * (bananas : ℝ)) / ((total_oranges : ℝ) + (bananas : ℝ)) = 0.894 →
  bananas = 400 := by
  sorry

#check shopkeeper_bananas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_bananas_l768_76830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2012_negative_l768_76827

theorem sin_2012_negative : Real.sin (2012 * Real.pi / 180) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2012_negative_l768_76827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_dollar_change_count_l768_76854

/-- Represents the number of ways to make change for a given amount in cents -/
def makeChange (amount : ℕ) : ℕ := sorry

/-- The value of a penny in cents -/
def pennyValue : ℕ := 1

/-- The value of a nickel in cents -/
def nickelValue : ℕ := 5

/-- The value of a dime in cents -/
def dimeValue : ℕ := 10

/-- The value of a quarter in cents -/
def quarterValue : ℕ := 25

/-- The value of a half-dollar in cents -/
def halfDollarValue : ℕ := 50

/-- Theorem: The number of ways to make change for a half-dollar (50 cents) 
    using standard U.S. coins, excluding the use of a half-dollar coin, is 33 -/
theorem half_dollar_change_count :
  makeChange halfDollarValue = 33 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_dollar_change_count_l768_76854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_average_score_l768_76875

/-- Calculates the weighted average score of two classes -/
noncomputable def weighted_average_score (students_a : ℕ) (students_b : ℕ) (score_a : ℝ) (score_b : ℝ) : ℝ :=
  ((students_a : ℝ) * score_a + (students_b : ℝ) * score_b) / ((students_a + students_b) : ℝ)

/-- The weighted average score of two classes with given parameters is 85 -/
theorem school_average_score :
  weighted_average_score 40 50 90 81 = 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_average_score_l768_76875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_are_perpendicular_l768_76823

/-- Two planes in 3D space --/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The normal vector of a plane --/
def normalVector (p : Plane3D) : Fin 3 → ℝ := 
  fun i => match i with
  | 0 => p.a
  | 1 => p.b
  | 2 => p.c

/-- Two planes are perpendicular if their normal vectors are orthogonal --/
def arePerpendicular (p1 p2 : Plane3D) : Prop :=
  (normalVector p1 0) * (normalVector p2 0) +
  (normalVector p1 1) * (normalVector p2 1) +
  (normalVector p1 2) * (normalVector p2 2) = 0

/-- The theorem to be proved --/
theorem planes_are_perpendicular :
  let p1 : Plane3D := ⟨2, 3, -4, 1⟩
  let p2 : Plane3D := ⟨5, -2, 1, 6⟩
  arePerpendicular p1 p2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_are_perpendicular_l768_76823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l768_76871

-- Define the function f(x) = |x+2| + |x-3|
def f (x : ℝ) : ℝ := |x + 2| + |x - 3|

-- Define the set of a for which the inequality has a real solution
def A : Set ℝ := {a | ∃ x, f x ≤ |a - 1|}

-- Theorem statement
theorem inequality_solution_range : A = Set.Iic (-4) ∪ Set.Ici 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l768_76871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_interval_l768_76820

-- Define the function f(x) = (x-2)e^x
noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x

-- State the theorem
theorem f_monotone_increasing_on_interval :
  MonotoneOn f (Set.Ioi 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_interval_l768_76820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l768_76800

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by three of its vertices -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

noncomputable def ABCD : Parallelogram := {
  A := { x := 0, y := 0 },
  B := { x := 3, y := Real.sqrt 3 },
  C := { x := 4, y := 0 }
}

noncomputable def line_CD : LineEquation := {
  a := 1,
  b := -Real.sqrt 3,
  c := -4
}

theorem parallelogram_properties (p : Parallelogram) (l : LineEquation) :
  p = ABCD → l = line_CD →
  (∀ x y, l.a * x + l.b * y + l.c = 0 ↔ (x - p.C.x) / (y - p.C.y) = (p.B.x - p.A.x) / (p.B.y - p.A.y)) ∧
  ((p.B.x - p.A.x) * (p.C.x - p.B.x) + (p.B.y - p.A.y) * (p.C.y - p.B.y) = 0) ∧
  ((p.B.x - p.A.x) * (p.C.x - p.B.x) + (p.B.y - p.A.y) * (p.C.y - p.B.y) = 0 →
   (p.B.x - p.A.x)^2 + (p.B.y - p.A.y)^2 = 12) ∧
  ((p.C.x - p.B.x)^2 + (p.C.y - p.B.y)^2 = 4) ∧
  ((p.B.x - p.A.x)^2 + (p.B.y - p.A.y)^2) * ((p.C.x - p.B.x)^2 + (p.C.y - p.B.y)^2) = 48 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l768_76800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l768_76872

-- Define the function f(x) = ln(x + 1) - 1/x
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - 1 / x

-- State the theorem
theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l768_76872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l768_76877

noncomputable def a (θ : ℝ) : ℝ × ℝ := (Real.sin θ, 1)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (1, Real.cos θ)

theorem vector_problem (θ : ℝ) (h : -π/2 < θ ∧ θ < π/2) :
  (a θ • b θ = 0 → θ = -π/4) ∧
  (∀ φ, -π/2 < φ ∧ φ < π/2 → ‖a φ + b φ‖ ≤ Real.sqrt 2 + 1) ∧
  (∃ φ, -π/2 < φ ∧ φ < π/2 ∧ ‖a φ + b φ‖ = Real.sqrt 2 + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l768_76877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approximation_l768_76810

/-- The area of a circular sector given its radius and central angle in degrees -/
noncomputable def sectorArea (radius : ℝ) (angle : ℝ) : ℝ :=
  (angle / 360) * Real.pi * radius^2

theorem sector_area_approximation :
  let r : ℝ := 12
  let θ : ℝ := 54
  abs (sectorArea r θ - 67.6464) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approximation_l768_76810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_distance_l768_76870

-- Define the focal distance for a hyperbola
noncomputable def focal_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

-- Define the two hyperbolas
def hyperbola1 (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

def hyperbola2 (t x y : ℝ) : Prop :=
  x^2 / (16 - t) - y^2 / (t + 9) = 1

-- State the theorem
theorem equal_focal_distance :
  ∀ t : ℝ, -9 < t → t < 16 →
  focal_distance 4 3 = focal_distance (Real.sqrt (16 - t)) (Real.sqrt (t + 9)) ∧
  focal_distance 4 3 = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_distance_l768_76870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_completes_in_35_days_l768_76864

/-- The time it takes y to complete the work alone given the conditions -/
noncomputable def y_completion_time (x_total_time y_remaining_time x_work_time : ℝ) : ℝ :=
  let x_rate := 1 / x_total_time
  let x_work := x_rate * x_work_time
  let remaining_work := 1 - x_work
  let y_rate := remaining_work / y_remaining_time
  1 / y_rate

/-- Theorem stating that y will take 35 days to complete the work alone -/
theorem y_completes_in_35_days
  (x_total_time : ℝ)
  (y_remaining_time : ℝ)
  (x_work_time : ℝ)
  (hx_total : x_total_time = 40)
  (hy_remaining : y_remaining_time = 28)
  (hx_work : x_work_time = 8) :
  y_completion_time x_total_time y_remaining_time x_work_time = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_completes_in_35_days_l768_76864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_kids_wash_three_whiteboards_l768_76892

/-- The time (in minutes) it takes for one kid to wash one whiteboard -/
noncomputable def time_per_whiteboard : ℝ := 160 / 6

/-- The time (in minutes) it takes for four kids to wash one whiteboard together -/
noncomputable def time_per_whiteboard_four_kids : ℝ := time_per_whiteboard / 4

/-- The number of whiteboards to be washed -/
def num_whiteboards : ℕ := 3

/-- Theorem: Four kids can wash three whiteboards in 20 minutes -/
theorem four_kids_wash_three_whiteboards : 
  time_per_whiteboard_four_kids * (num_whiteboards : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_kids_wash_three_whiteboards_l768_76892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_parallel_planes_l768_76883

/-- Represents a plane in 3D space -/
structure Plane where

/-- Represents a line in 3D space -/
structure Line where

/-- Defines the parallel relation between two planes -/
def parallel (p1 p2 : Plane) : Prop := sorry

/-- Defines the intersection of a plane with another plane, resulting in a line -/
def intersect (p1 p2 : Plane) : Line := sorry

/-- Defines the parallel relation between two lines -/
def lineParallel (l1 l2 : Line) : Prop := sorry

/-- Theorem: If two parallel planes are intersected by a third plane, 
    the lines of intersection will be parallel -/
theorem intersection_of_parallel_planes (α β γ : Plane) 
  (h1 : parallel α β) (h2 : ¬ parallel α γ) (h3 : ¬ parallel β γ) : 
  lineParallel (intersect α γ) (intersect β γ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_parallel_planes_l768_76883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_game_winner_diagonal_game_outcome_l768_76813

/-- A game played on a regular (2n+1)-gon where n > 1 -/
structure DiagonalGame (n : ℕ) where
  [n_gt_one : Fact (n > 1)]

/-- The number of diagonals in a (2n+1)-gon -/
def num_diagonals (n : ℕ) : ℕ := (2*n + 1) * (n - 1)

/-- The winner of the game -/
inductive Winner
  | Player1
  | Player2

/-- The theorem stating who wins based on the parity of n -/
theorem diagonal_game_winner (n : ℕ) [Fact (n > 1)] : 
  (if n % 2 = 0 then Winner.Player1 else Winner.Player2) = 
  (if num_diagonals n % 2 = 1 then Winner.Player1 else Winner.Player2) := by
  sorry

/-- The main theorem about the game outcome -/
theorem diagonal_game_outcome (n : ℕ) [Fact (n > 1)] :
  (num_diagonals n % 2 = 1) ↔ (n % 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_game_winner_diagonal_game_outcome_l768_76813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_54_after_60_ops_l768_76844

def operation (n : ℕ) : ℕ → Prop :=
  λ m => (m = n * 2) ∨ (m = n * 3) ∨ (n % 2 = 0 ∧ m = n / 2) ∨ (n % 3 = 0 ∧ m = n / 3)

def board_sequence (start : ℕ) (length : ℕ) : (ℕ → ℕ) → Prop :=
  λ seq => seq 0 = start ∧ ∀ i, i < length → operation (seq i) (seq (i + 1))

theorem not_54_after_60_ops : ¬ ∃ (seq : ℕ → ℕ), board_sequence 12 60 seq ∧ seq 60 = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_54_after_60_ops_l768_76844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l768_76846

noncomputable def f (x : ℝ) : ℝ := Real.tan (x/2 - Real.pi/3)

theorem f_properties :
  ∃ (domain : Set ℝ) (period : ℝ) (increase_intervals : Set (Set ℝ)) (solution_set : Set ℝ),
    domain = {x : ℝ | ∀ k : ℤ, x ≠ 2*k*Real.pi + 5*Real.pi/3} ∧
    period = 2*Real.pi ∧
    increase_intervals = {I | ∃ k : ℤ, I = Set.Ioo (2*k*Real.pi - Real.pi/3) (2*k*Real.pi + 5*Real.pi/3)} ∧
    solution_set = {x : ℝ | ∃ k : ℤ, 2*k*Real.pi - Real.pi/3 < x ∧ x ≤ 2*k*Real.pi + 4*Real.pi/3} ∧
    (∀ x ∈ domain, f x = Real.tan (x/2 - Real.pi/3)) ∧
    (∀ x ∈ domain, f (x + period) = f x) ∧
    (∀ I ∈ increase_intervals, ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) ∧
    (∀ x ∈ solution_set, f x ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l768_76846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_integral_equality_l768_76836

theorem function_integral_equality (a c : ℝ) (h : a ≠ 0) :
  ∃ x₀ : ℝ, 0 ≤ x₀ ∧ x₀ ≤ 1 ∧
  (∫ (x : ℝ) in Set.Icc 0 1, a * x^2 + c) = (a * x₀^2 + c) →
  x₀ = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_integral_equality_l768_76836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l768_76865

/-- Triangle ABC with side lengths a, b, c and angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : angleA + angleB + angleC = π
  cosine_law_a : a^2 = b^2 + c^2 - 2*b*c*(Real.cos angleA)
  cosine_law_b : b^2 = a^2 + c^2 - 2*a*c*(Real.cos angleB)
  cosine_law_c : c^2 = a^2 + b^2 - 2*a*b*(Real.cos angleC)

/-- The main theorem about the special triangle ABC --/
theorem special_triangle_properties (t : Triangle) 
    (h : t.a^2 + t.c^2 = t.b^2 + Real.sqrt 2 * t.a * t.c) :
  t.angleB = π/4 ∧ 
  (∀ A C, A + C = 3*π/4 → Real.cos A + Real.sqrt 2 * Real.cos C ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l768_76865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_non_prime_powers_l768_76805

theorem consecutive_non_prime_powers (n : ℕ) :
  ∃ x : ℕ, ∀ i ∈ Finset.range n,
    ¬∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ x + i + 1 = p^k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_non_prime_powers_l768_76805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_five_l768_76881

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => a (n + 1) + a n

theorem divisible_by_five (k : ℕ) : 5 ∣ a (5 * k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_five_l768_76881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_30_l768_76807

theorem remainder_sum_mod_30 (a b c d : ℕ) 
  (ha : a % 30 = 15)
  (hb : b % 30 = 7)
  (hc : c % 30 = 22)
  (hd : d % 30 = 6) :
  (a + b + c + d) % 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_30_l768_76807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l768_76863

-- Define the power function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x ^ a

-- State the theorem
theorem power_function_value (a : ℝ) :
  (f a 3 = Real.sqrt 3 / 3) → (f a 9 = 1 / 3) := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l768_76863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_subtraction_l768_76885

def repeating_decimal (whole : ℕ) (rep : ℕ) : ℚ :=
  (whole : ℚ) / 999

theorem repeating_decimal_subtraction :
  repeating_decimal 567 567 - repeating_decimal 234 234 - repeating_decimal 891 891 = -186 / 333 := by
  -- Expand the definition of repeating_decimal
  unfold repeating_decimal
  -- Perform arithmetic operations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_subtraction_l768_76885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walts_running_speed_l768_76857

/-- Proves that Walt's running speed was 6 miles per hour given the problem conditions --/
theorem walts_running_speed (total_distance : ℝ) (lionels_speed : ℝ) (walt_delay : ℝ) (lionels_distance : ℝ)
  (h1 : total_distance = 48)
  (h2 : lionels_speed = 2)
  (h3 : walt_delay = 2)
  (h4 : lionels_distance = 15) :
  (total_distance - lionels_distance) / (lionels_distance / lionels_speed - walt_delay) = 6 := by
  sorry

#check walts_running_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walts_running_speed_l768_76857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seminar_seating_equation_l768_76838

/-- Represents the number of rows in the hall -/
def x : ℕ → ℕ := fun _ => 0  -- Placeholder definition

/-- The total number of parents attending the seminar -/
def total_parents (x : ℕ) : ℕ := 30 * x + 6

/-- Theorem stating that the equation 30x + 6 = 31x - 15 correctly represents the situation -/
theorem seminar_seating_equation (x : ℕ) : total_parents x = 31 * x - 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seminar_seating_equation_l768_76838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_guide_arrangement_l768_76861

/-- The number of ways to arrange tourists and tour guides in a row --/
def arrangementCount (tourists : ℕ) (tourGuides : ℕ) : ℕ :=
  (tourists + 1) * Nat.factorial tourGuides

/-- Theorem: Arranging 5 tourists and 2 tour guides with guides next to each other --/
theorem tourist_guide_arrangement :
  arrangementCount 5 2 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_guide_arrangement_l768_76861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_statements_true_l768_76859

noncomputable def f (x : ℝ) : ℝ := Real.cos (5 * Real.pi / 2 - 2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)
noncomputable def h (x : ℝ) : ℝ := Real.sin (2 * x + 5 * Real.pi / 4)
noncomputable def j (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3)
noncomputable def k (x : ℝ) : ℝ := Real.cos (2 * x)

theorem two_statements_true : 
  (∃ (S : Finset (Prop)), S.card = 2 ∧ S ⊆ {
    (∀ x, f x = f (-x)),
    (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), ∀ y ∈ Set.Icc (-Real.pi/4) (Real.pi/4), x ≤ y → g x ≤ g y),
    (∀ x, h (Real.pi/8 + x) = h (Real.pi/8 - x)),
    (∀ x, j (x + Real.pi/3) = k x)
  }) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_statements_true_l768_76859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l768_76889

/-- The time (in seconds) it takes for a train to pass a man moving in the opposite direction -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_mps := relative_speed * (1000 / 3600)
  train_length / relative_speed_mps

/-- Theorem stating that the time for a 200-meter long train moving at 60 kmph to pass 
    a man moving at 6 kmph in the opposite direction is approximately 10.91 seconds -/
theorem train_passing_time_approx :
  ∃ ε > 0, |train_passing_time 200 60 6 - 10.91| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l768_76889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_one_sufficient_not_necessary_for_cos_zero_l768_76887

theorem sin_one_sufficient_not_necessary_for_cos_zero :
  (∀ x : ℝ, Real.sin x ^ 2 + Real.cos x ^ 2 = 1) →
  ((∀ x : ℝ, Real.sin x = 1 → Real.cos x = 0) ∧
   ¬(∀ x : ℝ, Real.cos x = 0 → Real.sin x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_one_sufficient_not_necessary_for_cos_zero_l768_76887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l768_76833

noncomputable def is_ascending (l : List ℝ) : Prop :=
  ∀ i j, i < j → i < l.length → j < l.length → l[i]! ≤ l[j]!

noncomputable def median (l : List ℝ) : ℝ :=
  if l.length % 2 = 0
  then (l[l.length / 2 - 1]! + l[l.length / 2]!) / 2
  else l[l.length / 2]!

theorem find_x (l : List ℝ) (x : ℝ) :
  l = [23, 28, 30, x, 34, 39] →
  is_ascending l →
  median l = 31 →
  x = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l768_76833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_combinatorial_identity_l768_76811

def P (x : ℕ) : ℕ := x * (x - 1) * (x - 2)

def C (n m : ℕ) : ℕ := n.choose m

theorem inequality_solution (x : ℕ) (h : x ≥ 3) :
  3 * P x ^ 3 ≤ 2 * P (x + 1) ^ 2 + 6 * P x ^ 2 ↔ x = 3 ∨ x = 4 ∨ x = 5 :=
sorry

theorem combinatorial_identity (m : ℕ) :
  (1 : ℚ) / C 5 m - 1 / C 6 m = 7 / (10 * C 7 m) → C 8 m = 28 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_combinatorial_identity_l768_76811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_y_quicker_by_1_8_minutes_l768_76879

-- Define the routes and their properties
noncomputable def route_x_distance : ℝ := 8
noncomputable def route_x_speed : ℝ := 40
noncomputable def route_y_total_distance : ℝ := 7
noncomputable def route_y_construction_distance : ℝ := 1
noncomputable def route_y_regular_distance : ℝ := route_y_total_distance - route_y_construction_distance
noncomputable def route_y_regular_speed : ℝ := 50
noncomputable def route_y_construction_speed : ℝ := 20

-- Define the time calculation function
noncomputable def time_in_minutes (distance : ℝ) (speed : ℝ) : ℝ :=
  (distance / speed) * 60

-- Theorem statement
theorem route_y_quicker_by_1_8_minutes :
  let time_x := time_in_minutes route_x_distance route_x_speed
  let time_y_regular := time_in_minutes route_y_regular_distance route_y_regular_speed
  let time_y_construction := time_in_minutes route_y_construction_distance route_y_construction_speed
  let time_y := time_y_regular + time_y_construction
  time_x - time_y = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_y_quicker_by_1_8_minutes_l768_76879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_derivative_l768_76850

open Real

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => log x / log 2) x = 1 / (x * log 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_derivative_l768_76850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l768_76897

theorem line_slope (θ : ℝ) (h : Real.cos θ = 4/5) : Real.tan θ = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l768_76897
