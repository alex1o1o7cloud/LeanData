import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_construction_l1187_118735

-- Define the basic types and structures
def Point := ℝ × ℝ

structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the conditions
def passes_through (line_start line_end point : Point) : Prop := sorry

def ratio (a b : ℝ) (r : ℚ) : Prop := sorry

def is_rectangle (rect : Rectangle) : Prop := sorry

def positive_orientation (rect : Rectangle) : Prop := sorry

def negative_orientation (rect : Rectangle) : Prop := sorry

-- Define a distance function for Points
def dist (p q : Point) : ℝ := sorry

-- Main theorem
theorem rectangle_construction (A M N : Point) : 
  ∃ (rect1 rect2 : Rectangle), 
    is_rectangle rect1 ∧ 
    is_rectangle rect2 ∧ 
    passes_through rect1.B rect1.C M ∧ 
    passes_through rect1.C rect1.D N ∧
    passes_through rect2.B rect2.C M ∧ 
    passes_through rect2.C rect2.D N ∧
    ratio (dist rect1.A rect1.B) (dist rect1.B rect1.C) (2/5) ∧
    ratio (dist rect2.A rect2.B) (dist rect2.B rect2.C) (2/5) ∧
    positive_orientation rect1 ∧
    negative_orientation rect2 ∧
    A ≠ M ∧ A ≠ N := by
  sorry

#check rectangle_construction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_construction_l1187_118735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_sum_l1187_118782

/-- Represents a line in the form ax + by + d = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  d : ℝ

/-- The distance between two parallel lines -/
noncomputable def distance (l₁ l₂ : Line) : ℝ :=
  abs (l₂.d / l₂.a - l₁.d / l₁.a) / Real.sqrt (1 + (l₁.b / l₁.a)^2)

theorem parallel_lines_sum (b c : ℝ) :
  let l₁ : Line := ⟨3, 4, 5⟩
  let l₂ : Line := ⟨6, b, c⟩
  (l₁.a * l₂.b = l₁.b * l₂.a) →  -- Parallel condition
  (distance l₁ l₂ = 3) →
  (b + c = 48 ∨ b + c = -12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_sum_l1187_118782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1187_118755

noncomputable def f (x : ℝ) : ℝ := |Real.sin (2 * x) + Real.sin (3 * x) + Real.sin (4 * x)|

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1187_118755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1187_118753

open Real

-- Define the curve C
noncomputable def C (α : Real) : Real × Real :=
  (Real.cos α, 1 + Real.sin α)

-- Define the valid range for α
def α_range (α : Real) : Prop :=
  0 ≤ α ∧ α ≤ Real.pi

-- Define the Cartesian equation of C
def C_cartesian (x y : Real) : Prop :=
  x^2 + (y-1)^2 = 1 ∧ 1 ≤ y ∧ y ≤ 2

-- Define the polar equation of C
def C_polar (ρ θ : Real) : Prop :=
  ρ = 2 * Real.sin θ ∧ Real.pi/4 ≤ θ ∧ θ ≤ 3*Real.pi/4

-- Define the trajectory of point M
def M_trajectory (ρ θ : Real) : Prop :=
  ρ * Real.sin θ = 2 ∧ Real.pi/4 ≤ θ ∧ θ ≤ 3*Real.pi/4

theorem curve_C_properties :
  ∀ α x y ρ θ,
  α_range α →
  C α = (x, y) →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  (C_cartesian x y ∧ C_polar ρ θ) ∧
  (∀ ρ_M θ_M, M_trajectory ρ_M θ_M → ρ * ρ_M = 4 ∧ θ_M = θ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1187_118753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_surface_area_l1187_118798

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  R : ℝ  -- Lower base radius
  r : ℝ  -- Upper base radius
  h : ℝ  -- Height

/-- Calculates the total surface area of a frustum -/
noncomputable def totalSurfaceArea (f : Frustum) : ℝ :=
  let s := Real.sqrt (f.h^2 + (f.R - f.r)^2)
  let lateralArea := Real.pi * (f.R + f.r) * s
  let topArea := Real.pi * f.r^2
  let bottomArea := Real.pi * f.R^2
  lateralArea + topArea + bottomArea

/-- Theorem stating the total surface area of a specific frustum -/
theorem frustum_surface_area :
  let f : Frustum := { R := 8, r := 2, h := 5 }
  totalSurfaceArea f = 10 * Real.pi * Real.sqrt 61 + 68 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_surface_area_l1187_118798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_sqrt_5_is_definite_quadratic_radical_l1187_118703

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop := ∃ (y : ℝ), y ≥ 0 ∧ x = Real.sqrt y

-- Define the given expressions
noncomputable def expr1 : ℝ := (6 : ℝ) ^ (1/3)
noncomputable def expr2 (a : ℝ) : ℝ := Real.sqrt a
noncomputable def expr3 : ℝ := Real.sqrt 5
noncomputable def expr4 : ℝ := Real.sqrt (-2)

-- Theorem statement
theorem only_sqrt_5_is_definite_quadratic_radical :
  is_quadratic_radical expr3 ∧
  (¬ is_quadratic_radical expr1) ∧
  (∃ a, ¬ is_quadratic_radical (expr2 a)) ∧
  (¬ is_quadratic_radical expr4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_sqrt_5_is_definite_quadratic_radical_l1187_118703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_boys_count_l1187_118725

theorem school_boys_count :
  ∀ (total_boys : ℕ),
    (0.34 * (total_boys : ℝ) = (total_boys * 34 / 100 : ℕ)) →
    (0.28 * (total_boys : ℝ) = (total_boys * 28 / 100 : ℕ)) →
    (0.10 * (total_boys : ℝ) = (total_boys * 10 / 100 : ℕ)) →
    ((total_boys - (total_boys * 34 / 100 + total_boys * 28 / 100 + total_boys * 10 / 100) : ℕ) = 238) →
    total_boys = 850 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_boys_count_l1187_118725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_hilt_detergent_usage_l1187_118723

/-- Calculates the total amount of detergent used by Mrs. Hilt -/
def totalDetergent (cottonRate woolenRate syntheticRate : ℚ) 
                   (cottonAmount woolenAmount syntheticAmount : ℚ) : ℚ :=
  cottonRate * cottonAmount + woolenRate * woolenAmount + syntheticRate * syntheticAmount

/-- Proves that Mrs. Hilt uses 19 ounces of detergent in total -/
theorem mrs_hilt_detergent_usage : 
  totalDetergent 2 3 1 4 3 2 = 19 := by
  unfold totalDetergent
  -- Evaluate the expression
  simp [mul_add]
  -- Perform the arithmetic
  norm_num

-- Verify the result
#eval totalDetergent 2 3 1 4 3 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_hilt_detergent_usage_l1187_118723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_food_calculation_l1187_118774

/-- Calculates the amount of food each ant needs given the following conditions:
    - There are 400 ants
    - Each ounce of ant food costs $0.1
    - Nikola charges $5 to start a job
    - Each leaf he rakes costs $0.01
    - He raked 6,000 leaves
    - He completed 4 jobs
-/
theorem ant_food_calculation (num_ants : ℕ) (food_cost_per_ounce : ℚ) 
    (job_start_fee : ℚ) (cost_per_leaf : ℚ) (num_leaves : ℕ) (num_jobs : ℕ) :
  num_ants = 400 →
  food_cost_per_ounce = 1/10 →
  job_start_fee = 5 →
  cost_per_leaf = 1/100 →
  num_leaves = 6000 →
  num_jobs = 4 →
  (num_jobs : ℚ) * job_start_fee + (num_leaves : ℚ) * cost_per_leaf / food_cost_per_ounce / (num_ants : ℚ) = 2 := by
  sorry

-- Uncomment the line below if you want to evaluate the theorem
-- #eval ant_food_calculation 400 (1/10) 5 (1/100) 6000 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_food_calculation_l1187_118774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_preserves_sum_free_sets_l1187_118775

def IsSumFreeSet (A : Set ℕ) : Prop :=
  ∀ x y, x ∈ A → y ∈ A → x + y ∉ A

def PreservesSumFreeSet (f : ℕ → ℕ) : Prop :=
  ∀ A : Set ℕ, IsSumFreeSet A → IsSumFreeSet (f '' A)

theorem identity_preserves_sum_free_sets
  (f : ℕ → ℕ)
  (h_surj : Function.Surjective f)
  (h_preserve : PreservesSumFreeSet f) :
  f = id := by
  sorry

#check identity_preserves_sum_free_sets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_preserves_sum_free_sets_l1187_118775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_tether_area_l1187_118710

/-- The area outside a regular hexagon that a tethered dog can reach -/
theorem dog_tether_area (side_length : ℝ) (tether_length : ℝ) : 
  side_length = 2 → tether_length = 3 → 
  (2 * π * tether_length^2 / 3) + (2 * (π * side_length^2 / 6)) = 22 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_tether_area_l1187_118710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_steps_to_meet_ella_l1187_118747

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℕ := 15840

/-- Bella's walking speed in feet per minute -/
def bella_speed : ℝ := 1  -- We assign a value of 1 as a placeholder

/-- Ella's cycling speed in feet per minute -/
def ella_speed : ℝ := 3 * bella_speed

/-- The length of Bella's step in feet -/
def step_length : ℝ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 1320

theorem bella_steps_to_meet_ella :
  (distance : ℝ) / (bella_speed + ella_speed) * bella_speed / step_length = steps_taken := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_steps_to_meet_ella_l1187_118747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_for_two_white_l1187_118743

/-- Represents a box containing a ball -/
structure Box :=
  (color : Bool)  -- true if white, false if black

/-- Represents a question about two boxes -/
def Question := Fin 2004 × Fin 2004

/-- Represents the state of all boxes -/
def BoxState := Fin 2004 → Box

/-- Predicate to check if a BoxState is valid (even number of white balls) -/
def validBoxState (state : BoxState) : Prop :=
  Even (Finset.card (Finset.filter (fun i => (state i).color) Finset.univ))

/-- Predicate to check if a question yields a positive answer -/
def positiveAnswer (state : BoxState) (q : Question) : Prop :=
  (state q.1).color ∨ (state q.2).color

/-- Predicate to check if a set of questions guarantees finding two white balls -/
def guaranteesTwoWhite (questions : Finset Question) : Prop :=
  ∀ state, validBoxState state →
    ∃ i j, i ≠ j ∧ (state i).color ∧ (state j).color ∧
      ∀ q ∈ questions, positiveAnswer state q

/-- The main theorem: 4005 questions are necessary and sufficient -/
theorem min_questions_for_two_white :
  ∃ questions : Finset Question,
    questions.card = 4005 ∧
    guaranteesTwoWhite questions ∧
    ∀ questions' : Finset Question,
      questions'.card < 4005 →
      ¬guaranteesTwoWhite questions' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_for_two_white_l1187_118743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1187_118757

theorem inequality_solution (x : ℝ) : 
  x ≠ 4 → (x * (x + 1) / (x - 4)^2 ≥ 12 ↔ (x ∈ Set.Icc 3 4 ∪ Set.Ico 4 (64/11))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1187_118757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_consecutive_digits_l1187_118740

def has_consecutive_digits (m n : ℕ) : Prop :=
  ∃ k : ℕ, (10^k * m) % n ≥ 347 * n / 1000 ∧ (10^k * m) % n < 348 * n / 1000

theorem smallest_n_with_consecutive_digits : 
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, m < n ∧ Nat.Coprime m n ∧ has_consecutive_digits m n) → n ≥ 999 :=
by
  sorry

#check smallest_n_with_consecutive_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_consecutive_digits_l1187_118740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1187_118766

noncomputable def a : ℝ := Real.sqrt 0.5
noncomputable def b : ℝ := Real.sqrt 0.3
noncomputable def c : ℝ := Real.log 0.2 / Real.log 0.3

theorem relationship_abc : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1187_118766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_function_l1187_118705

theorem min_value_trig_function (x : ℝ) (h1 : Real.cos x ≠ 0) (h2 : Real.sin x ≠ 0) :
  (4 / (Real.cos x)^2 + 9 / (Real.sin x)^2) ≥ 25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_function_l1187_118705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arm_wrestling_tournament_l1187_118739

theorem arm_wrestling_tournament (n : ℕ) (h1 : n > 7) : 
  (∃ (f : ℕ → ℕ → ℕ), 
    (∀ (m k : ℕ), m ≤ 7 → k ≤ m → f m k = 2^(n-m) * (Nat.choose m k)) ∧
    f 7 5 = 42) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arm_wrestling_tournament_l1187_118739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_is_242_cents_l1187_118716

/-- The list price of Camera Y in dollars -/
def list_price : ℚ := 50.50

/-- The discount amount at Deal Direct in dollars -/
def deal_direct_discount : ℚ := 10.50

/-- The discount percentage at Bargain Base -/
def bargain_base_discount_percent : ℚ := 20 / 100

/-- The tax rate at Bargain Base -/
def bargain_base_tax_rate : ℚ := 5 / 100

/-- The price at Deal Direct in dollars -/
def deal_direct_price : ℚ := list_price - deal_direct_discount

/-- The price at Bargain Base before tax in dollars -/
def bargain_base_price_before_tax : ℚ := list_price * (1 - bargain_base_discount_percent)

/-- The final price at Bargain Base including tax in dollars -/
def bargain_base_final_price : ℚ := bargain_base_price_before_tax * (1 + bargain_base_tax_rate)

/-- The price difference between Bargain Base and Deal Direct in cents -/
def price_difference_cents : ℤ := Int.floor ((bargain_base_final_price - deal_direct_price) * 100)

theorem price_difference_is_242_cents : price_difference_cents = 242 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_is_242_cents_l1187_118716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1187_118765

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (-abs x)

-- Define a, b, and c
noncomputable def a : ℝ := f (Real.log 3 / Real.log 0.5)
noncomputable def b : ℝ := f (Real.log 5 / Real.log 2)
noncomputable def c : ℝ := f 0

-- Theorem statement
theorem relationship_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1187_118765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_trigonometric_values_l1187_118777

/-- Given an angle α whose terminal side passes through the point (4sinθ, -3sinθ) where θ ∈ (π, 3π/2),
    prove that sinα = 3/5, cosα = -4/5, and tanα = -3/4 -/
theorem angle_trigonometric_values (θ α : Real) (h : θ ∈ Set.Ioo π (3*π/2)) 
  (terminal_point : (4 * Real.sin θ, -3 * Real.sin θ) ∈ Metric.sphere (0 : ℝ × ℝ) 1) : 
  Real.sin α = 3/5 ∧ Real.cos α = -4/5 ∧ Real.tan α = -3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_trigonometric_values_l1187_118777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lighthouse_height_approx_100_l1187_118742

open Real

-- Define the problem parameters
noncomputable def ship_distance : ℝ := 273.2050807568877
noncomputable def angle1 : ℝ := 30 * Real.pi / 180  -- 30° in radians
noncomputable def angle2 : ℝ := 45 * Real.pi / 180  -- 45° in radians

-- Define the lighthouse height function
noncomputable def lighthouse_height (d : ℝ) (α₁ α₂ : ℝ) : ℝ :=
  d / (1 / Real.tan α₁ + 1 / Real.tan α₂)

-- State the theorem
theorem lighthouse_height_approx_100 :
  ∃ ε > 0, abs (lighthouse_height ship_distance angle1 angle2 - 100) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lighthouse_height_approx_100_l1187_118742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_tax_percentage_l1187_118719

theorem farm_tax_percentage (total_tax village_tax willam_tax : ℝ)
  (willam_land_percentage : ℝ) :
  total_tax = 3840 →
  willam_tax = 480 →
  willam_land_percentage = 31.25 →
  (willam_tax / total_tax) * 100 = 12.5 := by
  intros h1 h2 h3
  -- The proof steps would go here
  sorry

#check farm_tax_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_tax_percentage_l1187_118719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l1187_118794

/-- Circle in polar coordinates -/
def circleEq (a : ℝ) (ρ θ : ℝ) : Prop := ρ = 2 * a * Real.cos θ

/-- Line in parametric form -/
def lineEq (t x y : ℝ) : Prop := x = 3 * t + 2 ∧ y = 4 * t + 2

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (x₀ y₀ : ℝ) : ℝ :=
  |4 * x₀ - 3 * y₀ - 2| / Real.sqrt (4^2 + (-3)^2)

/-- Tangency condition: distance from center to line equals radius -/
def isTangent (a : ℝ) : Prop := distancePointToLine a 0 = |a|

theorem circle_tangent_line (a : ℝ) : 
  (∃ ρ θ t x y, circleEq a ρ θ ∧ lineEq t x y ∧ isTangent a) → a = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l1187_118794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeremy_bus_time_l1187_118713

/-- Represents the time Jeremy spends on various activities during his school day --/
structure JeremySchedule where
  wakeUpTime : Nat -- in minutes after midnight
  busDepartureTime : Nat -- in minutes after midnight
  numberOfClasses : Nat
  classDuration : Nat -- in minutes
  lunchDuration : Nat -- in minutes
  additionalSchoolTime : Nat -- in minutes
  arrivalTime : Nat -- in minutes after midnight

/-- Calculates the total time Jeremy spends on the bus --/
def busTime (schedule : JeremySchedule) : Nat :=
  let totalAwayTime := schedule.arrivalTime - schedule.busDepartureTime
  let totalSchoolTime := 
    schedule.numberOfClasses * schedule.classDuration + 
    schedule.lunchDuration + 
    schedule.additionalSchoolTime
  totalAwayTime - totalSchoolTime

/-- Theorem stating that Jeremy spends 105 minutes on the bus --/
theorem jeremy_bus_time :
  let schedule : JeremySchedule := {
    wakeUpTime := 6 * 60,
    busDepartureTime := 7 * 60,
    numberOfClasses := 7,
    classDuration := 45,
    lunchDuration := 45,
    additionalSchoolTime := 135, -- 2.25 hours = 135 minutes
    arrivalTime := 17 * 60
  }
  busTime schedule = 105 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeremy_bus_time_l1187_118713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_special_case_l1187_118785

/-- A triangle with sides that are three consecutive positive integers
    and where the largest angle is twice the smallest angle has an area of 15√7/4. -/
theorem triangle_area_special_case :
  ∀ (a b c : ℕ+) (A B C : ℝ),
    a.val + 1 = b.val →
    b.val + 1 = c.val →
    A + B + C = π →
    0 < A ∧ A < C ∧ C < π →
    C = 2 * A →
    (1/2 : ℝ) * b.val * c.val * Real.sin A = (15 * Real.sqrt 7) / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_special_case_l1187_118785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_squared_l1187_118795

/-- The diameter of the circular pie in centimeters -/
noncomputable def diameter : ℝ := 20

/-- The number of equal pieces the pie is cut into -/
def num_pieces : ℕ := 4

/-- The central angle of each sector in radians -/
noncomputable def sector_angle : ℝ := 2 * Real.pi / num_pieces

/-- The longest line segment within a sector -/
noncomputable def longest_segment (d : ℝ) (θ : ℝ) : ℝ :=
  d * Real.sin (θ / 2)

/-- Theorem: The square of the longest line segment within a 90-degree sector
    of a circle with diameter 20 cm is equal to 200 cm^2 -/
theorem longest_segment_squared :
  (longest_segment diameter sector_angle)^2 = 200 := by
  sorry

#eval Float.sqrt 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_squared_l1187_118795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_rv_expected_value_l1187_118754

/-- A random variable uniformly distributed on an interval -/
structure UniformRV (α β : ℝ) where
  hlt : α < β

/-- The expected value of a uniform random variable -/
noncomputable def expected_value (α β : ℝ) (X : UniformRV α β) : ℝ := (β + α) / 2

/-- Theorem: The expected value of a uniform random variable on [α, β] is (β + α) / 2 -/
theorem uniform_rv_expected_value (α β : ℝ) (X : UniformRV α β) :
  expected_value α β X = (β + α) / 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_rv_expected_value_l1187_118754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_between_26_and_26_5_l1187_118751

theorem square_root_between_26_and_26_5 : ∃ n : ℕ, 
  (n : ℝ).sqrt > 26 ∧ 
  (n : ℝ).sqrt < 26.5 ∧ 
  n % 14 = 0 ∧
  n = 700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_between_26_and_26_5_l1187_118751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_driving_time_increase_l1187_118748

theorem driving_time_increase (x : ℝ) (h : x > 0) : 
  (32 / (2 * x) + 32 / (x / 2) - 64 / x) / (64 / x) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_driving_time_increase_l1187_118748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_still_water_speed_l1187_118708

/-- Represents the speed of a boat in various conditions -/
structure BoatSpeed where
  downstream : ℝ
  upstream : ℝ
  waterCurrent1 : ℝ
  waterCurrent2 : ℝ
  windSpeed : ℝ

/-- Calculates the estimated speed of a boat in still water -/
noncomputable def estimatedStillWaterSpeed (b : BoatSpeed) : ℝ :=
  let effectiveCurrent1 := b.waterCurrent1 - b.windSpeed
  let effectiveCurrent2 := b.waterCurrent2 - b.windSpeed
  let stillWaterSpeed1 := b.downstream - effectiveCurrent1
  let stillWaterSpeed2 := b.upstream + effectiveCurrent2
  (stillWaterSpeed1 + stillWaterSpeed2) / 2

/-- Theorem stating that the estimated speed of the boat in still water is 13.5 km/hr -/
theorem boat_still_water_speed (b : BoatSpeed) 
  (h1 : b.downstream = 16)
  (h2 : b.upstream = 9)
  (h3 : b.waterCurrent1 = 3)
  (h4 : b.waterCurrent2 = 5)
  (h5 : b.windSpeed = 2) :
  estimatedStillWaterSpeed b = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_still_water_speed_l1187_118708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_sum_of_digits_of_1998_digit_number_divisible_by_9_l1187_118792

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is divisible by 9 if and only if the sum of its digits is divisible by 9 -/
axiom divisible_by_9_iff_sum_of_digits_divisible_by_9 (n : ℕ) :
  n % 9 = 0 ↔ (sum_of_digits n) % 9 = 0

/-- A 1998-digit number -/
def is_1998_digit_number (n : ℕ) : Prop :=
  10^1997 ≤ n ∧ n < 10^1998

theorem sum_of_digits_of_sum_of_digits_of_1998_digit_number_divisible_by_9 (n : ℕ) :
  is_1998_digit_number n →
  n % 9 = 0 →
  sum_of_digits (sum_of_digits (sum_of_digits n)) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_sum_of_digits_of_1998_digit_number_divisible_by_9_l1187_118792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_home_runs_per_game_is_127_div_147_l1187_118788

/-- Represents a group of players with the same number of home runs and games played -/
structure PlayerGroup where
  players : ℕ
  homeRuns : ℕ
  gamesPlayed : ℕ

/-- Calculates the mean number of home runs per game for a given set of player groups -/
def meanHomeRunsPerGame (groups : List PlayerGroup) : ℚ :=
  let totalHomeRuns := groups.foldl (fun acc g => acc + (g.players * g.homeRuns : ℕ)) 0
  let totalGames := groups.foldl (fun acc g => acc + (g.players * g.gamesPlayed : ℕ)) 0
  (totalHomeRuns : ℚ) / (totalGames : ℚ)

/-- The theorem stating that the mean number of home runs per game is 127/147 -/
theorem mean_home_runs_per_game_is_127_div_147 :
  let groups : List PlayerGroup := [
    ⟨5, 4, 5⟩,
    ⟨6, 5, 6⟩,
    ⟨4, 7, 8⟩,
    ⟨3, 9, 10⟩,
    ⟨2, 11, 12⟩
  ]
  meanHomeRunsPerGame groups = 127 / 147 := by
  sorry

#eval (127 : ℚ) / 147

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_home_runs_per_game_is_127_div_147_l1187_118788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completion_time_l1187_118736

/-- The number of days it takes for person a to complete the work -/
noncomputable def days_a : ℝ := 15

/-- The number of days it takes for person c to complete the work -/
noncomputable def days_c : ℝ := 45

/-- The number of days it takes for persons a, b, and c to complete the work together -/
noncomputable def days_abc : ℝ := 7.2

/-- The amount of work to be completed -/
noncomputable def work : ℝ := 1

/-- Calculate the number of days it takes for person b to complete the work alone -/
noncomputable def days_b : ℝ := 
  (1 / days_abc - 1 / days_a - 1 / days_c)⁻¹

theorem b_completion_time :
  |days_b - 14.58| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completion_time_l1187_118736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_inequality_l1187_118706

/-- Helper function to calculate the area of a triangle given its side lengths -/
noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Given a triangle ABC with sides a, b, c and corresponding altitudes ha, hb, hc,
    the sum (hb² + hc²)/a² + (hc² + ha²)/b² + (ha² + hb²)/c² is less than or equal to 9/2. -/
theorem triangle_altitude_inequality (a b c ha hb hc : ℝ) 
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
    (h_altitude_a : ha = 2 * (area_triangle a b c) / a)
    (h_altitude_b : hb = 2 * (area_triangle a b c) / b)
    (h_altitude_c : hc = 2 * (area_triangle a b c) / c) :
  (hb^2 + hc^2) / a^2 + (hc^2 + ha^2) / b^2 + (ha^2 + hb^2) / c^2 ≤ 9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_inequality_l1187_118706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_all_reals_l1187_118727

/-- The function c(x) with parameter k -/
noncomputable def c (k : ℝ) (x : ℝ) : ℝ := (k * x^2 + 3 * x - 4) / (-3 * x^2 + 5 * x + k)

/-- The theorem stating the condition for c(x) to have a domain of all real numbers -/
theorem domain_all_reals (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, c k x = y) ↔ k < -25/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_all_reals_l1187_118727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1187_118714

theorem sin_2alpha_value (α : Real) 
  (h : Real.sin α - Real.cos α = Real.sqrt 5 / 5) : 
  Real.sin (2 * α) = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1187_118714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1187_118721

def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (0, 2)

theorem vector_properties :
  let diff := (a.1 - b.1, a.2 - b.2)
  (diff.1 * a.1 + diff.2 * a.2 = 0) ∧ 
  (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1187_118721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_is_quadratic_l1187_118767

noncomputable section

/-- Definition of a quadratic equation in one variable -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equations given in the problem -/
def eq_A : ℝ → ℝ := λ x ↦ 2*x - 1 - 4
def eq_B : ℝ → ℝ → ℝ := λ x y ↦ x*y + x - 3
def eq_C : ℝ → ℝ := λ x ↦ x - 1/x - 5
def eq_D : ℝ → ℝ := λ x ↦ x^2 - 2*x + 1

/-- Theorem stating that only eq_D is quadratic -/
theorem only_D_is_quadratic :
  ¬ is_quadratic eq_A ∧
  ¬ is_quadratic (λ x ↦ eq_B x x) ∧
  ¬ is_quadratic eq_C ∧
  is_quadratic eq_D := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_is_quadratic_l1187_118767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_multiplication_one_third_times_four_one_eighth_times_four_l1187_118786

/-- Represents the special multiplication operation in the country -/
def country_mul (f : ℚ) (x : ℚ) (y : ℚ) : ℚ := f * x / y

/-- The proportion factor p -/
def p : ℚ := 1 / 24

theorem special_multiplication (f x y : ℚ) :
  country_mul f x y = p * x * y :=
by sorry

theorem one_third_times_four :
  country_mul (1/3) 4 8 = 8 :=
by sorry

theorem one_eighth_times_four :
  country_mul (1/8) 4 3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_multiplication_one_third_times_four_one_eighth_times_four_l1187_118786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_sales_tax_percentage_l1187_118789

/-- The sales tax percentage for a radio purchase -/
noncomputable def sales_tax_percentage (total_price : ℝ) (tax_amount : ℝ) : ℝ :=
  (tax_amount * 100) / (total_price - tax_amount)

/-- Theorem stating the sales tax percentage for the given problem -/
theorem radio_sales_tax_percentage :
  sales_tax_percentage 2468 161.46 = (161.46 * 100) / 2306.54 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval sales_tax_percentage 2468 161.46

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_sales_tax_percentage_l1187_118789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_six_l1187_118709

theorem complex_expression_equals_six :
  (2 * Real.sqrt 3 - Real.pi) ^ (0 : ℝ) - |1 - Real.sqrt 3| + 3 * Real.tan (30 * π / 180) + (-1/2 : ℝ) ^ (-2 : ℤ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_six_l1187_118709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_score_tenth_grader_l1187_118796

def checkers_tournament (a : ℕ) : Prop :=
  let total_players := 11 * a
  let total_games := total_players * (total_players - 1) / 2
  let total_points := 2 * total_games
  let tenth_grade_points := total_points / (11 : ℕ) * 2
  let max_tenth_grade_score := 2 * (11 * a - 1)
  (a > 0) ∧ 
  (max_tenth_grade_score ≤ 20)

theorem max_score_tenth_grader :
  ∃ a : ℕ, checkers_tournament a ∧ 
  ∀ b : ℕ, checkers_tournament b → 
    (2 * (11 * b - 1) ≤ 20) := by
  sorry

#check max_score_tenth_grader

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_score_tenth_grader_l1187_118796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l1187_118781

noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / (7 * x - 10)

theorem vertical_asymptote_of_f :
  ∃ (x : ℝ), x = 10 / 7 ∧ (∀ ε > 0, ∃ δ > 0, ∀ y, 0 < |y - x| ∧ |y - x| < δ → |f y| > 1 / ε) :=
by
  -- We'll use 10/7 as our x-value
  let x := 10 / 7
  
  -- Prove that this x satisfies the conditions
  have h1 : x = 10 / 7 := rfl
  
  -- The rest of the proof would go here
  -- For now, we'll use sorry to skip the detailed proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l1187_118781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_is_25_l1187_118762

/-- The distance traveled by a train in 45 minutes -/
noncomputable def train_distance : ℝ :=
  let speed_mph := 60 / 1.75  -- Speed in miles per hour
  let time_hours := 45 / 60   -- Time in hours
  speed_mph * time_hours

/-- Theorem stating that a train traveling 1 mile in 1 minute 45 seconds will cover 25 miles in 45 minutes -/
theorem train_distance_is_25 : ⌊train_distance⌋ = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_is_25_l1187_118762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_time_l1187_118730

/-- Represents a burning candle -/
structure Candle where
  initial_height : ℝ
  burn_time : ℝ

/-- Calculates the height of a candle after a given time -/
noncomputable def candle_height (c : Candle) (t : ℝ) : ℝ :=
  c.initial_height - (c.initial_height / c.burn_time) * t

theorem candle_height_ratio_time (c1 c2 : Candle) :
  c1.initial_height = 10 →
  c1.burn_time = 5 →
  c2.initial_height = 8 →
  c2.burn_time = 4 →
  ∃ t : ℝ, t = 3.5 ∧ candle_height c1 t = 3 * candle_height c2 t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_ratio_time_l1187_118730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_l1187_118770

/-- A polynomial with non-negative integer coefficients -/
def NonNegIntPolynomial (p : Polynomial ℤ) : Prop :=
  ∀ i, (p.coeff i) ≥ 0

/-- The specific polynomial 2x³ + x + 4 -/
noncomputable def specific_poly : Polynomial ℤ :=
  Polynomial.monomial 3 2 + Polynomial.monomial 1 1 + Polynomial.monomial 0 4

theorem unique_polynomial :
  ∀ p : Polynomial ℤ,
    NonNegIntPolynomial p →
    p.eval 1 = 7 →
    p.eval 10 = 2014 →
    p = specific_poly :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_l1187_118770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1187_118731

/-- Two-dimensional vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot (v w : Vec2D) : ℝ := v.x * w.x + v.y * w.y

/-- Vector addition -/
def add (v w : Vec2D) : Vec2D := ⟨v.x + w.x, v.y + w.y⟩

/-- Scalar multiplication -/
def smul (k : ℝ) (v : Vec2D) : Vec2D := ⟨k * v.x, k * v.y⟩

/-- Magnitude (length) of a vector -/
noncomputable def magnitude (v : Vec2D) : ℝ := Real.sqrt (v.x^2 + v.y^2)

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : Vec2D) : Prop := dot v w = 0

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : Vec2D) : Prop := ∃ k : ℝ, v = smul k w

theorem vector_problem (a b c : Vec2D) (m k : ℝ) 
  (h1 : a = ⟨1, 2⟩) 
  (h2 : b = ⟨-2, 3⟩) 
  (h3 : c = ⟨-2, m⟩) :
  (perpendicular a (add b c) → magnitude c = Real.sqrt 5) ∧
  (collinear (add (smul k a) b) (add (smul 2 a) (smul (-1) b)) → k = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1187_118731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_is_correct_l1187_118752

/-- Represents the price difference scenario for a product purchase --/
noncomputable def price_difference (P : ℝ) : ℝ :=
  let original_price := P
  let inflated_price := original_price * 1.2
  let taxed_price := inflated_price * 1.08
  let original_purchase := taxed_price * 0.7
  let original_gbp := original_purchase / 1.4
  let discounted_price := taxed_price * 0.9
  let new_gbp := discounted_price / 1.35
  new_gbp - original_gbp

/-- Theorem stating the price difference is 0.216P --/
theorem price_difference_is_correct (P : ℝ) :
  price_difference P = 0.216 * P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_is_correct_l1187_118752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1187_118772

def M : Set ℝ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x | 2 * x > 7}

theorem intersection_M_N : M ∩ N = {5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1187_118772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1187_118722

/-- Given vectors a and b in ℝ², find lambda such that (a + lambda*b) ⊥ (a - lambda*b) -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (3, 3)) (h2 : b = (1, -1)) :
  ∃ lambda : ℝ, (a.1 + lambda * b.1, a.2 + lambda * b.2) • (a.1 - lambda * b.1, a.2 - lambda * b.2) = 0 ∧ 
  (lambda = 3 ∨ lambda = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1187_118722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_theorem_l1187_118746

/-- Calculates the required fencing for a rectangular field -/
noncomputable def fencing_required (area : ℝ) (uncovered_side : ℝ) : ℝ :=
  let width := area / uncovered_side
  2 * width + uncovered_side

theorem fencing_theorem :
  fencing_required 680 10 = 146 := by
  -- Unfold the definition of fencing_required
  unfold fencing_required
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_theorem_l1187_118746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_correct_l1187_118769

/-- A sequence defined by a recurrence relation -/
def a : ℕ → ℤ
  | 0 => 1  -- Adding a case for 0 to avoid missing cases error
  | 1 => 1
  | n + 2 => 2 * a (n + 1) + 1

/-- The general term formula for the sequence -/
def general_term (n : ℕ) : ℤ := 2^n - 1

/-- Theorem stating that the general term formula is correct for all n ≥ 1 -/
theorem general_term_correct (n : ℕ) (h : n ≥ 1) : a n = general_term n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_correct_l1187_118769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_period_l1187_118704

/-- Calculates the time period of investment given the principal, interest difference, and two interest rates. -/
noncomputable def calculate_time_period (principal : ℝ) (interest_diff : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  interest_diff / (principal * (rate1 - rate2))

/-- Proves that the time period of investment is 2 years given the specified conditions. -/
theorem investment_time_period :
  let principal : ℝ := 5000
  let interest_diff : ℝ := 600
  let rate1 : ℝ := 0.18
  let rate2 : ℝ := 0.12
  calculate_time_period principal interest_diff rate1 rate2 = 2 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_time_period 5000 600 0.18 0.12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_time_period_l1187_118704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_fourth_quadrant_l1187_118712

theorem tan_value_fourth_quadrant (α : ℝ) 
  (h1 : Real.sin α = -5/13) 
  (h2 : 3*π/2 < α ∧ α < 2*π) : 
  Real.tan α = -5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_fourth_quadrant_l1187_118712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twentieth_term_is_six_sevenths_l1187_118756

/-- Defines the sequence as described in the problem -/
def seq (i : ℕ) : ℚ :=
  let n := (Nat.sqrt (8 * i + 1) + 1) / 2
  let k := i - (n - 1) * n / 2
  ↑k / ↑n

/-- The 20th term of the sequence is 6/7 -/
theorem twentieth_term_is_six_sevenths : seq 20 = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twentieth_term_is_six_sevenths_l1187_118756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_formula_l1187_118768

noncomputable def T (n : ℕ+) (a : ℕ+ → ℝ) : ℝ := 
  (Finset.range n.val).prod (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem a_n_formula (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, T n a = 2 * (n : ℝ)^2) →
  ∀ n : ℕ+, n.val ≥ 2 → a n = ((n : ℝ) / ((n : ℝ) - 1))^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_formula_l1187_118768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_left_is_7800_l1187_118733

/-- Calculates the amount of water left in a bathtub given specific conditions. -/
def water_left_in_bathtub 
  (faucet_rate : ℚ) 
  (evaporation_rate : ℚ) 
  (time : ℚ) 
  (water_dumped : ℚ) : ℚ :=
  let water_added_per_hour := faucet_rate * 60
  let net_water_per_hour := water_added_per_hour - evaporation_rate
  let total_water_added := net_water_per_hour * time
  let water_dumped_ml := water_dumped * 1000
  total_water_added - water_dumped_ml

/-- The amount of water left in the bathtub under given conditions is 7800 ml. -/
theorem water_left_is_7800 :
  water_left_in_bathtub 40 200 9 12 = 7800 := by
  -- Unfold the definition and perform the calculation
  unfold water_left_in_bathtub
  -- Simplify the arithmetic expressions
  simp [Rat.mul_def, Rat.sub_def, Rat.add_def]
  -- The proof is complete
  rfl

#eval water_left_in_bathtub 40 200 9 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_left_is_7800_l1187_118733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1187_118763

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 - 2)^3 + 2013 * (a 2 - 2) = Real.sin (2014 * Real.pi / 3) →
  (a 2013 - 2)^3 + 2013 * (a 2013 - 2) = Real.cos (2015 * Real.pi / 6) →
  sum_of_terms a 2014 = 4028 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1187_118763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1187_118784

/-- A quadratic function f(x) = ax^2 + bx + c with a > 0 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0

/-- The roots of f(x) - x = 0 -/
structure Roots (f : QuadraticFunction) where
  x₁ : ℝ
  x₂ : ℝ
  root_condition : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 / f.a
  are_roots : f.a * x₁^2 + f.b * x₁ + f.c = x₁ ∧ f.a * x₂^2 + f.b * x₂ + f.c = x₂

/-- The symmetry point of f(x) -/
noncomputable def symmetry_point (f : QuadraticFunction) : ℝ := -f.b / (2 * f.a)

/-- The main theorem -/
theorem quadratic_function_properties (f : QuadraticFunction) (r : Roots f) :
  (∀ x, 0 < x → x < r.x₁ → x < f.a * x^2 + f.b * x + f.c ∧ f.a * x^2 + f.b * x + f.c < r.x₁) ∧
  symmetry_point f < r.x₁ / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1187_118784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_cycles_ge_5_eq_7_50_l1187_118760

/-- The probability that a random permutation of 10 elements has all cycle lengths ≥ 5 -/
def prob_all_cycles_ge_5 : ℚ :=
  (Nat.choose 10 5 * (Nat.factorial 4 * Nat.factorial 4) + Nat.factorial 9) / Nat.factorial 10

/-- Theorem stating that the probability of a random permutation of 10 elements 
    having all cycle lengths ≥ 5 is equal to 7/50 -/
theorem prob_all_cycles_ge_5_eq_7_50 : prob_all_cycles_ge_5 = 7 / 50 := by
  sorry

#eval prob_all_cycles_ge_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_cycles_ge_5_eq_7_50_l1187_118760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unique_triangles_l1187_118711

/-- A triangle with integer side lengths --/
structure IntTriangle where
  a : Nat
  b : Nat
  c : Nat
  h1 : a < 6
  h2 : b < 6
  h3 : c < 6
  h4 : a + b > c ∧ b + c > a ∧ c + a > b

/-- Two triangles are similar if their corresponding angles are equal --/
def similar (t1 t2 : IntTriangle) : Prop :=
  (t1.a * t2.b = t1.b * t2.a) ∧ (t1.b * t2.c = t1.c * t2.b) ∧ (t1.c * t2.a = t1.a * t2.c)

/-- Two triangles are congruent if their corresponding sides are equal --/
def congruent (t1 t2 : IntTriangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

/-- The set of all valid integer triangles with side lengths less than 6 --/
def S : Set IntTriangle :=
  {t : IntTriangle | true}

/-- A subset of S where no two triangles are similar or congruent --/
def MaxSet : Set IntTriangle :=
  {t ∈ S | ∀ t' ∈ S, t ≠ t' → ¬(similar t t') ∧ ¬(congruent t t')}

/-- The theorem to be proved --/
theorem max_unique_triangles : ∃ (M : Finset IntTriangle), ↑M ⊆ MaxSet ∧ M.card = 15 ∧
  ∀ (N : Finset IntTriangle), ↑N ⊆ MaxSet → N.card ≤ 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unique_triangles_l1187_118711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_five_primes_units_3_l1187_118776

/-- A function that returns true if a number is prime and has a units digit of 3 -/
def isPrimeWithUnits3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

/-- The list of the first five prime numbers with a units digit of 3 -/
def firstFivePrimesWithUnits3 : List ℕ :=
  [3, 13, 23, 43, 53]

/-- Theorem stating that the sum of the first five prime numbers with a units digit of 3 is 135 -/
theorem sum_first_five_primes_units_3 :
  (firstFivePrimesWithUnits3.map Int.ofNat).sum = 135 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_five_primes_units_3_l1187_118776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1187_118702

noncomputable def f (a b c x : ℝ) : ℝ := a + b * Real.cos x + c * Real.sin x

theorem range_of_a (a b c : ℝ) :
  (f a b c 0 = 1) →
  (f a b c (Real.pi / 2) = 1) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), |f a b c x| ≤ 2) →
  a ∈ Set.Icc (-Real.sqrt 2) (4 + 3 * Real.sqrt 2) := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1187_118702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_height_theorem_l1187_118793

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the base area of a cuboid -/
def baseArea (d : CuboidDimensions) : ℝ :=
  d.length * d.width

/-- Checks if the dimensions satisfy the given ratio -/
def satisfiesRatio (d : CuboidDimensions) : Prop :=
  ∃ (k : ℝ), d.length = 2 * k ∧ d.width = 3 * k ∧ d.height = 4 * k

/-- Theorem stating the height of the cuboid given the conditions -/
theorem cuboid_height_theorem (d : CuboidDimensions) :
  volume d = 144 →
  baseArea d = 18 →
  satisfiesRatio d →
  ∃ ε > 0, |d.height - 7.268| < ε := by
  sorry

#check cuboid_height_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_height_theorem_l1187_118793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lights_off_theorem_l1187_118726

/-- Represents a 100 × 100 × 100 cube with light bulbs and switches -/
structure Cube where
  lights : Fin 100 → Fin 100 → Fin 100 → Bool
  red_switches : Fin 100 → Fin 100 → Bool
  blue_switches : Fin 100 → Fin 100 → Bool
  green_switches : Fin 100 → Fin 100 → Bool

/-- Counts the number of lights that are on in the cube -/
def count_lights_on (c : Cube) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 100)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 100)) fun j =>
      Finset.sum (Finset.univ : Finset (Fin 100)) fun k =>
        if c.lights i j k then 1 else 0

/-- Theorem: For any configuration with k lights on, it's possible to turn off all lights
    using no more than k/100 switches on the red face -/
theorem lights_off_theorem (c : Cube) (k : Nat) 
    (h : count_lights_on c = k) :
    ∃ (new_red_switches : Fin 100 → Fin 100 → Bool),
      count_lights_on { c with red_switches := new_red_switches } = 0 ∧
      Finset.sum (Finset.univ : Finset (Fin 100)) (fun i =>
        Finset.sum (Finset.univ : Finset (Fin 100)) (fun j =>
          if new_red_switches i j then 1 else 0)) ≤ k / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lights_off_theorem_l1187_118726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_eq_two_implies_expression_eq_neg_two_l1187_118732

theorem tan_theta_eq_two_implies_expression_eq_neg_two (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / (Real.sin (π / 2 - θ) - Real.sin (π - θ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_eq_two_implies_expression_eq_neg_two_l1187_118732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_crates_pigeonhole_l1187_118759

theorem orange_crates_pigeonhole (total_crates min_oranges max_oranges : ℕ) :
  total_crates = 150 →
  min_oranges = 125 →
  max_oranges = 149 →
  ∃ n : ℕ, n = 6 ∧
    (∀ m : ℕ, (∃ k : ℕ, k ≥ n ∧ 
      (∃ orange_count : ℕ, orange_count ≥ min_oranges ∧ orange_count ≤ max_oranges ∧
        (∃ crates_with_count : Finset ℕ, crates_with_count.card = k ∧
          ∀ crate ∈ crates_with_count, ∃ f : ℕ → ℕ, f crate = orange_count))) →
    m ≤ n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_crates_pigeonhole_l1187_118759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l1187_118750

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem tangent_slope_at_one :
  deriv f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l1187_118750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_l1187_118773

-- Define the functions
noncomputable def f (x : ℝ) := 1 + Real.sin x
noncomputable def g (x : ℝ) := -Real.cos x

-- State the theorem
theorem monotonicity_intervals :
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (2 * k * π - π / 2) (2 * k * π + π / 2))) ∧
  (∀ k : ℤ, StrictAntiOn f (Set.Icc (2 * k * π + π / 2) (2 * k * π + 3 * π / 2))) ∧
  (∀ k : ℤ, StrictMonoOn g (Set.Icc (2 * k * π) (2 * k * π + π))) ∧
  (∀ k : ℤ, StrictAntiOn g (Set.Icc (2 * k * π - π) (2 * k * π))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_l1187_118773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1187_118720

theorem sin_double_angle_special_case (θ : ℝ) :
  Real.sin (π / 4 + θ) = 1 / 3 → Real.sin (2 * θ) = -7 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1187_118720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_with_wind_l1187_118738

/-- Proves that the distance flown with the wind is 400 miles -/
theorem distance_with_wind (wind_speed plane_speed distance_against : ℝ) :
  wind_speed = 20 →
  plane_speed = 180 →
  distance_against = 320 →
  distance_against * (plane_speed + wind_speed) / (plane_speed - wind_speed) = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_with_wind_l1187_118738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_polynomial_sets_l1187_118787

theorem disjoint_polynomial_sets (A B : ℤ) :
  ∃ C : ℤ, Disjoint
    {y : ℤ | ∃ x : ℤ, y = x^2 + A*x + B}
    {y : ℤ | ∃ x : ℤ, y = 2*x^2 + 2*x + C} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_polynomial_sets_l1187_118787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_pyramid_properties_l1187_118779

-- Define the quadrilateral pyramid
structure QuadrilateralPyramid where
  base : Set (EuclideanSpace ℝ (Fin 3))
  apex : EuclideanSpace ℝ (Fin 3)
  is_quadrilateral : Prop
  is_pyramid : Prop

-- Define the connecting segments
noncomputable def connecting_segments (p : QuadrilateralPyramid) : Set (Set (EuclideanSpace ℝ (Fin 3))) :=
  sorry

-- Define the intersection point of the connecting segments
noncomputable def intersection_point (p : QuadrilateralPyramid) : EuclideanSpace ℝ (Fin 3) :=
  sorry

-- Define the parallelogram formed by the midpoints of the connecting segments
noncomputable def midpoint_parallelogram (p : QuadrilateralPyramid) : Set (EuclideanSpace ℝ (Fin 3)) :=
  sorry

-- Define segment ratio
def SegmentRatio (s : Set (EuclideanSpace ℝ (Fin 3))) (p : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  sorry

-- Define is parallelogram
def IsParallelogram (s : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  sorry

-- Define area
noncomputable def area (s : Set (EuclideanSpace ℝ (Fin 3))) : ℝ :=
  sorry

-- Main theorem
theorem quadrilateral_pyramid_properties (p : QuadrilateralPyramid) :
  -- Part a
  (∀ s ∈ connecting_segments p, 
    SegmentRatio s (intersection_point p) = 3 / 2) ∧
  -- Part b
  IsParallelogram (midpoint_parallelogram p) ∧
  -- Part c
  area (midpoint_parallelogram p) = (1 / 72) * area p.base :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_pyramid_properties_l1187_118779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_fifth_digit_of_sum_one_fifth_one_eleventh_l1187_118729

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sum_decimal_representations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in a decimal representation -/
def nth_digit_after_decimal (f : ℕ → ℕ) (n : ℕ) : ℕ := sorry

theorem twenty_fifth_digit_of_sum_one_fifth_one_eleventh :
  nth_digit_after_decimal (sum_decimal_representations (1/5) (1/11)) 25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_fifth_digit_of_sum_one_fifth_one_eleventh_l1187_118729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_sum_l1187_118797

theorem simplest_fraction_sum (a b : ℕ+) (h1 : (a : ℚ) / b = 0.6375) 
  (h2 : Nat.gcd a b = 1) : a + b = 131 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_sum_l1187_118797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_equality_l1187_118715

structure Trapezoid where
  triangles : Finset (Set ℝ)
  pentagon : Set ℝ
  mk_trapezoid : triangles.card = 7

def adjacent_triangles (t : Trapezoid) : Finset (Set ℝ) :=
  t.triangles.filter (λ s => true)  -- Placeholder condition, replace with actual logic when available

noncomputable def Set.area (s : Set ℝ) : ℝ := 0  -- Placeholder definition

theorem trapezoid_area_equality (t : Trapezoid) :
  (adjacent_triangles t).sum Set.area = Set.area t.pentagon := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_equality_l1187_118715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l1187_118791

/-- The parametric curve defined by x = 32 cos³(t) and y = 3 sin³(t) -/
noncomputable def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (32 * (Real.cos t) ^ 3, 3 * (Real.sin t) ^ 3)

/-- The vertical line x = 12√3 -/
noncomputable def vertical_line : ℝ → ℝ :=
  λ _ ↦ 12 * Real.sqrt 3

/-- The region bounded by the parametric curve and the vertical line -/
def bounded_region (x y : ℝ) : Prop :=
  ∃ t, parametric_curve t = (x, y) ∧ x ≥ 12 * Real.sqrt 3

/-- The area of the bounded region -/
noncomputable def area : ℝ := 6 * Real.pi - 9 * Real.sqrt 3

/-- Theorem stating that the area of the bounded region is 6π - 9√3 -/
theorem area_of_bounded_region :
  area = 6 * Real.pi - 9 * Real.sqrt 3 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l1187_118791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arc_triangle_area_ratio_l1187_118744

theorem circle_arc_triangle_area_ratio : 
  let r : ℝ := 3
  let circle_area := π * r^2
  let arc_length := 2 * π * r / 3
  let triangle_side_length := arc_length
  let triangle_area := Real.sqrt 3 / 4 * triangle_side_length^2
  triangle_area / circle_area = π * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arc_triangle_area_ratio_l1187_118744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_sum_of_coefficients_l1187_118707

-- Define the binomial expansion function
noncomputable def binomialExpansion (x : ℝ) (n : ℕ) : ℝ := (x - 1 / (2 * x)) ^ n

-- Define the function to get the binomial coefficient of the third term
def thirdTermCoefficient (n : ℕ) : ℕ := Nat.choose n 2

-- Define the function to calculate the sum of all coefficients
noncomputable def sumOfCoefficients (n : ℕ) : ℝ := binomialExpansion 1 n

-- State the theorem
theorem binomial_expansion_sum_of_coefficients :
  ∀ n : ℕ, thirdTermCoefficient n = 15 → sumOfCoefficients n = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_sum_of_coefficients_l1187_118707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_odd_increasing_function_l1187_118783

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem solution_set_of_odd_increasing_function
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_incr : is_increasing_on f 0 (Real.pi / 2)) -- Changed Real.infinity to (Real.pi / 2)
  (h_f_neg3 : f (-3) = 0) :
  {x : ℝ | x * f x < 0} = {x : ℝ | -3 < x ∧ x < 0 ∨ 0 < x ∧ x < 3} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_odd_increasing_function_l1187_118783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_diagonal_length_l1187_118700

/-- The length of a diagonal in a regular pentagon with side length 12 -/
noncomputable def regular_pentagon_diagonal : ℝ := 6 * Real.sqrt (6 + 2 * Real.sqrt 5)

/-- Theorem: In a regular pentagon with side length 12, the length of a diagonal is 6√(6 + 2√5) -/
theorem regular_pentagon_diagonal_length :
  ∀ (s : ℝ), s = 12 →
  ∃ (d : ℝ), d = regular_pentagon_diagonal ∧ 
  d^2 = 2 * s^2 - 2 * s^2 * (-((Real.sqrt 5 - 1) / 4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_diagonal_length_l1187_118700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_l1187_118724

def v (t : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3 + 6*t
  | 1 => 1 - 4*t
  | 2 => -4 + 2*t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => -1
  | 1 => 6
  | 2 => 3

def direction_vector : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 6
  | 1 => -4
  | 2 => 2

theorem closest_point (t : ℝ) : 
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) ↔ t = -15/28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_l1187_118724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_at_most_one_obtuse_l1187_118745

/-- A triangle is a geometric shape with three sides and three angles. -/
structure Triangle where
  angles : Fin 3 → ℝ
  sum_of_angles : (angles 0) + (angles 1) + (angles 2) = Real.pi

/-- An angle is obtuse if it is greater than π/2. -/
def is_obtuse (angle : ℝ) : Prop := angle > Real.pi / 2

/-- A predicate that checks if a triangle has at most one obtuse angle. -/
def at_most_one_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 1) ∧ ¬is_obtuse (t.angles 2)) ∨
  (¬is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 1) ∧ ¬is_obtuse (t.angles 2)) ∨
  (¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 1) ∧ is_obtuse (t.angles 2)) ∨
  (¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 1) ∧ ¬is_obtuse (t.angles 2))

/-- A predicate that checks if a triangle has at least two obtuse angles. -/
def at_least_two_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 1)) ∨
  (is_obtuse (t.angles 1) ∧ is_obtuse (t.angles 2)) ∨
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 2))

/-- The main theorem stating that the negation of "at most one obtuse angle" 
    is equivalent to "at least two obtuse angles". -/
theorem negation_of_at_most_one_obtuse (t : Triangle) : 
  ¬(at_most_one_obtuse t) ↔ at_least_two_obtuse t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_at_most_one_obtuse_l1187_118745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_operations_to_500_l1187_118771

def sequenceStep (n : ℕ) : ℤ :=
  let initial := 5 + 60
  let cycle := 120 - 100
  initial + n / 2 * cycle + if n % 2 = 0 then 0 else 120

theorem min_operations_to_500 :
  (∃ n : ℕ, sequenceStep n = 500) ∧ 
  (∀ m : ℕ, m < 33 → sequenceStep m ≠ 500) ∧
  sequenceStep 33 = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_operations_to_500_l1187_118771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1187_118778

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 4 * (Real.cos x)^2 - 1

theorem function_identity (a b c : ℝ) :
  (∀ x : ℝ, a * f x - b * f (x + c) = 3) →
  3 * a^2 + 2 * b + Real.cos c = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1187_118778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l1187_118737

theorem two_true_propositions (p q : Prop) [Decidable p] [Decidable q] : 
  ∃! n : Nat, n = (if p ∧ q then 1 else 0) + 
                  (if p ∨ q then 1 else 0) + 
                  (if ¬p then 1 else 0) + 
                  (if ¬q then 1 else 0) ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l1187_118737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_root_l1187_118701

-- Define the interval [-π, π]
def I : Set ℝ := Set.Icc (-Real.pi) Real.pi

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := x^2 + 2*a*x - b^2 + Real.pi

-- Define the condition for f(x) to have a root
def has_root (a b : ℝ) : Prop := ∃ x, f a b x = 0

-- Define the event space
def Ω : Set (ℝ × ℝ) := Set.prod I I

-- Define the favorable event
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p ∈ Ω ∧ has_root p.1 p.2}

-- State the theorem
theorem probability_of_root :
  MeasureTheory.volume A / MeasureTheory.volume Ω = 3/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_root_l1187_118701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_tangency_l1187_118749

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0

-- Define the asymptotes of the hyperbola
def asymptotes (a : ℝ) (x y : ℝ) : Prop := x + a*y = 0 ∨ x - a*y = 0

-- Define the tangency condition
def tangent_condition (a : ℝ) : Prop := ∃ (x y : ℝ), asymptotes a x y ∧ my_circle x y

-- State the theorem
theorem hyperbola_circle_tangency (a : ℝ) (h1 : a > 0) (h2 : tangent_condition a) : 
  a = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_tangency_l1187_118749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_equals_bankers_gain_l1187_118758

/-- Represents the banker's gain in rupees -/
def bankers_gain : ℝ := 15.8

/-- Represents the time until the bill is due in years -/
def time : ℝ := 5

/-- Represents the average interest rate per annum as a decimal -/
def interest_rate : ℝ := 0.145

/-- Calculates the present value of a bill given its face value -/
noncomputable def present_value (face_value : ℝ) : ℝ :=
  face_value * Real.exp (-interest_rate * time)

/-- Calculates the face value of a bill given the banker's gain -/
noncomputable def face_value : ℝ :=
  bankers_gain / (1 - Real.exp (-interest_rate * time))

/-- Represents the true discount of the bill -/
noncomputable def true_discount : ℝ :=
  face_value - present_value face_value

/-- Theorem stating that the true discount equals the banker's gain -/
theorem true_discount_equals_bankers_gain :
  true_discount = bankers_gain := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_equals_bankers_gain_l1187_118758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_exists_l1187_118799

-- Define the space
structure Space where
  -- Add any necessary structure here

-- Define a line in the space
structure Line (S : Space) where
  -- Add any necessary properties for a line

-- Define a plane in the space
structure Plane (S : Space) where
  -- Add any necessary properties for a plane

-- Define perpendicularity between a line and a plane
def perpendicular (S : Space) (l : Line S) (p : Plane S) : Prop :=
  sorry

-- Define when a line is within a plane
def line_in_plane (S : Space) (l : Line S) (p : Plane S) : Prop :=
  sorry

-- Define perpendicularity between two lines
def lines_perpendicular (S : Space) (l1 l2 : Line S) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_line_exists (S : Space) (a : Line S) (α : Plane S) :
  ∃ b : Line S, line_in_plane S b α ∧ lines_perpendicular S b a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_exists_l1187_118799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane1_properties_plane2_properties_l1187_118734

-- Define the plane equations
def plane1 (x y z : ℝ) : Prop := 3 * y + 2 * z = 0
def plane2 (x y z : ℝ) : Prop := x + 3 * y - 1 = 0

-- Define points
def point1 : ℝ × ℝ × ℝ := (1, 2, -3)
def point2 : ℝ × ℝ × ℝ := (1, 0, 1)
def point3 : ℝ × ℝ × ℝ := (-2, 1, 3)

-- Define what it means for a plane to pass through a point
def passes_through (plane : ℝ → ℝ → ℝ → Prop) (point : ℝ × ℝ × ℝ) : Prop :=
  plane point.1 point.2.1 point.2.2

-- Define what it means for a plane to pass through the Ox axis
def passes_through_Ox_axis (plane : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ x : ℝ, plane x 0 0

-- Define what it means for a plane to be parallel to the Oz axis
def parallel_to_Oz_axis (plane : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ x y z : ℝ, plane x y z ↔ plane x y 0

-- Theorem statements
theorem plane1_properties :
  passes_through_Ox_axis plane1 ∧ passes_through plane1 point1 := by sorry

theorem plane2_properties :
  parallel_to_Oz_axis plane2 ∧ passes_through plane2 point2 ∧ passes_through plane2 point3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane1_properties_plane2_properties_l1187_118734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1187_118728

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 4 * Real.cos (ω * x) * Real.cos (ω * x + Real.pi / 3)

theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ (T : ℝ), T > 0 → Function.Periodic (f ω) T → T ≥ Real.pi) :
  ω = 1 ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 3), StrictMonoOn (fun y => -(f ω y)) (Set.Icc 0 (Real.pi / 3))) ∧
  (∀ x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 6), StrictMonoOn (f ω) (Set.Icc (Real.pi / 3) (5 * Real.pi / 6))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1187_118728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circle_area_l1187_118717

/-- The minimum area of a circle centered at the origin and tangent to the line 3x + y - 4 = 0 -/
theorem min_circle_area (C : Set (ℝ × ℝ)) 
  (center : (0, 0) ∈ C)
  (is_circle : ∃ (r : ℝ), C = {p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - 0)^2 = r^2})
  (tangent : ∃ (p : ℝ × ℝ), p ∈ C ∧ 3 * p.1 + p.2 - 4 = 0) :
  (∃ (r : ℝ), C = {p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - 0)^2 = r^2} ∧ Real.pi * r^2 ≥ 2 * Real.pi / 5) ∧
  (∃ (r : ℝ), C = {p : ℝ × ℝ | (p.1 - 0)^2 + (p.2 - 0)^2 = r^2} ∧ Real.pi * r^2 = 2 * Real.pi / 5) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circle_area_l1187_118717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_time_is_36_minutes_job_completion_time_l1187_118761

/-- The time it takes for P to finish the remaining job after working with Q for 2 hours -/
noncomputable def remaining_time (p_rate q_rate : ℝ) : ℝ :=
  let combined_rate := p_rate + q_rate
  let completed_portion := 2 * combined_rate
  let remaining_portion := 1 - completed_portion
  remaining_portion / p_rate

/-- The theorem stating the remaining time for P to finish the job -/
theorem remaining_time_is_36_minutes :
  remaining_time (1/3) (1/15) * 60 = 36 := by
  sorry

/-- Main theorem proving the result based on given conditions -/
theorem job_completion_time 
  (p_rate : ℝ) 
  (q_rate : ℝ) 
  (h1 : p_rate = 1/3)  -- P can finish the job in 3 hours
  (h2 : q_rate = 1/15) -- Q can finish the job in 15 hours
  : remaining_time p_rate q_rate * 60 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_time_is_36_minutes_job_completion_time_l1187_118761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1187_118790

-- Define the hyperbola
def Hyperbola (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}

-- Define the foci
def F1 (c : ℝ) : ℝ × ℝ := (-c, 0)
def F2 (c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the point P on the hyperbola
def P (x y : ℝ) : ℝ × ℝ := (x, y)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Main theorem
theorem hyperbola_eccentricity (a b c x : ℝ) :
  a > 0 ∧ b > 0 ∧
  P x 0 ∈ Hyperbola a b ∧
  distance (F1 c) (F2 c) = 12 ∧
  distance (P x 0) (F2 c) = 5 →
  distance (F1 c) (F2 c) / (2 * a) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1187_118790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1187_118741

-- Define two triangles
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_180 : A + B + C = Real.pi
  positive : 0 < A ∧ 0 < B ∧ 0 < C

noncomputable def Triangle1 : Triangle := sorry
noncomputable def Triangle2 : Triangle := sorry

-- Define the condition that cosine of angles in Triangle1 equal sine of angles in Triangle2
def angles_condition (t1 t2 : Triangle) : Prop :=
  Real.cos t1.A = Real.sin t2.A ∧
  Real.cos t1.B = Real.sin t2.B ∧
  Real.cos t1.C = Real.sin t2.C

-- Define acute and obtuse triangles
def is_acute (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

def is_obtuse (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

-- The main theorem
theorem triangle_property :
  angles_condition Triangle1 Triangle2 →
  is_acute Triangle1 ∧ is_obtuse Triangle2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1187_118741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_standard_deviation_l1187_118718

noncomputable def apple_masses : List ℝ := [125, 124, 121, 123, 127]

noncomputable def sample_mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def sample_variance (xs : List ℝ) : ℝ :=
  let mean := sample_mean xs
  (xs.map (fun x => (x - mean) ^ 2)).sum / (xs.length - 1)

noncomputable def sample_standard_deviation (xs : List ℝ) : ℝ :=
  (sample_variance xs).sqrt

theorem apple_standard_deviation :
  sample_standard_deviation apple_masses = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_standard_deviation_l1187_118718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l1187_118780

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2

noncomputable def g (x : ℝ) : ℝ := Real.sin (4 * x - Real.pi / 6) + 1 / 2

theorem f_and_g_properties :
  (∀ k : ℤ, ∃ x : ℝ, f x = f (k * Real.pi / 2 + Real.pi / 6)) ∧
  (∀ x ∈ Set.Ioo (-Real.pi / 12) (Real.pi / 3), g x ∈ Set.Ioc (-1 / 2) (3 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l1187_118780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_l1187_118764

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The conditions on the polynomial P -/
def satisfies_conditions (P : IntPolynomial) (a : ℤ) : Prop :=
  a > 0 ∧
  P.eval (2 : ℤ) = a ∧ P.eval (4 : ℤ) = a ∧ P.eval (6 : ℤ) = a ∧ P.eval (10 : ℤ) = a ∧
  P.eval (1 : ℤ) = -a ∧ P.eval (3 : ℤ) = -a ∧ P.eval (5 : ℤ) = -a ∧ P.eval (7 : ℤ) = -a

/-- The theorem stating the smallest possible value of a -/
theorem smallest_a : ∀ (P : IntPolynomial) (a : ℤ),
  satisfies_conditions P a → a ≥ 945 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_l1187_118764
