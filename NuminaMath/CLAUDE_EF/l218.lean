import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_existence_l218_21846

-- Define the circle
variable (k : Set (ℝ × ℝ))

-- Define points on the circle
variable (A B C : ℝ × ℝ)

-- Define the property of being on the circle
def on_circle (p : ℝ × ℝ) (k : Set (ℝ × ℝ)) : Prop := p ∈ k

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem inscribed_circle_existence :
  on_circle A k → on_circle B k → on_circle C k →
  ∃ D : ℝ × ℝ, on_circle D k ∧ 
    distance A B + distance C D = distance A D + distance B C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_existence_l218_21846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_relationships_l218_21868

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity between a line and another line
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of perpendicularity

-- Define parallel, intersecting, and skew relationships between lines
def parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of parallel lines

def intersecting (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of intersecting lines

def skew (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of skew lines

-- Theorem statement
theorem perpendicular_lines_relationships (l1 l2 l3 : Line3D) 
  (h1 : perpendicular l1 l3) (h2 : perpendicular l2 l3) :
  (parallel l1 l2 ∨ intersecting l1 l2 ∨ skew l1 l2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_relationships_l218_21868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_pigeonhole_l218_21892

theorem friend_pigeonhole (n : ℕ) (h : n ≥ 2) :
  ∃ (friendship : Fin n → Fin n → Bool),
    (∀ i j : Fin n, friendship i j = friendship j i) ∧  -- Friendship is symmetric
    (∀ i : Fin n, friendship i i = false) →              -- No one is their own friend
    ∃ i j : Fin n, i ≠ j ∧
      (Finset.filter (fun k => friendship i k) (Finset.univ : Finset (Fin n))).card =
      (Finset.filter (fun k => friendship j k) (Finset.univ : Finset (Fin n))).card :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_pigeonhole_l218_21892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l218_21824

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def areCollinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b ∨ b = k • a

/-- The magnitude (length) of a vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_collinearity (a b : ℝ × ℝ) :
  magnitude (a - b) = magnitude a + magnitude b → areCollinear a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l218_21824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_composition_unique_root_l218_21889

theorem sin_composition_unique_root :
  ∃! x : ℝ, Real.sin (Real.sin x) = x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_composition_unique_root_l218_21889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersecting_3subsets_correct_l218_21830

def max_intersecting_3subsets (n : ℕ) : ℕ :=
  match n with
  | 3 => 1
  | 4 => 4
  | 5 => 10
  | m => Nat.choose (m - 1) 2

theorem max_intersecting_3subsets_correct (n : ℕ) :
  max_intersecting_3subsets n = 
    if n < 6 then
      match n with
      | 3 => 1
      | 4 => 4
      | 5 => 10
      | _ => 0  -- This case should never occur due to the condition n < 6
    else
      Nat.choose (n - 1) 2 := by
  sorry  -- Proof omitted

#check max_intersecting_3subsets
#check max_intersecting_3subsets_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersecting_3subsets_correct_l218_21830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l218_21878

noncomputable def f (x : Real) : Real := 3 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2

theorem triangle_properties (a b c A B C : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- If f(A) = 5, then A = π/3
  (f A = 5 → A = π/3) ∧
  -- If a = 2, then the maximum possible area of triangle ABC is √3
  (a = 2 → (∀ area : Real, area ≤ Real.sqrt 3 ∧ 
    ∃ b' c' : Real, area = (1/2) * b' * c' * Real.sin A ∧ 
    2^2 = b'^2 + c'^2 - 2*b'*c'*Real.cos A)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l218_21878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l218_21859

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- A line passing through the right focus of the hyperbola -/
structure FocusLine where
  slope : ℝ

theorem hyperbola_eccentricity_range (h : Hyperbola) 
  (l1 : FocusLine) (l2 : FocusLine)
  (h_slope1 : l1.slope = 1)
  (h_slope2 : l2.slope = 3)
  (h_intersect1 : ∃ x y, x^2 / h.a^2 - y^2 / h.b^2 = 1 ∧ y = l1.slope * (x - h.a * eccentricity h))
  (h_intersect2 : ∃ x1 y1 x2 y2, 
    x1^2 / h.a^2 - y1^2 / h.b^2 = 1 ∧ 
    x2^2 / h.a^2 - y2^2 / h.b^2 = 1 ∧ 
    y1 = l2.slope * (x1 - h.a * eccentricity h) ∧
    y2 = l2.slope * (x2 - h.a * eccentricity h) ∧
    x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2) :
  Real.sqrt 2 < eccentricity h ∧ eccentricity h < Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l218_21859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_formula_l218_21810

/-- The Fibonacci sequence -/
def fib : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The conjugate of the golden ratio -/
noncomputable def ψ : ℝ := (1 - Real.sqrt 5) / 2

/-- Properties of φ and ψ -/
axiom φ_square : φ^2 = φ + 1
axiom ψ_square : ψ^2 = ψ + 1

/-- The main theorem -/
theorem fib_formula (n : ℕ) : (fib n : ℝ) = (φ^n - ψ^n) / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_formula_l218_21810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l218_21806

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / (x - 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > -1 ∧ x ≠ 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l218_21806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_induction_step_l218_21850

/-- The product of consecutive integers from n+1 to n+n -/
def leftSide (n : ℕ) : ℕ := (n + 1).factorial / n.factorial

/-- The given equation holds for all positive natural numbers -/
axiom equation_holds (n : ℕ) : n > 0 → leftSide n = 2^n * (2 * n - 1).factorial / n.factorial

/-- The ratio of leftSide for consecutive positive integers -/
def leftSideRatio (k : ℕ) : ℚ := leftSide (k + 1) / leftSide k

theorem induction_step (k : ℕ) (h : k > 0) : leftSideRatio k = 2 * (2 * k + 1) / (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_induction_step_l218_21850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_composition_l218_21847

/-- Given the composition of two alloys and their mixture, prove the ratio of lead to tin in Alloy A -/
theorem alloy_composition (alloy_A_mass alloy_B_mass new_alloy_tin_mass : ℝ)
  (alloy_B_tin_ratio alloy_B_copper_ratio : ℝ) :
  alloy_A_mass = 100 →
  alloy_B_mass = 200 →
  new_alloy_tin_mass = 117.5 →
  alloy_B_tin_ratio = 2 →
  alloy_B_copper_ratio = 3 →
  (let alloy_B_tin_mass := (alloy_B_tin_ratio / (alloy_B_tin_ratio + alloy_B_copper_ratio)) * alloy_B_mass
   let alloy_A_tin_mass := new_alloy_tin_mass - alloy_B_tin_mass
   let alloy_A_lead_mass := alloy_A_mass - alloy_A_tin_mass
   alloy_A_lead_mass / alloy_A_tin_mass = 5 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_composition_l218_21847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_19_l218_21894

/-- Function to construct the number with n threes inserted between zeros in 12008 -/
def insert_threes (n : ℕ) : ℕ :=
  12000 * 10^(n + 1) + 3 * ((10^n - 1) / 9) * 10 + 8

/-- Theorem stating that the constructed number is always divisible by 19 -/
theorem divisible_by_19 (n : ℕ) : 19 ∣ insert_threes n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_19_l218_21894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_area_for_specific_rectangles_l218_21840

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The minimum square area that can contain two non-overlapping rectangles -/
noncomputable def minSquareArea (r1 r2 : Rectangle) : ℝ :=
  (max (r1.width + r2.width) (r1.height + r2.height)) ^ 2

/-- Theorem stating the minimum square area for specific rectangles -/
theorem min_square_area_for_specific_rectangles :
  let r1 : Rectangle := ⟨3, 4⟩
  let r2 : Rectangle := ⟨4, 5⟩
  minSquareArea r1 r2 = 81 := by
  sorry

#check min_square_area_for_specific_rectangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_area_for_specific_rectangles_l218_21840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l218_21822

def a : ℕ → ℚ
| 0 => -1/2
| n + 1 => -1 / (a n + 2)

def b (n : ℕ) : ℚ := 1 / (a n + 1)

theorem b_formula (n : ℕ) : b n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l218_21822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_gain_percentage_difference_l218_21839

/-- The percentage difference in gain between two selling prices -/
noncomputable def percentage_difference_in_gain (cost_price selling_price_1 selling_price_2 : ℝ) : ℝ :=
  let gain_1 := selling_price_1 - cost_price
  let gain_2 := selling_price_2 - cost_price
  ((gain_2 - gain_1) / gain_1) * 100

/-- Theorem stating the percentage difference in gain for the given problem -/
theorem article_gain_percentage_difference :
  percentage_difference_in_gain 200 340 350 = (1 / 14) * 100 := by
  sorry

#eval (1 / 14 : ℚ) * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_gain_percentage_difference_l218_21839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_intersection_l218_21845

/-- The distance between two points in a 2D plane -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The angle between three points in a 2D plane -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  Real.arccos ((distance p1 p2)^2 + (distance p2 p3)^2 - (distance p1 p3)^2) /
    (2 * distance p1 p2 * distance p2 p3)

/-- Theorem: Conditions for intersection of ellipse and circle -/
theorem ellipse_circle_intersection
  (F₁ F₂ Q : ℝ × ℝ) -- F₁, F₂ are foci, Q is an intersection point
  (a : ℝ) -- Half of the major axis length
  (h1 : distance F₁ Q + distance Q F₂ = 2 * a) -- Ellipse condition
  (h2 : 2 * a * Real.sin ((angle F₁ Q F₂) / 2) ≤ distance F₁ F₂) -- Lower bound
  (h3 : distance F₁ F₂ < 2 * a) -- Upper bound
  : True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_intersection_l218_21845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_right_angle_l218_21834

/-- Represents the angle between the hour hand and minute hand at a given time --/
def angleBetweenHands (hours : ℝ) (minutes : ℝ) : ℝ := 
  abs (30 * hours - 5.5 * minutes)

/-- The time in minutes past 2 o'clock when the hands form a right angle --/
noncomputable def rightAngleTime : ℝ := 300 / 11

theorem clock_right_angle :
  2 < 2 + rightAngleTime / 60 ∧
  2 + rightAngleTime / 60 < 3 ∧
  angleBetweenHands (2 + rightAngleTime / 60) rightAngleTime = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_right_angle_l218_21834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_sphere_to_hemisphere_l218_21814

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Define the volume of a hemisphere
noncomputable def hemisphere_volume (r : ℝ) : ℝ := (1/2) * (4/3) * Real.pi * r^3

-- Theorem statement
theorem volume_ratio_sphere_to_hemisphere (r : ℝ) (h : r > 0) :
  (sphere_volume r) / (hemisphere_volume (3 * r)) = 1 / 13.5 :=
by
  -- Unfold the definitions
  unfold sphere_volume hemisphere_volume
  -- Simplify the expression
  simp [Real.pi_pos, h]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_sphere_to_hemisphere_l218_21814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_approx_5_53_l218_21800

/-- Calculate the distance covered given speed and time -/
noncomputable def distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Convert minutes to hours -/
noncomputable def minutesToHours (minutes : ℝ) : ℝ :=
  minutes / 60

/-- Calculate total distance covered by a man walking uphill and downhill -/
noncomputable def totalDistance (uphillSpeed uphillTime downhillSpeed downhillTime : ℝ) : ℝ :=
  let uphillDistance := distance uphillSpeed (minutesToHours uphillTime)
  let downhillDistance := distance downhillSpeed (minutesToHours downhillTime)
  uphillDistance + downhillDistance

/-- Theorem stating that the total distance covered is approximately 5.53 km -/
theorem total_distance_approx_5_53 :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |totalDistance 8 25 12 11 - 5.53| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_approx_5_53_l218_21800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yadav_finances_theorem_l218_21863

/-- Represents Mr. Yadav's financial situation -/
structure YadavFinances where
  monthly_salary : ℚ
  consumable_percentage : ℚ
  yearly_savings : ℚ
  monthly_clothes_transport : ℚ

/-- Calculates the percentage of remaining salary spent on clothes and transport -/
def remaining_salary_percentage (y : YadavFinances) : ℚ :=
  (y.monthly_clothes_transport / ((1 - y.consumable_percentage) * y.monthly_salary)) * 100

/-- Theorem stating that the percentage of remaining salary spent on clothes and transport is 50% -/
theorem yadav_finances_theorem (y : YadavFinances) 
  (h1 : y.consumable_percentage = 6/10)
  (h2 : y.yearly_savings = 46800)
  (h3 : y.monthly_clothes_transport = 3900) :
  remaining_salary_percentage y = 50 := by
  sorry

#eval remaining_salary_percentage { 
  monthly_salary := 19500, 
  consumable_percentage := 6/10, 
  yearly_savings := 46800, 
  monthly_clothes_transport := 3900 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yadav_finances_theorem_l218_21863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_range_theorem_l218_21829

-- Define the custom operation
noncomputable def custom_op (a b : ℝ) : ℝ :=
  if a > b then a * b + b else a * b - b

-- Theorem statement
theorem custom_op_range_theorem :
  ∀ x : ℝ, custom_op 3 (x + 2) > 0 ↔ (-2 < x ∧ x < 1) ∨ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_range_theorem_l218_21829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_l218_21855

/-- The total number of products -/
def total_products : ℕ := 7

/-- The number of genuine products -/
def genuine_products : ℕ := 4

/-- The number of defective products -/
def defective_products : ℕ := 3

/-- Event A: Selecting a defective product on the first draw -/
noncomputable def event_A : ℝ := defective_products / total_products

/-- Event B: Selecting a genuine product on the second draw -/
noncomputable def event_B : ℝ := genuine_products / (total_products - 1)

/-- The probability of selecting a genuine product on the second draw 
    given that a defective product was selected on the first draw -/
theorem conditional_probability : 
  (event_A * event_B) / event_A = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_l218_21855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workout_solution_exists_l218_21896

/-- Represents John's workout parameters -/
structure WorkoutParams where
  bikeDistance : ℝ
  runDistance : ℝ
  totalTime : ℝ
  transitionTime : ℝ

/-- Calculates the time spent biking -/
noncomputable def bikeTime (x : ℝ) (params : WorkoutParams) : ℝ :=
  params.bikeDistance / (3 * x + 2)

/-- Calculates the time spent running -/
noncomputable def runTime (x : ℝ) (params : WorkoutParams) : ℝ :=
  params.runDistance / (1.2 * x)

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- Theorem stating the existence of a solution for John's workout -/
theorem workout_solution_exists (params : WorkoutParams)
  (h1 : params.bikeDistance = 30)
  (h2 : params.runDistance = 10)
  (h3 : params.totalTime = 3)
  (h4 : params.transitionTime = 1/6) :
  ∃ x : ℝ,
    bikeTime x params + runTime x params = params.totalTime - params.transitionTime ∧
    roundToHundredth (1.2 * x) = 9.73 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workout_solution_exists_l218_21896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CaOH2_l218_21841

/-- Represents the mass percentage of an element in a compound -/
noncomputable def mass_percentage (element_mass : ℝ) (compound_mass : ℝ) : ℝ :=
  (element_mass / compound_mass) * 100

/-- The molar mass of calcium (Ca) in g/mol -/
def molar_mass_Ca : ℝ := 40.08

/-- The molar mass of oxygen (O) in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- The molar mass of hydrogen (H) in g/mol -/
def molar_mass_H : ℝ := 1.01

/-- The molar mass of calcium hydroxide Ca(OH)₂ in g/mol -/
def molar_mass_CaOH2 : ℝ := molar_mass_Ca + 2 * molar_mass_O + 2 * molar_mass_H

/-- The mass of oxygen in calcium hydroxide Ca(OH)₂ in g/mol -/
def mass_O_in_CaOH2 : ℝ := 2 * molar_mass_O

theorem mass_percentage_O_in_CaOH2 :
  ∃ ε > 0, |mass_percentage mass_O_in_CaOH2 molar_mass_CaOH2 - 43.19| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CaOH2_l218_21841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_apples_grams_l218_21804

/-- Calculates the amount of apples Susan has after sharing with her friends -/
noncomputable def susan_apples_after_sharing (phillip_apples : ℝ) (ben_extra : ℝ) (apple_weight : ℝ) 
  (tom_ratio : ℝ) (tom_extra_per_two : ℝ) (susan_extra : ℝ) (susan_share_ratio : ℝ) : ℝ :=
  let ben_apples := phillip_apples + ben_extra
  let tom_apples := tom_ratio * ben_apples
  let susan_apples := (1/2) * tom_apples + susan_extra
  let susan_apples_after := susan_apples * (1 - susan_share_ratio)
  susan_apples_after * apple_weight

/-- Theorem stating that Susan has 2128.359375 grams of apples after sharing -/
theorem susan_apples_grams : 
  susan_apples_after_sharing 38.25 8.5 150 (3/8) 75 7 0.1 = 2128.359375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_apples_grams_l218_21804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_passing_time_l218_21866

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_hr_to_m_s : ℝ := 1 / 3.6

/-- Calculate time (in seconds) for a train to pass a point -/
noncomputable def train_passing_time (length : ℝ) (speed_km_hr : ℝ) : ℝ :=
  length / (speed_km_hr * km_hr_to_m_s)

/-- Properties of the three trains -/
def train1_length : ℝ := 180
def train1_speed : ℝ := 36
def train2_length : ℝ := 250
def train2_speed : ℝ := 45
def train3_length : ℝ := 320
def train3_speed : ℝ := 54

/-- Total time for all trains to pass -/
noncomputable def total_passing_time : ℝ :=
  train_passing_time train1_length train1_speed +
  train_passing_time train2_length train2_speed +
  train_passing_time train3_length train3_speed

theorem trains_passing_time :
  |total_passing_time - 59.33| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_passing_time_l218_21866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l218_21888

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + Real.sqrt 2 * t, Real.sqrt 2 * t)

noncomputable def curve_C (θ : ℝ) : ℝ := Real.sin θ / (1 - Real.sin θ ^ 2)

noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - y - 1| / Real.sqrt 2

theorem min_distance_to_line : 
  ∃ (x y : ℝ), x ≠ 0 ∧ y = x^2 ∧ 
  (∀ (x' y' : ℝ), x' ≠ 0 → y' = x'^2 → 
    distance_to_line x y ≤ distance_to_line x' y') ∧
  distance_to_line x y = 3 * Real.sqrt 2 / 8 ∧
  x = 1/2 ∧ y = 1/4 := by
  sorry

#check min_distance_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l218_21888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l218_21873

/-- Represents a right cone -/
structure Cone where
  baseCircumference : ℝ
  height : ℝ

/-- The volume of a cone -/
noncomputable def Cone.volume (c : Cone) : ℝ :=
  (1 / 3) * Real.pi * ((c.baseCircumference / (2 * Real.pi))^2) * c.height

theorem cone_height_ratio (originalCone shorterCone : Cone) :
  originalCone.baseCircumference = 20 * Real.pi →
  originalCone.height = 40 →
  shorterCone.baseCircumference = originalCone.baseCircumference →
  shorterCone.volume = 320 * Real.pi →
  shorterCone.height / originalCone.height = 6 / 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l218_21873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_proof_l218_21864

/-- Given a function f such that f(x+1) = 2x + 4 for all x, 
    and f(a) = 8, prove that a = 3 -/
theorem function_value_proof (f : ℝ → ℝ) (a : ℝ) 
    (h1 : ∀ x, f (x + 1) = 2 * x + 4) 
    (h2 : f a = 8) : 
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_proof_l218_21864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l218_21881

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

-- Define the line
def line (x y : ℝ) : ℝ := 2*x + y - 10

-- Define the distance from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := |line x y| / Real.sqrt 5

-- Theorem statement
theorem min_distance_to_line :
  ∀ x y : ℝ, is_on_ellipse x y → 
  (∀ x' y' : ℝ, is_on_ellipse x' y' → distance_to_line x y ≤ distance_to_line x' y') ∧
  distance_to_line x y = Real.sqrt 5 :=
by
  sorry

#check min_distance_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l218_21881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_real_and_imag_parts_of_z_l218_21833

-- Define the complex number z
noncomputable def z : ℂ := (2 + Complex.I) / (2 - Complex.I) - Complex.I

-- Theorem statement
theorem sum_of_real_and_imag_parts_of_z :
  z.re + z.im = 2/5 := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_real_and_imag_parts_of_z_l218_21833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_complex_fraction_pure_imaginary_l218_21813

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- i is the square root of -1 -/
theorem i_squared : i * i = -1 := Complex.I_mul_I

/-- Definition of a pure imaginary number -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_fraction_pure_imaginary (a : ℝ) : 
  is_pure_imaginary ((a + i) / (1 - i)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_complex_fraction_pure_imaginary_l218_21813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l218_21875

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (60 - x)) + Real.sqrt (x * (3 - x))

theorem g_max_value :
  ∃ (N : ℝ) (x₁ : ℝ), 
    (∀ x, 0 ≤ x ∧ x ≤ 3 → g x ≤ N) ∧
    (0 ≤ x₁ ∧ x₁ ≤ 3) ∧
    g x₁ = N ∧
    N = 6 * Real.sqrt 5 ∧
    x₁ = 15 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l218_21875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_initial_interval_l218_21862

noncomputable def f (x : ℝ) : ℝ := 2^x - 3

theorem bisection_initial_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 ∧
  (∀ x ∈ Set.Ioo 1 2, ∀ y ∈ Set.Ioo 1 2, x < y → f x < f y) ∧
  f 1 < 0 ∧ f 2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_initial_interval_l218_21862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_operations_l218_21893

theorem arithmetic_operations : 
  (-7 : ℤ) + 5 - (-10) = 8 ∧ (3 : ℚ) / (-1/3) * (-3) = 27 := by
  -- Split the conjunction
  constructor
  -- Prove the first part
  · ring
  -- Prove the second part
  · field_simp
    ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_operations_l218_21893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_divides_n2n_plus_1_l218_21808

theorem infinitely_many_n_divides_n2n_plus_1 (p : Nat) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, p ∣ (f k) * 2^(f k) + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_divides_n2n_plus_1_l218_21808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l218_21858

theorem power_equality (x : ℝ) : (2 : ℝ)^10 = (32 : ℝ)^x → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l218_21858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l218_21869

-- Define the functions and constant
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := x^a

-- State the theorem
theorem function_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  h a 2 > f a 2 ∧ f a 2 > g a 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l218_21869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_upper_bound_l218_21837

/-- Sum of digits function -/
def S (x : ℕ) : ℕ := sorry

/-- Sequence definition -/
def x (k : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => S (k * x k n)

/-- Main theorem -/
theorem x_upper_bound (k : ℕ) (h : k > 0) (n : ℕ) :
  x k n < 27 * Real.sqrt (k : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_upper_bound_l218_21837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_relation_l218_21891

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - y + Real.sqrt 2 - 1 = 0

-- Define the slope of a line
noncomputable def line_slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

theorem ellipse_relation (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : eccentricity a b = Real.sqrt 6 / 3)
  (h4 : ∃ x y, circle_eq x y ∧ tangent_line x y)
  (m n : ℝ) (h5 : m ≠ 3)
  (h6 : ∃ x1 y1 x2 y2, ellipse x1 y1 a b ∧ ellipse x2 y2 a b)
  (h7 : ∃ k1 k2 k3, 
    k1 = line_slope x1 y1 3 2 ∧
    k2 = line_slope 3 2 m n ∧
    k3 = line_slope x2 y2 3 2 ∧
    k1 + k3 = 2 * k2) :
  m - n - 1 = 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_relation_l218_21891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l218_21882

def a : Fin 2 → ℝ := ![4, 3]
def b : Fin 2 → ℝ := ![-1, 2]

def m (l : ℝ) : Fin 2 → ℝ := fun i => a i - l * b i
def n : Fin 2 → ℝ := fun i => a i + b i

def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

theorem perpendicular_vectors_lambda (l : ℝ) : 
  dot_product (m l) n = 0 → l = 27/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l218_21882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_properties_l218_21851

noncomputable def f (x : ℝ) := Real.cos x ^ 2

theorem cos_squared_properties :
  (∀ x, f x = f (-x)) ∧
  (∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ π) ∧
  (∀ x, f (x + π) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_properties_l218_21851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_odd_function_l218_21853

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/a - 1/(a^x + 1)

theorem range_of_odd_function (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, f a x = -f a (-x)) : 
  Set.range (f a) = Set.Ioo (-1/2) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_odd_function_l218_21853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_cost_theorem_l218_21820

/-- Represents the cost of tickets for a baseball game -/
structure TicketCost where
  adult : ℚ
  child : ℚ

/-- Calculates the total cost for a given number of adult and child tickets -/
def totalCost (t : TicketCost) (numAdult : ℕ) (numChild : ℕ) : ℚ :=
  t.adult * numAdult + t.child * numChild

/-- Theorem stating the cost of 10 adult and 15 child tickets given the conditions -/
theorem ticket_cost_theorem (t : TicketCost) 
    (h1 : t.child = t.adult / 2)
    (h2 : totalCost t 6 8 = 46.50) :
    totalCost t 10 15 = 81.375 := by
  sorry

#eval (81.375 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_cost_theorem_l218_21820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_in_cylinder_l218_21880

/-- Given a cylindrical jar with base radius R and a sphere of radius r,
    if the water surface becomes tangent to the sphere when placed in the jar,
    then the radius x of a new sphere that produces the same effect satisfies
    certain conditions. -/
theorem sphere_in_cylinder (R r : ℝ) (h_pos_R : R > 0) (h_pos_r : r > 0) :
  let x := (-r + Real.sqrt (6 * R^2 - 3 * r^2)) / 2
  ∀ x,
    (x < r ∧ r ≤ R ↔ 1 / Real.sqrt 2 < r / R ∧ r / R ≤ 1) ∧
    (r < x ∧ x ≤ R ↔ (Real.sqrt 3 - 1) / 2 ≤ r / R ∧ r / R < 1 / Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_in_cylinder_l218_21880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gross_profit_calculation_l218_21828

/-- Given a sales price and a gross profit percentage, calculate the gross profit. -/
noncomputable def calculate_gross_profit (sales_price : ℝ) (gross_profit_percentage : ℝ) : ℝ :=
  let cost := sales_price / (1 + gross_profit_percentage)
  cost * gross_profit_percentage

/-- Theorem: The gross profit is $30 when the sales price is $54 and the gross profit is 125% of the cost. -/
theorem gross_profit_calculation :
  calculate_gross_profit 54 1.25 = 30 := by
  -- Unfold the definition of calculate_gross_profit
  unfold calculate_gross_profit
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gross_profit_calculation_l218_21828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intervals_increase_equal_range_solution_is_two_l218_21844

/-- The function f(x) defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2)^(-x^2 + 2*m*x - m^2 - 1)

/-- The theorem stating the equivalence of the intervals of increase and range for m = 2 -/
theorem intervals_increase_equal_range (m : ℝ) :
  (∀ x y : ℝ, x ≥ m → y < m → f m x > f m y) ↔ m = 2 := by
  sorry

/-- The main theorem proving the solution -/
theorem solution_is_two :
  ∃! m : ℝ, (∀ x y : ℝ, x ≥ m → y < m → f m x > f m y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intervals_increase_equal_range_solution_is_two_l218_21844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_l218_21801

/-- Custom operation @ for positive integers -/
def custom_op (k : Nat) (j : Nat) : Nat :=
  (List.range j).foldl (fun acc i => acc * (k + i)) k

/-- Theorem stating the value of b given a and e -/
theorem find_b (a : Nat) (e : Rat) (b : Nat) : 
  a = 2020 → e = 1/2 → e = a / b → b = 4040 := by
  sorry

#eval custom_op 6 4  -- This should output 3024 (6 * 7 * 8 * 9)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_l218_21801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_ln2_l218_21807

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then Real.exp x else x + 1

-- State the theorem
theorem f_composition_ln2 : f (f (Real.log 2)) = 3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_ln2_l218_21807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oh_squared_equals_850_l218_21852

/-- Triangle ABC with circumcenter O and orthocenter H -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ
  H : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: OH² = 850 for a triangle with given conditions -/
theorem oh_squared_equals_850 (t : Triangle)
  (h_circum : t.O = (0, 0))
  (h_radius : t.R = 10)
  (h_sides : t.a^2 + t.b^2 + t.c^2 = 50)
  (h_o_circum : distance t.O t.A = t.R ∧ distance t.O t.B = t.R ∧ distance t.O t.C = t.R)
  (h_h_ortho : (t.A.1 - t.B.1) * (t.H.1 - t.C.1) + (t.A.2 - t.B.2) * (t.H.2 - t.C.2) = 0 ∧
               (t.B.1 - t.C.1) * (t.H.1 - t.A.1) + (t.B.2 - t.C.2) * (t.H.2 - t.A.2) = 0 ∧
               (t.C.1 - t.A.1) * (t.H.1 - t.B.1) + (t.C.2 - t.A.2) * (t.H.2 - t.B.2) = 0) :
  (distance t.O t.H)^2 = 850 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oh_squared_equals_850_l218_21852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quinary_to_decimal_l218_21854

/-- Converts a list of digits in base b to a natural number. -/
def toNatCustom (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The base-5 representation of the number -/
def quinaryDigits : List Nat := [1, 2, 3, 4]

/-- Theorem: The base-10 representation of 1234 in base-5 is 194 -/
theorem quinary_to_decimal : toNatCustom quinaryDigits 5 = 194 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quinary_to_decimal_l218_21854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l218_21825

-- Define the power function f
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- Define the function g
noncomputable def g (α : ℝ) (x : ℝ) : ℝ := (x - 2) * f α x

-- Theorem statement
theorem min_value_of_g (α : ℝ) :
  f α 2 = 1/2 →
  ∃ (x : ℝ), x ∈ Set.Icc (1/2 : ℝ) 1 ∧
    (∀ (y : ℝ), y ∈ Set.Icc (1/2 : ℝ) 1 → g α x ≤ g α y) ∧
    g α x = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l218_21825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radical_identification_l218_21885

-- Define what it means for an expression to be always a quadratic radical
def always_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, f x = y

-- Define the expressions
noncomputable def expr1 (x : ℝ) : ℝ := Real.sqrt (-x)
noncomputable def expr2 (x : ℝ) : ℝ := Real.sqrt x
noncomputable def expr3 (x : ℝ) : ℝ := Real.sqrt (x^2 + 1)
noncomputable def expr4 (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)

-- State the theorem
theorem quadratic_radical_identification :
  always_quadratic_radical expr3 ∧
  ¬always_quadratic_radical expr1 ∧
  ¬always_quadratic_radical expr2 ∧
  ¬always_quadratic_radical expr4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radical_identification_l218_21885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_max_volume_height_l218_21870

-- Define the cone structure
structure Cone where
  height : ℝ
  baseRadius : ℝ

-- Define the sphere radius
def sphereRadius : ℝ := 1

-- Define the condition that the cone's vertex and base circumference are on the sphere
def onSphere (c : Cone) : Prop :=
  sphereRadius ^ 2 = (sphereRadius - c.height) ^ 2 + c.baseRadius ^ 2

-- Define the volume of the cone
noncomputable def coneVolume (c : Cone) : ℝ :=
  (1 / 3) * Real.pi * c.baseRadius ^ 2 * c.height

-- State the theorem
theorem cone_max_volume_height :
  ∃ (c : Cone), onSphere c ∧ 
  (∀ (c' : Cone), onSphere c' → coneVolume c' ≤ coneVolume c) ∧
  c.height = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_max_volume_height_l218_21870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_plate_probability_value_l218_21832

/-- The probability of selecting two plates of the same color from a set of 7 red plates and 5 green plates. -/
def same_color_plate_probability : ℚ := 
  let red_plates : ℕ := 7
  let green_plates : ℕ := 5
  let total_plates : ℕ := red_plates + green_plates
  let total_selections : ℕ := Nat.choose total_plates 2
  let red_selections : ℕ := Nat.choose red_plates 2
  let green_selections : ℕ := Nat.choose green_plates 2
  (red_selections + green_selections : ℚ) / total_selections

/-- Theorem stating that the probability is equal to 31/66 -/
theorem same_color_plate_probability_value : same_color_plate_probability = 31 / 66 := by
  -- Unfold the definition and simplify
  unfold same_color_plate_probability
  -- Perform the calculations
  norm_num
  -- The proof is complete
  rfl

#eval same_color_plate_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_plate_probability_value_l218_21832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l218_21890

-- Define the function f(x) = 2^x + 2
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + 2

-- Theorem stating the properties of f
theorem f_properties :
  (∀ y : ℝ, y > 2 → ∃ x : ℝ, f x = y) ∧ 
  (∀ x : ℝ, f x > 2) ∧
  (∀ x : ℝ, HasDerivAt f (Real.exp (x * Real.log 2) * Real.log 2) x) ∧
  (∀ x : ℝ, Real.exp (x * Real.log 2) * Real.log 2 > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l218_21890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poll_participants_proof_l218_21826

def poll_participants : ℕ := 260

theorem poll_participants_proof (
  initial_A_percent : ℚ := 35/100
) (initial_B_percent : ℚ := 65/100
) (additional_votes : ℕ := 80
) (final_B_percent : ℚ := 45/100
) :
  initial_A_percent + initial_B_percent = 1 →
  (∃ (x : ℕ), final_B_percent * (x + additional_votes) = initial_B_percent * x) →
  poll_participants = 260 := by
  intro h1 h2
  -- The proof goes here
  sorry

#check poll_participants_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poll_participants_proof_l218_21826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l218_21865

/-- Given a triangle ABC where angles A, B, C form an arithmetic progression
    and b^2 - a^2 = ac, prove that the radian measure of angle B is 2π/7 -/
theorem triangle_angle_measure (A B C a b c : ℝ) : 
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Arithmetic progression condition
  B - A = C - B →
  -- Given condition
  b^2 - a^2 = a * c →
  -- Conclusion
  B = 2 * Real.pi / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l218_21865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mode_difference_l218_21802

def dataset : List ℕ := [21, 25, 25, 26, 26, 26, 33, 33, 33, 37, 37, 37, 37, 40, 42, 45, 48, 48, 51, 55, 55, 59, 59, 59, 59, 59]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem median_mode_difference :
  |((median dataset : ℚ) - (mode dataset : ℚ))| = 22 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mode_difference_l218_21802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l218_21835

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 2 * sin (x / 2 + π / 3)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  T = 4 * π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l218_21835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_income_is_39000_main_theorem_l218_21803

/-- Represents the tax structure and Laura's income --/
structure TaxInfo where
  p : ℚ  -- Base tax rate
  income : ℚ  -- Laura's annual income

/-- Calculates the total tax based on the given tax structure --/
noncomputable def totalTax (info : TaxInfo) : ℚ :=
  let baseTax := min info.income 35000 * (info.p / 100)
  let extraTax := max (info.income - 35000) 0 * ((info.p + 3) / 100)
  baseTax + extraTax

/-- Theorem stating that Laura's income is $39000 --/
theorem laura_income_is_39000 (info : TaxInfo) :
  (totalTax info = info.income * (info.p + 0.3) / 100) →
  info.income = 39000 := by
  sorry

/-- Main theorem to be proved --/
theorem main_theorem : ∃ (info : TaxInfo), 
  (totalTax info = info.income * (info.p + 0.3) / 100) ∧
  info.income = 39000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_income_is_39000_main_theorem_l218_21803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attendees_l218_21843

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

-- Define the people
inductive Person
| Anna
| Bill
| Carl
| Dana
| Evan

-- Define a function to represent availability
def isAvailable (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Tuesday => true
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Thursday => true
  | Person.Anna, Day.Friday => false
  | Person.Bill, Day.Monday => true
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Wednesday => true
  | Person.Bill, Day.Thursday => false
  | Person.Bill, Day.Friday => true
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Wednesday => true
  | Person.Carl, Day.Thursday => true
  | Person.Carl, Day.Friday => false
  | Person.Dana, Day.Monday => true
  | Person.Dana, Day.Tuesday => true
  | Person.Dana, Day.Wednesday => false
  | Person.Dana, Day.Thursday => true
  | Person.Dana, Day.Friday => true
  | Person.Evan, Day.Monday => false
  | Person.Evan, Day.Tuesday => true
  | Person.Evan, Day.Wednesday => false
  | Person.Evan, Day.Thursday => true
  | Person.Evan, Day.Friday => true

-- Define a function to count the number of available people for a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (fun p => isAvailable p d) [Person.Anna, Person.Bill, Person.Carl, Person.Dana, Person.Evan]).length

-- Theorem: Tuesday and Friday have the maximum number of attendees
theorem max_attendees :
  (countAvailable Day.Tuesday = 3 ∧ countAvailable Day.Friday = 3) ∧
  (∀ d : Day, countAvailable d ≤ 3) := by
  sorry

#eval countAvailable Day.Monday
#eval countAvailable Day.Tuesday
#eval countAvailable Day.Wednesday
#eval countAvailable Day.Thursday
#eval countAvailable Day.Friday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_attendees_l218_21843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_purchase_equations_l218_21898

/-- Represents the value of one acre of good field in coins -/
def good_field_value : ℚ := 300

/-- Represents the value of seven acres of bad field in coins -/
def bad_field_value : ℚ := 500

/-- Represents the total area bought in acres -/
def total_area : ℚ := 100

/-- Represents the total cost in coins -/
def total_cost : ℚ := 10000

/-- Theorem stating the system of equations for the field purchase problem -/
theorem field_purchase_equations :
  ∃ (x y : ℚ),
    (x + y = total_area) ∧
    (good_field_value * x + (bad_field_value / 7) * y = total_cost) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_purchase_equations_l218_21898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l218_21895

theorem matrix_transformation (v : Fin 3 → ℝ) : 
  let N : Matrix (Fin 3) (Fin 3) ℝ := ![![2, 0, 0], ![0, 3, 0], ![0, 0, 4]]
  N.mulVec v = fun i => (i.val + 1) * v i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l218_21895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_symmetry_axes_different_l218_21821

noncomputable section

/-- Two quadratic polynomials -/
def f (x : ℝ) : ℝ := (1/8) * (x^2 + 6*x - 25)
def g (x : ℝ) : ℝ := (1/8) * (31 - x^2)

/-- Intersection points -/
def x₁ : ℝ := -7
def x₂ : ℝ := 4

/-- Axes of symmetry -/
def axis_f : ℝ := -3
def axis_g : ℝ := 0

/-- Theorem: The axes of symmetry of two quadratic polynomials that intersect at two points
    with perpendicular tangents are not necessarily the same -/
theorem quadratic_symmetry_axes_different :
  (f x₁ = g x₁) ∧ (f x₂ = g x₂) ∧  -- Intersection at two points
  ((deriv f x₁) * (deriv g x₁) = -1) ∧ ((deriv f x₂) * (deriv g x₂) = -1) ∧  -- Perpendicular tangents
  (axis_f ≠ axis_g) :=  -- Different axes of symmetry
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_symmetry_axes_different_l218_21821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_theorem_l218_21827

/-- Represents the properties of a rectangular floor -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  area : ℝ
  paintingCost : ℝ
  paintingRate : ℝ

/-- Theorem about the length of a rectangular floor given specific conditions -/
theorem floor_length_theorem (floor : RectangularFloor) 
  (h1 : floor.length = 3 * floor.breadth)
  (h2 : floor.area = floor.length * floor.breadth)
  (h3 : floor.paintingCost = 361)
  (h4 : floor.paintingRate = 3.00001)
  (h5 : floor.area = floor.paintingCost / floor.paintingRate) :
  ∃ ε > 0, |floor.length - 18.99| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_length_theorem_l218_21827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l218_21819

-- Define the slant height of the cone
noncomputable def slant_height : ℝ := 10

-- Define the lateral surface area of the cone
noncomputable def lateral_surface_area (r : ℝ) : ℝ := (1/2) * Real.pi * r^2

-- Theorem stating that the lateral surface area of the cone is 50π
theorem cone_lateral_surface_area :
  lateral_surface_area slant_height = 50 * Real.pi := by
  -- Unfold the definition of lateral_surface_area
  unfold lateral_surface_area
  -- Substitute the value of slant_height
  simp [slant_height]
  -- Simplify the arithmetic
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l218_21819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l218_21831

noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 1)
def b : ℝ × ℝ := (0, -1)
noncomputable def c (k : ℝ) : ℝ × ℝ := (Real.sqrt 3, k)

theorem perpendicular_vectors (k : ℝ) :
  (a.1 - 2 * b.1) * (c k).1 + (a.2 - 2 * b.2) * (c k).2 = 0 → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l218_21831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_2sinx_minus_1_domain_of_sqrt_tanx_minus_sqrt3_l218_21812

open Real

-- Function 1: y = √(2sin x - 1)
def domain1 (x : ℝ) : Prop :=
  ∃ k : ℤ, x ∈ Set.Icc (π/6 + 2*π*k) (5*π/6 + 2*π*k)

theorem domain_of_sqrt_2sinx_minus_1 (x : ℝ) :
  (∃ y : ℝ, y = sqrt (2 * sin x - 1)) ↔ domain1 x :=
sorry

-- Function 2: y = √(tan x - √3)
def domain2 (x : ℝ) : Prop :=
  ∃ k : ℤ, x ∈ Set.Ico (π/3 + π*k) (π/2 + π*k)

theorem domain_of_sqrt_tanx_minus_sqrt3 (x : ℝ) :
  (∃ y : ℝ, y = sqrt (tan x - sqrt 3)) ↔ domain2 x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_2sinx_minus_1_domain_of_sqrt_tanx_minus_sqrt3_l218_21812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_scaling_l218_21871

def dataset (a : Fin 5 → ℝ) : Set ℝ := {x | ∃ i, x = a i}

noncomputable def variance (s : Set ℝ) : ℝ := sorry

theorem variance_scaling (a : Fin 5 → ℝ) (h : variance (dataset a) = 1) :
  variance (dataset (fun i ↦ 2 * a i)) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_scaling_l218_21871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_x_over_x_plus_one_l218_21861

theorem max_value_sqrt_x_over_x_plus_one :
  ∃ (max : ℝ), max = 1/2 ∧
  (∀ x : ℝ, x ≥ 0 → Real.sqrt x / (x + 1) ≤ max) ∧
  (∃ x : ℝ, x ≥ 0 ∧ Real.sqrt x / (x + 1) = max) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_x_over_x_plus_one_l218_21861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l218_21842

theorem quadratic_root_difference (a b c : ℝ) (p q : ℕ) (h1 : a = 2) (h2 : b = -5) (h3 : c = -7) 
  (h4 : q = 2) (h5 : p = 81) :
  let Δ := b^2 - 4*a*c
  let x₁ := (-b + Real.sqrt Δ) / (2*a)
  let x₂ := (-b - Real.sqrt Δ) / (2*a)
  (x₁ - x₂ = Real.sqrt p / q) ∧ (p + q = 83) := by
  sorry

#check quadratic_root_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l218_21842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_theorem_l218_21883

/-- The foot of the perpendicular from the origin to a plane -/
structure FootOfPerpendicular where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Coefficients of the plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if the given coefficients satisfy the required conditions -/
def validCoefficients (coeffs : PlaneCoefficients) : Prop :=
  coeffs.A > 0 ∧ Int.gcd (Int.natAbs coeffs.A) (Int.gcd (Int.natAbs coeffs.B) (Int.gcd (Int.natAbs coeffs.C) (Int.natAbs coeffs.D))) = 1

/-- Check if the given coefficients form the equation of the plane with the given foot of perpendicular -/
def isCorrectPlaneEquation (foot : FootOfPerpendicular) (coeffs : PlaneCoefficients) : Prop :=
  coeffs.A * foot.x + coeffs.B * foot.y + coeffs.C * foot.z + coeffs.D = 0

/-- The main theorem statement -/
theorem plane_equation_theorem (foot : FootOfPerpendicular) 
    (h_foot : foot.x = 8 ∧ foot.y = -6 ∧ foot.z = 5) : 
    ∃ (coeffs : PlaneCoefficients), 
      validCoefficients coeffs ∧ 
      isCorrectPlaneEquation foot coeffs ∧ 
      coeffs.A = 8 ∧ coeffs.B = -6 ∧ coeffs.C = 5 ∧ coeffs.D = -125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_theorem_l218_21883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_on_ellipse_l218_21886

/-- Represents a point on an ellipse -/
structure EllipsePoint where
  x : ℝ
  y : ℝ
  on_ellipse : x^2/4 + y^2/3 = 1

/-- The right focus of the ellipse -/
def F : ℝ × ℝ := (1, 0)

/-- Distance from a point to the right focus -/
noncomputable def dist_to_focus (p : EllipsePoint) : ℝ :=
  Real.sqrt ((p.x - F.1)^2 + (p.y - F.2)^2)

/-- Theorem stating the maximum number of points on the ellipse
    forming an arithmetic sequence with the given conditions -/
theorem max_points_on_ellipse (n : ℕ) 
  (points : Fin n → EllipsePoint) 
  (is_arith_seq : ∃ (d : ℝ), d ≥ 1/100 ∧ 
    ∀ (i j : Fin n), dist_to_focus (points j) - dist_to_focus (points i) = d * (j - i)) :
  n ≤ 201 := by
  sorry

#check max_points_on_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_on_ellipse_l218_21886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_about_decimal_places_l218_21815

theorem incorrect_statement_about_decimal_places : ¬(∃ (n : ℚ), 56.769 = 50 + n + 0.769 ∧ n = 10 * 0.6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_about_decimal_places_l218_21815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_and_sum_l218_21823

theorem largest_solution_and_sum :
  ∃ (a b c : ℕ) (x : ℝ),
    (4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 13*x - 6) ∧
    (x = a + Real.sqrt (b + Real.sqrt c)) ∧
    (a + b + c = 379) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_and_sum_l218_21823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l218_21876

def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 - x + (1/16) * a > 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x > 0 → Real.sqrt (2*x + 1) < 1 + a*x

theorem range_of_a (a : ℝ) :
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → (1 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l218_21876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_integers_with_conditions_l218_21872

-- Define the range
def lower_bound : ℕ := 3000
def upper_bound : ℕ := 6000

-- Define a function to check if a number has four different digits
def has_four_different_digits (n : ℕ) : Bool :=
  let digits := n.digits 10
  digits.length = 4 && digits.toFinset.card = 4

-- Define a function to check if the hundreds digit is non-zero
def has_nonzero_hundreds_digit (n : ℕ) : Bool :=
  (n / 100) % 10 ≠ 0

-- Define the main theorem
theorem count_even_integers_with_conditions :
  (Finset.filter (λ n : ℕ =>
    n % 2 = 0 ∧
    has_four_different_digits (n + lower_bound) ∧
    has_nonzero_hundreds_digit (n + lower_bound)
  ) (Finset.range (upper_bound - lower_bound + 1))).card = 784 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_integers_with_conditions_l218_21872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_inequality_l218_21879

/-- The function g(x) defined in the problem -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - (1/2) * m * x^2 - (m - 4) * x

/-- The derivative of g(x) -/
noncomputable def g' (m : ℝ) (x : ℝ) : ℝ := 4 / x - m * x + (4 - m)

theorem slope_inequality {m x₀ x₁ x₂ : ℝ} (hm : m > 0) (hx₀ : x₀ > 0) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
    (hne : x₁ ≠ x₂) (hk : (g m x₁ - g m x₂) / (x₁ - x₂) = g' m x₀) :
  x₁ + x₂ > 2 * x₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_inequality_l218_21879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_construction_completed_l218_21838

/-- Represents the length of road built after n months -/
noncomputable def roadLength : ℕ → ℝ
  | 0 => 0
  | n + 1 => min (roadLength n + 1 / (roadLength n + 1)^10) 100

/-- The total road length to be built -/
def totalRoadLength : ℝ := 100

/-- The number of months required to complete the road -/
def completionTime : ℕ := 100^11

/-- Theorem stating that the road construction will be completed -/
theorem road_construction_completed :
  ∃ n : ℕ, roadLength n = totalRoadLength := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_construction_completed_l218_21838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_m_range_l218_21811

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1/x - 1 else -x^3 + 1

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, f x ≥ m^2 + 2*m - 2

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (m^2 - 1)^x < (m^2 - 1)^y

theorem f_minimum_and_m_range :
  (∀ x : ℝ, f x ≥ 1) ∧
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ 
    m < -3 ∨ (-Real.sqrt 2 ≤ m ∧ m ≤ 1) ∨ m > Real.sqrt 2) :=
by sorry

#check f_minimum_and_m_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_m_range_l218_21811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_jump_iff_n_mod_four_eq_two_l218_21805

/-- A regular polygon with 2n sides inscribed in a circle. -/
structure RegularPolygon (n : ℕ) where
  n_ge_two : n ≥ 2

/-- The type of jump where all frogs simultaneously move to an adjacent vertex. -/
def Jump (n : ℕ) (p : RegularPolygon n) := Unit

/-- Predicate to check if a line segment passes through the center of the circle. -/
def LinePassesThroughCenter (n : ℕ) (p : RegularPolygon n) (v1 v2 : Fin (2 * n)) : Prop := sorry

/-- Predicate to check if there exists a jump after which no line segment
    connecting any two different vertices with frogs passes through the center. -/
def ExistsValidJump (n : ℕ) (p : RegularPolygon n) : Prop := 
  ∃ (j : Jump n p), ∀ (v1 v2 : Fin (2 * n)), v1 ≠ v2 → ¬LinePassesThroughCenter n p v1 v2

/-- The main theorem stating the condition for the existence of a valid jump. -/
theorem valid_jump_iff_n_mod_four_eq_two (n : ℕ) :
  (∃ (p : RegularPolygon n), ExistsValidJump n p) ↔ n % 4 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_jump_iff_n_mod_four_eq_two_l218_21805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_domino_moves_9x9_l218_21818

/-- Represents a square on the checkerboard -/
inductive Square
| Black
| White

/-- Represents a 9x9 checkerboard -/
def Checkerboard := Fin 9 → Fin 9 → Square

/-- Creates a 9x9 checkerboard with alternating black and white squares, starting with black in the bottom left -/
def create_checkerboard : Checkerboard :=
  fun i j => if (i.val + j.val) % 2 = 0 then Square.Black else Square.White

/-- Represents a domino move on the checkerboard -/
structure DominoMove where
  row : Fin 9
  col : Fin 9
  horizontal : Bool

/-- Predicate to check if a list of moves covers the entire board -/
def covers_board (moves : List DominoMove) (board : Checkerboard) : Prop :=
  sorry -- Definition of covers_board goes here

/-- Theorem: The minimum number of domino moves required to cover a 9x9 checkerboard is 48 -/
theorem min_domino_moves_9x9 (board : Checkerboard) :
  board = create_checkerboard →
  ∃ (moves : List DominoMove), moves.length = 48 ∧ covers_board moves board := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_domino_moves_9x9_l218_21818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_balance_is_171_l218_21899

/-- Calculates the new credit card balance after transactions -/
noncomputable def newCreditCardBalance (initialBalance groceriesCost bathTowelsReturn : ℚ) : ℚ :=
  initialBalance + groceriesCost + (groceriesCost / 2) - bathTowelsReturn

/-- Theorem stating that the new balance is $171.00 given the specified transactions -/
theorem new_balance_is_171 :
  newCreditCardBalance 126 60 45 = 171 := by
  -- Unfold the definition of newCreditCardBalance
  unfold newCreditCardBalance
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_balance_is_171_l218_21899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_shift_theorem_l218_21897

/-- Given that a, b, and c are the roots of x³ - 4x² - 6x + 8 = 0,
    prove that a - 3, b - 3, and c - 3 are the roots of x³ + 5x² - 3x - 19 = 0 -/
theorem root_shift_theorem (a b c : ℂ) :
  (∀ x : ℂ, x^3 - 4*x^2 - 6*x + 8 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 5*x^2 - 3*x - 19 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_shift_theorem_l218_21897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_l218_21849

/-- Predicate to check if three complex numbers form an equilateral triangle -/
def IsEquilateralTriangle (x y z : ℂ) : Prop :=
  Complex.abs (x - y) = Complex.abs (y - z) ∧ 
  Complex.abs (y - z) = Complex.abs (z - x) ∧
  Complex.abs (z - x) = Complex.abs (x - y)

/-- Given complex numbers x, y, and z forming an equilateral triangle with side length 20
    in the complex plane, if |x + y + z| = 40, then |xy + xz + yz| = 1600/3 -/
theorem equilateral_triangle_complex (x y z : ℂ) 
    (h_equilateral : IsEquilateralTriangle x y z)
    (h_side_length : ∀ (a b : ℂ), (a, b) ∈ [(x, y), (y, z), (z, x)] → Complex.abs (a - b) = 20)
    (h_sum_abs : Complex.abs (x + y + z) = 40) :
    Complex.abs (x * y + x * z + y * z) = 1600 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_complex_l218_21849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l218_21856

noncomputable def train_length : ℝ := 110
noncomputable def train_speed : ℝ := 50
noncomputable def man_speed : ℝ := 5
noncomputable def km_per_hour_to_m_per_sec : ℝ := 5 / 18

noncomputable def relative_speed : ℝ := train_speed + man_speed
noncomputable def relative_speed_m_per_sec : ℝ := relative_speed * km_per_hour_to_m_per_sec
noncomputable def time_to_pass : ℝ := train_length / relative_speed_m_per_sec

theorem train_passing_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs (time_to_pass - 7.20) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l218_21856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_line_l218_21816

/-- The distance from the right focus of the hyperbola x²/4 - y²/5 = 1 to the line x + 2y - 8 = 0 is √5 -/
theorem distance_from_focus_to_line : 
  let hyperbola : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 / 4 - y^2 / 5 = 1
  let line : ℝ × ℝ → Prop := fun (x, y) ↦ x + 2*y - 8 = 0
  let right_focus : ℝ × ℝ := (3, 0)
  let distance_formula : ℝ × ℝ → ℝ := fun (x, y) ↦ 
    |x + 2*y - 8| / Real.sqrt (1^2 + 2^2)
  distance_formula right_focus = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_focus_to_line_l218_21816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l218_21809

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ Real.sqrt (-x^2 + x + 2)

-- State the theorem
theorem monotonic_increase_interval :
  ∃ (a b : ℝ), a = 1/2 ∧ b = 2 ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
  (∀ x, x < a ∨ x > b → ¬(∀ y, x < y → f x < f y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l218_21809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_leq_two_minus_x_iff_l218_21874

def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ 0 then f x else -f (-x)

theorem g_leq_two_minus_x_iff (x : ℝ) :
  g x ≤ 2 - x ↔ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_leq_two_minus_x_iff_l218_21874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l218_21877

/-- Calculates the time (in seconds) for a train to cross a bridge. -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem: A train 180 meters long, running at 54 kmph, takes 56 seconds to cross a bridge 660 meters in length. -/
theorem train_crossing_bridge :
  train_crossing_time 180 54 660 = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l218_21877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowball_model_l218_21817

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem snowball_model (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 6) (h₃ : r₃ = 7) :
  (sphere_volume r₁ + sphere_volume r₂ + sphere_volume r₃ = (2492 / 3) * Real.pi) ∧
  (sphere_surface_area r₃ = 196 * Real.pi) := by
  sorry

#check snowball_model

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowball_model_l218_21817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_function_properties_l218_21857

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℤ := floor x

def is_odd (f : ℝ → ℤ) : Prop :=
  ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℤ) : Prop :=
  ∀ x, f (-x) = f x

theorem floor_function_properties :
  let S : Set ℝ := Set.Icc (-2) 3
  (¬ is_odd f ∧ ¬ is_even f) ∧
  (f '' S = {-2, -1, 0, 1, 2, 3}) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_function_properties_l218_21857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_at_tangency_point_l218_21884

/-- Two circles externally tangent at a point -/
structure ExternallyTangentCircles :=
  (Γ₁ Γ₂ : Set (ℝ × ℝ))
  (T : ℝ × ℝ)
  (h_tangent : T ∈ Γ₁ ∩ Γ₂)

/-- Common tangent to two circles -/
structure CommonTangent (etc : ExternallyTangentCircles) :=
  (l : Set (ℝ × ℝ))
  (P Q : ℝ × ℝ)
  (h_tangent_Γ₁ : P ∈ Γ₁ ∩ l)
  (h_tangent_Γ₂ : Q ∈ Γ₂ ∩ l)
  (h_not_through_T : etc.T ∉ l)

/-- Definition of a right angle -/
def RightAngle (A B C : ℝ × ℝ) : Prop :=
  let v1 := (A.1 - B.1, A.2 - B.2)
  let v2 := (C.1 - B.1, C.2 - B.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

/-- The theorem to be proved -/
theorem right_angle_at_tangency_point
  (etc : ExternallyTangentCircles)
  (ct : CommonTangent etc) :
  RightAngle ct.P etc.T ct.Q :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_at_tangency_point_l218_21884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_correct_answers_verify_solution_l218_21848

/-- Represents an examination with scoring rules and results. -/
structure Examination where
  total_questions : ℕ
  correct_score : ℤ
  wrong_score : ℤ
  total_score : ℤ

/-- Calculates the number of correct answers in an examination. -/
def correct_answers (exam : Examination) : ℤ :=
  (exam.total_score + exam.wrong_score * exam.total_questions) / (exam.correct_score - exam.wrong_score)

/-- Theorem stating that for the given examination parameters,
    the number of correct answers is 74. -/
theorem exam_correct_answers :
  let exam : Examination := {
    total_questions := 100,
    correct_score := 5,
    wrong_score := -2,
    total_score := 320
  }
  correct_answers exam = 74 := by
  -- The proof goes here
  sorry

/-- Verifies that the calculated number of correct answers satisfies the problem conditions. -/
theorem verify_solution (exam : Examination) (c : ℤ) (h : c = correct_answers exam) :
  c * exam.correct_score + (exam.total_questions - c) * exam.wrong_score = exam.total_score ∧
  0 ≤ c ∧ c ≤ exam.total_questions := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_correct_answers_verify_solution_l218_21848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_monotonicity_l218_21867

-- Define the function f
def f (m n x : ℝ) : ℝ := x^3 + 3*m*x^2 + n*x

-- Define the function g
noncomputable def g (m n x : ℝ) : ℝ := f m n x - x^3 - 3*Real.log x

-- State the theorem
theorem extremum_and_monotonicity 
  (m n : ℝ) 
  (h1 : f m n (-1) = 0) 
  (h2 : ∃ ε > 0, ∀ x ∈ Set.Ioo (-1-ε) (-1+ε), f m n x ≥ f m n (-1)) :
  (m = 2/3 ∧ n = 1) ∧ 
  (∀ x y, 0 < x ∧ x < 3/4 ∧ 3/4 < y → g m n x > g m n y) ∧
  (∀ x y, 3/4 < x ∧ x < y → g m n x < g m n y) := by
  sorry

#check extremum_and_monotonicity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_monotonicity_l218_21867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_subsequences_count_l218_21887

/-- An arithmetic sequence of 10 terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n < 10 → a (n + 1) = a n + d

/-- Four terms form an arithmetic sequence -/
def IsArithmeticSequence (x y z w : ℝ) : Prop :=
  y - x = z - y ∧ z - y = w - z

/-- The set of all 4-term subsequences of a 10-term sequence -/
def AllSubsequences (a : ℕ → ℝ) : Set (Fin 4 → ℝ) :=
  { s | ∃ i j k l : Fin 10, i < j ∧ j < k ∧ k < l ∧
    s 0 = a i ∧ s 1 = a j ∧ s 2 = a k ∧ s 3 = a l }

/-- The set of all arithmetic 4-term subsequences of a 10-term sequence -/
def ArithmeticSubsequences (a : ℕ → ℝ) : Set (Fin 4 → ℝ) :=
  { s ∈ AllSubsequences a | IsArithmeticSequence (s 0) (s 1) (s 2) (s 3) }

-- Add this instance to ensure Fintype for ArithmeticSubsequences
instance (a : ℕ → ℝ) : Fintype (ArithmeticSubsequences a) :=
  sorry

theorem arithmetic_subsequences_count (a : ℕ → ℝ) :
  ArithmeticSequence a → Fintype.card (ArithmeticSubsequences a) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_subsequences_count_l218_21887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_recipe_flour_amount_l218_21860

/-- A recipe with butter and flour measurements -/
structure Recipe where
  butter : ℚ
  flour : ℚ

/-- The ratio of flour to butter in a recipe -/
def flourToButter (r : Recipe) : ℚ := r.flour / r.butter

theorem original_recipe_flour_amount 
  (original : Recipe)
  (scaled : Recipe)
  (h1 : original.butter = 3)
  (h2 : scaled.butter = 6 * original.butter)
  (h3 : scaled.butter = 12)
  (h4 : scaled.flour = 24)
  : original.flour = 6 := by
  sorry

#eval flourToButter { butter := 3, flour := 6 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_recipe_flour_amount_l218_21860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_3n_eq_xk_plus_yk_l218_21836

theorem unique_solution_for_3n_eq_xk_plus_yk :
  ∀ (x y n k : ℕ), 
    (x > 0) → (y > 0) → (n > 0) → (k > 0) →
    (Nat.gcd x y = 1) →
    (3^n = x^k + y^k) →
    (x = 2 ∧ y = 1 ∧ k = 3 ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_3n_eq_xk_plus_yk_l218_21836
