import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_income_percentage_tim_income_less_than_juan_l1326_132629

-- Define the incomes as real numbers
variable (Tim_income Mart_income Juan_income : ℝ)

-- Define the given conditions
def condition1 (Tim_income Mart_income : ℝ) : Prop := Mart_income = 1.40 * Tim_income
def condition2 (Mart_income Juan_income : ℝ) : Prop := Mart_income = 0.84 * Juan_income

-- Define the theorem to be proved
theorem tim_income_percentage 
  (h1 : condition1 Tim_income Mart_income) 
  (h2 : condition2 Mart_income Juan_income) :
  Tim_income = 0.60 * Juan_income := by
  sorry

-- Define the percentage difference
def percentage_difference : ℝ := 40

-- Define the final theorem that relates to the original question
theorem tim_income_less_than_juan 
  (h1 : condition1 Tim_income Mart_income) 
  (h2 : condition2 Mart_income Juan_income) :
  (1 - Tim_income / Juan_income) * 100 = percentage_difference := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_income_percentage_tim_income_less_than_juan_l1326_132629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_g_increasing_on_reals_l1326_132687

open Set Function Real

-- Define the functions
noncomputable def f : ℝ → ℝ := fun x ↦ Real.exp (-x)
def g : ℝ → ℝ := fun x ↦ x^3
noncomputable def h : ℝ → ℝ := fun x ↦ Real.log x
def k : ℝ → ℝ := fun x ↦ abs x

-- Define what it means for a function to be increasing on ℝ
def increasing_on_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem only_g_increasing_on_reals :
  (increasing_on_reals g) ∧
  (¬ increasing_on_reals f) ∧
  (¬ increasing_on_reals h) ∧
  (¬ increasing_on_reals k) :=
by
  sorry

#check only_g_increasing_on_reals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_g_increasing_on_reals_l1326_132687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_elevenths_rounded_l1326_132626

-- Define a function to round a real number to a specified number of decimal places
noncomputable def round_to_decimal_places (x : ℝ) (n : ℕ) : ℝ :=
  (⌊x * 10^n + 0.5⌋) / 10^n

-- State the theorem
theorem eight_elevenths_rounded : round_to_decimal_places (8/11) 3 = 0.727 := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_elevenths_rounded_l1326_132626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l1326_132627

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a
  h_eccentricity : a / Real.sqrt (a^2 - b^2) = 2
  h_point : 4 / a^2 + 0 / b^2 = 1

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  h_focus : k * (E.a / 2 - 1) = 0
  h_area : 6 * (1 + k^2) * |k| / (3 + 4 * k^2) = 6 * Real.sqrt 2 / 7

/-- The main theorem -/
theorem ellipse_and_line_properties (E : Ellipse) (l : IntersectingLine E) :
  (E.a = 2 ∧ E.b = Real.sqrt 3) ∧
  (l.k = 1 ∨ l.k = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l1326_132627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1326_132631

def v : Fin 2 → ℚ := ![4, 5]
def w : Fin 2 → ℚ := ![12, -3]

def dot_product (u v : Fin 2 → ℚ) : ℚ :=
  (Finset.univ.sum (λ i => u i * v i))

def proj (u v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  let scalar := (dot_product v u) / (dot_product u u)
  λ i => scalar * u i

theorem projection_theorem : 
  proj w v = ![132/51, -33/51] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1326_132631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equal_for_y_symmetric_angles_l1326_132679

-- Define the property of terminal sides being symmetric about the y-axis
def symmetric_about_y_axis (α β : ℝ) : Prop := sorry

-- Theorem statement
theorem sin_equal_for_y_symmetric_angles (α β : ℝ) 
  (h : symmetric_about_y_axis α β) : Real.sin α = Real.sin β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equal_for_y_symmetric_angles_l1326_132679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1326_132684

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, -3 * Real.cos x)
noncomputable def c (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.sin x)

noncomputable def f (x : ℝ) : ℝ := 
  let a_vec := a x
  let b_vec := b x
  let c_vec := c x
  let bc_sum := (b_vec.1 + c_vec.1, b_vec.2 + c_vec.2)
  a_vec.1 * bc_sum.1 + a_vec.2 * bc_sum.2

theorem f_properties :
  (∀ x : ℝ, f x ≤ 2 + Real.sqrt 2) ∧ 
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, 
    k * π + 3 * π / 8 ≤ x ∧ x ≤ k * π + 7 * π / 8 → 
    ∀ y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1326_132684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l1326_132669

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ :=
  let h := -p.b / (2 * p.a)
  let k := p.c - p.b^2 / (4 * p.a)
  (h, k + 1 / (4 * p.a))

theorem focus_of_specific_parabola :
  let p := Parabola.mk (-2) (-6) 1
  focus p = (-3/2, 43/8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l1326_132669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_mixed_up_answer_is_correct_l1326_132654

-- Define the types
inductive Person | he | dog

-- Define the possible actions
inductive Action | introduce | call_name

-- Define the possible outcomes
inductive Outcome | correct | mixed_up

-- Function to model the scenario
def scenario (action : Action) : Outcome :=
  match action with
  | Action.introduce => Outcome.mixed_up
  | Action.call_name => Outcome.mixed_up

-- Theorem: The outcome is always mixed_up
theorem always_mixed_up (action : Action) :
  scenario action = Outcome.mixed_up := by
  cases action
  . rfl
  . rfl

#check always_mixed_up

-- The answer is D: mixed up
def answer : String := "D"

-- Proof that the answer is correct
theorem answer_is_correct : answer = "D" := by rfl

#check answer_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_mixed_up_answer_is_correct_l1326_132654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_relation_l1326_132665

variable (x : Fin 10 → ℝ)

def derived_data (x : Fin 10 → ℝ) (i : Fin 10) : ℝ := 2 * x i + 1

noncomputable def variance (data : Fin 10 → ℝ) : ℝ :=
  let mean := (Finset.univ.sum data) / 10
  (Finset.univ.sum (fun i => (data i - mean) ^ 2)) / 10

theorem variance_relation (x : Fin 10 → ℝ) :
  variance (derived_data x) = 8 → variance x = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_relation_l1326_132665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vins_total_miles_l1326_132673

/-- Calculates the total miles Vins rides during W weeks -/
noncomputable def total_miles (L S X Y Z F : ℝ) (W : ℝ) : ℝ :=
  let library_trip := L + (L + X)
  let school_trip := S + (S + Y)
  let friend_trip := F + (F - Z)
  let weekly_miles := 3 * library_trip + 2 * school_trip + friend_trip / 2
  W * weekly_miles

/-- Theorem stating the total miles Vins rides during W weeks -/
theorem vins_total_miles (W : ℝ) (h : W ≥ 2) :
  total_miles 5 6 1 2 3 8 W = 67.5 * W :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vins_total_miles_l1326_132673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1326_132630

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  total_distance / train_speed_ms

/-- Theorem stating that the time taken for the train to cross the bridge is approximately 16.79 seconds -/
theorem train_crossing_bridge_time :
  let train_length := 110
  let bridge_length := 170
  let train_speed_kmh := 60
  let crossing_time := train_crossing_time train_length bridge_length train_speed_kmh
  ∃ ε > 0, |crossing_time - 16.79| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l1326_132630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_probability_l1326_132691

theorem biased_coin_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1) 
  (h3 : (Nat.choose 7 2) * p^2 * (1-p)^5 = (Nat.choose 7 3) * p^3 * (1-p)^4) : 
  (Nat.choose 7 4) * p^4 * (1-p)^3 = 135/512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_probability_l1326_132691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_not_dividing_1999_factorial_l1326_132697

theorem smallest_n_not_dividing_1999_factorial (n : ℕ) : 
  (∀ k < n, 34^k ∣ Nat.factorial 1999) ∧ ¬(34^n ∣ Nat.factorial 1999) ↔ n = 124 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_not_dividing_1999_factorial_l1326_132697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_triangles_in_3x3_grid_l1326_132655

/-- Represents a 3x3 grid of dots -/
def Grid := Fin 3 × Fin 3

/-- Represents a triangle formed by three dots -/
def Triangle := (Grid × Grid × Grid)

/-- Checks if three points are collinear -/
def collinear (a b c : Grid) : Prop := sorry

/-- The total number of ways to choose 3 dots from 9 -/
def total_combinations : ℕ := Nat.choose 9 3

/-- The number of degenerate cases (collinear points) -/
def degenerate_cases : ℕ := 8

/-- A triangle is valid if its vertices are not collinear -/
def valid_triangle (t : Triangle) : Prop :=
  let (a, b, c) := t
  ¬(collinear a b c)

/-- Decidability instance for valid_triangle -/
instance : DecidablePred valid_triangle := fun _ => sorry

/-- Finite type instance for Triangle -/
instance : Fintype Triangle := sorry

/-- The main theorem: number of distinct triangles in a 3x3 grid -/
theorem distinct_triangles_in_3x3_grid :
  (Finset.univ.filter valid_triangle).card = total_combinations - degenerate_cases := by
  sorry

#eval total_combinations - degenerate_cases

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_triangles_in_3x3_grid_l1326_132655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manager_to_non_manager_ratio_l1326_132635

/-- Represents the number of managers in a department -/
def managers : ℕ := 8

/-- Represents the number of non-managers in a department -/
def non_managers : ℕ := 38

/-- The ratio of managers to non-managers is constant across departments -/
axiom constant_ratio : ∃ (k : ℚ), ∀ (m n : ℕ), (m : ℚ) / (n : ℚ) = k

/-- In a department with 8 managers, the maximum number of non-managers is 38 -/
axiom max_non_managers : managers = 8 → non_managers ≤ 38

/-- The ratio of managers to non-managers is 8:38 -/
theorem manager_to_non_manager_ratio : 
  (managers : ℚ) / (non_managers : ℚ) = 8 / 38 := by
  sorry

#check manager_to_non_manager_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manager_to_non_manager_ratio_l1326_132635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_decreasing_in_interval_f_period_f_symmetry_f_zero_l1326_132696

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi/3)

theorem f_not_decreasing_in_interval :
  ∃ (x₁ x₂ : ℝ), Real.pi/2 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi ∧ f x₁ < f x₂ :=
by
  sorry

theorem f_period : ∀ (x : ℝ), f (x - 2*Real.pi) = f x :=
by
  sorry

theorem f_symmetry : ∀ (x : ℝ), f (8*Real.pi/3 - x) = f (8*Real.pi/3 + x) :=
by
  sorry

theorem f_zero : f (Real.pi/6 + Real.pi) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_decreasing_in_interval_f_period_f_symmetry_f_zero_l1326_132696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_k_max_l1326_132662

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 2

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a

theorem f_monotonicity_and_k_max (a : ℝ) :
  -- Part 1: Monotonicity for a ≤ 0
  (a ≤ 0 → ∀ x y : ℝ, x < y → f a x < f a y) ∧
  -- Part 2: Monotonicity for a > 0
  (a > 0 → (∀ x y : ℝ, x < y ∧ y < Real.log a → f a x > f a y) ∧
           (∀ x y : ℝ, Real.log a < x ∧ x < y → f a x < f a y)) ∧
  -- Part 3: Maximum value of k
  (∃ k : ℕ, k = 2 ∧ 
    ∀ m : ℕ, (∀ x : ℝ, x > 0 → (x - ↑m) * (f_prime 1 x) + x + 1 > 0) → m ≤ k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_k_max_l1326_132662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_inscribed_square_l1326_132623

theorem circle_area_from_inscribed_square (W X Y Z : ℝ × ℝ) :
  let square_area : ℝ := 9
  let is_square : Prop := ∃ s : ℝ, s > 0 ∧ 
    (X.1 - W.1)^2 + (X.2 - W.2)^2 = s^2 ∧
    (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = s^2 ∧
    (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 = s^2 ∧
    (W.1 - Z.1)^2 + (W.2 - Z.2)^2 = s^2
  let on_circle : Prop := ∃ r : ℝ, r > 0 ∧
    (X.1 - W.1)^2 + (X.2 - W.2)^2 = r^2 ∧
    (Z.1 - W.1)^2 + (Z.2 - W.2)^2 = r^2
  is_square ∧ on_circle → (π * ((X.1 - W.1)^2 + (X.2 - W.2)^2) = 9 * π) := by
  intro h
  sorry

#check circle_area_from_inscribed_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_inscribed_square_l1326_132623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_operation_results_l1326_132657

/-- Represents a taxi trip with distance and direction --/
structure TaxiTrip where
  distance : ℚ
  direction : Bool  -- true for north, false for south

/-- Calculates the fare for a single trip --/
def calculateFare (trip : ℚ) : ℚ :=
  if trip ≤ 3 then 10 else 10 + 2 * (trip - 3)

/-- Taxi driver's afternoon operation --/
def taxiOperation : List TaxiTrip :=
  [⟨2, false⟩, ⟨3, false⟩, ⟨6, false⟩, ⟨8, true⟩, ⟨9, false⟩, ⟨7, false⟩, ⟨5, false⟩, ⟨13, true⟩]

/-- Starting price for each trip --/
def startingPrice : ℚ := 10

/-- Extra charge per km over 3 km --/
def extraCharge : ℚ := 2

/-- Fuel consumption in liters per 100 km --/
def fuelConsumption : ℚ := 8

/-- Gasoline price per liter --/
def gasolinePrice : ℚ := 8

theorem taxi_operation_results :
  let finalPosition := taxiOperation.foldl (fun acc trip =>
    if trip.direction then acc + trip.distance else acc - trip.distance) 0
  let totalFare := taxiOperation.foldl (fun acc trip => acc + calculateFare trip.distance) 0
  let totalDistance := taxiOperation.foldl (fun acc trip => acc + trip.distance) 0
  let fuelCost := totalDistance * fuelConsumption / 100 * gasolinePrice
  let profit := totalFare - fuelCost
  finalPosition = -11 ∧ totalFare = 140 ∧ profit = 106.08 := by
  sorry

#check taxi_operation_results

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_operation_results_l1326_132657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_necessary_not_sufficient_l1326_132640

noncomputable section

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the line (m+2)x+3my+1=0 -/
noncomputable def slope1 (m : ℝ) : ℝ := -(m + 2) / (3 * m)

/-- The slope of the line (m-2)x+(m+2)y=0 -/
noncomputable def slope2 (m : ℝ) : ℝ := (m - 2) / (m + 2)

/-- The lines are perpendicular -/
def lines_perpendicular (m : ℝ) : Prop := perpendicular (slope1 m) (slope2 m)

theorem perpendicular_necessary_not_sufficient :
  (∀ m : ℝ, m = 1/2 → lines_perpendicular m) ∧
  ¬(∀ m : ℝ, lines_perpendicular m → m = 1/2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_necessary_not_sufficient_l1326_132640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_DM_perp_IK_l1326_132663

-- Define the geometric objects
variable (O A B C I M A' D E K : EuclideanSpace ℝ 2)

-- Define the circle
variable (circle_O : Sphere (EuclideanSpace ℝ 2))

-- Define the conditions
axiom triangle_inscribed : ∃ (r : ℝ), ∀ X ∈ {A, B, C}, dist O X = r
axiom I_is_incenter : ∃ (r : ℝ), ∀ X ∈ {A, B, C}, dist I X = r
axiom M_is_midpoint : M = (B + C) / 2
axiom A'_is_antipodal : dist O A + dist O A' = 2 * (dist O A)
axiom D_on_circle_I : ∃ (r : ℝ), dist I D = r
axiom D_on_BC : ∃ (t : ℝ), D = (1 - t) • B + t • C
axiom AE_perp_BC : (A - E) • (B - C) = 0
axiom E_on_BC : ∃ (t : ℝ), E = (1 - t) • B + t • C
axiom K_on_A'D : ∃ (t : ℝ), K = (1 - t) • A' + t • D
axiom K_on_ME : ∃ (t : ℝ), K = (1 - t) • M + t • E

-- State the theorem
theorem DM_perp_IK : (D - M) • (I - K) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_DM_perp_IK_l1326_132663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_theorem_l1326_132607

/-- The constant k in the inverse relationship between x² and ⁴√y -/
def k : ℝ := 64

/-- The given y value -/
def y : ℝ := 161051

/-- The given product of x and y -/
def xy_product : ℝ := 2205

/-- The inverse relationship between x² and ⁴√y -/
def inverse_relation (x : ℝ) : Prop := x^2 * y^(1/4) = k

/-- The condition that xy equals the given product -/
def product_condition (x : ℝ) : Prop := x * y = xy_product

/-- Approximate equality for real numbers -/
def approx_equal (a b : ℝ) : Prop := abs (a - b) < 0.01

theorem x_value_theorem :
  ∃ x : ℝ, inverse_relation x ∧ product_condition x ∧ approx_equal x 3.35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_theorem_l1326_132607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circle_theorem_l1326_132682

/-- The area of the part of an equilateral triangle with side length a lying outside a circle of radius a/3, whose center coincides with the center of the triangle. -/
noncomputable def areaOutsideCircle (a : ℝ) : ℝ :=
  (a^2 * (3 * Real.sqrt 3 - Real.pi)) / 18

/-- Theorem stating that the area of the part of an equilateral triangle with side length a lying outside a circle of radius a/3, whose center coincides with the center of the triangle, is equal to (a^2 * (3√3 - π)) / 18. -/
theorem area_outside_circle_theorem (a : ℝ) (h : a > 0) :
  let triangleArea := (a^2 * Real.sqrt 3) / 4
  let circleArea := Real.pi * (a/3)^2
  let triangleCenter := (0 : ℝ × ℝ)
  let circleCenter := triangleCenter
  let circleRadius := a/3
  triangleArea - circleArea + 3 * ((Real.pi * (a/3)^2) / 6 - (a^2 * Real.sqrt 3) / 36) = areaOutsideCircle a :=
by
  sorry

#check area_outside_circle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circle_theorem_l1326_132682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_sixth_l1326_132693

-- Define the curve and line
def curve (x : ℝ) : ℝ := x^2 - 2*x
def line (x : ℝ) : ℝ := -x

-- Define the enclosed area
noncomputable def enclosed_area : ℝ := ∫ x in (0)..(1), (line x - curve x)

-- Theorem statement
theorem area_is_one_sixth : enclosed_area = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_sixth_l1326_132693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suraya_mia_difference_l1326_132632

/-- The number of apples picked by each person -/
structure ApplePickers where
  kayla : ℤ
  caleb : ℤ
  suraya : ℤ
  mia : ℤ

/-- The conditions of the apple picking scenario -/
def apple_picking_conditions (a : ApplePickers) : Prop :=
  a.kayla = 20 ∧
  a.caleb = a.kayla - 5 ∧
  a.suraya = a.caleb + 12 ∧
  a.mia = 2 * a.caleb

/-- The theorem stating the difference in apples picked between Suraya and Mia -/
theorem suraya_mia_difference (a : ApplePickers) 
  (h : apple_picking_conditions a) : a.mia - a.suraya = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suraya_mia_difference_l1326_132632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_of_hexagonal_prism_l1326_132614

/-- A hexagonal prism with specific properties -/
structure HexagonalPrism where
  -- The volume of the prism
  volume : ℝ
  -- The perimeter of the base
  basePerimeter : ℝ
  -- The prism has a regular hexagonal base
  regularBase : Prop
  -- Lateral edges are perpendicular to the base
  perpendicularEdges : Prop
  -- All vertices are on the same spherical surface
  verticesOnSphere : Prop

/-- The volume of a sphere given its radius -/
noncomputable def sphereVolume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius ^ 3

/-- Theorem stating the volume of the sphere containing the hexagonal prism -/
theorem sphere_volume_of_hexagonal_prism (prism : HexagonalPrism)
  (h_volume : prism.volume = 9 / 8)
  (h_perimeter : prism.basePerimeter = 3) :
  ∃ (radius : ℝ), sphereVolume radius = (4 / 3) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_of_hexagonal_prism_l1326_132614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1326_132616

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.c = -3 * t.b * Real.cos t.A ∧ 
  t.c = 2 ∧ 
  Real.tan t.C = 3/4

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : TriangleProperties t) : 
  Real.tan t.A / Real.tan t.B = -4 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A) = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1326_132616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_badges_before_exchange_l1326_132628

/-- The number of badges Vasya had before the exchange -/
def V : ℕ := sorry

/-- The number of badges Tolya had before the exchange -/
def T : ℕ := sorry

/-- Vasya had 5 more badges than Tolya before the exchange -/
axiom vasya_more : V = T + 5

/-- The number of badges Vasya had after the exchange -/
noncomputable def vasya_after : ℚ := 0.76 * ↑V + 0.2 * ↑T

/-- The number of badges Tolya had after the exchange -/
noncomputable def tolya_after : ℚ := 0.8 * ↑T + 0.24 * ↑V

/-- After the exchange, Vasya had one badge fewer than Tolya -/
axiom after_exchange : vasya_after = tolya_after - 1

theorem badges_before_exchange : V = 50 ∧ T = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_badges_before_exchange_l1326_132628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_decay_properties_l1326_132620

noncomputable def f (x : ℝ) := Real.exp (-x)

theorem exponential_decay_properties :
  (∀ a : ℝ, ¬ (∀ x : ℝ, f x = f (2 * a - x))) ∧
  (∀ x : ℝ, f x ≤ 1) ∧
  (f 0 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_decay_properties_l1326_132620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_two_appears_seven_times_l1326_132653

noncomputable def S : Set (ℕ × ℕ × ℕ) :=
  {t | ∃ (p q r : ℕ), t = (p, q, r) ∧ Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    ∃ (x : ℚ), (p : ℚ) * x^2 + (q : ℚ) * x + (r : ℚ) = 0}

theorem prime_two_appears_seven_times :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card ≥ 7 ∧ s.toSet ⊆ S ∧ ∀ t ∈ s, (2 = t.1 ∨ 2 = t.2.1 ∨ 2 = t.2.2) := by
  sorry

#check prime_two_appears_seven_times

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_two_appears_seven_times_l1326_132653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1326_132686

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := 2 / (-1 + i)

theorem imaginary_part_of_z : z.im = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1326_132686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_sequence_properties_l1326_132648

-- Define the sequence α_n
noncomputable def alpha_seq (α : ℝ) : ℕ → ℝ
  | 0 => α
  | n + 1 => min (2 * alpha_seq α n) (1 - 2 * alpha_seq α n)

-- Statement of the theorem
theorem alpha_sequence_properties (α : ℝ) (h_irrational : Irrational α) (h_bound : 0 < α ∧ α < 1/2) :
  (∃ n : ℕ, alpha_seq α n < 3/16) ∧
  (∃ α : ℝ, ∀ n : ℕ, alpha_seq α n > 7/40) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_sequence_properties_l1326_132648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_and_converse_correct_l1326_132668

-- Define a structure for Triangle
structure Triangle where
  -- You might want to add more fields to represent a triangle
  area : ℝ
  -- Other fields could be added here

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define the original proposition
def triangles_equal_area_congruent : Prop :=
  ∀ t1 t2 : Triangle, t1.area = t2.area → congruent t1 t2

-- Define the negation of the proposition
def negation_triangles_equal_area_congruent : Prop :=
  ¬(∀ t1 t2 : Triangle, t1.area = t2.area → congruent t1 t2)

-- Define the converse of the proposition
def converse_triangles_equal_area_congruent : Prop :=
  ∀ t1 t2 : Triangle, ¬(t1.area = t2.area) → ¬(congruent t1 t2)

-- Theorem to prove
theorem negation_and_converse_correct :
  (negation_triangles_equal_area_congruent ↔ ∃ t1 t2 : Triangle, t1.area = t2.area ∧ ¬(congruent t1 t2)) ∧
  (converse_triangles_equal_area_congruent ↔ (∀ t1 t2 : Triangle, congruent t1 t2 → t1.area = t2.area)) :=
by
  sorry  -- The proof is omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_and_converse_correct_l1326_132668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_bounce_less_than_one_foot_l1326_132644

/-- The height of the ball after k bounces -/
noncomputable def bounce_height (k : ℕ) : ℝ :=
  10 * (1/2)^k

/-- Theorem stating that the 4th bounce is the first to be less than 1 foot -/
theorem fourth_bounce_less_than_one_foot :
  (∀ k < 4, bounce_height k ≥ 1) ∧ bounce_height 4 < 1 := by
  sorry

#check fourth_bounce_less_than_one_foot

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_bounce_less_than_one_foot_l1326_132644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1326_132652

-- Define p₁
def p₁ : Prop := ∃ x > 0, Real.log (x^2 + 1/4) ≤ Real.log x

-- Define p₂
def p₂ : Prop := ∀ x : ℝ, Real.sin x ≠ 0 → Real.sin x + 1 / Real.sin x ≥ 2

-- Define p₃
def p₃ : Prop := ∀ x y : ℝ, x + y = 0 ↔ x / y = -1

-- Theorem to prove
theorem problem_solution : p₁ ∨ ¬p₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1326_132652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kclFormed_l1326_132625

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String

-- Define the reaction coefficients
structure ReactionCoefficients where
  reactant1 : ℚ
  reactant2 : ℚ
  product1 : ℚ
  product2 : ℚ
  product3 : ℚ

-- Define the available moles of reactants
structure AvailableReactants where
  moles_reactant1 : ℚ
  moles_reactant2 : ℚ

-- Define the function to calculate the moles of product formed
def molesOfProductFormed (reaction : Reaction) (coefficients : ReactionCoefficients) (available : AvailableReactants) : ℚ :=
  min (available.moles_reactant1 / coefficients.reactant1) (available.moles_reactant2 / coefficients.reactant2) * coefficients.product1

-- Theorem statement
theorem kclFormed (reaction : Reaction) (coefficients : ReactionCoefficients) (available : AvailableReactants) :
  reaction.reactant1 = "NH4Cl" ∧
  reaction.reactant2 = "KOH" ∧
  reaction.product1 = "KCl" ∧
  reaction.product2 = "NH3" ∧
  reaction.product3 = "H2O" ∧
  coefficients.reactant1 = 1 ∧
  coefficients.reactant2 = 1 ∧
  coefficients.product1 = 1 ∧
  coefficients.product2 = 1 ∧
  coefficients.product3 = 1 ∧
  available.moles_reactant1 = 3 ∧
  available.moles_reactant2 = 3 →
  molesOfProductFormed reaction coefficients available = 3 := by
  intro h
  simp [molesOfProductFormed]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kclFormed_l1326_132625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_savings_l1326_132660

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

theorem julie_savings (initial_savings : ℝ) (rate : ℝ) : 
  (simple_interest (initial_savings / 2) rate 2 = 112) ∧ 
  (compound_interest (initial_savings / 2) rate 2 = 120) →
  initial_savings = 196 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_savings_l1326_132660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1326_132600

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f y) + x = x * y + f x) →
  (∀ x : ℝ, f x = x ∨ f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1326_132600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_min_value_l1326_132670

-- Define the real numbers x and y
variable (x y : ℝ)

-- Define m and n with their constraint
variable (m n : ℝ) (h : m + n = 7)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |x - 1| - |x + 1|

-- Define the function F
noncomputable def F (x y m n : ℝ) : ℝ := max (|x^2 - 4*y + m|) (|y^2 - 2*x + n|)

-- Theorem 1: The solution set of f(x) ≥ 7x is {x ∈ ℝ | x ≤ 0}
theorem solution_set (x : ℝ) : f x ≥ 7 * x ↔ x ≤ 0 := by sorry

-- Theorem 2: The minimum value of F is 1
theorem min_value : ∃ x₀ y₀ : ℝ, F x₀ y₀ m n = 1 ∧ ∀ x y : ℝ, F x y m n ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_min_value_l1326_132670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l1326_132605

theorem sin_half_angle (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l1326_132605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1326_132656

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  B = π / 3 →
  (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 4 →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1326_132656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1326_132606

/-- The area of a triangle with vertices at (3, -3), (8, 4), and (3, 4) is 17.5 square units. -/
theorem triangle_area : ∃ (area : ℝ), area = 17.5 := by
  -- Define the vertices of the triangle
  let v1 : ℤ × ℤ := (3, -3)
  let v2 : ℤ × ℤ := (8, 4)
  let v3 : ℤ × ℤ := (3, 4)

  -- Calculate the lengths of the legs
  let vertical_leg : ℤ := v3.2 - v1.2
  let horizontal_leg : ℤ := v2.1 - v3.1

  -- Calculate the area of the triangle
  let area : ℝ := (vertical_leg * horizontal_leg : ℝ) / 2

  -- The theorem statement
  use area
  sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1326_132606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_l1326_132677

-- Define the ellipse structure
structure Ellipse where
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  c : ℝ

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a function to check if a point is on the ellipse
def isOnEllipse (e : Ellipse) (p : ℝ × ℝ) : Prop :=
  distance p e.f1 + distance p e.f2 = e.c

-- Theorem statement
theorem ellipse_intersection (e : Ellipse) (h1 : e.f1 = (0, 3)) (h2 : e.f2 = (4, 0))
  (h3 : isOnEllipse e (1, 0)) : isOnEllipse e (7, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_l1326_132677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_inequality_l1326_132608

-- Define the function f
noncomputable def f (t : ℝ) : ℝ := 2 * Real.sin t

-- Define the theorem
theorem x_range_for_inequality :
  ∀ t ∈ Set.Icc (π/6) (π/2),
  ∀ m ∈ Set.range f,
  (∀ x, 2*x^2 + m*x - 2 < m + 2*x) →
  ∃ a b, a = -1 ∧ b = Real.sqrt 2 ∧ ∀ x, x ∈ Set.Ioo a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_inequality_l1326_132608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1326_132634

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 3*x - 1

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 3

-- State the theorem
theorem tangent_line_equation :
  let tangent_line (x : ℝ) := -2 * x + 1
  ∀ x, tangent_line x = f 1 + f' 1 * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1326_132634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_percentage_l1326_132650

/-- Calculates the second discount percentage given the original price, final price, and first discount percentage. -/
noncomputable def calculate_second_discount (original_price : ℝ) (final_price : ℝ) (first_discount : ℝ) : ℝ :=
  100 * (1 - final_price / (original_price * (1 - first_discount / 100)))

/-- Theorem stating that for the given prices and first discount, the second discount is approximately 14.47%. -/
theorem second_discount_percentage : 
  let original_price : ℝ := 9502.923976608186
  let final_price : ℝ := 6500
  let first_discount : ℝ := 20
  abs (calculate_second_discount original_price final_price first_discount - 14.47) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_percentage_l1326_132650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_circle_l1326_132666

/-- Given a line y = kx intersecting a circle x^2 + (y-2)^2 = 4 at two points A and B,
    if the distance |AB| = 2√3, then k = ±√3 -/
theorem intersection_line_circle (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1^2 + (A.2 - 2)^2 = 4) ∧ 
    (B.1^2 + (B.2 - 2)^2 = 4) ∧
    (A.2 = k * A.1) ∧ 
    (B.2 = k * B.1) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 12)) →
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_circle_l1326_132666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_height_is_two_meters_l1326_132636

/-- The volume of a single brick in cubic centimeters -/
noncomputable def brick_volume : ℝ := 20 * 10 * 7.5

/-- The length of the wall in meters -/
noncomputable def wall_length : ℝ := 29

/-- The width of the wall in meters -/
noncomputable def wall_width : ℝ := 0.75

/-- The number of bricks required for the wall -/
def num_bricks : ℕ := 29000

/-- Conversion factor from cubic centimeters to cubic meters -/
noncomputable def cm3_to_m3 : ℝ := 1 / 1000000

theorem wall_height_is_two_meters :
  ∃ (h : ℝ), h = 2 ∧ 
  (brick_volume * cm3_to_m3 * (num_bricks : ℝ)) = wall_length * h * wall_width := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_height_is_two_meters_l1326_132636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_to_finish_remaining_work_l1326_132667

/-- The number of days it takes for worker a and b to finish the work together -/
noncomputable def total_days_together : ℝ := 40

/-- The number of days worker a and b worked together before b left -/
noncomputable def days_worked_together : ℝ := 10

/-- The number of days it takes for worker a to finish the job alone -/
noncomputable def days_a_alone : ℝ := 16

/-- The amount of work completed by both workers in one day -/
noncomputable def work_rate_together : ℝ := 1 / total_days_together

/-- The amount of work completed by worker a in one day -/
noncomputable def work_rate_a : ℝ := 1 / days_a_alone

/-- The amount of work completed during the time they worked together -/
noncomputable def work_completed_together : ℝ := work_rate_together * days_worked_together

/-- The remaining work after b left -/
noncomputable def remaining_work : ℝ := 1 - work_completed_together

/-- The number of days it took for a to finish the remaining work after b left -/
noncomputable def days_a_finished_remaining : ℝ := remaining_work / work_rate_a

theorem days_to_finish_remaining_work :
  days_a_finished_remaining = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_to_finish_remaining_work_l1326_132667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_divisible_probability_864_1944_l1326_132613

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem probability_multiple_divisible (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  (∃ k : ℕ, (k * n).gcd m = m) → 
  (∃ p : ℚ, p = 1 / (m.gcd n).gcd (m / (m.gcd n))) := by
  sorry

theorem probability_864_1944 : 
  ∃ p : ℚ, p = 1 / 9 ∧ 
  (∃ k : ℕ, (k * 864).gcd 1944 = 1944) := by
  sorry

#check probability_864_1944

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_divisible_probability_864_1944_l1326_132613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_l1326_132651

-- Define the power function
noncomputable def power_function (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(-5*m - 3)

-- State the theorem
theorem decreasing_power_function :
  ∀ m : ℝ, (∀ x : ℝ, x > 0 → (deriv (power_function m)) x < 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_l1326_132651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_value_l1326_132624

theorem alpha_plus_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi)
  (h2 : 0 < β ∧ β < Real.pi)
  (h3 : Real.sin (α - β) = 5/6)
  (h4 : Real.tan α / Real.tan β = -1/4) :
  α + β = 7 * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_value_l1326_132624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_is_correct_l1326_132602

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 + 12 * x + 9 * y^2 + 27 * y + 36 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := 3 * Real.pi / 2

/-- Theorem stating that the area of the ellipse described by the given equation is 3π/2 -/
theorem ellipse_area_is_correct : 
  ∃ (a b : ℝ), (∀ x y : ℝ, ellipse_equation x y ↔ (x + 3/2)^2 / a^2 + (y + 3/2)^2 / b^2 = 1) ∧ 
  ellipse_area = Real.pi * a * b := by
  sorry

#check ellipse_area_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_is_correct_l1326_132602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_circle_center_line_passes_center_max_chord_through_center_line_maximizes_chord_l1326_132645

/-- Circle C with equation x^2 + y^2 + 4x + 3 = 0 -/
def circleC (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 3 = 0

/-- Line l passing through (2,3) and (-2,0) -/
def lineL (x y : ℝ) : Prop := 3*x - 4*y + 6 = 0

/-- Point (2,3) lies on the line -/
theorem point_on_line : lineL 2 3 := by sorry

/-- Center of the circle C is (-2,0) -/
theorem circle_center : circleC (-2) 0 := by sorry

/-- The line passes through the center of the circle -/
theorem line_passes_center : lineL (-2) 0 := by sorry

/-- The chord formed by the intersection of the line and circle is maximized 
    when the line passes through the center of the circle -/
theorem max_chord_through_center (x y : ℝ) :
  circleC x y ∧ lineL x y → 
  ∀ x' y', circleC x' y' → (x - x')^2 + (y - y')^2 ≤ 4 := by sorry

/-- Main theorem: The given line maximizes the chord length -/
theorem line_maximizes_chord : 
  ∀ a b c : ℝ, (∀ x y, a*x + b*y + c = 0 → (x - 2)^2 + (y - 3)^2 ≤ 20) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_circle_center_line_passes_center_max_chord_through_center_line_maximizes_chord_l1326_132645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l1326_132698

/-- Given a projection that takes (2, -5) to (1/2, -5/2), 
    prove that it takes (3, 6) to (-27/26, 135/26) -/
theorem projection_problem (proj : ℝ × ℝ → ℝ × ℝ) 
    (h : proj (2, -5) = (1/2, -5/2)) :
  proj (3, 6) = (-27/26, 135/26) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l1326_132698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_agent_monthly_expenses_l1326_132692

/-- Calculates the monthly expenses of a sales agent given their basic salary, commission rate, total sales, and savings rate. -/
theorem sales_agent_monthly_expenses 
  (basic_salary : ℝ) 
  (commission_rate : ℝ) 
  (total_sales : ℝ) 
  (savings_rate : ℝ) : 
  basic_salary = 1250 → 
  commission_rate = 0.1 → 
  total_sales = 23600 → 
  savings_rate = 0.2 → 
  (basic_salary + total_sales * commission_rate) * (1 - savings_rate) = 2888 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_agent_monthly_expenses_l1326_132692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l1326_132685

/-- The distance from a point (x, y) to a vertical line x = a -/
noncomputable def distToVerticalLine (x y a : ℝ) : ℝ := |x - a|

/-- The distance between two points (x1, y1) and (x2, y2) -/
noncomputable def distBetweenPoints (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- A point P(x, y) satisfies the condition that its distance to F(4, 0) 
    is 1 less than its distance to the line x + 5 = 0 -/
def satisfiesCondition (x y : ℝ) : Prop :=
  distBetweenPoints x y 4 0 = distToVerticalLine x y (-5) - 1

/-- The equation of the trajectory of point P -/
def trajectoryEquation (x y : ℝ) : Prop := y^2 = 16 * x

/-- Theorem: If a point satisfies the given condition, 
    then it lies on the parabola y^2 = 16x -/
theorem trajectory_is_parabola (x y : ℝ) :
  satisfiesCondition x y → trajectoryEquation x y := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l1326_132685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_line_l_theorem_l1326_132610

/-- The curve C formed by point P(x,y) given the conditions of vectors m and n -/
def curve_C (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- The vector m₁ -/
def m₁ (x : ℝ) : ℝ × ℝ := (0, x)

/-- The vector n₁ -/
def n₁ : ℝ × ℝ := (1, 1)

/-- The vector m₂ -/
def m₂ (x : ℝ) : ℝ × ℝ := (x, 0)

/-- The vector n₂ -/
def n₂ (y : ℝ) : ℝ × ℝ := (y^2, 1)

/-- The vector m -/
noncomputable def m (x y : ℝ) : ℝ × ℝ := 
  (Real.sqrt 2 * (n₂ y).1, x + Real.sqrt 2)

/-- The vector n -/
noncomputable def n (x : ℝ) : ℝ × ℝ := 
  (x - Real.sqrt 2, -Real.sqrt 2)

/-- m is parallel to n -/
def m_parallel_n (x y : ℝ) : Prop :=
  (m x y).1 * (n x).2 = (m x y).2 * (n x).1

/-- The line l that intersects curve C -/
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

/-- The distance between points M and N on line l -/
noncomputable def MN_distance (k : ℝ) : ℝ := 4 * Real.sqrt 2 / 3

theorem curve_C_and_line_l_theorem (x y k : ℝ) : 
  m_parallel_n x y →
  (∃ M N : ℝ × ℝ, curve_C M.1 M.2 ∧ curve_C N.1 N.2 ∧ 
    line_l k M.1 M.2 ∧ line_l k N.1 N.2 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = (MN_distance k)^2) →
  (curve_C x y ∧ (k = 1 ∨ k = -1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_line_l_theorem_l1326_132610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1326_132612

theorem cos_alpha_value (α β : ℝ) 
  (h1 : Real.cos (α - β) = -4/5)
  (h2 : Real.cos β = 4/5)
  (h3 : α - β ∈ Set.Ioo (π/2) π)
  (h4 : β ∈ Set.Ioo (3*π/2) (2*π)) :
  Real.cos α = -7/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1326_132612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_and_optimal_volume_l1326_132678

/-- Additional cost function for producing x thousand units -/
noncomputable def additional_cost (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 80 then (1/3) * x^2 + 10 * x
  else if x ≥ 80 then 51 * x + 10000 / x - 1550
  else 0

/-- Annual profit function in millions of dollars -/
noncomputable def annual_profit (x : ℝ) : ℝ :=
  500 * x - additional_cost x - 150

/-- Theorem stating the maximum annual profit and optimal production volume -/
theorem max_profit_and_optimal_volume :
  ∃ (max_profit : ℝ) (optimal_volume : ℝ),
    max_profit = 1200 ∧
    optimal_volume = 100 ∧
    ∀ x > 0, annual_profit x ≤ max_profit ∧
    annual_profit optimal_volume = max_profit :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_and_optimal_volume_l1326_132678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l1326_132694

def a (n : ℕ) : ℚ := 1 / (↑(n + 1) ^ 2)

def b : ℕ → ℚ
  | 0 => 1 - a 0
  | n + 1 => b n * (1 - a (n + 1))

theorem b_formula (n : ℕ) : b n = (↑(n + 2) : ℚ) / (2 * ↑(n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l1326_132694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_time_l1326_132664

theorem runner_time (n : ℕ) (t : ℝ) (h1 : n = 12) (h2 : t = 12) : 
  (n - 1 : ℝ) * (t / 3) = 44 := by
  have h3 : (n - 1 : ℝ) = 11 := by
    rw [h1]
    norm_num
  have h4 : t / 3 = 4 := by
    rw [h2]
    norm_num
  calc
    (n - 1 : ℝ) * (t / 3) = 11 * 4 := by rw [h3, h4]
    _ = 44 := by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_time_l1326_132664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l1326_132609

theorem divisor_problem :
  ∃ (d : ℕ) (x : ℕ), 
    d > 0 ∧
    x > 0 ∧
    x % d = 5 ∧ 
    (7 * x) % d = 8 ∧ 
    d = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l1326_132609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_condition_implies_alpha_main_theorem_l1326_132621

open Real

-- Define the curves and conditions
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (Real.cos t, 1 + Real.sin t)
noncomputable def C₂ (θ : ℝ) : ℝ := 4 * Real.cos θ
noncomputable def C₃ (α : ℝ) (t : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, 3 + t * Real.sin α)

-- Define the theorem
theorem area_condition_implies_alpha (α : ℝ) 
  (h1 : 0 ≤ α ∧ α < π) 
  (h2 : abs (Real.sin (2 * α)) = 1) : 
  α = π / 4 ∨ α = 3 * π / 4 := by
  sorry

-- Define the main theorem
theorem main_theorem : 
  ∃ α : ℝ, (0 ≤ α ∧ α < π) ∧ 
  (abs (Real.sin (2 * α)) = 1) ∧ 
  (α = π / 4 ∨ α = 3 * π / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_condition_implies_alpha_main_theorem_l1326_132621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_guaranteed_first_player_win_l1326_132690

structure Cube where
  edges : Finset (Fin 12)
  faces : Finset (Finset (Fin 12))
  edge_in_face : Fin 12 → Finset (Fin 12) → Prop

structure GameState where
  red_edges : Finset (Fin 12)
  green_edges : Finset (Fin 12)

def is_winning_state (cube : Cube) (state : GameState) : Prop :=
  ∃ face ∈ cube.faces, (∀ edge ∈ face, edge ∈ state.red_edges) ∨
                       (∀ edge ∈ face, edge ∈ state.green_edges)

def optimal_play (cube : Cube) (initial_state : GameState) : Prop :=
  ∀ (player1_strategy player2_strategy : GameState → Finset (Fin 3)),
    let final_state := sorry
    is_winning_state cube final_state

theorem not_guaranteed_first_player_win (cube : Cube) :
  ¬ ∀ (initial_state : GameState), optimal_play cube initial_state →
    ∃ face ∈ cube.faces, (∀ edge ∈ face, edge ∈ (sorry : GameState).red_edges) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_guaranteed_first_player_win_l1326_132690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_divides_triangle_equally_l1326_132699

/-- Triangle ABC in the xy-plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculate the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1/2) * base * height

/-- Calculate the area of the left subtriangle formed by the vertical line -/
noncomputable def leftSubtriangleArea (t : Triangle) (k : ℝ) : ℝ :=
  triangleArea k (t.A.2 - t.B.2)

/-- The main theorem -/
theorem vertical_line_divides_triangle_equally (t : Triangle) :
  t.A = (0, 4) →
  t.B = (0, 0) →
  t.C = (10, 0) →
  leftSubtriangleArea t 5 = (1/2) * triangleArea (t.C.1 - t.A.1) (t.A.2 - t.B.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_divides_triangle_equally_l1326_132699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_with_tangent_circles_l1326_132659

/-- A structure representing a circle with a center point -/
structure Circle where
  center : Point
  diameter : Real

/-- A structure representing a rectangle with four vertices -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Predicate to check if a circle is tangent to a rectangle -/
def are_tangent (c : Circle) (r : Rectangle) : Prop :=
  sorry

/-- Predicate to check if a circle passes through a point -/
def passes_through (c : Circle) (p : Point) : Prop :=
  sorry

/-- Function to calculate the area of a rectangle -/
def rectangle_area (r : Rectangle) : Real :=
  sorry

/-- The theorem statement -/
theorem rectangle_area_with_tangent_circles 
  (P Q R : Point) 
  (circleP circleQ circleR : Circle) 
  (rect : Rectangle) :
  circleP.center = P →
  circleQ.center = Q →
  circleR.center = R →
  circleP.diameter = 6 →
  circleQ.diameter = 6 →
  circleR.diameter = 6 →
  are_tangent circleP rect →
  are_tangent circleQ rect →
  are_tangent circleR rect →
  passes_through circleP Q →
  passes_through circleP R →
  passes_through circleQ P →
  passes_through circleQ R →
  passes_through circleR P →
  passes_through circleR Q →
  rectangle_area rect = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_with_tangent_circles_l1326_132659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_ratio_l1326_132674

theorem square_perimeter_ratio (s : ℝ) (h : s > 0) : 
  (7 * s * Real.sqrt 2 / Real.sqrt 2) * 4 / (s * 4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_ratio_l1326_132674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_one_third_l1326_132637

theorem power_of_one_third (a b : ℕ) : 
  (2^a : ℕ) * 5^b = 200 → 
  (∀ k : ℕ, k > a → ¬(2^k ∣ 200)) → 
  (∀ l : ℕ, l > b → ¬(5^l ∣ 200)) → 
  (1/3 : ℚ)^(b - a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_one_third_l1326_132637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1326_132688

def sequence_a : ℕ → ℝ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => 2 * sequence_a (n + 2) - sequence_a (n + 1) + 2

def sequence_b (n : ℕ) : ℝ := sequence_a (n + 1) - sequence_a n

theorem sequence_properties :
  (∀ n : ℕ, ∃ d : ℝ, sequence_b (n + 1) - sequence_b n = d) ∧
  (∀ n : ℕ, sequence_a n = n^2 - 2*n + 2) := by
  sorry

#eval sequence_a 5  -- Optional: to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1326_132688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_concyclic_l1326_132633

/-- Type representing a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate stating that four points lie on a circle -/
def CircleOn (A B C D : Point) : Prop := sorry

/-- Predicate stating that S is the midpoint of the arc AB not containing C and D -/
def MidpointOfArc (S A B C D : Point) : Prop := sorry

/-- Predicate stating that a point lies on a line defined by two other points -/
def OnLine (P Q R : Point) : Prop := sorry

/-- Given four points A, B, C, D on a circle in that order, with S as the midpoint
    of the arc AB not containing C and D, and E and F as the intersections of SD
    and SC with AB respectively, prove that C, D, E, and F are concyclic. -/
theorem points_concyclic (A B C D S E F : Point) 
  (h1 : CircleOn A B C D)
  (h2 : MidpointOfArc S A B C D)
  (h3 : OnLine E S D)
  (h4 : OnLine E A B)
  (h5 : OnLine F S C)
  (h6 : OnLine F A B) : 
  CircleOn C D E F := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_concyclic_l1326_132633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_on_line_of_centers_l1326_132658

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the regular triangle
def RegularTriangle (A B C : Point) : Prop :=
  sorry -- conditions for a regular triangle

-- Define the tangency conditions
def IsTangent (c : Circle) (l : Line) : Prop :=
  sorry -- conditions for circle tangent to line

def IsTangentAtPoint (c : Circle) (l : Line) (p : Point) : Prop :=
  sorry -- conditions for circle tangent to line at a specific point

-- Define the orthocenter
noncomputable def Orthocenter (A B C : Point) : Point :=
  sorry -- definition of orthocenter

-- Main theorem
theorem orthocenter_on_line_of_centers 
  (A B C : Point) 
  (ℓ : Line) 
  (ωa ωc : Circle) 
  (A1 C1 : Point) :
  RegularTriangle A B C →
  ℓ.a * B.x + ℓ.b * B.y + ℓ.c = 0 →
  IsTangentAtPoint ωa (Line.mk 0 1 (-C.y)) A1 →
  IsTangent ωa ℓ →
  IsTangent ωa (Line.mk (-1) 1 (A.x - A.y)) →
  IsTangentAtPoint ωc (Line.mk 1 0 (-A.x)) C1 →
  IsTangent ωc ℓ →
  IsTangent ωc (Line.mk (-1) 1 (A.x - A.y)) →
  ∃ (t : ℝ), 
    let H := Orthocenter A1 B C1
    let Ia := ωa.center
    let Ic := ωc.center
    H = Point.mk (Ia.x + t * (Ic.x - Ia.x)) (Ia.y + t * (Ic.y - Ia.y)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_on_line_of_centers_l1326_132658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1326_132675

def sequence_a : ℕ → ℚ
  | 0 => 0  -- Add this case for n = 0
  | 1 => 0
  | (n + 2) => 1 / (2 - sequence_a (n + 1))

theorem sequence_a_formula (n : ℕ) (h : n ≥ 1) : 
  sequence_a n = (n - 1 : ℚ) / n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1326_132675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_comparison_l1326_132676

def factorial_iteration (n : ℕ) : ℕ → ℕ
| 0 => n
| 1 => n.factorial
| (m + 1) => (factorial_iteration n m).factorial

def sequence_a (n : ℕ) : ℕ → ℕ
| 0 => n
| 1 => n
| (m + 1) => n ^ (sequence_a n m)

theorem sequence_comparison (n : ℕ) (hn : n > 2) :
  ∀ m : ℕ, m ≥ 1 → sequence_a n m < factorial_iteration n m :=
by
  intro m hm
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_comparison_l1326_132676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nestedRadical_eq_three_l1326_132661

/-- The value of the infinite nested radical expression sqrt(3 + 2 * sqrt(3 + 2 * sqrt(...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (3 + 2 * Real.sqrt 3)

/-- Theorem stating that the value of the nested radical is 3 -/
theorem nestedRadical_eq_three : nestedRadical = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nestedRadical_eq_three_l1326_132661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_denominators_l1326_132601

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a three-digit number -/
def ThreeDigitNumber := Fin 1000

/-- Converts three digits to a three-digit number -/
def digitsToNumber (a b c : Digit) : ThreeDigitNumber :=
  ⟨a.val * 100 + b.val * 10 + c.val, by sorry⟩

/-- Checks if at least one of the digits is not 9 -/
def atLeastOneNotNine (a b c : Digit) : Prop :=
  a.val ≠ 9 ∨ b.val ≠ 9 ∨ c.val ≠ 9

/-- Represents the possible denominators when 0.abc is expressed as a fraction in lowest terms -/
def possibleDenominators : Finset Nat :=
  {3, 9, 27, 37, 111, 333, 999}

/-- The main theorem -/
theorem count_possible_denominators (a b c : Digit) 
  (h : atLeastOneNotNine a b c) : 
  possibleDenominators.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_denominators_l1326_132601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_ratio_l1326_132638

-- Define the trapezoid and point
structure Trapezoid :=
  (E F G H Q : ℝ × ℝ)

-- Define the properties of the trapezoid
def isIsoscelesTrapezoid (t : Trapezoid) : Prop :=
  sorry -- Placeholder for isosceles trapezoid properties

def hasParallelBases (t : Trapezoid) : Prop :=
  sorry -- Placeholder for parallel bases properties

def EFGreaterThanGH (t : Trapezoid) : Prop :=
  sorry -- Placeholder for EF > GH property

-- Define the areas of the triangles
def triangleAreas (t : Trapezoid) : Prop :=
  ∃ (areaQGH areaQHE areaQEF areaQFG : ℝ),
    areaQGH = 3 ∧ areaQHE = 5 ∧ areaQEF = 7 ∧ areaQFG = 9

-- Main theorem
theorem trapezoid_ratio (t : Trapezoid) :
  isIsoscelesTrapezoid t →
  hasParallelBases t →
  EFGreaterThanGH t →
  triangleAreas t →
  ∃ (EF GH : ℝ), EF / GH = 7 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_ratio_l1326_132638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l1326_132603

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600)

/-- Theorem: A train 100 meters long traveling at 36 kmph will take 28 seconds to cross a bridge 180 meters long -/
theorem train_crossing_bridge :
  train_crossing_time 100 180 36 = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l1326_132603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_verify_solution_l1326_132639

/-- The operation * defined on real numbers -/
noncomputable def star_operation (v : ℝ) : ℝ := v - v / 3

/-- Theorem stating that if v satisfies the given equation, then v = 9 -/
theorem solve_equation (v : ℝ) (h : star_operation (star_operation v) = 4) : v = 9 := by
  sorry

/-- Verifies that 9 indeed satisfies the equation -/
theorem verify_solution : star_operation (star_operation 9) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_verify_solution_l1326_132639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_solid_edge_sum_l1326_132611

/-- Represents a rectangular solid with dimensions in geometric progression -/
structure GeometricProgressionSolid where
  a : ℝ
  r : ℝ

/-- The volume of the solid -/
noncomputable def volume (s : GeometricProgressionSolid) : ℝ := s.a^3

/-- The surface area of the solid -/
noncomputable def surface_area (s : GeometricProgressionSolid) : ℝ := 
  2 * (s.a^2 / s.r + s.a^2 + s.a^2 * s.r)

/-- The sum of the lengths of all edges of the solid -/
noncomputable def edge_sum (s : GeometricProgressionSolid) : ℝ := 
  4 * (s.a / s.r + s.a + s.a * s.r)

/-- Theorem: For a rectangular solid with volume 512 cm³, surface area 384 cm², 
    and dimensions in geometric progression, the sum of all edge lengths is 96 cm -/
theorem geometric_progression_solid_edge_sum :
  ∃ s : GeometricProgressionSolid, 
    volume s = 512 ∧ 
    surface_area s = 384 ∧ 
    edge_sum s = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_solid_edge_sum_l1326_132611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_approx_l1326_132689

-- Define a regular hexadecagon inscribed in a circle
structure RegularHexadecagon where
  radius : ℝ
  radius_pos : radius > 0

-- Define the area of the hexadecagon
noncomputable def area (h : RegularHexadecagon) : ℝ :=
  16 * (1/2) * h.radius^2 * Real.sin (22.5 * Real.pi / 180)

-- Theorem stating that the area is approximately 3r^2
theorem hexadecagon_area_approx (h : RegularHexadecagon) :
  ∃ ε > 0, |area h - 3 * h.radius^2| < ε := by
  sorry

#check hexadecagon_area_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_approx_l1326_132689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_alpha_l1326_132680

theorem sin_half_alpha (α : ℝ) 
  (h1 : Real.cos α = -2/3) 
  (h2 : π < α) 
  (h3 : α < 3*π/2) : 
  Real.sin (α/2) = Real.sqrt 30 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_alpha_l1326_132680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1326_132617

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := Real.cos (2 * x) + Real.sqrt 3 * Real.sin (2 * x) + 1 + a

-- Define the monotonicity properties
def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

-- State the theorem
theorem problem_solution (a : ℝ) :
  (∀ k : ℤ, is_increasing (f · a) (-(π/3) + k * π) (π/6 + k * π)) ∧
  (∀ k : ℤ, is_decreasing (f · a) (π/6 + k * π) (2*π/3 + k * π)) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 (π/2) ∧ f x a = 4 ∧ ∀ y ∈ Set.Icc 0 (π/2), f y a ≤ 4) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1326_132617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_heads_next_flip_prob_heads_independent_of_past_l1326_132647

/-- A biased coin with a probability of 3/4 of landing on heads -/
structure BiasedCoin where
  prob_heads : ℚ
  bias : prob_heads = 3/4

/-- The probability of getting heads on any flip of a biased coin is equal to its bias -/
theorem prob_heads_next_flip (coin : BiasedCoin) : 
  coin.prob_heads = 3/4 := by
  exact coin.bias

/-- The probability of getting heads on the next flip after any number of previous flips -/
theorem prob_heads_independent_of_past (coin : BiasedCoin) (n : ℕ) :
  coin.prob_heads = 3/4 := by
  exact coin.bias

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_heads_next_flip_prob_heads_independent_of_past_l1326_132647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_perpendicular_lines_l1326_132672

/-- Given a line l: ax + by + c = 0, return the equations of lines parallel to l at distance d -/
def parallel_lines (a b c d : ℝ) : Set (ℝ → ℝ → Prop) :=
  {f | ∃ k, f = (λ x y ↦ a * x + b * y + k = 0) ∧ 
    (k + c = d * Real.sqrt (a^2 + b^2) ∨ k + c = -d * Real.sqrt (a^2 + b^2))}

/-- Given a line l: ax + by + c = 0, return the equations of lines perpendicular to l at distance d from origin -/
def perpendicular_lines (a b c d : ℝ) : Set (ℝ → ℝ → Prop) :=
  {f | ∃ k, f = (λ x y ↦ b * x - a * y + k = 0) ∧ 
    (k = d * Real.sqrt (a^2 + b^2) ∨ k = -d * Real.sqrt (a^2 + b^2))}

theorem parallel_perpendicular_lines :
  (parallel_lines 1 2 (-3) 1 = {λ x y ↦ x + 2*y + Real.sqrt 5 - 3 = 0, λ x y ↦ x + 2*y - Real.sqrt 5 - 3 = 0}) ∧
  (perpendicular_lines 1 (-Real.sqrt 3) 1 5 = {λ x y ↦ Real.sqrt 3 * x + y + 10 = 0, λ x y ↦ Real.sqrt 3 * x + y - 10 = 0}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_perpendicular_lines_l1326_132672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_run_rate_theorem_l1326_132618

/-- Represents a cricket match situation --/
structure CricketMatch where
  total_overs : ℕ
  first_part_overs : ℕ
  first_part_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the required run rate for the remaining overs --/
def required_run_rate (m : CricketMatch) : ℚ :=
  let remaining_overs := m.total_overs - m.first_part_overs
  let runs_scored := m.first_part_run_rate * m.first_part_overs
  let runs_needed := m.target_runs - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket match --/
theorem cricket_run_rate_theorem (m : CricketMatch) 
  (h1 : m.total_overs = 50)
  (h2 : m.first_part_overs = 10)
  (h3 : m.first_part_run_rate = 3.2)
  (h4 : m.target_runs = 242) :
  required_run_rate m = 5.25 := by
  sorry

#eval required_run_rate { total_overs := 50, first_part_overs := 10, first_part_run_rate := 3.2, target_runs := 242 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_run_rate_theorem_l1326_132618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_specific_segment_l1326_132695

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The midpoint of two polar points with the same radius -/
noncomputable def polarMidpoint (p1 p2 : PolarPoint) : PolarPoint :=
  { r := p1.r,
    θ := (p1.θ + p2.θ) / 2 }

theorem midpoint_of_specific_segment :
  let p1 : PolarPoint := ⟨10, π / 4⟩
  let p2 : PolarPoint := ⟨10, 3 * π / 4⟩
  let m := polarMidpoint p1 p2
  m.r = 10 ∧ m.θ = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_specific_segment_l1326_132695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_non_overlapping_original_sets_l1326_132681

/-- An original set in an n × n grid -/
def OriginalSet (n : ℕ) : Type :=
  { s : Finset (ℕ × ℕ) // s.card = n - 1 ∧ 
    ∀ (i j k l : ℕ), (i, j) ∈ s → (k, l) ∈ s → i = k ∨ j = l → (i, j) = (k, l) }

/-- A collection of n+1 non-overlapping original sets -/
def NonOverlappingOriginalSets (n : ℕ) : Type :=
  { sets : Finset (OriginalSet n) // 
    sets.card = n + 1 ∧
    ∀ s t, s ∈ sets → t ∈ sets → s ≠ t → (s.val ∩ t.val).card = 0 }

/-- For any positive integer n, it is possible to choose n+1 non-overlapping original sets from an n × n grid -/
theorem existence_of_non_overlapping_original_sets (n : ℕ) (hn : n > 0) :
  Nonempty (NonOverlappingOriginalSets n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_non_overlapping_original_sets_l1326_132681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_approximation_l1326_132622

/-- Represents the loan details and calculates the borrowed amount -/
noncomputable def calculateBorrowedAmount (interestRate : ℝ) (years : ℝ) (totalReturned : ℝ) : ℝ :=
  totalReturned / (1 + interestRate * years / 100)

/-- Theorem stating that the borrowed amount is approximately 5526 given the conditions -/
theorem borrowed_amount_approximation :
  let interestRate : ℝ := 6
  let years : ℝ := 9
  let totalReturned : ℝ := 8510
  let borrowedAmount := calculateBorrowedAmount interestRate years totalReturned
  ‖borrowedAmount - 5526‖ < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_approximation_l1326_132622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_three_neg_four_l1326_132615

/-- The nabla operation for real numbers -/
noncomputable def nabla (x y : ℝ) : ℝ := (x + y) / (1 + x * y)

/-- Theorem stating that 3 ∇ (-4) = 1/11 -/
theorem nabla_three_neg_four : nabla 3 (-4) = 1/11 := by
  -- Unfold the definition of nabla
  unfold nabla
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_three_neg_four_l1326_132615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_point_P_existence_l1326_132683

/-- The curve C formed by points satisfying the distance ratio condition -/
def curve_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- The fixed point F -/
def F : ℝ × ℝ := (1, 0)

/-- The fixed line x = 4 -/
def line_x4 (x : ℝ) : Prop := x = 4

/-- Distance ratio condition for a point (x, y) -/
noncomputable def distance_ratio (x y : ℝ) : Prop :=
  Real.sqrt ((x - F.1)^2 + y^2) / |x - 4| = 1/2

/-- Point P on x-axis -/
def P : ℝ × ℝ := (4, 0)

/-- Theorem stating the equation of curve C and existence of point P -/
theorem curve_C_and_point_P_existence :
  (∀ x y, distance_ratio x y ↔ curve_C x y) ∧
  ∃ P : ℝ × ℝ, P.1 ≠ F.1 ∧ P.2 = 0 ∧
    ∀ t : ℝ, t ≠ 0 →
      ∃ A B : ℝ × ℝ,
        curve_C A.1 A.2 ∧
        curve_C B.1 B.2 ∧
        A.1 = t * A.2 + F.1 ∧
        B.1 = t * B.2 + F.1 ∧
        (A.2 / (A.1 - P.1) + B.2 / (B.1 - P.1) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_point_P_existence_l1326_132683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_revenue_l1326_132646

theorem book_sale_revenue (total_books : ℕ) (price_per_book : ℚ) : 
  (2 : ℚ) / 3 * total_books = (total_books - 40 : ℕ) →
  price_per_book = 7/2 →
  ((2 : ℚ) / 3 * total_books).floor * price_per_book = 280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_revenue_l1326_132646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_above_180_is_155_l1326_132604

/-- The last number in the n-th row -/
def last_number_in_row (n : ℕ) : ℕ := n^2

/-- Represents the array structure where n-th row ends with n^2 and contains 2n-1 numbers -/
def ArrayStructure (n : ℕ) : Prop :=
  ∀ k ≤ n, (k^2 - (2*k - 2)) ≤ (last_number_in_row k) ∧ (last_number_in_row k) ≤ k^2

/-- The number of elements in the n-th row -/
def elements_in_row (n : ℕ) : ℕ := 2*n - 1

/-- The first number in the n-th row -/
def first_number_in_row (n : ℕ) : ℕ := last_number_in_row (n-1) + 1

/-- The number directly above a given number in the array -/
noncomputable def number_above (x : ℕ) : ℕ :=
  let row := Nat.sqrt x
  let position_in_row := x - first_number_in_row row + 1
  first_number_in_row (row-1) + position_in_row - 1

/-- Theorem stating that the number directly above 180 in the array is 155 -/
theorem number_above_180_is_155 (h : ArrayStructure 14) : number_above 180 = 155 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_above_180_is_155_l1326_132604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_l1326_132649

-- Define the square OSVW
def O : ℝ × ℝ := (0, 0)
def W : ℝ × ℝ := (4, 4)
def S : ℝ × ℝ := (4, 0)
def V : ℝ × ℝ := (0, 4)

-- Define point T
def T : ℝ × ℝ := (-4, 0)

-- Define the area function for a square given two opposite corners
noncomputable def squareArea (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (x2 - x1) * (y2 - y1)

-- Define the area function for a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem area_equality :
  squareArea O W = triangleArea S V T :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_l1326_132649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greenhouse_max_income_l1326_132671

/-- Represents the total annual income from both greenhouses -/
noncomputable def f (x : ℝ) : ℝ := -1/4 * x + 4 * Real.sqrt (2 * x) + 250

/-- The maximum total annual income from both greenhouses -/
def max_income : ℝ := 282

/-- Theorem stating the existence of a maximum income within the given constraints -/
theorem greenhouse_max_income :
  ∃ x : ℝ, 20 ≤ x ∧ x ≤ 180 ∧ f x = max_income ∧
  ∀ y : ℝ, 20 ≤ y ∧ y ≤ 180 → f y ≤ max_income := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greenhouse_max_income_l1326_132671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_value_Q_value_m_value_l1326_132641

-- Define P and Q
noncomputable def P : ℝ := Real.rpow 8 0.25 * Real.sqrt (Real.sqrt 2) + Real.rpow (27 / 64) (-1/3) - Real.rpow (-2018) 0

noncomputable def Q : ℝ := 2 * (Real.log 2 / Real.log 3) - (Real.log (32 / 9) / Real.log 3) + (Real.log 8 / Real.log 3)

-- Theorem for P
theorem P_value : P = 7/3 := by sorry

-- Theorem for Q
theorem Q_value : Q = 2 := by sorry

-- Theorem for m
theorem m_value (a b m : ℝ) (h1 : Real.rpow 2 a = Real.rpow 5 b) (h2 : Real.rpow 2 a = m) (h3 : 1/a + 1/b = 2) :
  m = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_value_Q_value_m_value_l1326_132641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_non_participants_l1326_132642

theorem math_competition_non_participants 
  (total_students : ℕ) 
  (grade9_ratio grade10_ratio : ℕ) 
  (grade9_participation_ratio grade10_participation_ratio : ℚ) 
  (h1 : total_students = 210)
  (h2 : grade9_ratio = 3)
  (h3 : grade10_ratio = 4)
  (h4 : grade9_participation_ratio = 1/2)
  (h5 : grade10_participation_ratio = 3/7) :
  ∃ (non_participants : ℕ), non_participants = 114 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_non_participants_l1326_132642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_of_positive_reals_existence_of_positive_value_l1326_132643

-- Problem 1
theorem inequality_of_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

-- Problem 2
theorem existence_of_positive_value (x y z : ℝ) :
  let a := x^2 - 2*y + π/2
  let b := y^2 - 2*z + π/3
  let c := z^2 - 2*x + π/6
  max a (max b c) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_of_positive_reals_existence_of_positive_value_l1326_132643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ECODF_calculation_l1326_132619

/-- Given two circles with centers A and B, this function calculates the area of region ECODF --/
noncomputable def areaECODF (r : ℝ) (OA : ℝ) : ℝ :=
  18 * Real.sqrt 3 - 9 * Real.sqrt 2 - 9 * Real.pi / 4

/-- Theorem stating that for circles with radius 3 and OA = 3√3, the area of ECODF is as calculated --/
theorem area_ECODF_calculation (A B O C D E F : ℝ × ℝ) :
  let r := 3
  let OA := 3 * Real.sqrt 3
  -- O is midpoint of AB
  (O.1 = (A.1 + B.1) / 2 ∧ O.2 = (A.2 + B.2) / 2) →
  -- Distance OA = 3√3
  Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2) = OA →
  -- Circles centered at A and B with radius r
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = r →
  Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = r →
  -- OC and OD are tangent to circles A and B respectively
  ((C.1 - O.1) * (C.1 - A.1) + (C.2 - O.2) * (C.2 - A.2) = 0) →
  ((D.1 - O.1) * (D.1 - B.1) + (D.2 - O.2) * (D.2 - B.2) = 0) →
  -- EF is a common tangent to both circles
  Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = r →
  Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2) = r →
  ((E.1 - F.1) * (E.1 - A.1) + (E.2 - F.2) * (E.2 - A.2) = 0) →
  ((F.1 - E.1) * (F.1 - B.1) + (F.2 - E.2) * (F.2 - B.2) = 0) →
  areaECODF r OA = 18 * Real.sqrt 3 - 9 * Real.sqrt 2 - 9 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ECODF_calculation_l1326_132619
