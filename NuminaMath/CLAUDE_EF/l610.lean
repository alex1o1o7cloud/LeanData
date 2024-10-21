import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l610_61091

noncomputable section

open Real

theorem triangle_tangent_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  a * cos B - b * cos A = 3/5 * c →
  tan A / tan B = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l610_61091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_in_region_l610_61068

def S : Set ℝ := {1, 2}

theorem three_points_in_region :
  ∃! n : ℕ, n = (Finset.filter (λ p : ℝ × ℝ ↦ p.1 + p.2 - 3 ≥ 0) 
    (Finset.product {1, 2} {1, 2})).card ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_in_region_l610_61068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_white_not_yellow_l610_61051

/-- The probability of drawing a red ball from the bag. -/
noncomputable def prob_red : ℝ := 1/2

/-- The probability of drawing a white ball from the bag. -/
noncomputable def prob_white : ℝ := 1/3

/-- The probability of drawing a yellow ball from the bag. -/
noncomputable def prob_yellow : ℝ := 1/6

/-- The number of times a ball is drawn from the bag. -/
def num_draws : ℕ := 3

/-- The probability of noting down the colors red and white but not yellow in 3 draws with replacement. -/
theorem prob_red_white_not_yellow : 
  (Nat.choose num_draws 2 * prob_red^2 * prob_white + 
   Nat.choose num_draws 1 * prob_red * prob_white^2) = 5/12 := by
  sorry

#check prob_red_white_not_yellow

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_white_not_yellow_l610_61051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_symmetry_l610_61055

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

noncomputable def shifted_f (φ : ℝ) (x : ℝ) : ℝ := f φ (x + Real.pi/6)

def symmetric_about_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

theorem sin_shift_symmetry (φ : ℝ) :
  symmetric_about_y_axis (shifted_f φ) → φ = Real.pi/6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_symmetry_l610_61055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_result_l610_61097

noncomputable def cube (x : ℝ) : ℝ := x^3

noncomputable def square (x : ℝ) : ℝ := x^2

noncomputable def reciprocate (x : ℝ) : ℝ := x⁻¹

noncomputable def operation_sequence (x : ℝ) : ℝ := reciprocate (square (cube x))

noncomputable def iterate_sequence (x : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => operation_sequence (iterate_sequence x n)

theorem sequence_result (x : ℝ) (n : ℕ) (h : x ≠ 0) :
  iterate_sequence x n = x ^ ((-6 : ℤ) ^ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_result_l610_61097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l610_61063

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3) + 2

theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, T' < T → ∃ x, f (x + T') ≠ f x)) ∧
  -- Maximum value is 3
  (∀ x, f x ≤ 3) ∧ (∃ x, f x = 3) ∧
  -- Strictly increasing on [-π/12 + kπ, 5π/12 + kπ] for k ∈ ℤ
  (∀ k : ℤ, ∀ x y, -Real.pi/12 + k*Real.pi ≤ x ∧ x < y ∧ y ≤ 5*Real.pi/12 + k*Real.pi → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l610_61063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_comparison_l610_61088

theorem parabola_comparison : ∀ (x : ℝ), 
  (1/2 : ℝ) > (-1/2 : ℝ) := by
  intro x
  norm_num

#check parabola_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_comparison_l610_61088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l610_61013

/-- Calculates the length of a train given its speed and time to cross a pole -/
noncomputable def train_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Theorem: A train with speed 300 km/hr crossing a pole in 33 seconds has a length of 2750 meters -/
theorem train_length_calculation :
  train_length 300 33 = 2750 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l610_61013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l610_61057

/-- A hyperbola with semi-major axis a, semi-minor axis b, and semi-focal length c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  focal_length : c^2 = a^2 + b^2

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- The distance from a focus to an asymptote of the hyperbola -/
noncomputable def focus_asymptote_distance (h : Hyperbola) : ℝ := 
  (h.b * h.c) / Real.sqrt (h.a^2 + h.b^2)

theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_dist : focus_asymptote_distance h = Real.sqrt 2 / 3 * h.c) : 
  eccentricity h = 3 * Real.sqrt 7 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l610_61057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_pi_l610_61099

-- Define the functions
noncomputable def f1 (x : ℝ) := Real.sin (abs x)
noncomputable def f2 (x : ℝ) := Real.cos (2 * x - Real.pi / 3)
noncomputable def f3 (x : ℝ) := abs (Real.cos x + 1 / 2)
noncomputable def f4 (x : ℝ) := Real.tan (x + Real.pi / 4)

-- Define the concept of minimum positive period
def has_min_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ (∀ x, f (x + p) = f x) ∧ (∀ q, 0 < q ∧ q < p → ∃ x, f (x + q) ≠ f x)

-- State the theorem
theorem min_period_pi :
  ¬(has_min_period f1 Real.pi) ∧
  (has_min_period f2 Real.pi) ∧
  ¬(has_min_period f3 Real.pi) ∧
  (has_min_period f4 Real.pi) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_pi_l610_61099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_implies_x_l610_61060

noncomputable def v (x : ℝ) : Fin 2 → ℝ := ![x, 4]
noncomputable def w : Fin 2 → ℝ := ![5, 2]

noncomputable def proj (u v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  let dot_product := (u 0) * (v 0) + (u 1) * (v 1)
  let scalar := dot_product / ((v 0)^2 + (v 1)^2)
  fun i => scalar * (v i)

theorem projection_implies_x (x : ℝ) :
  proj w (v x) = ![3, 1.2] → x = 47/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_implies_x_l610_61060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_drink_calories_l610_61054

/-- Represents the fruit drink composition and calorie content --/
structure FruitDrink where
  lemon_juice : ℚ
  sugar : ℚ
  water : ℚ
  orange_juice : ℚ
  orange_juice_calories : ℚ
  lemon_juice_calories : ℚ
  sugar_calories : ℚ

/-- Calculates the total calories in a given weight of the fruit drink --/
def calories_in_weight (drink : FruitDrink) (weight : ℚ) : ℚ :=
  let total_weight := drink.lemon_juice + drink.sugar + drink.water + drink.orange_juice
  let total_calories := 
    drink.lemon_juice * drink.lemon_juice_calories / 100 +
    drink.sugar * drink.sugar_calories / 100 +
    drink.orange_juice * drink.orange_juice_calories / 100
  total_calories * weight / total_weight

/-- Theorem stating that 250 grams of the specified fruit drink contains 225 calories --/
theorem fruit_drink_calories : 
  let drink : FruitDrink := {
    lemon_juice := 200,
    sugar := 150,
    water := 300,
    orange_juice := 100,
    orange_juice_calories := 45,
    lemon_juice_calories := 25,
    sugar_calories := 386
  }
  calories_in_weight drink 250 = 225 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_drink_calories_l610_61054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_solution_l610_61053

-- Define the power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x^α

-- State the theorem
theorem power_function_solution :
  ∃ α : ℝ,
  (power_function α 2 = 8) ∧
  ∃ x : ℝ, (power_function α x = 64 ∧ x = 4) :=
by
  -- We'll use α = 3 as our solution
  use 3
  constructor
  · -- Prove that power_function 3 2 = 8
    simp [power_function]
    norm_num
  · -- Prove that there exists an x such that power_function 3 x = 64 and x = 4
    use 4
    constructor
    · simp [power_function]
      norm_num
    · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_solution_l610_61053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_implies_m_eq_neg_two_l610_61094

/-- A function f is an inverse proportion function if it can be written as f(x) = k/x for some non-zero constant k -/
def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x : ℝ), x ≠ 0 → f x = k / x

/-- The function we're considering -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * (x ^ (m^2 - 5))

theorem inverse_proportion_implies_m_eq_neg_two :
  (∃ m : ℝ, is_inverse_proportion (f m)) → (∃ m : ℝ, m = -2 ∧ is_inverse_proportion (f m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_implies_m_eq_neg_two_l610_61094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l610_61015

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sin (x / 2) - 1 / 2) + Real.sqrt (1 / (x - 2))

def domain (x : ℝ) : Prop :=
  2 < x ∧ x < 5 * Real.pi / 3 ∧
  ∃ k : ℕ, Real.pi / 3 + 4 * k * Real.pi < x ∧ x < 5 * Real.pi / 3 + 4 * k * Real.pi

theorem f_domain :
  ∀ x : ℝ, f x ∈ Set.univ ↔ domain x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l610_61015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_900_l610_61029

/-- Calculates the length of a train given its speed, the speed of a person walking in the same direction, and the time it takes for the train to cross the person. -/
noncomputable def train_length (train_speed : ℝ) (person_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  let relative_speed := train_speed - person_speed
  let relative_speed_ms := relative_speed * 1000 / 3600
  relative_speed_ms * crossing_time

/-- Theorem stating that under the given conditions, the train length is approximately 900 meters. -/
theorem train_length_approx_900 :
  let train_speed := 63
  let person_speed := 3
  let crossing_time := 53.99568034557235
  ∃ ε > 0, abs (train_length train_speed person_speed crossing_time - 900) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_900_l610_61029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_form_l610_61044

/-- The distance between the two intersections of x = y^6 and x + y^3 = 1 -/
noncomputable def intersection_distance : ℝ :=
  2 * Real.rpow ((Real.sqrt 5 - 1) / 2) (1/3)

/-- The curves x = y^6 and x + y^3 = 1 intersect at two points -/
axiom two_intersections : ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ 
  y₁^6 + y₁^3 = 1 ∧ y₂^6 + y₂^3 = 1

/-- The distance between the intersections can be expressed in the form sqrt(u + v * sqrt(w)) -/
theorem distance_form : 
  ∃ (u v w : ℝ), intersection_distance = Real.sqrt (u + v * Real.sqrt w) ∧ 
  u = 0 ∧ v = 2 ∧ w = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_form_l610_61044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_implies_k_value_l610_61042

theorem expansion_coefficient_implies_k_value (k : ℕ+) :
  (Finset.range 7).sum (λ i ↦ (Nat.choose 6 i : ℝ) * k^i * (if i = 4 then 1 else 0)) < 120 →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_implies_k_value_l610_61042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l610_61083

/-- Geometric sequence sum -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Theorem: Common ratio of geometric sequence when sums form arithmetic sequence -/
theorem geometric_sequence_common_ratio
  (a₁ : ℝ) (q : ℝ) (n : ℕ) (h : q ≠ 1) :
  let S := geometric_sum a₁ q
  2 * S n = S (n + 1) + S (n + 2) →
  q = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l610_61083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PO_in_triangular_pyramid_l610_61037

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  P : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Theorem: Length of PO in the given triangular pyramid -/
theorem length_PO_in_triangular_pyramid 
  (pyramid : TriangularPyramid)
  (h_equilateral : distance pyramid.A pyramid.B = distance pyramid.B pyramid.C ∧ 
                   distance pyramid.B pyramid.C = distance pyramid.C pyramid.A)
  (h_side_length : distance pyramid.A pyramid.B = 4 * Real.sqrt 3)
  (h_PA : distance pyramid.P pyramid.A = 3)
  (h_PB : distance pyramid.P pyramid.B = 4)
  (h_PC : distance pyramid.P pyramid.C = 5)
  (O : Point3D)
  (h_O_center : O.x = (pyramid.A.x + pyramid.B.x + pyramid.C.x) / 3 ∧
                O.y = (pyramid.A.y + pyramid.B.y + pyramid.C.y) / 3 ∧
                O.z = (pyramid.A.z + pyramid.B.z + pyramid.C.z) / 3) :
  distance pyramid.P O = Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PO_in_triangular_pyramid_l610_61037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_negative_integers_less_than_pi_l610_61086

theorem non_negative_integers_less_than_pi :
  {x : ℕ | (x : ℝ) < Real.pi} = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_negative_integers_less_than_pi_l610_61086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_property_l610_61022

-- Define the inverse proportion function
noncomputable def f (x : ℝ) : ℝ := 6 / x

-- Theorem statement
theorem inverse_proportion_property (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f x₂ = y₂) 
  (h3 : x₁ < 0) 
  (h4 : 0 < x₂) : 
  y₁ < y₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_property_l610_61022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_for_perry_l610_61032

/-- Represents the income tax system in country X -/
structure TaxSystem where
  baseRate : ℚ  -- The tax rate for incomes up to $5000
  baseCap : ℚ   -- The income threshold ($5000)
  excessRate : ℚ -- The tax rate for income exceeding $5000

/-- Calculates the tax for a given income under the tax system -/
def calculateTax (sys : TaxSystem) (income : ℚ) : ℚ :=
  if income ≤ sys.baseCap then
    sys.baseRate * income
  else
    sys.baseRate * sys.baseCap + sys.excessRate * (income - sys.baseCap)

/-- The theorem stating the correct tax rate for the given conditions -/
theorem tax_rate_for_perry (sys : TaxSystem) (perryIncome perryTax : ℚ) :
  sys.baseCap = 5000 ∧
  sys.excessRate = 1/10 ∧
  perryIncome = 10550 ∧
  perryTax = 950 ∧
  calculateTax sys perryIncome = perryTax →
  sys.baseRate = 79/1000 := by
  sorry

#check tax_rate_for_perry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_for_perry_l610_61032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_last_number_l610_61087

def sequence_condition (a : ℕ → ℕ) : Prop :=
  ∀ k, k ≥ 3 → k ≤ 2020 →
    (a (k-1) ∣ a k) ∧
    ((a (k-1) + a (k-2)) ∣ a k)

theorem min_last_number (a : ℕ → ℕ) :
  sequence_condition a →
  a 2020 ≥ Nat.factorial 2019 :=
by
  sorry

#eval Nat.factorial 2019

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_last_number_l610_61087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_properties_l610_61028

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_properties :
  (∀ x : ℝ, floor (x + 2) = floor x + 2) ∧
  (∀ x y : ℝ, floor (x + 1/2 + y + 1/2) = floor x + floor y + 1) ∧
  (∀ x y : ℝ, floor (0.5 * x * 0.5 * y) = (⌊(0.25 : ℝ) * (floor x * floor y)⌋)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_properties_l610_61028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_similar_triangle_after_cuts_l610_61081

-- Define a triangle type
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  sum_180 : angle1 + angle2 + angle3 = 180

-- Define the original triangle
def originalTriangle : Triangle := {
  angle1 := 20,
  angle2 := 20,
  angle3 := 140,
  sum_180 := by norm_num
}

-- Define a function to represent cutting a triangle along a bisector
def cutAlongBisector (t : Triangle) (bisectedAngle : Fin 3) : Triangle := sorry

-- Define a property of similarity to the original triangle
def isSimilarToOriginal (t : Triangle) : Prop :=
  (t.angle1 = 20 ∧ t.angle2 = 20 ∧ t.angle3 = 140) ∨
  (t.angle1 = 20 ∧ t.angle2 = 140 ∧ t.angle3 = 20) ∨
  (t.angle1 = 140 ∧ t.angle2 = 20 ∧ t.angle3 = 20)

-- Helper function to apply cuts
def applyCuts (t : Triangle) : List (Fin 3) → Triangle
  | [] => t
  | (cut :: cuts) => applyCuts (cutAlongBisector t cut) cuts

-- Theorem stating the impossibility of obtaining a similar triangle
theorem no_similar_triangle_after_cuts :
  ∀ (cuts : List (Fin 3)), ¬(isSimilarToOriginal (applyCuts originalTriangle cuts)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_similar_triangle_after_cuts_l610_61081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_at_calculated_point_l610_61004

/-- Represents a train with a given speed in km/h -/
structure Train where
  speed : ℝ

/-- Represents the scenario of two trains moving towards each other -/
structure TrainScenario where
  train_a : Train
  train_b : Train
  distance : ℝ

/-- Converts km/h to m/s -/
noncomputable def kmh_to_ms (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

/-- Calculates the meeting point of two trains -/
noncomputable def meeting_point (scenario : TrainScenario) : ℝ :=
  let speed_a_ms := kmh_to_ms scenario.train_a.speed
  let speed_b_ms := kmh_to_ms scenario.train_b.speed
  let relative_speed := speed_a_ms + speed_b_ms
  let time := (scenario.distance * 1000) / relative_speed
  speed_a_ms * time / 1000

/-- Theorem stating that the trains meet at the calculated point -/
theorem trains_meet_at_calculated_point (scenario : TrainScenario) 
  (h1 : scenario.train_a.speed = 162)
  (h2 : scenario.train_b.speed = 120)
  (h3 : scenario.distance = 450) :
  ∃ ε > 0, |meeting_point scenario - 258.5691| < ε :=
by
  sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_at_calculated_point_l610_61004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_distance_l610_61076

/-- Given a parabola x² = ay with a > 0 and directrix l₁: x - y - 3 = 0,
    prove that if the minimum sum of distances from a point on the parabola
    to l₁ and the focus is 2√2, then a = 4. -/
theorem parabola_directrix_distance (a : ℝ) : 
  a > 0 → 
  (∃ l₁ : ℝ → ℝ → Prop, l₁ = λ x y ↦ x - y - 3 = 0) →
  (∃ parabola : ℝ → ℝ → Prop, parabola = λ x y ↦ x^2 = a*y) →
  (∃ min_distance : ℝ, min_distance = 2 * Real.sqrt 2) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_distance_l610_61076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_if_perpendicular_to_same_line_planes_parallel_from_skew_lines_l610_61059

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the basic relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (skew : Line → Line → Prop)

-- Theorem 1
theorem planes_parallel_if_perpendicular_to_same_line 
  (n : Line) (α β : Plane) :
  perpendicular n α → perpendicular n β → parallel α β :=
by sorry

-- Theorem 2
theorem planes_parallel_from_skew_lines 
  (m n : Line) (α β : Plane) :
  skew m n → 
  contains α n → 
  parallel_line_plane n β → 
  contains β m → 
  parallel_line_plane m α → 
  parallel α β :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_if_perpendicular_to_same_line_planes_parallel_from_skew_lines_l610_61059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_reflected_arcs_area_l610_61009

/-- The radius of a circle circumscribing a regular octagon with side length 1 -/
noncomputable def octagon_radius : ℝ := 1 / (2 * Real.sqrt ((Real.sqrt 2 - 1) / (2 * Real.sqrt 2)))

/-- The area of the region bounded by 8 reflected arcs in a regular octagon inscribed in a circle -/
noncomputable def reflected_arcs_area (r : ℝ) : ℝ := 4 * Real.sqrt 3 - Real.pi * r^2

theorem octagon_reflected_arcs_area :
  reflected_arcs_area octagon_radius = 4 * Real.sqrt 3 - Real.pi * octagon_radius^2 := by
  -- Expand the definition of reflected_arcs_area
  unfold reflected_arcs_area
  -- The equality holds by definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_reflected_arcs_area_l610_61009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l610_61079

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_odd : ∀ x : ℝ, f x + f (-x) = 0
axiom f_decreasing : ∀ x m : ℝ, m > 0 → f (x - m) > f x

-- Define the solution set
def solution_set : Set ℝ := {x | x < -2 ∨ x > 1}

-- State the theorem
theorem inequality_solution : 
  {x : ℝ | f (-2 + x) + f (x^2) < 0} = solution_set := by
  sorry

#check inequality_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l610_61079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_on_unit_sphere_l610_61000

def f (x y z : ℝ) : ℝ := 3*x + 5*y - z

theorem max_on_unit_sphere :
  let max_value := Real.sqrt 35
  let max_point := (3/max_value, 5/max_value, -1/max_value)
  ∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 →
    f x y z ≤ max_value ∧
    f max_point.1 max_point.2.1 max_point.2.2 = max_value :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_on_unit_sphere_l610_61000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carls_dad_contribution_l610_61074

/-- The amount of money Carl's dad gave him to buy the coat -/
def dads_contribution (weekly_saving : ℕ) (weeks : ℕ) (bill_fraction : ℚ) (coat_price : ℕ) : ℤ :=
  (coat_price : ℤ) - ((weekly_saving * weeks : ℕ) - (((weekly_saving * weeks : ℕ) : ℚ) * bill_fraction).floor)

/-- Theorem stating the amount Carl's dad gave him -/
theorem carls_dad_contribution :
  dads_contribution 25 6 (1/3) 170 = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carls_dad_contribution_l610_61074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_omega_l610_61050

/-- The smallest positive value of ω that satisfies the cosine equation -/
theorem smallest_omega : ∃ ω : ℝ, ω > 0 ∧ 
  (∀ x : ℝ, Real.cos (ω * x + π / 4) = Real.cos (ω * (x - 2 * π / 3) + π / 4)) ∧ 
  (∀ ω' : ℝ, ω' > 0 → 
    (∀ x : ℝ, Real.cos (ω' * x + π / 4) = Real.cos (ω' * (x - 2 * π / 3) + π / 4)) → 
    ω ≤ ω') ∧
  ω = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_omega_l610_61050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_drawers_five_balls_two_drawers_l610_61096

-- Define a function for the number of ways to distribute n balls into k drawers
def number_of_ways_to_distribute (n k : ℕ) : ℕ := k^n

theorem balls_in_drawers (n k : ℕ) : 
  number_of_ways_to_distribute n k = k^n :=
by
  -- The proof is trivial due to the definition
  rfl

-- Define the specific problem
def number_of_balls : ℕ := 5
def number_of_drawers : ℕ := 2

-- State the theorem for the specific problem
theorem five_balls_two_drawers : 
  number_of_ways_to_distribute number_of_balls number_of_drawers = 2^5 :=
by
  -- Unfold the definitions and apply the general theorem
  unfold number_of_balls number_of_drawers
  exact balls_in_drawers 5 2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_drawers_five_balls_two_drawers_l610_61096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_Q_value_l610_61045

-- Define the probability function Q
def Q (b : ℝ) : ℝ :=
  -- We replace Probability.measure with a placeholder function
  -- as it's not directly available in the current Mathlib version
  sorry

-- Theorem statement
theorem max_Q_value :
  ∃ (b : ℝ), 0 ≤ b ∧ b ≤ 1 ∧ ∀ (c : ℝ), 0 ≤ c ∧ c ≤ 1 → Q c ≤ Q b ∧ Q b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_Q_value_l610_61045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_shortest_side_l610_61007

theorem similar_triangle_shortest_side
  (leg1 : ℝ)
  (hyp1 : ℝ)
  (hyp2 : ℝ)
  (h_right_triangle : leg1^2 + (Real.sqrt (hyp1^2 - leg1^2))^2 = hyp1^2)
  (h_leg1 : leg1 = 15)
  (h_hyp1 : hyp1 = 39)
  (h_hyp2 : hyp2 = 117) :
  leg1 * (hyp2 / hyp1) = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_shortest_side_l610_61007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_eq_one_seventh_l610_61092

/-- A sequence defined by b₁ = 5, b₂ = 7, and bₙ = bₙ₋₁ / bₙ₋₂ for n ≥ 3 -/
def b : ℕ → ℚ
  | 0 => 5  -- We define b(0) to be 5 to handle the base case
  | 1 => 7
  | n+2 => b (n+1) / b n

/-- The 2023rd term of the sequence is 1/7 -/
theorem b_2023_eq_one_seventh : b 2022 = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_eq_one_seventh_l610_61092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l610_61033

/-- The standard equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

/-- The standard equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := 4 * x^2 - 4 * y^2 / 3 = 1

/-- The focal length of an ellipse -/
noncomputable def ellipse_focal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

/-- The focal length of a hyperbola -/
noncomputable def hyperbola_focal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

/-- The theorem stating that the hyperbola has the same foci as the ellipse and passes through the given point -/
theorem hyperbola_properties :
  (∀ x y, ellipse_equation x y → hyperbola_focal_length (1/2) (Real.sqrt 3/2) = ellipse_focal_length 3 (Real.sqrt 8)) ∧
  hyperbola_equation 2 (3/2 * Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l610_61033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l610_61019

/-- Represents the area of a triangle -/
def TriangleArea : Type := ℝ

/-- Represents a trapezoid divided by its diagonals -/
structure Trapezoid where
  /-- Area of the triangle formed by one base and the diagonals -/
  A : TriangleArea
  /-- Area of the triangle formed by the other base and the diagonals -/
  B : TriangleArea

/-- The area of the trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  (Real.sqrt t.A + Real.sqrt t.B) ^ 2

/-- Theorem stating that the area of the trapezoid is (√A + √B)² -/
theorem trapezoid_area_formula (t : Trapezoid) :
  trapezoidArea t = (Real.sqrt t.A + Real.sqrt t.B) ^ 2 := by
  -- The proof is trivial as it follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l610_61019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_K1_K2_equal_l610_61077

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def two_circles_problem (c1 c2 : Circle) (d : ℝ) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  -- Circles are non-intersecting
  d > c1.radius + c2.radius ∧
  -- Distance between centers is d
  ((x2 - x1)^2 + (y2 - y1)^2 = d^2)

-- Define the construction of K₁ and K₂
noncomputable def construct_K (c1 c2 : Circle) (d : ℝ) : ℝ :=
  2 * c1.radius * c2.radius / d

-- Theorem statement
theorem circles_K1_K2_equal (c1 c2 : Circle) (d : ℝ) :
  two_circles_problem c1 c2 d →
  construct_K c1 c2 d = construct_K c2 c1 d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_K1_K2_equal_l610_61077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_a_value_l610_61012

/-- A power function that never passes through the origin -/
noncomputable def powerFunction (a : ℝ) (x : ℝ) : ℝ := (a^2 - 9*a + 19) * x^(2*a - 9)

/-- The property that the function never passes through the origin -/
def neverPassesThroughOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f x ≠ 0

/-- Theorem stating that there exists a value of a for which the power function
    never passes through the origin and a = 3 -/
theorem power_function_a_value :
  ∃ a : ℝ, neverPassesThroughOrigin (powerFunction a) ∧ a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_a_value_l610_61012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_x_coord_l610_61017

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - log x

theorem tangent_intersection_x_coord (a : ℝ) :
  ∃ (t : ℝ), t > 0 ∧
  (f a t / t = 2*t + a - 1/t) ∧
  (∀ x > 0, f a t / t * x = f a x → x = t) ∧
  t = 1 := by
  sorry

#check tangent_intersection_x_coord

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_x_coord_l610_61017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_fixed_points_l610_61071

def P₁ (x : ℝ) : ℝ := x^2 - 2

def P : ℕ → ℝ → ℝ
| 0 => P₁
| n + 1 => λ x => P₁ (P n x)

theorem polynomial_fixed_points (n : ℕ) (h : n > 0) :
  ∀ x : ℝ, P n x = x ↔ 
    (∃ m : ℕ, m < 2^(n-1) ∧ x = 2 * Real.cos (2 * Real.pi * (m : ℝ) / (2^n - 1))) ∨
    (∃ k : ℕ, k < 2^(n-1) ∧ x = 2 * Real.cos (2 * Real.pi * (k : ℝ) / (2^n + 1))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_fixed_points_l610_61071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_looms_profit_decrease_matches_l610_61089

/-- Represents the number of looms employed by the textile manufacturing firm -/
def L : ℕ := 70

/-- The aggregate sales value of the output of the looms in rupees -/
def aggregate_sales : ℕ := 500000

/-- The monthly manufacturing expenses in rupees -/
def manufacturing_expenses : ℕ := 150000

/-- The monthly establishment charges in rupees -/
def establishment_charges : ℕ := 75000

/-- The decrease in profit when one loom breaks down for a month, in rupees -/
def profit_decrease : ℕ := 5000

/-- Theorem stating that the number of looms employed by the firm is 70 -/
theorem number_of_looms : L = 70 := by
  rfl

/-- Helper function to calculate profit per loom -/
def profit_per_loom : ℚ :=
  (aggregate_sales - manufacturing_expenses : ℚ) / L

/-- Proof that the profit decrease matches the profit per loom -/
theorem profit_decrease_matches : profit_per_loom = profit_decrease := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_looms_profit_decrease_matches_l610_61089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inclination_range_l610_61010

-- Define the function f(x) = (1/3)x³ - x²
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ := x^2 - 2*x

-- Theorem statement
theorem tangent_inclination_range :
  ∀ x : ℝ, ∃ α : ℝ,
    (0 ≤ α ∧ α < Real.pi / 2) ∨ (3 * Real.pi / 4 ≤ α ∧ α < Real.pi) ∧
    Real.tan α = f_derivative x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inclination_range_l610_61010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_expression_l610_61072

theorem quadratic_roots_expression (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 2020 = 0 → 
  x₂^2 - 4*x₂ - 2020 = 0 → 
  x₁^2 - 2*x₁ + 2*x₂ = 2028 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_expression_l610_61072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_point_Q_trajectory_minimum_distance_RQ_l610_61031

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define the line l
def l : Set (ℝ × ℝ) := {p | 3 * p.1 - 4 * p.2 + 5 = 0 ∨ p.1 = 1}

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the trajectory of Q
def Q_trajectory : Set (ℝ × ℝ) := {p | p.1^2 / 4 + p.2^2 = 4 ∧ p.1 ≠ 0}

-- Define point R
def R : ℝ × ℝ := (1, 0)

theorem circle_line_intersection :
  P ∈ l ∧ ∃ A B, A ∈ C ∩ l ∧ B ∈ C ∩ l ∧ distance A B = 2 * Real.sqrt 3 := by sorry

theorem point_Q_trajectory :
  ∀ M ∈ C, ∃ Q ∈ Q_trajectory, Q.1 = 2 * M.1 ∧ Q.2 = M.2 := by sorry

theorem minimum_distance_RQ :
  ∃ Q ∈ Q_trajectory, ∀ Q' ∈ Q_trajectory, distance R Q ≤ distance R Q' ∧
  distance R Q = Real.sqrt 33 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_point_Q_trajectory_minimum_distance_RQ_l610_61031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_for_odd_function_l610_61014

theorem angle_range_for_odd_function 
  (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_increasing : ∀ x y, 0 < x → x < y → f x < f y)
  (h_half : f (1/2) = 0)
  (A : ℝ)
  (h_triangle : 0 < A ∧ A < π)
  (h_cos : f (Real.cos A) < 0) :
  (π/3 < A ∧ A < π/2) ∨ (2*π/3 < A ∧ A < π) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_for_odd_function_l610_61014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l610_61006

-- Define the function f
noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x + φ) + Real.cos (2 * x + φ)

-- State the theorem
theorem f_properties (φ : ℝ) (h1 : |φ| < π/2) 
  (h2 : ∀ x, f x φ = f (-x) φ) : -- Symmetry about x = 0
  (∃ T > 0, ∀ x, f (x + T) φ = f x φ ∧ 
   ∀ S, (0 < S ∧ S < T) → ∃ x, f (x + S) φ ≠ f x φ) ∧ -- Minimum positive period is π
  (∀ x, x ∈ Set.Ioo 0 (π/2) → f (x + π) φ = f x φ) ∧ -- Period is π
  (∀ x y, x ∈ Set.Ioo 0 (π/2) → y ∈ Set.Ioo 0 (π/2) → x < y → f y φ < f x φ) -- Decreasing on (0, π/2)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l610_61006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_root_of_unity_l610_61073

theorem complex_root_of_unity (α : ℂ) (n : ℕ)
  (hn : n > 0)
  (h1 : Complex.abs α = 1)
  (h2 : Complex.abs (α + 1) = 1)
  (h3 : (1 + α)^n = 1) :
  α^n = 1 ∧ ∃ k : ℕ, n = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_root_of_unity_l610_61073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l610_61030

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem conditions
def problem_conditions (t : Triangle) : Prop :=
  let m : ℝ × ℝ := (t.a / 2, t.c / 2)
  let n : ℝ × ℝ := (Real.cos t.C, Real.cos t.A)
  0 < t.B ∧ t.B < Real.pi ∧  -- Angle B is between 0 and π
  t.A + t.B + t.C = Real.pi ∧  -- Sum of angles in a triangle
  (m.1 * n.1 + m.2 * n.2 = t.b * Real.cos t.B) ∧  -- Dot product condition
  (Real.cos ((t.A - t.C) / 2) = Real.sqrt 3 * Real.sin t.A) ∧
  (t.a^2 / 4 + t.c^2 / 4 = 5)  -- |m| = √5 condition

-- State the theorem
theorem triangle_problem (t : Triangle) (h : problem_conditions t) :
  t.B = Real.pi / 3 ∧ (1 / 2 * t.a * t.b = 2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l610_61030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_distance_l610_61067

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Projects a 3D point onto the xOy plane -/
def project_to_xOy (p : Point3D) : Point2D :=
  { x := p.x, y := p.y }

/-- Calculates the distance between two 2D points -/
noncomputable def distance_2D (p q : Point2D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

theorem projection_distance :
  let P : Point3D := { x := 1, y := 2, z := 3 }
  let Q : Point3D := { x := -3, y := 5, z := 2 }
  let P' := project_to_xOy P
  let Q' := project_to_xOy Q
  distance_2D P' Q' = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_distance_l610_61067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l610_61046

-- Define the original function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω*x) * Real.cos (ω*x) + (Real.sqrt 3 * Real.cos (2*ω*x)) / 2

-- State the theorem
theorem function_transformation (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x, f ω (x + 2) = f ω x) :
  ∃ g : ℝ → ℝ, ∀ x, g x = Real.sin (π*x/4 + π/4) ∧
    ∀ y, g y = f ω (y/2 - 1/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l610_61046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l610_61093

-- Define the expression
noncomputable def expression (x y : ℝ) : ℝ := (x / Real.sqrt y - y / Real.sqrt x) ^ 6

-- Theorem statement
theorem coefficient_of_x_cubed (x y : ℝ) (h1 : x ≠ 0) (h2 : y > 0) :
  ∃ c, c = 15 ∧ 
  ∃ f : ℝ → ℝ, (λ t => expression (t * x) y) = (λ t => c * t^3 * x^3 + f t) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l610_61093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_tan_angle_l610_61061

noncomputable section

-- Define the ellipse M
def ellipse_M (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the focus F and point D
def F : ℝ × ℝ := (1, 0)
def D : ℝ × ℝ := (1, 3/2)

-- Define the line x = 4
def line_x_4 (x : ℝ) : Prop := x = 4

-- Define the property of slopes forming an arithmetic sequence
def slopes_arithmetic_sequence (k_DA k_DC k_DB : ℝ) : Prop :=
  k_DA + k_DB = 2 * k_DC

-- Define angle function (placeholder)
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem ellipse_and_max_tan_angle :
  -- Part 1: Prove the equation of ellipse M
  (∀ x y : ℝ, ellipse_M x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  -- Part 2: Prove the maximum value of tan ∠DCF
  (∃ C : ℝ × ℝ, 
    line_x_4 C.1 ∧
    (∀ A B : ℝ × ℝ,
      ellipse_M A.1 A.2 → ellipse_M B.1 B.2 →
      (∃ k_DA k_DC k_DB : ℝ, slopes_arithmetic_sequence k_DA k_DC k_DB) →
      Real.tan (angle D C F) ≤ 8/15)) ∧
  (∃ C : ℝ × ℝ, 
    line_x_4 C.1 ∧
    Real.tan (angle D C F) = 8/15) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_tan_angle_l610_61061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l610_61066

/-- Profit function for project A -/
noncomputable def profit_A (x : ℝ) : ℝ := (2/5) * x

/-- Profit function for project B -/
noncomputable def profit_B (x : ℝ) : ℝ := -(1/5) * x^2 + 2 * x

/-- Total investment available -/
def total_investment : ℝ := 32

theorem investment_problem :
  /- Part 1: Profit from 10 thousand yuan investment in A -/
  (profit_A 10 = 4) ∧
  /- Part 2: Equal profit investment -/
  (∃ m : ℝ, m > 0 ∧ profit_A m = profit_B m ∧ m = 3) ∧
  /- Part 3: Maximum profit and optimal investment -/
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ total_investment ∧
    (∀ s : ℝ, 0 ≤ s ∧ s ≤ total_investment →
      profit_A (total_investment - t) + profit_B t ≥
      profit_A (total_investment - s) + profit_B s) ∧
    profit_A (total_investment - t) + profit_B t = 16 ∧
    t = 5 ∧ total_investment - t = 27) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l610_61066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l610_61008

/-- The function f(x) = x(ln x - 2ax) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.log x - 2 * a * x)

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.log x + 1 - 4 * a * x

theorem extreme_points_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁ > 0 ∧ x₂ > 0 ∧
    f_derivative a x₁ = 0 ∧ 
    f_derivative a x₂ = 0 ∧ 
    (∀ x : ℝ, x > 0 → f_derivative a x = 0 → (x = x₁ ∨ x = x₂))) →
  0 < a ∧ a < 1/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l610_61008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l610_61052

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  m : ℝ
  c : ℝ
  f : ℝ → ℝ := λ x => -x^2 + a*x + b
  range_condition : ∀ x, f x ≤ 0
  solution_set : ∀ x, f x > c - 1 ↔ m - 4 < x ∧ x < m + 1

/-- The main theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties (qf : QuadraticFunction) :
  qf.b = -(1/4) * (2*qf.m - 3)^2 ∧ qf.c = -21/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l610_61052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_iff_m_eq_11_l610_61034

/-- Three points are collinear if they lie on the same straight line. -/
def areCollinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

/-- The theorem states that the given points are collinear if and only if m = 11. -/
theorem points_collinear_iff_m_eq_11 (m : ℝ) : 
  areCollinear (1, m - 1) (3, m + 5) (6, 2 * m + 3) ↔ m = 11 := by
  sorry

#check points_collinear_iff_m_eq_11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_iff_m_eq_11_l610_61034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l610_61048

/-- The parabola y² = 2px with fixed points A(a, b) and B(-a, 0) -/
structure Parabola where
  p : ℝ
  a : ℝ
  b : ℝ
  h_ab_nonzero : a * b ≠ 0
  h_b_sq_neq_2pa : b^2 ≠ 2 * p * a

/-- A point on the parabola -/
def ParabolaPoint (parab : Parabola) :=
  {y : ℝ // y^2 = 2 * parab.p * (y^2 / (2 * parab.p))}

/-- The intersection of a line through A and a point on the parabola with the parabola -/
noncomputable def intersectionA (parab : Parabola) (M : ParabolaPoint parab) : ℝ × ℝ := sorry

/-- The intersection of a line through B and a point on the parabola with the parabola -/
noncomputable def intersectionB (parab : Parabola) (M : ParabolaPoint parab) : ℝ × ℝ := sorry

/-- The fixed point through which M₁M₂ always passes -/
noncomputable def fixedPoint (parab : Parabola) : ℝ × ℝ := (parab.a, 2 * parab.p * parab.a / parab.b)

/-- Predicate to check if a point lies on a line through two other points -/
def LineThrough (P Q R : ℝ × ℝ) : Prop := sorry

theorem fixed_point_theorem (parab : Parabola) (M : ParabolaPoint parab) :
  let M₁ := intersectionA parab M
  let M₂ := intersectionB parab M
  LineThrough M₁ M₂ (fixedPoint parab) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l610_61048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_at_one_l610_61078

open Real

/-- The function f(x) defined on the interval [0, 1] -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + (a - 1) * x^2 - x + 2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * (a - 1) * x - 1

theorem min_at_one (a : ℝ) :
  (∀ x, x ∈ Set.Icc 0 1 → f a 1 ≤ f a x) ↔ a ≤ 3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_at_one_l610_61078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_age_sum_l610_61018

/-- Represents a person's age -/
def Age := ℕ

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Checks if a natural number is a perfect square -/
def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem birthday_age_sum (alice_age bob_age carl_age : ℕ) : 
  carl_age = 2 →
  alice_age = bob_age + 2 →
  (∃ n : ℕ, n ≥ 0 ∧ (bob_age + n) % (carl_age + n) = 0 ∧ 
   (∀ m : ℕ, m < n → (bob_age + m) % (carl_age + m) ≠ 0) ∧
   (∀ k : ℕ, k > n → k < n + 7 → (bob_age + k) % (carl_age + k) = 0)) →
  (∃ future_age : ℕ, 
    future_age > alice_age ∧ 
    is_square future_age ∧
    (∀ m : ℕ, alice_age < m ∧ m < future_age → ¬is_square m) ∧
    sum_of_digits future_age = 9) := by
  sorry

#eval sum_of_digits 36  -- This should output 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_age_sum_l610_61018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_difference_values_l610_61011

theorem lcm_gcd_difference_values (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (Nat.lcm m n - Nat.gcd m n = 103) →
  (m + n ∈ ({21, 105, 309} : Set ℕ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_difference_values_l610_61011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l610_61026

/-- The set of complex numbers z satisfying |z-i|+|z+i|=3 forms an ellipse -/
theorem trajectory_is_ellipse : 
  {z : ℂ | Complex.abs (z - Complex.I) + Complex.abs (z + Complex.I) = 3} 
  = {z : ℂ | ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (z.re / a) ^ 2 + (z.im / b) ^ 2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l610_61026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player2_winning_strategy_l610_61095

/-- Represents a player in the game -/
inductive Player
| One
| Two

/-- Represents a position on the chessboard -/
structure Position where
  x : Nat
  y : Nat

/-- Represents the game state -/
structure GameState where
  player1Pos : Position
  player2Pos : Position
  currentPlayer : Player

/-- Defines a valid move in the game -/
def isValidMove (start finish : Position) : Prop :=
  (start.x = finish.x ∧ (start.y + 1 = finish.y ∨ start.y = finish.y + 1)) ∨
  (start.y = finish.y ∧ (start.x + 1 = finish.x ∨ start.x = finish.x + 1))

/-- Defines the winning condition for a player -/
def isWinningMove (player : Player) (pos : Position) (m n : Nat) : Prop :=
  match player with
  | Player.One => pos.y = n
  | Player.Two => pos.y = 1

/-- Theorem stating that Player 2 has a winning strategy -/
theorem player2_winning_strategy (m n : Nat) (h1 : m = 998) (h2 : n = 1998) :
  ∃ (strategy : GameState → Position),
    ∀ (initialState : GameState),
      initialState.player1Pos = ⟨1, 1⟩ →
      initialState.player2Pos = ⟨m, n⟩ →
      initialState.currentPlayer = Player.One →
      ∃ (finalState : GameState),
        (finalState.currentPlayer = Player.Two ∧
         isWinningMove Player.Two finalState.player2Pos m n) ∨
        (finalState.player2Pos = finalState.player1Pos) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player2_winning_strategy_l610_61095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l610_61082

-- Define the complex number
noncomputable def z : ℂ := 2 / (1 - Complex.I)

-- Theorem statement
theorem z_in_first_quadrant : 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l610_61082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_zeros_l610_61021

-- Define the function f(x) = sin(log x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

-- Theorem statement
theorem infinitely_many_zeros :
  ∃ (S : Set ℝ), (∀ x ∈ S, 0 < x ∧ x < 1 ∧ f x = 0) ∧ Set.Infinite S :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_zeros_l610_61021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l610_61069

/-- Given a function f: ℝ → ℝ with a tangent line y = x + 3 at the point (1, f(1)),
    prove that f(1) + f'(1) = 5 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : ∀ x, f 1 + (deriv f 1) * (x - 1) = x + 3) : 
  f 1 + deriv f 1 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l610_61069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_cos_alpha_l610_61084

theorem sin_2alpha_plus_cos_alpha (α : Real) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi / 2) :
  Real.sin (2 * α) + Real.cos α = (4 + Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_cos_alpha_l610_61084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l610_61016

noncomputable def z (m : ℝ) : ℂ := Real.log (m^2 - 2*m - 2) + (m^2 + 3*m + 2)*Complex.I

theorem z_properties (m : ℝ) :
  (∃ y : ℝ, z m = y*Complex.I ∧ y ≠ 0 ↔ m = 3) ∧
  (∃ x : ℝ, z m = x ↔ m = -1 ∨ m = -2) ∧
  (Real.log (m^2 - 2*m - 2) < 0 ∧ m^2 + 3*m + 2 > 0 ↔ -1 < m ∧ m < 3) :=
by sorry

#check z_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l610_61016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_time_calculation_l610_61070

-- Define the given conditions
noncomputable def upstream_distance : ℝ := 15 -- km
noncomputable def upstream_time : ℝ := 5 -- hours
noncomputable def current_speed_ratio : ℝ := 1 / 4 -- ratio of current speed to boat speed in still water

-- Define the function to calculate downstream time
noncomputable def downstream_time (d : ℝ) (t : ℝ) (r : ℝ) : ℝ :=
  let upstream_speed := d / t
  let boat_speed := upstream_speed / (1 - r)
  let downstream_speed := boat_speed * (1 + r)
  d / downstream_speed

-- Theorem statement
theorem downstream_time_calculation :
  downstream_time upstream_distance upstream_time current_speed_ratio = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_time_calculation_l610_61070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequences_count_l610_61005

theorem geometric_sequences_count : 
  (Finset.range 40).sum (λ y => (Finset.range y).card) = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequences_count_l610_61005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l610_61035

/-- Definition of a hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The inclination angle of an asymptote of a hyperbola -/
noncomputable def asymptote_angle (h : Hyperbola) : ℝ := Real.arctan (h.b / h.a)

/-- The distance from a focus to an asymptote of a hyperbola -/
noncomputable def focus_asymptote_distance (h : Hyperbola) : ℝ :=
  h.b * Real.sqrt (h.a^2 + h.b^2) / (h.a^2 + h.b^2)

/-- Theorem: A hyperbola with given properties has the standard equation (x²/12) - (y²/4) = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
  (angle_condition : asymptote_angle h = π/6)
  (distance_condition : focus_asymptote_distance h = 2) :
  ∀ x y, standard_equation h x y ↔ x^2/12 - y^2/4 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l610_61035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_point_theorem_l610_61001

theorem inverse_point_theorem (f : ℝ → ℝ) (h : f 0 = 1) :
  ∃ g : ℝ → ℝ, Function.LeftInverse g (f ∘ (λ x ↦ x + 3)) ∧ g 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_point_theorem_l610_61001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l610_61024

noncomputable def f (x : ℝ) := 2 * Real.cos x * (Real.sin x - Real.cos x) + 1

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (π/8) (3*π/4) → f x ≤ Real.sqrt 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (π/8) (3*π/4) → f x ≥ -1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (π/8) (3*π/4) ∧ f x = Real.sqrt 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (π/8) (3*π/4) ∧ f x = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l610_61024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l610_61002

-- Define the concept of a quadratic radical
def QuadraticRadical (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x, f x = Real.sqrt (g x)

-- Define the concept of simplest quadratic radical
def SimplestQuadraticRadical (f : ℝ → ℝ) (S : Set (ℝ → ℝ)) : Prop :=
  QuadraticRadical f ∧ f ∈ S ∧ ∀ g ∈ S, QuadraticRadical g → ¬(∃ h, QuadraticRadical h ∧ (∀ x, h x = g x) ∧ h ≠ g)

-- Define the given expressions
noncomputable def expr1 : ℝ → ℝ := λ x ↦ Real.sqrt (x^2 + 1)
noncomputable def expr2 : ℝ → ℝ := λ _ ↦ Real.sqrt 0.2
noncomputable def expr3 : ℝ → ℝ := λ a ↦ Real.sqrt (8*a)
noncomputable def expr4 : ℝ → ℝ := λ _ ↦ Real.sqrt 9

-- Theorem statement
theorem simplest_quadratic_radical :
  SimplestQuadraticRadical expr1 {expr1, expr2, expr3, expr4} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l610_61002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_geq_one_l610_61041

/-- A random variable following a normal distribution with mean -1 and variance 36 -/
noncomputable def ξ : Prob → ℝ := sorry

/-- The probability density function of ξ -/
noncomputable def pdf_ξ : ℝ → ℝ := sorry

/-- The cumulative distribution function of ξ -/
noncomputable def cdf_ξ : ℝ → ℝ := sorry

/-- The mean of ξ is -1 -/
axiom mean_ξ : ∫ x, x * pdf_ξ x = -1

/-- The variance of ξ is 36 -/
axiom var_ξ : ∫ x, (x - (-1))^2 * pdf_ξ x = 36

/-- The probability that ξ is between -3 and -1 is 0.4 -/
axiom prob_between : cdf_ξ (-1) - cdf_ξ (-3) = 0.4

/-- The theorem to be proved -/
theorem prob_geq_one : 1 - cdf_ξ 1 = 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_geq_one_l610_61041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oppositely_collinear_vectors_l610_61043

/-- Two vectors are oppositely collinear if they point in opposite directions -/
def OppositelyCollinear (v w : ℝ × ℝ) : Prop :=
  ∃ l : ℝ, l < 0 ∧ v = l • w

theorem oppositely_collinear_vectors (a b : ℝ × ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hnc : ¬ Collinear ℝ ({0, a, b} : Set (ℝ × ℝ))) :
  (∃ k : ℝ, OppositelyCollinear (k • a + b) (a + k • b)) ↔ k = -1 := by
  sorry

#check oppositely_collinear_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oppositely_collinear_vectors_l610_61043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l610_61049

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (3 + t, Real.sqrt 5 + t)

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2 * Real.sqrt 5 * y = 0

-- Define point P
def point_P : ℝ × ℝ := (3, Real.sqrt 5)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, line_l t₁ = A ∧ line_l t₂ = B ∧
  circle_C A.1 A.2 ∧ circle_C B.1 B.2

-- State the theorem
theorem intersection_distance_sum (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
  Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) = 3 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l610_61049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_specific_terms_l610_61080

def mySequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => (mySequence n)^2 + 1

theorem gcd_of_specific_terms : Nat.gcd (mySequence 999) (mySequence 2004) = 677 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_specific_terms_l610_61080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avery_work_time_l610_61020

noncomputable def avery_rate : ℝ := 1/3
noncomputable def tom_rate : ℝ := 1/2
noncomputable def tom_remaining_time : ℝ := 20.000000000000007 / 60

theorem avery_work_time :
  ∃ t : ℝ, t > 0 ∧ t < 3 ∧
  (avery_rate + tom_rate) * t + tom_rate * tom_remaining_time = 1 ∧
  t = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avery_work_time_l610_61020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_prime_l610_61038

-- Define the conditions
def is_valid_prime (p : ℕ) : Prop :=
  ∃ (x : ℝ),
    Nat.Prime p ∧
    0 < x ∧ x < 1 ∧
    ∃ (k : ℕ),
      Real.sqrt (p : ℝ) - k = x ∧
      ∃ (N : ℕ),
        (1 : ℝ) / x = N + (Real.sqrt (p : ℝ) - 31) / 75 ∧
        31 ≤ Real.sqrt (p : ℝ) ∧ Real.sqrt (p : ℝ) < 106

-- State the theorem
theorem unique_valid_prime :
  ∃! (p : ℕ), is_valid_prime p ∧ p = 2011 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_prime_l610_61038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_reciprocal_operations_for_16_l610_61039

noncomputable def reciprocal (x : ℝ) : ℝ := 1 / x

noncomputable def repeated_reciprocal (x : ℝ) : ℕ → ℝ
  | 0 => x
  | n + 1 => reciprocal (repeated_reciprocal x n)

theorem min_reciprocal_operations_for_16 :
  ∃ (n : ℕ), n > 0 ∧ repeated_reciprocal 16 n = 16 ∧
  ∀ (m : ℕ), 0 < m ∧ m < n → repeated_reciprocal 16 m ≠ 16 := by
  -- Proof goes here
  sorry

#eval Nat.min 2 3  -- This line is added to ensure some computable code exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_reciprocal_operations_for_16_l610_61039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_less_than_5_l610_61075

def dice_outcomes : Finset (Fin 6 × Fin 6) := Finset.univ

def sum_less_than_5 (outcome : Fin 6 × Fin 6) : Bool :=
  (outcome.1.val + outcome.2.val < 5)

theorem probability_sum_less_than_5 :
  (Finset.filter (fun x => sum_less_than_5 x) dice_outcomes).card /
  dice_outcomes.card = 1 / 6 := by
  sorry

#eval (Finset.filter (fun x => sum_less_than_5 x) dice_outcomes).card
#eval dice_outcomes.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_less_than_5_l610_61075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larry_wins_game_prob_l610_61003

/-- The probability that Larry wins the game -/
noncomputable def larry_wins_prob (larry_prob : ℝ) (julius_prob : ℝ) : ℝ :=
  larry_prob / (1 - (1 - larry_prob) * (1 - julius_prob))

/-- Theorem stating that Larry's winning probability is 6/7 under given conditions -/
theorem larry_wins_game_prob :
  larry_wins_prob (2/3) (1/3) = 6/7 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larry_wins_game_prob_l610_61003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l610_61056

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 2*x + 3)

-- Define the domain of f(x)
def domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem f_properties :
  (∀ x y, x ∈ domain → y ∈ domain → -1 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y) ∧ 
  (∀ x, x ∈ domain → f x ≤ 2) ∧
  (f (-1) = 0 ∧ f 3 = 0) ∧
  (∀ x, x ∈ domain → f x ≥ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l610_61056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_perimeter_l610_61065

/-- Given a triangle ABC with altitude CD = 2√3 and ∠BAC = 60°, 
    prove that its area is 12 and perimeter is 8√3 + 6 -/
theorem triangle_area_perimeter (A B C D : ℝ × ℝ) :
  let CD := 2 * Real.sqrt 3
  let angle_BAC := 60
  let AC := 4 * Real.sqrt 3
  let BC := 6
  let AB := 4 * Real.sqrt 3
  let area := (1/2) * AC * CD
  let perimeter := AC + BC + AB
  CD = 2 * Real.sqrt 3 ∧ angle_BAC = 60 → 
  area = 12 ∧ perimeter = 8 * Real.sqrt 3 + 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_perimeter_l610_61065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_powers_2023_l610_61040

-- Define i as a complex number with i² = -1
noncomputable def i : ℂ := Complex.I

-- Define the sum of powers of i from 0 to n
noncomputable def sum_powers (n : ℕ) : ℂ :=
  (Finset.range (n + 1)).sum (fun k => i ^ k)

-- Statement of the theorem
theorem sum_powers_2023 : sum_powers 2023 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_powers_2023_l610_61040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_distance_correct_l610_61023

/-- The distance Lindy travels when Jack and Christina meet -/
noncomputable def lindyDistance (initialDistance : ℝ) (jackSpeed christinaSpeed lindySpeed : ℝ) : ℝ :=
  let meetingTime := initialDistance / (jackSpeed + christinaSpeed)
  lindySpeed * meetingTime

theorem lindy_distance_correct :
  lindyDistance 270 4 5 8 = 240 := by
  unfold lindyDistance
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_distance_correct_l610_61023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l610_61090

/-- The curve y = x³ - √3x + b, where b is a real constant -/
noncomputable def curve (b : ℝ) (x : ℝ) : ℝ := x^3 - Real.sqrt 3 * x + b

/-- The derivative of the curve -/
noncomputable def curve_derivative (x : ℝ) : ℝ := 3 * x^2 - Real.sqrt 3

/-- The angle of inclination of the tangent at a point on the curve -/
noncomputable def angle_of_inclination (x : ℝ) : ℝ := Real.arctan (curve_derivative x)

/-- The theorem stating the range of the angle of inclination -/
theorem angle_of_inclination_range (b : ℝ) :
  ∀ x : ℝ, angle_of_inclination x ∈ Set.union (Set.Icc 0 (Real.pi / 2)) (Set.Ico (2 * Real.pi / 3) Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l610_61090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_minimizes_sum_distances_l610_61062

-- Define a regular pentagon
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : ∀ i j : Fin 5, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

-- Define the center of a regular pentagon
noncomputable def center (p : RegularPentagon) : ℝ × ℝ :=
  let sum := (Finset.univ.sum fun i => p.vertices i)
  (sum.1 / 5, sum.2 / 5)

-- Define the sum of distances from a point to all vertices
noncomputable def sum_distances (p : RegularPentagon) (point : ℝ × ℝ) : ℝ :=
  Finset.univ.sum fun i => dist point (p.vertices i)

-- The theorem to prove
theorem center_minimizes_sum_distances (p : RegularPentagon) :
  ∀ point : ℝ × ℝ, sum_distances p (center p) ≤ sum_distances p point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_minimizes_sum_distances_l610_61062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l610_61047

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 - Real.sqrt 3 * Real.cos x - Real.sin x

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A : ℝ)
  (area : ℝ)

-- State the theorem
theorem triangle_problem :
  ∀ (t : Triangle),
    t.A > 0 ∧ t.A < Real.pi ∧
    f t.A = 4 ∧
    t.c = 3 ∧
    t.area = (3 * Real.sqrt 3) / 4 →
    (∀ x, f x ≥ 2) ∧
    t.a + t.b + t.c = 2 * Real.sqrt 3 + 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l610_61047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_umbrella_number_count_l610_61058

def is_umbrella_number (n : ℕ) : Bool :=
  n ≥ 100 && n < 1000 &&
  (n / 10 % 10 > n % 10) &&
  (n / 10 % 10 > n / 100)

def valid_digits : List ℕ := [0, 1, 2, 3, 4, 5, 6]

def is_valid_number (n : ℕ) : Bool :=
  n ≥ 100 && n < 1000 &&
  (valid_digits.contains (n / 100)) &&
  (valid_digits.contains (n / 10 % 10)) &&
  (valid_digits.contains (n % 10)) &&
  (n / 100 ≠ n / 10 % 10) &&
  (n / 100 ≠ n % 10) &&
  (n / 10 % 10 ≠ n % 10)

theorem umbrella_number_count :
  (List.filter (λ n => is_umbrella_number n && is_valid_number n) (List.range 1000)).length = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_umbrella_number_count_l610_61058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yw_length_in_right_triangle_with_circle_l610_61098

/-- A right triangle with a circle intersecting one side -/
structure RightTriangleWithCircle where
  -- Points of the triangle
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  -- Point where circle intersects XZ
  W : ℝ × ℝ
  -- Y is the right angle
  right_angle_at_Y : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0
  -- Circle with diameter YZ intersects XZ at W
  W_on_XZ : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ W = (t * X.1 + (1 - t) * Z.1, t * X.2 + (1 - t) * Z.2)
  W_on_circle : (W.1 - Y.1)^2 + (W.2 - Y.2)^2 = ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2) / 4

/-- The main theorem -/
theorem yw_length_in_right_triangle_with_circle 
  (triangle : RightTriangleWithCircle) 
  (area : ℝ) 
  (xz_length : ℝ) :
  area = 195 →
  xz_length = 30 →
  xz_length = Real.sqrt ((triangle.X.1 - triangle.Z.1)^2 + (triangle.X.2 - triangle.Z.2)^2) →
  Real.sqrt ((triangle.Y.1 - triangle.W.1)^2 + (triangle.Y.2 - triangle.W.2)^2) = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_yw_length_in_right_triangle_with_circle_l610_61098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionAreaTheorem_l610_61025

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  baseArea : ℝ
  volume : ℝ

/-- Represents a cross-section of the pyramid -/
structure CrossSection where
  area : ℝ

/-- The area of the cross-section closest to the base when the pyramid is divided into three equal volumes -/
noncomputable def crossSectionArea (pyramid : TriangularPyramid) : CrossSection :=
  { area := pyramid.baseArea / Real.rpow 9 (1/3) }

/-- Theorem stating the relationship between the base area and the cross-section area -/
theorem crossSectionAreaTheorem (pyramid : TriangularPyramid) 
  (h1 : pyramid.baseArea = 18) 
  (h2 : ∃ (v1 v2 : ℝ), v1 + v2 + pyramid.volume / 3 = pyramid.volume ∧ v1 = v2) :
  (crossSectionArea pyramid).area = 18 / Real.rpow 9 (1/3) := by
  sorry

#check crossSectionAreaTheorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionAreaTheorem_l610_61025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_total_proof_l610_61085

def shoe_price : ℝ := 200
def shoe_discount : ℝ := 0.3
def shirt_price : ℝ := 80
def shirt_quantity : ℕ := 2
def final_discount : ℝ := 0.05

theorem shopping_total_proof :
  (shoe_price * (1 - shoe_discount) + shirt_price * shirt_quantity) * (1 - final_discount) = 285 := by
  -- Calculation steps
  have discounted_shoe_price := shoe_price * (1 - shoe_discount)
  have total_shirt_price := shirt_price * shirt_quantity
  have subtotal := discounted_shoe_price + total_shirt_price
  have final_price := subtotal * (1 - final_discount)
  
  -- Proof steps (to be filled in)
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_total_proof_l610_61085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l610_61064

-- Define the function g(x) with parameter m
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := Real.exp x - m * x - Real.cos x

-- Theorem statement
theorem g_properties :
  ∃ (m : ℝ), m ≥ 0 ∧ 
  (∀ x, g m x ≥ 0) ∧ 
  (∃ ε > 0, ∀ x, |x| < ε → g m x ≥ g m 0) ∧
  (∀ m' : ℝ, m' ≥ 0 → m' ≠ m → 
    (∃ x, g m' x < 0) ∨ 
    (∀ ε > 0, ∃ x, |x| < ε ∧ g m' x < g m' 0)) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l610_61064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_sphere_properties_main_result_l610_61027

/-- A rectangular parallelepiped with an inscribed sphere -/
structure ParallelepipedWithSphere where
  k : ℝ  -- ratio of parallelepiped volume to sphere volume

/-- The angles at the base of the parallelepiped -/
noncomputable def base_angles (p : ParallelepipedWithSphere) : ℝ × ℝ :=
  (Real.arcsin (6 / (Real.pi * p.k)), Real.pi - Real.arcsin (6 / (Real.pi * p.k)))

/-- The constraint on the ratio k -/
def k_constraint (p : ParallelepipedWithSphere) : Prop :=
  p.k ≥ 6 / Real.pi

/-- Theorem stating the properties of the parallelepiped with inscribed sphere -/
theorem parallelepiped_sphere_properties (p : ParallelepipedWithSphere) :
  (base_angles p).1 > 0 ∧ 
  (base_angles p).1 < Real.pi / 2 ∧
  (base_angles p).2 > Real.pi / 2 ∧
  (base_angles p).2 < Real.pi ∧
  k_constraint p := by
  sorry

/-- The main result combining all properties -/
theorem main_result (p : ParallelepipedWithSphere) :
  let (α₁, α₂) := base_angles p
  α₁ = Real.arcsin (6 / (Real.pi * p.k)) ∧
  α₂ = Real.pi - Real.arcsin (6 / (Real.pi * p.k)) ∧
  k_constraint p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_sphere_properties_main_result_l610_61027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seminar_total_cost_l610_61036

/-- Calculates the total amount spent on a professional development seminar --/
theorem seminar_total_cost
  (regular_fee : ℝ)
  (discount_rate : ℝ)
  (num_teachers : ℕ)
  (food_allowance : ℝ)
  (tax_rate : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_rate = 0.075)
  (h3 : num_teachers = 22)
  (h4 : food_allowance = 10)
  (h5 : tax_rate = 0.06)
  : (regular_fee * (1 - discount_rate) * num_teachers * (1 + tax_rate) +
     food_allowance * num_teachers) = 3455.65 := by
  -- Calculate the discounted fee
  have discounted_fee : ℝ := regular_fee * (1 - discount_rate)
  
  -- Calculate the total discounted fee
  have total_discounted_fee : ℝ := discounted_fee * num_teachers
  
  -- Calculate the total tax
  have total_tax : ℝ := total_discounted_fee * tax_rate
  
  -- Calculate the total seminar fee
  have total_seminar_fee : ℝ := total_discounted_fee + total_tax
  
  -- Calculate the total food allowance
  have total_food_allowance : ℝ := food_allowance * num_teachers
  
  -- Prove the total cost
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seminar_total_cost_l610_61036
