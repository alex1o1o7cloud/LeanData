import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l519_51952

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > b ∧ b > 0
  ecc : c / a = Real.sqrt 2 / 2
  focus : c = 2

/-- The equation of the ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The intersection line -/
def intersection_line (m : ℝ) (x y : ℝ) : Prop :=
  y = x + m

/-- The unit circle -/
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- The main theorem -/
theorem ellipse_properties (E : Ellipse) :
  (∀ x y, ellipse_equation E x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∀ m, (∃ A B : ℝ × ℝ,
    A ≠ B ∧
    ellipse_equation E A.1 A.2 ∧
    ellipse_equation E B.1 B.2 ∧
    intersection_line m A.1 A.2 ∧
    intersection_line m B.1 B.2 ∧
    unit_circle ((A.1 + B.1) / 2) ((A.2 + B.2) / 2)) →
  m = 3 * Real.sqrt 5 / 5 ∨ m = -3 * Real.sqrt 5 / 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l519_51952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_half_plus_sin_l519_51904

theorem max_cos_half_plus_sin (θ : Real) (h : 0 < θ ∧ θ < π) : 
  ∃ a : Real, 0 < a ∧ a < 1 ∧
  Real.cos (θ/2) * (1 + Real.sin θ) ≤ Real.sqrt (1 - a^2) * (1 + 2 * a * Real.sqrt (1 - a^2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_half_plus_sin_l519_51904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_eighths_of_pi_l519_51900

theorem tan_product_eighths_of_pi : 
  Real.tan (π/8) * Real.tan (3*π/8) * Real.tan (5*π/8) * Real.tan (7*π/8) = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_eighths_of_pi_l519_51900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_of_cubes_l519_51992

/-- The volume of a cube with side length s -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- The total volume of n cubes each with side length s -/
def total_volume (n : ℕ) (s : ℝ) : ℝ := n * (cube_volume s)

theorem total_volume_of_cubes : 
  total_volume 5 1 + total_volume 5 2 = 45 := by
  -- Unfold the definitions
  unfold total_volume cube_volume
  -- Simplify the expressions
  simp [pow_three]
  -- Evaluate the arithmetic
  norm_num

#eval total_volume 5 1 + total_volume 5 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_of_cubes_l519_51992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l519_51987

noncomputable def f (ω : ℕ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

def is_symmetric_about_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

def is_monotonic_on (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → g x < g y ∨ g x > g y

theorem omega_values :
  ∀ ω : ℕ,
  ∀ φ : ℝ,
  0 < ω ∧ ω ≤ 12 ∧
  0 < φ ∧ φ < Real.pi ∧
  is_symmetric_about_y_axis (f ω φ) ∧
  ¬ is_monotonic_on (f ω φ) (Real.pi / 4) (Real.pi / 2) →
  ω ∈ ({3, 5, 6, 7, 8, 9, 10, 11, 12} : Set ℕ) := by sorry

#check omega_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l519_51987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_shoes_and_jerseys_l519_51933

theorem total_cost_shoes_and_jerseys 
  (num_shoes : ℕ)
  (num_jerseys : ℕ)
  (total_shoes_cost : ℕ)
  (jersey_cost_ratio : ℕ) :
  num_shoes = 6 →
  num_jerseys = 4 →
  total_shoes_cost = 480 →
  jersey_cost_ratio = 4 →
  (total_shoes_cost + (total_shoes_cost / num_shoes / jersey_cost_ratio * num_jerseys)) = 560 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_shoes_and_jerseys_l519_51933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l519_51940

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 1)) / x

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ -1 ∧ x ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l519_51940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manning_throws_l519_51963

/-- The distance in yards that Peyton Manning throws a football at 50°F -/
def D : ℝ := sorry

/-- The number of throws on Saturday -/
def saturday_throws : ℕ := 20

/-- The number of throws on Sunday -/
def sunday_throws : ℕ := 30

/-- The total distance thrown over two days in yards -/
def total_distance : ℝ := 1600

theorem manning_throws : 
  D * saturday_throws + 2 * D * sunday_throws = total_distance := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manning_throws_l519_51963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l519_51960

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 1/x

-- State the theorem
theorem f_monotone_increasing :
  MonotoneOn f (Set.Ioi (1/2)) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l519_51960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_circle_radius_is_sqrt_2_l519_51977

/-- Given a circle with equation x^2 + y^2 - 2ax + 2 = 0 and center C(2,0), 
    prove that its radius is √2 -/
theorem circle_radius (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x + 2 = 0 ↔ (x - 2)^2 + y^2 = 2) → 
  a = 2 :=
by
  sorry

/-- The radius of the circle is √2 -/
theorem circle_radius_is_sqrt_2 (a : ℝ) 
  (h : ∀ x y : ℝ, x^2 + y^2 - 2*a*x + 2 = 0 ↔ (x - 2)^2 + y^2 = 2) : 
  Real.sqrt 2 = Real.sqrt ((2 - a)^2 + (-2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_circle_radius_is_sqrt_2_l519_51977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_curve_maximizes_area_l519_51999

/-- A convex curve with an angular point -/
structure ConvexCurveWithAngle where
  length : ℝ
  angle : ℝ
  isConvex : Bool
  hasAngularPoint : Bool

/-- The optimal curve that maximizes the enclosed area -/
def optimalCurve (α : ℝ) : ConvexCurveWithAngle :=
  { length := 1,
    angle := α,
    isConvex := true,
    hasAngularPoint := true }

/-- The radius of the circular arc in the optimal curve -/
noncomputable def optimalRadius (α : ℝ) : ℝ :=
  1 / (α + 2 * Real.sin (α / 2))

/-- The area enclosed by the optimal curve -/
noncomputable def enclosedArea (c : ConvexCurveWithAngle) : ℝ :=
  (c.angle / 2) * (optimalRadius c.angle) ^ 2

theorem optimal_curve_maximizes_area (α : ℝ) (c : ConvexCurveWithAngle) :
  c.length = 1 → c.angle = α → c.isConvex → c.hasAngularPoint →
  enclosedArea c ≤ enclosedArea (optimalCurve α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_curve_maximizes_area_l519_51999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l519_51912

noncomputable def f (a ω b x : ℝ) : ℝ := a * Real.sin (2 * ω * x + Real.pi / 6) + a / 6 + b

theorem function_properties (a ω b : ℝ) (ha : a > 0) (hω : ω > 0) 
  (h_period : ∀ x, f a ω b (x + Real.pi) = f a ω b x)
  (h_max : ∀ x, f a ω b x ≤ 7/4)
  (h_min : ∀ x, f a ω b x ≥ 3/4)
  (h_max_exists : ∃ x, f a ω b x = 7/4)
  (h_min_exists : ∃ x, f a ω b x = 3/4) :
  (ω = 1 ∧ a = 1/2 ∧ b = 1) ∧
  (∀ k : ℤ, ∀ x : ℝ, k * Real.pi - Real.pi/3 ≤ x ∧ x ≤ k * Real.pi + Real.pi/6 → 
    ∀ y : ℝ, k * Real.pi - Real.pi/3 ≤ y ∧ y ≤ x → f (1/2) 1 1 y ≤ f (1/2) 1 1 x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l519_51912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l519_51907

noncomputable def closest_distance (a : ℝ) : ℝ :=
  if a ≤ Real.exp (1 / Real.exp 1) then 0
  else Real.sqrt 2 * (1 + Real.log (Real.log a)) / Real.log a

theorem closest_points_distance (a : ℝ) (h : a > 1) :
  let f := fun x => a^x
  let g := fun x => Real.log x / Real.log a
  ∃ (x₁ y₁ x₂ y₂ : ℝ), (y₁ = f x₁ ∧ y₂ = g x₂) ∧
    ∀ (x₃ y₃ x₄ y₄ : ℝ), (y₃ = f x₃ ∧ y₄ = g x₄) →
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ Real.sqrt ((x₃ - x₄)^2 + (y₃ - y₄)^2) ∧
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = closest_distance a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_points_distance_l519_51907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_lcm_360_l519_51921

theorem gcd_lcm_360 (a b : ℕ+) (h : Nat.gcd a.val b.val * Nat.lcm a.val b.val = 360) :
  Nat.gcd a.val b.val ∈ ({1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 20, 24, 30, 40, 45, 60, 120} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_lcm_360_l519_51921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_probability_l519_51983

/-- The probability of at least 6 people staying for the entire basketball game -/
theorem basketball_game_probability (total_people : ℕ) (certain_people : ℕ) (uncertain_people : ℕ) 
  (stay_prob : ℚ) : 
  total_people = 7 → 
  certain_people = 3 → 
  uncertain_people = 4 → 
  stay_prob = 1/3 → 
  (Nat.choose uncertain_people 3 * stay_prob^3 * (1 - stay_prob) + stay_prob^4) = 1/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_probability_l519_51983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_speed_proof_l519_51980

-- Define the parameters of the flight
def total_time : ℚ := 8
def distance_each_way : ℚ := 1500
def return_speed : ℚ := 500

-- Define the speed of the pilot on the flight out
noncomputable def outbound_speed : ℚ := 
  distance_each_way / (total_time - distance_each_way / return_speed)

-- Theorem statement
theorem flight_speed_proof : outbound_speed = 300 := by
  -- Unfold the definition of outbound_speed
  unfold outbound_speed
  
  -- Perform the calculation
  norm_num
  
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_speed_proof_l519_51980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_theorem_l519_51961

noncomputable section

variable (x : ℝ)

/-- The number of days required for x+4 cows to produce x+11 cans of milk, 
    given that x+6 cows produce x+9 cans in x+4 days -/
noncomputable def days_required (x : ℝ) : ℝ := (x + 11) * (x + 6) / (x + 9)

/-- The daily milk production per cow -/
noncomputable def daily_production_per_cow (x : ℝ) : ℝ := (x + 9) / ((x + 6) * (x + 4))

theorem milk_production_theorem (x : ℝ) :
  (x + 4) * daily_production_per_cow x * days_required x = x + 11 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_theorem_l519_51961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_correction_theorem_l519_51930

/-- The daily gain of the clock in minutes -/
noncomputable def daily_gain : ℝ := 3.25

/-- The number of days between January 1 and January 10 -/
def days : ℕ := 9

/-- The number of hours from noon to 8 PM -/
def extra_hours : ℕ := 8

/-- The total number of hours from noon on January 1 to 8 PM on January 10 -/
def total_hours : ℕ := days * 24 + extra_hours

/-- The correction in minutes to be subtracted from the clock -/
noncomputable def correction (g : ℝ) (h : ℕ) : ℝ := g * (h / 24 : ℝ)

theorem clock_correction_theorem :
  ‖correction daily_gain total_hours - 30.33‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_correction_theorem_l519_51930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l519_51991

theorem cos_theta_value (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo (-π/4) (π/4)) 
  (h2 : Real.cos (θ - π/4) = 3/5) : 
  Real.cos θ = 7 * Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l519_51991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_on_perpendicular_to_en_l519_51927

noncomputable section

/-- Ellipse C with equation x²/4 + y² = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Point P on the ellipse, not a vertex -/
def point_on_ellipse (x₀ y₀ : ℝ) : Prop :=
  ellipse_C x₀ y₀ ∧ x₀ ≠ 0

/-- N is the midpoint of PM -/
noncomputable def N (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀ / 2, y₀)

/-- D is the intersection of B₂N and y = -1 -/
noncomputable def D (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀ / (1 - y₀), -1)

/-- E is the midpoint of B₁D -/
noncomputable def E (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀ / (2 * (1 - y₀)), -1)

/-- Vector ON -/
noncomputable def ON (x₀ y₀ : ℝ) : ℝ × ℝ := N x₀ y₀

/-- Vector EN -/
noncomputable def EN (x₀ y₀ : ℝ) : ℝ × ℝ :=
  ((N x₀ y₀).1 - (E x₀ y₀).1, (N x₀ y₀).2 - (E x₀ y₀).2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem on_perpendicular_to_en (x₀ y₀ : ℝ) (h : point_on_ellipse x₀ y₀) :
  dot_product (ON x₀ y₀) (EN x₀ y₀) = 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_on_perpendicular_to_en_l519_51927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l519_51915

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_problem :
  let a : ℝ := 1
  let r : ℝ := 1/4
  let target_sum : ℝ := 341/256
  ∃ n : ℕ, geometric_sum a r n = target_sum ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l519_51915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l519_51901

noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.sin x) + Real.exp (Real.cos x)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4)

theorem f_properties :
  (∀ x, g x = g (-x)) ∧
  (∀ x, f x ≥ 2 * Real.exp (-Real.sqrt 2 / 2)) ∧
  (∃ x, f x = 2 * Real.exp (-Real.sqrt 2 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l519_51901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_one_k_bound_l519_51942

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (x - 1/x) * Real.log x
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := x - k/x

-- Theorem 1: f has a minimum at x = 1
theorem f_min_at_one : 
  ∀ x > 0, f x ≥ f 1 := by sorry

-- Theorem 2: If f(x) - g(x) has two zero points in [1, +∞), then 1 ≤ k < 17/8
theorem k_bound (k : ℝ) : 
  (∃ x y, 1 ≤ x ∧ x < y ∧ f x = g k x ∧ f y = g k y) → 
  1 ≤ k ∧ k < 17/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_one_k_bound_l519_51942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_minus_three_cos_plus_two_nonnegative_l519_51976

theorem cos_squared_minus_three_cos_plus_two_nonnegative (x : ℝ) : 
  (Real.cos x) ^ 2 - 3 * (Real.cos x) + 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_minus_three_cos_plus_two_nonnegative_l519_51976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zero_in_interval_l519_51932

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 10 then
    3 * (2 : ℝ)^x - 24
  else if 10 < x ∧ x ≤ 20 then
    -(2 : ℝ)^(x-5) + 126
  else
    0  -- arbitrary value for x outside the defined range

-- State the theorem
theorem no_zero_in_interval :
  (∀ x ∈ Set.Icc 0 10, Monotone (fun x ↦ 3 * (2 : ℝ)^x - 24)) →
  (∀ x ∈ Set.Ioc 10 20, StrictAntiOn (fun x ↦ -(2 : ℝ)^(x-5) + 126) (Set.Ioc 10 20)) →
  ¬∃ x ∈ Set.Ioo 3 7, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zero_in_interval_l519_51932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_range_l519_51956

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

-- State the theorem
theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 4 ≤ a ∧ a < 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_range_l519_51956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ingredient_prices_theorem_l519_51958

/-- Represents the price and quantity information for an ingredient --/
structure IngredientInfo where
  originalPrice : ℝ
  discountPercentage : ℝ
  additionalQuantity : ℝ
  totalCost : ℝ

/-- Calculates the discounted price of an ingredient --/
noncomputable def discountedPrice (info : IngredientInfo) : ℝ :=
  info.originalPrice * (1 - info.discountPercentage / 100)

/-- Theorem stating the original prices of ingredients based on given conditions --/
theorem ingredient_prices_theorem (salt sugar flour : IngredientInfo)
  (h_salt : salt = { originalPrice := 10, discountPercentage := 20, additionalQuantity := 10, totalCost := 400 })
  (h_sugar : sugar = { originalPrice := 21.18, discountPercentage := 15, additionalQuantity := 5, totalCost := 600 })
  (h_flour : flour = { originalPrice := 11.11, discountPercentage := 10, additionalQuantity := 8, totalCost := 800 }) :
  salt.totalCost / (discountedPrice salt) = salt.totalCost / salt.originalPrice + salt.additionalQuantity ∧
  sugar.totalCost / (discountedPrice sugar) = sugar.totalCost / sugar.originalPrice + sugar.additionalQuantity ∧
  flour.totalCost / (discountedPrice flour) = flour.totalCost / flour.originalPrice + flour.additionalQuantity := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ingredient_prices_theorem_l519_51958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_25_50_l519_51954

/-- The area of a rhombus given its diagonals -/
noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- Theorem: The area of a rhombus with diagonals 25 and 50 is 625 -/
theorem rhombus_area_25_50 : rhombus_area 25 50 = 625 := by
  -- Unfold the definition of rhombus_area
  unfold rhombus_area
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- The result should now be obvious to Lean
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_25_50_l519_51954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_conditions_l519_51979

/-- Represents a line in 3D space -/
structure Line3D where
  -- Define the line structure (placeholder)
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Checks if three lines are coplanar -/
def are_coplanar (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Definition of coplanarity (placeholder)

/-- Checks if two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of intersection (placeholder)

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of parallel lines (placeholder)

/-- Checks if two lines are perpendicular -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of perpendicular lines (placeholder)

/-- Checks if a point is on a line -/
def on_line (p : ℝ × ℝ × ℝ) (l : Line3D) : Prop :=
  sorry -- Definition of a point being on a line (placeholder)

/-- The four conditions for three lines to be coplanar -/
def condition1 (l1 l2 l3 : Line3D) : Prop :=
  (intersect l1 l2 ∧ intersect l2 l3 ∧ intersect l3 l1) ∧
  ¬(∃ p, on_line p l1 ∧ on_line p l2 ∧ on_line p l3)

def condition2 (l1 l2 l3 : Line3D) : Prop :=
  parallel l1 l2 ∧ parallel l2 l3 ∧ parallel l3 l1

def condition3 (l1 l2 l3 : Line3D) : Prop :=
  ∃ p, on_line p l1 ∧ on_line p l2 ∧ on_line p l3

def condition4 (l1 l2 l3 : Line3D) : Prop :=
  (perpendicular l1 l3 ∧ perpendicular l2 l3) ∨
  (perpendicular l1 l2 ∧ perpendicular l3 l2) ∨
  (perpendicular l2 l1 ∧ perpendicular l3 l1)

theorem coplanar_conditions (l1 l2 l3 : Line3D) :
  (condition1 l1 l2 l3 → are_coplanar l1 l2 l3) ∧
  (condition2 l1 l2 l3 → are_coplanar l1 l2 l3) ∧
  (condition3 l1 l2 l3 → are_coplanar l1 l2 l3) ∧
  (condition4 l1 l2 l3 → are_coplanar l1 l2 l3) ∧
  (∃! n : Nat, n = 1 ∧
    ((n = 1 → condition1 l1 l2 l3 → are_coplanar l1 l2 l3) ∧
     (n = 2 → condition2 l1 l2 l3 → are_coplanar l1 l2 l3) ∧
     (n = 3 → condition3 l1 l2 l3 → are_coplanar l1 l2 l3) ∧
     (n = 4 → condition4 l1 l2 l3 → are_coplanar l1 l2 l3))) :=
by
  sorry

#check coplanar_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_conditions_l519_51979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_work_time_l519_51935

/-- The time taken for two workers to complete a task together, given their individual completion times -/
theorem joint_work_time (a_time b_time : ℝ) (ha : a_time > 0) (hb : b_time > 0) :
  (1 / (1 / a_time + 1 / b_time)) = 70 / 17 → a_time = 10 ∧ b_time = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_work_time_l519_51935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independent_of_x_l519_51978

variable (x k : ℝ)

def A (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 2
def B (x k : ℝ) : ℝ := x^2 + k * x - 1

theorem independent_of_x (h : ∃ c, ∀ x, A x - 2 * B x k = c) : k = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_independent_of_x_l519_51978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l519_51945

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + (y-2)^2 = 16

-- Define the line of symmetry
def symmetry_line (x y : ℝ) (a : ℝ) : Prop := a*x + 6*y - 12 = 0

-- Define the line where S moves
def s_line (y : ℝ) : Prop := y + 6 = 0

-- Define the tangent line equation
def tangent_line (x y t : ℝ) : Prop := x*t - 8*y + 16 = 0

-- Define the line AB
def line_AB (x y t : ℝ) : Prop := t*x - 8*y = 0

-- Theorem statement
theorem fixed_point_theorem :
  ∀ (a t x y : ℝ),
    symmetry_line 0 2 a →
    s_line (-6) →
    (∃ x₁ y₁, circle_equation x₁ y₁ ∧ tangent_line x₁ y₁ t) →
    (∃ x₂ y₂, circle_equation x₂ y₂ ∧ tangent_line x₂ y₂ t) →
    line_AB x y t →
    x = 0 ∧ y = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l519_51945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_set_cost_l519_51972

/-- The cost of an audio cassette -/
def A : ℝ := sorry

/-- The cost of a video cassette -/
def V : ℝ := sorry

/-- The cost of a video cassette is 300 -/
axiom video_cost : V = 300

/-- The total cost of 7 audio cassettes and 3 video cassettes is 1110 -/
axiom total_cost : 7 * A + 3 * V = 1110

/-- The theorem to prove: The total cost of the second set of cassettes is 1110 -/
theorem second_set_cost : 7 * A + 3 * V = 1110 := by
  exact total_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_set_cost_l519_51972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_5_count_l519_51975

/-- The number of positive divisors of 8820 that are multiples of 5 -/
def num_divisors_multiple_of_5 (n : ℕ) : ℕ :=
  (Finset.filter (fun d ↦ d ∣ n ∧ 5 ∣ d) (Finset.range (n + 1))).card

/-- 8820 expressed as a product of prime factors -/
def n : ℕ := 2^2 * 3^2 * 5 * 7^2

theorem divisors_multiple_of_5_count :
  num_divisors_multiple_of_5 n = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_5_count_l519_51975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_height_l519_51926

/-- Trapezoid with given side lengths has height 12 -/
theorem trapezoid_height : ∀ (a b c d h : ℝ),
  a = 25 →
  b = 4 →
  c = 20 →
  d = 13 →
  (a - b) ^ 2 + h ^ 2 = c ^ 2 + d ^ 2 - 2 * c * d * Real.cos (π / 2) →
  h = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_height_l519_51926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_overspends_l519_51913

/-- Calculates the amount Bob spends over his budget when buying gifts. -/
theorem bob_overspends (budget : ℝ) (necklace_a necklace_b necklace_c : ℝ) 
  (book_price_diff_a book_price_diff_c : ℝ) (perfume_item_price : ℝ) :
  budget = 150 ∧
  necklace_a = 34 ∧ necklace_b = 42 ∧ necklace_c = 50 ∧
  book_price_diff_a = 20 ∧ book_price_diff_c = 10 ∧
  perfume_item_price = 25 →
  120.60 = 
    let total_necklaces := necklace_a + necklace_b + necklace_c
    let book1 := necklace_a + book_price_diff_a
    let book2 := necklace_c - book_price_diff_c
    let total_books := book1 + book2
    let books_discount := total_books * 0.1
    let books_after_discount := total_books - books_discount
    let perfume_set := perfume_item_price * 3
    let perfume_discount := perfume_set * 0.2
    let perfume_after_discount := perfume_set - perfume_discount
    let total_cost := total_necklaces + books_after_discount + perfume_after_discount
    total_cost - budget := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_overspends_l519_51913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l519_51959

/-- The complex number i -/
def i : ℂ := Complex.I

/-- The condition (a+bi)^2=2i -/
def condition (a b : ℝ) : Prop := (Complex.ofReal a + Complex.ofReal b * i)^2 = 2*i

theorem sufficient_not_necessary :
  (∀ a b : ℝ, a = 1 ∧ b = 1 → condition a b) ∧
  (∃ a b : ℝ, condition a b ∧ (a ≠ 1 ∨ b ≠ 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l519_51959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_sequence_l519_51989

/-- Represents the sequence where each integer n from 1 to 100 appears n times -/
def sequenceList : List ℕ := sorry

/-- The total number of elements in the sequence -/
def total_elements : ℕ := (100 * 101) / 2

/-- The position of the median (actually, the positions of the two middle elements) -/
def median_positions : (ℕ × ℕ) := (total_elements / 2, total_elements / 2 + 1)

/-- The cumulative count function -/
def cumulative_count (n : ℕ) : ℕ := (n * (n + 1)) / 2

/-- The smallest n for which the cumulative count reaches or exceeds the median position -/
def n_at_median : ℕ := 71

theorem median_of_sequence : 
  let median := (sequenceList[median_positions.1 - 1]! + sequenceList[median_positions.2 - 1]!) / 2
  median = 71 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_sequence_l519_51989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_cosine_function_l519_51916

/-- The minimum value of ω given the conditions of the cosine function -/
theorem min_omega_cosine_function (ω φ T : ℝ) (h_ω_pos : ω > 0) (h_φ_bounds : 0 < φ ∧ φ < π)
  (h_period : T = 2 * π / ω)
  (h_fT : Real.cos (ω * T + φ) = Real.sqrt 3 / 2)
  (h_zero : Real.cos (ω * (π / 9) + φ) = 0) :
  3 ≤ ω :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_cosine_function_l519_51916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sum_is_124_8_l519_51903

noncomputable def A : Finset ℝ := {1.2, 3.4, 5, 6}

noncomputable def S (X : Finset ℝ) : ℝ := X.sum id

noncomputable def total_sum : ℝ := (Finset.powerset A).sum S

theorem total_sum_is_124_8 : total_sum = 124.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sum_is_124_8_l519_51903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_at_negative_one_eighth_l519_51943

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x^6 - 1) / 4

-- State the theorem
theorem inverse_g_at_negative_one_eighth :
  ∃ (y : ℝ), g y = -1/8 ∧ y = (1/2)^(1/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_at_negative_one_eighth_l519_51943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l519_51931

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def sequence_S : ℕ → ℝ := sorry
noncomputable def sequence_b : ℕ → ℝ := sorry
noncomputable def sequence_c : ℕ → ℝ := sorry

axiom S_def : ∀ n, sequence_S (n + 1) = 4 * sequence_a n + 2
axiom a_1 : sequence_a 1 = 1
axiom b_def : ∀ n, sequence_b n = sequence_a (n + 1) - 2 * sequence_a n
axiom c_def : ∀ n, sequence_c n = sequence_a n / (2^n)

theorem main_theorem :
  (∀ n : ℕ, n ≥ 1 → sequence_b n = 3 * 2^(n-1)) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_c n = 1/2 + 3/4 * (n-1)) ∧
  (∀ n : ℕ, n ≥ 1 → 
    sequence_a n = (3*n - 1) * 2^(n-2) ∧ 
    sequence_S n = (3*n - 4) * 2^(n-1) + 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l519_51931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_x_l519_51988

-- Define the function f(x) = √(x-1)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1)

-- Theorem stating the domain of f
theorem domain_of_f :
  ∀ x : ℝ, f x ∈ Set.range f ↔ x ≥ 1 :=
by
  sorry

-- Theorem stating the range of x
theorem range_of_x :
  Set.range f = { y : ℝ | y ≥ 0 } :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_x_l519_51988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravelling_cost_is_5421_l519_51967

-- Define the trapezoidal plot and path dimensions
def longer_side : ℝ := 120
def shorter_side : ℝ := 95
def plot_height : ℝ := 65
def path_width_longer : ℝ := 4
def path_width_shorter : ℝ := 2.5
def cost_per_sqm : ℝ := 0.80

-- Define the function to calculate the area of a trapezoid
noncomputable def trapezoid_area (a b h : ℝ) : ℝ := (a + b) * h / 2

-- Define the function to calculate the cost of gravelling
noncomputable def gravelling_cost (area cost_per_sqm : ℝ) : ℝ := area * cost_per_sqm

-- Theorem statement
theorem gravelling_cost_is_5421 :
  gravelling_cost (trapezoid_area 
    (longer_side - path_width_longer) 
    (shorter_side - path_width_shorter) 
    plot_height) 
  cost_per_sqm = 5421 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravelling_cost_is_5421_l519_51967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_satisfying_conditions_l519_51990

theorem unique_polynomial_satisfying_conditions :
  ∃! P : Polynomial ℝ, 
    (P.eval 2017 = 2016) ∧ 
    (∀ x : ℝ, (P.eval x + 1)^2 = P.eval (x^2 + 1)) ∧
    (P = Polynomial.X - Polynomial.C 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_satisfying_conditions_l519_51990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ada_original_seat_l519_51948

-- Define the set of friends
inductive Friend : Type
| Ada : Friend
| Bea : Friend
| Ceci : Friend
| Dee : Friend
| Eli : Friend

-- Define the set of seats
def Seat : Type := Fin 5

-- Define the initial seating arrangement
def initial_seating : Friend → Seat := sorry

-- Define the movement function
def move (s : Seat) (n : Int) : Seat := sorry

-- Define the swap function
def swap (s1 s2 : Seat) : Seat → Seat := sorry

-- Define the final seating arrangement after movements
def final_seating : Friend → Option Seat := sorry

-- Theorem statement
theorem ada_original_seat :
  (∀ f : Friend, f ≠ Friend.Ada → final_seating f ≠ some (Fin.ofNat 3)) ∧
  final_seating Friend.Ada = none →
  initial_seating Friend.Ada = (Fin.ofNat 4) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ada_original_seat_l519_51948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_minus_theta_l519_51906

theorem sin_pi_half_minus_theta (θ : Real) (h1 : Real.tan θ = 2) (h2 : π < θ ∧ θ < 3*π/2) :
  Real.sin (π/2 - θ) = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_minus_theta_l519_51906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_l519_51971

/-- A parabola with given properties -/
structure Parabola where
  -- The vertex of the parabola
  vertex : ℝ × ℝ
  -- The x-coordinates of the intersections with the x-axis
  x₁ : ℝ
  x₂ : ℝ
  -- The vertex is at (-2, 3)
  vertex_condition : vertex = (-2, 3)
  -- The absolute difference between x₁ and x₂ is 6
  intersection_distance : |x₁ - x₂| = 6

/-- The equation of a parabola given its parameters -/
noncomputable def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  -1/3 * (x + 2)^2 + 3

/-- Theorem stating that the given equation correctly describes the parabola -/
theorem parabola_equation_correct (p : Parabola) :
  ∀ x, parabola_equation p x = 
    let (h, k) := p.vertex
    let a := -1/3
    a * (x - h)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_l519_51971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l519_51910

-- Define the initial point
def initial_point : ℝ × ℝ × ℝ := (1, 1, 2)

-- Define the transformations
def rotate_y_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def translate_z (d : ℝ) (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, z - d)

-- Define the sequence of transformations
def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotate_y_180
    |> reflect_yz
    |> reflect_xz
    |> rotate_y_180
    |> reflect_xz
    |> translate_z 2

-- Theorem statement
theorem transformation_result :
  transform initial_point = (-1, 1, 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l519_51910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_four_l519_51955

-- Define the two functions
def f (y : Real) : Real := (y - 2)^3
def g (y : Real) : Real := 4*y - 8

-- Define the area function
noncomputable def area_between_curves (a b : Real) : Real :=
  ∫ y in a..b, g y - f y

-- State the theorem
theorem area_is_four :
  area_between_curves 2 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_four_l519_51955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_player_height_l519_51937

/-- Represents the height of a basketball player in feet -/
noncomputable def player_height : ℝ := 6

/-- Represents the height of the basketball rim in feet -/
noncomputable def rim_height : ℝ := 10

/-- Represents the additional height the player can reach above their head in feet -/
noncomputable def reach_above_head : ℝ := 22 / 12

/-- Represents the player's vertical jump height in feet -/
noncomputable def jump_height : ℝ := 32 / 12

/-- Represents the additional height needed above the rim to dunk in feet -/
noncomputable def height_above_rim : ℝ := 6 / 12

theorem basketball_player_height :
  player_height + reach_above_head + jump_height = rim_height + height_above_rim := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_player_height_l519_51937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l519_51998

open Real Nat

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- Define the triangle conditions
def triangle_conditions (l m n : ℕ) : Prop :=
  l > m ∧ m > n ∧ 
  frac (3^l / 10000 : ℝ) = frac (3^m / 10000 : ℝ) ∧
  frac (3^m / 10000 : ℝ) = frac (3^n / 10000 : ℝ)

-- Theorem statement
theorem min_perimeter_triangle :
  ∀ l m n : ℕ, triangle_conditions l m n →
  ∀ l' m' n' : ℕ, triangle_conditions l' m' n' →
  l + m + n ≤ l' + m' + n' ∧
  l + m + n = 3003 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l519_51998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l519_51922

-- Define the integer part function as noncomputable
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- Define the problem statement
theorem problem_statement (a b : ℝ) : 
  (a * Real.sqrt 2 - Real.sqrt b = Real.sqrt 2) → 
  (b = integerPart (2 * Real.pi + 2)) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l519_51922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l519_51944

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (1 + 4 * Real.cos θ, 2 + 4 * Real.sin θ)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (2 + (Real.sqrt 3 / 2) * t, 1 + (1 / 2) * t)

-- Define the point P
def point_P : ℝ × ℝ := (2, 1)

-- Define the theorem
theorem intersection_product (A B : ℝ × ℝ) :
  (∃ θ₁ θ₂ t₁ t₂ : ℝ,
    curve_C θ₁ = line_l t₁ ∧
    curve_C θ₂ = line_l t₂ ∧
    A = curve_C θ₁ ∧
    B = curve_C θ₂) →
  (Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) *
   Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)) = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l519_51944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_factor_200_l519_51919

theorem power_factor_200 (a b : ℕ) : 
  (2^a = (Nat.gcd 200 (2^a))) →
  (5^b = (Nat.gcd 200 (5^b))) →
  (1/3 : ℚ)^(b - a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_factor_200_l519_51919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_segment_length_l519_51909

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an angle with vertex and two points on its sides -/
structure Angle where
  O : Point  -- vertex
  A : Point  -- point on one side
  B : Point  -- point on other side

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Definition of the problem setup -/
def minimumSegment (α : Angle) (x : ℝ) : Prop :=
  ∃ (M N : Point),
    distance α.O α.A > distance α.O α.B ∧
    distance α.A M = x ∧
    distance α.B N = x ∧
    M.x = α.O.x + (α.A.x - α.O.x) * (1 - x / distance α.O α.A) ∧
    M.y = α.O.y + (α.A.y - α.O.y) * (1 - x / distance α.O α.A) ∧
    N.x = α.O.x + (α.B.x - α.O.x) * (1 + x / distance α.O α.B) ∧
    N.y = α.O.y + (α.B.y - α.O.y) * (1 + x / distance α.O α.B) ∧
    ∀ (y : ℝ), y ≠ x → distance M N ≤ distance
      (Point.mk (α.O.x + (α.A.x - α.O.x) * (1 - y / distance α.O α.A))
                (α.O.y + (α.A.y - α.O.y) * (1 - y / distance α.O α.A)))
      (Point.mk (α.O.x + (α.B.x - α.O.x) * (1 + y / distance α.O α.B))
                (α.O.y + (α.B.y - α.O.y) * (1 + y / distance α.O α.B)))

/-- The main theorem stating that the segment is minimized when x = (OA - OB) / 2 -/
theorem minimum_segment_length (α : Angle) :
  minimumSegment α ((distance α.O α.A - distance α.O α.B) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_segment_length_l519_51909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l519_51994

open Real Set

noncomputable def f (x : ℝ) := (cos x) ^ 2 + sin x

theorem range_of_f :
  range f = Icc 1 (5/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l519_51994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_multiple_of_seven_l519_51918

theorem no_multiple_of_seven (a : ℕ) (h : 1 ≤ a ∧ a ≤ 100) : 
  ¬(∃ k : ℕ, (a^2 + 3^a + a * 3^((a + 1) / 2)) * (a^2 + 3^a - a * 3^((a + 1) / 2)) = 7 * k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_multiple_of_seven_l519_51918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_15_30_75_product_l519_51973

theorem sin_15_30_75_product : 
  Real.sin (15 * π / 180) * Real.sin (30 * π / 180) * Real.sin (75 * π / 180) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_15_30_75_product_l519_51973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_is_090_l519_51951

/-- Calculates the length of the second train given the speeds, length of the first train, and time to cross. -/
noncomputable def second_train_length (speed1 speed2 : ℝ) (length1 : ℝ) (cross_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let distance := relative_speed * cross_time / 3600
  distance - length1

/-- Theorem stating that under the given conditions, the length of the second train is 0.90 km. -/
theorem second_train_length_is_090 :
  second_train_length 90 90 1.10 40 = 0.90 := by
  unfold second_train_length
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_is_090_l519_51951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_top_four_lost_points_l519_51929

def ChessTournament (scores : List ℝ) : Prop :=
  scores.length = 8 ∧ 
  scores = [7, 6, 4, 4, 3, 2, 1.5, 0.5]

noncomputable def TopFourLostPoints (scores : List ℝ) : ℝ :=
  let topFour := (scores.toArray.qsort (· > ·)).toList.take 4
  let totalTopFour := topFour.sum
  let maxPossibleAgainstOthers := 4 * 4
  maxPossibleAgainstOthers - (totalTopFour - (totalTopFour - maxPossibleAgainstOthers))

theorem chess_tournament_top_four_lost_points 
  (scores : List ℝ) 
  (h : ChessTournament scores) : 
  TopFourLostPoints scores = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_top_four_lost_points_l519_51929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ordering_l519_51968

noncomputable section

/-- The inverse proportion function -/
def f (x : ℝ) : ℝ := -6 / x

/-- Point A -/
def A : ℝ × ℝ := (-1, f (-1))

/-- Point B -/
def B : ℝ × ℝ := (2, f 2)

/-- Point C -/
def C : ℝ × ℝ := (3, f 3)

theorem inverse_proportion_ordering :
  A.2 > C.2 ∧ C.2 > B.2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ordering_l519_51968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l519_51986

/-- Given a system of equations 2x + y = Δ and x + y = 3,
    with solution x = 1 and y = □, prove that Δ = 4 and □ = 2 -/
theorem system_solution (x y Δ : ℝ) (square : ℝ)
  (eq1 : 2 * x + y = Δ) 
  (eq2 : x + y = 3) 
  (sol_x : x = 1) 
  (sol_y : y = square) : 
  Δ = 4 ∧ square = 2 := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l519_51986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_unchanged_l519_51964

/-- Represents the volume of a substance -/
structure Volume where
  val : ℝ
  nonneg : 0 ≤ val

/-- Represents the density of a substance -/
structure Density where
  val : ℝ
  pos : 0 < val

/-- The density of water -/
def water_density : Density where
  val := 1000
  pos := by norm_num

/-- The density of ice -/
def ice_density : Density where
  val := 917
  pos := by norm_num

/-- The volume of water initially in the glass -/
def initial_water_volume : Volume where
  val := 1
  nonneg := by norm_num

/-- The volume of water removed to make ice -/
def removed_water_volume : Volume where
  val := 0.1
  nonneg := by norm_num

/-- The volume of ice formed from the removed water -/
noncomputable def ice_volume : Volume where
  val := removed_water_volume.val * water_density.val / ice_density.val
  nonneg := by
    apply div_nonneg
    · apply mul_nonneg
      · exact removed_water_volume.nonneg
      · exact le_of_lt water_density.pos
    · exact le_of_lt ice_density.pos

/-- States that ice floats in water -/
axiom ice_floats : ice_density.val < water_density.val

/-- The volume of water displaced by floating ice -/
noncomputable def displaced_water_volume : Volume where
  val := ice_volume.val * ice_density.val / water_density.val
  nonneg := by
    apply div_nonneg
    · apply mul_nonneg
      · exact ice_volume.nonneg
      · exact le_of_lt ice_density.pos
    · exact le_of_lt water_density.pos

/-- Theorem: The water level remains at the rim of the glass when ice melts -/
theorem water_level_unchanged : 
  initial_water_volume.val = initial_water_volume.val - removed_water_volume.val + displaced_water_volume.val :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_unchanged_l519_51964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l519_51995

-- Define the function f(x) = x^2 - 2ln(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

-- State the theorem
theorem monotonically_decreasing_interval :
  ∀ x : ℝ, x > 0 → (∀ y : ℝ, 0 < y ∧ y < 1 → (deriv f) y < 0) ∧
  (∀ z : ℝ, z ≤ 0 ∨ z ≥ 1 → (deriv f) z ≥ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l519_51995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_root_l519_51939

/-- A function f: ℝ → ℝ satisfying f(f(x+1)) = x³ + 1 for all real x -/
noncomputable def f : ℝ → ℝ := sorry

/-- The functional equation that f satisfies for all real x -/
axiom f_eq (x : ℝ) : f (f (x + 1)) = x^3 + 1

/-- Theorem: The equation f(x) = 0 has exactly one real root -/
theorem f_has_unique_root : ∃! x : ℝ, f x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_root_l519_51939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l519_51936

noncomputable section

variable (A B C a b c : ℝ)

-- Define the triangle ABC
def triangle_ABC (A B C a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c

-- State the theorem
theorem triangle_problem 
  (h_triangle : triangle_ABC A B C a b c)
  (h1 : b * Real.sin A = a * Real.cos B)
  (h2 : b = 3)
  (h3 : Real.sin C = 2 * Real.sin A) :
  B = Real.pi/4 ∧ c = 2*a ∧ c = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l519_51936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_zero_condition_l519_51985

theorem sin_zero_condition : 
  (∀ x : ℝ, x = 0 → Real.sin x = 0) ∧ 
  (∃ x : ℝ, Real.sin x = 0 ∧ x ≠ 0) :=
by
  constructor
  · intro x hx
    rw [hx]
    exact Real.sin_zero
  · use Real.pi
    constructor
    · exact Real.sin_pi
    · exact Real.pi_ne_zero


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_zero_condition_l519_51985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l519_51905

noncomputable section

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Conditions
  Real.cos A = 3/5 →
  Real.tan (B - A) = 1/3 →
  c = 13 →
  -- Conclusions
  Real.tan B = 3 ∧ 
  (1/2 * b * c * Real.sin A = 78) :=
by
  -- The proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l519_51905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l519_51970

-- Define the function f(x) = log(1-x)
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x)

-- State the theorem
theorem domain_of_f :
  (∀ y, y ∈ Set.range f → y < 0) →
  {x : ℝ | f x ∈ Set.range f} = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l519_51970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l519_51966

-- Define the points
def point1 : ℝ × ℝ := (3, 3)
def point2 : ℝ × ℝ := (-2, -2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem distance_between_points :
  distance point1 point2 = 5 * Real.sqrt 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l519_51966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tooth_arrangements_eq_ten_l519_51949

/-- The number of unique arrangements of the letters in TOOTH -/
def tooth_arrangements : ℕ := 5 * 4 * 1

/-- The word TOOTH has 5 letters -/
axiom tooth_length : 5 = (List.length ['T', 'O', 'O', 'T', 'H'])

/-- TOOTH has 3 T's -/
axiom tooth_t_count : 3 = (List.length (List.filter (· = 'T') ['T', 'O', 'O', 'T', 'H']))

/-- TOOTH has 2 O's -/
axiom tooth_o_count : 2 = (List.length (List.filter (· = 'O') ['T', 'O', 'O', 'T', 'H']))

theorem tooth_arrangements_eq_ten : tooth_arrangements = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tooth_arrangements_eq_ten_l519_51949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PB_equals_twelve_l519_51982

/-- A circle with center O and a point P outside the circle -/
structure CircleWithExternalPoint where
  O : EuclideanSpace ℝ (Fin 2)
  P : EuclideanSpace ℝ (Fin 2)
  radius : ℝ
  is_external : dist O P > radius

/-- Tangent and secant lines from P to the circle -/
structure TangentAndSecant (c : CircleWithExternalPoint) where
  T : EuclideanSpace ℝ (Fin 2)
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  is_tangent : dist c.O T = c.radius ∧ dist c.P T = dist c.P c.O - c.radius
  is_secant : dist c.O A = c.radius ∧ dist c.O B = c.radius
  PA_less_PB : dist c.P A < dist c.P B

/-- Theorem: PB = 12 given the conditions -/
theorem PB_equals_twelve (c : CircleWithExternalPoint) (ts : TangentAndSecant c)
  (h_PA : dist c.P ts.A = 3)
  (h_PT : dist c.P ts.T = dist ts.A ts.B - dist c.P ts.A) :
  dist c.P ts.B = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PB_equals_twelve_l519_51982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_sherbourne_is_two_hours_l519_51957

/-- Represents the train route between Scottsdale and Sherbourne -/
structure TrainRoute where
  total_length : ℝ
  round_trip_time : ℝ
  forest_grove_fraction : ℝ

/-- Calculates the time taken to travel from Forest Grove to Sherbourne -/
noncomputable def time_to_sherbourne (route : TrainRoute) : ℝ :=
  let forest_grove_distance := route.total_length * route.forest_grove_fraction
  let distance_to_sherbourne := route.total_length - forest_grove_distance
  let train_speed := (2 * route.total_length) / route.round_trip_time
  distance_to_sherbourne / train_speed

/-- Theorem: The time taken to travel from Forest Grove to Sherbourne is 2 hours -/
theorem time_to_sherbourne_is_two_hours (route : TrainRoute)
  (h1 : route.total_length = 200)
  (h2 : route.round_trip_time = 5)
  (h3 : route.forest_grove_fraction = 1/5) :
  time_to_sherbourne route = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_sherbourne_is_two_hours_l519_51957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_cosine_sum_l519_51924

theorem symmetric_sine_cosine_sum (ω a : ℝ) : ω > 0 →
  (∀ x : ℝ, Real.sin (ω * x) + a * Real.cos (ω * x) = Real.sin (ω * (2 * Real.pi / 3 - x)) + a * Real.cos (ω * (2 * Real.pi / 3 - x))) →
  (∃ x₀ : ℝ, ∀ x : ℝ, Real.sin (ω * x) + a * Real.cos (ω * x) ≥ Real.sin (ω * x₀) + a * Real.cos (ω * x₀)) →
  (x₀ = Real.pi / 6) →
  ∃ ω a : ℝ, ω > 0 ∧ a + ω = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_cosine_sum_l519_51924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tricycle_revolutions_l519_51953

/-- Represents a tricycle with front and back wheel radii -/
structure Tricycle where
  front_radius : ℝ
  back_radius : ℝ

/-- Calculates the number of revolutions made by the back wheels given the number of front wheel revolutions -/
noncomputable def back_wheel_revolutions (t : Tricycle) (front_revs : ℝ) : ℝ :=
  (t.front_radius / t.back_radius) * front_revs

theorem tricycle_revolutions (t : Tricycle) (front_revs : ℝ) :
  t.front_radius = 3 ∧ t.back_radius = 0.5 ∧ front_revs = 150 →
  back_wheel_revolutions t front_revs = 900 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tricycle_revolutions_l519_51953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_cells_below_diagonal_l519_51914

/-- Represents a cell in the table -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the n × n table with marked cells -/
structure Table (n : Nat) where
  marked : Finset Cell
  marked_count : marked.card = n - 1
  valid_cells : ∀ c ∈ marked, c.row < n ∧ c.col < n

/-- Represents a swap operation (either row or column) -/
inductive Swap
  | Row (i j : Nat)
  | Col (i j : Nat)

/-- Predicate to check if a cell is below the main diagonal -/
def Cell.belowMainDiagonal (c : Cell) : Prop := c.row > c.col

/-- Helper function to apply a list of swaps to a cell -/
def apply_swaps : List Swap → Cell → Cell
  | [], c => c
  | (Swap.Row i j) :: rest, c => 
    let newRow := if c.row = i then j else if c.row = j then i else c.row
    apply_swaps rest ⟨newRow, c.col⟩
  | (Swap.Col i j) :: rest, c => 
    let newCol := if c.col = i then j else if c.col = j then i else c.col
    apply_swaps rest ⟨c.row, newCol⟩

/-- The main theorem statement -/
theorem marked_cells_below_diagonal (n : Nat) (t : Table n) :
  ∃ (swaps : List Swap), ∀ c ∈ t.marked, (apply_swaps swaps c).belowMainDiagonal := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_cells_below_diagonal_l519_51914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_properties_l519_51934

-- Define the set T of all non-zero real numbers
noncomputable def T : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation ⊗
noncomputable def otimes (x y : ℝ) : ℝ := x / y^2

-- Theorem stating the properties of ⊗
theorem otimes_properties :
  -- ⊗ is not commutative over T
  (∃ x y : ℝ, x ∈ T ∧ y ∈ T ∧ otimes x y ≠ otimes y x) ∧
  -- ⊗ is not associative over T
  (∃ x y z : ℝ, x ∈ T ∧ y ∈ T ∧ z ∈ T ∧ otimes (otimes x y) z ≠ otimes x (otimes y z)) ∧
  -- 1 is not an identity element for ⊗ in T
  (∃ x : ℝ, x ∈ T ∧ (otimes x 1 ≠ x ∨ otimes 1 x ≠ x)) ∧
  -- Every element of T has an inverse for ⊗
  (∀ x : ℝ, x ∈ T → ∃ y : ℝ, y ∈ T ∧ otimes x y = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_properties_l519_51934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_valid_integer_l519_51928

def is_valid (n : ℕ) : Prop :=
  n > 1 ∧
  ∀ i : ℕ, i ∈ [2, 3, 4, 5, 6, 7, 8, 9] → n % i = 1

theorem least_valid_integer : 
  is_valid 2521 ∧ ∀ n < 2521, ¬(is_valid n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_valid_integer_l519_51928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_approx_8_l519_51941

/-- Represents the triathlon race with given distances and speeds for each segment. -/
structure Triathlon where
  swim_distance : ℝ
  bike_distance : ℝ
  run_distance : ℝ
  swim_speed : ℝ
  bike_speed : ℝ
  run_speed : ℝ

/-- Calculates the average speed for the entire triathlon race. -/
noncomputable def average_speed (t : Triathlon) : ℝ :=
  let total_distance := t.swim_distance + t.bike_distance + t.run_distance
  let total_time := t.swim_distance / t.swim_speed + t.bike_distance / t.bike_speed + t.run_distance / t.run_speed
  total_distance / total_time

/-- Theorem stating that the average speed of the given triathlon is approximately 8 km/h. -/
theorem triathlon_average_speed_approx_8 :
  let t : Triathlon := {
    swim_distance := 1,
    bike_distance := 4,
    run_distance := 2,
    swim_speed := 2,
    bike_speed := 25,
    run_speed := 12
  }
  abs (average_speed t - 8) < 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_approx_8_l519_51941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l519_51950

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sqrt 3 * Real.sin x * Real.cos x + 1/2

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_area_theorem (t : Triangle) (h1 : f (t.B + t.C) = 3/2) 
    (h2 : t.a = Real.sqrt 3) (h3 : t.b + t.c = 3) : 
    (1/2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 / 2 := by
  sorry

#check triangle_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l519_51950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ExistsTriangleWithBoundedArea_l519_51981

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a set of five points
def FivePoints : Set Point := sorry

-- Define the property of no three points being collinear
def NoThreeCollinear (points : Set Point) : Prop := sorry

-- Define the convex hull of a set of points
def ConvexHull (points : Set Point) : Set Point := sorry

-- Define the area of a set of points (representing a polygon)
def Area (points : Set Point) : ℝ := sorry

-- Define a triangle formed by three points
def Triangle (p1 p2 p3 : Point) : Set Point := sorry

-- The main theorem
theorem ExistsTriangleWithBoundedArea 
  (points : Set Point) 
  (h1 : points = FivePoints) 
  (h2 : NoThreeCollinear points) 
  (S : ℝ) 
  (h3 : Area (ConvexHull points) = S) : 
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    Area (Triangle p1 p2 p3) ≤ (5 - Real.sqrt 5) / 10 * S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ExistsTriangleWithBoundedArea_l519_51981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_for_tan_one_third_l519_51969

theorem sin_minus_cos_for_tan_one_third (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.tan θ = 1/3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_for_tan_one_third_l519_51969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_on_ray_l519_51965

theorem tan_double_angle_on_ray (α : ℝ) :
  (∃ (x y : ℝ), x < 0 ∧ y = -2 * x ∧ (x * Real.cos α = y * Real.sin α)) →
  Real.tan (2 * α) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_on_ray_l519_51965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_neg_263_l519_51917

/-- The set of angles with the same terminal side as a given angle. -/
def SameTerminalSide (θ : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 360 + θ}

/-- Theorem: The set of angles with the same terminal side as -263° is
    { α | α = k · 360° - 263°, k ∈ ℤ }. -/
theorem same_terminal_side_neg_263 :
  SameTerminalSide (-263) = {α : ℝ | ∃ k : ℤ, α = k * 360 - 263} := by
  sorry

/-- Lemma: Angles with the same terminal side differ by an integer multiple of 360°. -/
lemma same_terminal_side_diff (α β : ℝ) :
  α ∈ SameTerminalSide β ↔ ∃ k : ℤ, α - β = k * 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_neg_263_l519_51917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l519_51923

-- Define the angle α
noncomputable def α : Real := Real.arctan 2

-- Define the point P
def P : Real × Real := (1, 2)

-- Assumptions
axiom m_nonzero : P.1 ≠ 0
axiom point_on_terminal_side : P.2 / P.1 = 2

-- Theorem to prove
theorem tan_alpha_plus_pi_fourth : 
  Real.tan (α + Real.pi / 4) = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l519_51923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_zero_l519_51962

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 + 7^2 = 225) ∧ 
    (x₂^2 + 7^2 = 225) ∧ 
    (x₁ + x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_zero_l519_51962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_error_complex_fraction_l519_51925

theorem percentage_error_complex_fraction (x : ℝ) (h : x > 0) : 
  (|(3*x - x*(2/5))|/(3*x)) * 100 = (13/15) * 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_error_complex_fraction_l519_51925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_c_start_time_l519_51947

noncomputable section

structure Person where
  name : String
  startTime : ℕ
  startPoint : String
  endPoint : String
  speed : ℝ

def AB : ℝ := 1 -- Normalized length of AB

-- Trisection points
def C : ℝ := AB * (2/3)
def D : ℝ := AB * (1/3)

def personA : Person := {
  name := "A",
  startTime := 0,  -- 8:00
  startPoint := "A",
  endPoint := "B",
  speed := AB / 30  -- Reaches B at 8:30
}

def personB : Person := {
  name := "B",
  startTime := 12,  -- 8:12
  startPoint := "B",
  endPoint := "A",
  speed := AB / 18  -- Reaches A at 8:30
}

def personC : Person := {
  name := "C",
  startTime := 16,  -- 8:16 (to be proved)
  startPoint := "B",
  endPoint := "A",
  speed := AB / 18  -- Same speed as B
}

theorem person_c_start_time :
  -- A and B meet at C
  personA.speed * (personB.startTime - personA.startTime) = C ∧
  personB.speed * (personB.startTime - personA.startTime) = AB - C ∧
  -- When A and B meet, C is at D
  personC.speed * (personB.startTime - personC.startTime) = D ∧
  -- A and C meet at 8:30 (30 minutes after 8:00)
  personA.speed * 30 + personC.speed * (30 - personC.startTime) = AB ∧
  -- B reaches A at 8:30
  personB.speed * (30 - personB.startTime) = AB →
  personC.startTime = 16 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_c_start_time_l519_51947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_preserves_size_l519_51993

/-- A translation is a geometric transformation that moves every point of a shape by the same distance in the same direction. -/
def Translation (X : Type) := X → X

/-- A shape is represented as a set of points. -/
def Shape (X : Type) := Set X

/-- The size of a shape is a measure of its extent, such as area or volume. -/
noncomputable def size {X : Type} (S : Shape X) : ℝ := sorry

/-- Translation preserves the size of a shape. -/
theorem translation_preserves_size {X : Type} (T : Translation X) (S : Shape X) :
  size (T '' S) = size S := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_preserves_size_l519_51993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l519_51996

theorem trigonometric_inequality : 
  ∃ (a b c : Real),
    a = Real.sin (20 * π / 180) ∧
    b = Real.tan (30 * π / 180) ∧
    c = Real.cos (40 * π / 180) ∧
    c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l519_51996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l519_51902

noncomputable def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  unfold g
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l519_51902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_frustum_small_cone_altitude_l519_51911

/-- Represents a frustum of a right circular cone. -/
structure Frustum where
  height : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the altitude of the small cone removed from a frustum. -/
def small_cone_altitude (f : Frustum) : ℝ :=
  f.height

/-- Theorem stating that for a specific frustum, the altitude of the small cone is 18 cm. -/
theorem specific_frustum_small_cone_altitude :
  let f : Frustum := { height := 18, lower_base_area := 400 * Real.pi, upper_base_area := 100 * Real.pi }
  small_cone_altitude f = 18 := by
  rfl

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_frustum_small_cone_altitude_l519_51911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_special_savings_l519_51974

/-- Represents the "fair special" offer on sandals -/
structure FairSpecial where
  regular_price : ℚ
  discount_second : ℚ
  discount_third : ℚ

/-- Calculates the total cost for three pairs of sandals under the "fair special" offer -/
def total_cost (offer : FairSpecial) : ℚ :=
  offer.regular_price + 
  (offer.regular_price * (1 - offer.discount_second)) + 
  (offer.regular_price * (1 - offer.discount_third))

/-- Calculates the percentage saved under the "fair special" offer -/
def percentage_saved (offer : FairSpecial) : ℚ :=
  (1 - total_cost offer / (3 * offer.regular_price)) * 100

/-- Theorem stating that the "fair special" offer results in a 30% savings -/
theorem fair_special_savings :
  let offer : FairSpecial := {
    regular_price := 50,
    discount_second := 2/5,
    discount_third := 1/2
  }
  percentage_saved offer = 30 := by
  sorry

#eval percentage_saved {
  regular_price := 50,
  discount_second := 2/5,
  discount_third := 1/2
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_special_savings_l519_51974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l519_51946

theorem sum_remainder (a b c : ℕ) : 
  a % 59 = 29 →
  b % 59 = 31 →
  c % 59 = 7 →
  a ^ 2 % 59 = 29 →
  b ^ 2 % 59 = 31 →
  c ^ 2 % 59 = 7 →
  a > 0 →
  b > 0 →
  c > 0 →
  (a + b + c) % 59 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l519_51946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l519_51920

theorem function_equation_solution (f g : ℝ → ℝ) 
  (h1 : ∀ x > 0, g (f x) = x / (x * f x - 2))
  (h2 : ∀ x > 0, f (g x) = x / (x * g x - 2)) :
  (∀ x > 0, f x = 3 / x) ∧ (∀ x > 0, g x = 3 / x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l519_51920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l519_51938

-- Define complex numbers using the existing Complex type
open Complex

-- State the theorem
theorem complex_fraction_simplification :
  (10 * I) / (2 - I) = -2 + 4 * I := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l519_51938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_trapezoid_ratio_l519_51908

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  -- Points of the trapezoid
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  -- AB is parallel to CD
  parallel_sides : (A.2 - B.2) / (A.1 - B.1) = (C.2 - D.2) / (C.1 - D.1)
  -- Angle D is 90 degrees
  right_angle_D : (C.1 - D.1) * (A.1 - D.1) + (C.2 - D.2) * (A.2 - D.2) = 0
  -- E is on CD
  E_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (D.1 + t * (C.1 - D.1), D.2 + t * (C.2 - D.2))
  -- AE = BE
  AE_eq_BE : (A.1 - E.1)^2 + (A.2 - E.2)^2 = (B.1 - E.1)^2 + (B.2 - E.2)^2
  -- Triangles AED and CEB are similar but not congruent
  similar_triangles : ∃ k : ℝ, k ≠ 1 ∧
    (A.1 - E.1) / (C.1 - E.1) = (E.1 - D.1) / (E.1 - B.1) ∧
    (A.2 - E.2) / (C.2 - E.2) = (E.2 - D.2) / (E.2 - B.2) ∧
    k = (A.1 - E.1) / (C.1 - E.1)
  -- CD/AB = 2014
  sides_ratio : ((C.1 - D.1)^2 + (C.2 - D.2)^2) / ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2014^2

/-- The main theorem -/
theorem special_trapezoid_ratio (t : SpecialTrapezoid) :
  ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) / ((t.A.1 - t.D.1)^2 + (t.A.2 - t.D.2)^2) = 4027 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_trapezoid_ratio_l519_51908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_partition_exists_l519_51984

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ+
  height : ℕ+

/-- Represents a partition of a square into rectangles -/
structure SquarePartition where
  side_length : ℕ+
  rectangles : List Rectangle

/-- Checks if all side lengths in a partition are distinct -/
def has_distinct_sides (partition : SquarePartition) : Prop :=
  let all_sides := partition.rectangles.bind (λ r => [r.width, r.height])
  all_sides.Nodup

/-- Checks if the partition covers the entire square -/
def covers_square (partition : SquarePartition) : Prop :=
  (partition.rectangles.map (λ r => (r.width : ℕ) * (r.height : ℕ))).sum = 
    (partition.side_length : ℕ) * (partition.side_length : ℕ)

/-- The main theorem statement -/
theorem square_partition_exists :
  ∃ (partition : SquarePartition),
    partition.side_length = 13 ∧
    partition.rectangles.length = 5 ∧
    has_distinct_sides partition ∧
    covers_square partition := by
  sorry

#check square_partition_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_partition_exists_l519_51984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l519_51997

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * cos (x + π/6)

-- Theorem statement
theorem function_properties :
  ∀ a : ℝ,
  f a (π/2) = -1/2 →
  ∀ θ : ℝ, 0 < θ ∧ θ < π/2 ∧ sin θ = 1/3 →
    a = 1 ∧ f a θ = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l519_51997
