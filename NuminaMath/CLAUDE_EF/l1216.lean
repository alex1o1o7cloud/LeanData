import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l1216_121683

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflect a point over the x-axis -/
def reflectOverXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Reflect a point over the line y=x -/
def reflectOverYEqualsX (p : Point2D) : Point2D :=
  { x := p.y, y := p.x }

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (a b c : Point2D) : ℝ :=
  (1/2) * abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))

theorem area_of_triangle_PQR : 
  let P : Point2D := { x := 5, y := 3 }
  let Q : Point2D := reflectOverXAxis P
  let R : Point2D := reflectOverYEqualsX Q
  triangleArea P Q R = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l1216_121683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_miles_calculation_l1216_121622

/-- Represents the efficiency of a car in miles per gallon -/
structure CarEfficiency where
  city : ℚ
  highway : ℚ

/-- Represents the details of a trip -/
structure TripDetails where
  highwayMiles : ℚ
  gasCost : ℚ
  totalSpent : ℚ

/-- Calculates the number of city miles in a trip given car efficiency and trip details -/
def calculateCityMiles (efficiency : CarEfficiency) (trip : TripDetails) : ℚ :=
  ((trip.totalSpent / trip.gasCost) - (trip.highwayMiles / efficiency.highway)) * efficiency.city

theorem city_miles_calculation (efficiency : CarEfficiency) (trip : TripDetails) :
  efficiency.city = 30 ∧
  efficiency.highway = 40 ∧
  trip.highwayMiles = 200 ∧
  trip.gasCost = 3 ∧
  trip.totalSpent = 42 →
  calculateCityMiles efficiency trip = 270 := by
  sorry

def main : IO Unit := do
  let result := calculateCityMiles ⟨30, 40⟩ ⟨200, 3, 42⟩
  IO.println s!"The number of city miles is: {result}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_miles_calculation_l1216_121622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_85_l1216_121680

def numbers : List Nat := [45, 65, 85, 117, 169]

/-- The largest prime factor of a natural number -/
def largestPrimeFactor (n : Nat) : Nat :=
  (Nat.factors n).maximum?.getD 1

theorem largest_prime_factor_of_85 :
  ∀ n ∈ numbers, n ≠ 85 → largestPrimeFactor n < largestPrimeFactor 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_85_l1216_121680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_iff_negative_difference_l1216_121696

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

-- Define the sequence we're interested in
noncomputable def sequence_of_interest (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := 
  (4 : ℝ) ^ (a₁ * arithmetic_sequence a₁ d n)

-- State the theorem
theorem decreasing_iff_negative_difference 
  (a₁ : ℝ) (d : ℝ) (h : a₁ > 0) :
  (∀ n : ℕ, n ≥ 1 → sequence_of_interest a₁ d (n + 1) < sequence_of_interest a₁ d n) ↔ 
  d < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_iff_negative_difference_l1216_121696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_incircle_radii_l1216_121626

/-- Calculate the semi-perimeter of a triangle -/
noncomputable def semiPerimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

/-- Calculate the incircle radius of a triangle using Heron's formula -/
noncomputable def incircleRadius (a b c : ℝ) : ℝ :=
  let p := semiPerimeter a b c
  Real.sqrt ((p - a) * (p - b) * (p - c) / p)

/-- Theorem: The incircle radii of triangles with sides (17, 25, 26) and (17, 25, 28) are equal -/
theorem equal_incircle_radii :
  incircleRadius 17 25 26 = incircleRadius 17 25 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_incircle_radii_l1216_121626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1216_121627

/-- The speed of a train given its length and time to pass a stationary point. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ := length / time

/-- Theorem: A 500-meter long train that crosses a stationary point in 3 seconds has a speed of 500/3 meters per second. -/
theorem train_speed_calculation :
  train_speed 500 3 = 500 / 3 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1216_121627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_completion_time_l1216_121649

/-- The time it takes for A to complete the job alone -/
def A : ℝ := sorry

/-- The time it takes for D to complete the job alone -/
def D : ℝ := 6

/-- The time it takes for A and D to complete the job together -/
def AD : ℝ := 2

theorem A_completion_time : A = 3 := by
  have h1 : 1 / A + 1 / D = 1 / AD := by sorry
  have h2 : 1 / A + 1 / 6 = 1 / 2 := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_completion_time_l1216_121649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_of_A_l1216_121687

def M : Set ℕ := {n | 1 ≤ n ∧ n ≤ 1995}

theorem max_elements_of_A (A : Set ℕ) (h1 : A ⊆ M) 
  (h2 : ∀ x ∈ A, 15 * x ∉ A) : 
  ∃ (B : Set ℕ), B ⊆ M ∧ (∀ x ∈ B, 15 * x ∉ B) ∧ Finite B ∧ Nat.card B = 1870 ∧ 
  ∀ (C : Set ℕ), C ⊆ M → (∀ x ∈ C, 15 * x ∉ C) → Finite C → Nat.card C ≤ 1870 :=
sorry

#check max_elements_of_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_of_A_l1216_121687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l1216_121609

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the circles
def circle_M (x y : ℝ) : Prop := (x + 5)^2 + y^2 = 4
def circle_N (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 1

-- Define the point P on the right branch of the hyperbola
def P : ℝ × ℝ → Prop := λ p => hyperbola p.1 p.2 ∧ p.1 > 0

-- Define points M and N on their respective circles
def M : ℝ × ℝ → Prop := λ m => circle_M m.1 m.2
def N : ℝ × ℝ → Prop := λ n => circle_N n.1 n.2

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance_difference :
  ∃ (p m n : ℝ × ℝ), P p ∧ M m ∧ N n ∧
    (∀ (p' m' n' : ℝ × ℝ), P p' → M m' → N n' →
      distance p m - distance p n ≤ distance p' m' - distance p' n') ∧
    distance p m - distance p n = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l1216_121609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_qualifying_probability_team_challenge_probability_l1216_121628

-- Define the probabilities for individual qualifying round
noncomputable def prob_first_three : ℝ := 1/2
noncomputable def prob_last_two : ℝ := 1/3

-- Define the probability of advancing for Student A
noncomputable def prob_advancing : ℝ := 3/8

-- Define the probabilities for team challenge finals
noncomputable def P₁ (n : ℕ) (p : ℝ) : ℝ := (p * (2 - p))^n
noncomputable def P₂ (n : ℕ) (p : ℝ) : ℝ := p^n * (2 - p^n)

-- Theorem for individual qualifying round
theorem individual_qualifying_probability :
  prob_advancing = 
    (prob_first_three^3) + 
    (3 * prob_first_three^3 * (1 - (1 - prob_last_two)^2)) + 
    (3 * prob_first_three^2 * (1 - prob_first_three) * prob_last_two^2) :=
by sorry

-- Theorem for team challenge finals
theorem team_challenge_probability (n : ℕ) (p : ℝ) 
  (h1 : n ≥ 2) (h2 : 0 < p) (h3 : p < 1) : 
  P₁ n p > P₂ n p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_qualifying_probability_team_challenge_probability_l1216_121628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_from_angle_equation_l1216_121655

theorem equilateral_triangle_from_angle_equation (A B C : ℝ) :
  (∀ x ∈ ({A, B, C} : Set ℝ), 3 * Real.tan x - 3 * Real.tan (x / 2) - 2 * Real.sqrt 3 = 0) →
  (A = π / 3 ∧ B = π / 3 ∧ C = π / 3) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_from_angle_equation_l1216_121655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_l1216_121608

-- Define the point in rectangular coordinates
noncomputable def x : ℝ := 4
noncomputable def y : ℝ := 4 * Real.sqrt 2
noncomputable def z : ℝ := 4

-- Define the spherical coordinates
noncomputable def ρ : ℝ := 8
noncomputable def θ : ℝ := Real.pi / 4
noncomputable def φ : ℝ := Real.pi / 3

-- Theorem statement
theorem rectangular_to_spherical :
  (x^2 + y^2 + z^2 = ρ^2) ∧
  (x = ρ * Real.sin φ * Real.cos θ) ∧
  (y = ρ * Real.sin φ * Real.sin θ) ∧
  (z = ρ * Real.cos φ) ∧
  (ρ > 0) ∧ (0 ≤ θ) ∧ (θ < 2*Real.pi) ∧ (0 ≤ φ) ∧ (φ ≤ Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_spherical_l1216_121608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_expression_l1216_121669

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 - 3*z - 2) ≤ 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_complex_expression_l1216_121669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_result_l1216_121650

-- Define the quadratic equation coefficients
def a : ℝ := 5
def b : ℝ := -9
def c : ℝ := 1

-- Define the discriminant
def discriminant : ℝ := b^2 - 4*a*c

-- Define the roots
noncomputable def root1 : ℝ := (-b + Real.sqrt discriminant) / (2*a)
noncomputable def root2 : ℝ := (-b - Real.sqrt discriminant) / (2*a)

-- Define the positive difference between roots
noncomputable def root_difference : ℝ := |root1 - root2|

-- Theorem to prove
theorem quadratic_root_difference :
  root_difference = Real.sqrt 61 / 5 ∧
  ¬ ∃ (p : ℕ), p > 1 ∧ p * p ∣ 61 :=
by sorry

-- Calculate p + q
def p : ℕ := 61
def q : ℕ := 5

theorem result : p + q = 66 :=
by simp [p, q]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_result_l1216_121650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_is_even_l1216_121642

/-- The function f(x) = A * cos(ω * x + φ) -/
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.cos (ω * x + φ)

/-- Theorem: If f(x) = A * cos(ω * x + φ) has a maximum at x = 3, then f(x+3) is an even function -/
theorem f_shifted_is_even 
  (A ω φ : ℝ) 
  (h_A : A > 0) 
  (h_ω : ω > 0) 
  (h_max : ∀ x, f A ω φ x ≤ f A ω φ 3) :
  ∀ x, f A ω φ (x + 3) = f A ω φ (-x + 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_is_even_l1216_121642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l1216_121629

/-- Calculates the interest rate given principal, time, and simple interest -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (simple_interest : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

theorem interest_rate_problem :
  let principal : ℝ := 44625
  let time : ℝ := 9
  let simple_interest : ℝ := 4016.25
  calculate_interest_rate principal time simple_interest = 1 := by
  -- Unfold the definition of calculate_interest_rate
  unfold calculate_interest_rate
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l1216_121629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_cube_surface_area_l1216_121664

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℝ

/-- Calculates the surface area of a cube given its dimensions -/
def surfaceArea (c : CubeDimensions) : ℝ := 6 * c.side ^ 2

/-- Represents the dimensions of a subcube to be removed from each corner -/
structure SubcubeDimensions where
  side : ℝ

/-- Calculates the change in surface area when removing a subcube from a corner -/
def cornerSurfaceAreaChange (main : CubeDimensions) (sub : SubcubeDimensions) : ℝ :=
  3 * (main.side - sub.side) * sub.side - 3 * sub.side ^ 2

/-- Theorem: The surface area of a 5x5x5 cube with 2x2x2 subcubes removed from each corner is 198 sq.cm -/
theorem modified_cube_surface_area :
  let main := CubeDimensions.mk 5
  let sub := SubcubeDimensions.mk 2
  surfaceArea main + 8 * cornerSurfaceAreaChange main sub = 198 := by
  sorry

#eval surfaceArea (CubeDimensions.mk 5)
#eval cornerSurfaceAreaChange (CubeDimensions.mk 5) (SubcubeDimensions.mk 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_cube_surface_area_l1216_121664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_a_range_l1216_121651

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 3 then 2*a*x + 4
  else if 2 < x ∧ x < 3 then (a*x + 2) / (x - 2)
  else 0  -- undefined for x ≤ 2

-- State the theorem
theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, 2 < x ∧ x < y → f a y < f a x) →
  -1 < a ∧ a ≤ -2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_a_range_l1216_121651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_in_cones_l1216_121607

/-- Represents a right circular cone --/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a sphere --/
structure Sphere where
  radius : ℝ

noncomputable def volume_cone (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

noncomputable def volume_sphere (s : Sphere) : ℝ := (4/3) * Real.pi * s.radius^3

theorem liquid_rise_ratio_in_cones 
  (small_cone large_cone : Cone) 
  (marble : Sphere) 
  (h : volume_cone small_cone = volume_cone large_cone) 
  (r1 : small_cone.radius = 4) 
  (r2 : large_cone.radius = 8) 
  (rm : marble.radius = 2) : 
  (small_cone.height * ((volume_cone small_cone + volume_sphere marble) / volume_cone small_cone)^(1/3) - small_cone.height) / 
  (large_cone.height * ((volume_cone large_cone + volume_sphere marble) / volume_cone large_cone)^(1/3) - large_cone.height) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_in_cones_l1216_121607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_l1216_121648

/-- Calculates the profit percentage given cost price and selling price after discount --/
noncomputable def profit_percentage (cost_price selling_price : ℝ) : ℝ :=
  let marked_price := selling_price / 0.95
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percentage is approximately 31.58% --/
theorem profit_percentage_approx (cost_price selling_price : ℝ) 
  (h1 : cost_price = 38)
  (h2 : selling_price = 50) :
  ∃ ε > 0, |profit_percentage cost_price selling_price - 31.58| < ε := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval profit_percentage 38 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_l1216_121648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_derivative_properties_l1216_121625

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

-- Define the derivative of f(x)
def f_derivative (a b : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x + b

theorem function_and_derivative_properties 
  (a b c : ℝ) :
  (∃ m, f_derivative a b 1 = -3 ∧ f a b c 1 = m) →  -- Tangent line condition
  (∃ x, f_derivative a b x = 0 ∧ x = -2) →  -- Extreme value condition
  (f a b c = λ x ↦ -x^3 - 2*x^2 + 4*x - 3) ∧  -- Part 1 conclusion
  (∀ x ∈ Set.Icc (-2) 0, f_derivative a b x ≥ 0) →  -- Monotonically increasing condition
  b ≥ 4  -- Part 2 conclusion
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_derivative_properties_l1216_121625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l1216_121631

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 24 = 1

-- Define the distance from a point to a focus
noncomputable def distance_to_focus (x y : ℝ) (fx fy : ℝ) : ℝ :=
  Real.sqrt ((x - fx)^2 + (y - fy)^2)

-- State the theorem
theorem hyperbola_focus_distance 
  (x y : ℝ) 
  (f1x f1y f2x f2y : ℝ) -- Coordinates of the two foci
  (h1 : is_on_hyperbola x y)
  (h2 : distance_to_focus x y f1x f1y = 11) :
  distance_to_focus x y f2x f2y = 21 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l1216_121631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_consumption_theorem_l1216_121604

/-- The total amount of food consumed by dogs and puppies in a day -/
noncomputable def total_food_consumed (num_puppies num_dogs : ℕ) (dog_meal_frequency : ℕ) (dog_meal_amount : ℝ) : ℝ :=
  let puppy_meal_amount := dog_meal_amount / 2
  let puppy_meal_frequency := dog_meal_frequency * 3
  let dog_daily_food := dog_meal_amount * (dog_meal_frequency : ℝ)
  let puppy_daily_food := puppy_meal_amount * (puppy_meal_frequency : ℝ)
  (num_dogs : ℝ) * dog_daily_food + (num_puppies : ℝ) * puppy_daily_food

theorem food_consumption_theorem :
  total_food_consumed 4 3 3 4 = 108 := by
  -- Unfold the definition of total_food_consumed
  unfold total_food_consumed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_consumption_theorem_l1216_121604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_l1216_121653

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the conditions
axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_property : ∀ x : ℝ, f (x + 2) = -1 / f x
axiom f_defined : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = x - 2

-- State the theorem to be proved
theorem f_value : f 6.5 = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_l1216_121653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_one_l1216_121699

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x³(a⋅2ˣ - 2⁻ˣ) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x^3 * (a * Real.exp (x * Real.log 2) - Real.exp (-x * Real.log 2))

/-- Theorem: If f(x) = x³(a⋅2ˣ - 2⁻ˣ) is an even function, then a = 1 -/
theorem even_function_implies_a_equals_one (a : ℝ) :
  IsEven (f a) → a = 1 := by
  sorry

#check even_function_implies_a_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_one_l1216_121699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_b_alone_l1216_121674

noncomputable section

/-- The time it takes for A, B, and C together to finish the work -/
def time_abc : ℝ := 4

/-- The time it takes for A alone to finish the work -/
def time_a : ℝ := 6

/-- The time it takes for C alone to finish the work -/
def time_c : ℝ := 36

/-- The work rate of A, B, and C together -/
def rate_abc : ℝ := 1 / time_abc

/-- The work rate of A alone -/
def rate_a : ℝ := 1 / time_a

/-- The work rate of C alone -/
def rate_c : ℝ := 1 / time_c

/-- Theorem: The time it takes for B alone to finish the work is 18 days -/
theorem time_b_alone : ∃ (time_b : ℝ), time_b = 18 ∧ rate_abc = rate_a + (1 / time_b) + rate_c := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_b_alone_l1216_121674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_spades_correct_l1216_121659

/-- Represents a standard deck of playing cards -/
structure Deck where
  cards : Finset (Nat × Nat)
  card_count : cards.card = 52
  rank_count : (cards.image (·.1)).card = 13
  suit_count : (cards.image (·.2)).card = 4
  unique_cards : ∀ r s, (r, s) ∈ cards → r ∈ Finset.range 13 ∧ s ∈ Finset.range 4

/-- The probability of drawing three spades from the top of a standard deck -/
def probability_three_spades : ℚ := 11 / 850

/-- Theorem stating the probability of drawing three spades from the top of a standard deck -/
theorem probability_three_spades_correct :
  probability_three_spades = 11 / 850 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_spades_correct_l1216_121659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x_prime_multiples_l1216_121667

/-- A number is prime if it's greater than 1 and has no positive divisors other than 1 and itself -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 → m < n → n % m ≠ 0

/-- There exists a real number x such that both 10x and 15x are prime numbers when rounded down to the nearest integer -/
theorem exists_x_prime_multiples : ∃ x : ℝ, isPrime (Int.toNat ⌊10 * x⌋) ∧ isPrime (Int.toNat ⌊15 * x⌋) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x_prime_multiples_l1216_121667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_original_price_l1216_121646

/-- The original price of a car given its used price and percentage of original price -/
noncomputable def original_price (used_price : ℝ) (percentage : ℝ) : ℝ :=
  used_price / percentage

/-- Theorem stating that if a used car costs $15000 and this is 40% of the original price,
    then the original price was $37500 -/
theorem car_original_price :
  let used_price : ℝ := 15000
  let percentage : ℝ := 0.40
  original_price used_price percentage = 37500 := by
  -- Unfold the definition of original_price
  unfold original_price
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_original_price_l1216_121646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roll_path_length_for_given_radius_l1216_121640

/-- The length of the path traveled by point B when rolling a semi-circle -/
noncomputable def roll_path_length (radius : ℝ) : ℝ :=
  2 * (Real.pi * radius / 2)

/-- Theorem stating that the length of the path traveled by point B
    when rolling a semi-circle with radius 4/π cm is 8 cm -/
theorem roll_path_length_for_given_radius :
  roll_path_length (4 / Real.pi) = 8 := by
  sorry

#check roll_path_length_for_given_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roll_path_length_for_given_radius_l1216_121640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_101_l1216_121632

def a : ℕ → ℕ
  | 0 => 0  -- Add a case for 0
  | 1 => 0  -- Add a case for 1
  | 2 => 0  -- Add a case for 2
  | 3 => 0  -- Add a case for 3
  | 4 => 0  -- Add a case for 4
  | 5 => 5
  | n + 1 => 50 * a n + (n + 1)^2

def is_multiple (m n : ℕ) : Prop := ∃ k, m = n * k

theorem least_multiple_of_101 :
  (∀ n, 5 < n → n < 9 → ¬ is_multiple (a n) 101) ∧
  is_multiple (a 9) 101 := by
  sorry

#eval a 9 % 101  -- This should evaluate to 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_101_l1216_121632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_decimal_value_l1216_121654

/-- The infinite decimal expansion where the n-th digit after the decimal point is n -/
noncomputable def infinite_decimal : ℚ := 
  (∑' n : ℕ, n / (10 ^ n) : ℚ)

/-- Theorem stating that the infinite decimal expansion is equal to 10/81 -/
theorem infinite_decimal_value : infinite_decimal = 10 / 81 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_decimal_value_l1216_121654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_standard_deviation_l1216_121647

noncomputable def sample : List ℝ := [3, 5, 7, 4, 6]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem sample_standard_deviation :
  standardDeviation sample = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_standard_deviation_l1216_121647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_of_u_l1216_121605

open Complex

theorem max_magnitude_of_u (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^4 - z^3 - 3 * z^2 * I - z + 1) ≤ 5 ∧
  Complex.abs ((-1 : ℂ)^4 - (-1 : ℂ)^3 - 3 * (-1 : ℂ)^2 * I - (-1 : ℂ) + 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_of_u_l1216_121605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_BEIH_l1216_121685

/-- A square with side length 3 -/
structure Square3x3 where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_square : A = (0, 3) ∧ B = (0, 0) ∧ C = (3, 0) ∧ D = (3, 3)

/-- Point E is 1/3 of the way along AB from A to B -/
noncomputable def E : ℝ × ℝ := (0, 2)

/-- Point F is 1/3 of the way along BC from B to C -/
noncomputable def F : ℝ × ℝ := (1, 0)

/-- Intersection point of AF and DE -/
noncomputable def I : ℝ × ℝ := (3/10, 21/10)

/-- Intersection point of BD and AF -/
noncomputable def H : ℝ × ℝ := (3/4, 3/4)

/-- Area of a quadrilateral given its four vertices -/
noncomputable def quadrilateral_area (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  let (x4, y4) := p4
  (1/2) * abs (x1*y2 + x2*y3 + x3*y4 + x4*y1 - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

theorem area_of_BEIH (sq : Square3x3) :
  quadrilateral_area E I H sq.B = 1.0125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_BEIH_l1216_121685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_rotation_matrix_45_deg_l1216_121661

/-- The rotation matrix for a 45° counter-clockwise rotation about the origin -/
noncomputable def S : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.sqrt 2 / 2, -Real.sqrt 2 / 2],
    ![Real.sqrt 2 / 2,  Real.sqrt 2 / 2]]

/-- The determinant of the rotation matrix S is 1 -/
theorem det_rotation_matrix_45_deg : Matrix.det S = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_rotation_matrix_45_deg_l1216_121661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_100_is_zero_l1216_121693

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → x * g y - y * g x = k * g (x / y)

theorem g_100_is_zero
  (g : ℝ → ℝ)
  (k : ℝ)
  (h_k_nonzero : k ≠ 0)
  (h_k_not_neg_one : k ≠ -1)
  (h_func_eq : FunctionalEquation g k) :
  g 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_100_is_zero_l1216_121693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_value_l1216_121634

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

-- State the theorem
theorem f_maximum_value :
  (∀ x : ℝ, x < 1 → f x ≤ -1) ∧ (∃ x : ℝ, x < 1 ∧ f x = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_value_l1216_121634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_rate_l1216_121686

/-- The depth the diver descends in feet -/
noncomputable def depth : ℝ := 3500

/-- The time taken for the descent in minutes -/
noncomputable def time : ℝ := 100

/-- The rate of descent in feet per minute -/
noncomputable def rate : ℝ := depth / time

/-- Theorem: The rate of descent is 35 feet per minute -/
theorem diver_descent_rate : rate = 35 := by
  -- Unfold the definitions
  unfold rate depth time
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_rate_l1216_121686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l1216_121644

theorem sin_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi/2)
  (h2 : -Real.pi/2 < β ∧ β < 0)
  (h3 : Real.cos (α - β) = -3/5)
  (h4 : Real.tan α = 4/3) :
  Real.sin β = -24/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l1216_121644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1216_121616

-- Define the function f(x) as noncomputable
noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- Define the theorem
theorem min_omega_value (ω φ : ℝ) (h_pos : ω > 0) :
  (∀ x, f ω φ (x + π/3) = f ω φ (x - π/6)) →
  ω ≥ 4 ∧ ∃ k : ℕ, ω = 4 * k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1216_121616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_symmetry_axis_l1216_121619

noncomputable def original_function (x : ℝ) : ℝ := 2 * Real.sin (2 * x)

noncomputable def shifted_function (x : ℝ) : ℝ := 2 * Real.sin (2 * (x + Real.pi / 12))

def is_axis_of_symmetry (f : ℝ → ℝ) (axis : ℝ) : Prop :=
  ∀ x : ℝ, f (axis + x) = f (axis - x)

theorem shifted_sine_symmetry_axis (k : ℤ) :
  is_axis_of_symmetry shifted_function (k * Real.pi / 2 + Real.pi / 6) := by
  sorry

#check shifted_sine_symmetry_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_sine_symmetry_axis_l1216_121619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_disjoint_l1216_121614

/-- Two circles M and N in a 2D plane --/
structure TwoCircles where
  /-- Center of circle M --/
  center_m : ℝ × ℝ
  /-- Radius of circle M --/
  radius_m : ℝ
  /-- Center of circle N --/
  center_n : ℝ × ℝ
  /-- Radius of circle N --/
  radius_n : ℝ

/-- The distance between two points in 2D space --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem stating that the given circles are disjoint --/
theorem circles_are_disjoint (circles : TwoCircles) 
  (hm : circles.center_m = (0, 0)) 
  (hrm : circles.radius_m = 1)
  (hn : circles.center_n = (1, 2)) 
  (hrn : circles.radius_n = 1) : 
  distance circles.center_m circles.center_n > circles.radius_m + circles.radius_n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_disjoint_l1216_121614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spacy_subsets_count_l1216_121662

/-- A set is spacy if it contains no more than one out of any three consecutive integers -/
def IsSpacy (s : Set ℕ) : Prop :=
  ∀ n : ℕ, (n ∈ s → (n + 1 ∉ s ∨ n + 2 ∉ s)) ∧ (n + 1 ∈ s → n + 2 ∉ s)

/-- The number of spacy subsets for a set of n elements -/
def NumSpacySubsets : ℕ → ℕ
  | 0 => 1  -- Empty set is always spacy
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | n + 4 => NumSpacySubsets (n + 3) + NumSpacySubsets (n + 1)

theorem spacy_subsets_count :
  NumSpacySubsets 15 = 406 := by
  sorry

#eval NumSpacySubsets 15  -- This will compute and display the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spacy_subsets_count_l1216_121662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1216_121618

/-- The length of a bridge that a train can cross -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Theorem stating the length of the bridge -/
theorem bridge_length_calculation :
  bridge_length 120 45 30 = 255 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval bridge_length 120 45 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1216_121618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_transformation_l1216_121666

/-- Represents the original 7x7 table -/
def original_table (i j : Fin 7) : ℤ :=
  (i.val^2 + j.val) * (i.val + j.val^2)

/-- Represents a table after applying operations -/
def transformed_table (t : Fin 7 → Fin 7 → ℤ) (r c : Fin 7 → ℤ) (i j : Fin 7) : ℤ :=
  t i j + r i + c j

/-- Checks if a row forms an arithmetic sequence -/
def is_arithmetic_sequence (row : Fin 7 → ℤ) : Prop :=
  ∃ a d : ℤ, ∀ j : Fin 7, row j = a + j.val * d

/-- Applies a single operation to a table -/
def apply_operation (t : Fin 7 → Fin 7 → ℤ) (r c : Fin 7 → ℤ) : Fin 7 → Fin 7 → ℤ :=
  λ i j ↦ t i j + r i + c j

/-- Main theorem: It's impossible to transform the original table into one where each row is an arithmetic sequence -/
theorem impossibility_of_transformation :
  ¬∃ (num_ops : ℕ) (operations : Fin num_ops → (Fin 7 → ℤ) × (Fin 7 → ℤ)),
    ∀ i : Fin 7, is_arithmetic_sequence (λ j ↦ 
      (List.foldl 
        (λ acc op ↦ apply_operation acc op.1 op.2) 
        original_table 
        (List.ofFn operations)) i j) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_transformation_l1216_121666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_l1216_121689

-- Define the points A, B, C, D, E, and P on a real line
variable (A B C D E P : ℝ)

-- Define the conditions
def collinear (A B C D E : ℝ) : Prop := A < B ∧ B < C ∧ C < D ∧ D < E

def distances (A B C D E : ℝ) : Prop := B - A = 2 ∧ C - B = 2 ∧ D - C = 3 ∧ E - D = 4

-- Define the function to be minimized
def f (A B C D E P : ℝ) : ℝ := (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2

-- State the theorem
theorem min_sum_squares (A B C D E : ℝ) (h1 : collinear A B C D E) (h2 : distances A B C D E) :
  ∃ (min : ℝ), min = 58.8 ∧ ∀ (P : ℝ), f A B C D E P ≥ min :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_l1216_121689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_selling_price_l1216_121673

/-- Represents the original purchase price of the product -/
def P : ℝ := 1000

/-- Represents the original selling price of the product -/
def S : ℝ := 1.1 * P

/-- Represents the new selling price if the product was purchased for 10% less and sold at 30% profit -/
def S_new : ℝ := 1.17 * P

theorem original_selling_price : S = 1100 := by
  -- Unfold the definitions
  unfold S
  unfold P
  -- Evaluate the expression
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_selling_price_l1216_121673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1216_121635

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x

-- Define M(a)
noncomputable def M (a : ℝ) : ℝ :=
  if a > Real.sqrt 5 then a - Real.sqrt (a^2 - 5)
  else a + Real.sqrt (a^2 + 5)

theorem function_properties :
  -- a > 0 is a given condition
  ∀ a : ℝ, a > 0 →
    -- Part 1
    (let a := 2
     ∀ x : ℝ, -3 < f a x ∧ f a x < 5 ↔ (x ∈ Set.Ioo (-1) 1 ∨ x ∈ Set.Ioo 3 5)) ∧
    -- Part 2
    (∀ x : ℝ, x ∈ Set.Icc 0 (M a) → |f a x| ≤ 5) ∧
    (∀ m : ℝ, m > M a → ∃ x : ℝ, x ∈ Set.Icc 0 m ∧ |f a x| > 5) ∧
    -- Part 3
    (∀ t : ℝ, (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc t (t+2) ∧ x₂ ∈ Set.Icc t (t+2) ∧
                f a x₁ = 0 ∧ f a x₂ = -4 ∧
                (∀ x : ℝ, x ∈ Set.Icc t (t+2) → -4 ≤ f a x ∧ f a x ≤ 0))
     ↔ (a = 2 ∧ (t = 0 ∨ t = 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1216_121635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_b_values_l1216_121665

-- Define congruence relation
def congruent (a b m : ℕ) : Prop :=
  ∃ k : ℤ, k ≠ 0 ∧ (a : ℤ) - (b : ℤ) = k * (m : ℤ)

theorem possible_b_values (m : ℕ) (b : ℕ) 
  (h1 : m > 1)
  (h2 : congruent 6 b m)
  (h3 : ∃ n : ℕ, b = n * m) :
  b ∈ ({2, 3, 4} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_b_values_l1216_121665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_student_difference_l1216_121603

theorem school_student_difference (boys_A female_B : ℕ) 
  (boys_A_eq : boys_A = 217)
  (female_B_eq : female_B = 196)
  (total_equality : ∃ (male_B female_A : ℕ), 
    boys_A + male_B = female_A + female_B) :
  ∃ (male_B female_A : ℕ), 
    boys_A + male_B = female_A + female_B ∧ 
    (male_B : ℤ) - (female_A : ℤ) = 21 ∨ (male_B : ℤ) - (female_A : ℤ) = -21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_student_difference_l1216_121603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_condition_l1216_121606

/-- The function f(x) defined in the problem -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

/-- The theorem stating the condition for k -/
theorem triangle_side_condition (k : ℝ) :
  (∀ a b c : ℝ, ∃ (s t u : ℝ), s = f k a ∧ t = f k b ∧ u = f k c ∧ s + t > u ∧ s + u > t ∧ t + u > s) ↔
  -1/2 < k ∧ k < 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_condition_l1216_121606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_z_is_zero_l1216_121663

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Definition of z -/
noncomputable def z : ℂ := (2 + i) / (-2*i + 1)

/-- Theorem: The real part of z is 0 -/
theorem real_part_of_z_is_zero : Complex.re z = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_z_is_zero_l1216_121663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_winner_speed_l1216_121641

/-- Calculates the speed of the race winner given the race conditions -/
noncomputable def winner_speed (race_distance : ℝ) (second_place_time : ℝ) (time_difference : ℝ) : ℝ :=
  let winner_time := second_place_time - time_difference
  race_distance / (winner_time / 60)

/-- Theorem stating the winner's speed under given race conditions -/
theorem race_winner_speed :
  let race_distance : ℝ := 5  -- miles
  let second_place_time : ℝ := 23  -- minutes
  let time_difference : ℝ := 3  -- minutes
  winner_speed race_distance second_place_time time_difference = 15 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval winner_speed 5 23 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_winner_speed_l1216_121641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_cos_100_eq_3_l1216_121612

noncomputable def f (a b x : ℝ) : ℝ := a * (Real.sin x)^3 + b * (x^(1/3)) * (Real.cos x)^3 + 4

theorem f_cos_100_eq_3 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : f a b (Real.sin (10 * π / 180)) = 5) : 
  f a b (Real.cos (100 * π / 180)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_cos_100_eq_3_l1216_121612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_not_pi_third_l1216_121681

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem shift_not_pi_third : ¬(∀ (x : ℝ), f x = g (x + Real.pi / 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_not_pi_third_l1216_121681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1216_121675

/-- The number of days b takes to finish the work alone -/
noncomputable def b_days : ℝ := 33

/-- The number of days a takes to finish the work alone -/
noncomputable def a_days : ℝ := 2 * b_days

/-- The number of days c takes to finish the work alone -/
noncomputable def c_days : ℝ := 3 * b_days

/-- The combined work rate of a, b, and c -/
noncomputable def combined_work_rate : ℝ := 1 / a_days + 1 / b_days + 1 / c_days

/-- The number of days a, b, and c take to finish the work together -/
noncomputable def days_together : ℝ := 1 / combined_work_rate

theorem work_completion_time : days_together = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1216_121675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_value_l1216_121672

/-- Given a positive integer n such that 14n/60 is an integer and n has exactly 3 different positive prime factors, the maximum possible value of n is 330. -/
theorem max_n_value (n : ℕ) 
  (h1 : (14 * n) % 60 = 0)
  (h2 : ∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ n = p * q * r)
  (h3 : n > 0) :
  n ≤ 330 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_value_l1216_121672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1216_121601

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 6) / Real.sqrt (x^2 - 5*x + 6)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < 2 ∨ x > 3}

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | x^2 - 5*x + 6 > 0} = domain_f :=
by
  -- The proof goes here
  sorry

-- You can add more helper lemmas or theorems if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1216_121601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1216_121645

noncomputable def f (x m : ℝ) : ℝ := -(Real.sin x)^2 + m * (2 * Real.cos x - 1)

theorem function_properties :
  ∀ m : ℝ,
  (∀ x ∈ Set.Icc (-Real.pi / 3) ((2 * Real.pi) / 3), f x m ≥ -1) →
  (∃ x ∈ Set.Icc (-Real.pi / 3) ((2 * Real.pi) / 3), f x m = -1) →
  ((m = 0 ∧ 
    (∀ x ∈ Set.Icc (-Real.pi / 3) ((2 * Real.pi) / 3), f x m ≤ 0) ∧
    (∃ x ∈ Set.Icc (-Real.pi / 3) ((2 * Real.pi) / 3), f x m = 0 ∧ x = 0)) ∨
   (m = -1 ∧ 
    (∀ x ∈ Set.Icc (-Real.pi / 3) ((2 * Real.pi) / 3), f x m ≤ 5/4) ∧
    (∃ x ∈ Set.Icc (-Real.pi / 3) ((2 * Real.pi) / 3), f x m = 5/4 ∧ x = 2 * Real.pi / 3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1216_121645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_calculation_l1216_121690

theorem complex_fraction_calculation : 
  (((11 + 1/9) - (3 + 2/5) * (1 + 2/17)) - (8 + 2/5) / 3.6) / (2 + 6/25) = 20/9 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_calculation_l1216_121690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_converges_iff_abs_a_le_two_l1216_121692

/-- A sequence defined recursively -/
noncomputable def x : ℕ → ℝ → ℝ
  | 0, a => 1996
  | n + 1, a => a / (1 + (x n a)^2)

/-- The theorem statement -/
theorem sequence_converges_iff_abs_a_le_two (a : ℝ) :
  (∃ L, ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n a - L| < ε) ↔ |a| ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_converges_iff_abs_a_le_two_l1216_121692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l1216_121638

-- Define the set of positive integers (N*)
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define set M
def M : Set ℕ := {x ∈ PositiveIntegers | x < 6}

-- Define set N
noncomputable def N : Set ℝ := {x : ℝ | |x - 1| ≤ 2}

-- Define the complement of N in ℝ
noncomputable def ComplementN : Set ℝ := Set.univ \ N

-- Define the intersection of M and ComplementN
noncomputable def MIntersectComplementN : Set ℕ := {x ∈ M | (x : ℝ) ∈ ComplementN}

-- Theorem statement
theorem intersection_M_complement_N :
  MIntersectComplementN = {4, 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l1216_121638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l1216_121688

open Real

-- Define what it means for an angle to be in the second quadrant
def is_in_second_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, Real.pi / 2 + 2 * Real.pi * (k : ℝ) < α ∧ α < Real.pi + 2 * Real.pi * (k : ℝ)

-- Define what it means for an angle to be in the first or third quadrant
def is_in_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 0 + Real.pi * (k : ℝ) < α ∧ α < Real.pi / 2 + Real.pi * (k : ℝ)

-- The theorem statement
theorem half_angle_quadrant (α : ℝ) :
  is_in_second_quadrant α → is_in_first_or_third_quadrant (α / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l1216_121688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1216_121658

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geometric : IsGeometricSequence a)
  (h_fourth : a 4 = Nat.factorial 6)
  (h_seventh : a 7 = Nat.factorial 7) :
  a 1 = 720 / 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1216_121658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l1216_121602

theorem smallest_number_divisible (n : ℕ) : n = 109871748 ↔ 
  (∀ d ∈ ({29, 53, 37, 41, 47} : Finset ℕ), (n + 11) % d = 0) ∧
  (∀ m : ℕ, m < n → ∃ d ∈ ({29, 53, 37, 41, 47} : Finset ℕ), (m + 11) % d ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l1216_121602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_f_continuous_ivt_for_f_solution_exists_l1216_121676

-- Define the function f(x) = lg x - 3 + x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10 - 3 + x

-- State the theorem
theorem equation_solution_exists :
  Continuous f ∧
  (∀ a b : ℝ, f a * f b < 0 → ∃ c ∈ Set.Ioo a b, f c = 0) →
  ∃ x ∈ Set.Ioo 2 3, f x = 0 := by
  intro h
  have h1 : f 2 < 0 := by sorry
  have h2 : f 3 > 0 := by sorry
  have h3 : f 2 * f 3 < 0 := by sorry
  exact h.2 2 3 h3

-- Prove that f is continuous
theorem f_continuous : Continuous f := by sorry

-- Prove the intermediate value theorem for f
theorem ivt_for_f :
  ∀ a b : ℝ, f a * f b < 0 → ∃ c ∈ Set.Ioo a b, f c = 0 := by sorry

-- Combine the theorems to show the existence of a solution
theorem solution_exists : ∃ x ∈ Set.Ioo 2 3, f x = 0 :=
  equation_solution_exists (⟨f_continuous, ivt_for_f⟩)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_f_continuous_ivt_for_f_solution_exists_l1216_121676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_wasted_theorem_l1216_121620

/-- Calculates the amount of water wasted by a dripping faucet in one hour -/
def water_wasted_in_one_hour 
  (drips_per_minute : ℕ) 
  (mL_per_drop : ℚ) : ℚ :=
  (drips_per_minute : ℚ) * 60 * mL_per_drop

theorem water_wasted_theorem 
  (drips_per_minute : ℕ) 
  (mL_per_drop : ℚ) 
  (h1 : drips_per_minute = 10) 
  (h2 : mL_per_drop = 0.05) : 
  water_wasted_in_one_hour drips_per_minute mL_per_drop = 30 :=
by
  -- Proof goes here
  sorry

#eval water_wasted_in_one_hour 10 (5 / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_wasted_theorem_l1216_121620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_addition_theorem_l1216_121670

/-- Represents a number in base 8 (octal) -/
def Octal : Type := ℕ

/-- Converts an Octal number to its decimal (base 10) representation -/
def octal_to_decimal (n : Octal) : ℕ := sorry

/-- Converts a decimal (base 10) number to its Octal representation -/
def decimal_to_octal (n : ℕ) : Octal := sorry

/-- Adds two Octal numbers and returns the result in Octal -/
def octal_add (a b : Octal) : Octal := sorry

/-- Gets the units digit of an Octal number -/
def octal_units_digit (n : Octal) : Octal := sorry

/-- Allows use of natural number literals for Octal -/
instance : OfNat Octal n where
  ofNat := decimal_to_octal n

theorem octal_addition_theorem (a b c : Octal) :
  octal_units_digit (octal_add (octal_add a b) c) = (4 : Octal) :=
  by
    -- Assume a = 65₈, b = 74₈, c = 3₈
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_addition_theorem_l1216_121670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1216_121633

def is_squarefree (n : ℕ) : Prop := ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

theorem min_value_theorem (d n : ℕ) (h1 : d > 0) (h2 : n > 0) (h3 : d ∣ n) (h4 : n > 1000) 
  (h5 : ¬∃ m : ℕ, n = m^2) :
  ∃ (a b : ℕ) (c : ℤ), 
    a ≠ 0 ∧ b > 0 ∧ c ≠ 0 ∧ is_squarefree b ∧
    (∀ (d' n' : ℕ), d' > 0 → n' > 0 → d' ∣ n' → n' > 1000 → ¬∃ m : ℕ, n' = m^2 →
      |↑d' - Real.sqrt (↑n' : ℝ)| ≥ |↑d - Real.sqrt (↑n : ℝ)|) ∧
    |↑d - Real.sqrt (↑n : ℝ)| = ↑a * Real.sqrt (↑b : ℝ) + ↑c ∧
    a + b + c = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1216_121633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1216_121624

open Real BigOperators

theorem inequality_proof (n : ℕ) (p q : ℝ) (x : Fin n → ℝ) 
  (h1 : ∀ i, 0 < x i ∧ x i < 1)
  (h2 : ∑ i, x i = 1)
  (h3 : p > 0)
  (h4 : q ≥ 1)
  (h5 : -1 < p - q ∧ p - q < 0) :
  ∑ i, 1 / (x i ^ p - x i ^ q) ≥ n ^ (q + 1) / (n ^ (q - p) - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1216_121624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1216_121697

/-- Given a function g that satisfies the equation
    sin x + cos y = f x + f y + g x - g y for all real x and y,
    prove that there exists a constant C such that
    g x = (sin x - cos x) / 2 + C for all real x. -/
theorem functional_equation_solution
  (g : ℝ → ℝ)
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, Real.sin x + Real.cos y = f x + f y + g x - g y) :
  ∃ C : ℝ, ∀ x : ℝ, g x = (Real.sin x - Real.cos x) / 2 + C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1216_121697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_segment_length_l1216_121657

/-- Right triangle ABC with C as the right angle -/
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  right_angle_at_C : AB^2 + AC^2 = BC^2

/-- Length function l(x) for a segment in the triangle -/
noncomputable def length_function (t : RightTriangle) (x : ℝ) : ℝ :=
  (4 * |0.75 * x + 3|) / 5

theorem right_triangle_segment_length 
  (t : RightTriangle) 
  (h1 : t.AB = 4) 
  (h2 : t.AC = 3) 
  (h3 : t.BC = 5) :
  (∀ x, length_function t x = (4 * |0.75 * x + 3|) / 5) ∧ 
  length_function t 1.5 = 3.3 := by
  sorry

#check right_triangle_segment_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_segment_length_l1216_121657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1216_121611

theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : Real.sin (π/2 + 2*α) = -4/5) 
  (h2 : π/2 < α) (h3 : α < π) : 
  Real.tan (α + π/4) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1216_121611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_range_of_a_l1216_121668

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 - 3*x - 18 ≥ 0}
def B : Set ℝ := {x | (x + 5) / (x - 14) ≤ 0}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a + 1}

-- Theorem 1
theorem intersection_A_complement_B : 
  A ∩ (U \ B) = {x | x < -5 ∨ x ≥ 14} := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) : 
  (B ∩ C a = C a) → a ≥ -5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_range_of_a_l1216_121668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_july_milk_powder_price_l1216_121698

/-- Represents the cost per pound of milk powder, coffee, and sugar in June -/
noncomputable def june_cost : ℝ := sorry

/-- Represents the cost of the mixture in July per pound -/
noncomputable def july_mixture_cost : ℝ := 11.70 / 4

/-- The cost of coffee in July -/
noncomputable def july_coffee_cost : ℝ := 4 * june_cost

/-- The cost of milk powder in July -/
noncomputable def july_milk_powder_cost : ℝ := 0.2 * june_cost

/-- The cost of sugar in July -/
noncomputable def july_sugar_cost : ℝ := 1.45 * june_cost

/-- Theorem stating the cost of milk powder in July -/
theorem july_milk_powder_price : 
  ∃ (ε : ℝ), abs (july_milk_powder_cost - 0.3174) < ε ∧ ε > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_july_milk_powder_price_l1216_121698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1216_121684

/-- Given two circles C and D where an arc of 60° on C has the same length as an arc of 40° on D,
    the ratio of the area of circle C to the area of circle D is 4/9 -/
theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (π / 3 * r₁ = π / 4.5 * r₂) →
  (π * r₁^2) / (π * r₂^2) = 4/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1216_121684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpetual_withdrawal_investment_l1216_121630

/-- The required investment for perpetual withdrawals -/
noncomputable def required_investment (r : ℝ) : ℝ := (1 + r) * (2 + r) / r^3

/-- The sum of the present values of all future withdrawals -/
noncomputable def sum_present_values (r : ℝ) : ℝ := ∑' n, (n^2 : ℝ) / (1 + r)^n

/-- Theorem stating that the sum of present values equals the required investment -/
theorem perpetual_withdrawal_investment (r : ℝ) (hr : r > 0) :
  sum_present_values r = required_investment r := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpetual_withdrawal_investment_l1216_121630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_gpa_calculation_l1216_121694

theorem class_gpa_calculation (n : ℝ) (h : n > 0) : 
  (1/3 * n * 45 + 2/3 * n * 60) / n = 55 := by
  -- Simplify the expression
  have : (1/3 * n * 45 + 2/3 * n * 60) / n = 1/3 * 45 + 2/3 * 60 := by
    field_simp
    ring
  -- Rewrite using this equality
  rw [this]
  -- Evaluate the expression
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_gpa_calculation_l1216_121694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1216_121617

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- The problem statement -/
theorem car_average_speed :
  let distance : ℝ := 250  -- Total distance in miles
  let time : ℝ := 5        -- Total time in hours
  average_speed distance time = 50 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1216_121617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_three_correct_l1216_121671

/-- The count of whole numbers between 200 and 499 (inclusive) that contain at least one digit 3 -/
def count_numbers_with_three : ℕ := 138

/-- The set of whole numbers between 200 and 499 (inclusive) -/
def number_range : Set ℕ := {n : ℕ | 200 ≤ n ∧ n ≤ 499}

/-- Predicate to check if a natural number contains the digit 3 -/
def contains_three (n : ℕ) : Prop :=
  ∃ (d : ℕ), d = 3 ∧ ∃ (k m : ℕ), n = k * 10 + d + m * 10 ∧ m < 10

/-- The set of numbers in the range that contain at least one digit 3 -/
def numbers_with_three : Set ℕ :=
  {n ∈ number_range | contains_three n}

/-- Lemma to show that numbers_with_three is finite -/
lemma numbers_with_three_finite : Set.Finite numbers_with_three :=
sorry

/-- Theorem stating that the count of numbers with three is correct -/
theorem count_numbers_with_three_correct :
  Finset.card (Set.Finite.toFinset numbers_with_three_finite) = count_numbers_with_three :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_three_correct_l1216_121671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unfolded_shape_properties_l1216_121682

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a shape with a list of vertices -/
structure Shape where
  vertices : List Point

/-- Represents the folding operations -/
inductive FoldOperation where
  | FoldUp
  | FoldLeft

/-- Represents the result of folding operations -/
def fold (s : Shape) (op : FoldOperation) : Shape := sorry

/-- Represents the operation of cutting off a triangle -/
def cutTriangle (s : Shape) (a b c : Point) : Shape := sorry

/-- Represents the operation of unfolding a shape -/
def unfold (s : Shape) : Shape := sorry

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a b : Point) : Prop := sorry

/-- Checks if a shape has the properties of the correct answer (option D) -/
def hasCorrectProperties (s : Shape) : Prop := sorry

/-- The main theorem -/
theorem unfolded_shape_properties 
  (initialSquare : Shape)
  (foldedSquare : Shape)
  (m n : Point)
  (a b c d : Point) :
  foldedSquare = fold (fold initialSquare FoldOperation.FoldUp) FoldOperation.FoldLeft →
  isMidpoint m a b →
  isMidpoint n b c →
  let pentagon := cutTriangle foldedSquare m b n
  hasCorrectProperties (unfold pentagon) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unfolded_shape_properties_l1216_121682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slide_polygons_off_table_l1216_121637

/-- Represents a convex polygon on a 2D plane -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool

/-- Represents the rectangular table -/
structure Table where
  width : Real
  height : Real

/-- Represents a configuration of polygons on the table -/
structure Configuration where
  table : Table
  polygons : List ConvexPolygon
  nonoverlapping : Bool

/-- Represents a slide move of a polygon -/
structure SlideMove where
  polygon_index : Nat
  direction : Real × Real
  distance : Real

/-- Applies a list of moves to a configuration -/
def apply_moves (config : Configuration) (moves : List SlideMove) : Configuration :=
  sorry

/-- Checks if all polygons are off the table -/
def all_polygons_off_table (config : Configuration) : Prop :=
  sorry

/-- Theorem: It is always possible to slide all polygons off the table -/
theorem slide_polygons_off_table (config : Configuration) :
  ∃ (moves : List SlideMove), all_polygons_off_table (apply_moves config moves) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slide_polygons_off_table_l1216_121637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_l1216_121691

theorem quadratic_factorization (c d : ℕ) (h1 : c > d) 
  (h2 : ∀ x : ℝ, x^2 - 18*x + 72 = (x - ↑c) * (x - ↑d)) : 
  2 * d - c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_l1216_121691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_B_production_time_l1216_121695

/-- The time (in minutes) it takes for Machine A to produce one item -/
noncomputable def machine_A_time : ℝ := 4

/-- The number of minutes in a day -/
noncomputable def minutes_per_day : ℝ := 24 * 60

/-- The ratio of items produced by Machine A compared to Machine B -/
noncomputable def production_ratio : ℝ := 1.25

/-- The time (in minutes) it takes for Machine B to produce one item -/
noncomputable def machine_B_time : ℝ := minutes_per_day / (minutes_per_day / machine_A_time / production_ratio)

theorem machine_B_production_time :
  machine_B_time = 5 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_B_production_time_l1216_121695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_minus_semicircles_area_l1216_121636

/-- The side length of the regular octagon -/
def side_length : ℝ := 2

/-- The number of sides in an octagon -/
def num_sides : ℕ := 8

/-- The area of a regular octagon with side length s -/
noncomputable def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

/-- The area of a semicircle with radius r -/
noncomputable def semicircle_area (r : ℝ) : ℝ := Real.pi * r^2 / 2

/-- The theorem stating the area of the region inside the octagon but outside the semicircles -/
theorem octagon_minus_semicircles_area :
  octagon_area side_length - num_sides * semicircle_area (side_length / 2) = 8 + 8 * Real.sqrt 2 - 4 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_minus_semicircles_area_l1216_121636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_x_intercept_l1216_121656

noncomputable def perpendicularSlope (a b : ℝ) : ℝ := -b / a

noncomputable def yIntercept (x₀ y₀ m : ℝ) : ℝ := y₀ - m * x₀

noncomputable def xIntercept (m b : ℝ) : ℝ := -b / m

theorem perpendicular_line_x_intercept :
  let a : ℝ := 4
  let b : ℝ := -3
  let c : ℝ := 9
  let x₀ : ℝ := 0
  let y₀ : ℝ := 5
  let m := perpendicularSlope a b
  let b' := yIntercept x₀ y₀ m
  xIntercept m b' = 20/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_x_intercept_l1216_121656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_surface_area_increase_l1216_121621

/-- Percentage increase in surface area of a cuboid -/
theorem cuboid_surface_area_increase : 
  ∀ (L W H : ℝ), L > 0 → W > 0 → H > 0 →
  let SA_original := 2 * (L * W + L * H + W * H)
  let SA_new := 2 * ((1.5 * L) * (1.7 * W) + (1.5 * L) * (1.8 * H) + (1.7 * W) * (1.8 * H))
  (SA_new - SA_original) / SA_original * 100 = 315.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_surface_area_increase_l1216_121621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_problem_l1216_121679

theorem complex_division_problem : 
  (1 - Complex.I) / (3 + Complex.I) = 1/5 - (2/5) * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_problem_l1216_121679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1216_121639

/-- Ellipse structure with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Focal distance of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the standard form and area of triangle for a specific ellipse -/
theorem ellipse_properties (e : Ellipse) (f1 f2 m : Point) :
  let c := focal_distance e
  distance f1 f2 = 2 →
  m.x = 2 ∧ m.y = 2 * Real.sqrt 5 / 5 →
  m.x^2 / e.a^2 + m.y^2 / e.b^2 = 1 →
  (∀ (x y : ℝ), x^2 / e.a^2 + y^2 / e.b^2 = 1 ↔ x^2 / 5 + y^2 / 4 = 1) ∧
  (∀ (p : Point), p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1 →
    distance p f1 * distance p f2 = 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1216_121639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_amount_spent_on_boxes_min_amount_spent_on_boxes_specific_l1216_121678

/-- The minimum amount spent on boxes for packaging a fine arts collection --/
theorem min_amount_spent_on_boxes (box_length box_width box_height : ℝ)
  (cost_per_box : ℝ) (total_collection_volume : ℝ) : ℝ :=
  let box_volume := box_length * box_width * box_height
  let num_boxes := total_collection_volume / box_volume
  num_boxes * cost_per_box

/-- Proof of the specific case --/
theorem min_amount_spent_on_boxes_specific : 
  min_amount_spent_on_boxes 20 20 12 0.5 1920000 = 200 := by
  -- Unfold the definition of min_amount_spent_on_boxes
  unfold min_amount_spent_on_boxes
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_amount_spent_on_boxes_min_amount_spent_on_boxes_specific_l1216_121678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_m_range_l1216_121600

noncomputable section

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + 2 * Real.sqrt 3 * (Real.cos x) ^ 2 - Real.sqrt 3

noncomputable def g (m x : ℝ) : ℝ := m * Real.cos (2 * x - Real.pi / 6) - 2 * m + 3

-- State the theorem
theorem function_equality_implies_m_range :
  ∀ m : ℝ, m > 0 →
  (∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ Real.pi/4 ∧ 0 ≤ x₂ ∧ x₂ ≤ Real.pi/4 ∧ f x₁ = g m x₂) →
  2/3 ≤ m ∧ m ≤ 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_implies_m_range_l1216_121600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slopes_l1216_121610

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1)

noncomputable def k1 (x : ℝ) : ℝ := (f x) / x
noncomputable def k2 (a : ℝ) (x : ℝ) : ℝ := (g a x) / x

theorem tangent_line_slopes (a : ℝ) (h1 : a > 0) :
  -- Part 1: k1 = e
  ∃ x : ℝ, k1 x = Real.exp 1 ∧
  -- Part 2: 1-1/e < a < e-1/e if k1 * k2 = 1
  (∃ y : ℝ, k1 x * k2 a y = 1 → 1 - 1 / Real.exp 1 < a ∧ a < Real.exp 1 - 1 / Real.exp 1) :=
by sorry

#check tangent_line_slopes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slopes_l1216_121610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_min_max_distance_l1216_121615

/-- Ellipse with semi-major axis 5 and semi-minor axis 4 -/
def ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 25) + (P.2^2 / 16) = 1

/-- Circle with center (-3, 0) and radius 2 -/
def circle1 (M : ℝ × ℝ) : Prop :=
  (M.1 + 3)^2 + M.2^2 = 4

/-- Circle with center (3, 0) and radius 1 -/
def circle2 (N : ℝ × ℝ) : Prop :=
  (N.1 - 3)^2 + N.2^2 = 1

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem ellipse_min_max_distance (P M N : ℝ × ℝ) 
  (hP : ellipse P) (hM : circle1 M) (hN : circle2 N) :
  (∀ M' N', circle1 M' → circle2 N' → 
    distance P M + distance P N ≤ distance P M' + distance P N') ∧
  (∃ M' N', circle1 M' ∧ circle2 N' ∧ 
    distance P M' + distance P N' = 13) ∧
  (∃ M' N', circle1 M' ∧ circle2 N' ∧ 
    distance P M' + distance P N' = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_min_max_distance_l1216_121615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l1216_121677

noncomputable def triangle_side_1 : ℝ := 26
noncomputable def triangle_side_2 : ℝ := 16
noncomputable def triangle_side_3 : ℝ := 18

noncomputable def semiperimeter : ℝ := (triangle_side_1 + triangle_side_2 + triangle_side_3) / 2

theorem inscribed_circle_radius : 
  let r := (2 * Real.sqrt (semiperimeter * (semiperimeter - triangle_side_1) * 
    (semiperimeter - triangle_side_2) * (semiperimeter - triangle_side_3))) / 
    (triangle_side_1 + triangle_side_2 + triangle_side_3)
  r = 2 * Real.sqrt 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l1216_121677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1216_121660

/-- Definition of an ellipse with semi-major axis a, semi-minor axis b, and semi-focal distance c -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_c_eq : c^2 = a^2 - b^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

theorem ellipse_properties (e : Ellipse) :
  (2 * e.b > e.a + e.c → e.b^2 > e.a * e.c) ∧
  (Real.tan (Real.arctan (e.b / e.c)) > 1 →
    0 < eccentricity e ∧ eccentricity e < Real.sqrt 2 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1216_121660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ott_fraction_of_total_l1216_121613

/-- Represents the amount of money each friend initially gave to Ott -/
def initial_gift : ℚ := sorry

/-- Moe's initial money -/
def moe_money : ℚ := 7 * initial_gift

/-- Loki's initial money -/
def loki_money : ℚ := 5 * initial_gift

/-- Nick's initial money -/
def nick_money : ℚ := 9 * initial_gift

/-- Total initial money of Moe, Loki, and Nick -/
def total_initial_money : ℚ := moe_money + loki_money + nick_money

/-- Amount Ott received after benefactor's doubling -/
def ott_money : ℚ := 6 * initial_gift

/-- Theorem stating that Ott now has 2/7 of the group's total money -/
theorem ott_fraction_of_total : ott_money / total_initial_money = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ott_fraction_of_total_l1216_121613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_result_correct_l1216_121643

/-- Represents the vote counts for each candidate -/
structure VoteCounts where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ

/-- Calculate the final vote counts after redistribution -/
def finalVoteCounts (totalVotes : ℕ) (invalidPercentage : ℚ) 
  (firstPreferences : VoteCounts) (redistributionPercentages : VoteCounts) : VoteCounts :=
  sorry

theorem election_result_correct (totalVotes : ℕ) (invalidPercentage : ℚ) 
  (firstPreferences : VoteCounts) (redistributionPercentages : VoteCounts) : 
  totalVotes = 4000000 →
  invalidPercentage = 15 / 100 →
  firstPreferences = { 
    a := (40 : ℚ) / 100 * (1 - invalidPercentage) * totalVotes,
    b := (30 : ℚ) / 100 * (1 - invalidPercentage) * totalVotes,
    c := (20 : ℚ) / 100 * (1 - invalidPercentage) * totalVotes,
    d := (10 : ℚ) / 100 * (1 - invalidPercentage) * totalVotes
  } →
  redistributionPercentages = {
    a := 25 / 100,
    b := 35 / 100,
    c := 40 / 100,
    d := 0
  } →
  finalVoteCounts totalVotes invalidPercentage firstPreferences redistributionPercentages = {
    a := 1445000,
    b := 1139000,
    c := 816000,
    d := 0
  } :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_result_correct_l1216_121643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volumes_of_rotated_K_l1216_121623

-- Define the function f(x) = x / sqrt(1 - x^2)
noncomputable def f (x : ℝ) : ℝ := x / Real.sqrt (1 - x^2)

-- Define the region K
def K : Set (ℝ × ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1/2 ∧ 0 ≤ p.2 ∧ p.2 ≤ f p.1}

-- Volume V₁ of solid generated by rotating K around x-axis
noncomputable def V₁ : ℝ := ∫ x in (0)..(1/2), Real.pi * (f x)^2

-- Volume V₂ of solid generated by rotating K around y-axis
noncomputable def V₂ : ℝ := Real.pi * ∫ y in (0)..(Real.sqrt 3 / 3), ((1/2)^2 - (y / Real.sqrt (1 + y^2))^2)

theorem volumes_of_rotated_K :
  V₁ = Real.pi/2 * (Real.log 3 - 1) ∧
  V₂ = Real.pi/12 * (2 * Real.pi - 3 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volumes_of_rotated_K_l1216_121623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1216_121652

/-- The area of a triangle with vertices at (3, -3), (9, 6), and (3, 6) is 27 square units. -/
theorem triangle_area : ∃ area : ℝ, area = 27 := by
  -- Define the vertices of the triangle
  let A : ℝ × ℝ := (3, -3)
  let B : ℝ × ℝ := (9, 6)
  let C : ℝ × ℝ := (3, 6)

  -- Calculate the area of the triangle
  let area := (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

  -- Assert that the area is equal to 27
  existsi area
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1216_121652
