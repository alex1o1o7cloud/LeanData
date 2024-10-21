import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_time_difference_l764_76493

/-- Represents the time difference in rowing against and with a stream -/
noncomputable def timeDifference (D S : ℝ) : ℝ :=
  D / (4 * S)

/-- Theorem stating the time difference in rowing against and with a stream -/
theorem rowing_time_difference 
  (D B S : ℝ) 
  (h1 : B = 3 * S) 
  (h2 : D > 0) 
  (h3 : S > 0) : 
  D / (B - S) - D / (B + S) = timeDifference D S :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_time_difference_l764_76493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_satisfies_conditions_l764_76442

noncomputable def p (x : ℝ) : ℝ := (8/5) * x^2 + (16/5) * x - 24/5

theorem p_satisfies_conditions :
  (∃ (q : ℝ → ℝ), p = λ x ↦ (x - 1) * (x + 3) * q x) ∧
  (∀ x, x ≠ 1 → x ≠ -3 → (x^3 + x^2 - 4*x - 4) / p x ≠ 0) ∧
  p 2 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_satisfies_conditions_l764_76442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_sum_l764_76436

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3

noncomputable def g (x : ℝ) : ℝ := -f x

noncomputable def h (x : ℝ) : ℝ := f (-x)

noncomputable def a : ℕ := (Set.ncard {x : ℝ | f x = g x})

noncomputable def b : ℕ := (Set.ncard {x : ℝ | f x = h x})

theorem intersection_points_sum : 10 * a + b = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_sum_l764_76436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_f_and_g_l764_76478

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (Real.pi / 2 - x)

theorem max_distance_between_f_and_g :
  ∃ (C : ℝ), C = 2 * Real.sqrt 2 ∧ 
  (∀ m : ℝ, |f m - g m| ≤ C) ∧
  (∃ m : ℝ, |f m - g m| = C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_f_and_g_l764_76478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_sum_l764_76461

/-- A triangle with angles 60° and 45°, where the side opposite the 60° angle is 12 units -/
structure SpecialTriangle where
  /-- The side opposite the 60° angle -/
  side_a : ℝ
  /-- The side opposite the 45° angle -/
  side_b : ℝ
  /-- The side opposite the 75° angle (180° - 60° - 45°) -/
  side_c : ℝ
  /-- The side opposite the 60° angle is 12 units -/
  side_a_length : side_a = 12

/-- The sum of the two sides adjacent to the 60° angle is approximately 41.0 -/
theorem special_triangle_sum (t : SpecialTriangle) : 
  ∃ ε > 0, |t.side_b + t.side_c - 41.0| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_sum_l764_76461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_real_l764_76444

def f (x : ℝ) : ℝ := 2 * x + 3

theorem domain_of_f_is_real : Set.range f = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_real_l764_76444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l764_76408

theorem tan_alpha_value (α : ℝ) (h1 : Real.cos α = -1/2) (h2 : π/2 < α ∧ α < π) :
  Real.tan α = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l764_76408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABM_equals_angle_ABN_l764_76485

-- Define the parabola C: y^2 = 2x
def C (x y : ℝ) : Prop := y^2 = 2*x

-- Define points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (-2, 0)

-- Define a line l passing through A
def line_through_A (t : ℝ) (x y : ℝ) : Prop := x = t*y + 2

-- Define intersection points M and N
noncomputable def M (t : ℝ) : ℝ × ℝ := 
  let y := Real.sqrt (4 / (1 + t^2))
  (t * y + 2, y)

noncomputable def N (t : ℝ) : ℝ × ℝ := 
  let y := -Real.sqrt (4 / (1 + t^2))
  (t * y + 2, y)

-- Define the angle between two vectors
noncomputable def angle (v1 v2 : ℝ × ℝ) : ℝ := 
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2)))

-- Helper function for vector subtraction
def vsub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

-- Theorem statement
theorem angle_ABM_equals_angle_ABN (t : ℝ) :
  angle (vsub (M t) B) (vsub A B) = angle (vsub (N t) B) (vsub A B) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABM_equals_angle_ABN_l764_76485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_theorem_l764_76452

/-- Represents a bus stop -/
structure BusStop where
  id : Nat

/-- Represents a passenger's journey -/
structure Journey where
  start : BusStop
  finish : BusStop

/-- Represents a bus route -/
structure BusRoute where
  stops : List BusStop
  journeys : List Journey
  num_stops : Nat
  max_capacity : Nat

/-- Predicate to check if a pair of stops satisfies the no-direct-journey condition -/
def no_direct_journey (route : BusRoute) (a b : BusStop) : Prop :=
  ∀ j ∈ route.journeys, ¬(j.start = a ∧ j.finish = b)

/-- Theorem statement for the bus stop problem -/
theorem bus_stop_theorem (route : BusRoute) 
  (h_stops : route.num_stops = 14)
  (h_capacity : route.max_capacity = 25) : 
  (∃ (a₁ b₁ a₂ b₂ a₃ b₃ a₄ b₄ : BusStop),
    a₁ ∈ route.stops ∧ b₁ ∈ route.stops ∧
    a₂ ∈ route.stops ∧ b₂ ∈ route.stops ∧
    a₃ ∈ route.stops ∧ b₃ ∈ route.stops ∧
    a₄ ∈ route.stops ∧ b₄ ∈ route.stops ∧
    a₁ ≠ b₁ ∧ a₂ ≠ b₂ ∧ a₃ ≠ b₃ ∧ a₄ ≠ b₄ ∧
    no_direct_journey route a₁ b₁ ∧
    no_direct_journey route a₂ b₂ ∧
    no_direct_journey route a₃ b₃ ∧
    no_direct_journey route a₄ b₄) ∧
  (¬∃ (a₁ b₁ a₂ b₂ a₃ b₃ a₄ b₄ a₅ b₅ : BusStop),
    a₁ ∈ route.stops ∧ b₁ ∈ route.stops ∧
    a₂ ∈ route.stops ∧ b₂ ∈ route.stops ∧
    a₃ ∈ route.stops ∧ b₃ ∈ route.stops ∧
    a₄ ∈ route.stops ∧ b₄ ∈ route.stops ∧
    a₅ ∈ route.stops ∧ b₅ ∈ route.stops ∧
    a₁ ≠ b₁ ∧ a₂ ≠ b₂ ∧ a₃ ≠ b₃ ∧ a₄ ≠ b₄ ∧ a₅ ≠ b₅ ∧
    no_direct_journey route a₁ b₁ ∧
    no_direct_journey route a₂ b₂ ∧
    no_direct_journey route a₃ b₃ ∧
    no_direct_journey route a₄ b₄ ∧
    no_direct_journey route a₅ b₅) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stop_theorem_l764_76452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_of_differences_l764_76400

/-- Given an arithmetic sequence {a_n} with common difference 3,
    S_n is the sum of the first n terms of {a_n} -/
noncomputable def S (a₁ : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a₁ + (n - 1) * 3)

/-- The theorem states that the sequence formed by S_20 - S_10, S_30 - S_20, S_40 - S_30
    is an arithmetic sequence with common difference 300 -/
theorem arithmetic_sequence_of_differences (a₁ : ℝ) :
  let diff (k : ℕ) := S a₁ (10 * (k + 2)) - S a₁ (10 * (k + 1))
  ∀ k : ℕ, diff (k + 1) - diff k = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_of_differences_l764_76400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_l764_76425

/-- Represents a quadratic function f(x) = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of the parabola -/
noncomputable def vertex (f : QuadraticFunction) : ℝ × ℝ :=
  (-f.b / (2 * f.a), f.a * (-f.b / (2 * f.a))^2 + f.b * (-f.b / (2 * f.a)) + f.c)

/-- A point lies between two other points on the real line -/
def between (x y z : ℝ) : Prop := (x < y ∧ y < z) ∨ (z < y ∧ y < x)

theorem parabola_property (f : QuadraticFunction) 
  (h1 : vertex f = (-1, 4))
  (h2 : ∃ x, between 2 x 3 ∧ f.a * x^2 + f.b * x + f.c = 0) :
  f.b = 2 * f.a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_l764_76425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l764_76427

theorem sin_2alpha_value (α t : ℝ) : 
  (Real.cos α)^2 - t * (Real.cos α) + t = 0 →
  (Real.sin α)^2 - t * (Real.sin α) + t = 0 →
  Real.sin (2 * α) = 2 - 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l764_76427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l764_76479

theorem complex_power_sum (z : ℂ) (h : z + z⁻¹ = 2 * Real.sqrt 2) : z^24 + (z^24)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l764_76479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_theorem_l764_76495

noncomputable section

-- Define the triangle
def triangle_sides : ℝ × ℝ × ℝ := (3, 3, Real.sqrt 6)

-- Define the angles in radians
def angle1 : ℝ := Real.arccos (2/3)
def angle2 : ℝ := (Real.pi - angle1) / 2
def angle3 : ℝ := angle2

-- Convert radians to degrees
def rad_to_deg (x : ℝ) : ℝ := x * 180 / Real.pi

-- Define the angles in degrees
def angle1_deg : ℝ := rad_to_deg angle1
def angle2_deg : ℝ := rad_to_deg angle2
def angle3_deg : ℝ := rad_to_deg angle3

-- Theorem statement
theorem triangle_angles_theorem (ε : ℝ) (hε : ε > 0) : 
  let (a, b, c) := triangle_sides
  a + b > c ∧ a + c > b ∧ b + c > a ∧
  abs (angle1_deg - 48.19) < ε ∧
  abs (angle2_deg - 65.905) < ε ∧
  abs (angle3_deg - 65.905) < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_theorem_l764_76495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_is_ten_l764_76438

/-- A data point representing (x, y) coordinates. -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- The linear relationship between x and y. -/
def linearRelation (x : ℝ) : ℝ := 2.1 * x - 0.3

/-- The given data points. -/
def dataPoints : List DataPoint := [
  ⟨1, 2⟩,
  ⟨2, 3⟩,
  ⟨3, 7⟩,
  ⟨4, 8⟩,
  ⟨5, 10⟩  -- We use 10 here as a placeholder for 'a'
]

/-- Calculate the mean of a list of real numbers. -/
noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

theorem value_of_a_is_ten :
  let xs := dataPoints.map (·.x)
  let ys := dataPoints.map (·.y)
  let x_mean := mean xs
  let y_mean := mean ys
  x_mean = 3 ∧ y_mean = linearRelation x_mean → dataPoints.getLast?.map (·.y) = some 10 := by
  sorry

#check value_of_a_is_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_is_ten_l764_76438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_solutions_l764_76455

theorem quadruple_solutions (a b p n : ℕ) (k : ℕ) (h_prime : Nat.Prime p) (h_eq : a^3 + b^3 = p^n) :
  ((a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨
   (a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
   (a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_solutions_l764_76455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_above_110_l764_76467

-- Define the normal distribution
structure NormalDist (μ σ : ℝ) where
  value : ℝ

-- Define the probability function
noncomputable def prob {α : Type*} (p : Set α) : ℝ := sorry

-- Define the random variable ξ
def ξ : NormalDist 100 10 where
  value := 100

-- State the theorem
theorem estimate_above_110 :
  prob {x : ℝ | 90 ≤ x ∧ x ≤ 100} = 0.3 →
  prob {x : ℝ | x > 110} = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_above_110_l764_76467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l764_76402

-- Define the function f(x) = -1/x + log₂(x)
noncomputable def f (x : ℝ) : ℝ := -1/x + Real.log x / Real.log 2

-- Theorem statement
theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l764_76402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turnip_pull_theorem_l764_76424

/-- Represents the strength of an entity relative to a Mouse -/
structure Strength where
  value : ℕ
deriving Repr

instance : HAdd Strength Strength Strength where
  hAdd a b := ⟨a.value + b.value⟩

instance : HMul ℕ Strength Strength where
  hMul n s := ⟨n * s.value⟩

instance : LE Strength where
  le a b := a.value ≤ b.value

instance : LT Strength where
  lt a b := a.value < b.value

/-- The strength required to pull up the Turnip -/
def TurnipStrength : Strength := ⟨1237⟩

/-- The strength of a Mouse -/
def MouseStrength : Strength := ⟨1⟩

/-- The strength of a Cat relative to a Mouse -/
def CatStrength : Strength := 6 * MouseStrength

/-- The strength of a Dog relative to a Cat -/
def DogStrength : Strength := 5 * CatStrength

/-- The strength of a Granddaughter relative to a Dog -/
def GranddaughterStrength : Strength := 4 * DogStrength

/-- The strength of a Grandma relative to a Granddaughter -/
def GrandmaStrength : Strength := 3 * GranddaughterStrength

/-- The strength of a Grandpa relative to a Grandma -/
def GrandpaStrength : Strength := 2 * GrandmaStrength

/-- The combined strength of all entities except Mouse -/
def CombinedStrengthWithoutMouse : Strength :=
  GrandpaStrength + GrandmaStrength + GranddaughterStrength + DogStrength + CatStrength

/-- The combined strength of all entities including Mouse -/
def CombinedStrengthWithMouse : Strength :=
  CombinedStrengthWithoutMouse + MouseStrength

theorem turnip_pull_theorem :
  CombinedStrengthWithMouse ≥ TurnipStrength ∧
  CombinedStrengthWithoutMouse < TurnipStrength ∧
  TurnipStrength = 1237 * MouseStrength :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turnip_pull_theorem_l764_76424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_zero_l764_76410

/-- The function f(x) defined as (x + a) * ln((2x - 1) / (2x + 1)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

/-- The theorem stating that if f is even, then a must be 0 -/
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_zero_l764_76410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABCD_sum_p_q_l764_76462

noncomputable section

-- Define the points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (6, 1)
def D : ℝ × ℝ := (8, -1)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem perimeter_ABCD :
  distance A B + distance B C + distance C D + distance D A = 10 * Real.sqrt 2 + 2 * Real.sqrt 5 := by
  sorry

-- Define the perimeter
noncomputable def perimeter : ℝ := distance A B + distance B C + distance C D + distance D A

-- State the final result
theorem sum_p_q : ∃ (p q : ℤ), perimeter = p * Real.sqrt 2 + q * Real.sqrt 5 ∧ p + q = 12 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABCD_sum_p_q_l764_76462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_federal_return_charge_is_50_l764_76430

/-- The charge for a federal return at Kwik-e-Tax Center -/
def federal_return_charge : ℕ → Prop := sorry

/-- The charge for a state return at Kwik-e-Tax Center -/
def state_return_charge : ℕ := 30

/-- The charge for quarterly business taxes at Kwik-e-Tax Center -/
def quarterly_taxes_charge : ℕ := 80

/-- The number of federal returns sold -/
def federal_returns_sold : ℕ := 60

/-- The number of state returns sold -/
def state_returns_sold : ℕ := 20

/-- The number of quarterly returns sold -/
def quarterly_returns_sold : ℕ := 10

/-- The total revenue for the day -/
def total_revenue : ℕ := 4400

/-- Theorem stating that the charge for a federal return is $50 -/
theorem federal_return_charge_is_50 :
  federal_return_charge 50 ↔
    federal_returns_sold * 50 +
    state_returns_sold * state_return_charge +
    quarterly_returns_sold * quarterly_taxes_charge =
    total_revenue :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_federal_return_charge_is_50_l764_76430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_player_average_l764_76450

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  runsNeeded : ℕ
  averageIncrease : ℕ

/-- Calculates the current average runs per innings for a cricket player -/
def currentAverage (player : CricketPlayer) : ℚ :=
  let totalInnings := player.innings + 1
  let x := (player.innings * player.averageIncrease + player.runsNeeded) / player.averageIncrease
  x

/-- Theorem: The current average of the cricket player is 24 runs per innings -/
theorem cricket_player_average (player : CricketPlayer) 
  (h1 : player.innings = 8)
  (h2 : player.runsNeeded = 96)
  (h3 : player.averageIncrease = 8) :
  currentAverage player = 24 := by
  sorry

#eval currentAverage { innings := 8, runsNeeded := 96, averageIncrease := 8 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_player_average_l764_76450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_and_inequality_l764_76421

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

theorem odd_function_values_and_inequality (a b t : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →
  (∀ x, f a b (x^2 - x) + f a b (2*x^2 - t) < 0) →
  (a = 2 ∧ b = 1 ∧ t < -1/12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_and_inequality_l764_76421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_sum_of_composites_l764_76443

/-- A natural number is composite if it has a divisor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

/-- A natural number can be represented as the sum of two composite numbers -/
def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

/-- 11 is the largest natural number that cannot be represented as the sum of two composite numbers -/
theorem largest_non_sum_of_composites :
  (∀ m : ℕ, m > 11 → IsSumOfTwoComposites m) ∧
  ¬IsSumOfTwoComposites 11 ∧
  (∀ k : ℕ, k < 11 → IsSumOfTwoComposites k → k ≤ 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_sum_of_composites_l764_76443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_lines_l764_76426

noncomputable section

variable (a : ℝ)

def x (t : ℝ) : ℝ := a * (t * Real.sin t + Real.cos t)
def y (t : ℝ) : ℝ := a * (Real.sin t - t * Real.cos t)

def t₀ : ℝ := Real.pi / 4

theorem tangent_and_normal_lines :
  (∃ (x₀ y₀ : ℝ),
    x₀ = x a t₀ ∧
    y₀ = y a t₀ ∧
    (∀ (x' y' : ℝ),
      -- Tangent line equation
      y' = x' + (a * Real.sqrt 2 * Real.pi) / 4 ↔
        (y' - y₀) = (x' - x₀) * ((deriv (y a) t₀) / (deriv (x a) t₀))) ∧
    (∀ (x' y' : ℝ),
      -- Normal line equation
      y' = -x' + a * Real.sqrt 2 ↔
        (y' - y₀) = -(x' - x₀) / ((deriv (y a) t₀) / (deriv (x a) t₀)))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_lines_l764_76426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_range_l764_76472

/-- The range of slopes k for which there are at least three distinct points on the circle
    x^2 + y^2 - 4x - 4y = 0 with distance √2 from the line y = kx -/
theorem circle_line_distance_range :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 4*x - 4*y = 0}
  let line (k : ℝ) := {(x, y) : ℝ × ℝ | y = k*x}
  let distance (p : ℝ × ℝ) (k : ℝ) := |k * p.1 - p.2| / Real.sqrt (k^2 + 1)
  ∃ (p1 p2 p3 : ℝ × ℝ), p1 ∈ circle ∧ p2 ∈ circle ∧ p3 ∈ circle ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    distance p1 k = Real.sqrt 2 ∧ distance p2 k = Real.sqrt 2 ∧ distance p3 k = Real.sqrt 2
  ↔ 2 - Real.sqrt 3 ≤ k ∧ k ≤ 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_range_l764_76472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l764_76445

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2:ℝ)^(8*x+4) * (16:ℝ)^(x+2) = (8:ℝ)^(6*x+10) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l764_76445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l764_76422

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => 2 * sequence_a n / (1 + sequence_a n)

def sequence_b (n : ℕ) : ℚ := (n + 1 : ℚ) / sequence_a n

def S (n : ℕ) : ℚ := (Finset.range n).sum (λ k => sequence_b k)

theorem sequence_properties :
  (∀ n : ℕ, (1 / sequence_a n - 1) = (1 / 2 : ℚ) ^ n) ∧
  (∀ n : ℕ, sequence_a n = (2 ^ n : ℚ) / (1 + 2 ^ n : ℚ)) ∧
  (∀ n : ℕ, n ≥ 3 → S n > (n + 1)^2 / 2 + 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l764_76422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_coefficient_ratio_l764_76401

noncomputable def k1 : ℝ := Real.pi / 6

noncomputable def k2 : ℝ := Real.pi / 4

def k3 : ℝ := 1

variable (a : ℝ)

noncomputable def sphere_volume (a : ℝ) : ℝ := k1 * a^3

noncomputable def cylinder_volume (a : ℝ) : ℝ := k2 * a^3

def cube_volume (a : ℝ) : ℝ := k3 * a^3

theorem volume_coefficient_ratio :
  (k1 : ℝ) / k3 = Real.pi / 6 ∧ (k2 : ℝ) / k3 = Real.pi / 4 ∧ k3 / k3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_coefficient_ratio_l764_76401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_minus3_plus_i_to_1_minus_i_l764_76412

/-- The distance between two complex numbers in the complex plane -/
noncomputable def complex_distance (z₁ z₂ : ℂ) : ℝ :=
  Real.sqrt ((z₁.re - z₂.re)^2 + (z₁.im - z₂.im)^2)

/-- Theorem: The distance between -3 + i and 1 - i in the complex plane is 2√5 -/
theorem distance_minus3_plus_i_to_1_minus_i :
  complex_distance (-3 + I) (1 - I) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_minus3_plus_i_to_1_minus_i_l764_76412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_gt_bound_l764_76440

/-- The function f(x) defined as a(e^x + a) - x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

/-- Theorem stating that the minimum value of f(x) is greater than 2ln(a) + 3/2 when a > 0 --/
theorem f_min_gt_bound (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, f a x > 2 * Real.log a + 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_gt_bound_l764_76440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l764_76486

noncomputable def z : ℂ := 1 - Complex.I * Real.sqrt 2

theorem point_in_fourth_quadrant :
  let w := z + 1 / z
  (w.re > 0) ∧ (w.im < 0) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l764_76486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l764_76491

def a : ℕ → ℚ
  | 0 => 2  -- Added case for 0
  | n + 1 => a n / (1 + 3 * a n)

theorem a_10_value : a 10 = 2 / 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l764_76491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_properties_l764_76441

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp (Real.log 3 * x)) + (Real.exp (Real.log 3 * (-x)))

-- State the theorem
theorem even_function_properties :
  ∃ (a : ℝ), 
    (∀ (x : ℝ), f a x = f a (-x)) ∧  -- f is an even function
    (a = 1) ∧  -- a = 1
    (∀ (x y : ℝ), 0 < x → x < y → f a x < f a y) ∧  -- f is monotonically increasing on (0, +∞)
    (∀ (x : ℝ), f a (Real.log x / Real.log 10) < f a 1 ↔ 1/10 < x ∧ x < 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_properties_l764_76441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l764_76433

/-- The parabola y = x^2 - 8x + 18 -/
noncomputable def parabola (x : ℝ) : ℝ := x^2 - 8*x + 18

/-- The line y = 2x - 8 -/
noncomputable def line (x : ℝ) : ℝ := 2*x - 8

/-- The distance between a point (x, parabola x) on the parabola and the line -/
noncomputable def distance_to_line (x : ℝ) : ℝ :=
  |2*x - parabola x - 8| / Real.sqrt 5

theorem shortest_distance :
  ∃ (d : ℝ), d > 0 ∧ d = Real.sqrt (1/5) ∧
  ∀ (x : ℝ), distance_to_line x ≥ d := by
  sorry

#check shortest_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l764_76433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insect_start_position_l764_76448

/-- Represents the position of the insect after n jumps -/
def position (K₀ : ℤ) (n : ℕ) : ℤ :=
  K₀ + (n / 2 + 1) * n - if n % 2 = 0 then 2 * ((n / 2) * (n / 2)) else ((n / 2 + 1) * (n / 2 + 1))

/-- The theorem stating that if the insect ends at 2013 after 100 jumps, it must have started at 1963 -/
theorem insect_start_position (K₀ : ℤ) :
  position K₀ 100 = 2013 → K₀ = 1963 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_insect_start_position_l764_76448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_inclination_difference_obtuse_inclination_range_specific_line_inclination_l764_76498

-- Define the concept of inclination angle
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m

-- Define perpendicularity of lines
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_inclination_difference {α1 α2 : ℝ} 
  (h : perpendicular (Real.tan α1) (Real.tan α2)) : 
  |α1 - α2| = π / 2 := by sorry

theorem obtuse_inclination_range (a : ℝ) :
  inclination_angle (a^2 + 2*a) > π/2 → -2 < a ∧ a < 0 := by sorry

theorem specific_line_inclination :
  inclination_angle (-Real.tan (π/7)) = 6*π/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_inclination_difference_obtuse_inclination_range_specific_line_inclination_l764_76498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_notebooks_obtained_l764_76413

-- Define the parameters of the problem
def initial_money : ℕ := 150
def notebook_cost : ℕ := 4
def stickers_for_notebook : ℕ := 5

-- Function to calculate the total number of notebooks obtained
def total_notebooks (money : ℕ) (cost : ℕ) (exchange_rate : ℕ) : ℕ :=
  let initial_notebooks := money / cost
  let rec additional_notebooks (stickers : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then 0
    else
      let new_notebooks := stickers / exchange_rate
      if new_notebooks = 0 then 0
      else new_notebooks + additional_notebooks (stickers % exchange_rate + new_notebooks) (fuel - 1)
  initial_notebooks + additional_notebooks initial_notebooks initial_notebooks

-- Theorem statement
theorem max_notebooks_obtained :
  total_notebooks initial_money notebook_cost stickers_for_notebook = 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_notebooks_obtained_l764_76413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_perimeter_calculation_l764_76477

/-- The perimeter of a circular sector with two radii of length 7 and an arc of 5/6 of the full circle -/
noncomputable def sectorPerimeter : ℝ := 14 + (35 * Real.pi / 3)

/-- The length of an arc that is 5/6 of a circle with radius 7 -/
noncomputable def arcLength : ℝ := 5 / 6 * (2 * Real.pi * 7)

theorem sector_perimeter_calculation (radius : ℝ) (arcFraction : ℝ) 
  (h1 : radius = 7)
  (h2 : arcFraction = 5 / 6) :
  2 * radius + arcFraction * (2 * Real.pi * radius) = sectorPerimeter := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_perimeter_calculation_l764_76477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_price_for_original_revenue_min_sales_volume_after_reform_l764_76499

-- Define the original price and sales volume
noncomputable def original_price : ℝ := 25
noncomputable def original_volume : ℝ := 80000

-- Define the price-volume relationship
noncomputable def volume_change (price_change : ℝ) : ℝ := -2000 * price_change

-- Define the new revenue function
noncomputable def new_revenue (price : ℝ) : ℝ :=
  price * (original_volume + volume_change (price - original_price))

-- Define the original revenue
noncomputable def original_revenue : ℝ := original_price * original_volume

-- Define the investment functions
noncomputable def tech_innovation_cost (x : ℝ) : ℝ := (1 / 6) * (x^2 - 600)
noncomputable def fixed_ad_cost : ℝ := 50
noncomputable def variable_ad_cost (x : ℝ) : ℝ := x / 5

-- Define the total investment function
noncomputable def total_investment (x : ℝ) : ℝ :=
  tech_innovation_cost x + fixed_ad_cost + variable_ad_cost x

-- Theorem 1: Maximum price for original revenue
theorem max_price_for_original_revenue :
  ∃ (max_price : ℝ), max_price = 40 ∧
    ∀ (p : ℝ), p ≤ max_price → new_revenue p ≥ original_revenue := by
  sorry

-- Theorem 2: Minimum sales volume after reform
theorem min_sales_volume_after_reform :
  ∃ (min_volume : ℝ) (optimal_price : ℝ),
    min_volume = 10.2 * 10^6 ∧ optimal_price = 30 ∧
    min_volume * optimal_price ≥ original_revenue + total_investment optimal_price := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_price_for_original_revenue_min_sales_volume_after_reform_l764_76499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_l764_76409

theorem simplify_fraction (x : ℝ) (hx : x ≠ 0) :
  5 / (4 * x^2) - 4 * x^3 / 5 = (25 * x^2 - 16 * x^3) / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_l764_76409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_vertex_product_l764_76483

/-- A regular decagon in the complex plane -/
structure RegularDecagon where
  vertices : Fin 10 → ℂ
  is_regular : ∀ (i j : Fin 10), Complex.abs (vertices i - vertices j) = Complex.abs (vertices 0 - vertices 1)
  q1_at_2 : vertices 0 = 2
  q6_at_4 : vertices 5 = 4

/-- The product of all vertices of a regular decagon -/
def vertex_product (d : RegularDecagon) : ℂ :=
  Finset.prod (Finset.univ : Finset (Fin 10)) (λ i => d.vertices i)

/-- The theorem stating that the product of vertices equals 59048 -/
theorem regular_decagon_vertex_product (d : RegularDecagon) :
  vertex_product d = 59048 := by
  sorry

#check regular_decagon_vertex_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_vertex_product_l764_76483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_squares_l764_76403

theorem positive_difference_squares : (8^2 * 8^2) / 8 - (8^2 + 8^2) / 8 = 496 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_squares_l764_76403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l764_76463

noncomputable def f (x : ℝ) := Real.sqrt (3 - 2*x - x^2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l764_76463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l764_76447

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: Perimeter of triangle ADE in the given ellipse configuration -/
theorem ellipse_triangle_perimeter
  (C : Ellipse)
  (A D E : Point)
  (h_eccentricity : C.a * (1/2) = Real.sqrt (C.a^2 - C.b^2))
  (h_DE_perpendicular : ∃ (F₁ F₂ : Point), 
    F₁.x^2 / C.a^2 + F₁.y^2 / C.b^2 < 1 ∧
    F₂.x^2 / C.a^2 + F₂.y^2 / C.b^2 < 1 ∧
    (D.y - E.y) * (A.x - F₂.x) = (A.y - F₂.y) * (D.x - E.x))
  (h_DE_on_ellipse : D.x^2 / C.a^2 + D.y^2 / C.b^2 = 1 ∧
                     E.x^2 / C.a^2 + E.y^2 / C.b^2 = 1)
  (h_DE_length : distance D E = 6)
  : distance A D + distance A E + distance D E = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l764_76447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l764_76488

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define points A, B, C, D, and P
variable (A B C D P : ℝ × ℝ)

-- Define the perpendicularity conditions
axiom DA_perp_AB : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0
axiom CB_perp_AB : (C.1 - B.1) * (B.1 - A.1) + (C.2 - B.2) * (B.2 - A.2) = 0

-- Define the distance conditions
axiom DA_length : (D.1 - A.1)^2 + (D.2 - A.2)^2 = 18
axiom CB_length : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 2

-- P is on the ellipse
axiom P_on_ellipse : ellipse P.1 P.2

-- Define the area of triangle PCD
noncomputable def triangle_area (P C D : ℝ × ℝ) : ℝ :=
  abs ((P.1 - D.1) * (C.2 - D.2) - (P.2 - D.2) * (C.1 - D.1)) / 2

-- Theorem statement
theorem min_triangle_area :
  ∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧
  ∀ (Q : ℝ × ℝ), ellipse Q.1 Q.2 →
  triangle_area P C D ≤ triangle_area Q C D ∧
  triangle_area P C D = 4 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l764_76488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l764_76475

/-- The hyperbola equation -/
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- The parabola equation -/
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The asymptote equation of the hyperbola -/
def asymptote (a b x y : ℝ) : Prop := y = (b / a) * x

/-- The directrix equation of the parabola -/
def directrix (p x : ℝ) : Prop := x = -p / 2

theorem hyperbola_equation (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  (∃ x y : ℝ, hyperbola a b x y ∧ parabola p x y) →
  (distance (-a) 0 (p/2) 0 = 3) →
  (∃ x y : ℝ, asymptote a b x y ∧ directrix p x ∧ x = -1 ∧ y = -1) →
  (a = 2 ∧ b = 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l764_76475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ratio_inverse_l764_76432

theorem f_ratio_inverse (x : ℝ) (hx : x ≠ 0) : 
  (fun x => (1 + x^2) / (1 - x^2)) x / (fun x => (1 + x^2) / (1 - x^2)) (x⁻¹) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ratio_inverse_l764_76432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_evaluation_log_expression_in_terms_of_a_and_b_l764_76481

-- Problem 1
theorem complex_expression_evaluation :
  Real.sqrt ((1 : ℝ) * (2 + 1/4)) - (-0.96 : ℝ)^(0 : ℝ) - (3 + 3/8 : ℝ)^(-(2/3 : ℝ)) + (3/2 : ℝ)^(-(2 : ℝ)) + ((-32 : ℝ)^(-(4 : ℝ)))^(-(3/4 : ℝ)) = 5/2 := by
  sorry

-- Problem 2
theorem log_expression_in_terms_of_a_and_b (a b : ℝ) (h1 : (14 : ℝ)^a = 6) (h2 : (14 : ℝ)^b = 7) :
  Real.log 56 / Real.log 42 = (3 - 2*b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_evaluation_log_expression_in_terms_of_a_and_b_l764_76481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_correct_l764_76471

/-- The number of intersection points between r = 6 cos θ and r = 10 sin θ -/
def intersection_count : ℕ := 2

/-- The first curve in Cartesian coordinates -/
noncomputable def curve1 (θ : ℝ) : ℝ × ℝ :=
  (6 * Real.cos θ * Real.cos θ, 6 * Real.cos θ * Real.sin θ)

/-- The second curve in Cartesian coordinates -/
noncomputable def curve2 (θ : ℝ) : ℝ × ℝ :=
  (10 * Real.sin θ * Real.cos θ, 10 * Real.sin θ * Real.sin θ)

/-- Theorem stating that the number of intersection points is correct -/
theorem intersection_count_correct :
  ∃ (θ₁ θ₂ : ℝ), θ₁ ≠ θ₂ ∧ curve1 θ₁ = curve2 θ₁ ∧ curve1 θ₂ = curve2 θ₂ ∧
  ∀ (θ : ℝ), curve1 θ = curve2 θ → θ = θ₁ ∨ θ = θ₂ := by
  sorry

#check intersection_count_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_correct_l764_76471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l764_76420

noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 3 + Real.pi / 4)

theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = 3 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l764_76420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_theorem_l764_76480

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
noncomputable def satisfies_conditions (q : Quadrilateral) : Prop :=
  let (xa, ya) := q.A
  let (xb, yb) := q.B
  let (xc, yc) := q.C
  let (xd, yd) := q.D
  ∃ (r s : ℕ),
    -- BC = 10
    Real.sqrt ((xb - xc)^2 + (yb - yc)^2) = 10 ∧
    -- CD = 15
    Real.sqrt ((xc - xd)^2 + (yc - yd)^2) = 15 ∧
    -- AD = 12
    Real.sqrt ((xa - xd)^2 + (ya - yd)^2) = 12 ∧
    -- m∠A = m∠B = 60°
    Real.arccos ((xb - xa) * (xd - xa) + (yb - ya) * (yd - ya)) /
      (Real.sqrt ((xb - xa)^2 + (yb - ya)^2) * Real.sqrt ((xd - xa)^2 + (yd - ya)^2)) = Real.pi / 3 ∧
    Real.arccos ((xa - xb) * (xc - xb) + (ya - yb) * (yc - yb)) /
      (Real.sqrt ((xa - xb)^2 + (ya - yb)^2) * Real.sqrt ((xc - xb)^2 + (yc - yb)^2)) = Real.pi / 3 ∧
    -- AB = r + √s
    Real.sqrt ((xa - xb)^2 + (ya - yb)^2) = r + Real.sqrt (s : ℝ) ∧
    r > 0 ∧ s > 0

-- Theorem statement
theorem quadrilateral_theorem (q : Quadrilateral) :
  satisfies_conditions q → ∃ (r s : ℕ), r + s = 487 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_theorem_l764_76480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cone_slices_l764_76496

/-- Represents a right circular cone. -/
structure RightCircularCone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a cone slice. -/
structure ConeSlice where
  height : ℝ
  baseRadius : ℝ

/-- Calculate the volume of a cone slice. -/
noncomputable def volumeOfConeSlice (slice : ConeSlice) : ℝ :=
  (1/3) * Real.pi * slice.baseRadius^2 * slice.height

/-- Slice a cone into three equal-height pieces. -/
noncomputable def sliceCone (cone : RightCircularCone) : (ConeSlice × ConeSlice × ConeSlice) :=
  let sliceHeight := cone.height / 3
  let topSlice : ConeSlice := { height := sliceHeight, baseRadius := cone.baseRadius / 3 }
  let middleSlice : ConeSlice := { height := sliceHeight, baseRadius := 2 * cone.baseRadius / 3 }
  let bottomSlice : ConeSlice := { height := sliceHeight, baseRadius := cone.baseRadius }
  (topSlice, middleSlice, bottomSlice)

/-- The main theorem: the ratio of volumes of the smallest to largest slice is 1/27. -/
theorem volume_ratio_of_cone_slices (cone : RightCircularCone) :
  let (topSlice, _, bottomSlice) := sliceCone cone
  volumeOfConeSlice topSlice / volumeOfConeSlice bottomSlice = 1/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cone_slices_l764_76496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_root_l764_76446

theorem cubic_polynomial_root (x : ℝ) : x^3 = 3 + 8*x^2 - 12*x → x^3 - 6*x^2 + 12*x - 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_root_l764_76446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_of_f_l764_76460

/-- The rational function f(x) = (3x^2 - 2x - 4) / (2x - 5) -/
noncomputable def f (x : ℝ) : ℝ := (3*x^2 - 2*x - 4) / (2*x - 5)

/-- The proposed slant asymptote g(x) = 1.5x + 4.25 -/
def g (x : ℝ) : ℝ := 1.5*x + 4.25

/-- Theorem stating that g is the slant asymptote of f -/
theorem slant_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x > M, |f x - g x| < ε :=
by
  sorry

#check slant_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_of_f_l764_76460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equally_illuminated_points_l764_76453

/-- Two light sources with given intensity ratio and positions -/
structure LightSources (m n a : ℝ) where
  ratio : m / n > 0
  distance : a > 0

/-- Point equally illuminated by two light sources -/
def equallyIlluminatedPoint (ls : LightSources m n a) (x : ℝ) : Prop :=
  x * x / ((a - x) * (a - x)) = m / n

/-- Point outside the line equally illuminated by two light sources -/
def equallyIlluminatedPointOutside (ls : LightSources m n a) (b : ℝ) (x : ℝ) : Prop :=
  (b * b + x * x) / (b * b + (a - x) * (a - x)) = m / n

theorem equally_illuminated_points 
  (m n a b : ℝ) (ls : LightSources m n a) :
  (∃ x, equallyIlluminatedPoint ls x ↔ 
    x = a * (m + Real.sqrt (m * n)) / (m - n) ∨ 
    x = a * (m - Real.sqrt (m * n)) / (m - n)) ∧
  (∃ x, equallyIlluminatedPointOutside ls b x ↔ 
    x = (m * a + Real.sqrt (m * a^2 * n - b^2 * (n - m)^2)) / (n - m) ∨ 
    x = (m * a - Real.sqrt (m * a^2 * n - b^2 * (n - m)^2)) / (n - m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equally_illuminated_points_l764_76453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_98_div_3_l764_76451

-- Define the point of intersection
def intersection_point : ℝ × ℝ := (3, 1)

-- Define the slopes of the two lines
def slope1 : ℝ := 3
def slope2 : ℝ := -1

-- Define the y-coordinate of the horizontal line
def horizontal_line_y : ℝ := 8

-- Theorem statement
theorem triangle_area_is_98_div_3 :
  let line1 := λ (x : ℝ) => slope1 * (x - intersection_point.1) + intersection_point.2
  let line2 := λ (x : ℝ) => slope2 * (x - intersection_point.1) + intersection_point.2
  let x1 := (horizontal_line_y - intersection_point.2) / slope1 + intersection_point.1
  let x2 := (horizontal_line_y - intersection_point.2) / slope2 + intersection_point.1
  (1/2 : ℝ) * |x1 * (horizontal_line_y - intersection_point.2) +
              x2 * (intersection_point.2 - horizontal_line_y) +
              intersection_point.1 * (horizontal_line_y - horizontal_line_y)| = 98/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_98_div_3_l764_76451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_m_range_l764_76435

noncomputable def f (x m : ℝ) : ℝ := 2^(-|x - 1|) - m

theorem root_implies_m_range :
  ∀ m : ℝ, (∃ x : ℝ, f x m = 0) → m ∈ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_m_range_l764_76435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l764_76494

noncomputable section

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (-Real.sqrt 3, 0)
def focus2 : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the point that the ellipse passes through
def ellipse_point : ℝ × ℝ := (-1, Real.sqrt 3 / 2)

-- Define the fixed point A
def point_A : ℝ × ℝ := (1, 1/2)

-- Define a line through the origin
def line_through_origin (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define the area of triangle MAN
def triangle_area (k : ℝ) : ℝ := 
  2 * abs (k - 1/2) / Real.sqrt (1 + 4*k^2)

-- Theorem statement
theorem max_triangle_area :
  ∃ (k : ℝ), ∀ (k' : ℝ), triangle_area k ≥ triangle_area k' ∧ 
  triangle_area k = Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l764_76494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_existence_l764_76490

theorem subset_existence (A : Finset ℕ) : ∃ B : Finset ℕ, B ⊆ A ∧
  (∀ b₁ b₂ : ℕ, b₁ ∈ B → b₂ ∈ B → b₁ ≠ b₂ → 
    (¬(b₁ ∣ b₂) ∧ ¬(b₂ ∣ b₁) ∧ ¬((b₁+1) ∣ (b₂+1)) ∧ ¬((b₂+1) ∣ (b₁+1)))) ∧
  (∀ a : ℕ, a ∈ A → ∃ b : ℕ, b ∈ B ∧ ((a ∣ b) ∨ ((b+1) ∣ (a+1)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_existence_l764_76490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_third_iter_roots_f_iter_roots_l764_76423

def f (x : ℝ) := 2 * x^2 + x - 1

def f_iter : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ f_iter n

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem f_third_iter_roots :
  ∃ (r1 r2 r3 : ℝ), r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
  r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧
  (∀ x > 0, f_iter 3 x = x ↔ x = r1 ∨ x = r2 ∨ x = r3) :=
sorry

theorem f_iter_roots (n : ℕ) :
  ∃ (S : Finset ℝ), (∀ x ∈ S, x > 0) ∧ 
  Finset.card S = 2 * fib n ∧
  (∀ x > 0, f_iter n x = 0 ↔ x ∈ S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_third_iter_roots_f_iter_roots_l764_76423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l764_76454

theorem discount_calculation (P : ℝ) (h : P > 0) : 
  P * (1 - 0.3) * (1 - 0.2) = P * (1 - 0.44) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l764_76454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_daily_sales_l764_76469

def f (x : ℚ) : ℚ := 20 - (1 / 2) * |x - 10|

def g (x : ℚ) : ℚ := 80 - 2 * x

def y (x : ℚ) : ℚ := f x * g x

theorem min_daily_sales :
  ∀ x : ℚ, 0 ≤ x ∧ x ≤ 20 → y x ≥ 600 ∧ ∃ x₀ : ℚ, 0 ≤ x₀ ∧ x₀ ≤ 20 ∧ y x₀ = 600 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_daily_sales_l764_76469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dune_buggy_averages_l764_76482

/-- Represents the speed and fuel efficiency of a dune buggy in different terrains --/
structure DuneBuggy where
  flat_speed : ℝ
  downhill_speed_increase : ℝ
  uphill_speed_decrease : ℝ
  wind_effect : ℝ
  uphill_fuel_increase : ℝ
  downhill_fuel_decrease : ℝ

/-- Calculates the average speed and fuel efficiency of a dune buggy --/
noncomputable def calculate_averages (db : DuneBuggy) : ℝ × ℝ :=
  let flat_speed := db.flat_speed
  let downhill_speed := db.flat_speed + db.downhill_speed_increase
  let uphill_speed := db.flat_speed - db.uphill_speed_decrease
  
  let downhill_speed_with_wind := downhill_speed * (1 + db.wind_effect)
  let uphill_speed_with_wind := uphill_speed * (1 - db.wind_effect)
  
  let avg_speed := (flat_speed + downhill_speed_with_wind + uphill_speed_with_wind) / 3
  
  let flat_fuel := 1
  let downhill_fuel := 1 - db.downhill_fuel_decrease
  let uphill_fuel := 1 + db.uphill_fuel_increase
  
  let avg_fuel := (flat_fuel + downhill_fuel + uphill_fuel) / 3
  
  (avg_speed, avg_fuel)

theorem dune_buggy_averages (db : DuneBuggy) 
  (h1 : db.flat_speed = 60)
  (h2 : db.downhill_speed_increase = 12)
  (h3 : db.uphill_speed_decrease = 18)
  (h4 : db.wind_effect = 0.05)
  (h5 : db.uphill_fuel_increase = 0.2)
  (h6 : db.downhill_fuel_decrease = 0.1) :
  let (avg_speed, avg_fuel) := calculate_averages db
  avg_speed = 58.5 ∧ avg_fuel = 0.9667 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dune_buggy_averages_l764_76482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_l764_76487

-- Problem 1
theorem problem_1 : -2^2 + (3 : ℝ)^0 - (-1/2)⁻¹ = -1 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (x + 2) * (x - 1) - 3 * x * (x + 1) = 4 * (x^2 + x - 1/2) := by sorry

-- Problem 3
theorem problem_3 (a : ℝ) (h : a ≠ 0) : 2 * a^6 - a^2 * a^4 + (2 * a^4)^2 / a^4 = a^4 * (a^2 + 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_l764_76487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_polygons_l764_76404

/-- A polygon is represented as a list of points in the plane -/
def Polygon := List (ℝ × ℝ)

/-- Function to combine two polygons -/
def combine_polygons (p1 p2 : Polygon) : List Polygon := sorry

/-- Function to count the number of sides in a polygon -/
def count_sides (p : Polygon) : ℕ := sorry

/-- Predicate to check if a polygon can form all polygons from 3 to 100 sides -/
def can_form_all_polygons (p1 p2 : Polygon) : Prop :=
  ∀ n : ℕ, 3 ≤ n ∧ n ≤ 100 → ∃ p : Polygon, p ∈ (combine_polygons p1 p2) ∧ count_sides p = n

/-- Theorem stating the existence of two polygons that can form all polygons from 3 to 100 sides -/
theorem existence_of_polygons :
  ∃ p1 p2 : Polygon, can_form_all_polygons p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_polygons_l764_76404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l764_76428

open Real

/-- The original cosine function --/
noncomputable def f (x : ℝ) : ℝ := cos ((1/2) * x + π/6)

/-- The shifted cosine function --/
noncomputable def g (x φ : ℝ) : ℝ := cos ((1/2) * x - (1/2) * φ + π/6)

/-- Theorem stating the minimum positive value of φ --/
theorem min_shift_for_symmetry :
  ∃ (φ : ℝ), φ > 0 ∧
  (∀ (x : ℝ), g x φ = g (-x) φ) ∧
  (∀ (ψ : ℝ), ψ > 0 ∧ (∀ (x : ℝ), g x ψ = g (-x) ψ) → φ ≤ ψ) ∧
  φ = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l764_76428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_set_intersection_l764_76474

-- Define a symmetric set
def IsSymmetric (A : Set ℝ) : Prop :=
  ∀ x, x ∈ A → -x ∈ A

-- Define set A
def A : Set ℝ := {x | ∃ y, x = 2 * y ∨ x = 0 ∨ x = y^2 + y}

-- Define set B as natural numbers including 0
def B : Set ℝ := {x : ℝ | ∃ n : ℕ, x = n}

-- Theorem statement
theorem symmetric_set_intersection :
  IsSymmetric A → A ∩ B = {0, 6} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_set_intersection_l764_76474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_change_l764_76492

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_change 
  (P : ℝ) 
  (h1 : simple_interest P 4 5 = 1680) 
  (R' : ℝ) 
  (h2 : simple_interest P R' 4 = 1680) : 
  R' = 20 := by
  sorry

#check interest_rate_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_change_l764_76492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_eq_one_sixth_l764_76405

-- Define an obtuse angle
def is_obtuse_angle (α : ℝ) : Prop := Real.pi / 2 < α ∧ α < Real.pi

-- State the theorem
theorem sin_alpha_eq_one_sixth (α : ℝ) 
  (h_obtuse : is_obtuse_angle α) 
  (h_eq : 3 * Real.sin (2 * α) = Real.cos α) : 
  Real.sin α = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_eq_one_sixth_l764_76405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l764_76470

-- Define ObtuseTriangle as a structure
structure ObtuseTriangle (A B C : ℝ) : Prop where
  obtuse : A + B + C = Real.pi
  angle_sum_greater : A + B > Real.pi / 2

-- Define proposition p
def p : Prop := ∀ (A B C : ℝ) (h : ObtuseTriangle A B C), Real.sin A < Real.cos B

-- Define proposition q
def q : Prop := ∀ (x y : ℝ), x + y ≠ 2 → x ≠ -1 ∨ y ≠ 3

-- Theorem to prove
theorem correct_proposition : ¬p ∧ q := by
  constructor
  · -- Proof of ¬p
    sorry
  · -- Proof of q
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l764_76470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_point_on_circle_range_l764_76464

-- Define the line l passing through (-2,0) with inclination angle α
noncomputable def line_l (α : Real) : Set (Real × Real) :=
  {(x, y) | y = Real.tan α * (x + 2)}

-- Define the circle C
def circle_C : Set (Real × Real) :=
  {(x, y) | (x - 2)^2 + y^2 = 4}

-- Theorem for part (I)
theorem line_intersects_circle (α : Real) :
  (∃ p, p ∈ line_l α ∩ circle_C) ↔ 
  (0 ≤ α ∧ α ≤ π/6) ∨ (5*π/6 ≤ α ∧ α < π) :=
sorry

-- Theorem for part (II)
theorem point_on_circle_range (x y : Real) :
  (x, y) ∈ circle_C → -2 ≤ x + Real.sqrt 3 * y ∧ x + Real.sqrt 3 * y ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_point_on_circle_range_l764_76464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_for_even_g_l764_76437

-- Define the functions f and g
noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)
noncomputable def g (x φ : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3 + φ)

-- State the theorem
theorem phi_value_for_even_g (φ : ℝ) :
  (0 < φ) → (φ < Real.pi) →
  (∀ x, g x φ = g (-x) φ) →  -- g is an even function
  φ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_for_even_g_l764_76437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_test_results_l764_76419

noncomputable def scores : List ℝ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

noncomputable def average (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := average xs
  (xs.map (fun x => (x - μ) ^ 2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem shooting_test_results :
  average scores = 7 ∧ standardDeviation scores = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_test_results_l764_76419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_inequality_l764_76458

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/8 + y^2/2 = 1

-- Define the right vertex C
noncomputable def C : ℝ × ℝ := (2 * Real.sqrt 2, 0)

-- Define point A on the ellipse in the first quadrant
def A (m n : ℝ) : Prop := 
  ellipse m n ∧ m > 0 ∧ n > 0

-- Define point B symmetric to A about the origin
def B (m n : ℝ) : ℝ × ℝ := (-m, -n)

-- Define point D
noncomputable def D (m n : ℝ) : ℝ × ℝ := (m, (m - 2 * Real.sqrt 2) * n / (m + 2 * Real.sqrt 2))

-- Theorem statement
theorem ellipse_inequality (m n : ℝ) : 
  A m n → (m - 2 * Real.sqrt 2)^2 + n^2 < 
    ((m - 2 * Real.sqrt 2)^2 * ((10 * Real.sqrt 2 + 3 * m) / (2 * Real.sqrt 2 + m)) / 4) * 
    ((2 * Real.sqrt 2 + m)^2 + n^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_inequality_l764_76458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_unit_interval_sum_l764_76456

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 3

-- Define the theorem
theorem zero_in_unit_interval_sum (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  (∃ x, x ∈ Set.Icc (a : ℝ) (b : ℝ) ∧ f x = 0) → 
  b - a = 1 → 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_unit_interval_sum_l764_76456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_C_has_largest_shaded_area_l764_76417

-- Define the figures
noncomputable def figureA_shaded_area : ℝ := 4 - Real.pi
noncomputable def figureB_shaded_area : ℝ := 4 - Real.pi
noncomputable def figureC_shaded_area : ℝ := Real.pi - 2

-- Theorem statement
theorem figure_C_has_largest_shaded_area :
  figureC_shaded_area > figureA_shaded_area ∧
  figureC_shaded_area > figureB_shaded_area :=
by
  -- Split the conjunction
  apply And.intro
  -- Prove figureC_shaded_area > figureA_shaded_area
  · sorry
  -- Prove figureC_shaded_area > figureB_shaded_area
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_C_has_largest_shaded_area_l764_76417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l764_76465

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧  -- Smallest positive period is π
  (∀ x, f x ≤ 2) ∧                                                               -- Maximum value is 2
  (∀ x, f x ≥ -2) ∧                                                              -- Minimum value is -2
  (∀ k : ℤ, ∀ x y, k * Real.pi - Real.pi / 3 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + Real.pi / 6 → f x < f y) -- Monotonically increasing interval
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l764_76465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l764_76449

def a : ℕ → ℚ
  | 0 => 0  -- Add a case for 0 to make the function total
  | 1 => -1
  | 2 => 1
  | (n + 2) => (2 + (-1)^n) / 2 * a n

def S (n : ℕ) : ℚ :=
  if n % 2 = 0 then
    2 * (3/2)^(n/2) + 2 * (1/2)^(n/2) - 4
  else
    3 * (3/2)^((n-1)/2) + 2 * (1/2)^((n-1)/2) - 4

def b (n : ℕ) : ℚ := a (2*n - 1) + a (2*n)

theorem sequence_properties :
  (a 5 + a 6 = 2) ∧
  (∀ n : ℕ, n > 0 → (Finset.range n).sum a = S n) ∧
  (∀ i j k : ℕ, 0 < i ∧ i < j ∧ j < k →
    (b i + b k = 2 * b j ↔ i = 1 ∧ j = 2 ∧ k = 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l764_76449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_no_parallelogram_l764_76411

-- Define the ellipse M
def M (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
theorem ellipse_equation : 
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  M a b (-Real.sqrt 2) 0 ∧  -- Left focus at (-√2, 0)
  M a b (Real.sqrt 2) (Real.sqrt 3 / 3) ∧  -- Passes through (√2, √3/3)
  (∀ x y, M (Real.sqrt 3) 1 x y ↔ x^2 / 3 + y^2 = 1) :=
sorry

-- Define the non-existence of the parallelogram
theorem no_parallelogram :
  ¬∃ (A B C D : ℝ × ℝ),
    (A.2 = 2) ∧  -- Point A is on the line y = 2
    (M (Real.sqrt 3) 1 B.1 B.2) ∧
    (M (Real.sqrt 3) 1 C.1 C.2) ∧
    (M (Real.sqrt 3) 1 D.1 D.2) ∧  -- Points B, C, D are on the ellipse M
    ((D.2 - B.2) / (D.1 - B.1) = 1) ∧  -- Slope of BD is 1
    (A.1 - B.1 = C.1 - D.1) ∧ (A.2 - B.2 = C.2 - D.2) :=  -- ABCD is a parallelogram
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_no_parallelogram_l764_76411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_l764_76429

/-- Represents the fuel efficiency of a car in kilometers per gallon. -/
noncomputable def fuel_efficiency (distance : ℝ) (fuel : ℝ) : ℝ :=
  distance / fuel

/-- Theorem stating that the car's fuel efficiency is 40 km/gal given the provided conditions. -/
theorem car_fuel_efficiency :
  let distance := (150 : ℝ) -- kilometers
  let fuel := (3.75 : ℝ) -- gallons
  fuel_efficiency distance fuel = 40 := by
  -- Unfold the definitions
  unfold fuel_efficiency
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_l764_76429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_equal_dihedral_regular_faces_implies_regular_l764_76415

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  convex : Bool

/-- A regular polygon. -/
structure RegularPolygon where
  -- Add necessary fields here

/-- A dihedral angle of a polyhedron. -/
def DihedralAngle (p : ConvexPolyhedron) : Type := sorry

/-- A face of a polyhedron. -/
def Face (p : ConvexPolyhedron) : Type := sorry

/-- Predicate to check if all dihedral angles of a polyhedron are equal. -/
def AllDihedralAnglesEqual (p : ConvexPolyhedron) : Prop :=
  ∀ a b : DihedralAngle p, a = b

/-- Predicate to check if all faces of a polyhedron are regular polygons. -/
def AllFacesRegular (p : ConvexPolyhedron) : Prop :=
  ∀ f : Face p, ∃ r : RegularPolygon, sorry -- f is congruent to r

/-- Predicate to check if a polyhedron is regular. -/
def IsRegularPolyhedron (p : ConvexPolyhedron) : Prop := sorry

/-- Theorem stating that a convex polyhedron with equal dihedral angles and regular polygon faces is regular. -/
theorem convex_equal_dihedral_regular_faces_implies_regular 
  (p : ConvexPolyhedron) 
  (h_convex : p.convex)
  (h_dihedral : AllDihedralAnglesEqual p)
  (h_faces : AllFacesRegular p) : 
  IsRegularPolyhedron p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_equal_dihedral_regular_faces_implies_regular_l764_76415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eulers_formula_l764_76439

-- Define a triangle
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

-- Define the properties of a triangle
class TriangleProperties (t : Triangle) where
  circumcenter : EuclideanSpace ℝ (Fin 2)
  incenter : EuclideanSpace ℝ (Fin 2)
  circumradius : ℝ
  inradius : ℝ

-- State Euler's formula
theorem eulers_formula (t : Triangle) [tp : TriangleProperties t] :
  dist tp.circumcenter tp.incenter ^ 2 = tp.circumradius ^ 2 - 2 * tp.circumradius * tp.inradius :=
by
  sorry

#check eulers_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eulers_formula_l764_76439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sin_cos_l764_76476

theorem smallest_positive_solution_tan_sin_cos (x : ℝ) : 
  (x > 0 ∧ Real.tan (3 * x) - Real.sin (2 * x) = Real.cos (2 * x)) → x ≥ π / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sin_cos_l764_76476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_l764_76468

theorem divisible_by_three (d : Nat) : 
  d < 10 → (123450 + d) % 3 = 0 ↔ d ∈ ({0, 3, 6, 9} : Set Nat) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_l764_76468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_value_l764_76414

theorem cos_two_alpha_value (α : Real) 
  (h1 : 2 * Real.cos (2 * α) = Real.sin (α - π/4))
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.cos (2 * α) = Real.sqrt 15 / 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_value_l764_76414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_fifth_degree_polynomials_l764_76484

/-- A fifth degree polynomial function with leading coefficient 1 -/
def FifthDegreePolynomial (a b c d e : ℝ) : ℝ → ℝ := 
  fun x => x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- The difference between two fifth degree polynomials -/
def PolynomialDifference (p q : ℝ → ℝ) : ℝ → ℝ :=
  fun x => p x - q x

theorem max_intersection_points_fifth_degree_polynomials 
  (p q : ℝ → ℝ) 
  (hp : ∃ a b c d e, p = FifthDegreePolynomial a b c d e) 
  (hq : ∃ a b c d e, q = FifthDegreePolynomial a b c d e) 
  (hpq_diff : p ≠ q) :
  (∃ S : Finset ℝ, (∀ x ∈ S, p x = q x) ∧ S.card ≤ 4) ∧ 
  (∀ T : Finset ℝ, (∀ x ∈ T, p x = q x) → T.card ≤ 4) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_fifth_degree_polynomials_l764_76484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l764_76457

open Set
open Real

noncomputable def f (x : ℝ) : ℝ := 1 / (log x / log 10 - 1)

theorem domain_of_f :
  {x : ℝ | x > 0 ∧ log x / log 10 ≠ 1} = Ioo 0 10 ∪ Ioi 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l764_76457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_f_m_equals_neg_one_l764_76418

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^(2-m)

-- Define the domain of f
def domain (m : ℝ) : Set ℝ := Set.Icc (-3-m) (m^2-m)

-- State the theorem
theorem odd_function_f_m_equals_neg_one (m : ℝ) :
  (∀ x ∈ domain m, f m (-x) = -(f m x)) →  -- f is odd
  (∀ x ∈ domain m, f m x = x^(2-m)) →     -- definition of f
  m^2 - 2*m - 3 = 0 →                     -- condition from the problem
  f m m = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_f_m_equals_neg_one_l764_76418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_paving_stones_l764_76406

/-- The number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  (hall_length * hall_width * 100 / (stone_length * stone_width)).floor.toNat

/-- Theorem: The number of stones required to pave the given hall is 3600 -/
theorem hall_paving_stones :
  stones_required 36 15 3 5 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_paving_stones_l764_76406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_inclusion_l764_76431

-- Define the sets A, B, and C
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 2) + (x - 3)^(0 : ℕ)}
def B : Set ℝ := {x | 0 ≤ x - 1 ∧ x - 1 ≤ 4}
def C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 4 * m}

-- Theorem statement
theorem set_operations_and_inclusion :
  (A ∩ B = Set.Ioc 2 3 ∪ Set.Ioc 3 5) ∧
  (A ∪ B = Set.Ici 1) ∧
  (∀ m : ℝ, B ⊆ C m ↔ 5/4 < m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_inclusion_l764_76431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l764_76416

/-- The distance from the point (1,0) to the line y=x is √2/2 -/
theorem distance_point_to_line : 
  let point : ℝ × ℝ := (1, 0)
  let line_eq (x y : ℝ) := x = y
  |point.1 - point.2| / Real.sqrt 2 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l764_76416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_fourth_polyhedron_l764_76407

/-- Represents a sequence of polyhedra constructed from a regular tetrahedron -/
noncomputable def PolyhedraSequence : ℕ → ℝ
  | 0 => 1  -- P₀ has volume 1
  | n + 1 => PolyhedraSequence n + (3/4)^n * (1/2)  -- Volume increase at each step

/-- The volume of the 4th polyhedron in the sequence is 431/128 -/
theorem volume_of_fourth_polyhedron :
  PolyhedraSequence 4 = 431 / 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_fourth_polyhedron_l764_76407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_OA_l764_76459

/-- Parabola with focus F and point A -/
structure Parabola where
  p : ℝ
  F : ℝ × ℝ
  A : ℝ × ℝ
  h_p_pos : p > 0
  h_focus : F = (p/2, 0)
  h_on_parabola : A.2^2 = 2 * p * A.1
  h_angle : Real.cos (60 * π / 180) = (A.1 - F.1) / Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2)

/-- The distance between the origin and point A on the parabola -/
noncomputable def distance_OA (parab : Parabola) : ℝ :=
  Real.sqrt (parab.A.1^2 + parab.A.2^2)

theorem parabola_distance_OA (parab : Parabola) : 
  distance_OA parab = (Real.sqrt 21 / 2) * parab.p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_OA_l764_76459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_stride_difference_l764_76473

-- Define constants
def total_poles : ℕ := 61
def elmer_strides_per_gap : ℕ := 60
def oscar_leaps_per_gap : ℕ := 16
def total_distance_feet : ℝ := 7920

-- Define functions
def num_gaps : ℕ := total_poles - 1

def elmer_total_strides : ℕ := elmer_strides_per_gap * num_gaps
def oscar_total_leaps : ℕ := oscar_leaps_per_gap * num_gaps

noncomputable def elmer_stride_length : ℝ := total_distance_feet / elmer_total_strides
noncomputable def oscar_leap_length : ℝ := total_distance_feet / oscar_total_leaps

-- Theorem to prove
theorem leap_stride_difference : 
  oscar_leap_length - elmer_stride_length = 6.05 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_stride_difference_l764_76473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l764_76466

-- Define the speeds of the trains in km/h
noncomputable def speed1 : ℝ := 50
noncomputable def speed2 : ℝ := 40

-- Define the time to cross when running in the same direction (in seconds)
noncomputable def time_same_direction : ℝ := 50

-- Define the function to calculate the relative speed
noncomputable def relative_speed (s1 s2 : ℝ) (same_direction : Bool) : ℝ :=
  if same_direction then abs (s1 - s2) else s1 + s2

-- Define the function to convert km/h to m/s
noncomputable def kmh_to_ms (speed : ℝ) : ℝ := speed * (5/18)

-- Define the function to calculate the length of the trains
noncomputable def train_length (s1 s2 t : ℝ) : ℝ :=
  (kmh_to_ms (relative_speed s1 s2 true) * t) / 2

-- Define the function to calculate the time to cross
noncomputable def time_to_cross (s1 s2 l : ℝ) (same_direction : Bool) : ℝ :=
  (2 * l) / (kmh_to_ms (relative_speed s1 s2 same_direction))

-- Theorem statement
theorem train_crossing_time :
  let l := train_length speed1 speed2 time_same_direction
  abs (time_to_cross speed1 speed2 l false - 5.56) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l764_76466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l764_76434

theorem vector_decomposition (a b c d : ℝ × ℝ × ℝ) : 
  a = (1, 1, 3) →
  b = (2, -1, -6) →
  c = (5, 3, -1) →
  d = (-9, 2, 25) →
  d = (2 : ℝ) • a - (3 : ℝ) • b - c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l764_76434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_AEC_l764_76497

-- Define the square
noncomputable def square_side : ℝ := 2

-- Define the vertices of the square
noncomputable def A : ℝ × ℝ := (0, square_side)
noncomputable def B : ℝ × ℝ := (0, 0)
noncomputable def C : ℝ × ℝ := (square_side, 0)
noncomputable def D : ℝ × ℝ := (square_side, square_side)

-- Define C' on AD
noncomputable def C' : ℝ × ℝ := (square_side, 1/2)

-- Define E as the intersection of BC and AB
noncomputable def E : ℝ × ℝ := (8/7, 8/7)

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem stating the perimeter of triangle AEC' is 15/2
theorem perimeter_AEC'_is_15_over_2 :
  distance A E + distance E C' + distance C' A = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_AEC_l764_76497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_on_line_l764_76489

/-- Given points P, Q, and R in the xy-plane, prove that if R minimizes the total distance PR + RQ, then the y-coordinate of R is -3/5. -/
theorem minimize_distance_on_line (P Q R : ℝ × ℝ) : 
  P.1 = -2 ∧ P.2 = -3 ∧ 
  Q.1 = 3 ∧ Q.2 = 1 ∧ 
  R.1 = 1 ∧
  (∀ S : ℝ × ℝ, S.1 = 1 → (dist P R + dist R Q) ≤ (dist P S + dist S Q)) →
  R.2 = -3/5 := by
  sorry

#check minimize_distance_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_on_line_l764_76489
