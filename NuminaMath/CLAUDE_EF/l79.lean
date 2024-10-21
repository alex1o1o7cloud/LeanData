import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_abs_value_l79_7973

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := (2 - i) / (1 + i) - i^2016

theorem z_abs_value : Complex.abs z = Real.sqrt 10 / 2 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_abs_value_l79_7973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_A_l79_7992

-- Define the discriminant equation
noncomputable def discriminant (ξ : ℝ) : Prop := (1 - ξ)^2 + 4*ξ*(ξ + 2) = 0

-- Define the probabilities
noncomputable def prob_xi_neg_one : ℝ := 3/16
noncomputable def prob_xi_neg_one_fifth : ℝ := 1/4

-- Define the event A
def event_A (ξ : ℝ) : Prop := ξ = -1 ∨ ξ = -1/5

-- Theorem statement
theorem probability_of_A : 
  ∃ (ξ : ℝ), discriminant ξ ∧ 
  (∀ (x : ℝ), event_A x → (x = -1 → prob_xi_neg_one = 3/16) ∧ 
                         (x = -1/5 → prob_xi_neg_one_fifth = 1/4)) →
  prob_xi_neg_one + prob_xi_neg_one_fifth = 7/16 := by
  sorry

#check probability_of_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_A_l79_7992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_bound_l79_7991

def f (n : ℕ+) : ℚ := (Finset.range n.val).sum (λ i => 1 / (i + 1 : ℚ))

theorem harmonic_sum_bound :
  f 2 = 3/2 ∧
  f 4 > 2 ∧
  f 8 > 5/2 ∧
  f 16 > 3 →
  f 2048 > 13/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_bound_l79_7991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l79_7989

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2 + a * x

-- Theorem statement
theorem f_properties (a : ℝ) (h : a > 0) :
  -- Part 1: Tangent line equation when a = 2
  (∀ x y, x + y - 1 = 0 ↔ y = f 2 1 + (deriv (f 2)) 1 * (x - 1)) ∧
  -- Part 2: f(1/a) ≤ 0
  f a (1/a) ≤ 0 ∧
  -- Part 3: f has exactly one zero iff a = 1
  (∃! x, f a x = 0) ↔ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l79_7989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_3pi_l79_7970

-- Define the equation
def tanEquation (x : ℝ) : Prop := Real.tan x ^ 2 - 10 * Real.tan x + 2 = 0

-- Define the interval
def validInterval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2 * Real.pi

-- Theorem statement
theorem sum_of_roots_is_3pi :
  ∃ (S : Finset ℝ), (∀ x ∈ S, tanEquation x ∧ validInterval x) ∧
                    (S.sum id) = 3 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_3pi_l79_7970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l79_7909

/-- The sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) (lambda : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n ^ 2 = a (n + 1) * a (n - 1) + lambda * (a 2 - a 1) ^ 2

/-- Arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem part1 (a : ℕ → ℝ) (lambda d : ℝ) :
    RecurrenceSequence a lambda →
    ArithmeticSequence a d →
    d ≠ 0 →
    lambda = 1 := by sorry

theorem part2 (a : ℕ → ℝ) (lambda m : ℝ) :
    RecurrenceSequence a lambda →
    a 1 = 1 →
    a 2 = 2 →
    a 3 = 4 →
    (∃ r : ℝ, 3 ≤ r ∧ r ≤ 7 ∧ ∀ n : ℕ, n ≥ 1 → m * a n ≥ n - r) →
    m = 1 / 128 := by sorry

def PeriodicSequence (a : ℕ → ℝ) (T : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + T) = a n

def NonConstantSequence (a : ℕ → ℝ) : Prop :=
  ∃ m n : ℕ, a m ≠ a n

theorem part3 (a : ℕ → ℝ) (lambda : ℝ) :
    RecurrenceSequence a lambda →
    lambda ≠ 0 →
    NonConstantSequence a →
    (∃ T : ℕ, T ≥ 1 ∧ PeriodicSequence a T) →
    ∃ T : ℕ, T = 3 ∧ PeriodicSequence a T ∧
      ∀ S : ℕ, S ≥ 1 → PeriodicSequence a S → T ≤ S := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l79_7909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decrease_interval_of_composite_l79_7981

-- Define the function g(x) = (1/2)^x
noncomputable def g (x : ℝ) : ℝ := (1/2) ^ x

-- Define the property of f being symmetric to g with respect to y=x
def symmetric_to_g (f : ℝ → ℝ) : Prop := ∀ x y, f x = y ↔ g y = x

-- Define the interval of decrease for a function
def interval_of_decrease (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

-- State the theorem
theorem decrease_interval_of_composite (f : ℝ → ℝ) (h : symmetric_to_g f) :
  interval_of_decrease (fun x ↦ f (2*x - x^2)) 0 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decrease_interval_of_composite_l79_7981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l79_7971

theorem power_equation (a b : ℝ) (h1 : (10 : ℝ)^a = 8) (h2 : (10 : ℝ)^b = 2) : 
  (10 : ℝ)^(a - 2*b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l79_7971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_min_implies_a_neg_b_pos_l79_7988

/-- A function f(x) with a local minimum --/
noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - (1/2) * x^2 + b * x

/-- The existence of a local minimum for f(x) implies a < 0 and b > 0 --/
theorem local_min_implies_a_neg_b_pos (a b : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ IsLocalMin (f a b) x₀) → a < 0 ∧ b > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_min_implies_a_neg_b_pos_l79_7988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toll_road_distribution_l79_7922

structure Road where
  isToll : Bool

structure Town where
  roads : List Road

def Graph := List Town

structure Route where
  towns : List Town

def isTollRoad (r : Road) : Bool :=
  r.isToll

def countTollRoads (route : Route) : Nat :=
  route.towns.map (λ t => t.roads.filter isTollRoad |>.length) |>.sum

theorem toll_road_distribution (g : Graph) (southCapital northCapital : Town) 
    (h : ∀ r : Route, r.towns.head? = some southCapital → r.towns.getLast? = some northCapital → 
         countTollRoads r ≥ 10) :
    ∃ (distribution : Road → Fin 10), 
      ∀ r : Route, r.towns.head? = some southCapital → r.towns.getLast? = some northCapital →
        ∀ i : Fin 10, ∃ road ∈ r.towns.bind (λ t => t.roads), 
          isTollRoad road ∧ distribution road = i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toll_road_distribution_l79_7922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_modulo_13_l79_7972

theorem sum_remainder_modulo_13 (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_modulo_13_l79_7972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_are_intersecting_l79_7943

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 9

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (-1, 0)
def center2 : ℝ × ℝ := (2, 1)
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 10

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers > (radius2 - radius1) ∧
  distance_between_centers < (radius2 + radius1) := by
  sorry

-- Additional theorem to state the conclusion
theorem circles_are_intersecting : 
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_are_intersecting_l79_7943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_monotonicity_l79_7985

theorem exponential_monotonicity (a b : ℝ) (h : a > b) : (2 : ℝ)^a > (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_monotonicity_l79_7985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_tangent_difference_l79_7950

/-- The difference between the distance from a point on an ellipse to its left focus
    and the distance from that point to where its tangent touches an inscribed circle --/
theorem ellipse_focus_tangent_difference
  (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0)
  (A : ℝ × ℝ) (h_A_on_ellipse : (A.1^2 / a^2) + (A.2^2 / b^2) = 1)
  (h_A_first_quadrant : A.1 ≥ 0 ∧ A.2 ≥ 0)
  (P : ℝ × ℝ) (h_P_on_circle : P.1^2 + P.2^2 = b^2)
  (h_AP_tangent : (A.1 - P.1) * P.1 + (A.2 - P.2) * P.2 = 0) :
  let F := (Real.sqrt (a^2 - b^2), 0)
  ‖A - F‖ - ‖A - P‖ = a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_tangent_difference_l79_7950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_expressions_equality_l79_7919

theorem sqrt_expressions_equality : 
  (Real.sqrt 12 - (8 : Real)^(1/3) + abs (-Real.sqrt 3) - (Real.sqrt 3 - 2)^0 = 3 * Real.sqrt 3 - 3) ∧
  (Real.sqrt 27 - Real.sqrt 2 * Real.sqrt 8 + Real.sqrt ((-4)^2) = 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_expressions_equality_l79_7919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_transformed_plane_l79_7955

/-- The similarity transformation of a plane equation Ax + By + Cz + D = 0
    with coefficient k and center at the origin is Ax + By + Cz + k*D = 0 -/
def transform_plane (A B C D k : ℝ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + k * D = 0

/-- The point A with coordinates (3, 2, 4) -/
def point_A : ℝ × ℝ × ℝ := (3, 2, 4)

/-- The original plane equation: 2x - 3y + z - 6 = 0 -/
def original_plane (x y z : ℝ) : Prop :=
  2 * x - 3 * y + z - 6 = 0

/-- The similarity transformation coefficient -/
noncomputable def k : ℝ := 2 / 3

theorem point_on_transformed_plane :
  transform_plane 2 (-3) 1 (-6) k point_A.1 point_A.2.1 point_A.2.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_transformed_plane_l79_7955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_decreasing_is_decreasing_l79_7997

/-- Given a decreasing function f on [a, b], its inverse is decreasing on [f(b), f(a)] -/
theorem inverse_of_decreasing_is_decreasing 
  {f : ℝ → ℝ} {a b : ℝ} (h_dec : ∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f x > f y) 
  (h_bij : Function.Bijective f) :
  ∀ x y, x ∈ Set.Icc (f b) (f a) → y ∈ Set.Icc (f b) (f a) → x < y → 
    (Function.invFun f) x > (Function.invFun f) y := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_decreasing_is_decreasing_l79_7997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_x_range_l79_7910

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 1|

-- Theorem for the minimum value of f
theorem f_min_value : 
  ∀ x : ℝ, f x ≥ 2 ∧ ∃ y : ℝ, f y = 2 :=
sorry

-- Theorem for the range of x
theorem x_range (h : ∀ a b : ℝ, |2*a + b| + |a| - (1/2) * |a + b| * f x ≥ 0) : 
  x ∈ Set.Icc (-1/2) (1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_x_range_l79_7910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l79_7980

/-- Given vectors a and b, if (a + b) ⊥ (2a - λb), then λ = 2/9 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (lambda : ℝ) 
  (ha : a = (2, 1)) 
  (hb : b = (-3, 2)) 
  (h_perp : (a + b) • (2 • a - lambda • b) = 0) : 
  lambda = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l79_7980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_m_value_when_max_is_3_l79_7999

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := (Real.sin (Real.pi / 2 + x) - Real.sin x) ^ 2 + m

-- Theorem for the minimum positive period
theorem min_positive_period (m : ℝ) : 
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f x m = f (x + T) m) ∧ 
  (∀ T' : ℝ, T' > 0 → (∀ x, f x m = f (x + T') m) → T ≤ T') ∧ 
  T = Real.pi :=
sorry

-- Theorem for the value of m when maximum of f(x) is 3
theorem m_value_when_max_is_3 : 
  ∃ (m : ℝ), (∀ x, f x m ≤ 3) ∧ (∃ x, f x m = 3) ∧ m = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_m_value_when_max_is_3_l79_7999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_becky_winning_strategy_l79_7962

/-- The star game for two players -/
def StarGame (n : ℕ) : Prop :=
  n ≥ 5 ∧ 
  ∃ (strategy : Type), 
    (∃ (wins : strategy → Prop),
      ∀ (arrangement : Fin n → ℕ), 
        (∀ i : Fin n, arrangement i ∈ Finset.range n.succ) →
        (∀ i j : Fin n, i ≠ j → arrangement i ≠ arrangement j) →
        ∃ (s : strategy), wins s)

/-- Becky has a winning strategy iff n ≡ 2 (mod 4) -/
theorem becky_winning_strategy (n : ℕ) : 
  StarGame n ↔ (∃ k : ℕ, n = 4*k + 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_becky_winning_strategy_l79_7962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_less_than_Q_l79_7904

-- Define the variables and functions
variable (a : ℝ)

noncomputable def P (a : ℝ) : ℝ := Real.sqrt (a + 2) + Real.sqrt (a + 5)
noncomputable def Q (a : ℝ) : ℝ := Real.sqrt (a + 3) + Real.sqrt (a + 4)

-- State the theorem
theorem P_less_than_Q (h : a ≥ 0) : P a < Q a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_less_than_Q_l79_7904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_division_plus_third_l79_7938

/-- Proves that the division of repeating decimals 0.363636... by 0.090909...,
    plus 1/3, equals 13/3 -/
theorem repeating_decimal_division_plus_third : 
  (36 / 99) / (9 / 99) + (1 / 3) = 13 / 3 := by
  -- Convert repeating decimals to fractions
  have h1 : 0.363636 = 36 / 99 := by sorry
  have h2 : 0.090909 = 9 / 99 := by sorry
  
  -- Simplify fractions
  have h3 : 36 / 99 = 4 / 11 := by sorry
  have h4 : 9 / 99 = 1 / 11 := by sorry
  
  -- Perform division
  have h5 : (4 / 11) / (1 / 11) = 4 := by sorry
  
  -- Add 1/3 to the result
  have h6 : 4 + 1/3 = 13/3 := by sorry
  
  -- Combine steps
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_division_plus_third_l79_7938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l79_7947

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x - 2

-- State the theorem
theorem function_properties :
  (∀ x : ℝ, f (x + 1) = x - 1) →
  (∀ x : ℝ, f x = x - 2) ∧
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 ∧ x ≤ y → |f x| ≤ |f y|) ∧
  (∀ x y : ℝ, x ≤ 2 ∧ y ≤ 2 ∧ x ≤ y → |f x| ≥ |f y|) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l79_7947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patrol_sum_problem_solution_l79_7923

/-- Represents a police officer's patrol rate -/
structure PatrolRate where
  streets : ℕ
  hours : ℕ
  streets_per_hour : ℚ
  rate_calculation : streets_per_hour = (streets : ℚ) / hours

/-- Calculates the total streets patrolled in one hour by two officers -/
def total_streets_per_hour (officer_a : PatrolRate) (officer_b : PatrolRate) : ℚ :=
  officer_a.streets_per_hour + officer_b.streets_per_hour

theorem patrol_sum (officer_a : PatrolRate) (officer_b : PatrolRate) :
  total_streets_per_hour officer_a officer_b =
  (officer_a.streets : ℚ) / officer_a.hours + (officer_b.streets : ℚ) / officer_b.hours := by
  sorry

/-- The specific problem instance -/
def officer_a : PatrolRate := {
  streets := 36,
  hours := 4,
  streets_per_hour := 9,
  rate_calculation := by norm_num
}

def officer_b : PatrolRate := {
  streets := 55,
  hours := 5,
  streets_per_hour := 11,
  rate_calculation := by norm_num
}

theorem problem_solution :
  total_streets_per_hour officer_a officer_b = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_patrol_sum_problem_solution_l79_7923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_slope_theorem_l79_7965

/-- A parabola with equation y² = 2px -/
structure Parabola where
  p : ℝ

/-- The focus of a parabola -/
noncomputable def focus (c : Parabola) : ℝ × ℝ := (c.p / 2, 0)

/-- Slope of a line given two points -/
noncomputable def lineSlope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

/-- The theorem statement -/
theorem parabola_slope_theorem (c : Parabola) :
  let p : ℝ × ℝ := (-2, 1)  -- Point on axis of symmetry
  let f := focus c  -- Focus of the parabola
  c.p = 4 →         -- Derived from P being on axis of symmetry
  lineSlope p f = -1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_slope_theorem_l79_7965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_and_minimum_b_l79_7930

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

noncomputable def g (a x : ℝ) : ℝ := -x + 2 * a / (5 * a^2 - 4 * a + 1)

theorem fixed_points_and_minimum_b (a b : ℝ) (h : a ≠ 0) :
  -- Part 1: Fixed points when a = 1 and b = 3
  (a = 1 ∧ b = 3 → is_fixed_point (f 1 3) (-1) ∧ is_fixed_point (f 1 3) (-2)) ∧
  -- Part 2: Condition for always having two fixed points
  (∀ b : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_fixed_point (f a b) x₁ ∧ is_fixed_point (f a b) x₂) ↔ 0 < a ∧ a < 1) ∧
  -- Part 3: Minimum value of b
  (∀ b : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    is_fixed_point (f a b) x₁ ∧ 
    is_fixed_point (f a b) x₂ ∧ 
    g a ((x₁ + x₂) / 2) = (f a b x₁ + f a b x₂) / 2) →
    b ≥ -2 ∧ ∃ b₀ : ℝ, b₀ = -2 ∧ 
      ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
        is_fixed_point (f a b₀) x₁ ∧ 
        is_fixed_point (f a b₀) x₂ ∧ 
        g a ((x₁ + x₂) / 2) = (f a b₀ x₁ + f a b₀ x₂) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_and_minimum_b_l79_7930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_five_parts_equality_l79_7924

theorem sqrt_five_parts_equality :
  ∀ (a b : ℝ),
  (∃ (k : ℤ), (a : ℝ) = ↑k ∧ ↑k ≤ Real.sqrt 5 ∧ Real.sqrt 5 < ↑k + 1) →  -- a is the integer part of √5
  (b = Real.sqrt 5 - a) →                                                -- b is the decimal part of √5
  a - 2*b + Real.sqrt 5 = 6 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_five_parts_equality_l79_7924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_gravelling_cost_l79_7977

theorem road_gravelling_cost (lawn_length lawn_width road_width gravelling_cost_per_sqm : ℝ) 
  (h1 : lawn_length = 70)
  (h2 : lawn_width = 30)
  (h3 : road_width = 5)
  (h4 : gravelling_cost_per_sqm = 4) : 
  lawn_length * road_width + lawn_width * road_width - road_width * road_width * gravelling_cost_per_sqm = 1900 := by
  -- Define intermediate calculations
  let road_area_length := lawn_length * road_width
  let road_area_width := lawn_width * road_width
  let intersection_area := road_width * road_width
  let total_road_area := road_area_length + road_area_width - intersection_area
  let total_cost := total_road_area * gravelling_cost_per_sqm
  
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_gravelling_cost_l79_7977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_equals_five_l79_7934

theorem t_equals_five (P Q R S T U : ℕ) : 
  (Finset.range 7 \ {0} = {P, Q, R, S, T, U}) →
  (P + Q = 5) →
  (R = S + 5 ∨ S = R + 5) →
  (T > U) →
  (T = 5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_equals_five_l79_7934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_digit_swap_count_l79_7933

/-- Given that Jessica is 30 years old and Mark is older,
    this theorem states that there are exactly 25 possible pairs of
    Mark's current age (x) and the number of years (m) until their ages
    satisfy the digit-swapping property. -/
theorem age_digit_swap_count :
  let count := Finset.filter (fun p : ℕ × ℕ =>
    let (x, m) := p
    x > 30 ∧ 
    30 + m ≥ 10 ∧ 30 + m < 100 ∧
    x + m ≥ 10 ∧ x + m < 100 ∧
    (30 + m) % 10 * 10 + (30 + m) / 10 = x + m) (Finset.product (Finset.range 100) (Finset.range 100))
  Finset.card count = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_digit_swap_count_l79_7933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_quadruples_l79_7951

/-- The function to be minimized -/
def f (a b c d : ℤ) : ℤ :=
  a^2 + b^2 + c^2 + d^2 + a*b + c*d - a*c - b*d - b*c

/-- The set of quadruples that achieve the minimum value -/
def min_quadruples : Set (ℤ × ℤ × ℤ × ℤ) :=
  {(0, 1, -1, 1), (0, -1, 1, -1), (1, -1, 1, 0),
   (-1, 1, -1, 0), (1, 0, 0, 1), (-1, 0, 0, -1)}

/-- The theorem stating the minimum value and the quadruples that achieve it -/
theorem min_value_and_quadruples :
  ∀ a b c d : ℤ, a*d - b*c = 1 →
  (f a b c d ≥ 2 ∧
   (f a b c d = 2 ↔ (a, b, c, d) ∈ min_quadruples)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_quadruples_l79_7951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_endpoint_l79_7978

noncomputable def point := ℝ × ℝ

noncomputable def distance (p q : point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def parallel_to_x_axis (p q : point) : Prop :=
  p.2 = q.2

theorem segment_endpoint (A B : point) :
  parallel_to_x_axis A B →
  distance A B = 3 →
  A = (1, 2) →
  (B = (4, 2) ∨ B = (-2, 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_endpoint_l79_7978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_alphabetical_path_length_l79_7964

-- Define the graph structure
structure HikingMap where
  vertices : Set Char
  edges : Set (Char × Char)

-- Define the property of a valid path
def validPath (g : HikingMap) (path : List Char) : Prop :=
  path.Nodup ∧
  path.Sorted (· < ·) ∧
  ∀ i, i + 1 < path.length → (path[i]!, path[i+1]!) ∈ g.edges

-- Define the theorem
theorem max_alphabetical_path_length (g : HikingMap) :
  (g.vertices = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'}) →
  (∀ v ∈ g.vertices, ∀ w ∈ g.vertices, (v, w) ∈ g.edges → v < w) →
  (∃ path, validPath g path ∧ path.length = 10) ∧
  (∀ path, validPath g path → path.length ≤ 10) := by
  sorry

#check max_alphabetical_path_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_alphabetical_path_length_l79_7964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l79_7996

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x + 1 / (x + 1) - 1 / 2 else 1

theorem f_inequality_solution_set :
  ∀ x : ℝ, f (6 - x^2) > f x ↔ -Real.sqrt 5 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l79_7996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_identification_l79_7984

-- Define the four functions
noncomputable def f1 (x : ℝ) : ℝ := 2 * x
noncomputable def f2 (x : ℝ) : ℝ := x / 2
noncomputable def f3 (x : ℝ) : ℝ := 2 / x
noncomputable def f4 (x : ℝ) : ℝ := 2 / (x - 1)

-- Define inverse proportionality
def is_inverse_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x * x = k

-- Theorem statement
theorem inverse_proportion_identification :
  ¬ is_inverse_proportional f1 ∧
  ¬ is_inverse_proportional f2 ∧
  is_inverse_proportional f3 ∧
  ¬ is_inverse_proportional f4 := by
  sorry

#check inverse_proportion_identification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_identification_l79_7984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_is_85_percent_l79_7925

-- Define the class percentages and their corresponding test averages
def group1_percentage : Float := 0.45
def group1_average : Float := 0.95
def group2_percentage : Float := 0.50
def group2_average : Float := 0.78
def group3_percentage : Float := 1 - group1_percentage - group2_percentage
def group3_average : Float := 0.60

-- Define the overall class average
def class_average : Float :=
  group1_percentage * group1_average +
  group2_percentage * group2_average +
  group3_percentage * group3_average

-- Theorem to prove
theorem class_average_is_85_percent :
  Float.round (class_average * 100) = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_is_85_percent_l79_7925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_theta_value_l79_7944

noncomputable section

open Real

theorem angle_theta_value (θ m : ℝ) 
  (h1 : 4 * (sin θ)^2 - 4 * m * sin θ + 2 * m - 1 = 0)
  (h2 : 4 * (cos θ)^2 - 4 * m * cos θ + 2 * m - 1 = 0) 
  (h3 : 3 * π / 2 < θ) (h4 : θ < 2 * π) :
  θ = 5 * π / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_theta_value_l79_7944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_c_unique_l79_7956

/-- Functions s and c satisfying the given conditions -/
noncomputable def s : ℝ → ℝ := sorry
noncomputable def c : ℝ → ℝ := sorry

/-- Conditions for s and c -/
axiom s_deriv (x : ℝ) : deriv s x = c x
axiom c_deriv (x : ℝ) : deriv c x = -s x
axiom s_init : s 0 = 0
axiom c_init : c 0 = 1

/-- Theorem: s and c are unique and equal to sine and cosine -/
theorem s_c_unique : s = Real.sin ∧ c = Real.cos := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_c_unique_l79_7956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l79_7957

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

def valid_k : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

theorem equation_solution (x : ℝ) :
  (4 * (floor x : ℝ) = 25 * frac x - 4.5) ↔
  (∃ k : ℤ, k ∈ valid_k ∧ x = k + (8 * k + 9) / 50) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l79_7957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_fraction_sum_l79_7945

theorem max_fraction_sum (n : ℕ) (hn : n > 1) :
  let max_sum := 1 - 1 / ((2*n/3 + 7/6 : ℚ).floor * ((2*n/3 + 7/6 : ℚ).floor * (n - (2*n/3 + 1/6 : ℚ).floor) + 1))
  ∀ (a b c d : ℕ), b ≠ 0 → d ≠ 0 → 
    (a : ℚ) / b + (c : ℚ) / d < 1 → 
    a + c ≤ n → 
    (a : ℚ) / b + (c : ℚ) / d ≤ max_sum :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_fraction_sum_l79_7945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_P_l79_7946

-- Define the points A, B, and C
noncomputable def A : ℝ × ℝ := (0, 3)
noncomputable def B : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def C : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem locus_of_point_P (x y : ℝ) :
  let P : ℝ × ℝ := (x, y)
  distance P A = distance P B + distance P C →
  x^2 + (y - 1)^2 = 4 ∧ y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_P_l79_7946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_is_odd_and_period_pi_l79_7963

-- Define the tangent function as noncomputable
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- State the theorem
theorem tan_is_odd_and_period_pi :
  (∀ x : ℝ, tan (-x) = -tan x) ∧
  (∀ x : ℝ, tan (x + π) = tan x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_is_odd_and_period_pi_l79_7963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_into_three_equal_sum_sets_characterization_l79_7937

def is_divisible_into_three_equal_sum_sets (n : ℕ) : Prop :=
  ∃ (A B C : Finset ℕ), 
    A ∪ B ∪ C = Finset.range n.succ ∧ 
    A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
    (A.sum id = B.sum id) ∧ (B.sum id = C.sum id)

theorem divisible_into_three_equal_sum_sets_characterization (n : ℕ) :
  is_divisible_into_three_equal_sum_sets n ↔ 
  (n % 3 = 0 ∨ n % 3 = 2) ∧ n ≥ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_into_three_equal_sum_sets_characterization_l79_7937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l79_7969

-- Define a power function
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- State the theorem
theorem power_function_through_point :
  ∃ a : ℝ, power_function a 2 = Real.sqrt 2 ∧ a = 1/2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l79_7969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimal_third_side_l79_7911

/-- Given a triangle with two sides summing to d and enclosing an angle γ,
    the third side is minimized when the two given sides are equal. -/
theorem triangle_minimal_third_side 
  (d : ℝ) (γ : ℝ) (h_d : d > 0) (h_γ : 0 < γ ∧ γ < π) :
  ∃ (a b c : ℝ),
    a + b = d ∧
    c = Real.sqrt (2 * a * b * (1 - Real.cos γ)) ∧
    ∀ (a' b' : ℝ), 
      a' + b' = d → 
      Real.sqrt (2 * a' * b' * (1 - Real.cos γ)) ≥ c ∧
      Real.sqrt (2 * a' * b' * (1 - Real.cos γ)) = c ↔ a' = b' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimal_third_side_l79_7911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_a_value_l79_7931

theorem max_cos_a_value (a b c d : Real) 
  (h1 : Real.sin a = Real.tan (Real.pi / 2 - b))
  (h2 : Real.sin b = Real.tan (Real.pi / 2 - c))
  (h3 : Real.sin c = Real.tan (Real.pi / 2 - d))
  (h4 : Real.sin d = Real.tan (Real.pi / 2 - a)) :
  ∃ (max_cos_a : Real), 
    (∀ (x : Real), Real.cos a ≤ x → x ≤ max_cos_a) ∧ 
    max_cos_a = Real.sqrt ((Real.sqrt 5 - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_a_value_l79_7931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ray_l79_7917

-- Define the fixed points M and N
def M : ℝ × ℝ := (1, 0)
def N : ℝ × ℝ := (3, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the set of points P satisfying the condition
def trajectory : Set (ℝ × ℝ) :=
  {P | |distance P M - distance P N| = distance M N}

-- Define what it means for a set to be a ray
def IsRay (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (origin : ℝ × ℝ) (direction : ℝ × ℝ), 
    S = {p | ∃ t : ℝ, t ≥ 0 ∧ p = origin + t • direction}

-- Theorem statement
theorem trajectory_is_ray : 
  ∃ (r : Set (ℝ × ℝ)), IsRay r ∧ r = trajectory := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ray_l79_7917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l79_7976

theorem trig_identity (α β : ℝ) 
  (h1 : Real.sin α ^ 4 / Real.sin β ^ 2 + Real.cos α ^ 4 / Real.cos β ^ 2 = 1)
  (h2 : Real.sin α ≠ 0)
  (h3 : Real.cos α ≠ 0)
  (h4 : Real.sin β ≠ 0)
  (h5 : Real.cos β ≠ 0) :
  Real.sin β ^ 4 / Real.sin α ^ 2 + Real.cos β ^ 4 / Real.cos α ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l79_7976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l79_7958

open Real

theorem trigonometric_identity (α β γ : ℝ) 
  (h : cos (2 * α) = cos (2 * β) * cos (2 * γ)) : 
  1 + (tan (π/2 - (α + β)) * tan (π/2 - (α - β))) = 1 / (sin γ)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l79_7958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_properties_l79_7990

/-- Represents the number of students in a grade --/
def GradePopulation : Type := Nat

/-- Represents the number of students to be sampled --/
def SampleSize : Type := Nat

/-- Calculates the number of students to be sampled from a grade --/
def stratifiedSampleSize (gradePopulation : Nat) (totalPopulation : Nat) (totalSampleSize : Nat) : Nat :=
  (gradePopulation * totalSampleSize) / totalPopulation

/-- Theorem stating the properties of stratified sampling --/
theorem stratified_sampling_properties 
  (firstGradePopulation secondGradePopulation : Nat)
  (totalSampleSize : Nat)
  (h1 : firstGradePopulation = 1000)
  (h2 : secondGradePopulation = 1080)
  (h3 : totalSampleSize = 208) :
  let totalPopulation := firstGradePopulation + secondGradePopulation
  let firstGradeSample := stratifiedSampleSize firstGradePopulation totalPopulation totalSampleSize
  let secondGradeSample := stratifiedSampleSize secondGradePopulation totalPopulation totalSampleSize
  -- 1. Students from both grades can be selected
  (firstGradeSample > 0 ∧ secondGradeSample > 0) ∧
  -- 2. The number of students selected from each grade is proportional to the grade's population
  (firstGradeSample = 100 ∧ secondGradeSample = 108) ∧
  -- 3. The probability of selection is equal for students in both grades
  ((firstGradeSample : Rat) / firstGradePopulation = (secondGradeSample : Rat) / secondGradePopulation) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_properties_l79_7990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_expression_l79_7979

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 + 2*x + 2) :
  ∀ x, f x = x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_expression_l79_7979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l79_7974

-- Define the variables
noncomputable def a : ℝ := Real.sqrt 0.6
noncomputable def b : ℝ := (0.5 : ℝ) ^ (1/4)
noncomputable def c : ℝ := Real.log 0.4

-- State the theorem
theorem order_of_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l79_7974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_three_for_concatenated_range_l79_7902

def concatenate_range (start : Nat) (end_ : Nat) : Nat :=
  sorry

def highest_power_of_three (n : Nat) : Nat :=
  sorry

theorem highest_power_of_three_for_concatenated_range :
  highest_power_of_three (concatenate_range 31 68) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_three_for_concatenated_range_l79_7902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_1040_l79_7942

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The sum function as described in the problem -/
def sum (X Y Z : Digit) : ℕ :=
  111 * X.val + 10 * Y.val + 10 * Z.val + X.val + X.val + 11

theorem max_sum_is_1040 :
  ∃ (X Y Z : Digit),
    X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
    (∀ (W : Digit), W.val ≤ X.val) ∧
    sum X Y Z ≤ 1040 ∧
    (∀ (A B C : Digit),
      A ≠ B → A ≠ C → B ≠ C →
      (∀ (W : Digit), W.val ≤ A.val) →
      sum A B C ≤ sum X Y Z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_is_1040_l79_7942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_is_15_4_l79_7921

/-- The binomial expansion of (√x - 1/(2x))^6 where the 4th term has the largest coefficient -/
noncomputable def binomial_expansion (x : ℝ) : ℝ := (Real.sqrt x - 1 / (2 * x)) ^ 6

/-- The condition that the 4th term has the largest coefficient -/
axiom fourth_term_largest : ∀ x > 0, ∃ k > 0, ∀ i ≠ 3, 
  |Finset.sum (Finset.range 7) (λ j => Nat.choose 6 j * (Real.sqrt x)^(6-j) * (-1/(2*x))^j)| < k ∧
  |Finset.sum (Finset.range 7) (λ j => Nat.choose 6 j * (Real.sqrt x)^(6-j) * (-1/(2*x))^j)| = k

/-- The theorem stating that the constant term of the expansion is 15/4 -/
theorem constant_term_is_15_4 : 
  ∀ x > 0, Finset.sum (Finset.range 7) (λ j => Nat.choose 6 j * (Real.sqrt x)^(6-j) * (-1/(2*x))^j * x^(3/2*j-3)) = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_is_15_4_l79_7921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_limit_b_proof_l79_7966

/-- The upper limit of b given the conditions -/
def upper_limit_b : ℕ := 120

theorem upper_limit_b_proof (a b : ℕ) (h1 : 39 < a ∧ a < 51) (h2 : 49 < b) 
  (h3 : |((a : ℚ) / b) - 1/3| < 1e-14) : b ≤ upper_limit_b := by
  sorry

#check upper_limit_b_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_limit_b_proof_l79_7966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_grade_approx_72_7_l79_7903

/-- Calculate the average grade for a two-year period given the number of courses and average grades for each year -/
noncomputable def average_grade_two_years (courses_year1 courses_year2 : ℕ) (avg_grade_year1 avg_grade_year2 : ℝ) : ℝ :=
  let total_points := (courses_year1 : ℝ) * avg_grade_year1 + (courses_year2 : ℝ) * avg_grade_year2
  let total_courses := courses_year1 + courses_year2
  total_points / (total_courses : ℝ)

theorem average_grade_approx_72_7 :
  let courses_year1 : ℕ := 5
  let courses_year2 : ℕ := 6
  let avg_grade_year1 : ℝ := 40
  let avg_grade_year2 : ℝ := 100
  let result := average_grade_two_years courses_year1 courses_year2 avg_grade_year1 avg_grade_year2
  abs (result - 72.7) < 0.05 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_grade_approx_72_7_l79_7903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_first_21_multiples_of_8_l79_7967

theorem average_of_first_21_multiples_of_8 : 
  let multiples : List ℕ := (List.range 21).map (fun i => (i + 1) * 8)
  (multiples.sum / multiples.length : ℚ) = 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_first_21_multiples_of_8_l79_7967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_conditions_existence_of_positive_x0_l79_7935

noncomputable section

-- Use unicode lambda (λ) instead of the Greek letter
variables (l a : ℝ)

-- Use parentheses to clarify the function definition
def f (x : ℝ) : ℝ := x^2 / (l + x) - a * Real.log x

theorem min_value_conditions (hl : l > 0) (ha : a > 0) :
  (∀ x > 0, f l a x ≥ 0) ∧ f l a l = 0 →
  l = Real.exp (2/3) ∧ a = (3/4) * Real.exp (2/3) :=
sorry

theorem existence_of_positive_x0 (hl : l > 0) (ha : a > 0) :
  ∃ x₀ : ℝ, ∀ x > x₀, f l a x > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_conditions_existence_of_positive_x0_l79_7935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_increasing_function_l79_7954

noncomputable def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

noncomputable def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

noncomputable def sin_half (x : ℝ) : ℝ := Real.sin (x / 2)
noncomputable def sin (x : ℝ) : ℝ := Real.sin x
noncomputable def neg_tan (x : ℝ) : ℝ := -Real.tan x
noncomputable def neg_cos_double (x : ℝ) : ℝ := -Real.cos (2 * x)

theorem periodic_increasing_function :
  (has_period neg_cos_double π ∧ is_increasing_on neg_cos_double 0 (π / 2)) ∧
  ¬(has_period sin_half π ∧ is_increasing_on sin_half 0 (π / 2)) ∧
  ¬(has_period sin π ∧ is_increasing_on sin 0 (π / 2)) ∧
  ¬(has_period neg_tan π ∧ is_increasing_on neg_tan 0 (π / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_increasing_function_l79_7954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_red_ball_count_l79_7953

/-- Represents the colors of the balls -/
inductive Color
  | Red
  | Green
  | Blue
  | Yellow
  | Black
  | White

/-- The total number of balls initially -/
def totalBalls : ℕ := 10000

/-- The ratio of balls for each color -/
def colorRatio : Color → ℕ
  | Color.Red => 15
  | Color.Green => 13
  | Color.Blue => 17
  | Color.Yellow => 9
  | Color.Black => 7
  | Color.White => 23

/-- The sum of all color ratios -/
def totalRatio : ℕ := 
  (colorRatio Color.Red) + (colorRatio Color.Green) + (colorRatio Color.Blue) +
  (colorRatio Color.Yellow) + (colorRatio Color.Black) + (colorRatio Color.White)

/-- The initial number of balls of a given color -/
def initialBallCount (c : Color) : ℕ :=
  (colorRatio c * totalBalls) / totalRatio

/-- The number of balls removed or added for each color -/
def ballsChanged : Color → ℤ
  | Color.Red => 400
  | Color.Green => -250
  | Color.Yellow => -100
  | Color.Black => 200
  | Color.White => -500
  | Color.Blue => 0

/-- The final number of balls of a given color after changes -/
def finalBallCount (c : Color) : ℕ :=
  (initialBallCount c : ℤ) + ballsChanged c |>.toNat

theorem final_red_ball_count :
  finalBallCount Color.Red = 2185 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_red_ball_count_l79_7953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_fraction_equals_25_l79_7983

theorem power_fraction_equals_25 (n : ℕ) (x : ℝ) (h : x^(2*n) = 5) :
  (2*x^(3*n))^2 / (4*x^(2*n)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_fraction_equals_25_l79_7983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l79_7928

noncomputable def f (x : ℝ) : ℝ := 2 * x / (x - 1)

theorem range_of_f :
  ∀ y : ℝ, y ≠ 2 ↔ ∃ x : ℝ, x ≠ 1 ∧ f x = y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l79_7928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_max_value_condition_range_of_a_l79_7952

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x

-- Part 1
theorem tangent_line_at_one (x y : ℝ) :
  HasDerivAt (f 1) (1 + 1/x) x →
  f 1 1 = 1 →
  (2 * x - y - 1 = 0) ↔ y - 1 = 2 * (x - 1) :=
by sorry

-- Part 2
theorem max_value_condition (a : ℝ) :
  (∀ x > 0, f a x ≤ -2) →
  (∃ x > 0, f a x = -2) →
  a = -Real.exp 1 :=
by sorry

-- Part 3
theorem range_of_a (a : ℝ) :
  a < 0 →
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≤ g a x) →
  a ∈ Set.Icc ((1 - 2 * Real.exp 1) / ((Real.exp 1)^2 - Real.exp 1)) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_max_value_condition_range_of_a_l79_7952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_price_l79_7968

/-- Calculates the cost price of an article given its sale price (including tax),
    sales tax rate, and profit rate. -/
noncomputable def cost_price (sale_price : ℝ) (tax_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  sale_price / (1 + tax_rate) / (1 + profit_rate)

/-- Theorem stating that the cost price of an article is approximately 495.57
    given the conditions in the problem. -/
theorem article_cost_price :
  let sale_price : ℝ := 616
  let tax_rate : ℝ := 0.10
  let profit_rate : ℝ := 0.13
  abs (cost_price sale_price tax_rate profit_rate - 495.57) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_price_l79_7968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l79_7914

/-- Proves that the interest rate for a Rs. 1000 loan is 3% per year, given specific conditions. -/
theorem interest_rate_calculation (loan1 loan2 total_interest time rate2 : ℝ) 
  (h1 : loan1 = 1000)
  (h2 : loan2 = 1400)
  (h3 : total_interest = 390)
  (h4 : time = 3.9)
  (h5 : rate2 = 5)
  : ∃ x, loan1 * (time / 100) * x + loan2 * (rate2 / 100) * time = total_interest ∧ x = 3 := by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l79_7914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_line_l79_7995

/-- The distance from a point to a line in 2D space --/
noncomputable def distance_point_to_line (x y A B C : ℝ) : ℝ :=
  |A * x + B * y + C| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from point P(-5,7) to the line 12x + 5y - 3 = 0 is 28/13 --/
theorem distance_point_to_specific_line :
  distance_point_to_line (-5) 7 12 5 (-3) = 28/13 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_line_l79_7995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_mod_50_l79_7926

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the expression A
noncomputable def A : ℤ := floor (7/8) + floor (7^2/8) + floor (7^2019/8) + floor (7^2020/8)

-- State the theorem
theorem A_mod_50 : A % 50 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_mod_50_l79_7926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l79_7961

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sine rule for a triangle -/
axiom sine_rule (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The area formula for a triangle -/
noncomputable def triangle_area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * (Real.sin t.A - Real.sin t.B) = (t.c - t.b) * (Real.sin t.C + Real.sin t.B))
  (h2 : t.c = Real.sqrt 7)
  (h3 : triangle_area t = (3 * Real.sqrt 3) / 2) :
  t.C = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l79_7961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diplomat_languages_theorem_l79_7986

theorem diplomat_languages_theorem (total : ℕ) (french : ℕ) (not_russian : ℕ) (both_percent : ℚ) 
  (h_total : total = 150)
  (h_french : french = 17)
  (h_not_russian : not_russian = 32)
  (h_both_percent : both_percent = 1/10) :
  (total - (french + (total - not_russian) - (both_percent * ↑total).floor)) / ↑total = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diplomat_languages_theorem_l79_7986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_product_square_l79_7939

/-- A set of points in ℝ² forms a right triangle. -/
def IsRightTriangle (T : Set (ℝ × ℝ)) : Prop := sorry

/-- The area of a triangle. -/
noncomputable def Area (T : Set (ℝ × ℝ)) : ℝ := sorry

/-- The length of the hypotenuse of a right triangle. -/
noncomputable def Hypotenuse (T : Set (ℝ × ℝ)) : ℝ := sorry

/-- The length of one leg of a right triangle. -/
noncomputable def OneLeg (T : Set (ℝ × ℝ)) : ℝ := sorry

/-- The length of the shorter leg of a right triangle. -/
noncomputable def ShorterLeg (T : Set (ℝ × ℝ)) : ℝ := sorry

/-- Given two right triangles T₁ and T₂ with specific properties, 
    the square of the product of their hypotenuses is 4624. -/
theorem hypotenuse_product_square (T₁ T₂ : Set (ℝ × ℝ)) 
  (h₁ : IsRightTriangle T₁)
  (h₂ : IsRightTriangle T₂)
  (area₁ : Area T₁ = 2)
  (area₂ : Area T₂ = 8)
  (hyp_leg : Hypotenuse T₁ = OneLeg T₂)
  (leg_hyp : ShorterLeg T₁ = Hypotenuse T₂) :
  (Hypotenuse T₁ * Hypotenuse T₂)^2 = 4624 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_product_square_l79_7939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_243_equals_3_to_m_l79_7936

theorem root_243_equals_3_to_m (m : ℝ) : (243 : ℝ) ^ (1/2 : ℝ) = 3 ^ m → m = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_243_equals_3_to_m_l79_7936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l79_7960

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.log x

-- State the theorem
theorem f_inequality (a b : ℝ) (h : f (Real.exp a) > f b) :
  (a > 0 → Real.exp a - b > 0) ∧ (a < 0 → a - Real.log b < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l79_7960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_nonnegative_l79_7906

theorem sin_cos_nonnegative (θ : ℝ) : 
  Real.sin θ * Real.sqrt (1 - Real.cos θ ^ 2) + Real.cos θ * Real.sqrt (1 - Real.sin θ ^ 2) = 1 →
  ¬(Real.sin θ < 0 ∨ Real.cos θ < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_nonnegative_l79_7906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_satisfies_recurrence_R_base_case_l79_7905

/-- Equivalent resistance of an n-stage network with unit resistors -/
noncomputable def R (n : ℕ) : ℝ :=
  (Real.sqrt 3 - 1 - (5 - 3 * Real.sqrt 3) * (7 - 4 * Real.sqrt 3) ^ (n - 1)) /
  (1 - (7 - 4 * Real.sqrt 3) ^ n)

/-- The recurrence relation for the equivalent resistance -/
noncomputable def R_recurrence (R_prev : ℝ) : ℝ :=
  (2 + R_prev) / (3 + R_prev)

/-- Theorem stating that R satisfies the recurrence relation for all n > 1 -/
theorem R_satisfies_recurrence (n : ℕ) (h : n > 1) :
  R n = R_recurrence (R (n - 1)) := by
  sorry

/-- Theorem stating that R(1) = 1 -/
theorem R_base_case : R 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_satisfies_recurrence_R_base_case_l79_7905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_conditions_l79_7993

/-- Theorem: Non-existence of a function satisfying specific integral conditions -/
theorem no_function_satisfies_conditions (α : ℝ) :
  ¬ ∃ (f : ℝ → ℝ), 
    (ContinuousOn f (Set.Icc 0 1)) ∧ 
    (∀ x ∈ Set.Icc 0 1, f x > 0) ∧
    (∫ x in Set.Icc 0 1, f x = 1) ∧
    (∫ x in Set.Icc 0 1, x * f x = α) ∧
    (∫ x in Set.Icc 0 1, x^2 * f x = α^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_conditions_l79_7993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_production_theorem_l79_7975

noncomputable def production_cost (x : ℝ) : ℝ := x^2 / 10 - 2*x + 90

noncomputable def cost_per_ton (x : ℝ) : ℝ := production_cost x / x

noncomputable def profit (x : ℝ) : ℝ := 6*x - production_cost x

theorem factory_production_theorem :
  cost_per_ton 40 = 4.25 ∧ 
  ∃ (max_profit : ℝ), max_profit = 70 ∧ ∀ (x : ℝ), profit x ≤ max_profit := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_production_theorem_l79_7975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_and_extremum_l79_7916

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_x (f : QuadraticFunction) : ℝ := -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
noncomputable def vertex_y (f : QuadraticFunction) : ℝ :=
  f.a * (vertex_x f)^2 + f.b * (vertex_x f) + f.c

/-- Whether the quadratic function has a maximum (true) or minimum (false) -/
noncomputable def is_maximum (f : QuadraticFunction) : Prop := f.a < 0

theorem parabola_vertex_and_extremum (f : QuadraticFunction) 
  (h1 : f.a = -3) (h2 : f.b = 6) (h3 : f.c = 1) :
  vertex_x f = 1 ∧ vertex_y f = 4 ∧ is_maximum f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_and_extremum_l79_7916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_with_given_point_l79_7949

/-- Given that the terminal side of angle α passes through the point (-1, -2),
    prove the following statements about trigonometric functions of α. -/
theorem trig_identities_with_given_point (α : ℝ) : 
  (∃ (r : ℝ), r * Real.cos α = -1 ∧ r * Real.sin α = -2) →
  (Real.cos α = -Real.sqrt 5 / 5 ∧ 
   Real.tan α = 2 ∧
   (Real.sin (π - α) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / 
   (Real.tan (-α - π) * Real.sin (-π - α)) = 1 / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_with_given_point_l79_7949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l79_7912

theorem repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = 0.3846153846153846 → c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l79_7912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_exchange_correct_l79_7932

/-- Represents the number of people at the gathering -/
def x : ℕ := sorry

/-- The total number of gifts exchanged -/
def total_gifts : ℕ := 40

/-- The equation representing the gift exchange situation -/
def gift_exchange_equation (x : ℕ) : Prop :=
  x * (x - 1) = total_gifts

/-- Theorem stating that the gift exchange equation correctly represents the situation -/
theorem gift_exchange_correct (x : ℕ) : 
  gift_exchange_equation x ↔ 
  (∀ (i j : ℕ), i < x → j < x → i ≠ j → ∃! (gift : ℕ), gift = 1) ∧
  (Finset.sum (Finset.range x) (λ i => Finset.sum (Finset.range x) (λ j => if i ≠ j then 1 else 0))) = total_gifts :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gift_exchange_correct_l79_7932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l79_7918

/-- Ellipse parameters -/
structure EllipseParams where
  a : ℝ
  b : ℝ
  h : a > b
  k : b > 0

/-- Ellipse definition -/
noncomputable def Ellipse (p : EllipseParams) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 / p.a^2 + y^2 / p.b^2 = 1}

/-- Circle definition -/
noncomputable def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 + y^2 = r^2}

/-- Triangle area -/
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

theorem ellipse_and_triangle_properties
  (p : EllipseParams)
  (h_foci : ∃ F₁ F₂ : ℝ × ℝ, F₁ ∈ Ellipse p ∧ F₂ ∈ Ellipse p ∧ (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 8)
  (h_triangle : ∃ A B F₂ : ℝ × ℝ, A ∈ Ellipse p ∧ B ∈ Ellipse p ∧ F₂ ∈ Ellipse p ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 + (B.1 - F₂.1)^2 + (B.2 - F₂.2)^2 + (F₂.1 - A.1)^2 + (F₂.2 - A.2)^2 = 48) :
  (p.a = Real.sqrt 3 ∧ p.b = 1) ∧
  (∀ P : ℝ × ℝ, P ∈ Circle 2 →
    ∃ M N : ℝ × ℝ, M ∈ Circle 2 ∧ N ∈ Circle 2 ∧
      (∀ Q : ℝ × ℝ, Q ∈ Ellipse p → (Q.2 - P.2) * (Q.1 - M.1) = (Q.1 - P.1) * (Q.2 - M.2)) ∧
      (∀ Q : ℝ × ℝ, Q ∈ Ellipse p → (Q.2 - P.2) * (Q.1 - N.1) = (Q.1 - P.1) * (Q.2 - N.2)) ∧
      TriangleArea P M N ≤ 4) ∧
  (∃ P M N : ℝ × ℝ, P ∈ Circle 2 ∧ M ∈ Circle 2 ∧ N ∈ Circle 2 ∧
    (∀ Q : ℝ × ℝ, Q ∈ Ellipse p → (Q.2 - P.2) * (Q.1 - M.1) = (Q.1 - P.1) * (Q.2 - M.2)) ∧
    (∀ Q : ℝ × ℝ, Q ∈ Ellipse p → (Q.2 - P.2) * (Q.1 - N.1) = (Q.1 - P.1) * (Q.2 - N.2)) ∧
    TriangleArea P M N = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l79_7918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l79_7987

theorem cubic_equation_roots (a : ℝ) :
  let f := fun x : ℝ => x^3 - x - a
  ((∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) ↔ abs a < 2 * Real.sqrt 3 / 9) ∧
  ((∃! x : ℝ, f x = 0) ↔ abs a > 2 * Real.sqrt 3 / 9) ∧
  ((∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ ∀ z : ℝ, f z = 0 → z = x ∨ z = y) ↔ abs a = 2 * Real.sqrt 3 / 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l79_7987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_theorem_l79_7901

-- Define the variables
variable (a b t p q : ℝ)
-- Assume p ≠ q and t > 0 to avoid division by zero
variable (h1 : p ≠ q)
variable (h2 : t > 0)

-- Define the speed of the second object
noncomputable def speed_second : ℝ := (a * p * t + b * q) / (t * (q - p))

-- Define the speed of the first object
noncomputable def speed_first : ℝ := q * (a * t + b) / (t * (q - p))

-- Define the distance AB
noncomputable def distance_AB : ℝ := q * (a * t + b) / (q - p)

-- Theorem statement
theorem journey_theorem :
  -- The first object covers 'a' meters more per second than the second
  speed_first a b t p q = speed_second a b t p q + a ∧
  -- The first object completes the journey in 't' seconds
  speed_first a b t p q * t = distance_AB a b t p q ∧
  -- The second object covers 'b' meters more than (p/q) of the distance in the same time
  speed_second a b t p q * t = p / q * distance_AB a b t p q + b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_theorem_l79_7901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l79_7929

-- Define the function f(x) = 2x - x^2
def f (x : ℝ) : ℝ := 2 * x - x^2

-- Define the interval [-2, 3]
def I : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem a_range (a : ℝ) : (∀ x ∈ I, a < f x) → a ∈ Set.Iio (-8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l79_7929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_packing_height_difference_l79_7907

/-- The diameter of each cylindrical pipe in centimeters -/
def pipe_diameter : ℝ := 8

/-- The number of pipes in each crate -/
def num_pipes : ℕ := 300

/-- The height of the direct stacking method in Crate C -/
def direct_stack_height : ℝ := 30 * pipe_diameter

/-- The height between centers of consecutive rows in the staggered stacking method -/
noncomputable def staggered_row_height : ℝ := (Real.sqrt 3 / 2) * pipe_diameter

/-- The height of the staggered stacking method in Crate D -/
noncomputable def staggered_stack_height : ℝ := pipe_diameter + (29 * staggered_row_height)

/-- The difference in height between the two packing methods -/
noncomputable def height_difference : ℝ := direct_stack_height - staggered_stack_height

theorem packing_height_difference :
  height_difference = 232 - 116 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_packing_height_difference_l79_7907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l79_7915

/-- The distance between two points (a, 2a + 3) and (5, d) -/
noncomputable def distance (a d : ℝ) : ℝ :=
  Real.sqrt ((5 - a)^2 + (d - (2*a + 3))^2)

/-- Theorem stating that the distance between the points is as calculated -/
theorem distance_between_points (a d : ℝ) :
  distance a d = Real.sqrt ((5 - a)^2 + (d - (2*a + 3))^2) := by
  -- Unfold the definition of distance
  unfold distance
  -- The rest of the proof is trivial since it's just the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l79_7915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_l79_7959

theorem gcd_of_polynomial_and_multiple (y : ℤ) : 
  3456 ∣ y → 
  Nat.gcd (((5 * y + 4) * (9 * y + 1) * (12 * y + 6) * (3 * y + 9)).natAbs) y.natAbs = 216 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_l79_7959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_mult_formula_l79_7948

/-- Custom multiplication operation for positive integers -/
def custom_mult : ℕ+ → ℕ+ → ℕ := sorry

/-- Property 1: 1 * 1 = 2 -/
axiom custom_mult_one : custom_mult 1 1 = 2

/-- Property 2: (n+1) * 1 = n * 1 + 2^(n+1) -/
axiom custom_mult_succ (n : ℕ+) : 
  custom_mult (n + 1) 1 = custom_mult n 1 + 2^(n.val + 1)

/-- Theorem: For any positive integer n, n * 1 = 2^(n+1) - 2 -/
theorem custom_mult_formula (n : ℕ+) : 
  custom_mult n 1 = 2^(n.val + 1) - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_mult_formula_l79_7948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l79_7913

/-- The area of a triangle with base 12 cm and an angle of 30 degrees adjacent to the base is 72/√3 cm². -/
theorem triangle_area (base : ℝ) (angle : ℝ) (h_base : base = 12) (h_angle : angle = 30 * π / 180) :
  (1/2) * base * (base * Real.sin angle / Real.cos angle) * Real.sin angle = 72 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l79_7913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_minimization_l79_7900

theorem cylinder_surface_area_minimization (V : ℝ) (V_pos : 0 < V) :
  let ratio_for_min_area (vol : ℝ) : ℝ := 2
  ∀ (k : ℝ), k > 0 →
    ratio_for_min_area (k * V) = ratio_for_min_area V := by
  intro ratio_for_min_area k k_pos
  -- The proof goes here
  sorry

#check cylinder_surface_area_minimization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_minimization_l79_7900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l79_7998

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 9 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := 3 * x - 2 * y = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- State the theorem
theorem hyperbola_foci_distance (x y xf1 yf1 xf2 yf2 : ℝ) :
  hyperbola x y →
  asymptote x y →
  distance x y xf1 yf1 = 3 →
  (xf1 < xf2) →  -- Assuming F1 is the left focus and F2 is the right focus
  distance x y xf2 yf2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l79_7998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_sum_constant_l79_7941

/-- The curve C in the problem -/
def curve (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Point M -/
def M : ℝ × ℝ := (4, 0)

/-- Point N -/
def N : ℝ × ℝ := (1, 0)

/-- A line passing through N -/
structure Line where
  m : ℝ
  h : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ y = m * (x - 1)

/-- Intersection points of the line with the curve C -/
structure IntersectionPoints (l : Line) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_A : curve A.1 A.2 ∧ l.eq A.1 A.2
  h_B : curve B.1 B.2 ∧ l.eq B.1 B.2
  h_distinct : A ≠ B

/-- The point H where the line intersects the y-axis -/
noncomputable def H (l : Line) : ℝ × ℝ := (0, -1 / l.m)

/-- Definition of λ₁ and λ₂ -/
noncomputable def lambda₁ (l : Line) (ip : IntersectionPoints l) : ℝ :=
  let AN := (N.1 - ip.A.1, N.2 - ip.A.2)
  let MN := (N.1 - M.1, N.2 - M.2)
  (MN.1 * AN.1 + MN.2 * AN.2) / (AN.1^2 + AN.2^2)

noncomputable def lambda₂ (l : Line) (ip : IntersectionPoints l) : ℝ :=
  let BN := (N.1 - ip.B.1, N.2 - ip.B.2)
  let HB := (ip.B.1 - (H l).1, ip.B.2 - (H l).2)
  (HB.1 * BN.1 + HB.2 * BN.2) / (BN.1^2 + BN.2^2)

/-- The main theorem to prove -/
theorem lambda_sum_constant (l : Line) (ip : IntersectionPoints l) :
  lambda₁ l ip + lambda₂ l ip = -8/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_sum_constant_l79_7941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l79_7940

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then (1 - a) * x + a else (a - 3) * x^2 + 2

theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc 2 3 ∧ a ≠ 3 := by
  sorry

#check decreasing_f_implies_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l79_7940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polygon_angle_pair_l79_7982

/-- Represents the measure of an angle in degrees -/
def AngleMeasure : Type := { x : ℝ // x > 0 ∧ x < 180 }

/-- Proposition: There exists a unique pair of angle measure and integer multiplier
    satisfying the conditions for two different convex polygons -/
theorem unique_polygon_angle_pair :
  ∃! (x : AngleMeasure) (k : ℕ),
    k > 1 ∧
    x.val ≥ 60 ∧
    (k : ℝ) * x.val < 180 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polygon_angle_pair_l79_7982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_theorem_l79_7994

/-- Represents the time it takes to fill a tank using different combinations of pipes -/
structure TankFilling where
  xy : ℝ  -- Time for pipes X and Y together
  xz : ℝ  -- Time for pipes X and Z together
  yz : ℝ  -- Time for pipes Y and Z together

/-- Calculates the time it takes for all three pipes to fill the tank together -/
noncomputable def fillTime (t : TankFilling) : ℝ :=
  36 / 11

/-- Theorem stating that given the specific filling times for pairs of pipes,
    the time for all three pipes to fill the tank together is 36/11 hours -/
theorem fill_time_theorem (t : TankFilling) 
  (h_xy : t.xy = 3)
  (h_xz : t.xz = 6)
  (h_yz : t.yz = 4.5) :
  fillTime t = 36 / 11 := by
  sorry

#eval (36 : ℚ) / 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_theorem_l79_7994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l79_7908

theorem problem_solution (a b : ℕ) (h : (18 : ℕ) ^ a * 9 ^ (3 * a - 1) = (2 : ℕ) ^ 6 * (3 : ℕ) ^ b) : a = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l79_7908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l79_7927

/-- The circle defined by the equation x^2 - 6x + y^2 - 8y + 15 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - 6*p.1 + p.2^2 - 8*p.2 + 15 = 0}

/-- The shortest distance from the origin (0, 0) to the circle -/
noncomputable def shortestDistance : ℝ := 5 - Real.sqrt 10

/-- Theorem stating the existence of a point on the circle with the shortest distance to the origin -/
theorem shortest_distance_to_circle :
  ∃ (p : ℝ × ℝ), p ∈ Circle ∧
  ∀ (q : ℝ × ℝ), q ∈ Circle →
  Real.sqrt (p.1^2 + p.2^2) ≤ Real.sqrt (q.1^2 + q.2^2) ∧
  Real.sqrt (p.1^2 + p.2^2) = shortestDistance := by
  sorry

#check shortest_distance_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l79_7927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_cost_per_pound_l79_7920

/-- The cost of peanuts per pound given Frank's purchase scenario -/
theorem peanut_cost_per_pound : ℝ := by
  let initial_money : ℝ := 67
  let change : ℝ := 4
  let days : ℕ := 7
  let pounds_per_day : ℝ := 3
  let total_spent : ℝ := initial_money - change
  let total_pounds : ℝ := (days : ℝ) * pounds_per_day
  have h : total_spent / total_pounds = 3 := by sorry
  exact 3


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_cost_per_pound_l79_7920
