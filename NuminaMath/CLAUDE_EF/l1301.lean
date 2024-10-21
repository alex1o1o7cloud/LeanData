import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1301_130155

/-- Triangle ABC with given properties --/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  m : Real × Real
  n : Real × Real
  area : Real

/-- Theorem statement --/
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.m = (2 * abc.a - abc.c, abc.b))
  (h2 : abc.n = (Real.cos abc.C, Real.cos abc.B))
  (h3 : ∃ k : Real, abc.m = k • abc.n)
  (h4 : abc.area = Real.sqrt 3) :
  abc.B = π / 3 ∧ 
  (∃ (min_AC : Real), min_AC = 2 ∧ 
    ∀ (AC : Real), AC ≥ min_AC ∧ 
    (AC = min_AC → abc.a = abc.b ∧ abc.b = abc.c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1301_130155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l1301_130109

theorem largest_lambda : ∃ (lambda : ℝ), (lambda = 5/4) ∧
  (∀ (a b c d e : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → e ≥ 0 →
    a^2 + b^2 + c^2 + d^2 + e^2 ≥ a*b + lambda*b*c + c*d + d*e) ∧
  (∀ (lambda' : ℝ), lambda' > lambda →
    ∃ (a b c d e : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧
      a^2 + b^2 + c^2 + d^2 + e^2 < a*b + lambda'*b*c + c*d + d*e) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l1301_130109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_is_quadratic_sum_of_coefficients_l1301_130168

/-- Define the sequence v_n -/
def v : ℕ → ℚ
  | 0 => 7  -- Add this case to handle n = 0
  | 1 => 7
  | n + 1 => v n + (5 + 6 * (n - 1))

/-- v_n is a quadratic polynomial -/
theorem v_is_quadratic : ∃ (A B C : ℚ), ∀ n : ℕ, v n = A * n^2 + B * n + C := by sorry

/-- The sum of coefficients of v_n is 7 -/
theorem sum_of_coefficients : 
  ∃ (A B C : ℚ), (∀ n : ℕ, v n = A * n^2 + B * n + C) ∧ A + B + C = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_is_quadratic_sum_of_coefficients_l1301_130168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrigation_flows_l1301_130161

/-- Represents a channel in the irrigation system -/
structure Channel where
  name : String
  flow : ℝ

/-- Represents a node in the irrigation system -/
structure Node where
  name : String
  inflows : List Channel
  outflows : List Channel

/-- Represents the irrigation system -/
structure IrrigationSystem where
  channels : List Channel
  nodes : List Node
  q₀ : ℝ  -- Flow in channel AH

/-- The sum of flows between any two points is constant regardless of the path -/
axiom flow_sum_constant (sys : IrrigationSystem) (start end_ : Node) (path1 path2 : List Channel) :
  (path1.map (λ c => c.flow)).sum = (path2.map (λ c => c.flow)).sum

/-- Flow in channel AH is q₀ -/
axiom ah_flow (sys : IrrigationSystem) :
  (sys.channels.find? (λ c => c.name = "AH")).map (λ c => c.flow) = some sys.q₀

theorem irrigation_flows (sys : IrrigationSystem) :
  let ab := (sys.channels.find? (λ c => c.name = "AB")).map (λ c => c.flow)
  let bc := (sys.channels.find? (λ c => c.name = "BC")).map (λ c => c.flow)
  let total_inflow_a := (sys.nodes.find? (λ n => n.name = "A")).map (λ n => (n.inflows.map (λ c => c.flow)).sum)
  ab = some ((4/3) * sys.q₀) ∧
  bc = some ((2/3) * sys.q₀) ∧
  total_inflow_a = some ((7/3) * sys.q₀) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrigation_flows_l1301_130161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l1301_130154

def is_valid_configuration (n : ℕ) (config : List (ℕ × ℕ × Bool)) : Prop :=
  ∀ (a b : ℕ) (connected : Bool),
    (a, b, connected) ∈ config →
    (connected → ∃ (d : ℕ), d > 1 ∧ d ∣ (a + b) ∧ d ∣ n) ∧
    (¬connected → Nat.Coprime (a + b) n)

theorem smallest_valid_n :
  (∃ (config : List (ℕ × ℕ × Bool)), is_valid_configuration 15 config) ∧
  (∀ (m : ℕ), m < 15 → ¬∃ (config : List (ℕ × ℕ × Bool)), is_valid_configuration m config) :=
by
  sorry

#check smallest_valid_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l1301_130154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_linear_combination_l1301_130180

-- Define a conic section as a function from ℝ² to ℝ
def ConicSection := ℝ × ℝ → ℝ

-- Define the property of two conic sections having four common points
def hasFourCommonPoints (F₁ F₂ : ConicSection) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ : ℝ × ℝ),
    F₁ p₁ = 0 ∧ F₁ p₂ = 0 ∧ F₁ p₃ = 0 ∧ F₁ p₄ = 0 ∧
    F₂ p₁ = 0 ∧ F₂ p₂ = 0 ∧ F₂ p₃ = 0 ∧ F₂ p₄ = 0 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄

-- Define the property of a conic section passing through four points
def passesThrough4Points (F : ConicSection) (p₁ p₂ p₃ p₄ : ℝ × ℝ) : Prop :=
  F p₁ = 0 ∧ F p₂ = 0 ∧ F p₃ = 0 ∧ F p₄ = 0

-- Theorem statement
theorem conic_sections_linear_combination
  (F₁ F₂ : ConicSection)
  (h : hasFourCommonPoints F₁ F₂) :
  ∀ (F : ConicSection),
    (∃ (p₁ p₂ p₃ p₄ : ℝ × ℝ), passesThrough4Points F p₁ p₂ p₃ p₄ ∧
      F₁ p₁ = 0 ∧ F₁ p₂ = 0 ∧ F₁ p₃ = 0 ∧ F₁ p₄ = 0 ∧
      F₂ p₁ = 0 ∧ F₂ p₂ = 0 ∧ F₂ p₃ = 0 ∧ F₂ p₄ = 0) →
    ∃ (l m : ℝ), ∀ (x y : ℝ), F (x, y) = l * F₁ (x, y) + m * F₂ (x, y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_linear_combination_l1301_130180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_travel_time_l1301_130135

/-- Represents the speed of a particle at the nth mile -/
noncomputable def speed (n : ℕ) (k : ℝ) : ℝ := k / (n - 1)^2

/-- Represents the time taken to travel the nth mile -/
noncomputable def time (n : ℕ) (k : ℝ) : ℝ := 1 / speed n k

theorem particle_travel_time (n : ℕ) (h : n > 1) :
  ∃ k : ℝ, k > 0 ∧ time 3 k = 4 → time n k = (n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_travel_time_l1301_130135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_condition_l1301_130172

/-- Definition of an acute-angled triangle -/
def IsAcuteTriangle (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
  (a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (c^2 + a^2 > b^2)

theorem acute_triangle_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  IsAcuteTriangle a b c ↔ abs (a^2 - b^2) < c^2 ∧ c^2 < a^2 + b^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_condition_l1301_130172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_num_distribution_schemes_l1301_130148

/-- Represents the number of schools --/
def num_schools : ℕ := 5

/-- Represents the number of computers --/
def num_computers : ℕ := 6

/-- Represents the number of schools that must receive at least 2 computers --/
def schools_with_min_two : ℕ := 2

/-- Represents the minimum number of computers each of the specified schools must receive --/
def min_computers_per_school : ℕ := 2

/-- 
Helper function to calculate the number of ways to distribute computers
This function is not defined here, but would be implemented based on the problem constraints
--/
def num_ways_to_distribute (schools computers schools_with_min min_per_school : ℕ) : ℕ := 
  sorry

/-- 
Theorem stating that the number of ways to distribute the computers 
among the schools, given the constraints, is 15
--/
theorem num_distribution_schemes : 
  (num_ways_to_distribute num_schools num_computers schools_with_min_two min_computers_per_school) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_num_distribution_schemes_l1301_130148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1301_130129

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (a + 1) * x

-- State the theorem
theorem f_properties :
  -- Domain of f(x)
  ∀ (a : ℝ) (x : ℝ), x > 0 →
  -- Part 1: Monotonicity when a = 0
  (a = 0 → 
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂, 1 ≤ x₁ ∧ x₁ < x₂ → f a x₂ < f a x₁)) ∧
  -- Part 2: Range of a when maximum value > -2
  (a > -1 → 
    (∃ x, x > 0 ∧ ∀ y, y > 0 → f a y ≤ f a x) →
    (∃ x, x > 0 ∧ f a x > -2) →
    a < Real.exp 1 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1301_130129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_pure_imaginary_z₂_fourth_quadrant_l1301_130124

/-- Define complex number z₁ -/
def z₁ (m : ℝ) : ℂ := m * (m - 1) + (m - 1) * Complex.I

/-- Define complex number z₂ -/
def z₂ (m : ℝ) : ℂ := (m + 1) + (m^2 - 1) * Complex.I

/-- Theorem 1: If z₁ is a pure imaginary number, then m = 0 -/
theorem z₁_pure_imaginary (m : ℝ) : z₁ m = (m - 1) * Complex.I → m = 0 := by
  sorry

/-- Theorem 2: If z₂ is in the fourth quadrant, then -1 < m < 1 -/
theorem z₂_fourth_quadrant (m : ℝ) :
  (z₂ m).re > 0 ∧ (z₂ m).im < 0 → -1 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_pure_imaginary_z₂_fourth_quadrant_l1301_130124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_intersection_theorem_l1301_130117

-- Define the curve C
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l₁
def l₁ (x y m : ℝ) : Prop := x - y + m = 0

-- Define the arithmetic sequence condition
def arithmetic_sequence (x y : ℝ) : Prop :=
  ∃ (a : ℝ), Real.sqrt ((x-1)^2 + y^2) = a ∧ 2 = a + 1 ∧ Real.sqrt ((x+1)^2 + y^2) = a + 2

-- Define the intersection condition
def intersects_at_two_points (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ C x₁ y₁ ∧ C x₂ y₂ ∧ l₁ x₁ y₁ m ∧ l₁ x₂ y₂ m

-- Define the obtuse angle condition
def obtuse_angle (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ < 0

theorem curve_and_intersection_theorem :
  (∀ x y : ℝ, arithmetic_sequence x y → C x y) ∧
  (∀ m : ℝ, intersects_at_two_points m ∧
    (∀ x₁ y₁ x₂ y₂ : ℝ, C x₁ y₁ ∧ C x₂ y₂ ∧ l₁ x₁ y₁ m ∧ l₁ x₂ y₂ m → obtuse_angle x₁ y₁ x₂ y₂) →
    -2 * Real.sqrt 42 / 7 < m ∧ m < 2 * Real.sqrt 42 / 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_intersection_theorem_l1301_130117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_time_difference_l1301_130167

-- Define the cycling data for each day
structure CyclingDay where
  distance : ℚ
  speed : ℚ

-- Define the cycling schedule
def schedule : List CyclingDay := [
  { distance := 3, speed := 6 },
  { distance := 4, speed := 4 },
  { distance := 5, speed := 5 }
]

-- Calculate the total time spent cycling
noncomputable def totalTime (s : List CyclingDay) : ℚ :=
  s.foldl (fun acc day => acc + day.distance / day.speed) 0

-- Calculate the total distance cycled
def totalDistance (s : List CyclingDay) : ℚ :=
  s.foldl (fun acc day => acc + day.distance) 0

-- Calculate the time if cycling at a constant speed
noncomputable def constantSpeedTime (s : List CyclingDay) (speed : ℚ) : ℚ :=
  (totalDistance s) / speed

-- Main theorem
theorem cycling_time_difference :
  (totalTime schedule - constantSpeedTime schedule 5) * 60 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_time_difference_l1301_130167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1301_130162

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x^2 + (a+2)*x + 1) * ((3-2*a)*x^2 + 5*x + (3-2*a)) ≥ 0) →
  (a ∈ Set.Icc (-4 : ℝ) 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1301_130162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1301_130195

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.cos (x - Real.pi / 3)

theorem f_properties :
  (f (2 * Real.pi / 3) = -1 / 4) ∧
  (∀ x : ℝ, f x < 1 / 4 ↔ ∃ k : ℤ, k * Real.pi - 7 * Real.pi / 12 < x ∧ x < k * Real.pi - Real.pi / 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1301_130195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_product_l1301_130122

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_three_digit_product (n x y : ℕ) :
  n ≥ 100 ∧ n ≤ 999 ∧
  n = x * y * (10 * x + y) ∧
  is_prime x ∧ x ≤ 9 ∧
  ¬(3 ∣ (x + y)) ∧
  is_prime (10 * x + y) →
  n ≤ 795 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_product_l1301_130122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l1301_130175

theorem log_problem (x : ℝ) (h : Real.log 9 * x = 3 * Real.log 4) : 
  Real.log 64 / Real.log x = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_problem_l1301_130175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1301_130184

noncomputable def line1 (x y : ℝ) : Prop := 3 * y + 2 * x - 6 = 0
noncomputable def line2 (a x y : ℝ) : Prop := 4 * y + a * x + 5 = 0

noncomputable def slope1 : ℝ := -2 / 3
noncomputable def slope2 (a : ℝ) : ℝ := -a / 4

def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, line1 x y → line2 a x y → perpendicular slope1 (slope2 a)) →
  a = -6 := by
  sorry

#check perpendicular_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1301_130184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_seven_parts_product_l1301_130121

theorem sqrt_seven_parts_product (x y : ℝ) : 
  (x = ⌊Real.sqrt 7⌋) → 
  (y = Real.sqrt 7 - ⌊Real.sqrt 7⌋) → 
  (x + Real.sqrt 7) * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_seven_parts_product_l1301_130121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_log_sum_l1301_130114

noncomputable section

open Real

-- Define the given conditions
theorem max_log_sum (θ m n a b c d : ℝ) : 
  -- m and n are distinct real roots of the equation
  sin θ * m^2 + cos θ * m - 1 = 0 →
  sin θ * n^2 + cos θ * n - 1 = 0 →
  m ≠ n →
  -- Maximum distance d from circle to line MN
  d = 2 →
  -- Equation involving a, b, c, and d
  a * b * c + b^2 + c^2 = 4 * d →
  -- a, b, c are positive real numbers
  a > 0 → b > 0 → c > 0 →
  -- The maximum value of the logarithmic sum
  (∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' * b' * c' + b'^2 + c'^2 = 8 ∧
    ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x * y * z + y^2 + z^2 = 8 →
      (log a' / log 4 + log b' / log 2 + log c' / log 2 ≥ 
      log x / log 4 + log y / log 2 + log z / log 2)) ∧
  (log a / log 4 + log b / log 2 + log c / log 2 ≤ 3/2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_log_sum_l1301_130114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_ratio_adults_to_children_l1301_130106

/-- Admission fee for adults -/
def adult_fee : ℚ := 20

/-- Admission fee for children -/
def child_fee : ℚ := 10

/-- Total collected admission fees -/
def total_fee : ℚ := 1600

/-- Proposition: The ratio of adults to children closest to 1 is 53/54 -/
theorem closest_ratio_adults_to_children :
  ∀ a c : ℚ,
  a > 0 → c > 0 →
  adult_fee * a + child_fee * c = total_fee →
  ∀ a' c' : ℚ,
  a' > 0 → c' > 0 →
  adult_fee * a' + child_fee * c' = total_fee →
  |a / c - 1| ≤ |a' / c' - 1| →
  a / c = 53 / 54 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_ratio_adults_to_children_l1301_130106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l1301_130187

/-- The function f(x) = (3x^2 + 4x + 5) / (2x + 3) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 4 * x + 5) / (2 * x + 3)

/-- The proposed oblique asymptote function g(x) = (3/2)x + 1/4 -/
noncomputable def g (x : ℝ) : ℝ := (3/2) * x + 1/4

/-- Theorem stating that g is the oblique asymptote of f -/
theorem oblique_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x > M, |f x - g x| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l1301_130187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_exp_satisfies_l1301_130153

-- Define the four functions
noncomputable def f₁ : ℝ → ℝ := λ x => Real.log x
noncomputable def f₂ : ℝ → ℝ := λ x => Real.cos x
noncomputable def f₃ : ℝ → ℝ := λ x => Real.exp x
noncomputable def f₄ : ℝ → ℝ := λ x => Real.exp (Real.cos x)

-- Define the property that we want to check
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x₁ ∈ Set.range f, ∃! x₂, f x₁ * f x₂ = 1

-- State the theorem
theorem only_exp_satisfies :
  satisfies_condition f₃ ∧
  ¬satisfies_condition f₁ ∧
  ¬satisfies_condition f₂ ∧
  ¬satisfies_condition f₄ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_exp_satisfies_l1301_130153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_AB_l1301_130139

theorem det_AB {n : Type*} [Fintype n] [DecidableEq n] 
  (A B : Matrix n n ℝ) (h1 : Matrix.det A = -3) (h2 : Matrix.det B⁻¹ = 1/4) :
  Matrix.det (A * B) = -12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_AB_l1301_130139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_33150_l1301_130198

/-- Represents a standard deck of 52 playing cards -/
structure Deck where
  cards : Finset (Nat × Nat)
  card_count : cards.card = 52
  suit_count : ∀ s, 1 ≤ s ∧ s ≤ 4 → (cards.filter (λ c ↦ c.1 = s)).card = 13
  rank_count : ∀ r, 1 ≤ r ∧ r ≤ 13 → (cards.filter (λ c ↦ c.2 = r)).card = 4

/-- The probability of drawing a 4, then a 5, then a 6 of the same suit from a standard deck -/
def probability_sequential_draw (d : Deck) : ℚ :=
  (4 : ℚ) / 52 * (1 : ℚ) / 51 * (1 : ℚ) / 50

/-- Theorem stating that the probability is 1/33150 -/
theorem probability_is_one_33150 (d : Deck) :
  probability_sequential_draw d = 1 / 33150 := by
  sorry

#eval (1 : ℚ) / 33150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_33150_l1301_130198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1301_130123

noncomputable def sequence_a (n : ℕ) : ℝ :=
  1 / 3^n

noncomputable def sequence_b (n : ℕ) : ℝ :=
  n / sequence_a n

noncomputable def sum_b (n : ℕ) : ℝ :=
  ((2 * n - 1) * 3^(n + 1)) / 4 + 3 / 4

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → (Finset.range n).sum (fun i => 3^i * sequence_a (i + 1)) = n / 3) →
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 1 / 3^n) ∧
  (∀ n : ℕ, n ≥ 1 → sum_b n = ((2 * n - 1) * 3^(n + 1)) / 4 + 3 / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1301_130123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_sum_l1301_130113

/-- The sum of areas of an infinite sequence of circles with decreasing radii -/
theorem circle_area_sum (initial_radius ratio π : ℝ)
  (h1 : initial_radius = 2)
  (h2 : ratio = 1/3)
  (h3 : π > 0) :
  (∑' n, π * (initial_radius * ratio^n)^2) = 9 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_sum_l1301_130113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_office_persons_count_l1301_130176

/-- The number of persons in an office given specific age information -/
theorem office_persons_count (N : ℕ) 
  (average_age : N * 15 = N * 15)
  (sum_age_5 : 5 * 14 = 70)
  (sum_age_9 : 9 * 16 = 144)
  (age_15th : 41 = 41)
  (total_sum : 15 * N = 70 + 144 + 41) : 
  N = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_office_persons_count_l1301_130176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1301_130100

open Real

/-- The function we're minimizing -/
noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1) + 1 / (5 - x)

/-- The theorem stating the minimum value of the function -/
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo 1 5 ∧
  (∀ (y : ℝ), y ∈ Set.Ioo 1 5 → f x ≤ f y) ∧
  f x = (3 + 2 * Real.sqrt 2) / 4 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1301_130100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_planes_l1301_130164

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelP : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the lines and planes
variable (l m n : Line)
variable (α β : Plane)

-- Define the conditions
variable (hdiff_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n)
variable (hdiff_planes : α ≠ β)

-- State the theorem
theorem perpendicular_parallel_planes 
  (h1 : perpendicularLP m α)
  (h2 : parallelLP n β)
  (h3 : parallelP α β) :
  perpendicular m n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_planes_l1301_130164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_schedules_is_96_l1301_130169

/-- Represents the days of the week from Monday to Saturday -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents the teachers -/
inductive Teacher
| A
| B
| C
| D

/-- A schedule is a function from Day to Teacher -/
def Schedule := Day → Teacher

/-- Checks if two days are consecutive -/
def consecutive (d1 d2 : Day) : Prop :=
  match d1, d2 with
  | Day.Monday, Day.Tuesday => True
  | Day.Tuesday, Day.Wednesday => True
  | Day.Wednesday, Day.Thursday => True
  | Day.Thursday, Day.Friday => True
  | Day.Friday, Day.Saturday => True
  | _, _ => False

/-- Checks if a schedule is valid according to the problem conditions -/
def isValidSchedule (s : Schedule) : Prop :=
  (∀ d : Day, ∃ t : Teacher, s d = t) ∧
  (∃ d1 d2 d3 : Day, d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧
    s d1 = Teacher.A ∧ s d2 = Teacher.B ∧ s d3 = Teacher.C) ∧
  (∃ d1 d2 d3 : Day, d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧
    s d1 = Teacher.D ∧ s d2 = Teacher.D ∧ s d3 = Teacher.D) ∧
  (∃ d1 d2 : Day, consecutive d1 d2 ∧ s d1 = Teacher.D ∧ s d2 = Teacher.D)

/-- The number of valid schedules -/
def numberOfValidSchedules : ℕ := sorry

/-- The main theorem stating that the number of valid schedules is 96 -/
theorem number_of_valid_schedules_is_96 : numberOfValidSchedules = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_schedules_is_96_l1301_130169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1301_130146

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2*x - 1 else x^2 + x

-- Define the function g
noncomputable def g (x m : ℝ) : ℝ := f x - m

-- Theorem statement
theorem m_range (m : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ g x₁ m = 0 ∧ g x₂ m = 0 ∧ g x₃ m = 0) →
  m ∈ Set.Ioo (-1/4 : ℝ) 0 ∪ {0} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1301_130146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l1301_130127

theorem tan_half_sum (x y : ℝ) : 
  Real.cos x + Real.cos y = 3/5 → Real.sin x + Real.sin y = 8/13 → Real.tan ((x + y)/2) = 40/39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l1301_130127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_l1301_130134

theorem min_value_exponential_sum (x y z : ℝ) (h : x + 2*y + 3*z = 6) :
  Real.exp (x * Real.log 2) + Real.exp (y * Real.log 4) + Real.exp (z * Real.log 8) ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_exponential_sum_l1301_130134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l1301_130182

/-- The number of solutions to n^x + n^y = n^z with n^z < 2001 -/
def num_solutions : ℕ := 10

/-- Predicate for valid solutions -/
def is_valid_solution (n x y z : ℕ) : Prop :=
  n > 1 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ n^x + n^y = n^z ∧ n^z < 2001

/-- The main theorem -/
theorem count_solutions :
  (∃! (s : Finset (ℕ × ℕ × ℕ × ℕ)), 
    (∀ (n x y z : ℕ), (n, x, y, z) ∈ s ↔ is_valid_solution n x y z) ∧
    s.card = num_solutions) :=
sorry

/-- Helper lemma: n^x + n^y = n^z implies x = y when n > 1 -/
lemma equal_exponents (n x y z : ℕ) (hn : n > 1) (hxyz : n^x + n^y = n^z) : x = y :=
sorry

/-- Helper lemma: if n^x + n^x = n^z and n > 1, then n = 2 and z = x + 1 -/
lemma power_of_two (n x z : ℕ) (hn : n > 1) (hxz : n^x + n^x = n^z) : n = 2 ∧ z = x + 1 :=
sorry

/-- Helper lemma: count of solutions for 2^x + 2^x = 2^(x+1) with 2^(x+1) < 2001 -/
lemma count_power_of_two_solutions : 
  (Finset.range 10).card = num_solutions :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l1301_130182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1301_130131

/-- The time (in seconds) for a train to pass a stationary point -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  train_length / (train_speed_kmh * 1000 / 3600)

/-- Theorem: A 100-meter long train traveling at 30 km/h takes approximately 12 seconds to pass a stationary point -/
theorem train_passing_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |train_passing_time 100 30 - 12| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1301_130131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1301_130112

noncomputable def f (x : ℝ) := 2 * Real.cos x * (Real.sin x + Real.cos x)

theorem f_properties :
  (∀ x : ℝ, f x = Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4) + 1) ∧
  f (5 * Real.pi / 4) = 2 ∧
  (∀ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) → p ≥ Real.pi) ∧
  (∀ k : ℤ, ∀ x : ℝ, 
    k * Real.pi - 3 * Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 8 → 
    ∀ y : ℝ, k * Real.pi - 3 * Real.pi / 8 ≤ y ∧ y ≤ x → f y ≤ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1301_130112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_share_calculation_l1301_130197

/-- Calculates the amount received by the first party in a three-way division --/
noncomputable def calculate_first_share (total_amount : ℝ) (ratio_x ratio_y ratio_z : ℕ) : ℝ :=
  let total_parts := (ratio_x + ratio_y + ratio_z : ℝ)
  let part_value := total_amount / total_parts
  part_value * (ratio_x : ℝ)

/-- Theorem stating that given $5000 divided in the ratio 2:5:8, the first share is approximately $666.66 --/
theorem first_share_calculation :
  let total_amount : ℝ := 5000
  let ratio_x : ℕ := 2
  let ratio_y : ℕ := 5
  let ratio_z : ℕ := 8
  abs (calculate_first_share total_amount ratio_x ratio_y ratio_z - 666.66) < 0.01 := by
  sorry

-- This will not be evaluated due to the noncomputable nature of calculate_first_share
-- #eval calculate_first_share 5000 2 5 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_share_calculation_l1301_130197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_switches_equal_sum_to_15_l1301_130179

/-- The sequence of numbers representing the initial order of paper pieces -/
def initial_sequence : List Nat := List.range 16 |>.reverse |>.map (· + 1)

/-- The sequence of numbers representing the final order of paper pieces -/
def final_sequence : List Nat := List.range 16 |>.map (· + 1)

/-- The number of adjacent switches required to move a single piece from its initial position to its final position -/
def switches_for_piece (i : Nat) : Nat :=
  if i ≤ 16 then
    (initial_sequence.indexOf i) - (final_sequence.indexOf i)
  else
    0

/-- The theorem stating that the minimum number of switches is equal to the sum of integers from 1 to 15 -/
theorem min_switches_equal_sum_to_15 :
  (List.range 15 |>.map (· + 1) |>.sum) = (List.range 16 |>.map switches_for_piece |>.sum) := by
  sorry

#eval List.range 15 |>.map (· + 1) |>.sum
#eval List.range 16 |>.map switches_for_piece |>.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_switches_equal_sum_to_15_l1301_130179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_is_20_l1301_130130

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ -- Semi-major axis
  b : ℝ -- Semi-minor axis

/-- Checks if a point lies on the given ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Perimeter of triangle ABF is 20 -/
theorem triangle_perimeter_is_20 
  (e : Ellipse)
  (A B F : Point)
  (h1 : e.a^2 = 25 ∧ e.b^2 = 21)
  (h2 : A.x = -2 ∧ pointOnEllipse A e)
  (h3 : B.x = -2 ∧ pointOnEllipse B e)
  (h4 : F.x = 2 ∧ F.y = 0) -- Right focus
  : distance A B + distance B F + distance F A = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_is_20_l1301_130130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_lawn_fraction_l1301_130188

/-- Represents the time (in hours) it takes to mow the entire lawn -/
noncomputable def mowing_time : ℚ := 3

/-- Represents the fraction of the lawn mowed in one hour -/
noncomputable def mowing_rate : ℚ := 1 / mowing_time

/-- Represents the time John works alone (in hours) -/
noncomputable def work_time : ℚ := 1

/-- Theorem: Given John can mow the entire lawn in 3 hours and works for 1 hour alone,
    the fractional part of the lawn that remains to be mowed is 2/3 -/
theorem remaining_lawn_fraction :
  1 - (mowing_rate * work_time) = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_lawn_fraction_l1301_130188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_inequality_l1301_130102

theorem arithmetic_geometric_mean_inequality 
  (n : ℕ) 
  (x : Fin n → ℝ) 
  (h_n : n ≥ 2) 
  (h_pos : ∀ i, x i > 0) 
  (h_non_decreasing : ∀ i j : Fin n, i ≤ j → x i ≤ x j) 
  (h_non_increasing : ∀ i j : Fin n, i ≤ j → x i ≥ (x j) / ((j : ℝ) + 1)) : 
  (Finset.sum Finset.univ x / n) / 
  (Finset.prod Finset.univ x) ^ (1 / n : ℝ) ≤ 
  (n + 1 : ℝ) / (2 * (n.factorial : ℝ) ^ (1 / n : ℝ)) := 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_inequality_l1301_130102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_hyperbola_l1301_130181

/-- Given a real number s ≠ 0, the point (x, y) = ((s^2 + 1)/s, (s^2 - 1)/s) lies on a hyperbola. -/
theorem point_on_hyperbola (s : ℝ) (hs : s ≠ 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    ((s^2 + 1) / s)^2 / a^2 - ((s^2 - 1) / s)^2 / b^2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_hyperbola_l1301_130181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_approximation_l1301_130171

/-- Given a square and a rectangle with the following properties:
    1. The perimeter of the square equals the perimeter of the rectangle
    2. The length of the rectangle is 16 cm
    3. The side of the square is the diameter of a semicircle with circumference ≈ 21.99 cm
    Prove that the breadth of the rectangle is approximately 1.1 cm -/
theorem rectangle_breadth_approximation (s : ℝ) (b : ℝ) : 
  (4 * s = 2 * (16 + b)) →  -- Perimeters are equal
  (abs (21.99 - (π * s / 2 + s)) < 0.01) →  -- Semicircle circumference condition
  (abs (b - 1.1) < 0.01) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_approximation_l1301_130171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_motion_l1301_130158

/-- The law of motion for a material point attracted to a fixed point -/
theorem law_of_motion 
  (m : ℝ) (ω : ℝ) (x : ℝ → ℝ) (t : ℝ) 
  (h1 : m > 0)
  (h2 : ω > 0)
  (h3 : ∀ t, (deriv^[2] x) t + ω^2 * x t = 0) :
  ∃ (R α : ℝ), ∀ t, x t = R * Real.sin (ω * t + α) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_motion_l1301_130158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_growth_l1301_130185

theorem function_growth (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (hf : ∀ x, HasDerivAt f (f' x) x) 
  (h_growth : ∀ x, f' x > f x) (a : ℝ) (ha : a > 0) : 
  f a > Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_growth_l1301_130185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_non_n_good_polynomials_l1301_130193

/-- A polynomial is n-good if it satisfies certain conditions -/
def is_n_good (F : ℤ → ℤ) (n : ℕ) : Prop :=
  F 0 = 1 ∧
  (∀ c : ℕ, c > 0 → F c > 0) ∧
  (∃! (s : Finset ℕ), s.card = n ∧ ∀ c ∈ s, Nat.Prime (F c).natAbs)

/-- There exist infinitely many non-constant polynomials that are not n-good for any n -/
theorem infinitely_many_non_n_good_polynomials :
  ∃ (S : Set (ℤ → ℤ)), (∀ F ∈ S, ∃ A : ℤ, A ≠ 0 ∧ ∀ x, F x = A * x + 1) ∧
                        Set.Infinite S ∧
                        ∀ F ∈ S, ∀ n : ℕ, ¬(is_n_good F n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_non_n_good_polynomials_l1301_130193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distributive_laws_l1301_130143

noncomputable def avg (a b : ℝ) : ℝ := (a + b) / 2

theorem distributive_laws (x y z : ℝ) :
  (x + 2 * avg y z = avg (x + 2*y) (x + 2*z)) ∧
  (avg x (avg y z) = avg (avg x y) (avg x z)) ∧
  (x + avg y z ≠ avg x y + avg x z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distributive_laws_l1301_130143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_lambda_l1301_130145

/-- Given vectors a, b, c in ℝ², and l ∈ ℝ, if (a + l*b) is parallel to c, then l = 2 -/
theorem parallel_vector_lambda (a b c : ℝ × ℝ) (l : ℝ) :
  a = (2, 1) →
  b = (0, 1) →
  c = (2, 3) →
  (∃ (k : ℝ), k ≠ 0 ∧ a + l • b = k • c) →
  l = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_lambda_l1301_130145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_weight_loss_percentage_l1301_130133

/-- Calculates the percentage of weight lost during beef processing. -/
theorem beef_weight_loss_percentage 
  (initial_weight : ℝ) 
  (processed_weight : ℝ) 
  (h1 : initial_weight = 846.15)
  (h2 : processed_weight = 550) : 
  |((initial_weight - processed_weight) / initial_weight * 100 - 34.99)| < 0.01 := by
  sorry

#eval Float.abs ((846.15 - 550) / 846.15 * 100 - 34.99) < 0.01

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beef_weight_loss_percentage_l1301_130133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_congruence_divisibility_l1301_130132

theorem prime_congruence_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → Odd q → 
  (∃ x : ℤ, (q : ℤ) ∣ ((x + 1)^p - x^p)) ↔ q ≡ 1 [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_congruence_divisibility_l1301_130132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_of_line_l_coordinates_of_point_P_l1301_130104

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 4 = 0

-- Define point E
def point_E : ℝ × ℝ := (3, 4)

-- Define the condition for line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*x - 3*k + 4

-- Define the length of AB
noncomputable def length_AB : ℝ := 2 * Real.sqrt 3

-- Define the condition for point P
def point_P (x y : ℝ) : Prop := y + x - 1 = 0

-- Theorem 1: Equation of line l
theorem equation_of_line_l :
  ∃ (k : ℝ), (∀ x y : ℝ, line_l k x y ↔ (x = 3 ∨ y = (3/4)*x + 7/4)) := by
  sorry

-- Theorem 2: Coordinates of point P
theorem coordinates_of_point_P :
  ∃ (x y : ℝ), point_P x y ∧ x = 1/2 ∧ y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_of_line_l_coordinates_of_point_P_l1301_130104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_face_circumradius_value_m_plus_n_value_l1301_130150

/-- Represents a tetrahedron SABC with specific properties -/
structure Tetrahedron where
  /-- Circumradius of faces SAB, SBC, and SCA -/
  face_radius : ℝ
  /-- Radius of the inscribed sphere -/
  inradius : ℝ
  /-- Distance from S to the center of the inscribed sphere I -/
  SI : ℝ

/-- The largest possible circumradius of face ABC in the tetrahedron -/
noncomputable def largest_face_circumradius (t : Tetrahedron) : ℝ :=
  Real.sqrt (23185 / 1)

/-- Theorem stating the largest possible circumradius of face ABC -/
theorem largest_face_circumradius_value (t : Tetrahedron) 
    (h1 : t.face_radius = 108)
    (h2 : t.inradius = 35)
    (h3 : t.SI = 125) :
  largest_face_circumradius t = Real.sqrt (23185 / 1) := by
  sorry

/-- The sum of m and n in the expression R = √(m/n) -/
def m_plus_n : ℕ := 23186

/-- Theorem stating the value of m + n -/
theorem m_plus_n_value :
  m_plus_n = 23186 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_face_circumradius_value_m_plus_n_value_l1301_130150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_service_cost_is_correct_l1301_130118

-- Define the given constants and variables
def fuel_cost_per_liter : ℚ := 60/100
def mini_van_count : ℕ := 3
def truck_count : ℕ := 2
def total_cost : ℚ := 2991/10
def mini_van_tank_capacity : ℚ := 65
def truck_tank_capacity : ℚ := mini_van_tank_capacity * 22/10

-- Define the function to calculate the service cost per vehicle
noncomputable def service_cost_per_vehicle : ℚ :=
  let total_fuel_volume := mini_van_count * mini_van_tank_capacity + truck_count * truck_tank_capacity
  let total_fuel_cost := total_fuel_volume * fuel_cost_per_liter
  let total_service_cost := total_cost - total_fuel_cost
  let total_vehicles := mini_van_count + truck_count
  total_service_cost / total_vehicles

-- State the theorem
theorem service_cost_is_correct : service_cost_per_vehicle = 21/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_service_cost_is_correct_l1301_130118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_root_of_complex_equation_l1301_130177

def complex_equation (z : ℂ) : Prop :=
  z^2 = -36 + 64*Complex.I

theorem other_root_of_complex_equation :
  complex_equation (4 + 8*Complex.I) → complex_equation (-4 - 8*Complex.I) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_root_of_complex_equation_l1301_130177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_is_eight_point_nine_l1301_130196

/-- Represents a measuring device scale --/
structure Scale where
  start : ℝ
  finish : ℝ
  arrow_position : ℝ
  h_range : start < finish
  h_arrow : start ≤ arrow_position ∧ arrow_position ≤ finish

/-- Estimates the reading on a scale --/
def estimate_reading (s : Scale) : ℝ :=
  s.arrow_position

/-- Theorem stating that the estimated reading is 8.9 for the given scale --/
theorem estimate_is_eight_point_nine (s : Scale) 
  (h_start : s.start = 8.5)
  (h_finish : s.finish = 9.0)
  (h_arrow : s.arrow_position > (s.start + s.finish) / 2 ∧ s.arrow_position < s.finish)
  : estimate_reading s = 8.9 := by
  sorry

#check estimate_is_eight_point_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_is_eight_point_nine_l1301_130196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_existence_l1301_130108

noncomputable def f (a x : Real) : Real := Real.sin x ^ 2 + a * Real.cos x + (5/8) * a - 3/2

theorem max_value_existence :
  ∃ a : Real, a = 3/2 ∧
    (∀ x : Real, 0 ≤ x ∧ x ≤ Real.pi/2 → f a x ≤ 1) ∧
    (∃ x : Real, 0 ≤ x ∧ x ≤ Real.pi/2 ∧ f a x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_existence_l1301_130108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l1301_130165

/-- Represents the dimensions of a rectangular prism -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The original box dimensions -/
def originalBox : Dimensions := ⟨20, 15, 12⟩

/-- The dimensions of each cube removed from the corners -/
def removedCube : Dimensions := ⟨4, 4, 4⟩

/-- The dimensions of the additional rectangular prism removed -/
def removedPrism : Dimensions := ⟨8, 4, 2⟩

/-- The number of cubes removed from the corners -/
def numRemovedCubes : ℕ := 8

/-- Calculates the total volume removed -/
noncomputable def totalVolumeRemoved : ℝ :=
  (numRemovedCubes : ℝ) * volume removedCube + volume removedPrism

/-- Calculates the percentage of volume removed -/
noncomputable def percentageRemoved : ℝ :=
  (totalVolumeRemoved / volume originalBox) * 100

/-- Theorem stating that the percentage of volume removed is 16% -/
theorem volume_removed_percentage :
  percentageRemoved = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l1301_130165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_combination_l1301_130103

theorem square_root_combination (b : ℝ) : (Real.sqrt (5*b) = Real.sqrt (3+2*b)) → -b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_combination_l1301_130103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1301_130128

noncomputable section

/-- An ellipse C defined by mx^2 + (1-m)y^2 = 1 with its focus on the y-axis -/
def Ellipse (m : ℝ) := {(x, y) : ℝ × ℝ | m * x^2 + (1 - m) * y^2 = 1}

/-- The condition that the focus is on the y-axis -/
def FocusOnYAxis (m : ℝ) := 1 / (1 - m) > 1 / m

/-- The length of the minor axis of the ellipse -/
def MinorAxisLength (m : ℝ) := 2 * Real.sqrt (1 / m)

theorem ellipse_properties (m : ℝ) (h : FocusOnYAxis m) :
  (1/2 < m ∧ m < 1) ∧
  (∀ l, l = MinorAxisLength m → 2 < l ∧ l < 2 * Real.sqrt 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1301_130128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dorm_inequalities_l1301_130149

/-- Represents the number of dormitories -/
def x : ℕ → ℕ := fun _ => 0  -- Placeholder function, to be replaced with actual logic

/-- Represents the total number of students -/
def total_students (x : ℕ) : ℕ := 4 * x + 19

/-- Represents the number of students in the last dormitory when accommodating 6 per room -/
def last_dorm_students (x : ℕ) : ℤ := (4 * x + 19) - 6 * (x - 1)

/-- The theorem stating the system of inequalities that x must satisfy -/
theorem dorm_inequalities (x : ℕ) :
  1 ≤ last_dorm_students x ∧ last_dorm_students x ≤ 5 := by
  sorry

#check dorm_inequalities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dorm_inequalities_l1301_130149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_from_tan_cot_sum_l1301_130157

theorem tan_sum_from_tan_cot_sum (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 35)
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 40) : 
  Real.tan (x + y) = 280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_from_tan_cot_sum_l1301_130157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1301_130107

theorem power_equation_solution (x y : ℝ) 
  (h1 : (3 : ℝ)^x * (4 : ℝ)^y = 531441)
  (h2 : x - y = 12) : 
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1301_130107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_ratio_l1301_130125

theorem sphere_volume_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_surface_ratio : (4 * Real.pi * a^2, 4 * Real.pi * b^2, 4 * Real.pi * c^2) = (1, 2, 3)) :
  ((4/3 * Real.pi * a^3, 4/3 * Real.pi * b^3, 4/3 * Real.pi * c^3) = (1, 2 * Real.sqrt 2, 3 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_ratio_l1301_130125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_travel_distance_l1301_130144

/-- The total distance Lindy travels when Jack and Christina meet -/
noncomputable def lindyDistance (initialDistance jackSpeed christinaSpeed lindySpeed : ℝ) : ℝ :=
  let meetingTime := initialDistance / (jackSpeed + christinaSpeed)
  lindySpeed * meetingTime

/-- Theorem stating that Lindy travels 240 feet given the problem conditions -/
theorem lindy_travel_distance :
  lindyDistance 270 4 5 8 = 240 := by
  -- Unfold the definition of lindyDistance
  unfold lindyDistance
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lindy_travel_distance_l1301_130144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_from_origin_l1301_130101

noncomputable def distance_from_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

theorem farthest_point_from_origin :
  let points : List (ℝ × ℝ) := [(1, 5), (2, -3), (4, -1), (3, 3), (-2.5, 2)]
  (∀ p ∈ points, distance_from_origin 1 5 ≥ distance_from_origin p.1 p.2) ∧
  (∃ p ∈ points, p ≠ (1, 5) ∧ distance_from_origin 1 5 > distance_from_origin p.1 p.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_from_origin_l1301_130101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l1301_130116

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_theorem :
  (∀ x < 0, x * f' x - 2 * f x > 0) →
  (∀ x, f (x + 2023) - (x + 2023)^2 * f (-1) < 0 ↔ -2024 < x ∧ x < -2023) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_l1301_130116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_speed_B_to_C_l1301_130159

/-- Calculates the speed from B to C given the conditions of Tom's journey -/
noncomputable def speed_B_to_C (speed_A_to_B : ℝ) (avg_speed : ℝ) : ℝ :=
  let d := 1  -- We can use any positive real number for d
  let time_A_to_B := 2 * d / speed_A_to_B
  let total_distance := 3 * d
  let total_time := total_distance / avg_speed
  let time_B_to_C := total_time - time_A_to_B
  d / time_B_to_C

/-- Theorem stating that Tom's speed from B to C is 20 mph -/
theorem tom_speed_B_to_C : speed_B_to_C 60 36 = 20 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_speed_B_to_C_l1301_130159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_abc_l1301_130189

-- Define the constants
noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.6
noncomputable def b : ℝ := (0.3 : ℝ) ^ (0.6 : ℝ)
noncomputable def c : ℝ := (0.6 : ℝ) ^ (0.3 : ℝ)

-- State the theorem
theorem ordering_of_abc : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_abc_l1301_130189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_product_a_less_than_one_l1301_130142

open Nat BigOperators

def a (n : ℕ) : ℚ :=
  ∑ p in Nat.primeFactors n, 1 / (p : ℚ)

theorem sum_product_a_less_than_one (N : ℕ) (hN : N ≥ 2) :
  ∑ n in Finset.range (N - 1), ∏ i in Finset.range (n + 1), a (i + 2) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_product_a_less_than_one_l1301_130142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_fraction_l1301_130163

theorem real_part_of_complex_fraction : ∃ z : ℂ, z = (2 - I) / (1 + I) ∧ z.re = 1 / 2 := by
  -- Define the complex number z
  let z : ℂ := (2 - I) / (1 + I)
  
  -- State that the real part of z is equal to 1/2
  have h : z.re = 1 / 2 := by
    -- The proof would go here, but we'll use sorry for now
    sorry
  
  -- Conclude the theorem
  exact ⟨z, rfl, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_fraction_l1301_130163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_when_a_zero_f_positive_implies_a_positive_l1301_130191

/-- The function f(x) = (x - 1)ln(x) + ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.log x + a * x

theorem f_monotonicity_when_a_zero :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f 0 x₁ > f 0 x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f 0 x₁ < f 0 x₂) := by
  sorry

theorem f_positive_implies_a_positive (a : ℝ) :
  (∀ x, x > 0 → f a x > 0) → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_when_a_zero_f_positive_implies_a_positive_l1301_130191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l1301_130136

/-- Represents the length of a candle stub as a function of time -/
noncomputable def candleStub (initialLength burnTime : ℝ) (t : ℝ) : ℝ :=
  initialLength * (burnTime - t) / burnTime

/-- Theorem stating the existence of a time when one candle stub is three times the length of the other -/
theorem candle_lighting_time 
  (initialLength : ℝ) 
  (h_positive : initialLength > 0) :
  ∃ (t : ℝ), 
    0 ≤ t ∧ t ≤ 60 ∧
    candleStub initialLength 240 (60 - t) = 
    3 * candleStub initialLength 360 (60 - t) ∧
    t = 60 := by
  sorry

#check candle_lighting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l1301_130136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1301_130174

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The first term of our geometric series -/
noncomputable def a : ℝ := 2

/-- The common ratio of our geometric series -/
noncomputable def r : ℝ := 1 / 4

/-- Theorem: The sum of the infinite geometric series 2 + 2(1/4) + 2(1/4)² + 2(1/4)³ + ... is 8/3 -/
theorem geometric_series_sum : geometric_sum a r = 8 / 3 := by
  -- Unfold the definitions
  unfold geometric_sum a r
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1301_130174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_can_prevent_white_adjacency_l1301_130138

/-- Represents a position on the 23x23 grid -/
structure Position where
  x : Fin 23
  y : Fin 23

/-- Represents the state of the game board -/
structure GameState where
  white1 : Position
  white2 : Position
  black1 : Position
  black2 : Position

/-- Checks if two positions are adjacent -/
def adjacent (p1 p2 : Position) : Prop :=
  (p1.x = p2.x ∧ (p1.y.val + 1 = p2.y.val ∨ p2.y.val + 1 = p1.y.val)) ∨
  (p1.y = p2.y ∧ (p1.x.val + 1 = p2.x.val ∨ p2.x.val + 1 = p1.x.val))

/-- Represents a valid move on the board -/
def validMove (start finish : Position) : Prop :=
  adjacent start finish

/-- Represents the initial game state -/
def initialState : GameState :=
  { white1 := ⟨0, 0⟩,
    white2 := ⟨22, 22⟩,
    black1 := ⟨0, 22⟩,
    black2 := ⟨22, 0⟩ }

/-- The main theorem to be proved -/
theorem black_can_prevent_white_adjacency :
  ∃ (blackStrategy : GameState → Position → Position),
    ∀ (gameStates : Nat → GameState),
    gameStates 0 = initialState →
    (∀ n : Nat,
      (validMove (gameStates n).white1 (gameStates (n+1)).white1 ∧ (gameStates n).white2 = (gameStates (n+1)).white2) ∨
      (validMove (gameStates n).white2 (gameStates (n+1)).white2 ∧ (gameStates n).white1 = (gameStates (n+1)).white1)) →
    (∀ n : Nat,
      let nextBlackMove := if n % 2 = 0 then blackStrategy (gameStates (n+1)) (gameStates (n+1)).black1
                           else blackStrategy (gameStates (n+1)) (gameStates (n+1)).black2
      validMove (if n % 2 = 0 then (gameStates (n+1)).black1 else (gameStates (n+1)).black2) nextBlackMove) →
    ∀ n : Nat, ¬(adjacent (gameStates n).white1 (gameStates n).white2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_can_prevent_white_adjacency_l1301_130138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1301_130151

def sequenceList : List ℕ := [2, 5, 11, 20, 32, 47]

def differences (s : List ℕ) : List ℕ :=
  List.zipWith (fun a b => b - a) s (List.tail s)

def isConsecutiveMultiplesOf3 (l : List ℕ) : Prop :=
  ∀ (i : ℕ), i < l.length → l[i]! = (i + 1) * 3

theorem sequence_property : 
  isConsecutiveMultiplesOf3 (differences sequenceList) ∧ 
  sequenceList[4]! = 32 := by
  sorry

#eval sequenceList[4]!
#eval differences sequenceList

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l1301_130151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_range_l1301_130178

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 4

-- Define points A and B
def point_A (m : ℝ) : ℝ × ℝ := (-m, 0)
def point_B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the condition for ∠APB = 90°
def right_angle (A B P : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

-- Main theorem
theorem circle_point_range (m : ℝ) :
  m > 0 →
  (∃ P : ℝ × ℝ, circle_C P.1 P.2 ∧ right_angle (point_A m) (point_B m) P) →
  3 ≤ m ∧ m ≤ 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_range_l1301_130178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1301_130156

def acute_triangle (A B C : ℝ) : Prop := 0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2

theorem triangle_problem (A B C a b c : ℝ) 
  (h_acute : acute_triangle A B C)
  (h_cos : Real.cos (B + C) = -(Real.sqrt 3) / 3 * Real.sin (2 * A))
  (h_a : a = 7)
  (h_b : b = 5) :
  A = Real.pi/3 ∧ (1/2 * a * b * Real.sin C = 10 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1301_130156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trace_equality_for_commuting_matrices_l1301_130183

open Matrix

theorem trace_equality_for_commuting_matrices (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ) 
  (h_comm : A * B = B * A)
  (h_A_pow : A ^ n = 1)
  (h_B_pow : B ^ n = 1)
  (h_trace_AB : Matrix.trace (A * B) = n) :
  Matrix.trace A = Matrix.trace B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trace_equality_for_commuting_matrices_l1301_130183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l1301_130186

theorem solution_count : ∃ (S : Finset ℤ), 
  (∀ c ∈ S, 100 ≤ c ∧ c ≤ 2000) ∧
  (∀ c ∈ S, ∃ x : ℝ, 5 * ⌊x⌋ + 3 * ⌈x⌉ = c) ∧
  (∀ c : ℤ, 100 ≤ c ∧ c ≤ 2000 → (∃ x : ℝ, 5 * ⌊x⌋ + 3 * ⌈x⌉ = c) → c ∈ S) ∧
  S.card = 238 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l1301_130186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_intersection_is_hyperbola_l1301_130140

/-- Given a circle centered at the origin with radius a, prove that the locus of
    the intersection point P of lines A'M and AM' forms a hyperbola. -/
theorem locus_of_intersection_is_hyperbola (a : ℝ) (h : a > 0) :
  ∃ (P : ℝ × ℝ), 
    (P.1^2 / a^2) - (P.2^2 / (a/2)^2) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_intersection_is_hyperbola_l1301_130140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1301_130173

noncomputable def diamond (m n : ℝ) : ℝ := Real.sqrt (m^2 + n^2)

theorem diamond_calculation :
  (diamond (diamond 8 15) (diamond 15 (-8))) = 17 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1301_130173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_through_vertices_conic_tangent_to_sides_l1301_130194

-- Define a triangle in trilinear coordinates
structure TrilinearTriangle where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a conic in trilinear coordinates
structure Conic where
  p : ℝ
  q : ℝ
  r : ℝ

-- Theorem for conic passing through all vertices
theorem conic_through_vertices (t : TrilinearTriangle) (c : Conic) :
  ∃ p q r : ℝ, c.p * t.x * t.y + c.q * t.x * t.z + c.r * t.z * t.y = 0 :=
by sorry

-- Theorem for conic tangent to all sides or their extensions
theorem conic_tangent_to_sides (t : TrilinearTriangle) (c : Conic) :
  ∃ p q r : ℝ, 
    c.p * t.x^2 + c.q * t.y^2 + c.r * t.z^2 = 
    2 * ((Real.sqrt (c.p * c.q) * t.x * t.y) +
         (Real.sqrt (c.p * c.r) * t.x * t.z) +
         (Real.sqrt (c.q * c.r) * t.y * t.z)) ∨
    c.p * t.x^2 + c.q * t.y^2 + c.r * t.z^2 = 
    2 * ((Real.sqrt (c.p * c.q) * t.x * t.y) -
         (Real.sqrt (c.p * c.r) * t.x * t.z) -
         (Real.sqrt (c.q * c.r) * t.y * t.z)) ∨
    c.p * t.x^2 + c.q * t.y^2 + c.r * t.z^2 = 
    2 * (-(Real.sqrt (c.p * c.q) * t.x * t.y) +
         (Real.sqrt (c.p * c.r) * t.x * t.z) -
         (Real.sqrt (c.q * c.r) * t.y * t.z)) ∨
    c.p * t.x^2 + c.q * t.y^2 + c.r * t.z^2 = 
    2 * (-(Real.sqrt (c.p * c.q) * t.x * t.y) -
         (Real.sqrt (c.p * c.r) * t.x * t.z) +
         (Real.sqrt (c.q * c.r) * t.y * t.z)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_through_vertices_conic_tangent_to_sides_l1301_130194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_roots_distance_decreasing_before_vertex_y1_less_than_y2_l1301_130192

namespace ParabolaProperties

variable (m : ℝ)

/-- The parabola function --/
def f (x : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 9

/-- The derivative of the parabola function --/
def f' (x : ℝ) : ℝ := 2*x - 2*m

/-- Points where the parabola intersects the x-axis --/
def roots : Set ℝ := {x | f m x = 0}

/-- Statement: The minimum value of the parabola is -9 --/
theorem min_value : ∃ x₀, ∀ x, f m x ≥ f m x₀ ∧ f m x₀ = -9 := by sorry

/-- Statement: The length of the segment between the roots is 6 --/
theorem roots_distance : ∃ a b, a ∈ roots m ∧ b ∈ roots m ∧ |a - b| = 6 := by sorry

/-- Statement: For x < m-1, the derivative is negative --/
theorem decreasing_before_vertex : ∀ x, x < m - 1 → f' m x < 0 := by sorry

/-- Statement: y1 < y2 for points P(m+1, y1) and Q(m-3, y2) on the parabola --/
theorem y1_less_than_y2 : f m (m + 1) < f m (m - 3) := by sorry

end ParabolaProperties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_roots_distance_decreasing_before_vertex_y1_less_than_y2_l1301_130192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1301_130119

/-- Calculates the speed of a train in km/hr given its length, the platform length, and the time to pass the platform. -/
noncomputable def train_speed (train_length platform_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / time
  speed_ms * 3.6

/-- Theorem stating that a train of length 360 meters passing a platform of length 140 meters in 40 seconds has a speed of 45 km/hr. -/
theorem train_speed_calculation : train_speed 360 140 40 = 45 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check train_speed 360 140 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1301_130119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_P_l1301_130152

-- Define the curve in polar coordinates
noncomputable def C (θ : Real) : Real := 2 * Real.cos θ

-- Define the point P in polar coordinates
noncomputable def P : Real × Real := (1, Real.pi)

-- Helper function to calculate distance in polar coordinates
noncomputable def distance (p1 p2 : Real × Real) : Real :=
  let (r1, θ1) := p1
  let (r2, θ2) := p2
  Real.sqrt (r1^2 + r2^2 - 2*r1*r2*Real.cos (θ1 - θ2))

-- Statement of the theorem
theorem max_distance_to_P :
  ∃ (max_dist : Real),
    (∀ (θ : Real), distance (C θ, θ) P ≤ max_dist) ∧
    (∃ (θ_max : Real), distance (C θ_max, θ_max) P = max_dist) ∧
    max_dist = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_P_l1301_130152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l1301_130126

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + 1 / (x + 1)

-- State the theorem
theorem f_bounds (x : ℝ) (hx : x ∈ Set.Icc 0 1) : 
  f x ≥ 1 - x + x^2 ∧ 3/4 < f x ∧ f x ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l1301_130126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l1301_130137

/-- The function f(x) = sin(4x) + cos(6x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (4 * x) + Real.cos (6 * x)

/-- The period of a function is the smallest positive real number p such that
    f(x + p) = f(x) for all x -/
def isPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- The period of f(x) = sin(4x) + cos(6x) is π -/
theorem period_of_f : isPeriod f π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l1301_130137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_280_l1301_130190

/-- The length of a train given its speed, platform length, and time to cross the platform -/
noncomputable def train_length (speed : ℝ) (platform_length : ℝ) (time : ℝ) : ℝ :=
  speed * (5/18) * time - platform_length

/-- Theorem: The length of the train is 280 meters -/
theorem train_length_is_280 :
  train_length 72 240 26 = 280 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_280_l1301_130190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1301_130199

noncomputable section

/- Define the arithmetic sequence and its sum -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (a₁ + arithmetic_sequence a₁ d n) / 2

/- Define sets A and B -/
def set_A (a₁ d : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ n : ℕ+, p.1 = arithmetic_sequence a₁ d n ∧
                  p.2 = sum_arithmetic_sequence a₁ d n / n}

def set_B : Set (ℝ × ℝ) :=
  {p | (1/4) * p.1^2 - p.2^2 = 1}

/- Main theorem -/
theorem arithmetic_sequence_properties (a₁ d : ℝ) (h : d ≠ 0) :
  (∀ p ∈ set_A a₁ d, p.2 = (1/2) * (p.1 + a₁)) ∧
  (∃! p, p ∈ set_A a₁ d ∩ set_B) ∧
  (∃ a₁ : ℝ, a₁ ≠ 0 ∧ set_A a₁ d ∩ set_B = ∅) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1301_130199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_sixfold_property_l1301_130115

theorem smallest_number_with_sixfold_property : ∃ (n : ℕ), 
  (n = 153846) ∧ 
  (∀ m : ℕ, m < n → ¬(m % 10 = 6 ∧ 6 * 10^(Nat.log 10 m) + m / 10 = 4 * m)) ∧
  (153846 % 10 = 6) ∧
  (6 * 10^(Nat.log 10 153846) + 153846 / 10 = 4 * 153846) := by
  sorry

#check smallest_number_with_sixfold_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_sixfold_property_l1301_130115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_parallel_planes_l1301_130120

/-- A plane in 3D space -/
structure Plane where

/-- A line in 3D space -/
structure Line where

/-- Two planes are parallel -/
def parallel_planes (α β : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line) (α : Plane) : Prop := sorry

theorem perpendicular_to_parallel_planes (α β : Plane) (l : Line) :
  parallel_planes α β → perpendicular_line_plane l α → perpendicular_line_plane l β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_parallel_planes_l1301_130120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_grade_70_to_80_count_eighth_grade_median_ninth_grade_mode_ninth_grade_lower_variance_excellent_scores_estimate_l1301_130170

-- Define the score data for each grade
def eighth_grade_scores : List ℕ := [74, 76, 79, 81, 84, 86, 87, 90, 90, 93]
def ninth_grade_scores : List ℕ := [76, 81, 81, 83, 84, 84, 84, 85, 90, 92]

-- Define the score ranges and frequencies
def eighth_grade_ranges : List (ℕ × ℕ × ℕ) := [(70, 80, 3), (80, 90, 4), (90, 100, 3)]
def ninth_grade_ranges : List (ℕ × ℕ × ℕ) := [(70, 80, 1), (80, 90, 7), (90, 100, 2)]

-- Define the statistics for each grade
structure GradeStats where
  average : ℕ
  median : ℕ
  mode : ℕ
  variance : Float

def eighth_grade_stats : GradeStats := {
  average := 84,
  median := 85,
  mode := 90,
  variance := 36.4
}

def ninth_grade_stats : GradeStats := {
  average := 84,
  median := 84,
  mode := 84,
  variance := 18.4
}

-- Define the total number of students in each grade
def total_students : ℕ := 100

-- Theorem statements
theorem eighth_grade_70_to_80_count :
  (eighth_grade_ranges.find? (fun r => r.1 = 70 ∧ r.2 = 80)).map (fun x => x.2.2) = some 3 := by sorry

theorem eighth_grade_median :
  eighth_grade_stats.median = 85 := by sorry

theorem ninth_grade_mode :
  ninth_grade_stats.mode = 84 := by sorry

theorem ninth_grade_lower_variance :
  ninth_grade_stats.variance < eighth_grade_stats.variance := by sorry

theorem excellent_scores_estimate :
  let eighth_excellent := (eighth_grade_scores.filter (· ≥ 85)).length * 10
  let ninth_excellent := (ninth_grade_scores.filter (· ≥ 85)).length * 10
  eighth_excellent = 50 ∧ ninth_excellent = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_grade_70_to_80_count_eighth_grade_median_ninth_grade_mode_ninth_grade_lower_variance_excellent_scores_estimate_l1301_130170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_propositions_l1301_130141

noncomputable section

theorem three_true_propositions :
  (∀ x : ℝ, (2 : ℝ)^x > 0) ∧
  (∃ α β : ℝ, Real.sin (α + β) = Real.sin α + Real.sin β) ∧
  (∀ a b : ℝ, (a > b → ∀ c : ℝ, a * c^2 > b * c^2) ∧ ¬(∀ c : ℝ, a * c^2 > b * c^2 → a > b)) ∧
  (¬(∀ A B : ℝ, 0 < A ∧ A < π ∧ 0 < B ∧ B < π → (A > B → Real.sin A > Real.sin B))) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_propositions_l1301_130141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_encryption_decryption_l1301_130166

-- Define the encryption function
noncomputable def f (x : ℝ) : ℝ := 2^x - 2

-- State the theorem
theorem encryption_decryption :
  f 3 = 6 → ∃ x : ℝ, f x = 14 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_encryption_decryption_l1301_130166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_obtuse_angle_in_pentagon_l1301_130111

/-- Pentagon vertices -/
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (3, 1)
def C : ℝ × ℝ := (5, 1)
def D : ℝ × ℝ := (5, 6)
def E : ℝ × ℝ := (-1, 6)

/-- Area of the pentagon -/
def pentagon_area : ℝ := 23

/-- Probability that ∠AQB is obtuse when Q is randomly chosen from the interior of the pentagon -/
noncomputable def prob_obtuse_angle : ℝ := (5 * Real.pi) / 46

/-- Theorem stating the probability of ∠AQB being obtuse -/
theorem prob_obtuse_angle_in_pentagon :
  prob_obtuse_angle = (5 * Real.pi) / 46 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_obtuse_angle_in_pentagon_l1301_130111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_4_equal_coefficients_l1301_130147

-- Define the binomial expansion (marked as noncomputable due to Real.sqrt)
noncomputable def binomial_expansion (x : ℝ) := (x - 2 / Real.sqrt x) ^ 10

-- Define the coefficient of x^k in the expansion
def coefficient (k : ℕ) : ℤ := (-2)^k * (Nat.choose 10 k : ℤ)

-- Define the exponent of x in the general term
def exponent (k : ℕ) : ℚ := 10 - (3 * k) / 2

-- Statement I
theorem coefficient_of_x_4 :
  ∃ k : ℕ, exponent k = 4 ∧ coefficient k = 3360 :=
sorry

-- Statement II
theorem equal_coefficients :
  ∃ r : ℕ, r = 1 ∧ Nat.choose 10 (3*r - 1) = Nat.choose 10 (r + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_4_equal_coefficients_l1301_130147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1301_130160

noncomputable section

-- Define the function
def f (x : ℝ) : ℝ := (x^2 + 4*x + 3) / (x + 2)

-- Define the domain of the function
def domain : Set ℝ := {x : ℝ | x ≠ -2}

-- Define the range of the function
def range : Set ℝ := Set.Ioi (-1) ∪ Set.Iio (-1)

-- Theorem statement
theorem range_of_f :
  {y : ℝ | ∃ x ∈ domain, f x = y} = range := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1301_130160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_statement_is_minor_premise_l1301_130105

-- Define the structure of a syllogism
structure Syllogism where
  major_premise : String
  minor_premise : String
  conclusion : String

-- Define the specific syllogism from the problem
def school_syllogism : Syllogism :=
  { major_premise := "Xiushui First Middle School calls on all students to learn from Lei Feng to do good deeds, requiring each student to do at least one good deed",
    minor_premise := "Zhang San is a student in the second year of high school at Xiushui First Middle School",
    conclusion := "Zhang San must do at least one good deed" }

-- Theorem stating that the second statement is the minor premise
theorem second_statement_is_minor_premise :
  school_syllogism.minor_premise = "Zhang San is a student in the second year of high school at Xiushui First Middle School" := by
  rfl

#check second_statement_is_minor_premise

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_statement_is_minor_premise_l1301_130105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_is_five_l1301_130110

noncomputable def sample (a : ℝ) : List ℝ := [a, 3, 5, 7]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (λ x => (x - μ)^2)).sum / xs.length

theorem sample_variance_is_five (a b : ℝ) :
  b = mean (sample a) ∧ 
  a * b = 4 ∧ 
  a + b = 5 →
  variance (sample a) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_is_five_l1301_130110
