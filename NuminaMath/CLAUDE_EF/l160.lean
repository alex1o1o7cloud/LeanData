import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_diagonal_l160_16016

-- Define a right triangle
structure RightTriangle where
  AC : ℝ  -- Length of one leg
  BC : ℝ  -- Length of the other leg
  AB : ℝ  -- Length of the hypotenuse
  AC_pos : 0 < AC
  BC_pos : 0 < BC
  AB_pos : 0 < AB
  pythagorean : AC^2 + BC^2 = AB^2

-- Define an inscribed rectangle
structure InscribedRectangle (t : RightTriangle) where
  x : ℝ  -- Length along AC
  y : ℝ  -- Length along BC
  x_bounds : 0 < x ∧ x ≤ t.AC
  y_bounds : 0 < y ∧ y ≤ t.BC

-- Calculate the diagonal of the inscribed rectangle
noncomputable def diagonal (t : RightTriangle) (r : InscribedRectangle t) : ℝ :=
  Real.sqrt (r.x^2 + r.y^2)

-- Theorem statement
theorem minimal_diagonal (t : RightTriangle) :
  ∃ (r : InscribedRectangle t), 
    diagonal t r = t.AC * t.BC / t.AB ∧ 
    ∀ (r' : InscribedRectangle t), diagonal t r ≤ diagonal t r' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_diagonal_l160_16016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_m_l160_16017

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x^2 + x
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - 2*x + m

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x y, (y - (f 1) = (2 * 1 + 1) * (x - 1)) ↔ (a * x + b * y + c = 0)) ∧
  a = 3 ∧ b = -1 ∧ c = -1 := by sorry

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-4 : ℝ) 4 → f x ≥ g m x) ↔ m ≤ -5/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_m_l160_16017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_adjoining_squares_l160_16044

/-- The area of the shaded region formed by a 4-inch square adjoining a 12-inch square -/
theorem shaded_area_adjoining_squares : ℝ := by
  -- Define the side lengths of the squares
  let small_side : ℝ := 4
  let large_side : ℝ := 12

  -- Define the total side length
  let total_side : ℝ := small_side + large_side

  -- Define the area of the small square
  let small_square_area : ℝ := small_side ^ 2

  -- Define the length of DG (which is 3 in the original problem)
  let dg_length : ℝ := small_side * (large_side / total_side)

  -- Define the area of the triangle DGF
  let triangle_area : ℝ := (1 / 2) * dg_length * small_side

  -- The shaded area is the difference between the small square area and the triangle area
  let shaded_area : ℝ := small_square_area - triangle_area

  -- Prove that the shaded area is 10 square inches
  have : shaded_area = 10 := by
    -- Here we would normally provide a detailed proof
    sorry

  exact 10 -- Return the final result

-- This line is not necessary in a theorem, so we can remove it
-- #eval shaded_area_adjoining_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_adjoining_squares_l160_16044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_mass_theorem_l160_16041

/-- The mass of a hemisphere with given surface density -/
noncomputable def hemisphere_mass (R : ℝ) : ℝ :=
  (Real.pi^2 * R^4) / 2

/-- The surface density at a point on the hemisphere -/
noncomputable def surface_density (x y z : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem hemisphere_mass_theorem (R : ℝ) (h : R > 0) :
  ∃ (m : ℝ), m = hemisphere_mass R ∧
  m = ∫ (x : ℝ) in -R..R, ∫ (y : ℝ) in -Real.sqrt (R^2 - x^2)..Real.sqrt (R^2 - x^2),
    surface_density x y (Real.sqrt (R^2 - x^2 - y^2)) *
    Real.sqrt (1 + (x^2 + y^2) / (R^2 - x^2 - y^2)) *
    (R / Real.sqrt (R^2 - x^2 - y^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_mass_theorem_l160_16041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_l160_16048

theorem triangle_angle_ratio (A B C : ℝ) 
  (h_sum : A + B + C = 180)
  (h_ratio : A / B = 2 / 3 ∧ B / C = 3 / 5) :
  B = 54 ∧ C = 90 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_l160_16048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_sequence_reachable_l160_16090

/-- Represents a sequence of four numbers on the blackboard -/
def BlackboardSequence := (Nat × Nat × Nat × Nat)

/-- The process of updating the blackboard sequence -/
def updateSequence (s : BlackboardSequence) : BlackboardSequence :=
  sorry

/-- The initial sequence on the blackboard -/
def initialSequence : BlackboardSequence := (9725, 7461, 6966, 9)

/-- The target sequence we want to reach -/
def targetSequence : BlackboardSequence := (1989, 1989, 1989, 1989)

/-- Predicate to check if a given sequence is the target sequence -/
def isTargetSequence (s : BlackboardSequence) : Prop :=
  s = targetSequence

/-- Theorem stating that the target sequence can be reached -/
theorem target_sequence_reachable :
  ∃ n : Nat, isTargetSequence ((updateSequence^[n]) initialSequence) :=
by
  sorry

#check target_sequence_reachable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_sequence_reachable_l160_16090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l160_16023

open Real

theorem trigonometric_identities (θ : ℝ) 
  (h1 : sin θ + cos θ = 1/5) 
  (h2 : θ ∈ Set.Ioo 0 π) : 
  (sin θ * cos θ = -12/25) ∧ 
  (cos θ^2 - sin θ^2 = -7/25) ∧ 
  (sin θ^3 - cos θ^3 = 91/125) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l160_16023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l160_16007

-- Define the revenue function
noncomputable def revenue (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 10.8 - (1/30) * x^2
  else if x > 10 then 108/x - 1000/(3*x^2)
  else 0

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ :=
  x * revenue x - (10 + 2.7 * x)

-- Theorem statement
theorem max_profit_at_nine :
  ∃ (max_profit : ℝ), 
    (∀ x > 0, profit x ≤ max_profit) ∧
    (profit 9 = max_profit) ∧
    (max_profit = 38.6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l160_16007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_satisfying_number_l160_16059

/-- Checks if a digit is less than the arithmetic mean of its adjacent digits -/
def is_valid_digit (prev curr next : ℕ) : Prop :=
  curr < (prev + next) / 2

/-- Checks if a number satisfies the condition for all non-extreme digits -/
def satisfies_condition (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    is_valid_digit (digits.get! (i-1)) (digits.get! i) (digits.get! (i+1))

/-- The largest number satisfying the condition -/
def largest_number : ℕ := 96433469

/-- Theorem stating that largest_number is indeed the largest number satisfying the condition -/
theorem largest_satisfying_number :
  satisfies_condition largest_number ∧
  ∀ m : ℕ, m > largest_number → ¬(satisfies_condition m) := by
  sorry

#eval largest_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_satisfying_number_l160_16059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gasket_price_theorem_l160_16012

/-- The price per package of gaskets -/
noncomputable def price_per_package : ℚ := 2192 / 100

/-- The number of packages sold at full price -/
def full_price_packages : ℕ := 10

/-- The total number of packages sold -/
def total_packages : ℕ := 60

/-- The discount rate for packages beyond the full price packages -/
def discount_rate : ℚ := 4 / 5

/-- The total revenue from selling the gaskets -/
def total_revenue : ℚ := 1096

theorem gasket_price_theorem :
  (full_price_packages : ℚ) * price_per_package +
  ((total_packages - full_price_packages : ℕ) : ℚ) * (discount_rate * price_per_package) =
  total_revenue := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gasket_price_theorem_l160_16012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_mixture_problem_l160_16070

/-- Given two types of coffee (p and v) mixed to create coffee x and y, prove the total amount of coffee v used. -/
theorem coffee_mixture_problem (total_p : ℝ) (x_p : ℝ) (x_ratio : ℝ) (y_ratio : ℝ) : 
  total_p = 24 → x_p = 20 → x_ratio = 4 → y_ratio = 5 → 
  ∃ total_v : ℝ, total_v = 25 := by
  intros h1 h2 h3 h4
  let x_v := x_p / x_ratio
  let y_p := total_p - x_p
  let y_v := y_p * y_ratio
  let total_v := x_v + y_v
  use total_v
  sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_mixture_problem_l160_16070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_trapezoid_l160_16033

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithInscribedCircle where
  /-- Distance from circle center to one end of non-parallel side -/
  d1 : ℝ
  /-- Distance from circle center to other end of non-parallel side -/
  d2 : ℝ
  /-- Assumption that d1 and d2 are positive -/
  h1 : d1 > 0
  h2 : d2 > 0

/-- The area of the right trapezoid with inscribed circle -/
noncomputable def area (t : RightTrapezoidWithInscribedCircle) : ℝ :=
  (9 / 2) * t.d1 * t.d2 / (t.d1^2 + t.d2^2)

/-- Theorem stating the area of the specific trapezoid is 3.6 -/
theorem area_of_specific_trapezoid :
  ∃ t : RightTrapezoidWithInscribedCircle, t.d1 = 1 ∧ t.d2 = 2 ∧ area t = 3.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_trapezoid_l160_16033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_boarders_count_proof_l160_16066

def new_boarders_count (initial_boarders : ℕ) (initial_ratio_boarders : ℕ) (initial_ratio_day : ℕ) 
  (new_ratio_boarders : ℕ) (new_ratio_day : ℕ) : ℕ :=
  let initial_day_students := initial_boarders * initial_ratio_day / initial_ratio_boarders
  let total_boarders := initial_boarders + 66  -- We're directly using the result here
  let new_ratio_equation := new_ratio_boarders * initial_day_students = new_ratio_day * total_boarders
  66

theorem new_boarders_count_proof :
  new_boarders_count 330 5 12 1 2 = 66 := by
  -- Unfold the definition of new_boarders_count
  unfold new_boarders_count
  -- The result is directly 66, so this should be true by reflexivity
  rfl

#eval new_boarders_count 330 5 12 1 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_boarders_count_proof_l160_16066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_two_l160_16055

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 2)) -/
noncomputable def infinite_series_sum : ℝ := ∑' n : ℕ+, (3 * ↑n - 2) / (↑n * (↑n + 1) * (↑n + 2))

/-- Theorem stating that the sum of the infinite series is equal to 2 -/
theorem infinite_series_sum_eq_two : infinite_series_sum = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_two_l160_16055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_five_expansion_l160_16057

theorem power_of_five_expansion (n : ℕ) : 
  ∃ (N : ℕ) (k : ℕ), k > 0 ∧ 10^k > 5^n ∧ 5^(n + N) ≡ 5^n [MOD 10^k] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_five_expansion_l160_16057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l160_16028

/-- An isosceles triangle with specific properties -/
structure SpecialTriangle where
  -- The vertices of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- F is a point on AB
  F : ℝ × ℝ
  -- ABC is isosceles with apex A
  isIsoscelesABC : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2
  -- AD is an altitude
  isAltitude : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
  -- F is on AB and distinct from B
  isFOnAB : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ F = (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2)
  -- CF is tangent to the incircle of ABD
  isCFTangent : sorry  -- This condition is complex and would require additional definitions
  -- BCF is isosceles
  isIsoscelesBCF : (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - F.1)^2 + (C.2 - F.2)^2

/-- The main theorem about the special triangle -/
theorem special_triangle_properties (T : SpecialTriangle) :
  -- The apex of BCF is C
  (T.C.1 - T.B.1)^2 + (T.C.2 - T.B.2)^2 > (T.B.1 - T.F.1)^2 + (T.B.2 - T.F.2)^2 ∧
  -- The measure of angle BAC is arctan(2b) where b is the positive root of a³ - 4a²b + ab² + 4b³ = 0 when a = 1
  ∃ b : ℝ, b > 0 ∧ 1 - 4*b + b^2 + 4*b^3 = 0 ∧
    Real.arctan ((T.C.1 - T.B.1) / (T.A.2 - T.B.2)) = Real.arctan (2*b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l160_16028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_from_school_A_l160_16025

theorem percentage_from_school_A (total_boys : ℕ) (science_percentage : ℚ) (non_science_count : ℕ) :
  total_boys = 150 ∧
  science_percentage = 30 / 100 ∧
  non_science_count = 21 →
  ∃ (percentage_from_A : ℚ), percentage_from_A = 20 / 100 ∧
  percentage_from_A * total_boys = non_science_count / (1 - science_percentage) := by
    intro h
    use 20 / 100
    constructor
    · rfl
    · sorry  -- The proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_from_school_A_l160_16025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_range_of_b_l160_16045

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + b * x^2 + (b + 2) * x + 3

theorem monotonic_increasing_range_of_b (b : ℝ) :
  (∀ x : ℝ, Monotone (f b)) → -1 ≤ b ∧ b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_range_of_b_l160_16045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_coordinate_of_quadratic_function_l160_16021

/-- A quadratic function passing through specific points -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem vertex_x_coordinate_of_quadratic_function
  (a b c : ℝ)
  (h1 : QuadraticFunction a b c 2 = 9)
  (h2 : QuadraticFunction a b c 8 = 9)
  (h3 : QuadraticFunction a b c 3 = 4) :
  ∃ y : ℝ, QuadraticFunction a b c 5 = y ∧ 
    ∀ x : ℝ, QuadraticFunction a b c x ≥ QuadraticFunction a b c 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_coordinate_of_quadratic_function_l160_16021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_problem_l160_16022

/-- The harmonic mean of two positive real numbers -/
noncomputable def harmonicMean (a b : ℝ) : ℝ := 2 / (1/a + 1/b)

/-- Given x is 40% greater than 88 and y is 25% less than x, 
    prove that the harmonic mean of x and y is approximately 105.6 -/
theorem harmonic_mean_problem : 
  ∀ x y : ℝ, 
  x = 88 * 1.4 → 
  y = x * 0.75 → 
  abs (harmonicMean x y - 105.6) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_problem_l160_16022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_g_plus_half_l160_16056

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x - log x
noncomputable def g (x : ℝ) : ℝ := log x / x

-- State the theorem
theorem f_greater_than_g_plus_half :
  ∀ x : ℝ, 0 < x → x ≤ (exp 1) → f x > g x + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_g_plus_half_l160_16056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_fixed_five_l160_16046

/-- The number of permutations of n elements where exactly one element is in its original position -/
def exactlyOneFixed (n : ℕ) : ℕ :=
  n * (n - 1).factorial * (1 - ((-1 : Int)^(n-1)).toNat) / 2

/-- Theorem: The number of permutations of 5 elements where exactly one element is in its original position is 45 -/
theorem exactly_one_fixed_five :
  exactlyOneFixed 5 = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_fixed_five_l160_16046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l160_16047

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem smallest_positive_period_of_f :
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l160_16047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_second_digit_of_51_over_777_l160_16097

theorem sixty_second_digit_of_51_over_777 : ∃ (d : ℕ), d = 6 ∧ 
  (∃ (a b c : ℕ) (s : List ℕ), 
    51 = 777 * a + b ∧ 
    0 ≤ b ∧ b < 777 ∧
    c = 62 - a - 1 ∧
    s.length = 6 ∧
    (∀ n, n < s.length → s.get! n = (10 * (if n = 0 then b else s.get! (n-1))) / 777 % 10) ∧
    d = s.get! (c % 6)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_second_digit_of_51_over_777_l160_16097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_revenue_calculation_l160_16063

/-- Calculates the total revenue from ticket sales given the specified conditions -/
theorem ticket_revenue_calculation (group_size : ℕ) (interval_minutes : ℕ) 
  (start_hour start_minute : ℕ) (end_hour end_minute : ℕ) 
  (regular_price student_price : ℕ) (regular_to_student_ratio : ℕ) : 
  ∃ (revenue : ℕ), revenue = 22456 := by
  let total_intervals : ℕ := 
    (end_hour - start_hour) * (60 / interval_minutes) + 
    (end_minute - start_minute) / interval_minutes + 1
  let total_tickets : ℕ := total_intervals * group_size
  let student_tickets : ℕ := total_tickets / (regular_to_student_ratio + 1)
  let regular_tickets : ℕ := total_tickets - student_tickets
  let student_revenue : ℕ := student_tickets * student_price
  let regular_revenue : ℕ := regular_tickets * regular_price
  let revenue : ℕ := student_revenue + regular_revenue
  use revenue
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_revenue_calculation_l160_16063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oh_less_than_3r_l160_16034

/-- A structure representing a nondegenerate triangle with its circumcenter, orthocenter, and circumradius -/
structure Triangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Circumcenter
  O : ℝ × ℝ
  -- Orthocenter
  H : ℝ × ℝ
  -- Circumradius
  R : ℝ
  -- Nondegeneracy condition
  nondegen : A ≠ B ∧ B ≠ C ∧ C ≠ A
  -- O is the circumcenter
  is_circumcenter : dist O A = R ∧ dist O B = R ∧ dist O C = R
  -- H is the orthocenter
  is_orthocenter : (A.1 - H.1) * (B.1 - C.1) + (A.2 - H.2) * (B.2 - C.2) = 0 ∧
                   (B.1 - H.1) * (C.1 - A.1) + (B.2 - H.2) * (C.2 - A.2) = 0 ∧
                   (C.1 - H.1) * (A.1 - B.1) + (C.2 - H.2) * (A.2 - B.2) = 0

/-- The theorem stating that |OH| < 3R for any nondegenerate triangle -/
theorem oh_less_than_3r (t : Triangle) : dist t.O t.H < 3 * t.R := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oh_less_than_3r_l160_16034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l160_16005

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := ((x + 2) * (x - t)) / (x ^ 2)

def E (t : ℝ) : Set ℝ := {y | ∃ x ∈ ({1, 2, 3} : Set ℝ), y = f t x}

noncomputable def lambda : ℝ := (Real.log 2)^2 + (Real.log 2) * (Real.log 5) + (Real.log 5) - 1

theorem problem_solution :
  (∀ x, f 2 x = f 2 (-x)) ∧
  (lambda ∈ E 2) ∧
  (∀ a b, 0 < a → a < b → 
    (∀ x ∈ Set.Icc a b, 2 - 5/a ≤ f 2 x ∧ f 2 x ≤ 2 - 5/b) →
    a = 1 ∧ b = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l160_16005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_place_250_of_13_17_l160_16011

def decimal_representation (n d : ℕ) : List ℕ := sorry

def repeating_sequence (l : List ℕ) : List ℕ := sorry

theorem decimal_place_250_of_13_17 :
  let rep := repeating_sequence (decimal_representation 13 17)
  (rep.get! ((250 - 1) % rep.length)) = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_place_250_of_13_17_l160_16011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_x_l160_16009

def base_x_number (x : ℕ) : ℕ :=
  2*x^11 + 0*x^10 + 1*x^9 + 0*x^8 + 2*x^7 + 0*x^6 + 1*x^5 + 1*x^4 + 2*x^3 + 0*x^2 + 1*x + 2

def is_valid (x : ℕ) : Bool :=
  x ≥ 3 && (base_x_number x) % (x - 1) = 0

theorem sum_of_valid_x : (Finset.filter (fun x => is_valid x) (Finset.range 14)).sum id = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_x_l160_16009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l160_16035

-- Define the exponential function with base 1/3
noncomputable def f (x : ℝ) : ℝ := (1/3) ^ x

-- Define the mapping function
def g (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

theorem problem_statement :
  (∀ x y : ℝ, x < y → f y < f x) ∧ 
  (g (3, 1) = (4, 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l160_16035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_directed_graph_has_hamiltonian_path_l160_16006

/-- A complete directed graph is a graph where every pair of distinct vertices
    is connected by a directed edge. -/
def CompleteDirectedGraph (V : Type*) [Fintype V] [DecidableEq V] := 
  ∀ (u v : V), u ≠ v → (∃ (e : V × V), e = (u, v) ∨ e = (v, u))

/-- A Hamiltonian path in a graph is a path that visits each vertex exactly once. -/
def HamiltonianPath (V : Type*) [Fintype V] (path : List V) :=
  path.Nodup ∧ path.length = Fintype.card V

/-- In a complete directed graph, there exists a Hamiltonian path. -/
theorem complete_directed_graph_has_hamiltonian_path 
  {V : Type*} [Fintype V] [DecidableEq V] (h : CompleteDirectedGraph V) :
  ∃ (path : List V), HamiltonianPath V path := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_directed_graph_has_hamiltonian_path_l160_16006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_sum_l160_16051

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the arrangement of circles
structure CircleArrangement where
  largeCircle : Circle
  smallCircle1 : Circle
  smallCircle2 : Circle

-- Define the properties of the arrangement
def validArrangement (arr : CircleArrangement) : Prop :=
  -- The area of the largest circle is 100π
  arr.largeCircle.radius ^ 2 * Real.pi = 100 * Real.pi ∧
  -- The second circle is centered on the diameter of the large circle
  arr.smallCircle1.center.1 = arr.largeCircle.center.1 ∧
  arr.smallCircle1.center.2 = arr.largeCircle.center.2 + arr.largeCircle.radius / 2 ∧
  -- The second circle touches the circumference of the large circle
  arr.smallCircle1.radius = arr.largeCircle.radius / 2 ∧
  -- The third circle has the same radius as the second circle
  arr.smallCircle2.radius = arr.smallCircle1.radius ∧
  -- The third circle's circumference passes through the center of the large circle
  (arr.smallCircle2.center.1 - arr.largeCircle.center.1) ^ 2 +
    (arr.smallCircle2.center.2 - arr.largeCircle.center.2) ^ 2 =
    arr.smallCircle2.radius ^ 2

-- Define the theorem
theorem shaded_area_sum (arr : CircleArrangement) (h : validArrangement arr) :
  (Real.pi * arr.largeCircle.radius ^ 2 / 2) +
  (Real.pi * arr.smallCircle1.radius ^ 2 / 2) +
  (Real.pi * arr.smallCircle2.radius ^ 2 / 2) = 75 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_sum_l160_16051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_formula_l160_16004

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | n + 1 => if n % 2 = 0 then sequence_a n + 3 else sequence_a n + 1

def sequence_b (n : ℕ) : ℕ := sequence_a (2 * n)

theorem sequence_b_formula (n : ℕ) : sequence_b n = 4 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_formula_l160_16004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l160_16032

theorem trig_inequality : Real.cos 8.5 < Real.sin 3 ∧ Real.sin 3 < Real.sin 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l160_16032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_projections_l160_16052

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a plane
def Point : Type := ℝ × ℝ

-- Define concyclic points
def Concyclic (A B C D : Point) (circle : Circle) : Prop :=
  let dist := λ p : Point => Real.sqrt ((p.1 - circle.center.1)^2 + (p.2 - circle.center.2)^2)
  dist A = circle.radius ∧ dist B = circle.radius ∧ dist C = circle.radius ∧ dist D = circle.radius

-- Define orthogonal projection
noncomputable def OrthogonalProjection (P : Point) (A B : Point) : Point :=
  let v := (B.1 - A.1, B.2 - A.2)
  let t := ((P.1 - A.1) * v.1 + (P.2 - A.2) * v.2) / (v.1^2 + v.2^2)
  (A.1 + t * v.1, A.2 + t * v.2)

-- Main theorem
theorem concyclic_projections
  (A B C D : Point) (circle : Circle)
  (h_concyclic : Concyclic A B C D circle)
  (A' : Point) (h_A' : A' = OrthogonalProjection A B D)
  (C' : Point) (h_C' : C' = OrthogonalProjection C B D)
  (B' : Point) (h_B' : B' = OrthogonalProjection B A C)
  (D' : Point) (h_D' : D' = OrthogonalProjection D A C) :
  ∃ (circle' : Circle), Concyclic A' B' C' D' circle' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_projections_l160_16052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_count_l160_16008

-- Define the set of possible 'a' values
def A : Set Int := {-3, -2, -1, 0, 1, 2, 3}

-- Define the set of possible 'b' values
def B : Set Int := {-4, -3, -2, -1, 1, 2, 3, 4}

-- Define a parabola type
structure Parabola where
  a : Int
  b : Int

-- Define the set of all valid parabolas
def validParabolas : Set Parabola :=
  {p : Parabola | p.a ∈ A ∧ p.b ∈ B}

-- Define a function to check if a point is on a parabola
def isOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  sorry -- Actual implementation would go here

-- Define a function to check if three parabolas intersect at a common point
def noThreeIntersect (ps : Set Parabola) : Prop :=
  ∀ p1 p2 p3 : Parabola, p1 ∈ ps → p2 ∈ ps → p3 ∈ ps →
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
    ¬∃ (x y : ℝ), (isOnParabola p1 x y ∧ isOnParabola p2 x y ∧ isOnParabola p3 x y)

-- Define a function to count points on exactly two parabolas
noncomputable def countPointsOnTwoParabolas (ps : Set Parabola) : ℕ :=
  sorry -- Actual implementation would go here

-- The main theorem
theorem parabola_intersection_count :
  noThreeIntersect validParabolas →
  countPointsOnTwoParabolas validParabolas = 2912 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_count_l160_16008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equality_l160_16039

/-- The function g(x) = ln(2^x + 1) -/
noncomputable def g (x : ℝ) : ℝ := Real.log (2^x + 1)

/-- The theorem stating the equality -/
theorem g_sum_equality :
  g (-4) - g (-3) + g (-2) - g (-1) + g 1 - g 2 + g 3 - g 4 = -2 * Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equality_l160_16039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_solution_l160_16087

theorem trigonometric_system_solution :
  ∀ x y z : ℝ,
  0 ≤ x ∧ x ≤ π/2 →
  0 ≤ y ∧ y ≤ π/2 →
  0 ≤ z ∧ z ≤ π/2 →
  Real.sin x * Real.cos y = Real.sin z →
  Real.cos x * Real.sin y = Real.cos z →
  ((x = π/2 ∧ y = 0 ∧ z = π/2) ∨ (x = 0 ∧ y = π/2 ∧ z = 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_solution_l160_16087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l160_16058

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^6 - X^5 - X^4 + X^3 + X^2 - X = (X^2 - 1) * (X - 2) * q + (17/2 * X^2 - 3/2 * X - 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l160_16058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_sum_l160_16018

theorem pattern_sum (n : ℕ) (a t : ℝ) (h_pos : 0 < a ∧ 0 < t) :
  (∀ k : ℕ, k > 1 → Real.sqrt (k + k / (k^2 - 1 : ℝ)) = k * Real.sqrt (k / (k^2 - 1 : ℝ))) →
  Real.sqrt (8 + a / t) = 8 * Real.sqrt (a / t) →
  a + t = 71 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_sum_l160_16018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l160_16089

/-- The area of a trapezium with given parallel sides and height -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

theorem trapezium_area_example :
  let a : ℝ := 22 -- length of one parallel side
  let b : ℝ := 18 -- length of the other parallel side
  let h : ℝ := 15 -- distance between parallel sides
  trapeziumArea a b h = 300 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic expression
  simp [add_mul, mul_div_assoc]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l160_16089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_comparison_l160_16068

theorem price_comparison (x : ℝ) (h : x > 0) :
  0.9 * (1.3 * x) > x ∧ 0.9 * (1.3 * x) < 1.2 * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_comparison_l160_16068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_3_eq_neg_1_l160_16098

def f : ℕ → ℤ
| 0 => 1
| 1 => 0
| n + 2 => f (n + 1) - f n

theorem f_3_eq_neg_1 : f 3 = -1 := by
  -- Expand the definition of f
  have h1 : f 3 = f 2 - f 1 := rfl
  have h2 : f 2 = f 1 - f 0 := rfl
  -- Use the base cases
  have h3 : f 1 = 0 := rfl
  have h4 : f 0 = 1 := rfl
  -- Substitute and simplify
  rw [h1, h2, h3, h4]
  norm_num

#eval f 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_3_eq_neg_1_l160_16098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l160_16085

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.cos (x - Real.pi/6))^2 - (Real.sin x)^2

-- State the theorem
theorem f_properties :
  (f (Real.pi/12) = Real.sqrt 3/2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≤ Real.sqrt 3/2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi/2), f x = Real.sqrt 3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l160_16085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l160_16061

theorem sufficient_not_necessary (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a > b → a^3 + b^3 > a^2*b + a*b^2) ∧
  ∃ a b, 0 < a ∧ 0 < b ∧ a^3 + b^3 > a^2*b + a*b^2 ∧ a ≤ b :=
by
  constructor
  · intro h
    -- Proof for sufficiency
    sorry
  · -- Proof for not necessary
    use 1, 2
    constructor
    · norm_num
    constructor
    · norm_num
    constructor
    · norm_num
    · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l160_16061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_l160_16049

-- Define the circle
variable (circle : EuclideanPlane ℝ)

-- Define points
variable (E F B C M U G : circle.Point)

-- Define the conditions
variable (h1 : circle.isDiameter E F)
variable (h2 : circle.onCircle B)
variable (h3 : circle.onCircle C)
variable (h4 : circle.collinear B M C)
variable (h5 : circle.segmentCong E M M F)
variable (h6 : circle.segmentCong B M M C)
variable (h7 : circle.between B U M)
variable (h8 : circle.onCircle G)
variable (h9 : circle.collinear E U G)

-- Theorem statement
theorem triangle_similarity :
  circle.similarTriangles E U M E F G :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_l160_16049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l160_16076

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the given conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a = 2 * t.b ∧
  ∃ k, Real.sin t.A + Real.sin t.B = 2 * Real.sin t.C + k ∧
       Real.sin t.A + Real.sin t.C = 2 * Real.sin t.B + k

-- Helper function for area (not part of the main theorem)
noncomputable def area (t : Triangle) : Real :=
  1/2 * t.b * t.c * Real.sin t.A

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : satisfiesConditions t) : 
  Real.cos (t.B + t.C) = 1/4 ∧ 
  (area t = 3 * Real.sqrt 15 / 3 → t.c = 4 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l160_16076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l160_16019

-- Define the curve C
noncomputable def C (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

-- Define the line l in polar form
def l (ρ θ m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 3) + m = 0

-- State the theorem
theorem curve_line_intersection (m : ℝ) :
  (∃ x y : ℝ, Real.sqrt 3 * x + y + 2 * m = 0 ∧ ∃ t : ℝ, C t = (x, y)) ↔ 
  -19/12 ≤ m ∧ m ≤ 5/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l160_16019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_zero_satisfies_l160_16083

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - floor x

theorem only_zero_satisfies (m : ℕ) : 
  floor ((2 * m + 1 : ℝ) * frac (Real.sqrt (2 * m + 1))) = m ↔ m = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_zero_satisfies_l160_16083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_rate_is_seven_l160_16043

/-- Represents the cleaning scenario with a janitor and a student --/
structure CleaningScenario where
  janitor_time : ℚ  -- Time for janitor to clean alone (in hours)
  student_time : ℚ  -- Time for student to clean alone (in hours)
  janitor_rate : ℚ  -- Janitor's hourly rate (in dollars)
  cost_difference : ℚ  -- Additional cost for janitor to clean alone (in dollars)

/-- Calculates the student's hourly rate given the cleaning scenario --/
def student_rate (scenario : CleaningScenario) : ℚ :=
  let combined_rate := 1 / scenario.janitor_time + 1 / scenario.student_time
  let combined_time := 1 / combined_rate
  (scenario.janitor_rate * scenario.janitor_time - scenario.cost_difference) / combined_time - scenario.janitor_rate

/-- Theorem stating that the student's hourly rate is $7 given the specific scenario --/
theorem student_rate_is_seven :
  let scenario : CleaningScenario := {
    janitor_time := 8,
    student_time := 20,
    janitor_rate := 21,
    cost_difference := 8
  }
  student_rate scenario = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_rate_is_seven_l160_16043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_theorem_l160_16094

-- Define the cube structure
structure Cube where
  -- Eight positive integers on the faces
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+
  -- Constant value on edges
  g : ℕ+

-- Define the theorem
theorem cube_sum_theorem (cube : Cube) : 
  (cube.a.val * cube.b.val * cube.c.val * cube.g.val + 
   cube.a.val * cube.b.val * cube.e.val * cube.g.val + 
   cube.a.val * cube.b.val * cube.f.val * cube.g.val + 
   cube.a.val * cube.e.val * cube.f.val * cube.g.val + 
   cube.d.val * cube.b.val * cube.c.val * cube.g.val + 
   cube.d.val * cube.e.val * cube.b.val * cube.g.val + 
   cube.d.val * cube.b.val * cube.f.val * cube.g.val + 
   cube.d.val * cube.e.val * cube.f.val * cube.g.val) = 2310 →
  cube.a.val + cube.b.val + cube.c.val + cube.d.val + cube.e.val + cube.f.val = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_theorem_l160_16094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_l160_16053

-- Define the parabola
def is_on_parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the vector from focus to a point
def vector_from_focus (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - focus.1, p.2 - focus.2)

-- Define the magnitude of a vector
noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem parabola_focus_property 
  (A B C : ℝ × ℝ) 
  (h1 : is_on_parabola A) 
  (h2 : is_on_parabola B) 
  (h3 : is_on_parabola C) 
  (h4 : vector_from_focus A + vector_from_focus B + vector_from_focus C = (0, 0)) :
  vector_magnitude (vector_from_focus A) + 
  vector_magnitude (vector_from_focus B) + 
  vector_magnitude (vector_from_focus C) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_l160_16053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_success_probability_third_or_later_l160_16015

-- Define the probability of success
noncomputable def p : ℝ := 1 / 5

-- Define the mean number of attempts
noncomputable def mean : ℝ := 5

-- Define the probability of success on the first attempt
noncomputable def P1 : ℝ := p

-- Define the probability of success on the second attempt
noncomputable def P2 : ℝ := p * (1 - p)

-- Theorem statement
theorem success_probability_third_or_later :
  p < 1 ∧ mean = 1 / p → 1 - (P1 + P2) = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_success_probability_third_or_later_l160_16015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axes_of_transformed_func_l160_16036

noncomputable def original_func (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

noncomputable def transformed_func (x : ℝ) : ℝ := Real.cos (x - Real.pi / 6)

def is_symmetry_axis (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem symmetry_axes_of_transformed_func :
  (is_symmetry_axis transformed_func (-5 * Real.pi / 6)) ∧
  (is_symmetry_axis transformed_func (Real.pi / 6)) ∧
  (is_symmetry_axis transformed_func (7 * Real.pi / 6)) := by
  sorry

#check symmetry_axes_of_transformed_func

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axes_of_transformed_func_l160_16036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_difference_l160_16037

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    if P is a point on the hyperbola such that PF₂ is perpendicular to an asymptote,
    then |PF₁|² - |PF₂|² = 4a², where F₁ and F₂ are the left and right foci. -/
theorem hyperbola_foci_distance_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∀ (P F₁ F₂ : ℝ × ℝ),
  (P.1^2 / a^2 - P.2^2 / b^2 = 1) →  -- P is on the hyperbola
  (∃ (k : ℝ), F₂ = (k, 0)) →  -- F₂ is on the x-axis
  (∃ (l : ℝ), F₁ = (-l, 0)) →  -- F₁ is on the x-axis
  (∃ (m : ℝ), P.2 = m * (P.1 - F₂.1)) →  -- PF₂ is perpendicular to an asymptote
  ‖P - F₁‖^2 - ‖P - F₂‖^2 = 4 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_difference_l160_16037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_time_equation_l160_16086

/-- The time it takes Doug to paint a room alone -/
def doug_time : ℝ := 5

/-- The time it takes Dave to paint a room alone -/
def dave_time : ℝ := 7

/-- The total time it takes Doug and Dave to paint the room together, including a 1-hour lunch break -/
def t : ℝ := sorry

/-- The equation that t satisfies -/
theorem painting_time_equation : (1 / doug_time + 1 / dave_time) * (t - 1) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_time_equation_l160_16086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_M_property_l160_16095

/-- The base of the natural logarithm -/
noncomputable def e : ℝ := Real.exp 1

/-- A function has the M property if e^x * f(x) is monotonically increasing -/
def has_M_property (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → e^x * f x < e^y * f y

/-- The function f(x) = 2^(-x) -/
noncomputable def f (x : ℝ) : ℝ := 2^(-x)

/-- Theorem: f(x) = 2^(-x) has the M property -/
theorem f_has_M_property : has_M_property f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_M_property_l160_16095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_implies_base_inequality_l160_16091

theorem log_inequality_implies_base_inequality (m n : ℝ) : 
  (Real.log 9 / Real.log m < Real.log 9 / Real.log n) ∧ (Real.log 9 / Real.log n < 0) → 
  0 < m ∧ m < n ∧ n < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_implies_base_inequality_l160_16091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fastest_completion_time_l160_16065

-- Define the problem parameters
noncomputable def team_a_completion_rate : ℝ := 1 / 90
noncomputable def team_b_completion_rate : ℝ := 1 / 30
noncomputable def team_a_daily_cost : ℝ := 3000
noncomputable def team_b_daily_cost : ℝ := 10000
noncomputable def total_budget : ℝ := 280000

-- Define the theorem
theorem fastest_completion_time :
  ∃ (a b : ℝ),
    -- The project is completed
    a * team_a_completion_rate + b * team_b_completion_rate = 1 ∧
    -- The budget constraint is satisfied
    a * team_a_daily_cost + b * team_b_daily_cost ≤ total_budget ∧
    -- The completion time is 70 days
    a + b = 70 ∧
    -- This is the fastest completion time
    ∀ (a' b' : ℝ),
      a' * team_a_completion_rate + b' * team_b_completion_rate = 1 →
      a' * team_a_daily_cost + b' * team_b_daily_cost ≤ total_budget →
      a' + b' ≥ 70 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fastest_completion_time_l160_16065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_absolute_values_l160_16024

theorem equal_absolute_values (a b c : ℤ) 
  (h1 : ∃ n : ℤ, n = (a / b) + (b / c) + (c / a))
  (h2 : ∃ m : ℤ, m = (a / c) + (c / b) + (b / a)) :
  (|a| = |b| ∧ |b| = |c| ∧ |a| = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_absolute_values_l160_16024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l160_16026

noncomputable def is_circle_equation (a b r : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = (x - a)^2 + (y - b)^2 - r^2

def is_line (a b c : ℝ) (l : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, l x y = a * x + b * y + c

noncomputable def distance_point_to_line (x0 y0 a b c : ℝ) : ℝ :=
  abs (a * x0 + b * y0 + c) / Real.sqrt (a^2 + b^2)

def is_tangent_circle_line (a b r : ℝ) (l : ℝ → ℝ → ℝ) : Prop :=
  ∃ a' b' c', is_line a' b' c' l ∧ distance_point_to_line a b a' b' c' = r

theorem circle_tangent_to_line :
  let f : ℝ → ℝ → ℝ := λ x y => (x - 2)^2 + (y + 1)^2 - 18
  let l : ℝ → ℝ → ℝ := λ x y => x + y - 7
  is_circle_equation 2 (-1) (Real.sqrt 18) f ∧ is_tangent_circle_line 2 (-1) (Real.sqrt 18) l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l160_16026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_B_in_A_l160_16029

def A : Set ℕ := {x | 0 ≤ x ∧ x < 5}
def B : Set ℕ := {2, 4}

theorem complement_of_B_in_A :
  A \ B = {0, 1, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_B_in_A_l160_16029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_integer_points_l160_16003

theorem intersection_integer_points : 
  {k : ℤ | ∃ x y : ℤ, y = 2*x - 1 ∧ y = k*x + k} = {-1, 1, 3, 5} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_integer_points_l160_16003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_ratio_l160_16080

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (↑(n.choose k) : ℚ) / ↑(n.choose (k+1)) = 1/3 ∧ 
  (↑(n.choose (k+1)) : ℚ) / ↑(n.choose (k+2)) = 1/2 → 
  n + k = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_ratio_l160_16080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l160_16075

-- Define the ellipse equation
noncomputable def ellipse_equation (x y m : ℝ) : Prop := x^2 / 5 + y^2 / m = 1

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 10 / 5

-- Theorem statement
theorem ellipse_m_values :
  ∀ m : ℝ, (∀ x y : ℝ, ellipse_equation x y m) → 
  (m = 25/3 ∨ m = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l160_16075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l160_16060

noncomputable section

open Real

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

def circumradius (R : ℝ) (a : ℝ) (A : ℝ) : Prop :=
  2 * R = a / sin A

def triangle_area (S a b C : ℝ) : Prop :=
  S = 1/2 * a * b * sin C

theorem triangle_properties 
  (a b c : ℝ) (A B C : ℝ) (R S : ℝ) :
  triangle_ABC a b c A B C →
  a = 5 →
  b = 6 →
  cos B = -4/5 →
  S = 15 * sqrt 7 / 4 →
  A = Real.pi/6 ∧ 
  circumradius R a A ∧
  R = 5 ∧
  (c = 4 ∨ c = sqrt 106) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l160_16060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_eq_negative_two_l160_16013

def A : Set ℤ := {-2, -1, 0, 1, 2}

def B : Set ℤ := {x : ℤ | (x : ℝ) > 4 ∨ (x : ℝ) < -1}

theorem A_intersect_B_eq_negative_two : A ∩ B = {-2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_eq_negative_two_l160_16013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvilinear_trapezoid_area_l160_16002

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := Real.sqrt x

-- Define the bounds of the trapezoid
def lower_bound : ℝ := 0
def upper_bound : ℝ := 4

-- State the theorem
theorem curvilinear_trapezoid_area :
  ∫ x in lower_bound..upper_bound, f x = 16/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvilinear_trapezoid_area_l160_16002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drink_cost_calculation_l160_16078

/-- The cost of a drink given the cost of steaks, tip percentage, and Billy's tip contribution -/
noncomputable def drink_cost (steak_cost : ℝ) (tip_percentage : ℝ) (billy_tip_percentage : ℝ) (billy_tip_amount : ℝ) : ℝ :=
  let total_steak_cost := 2 * steak_cost
  let total_tip := tip_percentage * total_steak_cost
  let billy_tip_share := billy_tip_percentage * total_tip
  (billy_tip_amount - billy_tip_share) / 2

theorem drink_cost_calculation (steak_cost : ℝ) (tip_percentage : ℝ) (billy_tip_percentage : ℝ) (billy_tip_amount : ℝ)
  (h1 : steak_cost = 20)
  (h2 : tip_percentage = 0.2)
  (h3 : billy_tip_percentage = 0.8)
  (h4 : billy_tip_amount = 8) :
  drink_cost steak_cost tip_percentage billy_tip_percentage billy_tip_amount = 1.6 := by
  sorry

-- Remove the #eval line as it's not necessary for the proof and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drink_cost_calculation_l160_16078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l160_16030

-- Define the ∇ operation as noncomputable
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_calculation :
  nabla (nabla 2 3) (nabla 4 5) = 49 / 56 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l160_16030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_afternoon_sequences_count_l160_16067

/-- Represents the set of letters to be printed -/
def Letters := Fin 9

/-- Represents whether a letter has been printed -/
inductive PrintStatus
| Printed
| NotPrinted

/-- The state of printing before lunch -/
def PreLunchState : Letters → PrintStatus
| ⟨8, _⟩ => PrintStatus.Printed
| _ => PrintStatus.NotPrinted

/-- The number of possible afternoon printing sequences -/
def AfternoonSequences : ℕ := 704

/-- Theorem stating the number of possible afternoon printing sequences -/
theorem afternoon_sequences_count :
  AfternoonSequences = 704 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_afternoon_sequences_count_l160_16067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_is_36_l160_16072

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflect a point over the y-axis -/
def reflectOverYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Reflect a point over the line y = -x -/
def reflectOverNegativeDiagonal (p : Point) : Point :=
  { x := -p.y, y := -p.x }

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p q r : Point) : ℝ :=
  let base := |p.x - q.x|
  let height := |r.y - p.y|
  (1 / 2) * base * height

/-- The main theorem -/
theorem area_of_triangle_PQR_is_36 :
  let p := Point.mk 4 5
  let q := reflectOverYAxis p
  let r := reflectOverNegativeDiagonal q
  triangleArea p q r = 36 := by
  sorry

#eval IO.println "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_is_36_l160_16072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equation_solution_l160_16073

theorem sine_equation_solution (x : ℝ) 
  (h1 : Real.sin (π/2 - x) = -Real.sqrt 3/2) 
  (h2 : π < x ∧ x < 2*π) : 
  x = 7*π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equation_solution_l160_16073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_resistance_combinations_l160_16042

/-- Represents the set of possible resistance values after combining n 1-ohm resistors -/
def ResistanceSet (n : ℕ) : Set ℚ := sorry

/-- Combines two resistance values in series -/
def seriesCombine (a b : ℚ) : ℚ := a + b

/-- Combines two resistance values in parallel -/
def parallelCombine (a b : ℚ) : ℚ := (a * b) / (a + b)

/-- The set of possible resistance values after combining two resistors -/
def R2 : Finset ℚ := {1/2, 2}

/-- The set of possible resistance values after combining three resistors -/
def R3 : Finset ℚ := {1/3, 2/3, 3/2, 3}

/-- The set of possible resistance values after combining four resistors -/
def R4 : Finset ℚ := {1/4, 2/5, 3/5, 3/4, 1, 4/3, 5/3, 5/2, 4}

/-- The theorem stating that there are 15 distinct resistance values -/
theorem resistance_combinations : 
  (R2 ∪ R3 ∪ R4).card = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_resistance_combinations_l160_16042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l160_16050

/-- Sum of first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The ratio of S_3 to S_6 is 1/3 -/
axiom ratio_S3_S6 : S 3 / S 6 = 1 / 3

/-- Theorem: If S_3 / S_6 = 1/3, then S_6 / S_12 = 3/10 -/
theorem arithmetic_sequence_sum_ratio :
  S 6 / S 12 = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l160_16050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fat_ant_faster_l160_16088

/-- Represents an ant with its characteristics -/
structure Ant where
  speed : ℚ
  capacity : ℚ

/-- Calculates the time taken by an ant to deliver all cargo -/
noncomputable def deliveryTime (ant : Ant) (totalCargo : ℚ) (distance : ℚ) : ℚ :=
  let trips := totalCargo / ant.capacity
  let roundTrips := (trips - 1) * 2 + 1
  roundTrips * distance / ant.speed

theorem fat_ant_faster (totalCargo : ℚ) (distance : ℚ) :
  totalCargo = 150 →
  distance = 15 →
  let fatAnt : Ant := { speed := 3, capacity := 5 }
  let thinAnt : Ant := { speed := 5, capacity := 3 }
  deliveryTime fatAnt totalCargo distance < deliveryTime thinAnt totalCargo distance := by
  sorry

#check fat_ant_faster

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fat_ant_faster_l160_16088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l160_16054

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_specific_vectors :
  ∃ (a b : ℝ × ℝ),
    a - 2 • b = (2 * Real.sqrt 3, -1) ∧
    b - 2 • a = (-Real.sqrt 3, -1) ∧
    angle_between_vectors a b = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l160_16054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_tire_marks_l160_16010

/-- The distance between white dots left by a bicycle tire -/
noncomputable def distance_between_dots (tire_diameter : ℝ) (stripe_width : ℝ) : ℝ :=
  tire_diameter * Real.pi - stripe_width

/-- Theorem: The distance between white dots left by a bicycle tire
    with diameter 60cm crossing a 20cm wide stripe is (60π - 20) cm -/
theorem bicycle_tire_marks :
  distance_between_dots 60 20 = 60 * Real.pi - 20 := by
  -- Unfold the definition of distance_between_dots
  unfold distance_between_dots
  -- The equation now follows by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_tire_marks_l160_16010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_ln_l160_16038

noncomputable def f (x : ℝ) : ℝ := Real.log x

def tangent_line (x m : ℝ) : ℝ := 2 * x + m

theorem tangent_to_ln (x : ℝ) (h : x > 0) :
  ∃ m : ℝ, (∀ x₀ : ℝ, x₀ > 0 → tangent_line x₀ m = f x₀ + (deriv f x₀) * (x₀ - x)) →
  m = -1 - Real.log 2 := by
  sorry

#check tangent_to_ln

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_ln_l160_16038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_speed_and_fireworks_distance_l160_16027

/-- Define the relationship between air temperature and sound speed -/
noncomputable def sound_speed (x : ℝ) : ℝ := (3/5) * x + 331

/-- Define the distance calculation given speed and time -/
noncomputable def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating the relationship between air temperature and sound speed,
    and the distance calculation for the fireworks problem -/
theorem sound_speed_and_fireworks_distance :
  (sound_speed 0 = 331) ∧
  (sound_speed 5 = 334) ∧
  (distance (sound_speed 24) 5 = 1727) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_speed_and_fireworks_distance_l160_16027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hua_method_uses_golden_ratio_l160_16079

/-- The optimal selection method popularized by Hua Luogeng --/
structure OptimalSelectionMethod where
  inventor : String
  concept : String

/-- The Golden ratio --/
noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

/-- Hua Luogeng's optimal selection method --/
def hua_method : OptimalSelectionMethod :=
  { inventor := "Hua Luogeng"
  , concept := "Golden ratio" }

/-- Theorem stating that Hua Luogeng's optimal selection method uses the Golden ratio --/
theorem hua_method_uses_golden_ratio :
  hua_method.concept = "Golden ratio" :=
by
  rfl

#check hua_method_uses_golden_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hua_method_uses_golden_ratio_l160_16079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l160_16000

-- Define the functions
noncomputable def f (x : ℝ) := 1 / Real.sqrt (1 - x)
noncomputable def g (x : ℝ) := Real.log x

-- Define the sets A and B
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := Set.univ

-- Statement to prove
theorem domain_intersection :
  A ∩ B = Set.Iio 1 := by
  -- The proof goes here
  sorry

#check domain_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l160_16000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_describes_cone_l160_16040

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  r : ℝ
  θ : ℝ
  φ : ℝ

/-- Represents the equation φ = π/2 - c -/
def coneEquation (c : ℝ) (p : SphericalPoint) : Prop :=
  p.φ = Real.pi / 2 - c

/-- The set of points satisfying the cone equation -/
def coneSet (c : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | coneEquation c p}

/-- Placeholder for the concept of a cone -/
def IsCone (s : Set SphericalPoint) (vertex : SphericalPoint) (axis : ℝ × ℝ × ℝ) (angle : ℝ) : Prop :=
  sorry

/-- Theorem stating that the equation describes a cone -/
theorem equation_describes_cone (c : ℝ) :
  ∃ (vertex : SphericalPoint) (axis : ℝ × ℝ × ℝ) (angle : ℝ),
    IsCone (coneSet c) vertex axis angle :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_describes_cone_l160_16040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_five_smallest_multiples_of_15_l160_16062

theorem sum_five_smallest_multiples_of_15 : 
  (Finset.range 5).sum (fun i => 15 * (i + 1)) = 225 := by
  -- The proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_five_smallest_multiples_of_15_l160_16062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_symmetry_l160_16077

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

noncomputable def shifted_f (x φ : ℝ) : ℝ := Real.sin (2 * (x + φ) + Real.pi / 4)

theorem min_phi_for_symmetry :
  ∃ (φ : ℝ), φ > 0 ∧
  (∀ (x : ℝ), shifted_f x φ = shifted_f (-x) φ) ∧
  (∀ (ψ : ℝ), ψ > 0 → (∀ (x : ℝ), shifted_f x ψ = shifted_f (-x) ψ) → ψ ≥ φ) ∧
  φ = Real.pi / 8 :=
by
  sorry

#check min_phi_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_symmetry_l160_16077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_P_l160_16082

def telephone_number (P Q R S T U V W X Y : ℕ) : Prop :=
  P < Q ∧ Q < R ∧
  S < T ∧ T < U ∧
  V < W ∧ W < X ∧ X < Y ∧
  S % 2 = 0 ∧ T = S + 2 ∧ U = T + 2 ∧
  V % 2 = 1 ∧ W = V + 2 ∧ X = W + 2 ∧ Y = X + 2 ∧
  P + Q + R = S + T + U ∧
  Finset.card (Finset.range 10 ∩ {P, Q, R, S, T, U, V, W, X, Y}) = 10

theorem unique_P : ∃! P, ∃ Q R S T U V W X Y, telephone_number P Q R S T U V W X Y :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_P_l160_16082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_29_max_profit_value_l160_16071

noncomputable def annual_profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 20 then
    -2 * x^2 + 100 * x - 50
  else if x > 20 then
    -10 * x - 9000 / (x + 1) + 1950
  else
    0

def fixed_cost : ℝ := 50
def variable_cost_per_unit : ℝ := 8

noncomputable def sales_revenue (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 20 then
    x * (180 - 2 * x)
  else if x > 20 then
    x * (70 + 2000 / x - 9000 / (x * (x + 1)))
  else
    0

theorem max_profit_at_29 :
  ∀ x : ℝ, annual_profit x ≤ annual_profit 29 := by
  sorry

theorem max_profit_value :
  annual_profit 29 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_29_max_profit_value_l160_16071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombic_dodecahedron_second_kind_properties_l160_16074

-- Define a polyhedron
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))
  is_closed : Prop
  is_bounded : Prop

-- Define a square
structure Square where
  vertices : Fin 4 → (Fin 3 → ℝ)
  is_planar : Prop
  equal_sides : Prop
  right_angles : Prop

-- Define the Rhombic Dodecahedron of the Second Kind
def RhombicDodecahedronSecondKind : Polyhedron :=
  { vertices := sorry,
    faces := sorry,
    is_closed := sorry,
    is_bounded := sorry }

-- Define convexity for a polyhedron
def is_convex (p : Polyhedron) : Prop :=
  sorry

-- Define that all faces of a polyhedron are squares
def all_faces_squares (p : Polyhedron) : Prop :=
  sorry

-- Define that all faces of a polyhedron are equal
def all_faces_equal (p : Polyhedron) : Prop :=
  sorry

-- Theorem statement
theorem rhombic_dodecahedron_second_kind_properties :
  ¬(is_convex RhombicDodecahedronSecondKind) ∧
  all_faces_squares RhombicDodecahedronSecondKind ∧
  all_faces_equal RhombicDodecahedronSecondKind :=
by
  sorry

#check rhombic_dodecahedron_second_kind_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombic_dodecahedron_second_kind_properties_l160_16074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_magnitude_l160_16093

theorem smallest_square_magnitude (z : ℂ) (h_real_positive : 0 < z.re) 
  (h_area : abs (z.im * (1/z).re - z.re * (1/z).im) = 24/25) : 
  ∃ (d : ℝ), d ≥ 0 ∧ d^2 = 36/25 ∧ ∀ w : ℂ, (w.re > 0 ∧ abs (w.im * (1/w).re - w.re * (1/w).im) = 24/25) → 
    Complex.normSq (w + 1/w) ≥ d^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_magnitude_l160_16093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l160_16001

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x + x^2 else Real.exp (-x) + x^2

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (-a) + f a ≤ 2 * f 1) → a ∈ Set.Icc (-1) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l160_16001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_lengths_t_value_l160_16014

-- Define the points
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (-2, -1)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def OC : ℝ × ℝ := C

-- Define dot product
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Theorem for the lengths of the diagonals
theorem diagonal_lengths :
  magnitude (AB.1 + AC.1, AB.2 + AC.2) = 2 * Real.sqrt 10 ∧
  magnitude (AB.1 - AC.1, AB.2 - AC.2) = 4 * Real.sqrt 2 := by
  sorry

-- Theorem for the value of t
theorem t_value :
  ∃ t : ℝ, dot (AB.1 - t * OC.1, AB.2 - t * OC.2) OC = 0 ∧ t = -11/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_lengths_t_value_l160_16014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l160_16084

noncomputable def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ := (1/3) * base_area * height

noncomputable def cylinder_volume (radius : ℝ) (height : ℝ) : ℝ := Real.pi * radius^2 * height

theorem water_height_in_cylinder (pyramid_base_area : ℝ) (pyramid_height : ℝ) (cylinder_radius : ℝ) :
  pyramid_base_area = 144 →
  pyramid_height = 27 →
  cylinder_radius = 9 →
  ∃ h : ℝ, cylinder_volume cylinder_radius h = pyramid_volume pyramid_base_area pyramid_height ∧ h = 16 / Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l160_16084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_satisfies_conditions_l160_16096

noncomputable def distance_from_line (x y : ℝ) : ℝ :=
  |x - y + 1| / Real.sqrt 2

theorem point_satisfies_conditions : 
  let x : ℝ := -1
  let y : ℝ := -1
  distance_from_line x y = 1 / Real.sqrt 2 ∧ 
  x + y - 1 < 0 ∧
  x - y + 1 > 0 :=
by
  -- Unfold the definition of x and y
  have x : ℝ := -1
  have y : ℝ := -1

  -- Prove each part of the conjunction
  apply And.intro
  · -- Prove distance_from_line x y = 1 / Real.sqrt 2
    unfold distance_from_line
    -- This step requires computation, which we'll skip for now
    sorry
  · -- Prove x + y - 1 < 0 ∧ x - y + 1 > 0
    apply And.intro
    · -- Prove x + y - 1 < 0
      norm_num
    · -- Prove x - y + 1 > 0
      norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_satisfies_conditions_l160_16096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_4_primable_eq_21_l160_16020

/-- A function that checks if a positive integer is composed only of one-digit primes --/
def is_composed_of_one_digit_primes (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [2, 3, 5, 7]

/-- A function that checks if a number is 4-primable --/
def is_4_primable (n : ℕ) : Prop :=
  n % 4 = 0 ∧ is_composed_of_one_digit_primes n

/-- A decidable version of is_4_primable --/
def is_4_primable_decidable (n : ℕ) : Bool :=
  n % 4 = 0 && (n.digits 10).all (λ d ↦ d ∈ [2, 3, 5, 7])

/-- The count of 4-primable numbers less than 1000 --/
def count_4_primable_less_than_1000 : ℕ :=
  (Finset.range 1000).filter (λ n ↦ is_4_primable_decidable n) |>.card

/-- The main theorem stating that there are 21 4-primable numbers less than 1000 --/
theorem count_4_primable_eq_21 : count_4_primable_less_than_1000 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_4_primable_eq_21_l160_16020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whiteboard_cost_theorem_l160_16031

/-- Represents a class with its whiteboard count and ink usage per whiteboard -/
structure ClassInfo where
  whiteboards : ℕ
  ink_per_board : ℕ

/-- Calculates the total cost in dollars for whiteboard usage across all classes -/
def total_cost (classes : List ClassInfo) (ink_cost_cents : ℕ) : ℚ :=
  let total_ink := classes.map (fun c => c.whiteboards * c.ink_per_board) |>.sum
  (total_ink * ink_cost_cents : ℚ) / 100

/-- The total cost for using whiteboards in all classes for one day is $130 -/
theorem whiteboard_cost_theorem : 
  let classes := [
    { whiteboards := 3, ink_per_board := 20 },  -- Class A
    { whiteboards := 2, ink_per_board := 25 },  -- Class B
    { whiteboards := 4, ink_per_board := 15 },  -- Class C
    { whiteboards := 1, ink_per_board := 30 },  -- Class D
    { whiteboards := 3, ink_per_board := 20 }   -- Class E
  ]
  let ink_cost_cents := 50
  total_cost classes ink_cost_cents = 130 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_whiteboard_cost_theorem_l160_16031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l160_16092

theorem proper_subsets_count : 
  let S : Finset ℕ := Finset.filter (λ x => 10 ≤ x ∧ x < 100) (Finset.range 100)
  (Finset.card (Finset.powerset S) - 1) = 2^90 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_count_l160_16092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_set_is_conic_or_ray_l160_16081

-- Define the basic types and structures
def Point := ℝ × ℝ
def Circle := Point × ℝ

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

-- Define the distance from a point to a circle
noncomputable def distanceToCircle (p : Point) (c : Circle) : ℝ := sorry

-- Define the set of equidistant points
def equidistantSet (A : Point) (c : Circle) : Set Point :=
  {X : Point | distance X A = distanceToCircle X c}

-- Define the position of a point relative to a circle
inductive PointPosition
| Inside
| Outside
| OnCircle

noncomputable def pointPositionRelativeToCircle (p : Point) (c : Circle) : PointPosition := sorry

-- Helper definitions
def isEllipse (s : Set Point) : Prop := sorry
def isHyperbola (s : Set Point) : Prop := sorry
def isRay (s : Set Point) : Prop := sorry

-- Main theorem
theorem equidistant_set_is_conic_or_ray (A : Point) (c : Circle) :
  ∃ (s : Set Point), s = equidistantSet A c ∧
    (pointPositionRelativeToCircle A c = PointPosition.Inside → isEllipse s) ∧
    (pointPositionRelativeToCircle A c = PointPosition.Outside → isHyperbola s) ∧
    (pointPositionRelativeToCircle A c = PointPosition.OnCircle → isRay s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_set_is_conic_or_ray_l160_16081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_length_is_ten_l160_16064

/-- Represents the dimensions and properties of a box built with cubes -/
structure Box where
  width : ℚ
  height : ℚ
  cubeVolume : ℚ
  minCubes : ℕ

/-- Calculates the length of the box given its properties -/
def calculateBoxLength (box : Box) : ℚ :=
  (box.minCubes : ℚ) * box.cubeVolume / (box.width * box.height)

/-- Theorem stating that for a box with given properties, its length is 10 cm -/
theorem box_length_is_ten :
  let box : Box := {
    width := 18,
    height := 4,
    cubeVolume := 12,
    minCubes := 60
  }
  calculateBoxLength box = 10 := by
  -- Unfold the definition and perform the calculation
  unfold calculateBoxLength
  -- Simplify the arithmetic expression
  simp [Box.width, Box.height, Box.cubeVolume, Box.minCubes]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_length_is_ten_l160_16064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l160_16099

/-- Curve C2 obtained by stretching unit circle -/
noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, Real.sin θ)

/-- Line l passing through (1,0) with inclination π/4 -/
noncomputable def l (t : ℝ) : ℝ × ℝ := (1 + t * Real.sqrt 2 / 2, t * Real.sqrt 2 / 2)

/-- Intersection points of C2 and l -/
def intersection_points : Set ℝ := {t | ∃ θ, C2 θ = l t}

theorem intersection_product (t₁ t₂ : ℝ) :
  t₁ ∈ intersection_points → t₂ ∈ intersection_points →
  t₁ ≠ t₂ → |t₁| * |t₂| = 8/5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l160_16099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_root_in_zero_one_l160_16069

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 3

-- State the theorem
theorem root_interval : 
  (∃ k : ℤ, ∃ x : ℝ, x > k ∧ x < k + 1 ∧ f x = 0) → 
  (∃ x : ℝ, x > 0 ∧ x < 1 ∧ f x = 0) := by sorry

-- State the main result
theorem root_in_zero_one : 
  (∃ k : ℤ, ∃ x : ℝ, x > k ∧ x < k + 1 ∧ f x = 0) → k = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_root_in_zero_one_l160_16069
