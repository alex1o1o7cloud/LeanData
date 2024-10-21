import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l1176_117692

theorem sin_2x_value (x y : ℝ) 
  (h1 : Real.sin y = 2 * Real.cos x + (5/2) * Real.sin x)
  (h2 : Real.cos y = 2 * Real.sin x + (5/2) * Real.cos x) : 
  Real.sin (2*x) = -37/20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l1176_117692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_zero_interval_l1176_117600

/-- A quadratic function f(x) = ax^2 + bx + c has at least one zero in the interval (-2, 0),
    given that 2a + c/2 > b and 2^c < 1. -/
theorem quadratic_zero_interval (a b c : ℝ) (h1 : 2 * a + c / 2 > b) (h2 : (2 : ℝ)^c < 1) :
  ∃ x : ℝ, x ∈ Set.Ioo (-2 : ℝ) 0 ∧ a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_zero_interval_l1176_117600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_expansion_l1176_117693

/-- The coefficient of x in the expansion of (x - 2/x)^5 -/
def coefficient_of_x : ℤ := 40

/-- The binomial expression to be expanded -/
noncomputable def binomial_expression (x : ℝ) : ℝ := (x - 2/x)^5

theorem coefficient_of_x_in_expansion :
  coefficient_of_x = 40 := by
  -- The proof goes here
  sorry

#eval coefficient_of_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_in_expansion_l1176_117693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_plot_ratio_l1176_117687

/-- Represents a rectangular plot with given area and breadth -/
structure RectangularPlot where
  area : ℝ
  breadth : ℝ
  length_multiple_of_breadth : ℝ
  h1 : area = breadth * (length_multiple_of_breadth * breadth)
  h2 : breadth > 0

/-- The ratio of length to breadth for a rectangular plot with specific dimensions -/
def length_to_breadth_ratio (plot : RectangularPlot) : ℝ :=
  plot.length_multiple_of_breadth

/-- Theorem stating that for a rectangular plot with area 675 sq m and breadth 15 m,
    the ratio of length to breadth is 3:1 -/
theorem specific_plot_ratio :
  ∃ (plot : RectangularPlot),
    plot.area = 675 ∧
    plot.breadth = 15 ∧
    length_to_breadth_ratio plot = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_plot_ratio_l1176_117687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_permutations_l1176_117625

def banana_arrangements : ℕ := 60

theorem banana_permutations :
  banana_arrangements = 
    let total_letters : ℕ := 6
    let num_A : ℕ := 3
    let num_N : ℕ := 2
    let num_B : ℕ := 1
    Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N * Nat.factorial num_B)
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_permutations_l1176_117625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_danielle_popsicle_sticks_l1176_117685

/-- Calculates the number of popsicle sticks left after making popsicles --/
def popsicle_sticks_left (initial_budget : ℚ) (mold_cost : ℚ) (stick_pack_cost : ℚ) 
  (juice_bottle_cost : ℚ) (sticks_per_pack : ℕ) (popsicles_per_bottle : ℕ) : ℕ :=
  let remaining_budget := initial_budget - mold_cost - stick_pack_cost
  let juice_bottles := (remaining_budget / juice_bottle_cost).floor
  let popsicles_made := juice_bottles * popsicles_per_bottle
  (sticks_per_pack - popsicles_made).toNat

theorem danielle_popsicle_sticks : 
  popsicle_sticks_left 10 3 1 2 100 20 = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_danielle_popsicle_sticks_l1176_117685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_interval_subset_domain_l1176_117697

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (3 - 2*x - x^2) / Real.log (1/2)

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂ := by
  sorry

-- Define the domain of f(x)
def domain_f : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}

-- State that the interval (-1, 1) is a subset of the domain of f(x)
theorem interval_subset_domain : Set.Ioo (-1 : ℝ) 1 ⊆ domain_f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_interval_subset_domain_l1176_117697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_tangent_slope_range_l1176_117649

/-- Given a hyperbola x²/12 - y²/4 = 1, prove that if a line passing through its right focus
    intersects the right branch of the hyperbola at only one point, then the slope of this line
    is in the range [-√3/3, √3/3]. -/
theorem hyperbola_tangent_slope_range (x y : ℝ) (m : ℝ) :
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 12 - y^2 / 4 = 1
  let right_focus := (2 * Real.sqrt 3, 0)
  let line := fun (x : ℝ) ↦ m * (x - right_focus.1) + right_focus.2
  (∃! x₀, x₀ > right_focus.1 ∧ hyperbola x₀ (line x₀)) →
  m ∈ Set.Icc (-Real.sqrt 3 / 3) (Real.sqrt 3 / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_tangent_slope_range_l1176_117649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_sqrt_two_l1176_117605

noncomputable section

/-- Two lines are parallel if their slopes are equal and not undefined -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2 ∧ m1 ≠ 0 ∧ m2 ≠ 0

/-- The slope of the line x + 2ay = 2a + 2 -/
noncomputable def slope1 (a : ℝ) : ℝ := -1 / (2 * a)

/-- The slope of the line ax + 2y = a + 1 -/
noncomputable def slope2 (a : ℝ) : ℝ := -a / 2

/-- The y-intercept of the line x + 2ay = 2a + 2 -/
noncomputable def intercept1 (a : ℝ) : ℝ := (2 * a + 2) / (2 * a)

/-- The y-intercept of the line ax + 2y = a + 1 -/
noncomputable def intercept2 (a : ℝ) : ℝ := (a + 1) / 2

theorem parallel_lines_imply_a_sqrt_two :
  ∀ a : ℝ, parallel (slope1 a) (slope2 a) ∧ intercept1 a ≠ intercept2 a → a = Real.sqrt 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_sqrt_two_l1176_117605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l1176_117681

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y - 13 = 0
def C₂ (x y a : ℝ) : Prop := x^2 + y^2 - 2*a*x - 6*y + a^2 + 1 = 0
def l (x y m : ℝ) : Prop := (m+1)*x + y - 7*m - 7 = 0

-- State the theorem
theorem circle_tangency (a m : ℝ) :
  (a > 0) →
  (∃ x₁ y₁, C₁ x₁ y₁) →
  (∃ x₂ y₂, C₂ x₂ y₂ a) →
  (∀ x₁ y₁ x₂ y₂, C₁ x₁ y₁ → C₂ x₂ y₂ a → (x₁ - x₂)^2 + (y₁ - y₂)^2 = (3*Real.sqrt 2 + 2*Real.sqrt 2)^2) →
  (∃ x₃ y₃, C₂ x₃ y₃ a ∧ l x₃ y₃ m) →
  (m = 0 ∨ m = -8/7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l1176_117681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_locus_l1176_117699

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line passing through the center
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define an external point on the line
def ExternalPoint (l : Line) (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p = (l.point1.1 + t * (l.point2.1 - l.point1.1), 
               l.point1.2 + t * (l.point2.2 - l.point1.2))

-- Define the angle bisector
def AngleBisector (l : Line) (p : ℝ × ℝ) (c : Circle) : Set (ℝ × ℝ) :=
  sorry

-- Define the foot of perpendicular
def FootOfPerpendicular (c : Circle) (b : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

-- Define a set of points on a line
def LineSet (l : Line) : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p = (l.point1.1 + t * (l.point2.1 - l.point1.1), 
                     l.point1.2 + t * (l.point2.2 - l.point1.2))}

-- Define parallel sets
def Parallel (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  sorry

-- Define distance between a point and a set
def dist (p : ℝ × ℝ) (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

-- State the theorem
theorem ellipse_locus 
  (c : Circle) 
  (e : Line) 
  (h1 : e.point1 = c.center ∨ e.point2 = c.center) 
  (p : ℝ × ℝ) 
  (h2 : ExternalPoint e p) :
  ∃ (d1 d2 : Line),
    (∀ (b : Set (ℝ × ℝ)), 
      b = AngleBisector e p c → 
      (FootOfPerpendicular c b ∈ LineSet d1 ∨ FootOfPerpendicular c b ∈ LineSet d2)) ∧
    (Parallel (LineSet d1) (LineSet d2)) ∧
    (∃ (q1 q2 : ℝ × ℝ), 
      q1 ∈ LineSet d1 ∧ 
      q2 ∈ LineSet d2 ∧ 
      dist q1 (LineSet e) = c.radius / 2 ∧ 
      dist q2 (LineSet e) = c.radius / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_locus_l1176_117699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1176_117643

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin (x/2) + Real.cos (x/2))^2 - 2 * Real.sqrt 3 * (Real.cos (x/2))^2 + Real.sqrt 3

-- State the theorem
theorem f_range :
  ∀ x ∈ Set.Icc 0 Real.pi,
    ∃ y ∈ Set.Icc (1 - Real.sqrt 3) 3,
      f x = y ∧
      ∀ z, f x = z → z ∈ Set.Icc (1 - Real.sqrt 3) 3 :=
by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1176_117643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_conditions_l1176_117622

theorem divisibility_conditions (n : ℕ) : 
  (∀ x : ℤ, (x^2 + x + 1) ∣ ((x + 1)^n - x^n - 1)) = (n % 6 = 1 ∨ n % 6 = 5) ∧
  (∀ x : ℤ, (x^2 + x + 1) ∣ ((x + 1)^n + x^n + 1)) = (n % 6 = 2 ∨ n % 6 = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_conditions_l1176_117622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1176_117639

theorem sin_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1176_117639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_self_descriptive_number_l1176_117678

def is_valid_number (n : ℕ) : Prop :=
  ∃ (digits : List ℕ),
    digits.length = 10 ∧
    n = digits.foldl (λ acc d => acc * 10 + d) 0 ∧
    ∀ i ∈ Finset.range 10, digits.get? i = some ((digits.filter (λ d => d = i)).length)

theorem unique_self_descriptive_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 6210001000 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_self_descriptive_number_l1176_117678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_change_amount_l1176_117608

/-- The amount of change in cents -/
def change : ℕ := 0  -- Initialize with a default value

/-- Condition: The change is less than one dollar -/
axiom change_less_than_dollar : change < 100

/-- Condition: Using maximum quarters and the rest in nickels requires 2 additional nickels -/
axiom quarters_condition : change % 25 = 10

/-- Condition: Using maximum dimes and the rest in nickels requires 4 additional nickels -/
axiom dimes_condition : change % 10 = 20

/-- Theorem: The only amount of change satisfying all conditions is 60 cents -/
theorem unique_change_amount : change = 60 := by
  sorry  -- Proof is omitted for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_change_amount_l1176_117608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_intersection_and_distance_l1176_117689

/-- The equation of the line passing through the intersection of two given lines
    and having a specific distance from a given point. -/
theorem line_equation_from_intersection_and_distance
  (l₁ l₂ : Set (ℝ × ℝ)) -- Two lines given by their equations
  (P : ℝ × ℝ) -- A point
  (d : ℝ) -- Distance from the point to the line
  (h₁ : l₁ = {p : ℝ × ℝ | 7 * p.1 + 5 * p.2 - 24 = 0})
  (h₂ : l₂ = {p : ℝ × ℝ | p.1 - p.2 = 0})
  (h₃ : P = (5, 1))
  (h₄ : d = Real.sqrt 10)
  : ∃ (a b c : ℝ), {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0} = 
    {p : ℝ × ℝ | 3 * p.1 - p.2 - 4 = 0} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_intersection_and_distance_l1176_117689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_and_a_value_l1176_117628

-- Define the circles and points
def origin : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (12, 5)
def S : ℝ → ℝ × ℝ := λ a => (a, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_radii_and_a_value :
  ∃ (r1 r2 : ℝ),
    r1 > 0 ∧
    r2 > 0 ∧
    distance origin P = r1 ∧
    (∃ a, distance origin (S a) = r2) ∧
    r1 - r2 = 5 →
    ∃ a, S a = (8, 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_and_a_value_l1176_117628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_three_digit_numbers_l1176_117672

def valid_digits : Finset Nat := {0, 2, 3, 4, 6, 8}
def valid_nonzero_digits : Finset Nat := {2, 3, 4, 6, 8}

def is_valid_three_digit_number (n : Nat) : Bool :=
  n ≥ 100 && n < 1000 &&
  (n / 100) ∈ valid_nonzero_digits &&
  ((n / 10) % 10) ∈ valid_digits &&
  (n % 10) ∈ valid_digits

theorem count_valid_three_digit_numbers :
  (Finset.filter (fun n => is_valid_three_digit_number n) (Finset.range 1000)).card = 180 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_three_digit_numbers_l1176_117672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_product_bound_l1176_117610

theorem triangle_sine_product_bound (A B C : ℝ) 
  (h_triangle : A + B + C = Real.pi) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) : 
  Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) ≤ 1/8 ∧ 
  ∃ (A' B' C' : ℝ), A' + B' + C' = Real.pi ∧ 
    0 < A' ∧ 0 < B' ∧ 0 < C' ∧ 
    Real.sin (A'/2) * Real.sin (B'/2) * Real.sin (C'/2) = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_product_bound_l1176_117610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1176_117654

theorem inequality_proof (a b c l m : ℝ) (m n : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hl : 0 < l) (hm : 0 < m)
  (sum_one : a + b + c = 1) (hm2 : m ≥ 2) (hn : n ≥ 2) :
  (a^m + b^n) / (l*b + m*c) + (b^m + c^n) / (l*c + m*a) + (c^m + a^n) / (l*a + m*b) 
  ≥ (3^(2-m) + 3^(2-n)) / (l + m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1176_117654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_satisfying_conditions_l1176_117601

/-- The set of divisors of a positive integer n -/
def divisors (n : ℕ+) : Finset ℕ+ :=
  sorry

/-- The kth divisor of n, where divisors are sorted in ascending order -/
def kth_divisor (n : ℕ+) (k : ℕ) : ℕ+ :=
  sorry

theorem unique_n_satisfying_conditions : ∃! (n : ℕ+),
  (∃ (k : ℕ), k ≥ 22 ∧ (divisors n).card = k) ∧
  (kth_divisor n 7)^2 + (kth_divisor n 10)^2 = (n.val / (kth_divisor n 22).val)^2 ∧
  n = 2040 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_satisfying_conditions_l1176_117601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1176_117694

/-- The additional time needed to fill a cistern with a leak -/
noncomputable def additional_fill_time (normal_fill_time : ℝ) (empty_time : ℝ) : ℝ :=
  (1 / (1 / normal_fill_time - 1 / empty_time)) - normal_fill_time

theorem cistern_fill_time (normal_fill_time empty_time : ℝ) 
  (h1 : normal_fill_time = 12)
  (h2 : empty_time = 84) :
  additional_fill_time normal_fill_time empty_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1176_117694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_l1176_117652

-- Define the original line
def original_line (x y : ℝ) : Prop := 4 * x + 3 * y - 5 = 0

-- Define the two parallel lines
def parallel_line1 (x y : ℝ) : Prop := 4 * x + 3 * y + 10 = 0
def parallel_line2 (x y : ℝ) : Prop := 4 * x + 3 * y - 20 = 0

-- Define the distance between two lines
noncomputable def distance_between_lines (A B C1 C2 : ℝ) : ℝ := 
  abs (C2 - C1) / Real.sqrt (A^2 + B^2)

theorem parallel_lines_at_distance : 
  (∀ x y : ℝ, parallel_line1 x y → original_line x y → 
    distance_between_lines 4 3 (-10) 5 = 3) ∧
  (∀ x y : ℝ, parallel_line2 x y → original_line x y → 
    distance_between_lines 4 3 20 5 = 3) := by
  sorry

#check parallel_lines_at_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_l1176_117652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonic_functions_l1176_117626

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := -2 * x + 1
noncomputable def g (x : ℝ) : ℝ := x^3
noncomputable def h (x : ℝ) : ℝ := Real.log x / Real.log 10
noncomputable def k (x : ℝ) : ℝ := 1 / x

-- Define monotonicity
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Theorem statement
theorem not_monotonic_functions :
  Monotonic f ∧ Monotonic g ∧ Monotonic h ∧ ¬Monotonic k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonic_functions_l1176_117626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1176_117670

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.sqrt (x + 1)

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ -1 ∧ f x = y) ↔ y ≥ -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1176_117670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l1176_117614

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := x^3 - x^2 - 2

/-- The point of tangency -/
def point : ℝ × ℝ := (2, 2)

/-- The equation of the tangent line in general form -/
def tangent_line (x y : ℝ) : Prop := 8*x - y - 14 = 0

/-- Theorem stating that the tangent_line is indeed the tangent to f at point -/
theorem tangent_line_is_correct : 
  ∃ (m b : ℝ), (∀ x y, tangent_line x y ↔ y = m*x + b) ∧ 
               tangent_line point.1 point.2 ∧
               (deriv f point.1 = m) := by
  sorry

#check tangent_line_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l1176_117614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_obtuse_l1176_117691

-- Define the triangle with its altitudes
structure Triangle where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ

-- Define the property of having altitudes 8, 10, and 25
def has_given_altitudes (t : Triangle) : Prop :=
  t.h₁ = 8 ∧ t.h₂ = 10 ∧ t.h₃ = 25

-- Define what it means for an angle to be obtuse
def is_obtuse_angle (θ : ℝ) : Prop :=
  θ > Real.pi / 2 ∧ θ < Real.pi

-- Helper definition for is_angle_of
def is_angle_of (t : Triangle) (θ : ℝ) : Prop :=
  sorry  -- This would typically be defined based on the triangle's properties

-- Theorem statement
theorem largest_angle_is_obtuse (t : Triangle) 
  (h : has_given_altitudes t) : 
  ∃ θ, is_obtuse_angle θ ∧ 
  ∀ φ, (is_angle_of t φ → φ ≤ θ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_obtuse_l1176_117691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_division_theorem_l1176_117659

/-- Represents a number in base 6 --/
structure Base6 where
  value : ℕ
  isValid : value < 6^64 := by sorry

/-- Converts a base 6 number to its decimal (base 10) equivalent --/
def toDecimal (n : Base6) : ℕ := n.value

/-- Converts a decimal (base 10) number to its base 6 equivalent --/
def toBase6 (n : ℕ) : Base6 := ⟨n % (6^64), by sorry⟩

/-- Performs division in base 6 --/
def divBase6 (a b : Base6) : Base6 := 
  toBase6 (toDecimal a / toDecimal b)

theorem base6_division_theorem :
  let dividend : Base6 := toBase6 2045
  let divisor : Base6 := toBase6 14
  let quotient : Base6 := toBase6 51
  divBase6 dividend divisor = quotient := by sorry

#eval toDecimal (divBase6 (toBase6 2045) (toBase6 14))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_division_theorem_l1176_117659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_increasing_on_positive_reals_l1176_117674

-- Define the interval (0, +∞)
noncomputable def OpenPositiveReals : Set ℝ := {x : ℝ | x > 0}

-- Define the functions
def f_A : ℝ → ℝ := λ x ↦ 3 - x
def f_B : ℝ → ℝ := λ x ↦ x^2 - 3*x
noncomputable def f_C : ℝ → ℝ := λ x ↦ 1/x
def f_D : ℝ → ℝ := λ x ↦ |x|

-- Define what it means for a function to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

-- Theorem statement
theorem absolute_value_increasing_on_positive_reals :
  ¬IncreasingOn f_A OpenPositiveReals ∧
  ¬IncreasingOn f_B OpenPositiveReals ∧
  ¬IncreasingOn f_C OpenPositiveReals ∧
  IncreasingOn f_D OpenPositiveReals := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_increasing_on_positive_reals_l1176_117674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1176_117651

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3) + 1

theorem f_properties :
  ∃ (t : ℝ),
    (∀ (x : ℝ), f (t + x) = f (t - x)) ∧
    t > 0 ∧
    (∀ (t' : ℝ), (∀ (x : ℝ), f (t' + x) = f (t' - x)) ∧ t' > 0 → t ≤ t') ∧
    t = Real.pi / 12 ∧
    (∃ (a b : ℝ),
      a < b ∧
      (∃ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
        a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧ x₅ < x₆ ∧ x₆ ≤ b ∧
        f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0 ∧ f x₆ = 0) ∧
      b - a = 7 * Real.pi / 3 ∧
      (∀ (a' b' : ℝ),
        a' < b' ∧
        (∃ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
          a' ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧ x₅ < x₆ ∧ x₆ ≤ b' ∧
          f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0 ∧ f x₆ = 0) →
        b' - a' ≥ 7 * Real.pi / 3)) ∧
    (∀ (m : ℝ),
      (∃ (x₁ x₂ : ℝ),
        -Real.pi/12 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.pi/6 ∧
        m * f x₁ - 2 = 0 ∧ m * f x₂ - 2 = 0) ↔
      2/3 < m ∧ m ≤ Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1176_117651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l1176_117696

/-- The maximum area of a triangle with sides satisfying a^2 + b^2 + 3c^2 = 7 -/
theorem triangle_max_area (a b c : ℝ) (h : a^2 + b^2 + 3*c^2 = 7) :
  ∃ (S : ℝ), S = Real.sqrt 7 / 4 ∧ ∀ (A : ℝ), A ≤ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l1176_117696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aquarium_water_ratio_l1176_117609

/-- Represents an aquarium with given dimensions and water levels -/
structure Aquarium where
  length : ℚ
  width : ℚ
  height : ℚ
  initialFillRatio : ℚ
  spillRatio : ℚ
  finalWaterAmount : ℚ

/-- Calculates the ratio of final water amount to the amount after spilling -/
def waterRatio (a : Aquarium) : ℚ :=
  let totalVolume := a.length * a.width * a.height
  let initialWater := totalVolume * a.initialFillRatio
  let afterSpill := initialWater * (1 - a.spillRatio)
  a.finalWaterAmount / afterSpill

/-- Theorem stating that the water ratio for the given aquarium is 3 -/
theorem aquarium_water_ratio :
  let a : Aquarium := {
    length := 4,
    width := 6,
    height := 3,
    initialFillRatio := 1/2,
    spillRatio := 1/2,
    finalWaterAmount := 54
  }
  waterRatio a = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_aquarium_water_ratio_l1176_117609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_7x_mod_9_l1176_117663

theorem remainder_7x_mod_9 (x : ℕ) (h : x % 9 = 5) : (7 * x) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_7x_mod_9_l1176_117663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_bounds_l1176_117613

/-- A regular hexagon with side length 2 -/
structure RegularHexagon where
  side_length : ℝ
  is_regular : side_length = 2

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The path of the ball in the hexagon -/
structure BallPath (h : RegularHexagon) where
  start : Point
  hits : List Point
  hits_sequence : hits.length = 6

/-- The angle θ between BP and PQ -/
noncomputable def theta (h : RegularHexagon) (path : BallPath h) : ℝ :=
  sorry  -- Placeholder for the actual angle calculation

/-- The theorem stating the bounds of θ -/
theorem theta_bounds (h : RegularHexagon) (path : BallPath h) :
  Real.arctan (3 * Real.sqrt 3 / 10) ≤ theta h path ∧
  theta h path ≤ Real.arctan (3 * Real.sqrt 3 / 8) := by
  sorry

#check theta_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_bounds_l1176_117613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1176_117658

/-- An increasing geometric sequence with given properties -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  (a 1 * a 2 = 8) ∧ (a 1 + a 2 = 6) ∧ (∀ n : ℕ+, a (n + 1) > a n)

/-- The general term of the sequence -/
noncomputable def GeneralTerm (n : ℕ+) : ℝ := 2^(n.val : ℝ)

/-- The b_n sequence derived from a_n -/
def BSequence (a : ℕ+ → ℝ) (n : ℕ+) : ℝ := 2 * a n + 3

/-- The sum of the first n terms of the b_n sequence -/
noncomputable def SumBSequence (n : ℕ+) : ℝ := 2^((n.val : ℝ) + 2) - 4 + 3 * n.val

theorem geometric_sequence_properties (a : ℕ+ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ+, a n = GeneralTerm n) ∧
  (∀ n : ℕ+, (Finset.sum (Finset.range n.val) (fun i => BSequence a ⟨i + 1, Nat.succ_pos i⟩)) = SumBSequence n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1176_117658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l1176_117655

noncomputable def rotation_matrix (θ : Real) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

theorem smallest_rotation_power :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (m : ℕ), m > 0 → (rotation_matrix (2 * Real.pi / 3))^m = 1 → n ≤ m) ∧
  (rotation_matrix (2 * Real.pi / 3))^n = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l1176_117655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_length_eq_150_l1176_117653

/-- A rectangular garden with given perimeter and breadth -/
structure RectangularGarden where
  perimeter : ℝ
  breadth : ℝ

/-- The length of a rectangular garden -/
noncomputable def garden_length (g : RectangularGarden) : ℝ :=
  (g.perimeter / 2) - g.breadth

/-- Theorem: For a rectangular garden with perimeter 600 m and breadth 150 m, the length is 150 m -/
theorem garden_length_eq_150 (g : RectangularGarden) 
  (h1 : g.perimeter = 600) 
  (h2 : g.breadth = 150) : 
  garden_length g = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_length_eq_150_l1176_117653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l1176_117683

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then Real.exp x + x^2 else -Real.exp (-x) - x^2

theorem tangent_slope_at_one (h : ∀ x, f (-x) = -f x) :
  (deriv f) 1 = Real.exp (-1) - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l1176_117683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_resulting_polygon_sides_l1176_117631

/-- Represents a polygon with a given number of sides. -/
structure Polygon :=
  (sides : ℕ)

/-- The sequence of polygons from square to decagon. -/
def polygon_sequence : List Polygon :=
  [⟨4⟩, ⟨5⟩, ⟨6⟩, ⟨7⟩, ⟨8⟩, ⟨9⟩, ⟨10⟩]

/-- Calculates the number of exposed sides for a given polygon in the sequence. -/
def exposed_sides (p : Polygon) (index : ℕ) : ℕ :=
  if index = 0 ∨ index = (polygon_sequence.length - 1) then
    p.sides - 1
  else
    p.sides - 2

/-- Calculates the total number of exposed sides in the polygon sequence. -/
def total_exposed_sides : ℕ :=
  (List.enum polygon_sequence).map
    (fun (index, p) => exposed_sides p index)
  |>.sum

/-- The theorem stating that the resulting polygon has 37 sides. -/
theorem resulting_polygon_sides :
  total_exposed_sides = 37 := by
  sorry

#eval total_exposed_sides

end NUMINAMATH_CALUDE_ERRORFEEDBACK_resulting_polygon_sides_l1176_117631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_circuit_current_l1176_117634

/-- Represents a complex number -/
structure MyComplex : Type :=
  (re : ℝ) (im : ℝ)

/-- Addition of complex numbers -/
def MyComplex.add (z w : MyComplex) : MyComplex :=
  ⟨z.re + w.re, z.im + w.im⟩

/-- Multiplication of complex numbers -/
def MyComplex.mul (z w : MyComplex) : MyComplex :=
  ⟨z.re * w.re - z.im * w.im, z.re * w.im + z.im * w.re⟩

/-- Division of complex numbers -/
noncomputable def MyComplex.div (z w : MyComplex) : MyComplex :=
  let denom := w.re * w.re + w.im * w.im
  ⟨(z.re * w.re + z.im * w.im) / denom, (z.im * w.re - z.re * w.im) / denom⟩

/-- The impedance in the AC circuit -/
def Z : MyComplex := ⟨2, 4⟩

/-- The voltage in the AC circuit as a function of k -/
def V (k : ℝ) : MyComplex := ⟨4 - 2 * k, 0⟩

/-- The current in the AC circuit as a function of k -/
noncomputable def I (k : ℝ) : MyComplex := MyComplex.div (V k) Z

theorem ac_circuit_current (k : ℝ) :
  I k = ⟨(2 - k) / 5, -(4 - 2 * k) / 5⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_circuit_current_l1176_117634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chipmunk_stored_91_seeds_l1176_117633

/-- Represents the number of holes dug by the chipmunk -/
def chipmunk_holes : ℕ := sorry

/-- Represents the number of holes dug by the rabbit -/
def rabbit_holes : ℕ := sorry

/-- The number of seeds stored in each chipmunk hole -/
def chipmunk_seeds_per_hole : ℕ := 5

/-- The number of seeds stored in each rabbit hole -/
def rabbit_seeds_per_hole : ℕ := 7

/-- The rabbit dug 5 fewer holes than the chipmunk -/
axiom hole_difference : rabbit_holes = chipmunk_holes - 5

/-- Both animals stored the same number of seeds -/
axiom equal_seeds : chipmunk_holes * chipmunk_seeds_per_hole = rabbit_holes * rabbit_seeds_per_hole

/-- The theorem to prove: the chipmunk stored 91 seeds -/
theorem chipmunk_stored_91_seeds : chipmunk_holes * chipmunk_seeds_per_hole = 91 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chipmunk_stored_91_seeds_l1176_117633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subcommittee_count_l1176_117617

/-- The number of ways to select a three-person sub-committee from a committee of 8 people,
    where one specific person (the chairperson) must be included. -/
theorem subcommittee_count : Nat.choose 7 2 = 21 := by
  -- Define the total number of people in the committee
  let total_members : Nat := 8
  -- Define the size of the sub-committee
  let subcommittee_size : Nat := 3
  -- Define that one person (the chairperson) must be included
  let required_member : Nat := 1
  -- Calculate the number of ways to select the remaining members
  let remaining_selections := Nat.choose (total_members - required_member) (subcommittee_size - required_member)
  -- Prove that the number of ways to form the sub-committee is equal to 21
  have h : remaining_selections = 21 := by
    -- This is where the actual proof would go
    sorry
  exact h

#eval Nat.choose 7 2  -- This will evaluate to 21

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subcommittee_count_l1176_117617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_lambda_l1176_117621

theorem equilateral_triangle_lambda (ω : ℂ) (lambda : ℝ) : 
  Complex.abs ω = 3 → 
  lambda > 1 →
  (∃ (z : ℂ), Complex.abs z = 1 ∧ 
    Complex.abs (ω - ω^3) = Complex.abs (ω^3 - lambda*ω) ∧ 
    Complex.abs (lambda*ω - ω) = Complex.abs (ω - ω^3)) →
  lambda = 1 + Real.sqrt (32/3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_lambda_l1176_117621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_pairs_l1176_117682

def central_angle (n : ℕ) (i j : Fin n) : ℝ := sorry

theorem central_angle_pairs (n : ℕ) (h : n = 21) :
  let G := { e : Fin n → Fin n → Prop | ∀ i j, e i j ↔ central_angle n i j > 120 }
  let num_pairs := n.choose 2
  let max_edges := n^2 / 4
  num_pairs - max_edges ≥ 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_pairs_l1176_117682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_freddy_speed_ratio_l1176_117612

/-- Represents the average speed ratio of two travelers given their distances and times -/
noncomputable def average_speed_ratio (distance_eddy distance_freddy time_eddy time_freddy : ℝ) : ℝ :=
  (distance_eddy / time_eddy) / (distance_freddy / time_freddy)

/-- Theorem: The ratio of Eddy's average speed to Freddy's average speed is 4:1 -/
theorem eddy_freddy_speed_ratio :
  average_speed_ratio 900 300 3 4 = 4 := by
  -- Unfold the definition of average_speed_ratio
  unfold average_speed_ratio
  -- Simplify the expression
  simp [div_div_eq_mul_div]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_freddy_speed_ratio_l1176_117612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l1176_117624

theorem sin_cos_value (α : ℝ) (h : Real.cos α - Real.sin α = 1/2) :
  Real.sin α * Real.cos α = 3/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l1176_117624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_equals_cos_l1176_117645

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => Real.cos
  | n + 1 => fun x => deriv (f n) x

theorem f_2016_equals_cos : f 2016 = f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_equals_cos_l1176_117645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_train_length_is_150_l1176_117604

noncomputable section

-- Define the given conditions
def long_train_length : ℚ := 200
def speed_train1 : ℚ := 40
def speed_train2 : ℚ := 46
def crossing_time : ℚ := 210

-- Define the function to calculate the length of the shorter train
def shorter_train_length : ℚ :=
  let speed1_ms : ℚ := speed_train1 * (1000 / 3600)
  let speed2_ms : ℚ := speed_train2 * (1000 / 3600)
  let relative_speed : ℚ := speed2_ms - speed1_ms
  let total_length : ℚ := relative_speed * crossing_time
  total_length - long_train_length

-- Theorem statement
theorem shorter_train_length_is_150 :
  shorter_train_length = 150 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_train_length_is_150_l1176_117604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_is_five_l1176_117657

-- Define the triangle DEF
structure Triangle (DE DF EF : ℝ) : Prop where
  side_de : DE = 13
  side_df : DF = 13
  side_ef : EF = 24

-- Define the median DM
noncomputable def median_DM (DE DF EF : ℝ) : ℝ := 
  Real.sqrt (DE^2 - (EF/2)^2)

-- Theorem statement
theorem median_length_is_five (DE DF EF : ℝ) (t : Triangle DE DF EF) : 
  median_DM DE DF EF = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_is_five_l1176_117657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1176_117640

theorem remainder_theorem (x : ℕ) (h : 11 * x ≡ 1 [ZMOD 27]) : (13 + x) % 27 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1176_117640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l1176_117606

theorem tan_sum_problem (x y : Real) 
  (h1 : Real.tan x + Real.tan y = -20)
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 30) : 
  Real.tan (x + y) = -12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l1176_117606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_count_l1176_117615

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- An isosceles triangle inscribed in a circle -/
structure IsoscelesTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Checks if a point is inside a circle -/
def isInside (p : ℝ × ℝ) (c : Circle) : Prop :=
  distance p c.O < c.r

/-- Checks if a triangle is inscribed in a circle -/
def isInscribed (t : IsoscelesTriangle) (c : Circle) : Prop :=
  distance t.A c.O = c.r ∧ distance t.B c.O = c.r ∧ distance t.C c.O = c.r

/-- Checks if a triangle is isosceles with vertex angle 45° -/
def isIsosceles45 (t : IsoscelesTriangle) : Prop :=
  distance t.A t.B = distance t.A t.C ∧ 
  Real.arccos ((distance t.B t.C)^2 / (2 * (distance t.A t.B)^2)) = Real.pi/4

/-- Checks if one leg of the triangle passes through a point -/
def legPassesThrough (t : IsoscelesTriangle) (p : ℝ × ℝ) : Prop :=
  distance t.A p + distance p t.B = distance t.A t.B ∨ 
  distance t.A p + distance p t.C = distance t.A t.C

theorem isosceles_triangles_count 
  (c : Circle) (P : ℝ × ℝ) (h : isInside P c) :
  (∃ (t1 t2 t3 t4 : IsoscelesTriangle), 
    (∀ i : IsoscelesTriangle, isInscribed i c ∧ isIsosceles45 i ∧ legPassesThrough i P → 
      i = t1 ∨ i = t2 ∨ i = t3 ∨ i = t4) ∧
    t1 ≠ t2 ∧ t1 ≠ t3 ∧ t1 ≠ t4 ∧ t2 ≠ t3 ∧ t2 ≠ t4 ∧ t3 ≠ t4) ↔ 
  distance P c.O ≥ c.r * Real.sin (Real.pi/8) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_count_l1176_117615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_red_one_white_without_replacement_prob_two_red_with_replacement_l1176_117627

/-- Represents the number of red balls in the box -/
def num_red_balls : ℕ := 2

/-- Represents the number of white balls in the box -/
def num_white_balls : ℕ := 4

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := num_red_balls + num_white_balls

/-- Represents the probability of drawing a red ball in a single draw -/
def prob_red : ℚ := num_red_balls / total_balls

/-- Represents the probability of drawing a white ball in a single draw -/
def prob_white : ℚ := num_white_balls / total_balls

theorem prob_one_red_one_white_without_replacement :
  (num_red_balls / total_balls) * (num_white_balls / (total_balls - 1)) +
  (num_white_balls / total_balls) * (num_red_balls / (total_balls - 1)) = 8 / 15 := by
  sorry

theorem prob_two_red_with_replacement :
  (Nat.choose 5 2 : ℚ) * (prob_red ^ 2) * ((1 - prob_red) ^ 3) = 80 / 243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_red_one_white_without_replacement_prob_two_red_with_replacement_l1176_117627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_unit_area_and_radius_l1176_117667

/-- The area of a circular sector in terms of its radius and central angle. -/
noncomputable def sectorArea (r : ℝ) (θ : ℝ) : ℝ := Real.pi * r^2 * θ / (2 * Real.pi)

/-- Theorem: Given a circular sector with area 1 and radius 1, its central angle is 2 radians. -/
theorem central_angle_of_unit_area_and_radius (A : ℝ) (r : ℝ) (θ : ℝ) 
  (h1 : A = 1) (h2 : r = 1) (h3 : A = sectorArea r θ) : θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_unit_area_and_radius_l1176_117667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1176_117668

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos x ^ 2 - Real.sin x ^ 2) + 2 * Real.sin x * Real.cos x

-- State the theorem
theorem f_properties :
  -- 1. The smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- 2. For x ∈ [-π/3, π/3], the range is [-√3, 2]
  (∀ (y : ℝ), (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/3) (Real.pi/3) ∧ f x = y) ↔ y ∈ Set.Icc (-Real.sqrt 3) 2) ∧
  -- 3. For x ∈ [-π/3, π/3], f(x) is strictly decreasing on [π/12, 7π/12]
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc (Real.pi/12) (7*Real.pi/12) ∧ x₂ ∈ Set.Icc (Real.pi/12) (7*Real.pi/12) ∧ x₁ < x₂ → f x₁ > f x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1176_117668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_max_distance_l1176_117676

-- Define the curve C
noncomputable def C : ℝ → ℝ × ℝ := fun θ => (3 * Real.cos θ, Real.sin θ)

-- Define the line l
def l (a : ℝ) : ℝ → ℝ × ℝ := fun t => (a + 4 * t, 1 - t)

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (p : ℝ × ℝ) (a : ℝ) : ℝ :=
  let (x, y) := p
  abs (x + 4 * y - a - 4) / Real.sqrt 17

theorem intersection_points_and_max_distance :
  (∃ θ₁ θ₂, C θ₁ = l (-1) (3/4) ∧ C θ₂ = l (-1) (-21/100)) ∧
  (∀ θ, distance_point_to_line (C θ) (-16) ≤ Real.sqrt 17) ∧
  (∃ θ, distance_point_to_line (C θ) (-16) = Real.sqrt 17) ∧
  (∀ θ, distance_point_to_line (C θ) 8 ≤ Real.sqrt 17) ∧
  (∃ θ, distance_point_to_line (C θ) 8 = Real.sqrt 17) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_max_distance_l1176_117676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_of_exponentials_l1176_117637

theorem inequality_of_exponentials (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_eq1 : (2 : ℝ)^x = (3 : ℝ)^y) (h_eq2 : (3 : ℝ)^y = (5 : ℝ)^z) : 3*y < 2*x ∧ 2*x < 5*z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_of_exponentials_l1176_117637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1176_117671

-- Define the triangle and vectors
def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0

noncomputable def vector_m (A : Real) : Real × Real := (Real.sqrt 3, Real.cos A + 1)
noncomputable def vector_n (A : Real) : Real × Real := (Real.sin A, -1)

-- Define perpendicularity of vectors
def perpendicular (v w : Real × Real) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- State the theorem
theorem triangle_problem (A B C a b c : Real) :
  triangle_ABC A B C a b c →
  perpendicular (vector_m A) (vector_n A) →
  a = 2 →
  Real.cos B = Real.sqrt 3 / 3 →
  A = Real.pi / 3 ∧ b = 4 * Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1176_117671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ways_formula_l1176_117623

/-- The number of ways to form the product of n distinct letters -/
def product_ways (n : ℕ) : ℕ := 
  if n ≤ 1 then 1 else (2 * n - 2).factorial

/-- Theorem stating that the number of ways to form the product of n distinct letters is (2n-2)! for n > 1 -/
theorem product_ways_formula (n : ℕ) (h : n > 1) : product_ways n = (2 * n - 2).factorial := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ways_formula_l1176_117623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_equals_12pi_l1176_117632

-- Define the circle and sector
def P : ℝ × ℝ := (2, -1)
def Q : ℝ × ℝ := (-4, 5)
def central_angle : ℝ := 60

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Function to calculate area of a sector
noncomputable def sector_area (radius : ℝ) (angle : ℝ) : ℝ :=
  (Real.pi * radius^2) * (angle / 360)

-- Theorem statement
theorem sector_area_equals_12pi :
  sector_area (distance P Q) central_angle = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_equals_12pi_l1176_117632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_intersection_count_l1176_117662

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle
def circle_F (x y : ℝ) : Prop := (x-1)^2 + y^2 = 9

-- Define point P
noncomputable def P : ℝ × ℝ := (4, 0)

-- Define point A
noncomputable def A : ℝ × ℝ := (2, 2*Real.sqrt 2)

-- Define point B
noncomputable def B : ℝ × ℝ := (2, -2*Real.sqrt 2)

-- Theorem for part (I)
theorem chord_length :
  ∀ (x y : ℝ), parabola x y ∧ circle_F x y →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 :=
by
  sorry

-- Theorem for part (II)
theorem intersection_count :
  ∃! (n : ℕ), n = 2 ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    (y₁ - A.2) * (P.1 - A.1) = (x₁ - A.1) * (P.2 - A.2) ∧
    (y₂ - A.2) * (P.1 - A.1) = (x₂ - A.1) * (P.2 - A.2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_intersection_count_l1176_117662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1176_117644

theorem expression_evaluation (k : ℤ) :
  (2 : ℚ)^(-(3*k+2)) - 3*(2 : ℚ)^(-(3*k)) + (2 : ℚ)^(-3*k) = -7/4 * (2 : ℚ)^(-3*k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1176_117644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_thickness_estimate_l1176_117660

/-- The number of squares on which rice is placed -/
def num_squares : ℕ := 64

/-- The volume of rice grains in cubic meters per 10^7 grains -/
def rice_volume_per_10_7_grains : ℝ := 1

/-- The global arable land area in square meters -/
def global_arable_land : ℝ := 1.5 * 10^13

/-- Approximation of lg 2 -/
def lg2_approx : ℝ := 0.30

/-- Approximation of lg 3 -/
def lg3_approx : ℝ := 0.48

/-- The thickness of rice grains covering the global arable land -/
def rice_thickness : ℝ := 0.1

/-- Theorem stating that the thickness of rice grains covering the global arable land
    is approximately 0.1 meter, given the conditions of the problem -/
theorem rice_thickness_estimate :
  ∃ (total_grains : ℕ) (total_volume : ℝ),
    total_grains = 2^num_squares - 1 ∧
    total_volume = (total_grains : ℝ) / 10^7 ∧
    abs (rice_thickness - total_volume / global_arable_land) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_thickness_estimate_l1176_117660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_helmet_purchase_solution_l1176_117688

/-- Represents the store's helmet purchase scenario -/
structure HelmetPurchase where
  typeA_count : ℕ
  typeB_count : ℕ
  total_cost : ℕ
  price_difference : ℕ
  promotion_typeA_discount : ℚ
  promotion_typeB_discount : ℕ
  promotion_total : ℕ

/-- Theorem stating the solution to the helmet purchase problem -/
theorem helmet_purchase_solution (hp : HelmetPurchase)
  (h1 : hp.typeA_count = 20)
  (h2 : hp.typeB_count = 30)
  (h3 : hp.total_cost = 2920)
  (h4 : hp.price_difference = 11)
  (h5 : hp.promotion_typeA_discount = 1/5)
  (h6 : hp.promotion_typeB_discount = 6)
  (h7 : hp.promotion_total = 40) :
  ∃ (priceA priceB : ℕ) (promotionA : ℕ) (minCost : ℕ),
    priceA = 65 ∧
    priceB = 54 ∧
    promotionA = 14 ∧
    minCost = 1976 ∧
    priceA = priceB + hp.price_difference ∧
    hp.typeA_count * priceA + hp.typeB_count * priceB = hp.total_cost ∧
    promotionA ≥ (hp.promotion_total - promotionA) / 2 ∧
    minCost = promotionA * (priceA * (1 - hp.promotion_typeA_discount)).floor +
              (hp.promotion_total - promotionA) * (priceB - hp.promotion_typeB_discount) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_helmet_purchase_solution_l1176_117688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spitting_distance_l1176_117669

/-- The spitting distance problem -/
theorem spitting_distance 
  (billy_distance : ℝ) 
  (madison_percentage : ℝ) 
  (ryan_percentage : ℝ) 
  (h1 : billy_distance = 30) 
  (h2 : madison_percentage = 0.2) 
  (h3 : ryan_percentage = 0.5) : 
  let madison_distance := billy_distance * (1 + madison_percentage)
  let ryan_distance := madison_distance * (1 - ryan_percentage)
  ryan_distance = 18 := by 
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spitting_distance_l1176_117669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_value_l1176_117656

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x + x^2010

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| n+1 => deriv (f_n n)

theorem f_2011_value (x : ℝ) : f_n 2011 x = -Real.cos x + Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_value_l1176_117656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_six_times_one_half_l1176_117638

noncomputable def g (x : ℝ) : ℝ := 1 / (1 - x)

theorem g_six_times_one_half : g (g (g (g (g (g (1/2)))))) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_six_times_one_half_l1176_117638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1176_117603

theorem triangle_area (a b c : ℝ) (h1 : a = 4) (h2 : b = 3) (h3 : c = 3) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1176_117603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_ratio_complex_quadratic_l1176_117648

theorem root_ratio_complex_quadratic :
  ∃ (x₁ x₂ : ℂ),
    (x₁.im = 0) ∧
    (x₁^2 + (1 + I) * x₁ - 6 + 3 * I = 0) ∧
    (x₂^2 + (1 + I) * x₂ - 6 + 3 * I = 0) ∧
    (x₁ / x₂ = -6/5 - 3/5 * I) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_ratio_complex_quadratic_l1176_117648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_formation_problem_l1176_117618

-- Define the characteristics of a set
def is_set (S : Type) : Prop :=
  (∀ x y : S, x = y ∨ x ≠ y) ∧  -- Definiteness
  (∀ x y : S, True)             -- Unorderedness and distinctness

-- Define the conditions
def rational_ge_sqrt3 : Type := { x : ℚ // x^2 ≥ 3 }
def tall_students (school : Type) : Type := { s : school // True }  -- Cannot be defined precisely
def squares : Type := { n : ℕ // ∃ m : ℕ, n = m^2 }
def no_real_roots_quadratic : Type := { p : ℝ × ℝ × ℝ // let (a, b, c) := p; a ≠ 0 ∧ b^2 - 4*a*c < 0 }

-- Theorem statement
theorem set_formation_problem :
  is_set rational_ge_sqrt3 ∧
  (∀ school : Type, is_set (tall_students school)) ∧
  is_set squares ∧
  is_set no_real_roots_quadratic :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_formation_problem_l1176_117618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1176_117665

/-- If the cosine of an angle is negative and its tangent is positive, 
    then the terminal side of the angle is in the third quadrant. -/
theorem angle_in_third_quadrant (α : Real) 
  (h1 : Real.cos α < 0) 
  (h2 : Real.tan α > 0) : 
  α ∈ Set.Ioo (π) (3 * π / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1176_117665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heated_water_properties_l1176_117675

-- Define the state of water
structure WaterState where
  temperature : ℝ
  ionProduct : ℝ
  pH : ℝ
  hIonConcentration : ℝ
  ohIonConcentration : ℝ

-- Define the property of being neutral
def isNeutral (w : WaterState) : Prop :=
  w.hIonConcentration = w.ohIonConcentration

-- Define the relationship between temperature and ion concentrations
axiom temperature_increase_promotes_ionization (w1 w2 : WaterState) :
  w2.temperature > w1.temperature →
  w2.hIonConcentration > w1.hIonConcentration ∧
  w2.ohIonConcentration > w1.ohIonConcentration

-- Define ion product
def ionProduct (w : WaterState) : ℝ :=
  w.hIonConcentration * w.ohIonConcentration

-- Define pH calculation
noncomputable def pH (w : WaterState) : ℝ :=
  -Real.log w.hIonConcentration

-- Theorem to prove
theorem heated_water_properties (w1 w2 : WaterState) 
  (h_temp : w2.temperature > w1.temperature)
  (h_neutral1 : isNeutral w1)
  (h_neutral2 : isNeutral w2) :
  ionProduct w2 > ionProduct w1 ∧ 
  w2.pH < w1.pH ∧
  isNeutral w2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heated_water_properties_l1176_117675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jumps_l1176_117664

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℕ
  y : ℕ

/-- Defines the possible jumps a frog can make -/
inductive Jump : Point → Point → Prop where
  | double_x {a b : ℕ} : Jump ⟨a, b⟩ ⟨2 * a, b⟩
  | double_y {a b : ℕ} : Jump ⟨a, b⟩ ⟨a, 2 * b⟩
  | subtract_x {a b : ℕ} : a > b → Jump ⟨a, b⟩ ⟨a - b, b⟩
  | subtract_y {a b : ℕ} : b > a → Jump ⟨a, b⟩ ⟨a, b - a⟩

/-- Defines reachability from one point to another through a sequence of jumps -/
def Reachable (start finish : Point) : Prop :=
  ∃ (path : List Point), path.head? = some start ∧ path.getLast? = some finish ∧
    ∀ (i : ℕ) (p q : Point), i + 1 < path.length → 
      path[i]? = some p → path[i + 1]? = some q → Jump p q

/-- The main theorem to prove -/
theorem frog_jumps :
  (Reachable ⟨1, 1⟩ ⟨24, 40⟩) ∧
  (Reachable ⟨1, 1⟩ ⟨200, 4⟩) ∧
  ¬(Reachable ⟨1, 1⟩ ⟨40, 60⟩) ∧
  ¬(Reachable ⟨1, 1⟩ ⟨24, 60⟩) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jumps_l1176_117664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l1176_117647

theorem sin_2x_value (x : ℝ) (t : ℝ) (h_t : t ≠ 0) :
  let a : ℝ × ℝ := (Real.cos x, -Real.sin x)
  let b : ℝ × ℝ := (-Real.cos (π / 2 - x), Real.cos x)
  a = t • b →
  Real.sin (2 * x) = 1 ∨ Real.sin (2 * x) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l1176_117647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1176_117611

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define focal points
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define the locus of point M
noncomputable def locus_M (x y : ℝ) : Prop := y^2 = 8*x

-- Define the area of quadrilateral ABCD
noncomputable def area_ABCD (k : ℝ) : ℝ := 16 * (1 + k^2)^2 / (2*k^4 + 5*k^2 + 2)

theorem ellipse_properties :
  -- Part 1: Prove that the locus of M forms the given parabola
  (∀ x y : ℝ, locus_M x y ↔ 
    ∃ P : ℝ × ℝ, 
      (P.1 = F1.1 ∧ (y - P.2) = 0) ∧ 
      ((x - P.1) * (y - P.2) = 0) ∧
      ((x - (P.1 + F2.1)/2)^2 + (y - (P.2 + F2.2)/2)^2 = ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) / 4)) ∧
  -- Part 2: Prove that the minimum area of ABCD is 64/9
  (∃ k_min : ℝ, ∀ k : ℝ, area_ABCD k ≥ area_ABCD k_min ∧ area_ABCD k_min = 64/9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1176_117611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_is_sqrt_5_l1176_117680

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (φ : ℝ) : ℝ × ℝ := (1 + Real.cos φ, Real.sin φ)
def C₂ (θ : ℝ) : ℝ × ℝ := (Real.sin θ * Real.cos θ, Real.sin θ * Real.sin θ)

-- Define the intersection point A
noncomputable def A : ℝ × ℝ := (2 / Real.sqrt 5, 4 / 5)

-- Define the points M and N
noncomputable def M : ℝ × ℝ := (-(3 / Real.sqrt 5), 4 / Real.sqrt 5)
noncomputable def N : ℝ × ℝ := (4 / 5, -(2 / 5))

-- Theorem statement
theorem length_MN_is_sqrt_5 :
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_is_sqrt_5_l1176_117680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_PT_l1176_117619

/-- Given a plane with points P(0,4), Q(6,0), R(1,0), and S(5,3),
    where line segments PQ and RS intersect at point T,
    the length of segment PT is √((57/17)² + (-79/51)²). -/
theorem length_of_PT (P Q R S T : ℝ × ℝ) : 
  P = (0, 4) →
  Q = (6, 0) →
  R = (1, 0) →
  S = (5, 3) →
  (∃ t : ℝ, t ∈ Set.Ioo 0 1 ∧ 
    T = (1 - t) • P + t • Q ∧
    T = (1 - t) • R + t • S) →
  Real.sqrt ((57 / 17)^2 + (-79 / 51)^2) = Real.sqrt ((T.1 - P.1)^2 + (T.2 - P.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_PT_l1176_117619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_points_on_line_minimum_distance_to_interior_hyperbola_circle_intersection_l1176_117635

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the circle (renamed to avoid conflict)
def circleEq (r x y : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem exterior_points_on_line :
  ∀ x y : ℝ, line x y → x^2 - y^2 < 1 := by sorry

theorem minimum_distance_to_interior :
  ∀ x y : ℝ, x^2 - y^2 ≥ 1 → distance x y 0 (-1) ≥ Real.sqrt 6 / 2 := by sorry

theorem hyperbola_circle_intersection (a b r : ℝ) :
  hyperbola a b 2 1 →
  (∀ x y : ℝ, hyperbola a b x y ∧ circleEq r x y → distance x y 0 0 = Real.sqrt 2 * r) →
  r^2 = 8 * b^2 / (b^2 - 3) ∧ r > 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_points_on_line_minimum_distance_to_interior_hyperbola_circle_intersection_l1176_117635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_power_of_three_l1176_117650

theorem largest_integer_power_of_three (n : ℕ) : n ≤ 11 ↔ ∃ x : ℝ, n ≤ (3 : ℝ)^(x * (3 - x)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_power_of_three_l1176_117650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exp_inequality_l1176_117698

theorem negation_of_exp_inequality :
  (¬ ∀ x : ℝ, x > 0 → Real.exp x > x + 1) ↔ (∃ x : ℝ, x > 0 ∧ Real.exp x < x + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exp_inequality_l1176_117698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quilt_length_theorem_l1176_117673

noncomputable def quilt_length (width : ℝ) (patch_area : ℝ) (first_patch_cost : ℝ) (later_patch_cost : ℝ) (total_cost : ℝ) : ℝ :=
  let first_patches := 10
  let first_patches_cost := first_patches * first_patch_cost
  let remaining_cost := total_cost - first_patches_cost
  let additional_patches := remaining_cost / later_patch_cost
  let total_patches := first_patches + additional_patches
  let total_area := total_patches * patch_area
  total_area / width

theorem quilt_length_theorem :
  quilt_length 20 4 10 5 450 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quilt_length_theorem_l1176_117673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1176_117695

noncomputable def Ellipse (a b : ℝ) (h : a > b ∧ b > 0) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

noncomputable def ChordLength (a b : ℝ) : ℝ := 2 * b^2 / a

noncomputable def SlopePA (x y : ℝ) : ℝ := y / (x + 2)

noncomputable def SlopePB (x y : ℝ) : ℝ := y / (x - 2)

theorem ellipse_properties (a b : ℝ) (h : a > b ∧ b > 0) :
  Eccentricity a b = 1/2 ∧ ChordLength a b = 3 →
  (∀ x y, (x, y) ∈ Ellipse a b h → x^2/4 + y^2/2 = 1) ∧
  (∀ x y, (x, y) ∈ Ellipse a b h ∧ x ≠ 2 ∧ x ≠ -2 →
    SlopePA x y * SlopePB x y = -1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1176_117695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_from_origin_l1176_117661

/-- Circle C₁ in the Cartesian plane -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- Circle C₂ in the Cartesian plane -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 9 = 0

/-- Point P that forms equal length tangents to C₁ and C₂ -/
def P_locus (x y : ℝ) : Prop := 3*x + 4*y - 4 = 0

/-- The minimum distance from the origin to point P -/
noncomputable def min_distance : ℝ := 4/5

/-- Theorem stating the minimum distance from origin to P -/
theorem min_distance_from_origin :
  ∀ x y : ℝ, P_locus x y → Real.sqrt (x^2 + y^2) ≥ min_distance :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_from_origin_l1176_117661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_owner_profit_percentage_l1176_117646

-- Define the problem parameters
def num_pens : ℕ := 500
def base_cost : ℕ := 450
def tax_rate : ℚ := 5 / 100
def shipping_fee : ℚ := 20
def discount_rate : ℚ := 3 / 100

-- Define the calculation steps
def cost_price : ℚ := base_cost + (base_cost * tax_rate) + shipping_fee
def selling_price : ℚ := num_pens * (1 - discount_rate)
def profit : ℚ := selling_price - cost_price
def profit_percentage : ℚ := (profit / cost_price) * 100

-- The theorem to be proved
theorem store_owner_profit_percentage :
  ∃ (ε : ℚ), abs (profit_percentage - (-152 / 100)) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_owner_profit_percentage_l1176_117646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_is_inverse_e_l1176_117677

/-- The point where the tangent line touches the curve y = ln(x) -/
noncomputable def tangent_point : ℝ := Real.exp 1

/-- The equation of the tangent line y = kx -/
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x

/-- The natural logarithm curve -/
noncomputable def ln_curve (x : ℝ) : ℝ := Real.log x

theorem tangent_slope_is_inverse_e (k : ℝ) :
  (∀ x, tangent_line k x = ln_curve x → x = tangent_point) →
  tangent_line k 0 = ln_curve 0 →
  k = 1 / Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_is_inverse_e_l1176_117677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_fraction_l1176_117686

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem simple_interest_fraction :
  ∀ (principal : ℝ), principal > 0 →
  simple_interest principal 5 4 = principal / 5 :=
by
  intro principal h_pos
  unfold simple_interest
  simp [mul_assoc, mul_comm, mul_div_cancel']
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_fraction_l1176_117686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_seven_acute_triangles_l1176_117641

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of 5 points in a plane -/
def FivePoints : Type := Fin 5 → Point

/-- Predicate to check if three points are collinear -/
def areCollinear (p q r : Point) : Prop := sorry

/-- Predicate to check if a triangle is acute -/
def isAcuteTriangle (p q r : Point) : Prop := sorry

/-- The number of acute triangles formed by a set of 5 points -/
def numAcuteTriangles (points : FivePoints) : ℕ := sorry

/-- The main theorem -/
theorem at_most_seven_acute_triangles (points : FivePoints) 
  (h : ∀ (i j k : Fin 5), i ≠ j → j ≠ k → i ≠ k → ¬areCollinear (points i) (points j) (points k)) :
  numAcuteTriangles points ≤ 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_seven_acute_triangles_l1176_117641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_floor_equation_l1176_117684

theorem min_floor_equation (n : ℕ) : 
  (∀ k : ℕ, k^2 + Int.floor (n / k^2 : ℚ) ≥ 1991) ∧ 
  (∃ k : ℕ, k^2 + Int.floor (n / k^2 : ℚ) = 1991) ↔ 
  1024 * 967 ≤ n ∧ n ≤ 1024 * 967 + 1023 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_floor_equation_l1176_117684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_from_asymptotes_l1176_117636

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + (h.b / h.a)^2)

/-- The slope of the asymptotes of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ := h.b / h.a

theorem hyperbola_eccentricity_from_asymptotes (h : Hyperbola) 
  (h_asymptote : asymptote_slope h = 1/3) : 
  eccentricity h = Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_from_asymptotes_l1176_117636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_excellent_proportion_l1176_117620

/-- Represents the proportion of students with excellent grade -/
noncomputable def excellent_proportion (total : ℝ) (excellent : ℝ) : ℝ := excellent / total

/-- Represents the average score of the class -/
noncomputable def class_average (total : ℝ) (excellent : ℝ) : ℝ :=
  (excellent * 95 + (total - excellent) * 80) / total

/-- Theorem stating the minimum proportion of students with excellent grades -/
theorem minimum_excellent_proportion (total : ℝ) (excellent : ℝ) 
  (h_total_pos : total > 0)
  (h_excellent_nonneg : excellent ≥ 0)
  (h_excellent_le_total : excellent ≤ total)
  (h_class_avg : class_average total excellent ≥ 90) :
  excellent_proportion total excellent ≥ 2/3 := by
  sorry

#check minimum_excellent_proportion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_excellent_proportion_l1176_117620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1176_117607

/-- An arithmetic sequence satisfying given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 + a 3 = 14 ∧ a 4 - a 1 = 6

/-- A geometric sequence satisfying given conditions -/
def geometric_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) (m : ℕ) : Prop :=
  b 2 = a 1 ∧ b 3 = a 3 ∧ b 6 = a m

/-- The main theorem stating the properties of the sequences -/
theorem sequence_properties (a : ℕ → ℝ) (b : ℕ → ℝ) (m : ℕ) 
    (h_arith : arithmetic_sequence a) (h_geom : geometric_sequence b a m) :
    (∀ n : ℕ, a n = 2 * (n : ℝ) + 2) ∧ m = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1176_117607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jungkook_eunji_between_l1176_117602

/-- The number of students between two positions in a line-up -/
def studentsBetween (total : ℕ) (pos1 : ℕ) (pos2 : ℕ) : ℕ :=
  Int.natAbs (pos1 - pos2) - 1

theorem jungkook_eunji_between :
  let total_students : ℕ := 28
  let eunji_position : ℕ := 20
  let jungkook_position : ℕ := 14
  studentsBetween total_students jungkook_position eunji_position = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jungkook_eunji_between_l1176_117602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_rental_savings_correct_l1176_117666

/-- Calculates the savings from renting a parking space monthly instead of weekly for a year -/
noncomputable def parkingRentalSavings : ℝ :=
  let weeklyRate : ℝ := 10
  let nonPeakDiscount : ℝ := 2
  let nonPeakMonths : ℕ := 4
  let peakMonthlyRate : ℝ := 40
  let nonPeakMonthlyRate : ℝ := 35
  let weeksInYear : ℝ := 52
  let monthsInYear : ℕ := 12

  let nonPeakWeeks : ℝ := (nonPeakMonths : ℝ) * weeksInYear / (monthsInYear : ℝ)
  let peakWeeks : ℝ := weeksInYear - nonPeakWeeks

  let weeklyTotal : ℝ := nonPeakWeeks * (weeklyRate - nonPeakDiscount) + peakWeeks * weeklyRate
  let monthlyTotal : ℝ := (nonPeakMonths : ℝ) * nonPeakMonthlyRate + ((monthsInYear - nonPeakMonths) : ℝ) * peakMonthlyRate

  weeklyTotal - monthlyTotal

theorem parking_rental_savings_correct : 
  ∃ (ε : ℝ), abs (parkingRentalSavings - 25.34) < ε ∧ ε < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_rental_savings_correct_l1176_117666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1176_117690

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

/-- Given a positive geometric sequence with a₁ = 1 and S₅ = 5S₃ - 4, prove S₄ = 15 -/
theorem geometric_sequence_sum (q : ℝ) (h₁ : q > 0) (h₂ : q ≠ 1) :
  geometricSum 1 q 5 = 5 * geometricSum 1 q 3 - 4 →
  geometricSum 1 q 4 = 15 := by
  sorry

#check geometric_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1176_117690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escape_strategy_exists_l1176_117642

/-- Represents a figure that can be placed on a captive's forehead -/
structure Figure : Type :=
  (id : ℕ)

/-- Represents a captive with a figure on their forehead -/
structure Captive : Type :=
  (id : ℕ)
  (figure : Figure)

/-- Represents the state of the game -/
structure GameState : Type :=
  (captives : List Captive)
  (figures : List Figure)

/-- A function that determines if a figure guess is correct for a given captive -/
def is_correct_guess (state : GameState) (captive : Captive) (guess : Figure) : Prop :=
  captive.figure = guess

/-- A strategy function that takes the game state and a captive, and returns a guess -/
noncomputable def strategy (state : GameState) (captive : Captive) : Figure :=
  sorry

/-- The main theorem stating that there exists a strategy that guarantees at least one correct guess -/
theorem escape_strategy_exists (state : GameState) : 
  (state.captives.length ≥ 3) →
  (∀ f₁ f₂ : Figure, f₁ ≠ f₂ → 
    (state.captives.filter (λ c => c.figure.id = f₁.id)).length ≠ 
    (state.captives.filter (λ c => c.figure.id = f₂.id)).length) →
  ∃ c : Captive, c ∈ state.captives ∧ is_correct_guess state c (strategy state c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escape_strategy_exists_l1176_117642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ceiling_sqrt_equals_154_l1176_117630

open Real

/-- Ceiling function -/
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

/-- Sum of ceiling of square roots from 5 to 36 -/
noncomputable def sum_ceiling_sqrt : ℕ :=
  (Finset.range 32).sum (fun n => (ceiling (Real.sqrt ((n + 5 : ℕ) : ℝ))).toNat)

/-- Theorem stating that the sum of ceiling of square roots from 5 to 36 equals 154 -/
theorem sum_ceiling_sqrt_equals_154 : sum_ceiling_sqrt = 154 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ceiling_sqrt_equals_154_l1176_117630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_difference_specific_quadrilateral_l1176_117679

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicTangentialQuadrilateral where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Conditions
  cyclic : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
  tangential : ∃ (r : ℝ), r > 0 ∧ r * (a + b + c + d) = (a * c + b * d)

/-- The absolute difference between segments created by the inscribed circle's tangency point -/
noncomputable def tangentPointDifference (q : CyclicTangentialQuadrilateral) : ℝ :=
  let s := (q.a + q.b + q.c + q.d) / 2
  let K := Real.sqrt ((s - q.a) * (s - q.b) * (s - q.c) * (s - q.d))
  let r := K / s
  let n := (70 + Real.sqrt (70 * 70 + 4 * r * r)) / 2
  |n - (q.c - n)|

/-- The main theorem -/
theorem tangent_point_difference_specific_quadrilateral :
  ∃ (q : CyclicTangentialQuadrilateral),
    q.a = 80 ∧ q.b = 100 ∧ q.c = 140 ∧ q.d = 120 ∧
    abs (tangentPointDifference q - 50.726) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_difference_specific_quadrilateral_l1176_117679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_not_sophomores_l1176_117629

theorem percentage_not_sophomores (total_students : ℕ) 
  (junior_percentage : ℚ) (seniors : ℕ) (freshman_sophomore_diff : ℕ) :
  total_students = 800 →
  junior_percentage = 22 / 100 →
  seniors = 160 →
  freshman_sophomore_diff = 64 →
  (↑(total_students - (total_students / 4)) / ↑total_students) * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_not_sophomores_l1176_117629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_40_l1176_117616

/-- A rectangle ABCD with a circle touching sides AB and AD, passing through C, and intersecting DC at N -/
structure CircleTouchingRectangle where
  /-- The length of side AB -/
  ab : ℝ
  /-- The length of side AD -/
  ad : ℝ
  /-- The x-coordinate of point N where the circle intersects DC -/
  n_x : ℝ
  /-- The circle touches AB and AD, passes through C, and intersects DC at N -/
  circle_properties : True

/-- The area of trapezoid ABND in a CircleTouchingRectangle -/
noncomputable def trapezoid_area (r : CircleTouchingRectangle) : ℝ :=
  (r.ab + r.n_x) / 2 * r.ad

/-- Theorem: The area of trapezoid ABND is 40 square units -/
theorem trapezoid_area_is_40 (r : CircleTouchingRectangle) 
    (h1 : r.ab = 9) 
    (h2 : r.ad = 8)
    (h3 : r.n_x = 1) : 
  trapezoid_area r = 40 := by
  unfold trapezoid_area
  rw [h1, h2, h3]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_40_l1176_117616
