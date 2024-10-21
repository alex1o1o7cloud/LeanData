import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_2310_smallest_coprime_to_2310_is_correct_l601_60106

/-- The smallest integer greater than 1 that is coprime to 2310 -/
def smallest_coprime_to_2310 : ℕ := 13

/-- 2310 is the product of its prime factors -/
theorem factor_2310 : 2310 = 2 * 3 * 5 * 7 * 11 := by sorry

theorem smallest_coprime_to_2310_is_correct :
  smallest_coprime_to_2310 > 1 ∧
  Nat.Coprime smallest_coprime_to_2310 2310 ∧
  ∀ n : ℕ, 1 < n → n < smallest_coprime_to_2310 → ¬Nat.Coprime n 2310 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_2310_smallest_coprime_to_2310_is_correct_l601_60106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_in_unit_interval_condition_on_c_l601_60147

-- Define the function f
def f (x : ℝ) : ℝ := -3 * x^2 - 3 * x + 18

-- Define the function g
def g (c : ℝ) (x : ℝ) : ℝ := -3 * x^2 + 5 * x + c

-- Theorem for the range of f in [0,1]
theorem range_of_f_in_unit_interval :
  Set.range (fun x => f x) ∩ Set.Icc 0 1 = Set.Icc 12 18 := by sorry

-- Theorem for the condition on c
theorem condition_on_c :
  ∀ c : ℝ, (∀ x ∈ Set.Icc 1 4, g c x ≤ 0) ↔ c ≤ -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_in_unit_interval_condition_on_c_l601_60147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt5_sqrt13_sum_opposite_of_x_minus_y_plus_sqrt3_sum_of_decimal_parts_l601_60178

-- Question 1
theorem sqrt5_sqrt13_sum (a b : ℝ) : 
  (a = Real.sqrt 5 - 2) → (b = 3) → a + b + 5 = Real.sqrt 5 + 6 := by sorry

-- Question 2
theorem opposite_of_x_minus_y_plus_sqrt3 (x : ℤ) (y : ℝ) :
  (10 + Real.sqrt 3 = ↑x + y) → (0 < y) → (y < 1) → -(↑x - y + Real.sqrt 3) = -12 := by sorry

-- Question 3
theorem sum_of_decimal_parts (a b : ℝ) :
  (a = 5 + Real.sqrt 11 - 8) → (b = 5 - Real.sqrt 11 - 1) → a + b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt5_sqrt13_sum_opposite_of_x_minus_y_plus_sqrt3_sum_of_decimal_parts_l601_60178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_problem_l601_60140

theorem binomial_coefficient_problem (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (x^2 + a/x)^5 = -1) →
  (∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → 
    ∃ f : ℝ → ℝ, (x^2 + a/x)^5 = c*x + f x ∧ (∀ y : ℝ, y ≠ 0 → f y ≠ y)) →
  c = -80 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_problem_l601_60140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l601_60169

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 2 * Real.sqrt 3 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  (abs (Real.sqrt 3 * x - y + 2 * Real.sqrt 3)) / 2

-- Theorem statement
theorem distance_range :
  ∀ x y : ℝ, curve_C x y →
  2 * Real.sqrt 3 - 1 ≤ distance_to_line x y ∧ 
  distance_to_line x y ≤ 2 * Real.sqrt 3 + 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l601_60169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_parallel_to_tangent_l601_60121

-- Define the curve
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the point M
def M : ℝ × ℝ := (1, 1)

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 2 * x - y + 4 = 0

theorem line_through_P_parallel_to_tangent :
  (line_equation P.1 P.2) ∧
  (∃ m : ℝ, (∀ x y : ℝ, line_equation x y ↔ y - P.2 = m * (x - P.1)) ∧
            m = (deriv f) M.1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_parallel_to_tangent_l601_60121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l601_60162

-- Define the circles
def Circle (center : ℝ × ℝ) (radius : ℝ) : Type := 
  {p : ℝ × ℝ // Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) = radius}

-- Define the problem setup
structure CircleProblem where
  center : ℝ × ℝ
  smallRadius : ℝ
  largeRadius : ℝ
  smallArc : ℝ  -- Length of 60-degree arc on small circle
  largeArc : ℝ  -- Length of 40-degree arc on large circle
  h1 : smallArc = 60 / 360 * (2 * Real.pi * smallRadius)
  h2 : largeArc = 40 / 360 * (2 * Real.pi * largeRadius)
  h3 : smallArc = largeArc  -- The arcs have the same length

-- Define the theorem
theorem circle_area_ratio (problem : CircleProblem) :
  (Real.pi * problem.smallRadius^2) / (Real.pi * problem.largeRadius^2) = 4 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l601_60162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_sum_of_coefficients_l601_60161

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem even_function_sum_of_coefficients 
  (a b : ℝ) 
  (h1 : ∀ x ∈ Set.Icc (a - 1) (3 * a), f a b x = f a b (-x)) 
  (h2 : Set.Nonempty (Set.Icc (a - 1) (3 * a))) :
  a + b = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_sum_of_coefficients_l601_60161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_perimeter_l601_60142

/-- An isosceles trapezoid with given base lengths and height -/
structure IsoscelesTrapezoid where
  baseAD : ℝ
  baseBC : ℝ
  height : ℝ

/-- Calculate the perimeter of an isosceles trapezoid -/
noncomputable def perimeter (t : IsoscelesTrapezoid) : ℝ :=
  t.baseAD + t.baseBC + 2 * Real.sqrt ((t.baseAD - t.baseBC)^2 / 4 + t.height^2)

/-- Theorem stating that the perimeter of the specific isosceles trapezoid is 160 + 2√765 -/
theorem specific_trapezoid_perimeter :
  let t : IsoscelesTrapezoid := { baseAD := 98, baseBC := 62, height := 21 }
  perimeter t = 160 + 2 * Real.sqrt 765 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_perimeter_l601_60142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_inequality_solution_set_l601_60143

noncomputable section

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Axioms
axiom f_domain (x : ℝ) : x > 0 → f x ≠ 0

axiom f_functional_equation (x y : ℝ) : x > 0 → y > 0 → f (x * y) = f x + f y

axiom f_positive (x : ℝ) : x > 1 → f x > 0

axiom f_9_eq_8 : f 9 = 8

-- Theorems
theorem f_monotone_increasing (x₁ x₂ : ℝ) : 
  x₁ > 0 → x₂ > 0 → x₁ > x₂ → f x₁ > f x₂ := by sorry

theorem f_inequality_solution_set :
  {x : ℝ | f (x^2 - 2*x) - f (2 - x) < 4} = Set.Ioo (-3) 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_inequality_solution_set_l601_60143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_l601_60113

variable (C : ℝ)
variable (x y : ℝ → ℝ)

noncomputable def solution (C : ℝ) (y : ℝ) : ℝ := 
  C * Real.exp (Real.sin y) - 2 * (1 + Real.sin y)

theorem solution_satisfies_diff_eq (C : ℝ) :
  ∀ y, (x y) = solution C y →
    (deriv x y) = 1 / ((x y) * Real.cos y + Real.sin (2 * y)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_l601_60113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_f_range_integers_min_difference_bounds_l601_60149

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Theorem 1: f(1/x) = -f(x)
theorem f_inverse (x : ℝ) (hx : x ≠ -1 ∧ x ≠ 0) : f (1/x) = -f x := by
  sorry

-- Theorem 2: Range of f(x) for x ∈ ℤ is [-3, 1]
theorem f_range_integers :
  ∀ (x : ℤ), -3 ≤ f (↑x) ∧ f (↑x) ≤ 1 := by
  sorry

-- Theorem 3: The minimum value of M - m is 4
theorem min_difference_bounds :
  ∃ (m M : ℝ), (∀ (x : ℤ), m ≤ f (↑x) ∧ f (↑x) ≤ M) ∧
  (∀ (m' M' : ℝ), (∀ (x : ℤ), m' ≤ f (↑x) ∧ f (↑x) ≤ M') → M - m ≤ M' - m') ∧
  M - m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_f_range_integers_min_difference_bounds_l601_60149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_condition_l601_60168

/-- A function that satisfies f(x)f(y) = f(x+y) for all real x and y -/
noncomputable def f (x : ℝ) : ℝ := 2^x

/-- Theorem stating that f satisfies the given condition -/
theorem f_satisfies_condition : ∀ x y : ℝ, f x * f y = f (x + y) := by
  intro x y
  simp [f]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_condition_l601_60168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_and_inverse_l601_60164

open Matrix

variable {α : Type*} [Field α]
variable {n : ℕ}

theorem det_product_and_inverse {A B : Matrix (Fin n) (Fin n) α} 
  (h1 : Matrix.det A = 5) (h2 : Matrix.det B = 7) :
  Matrix.det (A * B) = 35 ∧ Matrix.det (A⁻¹) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_and_inverse_l601_60164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_circle_origin_l601_60174

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Line l with slope k -/
structure Line where
  k : ℝ

/-- Point on the ellipse -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

def onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def onLine (p : Point) (l : Line) : Prop :=
  p.y = l.k * p.x - Real.sqrt 3

theorem ellipse_intersection_circle_origin (e : Ellipse) (l : Line) :
  eccentricity e = Real.sqrt 3 / 2 →
  e.a = 1 →
  ∃ A B : Point,
    onEllipse A e ∧ onEllipse B e ∧
    onLine A l ∧ onLine B l ∧
    A.x * B.x + A.y * B.y = 0 →
  l.k = Real.sqrt 11 / 2 ∨ l.k = -Real.sqrt 11 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_circle_origin_l601_60174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_room_together_l601_60127

/-- Represents the time it takes to paint a room together given individual painting rates -/
noncomputable def paint_time_together (taylor_rate : ℝ) (jennifer_rate : ℝ) : ℝ :=
  1 / (1 / taylor_rate + 1 / jennifer_rate)

/-- Theorem stating that Taylor and Jennifer can paint a room together in 60/11 hours -/
theorem paint_room_together :
  let taylor_rate : ℝ := 12
  let jennifer_rate : ℝ := 10
  paint_time_together taylor_rate jennifer_rate = 60 / 11 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_room_together_l601_60127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_common_points_l601_60198

/-- The first equation representing a set of points in the plane -/
def equation1 (x y : ℝ) : Prop := (2*x - y + 3) * (4*x + y - 5) = 0

/-- The second equation representing a set of points in the plane -/
def equation2 (x y : ℝ) : Prop := (x + y - 3) * (3*x - 4*y + 8) = 0

/-- A point satisfies both equations -/
def is_common_point (p : ℝ × ℝ) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

/-- The set of all common points -/
def common_points : Set (ℝ × ℝ) :=
  {p | is_common_point p}

/-- The theorem stating that there are exactly 4 common points -/
theorem four_common_points : ∃ (s : Finset (ℝ × ℝ)), s.card = 4 ∧ ∀ p, p ∈ s ↔ p ∈ common_points := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_common_points_l601_60198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l601_60158

-- Define the function f(x)
def f (x : ℝ) := x^3 - 3*x^2 + 1

-- Define the domain
def domain : Set ℝ := Set.Icc (-1) 3

-- Theorem statement
theorem f_properties :
  ∃ (increasing_intervals : Set (Set ℝ)) (decreasing_interval : Set ℝ) (max_value min_value : ℝ),
    -- Intervals of increase
    increasing_intervals = {Set.Ioo (-1) 0, Set.Ioo 2 3} ∧
    -- Interval of decrease
    decreasing_interval = Set.Ioo 0 2 ∧
    -- Maximum value
    max_value = 1 ∧
    -- Minimum value
    min_value = -3 ∧
    -- Function is increasing on the specified intervals
    (∀ i ∈ increasing_intervals, ∀ x y, x ∈ i → y ∈ i → x < y → f x < f y) ∧
    -- Function is decreasing on the specified interval
    (∀ x y, x ∈ decreasing_interval → y ∈ decreasing_interval → x < y → f x > f y) ∧
    -- Maximum value property
    (∀ x, x ∈ domain → f x ≤ max_value) ∧
    -- Minimum value property
    (∀ x, x ∈ domain → f x ≥ min_value) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l601_60158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_of_exponential_l601_60175

-- Define the exponential function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the inverse function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem statement
theorem inverse_function_of_exponential 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : f a 2 = 4) :
  Function.LeftInverse g (f a) ∧ Function.RightInverse g (f a) := by
  sorry

#check inverse_function_of_exponential

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_of_exponential_l601_60175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l601_60115

theorem solve_exponential_equation (x : ℝ) : (6 : ℝ)^(x + 2) = 1 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l601_60115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l601_60157

noncomputable def f (x : ℝ) := Real.sin x ^ 2 - Real.cos x ^ 2

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l601_60157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l601_60186

theorem sin_minus_cos_value (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = 4/3) 
  (h2 : π/4 < θ) 
  (h3 : θ < π/2) : 
  Real.sin θ - Real.cos θ = Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l601_60186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_people_for_cheaper_package_l601_60197

/-- Represents the cost calculation for badminton venue billing schemes -/
structure BadmintonBilling where
  hourlyRate : ℕ → ℕ → ℕ  -- Function that takes number of people and hours, returns cost
  entranceFee : ℕ → ℕ     -- Function that takes number of people, returns entrance fee cost

/-- Package billing scheme -/
def packageScheme : BadmintonBilling :=
  { hourlyRate := λ _ h ↦ 90 * h,
    entranceFee := λ n ↦ 10 * n }

/-- Per person billing scheme -/
def perPersonScheme : BadmintonBilling :=
  { hourlyRate := λ n h ↦ if h ≤ 3 then 54 * n else 54 * n + 8 * n * (h - 3),
    entranceFee := λ _ ↦ 0 }

/-- Calculate total cost for a given billing scheme -/
def totalCost (scheme : BadmintonBilling) (people : ℕ) (hours : ℕ) : ℕ :=
  scheme.hourlyRate people hours + scheme.entranceFee people

/-- Theorem: The minimum number of people for package scheme to be cheaper is 8 -/
theorem min_people_for_cheaper_package : 
  ∀ n : ℕ, n ≥ 8 ↔ 
    totalCost packageScheme n 6 < totalCost perPersonScheme n 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_people_for_cheaper_package_l601_60197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l601_60114

-- Define the function f(x) = ln x + 2x - 6
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

-- Theorem statement
theorem root_in_interval :
  ∃ r : ℝ, r ∈ Set.Ioo 2 3 ∧ f r = 0 := by
  sorry

-- Additional lemmas to support the main theorem
lemma f_continuous : Continuous f := by
  sorry

lemma f_monotone : StrictMono f := by
  sorry

lemma f_neg_at_two : f 2 < 0 := by
  sorry

lemma f_pos_at_three : f 3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l601_60114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l601_60144

/-- The volume of a regular triangular pyramid with a circumscribed sphere of radius R
    and an angle γ between lateral faces. -/
noncomputable def regularTriangularPyramidVolume (R : ℝ) (γ : ℝ) : ℝ :=
  (2 * R^3 * Real.sqrt 3 * (Real.cos (γ/2))^2 * (1 - 2 * Real.cos γ)) /
  (27 * (Real.sin (γ/2))^3)

/-- Theorem stating the volume of a regular triangular pyramid with a circumscribed sphere. -/
theorem regular_triangular_pyramid_volume
  (R : ℝ) (γ : ℝ) (h₁ : R > 0) (h₂ : 0 < γ ∧ γ < π) :
  regularTriangularPyramidVolume R γ =
    (2 * R^3 * Real.sqrt 3 * (Real.cos (γ/2))^2 * (1 - 2 * Real.cos γ)) /
    (27 * (Real.sin (γ/2))^3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l601_60144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_is_42_l601_60163

def sit_ups : List (Nat × Nat) := [(35, 3), (38, 5), (42, 7), (45, 4), (48, 4)]

def total_students : Nat := sit_ups.foldl (fun acc (_, count) => acc + count) 0

def median_position (n : Nat) : Nat := (n + 1) / 2

def cumulative_count (data : List (Nat × Nat)) : List (Nat × Nat) :=
  data.scanl (fun (_, acc) (value, count) => (value, acc + count)) (0, 0)

theorem median_is_42 : 
  total_students = 23 → 
  (cumulative_count sit_ups).find? (fun (_, count) => count ≥ median_position total_students) = some (42, 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_is_42_l601_60163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_and_double_sum_l601_60180

theorem tan_sum_and_double_sum (α β : Real) 
  (h1 : Real.tan α = 1/7) (h2 : Real.tan β = 1/3) : 
  Real.tan (α + β) = 1/2 ∧ Real.tan (α + 2*β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_and_double_sum_l601_60180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_396_l601_60185

def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_of_distinct_prime_factors_396 : 
  sum_of_distinct_prime_factors 396 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_396_l601_60185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_max_triangle_area_l601_60105

-- Define the line l and parabola C
def line_l (x y b : ℝ) : Prop := y = 2 * x + b
def parabola_C (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the tangency condition
def is_tangent (b : ℝ) : Prop :=
  ∃ x y, line_l x y b ∧ parabola_C x y ∧
  ∀ x' y', line_l x' y' b ∧ parabola_C x' y' → x = x' ∧ y = y'

-- Define the intersection points A and B
def intersection_points (b : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, 
    line_l x₁ y₁ b ∧ parabola_C x₁ y₁ ∧
    line_l x₂ y₂ b ∧ parabola_C x₂ y₂ ∧
    x₁ ≠ x₂

-- Define the area of triangle ABP
noncomputable def triangle_area (x₁ y₁ x₂ y₂ x y : ℝ) : ℝ :=
  (1/2) * abs ((x₁ - x) * (y₂ - y) - (x₂ - x) * (y₁ - y))

-- Theorem 1: Tangent line condition
theorem tangent_line_b : is_tangent (-4) := by sorry

-- Theorem 2: Maximum area of triangle ABP
theorem max_triangle_area :
  ∃ x₁ y₁ x₂ y₂, 
    intersection_points 1 ∧
    (∀ x y, parabola_C x y → 
      triangle_area x₁ y₁ x₂ y₂ x y ≤ 10 * Real.sqrt 5) ∧
    (∃ x y, parabola_C x y ∧ 
      triangle_area x₁ y₁ x₂ y₂ x y = 10 * Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_max_triangle_area_l601_60105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l601_60154

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => (3 * sequence_a n - 1) / (4 * sequence_a n + 7)

theorem sequence_a_general_term (n : ℕ) :
  sequence_a n = (9 - 4 * (n + 1)) / (2 + 8 * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l601_60154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l601_60126

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 1 then -3 * x + 4 else 4 * x - 6

-- Theorem statement
theorem g_values :
  g (-2) = 10 ∧ g 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l601_60126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_when_m_eq_2_no_containment_for_any_m_l601_60130

/-- Definition of circle C₁ -/
def C₁ (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x + 4*y + m^2 - 5 = 0

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x = 0

/-- The center of C₁ -/
def center_C₁ (m : ℝ) : ℝ × ℝ := (m, -2)

/-- The radius of C₁ -/
def radius_C₁ : ℝ := 3

/-- The center of C₂ -/
def center_C₂ : ℝ × ℝ := (-2, 0)

/-- The radius of C₂ -/
def radius_C₂ : ℝ := 2

/-- Distance between centers of C₁ and C₂ -/
noncomputable def distance_between_centers (m : ℝ) : ℝ :=
  Real.sqrt ((m + 2)^2 + 4)

/-- Theorem: When m = 2, circles C₁ and C₂ intersect -/
theorem circles_intersect_when_m_eq_2 :
  radius_C₁ - radius_C₂ < distance_between_centers 2 ∧
  distance_between_centers 2 < radius_C₁ + radius_C₂ := by
  sorry

/-- Theorem: There does not exist a real number m such that C₁ and C₂ are both contained within each other -/
theorem no_containment_for_any_m :
  ¬ ∃ m : ℝ, distance_between_centers m < |radius_C₁ - radius_C₂| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_when_m_eq_2_no_containment_for_any_m_l601_60130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_exists_l601_60177

/-- Represents a distribution of balls into boxes -/
structure BallDistribution where
  boxes : List (Nat × Nat)  -- List of (color, count) pairs
  student_count : Nat

/-- Checks if a distribution is valid according to the problem constraints -/
def is_valid_distribution (d : BallDistribution) : Prop :=
  d.boxes.all (λ b => b.2 ≥ 10) ∧
  (d.boxes.map (λ b => b.2)).sum = 800 ∧
  d.boxes.length ≤ 20 ∧
  d.student_count = 20

/-- Checks if a distribution results in equal balls for all students -/
def has_equal_distribution (d : BallDistribution) : Prop :=
  ∃ n : Nat, (d.boxes.map (λ b => b.2)).sum = n * d.student_count

/-- Main theorem: There exists a valid distribution with equal balls for all students -/
theorem ball_distribution_exists : 
  ∃ d : BallDistribution, is_valid_distribution d ∧ has_equal_distribution d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_exists_l601_60177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sqrt_equal_six_l601_60133

theorem factorial_sqrt_equal_six : Real.sqrt (3 * 2 * 1 * (3 * 2 * 1)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sqrt_equal_six_l601_60133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_l601_60187

open Real Set

-- Define the function
noncomputable def f (x : ℝ) := 2 * sin (π / 6 - 2 * x)

-- Define the domain
def domain : Set ℝ := Icc 0 π

-- Theorem statement
theorem f_increasing_intervals :
  ∀ x ∈ domain, 
    (x ∈ Icc 0 (π/3) ∨ x ∈ Icc (5*π/6) π) → 
    ∃ ε > 0, ∀ h ∈ Ioo 0 ε, x + h ∈ domain → f (x + h) > f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_l601_60187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_vote_ratio_l601_60123

theorem election_vote_ratio : 
  ∀ (eliot_votes shaun_votes randy_votes : ℕ),
    randy_votes = 16 →
    shaun_votes = 5 * randy_votes →
    eliot_votes = 160 →
    eliot_votes / shaun_votes = 2 :=
  fun eliot_votes shaun_votes randy_votes h1 h2 h3 =>
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_vote_ratio_l601_60123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skateboard_speed_theorem_l601_60167

/-- Calculates the speed in miles per hour given distance in feet and time in seconds -/
noncomputable def speed_mph (distance_feet : ℝ) (time_seconds : ℝ) : ℝ :=
  let feet_per_second := distance_feet / time_seconds
  let feet_per_mile := 5280
  let seconds_per_hour := 3600
  feet_per_second * (seconds_per_hour / feet_per_mile)

/-- Theorem stating that the speed calculated from 330 feet in 45 seconds is approximately 10.75 mph -/
theorem skateboard_speed_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |speed_mph 330 45 - 10.75| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skateboard_speed_theorem_l601_60167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_problem_l601_60150

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem ellipse_and_line_problem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ellipse a b 1 (3/2) ∧ eccentricity a b = 1/2 →
  (∃ (x y : ℝ), ellipse a b x y ∧
    (∃ (c : ℝ), triangle_area c 0 x y 1 (3/2) = 12 * Real.sqrt 2 / 7) →
    ((x - y + 1 = 0) ∨ (x + y + 1 = 0))) ∧
  (a^2 = 4 ∧ b^2 = 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_problem_l601_60150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l601_60146

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

/-- Main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 * Real.sin t.C = t.a * t.c * Real.cos t.B * Real.sin t.C + area t)
  (h2 : t.b * Real.sin t.C + t.c * Real.sin t.B = 6 * Real.sin t.B) :
  t.C = π/3 ∧ t.a + t.b + t.c ≤ 9 := by
  sorry

#check triangle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l601_60146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_value_l601_60128

/-- The sum of the infinite series Σ(n=0 to ∞) of (n+2) * (1/2000)^n -/
noncomputable def infinite_series_sum : ℝ := ∑' n, (n + 2) * (1 / 2000) ^ n

/-- The infinite series sum is equal to 8000000/3996001 -/
theorem infinite_series_sum_value : infinite_series_sum = 8000000 / 3996001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_value_l601_60128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_of_revolution_properties_l601_60173

/-- The volume of a solid of revolution formed by rotating an equilateral triangle -/
noncomputable def volume_of_revolution (a φ : ℝ) : ℝ :=
  (1/2) * Real.pi * a^2 * Real.sin (φ + Real.pi/6)

/-- The surface area of a solid of revolution formed by rotating an equilateral triangle -/
noncomputable def surface_area_of_revolution (a φ : ℝ) : ℝ :=
  2 * Real.pi * a^2 * Real.sqrt 3 * Real.sin (φ + Real.pi/6)

/-- Theorem stating the volume and surface area of the solid of revolution -/
theorem solid_of_revolution_properties (a φ : ℝ) (h_a : a > 0) :
  let V := volume_of_revolution a φ
  let S := surface_area_of_revolution a φ
  (V = (1/2) * Real.pi * a^2 * Real.sin (φ + Real.pi/6)) ∧
  (S = 2 * Real.pi * a^2 * Real.sqrt 3 * Real.sin (φ + Real.pi/6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_of_revolution_properties_l601_60173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_implies_a_one_f_increasing_l601_60107

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Theorem 1: If f is odd, then a = 1
theorem f_odd_implies_a_one (a : ℝ) :
  (∀ x, f a x = -f a (-x)) → a = 1 := by
  sorry

-- Theorem 2: f is increasing for any a
theorem f_increasing (a : ℝ) :
  ∀ x y, x < y → f a x < f a y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_implies_a_one_f_increasing_l601_60107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l601_60141

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6) - 1

theorem symmetry_center_of_g :
  ∃ (k : ℤ), g (Real.pi / 6 + k * Real.pi / 2) = g (Real.pi / 6 - k * Real.pi / 2) ∧
             g (Real.pi / 6) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l601_60141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_l601_60103

theorem geometric_series_ratio (n : ℕ) : 
  let S := Finset.sum (Finset.range n) (λ i => (2 : ℝ) ^ i)
  let ratio := (2 : ℝ) ^ n / S
  1 < ratio ∧ ratio < 1.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_l601_60103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_depth_is_eight_meters_l601_60104

/-- Represents a rectangular pond with given dimensions and volume -/
structure RectangularPond where
  length : ℝ
  width : ℝ
  volume : ℝ

/-- Calculates the depth of a rectangular pond -/
noncomputable def pondDepth (pond : RectangularPond) : ℝ :=
  pond.volume / (pond.length * pond.width)

/-- Theorem: The depth of the specified pond is 8 meters -/
theorem pond_depth_is_eight_meters :
  let pond : RectangularPond := {
    length := 20,
    width := 10,
    volume := 1600
  }
  pondDepth pond = 8 := by
  -- Unfold the definition of pondDepth
  unfold pondDepth
  -- Perform the calculation
  simp [RectangularPond.volume, RectangularPond.length, RectangularPond.width]
  -- The result should now be obvious
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_depth_is_eight_meters_l601_60104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_toy_probability_l601_60194

/-- Represents the toy vending machine -/
structure ToyMachine where
  numToys : Nat
  minCost : Rat
  costIncrement : Rat

/-- Represents Sam's initial state -/
structure SamInitialState where
  halfDollarCoins : Nat
  twentyDollarBills : Nat

/-- The probability problem for Sam buying his favorite toy -/
def samToyProblem (machine : ToyMachine) (samState : SamInitialState) (favoriteToyPrice : Rat) : Prop :=
  machine.numToys = 9 ∧
  machine.minCost = 1/2 ∧
  machine.costIncrement = 1/2 ∧
  samState.halfDollarCoins = 10 ∧
  samState.twentyDollarBills = 1 ∧
  favoriteToyPrice = 7/2 →
  (1 : Rat) - (8 * Nat.factorial 8 + Nat.factorial 7) / Nat.factorial 9 = 7/8

theorem sam_toy_probability (machine : ToyMachine) (samState : SamInitialState) (favoriteToyPrice : Rat) :
  samToyProblem machine samState favoriteToyPrice := by
  sorry

#check sam_toy_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_toy_probability_l601_60194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisibility_condition_l601_60112

theorem unique_divisibility_condition (k m : ℕ) :
  (∃! n : ℕ, n > 0 ∧ (n ^ m : ℕ) ∣ (5 ^ (n ^ k) + 1)) ↔ k + 2 ≤ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisibility_condition_l601_60112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_ratio_l601_60165

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

noncomputable def common_ratio (a : ℕ → ℝ) : ℝ :=
  a 2 / a 1

theorem arithmetic_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_incr : is_increasing_sequence a)
  (h_pos : a 1 > 0)
  (h_eq : ∀ n : ℕ, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  common_ratio a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_ratio_l601_60165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_good_condition_approx_88_94_l601_60193

/-- Represents the fruit inventory and quality data --/
structure FruitInventory where
  totalFruits : ℕ
  oranges : ℕ
  bananas : ℕ
  apples : ℕ
  pears : ℕ
  strawberries : ℕ
  kiwis : ℕ
  orangesRottenPercent : ℚ
  bananasRottenPercent : ℚ
  applesRottenPercent : ℚ
  pearsRottenPercent : ℚ
  strawberriesRottenPercent : ℚ
  kiwisRottenPercent : ℚ

/-- Calculates the percentage of fruits in good condition --/
def percentageGoodCondition (inventory : FruitInventory) : ℚ :=
  let totalRotten := (inventory.oranges * inventory.orangesRottenPercent +
                      inventory.bananas * inventory.bananasRottenPercent +
                      inventory.apples * inventory.applesRottenPercent +
                      inventory.pears * inventory.pearsRottenPercent +
                      inventory.strawberries * inventory.strawberriesRottenPercent +
                      inventory.kiwis * inventory.kiwisRottenPercent) / 100
  let goodCondition := inventory.totalFruits - totalRotten.floor
  (goodCondition * 100) / inventory.totalFruits

/-- Theorem stating that the percentage of fruits in good condition is approximately 88.94% --/
theorem percentage_good_condition_approx_88_94 (inventory : FruitInventory)
  (h1 : inventory.totalFruits = 2450)
  (h2 : inventory.oranges = 600)
  (h3 : inventory.bananas = 500)
  (h4 : inventory.apples = 450)
  (h5 : inventory.pears = 400)
  (h6 : inventory.strawberries = 300)
  (h7 : inventory.kiwis = 200)
  (h8 : inventory.orangesRottenPercent = 14)
  (h9 : inventory.bananasRottenPercent = 8)
  (h10 : inventory.applesRottenPercent = 10)
  (h11 : inventory.pearsRottenPercent = 11)
  (h12 : inventory.strawberriesRottenPercent = 16)
  (h13 : inventory.kiwisRottenPercent = 5) :
  ∃ ε > 0, |percentageGoodCondition inventory - 8894/100| < ε ∧ ε < 1/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_good_condition_approx_88_94_l601_60193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_x_coordinate_l601_60195

/-- The area of a rhombus given its diagonals -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- The coordinates of the rhombus vertices -/
def rhombusVertices (x : ℝ) : List (ℝ × ℝ) := [(0, 3.5), (x, 0), (0, -3.5), (-8, 0)]

/-- Theorem: For a rhombus with given vertices and area, the x-coordinate is 8 -/
theorem rhombus_x_coordinate (x : ℝ) : 
  rhombusArea 7 (x + 8) = 56 → x = 8 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_x_coordinate_l601_60195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l601_60196

-- Define the function f(x) = x^2e^x
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

-- Define the domain of f
def domain : Set ℝ := Set.Icc (-1) 1

-- Theorem statement
theorem f_monotone_increasing :
  ∀ x ∈ domain, ∀ y ∈ domain, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → x < y → f x < f y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l601_60196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_approx_l601_60101

/-- The time for two trains to cross each other when traveling in opposite directions -/
noncomputable def time_to_cross (length_A length_B : ℝ) (time_A time_B : ℝ) : ℝ :=
  let speed_A := length_A / time_A
  let speed_B := length_B / time_B
  let relative_speed := speed_A + speed_B
  let combined_length := length_A + length_B
  combined_length / relative_speed

/-- Theorem stating that the time for the given trains to cross each other is approximately 11.43 seconds -/
theorem trains_crossing_time_approx :
  ∃ ε > 0, |time_to_cross 150 90 10 15 - 11.43| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_approx_l601_60101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_beta_l601_60145

theorem smallest_beta : ∃ (α : ℕ), (16 : ℚ)/37 < (α : ℚ)/23 ∧ (α : ℚ)/23 < 7/16 ∧
  ∀ (β : ℕ) (γ : ℕ), 0 < β → β < 23 → ¬((16 : ℚ)/37 < (γ : ℚ)/β ∧ (γ : ℚ)/β < 7/16) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_beta_l601_60145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_theorem_l601_60108

theorem max_distance_theorem (a b c : ℝ × ℝ) 
  (h1 : ‖a‖ = 4)
  (h2 : ‖b‖ = 2 * Real.sqrt 2)
  (h3 : a • b = π / 4)
  (h4 : (c - a) • (c - b) = -1) :
  (‖c - a‖ ≤ Real.sqrt 2 + 1) ∧ (∃ c' : ℝ × ℝ, ‖c' - a‖ = Real.sqrt 2 + 1 ∧ (c' - a) • (c' - b) = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_theorem_l601_60108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_owner_speed_approx_l601_60125

/-- Calculates the speed of the owner's bike given the thief's speed, head start time, and total time until overtaking. -/
noncomputable def ownerSpeed (thiefSpeed : ℝ) (headStartTime : ℝ) (totalTime : ℝ) : ℝ :=
  let chaseTime := totalTime - headStartTime
  let initialDistance := thiefSpeed * headStartTime
  let additionalDistance := thiefSpeed * chaseTime
  let totalDistance := initialDistance + additionalDistance
  totalDistance / chaseTime

/-- Theorem stating that given the problem conditions, the owner's speed is approximately 51.43 kmph. -/
theorem owner_speed_approx (thiefSpeed : ℝ) (headStartTime : ℝ) (totalTime : ℝ)
    (h1 : thiefSpeed = 45)
    (h2 : headStartTime = 0.5)
    (h3 : totalTime = 4) :
    ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |ownerSpeed thiefSpeed headStartTime totalTime - 51.43| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_owner_speed_approx_l601_60125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l601_60160

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |Real.log (x + a)|

-- State the theorem
theorem range_of_a_for_increasing_f :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f a x₁ ≤ f a x₂) →
  a ∈ Set.Ici 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l601_60160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l601_60148

theorem right_triangle_hypotenuse (a b : ℝ) 
  (h_right : a > 0 ∧ b > 0)
  (h_median1 : Real.sqrt (b^2 + (a/2)^2) = 6)
  (h_median2 : Real.sqrt (a^2 + (b/2)^2) = Real.sqrt 27) :
  Real.sqrt ((2*a)^2 + (2*b)^2) = 2 * Real.sqrt 25.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l601_60148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_fixed_point_l601_60172

/-- Definition of the ellipse C -/
def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of a point on the ellipse -/
def point_on_ellipse (a b : ℝ) : Prop :=
  ellipse 1 (3/2) a b

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Definition of the line l -/
def line (x y k m : ℝ) : Prop :=
  y = k * x + m

/-- Definition of the right vertex D -/
def right_vertex (a : ℝ) : ℝ × ℝ :=
  (a, 0)

/-- Definition of the dot product of vectors DA and DB being zero -/
def perpendicular_condition (xa ya xb yb a : ℝ) : Prop :=
  (xa - a) * (xb - a) + ya * yb = 0

/-- The main theorem -/
theorem ellipse_intersection_fixed_point 
  (a b k m : ℝ) 
  (h1 : point_on_ellipse a b) 
  (h2 : eccentricity a b = 1/2) 
  (h3 : ∃ (xa ya xb yb : ℝ), 
    ellipse xa ya a b ∧ 
    ellipse xb yb a b ∧ 
    line xa ya k m ∧ 
    line xb yb k m ∧ 
    perpendicular_condition xa ya xb yb a) :
  line (2/7) 0 k m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_fixed_point_l601_60172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_mixed_segments_l601_60199

/-- A type representing a point on the number line that can be either rational or irrational -/
inductive Point
| rational (q : ℚ)
| irrational (x : ℝ)

/-- A function that determines if a point is rational -/
def is_rational (p : Point) : Bool :=
  match p with
  | Point.rational _ => true
  | Point.irrational _ => false

/-- A function that determines if two adjacent points form a segment with one rational and one irrational endpoint -/
def is_mixed_segment (p1 p2 : Point) : Bool :=
  (is_rational p1 && !is_rational p2) || (!is_rational p1 && is_rational p2)

theorem odd_mixed_segments (n : ℕ) (points : List Point) : 
  points.length = n + 2 →
  points.head? = some (Point.rational 1) →
  points.getLast? = some (Point.irrational (Real.sqrt 2)) →
  (∀ p ∈ points, ∃ x : ℝ, (x > 1 ∧ x < Real.sqrt 2) ∧ 
    (is_rational p = true → ∃ q : ℚ, x = q) ∧
    (is_rational p = false → ∀ q : ℚ, x ≠ q)) →
  Odd (List.countP (λ (pair : Point × Point) => is_mixed_segment pair.1 pair.2 = true) (points.zip (points.tail))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_mixed_segments_l601_60199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_shaded_areas_different_l601_60120

/-- Represents a square with a specific partition and shading pattern -/
structure PartitionedSquare where
  totalArea : ℝ
  shadedArea : ℝ

/-- Square I with diagonal partition and corner shading -/
noncomputable def squareI : PartitionedSquare where
  totalArea := 1
  shadedArea := 1/4

/-- Square II with midpoint partition and central shading -/
noncomputable def squareII : PartitionedSquare where
  totalArea := 1
  shadedArea := 1/4

/-- Square III with diagonal and midpoint line, trapezoid shading -/
noncomputable def squareIII : PartitionedSquare where
  totalArea := 1
  shadedArea := 3/8

/-- Theorem stating that the shaded areas of all three squares are different -/
theorem all_shaded_areas_different :
  squareI.shadedArea ≠ squareII.shadedArea ∧
  squareI.shadedArea ≠ squareIII.shadedArea ∧
  squareII.shadedArea ≠ squareIII.shadedArea := by
  sorry

#check all_shaded_areas_different

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_shaded_areas_different_l601_60120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_impossibility_l601_60156

/-- A triangle is represented as a set of points in ℝ × ℝ -/
def Triangle : Type := Set (ℝ × ℝ)

theorem triangle_division_impossibility : ∃ t : Triangle, ¬ (∃ (parts : Finset (Set (ℝ × ℝ))) (square : Set (ℝ × ℝ)), 
  (parts.card = 1000) ∧ 
  (∀ p ∈ parts, p ⊆ t) ∧
  (parts.sup id = square) ∧
  (∃ s : ℝ, square = {(x, y) | 0 ≤ x ∧ x ≤ s ∧ 0 ≤ y ∧ y ≤ s})) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_impossibility_l601_60156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_range_l601_60171

noncomputable def circle_center : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)
noncomputable def circle_radius : ℝ := Real.sqrt 3

noncomputable def line_param (α t : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, 2 + t * Real.sin α)

noncomputable def chord_length (α : ℝ) : ℝ := 2 * Real.sqrt (2 + Real.sin (2 * α))

theorem chord_length_range :
  ∀ α : ℝ, 0 ≤ α ∧ α ≤ Real.pi / 4 →
  2 * Real.sqrt 2 ≤ chord_length α ∧ chord_length α < 2 * Real.sqrt 3 :=
by sorry

#check chord_length_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_range_l601_60171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parakeets_in_pet_store_l601_60152

/-- Given a number of cages and a total number of birds, calculate the number of parakeets per cage -/
def parakeets_per_cage (num_cages : ℕ) (total_birds : ℕ) : ℕ :=
  (total_birds / num_cages) / 2

/-- Represents the number of birds in a cage -/
structure Cage where
  parakeets : ℕ
  parrots : ℕ

theorem parakeets_in_pet_store (num_cages : ℕ) (total_birds : ℕ) 
  (h1 : num_cages = 9)
  (h2 : total_birds = 36)
  (h3 : ∀ cage : Cage, cage.parakeets = cage.parrots) :
  parakeets_per_cage num_cages total_birds = 2 := by
  sorry

#eval parakeets_per_cage 9 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parakeets_in_pet_store_l601_60152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_tangent_circles_l601_60109

-- Define the types for points and lines
variable (Point : Type) (Line : Type)

-- Define the necessary geometric relations
variable (collinear : Point → Point → Point → Prop)
variable (orthocenter : Point → Point → Point → Point → Prop)
variable (on_line : Point → Line → Prop)
variable (intersection : Line → Line → Point)
variable (perpendicular : Line → Line → Prop)
variable (equal_length : Point → Point → Point → Point → Prop)
variable (circle : Point → Point → Type)
variable (circumcircle : Point → Point → Point → Type)
variable (tangent : Type → Type → Prop)

-- Define the given points and lines
variable (A B C H E F X Y T : Point)
variable (AB AC BC EF EH FH HT : Line)

-- State the theorem
theorem orthocenter_tangent_circles 
  (h1 : orthocenter H A B C)
  (h2 : on_line E AB)
  (h3 : on_line F AC)
  (h4 : equal_length B E B H)
  (h5 : equal_length C F C H)
  (h6 : X = intersection EH BC)
  (h7 : Y = intersection FH BC)
  (h8 : perpendicular HT EF)
  (h9 : on_line T EF)
  (h10 : on_line T HT) :
  tangent (circumcircle T X Y) (circle B C) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_tangent_circles_l601_60109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l601_60118

noncomputable def f (x : ℝ) := 2 * Real.sin x - 3 * x

theorem range_of_a (a : ℝ) : 
  (∀ m ∈ Set.Icc (-2 : ℝ) 2, f (m * a - 3) + f (a^2) > 0) ↔ a ∈ Set.Ioo (-1 : ℝ) 1 :=
sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l601_60118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_post_height_l601_60110

/-- Represents the properties of a cylindrical post and a squirrel's path on it -/
structure SpiralPath where
  post_circumference : ℚ
  total_distance : ℚ
  rise_per_circuit : ℚ

/-- Calculates the height of the post based on the spiral path properties -/
def post_height (path : SpiralPath) : ℚ :=
  (path.total_distance / path.post_circumference) * path.rise_per_circuit

/-- Theorem stating that for the given conditions, the post height is 16 feet -/
theorem squirrel_post_height :
  let path : SpiralPath := {
    post_circumference := 2,
    total_distance := 8,
    rise_per_circuit := 4
  }
  post_height path = 16 := by
  -- Unfold the definition of post_height
  unfold post_height
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_post_height_l601_60110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_parametric_equation_l601_60102

/-- The parametric equation of the trajectory of the midpoint of a line segment
    from the origin to a point on a circle. -/
theorem midpoint_trajectory_parametric_equation :
  ∃ P Q : ℝ → ℝ × ℝ, 
    (∀ θ : ℝ, (P θ).1 = 6 + 2 * Real.cos θ ∧ (P θ).2 = 4 + 2 * Real.sin θ) ∧
    (∀ θ : ℝ, (Q θ).1 = ((P θ).1) / 2 ∧ (Q θ).2 = ((P θ).2) / 2) ∧
    (∀ θ : ℝ, (Q θ).1 = 3 + Real.cos θ ∧ (Q θ).2 = 2 + Real.sin θ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_parametric_equation_l601_60102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_points_l601_60129

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the diameter endpoints
def DiameterEndpoints (center : ℝ × ℝ) (radius : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  ((center.1 - radius, center.2), (center.1 + radius, center.2))

-- Define the condition for sum of squared distances
def SumSquaredDistances (p : ℝ × ℝ) (a b : ℝ × ℝ) : ℝ :=
  (p.1 - a.1)^2 + (p.2 - a.2)^2 + (p.1 - b.1)^2 + (p.2 - b.2)^2

-- Define the set of points satisfying all conditions
def ValidPoints (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | p ∈ Circle center (Real.sqrt 2) ∧
       let (a, b) := DiameterEndpoints center (Real.sqrt 2)
       SumSquaredDistances p a b = 4 ∧
       p.2 > center.2}

-- Theorem statement
theorem infinitely_many_valid_points (center : ℝ × ℝ) : 
  Set.Infinite (ValidPoints center) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_points_l601_60129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l601_60151

theorem inequality_proof (a b c : ℝ) : 
  a = (3/5)^(-(1/3 : ℝ)) → b = (3/5)^(-(1/2 : ℝ)) → c = (4/3)^(-(1/2 : ℝ)) → c < a ∧ a < b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l601_60151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_a_and_b_l601_60159

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + 2*a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + 2*a = 0) :
  a + b ≥ Real.sqrt (3 * (24 : ℝ)^(1/3)) + (24 : ℝ)^(1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_a_and_b_l601_60159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l601_60166

noncomputable def f (x : ℝ) : ℝ := x / (x^2 - x + 2)

theorem range_of_f :
  Set.range f = Set.Icc (-1/7 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l601_60166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l601_60111

theorem min_value_trig_expression (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (Real.sin x + Real.tan x)^2 + (Real.cos x + (1 / Real.tan x))^2 ≥ 5 ∧
  ((Real.sin x + Real.tan x)^2 + (Real.cos x + (1 / Real.tan x))^2 = 5 ↔ x = π / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l601_60111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximation_l601_60181

/-- Calculates the simple interest rate given the principal, time, and total amount -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (total_amount : ℝ) : ℝ :=
  ((total_amount - principal) * 100) / (principal * time)

theorem interest_rate_approximation :
  let principal : ℝ := 5396.103896103896
  let time : ℝ := 9
  let total_amount : ℝ := 8310
  let calculated_rate := calculate_interest_rate principal time total_amount
  ∃ ε > 0, |calculated_rate - 6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximation_l601_60181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_intensity_at_10m_l601_60135

/-- The intensity of light at a certain distance from a light source -/
noncomputable def intensity (k : ℝ) (d : ℝ) : ℝ := k / d^2

theorem light_intensity_at_10m 
  (h1 : intensity 5000 5 = 200) 
  (h2 : ∀ (k d : ℝ), intensity k d = k / d^2) :
  intensity 5000 10 = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_intensity_at_10m_l601_60135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_finish_time_l601_60182

/-- The time (in days) it takes for workers a, b, and c to finish a job together -/
noncomputable def time_abc : ℝ := 6

/-- The time (in days) it takes for worker c to finish the job alone -/
noncomputable def time_c : ℝ := 10

/-- The work rate (jobs per day) of workers a and b combined -/
noncomputable def rate_ab : ℝ := 1 / time_abc - 1 / time_c

/-- The time (in days) it takes for workers a and b to finish the job together -/
noncomputable def time_ab : ℝ := 1 / rate_ab

theorem ab_finish_time :
  time_ab = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_finish_time_l601_60182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_not_90_degrees_l601_60122

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop := m * x + y - 1 = 0

-- Define the slope angle of a line
noncomputable def slope_angle (m : ℝ) : ℝ := Real.arctan (-m)

-- Theorem: The slope angle of the line mx + y - 1 = 0 cannot be 90°
theorem slope_angle_not_90_degrees (m : ℝ) : slope_angle m ≠ Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_not_90_degrees_l601_60122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_rotation_surface_area_l601_60155

/-- The surface area of a solid formed by rotating an equilateral triangle around one of its sides -/
theorem equilateral_triangle_rotation_surface_area (a : ℝ) (h : a > 0) :
  (2 : ℝ) * π * ((Real.sqrt 3 / 2) * a) * a = Real.sqrt 3 * π * a^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_rotation_surface_area_l601_60155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l601_60188

/-- A circle C1 passes through the point (0,2) and is tangent to another circle C2 
    defined by the equation x^2 + (y - 3)^2 = 5 at the point (1,3). 
    This theorem states that the center of C1 is (2,3). -/
theorem circle_center (C1 C2 : Set (ℝ × ℝ)) : 
  (∃ (center : ℝ × ℝ) (radius : ℝ), 
    C1 = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}) →
  (0, 2) ∈ C1 →
  C2 = {p : ℝ × ℝ | p.1^2 + (p.2 - 3)^2 = 5} →
  (∃ (t : ℝ × ℝ), t ∈ C1 ∧ t ∈ C2 ∧ 
    (∀ p ∈ C1, p ≠ t → (∃ ε > 0, Set.inter (Metric.ball t ε) (Set.inter C1 C2) = {t}))) →
  (1, 3) ∈ C1 →
  (1, 3) ∈ C2 →
  (∃ (center : ℝ × ℝ) (radius : ℝ), 
    C1 = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2} ∧
    center = (2, 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l601_60188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_implies_k_leq_one_l601_60116

/-- The proposition that for all x in [0, π/2], e^x * sin(x) ≥ k*x is true -/
def proposition (k : ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi/2 → Real.exp x * Real.sin x ≥ k * x

/-- If the proposition is true, then k ≤ 1 -/
theorem proposition_implies_k_leq_one (k : ℝ) : proposition k → k ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_implies_k_leq_one_l601_60116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_equation_of_l_when_OP_eq_OM_l601_60184

-- Define the circle C
noncomputable def C (x y : ℝ) : Prop := x^2 + y^2 - 8*x = 0

-- Define point P
def P : ℝ × ℝ := (2, 2)

-- Define the moving line l passing through P
noncomputable def l (m : ℝ) (x y : ℝ) : Prop := y - P.2 = m * (x - P.1)

-- Define the midpoint M of segment AB
noncomputable def M (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem for the trajectory of point M
theorem trajectory_of_M (x y : ℝ) :
  (∃ m A B, l m A.1 A.2 ∧ l m B.1 B.2 ∧ C A.1 A.2 ∧ C B.1 B.2 ∧ (x, y) = M A B) →
  (x - 3)^2 + (y - 1)^2 = 2 := by
  sorry

-- Theorem for the equation of line l when |OP| = |OM|
theorem equation_of_l_when_OP_eq_OM (x y : ℝ) :
  (∃ m A B, l m A.1 A.2 ∧ l m B.1 B.2 ∧ C A.1 A.2 ∧ C B.1 B.2 ∧
   let M := M A B
   (O.1 - P.1)^2 + (O.2 - P.2)^2 = (O.1 - M.1)^2 + (O.2 - M.2)^2) →
  3*x + y - 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_equation_of_l_when_OP_eq_OM_l601_60184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tuna_cost_is_two_l601_60119

/-- The cost of a pack of tuna given the following conditions:
  * 5 packs of tuna were bought
  * 4 bottles of water were bought
  * Each bottle of water costs $1.5
  * The total cost of shopping is $56
  * $40 was spent on different goods
-/
noncomputable def tuna_cost : ℝ :=
  let num_tuna : ℕ := 5
  let num_water : ℕ := 4
  let water_cost : ℝ := 1.5
  let total_cost : ℝ := 56
  let other_goods_cost : ℝ := 40
  (total_cost - other_goods_cost - num_water * water_cost) / num_tuna

theorem tuna_cost_is_two : tuna_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tuna_cost_is_two_l601_60119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_relation_l601_60192

/-- The number of positive divisors of a positive integer -/
def d (n : ℕ) : ℕ := sorry

/-- Theorem: For any positive integer k, there exists a positive integer n 
    such that d(n^2) = k · d(n) if and only if k is odd -/
theorem divisor_count_relation (k : ℕ) :
  (∃ n : ℕ, n > 0 ∧ d (n^2) = k * d n) ↔ k % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_relation_l601_60192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_approximation_l601_60191

/-- The distance between two observers in miles -/
def distance_between_observers : ℝ := 15

/-- The angle of elevation for the first observer in radians -/
noncomputable def angle1 : ℝ := 40 * Real.pi / 180

/-- The angle of elevation for the second observer in radians -/
noncomputable def angle2 : ℝ := 45 * Real.pi / 180

/-- The altitude of the airplane in miles -/
noncomputable def altitude : ℝ := distance_between_observers * (1 / Real.tan angle1)

theorem airplane_altitude_approximation :
  abs (altitude - 17) < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_approximation_l601_60191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_picked_last_year_l601_60183

def roses_last_year : ℕ → Prop := λ r => True

def roses_this_year (r : ℕ) : ℕ := r / 2

def roses_needed (r : ℕ) : ℕ := 2 * r

def rose_price : ℕ := 3

def money_spent : ℕ := 54

theorem roses_picked_last_year (r : ℕ) : 
  roses_last_year r →
  roses_needed r - roses_this_year r = money_spent / rose_price →
  r = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_picked_last_year_l601_60183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l601_60139

noncomputable def h (x : ℝ) : ℝ := 3 / (1 + 9 * x^2)

theorem h_range : Set.range h = Set.Ioo 0 3 ∪ {3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l601_60139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_sum_is_530_div_17_l601_60117

/-- The line equation forming a triangle with the x and y axes -/
def line_equation (x y : ℝ) : Prop := 15 * x + 8 * y = 120

/-- The triangle formed by the line and the axes -/
structure Triangle where
  a : ℝ  -- x-intercept
  b : ℝ  -- y-intercept
  h : line_equation a 0 ∧ line_equation 0 b

/-- The sum of the altitudes of the triangle -/
noncomputable def altitude_sum (t : Triangle) : ℝ :=
  t.a + t.b + (2 * (t.a * t.b) / Real.sqrt ((t.a ^ 2) + (t.b ^ 2)))

/-- Theorem: The sum of the altitudes of the triangle is 530/17 -/
theorem altitude_sum_is_530_div_17 (t : Triangle) : altitude_sum t = 530 / 17 := by
  sorry

#eval "Theorem stated and proof skipped with sorry"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_sum_is_530_div_17_l601_60117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_ratio_eq_one_minus_sqrt_two_over_eight_l601_60137

/-- Represents a rectangular sheet of paper. -/
structure Paper where
  width : ℝ
  length : ℝ
  area : ℝ
  length_eq_twice_width : length = 2 * width
  area_eq_length_times_width : area = length * width

/-- Represents the folded paper. -/
structure FoldedPaper where
  original : Paper
  new_area : ℝ

/-- The ratio of the new area to the original area after folding. -/
noncomputable def foldRatio (p : FoldedPaper) : ℝ :=
  p.new_area / p.original.area

/-- Theorem stating the ratio of areas after folding. -/
theorem fold_ratio_eq_one_minus_sqrt_two_over_eight (p : FoldedPaper) :
  foldRatio p = 1 - Real.sqrt 2 / 8 := by
  sorry

#check fold_ratio_eq_one_minus_sqrt_two_over_eight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_ratio_eq_one_minus_sqrt_two_over_eight_l601_60137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_x_l601_60132

theorem tan_pi_4_plus_x (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.cos x = 4/5) : 
  Real.tan (π/4 + x) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_x_l601_60132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_value_l601_60124

theorem cos_double_angle_special_value (α : ℝ) 
  (h1 : Real.cos (α - π/3) = 2/3) 
  (h2 : 0 < α ∧ α < π/2) : 
  Real.cos (2*α - 2*π/3) = -1/9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_value_l601_60124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_q_l601_60170

noncomputable section

/-- A cubic polynomial Q(x) = x^3 + px^2 + qx + r -/
def Q (p q r : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + r

/-- The mean of the zeros of Q(x) -/
def meanZeros (p : ℝ) : ℝ := -p / 3

/-- The product of the zeros of Q(x) -/
def productZeros (r : ℝ) : ℝ := r

/-- The sum of the coefficients of Q(x) -/
def sumCoefficients (p q r : ℝ) : ℝ := 1 + p + q + r

/-- The y-intercept of Q(x) -/
def yIntercept (r : ℝ) : ℝ := r

theorem cubic_polynomial_q (p q r : ℝ) :
  meanZeros p = productZeros r ∧
  productZeros r = sumCoefficients p q r ∧
  yIntercept r = 3 →
  q = 5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_q_l601_60170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_bounds_l601_60131

open Real

noncomputable def g (x : ℝ) : ℝ := sin (2 * x - π / 3)

theorem g_range_bounds (a b : ℝ) (h₁ : a < b) (h₂ : ∀ x ∈ Set.Icc a b, -1/2 ≤ g x ∧ g x ≤ 1) :
  π / 3 ≤ b - a ∧ b - a ≤ 2 * π / 3 := by
  sorry

#check g_range_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_bounds_l601_60131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_road_fuel_collection_l601_60189

/-- Represents a car on the circular road -/
structure Car where
  position : ℝ  -- Position on the circular road
  fuel : ℝ      -- Amount of fuel in the car

/-- Represents the circular road with cars -/
structure CircularRoad where
  length : ℝ    -- Length of the circular road
  cars : List Car

/-- Theorem: If the total fuel is sufficient for one car to complete the road,
    then there exists a car that can complete the road by collecting fuel. -/
theorem circular_road_fuel_collection (road : CircularRoad) :
  (road.cars.map Car.fuel).sum ≥ road.length →
  ∃ (start_car : Car), start_car ∈ road.cars ∧
    ∃ (route : List Car), route ≠ [] ∧
                          route.head? = some start_car ∧
                          (route.reverse).head? = some start_car ∧
                          (route.map Car.fuel).sum ≥ road.length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_road_fuel_collection_l601_60189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l601_60176

/-- The area of a rhombus with vertices at (0, 7.5), (8, 0), (0, -7.5), and (-8, 0) is 120 square units. -/
theorem rhombus_area : 
  ∀ (x1 y1 x2 y2 : ℝ),
  x1 = 0 ∧ y1 = 7.5 ∧ 
  x2 = 8 ∧ y2 = 0 →
  let d1 := |y1 - (-y1)|
  let d2 := |x2 - (-x2)|
  (d1 * d2) / 2 = 120 := by
  sorry

#check rhombus_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l601_60176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_relation_l601_60153

theorem cubic_roots_relation (a b c r s t : ℝ) : 
  (∀ x, x^3 + 3*x^2 + 4*x + 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (∀ x, x^3 + r*x^2 + s*x + t = 0 ↔ (x = a + b ∨ x = b + c ∨ x = c + a)) →
  t = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_relation_l601_60153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l601_60100

noncomputable def a : ℝ × ℝ × ℝ := (3, -4, -Real.sqrt 11)

noncomputable def b (x y : ℝ) : ℝ × ℝ × ℝ := (Real.sin (2*x) * Real.cos y, Real.cos (2*x) * Real.cos y, Real.sin y)

noncomputable def f (x y : ℝ) : ℝ := 
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b x y
  a₁ * b₁ + a₂ * b₂ + a₃ * b₃

theorem f_range : 
  (∀ x y : ℝ, (f x y) ≥ -6 ∧ (f x y) ≤ 6) ∧ 
  (∃ x₁ y₁ : ℝ, f x₁ y₁ = -6) ∧ 
  (∃ x₂ y₂ : ℝ, f x₂ y₂ = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l601_60100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_is_minus_sqrt_two_plus_three_l601_60190

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x + 3 else a * x + b

-- State the theorem
theorem sum_of_a_and_b_is_minus_sqrt_two_plus_three (a b : ℝ) :
  (∀ x₁ : ℝ, x₁ ≠ 0 → ∃! x₂ : ℝ, x₂ ≠ 0 ∧ x₂ ≠ x₁ ∧ f a b x₁ = f a b x₂) →
  f a b (Real.sqrt 3 * a) = f a b (4 * b) →
  a + b = -Real.sqrt 2 + 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_is_minus_sqrt_two_plus_three_l601_60190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_170_l601_60179

/-- The length of a train given its speed, platform length, and time to cross the platform. -/
noncomputable def train_length (train_speed_kmph : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (5/18)
  let total_distance := train_speed_mps * crossing_time
  total_distance - platform_length

/-- Theorem stating that the length of the train is 170 meters under given conditions. -/
theorem train_length_is_170 :
  train_length 72 350 26 = 170 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_length 72 350 26

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_170_l601_60179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_c_on_b_l601_60134

noncomputable def a : ℝ × ℝ := (2, 3)
noncomputable def b : ℝ × ℝ := (-4, 7)

noncomputable def projection (u v : ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2) / Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem projection_c_on_b :
  ∀ c : ℝ × ℝ, a.1 + c.1 = 0 ∧ a.2 + c.2 = 0 →
  projection c b = -Real.sqrt 65 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_c_on_b_l601_60134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l601_60136

/-- The length of a train given specific conditions --/
theorem train_length_problem (v1 v2 t L2 : ℝ) (h1 : v1 = 100) (h2 : v2 = 60) (h3 : t = 18) (h4 : L2 = 300) : 
  (v1 + v2) * (5/18) * t - L2 = 420 := by
  -- Define relative speed
  let v_rel := (v1 + v2) * (5/18)
  -- Define length of first train
  let L1 := v_rel * t - L2
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l601_60136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_point_coplanar_l601_60138

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of a tetrahedron -/
structure Tetrahedron where
  O : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Check if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Vector from one point to another -/
def vectorBetween (p1 p2 : Point3D) : Vector3D := sorry

/-- Scalar multiplication of a vector -/
def scalarMult (s : ℝ) (v : Vector3D) : Vector3D := sorry

/-- Addition of vectors -/
def vectorAdd (v1 v2 : Vector3D) : Vector3D := sorry

theorem tetrahedron_point_coplanar (tetra : Tetrahedron) (P : Point3D) (l : ℝ) :
  vectorBetween tetra.O P = 
    vectorAdd 
      (vectorAdd 
        (scalarMult (1/4) (vectorBetween tetra.O tetra.A))
        (scalarMult l (vectorBetween tetra.O tetra.B)))
      (scalarMult (1/6) (vectorBetween tetra.O tetra.C)) →
  areCoplanar P tetra.A tetra.B tetra.C →
  l = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_point_coplanar_l601_60138
