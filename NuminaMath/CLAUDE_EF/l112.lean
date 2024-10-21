import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l112_11271

theorem exponential_inequality (a b c : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 3/2) : 
  (3 : ℝ)^a + (9 : ℝ)^b + (27 : ℝ)^c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l112_11271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l112_11228

/-- Given a principal amount and an interest rate, calculates the simple interest for a given time period. -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Given a principal amount and an interest rate, calculates the compound interest for a given time period. -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Theorem stating that if the simple interest for 2 years is 50 and the compound interest for 2 years is 51.25,
    then the annual interest rate is 5%. -/
theorem interest_rate_is_five_percent (P : ℝ) (r : ℝ) 
    (h1 : simple_interest P r 2 = 50)
    (h2 : compound_interest P r 2 = 51.25) : 
    r = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l112_11228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_sum_l112_11245

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals for computability)
  d : ℚ      -- Common difference
  h1 : ∀ n, a (n + 1) = a n + d  -- Definition of arithmetic sequence
  h2 : a 11 / a 10 < -1  -- Given condition
  h3 : ∃ k : ℕ, ∀ n : ℕ, (n : ℚ) * (a 1 + a n) / 2 ≤ (k : ℚ) * (a 1 + a k) / 2  -- S_n has a maximum value

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The theorem to be proved -/
theorem smallest_positive_sum (seq : ArithmeticSequence) :
  ∃ (m : ℕ), S seq 19 > 0 ∧ (∀ n, S seq n > 0 → S seq 19 ≤ S seq n) ∧ m = 19 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_sum_l112_11245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s₁_less_than_s₂_l112_11293

/-- A triangle in a 2D plane. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle. -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- Distance between two points. -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Sum of distances from centroid to vertices. -/
noncomputable def s₁ (t : Triangle) : ℝ :=
  let O := centroid t
  distance O t.A + distance O t.B + distance O t.C

/-- Three times the perimeter of the triangle. -/
noncomputable def s₂ (t : Triangle) : ℝ :=
  3 * (distance t.A t.B + distance t.B t.C + distance t.C t.A)

/-- Theorem stating that s₁ < s₂ for any triangle. -/
theorem s₁_less_than_s₂ (t : Triangle) : s₁ t < s₂ t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s₁_less_than_s₂_l112_11293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x1_greater_f_x2_l112_11208

-- Define the function f and its second derivative
variable (f : ℝ → ℝ)
variable (f'' : ℝ → ℝ)

-- Define the conditions
axiom symmetric : ∀ x : ℝ, f (1 + (1 - x)) = f x
axiom second_derivative : ∀ x : ℝ, (x - 1) * f'' x < 0

-- State the theorem to be proved
theorem f_x1_greater_f_x2 (x₁ x₂ : ℝ) (h1 : x₁ < x₂) (h2 : x₁ + x₂ > 2) :
  f x₁ > f x₂ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x1_greater_f_x2_l112_11208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_negative_2023_opposite_of_negative_2023_is_2023_l112_11276

theorem opposite_of_negative_2023 : (-2023 : ℝ) + 2023 = 0 := by
  ring

#check opposite_of_negative_2023

def opposite (x : ℝ) : ℝ := -x

theorem opposite_of_negative_2023_is_2023 : opposite (-2023 : ℝ) = 2023 := by
  rw [opposite]
  ring

#check opposite_of_negative_2023_is_2023

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_negative_2023_opposite_of_negative_2023_is_2023_l112_11276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P₀_volume_volume_increase_volume_P₃_l112_11254

/-- Represents a sequence of polyhedra -/
structure Polyhedron where
  volume : ℚ

/-- Constructs the next polyhedron in the sequence -/
def next_polyhedron (p : Polyhedron) : Polyhedron :=
  { volume := p.volume + (2 / 9) * (p.volume - 1) }

/-- The initial tetrahedron P₀ -/
def P₀ : Polyhedron :=
  { volume := 1 }

/-- The sequence of polyhedra -/
def P : ℕ → Polyhedron
  | 0 => P₀
  | n + 1 => next_polyhedron (P n)

/-- The volume of P₀ is 1 -/
theorem P₀_volume : P₀.volume = 1 :=
  rfl

/-- The volume increase at each step -/
theorem volume_increase (n : ℕ) : 
  (P (n + 1)).volume - (P n).volume = (2 / 9) * ((P n).volume - (P (n - 1)).volume) :=
  sorry

/-- The main theorem: the volume of P₃ is 22615/6561 -/
theorem volume_P₃ : (P 3).volume = 22615 / 6561 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P₀_volume_volume_increase_volume_P₃_l112_11254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_w_div_n_eq_one_over_e_l112_11299

/-- The expected number of white squares remaining after the process described in the problem. -/
noncomputable def w : ℕ → ℝ := sorry

/-- The limit of w(N)/N as N approaches infinity is 1/e. -/
theorem limit_w_div_n_eq_one_over_e :
  ∀ ε > 0, ∃ N₀ : ℕ, ∀ N ≥ N₀, |w N / N - 1 / Real.exp 1| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_w_div_n_eq_one_over_e_l112_11299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_sqrt_sum_infinite_l112_11278

theorem rational_sqrt_sum_infinite : 
  ∃ S : Set ℚ, (Set.Infinite S) ∧ 
  (∀ x ∈ S, ∃ r : ℚ, (Real.sqrt (x - 1) + Real.sqrt (4 * x + 1) : ℝ) = r) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_sqrt_sum_infinite_l112_11278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_difference_is_ten_percent_l112_11213

-- Define the exchange rate
noncomputable def euro_to_dollar : ℚ := 6/5

-- Define Diana's and Etienne's money
noncomputable def diana_dollars : ℚ := 600
noncomputable def etienne_euros : ℚ := 450

-- Convert Etienne's euros to dollars
noncomputable def etienne_dollars : ℚ := etienne_euros * euro_to_dollar

-- Calculate the percentage difference
noncomputable def percentage_difference : ℚ := (diana_dollars - etienne_dollars) / diana_dollars * 100

-- Theorem statement
theorem money_difference_is_ten_percent :
  percentage_difference = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_difference_is_ten_percent_l112_11213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_in_rectangle_percentage_l112_11229

/-- The percentage of a rectangle's area covered by a square, given specific proportions -/
theorem square_in_rectangle_percentage : 
  ∀ (s : ℝ), s > 0 →
  (s^2 / (3*s * 9*s)) * 100 = 100 / 27 := by
  intro s hs
  field_simp
  ring

#eval (100 : ℚ) / 27

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_in_rectangle_percentage_l112_11229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_number_choices_l112_11201

/-- Theorem about consecutive number choices -/
theorem consecutive_number_choices : ℕ := by
  -- Define the total number of elements
  let n : ℕ := 49

  -- Define the number of elements to choose
  let k : ℕ := 6

  -- Define the number of ways to choose k numbers from n numbers
  let total_choices : ℕ := Nat.choose n k

  -- Define the number of ways to choose k numbers from (n - k + 1) numbers
  let non_consecutive_choices : ℕ := Nat.choose (n - k + 1) k

  -- The number of ways to choose k numbers with at least two consecutive numbers
  let consecutive_choices : ℕ := total_choices - non_consecutive_choices

  -- Theorem statement
  have : consecutive_choices = 6924764 := by sorry

  exact 6924764

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_number_choices_l112_11201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_a_greater_than_one_l112_11285

noncomputable section

variable (f : ℝ → ℝ)

-- f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- f is monotonically increasing on [0, +∞)
axiom f_increasing_nonneg : ∀ x y, 0 ≤ x → x < y → f x < f y

-- f is monotonically increasing on ℝ
theorem f_increasing : ∀ x y, x < y → f x < f y := by
  sorry

-- The main theorem
theorem a_greater_than_one (a : ℝ) (h : f a < f (2*a - 1)) : a > 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_a_greater_than_one_l112_11285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l112_11263

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

-- State the theorem
theorem max_value_of_f :
  ∃ a : ℝ, (∀ x : ℝ, f x ≤ a) ∧ (∃ x : ℝ, f x = a) ∧ a = 2 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l112_11263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relationship_l112_11283

theorem sin_cos_relationship (α β : Real) :
  (Real.sin α + Real.cos β = 0) → (Real.sin α)^2 + (Real.sin β)^2 = 1 ∧
  ¬ ((Real.sin α)^2 + (Real.sin β)^2 = 1 → Real.sin α + Real.cos β = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relationship_l112_11283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_transformed_data_l112_11237

noncomputable def data_set := Fin 3 → ℝ

noncomputable def average (s : data_set) : ℝ := (s 0 + s 1 + s 2) / 3

noncomputable def variance (s : data_set) : ℝ := 
  ((s 0 - average s)^2 + (s 1 - average s)^2 + (s 2 - average s)^2) / 3

def transform (s : data_set) : data_set :=
  fun i => 3 * s i - 2

theorem variance_of_transformed_data (s : data_set) 
  (h_avg : average s = 4) 
  (h_var : variance s = 3) : 
  variance (transform s) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_transformed_data_l112_11237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_groups_count_l112_11257

/-- Represents a group of children in the photography club --/
structure ChildrenGroup where
  boys : ℕ
  girls : ℕ

/-- Represents the photography session --/
structure PhotographySession where
  groups : List ChildrenGroup
  group_count : groups.length = 100
  total_children : List.sum (List.map (λ g => g.boys + g.girls) groups) = 300

theorem mixed_groups_count (session : PhotographySession) 
  (h1 : List.sum (List.map (λ g => g.boys * (g.boys - 1)) session.groups) = 100)
  (h2 : List.sum (List.map (λ g => g.girls * (g.girls - 1)) session.groups) = 56) :
  List.sum (List.map (λ g => if g.boys > 0 ∧ g.girls > 0 then 1 else 0) session.groups) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_groups_count_l112_11257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_correct_l112_11280

noncomputable section

/-- A parabola with equation y = 4x^2 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 4 * p.1^2}

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, 1/16)

/-- The directrix of the parabola -/
def directrix : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -1/16}

/-- Distance squared between two points -/
def distanceSquared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Distance squared from a point to a line y = k -/
def distanceToLineSquared (p : ℝ × ℝ) (k : ℝ) : ℝ :=
  (p.2 - k)^2

theorem parabola_focus_correct :
  ∀ p ∈ Parabola,
    distanceSquared p focus = distanceToLineSquared p (-1/16) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_correct_l112_11280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_C_l112_11274

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real  -- Area of the triangle

/-- The triangle satisfies the given conditions -/
class TriangleConditions (t : Triangle) where
  angle_sum : t.A + t.B + t.C = Real.pi
  area_formula : Real.sin (t.A + t.C) = (2 * t.S) / (t.b^2 - t.c^2)
  arithmetic_sequence : t.B = (t.A + t.C) / 2

theorem triangle_angle_C (t : Triangle) [TriangleConditions t] : t.C = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_C_l112_11274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_distinct_and_expression_integer_l112_11288

/-- The polynomial x^3 - x^2 - x - 1 -/
def f (x : ℝ) : ℝ := x^3 - x^2 - x - 1

/-- The roots of the polynomial f -/
noncomputable def roots : Set ℝ := {x | f x = 0}

/-- The expression to be proven as an integer -/
noncomputable def expression (a b c : ℝ) : ℝ :=
  (a^1982 - b^1982) / (a - b) + (b^1982 - c^1982) / (b - c) + (c^1982 - a^1982) / (c - a)

theorem roots_distinct_and_expression_integer :
  ∀ a b c, a ∈ roots → b ∈ roots → c ∈ roots →
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ ∃ n : ℤ, expression a b c = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_distinct_and_expression_integer_l112_11288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_pairs_equation_l112_11272

theorem positive_integer_pairs_equation :
  ∀ x y : ℕ+, x + y + x * y = 2006 ↔ 
    (x.val, y.val) ∈ ({(2, 668), (668, 2), (8, 222), (222, 8)} : Set (ℕ × ℕ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_pairs_equation_l112_11272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_exists_unique_l112_11286

/-- Triangle with orthocenter and incenter -/
structure TriangleWithCenters where
  A : ℝ × ℝ  -- Vertex A of the triangle
  B : ℝ × ℝ  -- Vertex B of the triangle
  C : ℝ × ℝ  -- Vertex C of the triangle
  M : ℝ × ℝ  -- Orthocenter
  O : ℝ × ℝ  -- Incenter
  ρ : ℝ      -- Incircle radius

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: A unique triangle exists given AM, AO, and ρ -/
theorem triangle_exists_unique (am ao ρ : ℝ) (h_pos : am > 0 ∧ ao > 0 ∧ ρ > 0) :
  ∃! t : TriangleWithCenters, 
    distance t.A t.M = am ∧ 
    distance t.A t.O = ao ∧ 
    t.ρ = ρ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_exists_unique_l112_11286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_rotation_path_length_l112_11244

/-- Represents a triangle with side lengths -/
structure Triangle where
  sideAB : ℝ
  sideAP : ℝ
  sideBP : ℝ

/-- Represents a square with side length -/
structure Square where
  sideLength : ℝ

/-- Calculates the total path length of vertex P in the rotation process -/
noncomputable def totalPathLength (t : Triangle) (s : Square) : ℝ :=
  4 * (2 * Real.pi / 3) * t.sideBP

theorem isosceles_triangle_rotation_path_length 
  (t : Triangle) 
  (s : Square) 
  (h1 : t.sideAB = 3) 
  (h2 : t.sideAP = 3) 
  (h3 : t.sideBP = 4) 
  (h4 : s.sideLength = 8) :
  totalPathLength t s = 32 * Real.pi / 3 := by
  sorry

#check isosceles_triangle_rotation_path_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_rotation_path_length_l112_11244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l112_11241

theorem negation_of_universal_proposition :
  (¬∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l112_11241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l112_11226

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℝ) 
  (ha : arithmetic_sequence a) 
  (hb : arithmetic_sequence b) 
  (h_ratio : ∀ n : ℕ, sum_of_arithmetic_sequence a n / sum_of_arithmetic_sequence b n = (2 * n + 1) / (3 * n + 2)) :
  (a 3 + a 11 + a 19) / (b 7 + b 15) = 129 / 130 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l112_11226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_B_faster_l112_11262

-- Define the routes
structure Route where
  total_distance : ℝ
  normal_speed : ℝ
  slow_distance : ℝ
  slow_speed : ℝ

-- Define the two routes
def route_A : Route :=
  { total_distance := 8
  , normal_speed := 40
  , slow_distance := 1
  , slow_speed := 10 }

def route_B : Route :=
  { total_distance := 7
  , normal_speed := 50
  , slow_distance := 1
  , slow_speed := 25 }

-- Function to calculate travel time for a route
noncomputable def travel_time (r : Route) : ℝ :=
  let normal_distance := r.total_distance - r.slow_distance
  (normal_distance / r.normal_speed + r.slow_distance / r.slow_speed) * 60

-- Theorem statement
theorem route_B_faster : 
  travel_time route_A - travel_time route_B = 6.9 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_B_faster_l112_11262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_inverse_uniqueness_l112_11236

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

-- State the theorems to be proved
theorem f_properties :
  (f 2 = 4) ∧
  (f (1/2) = 1/4) ∧
  (f (f (-1)) = 1) ∧
  (∃ a₁ a₂ : ℝ, f a₁ = 3 ∧ f a₂ = 3 ∧ a₁ = 1 ∧ a₂ = Real.sqrt 3) := by
  sorry

-- Additional theorem for the uniqueness of solutions
theorem f_inverse_uniqueness :
  ∀ a : ℝ, f a = 3 → (a = 1 ∨ a = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_inverse_uniqueness_l112_11236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l112_11296

theorem trigonometric_equation_solution (x : ℝ) :
  Real.sin x ^ 4 + Real.sin (2 * x) ^ 4 + Real.sin (3 * x) ^ 4 = 
  Real.cos x ^ 4 + Real.cos (2 * x) ^ 4 + Real.cos (3 * x) ^ 4 →
  ∃ k : ℤ, x = (2 * k + 1) * Real.pi / 8 ∨ x = (2 * k + 1) * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l112_11296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oscillating_sequence_golden_ratio_l112_11258

/-- Definition of a p-oscillating sequence -/
def IsOscillatingSequence (x : ℕ → ℝ) (p : ℝ) : Prop :=
  ∀ n : ℕ, (x (n + 1) - p) * (x n - p) < 0

/-- Definition of the sequence c_n -/
noncomputable def c : ℕ → ℝ
  | 0 => 1  -- Add case for 0
  | 1 => 1
  | n + 2 => 1 / (c (n + 1) + 1)

/-- Theorem: If c is a p-oscillating sequence, then p = (√5 - 1) / 2 -/
theorem oscillating_sequence_golden_ratio :
  ∃ p, IsOscillatingSequence c p → p = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oscillating_sequence_golden_ratio_l112_11258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_to_initial_probability_probability_after_2020_steps_l112_11211

/-- Represents the state of money distribution among players -/
def MoneyState := Vector ℕ 4

/-- The initial state where each player has $1 -/
def initialState : MoneyState := Vector.cons 1 (Vector.cons 1 (Vector.cons 1 (Vector.cons 1 Vector.nil)))

/-- The set of all possible transitions from one state to another -/
def allTransitions : Finset (MoneyState → MoneyState) := sorry

/-- The probability of transitioning from one state to another -/
def transitionProbability (s t : MoneyState) : ℚ := sorry

/-- Theorem: The probability of returning to the initial state after one step is 24/81 -/
theorem return_to_initial_probability :
  transitionProbability initialState initialState = 24 / 81 := by sorry

/-- The probability of being in the initial state after n steps -/
def probabilityAfterNSteps (n : ℕ) : ℚ := sorry

/-- Theorem: The probability of being in the initial state after 2020 steps is (24/81)^2020 -/
theorem probability_after_2020_steps :
  probabilityAfterNSteps 2020 = (24 / 81) ^ 2020 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_to_initial_probability_probability_after_2020_steps_l112_11211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_behind_windy_l112_11227

/-- Represents the race conditions and results -/
structure RaceData where
  race_distance : ℝ
  first_race_lead : ℝ
  second_race_handicap : ℝ
  windy_speed_reduction : ℝ

/-- Calculates the distance Sunny finishes behind Windy in the second race -/
noncomputable def second_race_result (data : RaceData) : ℝ :=
  let sunny_speed_ratio := (data.race_distance) / (data.race_distance - data.first_race_lead)
  let windy_second_race_distance := data.windy_speed_reduction * sunny_speed_ratio * (data.race_distance + data.second_race_handicap)
  windy_second_race_distance - (data.race_distance + data.second_race_handicap)

/-- Theorem stating that Sunny finishes between 14 and 15 meters behind Windy in the second race -/
theorem sunny_behind_windy (data : RaceData) 
  (h1 : data.race_distance = 120)
  (h2 : data.first_race_lead = 18)
  (h3 : data.second_race_handicap = 18)
  (h4 : data.windy_speed_reduction = 0.96) :
  14 < second_race_result data ∧ second_race_result data < 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_behind_windy_l112_11227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_parabola_l112_11219

/-- The value of k for which the line 4x + 7y + k = 0 is tangent to the parabola y² = 16x -/
def tangent_k : ℝ := 49

/-- The line equation -/
def line (x y k : ℝ) : Prop := 4 * x + 7 * y + k = 0

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

theorem line_tangent_to_parabola :
  ∃! k, ∀ x y, line x y k ∧ parabola x y → 
    (∃! (p : ℝ × ℝ), line p.1 p.2 k ∧ parabola p.1 p.2) ∧
    k = tangent_k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_parabola_l112_11219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l112_11223

/-- Square ABCD with side length s -/
structure SquareABCD where
  s : ℝ
  s_pos : s > 0

/-- Square EFGH inscribed in ABCD -/
structure SquareEFGH (ABCD : SquareABCD) where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  on_sides : E.1 ∈ Set.Icc 0 ABCD.s ∧ 
             F.2 ∈ Set.Icc 0 ABCD.s ∧ 
             G.1 ∈ Set.Icc 0 ABCD.s ∧ 
             H.2 ∈ Set.Icc 0 ABCD.s
  E_on_AB : E.2 = 0 ∧ E.1 = 3 * (ABCD.s - E.1)

/-- Function to calculate the area of a square given its side length -/
def area (side : ℝ) : ℝ := side * side

/-- The ratio of areas theorem -/
theorem area_ratio (ABCD : SquareABCD) (EFGH : SquareEFGH ABCD) :
  (area (Real.sqrt 2 * (ABCD.s / 4))) / (area ABCD.s) = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l112_11223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l112_11269

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)

/-- The theorem stating that the only function satisfying the equation is f(x) = x - 1 -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → ∀ x : ℝ, f x = x - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l112_11269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_consumption_relation_l112_11235

/-- Represents the remaining fuel in liters -/
def y : ℝ → ℝ := sorry

/-- Represents the driving time in hours -/
def t : ℝ := sorry

/-- The initial amount of fuel in liters -/
def initial_fuel : ℝ := 40

/-- The fuel consumption rate in liters per hour -/
def consumption_rate : ℝ := 5

/-- Theorem stating that y = 40 - 5t correctly describes the relationship
    between remaining fuel and driving time -/
theorem fuel_consumption_relation :
  y t = initial_fuel - consumption_rate * t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_consumption_relation_l112_11235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l112_11210

/-- The radius of the inscribed circle in a triangle with side lengths a, b, and c. -/
noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

theorem inscribed_circle_radius_specific_triangle :
  inscribedCircleRadius 26 16 18 = 2 * Real.sqrt 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l112_11210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l112_11290

theorem divisor_problem (n : ℕ+) (h1 : Nat.card (Nat.divisors n) = 72) 
  (h2 : Nat.card (Nat.divisors (5 * n)) = 90) :
  ∃ k : ℕ, k = 3 ∧ 5^k ∣ n ∧ ∀ m : ℕ, 5^m ∣ n → m ≤ k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l112_11290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l112_11265

noncomputable def z : ℂ := -2 * (Complex.exp (Complex.I * (2016 * Real.pi / 180)))

theorem z_in_fourth_quadrant : 
  z.re < 0 ∧ z.im > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l112_11265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_fourth_quadrant_l112_11247

theorem sin_plus_cos_fourth_quadrant (α : ℝ) 
  (h1 : Real.sin (2 * α) = -Real.sqrt 2 / 2) 
  (h2 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) : 
  Real.sin α + Real.cos α = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_fourth_quadrant_l112_11247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_ratio_theorem_l112_11248

/-- Represents a right circular cone -/
structure Cone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- The ratio of height to radius of the cone -/
noncomputable def heightRadiusRatio (cone : Cone) : ℝ := cone.h / cone.r

/-- Condition for the bead's path after 10 rotations -/
def beadPathCondition (cone : Cone) : Prop :=
  2 * Real.pi * Real.sqrt (cone.r^2 + cone.h^2) = 20 * Real.pi * cone.r

theorem cone_ratio_theorem (cone : Cone) :
  beadPathCondition cone →
  heightRadiusRatio cone = 3 * Real.sqrt 11 := by
  sorry

#check cone_ratio_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_ratio_theorem_l112_11248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_parallel_line_through_point_l112_11291

-- Define the necessary structures
structure Point where

structure Line where

structure Plane where

-- Define the relationships
def parallel_line_plane (l : Line) (a : Plane) : Prop := sorry

def point_in_plane (A : Point) (a : Plane) : Prop := sorry

def line_in_plane (l : Line) (a : Plane) : Prop := sorry

def line_through_point (l : Line) (A : Point) : Prop := sorry

def parallel_lines (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem unique_parallel_line_through_point 
  (l : Line) (a : Plane) (A : Point)
  (h1 : parallel_line_plane l a)
  (h2 : point_in_plane A a) :
  ∃! m : Line, line_in_plane m a ∧ line_through_point m A ∧ parallel_lines m l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_parallel_line_through_point_l112_11291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increased_flow_increases_time_l112_11260

/-- Represents the round trip time for a ferry journey -/
noncomputable def roundTripTime (distance : ℝ) (ferrySpeed : ℝ) (waterFlowSpeed : ℝ) : ℝ :=
  (2 * distance * ferrySpeed) / (ferrySpeed^2 - waterFlowSpeed^2)

/-- Proves that increasing water flow speed increases round trip time -/
theorem increased_flow_increases_time 
  (distance : ℝ) (ferrySpeed : ℝ) (initialFlow : ℝ) (increasedFlow : ℝ) 
  (h1 : distance > 0) 
  (h2 : ferrySpeed > 0) 
  (h3 : 0 ≤ initialFlow) 
  (h4 : initialFlow < ferrySpeed) 
  (h5 : initialFlow < increasedFlow) 
  (h6 : increasedFlow < ferrySpeed) :
  roundTripTime distance ferrySpeed initialFlow < roundTripTime distance ferrySpeed increasedFlow :=
by
  sorry

#check increased_flow_increases_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increased_flow_increases_time_l112_11260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_pairs_l112_11230

theorem infinite_coprime_pairs (a b : ℤ) (h : a ≠ b) :
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ n, Int.gcd (a + f n) (b + f n) = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_pairs_l112_11230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l112_11205

/-- The circle with center (1, 1) and radius 1 -/
def Circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

/-- The line x - y = 2 -/
def Line (x y : ℝ) : Prop := x - y = 2

/-- The distance from a point (x, y) to the line x - y = 2 -/
noncomputable def distanceToLine (x y : ℝ) : ℝ := |x - y - 2| / Real.sqrt 2

/-- The maximum distance from a point on the circle to the line -/
noncomputable def maxDistance : ℝ := 1 + Real.sqrt 2

theorem max_distance_circle_to_line :
  ∀ x y : ℝ, Circle x y → distanceToLine x y ≤ maxDistance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l112_11205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_l112_11255

/-- Converts meters per second to kilometers per hour -/
noncomputable def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3600 / 1000

theorem speed_conversion (speed_mps : ℝ) :
  speed_mps = 22 → mps_to_kmph speed_mps = 79.2 := by
  intro h
  rw [mps_to_kmph, h]
  norm_num
  -- The proof is completed automatically by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_l112_11255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_percentage_calculation_l112_11256

/-- Calculates the percentage of stock sold given the cash realized, brokerage rate, and total amount including brokerage -/
noncomputable def calculate_stock_percentage (cash_realized : ℝ) (brokerage_rate : ℝ) (total_amount : ℝ) : ℝ :=
  let total_before_brokerage := total_amount / (1 - brokerage_rate / 100)
  let brokerage_fee := total_before_brokerage - cash_realized
  (brokerage_fee / total_before_brokerage) * (100 / brokerage_rate)

/-- Theorem stating that given the specified conditions, the percentage of stock sold is approximately 7.39% -/
theorem stock_percentage_calculation :
  let cash_realized : ℝ := 108.25
  let brokerage_rate : ℝ := 0.25
  let total_amount : ℝ := 108
  let result := calculate_stock_percentage cash_realized brokerage_rate total_amount
  abs (result - 7.39) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_percentage_calculation_l112_11256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l112_11246

theorem polynomial_division_theorem :
  let f : Polynomial ℚ := X^6 - 4*X^5 + 5*X^4 - 27*X^3 + 13*X^2 - 16*X + 12
  let g : Polynomial ℚ := X - 3
  let q : Polynomial ℚ := X^5 - X^4 + 2*X^3 - 21*X^2 - 50*X - 166
  let r : Polynomial ℚ := -486
  f = g * q + r := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l112_11246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_y1_value_l112_11289

/-- Represents a quadrilateral in a 2D coordinate system -/
structure Quadrilateral where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ
  x3 : ℝ
  y3 : ℝ
  x4 : ℝ
  y4 : ℝ

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (h b1 b2 : ℝ) : ℝ :=
  (1/2) * h * (b1 + b2)

/-- Main theorem: If a quadrilateral with the given coordinates has an area of 76, then y1 = -3 -/
theorem quadrilateral_y1_value (q : Quadrilateral) 
    (h1 : q.x1 = 4 ∧ q.x2 = 4 ∧ q.x3 = 12 ∧ q.x4 = 12)
    (h2 : q.y2 = 7 ∧ q.y3 = 2 ∧ q.y4 = -7)
    (h3 : trapezoidArea (q.x3 - q.x1) (q.y2 - q.y1) (q.y3 - q.y4) = 76) :
  q.y1 = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_y1_value_l112_11289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_norm_condition_smallest_norm_v_l112_11279

open Real

def v : ℝ × ℝ := sorry

-- Define the given condition
theorem norm_condition : ‖v + (4, 2)‖ = 10 := sorry

-- Theorem to prove
theorem smallest_norm_v :
  ∀ w : ℝ × ℝ, ‖w + (4, 2)‖ = 10 → ‖v‖ ≤ ‖w‖ ∧ ‖v‖ = 10 - 2 * sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_norm_condition_smallest_norm_v_l112_11279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_did_good_deed_l112_11212

-- Define the individuals
inductive Person : Type
  | A
  | B
  | C

-- Define the person who did the good deed
variable (goodDeedDoer : Person)

-- Define the statements made by each person
def statement (p : Person) : Prop :=
  match p with
  | Person.A => goodDeedDoer = Person.B
  | Person.B => goodDeedDoer ≠ Person.B
  | Person.C => goodDeedDoer ≠ Person.C

-- Only one person did the good deed
axiom unique_doer : ∀ (p q : Person), goodDeedDoer = p → goodDeedDoer = q → p = q

-- Only one statement is true
axiom one_true_statement : ∃! (p : Person), statement goodDeedDoer p

-- Theorem: C did the good deed
theorem c_did_good_deed : goodDeedDoer = Person.C := by
  sorry

#check c_did_good_deed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_did_good_deed_l112_11212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l112_11209

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - 1 + Real.sin x ^ 2

/-- The transformed function g(x) -/
noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi/8)

/-- Theorem stating that g(x) is an even function -/
theorem g_is_even : ∀ x : ℝ, g x = g (-x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l112_11209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_minus_cot_l112_11204

noncomputable def f (x : ℝ) : ℝ := Real.tan x - (1 / Real.tan x)

theorem period_of_tan_minus_cot :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_minus_cot_l112_11204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_line_l112_11234

/-- The chord length of a circle intercepted by a line --/
theorem chord_length_circle_line : 
  ∃ (p q : ℝ × ℝ), 
    ((p.1 - 1)^2 + p.2^2 = 1) ∧
    (p.2 = p.1) ∧
    ((q.1 - 1)^2 + q.2^2 = 1) ∧
    (q.2 = q.1) ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_line_l112_11234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_euclid_field_theorem_l112_11220

/-- Represents a right triangle with a square in one corner --/
structure FieldTriangle where
  leg1 : ℝ
  leg2 : ℝ
  square_distance : ℝ

/-- Calculates the fraction of the triangle not covered by the square --/
noncomputable def planted_fraction (field : FieldTriangle) : ℝ :=
  let triangle_area := (field.leg1 * field.leg2) / 2
  let square_side := field.square_distance
  let square_area := square_side * square_side
  let planted_area := triangle_area - square_area
  planted_area / triangle_area

/-- Theorem statement for the given problem --/
theorem farmer_euclid_field_theorem (field : FieldTriangle) 
  (h1 : field.leg1 = 5)
  (h2 : field.leg2 = 12)
  (h3 : field.square_distance = 3) :
  planted_fraction field = 7/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_euclid_field_theorem_l112_11220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kolakoski_property_l112_11251

/-- Kolakoski sequence -/
def kolakoski : ℕ → ℕ := sorry

/-- Sum of the first n terms of the Kolakoski sequence -/
def kolakoski_sum (n : ℕ) : ℕ := (Finset.range n).sum (λ i => kolakoski i)

theorem kolakoski_property :
  kolakoski_sum 50 = 75 ∧
  kolakoski 49 = 2 →
  kolakoski 72 = 1 ∧
  kolakoski 73 = 2 ∧
  kolakoski 74 = 2 ∧
  kolakoski 75 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kolakoski_property_l112_11251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l112_11264

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (16 - 4^x)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iic 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l112_11264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_triangle_l112_11295

/-- A convex n-gon in a unit square -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)
  inside_unit_square : ∀ i, vertices i ∈ (Set.Icc 0 1 : Set ℝ) ×ˢ (Set.Icc 0 1 : Set ℝ)

/-- The area of a triangle formed by three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem: There exists a triangle with area less than 80/n^3 -/
theorem exists_small_triangle {n : ℕ} (K : ConvexNGon n) :
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    triangle_area (K.vertices i) (K.vertices j) (K.vertices k) < 80 / n^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_triangle_l112_11295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cold_medicine_box_properties_l112_11232

/-- Represents a box of cold medicine granules -/
structure ColdMedicineBox where
  packets : ℕ
  acetaminophenPerPacket : ℚ

/-- Calculates the total grams of acetaminophen in a box -/
def totalAcetaminophen (box : ColdMedicineBox) : ℚ :=
  box.packets * box.acetaminophenPerPacket

/-- Calculates the grams of acetaminophen in half a packet -/
def halfPacketAcetaminophen (box : ColdMedicineBox) : ℚ :=
  box.acetaminophenPerPacket / 2

/-- Theorem stating the properties of the cold medicine box -/
theorem cold_medicine_box_properties (box : ColdMedicineBox) 
    (h1 : box.packets = 9)
    (h2 : box.acetaminophenPerPacket = 1/5) : 
    totalAcetaminophen box = 9/5 ∧ halfPacketAcetaminophen box = 1/10 := by
  sorry

#eval (9 : ℚ) * (1/5 : ℚ)
#eval (1/5 : ℚ) / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cold_medicine_box_properties_l112_11232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_l112_11253

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) + sequence_a (n + 1) / (2^(n + 1))

theorem sequence_a_bounds (n : ℕ) (h : n ≥ 3) :
  2 - 1 / (2^(n-1)) < sequence_a n ∧ sequence_a n < (3/2)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_bounds_l112_11253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_skew_angle_in_regular_tetrahedron_l112_11240

/-- A regular tetrahedron is a tetrahedron with all edges equal -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- The angle between a face median and a skew edge in a regular tetrahedron -/
noncomputable def median_skew_angle (t : RegularTetrahedron) : ℝ :=
  Real.arccos (Real.sqrt 3 / 6)

/-- Theorem: In a regular tetrahedron, the angle between a face median and a skew edge is arccos(√3/6) -/
theorem median_skew_angle_in_regular_tetrahedron (t : RegularTetrahedron) :
  median_skew_angle t = Real.arccos (Real.sqrt 3 / 6) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_skew_angle_in_regular_tetrahedron_l112_11240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_pi_symmetry_point_shift_symmetry_f_properties_l112_11217

-- Define the function f as noncomputable due to use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x)^2

-- Statement 1: Minimum positive period is π
theorem min_period_pi (x : ℝ) : f (x + π) = f x := by sorry

-- Statement 2: Symmetry about (-π/12, -√3/2)
theorem symmetry_point (x : ℝ) : 
  f (-π/12 + x) + f (-π/12 - x) = -Real.sqrt 3 := by sorry

-- Statement 3: Shifting by 2π/3 creates symmetry about y-axis
theorem shift_symmetry (x : ℝ) : 
  f (x + 2*π/3) = f (-x + 2*π/3) := by sorry

-- Main theorem combining all statements
theorem f_properties : 
  (∀ x, f (x + π) = f x) ∧ 
  (∀ x, f (-π/12 + x) + f (-π/12 - x) = -Real.sqrt 3) ∧
  (∀ x, f (x + 2*π/3) = f (-x + 2*π/3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_pi_symmetry_point_shift_symmetry_f_properties_l112_11217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_questions_minimum_l112_11242

/-- The number of people taking the test -/
def n : ℕ := 10

/-- The size of the group that can answer all questions -/
def k : ℕ := 5

/-- The size of the group that cannot answer all questions -/
def m : ℕ := 4

/-- The minimum number of questions in the test -/
def min_questions : ℕ := Nat.choose n m

/-- Represents whether a person can answer a question -/
def answers (p : Fin n) (q : ℕ) : Prop := sorry

theorem test_questions_minimum (people : Finset (Fin n)) :
  (∀ (group : Finset (Fin n)), group.card = k → 
    ∃ (questions : Finset ℕ), questions.card = min_questions ∧ 
      (∀ q ∈ questions, ∃ p ∈ group, answers p q)) ∧
  (∀ (group : Finset (Fin n)), group.card = m → 
    ∀ (questions : Finset ℕ), questions.card = min_questions → 
      ∃ q ∈ questions, ∀ p ∈ group, ¬(answers p q)) :=
by sorry

#eval min_questions -- Should output 210

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_questions_minimum_l112_11242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_l112_11218

theorem cosine_difference (α β : ℝ) : 
  0 < α → 0 < β → α + β < π → 
  Real.cos α = 1/3 → Real.cos (α + β) = -1/3 → 
  Real.cos (α - β) = 23/27 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_l112_11218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l112_11273

/-- Calculates the length of a train given its speed and time to pass a point. -/
noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600) * time_s

/-- Theorem stating that a train moving at 72 km/h passing a point in 5 seconds is 100 meters long. -/
theorem train_length_calculation :
  train_length 72 5 = 100 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l112_11273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_63_rearrangements_l112_11224

def is_multiple_of_63 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 63 * k

def is_rearrangement (a b : ℕ) : Prop :=
  ∃ perm : List.Perm (Nat.digits 10 a) (Nat.digits 10 b), True

theorem smallest_number_with_63_rearrangements : 
  (∀ n : ℕ, n < 111888 → ¬(∀ m : ℕ, is_rearrangement n m → is_multiple_of_63 m)) ∧
  (∀ m : ℕ, is_rearrangement 111888 m → is_multiple_of_63 m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_63_rearrangements_l112_11224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_with_adjacent_pair_l112_11238

/-- The number of ways to arrange n people around a round table -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways two people can sit next to each other -/
def adjacentArrangements : ℕ := 2

theorem seating_arrangements_with_adjacent_pair (n : ℕ) (h : n = 10) :
  circularArrangements (n - 1) * adjacentArrangements = 80640 := by
  sorry

#eval circularArrangements 9 * adjacentArrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_with_adjacent_pair_l112_11238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_l112_11222

theorem inverse_proposition :
  (∀ x : ℝ, x^2 > 0 → x < 0) ↔ (∀ x : ℝ, x < 0 → x^2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_l112_11222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_reaches_one_l112_11207

def sequence_a (h : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => if sequence_a h n % 2 = 0 then sequence_a h n / 2 else sequence_a h n + h

theorem sequence_reaches_one (h : ℕ) :
  (∃ n : ℕ, n > 0 ∧ sequence_a h n = 1) ↔ h % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_reaches_one_l112_11207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_chord_length_l112_11287

/-- Given a parabola y² = 4x with focal chord endpoints A(x₁, y₁) and B(x₂, y₂) perpendicular to the x-axis, 
    the length of AB is 4. -/
theorem parabola_focal_chord_length (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 4 * x₁ → -- Endpoint A satisfies parabola equation
  y₂^2 = 4 * x₂ → -- Endpoint B satisfies parabola equation
  x₁ = x₂ →      -- AB is perpendicular to x-axis
  |y₁ - y₂| = 4  -- Length of AB
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_chord_length_l112_11287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_volume_in_cylinder_l112_11292

noncomputable section

def cylinder_radius : ℝ := 10
def cylinder_height : ℝ := 30
def cone_radius : ℝ := 10
def cone_height : ℝ := 10
def num_cones : ℕ := 3

noncomputable def cylinder_volume : ℝ := Real.pi * cylinder_radius^2 * cylinder_height
noncomputable def cone_volume : ℝ := (1/3) * Real.pi * cone_radius^2 * cone_height
noncomputable def total_cone_volume : ℝ := num_cones * cone_volume

theorem empty_volume_in_cylinder :
  cylinder_volume - total_cone_volume = 2000 * Real.pi := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_volume_in_cylinder_l112_11292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_triangle_leg_length_l112_11268

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  leg : ℝ
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = leg * Real.sqrt 2

/-- The main isosceles right triangle -/
noncomputable def main_triangle : IsoscelesRightTriangle :=
  { leg := Real.sqrt 2
    hypotenuse := 2
    hypotenuse_eq := by sorry }

/-- The area of an isosceles right triangle -/
noncomputable def area (t : IsoscelesRightTriangle) : ℝ := t.leg^2 / 2

/-- The smaller isosceles right triangle -/
noncomputable def small_triangle : IsoscelesRightTriangle :=
  { leg := Real.sqrt (2/3)
    hypotenuse := Real.sqrt (4/3)
    hypotenuse_eq := by sorry }

theorem small_triangle_leg_length :
  (area main_triangle = 3 * area small_triangle) →
  small_triangle.leg = Real.sqrt (2/3) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_triangle_leg_length_l112_11268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_difference_l112_11203

theorem cube_surface_area_difference :
  ∀ (large_cube_volume small_cube_volume num_small_cubes : ℝ),
    large_cube_volume = 125 →
    small_cube_volume = 1 →
    num_small_cubes = 125 →
    let large_cube_side := large_cube_volume ^ (1/3)
    let small_cube_side := small_cube_volume ^ (1/3)
    let large_cube_surface_area := 6 * large_cube_side^2
    let small_cube_surface_area := 6 * small_cube_side^2
    let total_small_cubes_surface_area := num_small_cubes * small_cube_surface_area
    total_small_cubes_surface_area - large_cube_surface_area = 600 := by
  intros large_cube_volume small_cube_volume num_small_cubes h1 h2 h3
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_difference_l112_11203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_equality_l112_11252

theorem cos_double_angle_equality (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.cos (2 * α) = 2 * Real.cos (α + π/4)) : 
  Real.sin (2 * α) = 1 ∧ Real.tan α = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_equality_l112_11252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_prime_eval_l112_11275

/-- A non-constant polynomial with integer coefficients. -/
def NonConstantIntPoly : Type := 
  {p : Polynomial ℤ // ∃ (a b : ℤ), a ≠ b ∧ p.eval a ≠ p.eval b}

/-- Theorem stating that for any non-constant polynomial with integer coefficients,
    there exists an integer n such that P(n^2 + 2020) is not prime. -/
theorem exists_non_prime_eval (P : NonConstantIntPoly) : 
  ∃ n : ℤ, ¬ Nat.Prime (Int.natAbs (P.val.eval (n^2 + 2020))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_prime_eval_l112_11275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_in_angle_l112_11233

theorem path_length_in_angle (a b : ℝ) (h : 0 < a ∧ a < b) :
  let angle := 60 * π / 180
  let total_path_length := a + 2 * Real.sqrt ((a^2 + a*b + b^2) / 3)
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 0 ∧
    x * Real.sin angle = a ∧
    y * Real.sin angle = b ∧
    x * Real.cos angle + y * Real.cos angle = x + y ∧
    total_path_length = x + Real.sqrt (a^2 + (y - x)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_in_angle_l112_11233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_number_mean_l112_11221

def numbers : List ℕ := [1451, 1723, 1987, 2056, 2191, 2212]

theorem remaining_number_mean (group_of_five : List ℕ) 
  (h1 : group_of_five.length = 5)
  (h2 : group_of_five.sum / group_of_five.length = 1900)
  (h3 : ∀ x, x ∈ group_of_five → x ∈ numbers)
  (h4 : ∃! x, x ∈ numbers ∧ x ∉ group_of_five) :
  ∃ x, x ∈ numbers ∧ x ∉ group_of_five ∧ x = 2120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_number_mean_l112_11221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AP_is_one_third_l112_11214

-- Define the square
def square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the circle
def circle_omega : Set (ℝ × ℝ) :=
  {p | (p.1 - 0)^2 + (p.2 - 1/3)^2 = 1/9}

-- Define points
def A : ℝ × ℝ := (0, 1)
def M : ℝ × ℝ := (0, 0)

-- Define the line AM
def line_AM : Set (ℝ × ℝ) :=
  {p | p.1 = 0}

-- Define point P
noncomputable def P : ℝ × ℝ := (0, 2/3)

-- Theorem statement
theorem length_AP_is_one_third :
  P ∈ circle_omega ∧ P ∈ line_AM ∧ P ≠ M →
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AP_is_one_third_l112_11214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_one_l112_11215

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 - 6 * x + 7) / (3 * x - 1.5)

theorem f_has_minimum_one :
  ∃ (x : ℝ), -3 < x ∧ x < 2 ∧ x ≠ 0.5 ∧
  (∀ (y : ℝ), -3 < y ∧ y < 2 ∧ y ≠ 0.5 → f x ≤ f y) ∧
  f x = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_one_l112_11215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billing_cost_effectiveness_method_b_better_for_350_l112_11231

/-- Billing amount for Method A -/
noncomputable def billing_a (t : ℝ) : ℝ :=
  if t ≤ 200 then 78 else 0.25 * t + 28

/-- Billing amount for Method B -/
noncomputable def billing_b (t : ℝ) : ℝ :=
  if t ≤ 500 then 108 else 0.19 * t + 13

/-- Theorem stating the cost-effectiveness of billing methods A and B -/
theorem billing_cost_effectiveness :
  ∀ t : ℝ, t ≥ 0 →
    (t < 320 → billing_a t < billing_b t) ∧
    (t > 320 → billing_b t < billing_a t) :=
by sorry

/-- Corollary: Method B is more cost-effective for 350 minutes of outgoing calls -/
theorem method_b_better_for_350 :
  billing_b 350 < billing_a 350 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billing_cost_effectiveness_method_b_better_for_350_l112_11231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_day_hike_distance_l112_11277

/-- Calculates the distance hiked on the third day of a three-day hike --/
noncomputable def third_day_distance (total_distance : ℝ) (first_day_distance : ℝ) (detour_distance : ℝ) 
  (after_river_distance : ℝ) (lost_time : ℝ) (hiking_pace : ℝ) : ℝ :=
  let second_day_distance := total_distance / 2
  let remaining_distance := total_distance - first_day_distance - second_day_distance
  let distance_before_storm := detour_distance + after_river_distance
  let missed_distance := lost_time * hiking_pace
  remaining_distance - distance_before_storm + missed_distance

theorem third_day_hike_distance : 
  third_day_distance 50 10 3 4 2 3 = 14 := by
  unfold third_day_distance
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_day_hike_distance_l112_11277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l112_11206

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 3)

theorem min_value_of_f (φ : ℝ) (h1 : |φ| < Real.pi / 2) 
  (h2 : ∀ x, Real.sin (2 * (x + Real.pi / 6) + φ) = -Real.sin (-2 * (x + Real.pi / 6) - φ)) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.sqrt 3 / 2) ∧ 
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -Real.sqrt 3 / 2) := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l112_11206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_8_l112_11284

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem f_value_at_pi_over_8 (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + Real.pi) = f ω x) :
  f ω (Real.pi / 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_8_l112_11284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l112_11225

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + 2 * Real.sin (x + Real.pi / 4) * Real.sin (x - Real.pi / 4)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  (∀ (x y : ℝ), -Real.pi/4 ≤ x ∧ x < y ∧ y ≤ Real.pi/4 ∧ -Real.pi/6 ≤ x → f x < f y) ∧
  (∀ (x : ℝ), -Real.pi/4 ≤ x ∧ x < -Real.pi/6 → ∃ (y : ℝ), -Real.pi/4 ≤ y ∧ y < x ∧ f y > f x) ∧
  (∀ (x : ℝ), Real.pi/4 < x ∧ x ≤ Real.pi/4 → ∃ (y : ℝ), x < y ∧ y ≤ Real.pi/4 ∧ f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l112_11225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l112_11202

/-- The function f(x) defined for x > 1 -/
noncomputable def f (x : ℝ) : ℝ := x - 9 / (2 - 2*x)

/-- Theorem stating that f(x) has a minimum value of 3√2 + 1 for x > 1 -/
theorem f_min_value : ∀ x > 1, f x ≥ 3 * Real.sqrt 2 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l112_11202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_cosine_theorem_l112_11267

/-- A regular triangular pyramid -/
structure RegularTriangularPyramid where
  side_length : ℝ
  height : ℝ

/-- A cross section of a regular triangular pyramid -/
structure CrossSection (pyramid : RegularTriangularPyramid) where
  is_through_edge_and_base_center : Prop
  is_equilateral_triangle : Prop

/-- The angle between the lateral face and the base of the pyramid -/
noncomputable def lateral_base_angle (pyramid : RegularTriangularPyramid) : ℝ :=
  Real.arccos (pyramid.height / (Real.sqrt ((pyramid.side_length^2 / 4) + pyramid.height^2)))

theorem cross_section_cosine_theorem (pyramid : RegularTriangularPyramid) 
  (cross_section : CrossSection pyramid) :
  cross_section.is_through_edge_and_base_center →
  cross_section.is_equilateral_triangle →
  (Real.cos (lateral_base_angle pyramid) = 1/3 ∨ 
   Real.cos (lateral_base_angle pyramid) = Real.sqrt 6 / 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_cosine_theorem_l112_11267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_properties_l112_11243

/-- Represents the linear regression model for weight and height -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- The sample data used for regression -/
def sample : List (ℝ × ℝ) := sorry

/-- The linear regression model for the given sample -/
def model : LinearRegression := sorry

/-- Mean of x values in the sample -/
noncomputable def mean_x : ℝ := sorry

/-- Mean of y values in the sample -/
noncomputable def mean_y : ℝ := sorry

/-- Theorem stating properties of the linear regression model -/
theorem linear_regression_properties :
  model.slope = 0.85 ∧ 
  model.intercept = -85.71 →
  (0 < model.slope) ∧ 
  (model.slope * mean_x + model.intercept = mean_y) ∧
  (∀ δ : ℝ, model.slope * (mean_x + δ) + model.intercept = mean_y + model.slope * δ) ∧
  (∀ x y : ℝ, model.slope * x + model.intercept = y → 
    ∃ ε : ℝ, y + ε = model.slope * x + model.intercept ∧ ε ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_properties_l112_11243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_range_l112_11249

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 3 then 2*x - x^2
  else if -2 ≤ x ∧ x ≤ 0 then x^2 + 6*x
  else 0  -- Default value for x outside the specified ranges

-- State the theorem
theorem f_value_range :
  ∀ y : ℝ, (∃ x : ℝ, (0 < x ∧ x ≤ 3 ∨ -2 ≤ x ∧ x ≤ 0) ∧ f x = y) ↔ -8 ≤ y ∧ y ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_range_l112_11249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l112_11282

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the specific conditions for our triangle
def SpecialTriangle (t : Triangle) : Prop :=
  t.B = 2 * t.C ∧ 2 * t.b = 3 * t.c

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  -- Part I: Prove cos(C) = 3/4
  Real.cos t.C = 3/4 ∧
  -- Part II: Prove area = (15√7)/4 when c = 4
  (t.c = 4 → t.a * t.b * Real.sin t.C / 2 = 15 * Real.sqrt 7 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l112_11282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_positive_necessary_not_sufficient_l112_11216

/-- Predicate to check if the equation mx^2 + ny^2 = 1 represents an ellipse -/
def IsEllipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

/-- Given constants m and n, prove that "mn > 0" is a necessary but not sufficient condition
    for "the curve of the equation mx^2 + ny^2 = 1 is an ellipse" -/
theorem mn_positive_necessary_not_sufficient (m n : ℝ) :
  (∃ (x y : ℝ), m * x^2 + n * y^2 = 1 ∧ IsEllipse m n) →
  (m * n > 0 ∧ ¬(m * n > 0 → IsEllipse m n)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_positive_necessary_not_sufficient_l112_11216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_lines_exist_l112_11297

/-- Two points in a plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in a plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (p : Point2D) (l : Line2D) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- The main theorem -/
theorem three_tangent_lines_exist (A B : Point2D) (h : distance A B = 5) :
  ∃! (lines : Finset Line2D), lines.card = 3 ∧
    ∀ l ∈ lines, distancePointToLine A l = 2 ∧ distancePointToLine B l = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangent_lines_exist_l112_11297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l112_11294

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The left focus of the hyperbola -/
noncomputable def leftFocus (h : Hyperbola a b) : Point :=
  { x := -Real.sqrt (a^2 + b^2), y := 0 }

/-- The right focus of the hyperbola -/
noncomputable def rightFocus (h : Hyperbola a b) : Point :=
  { x := Real.sqrt (a^2 + b^2), y := 0 }

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: The slope of the asymptote of a hyperbola is ±b/a -/
theorem hyperbola_asymptote_slope (h : Hyperbola a b) 
  (A B : Point) 
  (tangent_intersects_asymptote : A.x < 0 ∧ A.y > 0)
  (tangent_intersects_hyperbola : B.x > 0)
  (AB_equals_BF2 : distance A B = distance B (rightFocus h))
  (F1_on_circle : distance (leftFocus h) ⟨0, 0⟩ = a) :
  (b / a = 2 ∨ b / a = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l112_11294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l112_11261

theorem sum_remainder (s : ℕ) (hs : s > 0) : 
  let base := s + 1
  let sum := (Finset.range s).sum (λ i => (i + 1) * (base ^ (i + 1) - 1) / (base - 1))
  sum % (s - 1) = if s % 2 = 0 then 1 else (s + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l112_11261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_sin_false_a_lt_b_not_sufficient_necessary_for_square_negation_exists_quadratic_less_zero_unique_zero_ln_x_plus_x_l112_11270

-- Statement 1
theorem converse_sin_false : 
  ¬(∀ x : ℝ, Real.sin x = 1/2 → x = π/6) :=
sorry

-- Statement 2
theorem a_lt_b_not_sufficient_necessary_for_square : 
  ∃ a b : ℝ, (a < b ∧ a^2 ≥ b^2) ∨ (a ≥ b ∧ a^2 < b^2) :=
sorry

-- Statement 3
theorem negation_exists_quadratic_less_zero : 
  (¬∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
sorry

-- Statement 4
theorem unique_zero_ln_x_plus_x :
  ∃! x : ℝ, x > 1 ∧ x < 2 ∧ Real.log x + x - 3/2 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_sin_false_a_lt_b_not_sufficient_necessary_for_square_negation_exists_quadratic_less_zero_unique_zero_ln_x_plus_x_l112_11270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_equals_one_monotonicity_intervals_l112_11239

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 - 3 * x

-- Define the derivative of f(x)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * x - 3

theorem tangent_line_parallel_implies_a_equals_one (a : ℝ) :
  (f' a 1 = -4) → a = 1 := by sorry

theorem monotonicity_intervals (x : ℝ) :
  let a := 1
  (x < -1 ∨ x > 3 → f' a x > 0) ∧
  (-1 < x ∧ x < 3 → f' a x < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_equals_one_monotonicity_intervals_l112_11239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_management_percentage_is_six_percent_l112_11266

/-- Represents the business partnership between a working partner and a sleeping partner -/
structure Partnership where
  a_investment : ℚ
  b_investment : ℚ
  total_profit : ℚ
  a_total_received : ℚ

/-- Calculates the percentage of profit that the working partner receives for managing the business -/
noncomputable def management_percentage (p : Partnership) : ℚ :=
  let total_investment := p.a_investment + p.b_investment
  let a_profit_share_ratio := p.a_investment / total_investment
  let remaining_profit := p.total_profit * (1 - (p.a_total_received - a_profit_share_ratio * p.total_profit) / p.total_profit)
  ((p.a_total_received - a_profit_share_ratio * remaining_profit) / p.total_profit) * 100

/-- The main theorem stating that the management percentage is 6% for the given conditions -/
theorem management_percentage_is_six_percent (p : Partnership) 
  (h1 : p.a_investment = 2000)
  (h2 : p.b_investment = 3000)
  (h3 : p.total_profit = 9600)
  (h4 : p.a_total_received = 4416) :
  management_percentage p = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_management_percentage_is_six_percent_l112_11266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_count_l112_11298

def digits : List Nat := [1, 3, 5, 7]

def valid_permutation (perm : List Nat) : Bool :=
  perm.length = 4 && perm.toFinset = digits.toFinset

def expr_value (perm : List Nat) : Nat :=
  match perm with
  | [a, b, c, d] => (a * b) + (c * d)
  | _ => 0

def distinct_values : Finset Nat :=
  (List.permutations digits).filter valid_permutation
    |>.map expr_value
    |>.toFinset

theorem distinct_values_count :
  distinct_values.card = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_count_l112_11298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_amount_proof_l112_11250

/-- Calculates the number of tablespoons of soap needed for a bubble mix recipe. -/
noncomputable def soap_needed (container_capacity : ℝ) (soap_per_cup : ℝ) (ounces_per_cup : ℝ) : ℝ :=
  (container_capacity / ounces_per_cup) * soap_per_cup

/-- Proves that the amount of soap needed for the given recipe and container is 15 tablespoons. -/
theorem soap_amount_proof :
  soap_needed 40 3 8 = 15 := by
  -- Unfold the definition of soap_needed
  unfold soap_needed
  -- Simplify the arithmetic
  simp [div_mul_eq_mul_div]
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_amount_proof_l112_11250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l112_11281

noncomputable def f (x : ℝ) : ℝ := (x + 2)^0 / (x + 1)

theorem domain_of_f :
  {x : ℝ | f x ≠ 0 ∧ (x + 1 ≠ 0)} = {x : ℝ | x ≠ -1 ∧ x ≠ -2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l112_11281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_work_time_l112_11200

-- Define the work completion time for the man and son together
noncomputable def combined_time : ℝ := 4

-- Define the work completion time for the son alone
noncomputable def son_time : ℝ := 20

-- Define the work completion rate for the man and son together
noncomputable def combined_rate : ℝ := 1 / combined_time

-- Define the work completion rate for the son alone
noncomputable def son_rate : ℝ := 1 / son_time

-- State the theorem
theorem man_work_time :
  ∃ (man_time : ℝ), 
    man_time > 0 ∧ 
    (1 / man_time) + son_rate = combined_rate ∧
    man_time = 5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_work_time_l112_11200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l112_11259

/-- An ellipse with center at the origin, right focus at (√15, 0), and 
    an intersection point with y = x at (2, 2) has the equation (x²/20) + (y²/5) = 1 -/
theorem ellipse_equation (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (x^2 / 20 + y^2 / 5 = 1)) →
  F = (Real.sqrt 15, 0) →
  (2, 2) ∈ C →
  (∀ (x y : ℝ), (x, y) ∈ C → x^2 / 20 + y^2 / 5 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l112_11259
