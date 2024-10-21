import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_squares_l1356_135656

def sequence_a : ℕ → ℕ
  | 0 => 45  -- We define a₀ = 45 to handle the zero case
  | 1 => 45
  | n+2 => (sequence_a (n+1))^2 + 15 * (sequence_a (n+1))

theorem no_perfect_squares (n : ℕ) : ∃ (k : ℕ), sequence_a n ≠ k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_squares_l1356_135656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_encryption_theorem_l1356_135622

def encryption_algorithm (i : ℕ) (A : List ℤ) : List ℤ :=
  if i % 2 = 1 then
    A.map (fun x => if x % 2 = 1 then x + i else x - 1)
  else
    A.map (fun x => if x % 2 = 0 then x + 2*i else x - 2)

def B : ℕ → List ℤ
  | 0 => [2, 0, 2, 3, 5, 7]
  | n + 1 => encryption_algorithm (n + 1) (B n)

theorem encryption_theorem :
  B 2 = [-1, -3, -1, 8, 10, 12] ∧
  ∀ n : ℕ, (B (2*n)).sum = 9*n^2 + 4*n + 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_encryption_theorem_l1356_135622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_used_in_peanut_butter_l1356_135677

def oil_to_peanuts_ratio : ℚ := 2 / 8
def total_weight : ℚ := 20
def batch_weight : ℚ := 2 + 8

theorem oil_used_in_peanut_butter : 
  (total_weight / batch_weight) * (oil_to_peanuts_ratio * batch_weight) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_used_in_peanut_butter_l1356_135677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l1356_135689

/-- The function g defined for positive real numbers a, b, and c -/
noncomputable def g (a b c : ℝ) : ℝ := a^2 / (a^2 + b^2) + b^2 / (b^2 + c^2) + c^2 / (c^2 + a^2)

/-- Theorem stating that g(a,b,c) is strictly between 1 and 2 for positive a, b, and c -/
theorem g_bounds {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  1 < g a b c ∧ g a b c < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l1356_135689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_insertion_theorem_l1356_135652

/-- Represents a natural number with a specific number of digits -/
structure DigitNumber (n : ℕ) where
  value : ℕ
  digit_count : (Nat.repr value).length = n

/-- Represents the result of deleting digits from a number -/
structure DeletionResult where
  original : DigitNumber 2007
  deleted : Fin 2007 → Bool
  result : DigitNumber 2000

/-- Represents the result of inserting digits into a number -/
structure InsertionResult where
  original : DigitNumber 2007
  inserted : Fin 2014 → Bool
  result : DigitNumber 2014

/-- The main theorem statement -/
theorem digit_insertion_theorem 
  (n1 n2 : DigitNumber 2007) 
  (d1 : DeletionResult) 
  (d2 : DeletionResult) :
  d1.original = n1 →
  d2.original = n2 →
  d1.result = d2.result →
  ∃ (i1 i2 : InsertionResult), 
    i1.original = n1 ∧ 
    i2.original = n2 ∧ 
    i1.result = i2.result := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_insertion_theorem_l1356_135652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_cost_theorem_l1356_135693

/-- Represents the cost and ratio of a metal in an alloy. -/
structure Metal where
  cost : ℚ  -- Cost per kg in Rs.
  ratio : ℚ  -- Ratio in the alloy

/-- Calculates the cost of an alloy given its components. -/
def alloyCost (metals : List Metal) : ℚ :=
  let totalCost := (metals.map (λ m => m.cost * m.ratio)).sum
  let totalRatio := (metals.map (λ m => m.ratio)).sum
  totalCost / totalRatio

/-- Theorem stating that the given mixture results in the desired alloy cost. -/
theorem alloy_cost_theorem (metalA metalB metalC : Metal) 
  (hA : metalA.cost = 68 ∧ metalA.ratio = 6)
  (hB : metalB.cost = 96 ∧ metalB.ratio = 1)
  (hC : metalC.cost = 110 ∧ metalC.ratio = 0) :
  alloyCost [metalA, metalB, metalC] = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_cost_theorem_l1356_135693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_of_N_l1356_135687

def M : Finset ℕ := {0, 2, 3, 7}

def N : Finset ℕ := Finset.image (λ (p : ℕ × ℕ) => p.1 * p.2) (M.product M)

theorem max_subsets_of_N : Finset.card (Finset.powerset N) = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_of_N_l1356_135687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_6x_plus_9_times_sin_2x_l1356_135673

theorem integral_x_squared_plus_6x_plus_9_times_sin_2x : 
  ∫ x in (-3)..0, (x^2 + 6*x + 9) * Real.sin (2*x) = -(17 + Real.cos 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_6x_plus_9_times_sin_2x_l1356_135673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_count_theorem_l1356_135662

/-- The number of distinct, natural-number factors of 4^5 * 5^2 * 6^3 -/
def num_factors : ℕ := 168

/-- The given number -/
def given_number : ℕ := 4^5 * 5^2 * 6^3

theorem factor_count_theorem :
  (Finset.filter (· ∣ given_number) (Finset.range (given_number + 1))).card = num_factors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_count_theorem_l1356_135662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_curve_bounded_region_l1356_135651

/-- The equation of the curve bounding the region -/
def curve_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 50 * abs x = 500

/-- The set of points satisfying the curve equation -/
def curve_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve_equation p.1 p.2}

/-- The bounded region -/
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x₁ x₂ y₁ y₂ : ℝ), 
    x₁ ≤ p.1 ∧ p.1 ≤ x₂ ∧ y₁ ≤ p.2 ∧ p.2 ≤ y₂ ∧
    (x₁, y₁) ∈ curve_set ∧ (x₂, y₂) ∈ curve_set}

/-- The area of the bounded region -/
noncomputable def area_of_bounded_region : ℝ :=
  (MeasureTheory.volume bounded_region).toReal

theorem area_of_curve_bounded_region :
  area_of_bounded_region = 2500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_curve_bounded_region_l1356_135651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1356_135631

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 3)

theorem domain_of_f :
  Set.range f = {y | ∃ x, f x = y ∧ ∀ k : ℤ, x ≠ Real.pi / 12 + k * Real.pi / 2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1356_135631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_is_odd_l1356_135699

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b x : ℝ, f x = b * f a + a * f b

/-- The theorem statement -/
theorem special_function_is_odd (f : ℝ → ℝ) 
  (h1 : special_function f) 
  (h2 : ∃ x, f x ≠ 0) : 
  ∀ x, f (-x) = -f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_is_odd_l1356_135699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_set_forming_option_l1356_135600

/-- A predicate that determines if a description is definite enough to form a set -/
def isDefiniteDescription (description : String) : Prop := sorry

/-- The list of given options -/
def options : List String := [
  "Students with a higher level of basketball skills in the school",
  "Tall trees in the campus",
  "All the countries in the European Union in 2012",
  "Economically developed cities in China"
]

/-- Theorem stating that only one option can form a set -/
theorem only_one_set_forming_option :
  ∃! option, option ∈ options ∧ isDefiniteDescription option ∧ 
  option = "All the countries in the European Union in 2012" := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_set_forming_option_l1356_135600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_equation_route_equation_correct_l1356_135641

/-- Represents the equation for the route problem -/
theorem route_equation (x : ℝ) : Prop :=
  25 / x - 21 / (1.4 * x) = 1 / 3

/-- The length of route a in kilometers -/
def route_a_length : ℚ := 25

/-- The length of route b in kilometers -/
def route_b_length : ℚ := 21

/-- The speed increase factor for route b compared to route a -/
def speed_increase_factor : ℚ := 14/10

/-- The time saved in hours by taking route b instead of route a -/
def time_saved : ℚ := 1 / 3

/-- Theorem stating that the route_equation correctly represents the given conditions -/
theorem route_equation_correct (x : ℝ) (hx : x > 0) :
  route_equation x ↔
  (route_a_length / x - route_b_length / (speed_increase_factor * x) = time_saved) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_equation_route_equation_correct_l1356_135641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_values_l1356_135604

-- Define the line l
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, 1 + t * Real.sin α)

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 = 4 * y

-- Define the intersection condition
def intersects (α : ℝ) : Prop :=
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ 
    curve_C (line_l α t₁).1 (line_l α t₁).2 ∧
    curve_C (line_l α t₂).1 (line_l α t₂).2

-- Define the distance between intersection points
noncomputable def distance (α : ℝ) : ℝ :=
  let t₁ := (4 * Real.sin α + Real.sqrt (16 * Real.sin α^2 + 16 * Real.cos α^2)) / (2 * Real.cos α^2)
  let t₂ := (4 * Real.sin α - Real.sqrt (16 * Real.sin α^2 + 16 * Real.cos α^2)) / (2 * Real.cos α^2)
  abs (t₁ - t₂)

-- Theorem statement
theorem intersection_angle_values (α : ℝ) 
  (h₁ : 0 ≤ α ∧ α < Real.pi)
  (h₂ : intersects α)
  (h₃ : distance α = 8) :
  α = Real.pi / 4 ∨ α = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_values_l1356_135604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_equation_l1356_135613

/-- Given a line and a point, find the equation of the symmetric line --/
theorem symmetric_line_equation (a b c : ℝ) (x₀ y₀ : ℝ) :
  let original_line := λ (x y : ℝ) ↦ a * x + b * y + c = 0
  let symmetry_point := (x₀, y₀)
  let symmetric_line := λ (x y : ℝ) ↦ a * (2 * x₀ - x) + b * (2 * y₀ - y) + c = 0
  (original_line 0 0 ∧ a = 2 ∧ b = 3 ∧ c = -6 ∧ x₀ = 1 ∧ y₀ = -1) →
  symmetric_line = λ (x y : ℝ) ↦ 2 * x + 3 * y + 8 = 0 :=
by
  intro h
  sorry  -- Placeholder for the actual proof

#check symmetric_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_equation_l1356_135613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_upper_bound_l1356_135601

theorem sin_upper_bound :
  (¬ ∃ α, Real.sin α > 1) ∧ (∀ α, Real.sin α ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_upper_bound_l1356_135601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triplets_l1356_135657

theorem count_triplets (n : ℕ) : 
  (Finset.sum (Finset.range (n + 1)) (λ z : ℕ => (z - 1)^2)) = n * (n + 1) * (2 * n + 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triplets_l1356_135657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_is_centroid_l1356_135635

def harry : ℚ × ℚ := (10, -3)
def sandy : ℚ × ℚ := (2, 7)
def ron : ℚ × ℚ := (6, 1)

def centroid (p1 p2 p3 : ℚ × ℚ) : ℚ × ℚ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

theorem meeting_point_is_centroid :
  centroid harry sandy ron = (6, 5/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_is_centroid_l1356_135635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_derivative_bound_range_l1356_135650

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := x - a * Real.log x + b

-- Theorem 1
theorem tangent_line_values (a b : ℝ) :
  (∀ x, deriv (f a b) x = 2) ∧ 
  (f a b 1 = 5) → 
  a = -1 ∧ b = 4 := by sorry

-- Theorem 2
theorem derivative_bound_range (a : ℝ) :
  (∀ x ∈ Set.Icc (2 : ℝ) 3, |deriv (f a 0) x| < 3 / x^2) →
  a ∈ Set.Icc (2 : ℝ) (7/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_derivative_bound_range_l1356_135650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_ellipse_l1356_135672

-- Define the two fixed points
def point1 : ℝ × ℝ := (0, 4)
def point2 : ℝ × ℝ := (6, -2)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the equation
def ellipse_equation (x y : ℝ) : Prop :=
  distance (x, y) point1 + distance (x, y) point2 = 12

-- Theorem statement
theorem is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), ellipse_equation x y ↔
    (x^2 / a^2) + (y^2 / b^2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_ellipse_l1356_135672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_squared_length_l1356_135605

noncomputable def a (x : ℝ) : ℝ := 3 * x + 1
noncomputable def b (x : ℝ) : ℝ := -2 * x + 6
noncomputable def c (x : ℝ) : ℝ := x - 3

noncomputable def m (x : ℝ) : ℝ := (a x + b x + c x) / 3

theorem graph_squared_length : 
  (4 - (-2))^2 + (m 4 - m (-2))^2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_squared_length_l1356_135605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_2_range_of_a_for_inequality_l1356_135609

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := Real.sqrt (x^2 - 2*x + 1) + |x + a|

-- Part 1
theorem min_value_when_a_is_2 :
  ∃ (min : ℝ), min = 3 ∧ ∀ x, f x 2 ≥ min := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  ∀ a, (∀ x ∈ Set.Icc (2/3) 1, f x a ≤ x) ↔ a ∈ Set.Icc (-1) (-1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_2_range_of_a_for_inequality_l1356_135609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_count_l1356_135606

/-- Represents the number of red balls in the bag -/
def x : ℕ := sorry

/-- Represents the number of black balls added to the bag -/
def black_balls : ℕ := 10

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := x + black_balls

/-- Represents the probability of drawing a black ball -/
def prob_black : ℚ := 2/5

theorem red_balls_count : 
  (black_balls : ℚ) / total_balls = prob_black → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_count_l1356_135606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_open_interval_min_a_value_b_range_l1356_135692

-- Define the set M
def M : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define the set N
def N (a b : ℝ) : Set ℝ := {x | a < x ∧ x < b}

-- Theorem 1: M is equivalent to the open interval (-1, 2)
theorem M_equals_open_interval : M = Set.Ioo (-1) 2 := by sorry

-- Theorem 2: If M ⊇ N, then the minimum value of a is -1
theorem min_a_value (a b : ℝ) (h : M ⊇ N a b) : 
  ∀ ε > 0, ∃ a' ≥ -1 - ε, M ⊇ N a' b := by sorry

-- Theorem 3: If M ∩ N = M, then b ∈ [2, +∞)
theorem b_range (a b : ℝ) (h : M ∩ N a b = M) : b ∈ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_open_interval_min_a_value_b_range_l1356_135692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l1356_135663

/-- The circle centered at (-1, 0) with radius 1 -/
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

/-- The line 3x + 4y - 2 = 0 -/
def my_line (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0

/-- Point P -/
def P : ℝ × ℝ := (-2, 2)

theorem line_tangent_to_circle :
  (∀ x y, my_line x y → (x, y) = P ∨ (∃ t, x = -2 + 3*t ∧ y = 2 - 4*t)) ∧
  (∃! x y, my_line x y ∧ my_circle x y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l1356_135663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_1000_eq_neg_one_l1356_135691

def mySequence (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | 1 => 5
  | n + 2 => mySequence (n + 1) - mySequence n

theorem mySequence_1000_eq_neg_one : mySequence 999 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_1000_eq_neg_one_l1356_135691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1356_135624

/-- Regular hexagon with side length 1 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ := 1)

/-- Point on side AF of the hexagon -/
def PointOnAF (hex : RegularHexagon) := {P : ℝ × ℝ // ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • hex.A + t • hex.F}

/-- Dot product of vectors PD and PE -/
def dotProduct (hex : RegularHexagon) (P : PointOnAF hex) : ℝ :=
  let PD := (hex.D.1 - P.val.1, hex.D.2 - P.val.2)
  let PE := (hex.E.1 - P.val.1, hex.E.2 - P.val.2)
  PD.1 * PE.1 + PD.2 * PE.2

/-- Theorem: The minimum value of PD · PE is 3/2 -/
theorem min_dot_product (hex : RegularHexagon) :
  ∃ min : ℝ, min = (3/2 : ℝ) ∧ ∀ P : PointOnAF hex, dotProduct hex P ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1356_135624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_riemann_zeta_sum_l1356_135653

-- Define the Riemann zeta function
noncomputable def zeta (x : ℝ) : ℝ := ∑' n, (1 : ℝ) / n^x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- State the theorem
theorem riemann_zeta_sum :
  (∑' k : ℕ, frac (zeta (2 * (k + 3 : ℕ)))) = 1 / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_riemann_zeta_sum_l1356_135653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arith_geom_triple_exists_unique_l1356_135683

/-- A structure representing a triple of real numbers that can form both
    an arithmetic and geometric sequence. -/
structure ArithGeomTriple where
  a : ℝ
  b : ℝ
  c : ℝ
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c
  arith_seq : ∃ (x y z : ℝ), ({x, y, z} : Set ℝ) = {a, b, c} ∧ y - x = z - y
  geom_seq : ∃ (x y z : ℝ), ({x, y, z} : Set ℝ) = {a, b, c} ∧ y * y = x * z
  sum_six : a + b + c = 6

/-- The theorem stating the existence and uniqueness of the ArithGeomTriple. -/
theorem arith_geom_triple_exists_unique :
  ∃! (t : ArithGeomTriple), ({t.a, t.b, t.c} : Set ℝ) = {-4, 2, 8} ∨ ({t.a, t.b, t.c} : Set ℝ) = {8, 2, -4} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arith_geom_triple_exists_unique_l1356_135683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1356_135696

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.sin C / (Real.sin A * Real.cos B) = 2 * c / a →
  Real.cos A = 1 / 4 →
  B = π / 3 ∧ Real.sin C = (Real.sqrt 15 + Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1356_135696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_sale_savings_l1356_135643

/-- Calculates the percentage saved in a ticket sale -/
theorem ticket_sale_savings (original_price : ℝ) (num_tickets : ℕ) (sale_price_ratio : ℕ) : 
  original_price > 0 → 
  num_tickets > 0 → 
  sale_price_ratio > 0 → 
  sale_price_ratio < num_tickets →
  let sale_price := (sale_price_ratio : ℝ) / (num_tickets : ℝ) * original_price
  let savings := original_price - sale_price
  let savings_percentage := savings / original_price * 100
  (num_tickets = 20 ∧ sale_price_ratio = 4 ∧ original_price = 600) → 
  savings_percentage = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_sale_savings_l1356_135643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_sine_symmetry_l1356_135639

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x)

-- Define the translated function
noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 12)

-- Define the center of symmetry
noncomputable def center_of_symmetry (k : ℤ) : ℝ × ℝ := (k * Real.pi / 2 + Real.pi / 12, 0)

-- Theorem statement
theorem translated_sine_symmetry (k : ℤ) :
  ∀ x : ℝ, g ((center_of_symmetry k).1 + x) = g ((center_of_symmetry k).1 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_sine_symmetry_l1356_135639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_greater_than_two_l1356_135670

/-- The function f(x) = ax - (2a+1)/x where a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (2 * a + 1) / x

/-- Theorem: If f(m^2 + 1) > f(m^2 - m + 3) for a > 0, then m > 2 -/
theorem m_greater_than_two (a m : ℝ) (ha : a > 0) 
  (h : f a (m^2 + 1) > f a (m^2 - m + 3)) : m > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_greater_than_two_l1356_135670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_P_l1356_135665

/-- The equation of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The x-coordinate of the focus -/
noncomputable def focus_x : ℝ := Real.sqrt 3

/-- The condition for equal angles -/
def equal_angles (p : ℝ) : Prop :=
  ∀ (x_A y_A x_B y_B : ℝ),
    ellipse x_A y_A → ellipse x_B y_B →
    y_A * (x_B - focus_x) = y_B * (x_A - focus_x) →
    y_A / (x_A - p) = -y_B / (x_B - p)

/-- The theorem stating the unique point P -/
theorem unique_point_P :
  ∃! p : ℝ, p > 0 ∧ equal_angles p ∧ p = 2 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_P_l1356_135665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_solutions_l1356_135612

def is_valid_solution (n : ℕ) : Prop :=
  ∃ (A B C D : ℕ),
    A ≥ 1 ∧ A ≤ 9 ∧
    B ≥ 0 ∧ B ≤ 9 ∧
    C ≥ 0 ∧ C ≤ 9 ∧
    D ≥ 0 ∧ D ≤ 9 ∧
    n = A * 1000 + D * 100 + D * 10 + C ∧
    (A * 100 + B * 10 + C) * (A * 10 + D) = n

theorem valid_solutions :
  ∀ n : ℕ, is_valid_solution n ↔ n ∈ ({1000, 1011, 1044, 1055} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_solutions_l1356_135612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_digit_sum_ratio_l1356_135671

/-- S(n) denotes the sum of digits of n -/
def S (n : ℕ+) : ℕ := sorry

/-- The theorem states that the largest possible value of S(n) / S(16n) is 13 -/
theorem largest_digit_sum_ratio :
  (∀ n : ℕ+, (S n : ℚ) / (S (16 * n) : ℚ) ≤ 13) ∧
  (∃ n : ℕ+, (S n : ℚ) / (S (16 * n) : ℚ) = 13) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_digit_sum_ratio_l1356_135671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_squared_sum_l1356_135626

open Real
open BigOperators

theorem cosine_squared_sum : 
  (∑ k in Finset.range 30, (cos ((3 + 6 * k : ℝ) * π / 180)) ^ 2) = 29/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_squared_sum_l1356_135626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1356_135602

theorem trigonometric_identity (α : ℝ) :
  (Real.sin (2 * Real.pi + α / 4) * (Real.cos (α / 8) / Real.sin (α / 8)) - Real.cos (2 * Real.pi + α / 4)) /
  (Real.cos (α / 4 - 3 * Real.pi) * (Real.cos (α / 8) / Real.sin (α / 8)) + Real.cos (7 * Real.pi / 2 - α / 4)) =
  -Real.tan (α / 8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1356_135602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_odd_function_property_l1356_135682

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Statement 1
theorem decreasing_function (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0) :
  ∀ x y : ℝ, x < y → f y < f x := by
  sorry

-- Definition of odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Statement 3
theorem odd_function_property (h : is_odd f) :
  is_odd (fun x ↦ f x - f (|x|)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_odd_function_property_l1356_135682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1356_135642

-- Define the curve C
noncomputable def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the condition for points A and B
def pointsOnC (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  C x₁ y₁ ∧ C x₂ y₂ ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = 4

-- Define the area of triangle ABQ
noncomputable def triangleABQArea (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  let k := (y₂ - y₁) / (x₂ - x₁)
  let m := y₁ - k * x₁
  4 * (1 + 1/k^2) * Real.sqrt (2 - 1/k^2)

-- Theorem statement
theorem max_triangle_area :
  ∀ x₁ y₁ x₂ y₂ : ℝ, pointsOnC x₁ y₁ x₂ y₂ →
  ∃ maxArea : ℝ, maxArea = 8 ∧
  ∀ a b : ℝ, pointsOnC a b x₂ y₂ →
  triangleABQArea a b x₂ y₂ ≤ maxArea := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1356_135642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1356_135697

noncomputable def vector_BA : ℝ × ℝ := (Real.sqrt 3 / 2, 1 / 2)
def vector_BC : ℝ × ℝ := (0, 1)

theorem angle_between_vectors :
  Real.arccos ((vector_BA.1 * vector_BC.1 + vector_BA.2 * vector_BC.2) /
    (Real.sqrt (vector_BA.1^2 + vector_BA.2^2) * Real.sqrt (vector_BC.1^2 + vector_BC.2^2))) = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1356_135697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_special_expansion_l1356_135678

/-- 
Given an expression (√x - 1/(2∛x))^n, if the binomial coefficients of the third and fourth terms 
in its expansion are equal and are the maximum, then the constant term in the expansion is -5/4.
-/
theorem constant_term_of_special_expansion (x : ℝ) (n : ℕ) : 
  (∃ k : ℕ, k > 2 ∧ 
    Nat.choose n k = Nat.choose n (k + 1) ∧ 
    (∀ m : ℕ, Nat.choose n m ≤ Nat.choose n k)) →
  (-1/2 : ℝ)^3 * (Nat.choose 5 3 : ℝ) = -5/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_special_expansion_l1356_135678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_inequality_l1356_135688

/-- The function f(x) = |mx-2| - |mx+1| where m ∈ ℝ -/
def f (m : ℝ) (x : ℝ) : ℝ := |m*x - 2| - |m*x + 1|

/-- The maximum value of f(x) for any real m -/
noncomputable def n : ℝ := sSup {y | ∃ (m x : ℝ), f m x = y}

theorem max_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = n) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_inequality_l1356_135688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l1356_135634

-- Define the triangle's side lengths
noncomputable def a : ℝ := 30
noncomputable def b : ℝ := 26
noncomputable def c : ℝ := 10

-- Define the semi-perimeter
noncomputable def s : ℝ := (a + b + c) / 2

-- Define the area using Heron's formula
noncomputable def area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem triangle_area_approx : 
  ∃ ε > 0, abs (area - 126.72) < ε :=
by
  -- We'll use 0.01 as our epsilon
  use 0.01
  -- Split the goal into two parts
  constructor
  -- Prove ε > 0
  · norm_num
  -- Prove the inequality
  · sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l1356_135634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1356_135640

theorem polynomial_divisibility
  (F : Polynomial ℤ) (A : Finset ℤ)
  (h_div : ∀ n : ℤ, ∃ a ∈ A, a ∣ F.eval n) :
  ∃ a ∈ A, ∀ n : ℤ, a ∣ F.eval n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1356_135640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_implies_x_value_l1356_135611

/-- Given vectors a and b, if their dot product is 2, then the second component of b is 5. -/
theorem dot_product_implies_x_value (a b : ℝ × ℝ × ℝ) : 
  a = (-3, 2, 5) → 
  (Prod.fst b) = 1 → 
  (Prod.snd (Prod.snd b)) = -1 → 
  (Prod.fst a) * (Prod.fst b) + (Prod.fst (Prod.snd a)) * (Prod.fst (Prod.snd b)) + (Prod.snd (Prod.snd a)) * (Prod.snd (Prod.snd b)) = 2 → 
  (Prod.fst (Prod.snd b)) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_implies_x_value_l1356_135611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_buyers_difference_l1356_135633

/-- Represents the cost of a pencil in cents -/
def pencil_cost : ℕ := sorry

/-- Represents the number of seventh graders who bought pencils -/
def seventh_graders : ℕ := sorry

/-- Represents the number of sixth graders who bought pencils -/
def sixth_graders : ℕ := sorry

/-- The total amount paid by seventh graders in cents -/
def seventh_graders_total : ℕ := 221

/-- The total amount paid by sixth graders in cents -/
def sixth_graders_total : ℕ := 286

/-- The total number of sixth graders -/
def total_sixth_graders : ℕ := 35

theorem pencil_buyers_difference : 
  pencil_cost > 0 ∧ 
  pencil_cost * seventh_graders = seventh_graders_total ∧
  pencil_cost * sixth_graders = sixth_graders_total ∧
  sixth_graders ≤ total_sixth_graders →
  sixth_graders - seventh_graders = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_buyers_difference_l1356_135633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l1356_135615

-- Define the clock parameters
noncomputable def hour_hand_speed : ℝ := 30  -- degrees per hour
noncomputable def minute_hand_speed : ℝ := 6 -- degrees per minute

-- Define the time
noncomputable def hours : ℝ := 3
noncomputable def minutes : ℝ := 30

-- Define the positions of the hands
noncomputable def hour_hand_position : ℝ := hours * hour_hand_speed + (minutes / 60) * hour_hand_speed
noncomputable def minute_hand_position : ℝ := minutes * minute_hand_speed

-- Theorem to prove
theorem clock_angle_at_3_30 :
  |minute_hand_position - hour_hand_position| = 75 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l1356_135615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_sum_max_value_trig_sum_achievable_l1356_135610

theorem max_value_trig_sum (θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ) :
  Real.sin θ₁ * Real.cos θ₂ + Real.sin θ₂ * Real.cos θ₃ + Real.sin θ₃ * Real.cos θ₄ + 
  Real.sin θ₄ * Real.cos θ₅ + Real.sin θ₅ * Real.cos θ₆ ≤ 5/2 :=
by sorry

theorem max_value_trig_sum_achievable :
  ∃ θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ,
    Real.sin θ₁ * Real.cos θ₂ + Real.sin θ₂ * Real.cos θ₃ + Real.sin θ₃ * Real.cos θ₄ + 
    Real.sin θ₄ * Real.cos θ₅ + Real.sin θ₅ * Real.cos θ₆ = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_sum_max_value_trig_sum_achievable_l1356_135610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_evaluation_result_final_result_l1356_135675

theorem expression_simplification (a b : ℝ) : (a + b)^2 + b*(a - b) - 3*a*b = a^2 := by
  ring  -- This tactic will simplify the algebraic expression

theorem evaluation_result : (2 : ℝ)^2 = 4 := by
  norm_num  -- This tactic will evaluate the numerical expression

theorem final_result : (2 + 2023 : ℝ)^2 + 2023*(2 - 2023) - 3*2*2023 = 4 := by
  have h1 := expression_simplification 2 2023
  rw [h1]
  exact evaluation_result

#check final_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_evaluation_result_final_result_l1356_135675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_close_points_l1356_135655

-- Define a regular tetrahedron with edge length 1
structure RegularTetrahedron where
  edgeLength : ℝ
  isOne : edgeLength = 1

-- Define a point on the surface of the tetrahedron
structure SurfacePoint where
  x : ℝ
  y : ℝ
  z : ℝ
  onSurface : Bool

-- Define a set of nine points on the surface
def NinePoints (t : RegularTetrahedron) :=
  { points : Finset SurfacePoint // points.card = 9 ∧ ∀ p ∈ points, p.onSurface }

-- Define the distance between two points
noncomputable def distance (p1 p2 : SurfacePoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Theorem statement
theorem exist_close_points (t : RegularTetrahedron) (points : NinePoints t) :
  ∃ p1 p2, p1 ∈ points.val ∧ p2 ∈ points.val ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_close_points_l1356_135655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_average_age_l1356_135685

theorem population_average_age 
  (ratio_women_men : ℚ) 
  (avg_age_women : ℚ) 
  (avg_age_men : ℚ) :
  ratio_women_men = 12 / 11 →
  avg_age_women = 30 →
  avg_age_men = 35 →
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 745 / 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_average_age_l1356_135685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_largest_odd_divisors_l1356_135694

def largest_odd_divisor : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => if (n + 2) % 2 = 0 then largest_odd_divisor ((n + 2) / 2) else n + 2

def S (n : ℕ) : ℕ :=
  (List.range (2^n)).map (λ k => largest_odd_divisor (k + 1)) |>.sum

theorem sum_largest_odd_divisors (n : ℕ) : 3 * S n = 4^n + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_largest_odd_divisors_l1356_135694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_zone_points_l1356_135659

/-- Represents a circular target with multiple zones -/
structure Target where
  zones : Nat
  bullseyeRadius : ℝ
  bullseyePoints : ℝ

/-- Calculates the area of a circular region given its outer and inner radii -/
noncomputable def ringArea (outerRadius innerRadius : ℝ) : ℝ :=
  Real.pi * (outerRadius^2 - innerRadius^2)

/-- Calculates the points scored for hitting a zone based on its area and the bullseye points -/
noncomputable def zonePoints (target : Target) (zoneArea bullseyeArea : ℝ) : ℝ :=
  target.bullseyePoints * bullseyeArea / zoneArea

/-- Theorem: In a 5-zone target where ring widths equal bullseye radius,
    if bullseye scores 315 points, then the blue (4th) zone scores 35 points -/
theorem blue_zone_points (target : Target) :
  target.zones = 5 →
  target.bullseyePoints = 315 →
  let bullseyeArea := Real.pi * target.bullseyeRadius^2
  let blueZoneArea := ringArea (5 * target.bullseyeRadius) (4 * target.bullseyeRadius)
  zonePoints target blueZoneArea bullseyeArea = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_zone_points_l1356_135659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_one_l1356_135618

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then 2*x + 4
  else if x < 2 then x - 10
  else 3*x - 5

-- Theorem statement
theorem solutions_of_f_eq_one :
  {x : ℝ | f x = 1} = {-3/2, 2} := by
  sorry

#check solutions_of_f_eq_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_one_l1356_135618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_approx_root_2_6_l1356_135603

-- Define the function f(x) = log₁₀(x) + x - 3
noncomputable def f (x : ℝ) := Real.log x / Real.log 10 + x - 3

-- Define the approximations given in the problem
def log_10_2_5 : ℝ := 0.398
def log_10_2_75 : ℝ := 0.439
def log_10_2_5625 : ℝ := 0.409

-- Theorem statement
theorem root_exists_in_interval :
  ∃ x : ℝ, 2.5 < x ∧ x < 2.7 ∧ f x = 0 := by
  sorry

-- Theorem stating that 2.6 is an approximate root with 0.1 accuracy
theorem approx_root_2_6 :
  ∃ x : ℝ, |x - 2.6| < 0.1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_approx_root_2_6_l1356_135603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlys_backstroke_practice_l1356_135658

/-- Represents Carly's swimming practice schedule and calculates her daily backstroke practice time. -/
noncomputable def carlys_swimming_practice (butterfly_hours_per_day : ℝ) (butterfly_days_per_week : ℕ) 
  (backstroke_days_per_week : ℕ) (total_hours_per_month : ℝ) (weeks_per_month : ℕ) : ℝ :=
  let butterfly_hours_per_month := butterfly_hours_per_day * (butterfly_days_per_week : ℝ) * (weeks_per_month : ℝ)
  let backstroke_hours_per_month := total_hours_per_month - butterfly_hours_per_month
  backstroke_hours_per_month / ((backstroke_days_per_week : ℝ) * (weeks_per_month : ℝ))

/-- Theorem stating that Carly practices backstroke for 2 hours a day. -/
theorem carlys_backstroke_practice : 
  carlys_swimming_practice 3 4 6 96 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlys_backstroke_practice_l1356_135658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_f_strictly_increasing_l1356_135661

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (4 * x - Real.pi / 3)

-- Theorem for part (I)
theorem f_at_pi : f Real.pi = 1 / 2 := by sorry

-- Theorem for part (II)
theorem f_strictly_increasing (k : ℤ) :
  StrictMonoOn f (Set.Icc ((k : ℝ) * Real.pi / 2 - Real.pi / 6) ((k : ℝ) * Real.pi / 2 + Real.pi / 12)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_f_strictly_increasing_l1356_135661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1356_135674

/-- The minimum area of a triangle with vertices A(0,0), B(30,18), and C with integer coordinates --/
theorem min_triangle_area : 
  ∃ (p q : ℤ), 
    let A : ℝ × ℝ := (0, 0)
    let B : ℝ × ℝ := (30, 18)
    let C : ℝ × ℝ := (↑p, ↑q)
    (∀ (x y : ℤ), abs (3 * x - 5 * y) ≥ 1) ∧ 
    abs ((30 * ↑q - 18 * ↑p) : ℝ) / 2 = 1 / 2 := by
  sorry

#check min_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1356_135674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_comparison_theorem_l1356_135637

theorem product_comparison_theorem (S : Finset ℝ) : 
  (S.card = 10) → 
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y → x > 0 ∧ y > 0) → 
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y) → 
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    ((∀ d e, d ∈ S ∧ e ∈ S ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c ∧ e ≠ a ∧ e ≠ b ∧ e ≠ c → a * b * c > d * e) ∨
    (∀ d e f g, d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ 
      d ≠ a ∧ d ≠ b ∧ d ≠ c ∧ e ≠ a ∧ e ≠ b ∧ e ≠ c ∧ 
      f ≠ a ∧ f ≠ b ∧ f ≠ c ∧ g ≠ a ∧ g ≠ b ∧ g ≠ c → a * b * c > d * e * f * g)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_comparison_theorem_l1356_135637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1356_135679

noncomputable def f (x a θ : ℝ) : ℝ := a + 2 * (Real.cos x)^2 * Real.cos (2*x + θ)

theorem problem_solution (a θ α : ℝ) :
  (∀ x, f x a θ = -f (-x) a θ) →  -- f is an odd function
  f (π/4) a θ = 0 →
  θ ∈ Set.Ioo 0 π →
  f (α/4) a θ = -2/5 →
  α ∈ Set.Ioo (π/2) π →
  (a = -1 ∧ θ = π/2) ∧
  Real.sin (α + π/3) = (4 - 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1356_135679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_square_root_l1356_135607

theorem simplest_square_root : 
  let options : List ℝ := [Real.sqrt 8, Real.sqrt (1/2), Real.sqrt 10, Real.sqrt 1.5]
  ∀ x ∈ options, x ≠ Real.sqrt 10 → ∃ y : ℝ, y * y = (Real.sqrt x) * (Real.sqrt x) ∧ y ≠ Real.sqrt x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_square_root_l1356_135607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_70to79_is_25_58_l1356_135680

-- Define the score ranges
inductive ScoreRange
| Below60
| Range60to69
| Range70to79
| Range80to89
| Range90to100

-- Define the frequency for each range
def frequency (range : ScoreRange) : Nat :=
  match range with
  | ScoreRange.Below60 => 7
  | ScoreRange.Range60to69 => 5
  | ScoreRange.Range70to79 => 11
  | ScoreRange.Range80to89 => 12
  | ScoreRange.Range90to100 => 8

-- Define the total number of students
def totalStudents : Nat :=
  frequency ScoreRange.Below60 +
  frequency ScoreRange.Range60to69 +
  frequency ScoreRange.Range70to79 +
  frequency ScoreRange.Range80to89 +
  frequency ScoreRange.Range90to100

-- Define the percentage of students in the 70%-79% range
noncomputable def percentageIn70to79Range : ℝ :=
  (frequency ScoreRange.Range70to79 : ℝ) / (totalStudents : ℝ) * 100

-- Theorem statement
theorem percentage_70to79_is_25_58 :
  ∃ ε > 0, |percentageIn70to79Range - 25.58| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_70to79_is_25_58_l1356_135680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_difference_l1356_135686

/-- The difference between the area of the region inside a circle but outside an inscribed equilateral triangle and the area of the region inside the triangle but outside the circle -/
theorem circle_triangle_area_difference (r : ℝ) (s : ℝ) : 
  r = 3 → s = 6 → (π * r^2) - (Real.sqrt 3 / 4 * s^2) = 9 * (π - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_difference_l1356_135686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_times_theorem_l1356_135664

/-- Represents a dock in the river system -/
inductive Dock : Type
  | A : Dock
  | B : Dock
  | C : Dock

/-- Represents the direction of travel relative to the current -/
inductive Direction : Type
  | WithCurrent : Direction
  | AgainstCurrent : Direction

/-- The travel time for a given distance and direction -/
noncomputable def travelTime (distance : ℝ) (direction : Direction) : ℝ :=
  match direction with
  | Direction.WithCurrent => (18 / 3) * distance
  | Direction.AgainstCurrent => (30 / 3) * distance

/-- The possible travel times between docks -/
def possibleTravelTimes : Set ℝ := {24, 72}

/-- Theorem stating that the possible travel times between docks are either 24 or 72 minutes -/
theorem travel_times_theorem :
  ∀ (start finish : Dock) (distance : ℝ) (direction : Direction),
    distance > 0 →
    travelTime distance direction ∈ possibleTravelTimes :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_times_theorem_l1356_135664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equality_at_130_degrees_l1356_135619

theorem tan_equality_at_130_degrees :
  let x : ℝ := 130
  0 < x ∧ x < 180 →
  Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x) :=
by
  intro x h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equality_at_130_degrees_l1356_135619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_score_is_80_l1356_135614

def scores : List (ℕ × ℕ) := [(50, 2), (60, 3), (70, 7), (80, 14), (90, 13), (100, 3)]

def total_students : ℕ := 42

theorem median_score_is_80 :
  let cumulative := scores.scanl (fun acc (score, count) => acc + count) 0
  let median_index := total_students / 2
  cumulative.get? median_index = some 26 ∧
  cumulative.get? (median_index - 1) = some 12 →
  scores.get? (median_index - 1) = some (80, 14) := by
  sorry

#eval scores.scanl (fun acc (score, count) => acc + count) 0
#eval total_students / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_score_is_80_l1356_135614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_frustum_lateral_surface_area_l1356_135648

/-- The lateral surface area of a frustum of a right circular cone. -/
noncomputable def frustumLateralSurfaceArea (r1 r2 h : ℝ) : ℝ :=
  Real.pi * (r1 + r2) * Real.sqrt (h^2 + (r1 - r2)^2)

/-- Theorem: The lateral surface area of a specific frustum -/
theorem specific_frustum_lateral_surface_area :
  frustumLateralSurfaceArea 8 2 6 = 60 * Real.sqrt 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_frustum_lateral_surface_area_l1356_135648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_correct_max_k_value_l1356_135647

/-- Sequence a_n -/
noncomputable def a (n : ℕ+) : ℝ := n + 5

/-- Sum of first n terms of a_n -/
noncomputable def S (n : ℕ+) : ℝ := (1/2 : ℝ) * n^2 + (11/2 : ℝ) * n

/-- Sequence b_n -/
noncomputable def b (n : ℕ+) : ℝ := 3 / ((2 * a n - 11) * (2 * a (n + 1) - 11))

/-- Sum of first n terms of b_n -/
noncomputable def T (n : ℕ+) : ℝ := (3 * n) / (2 * n + 1)

/-- The point (n, S_n/n) lies on the line y = (1/2)x + 11/2 -/
axiom point_on_line (n : ℕ+) : S n / n = (1/2 : ℝ) * n + 11/2

theorem a_formula_correct (n : ℕ+) : a n = n + 5 := by sorry

theorem max_k_value : 
  (∃ (k : ℕ), k = 19 ∧ 
    (∀ (n : ℕ+), T n > k / 20) ∧
    (∀ (m : ℕ), m > k → ∃ (n : ℕ+), T n ≤ m / 20)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_correct_max_k_value_l1356_135647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_segment_length_l1356_135625

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ
deriving Inhabited

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents Crystal's run -/
noncomputable def crystalRun : List Point :=
  [⟨0, 0⟩,  -- Starting point
   ⟨0, 2⟩,  -- After going north
   ⟨-Real.sqrt 2, 2 + Real.sqrt 2⟩,  -- After going northwest
   ⟨-2 * Real.sqrt 2, 2⟩]  -- After going southwest

theorem final_segment_length :
  distance (crystalRun.get! 3) (crystalRun.get! 0) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_segment_length_l1356_135625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_length_l1356_135646

/-- Represents a quadrilateral with a diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  (q.diagonal * q.offset1 + q.diagonal * q.offset2) / 2

/-- Theorem: Given a quadrilateral with diagonal 24, offset1 9, and area 180,
    the second offset (offset2) must be 6 -/
theorem second_offset_length (q : Quadrilateral) 
    (h1 : q.diagonal = 24)
    (h2 : q.offset1 = 9)
    (h3 : area q = 180) :
    q.offset2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_length_l1356_135646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_divisibility_impossible_l1356_135628

structure Cube where
  vertices : Fin 8 → ℕ

def initial_cube : Cube where
  vertices := λ i => if i = 0 then 1 else 0

def add_to_edge (c : Cube) (e : Fin 12) : Cube :=
  sorry

def all_divisible_by (c : Cube) (n : ℕ) : Prop :=
  ∀ i, c.vertices i % n = 0

theorem cube_divisibility_impossible :
  ¬ (∃ (ops : List (Fin 12)), 
      let final_cube := ops.foldl add_to_edge initial_cube
      (all_divisible_by final_cube 2) ∨ (all_divisible_by final_cube 3)) :=
sorry

#check cube_divisibility_impossible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_divisibility_impossible_l1356_135628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earnings_left_over_l1356_135690

/-- Represents the percentage of Dhoni's earnings spent on rent -/
noncomputable def rent_percentage : ℝ := 40

/-- Represents the percentage less spent on dishwasher compared to rent -/
noncomputable def dishwasher_percentage_less : ℝ := 20

/-- Calculates the percentage of earnings spent on dishwasher -/
noncomputable def dishwasher_percentage : ℝ := rent_percentage * (1 - dishwasher_percentage_less / 100)

/-- Theorem: Given the rent and dishwasher expenses, the percentage left over is 28% -/
theorem earnings_left_over : 
  100 - (rent_percentage + dishwasher_percentage) = 28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earnings_left_over_l1356_135690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l1356_135627

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The common ratio of a geometric sequence -/
noncomputable def CommonRatio (a : ℕ → ℝ) : ℝ :=
  a 2 / a 1

theorem arithmetic_geometric_ratio (a : ℕ → ℝ) :
  ArithmeticSequence a →
  GeometricSequence (λ n ↦ a (n + 1)) →
  (CommonRatio (λ n ↦ a (n + 1)) = 3 ∨ CommonRatio (λ n ↦ a (n + 1)) = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l1356_135627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_inequality_l1356_135649

/-- The function f(x) = e^x - 1/2 * x^2 - x -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1/2 * x^2 - x

theorem f_minimum_and_inequality (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x ≥ 1) ∧
  (∀ x : ℝ, x ≥ 0 → f x ≥ a * x + 1) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_and_inequality_l1356_135649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sphere_radius_l1356_135666

/-- Represents a truncated cone -/
structure TruncatedCone where
  bottomRadius : ℝ
  topRadius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Represents a generic surface -/
structure Surface where
  dummy : Unit

/-- Predicate to check if a sphere is tangent to a surface -/
def Sphere.isTangentTo (s : Sphere) (surface : Surface) : Prop :=
  sorry

/-- 
Given a truncated cone with specific dimensions, 
this theorem proves that a sphere tangent to its top, bottom, and lateral surface
has a radius of 4.5
-/
theorem tangent_sphere_radius 
  (cone : TruncatedCone)
  (sphere : Sphere)
  (h1 : cone.bottomRadius = 24)
  (h2 : cone.topRadius = 6)
  (h3 : cone.height = 15)
  (h4 : sphere.isTangentTo ⟨()⟩)  -- Representing top surface
  (h5 : sphere.isTangentTo ⟨()⟩)  -- Representing bottom surface
  (h6 : sphere.isTangentTo ⟨()⟩)  -- Representing lateral surface
  : sphere.radius = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sphere_radius_l1356_135666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_inverse_sum_l1356_135623

theorem nested_inverse_sum : ((((3 : ℚ) + 2)⁻¹ + 2)⁻¹ + 2)⁻¹ + 2 = 65 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_inverse_sum_l1356_135623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_to_circle_l1356_135620

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

-- Define the axis of the parabola
noncomputable def parabola_axis (p : ℝ) : ℝ := -p/2

-- Define the condition for a line being tangent to the circle
def is_tangent (line : ℝ → ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ line x = y ∧ 
  ∀ (x' y' : ℝ), circle_eq x' y' → (x' - x)^2 + (y' - y)^2 > 0

-- State the theorem
theorem parabola_tangent_to_circle :
  ∃ (p : ℝ), parabola p 2 0 ∧ is_tangent (λ x ↦ parabola_axis p) → p = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_to_circle_l1356_135620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_implies_k_range_l1356_135667

-- Define the line y = kx + 1
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the circle (x-1)^2 + (y-1)^2 = 1
def circleEq (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem intersection_distance_implies_k_range (k : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    circleEq x1 y1 ∧ circleEq x2 y2 ∧
    y1 = line k x1 ∧ y2 = line k x2 ∧
    distance x1 y1 x2 y2 ≥ Real.sqrt 2) →
  -1 ≤ k ∧ k ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_implies_k_range_l1356_135667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1356_135681

theorem problem_solution (a b m : ℝ) 
  (h1 : (1/2:ℝ)^a = 3^b)
  (h2 : (1/2:ℝ)^a = m)
  (h3 : 3^b = m)
  (h4 : 1/a - 1/b = 2) :
  m = Real.sqrt 6 / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1356_135681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_given_two_solutions_main_theorem_l1356_135608

-- Define the function f(x) = |2^x - a|
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |2^x - a|

-- State the theorem
theorem range_of_a_given_two_solutions (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 1 ∧ f a x₂ = 1) →
  a > 1 ∧ ∀ y > 1, ∃ x : ℝ, f a x = 1 :=
by sorry

-- Define the range of a
def range_of_a : Set ℝ := {y : ℝ | y > 1}

-- State the main theorem
theorem main_theorem :
  {a : ℝ | ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 1 ∧ f a x₂ = 1} = range_of_a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_given_two_solutions_main_theorem_l1356_135608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1356_135629

/-- For any triangle ABC with circumradius R and inradius r, 
    the following inequality holds. -/
theorem triangle_inequality (A B C R r : ℝ) : 
  0 < R ∧ 0 < r ∧ 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  (3 * r) / (2 * R) ≤ 
    Real.sin (A/2) + Real.sin (B/2) + Real.sin (C/2) - 
    (Real.sin (A/2))^2 - (Real.sin (B/2))^2 - (Real.sin (C/2))^2 ∧
  Real.sin (A/2) + Real.sin (B/2) + Real.sin (C/2) - 
    (Real.sin (A/2))^2 - (Real.sin (B/2))^2 - (Real.sin (C/2))^2 ≤ 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1356_135629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_ab_fixed_point_l1356_135660

/-- Ellipse properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Given ellipse with specific properties -/
noncomputable def given_ellipse : Ellipse where
  a := Real.sqrt 5
  b := 1
  h := by
    constructor
    · sorry -- Proof that √5 > 1
    · exact Real.zero_lt_one

/-- Eccentricity of the ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 2 * Real.sqrt 5 / 5

/-- Maximum area of triangle AF₁F₂ -/
def max_triangle_area : ℝ := 2

/-- Theorem: Equation of the ellipse -/
theorem ellipse_equation (e : Ellipse) (x y : ℝ) :
  (x^2 / e.a^2) + (y^2 / e.b^2) = 1 ↔ (x^2 / 5) + y^2 = 1 := by
  sorry

/-- Theorem: Line AB passes through a fixed point -/
theorem line_ab_fixed_point (e : Ellipse) (A B : ℝ × ℝ) :
  A.1^2 / 5 + A.2^2 = 1 →
  B.1^2 / 5 + B.2^2 = 1 →
  A.2 * B.2 > 0 →
  ∃ (k : ℝ), A.2 - B.2 = k * (A.1 - B.1) →
  ∃ (t : ℝ), A.2 - B.2 = k * (A.1 - B.1) ∧ 0 = k * (2.5 - t) + (A.2 - k * A.1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_ab_fixed_point_l1356_135660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_at_10_l1356_135684

/-- An arithmetic sequence with common difference d and first term a1 -/
noncomputable def arithmetic_sequence (d : ℝ) (a1 : ℝ) (n : ℕ) : ℝ := a1 + d * (n - 1)

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (d : ℝ) (a1 : ℝ) (n : ℕ) : ℝ := n * a1 + n * (n - 1) / 2 * d

/-- Theorem stating that the sum is largest when n = 10 -/
theorem largest_sum_at_10 (d : ℝ) (a1 : ℝ) (h1 : d < 0) (h2 : S d a1 8 = S d a1 12) :
  ∀ n : ℕ, S d a1 10 ≥ S d a1 n := by
  sorry

#check largest_sum_at_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_at_10_l1356_135684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_matrix_relation_l1356_135638

-- Define the Fibonacci sequence
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- Define the matrix A
def A (n : ℕ) : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![fib (n + 2), fib (n + 1), fib n],
    ![fib (n + 1), fib n, fib (n - 1)],
    ![fib n, fib (n - 1), fib (n - 2)]]

-- Define the original matrix
def M : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 1, 0],
    ![1, 0, 1],
    ![0, 1, 1]]

theorem fibonacci_matrix_relation (n : ℕ) (h : n > 0) :
  M ^ n = A n ∧ Matrix.det (A n) = (-1 : ℤ) ^ (n + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_matrix_relation_l1356_135638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_exists_l1356_135617

/-- A hyperbola with equation x² - y²/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The circle with equation (x-2)² + y² = 3 -/
def circle' (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 3

/-- The asymptote of the hyperbola -/
def asymptote (x y : ℝ) : Prop := x = y/Real.sqrt 3 ∨ x = -y/Real.sqrt 3

/-- Theorem stating the existence of a hyperbola satisfying the given conditions -/
theorem hyperbola_exists : 
  ∃ (c : ℝ), 
    (∀ x y, hyperbola x y → (x = c ∨ x = -c)) ∧ 
    (∃ x y, asymptote x y ∧ circle' x y) := by
  sorry

#check hyperbola_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_exists_l1356_135617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_volume_ratio_l1356_135676

/-- The ratio of the volume of a regular n-gon pyramid to the volume of its circumscribed sphere -/
noncomputable def volume_ratio (n : ℕ) (α : ℝ) : ℝ :=
  (n * Real.sin α ^ 2 * Real.sin (2 * α) ^ 2 * Real.sin (2 * Real.pi / n)) / (4 * Real.pi)

/-- Theorem stating the ratio of volumes for a regular n-gon pyramid and its circumscribed sphere -/
theorem pyramid_sphere_volume_ratio (n : ℕ) (α : ℝ) :
  let pyramid_volume := (1 / 3) * n * Real.sin α ^ 2 * Real.sin (2 * α) ^ 2 * Real.sin (2 * Real.pi / n)
  let sphere_volume := (4 / 3) * Real.pi
  pyramid_volume / sphere_volume = volume_ratio n α := by
  sorry

#check pyramid_sphere_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_volume_ratio_l1356_135676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_catch_up_distance_l1356_135616

/-- The distance Timothy's mother must drive to catch up with him -/
noncomputable def catch_up_distance (timothy_speed : ℝ) (mother_speed : ℝ) (delay : ℝ) : ℝ :=
  let initial_distance := timothy_speed * delay
  let catch_up_time := initial_distance / (mother_speed - timothy_speed)
  mother_speed * catch_up_time

theorem mother_catch_up_distance :
  catch_up_distance 6 36 (1/4) = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_catch_up_distance_l1356_135616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1356_135668

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 
  Real.sqrt (a * (Real.cos x)^2 + b * (Real.sin x)^2) + 
  Real.sqrt (a * (Real.sin x)^2 + b * (Real.cos x)^2)

-- State the theorem
theorem f_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  ∀ x, Real.sqrt a + Real.sqrt b ≤ f a b x ∧ f a b x ≤ Real.sqrt (2 * (a + b)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1356_135668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_vessel_problem_l1356_135654

/-- Represents the properties of a cylindrical vessel and its environment --/
structure CylindricalVessel where
  radius : ℝ
  height : ℝ
  initialTemp : ℝ
  finalTemp : ℝ
  atmPressure : ℝ
  mercuryDensity : ℝ
  airExpansionCoeff : ℝ

/-- Calculates the force on the glass plate and the density ratio --/
noncomputable def calculateForceAndDensityRatio (vessel : CylindricalVessel) : ℝ × ℝ :=
  sorry

/-- Theorem statement for the cylindrical vessel problem --/
theorem cylindrical_vessel_problem (vessel : CylindricalVessel)
  (h_radius : vessel.radius = 0.08)
  (h_height : vessel.height = 0.2416)
  (h_initialTemp : vessel.initialTemp = 1473)  -- in Kelvin
  (h_finalTemp : vessel.finalTemp = 289)       -- in Kelvin
  (h_atmPressure : vessel.atmPressure = 745.6)
  (h_mercuryDensity : vessel.mercuryDensity = 13.56)
  (h_airExpansionCoeff : vessel.airExpansionCoeff = 1 / 273) :
  let (force, densityRatio) := calculateForceAndDensityRatio vessel
  (abs (force - 16.64) < 0.01) ∧ (abs (densityRatio - 0.196) < 0.001) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_vessel_problem_l1356_135654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soft_drink_pack_contains_11_cans_l1356_135644

/-- The number of cans in a pack of soft drinks -/
def cans_in_pack (pack_cost : ℚ) (can_cost : ℚ) : ℕ :=
  (pack_cost / can_cost).floor.toNat

/-- Theorem stating that a pack costing $2.99 with individual cans costing $0.25 contains 11 cans -/
theorem soft_drink_pack_contains_11_cans :
  cans_in_pack (299/100) (1/4) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soft_drink_pack_contains_11_cans_l1356_135644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1356_135645

-- Define the triangle ABC
noncomputable def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Add any necessary conditions for a valid triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the condition given in the problem
noncomputable def TriangleCondition (A B C : ℝ) (a b c : ℝ) : Prop :=
  (2 * a - c) * Real.cos B - b * Real.cos C = 0

-- Define the function f(x)
noncomputable def f (x B : ℝ) : ℝ :=
  2 * Real.sin x * Real.cos x * Real.cos B - (Real.sqrt 3 / 2) * Real.cos (2 * x)

-- State the theorem
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h1 : Triangle A B C a b c) (h2 : TriangleCondition A B C a b c) :
  (B = π / 3) ∧ 
  (∀ x, f x B ≤ 1) ∧
  (∃ S : Set ℝ, S = {x | ∃ k : ℤ, x = k * π + 5 * π / 12} ∧ 
    ∀ x, x ∈ S ↔ f x B = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1356_135645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_implies_x_values_l1356_135630

-- Define the vector type
def MyVector := ℝ × ℝ

-- Define the vectors a and b
def a (x : ℝ) : MyVector := (x, 3 - x)
def b (x : ℝ) : MyVector := (-1, 3 - x)

-- Define parallel vectors
def parallel (v w : MyVector) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

-- State the theorem
theorem vectors_parallel_implies_x_values (x : ℝ) :
  parallel (a x) (b x) → x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_implies_x_values_l1356_135630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1356_135695

noncomputable def integerPart (x : ℝ) : ℤ := Int.floor x

noncomputable def fractionalPart (x : ℝ) : ℝ := x - (Int.floor x : ℝ)

theorem equation_solution (x : ℝ) (h : x > 1) :
  (1 / (integerPart x : ℝ) + 1 / fractionalPart x = x) ↔
  ∃ (n : ℕ), n > 1 ∧ x = n + 1 / n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1356_135695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_12345_is_integer_with_units_digit_4_l1356_135621

noncomputable def c : ℝ := 4 + Real.sqrt 15
noncomputable def d : ℝ := 4 - Real.sqrt 15

def S : ℕ → ℝ
  | 0 => 1
  | 1 => 4
  | (n + 2) => 8 * S (n + 1) - S n

theorem S_12345_is_integer_with_units_digit_4 :
  ∃ (k : ℤ), S 12345 = k ∧ k % 10 = 4 := by
  sorry

#eval S 5  -- This line is added to check if the function works for small values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_12345_is_integer_with_units_digit_4_l1356_135621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1356_135636

/-- The time (in seconds) for a train to pass a person moving in the opposite direction -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) : ℝ :=
  train_length / ((train_speed + person_speed) * (5 / 18))

/-- Theorem stating that the time for a 220m train moving at 60 km/hr to pass a person
    moving at 6 km/hr in the opposite direction is approximately 12 seconds -/
theorem train_passing_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |train_passing_time 220 60 6 - 12| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l1356_135636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_container_volume_l1356_135669

/-- The volume of a conical container formed from a sector of a square sheet -/
theorem conical_container_volume 
  (side_length : ℝ) 
  (central_angle : ℝ) 
  (h : side_length = 8) 
  (θ : central_angle = π/4) : 
  (1/3) * π * ((central_angle * side_length) / (2 * π))^2 * 
  Real.sqrt (side_length^2 - ((central_angle * side_length) / (2 * π))^2) = Real.sqrt 7 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_container_volume_l1356_135669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_speed_theorem_l1356_135632

/-- Represents the speed of a man rowing across three streams with wind --/
noncomputable def effective_speed (with_stream : ℝ) (against_stream : ℝ) (stream1 : ℝ) (stream2 : ℝ) (stream3 : ℝ) (wind : ℝ) : ℝ :=
  let speed1 := with_stream + stream1 - wind
  let speed2 := against_stream - stream2 - wind
  let speed3 := with_stream + stream3 - wind
  (speed1 + speed2 + speed3) / 3

/-- Theorem stating that the effective speed is 13 km/h given the problem conditions --/
theorem rowing_speed_theorem :
  effective_speed 16 6 2 (-1) 3 1 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_speed_theorem_l1356_135632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_circles_theorem_l1356_135698

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle in 2D space
structure Circle where
  center : Point2D
  radius : ℝ

-- Define a function to check if three points are collinear
def collinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

-- Define a function to check if two circles intersect at right angles
def perpendicular_intersection (c1 c2 : Circle) : Prop :=
  sorry -- Definition of perpendicular intersection

-- Define a function to calculate the angle at the base of an isosceles triangle
noncomputable def base_angle (center : Point2D) (p1 p2 : Point2D) : ℝ :=
  sorry -- Calculation of base angle

-- Define a membership relation for points on a circle
def on_circle (p : Point2D) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Main theorem
theorem perpendicular_circles_theorem (A B C : Point2D) :
  ¬collinear A B C →
  ∃ (c1 c2 c3 : Circle),
    (c1.center ≠ A ∧ c1.center ≠ B ∧ c1.center ≠ C) ∧
    (c2.center ≠ A ∧ c2.center ≠ B ∧ c2.center ≠ C) ∧
    (c3.center ≠ A ∧ c3.center ≠ B ∧ c3.center ≠ C) ∧
    (on_circle A c2 ∧ on_circle A c3) ∧
    (on_circle B c1 ∧ on_circle B c3) ∧
    (on_circle C c1 ∧ on_circle C c2) ∧
    perpendicular_intersection c1 c2 ∧
    perpendicular_intersection c2 c3 ∧
    perpendicular_intersection c3 c1 ↔
    base_angle c1.center B C +
    base_angle c2.center A C +
    base_angle c3.center A B = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_circles_theorem_l1356_135698
