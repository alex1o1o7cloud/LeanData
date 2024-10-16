import Mathlib

namespace NUMINAMATH_CALUDE_division_remainder_problem_l1888_188870

theorem division_remainder_problem 
  (P D Q R D' Q' R' C : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R')
  (h3 : R < D)
  (h4 : R' < D') :
  P % ((D + C) * D') = D' * C * R' + D * R' + C * R' + R := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1888_188870


namespace NUMINAMATH_CALUDE_eighth_root_of_390625000000000_l1888_188881

theorem eighth_root_of_390625000000000 : (390625000000000 : ℝ) ^ (1/8 : ℝ) = 101 := by
  sorry

end NUMINAMATH_CALUDE_eighth_root_of_390625000000000_l1888_188881


namespace NUMINAMATH_CALUDE_f_second_derivative_at_zero_l1888_188876

-- Define the function f
def f (x : ℝ) (f''_1 : ℝ) : ℝ := x^3 - 2 * x * f''_1

-- State the theorem
theorem f_second_derivative_at_zero (f''_1 : ℝ) : 
  (deriv (deriv (f · f''_1))) 0 = -2 :=
sorry

end NUMINAMATH_CALUDE_f_second_derivative_at_zero_l1888_188876


namespace NUMINAMATH_CALUDE_crypto_encoding_l1888_188815

/-- Represents the encoding of digits in the cryptographic system -/
inductive Digit
| A
| B
| C
| D

/-- Converts a Digit to its corresponding base-4 value -/
def digit_to_base4 : Digit → Nat
| Digit.A => 3
| Digit.B => 1
| Digit.C => 0
| Digit.D => 2

/-- Converts a three-digit code to its base-10 value -/
def code_to_base10 (d₁ d₂ d₃ : Digit) : Nat :=
  16 * (digit_to_base4 d₁) + 4 * (digit_to_base4 d₂) + (digit_to_base4 d₃)

/-- The main theorem stating the result of the cryptographic encoding -/
theorem crypto_encoding :
  code_to_base10 Digit.B Digit.C Digit.D + 1 = code_to_base10 Digit.B Digit.D Digit.A ∧
  code_to_base10 Digit.B Digit.D Digit.A + 1 = code_to_base10 Digit.B Digit.C Digit.A →
  code_to_base10 Digit.D Digit.A Digit.C = 44 :=
by sorry

end NUMINAMATH_CALUDE_crypto_encoding_l1888_188815


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_l1888_188882

-- Define the basic geometric objects
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the geometric relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_to_line_are_parallel
  (α β : Plane) (m : Line) (h_diff : α ≠ β) :
  perpendicular m α → perpendicular m β → parallel α β := by sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_l1888_188882


namespace NUMINAMATH_CALUDE_peanut_distribution_theorem_l1888_188873

/-- Represents the distribution of peanuts among three people -/
structure PeanutDistribution where
  alex : ℕ
  betty : ℕ
  charlie : ℕ

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r > 0 ∧ b = a * r ∧ c = b * r

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression (a b c : ℕ) : Prop :=
  ∃ d : ℤ, b = a + d ∧ c = b + d

/-- The main theorem about the peanut distribution -/
theorem peanut_distribution_theorem (init : PeanutDistribution) 
  (h_total : init.alex + init.betty + init.charlie = 444)
  (h_order : init.alex < init.betty ∧ init.betty < init.charlie)
  (h_geometric : is_geometric_progression init.alex init.betty init.charlie)
  (final : PeanutDistribution)
  (h_eating : final.alex = init.alex - 5 ∧ final.betty = init.betty - 9 ∧ final.charlie = init.charlie - 25)
  (h_arithmetic : is_arithmetic_progression final.alex final.betty final.charlie) :
  init.alex = 108 := by
  sorry

end NUMINAMATH_CALUDE_peanut_distribution_theorem_l1888_188873


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1888_188827

theorem rectangular_to_polar_conversion :
  let x : ℝ := 3
  let y : ℝ := -3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 ∧ y < 0 then 2 * π + Real.arctan (y / x) else Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π →
  r = 3 * Real.sqrt 2 ∧ θ = 7 * π / 4 := by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1888_188827


namespace NUMINAMATH_CALUDE_combinations_equal_twenty_l1888_188880

/-- The number of available colors -/
def num_colors : ℕ := 5

/-- The number of available painting methods -/
def num_methods : ℕ := 4

/-- The total number of combinations of colors and painting methods -/
def total_combinations : ℕ := num_colors * num_methods

/-- Theorem stating that the total number of combinations is 20 -/
theorem combinations_equal_twenty : total_combinations = 20 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_twenty_l1888_188880


namespace NUMINAMATH_CALUDE_binomial_expansion_terms_l1888_188895

theorem binomial_expansion_terms (n : ℕ+) : 
  (Finset.range (2*n + 1)).card = 2*n + 1 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_terms_l1888_188895


namespace NUMINAMATH_CALUDE_cost_per_block_l1888_188898

/-- Proves that the cost per piece of concrete block is $2 -/
theorem cost_per_block (blocks_per_section : ℕ) (num_sections : ℕ) (total_cost : ℚ) :
  blocks_per_section = 30 →
  num_sections = 8 →
  total_cost = 480 →
  total_cost / (blocks_per_section * num_sections) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_block_l1888_188898


namespace NUMINAMATH_CALUDE_g_evaluation_l1888_188819

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem g_evaluation : 3 * g 2 + 4 * g (-4) = 327 := by
  sorry

end NUMINAMATH_CALUDE_g_evaluation_l1888_188819


namespace NUMINAMATH_CALUDE_work_completion_days_l1888_188803

/-- Calculates the initial number of days planned to complete a work given the total number of men,
    number of absent men, and the number of days taken by the remaining men. -/
def initialDays (totalMen : ℕ) (absentMen : ℕ) (daysWithAbsent : ℕ) : ℕ :=
  (totalMen - absentMen) * daysWithAbsent / totalMen

/-- Proves that given 20 men where 10 become absent and the remaining 10 complete the work in 40 days,
    the original plan was to complete the work in 20 days. -/
theorem work_completion_days :
  initialDays 20 10 40 = 20 := by
  sorry

#eval initialDays 20 10 40

end NUMINAMATH_CALUDE_work_completion_days_l1888_188803


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l1888_188824

/-- Given two parallel vectors a and b, prove that x + y = -9 -/
theorem parallel_vectors_sum (x y : ℝ) :
  let a : Fin 3 → ℝ := ![-1, 2, 1]
  let b : Fin 3 → ℝ := ![3, x, y]
  (∃ (k : ℝ), ∀ i, b i = k * (a i)) →
  x + y = -9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l1888_188824


namespace NUMINAMATH_CALUDE_pages_printed_for_fifty_dollars_l1888_188859

/-- Given the cost of 9 cents for 7 pages, prove that the maximum number of whole pages
    that can be printed for $50 is 3888. -/
theorem pages_printed_for_fifty_dollars (cost_per_seven_pages : ℚ) 
  (h1 : cost_per_seven_pages = 9/100) : 
  ⌊(50 * 100 * 7) / (cost_per_seven_pages * 7)⌋ = 3888 := by
  sorry

end NUMINAMATH_CALUDE_pages_printed_for_fifty_dollars_l1888_188859


namespace NUMINAMATH_CALUDE_function_properties_l1888_188822

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (2 * ω * x - Real.pi / 6) + 2 * (Real.cos (ω * x))^2 - 1

theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x, f ω (x + Real.pi / ω) = f ω x) : 
  ω = 1 ∧ 
  (∀ x ∈ Set.Icc 0 (7 * Real.pi / 12), f ω x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 (7 * Real.pi / 12), f ω x ≥ -Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 (7 * Real.pi / 12), f ω x = 1) ∧
  (∃ x ∈ Set.Icc 0 (7 * Real.pi / 12), f ω x = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1888_188822


namespace NUMINAMATH_CALUDE_consecutive_non_primes_l1888_188810

theorem consecutive_non_primes (n : ℕ) : 
  ∃ k : ℕ, ∀ i : ℕ, i ∈ Finset.range n → ¬ Prime (k + i + 2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_non_primes_l1888_188810


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1888_188851

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1888_188851


namespace NUMINAMATH_CALUDE_bikes_in_parking_lot_l1888_188867

theorem bikes_in_parking_lot :
  let num_cars : ℕ := 10
  let total_wheels : ℕ := 44
  let wheels_per_car : ℕ := 4
  let wheels_per_bike : ℕ := 2
  let num_bikes : ℕ := (total_wheels - num_cars * wheels_per_car) / wheels_per_bike
  num_bikes = 2 := by sorry

end NUMINAMATH_CALUDE_bikes_in_parking_lot_l1888_188867


namespace NUMINAMATH_CALUDE_projection_magnitude_l1888_188874

def a : Fin 2 → ℝ := ![1, -1]
def b : Fin 2 → ℝ := ![2, -1]

theorem projection_magnitude :
  ‖((a + b) • a / (a • a)) • a‖ = (5 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_projection_magnitude_l1888_188874


namespace NUMINAMATH_CALUDE_max_odd_sums_for_given_range_l1888_188838

def max_odd_sums (n : ℕ) (start : ℕ) : ℕ :=
  if n ≤ 2 then 0
  else ((n - 2) / 2) + 1

theorem max_odd_sums_for_given_range :
  max_odd_sums 998 1000 = 499 := by
  sorry

end NUMINAMATH_CALUDE_max_odd_sums_for_given_range_l1888_188838


namespace NUMINAMATH_CALUDE_circle_diameter_points_exist_l1888_188858

/-- Represents a point on the circumference of a circle -/
structure CirclePoint where
  angle : ℝ
  property : 0 ≤ angle ∧ angle < 2 * Real.pi

/-- Represents an arc on the circumference of a circle -/
structure Arc where
  start : CirclePoint
  length : ℝ
  property : 0 < length ∧ length ≤ 2 * Real.pi

/-- The main theorem statement -/
theorem circle_diameter_points_exist (k : ℕ) (points : Finset CirclePoint) (arcs : Finset Arc) :
  points.card = 3 * k →
  arcs.card = 3 * k →
  (∃ (s₁ : Finset Arc), s₁.card = k ∧ ∀ a ∈ s₁, a.length = 1) →
  (∃ (s₂ : Finset Arc), s₂.card = k ∧ ∀ a ∈ s₂, a.length = 2) →
  (∃ (s₃ : Finset Arc), s₃.card = k ∧ ∀ a ∈ s₃, a.length = 3) →
  ∃ (p₁ p₂ : CirclePoint), p₁ ∈ points ∧ p₂ ∈ points ∧ abs (p₁.angle - p₂.angle) = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_points_exist_l1888_188858


namespace NUMINAMATH_CALUDE_max_m_value_l1888_188866

theorem max_m_value (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + m*x₁ + 6 = 0 ∧ x₂^2 + m*x₂ + 6 = 0 ∧ |x₁ - x₂| = Real.sqrt 85) →
  m ≤ Real.sqrt 109 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l1888_188866


namespace NUMINAMATH_CALUDE_weight_replacement_l1888_188802

theorem weight_replacement (n : ℕ) (old_weight new_weight avg_increase : ℝ) :
  n = 8 →
  new_weight = 95 →
  avg_increase = 2.5 →
  old_weight = new_weight - n * avg_increase →
  old_weight = 75 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l1888_188802


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1888_188893

theorem line_passes_through_fixed_point (a b : ℝ) (h : a + 2 * b = 1) :
  ∃ (x y : ℝ), x = 1/2 ∧ y = -1/6 ∧ a * x + 3 * y + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1888_188893


namespace NUMINAMATH_CALUDE_valid_B_l1888_188848

-- Define set A
def A : Set ℝ := {x | x ≥ 0}

-- Define the property that A ∩ B = B
def intersectionProperty (B : Set ℝ) : Prop := A ∩ B = B

-- Define the set {1,2}
def candidateB : Set ℝ := {1, 2}

-- Theorem statement
theorem valid_B : intersectionProperty candidateB := by sorry

end NUMINAMATH_CALUDE_valid_B_l1888_188848


namespace NUMINAMATH_CALUDE_cu_atom_count_l1888_188837

/-- Represents the number of atoms of an element in a compound -/
structure AtomCount where
  count : ℕ

/-- Represents the atomic weight of an element in amu -/
structure AtomicWeight where
  weight : ℝ

/-- Represents a chemical compound -/
structure Compound where
  cu : AtomCount
  c : AtomCount
  o : AtomCount
  molecularWeight : ℝ

def cuAtomicWeight : AtomicWeight := ⟨63.55⟩
def cAtomicWeight : AtomicWeight := ⟨12.01⟩
def oAtomicWeight : AtomicWeight := ⟨16.00⟩

def compoundWeight (cpd : Compound) : ℝ :=
  cpd.cu.count * cuAtomicWeight.weight +
  cpd.c.count * cAtomicWeight.weight +
  cpd.o.count * oAtomicWeight.weight

theorem cu_atom_count (cpd : Compound) :
  cpd.c = ⟨1⟩ ∧ cpd.o = ⟨3⟩ ∧ cpd.molecularWeight = 124 →
  cpd.cu = ⟨1⟩ :=
by sorry

end NUMINAMATH_CALUDE_cu_atom_count_l1888_188837


namespace NUMINAMATH_CALUDE_rebus_solution_exists_and_unique_l1888_188807

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_distinct (a b c d e f g h i j : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

theorem rebus_solution_exists_and_unique :
  ∃! (a b c d e f g h i j : ℕ),
    is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
    is_valid_digit d ∧ is_valid_digit e ∧ is_valid_digit f ∧
    is_valid_digit g ∧ is_valid_digit h ∧ is_valid_digit i ∧ is_valid_digit j ∧
    are_distinct a b c d e f g h i j ∧
    100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 * g + 100 * h + 10 * i + j :=
sorry

end NUMINAMATH_CALUDE_rebus_solution_exists_and_unique_l1888_188807


namespace NUMINAMATH_CALUDE_rectangle_ribbon_length_l1888_188836

/-- The length of ribbon required to form a rectangle -/
def ribbon_length (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: The length of ribbon required to form a rectangle with length 20 feet and width 15 feet is 70 feet -/
theorem rectangle_ribbon_length : 
  ribbon_length 20 15 = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ribbon_length_l1888_188836


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1888_188833

/-- Given a circle D with equation x^2 + 10x + 2y^2 - 8y = 18,
    prove that the sum of its center coordinates and radius is -3 + √38 -/
theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), 
    (∀ (x y : ℝ), x^2 + 10*x + 2*y^2 - 8*y = 18 ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
    a + b + r = -3 + Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1888_188833


namespace NUMINAMATH_CALUDE_namjoon_books_l1888_188813

/-- The number of books Namjoon has in total -/
def total_books (a b c : ℕ) : ℕ := a + b + c

/-- Theorem stating the total number of books Namjoon has -/
theorem namjoon_books :
  ∀ (a b c : ℕ),
  a = 35 →
  b = a - 16 →
  c = b + 35 →
  total_books a b c = 108 := by
  sorry

end NUMINAMATH_CALUDE_namjoon_books_l1888_188813


namespace NUMINAMATH_CALUDE_window_length_l1888_188820

/-- Given a rectangular window with width 10 feet and area 60 square feet, its length is 6 feet -/
theorem window_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 10 → area = 60 → area = length * width → length = 6 := by
  sorry

end NUMINAMATH_CALUDE_window_length_l1888_188820


namespace NUMINAMATH_CALUDE_quadrilateral_division_theorem_l1888_188887

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  /-- The sum of internal angles is 360 degrees -/
  angle_sum : ℝ
  angle_sum_eq : angle_sum = 360

/-- A diagonal of a quadrilateral -/
structure Diagonal (Q : ConvexQuadrilateral) where
  /-- The diagonal divides the quadrilateral into two triangles -/
  divides_into_triangles : Prop

/-- A triangle formed by a diagonal in a quadrilateral -/
structure Triangle (Q : ConvexQuadrilateral) (D : Diagonal Q) where
  /-- The sum of angles in the triangle is 180 degrees -/
  angle_sum : ℝ
  angle_sum_eq : angle_sum = 180

/-- Theorem: In a convex quadrilateral, it's impossible to divide it by a diagonal into two acute triangles, 
    while it's possible to divide it into two right triangles or two obtuse triangles -/
theorem quadrilateral_division_theorem (Q : ConvexQuadrilateral) :
  (∃ D : Diagonal Q, ∃ T1 T2 : Triangle Q D, 
    T1.angle_sum < 180 ∧ T2.angle_sum < 180) → False
  ∧ (∃ D : Diagonal Q, ∃ T1 T2 : Triangle Q D, 
    T1.angle_sum = 180 ∧ T2.angle_sum = 180)
  ∧ (∃ D : Diagonal Q, ∃ T1 T2 : Triangle Q D, 
    T1.angle_sum > 180 ∧ T2.angle_sum > 180) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_division_theorem_l1888_188887


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l1888_188835

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (x^2 - 9) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l1888_188835


namespace NUMINAMATH_CALUDE_square_root_equation_l1888_188845

theorem square_root_equation (x : ℝ) :
  Real.sqrt (3 * x + 4) = 12 → x = 140 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l1888_188845


namespace NUMINAMATH_CALUDE_max_handshakes_l1888_188894

theorem max_handshakes (n : ℕ) (h : n = 25) : 
  (n * (n - 1)) / 2 = 300 := by
  sorry

#check max_handshakes

end NUMINAMATH_CALUDE_max_handshakes_l1888_188894


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_l1888_188846

theorem roots_sum_reciprocal (m n : ℝ) : 
  m^2 + 2*m - 3 = 0 → n^2 + 2*n - 3 = 0 → 1/m + 1/n = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_l1888_188846


namespace NUMINAMATH_CALUDE_quadratic_expression_rewrite_l1888_188871

theorem quadratic_expression_rewrite :
  ∃ (c p q : ℚ),
    (∀ k, 8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q) ∧
    q / p = -142 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_rewrite_l1888_188871


namespace NUMINAMATH_CALUDE_river_current_speed_l1888_188899

/-- Theorem: Given a swimmer's speed in still water and the ratio of upstream to downstream swimming time, we can determine the speed of the river's current. -/
theorem river_current_speed 
  (swimmer_speed : ℝ) 
  (upstream_downstream_ratio : ℝ) 
  (h1 : swimmer_speed = 10) 
  (h2 : upstream_downstream_ratio = 3) : 
  ∃ (current_speed : ℝ), current_speed = 5 ∧ 
  (swimmer_speed + current_speed) * upstream_downstream_ratio = 
  (swimmer_speed - current_speed) * (upstream_downstream_ratio + 1) := by
  sorry

end NUMINAMATH_CALUDE_river_current_speed_l1888_188899


namespace NUMINAMATH_CALUDE_smallest_value_theorem_l1888_188861

theorem smallest_value_theorem (a b : ℕ+) (h : a.val^2 - b.val^2 = 16) :
  (∀ (c d : ℕ+), c.val^2 - d.val^2 = 16 →
    (a.val + b.val : ℚ) / (a.val - b.val : ℚ) + (a.val - b.val : ℚ) / (a.val + b.val : ℚ) ≤
    (c.val + d.val : ℚ) / (c.val - d.val : ℚ) + (c.val - d.val : ℚ) / (c.val + d.val : ℚ)) ∧
  (a.val + b.val : ℚ) / (a.val - b.val : ℚ) + (a.val - b.val : ℚ) / (a.val + b.val : ℚ) = 9/4 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_theorem_l1888_188861


namespace NUMINAMATH_CALUDE_bob_works_five_days_per_week_l1888_188885

/-- Represents the number of hours Bob works per day -/
def daily_hours : ℕ := 10

/-- Represents the total number of hours Bob works in a month -/
def monthly_hours : ℕ := 200

/-- Represents the average number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Calculates the number of days Bob works per week -/
def days_worked_per_week : ℚ :=
  (monthly_hours / daily_hours) / weeks_per_month

theorem bob_works_five_days_per_week :
  days_worked_per_week = 5 := by sorry

end NUMINAMATH_CALUDE_bob_works_five_days_per_week_l1888_188885


namespace NUMINAMATH_CALUDE_train_distance_l1888_188831

/-- Proves that a train traveling at 3 m/s for 9 seconds covers 27 meters -/
theorem train_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 3 → time = 9 → distance = speed * time → distance = 27 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_l1888_188831


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1888_188826

/-- 
In a Cartesian coordinate system, the coordinates of a point (2, -3) 
with respect to the origin are (2, -3).
-/
theorem point_coordinates_wrt_origin : 
  let point : ℝ × ℝ := (2, -3)
  point = (2, -3) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1888_188826


namespace NUMINAMATH_CALUDE_subset_implies_a_nonpositive_l1888_188865

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + a ≤ 0}
def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}

-- Theorem statement
theorem subset_implies_a_nonpositive (a : ℝ) (h : B ⊆ A a) : a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_nonpositive_l1888_188865


namespace NUMINAMATH_CALUDE_logarithm_inequality_and_root_comparison_l1888_188818

theorem logarithm_inequality_and_root_comparison : 
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log (((a + b) / 2) : ℝ) ≥ (Real.log a + Real.log b) / 2) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_inequality_and_root_comparison_l1888_188818


namespace NUMINAMATH_CALUDE_joan_has_77_balloons_l1888_188853

/-- The number of balloons Joan has after giving some away and receiving more -/
def joans_balloons (initial_blue initial_red mark_blue mark_red sarah_blue additional_red : ℕ) : ℕ :=
  (initial_blue - mark_blue - sarah_blue) + (initial_red - mark_red + additional_red)

/-- Theorem stating that Joan has 77 balloons given the problem conditions -/
theorem joan_has_77_balloons :
  joans_balloons 72 48 15 10 24 6 = 77 := by
  sorry

#eval joans_balloons 72 48 15 10 24 6

end NUMINAMATH_CALUDE_joan_has_77_balloons_l1888_188853


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l1888_188821

theorem constant_ratio_problem (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) (k : ℝ) :
  (3 * x₁ - 4) / (y₁ + 15) = k →
  (3 * x₂ - 4) / (y₂ + 15) = k →
  x₁ = 2 →
  y₁ = 3 →
  y₂ = 12 →
  x₂ = 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l1888_188821


namespace NUMINAMATH_CALUDE_nabla_problem_l1888_188891

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^(a - 1)

-- State the theorem
theorem nabla_problem : nabla (nabla 2 3) 4 = 1027 := by
  sorry

end NUMINAMATH_CALUDE_nabla_problem_l1888_188891


namespace NUMINAMATH_CALUDE_quadrilateral_rod_count_quadrilateral_rod_count_is_17_l1888_188883

theorem quadrilateral_rod_count : ℕ → Prop :=
  fun n =>
    let rods : Finset ℕ := Finset.range 30
    let used_rods : Finset ℕ := {3, 7, 15}
    let valid_rods : Finset ℕ := 
      rods.filter (fun x => 
        x > 5 ∧ x < 25 ∧ x ∉ used_rods)
    n = valid_rods.card

theorem quadrilateral_rod_count_is_17 :
  quadrilateral_rod_count 17 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_rod_count_quadrilateral_rod_count_is_17_l1888_188883


namespace NUMINAMATH_CALUDE_unique_plane_through_skew_lines_l1888_188825

/-- Two lines in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a simplified representation

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields to define a plane in 3D space
  -- This is a simplified representation

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Define the condition for two lines to be skew
  sorry

/-- Predicate to check if a plane passes through a line -/
def passes_through (p : Plane3D) (l : Line3D) : Prop :=
  -- Define the condition for a plane to pass through a line
  sorry

/-- Predicate to check if a plane is parallel to a line -/
def is_parallel_to (p : Plane3D) (l : Line3D) : Prop :=
  -- Define the condition for a plane to be parallel to a line
  sorry

/-- Theorem stating the existence and uniqueness of a plane passing through one skew line and parallel to another -/
theorem unique_plane_through_skew_lines (l1 l2 : Line3D) (h : are_skew l1 l2) :
  ∃! p : Plane3D, passes_through p l1 ∧ is_parallel_to p l2 :=
sorry

end NUMINAMATH_CALUDE_unique_plane_through_skew_lines_l1888_188825


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l1888_188890

/-- Given two people moving in opposite directions, with one person's speed and their final separation known, prove the other person's speed. -/
theorem opposite_direction_speed 
  (riya_speed : ℝ) 
  (total_separation : ℝ) 
  (time : ℝ) 
  (h1 : riya_speed = 21)
  (h2 : total_separation = 43)
  (h3 : time = 1) :
  let priya_speed := total_separation / time - riya_speed
  priya_speed = 22 := by sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l1888_188890


namespace NUMINAMATH_CALUDE_sum_of_prime_divisors_2018_l1888_188806

theorem sum_of_prime_divisors_2018 : ∃ p q : Nat, 
  p.Prime ∧ q.Prime ∧ 
  p ≠ q ∧
  p * q = 2018 ∧
  (∀ r : Nat, r.Prime → r ∣ 2018 → r = p ∨ r = q) ∧
  p + q = 1011 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_divisors_2018_l1888_188806


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1888_188844

/-- Given two vectors OA and OB in R², prove that if they are perpendicular
    and OA = (-1, 2) and OB = (3, m), then m = 3/2. -/
theorem perpendicular_vectors_m_value (OA OB : ℝ × ℝ) (m : ℝ) :
  OA = (-1, 2) →
  OB = (3, m) →
  OA.1 * OB.1 + OA.2 * OB.2 = 0 →
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1888_188844


namespace NUMINAMATH_CALUDE_constant_term_product_l1888_188801

-- Define polynomials p, q, and r
variable (p q r : ℝ → ℝ)

-- State the theorem
theorem constant_term_product (h1 : ∀ x, r x = p x * q x) 
                               (h2 : p 0 = 5) 
                               (h3 : r 0 = -10) : 
  q 0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_product_l1888_188801


namespace NUMINAMATH_CALUDE_parallel_lines_length_l1888_188860

-- Define the parallel lines and their lengths
def AB : ℝ := 120
def CD : ℝ := 80
def GH : ℝ := 140

-- Define the property of parallel lines
def parallel (a b c d : ℝ) : Prop := sorry

-- Theorem statement
theorem parallel_lines_length (EF : ℝ) 
  (h1 : parallel AB CD EF GH) : EF = 80 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_length_l1888_188860


namespace NUMINAMATH_CALUDE_remaining_pie_portion_l1888_188868

/-- Proves that given Carlos took 60% of a whole pie and Maria took one quarter of the remainder, 
    the portion of the whole pie left is 30%. -/
theorem remaining_pie_portion 
  (carlos_portion : Real) 
  (maria_portion : Real) 
  (h1 : carlos_portion = 0.6) 
  (h2 : maria_portion = 0.25 * (1 - carlos_portion)) : 
  1 - carlos_portion - maria_portion = 0.3 := by
sorry

end NUMINAMATH_CALUDE_remaining_pie_portion_l1888_188868


namespace NUMINAMATH_CALUDE_adam_nuts_purchase_l1888_188842

theorem adam_nuts_purchase (nuts_price dried_fruits_price dried_fruits_weight total_cost : ℝ) 
  (h1 : nuts_price = 12)
  (h2 : dried_fruits_price = 8)
  (h3 : dried_fruits_weight = 2.5)
  (h4 : total_cost = 56) :
  ∃ (nuts_weight : ℝ), 
    nuts_weight * nuts_price + dried_fruits_weight * dried_fruits_price = total_cost ∧ 
    nuts_weight = 3 := by
sorry


end NUMINAMATH_CALUDE_adam_nuts_purchase_l1888_188842


namespace NUMINAMATH_CALUDE_min_sum_position_max_product_position_l1888_188884

/-- Represents the special number with 1991 nines between two ones -/
def specialNumber : ℕ := 1 * 10^1992 + 1

/-- Calculates the sum when splitting the number at position m -/
def sumAtPosition (m : ℕ) : ℕ := 
  (2 * 10^m - 1) + (10^(1992 - m) - 9)

/-- Calculates the product when splitting the number at position m -/
def productAtPosition (m : ℕ) : ℕ := 
  (2 * 10^m - 1) * (10^(1992 - m) - 9)

theorem min_sum_position : 
  ∀ m : ℕ, m ≠ 996 → m ≠ 997 → sumAtPosition m > sumAtPosition 996 :=
sorry

theorem max_product_position : 
  ∀ m : ℕ, m ≠ 995 → m ≠ 996 → productAtPosition m < productAtPosition 995 :=
sorry

end NUMINAMATH_CALUDE_min_sum_position_max_product_position_l1888_188884


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_12_l1888_188888

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |> List.sum

theorem last_two_digits_sum_factorials_12 :
  lastTwoDigits (sumFactorials 12) = 13 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_12_l1888_188888


namespace NUMINAMATH_CALUDE_min_time_to_finish_tasks_l1888_188889

def wash_rice_time : ℕ := 2
def cook_porridge_time : ℕ := 10
def wash_vegetables_time : ℕ := 3
def chop_vegetables_time : ℕ := 5

def total_vegetable_time : ℕ := wash_vegetables_time + chop_vegetables_time

theorem min_time_to_finish_tasks : ℕ := by
  have h1 : wash_rice_time + cook_porridge_time = 12 := by sorry
  have h2 : total_vegetable_time ≤ cook_porridge_time := by sorry
  exact 12

end NUMINAMATH_CALUDE_min_time_to_finish_tasks_l1888_188889


namespace NUMINAMATH_CALUDE_graph_intersects_x_equals_one_at_most_once_f_equals_g_l1888_188834

-- Define a general function type
def RealFunction := ℝ → ℝ

-- Statement 1: The graph of y = f(x) intersects with x = 1 at most at one point
theorem graph_intersects_x_equals_one_at_most_once (f : RealFunction) :
  ∃! y, f 1 = y :=
sorry

-- Statement 2: f(x) = x^2 - 2x + 1 and g(t) = t^2 - 2t + 1 are the same function
def f (x : ℝ) : ℝ := x^2 - 2*x + 1
def g (t : ℝ) : ℝ := t^2 - 2*t + 1

theorem f_equals_g : f = g :=
sorry

end NUMINAMATH_CALUDE_graph_intersects_x_equals_one_at_most_once_f_equals_g_l1888_188834


namespace NUMINAMATH_CALUDE_zoo_animals_count_l1888_188847

/-- The number of ostriches in the zoo -/
def num_ostriches : ℕ := 15

/-- The number of sika deer in the zoo -/
def num_deer : ℕ := 23

/-- The number of legs an ostrich has -/
def ostrich_legs : ℕ := 2

/-- The number of legs a sika deer has -/
def deer_legs : ℕ := 4

/-- The total number of legs when counted normally -/
def total_legs : ℕ := 122

/-- The total number of legs when the numbers of animals are swapped -/
def swapped_legs : ℕ := 106

theorem zoo_animals_count : 
  (num_ostriches * ostrich_legs + num_deer * deer_legs = total_legs) ∧
  (num_deer * ostrich_legs + num_ostriches * deer_legs = swapped_legs) := by
  sorry

#check zoo_animals_count

end NUMINAMATH_CALUDE_zoo_animals_count_l1888_188847


namespace NUMINAMATH_CALUDE_new_barbell_total_cost_l1888_188862

def old_barbell_cost : ℝ := 250

def new_barbell_cost_increase_percentage : ℝ := 0.3

def sales_tax_percentage : ℝ := 0.1

def new_barbell_cost_before_tax : ℝ := old_barbell_cost * (1 + new_barbell_cost_increase_percentage)

def sales_tax : ℝ := new_barbell_cost_before_tax * sales_tax_percentage

def total_cost : ℝ := new_barbell_cost_before_tax + sales_tax

theorem new_barbell_total_cost : total_cost = 357.50 := by
  sorry

end NUMINAMATH_CALUDE_new_barbell_total_cost_l1888_188862


namespace NUMINAMATH_CALUDE_problem_solution_l1888_188814

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / (x^2 + 3)

theorem problem_solution (a : ℝ) :
  (∀ x, deriv (f a) x = (a * (x^2 + 3) - a * x * (2 * x)) / (x^2 + 3)^2) →
  deriv (f a) 1 = 1/2 →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1888_188814


namespace NUMINAMATH_CALUDE_a_b_equality_l1888_188850

theorem a_b_equality (a b : ℝ) 
  (h1 : a * b = 1) 
  (h2 : (a + b + 2) / 4 = 1 / (a + 1) + 1 / (b + 1)) : 
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_a_b_equality_l1888_188850


namespace NUMINAMATH_CALUDE_ceiling_floor_expression_l1888_188855

theorem ceiling_floor_expression : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ - 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_expression_l1888_188855


namespace NUMINAMATH_CALUDE_angle_sum_equality_l1888_188878

theorem angle_sum_equality (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 1/7) (h4 : Real.tan β = 3/79) : 5*α + 2*β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_equality_l1888_188878


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l1888_188808

theorem perfect_square_quadratic (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + 2*(m-3)*x + 16 = y^2) → (m = 7 ∨ m = -1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l1888_188808


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1888_188856

theorem sufficient_not_necessary_condition : 
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧ 
  (∃ a b : ℝ, a * b > 1 ∧ (a ≤ 1 ∨ b ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1888_188856


namespace NUMINAMATH_CALUDE_base5_of_89_l1888_188843

-- Define a function to convert a natural number to its base-5 representation
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

-- Theorem stating that 89 in base-5 is equivalent to [4, 2, 3]
theorem base5_of_89 : toBase5 89 = [4, 2, 3] := by sorry

end NUMINAMATH_CALUDE_base5_of_89_l1888_188843


namespace NUMINAMATH_CALUDE_masha_floor_number_l1888_188812

/-- Represents a multi-story apartment building -/
structure ApartmentBuilding where
  floors : ℕ
  entrances : ℕ
  apartments_per_floor : ℕ

/-- Calculates the floor number given an apartment number and building structure -/
def floor_number (building : ApartmentBuilding) (apartment_number : ℕ) : ℕ :=
  sorry

theorem masha_floor_number :
  let building := ApartmentBuilding.mk 17 4 5
  let masha_apartment := 290
  floor_number building masha_apartment = 7 := by
  sorry

end NUMINAMATH_CALUDE_masha_floor_number_l1888_188812


namespace NUMINAMATH_CALUDE_unique_solution_triple_l1888_188869

theorem unique_solution_triple (x y z : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x^2 - y = (z - 1)^2 ∧
  y^2 - z = (x - 1)^2 ∧
  z^2 - x = (y - 1)^2 →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_triple_l1888_188869


namespace NUMINAMATH_CALUDE_binomial_9_choose_5_l1888_188817

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by sorry

end NUMINAMATH_CALUDE_binomial_9_choose_5_l1888_188817


namespace NUMINAMATH_CALUDE_complex_product_real_l1888_188823

theorem complex_product_real (x : ℝ) : 
  let z₁ : ℂ := 1 + I
  let z₂ : ℂ := x - I
  (z₁ * z₂).im = 0 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l1888_188823


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1888_188800

theorem contrapositive_equivalence (p q : Prop) :
  (p → q) → (¬q → ¬p) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1888_188800


namespace NUMINAMATH_CALUDE_range_of_a_proposition_holds_l1888_188841

/-- The proposition that the inequality ax^2 - 2ax - 3 ≥ 0 does not hold for all real x -/
def proposition (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬(a * x^2 - 2 * a * x - 3 ≥ 0)

/-- The theorem stating that if the proposition holds, then a is in the range (-3, 0] -/
theorem range_of_a (a : ℝ) (h : proposition a) : -3 < a ∧ a ≤ 0 := by
  sorry

/-- The theorem stating that if a is in the range (-3, 0], then the proposition holds -/
theorem proposition_holds (a : ℝ) (h : -3 < a ∧ a ≤ 0) : proposition a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_proposition_holds_l1888_188841


namespace NUMINAMATH_CALUDE_grade_distribution_l1888_188852

theorem grade_distribution (total_students : ℕ) 
  (prob_A : ℝ) (prob_B : ℝ) (prob_C : ℝ) 
  (h1 : prob_A = 0.8 * prob_B) 
  (h2 : prob_C = 1.2 * prob_B) 
  (h3 : prob_A + prob_B + prob_C = 1) 
  (h4 : total_students = 40) :
  ∃ (A B C : ℕ), 
    A + B + C = total_students ∧ 
    A = 10 ∧ 
    B = 14 ∧ 
    C = 16 := by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_l1888_188852


namespace NUMINAMATH_CALUDE_square_sum_ge_product_sum_abs_diff_product_gt_abs_diff_l1888_188863

-- Theorem 1
theorem square_sum_ge_product_sum (a b : ℝ) : a^2 + b^2 ≥ a*b + a + b - 1 := by
  sorry

-- Theorem 2
theorem abs_diff_product_gt_abs_diff (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) :
  |1 - a*b| > |a - b| := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_product_sum_abs_diff_product_gt_abs_diff_l1888_188863


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l1888_188877

theorem consecutive_integers_product (n : ℕ) : 
  n > 0 ∧ (n + (n + 1) < 150) → n * (n + 1) ≤ 5550 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l1888_188877


namespace NUMINAMATH_CALUDE_expression_value_l1888_188864

theorem expression_value : ∃ x : ℕ, (8000 * 6000 : ℕ) = 480 * x ∧ x = 100000 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1888_188864


namespace NUMINAMATH_CALUDE_train_length_l1888_188805

/-- The length of a train given its speed, bridge length, and time to pass the bridge -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  bridge_length = 140 →
  passing_time = 40 →
  (train_speed * passing_time) - bridge_length = 360 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1888_188805


namespace NUMINAMATH_CALUDE_product_of_good_is_good_l1888_188811

/-- A positive integer is good if it can be represented as ax^2 + bxy + cy^2 
    with b^2 - 4ac = -20 for some integers a, b, c, x, y -/
def is_good (n : ℕ+) : Prop :=
  ∃ (a b c x y : ℤ), (n : ℤ) = a * x^2 + b * x * y + c * y^2 ∧ b^2 - 4 * a * c = -20

/-- The product of two good numbers is also a good number -/
theorem product_of_good_is_good (n1 n2 : ℕ+) (h1 : is_good n1) (h2 : is_good n2) :
  is_good (n1 * n2) :=
sorry

end NUMINAMATH_CALUDE_product_of_good_is_good_l1888_188811


namespace NUMINAMATH_CALUDE_sum_of_derivatives_positive_l1888_188840

def f (x : ℝ) : ℝ := -x^2 - x^4 - x^6

theorem sum_of_derivatives_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ < 0) (h₂ : x₂ + x₃ < 0) (h₃ : x₃ + x₁ < 0) : 
  deriv f x₁ + deriv f x₂ + deriv f x₃ > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_derivatives_positive_l1888_188840


namespace NUMINAMATH_CALUDE_double_counted_page_l1888_188892

/-- Given a book with 62 pages, prove that if the sum of all page numbers
    plus an additional count of one page number equals 1997,
    then the page number that was counted twice is 44. -/
theorem double_counted_page (n : ℕ) (x : ℕ) : 
  n = 62 → 
  (n * (n + 1)) / 2 + x = 1997 → 
  x = 44 := by
sorry

end NUMINAMATH_CALUDE_double_counted_page_l1888_188892


namespace NUMINAMATH_CALUDE_average_temperature_proof_l1888_188897

def daily_temperatures : List ℝ := [40, 50, 65, 36, 82, 72, 26]
def days_in_week : ℕ := 7

theorem average_temperature_proof :
  (daily_temperatures.sum / days_in_week : ℝ) = 53 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_proof_l1888_188897


namespace NUMINAMATH_CALUDE_zeros_count_theorem_l1888_188857

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def has_unique_zero_in_interval (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  f c = 0 ∧ ∀ x, x ∈ Set.Icc a b → f x = 0 → x = c

def count_zeros_in_interval (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem zeros_count_theorem (f : ℝ → ℝ) :
  is_even_function f →
  (∀ x, f (5 + x) = f (5 - x)) →
  has_unique_zero_in_interval f 0 5 1 →
  count_zeros_in_interval f (-2012) 2012 = 806 :=
sorry

end NUMINAMATH_CALUDE_zeros_count_theorem_l1888_188857


namespace NUMINAMATH_CALUDE_problem_solution_l1888_188829

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (exp x + a)

def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem problem_solution :
  (∃ a, is_odd (f a)) →
  (∃ a, a = 0 ∧ is_odd (f a)) ∧
  (∀ m : ℝ,
    (m > 1/exp 1 + exp 2 → ¬∃ x, (log x) / x = x^2 - 2 * (exp 1) * x + m) ∧
    (m = 1/exp 1 + exp 2 → ∃! x, x = exp 1 ∧ (log x) / x = x^2 - 2 * (exp 1) * x + m) ∧
    (m < 1/exp 1 + exp 2 → ∃ x y, x ≠ y ∧ (log x) / x = x^2 - 2 * (exp 1) * x + m ∧ (log y) / y = y^2 - 2 * (exp 1) * y + m))
    := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1888_188829


namespace NUMINAMATH_CALUDE_product_equality_l1888_188849

theorem product_equality (a b c : ℝ) (h : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) :
  6 * 15 * 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1888_188849


namespace NUMINAMATH_CALUDE_min_value_expression_l1888_188886

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a + b) / c + (b + c) / a + (c + a) / b + 3 > 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1888_188886


namespace NUMINAMATH_CALUDE_segment_translation_l1888_188828

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation function
def translate_left (p : Point) (units : ℝ) : Point :=
  (p.1 - units, p.2)

-- Define the problem statement
theorem segment_translation :
  let A : Point := (-1, 4)
  let B : Point := (-4, 1)
  let A₁ : Point := translate_left A 4
  let B₁ : Point := translate_left B 4
  A₁ = (-5, 4) ∧ B₁ = (-8, 1) := by sorry

end NUMINAMATH_CALUDE_segment_translation_l1888_188828


namespace NUMINAMATH_CALUDE_incorrect_statement_l1888_188830

theorem incorrect_statement : ¬(∀ m : ℝ, (∃ x : ℝ, x^2 + x - m = 0) → m > 0) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1888_188830


namespace NUMINAMATH_CALUDE_rectangle_areas_sum_l1888_188832

theorem rectangle_areas_sum : 
  let width := 3
  let lengths := [1, 8, 27, 64, 125, 216]
  let areas := lengths.map (λ l => width * l)
  areas.sum = 1323 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_areas_sum_l1888_188832


namespace NUMINAMATH_CALUDE_katies_sister_candy_l1888_188896

theorem katies_sister_candy (katie_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ) :
  katie_candy = 10 →
  eaten_candy = 9 →
  remaining_candy = 7 →
  ∃ sister_candy : ℕ, sister_candy = 6 ∧ katie_candy + sister_candy = eaten_candy + remaining_candy :=
by sorry

end NUMINAMATH_CALUDE_katies_sister_candy_l1888_188896


namespace NUMINAMATH_CALUDE_geese_in_marsh_l1888_188854

theorem geese_in_marsh (total_birds ducks : ℕ) (h1 : total_birds = 95) (h2 : ducks = 37) :
  total_birds - ducks = 58 := by
  sorry

end NUMINAMATH_CALUDE_geese_in_marsh_l1888_188854


namespace NUMINAMATH_CALUDE_octal_5374_to_decimal_l1888_188809

def octal_to_decimal (a b c d : Nat) : Nat :=
  d * 8^0 + c * 8^1 + b * 8^2 + a * 8^3

theorem octal_5374_to_decimal :
  octal_to_decimal 5 3 7 4 = 2812 := by
  sorry

end NUMINAMATH_CALUDE_octal_5374_to_decimal_l1888_188809


namespace NUMINAMATH_CALUDE_smallest_positive_integer_x_l1888_188879

theorem smallest_positive_integer_x (x : ℕ+) : (2 * (x : ℝ)^2 < 50) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_x_l1888_188879


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l1888_188816

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- The main theorem -/
theorem tangent_line_y_intercept (c1 c2 : Circle) (l : Line) :
  c1.center = (3, 0) →
  c1.radius = 3 →
  c2.center = (8, 0) →
  c2.radius = 2 →
  is_tangent l c1 →
  is_tangent l c2 →
  l.y_intercept = 15 * Real.sqrt 26 / 26 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l1888_188816


namespace NUMINAMATH_CALUDE_max_value_of_y_l1888_188875

def y (x : ℝ) : ℝ := |x + 1| - 2 * |x| + |x - 2|

theorem max_value_of_y :
  ∃ (α : ℝ), α = 3 ∧ ∀ x, -1 ≤ x ∧ x ≤ 2 → y x ≤ α :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_y_l1888_188875


namespace NUMINAMATH_CALUDE_fraction_simplification_l1888_188804

theorem fraction_simplification (a b : ℝ) (h : a ≠ b ∧ a ≠ -b) :
  (5 * a + 3 * b) / (a^2 - b^2) - (2 * a) / (a^2 - b^2) = 3 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1888_188804


namespace NUMINAMATH_CALUDE_ashok_pyarelal_capital_ratio_l1888_188872

/-- Given a total loss and Pyarelal's loss, calculate the ratio of Ashok's capital to Pyarelal's capital -/
theorem ashok_pyarelal_capital_ratio 
  (total_loss : ℕ) 
  (pyarelal_loss : ℕ) 
  (h1 : total_loss = 1600) 
  (h2 : pyarelal_loss = 1440) : 
  ∃ (a p : ℕ), a ≠ 0 ∧ p ≠ 0 ∧ a / p = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ashok_pyarelal_capital_ratio_l1888_188872


namespace NUMINAMATH_CALUDE_ceiling_minus_x_l1888_188839

theorem ceiling_minus_x (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ f : ℝ, 0 < f ∧ f < 1 ∧ ⌈x⌉ - x = 1 - f := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_l1888_188839
