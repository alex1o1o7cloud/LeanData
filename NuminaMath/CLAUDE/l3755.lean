import Mathlib

namespace NUMINAMATH_CALUDE_rhombus_area_in_hexagon_l3755_375552

/-- A regular hexagon -/
structure RegularHexagon where
  area : ℝ

/-- The total area of rhombuses that can be formed within a regular hexagon -/
def total_rhombus_area (h : RegularHexagon) : ℝ :=
  sorry

/-- Theorem: In a regular hexagon with area 80, the total area of rhombuses is 45 -/
theorem rhombus_area_in_hexagon (h : RegularHexagon) 
  (h_area : h.area = 80) : total_rhombus_area h = 45 :=
  sorry

end NUMINAMATH_CALUDE_rhombus_area_in_hexagon_l3755_375552


namespace NUMINAMATH_CALUDE_volume_of_region_l3755_375503

-- Define the function f
def f (x y z : ℝ) : ℝ := |x - y + z| + |x - y - z| + |x + y - z| + |-x + y - z|

-- Define the region R
def R : Set (ℝ × ℝ × ℝ) := {(x, y, z) | f x y z ≤ 6}

-- Theorem statement
theorem volume_of_region : MeasureTheory.volume R = 36 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_region_l3755_375503


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3755_375540

theorem linear_equation_solution (b : ℝ) : 
  (∀ x y : ℝ, x - 2*y + b = 0 → y = (1/2)*x + b - 1) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3755_375540


namespace NUMINAMATH_CALUDE_sara_quarters_l3755_375568

theorem sara_quarters (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 21 → additional = 49 → total = initial + additional → total = 70 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l3755_375568


namespace NUMINAMATH_CALUDE_value_of_expression_l3755_375551

theorem value_of_expression : 70 * Real.sqrt ((8^10 + 4^10) / (8^4 + 4^11)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3755_375551


namespace NUMINAMATH_CALUDE_class_size_problem_l3755_375511

theorem class_size_problem (total : ℕ) : 
  (total / 3 : ℚ) + 26 = total → total = 39 :=
by sorry

end NUMINAMATH_CALUDE_class_size_problem_l3755_375511


namespace NUMINAMATH_CALUDE_ab_value_l3755_375549

theorem ab_value (a b : ℤ) (h1 : |a| = 7) (h2 : b = 5) (h3 : a + b < 0) : a * b = -35 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3755_375549


namespace NUMINAMATH_CALUDE_bridge_length_l3755_375573

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 145 →
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  (train_speed * crossing_time) - train_length = 230 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l3755_375573


namespace NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l3755_375562

theorem arithmetic_mean_sqrt2 :
  (Real.sqrt 2 + 1 + (Real.sqrt 2 - 1)) / 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l3755_375562


namespace NUMINAMATH_CALUDE_mean_squares_sum_l3755_375572

theorem mean_squares_sum (a b c : ℝ) : 
  (a + b + c) / 3 = 12 →
  (a * b * c) ^ (1/3 : ℝ) = 5 →
  3 / (1/a + 1/b + 1/c) = 4 →
  a^2 + b^2 + c^2 = 1108.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_squares_sum_l3755_375572


namespace NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l3755_375542

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem thirty_sided_polygon_diagonals :
  num_diagonals 30 = 405 := by
  sorry

#eval num_diagonals 30  -- This will evaluate to 405

end NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l3755_375542


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3755_375581

theorem polynomial_simplification (p : ℝ) :
  (5 * p^4 + 4 * p^3 - 7 * p^2 + 9 * p - 3) + (-8 * p^4 + 2 * p^3 - p^2 - 3 * p + 4) =
  -3 * p^4 + 6 * p^3 - 8 * p^2 + 6 * p + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3755_375581


namespace NUMINAMATH_CALUDE_base7_product_l3755_375513

/-- Converts a base 7 number represented as a list of digits to its decimal (base 10) equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 7 * acc) 0

/-- Converts a decimal (base 10) number to its base 7 representation as a list of digits -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- The problem statement -/
theorem base7_product : 
  toBase7 (toDecimal [1,3,2,4] * toDecimal [2,3]) = [3,1,4,1,5] := by
  sorry

end NUMINAMATH_CALUDE_base7_product_l3755_375513


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3755_375571

/-- Given a function g: ℝ → ℝ satisfying the functional equation
    (g x * g y - g (x * y)) / 4 = x + y + 3 for all x, y ∈ ℝ,
    prove that g x = x + 4 for all x ∈ ℝ. -/
theorem functional_equation_solution (g : ℝ → ℝ)
    (h : ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 3) :
  ∀ x : ℝ, g x = x + 4 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3755_375571


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l3755_375528

/-- The rowing speed of a man in still water, given his speeds with and against the stream -/
theorem mans_rowing_speed (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 18)
  (h2 : speed_against_stream = 8) :
  (speed_with_stream + speed_against_stream) / 2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_l3755_375528


namespace NUMINAMATH_CALUDE_quaternary_201_equals_33_l3755_375570

/-- Converts a quaternary (base 4) number to its decimal equivalent -/
def quaternary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The quaternary representation of the number -/
def quaternary_201 : List Nat := [1, 0, 2]

theorem quaternary_201_equals_33 :
  quaternary_to_decimal quaternary_201 = 33 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_201_equals_33_l3755_375570


namespace NUMINAMATH_CALUDE_egyptian_fraction_sum_l3755_375564

theorem egyptian_fraction_sum : ∃ (b₂ b₃ b₄ b₅ b₆ : ℕ),
  (4 : ℚ) / 9 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ = 9 ∧
  b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₂ ≠ b₆ ∧
  b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧
  b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧
  b₅ ≠ b₆ := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_sum_l3755_375564


namespace NUMINAMATH_CALUDE_monday_attendance_l3755_375558

theorem monday_attendance (tuesday : ℕ) (wed_to_fri : ℕ) (average : ℕ) (days : ℕ)
  (h1 : tuesday = 15)
  (h2 : wed_to_fri = 10)
  (h3 : average = 11)
  (h4 : days = 5) :
  ∃ (monday : ℕ), monday + tuesday + 3 * wed_to_fri = average * days ∧ monday = 10 := by
  sorry

end NUMINAMATH_CALUDE_monday_attendance_l3755_375558


namespace NUMINAMATH_CALUDE_midpoint_of_specific_segment_l3755_375563

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Calculate the midpoint of two points in polar coordinates -/
def polarMidpoint (p1 p2 : PolarPoint) : PolarPoint :=
  sorry

theorem midpoint_of_specific_segment :
  let p1 : PolarPoint := ⟨10, π/4⟩
  let p2 : PolarPoint := ⟨10, 3*π/4⟩
  let midpoint := polarMidpoint p1 p2
  midpoint.r = 5 * Real.sqrt 2 ∧ midpoint.θ = π/2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_of_specific_segment_l3755_375563


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3755_375579

theorem inequality_equivalence (x : ℝ) : 2 * x - 3 < x + 1 ↔ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3755_375579


namespace NUMINAMATH_CALUDE_solve_for_a_l3755_375596

theorem solve_for_a (a b c : ℝ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 8) 
  (eq3 : c = 4) : 
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l3755_375596


namespace NUMINAMATH_CALUDE_max_sock_price_l3755_375543

theorem max_sock_price (total_money : ℕ) (entrance_fee : ℕ) (num_socks : ℕ) (tax_rate : ℚ) :
  total_money = 180 →
  entrance_fee = 3 →
  num_socks = 20 →
  tax_rate = 6 / 100 →
  ∃ (max_price : ℕ), 
    max_price = 8 ∧
    (max_price : ℚ) * num_socks * (1 + tax_rate) + entrance_fee ≤ total_money ∧
    ∀ (price : ℕ), 
      price > max_price → 
      (price : ℚ) * num_socks * (1 + tax_rate) + entrance_fee > total_money :=
by sorry

end NUMINAMATH_CALUDE_max_sock_price_l3755_375543


namespace NUMINAMATH_CALUDE_square_sum_inequality_l3755_375589

theorem square_sum_inequality (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (2 + a) * (2 + b) ≥ c * d := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l3755_375589


namespace NUMINAMATH_CALUDE_lucy_balance_l3755_375519

/-- Calculates the final balance after a deposit and withdrawal --/
def final_balance (initial : ℕ) (deposit : ℕ) (withdrawal : ℕ) : ℕ :=
  initial + deposit - withdrawal

/-- Proves that Lucy's final balance is $76 --/
theorem lucy_balance : final_balance 65 15 4 = 76 := by
  sorry

end NUMINAMATH_CALUDE_lucy_balance_l3755_375519


namespace NUMINAMATH_CALUDE_rectangular_cross_section_shapes_l3755_375590

/-- Enumeration of the geometric shapes in question -/
inductive GeometricShape
  | RectangularPrism
  | Cylinder
  | Cone
  | Cube

/-- Predicate to determine if a shape can have a rectangular cross-section -/
def has_rectangular_cross_section (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.RectangularPrism => true
  | GeometricShape.Cylinder => true
  | GeometricShape.Cone => false
  | GeometricShape.Cube => true

/-- The set of shapes that can have a rectangular cross-section -/
def shapes_with_rectangular_cross_section : Set GeometricShape :=
  {shape | has_rectangular_cross_section shape}

/-- Theorem stating which shapes can have a rectangular cross-section -/
theorem rectangular_cross_section_shapes :
  shapes_with_rectangular_cross_section =
    {GeometricShape.RectangularPrism, GeometricShape.Cylinder, GeometricShape.Cube} :=
by sorry


end NUMINAMATH_CALUDE_rectangular_cross_section_shapes_l3755_375590


namespace NUMINAMATH_CALUDE_room_perimeter_l3755_375537

theorem room_perimeter (breadth : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) : 
  length = 3 * breadth →
  area = 12 →
  area = length * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 16 := by
  sorry

end NUMINAMATH_CALUDE_room_perimeter_l3755_375537


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l3755_375501

theorem sum_of_roots_zero (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 6*x^2 - x + 6
  ∃ a b c d : ℝ, (∀ x, f x = (x^2 + a*x + b) * (x^2 + c*x + d)) →
  (a + b + c + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l3755_375501


namespace NUMINAMATH_CALUDE_min_value_theorem_l3755_375576

theorem min_value_theorem (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : (b + c) / a + (a + c) / b = (a + b) / c + 1) :
  (∀ x y z, 0 < x ∧ 0 < y ∧ 0 < z ∧ (y + z) / x + (x + z) / y = (x + y) / z + 1 → (a + b) / c ≤ (x + y) / z) ∧
  (a + b) / c = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3755_375576


namespace NUMINAMATH_CALUDE_high_schooler_pairs_l3755_375510

theorem high_schooler_pairs (n : ℕ) (h : 10 ≤ n ∧ n ≤ 15) : 
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 15 → n * (n - 1) / 2 ≤ m * (m - 1) / 2) → n = 10 ∧
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 15 → m * (m - 1) / 2 ≤ n * (n - 1) / 2) → n = 15 :=
by sorry

end NUMINAMATH_CALUDE_high_schooler_pairs_l3755_375510


namespace NUMINAMATH_CALUDE_block_distribution_l3755_375545

theorem block_distribution (n : ℕ) (h : n > 0) (h_divides : n ∣ 49) :
  ∃ (blocks_per_color : ℕ), blocks_per_color > 0 ∧ blocks_per_color * n = 49 := by
  sorry

end NUMINAMATH_CALUDE_block_distribution_l3755_375545


namespace NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l3755_375587

/-- Determines if the equation x^2+y^2-4x+2y+m=0 represents a circle -/
def is_circle (m : ℝ) : Prop := (-4)^2 + 2^2 - 4*m > 0

/-- The condition m=0 is sufficient but not necessary for the equation to represent a circle -/
theorem m_zero_sufficient_not_necessary : 
  (is_circle 0) ∧ (∃ m : ℝ, m ≠ 0 ∧ is_circle m) :=
sorry

end NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l3755_375587


namespace NUMINAMATH_CALUDE_wong_valentines_l3755_375512

/-- The number of Valentines Mrs. Wong initially had -/
def initial_valentines : ℕ := 30

/-- The number of Valentines Mrs. Wong gave to her children -/
def given_valentines : ℕ := 8

/-- The number of Valentines Mrs. Wong has left -/
def remaining_valentines : ℕ := initial_valentines - given_valentines

theorem wong_valentines : remaining_valentines = 22 := by
  sorry

end NUMINAMATH_CALUDE_wong_valentines_l3755_375512


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l3755_375538

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- Theorem statement
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l3755_375538


namespace NUMINAMATH_CALUDE_consecutive_numbers_sequence_l3755_375548

def is_valid_sequence (a b : ℕ) : Prop :=
  let n := b - a + 1
  let sum := n * (a + b) / 2
  let mean := sum / n
  let sum_without_122_123 := sum - 122 - 123
  let mean_without_122_123 := sum_without_122_123 / (n - 2)
  (mean = 85) ∧
  (mean = (70 + 82 + 103) / 3) ∧
  (mean_without_122_123 + 1 = mean) ∧
  (a = 47) ∧
  (b = 123)

theorem consecutive_numbers_sequence :
  ∃ a b : ℕ, is_valid_sequence a b :=
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sequence_l3755_375548


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l3755_375508

/-- Given a plane vector a = (2,0), |b| = 2, and a ⋅ b = 2, prove |a - 2b| = 2√3 -/
theorem vector_magnitude_proof (b : ℝ × ℝ) :
  let a : ℝ × ℝ := (2, 0)
  (norm b = 2) →
  (a.1 * b.1 + a.2 * b.2 = 2) →
  norm (a - 2 • b) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l3755_375508


namespace NUMINAMATH_CALUDE_moores_law_transistor_count_l3755_375554

/-- Moore's law calculation for transistor count --/
theorem moores_law_transistor_count 
  (initial_year : Nat) 
  (final_year : Nat) 
  (initial_transistors : Nat) 
  (doubling_period : Nat) 
  (h1 : initial_year = 1985)
  (h2 : final_year = 2010)
  (h3 : initial_transistors = 300000)
  (h4 : doubling_period = 2) :
  let years_passed := final_year - initial_year
  let doublings := years_passed / doubling_period
  initial_transistors * (2 ^ doublings) = 1228800000 :=
by sorry

end NUMINAMATH_CALUDE_moores_law_transistor_count_l3755_375554


namespace NUMINAMATH_CALUDE_problem_solution_l3755_375518

theorem problem_solution (m n : ℝ) (h : 2 * m - n = 3) :
  (∀ x : ℝ, |x| + |n + 3| ≥ 9 → x ≤ -3 ∨ x ≥ 3) ∧
  (∃ min : ℝ, min = 3 ∧ ∀ x y : ℝ, 2 * x - y = 3 →
    |5/3 * x - 1/3 * y| + |1/3 * x - 2/3 * y| ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3755_375518


namespace NUMINAMATH_CALUDE_min_total_cost_l3755_375500

/-- Represents the number of rooms of each type -/
structure RoomAllocation where
  triple : ℕ
  double : ℕ
  single : ℕ

/-- Calculates the total cost for a given room allocation -/
def totalCost (a : RoomAllocation) : ℕ :=
  300 * a.triple + 300 * a.double + 200 * a.single

/-- Checks if a room allocation is valid for the given constraints -/
def isValidAllocation (a : RoomAllocation) : Prop :=
  a.triple + a.double + a.single = 20 ∧
  3 * a.triple + 2 * a.double + a.single = 50

/-- Theorem: The minimum total cost for the given constraints is 5500 yuan -/
theorem min_total_cost :
  ∃ (a : RoomAllocation), isValidAllocation a ∧
    totalCost a = 5500 ∧
    ∀ (b : RoomAllocation), isValidAllocation b → totalCost a ≤ totalCost b :=
  sorry

end NUMINAMATH_CALUDE_min_total_cost_l3755_375500


namespace NUMINAMATH_CALUDE_problem_solution_l3755_375593

theorem problem_solution : 
  (2 + 3 + 4) / 3 = (1990 + 1991 + 1992) / N → N = 1991 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3755_375593


namespace NUMINAMATH_CALUDE_ordered_pairs_satisfying_conditions_l3755_375583

theorem ordered_pairs_satisfying_conditions :
  ∀ a b : ℕ+,
  (a.val^2 + b.val^2 + 25 = 15 * a.val * b.val) ∧
  (Nat.Prime (a.val^2 + a.val * b.val + b.val^2)) →
  ((a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ordered_pairs_satisfying_conditions_l3755_375583


namespace NUMINAMATH_CALUDE_golden_ratio_triangle_geometric_mean_l3755_375532

/-- Golden ratio division of a line segment -/
structure GoldenRatioDivision (α : Type*) [LinearOrderedField α] where
  ab : α
  bc : α
  golden_ratio : (ab - bc) / bc = bc / ab

/-- Right triangle with golden ratio hypotenuse -/
structure GoldenRatioTriangle (α : Type*) [LinearOrderedField α] extends GoldenRatioDivision α where
  other_leg : α
  pythagorean : other_leg^2 + bc^2 = ab^2

/-- The other leg is the geometric mean of the hypotenuse and the first leg -/
theorem golden_ratio_triangle_geometric_mean 
  {α : Type*} [LinearOrderedField α] (t : GoldenRatioTriangle α) :
  t.other_leg^2 = t.ab * t.bc :=
by sorry

end NUMINAMATH_CALUDE_golden_ratio_triangle_geometric_mean_l3755_375532


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3755_375530

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = 3/2 * x ∨ y = -3/2 * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(3/2)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3755_375530


namespace NUMINAMATH_CALUDE_total_rounded_to_nearest_dollar_l3755_375595

-- Define the rounding function
def roundToNearestDollar (x : ℚ) : ℤ :=
  if x - x.floor < 1/2 then x.floor else x.ceil

-- Define the purchases
def purchase1 : ℚ := 299/100
def purchase2 : ℚ := 651/100
def purchase3 : ℚ := 1049/100

-- Theorem statement
theorem total_rounded_to_nearest_dollar :
  (roundToNearestDollar purchase1 + 
   roundToNearestDollar purchase2 + 
   roundToNearestDollar purchase3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_rounded_to_nearest_dollar_l3755_375595


namespace NUMINAMATH_CALUDE_donut_selections_l3755_375507

theorem donut_selections (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 4) : 
  Nat.choose (n + k - 1) (k - 1) = 84 := by
  sorry

end NUMINAMATH_CALUDE_donut_selections_l3755_375507


namespace NUMINAMATH_CALUDE_total_games_proof_l3755_375555

/-- The number of baseball games Benny's high school played -/
def total_games (games_attended games_missed : ℕ) : ℕ :=
  games_attended + games_missed

/-- Theorem stating that the total number of games is the sum of attended and missed games -/
theorem total_games_proof (games_attended games_missed : ℕ) :
  total_games games_attended games_missed = games_attended + games_missed :=
by sorry

end NUMINAMATH_CALUDE_total_games_proof_l3755_375555


namespace NUMINAMATH_CALUDE_complex_expression_result_l3755_375529

theorem complex_expression_result : 
  ((3 / 4) * (1 / 2) * (2 / 5) * 5020) - ((2^3) * (4/5) * 250) + Real.sqrt 900 = -817 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_result_l3755_375529


namespace NUMINAMATH_CALUDE_vasyas_numbers_l3755_375527

theorem vasyas_numbers (x y : ℝ) : x + y = x * y ∧ x + y = x / y ∧ x * y = x / y → x = (1 : ℝ) / 2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l3755_375527


namespace NUMINAMATH_CALUDE_opposite_signs_abs_sum_less_abs_diff_l3755_375547

theorem opposite_signs_abs_sum_less_abs_diff
  (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_abs_sum_less_abs_diff_l3755_375547


namespace NUMINAMATH_CALUDE_triangle_problem_l3755_375525

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.c = 5/2)
  (h2 : t.b = Real.sqrt 6)
  (h3 : 4 * t.a - 3 * Real.sqrt 6 * Real.cos t.A = 0) :
  t.a = 3/2 ∧ t.B = 2 * t.A := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3755_375525


namespace NUMINAMATH_CALUDE_repeating_decimal_56_l3755_375574

def repeating_decimal (a b : ℕ) : ℚ :=
  (a : ℚ) / (99 : ℚ)

theorem repeating_decimal_56 :
  repeating_decimal 56 = 56 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_56_l3755_375574


namespace NUMINAMATH_CALUDE_combination_sum_equals_55_l3755_375514

-- Define the combination function
def combination (n r : ℕ) : ℕ :=
  if r ≤ n then
    Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))
  else
    0

-- State the theorem
theorem combination_sum_equals_55 :
  combination 10 9 + combination 10 8 = 55 :=
sorry

end NUMINAMATH_CALUDE_combination_sum_equals_55_l3755_375514


namespace NUMINAMATH_CALUDE_min_value_of_y_l3755_375567

theorem min_value_of_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x z : ℝ, x > 0 → z > 0 → x + z = 2 → 1/x + 4/z ≥ 1/a + 4/b) ∧ 1/a + 4/b = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_y_l3755_375567


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3755_375560

theorem quadratic_inequality (m : ℝ) :
  (∀ x : ℝ, m * x^2 + 2 * m * x - 1 < 0) ↔ (-1 < m ∧ m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3755_375560


namespace NUMINAMATH_CALUDE_cherry_pie_degrees_l3755_375577

theorem cherry_pie_degrees (total_students : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ) 
  (h1 : total_students = 36)
  (h2 : chocolate = 12)
  (h3 : apple = 8)
  (h4 : blueberry = 6)
  (h5 : (total_students - (chocolate + apple + blueberry)) % 2 = 0) :
  (((total_students - (chocolate + apple + blueberry)) / 2) : ℚ) / total_students * 360 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pie_degrees_l3755_375577


namespace NUMINAMATH_CALUDE_students_without_gift_l3755_375535

theorem students_without_gift (total_students : ℕ) (h : total_students = 2016) :
  (∃ (no_gift : ℕ), no_gift = Nat.totient total_students ∧
    ∀ (n : ℕ), n ≥ 2 → no_gift = total_students - (total_students / n) * n) := by
  sorry

end NUMINAMATH_CALUDE_students_without_gift_l3755_375535


namespace NUMINAMATH_CALUDE_pear_count_l3755_375523

theorem pear_count (initial_apples : ℕ) (apple_removal_rate : ℚ) (pear_removal_rate : ℚ) :
  initial_apples = 160 →
  apple_removal_rate = 3/4 →
  pear_removal_rate = 1/3 →
  (initial_apples * (1 - apple_removal_rate) : ℚ) = (1/2 : ℚ) * (initial_pears * (1 - pear_removal_rate) : ℚ) →
  initial_pears = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_pear_count_l3755_375523


namespace NUMINAMATH_CALUDE_particular_number_l3755_375526

theorem particular_number (x : ℤ) (h : x - 7 = 2) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_l3755_375526


namespace NUMINAMATH_CALUDE_football_team_progress_l3755_375509

theorem football_team_progress (lost_yards gained_yards : ℤ) (h1 : lost_yards = 5) (h2 : gained_yards = 7) :
  gained_yards - lost_yards = 2 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l3755_375509


namespace NUMINAMATH_CALUDE_power_of_power_three_cubed_squared_l3755_375586

theorem power_of_power_three_cubed_squared : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_cubed_squared_l3755_375586


namespace NUMINAMATH_CALUDE_winter_sales_proof_l3755_375517

/-- Represents the sales of hamburgers in millions for each season and the total year -/
structure HamburgerSales where
  spring_summer : ℝ
  fall : ℝ
  winter : ℝ
  total : ℝ

/-- Given the conditions of the hamburger sales, prove that winter sales are 4 million -/
theorem winter_sales_proof (sales : HamburgerSales) 
  (h1 : sales.total = 20)
  (h2 : sales.spring_summer = 0.6 * sales.total)
  (h3 : sales.fall = 0.2 * sales.total)
  (h4 : sales.total = sales.spring_summer + sales.fall + sales.winter) :
  sales.winter = 4 := by
  sorry

#check winter_sales_proof

end NUMINAMATH_CALUDE_winter_sales_proof_l3755_375517


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3755_375598

-- Problem 1
theorem problem_1 : 3 * Real.sqrt 3 + Real.sqrt 8 - Real.sqrt 2 + Real.sqrt 27 = 6 * Real.sqrt 3 + Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 : (1/2) * (Real.sqrt 3 + Real.sqrt 5) - (3/4) * (Real.sqrt 5 - Real.sqrt 12) = 2 * Real.sqrt 3 - (1/4) * Real.sqrt 5 := by sorry

-- Problem 3
theorem problem_3 : (2 * Real.sqrt 5 + Real.sqrt 6) * (2 * Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - Real.sqrt 6)^2 = 3 + 2 * Real.sqrt 30 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3755_375598


namespace NUMINAMATH_CALUDE_system_solution_l3755_375541

theorem system_solution (x y z : ℝ) : 
  (x^2 + y^2 - z*(x + y) = 2 ∧
   y^2 + z^2 - x*(y + z) = 4 ∧
   z^2 + x^2 - y*(z + x) = 8) ↔
  ((x = 1 ∧ y = -1 ∧ z = 2) ∨ (x = -1 ∧ y = 1 ∧ z = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3755_375541


namespace NUMINAMATH_CALUDE_A_equals_2B_l3755_375524

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x - 2 * B^2
def g (B x : ℝ) : ℝ := B * x

-- State the theorem
theorem A_equals_2B (A B : ℝ) (h1 : B ≠ 0) (h2 : f A B (g B 1) = 0) : A = 2 * B := by
  sorry

end NUMINAMATH_CALUDE_A_equals_2B_l3755_375524


namespace NUMINAMATH_CALUDE_min_students_class_5_7_l3755_375546

theorem min_students_class_5_7 (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k + 3) ∧ 
  (∃ m : ℕ, n = 8 * m + 3) → 
  n ≥ 59 :=
sorry

end NUMINAMATH_CALUDE_min_students_class_5_7_l3755_375546


namespace NUMINAMATH_CALUDE_daughters_teachers_l3755_375531

/-- Proves the number of daughter's teachers given total spent, cost per gift, and son's teachers -/
theorem daughters_teachers (total_spent : ℕ) (cost_per_gift : ℕ) (sons_teachers : ℕ) 
  (h1 : total_spent = 70) 
  (h2 : cost_per_gift = 10) 
  (h3 : sons_teachers = 3) :
  total_spent / cost_per_gift - sons_teachers = 4 := by
  sorry

#check daughters_teachers

end NUMINAMATH_CALUDE_daughters_teachers_l3755_375531


namespace NUMINAMATH_CALUDE_william_hot_dogs_l3755_375506

/-- The number of hot dogs William sold during the first three innings -/
def first_innings_sales : ℕ := 19

/-- The number of hot dogs William sold during the next three innings -/
def next_innings_sales : ℕ := 27

/-- The number of hot dogs William had left to sell -/
def remaining_hot_dogs : ℕ := 45

/-- The total number of hot dogs William had at first -/
def total_hot_dogs : ℕ := first_innings_sales + next_innings_sales + remaining_hot_dogs

theorem william_hot_dogs : total_hot_dogs = 91 := by sorry

end NUMINAMATH_CALUDE_william_hot_dogs_l3755_375506


namespace NUMINAMATH_CALUDE_q_div_p_equals_162_l3755_375585

/-- The number of slips in the hat -/
def total_slips : ℕ := 40

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The number of slips with each number -/
def slips_per_number : ℕ := 4

/-- The probability that all four drawn slips bear the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

/-- The probability that two slips bear a number a and the other two bear a number b ≠ a -/
def q : ℚ := ((Nat.choose distinct_numbers 2 : ℚ) * 
              (Nat.choose slips_per_number 2 : ℚ) * 
              (Nat.choose slips_per_number 2 : ℚ)) / 
             (Nat.choose total_slips drawn_slips : ℚ)

/-- Theorem stating that q/p = 162 -/
theorem q_div_p_equals_162 : q / p = 162 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_162_l3755_375585


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l3755_375544

theorem sin_product_equals_one_sixteenth : 
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * 
  Real.sin (54 * π / 180) * Real.sin (78 * π / 180) = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l3755_375544


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3755_375505

theorem trigonometric_identity (α β : Real) (h : α + β = Real.pi / 3) :
  Real.tan α + Real.tan β + Real.sqrt 3 * Real.tan α * Real.tan β = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3755_375505


namespace NUMINAMATH_CALUDE_white_longer_than_blue_l3755_375580

/-- The length of the white line in inches -/
def white_line_length : ℝ := 7.666666666666667

/-- The length of the blue line in inches -/
def blue_line_length : ℝ := 3.3333333333333335

/-- The difference in length between the white and blue lines -/
def length_difference : ℝ := white_line_length - blue_line_length

theorem white_longer_than_blue :
  length_difference = 4.333333333333333 := by sorry

end NUMINAMATH_CALUDE_white_longer_than_blue_l3755_375580


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3755_375556

theorem isosceles_triangle_base_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- triangle sides are positive
  a = b →                  -- isosceles triangle condition
  a = 5 →                  -- given leg length
  a + b > c →              -- triangle inequality
  c + a > b →              -- triangle inequality
  c ≠ 11                   -- base cannot be 11
  := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3755_375556


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3755_375584

/-- Given real numbers a, b, c, and a positive number m satisfying the condition,
    the quadratic equation has a root between 0 and 1. -/
theorem quadratic_root_in_unit_interval (a b c m : ℝ) (hm : m > 0) 
    (h : a / (m + 2) + b / (m + 1) + c / m = 0) :
    ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3755_375584


namespace NUMINAMATH_CALUDE_square_root_fraction_equals_one_l3755_375539

theorem square_root_fraction_equals_one : 
  Real.sqrt (3^2 + 4^2) / Real.sqrt (20 + 5) = 1 := by sorry

end NUMINAMATH_CALUDE_square_root_fraction_equals_one_l3755_375539


namespace NUMINAMATH_CALUDE_range_of_a_l3755_375504

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0) → (a > 3 ∨ a < -1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3755_375504


namespace NUMINAMATH_CALUDE_queen_probability_l3755_375575

/-- A deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_queens : ℕ)

/-- The probability of drawing a specific card from a deck -/
def draw_probability (d : Deck) (num_specific_cards : ℕ) : ℚ :=
  num_specific_cards / d.total_cards

/-- Our specific deck -/
def modified_deck : Deck :=
  { total_cards := 54,
    num_queens := 4 }

theorem queen_probability :
  draw_probability modified_deck modified_deck.num_queens = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_queen_probability_l3755_375575


namespace NUMINAMATH_CALUDE_trig_expression_evaluation_l3755_375565

open Real

theorem trig_expression_evaluation (x : ℝ) 
  (f : ℝ → ℝ) 
  (hf : f = fun x => sin x + cos x) 
  (hf' : deriv f = fun x => 3 * f x) : 
  (sin x)^2 - 3 / ((cos x)^2 + 1) = -14/9 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_evaluation_l3755_375565


namespace NUMINAMATH_CALUDE_female_average_score_l3755_375569

theorem female_average_score (total_average : ℝ) (male_average : ℝ) (male_count : ℕ) (female_count : ℕ) :
  total_average = 90 →
  male_average = 82 →
  male_count = 8 →
  female_count = 32 →
  (male_count * male_average + female_count * ((male_count + female_count) * total_average - male_count * male_average) / female_count) / (male_count + female_count) = 90 →
  ((male_count + female_count) * total_average - male_count * male_average) / female_count = 92 := by
sorry

end NUMINAMATH_CALUDE_female_average_score_l3755_375569


namespace NUMINAMATH_CALUDE_water_bill_ratio_l3755_375594

def electricity_bill : ℚ := 60
def gas_bill : ℚ := 40
def water_bill : ℚ := 40
def internet_bill : ℚ := 25

def gas_bill_paid : ℚ := (3/4) * gas_bill + 5
def internet_bill_paid : ℚ := 4 * 5

def total_remaining : ℚ := 30

def water_bill_paid : ℚ := water_bill - (total_remaining - (gas_bill - gas_bill_paid) - (internet_bill - internet_bill_paid))

theorem water_bill_ratio : water_bill_paid / water_bill = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_water_bill_ratio_l3755_375594


namespace NUMINAMATH_CALUDE_four_digit_equation_solutions_l3755_375566

/-- Represents a four-digit number ABCD as a pair of two-digit numbers (AB, CD) -/
def FourDigitNumber := Nat × Nat

/-- Checks if a pair of numbers represents a valid four-digit number -/
def isValidFourDigitNumber (n : FourDigitNumber) : Prop :=
  10 ≤ n.1 ∧ n.1 ≤ 99 ∧ 10 ≤ n.2 ∧ n.2 ≤ 99

/-- Converts a pair of two-digit numbers to a four-digit number -/
def toNumber (n : FourDigitNumber) : Nat :=
  100 * n.1 + n.2

/-- The equation that the four-digit number must satisfy -/
def satisfiesEquation (n : FourDigitNumber) : Prop :=
  toNumber n = n.1 * n.2 + n.1 * n.1

theorem four_digit_equation_solutions :
  ∀ n : FourDigitNumber, 
    isValidFourDigitNumber n ∧ satisfiesEquation n ↔ 
    n = (12, 96) ∨ n = (34, 68) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_equation_solutions_l3755_375566


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l3755_375559

-- Define the original proposition
def original_proposition (x : ℝ) : Prop := x > 2 → x > 1

-- Define the inverse proposition
def inverse_proposition (x : ℝ) : Prop := x > 1 → x > 2

-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition
theorem inverse_of_proposition :
  (∀ x, original_proposition x) ↔ (∀ x, inverse_proposition x) :=
sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l3755_375559


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l3755_375521

/-- Given two similar right triangles, where one has legs 12 and 9, and the other has legs x and 6, prove that x = 8 -/
theorem similar_triangles_leg_length : 
  ∀ x : ℝ, 
  (12 : ℝ) / x = (9 : ℝ) / 6 → 
  x = 8 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l3755_375521


namespace NUMINAMATH_CALUDE_bob_has_ten_candies_l3755_375536

/-- The number of candies Bob has after trick-or-treating. -/
def bob_candies (mary_candies sue_candies john_candies sam_candies total_candies : ℕ) : ℕ :=
  total_candies - (mary_candies + sue_candies + john_candies + sam_candies)

/-- Theorem stating that Bob has 10 candies given the conditions from the problem. -/
theorem bob_has_ten_candies :
  bob_candies 5 20 5 10 50 = 10 := by sorry

end NUMINAMATH_CALUDE_bob_has_ten_candies_l3755_375536


namespace NUMINAMATH_CALUDE_product_expansion_l3755_375582

theorem product_expansion (x : ℝ) : 5*(x-6)*(x+9) + 3*x = 5*x^2 + 18*x - 270 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3755_375582


namespace NUMINAMATH_CALUDE_left_square_side_length_l3755_375588

/-- Given three squares with specific relationships between their side lengths,
    prove that the side length of the left square is 8 cm. -/
theorem left_square_side_length (x : ℝ) : 
  x > 0 →  -- Ensure positive side length
  x + (x + 17) + (x + 11) = 52 →  -- Sum of side lengths
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_left_square_side_length_l3755_375588


namespace NUMINAMATH_CALUDE_permutation_combination_problem_l3755_375534

-- Define the permutation function
def A (n : ℕ) (k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

-- Define the combination function
def C (n : ℕ) (k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem permutation_combination_problem :
  (2 * A 8 5 + 7 * A 8 4) / (A 8 8 + A 9 5) = 5 / 11 ∧
  C 200 192 + C 200 196 + 2 * C 200 197 = 67331650 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_problem_l3755_375534


namespace NUMINAMATH_CALUDE_line_quadrant_theorem_l3755_375522

/-- Represents a line in the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a line passes through a given quadrant -/
def passes_through_quadrant (l : Line) (q : ℕ) : Prop :=
  match q with
  | 1 => ∃ x > 0, l.slope * x + l.intercept > 0
  | 2 => ∃ x < 0, l.slope * x + l.intercept > 0
  | 3 => ∃ x < 0, l.slope * x + l.intercept < 0
  | 4 => ∃ x > 0, l.slope * x + l.intercept < 0
  | _ => False

/-- The main theorem -/
theorem line_quadrant_theorem (a b : ℝ) (h1 : a < 0) (h2 : b > 0) 
  (h3 : passes_through_quadrant (Line.mk a b) 1)
  (h4 : passes_through_quadrant (Line.mk a b) 2)
  (h5 : passes_through_quadrant (Line.mk a b) 4) :
  ¬ passes_through_quadrant (Line.mk b a) 2 := by
  sorry

end NUMINAMATH_CALUDE_line_quadrant_theorem_l3755_375522


namespace NUMINAMATH_CALUDE_even_increasing_inequality_l3755_375597

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_left (f : ℝ → ℝ) : Prop := 
  ∀ x y, x ≤ y ∧ y ≤ -1 → f x ≤ f y

-- State the theorem
theorem even_increasing_inequality 
  (h_even : is_even f) 
  (h_incr : increasing_on_left f) : 
  f 2 < f (-1.5) ∧ f (-1.5) < f (-1) := by
sorry

end NUMINAMATH_CALUDE_even_increasing_inequality_l3755_375597


namespace NUMINAMATH_CALUDE_triangle_angle_sixty_degrees_l3755_375592

theorem triangle_angle_sixty_degrees (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  -- a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
  -- Given condition
  (2 * b - c) * Real.cos A = a * Real.cos C →
  -- Conclusion
  A = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sixty_degrees_l3755_375592


namespace NUMINAMATH_CALUDE_larger_number_is_322_l3755_375553

def is_hcf (a b h : ℕ) : Prop := h ∣ a ∧ h ∣ b ∧ ∀ d : ℕ, d ∣ a → d ∣ b → d ≤ h

def is_lcm (a b l : ℕ) : Prop := a ∣ l ∧ b ∣ l ∧ ∀ m : ℕ, a ∣ m → b ∣ m → l ∣ m

theorem larger_number_is_322 (a b : ℕ) (h : a > 0 ∧ b > 0) :
  is_hcf a b 23 → is_lcm a b (23 * 13 * 14) → max a b = 322 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_322_l3755_375553


namespace NUMINAMATH_CALUDE_water_left_in_cooler_l3755_375520

/-- Calculates the remaining water in a cooler after filling Dixie cups for a meeting --/
theorem water_left_in_cooler 
  (initial_gallons : ℕ) 
  (ounces_per_cup : ℕ) 
  (rows : ℕ) 
  (chairs_per_row : ℕ) 
  (ounces_per_gallon : ℕ) 
  (h1 : initial_gallons = 3)
  (h2 : ounces_per_cup = 6)
  (h3 : rows = 5)
  (h4 : chairs_per_row = 10)
  (h5 : ounces_per_gallon = 128) : 
  initial_gallons * ounces_per_gallon - rows * chairs_per_row * ounces_per_cup = 84 := by
  sorry

end NUMINAMATH_CALUDE_water_left_in_cooler_l3755_375520


namespace NUMINAMATH_CALUDE_cubic_roots_nature_l3755_375502

-- Define the cubic polynomial
def cubic_poly (x : ℝ) : ℝ := x^3 - 5*x^2 + 8*x - 4

-- Theorem statement
theorem cubic_roots_nature :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c < 0 ∧
    (∀ x : ℝ, cubic_poly x = 0 ↔ (x = a ∨ x = b ∨ x = c))) :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_nature_l3755_375502


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l3755_375591

theorem angle_sum_in_circle (x : ℝ) : 
  (6*x + 7*x + 3*x + 2*x + 4*x = 360) → x = 180/11 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l3755_375591


namespace NUMINAMATH_CALUDE_original_number_proof_l3755_375533

theorem original_number_proof (x : ℝ) : 
  268 * x = 19832 ∧ 2.68 * x = 1.9832 → x = 74 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3755_375533


namespace NUMINAMATH_CALUDE_heron_height_calculation_l3755_375516

theorem heron_height_calculation (a b c : ℝ) (h : ℝ) :
  a = 20 ∧ b = 99 ∧ c = 101 →
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  h = 2 * area / b →
  h = 20 := by
  sorry

end NUMINAMATH_CALUDE_heron_height_calculation_l3755_375516


namespace NUMINAMATH_CALUDE_intersection_value_l3755_375550

theorem intersection_value (A B : Set ℝ) (m : ℝ) : 
  A = {-1, 1, 3} → 
  B = {1, m} → 
  A ∩ B = {1, 3} → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_value_l3755_375550


namespace NUMINAMATH_CALUDE_arun_weight_average_l3755_375561

-- Define Arun's weight as a real number
def arun_weight : ℝ := sorry

-- Define the conditions on Arun's weight
def condition1 : Prop := 65 < arun_weight ∧ arun_weight < 72
def condition2 : Prop := 60 < arun_weight ∧ arun_weight < 70
def condition3 : Prop := arun_weight ≤ 68

-- Theorem to prove
theorem arun_weight_average :
  condition1 ∧ condition2 ∧ condition3 →
  (65 + 68) / 2 = 66.5 :=
by sorry

end NUMINAMATH_CALUDE_arun_weight_average_l3755_375561


namespace NUMINAMATH_CALUDE_indeterminate_or_l3755_375599

theorem indeterminate_or (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : 
  ¬∀ (b : Bool), (p ∨ q) = b :=
by
  sorry

end NUMINAMATH_CALUDE_indeterminate_or_l3755_375599


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_equivalence_l3755_375578

/-- Represents a quadratic function in the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a quadratic function in vertex form a(x - m)² + n -/
structure VertexForm where
  a : ℝ
  m : ℝ
  n : ℝ

/-- The vertex of a quadratic function -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Theorem stating the equivalence of the standard and vertex forms of a specific quadratic function,
    and identifying its vertex -/
theorem quadratic_vertex_form_equivalence :
  let f : QuadraticFunction := { a := 2, b := -12, c := -12 }
  let v : VertexForm := { a := 2, m := 3, n := -30 }
  let vertex : Vertex := { x := 3, y := -30 }
  (∀ x, 2 * x^2 - 12 * x - 12 = 2 * (x - 3)^2 - 30) ∧
  (vertex.x = -f.b / (2 * f.a) ∧ vertex.y = f.c - f.b^2 / (4 * f.a)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_equivalence_l3755_375578


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3755_375557

/-- The arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

/-- The absolute difference between two integers -/
def absDiff (a b : ℤ) : ℕ := (a - b).natAbs

theorem arithmetic_sequence_difference :
  let a := -10  -- First term of the sequence
  let d := 11   -- Common difference of the sequence
  absDiff (arithmeticSequence a d 2025) (arithmeticSequence a d 2010) = 165 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3755_375557


namespace NUMINAMATH_CALUDE_angle_measure_problem_l3755_375515

/-- Given two supplementary angles C and D, where C is 5 times D, prove that the measure of angle C is 150°. -/
theorem angle_measure_problem (C D : ℝ) : 
  C + D = 180 →  -- C and D are supplementary
  C = 5 * D →    -- C is 5 times D
  C = 150 :=     -- The measure of angle C is 150°
by
  sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l3755_375515
