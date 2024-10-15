import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1059_105966

/-- Given that a, b, c, and d form a geometric sequence,
    prove that a+b, b+c, c+d form a geometric sequence -/
theorem geometric_sequence_sum (a b c d : ℝ) 
  (h : ∃ (r : ℝ), b = a * r ∧ c = b * r ∧ d = c * r) : 
  ∃ (q : ℝ), (b + c) = (a + b) * q ∧ (c + d) = (b + c) * q := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1059_105966


namespace NUMINAMATH_CALUDE_function_zeros_count_l1059_105950

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_zeros_count
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_periodic : is_periodic f 3)
  (h_sin : ∀ x ∈ Set.Ioo 0 (3/2), f x = Real.sin (Real.pi * x))
  (h_zero : f (3/2) = 0) :
  ∃ S : Finset ℝ, S.card = 7 ∧ (∀ x ∈ S, x ∈ Set.Icc 0 6 ∧ f x = 0) ∧
    (∀ x ∈ Set.Icc 0 6, f x = 0 → x ∈ S) :=
sorry

end NUMINAMATH_CALUDE_function_zeros_count_l1059_105950


namespace NUMINAMATH_CALUDE_discount_percentage_l1059_105942

def ticket_price : ℝ := 25
def sale_price : ℝ := 18.75

theorem discount_percentage : 
  (ticket_price - sale_price) / ticket_price * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l1059_105942


namespace NUMINAMATH_CALUDE_islander_liar_count_l1059_105918

/-- Represents the type of islander: either a knight or a liar -/
inductive IslanderType
| Knight
| Liar

/-- Represents a group of islanders making a statement -/
structure IslanderGroup where
  size : Nat
  statement : Nat

/-- The total number of islanders -/
def totalIslanders : Nat := 19

/-- The three groups of islanders making statements -/
def groups : List IslanderGroup := [
  { size := 3, statement := 3 },
  { size := 6, statement := 6 },
  { size := 9, statement := 9 }
]

/-- Determines if a statement is true given the actual number of liars -/
def isStatementTrue (statement : Nat) (actualLiars : Nat) : Bool :=
  statement == actualLiars

/-- Determines if an islander is telling the truth based on their type and statement -/
def isTellingTruth (type : IslanderType) (statementTrue : Bool) : Bool :=
  match type with
  | IslanderType.Knight => statementTrue
  | IslanderType.Liar => ¬statementTrue

/-- The main theorem to prove -/
theorem islander_liar_count :
  ∀ (liarCount : Nat),
  (liarCount ≤ totalIslanders) →
  (∀ (group : IslanderGroup),
    group ∈ groups →
    (∀ (type : IslanderType),
      (isTellingTruth type (isStatementTrue group.statement liarCount)) →
      (type = IslanderType.Knight ↔ liarCount = group.statement))) →
  (liarCount = 9 ∨ liarCount = 18 ∨ liarCount = 19) :=
sorry

end NUMINAMATH_CALUDE_islander_liar_count_l1059_105918


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l1059_105995

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 10)
  (h2 : sum_first_two = 6) :
  ∃ (a : ℝ), 
    (a = 10 - 10 * Real.sqrt (2/5) ∨ a = 10 + 10 * Real.sqrt (2/5)) ∧ 
    (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l1059_105995


namespace NUMINAMATH_CALUDE_product_b3_b17_l1059_105976

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- The main theorem -/
theorem product_b3_b17 (a b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_cond : 3 * a 1 - (a 8)^2 + 3 * a 15 = 0)
  (h_eq : a 8 = b 10) :
  b 3 * b 17 = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_b3_b17_l1059_105976


namespace NUMINAMATH_CALUDE_decimal_to_binary_51_l1059_105968

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec toBinaryAux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
    toBinaryAux n

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 51

/-- The expected binary representation -/
def expectedBinary : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the binary representation of 51 is [1,1,0,0,1,1] -/
theorem decimal_to_binary_51 : toBinary decimalNumber = expectedBinary := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_51_l1059_105968


namespace NUMINAMATH_CALUDE_area_ratio_bounds_l1059_105978

-- Define an equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a line passing through the centroid
structure CentroidLine where
  angle : ℝ  -- Angle of the line with respect to a reference

-- Define the two parts created by the line
structure TriangleParts where
  part1 : ℝ
  part2 : ℝ
  parts_positive : part1 > 0 ∧ part2 > 0
  parts_sum : part1 + part2 = 1  -- Normalized to total area 1

-- Main theorem
theorem area_ratio_bounds (t : EquilateralTriangle) (l : CentroidLine) 
  (p : TriangleParts) : 
  4/5 ≤ min (p.part1 / p.part2) (p.part2 / p.part1) ∧ 
  max (p.part1 / p.part2) (p.part2 / p.part1) ≤ 5/4 :=
sorry

end NUMINAMATH_CALUDE_area_ratio_bounds_l1059_105978


namespace NUMINAMATH_CALUDE_project_rotation_lcm_l1059_105958

theorem project_rotation_lcm : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 12)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_project_rotation_lcm_l1059_105958


namespace NUMINAMATH_CALUDE_zeros_in_square_expansion_l1059_105945

theorem zeros_in_square_expansion (n : ℕ) : 
  (∃ k : ℕ, (10^15 - 3)^2 = k * 10^n ∧ k % 10 ≠ 0) → n = 29 :=
sorry

end NUMINAMATH_CALUDE_zeros_in_square_expansion_l1059_105945


namespace NUMINAMATH_CALUDE_ellipse_cosine_theorem_l1059_105996

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let a := 2
  let c := Real.sqrt 3
  F₁ = (-c, 0) ∧ F₂ = (c, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ellipse x y

-- Define the distance ratio condition
def distance_ratio (P F₁ F₂ : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  d₁ = 3 * d₂

-- Theorem statement
theorem ellipse_cosine_theorem (P F₁ F₂ : ℝ × ℝ) :
  foci F₁ F₂ →
  point_on_ellipse P →
  distance_ratio P F₁ F₂ →
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  let d₃ := Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)
  (d₁^2 + d₂^2 - d₃^2) / (2 * d₁ * d₂) = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_cosine_theorem_l1059_105996


namespace NUMINAMATH_CALUDE_tax_reduction_theorem_l1059_105980

theorem tax_reduction_theorem (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 100) : 
  (1 - x / 100) * 1.1 = 0.825 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_theorem_l1059_105980


namespace NUMINAMATH_CALUDE_triangle_inequality_inner_point_l1059_105957

/-- Given a triangle ABC and a point P on side AB, prove that PC · AB < PA · BC + PB · AC -/
theorem triangle_inequality_inner_point (A B C P : ℝ × ℝ) : 
  (P.1 > A.1 ∧ P.1 < B.1) → -- P is an inner point of AB
  (dist P C * dist A B < dist P A * dist B C + dist P B * dist A C) := by
  sorry

#check triangle_inequality_inner_point

end NUMINAMATH_CALUDE_triangle_inequality_inner_point_l1059_105957


namespace NUMINAMATH_CALUDE_original_fraction_is_two_thirds_l1059_105936

theorem original_fraction_is_two_thirds :
  ∀ (a b : ℕ), 
    a ≠ 0 → b ≠ 0 →
    (a^3 : ℚ) / (b + 3 : ℚ) = 2 * (a : ℚ) / (b : ℚ) →
    (∀ d : ℕ, d ≠ 0 → d ∣ a ∧ d ∣ b → d = 1) →
    a = 2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_is_two_thirds_l1059_105936


namespace NUMINAMATH_CALUDE_washers_remaining_l1059_105970

/-- Calculates the number of washers remaining after a plumbing job. -/
theorem washers_remaining
  (total_pipe_length : ℕ)
  (pipe_per_bolt : ℕ)
  (washers_per_bolt : ℕ)
  (initial_washers : ℕ)
  (h1 : total_pipe_length = 40)
  (h2 : pipe_per_bolt = 5)
  (h3 : washers_per_bolt = 2)
  (h4 : initial_washers = 20) :
  initial_washers - (total_pipe_length / pipe_per_bolt * washers_per_bolt) = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_washers_remaining_l1059_105970


namespace NUMINAMATH_CALUDE_exam_disturbance_probability_l1059_105917

theorem exam_disturbance_probability :
  let n : ℕ := 6  -- number of students
  let p_undisturbed : ℚ := 2 / n * 2 / (n - 1) * 2 / (n - 2) * 2 / (n - 3)
  (1 : ℚ) - p_undisturbed = 43 / 45 :=
by sorry

end NUMINAMATH_CALUDE_exam_disturbance_probability_l1059_105917


namespace NUMINAMATH_CALUDE_picture_area_l1059_105959

theorem picture_area (x y : ℤ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : (3*x + 2)*(y + 4) - x*y = 62) : 
  x * y = 10 := by
sorry

end NUMINAMATH_CALUDE_picture_area_l1059_105959


namespace NUMINAMATH_CALUDE_square_difference_equal_l1059_105964

theorem square_difference_equal (a b : ℝ) : (a - b)^2 = (b - a)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equal_l1059_105964


namespace NUMINAMATH_CALUDE_square_pentagon_side_ratio_l1059_105951

theorem square_pentagon_side_ratio (perimeter : ℝ) (square_side : ℝ) (pentagon_side : ℝ)
  (h1 : perimeter > 0)
  (h2 : 4 * square_side = perimeter)
  (h3 : 5 * pentagon_side = perimeter) :
  pentagon_side / square_side = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_square_pentagon_side_ratio_l1059_105951


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1059_105962

theorem problem_1 (x y : ℝ) : 3 * (x - y)^2 - 6 * (x - y)^2 + 2 * (x - y)^2 = -(x - y)^2 := by
  sorry

theorem problem_2 (a b : ℝ) (h : a^2 - 2*b = 2) : 4*a^2 - 8*b - 9 = -1 := by
  sorry

theorem problem_3 (a b c d : ℝ) (h1 : a - 2*b = 4) (h2 : b - c = -5) (h3 : 3*c + d = 10) :
  (a + 3*c) - (2*b + c) + (b + d) = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1059_105962


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l1059_105924

theorem min_value_exponential_sum (x : ℝ) : 16^x + 4^x - 2^x + 1 ≥ (3/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l1059_105924


namespace NUMINAMATH_CALUDE_train_length_proof_l1059_105911

/-- The length of each train in meters -/
def train_length : ℝ := 62.5

/-- The speed of the faster train in km/hr -/
def fast_train_speed : ℝ := 46

/-- The speed of the slower train in km/hr -/
def slow_train_speed : ℝ := 36

/-- The time taken for the faster train to completely pass the slower train in seconds -/
def overtake_time : ℝ := 45

theorem train_length_proof :
  let relative_speed := (fast_train_speed - slow_train_speed) * 1000 / 3600
  let distance_covered := relative_speed * overtake_time
  2 * train_length = distance_covered := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l1059_105911


namespace NUMINAMATH_CALUDE_positive_integer_expression_l1059_105953

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem positive_integer_expression (m n : ℕ) :
  ∃ k : ℕ+, k = (factorial (2 * m) * factorial (2 * n)) / (factorial m * factorial n * factorial (m + n)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_expression_l1059_105953


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1059_105902

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π/4 - α) = -3/5) : 
  Real.sin (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1059_105902


namespace NUMINAMATH_CALUDE_shirt_price_proof_l1059_105961

theorem shirt_price_proof (P : ℝ) : 
  (0.75 * (0.75 * P) = 18) → P = 32 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_proof_l1059_105961


namespace NUMINAMATH_CALUDE_insurance_coverage_percentage_l1059_105969

theorem insurance_coverage_percentage
  (frames_cost : ℝ)
  (lenses_cost : ℝ)
  (coupon_value : ℝ)
  (final_cost : ℝ)
  (h1 : frames_cost = 200)
  (h2 : lenses_cost = 500)
  (h3 : coupon_value = 50)
  (h4 : final_cost = 250) :
  (((frames_cost + lenses_cost - coupon_value) - final_cost) / lenses_cost) * 100 = 80 :=
by sorry

end NUMINAMATH_CALUDE_insurance_coverage_percentage_l1059_105969


namespace NUMINAMATH_CALUDE_intersection_M_N_l1059_105920

def M : Set ℕ := {x | x > 0 ∧ x ≤ 2}
def N : Set ℕ := {2, 6}

theorem intersection_M_N : M ∩ N = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1059_105920


namespace NUMINAMATH_CALUDE_range_of_trig_function_l1059_105930

open Real

theorem range_of_trig_function :
  let f : ℝ → ℝ := λ x ↦ 2 * cos x + sin (2 * x)
  ∃ (a b : ℝ), a = -3 * Real.sqrt 3 / 2 ∧ b = 3 * Real.sqrt 3 / 2 ∧
    (∀ x, f x ∈ Set.Icc a b) ∧
    (∀ y ∈ Set.Icc a b, ∃ x, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_range_of_trig_function_l1059_105930


namespace NUMINAMATH_CALUDE_sum_of_bases_l1059_105973

-- Define the fractions F₁ and F₂
def F₁ (R : ℕ) : ℚ := (4 * R + 5) / (R^2 - 1)
def F₂ (R : ℕ) : ℚ := (5 * R + 4) / (R^2 - 1)

-- Define the conditions
def condition1 (R₁ : ℕ) : Prop := F₁ R₁ = 5 / 11
def condition2 (R₁ : ℕ) : Prop := F₂ R₁ = 6 / 11
def condition3 (R₂ : ℕ) : Prop := F₁ R₂ = 3 / 7
def condition4 (R₂ : ℕ) : Prop := F₂ R₂ = 4 / 7

-- State the theorem
theorem sum_of_bases (R₁ R₂ : ℕ) :
  condition1 R₁ → condition2 R₁ → condition3 R₂ → condition4 R₂ → R₁ + R₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_bases_l1059_105973


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l1059_105922

theorem number_exceeding_percentage : ∃ x : ℝ, x = 0.16 * x + 105 ∧ x = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l1059_105922


namespace NUMINAMATH_CALUDE_max_xy_value_l1059_105941

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 2 * y = 110) : x * y ≤ 216 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l1059_105941


namespace NUMINAMATH_CALUDE_unique_q_value_l1059_105943

theorem unique_q_value (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p*q = 4) : q = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_q_value_l1059_105943


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l1059_105998

/-- A dodecahedron is a 3-dimensional figure with specific properties -/
structure Dodecahedron where
  faces : ℕ
  vertices : ℕ
  faces_per_vertex : ℕ
  faces_are_pentagonal : Prop
  h_faces : faces = 12
  h_vertices : vertices = 20
  h_faces_per_vertex : faces_per_vertex = 3

/-- The number of interior diagonals in a dodecahedron -/
def interior_diagonals (d : Dodecahedron) : ℕ :=
  (d.vertices * (d.vertices - 1 - d.faces_per_vertex)) / 2

/-- Theorem stating the number of interior diagonals in a dodecahedron -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  interior_diagonals d = 160 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l1059_105998


namespace NUMINAMATH_CALUDE_percentage_reduction_proof_price_increase_proof_l1059_105908

-- Define the initial price
def initial_price : ℝ := 50

-- Define the final price after two reductions
def final_price : ℝ := 32

-- Define the profit per kilogram before price increase
def initial_profit : ℝ := 10

-- Define the initial daily sales volume
def initial_sales : ℝ := 500

-- Define the target daily profit
def target_profit : ℝ := 6000

-- Define the maximum allowed price increase
def max_price_increase : ℝ := 8

-- Define the sales volume decrease per yuan of price increase
def sales_decrease_rate : ℝ := 20

-- Theorem for the percentage reduction
theorem percentage_reduction_proof :
  ∃ x : ℝ, x > 0 ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price ∧ x = 1/5 :=
sorry

-- Theorem for the required price increase
theorem price_increase_proof :
  ∃ y : ℝ, 0 < y ∧ y ≤ max_price_increase ∧
  (initial_profit + y) * (initial_sales - sales_decrease_rate * y) = target_profit ∧
  y = 5 :=
sorry

end NUMINAMATH_CALUDE_percentage_reduction_proof_price_increase_proof_l1059_105908


namespace NUMINAMATH_CALUDE_students_playing_neither_l1059_105915

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 35 →
  football = 26 →
  tennis = 20 →
  both = 17 →
  total - (football + tennis - both) = 6 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_neither_l1059_105915


namespace NUMINAMATH_CALUDE_special_function_sum_property_l1059_105960

/-- A function satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (f 0 = 0) ∧
  (∀ x y, x ∈ ({x | x < -1} ∪ {x | x > 1}) → 
          y ∈ ({x | x < -1} ∪ {x | x > 1}) → 
          f (1/x) + f (1/y) = f ((x+y)/(1+x*y))) ∧
  (∀ x, x ∈ {x | -1 < x ∧ x < 0} → f x > 0)

/-- The theorem to be proved -/
theorem special_function_sum_property (f : ℝ → ℝ) (hf : SpecialFunction f) :
  ∑' (n : ℕ), f (1 / (n^2 + 7*n + 11)) > f (1/2) := by
  sorry

end NUMINAMATH_CALUDE_special_function_sum_property_l1059_105960


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l1059_105926

/-- Calculates the cost of gas per gallon given Carla's trip details --/
theorem gas_cost_per_gallon 
  (distance_to_grocery : ℝ) 
  (distance_to_school : ℝ) 
  (distance_to_soccer : ℝ) 
  (miles_per_gallon : ℝ) 
  (total_gas_cost : ℝ) 
  (h1 : distance_to_grocery = 8) 
  (h2 : distance_to_school = 6) 
  (h3 : distance_to_soccer = 12) 
  (h4 : miles_per_gallon = 25) 
  (h5 : total_gas_cost = 5) :
  (total_gas_cost / ((distance_to_grocery + distance_to_school + distance_to_soccer + 2 * distance_to_soccer) / miles_per_gallon)) = 2.5 := by
sorry

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l1059_105926


namespace NUMINAMATH_CALUDE_trapezoid_existence_l1059_105974

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A set of marked vertices in a polygon -/
def MarkedVertices (n : ℕ) (m : ℕ) := Fin m → Fin n

/-- Four points form a trapezoid if two sides are parallel and not all four points are collinear -/
def IsTrapezoid (a b c d : ℝ × ℝ) : Prop := sorry

theorem trapezoid_existence (polygon : RegularPolygon 2015) (marked : MarkedVertices 2015 64) :
  ∃ (a b c d : Fin 64), IsTrapezoid (polygon.vertices (marked a)) (polygon.vertices (marked b)) 
                                    (polygon.vertices (marked c)) (polygon.vertices (marked d)) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_existence_l1059_105974


namespace NUMINAMATH_CALUDE_expression_undefined_at_13_l1059_105982

-- Define the numerator and denominator of the expression
def numerator (x : ℝ) : ℝ := 3 * x^3 - 5
def denominator (x : ℝ) : ℝ := x^2 - 26 * x + 169

-- Theorem stating that the expression is undefined when x = 13
theorem expression_undefined_at_13 : denominator 13 = 0 := by sorry

end NUMINAMATH_CALUDE_expression_undefined_at_13_l1059_105982


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l1059_105923

theorem consecutive_integers_sqrt_3 (a b : ℤ) : 
  (b = a + 1) → (↑a < Real.sqrt 3 ∧ Real.sqrt 3 < ↑b) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l1059_105923


namespace NUMINAMATH_CALUDE_student_square_substitution_l1059_105921

theorem student_square_substitution (a b : ℕ) : 
  (a + 2 * b - 3)^2 = a^2 + 4 * b^2 - 9 → a = 3 ∧ ∀ n : ℕ, (3 + 2 * n - 3)^2 = 3^2 + 4 * n^2 - 9 :=
by sorry

end NUMINAMATH_CALUDE_student_square_substitution_l1059_105921


namespace NUMINAMATH_CALUDE_x_over_y_value_l1059_105954

theorem x_over_y_value (x y z : ℝ) 
  (eq1 : x + y = 2 * x + z)
  (eq2 : x - 2 * y = 4 * z)
  (eq3 : x + y + z = 21)
  (eq4 : y / z = 6) :
  x / y = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_value_l1059_105954


namespace NUMINAMATH_CALUDE_mike_ride_length_l1059_105901

-- Define the taxi fare structure
structure TaxiFare where
  start_fee : ℝ
  per_mile_fee : ℝ
  toll_fee : ℝ

-- Define the problem parameters
def mike_fare : TaxiFare := ⟨2.5, 0.25, 0⟩
def annie_fare : TaxiFare := ⟨2.5, 0.25, 5⟩
def annie_miles : ℝ := 14

-- Define the function to calculate the total fare
def total_fare (fare : TaxiFare) (miles : ℝ) : ℝ :=
  fare.start_fee + fare.per_mile_fee * miles + fare.toll_fee

-- Theorem statement
theorem mike_ride_length :
  ∃ (mike_miles : ℝ),
    total_fare mike_fare mike_miles = total_fare annie_fare annie_miles ∧
    mike_miles = 34 := by
  sorry

end NUMINAMATH_CALUDE_mike_ride_length_l1059_105901


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1059_105999

theorem inequality_system_solution : 
  let S := {x : ℤ | x > 0 ∧ 5 + 3*x < 13 ∧ (x+2)/3 - (x-1)/2 ≤ 2}
  S = {1, 2} := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1059_105999


namespace NUMINAMATH_CALUDE_count_nines_to_800_l1059_105975

/-- Count of digit 9 occurrences in integers from 1 to n -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Theorem: The count of digit 9 occurrences in integers from 1 to 800 is 160 -/
theorem count_nines_to_800 : count_nines 800 = 160 := by sorry

end NUMINAMATH_CALUDE_count_nines_to_800_l1059_105975


namespace NUMINAMATH_CALUDE_equation_solution_l1059_105937

theorem equation_solution : 
  ∃! x : ℝ, (3*x - 2 ≥ 0) ∧ (Real.sqrt (3*x - 2) + 9 / Real.sqrt (3*x - 2) = 6) ∧ (x = 11/3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1059_105937


namespace NUMINAMATH_CALUDE_painting_time_proof_l1059_105900

/-- Given a painting job with a total number of rooms, time per room, and rooms already painted,
    calculate the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Theorem stating that for the given painting scenario, the time to paint the remaining rooms is 16 hours. -/
theorem painting_time_proof :
  time_to_paint_remaining 10 8 8 = 16 := by
  sorry


end NUMINAMATH_CALUDE_painting_time_proof_l1059_105900


namespace NUMINAMATH_CALUDE_f_properties_l1059_105992

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ a then (1/a) * x
  else if a < x ∧ x ≤ 1 then (1/(1-a)) * (1-x)
  else 0

def is_turning_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f (f x) = x ∧ f x ≠ x

theorem f_properties (a : ℝ) (h : 0 < a ∧ a < 1) :
  (f (1/2) (f (1/2) (4/5)) = 4/5 ∧ is_turning_point (f (1/2)) (4/5)) ∧
  (∀ x : ℝ, a < x → x ≤ 1 →
    f a (f a x) = if x < a^2 - a + 1
                  then 1/(1-a) * (1 - 1/(1-a) * (1-x))
                  else 1/(a*(1-a)) * (1-x)) ∧
  (is_turning_point (f a) (1/(2-a)) ∧ is_turning_point (f a) (1/(1+a-a^2))) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l1059_105992


namespace NUMINAMATH_CALUDE_inverse_mod_59_l1059_105984

theorem inverse_mod_59 (h : (17⁻¹ : ZMod 59) = 23) : (42⁻¹ : ZMod 59) = 36 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_59_l1059_105984


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l1059_105906

theorem smallest_n_satisfying_inequality : ∃ n : ℕ, 
  (∀ m : ℕ, 2006^1003 < m^2006 → n ≤ m) ∧ 2006^1003 < n^2006 ∧ n = 45 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l1059_105906


namespace NUMINAMATH_CALUDE_cm_per_inch_l1059_105928

/-- Theorem: Given the map scale and measured distance, prove the number of centimeters in one inch -/
theorem cm_per_inch (map_scale_inches : Real) (map_scale_miles : Real) 
  (measured_cm : Real) (measured_miles : Real) :
  map_scale_inches = 1.5 →
  map_scale_miles = 24 →
  measured_cm = 47 →
  measured_miles = 296.06299212598424 →
  (measured_cm / (measured_miles / (map_scale_miles / map_scale_inches))) = 2.54 :=
by sorry

end NUMINAMATH_CALUDE_cm_per_inch_l1059_105928


namespace NUMINAMATH_CALUDE_initial_deck_size_l1059_105939

theorem initial_deck_size (red_cards : ℕ) (black_cards : ℕ) : 
  (red_cards : ℚ) / (red_cards + black_cards) = 1/3 →
  (red_cards : ℚ) / (red_cards + black_cards + 4) = 1/4 →
  red_cards + black_cards = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_deck_size_l1059_105939


namespace NUMINAMATH_CALUDE_final_mixture_concentration_l1059_105904

/-- Represents a vessel with a given capacity and alcohol concentration -/
structure Vessel where
  capacity : ℝ
  alcoholConcentration : ℝ

/-- Calculates the amount of alcohol in a vessel -/
def alcoholAmount (v : Vessel) : ℝ := v.capacity * v.alcoholConcentration

/-- Theorem: The concentration of the final mixture is (5.75 / 18) * 100% -/
theorem final_mixture_concentration 
  (vessel1 : Vessel)
  (vessel2 : Vessel)
  (vessel3 : Vessel)
  (vessel4 : Vessel)
  (finalVessel : ℝ) :
  vessel1.capacity = 2 →
  vessel1.alcoholConcentration = 0.25 →
  vessel2.capacity = 6 →
  vessel2.alcoholConcentration = 0.40 →
  vessel3.capacity = 3 →
  vessel3.alcoholConcentration = 0.55 →
  vessel4.capacity = 4 →
  vessel4.alcoholConcentration = 0.30 →
  finalVessel = 18 →
  (alcoholAmount vessel1 + alcoholAmount vessel2 + alcoholAmount vessel3 + alcoholAmount vessel4) / finalVessel = 5.75 / 18 := by
  sorry

#eval (5.75 / 18) * 100 -- Approximately 31.94%

end NUMINAMATH_CALUDE_final_mixture_concentration_l1059_105904


namespace NUMINAMATH_CALUDE_triangle_inequality_proof_l1059_105956

theorem triangle_inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > 
  a^3 + b^3 + c^3 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_proof_l1059_105956


namespace NUMINAMATH_CALUDE_fair_spending_l1059_105971

theorem fair_spending (initial_amount : ℝ) (ride_fraction : ℝ) (dessert_cost : ℝ) : 
  initial_amount = 30 →
  ride_fraction = 1/2 →
  dessert_cost = 5 →
  initial_amount - (ride_fraction * initial_amount) - dessert_cost = 10 := by
sorry

end NUMINAMATH_CALUDE_fair_spending_l1059_105971


namespace NUMINAMATH_CALUDE_computer_price_proof_l1059_105916

theorem computer_price_proof (P : ℝ) 
  (h1 : 1.3 * P = 364)
  (h2 : 2 * P = 560) : 
  P = 280 := by sorry

end NUMINAMATH_CALUDE_computer_price_proof_l1059_105916


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1059_105934

/-- Given a line 5x + 12y = 60, the minimum distance from the origin (0, 0) to any point (x, y) on this line is 60/13 -/
theorem min_distance_to_line : 
  let line := {(x, y) : ℝ × ℝ | 5 * x + 12 * y = 60}
  ∃ (d : ℝ), d = 60 / 13 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ line → 
      d ≤ Real.sqrt ((p.1 ^ 2) + (p.2 ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1059_105934


namespace NUMINAMATH_CALUDE_camp_wonka_ratio_l1059_105988

theorem camp_wonka_ratio : 
  ∀ (total_campers : ℕ) (boys girls : ℕ) (marshmallows : ℕ),
    total_campers = 96 →
    girls = total_campers / 3 →
    boys = total_campers - girls →
    marshmallows = 56 →
    (boys : ℚ) * (1/2) + (girls : ℚ) * (3/4) = marshmallows →
    (boys : ℚ) / (total_campers : ℚ) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_camp_wonka_ratio_l1059_105988


namespace NUMINAMATH_CALUDE_inequality_generalization_l1059_105991

theorem inequality_generalization (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + n^n / x^n ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_generalization_l1059_105991


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1059_105912

/-- The number of dots on each side of the square grid -/
def gridSize : ℕ := 5

/-- The number of different rectangles that can be formed in a square grid -/
def numRectangles (n : ℕ) : ℕ := (n + 1).choose 2 * (n + 1).choose 2

/-- Theorem stating the number of rectangles in a 5x5 grid -/
theorem rectangles_in_5x5_grid : numRectangles gridSize = 225 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l1059_105912


namespace NUMINAMATH_CALUDE_cookie_jar_solution_l1059_105955

def cookie_jar_problem (initial_amount doris_spent martha_spent remaining : ℕ) : Prop :=
  doris_spent = 6 ∧
  martha_spent = doris_spent / 2 ∧
  remaining = 15 ∧
  initial_amount = doris_spent + martha_spent + remaining

theorem cookie_jar_solution :
  ∃ initial_amount doris_spent martha_spent remaining,
    cookie_jar_problem initial_amount doris_spent martha_spent remaining ∧
    initial_amount = 24 := by sorry

end NUMINAMATH_CALUDE_cookie_jar_solution_l1059_105955


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_product_l1059_105905

theorem largest_divisor_of_consecutive_even_product : ∃ (n : ℕ), 
  (∀ (k : ℕ), k > 0 → 16 ∣ (2*k) * (2*k + 2) * (2*k + 4)) ∧ 
  (∀ (m : ℕ), m > 16 → ∃ (j : ℕ), j > 0 ∧ ¬(m ∣ (2*j) * (2*j + 2) * (2*j + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_product_l1059_105905


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1059_105987

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1059_105987


namespace NUMINAMATH_CALUDE_seating_probability_l1059_105967

/-- Represents the number of delegates -/
def num_delegates : ℕ := 9

/-- Represents the number of countries -/
def num_countries : ℕ := 3

/-- Represents the number of delegates per country -/
def delegates_per_country : ℕ := 3

/-- Calculates the total number of seating arrangements -/
def total_arrangements : ℕ := (num_delegates.factorial) / ((delegates_per_country.factorial) ^ num_countries)

/-- Calculates the number of unwanted arrangements (where at least one country's delegates sit together) -/
def unwanted_arrangements : ℕ := 
  num_countries * num_delegates * ((num_delegates - delegates_per_country).factorial / ((delegates_per_country.factorial) ^ (num_countries - 1))) -
  (num_countries.choose 2) * num_delegates * (num_delegates - 2 * delegates_per_country + 1) +
  num_delegates * 2

/-- The probability that each delegate sits next to at least one delegate from another country -/
def probability : ℚ := (total_arrangements - unwanted_arrangements : ℚ) / total_arrangements

theorem seating_probability : probability = 41 / 56 := by
  sorry

end NUMINAMATH_CALUDE_seating_probability_l1059_105967


namespace NUMINAMATH_CALUDE_probability_less_than_20_l1059_105947

theorem probability_less_than_20 (total : ℕ) (over_30 : ℕ) (h1 : total = 120) (h2 : over_30 = 90) :
  let under_20 := total - over_30
  (under_20 : ℚ) / total = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_20_l1059_105947


namespace NUMINAMATH_CALUDE_fraction_simplification_l1059_105919

theorem fraction_simplification :
  (6 + 6 + 6 + 6) / ((-2) * (-2) * (-2) * (-2)) = (4 * 6) / ((-2)^4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1059_105919


namespace NUMINAMATH_CALUDE_min_prime_sum_l1059_105914

theorem min_prime_sum (m n p : ℕ) : 
  m.Prime ∧ n.Prime ∧ p.Prime →
  ∃ k : ℕ, k = 47 + m ∧ k = 53 + n ∧ k = 71 + p →
  m + n + p ≥ 57 :=
by sorry

end NUMINAMATH_CALUDE_min_prime_sum_l1059_105914


namespace NUMINAMATH_CALUDE_unique_solution_n_times_s_l1059_105933

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x * f y + 2 * x) = 2 * x * y + f x

/-- The theorem stating that f(3) = -2 is the only solution -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 3 = -2 := by
  sorry

/-- The number of possible values for f(3) -/
def n : ℕ := 1

/-- The sum of all possible values for f(3) -/
def s : ℝ := -2

/-- The product of n and s -/
theorem n_times_s : n * s = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_n_times_s_l1059_105933


namespace NUMINAMATH_CALUDE_circle_symmetry_l1059_105948

/-- Given two circles and a line of symmetry, prove the value of a parameter -/
theorem circle_symmetry (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - a*x + 2*y + 1 = 0 ↔ 
    ∃ x' y' : ℝ, x'^2 + y'^2 = 1 ∧ 
    (x + x')/2 - (y + y')/2 = 1 ∧
    (x - x')^2 + (y - y')^2 = ((x + x')/2 - x)^2 + ((y + y')/2 - y)^2) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1059_105948


namespace NUMINAMATH_CALUDE_union_of_sets_l1059_105913

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem union_of_sets :
  ∃ x : ℝ, (A x ∩ B x = {9}) → (A x ∪ B x = {-8, -7, -4, 4, 9}) := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1059_105913


namespace NUMINAMATH_CALUDE_unique_factorization_2210_l1059_105993

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_factorization (a b : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧ a * b = 2210

theorem unique_factorization_2210 :
  ∃! p : ℕ × ℕ, valid_factorization p.1 p.2 ∧ p.1 ≤ p.2 :=
sorry

end NUMINAMATH_CALUDE_unique_factorization_2210_l1059_105993


namespace NUMINAMATH_CALUDE_pages_left_to_read_l1059_105981

theorem pages_left_to_read (total_pages read_pages : ℕ) : 
  total_pages = 17 → read_pages = 11 → total_pages - read_pages = 6 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l1059_105981


namespace NUMINAMATH_CALUDE_point_on_line_l1059_105979

/-- Given a line with equation x = 2y + 3 and two points (m, n) and (m + 2, n + k) on this line,
    prove that k = 0. -/
theorem point_on_line (m n k : ℝ) : 
  (m = 2 * n + 3) → 
  (m + 2 = 2 * (n + k) + 3) → 
  k = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1059_105979


namespace NUMINAMATH_CALUDE_statement_D_not_always_true_l1059_105944

-- Define the space
variable (Space : Type)

-- Define lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the specific lines and planes
variable (a b c : Line)
variable (α β : Plane)

-- State the theorem
theorem statement_D_not_always_true :
  ¬(∀ (b c : Line) (α : Plane),
    (subset b α ∧ ¬subset c α ∧ parallel_line_plane c α) → parallel b c) :=
by sorry

end NUMINAMATH_CALUDE_statement_D_not_always_true_l1059_105944


namespace NUMINAMATH_CALUDE_water_formed_moles_l1059_105977

-- Define the chemical species
inductive ChemicalSpecies
| NaOH
| HCl
| H2O
| NaCl

-- Define a function to represent the stoichiometric coefficient in the balanced equation
def stoichiometric_coefficient (reactant product : ChemicalSpecies) : ℕ :=
  match reactant, product with
  | ChemicalSpecies.NaOH, ChemicalSpecies.H2O => 1
  | ChemicalSpecies.HCl, ChemicalSpecies.H2O => 1
  | _, _ => 0

-- Define the given amounts of reactants
def initial_NaOH : ℕ := 2
def initial_HCl : ℕ := 2

-- State the theorem
theorem water_formed_moles :
  min initial_NaOH initial_HCl = 
  stoichiometric_coefficient ChemicalSpecies.NaOH ChemicalSpecies.H2O * 
  stoichiometric_coefficient ChemicalSpecies.HCl ChemicalSpecies.H2O * 2 :=
by sorry

end NUMINAMATH_CALUDE_water_formed_moles_l1059_105977


namespace NUMINAMATH_CALUDE_part_one_part_two_l1059_105989

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem part_one : 
  ∀ x : ℝ, (p x 1 ∧ q x) → (2 < x ∧ x < 3) :=
sorry

-- Part 2
theorem part_two :
  (∀ x a : ℝ, q x → p x a) ∧ 
  (∃ x a : ℝ, p x a ∧ ¬(q x)) →
  (∀ a : ℝ, 1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1059_105989


namespace NUMINAMATH_CALUDE_max_consecutive_sum_l1059_105986

/-- The sum of n consecutive integers starting from a -/
def sum_consecutive (a : ℤ) (n : ℕ) : ℤ := n * (2 * a + n - 1) / 2

/-- The target sum -/
def target_sum : ℕ := 528

/-- The maximum number of consecutive integers summing to the target -/
def max_consecutive : ℕ := 1056

theorem max_consecutive_sum :
  (∃ a : ℤ, sum_consecutive a max_consecutive = target_sum) ∧
  (∀ n : ℕ, n > max_consecutive → ¬∃ a : ℤ, sum_consecutive a n = target_sum) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_l1059_105986


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l1059_105994

/-- Given information about blanket purchases and average price, 
    prove the unknown rate of two blankets. -/
theorem unknown_blanket_rate 
  (blanket_count_1 blanket_count_2 blanket_count_unknown : ℕ)
  (price_1 price_2 average_price : ℚ)
  (h1 : blanket_count_1 = 3)
  (h2 : blanket_count_2 = 3)
  (h3 : blanket_count_unknown = 2)
  (h4 : price_1 = 100)
  (h5 : price_2 = 150)
  (h6 : average_price = 150)
  (h7 : (blanket_count_1 * price_1 + blanket_count_2 * price_2 + 
         blanket_count_unknown * unknown_rate) / 
        (blanket_count_1 + blanket_count_2 + blanket_count_unknown) = 
        average_price) :
  unknown_rate = 225 := by
  sorry


end NUMINAMATH_CALUDE_unknown_blanket_rate_l1059_105994


namespace NUMINAMATH_CALUDE_auto_finance_fraction_l1059_105952

theorem auto_finance_fraction (total_credit auto_credit finance_credit : ℝ) 
  (h1 : total_credit = 291.6666666666667)
  (h2 : auto_credit = 0.36 * total_credit)
  (h3 : finance_credit = 35) :
  finance_credit / auto_credit = 1/3 := by
sorry

end NUMINAMATH_CALUDE_auto_finance_fraction_l1059_105952


namespace NUMINAMATH_CALUDE_painting_cost_in_cny_l1059_105907

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 5

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℝ := 8

/-- Cost of the painting in Namibian dollars -/
def painting_cost_nad : ℝ := 150

/-- Theorem stating the cost of the painting in Chinese yuan -/
theorem painting_cost_in_cny :
  (painting_cost_nad / usd_to_nad) * usd_to_cny = 240 := by
  sorry

end NUMINAMATH_CALUDE_painting_cost_in_cny_l1059_105907


namespace NUMINAMATH_CALUDE_range_of_H_l1059_105990

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

-- State the theorem about the range of H
theorem range_of_H :
  ∀ y : ℝ, (∃ x : ℝ, H x = y) ↔ y ∈ Set.Icc (-4) 4 := by sorry

end NUMINAMATH_CALUDE_range_of_H_l1059_105990


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l1059_105983

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1/2 : ℂ) + (1/3 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1/2 : ℂ) - (1/3 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l1059_105983


namespace NUMINAMATH_CALUDE_masha_room_number_l1059_105965

theorem masha_room_number 
  (total_rooms : ℕ) 
  (masha_room : ℕ) 
  (alina_room : ℕ) 
  (h1 : total_rooms = 10000)
  (h2 : 1 ≤ masha_room ∧ masha_room < alina_room ∧ alina_room ≤ total_rooms)
  (h3 : masha_room + alina_room = 2022)
  (h4 : (((alina_room - masha_room - 1) * (masha_room + alina_room)) / 2) = 3033) :
  masha_room = 1009 := by
sorry

end NUMINAMATH_CALUDE_masha_room_number_l1059_105965


namespace NUMINAMATH_CALUDE_modulus_of_z_l1059_105997

theorem modulus_of_z (a : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : ∃ (b : ℝ), (2 - i) / (a + i) = b * i) :
  Complex.abs (2 * a + Complex.I * Real.sqrt 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1059_105997


namespace NUMINAMATH_CALUDE_mens_average_weight_l1059_105932

theorem mens_average_weight (num_men : ℕ) (num_women : ℕ) (avg_women : ℝ) (avg_total : ℝ) :
  num_men = 8 →
  num_women = 6 →
  avg_women = 120 →
  avg_total = 160 →
  let total_people := num_men + num_women
  let avg_men := (avg_total * total_people - avg_women * num_women) / num_men
  avg_men = 190 := by
sorry

end NUMINAMATH_CALUDE_mens_average_weight_l1059_105932


namespace NUMINAMATH_CALUDE_max_value_expression_l1059_105935

theorem max_value_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (2 * x^2 + 2 * y^2 + 2) ≤ Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l1059_105935


namespace NUMINAMATH_CALUDE_reciprocal_sum_one_third_three_fourths_l1059_105909

theorem reciprocal_sum_one_third_three_fourths (x : ℚ) :
  x = (1/3 + 3/4)⁻¹ → x = 12/13 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_one_third_three_fourths_l1059_105909


namespace NUMINAMATH_CALUDE_milburg_children_count_l1059_105927

/-- The number of children in Milburg -/
def children_count (total_population : ℕ) (adult_count : ℕ) : ℕ :=
  total_population - adult_count

/-- Theorem: The number of children in Milburg is 2987 -/
theorem milburg_children_count :
  children_count 5256 2269 = 2987 := by
  sorry

end NUMINAMATH_CALUDE_milburg_children_count_l1059_105927


namespace NUMINAMATH_CALUDE_parabolas_similar_l1059_105910

/-- Two parabolas are similar if there exists a homothety that transforms one into the other -/
theorem parabolas_similar (a : ℝ) : 
  (∃ (x y : ℝ), y = 2 * x^2 → (∃ (x' y' : ℝ), y' = x'^2 ∧ x' = 2*x ∧ y' = 2*y)) := by
  sorry

#check parabolas_similar

end NUMINAMATH_CALUDE_parabolas_similar_l1059_105910


namespace NUMINAMATH_CALUDE_two_pants_three_tops_six_looks_l1059_105972

/-- The number of possible looks given a number of pants and tops -/
def number_of_looks (pants : ℕ) (tops : ℕ) : ℕ := pants * tops

/-- Theorem stating that 2 pairs of pants and 3 pairs of tops result in 6 looks -/
theorem two_pants_three_tops_six_looks : 
  number_of_looks 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_pants_three_tops_six_looks_l1059_105972


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l1059_105938

theorem minimum_value_theorem (m n : ℝ) (a : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x y : ℝ, y = a^(x+2) - 2 ∧ y = -n/m * x - 1/m) →
  1/m + 1/n ≥ 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l1059_105938


namespace NUMINAMATH_CALUDE_parallelogram_smaller_angle_l1059_105903

theorem parallelogram_smaller_angle (x : ℝ) : 
  x > 0 ∧ x < 180 ∧ x + (x + 90) = 180 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_smaller_angle_l1059_105903


namespace NUMINAMATH_CALUDE_system_no_solution_l1059_105985

/-- The coefficient matrix of the system of equations -/
def A (n : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![n, 1, 1;
     1, n, 1;
     1, 1, n]

/-- The theorem stating that the system has no solution iff n = -2 -/
theorem system_no_solution (n : ℝ) :
  (∀ x y z : ℝ, n * x + y + z ≠ 1 ∨ x + n * y + z ≠ 1 ∨ x + y + n * z ≠ 1) ↔ n = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_no_solution_l1059_105985


namespace NUMINAMATH_CALUDE_all_cells_happy_l1059_105929

def Board := Fin 10 → Fin 10 → Bool

def isBlue (board : Board) (i j : Fin 10) : Bool :=
  (i.val + j.val) % 2 = 0

def neighbors (i j : Fin 10) : List (Fin 10 × Fin 10) :=
  [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    |> List.filter (fun (x, y) => x.val < 10 && y.val < 10)

def countBlueNeighbors (board : Board) (i j : Fin 10) : Nat :=
  (neighbors i j).filter (fun (x, y) => isBlue board x y) |>.length

theorem all_cells_happy (board : Board) :
  ∀ i j : Fin 10, countBlueNeighbors board i j = 2 := by
  sorry

#check all_cells_happy

end NUMINAMATH_CALUDE_all_cells_happy_l1059_105929


namespace NUMINAMATH_CALUDE_line_parallel_plane_implies_parallel_to_all_lines_false_l1059_105949

/-- A line in 3D space --/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space --/
structure Plane3D where
  -- Define properties of a plane

/-- Defines when a line is parallel to a plane --/
def line_parallel_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when a line is contained in a plane --/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when two lines are parallel --/
def lines_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- The statement to be proven false --/
theorem line_parallel_plane_implies_parallel_to_all_lines_false :
  ¬ (∀ (l : Line3D) (p : Plane3D),
    line_parallel_plane l p →
    ∀ (l' : Line3D), line_in_plane l' p →
    lines_parallel l l') :=
  sorry

end NUMINAMATH_CALUDE_line_parallel_plane_implies_parallel_to_all_lines_false_l1059_105949


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1059_105946

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x + 3 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 3 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1059_105946


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l1059_105931

/-- Represents the number of handshakes in a soccer tournament -/
def tournament_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  n.choose 2 + k

/-- Theorem stating the minimum number of coach handshakes -/
theorem min_coach_handshakes :
  ∃ (n : ℕ) (k : ℕ), tournament_handshakes n k = 406 ∧ k = 0 ∧ 
  ∀ (m : ℕ) (j : ℕ), tournament_handshakes m j = 406 → j ≥ k :=
by sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l1059_105931


namespace NUMINAMATH_CALUDE_simple_interest_principal_l1059_105963

/-- Simple interest calculation -/
theorem simple_interest_principal (interest : ℚ) (rate : ℚ) (time : ℚ) :
  interest = 8625 →
  rate = 50 / 3 →
  time = 3 / 4 →
  ∃ principal : ℚ, principal = 69000 ∧ interest = principal * rate * time / 100 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l1059_105963


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l1059_105940

theorem sum_of_special_numbers (a b c : ℤ) : 
  (∀ n : ℕ, n ≥ a) → 
  (∀ m : ℤ, m < 0 → m ≤ b) → 
  (c = -c) → 
  a + b + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l1059_105940


namespace NUMINAMATH_CALUDE_probability_half_correct_l1059_105925

/-- The probability of getting exactly k successes in n trials with probability p for each trial. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of questions in the test -/
def num_questions : ℕ := 20

/-- The number of choices for each question -/
def num_choices : ℕ := 3

/-- The probability of guessing a question correctly -/
def prob_correct : ℚ := 1 / num_choices

/-- The number of questions to get correct -/
def target_correct : ℕ := num_questions / 2

theorem probability_half_correct :
  binomial_probability num_questions target_correct prob_correct = 189399040 / 3486784401 := by
  sorry

end NUMINAMATH_CALUDE_probability_half_correct_l1059_105925
