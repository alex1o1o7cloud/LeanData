import Mathlib

namespace NUMINAMATH_CALUDE_line_perpendicular_range_l3107_310770

/-- Given a line l: x - y + a = 0 and points A(-2,0) and B(2,0),
    if there exists a point P on line l such that AP ⊥ BP,
    then -2√2 ≤ a ≤ 2√2. -/
theorem line_perpendicular_range (a : ℝ) :
  (∃ (P : ℝ × ℝ), 
    (P.1 - P.2 + a = 0) ∧ 
    ((P.1 + 2) * (P.1 - 2) + P.2 * P.2 = 0)) →
  -2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_line_perpendicular_range_l3107_310770


namespace NUMINAMATH_CALUDE_sum_of_squares_l3107_310748

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 22) (h2 : x * y = 40) : x^2 + y^2 = 404 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3107_310748


namespace NUMINAMATH_CALUDE_not_sum_of_consecutive_iff_power_of_two_l3107_310789

/-- A natural number is a power of 2 -/
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

/-- A natural number can be expressed as the sum of consecutive natural numbers -/
def is_sum_of_consecutive (n : ℕ) : Prop :=
  ∃ (start : ℕ) (length : ℕ+), n = (length : ℕ) * (2 * start + length - 1) / 2

/-- 
Theorem: A natural number cannot be expressed as the sum of consecutive natural numbers 
if and only if it is a power of 2
-/
theorem not_sum_of_consecutive_iff_power_of_two (n : ℕ) :
  ¬(is_sum_of_consecutive n) ↔ is_power_of_two n := by sorry

end NUMINAMATH_CALUDE_not_sum_of_consecutive_iff_power_of_two_l3107_310789


namespace NUMINAMATH_CALUDE_cinema_tickets_l3107_310760

theorem cinema_tickets (x y : ℕ) : 
  x + y = 35 →
  24 * x + 18 * y = 750 →
  x = 20 ∧ y = 15 := by
sorry

end NUMINAMATH_CALUDE_cinema_tickets_l3107_310760


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_equation_l3107_310704

/-- Given a parabola and a hyperbola in the Cartesian coordinate plane, 
    with a point of intersection and a condition on the focus of the parabola, 
    prove that the asymptotes of the hyperbola have a specific equation. -/
theorem hyperbola_asymptotes_equation (b : ℝ) (A : ℝ × ℝ) (F : ℝ × ℝ) :
  b > 0 →
  A.2^2 = 4 * A.1 →
  A.1^2 / 4 - A.2^2 / b^2 = 1 →
  F = (1, 0) →
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = 25 →
  ∃ (k : ℝ), k = 2 * Real.sqrt 3 / 3 ∧
    (∀ (x y : ℝ), (x^2 / 4 - y^2 / b^2 = 1) → (y = k * x ∨ y = -k * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_equation_l3107_310704


namespace NUMINAMATH_CALUDE_jet_flight_time_l3107_310797

/-- Given a jet flying with and against wind, calculate the time taken with tail wind -/
theorem jet_flight_time (distance : ℝ) (return_time : ℝ) (wind_speed : ℝ) 
  (h1 : distance = 2000)
  (h2 : return_time = 5)
  (h3 : wind_speed = 50) : 
  ∃ (jet_speed : ℝ) (tail_wind_time : ℝ),
    (jet_speed + wind_speed) * tail_wind_time = distance ∧
    (jet_speed - wind_speed) * return_time = distance ∧
    tail_wind_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_jet_flight_time_l3107_310797


namespace NUMINAMATH_CALUDE_complex_parts_of_z_l3107_310784

theorem complex_parts_of_z : ∃ z : ℂ, z = Complex.I ^ 2 + Complex.I ∧ z.re = -1 ∧ z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_parts_of_z_l3107_310784


namespace NUMINAMATH_CALUDE_expansion_distinct_terms_l3107_310776

/-- The number of distinct terms in the expansion of a product of two sums -/
def distinctTermsInExpansion (n m : ℕ) : ℕ := n * m

/-- Theorem: The number of distinct terms in the expansion of (a+b+c)(d+e+f+g+h+i) is 18 -/
theorem expansion_distinct_terms :
  distinctTermsInExpansion 3 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_expansion_distinct_terms_l3107_310776


namespace NUMINAMATH_CALUDE_skittles_bought_proof_l3107_310739

/-- The number of Skittles Brenda initially had -/
def initial_skittles : ℕ := 7

/-- The number of Skittles Brenda ended up with -/
def final_skittles : ℕ := 15

/-- The number of Skittles Brenda bought -/
def bought_skittles : ℕ := final_skittles - initial_skittles

theorem skittles_bought_proof :
  bought_skittles = final_skittles - initial_skittles :=
by sorry

end NUMINAMATH_CALUDE_skittles_bought_proof_l3107_310739


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l3107_310779

def repeating_decimal (c d : ℕ) : ℚ :=
  1 + (c * 10 + d : ℚ) / 99

theorem mistaken_multiplication (c d : ℕ) 
  (h1 : c < 10) (h2 : d < 10) :
  90 * repeating_decimal c d - 90 * (1 + (c * 10 + d : ℚ) / 100) = 0.9 → 
  c = 9 ∧ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l3107_310779


namespace NUMINAMATH_CALUDE_stacy_berries_l3107_310735

theorem stacy_berries (skylar_berries steve_berries stacy_berries : ℕ) : 
  skylar_berries = 20 →
  steve_berries = skylar_berries / 2 →
  stacy_berries = 3 * steve_berries + 2 →
  stacy_berries = 32 := by
  sorry

end NUMINAMATH_CALUDE_stacy_berries_l3107_310735


namespace NUMINAMATH_CALUDE_cone_from_sector_l3107_310762

/-- Proves that a cone formed from a 300° sector of a circle with radius 8 
    has a base radius of 20/3 and a slant height of 8 -/
theorem cone_from_sector (sector_angle : Real) (circle_radius : Real) 
    (cone_base_radius : Real) (cone_slant_height : Real) : 
    sector_angle = 300 ∧ 
    circle_radius = 8 ∧ 
    cone_base_radius = 20 / 3 ∧ 
    cone_slant_height = circle_radius → 
    cone_base_radius * 2 * π = sector_angle / 360 * (2 * π * circle_radius) ∧
    cone_slant_height = circle_radius := by
  sorry

end NUMINAMATH_CALUDE_cone_from_sector_l3107_310762


namespace NUMINAMATH_CALUDE_factorial_sum_unique_solution_l3107_310730

theorem factorial_sum_unique_solution :
  ∀ n a b c : ℕ, n.factorial = a.factorial + b.factorial + c.factorial →
  n = 3 ∧ a = 2 ∧ b = 2 ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_unique_solution_l3107_310730


namespace NUMINAMATH_CALUDE_point_in_region_l3107_310796

def point : ℝ × ℝ := (0, -2)

theorem point_in_region (x y : ℝ) (h : (x, y) = point) : 
  x + y - 1 < 0 ∧ x - y + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_l3107_310796


namespace NUMINAMATH_CALUDE_line_parameterization_l3107_310716

/-- The line equation y = (3/4)x - 2 parameterized as (x, y) = (-3, v) + u(m, -8) -/
def line_equation (x y : ℝ) : Prop :=
  y = (3/4) * x - 2

/-- The parametric form of the line -/
def parametric_form (x y u v m : ℝ) : Prop :=
  x = -3 + u * m ∧ y = v - 8 * u

/-- Theorem stating that v = -17/4 and m = -16/9 satisfy the line equation and parametric form -/
theorem line_parameterization :
  ∃ (v m : ℝ), v = -17/4 ∧ m = -16/9 ∧
  (∀ (x y u : ℝ), parametric_form x y u v m → line_equation x y) :=
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3107_310716


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l3107_310777

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), 
    (7 * a) % 77 = 1 ∧ 
    (13 * b) % 77 = 1 ∧ 
    ((3 * a + 9 * b) % 77) = 10 :=
by sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l3107_310777


namespace NUMINAMATH_CALUDE_triple_sharp_40_l3107_310753

-- Define the # function
def sharp (N : ℝ) : ℝ := 0.7 * N + 2

-- State the theorem
theorem triple_sharp_40 : sharp (sharp (sharp 40)) = 18.1 := by
  sorry

end NUMINAMATH_CALUDE_triple_sharp_40_l3107_310753


namespace NUMINAMATH_CALUDE_binary_11100_to_quaternary_l3107_310799

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its quaternary representation (as a list of digits) -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary number 11100₂ -/
def binary_11100 : List Bool := [true, true, true, false, false]

theorem binary_11100_to_quaternary :
  decimal_to_quaternary (binary_to_decimal binary_11100) = [1, 3, 0] :=
sorry

end NUMINAMATH_CALUDE_binary_11100_to_quaternary_l3107_310799


namespace NUMINAMATH_CALUDE_vector_addition_l3107_310744

/-- Given two vectors OA and AB in 2D space, prove that OB is their sum. -/
theorem vector_addition (OA AB : ℝ × ℝ) (h1 : OA = (-2, 3)) (h2 : AB = (-1, -4)) :
  OA + AB = (-3, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l3107_310744


namespace NUMINAMATH_CALUDE_star_properties_l3107_310714

def star (x y : ℤ) : ℤ := (x + 2) * (y + 2) - 3

theorem star_properties :
  (∀ x y : ℤ, star x y = star y x) ∧
  (∃ x y z : ℤ, star x (y + z) ≠ star x y + star x z) ∧
  (∃ x : ℤ, star (x - 2) (x + 2) ≠ star x x - 3) ∧
  (∃ x : ℤ, star x 1 ≠ x) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l3107_310714


namespace NUMINAMATH_CALUDE_paths_to_2005_l3107_310793

/-- Represents a position on the 5x5 grid --/
inductive Position
| Center : Position
| Side : Position
| Corner : Position
| Edge : Position

/-- Represents the possible moves on the grid --/
def possibleMoves : Position → List Position
| Position.Center => [Position.Side, Position.Corner]
| Position.Side => [Position.Side, Position.Corner, Position.Edge]
| Position.Corner => [Position.Side, Position.Edge]
| Position.Edge => []

/-- Counts the number of paths to form 2005 on the given grid --/
def countPaths : ℕ :=
  let initialSideMoves := 4
  let initialCornerMoves := 4
  let sideToSideMoves := 2
  let sideToCornerMoves := 2
  let cornerToSideMoves := 2
  let sideToEdgeMoves := 3
  let cornerToEdgeMoves := 5

  let sideSidePaths := initialSideMoves * sideToSideMoves * sideToEdgeMoves
  let sideCornerPaths := initialSideMoves * sideToCornerMoves * cornerToEdgeMoves
  let cornerSidePaths := initialCornerMoves * cornerToSideMoves * sideToEdgeMoves

  sideSidePaths + sideCornerPaths + cornerSidePaths

/-- Theorem stating that there are 88 paths to form 2005 on the given grid --/
theorem paths_to_2005 : countPaths = 88 := by sorry

end NUMINAMATH_CALUDE_paths_to_2005_l3107_310793


namespace NUMINAMATH_CALUDE_distribute_objects_eq_144_l3107_310772

-- Define the number of objects and containers
def n : ℕ := 4

-- Define the function to calculate the number of ways to distribute objects
def distribute_objects : ℕ := sorry

-- Theorem statement
theorem distribute_objects_eq_144 : distribute_objects = 144 := by sorry

end NUMINAMATH_CALUDE_distribute_objects_eq_144_l3107_310772


namespace NUMINAMATH_CALUDE_least_possible_third_side_l3107_310747

theorem least_possible_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a = 8 ∧ b = 15) ∨ (a = 8 ∧ c = 15) ∨ (b = 8 ∧ c = 15) →
  a^2 + b^2 = c^2 →
  Real.sqrt 161 ≤ min a (min b c) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_third_side_l3107_310747


namespace NUMINAMATH_CALUDE_brad_balloons_l3107_310746

theorem brad_balloons (total red blue : ℕ) (h1 : total = 50) (h2 : red = 12) (h3 : blue = 7) :
  total - (red + blue) = 31 := by
  sorry

end NUMINAMATH_CALUDE_brad_balloons_l3107_310746


namespace NUMINAMATH_CALUDE_correct_boat_equation_l3107_310718

/-- Represents the scenario of boats and students during the Qingming Festival outing. -/
structure BoatScenario where
  totalBoats : ℕ
  largeboatCapacity : ℕ
  smallboatCapacity : ℕ
  totalStudents : ℕ

/-- The equation representing the boat scenario. -/
def boatEquation (scenario : BoatScenario) (x : ℕ) : Prop :=
  scenario.largeboatCapacity * (scenario.totalBoats - x) + scenario.smallboatCapacity * x = scenario.totalStudents

/-- Theorem stating that the given equation correctly represents the boat scenario. -/
theorem correct_boat_equation (scenario : BoatScenario) (h1 : scenario.totalBoats = 8) 
    (h2 : scenario.largeboatCapacity = 6) (h3 : scenario.smallboatCapacity = 4) 
    (h4 : scenario.totalStudents = 38) : 
  boatEquation scenario = fun x => 6 * (8 - x) + 4 * x = 38 := by
  sorry


end NUMINAMATH_CALUDE_correct_boat_equation_l3107_310718


namespace NUMINAMATH_CALUDE_rectangle_to_square_l3107_310706

theorem rectangle_to_square (area : ℝ) (reduction : ℝ) (side : ℝ) : 
  area = 600 →
  reduction = 10 →
  (side + reduction) * side = area →
  side * side = area →
  side = 20 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l3107_310706


namespace NUMINAMATH_CALUDE_f_negative_three_value_l3107_310720

/-- Given a function f(x) = a*sin(x) + b*tan(x) + x^3 + 1, 
    if f(3) = 7, then f(-3) = -5 -/
theorem f_negative_three_value 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = a * Real.sin x + b * Real.tan x + x^3 + 1) 
  (h2 : f 3 = 7) : 
  f (-3) = -5 := by sorry

end NUMINAMATH_CALUDE_f_negative_three_value_l3107_310720


namespace NUMINAMATH_CALUDE_solution_equation_l3107_310743

theorem solution_equation (x : ℝ) (hx : x ≠ 0) :
  (9 * x)^10 = (18 * x)^5 → x = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_solution_equation_l3107_310743


namespace NUMINAMATH_CALUDE_inequality_proof_l3107_310780

def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2*|a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3107_310780


namespace NUMINAMATH_CALUDE_opposite_abs_power_l3107_310736

theorem opposite_abs_power (x y : ℝ) : 
  |x - 2| + |y + 3| = 0 → (x + y)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_abs_power_l3107_310736


namespace NUMINAMATH_CALUDE_not_linear_in_M_f_expression_for_negative_range_sin_k_in_M_iff_l3107_310786

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∃ (T : ℝ) (hT : T ≠ 0), ∀ x, f (x + T) = T * f x}

-- Theorem 1
theorem not_linear_in_M : ¬(λ x : ℝ => x) ∈ M := by sorry

-- Theorem 2
theorem f_expression_for_negative_range 
  (f : ℝ → ℝ) (hf : f ∈ M) (hT : ∃ T, T = 2 ∧ ∀ x, 1 < x → x < 2 → f x = x + Real.log x) :
  ∀ x, -3 < x → x < -2 → f x = (1/4) * (x + 4 + Real.log (x + 4)) := by sorry

-- Theorem 3
theorem sin_k_in_M_iff (k : ℝ) : 
  (λ x : ℝ => Real.sin (k * x)) ∈ M ↔ ∃ m : ℤ, k = m * Real.pi := by sorry

end NUMINAMATH_CALUDE_not_linear_in_M_f_expression_for_negative_range_sin_k_in_M_iff_l3107_310786


namespace NUMINAMATH_CALUDE_min_value_expression_l3107_310798

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b * c = 1) (h2 : a / b = 2) :
  a^2 + 4*a*b + 9*b^2 + 8*b*c + 3*c^2 ≥ 3 * (63 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3107_310798


namespace NUMINAMATH_CALUDE_burt_basil_profit_l3107_310773

/-- Calculate the net profit from Burt's basil plants -/
theorem burt_basil_profit :
  let seed_cost : ℕ := 200  -- in cents
  let soil_cost : ℕ := 800  -- in cents
  let total_plants : ℕ := 20
  let price_per_plant : ℕ := 500  -- in cents
  
  let total_cost : ℕ := seed_cost + soil_cost
  let total_revenue : ℕ := total_plants * price_per_plant
  let net_profit : ℤ := total_revenue - total_cost
  
  net_profit = 9000  -- 90.00 in cents
  := by sorry

end NUMINAMATH_CALUDE_burt_basil_profit_l3107_310773


namespace NUMINAMATH_CALUDE_base_conversion_2869_to_base_7_l3107_310775

theorem base_conversion_2869_to_base_7 :
  2869 = 1 * (7^4) + 1 * (7^3) + 2 * (7^2) + 3 * (7^1) + 6 * (7^0) :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_2869_to_base_7_l3107_310775


namespace NUMINAMATH_CALUDE_parallelogram_area_l3107_310750

/-- The area of a parallelogram with diagonals intersecting at a 60° angle
    and two sides of lengths 6 and 8 is equal to 14√3. -/
theorem parallelogram_area (a b : ℝ) : 
  (a^2 + b^2 - a*b = 36) →  -- From the side of length 6
  (a^2 + b^2 + a*b = 64) →  -- From the side of length 8
  2 * a * b * (Real.sqrt 3 / 2) = 14 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3107_310750


namespace NUMINAMATH_CALUDE_pascal_ratio_98_l3107_310737

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Pascal's Triangle property: each entry is the sum of the two entries directly above it -/
axiom pascal_property (n k : ℕ) : binomial (n + 1) k = binomial n (k - 1) + binomial n k

/-- Three consecutive entries in Pascal's Triangle -/
def consecutive_entries (n r : ℕ) : (ℕ × ℕ × ℕ) :=
  (binomial n r, binomial n (r + 1), binomial n (r + 2))

/-- Ratio of three numbers -/
def in_ratio (a b c : ℕ) (x y z : ℕ) : Prop :=
  a * y = b * x ∧ b * z = c * y

theorem pascal_ratio_98 : ∃ r : ℕ, in_ratio (binomial 98 r) (binomial 98 (r + 1)) (binomial 98 (r + 2)) 4 5 6 := by
  sorry

end NUMINAMATH_CALUDE_pascal_ratio_98_l3107_310737


namespace NUMINAMATH_CALUDE_max_common_segment_length_theorem_l3107_310719

/-- The maximum length of the common initial segment of two sequences with coprime periods -/
def max_common_segment_length (m n : ℕ) : ℕ :=
  m + n - 2

/-- Theorem stating that for two sequences with coprime periods m and n,
    the maximum length of their common initial segment is m + n - 2 -/
theorem max_common_segment_length_theorem (m n : ℕ) (h : Nat.Coprime m n) :
  max_common_segment_length m n = m + n - 2 := by
  sorry


end NUMINAMATH_CALUDE_max_common_segment_length_theorem_l3107_310719


namespace NUMINAMATH_CALUDE_divisibility_by_p_squared_l3107_310738

theorem divisibility_by_p_squared (p : ℕ) (h_prime : Nat.Prime p) (h_greater_than_three : p > 3) :
  ∃ k : ℤ, (p + 1 : ℤ)^(p - 1) - 1 = k * p^2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_p_squared_l3107_310738


namespace NUMINAMATH_CALUDE_colored_dodecahedron_constructions_l3107_310728

/-- The number of faces in a dodecahedron -/
def num_faces : ℕ := 12

/-- The number of rotational symmetries considered for simplification -/
def rotational_symmetries : ℕ := 5

/-- The number of distinguishable ways to construct a colored dodecahedron -/
def distinguishable_constructions : ℕ := Nat.factorial (num_faces - 1) / rotational_symmetries

/-- Theorem stating the number of distinguishable ways to construct a colored dodecahedron -/
theorem colored_dodecahedron_constructions :
  distinguishable_constructions = 7983360 := by
  sorry

#eval distinguishable_constructions

end NUMINAMATH_CALUDE_colored_dodecahedron_constructions_l3107_310728


namespace NUMINAMATH_CALUDE_handshake_problem_l3107_310742

theorem handshake_problem (n : ℕ) (total_handshakes : ℕ) 
  (h1 : n = 12) 
  (h2 : total_handshakes = 66) : 
  total_handshakes = n * (n - 1) / 2 ∧ 
  (total_handshakes / (n - 1) : ℚ) = 6 := by
sorry

end NUMINAMATH_CALUDE_handshake_problem_l3107_310742


namespace NUMINAMATH_CALUDE_smallest_z_for_inequality_l3107_310756

theorem smallest_z_for_inequality : ∃ (z : ℕ), (∀ (y : ℕ), 27 ^ y > 3 ^ 24 → z ≤ y) ∧ 27 ^ z > 3 ^ 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_z_for_inequality_l3107_310756


namespace NUMINAMATH_CALUDE_additional_cars_needed_l3107_310740

def current_cars : ℕ := 37
def cars_per_row : ℕ := 9

theorem additional_cars_needed :
  let next_multiple := ((current_cars + cars_per_row - 1) / cars_per_row) * cars_per_row
  next_multiple - current_cars = 8 := by
  sorry

end NUMINAMATH_CALUDE_additional_cars_needed_l3107_310740


namespace NUMINAMATH_CALUDE_remainder_problem_l3107_310729

theorem remainder_problem (m : ℤ) (h : m % 5 = 3) : (4 * m + 5) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3107_310729


namespace NUMINAMATH_CALUDE_pancake_diameter_l3107_310715

/-- The diameter of a circular object with radius 7 centimeters is 14 centimeters. -/
theorem pancake_diameter (r : ℝ) (h : r = 7) : 2 * r = 14 := by
  sorry

end NUMINAMATH_CALUDE_pancake_diameter_l3107_310715


namespace NUMINAMATH_CALUDE_mars_radius_scientific_notation_l3107_310788

theorem mars_radius_scientific_notation :
  3395000 = 3.395 * (10 ^ 6) := by sorry

end NUMINAMATH_CALUDE_mars_radius_scientific_notation_l3107_310788


namespace NUMINAMATH_CALUDE_sum_inequality_l3107_310790

theorem sum_inequality (m n : ℕ+) (a : Fin m → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_range : ∀ i, a i ∈ Finset.range n)
  (h_sum : ∀ i j, i ≤ j → a i + a j ≤ n → ∃ k, a i + a j = a k) :
  (Finset.sum (Finset.range m) (λ i => a i)) / m ≥ (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3107_310790


namespace NUMINAMATH_CALUDE_certain_amount_proof_l3107_310764

theorem certain_amount_proof (x : ℝ) (A : ℝ) : 
  x = 780 → 
  (0.25 * x) = (0.15 * 1500 - A) → 
  A = 30 := by
sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l3107_310764


namespace NUMINAMATH_CALUDE_shopping_mall_problem_l3107_310717

/-- Shopping mall goods purchasing and selling problem -/
theorem shopping_mall_problem 
  (cost_A_1_B_2 : ℝ) 
  (cost_A_3_B_2 : ℝ) 
  (sell_price_A : ℝ) 
  (sell_price_B : ℝ) 
  (total_units : ℕ) 
  (profit_lower : ℝ) 
  (profit_upper : ℝ) 
  (planned_units : ℕ) 
  (actual_profit : ℝ)
  (h1 : cost_A_1_B_2 = 320)
  (h2 : cost_A_3_B_2 = 520)
  (h3 : sell_price_A = 120)
  (h4 : sell_price_B = 140)
  (h5 : total_units = 50)
  (h6 : profit_lower = 1350)
  (h7 : profit_upper = 1375)
  (h8 : planned_units = 46)
  (h9 : actual_profit = 1220) :
  ∃ (cost_A cost_B : ℝ) (m : ℕ) (b : ℕ),
    cost_A = 100 ∧ 
    cost_B = 110 ∧ 
    13 ≤ m ∧ m ≤ 15 ∧
    b ≥ 32 := by sorry

end NUMINAMATH_CALUDE_shopping_mall_problem_l3107_310717


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3107_310723

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry with respect to x-axis
def symmetricXAxis (p q : Point2D) : Prop :=
  p.x = q.x ∧ p.y = -q.y

-- Theorem statement
theorem symmetric_point_coordinates :
  ∀ (B A : Point2D),
    B.x = 4 ∧ B.y = -1 →
    symmetricXAxis A B →
    A.x = 4 ∧ A.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3107_310723


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l3107_310741

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop :=
  ∃ a b : ℤ, b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop :=
  ¬(IsRational x)

-- Theorem stating that √2 is irrational
theorem sqrt_two_irrational : IsIrrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l3107_310741


namespace NUMINAMATH_CALUDE_cone_volume_l3107_310710

/-- Given a cone with slant height 1 and lateral surface area 2π/3, its volume is 4√5π/81 -/
theorem cone_volume (s : Real) (A : Real) (V : Real) : 
  s = 1 → A = (2/3) * Real.pi → V = (4 * Real.sqrt 5 / 81) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l3107_310710


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_two_l3107_310751

theorem log_expression_equals_negative_two :
  (Real.log 64 / Real.log 32) / (Real.log 2 / Real.log 32) -
  (Real.log 256 / Real.log 16) / (Real.log 2 / Real.log 16) = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_two_l3107_310751


namespace NUMINAMATH_CALUDE_all_terms_even_l3107_310791

theorem all_terms_even (p q : ℤ) (hp : Even p) (hq : Even q) :
  ∀ k : ℕ, k ≤ 8 → Even (Nat.choose 8 k * p^(8 - k) * q^k) := by sorry

end NUMINAMATH_CALUDE_all_terms_even_l3107_310791


namespace NUMINAMATH_CALUDE_inequality_holds_l3107_310795

open Real

/-- A function satisfying the given differential equation -/
def SatisfiesDiffEq (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, x * (deriv^[2] f x) + 2 * f x = 1 / x^2

theorem inequality_holds (f : ℝ → ℝ) (hf : SatisfiesDiffEq f) :
  f 2 / 9 < f 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3107_310795


namespace NUMINAMATH_CALUDE_intersection_equals_open_closed_interval_l3107_310711

-- Define set A
def A : Set ℝ := {x | x^2 - 1 ≤ 0}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_open_closed_interval :
  A_intersect_B = Set.Ioo 0 1 ∪ Set.Ioc 1 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_equals_open_closed_interval_l3107_310711


namespace NUMINAMATH_CALUDE_area1_is_linear_area2_is_quadratic_l3107_310734

-- Define the rectangles
def rectangle1_length : ℝ := 10
def rectangle1_width : ℝ := 5
def rectangle2_length : ℝ := 30
def rectangle2_width : ℝ := 20

-- Define the area functions
def area1 (x : ℝ) : ℝ := (rectangle1_length - x) * rectangle1_width
def area2 (x : ℝ) : ℝ := (rectangle2_length + x) * (rectangle2_width + x)

-- Theorem statements
theorem area1_is_linear :
  ∃ (m b : ℝ), ∀ x, area1 x = m * x + b :=
sorry

theorem area2_is_quadratic :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, area2 x = a * x^2 + b * x + c) :=
sorry

end NUMINAMATH_CALUDE_area1_is_linear_area2_is_quadratic_l3107_310734


namespace NUMINAMATH_CALUDE_max_x_minus_y_l3107_310732

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), w = x - y → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l3107_310732


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3107_310701

theorem modulus_of_complex_number :
  let z : ℂ := (1 + 2*Complex.I) / Complex.I^2
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3107_310701


namespace NUMINAMATH_CALUDE_parabola_shift_left_one_unit_l3107_310713

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := 2 * p.a * h + p.b,
    c := p.a * h^2 + p.b * h + p.c }

theorem parabola_shift_left_one_unit :
  let original := Parabola.mk 1 0 2
  let shifted := shift_horizontal original (-1)
  shifted = Parabola.mk 1 2 3 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_left_one_unit_l3107_310713


namespace NUMINAMATH_CALUDE_more_men_than_women_count_l3107_310722

def num_men : ℕ := 6
def num_women : ℕ := 4
def group_size : ℕ := 5

def select_group (m w : ℕ) : ℕ := Nat.choose num_men m * Nat.choose num_women w

theorem more_men_than_women_count : 
  (select_group 3 2) + (select_group 4 1) + (select_group 5 0) = 186 :=
by sorry

end NUMINAMATH_CALUDE_more_men_than_women_count_l3107_310722


namespace NUMINAMATH_CALUDE_is_quadratic_equation_example_l3107_310745

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation 2x^2 = 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 1

/-- Theorem: The equation 2x^2 = 1 is a quadratic equation in one variable -/
theorem is_quadratic_equation_example : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_is_quadratic_equation_example_l3107_310745


namespace NUMINAMATH_CALUDE_max_correct_answers_is_19_l3107_310761

/-- Represents the result of an exam -/
structure ExamResult where
  total_questions : Nat
  correct_answers : Nat
  wrong_answers : Nat
  unanswered : Nat
  score : Int

/-- Checks if an ExamResult is valid according to the given scoring system -/
def is_valid_result (result : ExamResult) : Prop :=
  result.total_questions = 25 ∧
  result.correct_answers + result.wrong_answers + result.unanswered = result.total_questions ∧
  4 * result.correct_answers - result.wrong_answers = result.score

/-- Theorem: The maximum number of correct answers for a score of 70 is 19 -/
theorem max_correct_answers_is_19 :
  ∀ result : ExamResult,
    is_valid_result result →
    result.score = 70 →
    result.correct_answers ≤ 19 ∧
    ∃ optimal_result : ExamResult,
      is_valid_result optimal_result ∧
      optimal_result.score = 70 ∧
      optimal_result.correct_answers = 19 :=
by sorry

#check max_correct_answers_is_19

end NUMINAMATH_CALUDE_max_correct_answers_is_19_l3107_310761


namespace NUMINAMATH_CALUDE_intersection_point_polar_radius_l3107_310727

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 - y^2 = 4 ∧ y ≠ 0

-- Define the line l₃ in polar form
def l₃ (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) - Real.sqrt 2 = 0

-- Define the intersection point M
def M (x y : ℝ) : Prop := C x y ∧ x + y = Real.sqrt 2

-- Theorem statement
theorem intersection_point_polar_radius :
  ∀ x y : ℝ, M x y → x^2 + y^2 = 5 := by sorry

end NUMINAMATH_CALUDE_intersection_point_polar_radius_l3107_310727


namespace NUMINAMATH_CALUDE_area_ratio_of_concentric_circles_l3107_310767

/-- Two concentric circles with center O -/
structure ConcentricCircles where
  O : Point
  r₁ : ℝ  -- radius of smaller circle
  r₂ : ℝ  -- radius of larger circle
  h : 0 < r₁ ∧ r₁ < r₂

/-- The length of an arc on a circle -/
def arcLength (r : ℝ) (θ : ℝ) : ℝ := r * θ

theorem area_ratio_of_concentric_circles (C : ConcentricCircles) :
  arcLength C.r₁ (π/3) = arcLength C.r₂ (π/4) →
  (C.r₁^2 / C.r₂^2 : ℝ) = 9/16 := by
  sorry

#check area_ratio_of_concentric_circles

end NUMINAMATH_CALUDE_area_ratio_of_concentric_circles_l3107_310767


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_is_56_l3107_310787

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSide := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSide) * (box.width / cubeSide) * (box.depth / cubeSide)

/-- Theorem stating that the smallest number of cubes to fill the given box is 56 -/
theorem smallest_number_of_cubes_is_56 :
  smallestNumberOfCubes ⟨35, 20, 10⟩ = 56 := by
  sorry

#eval smallestNumberOfCubes ⟨35, 20, 10⟩

end NUMINAMATH_CALUDE_smallest_number_of_cubes_is_56_l3107_310787


namespace NUMINAMATH_CALUDE_math_test_blank_questions_l3107_310700

theorem math_test_blank_questions 
  (total_questions : ℕ) 
  (word_problems : ℕ) 
  (addition_subtraction_problems : ℕ)
  (questions_answered : ℕ) 
  (h1 : total_questions = 45)
  (h2 : word_problems = 17)
  (h3 : addition_subtraction_problems = 28)
  (h4 : questions_answered = 38)
  (h5 : word_problems + addition_subtraction_problems = total_questions) :
  total_questions - questions_answered = 7 := by
  sorry

end NUMINAMATH_CALUDE_math_test_blank_questions_l3107_310700


namespace NUMINAMATH_CALUDE_inequality_proof_l3107_310765

theorem inequality_proof (x : ℝ) (h1 : (3/2 : ℝ) ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3107_310765


namespace NUMINAMATH_CALUDE_triangle_area_main_theorem_l3107_310733

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the area of a specific triangle -/
theorem triangle_area (t : Triangle) 
  (h1 : t.a^2 = t.b^2 + t.c^2 - t.b * t.c) 
  (h2 : t.b * t.c = 16) : 
  (1/2) * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 := by
  sorry

/-- Main theorem proving the area of the triangle -/
theorem main_theorem : 
  ∃ (t : Triangle), 
    t.a^2 = t.b^2 + t.c^2 - t.b * t.c ∧ 
    t.b * t.c = 16 ∧ 
    (1/2) * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_main_theorem_l3107_310733


namespace NUMINAMATH_CALUDE_vector_subtraction_l3107_310768

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (-7, 0, 1)
def b : ℝ × ℝ × ℝ := (6, 2, -1)

-- State the theorem
theorem vector_subtraction :
  (a.1 - 5 * b.1, a.2.1 - 5 * b.2.1, a.2.2 - 5 * b.2.2) = (-37, -10, 6) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3107_310768


namespace NUMINAMATH_CALUDE_grid_problem_l3107_310771

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- Represents the grid of numbers -/
structure NumberGrid where
  row : ArithmeticSequence
  col1 : ArithmeticSequence
  col2 : ArithmeticSequence

/-- The main theorem -/
theorem grid_problem (g : NumberGrid) :
  g.row.first = 16 ∧
  g.col1.first + g.col1.diff = 10 ∧
  g.col1.first + 2 * g.col1.diff = 19 ∧
  g.col2.first + 4 * g.col2.diff = -13 ∧
  g.row.first + 6 * g.row.diff = g.col2.first + 4 * g.col2.diff →
  g.col2.first = -36.75 := by
  sorry

end NUMINAMATH_CALUDE_grid_problem_l3107_310771


namespace NUMINAMATH_CALUDE_fir_tree_needles_l3107_310769

/-- Represents the number of fir trees in the forest -/
def num_trees : ℕ := 710000

/-- Represents the maximum number of needles a tree can have -/
def max_needles : ℕ := 100000

/-- Represents the minimum number of trees we want to prove have the same number of needles -/
def min_same_needles : ℕ := 7

theorem fir_tree_needles :
  ∃ (n : ℕ) (trees : Finset (Fin num_trees)),
    n ≤ max_needles ∧
    trees.card ≥ min_same_needles ∧
    ∀ t ∈ trees, (fun i => i.val) t = n :=
by sorry

end NUMINAMATH_CALUDE_fir_tree_needles_l3107_310769


namespace NUMINAMATH_CALUDE_obtuse_triangle_count_and_largest_perimeter_l3107_310707

/-- Represents a triangle with side lengths in arithmetic progression --/
structure ArithmeticTriangle where
  a : ℕ  -- middle length
  d : ℕ  -- common difference

/-- Checks if the triangle is obtuse --/
def ArithmeticTriangle.isObtuse (t : ArithmeticTriangle) : Prop :=
  (t.a - t.d)^2 + t.a^2 < (t.a + t.d)^2

/-- Checks if the triangle satisfies the given conditions --/
def ArithmeticTriangle.isValid (t : ArithmeticTriangle) : Prop :=
  t.d > 0 ∧ t.a > t.d ∧ t.a + t.d ≤ 50

/-- Counts the number of valid obtuse triangles --/
def countValidObtuseTriangles : ℕ := sorry

/-- Finds the triangle with the largest perimeter --/
def largestPerimeterTriangle : ArithmeticTriangle := sorry

theorem obtuse_triangle_count_and_largest_perimeter :
  countValidObtuseTriangles = 157 ∧
  let t := largestPerimeterTriangle
  t.a - t.d = 29 ∧ t.a = 39 ∧ t.a + t.d = 50 := by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_count_and_largest_perimeter_l3107_310707


namespace NUMINAMATH_CALUDE_pancake_cost_l3107_310712

/-- The cost of a stack of pancakes satisfies the given conditions -/
theorem pancake_cost (pancake_stacks : ℕ) (bacon_slices : ℕ) (bacon_price : ℚ) (total_raised : ℚ) :
  pancake_stacks = 60 →
  bacon_slices = 90 →
  bacon_price = 2 →
  total_raised = 420 →
  ∃ (P : ℚ), P * pancake_stacks + bacon_price * bacon_slices = total_raised ∧ P = 4 :=
by sorry

end NUMINAMATH_CALUDE_pancake_cost_l3107_310712


namespace NUMINAMATH_CALUDE_average_movers_to_texas_l3107_310725

/-- The number of people moving to Texas over 5 days -/
def total_people : ℕ := 3500

/-- The number of days -/
def num_days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving per hour -/
def average_per_hour : ℚ :=
  total_people / (num_days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem average_movers_to_texas :
  round_to_nearest average_per_hour = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_movers_to_texas_l3107_310725


namespace NUMINAMATH_CALUDE_no_integer_solution_l3107_310759

theorem no_integer_solution : ¬ ∃ (a b : ℤ), a^2 + b^2 = 10^100 + 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3107_310759


namespace NUMINAMATH_CALUDE_trip_length_satisfies_conditions_l3107_310782

/-- Represents the total trip length in miles -/
def total_trip_length : ℝ := 180

/-- Represents the distance traveled on battery power in miles -/
def battery_distance : ℝ := 60

/-- Represents the fuel consumption rate in gallons per mile when using gasoline -/
def fuel_consumption_rate : ℝ := 0.03

/-- Represents the average fuel efficiency for the entire trip in miles per gallon -/
def average_fuel_efficiency : ℝ := 50

/-- Theorem stating that the total trip length satisfies the given conditions -/
theorem trip_length_satisfies_conditions :
  (total_trip_length / (fuel_consumption_rate * (total_trip_length - battery_distance)) = average_fuel_efficiency) ∧
  (total_trip_length > battery_distance) := by
  sorry

#check trip_length_satisfies_conditions

end NUMINAMATH_CALUDE_trip_length_satisfies_conditions_l3107_310782


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3107_310705

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let new_length := 1.3 * l
  let new_width := 1.2 * w
  let original_area := l * w
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.56 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l3107_310705


namespace NUMINAMATH_CALUDE_seven_by_seven_dissection_l3107_310708

theorem seven_by_seven_dissection :
  ∀ (a b : ℕ),
  (3 * a + 4 * b = 7 * 7) →
  (b = 1) := by
sorry

end NUMINAMATH_CALUDE_seven_by_seven_dissection_l3107_310708


namespace NUMINAMATH_CALUDE_triangle_area_is_20_16_l3107_310778

/-- Represents a line in 2D space --/
structure Line where
  slope : ℚ
  point : ℚ × ℚ

/-- Calculates the area of a triangle given three points --/
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

/-- Finds the intersection point of a line with the line x + y = 12 --/
def intersectionWithSum12 (l : Line) : ℚ × ℚ :=
  let (a, b) := l.point
  let m := l.slope
  let x := (12 - b + m*a) / (m + 1)
  (x, 12 - x)

theorem triangle_area_is_20_16 (l1 l2 : Line) :
  l1.point = (4, 4) →
  l2.point = (4, 4) →
  l1.slope = 2/3 →
  l2.slope = 3/2 →
  let p1 := (4, 4)
  let p2 := intersectionWithSum12 l1
  let p3 := intersectionWithSum12 l2
  triangleArea p1 p2 p3 = 20.16 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_is_20_16_l3107_310778


namespace NUMINAMATH_CALUDE_square_area_increase_l3107_310702

theorem square_area_increase (a : ℝ) (ha : a > 0) : 
  let side_b := 2 * a
  let side_c := side_b * 1.4
  let area_a := a ^ 2
  let area_b := side_b ^ 2
  let area_c := side_c ^ 2
  (area_c - (area_a + area_b)) / (area_a + area_b) = 0.568 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l3107_310702


namespace NUMINAMATH_CALUDE_stock_price_calculation_l3107_310749

theorem stock_price_calculation (initial_price : ℝ) : 
  let first_year_increase : ℝ := 1.5
  let second_year_decrease : ℝ := 0.3
  let third_year_increase : ℝ := 0.2
  let price_after_first_year : ℝ := initial_price * (1 + first_year_increase)
  let price_after_second_year : ℝ := price_after_first_year * (1 - second_year_decrease)
  let final_price : ℝ := price_after_second_year * (1 + third_year_increase)
  initial_price = 120 → final_price = 252 := by
sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l3107_310749


namespace NUMINAMATH_CALUDE_count_terminating_decimals_is_40_l3107_310724

/-- 
Counts the number of integers n between 1 and 120 inclusive 
for which the decimal representation of n/120 terminates.
-/
def count_terminating_decimals : ℕ :=
  let max : ℕ := 120
  let prime_factors : Multiset ℕ := {2, 2, 2, 3, 5}
  sorry

/-- The count of terminating decimals is 40 -/
theorem count_terminating_decimals_is_40 : 
  count_terminating_decimals = 40 := by sorry

end NUMINAMATH_CALUDE_count_terminating_decimals_is_40_l3107_310724


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_count_l3107_310752

/-- Represents a sequence of consecutive odd integers -/
structure ConsecutiveOddIntegers where
  n : ℕ  -- number of integers in the sequence
  first : ℤ  -- first (least) integer in the sequence
  avg : ℚ  -- average of the integers in the sequence

/-- Theorem: Given the conditions, the number of consecutive odd integers is 8 -/
theorem consecutive_odd_integers_count
  (seq : ConsecutiveOddIntegers)
  (h1 : seq.first = 407)
  (h2 : seq.avg = 414)
  : seq.n = 8 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_count_l3107_310752


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l3107_310731

-- Problem 1
theorem equation_one_solution (x : ℝ) : 
  (5 / (x - 1) = 1 / (2 * x + 1)) ↔ x = -2/3 :=
sorry

-- Problem 2
theorem equation_two_no_solution : 
  ¬∃ (x : ℝ), (1 / (x - 2) + 2 = (1 - x) / (2 - x)) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l3107_310731


namespace NUMINAMATH_CALUDE_prime_remainder_30_l3107_310758

theorem prime_remainder_30 (p : ℕ) (h : Prime p) : 
  let r := p % 30
  Prime r ∨ r = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_remainder_30_l3107_310758


namespace NUMINAMATH_CALUDE_union_of_M_and_complement_of_N_l3107_310781

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def M : Finset ℕ := {1, 3, 4}
def N : Finset ℕ := {3, 5, 6}

theorem union_of_M_and_complement_of_N :
  M ∪ (U \ N) = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_complement_of_N_l3107_310781


namespace NUMINAMATH_CALUDE_pace_difference_is_one_minute_l3107_310757

-- Define the race parameters
def square_length : ℚ := 3/4
def num_laps : ℕ := 7
def this_year_time : ℚ := 42
def last_year_time : ℚ := 47.25

-- Define the total race distance
def race_distance : ℚ := square_length * num_laps

-- Define the average pace for this year and last year
def this_year_pace : ℚ := this_year_time / race_distance
def last_year_pace : ℚ := last_year_time / race_distance

-- Theorem statement
theorem pace_difference_is_one_minute :
  last_year_pace - this_year_pace = 1 := by sorry

end NUMINAMATH_CALUDE_pace_difference_is_one_minute_l3107_310757


namespace NUMINAMATH_CALUDE_nancy_clay_pots_l3107_310785

/-- The number of clay pots Nancy created on Monday -/
def monday_pots : ℕ := 12

/-- The number of clay pots Nancy created on Tuesday -/
def tuesday_pots : ℕ := 2 * monday_pots

/-- The number of clay pots Nancy created on Wednesday -/
def wednesday_pots : ℕ := 14

/-- The total number of clay pots Nancy created by the end of the week -/
def total_pots : ℕ := monday_pots + tuesday_pots + wednesday_pots

theorem nancy_clay_pots : total_pots = 50 := by sorry

end NUMINAMATH_CALUDE_nancy_clay_pots_l3107_310785


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3107_310754

/-- The fixed point that the line (a+3)x + (2a-1)y + 7 = 0 passes through for all real a -/
def fixed_point : ℝ × ℝ := (-2, 1)

/-- The line equation as a function of a, x, and y -/
def line_equation (a x y : ℝ) : ℝ := (a + 3) * x + (2 * a - 1) * y + 7

theorem fixed_point_on_line :
  ∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3107_310754


namespace NUMINAMATH_CALUDE_exchange_impossibility_l3107_310766

theorem exchange_impossibility : ¬ ∃ (N : ℕ), 5 * N = 2001 := by sorry

end NUMINAMATH_CALUDE_exchange_impossibility_l3107_310766


namespace NUMINAMATH_CALUDE_solution_set_g_range_of_m_l3107_310794

-- Define the functions f and g
def f (x : ℝ) := x^2 - 2*x - 8
def g (x : ℝ) := 2*x^2 - 4*x - 16

-- Theorem for the solution set of g(x) < 0
theorem solution_set_g (x : ℝ) : g x < 0 ↔ -2 < x ∧ x < 4 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x > 2, f x ≥ (m + 2) * x - m - 15) → m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_g_range_of_m_l3107_310794


namespace NUMINAMATH_CALUDE_daps_equivalent_to_dips_l3107_310763

/-- The number of daps equivalent to one dop -/
def daps_per_dop : ℚ := 5 / 4

/-- The number of dips equivalent to one dop -/
def dips_per_dop : ℚ := 3

/-- The number of dips we want to convert to daps -/
def target_dips : ℚ := 54

/-- Theorem stating the equivalence between daps and dips -/
theorem daps_equivalent_to_dips : 
  (target_dips * daps_per_dop) / dips_per_dop = 22.5 := by sorry

end NUMINAMATH_CALUDE_daps_equivalent_to_dips_l3107_310763


namespace NUMINAMATH_CALUDE_intersection_M_N_union_complements_M_N_l3107_310774

open Set

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x ≥ 1}

-- Define set N
def N : Set ℝ := {x | 0 ≤ x ∧ x < 5}

-- Theorem for the intersection of M and N
theorem intersection_M_N : M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 5} := by sorry

-- Theorem for the union of complements of M and N
theorem union_complements_M_N : (U \ M) ∪ (U \ N) = {x : ℝ | x < 1 ∨ x ≥ 5} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_union_complements_M_N_l3107_310774


namespace NUMINAMATH_CALUDE_kho_kho_only_players_l3107_310726

theorem kho_kho_only_players (total : ℕ) (kabadi : ℕ) (both : ℕ) (kho_kho_only : ℕ) : 
  total = 45 → kabadi = 10 → both = 5 → kho_kho_only = total - kabadi + both :=
by
  sorry

end NUMINAMATH_CALUDE_kho_kho_only_players_l3107_310726


namespace NUMINAMATH_CALUDE_intersection_M_N_l3107_310709

-- Define the sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {y | y > -1}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo (-1) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3107_310709


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_expression_simplification_l3107_310755

-- Problem 1
theorem quadratic_equation_solution (x : ℝ) :
  x * (x + 6) = 8 * (x + 3) ↔ x = 6 ∨ x = -4 := by sorry

-- Problem 2
theorem expression_simplification (a b : ℝ) (h : a ≠ b) :
  (3 * a^2 - 3 * b^2) / (a^2 * b + a * b^2) / (1 - (a^2 + b^2) / (2 * a * b)) = -6 / (a - b) := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_expression_simplification_l3107_310755


namespace NUMINAMATH_CALUDE_tiffany_lives_l3107_310783

/-- Calculates the final number of lives in a video game scenario -/
def final_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Proves that for the given scenario, the final number of lives is 56 -/
theorem tiffany_lives : final_lives 43 14 27 = 56 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_lives_l3107_310783


namespace NUMINAMATH_CALUDE_katie_total_marbles_l3107_310703

/-- The number of marbles Katie has -/
def total_marbles (pink orange purple : ℕ) : ℕ := pink + orange + purple

/-- The properties of Katie's marble collection -/
def katie_marbles (pink orange purple : ℕ) : Prop :=
  pink = 13 ∧ orange = pink - 9 ∧ purple = 4 * orange

theorem katie_total_marbles :
  ∀ pink orange purple : ℕ,
    katie_marbles pink orange purple →
    total_marbles pink orange purple = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_katie_total_marbles_l3107_310703


namespace NUMINAMATH_CALUDE_age_difference_is_four_l3107_310721

/-- Gladys' current age -/
def gladys_age : ℕ := 40 - 10

/-- Juanico's current age -/
def juanico_age : ℕ := 41 - 30

/-- The difference between half of Gladys' age and Juanico's age -/
def age_difference : ℕ := (gladys_age / 2) - juanico_age

theorem age_difference_is_four : age_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_four_l3107_310721


namespace NUMINAMATH_CALUDE_intersection_count_possibilities_l3107_310792

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define a line
def Line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define intersection of two lines
def LinesIntersect (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ x y, Line l1.1 l1.2.1 l1.2.2 x y ∧ Line l2.1 l2.2.1 l2.2.2 x y

-- Define when a line is not tangent to the ellipse
def NotTangent (l : ℝ × ℝ × ℝ) : Prop :=
  ∃ x1 y1 x2 y2, x1 ≠ x2 ∧ y1 ≠ y2 ∧
    Line l.1 l.2.1 l.2.2 x1 y1 ∧ Line l.1 l.2.1 l.2.2 x2 y2 ∧
    Ellipse x1 y1 ∧ Ellipse x2 y2

-- Define the number of intersection points
def IntersectionCount (l1 l2 : ℝ × ℝ × ℝ) : ℕ :=
  sorry

-- Theorem statement
theorem intersection_count_possibilities
  (l1 l2 : ℝ × ℝ × ℝ)
  (h1 : LinesIntersect l1 l2)
  (h2 : NotTangent l1)
  (h3 : NotTangent l2) :
  (IntersectionCount l1 l2 = 2) ∨
  (IntersectionCount l1 l2 = 3) ∨
  (IntersectionCount l1 l2 = 4) :=
sorry

end NUMINAMATH_CALUDE_intersection_count_possibilities_l3107_310792
