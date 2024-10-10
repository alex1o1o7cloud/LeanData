import Mathlib

namespace meshed_gears_speed_proportion_l763_76321

/-- Represents a gear with number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Proves that for four meshed gears, their angular speeds are proportional to yzw : xzw : xyw : xyz -/
theorem meshed_gears_speed_proportion
  (A B C D : Gear)
  (h_mesh : A.teeth * A.speed = B.teeth * B.speed ∧
            B.teeth * B.speed = C.teeth * C.speed ∧
            C.teeth * C.speed = D.teeth * D.speed) :
  ∃ (k : ℝ), k ≠ 0 ∧
    A.speed = k * (B.teeth * C.teeth * D.teeth) ∧
    B.speed = k * (A.teeth * C.teeth * D.teeth) ∧
    C.speed = k * (A.teeth * B.teeth * D.teeth) ∧
    D.speed = k * (A.teeth * B.teeth * C.teeth) :=
sorry

end meshed_gears_speed_proportion_l763_76321


namespace drug_molecule_diameter_scientific_notation_l763_76352

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem drug_molecule_diameter_scientific_notation :
  toScientificNotation 0.00000008 = ScientificNotation.mk 8 (-8) sorry := by
  sorry

end drug_molecule_diameter_scientific_notation_l763_76352


namespace blocks_required_for_specified_wall_l763_76327

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  height : ℕ
  length₁ : ℕ
  length₂ : ℕ

/-- Calculates the number of blocks required for a wall with given specifications --/
def calculateBlocksRequired (wall : WallDimensions) (block : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating that the number of blocks required for the specified wall is 404 --/
theorem blocks_required_for_specified_wall :
  let wall := WallDimensions.mk 150 8
  let block := BlockDimensions.mk 1 3 2
  calculateBlocksRequired wall block = 404 :=
by sorry

end blocks_required_for_specified_wall_l763_76327


namespace value_of_b_is_two_l763_76326

def f (x : ℝ) := x^2 - 2*x + 2

theorem value_of_b_is_two :
  ∃ b : ℝ, b > 1 ∧
  (∀ x, x ∈ Set.Icc 1 b ↔ f x ∈ Set.Icc 1 b) ∧
  b = 2 := by
sorry

end value_of_b_is_two_l763_76326


namespace sqrt_equation_solutions_l763_76306

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
  sorry

end sqrt_equation_solutions_l763_76306


namespace trapezoid_sides_l763_76365

-- Define the right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2
  is_345 : a = 3 ∧ b = 4 ∧ c = 5

-- Define the perpendicular line
def perpendicular_line (t : RightTriangle) (d : ℝ) : Prop :=
  d = 1 ∨ d = t.c - 1

-- Define the trapezoid formed by the perpendicular line
structure Trapezoid where
  s1 : ℝ
  s2 : ℝ
  s3 : ℝ
  s4 : ℝ

-- Theorem statement
theorem trapezoid_sides (t : RightTriangle) (d : ℝ) (trap : Trapezoid) 
  (h1 : perpendicular_line t d) :
  (trap.s1 = trap.s4 ∧ trap.s2 = trap.s3) ∧
  ((trap.s1 = 3 ∧ trap.s2 = 3/2) ∨ (trap.s1 = 4 ∧ trap.s2 = 4/3)) := by
  sorry

end trapezoid_sides_l763_76365


namespace parallelogram_base_l763_76300

theorem parallelogram_base (area height : ℝ) (h1 : area = 375) (h2 : height = 15) :
  area / height = 25 := by
  sorry

end parallelogram_base_l763_76300


namespace rectangle_area_l763_76351

/-- A rectangle with length thrice its breadth and perimeter 56 meters has an area of 147 square meters. -/
theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 56) : l * b = 147 := by
  sorry

end rectangle_area_l763_76351


namespace bus_rental_solution_l763_76329

/-- Represents the bus rental problem for a school study tour. -/
structure BusRentalProblem where
  capacityA : Nat  -- Capacity of A type bus
  capacityB : Nat  -- Capacity of B type bus
  extraPeople : Nat  -- People without seats in original plan
  fewerBusesB : Nat  -- Number of fewer B type buses needed
  totalBuses : Nat  -- Total number of buses to be rented
  maxTypeB : Nat  -- Maximum number of B type buses
  feeA : Nat  -- Rental fee for A type bus
  feeB : Nat  -- Rental fee for B type bus

/-- Represents a bus rental scheme. -/
structure RentalScheme where
  numA : Nat  -- Number of A type buses
  numB : Nat  -- Number of B type buses

/-- The main theorem about the bus rental problem. -/
theorem bus_rental_solution (p : BusRentalProblem)
  (h1 : p.capacityA = 45)
  (h2 : p.capacityB = 60)
  (h3 : p.extraPeople = 30)
  (h4 : p.fewerBusesB = 6)
  (h5 : p.totalBuses = 25)
  (h6 : p.maxTypeB = 7)
  (h7 : p.feeA = 220)
  (h8 : p.feeB = 300) :
  ∃ (originalA totalPeople : Nat) (schemes : List RentalScheme) (bestScheme : RentalScheme),
    originalA = 26 ∧
    totalPeople = 1200 ∧
    schemes = [⟨20, 5⟩, ⟨19, 6⟩, ⟨18, 7⟩] ∧
    bestScheme = ⟨20, 5⟩ ∧
    (∀ scheme ∈ schemes, 
      scheme.numA + scheme.numB = p.totalBuses ∧
      scheme.numB ≤ p.maxTypeB ∧
      scheme.numA * p.capacityA + scheme.numB * p.capacityB ≥ totalPeople) ∧
    (∀ scheme ∈ schemes,
      scheme.numA * p.feeA + scheme.numB * p.feeB ≥ 
      bestScheme.numA * p.feeA + bestScheme.numB * p.feeB) := by
  sorry

end bus_rental_solution_l763_76329


namespace trapezoid_square_area_equality_l763_76304

/-- Given a trapezoid with upper side 15 cm, lower side 9 cm, and height 12 cm,
    the side length of a square with the same area as the trapezoid is 12 cm. -/
theorem trapezoid_square_area_equality (upper_side lower_side height : ℝ) 
    (h1 : upper_side = 15)
    (h2 : lower_side = 9)
    (h3 : height = 12) :
    ∃ (square_side : ℝ), 
      (1/2 * (upper_side + lower_side) * height = square_side^2) ∧ 
      square_side = 12 := by
  sorry

end trapezoid_square_area_equality_l763_76304


namespace nori_crayon_problem_l763_76334

/-- Given the initial number of crayon boxes, crayons per box, crayons given to Mae, and crayons left,
    calculate the difference between crayons given to Lea and Mae. -/
def crayon_difference (boxes : ℕ) (crayons_per_box : ℕ) (given_to_mae : ℕ) (crayons_left : ℕ) : ℕ :=
  boxes * crayons_per_box - given_to_mae - crayons_left - given_to_mae

theorem nori_crayon_problem :
  crayon_difference 4 8 5 15 = 7 := by
  sorry

end nori_crayon_problem_l763_76334


namespace no_preimage_range_l763_76328

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := Set.univ

def f (x : ℝ) : ℝ := -x^2 + 2*x - 1

theorem no_preimage_range :
  {p : ℝ | p ∈ B ∧ ∀ x ∈ A, f x ≠ p} = {p : ℝ | p ≥ -1} := by
  sorry

end no_preimage_range_l763_76328


namespace courtyard_paving_l763_76337

theorem courtyard_paving (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (brick_width : ℝ) :
  courtyard_length = 25 →
  courtyard_width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  ⌈(courtyard_length * courtyard_width) / (brick_length * brick_width)⌉ = 20000 := by
  sorry

end courtyard_paving_l763_76337


namespace pentagon_area_l763_76396

/-- A point on a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- A pentagon on a grid -/
structure GridPentagon where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint
  v4 : GridPoint
  v5 : GridPoint

/-- Count the number of integer points strictly inside a polygon -/
def countInteriorPoints (p : GridPentagon) : Int :=
  sorry

/-- Count the number of integer points on the boundary of a polygon -/
def countBoundaryPoints (p : GridPentagon) : Int :=
  sorry

/-- Calculate the area of a polygon using Pick's theorem -/
def polygonArea (p : GridPentagon) : Int :=
  countInteriorPoints p + (countBoundaryPoints p / 2) - 1

theorem pentagon_area :
  let p : GridPentagon := {
    v1 := { x := 0, y := 1 },
    v2 := { x := 2, y := 5 },
    v3 := { x := 6, y := 3 },
    v4 := { x := 5, y := 0 },
    v5 := { x := 1, y := 0 }
  }
  polygonArea p = 17 := by
  sorry

end pentagon_area_l763_76396


namespace special_triangle_sum_l763_76307

/-- Triangle ABC with given side lengths and circles P and Q with specific properties -/
structure SpecialTriangle where
  -- Side lengths of triangle ABC
  AB : ℝ
  AC : ℝ
  BC : ℝ
  -- Radius of circle P
  radiusP : ℝ
  -- Radius of circle Q (to be determined)
  radiusQ : ℝ
  -- Conditions
  isosceles : AB = AC
  tangentP : radiusP < AB ∧ radiusP < BC
  tangentQ : radiusQ < AB ∧ radiusQ < BC
  externalTangent : radiusQ + radiusP < BC
  -- Representation of radiusQ
  m : ℕ
  n : ℕ
  k : ℕ
  radiusQForm : radiusQ = m - n * Real.sqrt k
  kPrime : Nat.Prime k

/-- The main theorem stating the sum of m and nk for the special triangle -/
theorem special_triangle_sum (t : SpecialTriangle) 
  (h1 : t.AB = 130)
  (h2 : t.BC = 150)
  (h3 : t.radiusP = 20) :
  t.m + t.n * t.k = 386 := by
  sorry

end special_triangle_sum_l763_76307


namespace dartboard_central_angles_l763_76371

/-- Represents a region on a circular dartboard -/
structure DartboardRegion where
  probability : ℚ
  centralAngle : ℚ

/-- Theorem: Given the probabilities of hitting regions A and B on a circular dartboard,
    prove that their central angles are 45° and 30° respectively -/
theorem dartboard_central_angles 
  (regionA regionB : DartboardRegion)
  (hA : regionA.probability = 1/8)
  (hB : regionB.probability = 1/12)
  (h_total : regionA.centralAngle + regionB.centralAngle ≤ 360) :
  regionA.centralAngle = 45 ∧ regionB.centralAngle = 30 := by
  sorry

end dartboard_central_angles_l763_76371


namespace sqrt_sum_rational_implies_both_rational_l763_76356

theorem sqrt_sum_rational_implies_both_rational 
  (a b : ℚ) 
  (h : ∃ (q : ℚ), q = Real.sqrt a + Real.sqrt b) : 
  (∃ (r : ℚ), r = Real.sqrt a) ∧ (∃ (s : ℚ), s = Real.sqrt b) := by
sorry

end sqrt_sum_rational_implies_both_rational_l763_76356


namespace expression_simplification_l763_76332

theorem expression_simplification (a b c k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hsum : a + b + c = 0) :
  (k * a^2 * b^2 + k * a^2 * c^2 + k * b^2 * c^2) / 
  ((a^2 - b*c)*(b^2 - a*c) + (a^2 - b*c)*(c^2 - a*b) + (b^2 - a*c)*(c^2 - a*b)) = k/3 := by
  sorry

end expression_simplification_l763_76332


namespace sequence_divergence_criterion_l763_76354

/-- Given a sequence xₙ and a limit point a, prove that for every ε > 0,
    there exists a number k such that for all n > k, |xₙ - a| ≥ ε -/
theorem sequence_divergence_criterion 
  (x : ℕ → ℝ) (a : ℝ) : 
  ∀ ε > 0, ∃ k : ℕ, ∀ n > k, |x n - a| ≥ ε := by
  sorry

end sequence_divergence_criterion_l763_76354


namespace valid_sequence_count_l763_76309

/-- Represents a sequence of coin tosses -/
def CoinSequence := List Bool

/-- Counts the number of occurrences of a specific subsequence in a coin sequence -/
def countSubsequence (seq : CoinSequence) (subseq : CoinSequence) : Nat :=
  sorry

/-- Checks if a coin sequence satisfies the given conditions -/
def isValidSequence (seq : CoinSequence) : Prop :=
  (countSubsequence seq [true, true] = 3) ∧
  (countSubsequence seq [true, false] = 4) ∧
  (countSubsequence seq [false, true] = 5) ∧
  (countSubsequence seq [false, false] = 6)

/-- The number of valid coin sequences -/
def validSequenceCount : Nat :=
  sorry

theorem valid_sequence_count :
  validSequenceCount = 16170 :=
sorry

end valid_sequence_count_l763_76309


namespace range_of_a_l763_76345

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, x - a ≤ 0) → a ∈ Set.Ici 2 :=
by sorry

end range_of_a_l763_76345


namespace six_digit_numbers_with_at_least_two_zeros_l763_76367

/-- The number of digits in the numbers we're considering -/
def num_digits : ℕ := 6

/-- The total number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 9 * 10^5

/-- The number of 6-digit numbers with no zeros -/
def numbers_with_no_zeros : ℕ := 9^6

/-- The number of 6-digit numbers with exactly one zero -/
def numbers_with_one_zero : ℕ := 5 * 9^5

/-- The number of 6-digit numbers with at least two zeros -/
def numbers_with_at_least_two_zeros : ℕ := total_six_digit_numbers - (numbers_with_no_zeros + numbers_with_one_zero)

theorem six_digit_numbers_with_at_least_two_zeros :
  numbers_with_at_least_two_zeros = 73314 := by
  sorry

end six_digit_numbers_with_at_least_two_zeros_l763_76367


namespace chicken_duck_count_l763_76375

theorem chicken_duck_count : ∃ (chickens ducks : ℕ),
  chickens + ducks = 239 ∧
  ducks = 3 * chickens + 15 ∧
  chickens = 56 ∧
  ducks = 183 := by
  sorry

end chicken_duck_count_l763_76375


namespace puzzle_piece_ratio_l763_76366

theorem puzzle_piece_ratio (total pieces : ℕ) (border : ℕ) (trevor : ℕ) (missing : ℕ) :
  total = 500 →
  border = 75 →
  trevor = 105 →
  missing = 5 →
  ∃ (joe : ℕ), joe = total - border - trevor - missing ∧ joe = 3 * trevor :=
by sorry

end puzzle_piece_ratio_l763_76366


namespace hexagon_area_equal_perimeter_l763_76311

/-- The area of a regular hexagon with the same perimeter as a square of area 16 -/
theorem hexagon_area_equal_perimeter (square_area : ℝ) (square_side : ℝ) (hex_side : ℝ) :
  square_area = 16 →
  square_side^2 = square_area →
  4 * square_side = 6 * hex_side →
  (3 * hex_side^2 * Real.sqrt 3) / 2 = (32 * Real.sqrt 3) / 3 :=
by sorry

end hexagon_area_equal_perimeter_l763_76311


namespace tangerine_consumption_change_l763_76385

/-- Demand function before embargo -/
def initial_demand (p : ℝ) : ℝ := 50 - p

/-- Demand function after embargo -/
def new_demand (p : ℝ) : ℝ := 2.5 * (50 - p)

/-- Marginal cost (constant) -/
def marginal_cost : ℝ := 5

/-- Initial equilibrium quantity under perfect competition -/
def initial_equilibrium_quantity : ℝ := initial_demand marginal_cost

/-- New equilibrium quantity under monopoly -/
noncomputable def new_equilibrium_quantity : ℝ := 56.25

theorem tangerine_consumption_change :
  new_equilibrium_quantity / initial_equilibrium_quantity = 1.25 := by sorry

end tangerine_consumption_change_l763_76385


namespace quadratic_one_root_l763_76312

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem quadratic_one_root : ∃! x : ℝ, f x = 0 := by
  sorry

end quadratic_one_root_l763_76312


namespace divisors_multiple_of_five_3780_l763_76308

/-- The number of positive divisors of 3780 that are multiples of 5 -/
def divisors_multiple_of_five (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d % 5 = 0) (Nat.divisors n)).card

/-- Theorem stating that the number of positive divisors of 3780 that are multiples of 5 is 24 -/
theorem divisors_multiple_of_five_3780 :
  divisors_multiple_of_five 3780 = 24 := by
  sorry

end divisors_multiple_of_five_3780_l763_76308


namespace johns_journey_cost_l763_76333

/-- Calculates the total cost of John's journey given the specified conditions. -/
theorem johns_journey_cost : 
  let rental_cost : ℚ := 150
  let rental_discount : ℚ := 0.15
  let gas_cost_per_gallon : ℚ := 3.5
  let gas_gallons : ℚ := 8
  let driving_cost_per_mile : ℚ := 0.5
  let initial_distance : ℚ := 320
  let additional_distance : ℚ := 50
  let toll_fees : ℚ := 15
  let parking_cost_per_day : ℚ := 20
  let parking_days : ℚ := 3
  let meals_lodging_cost_per_day : ℚ := 70
  let meals_lodging_days : ℚ := 2

  let discounted_rental := rental_cost * (1 - rental_discount)
  let total_gas_cost := gas_cost_per_gallon * gas_gallons
  let total_distance := initial_distance + additional_distance
  let total_driving_cost := driving_cost_per_mile * total_distance
  let total_parking_cost := parking_cost_per_day * parking_days
  let total_meals_lodging := meals_lodging_cost_per_day * meals_lodging_days

  let total_cost := discounted_rental + total_gas_cost + total_driving_cost + 
                    toll_fees + total_parking_cost + total_meals_lodging

  total_cost = 555.5 := by sorry

end johns_journey_cost_l763_76333


namespace octal_to_decimal_l763_76318

theorem octal_to_decimal : (3 * 8^2 + 6 * 8^1 + 7 * 8^0) = 247 := by
  sorry

end octal_to_decimal_l763_76318


namespace sum_of_y_coordinates_on_y_axis_l763_76346

-- Define the circle
def circle_center : ℝ × ℝ := (-3, 5)
def circle_radius : ℝ := 15

-- Define a function to represent the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Theorem statement
theorem sum_of_y_coordinates_on_y_axis :
  ∃ y₁ y₂ : ℝ, 
    circle_equation 0 y₁ ∧ 
    circle_equation 0 y₂ ∧ 
    y₁ ≠ y₂ ∧ 
    y₁ + y₂ = 10 :=
sorry

end sum_of_y_coordinates_on_y_axis_l763_76346


namespace equal_earnings_l763_76376

theorem equal_earnings (t : ℝ) : 
  (t - 6) * (2 * t - 5) = (2 * t - 8) * (t - 5) → t = 10 := by
  sorry

end equal_earnings_l763_76376


namespace rectangle_dimensions_area_l763_76394

theorem rectangle_dimensions_area (x : ℝ) : 
  (x - 2) * (2 * x + 5) = 8 * x - 6 → x = 4 :=
by sorry

end rectangle_dimensions_area_l763_76394


namespace mod_equivalence_unique_solution_l763_76380

theorem mod_equivalence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [ZMOD 10] ∧ n = 2 := by
  sorry

end mod_equivalence_unique_solution_l763_76380


namespace union_of_A_and_B_l763_76302

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - x - 6 ≤ 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 ≤ x ∧ x < 4} := by
  sorry

end union_of_A_and_B_l763_76302


namespace inequality_implication_l763_76341

theorem inequality_implication (x y : ℝ) (h : x > y) : 2*x - 1 > 2*y - 1 := by
  sorry

end inequality_implication_l763_76341


namespace sqrt_two_times_sqrt_three_equals_sqrt_six_l763_76350

theorem sqrt_two_times_sqrt_three_equals_sqrt_six :
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_two_times_sqrt_three_equals_sqrt_six_l763_76350


namespace four_digit_reverse_pairs_l763_76320

def reverse_digits (n : ℕ) : ℕ :=
  let digits := String.toList (toString n)
  String.toNat! (String.mk (List.reverse digits))

def ends_with_three_zeros (n : ℕ) : Prop :=
  n % 1000 = 0

theorem four_digit_reverse_pairs : 
  ∀ (a b : ℕ), 
    1000 ≤ a ∧ a < 10000 ∧
    1000 ≤ b ∧ b < 10000 ∧
    b = reverse_digits a ∧
    ends_with_three_zeros (a * b) →
    ((a = 5216 ∧ b = 6125) ∨
     (a = 5736 ∧ b = 6375) ∨
     (a = 5264 ∧ b = 4625) ∨
     (a = 5784 ∧ b = 4875))
  := by sorry

end four_digit_reverse_pairs_l763_76320


namespace power_five_2023_mod_11_l763_76301

theorem power_five_2023_mod_11 : 5^2023 % 11 = 4 := by
  sorry

end power_five_2023_mod_11_l763_76301


namespace max_min_values_on_interval_l763_76344

-- Define the function f(x) = 3x - x³
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Define the interval [2, 3]
def interval : Set ℝ := { x | 2 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem max_min_values_on_interval :
  (∀ x ∈ interval, f x ≤ f 2) ∧
  (∀ x ∈ interval, f 3 ≤ f x) ∧
  (f 2 = -2) ∧
  (f 3 = -18) :=
sorry

end max_min_values_on_interval_l763_76344


namespace boys_age_l763_76393

theorem boys_age (current_age : ℕ) : 
  (current_age = 2 * (current_age - 5)) → current_age = 10 := by
  sorry

end boys_age_l763_76393


namespace m_value_proof_l763_76317

theorem m_value_proof (m : ℝ) : 
  (m > 0) ∧ 
  (∀ x : ℝ, (x / (x - 1) < 0 → 0 < x ∧ x < m)) ∧ 
  (∃ x : ℝ, 0 < x ∧ x < m ∧ x / (x - 1) ≥ 0) →
  m = 1/2 := by
sorry

end m_value_proof_l763_76317


namespace divisibility_implies_value_l763_76360

theorem divisibility_implies_value (p q r : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, x^4 + 6*x^3 + 8*p*x^2 + 6*q*x + r = (x^3 + 4*x^2 + 16*x + 4) * k) →
  (p + q) * r = 56 := by
  sorry

end divisibility_implies_value_l763_76360


namespace wire_cutting_l763_76323

theorem wire_cutting (total_length : ℝ) (difference : ℝ) (longer_piece : ℝ) : 
  total_length = 30 →
  difference = 2 →
  longer_piece = total_length / 2 + difference / 2 →
  longer_piece = 16 :=
by
  sorry

end wire_cutting_l763_76323


namespace simplify_expression_l763_76395

theorem simplify_expression (a b : ℝ) : (22*a + 60*b) + (10*a + 29*b) - (9*a + 50*b) = 23*a + 39*b := by
  sorry

end simplify_expression_l763_76395


namespace billboard_perimeter_l763_76331

theorem billboard_perimeter (area : ℝ) (short_side : ℝ) (perimeter : ℝ) : 
  area = 104 → 
  short_side = 8 → 
  perimeter = 2 * (area / short_side + short_side) →
  perimeter = 42 := by
sorry


end billboard_perimeter_l763_76331


namespace performing_arts_school_l763_76390

theorem performing_arts_school (total : ℕ) (cant_sing cant_dance cant_act : ℕ) :
  total = 150 ∧
  cant_sing = 80 ∧
  cant_dance = 110 ∧
  cant_act = 60 →
  ∃ (all_talents : ℕ),
    all_talents = total - ((total - cant_sing) + (total - cant_dance) + (total - cant_act) - total) ∧
    all_talents = 50 := by
  sorry

end performing_arts_school_l763_76390


namespace max_min_f_on_interval_l763_76343

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  let a := -3
  let b := 0
  ∃ (x_max x_min : ℝ),
    x_max ∈ Set.Icc a b ∧
    x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 3 ∧
    f x_min = -17 :=
by sorry

end max_min_f_on_interval_l763_76343


namespace share_ratio_l763_76372

theorem share_ratio (total amount : ℕ) (a_share b_share c_share : ℕ) 
  (h1 : total = 595)
  (h2 : a_share = 420)
  (h3 : b_share = 105)
  (h4 : c_share = 70)
  (h5 : total = a_share + b_share + c_share)
  (h6 : b_share = c_share / 4) :
  a_share / b_share = 4 := by
sorry

end share_ratio_l763_76372


namespace first_part_length_l763_76355

/-- Given a trip with the following conditions:
  * Total distance is 50 km
  * First part is traveled at 66 km/h
  * Second part is traveled at 33 km/h
  * Average speed of the entire trip is 44 km/h
  Prove that the length of the first part of the trip is 25 km -/
theorem first_part_length (total_distance : ℝ) (speed1 speed2 avg_speed : ℝ) 
  (h1 : total_distance = 50)
  (h2 : speed1 = 66)
  (h3 : speed2 = 33)
  (h4 : avg_speed = 44)
  (h5 : ∃ x : ℝ, x > 0 ∧ x < total_distance ∧ 
        avg_speed = total_distance / (x / speed1 + (total_distance - x) / speed2)) :
  ∃ x : ℝ, x = 25 ∧ x > 0 ∧ x < total_distance ∧ 
    avg_speed = total_distance / (x / speed1 + (total_distance - x) / speed2) := by
  sorry

end first_part_length_l763_76355


namespace arithmetic_sqrt_of_neg_four_squared_l763_76388

theorem arithmetic_sqrt_of_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end arithmetic_sqrt_of_neg_four_squared_l763_76388


namespace equation_solutions_l763_76324

-- Define the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos (2 * x)

-- State the theorem
theorem equation_solutions :
  ∃! (solutions : Finset ℝ), solutions.card = 60 ∧ ∀ x, x ∈ solutions ↔ equation x :=
sorry

end equation_solutions_l763_76324


namespace dice_sum_not_22_l763_76310

theorem dice_sum_not_22 (a b c d e : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  1 ≤ e ∧ e ≤ 6 →
  a * b * c * d * e = 432 →
  a + b + c + d + e ≠ 22 := by
sorry

end dice_sum_not_22_l763_76310


namespace no_equal_sums_l763_76381

theorem no_equal_sums : ¬∃ (n : ℕ+), 
  (5 * n * (n + 1) : ℚ) / 2 = (5 * n * (n + 7) : ℚ) / 2 := by
  sorry

end no_equal_sums_l763_76381


namespace eventually_point_difference_exceeds_50_l763_76347

/-- Represents a player in the tournament -/
structure Player where
  id : Nat
  points : Int

/-- Represents the state of the tournament on a given day -/
structure TournamentDay where
  players : Vector Player 200
  day : Nat

/-- Function to sort players by their points -/
def sortPlayers (players : Vector Player 200) : Vector Player 200 := sorry

/-- Function to play matches for a day and update points -/
def playMatches (t : TournamentDay) : TournamentDay := sorry

/-- Predicate to check if the point difference exceeds 50 -/
def pointDifferenceExceeds50 (t : TournamentDay) : Prop :=
  ∃ i j, i < 200 ∧ j < 200 ∧ (t.players.get i).points - (t.players.get j).points > 50

/-- The main theorem to be proved -/
theorem eventually_point_difference_exceeds_50 :
  ∃ n : Nat, ∃ t : TournamentDay, t.day = n ∧ pointDifferenceExceeds50 t :=
sorry

end eventually_point_difference_exceeds_50_l763_76347


namespace remove_five_for_target_average_l763_76362

def original_list : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def target_average : Rat := 41/5  -- 8.2 as a rational number

theorem remove_five_for_target_average :
  let remaining_list := original_list.filter (· ≠ 5)
  (remaining_list.sum : Rat) / remaining_list.length = target_average := by
  sorry

end remove_five_for_target_average_l763_76362


namespace hana_stamp_collection_l763_76303

/-- The value of Hana's entire stamp collection -/
def total_value : ℚ := 980 / 27

/-- The fraction of the collection Hana sold -/
def sold_fraction : ℚ := 4/7 + 1/3 * (3/7) + 1/5 * (2/7)

/-- The amount Hana earned from her sales -/
def earned_amount : ℚ := 28

theorem hana_stamp_collection :
  sold_fraction * total_value = earned_amount :=
sorry

end hana_stamp_collection_l763_76303


namespace count_leap_years_l763_76339

def is_leap_year (year : ℕ) : Bool :=
  if year % 100 = 0 then year % 400 = 0 else year % 4 = 0

def years : List ℕ := [1964, 1978, 1995, 1996, 2001, 2100]

theorem count_leap_years : (years.filter is_leap_year).length = 2 := by
  sorry

end count_leap_years_l763_76339


namespace cone_base_circumference_l763_76398

/-- The circumference of the base of a right circular cone formed from a circular piece of paper 
    with radius 6 inches, after removing a 180° sector, is equal to 6π inches. -/
theorem cone_base_circumference (r : ℝ) (h : r = 6) : 
  (2 * π * r) * (1/2) = 6 * π := by sorry

end cone_base_circumference_l763_76398


namespace hexagon_percentage_is_62_5_l763_76338

/-- Represents the tiling pattern of the plane -/
structure TilingPattern where
  /-- The number of smaller squares in each large square -/
  total_squares : ℕ
  /-- The number of smaller squares used to form hexagons in each large square -/
  hexagon_squares : ℕ

/-- Calculates the percentage of the plane enclosed by hexagons -/
def hexagon_percentage (pattern : TilingPattern) : ℚ :=
  (pattern.hexagon_squares : ℚ) / (pattern.total_squares : ℚ) * 100

/-- The theorem stating that the percentage of the plane enclosed by hexagons is 62.5% -/
theorem hexagon_percentage_is_62_5 (pattern : TilingPattern) 
  (h1 : pattern.total_squares = 16)
  (h2 : pattern.hexagon_squares = 10) : 
  hexagon_percentage pattern = 62.5 := by
  sorry

#eval hexagon_percentage { total_squares := 16, hexagon_squares := 10 }

end hexagon_percentage_is_62_5_l763_76338


namespace two_apples_per_slice_l763_76379

/-- The number of apples in each slice of pie -/
def apples_per_slice (total_apples : ℕ) (num_pies : ℕ) (slices_per_pie : ℕ) : ℚ :=
  total_apples / (num_pies * slices_per_pie)

/-- Theorem: Given the conditions, prove that there are 2 apples in each slice of pie -/
theorem two_apples_per_slice :
  let total_apples : ℕ := 4 * 12  -- 4 dozen apples
  let num_pies : ℕ := 4
  let slices_per_pie : ℕ := 6
  apples_per_slice total_apples num_pies slices_per_pie = 2 := by
  sorry

end two_apples_per_slice_l763_76379


namespace factorization_cubic_minus_xy_squared_l763_76340

theorem factorization_cubic_minus_xy_squared (x y : ℝ) : 
  x^3 - x*y^2 = x*(x+y)*(x-y) := by sorry

end factorization_cubic_minus_xy_squared_l763_76340


namespace puzzle_solution_l763_76357

theorem puzzle_solution (x y : ℤ) (h1 : 3 * x + 4 * y = 150) (h2 : x = 15 ∨ y = 15) : 
  (x ≠ 15 → x = 30) ∧ (y ≠ 15 → y = 30) :=
by sorry

end puzzle_solution_l763_76357


namespace subset_implies_m_equals_one_l763_76319

def A (m : ℝ) : Set ℝ := {-1, 2, 2*m-1}
def B (m : ℝ) : Set ℝ := {2, m^2}

theorem subset_implies_m_equals_one (m : ℝ) :
  B m ⊆ A m → m = 1 := by
  sorry

end subset_implies_m_equals_one_l763_76319


namespace number_equation_solution_l763_76348

theorem number_equation_solution :
  ∃ x : ℚ, (35 + 3 * x = 51) ∧ (x = 16 / 3) :=
by sorry

end number_equation_solution_l763_76348


namespace seven_eighths_of_48_l763_76335

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by sorry

end seven_eighths_of_48_l763_76335


namespace sum_of_reciprocal_products_l763_76368

theorem sum_of_reciprocal_products (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 2022*a^2 + 1011 = 0 →
  b^3 - 2022*b^2 + 1011 = 0 →
  c^3 - 2022*c^2 + 1011 = 0 →
  1/(a*b) + 1/(b*c) + 1/(a*c) = -2 := by
sorry

end sum_of_reciprocal_products_l763_76368


namespace cake_mix_buyers_l763_76325

theorem cake_mix_buyers (total : ℕ) (muffin : ℕ) (both : ℕ) (neither_prob : ℚ) 
  (h1 : total = 100)
  (h2 : muffin = 40)
  (h3 : both = 19)
  (h4 : neither_prob = 29/100) :
  ∃ cake : ℕ, cake = 50 ∧ 
    cake + muffin - both = total - (neither_prob * total).num := by
  sorry

end cake_mix_buyers_l763_76325


namespace trajectory_equation_l763_76363

/-- The circle C -/
def C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

/-- The fixed point F -/
def F : ℝ × ℝ := (2, 0)

/-- Predicate for a point being on the trajectory of Q -/
def on_trajectory (x y : ℝ) : Prop :=
  ∃ (px py : ℝ),
    C px py ∧
    ∃ (qx qy : ℝ),
      -- Q is on the perpendicular bisector of PF
      (qx - px)^2 + (qy - py)^2 = (qx - F.1)^2 + (qy - F.2)^2 ∧
      -- Q is on the line CP
      (qx + 2) * py = (qy) * (px + 2) ∧
      -- Q is the point we're considering
      qx = x ∧ qy = y

/-- The main theorem -/
theorem trajectory_equation :
  ∀ x y : ℝ, on_trajectory x y ↔ x^2 - y^2/3 = 1 := by sorry

end trajectory_equation_l763_76363


namespace correct_lineup_count_l763_76359

-- Define the total number of players
def total_players : ℕ := 15

-- Define the number of players in the starting lineup
def lineup_size : ℕ := 6

-- Define the number of guaranteed players (All-Stars)
def guaranteed_players : ℕ := 3

-- Define the number of goalkeepers
def goalkeepers : ℕ := 1

-- Define the function to calculate the number of possible lineups
def possible_lineups : ℕ := Nat.choose (total_players - guaranteed_players - goalkeepers) (lineup_size - guaranteed_players - goalkeepers)

-- Theorem statement
theorem correct_lineup_count : possible_lineups = 55 := by
  sorry

end correct_lineup_count_l763_76359


namespace integer_pairs_satisfying_inequality_l763_76399

theorem integer_pairs_satisfying_inequality :
  ∀ a b : ℕ+, 
    (11 * a * b ≤ a^3 - b^3 ∧ a^3 - b^3 ≤ 12 * a * b) ↔ 
    ((a = 30 ∧ b = 25) ∨ (a = 8 ∧ b = 4)) := by
  sorry

end integer_pairs_satisfying_inequality_l763_76399


namespace class_average_marks_l763_76387

theorem class_average_marks (students1 students2 : ℕ) (avg2 combined_avg : ℚ) :
  students1 = 12 →
  students2 = 28 →
  avg2 = 60 →
  combined_avg = 54 →
  (students1 : ℚ) * (40 : ℚ) + (students2 : ℚ) * avg2 = (students1 + students2 : ℚ) * combined_avg :=
by sorry

end class_average_marks_l763_76387


namespace phone_number_A_value_l763_76369

def phone_number (A B C D E F G H I J : ℕ) : Prop :=
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  G > H ∧ H > I ∧ I > J ∧
  D % 2 = 0 ∧ E % 2 = 0 ∧ F % 2 = 0 ∧
  D = E + 2 ∧ E = F + 2 ∧
  G % 2 = 1 ∧ H % 2 = 1 ∧ I % 2 = 1 ∧ J % 2 = 1 ∧
  G = H + 2 ∧ H = I + 2 ∧ I = J + 4 ∧
  J = 1 ∧
  A + B + C = 11 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J

theorem phone_number_A_value :
  ∀ A B C D E F G H I J : ℕ,
  phone_number A B C D E F G H I J →
  A = 8 := by
sorry

end phone_number_A_value_l763_76369


namespace num_true_props_l763_76315

-- Define the propositions as boolean variables
def prop1 : Bool := true  -- All lateral edges of a regular pyramid are equal
def prop2 : Bool := false -- The lateral faces of a right prism are all congruent rectangles
def prop3 : Bool := true  -- The generatrix of a cylinder is perpendicular to the base
def prop4 : Bool := true  -- The section obtained by cutting a cone with a plane passing through the axis of rotation is always a congruent isosceles triangle

-- Define a function to count true propositions
def countTrueProps (p1 p2 p3 p4 : Bool) : Nat :=
  (if p1 then 1 else 0) + (if p2 then 1 else 0) + (if p3 then 1 else 0) + (if p4 then 1 else 0)

-- Theorem stating that the number of true propositions is 3
theorem num_true_props : countTrueProps prop1 prop2 prop3 prop4 = 3 := by
  sorry

end num_true_props_l763_76315


namespace inequality_proof_l763_76358

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  a^2 / (b + c + d) + b^2 / (c + d + a) + c^2 / (d + a + b) + d^2 / (a + b + c) ≥ 2/3 := by
  sorry

end inequality_proof_l763_76358


namespace f_properties_l763_76353

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def decreasing_on_8_to_inf (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 8 → y > x → f y < f x

def f_plus_8_is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 8) = f (-x + 8)

-- State the theorem
theorem f_properties (h1 : decreasing_on_8_to_inf f) (h2 : f_plus_8_is_even f) :
  f 7 = f 9 ∧ f 7 > f 10 := by sorry

end f_properties_l763_76353


namespace total_study_time_is_three_l763_76361

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time Sam spends studying Science in minutes -/
def science_time : ℕ := 60

/-- The time Sam spends studying Math in minutes -/
def math_time : ℕ := 80

/-- The time Sam spends studying Literature in minutes -/
def literature_time : ℕ := 40

/-- The total time Sam spends studying in hours -/
def total_study_time : ℚ :=
  (science_time + math_time + literature_time) / minutes_per_hour

theorem total_study_time_is_three : total_study_time = 3 := by
  sorry

end total_study_time_is_three_l763_76361


namespace exists_unique_q_l763_76391

/-- Polynomial function g(x) -/
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p*x^4 + q*x^3 + r*x^2 + s*x + t

/-- Theorem stating the existence and uniqueness of q -/
theorem exists_unique_q :
  ∃! q : ℝ, ∃ p r s t : ℝ,
    g p q r s t 0 = 3 ∧
    g p q r s t (-2) = 0 ∧
    g p q r s t 1 = 0 ∧
    g p q r s t (-1) = -1 :=
by
  sorry

end exists_unique_q_l763_76391


namespace rectangular_area_equation_l763_76392

/-- Represents a rectangular area with length and width in meters -/
structure RectangularArea where
  length : ℝ
  width : ℝ

/-- The area of a rectangle is the product of its length and width -/
def area (r : RectangularArea) : ℝ := r.length * r.width

theorem rectangular_area_equation (x : ℝ) :
  let r : RectangularArea := { length := x, width := x - 6 }
  area r = 720 → x * (x - 6) = 720 := by
  sorry

end rectangular_area_equation_l763_76392


namespace intersection_k_value_l763_76378

-- Define the lines
def line1 (x y k : ℝ) : Prop := 2*x + 3*y - k = 0
def line2 (x y k : ℝ) : Prop := x - k*y + 12 = 0

-- Define the condition that the intersection point lies on the y-axis
def intersection_on_y_axis (k : ℝ) : Prop :=
  ∃ y : ℝ, line1 0 y k ∧ line2 0 y k

-- Theorem statement
theorem intersection_k_value :
  ∀ k : ℝ, intersection_on_y_axis k → (k = 6 ∨ k = -6) :=
by sorry

end intersection_k_value_l763_76378


namespace systematic_sampling_theorem_l763_76382

/-- Systematic sampling function that generates a sample number for a given group -/
def sampleNumber (x : ℕ) (k : ℕ) : ℕ :=
  (x + 33 * k) % 100 + 100 * k

/-- Generates the full sample of 10 numbers given an initial value x -/
def generateSample (x : ℕ) : List ℕ :=
  List.range 10 |>.map (sampleNumber x)

/-- Checks if a number ends with the digits 87 -/
def endsWith87 (n : ℕ) : Bool :=
  n % 100 = 87

/-- Set of possible x values that result in a sample number ending with 87 -/
def possibleXValues : Set ℕ :=
  {x | x ∈ Finset.range 100 ∧ ∃ k, k ∈ Finset.range 10 ∧ endsWith87 (sampleNumber x k)}

theorem systematic_sampling_theorem :
  (generateSample 24 = [24, 157, 290, 423, 556, 689, 822, 955, 88, 221]) ∧
  (possibleXValues = {21, 22, 23, 54, 55, 56, 87, 88, 89, 90}) := by
  sorry

end systematic_sampling_theorem_l763_76382


namespace lyndees_friends_l763_76322

theorem lyndees_friends (total_chicken : ℕ) (lyndee_ate : ℕ) (friend_ate : ℕ) 
  (h1 : total_chicken = 11)
  (h2 : lyndee_ate = 1)
  (h3 : friend_ate = 2)
  (h4 : total_chicken = lyndee_ate + friend_ate * (total_chicken - lyndee_ate) / friend_ate) :
  (total_chicken - lyndee_ate) / friend_ate = 5 := by
  sorry

end lyndees_friends_l763_76322


namespace student_arrangements_l763_76370

def num_students : ℕ := 6

-- Condition 1: A not at head, B not at tail
def condition1 (arrangements : ℕ) : Prop :=
  arrangements = 504

-- Condition 2: A, B, and C not adjacent
def condition2 (arrangements : ℕ) : Prop :=
  arrangements = 144

-- Condition 3: A and B adjacent, C and D adjacent
def condition3 (arrangements : ℕ) : Prop :=
  arrangements = 96

-- Condition 4: Neither A nor B adjacent to C
def condition4 (arrangements : ℕ) : Prop :=
  arrangements = 288

theorem student_arrangements :
  ∃ (arr1 arr2 arr3 arr4 : ℕ),
    condition1 arr1 ∧
    condition2 arr2 ∧
    condition3 arr3 ∧
    condition4 arr4 :=
  by sorry

end student_arrangements_l763_76370


namespace successive_discounts_equivalent_to_single_l763_76373

/-- Represents a discount as a fraction between 0 and 1 -/
def Discount := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- Apply a discount to a price -/
def applyDiscount (price : ℝ) (discount : Discount) : ℝ :=
  price * (1 - discount.val)

/-- Apply two successive discounts -/
def applySuccessiveDiscounts (price : ℝ) (d1 d2 : Discount) : ℝ :=
  applyDiscount (applyDiscount price d1) d2

theorem successive_discounts_equivalent_to_single (price : ℝ) :
  let d1 : Discount := ⟨0.1, by norm_num⟩
  let d2 : Discount := ⟨0.2, by norm_num⟩
  let singleDiscount : Discount := ⟨0.28, by norm_num⟩
  applySuccessiveDiscounts price d1 d2 = applyDiscount price singleDiscount := by
  sorry

end successive_discounts_equivalent_to_single_l763_76373


namespace seashell_collection_l763_76397

theorem seashell_collection (x y : ℝ) : 
  let initial := x
  let additional := y
  let total := initial + additional
  let after_jessica := (2/3) * total
  let after_henry := (3/4) * after_jessica
  after_henry = (1/2) * total
  := by sorry

end seashell_collection_l763_76397


namespace cannot_simplify_further_l763_76336

theorem cannot_simplify_further (x : ℝ) : 
  Real.sqrt (x^6 + x^4 + 1) = Real.sqrt (x^6 + x^4 + 1) := by sorry

end cannot_simplify_further_l763_76336


namespace harriets_age_l763_76314

/-- Given information about Peter and Harriet's ages, prove Harriet's current age -/
theorem harriets_age (peter_mother_age : ℕ) (peter_age : ℕ) (harriet_age : ℕ) : 
  peter_mother_age = 60 →
  peter_age = peter_mother_age / 2 →
  peter_age + 4 = 2 * (harriet_age + 4) →
  harriet_age = 13 := by sorry

end harriets_age_l763_76314


namespace problem_statement_l763_76384

theorem problem_statement (x y : ℝ) (h1 : x = 2 * y) (h2 : y ≠ 0) :
  (x + 2 * y) - (2 * x + y) = -y := by
  sorry

end problem_statement_l763_76384


namespace triangle_area_l763_76383

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 6 → 
  a = 2 * c → 
  B = π / 3 → 
  (1 / 2) * a * c * Real.sin B = 6 * Real.sqrt 3 :=
by sorry

end triangle_area_l763_76383


namespace device_marked_price_device_marked_price_is_59_l763_76342

theorem device_marked_price (initial_price : ℝ) (purchase_discount : ℝ) 
  (profit_percentage : ℝ) (sale_discount : ℝ) : ℝ :=
  let purchase_price := initial_price * (1 - purchase_discount)
  let selling_price := purchase_price * (1 + profit_percentage)
  selling_price / (1 - sale_discount)

theorem device_marked_price_is_59 :
  device_marked_price 50 0.15 0.25 0.10 = 59 := by
  sorry

end device_marked_price_device_marked_price_is_59_l763_76342


namespace isosceles_triangle_perimeter_l763_76386

-- Define the isosceles triangle
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  area : ℝ

-- Define the conditions of the problem
def triangle : IsoscelesTriangle :=
  { side1 := 6,
    side2 := 8,
    area := 12 }

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) 
  (h1 : t = triangle) : 
  (2 * t.side1 + t.side2 = 20) ∨ (2 * t.side2 + t.side1 = 20) :=
sorry

end isosceles_triangle_perimeter_l763_76386


namespace spinner_sections_l763_76377

theorem spinner_sections (n : ℕ) (n_pos : n > 0) : 
  (1 - 1 / n : ℚ) ^ 2 = 559 / 1000 → n = 4 := by
  sorry

end spinner_sections_l763_76377


namespace hexagonal_pyramid_height_l763_76389

/-- Represents a right hexagonal pyramid with three parallel cross sections. -/
structure HexagonalPyramid where
  /-- Height of the smallest cross section from the apex -/
  x : ℝ
  /-- Area of the smallest cross section -/
  area₁ : ℝ
  /-- Area of the middle cross section -/
  area₂ : ℝ
  /-- Area of the largest cross section -/
  area₃ : ℝ
  /-- The areas are in the correct ratio -/
  area_ratio₁ : area₁ / area₂ = 9 / 20
  /-- The areas are in the correct ratio -/
  area_ratio₂ : area₂ / area₃ = 5 / 9
  /-- The heights are in arithmetic progression -/
  height_progression : x + 20 - (x + 10) = (x + 10) - x

/-- The height of the smallest cross section from the apex in a right hexagonal pyramid
    with specific cross-sectional areas at specific heights. -/
theorem hexagonal_pyramid_height (p : HexagonalPyramid) : p.x = 100 / (10 - 3 * Real.sqrt 5) := by
  sorry

end hexagonal_pyramid_height_l763_76389


namespace quadratic_convergence_l763_76313

-- Define the quadratic function
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the property that |f(x)| ≤ 1/2 for all x in [3, 5]
def bounded_on_interval (p q : ℝ) : Prop :=
  ∀ x : ℝ, 3 ≤ x ∧ x ≤ 5 → |f p q x| ≤ 1/2

-- Define the repeated application of f
def f_iterate (p q : ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n+1 => f p q (f_iterate p q n x)

-- State the theorem
theorem quadratic_convergence (p q : ℝ) (h : bounded_on_interval p q) :
  f_iterate p q 2017 ((7 + Real.sqrt 15) / 2) = (7 - Real.sqrt 15) / 2 :=
sorry

end quadratic_convergence_l763_76313


namespace sin_450_degrees_l763_76349

theorem sin_450_degrees : Real.sin (450 * π / 180) = 1 := by sorry

end sin_450_degrees_l763_76349


namespace complex_equation_sum_l763_76330

theorem complex_equation_sum (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + 2 * i) / i = b + i →
  a + b = 1 :=
by sorry

end complex_equation_sum_l763_76330


namespace carlton_outfits_l763_76374

/-- Represents Carlton's wardrobe and outfit combinations -/
structure Wardrobe where
  button_up_shirts : ℕ
  sweater_vests : ℕ
  outfits : ℕ

/-- Calculates the number of outfits for Carlton -/
def calculate_outfits (w : Wardrobe) : Prop :=
  w.button_up_shirts = 3 ∧
  w.sweater_vests = 2 * w.button_up_shirts ∧
  w.outfits = w.button_up_shirts * w.sweater_vests

/-- Theorem stating that Carlton has 18 outfits -/
theorem carlton_outfits :
  ∃ w : Wardrobe, calculate_outfits w ∧ w.outfits = 18 := by
  sorry


end carlton_outfits_l763_76374


namespace multiples_of_seven_between_20_and_150_l763_76305

theorem multiples_of_seven_between_20_and_150 : 
  (Finset.filter (fun n => n % 7 = 0 ∧ n > 20 ∧ n < 150) (Finset.range 150)).card = 19 := by
sorry

end multiples_of_seven_between_20_and_150_l763_76305


namespace min_sum_triangular_grid_l763_76364

/-- Represents a triangular grid with 16 cells --/
structure TriangularGrid :=
  (cells : Fin 16 → ℕ)

/-- Checks if a number is prime --/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Represents the layers of the triangular grid --/
def layers : List (List (Fin 16)) := sorry

/-- The sum of numbers in each layer is prime --/
def layerSumsPrime (grid : TriangularGrid) : Prop :=
  ∀ layer ∈ layers, isPrime (layer.map grid.cells).sum

/-- The theorem stating the minimum sum of all numbers in the grid --/
theorem min_sum_triangular_grid :
  ∀ grid : TriangularGrid, layerSumsPrime grid →
  (Finset.univ.sum (λ i => grid.cells i) ≥ 22) :=
sorry

end min_sum_triangular_grid_l763_76364


namespace altitude_equation_l763_76316

/-- Given a triangle ABC with side equations:
    AB: 3x + 4y + 12 = 0
    BC: 4x - 3y + 16 = 0
    CA: 2x + y - 2 = 0
    The altitude from A to BC has the equation x - 2y + 4 = 0 -/
theorem altitude_equation (x y : ℝ) :
  (3 * x + 4 * y + 12 = 0) →  -- AB
  (4 * x - 3 * y + 16 = 0) →  -- BC
  (2 * x + y - 2 = 0) →       -- CA
  (x - 2 * y + 4 = 0)         -- Altitude from A to BC
:= by sorry

end altitude_equation_l763_76316
