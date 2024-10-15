import Mathlib

namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l2633_263371

theorem cubic_sum_over_product (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x^3 + 1/(y+2016) = y^3 + 1/(z+2016) ∧ 
  y^3 + 1/(z+2016) = z^3 + 1/(x+2016) → 
  (x^3 + y^3 + z^3) / (x*y*z) = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l2633_263371


namespace NUMINAMATH_CALUDE_box_third_side_l2633_263308

/-- A rectangular box with known properties -/
structure Box where
  cubes : ℕ  -- Number of cubes that fit in the box
  cube_volume : ℕ  -- Volume of each cube in cubic centimetres
  side1 : ℕ  -- Length of first known side in centimetres
  side2 : ℕ  -- Length of second known side in centimetres

/-- The length of the third side of the box -/
def third_side (b : Box) : ℚ :=
  (b.cubes * b.cube_volume : ℚ) / (b.side1 * b.side2)

/-- Theorem stating that the third side of the given box is 6 centimetres -/
theorem box_third_side :
  let b : Box := { cubes := 24, cube_volume := 27, side1 := 9, side2 := 12 }
  third_side b = 6 := by sorry

end NUMINAMATH_CALUDE_box_third_side_l2633_263308


namespace NUMINAMATH_CALUDE_quagga_placements_l2633_263343

/-- Represents a chessboard --/
def Chessboard := Fin 8 × Fin 8

/-- Represents a quagga's move --/
def QuaggaMove := (Int × Int) × (Int × Int)

/-- Defines the valid moves for a quagga --/
def validQuaggaMoves : List QuaggaMove :=
  [(( 6,  0), ( 0,  5)), (( 6,  0), ( 0, -5)),
   ((-6,  0), ( 0,  5)), ((-6,  0), ( 0, -5)),
   (( 0,  6), ( 5,  0)), (( 0,  6), (-5,  0)),
   (( 0, -6), ( 5,  0)), (( 0, -6), (-5,  0))]

/-- Checks if a move is valid on the chessboard --/
def isValidMove (start : Chessboard) (move : QuaggaMove) : Bool :=
  let ((dx1, dy1), (dx2, dy2)) := move
  let (x, y) := start
  let x1 := x + dx1
  let y1 := y + dy1
  let x2 := x1 + dx2
  let y2 := y1 + dy2
  0 ≤ x2 ∧ x2 < 8 ∧ 0 ≤ y2 ∧ y2 < 8

/-- Represents a placement of quaggas on the chessboard --/
def QuaggaPlacement := List Chessboard

/-- Checks if a placement is valid (no quaggas attack each other) --/
def isValidPlacement (placement : QuaggaPlacement) : Bool :=
  sorry

/-- The main theorem to prove --/
theorem quagga_placements :
  (∃ (placements : List QuaggaPlacement),
    placements.length = 68 ∧
    ∀ p ∈ placements,
      p.length = 51 ∧
      isValidPlacement p) :=
sorry

end NUMINAMATH_CALUDE_quagga_placements_l2633_263343


namespace NUMINAMATH_CALUDE_village_population_percentage_l2633_263302

theorem village_population_percentage : 
  let total_population : ℕ := 25600
  let part_population : ℕ := 23040
  (part_population : ℚ) / total_population * 100 = 90 := by sorry

end NUMINAMATH_CALUDE_village_population_percentage_l2633_263302


namespace NUMINAMATH_CALUDE_power_sum_difference_l2633_263331

theorem power_sum_difference : 2^4 + 2^4 + 2^4 - 2^2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l2633_263331


namespace NUMINAMATH_CALUDE_ride_time_is_36_seconds_l2633_263324

/-- Represents the escalator problem with given conditions -/
structure EscalatorProblem where
  theo_walk_time_non_operating : ℝ
  theo_walk_time_operating : ℝ
  escalator_efficiency : ℝ
  theo_walk_time_non_operating_eq : theo_walk_time_non_operating = 80
  theo_walk_time_operating_eq : theo_walk_time_operating = 30
  escalator_efficiency_eq : escalator_efficiency = 0.75

/-- Calculates the time it takes Theo to ride down the operating escalator while standing still -/
def ride_time (problem : EscalatorProblem) : ℝ :=
  problem.theo_walk_time_non_operating * problem.escalator_efficiency

/-- Theorem stating that the ride time for Theo is 36 seconds -/
theorem ride_time_is_36_seconds (problem : EscalatorProblem) :
  ride_time problem = 36 := by
  sorry

#eval ride_time { theo_walk_time_non_operating := 80,
                  theo_walk_time_operating := 30,
                  escalator_efficiency := 0.75,
                  theo_walk_time_non_operating_eq := rfl,
                  theo_walk_time_operating_eq := rfl,
                  escalator_efficiency_eq := rfl }

end NUMINAMATH_CALUDE_ride_time_is_36_seconds_l2633_263324


namespace NUMINAMATH_CALUDE_daps_dops_dips_equivalence_l2633_263389

/-- Given that 5 daps are equivalent to 4 dops and 3 dops are equivalent to 11 dips,
    prove that 22.5 daps are equivalent to 66 dips. -/
theorem daps_dops_dips_equivalence 
  (h1 : (5 : ℚ) / 4 = daps_per_dop) 
  (h2 : (3 : ℚ) / 11 = dops_per_dip) : 
  (66 : ℚ) * daps_per_dop * dops_per_dip = (45 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_daps_dops_dips_equivalence_l2633_263389


namespace NUMINAMATH_CALUDE_distance_from_center_to_point_l2633_263340

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 6*x - 8*y + 18

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, -4)

-- Define the point
def point : ℝ × ℝ := (3, -2)

-- Theorem statement
theorem distance_from_center_to_point : 
  let (cx, cy) := circle_center
  let (px, py) := point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_from_center_to_point_l2633_263340


namespace NUMINAMATH_CALUDE_ellipse_b_value_l2633_263334

/-- The value of b for an ellipse with given properties -/
theorem ellipse_b_value (b : ℝ) (h1 : 0 < b) (h2 : b < 3) :
  (∀ x y : ℝ, x^2 / 9 + y^2 / b^2 = 1 →
    ∃ F₁ F₂ : ℝ × ℝ, 
      (F₁.1 < 0 ∧ F₂.1 > 0) ∧ 
      (∀ A B : ℝ × ℝ, 
        (A.1^2 / 9 + A.2^2 / b^2 = 1) ∧ 
        (B.1^2 / 9 + B.2^2 / b^2 = 1) ∧
        (∃ k : ℝ, A.2 = k * (A.1 - F₁.1) ∧ B.2 = k * (B.1 - F₁.1)) →
        (dist A F₁ + dist A F₂ = 6) ∧ 
        (dist B F₁ + dist B F₂ = 6) ∧
        (dist B F₂ + dist A F₂ ≤ 10))) →
  b = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_b_value_l2633_263334


namespace NUMINAMATH_CALUDE_marathon_average_time_l2633_263322

theorem marathon_average_time (casey_time : ℝ) (zendaya_factor : ℝ) : 
  casey_time = 6 → 
  zendaya_factor = 1/3 → 
  let zendaya_time := casey_time + zendaya_factor * casey_time
  (casey_time + zendaya_time) / 2 = 7 := by
sorry

end NUMINAMATH_CALUDE_marathon_average_time_l2633_263322


namespace NUMINAMATH_CALUDE_exists_special_function_l2633_263351

/-- The number of divisors of a natural number -/
def number_of_divisors (n : ℕ) : ℕ := sorry

/-- The existence of a function with specific properties -/
theorem exists_special_function : 
  ∃ f : ℕ → ℕ, 
    (∃ n : ℕ, f n ≠ n) ∧ 
    (∀ m n : ℕ, (number_of_divisors m = f n) ↔ (number_of_divisors (f m) = n)) :=
sorry

end NUMINAMATH_CALUDE_exists_special_function_l2633_263351


namespace NUMINAMATH_CALUDE_pascal_triangle_51st_row_third_number_l2633_263378

theorem pascal_triangle_51st_row_third_number : 
  let n : ℕ := 51
  let k : ℕ := 2
  Nat.choose n k = 1275 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_51st_row_third_number_l2633_263378


namespace NUMINAMATH_CALUDE_system_solution_l2633_263385

theorem system_solution (x y z u : ℝ) 
  (eq1 : x + y = 4)
  (eq2 : x * z + y * u = 7)
  (eq3 : x * z^2 + y * u^2 = 12)
  (eq4 : x * z^3 + y * u^3 = 21) :
  z = 7/3 ∧ y = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2633_263385


namespace NUMINAMATH_CALUDE_marbles_lost_vs_found_l2633_263317

theorem marbles_lost_vs_found (initial : ℕ) (lost : ℕ) (found : ℕ) : 
  initial = 4 → lost = 16 → found = 8 → lost - found = 8 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_vs_found_l2633_263317


namespace NUMINAMATH_CALUDE_expression_factorization_l2633_263360

theorem expression_factorization (x : ℝ) :
  (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10) = 5 * x * (5 * x^2 + 19) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2633_263360


namespace NUMINAMATH_CALUDE_coltons_remaining_stickers_l2633_263330

/-- The number of stickers Colton has left after giving some to his friends. -/
def stickers_left (initial : ℕ) (per_friend : ℕ) (num_friends : ℕ) (mandy_extra : ℕ) (justin_less : ℕ) : ℕ :=
  let friends_total := per_friend * num_friends
  let mandy_stickers := friends_total + mandy_extra
  let justin_stickers := mandy_stickers - justin_less
  initial - (friends_total + mandy_stickers + justin_stickers)

/-- Theorem stating that Colton has 42 stickers left given the problem conditions. -/
theorem coltons_remaining_stickers :
  stickers_left 72 4 3 2 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_coltons_remaining_stickers_l2633_263330


namespace NUMINAMATH_CALUDE_hotel_room_number_contradiction_l2633_263365

theorem hotel_room_number_contradiction : 
  ¬ ∃ (a b c : ℕ), 
    0 < a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    100 * a + 10 * b + c = (a + 1) * (b + 1) * c :=
by sorry

end NUMINAMATH_CALUDE_hotel_room_number_contradiction_l2633_263365


namespace NUMINAMATH_CALUDE_walking_distance_l2633_263307

theorem walking_distance (speed1 speed2 time_diff : ℝ) (h1 : speed1 = 4)
  (h2 : speed2 = 3) (h3 : time_diff = 1/2) :
  let distance := speed1 * (time_diff + distance / speed2)
  distance = 6 := by sorry

end NUMINAMATH_CALUDE_walking_distance_l2633_263307


namespace NUMINAMATH_CALUDE_center_numbers_l2633_263323

def numbers : List ℕ := [9, 12, 18, 24, 36, 48, 96]

def is_valid_center (x : ℕ) (nums : List ℕ) : Prop :=
  x ∈ nums ∧
  ∃ (a b c d e f : ℕ),
    a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ d ∈ nums ∧ e ∈ nums ∧ f ∈ nums ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
    a * x * d = b * x * e ∧ b * x * e = c * x * f

theorem center_numbers :
  ∀ x ∈ numbers, is_valid_center x numbers ↔ x = 12 ∨ x = 96 := by
  sorry

end NUMINAMATH_CALUDE_center_numbers_l2633_263323


namespace NUMINAMATH_CALUDE_florist_roses_count_l2633_263377

/-- Calculates the final number of roses a florist has after selling and picking more. -/
def final_roses (initial : ℕ) (sold : ℕ) (picked : ℕ) : ℕ :=
  initial - sold + picked

/-- Proves that given the initial conditions, the florist ends up with 56 roses. -/
theorem florist_roses_count : final_roses 50 15 21 = 56 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_count_l2633_263377


namespace NUMINAMATH_CALUDE_sector_central_angle_l2633_263380

-- Define the sector
structure Sector where
  perimeter : ℝ
  area : ℝ

-- Theorem statement
theorem sector_central_angle (s : Sector) (h1 : s.perimeter = 6) (h2 : s.area = 2) :
  ∃ θ : ℝ, (θ = 1 ∨ θ = 4) ∧ 
  (∃ r : ℝ, r > 0 ∧ θ * r + 2 * r = s.perimeter ∧ 1/2 * r^2 * θ = s.area) :=
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2633_263380


namespace NUMINAMATH_CALUDE_canister_capacity_ratio_l2633_263352

/-- Represents the ratio of capacities between two canisters -/
structure CanisterRatio where
  c : ℝ  -- Capacity of canister C
  d : ℝ  -- Capacity of canister D

/-- Theorem stating the ratio of canister capacities given the problem conditions -/
theorem canister_capacity_ratio (r : CanisterRatio) 
  (hc_half : r.c / 2 = r.c - (r.d / 3 - r.d / 12)) 
  (hd_third : r.d / 3 > 0) 
  (hc_positive : r.c > 0) 
  (hd_positive : r.d > 0) :
  r.d / r.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_canister_capacity_ratio_l2633_263352


namespace NUMINAMATH_CALUDE_ernies_income_ratio_l2633_263370

def ernies_previous_income : ℕ := 6000
def jacks_income : ℕ := 2 * ernies_previous_income
def combined_income : ℕ := 16800
def ernies_current_income : ℕ := combined_income - jacks_income

theorem ernies_income_ratio :
  (ernies_current_income : ℚ) / ernies_previous_income = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ernies_income_ratio_l2633_263370


namespace NUMINAMATH_CALUDE_carlos_class_size_l2633_263388

theorem carlos_class_size (n : ℕ) (carlos : ℕ) :
  (carlos = 75) →
  (n - carlos = 74) →
  (carlos - 1 = 74) →
  n = 149 := by
  sorry

end NUMINAMATH_CALUDE_carlos_class_size_l2633_263388


namespace NUMINAMATH_CALUDE_sin_75_cos_15_minus_1_l2633_263369

theorem sin_75_cos_15_minus_1 : 
  2 * Real.sin (75 * π / 180) * Real.cos (15 * π / 180) - 1 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_cos_15_minus_1_l2633_263369


namespace NUMINAMATH_CALUDE_total_crayons_count_l2633_263368

/-- The number of children -/
def num_children : ℕ := 7

/-- The number of crayons each child has -/
def crayons_per_child : ℕ := 8

/-- The total number of crayons -/
def total_crayons : ℕ := num_children * crayons_per_child

theorem total_crayons_count : total_crayons = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_count_l2633_263368


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2633_263342

theorem sum_of_coefficients (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^10 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                           a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 1023 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2633_263342


namespace NUMINAMATH_CALUDE_consecutive_integers_product_210_l2633_263392

/-- Given three consecutive integers whose product is 210 and whose sum of squares is minimized,
    the sum of the two smallest of these integers is 11. -/
theorem consecutive_integers_product_210 (n : ℤ) :
  (n - 1) * n * (n + 1) = 210 ∧
  ∀ m : ℤ, (m - 1) * m * (m + 1) = 210 → 
    (n - 1)^2 + n^2 + (n + 1)^2 ≤ (m - 1)^2 + m^2 + (m + 1)^2 →
  (n - 1) + n = 11 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_210_l2633_263392


namespace NUMINAMATH_CALUDE_equipment_cost_proof_l2633_263326

/-- The number of players on the team -/
def num_players : ℕ := 16

/-- The cost of a jersey in dollars -/
def jersey_cost : ℚ := 25

/-- The cost of shorts in dollars -/
def shorts_cost : ℚ := 152/10

/-- The cost of socks in dollars -/
def socks_cost : ℚ := 68/10

/-- The total cost of equipment for all players on the team -/
def total_cost : ℚ := num_players * (jersey_cost + shorts_cost + socks_cost)

theorem equipment_cost_proof : total_cost = 752 := by
  sorry

end NUMINAMATH_CALUDE_equipment_cost_proof_l2633_263326


namespace NUMINAMATH_CALUDE_combined_salaries_l2633_263383

/-- Given the salary of A and the average salary of A, B, C, D, and E,
    prove the combined salaries of B, C, D, and E. -/
theorem combined_salaries
  (salary_A : ℕ)
  (average_salary : ℕ)
  (h1 : salary_A = 10000)
  (h2 : average_salary = 8400) :
  salary_A + (4 * ((5 * average_salary) - salary_A)) = 42000 :=
by sorry

end NUMINAMATH_CALUDE_combined_salaries_l2633_263383


namespace NUMINAMATH_CALUDE_binomial_expansion_ratio_l2633_263335

theorem binomial_expansion_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ) : 
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃ + a₅) = -122 / 121 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_ratio_l2633_263335


namespace NUMINAMATH_CALUDE_quadratic_polynomial_determination_l2633_263364

/-- A type representing quadratic polynomials -/
def QuadraticPolynomial := ℝ → ℝ

/-- A function that evaluates a quadratic polynomial at a given point -/
def evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ := p x

/-- A function that checks if two polynomials agree at a given point -/
def agree (p q : QuadraticPolynomial) (x : ℝ) : Prop := evaluate p x = evaluate q x

theorem quadratic_polynomial_determination (n : ℕ) (h : n > 1) :
  ∃ (C : ℝ), C > 0 ∧
  ∀ (polynomials : Finset QuadraticPolynomial),
  polynomials.card = n →
  ∃ (points : Finset ℝ),
  points.card = 2 * n^2 + 1 ∧
  ∃ (p : QuadraticPolynomial),
  p ∈ polynomials ∧
  ∀ (q : QuadraticPolynomial),
  (∀ (x : ℝ), x ∈ points → agree p q x) → p = q :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_determination_l2633_263364


namespace NUMINAMATH_CALUDE_x_minus_25_is_perfect_square_l2633_263379

/-- Represents the number of zeros after the first 1 in the definition of x -/
def zeros_after_first_one : ℕ := 2011

/-- Represents the number of zeros after the second 1 in the definition of x -/
def zeros_after_second_one : ℕ := 2012

/-- Defines x as described in the problem -/
def x : ℕ := 
  10^(zeros_after_second_one + 3) + 
  10^(zeros_after_first_one + zeros_after_second_one + 2) + 
  50

/-- States that x - 25 is a perfect square -/
theorem x_minus_25_is_perfect_square : 
  ∃ n : ℕ, x - 25 = n^2 := by sorry

end NUMINAMATH_CALUDE_x_minus_25_is_perfect_square_l2633_263379


namespace NUMINAMATH_CALUDE_emani_money_l2633_263366

/-- Proves that Emani has $150, given the conditions of the problem -/
theorem emani_money :
  (∀ (emani howard : ℕ),
    emani = howard + 30 →
    emani + howard = 2 * 135 →
    emani = 150) :=
by sorry

end NUMINAMATH_CALUDE_emani_money_l2633_263366


namespace NUMINAMATH_CALUDE_lattice_point_bounds_l2633_263303

/-- The minimum number of points in ℤ^d such that any set of these points
    will contain n points whose centroid is a lattice point -/
def f (n d : ℕ) : ℕ :=
  sorry

theorem lattice_point_bounds (n d : ℕ) (hn : n > 0) (hd : d > 0) :
  (n - 1) * 2^d + 1 ≤ f n d ∧ f n d ≤ (n - 1) * n^d + 1 :=
by sorry

end NUMINAMATH_CALUDE_lattice_point_bounds_l2633_263303


namespace NUMINAMATH_CALUDE_complex_power_result_l2633_263337

theorem complex_power_result : (3 * (Complex.cos (30 * Real.pi / 180)) + 3 * Complex.I * (Complex.sin (30 * Real.pi / 180)))^4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_result_l2633_263337


namespace NUMINAMATH_CALUDE_xy_less_than_one_necessary_not_sufficient_l2633_263344

theorem xy_less_than_one_necessary_not_sufficient (x y : ℝ) :
  (0 < x ∧ x < 1/y) → (x*y < 1) ∧
  ¬(∀ x y : ℝ, x*y < 1 → (0 < x ∧ x < 1/y)) :=
by sorry

end NUMINAMATH_CALUDE_xy_less_than_one_necessary_not_sufficient_l2633_263344


namespace NUMINAMATH_CALUDE_gcd_91_72_l2633_263329

theorem gcd_91_72 : Nat.gcd 91 72 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_91_72_l2633_263329


namespace NUMINAMATH_CALUDE_range_of_a_l2633_263382

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x| ≤ 4) → -4 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2633_263382


namespace NUMINAMATH_CALUDE_height_difference_l2633_263306

/-- Given three heights in ratio 4 : 5 : 6 with the shortest being 120 cm, 
    prove that the sum of shortest and tallest minus the middle equals 150 cm -/
theorem height_difference (h₁ h₂ h₃ : ℝ) : 
  h₁ / h₂ = 4 / 5 → 
  h₂ / h₃ = 5 / 6 → 
  h₁ = 120 → 
  h₁ + h₃ - h₂ = 150 := by
sorry

end NUMINAMATH_CALUDE_height_difference_l2633_263306


namespace NUMINAMATH_CALUDE_roots_of_equation_l2633_263333

theorem roots_of_equation : 
  let f : ℝ → ℝ := fun y ↦ (2 * y + 1) * (2 * y - 3)
  ∃ y₁ y₂ : ℝ, y₁ = -1/2 ∧ y₂ = 3/2 ∧ (∀ y : ℝ, f y = 0 ↔ y = y₁ ∨ y = y₂) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2633_263333


namespace NUMINAMATH_CALUDE_cable_length_l2633_263313

/-- The length of a curve defined by the intersection of a sphere and a plane --/
theorem cable_length (x y z : ℝ) : 
  x + y + z = 10 →
  x * y + y * z + z * x = 18 →
  ∃ (curve_length : ℝ), curve_length = 4 * Real.pi * Real.sqrt (23 / 3) :=
by sorry

end NUMINAMATH_CALUDE_cable_length_l2633_263313


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2633_263375

/-- A parabola with vertex at the origin and focus on the x-axis. -/
structure Parabola where
  /-- The x-coordinate of the focus -/
  p : ℝ
  /-- The parabola passes through this point -/
  point : ℝ × ℝ

/-- The distance from the focus to the directrix for a parabola -/
def focusDirectrixDistance (c : Parabola) : ℝ :=
  c.p

theorem parabola_focus_directrix_distance 
  (c : Parabola) 
  (h1 : c.point = (1, 3)) : 
  focusDirectrixDistance c = 9/2 := by
  sorry

#check parabola_focus_directrix_distance

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2633_263375


namespace NUMINAMATH_CALUDE_robin_gum_total_l2633_263321

/-- Calculate the total number of gum pieces Robin has after his purchases -/
theorem robin_gum_total (initial_packages : ℕ) (initial_pieces_per_package : ℕ)
  (local_packages : ℚ) (local_pieces_per_package : ℕ)
  (foreign_packages : ℕ) (foreign_pieces_per_package : ℕ)
  (exchange_rate : ℚ) (foreign_purchase_dollars : ℕ) :
  initial_packages = 27 →
  initial_pieces_per_package = 18 →
  local_packages = 15.5 →
  local_pieces_per_package = 12 →
  foreign_packages = 8 →
  foreign_pieces_per_package = 25 →
  exchange_rate = 1.2 →
  foreign_purchase_dollars = 50 →
  (initial_packages * initial_pieces_per_package +
   ⌊local_packages⌋ * local_pieces_per_package +
   foreign_packages * foreign_pieces_per_package) = 872 := by
  sorry

#check robin_gum_total

end NUMINAMATH_CALUDE_robin_gum_total_l2633_263321


namespace NUMINAMATH_CALUDE_addition_problem_l2633_263353

theorem addition_problem : ∃ x : ℝ, 37 + x = 52 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_addition_problem_l2633_263353


namespace NUMINAMATH_CALUDE_bathroom_tiling_savings_janet_bathroom_savings_l2633_263345

/-- Calculates the savings when choosing the least expensive tiles over the most expensive ones for a bathroom tiling project. -/
theorem bathroom_tiling_savings (wall1_length wall1_width wall2_length wall2_width wall3_length wall3_width : ℕ)
  (tiles_per_sqft : ℕ) (cheap_tile_cost expensive_tile_cost : ℚ) : ℚ :=
  let total_area := wall1_length * wall1_width + wall2_length * wall2_width + wall3_length * wall3_width
  let total_tiles := total_area * tiles_per_sqft
  let expensive_total := total_tiles * expensive_tile_cost
  let cheap_total := total_tiles * cheap_tile_cost
  expensive_total - cheap_total

/-- The savings for Janet's specific bathroom tiling project is $2,400. -/
theorem janet_bathroom_savings : 
  bathroom_tiling_savings 5 8 7 8 6 9 4 11 15 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_tiling_savings_janet_bathroom_savings_l2633_263345


namespace NUMINAMATH_CALUDE_solution_set_equality_l2633_263357

/-- The set of real numbers a for which the solution set of |x - 2| < a is a subset of (-2, 1] -/
def A : Set ℝ := {a | ∀ x, |x - 2| < a → -2 < x ∧ x ≤ 1}

/-- The theorem stating that A is equal to (-∞, 0] -/
theorem solution_set_equality : A = Set.Iic 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l2633_263357


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2633_263300

theorem complex_fraction_sum (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -2) (hb : b ≠ -2) (hc : c ≠ -2) (hd : d ≠ -2)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2633_263300


namespace NUMINAMATH_CALUDE_expected_pairs_value_l2633_263381

/-- The number of boys in the lineup -/
def num_boys : ℕ := 9

/-- The number of girls in the lineup -/
def num_girls : ℕ := 15

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of adjacent pairs in the lineup -/
def num_pairs : ℕ := total_people - 1

/-- The probability of a boy-girl or girl-boy pair at any given adjacent position -/
def pair_probability : ℚ := 
  (num_boys * num_girls + num_girls * num_boys) / (total_people * (total_people - 1))

/-- The expected number of boy-girl or girl-boy pairs in a random permutation -/
def expected_pairs : ℚ := num_pairs * pair_probability

theorem expected_pairs_value : expected_pairs = 3105 / 276 := by sorry

end NUMINAMATH_CALUDE_expected_pairs_value_l2633_263381


namespace NUMINAMATH_CALUDE_sum_y_invariant_under_rotation_l2633_263396

/-- A rectangle in 2D space -/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  is_opposite : v1 ≠ v2

/-- The sum of y-coordinates of two points -/
def sum_y (p1 p2 : ℝ × ℝ) : ℝ := p1.2 + p2.2

/-- Theorem: The sum of y-coordinates of the other two vertices of a rectangle
    remains unchanged after a 90-degree rotation around its center -/
theorem sum_y_invariant_under_rotation (r : Rectangle) 
    (h1 : r.v1 = (5, 20))
    (h2 : r.v2 = (11, -8)) :
    ∃ (v3 v4 : ℝ × ℝ), sum_y v3 v4 = 12 ∧ 
    (∀ (v3' v4' : ℝ × ℝ), sum_y v3' v4' = 12) :=
  sorry

#check sum_y_invariant_under_rotation

end NUMINAMATH_CALUDE_sum_y_invariant_under_rotation_l2633_263396


namespace NUMINAMATH_CALUDE_sibling_ages_sum_l2633_263301

theorem sibling_ages_sum (a b : ℕ+) : 
  a < b → 
  a * b * b * b = 216 → 
  a + b + b + b = 19 := by
sorry

end NUMINAMATH_CALUDE_sibling_ages_sum_l2633_263301


namespace NUMINAMATH_CALUDE_tan_difference_pi_12_5pi_12_l2633_263314

theorem tan_difference_pi_12_5pi_12 : 
  Real.tan (π / 12) - Real.tan (5 * π / 12) = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_pi_12_5pi_12_l2633_263314


namespace NUMINAMATH_CALUDE_orange_harvest_calculation_l2633_263339

/-- The number of sacks of oranges harvested per day -/
def sacks_per_day : ℕ := 38

/-- The number of days of harvest -/
def harvest_days : ℕ := 49

/-- The total number of sacks harvested after the given number of days -/
def total_sacks : ℕ := sacks_per_day * harvest_days

theorem orange_harvest_calculation :
  total_sacks = 1862 := by sorry

end NUMINAMATH_CALUDE_orange_harvest_calculation_l2633_263339


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l2633_263397

/-- The amount of money Chris had before his birthday -/
def money_before : ℕ := sorry

/-- The amount Chris received from his aunt and uncle -/
def aunt_uncle_gift : ℕ := 20

/-- The amount Chris received from his parents -/
def parents_gift : ℕ := 75

/-- The amount Chris received from his grandmother -/
def grandmother_gift : ℕ := 25

/-- The total amount Chris had after his birthday -/
def total_after : ℕ := 279

/-- Theorem stating that Chris had $159 before his birthday -/
theorem chris_money_before_birthday :
  money_before + aunt_uncle_gift + parents_gift + grandmother_gift = total_after ∧
  money_before = 159 := by sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l2633_263397


namespace NUMINAMATH_CALUDE_matrix_equation_satisfied_l2633_263312

/-- The matrix M that satisfies the given equation -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]

/-- The right-hand side matrix of the equation -/
def RHS : Matrix (Fin 2) (Fin 2) ℝ := !![10, 20; 5, 10]

/-- Theorem stating that M satisfies the given matrix equation -/
theorem matrix_equation_satisfied :
  M^3 - 4 • M^2 + 5 • M = RHS := by sorry

end NUMINAMATH_CALUDE_matrix_equation_satisfied_l2633_263312


namespace NUMINAMATH_CALUDE_cement_truck_loads_l2633_263358

theorem cement_truck_loads (total material_truck_loads sand_truck_loads dirt_truck_loads : ℚ)
  (h1 : total = 0.67)
  (h2 : sand_truck_loads = 0.17)
  (h3 : dirt_truck_loads = 0.33)
  : total - (sand_truck_loads + dirt_truck_loads) = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_cement_truck_loads_l2633_263358


namespace NUMINAMATH_CALUDE_A_intersect_B_is_empty_l2633_263315

-- Define set A
def A : Set ℝ := {x : ℝ | |x| ≥ 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - x - 2 < 0}

-- Theorem statement
theorem A_intersect_B_is_empty : A ∩ B = ∅ := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_is_empty_l2633_263315


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_fifth_powers_l2633_263362

theorem divisibility_of_sum_of_fifth_powers (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (y - z) * (z - x) * (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_fifth_powers_l2633_263362


namespace NUMINAMATH_CALUDE_product_of_one_plus_tangents_17_and_28_l2633_263384

theorem product_of_one_plus_tangents_17_and_28 :
  (1 + Real.tan (17 * π / 180)) * (1 + Real.tan (28 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_tangents_17_and_28_l2633_263384


namespace NUMINAMATH_CALUDE_cranberry_juice_price_per_ounce_l2633_263367

/-- Given a can of cranberry juice with volume in ounces and price in cents,
    calculate the price per ounce in cents. -/
def price_per_ounce (volume : ℕ) (price : ℕ) : ℚ :=
  price / volume

/-- Theorem stating that the price per ounce of cranberry juice is 7 cents
    given that a 12 ounce can sells for 84 cents. -/
theorem cranberry_juice_price_per_ounce :
  price_per_ounce 12 84 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cranberry_juice_price_per_ounce_l2633_263367


namespace NUMINAMATH_CALUDE_talia_father_age_l2633_263356

/-- Represents the ages of Talia and her parents -/
structure FamilyAges where
  talia : ℕ
  mom : ℕ
  dad : ℕ

/-- Conditions given in the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.talia + 7 = 20 ∧
  ages.mom = 3 * ages.talia ∧
  ages.dad + 3 = ages.mom

/-- Theorem stating that Talia's father is 36 years old -/
theorem talia_father_age (ages : FamilyAges) :
  problem_conditions ages → ages.dad = 36 := by
  sorry


end NUMINAMATH_CALUDE_talia_father_age_l2633_263356


namespace NUMINAMATH_CALUDE_f_properties_l2633_263336

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - x + 1

theorem f_properties :
  (∀ a : ℝ, (∀ x > 0, x * (Real.log x + 1/x) ≤ x^2 + a*x + 1) ↔ a ≥ -1) ∧
  (∀ x > 0, (x - 1) * f x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2633_263336


namespace NUMINAMATH_CALUDE_john_walks_farther_l2633_263373

/-- John's walking distance to school in miles -/
def john_distance : ℝ := 1.74

/-- Nina's walking distance to school in miles -/
def nina_distance : ℝ := 1.235

/-- The difference between John's and Nina's walking distances -/
def distance_difference : ℝ := john_distance - nina_distance

theorem john_walks_farther : distance_difference = 0.505 := by sorry

end NUMINAMATH_CALUDE_john_walks_farther_l2633_263373


namespace NUMINAMATH_CALUDE_integer_decimal_parts_sum_l2633_263304

theorem integer_decimal_parts_sum (m n : ℝ) : 
  (∃ k : ℤ, 7 + Real.sqrt 13 = k + m ∧ k ≤ 7 + Real.sqrt 13 ∧ 7 + Real.sqrt 13 < k + 1) →
  (∃ j : ℤ, Real.sqrt 13 = j + n ∧ j ≤ Real.sqrt 13 ∧ Real.sqrt 13 < j + 1) →
  m + n = 7 + Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_integer_decimal_parts_sum_l2633_263304


namespace NUMINAMATH_CALUDE_eva_marks_ratio_l2633_263348

/-- Represents the marks Eva scored in a subject for a semester -/
structure Marks where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Represents Eva's marks for both semesters -/
structure YearlyMarks where
  first_semester : Marks
  second_semester : Marks

def total_marks (ym : YearlyMarks) : ℕ :=
  ym.first_semester.maths + ym.first_semester.arts + ym.first_semester.science +
  ym.second_semester.maths + ym.second_semester.arts + ym.second_semester.science

theorem eva_marks_ratio :
  ∀ (ym : YearlyMarks),
    ym.second_semester.maths = 80 →
    ym.second_semester.arts = 90 →
    ym.second_semester.science = 90 →
    ym.first_semester.maths = ym.second_semester.maths + 10 →
    ym.first_semester.arts = ym.second_semester.arts - 15 →
    ym.first_semester.science < ym.second_semester.science →
    total_marks ym = 485 →
    ∃ (x : ℕ), 
      ym.second_semester.science - ym.first_semester.science = x ∧
      x = 30 ∧
      (x : ℚ) / ym.second_semester.science = 1 / 3 :=
by sorry


end NUMINAMATH_CALUDE_eva_marks_ratio_l2633_263348


namespace NUMINAMATH_CALUDE_given_segments_proportionate_l2633_263386

/-- A set of line segments is proportionate if the product of any two segments
    equals the product of the remaining two segments. -/
def IsProportionate (a b c d : ℝ) : Prop :=
  a * d = b * c

/-- The given set of line segments. -/
def LineSegments : (ℝ × ℝ × ℝ × ℝ) :=
  (3, 6, 4, 8)

/-- Theorem stating that the given set of line segments is proportionate. -/
theorem given_segments_proportionate :
  let (a, b, c, d) := LineSegments
  IsProportionate a b c d := by
  sorry

end NUMINAMATH_CALUDE_given_segments_proportionate_l2633_263386


namespace NUMINAMATH_CALUDE_chromosome_set_variation_l2633_263393

/-- Represents the types of chromosome number variations -/
inductive ChromosomeVariationType
| IndividualChange
| SetChange

/-- Represents the form of chromosome changes -/
inductive ChromosomeChangeForm
| Individual
| Set

/-- Definition of chromosome number variation -/
structure ChromosomeVariation where
  type : ChromosomeVariationType
  form : ChromosomeChangeForm

/-- Theorem stating that one type of chromosome number variation involves
    doubling or halving of chromosomes in the form of chromosome sets -/
theorem chromosome_set_variation :
  ∃ (cv : ChromosomeVariation),
    cv.type = ChromosomeVariationType.SetChange ∧
    cv.form = ChromosomeChangeForm.Set :=
sorry

end NUMINAMATH_CALUDE_chromosome_set_variation_l2633_263393


namespace NUMINAMATH_CALUDE_total_oranges_bought_l2633_263355

/-- The number of times Stephanie went to the store last month -/
def store_visits : ℕ := 8

/-- The number of oranges Stephanie buys each time she goes to the store -/
def oranges_per_visit : ℕ := 2

/-- Theorem: The total number of oranges Stephanie bought last month is 16 -/
theorem total_oranges_bought : store_visits * oranges_per_visit = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_bought_l2633_263355


namespace NUMINAMATH_CALUDE_unique_digit_solution_l2633_263399

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem unique_digit_solution (A M C : ℕ) 
  (h_A : is_digit A) (h_M : is_digit M) (h_C : is_digit C)
  (h_eq : (100*A + 10*M + C) * (A + M + C) = 2005) : 
  A = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l2633_263399


namespace NUMINAMATH_CALUDE_debt_payment_average_l2633_263390

theorem debt_payment_average : 
  let total_payments : ℕ := 52
  let first_payment_count : ℕ := 25
  let first_payment_amount : ℚ := 500
  let additional_amount : ℚ := 100
  let second_payment_count : ℕ := total_payments - first_payment_count
  let second_payment_amount : ℚ := first_payment_amount + additional_amount
  let total_amount : ℚ := first_payment_count * first_payment_amount + 
                          second_payment_count * second_payment_amount
  let average_payment : ℚ := total_amount / total_payments
  average_payment = 551.92 := by
sorry

end NUMINAMATH_CALUDE_debt_payment_average_l2633_263390


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2633_263372

theorem root_sum_theorem (m n : ℝ) : 
  (∀ x, x^2 - (m+n)*x + m*n = 0 ↔ x = m ∨ x = n) → 
  m = 2*n → 
  m + n = 3*n :=
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2633_263372


namespace NUMINAMATH_CALUDE_mary_coins_l2633_263309

theorem mary_coins (dimes quarters : ℕ) 
  (h1 : quarters = 2 * dimes + 7)
  (h2 : (0.10 : ℚ) * dimes + (0.25 : ℚ) * quarters = 10.15) : quarters = 35 := by
  sorry

end NUMINAMATH_CALUDE_mary_coins_l2633_263309


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_specific_case_l2633_263319

def binomial_expansion (a b : ℝ) (n : ℕ) := (a + b)^n

def fourth_term_coefficient (a b : ℝ) (n : ℕ) : ℝ :=
  Nat.choose n 3 * a^(n-3) * b^3

theorem fourth_term_coefficient_specific_case :
  fourth_term_coefficient (1/2 * Real.sqrt x) (2/(3*x)) 6 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_specific_case_l2633_263319


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l2633_263311

theorem ferris_wheel_capacity (total_people : ℕ) (num_seats : ℕ) (people_per_seat : ℕ) : 
  total_people = 18 → num_seats = 2 → people_per_seat = total_people / num_seats → people_per_seat = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l2633_263311


namespace NUMINAMATH_CALUDE_special_function_value_l2633_263347

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 2))

/-- The main theorem stating that f(2010) = 2 for any function satisfying the conditions -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) : f 2010 = 2 := by
  sorry


end NUMINAMATH_CALUDE_special_function_value_l2633_263347


namespace NUMINAMATH_CALUDE_weaving_problem_l2633_263332

/-- Represents the daily increase in cloth production -/
def daily_increase : ℚ := 16 / 29

/-- Represents the number of days of weaving -/
def days : ℕ := 30

/-- Represents the amount woven on the first day -/
def first_day_production : ℚ := 5

/-- Represents the total amount of cloth woven -/
def total_production : ℚ := 390

theorem weaving_problem :
  first_day_production * days + (days * (days - 1) / 2) * daily_increase = total_production := by
  sorry

end NUMINAMATH_CALUDE_weaving_problem_l2633_263332


namespace NUMINAMATH_CALUDE_sharon_coffee_pods_l2633_263395

/-- Calculates the number of pods in a box given vacation details and spending -/
def pods_per_box (vacation_days : ℕ) (daily_pods : ℕ) (total_spent : ℕ) (price_per_box : ℕ) : ℕ :=
  let total_pods := vacation_days * daily_pods
  let boxes_bought := total_spent / price_per_box
  total_pods / boxes_bought

/-- Proves that the number of pods in a box is 30 given the specific vacation details -/
theorem sharon_coffee_pods :
  pods_per_box 40 3 32 8 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sharon_coffee_pods_l2633_263395


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2633_263398

theorem complex_expression_evaluation :
  let a : ℂ := 1 + 2*I
  let b : ℂ := 2 + I
  a * b - 2 * b^2 = -6 - 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2633_263398


namespace NUMINAMATH_CALUDE_compound_interest_problem_l2633_263346

theorem compound_interest_problem :
  ∃ (P r : ℝ), P > 0 ∧ r > 0 ∧ 
  P * (1 + r)^2 = 8000 ∧
  P * (1 + r)^3 = 9261 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l2633_263346


namespace NUMINAMATH_CALUDE_stating_max_correct_is_38_l2633_263391

/-- Represents the result of a multiple choice contest. -/
structure ContestResult where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- 
Calculates the maximum number of correctly answered questions in a contest.
-/
def max_correct_answers (result : ContestResult) : ℕ :=
  sorry

/-- 
Theorem stating that for the given contest parameters, 
the maximum number of correctly answered questions is 38.
-/
theorem max_correct_is_38 : 
  let result : ContestResult := {
    total_questions := 60,
    correct_points := 5,
    incorrect_points := -2,
    total_score := 150
  }
  max_correct_answers result = 38 := by
  sorry

end NUMINAMATH_CALUDE_stating_max_correct_is_38_l2633_263391


namespace NUMINAMATH_CALUDE_divisibility_by_900_l2633_263338

theorem divisibility_by_900 (n : ℕ) : ∃ k : ℤ, 6^(2*(n+1)) - 2^(n+3) * 3^(n+2) + 36 = 900 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_900_l2633_263338


namespace NUMINAMATH_CALUDE_sets_theorem_l2633_263305

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 = 0}
def B : Set ℝ := {y | ∃ x, y = x^2 - 4}

-- Statement to prove
theorem sets_theorem :
  (A ∩ B = A) ∧ (A ∪ B = B) := by sorry

end NUMINAMATH_CALUDE_sets_theorem_l2633_263305


namespace NUMINAMATH_CALUDE_x_y_not_congruent_l2633_263316

def x : ℕ → ℕ
  | 0 => 365
  | n + 1 => x n * (x n ^ 1986 + 1) + 1622

def y : ℕ → ℕ
  | 0 => 16
  | n + 1 => y n * (y n ^ 3 + 1) - 1952

theorem x_y_not_congruent (n k : ℕ) : x n % 1987 ≠ y k % 1987 := by
  sorry

end NUMINAMATH_CALUDE_x_y_not_congruent_l2633_263316


namespace NUMINAMATH_CALUDE_a_range_l2633_263310

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

theorem a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) →
  (3/2 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_a_range_l2633_263310


namespace NUMINAMATH_CALUDE_cos_2theta_value_l2633_263320

theorem cos_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ) ^ (2 * n) = 3) :
  Real.cos (2 * θ) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_2theta_value_l2633_263320


namespace NUMINAMATH_CALUDE_ivory_josh_riddle_difference_l2633_263318

theorem ivory_josh_riddle_difference :
  ∀ (ivory_riddles josh_riddles taso_riddles : ℕ),
    josh_riddles = 8 →
    taso_riddles = 24 →
    taso_riddles = 2 * ivory_riddles →
    ivory_riddles - josh_riddles = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ivory_josh_riddle_difference_l2633_263318


namespace NUMINAMATH_CALUDE_square_root_calculation_l2633_263350

theorem square_root_calculation : Real.sqrt (5^2 - 4^2 - 3^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_root_calculation_l2633_263350


namespace NUMINAMATH_CALUDE_sum_positive_not_sufficient_nor_necessary_for_product_positive_l2633_263328

theorem sum_positive_not_sufficient_nor_necessary_for_product_positive :
  ∃ (a b : ℝ), (a + b > 0 ∧ a * b ≤ 0) ∧ ∃ (c d : ℝ), (c + d ≤ 0 ∧ c * d > 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_not_sufficient_nor_necessary_for_product_positive_l2633_263328


namespace NUMINAMATH_CALUDE_inequality_proof_l2633_263341

theorem inequality_proof (α β γ : ℝ) 
  (h1 : β * γ ≠ 0) 
  (h2 : (1 - γ^2) / (β * γ) ≥ 0) : 
  10 * (α^2 + β^2 + γ^2 - β * γ^2) ≥ 2 * α * β + 5 * α * γ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2633_263341


namespace NUMINAMATH_CALUDE_cover_ways_eq_fib_succ_l2633_263361

/-- The number of ways to cover a 2 × n grid with 1 × 2 tiles -/
def cover_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n+2 => cover_ways (n+1) + cover_ways n

/-- The Fibonacci sequence -/
def fib (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n+2 => fib (n+1) + fib n

theorem cover_ways_eq_fib_succ (n : ℕ) : cover_ways n = fib (n+1) := by
  sorry

#eval cover_ways 10  -- Should evaluate to 89

end NUMINAMATH_CALUDE_cover_ways_eq_fib_succ_l2633_263361


namespace NUMINAMATH_CALUDE_andrei_apple_spending_l2633_263359

/-- Calculates Andrei's monthly spending on apples after price increase and discount -/
def andreiMonthlySpending (originalPrice : ℚ) (priceIncrease : ℚ) (discount : ℚ) (kgPerMonth : ℚ) : ℚ :=
  let newPrice := originalPrice * (1 + priceIncrease)
  let discountedPrice := newPrice * (1 - discount)
  discountedPrice * kgPerMonth

/-- Theorem stating that Andrei's monthly spending on apples is 99 rubles -/
theorem andrei_apple_spending :
  andreiMonthlySpending 50 (1/10) (1/10) 2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_andrei_apple_spending_l2633_263359


namespace NUMINAMATH_CALUDE_problem_solution_l2633_263394

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x + 1| - |x - a|

def g (a : ℝ) (x : ℝ) : ℝ := f a x + 3 * |x - a|

theorem problem_solution :
  (∀ x : ℝ, f 2 x ≥ 0 ↔ x ≤ -4 ∨ x ≥ 0) ∧
  (∃ t : ℝ, (∀ x : ℝ, g 1 x ≥ t) ∧
   (∀ m n : ℝ, m > 0 → n > 0 → 2/m + 1/(2*n) = t →
    m + n ≥ 9/8) ∧
   (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ 2/m + 1/(2*n) = t ∧ m + n = 9/8)) :=
by sorry

end

end NUMINAMATH_CALUDE_problem_solution_l2633_263394


namespace NUMINAMATH_CALUDE_randys_trip_length_l2633_263349

theorem randys_trip_length :
  ∀ (total_length : ℚ),
    (1 / 3 : ℚ) * total_length + 20 + (1 / 5 : ℚ) * total_length = total_length →
    total_length = 300 / 7 := by
  sorry

end NUMINAMATH_CALUDE_randys_trip_length_l2633_263349


namespace NUMINAMATH_CALUDE_triangle_area_with_given_conditions_l2633_263374

/-- Given a triangle DEF with inradius r, circumradius R, and angles D, E, F,
    prove that if r = 2, R = 9, and 2cos(E) = cos(D) + cos(F), then the area of the triangle is 54. -/
theorem triangle_area_with_given_conditions (D E F : Real) (r R : Real) :
  r = 2 →
  R = 9 →
  2 * Real.cos E = Real.cos D + Real.cos F →
  ∃ (area : Real), area = 54 ∧ area = r * (Real.sin D + Real.sin E + Real.sin F) * R / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_with_given_conditions_l2633_263374


namespace NUMINAMATH_CALUDE_fourth_grade_agreement_l2633_263354

theorem fourth_grade_agreement (third_grade : ℕ) (total : ℕ) (h1 : third_grade = 154) (h2 : total = 391) :
  total - third_grade = 237 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_agreement_l2633_263354


namespace NUMINAMATH_CALUDE_linear_function_inequality_solution_l2633_263327

/-- Given a linear function y = kx + b, prove that under certain conditions, 
    the solution set of an inequality is x < 1 -/
theorem linear_function_inequality_solution 
  (k b n : ℝ) 
  (h_k : k ≠ 0)
  (h_n : n > 2)
  (h_y_neg1 : k * (-1) + b = n)
  (h_y_1 : k * 1 + b = 2) :
  {x : ℝ | (k - 2) * x + b > 0} = {x : ℝ | x < 1} := by
sorry

end NUMINAMATH_CALUDE_linear_function_inequality_solution_l2633_263327


namespace NUMINAMATH_CALUDE_crayon_distribution_l2633_263376

theorem crayon_distribution (total benny fred jason sarah : ℕ) : 
  total = 96 →
  benny = 12 →
  fred = 2 * benny →
  jason = 3 * sarah →
  fred + benny + jason + sarah = total →
  (fred = 24 ∧ benny = 12 ∧ jason = 45 ∧ sarah = 15) :=
by sorry

end NUMINAMATH_CALUDE_crayon_distribution_l2633_263376


namespace NUMINAMATH_CALUDE_g_expression_l2633_263387

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the relationship between f and g
def g_relation (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 2) = f x

-- Theorem statement
theorem g_expression (g : ℝ → ℝ) (h : g_relation g) :
  ∀ x, g x = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l2633_263387


namespace NUMINAMATH_CALUDE_efficiency_ratio_l2633_263363

-- Define the efficiencies of workers a, b, c, and d
def efficiency_a : ℚ := 1 / 18
def efficiency_b : ℚ := 1 / 36
def efficiency_c : ℚ := 1 / 20
def efficiency_d : ℚ := 1 / 30

-- Theorem statement
theorem efficiency_ratio :
  -- a and b together have the same efficiency as c and d together
  efficiency_a + efficiency_b = efficiency_c + efficiency_d →
  -- The ratio of a's efficiency to b's efficiency is 2:1
  efficiency_a / efficiency_b = 2 := by
sorry

end NUMINAMATH_CALUDE_efficiency_ratio_l2633_263363


namespace NUMINAMATH_CALUDE_characterize_valid_functions_l2633_263325

def is_valid_function (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, a + f b + f (f c) = 0 →
    f a ^ 3 + b * (f b) ^ 2 + c ^ 2 * f c = 3 * a * b * c

theorem characterize_valid_functions :
  ∀ f : ℝ → ℝ, is_valid_function f →
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) ∨ (∀ x : ℝ, f x = 0) :=
sorry

end NUMINAMATH_CALUDE_characterize_valid_functions_l2633_263325
