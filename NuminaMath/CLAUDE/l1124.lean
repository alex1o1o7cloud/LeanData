import Mathlib

namespace NUMINAMATH_CALUDE_smallest_perimeter_l1124_112492

/-- Triangle PQR with positive integer side lengths, PQ = PR, and J is the intersection of angle bisectors of ∠Q and ∠R with QJ = 10 -/
structure IsoscelesTriangle where
  PQ : ℕ+
  QR : ℕ+
  J : ℝ × ℝ
  QJ_length : ℝ
  qj_eq_10 : QJ_length = 10

/-- The smallest possible perimeter of triangle PQR is 96 -/
theorem smallest_perimeter (t : IsoscelesTriangle) : 
  ∃ (min_perimeter : ℕ), min_perimeter = 96 ∧ 
  ∀ (perimeter : ℕ), perimeter ≥ min_perimeter :=
by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l1124_112492


namespace NUMINAMATH_CALUDE_prove_complex_circle_theorem_l1124_112453

def complex_circle_theorem (z : ℂ) : Prop :=
  Complex.abs (z - Complex.I) = Real.sqrt 5 →
  ∃ (center : ℂ) (radius : ℝ),
    center = Complex.mk 0 1 ∧
    radius = Real.sqrt 5 ∧
    Complex.abs (z - center) = radius

theorem prove_complex_circle_theorem :
  ∀ z : ℂ, complex_circle_theorem z :=
by
  sorry

end NUMINAMATH_CALUDE_prove_complex_circle_theorem_l1124_112453


namespace NUMINAMATH_CALUDE_plumbing_job_washers_remaining_l1124_112444

/-- Calculates the number of washers remaining after a plumbing job. -/
def washers_remaining (copper_pipe : ℕ) (pvc_pipe : ℕ) (steel_pipe : ℕ) 
  (copper_bolt_length : ℕ) (pvc_bolt_length : ℕ) (steel_bolt_length : ℕ)
  (copper_washers_per_bolt : ℕ) (pvc_washers_per_bolt : ℕ) (steel_washers_per_bolt : ℕ)
  (total_washers : ℕ) : ℕ :=
  let copper_bolts := (copper_pipe + copper_bolt_length - 1) / copper_bolt_length
  let pvc_bolts := (pvc_pipe + pvc_bolt_length - 1) / pvc_bolt_length * 2
  let steel_bolts := (steel_pipe + steel_bolt_length - 1) / steel_bolt_length
  let washers_used := copper_bolts * copper_washers_per_bolt + 
                      pvc_bolts * pvc_washers_per_bolt + 
                      steel_bolts * steel_washers_per_bolt
  total_washers - washers_used

theorem plumbing_job_washers_remaining :
  washers_remaining 40 30 20 5 10 8 2 3 4 80 = 43 := by
  sorry

end NUMINAMATH_CALUDE_plumbing_job_washers_remaining_l1124_112444


namespace NUMINAMATH_CALUDE_quadratic_transformation_has_integer_roots_l1124_112466

/-- Represents a quadratic polynomial ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Checks if a quadratic polynomial has integer roots -/
def has_integer_roots (p : QuadraticPolynomial) : Prop :=
  ∃ (x : ℤ), p.a * x^2 + p.b * x + p.c = 0

/-- Represents a single step in the transformation process -/
inductive TransformationStep
  | IncreaseX
  | DecreaseX
  | IncreaseConstant
  | DecreaseConstant

/-- Applies a transformation step to a polynomial -/
def apply_step (p : QuadraticPolynomial) (step : TransformationStep) : QuadraticPolynomial :=
  match step with
  | TransformationStep.IncreaseX => { a := p.a, b := p.b + 1, c := p.c }
  | TransformationStep.DecreaseX => { a := p.a, b := p.b - 1, c := p.c }
  | TransformationStep.IncreaseConstant => { a := p.a, b := p.b, c := p.c + 1 }
  | TransformationStep.DecreaseConstant => { a := p.a, b := p.b, c := p.c - 1 }

theorem quadratic_transformation_has_integer_roots 
  (initial : QuadraticPolynomial)
  (final : QuadraticPolynomial)
  (h_initial : initial = { a := 1, b := 10, c := 20 })
  (h_final : final = { a := 1, b := 20, c := 10 })
  (h_transform : ∃ (steps : List TransformationStep), 
    final = steps.foldl apply_step initial) :
  ∃ (intermediate : QuadraticPolynomial),
    (∃ (steps : List TransformationStep), intermediate = steps.foldl apply_step initial) ∧
    has_integer_roots intermediate :=
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_has_integer_roots_l1124_112466


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l1124_112451

/-- The distance between the centers of two circles with polar equations ρ = 2cos(θ) and ρ = 4sin(θ) is √5. -/
theorem distance_between_circle_centers :
  let circle1 : ℝ → ℝ := fun θ ↦ 2 * Real.cos θ
  let circle2 : ℝ → ℝ := fun θ ↦ 4 * Real.sin θ
  let center1 : ℝ × ℝ := (1, 0)
  let center2 : ℝ × ℝ := (0, 2)
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = 5 := by
  sorry

#check distance_between_circle_centers

end NUMINAMATH_CALUDE_distance_between_circle_centers_l1124_112451


namespace NUMINAMATH_CALUDE_malfunctioning_clock_correct_time_l1124_112462

/-- Represents a 12-hour digital clock with a malfunction where '2' is displayed as '5' -/
structure MalfunctioningClock where
  /-- The number of hours in the clock (12) -/
  total_hours : ℕ
  /-- The number of minutes per hour (60) -/
  minutes_per_hour : ℕ
  /-- The number of hours affected by the malfunction -/
  incorrect_hours : ℕ
  /-- The number of minutes per hour affected by the malfunction -/
  incorrect_minutes : ℕ

/-- The fraction of the day a malfunctioning clock shows the correct time -/
def correct_time_fraction (clock : MalfunctioningClock) : ℚ :=
  ((clock.total_hours - clock.incorrect_hours : ℚ) / clock.total_hours) *
  ((clock.minutes_per_hour - clock.incorrect_minutes : ℚ) / clock.minutes_per_hour)

theorem malfunctioning_clock_correct_time :
  ∃ (clock : MalfunctioningClock),
    clock.total_hours = 12 ∧
    clock.minutes_per_hour = 60 ∧
    clock.incorrect_hours = 2 ∧
    clock.incorrect_minutes = 15 ∧
    correct_time_fraction clock = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_malfunctioning_clock_correct_time_l1124_112462


namespace NUMINAMATH_CALUDE_min_sum_squares_l1124_112435

def S : Finset Int := {-8, -6, -4, -1, 3, 5, 7, 10}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1124_112435


namespace NUMINAMATH_CALUDE_quadratic_derivative_bound_l1124_112430

theorem quadratic_derivative_bound (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |2 * a * x + b| ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_derivative_bound_l1124_112430


namespace NUMINAMATH_CALUDE_pebbles_distribution_l1124_112493

/-- The number of pebbles in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of pebbles Janice had -/
def janice_dozens : ℕ := 3

/-- The total number of pebbles Janice had -/
def total_pebbles : ℕ := janice_dozens * dozen

/-- The number of friends who received pebbles -/
def num_friends : ℕ := 9

/-- The number of pebbles each friend received -/
def pebbles_per_friend : ℕ := total_pebbles / num_friends

theorem pebbles_distribution :
  pebbles_per_friend = 4 :=
sorry

end NUMINAMATH_CALUDE_pebbles_distribution_l1124_112493


namespace NUMINAMATH_CALUDE_medium_birdhouse_price_l1124_112498

/-- The price of a large birdhouse -/
def large_price : ℕ := 22

/-- The price of a small birdhouse -/
def small_price : ℕ := 7

/-- The number of large birdhouses sold -/
def large_sold : ℕ := 2

/-- The number of medium birdhouses sold -/
def medium_sold : ℕ := 2

/-- The number of small birdhouses sold -/
def small_sold : ℕ := 3

/-- The total amount made from all birdhouses -/
def total_amount : ℕ := 97

/-- The price of a medium birdhouse -/
def medium_price : ℕ := 16

theorem medium_birdhouse_price : 
  large_price * large_sold + medium_price * medium_sold + small_price * small_sold = total_amount :=
by sorry

end NUMINAMATH_CALUDE_medium_birdhouse_price_l1124_112498


namespace NUMINAMATH_CALUDE_diagonals_properties_l1124_112488

/-- Number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem diagonals_properties :
  (∀ n : ℕ, n ≥ 4 → num_diagonals n = n * (n - 3) / 2) →
  num_diagonals 4 = 2 →
  num_diagonals 5 = 5 ∧
  num_diagonals 6 - num_diagonals 5 = 4 ∧
  ∀ n : ℕ, n ≥ 4 → num_diagonals (n + 1) - num_diagonals n = n - 1 :=
by sorry

end NUMINAMATH_CALUDE_diagonals_properties_l1124_112488


namespace NUMINAMATH_CALUDE_garden_vegetables_theorem_l1124_112499

/-- Represents the quantities of vegetables in a garden -/
structure GardenVegetables where
  tomatoes : ℕ
  potatoes : ℕ
  cabbages : ℕ
  eggplants : ℕ

/-- Calculates the final quantities of vegetables after changes -/
def finalQuantities (initial : GardenVegetables) 
  (tomatoesPicked potatoes_sold cabbagesBought eggplantsPlanted : ℕ) : GardenVegetables :=
  { tomatoes := initial.tomatoes - min initial.tomatoes tomatoesPicked,
    potatoes := initial.potatoes - min initial.potatoes potatoes_sold,
    cabbages := initial.cabbages + cabbagesBought,
    eggplants := initial.eggplants + eggplantsPlanted }

theorem garden_vegetables_theorem (initial : GardenVegetables) 
  (tomatoesPicked potatoes_sold cabbagesBought eggplantsPlanted : ℕ) :
  initial.tomatoes = 177 → 
  initial.potatoes = 12 → 
  initial.cabbages = 25 → 
  initial.eggplants = 10 → 
  tomatoesPicked = 53 → 
  potatoes_sold = 15 → 
  cabbagesBought = 32 → 
  eggplantsPlanted = 18 → 
  finalQuantities initial tomatoesPicked potatoes_sold cabbagesBought eggplantsPlanted = 
    { tomatoes := 124, potatoes := 0, cabbages := 57, eggplants := 28 } := by
  sorry

end NUMINAMATH_CALUDE_garden_vegetables_theorem_l1124_112499


namespace NUMINAMATH_CALUDE_toms_next_birthday_l1124_112459

theorem toms_next_birthday (sally tom jenny : ℝ) 
  (h1 : sally = 1.25 * tom)  -- Sally is 25% older than Tom
  (h2 : tom = 0.7 * jenny)   -- Tom is 30% younger than Jenny
  (h3 : sally + tom + jenny = 30)  -- Sum of ages is 30
  : ⌊tom⌋ + 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_toms_next_birthday_l1124_112459


namespace NUMINAMATH_CALUDE_division_fraction_problem_l1124_112417

theorem division_fraction_problem : (1 / 60) / ((2 / 3) - (1 / 5) - (2 / 5)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_division_fraction_problem_l1124_112417


namespace NUMINAMATH_CALUDE_specific_arrangement_surface_area_l1124_112437

/-- Represents a cube arrangement with two layers --/
structure CubeArrangement where
  totalCubes : Nat
  layerSize : Nat
  cubeEdgeLength : Real

/-- Calculates the exposed surface area of the cube arrangement --/
def exposedSurfaceArea (arrangement : CubeArrangement) : Real :=
  sorry

/-- Theorem stating that the exposed surface area of the specific arrangement is 49 square meters --/
theorem specific_arrangement_surface_area :
  let arrangement : CubeArrangement := {
    totalCubes := 18,
    layerSize := 9,
    cubeEdgeLength := 1
  }
  exposedSurfaceArea arrangement = 49 := by sorry

end NUMINAMATH_CALUDE_specific_arrangement_surface_area_l1124_112437


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1124_112464

/-- Given two vectors a and b in R², prove that if they are perpendicular
    and a = (1, -2) and b = (m, m+2), then m = -4. -/
theorem perpendicular_vectors_m_value
  (a b : ℝ × ℝ)
  (h1 : a = (1, -2))
  (h2 : ∃ m : ℝ, b = (m, m + 2))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  ∃ m : ℝ, b = (m, m + 2) ∧ m = -4 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1124_112464


namespace NUMINAMATH_CALUDE_cube_sum_l1124_112440

theorem cube_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_l1124_112440


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1124_112410

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±2√2x -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_equation : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptotes : ∀ x, ∃ y, y = 2 * Real.sqrt 2 * x ∨ y = -2 * Real.sqrt 2 * x) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1124_112410


namespace NUMINAMATH_CALUDE_adams_account_balance_l1124_112449

theorem adams_account_balance :
  let initial_savings : ℚ := 1579.37
  let monday_earnings : ℚ := 21.85
  let tuesday_earnings : ℚ := 33.28
  let wednesday_spending : ℚ := 87.41
  let final_balance : ℚ := initial_savings + monday_earnings + tuesday_earnings - wednesday_spending
  final_balance = 1547.09 := by sorry

end NUMINAMATH_CALUDE_adams_account_balance_l1124_112449


namespace NUMINAMATH_CALUDE_sqrt_sum_quotient_l1124_112489

theorem sqrt_sum_quotient : 
  (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 1.44) / (Real.sqrt 0.49) = 185/63 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_quotient_l1124_112489


namespace NUMINAMATH_CALUDE_quadratic_properties_l1124_112404

def f (x : ℝ) := -2 * x^2 + 4 * x + 3

theorem quadratic_properties :
  (∀ x y, x < y → f x > f y) ∧
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (f 1 = 5) ∧
  (∀ x, x > 1 → ∀ y, y > x → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1124_112404


namespace NUMINAMATH_CALUDE_largest_k_phi_sigma_power_two_l1124_112467

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem largest_k_phi_sigma_power_two :
  (∀ k : ℕ, k > 31 → phi (sigma (2^k)) ≠ 2^k) ∧
  phi (sigma (2^31)) = 2^31 := by sorry

end NUMINAMATH_CALUDE_largest_k_phi_sigma_power_two_l1124_112467


namespace NUMINAMATH_CALUDE_three_vectors_with_zero_sum_and_unit_difference_l1124_112479

theorem three_vectors_with_zero_sum_and_unit_difference (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :
  ∃ (a b c : α), 
    a + b + c = 0 ∧ 
    ‖a + b - c‖ = 1 ∧ 
    ‖b + c - a‖ = 1 ∧ 
    ‖c + a - b‖ = 1 ∧
    ‖a‖ = (1 : ℝ) / 2 ∧ 
    ‖b‖ = (1 : ℝ) / 2 ∧ 
    ‖c‖ = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_three_vectors_with_zero_sum_and_unit_difference_l1124_112479


namespace NUMINAMATH_CALUDE_basil_seed_cost_l1124_112408

/-- Represents the cost structure and profit for Burt's basil plant business -/
structure BasilBusiness where
  seed_cost : ℝ
  soil_cost : ℝ
  plants : ℕ
  price_per_plant : ℝ
  net_profit : ℝ

/-- Calculates the total revenue from selling basil plants -/
def total_revenue (b : BasilBusiness) : ℝ :=
  b.plants * b.price_per_plant

/-- Calculates the total expenses for the basil business -/
def total_expenses (b : BasilBusiness) : ℝ :=
  b.seed_cost + b.soil_cost

/-- Theorem stating that given the conditions, the seed cost is $2.00 -/
theorem basil_seed_cost (b : BasilBusiness) 
  (h1 : b.soil_cost = 8)
  (h2 : b.plants = 20)
  (h3 : b.price_per_plant = 5)
  (h4 : b.net_profit = 90)
  (h5 : total_revenue b - total_expenses b = b.net_profit) :
  b.seed_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_basil_seed_cost_l1124_112408


namespace NUMINAMATH_CALUDE_division_simplification_l1124_112454

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  6 * x^3 * y^2 / (-3 * x * y) = -2 * x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l1124_112454


namespace NUMINAMATH_CALUDE_base5_multiplication_l1124_112460

/-- Converts a base 5 number to its decimal equivalent -/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to its base 5 representation -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base5_multiplication (a b : List Nat) :
  decimalToBase5 (base5ToDecimal a * base5ToDecimal b) = [2, 1, 3, 4] :=
  by sorry

end NUMINAMATH_CALUDE_base5_multiplication_l1124_112460


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l1124_112409

/-- The mapping f from ℝ² to ℝ² defined by f(x, y) = (x+2y, 2x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2*p.2, 2*p.1 - p.2)

/-- The theorem stating that (-1/3, 5/3) is the pre-image of (3, 1) under the mapping f -/
theorem preimage_of_3_1 :
  f (-1/3, 5/3) = (3, 1) :=
by sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l1124_112409


namespace NUMINAMATH_CALUDE_max_value_problem_l1124_112421

theorem max_value_problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  ∃ (max : ℝ), max = 1 ∧ x + y^3 + z^2 ≤ max ∧ ∃ (x' y' z' : ℝ), x' + y'^3 + z'^2 = max :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l1124_112421


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l1124_112419

theorem min_value_sum_squares (x y z : ℝ) (h : x + 2*y + Real.sqrt 3 * z = 1) : 
  ∀ (a b c : ℝ), a^2 + b^2 + c^2 ≥ (1/8 : ℝ) ∧ 
  (∃ (x₀ y₀ z₀ : ℝ), x₀^2 + y₀^2 + z₀^2 = (1/8 : ℝ) ∧ x₀ + 2*y₀ + Real.sqrt 3 * z₀ = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l1124_112419


namespace NUMINAMATH_CALUDE_smallest_divisor_exponent_l1124_112476

def polynomial (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_divisor_exponent :
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (z : ℂ), polynomial z = 0 → z^k = 1) ∧
  (∀ (m : ℕ), m > 0 → m < k → ∃ (w : ℂ), polynomial w = 0 ∧ w^m ≠ 1) ∧
  k = 120 :=
sorry

end NUMINAMATH_CALUDE_smallest_divisor_exponent_l1124_112476


namespace NUMINAMATH_CALUDE_right_triangle_angle_sum_l1124_112457

theorem right_triangle_angle_sum (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = 180) (h5 : A + B = C) : C = 90 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_sum_l1124_112457


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1124_112494

/-- The standard equation of a hyperbola with given foci and real axis length -/
theorem hyperbola_equation (x y : ℝ) : 
  let foci_distance : ℝ := 8
  let real_axis_length : ℝ := 4
  let a : ℝ := real_axis_length / 2
  let c : ℝ := foci_distance / 2
  let b_squared : ℝ := c^2 - a^2
  x^2 / a^2 - y^2 / b_squared = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1124_112494


namespace NUMINAMATH_CALUDE_field_division_l1124_112475

theorem field_division (total_area : ℝ) (smaller_area larger_area : ℝ) : 
  total_area = 900 →
  smaller_area + larger_area = total_area →
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area = 405 := by
sorry

end NUMINAMATH_CALUDE_field_division_l1124_112475


namespace NUMINAMATH_CALUDE_floor_of_3_point_9_l1124_112480

theorem floor_of_3_point_9 : ⌊(3.9 : ℝ)⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_of_3_point_9_l1124_112480


namespace NUMINAMATH_CALUDE_systematic_sampling_constant_difference_l1124_112463

/-- Represents a sequence of 5 numbers -/
structure Sequence :=
  (numbers : Fin 5 → Nat)

/-- Checks if a sequence has a constant difference between consecutive elements -/
def hasConstantDifference (s : Sequence) (d : Nat) : Prop :=
  ∀ i : Fin 4, s.numbers (i.succ) - s.numbers i = d

/-- Systematic sampling function -/
def systematicSample (totalStudents : Nat) (sampleSize : Nat) : Sequence :=
  sorry

theorem systematic_sampling_constant_difference :
  let totalStudents : Nat := 55
  let sampleSize : Nat := 5
  let sampledSequence := systematicSample totalStudents sampleSize
  hasConstantDifference sampledSequence (totalStudents / sampleSize) :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_constant_difference_l1124_112463


namespace NUMINAMATH_CALUDE_intersection_passes_through_center_l1124_112458

-- Define the cube
def Cube : Type := Unit

-- Define a point in 3D space
def Point : Type := Unit

-- Define a plane
def Plane : Type := Unit

-- Define a hexagon
structure Hexagon :=
  (A B C D E F : Point)

-- Define the intersection of a cube and a plane
def intersection (c : Cube) (p : Plane) : Hexagon := sorry

-- Define the center of a cube
def center (c : Cube) : Point := sorry

-- Define a function to check if three lines intersect at a point
def intersect_at (p1 p2 p3 p4 p5 p6 : Point) (O : Point) : Prop := sorry

-- Theorem statement
theorem intersection_passes_through_center (c : Cube) (p : Plane) :
  let h := intersection c p
  intersect_at h.A h.D h.B h.E h.C h.F (center c) := by sorry

end NUMINAMATH_CALUDE_intersection_passes_through_center_l1124_112458


namespace NUMINAMATH_CALUDE_double_acute_angle_less_than_180_degrees_l1124_112432

theorem double_acute_angle_less_than_180_degrees (α : Real) :
  (0 < α ∧ α < Real.pi / 2) → 2 * α < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_double_acute_angle_less_than_180_degrees_l1124_112432


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_special_hexagon_perimeter_is_4_sqrt_6_l1124_112445

/-- An equilateral hexagon with specific angle and area properties -/
structure SpecialHexagon where
  -- The hexagon is equilateral
  equilateral : Bool
  -- Alternating interior angles are 120° and 60°
  alternating_angles : Bool
  -- The area of the hexagon
  area : ℝ
  -- Conditions on the hexagon
  h_equilateral : equilateral = true
  h_alternating_angles : alternating_angles = true
  h_area : area = 12

/-- The perimeter of a SpecialHexagon is 4√6 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : ℝ :=
  4 * Real.sqrt 6

/-- The perimeter of a SpecialHexagon with area 12 is 4√6 -/
theorem special_hexagon_perimeter_is_4_sqrt_6 (h : SpecialHexagon) :
  special_hexagon_perimeter h = 4 * Real.sqrt 6 := by
  sorry

#check special_hexagon_perimeter_is_4_sqrt_6

end NUMINAMATH_CALUDE_special_hexagon_perimeter_special_hexagon_perimeter_is_4_sqrt_6_l1124_112445


namespace NUMINAMATH_CALUDE_negation_at_most_three_l1124_112441

theorem negation_at_most_three (x : ℝ) : ¬(x ≤ 3) ↔ x > 3 := by sorry

end NUMINAMATH_CALUDE_negation_at_most_three_l1124_112441


namespace NUMINAMATH_CALUDE_point_on_line_extension_l1124_112407

theorem point_on_line_extension (A B C D : EuclideanSpace ℝ (Fin 2)) :
  (D - A) = 2 • (B - A) - (C - A) →
  ∃ t : ℝ, t > 1 ∧ D = C + t • (B - C) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_extension_l1124_112407


namespace NUMINAMATH_CALUDE_black_ball_probability_l1124_112469

theorem black_ball_probability (total : ℕ) (white yellow black : ℕ) :
  total = white + yellow + black →
  white = 10 →
  yellow = 5 →
  black = 10 →
  (black : ℚ) / (yellow + black) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_black_ball_probability_l1124_112469


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1124_112402

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_condition : a + b + c = 24)
  (sum_squares_condition : a^2 + b^2 + c^2 = 392)
  (sum_cubes_condition : a^3 + b^3 + c^3 = 2760) :
  a * b * c = 1844 := by sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1124_112402


namespace NUMINAMATH_CALUDE_remaining_movies_to_watch_l1124_112465

theorem remaining_movies_to_watch (total_movies watched_movies : ℕ) : 
  total_movies = 8 → watched_movies = 4 → total_movies - watched_movies = 4 :=
by sorry

end NUMINAMATH_CALUDE_remaining_movies_to_watch_l1124_112465


namespace NUMINAMATH_CALUDE_not_right_triangle_l1124_112427

theorem not_right_triangle (a b c : ℝ) (ha : a = 1/3) (hb : b = 1/4) (hc : c = 1/5) :
  ¬ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l1124_112427


namespace NUMINAMATH_CALUDE_unique_root_condition_l1124_112422

/-- The system of equations has only one root if and only if m is 0 or 2 -/
theorem unique_root_condition (m : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^2 = 2*|p.1| ∧ |p.1| - p.2 - m = 1 - p.2^2) ↔ (m = 0 ∨ m = 2) :=
sorry

end NUMINAMATH_CALUDE_unique_root_condition_l1124_112422


namespace NUMINAMATH_CALUDE_lilith_water_bottle_price_l1124_112434

/-- The regular price per water bottle in Lilith's town -/
def regularPrice : ℚ := 185 / 100

theorem lilith_water_bottle_price :
  let initialBottles : ℕ := 60
  let initialPrice : ℚ := 2
  let shortfall : ℚ := 9
  (initialBottles : ℚ) * regularPrice = initialBottles * initialPrice - shortfall :=
by sorry

end NUMINAMATH_CALUDE_lilith_water_bottle_price_l1124_112434


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1124_112424

theorem complex_equation_solution :
  ∃ z : ℂ, (5 : ℂ) + 2 * Complex.I * z = (4 : ℂ) - 6 * Complex.I * z ∧ z = Complex.I / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1124_112424


namespace NUMINAMATH_CALUDE_village_population_l1124_112452

/-- Given that 40% of a village's population is 23040, prove that the total population is 57600. -/
theorem village_population (population : ℕ) (h : (40 : ℕ) * population = 100 * 23040) : population = 57600 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1124_112452


namespace NUMINAMATH_CALUDE_milan_phone_bill_l1124_112412

/-- Calculates the number of minutes billed given the total bill, monthly fee, and cost per minute -/
def minutes_billed (total_bill monthly_fee cost_per_minute : ℚ) : ℚ :=
  (total_bill - monthly_fee) / cost_per_minute

/-- Proves that given the specified conditions, the number of minutes billed is 178 -/
theorem milan_phone_bill :
  let total_bill : ℚ := 23.36
  let monthly_fee : ℚ := 2
  let cost_per_minute : ℚ := 0.12
  minutes_billed total_bill monthly_fee cost_per_minute = 178 := by
  sorry

end NUMINAMATH_CALUDE_milan_phone_bill_l1124_112412


namespace NUMINAMATH_CALUDE_find_original_number_l1124_112405

/-- A four-digit number is between 1000 and 9999 inclusive -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem find_original_number (N : ℕ) (h1 : FourDigitNumber N) (h2 : N - 3 - 57 = 1819) : N = 1879 := by
  sorry

end NUMINAMATH_CALUDE_find_original_number_l1124_112405


namespace NUMINAMATH_CALUDE_vector_problem_l1124_112429

/-- Given vectors a and b, if vector c satisfies the conditions, then c equals the expected result. -/
theorem vector_problem (a b c : ℝ × ℝ) : 
  a = (1, 2) → 
  b = (2, -3) → 
  (∃ (k : ℝ), c + a = k • b) → -- (c+a) ∥ b
  (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = 0) → -- c ⟂ (a+b)
  c = (-7/9, -7/3) := by
sorry


end NUMINAMATH_CALUDE_vector_problem_l1124_112429


namespace NUMINAMATH_CALUDE_fold_cut_result_l1124_112436

/-- Represents the possible number of parts after cutting a folded square --/
inductive CutResult
  | OppositeMiddle : CutResult
  | AdjacentMiddle : CutResult

/-- Represents the dimensions of the original rectangle --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the result of folding and cutting the rectangle --/
def fold_and_cut (rect : Rectangle) (cut : CutResult) : Set ℕ :=
  match cut with
  | CutResult.OppositeMiddle => {11, 13}
  | CutResult.AdjacentMiddle => {31, 36, 37, 43}

/-- Theorem stating the result of folding and cutting the specific rectangle --/
theorem fold_cut_result (rect : Rectangle) (h1 : rect.width = 10) (h2 : rect.height = 12) :
  (fold_and_cut rect CutResult.OppositeMiddle = {11, 13}) ∧
  (fold_and_cut rect CutResult.AdjacentMiddle = {31, 36, 37, 43}) := by
  sorry

#check fold_cut_result

end NUMINAMATH_CALUDE_fold_cut_result_l1124_112436


namespace NUMINAMATH_CALUDE_zeros_of_f_l1124_112416

def f (x : ℝ) : ℝ := (x - 1) * (x^2 - 2*x - 3)

theorem zeros_of_f :
  {x : ℝ | f x = 0} = {1, -1, 3} := by sorry

end NUMINAMATH_CALUDE_zeros_of_f_l1124_112416


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_problem_solution_l1124_112472

def geometric_series (a : ℝ) (r : ℝ) : ℕ → ℝ := λ n => a * r^n

theorem infinite_geometric_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∑' n, geometric_series a r n = a / (1 - r) :=
sorry

theorem problem_solution :
  ∑' n, geometric_series (1/4) (1/3) n = 3/8 :=
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_problem_solution_l1124_112472


namespace NUMINAMATH_CALUDE_avg_temp_MTWT_is_48_l1124_112439

/-- The average temperature for Monday, Tuesday, Wednesday, and Thursday -/
def avg_temp_MTWT : ℝ := sorry

/-- The average temperature for some days -/
def avg_temp_some_days : ℝ := 48

/-- The average temperature for Tuesday, Wednesday, Thursday, and Friday -/
def avg_temp_TWTF : ℝ := 40

/-- The temperature on Monday -/
def temp_Monday : ℝ := 42

/-- The temperature on Friday -/
def temp_Friday : ℝ := 10

/-- The theorem stating that the average temperature for Monday, Tuesday, Wednesday, and Thursday is 48 degrees -/
theorem avg_temp_MTWT_is_48 : avg_temp_MTWT = 48 := by sorry

end NUMINAMATH_CALUDE_avg_temp_MTWT_is_48_l1124_112439


namespace NUMINAMATH_CALUDE_ferry_tourist_sum_l1124_112491

/-- The number of trips made by the ferry -/
def num_trips : ℕ := 15

/-- The initial number of tourists -/
def initial_tourists : ℕ := 100

/-- The decrease in number of tourists per trip -/
def tourist_decrease : ℕ := 2

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a : ℤ) (d : ℤ) : ℤ :=
  (n : ℤ) * (2 * a + (n - 1) * d) / 2

theorem ferry_tourist_sum :
  arithmetic_sum num_trips initial_tourists (-tourist_decrease) = 1290 :=
sorry

end NUMINAMATH_CALUDE_ferry_tourist_sum_l1124_112491


namespace NUMINAMATH_CALUDE_calculation_proof_l1124_112446

theorem calculation_proof :
  (1 * (-8) - 9 - (-3) + (-6) = -20) ∧
  (-2^2 + 3 * (-1)^2023 - |1 - 5| / 2 = -9) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1124_112446


namespace NUMINAMATH_CALUDE_cricketer_score_l1124_112477

theorem cricketer_score : ∀ (total_score : ℝ),
  (12 * 4 + 2 * 6 : ℝ) + 0.55223880597014926 * total_score = total_score →
  total_score = 134 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_score_l1124_112477


namespace NUMINAMATH_CALUDE_integral_sqrt_a_squared_minus_x_squared_l1124_112471

theorem integral_sqrt_a_squared_minus_x_squared (a : ℝ) (ha : a > 0) :
  ∫ x in -a..a, Real.sqrt (a^2 - x^2) = (1/2) * π * a^2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_a_squared_minus_x_squared_l1124_112471


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_four_zeros_l1124_112406

theorem smallest_multiplier_for_four_zeros (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < n → ¬(10000 ∣ (975 * 935 * 972 * m))) →
  (10000 ∣ (975 * 935 * 972 * n)) →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_four_zeros_l1124_112406


namespace NUMINAMATH_CALUDE_box_counting_l1124_112443

theorem box_counting (initial_boxes : ℕ) (boxes_per_operation : ℕ) (final_nonempty_boxes : ℕ) : 
  initial_boxes = 2013 → 
  boxes_per_operation = 13 → 
  final_nonempty_boxes = 2013 →
  initial_boxes + boxes_per_operation * final_nonempty_boxes = 28182 := by
  sorry

#check box_counting

end NUMINAMATH_CALUDE_box_counting_l1124_112443


namespace NUMINAMATH_CALUDE_bigger_part_is_34_l1124_112438

theorem bigger_part_is_34 (x y : ℝ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) :
  max x y = 34 := by
  sorry

end NUMINAMATH_CALUDE_bigger_part_is_34_l1124_112438


namespace NUMINAMATH_CALUDE_symmetry_line_l1124_112483

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a circle --/
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a point is on a line --/
def onLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if two circles are symmetric with respect to a line --/
def symmetric (c1 c2 : Circle) (l : Line) : Prop :=
  ∀ p : ℝ × ℝ, onCircle p c1 → 
    ∃ q : ℝ × ℝ, onCircle q c2 ∧ onLine ((p.1 + q.1) / 2, (p.2 + q.2) / 2) l

/-- The main theorem --/
theorem symmetry_line : 
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (2, -2), radius := 3 }
  let l : Line := { a := 1, b := -1, c := -2 }
  symmetric c1 c2 l := by sorry

end NUMINAMATH_CALUDE_symmetry_line_l1124_112483


namespace NUMINAMATH_CALUDE_alice_coins_value_l1124_112473

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a half-dollar in cents -/
def half_dollar_value : ℕ := 50

/-- The number of pennies Alice has -/
def num_pennies : ℕ := 2

/-- The number of nickels Alice has -/
def num_nickels : ℕ := 3

/-- The number of dimes Alice has -/
def num_dimes : ℕ := 4

/-- The number of half-dollars Alice has -/
def num_half_dollars : ℕ := 1

/-- The total value of Alice's coins in cents -/
def total_cents : ℕ :=
  num_pennies * penny_value +
  num_nickels * nickel_value +
  num_dimes * dime_value +
  num_half_dollars * half_dollar_value

/-- The value of one dollar in cents -/
def dollar_in_cents : ℕ := 100

theorem alice_coins_value :
  (total_cents : ℚ) / (dollar_in_cents : ℚ) = 107 / 100 := by
  sorry

end NUMINAMATH_CALUDE_alice_coins_value_l1124_112473


namespace NUMINAMATH_CALUDE_valid_outfit_count_l1124_112484

/-- The number of shirts available. -/
def num_shirts : ℕ := 7

/-- The number of pants available. -/
def num_pants : ℕ := 5

/-- The number of hats available. -/
def num_hats : ℕ := 7

/-- The number of colors available for pants. -/
def num_pants_colors : ℕ := 5

/-- The number of colors available for shirts and hats. -/
def num_shirt_hat_colors : ℕ := 7

/-- The number of valid outfit choices. -/
def num_valid_outfits : ℕ := num_shirts * num_pants * num_hats - num_pants_colors

theorem valid_outfit_count : num_valid_outfits = 240 := by
  sorry

end NUMINAMATH_CALUDE_valid_outfit_count_l1124_112484


namespace NUMINAMATH_CALUDE_pen_cost_calculation_l1124_112468

theorem pen_cost_calculation (pack_size : ℕ) (pack_cost : ℚ) (desired_pens : ℕ) : 
  pack_size = 150 → pack_cost = 45 → desired_pens = 3600 →
  (desired_pens : ℚ) * (pack_cost / pack_size) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_calculation_l1124_112468


namespace NUMINAMATH_CALUDE_car_speed_problem_l1124_112431

theorem car_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_fraction : ℝ) 
  (h1 : distance = 720)
  (h2 : original_time = 8)
  (h3 : new_time_fraction = 5/8) :
  let new_time := new_time_fraction * original_time
  let new_speed := distance / new_time
  new_speed = 144 := by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1124_112431


namespace NUMINAMATH_CALUDE_kiras_cat_kibble_l1124_112486

/-- Calculates the amount of kibble Kira initially filled her cat's bowl with. -/
def initial_kibble_amount (eating_rate : ℚ) (time_away : ℚ) (kibble_left : ℚ) : ℚ :=
  (time_away / 4) * eating_rate + kibble_left

/-- Theorem stating that given the conditions, Kira initially filled the bowl with 3 pounds of kibble. -/
theorem kiras_cat_kibble : initial_kibble_amount 1 8 1 = 3 := by
  sorry

#eval initial_kibble_amount 1 8 1

end NUMINAMATH_CALUDE_kiras_cat_kibble_l1124_112486


namespace NUMINAMATH_CALUDE_quartic_polynomial_e_value_l1124_112414

/-- A polynomial of degree 4 with integer coefficients -/
structure QuarticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  e : ℤ

/-- The sum of coefficients of the polynomial -/
def QuarticPolynomial.sumCoeffs (p : QuarticPolynomial) : ℤ :=
  p.a + p.b + p.c + p.e

/-- Predicate for a polynomial having all negative integer roots -/
def hasAllNegativeIntegerRoots (p : QuarticPolynomial) : Prop :=
  ∃ (s₁ s₂ s₃ s₄ : ℕ+), 
    p.a = s₁ + s₂ + s₃ + s₄ ∧
    p.b = s₁*s₂ + s₁*s₃ + s₁*s₄ + s₂*s₃ + s₂*s₄ + s₃*s₄ ∧
    p.c = s₁*s₂*s₃ + s₁*s₂*s₄ + s₁*s₃*s₄ + s₂*s₃*s₄ ∧
    p.e = s₁*s₂*s₃*s₄

theorem quartic_polynomial_e_value (p : QuarticPolynomial) 
  (h1 : hasAllNegativeIntegerRoots p) 
  (h2 : p.sumCoeffs = 2023) : 
  p.e = 1540 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_e_value_l1124_112414


namespace NUMINAMATH_CALUDE_cubic_root_function_l1124_112442

theorem cubic_root_function (k : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = k * x^(1/3)) →
  (∃ y : ℝ, y = 4 * Real.sqrt 3 ∧ 64^(1/3) * k = y) →
  (∃ y : ℝ, y = 2 * Real.sqrt 3 ∧ 8^(1/3) * k = y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_function_l1124_112442


namespace NUMINAMATH_CALUDE_power_of_2_probability_l1124_112426

/-- A number is a four-digit number in base 4 if it's between 1000₄ and 3333₄ inclusive -/
def IsFourDigitBase4 (n : ℕ) : Prop :=
  64 ≤ n ∧ n ≤ 255

/-- The count of four-digit numbers in base 4 -/
def CountFourDigitBase4 : ℕ := 255 - 64 + 1

/-- A number is a power of 2 if its log base 2 is an integer -/
def IsPowerOf2 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- The count of powers of 2 that are four-digit numbers in base 4 -/
def CountPowerOf2FourDigitBase4 : ℕ := 2

/-- The probability of a randomly chosen four-digit number in base 4 being a power of 2 -/
def ProbabilityPowerOf2FourDigitBase4 : ℚ :=
  CountPowerOf2FourDigitBase4 / CountFourDigitBase4

theorem power_of_2_probability :
  ProbabilityPowerOf2FourDigitBase4 = 1 / 96 := by
  sorry

end NUMINAMATH_CALUDE_power_of_2_probability_l1124_112426


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l1124_112455

/-- The total number of pencils after adding more -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Proof that the total number of pencils is 72 -/
theorem pencils_in_drawer : total_pencils 27 45 = 72 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l1124_112455


namespace NUMINAMATH_CALUDE_no_rectangle_solution_l1124_112478

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem no_rectangle_solution : ¬∃ (x y : ℕ), 
  is_prime x ∧ is_prime y ∧ 
  x < y ∧ y < 6 ∧ 
  2 * (x + y) = 21 ∧ 
  x * y = 45 :=
sorry

end NUMINAMATH_CALUDE_no_rectangle_solution_l1124_112478


namespace NUMINAMATH_CALUDE_triangle_side_length_l1124_112497

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = 4 * A ∧  -- Given condition
  a = 30 ∧  -- Given side length
  c = 48 ∧  -- Given side length
  a / Real.sin A = c / Real.sin C ∧  -- Law of Sines
  b / Real.sin B = a / Real.sin A ∧  -- Law of Sines
  ∃ x : ℝ, 4 * x^3 - 4 * x - 8 / 5 = 0 ∧ x = Real.cos A  -- Equation for cosA
  →
  b = 30 * (5 - 20 * Real.sin A ^ 2 + 16 * Real.sin A ^ 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1124_112497


namespace NUMINAMATH_CALUDE_order_of_values_l1124_112400

theorem order_of_values : ∃ (a b c : ℝ),
  a = Real.exp 0.2 - 1 ∧
  b = Real.log 1.2 ∧
  c = Real.tan 0.2 ∧
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_order_of_values_l1124_112400


namespace NUMINAMATH_CALUDE_mode_of_data_set_l1124_112481

def data_set : List Int := [-1, 0, 2, -1, 3]

def mode (l : List α) [DecidableEq α] : Option α :=
  l.argmax (fun x => l.count x)

theorem mode_of_data_set :
  mode data_set = some (-1) := by
  sorry

end NUMINAMATH_CALUDE_mode_of_data_set_l1124_112481


namespace NUMINAMATH_CALUDE_largest_angle_in_3_4_5_ratio_triangle_l1124_112403

theorem largest_angle_in_3_4_5_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    b = (4/3) * a →
    c = (5/3) * a →
    a + b + c = 180 →
    c = 75 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_3_4_5_ratio_triangle_l1124_112403


namespace NUMINAMATH_CALUDE_jolene_bicycle_fundraising_l1124_112425

/-- Proves that Jolene raises enough money to buy the bicycle with some extra --/
theorem jolene_bicycle_fundraising (
  bicycle_cost : ℕ)
  (babysitting_families : ℕ)
  (babysitting_rate : ℕ)
  (car_washing_neighbors : ℕ)
  (car_washing_rate : ℕ)
  (dog_walking_count : ℕ)
  (dog_walking_rate : ℕ)
  (cash_gift : ℕ)
  (h1 : bicycle_cost = 250)
  (h2 : babysitting_families = 4)
  (h3 : babysitting_rate = 30)
  (h4 : car_washing_neighbors = 5)
  (h5 : car_washing_rate = 12)
  (h6 : dog_walking_count = 3)
  (h7 : dog_walking_rate = 15)
  (h8 : cash_gift = 40) :
  let total_raised := babysitting_families * babysitting_rate +
                      car_washing_neighbors * car_washing_rate +
                      dog_walking_count * dog_walking_rate +
                      cash_gift
  ∃ (extra : ℕ), total_raised = 265 ∧ total_raised > bicycle_cost ∧ extra = total_raised - bicycle_cost ∧ extra = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_jolene_bicycle_fundraising_l1124_112425


namespace NUMINAMATH_CALUDE_point_division_theorem_l1124_112448

/-- Given a line segment CD and a point Q on CD such that CQ:QD = 3:5,
    prove that Q = (5/8)*C + (3/8)*D -/
theorem point_division_theorem (C D Q : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D) 
  (h2 : ∃ k : ℝ, k > 0 ∧ (Q - C) = k • (3 • (D - C))) :
  Q = (5/8) • C + (3/8) • D :=
sorry

end NUMINAMATH_CALUDE_point_division_theorem_l1124_112448


namespace NUMINAMATH_CALUDE_range_of_m_l1124_112433

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 < x ∧ x < m + 1}

-- State the theorem
theorem range_of_m (m : ℝ) : (B m ⊆ A) → m ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1124_112433


namespace NUMINAMATH_CALUDE_f_negative_m_value_l1124_112495

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 + x + 9) / (x^2 + 3)

theorem f_negative_m_value (m : ℝ) (h : f m = 10) : f (-m) = -4 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_m_value_l1124_112495


namespace NUMINAMATH_CALUDE_set_A_properties_l1124_112450

def A : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2^k ∧ k ≥ 1}

theorem set_A_properties :
  (∀ a ∈ A, ∀ b : ℕ, b > 0 → b < 2*a - 1 → ¬(2*a ∣ b*(b+1))) ∧
  (∀ a : ℕ, a > 1 → a ∉ A → ∃ b : ℕ, b > 0 ∧ b < 2*a - 1 ∧ (2*a ∣ b*(b+1))) :=
by sorry

end NUMINAMATH_CALUDE_set_A_properties_l1124_112450


namespace NUMINAMATH_CALUDE_polynomial_property_l1124_112487

theorem polynomial_property (P : ℤ → ℤ) (h_poly : ∀ a b : ℤ, ∃ c : ℤ, P a - P b = c * (a - b)) :
  P 1 = 2019 →
  P 2019 = 1 →
  ∃ k : ℤ, P k = k →
  k = 1010 :=
sorry

end NUMINAMATH_CALUDE_polynomial_property_l1124_112487


namespace NUMINAMATH_CALUDE_abc_product_values_l1124_112420

theorem abc_product_values (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a + 1/b = 5) (eq2 : b + 1/c = 2) (eq3 : c + 1/a = 8/3) :
  a * b * c = 1 ∨ a * b * c = 37/3 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_values_l1124_112420


namespace NUMINAMATH_CALUDE_jerry_thermostat_problem_l1124_112485

/-- Calculates the final temperature after a series of adjustments --/
def finalTemperature (initial : ℝ) : ℝ :=
  let doubled := initial * 2
  let afterDad := doubled - 30
  let afterMom := afterDad * 0.7  -- Reducing by 30% is equivalent to multiplying by 0.7
  let final := afterMom + 24
  final

/-- Theorem stating that the final temperature is 59 degrees --/
theorem jerry_thermostat_problem : finalTemperature 40 = 59 := by
  sorry

end NUMINAMATH_CALUDE_jerry_thermostat_problem_l1124_112485


namespace NUMINAMATH_CALUDE_tabitha_money_to_mom_l1124_112474

/-- The amount of money Tabitha gave her mom -/
def money_given_to_mom (initial_amount : ℚ) (item_cost : ℚ) (num_items : ℕ) (final_amount : ℚ) : ℚ :=
  initial_amount - 2 * (final_amount + item_cost * num_items)

/-- Theorem stating the amount of money Tabitha gave her mom -/
theorem tabitha_money_to_mom :
  money_given_to_mom 25 0.5 5 6 = 8 := by
  sorry

#eval money_given_to_mom 25 0.5 5 6

end NUMINAMATH_CALUDE_tabitha_money_to_mom_l1124_112474


namespace NUMINAMATH_CALUDE_base4_multiplication_division_l1124_112490

-- Define a function to convert from base 4 to base 10
def base4ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 4
def base10ToBase4 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem base4_multiplication_division :
  base10ToBase4 (base4ToBase10 132 * base4ToBase10 22 / base4ToBase10 3) = 154 := by sorry

end NUMINAMATH_CALUDE_base4_multiplication_division_l1124_112490


namespace NUMINAMATH_CALUDE_count_numbers_with_6_or_7_correct_l1124_112470

/-- The number of integers from 1 to 729 (inclusive) in base 9 that contain at least one digit 6 or 7 -/
def count_numbers_with_6_or_7 : ℕ := 386

/-- The total number of integers we're considering -/
def total_numbers : ℕ := 729

/-- The base of the number system we're using -/
def base : ℕ := 9

/-- The number of digits available that are neither 6 nor 7 -/
def digits_without_6_or_7 : ℕ := 7

theorem count_numbers_with_6_or_7_correct :
  count_numbers_with_6_or_7 = total_numbers - digits_without_6_or_7^3 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_6_or_7_correct_l1124_112470


namespace NUMINAMATH_CALUDE_min_value_inequality_l1124_112423

theorem min_value_inequality (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a * b * c = 1) :
  1 / (a^2 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1124_112423


namespace NUMINAMATH_CALUDE_trapezoid_area_division_l1124_112447

/-- Represents a trapezoid with a diagonal and a parallel line -/
structure Trapezoid where
  /-- The ratio in which the diagonal divides the area -/
  diagonal_ratio : Rat
  /-- The ratio in which the parallel line divides the area -/
  parallel_line_ratio : Rat

/-- Theorem about area division in a specific trapezoid -/
theorem trapezoid_area_division (T : Trapezoid) 
  (h : T.diagonal_ratio = 3 / 7) : 
  T.parallel_line_ratio = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_division_l1124_112447


namespace NUMINAMATH_CALUDE_cube_root_floor_product_limit_l1124_112496

def cube_root_floor_product (n : ℕ) : ℚ :=
  (Finset.range n).prod (λ i => ⌊(3 * i + 1 : ℚ)^(1/3)⌋) /
  (Finset.range n).prod (λ i => ⌊(3 * i + 2 : ℚ)^(1/3)⌋)

theorem cube_root_floor_product_limit : 
  cube_root_floor_product 167 = 1/8 := by sorry

end NUMINAMATH_CALUDE_cube_root_floor_product_limit_l1124_112496


namespace NUMINAMATH_CALUDE_inequality_assignment_exists_l1124_112413

/-- Represents the inequality symbols on even-positioned cards -/
def InequalitySequence := Fin 50 → Bool

/-- Represents the assignment of numbers to odd-positioned cards -/
def NumberAssignment := Fin 51 → Fin 51

/-- Checks if a number assignment satisfies the given inequality sequence -/
def is_valid_assignment (ineq : InequalitySequence) (assign : NumberAssignment) : Prop :=
  ∀ i : Fin 50, 
    (ineq i = true → assign i < assign (i + 1)) ∧
    (ineq i = false → assign i > assign (i + 1))

/-- The main theorem stating that a valid assignment always exists -/
theorem inequality_assignment_exists (ineq : InequalitySequence) :
  ∃ (assign : NumberAssignment), is_valid_assignment ineq assign ∧ Function.Bijective assign :=
sorry

end NUMINAMATH_CALUDE_inequality_assignment_exists_l1124_112413


namespace NUMINAMATH_CALUDE_polynomial_sum_l1124_112482

-- Define the polynomial g(x)
def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) :
  g a b c d (1 + I) = 0 → g a b c d (3*I) = 0 → a + b + c + d = 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l1124_112482


namespace NUMINAMATH_CALUDE_min_value_binomial_distribution_l1124_112418

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- The expected value of a binomial distribution -/
def expectedValue (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: The minimum value of 1/p + 1/q for a binomial distribution
    with E(X) = 4 and D(X) = q is 9/4 -/
theorem min_value_binomial_distribution 
  (X : BinomialDistribution) 
  (h_exp : expectedValue X = 4)
  (h_var : variance X = q)
  : (1 / X.p + 1 / q) ≥ 9/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_binomial_distribution_l1124_112418


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l1124_112461

def line1 (x : ℝ) : ℝ := -x
def line2 (x : ℝ) : ℝ := 2*x - 1

def intersection_point : ℝ × ℝ :=
  let x := 1
  let y := -1
  (x, y)

theorem intersection_in_fourth_quadrant :
  let (x, y) := intersection_point
  x > 0 ∧ y < 0 ∧ line1 x = y ∧ line2 x = y :=
sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l1124_112461


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1124_112415

-- Define the quadratic equation
def quadratic_equation (m n x : ℝ) : Prop := x^2 + m*x + n = 0

-- Define the condition for two real roots
def has_two_real_roots (m n : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m n x₁ ∧ quadratic_equation m n x₂

-- Define the condition for negative roots
def has_negative_roots (m n : ℝ) : Prop := ∀ x : ℝ, quadratic_equation m n x → x < 0

-- Define the inequality
def inequality (m n t : ℝ) : Prop := t ≤ (m-1)^2 + (n-1)^2 + (m-n)^2

-- Theorem statement
theorem quadratic_equation_properties :
  ∀ m n : ℝ, has_two_real_roots m n →
  (∃ t : ℝ, (n = 3 - m ∧ has_negative_roots m n) → 2 ≤ m ∧ m < 3) ∧
  (∃ t_max : ℝ, t_max = 9/8 ∧ ∀ t : ℝ, inequality m n t → t ≤ t_max) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1124_112415


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l1124_112456

theorem fractional_inequality_solution_set (x : ℝ) : 
  (x - 1) / (x + 2) > 1 ↔ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l1124_112456


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_three_exists_141_max_141_solution_l1124_112428

theorem greatest_integer_with_gcf_three (n : ℕ) : n < 150 ∧ Nat.gcd n 24 = 3 → n ≤ 141 :=
by
  sorry

theorem exists_141 : 141 < 150 ∧ Nat.gcd 141 24 = 3 :=
by
  sorry

theorem max_141 : ∀ m, m < 150 ∧ Nat.gcd m 24 = 3 → m ≤ 141 :=
by
  sorry

theorem solution : (∃ n, n < 150 ∧ Nat.gcd n 24 = 3 ∧ ∀ m, m < 150 ∧ Nat.gcd m 24 = 3 → m ≤ n) ∧
                   (∀ n, n < 150 ∧ Nat.gcd n 24 = 3 ∧ ∀ m, m < 150 ∧ Nat.gcd m 24 = 3 → m ≤ n → n = 141) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_three_exists_141_max_141_solution_l1124_112428


namespace NUMINAMATH_CALUDE_seating_probability_is_two_sevenths_l1124_112411

/-- The number of boys to be seated -/
def num_boys : ℕ := 5

/-- The number of girls to be seated -/
def num_girls : ℕ := 6

/-- The total number of chairs -/
def total_chairs : ℕ := 11

/-- The probability of seating boys and girls with the given condition -/
def seating_probability : ℚ :=
  2 / 7

/-- Theorem stating that the probability of seating boys and girls
    such that there are no more boys than girls at any point is 2/7 -/
theorem seating_probability_is_two_sevenths :
  seating_probability = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_seating_probability_is_two_sevenths_l1124_112411


namespace NUMINAMATH_CALUDE_total_commute_time_is_16_l1124_112401

-- Define the time it takes to walk and bike to work
def walk_time : ℕ := 2
def bike_time : ℕ := 1

-- Define the number of times Roque walks and bikes to work per week
def walk_trips : ℕ := 3
def bike_trips : ℕ := 2

-- Define the total commuting time
def total_commute_time : ℕ := 
  2 * (walk_time * walk_trips + bike_time * bike_trips)

-- Theorem statement
theorem total_commute_time_is_16 : total_commute_time = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_commute_time_is_16_l1124_112401
