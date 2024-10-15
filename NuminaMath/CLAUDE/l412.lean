import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l412_41218

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 3) + 2 / (y + 3) = 1 / 4) :
  2 * x + 3 * y ≥ 16 * Real.sqrt 3 - 16 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧
    1 / (x₀ + 3) + 2 / (y₀ + 3) = 1 / 4 ∧
    2 * x₀ + 3 * y₀ = 16 * Real.sqrt 3 - 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l412_41218


namespace NUMINAMATH_CALUDE_polynomial_transformation_l412_41239

theorem polynomial_transformation (x y : ℝ) (hx : x ≠ 0) :
  y = x + 1/x →
  (x^4 - x^3 - 6*x^2 - x + 1 = 0) ↔ (x^2*(y^2 - y - 8) = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l412_41239


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l412_41214

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 3^5 - 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l412_41214


namespace NUMINAMATH_CALUDE_mady_balls_equals_ternary_sum_l412_41267

/-- Represents the state of a box in Mady's game -/
inductive BoxState
| Empty : BoxState
| OneBall : BoxState
| TwoBalls : BoxState

/-- Converts a natural number to its ternary (base 3) representation -/
def toTernary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then []
      else (m % 3) :: aux (m / 3)
    aux n |>.reverse

/-- Simulates Mady's ball-placing process for a given number of steps -/
def madyProcess (steps : ℕ) : List BoxState :=
  sorry

/-- Counts the number of balls in the final state -/
def countBalls (state : List BoxState) : ℕ :=
  sorry

/-- The main theorem: The number of balls after 2023 steps equals the sum of ternary digits of 2023 -/
theorem mady_balls_equals_ternary_sum :
  countBalls (madyProcess 2023) = (toTernary 2023).sum := by
  sorry

end NUMINAMATH_CALUDE_mady_balls_equals_ternary_sum_l412_41267


namespace NUMINAMATH_CALUDE_solution_in_interval_l412_41232

theorem solution_in_interval (x₀ : ℝ) : 
  (Real.log x₀ + x₀ - 3 = 0) → (2 < x₀ ∧ x₀ < 2.5) := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l412_41232


namespace NUMINAMATH_CALUDE_set_A_equality_l412_41259

def A : Set ℕ := {x | x < 3}

theorem set_A_equality : A = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_A_equality_l412_41259


namespace NUMINAMATH_CALUDE_unique_solution_for_system_l412_41291

theorem unique_solution_for_system :
  ∃! (x y : ℝ), (x + y = (7 - x) + (7 - y)) ∧ (x - y = (x + 1) + (y + 1)) ∧ x = 8 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_system_l412_41291


namespace NUMINAMATH_CALUDE_cord_lengths_l412_41250

theorem cord_lengths (total_length : ℝ) (a b c : ℝ) : 
  total_length = 60 → -- Total length is 60 decimeters
  a + b + c = total_length * 10 → -- Sum of parts equals total length in cm
  b = a + 1 → -- Second part is 1 cm more than first
  c = b + 1 → -- Third part is 1 cm more than second
  (a, b, c) = (199, 200, 201) := by sorry

end NUMINAMATH_CALUDE_cord_lengths_l412_41250


namespace NUMINAMATH_CALUDE_cos_75_cos_15_plus_sin_75_sin_15_l412_41243

theorem cos_75_cos_15_plus_sin_75_sin_15 :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) +
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_cos_15_plus_sin_75_sin_15_l412_41243


namespace NUMINAMATH_CALUDE_constant_term_expansion_l412_41215

theorem constant_term_expansion (a : ℝ) : 
  (∃ k : ℝ, k = -40 ∧ k = (a * 40 + 1 * 1)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l412_41215


namespace NUMINAMATH_CALUDE_four_integers_with_many_divisors_l412_41231

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem four_integers_with_many_divisors :
  ∃ (a b c d : ℕ),
    a > 0 ∧ a ≤ 70000 ∧ count_divisors a > 100 ∧
    b > 0 ∧ b ≤ 70000 ∧ count_divisors b > 100 ∧
    c > 0 ∧ c ≤ 70000 ∧ count_divisors c > 100 ∧
    d > 0 ∧ d ≤ 70000 ∧ count_divisors d > 100 :=
by
  use 69300, 50400, 60480, 55440
  sorry

end NUMINAMATH_CALUDE_four_integers_with_many_divisors_l412_41231


namespace NUMINAMATH_CALUDE_base_7_addition_l412_41272

/-- Addition in base 7 -/
def add_base_7 (a b : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 7 -/
def to_base_7 (n : ℕ) : ℕ := sorry

/-- Conversion from base 7 to base 10 -/
def from_base_7 (n : ℕ) : ℕ := sorry

theorem base_7_addition : add_base_7 (from_base_7 25) (from_base_7 256) = from_base_7 544 := by
  sorry

end NUMINAMATH_CALUDE_base_7_addition_l412_41272


namespace NUMINAMATH_CALUDE_trapezoid_properties_l412_41298

-- Define the trapezoid and its properties
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  h : ℝ
  parallel_AB_CD : AB ≠ CD

-- Define the midpoints
def midpoint_M (t : Trapezoid) : ℝ × ℝ := sorry
def midpoint_N (t : Trapezoid) : ℝ × ℝ := sorry
def midpoint_P (t : Trapezoid) : ℝ × ℝ := sorry

-- Define the length of MN
def length_MN (t : Trapezoid) : ℝ := sorry

-- Define the area of triangle MNP
def area_MNP (t : Trapezoid) : ℝ := sorry

-- Theorem statement
theorem trapezoid_properties (t : Trapezoid) 
  (h_AB : t.AB = 15) 
  (h_CD : t.CD = 24) 
  (h_h : t.h = 14) : 
  length_MN t = 4.5 ∧ area_MNP t = 15.75 := by sorry

end NUMINAMATH_CALUDE_trapezoid_properties_l412_41298


namespace NUMINAMATH_CALUDE_remainder_of_sum_times_three_div_six_l412_41234

/-- The sum of an arithmetic sequence with first term a, common difference d, and n terms -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- The number of terms in the arithmetic sequence with first term 2, common difference 6, and last term 266 -/
def n : ℕ := 45

/-- The first term of the arithmetic sequence -/
def a : ℕ := 2

/-- The common difference of the arithmetic sequence -/
def d : ℕ := 6

/-- The last term of the arithmetic sequence -/
def last_term : ℕ := 266

theorem remainder_of_sum_times_three_div_six :
  (3 * arithmetic_sum a d n) % 6 = 0 :=
sorry

end NUMINAMATH_CALUDE_remainder_of_sum_times_three_div_six_l412_41234


namespace NUMINAMATH_CALUDE_equal_surface_area_implies_L_value_l412_41297

/-- Given a cube with edge length 30 and a rectangular solid with edge lengths 20, 30, and L,
    if their surface areas are equal, then L = 42. -/
theorem equal_surface_area_implies_L_value (L : ℝ) : 
  (6 * 30 * 30 = 2 * 20 * 30 + 2 * 20 * L + 2 * 30 * L) → L = 42 := by
  sorry

#check equal_surface_area_implies_L_value

end NUMINAMATH_CALUDE_equal_surface_area_implies_L_value_l412_41297


namespace NUMINAMATH_CALUDE_nth_equation_proof_l412_41245

theorem nth_equation_proof (n : ℕ) : 
  n^2 + (n+1)^2 = (n*(n+1)+1)^2 - (n*(n+1))^2 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_proof_l412_41245


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l412_41265

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l412_41265


namespace NUMINAMATH_CALUDE_divisor_count_equality_l412_41247

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Theorem: For all positive integers n and k, there exists a positive integer s
    such that the number of positive divisors of sn equals the number of positive divisors of sk
    if and only if n does not divide k and k does not divide n -/
theorem divisor_count_equality (n k : ℕ+) :
  (∃ s : ℕ+, num_divisors (s * n) = num_divisors (s * k)) ↔ (¬(n ∣ k) ∧ ¬(k ∣ n)) :=
sorry

end NUMINAMATH_CALUDE_divisor_count_equality_l412_41247


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l412_41264

/-- The number of revolutions needed for the second horse to cover the same distance as the first horse on a merry-go-round. -/
theorem merry_go_round_revolutions (r₁ r₂ : ℝ) (n₁ : ℕ) (h₁ : r₁ = 30) (h₂ : r₂ = 10) (h₃ : n₁ = 25) :
  (r₁ * n₁ : ℝ) / r₂ = 75 := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l412_41264


namespace NUMINAMATH_CALUDE_hundred_guests_at_reunions_l412_41206

/-- The number of guests attending at least one of two reunions -/
def guests_at_reunions (oates_guests yellow_guests both_guests : ℕ) : ℕ :=
  oates_guests + yellow_guests - both_guests

/-- Theorem: Given the conditions of the problem, 100 guests attend at least one reunion -/
theorem hundred_guests_at_reunions :
  let oates_guests : ℕ := 42
  let yellow_guests : ℕ := 65
  let both_guests : ℕ := 7
  guests_at_reunions oates_guests yellow_guests both_guests = 100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_guests_at_reunions_l412_41206


namespace NUMINAMATH_CALUDE_hannah_measuring_spoons_l412_41223

/-- The number of measuring spoons Hannah bought -/
def num_measuring_spoons : ℕ := 2

/-- The price of each cookie in dollars -/
def cookie_price : ℚ := 4/5

/-- The number of cookies sold -/
def num_cookies : ℕ := 40

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The number of cupcakes sold -/
def num_cupcakes : ℕ := 30

/-- The price of each measuring spoon in dollars -/
def spoon_price : ℚ := 13/2

/-- The amount of money left after buying measuring spoons in dollars -/
def money_left : ℚ := 79

theorem hannah_measuring_spoons :
  (num_cookies * cookie_price + num_cupcakes * cupcake_price - money_left) / spoon_price = num_measuring_spoons := by
  sorry

end NUMINAMATH_CALUDE_hannah_measuring_spoons_l412_41223


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l412_41270

-- Define proposition p
def p : Prop := ∀ (x a : ℝ), x^2 + a*x + a^2 ≥ 0

-- Define proposition q
def q : Prop := ∃ (x₀ : ℕ), x₀ > 0 ∧ 2*x₀^2 - 1 ≤ 0

-- Theorem statement
theorem p_or_q_is_true : p ∨ q := by sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l412_41270


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_P_x_eq_x_l412_41248

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- Property: for any integers a and b, b - a divides P(b) - P(a) -/
def IntegerCoefficientProperty (P : IntPolynomial) : Prop :=
  ∀ a b : ℤ, (b - a) ∣ (P b - P a)

theorem no_integer_solutions_for_P_x_eq_x
  (P : IntPolynomial)
  (h_int_coeff : IntegerCoefficientProperty P)
  (h_P_3 : P 3 = 4)
  (h_P_4 : P 4 = 3) :
  ¬∃ x : ℤ, P x = x :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_P_x_eq_x_l412_41248


namespace NUMINAMATH_CALUDE_circle_point_selection_eq_258_l412_41240

/-- The number of ways to select 8 points from 24 equally spaced points on a circle,
    such that no two selected points have an arc length of 3 or 8 between them. -/
def circle_point_selection : ℕ :=
  2^8 + 2

/-- Proves that the number of valid selections is 258. -/
theorem circle_point_selection_eq_258 : circle_point_selection = 258 := by
  sorry

end NUMINAMATH_CALUDE_circle_point_selection_eq_258_l412_41240


namespace NUMINAMATH_CALUDE_quadratic_function_property_l412_41276

theorem quadratic_function_property (a b : ℝ) (h1 : a ≠ b) : 
  let f := fun x => x^2 + a*x + b
  (f a = f b) → f 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l412_41276


namespace NUMINAMATH_CALUDE_basic_computer_price_l412_41260

/-- Given the price of a basic computer and a printer, prove that the basic computer costs $2000 -/
theorem basic_computer_price
  (basic_computer printer : ℝ)
  (total_price : basic_computer + printer = 2500)
  (enhanced_total : ∃ (enhanced_total : ℝ), enhanced_total = basic_computer + 500 + printer)
  (printer_ratio : printer = (1/6) * (basic_computer + 500 + printer)) :
  basic_computer = 2000 := by
  sorry

end NUMINAMATH_CALUDE_basic_computer_price_l412_41260


namespace NUMINAMATH_CALUDE_inequality_proof_l412_41256

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  1 ≤ (x / (1 + y*z) + y / (1 + z*x) + z / (1 + x*y)) ∧ 
  (x / (1 + y*z) + y / (1 + z*x) + z / (1 + x*y)) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l412_41256


namespace NUMINAMATH_CALUDE_system_solutions_l412_41283

/-- The system of equations has only two solutions -/
theorem system_solutions (x y z : ℝ) : 
  (2 * x^2 / (1 + x^2) = y ∧ 
   2 * y^2 / (1 + y^2) = z ∧ 
   2 * z^2 / (1 + z^2) = x) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l412_41283


namespace NUMINAMATH_CALUDE_range_of_a_l412_41224

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |1/2 * x^3 - a*x| ≤ 1) ↔ -1/2 ≤ a ∧ a ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l412_41224


namespace NUMINAMATH_CALUDE_total_octopus_legs_l412_41274

/-- The number of legs an octopus has -/
def legs_per_octopus : ℕ := 8

/-- The number of octopuses Carson saw -/
def octopuses_seen : ℕ := 5

/-- The total number of octopus legs Carson saw -/
def total_legs : ℕ := octopuses_seen * legs_per_octopus

theorem total_octopus_legs : total_legs = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_octopus_legs_l412_41274


namespace NUMINAMATH_CALUDE_probability_three_unused_rockets_expected_targets_hit_l412_41211

/-- Represents a rocket artillery system -/
structure RocketSystem where
  totalRockets : ℕ
  maxShotsPerTarget : ℕ
  hitProbability : ℝ

/-- Calculates the probability of having exactly 3 unused rockets after firing at 5 targets -/
def probabilityThreeUnusedRockets (system : RocketSystem) : ℝ :=
  10 * system.hitProbability^3 * (1 - system.hitProbability)^2

/-- Calculates the expected number of targets hit when firing at 9 targets -/
def expectedTargetsHit (system : RocketSystem) : ℝ :=
  10 * system.hitProbability - system.hitProbability^10

/-- Theorem stating the probability of having exactly 3 unused rockets after firing at 5 targets -/
theorem probability_three_unused_rockets 
  (system : RocketSystem) 
  (h1 : system.totalRockets = 10) 
  (h2 : system.maxShotsPerTarget = 2) :
  probabilityThreeUnusedRockets system = 10 * system.hitProbability^3 * (1 - system.hitProbability)^2 := by
  sorry

/-- Theorem stating the expected number of targets hit when firing at 9 targets -/
theorem expected_targets_hit 
  (system : RocketSystem) 
  (h1 : system.totalRockets = 10) 
  (h2 : system.maxShotsPerTarget = 2) :
  expectedTargetsHit system = 10 * system.hitProbability - system.hitProbability^10 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_unused_rockets_expected_targets_hit_l412_41211


namespace NUMINAMATH_CALUDE_tv_sales_decrease_l412_41253

/-- Proves that a 70% price increase and 36% revenue increase results in a 20% sales decrease -/
theorem tv_sales_decrease (initial_price initial_quantity : ℝ) 
  (initial_price_positive : initial_price > 0)
  (initial_quantity_positive : initial_quantity > 0) : 
  let new_price := 1.7 * initial_price
  let new_revenue := 1.36 * (initial_price * initial_quantity)
  let new_quantity := new_revenue / new_price
  (initial_quantity - new_quantity) / initial_quantity = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_tv_sales_decrease_l412_41253


namespace NUMINAMATH_CALUDE_sum_is_composite_l412_41208

theorem sum_is_composite (a b c d : ℕ+) 
  (h : a^2 + b^2 + a*b = c^2 + d^2 + c*d) : 
  ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ (a : ℕ) + b + c + d = k * m :=
sorry

end NUMINAMATH_CALUDE_sum_is_composite_l412_41208


namespace NUMINAMATH_CALUDE_capri_sun_cost_per_pouch_l412_41203

/-- Calculates the cost per pouch in cents given the number of boxes, pouches per box, and total cost in dollars. -/
def cost_per_pouch (boxes : ℕ) (pouches_per_box : ℕ) (total_cost_dollars : ℕ) : ℕ :=
  (total_cost_dollars * 100) / (boxes * pouches_per_box)

/-- Proves that for 10 boxes with 6 pouches each, costing $12 in total, each pouch costs 20 cents. -/
theorem capri_sun_cost_per_pouch :
  cost_per_pouch 10 6 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_capri_sun_cost_per_pouch_l412_41203


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l412_41275

/-- Given a triangle ABC with vertices A(x₁, y₁), B(x₂, y₂), C(x₃, y₃),
    and points E on AC and F on AB such that AE:EC = n:l and AF:FB = m:l,
    prove that the intersection point P of BE and CF has coordinates
    ((lx₁ + mx₂ + nx₃)/(l + m + n), (ly₁ + my₂ + ny₃)/(l + m + n)) -/
theorem intersection_point_coordinates
  (x₁ y₁ x₂ y₂ x₃ y₃ l m n : ℝ)
  (h₁ : m ≠ -l)
  (h₂ : n ≠ -l)
  (h₃ : l + m + n ≠ 0) :
  let A := (x₁, y₁)
  let B := (x₂, y₂)
  let C := (x₃, y₃)
  let E := ((l * x₁ + n * x₃) / (l + n), (l * y₁ + n * y₃) / (l + n))
  let F := ((l * x₁ + m * x₂) / (l + m), (l * y₁ + m * y₂) / (l + m))
  let P := ((l * x₁ + m * x₂ + n * x₃) / (l + m + n), (l * y₁ + m * y₂ + n * y₃) / (l + m + n))
  ∃ (t : ℝ), (P.1 - E.1) / (B.1 - E.1) = t ∧ (P.2 - E.2) / (B.2 - E.2) = t ∧
             (P.1 - F.1) / (C.1 - F.1) = (1 - t) ∧ (P.2 - F.2) / (C.2 - F.2) = (1 - t) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l412_41275


namespace NUMINAMATH_CALUDE_small_pizza_has_four_slices_l412_41226

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := sorry

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 8

/-- The number of small pizzas purchased -/
def small_pizzas_bought : ℕ := 3

/-- The number of large pizzas purchased -/
def large_pizzas_bought : ℕ := 2

/-- The number of slices George eats -/
def george_slices : ℕ := 3

/-- The number of slices Bob eats -/
def bob_slices : ℕ := george_slices + 1

/-- The number of slices Susie eats -/
def susie_slices : ℕ := bob_slices / 2

/-- The number of slices Bill eats -/
def bill_slices : ℕ := 3

/-- The number of slices Fred eats -/
def fred_slices : ℕ := 3

/-- The number of slices Mark eats -/
def mark_slices : ℕ := 3

/-- The number of slices left over -/
def leftover_slices : ℕ := 10

theorem small_pizza_has_four_slices : small_pizza_slices = 4 := by
  sorry

end NUMINAMATH_CALUDE_small_pizza_has_four_slices_l412_41226


namespace NUMINAMATH_CALUDE_jennifer_total_distance_l412_41279

/-- Represents the distances and changes for Jennifer's museum visits -/
structure MuseumDistances where
  first_museum : ℕ
  second_museum : ℕ
  cultural_center : ℕ
  traffic_increase : ℕ
  bus_decrease : ℕ
  bicycle_decrease : ℕ

/-- Calculates the total distance for Jennifer's museum visits -/
def total_distance (d : MuseumDistances) : ℕ :=
  (d.second_museum + d.traffic_increase) + 
  (d.cultural_center - d.bus_decrease) + 
  (d.first_museum - d.bicycle_decrease)

/-- Theorem stating that Jennifer's total distance is 32 miles -/
theorem jennifer_total_distance :
  ∀ d : MuseumDistances,
  d.first_museum = 5 ∧
  d.second_museum = 15 ∧
  d.cultural_center = 10 ∧
  d.traffic_increase = 5 ∧
  d.bus_decrease = 2 ∧
  d.bicycle_decrease = 1 →
  total_distance d = 32 :=
by sorry

end NUMINAMATH_CALUDE_jennifer_total_distance_l412_41279


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l412_41236

/-- Proves that for a hyperbola x²/a² - y²/b² = 1 with a > b, 
    if the angle between its asymptotes is 45°, then a/b = 1 + √2 -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (Real.pi / 4 = Real.arctan ((b/a - (-b/a)) / (1 + (b/a) * (-b/a)))) →
  a / b = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l412_41236


namespace NUMINAMATH_CALUDE_bus_trip_speed_l412_41287

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) 
  (h1 : distance = 280)
  (h2 : speed_increase = 5)
  (h3 : time_decrease = 1)
  (h4 : distance / speed - time_decrease = distance / (speed + speed_increase)) :
  speed = 35 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l412_41287


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l412_41284

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-3, -2)) : 
  |P.2| = 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l412_41284


namespace NUMINAMATH_CALUDE_larger_cross_section_distance_l412_41262

/-- Represents a right octagonal pyramid -/
structure RightOctagonalPyramid where
  -- We don't need to define the full structure, just what's necessary for the problem

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  area : ℝ
  distance_from_apex : ℝ

theorem larger_cross_section_distance
  (pyramid : RightOctagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h_area1 : cs1.area = 256 * Real.sqrt 2)
  (h_area2 : cs2.area = 576 * Real.sqrt 2)
  (h_distance : cs2.distance_from_apex - cs1.distance_from_apex = 10)
  (h_parallel : True)  -- Assuming parallel, but not used in the proof
  (h_larger : cs2.area > cs1.area) :
  cs2.distance_from_apex = 30 := by
sorry


end NUMINAMATH_CALUDE_larger_cross_section_distance_l412_41262


namespace NUMINAMATH_CALUDE_equation_solution_l412_41205

theorem equation_solution :
  let f : ℂ → ℂ := λ x => x^3 + 4*x^2*Real.sqrt 3 + 12*x + 4*Real.sqrt 3 + x^2 - 1
  ∀ x : ℂ, f x = 0 ↔ x = 0 ∨ x = -Real.sqrt 3 ∨ x = (-Real.sqrt 3 + Complex.I)/2 ∨ x = (-Real.sqrt 3 - Complex.I)/2 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l412_41205


namespace NUMINAMATH_CALUDE_max_value_of_f_l412_41233

/-- The function f(x) = -5x^2 + 25x - 15 -/
def f (x : ℝ) : ℝ := -5 * x^2 + 25 * x - 15

/-- Theorem stating that the maximum value of f(x) is 750 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 750 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l412_41233


namespace NUMINAMATH_CALUDE_multiplication_addition_difference_l412_41225

theorem multiplication_addition_difference : 
  (2 : ℚ) / 3 * (3 : ℚ) / 2 - ((2 : ℚ) / 3 + (3 : ℚ) / 2) = -(7 : ℚ) / 6 :=
by sorry

end NUMINAMATH_CALUDE_multiplication_addition_difference_l412_41225


namespace NUMINAMATH_CALUDE_opposite_numbers_with_equation_l412_41249

theorem opposite_numbers_with_equation (x y : ℝ) : 
  x + y = 0 → (x + 2)^2 - (y + 2)^2 = 4 → x = 1/2 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_with_equation_l412_41249


namespace NUMINAMATH_CALUDE_triangle_theorem_l412_41271

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c) 
  (h2 : t.c = 3 * t.a) : 
  t.B = π/3 ∧ Real.sin t.A = Real.sqrt 21 / 14 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l412_41271


namespace NUMINAMATH_CALUDE_earliest_retirement_is_2009_l412_41294

/-- Rule of 70 provision: An employee can retire when age + years of employment ≥ 70 -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year the employee was hired -/
def hire_year : ℕ := 1990

/-- The employee's age when hired -/
def hire_age : ℕ := 32

/-- The earliest retirement year satisfies the rule of 70 -/
def earliest_retirement_year (year : ℕ) : Prop :=
  rule_of_70 (hire_age + (year - hire_year)) (year - hire_year) ∧
  ∀ y < year, ¬rule_of_70 (hire_age + (y - hire_year)) (y - hire_year)

/-- Theorem: The earliest retirement year for the employee is 2009 -/
theorem earliest_retirement_is_2009 : earliest_retirement_year 2009 := by
  sorry

end NUMINAMATH_CALUDE_earliest_retirement_is_2009_l412_41294


namespace NUMINAMATH_CALUDE_total_is_99_l412_41213

/-- The total number of ducks and ducklings in a flock --/
def total_ducks_and_ducklings : ℕ → ℕ → ℕ → ℕ := fun a b c => 
  (2 + 6 + 9) + (2 * a + 6 * b + 9 * c)

/-- Theorem: The total number of ducks and ducklings is 99 --/
theorem total_is_99 : total_ducks_and_ducklings 5 3 6 = 99 := by
  sorry

end NUMINAMATH_CALUDE_total_is_99_l412_41213


namespace NUMINAMATH_CALUDE_leftHandedJazzLoversCount_l412_41216

/-- Represents a club with members having different characteristics -/
structure Club where
  total : ℕ
  leftHanded : ℕ
  jazzLovers : ℕ
  rightHandedNonJazz : ℕ

/-- The number of left-handed jazz lovers in the club -/
def leftHandedJazzLovers (c : Club) : ℕ :=
  c.total - (c.leftHanded + c.jazzLovers - c.rightHandedNonJazz)

/-- Theorem stating the number of left-handed jazz lovers in the given club -/
theorem leftHandedJazzLoversCount (c : Club) 
  (h1 : c.total = 20)
  (h2 : c.leftHanded = 8)
  (h3 : c.jazzLovers = 15)
  (h4 : c.rightHandedNonJazz = 2) :
  leftHandedJazzLovers c = 5 := by
  sorry

#eval leftHandedJazzLovers { total := 20, leftHanded := 8, jazzLovers := 15, rightHandedNonJazz := 2 }

end NUMINAMATH_CALUDE_leftHandedJazzLoversCount_l412_41216


namespace NUMINAMATH_CALUDE_max_prob_with_highest_prob_player_second_l412_41258

/-- Represents a chess player with a winning probability -/
structure Player where
  winProb : ℝ

/-- Represents the order of games played -/
inductive GameOrder
| ABC
| ACB
| BAC
| BCA
| CAB
| CBA

/-- Calculates the probability of winning two consecutive games given a game order -/
def probTwoConsecutiveWins (p₁ p₂ p₃ : ℝ) (order : GameOrder) : ℝ :=
  match order with
  | GameOrder.ABC => 2 * (p₁ * p₂)
  | GameOrder.ACB => 2 * (p₁ * p₃)
  | GameOrder.BAC => 2 * (p₂ * p₁)
  | GameOrder.BCA => 2 * (p₂ * p₃)
  | GameOrder.CAB => 2 * (p₃ * p₁)
  | GameOrder.CBA => 2 * (p₃ * p₂)

theorem max_prob_with_highest_prob_player_second 
  (A B C : Player) 
  (h₁ : 0 < A.winProb) 
  (h₂ : A.winProb < B.winProb) 
  (h₃ : B.winProb < C.winProb) :
  ∀ (order : GameOrder), 
    probTwoConsecutiveWins A.winProb B.winProb C.winProb order ≤ 
    max (probTwoConsecutiveWins A.winProb B.winProb C.winProb GameOrder.CAB)
        (probTwoConsecutiveWins A.winProb B.winProb C.winProb GameOrder.CBA) :=
by sorry

end NUMINAMATH_CALUDE_max_prob_with_highest_prob_player_second_l412_41258


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l412_41293

def calculate_money_left (initial_amount : ℝ) (candy_bars : ℕ) (chips : ℕ) (soft_drinks : ℕ)
  (candy_bar_price : ℝ) (chips_price : ℝ) (soft_drink_price : ℝ)
  (candy_discount : ℝ) (chips_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let candy_cost := candy_bars * candy_bar_price
  let chips_cost := chips * chips_price
  let soft_drinks_cost := soft_drinks * soft_drink_price
  let total_before_discounts := candy_cost + chips_cost + soft_drinks_cost
  let candy_discount_amount := candy_cost * candy_discount
  let chips_discount_amount := chips_cost * chips_discount
  let total_after_discounts := total_before_discounts - candy_discount_amount - chips_discount_amount
  let tax_amount := total_after_discounts * sales_tax
  let final_cost := total_after_discounts + tax_amount
  initial_amount - final_cost

theorem money_left_after_purchase :
  calculate_money_left 200 25 10 15 3 2.5 1.75 0.1 0.05 0.06 = 75.45 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l412_41293


namespace NUMINAMATH_CALUDE_tangent_slope_range_implies_y_coordinate_range_l412_41289

/-- The curve C defined by y = x^2 - x + 1 -/
def C : ℝ → ℝ := λ x => x^2 - x + 1

/-- The derivative of C -/
def C' : ℝ → ℝ := λ x => 2*x - 1

theorem tangent_slope_range_implies_y_coordinate_range :
  ∀ x y : ℝ,
  y = C x →
  -1 ≤ C' x ∧ C' x ≤ 3 →
  3/4 ≤ y ∧ y ≤ 3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_range_implies_y_coordinate_range_l412_41289


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l412_41229

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal -/
def symmetricYAxis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetric_points_sum (a b : ℝ) :
  symmetricYAxis (a, 3) (4, b) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l412_41229


namespace NUMINAMATH_CALUDE_hyperbola_equation_l412_41230

/-- Given a hyperbola with asymptote x + √3y = 0 and one focus at (4, 0),
    its standard equation is x²/12 - y²/4 = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧
    (x + Real.sqrt 3 * y = 0 → y = -(1 / Real.sqrt 3) * x) ∧  -- Asymptote condition
    c = 4 ∧                                                   -- Focus condition
    c^2 = a^2 + b^2 ∧                                         -- Hyperbola property
    b/a = Real.sqrt 3 / 3) →                                  -- Derived from asymptote
  x^2 / 12 - y^2 / 4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l412_41230


namespace NUMINAMATH_CALUDE_equation_solutions_l412_41207

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)
  let g (x : ℝ) := (x - 3) * (x - 7) * (x - 3)
  { x : ℝ | x ≠ 3 ∧ x ≠ 7 ∧ f x / g x = 1 } = { 3 + Real.sqrt 3, 3 + Real.sqrt 5, 3 - Real.sqrt 5 } :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l412_41207


namespace NUMINAMATH_CALUDE_problem_statement_l412_41281

theorem problem_statement (a b : ℝ) :
  (a + 1)^2 + Real.sqrt (b - 2) = 0 → a - b = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l412_41281


namespace NUMINAMATH_CALUDE_alien_abduction_l412_41246

theorem alien_abduction (P : ℕ) : 
  (80 : ℚ) / 100 * P + 40 = P → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_alien_abduction_l412_41246


namespace NUMINAMATH_CALUDE_last_four_average_l412_41285

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 70 →
  ((list.take 3).sum / 3 : ℝ) = 65 →
  ((list.drop 3).sum / 4 : ℝ) = 73.75 := by
  sorry

end NUMINAMATH_CALUDE_last_four_average_l412_41285


namespace NUMINAMATH_CALUDE_distinct_naturals_reciprocal_sum_l412_41292

theorem distinct_naturals_reciprocal_sum (x y z : ℕ) : 
  x < y ∧ y < z ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  ∃ (a : ℕ), (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = a
  →
  x = 2 ∧ y = 3 ∧ z = 6 := by
sorry

end NUMINAMATH_CALUDE_distinct_naturals_reciprocal_sum_l412_41292


namespace NUMINAMATH_CALUDE_locust_jump_equivalence_l412_41210

/-- A type representing the position of a locust on a line -/
def Position := ℝ

/-- A type representing a configuration of locusts -/
def Configuration := List Position

/-- A function that represents a jump to the right -/
def jumpRight (config : Configuration) (i j : ℕ) : Configuration :=
  sorry

/-- A function that represents a jump to the left -/
def jumpLeft (config : Configuration) (i j : ℕ) : Configuration :=
  sorry

/-- A predicate that checks if all locusts are 1 unit apart -/
def isUnitApart (config : Configuration) : Prop :=
  sorry

theorem locust_jump_equivalence (initial : Configuration) 
  (h : ∃ (final : Configuration), (∀ i j, jumpRight initial i j = final) ∧ isUnitApart final) :
  ∃ (final : Configuration), (∀ i j, jumpLeft initial i j = final) ∧ isUnitApart final :=
sorry

end NUMINAMATH_CALUDE_locust_jump_equivalence_l412_41210


namespace NUMINAMATH_CALUDE_f_three_point_five_l412_41278

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property that f(x+2) is an odd function
axiom f_odd (x : ℝ) : f (-(x + 2)) = -f (x + 2)

-- Define the property that f(x) = 2x for x ∈ (0,2)
axiom f_linear (x : ℝ) : x > 0 → x < 2 → f x = 2 * x

-- Theorem to prove
theorem f_three_point_five : f 3.5 = -1 := by sorry

end NUMINAMATH_CALUDE_f_three_point_five_l412_41278


namespace NUMINAMATH_CALUDE_class_composition_unique_l412_41252

/-- Represents a pair of numbers written by a student -/
structure Answer :=
  (classmates : Nat)
  (girls : Nat)

/-- Represents the class composition -/
structure ClassComposition :=
  (boys : Nat)
  (girls : Nat)

/-- Checks if an answer is valid given the actual class composition -/
def isValidAnswer (actual : ClassComposition) (answer : Answer) : Prop :=
  (answer.classmates = actual.boys + actual.girls - 1 ∧ 
   (answer.girls = actual.girls ∨ answer.girls = actual.girls + 4 ∨ answer.girls = actual.girls - 4)) ∨
  (answer.girls = actual.girls ∧ 
   (answer.classmates = actual.boys + actual.girls - 1 ∨ 
    answer.classmates = actual.boys + actual.girls + 3 ∨ 
    answer.classmates = actual.boys + actual.girls - 5))

theorem class_composition_unique :
  ∃! comp : ClassComposition,
    isValidAnswer comp ⟨15, 18⟩ ∧
    isValidAnswer comp ⟨15, 10⟩ ∧
    isValidAnswer comp ⟨12, 13⟩ ∧
    comp.boys = 16 ∧
    comp.girls = 14 := by sorry

end NUMINAMATH_CALUDE_class_composition_unique_l412_41252


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l412_41235

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptote y = x -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b = a) : 
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l412_41235


namespace NUMINAMATH_CALUDE_expression_evaluation_l412_41238

theorem expression_evaluation :
  let a : ℝ := 1
  let b : ℝ := 2
  2*(a+b)*(a-b) - (a+b)^2 + a*(2*a+b) = -11 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l412_41238


namespace NUMINAMATH_CALUDE_trigonometric_sum_simplification_l412_41254

open Real BigOperators

theorem trigonometric_sum_simplification (n : ℕ) (α : ℝ) :
  (cos α + ∑ k in Finset.range (n - 1), (n.choose k) * cos ((k + 1) * α) + cos ((n + 1) * α) = 
   2^n * (cos (α / 2))^n * cos ((n + 2) * α / 2)) ∧
  (sin α + ∑ k in Finset.range (n - 1), (n.choose k) * sin ((k + 1) * α) + sin ((n + 1) * α) = 
   2^n * (cos (α / 2))^n * sin ((n + 2) * α / 2)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_simplification_l412_41254


namespace NUMINAMATH_CALUDE_clock_malfunction_proof_l412_41220

/-- Represents a time in HH:MM format -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Represents a single digit change due to malfunction -/
inductive DigitChange
  | Increase
  | Decrease
  | NoChange

/-- Applies a digit change to a number -/
def applyDigitChange (n : Nat) (change : DigitChange) : Nat :=
  match change with
  | DigitChange.Increase => (n + 1) % 10
  | DigitChange.Decrease => (n + 9) % 10
  | DigitChange.NoChange => n

/-- Applies changes to all digits of a time -/
def applyChanges (t : Time) (h1 h2 m1 m2 : DigitChange) : Time :=
  let newHours := applyDigitChange (t.hours / 10) h1 * 10 + applyDigitChange (t.hours % 10) h2
  let newMinutes := applyDigitChange (t.minutes / 10) m1 * 10 + applyDigitChange (t.minutes % 10) m2
  ⟨newHours, newMinutes, sorry⟩

theorem clock_malfunction_proof :
  ∃ (original : Time) (h1 h2 m1 m2 : DigitChange),
    applyChanges original h1 h2 m1 m2 = ⟨20, 50, sorry⟩ ∧
    original = ⟨19, 49, sorry⟩ :=
  sorry

end NUMINAMATH_CALUDE_clock_malfunction_proof_l412_41220


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_l412_41282

/-- The cubic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 5 * x^2 + 3 * x - 2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 10 * x + 3

theorem decreasing_interval_of_f (a : ℝ) :
  (f' a 3 = 0) →
  (∀ x : ℝ, x ∈ Set.Icc (1/3 : ℝ) 3 ↔ f' a x ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_l412_41282


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l412_41296

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l412_41296


namespace NUMINAMATH_CALUDE_water_jars_count_l412_41227

/-- Represents the number of jars of each size -/
def num_jars : ℕ := 4

/-- Represents the total volume of water in gallons -/
def total_water : ℕ := 7

/-- Represents the volume of water in quarts -/
def water_in_quarts : ℕ := total_water * 4

theorem water_jars_count :
  num_jars * 3 = 12 ∧
  num_jars * (1 + 2 + 4) = water_in_quarts :=
by sorry

#check water_jars_count

end NUMINAMATH_CALUDE_water_jars_count_l412_41227


namespace NUMINAMATH_CALUDE_extraneous_root_implies_a_value_l412_41277

/-- The equation has an extraneous root if x = 3 is a solution to the polynomial form of the equation -/
def has_extraneous_root (a : ℚ) : Prop :=
  ∃ x : ℚ, x = 3 ∧ x - 2*a = 2*(x - 3)

/-- The original equation -/
def original_equation (x a : ℚ) : Prop :=
  x / (x - 3) - 2*a / (x - 3) = 2

theorem extraneous_root_implies_a_value :
  ∀ a : ℚ, has_extraneous_root a → a = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_extraneous_root_implies_a_value_l412_41277


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l412_41237

-- Define the vectors a and b
def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (6, -4)

-- Define the dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the scalar multiplication for 2D vectors
def scalar_mult (t : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (t * v.1, t * v.2)

-- Define vector addition for 2D vectors
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Theorem statement
theorem perpendicular_vectors (t : ℝ) : 
  dot_product a (vector_add (scalar_mult t a) b) = 0 → t = -5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l412_41237


namespace NUMINAMATH_CALUDE_concert_ticket_revenue_l412_41244

theorem concert_ticket_revenue 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (full_price : ℕ) 
  (discounted_price : ℕ) 
  (full_price_tickets : ℕ) 
  (discounted_tickets : ℕ) :
  total_tickets = 200 →
  total_revenue = 2800 →
  discounted_price = (3 * full_price) / 4 →
  total_tickets = full_price_tickets + discounted_tickets →
  total_revenue = full_price * full_price_tickets + discounted_price * discounted_tickets →
  full_price_tickets * full_price = 680 :=
by sorry

end NUMINAMATH_CALUDE_concert_ticket_revenue_l412_41244


namespace NUMINAMATH_CALUDE_awards_distribution_theorem_l412_41290

/-- The number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 6 awards to 3 students results in 465 ways -/
theorem awards_distribution_theorem :
  distribute_awards 6 3 = 465 :=
by sorry

end NUMINAMATH_CALUDE_awards_distribution_theorem_l412_41290


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_implies_a_range_l412_41219

/-- A point P(x, y) is in the second quadrant if x < 0 and y > 0 -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point P as a function of a -/
def x_coord (a : ℝ) : ℝ := 2 * a + 1

/-- The y-coordinate of point P as a function of a -/
def y_coord (a : ℝ) : ℝ := 1 - a

/-- Theorem: If P(2a+1, 1-a) is in the second quadrant, then a < -1/2 -/
theorem point_in_second_quadrant_implies_a_range (a : ℝ) :
  in_second_quadrant (x_coord a) (y_coord a) → a < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_implies_a_range_l412_41219


namespace NUMINAMATH_CALUDE_max_sum_in_t_grid_l412_41200

/-- A T-shaped grid represented as a list of 5 integers -/
def TGrid := List Int

/-- Check if a T-shaped grid is valid (contains exactly the numbers 2, 5, 8, 11, 14) -/
def isValidTGrid (grid : TGrid) : Prop :=
  grid.length = 5 ∧ grid.toFinset = {2, 5, 8, 11, 14}

/-- Calculate the vertical sum of a T-shaped grid -/
def verticalSum (grid : TGrid) : Int :=
  match grid with
  | [a, b, c, _, _] => a + b + c
  | _ => 0

/-- Calculate the horizontal sum of a T-shaped grid -/
def horizontalSum (grid : TGrid) : Int :=
  match grid with
  | [_, b, _, d, e] => b + d + e
  | _ => 0

/-- Check if a T-shaped grid satisfies the sum condition -/
def satisfiesSumCondition (grid : TGrid) : Prop :=
  verticalSum grid = horizontalSum grid

/-- The main theorem: The maximum sum in a valid T-shaped grid is 33 -/
theorem max_sum_in_t_grid :
  ∀ (grid : TGrid),
    isValidTGrid grid →
    satisfiesSumCondition grid →
    (verticalSum grid ≤ 33 ∧ horizontalSum grid ≤ 33) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_in_t_grid_l412_41200


namespace NUMINAMATH_CALUDE_inequality_proof_l412_41209

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l412_41209


namespace NUMINAMATH_CALUDE_inequality_range_l412_41273

open Real

theorem inequality_range (a : ℝ) : 
  (∀ x > 1, a * log x > 1 - 1/x) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l412_41273


namespace NUMINAMATH_CALUDE_odd_function_property_l412_41228

/-- Given a function f(x) = x^5 + ax^3 + bx, where a and b are real constants,
    if f(-2) = 10, then f(2) = -10 -/
theorem odd_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^5 + a*x^3 + b*x
  f (-2) = 10 → f 2 = -10 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l412_41228


namespace NUMINAMATH_CALUDE_library_books_problem_l412_41269

theorem library_books_problem (initial_books : ℕ) : 
  initial_books - 227 + 56 - 35 = 29 → initial_books = 235 :=
by sorry

end NUMINAMATH_CALUDE_library_books_problem_l412_41269


namespace NUMINAMATH_CALUDE_roots_sum_bound_l412_41241

theorem roots_sum_bound (u v : ℂ) : 
  u ≠ v → 
  u^2023 = 1 → 
  v^2023 = 1 → 
  Complex.abs (u + v) < Real.sqrt (2 + Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_bound_l412_41241


namespace NUMINAMATH_CALUDE_probability_theorem_l412_41202

/-- The probability that the straight-line distance between two randomly chosen points
    on the sides of a square with side length 2 is at least 1 -/
def probability_distance_at_least_one (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- A square with side length 2 -/
def square_side_two : Set (ℝ × ℝ) :=
  sorry

theorem probability_theorem :
  probability_distance_at_least_one square_side_two = (26 - Real.pi) / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l412_41202


namespace NUMINAMATH_CALUDE_base_conversion_sum_fraction_l412_41242

/-- Given that 546 in base 7 is equal to xy9 in base 10, where x and y are single digits,
    prove that (x + y + 9) / 21 = 6 / 7 -/
theorem base_conversion_sum_fraction :
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ 
  (5 * 7^2 + 4 * 7 + 6 : ℕ) = x * 100 + y * 10 + 9 →
  (x + y + 9 : ℚ) / 21 = 6 / 7 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_sum_fraction_l412_41242


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_sufficient_not_necessary_negation_equivalence_disjunction_not_both_true_l412_41204

-- 1. Contrapositive
theorem contrapositive_equivalence :
  (∀ x : ℝ, x ≠ 2 → x^2 - 5*x + 6 ≠ 0) ↔
  (∀ x : ℝ, x^2 - 5*x + 6 = 0 → x = 2) := by sorry

-- 2. Sufficient but not necessary condition
theorem sufficient_not_necessary :
  (∀ x : ℝ, x < 1 → x^2 - 3*x + 2 > 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ x ≥ 1) := by sorry

-- 3. Negation of universal quantifier
theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≠ 0) ↔
  (∃ x : ℝ, x^2 + x + 1 = 0) := by sorry

-- 4. Disjunction does not imply both true
theorem disjunction_not_both_true :
  ∃ (p q : Prop), (p ∨ q) ∧ ¬(p ∧ q) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_sufficient_not_necessary_negation_equivalence_disjunction_not_both_true_l412_41204


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l412_41295

theorem quadratic_equation_root (b : ℝ) :
  (∃ x : ℝ, 2 * x^2 + b * x - 119 = 0) ∧ (2 * 7^2 + b * 7 - 119 = 0) →
  b = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l412_41295


namespace NUMINAMATH_CALUDE_quadratic_root_coefficient_relation_l412_41255

/-- 
For a quadratic equation x^2 + px + q = 0 with roots α and β, 
this theorem states the relationship between the roots and the coefficients.
-/
theorem quadratic_root_coefficient_relation (p q α β : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = α ∨ x = β) → 
  (α + β = -p ∧ α * β = q) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_coefficient_relation_l412_41255


namespace NUMINAMATH_CALUDE_trig_identity_l412_41268

theorem trig_identity (α : Real) (h : Real.sin α + Real.cos α = Real.sqrt 2) :
  Real.tan α + Real.cos α / Real.sin α = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l412_41268


namespace NUMINAMATH_CALUDE_kyle_age_l412_41217

/-- Given the ages of several people and their relationships, prove Kyle's age --/
theorem kyle_age (david sandra casey fiona julian shelley kyle frederick tyson : ℕ) 
  (h1 : shelley = kyle - 3)
  (h2 : shelley = julian + 4)
  (h3 : julian = frederick - 20)
  (h4 : julian = fiona + 5)
  (h5 : frederick = 2 * tyson)
  (h6 : tyson = 2 * casey)
  (h7 : casey = fiona - 2)
  (h8 : 2 * casey = sandra)
  (h9 : sandra = david + 4)
  (h10 : david = 16) : 
  kyle = 23 := by sorry

end NUMINAMATH_CALUDE_kyle_age_l412_41217


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l412_41288

theorem opposite_of_negative_five : -((-5) : ℤ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l412_41288


namespace NUMINAMATH_CALUDE_star_difference_l412_41251

def star (x y : ℤ) : ℤ := x * y - 3 * x

theorem star_difference : (star 6 2) - (star 2 6) = -12 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_l412_41251


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l412_41299

/-- Given two concentric circles with radii A and B (A > B), if the area between
    the circles is 15π square meters, then the length of a chord of the larger
    circle that is tangent to the smaller circle is 2√15 meters. -/
theorem chord_length_concentric_circles (A B : ℝ) (h1 : A > B) (h2 : A > 0) (h3 : B > 0)
    (h4 : π * A^2 - π * B^2 = 15 * π) :
    ∃ (c : ℝ), c^2 = 4 * 15 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l412_41299


namespace NUMINAMATH_CALUDE_consecutive_color_groups_probability_l412_41222

-- Define the number of pencils of each color
def green_pencils : ℕ := 4
def orange_pencils : ℕ := 3
def blue_pencils : ℕ := 5

-- Define the total number of pencils
def total_pencils : ℕ := green_pencils + orange_pencils + blue_pencils

-- Define the probability of the specific selection
def probability_consecutive_color_groups : ℚ :=
  (Nat.factorial 3 * Nat.factorial green_pencils * Nat.factorial orange_pencils * Nat.factorial blue_pencils) /
  Nat.factorial total_pencils

-- Theorem statement
theorem consecutive_color_groups_probability :
  probability_consecutive_color_groups = 1 / 4620 :=
sorry

end NUMINAMATH_CALUDE_consecutive_color_groups_probability_l412_41222


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l412_41221

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of nitrogen atoms in the compound -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in the compound -/
def num_O : ℕ := 5

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := (num_N : ℝ) * atomic_weight_N + (num_O : ℝ) * atomic_weight_O

theorem molecular_weight_calculation :
  molecular_weight = 108.02 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l412_41221


namespace NUMINAMATH_CALUDE_second_graders_count_l412_41266

/-- The number of second graders wearing blue shirts -/
def second_graders : ℕ := sorry

/-- The cost of a blue shirt for second graders -/
def blue_shirt_cost : ℚ := 560 / 100

/-- The number of kindergartners -/
def kindergartners : ℕ := 101

/-- The cost of an orange shirt for kindergartners -/
def orange_shirt_cost : ℚ := 580 / 100

/-- The number of first graders -/
def first_graders : ℕ := 113

/-- The cost of a yellow shirt for first graders -/
def yellow_shirt_cost : ℚ := 500 / 100

/-- The number of third graders -/
def third_graders : ℕ := 108

/-- The cost of a green shirt for third graders -/
def green_shirt_cost : ℚ := 525 / 100

/-- The total amount spent on all shirts -/
def total_spent : ℚ := 231700 / 100

/-- Theorem stating that the number of second graders wearing blue shirts is 107 -/
theorem second_graders_count : second_graders = 107 := by
  sorry

end NUMINAMATH_CALUDE_second_graders_count_l412_41266


namespace NUMINAMATH_CALUDE_interior_angles_sum_l412_41212

theorem interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 3240) → (180 * ((n + 4) - 2) = 3960) := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l412_41212


namespace NUMINAMATH_CALUDE_inequality_proof_l412_41280

theorem inequality_proof (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (h_prod : a * b * c * d * e = 1) : 
  (d * e) / (a * (b + 1)) + (e * a) / (b * (c + 1)) + 
  (a * b) / (c * (d + 1)) + (b * c) / (d * (e + 1)) + 
  (c * d) / (e * (a + 1)) ≥ 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l412_41280


namespace NUMINAMATH_CALUDE_refrigerator_loss_percentage_l412_41257

/-- Represents the loss percentage on the refrigerator -/
def loss_percentage : ℝ := 4

/-- Represents the cost price of the refrigerator in Rupees -/
def refrigerator_cp : ℝ := 15000

/-- Represents the cost price of the mobile phone in Rupees -/
def mobile_cp : ℝ := 8000

/-- Represents the profit percentage on the mobile phone -/
def mobile_profit_percentage : ℝ := 9

/-- Represents the overall profit in Rupees -/
def overall_profit : ℝ := 120

/-- Theorem stating that given the conditions, the loss percentage on the refrigerator is 4% -/
theorem refrigerator_loss_percentage :
  let mobile_sp := mobile_cp * (1 + mobile_profit_percentage / 100)
  let total_cp := refrigerator_cp + mobile_cp
  let total_sp := total_cp + overall_profit
  let refrigerator_sp := total_sp - mobile_sp
  let loss := refrigerator_cp - refrigerator_sp
  loss_percentage = (loss / refrigerator_cp) * 100 := by
  sorry


end NUMINAMATH_CALUDE_refrigerator_loss_percentage_l412_41257


namespace NUMINAMATH_CALUDE_hidden_numbers_average_l412_41263

/-- Given three cards with visible numbers and hidden consecutive odd numbers,
    if the sum of numbers on each card is equal, then the average of hidden numbers is 18. -/
theorem hidden_numbers_average (v₁ v₂ v₃ h₁ h₂ h₃ : ℕ) : 
  v₁ = 30 ∧ v₂ = 42 ∧ v₃ = 36 →  -- visible numbers
  h₂ = h₁ + 2 ∧ h₃ = h₂ + 2 →    -- hidden numbers are consecutive odd
  v₁ + h₁ = v₂ + h₂ ∧ v₂ + h₂ = v₃ + h₃ →  -- sum on each card is equal
  (h₁ + h₂ + h₃) / 3 = 18 :=
by sorry

end NUMINAMATH_CALUDE_hidden_numbers_average_l412_41263


namespace NUMINAMATH_CALUDE_larger_cuboid_length_l412_41261

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- The dimensions of the smaller cuboid -/
def smallerCuboid : CuboidDimensions :=
  { length := 5, width := 4, height := 3 }

/-- The number of smaller cuboids that can be formed from the larger cuboid -/
def numberOfSmallerCuboids : ℕ := 32

/-- The width of the larger cuboid -/
def largerCuboidWidth : ℝ := 10

/-- The height of the larger cuboid -/
def largerCuboidHeight : ℝ := 12

theorem larger_cuboid_length :
  ∃ (largerLength : ℝ),
    cuboidVolume { length := largerLength, width := largerCuboidWidth, height := largerCuboidHeight } =
    (numberOfSmallerCuboids : ℝ) * cuboidVolume smallerCuboid ∧
    largerLength = 16 := by
  sorry

end NUMINAMATH_CALUDE_larger_cuboid_length_l412_41261


namespace NUMINAMATH_CALUDE_subtract_decimals_l412_41286

theorem subtract_decimals : 34.25 - 0.45 = 33.8 := by
  sorry

end NUMINAMATH_CALUDE_subtract_decimals_l412_41286


namespace NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l412_41201

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- 
If for an arithmetic sequence, 2S₃ = 3S₂ + 6, 
then the common difference is 2 
-/
theorem arithmetic_seq_common_diff 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l412_41201
