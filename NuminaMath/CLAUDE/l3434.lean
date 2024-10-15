import Mathlib

namespace NUMINAMATH_CALUDE_sphere_expansion_l3434_343474

theorem sphere_expansion (r₁ r₂ : ℝ) (h : r₁ > 0) (h' : r₂ > 0) :
  (4 * π * r₂^2) = 4 * (4 * π * r₁^2) →
  ((4 / 3) * π * r₂^3) = 8 * ((4 / 3) * π * r₁^3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_expansion_l3434_343474


namespace NUMINAMATH_CALUDE_omega_range_l3434_343470

open Real

theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∃! (r₁ r₂ r₃ : ℝ), 0 < r₁ ∧ r₁ < r₂ ∧ r₂ < r₃ ∧ r₃ < π ∧
    (∀ x, sin (ω * x) - Real.sqrt 3 * cos (ω * x) = -1 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃)) →
  13/6 < ω ∧ ω ≤ 7/2 :=
by sorry

end NUMINAMATH_CALUDE_omega_range_l3434_343470


namespace NUMINAMATH_CALUDE_symmetry_axis_l3434_343482

-- Define a function f with the given property
def f (x : ℝ) : ℝ := sorry

-- State the condition that f(x) = f(3 - x) for all x
axiom f_symmetry (x : ℝ) : f x = f (3 - x)

-- Define the concept of an axis of symmetry
def is_axis_of_symmetry (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem stating that x = 1.5 is an axis of symmetry for f
theorem symmetry_axis : is_axis_of_symmetry 1.5 f := by sorry

end NUMINAMATH_CALUDE_symmetry_axis_l3434_343482


namespace NUMINAMATH_CALUDE_pentagon_cannot_tile_floor_l3434_343445

-- Define a function to calculate the interior angle of a regular polygon
def interior_angle (n : ℕ) : ℚ :=
  180 - 360 / n

-- Define a function to check if an angle can divide 360° evenly
def divides_360 (angle : ℚ) : Prop :=
  ∃ k : ℕ, k * angle = 360

-- Theorem statement
theorem pentagon_cannot_tile_floor :
  divides_360 (interior_angle 6) ∧
  divides_360 90 ∧
  divides_360 60 ∧
  ¬ divides_360 (interior_angle 5) := by
  sorry

end NUMINAMATH_CALUDE_pentagon_cannot_tile_floor_l3434_343445


namespace NUMINAMATH_CALUDE_books_read_together_l3434_343415

theorem books_read_together (tony_books dean_books breanna_books tony_dean_overlap total_different : ℕ)
  (h1 : tony_books = 23)
  (h2 : dean_books = 12)
  (h3 : breanna_books = 17)
  (h4 : tony_dean_overlap = 3)
  (h5 : total_different = 47) :
  tony_books + dean_books + breanna_books - tony_dean_overlap - total_different = 2 :=
by sorry

end NUMINAMATH_CALUDE_books_read_together_l3434_343415


namespace NUMINAMATH_CALUDE_experienced_sailors_monthly_earnings_l3434_343458

/-- Calculate the total combined monthly earnings of experienced sailors --/
theorem experienced_sailors_monthly_earnings
  (total_sailors : ℕ)
  (inexperienced_sailors : ℕ)
  (inexperienced_hourly_wage : ℚ)
  (weekly_hours : ℕ)
  (weeks_per_month : ℕ)
  (h_total : total_sailors = 17)
  (h_inexperienced : inexperienced_sailors = 5)
  (h_wage : inexperienced_hourly_wage = 10)
  (h_hours : weekly_hours = 60)
  (h_weeks : weeks_per_month = 4) :
  let experienced_sailors := total_sailors - inexperienced_sailors
  let experienced_hourly_wage := inexperienced_hourly_wage * (1 + 1/5)
  let weekly_earnings := experienced_hourly_wage * weekly_hours
  let total_monthly_earnings := weekly_earnings * experienced_sailors * weeks_per_month
  total_monthly_earnings = 34560 :=
by sorry

end NUMINAMATH_CALUDE_experienced_sailors_monthly_earnings_l3434_343458


namespace NUMINAMATH_CALUDE_inequality_holds_l3434_343467

theorem inequality_holds (x y : ℝ) (h : 2 * y + 5 * x = 10) : 3 * x * y - x^2 - y^2 < 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3434_343467


namespace NUMINAMATH_CALUDE_aunts_gift_l3434_343430

theorem aunts_gift (grandfather_gift : ℕ) (bank_deposit : ℕ) (total_gift : ℕ) :
  grandfather_gift = 150 →
  bank_deposit = 45 →
  bank_deposit * 5 = total_gift →
  total_gift - grandfather_gift = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_aunts_gift_l3434_343430


namespace NUMINAMATH_CALUDE_pendant_sales_theorem_l3434_343483

/-- Parameters for the Asian Games mascot pendant sales problem -/
structure PendantSales where
  cost : ℝ             -- Cost price of each pendant
  initial_price : ℝ    -- Initial selling price
  initial_sales : ℝ    -- Initial monthly sales
  price_sensitivity : ℝ -- Daily sales decrease per 1 yuan price increase

/-- Calculate profit based on price increase -/
def profit (p : PendantSales) (x : ℝ) : ℝ :=
  (p.initial_price + x - p.cost) * (p.initial_sales - 30 * p.price_sensitivity * x)

/-- Theorem for the Asian Games mascot pendant sales problem -/
theorem pendant_sales_theorem (p : PendantSales) 
  (h1 : p.cost = 13)
  (h2 : p.initial_price = 20)
  (h3 : p.initial_sales = 200)
  (h4 : p.price_sensitivity = 10) :
  (∃ x : ℝ, x^2 - 13*x + 22 = 0 ∧ profit p x = 1620) ∧
  (∃ x : ℝ, x = 53/2 ∧ ∀ y : ℝ, profit p y ≤ profit p x) ∧
  profit p (13/2) = 3645/2 := by
  sorry


end NUMINAMATH_CALUDE_pendant_sales_theorem_l3434_343483


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_63n_l3434_343479

theorem smallest_n_for_sqrt_63n (n : ℕ) : n > 0 ∧ ∃ k : ℕ, k > 0 ∧ k^2 = 63 * n → n ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_63n_l3434_343479


namespace NUMINAMATH_CALUDE_alyssa_ate_25_limes_l3434_343489

/-- The number of limes Mike picked -/
def mike_limes : ℝ := 32.0

/-- The number of limes left -/
def limes_left : ℝ := 7

/-- The number of limes Alyssa ate -/
def alyssa_limes : ℝ := mike_limes - limes_left

/-- Proof that Alyssa ate 25.0 limes -/
theorem alyssa_ate_25_limes : alyssa_limes = 25.0 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_ate_25_limes_l3434_343489


namespace NUMINAMATH_CALUDE_balance_proof_l3434_343497

def initial_balance : ℕ := 27004
def transferred_amount : ℕ := 69
def remaining_balance : ℕ := 26935

theorem balance_proof : initial_balance = transferred_amount + remaining_balance := by
  sorry

end NUMINAMATH_CALUDE_balance_proof_l3434_343497


namespace NUMINAMATH_CALUDE_least_product_of_primes_over_50_l3434_343468

theorem least_product_of_primes_over_50 :
  ∃ p q : ℕ,
    p.Prime ∧ q.Prime ∧
    p > 50 ∧ q > 50 ∧
    p ≠ q ∧
    p * q = 3127 ∧
    ∀ r s : ℕ,
      r.Prime → s.Prime →
      r > 50 → s > 50 →
      r ≠ s →
      r * s ≥ 3127 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_primes_over_50_l3434_343468


namespace NUMINAMATH_CALUDE_base_10_423_equals_base_5_3143_l3434_343471

def base_10_to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base_10_423_equals_base_5_3143 :
  base_10_to_base_5 423 = [3, 1, 4, 3] := by
  sorry

end NUMINAMATH_CALUDE_base_10_423_equals_base_5_3143_l3434_343471


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3434_343429

-- Define the second quadrant
def second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 2 * Real.pi + Real.pi / 2 < α ∧ α < k * 2 * Real.pi + Real.pi

-- Define the first quadrant
def first_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 2 * Real.pi < α ∧ α < n * 2 * Real.pi + Real.pi / 2

-- Define the third quadrant
def third_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 2 * Real.pi + Real.pi < α ∧ α < n * 2 * Real.pi + 3 * Real.pi / 2

-- Theorem statement
theorem half_angle_quadrant (α : Real) :
  second_quadrant α → first_quadrant (α / 2) ∨ third_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3434_343429


namespace NUMINAMATH_CALUDE_cells_intersected_303x202_l3434_343414

/-- Represents a grid rectangle with diagonals --/
structure GridRectangle where
  length : Nat
  width : Nat

/-- Calculates the number of cells intersected by diagonals in a grid rectangle --/
def cells_intersected_by_diagonals (grid : GridRectangle) : Nat :=
  let small_rectangles := (grid.length / 3) * (grid.width / 2)
  let cells_per_diagonal := small_rectangles * 4
  let total_cells := cells_per_diagonal * 2
  total_cells - 2

/-- Theorem stating that in a 303 x 202 grid rectangle, 806 cells are intersected by diagonals --/
theorem cells_intersected_303x202 :
  cells_intersected_by_diagonals ⟨303, 202⟩ = 806 := by
  sorry

end NUMINAMATH_CALUDE_cells_intersected_303x202_l3434_343414


namespace NUMINAMATH_CALUDE_product_of_three_rationals_l3434_343409

theorem product_of_three_rationals (a b c : ℚ) :
  a * b * c < 0 → (a < 0 ∧ b ≥ 0 ∧ c ≥ 0) ∨
                   (a ≥ 0 ∧ b < 0 ∧ c ≥ 0) ∨
                   (a ≥ 0 ∧ b ≥ 0 ∧ c < 0) ∨
                   (a < 0 ∧ b < 0 ∧ c < 0) :=
by sorry

end NUMINAMATH_CALUDE_product_of_three_rationals_l3434_343409


namespace NUMINAMATH_CALUDE_sara_height_l3434_343444

/-- Given the heights of Mark, Roy, Joe, and Sara, prove Sara's height is 45 inches -/
theorem sara_height (mark_height joe_height roy_height sara_height : ℕ) 
  (h1 : mark_height = 34)
  (h2 : roy_height = mark_height + 2)
  (h3 : joe_height = roy_height + 3)
  (h4 : sara_height = joe_height + 6) :
  sara_height = 45 := by
  sorry

end NUMINAMATH_CALUDE_sara_height_l3434_343444


namespace NUMINAMATH_CALUDE_roots_sum_cubic_l3434_343422

theorem roots_sum_cubic (a b c : ℂ) : 
  a^3 + 2*a^2 + 3*a + 4 = 0 →
  b^3 + 2*b^2 + 3*b + 4 = 0 →
  c^3 + 2*c^2 + 3*c + 4 = 0 →
  (a^3 - b^3) / (a - b) + (b^3 - c^3) / (b - c) + (c^3 - a^3) / (c - a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_cubic_l3434_343422


namespace NUMINAMATH_CALUDE_largest_integer_solution_l3434_343440

theorem largest_integer_solution : 
  ∀ x : ℤ, (3 * x - 4 : ℚ) / 2 < x - 1 → x ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l3434_343440


namespace NUMINAMATH_CALUDE_max_goats_l3434_343484

/-- Represents the number of coconuts Max can trade for one crab -/
def coconuts_per_crab : ℕ := 3

/-- Represents the number of crabs Max can trade for one goat -/
def crabs_per_goat : ℕ := 6

/-- Represents the initial number of coconuts Max has -/
def initial_coconuts : ℕ := 342

/-- Calculates the number of goats Max will have after trading all his coconuts -/
def goats_from_coconuts (coconuts : ℕ) (coconuts_per_crab : ℕ) (crabs_per_goat : ℕ) : ℕ :=
  (coconuts / coconuts_per_crab) / crabs_per_goat

/-- Theorem stating that Max will end up with 19 goats -/
theorem max_goats : 
  goats_from_coconuts initial_coconuts coconuts_per_crab crabs_per_goat = 19 := by
  sorry

end NUMINAMATH_CALUDE_max_goats_l3434_343484


namespace NUMINAMATH_CALUDE_nested_radical_simplification_l3434_343404

theorem nested_radical_simplification (a b m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hpos : a + 2 * Real.sqrt b > 0)
  (hm : m > 0) (hn : n > 0) (hmn_sum : m + n = a) (hmn_prod : m * n = b) :
  Real.sqrt (a + 2 * Real.sqrt b) = Real.sqrt m + Real.sqrt n ∧
  Real.sqrt (a - 2 * Real.sqrt b) = |Real.sqrt m - Real.sqrt n| :=
by sorry

end NUMINAMATH_CALUDE_nested_radical_simplification_l3434_343404


namespace NUMINAMATH_CALUDE_remaining_trees_correct_l3434_343452

/-- The number of oak trees remaining in the park after cutting down damaged trees -/
def remaining_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that the number of remaining trees is correct -/
theorem remaining_trees_correct (initial : ℕ) (cut_down : ℕ) 
  (h : cut_down ≤ initial) : 
  remaining_trees initial cut_down = initial - cut_down :=
by sorry

end NUMINAMATH_CALUDE_remaining_trees_correct_l3434_343452


namespace NUMINAMATH_CALUDE_great_white_shark_teeth_l3434_343405

/-- The number of teeth of different shark species -/
def shark_teeth : ℕ → ℕ
| 0 => 180  -- tiger shark
| 1 => shark_teeth 0 / 6  -- hammerhead shark
| 2 => 2 * (shark_teeth 0 + shark_teeth 1)  -- great white shark
| _ => 0  -- other sharks (not relevant for this problem)

theorem great_white_shark_teeth : shark_teeth 2 = 420 := by
  sorry

end NUMINAMATH_CALUDE_great_white_shark_teeth_l3434_343405


namespace NUMINAMATH_CALUDE_multiplication_problem_l3434_343428

theorem multiplication_problem : 7 * (1 / 11) * 33 = 21 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l3434_343428


namespace NUMINAMATH_CALUDE_boxes_needed_proof_l3434_343461

/-- The number of chocolate bars Tom needs to sell -/
def total_bars : ℕ := 849

/-- The number of chocolate bars in each box -/
def bars_per_box : ℕ := 5

/-- The minimum number of boxes needed to contain all the bars -/
def min_boxes_needed : ℕ := (total_bars + bars_per_box - 1) / bars_per_box

theorem boxes_needed_proof : min_boxes_needed = 170 := by
  sorry

end NUMINAMATH_CALUDE_boxes_needed_proof_l3434_343461


namespace NUMINAMATH_CALUDE_smallest_x_value_l3434_343416

theorem smallest_x_value (x y : ℕ+) (h : (9 : ℚ) / 10 = y / (275 + x)) : 
  x ≥ 5 ∧ ∃ (y' : ℕ+), (9 : ℚ) / 10 = y' / (275 + 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3434_343416


namespace NUMINAMATH_CALUDE_matrix_operation_result_l3434_343426

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 6, 4]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-1, 8; -7, 3]

theorem matrix_operation_result :
  (2 : ℤ) • (A + B) = !![8, 10; -2, 14] := by sorry

end NUMINAMATH_CALUDE_matrix_operation_result_l3434_343426


namespace NUMINAMATH_CALUDE_solve_for_q_l3434_343400

theorem solve_for_q (k l q : ℚ) : 
  (7 / 8 = k / 96) → 
  (7 / 8 = (k + l) / 112) → 
  (7 / 8 = (q - l) / 144) → 
  q = 140 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l3434_343400


namespace NUMINAMATH_CALUDE_max_abs_z_purely_imaginary_l3434_343486

theorem max_abs_z_purely_imaginary (z : ℂ) :
  (∃ (t : ℝ), (z - Complex.I) / (z - 1) = Complex.I * t) → Complex.abs z ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_purely_imaginary_l3434_343486


namespace NUMINAMATH_CALUDE_max_perimeter_special_triangle_max_perimeter_achievable_l3434_343473

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Theorem: The maximum perimeter of a triangle ABC where a^2 = b^2 + c^2 - bc and a = 2 is 6 -/
theorem max_perimeter_special_triangle :
  ∀ t : Triangle,
    t.a^2 = t.b^2 + t.c^2 - t.b * t.c →
    t.a = 2 →
    perimeter t ≤ 6 :=
by
  sorry

/-- Corollary: There exists a triangle satisfying the conditions with perimeter equal to 6 -/
theorem max_perimeter_achievable :
  ∃ t : Triangle,
    t.a^2 = t.b^2 + t.c^2 - t.b * t.c ∧
    t.a = 2 ∧
    perimeter t = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_special_triangle_max_perimeter_achievable_l3434_343473


namespace NUMINAMATH_CALUDE_vasil_can_win_more_l3434_343448

/-- Represents the possible objects on a coin side -/
inductive Object
| Scissors
| Paper
| Rock

/-- Represents a coin with two sides -/
structure Coin where
  side1 : Object
  side2 : Object

/-- The set of available coins -/
def coins : List Coin := [
  ⟨Object.Scissors, Object.Paper⟩,
  ⟨Object.Rock, Object.Scissors⟩,
  ⟨Object.Paper, Object.Rock⟩
]

/-- Determines if object1 beats object2 -/
def beats (object1 object2 : Object) : Bool :=
  match object1, object2 with
  | Object.Scissors, Object.Paper => true
  | Object.Paper, Object.Rock => true
  | Object.Rock, Object.Scissors => true
  | _, _ => false

/-- Calculates the probability of Vasil winning against Asya -/
def winProbability (asyaCoin vasilCoin : Coin) : Rat :=
  sorry

/-- Theorem stating that Vasil can choose a coin to have a higher winning probability -/
theorem vasil_can_win_more : ∃ (strategy : Coin → Coin),
  ∀ (asyaChoice : Coin),
    asyaChoice ∈ coins →
    winProbability asyaChoice (strategy asyaChoice) > 1/2 :=
  sorry

end NUMINAMATH_CALUDE_vasil_can_win_more_l3434_343448


namespace NUMINAMATH_CALUDE_price_restoration_l3434_343498

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) 
  (h1 : reduced_price = 0.9 * original_price) : 
  (11 + 1/9) / 100 * reduced_price = original_price := by
sorry

end NUMINAMATH_CALUDE_price_restoration_l3434_343498


namespace NUMINAMATH_CALUDE_two_cars_meeting_on_highway_l3434_343403

/-- Theorem: Two cars meeting on a highway --/
theorem two_cars_meeting_on_highway 
  (highway_length : ℝ) 
  (time : ℝ) 
  (speed_car2 : ℝ) 
  (h1 : highway_length = 105) 
  (h2 : time = 3) 
  (h3 : speed_car2 = 20) : 
  ∃ (speed_car1 : ℝ), 
    speed_car1 * time + speed_car2 * time = highway_length ∧ 
    speed_car1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_two_cars_meeting_on_highway_l3434_343403


namespace NUMINAMATH_CALUDE_xuzhou_metro_scientific_notation_l3434_343435

theorem xuzhou_metro_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 31900 = a * (10 : ℝ) ^ n ∧ a = 3.19 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_xuzhou_metro_scientific_notation_l3434_343435


namespace NUMINAMATH_CALUDE_mom_tshirt_packages_l3434_343496

/-- The number of t-shirts in each package -/
def package_size : ℕ := 13

/-- The total number of t-shirts mom buys -/
def total_tshirts : ℕ := 39

/-- The number of packages mom will have -/
def num_packages : ℕ := total_tshirts / package_size

theorem mom_tshirt_packages : num_packages = 3 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_packages_l3434_343496


namespace NUMINAMATH_CALUDE_cube_volume_from_lateral_area_l3434_343459

/-- 
Given a cube with lateral surface area of 100 square units, 
prove that its volume is 125 cubic units.
-/
theorem cube_volume_from_lateral_area : 
  ∀ s : ℝ, 
  (4 * s^2 = 100) → 
  (s^3 = 125) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_lateral_area_l3434_343459


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_30_l3434_343493

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ,
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p > 30 ∧ 
    q > 30 ∧ 
    p ≠ q ∧
    p * q = 1147 ∧
    ∀ a b : ℕ, 
      Nat.Prime a → 
      Nat.Prime b → 
      a > 30 → 
      b > 30 → 
      a ≠ b → 
      a * b ≥ 1147 :=
by
  sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_30_l3434_343493


namespace NUMINAMATH_CALUDE_function_composition_nonnegative_implies_a_lower_bound_l3434_343424

theorem function_composition_nonnegative_implies_a_lower_bound 
  (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + 2 * x + 1) 
  (h2 : ∀ x, f (f x) ≥ 0) : 
  a ≥ (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_function_composition_nonnegative_implies_a_lower_bound_l3434_343424


namespace NUMINAMATH_CALUDE_circle_symmetry_l3434_343412

/-- The equation of the original circle -/
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y + 1 = 0

/-- The equation of the line of symmetry -/
def symmetry_line (x y : ℝ) : Prop :=
  x - y + 3 = 0

/-- The equation of the symmetric circle -/
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 2)^2 = 1

/-- Theorem stating that the symmetric_circle is indeed symmetric to the original_circle
    with respect to the symmetry_line -/
theorem circle_symmetry :
  ∀ (x y : ℝ), original_circle x y ↔ 
  ∃ (x' y' : ℝ), symmetric_circle x' y' ∧ 
  ((x + x')/2 - (y + y')/2 + 3 = 0) ∧
  ((y' - y)/(x' - x) = -1) :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3434_343412


namespace NUMINAMATH_CALUDE_monotonic_range_of_a_l3434_343411

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 2

theorem monotonic_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, Monotone (f a)) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_range_of_a_l3434_343411


namespace NUMINAMATH_CALUDE_all_statements_correct_l3434_343469

-- Define chemical elements and their atomic masses
def H : ℝ := 1
def O : ℝ := 16
def S : ℝ := 32
def N : ℝ := 14
def C : ℝ := 12

-- Define molecules and their molar masses
def H2SO4_mass : ℝ := 2 * H + S + 4 * O
def NO_mass : ℝ := N + O
def NO2_mass : ℝ := N + 2 * O
def O2_mass : ℝ := 2 * O
def O3_mass : ℝ := 3 * O
def CO_mass : ℝ := C + O
def CO2_mass : ℝ := C + 2 * O

-- Define the number of atoms in 2 mol of NO and NO2
def NO_atoms : ℕ := 2
def NO2_atoms : ℕ := 3

-- Theorem stating all given statements are correct
theorem all_statements_correct :
  (H2SO4_mass = 98) ∧
  (2 * NO_atoms ≠ 2 * NO2_atoms) ∧
  (∀ m : ℝ, m > 0 → m / O2_mass * 2 = m / O3_mass * 3) ∧
  (∀ n : ℝ, n > 0 → n * (CO_mass / C) = n * (CO2_mass / C)) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_correct_l3434_343469


namespace NUMINAMATH_CALUDE_sum_of_divisors_154_l3434_343450

/-- The sum of all positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all positive divisors of 154 is 288 -/
theorem sum_of_divisors_154 : sum_of_divisors 154 = 288 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_154_l3434_343450


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l3434_343438

/-- Represents the number of water heaters in a sample from a specific factory -/
structure FactorySample where
  total : ℕ
  factory_a : ℕ
  factory_b : ℕ
  sample_size : ℕ

/-- Calculates the stratified sample size for a factory -/
def stratified_sample_size (total : ℕ) (factory : ℕ) (sample_size : ℕ) : ℕ :=
  (factory * sample_size) / total

/-- Theorem stating the correct stratified sample sizes for factories A and B -/
theorem correct_stratified_sample (fs : FactorySample) 
  (h1 : fs.total = 98)
  (h2 : fs.factory_a = 56)
  (h3 : fs.factory_b = 42)
  (h4 : fs.sample_size = 14) :
  stratified_sample_size fs.total fs.factory_a fs.sample_size = 8 ∧
  stratified_sample_size fs.total fs.factory_b fs.sample_size = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l3434_343438


namespace NUMINAMATH_CALUDE_max_xy_under_constraint_l3434_343406

theorem max_xy_under_constraint (x y : ℕ+) (h : 27 * x.val + 35 * y.val ≤ 945) :
  x.val * y.val ≤ 234 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_under_constraint_l3434_343406


namespace NUMINAMATH_CALUDE_multiples_of_4_or_6_in_100_l3434_343464

theorem multiples_of_4_or_6_in_100 :
  let S := Finset.range 100
  (S.filter (fun n => n % 4 = 0 ∨ n % 6 = 0)).card = 33 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_4_or_6_in_100_l3434_343464


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l3434_343462

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l3434_343462


namespace NUMINAMATH_CALUDE_luke_candy_purchase_luke_candy_purchase_result_l3434_343453

/-- The number of candy pieces Luke can buy given his tickets and candy cost -/
theorem luke_candy_purchase (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) : ℕ :=
  by
  have h1 : whack_a_mole_tickets = 2 := by sorry
  have h2 : skee_ball_tickets = 13 := by sorry
  have h3 : candy_cost = 3 := by sorry
  
  have total_tickets : ℕ := whack_a_mole_tickets + skee_ball_tickets
  
  exact total_tickets / candy_cost

/-- Proof that Luke can buy 5 pieces of candy -/
theorem luke_candy_purchase_result : luke_candy_purchase 2 13 3 = 5 := by sorry

end NUMINAMATH_CALUDE_luke_candy_purchase_luke_candy_purchase_result_l3434_343453


namespace NUMINAMATH_CALUDE_eighth_diagram_fully_shaded_l3434_343475

/-- The number of shaded squares in the n-th diagram -/
def shaded_squares (n : ℕ) : ℕ := n^2

/-- The total number of squares in the n-th diagram -/
def total_squares (n : ℕ) : ℕ := n^2

/-- The fraction of shaded squares in the n-th diagram -/
def shaded_fraction (n : ℕ) : ℚ :=
  (shaded_squares n : ℚ) / (total_squares n : ℚ)

theorem eighth_diagram_fully_shaded :
  shaded_fraction 8 = 1 := by sorry

end NUMINAMATH_CALUDE_eighth_diagram_fully_shaded_l3434_343475


namespace NUMINAMATH_CALUDE_eighth_group_student_number_l3434_343423

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  students_per_group : ℕ
  selected_number : ℕ
  selected_group : ℕ

/-- Calculates the number of the student in a given group -/
def student_number (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.selected_number + (group - s.selected_group) * s.students_per_group

/-- Theorem: In the given systematic sampling, the student from the 8th group has number 37 -/
theorem eighth_group_student_number (s : SystematicSampling) 
    (h1 : s.total_students = 50)
    (h2 : s.num_groups = 10)
    (h3 : s.students_per_group = 5)
    (h4 : s.selected_number = 12)
    (h5 : s.selected_group = 3) :
    student_number s 8 = 37 := by
  sorry


end NUMINAMATH_CALUDE_eighth_group_student_number_l3434_343423


namespace NUMINAMATH_CALUDE_negative_abs_opposite_double_negative_l3434_343465

-- Define the property of being opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- State the theorem
theorem negative_abs_opposite_double_negative :
  are_opposite (-|(-3 : ℝ)|) (-(-3)) :=
sorry

end NUMINAMATH_CALUDE_negative_abs_opposite_double_negative_l3434_343465


namespace NUMINAMATH_CALUDE_expression_simplification_l3434_343443

theorem expression_simplification :
  Real.sqrt 12 - 2 * Real.cos (30 * π / 180) - (1/3)⁻¹ = Real.sqrt 3 - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3434_343443


namespace NUMINAMATH_CALUDE_zoo_arrangements_l3434_343408

/-- The number of letters in the word "ZOO₁M₁O₂M₂O₃" -/
def word_length : ℕ := 7

/-- The number of distinct arrangements of the letters in "ZOO₁M₁O₂M₂O₃" -/
def num_arrangements : ℕ := Nat.factorial word_length

theorem zoo_arrangements :
  num_arrangements = 5040 := by sorry

end NUMINAMATH_CALUDE_zoo_arrangements_l3434_343408


namespace NUMINAMATH_CALUDE_min_value_problem_l3434_343456

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2/y = 3) :
  ∃ (m : ℝ), m = 8/3 ∧ ∀ z, z = 2/x + y → z ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3434_343456


namespace NUMINAMATH_CALUDE_gcd_repeating_even_three_digit_l3434_343421

theorem gcd_repeating_even_three_digit : 
  ∃ g : ℕ, ∀ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ Even n → 
    g = Nat.gcd (1001 * n) (Nat.gcd (1001 * (n + 2)) (1001 * (n + 4))) ∧ 
    g = 2002 := by
  sorry

end NUMINAMATH_CALUDE_gcd_repeating_even_three_digit_l3434_343421


namespace NUMINAMATH_CALUDE_find_m_l3434_343449

def A (m : ℕ) : Set ℝ := {x : ℝ | (m * x - 1) / x < 0}

def B : Set ℝ := {x : ℝ | 2 * x^2 - x < 0}

def is_necessary_not_sufficient (A B : Set ℝ) : Prop :=
  B ⊆ A ∧ A ≠ B

theorem find_m :
  ∃ (m : ℕ), m > 0 ∧ m < 6 ∧ is_necessary_not_sufficient (A m) B ∧ m = 1 :=
by sorry

end NUMINAMATH_CALUDE_find_m_l3434_343449


namespace NUMINAMATH_CALUDE_candy_distribution_l3434_343460

theorem candy_distribution (N : ℕ) : N > 1 ∧ 
  N % 2 = 1 ∧ 
  N % 3 = 1 ∧ 
  N % 5 = 1 ∧ 
  (∀ m : ℕ, m > 1 → m % 2 = 1 → m % 3 = 1 → m % 5 = 1 → m ≥ N) → 
  N = 31 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3434_343460


namespace NUMINAMATH_CALUDE_inequality_solution_l3434_343476

def inequality (x : ℝ) : Prop :=
  (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 2

def solution_set (x : ℝ) : Prop :=
  (0 < x ∧ x ≤ 0.5) ∨ (x ≥ 6)

theorem inequality_solution : ∀ x : ℝ, inequality x ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3434_343476


namespace NUMINAMATH_CALUDE_double_points_on_quadratic_l3434_343410

/-- A double point is a point where the ordinate is twice its abscissa. -/
def is_double_point (x y : ℝ) : Prop := y = 2 * x

/-- The quadratic function y = x² + 2mx - m -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x - m

theorem double_points_on_quadratic (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_double_point x₁ y₁ ∧
    is_double_point x₂ y₂ ∧
    y₁ = quadratic_function m x₁ ∧
    y₂ = quadratic_function m x₂ ∧
    x₁ < 1 ∧ 1 < x₂ →
    m < 1 :=
sorry

end NUMINAMATH_CALUDE_double_points_on_quadratic_l3434_343410


namespace NUMINAMATH_CALUDE_simplify_expression_l3434_343487

theorem simplify_expression : (18 * (10^10)) / (6 * (10^4)) * 2 = 6 * (10^6) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3434_343487


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3434_343434

/-- Given a rhombus with diagonals of 10 inches and 24 inches, its perimeter is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3434_343434


namespace NUMINAMATH_CALUDE_fraction_exists_l3434_343490

theorem fraction_exists : ∃ n : ℕ, (n : ℚ) / 22 = 9545 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_exists_l3434_343490


namespace NUMINAMATH_CALUDE_probability_of_vowels_in_all_sets_l3434_343455

-- Define the sets
def set1 : Finset Char := {'a', 'b', 'o', 'd', 'e', 'f', 'g'}
def set2 : Finset Char := {'k', 'l', 'm', 'n', 'u', 'p', 'r', 's'}
def set3 : Finset Char := {'t', 'v', 'w', 'i', 'x', 'y', 'z'}
def set4 : Finset Char := {'a', 'c', 'e', 'u', 'g', 'h', 'j'}

-- Define vowels
def vowels : Finset Char := {'a', 'e', 'i', 'o', 'u'}

-- Function to count vowels in a set
def countVowels (s : Finset Char) : Nat :=
  (s ∩ vowels).card

-- Theorem statement
theorem probability_of_vowels_in_all_sets :
  let p1 := (countVowels set1 : ℚ) / set1.card
  let p2 := (countVowels set2 : ℚ) / set2.card
  let p3 := (countVowels set3 : ℚ) / set3.card
  let p4 := (countVowels set4 : ℚ) / set4.card
  p1 * p2 * p3 * p4 = 9 / 2744 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_vowels_in_all_sets_l3434_343455


namespace NUMINAMATH_CALUDE_product_difference_bound_l3434_343419

theorem product_difference_bound (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h1 : x * y - z = x * z - y) (h2 : x * z - y = y * z - x) : 
  x * y - z ≥ -1/4 := by
sorry

end NUMINAMATH_CALUDE_product_difference_bound_l3434_343419


namespace NUMINAMATH_CALUDE_not_prime_n4_plus_n2_plus_1_l3434_343441

theorem not_prime_n4_plus_n2_plus_1 (n : ℕ) (h : n > 1) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + n^2 + 1 = a * b :=
by
  sorry

end NUMINAMATH_CALUDE_not_prime_n4_plus_n2_plus_1_l3434_343441


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_three_fifths_l3434_343439

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol_fraction : ℚ
  water_fraction : ℚ
  sum_is_one : alcohol_fraction + water_fraction = 1

/-- The ratio of alcohol to water in a mixture -/
def alcohol_to_water_ratio (m : Mixture) : ℚ := m.alcohol_fraction / m.water_fraction

/-- Theorem stating that for a mixture with 3/5 alcohol and 2/5 water, 
    the ratio of alcohol to water is 3:2 -/
theorem alcohol_water_ratio_three_fifths 
  (m : Mixture) 
  (h1 : m.alcohol_fraction = 3/5) 
  (h2 : m.water_fraction = 2/5) : 
  alcohol_to_water_ratio m = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_alcohol_water_ratio_three_fifths_l3434_343439


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l3434_343477

theorem quadratic_root_sum (a b c : ℤ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = a ∨ x = b) →
  a + b + c = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l3434_343477


namespace NUMINAMATH_CALUDE_bead_necklaces_sold_l3434_343466

/-- Proves that the number of bead necklaces sold is 4, given the conditions of the problem -/
theorem bead_necklaces_sold (gem_necklaces : ℕ) (price_per_necklace : ℕ) (total_earnings : ℕ) 
  (h1 : gem_necklaces = 3)
  (h2 : price_per_necklace = 3)
  (h3 : total_earnings = 21) :
  total_earnings - gem_necklaces * price_per_necklace = 4 * price_per_necklace :=
by sorry

end NUMINAMATH_CALUDE_bead_necklaces_sold_l3434_343466


namespace NUMINAMATH_CALUDE_drug_effectiveness_max_effective_hours_l3434_343437

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x ≤ 4 then -1/2 * x^2 + 2*x + 8
  else if 4 < x ∧ x ≤ 16 then -x/2 - Real.log x / Real.log 2 + 12
  else 0

def is_effective (m : ℝ) (x : ℝ) : Prop := m * f x ≥ 12

theorem drug_effectiveness (m : ℝ) (h : m > 0) :
  (∀ x, 0 < x ∧ x ≤ 8 → is_effective m x) ↔ m ≥ 12/5 :=
sorry

theorem max_effective_hours :
  ∃ k : ℕ, k = 6 ∧ 
  (∀ x, 0 < x ∧ x ≤ ↑k → is_effective 2 x) ∧
  (∀ k' : ℕ, k' > k → ∃ x, 0 < x ∧ x ≤ ↑k' ∧ ¬is_effective 2 x) :=
sorry

end NUMINAMATH_CALUDE_drug_effectiveness_max_effective_hours_l3434_343437


namespace NUMINAMATH_CALUDE_expression_value_l3434_343432

theorem expression_value : 3^(2+4+6) - (3^2 + 3^4 + 3^6) + (3^2 * 3^4 * 3^6) = 1062242 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3434_343432


namespace NUMINAMATH_CALUDE_find_divisor_l3434_343492

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) (divisor : Nat) :
  dividend = quotient * divisor + remainder →
  dividend = 109 →
  quotient = 9 →
  remainder = 1 →
  divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3434_343492


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1581_l3434_343499

def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_1581 : 
  largest_prime_factor 1581 = 113 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1581_l3434_343499


namespace NUMINAMATH_CALUDE_corrected_mean_l3434_343472

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 →
  original_mean = 40 →
  incorrect_value = 15 →
  correct_value = 45 →
  let original_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := original_sum + difference
  corrected_sum / n = 40.6 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l3434_343472


namespace NUMINAMATH_CALUDE_xy_value_l3434_343401

theorem xy_value (x y : ℝ) (h1 : |x| = 2) (h2 : y = 3) (h3 : x * y < 0) : x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3434_343401


namespace NUMINAMATH_CALUDE_bob_remaining_corn_l3434_343427

theorem bob_remaining_corn (initial_bushels : ℕ) (ears_per_bushel : ℕ) 
  (terry_bushels jerry_bushels linda_bushels : ℕ) (stacy_ears : ℕ) : 
  initial_bushels = 50 →
  ears_per_bushel = 14 →
  terry_bushels = 8 →
  jerry_bushels = 3 →
  linda_bushels = 12 →
  stacy_ears = 21 →
  initial_bushels * ears_per_bushel - 
  (terry_bushels * ears_per_bushel + jerry_bushels * ears_per_bushel + 
   linda_bushels * ears_per_bushel + stacy_ears) = 357 :=
by sorry

end NUMINAMATH_CALUDE_bob_remaining_corn_l3434_343427


namespace NUMINAMATH_CALUDE_trivia_game_points_per_question_l3434_343480

theorem trivia_game_points_per_question 
  (first_half_correct : ℕ) 
  (second_half_correct : ℕ) 
  (final_score : ℕ) 
  (h1 : first_half_correct = 6)
  (h2 : second_half_correct = 4)
  (h3 : final_score = 30) :
  final_score / (first_half_correct + second_half_correct) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_points_per_question_l3434_343480


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3434_343446

theorem contrapositive_equivalence (x y : ℝ) :
  (((x - 1) * (y + 2) ≠ 0 → x ≠ 1 ∧ y ≠ -2) ↔
   (x = 1 ∨ y = -2 → (x - 1) * (y + 2) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3434_343446


namespace NUMINAMATH_CALUDE_max_a_value_l3434_343417

/-- A lattice point in an xy-coordinate system -/
def LatticePoint (x y : ℤ) : Prop := True

/-- The line equation y = mx + 5 -/
def LineEquation (m : ℚ) (x y : ℤ) : Prop := y = m * x + 5

/-- The condition for x -/
def XCondition (x : ℤ) : Prop := 0 < x ∧ x ≤ 150

/-- The condition for m -/
def MCondition (m a : ℚ) : Prop := 1/3 < m ∧ m < a

/-- The main theorem -/
theorem max_a_value : 
  ∃ (a : ℚ), a = 52/151 ∧ 
  (∀ (m : ℚ), MCondition m a → 
    ∀ (x y : ℤ), XCondition x → LatticePoint x y → ¬LineEquation m x y) ∧
  (∀ (a' : ℚ), a' > a → 
    ∃ (m : ℚ), MCondition m a' ∧
    ∃ (x y : ℤ), XCondition x ∧ LatticePoint x y ∧ LineEquation m x y) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l3434_343417


namespace NUMINAMATH_CALUDE_intersection_distance_implies_a_value_l3434_343457

-- Define the curve C
def curve_C (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x ∧ a > 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the intersection points
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ curve_C a x y ∧ line_l x y}

-- Theorem statement
theorem intersection_distance_implies_a_value (a : ℝ) :
  (∃ (A B : ℝ × ℝ), A ∈ intersection_points a ∧ B ∈ intersection_points a ∧ 
   A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 10) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_implies_a_value_l3434_343457


namespace NUMINAMATH_CALUDE_grape_yield_after_change_l3434_343413

/-- Represents the number of jars that can be made from one can of juice -/
structure JuiceYield where
  apple : ℚ
  grape : ℚ

/-- Represents the recipe for the beverage -/
structure Recipe where
  apple : ℚ
  grape : ℚ

/-- The initial recipe yield -/
def initial_yield : JuiceYield :=
  { apple := 6,
    grape := 10 }

/-- The changed recipe yield for apple juice -/
def changed_apple_yield : ℚ := 5

/-- Theorem stating that after the recipe change, one can of grape juice makes 15 jars -/
theorem grape_yield_after_change
  (initial : JuiceYield)
  (changed_apple : ℚ)
  (h_initial : initial = initial_yield)
  (h_changed_apple : changed_apple = changed_apple_yield)
  : ∃ (changed : JuiceYield), changed.grape = 15 :=
sorry

end NUMINAMATH_CALUDE_grape_yield_after_change_l3434_343413


namespace NUMINAMATH_CALUDE_g_satisfies_conditions_g_value_at_neg_two_l3434_343463

/-- A cubic polynomial -/
def CubicPolynomial (α : Type*) [Field α] := α → α

/-- The polynomial f(x) = x^3 - 2x^2 + 5 -/
def f : CubicPolynomial ℝ := λ x ↦ x^3 - 2*x^2 + 5

/-- The polynomial g, which is defined by the problem conditions -/
noncomputable def g : CubicPolynomial ℝ := sorry

/-- The roots of f -/
noncomputable def roots_f : Finset ℝ := sorry

theorem g_satisfies_conditions :
  (g 0 = 1) ∧ 
  (∀ r ∈ roots_f, ∃ s, g s = 0 ∧ s = r^2) ∧
  (∀ s, g s = 0 → ∃ r ∈ roots_f, s = r^2) := sorry

theorem g_value_at_neg_two : g (-2) = 24.2 := sorry

end NUMINAMATH_CALUDE_g_satisfies_conditions_g_value_at_neg_two_l3434_343463


namespace NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l3434_343418

-- Define the function f
def f (x m n : ℝ) : ℝ := 2 * x^3 + 3 * m * x^2 + 3 * n * x - 6

-- Define the derivative of f
def f' (x m n : ℝ) : ℝ := 6 * x^2 + 6 * m * x + 3 * n

theorem extreme_values_and_monotonicity :
  ∃ (m n : ℝ),
    (f' 1 m n = 0 ∧ f' 2 m n = 0) ∧
    (m = -3 ∧ n = 4) ∧
    (∀ x, x < 1 → (f' x m n > 0)) ∧
    (∀ x, 1 < x ∧ x < 2 → (f' x m n < 0)) ∧
    (∀ x, x > 2 → (f' x m n > 0)) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l3434_343418


namespace NUMINAMATH_CALUDE_rice_bags_weight_analysis_l3434_343488

def standard_weight : ℝ := 50
def num_bags : ℕ := 10
def weight_deviations : List ℝ := [0.5, 0.3, 0, -0.2, -0.3, 1.1, -0.7, -0.2, 0.6, 0.7]

theorem rice_bags_weight_analysis :
  let total_deviation : ℝ := weight_deviations.sum
  let total_weight : ℝ := (standard_weight * num_bags) + total_deviation
  let average_weight : ℝ := total_weight / num_bags
  (total_deviation = 1.7) ∧ 
  (total_weight = 501.7) ∧ 
  (average_weight = 50.17) := by
sorry

end NUMINAMATH_CALUDE_rice_bags_weight_analysis_l3434_343488


namespace NUMINAMATH_CALUDE_unique_x_intercept_l3434_343481

theorem unique_x_intercept (x : ℝ) : 
  ∃! x, (x - 4) * (x^2 + 4*x + 13) = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_intercept_l3434_343481


namespace NUMINAMATH_CALUDE_marcy_makeup_count_l3434_343478

/-- The number of people Marcy can paint with one tube of lip gloss -/
def people_per_tube : ℕ := 3

/-- The number of tubs of lip gloss Marcy brings -/
def tubs : ℕ := 6

/-- The number of tubes of lip gloss in each tub -/
def tubes_per_tub : ℕ := 2

/-- The total number of people Marcy is painting with makeup -/
def total_people : ℕ := tubs * tubes_per_tub * people_per_tube

theorem marcy_makeup_count : total_people = 36 := by
  sorry

end NUMINAMATH_CALUDE_marcy_makeup_count_l3434_343478


namespace NUMINAMATH_CALUDE_sachins_age_l3434_343454

theorem sachins_age (sachin_age rahul_age : ℝ) 
  (h1 : rahul_age = sachin_age + 9)
  (h2 : sachin_age / rahul_age = 7 / 9) :
  sachin_age = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_sachins_age_l3434_343454


namespace NUMINAMATH_CALUDE_eighth_week_hours_l3434_343491

def hours_worked : List ℕ := [9, 13, 8, 14, 12, 10, 11]
def total_weeks : ℕ := 8
def target_average : ℕ := 12

theorem eighth_week_hours : 
  ∃ x : ℕ, 
    (List.sum hours_worked + x) / total_weeks = target_average ∧ 
    x = 19 := by
  sorry

end NUMINAMATH_CALUDE_eighth_week_hours_l3434_343491


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l3434_343407

theorem dining_bill_calculation (total_bill : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
  (h1 : total_bill = 198)
  (h2 : tax_rate = 0.1)
  (h3 : tip_rate = 0.2) : 
  ∃ (food_price : ℝ), 
    food_price * (1 + tax_rate) * (1 + tip_rate) = total_bill ∧ 
    food_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l3434_343407


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3434_343402

-- Define the operation
def determinant (a b c d : ℂ) : ℂ := a * d - b * c

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), determinant 1 (-1) z (z * Complex.I) = 4 + 2 * Complex.I ∧ z = 3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3434_343402


namespace NUMINAMATH_CALUDE_head_start_calculation_l3434_343451

/-- Prove that given A runs 1 ¾ times as fast as B, and A and B reach a winning post 196 m away at the same time, the head start A gives B is 84 meters. -/
theorem head_start_calculation (speed_a speed_b head_start : ℝ) 
  (h1 : speed_a = (7/4) * speed_b)
  (h2 : (196 - head_start) / speed_b = 196 / speed_a) :
  head_start = 84 := by
  sorry

end NUMINAMATH_CALUDE_head_start_calculation_l3434_343451


namespace NUMINAMATH_CALUDE_population_growth_two_periods_l3434_343495

/-- Theorem: Population growth over two periods --/
theorem population_growth_two_periods (P : ℝ) (h : P > 0) :
  let first_half := P * 3
  let second_half := first_half * 4
  (second_half - P) / P * 100 = 1100 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_two_periods_l3434_343495


namespace NUMINAMATH_CALUDE_cubic_sum_l3434_343442

theorem cubic_sum (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 8) / a = (b^3 + 8) / b ∧ (b^3 + 8) / b = (c^3 + 8) / c) : 
  a^3 + b^3 + c^3 = -24 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_l3434_343442


namespace NUMINAMATH_CALUDE_f_solution_sets_l3434_343494

/-- The function f(x) = ax^2 - (a+c)x + c -/
def f (a c x : ℝ) : ℝ := a * x^2 - (a + c) * x + c

theorem f_solution_sets :
  /- Part 1 -/
  (∀ a c : ℝ, a > 0 → (∀ x : ℝ, f a c x = f a c (-2 - x)) →
    {x : ℝ | f a c x > 0} = {x : ℝ | x < -3 ∨ x > 1}) ∧
  /- Part 2 -/
  (∀ a : ℝ, a ≥ 0 → f a 1 0 = 1 →
    {x : ℝ | f a 1 x > 0} =
      if a = 0 then {x : ℝ | x > 1}
      else if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1/a}
      else if a > 1 then {x : ℝ | 1/a < x ∧ x < 1}
      else ∅) :=
by sorry

end NUMINAMATH_CALUDE_f_solution_sets_l3434_343494


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3434_343425

def A : Set ℕ := {2, 4}
def B : Set ℕ := {3, 4}

theorem intersection_of_A_and_B : A ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3434_343425


namespace NUMINAMATH_CALUDE_sozopolian_inequality_sozopolian_equality_l3434_343436

/-- Definition of a Sozopolian set -/
def is_sozopolian (p a b c : ℕ) : Prop :=
  Nat.Prime p ∧ p % 2 = 1 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a * b + 1) % p = 0 ∧
  (b * c + 1) % p = 0 ∧
  (c * a + 1) % p = 0

theorem sozopolian_inequality (p a b c : ℕ) :
  is_sozopolian p a b c → p + 2 ≤ (a + b + c) / 3 :=
sorry

theorem sozopolian_equality (p : ℕ) :
  (∃ a b c : ℕ, is_sozopolian p a b c ∧ p + 2 = (a + b + c) / 3) ↔ p = 5 :=
sorry

end NUMINAMATH_CALUDE_sozopolian_inequality_sozopolian_equality_l3434_343436


namespace NUMINAMATH_CALUDE_stem_and_leaf_preserves_info_l3434_343485

/-- Represents different types of statistical charts -/
inductive StatChart
  | BarChart
  | PieChart
  | LineChart
  | StemAndLeafPlot

/-- Predicate to determine if a chart loses information -/
def loses_information (chart : StatChart) : Prop :=
  match chart with
  | StatChart.BarChart => True
  | StatChart.PieChart => True
  | StatChart.LineChart => True
  | StatChart.StemAndLeafPlot => False

/-- Theorem stating that only the stem-and-leaf plot does not lose information -/
theorem stem_and_leaf_preserves_info :
  ∀ (chart : StatChart), ¬(loses_information chart) ↔ chart = StatChart.StemAndLeafPlot :=
by sorry


end NUMINAMATH_CALUDE_stem_and_leaf_preserves_info_l3434_343485


namespace NUMINAMATH_CALUDE_comic_arrangement_count_l3434_343447

/-- The number of different Batman comic books --/
def batman_comics : ℕ := 8

/-- The number of different Superman comic books --/
def superman_comics : ℕ := 7

/-- The number of different Wonder Woman comic books --/
def wonder_woman_comics : ℕ := 5

/-- The total number of comic books --/
def total_comics : ℕ := batman_comics + superman_comics + wonder_woman_comics

/-- The number of different comic book types --/
def comic_types : ℕ := 3

theorem comic_arrangement_count :
  (Nat.factorial batman_comics) * (Nat.factorial superman_comics) * 
  (Nat.factorial wonder_woman_comics) * (Nat.factorial comic_types) = 12203212800 := by
  sorry

end NUMINAMATH_CALUDE_comic_arrangement_count_l3434_343447


namespace NUMINAMATH_CALUDE_consecutive_natural_even_product_inequality_l3434_343431

theorem consecutive_natural_even_product_inequality (n : ℕ) (m : ℕ) (h : Even m) (h_pos : m > 0) :
  n * (n + 1) ≠ m * (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_natural_even_product_inequality_l3434_343431


namespace NUMINAMATH_CALUDE_five_n_plus_three_composite_l3434_343420

theorem five_n_plus_three_composite (n : ℕ) 
  (h1 : ∃ k : ℕ, 2 * n + 1 = k^2) 
  (h2 : ∃ m : ℕ, 3 * n + 1 = m^2) : 
  ¬(Nat.Prime (5 * n + 3)) :=
sorry

end NUMINAMATH_CALUDE_five_n_plus_three_composite_l3434_343420


namespace NUMINAMATH_CALUDE_square_sum_value_l3434_343433

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 6) : a^2 + b^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l3434_343433
