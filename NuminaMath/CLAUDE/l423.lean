import Mathlib

namespace NUMINAMATH_CALUDE_range_of_g_l423_42334

def f (x : ℝ) : ℝ := 4 * x + 1

def g (x : ℝ) : ℝ := f (f (f x))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 →
  ∃ y : ℝ, g y = x ∧ -43 ≤ x ∧ x ≤ 213 :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l423_42334


namespace NUMINAMATH_CALUDE_no_isosceles_triangles_l423_42395

-- Define a point on a 2D grid
structure Point where
  x : Int
  y : Int

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Calculate the squared distance between two points
def squaredDistance (p1 p2 : Point) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Check if a triangle is isosceles
def isIsosceles (t : Triangle) : Bool :=
  let d1 := squaredDistance t.a t.b
  let d2 := squaredDistance t.b t.c
  let d3 := squaredDistance t.c t.a
  d1 = d2 || d2 = d3 || d3 = d1

-- Define the five triangles
def triangle1 : Triangle := ⟨⟨2, 7⟩, ⟨5, 7⟩, ⟨5, 3⟩⟩
def triangle2 : Triangle := ⟨⟨4, 2⟩, ⟨7, 2⟩, ⟨4, 6⟩⟩
def triangle3 : Triangle := ⟨⟨2, 1⟩, ⟨2, 4⟩, ⟨7, 1⟩⟩
def triangle4 : Triangle := ⟨⟨7, 5⟩, ⟨9, 8⟩, ⟨9, 9⟩⟩
def triangle5 : Triangle := ⟨⟨8, 2⟩, ⟨8, 5⟩, ⟨10, 1⟩⟩

-- Theorem: None of the given triangles are isosceles
theorem no_isosceles_triangles : 
  ¬(isIsosceles triangle1 ∨ isIsosceles triangle2 ∨ isIsosceles triangle3 ∨ 
    isIsosceles triangle4 ∨ isIsosceles triangle5) := by
  sorry

end NUMINAMATH_CALUDE_no_isosceles_triangles_l423_42395


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l423_42307

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) ↔ a ∈ Set.Iio (-1) ∪ Set.Ioi 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l423_42307


namespace NUMINAMATH_CALUDE_rs_value_l423_42360

theorem rs_value (r s : ℝ) (hr : r > 0) (hs : s > 0)
  (h1 : r^3 + s^3 = 1) (h2 : r^6 + s^6 = 15/16) :
  r * s = 1 / Real.rpow 48 (1/3) :=
sorry

end NUMINAMATH_CALUDE_rs_value_l423_42360


namespace NUMINAMATH_CALUDE_moving_trips_l423_42380

theorem moving_trips (total_time : ℕ) (fill_time : ℕ) (drive_time : ℕ) : 
  total_time = 7 * 60 ∧ fill_time = 15 ∧ drive_time = 30 →
  (total_time / (fill_time + 2 * drive_time) : ℕ) = 5 := by
sorry

end NUMINAMATH_CALUDE_moving_trips_l423_42380


namespace NUMINAMATH_CALUDE_square_side_length_equal_perimeter_l423_42313

theorem square_side_length_equal_perimeter (r : ℝ) (s : ℝ) :
  r = 3 →  -- radius of the circle is 3 units
  4 * s = 2 * Real.pi * r →  -- perimeters are equal
  s = 3 * Real.pi / 2 :=  -- side length of the square
by
  sorry

end NUMINAMATH_CALUDE_square_side_length_equal_perimeter_l423_42313


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l423_42358

theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + y^2 = 3/4}
  let tangent_line := {(x, y) : ℝ × ℝ | ∃ k, y = k * x ∧ k^2 = 3}
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let e := Real.sqrt (1 + b^2 / a^2)
  (∃ p q : ℝ × ℝ, p ∈ tangent_line ∧ q ∈ tangent_line ∧ p ∈ hyperbola ∧ q ∈ hyperbola ∧ p ≠ q) →
  e > 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l423_42358


namespace NUMINAMATH_CALUDE_sunset_increase_calculation_l423_42355

/-- The daily increase in sunset time, given initial and final sunset times over a period. -/
def daily_sunset_increase (initial_time final_time : ℕ) (days : ℕ) : ℚ :=
  (final_time - initial_time) / days

/-- Theorem stating that the daily sunset increase is 1.2 minutes under given conditions. -/
theorem sunset_increase_calculation :
  let initial_time := 18 * 60  -- 6:00 PM in minutes since midnight
  let final_time := 18 * 60 + 48  -- 6:48 PM in minutes since midnight
  let days := 40
  daily_sunset_increase initial_time final_time days = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_sunset_increase_calculation_l423_42355


namespace NUMINAMATH_CALUDE_salary_raise_percentage_l423_42361

/-- Calculates the percentage raise given the original and new salaries. -/
def percentage_raise (original : ℚ) (new : ℚ) : ℚ :=
  (new - original) / original * 100

/-- Proves that the percentage raise from $500 to $530 is 6%. -/
theorem salary_raise_percentage :
  percentage_raise 500 530 = 6 := by
  sorry

end NUMINAMATH_CALUDE_salary_raise_percentage_l423_42361


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l423_42363

theorem smallest_n_for_roots_of_unity : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∀ z : ℂ, z^4 - z^2 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^4 - z^2 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l423_42363


namespace NUMINAMATH_CALUDE_viaduct_laying_speed_l423_42303

/-- Proves that the original daily laying length is 300 meters given the conditions of the viaduct construction. -/
theorem viaduct_laying_speed 
  (total_length : ℝ) 
  (total_days : ℝ) 
  (initial_length : ℝ) 
  (h1 : total_length = 4800)
  (h2 : total_days = 9)
  (h3 : initial_length = 600)
  (h4 : ∃ (x : ℝ), (initial_length / x) + ((total_length - initial_length) / (2 * x)) = total_days)
  : ∃ (x : ℝ), x = 300 ∧ (initial_length / x) + ((total_length - initial_length) / (2 * x)) = total_days :=
sorry

end NUMINAMATH_CALUDE_viaduct_laying_speed_l423_42303


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l423_42327

/-- If (a - i) / (1 + i) is a pure imaginary number where a ∈ ℝ, then 3a + 4i is in the first quadrant of the complex plane. -/
theorem complex_number_quadrant (a : ℝ) :
  (((a : ℂ) - I) / (1 + I)).im ≠ 0 ∧ (((a : ℂ) - I) / (1 + I)).re = 0 →
  (3 * a : ℝ) > 0 ∧ 4 > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l423_42327


namespace NUMINAMATH_CALUDE_sweep_probability_is_one_third_l423_42367

/-- Represents the positions of flies on a clock -/
inductive ClockPosition
  | twelve
  | three
  | six
  | nine

/-- Represents a time interval in minutes -/
def TimeInterval : ℕ := 20

/-- Calculates the number of favorable intervals where exactly two flies are swept -/
def favorableIntervals : ℕ := 4 * 5

/-- Total minutes in an hour -/
def totalMinutes : ℕ := 60

/-- The probability of sweeping exactly two flies in the given time interval -/
def sweepProbability : ℚ := favorableIntervals / totalMinutes

theorem sweep_probability_is_one_third :
  sweepProbability = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_sweep_probability_is_one_third_l423_42367


namespace NUMINAMATH_CALUDE_garden_area_difference_l423_42369

-- Define the dimensions of the gardens
def karl_length : ℕ := 30
def karl_width : ℕ := 50
def makenna_length : ℕ := 35
def makenna_width : ℕ := 45

-- Define the areas of the gardens
def karl_area : ℕ := karl_length * karl_width
def makenna_area : ℕ := makenna_length * makenna_width

-- Theorem statement
theorem garden_area_difference :
  makenna_area - karl_area = 75 ∧ makenna_area > karl_area := by
  sorry

end NUMINAMATH_CALUDE_garden_area_difference_l423_42369


namespace NUMINAMATH_CALUDE_product_sum_puzzle_l423_42318

theorem product_sum_puzzle :
  ∃ (a b c : ℤ), (a * b + c = 40) ∧ (a + b ≠ 18) ∧
  (∃ (a' b' c' : ℤ), (a' * b' + c' = 40) ∧ (a' + b' ≠ 18) ∧ (c' ≠ c)) :=
by sorry

end NUMINAMATH_CALUDE_product_sum_puzzle_l423_42318


namespace NUMINAMATH_CALUDE_negation_of_both_even_l423_42308

theorem negation_of_both_even (a b : ℤ) : 
  ¬(Even a ∧ Even b) ↔ ¬(Even a) ∨ ¬(Even b) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_both_even_l423_42308


namespace NUMINAMATH_CALUDE_c_work_rate_l423_42390

def work_rate (days : ℚ) : ℚ := 1 / days

theorem c_work_rate 
  (ab_days : ℚ) 
  (bc_days : ℚ) 
  (ca_days : ℚ) 
  (hab : ab_days = 3) 
  (hbc : bc_days = 4) 
  (hca : ca_days = 6) : 
  ∃ (c_rate : ℚ), 
    c_rate = 1 / 24 ∧ 
    c_rate + work_rate ab_days = work_rate ca_days ∧
    c_rate + work_rate bc_days - work_rate ab_days = work_rate bc_days := by
  sorry

end NUMINAMATH_CALUDE_c_work_rate_l423_42390


namespace NUMINAMATH_CALUDE_books_remaining_l423_42398

/-- Calculates the number of books remaining in Tracy's charity book store -/
theorem books_remaining (initial_books : ℕ) (donors : ℕ) (books_per_donor : ℕ) (borrowed_books : ℕ) : 
  initial_books = 300 → 
  donors = 10 → 
  books_per_donor = 5 → 
  borrowed_books = 140 → 
  initial_books + donors * books_per_donor - borrowed_books = 210 := by
sorry

end NUMINAMATH_CALUDE_books_remaining_l423_42398


namespace NUMINAMATH_CALUDE_min_omega_value_l423_42391

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem min_omega_value (ω φ : ℝ) (h_ω_pos : ω > 0) 
  (h_exists : ∃ x₀ : ℝ, f ω φ (x₀ + 2) - f ω φ x₀ = 4) :
  ω ≥ Real.pi / 2 ∧ ∀ ω' > 0, (∃ x₀' : ℝ, f ω' φ (x₀' + 2) - f ω' φ x₀' = 4) → ω' ≥ Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_min_omega_value_l423_42391


namespace NUMINAMATH_CALUDE_triangle_perimeter_in_divided_square_l423_42376

/-- Given a square of side z divided into a smaller square of side w and four congruent triangles,
    the perimeter of one of these triangles is h + z, where h is the height of the triangle. -/
theorem triangle_perimeter_in_divided_square (z w h : ℝ) :
  z > 0 → w > 0 → h > 0 →
  h + (z - h) = z →  -- The height plus the base of the triangle equals the side of the larger square
  w^2 = h^2 + (z - h)^2 →  -- Pythagoras theorem for the triangle
  (h + z : ℝ) = 2 * h + (z - h) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_in_divided_square_l423_42376


namespace NUMINAMATH_CALUDE_jonathan_book_purchase_l423_42345

theorem jonathan_book_purchase (dictionary_cost dinosaur_book_cost cookbook_cost savings : ℕ) 
  (h1 : dictionary_cost = 11)
  (h2 : dinosaur_book_cost = 19)
  (h3 : cookbook_cost = 7)
  (h4 : savings = 8) :
  dictionary_cost + dinosaur_book_cost + cookbook_cost - savings = 29 := by
  sorry

end NUMINAMATH_CALUDE_jonathan_book_purchase_l423_42345


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l423_42342

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 1

-- State the theorem
theorem quadratic_symmetry (a : ℝ) :
  (f a 1 = 2) → (f a (-1) = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l423_42342


namespace NUMINAMATH_CALUDE_golf_tournament_total_cost_l423_42316

/-- The cost of the golf tournament given the electricity bill cost and additional expenses -/
def golf_tournament_cost (electricity_bill : ℝ) (cell_phone_additional : ℝ) : ℝ :=
  let cell_phone_expense := electricity_bill + cell_phone_additional
  let tournament_additional_cost := 0.2 * cell_phone_expense
  cell_phone_expense + tournament_additional_cost

/-- Theorem stating the total cost of the golf tournament -/
theorem golf_tournament_total_cost :
  golf_tournament_cost 800 400 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_golf_tournament_total_cost_l423_42316


namespace NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l423_42364

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The sum of the interior angles of a hexagon is 720 degrees -/
theorem hexagon_interior_angles_sum : 
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l423_42364


namespace NUMINAMATH_CALUDE_max_perfect_matchings_20gon_l423_42344

/-- Represents a convex polygon with 2n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin (2 * n) → ℝ × ℝ

/-- Represents a triangulation of a convex polygon -/
structure Triangulation (n : ℕ) where
  polygon : ConvexPolygon n
  diagonals : Fin (2 * n - 3) → Fin (2 * n) × Fin (2 * n)

/-- Represents a perfect matching in a triangulation -/
structure PerfectMatching (n : ℕ) where
  triangulation : Triangulation n
  edges : Fin n → Fin (4 * n - 3)

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => fib (n + 1) + fib n

/-- The maximum number of perfect matchings for a convex 2n-gon -/
def maxPerfectMatchings (n : ℕ) : ℕ := fib n

/-- The theorem statement -/
theorem max_perfect_matchings_20gon :
  maxPerfectMatchings 10 = 89 := by sorry

end NUMINAMATH_CALUDE_max_perfect_matchings_20gon_l423_42344


namespace NUMINAMATH_CALUDE_mcpherson_contribution_l423_42374

/-- Calculate Mr. McPherson's contribution to rent and expenses --/
theorem mcpherson_contribution
  (current_rent : ℝ)
  (rent_increase_rate : ℝ)
  (current_monthly_expenses : ℝ)
  (monthly_expenses_increase_rate : ℝ)
  (mrs_mcpherson_contribution_rate : ℝ)
  (h1 : current_rent = 1200)
  (h2 : rent_increase_rate = 0.05)
  (h3 : current_monthly_expenses = 100)
  (h4 : monthly_expenses_increase_rate = 0.03)
  (h5 : mrs_mcpherson_contribution_rate = 0.30) :
  ∃ (mr_mcpherson_contribution : ℝ),
    mr_mcpherson_contribution = 1747.20 ∧
    mr_mcpherson_contribution =
      (1 - mrs_mcpherson_contribution_rate) *
      (current_rent * (1 + rent_increase_rate) +
       12 * current_monthly_expenses * (1 + monthly_expenses_increase_rate)) :=
by
  sorry

end NUMINAMATH_CALUDE_mcpherson_contribution_l423_42374


namespace NUMINAMATH_CALUDE_tangent_line_at_one_two_l423_42384

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (-x - 1) - x
  else Real.exp (x - 1) + x

-- State the theorem
theorem tangent_line_at_one_two :
  (∀ x : ℝ, f x = f (-x)) → -- f is even
  f 1 = 2 → -- (1, 2) lies on the curve
  ∃ m : ℝ, ∀ x : ℝ, (HasDerivAt f m 1 ∧ m = 2) → 
    2 = m * (1 - 1) + f 1 ∧ -- Point-slope form at (1, 2)
    ∀ y : ℝ, y = 2 * x ↔ y - f 1 = m * (x - 1) -- Tangent line equation
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_two_l423_42384


namespace NUMINAMATH_CALUDE_two_numbers_problem_l423_42301

theorem two_numbers_problem (A B : ℝ) (h1 : A + B = 40) (h2 : A * B = 375) (h3 : A / B = 3/2) 
  (h4 : A > 0) (h5 : B > 0) : A = 24 ∧ B = 16 ∧ A - B = 8 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l423_42301


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l423_42305

/-- Given an angle in the second quadrant, its half angle can only be in the first or third quadrant -/
theorem half_angle_quadrant (θ : Real) :
  π / 2 < θ ∧ θ < π →
  (0 < θ / 2 ∧ θ / 2 < π / 2) ∨ (π < θ / 2 ∧ θ / 2 < 3 * π / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l423_42305


namespace NUMINAMATH_CALUDE_tangent_line_inclination_l423_42326

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b

-- Define the derivative of f(x)
def f_prime (a x : ℝ) : ℝ := 3*x^2 - 2*a*x

-- Theorem statement
theorem tangent_line_inclination (a b : ℝ) :
  (f_prime a 1 = -1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_inclination_l423_42326


namespace NUMINAMATH_CALUDE_simplify_expression_l423_42315

theorem simplify_expression (a b : ℝ) : a * (4 * a - b) - (2 * a + b) * (2 * a - b) = b^2 - a * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l423_42315


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l423_42347

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : GeometricSequence a)
    (h_product : a 1 * a 7 * a 13 = 8) :
  a 3 * a 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l423_42347


namespace NUMINAMATH_CALUDE_circle_center_sum_l423_42311

theorem circle_center_sum (x y : ℝ) : 
  (∀ a b : ℝ, (a - x)^2 + (b - y)^2 = (a^2 + b^2 - 12*a + 4*b - 10)) → 
  x + y = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l423_42311


namespace NUMINAMATH_CALUDE_paolo_coconuts_l423_42333

theorem paolo_coconuts (paolo dante : ℕ) : 
  dante = 3 * paolo →  -- Dante has thrice as many coconuts as Paolo
  dante - 10 = 32 →    -- Dante had 32 coconuts left after selling 10
  paolo = 14 :=        -- Paolo had 14 coconuts
by
  sorry

end NUMINAMATH_CALUDE_paolo_coconuts_l423_42333


namespace NUMINAMATH_CALUDE_production_calculation_l423_42389

/-- Calculates the production given the number of workers, hours per day, 
    number of days, efficiency factor, and base production rate -/
def calculate_production (workers : ℕ) (hours_per_day : ℕ) (days : ℕ) 
                         (efficiency_factor : ℚ) (base_rate : ℚ) : ℚ :=
  (workers : ℚ) * (hours_per_day : ℚ) * (days : ℚ) * efficiency_factor * base_rate

theorem production_calculation :
  let initial_workers : ℕ := 10
  let initial_hours : ℕ := 6
  let initial_days : ℕ := 5
  let initial_production : ℕ := 200
  let new_workers : ℕ := 8
  let new_hours : ℕ := 7
  let new_days : ℕ := 4
  let efficiency_increase : ℚ := 11/10

  let base_rate : ℚ := (initial_production : ℚ) / 
    ((initial_workers : ℚ) * (initial_hours : ℚ) * (initial_days : ℚ))

  let new_production : ℚ := calculate_production new_workers new_hours new_days 
                            efficiency_increase base_rate

  new_production = 198 :=
by sorry

end NUMINAMATH_CALUDE_production_calculation_l423_42389


namespace NUMINAMATH_CALUDE_trig_identity_proof_l423_42370

theorem trig_identity_proof (α : Real) (h : Real.tan α = 2) :
  4 * (Real.sin α)^2 - 3 * (Real.sin α) * (Real.cos α) - 5 * (Real.cos α)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l423_42370


namespace NUMINAMATH_CALUDE_correct_operation_l423_42366

theorem correct_operation (a b : ℝ) : 4 * a^2 * b - 2 * b * a^2 = 2 * a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l423_42366


namespace NUMINAMATH_CALUDE_square_sum_from_sum_and_product_l423_42368

theorem square_sum_from_sum_and_product (a b : ℝ) 
  (h1 : a + b = 5) (h2 : a * b = 6) : a^2 + b^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_sum_and_product_l423_42368


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l423_42336

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def consecutive_nonprimes (start : ℕ) : Prop :=
  ∀ k : ℕ, k < 6 → ¬(is_prime (start + k))

theorem smallest_prime_after_six_nonprimes :
  ∀ p : ℕ, is_prime p →
    (∃ start : ℕ, consecutive_nonprimes start ∧ start + 6 < p) →
    p ≥ 127 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l423_42336


namespace NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l423_42356

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if the perimeter is equal to 3(sin A + sin B + sin C),
    then the diameter of its circumcircle is 3. -/
theorem triangle_circumcircle_diameter
  (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a + b + c = 3 * (Real.sin A + Real.sin B + Real.sin C) →
  a / Real.sin A = 2 * R →
  b / Real.sin B = 2 * R →
  c / Real.sin C = 2 * R →
  2 * R = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l423_42356


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l423_42373

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan ((100 : ℝ) * π / 180 - x * π / 180) =
    (Real.sin ((100 : ℝ) * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos ((100 : ℝ) * π / 180) - Real.cos (x * π / 180)) ∧
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l423_42373


namespace NUMINAMATH_CALUDE_problem_statement_l423_42386

theorem problem_statement : (2351 - 2250)^2 / 121 = 84 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l423_42386


namespace NUMINAMATH_CALUDE_certain_number_proof_l423_42302

theorem certain_number_proof (n : ℕ) (h1 : n > 0) :
  let m := 72 * 14
  Nat.gcd m 72 = 72 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l423_42302


namespace NUMINAMATH_CALUDE_nancy_bathroom_flooring_l423_42378

/-- Represents the dimensions of a rectangular area -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The central area of Nancy's bathroom -/
def central_area : Rectangle := { length := 10, width := 10 }

/-- The hallway area of Nancy's bathroom -/
def hallway : Rectangle := { length := 6, width := 4 }

/-- The total area of hardwood flooring in Nancy's bathroom -/
def total_flooring_area : ℝ := area central_area + area hallway

theorem nancy_bathroom_flooring :
  total_flooring_area = 124 := by sorry

end NUMINAMATH_CALUDE_nancy_bathroom_flooring_l423_42378


namespace NUMINAMATH_CALUDE_hyperbola_equation_l423_42375

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (m : ℝ), m * 2 = Real.sqrt 3) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c = Real.sqrt 7) →
  a = 2 ∧ b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l423_42375


namespace NUMINAMATH_CALUDE_triangle_special_condition_l423_42306

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_special_condition (t : Triangle) :
  t.a^2 = 3*t.b^2 + 3*t.c^2 - 2*Real.sqrt 3*t.b*t.c*Real.sin t.A →
  t.C = π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_condition_l423_42306


namespace NUMINAMATH_CALUDE_min_value_C_squared_minus_D_squared_l423_42352

theorem min_value_C_squared_minus_D_squared :
  ∀ (x y z : ℝ), 
  x ≥ 0 → y ≥ 0 → z ≥ 0 →
  x ≤ 1 → y ≤ 2 → z ≤ 3 →
  let C := Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12)
  let D := Real.sqrt (x + 1) + Real.sqrt (y + 2) + Real.sqrt (z + 3)
  ∀ (C' D' : ℝ), C = C' → D = D' →
  C' ^ 2 - D' ^ 2 ≥ 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_C_squared_minus_D_squared_l423_42352


namespace NUMINAMATH_CALUDE_units_digit_of_17_times_24_l423_42371

theorem units_digit_of_17_times_24 : (17 * 24) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_17_times_24_l423_42371


namespace NUMINAMATH_CALUDE_symmetry_sum_l423_42385

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposite
    and their y-coordinates are equal -/
def symmetric_wrt_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetry_sum (a b : ℝ) :
  symmetric_wrt_y_axis (a, -3) (4, b) → a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l423_42385


namespace NUMINAMATH_CALUDE_race_time_calculation_l423_42393

/-- A theorem about a race between two runners --/
theorem race_time_calculation (race_distance : ℝ) (b_time : ℝ) (a_lead : ℝ) (a_time : ℝ) : 
  race_distance = 120 →
  b_time = 45 →
  a_lead = 24 →
  a_time = 56.25 →
  (race_distance / a_time = (race_distance - a_lead) / b_time) := by
sorry

end NUMINAMATH_CALUDE_race_time_calculation_l423_42393


namespace NUMINAMATH_CALUDE_coefficients_of_given_equation_l423_42353

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given quadratic equation 2x² + 3x - 1 = 0 -/
def givenEquation : QuadraticEquation :=
  { a := 2, b := 3, c := -1 }

theorem coefficients_of_given_equation :
  givenEquation.a = 2 ∧ givenEquation.b = 3 ∧ givenEquation.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_given_equation_l423_42353


namespace NUMINAMATH_CALUDE_rectangle_area_l423_42321

/-- The area of a rectangle with length 47.3 cm and width 24 cm is 1135.2 square centimeters. -/
theorem rectangle_area : 
  let length : ℝ := 47.3
  let width : ℝ := 24
  length * width = 1135.2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l423_42321


namespace NUMINAMATH_CALUDE_rachel_furniture_assembly_l423_42328

/-- The number of tables Rachel bought -/
def num_tables : ℕ := 3

theorem rachel_furniture_assembly :
  ∀ (chairs tables : ℕ) (time_per_piece total_time : ℕ),
  chairs = 7 →
  time_per_piece = 4 →
  total_time = 40 →
  total_time = time_per_piece * (chairs + tables) →
  tables = num_tables :=
by sorry

end NUMINAMATH_CALUDE_rachel_furniture_assembly_l423_42328


namespace NUMINAMATH_CALUDE_value_of_a_l423_42346

theorem value_of_a (a : ℝ) : 
  let A : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}
  1 ∈ A → a = -1 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l423_42346


namespace NUMINAMATH_CALUDE_supplementary_angles_problem_l423_42312

theorem supplementary_angles_problem (x y : ℝ) : 
  x + y = 180 → 
  y = x + 18 → 
  y = 99 := by
sorry

end NUMINAMATH_CALUDE_supplementary_angles_problem_l423_42312


namespace NUMINAMATH_CALUDE_smallest_m_perfect_square_and_cube_l423_42332

theorem smallest_m_perfect_square_and_cube : ∃ (m : ℕ), 
  (m > 0) ∧ 
  (∃ (k : ℕ), 5 * m = k * k) ∧ 
  (∃ (l : ℕ), 3 * m = l * l * l) ∧ 
  (∀ (n : ℕ), n > 0 → 
    (∃ (k : ℕ), 5 * n = k * k) → 
    (∃ (l : ℕ), 3 * n = l * l * l) → 
    m ≤ n) ∧
  m = 243 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_perfect_square_and_cube_l423_42332


namespace NUMINAMATH_CALUDE_parallel_vector_with_given_magnitude_l423_42325

/-- Given two vectors a and b in ℝ², where a = (2,1) and b is parallel to a with magnitude 2√5,
    prove that b must be either (4,2) or (-4,-2). -/
theorem parallel_vector_with_given_magnitude (a b : ℝ × ℝ) :
  a = (2, 1) →
  (∃ k : ℝ, b = (k * a.1, k * a.2)) →
  Real.sqrt ((b.1)^2 + (b.2)^2) = 2 * Real.sqrt 5 →
  b = (4, 2) ∨ b = (-4, -2) := by
  sorry

#check parallel_vector_with_given_magnitude

end NUMINAMATH_CALUDE_parallel_vector_with_given_magnitude_l423_42325


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l423_42377

/-- Given a paint mixture ratio of 5:3:7 for yellow:blue:red,
    if 21 quarts of red paint are used, then 9 quarts of blue paint should be used. -/
theorem paint_mixture_ratio (yellow blue red : ℚ) (red_quarts : ℚ) :
  yellow = 5 →
  blue = 3 →
  red = 7 →
  red_quarts = 21 →
  (blue / red) * red_quarts = 9 := by
  sorry


end NUMINAMATH_CALUDE_paint_mixture_ratio_l423_42377


namespace NUMINAMATH_CALUDE_magnitude_of_B_area_of_triangle_l423_42338

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def triangleCondition (t : Triangle) : Prop :=
  2 * t.b * Real.sin t.B = (2 * t.a + t.c) * Real.sin t.A + (2 * t.c + t.a) * Real.sin t.C

-- Theorem for part I
theorem magnitude_of_B (t : Triangle) (h : triangleCondition t) : t.B = 2 * Real.pi / 3 := by
  sorry

-- Theorem for part II
theorem area_of_triangle (t : Triangle) (h1 : triangleCondition t) (h2 : t.b = Real.sqrt 3) (h3 : t.A = Real.pi / 4) :
  (1 / 2) * t.b * t.c * Real.sin t.A = (3 - Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_B_area_of_triangle_l423_42338


namespace NUMINAMATH_CALUDE_smaller_cone_height_equals_frustum_height_l423_42343

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  height : ℝ
  larger_base_area : ℝ
  smaller_base_area : ℝ

/-- Calculates the height of the smaller cone removed to form the frustum -/
def smaller_cone_height (f : Frustum) : ℝ :=
  f.height

/-- Theorem stating that the height of the smaller cone is equal to the frustum's height -/
theorem smaller_cone_height_equals_frustum_height (f : Frustum)
  (h1 : f.height = 18)
  (h2 : f.larger_base_area = 400 * Real.pi)
  (h3 : f.smaller_base_area = 100 * Real.pi) :
  smaller_cone_height f = f.height :=
by sorry

end NUMINAMATH_CALUDE_smaller_cone_height_equals_frustum_height_l423_42343


namespace NUMINAMATH_CALUDE_f_increasing_interval_l423_42339

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 9) / Real.log (1/3)

theorem f_increasing_interval :
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ < -3 → f x₁ < f x₂ :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_interval_l423_42339


namespace NUMINAMATH_CALUDE_emilys_flowers_l423_42379

theorem emilys_flowers (flower_cost : ℕ) (total_spent : ℕ) : 
  flower_cost = 3 →
  total_spent = 12 →
  ∃ (roses daisies : ℕ), 
    roses = daisies ∧ 
    roses + daisies = total_spent / flower_cost :=
by
  sorry

end NUMINAMATH_CALUDE_emilys_flowers_l423_42379


namespace NUMINAMATH_CALUDE_special_polynomial_value_l423_42320

/-- A polynomial of degree n satisfying the given condition -/
def SpecialPolynomial (n : ℕ) : (ℕ → ℚ) := fun k => 1 / (Nat.choose (n+1) k)

/-- The theorem stating the value of p(n+1) for the special polynomial -/
theorem special_polynomial_value (n : ℕ) :
  let p := SpecialPolynomial n
  p (n+1) = if Even n then 1 else 0 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_value_l423_42320


namespace NUMINAMATH_CALUDE_circles_symmetric_line_l423_42335

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y + 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y + 7 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Theorem statement
theorem circles_symmetric_line :
  ∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → line_l x y :=
by sorry

end NUMINAMATH_CALUDE_circles_symmetric_line_l423_42335


namespace NUMINAMATH_CALUDE_not_polite_and_power_of_two_polite_or_power_of_two_l423_42388

/-- A number is polite if it can be written as the sum of consecutive integers from m to n, where m < n. -/
def IsPolite (N : ℕ) : Prop :=
  ∃ m n : ℕ, m < n ∧ N = (n * (n + 1) - m * (m - 1)) / 2

/-- A number is a power of two if it can be written as 2^ℓ for some non-negative integer ℓ. -/
def IsPowerOfTwo (N : ℕ) : Prop :=
  ∃ ℓ : ℕ, N = 2^ℓ

/-- No number is both polite and a power of two. -/
theorem not_polite_and_power_of_two (N : ℕ) : ¬(IsPolite N ∧ IsPowerOfTwo N) := by
  sorry

/-- Every positive integer is either polite or a power of two. -/
theorem polite_or_power_of_two (N : ℕ) : N > 0 → IsPolite N ∨ IsPowerOfTwo N := by
  sorry

end NUMINAMATH_CALUDE_not_polite_and_power_of_two_polite_or_power_of_two_l423_42388


namespace NUMINAMATH_CALUDE_password_factorization_l423_42392

theorem password_factorization (a b c d : ℝ) :
  (a^2 - b^2) * c^2 - (a^2 - b^2) * d^2 = (a + b) * (a - b) * (c + d) * (c - d) := by
  sorry

end NUMINAMATH_CALUDE_password_factorization_l423_42392


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l423_42348

/-- An isosceles trapezoid with perpendicular diagonals -/
structure IsoscelesTrapezoid where
  /-- The length of the longer base -/
  a : ℝ
  /-- The length of the shorter base -/
  b : ℝ
  /-- The height of the trapezoid -/
  h : ℝ
  /-- The condition that the trapezoid is isosceles -/
  isIsosceles : True
  /-- The condition that the diagonals are perpendicular -/
  diagonalsPerpendicular : True
  /-- The midline length is 5 -/
  midline_eq : (a + b) / 2 = 5

/-- The area of an isosceles trapezoid with perpendicular diagonals and midline length 5 is 25 -/
theorem isosceles_trapezoid_area (T : IsoscelesTrapezoid) : (T.a + T.b) * T.h / 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l423_42348


namespace NUMINAMATH_CALUDE_kim_thursday_sales_l423_42362

/-- The number of boxes Kim sold on Tuesday -/
def tuesday_sales : ℕ := 4800

/-- The number of boxes Kim sold on Wednesday -/
def wednesday_sales : ℕ := tuesday_sales / 2

/-- The number of boxes Kim sold on Thursday -/
def thursday_sales : ℕ := wednesday_sales / 2

/-- Theorem stating that Kim sold 1200 boxes on Thursday -/
theorem kim_thursday_sales : thursday_sales = 1200 := by
  sorry

end NUMINAMATH_CALUDE_kim_thursday_sales_l423_42362


namespace NUMINAMATH_CALUDE_A_share_is_one_third_l423_42341

structure Partnership where
  initial_investment : ℝ
  total_gain : ℝ

def investment_share (p : Partnership) (months : ℝ) (multiplier : ℝ) : ℝ :=
  p.initial_investment * multiplier * months

theorem A_share_is_one_third (p : Partnership) :
  p.total_gain = 12000 →
  investment_share p 12 1 = investment_share p 6 2 →
  investment_share p 12 1 = investment_share p 4 3 →
  investment_share p 12 1 = p.total_gain / 3 := by
sorry

end NUMINAMATH_CALUDE_A_share_is_one_third_l423_42341


namespace NUMINAMATH_CALUDE_four_digit_multiples_of_seven_l423_42382

theorem four_digit_multiples_of_seven : 
  (Finset.filter (fun n => n % 7 = 0) (Finset.range 9000)).card = 1286 :=
by
  sorry


end NUMINAMATH_CALUDE_four_digit_multiples_of_seven_l423_42382


namespace NUMINAMATH_CALUDE_no_cracked_seashells_l423_42322

theorem no_cracked_seashells (tim_shells sally_shells total_shells : ℕ) 
  (h1 : tim_shells = 37)
  (h2 : sally_shells = 13)
  (h3 : total_shells = 50)
  (h4 : tim_shells + sally_shells = total_shells) :
  total_shells - (tim_shells + sally_shells) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_cracked_seashells_l423_42322


namespace NUMINAMATH_CALUDE_thief_speed_calculation_l423_42340

/-- The speed of the thief's car in km/h -/
def thief_speed : ℝ := 43.75

/-- The head start time of the thief in hours -/
def head_start : ℝ := 0.5

/-- The speed of the owner's bike in km/h -/
def owner_speed : ℝ := 50

/-- The total time until the owner overtakes the thief in hours -/
def total_time : ℝ := 4

theorem thief_speed_calculation :
  thief_speed * total_time = owner_speed * (total_time - head_start) := by sorry

#check thief_speed_calculation

end NUMINAMATH_CALUDE_thief_speed_calculation_l423_42340


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_two_right_triangle_condition_l423_42331

def f (a : ℝ) (x : ℝ) := |x + 1| - |a * x - 3|

theorem solution_set_when_a_eq_two :
  {x : ℝ | f 2 x > 1} = {x : ℝ | 1 < x ∧ x < 3} := by sorry

theorem right_triangle_condition (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, f a x = y ∧ f a y = 0 ∧ x ≠ y ∧ (x - y)^2 + (f a x)^2 = (x - y)^2 + y^2) →
  a = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_two_right_triangle_condition_l423_42331


namespace NUMINAMATH_CALUDE_projection_matrix_values_l423_42396

def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

theorem projection_matrix_values :
  ∀ (a c : ℚ),
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![a, 18/45; c, 27/45]
  is_projection_matrix P →
  a = 1/5 ∧ c = 2/5 := by
sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l423_42396


namespace NUMINAMATH_CALUDE_roots_equation_value_l423_42359

theorem roots_equation_value (x₁ x₂ : ℝ) 
  (h₁ : 3 * x₁^2 - 2 * x₁ - 4 = 0)
  (h₂ : 3 * x₂^2 - 2 * x₂ - 4 = 0)
  (h₃ : x₁ ≠ x₂) :
  3 * x₁^2 + 2 * x₂ = 16/3 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_value_l423_42359


namespace NUMINAMATH_CALUDE_parallelogram_opposite_sides_parallel_equal_l423_42323

-- Define a parallelogram
structure Parallelogram :=
  (vertices : Fin 4 → ℝ × ℝ)
  (is_parallelogram : 
    (vertices 0 - vertices 1 = vertices 3 - vertices 2) ∧
    (vertices 0 - vertices 3 = vertices 1 - vertices 2))

-- Define the property of having parallel and equal opposite sides
def has_parallel_equal_opposite_sides (p : Parallelogram) : Prop :=
  (p.vertices 0 - p.vertices 1 = p.vertices 3 - p.vertices 2) ∧
  (p.vertices 0 - p.vertices 3 = p.vertices 1 - p.vertices 2)

-- Theorem stating that all parallelograms have parallel and equal opposite sides
theorem parallelogram_opposite_sides_parallel_equal (p : Parallelogram) :
  has_parallel_equal_opposite_sides p :=
by
  sorry

-- Note: Rectangles, rhombuses, and squares are special cases of parallelograms,
-- so this theorem applies to them as well.

end NUMINAMATH_CALUDE_parallelogram_opposite_sides_parallel_equal_l423_42323


namespace NUMINAMATH_CALUDE_tan_105_degrees_l423_42383

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l423_42383


namespace NUMINAMATH_CALUDE_slow_dancers_count_l423_42387

theorem slow_dancers_count (total_kids : ℕ) (non_slow_dancers : ℕ) : 
  total_kids = 140 → 
  non_slow_dancers = 10 → 
  (total_kids / 4 : ℕ) - non_slow_dancers = 25 := by
  sorry

end NUMINAMATH_CALUDE_slow_dancers_count_l423_42387


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l423_42349

theorem sqrt_sum_problem (x : ℝ) (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) :
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l423_42349


namespace NUMINAMATH_CALUDE_no_perfect_squares_l423_42300

theorem no_perfect_squares (n : ℕ+) : 
  ¬(∃ (a b c : ℕ), (2 * n.val^2 - 1 = a^2) ∧ (3 * n.val^2 - 1 = b^2) ∧ (6 * n.val^2 - 1 = c^2)) :=
by sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l423_42300


namespace NUMINAMATH_CALUDE_hostel_provisions_l423_42394

-- Define the initial number of men
def initial_men : ℕ := 250

-- Define the number of days provisions last initially
def initial_days : ℕ := 36

-- Define the number of men who left
def men_left : ℕ := 50

-- Define the number of days provisions last after men left
def new_days : ℕ := 45

-- Theorem statement
theorem hostel_provisions :
  initial_men * initial_days = (initial_men - men_left) * new_days :=
by sorry

end NUMINAMATH_CALUDE_hostel_provisions_l423_42394


namespace NUMINAMATH_CALUDE_min_cut_length_40x30_paper_l423_42330

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the problem setup -/
structure PaperCutProblem where
  paper : Rectangle
  inner_rectangle : Rectangle
  num_cuts : ℕ

/-- The minimum total length of cuts for the given problem -/
def min_cut_length (problem : PaperCutProblem) : ℕ := 
  2 * problem.paper.width + 2 * problem.paper.height

/-- Theorem stating the minimum cut length for the specific problem -/
theorem min_cut_length_40x30_paper (problem : PaperCutProblem) 
  (h1 : problem.paper = ⟨40, 30⟩) 
  (h2 : problem.inner_rectangle = ⟨10, 5⟩) 
  (h3 : problem.num_cuts = 4) : 
  min_cut_length problem = 140 := by
  sorry

#check min_cut_length_40x30_paper

end NUMINAMATH_CALUDE_min_cut_length_40x30_paper_l423_42330


namespace NUMINAMATH_CALUDE_half_dollar_percentage_l423_42365

def nickel_value : ℚ := 5
def quarter_value : ℚ := 25
def half_dollar_value : ℚ := 50

def num_nickels : ℕ := 75
def num_half_dollars : ℕ := 40
def num_quarters : ℕ := 30

def total_value : ℚ := 
  num_nickels * nickel_value + 
  num_half_dollars * half_dollar_value + 
  num_quarters * quarter_value

def half_dollar_total : ℚ := num_half_dollars * half_dollar_value

theorem half_dollar_percentage : 
  (half_dollar_total / total_value) * 100 = 64 := by sorry

end NUMINAMATH_CALUDE_half_dollar_percentage_l423_42365


namespace NUMINAMATH_CALUDE_range_of_f_l423_42350

-- Define the function
def f (x : ℝ) : ℝ := -x^2 - 6*x - 5

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Iic 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l423_42350


namespace NUMINAMATH_CALUDE_probability_calculation_l423_42324

/-- The probability of selecting exactly 2 purple and 2 orange marbles -/
def probability_two_purple_two_orange : ℚ :=
  66 / 1265

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 8

/-- The number of purple marbles in the bag -/
def purple_marbles : ℕ := 12

/-- The number of orange marbles in the bag -/
def orange_marbles : ℕ := 5

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := green_marbles + purple_marbles + orange_marbles

/-- The number of marbles selected -/
def selected_marbles : ℕ := 4

theorem probability_calculation :
  probability_two_purple_two_orange = 
    (Nat.choose purple_marbles 2 * Nat.choose orange_marbles 2) / 
    Nat.choose total_marbles selected_marbles :=
by
  sorry

end NUMINAMATH_CALUDE_probability_calculation_l423_42324


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l423_42354

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ a₂ a₃ : ℕ) (h1 : a₁ = 3) (h2 : a₂ = 7) (h3 : a₃ = 11) :
  arithmetic_sequence a₁ (a₂ - a₁) 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l423_42354


namespace NUMINAMATH_CALUDE_evaluate_expression_l423_42310

theorem evaluate_expression : 4^4 - 4 * 4^3 + 6 * 4^2 - 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l423_42310


namespace NUMINAMATH_CALUDE_parallel_lines_angle_measure_l423_42337

/-- Given two parallel lines intersected by a transversal, 
    if one angle is (x+40)° and the other is (3x-40)°, 
    then the first angle measures 85°. -/
theorem parallel_lines_angle_measure :
  ∀ (x : ℝ) (α β : ℝ),
  α = x + 40 →
  β = 3*x - 40 →
  α + β = 180 →
  α = 85 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_angle_measure_l423_42337


namespace NUMINAMATH_CALUDE_min_area_APQB_l423_42329

/-- Parabola Γ defined by y² = 8x -/
def Γ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- Focus of the parabola Γ -/
def F : ℝ × ℝ := (2, 0)

/-- Line l passing through F -/
def l (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = m * p.2 + 2}

/-- Points A and B are intersections of Γ and l -/
def A (m : ℝ) : ℝ × ℝ := sorry

def B (m : ℝ) : ℝ × ℝ := sorry

/-- Tangent line to Γ at point (x, y) -/
def tangentLine (x y : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 * y = 4 * (p.1 + x)}

/-- Point P is the intersection of tangent at A with y-axis -/
def P (m : ℝ) : ℝ := sorry

/-- Point Q is the intersection of tangent at B with y-axis -/
def Q (m : ℝ) : ℝ := sorry

/-- Area of quadrilateral APQB -/
def areaAPQB (m : ℝ) : ℝ := sorry

/-- The minimum area of quadrilateral APQB is 12 -/
theorem min_area_APQB : 
  ∀ m : ℝ, areaAPQB m ≥ 12 ∧ ∃ m₀ : ℝ, areaAPQB m₀ = 12 :=
sorry

end NUMINAMATH_CALUDE_min_area_APQB_l423_42329


namespace NUMINAMATH_CALUDE_car_distance_in_yards_l423_42309

/-- Proves the distance traveled by a car in yards over 60 minutes -/
theorem car_distance_in_yards
  (b : ℝ) (s : ℝ) (h_s_pos : s > 0) :
  let feet_per_s_seconds : ℝ := 5 * b / 12
  let seconds_in_hour : ℝ := 60 * 60
  let feet_in_yard : ℝ := 3
  let distance_feet : ℝ := feet_per_s_seconds * seconds_in_hour / s
  let distance_yards : ℝ := distance_feet / feet_in_yard
  distance_yards = 500 * b / s :=
by sorry


end NUMINAMATH_CALUDE_car_distance_in_yards_l423_42309


namespace NUMINAMATH_CALUDE_choose_four_from_nine_l423_42351

theorem choose_four_from_nine (n : ℕ) (k : ℕ) : n = 9 ∧ k = 4 → Nat.choose n k = 126 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_nine_l423_42351


namespace NUMINAMATH_CALUDE_cosine_of_angle_l423_42314

def a : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, -3)

theorem cosine_of_angle (t : ℝ) : 
  let b : ℝ × ℝ := (3, t)
  (b.1 * c.1 + b.2 * c.2 = 0) →  -- b ⊥ c
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = Real.sqrt 2 / 10 := by
sorry

end NUMINAMATH_CALUDE_cosine_of_angle_l423_42314


namespace NUMINAMATH_CALUDE_time_to_write_rearrangements_l423_42317

def name_length : ℕ := 5
def rearrangements_per_minute : ℕ := 20

theorem time_to_write_rearrangements :
  (Nat.factorial name_length / rearrangements_per_minute : ℚ) / 60 = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_time_to_write_rearrangements_l423_42317


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l423_42304

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_angle_calculation (t : Triangle) :
  t.A = 60 ∧ t.B = 2 * t.C ∧ t.A + t.B + t.C = 180 → t.B = 80 :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_calculation_l423_42304


namespace NUMINAMATH_CALUDE_sum_of_first_10_common_elements_l423_42357

/-- Arithmetic progression with first term 5 and common difference 3 -/
def ap (n : ℕ) : ℕ := 5 + 3 * n

/-- Geometric progression with first term 10 and common ratio 2 -/
def gp (k : ℕ) : ℕ := 10 * 2^k

/-- The sequence of common elements between ap and gp -/
def common_sequence (n : ℕ) : ℕ := 20 * 4^n

theorem sum_of_first_10_common_elements : 
  (Finset.range 10).sum common_sequence = 6990500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_10_common_elements_l423_42357


namespace NUMINAMATH_CALUDE_candidate_a_vote_percentage_l423_42381

theorem candidate_a_vote_percentage
  (total_voters : ℕ)
  (democrat_percentage : ℚ)
  (republican_percentage : ℚ)
  (democrat_for_a_percentage : ℚ)
  (republican_for_a_percentage : ℚ)
  (h1 : democrat_percentage = 60 / 100)
  (h2 : republican_percentage = 1 - democrat_percentage)
  (h3 : democrat_for_a_percentage = 70 / 100)
  (h4 : republican_for_a_percentage = 20 / 100)
  : (democrat_percentage * democrat_for_a_percentage +
     republican_percentage * republican_for_a_percentage) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_candidate_a_vote_percentage_l423_42381


namespace NUMINAMATH_CALUDE_book_sale_gain_percentage_l423_42399

def total_cost : ℚ := 420
def cost_loss_book : ℚ := 245
def loss_percentage : ℚ := 15 / 100

theorem book_sale_gain_percentage :
  let cost_gain_book := total_cost - cost_loss_book
  let selling_price := cost_loss_book * (1 - loss_percentage)
  let gain_percentage := (selling_price - cost_gain_book) / cost_gain_book * 100
  gain_percentage = 19 := by sorry

end NUMINAMATH_CALUDE_book_sale_gain_percentage_l423_42399


namespace NUMINAMATH_CALUDE_subset_cardinality_inequality_l423_42397

theorem subset_cardinality_inequality (n m : ℕ) (A : Fin m → Finset (Fin n)) :
  (∀ i : Fin m, ¬ (30 ∣ (A i).card)) →
  (∀ i j : Fin m, i ≠ j → (30 ∣ (A i ∩ A j).card)) →
  2 * m - m / 30 ≤ 3 * n :=
by sorry

end NUMINAMATH_CALUDE_subset_cardinality_inequality_l423_42397


namespace NUMINAMATH_CALUDE_total_spent_equals_42_33_l423_42319

/-- The total amount Joan spent on clothing -/
def total_spent : ℚ := 15 + 14.82 + 12.51

/-- Theorem stating that the total amount spent is equal to $42.33 -/
theorem total_spent_equals_42_33 : total_spent = 42.33 := by sorry

end NUMINAMATH_CALUDE_total_spent_equals_42_33_l423_42319


namespace NUMINAMATH_CALUDE_oliver_birthday_gift_l423_42372

/-- The amount of money Oliver's friend gave him on his birthday --/
def friend_gift (initial_amount savings frisbee_cost puzzle_cost final_amount : ℕ) : ℕ :=
  final_amount - (initial_amount + savings - frisbee_cost - puzzle_cost)

theorem oliver_birthday_gift :
  friend_gift 9 5 4 3 15 = 8 :=
by sorry

end NUMINAMATH_CALUDE_oliver_birthday_gift_l423_42372
