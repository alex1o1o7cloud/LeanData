import Mathlib

namespace NUMINAMATH_CALUDE_computer_price_reduction_l1193_119307

/-- Given a computer with original price x, after reducing it by m yuan and then by 20%,
    resulting in a final price of n yuan, prove that the original price x is equal to (5/4)n + m. -/
theorem computer_price_reduction (x m n : ℝ) (h : (x - m) * (1 - 0.2) = n) :
  x = (5/4) * n + m := by
  sorry

end NUMINAMATH_CALUDE_computer_price_reduction_l1193_119307


namespace NUMINAMATH_CALUDE_roberto_outfits_l1193_119345

/-- The number of pairs of trousers Roberto has -/
def trousers : ℕ := 5

/-- The number of shirts Roberto has -/
def shirts : ℕ := 6

/-- The number of jackets Roberto has -/
def jackets : ℕ := 4

/-- The number of pairs of shoes Roberto has -/
def shoes : ℕ := 3

/-- The number of jackets with shoe restrictions -/
def restricted_jackets : ℕ := 1

/-- The number of shoes that can be worn with the restricted jacket -/
def shoes_per_restricted_jacket : ℕ := 2

/-- The total number of outfits Roberto can put together -/
def total_outfits : ℕ := trousers * shirts * (
  (jackets - restricted_jackets) * shoes +
  restricted_jackets * shoes_per_restricted_jacket
)

theorem roberto_outfits :
  total_outfits = 330 :=
by sorry

end NUMINAMATH_CALUDE_roberto_outfits_l1193_119345


namespace NUMINAMATH_CALUDE_train_length_l1193_119392

/-- The length of a train given its speed and time to pass a point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 36 → time_s = 5.5 → speed_kmh * (1000 / 3600) * time_s = 55 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1193_119392


namespace NUMINAMATH_CALUDE_area_of_U_l1193_119385

/-- A regular octagon centered at the origin in the complex plane -/
def regularOctagon : Set ℂ :=
  sorry

/-- The distance between opposite sides of the octagon is 2 units -/
def oppositeDistanceIs2 : ℝ :=
  sorry

/-- One pair of sides of the octagon is parallel to the real axis -/
def sideParallelToRealAxis : Prop :=
  sorry

/-- The region outside the octagon -/
def T : Set ℂ :=
  {z : ℂ | z ∉ regularOctagon}

/-- The set of reciprocals of points in T -/
def U : Set ℂ :=
  {w : ℂ | ∃ z ∈ T, w = 1 / z}

/-- The area of a set in the complex plane -/
def area : Set ℂ → ℝ :=
  sorry

theorem area_of_U : area U = π / 2 :=
  sorry

end NUMINAMATH_CALUDE_area_of_U_l1193_119385


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1193_119387

/-- Proves that given a sum P put at simple interest for 4 years, 
    if increasing the interest rate by 2% results in $56 more interest, 
    then P = $700. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 2) * 4) / 100 - (P * R * 4) / 100 = 56 → P = 700 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1193_119387


namespace NUMINAMATH_CALUDE_square_divisibility_l1193_119310

theorem square_divisibility (n : ℕ+) (h : ∀ q : ℕ+, q ∣ n → q ≤ 12) :
  144 ∣ n^2 := by
sorry

end NUMINAMATH_CALUDE_square_divisibility_l1193_119310


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l1193_119395

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) : 
  (x * y ≥ 64 ∧ x + y ≥ 18) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ * y₁ = 64 ∧ x₂ + y₂ = 18 ∧
   2/x₁ + 8/y₁ = 1 ∧ 2/x₂ + 8/y₂ = 1 ∧
   x₁ > 0 ∧ y₁ > 0 ∧ x₂ > 0 ∧ y₂ > 0) :=
by sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l1193_119395


namespace NUMINAMATH_CALUDE_three_face_painted_subcubes_count_l1193_119311

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  painted_faces : ℕ := 6

/-- Counts the number of subcubes with at least three painted faces -/
def count_three_face_painted_subcubes (c : PaintedCube 4) : ℕ :=
  8

/-- Theorem: In a 4x4x4 painted cube, there are exactly 8 subcubes with at least three painted faces -/
theorem three_face_painted_subcubes_count (c : PaintedCube 4) :
  count_three_face_painted_subcubes c = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_face_painted_subcubes_count_l1193_119311


namespace NUMINAMATH_CALUDE_store_rooms_problem_l1193_119381

theorem store_rooms_problem (x : ℕ) : 
  (∃ (total_guests : ℕ), 
    total_guests = 7 * x + 7 ∧ 
    total_guests = 9 * (x - 1)) → 
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_store_rooms_problem_l1193_119381


namespace NUMINAMATH_CALUDE_prop_p_prop_q_l1193_119354

-- Define the set of real numbers excluding 1
def RealExcludingOne : Set ℝ := {x : ℝ | x ∈ (Set.Ioo 0 1) ∪ (Set.Ioi 1)}

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Proposition p
theorem prop_p : ∀ a ∈ RealExcludingOne, log a 1 = 0 := by sorry

-- Proposition q
theorem prop_q : ∀ x : ℕ, x^3 ≥ x^2 := by sorry

end NUMINAMATH_CALUDE_prop_p_prop_q_l1193_119354


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1193_119371

-- Define the sets A and B
def A : Set ℝ := {x | |x| < 3}
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1193_119371


namespace NUMINAMATH_CALUDE_prime_square_sum_l1193_119366

theorem prime_square_sum (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ ∃ (n : ℕ), p^q + p^r = n^2 ↔ 
  ((p = 2 ∧ q = 2 ∧ r = 5) ∨ 
   (p = 2 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 3) ∨ 
   (p = 3 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 2 ∧ q = r ∧ q ≥ 3 ∧ Prime q)) :=
by sorry

end NUMINAMATH_CALUDE_prime_square_sum_l1193_119366


namespace NUMINAMATH_CALUDE_cube_pyramid_equal_volume_l1193_119359

/-- Given a cube with edge length 6 and a square-based pyramid with base edge length 10,
    if their volumes are equal, then the height of the pyramid is 162/25. -/
theorem cube_pyramid_equal_volume (h : ℚ) : 
  (6 : ℚ)^3 = (1/3 : ℚ) * 10^2 * h → h = 162/25 := by
  sorry

end NUMINAMATH_CALUDE_cube_pyramid_equal_volume_l1193_119359


namespace NUMINAMATH_CALUDE_supermarket_product_sales_l1193_119305

-- Define the linear function
def sales_quantity (x : ℝ) : ℝ := -2 * x + 200

-- Define the profit function
def profit (x : ℝ) : ℝ := (sales_quantity x) * (x - 60)

-- Define the given data points
def data_points : List (ℝ × ℝ) := [(65, 70), (70, 60), (75, 50), (80, 40)]

theorem supermarket_product_sales :
  -- 1. The function fits the given data points
  (∀ (point : ℝ × ℝ), point ∈ data_points → sales_quantity point.1 = point.2) ∧
  -- 2. The selling price of 70 or 90 dollars per kilogram results in a daily profit of $600
  (profit 70 = 600 ∧ profit 90 = 600) ∧
  -- 3. The maximum daily profit is $800, achieved at a selling price of 80 dollars per kilogram
  (∀ (x : ℝ), profit x ≤ 800) ∧ (profit 80 = 800) :=
by sorry

end NUMINAMATH_CALUDE_supermarket_product_sales_l1193_119305


namespace NUMINAMATH_CALUDE_circle_chords_and_regions_l1193_119388

/-- The number of chords that can be drawn between n points on a circle's circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- The number of regions formed inside a circle by chords connecting n points on its circumference -/
def num_regions (n : ℕ) : ℕ := 1 + n.choose 2 + n.choose 4

theorem circle_chords_and_regions (n : ℕ) (h : n = 10) :
  num_chords n = 45 ∧ num_regions n = 256 := by
  sorry

#eval num_chords 10
#eval num_regions 10

end NUMINAMATH_CALUDE_circle_chords_and_regions_l1193_119388


namespace NUMINAMATH_CALUDE_min_phi_for_odd_function_l1193_119339

open Real

theorem min_phi_for_odd_function (φ : ℝ) : 
  (φ > 0 ∧ 
   (∀ x, cos (π * x - π * φ - π / 3) = -cos (π * (-x) - π * φ - π / 3))) 
  ↔ 
  φ = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_min_phi_for_odd_function_l1193_119339


namespace NUMINAMATH_CALUDE_general_term_is_2n_l1193_119355

/-- An increasing arithmetic sequence with specific properties -/
def IncreasingArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) > a n) ∧ 
  (∃ d > 0, ∀ n, a (n + 1) = a n + d) ∧
  (a 1 = 2) ∧
  (a 2 ^ 2 = a 5 + 6)

/-- The general term of the sequence is 2n -/
theorem general_term_is_2n (a : ℕ → ℝ) 
    (h : IncreasingArithmeticSequence a) : 
    ∀ n : ℕ, a n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_general_term_is_2n_l1193_119355


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1193_119327

theorem nested_fraction_evaluation :
  2 + (3 / (4 + (5 / (6 + 7/8)))) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1193_119327


namespace NUMINAMATH_CALUDE_mark_spent_40_l1193_119338

/-- The total amount Mark spent on tomatoes and apples -/
def total_spent (tomato_price : ℝ) (tomato_weight : ℝ) (apple_price : ℝ) (apple_weight : ℝ) : ℝ :=
  tomato_price * tomato_weight + apple_price * apple_weight

/-- Theorem stating that Mark spent $40 in total -/
theorem mark_spent_40 : 
  total_spent 5 2 6 5 = 40 := by sorry

end NUMINAMATH_CALUDE_mark_spent_40_l1193_119338


namespace NUMINAMATH_CALUDE_oak_grove_books_after_donations_l1193_119360

/-- Represents the number of books in Oak Grove libraries -/
structure OakGroveLibraries where
  public_library : ℕ
  school_libraries : ℕ
  community_center : ℕ

/-- Calculates the total number of books after donations -/
def total_books_after_donations (libs : OakGroveLibraries) (public_donation : ℕ) (community_donation : ℕ) : ℕ :=
  libs.public_library + libs.school_libraries + libs.community_center - public_donation - community_donation

/-- Theorem stating the total number of books after donations -/
theorem oak_grove_books_after_donations :
  let initial_libraries : OakGroveLibraries := {
    public_library := 1986,
    school_libraries := 5106,
    community_center := 3462
  }
  let public_donation : ℕ := 235
  let community_donation : ℕ := 328
  total_books_after_donations initial_libraries public_donation community_donation = 9991 := by
  sorry


end NUMINAMATH_CALUDE_oak_grove_books_after_donations_l1193_119360


namespace NUMINAMATH_CALUDE_gasoline_consumption_rate_l1193_119347

/-- Represents the gasoline consumption problem --/
structure GasolineProblem where
  initial_gasoline : ℝ
  supermarket_distance : ℝ
  farm_distance : ℝ
  partial_farm_trip : ℝ
  final_gasoline : ℝ

/-- Calculates the total distance traveled --/
def total_distance (p : GasolineProblem) : ℝ :=
  2 * p.supermarket_distance + 2 * p.partial_farm_trip + p.farm_distance

/-- Calculates the total gasoline consumed --/
def gasoline_consumed (p : GasolineProblem) : ℝ :=
  p.initial_gasoline - p.final_gasoline

/-- Theorem stating the gasoline consumption rate --/
theorem gasoline_consumption_rate (p : GasolineProblem) 
  (h1 : p.initial_gasoline = 12)
  (h2 : p.supermarket_distance = 5)
  (h3 : p.farm_distance = 6)
  (h4 : p.partial_farm_trip = 2)
  (h5 : p.final_gasoline = 2) :
  total_distance p / gasoline_consumed p = 2 := by sorry

end NUMINAMATH_CALUDE_gasoline_consumption_rate_l1193_119347


namespace NUMINAMATH_CALUDE_no_blue_in_red_triangle_l1193_119398

-- Define the color of a point
inductive Color
| Red
| Blue

-- Define a point in the plane with integer coordinates
structure Point where
  x : Int
  y : Int

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define a predicate for a point being inside a triangle
def inside_triangle (p a b c : Point) : Prop := sorry

-- State the conditions
axiom condition1 : ∀ (p : Point), coloring p = Color.Red ∨ coloring p = Color.Blue

axiom condition2 : ∀ (p q : Point),
  coloring p = Color.Red → coloring q = Color.Red →
  ∀ (r : Point), inside_triangle r p q q → coloring r ≠ Color.Blue

axiom condition3 : ∀ (p q : Point),
  coloring p = Color.Blue → coloring q = Color.Blue →
  distance p q = 2 →
  coloring {x := (p.x + q.x) / 2, y := (p.y + q.y) / 2} = Color.Blue

-- State the theorem
theorem no_blue_in_red_triangle (a b c : Point) :
  coloring a = Color.Red → coloring b = Color.Red → coloring c = Color.Red →
  ∀ (p : Point), inside_triangle p a b c → coloring p ≠ Color.Blue :=
sorry

end NUMINAMATH_CALUDE_no_blue_in_red_triangle_l1193_119398


namespace NUMINAMATH_CALUDE_expansion_coefficients_l1193_119341

theorem expansion_coefficients (m : ℝ) (n : ℕ) :
  m > 0 →
  (1 : ℝ) + n + (n * (n - 1) / 2) = 37 →
  m ^ 2 * (Nat.choose n 6) = 112 →
  n = 8 ∧ m = 2 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l1193_119341


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1193_119300

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.sin x > 1) ↔ (∃ x₀ : ℝ, Real.sin x₀ ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1193_119300


namespace NUMINAMATH_CALUDE_cube_nested_square_root_l1193_119348

theorem cube_nested_square_root : (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_nested_square_root_l1193_119348


namespace NUMINAMATH_CALUDE_fib_150_mod_7_l1193_119318

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_150_mod_7 : fib 150 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_7_l1193_119318


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1193_119336

/-- The x-intercept of a line is the point where the line crosses the x-axis (i.e., where y = 0) -/
def x_intercept (a b c : ℚ) : ℚ × ℚ :=
  let x := c / a
  (x, 0)

/-- The line equation is in the form ax + by = c -/
def line_equation (a b c : ℚ) (x y : ℚ) : Prop :=
  a * x + b * y = c

theorem x_intercept_of_line :
  x_intercept 5 (-7) 35 = (7, 0) ∧
  line_equation 5 (-7) 35 (x_intercept 5 (-7) 35).1 (x_intercept 5 (-7) 35).2 :=
sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1193_119336


namespace NUMINAMATH_CALUDE_charles_whistles_l1193_119394

/-- Given that Sean has 45 whistles and 32 more whistles than Charles,
    prove that Charles has 13 whistles. -/
theorem charles_whistles (sean_whistles : ℕ) (difference : ℕ) :
  sean_whistles = 45 →
  difference = 32 →
  sean_whistles - difference = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_charles_whistles_l1193_119394


namespace NUMINAMATH_CALUDE_complex_number_computations_l1193_119323

theorem complex_number_computations :
  let z₁ : ℂ := 1 + 2*I
  let z₂ : ℂ := (1 + I) / (1 - I)
  let z₃ : ℂ := (Real.sqrt 2 + Real.sqrt 3 * I) / (Real.sqrt 3 - Real.sqrt 2 * I)
  (z₁^2 = -3 + 4*I) ∧
  (z₂^6 + z₃ = -1 + Real.sqrt 6 / 5 + ((Real.sqrt 3 + Real.sqrt 2) / 5) * I) := by
sorry

end NUMINAMATH_CALUDE_complex_number_computations_l1193_119323


namespace NUMINAMATH_CALUDE_expression_evaluation_l1193_119367

theorem expression_evaluation : (255^2 - 231^2 - (231^2 - 207^2)) / 24 = 48 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1193_119367


namespace NUMINAMATH_CALUDE_house_rooms_l1193_119380

theorem house_rooms (outlets_per_room : ℕ) (total_outlets : ℕ) (h1 : outlets_per_room = 6) (h2 : total_outlets = 42) :
  total_outlets / outlets_per_room = 7 := by
  sorry

end NUMINAMATH_CALUDE_house_rooms_l1193_119380


namespace NUMINAMATH_CALUDE_function_inequality_l1193_119362

def f (x : ℝ) := x^2 - 2*x

theorem function_inequality (a : ℝ) : 
  (∃ x ∈ Set.Icc 2 4, f x ≤ a^2 + 2*a) → a ∈ Set.Iic (-2) ∪ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1193_119362


namespace NUMINAMATH_CALUDE_sum_odd_integers_21_to_51_l1193_119346

/-- The sum of all odd integers from 21 through 51, inclusive, is 576. -/
theorem sum_odd_integers_21_to_51 : 
  (Finset.range 16).sum (fun i => 21 + 2 * i) = 576 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_21_to_51_l1193_119346


namespace NUMINAMATH_CALUDE_is_circle_center_l1193_119390

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y - 55 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (3, -1)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center : ∀ (x y : ℝ), 
  circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 65 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l1193_119390


namespace NUMINAMATH_CALUDE_penultimate_digit_of_quotient_l1193_119342

theorem penultimate_digit_of_quotient : ∃ k : ℕ, 
  (4^1994 + 7^1994) / 10 = k * 10 + 1 := by
  sorry

end NUMINAMATH_CALUDE_penultimate_digit_of_quotient_l1193_119342


namespace NUMINAMATH_CALUDE_four_sharp_40_l1193_119308

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem four_sharp_40 : sharp (sharp (sharp (sharp 40))) = 9.536 := by
  sorry

end NUMINAMATH_CALUDE_four_sharp_40_l1193_119308


namespace NUMINAMATH_CALUDE_mean_inequality_for_close_numbers_l1193_119302

theorem mean_inequality_for_close_numbers
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (hxy : x ≠ y)
  (hyz : y ≠ z)
  (hxz : x ≠ z)
  (hclose : ∃ (ε δ : ℝ), ε > 0 ∧ δ > 0 ∧ ε < 1 ∧ δ < 1 ∧ x = y + ε ∧ z = y - δ) :
  (x + y) / 2 > Real.sqrt (x * y) ∧ Real.sqrt (x * y) > 2 * y * z / (y + z) :=
sorry

end NUMINAMATH_CALUDE_mean_inequality_for_close_numbers_l1193_119302


namespace NUMINAMATH_CALUDE_sqrt_division_equality_l1193_119349

theorem sqrt_division_equality : Real.sqrt 2 / Real.sqrt 3 = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_equality_l1193_119349


namespace NUMINAMATH_CALUDE_reciprocal_minus_one_l1193_119373

theorem reciprocal_minus_one (x : ℝ) : (1 / x = -1) → |-x - 1| = 0 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_minus_one_l1193_119373


namespace NUMINAMATH_CALUDE_train_travel_theorem_l1193_119316

/-- Represents the distance traveled by a train -/
def train_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem train_travel_theorem (initial_distance initial_time final_time : ℝ) 
  (h1 : initial_distance = 300)
  (h2 : initial_time = 20)
  (h3 : final_time = 600) : 
  train_distance (initial_distance / initial_time) final_time = 9000 := by
  sorry

#check train_travel_theorem

end NUMINAMATH_CALUDE_train_travel_theorem_l1193_119316


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1193_119356

theorem average_speed_calculation (d₁ d₂ d₃ v₁ v₂ v₃ : ℝ) 
  (h₁ : d₁ = 30) (h₂ : d₂ = 50) (h₃ : d₃ = 40)
  (h₄ : v₁ = 30) (h₅ : v₂ = 50) (h₆ : v₃ = 60) : 
  (d₁ + d₂ + d₃) / ((d₁ / v₁) + (d₂ / v₂) + (d₃ / v₃)) = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1193_119356


namespace NUMINAMATH_CALUDE_sequence_increasing_l1193_119365

theorem sequence_increasing (n : ℕ+) : 
  let a : ℕ+ → ℚ := fun k => k / (k + 2)
  a n < a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_l1193_119365


namespace NUMINAMATH_CALUDE_passes_through_point1_passes_through_point2_unique_line_l1193_119335

/-- A line passing through two points (-1, 0) and (0, 2) -/
def line (x y : ℝ) : Prop := y = 2 * x + 2

/-- The line passes through the point (-1, 0) -/
theorem passes_through_point1 : line (-1) 0 := by sorry

/-- The line passes through the point (0, 2) -/
theorem passes_through_point2 : line 0 2 := by sorry

/-- The equation y = 2x + 2 represents the unique line passing through (-1, 0) and (0, 2) -/
theorem unique_line : ∀ (x y : ℝ), (y = 2 * x + 2) ↔ line x y := by sorry

end NUMINAMATH_CALUDE_passes_through_point1_passes_through_point2_unique_line_l1193_119335


namespace NUMINAMATH_CALUDE_difference_solution_equation_problems_l1193_119328

/-- Definition of a difference solution equation -/
def is_difference_solution_equation (a b : ℝ) : Prop :=
  ∃ x : ℝ, a * x = b ∧ x = b - a

theorem difference_solution_equation_problems :
  -- Part 1
  is_difference_solution_equation 2 4 ∧
  -- Part 2
  (∀ a b : ℝ, is_difference_solution_equation 4 (a * b + a) →
    3 * (a * b + a) = 16) ∧
  -- Part 3
  (∀ m n : ℝ, is_difference_solution_equation 4 (m * n + m) ∧
    is_difference_solution_equation (-2) (m * n + n) →
    3 * (m * n + m) - 9 * (m * n + n)^2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_difference_solution_equation_problems_l1193_119328


namespace NUMINAMATH_CALUDE_f_property_l1193_119372

def property_P (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 ≤ k ∧ k + 1 < n ∧
  2 * Nat.choose n k = Nat.choose n (k - 1) + Nat.choose n (k + 1)

theorem f_property :
  (property_P 7) ∧
  (∀ n : ℕ, n ≤ 2016 → property_P n → n ≤ 1934) ∧
  (property_P 1934) :=
sorry

end NUMINAMATH_CALUDE_f_property_l1193_119372


namespace NUMINAMATH_CALUDE_units_digit_of_1389_pow_1247_l1193_119378

theorem units_digit_of_1389_pow_1247 (n : ℕ) :
  n = 1389^1247 → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_1389_pow_1247_l1193_119378


namespace NUMINAMATH_CALUDE_book_selection_l1193_119329

theorem book_selection (n m k : ℕ) (hn : n = 8) (hm : m = 5) (hk : k = 1) :
  (Nat.choose (n - k) (m - k)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_l1193_119329


namespace NUMINAMATH_CALUDE_sum_of_y_coefficients_l1193_119358

-- Define the polynomials
def p (x y : ℝ) := 5*x + 3*y - 2
def q (x y : ℝ) := 2*x + 5*y + 7

-- Define the expanded product
def expanded_product (x y : ℝ) := p x y * q x y

-- Define a function to extract coefficients of terms with y
def y_coefficients (x y : ℝ) : List ℝ := 
  [31, 15, 11]  -- Coefficients of xy, y², and y respectively

-- Theorem statement
theorem sum_of_y_coefficients :
  (y_coefficients 0 0).sum = 57 :=
sorry

end NUMINAMATH_CALUDE_sum_of_y_coefficients_l1193_119358


namespace NUMINAMATH_CALUDE_solution_for_x_l1193_119370

theorem solution_for_x (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0)
  (eq1 : x + 1 / z = 15) (eq2 : z + 1 / x = 9 / 20) :
  x = (15 + 5 * Real.sqrt 11) / 2 ∨ x = (15 - 5 * Real.sqrt 11) / 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_for_x_l1193_119370


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1193_119351

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 9 ∧ (101054 - k) % 10 = 0 ∧ ∀ (m : ℕ), m < k → (101054 - m) % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1193_119351


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1193_119363

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.sin (10 * π / 180) - 
  Real.cos (10 * π / 180) * Real.sin (70 * π / 180) = 
  -(Real.sqrt 3 / 2) := by sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1193_119363


namespace NUMINAMATH_CALUDE_andrew_flooring_planks_l1193_119364

/-- The number of wooden planks Andrew bought for his flooring project -/
def total_planks : ℕ := 91

/-- The number of planks used in Andrew's bedroom -/
def bedroom_planks : ℕ := 8

/-- The number of planks used in the living room -/
def living_room_planks : ℕ := 20

/-- The number of planks used in the kitchen -/
def kitchen_planks : ℕ := 11

/-- The number of planks used in the dining room -/
def dining_room_planks : ℕ := 13

/-- The number of planks used in the guest bedroom -/
def guest_bedroom_planks : ℕ := bedroom_planks - 2

/-- The number of planks used in each hallway -/
def hallway_planks : ℕ := 4

/-- The number of planks used in the study -/
def study_planks : ℕ := guest_bedroom_planks + 3

/-- The number of planks ruined in each bedroom -/
def bedroom_ruined_planks : ℕ := 3

/-- The number of planks ruined in the living room -/
def living_room_ruined_planks : ℕ := 2

/-- The number of planks ruined in the study -/
def study_ruined_planks : ℕ := 1

/-- The number of leftover planks -/
def leftover_planks : ℕ := 7

/-- The number of hallways -/
def number_of_hallways : ℕ := 2

theorem andrew_flooring_planks :
  total_planks = 
    bedroom_planks + bedroom_ruined_planks +
    living_room_planks + living_room_ruined_planks +
    kitchen_planks +
    dining_room_planks +
    guest_bedroom_planks + bedroom_ruined_planks +
    (hallway_planks * number_of_hallways) +
    study_planks + study_ruined_planks +
    leftover_planks :=
by sorry

end NUMINAMATH_CALUDE_andrew_flooring_planks_l1193_119364


namespace NUMINAMATH_CALUDE_marble_remainder_l1193_119397

theorem marble_remainder (r p : ℕ) : 
  r % 8 = 5 → p % 8 = 6 → (r + p) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_remainder_l1193_119397


namespace NUMINAMATH_CALUDE_mountain_height_proof_l1193_119306

def mountain_height (h : ℝ) : Prop :=
  h > 7900 ∧ h < 8000

theorem mountain_height_proof (h : ℝ) 
  (peter_false : ¬(h ≥ 8000))
  (mary_false : ¬(h ≤ 7900))
  (john_false : ¬(h ≤ 7500)) :
  mountain_height h :=
sorry

end NUMINAMATH_CALUDE_mountain_height_proof_l1193_119306


namespace NUMINAMATH_CALUDE_spheres_in_base_of_165_pyramid_l1193_119320

/-- The number of spheres in a regular triangular pyramid with n levels -/
def pyramid_spheres (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The number of spheres in the base of a regular triangular pyramid with n levels -/
def base_spheres (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: In a regular triangular pyramid with exactly 165 identical spheres,
    the number of spheres in the base is 45 -/
theorem spheres_in_base_of_165_pyramid :
  ∃ n : ℕ, pyramid_spheres n = 165 ∧ base_spheres n = 45 :=
sorry

end NUMINAMATH_CALUDE_spheres_in_base_of_165_pyramid_l1193_119320


namespace NUMINAMATH_CALUDE_apprentice_work_time_l1193_119312

/-- Proves that given the master's and apprentice's production rates, 
    the apprentice needs 4 hours to match the master's 3-hour output. -/
theorem apprentice_work_time 
  (master_rate : ℕ) 
  (apprentice_rate : ℕ) 
  (master_time : ℕ) 
  (h1 : master_rate = 64)
  (h2 : apprentice_rate = 48)
  (h3 : master_time = 3) :
  (master_rate * master_time) / apprentice_rate = 4 := by
  sorry

#check apprentice_work_time

end NUMINAMATH_CALUDE_apprentice_work_time_l1193_119312


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l1193_119301

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^23 + (i^105 * i^17) = -i - 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l1193_119301


namespace NUMINAMATH_CALUDE_exists_function_satisfying_condition_l1193_119369

theorem exists_function_satisfying_condition : 
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 2*x) = |x + 1| := by
  sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_condition_l1193_119369


namespace NUMINAMATH_CALUDE_one_third_of_recipe_flour_l1193_119315

theorem one_third_of_recipe_flour (original_flour : ℚ) (reduced_flour : ℚ) : 
  original_flour = 17/3 → reduced_flour = original_flour / 3 → reduced_flour = 17/9 := by
  sorry

#check one_third_of_recipe_flour

end NUMINAMATH_CALUDE_one_third_of_recipe_flour_l1193_119315


namespace NUMINAMATH_CALUDE_exam_score_calculation_l1193_119324

theorem exam_score_calculation (total_questions : ℕ) (total_marks : ℤ) (correct_answers : ℕ) (marks_per_wrong : ℤ) :
  total_questions = 80 →
  total_marks = 130 →
  correct_answers = 42 →
  marks_per_wrong = -1 →
  ∃ (marks_per_correct : ℤ),
    marks_per_correct * correct_answers + marks_per_wrong * (total_questions - correct_answers) = total_marks ∧
    marks_per_correct = 4 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l1193_119324


namespace NUMINAMATH_CALUDE_derivative_at_negative_one_l1193_119384

/-- Given a function f(x) = ax^4 + bx^2 + c where f'(1) = 2, prove that f'(-1) = -2 -/
theorem derivative_at_negative_one (a b c : ℝ) :
  let f := fun x : ℝ => a * x^4 + b * x^2 + c
  (deriv f) 1 = 2 → (deriv f) (-1) = -2 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_negative_one_l1193_119384


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1193_119386

/-- Given a hyperbola with equation x²/144 - y²/81 = 1, its asymptotes are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ,
  x^2 / 144 - y^2 / 81 = 1 →
  ∃ m : ℝ, m > 0 ∧ (y = m * x ∨ y = -m * x) ∧ m = 3/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1193_119386


namespace NUMINAMATH_CALUDE_circle_equation_center_radius_l1193_119393

/-- Given a circle equation, prove its center and radius -/
theorem circle_equation_center_radius 
  (x y : ℝ) 
  (h : x^2 - 2*x + y^2 + 6*y = 6) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (1, -3) ∧ 
    radius = 4 ∧ 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_center_radius_l1193_119393


namespace NUMINAMATH_CALUDE_problem_solution_l1193_119304

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := a^x
def g (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + m
def h (m : ℝ) (x : ℝ) : ℝ := 2^(2*x) + m - 2*m*2^x

-- Define the minimum value function
def H (m : ℝ) : ℝ :=
  if m < 1 then 1 - m
  else if m ≤ 2 then m - m^2
  else 4 - 3*m

theorem problem_solution :
  ∀ (a m : ℝ),
  (a > 0 ∧ a ≠ 1 ∧ m > 0) →
  (∀ (x : ℝ), x ∈ Set.Icc (-1) 1 → f a x ≤ 5/2 ∧ f a x ≥ 0) →
  (f a 1 + f a (-1) = 5/2) →
  (a = 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 1 → h m x ≥ H m) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 1 → |1 - m*(2^x + m/2^x)| ≤ 1 → m ∈ Set.Icc 0 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1193_119304


namespace NUMINAMATH_CALUDE_oriented_knight_moves_l1193_119344

/-- An Oriented Knight's move on a chess board -/
inductive OrientedKnightMove
| right_up : OrientedKnightMove  -- Two squares right, one square up
| up_right : OrientedKnightMove  -- Two squares up, one square right

/-- A sequence of Oriented Knight moves -/
def MoveSequence := List OrientedKnightMove

/-- The size of the chess board -/
def boardSize : ℕ := 16

/-- Checks if a sequence of moves is valid (reaches the top-right corner) -/
def isValidSequence (moves : MoveSequence) : Prop :=
  let finalPosition := moves.foldl
    (fun pos move => match move with
      | OrientedKnightMove.right_up => (pos.1 + 2, pos.2 + 1)
      | OrientedKnightMove.up_right => (pos.1 + 1, pos.2 + 2))
    (0, 0)
  finalPosition = (boardSize - 1, boardSize - 1)

/-- The number of valid move sequences for an Oriented Knight -/
def validSequenceCount : ℕ := 252

theorem oriented_knight_moves :
  (validSequences : Finset MoveSequence).card = validSequenceCount :=
by
  sorry

end NUMINAMATH_CALUDE_oriented_knight_moves_l1193_119344


namespace NUMINAMATH_CALUDE_geometric_sequence_unique_solution_l1193_119382

/-- A geometric sequence is defined by its first term and common ratio. -/
structure GeometricSequence where
  first_term : ℚ
  common_ratio : ℚ

/-- Get the nth term of a geometric sequence. -/
def GeometricSequence.nth_term (seq : GeometricSequence) (n : ℕ) : ℚ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem geometric_sequence_unique_solution :
  ∃! (seq : GeometricSequence),
    seq.nth_term 2 = 37 + 1/3 ∧
    seq.nth_term 6 = 2 + 1/3 ∧
    seq.first_term = 74 + 2/3 ∧
    seq.common_ratio = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_unique_solution_l1193_119382


namespace NUMINAMATH_CALUDE_log_equation_solution_l1193_119319

theorem log_equation_solution (s : ℝ) (h : s > 0) :
  (4 * Real.log s / Real.log 3 = Real.log (4 * s^2) / Real.log 3) → s = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1193_119319


namespace NUMINAMATH_CALUDE_zero_point_range_and_solution_l1193_119353

def f (a x : ℝ) : ℝ := a * x^3 - 2 * a * x + 3 * a - 4

theorem zero_point_range_and_solution :
  (∃ (a : ℝ), ∃ (x : ℝ), x ∈ Set.Ioo (-1 : ℝ) 1 ∧ f a x = 0) ∧
  (∀ (a : ℝ), (∃ (x : ℝ), x ∈ Set.Ioo (-1 : ℝ) 1 ∧ f a x = 0) →
    a ∈ Set.Icc (12 * (27 - 4 * Real.sqrt 6) / 211) (12 * (27 + 4 * Real.sqrt 6) / 211)) ∧
  (f (32/17) (1/2) = 0) :=
sorry

end NUMINAMATH_CALUDE_zero_point_range_and_solution_l1193_119353


namespace NUMINAMATH_CALUDE_cubic_root_product_l1193_119389

theorem cubic_root_product (a b c : ℝ) : 
  (a^3 - 15*a^2 + 22*a - 8 = 0) ∧ 
  (b^3 - 15*b^2 + 22*b - 8 = 0) ∧ 
  (c^3 - 15*c^2 + 22*c - 8 = 0) → 
  (2+a)*(2+b)*(2+c) = 120 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l1193_119389


namespace NUMINAMATH_CALUDE_workers_in_second_group_l1193_119313

theorem workers_in_second_group 
  (wages_group1 : ℕ) 
  (workers_group1 : ℕ) 
  (days_group1 : ℕ) 
  (wages_group2 : ℕ) 
  (days_group2 : ℕ) 
  (h1 : wages_group1 = 9450) 
  (h2 : workers_group1 = 15) 
  (h3 : days_group1 = 6) 
  (h4 : wages_group2 = 9975) 
  (h5 : days_group2 = 5) : 
  (wages_group2 / (wages_group1 / (workers_group1 * days_group1) * days_group2)) = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_workers_in_second_group_l1193_119313


namespace NUMINAMATH_CALUDE_sum_divisors_400_has_one_prime_factor_l1193_119340

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the positive divisors of 400 has exactly one distinct prime factor -/
theorem sum_divisors_400_has_one_prime_factor :
  num_distinct_prime_factors (sum_of_divisors 400) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_divisors_400_has_one_prime_factor_l1193_119340


namespace NUMINAMATH_CALUDE_not_necessarily_even_increasing_on_reals_max_at_turning_point_sqrt_convexity_l1193_119330

-- 1. A function f: ℝ → ℝ that satisfies f(-2) = f(2) is not necessarily an even function
theorem not_necessarily_even (f : ℝ → ℝ) (h : f (-2) = f 2) :
  ¬ ∀ x, f (-x) = f x :=
sorry

-- 2. If f: ℝ → ℝ is monotonically increasing on (-∞, 0] and [0, +∞), then f is increasing on ℝ
theorem increasing_on_reals (f : ℝ → ℝ)
  (h1 : ∀ x y, x ≤ y → x ≤ 0 → y ≤ 0 → f x ≤ f y)
  (h2 : ∀ x y, x ≤ y → 0 ≤ x → 0 ≤ y → f x ≤ f y) :
  ∀ x y, x ≤ y → f x ≤ f y :=
sorry

-- 3. If f: [a, b] → ℝ (where a < c < b) is increasing on [a, c) and decreasing on [c, b],
--    then f(c) is the maximum value of f on [a, b]
theorem max_at_turning_point {a b c : ℝ} (h : a < c ∧ c < b) (f : ℝ → ℝ)
  (h1 : ∀ x y, a ≤ x → x < y → y < c → f x ≤ f y)
  (h2 : ∀ x y, c < x → x < y → y ≤ b → f y ≤ f x) :
  ∀ x, a ≤ x → x ≤ b → f x ≤ f c :=
sorry

-- 4. For f(x) = √x and any x₁, x₂ ∈ (0, +∞), (f(x₁) + f(x₂))/2 ≤ f((x₁ + x₂)/2)
theorem sqrt_convexity (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : 0 < x₂) :
  (Real.sqrt x₁ + Real.sqrt x₂) / 2 ≤ Real.sqrt ((x₁ + x₂) / 2) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_even_increasing_on_reals_max_at_turning_point_sqrt_convexity_l1193_119330


namespace NUMINAMATH_CALUDE_exists_triangle_altitudes_form_triangle_but_not_bisectors_l1193_119309

/-- A triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The altitude triangle formed by the altitudes of the original triangle. -/
def AltitudeTriangle (t : Triangle) : Triangle := sorry

/-- The angle bisectors of a triangle. -/
def AngleBisectors (t : Triangle) : Fin 3 → ℝ := sorry

/-- Predicate to check if three lengths can form a triangle. -/
def CanFormTriangle (l₁ l₂ l₃ : ℝ) : Prop :=
  l₁ + l₂ > l₃ ∧ l₂ + l₃ > l₁ ∧ l₃ + l₁ > l₂

theorem exists_triangle_altitudes_form_triangle_but_not_bisectors :
  ∃ t : Triangle,
    CanFormTriangle (AltitudeTriangle t).a (AltitudeTriangle t).b (AltitudeTriangle t).c ∧
    ¬CanFormTriangle (AngleBisectors (AltitudeTriangle t) 0)
                     (AngleBisectors (AltitudeTriangle t) 1)
                     (AngleBisectors (AltitudeTriangle t) 2) :=
sorry

end NUMINAMATH_CALUDE_exists_triangle_altitudes_form_triangle_but_not_bisectors_l1193_119309


namespace NUMINAMATH_CALUDE_smallest_n_for_factorization_factorization_exists_for_31_l1193_119396

theorem smallest_n_for_factorization : 
  ∀ n : ℤ, n < 31 → 
  ¬∃ A B : ℤ, ∀ x : ℝ, 5 * x^2 + n * x + 48 = (5 * x + A) * (x + B) :=
by sorry

theorem factorization_exists_for_31 : 
  ∃ A B : ℤ, ∀ x : ℝ, 5 * x^2 + 31 * x + 48 = (5 * x + A) * (x + B) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_factorization_factorization_exists_for_31_l1193_119396


namespace NUMINAMATH_CALUDE_hundredth_digit_is_one_l1193_119321

/-- The decimal representation of 7/33 has a repeating pattern of length 2 -/
def decimal_rep_period (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 
    (7 : ℚ) / 33 = (a * 10 + b : ℚ) / 100 + (7 : ℚ) / (33 * 100)

/-- The 100th digit after the decimal point in 7/33 -/
def hundredth_digit : ℕ :=
  sorry

theorem hundredth_digit_is_one :
  decimal_rep_period 2 → hundredth_digit = 1 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_digit_is_one_l1193_119321


namespace NUMINAMATH_CALUDE_median_and_altitude_lengths_l1193_119383

/-- Right triangle DEF with given side lengths and midpoint N -/
structure RightTriangleDEF where
  DE : ℝ
  DF : ℝ
  N : ℝ × ℝ
  is_right_angle : DE^2 + DF^2 = (DE + DF)^2 / 2
  side_lengths : DE = 6 ∧ DF = 8
  N_is_midpoint : N = ((DE + DF) / 2, DF / 2)

/-- Theorem about median and altitude lengths in the right triangle -/
theorem median_and_altitude_lengths (t : RightTriangleDEF) :
  let DN := Real.sqrt ((t.DE + t.N.1)^2 + t.N.2^2)
  let altitude := 2 * (t.DE * t.DF) / (t.DE + t.DF)
  DN = 5 ∧ altitude = 4.8 := by
  sorry


end NUMINAMATH_CALUDE_median_and_altitude_lengths_l1193_119383


namespace NUMINAMATH_CALUDE_max_homework_time_l1193_119325

def homework_time (biology_time : ℕ) : ℕ :=
  let history_time := 2 * biology_time
  let geography_time := 3 * history_time
  biology_time + history_time + geography_time

theorem max_homework_time :
  homework_time 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_max_homework_time_l1193_119325


namespace NUMINAMATH_CALUDE_polynomial_value_at_three_l1193_119399

theorem polynomial_value_at_three : 
  let x : ℝ := 3
  x^6 - 6*x^2 + 7*x = 696 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_three_l1193_119399


namespace NUMINAMATH_CALUDE_difference_of_squares_2023_2022_l1193_119374

theorem difference_of_squares_2023_2022 : 2023^2 - 2022^2 = 4045 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_2023_2022_l1193_119374


namespace NUMINAMATH_CALUDE_cupcake_distribution_l1193_119332

/-- Given the initial number of cupcakes, the number of eaten cupcakes, and the number of packages,
    calculate the number of cupcakes in each package. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Theorem stating that given 71 initial cupcakes, 43 eaten cupcakes, and 4 packages,
    the number of cupcakes in each package is 7. -/
theorem cupcake_distribution :
  cupcakes_per_package 71 43 4 = 7 := by
  sorry


end NUMINAMATH_CALUDE_cupcake_distribution_l1193_119332


namespace NUMINAMATH_CALUDE_math_club_pair_sequences_l1193_119379

/-- The number of students in the Math Club -/
def num_students : ℕ := 12

/-- The number of sessions per week -/
def sessions_per_week : ℕ := 3

/-- The number of students selected per session -/
def students_per_session : ℕ := 2

/-- The number of different pair sequences that can be selected in one week -/
def pair_sequences_per_week : ℕ := (num_students * (num_students - 1)) ^ sessions_per_week

theorem math_club_pair_sequences :
  pair_sequences_per_week = 2299968 :=
sorry

end NUMINAMATH_CALUDE_math_club_pair_sequences_l1193_119379


namespace NUMINAMATH_CALUDE_xiao_ying_performance_l1193_119357

def regular_weight : ℝ := 0.20
def midterm_weight : ℝ := 0.30
def final_weight : ℝ := 0.50
def regular_score : ℝ := 85
def midterm_score : ℝ := 90
def final_score : ℝ := 92

def semester_performance : ℝ :=
  regular_weight * regular_score +
  midterm_weight * midterm_score +
  final_weight * final_score

theorem xiao_ying_performance :
  semester_performance = 90 := by sorry

end NUMINAMATH_CALUDE_xiao_ying_performance_l1193_119357


namespace NUMINAMATH_CALUDE_gift_combinations_count_l1193_119331

/-- The number of different gift packaging combinations -/
def gift_combinations (wrapping_paper : ℕ) (ribbon : ℕ) (gift_card : ℕ) (gift_box : ℕ) : ℕ :=
  wrapping_paper * ribbon * gift_card * gift_box

/-- Theorem stating the number of gift packaging combinations -/
theorem gift_combinations_count :
  gift_combinations 10 3 4 5 = 600 := by
  sorry

end NUMINAMATH_CALUDE_gift_combinations_count_l1193_119331


namespace NUMINAMATH_CALUDE_chucks_team_lead_l1193_119343

/-- Represents a team in the basketball match -/
inductive Team
| ChucksTeam
| YellowTeam

/-- Represents a quarter in the basketball match -/
inductive Quarter
| First
| Second
| Third
| Fourth

/-- Calculates the score for a given team in a given quarter -/
def quarterScore (team : Team) (quarter : Quarter) : ℤ :=
  match team, quarter with
  | Team.ChucksTeam, Quarter.First => 23
  | Team.ChucksTeam, Quarter.Second => 18
  | Team.ChucksTeam, Quarter.Third => 19
  | Team.ChucksTeam, Quarter.Fourth => 17
  | Team.YellowTeam, Quarter.First => 24
  | Team.YellowTeam, Quarter.Second => 19
  | Team.YellowTeam, Quarter.Third => 14
  | Team.YellowTeam, Quarter.Fourth => 16

/-- Points gained from technical fouls -/
def technicalFoulPoints (team : Team) : ℤ :=
  match team with
  | Team.ChucksTeam => 3
  | Team.YellowTeam => 2

/-- Calculates the total score for a team -/
def totalScore (team : Team) : ℤ :=
  quarterScore team Quarter.First +
  quarterScore team Quarter.Second +
  quarterScore team Quarter.Third +
  quarterScore team Quarter.Fourth +
  technicalFoulPoints team

/-- The main theorem stating Chuck's Team's lead -/
theorem chucks_team_lead :
  totalScore Team.ChucksTeam - totalScore Team.YellowTeam = 5 := by
  sorry


end NUMINAMATH_CALUDE_chucks_team_lead_l1193_119343


namespace NUMINAMATH_CALUDE_journey_speeds_correct_l1193_119377

/-- Represents the speeds and meeting times of pedestrians and cyclists --/
structure JourneyData where
  distance : ℝ
  pedestrian_start : ℝ
  cyclist1_start : ℝ
  cyclist2_start : ℝ
  pedestrian_speed : ℝ
  cyclist_speed : ℝ

/-- Checks if the given speeds satisfy the journey conditions --/
def satisfies_conditions (data : JourneyData) : Prop :=
  let first_meeting_time := data.cyclist1_start + (data.distance / 2 - data.pedestrian_speed * (data.cyclist1_start - data.pedestrian_start)) / (data.cyclist_speed - data.pedestrian_speed)
  let second_meeting_time := first_meeting_time + 1
  let pedestrian_distance_at_second_meeting := data.pedestrian_speed * (second_meeting_time - data.pedestrian_start)
  let cyclist2_distance := data.cyclist_speed * (second_meeting_time - data.cyclist2_start)
  first_meeting_time - data.pedestrian_start > 0 ∧
  first_meeting_time - data.cyclist1_start > 0 ∧
  second_meeting_time - data.cyclist2_start > 0 ∧
  pedestrian_distance_at_second_meeting + cyclist2_distance = data.distance

/-- The main theorem stating that the given speeds satisfy the journey conditions --/
theorem journey_speeds_correct : ∃ (data : JourneyData),
  data.distance = 40 ∧
  data.pedestrian_start = 0 ∧
  data.cyclist1_start = 10/3 ∧
  data.cyclist2_start = 4.5 ∧
  data.pedestrian_speed = 5 ∧
  data.cyclist_speed = 30 ∧
  satisfies_conditions data := by
  sorry


end NUMINAMATH_CALUDE_journey_speeds_correct_l1193_119377


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l1193_119350

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The problem statement -/
theorem pirate_loot_sum : 
  let silverware := base5ToBase10 [4, 1, 2, 3]
  let gemstones := base5ToBase10 [2, 2, 0, 3]
  let fine_silk := base5ToBase10 [2, 0, 2]
  silverware + gemstones + fine_silk = 873 := by
  sorry


end NUMINAMATH_CALUDE_pirate_loot_sum_l1193_119350


namespace NUMINAMATH_CALUDE_no_equilateral_triangle_2D_exists_regular_tetrahedron_3D_l1193_119376

-- Define a 2D point with integer coordinates
structure Point2D where
  x : ℤ
  y : ℤ

-- Define a 3D point with integer coordinates
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

-- Function to calculate the square of the distance between two 2D points
def distanceSquared2D (p1 p2 : Point2D) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to calculate the square of the distance between two 3D points
def distanceSquared3D (p1 p2 : Point3D) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

-- Theorem: No equilateral triangle exists with vertices at integer coordinate points in 2D
theorem no_equilateral_triangle_2D :
  ¬∃ (a b c : Point2D), 
    distanceSquared2D a b = distanceSquared2D b c ∧
    distanceSquared2D b c = distanceSquared2D c a ∧
    distanceSquared2D c a = distanceSquared2D a b :=
sorry

-- Theorem: A regular tetrahedron exists with vertices at integer coordinate points in 3D
theorem exists_regular_tetrahedron_3D :
  ∃ (a b c d : Point3D),
    distanceSquared3D a b = distanceSquared3D b c ∧
    distanceSquared3D b c = distanceSquared3D c d ∧
    distanceSquared3D c d = distanceSquared3D d a ∧
    distanceSquared3D d a = distanceSquared3D a b ∧
    distanceSquared3D a c = distanceSquared3D b d :=
sorry

end NUMINAMATH_CALUDE_no_equilateral_triangle_2D_exists_regular_tetrahedron_3D_l1193_119376


namespace NUMINAMATH_CALUDE_product_inequality_l1193_119337

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1193_119337


namespace NUMINAMATH_CALUDE_foal_count_l1193_119352

def animal_count : ℕ := 11
def leg_count : ℕ := 30
def turkey_legs : ℕ := 2
def foal_legs : ℕ := 4

theorem foal_count (t f : ℕ) : 
  t + f = animal_count → 
  turkey_legs * t + foal_legs * f = leg_count → 
  f = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_foal_count_l1193_119352


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1193_119317

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 6 * I
  let z₂ : ℂ := 4 - 6 * I
  (z₁ / z₂) + (z₂ / z₁) = (-10 : ℚ) / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1193_119317


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1193_119333

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- The area of the triangle -/
  area : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- Assumption that the hypotenuse is positive -/
  hypotenuse_pos : hypotenuse > 0
  /-- Assumption that the area is positive -/
  area_pos : area > 0
  /-- Assumption that the radius is positive -/
  radius_pos : radius > 0

/-- Theorem stating that for a right-angled triangle with hypotenuse 9 and area 36,
    the radius of the inscribed circle is 3 -/
theorem inscribed_circle_radius
  (triangle : RightTriangleWithInscribedCircle)
  (h1 : triangle.hypotenuse = 9)
  (h2 : triangle.area = 36) :
  triangle.radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1193_119333


namespace NUMINAMATH_CALUDE_investment_principal_calculation_l1193_119326

/-- Proves that given a monthly interest payment of $234 and a simple annual interest rate of 9%,
    the principal amount of the investment is $31,200. -/
theorem investment_principal_calculation (monthly_interest : ℝ) (annual_rate : ℝ) :
  monthly_interest = 234 →
  annual_rate = 0.09 →
  (monthly_interest * 12) / annual_rate = 31200 := by
  sorry

end NUMINAMATH_CALUDE_investment_principal_calculation_l1193_119326


namespace NUMINAMATH_CALUDE_power_equation_solution_l1193_119303

theorem power_equation_solution (n : ℕ) : 2^n = 2 * 4^2 * 16^3 → n = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1193_119303


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l1193_119368

/-- The function f(x) = x^3 - x + a --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x + a

/-- The condition a^2 - a = 0 --/
def condition (a : ℝ) : Prop := a^2 - a = 0

/-- f is an increasing function --/
def is_increasing (a : ℝ) : Prop := ∀ x y, x < y → f a x < f a y

theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ a, condition a → is_increasing a) ∧
  ¬(∀ a, is_increasing a → condition a) :=
sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l1193_119368


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1193_119322

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m : ℝ) : Prop :=
  (m - 1) / 1 = 2 / (-(m + 2))

theorem sufficient_not_necessary_condition :
  (are_parallel (-1)) ∧ (∃ m : ℝ, m ≠ -1 ∧ are_parallel m) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1193_119322


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1193_119391

theorem smallest_positive_integer_with_remainders : ∃ (b : ℕ), b > 0 ∧
  b % 3 = 2 ∧ b % 5 = 3 ∧ 
  ∀ (x : ℕ), x > 0 ∧ x % 3 = 2 ∧ x % 5 = 3 → b ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1193_119391


namespace NUMINAMATH_CALUDE_rectangular_prism_space_diagonal_l1193_119314

/-- A rectangular prism with given surface area and edge length sum has a space diagonal of length 5 -/
theorem rectangular_prism_space_diagonal : 
  ∀ (x y z : ℝ), 
  (2 * x * y + 2 * y * z + 2 * x * z = 11) →
  (4 * (x + y + z) = 24) →
  Real.sqrt (x^2 + y^2 + z^2) = 5 := by
sorry


end NUMINAMATH_CALUDE_rectangular_prism_space_diagonal_l1193_119314


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1193_119361

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 3 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 + m*y + 3 = 0 ∧ y = 3 ∧ m = -4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1193_119361


namespace NUMINAMATH_CALUDE_largest_angle_ABC_l1193_119334

theorem largest_angle_ABC (AC BC : ℝ) (angle_BAC : ℝ) : 
  AC = 5 * Real.sqrt 2 →
  BC = 5 →
  angle_BAC = 30 * π / 180 →
  ∃ (angle_ABC : ℝ), 
    angle_ABC ≤ 135 * π / 180 ∧
    ∀ (other_angle_ABC : ℝ), 
      (AC / Real.sin angle_BAC = BC / Real.sin other_angle_ABC) →
      other_angle_ABC ≤ angle_ABC := by
sorry

end NUMINAMATH_CALUDE_largest_angle_ABC_l1193_119334


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1193_119375

/-- Given a circle D with equation x^2 + 14y + 63 = -y^2 - 12x, 
    where (a, b) is the center and r is the radius, 
    prove that a + b + r = -13 + √22 -/
theorem circle_center_radius_sum (x y a b r : ℝ) : 
  (∀ x y, x^2 + 14*y + 63 = -y^2 - 12*x) →
  ((x - a)^2 + (y - b)^2 = r^2) →
  a + b + r = -13 + Real.sqrt 22 := by
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1193_119375
