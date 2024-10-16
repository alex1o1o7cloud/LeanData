import Mathlib

namespace NUMINAMATH_CALUDE_cross_product_perpendicular_l695_69536

/-- The cross product of two 3D vectors -/
def cross_product (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => match i with
    | 0 => v 1 * w 2 - v 2 * w 1
    | 1 => v 2 * w 0 - v 0 * w 2
    | 2 => v 0 * w 1 - v 1 * w 0

/-- The dot product of two 3D vectors -/
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

theorem cross_product_perpendicular (v w : Fin 3 → ℝ) :
  let v1 : Fin 3 → ℝ := fun i => match i with
    | 0 => 3
    | 1 => -2
    | 2 => 4
  let v2 : Fin 3 → ℝ := fun i => match i with
    | 0 => 1
    | 1 => 5
    | 2 => -3
  let cp := cross_product v1 v2
  cp 0 = -14 ∧ cp 1 = 13 ∧ cp 2 = 17 ∧
  dot_product cp v1 = 0 ∧ dot_product cp v2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_cross_product_perpendicular_l695_69536


namespace NUMINAMATH_CALUDE_finleys_age_l695_69597

/-- Given the ages of Jill, Roger, and Finley, prove that Finley is 55 years old. -/
theorem finleys_age (jill roger finley : ℕ) : 
  jill = 20 ∧ 
  roger = 2 * jill + 5 ∧ 
  (roger + 15) - (jill + 15) = finley - 30 → 
  finley = 55 := by
  sorry

end NUMINAMATH_CALUDE_finleys_age_l695_69597


namespace NUMINAMATH_CALUDE_function_value_at_five_l695_69539

/-- Given a function g : ℝ → ℝ satisfying g(x) + 3g(1-x) = 4x^2 - 1 for all x,
    prove that g(5) = 11.25 -/
theorem function_value_at_five (g : ℝ → ℝ) 
    (h : ∀ x, g x + 3 * g (1 - x) = 4 * x^2 - 1) : 
    g 5 = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_five_l695_69539


namespace NUMINAMATH_CALUDE_pole_length_reduction_l695_69573

theorem pole_length_reduction (original_length : ℝ) (reduction_percentage : ℝ) (new_length : ℝ) :
  original_length = 20 →
  reduction_percentage = 30 →
  new_length = original_length * (1 - reduction_percentage / 100) →
  new_length = 14 :=
by sorry

end NUMINAMATH_CALUDE_pole_length_reduction_l695_69573


namespace NUMINAMATH_CALUDE_largest_invalid_sum_l695_69576

def is_valid_sum (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b % 6 ≠ 0 ∧ n = 42 * a + b

theorem largest_invalid_sum : 
  (∀ m : ℕ, m > 252 → is_valid_sum m) ∧ ¬ is_valid_sum 252 :=
sorry

end NUMINAMATH_CALUDE_largest_invalid_sum_l695_69576


namespace NUMINAMATH_CALUDE_driptown_rainfall_2011_l695_69543

/-- The total rainfall in Driptown in 2011 -/
def total_rainfall_2011 (avg_2010 avg_increase : ℝ) : ℝ :=
  12 * (avg_2010 + avg_increase)

/-- Theorem: The total rainfall in Driptown in 2011 was 468 mm -/
theorem driptown_rainfall_2011 :
  total_rainfall_2011 37.2 1.8 = 468 := by
  sorry

end NUMINAMATH_CALUDE_driptown_rainfall_2011_l695_69543


namespace NUMINAMATH_CALUDE_expression_positivity_l695_69553

theorem expression_positivity (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_positivity_l695_69553


namespace NUMINAMATH_CALUDE_increasing_digits_mod_1000_l695_69510

/-- The number of ways to distribute n identical objects into k distinct boxes -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of 8-digit positive integers with digits in increasing order (1-8, repetition allowed) -/
def M : ℕ := starsAndBars 8 8

theorem increasing_digits_mod_1000 : M % 1000 = 435 := by sorry

end NUMINAMATH_CALUDE_increasing_digits_mod_1000_l695_69510


namespace NUMINAMATH_CALUDE_smallest_coinciding_triangle_l695_69545

/-- Represents the type of isosceles triangle -/
inductive TriangleType
  | Acute
  | Right

/-- Returns the vertex angle of a triangle based on its type -/
def vertexAngle (t : TriangleType) : ℕ :=
  match t with
  | TriangleType.Acute => 30
  | TriangleType.Right => 90

/-- Returns the type of the n-th triangle in the sequence -/
def nthTriangleType (n : ℕ) : TriangleType :=
  if n % 3 = 0 then TriangleType.Right else TriangleType.Acute

/-- Calculates the sum of vertex angles for the first n triangles -/
def sumOfAngles (n : ℕ) : ℕ :=
  List.range n |> List.map (fun i => vertexAngle (nthTriangleType (i + 1))) |> List.sum

/-- The main theorem to prove -/
theorem smallest_coinciding_triangle : 
  (∀ k < 23, sumOfAngles k % 360 ≠ 0) ∧ sumOfAngles 23 % 360 = 0 := by
  sorry


end NUMINAMATH_CALUDE_smallest_coinciding_triangle_l695_69545


namespace NUMINAMATH_CALUDE_division_problem_l695_69524

theorem division_problem (a b c : ℝ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 5/2) : 
  c / a = 2/15 := by sorry

end NUMINAMATH_CALUDE_division_problem_l695_69524


namespace NUMINAMATH_CALUDE_empty_seats_arrangements_l695_69540

/-- The number of chairs in a row -/
def num_chairs : ℕ := 8

/-- The number of students taking seats -/
def num_students : ℕ := 4

/-- The number of empty seats -/
def num_empty_seats : ℕ := num_chairs - num_students

/-- Calculates the number of ways to arrange all empty seats adjacent to each other -/
def adjacent_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial (n - k + 1)

/-- Calculates the number of ways to arrange all empty seats not adjacent to each other -/
def non_adjacent_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial k * Nat.choose (n - k + 1) k

theorem empty_seats_arrangements :
  adjacent_arrangements num_chairs num_empty_seats = 120 ∧
  non_adjacent_arrangements num_chairs num_empty_seats = 120 := by
  sorry

end NUMINAMATH_CALUDE_empty_seats_arrangements_l695_69540


namespace NUMINAMATH_CALUDE_brian_shoe_count_l695_69531

theorem brian_shoe_count :
  ∀ (b e j : ℕ),
    j = e / 2 →
    e = 3 * b →
    b + e + j = 121 →
    b = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_brian_shoe_count_l695_69531


namespace NUMINAMATH_CALUDE_motorcyclist_cyclist_problem_l695_69571

/-- The distance between two points A and B, given the conditions of the problem -/
def distance_AB : ℝ := 20

theorem motorcyclist_cyclist_problem (x : ℝ) 
  (h1 : x > 0) -- Ensure distance is positive
  (h2 : x - 4 > 0) -- Ensure meeting point is between A and B
  (h3 : (x - 4) / 4 = x / (x - 15)) -- Ratio of speeds equation
  : x = distance_AB := by
  sorry

end NUMINAMATH_CALUDE_motorcyclist_cyclist_problem_l695_69571


namespace NUMINAMATH_CALUDE_black_to_white_area_ratio_l695_69527

/-- The ratio of black to white area in concentric circles with radii 2, 4, 6, and 8 -/
theorem black_to_white_area_ratio : Real := by
  -- Define the radii of the circles
  let r1 : Real := 2
  let r2 : Real := 4
  let r3 : Real := 6
  let r4 : Real := 8

  -- Define the areas of the circles
  let A1 : Real := Real.pi * r1^2
  let A2 : Real := Real.pi * r2^2
  let A3 : Real := Real.pi * r3^2
  let A4 : Real := Real.pi * r4^2

  -- Define the areas of the black and white regions
  let black_area : Real := A1 + (A3 - A2)
  let white_area : Real := (A2 - A1) + (A4 - A3)

  -- Prove that the ratio of black area to white area is 3/5
  have h : black_area / white_area = 3 / 5 := by sorry

  exact 3 / 5

end NUMINAMATH_CALUDE_black_to_white_area_ratio_l695_69527


namespace NUMINAMATH_CALUDE_division_problem_l695_69514

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 1375 → 
  divisor = 66 → 
  remainder = 55 → 
  dividend = divisor * quotient + remainder → 
  quotient = 20 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l695_69514


namespace NUMINAMATH_CALUDE_pascal_triangle_row17_element5_l695_69509

theorem pascal_triangle_row17_element5 : Nat.choose 17 4 = 2380 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row17_element5_l695_69509


namespace NUMINAMATH_CALUDE_poly_descending_order_l695_69591

/-- The original polynomial -/
def original_poly (x y : ℝ) : ℝ := 2 * x^2 * y - 3 * x^3 - x * y^3 + 1

/-- The polynomial arranged in descending order of x -/
def descending_poly (x y : ℝ) : ℝ := -3 * x^3 + 2 * x^2 * y - x * y^3 + 1

/-- Theorem stating that the original polynomial is equal to the descending order polynomial -/
theorem poly_descending_order : ∀ x y : ℝ, original_poly x y = descending_poly x y := by
  sorry

end NUMINAMATH_CALUDE_poly_descending_order_l695_69591


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l695_69595

theorem regular_polygon_properties :
  ∀ n : ℕ,
  (n ≥ 3) →
  (n - 2) * 180 = 3 * 360 + 180 →
  n = 9 ∧ ((n - 2) * 180 / n : ℚ) = 140 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l695_69595


namespace NUMINAMATH_CALUDE_monday_walking_speed_l695_69583

/-- Represents Jonathan's exercise routine for a week -/
structure ExerciseRoutine where
  monday_speed : ℝ
  wednesday_speed : ℝ
  friday_speed : ℝ
  distance_per_day : ℝ
  total_time : ℝ

/-- Theorem stating that Jonathan's Monday walking speed is 2 miles per hour -/
theorem monday_walking_speed (routine : ExerciseRoutine) 
  (h1 : routine.wednesday_speed = 3)
  (h2 : routine.friday_speed = 6)
  (h3 : routine.distance_per_day = 6)
  (h4 : routine.total_time = 6)
  (h5 : routine.distance_per_day / routine.monday_speed + 
        routine.distance_per_day / routine.wednesday_speed + 
        routine.distance_per_day / routine.friday_speed = routine.total_time) :
  routine.monday_speed = 2 := by
  sorry

#check monday_walking_speed

end NUMINAMATH_CALUDE_monday_walking_speed_l695_69583


namespace NUMINAMATH_CALUDE_wire_cutting_l695_69511

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) :
  total_length = 70 →
  ratio = 2 / 3 →
  shorter_piece + (shorter_piece + ratio * shorter_piece) = total_length →
  shorter_piece = 26.25 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l695_69511


namespace NUMINAMATH_CALUDE_cubic_inequality_for_negative_numbers_l695_69529

theorem cubic_inequality_for_negative_numbers (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(a^3 > b^3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_for_negative_numbers_l695_69529


namespace NUMINAMATH_CALUDE_students_spend_two_dollars_l695_69551

/-- The price of one pencil in cents -/
def pencil_price : ℕ := 20

/-- The number of pencils Tolu wants -/
def tolu_pencils : ℕ := 3

/-- The number of pencils Robert wants -/
def robert_pencils : ℕ := 5

/-- The number of pencils Melissa wants -/
def melissa_pencils : ℕ := 2

/-- The total amount spent by the students in dollars -/
def total_spent : ℚ := (pencil_price * (tolu_pencils + robert_pencils + melissa_pencils)) / 100

theorem students_spend_two_dollars : total_spent = 2 := by
  sorry

end NUMINAMATH_CALUDE_students_spend_two_dollars_l695_69551


namespace NUMINAMATH_CALUDE_divisors_sum_8_implies_one_zero_l695_69566

def has_three_smallest_distinct_divisors_sum_8 (A : ℕ+) : Prop :=
  ∃ d₁ d₂ d₃ : ℕ+, 
    d₁ < d₂ ∧ d₂ < d₃ ∧
    d₁ ∣ A ∧ d₂ ∣ A ∧ d₃ ∣ A ∧
    d₁.val + d₂.val + d₃.val = 8 ∧
    ∀ d : ℕ+, d ∣ A → d ≤ d₃ → d = d₁ ∨ d = d₂ ∨ d = d₃

def ends_with_one_zero (A : ℕ+) : Prop :=
  ∃ k : ℕ, A.val = 10 * k ∧ k % 10 ≠ 0

theorem divisors_sum_8_implies_one_zero (A : ℕ+) :
  has_three_smallest_distinct_divisors_sum_8 A → ends_with_one_zero A :=
sorry

end NUMINAMATH_CALUDE_divisors_sum_8_implies_one_zero_l695_69566


namespace NUMINAMATH_CALUDE_functional_equation_solution_l695_69584

/-- Given a function g : ℝ → ℝ satisfying the functional equation
    2g(x) - 3g(1/x) = x^2 for all x ≠ 0, prove that g(2) = 8.25 -/
theorem functional_equation_solution (g : ℝ → ℝ) 
    (h : ∀ x : ℝ, x ≠ 0 → 2 * g x - 3 * g (1/x) = x^2) : 
  g 2 = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l695_69584


namespace NUMINAMATH_CALUDE_missing_score_proof_l695_69500

theorem missing_score_proof (known_scores : List ℝ) (mean : ℝ) : 
  known_scores = [81, 73, 86, 73] →
  mean = 79.2 →
  ∃ (missing_score : ℝ), 
    (List.sum known_scores + missing_score) / 5 = mean ∧
    missing_score = 83 := by
  sorry

end NUMINAMATH_CALUDE_missing_score_proof_l695_69500


namespace NUMINAMATH_CALUDE_expression_simplification_l695_69555

theorem expression_simplification (a : ℤ) (h : a = 2020) : 
  (a^4 - 3*a^3*(a+1) + 4*a^2*(a+1)^2 - (a+1)^4 + 1) / (a*(a+1)) = a^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l695_69555


namespace NUMINAMATH_CALUDE_score_ordering_l695_69570

-- Define the set of people
inductive Person : Type
| M : Person  -- Marty
| Q : Person  -- Quay
| S : Person  -- Shana
| Z : Person  -- Zane
| K : Person  -- Kaleana

-- Define a function to represent the score of each person
variable (score : Person → ℕ)

-- Define the conditions from the problem
def marty_condition (score : Person → ℕ) : Prop :=
  ∃ p : Person, score Person.M > score p

def quay_condition (score : Person → ℕ) : Prop :=
  score Person.Q = score Person.K

def shana_condition (score : Person → ℕ) : Prop :=
  ∃ p : Person, score Person.S < score p

def zane_condition (score : Person → ℕ) : Prop :=
  (score Person.Z < score Person.S) ∨ (score Person.Z > score Person.M)

-- Theorem statement
theorem score_ordering (score : Person → ℕ) :
  marty_condition score →
  quay_condition score →
  shana_condition score →
  zane_condition score →
  (score Person.Z < score Person.S) ∧
  (score Person.S < score Person.Q) ∧
  (score Person.Q < score Person.M) :=
sorry

end NUMINAMATH_CALUDE_score_ordering_l695_69570


namespace NUMINAMATH_CALUDE_bottom_price_is_3350_l695_69561

/-- The price of a bottom pajama in won -/
def bottom_price : ℕ := sorry

/-- The price of a top pajama in won -/
def top_price : ℕ := sorry

/-- The number of pajama sets bought -/
def num_sets : ℕ := 3

/-- The total amount paid in won -/
def total_paid : ℕ := 21000

/-- The price difference between top and bottom in won -/
def price_difference : ℕ := 300

theorem bottom_price_is_3350 : 
  bottom_price = 3350 ∧ 
  top_price = bottom_price + price_difference ∧
  num_sets * (bottom_price + top_price) = total_paid := by
  sorry

end NUMINAMATH_CALUDE_bottom_price_is_3350_l695_69561


namespace NUMINAMATH_CALUDE_second_bush_pink_roses_l695_69557

def rose_problem (red : ℕ) (yellow : ℕ) (orange : ℕ) (total_picked : ℕ) : ℕ :=
  let red_picked := red / 2
  let yellow_picked := yellow / 4
  let orange_picked := orange / 4
  let pink_picked := total_picked - red_picked - yellow_picked - orange_picked
  2 * pink_picked

theorem second_bush_pink_roses :
  rose_problem 12 20 8 22 = 18 := by
  sorry

end NUMINAMATH_CALUDE_second_bush_pink_roses_l695_69557


namespace NUMINAMATH_CALUDE_sum_g_one_neg_one_l695_69562

/-- Given two functions f and g defined on real numbers satisfying certain conditions,
    prove that g(1) + g(-1) = -1. -/
theorem sum_g_one_neg_one (f g : ℝ → ℝ) 
    (h1 : ∀ x y, f (x - y) = f x * g y - g x * f y)
    (h2 : f (-2) = f 1)
    (h3 : f 1 ≠ 0) : 
  g 1 + g (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_sum_g_one_neg_one_l695_69562


namespace NUMINAMATH_CALUDE_mosquitoes_to_cause_death_l695_69585

/-- Represents the number of drops of blood sucked by a mosquito of a given species -/
def drops_per_species : Fin 3 → ℕ
  | 0 => 20  -- Species A
  | 1 => 25  -- Species B
  | 2 => 30  -- Species C
  | _ => 0   -- This case is unreachable, but needed for completeness

/-- The number of drops of blood per liter -/
def drops_per_liter : ℕ := 5000

/-- The number of liters of blood loss that causes death -/
def lethal_blood_loss : ℕ := 3

/-- The total number of drops of blood that cause death -/
def lethal_drops : ℕ := lethal_blood_loss * drops_per_liter

/-- The theorem stating the number of mosquitoes of each species required to cause death -/
theorem mosquitoes_to_cause_death :
  ∃ n : ℕ, n > 0 ∧ 
  (n * drops_per_species 0 + n * drops_per_species 1 + n * drops_per_species 2 = lethal_drops) ∧
  n = 200 := by
  sorry

end NUMINAMATH_CALUDE_mosquitoes_to_cause_death_l695_69585


namespace NUMINAMATH_CALUDE_equal_cell_squares_count_l695_69518

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the 5x5 grid configuration -/
def Grid : Type := Fin 5 → Fin 5 → Cell

/-- The specific grid configuration given in the problem -/
def problem_grid : Grid := sorry

/-- A square in the grid -/
structure Square where
  top_left : Fin 5 × Fin 5
  size : Nat

/-- Checks if a square has equal number of black and white cells -/
def has_equal_cells (g : Grid) (s : Square) : Bool := sorry

/-- Counts the number of squares with equal black and white cells -/
def count_equal_squares (g : Grid) : Nat := sorry

/-- The main theorem -/
theorem equal_cell_squares_count :
  count_equal_squares problem_grid = 16 := by sorry

end NUMINAMATH_CALUDE_equal_cell_squares_count_l695_69518


namespace NUMINAMATH_CALUDE_parallel_lines_m_l695_69598

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  (2 : ℝ) / (m + 1) = m / 3

/-- The problem statement -/
theorem parallel_lines_m (m : ℝ) : parallel_lines m → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_l695_69598


namespace NUMINAMATH_CALUDE_square_root_division_l695_69507

theorem square_root_division (x : ℝ) : (Real.sqrt 3600 / x = 4) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_root_division_l695_69507


namespace NUMINAMATH_CALUDE_sin_cos_product_given_tan_l695_69513

theorem sin_cos_product_given_tan (θ : Real) (h : Real.tan θ = 2) :
  Real.sin θ * Real.cos θ = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_given_tan_l695_69513


namespace NUMINAMATH_CALUDE_vector_magnitude_l695_69534

theorem vector_magnitude (a b : ℝ × ℝ) :
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  a - b = (Real.sqrt 3, Real.sqrt 2) →
  ‖a + 2 • b‖ = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l695_69534


namespace NUMINAMATH_CALUDE_shortest_path_general_drinking_horse_l695_69578

-- Define the points and the line
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (4, 4)
def l (x y : ℝ) : Prop := x - y + 1 = 0

-- State the theorem
theorem shortest_path_general_drinking_horse :
  ∃ (P : ℝ × ℝ), l P.1 P.2 ∧
    Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) +
    Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) =
    2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_shortest_path_general_drinking_horse_l695_69578


namespace NUMINAMATH_CALUDE_perpendicular_planes_counterexample_l695_69532

/-- A type representing a plane in 3D space -/
structure Plane :=
  (normal : ℝ × ℝ × ℝ)
  (point : ℝ × ℝ × ℝ)

/-- Perpendicularity relation between planes -/
def perpendicular (p q : Plane) : Prop :=
  ∃ (k : ℝ), p.normal = k • q.normal

theorem perpendicular_planes_counterexample :
  ∃ (α β γ : Plane),
    α ≠ β ∧ β ≠ γ ∧ α ≠ γ ∧
    perpendicular α β ∧
    perpendicular β γ ∧
    ¬ perpendicular α γ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_counterexample_l695_69532


namespace NUMINAMATH_CALUDE_geometric_sequence_term_count_l695_69546

/-- Given a geometric sequence {a_n} with a_1 = 1, q = 1/2, and a_n = 1/64, prove that the number of terms n is 7. -/
theorem geometric_sequence_term_count (a : ℕ → ℚ) :
  a 1 = 1 →
  (∀ k : ℕ, a (k + 1) = a k * (1/2)) →
  (∃ n : ℕ, a n = 1/64) →
  ∃ n : ℕ, n = 7 ∧ a n = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_term_count_l695_69546


namespace NUMINAMATH_CALUDE_work_completion_time_l695_69579

/-- Proves that if A completes a work in 10 days, and A and B together complete the work in 
    2.3076923076923075 days, then B completes the work alone in 3 days. -/
theorem work_completion_time (a_time b_time combined_time : ℝ) 
    (ha : a_time = 10)
    (hc : combined_time = 2.3076923076923075)
    (h_combined : 1 / a_time + 1 / b_time = 1 / combined_time) : 
  b_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l695_69579


namespace NUMINAMATH_CALUDE_max_members_is_414_l695_69593

/-- The number of members in the dance group. -/
def m : ℕ := 414

/-- Represents the condition that when arranged in a square formation, there are 11 members left over. -/
def square_formation_condition (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k^2 + 11

/-- Represents the condition that when arranged in a formation with 5 more rows than columns, there are no members left over. -/
def rectangular_formation_condition (m : ℕ) : Prop :=
  ∃ n : ℕ, m = n * (n + 5)

/-- Theorem stating that 414 is the maximum number of members satisfying both conditions. -/
theorem max_members_is_414 :
  square_formation_condition m ∧
  rectangular_formation_condition m ∧
  ∀ x > m, ¬(square_formation_condition x ∧ rectangular_formation_condition x) :=
by sorry

end NUMINAMATH_CALUDE_max_members_is_414_l695_69593


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l695_69554

/-- The number of valid combinations for a wizard's elixir --/
def validCombinations (herbs : ℕ) (gems : ℕ) (incompatible : ℕ) : ℕ :=
  herbs * gems - incompatible

/-- Theorem: Given 4 herbs, 6 gems, and 3 invalid combinations, 
    the number of valid combinations is 21 --/
theorem wizard_elixir_combinations : 
  validCombinations 4 6 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l695_69554


namespace NUMINAMATH_CALUDE_divided_isosceles_triangle_theorem_l695_69563

/-- An isosceles triangle with a parallel line dividing it -/
structure DividedIsoscelesTriangle where
  /-- The length of the base of the isosceles triangle -/
  base : ℝ
  /-- The length of the parallel line dividing the triangle -/
  parallel_line : ℝ
  /-- The ratio of the area of the smaller region to the whole triangle -/
  area_ratio : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The parallel line is positive and not longer than the base -/
  parallel_line_bounds : 0 < parallel_line ∧ parallel_line ≤ base
  /-- The area ratio is between 0 and 1 -/
  area_ratio_bounds : 0 < area_ratio ∧ area_ratio < 1
  /-- The parallel line divides the triangle according to the area ratio -/
  division_property : (parallel_line / base) ^ 2 = area_ratio

/-- The theorem stating the properties of the divided isosceles triangle -/
theorem divided_isosceles_triangle_theorem (t : DividedIsoscelesTriangle) 
  (h_base : t.base = 24)
  (h_ratio : t.area_ratio = 1/4) : 
  t.parallel_line = 12 := by
  sorry

end NUMINAMATH_CALUDE_divided_isosceles_triangle_theorem_l695_69563


namespace NUMINAMATH_CALUDE_property_necessary_not_sufficient_l695_69505

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define the property that f(x+1) > f(x) for all x ∈ ℝ
def property_f (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) > f x

-- Define what it means for a function to be increasing on ℝ
def increasing_on_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Theorem stating that the property is necessary but not sufficient
-- for the function to be increasing on ℝ
theorem property_necessary_not_sufficient :
  (∀ f : ℝ → ℝ, increasing_on_reals f → property_f f) ∧
  ¬(∀ f : ℝ → ℝ, property_f f → increasing_on_reals f) :=
sorry

end NUMINAMATH_CALUDE_property_necessary_not_sufficient_l695_69505


namespace NUMINAMATH_CALUDE_calculation_proof_l695_69544

theorem calculation_proof :
  ((125 + 17) * 8 = 1136) ∧ ((458 - (85 + 28)) / 23 = 15) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l695_69544


namespace NUMINAMATH_CALUDE_complement_of_union_l695_69559

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l695_69559


namespace NUMINAMATH_CALUDE_w_change_factor_l695_69541

theorem w_change_factor (w w' m z : ℝ) (h_pos_m : m > 0) (h_pos_z : z > 0) :
  let q := 5 * w / (4 * m * z^2)
  let q' := 5 * w' / (4 * (2 * m) * (3 * z)^2)
  q' = 0.2222222222222222 * q → w' = 4 * w := by
  sorry

end NUMINAMATH_CALUDE_w_change_factor_l695_69541


namespace NUMINAMATH_CALUDE_batsman_average_after_25th_innings_l695_69548

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: A batsman's average after 25 innings -/
theorem batsman_average_after_25th_innings 
  (stats : BatsmanStats)
  (h1 : stats.innings = 24)
  (h2 : newAverage stats 80 = stats.average + 3)
  : newAverage stats 80 = 8 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_after_25th_innings_l695_69548


namespace NUMINAMATH_CALUDE_least_possible_z_minus_x_l695_69517

theorem least_possible_z_minus_x (x y z : ℤ) 
  (h1 : x < y ∧ y < z)
  (h2 : y - x > 9)
  (h3 : Even x)
  (h4 : Odd y ∧ Odd z) : 
  (∀ (a b c : ℤ), a < b ∧ b < c ∧ b - a > 9 ∧ Even a ∧ Odd b ∧ Odd c → c - a ≥ 13) ∧
  (∃ (a b c : ℤ), a < b ∧ b < c ∧ b - a > 9 ∧ Even a ∧ Odd b ∧ Odd c ∧ c - a = 13) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_z_minus_x_l695_69517


namespace NUMINAMATH_CALUDE_find_alpha_l695_69558

theorem find_alpha (α β : ℝ) (h1 : α + β = 11) (h2 : α * β = 24) (h3 : α > β) : α = 8 := by
  sorry

end NUMINAMATH_CALUDE_find_alpha_l695_69558


namespace NUMINAMATH_CALUDE_fountain_area_l695_69599

theorem fountain_area (AB CD : ℝ) (h1 : AB = 20) (h2 : CD = 12) : ∃ (area : ℝ), area = 244 * π := by
  sorry

end NUMINAMATH_CALUDE_fountain_area_l695_69599


namespace NUMINAMATH_CALUDE_banana_production_theorem_l695_69506

/-- The total banana production from two islands -/
def total_banana_production (jakies_production : ℕ) (nearby_production : ℕ) : ℕ :=
  jakies_production + nearby_production

/-- Theorem stating the total banana production from Jakies Island and a nearby island -/
theorem banana_production_theorem (nearby_production : ℕ) 
  (h1 : nearby_production = 9000)
  (h2 : ∃ (jakies_production : ℕ), jakies_production = 10 * nearby_production) :
  ∃ (total_production : ℕ), total_production = 99000 ∧ 
  total_production = total_banana_production (10 * nearby_production) nearby_production :=
by
  sorry


end NUMINAMATH_CALUDE_banana_production_theorem_l695_69506


namespace NUMINAMATH_CALUDE_extreme_value_condition_l695_69528

/-- The function f(x) defined as x^3 + ax^2 + 3x - 9 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

/-- The derivative of f(x) with respect to x -/
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem extreme_value_condition (a : ℝ) : 
  (∀ x : ℝ, f a x = f_prime a x) → 
  f_prime a (-3) = 0 → 
  a = 5 := by sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l695_69528


namespace NUMINAMATH_CALUDE_gcf_of_180_252_315_l695_69520

theorem gcf_of_180_252_315 : Nat.gcd 180 (Nat.gcd 252 315) = 9 := by sorry

end NUMINAMATH_CALUDE_gcf_of_180_252_315_l695_69520


namespace NUMINAMATH_CALUDE_vector_product_l695_69501

theorem vector_product (m n : ℝ) : 
  let a : Fin 2 → ℝ := ![m, n]
  let b : Fin 2 → ℝ := ![-1, 2]
  (∃ (k : ℝ), a = k • b) → 
  (‖a‖ = 2 * ‖b‖) →
  m * n = -8 := by sorry

end NUMINAMATH_CALUDE_vector_product_l695_69501


namespace NUMINAMATH_CALUDE_find_X_l695_69504

theorem find_X : ∃ X : ℝ, 
  1.5 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1200.0000000000002 ∧ 
  X = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_find_X_l695_69504


namespace NUMINAMATH_CALUDE_salary_increase_l695_69547

/-- Prove that adding a manager's salary increases the average salary by 100 --/
theorem salary_increase (num_employees : ℕ) (avg_salary : ℚ) (manager_salary : ℚ) :
  num_employees = 20 →
  avg_salary = 1700 →
  manager_salary = 3800 →
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 100 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l695_69547


namespace NUMINAMATH_CALUDE_sin_600_plus_tan_240_l695_69560

theorem sin_600_plus_tan_240 : Real.sin (600 * π / 180) + Real.tan (240 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_plus_tan_240_l695_69560


namespace NUMINAMATH_CALUDE_ron_has_two_friends_l695_69526

/-- The number of Ron's friends eating pizza -/
def num_friends (total_slices : ℕ) (slices_per_person : ℕ) : ℕ :=
  total_slices / slices_per_person - 1

/-- Theorem: Given a 12-slice pizza and 4 slices per person, Ron has 2 friends -/
theorem ron_has_two_friends : num_friends 12 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ron_has_two_friends_l695_69526


namespace NUMINAMATH_CALUDE_carton_width_l695_69565

/-- Represents the dimensions of a rectangular carton -/
structure CartonDimensions where
  length : ℝ
  width : ℝ

/-- Given a carton with dimensions 25 inches by 60 inches, its width is 25 inches -/
theorem carton_width (c : CartonDimensions) 
  (h1 : c.length = 60) 
  (h2 : c.width = 25) : 
  c.width = 25 := by
  sorry

end NUMINAMATH_CALUDE_carton_width_l695_69565


namespace NUMINAMATH_CALUDE_x_range_l695_69574

theorem x_range (x : ℝ) (h1 : x^2 - 8*x + 12 < 0) (h2 : x > 3) : 3 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l695_69574


namespace NUMINAMATH_CALUDE_items_left_in_cart_l695_69594

def initial_items : ℕ := 18
def deleted_items : ℕ := 10

theorem items_left_in_cart : initial_items - deleted_items = 8 := by
  sorry

end NUMINAMATH_CALUDE_items_left_in_cart_l695_69594


namespace NUMINAMATH_CALUDE_assignment_count_theorem_l695_69582

/-- The number of ways to assign 4 distinct objects to 3 distinct groups,
    where each group must contain at least one object. -/
def assignment_count : ℕ := 36

/-- The number of distinct objects to be assigned. -/
def num_objects : ℕ := 4

/-- The number of distinct groups to which objects are assigned. -/
def num_groups : ℕ := 3

theorem assignment_count_theorem :
  (∀ assignment : Fin num_objects → Fin num_groups,
    (∀ g : Fin num_groups, ∃ o : Fin num_objects, assignment o = g) →
    ∃! c : ℕ, c = assignment_count) :=
sorry

end NUMINAMATH_CALUDE_assignment_count_theorem_l695_69582


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l695_69549

-- Define the types of cows
inductive CowType
| Holstein
| Jersey

-- Define the cost of each cow type
def cow_cost : CowType → Nat
| CowType.Holstein => 260
| CowType.Jersey => 170

-- Define the number of hearts in a standard deck
def hearts_in_deck : Nat := 52

-- Define the number of cows
def total_cows : Nat := 2 * hearts_in_deck

-- Define the ratio of Holstein to Jersey cows
def holstein_ratio : Nat := 3
def jersey_ratio : Nat := 2

-- Define the sales tax rate
def sales_tax_rate : Rat := 5 / 100

-- Define the transportation cost per cow
def transport_cost_per_cow : Nat := 20

-- Define the function to calculate the total cost
def total_cost : ℚ :=
  let holstein_count := (holstein_ratio * total_cows) / (holstein_ratio + jersey_ratio)
  let jersey_count := (jersey_ratio * total_cows) / (holstein_ratio + jersey_ratio)
  let base_cost := holstein_count * cow_cost CowType.Holstein + jersey_count * cow_cost CowType.Jersey
  let sales_tax := base_cost * sales_tax_rate
  let transport_cost := total_cows * transport_cost_per_cow
  (base_cost + sales_tax + transport_cost : ℚ)

-- Theorem statement
theorem total_cost_is_correct : total_cost = 26324.50 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l695_69549


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l695_69522

theorem circle_ratio_after_increase (r : ℝ) (h : r > 0) :
  let new_radius := r + 2
  let new_circumference := 2 * Real.pi * new_radius
  let new_diameter := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l695_69522


namespace NUMINAMATH_CALUDE_factorization_3a_squared_minus_3_l695_69538

theorem factorization_3a_squared_minus_3 (a : ℝ) : 3 * a^2 - 3 = 3 * (a - 1) * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3a_squared_minus_3_l695_69538


namespace NUMINAMATH_CALUDE_towel_sets_cost_l695_69556

def guest_sets : ℕ := 2
def master_sets : ℕ := 4
def guest_price : ℚ := 40
def master_price : ℚ := 50
def discount_rate : ℚ := 0.2

def total_cost : ℚ := guest_sets * guest_price + master_sets * master_price
def discount_amount : ℚ := total_cost * discount_rate
def final_cost : ℚ := total_cost - discount_amount

theorem towel_sets_cost : final_cost = 224 := by
  sorry

end NUMINAMATH_CALUDE_towel_sets_cost_l695_69556


namespace NUMINAMATH_CALUDE_find_X_l695_69516

theorem find_X : ∃ X : ℕ, X = 555 * 465 * (3 * (555 - 465)) + (555 - 465)^2 ∧ X = 69688350 := by
  sorry

end NUMINAMATH_CALUDE_find_X_l695_69516


namespace NUMINAMATH_CALUDE_mandy_school_ratio_l695_69530

theorem mandy_school_ratio : 
  ∀ (researched applied accepted : ℕ),
    researched = 42 →
    accepted = 7 →
    2 * accepted = applied →
    (applied : ℚ) / researched = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_mandy_school_ratio_l695_69530


namespace NUMINAMATH_CALUDE_remainder_98_102_div_8_l695_69569

theorem remainder_98_102_div_8 : (98 * 102) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_102_div_8_l695_69569


namespace NUMINAMATH_CALUDE_remaining_milk_l695_69535

-- Define the initial amount of milk
def initial_milk : ℚ := 5

-- Define the amount given away
def given_away : ℚ := 2 + 3/4

-- Theorem statement
theorem remaining_milk :
  initial_milk - given_away = 2 + 1/4 := by sorry

end NUMINAMATH_CALUDE_remaining_milk_l695_69535


namespace NUMINAMATH_CALUDE_diamond_brace_to_ring_ratio_l695_69589

def total_worth : ℕ := 14000
def ring_cost : ℕ := 4000
def car_cost : ℕ := 2000

def diamond_brace_cost : ℕ := total_worth - (ring_cost + car_cost)

theorem diamond_brace_to_ring_ratio :
  diamond_brace_cost / ring_cost = 2 := by sorry

end NUMINAMATH_CALUDE_diamond_brace_to_ring_ratio_l695_69589


namespace NUMINAMATH_CALUDE_power_multiplication_l695_69586

theorem power_multiplication : 2^3 * 5^3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l695_69586


namespace NUMINAMATH_CALUDE_smallest_angle_when_largest_is_120_l695_69519

/-- Represents a trapezoid with angles in arithmetic sequence -/
structure ArithmeticTrapezoid where
  /-- The smallest angle of the trapezoid -/
  smallest_angle : ℝ
  /-- The common difference between consecutive angles -/
  angle_difference : ℝ
  /-- The sum of all angles in the trapezoid is 360° -/
  angle_sum : smallest_angle + (smallest_angle + angle_difference) + 
              (smallest_angle + 2 * angle_difference) + 
              (smallest_angle + 3 * angle_difference) = 360

theorem smallest_angle_when_largest_is_120 (t : ArithmeticTrapezoid) 
  (h : t.smallest_angle + 3 * t.angle_difference = 120) : 
  t.smallest_angle = 60 := by
  sorry

#check smallest_angle_when_largest_is_120

end NUMINAMATH_CALUDE_smallest_angle_when_largest_is_120_l695_69519


namespace NUMINAMATH_CALUDE_chord_cosine_l695_69568

theorem chord_cosine (r : ℝ) (θ φ : ℝ) : 
  r > 0 →
  θ > 0 →
  φ > 0 →
  θ + φ < π →
  8^2 = 2 * r^2 * (1 - Real.cos θ) →
  15^2 = 2 * r^2 * (1 - Real.cos φ) →
  17^2 = 2 * r^2 * (1 - Real.cos (θ + φ)) →
  Real.cos θ = 161 / 225 := by
sorry

end NUMINAMATH_CALUDE_chord_cosine_l695_69568


namespace NUMINAMATH_CALUDE_right_angled_triangle_sets_l695_69552

theorem right_angled_triangle_sets : 
  (3^2 + 4^2 = 5^2) ∧ 
  (30^2 + 40^2 = 50^2) ∧ 
  ((0.3 : ℝ)^2 + (0.4 : ℝ)^2 = (0.5 : ℝ)^2) ∧ 
  (Real.sqrt 3)^2 + (Real.sqrt 4)^2 ≠ (Real.sqrt 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_sets_l695_69552


namespace NUMINAMATH_CALUDE_stock_change_theorem_l695_69542

theorem stock_change_theorem (x : ℝ) (h : x > 0) :
  let day1_value := x * (1 - 0.3)
  let day2_value := day1_value * (1 + 0.5)
  (day2_value - x) / x = 0.05 := by sorry

end NUMINAMATH_CALUDE_stock_change_theorem_l695_69542


namespace NUMINAMATH_CALUDE_percent_not_working_projects_l695_69580

/-- Represents the survey results of employees working on projects -/
structure ProjectSurvey where
  total : ℕ
  projectA : ℕ
  projectB : ℕ
  bothProjects : ℕ

/-- Calculates the percentage of employees not working on either project -/
def percentNotWorking (survey : ProjectSurvey) : ℚ :=
  let workingOnEither := survey.projectA + survey.projectB - survey.bothProjects
  let notWorking := survey.total - workingOnEither
  (notWorking : ℚ) / survey.total * 100

/-- The theorem stating the percentage of employees not working on either project -/
theorem percent_not_working_projects (survey : ProjectSurvey) 
    (h1 : survey.total = 150)
    (h2 : survey.projectA = 90)
    (h3 : survey.projectB = 50)
    (h4 : survey.bothProjects = 30) :
    percentNotWorking survey = 26.67 := by
  sorry


end NUMINAMATH_CALUDE_percent_not_working_projects_l695_69580


namespace NUMINAMATH_CALUDE_cos_equality_implies_77_l695_69550

theorem cos_equality_implies_77 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) 
  (h3 : Real.cos (n * π / 180) = Real.cos (283 * π / 180)) : n = 77 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_implies_77_l695_69550


namespace NUMINAMATH_CALUDE_ab_value_l695_69503

theorem ab_value (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 27) 
  (h3 : a + b + c = 10) : 
  a * b = 9 := by
sorry

end NUMINAMATH_CALUDE_ab_value_l695_69503


namespace NUMINAMATH_CALUDE_train_speed_in_km_hr_l695_69572

-- Define the given parameters
def train_length : ℝ := 50
def platform_length : ℝ := 250
def crossing_time : ℝ := 15

-- Define the conversion factor from m/s to km/hr
def m_s_to_km_hr : ℝ := 3.6

-- Theorem statement
theorem train_speed_in_km_hr :
  let total_distance := train_length + platform_length
  let speed_m_s := total_distance / crossing_time
  speed_m_s * m_s_to_km_hr = 72 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_in_km_hr_l695_69572


namespace NUMINAMATH_CALUDE_probability_colored_ball_l695_69525

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of red balls
def red_balls : ℕ := 2

-- Define the number of blue balls
def blue_balls : ℕ := 5

-- Define the number of white balls
def white_balls : ℕ := 3

-- Theorem: The probability of drawing a colored ball is 7/10
theorem probability_colored_ball :
  (red_balls + blue_balls : ℚ) / total_balls = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_colored_ball_l695_69525


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l695_69590

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x * (x + 1) > 0}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l695_69590


namespace NUMINAMATH_CALUDE_total_crayons_l695_69596

/-- Given that each child has 6 crayons and there are 12 children, prove that the total number of crayons is 72. -/
theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) (h1 : crayons_per_child = 6) (h2 : num_children = 12) :
  crayons_per_child * num_children = 72 := by
  sorry

#check total_crayons

end NUMINAMATH_CALUDE_total_crayons_l695_69596


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l695_69592

/-- Given a hyperbola with asymptotes y = ±(3/4)x, its eccentricity is either 5/4 or 5/3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h : b / a = 3 / 4 ∨ a / b = 3 / 4) :
  let e := c / a
  c^2 = a^2 + b^2 →
  e = 5 / 4 ∨ e = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l695_69592


namespace NUMINAMATH_CALUDE_root_sum_equals_three_l695_69577

-- Define the equations
def equation1 (x : ℝ) : Prop := x + Real.log x = 3
def equation2 (x : ℝ) : Prop := x + (10 : ℝ)^x = 3

-- State the theorem
theorem root_sum_equals_three :
  ∀ x₁ x₂ : ℝ, equation1 x₁ → equation2 x₂ → x₁ + x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_equals_three_l695_69577


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l695_69588

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ ∀ n, ¬p n :=
by sorry

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, 3^n > 2018) ↔ (∀ n : ℕ, 3^n ≤ 2018) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l695_69588


namespace NUMINAMATH_CALUDE_baseball_cards_per_pack_l695_69512

theorem baseball_cards_per_pack : 
  ∀ (num_people : ℕ) (cards_per_person : ℕ) (total_packs : ℕ),
    num_people = 4 →
    cards_per_person = 540 →
    total_packs = 108 →
    (num_people * cards_per_person) / total_packs = 20 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_per_pack_l695_69512


namespace NUMINAMATH_CALUDE_marys_speed_l695_69587

theorem marys_speed (mary_hill_length ann_hill_length ann_speed time_difference : ℝ) 
  (h1 : mary_hill_length = 630)
  (h2 : ann_hill_length = 800)
  (h3 : ann_speed = 40)
  (h4 : time_difference = 13)
  (h5 : ann_hill_length / ann_speed = mary_hill_length / mary_speed + time_difference) :
  mary_speed = 90 :=
by
  sorry

#check marys_speed

end NUMINAMATH_CALUDE_marys_speed_l695_69587


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l695_69508

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_of_M_and_N : M ∩ N = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l695_69508


namespace NUMINAMATH_CALUDE_sandy_clothes_cost_l695_69564

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℚ := 13.99

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℚ := 12.14

/-- The amount Sandy spent on a jacket -/
def jacket_cost : ℚ := 7.43

/-- The total amount Sandy spent on clothes -/
def total_cost : ℚ := shorts_cost + shirt_cost + jacket_cost

/-- Theorem stating that the total amount Sandy spent on clothes is $33.56 -/
theorem sandy_clothes_cost : total_cost = 33.56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_clothes_cost_l695_69564


namespace NUMINAMATH_CALUDE_ralphs_socks_l695_69537

theorem ralphs_socks (x y z : ℕ) : 
  x + y + z = 12 →  -- Total pairs of socks
  x + 3*y + 4*z = 24 →  -- Total cost
  x ≥ 1 →  -- At least one pair of $1 socks
  y ≥ 1 →  -- At least one pair of $3 socks
  z ≥ 1 →  -- At least one pair of $4 socks
  x = 7  -- Number of $1 socks Ralph bought
  := by sorry

end NUMINAMATH_CALUDE_ralphs_socks_l695_69537


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_line_equation_with_equal_intercepts_l695_69515

-- Define the line equation
def line_equation (m x y : ℝ) : Prop :=
  (m + 2) * x - (m + 1) * y - 3 * m - 7 = 0

-- Theorem 1: The line passes through the point (4, 1) for all real m
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m 4 1 :=
sorry

-- Theorem 2: When x and y intercepts are equal, the line equation becomes x+y-5=0
theorem line_equation_with_equal_intercepts :
  ∀ m : ℝ, 
    (∃ k : ℝ, k ≠ 0 ∧ line_equation m k 0 ∧ line_equation m 0 (-k)) →
    ∃ c : ℝ, ∀ x y : ℝ, line_equation m x y ↔ x + y - 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_line_equation_with_equal_intercepts_l695_69515


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l695_69581

theorem sum_of_squares_lower_bound (x y z m : ℝ) (h : x + y + z = m) :
  x^2 + y^2 + z^2 ≥ m^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l695_69581


namespace NUMINAMATH_CALUDE_zeros_in_decimal_representation_l695_69533

theorem zeros_in_decimal_representation (n : ℕ) : 
  (∃ k : ℕ, (1 : ℚ) / (25^10 : ℚ) = (1 : ℚ) / (10^k : ℚ)) ∧ 
  (∀ m : ℕ, m < 20 → (1 : ℚ) / (25^10 : ℚ) < (1 : ℚ) / (10^m : ℚ)) ∧
  (1 : ℚ) / (25^10 : ℚ) ≥ (1 : ℚ) / (10^20 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_zeros_in_decimal_representation_l695_69533


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l695_69567

theorem inserted_numbers_sum (x y : ℝ) : 
  10 < x ∧ x < y ∧ y < 39 ∧  -- x and y are between 10 and 39
  (x / 10 = y / x) ∧         -- 10, x, y form a geometric sequence
  (y - x = 39 - y) →         -- x, y, 39 form an arithmetic sequence
  x + y = 11.25 :=           -- sum of x and y is 11¼
by sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l695_69567


namespace NUMINAMATH_CALUDE_fraction_problem_l695_69502

theorem fraction_problem (N : ℝ) (f : ℝ) (h1 : N = 12) (h2 : 1 + f * N = 0.75 * N) : f = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l695_69502


namespace NUMINAMATH_CALUDE_technician_round_trip_l695_69575

/-- Represents the percentage of a round-trip journey completed -/
def round_trip_percentage (outbound_percent : ℝ) (return_percent : ℝ) : ℝ :=
  (outbound_percent + return_percent * outbound_percent) * 50

/-- Theorem stating that completing the outbound journey and 10% of the return journey
    results in 55% of the round-trip being completed -/
theorem technician_round_trip :
  round_trip_percentage 100 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_technician_round_trip_l695_69575


namespace NUMINAMATH_CALUDE_cube_surface_covering_l695_69523

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  side_length : ℝ
  height : ℝ
  area : ℝ

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where
  side_length : ℝ
  surface_area : ℝ

/-- A covering is a collection of shapes that cover a surface. -/
structure Covering where
  shapes : List Rhombus
  total_area : ℝ

/-- Theorem: It is possible to cover the surface of a cube with fewer than six rhombuses. -/
theorem cube_surface_covering (c : Cube) : 
  ∃ (cov : Covering), cov.shapes.length < 6 ∧ cov.total_area = c.surface_area := by
  sorry


end NUMINAMATH_CALUDE_cube_surface_covering_l695_69523


namespace NUMINAMATH_CALUDE_fraction_equality_l695_69521

theorem fraction_equality : (1622^2 - 1615^2) / (1629^2 - 1608^2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l695_69521
