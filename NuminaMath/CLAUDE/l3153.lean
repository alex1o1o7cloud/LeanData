import Mathlib

namespace NUMINAMATH_CALUDE_two_integers_sum_l3153_315308

theorem two_integers_sum (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ x - y = 4 ∧ x * y = 156 → x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_l3153_315308


namespace NUMINAMATH_CALUDE_solution_implies_k_value_l3153_315368

theorem solution_implies_k_value (x y k : ℝ) :
  x = -3 → y = 2 → 2 * x + k * y = 0 → k = 3 := by sorry

end NUMINAMATH_CALUDE_solution_implies_k_value_l3153_315368


namespace NUMINAMATH_CALUDE_horner_v1_at_negative_two_l3153_315366

def horner_polynomial (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

def horner_v0 : ℝ := 1

def horner_v1 (x : ℝ) : ℝ := horner_v0 * x - 5

theorem horner_v1_at_negative_two :
  horner_v1 (-2) = -7 :=
sorry

end NUMINAMATH_CALUDE_horner_v1_at_negative_two_l3153_315366


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l3153_315314

theorem triangle_angle_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- All angles are positive
  a + b + c = 180 →        -- Sum of angles is 180 degrees
  a = 20 →                 -- Smallest angle is 20 degrees
  c = 5 * a →              -- Largest angle is 5 times the smallest
  a ≤ b ∧ b ≤ c →          -- a is smallest, c is largest
  b / a = 3 :=             -- Ratio of middle to smallest is 3:1
by sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l3153_315314


namespace NUMINAMATH_CALUDE_room_width_is_correct_l3153_315385

/-- The width of a room satisfying given conditions -/
def room_width : ℝ :=
  let length : ℝ := 25
  let height : ℝ := 12
  let door_area : ℝ := 6 * 3
  let window_area : ℝ := 4 * 3
  let num_windows : ℕ := 3
  let whitewash_cost_per_sqft : ℝ := 3
  let total_whitewash_cost : ℝ := 2718
  15

/-- Theorem stating that the room width is correct given the conditions -/
theorem room_width_is_correct :
  let length : ℝ := 25
  let height : ℝ := 12
  let door_area : ℝ := 6 * 3
  let window_area : ℝ := 4 * 3
  let num_windows : ℕ := 3
  let whitewash_cost_per_sqft : ℝ := 3
  let total_whitewash_cost : ℝ := 2718
  let width := room_width
  whitewash_cost_per_sqft * (2 * (length * height) + 2 * (width * height) - door_area - num_windows * window_area) = total_whitewash_cost :=
by
  sorry


end NUMINAMATH_CALUDE_room_width_is_correct_l3153_315385


namespace NUMINAMATH_CALUDE_trig_identities_l3153_315311

theorem trig_identities (x : Real) 
  (h1 : 0 < x) (h2 : x < Real.pi) 
  (h3 : Real.sin x + Real.cos x = 7/13) : 
  (Real.sin x * Real.cos x = -60/169) ∧ 
  ((5 * Real.sin x + 4 * Real.cos x) / (15 * Real.sin x - 7 * Real.cos x) = 8/43) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l3153_315311


namespace NUMINAMATH_CALUDE_fraction_simplification_l3153_315350

theorem fraction_simplification (m : ℝ) (h : m ≠ 0) : 
  (3 * m^3) / (6 * m^2) = m / 2 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3153_315350


namespace NUMINAMATH_CALUDE_fixed_points_exist_l3153_315362

-- Define the fixed point F and the line l
def F : ℝ × ℝ := (1, 0)
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 4}

-- Define the trajectory E
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

-- Define the point A where E intersects the negative x-axis
def A : ℝ × ℝ := (-2, 0)

-- Define a function to represent a line through F that intersects E at two points
def line_through_F (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = m * p.2 + 1}

-- Define the theorem to be proved
theorem fixed_points_exist (m : ℝ) (hm : m ≠ 0) : 
  ∃ (B C M N : ℝ × ℝ) (Q : ℝ × ℝ),
    B ∈ E ∩ line_through_F m ∧
    C ∈ E ∩ line_through_F m ∧
    M ∈ l ∧
    N ∈ l ∧
    (Q = (1, 0) ∨ Q = (7, 0)) ∧
    ((Q.1 - M.1) * (Q.1 - N.1) + (Q.2 - M.2) * (Q.2 - N.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_exist_l3153_315362


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3153_315318

theorem inequality_system_solution (x : ℝ) :
  (x - 4 > (3/2) * x - 3) ∧
  ((2 + x) / 3 - 1 ≤ (1 + x) / 2) →
  -5 ≤ x ∧ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3153_315318


namespace NUMINAMATH_CALUDE_rabbit_speed_l3153_315379

/-- Proves that given a dog running at 24 miles per hour chasing a rabbit with a 0.6-mile head start,
    if it takes the dog 4 minutes to catch up to the rabbit, then the rabbit's speed is 15 miles per hour. -/
theorem rabbit_speed (dog_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  dog_speed = 24 →
  head_start = 0.6 →
  catch_up_time = 4 / 60 →
  ∃ (rabbit_speed : ℝ),
    rabbit_speed * catch_up_time = dog_speed * catch_up_time - head_start ∧
    rabbit_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_speed_l3153_315379


namespace NUMINAMATH_CALUDE_distance_and_intersection_l3153_315304

def point1 : ℝ × ℝ := (3, 7)
def point2 : ℝ × ℝ := (-5, 3)

theorem distance_and_intersection :
  let distance := Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2)
  let slope := (point2.2 - point1.2) / (point2.1 - point1.1)
  let y_intercept := point1.2 - slope * point1.1
  let line := fun x => slope * x + y_intercept
  (distance = 4 * Real.sqrt 5) ∧
  (line (-1) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_distance_and_intersection_l3153_315304


namespace NUMINAMATH_CALUDE_car_trip_distance_l3153_315312

theorem car_trip_distance (D : ℝ) 
  (h1 : D / 2 + D / 2 = D)  -- First stop at 1/2 of total distance
  (h2 : D / 2 - (D / 2) / 4 + (D / 2) / 4 = D / 2)  -- Second stop at 1/4 of remaining distance
  (h3 : D - D / 2 - (D / 2) / 4 = 105)  -- Remaining distance after second stop is 105 miles
  : D = 280 := by
sorry

end NUMINAMATH_CALUDE_car_trip_distance_l3153_315312


namespace NUMINAMATH_CALUDE_expression_value_l3153_315331

theorem expression_value : 
  ∃ (m : ℕ), (3^1005 + 7^1006)^2 - (3^1005 - 7^1006)^2 = m * 10^1006 ∧ m = 280 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3153_315331


namespace NUMINAMATH_CALUDE_muffins_per_box_l3153_315389

theorem muffins_per_box (total_muffins : ℕ) (available_boxes : ℕ) (additional_boxes : ℕ) :
  total_muffins = 95 →
  available_boxes = 10 →
  additional_boxes = 9 →
  (total_muffins / (available_boxes + additional_boxes) : ℚ) = 5 := by
sorry

end NUMINAMATH_CALUDE_muffins_per_box_l3153_315389


namespace NUMINAMATH_CALUDE_sum_110_terms_l3153_315388

-- Define an arithmetic sequence type
def ArithmeticSequence := ℕ → ℤ

-- Define the sum of the first n terms of an arithmetic sequence
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (List.range n).map seq |>.sum

-- Define the properties of our specific arithmetic sequence
def special_arithmetic_sequence (seq : ArithmeticSequence) : Prop :=
  sum_n_terms seq 10 = 100 ∧ sum_n_terms seq 100 = 10

-- State the theorem
theorem sum_110_terms (seq : ArithmeticSequence) 
  (h : special_arithmetic_sequence seq) : 
  sum_n_terms seq 110 = -110 := by
  sorry

end NUMINAMATH_CALUDE_sum_110_terms_l3153_315388


namespace NUMINAMATH_CALUDE_yoongi_calculation_l3153_315320

theorem yoongi_calculation (x : ℕ) : 
  (x ≥ 10 ∧ x < 100) → (x - 35 = 27) → (x - 53 = 9) := by
  sorry

end NUMINAMATH_CALUDE_yoongi_calculation_l3153_315320


namespace NUMINAMATH_CALUDE_incorrect_conclusion_l3153_315354

theorem incorrect_conclusion (a b c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b > c) (h4 : c > 0) :
  ¬((a / b) > (a / c)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_conclusion_l3153_315354


namespace NUMINAMATH_CALUDE_nabla_four_seven_l3153_315393

-- Define the nabla operation
def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

-- Theorem statement
theorem nabla_four_seven : nabla 4 7 = 11 / 29 := by
  sorry

end NUMINAMATH_CALUDE_nabla_four_seven_l3153_315393


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3153_315328

/-- The perimeter of a rhombus with diagonals 8 and 30 inches is 4√241 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * s = 4 * Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3153_315328


namespace NUMINAMATH_CALUDE_amount_to_hand_in_l3153_315356

/-- Represents the contents of Jack's till -/
structure TillContents where
  usd_100: Nat
  usd_50: Nat
  usd_20: Nat
  usd_10: Nat
  usd_5: Nat
  usd_1: Nat
  quarters: Nat
  dimes: Nat
  nickels: Nat
  pennies: Nat
  euro_5: Nat
  gbp_10: Nat

/-- Exchange rates -/
def euro_to_usd : Rat := 118/100
def gbp_to_usd : Rat := 139/100

/-- The amount to be left in the till -/
def amount_to_leave : Rat := 300

/-- Calculate the total amount in USD -/
def total_amount (contents : TillContents) : Rat :=
  contents.usd_100 * 100 +
  contents.usd_50 * 50 +
  contents.usd_20 * 20 +
  contents.usd_10 * 10 +
  contents.usd_5 * 5 +
  contents.usd_1 +
  contents.quarters * (1/4) +
  contents.dimes * (1/10) +
  contents.nickels * (1/20) +
  contents.pennies * (1/100) +
  contents.euro_5 * 5 * euro_to_usd +
  contents.gbp_10 * 10 * gbp_to_usd

/-- Calculate the total amount of coins -/
def total_coins (contents : TillContents) : Rat :=
  contents.quarters * (1/4) +
  contents.dimes * (1/10) +
  contents.nickels * (1/20) +
  contents.pennies * (1/100)

/-- Jack's till contents -/
def jacks_till : TillContents := {
  usd_100 := 2,
  usd_50 := 1,
  usd_20 := 5,
  usd_10 := 3,
  usd_5 := 7,
  usd_1 := 27,
  quarters := 42,
  dimes := 19,
  nickels := 36,
  pennies := 47,
  euro_5 := 20,
  gbp_10 := 25
}

theorem amount_to_hand_in :
  total_amount jacks_till - (amount_to_leave + total_coins jacks_till) = 607.5 := by
  sorry

end NUMINAMATH_CALUDE_amount_to_hand_in_l3153_315356


namespace NUMINAMATH_CALUDE_factorization_equality_l3153_315369

theorem factorization_equality (x y : ℝ) :
  (2*x - y) * (x + 3*y) - (2*x + 3*y) * (y - 2*x) = 3 * (2*x - y) * (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3153_315369


namespace NUMINAMATH_CALUDE_chair_probability_l3153_315345

/- Define the number of chairs -/
def total_chairs : ℕ := 10

/- Define the number of broken chairs -/
def broken_chairs : ℕ := 2

/- Define the number of usable chairs -/
def usable_chairs : ℕ := total_chairs - broken_chairs

/- Define the number of adjacent pairs in usable chairs -/
def adjacent_pairs : ℕ := usable_chairs - 1 - 1  -- Subtract 1 for the gap between 4 and 7

/- Define the probability of not sitting next to each other -/
def prob_not_adjacent : ℚ := 11 / 14

theorem chair_probability : 
  prob_not_adjacent = 1 - (adjacent_pairs : ℚ) / (usable_chairs.choose 2) :=
by sorry

end NUMINAMATH_CALUDE_chair_probability_l3153_315345


namespace NUMINAMATH_CALUDE_checkerboard_triangle_area_theorem_l3153_315326

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A triangle on a 2D grid -/
structure GridTriangle where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint

/-- The area of a triangle -/
def triangleArea (t : GridTriangle) : ℝ :=
  sorry

/-- Whether two triangles are similar -/
def areSimilar (t1 t2 : GridTriangle) : Prop :=
  sorry

/-- The area of the white part of a triangle -/
def whiteArea (t : GridTriangle) : ℝ :=
  sorry

/-- The area of the black part of a triangle -/
def blackArea (t : GridTriangle) : ℝ :=
  sorry

/-- The main theorem -/
theorem checkerboard_triangle_area_theorem (X : GridTriangle) (S : ℝ) 
  (h : triangleArea X = S) : 
  ∃ Y : GridTriangle, areSimilar X Y ∧ whiteArea Y = S ∧ blackArea Y = S :=
sorry

end NUMINAMATH_CALUDE_checkerboard_triangle_area_theorem_l3153_315326


namespace NUMINAMATH_CALUDE_homes_numbering_twos_l3153_315342

/-- In a city with 100 homes numbered from 1 to 100, 
    the number of 2's used in the numbering is 20. -/
theorem homes_numbering_twos (homes : Nat) (twos_used : Nat) : 
  homes = 100 → twos_used = 20 := by
  sorry

#check homes_numbering_twos

end NUMINAMATH_CALUDE_homes_numbering_twos_l3153_315342


namespace NUMINAMATH_CALUDE_triangle_ABC_is_obtuse_l3153_315324

theorem triangle_ABC_is_obtuse (A B C : Real) (hA : A = 10) (hB : B = 60) 
  (hsum : A + B + C = 180) : C > 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_is_obtuse_l3153_315324


namespace NUMINAMATH_CALUDE_simplify_expression_l3153_315363

theorem simplify_expression (x : ℝ) : 3*x + 4 - x + 8 = 2*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3153_315363


namespace NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l3153_315319

theorem quadratic_equation_with_given_roots :
  ∀ (f : ℝ → ℝ),
  (∀ x, f x = 0 ↔ x = -5 ∨ x = 7) →
  (∀ x, f x = (x + 5) * (x - 7)) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l3153_315319


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l3153_315339

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^8 + 21 * x^4 * y^3 + 49 * y^6) = 27 * x^12 - 343 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l3153_315339


namespace NUMINAMATH_CALUDE_unique_solution_square_equation_l3153_315338

theorem unique_solution_square_equation :
  ∃! x : ℝ, (10 - x)^2 = x^2 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_square_equation_l3153_315338


namespace NUMINAMATH_CALUDE_parametric_to_standard_equation_l3153_315382

/-- Given parametric equations x = 1 + (1/2)t and y = 5 + (√3/2)t,
    prove they are equivalent to the standard equation √3x - y + 5 - √3 = 0 -/
theorem parametric_to_standard_equation 
  (t x y : ℝ) 
  (h1 : x = 1 + (1/2) * t) 
  (h2 : y = 5 + (Real.sqrt 3 / 2) * t) :
  Real.sqrt 3 * x - y + 5 - Real.sqrt 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_parametric_to_standard_equation_l3153_315382


namespace NUMINAMATH_CALUDE_probability_is_31_145_l3153_315344

-- Define the shoe collection
def total_pairs : ℕ := 15
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 5
def red_pairs : ℕ := 2

-- Define the probability function
def probability_same_color_different_foot : ℚ :=
  -- Black shoes probability
  (2 * black_pairs : ℚ) / (2 * total_pairs) * (black_pairs : ℚ) / (2 * total_pairs - 1) +
  -- Brown shoes probability
  (2 * brown_pairs : ℚ) / (2 * total_pairs) * (brown_pairs : ℚ) / (2 * total_pairs - 1) +
  -- Red shoes probability
  (2 * red_pairs : ℚ) / (2 * total_pairs) * (red_pairs : ℚ) / (2 * total_pairs - 1)

-- Theorem statement
theorem probability_is_31_145 : probability_same_color_different_foot = 31 / 145 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_31_145_l3153_315344


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_8_l3153_315394

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

theorem least_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → 181 ≤ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_8_l3153_315394


namespace NUMINAMATH_CALUDE_max_value_theorem_l3153_315315

/-- Given a line ax + 2by - 1 = 0 intercepting a chord of length 2√3 on the circle x^2 + y^2 = 4,
    the maximum value of 3a + 2b is √10. -/
theorem max_value_theorem (a b : ℝ) : 
  (∃ x y : ℝ, a * x + 2 * b * y - 1 = 0 ∧ x^2 + y^2 = 4) →  -- Line intersects circle
  (∃ x₁ y₁ x₂ y₂ : ℝ, a * x₁ + 2 * b * y₁ - 1 = 0 ∧ 
                     a * x₂ + 2 * b * y₂ - 1 = 0 ∧
                     x₁^2 + y₁^2 = 4 ∧
                     x₂^2 + y₂^2 = 4 ∧
                     (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) →  -- Chord length is 2√3
  (a^2 + 4 * b^2 = 1) →  -- Distance from center to line is 1
  (∀ c : ℝ, 3 * a + 2 * b ≤ c → c ≥ Real.sqrt 10) ∧ 
  (∃ a₀ b₀ : ℝ, 3 * a₀ + 2 * b₀ = Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3153_315315


namespace NUMINAMATH_CALUDE_smallest_angle_satisfying_equation_l3153_315309

theorem smallest_angle_satisfying_equation :
  let y := Real.pi / 18
  (∀ z : Real, 0 < z ∧ z < y → Real.sin (4 * z) * Real.sin (5 * z) ≠ Real.cos (4 * z) * Real.cos (5 * z)) ∧
  Real.sin (4 * y) * Real.sin (5 * y) = Real.cos (4 * y) * Real.cos (5 * y) := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_satisfying_equation_l3153_315309


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3153_315360

theorem tan_alpha_value (α : Real) 
  (h1 : Real.sin (π - α) = Real.sqrt 5 / 5)
  (h2 : π / 2 < α ∧ α < π) : 
  Real.tan α = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3153_315360


namespace NUMINAMATH_CALUDE_f_log_4_9_l3153_315371

/-- A function that is even and equals 2^x for negative x -/
def f (x : ℝ) : ℝ := sorry

/-- f is an even function -/
axiom f_even : ∀ x : ℝ, f x = f (-x)

/-- f(x) = 2^x for x < 0 -/
axiom f_neg : ∀ x : ℝ, x < 0 → f x = 2^x

/-- The main theorem: f(log_4(9)) = 1/3 -/
theorem f_log_4_9 : f (Real.log 9 / Real.log 4) = 1/3 := by sorry

end NUMINAMATH_CALUDE_f_log_4_9_l3153_315371


namespace NUMINAMATH_CALUDE_twenty_men_handshakes_l3153_315384

/-- The number of handshakes in a complete graph with n vertices -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating that 20 men result in 190 handshakes -/
theorem twenty_men_handshakes :
  ∃ n : ℕ, n > 0 ∧ handshakes n = 190 ∧ n = 20 := by
  sorry

#check twenty_men_handshakes

end NUMINAMATH_CALUDE_twenty_men_handshakes_l3153_315384


namespace NUMINAMATH_CALUDE_students_history_not_statistics_l3153_315337

/-- Given a group of students with the following properties:
  * There are 89 students in total
  * 36 students are taking history
  * 32 students are taking statistics
  * 59 students are taking history or statistics or both
  This theorem proves that 27 students are taking history but not statistics. -/
theorem students_history_not_statistics 
  (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ)
  (h_total : total = 89)
  (h_history : history = 36)
  (h_statistics : statistics = 32)
  (h_history_or_statistics : history_or_statistics = 59) :
  history - (history + statistics - history_or_statistics) = 27 := by
  sorry

end NUMINAMATH_CALUDE_students_history_not_statistics_l3153_315337


namespace NUMINAMATH_CALUDE_simplify_expression_l3153_315399

theorem simplify_expression : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 49) = (3 + 2 * Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3153_315399


namespace NUMINAMATH_CALUDE_f_nonnegative_when_a_is_one_f_has_two_zeros_iff_a_in_open_unit_interval_l3153_315397

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1 - 2 * log x

theorem f_nonnegative_when_a_is_one (x : ℝ) (h : x > 0) :
  f 1 x ≥ 0 := by sorry

theorem f_has_two_zeros_iff_a_in_open_unit_interval (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔
  0 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_when_a_is_one_f_has_two_zeros_iff_a_in_open_unit_interval_l3153_315397


namespace NUMINAMATH_CALUDE_third_quadrant_condition_l3153_315307

-- Define the complex number z
def z (a : ℝ) : ℂ := Complex.mk (a - 1) (a + 1)

-- Define the condition for a point to be in the third quadrant
def in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

-- Theorem statement
theorem third_quadrant_condition (a : ℝ) :
  in_third_quadrant (z a) ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_third_quadrant_condition_l3153_315307


namespace NUMINAMATH_CALUDE_box_weight_difference_l3153_315381

theorem box_weight_difference (first_box_weight third_box_weight : ℕ) 
  (h1 : first_box_weight = 2)
  (h2 : third_box_weight = 13) : 
  third_box_weight - first_box_weight = 11 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_difference_l3153_315381


namespace NUMINAMATH_CALUDE_vector_sum_parallel_l3153_315364

def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, 1)

theorem vector_sum_parallel (m : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ (a + 2 • b m) = k • (2 • a - b m)) → 
  (a + b m) = (-3/2, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_parallel_l3153_315364


namespace NUMINAMATH_CALUDE_polynomial_multiplication_equality_l3153_315377

-- Define the polynomials
def p (y : ℝ) : ℝ := 2*y - 1
def q (y : ℝ) : ℝ := 5*y^12 - 3*y^11 + y^9 - 4*y^8
def r (y : ℝ) : ℝ := 10*y^13 - 11*y^12 + 3*y^11 + y^10 - 9*y^9 + 4*y^8

-- Theorem statement
theorem polynomial_multiplication_equality :
  ∀ y : ℝ, p y * q y = r y :=
by sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_equality_l3153_315377


namespace NUMINAMATH_CALUDE_bus_row_capacity_l3153_315305

/-- Represents a bus with a given number of rows and total capacity -/
structure Bus where
  rows : ℕ
  capacity : ℕ

/-- Calculates the number of children each row can accommodate -/
def childrenPerRow (bus : Bus) : ℕ := bus.capacity / bus.rows

/-- Theorem: Given a bus with 9 rows and a capacity of 36 children,
    prove that each row can accommodate 4 children -/
theorem bus_row_capacity (bus : Bus) 
    (h_rows : bus.rows = 9) 
    (h_capacity : bus.capacity = 36) : 
    childrenPerRow bus = 4 := by
  sorry

end NUMINAMATH_CALUDE_bus_row_capacity_l3153_315305


namespace NUMINAMATH_CALUDE_simplify_expression_l3153_315386

theorem simplify_expression : (2^8 + 7^3) * (2^2 - (-2)^3)^5 = 149062368 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3153_315386


namespace NUMINAMATH_CALUDE_number_equals_two_l3153_315378

theorem number_equals_two : ∃ x : ℝ, 0.4 * x = 0.8 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_number_equals_two_l3153_315378


namespace NUMINAMATH_CALUDE_percentage_of_number_l3153_315398

theorem percentage_of_number (x : ℝ) (h : (1/4) * (1/3) * (2/5) * x = 16) : 
  (40/100) * x = 192 := by sorry

end NUMINAMATH_CALUDE_percentage_of_number_l3153_315398


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l3153_315303

/-- A trapezoid with given side lengths -/
structure Trapezoid :=
  (EF : ℝ)
  (GH : ℝ)
  (EG : ℝ)
  (FH : ℝ)
  (is_trapezoid : EF ≠ GH)

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.EF + t.GH + t.EG + t.FH

/-- Theorem: The perimeter of the given trapezoid is 38 units -/
theorem trapezoid_perimeter :
  ∃ (t : Trapezoid), t.EF = 10 ∧ t.GH = 14 ∧ t.EG = 7 ∧ t.FH = 7 ∧ perimeter t = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l3153_315303


namespace NUMINAMATH_CALUDE_ratio_problem_l3153_315316

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 2 / 5)
  (h3 : c / d = 9) : 
  d / a = 5 / 54 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3153_315316


namespace NUMINAMATH_CALUDE_range_of_m_l3153_315332

/-- The function f(x) as defined in the problem -/
def f (a x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

/-- The theorem statement -/
theorem range_of_m (m : ℝ) :
  (∀ a ∈ Set.Icc (-3 : ℝ) 0,
    ∀ x₁ ∈ Set.Icc 0 2,
    ∀ x₂ ∈ Set.Icc 0 2,
    m - a * m^2 ≥ |f a x₁ - f a x₂|) →
  m ∈ Set.Ici 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3153_315332


namespace NUMINAMATH_CALUDE_ranas_speed_l3153_315321

/-- Proves that Rana's speed is 5 kmph given the problem conditions -/
theorem ranas_speed (circumference : ℝ) (ajith_speed : ℝ) (meeting_time : ℝ) 
  (h1 : circumference = 115)
  (h2 : ajith_speed = 4)
  (h3 : meeting_time = 115) :
  ∃ v : ℝ, v = 5 ∧ 
    (v * meeting_time - ajith_speed * meeting_time) / circumference = 1 :=
by sorry

end NUMINAMATH_CALUDE_ranas_speed_l3153_315321


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3153_315352

/-- Given a bus that stops for 45 minutes per hour and has an average speed of 15 km/hr including stoppages,
    prove that its average speed excluding stoppages is 60 km/hr. -/
theorem bus_speed_excluding_stoppages (stop_time : ℝ) (avg_speed_with_stops : ℝ) 
  (h1 : stop_time = 45) 
  (h2 : avg_speed_with_stops = 15) :
  let moving_time : ℝ := 60 - stop_time
  let speed_excluding_stops : ℝ := (avg_speed_with_stops * 60) / moving_time
  speed_excluding_stops = 60 := by
sorry

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3153_315352


namespace NUMINAMATH_CALUDE_distinct_groups_eq_seven_l3153_315346

/-- The number of distinct groups of 3 marbles Tom can choose -/
def distinct_groups : ℕ :=
  let red_marbles : ℕ := 1
  let green_marbles : ℕ := 1
  let blue_marbles : ℕ := 1
  let yellow_marbles : ℕ := 4
  let non_yellow_marbles : ℕ := red_marbles + green_marbles + blue_marbles
  let all_yellow_groups : ℕ := 1
  let two_yellow_groups : ℕ := non_yellow_marbles
  let one_yellow_groups : ℕ := Nat.choose non_yellow_marbles 2
  all_yellow_groups + two_yellow_groups + one_yellow_groups

theorem distinct_groups_eq_seven : distinct_groups = 7 := by
  sorry

end NUMINAMATH_CALUDE_distinct_groups_eq_seven_l3153_315346


namespace NUMINAMATH_CALUDE_triangle_angle_value_l3153_315370

theorem triangle_angle_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c →
  B = π / 3 ∨ B = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l3153_315370


namespace NUMINAMATH_CALUDE_knife_value_l3153_315340

theorem knife_value (n : ℕ) (k : ℕ) (m : ℕ) :
  (n * n = 20 * k + 10 + m) →
  (1 ≤ m) →
  (m ≤ 9) →
  (∃ b : ℕ, 10 - b = m + b) →
  (∃ b : ℕ, b = 2) :=
by sorry

end NUMINAMATH_CALUDE_knife_value_l3153_315340


namespace NUMINAMATH_CALUDE_parabola_rotation_l3153_315335

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Rotates a parabola by 180° around its vertex --/
def rotate180 (p : Parabola) : Parabola :=
  { a := -p.a, h := p.h, k := p.k }

theorem parabola_rotation (p : Parabola) (hp : p = { a := 2, h := 3, k := -2 }) :
  rotate180 p = { a := -2, h := 3, k := -2 } := by
  sorry

#check parabola_rotation

end NUMINAMATH_CALUDE_parabola_rotation_l3153_315335


namespace NUMINAMATH_CALUDE_root_in_interval_l3153_315390

-- Define the function f(x) = x^3 - x - 1
def f (x : ℝ) : ℝ := x^3 - x - 1

-- Theorem statement
theorem root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l3153_315390


namespace NUMINAMATH_CALUDE_petya_larger_than_vasya_l3153_315330

theorem petya_larger_than_vasya : 2^25 > 4^12 := by
  sorry

end NUMINAMATH_CALUDE_petya_larger_than_vasya_l3153_315330


namespace NUMINAMATH_CALUDE_boat_round_trip_equation_l3153_315306

/-- Represents the equation for a boat's round trip between two points -/
def boat_equation (distance : ℝ) (flow_speed : ℝ) (boat_speed : ℝ) (total_time : ℝ) : Prop :=
  (distance / (boat_speed + flow_speed)) + (distance / (boat_speed - flow_speed)) = total_time

/-- Theorem stating that the given equation correctly represents the boat's round trip -/
theorem boat_round_trip_equation : 
  ∀ (x : ℝ), x > 5 → boat_equation 60 5 x 8 :=
by sorry

end NUMINAMATH_CALUDE_boat_round_trip_equation_l3153_315306


namespace NUMINAMATH_CALUDE_first_interest_rate_is_five_percent_l3153_315323

-- Define the total amount, amounts lent at each rate, and the known interest rate
def total_amount : ℝ := 2500
def amount_first_rate : ℝ := 2000
def amount_second_rate : ℝ := total_amount - amount_first_rate
def second_rate : ℝ := 6

-- Define the total yearly annual income
def total_income : ℝ := 130

-- Define the first interest rate as a variable
variable (first_rate : ℝ)

-- Theorem statement
theorem first_interest_rate_is_five_percent :
  (amount_first_rate * first_rate / 100 + amount_second_rate * second_rate / 100 = total_income) →
  first_rate = 5 := by
sorry

end NUMINAMATH_CALUDE_first_interest_rate_is_five_percent_l3153_315323


namespace NUMINAMATH_CALUDE_worker_net_income_proof_l3153_315380

/-- Calculates the net income after tax for a tax resident worker --/
def netIncomeAfterTax (grossIncome : ℝ) (taxRate : ℝ) : ℝ :=
  grossIncome * (1 - taxRate)

/-- Proves that the net income after tax for a worker credited with 45000 and a 13% tax rate is 39150 --/
theorem worker_net_income_proof :
  let grossIncome : ℝ := 45000
  let taxRate : ℝ := 0.13
  netIncomeAfterTax grossIncome taxRate = 39150 := by
sorry

#eval netIncomeAfterTax 45000 0.13

end NUMINAMATH_CALUDE_worker_net_income_proof_l3153_315380


namespace NUMINAMATH_CALUDE_employee_bonuses_l3153_315396

theorem employee_bonuses :
  ∃ (x y z : ℝ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y + z = 2970 ∧
    y = (1/3) * x + 180 ∧
    z = (1/3) * y + 130 ∧
    x = 1800 ∧ y = 780 ∧ z = 390 := by
  sorry

end NUMINAMATH_CALUDE_employee_bonuses_l3153_315396


namespace NUMINAMATH_CALUDE_T_is_three_rays_l3153_315310

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set T as described in the problem -/
def T : Set Point :=
  {p : Point | (4 = p.x + 1 ∧ p.y - 5 ≤ 4) ∨
               (4 = p.y - 5 ∧ p.x + 1 ≤ 4) ∨
               (p.x + 1 = p.y - 5 ∧ 4 ≤ p.x + 1)}

/-- A ray starting from a point in a given direction -/
structure Ray where
  start : Point
  direction : ℝ × ℝ

/-- The three rays that should describe T -/
def threeRays : List Ray :=
  [{ start := ⟨3, 9⟩, direction := (0, -1) },   -- Vertically downward
   { start := ⟨3, 9⟩, direction := (-1, 0) },   -- Horizontally leftward
   { start := ⟨3, 9⟩, direction := (1, 1) }]    -- Diagonally upward

/-- Theorem stating that T is equivalent to three rays with a common point -/
theorem T_is_three_rays : 
  ∀ p : Point, p ∈ T ↔ ∃ r ∈ threeRays, ∃ t : ℝ, t ≥ 0 ∧ 
    p.x = r.start.x + t * r.direction.1 ∧ 
    p.y = r.start.y + t * r.direction.2 :=
sorry

end NUMINAMATH_CALUDE_T_is_three_rays_l3153_315310


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3153_315358

/-- Represents the composition of teachers in a school -/
structure TeacherComposition where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  junior : ℕ
  total_sum : total = senior + intermediate + junior

/-- Represents the sample of teachers -/
structure TeacherSample where
  size : ℕ
  senior : ℕ
  intermediate : ℕ
  junior : ℕ
  size_sum : size = senior + intermediate + junior

/-- Theorem stating the correct stratified sampling for the given teacher composition -/
theorem stratified_sampling_theorem 
  (school : TeacherComposition) 
  (sample : TeacherSample) 
  (h1 : school.total = 300) 
  (h2 : school.senior = 90) 
  (h3 : school.intermediate = 150) 
  (h4 : school.junior = 60) 
  (h5 : sample.size = 40) : 
  sample.senior = 12 ∧ sample.intermediate = 20 ∧ sample.junior = 8 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3153_315358


namespace NUMINAMATH_CALUDE_division_of_decimals_l3153_315355

theorem division_of_decimals : (0.45 : ℝ) / 0.005 = 90 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l3153_315355


namespace NUMINAMATH_CALUDE_parabola_properties_l3153_315375

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Two points on a parabola -/
structure ParabolaPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Theorem about properties of points on a parabola -/
theorem parabola_properties
  (E : Parabola)
  (pts : ParabolaPoints)
  (symmetry_line : Line)
  (h1 : E.equation = fun x y ↦ y^2 = 4*x)
  (h2 : pts.A.1 ≠ pts.B.1 ∨ pts.A.2 ≠ pts.B.2)
  (h3 : E.equation pts.A.1 pts.A.2 ∧ E.equation pts.B.1 pts.B.2)
  (h4 : symmetry_line.slope = k)
  (h5 : symmetry_line.intercept = 4)
  (h6 : ∃ x₀, pts.A.2 - pts.B.2 = -k * (pts.A.1 - pts.B.1) ∧ 
                pts.A.2 / (pts.A.1 - x₀) = pts.B.2 / (pts.B.1 - x₀)) :
  E.focus = (1, 0) ∧ 
  pts.A.1 + pts.B.1 = 4 ∧ 
  ∃ x₀ : ℝ, -2 < x₀ ∧ x₀ < 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3153_315375


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3153_315395

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 2) (hy : |y| = 3) (hxy : x > y) : x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3153_315395


namespace NUMINAMATH_CALUDE_engine_system_theorems_l3153_315376

/-- Engine connecting rod and crank system -/
structure EngineSystem where
  a : ℝ  -- length of crank OA
  b : ℝ  -- length of connecting rod AP
  α : ℝ  -- angle AOP
  β : ℝ  -- angle APO
  h : 0 < a ∧ 0 < b  -- positive lengths

/-- Theorems about the engine connecting rod and crank system -/
theorem engine_system_theorems (sys : EngineSystem) :
  -- Part 1
  sys.a * Real.sin sys.α = sys.b * Real.sin sys.β ∧
  -- Part 2
  (∀ β', Real.sin β' ≤ sys.a / sys.b) ∧
  -- Part 3
  ∀ x, x = sys.a * (1 - Real.cos sys.α) + sys.b * (1 - Real.cos sys.β) := by
  sorry

end NUMINAMATH_CALUDE_engine_system_theorems_l3153_315376


namespace NUMINAMATH_CALUDE_dividend_calculation_l3153_315301

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 16)
  (h_quotient : quotient = 9)
  (h_remainder : remainder = 5) :
  divisor * quotient + remainder = 149 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3153_315301


namespace NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l3153_315349

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence where the third term is 12 and the fourth term is 18, the second term is 8. -/
theorem second_term_of_geometric_sequence
    (a : ℕ → ℚ)
    (h_geometric : IsGeometricSequence a)
    (h_third_term : a 3 = 12)
    (h_fourth_term : a 4 = 18) :
    a 2 = 8 := by
  sorry

#check second_term_of_geometric_sequence

end NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l3153_315349


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3153_315357

theorem expand_and_simplify (y : ℝ) : 5 * (6 * y^2 - 3 * y + 2) = 30 * y^2 - 15 * y + 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3153_315357


namespace NUMINAMATH_CALUDE_interview_probability_l3153_315313

def total_students : ℕ := 30
def french_students : ℕ := 20
def spanish_students : ℕ := 24

theorem interview_probability :
  let both_classes := french_students + spanish_students - total_students
  let only_french := french_students - both_classes
  let only_spanish := spanish_students - both_classes
  let total_combinations := total_students.choose 2
  let unfavorable_combinations := only_french.choose 2 + only_spanish.choose 2
  (total_combinations - unfavorable_combinations : ℚ) / total_combinations = 25 / 29 := by
  sorry

end NUMINAMATH_CALUDE_interview_probability_l3153_315313


namespace NUMINAMATH_CALUDE_pascal_burger_ratio_l3153_315322

/-- The mass of fats in grams in a Pascal Burger -/
def mass_fats : ℕ := 32

/-- The mass of carbohydrates in grams in a Pascal Burger -/
def mass_carbs : ℕ := 48

/-- The ratio of fats to carbohydrates in a Pascal Burger -/
def fats_to_carbs_ratio : Rat := mass_fats / mass_carbs

theorem pascal_burger_ratio :
  fats_to_carbs_ratio = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_pascal_burger_ratio_l3153_315322


namespace NUMINAMATH_CALUDE_roots_transformation_l3153_315347

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 4*x^2 + 5

-- Define the roots of the original polynomial
def roots_original : Set ℝ := {r | original_poly r = 0}

-- Define the new polynomial
def new_poly (x : ℝ) : ℝ := x^3 - 12*x^2 + 135

-- Define the roots of the new polynomial
def roots_new : Set ℝ := {r | new_poly r = 0}

-- State the theorem
theorem roots_transformation :
  ∃ (r₁ r₂ r₃ : ℝ), roots_original = {r₁, r₂, r₃} →
    roots_new = {3*r₁, 3*r₂, 3*r₃} :=
sorry

end NUMINAMATH_CALUDE_roots_transformation_l3153_315347


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3153_315365

/-- Given a line segment with endpoints (3, 5) and (11, 21), 
    the sum of the coordinates of its midpoint is 20. -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 3
  let y₁ : ℝ := 5
  let x₂ : ℝ := 11
  let y₂ : ℝ := 21
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 20 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3153_315365


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3153_315336

theorem circle_center_and_radius :
  ∀ (x y : ℝ), (x - 1)^2 + (y + 5)^2 = 3 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -5) ∧ radius = Real.sqrt 3 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3153_315336


namespace NUMINAMATH_CALUDE_triangle_with_120_degree_angle_divisible_into_isosceles_l3153_315359

-- Define a triangle type
structure Triangle :=
  (a b c : ℝ)
  (sum_to_180 : a + b + c = 180)
  (all_positive : 0 < a ∧ 0 < b ∧ 0 < c)

-- Define an isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define the property of being divisible into two isosceles triangles
def DivisibleIntoTwoIsosceles (t : Triangle) : Prop :=
  ∃ (t1 t2 : Triangle), IsIsosceles t1 ∧ IsIsosceles t2

-- The main theorem
theorem triangle_with_120_degree_angle_divisible_into_isosceles
  (t : Triangle)
  (has_120_degree : t.a = 120 ∨ t.b = 120 ∨ t.c = 120)
  (divisible : DivisibleIntoTwoIsosceles t) :
  (t.b = 30 ∧ t.c = 15) ∨ (t.b = 45 ∧ t.c = 15) ∨
  (t.b = 15 ∧ t.c = 30) ∨ (t.b = 15 ∧ t.c = 45) :=
by sorry


end NUMINAMATH_CALUDE_triangle_with_120_degree_angle_divisible_into_isosceles_l3153_315359


namespace NUMINAMATH_CALUDE_polygon_with_one_degree_exterior_angles_l3153_315391

/-- The number of sides in a polygon where each exterior angle measures 1 degree -/
def polygon_sides : ℕ := 360

/-- The measure of each exterior angle in degrees -/
def exterior_angle : ℝ := 1

/-- The sum of exterior angles in any polygon in degrees -/
def sum_exterior_angles : ℝ := 360

theorem polygon_with_one_degree_exterior_angles :
  (sum_exterior_angles / exterior_angle : ℝ) = polygon_sides := by sorry

end NUMINAMATH_CALUDE_polygon_with_one_degree_exterior_angles_l3153_315391


namespace NUMINAMATH_CALUDE_circle_symmetry_l3153_315351

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  (x - 1/2)^2 + (y + 1)^2 = 5/4

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x - y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 3/2)^2 = 5/4

-- Theorem statement
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_circle x₁ y₁ →
    symmetric_circle x₂ y₂ →
    ∃ (x y : ℝ),
      symmetry_line x y ∧
      (x₁ + x₂ = 2*x) ∧
      (y₁ + y₂ = 2*y) :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3153_315351


namespace NUMINAMATH_CALUDE_camp_men_count_l3153_315387

/-- The number of days the food lasts initially -/
def initial_days : ℕ := 50

/-- The number of days the food lasts after more men join -/
def final_days : ℕ := 25

/-- The number of additional men who join -/
def additional_men : ℕ := 10

/-- The initial number of men in the camp -/
def initial_men : ℕ := 10

theorem camp_men_count :
  ∀ (food : ℕ),
  food = initial_men * initial_days ∧
  food = (initial_men + additional_men) * final_days →
  initial_men = 10 := by
sorry

end NUMINAMATH_CALUDE_camp_men_count_l3153_315387


namespace NUMINAMATH_CALUDE_ab_inequality_l3153_315334

theorem ab_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b = 2) : a * b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_inequality_l3153_315334


namespace NUMINAMATH_CALUDE_salary_increase_l3153_315317

/-- If a salary increases by 33.33% to $80, prove that the original salary was $60 -/
theorem salary_increase (original : ℝ) (increase_percent : ℝ) (new_salary : ℝ) :
  increase_percent = 33.33 ∧ new_salary = 80 ∧ new_salary = original * (1 + increase_percent / 100) →
  original = 60 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l3153_315317


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l3153_315325

/-- A figure composed of squares arranged in a specific pattern -/
structure SquareFigure where
  squareSideLength : ℝ
  rectangleWidth : ℕ
  rectangleHeight : ℕ
  lShapeOutward : ℕ
  lShapeDownward : ℕ

/-- Calculate the perimeter of the SquareFigure -/
def calculatePerimeter (figure : SquareFigure) : ℝ :=
  let bottomLength := figure.rectangleWidth * figure.squareSideLength
  let topLength := (figure.rectangleWidth + figure.lShapeOutward) * figure.squareSideLength
  let leftHeight := figure.rectangleHeight * figure.squareSideLength
  let rightHeight := (figure.rectangleHeight + figure.lShapeDownward) * figure.squareSideLength
  bottomLength + topLength + leftHeight + rightHeight

/-- Theorem stating that the perimeter of the specific figure is 26 units -/
theorem specific_figure_perimeter :
  let figure : SquareFigure := {
    squareSideLength := 2
    rectangleWidth := 3
    rectangleHeight := 2
    lShapeOutward := 2
    lShapeDownward := 1
  }
  calculatePerimeter figure = 26 := by
  sorry

end NUMINAMATH_CALUDE_specific_figure_perimeter_l3153_315325


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3153_315329

theorem scientific_notation_equality : 3790000 = 3.79 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3153_315329


namespace NUMINAMATH_CALUDE_smallest_number_l3153_315348

-- Define a function to convert a number from any base to decimal
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the numbers in their respective bases
def num1 : List Nat := [8, 5]
def base1 : Nat := 9

def num2 : List Nat := [2, 1, 0]
def base2 : Nat := 6

def num3 : List Nat := [1, 0, 0, 0]
def base3 : Nat := 4

def num4 : List Nat := [1, 1, 1, 1, 1, 1, 1]
def base4 : Nat := 2

-- Theorem statement
theorem smallest_number :
  to_decimal num3 base3 < to_decimal num1 base1 ∧
  to_decimal num3 base3 < to_decimal num2 base2 ∧
  to_decimal num3 base3 < to_decimal num4 base4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3153_315348


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_cube_root_between_8_and_8_1_l3153_315392

theorem unique_integer_divisible_by_18_with_cube_root_between_8_and_8_1 :
  ∃! n : ℕ+, 18 ∣ n ∧ (8 : ℝ) < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < (8.1 : ℝ) ∧ n = 522 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_cube_root_between_8_and_8_1_l3153_315392


namespace NUMINAMATH_CALUDE_chair_price_proof_l3153_315372

/-- The normal price of a chair -/
def normal_price : ℝ := 20

/-- The discounted price for the first 5 chairs -/
def discounted_price_first_5 : ℝ := 0.75 * normal_price

/-- The discounted price for chairs after the first 5 -/
def discounted_price_after_5 : ℝ := 0.5 * normal_price

/-- The number of chairs bought -/
def chairs_bought : ℕ := 8

/-- The total cost of all chairs bought -/
def total_cost : ℝ := 105

theorem chair_price_proof :
  5 * discounted_price_first_5 + (chairs_bought - 5) * discounted_price_after_5 = total_cost :=
sorry

end NUMINAMATH_CALUDE_chair_price_proof_l3153_315372


namespace NUMINAMATH_CALUDE_cookies_remaining_l3153_315341

/-- Represents the baked goods scenario --/
structure BakedGoods where
  cookies : ℕ
  brownies : ℕ
  cookie_price : ℚ
  brownie_price : ℚ

/-- Calculates the total value of baked goods --/
def total_value (bg : BakedGoods) : ℚ :=
  bg.cookies * bg.cookie_price + bg.brownies * bg.brownie_price

/-- Theorem stating the number of cookies remaining --/
theorem cookies_remaining (bg : BakedGoods) 
  (h1 : bg.brownies = 32)
  (h2 : bg.cookie_price = 1)
  (h3 : bg.brownie_price = 3/2)
  (h4 : total_value bg = 99) :
  bg.cookies = 51 := by
sorry


end NUMINAMATH_CALUDE_cookies_remaining_l3153_315341


namespace NUMINAMATH_CALUDE_difference_of_three_times_number_and_five_l3153_315361

theorem difference_of_three_times_number_and_five (x : ℝ) : 3 * x - 5 = 15 → 3 * x - 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_three_times_number_and_five_l3153_315361


namespace NUMINAMATH_CALUDE_functional_polynomial_characterization_l3153_315333

/-- A polynomial that satisfies the given functional equation -/
def FunctionalPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 1 + p x = (p (x - 1) + p (x + 1)) / 2

theorem functional_polynomial_characterization :
  ∀ p : ℝ → ℝ, FunctionalPolynomial p →
  ∃ b c : ℝ, ∀ x : ℝ, p x = x^2 + b*x + c :=
sorry

end NUMINAMATH_CALUDE_functional_polynomial_characterization_l3153_315333


namespace NUMINAMATH_CALUDE_expression_evaluation_l3153_315373

theorem expression_evaluation (a b c : ℚ) (h1 : a = 5) (h2 : b = -3) (h3 : c = 2) :
  3 / (a + b + c) = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3153_315373


namespace NUMINAMATH_CALUDE_square_area_with_circles_8_l3153_315343

/-- The area of a square containing four circles of radius r, with two circles touching each side of the square. -/
def square_area_with_circles (r : ℝ) : ℝ :=
  (4 * r) ^ 2

/-- Theorem: The area of a square containing four circles of radius 8 inches, 
    with two circles touching each side of the square, is 1024 square inches. -/
theorem square_area_with_circles_8 : 
  square_area_with_circles 8 = 1024 :=
by sorry

end NUMINAMATH_CALUDE_square_area_with_circles_8_l3153_315343


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l3153_315367

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l3153_315367


namespace NUMINAMATH_CALUDE_middle_carriages_passengers_l3153_315353

/-- Represents a train with carriages and passengers -/
structure Train where
  num_carriages : Nat
  total_passengers : Nat
  block_passengers : Nat
  block_size : Nat

/-- Calculates the number of passengers in the middle two carriages -/
def middle_two_passengers (t : Train) : Nat :=
  t.total_passengers - (4 * t.block_passengers - 3 * t.total_passengers)

/-- Theorem stating that for a train with given specifications, 
    the middle two carriages contain 96 passengers -/
theorem middle_carriages_passengers 
  (t : Train) 
  (h1 : t.num_carriages = 18) 
  (h2 : t.total_passengers = 700) 
  (h3 : t.block_passengers = 199) 
  (h4 : t.block_size = 5) : 
  middle_two_passengers t = 96 := by
  sorry

end NUMINAMATH_CALUDE_middle_carriages_passengers_l3153_315353


namespace NUMINAMATH_CALUDE_beta_interval_l3153_315383

theorem beta_interval (β : ℝ) : 
  (∃ k : ℤ, β = π/6 + 2*k*π) ∧ -2*π < β ∧ β < 2*π ↔ β = π/6 ∨ β = -11*π/6 := by
  sorry

end NUMINAMATH_CALUDE_beta_interval_l3153_315383


namespace NUMINAMATH_CALUDE_whole_number_between_l3153_315327

theorem whole_number_between : 
  ∀ N : ℕ, (6 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 7) → (N = 25 ∨ N = 26 ∨ N = 27) := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_l3153_315327


namespace NUMINAMATH_CALUDE_distance_between_trees_l3153_315374

/-- Given a yard of length 500 metres with 105 trees planted at equal distances,
    including one at each end, prove that the distance between two consecutive
    trees is 500/104 metres. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) 
    (h1 : yard_length = 500)
    (h2 : num_trees = 105) :
  let num_segments := num_trees - 1
  yard_length / num_segments = 500 / 104 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3153_315374


namespace NUMINAMATH_CALUDE_greatest_marble_difference_is_six_l3153_315300

/-- Represents a basket of marbles -/
structure Basket where
  color1 : String
  count1 : Nat
  color2 : String
  count2 : Nat

/-- Calculates the absolute difference between two natural numbers -/
def absDiff (a b : Nat) : Nat :=
  if a ≥ b then a - b else b - a

/-- Theorem: The greatest difference between marble counts in any basket is 6 -/
theorem greatest_marble_difference_is_six :
  let basketA : Basket := { color1 := "red", count1 := 4, color2 := "yellow", count2 := 2 }
  let basketB : Basket := { color1 := "green", count1 := 6, color2 := "yellow", count2 := 1 }
  let basketC : Basket := { color1 := "white", count1 := 3, color2 := "yellow", count2 := 9 }
  let diffA := absDiff basketA.count1 basketA.count2
  let diffB := absDiff basketB.count1 basketB.count2
  let diffC := absDiff basketC.count1 basketC.count2
  (max diffA (max diffB diffC)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_marble_difference_is_six_l3153_315300


namespace NUMINAMATH_CALUDE_abs_sum_equals_five_l3153_315302

theorem abs_sum_equals_five (a b c : ℝ) 
  (h1 : a^2 - b*c = 14)
  (h2 : b^2 - c*a = 14)
  (h3 : c^2 - a*b = -3) :
  |a + b + c| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_equals_five_l3153_315302
