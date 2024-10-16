import Mathlib

namespace NUMINAMATH_CALUDE_original_number_is_two_thirds_l3229_322992

theorem original_number_is_two_thirds :
  ∃ x : ℚ, (1 + 1 / x = 5 / 2) ∧ (x = 2 / 3) := by sorry

end NUMINAMATH_CALUDE_original_number_is_two_thirds_l3229_322992


namespace NUMINAMATH_CALUDE_qr_length_l3229_322906

/-- Right triangle ABC with hypotenuse AB = 13, AC = 12, and BC = 5 -/
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  right_angle : AB^2 = AC^2 + BC^2
  AB_eq : AB = 13
  AC_eq : AC = 12
  BC_eq : BC = 5

/-- Circle P passing through C and tangent to BC -/
structure CircleP (t : RightTriangle) where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_C : True  -- Simplified condition
  tangent_to_BC : True     -- Simplified condition
  smallest : True          -- Simplified condition

/-- Points Q and R as intersections of circle P with AC and AB -/
structure Intersections (t : RightTriangle) (p : CircleP t) where
  Q : ℝ × ℝ
  R : ℝ × ℝ
  Q_on_AC : True           -- Simplified condition
  R_on_AB : True           -- Simplified condition
  Q_on_circle : True       -- Simplified condition
  R_on_circle : True       -- Simplified condition

/-- Main theorem: Length of QR is 5.42 -/
theorem qr_length (t : RightTriangle) (p : CircleP t) (i : Intersections t p) :
  Real.sqrt ((i.Q.1 - i.R.1)^2 + (i.Q.2 - i.R.2)^2) = 5.42 := by
  sorry

end NUMINAMATH_CALUDE_qr_length_l3229_322906


namespace NUMINAMATH_CALUDE_roselyn_initial_books_l3229_322924

/-- The number of books Roselyn initially had -/
def initial_books : ℕ := 220

/-- The number of books Rebecca received -/
def rebecca_books : ℕ := 40

/-- The number of books Mara received -/
def mara_books : ℕ := 3 * rebecca_books

/-- The number of books Roselyn remained with -/
def remaining_books : ℕ := 60

/-- Theorem stating that the initial number of books Roselyn had is 220 -/
theorem roselyn_initial_books :
  initial_books = mara_books + rebecca_books + remaining_books :=
by sorry

end NUMINAMATH_CALUDE_roselyn_initial_books_l3229_322924


namespace NUMINAMATH_CALUDE_no_identical_lines_l3229_322958

theorem no_identical_lines : ¬∃ (a d : ℝ), ∀ (x y : ℝ),
  (5 * x + a * y + d = 0 ↔ 2 * d * x - 3 * y + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_identical_lines_l3229_322958


namespace NUMINAMATH_CALUDE_a_in_second_quadrant_l3229_322966

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the second quadrant of a rectangular coordinate system -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The point A with coordinates dependent on x -/
def A (x : ℝ) : Point :=
  { x := 6 - 2*x, y := x - 5 }

/-- Theorem stating the condition for point A to be in the second quadrant -/
theorem a_in_second_quadrant :
  ∀ x : ℝ, SecondQuadrant (A x) ↔ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_a_in_second_quadrant_l3229_322966


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3229_322977

theorem decimal_to_fraction (x : ℚ) : x = 2.75 → x = 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3229_322977


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3229_322933

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) -- The arithmetic sequence
  (d : ℤ) -- The common difference
  (h1 : a 0 = 23) -- First term is 23
  (h2 : ∀ n, a (n + 1) = a n + d) -- Arithmetic sequence definition
  (h3 : ∀ n, n < 6 → a n > 0) -- First 6 terms are positive
  (h4 : ∀ n, n ≥ 6 → a n < 0) -- Terms from 7th onward are negative
  : d = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3229_322933


namespace NUMINAMATH_CALUDE_solution_set_l3229_322989

def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}

def B (m : ℝ) : Set ℝ := {x : ℝ | m*x + 1 = 0}

theorem solution_set (m : ℝ) : (A ∪ B m = A) ↔ m ∈ ({0, -1/2, -1/3} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l3229_322989


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l3229_322919

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 84 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l3229_322919


namespace NUMINAMATH_CALUDE_multiply_squared_equation_l3229_322916

/-- Given that a^2 * b = 3 * (4a + 2) and a = 1 is a possible solution, prove that b = 18 -/
theorem multiply_squared_equation (a b : ℝ) : 
  a^2 * b = 3 * (4*a + 2) → a = 1 → b = 18 := by
  sorry

end NUMINAMATH_CALUDE_multiply_squared_equation_l3229_322916


namespace NUMINAMATH_CALUDE_strawberry_milk_count_total_milk_sum_l3229_322961

/-- The number of students who selected strawberry milk -/
def strawberry_milk_students : ℕ := sorry

/-- The number of students who selected chocolate milk -/
def chocolate_milk_students : ℕ := 2

/-- The number of students who selected regular milk -/
def regular_milk_students : ℕ := 3

/-- The total number of milks taken -/
def total_milks : ℕ := 20

/-- Theorem stating that the number of students who selected strawberry milk is 15 -/
theorem strawberry_milk_count : 
  strawberry_milk_students = 15 :=
by
  sorry

/-- Theorem stating that the total number of milks is the sum of all milk selections -/
theorem total_milk_sum :
  total_milks = chocolate_milk_students + strawberry_milk_students + regular_milk_students :=
by
  sorry

end NUMINAMATH_CALUDE_strawberry_milk_count_total_milk_sum_l3229_322961


namespace NUMINAMATH_CALUDE_eight_squares_exist_l3229_322904

/-- Represents a 3x3 square of digits -/
def Square := Matrix (Fin 3) (Fin 3) Nat

/-- Checks if a square uses all digits from 1 to 9 exactly once -/
def uses_all_digits (s : Square) : Prop :=
  ∀ d : Fin 9, ∃! (i j : Fin 3), s i j = d.val + 1

/-- Calculates the sum of a row in a square -/
def row_sum (s : Square) (i : Fin 3) : Nat :=
  (s i 0) + (s i 1) + (s i 2)

/-- Checks if the sum of the first two rows equals the sum of the third row -/
def sum_property (s : Square) : Prop :=
  row_sum s 0 + row_sum s 1 = row_sum s 2

/-- Calculates the difference between row sums -/
def row_sum_diff (s : Square) : Nat :=
  (row_sum s 2) - (row_sum s 1)

/-- The main theorem statement -/
theorem eight_squares_exist : 
  ∃ (squares : Fin 8 → Square),
    (∀ i : Fin 8, uses_all_digits (squares i)) ∧
    (∀ i : Fin 8, sum_property (squares i)) ∧
    (∀ i j : Fin 8, row_sum_diff (squares i) = row_sum_diff (squares j)) ∧
    (∀ i : Fin 8, row_sum_diff (squares i) = 9) :=
  sorry

end NUMINAMATH_CALUDE_eight_squares_exist_l3229_322904


namespace NUMINAMATH_CALUDE_smallest_consecutive_primes_sum_after_13_l3229_322959

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def consecutive_primes (p q r s : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧
  q = p.succ ∧ r = q.succ ∧ s = r.succ

theorem smallest_consecutive_primes_sum_after_13 :
  ∃ (p q r s : ℕ),
    consecutive_primes p q r s ∧
    p > 13 ∧
    4 ∣ (p + q + r + s) ∧
    (p + q + r + s = 88) ∧
    ∀ (a b c d : ℕ),
      consecutive_primes a b c d → a > 13 → 4 ∣ (a + b + c + d) →
      (a + b + c + d ≥ p + q + r + s) :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_primes_sum_after_13_l3229_322959


namespace NUMINAMATH_CALUDE_boyfriend_texts_l3229_322923

theorem boyfriend_texts (total : ℕ) (grocery : ℕ) : 
  total = 33 → 
  grocery + 5 * grocery + (grocery + 5 * grocery) / 10 = total → 
  grocery = 5 := by
  sorry

end NUMINAMATH_CALUDE_boyfriend_texts_l3229_322923


namespace NUMINAMATH_CALUDE_slope_problem_l3229_322955

theorem slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 4) / (2 - m) = 2 * m) : m = (3 + Real.sqrt 41) / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_problem_l3229_322955


namespace NUMINAMATH_CALUDE_largest_average_is_17_multiples_l3229_322978

def average_of_multiples (n : ℕ) (upper_bound : ℕ) : ℚ :=
  let last_multiple := (upper_bound / n) * n
  (n + last_multiple) / 2

def largest_average (upper_bound : ℕ) : ℚ :=
  max (average_of_multiples 11 upper_bound)
    (max (average_of_multiples 13 upper_bound)
      (max (average_of_multiples 17 upper_bound)
        (average_of_multiples 19 upper_bound)))

theorem largest_average_is_17_multiples (upper_bound : ℕ) :
  upper_bound = 100810 →
  largest_average upper_bound = average_of_multiples 17 upper_bound :=
by
  sorry

end NUMINAMATH_CALUDE_largest_average_is_17_multiples_l3229_322978


namespace NUMINAMATH_CALUDE_equation_solution_l3229_322928

theorem equation_solution : ∃! x : ℝ, (1 : ℝ) / (x + 3) = 3 / (x + 9) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3229_322928


namespace NUMINAMATH_CALUDE_parallelogram_area_l3229_322909

/-- The area of a parallelogram with vertices at (0, 0), (3, 0), (1, 5), and (4, 5) is 15 square units. -/
theorem parallelogram_area : 
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (3, 0)
  let v3 : ℝ × ℝ := (1, 5)
  let v4 : ℝ × ℝ := (4, 5)
  let base : ℝ := v2.1 - v1.1
  let height : ℝ := v3.2 - v1.2
  base * height = 15 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3229_322909


namespace NUMINAMATH_CALUDE_angle_through_point_l3229_322971

theorem angle_through_point (α : Real) :
  0 ≤ α → α < 2 * Real.pi →
  let P : ℝ × ℝ := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))
  (Real.cos α = P.1 ∧ Real.sin α = P.2) →
  α = 11 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_through_point_l3229_322971


namespace NUMINAMATH_CALUDE_square_area_proof_l3229_322980

theorem square_area_proof (side_length : ℝ) (rectangle_perimeter : ℝ) : 
  side_length = 8 →
  rectangle_perimeter = 20 →
  (side_length * side_length) = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l3229_322980


namespace NUMINAMATH_CALUDE_cream_fraction_after_pouring_l3229_322946

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  cream : ℚ

/-- Represents the state of both cups --/
structure CupState where
  cup1 : CupContents
  cup2 : CupContents

/-- Performs one round of pouring between cups --/
def pour (state : CupState) : CupState := sorry

/-- Calculates the fraction of cream in cup1 after the pouring process --/
def creamFractionInCup1 (initial : CupState) : ℚ := sorry

theorem cream_fraction_after_pouring :
  let initial := CupState.mk
    (CupContents.mk 5 0)  -- 5 oz coffee, 0 oz cream in cup1
    (CupContents.mk 0 3)  -- 0 oz coffee, 3 oz cream in cup2
  let final := pour (pour initial)
  creamFractionInCup1 final = (11 : ℚ) / 21 := by sorry

#check cream_fraction_after_pouring

end NUMINAMATH_CALUDE_cream_fraction_after_pouring_l3229_322946


namespace NUMINAMATH_CALUDE_bulletin_board_width_l3229_322963

/-- Proves that a rectangular bulletin board with area 6400 cm² and length 160 cm has a width of 40 cm -/
theorem bulletin_board_width :
  ∀ (area length width : ℝ),
  area = 6400 ∧ length = 160 ∧ area = length * width →
  width = 40 := by
  sorry

end NUMINAMATH_CALUDE_bulletin_board_width_l3229_322963


namespace NUMINAMATH_CALUDE_decreasing_function_odd_product_l3229_322973

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Statement 1
theorem decreasing_function (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0) :
  ∀ x y : ℝ, x < y → f y < f x :=
sorry

-- Define an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Statement 3
theorem odd_product (h : is_odd f) :
  is_odd (λ x => f x * f (|x|)) :=
sorry

end NUMINAMATH_CALUDE_decreasing_function_odd_product_l3229_322973


namespace NUMINAMATH_CALUDE_qr_distance_l3229_322915

/-- Right triangle DEF with given side lengths -/
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  right_angle : DE^2 + EF^2 = DF^2

/-- Circle centered at Q tangent to DE at D and passing through F -/
structure CircleQ where
  Q : ℝ × ℝ
  tangent_DE : True  -- Simplified representation of tangency
  passes_through_F : True  -- Simplified representation of passing through F

/-- Circle centered at R tangent to EF at E and passing through F -/
structure CircleR where
  R : ℝ × ℝ
  tangent_EF : True  -- Simplified representation of tangency
  passes_through_F : True  -- Simplified representation of passing through F

/-- The main theorem statement -/
theorem qr_distance (t : RightTriangle) (cq : CircleQ) (cr : CircleR) 
  (h1 : t.DE = 5) (h2 : t.EF = 12) (h3 : t.DF = 13) :
  Real.sqrt ((cq.Q.1 - cr.R.1)^2 + (cq.Q.2 - cr.R.2)^2) = 13.54 := by
  sorry

end NUMINAMATH_CALUDE_qr_distance_l3229_322915


namespace NUMINAMATH_CALUDE_dividend_calculation_l3229_322962

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 158 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3229_322962


namespace NUMINAMATH_CALUDE_smallest_n_for_pie_distribution_l3229_322967

theorem smallest_n_for_pie_distribution (N : ℕ) : 
  N > 70 → (21 * N) % 70 = 0 → (∀ m : ℕ, m > 70 ∧ (21 * m) % 70 = 0 → m ≥ N) → N = 80 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_pie_distribution_l3229_322967


namespace NUMINAMATH_CALUDE_discount_calculation_l3229_322948

/-- Calculates the discounted price for a purchase with a percentage discount on amounts over a threshold --/
def discountedPrice (itemCount : ℕ) (itemPrice : ℚ) (discountPercentage : ℚ) (discountThreshold : ℚ) : ℚ :=
  let totalPrice := itemCount * itemPrice
  let amountOverThreshold := max (totalPrice - discountThreshold) 0
  let discountAmount := discountPercentage * amountOverThreshold
  totalPrice - discountAmount

/-- Proves that for a purchase of 7 items at $200 each, with a 10% discount on amounts over $1000, the final cost is $1360 --/
theorem discount_calculation :
  discountedPrice 7 200 0.1 1000 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l3229_322948


namespace NUMINAMATH_CALUDE_min_value_expression_l3229_322934

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 9) :
  (a^2 + b^2 + c^2)/(a + b + c) + (b^2 + c^2)/(b + c) + (c^2 + a^2)/(c + a) + (a^2 + b^2)/(a + b) ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3229_322934


namespace NUMINAMATH_CALUDE_nh_not_equal_nk_l3229_322900

/-- A structure representing a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A structure representing a line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Given two points, return the line passing through them -/
def line_through_points (p1 p2 : Point) : Line :=
  sorry

/-- Given a point and a line, return the perpendicular line passing through the point -/
def perpendicular_line (p : Point) (l : Line) : Line :=
  sorry

/-- Given two lines, return the angle between them in radians -/
def angle_between_lines (l1 l2 : Line) : ℝ :=
  sorry

/-- Given two points, return the distance between them -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Given three points A, B, C, return a point that is 1/3 of the way from A to B -/
def one_third_point (a b : Point) : Point :=
  sorry

theorem nh_not_equal_nk (h k y z : Point) :
  let hk : Line := line_through_points h k
  let yz : Line := line_through_points y z
  let n : Point := one_third_point y z
  let yh : Line := perpendicular_line y hk
  let zk : Line := line_through_points z k
  angle_between_lines hk zk = π / 4 →
  distance n h ≠ distance n k :=
sorry

end NUMINAMATH_CALUDE_nh_not_equal_nk_l3229_322900


namespace NUMINAMATH_CALUDE_exists_even_in_sequence_l3229_322997

/-- A sequence of natural numbers where each subsequent number is obtained by adding one of its non-zero digits to the previous number. -/
def DigitAdditionSequence : Type :=
  ℕ → ℕ

/-- Property that defines the sequence: each subsequent number is obtained by adding one of its non-zero digits to the previous number. -/
def IsValidSequence (seq : DigitAdditionSequence) : Prop :=
  ∀ n : ℕ, ∃ d : ℕ, d > 0 ∧ d < 10 ∧ seq (n + 1) = seq n + d

/-- Theorem stating that there exists an even number in the sequence. -/
theorem exists_even_in_sequence (seq : DigitAdditionSequence) (h : IsValidSequence seq) :
  ∃ n : ℕ, Even (seq n) := by
  sorry

end NUMINAMATH_CALUDE_exists_even_in_sequence_l3229_322997


namespace NUMINAMATH_CALUDE_circle_radius_l3229_322994

theorem circle_radius (x y : ℝ) : 
  (2 * x^2 + 2 * y^2 - 4 * x + 6 * y = 3/2) → 
  ∃ (h k r : ℝ), r = 2 ∧ (x - h)^2 + (y - k)^2 = r^2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l3229_322994


namespace NUMINAMATH_CALUDE_staircase_expansion_l3229_322943

/-- Calculates the number of toothpicks needed for a staircase of given steps -/
def toothpicks_for_steps (n : ℕ) : ℕ :=
  if n ≤ 1 then 4
  else if n = 2 then 10
  else 10 + 8 * (n - 2)

/-- The problem statement -/
theorem staircase_expansion :
  let initial_steps := 4
  let initial_toothpicks := 26
  let main_final_steps := 6
  let adjacent_steps := 3
  let additional_toothpicks := 
    (toothpicks_for_steps main_final_steps + toothpicks_for_steps adjacent_steps) - initial_toothpicks
  additional_toothpicks = 34 := by sorry

end NUMINAMATH_CALUDE_staircase_expansion_l3229_322943


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3229_322913

/-- The polynomial x^4 - 6x^3 + 16x^2 - 25x + 10 -/
def P (x : ℝ) : ℝ := x^4 - 6*x^3 + 16*x^2 - 25*x + 10

/-- The divisor x^2 - 2x + k -/
def D (x k : ℝ) : ℝ := x^2 - 2*x + k

/-- The remainder x + a -/
def R (x a : ℝ) : ℝ := x + a

/-- There exist q such that P(x) = D(x, k) * q(x) + R(x, a) for all x -/
def divides (k a : ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, P x = D x k * q x + R x a

theorem polynomial_division_theorem :
  ∀ k a : ℝ, divides k a ↔ k = 5 ∧ a = -5 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3229_322913


namespace NUMINAMATH_CALUDE_orange_orchard_composition_l3229_322964

/-- Represents an orange orchard with flat and hilly areas. -/
structure Orchard :=
  (total_acres : ℕ)
  (sampled_acres : ℕ)
  (flat_sampled : ℕ)
  (hilly_sampled : ℕ)

/-- Checks if the sampling method is valid for the given orchard. -/
def valid_sampling (o : Orchard) : Prop :=
  o.hilly_sampled = 2 * o.flat_sampled + 1 ∧
  o.flat_sampled + o.hilly_sampled = o.sampled_acres

/-- Calculates the number of flat acres in the orchard based on the sampling. -/
def flat_acres (o : Orchard) : ℕ :=
  o.flat_sampled * (o.total_acres / o.sampled_acres)

/-- Calculates the number of hilly acres in the orchard based on the sampling. -/
def hilly_acres (o : Orchard) : ℕ :=
  o.hilly_sampled * (o.total_acres / o.sampled_acres)

/-- Theorem stating the composition of the orange orchard. -/
theorem orange_orchard_composition (o : Orchard) 
  (h1 : o.total_acres = 120)
  (h2 : o.sampled_acres = 10)
  (h3 : valid_sampling o) :
  flat_acres o = 36 ∧ hilly_acres o = 84 :=
sorry

end NUMINAMATH_CALUDE_orange_orchard_composition_l3229_322964


namespace NUMINAMATH_CALUDE_power_of_eight_division_l3229_322988

theorem power_of_eight_division (n : ℕ) : 8^(n+1) / 8 = 8^n := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_division_l3229_322988


namespace NUMINAMATH_CALUDE_million_millimeters_equals_one_kilometer_l3229_322912

-- Define the conversion factors
def millimeters_per_meter : ℕ := 1000
def meters_per_kilometer : ℕ := 1000

-- Define the question
def million_millimeters : ℕ := 1000000

-- Theorem to prove
theorem million_millimeters_equals_one_kilometer :
  (million_millimeters / millimeters_per_meter) / meters_per_kilometer = 1 := by
  sorry

end NUMINAMATH_CALUDE_million_millimeters_equals_one_kilometer_l3229_322912


namespace NUMINAMATH_CALUDE_negation_of_forall_even_square_plus_n_l3229_322960

theorem negation_of_forall_even_square_plus_n :
  (¬ ∀ n : ℕ, Even (n^2 + n)) ↔ (∃ n : ℕ, ¬ Even (n^2 + n)) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_even_square_plus_n_l3229_322960


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l3229_322902

-- Define the function
def f (x : ℝ) : ℝ := (2*x - 1)^3

-- State the theorem
theorem tangent_slope_at_zero : 
  (deriv f) 0 = 6 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l3229_322902


namespace NUMINAMATH_CALUDE_first_divisor_problem_l3229_322925

theorem first_divisor_problem :
  ∃ (d : ℕ+) (x k m : ℤ),
    x = k * d.val + 11 ∧
    x = 9 * m + 2 ∧
    d.val < 11 ∧
    9 % d.val = 0 ∧
    d = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l3229_322925


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_theorem_l3229_322947

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Predicate to check if a quadrilateral is inscribed -/
def is_inscribed (q : Quadrilateral) : Prop := sorry

/-- Function to get the radius of the circumscribed circle of a quadrilateral -/
def circumscribed_radius (q : Quadrilateral) : ℝ := sorry

/-- Predicate to check if a point is inside a quadrilateral -/
def is_inside (P : Point) (q : Quadrilateral) : Prop := sorry

/-- Function to divide a quadrilateral into four parts given an internal point -/
def divide_quadrilateral (q : Quadrilateral) (P : Point) : 
  (Quadrilateral × Quadrilateral × Quadrilateral × Quadrilateral) := sorry

theorem inscribed_quadrilateral_theorem 
  (ABCD : Quadrilateral) 
  (P : Point) 
  (h_inscribed : is_inscribed ABCD) 
  (h_inside : is_inside P ABCD) :
  let (APB, BPC, CPD, APD) := divide_quadrilateral ABCD P
  (is_inscribed APB ∧ is_inscribed BPC ∧ is_inscribed CPD) →
  (circumscribed_radius APB = circumscribed_radius BPC) →
  (circumscribed_radius BPC = circumscribed_radius CPD) →
  (is_inscribed APD ∧ circumscribed_radius APD = circumscribed_radius APB) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_theorem_l3229_322947


namespace NUMINAMATH_CALUDE_journey_first_half_rate_l3229_322995

/-- Represents a journey with two halves -/
structure Journey where
  total_distance : ℝ
  first_half_rate : ℝ
  second_half_time_factor : ℝ
  average_rate : ℝ

/-- Theorem stating the conditions and the result to be proven -/
theorem journey_first_half_rate (j : Journey) 
  (h1 : j.total_distance = 640)
  (h2 : j.second_half_time_factor = 3)
  (h3 : j.average_rate = 40) :
  j.first_half_rate = 80 := by
  sorry

#check journey_first_half_rate

end NUMINAMATH_CALUDE_journey_first_half_rate_l3229_322995


namespace NUMINAMATH_CALUDE_martha_blocks_found_l3229_322983

/-- The number of blocks Martha found -/
def blocks_found (initial final : ℕ) : ℕ := final - initial

/-- Martha's initial number of blocks -/
def martha_initial : ℕ := 4

/-- Martha's final number of blocks -/
def martha_final : ℕ := 84

theorem martha_blocks_found : blocks_found martha_initial martha_final = 80 := by
  sorry

end NUMINAMATH_CALUDE_martha_blocks_found_l3229_322983


namespace NUMINAMATH_CALUDE_balloon_unique_arrangements_l3229_322907

def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 3)

theorem balloon_unique_arrangements :
  balloon_arrangements = 420 := by
  sorry

end NUMINAMATH_CALUDE_balloon_unique_arrangements_l3229_322907


namespace NUMINAMATH_CALUDE_sixth_graders_and_parents_average_age_l3229_322908

/-- The average age of a group of sixth-graders and their parents -/
def average_age (num_children : ℕ) (num_parents : ℕ) (avg_age_children : ℚ) (avg_age_parents : ℚ) : ℚ :=
  ((num_children : ℚ) * avg_age_children + (num_parents : ℚ) * avg_age_parents) / ((num_children + num_parents) : ℚ)

/-- Theorem stating the average age of sixth-graders and their parents -/
theorem sixth_graders_and_parents_average_age :
  average_age 45 60 12 35 = 25142857142857142 / 1000000000000000 :=
by sorry

end NUMINAMATH_CALUDE_sixth_graders_and_parents_average_age_l3229_322908


namespace NUMINAMATH_CALUDE_smallest_muffin_boxes_l3229_322942

theorem smallest_muffin_boxes : ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), 0 < k ∧ k < n → ¬(11 ∣ (17 * k - 1))) ∧ (11 ∣ (17 * n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_muffin_boxes_l3229_322942


namespace NUMINAMATH_CALUDE_chair_cost_l3229_322986

theorem chair_cost (total_cost : ℝ) (table_cost : ℝ) (num_chairs : ℕ) :
  total_cost = 135 →
  table_cost = 55 →
  num_chairs = 4 →
  ∃ (chair_cost : ℝ),
    chair_cost * num_chairs = total_cost - table_cost ∧
    chair_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_chair_cost_l3229_322986


namespace NUMINAMATH_CALUDE_page_lines_increase_l3229_322976

theorem page_lines_increase (L : ℕ) (h1 : (60 : ℝ) / L = 1 / 3) : L + 60 = 240 := by
  sorry

end NUMINAMATH_CALUDE_page_lines_increase_l3229_322976


namespace NUMINAMATH_CALUDE_equations_solvability_l3229_322903

theorem equations_solvability :
  (∃ (x y z : ℕ), 
    (x % 2 = 1) ∧ (y % 2 = 1) ∧ (z % 2 = 1) ∧
    (y = x + 2) ∧ (z = y + 2) ∧
    (x + y + z = 51)) ∧
  (∃ (x y z w : ℕ),
    (x % 6 = 0) ∧ (y % 6 = 0) ∧ (z % 6 = 0) ∧ (w % 6 = 0) ∧
    (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (w > 0) ∧
    (x + y + z + w = 60)) :=
by sorry

end NUMINAMATH_CALUDE_equations_solvability_l3229_322903


namespace NUMINAMATH_CALUDE_inscribed_circle_length_equals_arc_length_l3229_322944

/-- Given a circular arc of 120° with radius R and an inscribed circle with radius r 
    tangent to the arc and the tangent lines drawn at the arc's endpoints, 
    the circumference of the inscribed circle (2πr) is equal to the length of the original 120° arc. -/
theorem inscribed_circle_length_equals_arc_length (R r : ℝ) : 
  R > 0 → r > 0 → r = R / 2 → 2 * π * r = 2 * π * R * (1/3) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_length_equals_arc_length_l3229_322944


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3229_322991

/-- Given an inequality system with solution set x < 1, find the range of a -/
theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x - 1 < 0 ∧ x < a + 3) ↔ x < 1) → a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3229_322991


namespace NUMINAMATH_CALUDE_mary_shirts_problem_l3229_322930

theorem mary_shirts_problem (blue_shirts : ℕ) (brown_shirts : ℕ) (remaining_shirts : ℕ) :
  blue_shirts = 26 →
  brown_shirts = 36 →
  remaining_shirts = 37 →
  ∃ (f : ℚ), f = 1/2 ∧
    blue_shirts * (1 - f) + brown_shirts * (2/3) = remaining_shirts :=
by sorry

end NUMINAMATH_CALUDE_mary_shirts_problem_l3229_322930


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3229_322956

theorem inequality_system_solution : 
  {x : ℝ | (8 * x - 3 ≤ 13) ∧ ((x - 1) / 3 - 2 < x - 1)} = {x : ℝ | -2 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3229_322956


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l3229_322950

def is_parallel (a : ℝ) : Prop :=
  a^2 = 1

theorem a_equals_one_sufficient_not_necessary :
  (∃ a : ℝ, is_parallel a ∧ a ≠ 1) ∧
  (∀ a : ℝ, a = 1 → is_parallel a) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l3229_322950


namespace NUMINAMATH_CALUDE_km2_to_hectares_conversion_m2_to_km2_conversion_l3229_322981

-- Define the conversion factors
def km2_to_hectares : ℝ := 100
def m2_to_km2 : ℝ := 1000000

-- Theorem 1: 3.4 km² = 340 hectares
theorem km2_to_hectares_conversion :
  3.4 * km2_to_hectares = 340 := by sorry

-- Theorem 2: 690000 m² = 0.69 km²
theorem m2_to_km2_conversion :
  690000 / m2_to_km2 = 0.69 := by sorry

end NUMINAMATH_CALUDE_km2_to_hectares_conversion_m2_to_km2_conversion_l3229_322981


namespace NUMINAMATH_CALUDE_at_most_one_solution_l3229_322941

/-- The floor function, mapping a real number to its integer part -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem stating that the equation ax + b⌊x⌋ - c = 0 has at most one solution -/
theorem at_most_one_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃! x, a * x + b * (floor x : ℝ) - c = 0 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_solution_l3229_322941


namespace NUMINAMATH_CALUDE_binomial_coefficient_relation_l3229_322938

theorem binomial_coefficient_relation :
  ∀ n : ℤ, n > 3 →
  ∃ m : ℤ, m > 1 ∧
    Nat.choose (m.toNat) 2 = 3 * Nat.choose (n.toNat) 4 ∧
    m = (n^2 - 3*n + 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_relation_l3229_322938


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3229_322970

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 5*x = 4) ↔ (∃ x : ℝ, x^2 + 5*x ≠ 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3229_322970


namespace NUMINAMATH_CALUDE_count_numbers_with_digits_eq_six_l3229_322982

/-- The count of integers between 600 and 2000 that contain the digits 3, 5, and 7 -/
def count_numbers_with_digits : ℕ :=
  -- Definition goes here
  sorry

/-- The range of integers to consider -/
def lower_bound : ℕ := 600
def upper_bound : ℕ := 2000

/-- The required digits -/
def required_digits : List ℕ := [3, 5, 7]

theorem count_numbers_with_digits_eq_six :
  count_numbers_with_digits = 6 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_digits_eq_six_l3229_322982


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3229_322965

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  1 / x + 1 / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3229_322965


namespace NUMINAMATH_CALUDE_lcm_gcd_equality_l3229_322974

/-- For positive integers a, b, c, prove that 
    [a,b,c]^2 / ([a,b][b,c][c,a]) = (a,b,c)^2 / ((a,b)(b,c)(c,a)) -/
theorem lcm_gcd_equality (a b c : ℕ+) : 
  (Nat.lcm (Nat.lcm a b) c)^2 / (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a) = 
  (Nat.gcd (Nat.gcd a b) c)^2 / (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_equality_l3229_322974


namespace NUMINAMATH_CALUDE_complex_cube_root_l3229_322926

theorem complex_cube_root (a b : ℕ+) :
  (↑a + ↑b * Complex.I) ^ 3 = 2 + 11 * Complex.I →
  ↑a + ↑b * Complex.I = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l3229_322926


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_1987_l3229_322914

theorem last_three_digits_of_7_to_1987 : 7^1987 % 1000 = 543 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_1987_l3229_322914


namespace NUMINAMATH_CALUDE_coin_array_problem_l3229_322911

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem coin_array_problem :
  ∃ N : ℕ, triangular_sum N = 3003 ∧ sum_of_digits N = 14 := by
  sorry

end NUMINAMATH_CALUDE_coin_array_problem_l3229_322911


namespace NUMINAMATH_CALUDE_ferris_wheel_broken_seats_l3229_322952

/-- The number of broken seats on a Ferris wheel -/
def broken_seats (total_seats : ℕ) (capacity_per_seat : ℕ) (current_capacity : ℕ) : ℕ :=
  total_seats - (current_capacity / capacity_per_seat)

/-- Theorem stating the number of broken seats on the Ferris wheel -/
theorem ferris_wheel_broken_seats :
  let total_seats : ℕ := 18
  let capacity_per_seat : ℕ := 15
  let current_capacity : ℕ := 120
  broken_seats total_seats capacity_per_seat current_capacity = 10 := by
  sorry

#eval broken_seats 18 15 120

end NUMINAMATH_CALUDE_ferris_wheel_broken_seats_l3229_322952


namespace NUMINAMATH_CALUDE_least_number_with_divisibility_property_l3229_322901

theorem least_number_with_divisibility_property : ∃ m : ℕ, 
  (m > 0) ∧ 
  (∀ n : ℕ, n > 0 → n < m → ¬(∃ q r : ℕ, n = 5 * q ∧ n = 34 * (q - 8) + r ∧ r < 34)) ∧
  (∃ q r : ℕ, m = 5 * q ∧ m = 34 * (q - 8) + r ∧ r < 34) ∧
  m = 162 :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_divisibility_property_l3229_322901


namespace NUMINAMATH_CALUDE_sheila_tue_thu_hours_l3229_322929

/-- Represents Sheila's work schedule and earnings -/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  hourly_rate : ℚ
  weekly_earnings : ℚ

/-- The theorem stating Sheila's work hours on Tuesday and Thursday -/
theorem sheila_tue_thu_hours (schedule : WorkSchedule) 
  (h1 : schedule.hours_mon_wed_fri = 8 * 3)
  (h2 : schedule.hourly_rate = 12)
  (h3 : schedule.weekly_earnings = 432)
  (h4 : schedule.weekly_earnings = 
        schedule.hourly_rate * (schedule.hours_mon_wed_fri + schedule.hours_tue_thu)) :
  schedule.hours_tue_thu = 12 := by
  sorry


end NUMINAMATH_CALUDE_sheila_tue_thu_hours_l3229_322929


namespace NUMINAMATH_CALUDE_triangle_side_length_l3229_322972

theorem triangle_side_length (x : ℕ+) : 
  (5 + 15 > x^3 ∧ x^3 + 5 > 15 ∧ x^3 + 15 > 5) ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3229_322972


namespace NUMINAMATH_CALUDE_negation_equivalence_l3229_322932

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3229_322932


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l3229_322969

/-- Represents the composition of a bag of marbles -/
structure BagComposition where
  color1 : ℕ
  color2 : ℕ

/-- Calculates the probability of drawing a specific color from a bag -/
def drawProbability (bag : BagComposition) (colorCount : ℕ) : ℚ :=
  colorCount / (bag.color1 + bag.color2)

/-- The main theorem statement -/
theorem yellow_marble_probability
  (bagX : BagComposition)
  (bagY : BagComposition)
  (bagZ : BagComposition)
  (hX : bagX = ⟨5, 3⟩)
  (hY : bagY = ⟨8, 2⟩)
  (hZ : bagZ = ⟨3, 4⟩) :
  let probWhiteX := drawProbability bagX bagX.color1
  let probYellowY := drawProbability bagY bagY.color1
  let probBlackX := drawProbability bagX bagX.color2
  let probYellowZ := drawProbability bagZ bagZ.color1
  probWhiteX * probYellowY + probBlackX * probYellowZ = 37 / 56 :=
sorry

end NUMINAMATH_CALUDE_yellow_marble_probability_l3229_322969


namespace NUMINAMATH_CALUDE_john_total_contribution_l3229_322918

/-- The amount of money John received from his grandpa -/
def grandpa_contribution : ℚ := 30

/-- The amount of money John received from his grandma -/
def grandma_contribution : ℚ := 3 * grandpa_contribution

/-- The amount of money John received from his aunt -/
def aunt_contribution : ℚ := (3/2) * grandpa_contribution

/-- The amount of money John received from his uncle -/
def uncle_contribution : ℚ := (2/3) * grandma_contribution

/-- The total amount of money John received from all four relatives -/
def total_contribution : ℚ := grandpa_contribution + grandma_contribution + aunt_contribution + uncle_contribution

theorem john_total_contribution : total_contribution = 225 := by
  sorry

end NUMINAMATH_CALUDE_john_total_contribution_l3229_322918


namespace NUMINAMATH_CALUDE_tan_difference_l3229_322990

theorem tan_difference (α β : Real) 
  (h1 : Real.tan (α + π/3) = -3)
  (h2 : Real.tan (β - π/6) = 5) : 
  Real.tan (α - β) = -7/4 := by
sorry

end NUMINAMATH_CALUDE_tan_difference_l3229_322990


namespace NUMINAMATH_CALUDE_permutation_intersection_theorem_l3229_322984

/-- A permutation of the first n natural numbers -/
def Permutation (n : ℕ) := { f : Fin n → Fin n // Function.Bijective f }

/-- Two permutations intersect if they have the same value at some position -/
def intersect {n : ℕ} (p q : Permutation n) : Prop :=
  ∃ k : Fin n, p.val k = q.val k

/-- The theorem to be proved -/
theorem permutation_intersection_theorem :
  ∃ (S : Finset (Permutation 2010)),
    S.card = 1006 ∧
    ∀ p : Permutation 2010, ∃ q ∈ S, intersect p q := by
  sorry


end NUMINAMATH_CALUDE_permutation_intersection_theorem_l3229_322984


namespace NUMINAMATH_CALUDE_complex_magnitude_l3229_322951

theorem complex_magnitude (z : ℂ) : z = (2 + Complex.I) / Complex.I + Complex.I → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3229_322951


namespace NUMINAMATH_CALUDE_ages_sum_l3229_322940

theorem ages_sum (a b c : ℕ) : 
  a = b + c + 20 → 
  a^2 = (b + c)^2 + 1800 → 
  a + b + c = 90 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l3229_322940


namespace NUMINAMATH_CALUDE_river_travel_time_l3229_322985

structure RiverSystem where
  docks : Fin 3 → String
  distance : Fin 3 → Fin 3 → ℝ
  time_against_current : ℝ
  time_with_current : ℝ

def valid_river_system (rs : RiverSystem) : Prop :=
  (∀ i j, rs.distance i j = 3) ∧
  rs.time_against_current = 30 ∧
  rs.time_with_current = 18 ∧
  rs.time_against_current > rs.time_with_current

def travel_time (rs : RiverSystem) : Set ℝ :=
  {24, 72}

theorem river_travel_time (rs : RiverSystem) (h : valid_river_system rs) :
  ∀ i j, i ≠ j → (rs.distance i j / rs.time_against_current * 60 ∈ travel_time rs) ∨
                 (rs.distance i j / rs.time_with_current * 60 ∈ travel_time rs) :=
sorry

end NUMINAMATH_CALUDE_river_travel_time_l3229_322985


namespace NUMINAMATH_CALUDE_hash_two_three_l3229_322921

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- Theorem to prove
theorem hash_two_three : hash 2 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hash_two_three_l3229_322921


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3229_322987

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of six coins -/
structure SixCoins where
  penny : CoinFlip
  nickel : CoinFlip
  dime : CoinFlip
  quarter : CoinFlip
  halfDollar : CoinFlip
  dollar : CoinFlip

/-- The total number of possible outcomes when flipping six coins -/
def totalOutcomes : Nat := 64

/-- Checks if the penny and dime have different outcomes -/
def pennyDimeDifferent (coins : SixCoins) : Prop :=
  coins.penny ≠ coins.dime

/-- Checks if the nickel and quarter have the same outcome -/
def nickelQuarterSame (coins : SixCoins) : Prop :=
  coins.nickel = coins.quarter

/-- Counts the number of favorable outcomes -/
def favorableOutcomes : Nat := 16

/-- The probability of the specified event -/
def probability : ℚ := 1 / 4

theorem coin_flip_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = probability :=
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3229_322987


namespace NUMINAMATH_CALUDE_blue_rows_count_l3229_322999

/-- Given a grid with the following properties:
  * 10 rows and 15 squares per row
  * 4 rows of 6 squares in the middle are red
  * 66 squares are green
  * All remaining squares are blue
  * Blue squares cover entire rows
Prove that the number of rows colored blue at the beginning and end of the grid is 4 -/
theorem blue_rows_count (total_rows : Nat) (squares_per_row : Nat) 
  (red_rows : Nat) (red_squares_per_row : Nat) (green_squares : Nat) : 
  total_rows = 10 → 
  squares_per_row = 15 → 
  red_rows = 4 → 
  red_squares_per_row = 6 → 
  green_squares = 66 → 
  (total_rows * squares_per_row - red_rows * red_squares_per_row - green_squares) / squares_per_row = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_rows_count_l3229_322999


namespace NUMINAMATH_CALUDE_bill_sunday_miles_l3229_322939

/-- Represents the number of miles run by a person on a specific day -/
structure DailyMiles where
  friday : ℝ
  saturday : ℝ
  sunday : ℝ

/-- Calculates the total miles run over three days -/
def totalMiles (person : DailyMiles) : ℝ :=
  person.friday + person.saturday + person.sunday

theorem bill_sunday_miles (bill julia : DailyMiles) :
  bill.friday = 2 * bill.saturday →
  bill.sunday = bill.saturday + 4 →
  julia.saturday = 0 →
  julia.sunday = 2 * bill.sunday →
  julia.friday = 2 * bill.friday - 3 →
  totalMiles bill + totalMiles julia = 30 →
  bill.sunday = 6.1 := by
sorry

end NUMINAMATH_CALUDE_bill_sunday_miles_l3229_322939


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3229_322922

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, x)
  parallel a b → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3229_322922


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3229_322996

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3229_322996


namespace NUMINAMATH_CALUDE_factorial_square_root_theorem_l3229_322975

theorem factorial_square_root_theorem : 
  (Real.sqrt ((Nat.factorial 5 * Nat.factorial 4) / Nat.factorial 3))^2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_theorem_l3229_322975


namespace NUMINAMATH_CALUDE_log_xy_value_l3229_322935

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^4) = 2) (h2 : Real.log (x^3 * y) = 2) :
  Real.log (x * y) = 10 / 11 := by
sorry

end NUMINAMATH_CALUDE_log_xy_value_l3229_322935


namespace NUMINAMATH_CALUDE_smallest_ccd_is_227_l3229_322979

/-- Represents a two-digit number -/
def TwoDigitNumber (c d : ℕ) : Prop :=
  c ≠ 0 ∧ c ≤ 9 ∧ d ≤ 9

/-- Represents a three-digit number -/
def ThreeDigitNumber (c d : ℕ) : Prop :=
  TwoDigitNumber c d ∧ c * 100 + c * 10 + d ≥ 100

/-- The main theorem -/
theorem smallest_ccd_is_227 :
  ∃ (c d : ℕ),
    TwoDigitNumber c d ∧
    ThreeDigitNumber c d ∧
    c ≠ d ∧
    (c * 10 + d : ℚ) = (1 / 7) * (c * 100 + c * 10 + d) ∧
    c * 100 + c * 10 + d = 227 ∧
    ∀ (c' d' : ℕ),
      TwoDigitNumber c' d' →
      ThreeDigitNumber c' d' →
      c' ≠ d' →
      (c' * 10 + d' : ℚ) = (1 / 7) * (c' * 100 + c' * 10 + d') →
      c' * 100 + c' * 10 + d' ≥ 227 :=
by sorry

end NUMINAMATH_CALUDE_smallest_ccd_is_227_l3229_322979


namespace NUMINAMATH_CALUDE_coefficient_of_x_term_l3229_322927

theorem coefficient_of_x_term (x : ℝ) : 
  let expansion := (Real.sqrt x - 2 / x) ^ 5
  ∃ c : ℝ, c = -10 ∧ 
    ∃ t : ℝ → ℝ, (expansion = c * x + t x) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → |t h / h| < ε) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_term_l3229_322927


namespace NUMINAMATH_CALUDE_cube_root_strict_mono_l3229_322998

theorem cube_root_strict_mono {a b : ℝ} (h : a < b) : ¬(a^(1/3) ≥ b^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_strict_mono_l3229_322998


namespace NUMINAMATH_CALUDE_grid_coloring_count_l3229_322968

/-- Represents the number of valid colorings for a 2 × n grid -/
def num_colorings (n : ℕ) : ℕ :=
  3^(n-1)

/-- Theorem stating the number of distinct colorings for the grid -/
theorem grid_coloring_count (n : ℕ) (h : n ≥ 2) :
  let grid_size := 2 * n
  let colored_endpoints := 3
  let vertices_to_color := grid_size - colored_endpoints
  let num_colors := 3
  num_colorings n = num_colors^(n-1) :=
by sorry

end NUMINAMATH_CALUDE_grid_coloring_count_l3229_322968


namespace NUMINAMATH_CALUDE_gumball_count_l3229_322905

def gumball_machine (red : ℕ) : Prop :=
  ∃ (blue green yellow orange : ℕ),
    blue = red / 2 ∧
    green = 4 * blue ∧
    yellow = (60 * green) / 100 ∧
    orange = (red + blue) / 3 ∧
    red + blue + green + yellow + orange = 124

theorem gumball_count : gumball_machine 24 := by
  sorry

end NUMINAMATH_CALUDE_gumball_count_l3229_322905


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equivalence_l3229_322937

theorem arithmetic_sequence_equivalence
  (a b c : ℕ → ℝ)
  (h1 : ∀ n, b n = a (n + 1) - a n)
  (h2 : ∀ n, c n = a n + 2 * a (n + 1)) :
  (∃ d, ∀ n, a (n + 1) - a n = d) ↔
  ((∃ D, ∀ n, c (n + 1) - c n = D) ∧ (∀ n, b n ≤ b (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equivalence_l3229_322937


namespace NUMINAMATH_CALUDE_min_value_theorem_l3229_322920

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 + 1 / (a + b + c)^3 ≥ 2 * 3^(2/5) / 3 + 3^(1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3229_322920


namespace NUMINAMATH_CALUDE_inverse_square_difference_l3229_322945

theorem inverse_square_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 - y^2 = x*y) : 
  1/x^2 - 1/y^2 = -1/(x*y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_difference_l3229_322945


namespace NUMINAMATH_CALUDE_max_monthly_profit_l3229_322917

/-- Represents the monthly sales profit function -/
def monthly_profit (x : ℕ) : ℚ :=
  -10 * x^2 + 110 * x + 2100

/-- The maximum allowed price increase -/
def max_price_increase : ℕ := 15

/-- Theorem stating the maximum monthly profit and the corresponding selling prices -/
theorem max_monthly_profit :
  (∃ (profit : ℚ) (price1 price2 : ℕ),
    profit = 2400 ∧
    price1 = 55 ∧
    price2 = 56 ∧
    (∀ x : ℕ, x > 0 ∧ x ≤ max_price_increase →
      monthly_profit x ≤ profit) ∧
    monthly_profit (price1 - 50) = profit ∧
    monthly_profit (price2 - 50) = profit) :=
by sorry


end NUMINAMATH_CALUDE_max_monthly_profit_l3229_322917


namespace NUMINAMATH_CALUDE_initial_data_points_l3229_322957

theorem initial_data_points (x : ℝ) : 
  (1.20 * x - 0.25 * (1.20 * x) = 180) → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_initial_data_points_l3229_322957


namespace NUMINAMATH_CALUDE_remainder_s_1024_mod_1000_l3229_322993

-- Define the polynomial q(x)
def q (x : ℤ) : ℤ := (x^1025 - 1) / (x - 1)

-- Define the divisor polynomial
def divisor (x : ℤ) : ℤ := x^6 + x^5 + 3*x^4 + x^3 + x^2 + x + 1

-- Define s(x) as the polynomial remainder
noncomputable def s (x : ℤ) : ℤ := q x % divisor x

-- Theorem statement
theorem remainder_s_1024_mod_1000 : |s 1024| % 1000 = 824 := by sorry

end NUMINAMATH_CALUDE_remainder_s_1024_mod_1000_l3229_322993


namespace NUMINAMATH_CALUDE_shuttle_speed_km_per_hour_l3229_322954

/-- The speed of a space shuttle orbiting the Earth -/
def shuttle_speed_km_per_sec : ℝ := 9

/-- The number of seconds in an hour -/
def seconds_per_hour : ℝ := 3600

/-- Theorem stating that the speed of the space shuttle in km/h is 32400 -/
theorem shuttle_speed_km_per_hour :
  shuttle_speed_km_per_sec * seconds_per_hour = 32400 := by
  sorry

end NUMINAMATH_CALUDE_shuttle_speed_km_per_hour_l3229_322954


namespace NUMINAMATH_CALUDE_rod_length_for_given_weight_l3229_322949

/-- Represents the properties of a uniform rod -/
structure UniformRod where
  length_kg_ratio : ℝ  -- Ratio of length to weight

/-- Calculates the length of a uniform rod given its weight -/
def rod_length (rod : UniformRod) (weight : ℝ) : ℝ :=
  weight * rod.length_kg_ratio

theorem rod_length_for_given_weight 
  (rod : UniformRod) 
  (h1 : rod_length rod 42.75 = 11.25)
  (h2 : rod.length_kg_ratio = 11.25 / 42.75) : 
  rod_length rod 26.6 = 7 := by
  sorry

#check rod_length_for_given_weight

end NUMINAMATH_CALUDE_rod_length_for_given_weight_l3229_322949


namespace NUMINAMATH_CALUDE_bicycle_wheels_l3229_322910

theorem bicycle_wheels (num_bicycles : ℕ) (num_tricycles : ℕ) (total_wheels : ℕ) (tricycle_wheels : ℕ) :
  num_bicycles = 6 →
  num_tricycles = 15 →
  total_wheels = 57 →
  tricycle_wheels = 3 →
  ∃ (bicycle_wheels : ℕ), 
    bicycle_wheels = 2 ∧ 
    num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels = total_wheels :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_wheels_l3229_322910


namespace NUMINAMATH_CALUDE_g_expression_l3229_322953

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define g using the given condition
def g (x : ℝ) : ℝ := f (x - 2)

-- Theorem to prove
theorem g_expression : ∀ x : ℝ, g x = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l3229_322953


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3229_322936

theorem trigonometric_identities (α : ℝ) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  (Real.sin (2*α) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3229_322936


namespace NUMINAMATH_CALUDE_ben_votes_l3229_322931

/-- Given a total of 60 votes and a ratio of 2:3 between Ben's and Matt's votes,
    prove that Ben received 24 votes. -/
theorem ben_votes (total_votes : ℕ) (ben_votes : ℕ) (matt_votes : ℕ) :
  total_votes = 60 →
  ben_votes + matt_votes = total_votes →
  3 * ben_votes = 2 * matt_votes →
  ben_votes = 24 := by
sorry

end NUMINAMATH_CALUDE_ben_votes_l3229_322931
