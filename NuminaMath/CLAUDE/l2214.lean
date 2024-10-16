import Mathlib

namespace NUMINAMATH_CALUDE_shaded_area_possibilities_l2214_221472

/-- Represents a rectangle on a grid --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the configuration of rectangles in the problem --/
structure Configuration where
  abcd : Rectangle
  pqrs : Rectangle
  qrst : Rectangle
  upper_right : Rectangle

/-- The main theorem statement --/
theorem shaded_area_possibilities (config : Configuration) : 
  (config.abcd.width * config.abcd.height = 33) →
  (config.abcd.width < 7 ∧ config.abcd.height < 7) →
  (config.abcd.width ≥ 1 ∧ config.abcd.height ≥ 1) →
  (config.pqrs.width < 7 ∧ config.pqrs.height < 7) →
  (config.qrst.width < 7 ∧ config.qrst.height < 7) →
  (config.upper_right.width < 7 ∧ config.upper_right.height < 7) →
  (config.qrst.width = config.qrst.height) →
  (config.pqrs.width < config.upper_right.height) →
  (config.pqrs.width ≠ config.pqrs.height) →
  (config.upper_right.width ≠ config.upper_right.height) →
  (∃ (shaded_area : ℕ), 
    shaded_area = config.abcd.width * config.abcd.height - 
      (config.pqrs.width * config.pqrs.height + 
       config.qrst.width * config.qrst.height + 
       config.upper_right.width * config.upper_right.height) ∧
    (shaded_area = 21 ∨ shaded_area = 20 ∨ shaded_area = 17)) :=
by
  sorry


end NUMINAMATH_CALUDE_shaded_area_possibilities_l2214_221472


namespace NUMINAMATH_CALUDE_book_cost_problem_l2214_221456

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) 
  (h1 : total_cost = 360)
  (h2 : loss_percent = 0.15)
  (h3 : gain_percent = 0.19)
  (h4 : ∃ (c1 c2 : ℝ), c1 + c2 = total_cost ∧ 
                       c1 * (1 - loss_percent) = c2 * (1 + gain_percent)) :
  ∃ (loss_book_cost : ℝ), loss_book_cost = 210 ∧ 
    ∃ (c2 : ℝ), loss_book_cost + c2 = total_cost ∧ 
    loss_book_cost * (1 - loss_percent) = c2 * (1 + gain_percent) :=
sorry

end NUMINAMATH_CALUDE_book_cost_problem_l2214_221456


namespace NUMINAMATH_CALUDE_pencil_cost_l2214_221437

theorem pencil_cost (total_spent notebook_cost ruler_cost num_pencils : ℕ) 
  (h1 : total_spent = 74)
  (h2 : notebook_cost = 35)
  (h3 : ruler_cost = 18)
  (h4 : num_pencils = 3) :
  (total_spent - notebook_cost - ruler_cost) / num_pencils = 7 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l2214_221437


namespace NUMINAMATH_CALUDE_number_count_l2214_221460

theorem number_count (total_average : ℝ) (first_six_average : ℝ) (last_six_average : ℝ) (middle_number : ℝ) :
  total_average = 9.9 →
  first_six_average = 10.5 →
  last_six_average = 11.4 →
  middle_number = 22.5 →
  ∃ (n : ℕ), n = 11 ∧ n % 2 = 1 ∧
  n * total_average = 6 * first_six_average + 6 * last_six_average - middle_number :=
by sorry


end NUMINAMATH_CALUDE_number_count_l2214_221460


namespace NUMINAMATH_CALUDE_symmetry_line_is_correct_l2214_221409

/-- The line of symmetry between two circles -/
def line_of_symmetry (c1 c2 : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  fun p => ∃ (q : ℝ × ℝ), c1 q ∧ c2 (2 * p.1 - q.1, 2 * p.2 - q.2)

/-- First circle: x^2 + y^2 = 9 -/
def circle1 : ℝ × ℝ → Prop :=
  fun p => p.1^2 + p.2^2 = 9

/-- Second circle: x^2 + y^2 - 4x + 4y - 1 = 0 -/
def circle2 : ℝ × ℝ → Prop :=
  fun p => p.1^2 + p.2^2 - 4*p.1 + 4*p.2 - 1 = 0

/-- The equation of the line of symmetry: x - y - 2 = 0 -/
def symmetry_line : ℝ × ℝ → Prop :=
  fun p => p.1 - p.2 - 2 = 0

theorem symmetry_line_is_correct : 
  line_of_symmetry circle1 circle2 = symmetry_line :=
sorry

end NUMINAMATH_CALUDE_symmetry_line_is_correct_l2214_221409


namespace NUMINAMATH_CALUDE_square_difference_l2214_221448

theorem square_difference (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : b^2 - a^2 = -15 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2214_221448


namespace NUMINAMATH_CALUDE_polynomial_sum_l2214_221453

def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

theorem polynomial_sum (a b c d : ℝ) : 
  (∃ (x : ℝ), f a b x = g c d x) ∧
  (f a b (-a/2) = g c d (-c/2)) ∧
  (g c d (-a/2) = 0) ∧
  (f a b (-c/2) = 0) ∧
  (f a b 50 = -200) ∧
  (g c d 50 = -200) →
  a + c = -200 := by sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2214_221453


namespace NUMINAMATH_CALUDE_ring_payment_possible_l2214_221468

/-- Represents a chain of rings -/
structure RingChain :=
  (size : ℕ)

/-- Represents a cut ring chain -/
structure CutRingChain :=
  (segments : List RingChain)
  (total_size : ℕ)

/-- Represents a daily payment -/
structure DailyPayment :=
  (rings_given : ℕ)
  (rings_taken : ℕ)

def is_valid_payment_sequence (payments : List DailyPayment) : Prop :=
  payments.length = 7 ∧
  ∀ p ∈ payments, p.rings_given - p.rings_taken = 1

def can_make_payments (chain : RingChain) : Prop :=
  ∃ (cut_chain : CutRingChain) (payments : List DailyPayment),
    chain.size = 7 ∧
    cut_chain.total_size = 7 ∧
    cut_chain.segments.length ≤ 3 ∧
    is_valid_payment_sequence payments

theorem ring_payment_possible :
  ∃ (chain : RingChain), can_make_payments chain :=
sorry

end NUMINAMATH_CALUDE_ring_payment_possible_l2214_221468


namespace NUMINAMATH_CALUDE_black_squares_in_45th_row_l2214_221441

/-- Represents the number of squares in the nth row of the stair-step pattern -/
def squares_in_row (n : ℕ) : ℕ := 2 * n + 1

/-- Represents the number of black squares in the nth row of the stair-step pattern -/
def black_squares_in_row (n : ℕ) : ℕ := (squares_in_row n - 1) / 2

/-- Theorem stating that the number of black squares in the 45th row is 45 -/
theorem black_squares_in_45th_row :
  black_squares_in_row 45 = 45 := by
  sorry

end NUMINAMATH_CALUDE_black_squares_in_45th_row_l2214_221441


namespace NUMINAMATH_CALUDE_mary_bought_24_cards_l2214_221431

/-- The number of baseball cards Mary bought -/
def cards_bought (initial_cards promised_cards remaining_cards : ℝ) : ℝ :=
  remaining_cards - (initial_cards - promised_cards)

/-- Theorem: Mary bought 24.0 baseball cards -/
theorem mary_bought_24_cards :
  cards_bought 18.0 26.0 32.0 = 24.0 := by
  sorry

end NUMINAMATH_CALUDE_mary_bought_24_cards_l2214_221431


namespace NUMINAMATH_CALUDE_inequality_satisfied_iff_m_in_range_l2214_221400

theorem inequality_satisfied_iff_m_in_range (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8*x + 20) / (m*x^2 + 2*(m+1)*x + 9*m + 4) < 0) ↔ 
  m < -1/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_satisfied_iff_m_in_range_l2214_221400


namespace NUMINAMATH_CALUDE_batsman_new_average_l2214_221459

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  latestScore : Nat
  averageIncrease : Nat

/-- Calculates the average score after the latest innings -/
def calculateNewAverage (stats : BatsmanStats) : Nat :=
  (stats.totalRuns + stats.latestScore) / stats.innings

/-- Theorem: Given the conditions, the batsman's new average is 43 -/
theorem batsman_new_average (stats : BatsmanStats) 
  (h1 : stats.innings = 12)
  (h2 : stats.latestScore = 65)
  (h3 : stats.averageIncrease = 2)
  (h4 : calculateNewAverage stats = (calculateNewAverage stats - stats.averageIncrease) + stats.averageIncrease) :
  calculateNewAverage stats = 43 := by
  sorry

#eval calculateNewAverage { innings := 12, totalRuns := 451, latestScore := 65, averageIncrease := 2 }

end NUMINAMATH_CALUDE_batsman_new_average_l2214_221459


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2214_221475

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 3*x + 2) * (x^2 + 7*x + 12) + (x^2 + 5*x - 6) = (x^2 + 5*x + 2) * (x^2 + 5*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2214_221475


namespace NUMINAMATH_CALUDE_student_test_score_l2214_221450

theorem student_test_score (max_marks : ℕ) (pass_percentage : ℚ) (fail_margin : ℕ) : 
  max_marks = 300 → 
  pass_percentage = 60 / 100 → 
  fail_margin = 100 → 
  ∃ (student_score : ℕ), 
    student_score = max_marks * pass_percentage - fail_margin ∧ 
    student_score = 80 := by
  sorry

end NUMINAMATH_CALUDE_student_test_score_l2214_221450


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2214_221411

theorem root_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → x₂^2 - 3*x₂ - 1 = 0 → x₁ ≠ x₂ → 
  (1/x₁ + 1/x₂ : ℝ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2214_221411


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l2214_221417

theorem base_10_to_base_7 :
  ∃ (a b c d : ℕ),
    804 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
    a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
    a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 6 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l2214_221417


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l2214_221458

/-- Calculates the ratio of average speed to still water speed for a boat in a river --/
theorem boat_speed_ratio 
  (v : ℝ) -- Boat speed in still water
  (c : ℝ) -- River current speed
  (d : ℝ) -- Distance traveled each way
  (h1 : v > 0)
  (h2 : c ≥ 0)
  (h3 : c < v)
  (h4 : d > 0)
  : (2 * d) / ((d / (v + c)) + (d / (v - c))) / v = 24 / 25 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l2214_221458


namespace NUMINAMATH_CALUDE_largest_x_abs_value_equation_l2214_221406

theorem largest_x_abs_value_equation : 
  (∃ (x : ℝ), |x - 8| = 15 ∧ ∀ (y : ℝ), |y - 8| = 15 → y ≤ x) → 
  (∃ (x : ℝ), |x - 8| = 15 ∧ ∀ (y : ℝ), |y - 8| = 15 → y ≤ 23) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_abs_value_equation_l2214_221406


namespace NUMINAMATH_CALUDE_mixed_grains_in_rice_l2214_221484

theorem mixed_grains_in_rice (total_stones : ℕ) (sample_size : ℕ) (mixed_in_sample : ℕ) :
  total_stones = 1536 →
  sample_size = 256 →
  mixed_in_sample = 18 →
  (total_stones * mixed_in_sample) / sample_size = 108 :=
by
  sorry

#check mixed_grains_in_rice

end NUMINAMATH_CALUDE_mixed_grains_in_rice_l2214_221484


namespace NUMINAMATH_CALUDE_larger_number_of_product_18_sum_15_l2214_221403

theorem larger_number_of_product_18_sum_15 (x y : ℝ) : 
  x * y = 18 → x + y = 15 → max x y = 12 := by
sorry

end NUMINAMATH_CALUDE_larger_number_of_product_18_sum_15_l2214_221403


namespace NUMINAMATH_CALUDE_bicycles_in_garage_l2214_221436

theorem bicycles_in_garage (tricycles unicycles total_wheels : ℕ) 
  (h1 : tricycles = 4)
  (h2 : unicycles = 7)
  (h3 : total_wheels = 25) : ∃ bicycles : ℕ, 
  bicycles * 2 + tricycles * 3 + unicycles * 1 = total_wheels ∧ bicycles = 3 := by
  sorry

end NUMINAMATH_CALUDE_bicycles_in_garage_l2214_221436


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2214_221447

def fraction_sequence : List ℚ := 
  [1/10, 2/10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10, 10/10, 11/10, 12/10, 13/10]

theorem sum_of_fractions : 
  fraction_sequence.sum = 91/10 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2214_221447


namespace NUMINAMATH_CALUDE_exponent_simplification_l2214_221433

theorem exponent_simplification :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l2214_221433


namespace NUMINAMATH_CALUDE_circle_radius_increase_l2214_221413

/-- If a circle's radius r is increased by n, and its new area is twice the original area,
    then r = n(√2 + 1) -/
theorem circle_radius_increase (n : ℝ) (h : n > 0) :
  ∃ (r : ℝ), r > 0 ∧ π * (r + n)^2 = 2 * π * r^2 → r = n * (Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increase_l2214_221413


namespace NUMINAMATH_CALUDE_fourth_root_of_256000000_l2214_221405

theorem fourth_root_of_256000000 : (400 : ℕ) ^ 4 = 256000000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_256000000_l2214_221405


namespace NUMINAMATH_CALUDE_parabola_directrix_l2214_221467

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := x^2 + 12*y = 0

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop := y = 3

/-- Theorem: The directrix of the parabola x^2 + 12y = 0 is y = 3 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → directrix_equation y :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2214_221467


namespace NUMINAMATH_CALUDE_divisible_by_five_l2214_221442

theorem divisible_by_five (a b : ℕ+) (h : 5 ∣ (a * b)) : 5 ∣ a ∨ 5 ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l2214_221442


namespace NUMINAMATH_CALUDE_tom_books_theorem_l2214_221416

def books_problem (initial_books sold_books bought_books : ℕ) : Prop :=
  let remaining_books := initial_books - sold_books
  let final_books := remaining_books + bought_books
  final_books = 39

theorem tom_books_theorem :
  books_problem 5 4 38 := by sorry

end NUMINAMATH_CALUDE_tom_books_theorem_l2214_221416


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l2214_221474

theorem angle_sum_theorem (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2)
  (h_equation : (1 + Real.sqrt 3 * Real.tan α) * (1 + Real.sqrt 3 * Real.tan β) = 4) :
  α + β = π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l2214_221474


namespace NUMINAMATH_CALUDE_right_triangle_vector_k_l2214_221402

-- Define a right-angled triangle ABC
structure RightTriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Define the theorem
theorem right_triangle_vector_k (k : ℝ) (triangle : RightTriangleABC) 
  (hBA : triangle.B.1 - triangle.A.1 = k ∧ triangle.B.2 - triangle.A.2 = 1)
  (hBC : triangle.B.1 - triangle.C.1 = 2 ∧ triangle.B.2 - triangle.C.2 = 3) :
  k = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_vector_k_l2214_221402


namespace NUMINAMATH_CALUDE_probability_second_draw_given_first_l2214_221401

/-- The probability of drawing a high-quality item on the second draw, given that the first draw was a high-quality item, when there are 5 high-quality items and 3 defective items in total. -/
theorem probability_second_draw_given_first (total_items : ℕ) (high_quality : ℕ) (defective : ℕ) :
  total_items = high_quality + defective →
  high_quality = 5 →
  defective = 3 →
  (high_quality - 1 : ℚ) / (total_items - 1) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_draw_given_first_l2214_221401


namespace NUMINAMATH_CALUDE_sequence_problem_l2214_221470

theorem sequence_problem (a : ℕ → ℚ) (m : ℕ) :
  a 1 = 1 →
  (∀ n : ℕ, n ≥ 1 → a n - a (n + 1) = a (n + 1) * a n) →
  8 * a m = 1 →
  m = 8 :=
by sorry

end NUMINAMATH_CALUDE_sequence_problem_l2214_221470


namespace NUMINAMATH_CALUDE_diagonal_division_l2214_221430

/-- A regular polygon with 2018 vertices, labeled clockwise from 1 to 2018 -/
structure RegularPolygon2018 where
  vertices : Fin 2018

/-- The number of vertices between two given vertices in a clockwise direction -/
def verticesBetween (a b : Fin 2018) : ℕ :=
  if b.val ≥ a.val then
    b.val - a.val + 1
  else
    (2018 - a.val) + b.val + 1

/-- The result of drawing diagonals in the polygon -/
def diagonalResult (p : RegularPolygon2018) : Prop :=
  let polygon1 := verticesBetween 18 1018
  let polygon2 := verticesBetween 1018 2000
  let polygon3 := verticesBetween 2000 18 + 1  -- Adding 1 for vertex 1018
  polygon1 = 1001 ∧ polygon2 = 983 ∧ polygon3 = 38

theorem diagonal_division (p : RegularPolygon2018) : diagonalResult p := by
  sorry

end NUMINAMATH_CALUDE_diagonal_division_l2214_221430


namespace NUMINAMATH_CALUDE_pig_profit_is_960_l2214_221419

/-- Calculates the profit from selling pigs given the specified conditions -/
def calculate_pig_profit (num_piglets : ℕ) (sale_price : ℕ) (feeding_cost : ℕ) 
  (months_group1 : ℕ) (months_group2 : ℕ) : ℕ :=
  let revenue := num_piglets * sale_price
  let cost_group1 := (num_piglets / 2) * feeding_cost * months_group1
  let cost_group2 := (num_piglets / 2) * feeding_cost * months_group2
  let total_cost := cost_group1 + cost_group2
  revenue - total_cost

/-- The profit from selling pigs under the given conditions is $960 -/
theorem pig_profit_is_960 : 
  calculate_pig_profit 6 300 10 12 16 = 960 := by
  sorry

end NUMINAMATH_CALUDE_pig_profit_is_960_l2214_221419


namespace NUMINAMATH_CALUDE_centroid_trace_area_centroid_trace_area_diameter_30_l2214_221432

/-- The area of the region bounded by the curve traced by the centroid of a triangle
    inscribed in a circle, where the base of the triangle is a diameter of the circle. -/
theorem centroid_trace_area (r : ℝ) (h : r > 0) : 
  (π * (r / 3)^2) = (25 * π / 9) * r^2 := by
  sorry

/-- The specific case where the diameter of the circle is 30 -/
theorem centroid_trace_area_diameter_30 : 
  (π * 5^2) = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_centroid_trace_area_centroid_trace_area_diameter_30_l2214_221432


namespace NUMINAMATH_CALUDE_f_4_3_2_1_l2214_221491

/-- The mapping f from (a₁, a₂, a₃, a₄) to (b₁, b₂, b₃, b₄) based on the equation
    x^4 + a₁x³ + a₂x² + a₃x + a₄ = (x+1)^4 + b₁(x+1)³ + b₂(x+1)² + b₃(x+1) + b₄ -/
def f (a₁ a₂ a₃ a₄ : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  sorry

theorem f_4_3_2_1 : f 4 3 2 1 = (0, -3, 4, -1) := by
  sorry

end NUMINAMATH_CALUDE_f_4_3_2_1_l2214_221491


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2214_221427

-- Define the log function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the condition for lgm < 1
def condition (m : ℝ) : Prop := log m < 1

-- Define the set {1, 2}
def set_B : Set ℝ := {1, 2}

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ m ∈ set_B, condition m) ∧
  (∃ m : ℝ, condition m ∧ m ∉ set_B) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2214_221427


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l2214_221469

def a : ℝ × ℝ := (2, 3)
def b (m : ℝ) : ℝ × ℝ := (m, -6)

theorem magnitude_of_sum (m : ℝ) (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) :
  Real.sqrt ((2 * a.1 + (b m).1)^2 + (2 * a.2 + (b m).2)^2) = 13 :=
by sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_l2214_221469


namespace NUMINAMATH_CALUDE_one_true_proposition_l2214_221493

-- Define the basic concepts
def Point : Type := ℝ × ℝ
def Triangle (A B C : Point) : Prop := True  -- Simplified definition
def Isosceles (A B C : Point) : Prop := True  -- Simplified definition

-- Define the original proposition
def original_prop (A B C : Point) : Prop :=
  A.1 = B.1 ∧ A.2 = B.2 → Isosceles A B C

-- Define the converse proposition
def converse_prop (A B C : Point) : Prop :=
  Isosceles A B C → A.1 = B.1 ∧ A.2 = B.2

-- Define the inverse proposition
def inverse_prop (A B C : Point) : Prop :=
  ¬(A.1 = B.1 ∧ A.2 = B.2) → ¬(Isosceles A B C)

-- Define the contrapositive proposition
def contrapositive_prop (A B C : Point) : Prop :=
  ¬(Isosceles A B C) → ¬(A.1 = B.1 ∧ A.2 = B.2)

-- The theorem to be proved
theorem one_true_proposition (A B C : Point) :
  (original_prop A B C) ∧
  (¬(converse_prop A B C) ∨ ¬(inverse_prop A B C)) ∧
  (contrapositive_prop A B C) :=
sorry

end NUMINAMATH_CALUDE_one_true_proposition_l2214_221493


namespace NUMINAMATH_CALUDE_tangent_slope_constraint_implies_a_range_l2214_221452

theorem tangent_slope_constraint_implies_a_range
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x, f x = -x^3 + a*x^2 + b)
  (h2 : ∀ x, (deriv f x) < 1) :
  -Real.sqrt 3 < a ∧ a < Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_constraint_implies_a_range_l2214_221452


namespace NUMINAMATH_CALUDE_logarithm_calculation_l2214_221466

theorem logarithm_calculation : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - (5 : ℝ) ^ (Real.log 3 / Real.log 5) = -1 :=
by sorry

-- Note: We cannot include the second part of the problem due to inconsistencies in the problem statement and solution.

end NUMINAMATH_CALUDE_logarithm_calculation_l2214_221466


namespace NUMINAMATH_CALUDE_toy_price_calculation_l2214_221483

theorem toy_price_calculation (toy_price : ℝ) : 
  (3 * toy_price + 2 * 5 + 5 * 6 = 70) → toy_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_toy_price_calculation_l2214_221483


namespace NUMINAMATH_CALUDE_min_value_theorem_l2214_221463

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (min_val : ℝ), min_val = 2 * Real.sqrt 3 ∧
   ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 →
   (2 * x^2 + 1) / (x * y) - 2 ≥ min_val) ∧
  (2 * a^2 + 1) / (a * b) - 2 = 2 * Real.sqrt 3 ↔ a = (Real.sqrt 3 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2214_221463


namespace NUMINAMATH_CALUDE_equality_of_squares_l2214_221445

theorem equality_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ b^2 * (c + a - b) = c^2 * (a + b - c)) :
  a = b ∧ b = c :=
sorry

end NUMINAMATH_CALUDE_equality_of_squares_l2214_221445


namespace NUMINAMATH_CALUDE_counterexample_exists_l2214_221424

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2214_221424


namespace NUMINAMATH_CALUDE_negative_two_cubed_l2214_221492

theorem negative_two_cubed : (-2 : ℤ)^3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_l2214_221492


namespace NUMINAMATH_CALUDE_always_uninfected_cell_l2214_221422

/-- Represents a square grid of cells -/
structure Grid (n : ℕ) where
  side_length : ℕ
  cells : Fin n → Fin n → Bool

/-- Represents the state of infection in the grid -/
structure InfectionState (n : ℕ) where
  grid : Grid n
  infected_cells : Set (Fin n × Fin n)

/-- A function to determine if a cell can be infected based on its neighbors -/
def can_be_infected (state : InfectionState n) (cell : Fin n × Fin n) : Bool :=
  sorry

/-- The perimeter of the infected region -/
def infected_perimeter (state : InfectionState n) : ℕ :=
  sorry

/-- Theorem stating that there will always be at least one uninfected cell -/
theorem always_uninfected_cell (n : ℕ) (initial_state : InfectionState n) :
  ∃ (cell : Fin n × Fin n), cell ∉ initial_state.infected_cells ∧
    ∀ (final_state : InfectionState n),
      (∀ (c : Fin n × Fin n), c ∉ initial_state.infected_cells →
        (c ∈ final_state.infected_cells → can_be_infected initial_state c)) →
      cell ∉ final_state.infected_cells :=
  sorry

end NUMINAMATH_CALUDE_always_uninfected_cell_l2214_221422


namespace NUMINAMATH_CALUDE_f_strictly_increasing_on_interval_l2214_221476

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

-- State the theorem
theorem f_strictly_increasing_on_interval :
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_on_interval_l2214_221476


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2214_221477

theorem cone_lateral_surface_area (l r : ℝ) (h1 : l = 5) (h2 : r = 2) :
  π * r * l = 10 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2214_221477


namespace NUMINAMATH_CALUDE_exponential_inequality_l2214_221438

theorem exponential_inequality (a x y : ℝ) :
  (a > 1 ∧ x > y → a^x > a^y) ∧ (a < 1 ∧ x > y → a^x < a^y) := by
sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2214_221438


namespace NUMINAMATH_CALUDE_three_layer_rug_area_l2214_221410

/-- Given three rugs with a total area, floor area covered when overlapped, and area covered by two layers,
    calculate the area covered by three layers. -/
theorem three_layer_rug_area (total_area floor_area two_layer_area : ℝ) 
    (h1 : total_area = 204)
    (h2 : floor_area = 140)
    (h3 : two_layer_area = 24) :
  total_area - floor_area = 64 := by
  sorry

#check three_layer_rug_area

end NUMINAMATH_CALUDE_three_layer_rug_area_l2214_221410


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2214_221434

theorem polynomial_divisibility (r s : ℚ) : 
  (∀ x : ℚ, (x + 3) * (x - 2) ∣ (x^5 - 2*x^4 + 3*x^3 - r*x^2 + s*x - 8)) →
  r = -482/15 ∧ s = -1024/15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2214_221434


namespace NUMINAMATH_CALUDE_jackson_williams_money_ratio_l2214_221471

/-- Given that Jackson and Williams have a total of $150 and Jackson has $125,
    prove that the ratio of Jackson's money to Williams' money is 5:1 -/
theorem jackson_williams_money_ratio :
  ∀ (jackson_money williams_money : ℝ),
    jackson_money + williams_money = 150 →
    jackson_money = 125 →
    jackson_money / williams_money = 5 := by
  sorry

end NUMINAMATH_CALUDE_jackson_williams_money_ratio_l2214_221471


namespace NUMINAMATH_CALUDE_farmer_water_capacity_l2214_221415

/-- Calculates the total water capacity of a single truck -/
def truckCapacity (tankCapacities : List ℕ) : ℕ :=
  tankCapacities.sum

/-- Calculates the amount of water in a truck given its capacity and fill percentage -/
def waterInTruck (capacity : ℕ) (fillPercentage : ℕ) : ℕ :=
  capacity * fillPercentage / 100

/-- Represents the problem of calculating total water capacity across multiple trucks -/
def waterCapacityProblem (tankCapacities : List ℕ) (fillPercentages : List ℕ) : Prop :=
  let capacity := truckCapacity tankCapacities
  let waterAmounts := fillPercentages.map (waterInTruck capacity)
  waterAmounts.sum = 2750

/-- The main theorem stating the solution to the water capacity problem -/
theorem farmer_water_capacity :
  waterCapacityProblem [200, 250, 300, 350] [100, 75, 50, 25, 0] := by
  sorry

#check farmer_water_capacity

end NUMINAMATH_CALUDE_farmer_water_capacity_l2214_221415


namespace NUMINAMATH_CALUDE_total_hamburger_combinations_l2214_221418

/-- The number of condiments available. -/
def num_condiments : ℕ := 10

/-- The number of choices for each condiment (include or not include). -/
def choices_per_condiment : ℕ := 2

/-- The number of choices for meat patties. -/
def meat_patty_choices : ℕ := 3

/-- The number of choices for bun types. -/
def bun_choices : ℕ := 3

/-- Theorem stating the total number of different hamburger combinations. -/
theorem total_hamburger_combinations :
  (choices_per_condiment ^ num_condiments) * meat_patty_choices * bun_choices = 9216 := by
  sorry


end NUMINAMATH_CALUDE_total_hamburger_combinations_l2214_221418


namespace NUMINAMATH_CALUDE_f_is_linear_equation_one_var_l2214_221435

/-- A linear equation with one variable is of the form ax + b = 0, where a and b are real numbers and a ≠ 0 -/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function f(x) = x - 1 -/
def f (x : ℝ) : ℝ := x - 1

theorem f_is_linear_equation_one_var :
  is_linear_equation_one_var f :=
sorry

end NUMINAMATH_CALUDE_f_is_linear_equation_one_var_l2214_221435


namespace NUMINAMATH_CALUDE_sum_of_ratios_l2214_221479

theorem sum_of_ratios (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x/y + y/z + z/x + y/x + z/y + x/z = 9)
  (h2 : x + y + z = 3) :
  x/y + y/z + z/x = 4.5 ∧ y/x + z/y + x/z = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_l2214_221479


namespace NUMINAMATH_CALUDE_rectangular_plot_fence_poles_l2214_221461

/-- Calculates the number of fence poles needed for a rectangular plot -/
def fence_poles (length width pole_distance : ℕ) : ℕ :=
  (2 * (length + width)) / pole_distance

/-- Theorem: A 90m by 40m rectangular plot with fence poles 5m apart needs 52 poles -/
theorem rectangular_plot_fence_poles :
  fence_poles 90 40 5 = 52 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_fence_poles_l2214_221461


namespace NUMINAMATH_CALUDE_fraction_value_l2214_221407

theorem fraction_value : 
  let a := 423134
  let b := 423133
  (a * 846267 - b) / (b * 846267 + a) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l2214_221407


namespace NUMINAMATH_CALUDE_sin_squared_3x_maximum_l2214_221426

open Real

theorem sin_squared_3x_maximum (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (3 * x) ^ 2) :
  ∃ x, x ∈ Set.Ioo 0 0.6 ∧ f x = 1 ∧ ∀ y ∈ Set.Ioo 0 0.6, f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_sin_squared_3x_maximum_l2214_221426


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2214_221498

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 - Complex.I) / Complex.I ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2214_221498


namespace NUMINAMATH_CALUDE_f_max_value_l2214_221414

/-- The function f(x) = -x^2 + 4x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + a

/-- Theorem: If f(x) has a minimum value of -2 on [0, 1], then its maximum value on [0, 1] is 1 -/
theorem f_max_value (a : ℝ) :
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f a x ≤ f a y) →
  (∃ x ∈ Set.Icc 0 1, f a x = -2) →
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f a y ≤ f a x) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = 1) :=
by sorry

#check f_max_value

end NUMINAMATH_CALUDE_f_max_value_l2214_221414


namespace NUMINAMATH_CALUDE_divisible_by_11_iff_valid_pair_l2214_221457

def is_valid_pair (a b : Nat) : Prop :=
  (a, b) ∈ [(8, 0), (9, 1), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9)]

def number_from_digits (a b : Nat) : Nat :=
  380000 + a * 1000 + 750 + b

theorem divisible_by_11_iff_valid_pair (a b : Nat) :
  a < 10 ∧ b < 10 →
  (number_from_digits a b) % 11 = 0 ↔ is_valid_pair a b := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_11_iff_valid_pair_l2214_221457


namespace NUMINAMATH_CALUDE_remainder_theorem_l2214_221420

theorem remainder_theorem (r : ℤ) : (r^11 - 1) % (r - 2) = 2047 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2214_221420


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2214_221428

theorem arithmetic_sequence_problem (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →
  a 2 + 4 * a 7 + a 12 = 96 →
  2 * a 3 + a 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2214_221428


namespace NUMINAMATH_CALUDE_average_initial_price_is_54_l2214_221421

/-- Represents the price and quantity of fruit. -/
structure FruitInfo where
  applePrice : ℕ
  orangePrice : ℕ
  totalFruit : ℕ
  orangesPutBack : ℕ
  avgPriceKept : ℕ

/-- Calculates the average price of initially selected fruit. -/
def averageInitialPrice (info : FruitInfo) : ℚ :=
  let apples := info.totalFruit - (info.totalFruit - info.orangesPutBack - 
    (info.avgPriceKept * (info.totalFruit - info.orangesPutBack) - 
    info.orangePrice * (info.totalFruit - info.orangesPutBack - info.orangesPutBack)) / 
    (info.applePrice - info.orangePrice))
  let oranges := info.totalFruit - apples
  (info.applePrice * apples + info.orangePrice * oranges) / info.totalFruit

/-- Theorem stating that the average initial price is 54 cents. -/
theorem average_initial_price_is_54 (info : FruitInfo) 
    (h1 : info.applePrice = 40)
    (h2 : info.orangePrice = 60)
    (h3 : info.totalFruit = 10)
    (h4 : info.orangesPutBack = 6)
    (h5 : info.avgPriceKept = 45) :
  averageInitialPrice info = 54 := by
  sorry

end NUMINAMATH_CALUDE_average_initial_price_is_54_l2214_221421


namespace NUMINAMATH_CALUDE_repeating_six_equals_two_thirds_l2214_221478

/-- The decimal representation of a number with infinitely repeating 6 after the decimal point -/
def repeating_six : ℚ := sorry

/-- Theorem stating that the repeating decimal 0.666... is equal to 2/3 -/
theorem repeating_six_equals_two_thirds : repeating_six = 2/3 := by sorry

end NUMINAMATH_CALUDE_repeating_six_equals_two_thirds_l2214_221478


namespace NUMINAMATH_CALUDE_distribute_volunteers_eq_twelve_l2214_221446

/-- The number of ways to distribute 8 volunteer positions to 3 schools -/
def distribute_volunteers : ℕ :=
  let total_positions := 8
  let num_schools := 3
  let total_partitions := Nat.choose (total_positions - 1) (num_schools - 1)
  let equal_allocations := 3 * 3  -- (1,1,6), (2,2,4), (3,3,2)
  total_partitions - equal_allocations

/-- Theorem: The number of ways to distribute 8 volunteer positions to 3 schools,
    with each school receiving at least one position and the allocations being unequal, is 12 -/
theorem distribute_volunteers_eq_twelve : distribute_volunteers = 12 := by
  sorry

end NUMINAMATH_CALUDE_distribute_volunteers_eq_twelve_l2214_221446


namespace NUMINAMATH_CALUDE_gasoline_distribution_impossible_l2214_221496

theorem gasoline_distribution_impossible : ¬∃ (x y z : ℝ), 
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 
  x + y + z = 50 ∧ 
  x = y + 10 ∧ 
  z + 26 = y := by
  sorry

end NUMINAMATH_CALUDE_gasoline_distribution_impossible_l2214_221496


namespace NUMINAMATH_CALUDE_two_sets_of_points_l2214_221444

/-- Given two sets of points in a plane, if the total number of connecting lines
    is 136 and the sum of connecting lines between the groups is 66,
    then one set contains 10 points and the other contains 7 points. -/
theorem two_sets_of_points (x y : ℕ) : 
  x + y = 17 ∧ 
  (x * (x - 1) + y * (y - 1)) / 2 = 136 ∧ 
  x * y = 66 →
  (x = 10 ∧ y = 7) ∨ (x = 7 ∧ y = 10) :=
by sorry

end NUMINAMATH_CALUDE_two_sets_of_points_l2214_221444


namespace NUMINAMATH_CALUDE_sum_of_integers_l2214_221404

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + 2*r = 10)
  (eq2 : q - r + s = 9)
  (eq3 : r - 2*s + p = 6)
  (eq4 : s - p + q = 7) :
  p + q + r + s = 32 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2214_221404


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2214_221451

/-- The number of apples initially in the cafeteria -/
def initial_apples : ℕ := sorry

/-- The number of apples handed out to students -/
def apples_to_students : ℕ := 8

/-- The number of pies made -/
def pies_made : ℕ := 6

/-- The number of apples required for each pie -/
def apples_per_pie : ℕ := 9

/-- Theorem stating that the initial number of apples is 62 -/
theorem cafeteria_apples : initial_apples = 62 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2214_221451


namespace NUMINAMATH_CALUDE_wall_width_correct_l2214_221440

/-- Represents the dimensions and properties of a wall -/
structure Wall where
  width : ℝ
  height : ℝ
  length : ℝ
  volume : ℝ

/-- The width of the wall given the conditions -/
def wall_width (w : Wall) : ℝ :=
  (384 : ℝ) ^ (1/3)

/-- Theorem stating that the calculated width satisfies the given conditions -/
theorem wall_width_correct (w : Wall) 
  (h_height : w.height = 6 * w.width)
  (h_length : w.length = 7 * w.height)
  (h_volume : w.volume = 16128) : 
  w.width = wall_width w := by
  sorry

#eval wall_width { width := 0, height := 0, length := 0, volume := 16128 }

end NUMINAMATH_CALUDE_wall_width_correct_l2214_221440


namespace NUMINAMATH_CALUDE_function_inequality_l2214_221412

/-- Given a function f(x) = x - 1 - ln x, if f(x) ≥ kx - 2 for all x > 0, 
    then k ≤ 1 - 1/e² -/
theorem function_inequality (k : ℝ) : 
  (∀ x > 0, x - 1 - Real.log x ≥ k * x - 2) → k ≤ 1 - 1 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2214_221412


namespace NUMINAMATH_CALUDE_congruence_remainders_l2214_221423

theorem congruence_remainders (x : ℤ) 
  (h1 : x ≡ 25 [ZMOD 35])
  (h2 : x ≡ 31 [ZMOD 42]) :
  (x ≡ 10 [ZMOD 15]) ∧ (x ≡ 13 [ZMOD 18]) := by
  sorry

end NUMINAMATH_CALUDE_congruence_remainders_l2214_221423


namespace NUMINAMATH_CALUDE_sugar_cups_correct_l2214_221480

/-- Represents the number of cups of sugar in the lemonade mixture -/
def sugar : ℕ := 28

/-- Represents the number of cups of water in the lemonade mixture -/
def water : ℕ := 56

/-- The total number of cups used in the mixture -/
def total_cups : ℕ := 84

/-- Theorem stating that the number of cups of sugar is correct given the conditions -/
theorem sugar_cups_correct :
  (sugar + water = total_cups) ∧ (2 * sugar = water) ∧ (sugar = 28) := by
  sorry

end NUMINAMATH_CALUDE_sugar_cups_correct_l2214_221480


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2214_221425

theorem complex_fraction_equality : (2 - I) / (1 - I) = 3/2 + I/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2214_221425


namespace NUMINAMATH_CALUDE_shortest_player_height_l2214_221486

theorem shortest_player_height (tallest_height shortest_height height_difference : ℝ) 
  (h1 : tallest_height = 77.75)
  (h2 : height_difference = 9.5)
  (h3 : tallest_height = shortest_height + height_difference) :
  shortest_height = 68.25 := by
  sorry

end NUMINAMATH_CALUDE_shortest_player_height_l2214_221486


namespace NUMINAMATH_CALUDE_problem_statement_l2214_221465

theorem problem_statement :
  let M := (Real.sqrt (3 + Real.sqrt 8) + Real.sqrt (3 - Real.sqrt 8)) / Real.sqrt (2 * Real.sqrt 2 + 1) - Real.sqrt (4 - 2 * Real.sqrt 3)
  M = 3 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2214_221465


namespace NUMINAMATH_CALUDE_function_and_tangent_line_l2214_221429

/-- Given a function f with the property that 
    f(x) = (1/4) * f'(1) * x^2 + 2 * f(1) * x - 4,
    prove that f(x) = 2x^2 + 4x - 4 and its tangent line
    at (0, f(0)) has the equation 4x - y - 4 = 0 -/
theorem function_and_tangent_line 
  (f : ℝ → ℝ) 
  (h : ∀ x, f x = (1/4) * (deriv f 1) * x^2 + 2 * (f 1) * x - 4) :
  (∀ x, f x = 2*x^2 + 4*x - 4) ∧ 
  (∃ a b c : ℝ, a = 4 ∧ b = -1 ∧ c = -4 ∧ 
    ∀ x y, y = (deriv f 0) * x + f 0 ↔ a*x + b*y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_and_tangent_line_l2214_221429


namespace NUMINAMATH_CALUDE_gcd_of_ones_l2214_221454

theorem gcd_of_ones (m n : ℕ+) :
  Nat.gcd ((10^(m.val) - 1) / 9) ((10^(n.val) - 1) / 9) = (10^(Nat.gcd m.val n.val) - 1) / 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_ones_l2214_221454


namespace NUMINAMATH_CALUDE_helmet_sales_theorem_l2214_221408

/-- Represents the monthly growth rate of helmet sales -/
def monthly_growth_rate : ℝ := sorry

/-- Represents the optimal selling price of helmets -/
def optimal_selling_price : ℝ := sorry

/-- April sales volume -/
def april_sales : ℝ := 100

/-- June sales volume -/
def june_sales : ℝ := 144

/-- Cost price per helmet -/
def cost_price : ℝ := 30

/-- Reference selling price -/
def reference_price : ℝ := 40

/-- Reference monthly sales volume -/
def reference_sales : ℝ := 600

/-- Sales volume decrease per yuan increase in price -/
def sales_decrease_rate : ℝ := 10

/-- Target monthly profit -/
def target_profit : ℝ := 10000

theorem helmet_sales_theorem :
  (april_sales * (1 + monthly_growth_rate)^2 = june_sales) ∧
  ((optimal_selling_price - cost_price) * 
   (reference_sales - sales_decrease_rate * (optimal_selling_price - reference_price)) = target_profit) ∧
  (monthly_growth_rate = 0.2) ∧
  (optimal_selling_price = 50) := by sorry

end NUMINAMATH_CALUDE_helmet_sales_theorem_l2214_221408


namespace NUMINAMATH_CALUDE_typing_speed_difference_l2214_221495

theorem typing_speed_difference (before_speed after_speed : ℕ) 
  (h1 : before_speed = 10) 
  (h2 : after_speed = 8) 
  (difference : ℕ) 
  (h3 : difference = 10) : 
  ∃ (minutes : ℕ), minutes * before_speed - minutes * after_speed = difference ∧ minutes = 5 :=
sorry

end NUMINAMATH_CALUDE_typing_speed_difference_l2214_221495


namespace NUMINAMATH_CALUDE_total_spending_is_correct_l2214_221439

-- Define the structure for a week's theater visit
structure WeekVisit where
  duration : Float
  price_per_hour : Float
  discount_rate : Float
  visit_count : Nat

-- Define the list of visits for 6 weeks
def theater_visits : List WeekVisit := [
  { duration := 3, price_per_hour := 5, discount_rate := 0.2, visit_count := 1 },
  { duration := 2.5, price_per_hour := 6, discount_rate := 0.1, visit_count := 1 },
  { duration := 4, price_per_hour := 4, discount_rate := 0, visit_count := 1 },
  { duration := 3, price_per_hour := 5, discount_rate := 0.2, visit_count := 1 },
  { duration := 3.5, price_per_hour := 6, discount_rate := 0.1, visit_count := 2 },
  { duration := 2, price_per_hour := 7, discount_rate := 0, visit_count := 1 }
]

-- Define the transportation cost per visit
def transportation_cost : Float := 3

-- Calculate the total cost for a single visit
def visit_cost (visit : WeekVisit) : Float :=
  let performance_cost := visit.duration * visit.price_per_hour
  let discount := performance_cost * visit.discount_rate
  let discounted_cost := performance_cost - discount
  discounted_cost + transportation_cost

-- Calculate the total spending for all visits
def total_spending : Float :=
  theater_visits.map (fun visit => visit_cost visit * visit.visit_count.toFloat) |>.sum

-- Theorem statement
theorem total_spending_is_correct : total_spending = 126.3 := by
  sorry

end NUMINAMATH_CALUDE_total_spending_is_correct_l2214_221439


namespace NUMINAMATH_CALUDE_book_club_picks_l2214_221449

theorem book_club_picks (total_members : ℕ) (meeting_weeks : ℕ) (guest_picks : ℕ) :
  total_members = 13 →
  meeting_weeks = 48 →
  guest_picks = 12 →
  (meeting_weeks - guest_picks) / total_members = 2 :=
by sorry

end NUMINAMATH_CALUDE_book_club_picks_l2214_221449


namespace NUMINAMATH_CALUDE_arianna_sleep_hours_l2214_221443

/-- Represents the number of hours in a day. -/
def hours_in_day : ℕ := 24

/-- Represents the number of hours Arianna spends at work. -/
def work_hours : ℕ := 6

/-- Represents the number of hours Arianna spends in class. -/
def class_hours : ℕ := 3

/-- Represents the number of hours Arianna spends at the gym. -/
def gym_hours : ℕ := 2

/-- Represents the number of hours Arianna spends on other daily chores. -/
def chore_hours : ℕ := 5

/-- Represents the number of hours Arianna sleeps. -/
def sleep_hours : ℕ := hours_in_day - (work_hours + class_hours + gym_hours + chore_hours)

/-- Theorem stating that Arianna sleeps for 8 hours a day. -/
theorem arianna_sleep_hours : sleep_hours = 8 := by
  sorry

end NUMINAMATH_CALUDE_arianna_sleep_hours_l2214_221443


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2214_221464

/-- Calculates the length of a train given its speed and the time it takes to cross a bridge of known length. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_ms := train_speed * 1000 / 3600
  let total_distance := speed_ms * crossing_time
  total_distance - bridge_length

/-- Proves that a train traveling at 40 km/h that crosses a 300-meter bridge in 45 seconds has a length of approximately 199.95 meters. -/
theorem train_length_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_length 40 300 45 - 199.95| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2214_221464


namespace NUMINAMATH_CALUDE_parallelogram_area_and_scaling_l2214_221485

theorem parallelogram_area_and_scaling :
  let base : ℝ := 6
  let height : ℝ := 20
  let area := base * height
  let scaled_base := 3 * base
  let scaled_height := 3 * height
  let scaled_area := scaled_base * scaled_height
  (area = 120) ∧ 
  (scaled_area = 9 * area) ∧ 
  (scaled_area = 1080) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_and_scaling_l2214_221485


namespace NUMINAMATH_CALUDE_prime_quadruplet_l2214_221455

theorem prime_quadruplet (p₁ p₂ p₃ p₄ : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
  p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧
  p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882 →
  ((p₁, p₂, p₃, p₄) = (2, 5, 19, 37) ∨ 
   (p₁, p₂, p₃, p₄) = (2, 11, 19, 31) ∨ 
   (p₁, p₂, p₃, p₄) = (2, 13, 19, 29)) :=
by sorry

end NUMINAMATH_CALUDE_prime_quadruplet_l2214_221455


namespace NUMINAMATH_CALUDE_shaded_area_sum_l2214_221499

/-- Represents a triangle in the hexagon -/
structure Triangle :=
  (size : Nat)
  (area : ℝ)

/-- The hexagon composed of equilateral triangles -/
structure Hexagon :=
  (unit_triangle : Triangle)
  (small : Triangle)
  (medium : Triangle)
  (large : Triangle)

/-- The theorem stating the area of the shaded part -/
theorem shaded_area_sum (h : Hexagon) 
  (h_unit : h.unit_triangle.area = 10)
  (h_small : h.small.size = 1)
  (h_medium : h.medium.size = 6)
  (h_large : h.large.size = 13) :
  h.small.area + h.medium.area + h.large.area = 110 := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_sum_l2214_221499


namespace NUMINAMATH_CALUDE_symmetry_implies_a_equals_one_monotonic_increasing_implies_a_leq_one_max_value_on_interval_l2214_221473

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Theorem 1: If f(1+x) = f(1-x) for all x, then a = 1
theorem symmetry_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, f a (1+x) = f a (1-x)) → a = 1 := by sorry

-- Theorem 2: If f is monotonically increasing on [1, +∞), then a ≤ 1
theorem monotonic_increasing_implies_a_leq_one (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x ∧ x < y → f a x < f a y) → a ≤ 1 := by sorry

-- Theorem 3: The maximum value of f on [-1, 1] is 2
theorem max_value_on_interval (a : ℝ) :
  ∃ m : ℝ, m = 2 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f a x ≤ m := by sorry

end NUMINAMATH_CALUDE_symmetry_implies_a_equals_one_monotonic_increasing_implies_a_leq_one_max_value_on_interval_l2214_221473


namespace NUMINAMATH_CALUDE_sin_90_degrees_l2214_221462

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l2214_221462


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2214_221482

theorem last_two_digits_product (n : ℤ) : 
  (n % 100 % 4 = 0) → 
  ((n % 100) / 10 + n % 10 = 13) → 
  ((n % 100) / 10 * (n % 10) = 42) := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2214_221482


namespace NUMINAMATH_CALUDE_certain_number_problem_l2214_221494

theorem certain_number_problem (x : ℝ) : 
  3 + x + 333 + 33.3 = 399.6 → x = 30.3 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2214_221494


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_max_l2214_221488

theorem quadratic_function_inequality_max (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ M : ℝ, M = Real.sqrt 6 - 2 ∧ 
    (∀ a' b' c' : ℝ, (∀ x : ℝ, a' * x^2 + b' * x + c' ≥ 2 * a' * x + b') → 
      b'^2 / (a'^2 + 2 * c'^2) ≤ M) ∧
    (∃ a' b' c' : ℝ, (∀ x : ℝ, a' * x^2 + b' * x + c' ≥ 2 * a' * x + b') ∧ 
      b'^2 / (a'^2 + 2 * c'^2) = M)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_max_l2214_221488


namespace NUMINAMATH_CALUDE_points_in_small_square_l2214_221489

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square region in a 2D plane -/
structure Square where
  center : Point
  side_length : ℝ

/-- Check if a point is inside a square -/
def is_point_in_square (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) ≤ s.side_length / 2 ∧
  abs (p.y - s.center.y) ≤ s.side_length / 2

/-- The main theorem -/
theorem points_in_small_square (points : Finset Point) 
    (h1 : points.card = 51)
    (h2 : ∀ p ∈ points, is_point_in_square p ⟨⟨0.5, 0.5⟩, 1⟩) :
    ∃ (small_square : Square),
      small_square.side_length = 0.2 ∧
      ∃ (p1 p2 p3 : Point),
        p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
        p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
        is_point_in_square p1 small_square ∧
        is_point_in_square p2 small_square ∧
        is_point_in_square p3 small_square :=
  sorry

end NUMINAMATH_CALUDE_points_in_small_square_l2214_221489


namespace NUMINAMATH_CALUDE_fuel_consumption_statements_correct_l2214_221487

/-- Represents the fuel consumption data for a car journey -/
structure FuelConsumptionData where
  initial_fuel : ℝ
  distance_interval : ℝ
  fuel_decrease_per_interval : ℝ
  total_distance : ℝ

/-- Theorem stating the correctness of all fuel consumption statements -/
theorem fuel_consumption_statements_correct
  (data : FuelConsumptionData)
  (h_initial : data.initial_fuel = 45)
  (h_interval : data.distance_interval = 50)
  (h_decrease : data.fuel_decrease_per_interval = 4)
  (h_total : data.total_distance = 500) :
  (data.initial_fuel = 45) ∧
  ((data.fuel_decrease_per_interval / data.distance_interval) * 100 = 8) ∧
  (∀ x y : ℝ, y = data.initial_fuel - (data.fuel_decrease_per_interval / data.distance_interval) * x) ∧
  (data.initial_fuel - (data.fuel_decrease_per_interval / data.distance_interval) * data.total_distance = 5) :=
by sorry


end NUMINAMATH_CALUDE_fuel_consumption_statements_correct_l2214_221487


namespace NUMINAMATH_CALUDE_evaluate_expression_l2214_221497

theorem evaluate_expression : 11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2214_221497


namespace NUMINAMATH_CALUDE_f_increasing_when_x_gt_1_l2214_221481

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^2

-- State the theorem
theorem f_increasing_when_x_gt_1 :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_f_increasing_when_x_gt_1_l2214_221481


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l2214_221490

/-- The function f(x) = a^(2-x) + 2 always passes through the point (2, 3) for all a > 0 and a ≠ 1 -/
theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(2 - x) + 2
  f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l2214_221490
