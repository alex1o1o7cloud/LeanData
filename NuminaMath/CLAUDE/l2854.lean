import Mathlib

namespace NUMINAMATH_CALUDE_basketball_wins_l2854_285475

theorem basketball_wins (x : ℚ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : x + (5/8)*x + (x + (5/8)*x) = 130) : x = 40 := by
  sorry

end NUMINAMATH_CALUDE_basketball_wins_l2854_285475


namespace NUMINAMATH_CALUDE_sin_double_angle_with_tan_two_l2854_285459

theorem sin_double_angle_with_tan_two (θ : Real) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_with_tan_two_l2854_285459


namespace NUMINAMATH_CALUDE_sqrt_product_and_difference_of_squares_l2854_285409

theorem sqrt_product_and_difference_of_squares :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y) ∧
  (∀ a b : ℝ, (a + b) * (a - b) = a^2 - b^2) ∧
  (Real.sqrt 3 * Real.sqrt 27 = 9) ∧
  ((Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) - (Real.sqrt 3 - 2)^2 = 4 * Real.sqrt 3 - 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_and_difference_of_squares_l2854_285409


namespace NUMINAMATH_CALUDE_total_chocolate_bars_l2854_285438

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 16

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 25

/-- The total number of chocolate bars in the large box -/
def total_bars : ℕ := num_small_boxes * bars_per_small_box

theorem total_chocolate_bars : total_bars = 400 := by
  sorry

end NUMINAMATH_CALUDE_total_chocolate_bars_l2854_285438


namespace NUMINAMATH_CALUDE_thousand_pow_ten_zeros_l2854_285445

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- 1000 is equal to 10^3 -/
axiom thousand_eq_ten_cubed : (1000 : ℕ) = 10^3

/-- The number of trailing zeros in 1000^10 is 30 -/
theorem thousand_pow_ten_zeros : trailingZeros (1000^10) = 30 := by sorry

end NUMINAMATH_CALUDE_thousand_pow_ten_zeros_l2854_285445


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2854_285433

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (a * x) / (x - 1) < (a - 1) / (x - 1)) ↔
  (a > 0 ∧ (a - 1) / a < x ∧ x < 1) ∨
  (a = 0 ∧ x < 1) ∨
  (a < 0 ∧ (x > (a - 1) / a ∨ x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2854_285433


namespace NUMINAMATH_CALUDE_principal_is_8000_l2854_285423

/-- The principal amount that satisfies the given compound interest conditions -/
def find_principal : ℝ := by sorry

/-- The annual interest rate -/
def interest_rate : ℝ := by sorry

theorem principal_is_8000 :
  find_principal = 8000 ∧
  find_principal * (1 + interest_rate)^2 = 8820 ∧
  find_principal * (1 + interest_rate)^3 = 9261 := by sorry

end NUMINAMATH_CALUDE_principal_is_8000_l2854_285423


namespace NUMINAMATH_CALUDE_y_value_l2854_285412

theorem y_value (y : ℚ) (h : (1 : ℚ) / 3 - (1 : ℚ) / 4 = 4 / y) : y = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2854_285412


namespace NUMINAMATH_CALUDE_equation_solution_l2854_285479

theorem equation_solution : 
  ∃ (S : Set ℝ), S = {x : ℝ | (x + 2)^4 + x^4 = 82} ∧ S = {-3, 1} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2854_285479


namespace NUMINAMATH_CALUDE_winston_remaining_cents_l2854_285434

/-- The number of cents in a quarter -/
def cents_per_quarter : ℕ := 25

/-- The number of cents in half a dollar -/
def half_dollar_cents : ℕ := 50

/-- The number of quarters Winston has -/
def winston_quarters : ℕ := 14

theorem winston_remaining_cents : 
  winston_quarters * cents_per_quarter - half_dollar_cents = 300 := by
  sorry

end NUMINAMATH_CALUDE_winston_remaining_cents_l2854_285434


namespace NUMINAMATH_CALUDE_pictures_per_album_l2854_285402

theorem pictures_per_album 
  (phone_pics : ℕ) 
  (camera_pics : ℕ) 
  (num_albums : ℕ) 
  (h1 : phone_pics = 35) 
  (h2 : camera_pics = 5) 
  (h3 : num_albums = 5) : 
  (phone_pics + camera_pics) / num_albums = 8 := by
  sorry

end NUMINAMATH_CALUDE_pictures_per_album_l2854_285402


namespace NUMINAMATH_CALUDE_brick_height_proof_l2854_285470

/-- Proves that given a wall of specific dimensions and bricks of specific dimensions,
    if a certain number of bricks are used, then the height of each brick is 6 cm. -/
theorem brick_height_proof (wall_length wall_height wall_thickness : ℝ)
                           (brick_length brick_width brick_height : ℝ)
                           (num_bricks : ℝ) :
  wall_length = 8 →
  wall_height = 6 →
  wall_thickness = 0.02 →
  brick_length = 0.05 →
  brick_width = 0.11 →
  brick_height = 0.06 →
  num_bricks = 2909.090909090909 →
  brick_height * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_brick_height_proof_l2854_285470


namespace NUMINAMATH_CALUDE_annas_vegetable_patch_area_l2854_285480

/-- Represents a rectangular enclosure with fence posts -/
structure FencedRectangle where
  total_posts : ℕ
  post_spacing : ℝ
  long_side_post_ratio : ℕ

/-- Calculates the area of a fenced rectangle -/
def calculate_area (fence : FencedRectangle) : ℝ :=
  let short_side_posts := (fence.total_posts + 4) / (2 * (fence.long_side_post_ratio + 1))
  let long_side_posts := fence.long_side_post_ratio * short_side_posts
  let short_side_length := (short_side_posts - 1) * fence.post_spacing
  let long_side_length := (long_side_posts - 1) * fence.post_spacing
  short_side_length * long_side_length

/-- Theorem stating that the area of Anna's vegetable patch is 144 square meters -/
theorem annas_vegetable_patch_area :
  let fence := FencedRectangle.mk 24 3 3
  calculate_area fence = 144 := by sorry

end NUMINAMATH_CALUDE_annas_vegetable_patch_area_l2854_285480


namespace NUMINAMATH_CALUDE_cell_growth_10_days_l2854_285474

/-- Calculates the number of cells after a given number of days, 
    starting with an initial population that doubles every two days. -/
def cell_population (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells * 2^(days / 2)

/-- Theorem stating that given 4 initial cells that double every two days for 10 days, 
    the final number of cells is 64. -/
theorem cell_growth_10_days : cell_population 4 10 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cell_growth_10_days_l2854_285474


namespace NUMINAMATH_CALUDE_room_width_proof_l2854_285449

/-- Given a rectangular room with known length, paving cost, and paving rate per square meter,
    prove that the width of the room is 2.75 meters. -/
theorem room_width_proof (length : ℝ) (paving_cost : ℝ) (paving_rate : ℝ) :
  length = 6.5 →
  paving_cost = 10725 →
  paving_rate = 600 →
  paving_cost / paving_rate / length = 2.75 := by
  sorry


end NUMINAMATH_CALUDE_room_width_proof_l2854_285449


namespace NUMINAMATH_CALUDE_probability_at_least_two_green_is_one_third_l2854_285428

/-- The probability of selecting at least two green apples when randomly choosing
    3 apples from a set of 10 apples, where 4 are green and 6 are red. -/
def probability_at_least_two_green : ℚ :=
  let total_apples : ℕ := 10
  let green_apples : ℕ := 4
  let red_apples : ℕ := 6
  let chosen_apples : ℕ := 3
  let total_ways := Nat.choose total_apples chosen_apples
  let ways_two_green := Nat.choose green_apples 2 * Nat.choose red_apples 1
  let ways_three_green := Nat.choose green_apples 3
  (ways_two_green + ways_three_green : ℚ) / total_ways

theorem probability_at_least_two_green_is_one_third :
  probability_at_least_two_green = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_green_is_one_third_l2854_285428


namespace NUMINAMATH_CALUDE_john_total_spend_l2854_285419

def calculate_total_spend (tshirt_price : ℝ) (tshirt_count : ℕ) (pants_price : ℝ) (pants_count : ℕ)
  (jacket_price : ℝ) (jacket_discount : ℝ) (hat_price : ℝ) (shoes_price : ℝ) (shoes_discount : ℝ)
  (clothes_tax_rate : ℝ) (shoes_tax_rate : ℝ) : ℝ :=
  let tshirt_total := tshirt_price * 2 + tshirt_price * 0.5
  let pants_total := pants_price * pants_count
  let jacket_total := jacket_price * (1 - jacket_discount)
  let shoes_total := shoes_price * (1 - shoes_discount)
  let clothes_subtotal := tshirt_total + pants_total + jacket_total + hat_price
  let total_before_tax := clothes_subtotal + shoes_total
  let clothes_tax := clothes_subtotal * clothes_tax_rate
  let shoes_tax := shoes_total * shoes_tax_rate
  total_before_tax + clothes_tax + shoes_tax

theorem john_total_spend :
  calculate_total_spend 20 3 50 2 80 0.25 15 60 0.1 0.05 0.08 = 294.57 := by
  sorry

end NUMINAMATH_CALUDE_john_total_spend_l2854_285419


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l2854_285452

theorem max_digits_product_5_4 : 
  ∀ a b : ℕ, 
  10000 ≤ a ∧ a < 100000 → 
  1000 ≤ b ∧ b < 10000 → 
  a * b < 1000000000 :=
by sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l2854_285452


namespace NUMINAMATH_CALUDE_square_field_area_l2854_285464

/-- Given a square field where a horse takes 10 hours to run around it at a speed of 16 km/h, 
    the area of the field is 1600 square kilometers. -/
theorem square_field_area (s : ℝ) : 
  s > 0 → -- s is positive (side length of square)
  (4 * s = 16 * 10) → -- perimeter equals distance traveled by horse
  s^2 = 1600 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l2854_285464


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_l2854_285411

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_increasing_iff (a : ℕ → ℝ) 
  (h : is_geometric_sequence a) :
  (a 1 < a 2 ∧ a 2 < a 3) ↔ is_increasing_sequence a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_l2854_285411


namespace NUMINAMATH_CALUDE_max_correct_answers_l2854_285482

/-- Represents the result of a math exam -/
structure ExamResult where
  totalQuestions : ℕ
  correctAnswers : ℕ
  wrongAnswers : ℕ
  unansweredQuestions : ℕ
  score : ℤ

/-- Calculates the score based on the exam rules -/
def calculateScore (result : ExamResult) : ℤ :=
  result.correctAnswers - 3 * result.wrongAnswers - 2 * result.unansweredQuestions

/-- Checks if the exam result is valid according to the given conditions -/
def isValidExamResult (result : ExamResult) : Prop :=
  result.totalQuestions = 100 ∧
  result.correctAnswers + result.wrongAnswers + result.unansweredQuestions = result.totalQuestions ∧
  calculateScore result = 50

/-- Theorem stating that the maximum number of correct answers is 87 -/
theorem max_correct_answers (result : ExamResult) :
  isValidExamResult result → result.correctAnswers ≤ 87 := by
  sorry

#check max_correct_answers

end NUMINAMATH_CALUDE_max_correct_answers_l2854_285482


namespace NUMINAMATH_CALUDE_power_mod_remainder_l2854_285420

theorem power_mod_remainder : 6^50 % 215 = 36 := by sorry

end NUMINAMATH_CALUDE_power_mod_remainder_l2854_285420


namespace NUMINAMATH_CALUDE_mowers_for_three_hours_l2854_285436

/-- The number of mowers required to drink a barrel of kvass in a given time -/
def mowers_required (initial_mowers : ℕ) (initial_hours : ℕ) (target_hours : ℕ) : ℕ :=
  (initial_mowers * initial_hours) / target_hours

/-- Theorem stating that 16 mowers are required to drink a barrel of kvass in 3 hours,
    given that 6 mowers can drink it in 8 hours -/
theorem mowers_for_three_hours :
  mowers_required 6 8 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_mowers_for_three_hours_l2854_285436


namespace NUMINAMATH_CALUDE_triangle_side_expression_zero_l2854_285444

theorem triangle_side_expression_zero (a b c : ℝ) 
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  |a - b - c| - |c - a + b| = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_expression_zero_l2854_285444


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2854_285451

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let ellipse_eccentricity := Real.sqrt (1 - (b / a) ^ 2)
  let hyperbola_eccentricity := Real.sqrt (1 + (b / a) ^ 2)
  ellipse_eccentricity = Real.sqrt 3 / 2 →
  hyperbola_eccentricity = Real.sqrt 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2854_285451


namespace NUMINAMATH_CALUDE_no_roots_in_larger_interval_l2854_285442

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the property of having exactly one root
def has_unique_root (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

-- Define the property of a root being within an open interval
def root_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ f x = 0

-- Theorem statement
theorem no_roots_in_larger_interval
  (h_unique : has_unique_root f)
  (h_16 : root_in_interval f 0 16)
  (h_8 : root_in_interval f 0 8)
  (h_4 : root_in_interval f 0 4)
  (h_2 : root_in_interval f 0 2) :
  ∀ x ∈ Set.Icc 2 16, f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_no_roots_in_larger_interval_l2854_285442


namespace NUMINAMATH_CALUDE_g_is_even_l2854_285429

-- Define a function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g as f(x) + f(-x)
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem: g is an even function
theorem g_is_even : ∀ x : ℝ, g f x = g f (-x) := by sorry

end NUMINAMATH_CALUDE_g_is_even_l2854_285429


namespace NUMINAMATH_CALUDE_subway_passenger_decrease_l2854_285425

theorem subway_passenger_decrease (initial : ℕ) (got_off : ℕ) (got_on : ℕ)
  (h1 : initial = 35)
  (h2 : got_off = 18)
  (h3 : got_on = 15) :
  initial - (initial - got_off + got_on) = 3 :=
by sorry

end NUMINAMATH_CALUDE_subway_passenger_decrease_l2854_285425


namespace NUMINAMATH_CALUDE_snow_probability_l2854_285465

theorem snow_probability (p1 p2 : ℝ) (h1 : p1 = 1/4) (h2 : p2 = 1/3) :
  let prob_no_snow := (1 - p1)^4 * (1 - p2)^3
  1 - prob_no_snow = 29/32 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l2854_285465


namespace NUMINAMATH_CALUDE_wayne_shrimp_appetizer_cost_l2854_285467

/-- Calculates the total cost of shrimp for an appetizer given the number of shrimp per guest,
    number of guests, cost per pound of shrimp, and number of shrimp per pound. -/
def shrimp_appetizer_cost (shrimp_per_guest : ℕ) (num_guests : ℕ) (cost_per_pound : ℚ) (shrimp_per_pound : ℕ) : ℚ :=
  (shrimp_per_guest * num_guests : ℚ) / shrimp_per_pound * cost_per_pound

/-- Proves that Wayne's shrimp appetizer will cost $170 given the specified conditions. -/
theorem wayne_shrimp_appetizer_cost :
  shrimp_appetizer_cost 5 40 17 20 = 170 := by
  sorry

end NUMINAMATH_CALUDE_wayne_shrimp_appetizer_cost_l2854_285467


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2854_285437

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given quadratic equation x^2 - 6x + 2m - 1 = 0 with two real roots -/
def givenEquation (m : ℝ) : QuadraticEquation :=
  { a := 1, b := -6, c := 2 * m - 1 }

theorem quadratic_roots_theorem (m : ℝ) :
  let eq := givenEquation m
  let x₁ : ℝ := 1
  let x₂ : ℝ := 5
  (∃ (x₁ x₂ : ℝ), x₁^2 - 6*x₁ + 2*m - 1 = 0 ∧ x₂^2 - 6*x₂ + 2*m - 1 = 0) →
  x₁ = 1 →
  (x₂ = 5 ∧ m = 3) ∧
  (∃ m' : ℝ, (x₁ - 1) * (x₂ - 1) = 6 / (m' - 5) ∧ m' = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2854_285437


namespace NUMINAMATH_CALUDE_hall_ratio_l2854_285499

/-- Given a rectangular hall with area 578 sq. m and difference between length and width 17 m,
    prove that the ratio of width to length is 1:2 -/
theorem hall_ratio (w l : ℝ) (hw : w > 0) (hl : l > 0) : 
  w * l = 578 → l - w = 17 → w / l = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hall_ratio_l2854_285499


namespace NUMINAMATH_CALUDE_number_of_subsets_A_l2854_285481

def U : Finset ℕ := {0, 1, 2}

theorem number_of_subsets_A (A : Finset ℕ) (h : U \ A = {2}) : Finset.card (Finset.powerset A) = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_subsets_A_l2854_285481


namespace NUMINAMATH_CALUDE_charles_pictures_l2854_285478

theorem charles_pictures (total_papers : ℕ) (pictures_before_work : ℕ) (pictures_after_work : ℕ) (papers_left : ℕ) :
  let total_yesterday := pictures_before_work + pictures_after_work
  let used_papers := total_papers - papers_left
  used_papers - total_yesterday = total_papers - papers_left - (pictures_before_work + pictures_after_work) :=
by sorry

end NUMINAMATH_CALUDE_charles_pictures_l2854_285478


namespace NUMINAMATH_CALUDE_julian_frederick_age_difference_l2854_285466

/-- Given the ages of Kyle, Julian, Frederick, and Tyson, prove that Julian is 20 years younger than Frederick. -/
theorem julian_frederick_age_difference :
  ∀ (kyle_age julian_age frederick_age tyson_age : ℕ),
  kyle_age = julian_age + 5 →
  frederick_age > julian_age →
  frederick_age = 2 * tyson_age →
  tyson_age = 20 →
  kyle_age = 25 →
  frederick_age - julian_age = 20 :=
by sorry

end NUMINAMATH_CALUDE_julian_frederick_age_difference_l2854_285466


namespace NUMINAMATH_CALUDE_good_couples_parity_l2854_285468

/-- Represents the color of a grid on the chess board -/
inductive Color
| Red
| Blue

/-- Converts a Color to an integer label -/
def color_to_label (c : Color) : Int :=
  match c with
  | Color.Red => 1
  | Color.Blue => -1

/-- Represents a chess board with m rows and n columns -/
structure ChessBoard (m n : Nat) where
  grid : Fin m → Fin n → Color
  m_ge_three : m ≥ 3
  n_ge_three : n ≥ 3

/-- Counts the number of "good couples" on the chess board -/
def count_good_couples (board : ChessBoard m n) : Nat :=
  sorry

/-- Calculates the product of labels for border grids (excluding corners) -/
def border_product (board : ChessBoard m n) : Int :=
  sorry

/-- Main theorem: The parity of good couples is determined by the border product -/
theorem good_couples_parity (m n : Nat) (board : ChessBoard m n) :
  Even (count_good_couples board) ↔ border_product board = 1 :=
  sorry

end NUMINAMATH_CALUDE_good_couples_parity_l2854_285468


namespace NUMINAMATH_CALUDE_shifted_line_equation_l2854_285471

/-- Represents a line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line vertically by a given amount -/
def vertical_shift (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift }

/-- The theorem stating that shifting y = -2x up by 5 units results in y = -2x + 5 -/
theorem shifted_line_equation :
  let original_line : Line := { slope := -2, intercept := 0 }
  let shifted_line := vertical_shift original_line 5
  shifted_line = { slope := -2, intercept := 5 } := by sorry

end NUMINAMATH_CALUDE_shifted_line_equation_l2854_285471


namespace NUMINAMATH_CALUDE_regression_correction_l2854_285443

/-- Represents a data point with x and y coordinates -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Represents a linear regression equation -/
structure RegressionEquation where
  slope : ℝ
  intercept : ℝ

/-- Represents the center of sample points -/
structure SampleCenter where
  x : ℝ
  y : ℝ

theorem regression_correction (data : List DataPoint) 
  (initial_eq : RegressionEquation) 
  (initial_center : SampleCenter)
  (incorrect_point1 incorrect_point2 correct_point1 correct_point2 : DataPoint)
  (corrected_slope : ℝ)
  (h1 : data.length = 8)
  (h2 : initial_eq.slope = 2 ∧ initial_eq.intercept = 5)
  (h3 : initial_center.x = 2)
  (h4 : incorrect_point1 = ⟨7, 3⟩ ∧ correct_point1 = ⟨3, 7⟩)
  (h5 : incorrect_point2 = ⟨4, -6⟩ ∧ correct_point2 = ⟨4, 6⟩)
  (h6 : corrected_slope = 13/3) :
  ∃ k : ℝ, k = 9/2 ∧ 
    ∀ x y : ℝ, y = corrected_slope * x + k → 
      ∃ center : SampleCenter, center.x = 3/2 ∧ center.y = 11 ∧
        y = corrected_slope * center.x + k := by
  sorry

end NUMINAMATH_CALUDE_regression_correction_l2854_285443


namespace NUMINAMATH_CALUDE_box_dimensions_l2854_285400

/-- A box with a square base and specific ribbon tying properties has dimensions 22 cm × 22 cm × 11 cm -/
theorem box_dimensions (s : ℝ) (b : ℝ) :
  s > 0 →
  6 * s + b = 156 →
  7 * s + b = 178 →
  s = 22 ∧ s / 2 = 11 :=
by sorry

end NUMINAMATH_CALUDE_box_dimensions_l2854_285400


namespace NUMINAMATH_CALUDE_increase_in_average_commission_l2854_285414

/-- Calculates the increase in average commission after a big sale -/
theorem increase_in_average_commission 
  (big_sale_commission : ℕ) 
  (new_average_commission : ℕ) 
  (total_sales : ℕ) 
  (h1 : big_sale_commission = 1000)
  (h2 : new_average_commission = 250)
  (h3 : total_sales = 6) :
  new_average_commission - (new_average_commission * total_sales - big_sale_commission) / (total_sales - 1) = 150 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_average_commission_l2854_285414


namespace NUMINAMATH_CALUDE_monotonic_increasing_iff_m_ge_four_thirds_l2854_285441

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

-- State the theorem
theorem monotonic_increasing_iff_m_ge_four_thirds (m : ℝ) :
  (∀ x : ℝ, Monotone (f m)) ↔ m ≥ 4/3 := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_iff_m_ge_four_thirds_l2854_285441


namespace NUMINAMATH_CALUDE_gcd_1987_2025_l2854_285426

theorem gcd_1987_2025 : Nat.gcd 1987 2025 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1987_2025_l2854_285426


namespace NUMINAMATH_CALUDE_tennis_racket_packaging_l2854_285410

/-- Given information about tennis racket packaging, prove the number of rackets in the other carton type. -/
theorem tennis_racket_packaging (total_cartons : ℕ) (total_rackets : ℕ) (three_racket_cartons : ℕ) 
  (h1 : total_cartons = 38)
  (h2 : total_rackets = 100)
  (h3 : three_racket_cartons = 24)
  : ∃ (other_carton_size : ℕ), 
    other_carton_size * (total_cartons - three_racket_cartons) + 3 * three_racket_cartons = total_rackets ∧ 
    other_carton_size = 2 :=
by sorry

end NUMINAMATH_CALUDE_tennis_racket_packaging_l2854_285410


namespace NUMINAMATH_CALUDE_steven_peach_apple_difference_l2854_285460

-- Define the number of peaches and apples Steven has
def steven_peaches : ℕ := 18
def steven_apples : ℕ := 11

-- Theorem to prove the difference between peaches and apples
theorem steven_peach_apple_difference :
  steven_peaches - steven_apples = 7 := by
  sorry

end NUMINAMATH_CALUDE_steven_peach_apple_difference_l2854_285460


namespace NUMINAMATH_CALUDE_g_difference_l2854_285457

theorem g_difference (x h : ℝ) : 
  let g := λ (t : ℝ) => 3 * t^3 - 4 * t + 5
  g (x + h) - g x = h * (9 * x^2 + 9 * x * h + 3 * h^2 - 4) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l2854_285457


namespace NUMINAMATH_CALUDE_value_of_a_l2854_285421

theorem value_of_a (a : ℝ) (h : a = -a) : a = 0 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2854_285421


namespace NUMINAMATH_CALUDE_leaves_blown_away_l2854_285407

theorem leaves_blown_away (initial_leaves left_leaves : ℕ) 
  (h1 : initial_leaves = 5678)
  (h2 : left_leaves = 1432) :
  initial_leaves - left_leaves = 4246 := by
  sorry

end NUMINAMATH_CALUDE_leaves_blown_away_l2854_285407


namespace NUMINAMATH_CALUDE_inverted_sand_height_is_25_l2854_285488

/-- Represents the container with frustum and cylinder components -/
structure Container where
  radius : ℝ
  frustumHeight : ℝ
  cylinderHeight : ℝ
  cylinderFillHeight : ℝ

/-- Calculates the total height of sand when the container is inverted -/
def invertedSandHeight (c : Container) : ℝ :=
  c.frustumHeight + c.cylinderFillHeight

/-- Theorem stating the height of sand when the container is inverted -/
theorem inverted_sand_height_is_25 (c : Container) 
  (h_radius : c.radius = 12)
  (h_frustum_height : c.frustumHeight = 20)
  (h_cylinder_height : c.cylinderHeight = 20)
  (h_cylinder_fill : c.cylinderFillHeight = 5) :
  invertedSandHeight c = 25 := by
  sorry

#check inverted_sand_height_is_25

end NUMINAMATH_CALUDE_inverted_sand_height_is_25_l2854_285488


namespace NUMINAMATH_CALUDE_base4_calculation_l2854_285496

/-- Converts a base 4 number to base 10 --/
def base4_to_base10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 --/
def base10_to_base4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Theorem: In base 4, (1230₄ + 32₄) ÷ 13₄ = 111₄ --/
theorem base4_calculation : 
  let a := base4_to_base10 [0, 3, 2, 1]  -- 1230₄
  let b := base4_to_base10 [2, 3]        -- 32₄
  let c := base4_to_base10 [3, 1]        -- 13₄
  base10_to_base4 ((a + b) / c) = [1, 1, 1] := by
  sorry


end NUMINAMATH_CALUDE_base4_calculation_l2854_285496


namespace NUMINAMATH_CALUDE_base7_subtraction_theorem_l2854_285403

/-- Represents a number in base 7 --/
def Base7 : Type := ℕ

/-- Converts a base 7 number to its decimal representation --/
def to_decimal (n : Base7) : ℕ := sorry

/-- Converts a decimal number to its base 7 representation --/
def from_decimal (n : ℕ) : Base7 := sorry

/-- Subtracts two base 7 numbers --/
def base7_subtract (a b : Base7) : Base7 := sorry

theorem base7_subtraction_theorem :
  base7_subtract (from_decimal 4321) (from_decimal 1234) = from_decimal 3054 := by
  sorry

end NUMINAMATH_CALUDE_base7_subtraction_theorem_l2854_285403


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l2854_285489

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- The number to be converted to scientific notation -/
def original_number : ℕ := 189130000000

/-- Function to convert a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem scientific_notation_correct :
  let sn := to_scientific_notation original_number
  sn.coefficient = 1.8913 ∧ sn.exponent = 11 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l2854_285489


namespace NUMINAMATH_CALUDE_mean_equality_implies_sum_l2854_285417

theorem mean_equality_implies_sum (y z : ℝ) : 
  (8 + 15 + 21) / 3 = (14 + y + z) / 3 → y + z = 30 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_sum_l2854_285417


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2854_285487

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 + 3*x + 1 < 0) ↔ (∀ x : ℝ, x > 0 → x^2 + 3*x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2854_285487


namespace NUMINAMATH_CALUDE_predicted_value_theorem_l2854_285477

/-- A linear regression model with given slope and sample centroid -/
structure LinearRegressionModel where
  slope : ℝ
  centroid_x : ℝ
  centroid_y : ℝ

/-- Calculate the predicted value of the dependent variable -/
def predict (model : LinearRegressionModel) (x : ℝ) : ℝ :=
  let intercept := model.centroid_y - model.slope * model.centroid_x
  model.slope * x + intercept

theorem predicted_value_theorem (model : LinearRegressionModel) 
  (h1 : model.slope = 1.23)
  (h2 : model.centroid_x = 4)
  (h3 : model.centroid_y = 5)
  (x : ℝ)
  (h4 : x = 10) :
  predict model x = 12.38 := by
  sorry

end NUMINAMATH_CALUDE_predicted_value_theorem_l2854_285477


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2854_285472

-- Define sets M and N
def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2854_285472


namespace NUMINAMATH_CALUDE_complex_inequality_complex_inequality_equality_condition_l2854_285458

theorem complex_inequality (z : ℂ) (h : Complex.abs z ≥ 1) :
  (Complex.abs (2 * z - 1))^5 / (25 * Real.sqrt 5) ≥ (Complex.abs (z - 1))^4 / 4 :=
by sorry

theorem complex_inequality_equality_condition (z : ℂ) :
  (Complex.abs (2 * z - 1))^5 / (25 * Real.sqrt 5) = (Complex.abs (z - 1))^4 / 4 ↔
  z = Complex.I ∨ z = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_inequality_complex_inequality_equality_condition_l2854_285458


namespace NUMINAMATH_CALUDE_floss_per_student_l2854_285450

/-- Proves that each student needs 5 yards of floss given the problem conditions -/
theorem floss_per_student 
  (num_students : ℕ) 
  (floss_per_packet : ℕ) 
  (leftover_floss : ℕ) 
  (total_floss : ℕ) :
  num_students = 20 →
  floss_per_packet = 35 →
  leftover_floss = 5 →
  total_floss = num_students * (total_floss / num_students) →
  total_floss % floss_per_packet = 0 →
  total_floss / num_students = 5 := by
sorry

end NUMINAMATH_CALUDE_floss_per_student_l2854_285450


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2854_285455

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  n > 0 ∧ exterior_angle = 15 → n * exterior_angle = 360 → n = 24 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2854_285455


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l2854_285427

theorem geometric_series_first_term (a r : ℝ) (h1 : r ≠ 1) (h2 : |r| < 1) : 
  (a / (1 - r) = 12) → ((a^2) / (1 - r^2) = 36) → a = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l2854_285427


namespace NUMINAMATH_CALUDE_all_terms_irrational_l2854_285497

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define the property of √2 and √3 being in the sequence
def sqrt2_sqrt3_in_sequence (a : ℕ → ℝ) : Prop :=
  ∃ m n : ℕ, a m = Real.sqrt 2 ∧ a n = Real.sqrt 3

-- Theorem statement
theorem all_terms_irrational
  (a : ℕ → ℝ)
  (h1 : is_arithmetic_sequence a)
  (h2 : sqrt2_sqrt3_in_sequence a) :
  ∀ n : ℕ, Irrational (a n) :=
sorry

end NUMINAMATH_CALUDE_all_terms_irrational_l2854_285497


namespace NUMINAMATH_CALUDE_fraction_of_employees_laid_off_l2854_285486

/-- Proves that the fraction of employees laid off is 1/3 given the initial conditions -/
theorem fraction_of_employees_laid_off 
  (initial_employees : ℕ) 
  (salary_per_employee : ℕ) 
  (total_paid_after_layoff : ℕ) 
  (h1 : initial_employees = 450)
  (h2 : salary_per_employee = 2000)
  (h3 : total_paid_after_layoff = 600000) :
  (initial_employees * salary_per_employee - total_paid_after_layoff) / (initial_employees * salary_per_employee) = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_of_employees_laid_off_l2854_285486


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2854_285432

-- Define the hyperbola
def hyperbola (m : ℤ) (x y : ℝ) : Prop :=
  x^2 / m^2 + y^2 / (m^2 - 4) = 1

-- Define the eccentricity
def eccentricity (m : ℤ) : ℝ :=
  2

-- Theorem statement
theorem hyperbola_eccentricity (m : ℤ) :
  ∃ (e : ℝ), e = eccentricity m ∧ 
  ∀ (x y : ℝ), hyperbola m x y → e = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2854_285432


namespace NUMINAMATH_CALUDE_club_members_count_l2854_285473

theorem club_members_count : ∃! n : ℕ, 
  200 ≤ n ∧ n ≤ 300 ∧ 
  (n + 4) % 10 = 0 ∧ 
  (n + 5) % 11 = 0 ∧ 
  n = 226 := by
  sorry

end NUMINAMATH_CALUDE_club_members_count_l2854_285473


namespace NUMINAMATH_CALUDE_number_of_lineups_is_4290_l2854_285493

/-- Represents the total number of players in the team -/
def total_players : ℕ := 15

/-- Represents the number of players in a starting lineup -/
def lineup_size : ℕ := 6

/-- Represents the number of players who refuse to play together -/
def incompatible_players : ℕ := 2

/-- Calculates the number of possible starting lineups -/
def number_of_lineups : ℕ :=
  let remaining_players := total_players - incompatible_players
  Nat.choose remaining_players (lineup_size - 1) * 2 +
  Nat.choose remaining_players lineup_size

/-- Theorem stating that the number of possible lineups is 4290 -/
theorem number_of_lineups_is_4290 :
  number_of_lineups = 4290 := by sorry

end NUMINAMATH_CALUDE_number_of_lineups_is_4290_l2854_285493


namespace NUMINAMATH_CALUDE_three_two_zero_zero_properties_l2854_285424

/-- Represents a number with its decimal representation -/
structure DecimalNumber where
  value : ℝ
  representation : String

/-- Counts the number of significant figures in a decimal representation -/
def countSignificantFigures (n : DecimalNumber) : ℕ :=
  sorry

/-- Determines the precision of a decimal representation -/
def getPrecision (n : DecimalNumber) : String :=
  sorry

/-- The main theorem about the number 3200 -/
theorem three_two_zero_zero_properties :
  let n : DecimalNumber := ⟨3200, "0.320"⟩
  countSignificantFigures n = 3 ∧ getPrecision n = "thousandth" := by
  sorry

end NUMINAMATH_CALUDE_three_two_zero_zero_properties_l2854_285424


namespace NUMINAMATH_CALUDE_abby_peeled_22_l2854_285494

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  abby_rate : ℕ
  homer_solo_time : ℕ

/-- Calculates the number of potatoes Abby peeled -/
def abby_peeled (scenario : PotatoPeeling) : ℕ :=
  let homer_solo := scenario.homer_rate * scenario.homer_solo_time
  let remaining := scenario.total_potatoes - homer_solo
  let combined_rate := scenario.homer_rate + scenario.abby_rate
  let combined_time := remaining / combined_rate
  scenario.abby_rate * combined_time

/-- The main theorem stating that Abby peeled 22 potatoes -/
theorem abby_peeled_22 (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 60)
  (h2 : scenario.homer_rate = 4)
  (h3 : scenario.abby_rate = 6)
  (h4 : scenario.homer_solo_time = 6) :
  abby_peeled scenario = 22 := by
  sorry

end NUMINAMATH_CALUDE_abby_peeled_22_l2854_285494


namespace NUMINAMATH_CALUDE_deck_size_proof_l2854_285448

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b) = 1/4 →
  (r : ℚ) / (r + b + 6) = 1/5 →
  r + b = 24 := by
sorry

end NUMINAMATH_CALUDE_deck_size_proof_l2854_285448


namespace NUMINAMATH_CALUDE_arithmetic_mean_theorem_l2854_285404

theorem arithmetic_mean_theorem (x a b : ℝ) (hx : x ≠ 0) (hb : b ≠ 0) :
  (1 / 2) * ((x * b + a) / x + (x * b - a) / x) = b := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_theorem_l2854_285404


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_theorem_l2854_285492

/-- A fourth-degree polynomial with real coefficients -/
def FourthDegreePolynomial : Type := ℝ → ℝ

/-- The condition that |g(x)| = 6 for x = 0, 1, 3, 4, 5 -/
def SatisfiesCondition (g : FourthDegreePolynomial) : Prop :=
  |g 0| = 6 ∧ |g 1| = 6 ∧ |g 3| = 6 ∧ |g 4| = 6 ∧ |g 5| = 6

theorem fourth_degree_polynomial_theorem (g : FourthDegreePolynomial) 
  (h : SatisfiesCondition g) : |g 7| = 106.8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_degree_polynomial_theorem_l2854_285492


namespace NUMINAMATH_CALUDE_rational_function_characterization_l2854_285440

theorem rational_function_characterization (f : ℚ → ℚ) 
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 := by sorry

end NUMINAMATH_CALUDE_rational_function_characterization_l2854_285440


namespace NUMINAMATH_CALUDE_train_speed_l2854_285435

/-- Proves that a train of given length crossing a bridge of given length in a given time has a specific speed in km/hr -/
theorem train_speed (train_length bridge_length : Real) (crossing_time : Real) : 
  train_length = 150 ∧ 
  bridge_length = 225 ∧ 
  crossing_time = 30 → 
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2854_285435


namespace NUMINAMATH_CALUDE_distance_between_vertices_l2854_285430

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the vertices of the hyperbola
def vertices : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Theorem: The distance between the vertices of the hyperbola is 8
theorem distance_between_vertices :
  ∀ (v1 v2 : ℝ × ℝ), v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 →
  Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l2854_285430


namespace NUMINAMATH_CALUDE_forest_to_street_ratio_l2854_285446

/-- The ratio of forest area to street area is 3:1 -/
theorem forest_to_street_ratio : 
  ∀ (street_side_length : ℝ) (trees_per_sq_meter : ℝ) (total_trees : ℝ),
  street_side_length = 100 →
  trees_per_sq_meter = 4 →
  total_trees = 120000 →
  (total_trees / trees_per_sq_meter) / (street_side_length ^ 2) = 3 := by
sorry

end NUMINAMATH_CALUDE_forest_to_street_ratio_l2854_285446


namespace NUMINAMATH_CALUDE_square_probability_is_correct_l2854_285483

/-- The number of squares in a 6x6 grid -/
def gridSize : ℕ := 36

/-- The number of squares to be selected -/
def selectCount : ℕ := 4

/-- The number of ways to select 4 squares from 36 squares -/
def totalSelections : ℕ := Nat.choose gridSize selectCount

/-- The number of ways to select 4 squares that form a square -/
def favorableSelections : ℕ := 105

/-- The probability of selecting 4 squares that form a square -/
def squareProbability : ℚ := favorableSelections / totalSelections

theorem square_probability_is_correct : squareProbability = 1 / 561 := by
  sorry

#eval squareProbability

end NUMINAMATH_CALUDE_square_probability_is_correct_l2854_285483


namespace NUMINAMATH_CALUDE_optimal_seating_l2854_285461

/-- Represents the conference hall seating problem --/
def ConferenceSeating (total_chairs : ℕ) (chairs_per_row : ℕ) (attendees : ℕ) : Prop :=
  ∃ (chairs_to_remove : ℕ),
    let remaining_chairs := total_chairs - chairs_to_remove
    remaining_chairs % chairs_per_row = 0 ∧
    remaining_chairs ≥ attendees ∧
    ∀ (n : ℕ), n < chairs_to_remove →
      (total_chairs - n) % chairs_per_row ≠ 0 ∨
      (total_chairs - n) < attendees ∨
      (total_chairs - n) - attendees > remaining_chairs - attendees

theorem optimal_seating :
  ConferenceSeating 156 13 100 ∧
  (∃ (chairs_to_remove : ℕ), chairs_to_remove = 52 ∧
    let remaining_chairs := 156 - chairs_to_remove
    remaining_chairs % 13 = 0 ∧
    remaining_chairs ≥ 100 ∧
    ∀ (n : ℕ), n < chairs_to_remove →
      (156 - n) % 13 ≠ 0 ∨
      (156 - n) < 100 ∨
      (156 - n) - 100 > remaining_chairs - 100) :=
by sorry

end NUMINAMATH_CALUDE_optimal_seating_l2854_285461


namespace NUMINAMATH_CALUDE_monotonicity_of_F_intersection_property_l2854_285408

noncomputable section

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) : ℝ := x * (Real.log x + 1)

def f' (x : ℝ) : ℝ := Real.log x + 2

def F (x : ℝ) (a : ℝ) : ℝ := a * x^2 + f' x

theorem monotonicity_of_F (x : ℝ) (a : ℝ) (h : x > 0) :
  (a ≥ 0 → StrictMono (F · a)) ∧
  (a < 0 → StrictMonoOn (F · a) (Set.Ioo 0 (Real.sqrt (-1 / (2 * a)))) ∧
           StrictAntiOn (F · a) (Set.Ioi (Real.sqrt (-1 / (2 * a))))) :=
sorry

theorem intersection_property (x₁ x₂ k : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ < x₂) :
  k = (f' x₂ - f' x₁) / (x₂ - x₁) → x₁ < 1 / k ∧ 1 / k < x₂ :=
sorry

end NUMINAMATH_CALUDE_monotonicity_of_F_intersection_property_l2854_285408


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2854_285413

theorem solve_linear_equation :
  ∀ x : ℚ, 3 * x + 8 = -4 * x - 16 → x = -24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2854_285413


namespace NUMINAMATH_CALUDE_louisa_average_speed_l2854_285453

/-- Proves that given the conditions of Louisa's travel, her average speed was 60 miles per hour -/
theorem louisa_average_speed :
  ∀ (v : ℝ), -- v represents the average speed in miles per hour
  v > 0 → -- speed is positive
  ∃ (t : ℝ), -- t represents the time for the 240-mile trip
  t > 0 → -- time is positive
  240 = v * t ∧ -- equation for the first day's travel
  420 = v * (t + 3) → -- equation for the second day's travel
  v = 60 := by
sorry

end NUMINAMATH_CALUDE_louisa_average_speed_l2854_285453


namespace NUMINAMATH_CALUDE_cereal_box_servings_l2854_285456

theorem cereal_box_servings (total_cereal : ℝ) (serving_size : ℝ) (h1 : total_cereal = 24.5) (h2 : serving_size = 1.75) : 
  ⌊total_cereal / serving_size⌋ = 14 := by
sorry

end NUMINAMATH_CALUDE_cereal_box_servings_l2854_285456


namespace NUMINAMATH_CALUDE_augmented_matrix_solution_l2854_285415

theorem augmented_matrix_solution (c₁ c₂ : ℝ) : 
  (∃ (x y : ℝ), 2 * x + 3 * y = c₁ ∧ y = c₂ ∧ x = 3 ∧ y = 5) → c₁ - c₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_augmented_matrix_solution_l2854_285415


namespace NUMINAMATH_CALUDE_perpendicular_necessary_and_sufficient_l2854_285495

/-- A plane -/
structure Plane where
  dummy : Unit

/-- A line in a plane -/
structure Line (α : Plane) where
  dummy : Unit

/-- Predicate for a line being straight -/
def isStraight (α : Plane) (l : Line α) : Prop :=
  sorry

/-- Predicate for a line being oblique -/
def isOblique (α : Plane) (m : Line α) : Prop :=
  sorry

/-- Predicate for two lines being perpendicular -/
def isPerpendicular (α : Plane) (l m : Line α) : Prop :=
  sorry

/-- Theorem stating that for a straight line l and an oblique line m on plane α,
    l being perpendicular to m is both necessary and sufficient -/
theorem perpendicular_necessary_and_sufficient (α : Plane) (l m : Line α)
    (h1 : isStraight α l) (h2 : isOblique α m) :
    isPerpendicular α l m ↔ True :=
  sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_and_sufficient_l2854_285495


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2854_285422

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  arithmetic_sequence a (-2) →
  a 1 + a 4 + a 7 = 50 →
  a 6 + a 9 + a 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2854_285422


namespace NUMINAMATH_CALUDE_negation_of_existence_cubic_inequality_negation_l2854_285406

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem cubic_inequality_negation : 
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_cubic_inequality_negation_l2854_285406


namespace NUMINAMATH_CALUDE_isosceles_triangle_proof_l2854_285431

/-- A triangle is acute-angled if all its angles are less than 90 degrees -/
def IsAcuteAngled (triangle : Set Point) : Prop :=
  sorry

/-- An altitude of a triangle is a line segment from a vertex perpendicular to the opposite side -/
def IsAltitude (segment : Set Point) (triangle : Set Point) : Prop :=
  sorry

/-- The area of a triangle -/
noncomputable def TriangleArea (triangle : Set Point) : ℝ :=
  sorry

/-- A triangle is isosceles if it has at least two equal sides -/
def IsIsosceles (triangle : Set Point) : Prop :=
  sorry

theorem isosceles_triangle_proof (A B C D E : Point) :
  let triangle := {A, B, C}
  IsAcuteAngled triangle →
  IsAltitude {A, D} triangle →
  IsAltitude {B, E} triangle →
  TriangleArea {B, D, E} ≤ TriangleArea {D, E, A} ∧
  TriangleArea {D, E, A} ≤ TriangleArea {E, A, B} ∧
  TriangleArea {E, A, B} ≤ TriangleArea {A, B, D} →
  IsIsosceles triangle :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_proof_l2854_285431


namespace NUMINAMATH_CALUDE_cyclist_distance_l2854_285401

/-- The distance traveled by a cyclist given the conditions of the problem -/
theorem cyclist_distance (distance_AB : ℝ) (pedestrian_speed : ℝ) 
  (h1 : distance_AB = 5)
  (h2 : pedestrian_speed > 0) : 
  let cyclist_speed := 2 * pedestrian_speed
  let time := distance_AB / pedestrian_speed
  cyclist_speed * time = 10 := by sorry

end NUMINAMATH_CALUDE_cyclist_distance_l2854_285401


namespace NUMINAMATH_CALUDE_average_weight_of_children_l2854_285485

/-- The average weight of all children given the weights of boys, girls, and toddlers -/
theorem average_weight_of_children 
  (num_boys : ℕ) (num_girls : ℕ) (num_toddlers : ℕ)
  (avg_weight_boys : ℝ) (avg_weight_girls : ℝ) (avg_weight_toddlers : ℝ)
  (h_num_boys : num_boys = 8)
  (h_num_girls : num_girls = 5)
  (h_num_toddlers : num_toddlers = 3)
  (h_avg_weight_boys : avg_weight_boys = 160)
  (h_avg_weight_girls : avg_weight_girls = 130)
  (h_avg_weight_toddlers : avg_weight_toddlers = 40)
  (h_total_children : num_boys + num_girls + num_toddlers = 16) :
  let total_weight := num_boys * avg_weight_boys + num_girls * avg_weight_girls + num_toddlers * avg_weight_toddlers
  total_weight / (num_boys + num_girls + num_toddlers) = 128.125 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_children_l2854_285485


namespace NUMINAMATH_CALUDE_class_size_quotient_l2854_285469

theorem class_size_quotient (N H J : ℝ) 
  (h1 : N / H = 1.2) 
  (h2 : H / J = 5/6) : 
  N / J = 1 := by
  sorry

end NUMINAMATH_CALUDE_class_size_quotient_l2854_285469


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l2854_285418

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∃ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l2854_285418


namespace NUMINAMATH_CALUDE_no_valid_m_l2854_285463

theorem no_valid_m : ¬ ∃ (m : ℕ+), (∃ (a b : ℕ+), (1806 : ℤ) = a * (m.val ^ 2 - 2) ∧ (1806 : ℤ) = b * (m.val ^ 2 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_m_l2854_285463


namespace NUMINAMATH_CALUDE_ice_cream_volume_l2854_285447

/-- The volume of ice cream in a cone with hemisphere and cylindrical topping -/
theorem ice_cream_volume (h_cone r_cone h_cylinder : ℝ) 
  (h_cone_pos : 0 < h_cone)
  (r_cone_pos : 0 < r_cone)
  (h_cylinder_pos : 0 < h_cylinder)
  (h_cone_val : h_cone = 12)
  (r_cone_val : r_cone = 3)
  (h_cylinder_val : h_cylinder = 2) :
  (1/3 * π * r_cone^2 * h_cone) +  -- Volume of cone
  (2/3 * π * r_cone^3) +           -- Volume of hemisphere
  (π * r_cone^2 * h_cylinder) =    -- Volume of cylinder
  72 * π := by
sorry


end NUMINAMATH_CALUDE_ice_cream_volume_l2854_285447


namespace NUMINAMATH_CALUDE_total_books_l2854_285498

/-- Given an initial number of books and a number of books bought, 
    the total number of books is equal to the sum of the initial number and the number bought. -/
theorem total_books (initial_books bought_books : ℕ) :
  initial_books + bought_books = initial_books + bought_books :=
by sorry

end NUMINAMATH_CALUDE_total_books_l2854_285498


namespace NUMINAMATH_CALUDE_seating_arrangements_eq_144_l2854_285416

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to choose k objects from n objects. -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else factorial n / (factorial k * factorial (n - k))

/-- The number of seating arrangements for 4 students and 2 teachers,
    where teachers cannot sit at either end and must not sit next to each other. -/
def seating_arrangements : ℕ := 
  let student_arrangements := factorial 4
  let teacher_positions := choose 3 2
  let teacher_arrangements := factorial 2
  student_arrangements * teacher_positions * teacher_arrangements

theorem seating_arrangements_eq_144 : seating_arrangements = 144 := by
  sorry

#eval seating_arrangements

end NUMINAMATH_CALUDE_seating_arrangements_eq_144_l2854_285416


namespace NUMINAMATH_CALUDE_nine_expressions_cover_1_to_13_l2854_285484

def nine_expressions : List (ℕ → Prop) :=
  [ (λ n => n = ((9 / 9) ^ (9 - 9))),
    (λ n => n = ((9 / 9) + (9 / 9))),
    (λ n => n = ((9 / 9) + (9 / 9) + (9 / 9))),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 2),
    (λ n => n = ((9 * 9 + 9) / 9 - 9) + 9),
    (λ n => n = ((9 / 9) + (9 / 9) + (9 / 9)) ^ 2 - 3),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 2 - (9 / 9)),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 3 - (9 / 9)),
    (λ n => n = 9),
    (λ n => n = (99 - 9) / 9),
    (λ n => n = 9 + (9 / 9) + (9 / 9)),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 3 - 4),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 2 + 9) ]

theorem nine_expressions_cover_1_to_13 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 13 → ∃ expr ∈ nine_expressions, expr n :=
sorry

end NUMINAMATH_CALUDE_nine_expressions_cover_1_to_13_l2854_285484


namespace NUMINAMATH_CALUDE_largest_solution_logarithm_equation_l2854_285439

theorem largest_solution_logarithm_equation (x : ℝ) : 
  (x > 0) → 
  (∀ y, y > 0 → (Real.log 2 / Real.log (2*y) + Real.log 2 / Real.log (4*y^2) = -1) → x ≥ y) →
  (Real.log 2 / Real.log (2*x) + Real.log 2 / Real.log (4*x^2) = -1) →
  1 / x^12 = 4096 := by
sorry

end NUMINAMATH_CALUDE_largest_solution_logarithm_equation_l2854_285439


namespace NUMINAMATH_CALUDE_hockey_league_games_l2854_285454

/-- The number of teams in the hockey league -/
def num_teams : ℕ := 15

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- Calculates the total number of games in the season -/
def total_games : ℕ := (num_teams * (num_teams - 1) / 2) * games_per_pair

/-- Theorem: The total number of games in the season is 1050 -/
theorem hockey_league_games : total_games = 1050 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l2854_285454


namespace NUMINAMATH_CALUDE_baseball_game_opponent_score_l2854_285405

theorem baseball_game_opponent_score :
  let team_scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let total_games : ℕ := team_scores.length
  let lost_games : ℕ := 8
  let won_games : ℕ := total_games - lost_games
  let lost_score_diff : ℕ := 2
  let won_score_ratio : ℕ := 3
  ∃ (opponent_scores : List ℕ),
    opponent_scores.length = total_games ∧
    (∀ i ∈ Finset.range lost_games,
      opponent_scores[i]! = team_scores[i]! + lost_score_diff) ∧
    (∀ i ∈ Finset.range won_games,
      team_scores[lost_games + i]! = won_score_ratio * opponent_scores[lost_games + i]!) ∧
    opponent_scores.sum = 78 :=
by sorry

end NUMINAMATH_CALUDE_baseball_game_opponent_score_l2854_285405


namespace NUMINAMATH_CALUDE_ramola_rank_from_last_l2854_285462

theorem ramola_rank_from_last (total_students : ℕ) (rank_from_start : ℕ) :
  total_students = 26 →
  rank_from_start = 14 →
  total_students - rank_from_start + 1 = 14 :=
by sorry

end NUMINAMATH_CALUDE_ramola_rank_from_last_l2854_285462


namespace NUMINAMATH_CALUDE_light_off_after_odd_presses_l2854_285476

def LightSwitch : Type := Bool

def press (state : LightSwitch) : LightSwitch :=
  !state

def press_n_times (state : LightSwitch) (n : ℕ) : LightSwitch :=
  match n with
  | 0 => state
  | m + 1 => press (press_n_times state m)

theorem light_off_after_odd_presses (n : ℕ) (h : Odd n) :
  press_n_times true n = false :=
sorry

end NUMINAMATH_CALUDE_light_off_after_odd_presses_l2854_285476


namespace NUMINAMATH_CALUDE_inequality_properties_l2854_285491

theorem inequality_properties (a b c d : ℝ) :
  (∀ (a b c : ℝ), c ≠ 0 → a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b c d : ℝ), a > b → c > d → a + c > b + d) ∧
  (∃ (a b c d : ℝ), a > b ∧ c > d ∧ ¬(a * c > b * d)) ∧
  (∃ (a b : ℝ), a > b ∧ ¬(1 / a > 1 / b)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_properties_l2854_285491


namespace NUMINAMATH_CALUDE_inequality_transformation_l2854_285490

theorem inequality_transformation (a b c : ℝ) (h1 : c ≠ 0) :
  a * c^2 > b * c^2 → a > b := by sorry

end NUMINAMATH_CALUDE_inequality_transformation_l2854_285490
