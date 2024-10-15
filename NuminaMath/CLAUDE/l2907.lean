import Mathlib

namespace NUMINAMATH_CALUDE_regular_polyhedra_similarity_l2907_290732

/-- A regular polyhedron -/
structure RegularPolyhedron where
  -- Add necessary fields here
  -- For example:
  vertices : Set (ℝ × ℝ × ℝ)
  faces : Set (Set (ℝ × ℝ × ℝ))
  -- Add more fields as needed

/-- Defines what it means for two polyhedra to be of the same combinatorial type -/
def same_combinatorial_type (P Q : RegularPolyhedron) : Prop :=
  sorry

/-- Defines what it means for faces to be of the same kind -/
def same_kind_faces (P Q : RegularPolyhedron) : Prop :=
  sorry

/-- Defines what it means for polyhedral angles to be of the same kind -/
def same_kind_angles (P Q : RegularPolyhedron) : Prop :=
  sorry

/-- Defines similarity between two polyhedra -/
def similar (P Q : RegularPolyhedron) : Prop :=
  sorry

/-- The main theorem: two regular polyhedra of the same combinatorial type
    with faces and polyhedral angles of the same kind are similar -/
theorem regular_polyhedra_similarity (P Q : RegularPolyhedron)
  (h1 : same_combinatorial_type P Q)
  (h2 : same_kind_faces P Q)
  (h3 : same_kind_angles P Q) :
  similar P Q :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polyhedra_similarity_l2907_290732


namespace NUMINAMATH_CALUDE_alices_number_l2907_290737

theorem alices_number : ∃ n : ℕ, 
  (180 ∣ n) ∧ 
  (45 ∣ n) ∧ 
  1000 ≤ n ∧ 
  n < 3000 ∧ 
  (∀ m : ℕ, (180 ∣ m) ∧ (45 ∣ m) ∧ 1000 ≤ m ∧ m < 3000 → n ≤ m) ∧
  n = 1260 :=
by sorry

end NUMINAMATH_CALUDE_alices_number_l2907_290737


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2907_290757

/-- A right-angled triangle with specific properties -/
structure RightTriangle where
  -- The lengths of the two legs
  a : ℝ
  b : ℝ
  -- The midpoints of the legs
  m : ℝ × ℝ
  n : ℝ × ℝ
  -- Conditions
  right_angle : a^2 + b^2 = (a + b)^2 / 2
  m_midpoint : m = (a/2, 0)
  n_midpoint : n = (0, b/2)
  xn_length : a^2 + (b/2)^2 = 22^2
  ym_length : b^2 + (a/2)^2 = 31^2

/-- The theorem to be proved -/
theorem right_triangle_hypotenuse (t : RightTriangle) : 
  Real.sqrt (t.a^2 + t.b^2) = 34 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2907_290757


namespace NUMINAMATH_CALUDE_initial_bushes_count_l2907_290798

/-- The number of orchid bushes to be planted today -/
def bushes_to_plant : ℕ := 4

/-- The final number of orchid bushes after planting -/
def final_bushes : ℕ := 6

/-- The initial number of orchid bushes in the park -/
def initial_bushes : ℕ := final_bushes - bushes_to_plant

theorem initial_bushes_count : initial_bushes = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_bushes_count_l2907_290798


namespace NUMINAMATH_CALUDE_inequality_proof_l2907_290725

theorem inequality_proof (α β γ : ℝ) : 1 - Real.sin (α / 2) ≥ 2 * Real.sin (β / 2) * Real.sin (γ / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2907_290725


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2907_290741

theorem quadratic_equation_solution (x : ℝ) : 
  (x - 2) * (x + 3) = 0 ↔ x = 2 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2907_290741


namespace NUMINAMATH_CALUDE_fraction_zero_solution_l2907_290713

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 - 16) / (4 - x) = 0 ∧ x ≠ 4 → x = -4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_solution_l2907_290713


namespace NUMINAMATH_CALUDE_area_is_40_l2907_290709

/-- Two perpendicular lines intersecting at point B -/
structure PerpendicularLines where
  B : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  perpendicular : Bool
  intersect_at_B : Bool
  y_intercept_product : ℝ

/-- The area of triangle BRS given two perpendicular lines -/
def triangle_area (lines : PerpendicularLines) : ℝ :=
  sorry

/-- Theorem stating that the area of triangle BRS is 40 -/
theorem area_is_40 (lines : PerpendicularLines) 
  (h1 : lines.B = (8, 6))
  (h2 : lines.perpendicular = true)
  (h3 : lines.intersect_at_B = true)
  (h4 : lines.y_intercept_product = -24)
  : triangle_area lines = 40 := by
  sorry

end NUMINAMATH_CALUDE_area_is_40_l2907_290709


namespace NUMINAMATH_CALUDE_function_inequality_l2907_290785

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x > 0, (x * log x) * deriv f x < f x) : 
  2 * f (sqrt e) > f e := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2907_290785


namespace NUMINAMATH_CALUDE_log_eight_x_equals_three_point_two_five_l2907_290702

theorem log_eight_x_equals_three_point_two_five (x : ℝ) :
  Real.log x / Real.log 8 = 3.25 → x = 32 * (2 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_log_eight_x_equals_three_point_two_five_l2907_290702


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2907_290752

/-- The eccentricity of a hyperbola with equation y²/9 - x²/4 = 1 is √13/3 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 13 / 3 ∧
  ∀ x y : ℝ, y^2 / 9 - x^2 / 4 = 1 → 
  e = Real.sqrt ((3:ℝ)^2 + (2:ℝ)^2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2907_290752


namespace NUMINAMATH_CALUDE_shopping_discount_theorem_l2907_290701

def shoe_price : ℝ := 60
def dress_price : ℝ := 120
def accessory_price : ℝ := 25

def shoe_discount : ℝ := 0.3
def dress_discount : ℝ := 0.15
def accessory_discount : ℝ := 0.5
def additional_discount : ℝ := 0.1

def shoe_quantity : ℕ := 3
def dress_quantity : ℕ := 2
def accessory_quantity : ℕ := 3

def discount_threshold : ℝ := 200

theorem shopping_discount_theorem :
  let total_before_discount := shoe_price * shoe_quantity + dress_price * dress_quantity + accessory_price * accessory_quantity
  let shoe_discounted := shoe_price * shoe_quantity * (1 - shoe_discount)
  let dress_discounted := dress_price * dress_quantity * (1 - dress_discount)
  let accessory_discounted := accessory_price * accessory_quantity * (1 - accessory_discount)
  let total_after_category_discounts := shoe_discounted + dress_discounted + accessory_discounted
  let final_total := 
    if total_before_discount > discount_threshold
    then total_after_category_discounts * (1 - additional_discount)
    else total_after_category_discounts
  final_total = 330.75 := by
  sorry

end NUMINAMATH_CALUDE_shopping_discount_theorem_l2907_290701


namespace NUMINAMATH_CALUDE_ball_count_proof_l2907_290707

/-- Given a box with balls where some are red, prove that if there are 12 red balls
    and the probability of drawing a red ball is 0.6, then the total number of balls is 20. -/
theorem ball_count_proof (total_balls : ℕ) (red_balls : ℕ) (prob_red : ℚ) 
    (h1 : red_balls = 12)
    (h2 : prob_red = 6/10)
    (h3 : (red_balls : ℚ) / total_balls = prob_red) : 
  total_balls = 20 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_proof_l2907_290707


namespace NUMINAMATH_CALUDE_percentage_relation_l2907_290700

theorem percentage_relation (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x / 100 * y = 12) (h2 : y / 100 * x = 9) : x = 400 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l2907_290700


namespace NUMINAMATH_CALUDE_min_distance_complex_l2907_290797

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z - (1 + 2*I)) = 2) :
  ∃ (min_val : ℝ), (∀ (w : ℂ), Complex.abs (w - (1 + 2*I)) = 2 → Complex.abs (w + 1) ≥ min_val) ∧
                   (∃ (z₀ : ℂ), Complex.abs (z₀ - (1 + 2*I)) = 2 ∧ Complex.abs (z₀ + 1) = min_val) ∧
                   min_val = 2 * Real.sqrt 2 - 2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_l2907_290797


namespace NUMINAMATH_CALUDE_greatest_common_piece_length_l2907_290776

theorem greatest_common_piece_length :
  let rope_lengths : List Nat := [45, 60, 75, 90]
  Nat.gcd (Nat.gcd (Nat.gcd 45 60) 75) 90 = 15 := by sorry

end NUMINAMATH_CALUDE_greatest_common_piece_length_l2907_290776


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l2907_290718

theorem least_positive_integer_multiple_of_53 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ ((3*y)^2 + 2*43*(3*y) + 43^2))) ∧
  (53 ∣ ((3*x)^2 + 2*43*(3*x) + 43^2)) ∧
  x = 21 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l2907_290718


namespace NUMINAMATH_CALUDE_expression_one_evaluation_l2907_290788

theorem expression_one_evaluation : 8 / (-2) - (-4) * (-3) = -16 := by sorry

end NUMINAMATH_CALUDE_expression_one_evaluation_l2907_290788


namespace NUMINAMATH_CALUDE_equation_solution_l2907_290728

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (1 - x) / (x - 2) = 1 - 3 / (x - 2) ↔ x = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2907_290728


namespace NUMINAMATH_CALUDE_weekly_earnings_increase_l2907_290799

/-- Calculates the percentage increase between two amounts -/
def percentageIncrease (originalAmount newAmount : ℚ) : ℚ :=
  ((newAmount - originalAmount) / originalAmount) * 100

theorem weekly_earnings_increase (originalAmount newAmount : ℚ) 
  (h1 : originalAmount = 40)
  (h2 : newAmount = 80) :
  percentageIncrease originalAmount newAmount = 100 := by
  sorry

#eval percentageIncrease 40 80

end NUMINAMATH_CALUDE_weekly_earnings_increase_l2907_290799


namespace NUMINAMATH_CALUDE_candy_bar_sales_l2907_290791

/-- The number of additional candy bars sold each day -/
def additional_candy_bars : ℕ := sorry

/-- The cost of each candy bar in cents -/
def candy_bar_cost : ℕ := 10

/-- The number of days Sol sells candy bars in a week -/
def selling_days : ℕ := 6

/-- The number of candy bars sold on the first day -/
def first_day_sales : ℕ := 10

/-- The total earnings in cents for the week -/
def total_earnings : ℕ := 1200

theorem candy_bar_sales :
  (first_day_sales * selling_days + 
   additional_candy_bars * (selling_days * (selling_days - 1) / 2)) * 
  candy_bar_cost = total_earnings :=
sorry

end NUMINAMATH_CALUDE_candy_bar_sales_l2907_290791


namespace NUMINAMATH_CALUDE_equation_decomposition_l2907_290780

-- Define the equation
def equation (x y : ℝ) : Prop := y^6 - 9*x^6 = 3*y^3 - 1

-- Define a parabola
def is_parabola (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Theorem statement
theorem equation_decomposition :
  ∃ f g : ℝ → ℝ, 
    (∀ x y, equation x y ↔ (y = f x ∨ y = g x)) ∧
    is_parabola f ∧ is_parabola g :=
sorry

end NUMINAMATH_CALUDE_equation_decomposition_l2907_290780


namespace NUMINAMATH_CALUDE_jerry_current_average_l2907_290786

def jerry_average_score (current_average : ℝ) (raise_by : ℝ) (fourth_test_score : ℝ) : Prop :=
  let total_score_needed := 4 * (current_average + raise_by)
  3 * current_average + fourth_test_score = total_score_needed

theorem jerry_current_average : 
  ∃ (A : ℝ), jerry_average_score A 2 89 ∧ A = 81 :=
sorry

end NUMINAMATH_CALUDE_jerry_current_average_l2907_290786


namespace NUMINAMATH_CALUDE_function_inequality_l2907_290720

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) : f a > Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2907_290720


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2907_290765

def A : Set ℝ := {x | x^2 - 2*x - 8 > 0}
def B : Set ℝ := {-3, -1, 1, 3, 5}

theorem intersection_of_A_and_B : A ∩ B = {-3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2907_290765


namespace NUMINAMATH_CALUDE_complex_fraction_real_l2907_290708

theorem complex_fraction_real (a : ℝ) : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (Complex.ofReal a - Complex.I) / (2 + Complex.I) ∈ Set.range Complex.ofReal →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l2907_290708


namespace NUMINAMATH_CALUDE_parabola_symmetry_l2907_290763

/-- Prove that if (2, 3) lies on the parabola y = ax^2 + 2ax + c, then (-4, 3) also lies on it -/
theorem parabola_symmetry (a c : ℝ) : 
  (3 = a * 2^2 + 2 * a * 2 + c) → (3 = a * (-4)^2 + 2 * a * (-4) + c) := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l2907_290763


namespace NUMINAMATH_CALUDE_lucky_number_properties_l2907_290743

def is_lucky_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  let ab := n / 100
  let cd := n % 100
  ab ≠ cd ∧ cd % ab = 0 ∧ n % cd = 0

def count_lucky_numbers : ℕ := sorry

def largest_odd_lucky_number : ℕ := sorry

theorem lucky_number_properties :
  count_lucky_numbers = 65 ∧
  largest_odd_lucky_number = 1995 ∧
  is_lucky_number largest_odd_lucky_number ∧
  (∀ n, is_lucky_number n → n % 2 = 1 → n ≤ largest_odd_lucky_number) := by sorry

end NUMINAMATH_CALUDE_lucky_number_properties_l2907_290743


namespace NUMINAMATH_CALUDE_arrangement_count_l2907_290712

theorem arrangement_count (boys girls : ℕ) (total_selected : ℕ) (girls_selected : ℕ) : 
  boys = 5 → girls = 3 → total_selected = 5 → girls_selected = 2 →
  (Nat.choose girls girls_selected) * (Nat.choose boys (total_selected - girls_selected)) * (Nat.factorial total_selected) = 3600 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l2907_290712


namespace NUMINAMATH_CALUDE_dividend_rate_is_14_percent_l2907_290764

/-- Calculates the rate of dividend given investment details and annual income -/
def rate_of_dividend (total_investment : ℚ) (share_face_value : ℚ) (share_quoted_price : ℚ) (annual_income : ℚ) : ℚ :=
  let number_of_shares := total_investment / share_quoted_price
  let dividend_per_share := annual_income / number_of_shares
  (dividend_per_share / share_face_value) * 100

/-- Theorem stating that given the specific investment details and annual income, the rate of dividend is 14% -/
theorem dividend_rate_is_14_percent :
  rate_of_dividend 4940 10 9.5 728 = 14 := by
  sorry

end NUMINAMATH_CALUDE_dividend_rate_is_14_percent_l2907_290764


namespace NUMINAMATH_CALUDE_train_length_calculation_l2907_290742

theorem train_length_calculation (platform_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) :
  platform_length = 400 →
  platform_crossing_time = 42 →
  pole_crossing_time = 18 →
  ∃ train_length : ℝ,
    train_length = 300 ∧
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2907_290742


namespace NUMINAMATH_CALUDE_vector_equation_l2907_290769

theorem vector_equation (a b : ℝ × ℝ) : 
  a = (1, 2) → 2 • a + b = (3, 2) → b = (1, -2) := by sorry

end NUMINAMATH_CALUDE_vector_equation_l2907_290769


namespace NUMINAMATH_CALUDE_bin_drawing_probability_l2907_290783

def bin_probability (black white : ℕ) : ℚ :=
  let total := black + white
  let favorable := (black.choose 2 * white) + (black * white.choose 2)
  favorable / total.choose 3

theorem bin_drawing_probability :
  bin_probability 10 4 = 60 / 91 := by
  sorry

end NUMINAMATH_CALUDE_bin_drawing_probability_l2907_290783


namespace NUMINAMATH_CALUDE_diplomats_language_theorem_l2907_290795

theorem diplomats_language_theorem (total : ℕ) (japanese : ℕ) (not_russian : ℕ) (both_percent : ℚ) :
  total = 120 →
  japanese = 20 →
  not_russian = 32 →
  both_percent = 1/10 →
  (↑(total - (japanese + (total - not_russian) - (both_percent * ↑total).num)) / ↑total : ℚ) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_diplomats_language_theorem_l2907_290795


namespace NUMINAMATH_CALUDE_expected_balls_in_original_position_l2907_290730

/-- Represents the number of balls in the circle. -/
def n : ℕ := 6

/-- Represents the number of swaps performed. -/
def k : ℕ := 3

/-- The probability of a specific ball being swapped in one swap. -/
def p : ℚ := 1 / 3

/-- The probability of a ball remaining in its original position after k swaps. -/
def prob_original_position (k : ℕ) (p : ℚ) : ℚ :=
  (1 - p)^k + k * p * (1 - p)^(k-1)

/-- The expected number of balls in their original positions after k swaps. -/
def expected_original_positions (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  n * prob_original_position k p

theorem expected_balls_in_original_position :
  expected_original_positions n k p = 84 / 27 :=
by sorry

end NUMINAMATH_CALUDE_expected_balls_in_original_position_l2907_290730


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l2907_290775

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l2907_290775


namespace NUMINAMATH_CALUDE_consecutive_product_prime_power_and_perfect_power_l2907_290793

theorem consecutive_product_prime_power_and_perfect_power (m : ℕ) : m ≥ 1 → (
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ m * (m + 1) = p^k) ↔ m = 1
) ∧ 
¬(∃ (a k : ℕ), a ≥ 1 ∧ k ≥ 2 ∧ m * (m + 1) = a^k) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_prime_power_and_perfect_power_l2907_290793


namespace NUMINAMATH_CALUDE_jason_initial_cards_l2907_290792

/-- The number of Pokemon cards Jason gave away. -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left. -/
def cards_left : ℕ := 4

/-- The initial number of Pokemon cards Jason had. -/
def initial_cards : ℕ := cards_given_away + cards_left

/-- Theorem stating that Jason initially had 13 Pokemon cards. -/
theorem jason_initial_cards : initial_cards = 13 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l2907_290792


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_l2907_290756

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem statement
theorem not_all_squares_congruent :
  ¬ (∀ s1 s2 : Square, congruent s1 s2) :=
sorry

end NUMINAMATH_CALUDE_not_all_squares_congruent_l2907_290756


namespace NUMINAMATH_CALUDE_average_remaining_is_70_l2907_290796

/-- Represents the number of travelers checks of each denomination -/
structure TravelersChecks where
  fifty : ℕ
  hundred : ℕ

/-- The problem setup for travelers checks -/
def travelersChecksProblem (tc : TravelersChecks) : Prop :=
  tc.fifty + tc.hundred = 30 ∧
  50 * tc.fifty + 100 * tc.hundred = 1800

/-- The average amount of remaining checks after spending 15 $50 checks -/
def averageRemainingAmount (tc : TravelersChecks) : ℚ :=
  (50 * (tc.fifty - 15) + 100 * tc.hundred) / (tc.fifty + tc.hundred - 15)

/-- Theorem stating that the average amount of remaining checks is $70 -/
theorem average_remaining_is_70 (tc : TravelersChecks) :
  travelersChecksProblem tc → averageRemainingAmount tc = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_remaining_is_70_l2907_290796


namespace NUMINAMATH_CALUDE_angle_AOC_equals_negative_150_l2907_290751

-- Define the rotation angles
def counterclockwise_rotation : ℝ := 120
def clockwise_rotation : ℝ := 270

-- Define the resulting angle
def angle_AOC : ℝ := counterclockwise_rotation - clockwise_rotation

-- Theorem statement
theorem angle_AOC_equals_negative_150 : angle_AOC = -150 := by
  sorry

end NUMINAMATH_CALUDE_angle_AOC_equals_negative_150_l2907_290751


namespace NUMINAMATH_CALUDE_square_of_binomial_coefficient_l2907_290703

/-- If bx^2 + 18x + 9 is the square of a binomial, then b = 9 -/
theorem square_of_binomial_coefficient (b : ℝ) : 
  (∃ t u : ℝ, ∀ x : ℝ, bx^2 + 18*x + 9 = (t*x + u)^2) → b = 9 :=
by sorry

end NUMINAMATH_CALUDE_square_of_binomial_coefficient_l2907_290703


namespace NUMINAMATH_CALUDE_trig_equality_proof_l2907_290723

theorem trig_equality_proof (x : ℝ) : 
  (Real.sin x * Real.cos (2 * x) + Real.cos x * Real.cos (4 * x) = 
   Real.sin (π / 4 + 2 * x) * Real.sin (π / 4 - 3 * x)) ↔ 
  (∃ n : ℤ, x = π / 12 * (4 * n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_trig_equality_proof_l2907_290723


namespace NUMINAMATH_CALUDE_tangent_line_at_1_l2907_290749

-- Define the function f
def f (x : ℝ) : ℝ := x^2 * (x - 2) + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

-- Theorem statement
theorem tangent_line_at_1 : 
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m*x + b ↔ x - y + 1 = 0) ∧ 
  (m = f' 1) ∧ 
  (f 1 = m*1 + b) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_1_l2907_290749


namespace NUMINAMATH_CALUDE_mika_stickers_l2907_290717

/-- The number of stickers Mika has left after various transactions --/
def stickers_left (initial bought birthday given_away used : ℕ) : ℕ :=
  initial + bought + birthday - given_away - used

/-- Theorem stating that Mika is left with 2 stickers --/
theorem mika_stickers :
  stickers_left 20 26 20 6 58 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_l2907_290717


namespace NUMINAMATH_CALUDE_arthur_walking_distance_l2907_290753

/-- Calculates the total distance walked given the number of blocks and block length -/
def total_distance (blocks_south : ℕ) (blocks_west : ℕ) (block_length : ℚ) : ℚ :=
  (blocks_south + blocks_west : ℚ) * block_length

/-- Theorem: Arthur's total walking distance is 4.5 miles -/
theorem arthur_walking_distance :
  let blocks_south : ℕ := 8
  let blocks_west : ℕ := 10
  let block_length : ℚ := 1/4
  total_distance blocks_south blocks_west block_length = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walking_distance_l2907_290753


namespace NUMINAMATH_CALUDE_line_parallel_or_in_plane_l2907_290777

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  
/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here

/-- Parallel relation between two lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Main theorem -/
theorem line_parallel_or_in_plane (a b : Line3D) (α : Plane3D) 
  (h1 : parallel_lines a b) (h2 : parallel_line_plane a α) :
  parallel_line_plane b α ∨ line_in_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_or_in_plane_l2907_290777


namespace NUMINAMATH_CALUDE_problem_1_l2907_290738

theorem problem_1 : -20 - (-14) - |(-18)| - 13 = -37 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2907_290738


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2907_290745

-- Define the repeating decimal 4.666...
def repeating_decimal : ℚ :=
  4 + (2 / 3)

-- Theorem statement
theorem repeating_decimal_as_fraction :
  repeating_decimal = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2907_290745


namespace NUMINAMATH_CALUDE_gumball_sales_total_l2907_290710

theorem gumball_sales_total (price1 price2 price3 price4 price5 : ℕ) 
  (h1 : price1 = 12)
  (h2 : price2 = 15)
  (h3 : price3 = 8)
  (h4 : price4 = 10)
  (h5 : price5 = 20) :
  price1 + price2 + price3 + price4 + price5 = 65 := by
  sorry

end NUMINAMATH_CALUDE_gumball_sales_total_l2907_290710


namespace NUMINAMATH_CALUDE_max_even_integer_quadratic_inequality_l2907_290724

theorem max_even_integer_quadratic_inequality :
  (∃ a : ℤ, a % 2 = 0 ∧ a^2 - 12*a + 32 ≤ 0) →
  (∀ a : ℤ, a % 2 = 0 ∧ a^2 - 12*a + 32 ≤ 0 → a ≤ 8) ∧
  (∃ a : ℤ, a = 8 ∧ a % 2 = 0 ∧ a^2 - 12*a + 32 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_even_integer_quadratic_inequality_l2907_290724


namespace NUMINAMATH_CALUDE_elise_comic_book_expense_l2907_290735

theorem elise_comic_book_expense (initial_amount : ℤ) (saved_amount : ℤ) 
  (puzzle_cost : ℤ) (amount_left : ℤ) :
  initial_amount = 8 →
  saved_amount = 13 →
  puzzle_cost = 18 →
  amount_left = 1 →
  initial_amount + saved_amount - puzzle_cost - amount_left = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_elise_comic_book_expense_l2907_290735


namespace NUMINAMATH_CALUDE_test_has_ten_four_point_questions_l2907_290784

/-- Represents a test with two-point and four-point questions -/
structure Test where
  total_points : ℕ
  total_questions : ℕ
  two_point_questions : ℕ
  four_point_questions : ℕ

/-- Checks if a test configuration is valid -/
def is_valid_test (t : Test) : Prop :=
  t.total_questions = t.two_point_questions + t.four_point_questions ∧
  t.total_points = 2 * t.two_point_questions + 4 * t.four_point_questions

/-- Theorem: A test with 100 points and 40 questions has 10 four-point questions -/
theorem test_has_ten_four_point_questions (t : Test) 
  (h1 : t.total_points = 100) 
  (h2 : t.total_questions = 40) 
  (h3 : is_valid_test t) : 
  t.four_point_questions = 10 := by
  sorry

end NUMINAMATH_CALUDE_test_has_ten_four_point_questions_l2907_290784


namespace NUMINAMATH_CALUDE_basketball_volleyball_cost_total_cost_proof_l2907_290704

/-- The cost of buying basketballs and volleyballs -/
theorem basketball_volleyball_cost (m n : ℝ) : ℝ :=
  3 * m + 7 * n

/-- Proof that the total cost of 3 basketballs and 7 volleyballs is 3m + 7n yuan -/
theorem total_cost_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  basketball_volleyball_cost m n = 3 * m + 7 * n :=
by sorry

end NUMINAMATH_CALUDE_basketball_volleyball_cost_total_cost_proof_l2907_290704


namespace NUMINAMATH_CALUDE_cooking_oil_problem_l2907_290766

theorem cooking_oil_problem (X : ℝ) : 
  (X - ((2/5) * X + 300)) - ((1/2) * (X - ((2/5) * X + 300)) - 200) = 800 →
  X = 2500 :=
by
  sorry

#check cooking_oil_problem

end NUMINAMATH_CALUDE_cooking_oil_problem_l2907_290766


namespace NUMINAMATH_CALUDE_video_votes_l2907_290794

theorem video_votes (up_votes : ℕ) (ratio_up : ℕ) (ratio_down : ℕ) (down_votes : ℕ) : 
  up_votes = 18 →
  ratio_up = 9 →
  ratio_down = 2 →
  up_votes * ratio_down = down_votes * ratio_up →
  down_votes = 4 := by
sorry

end NUMINAMATH_CALUDE_video_votes_l2907_290794


namespace NUMINAMATH_CALUDE_min_hours_sixth_week_l2907_290727

/-- The required average number of hours per week -/
def required_average : ℚ := 12

/-- The number of weeks -/
def total_weeks : ℕ := 6

/-- The hours worked in the first 5 weeks -/
def first_five_weeks : List ℕ := [9, 10, 14, 11, 8]

/-- The sum of hours worked in the first 5 weeks -/
def sum_first_five : ℕ := first_five_weeks.sum

theorem min_hours_sixth_week : 
  ∀ x : ℕ, 
    (sum_first_five + x : ℚ) / total_weeks ≥ required_average → 
    x ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_min_hours_sixth_week_l2907_290727


namespace NUMINAMATH_CALUDE_remainder_theorem_l2907_290722

-- Define the polynomial p(x)
variable (p : ℝ → ℝ)

-- Define the conditions
axiom p_div_x_minus_2 : ∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * q x + 2
axiom p_div_x_minus_3 : ∃ q : ℝ → ℝ, ∀ x, p x = (x - 3) * q x + 6

-- State the theorem
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * (x - 3) * q x + (4 * x - 6) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2907_290722


namespace NUMINAMATH_CALUDE_road_trash_cans_l2907_290770

/-- The number of trash cans on both sides of a road -/
def trashCanCount (roadLength : ℕ) (interval : ℕ) : ℕ :=
  2 * ((roadLength / interval) - 1)

/-- Theorem: The total number of trash cans on a 400-meter road with 20-meter intervals is 38 -/
theorem road_trash_cans :
  trashCanCount 400 20 = 38 := by
  sorry

end NUMINAMATH_CALUDE_road_trash_cans_l2907_290770


namespace NUMINAMATH_CALUDE_revision_cost_per_page_revision_cost_is_four_l2907_290719

/-- The cost per page for revision in a manuscript typing service --/
theorem revision_cost_per_page : ℝ → Prop :=
  fun x =>
    let total_pages : ℕ := 100
    let pages_revised_once : ℕ := 35
    let pages_revised_twice : ℕ := 15
    let pages_not_revised : ℕ := total_pages - pages_revised_once - pages_revised_twice
    let initial_typing_cost_per_page : ℝ := 6
    let total_cost : ℝ := 860
    (initial_typing_cost_per_page * total_pages + x * pages_revised_once + 2 * x * pages_revised_twice = total_cost) →
    x = 4

theorem revision_cost_is_four : revision_cost_per_page 4 := by
  sorry

end NUMINAMATH_CALUDE_revision_cost_per_page_revision_cost_is_four_l2907_290719


namespace NUMINAMATH_CALUDE_smallest_square_area_l2907_290782

/-- Given three squares arranged as described in the problem, 
    this theorem relates the area of the smallest square to that of the middle square. -/
theorem smallest_square_area 
  (largest_square_area : ℝ) 
  (middle_square_area : ℝ) 
  (h1 : largest_square_area = 1) 
  (h2 : 0 < middle_square_area) 
  (h3 : middle_square_area < 1) :
  ∃ (smallest_square_area : ℝ), 
    smallest_square_area = ((1 - middle_square_area) / 2)^2 ∧ 
    0 < smallest_square_area ∧
    smallest_square_area < middle_square_area := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l2907_290782


namespace NUMINAMATH_CALUDE_intersection_and_center_l2907_290758

-- Define the square ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (4, 4)
def D : ℝ × ℝ := (4, 0)

-- Define the lines
def line_from_A (x : ℝ) : ℝ := x
def line_from_B (x : ℝ) : ℝ := 4 - x

-- Define the intersection point
def intersection_point : ℝ × ℝ := (2, 2)

theorem intersection_and_center :
  (∀ x : ℝ, line_from_A x = line_from_B x → x = intersection_point.1) ∧
  (line_from_A intersection_point.1 = intersection_point.2) ∧
  (intersection_point.1 = (C.1 - A.1) / 2) ∧
  (intersection_point.2 = (C.2 - A.2) / 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_center_l2907_290758


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2907_290711

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  (x - y)^2 - (x + y)*(x - y) = -2*x*y + 2*y^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (12*a^2*b - 6*a*b^2) / (-3*a*b) = -4*a + 2*b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2907_290711


namespace NUMINAMATH_CALUDE_expression_value_l2907_290762

theorem expression_value (a b c d x : ℝ) : 
  (a = b) →  -- a and -b are opposite numbers
  (c * d = -1) →  -- c and -d are reciprocals
  (abs x = 3) →  -- absolute value of x is 3
  (x^3 + c * d * x^2 - (a - b) / 2 = 18 ∨ x^3 + c * d * x^2 - (a - b) / 2 = -36) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2907_290762


namespace NUMINAMATH_CALUDE_max_x_implies_a_value_l2907_290761

/-- Given that the maximum value of x satisfying (x² - 4x + a) + |x - 3| ≤ 5 is 3, prove that a = 8 -/
theorem max_x_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, (x^2 - 4*x + a) + |x - 3| ≤ 5 → x ≤ 3) ∧ 
  (∃ x : ℝ, x = 3 ∧ (x^2 - 4*x + a) + |x - 3| = 5) → 
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_max_x_implies_a_value_l2907_290761


namespace NUMINAMATH_CALUDE_fuel_savings_l2907_290781

theorem fuel_savings (old_efficiency : ℝ) (old_cost : ℝ) 
  (h_old_positive : old_efficiency > 0) (h_cost_positive : old_cost > 0) : 
  let new_efficiency := old_efficiency * (1 + 0.6)
  let new_cost := old_cost * 1.25
  let old_trip_cost := old_cost
  let new_trip_cost := (old_efficiency / new_efficiency) * new_cost
  let savings_percent := (old_trip_cost - new_trip_cost) / old_trip_cost * 100
  savings_percent = 21.875 := by
sorry


end NUMINAMATH_CALUDE_fuel_savings_l2907_290781


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2907_290790

/-- The y-intercept of the line 4x + 7y = 28 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) : 
  4 * x + 7 * y = 28 → (x = 0 ∧ y = 4) → (0, 4).fst = x ∧ (0, 4).snd = y := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2907_290790


namespace NUMINAMATH_CALUDE_money_distribution_l2907_290759

/-- Given the ratios of money between Ram and Gopal (7:17) and between Gopal and Krishan (7:17),
    and that Ram has Rs. 686, prove that Krishan has Rs. 4046. -/
theorem money_distribution (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 686 →
  krishan = 4046 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2907_290759


namespace NUMINAMATH_CALUDE_part_one_part_two_l2907_290754

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 3| + |x - a|

-- Part 1
theorem part_one : 
  {x : ℝ | f 4 x = 7} = Set.Icc (-3) 4 := by sorry

-- Part 2
theorem part_two (h : a > 0) :
  {x : ℝ | f a x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} → a = 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2907_290754


namespace NUMINAMATH_CALUDE_johns_final_push_time_l2907_290789

/-- The time of John's final push in a race, given the initial and final distances between
    John and Steve, and their respective speeds. -/
theorem johns_final_push_time 
  (initial_distance : ℝ) 
  (john_speed : ℝ) 
  (steve_speed : ℝ) 
  (final_distance : ℝ) : 
  initial_distance = 16 →
  john_speed = 4.2 →
  steve_speed = 3.7 →
  final_distance = 2 →
  ∃ t : ℝ, t = 15 / 7 ∧ john_speed * t = initial_distance + final_distance :=
by
  sorry

#check johns_final_push_time

end NUMINAMATH_CALUDE_johns_final_push_time_l2907_290789


namespace NUMINAMATH_CALUDE_payment_for_C_is_250_l2907_290748

/-- Calculates the payment for worker C given the work rates and total payment --/
def calculate_payment_C (a_rate : ℚ) (b_rate : ℚ) (total_rate : ℚ) (total_payment : ℚ) : ℚ :=
  let c_rate := total_rate - (a_rate + b_rate)
  c_rate * total_payment

/-- Theorem stating that C should be paid 250 given the problem conditions --/
theorem payment_for_C_is_250 
  (a_rate : ℚ) 
  (b_rate : ℚ) 
  (total_rate : ℚ) 
  (total_payment : ℚ) 
  (h1 : a_rate = 1/6) 
  (h2 : b_rate = 1/8) 
  (h3 : total_rate = 1/3) 
  (h4 : total_payment = 6000) : 
  calculate_payment_C a_rate b_rate total_rate total_payment = 250 := by
  sorry

#eval calculate_payment_C (1/6) (1/8) (1/3) 6000

end NUMINAMATH_CALUDE_payment_for_C_is_250_l2907_290748


namespace NUMINAMATH_CALUDE_sheep_buying_problem_l2907_290771

/-- Represents the sheep buying problem from "The Nine Chapters on the Mathematical Art" --/
theorem sheep_buying_problem (x y : ℤ) : 
  (∀ (contribution shortage : ℤ), contribution = 5 ∧ shortage = 45 → contribution * x + shortage = y) ∧
  (∀ (contribution surplus : ℤ), contribution = 7 ∧ surplus = 3 → contribution * x - surplus = y) ↔
  (5 * x + 45 = y ∧ 7 * x - 3 = y) :=
sorry


end NUMINAMATH_CALUDE_sheep_buying_problem_l2907_290771


namespace NUMINAMATH_CALUDE_sons_age_l2907_290729

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 28 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 26 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2907_290729


namespace NUMINAMATH_CALUDE_exam_max_marks_l2907_290734

theorem exam_max_marks :
  let pass_percentage : ℚ := 60 / 100
  let failing_score : ℕ := 210
  let failing_margin : ℕ := 90
  let max_marks : ℕ := 500
  (pass_percentage * max_marks : ℚ) = failing_score + failing_margin ∧
  max_marks = 500 := by
  sorry

end NUMINAMATH_CALUDE_exam_max_marks_l2907_290734


namespace NUMINAMATH_CALUDE_corn_kernel_problem_l2907_290721

theorem corn_kernel_problem (ears_per_stalk : ℕ) (num_stalks : ℕ) (total_kernels : ℕ) :
  ears_per_stalk = 4 →
  num_stalks = 108 →
  total_kernels = 237600 →
  ∃ X : ℕ,
    X * (ears_per_stalk * num_stalks / 2) +
    (X + 100) * (ears_per_stalk * num_stalks / 2) = total_kernels ∧
    X = 500 := by
  sorry

#check corn_kernel_problem

end NUMINAMATH_CALUDE_corn_kernel_problem_l2907_290721


namespace NUMINAMATH_CALUDE_adult_meal_cost_l2907_290726

/-- Calculates the cost of an adult meal given the total number of people,
    number of kids, and total cost for a group at a restaurant where kids eat free. -/
theorem adult_meal_cost (total_people : ℕ) (num_kids : ℕ) (total_cost : ℚ) :
  total_people = 12 →
  num_kids = 7 →
  total_cost = 15 →
  (total_cost / (total_people - num_kids) : ℚ) = 3 := by
  sorry

#check adult_meal_cost

end NUMINAMATH_CALUDE_adult_meal_cost_l2907_290726


namespace NUMINAMATH_CALUDE_g_one_equals_three_l2907_290767

-- Define f as an odd function
def f : ℝ → ℝ := sorry

-- Define g as an even function
def g : ℝ → ℝ := sorry

-- Axiom for odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Axiom for even function
axiom g_even : ∀ x : ℝ, g (-x) = g x

-- Given conditions
axiom condition1 : f (-1) + g 1 = 2
axiom condition2 : f 1 + g (-1) = 4

-- Theorem to prove
theorem g_one_equals_three : g 1 = 3 := by sorry

end NUMINAMATH_CALUDE_g_one_equals_three_l2907_290767


namespace NUMINAMATH_CALUDE_problem_statement_l2907_290760

theorem problem_statement (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (h : x = 1 / z^2) :
  (x - 1/x) * (z^2 + 1/z^2) = x^2 - z^4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2907_290760


namespace NUMINAMATH_CALUDE_optimal_move_is_MN_l2907_290736

/-- Represents a move in the game -/
inductive Move
| FG
| MN

/-- Represents the outcome of the game -/
structure Outcome :=
(player_score : ℕ)
(opponent_score : ℕ)

/-- The game state after 12 moves (6 by each player) -/
def initial_state : ℕ := 12

/-- Simulates the game outcome based on the chosen move -/
def simulate_game (move : Move) : Outcome :=
  match move with
  | Move.FG => ⟨1, 8⟩
  | Move.MN => ⟨8, 1⟩

/-- Determines if one outcome is better than another for the player -/
def is_better_outcome (o1 o2 : Outcome) : Prop :=
  o1.player_score > o2.player_score

theorem optimal_move_is_MN :
  let fg_outcome := simulate_game Move.FG
  let mn_outcome := simulate_game Move.MN
  is_better_outcome mn_outcome fg_outcome :=
by sorry


end NUMINAMATH_CALUDE_optimal_move_is_MN_l2907_290736


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2907_290733

/-- Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
    prove that S₉ = 81 when a₂ = 3 and S₄ = 16. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →
  (∀ n, a (n + 1) - a n = a 2 - a 1) →
  a 2 = 3 →
  S 4 = 16 →
  S 9 = 81 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2907_290733


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2907_290746

/-- Tetrahedron PQRS with given properties -/
structure Tetrahedron where
  -- Edge length QR
  qr : ℝ
  -- Area of face PQR
  area_pqr : ℝ
  -- Area of face QRS
  area_qrs : ℝ
  -- Angle between faces PQR and QRS (in radians)
  angle_pqr_qrs : ℝ

/-- The volume of the tetrahedron PQRS -/
def tetrahedron_volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the specific tetrahedron -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    qr := 15,
    area_pqr := 150,
    area_qrs := 90,
    angle_pqr_qrs := π / 4  -- 45° in radians
  }
  tetrahedron_volume t = 300 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2907_290746


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l2907_290750

theorem cubic_polynomial_roots : ∃ (r₁ r₂ : ℝ), 
  (∀ x : ℝ, x^3 - 7*x^2 + 8*x + 16 = 0 ↔ x = r₁ ∨ x = r₂) ∧
  r₁ = -1 ∧ r₂ = 4 ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - r₂| < δ → |x^3 - 7*x^2 + 8*x + 16| < ε * |x - r₂|^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l2907_290750


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2907_290740

theorem complex_equation_solution (a b : ℝ) : 
  (b : ℂ) + 5*I = 9 - a + a*I → b = 6 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2907_290740


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l2907_290779

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_of_power : 
  tens_digit ((4 + 3) ^ 12) + ones_digit ((4 + 3) ^ 12) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l2907_290779


namespace NUMINAMATH_CALUDE_expected_different_faces_formula_l2907_290744

/-- The number of sides on a fair die -/
def numSides : ℕ := 6

/-- The number of times the die is rolled -/
def numRolls : ℕ := 6

/-- The probability of a specific face not appearing in a single roll -/
def probNotAppear : ℚ := (numSides - 1) / numSides

/-- The expected number of different faces that appear when rolling a fair die -/
def expectedDifferentFaces : ℚ := numSides * (1 - probNotAppear ^ numRolls)

/-- Theorem stating the expected number of different faces when rolling a fair die -/
theorem expected_different_faces_formula :
  expectedDifferentFaces = (numSides^numRolls - (numSides - 1)^numRolls) / numSides^(numRolls - 1) :=
sorry

end NUMINAMATH_CALUDE_expected_different_faces_formula_l2907_290744


namespace NUMINAMATH_CALUDE_omars_kite_height_l2907_290773

/-- Omar's kite raising problem -/
theorem omars_kite_height 
  (omar_time : ℝ) 
  (jasper_rate_multiplier : ℝ) 
  (jasper_height : ℝ) 
  (jasper_time : ℝ) :
  omar_time = 12 →
  jasper_rate_multiplier = 3 →
  jasper_height = 600 →
  jasper_time = 10 →
  (omar_time * (jasper_height / jasper_time) / jasper_rate_multiplier) = 240 := by
  sorry

#check omars_kite_height

end NUMINAMATH_CALUDE_omars_kite_height_l2907_290773


namespace NUMINAMATH_CALUDE_cos_sin_identity_l2907_290787

theorem cos_sin_identity : 
  Real.cos (14 * π / 180) * Real.cos (59 * π / 180) + 
  Real.sin (14 * π / 180) * Real.sin (121 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l2907_290787


namespace NUMINAMATH_CALUDE_article_a_profit_percentage_l2907_290706

/-- Profit percentage calculation for Article A -/
theorem article_a_profit_percentage 
  (x : ℝ) -- selling price of Article A
  (y : ℝ) -- selling price of Article B
  (h1 : 0.5 * x = 0.8 * (x / 1.6)) -- condition for 20% loss at half price
  (h2 : 1.05 * y = 0.9 * x) -- condition for price equality after changes
  : (0.972 * x - (x / 1.6)) / (x / 1.6) * 100 = 55.52 := by sorry

end NUMINAMATH_CALUDE_article_a_profit_percentage_l2907_290706


namespace NUMINAMATH_CALUDE_water_volume_calculation_l2907_290705

/-- Given a volume of water that can be transferred into small hemisphere containers,
    this theorem proves the total volume of water. -/
theorem water_volume_calculation
  (hemisphere_volume : ℝ)
  (num_hemispheres : ℕ)
  (hemisphere_volume_is_4 : hemisphere_volume = 4)
  (num_hemispheres_is_2945 : num_hemispheres = 2945) :
  hemisphere_volume * num_hemispheres = 11780 :=
by sorry

end NUMINAMATH_CALUDE_water_volume_calculation_l2907_290705


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2907_290772

theorem inequality_solution_set (t m : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + t < 0 ↔ 1 < x ∧ x < m) → 
  t = 2 ∧ m = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2907_290772


namespace NUMINAMATH_CALUDE_fraction_equality_l2907_290731

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 2 / 3) :
  t / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2907_290731


namespace NUMINAMATH_CALUDE_simplify_expression_l2907_290774

theorem simplify_expression (x : ℝ) : (3*x)^4 + (4*x)*(x^5) = 81*x^4 + 4*x^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2907_290774


namespace NUMINAMATH_CALUDE_max_ab_value_l2907_290768

theorem max_ab_value (a b : ℝ) : 
  (∀ x : ℤ, (20 * x + a > 0 ∧ 15 * x - b ≤ 0) ↔ (x = 2 ∨ x = 3 ∨ x = 4)) →
  (∃ (a' b' : ℝ), a' * b' = -1200 ∧ ∀ (a'' b'' : ℝ), 
    (∀ x : ℤ, (20 * x + a'' > 0 ∧ 15 * x - b'' ≤ 0) ↔ (x = 2 ∨ x = 3 ∨ x = 4)) →
    a'' * b'' ≤ a' * b') :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l2907_290768


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l2907_290747

/-- A line is tangent to a parabola if and only if their intersection has exactly one point. -/
def is_tangent_line_to_parabola (a b c : ℝ) : Prop :=
  ∃! x : ℝ, (3 * x + 1)^2 = 12 * x

/-- The line y = 3x + 1 is tangent to the parabola y^2 = 12x. -/
theorem line_tangent_to_parabola : is_tangent_line_to_parabola 3 1 12 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l2907_290747


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l2907_290714

/-- Calculates the total wet surface area of a rectangular cistern -/
def totalWetSurfaceArea (length width depth : ℝ) : ℝ :=
  let bottomArea := length * width
  let longSideArea := 2 * depth * length
  let shortSideArea := 2 * depth * width
  bottomArea + longSideArea + shortSideArea

/-- Theorem stating that the total wet surface area of a specific cistern is 68.5 m² -/
theorem cistern_wet_surface_area :
  totalWetSurfaceArea 9 4 1.25 = 68.5 := by
  sorry

#eval totalWetSurfaceArea 9 4 1.25

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l2907_290714


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2907_290755

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Three terms form a geometric sequence -/
def FormGeometricSequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  FormGeometricSequence (a 3) (a 6) (a 9) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2907_290755


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l2907_290778

/-- Given a group of children with various emotional states and genders, 
    prove the number of boys who are neither happy nor sad. -/
theorem boys_neither_happy_nor_sad 
  (total_children : ℕ) 
  (happy_children : ℕ) 
  (sad_children : ℕ) 
  (neither_children : ℕ) 
  (total_boys : ℕ) 
  (total_girls : ℕ) 
  (happy_boys : ℕ) 
  (sad_girls : ℕ) 
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neither_children = 20)
  (h5 : total_boys = 17)
  (h6 : total_girls = 43)
  (h7 : happy_boys = 6)
  (h8 : sad_girls = 4)
  (h9 : total_children = happy_children + sad_children + neither_children)
  (h10 : total_children = total_boys + total_girls) :
  total_boys - (happy_boys + (sad_children - sad_girls)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l2907_290778


namespace NUMINAMATH_CALUDE_mashed_potatoes_count_l2907_290715

theorem mashed_potatoes_count : ∀ (bacon_count mashed_count : ℕ),
  bacon_count = 489 →
  bacon_count = mashed_count + 10 →
  mashed_count = 479 := by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_count_l2907_290715


namespace NUMINAMATH_CALUDE_intersecting_line_equations_l2907_290716

/-- A line passing through a point and intersecting a circle --/
structure IntersectingLine where
  -- The point through which the line passes
  point : ℝ × ℝ
  -- The center of the circle
  center : ℝ × ℝ
  -- The radius of the circle
  radius : ℝ
  -- The length of the chord formed by the intersection
  chord_length : ℝ

/-- The equations of the line given the conditions --/
def line_equations (l : IntersectingLine) : Set (ℝ → ℝ → Prop) :=
  { (λ x y => x = -4),
    (λ x y => 4*x + 3*y + 25 = 0) }

/-- Theorem stating that the given conditions result in the specified line equations --/
theorem intersecting_line_equations 
  (l : IntersectingLine)
  (h1 : l.point = (-4, -3))
  (h2 : l.center = (-1, -2))
  (h3 : l.radius = 5)
  (h4 : l.chord_length = 8) :
  ∃ (eq : ℝ → ℝ → Prop), eq ∈ line_equations l ∧ 
    ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | eq p.1 p.2} → 
      ((x + 1)^2 + (y + 2)^2 = 25 ∨ (x, y) = l.point) :=
sorry

end NUMINAMATH_CALUDE_intersecting_line_equations_l2907_290716


namespace NUMINAMATH_CALUDE_intersection_point_pq_rs_l2907_290739

/-- The intersection point of two lines in 3D space --/
def intersection_point (p q r s : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Given points in 3D space --/
def P : ℝ × ℝ × ℝ := (4, -3, 6)
def Q : ℝ × ℝ × ℝ := (20, -23, 14)
def R : ℝ × ℝ × ℝ := (-2, 7, -10)
def S : ℝ × ℝ × ℝ := (6, -11, 16)

/-- Theorem stating that the intersection point of lines PQ and RS is (180/19, -283/19, 202/19) --/
theorem intersection_point_pq_rs :
  intersection_point P Q R S = (180/19, -283/19, 202/19) := by sorry

end NUMINAMATH_CALUDE_intersection_point_pq_rs_l2907_290739
