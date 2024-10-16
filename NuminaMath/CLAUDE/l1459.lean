import Mathlib

namespace NUMINAMATH_CALUDE_investment_time_is_two_years_l1459_145929

/-- Calculates the time period of investment given the principal, interest rates, and interest difference. -/
def calculate_investment_time (principal : ℚ) (rate_high : ℚ) (rate_low : ℚ) (interest_diff : ℚ) : ℚ :=
  interest_diff / (principal * (rate_high - rate_low))

theorem investment_time_is_two_years 
  (principal : ℚ) 
  (rate_high : ℚ) 
  (rate_low : ℚ) 
  (interest_diff : ℚ) :
  principal = 2500 ∧ 
  rate_high = 18 / 100 ∧ 
  rate_low = 12 / 100 ∧ 
  interest_diff = 300 → 
  calculate_investment_time principal rate_high rate_low interest_diff = 2 :=
by
  sorry

#eval calculate_investment_time 2500 (18/100) (12/100) 300

end NUMINAMATH_CALUDE_investment_time_is_two_years_l1459_145929


namespace NUMINAMATH_CALUDE_power_inequality_l1459_145914

theorem power_inequality (a b : ℕ) (ha : a > 1) (hb : b > 2) :
  a^b + 1 ≥ b * (a + 1) ∧ (a^b + 1 = b * (a + 1) ↔ a = 2 ∧ b = 3) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l1459_145914


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l1459_145953

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -1) : 
  x^2 + y^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l1459_145953


namespace NUMINAMATH_CALUDE_jim_future_age_l1459_145964

/-- Tom's current age -/
def tom_current_age : ℕ := 37

/-- Tom's age 7 years ago -/
def tom_age_7_years_ago : ℕ := tom_current_age - 7

/-- Jim's age 7 years ago -/
def jim_age_7_years_ago : ℕ := tom_age_7_years_ago / 2 + 5

/-- Jim's current age -/
def jim_current_age : ℕ := jim_age_7_years_ago + 7

/-- Jim's age in 2 years -/
def jim_age_in_2_years : ℕ := jim_current_age + 2

theorem jim_future_age :
  tom_current_age = 37 →
  tom_age_7_years_ago = 30 →
  jim_age_7_years_ago = 20 →
  jim_current_age = 27 →
  jim_age_in_2_years = 29 := by
  sorry

end NUMINAMATH_CALUDE_jim_future_age_l1459_145964


namespace NUMINAMATH_CALUDE_digit_difference_1250_l1459_145919

/-- The number of digits in the base-b representation of a positive integer n -/
def num_digits (n : ℕ) (b : ℕ) : ℕ :=
  Nat.log b n + 1

/-- The theorem stating the difference in number of digits between base-4 and base-9 representations of 1250 -/
theorem digit_difference_1250 :
  num_digits 1250 4 - num_digits 1250 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_1250_l1459_145919


namespace NUMINAMATH_CALUDE_bisection_method_root_location_l1459_145904

def f (x : ℝ) := x^3 - 6*x^2 + 4

theorem bisection_method_root_location :
  (∃ r ∈ Set.Ioo 0 1, f r = 0) →
  (f 0 > 0) →
  (f 1 < 0) →
  (f (1/2) > 0) →
  ∃ r ∈ Set.Ioo (1/2) 1, f r = 0 := by sorry

end NUMINAMATH_CALUDE_bisection_method_root_location_l1459_145904


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1459_145961

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3, 4, 5}
  let B : Set ℕ := {2, 4, 5, 8, 10}
  A ∩ B = {2, 4, 5} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1459_145961


namespace NUMINAMATH_CALUDE_g_of_2_eq_0_l1459_145944

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 4

-- State the theorem
theorem g_of_2_eq_0 : g 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_eq_0_l1459_145944


namespace NUMINAMATH_CALUDE_fraction_equality_l1459_145932

theorem fraction_equality (x y : ℝ) (h : x ≠ -y) : (-x + y) / (-x - y) = (x - y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1459_145932


namespace NUMINAMATH_CALUDE_prob_red_or_green_is_two_thirds_l1459_145971

-- Define the number of balls of each color
def red_balls : ℕ := 2
def yellow_balls : ℕ := 3
def green_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + green_balls

-- Define the number of favorable outcomes (red or green balls)
def favorable_outcomes : ℕ := red_balls + green_balls

-- Define the probability of drawing a red or green ball
def prob_red_or_green : ℚ := favorable_outcomes / total_balls

-- Theorem statement
theorem prob_red_or_green_is_two_thirds : 
  prob_red_or_green = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_red_or_green_is_two_thirds_l1459_145971


namespace NUMINAMATH_CALUDE_businessmen_beverage_theorem_l1459_145916

theorem businessmen_beverage_theorem (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ) 
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 13)
  (h4 : both = 7) :
  total - (coffee + tea - both) = 9 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_beverage_theorem_l1459_145916


namespace NUMINAMATH_CALUDE_original_rectangle_perimeter_l1459_145911

/-- Given a rectangle with sides a and b, prove that if it's cut diagonally
    and then one piece is cut parallel to its shorter sides at the midpoints,
    resulting in a rectangle with perimeter 129 cm, then the perimeter of the
    original rectangle was 258 cm. -/
theorem original_rectangle_perimeter
  (a b : ℝ) 
  (h_positive : a > 0 ∧ b > 0)
  (h_final_perimeter : 2 * (a / 2 + b / 2) = 129) :
  2 * (a + b) = 258 :=
sorry

end NUMINAMATH_CALUDE_original_rectangle_perimeter_l1459_145911


namespace NUMINAMATH_CALUDE_line_intercept_sum_l1459_145992

/-- A line in the x-y plane -/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℚ := -l.y_intercept / l.slope

/-- The y-intercept of a line -/
def y_intercept (l : Line) : ℚ := l.y_intercept

/-- The sum of x-intercept and y-intercept of a line -/
def intercept_sum (l : Line) : ℚ := x_intercept l + y_intercept l

theorem line_intercept_sum :
  ∃ (l : Line), l.slope = -3 ∧ l.y_intercept = -13 ∧ intercept_sum l = -52/3 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l1459_145992


namespace NUMINAMATH_CALUDE_focus_of_our_parabola_l1459_145921

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- A parabola defined by its equation -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- The focus of a parabola given its equation -/
def focus_of_parabola (p : Parabola) : Focus := sorry

/-- The parabola x^2 + y = 0 -/
def our_parabola : Parabola :=
  { equation := fun x y => x^2 + y = 0 }

theorem focus_of_our_parabola :
  focus_of_parabola our_parabola = ⟨0, -1/4⟩ := by sorry

end NUMINAMATH_CALUDE_focus_of_our_parabola_l1459_145921


namespace NUMINAMATH_CALUDE_total_cats_l1459_145965

/-- Represents the Clevercat Academy with cats that can perform various tricks. -/
structure ClevercatAcademy where
  jump : ℕ
  fetch : ℕ
  spin : ℕ
  jump_fetch : ℕ
  fetch_spin : ℕ
  jump_spin : ℕ
  all_three : ℕ
  none : ℕ

/-- The theorem states that given the specific numbers of cats that can perform
    various combinations of tricks, the total number of cats in the academy is 99. -/
theorem total_cats (academy : ClevercatAcademy)
  (h_jump : academy.jump = 60)
  (h_fetch : academy.fetch = 35)
  (h_spin : academy.spin = 40)
  (h_jump_fetch : academy.jump_fetch = 20)
  (h_fetch_spin : academy.fetch_spin = 15)
  (h_jump_spin : academy.jump_spin = 22)
  (h_all_three : academy.all_three = 11)
  (h_none : academy.none = 10) :
  (academy.jump - academy.jump_fetch - academy.jump_spin + academy.all_three) +
  (academy.fetch - academy.jump_fetch - academy.fetch_spin + academy.all_three) +
  (academy.spin - academy.jump_spin - academy.fetch_spin + academy.all_three) +
  academy.jump_fetch + academy.fetch_spin + academy.jump_spin -
  2 * academy.all_three + academy.none = 99 :=
by sorry

end NUMINAMATH_CALUDE_total_cats_l1459_145965


namespace NUMINAMATH_CALUDE_sin_690_degrees_l1459_145987

theorem sin_690_degrees : Real.sin (690 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l1459_145987


namespace NUMINAMATH_CALUDE_simplify_expression_l1459_145946

theorem simplify_expression (x y : ℝ) : (3*x - 5*y) + (4*x + 5*y) = 7*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1459_145946


namespace NUMINAMATH_CALUDE_product_unit_digit_l1459_145972

def unit_digit (n : ℕ) : ℕ := n % 10

theorem product_unit_digit : 
  unit_digit (624 * 708 * 913 * 463) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_unit_digit_l1459_145972


namespace NUMINAMATH_CALUDE_fraction_power_cube_l1459_145931

theorem fraction_power_cube : (2 / 5 : ℚ) ^ 3 = 8 / 125 := by sorry

end NUMINAMATH_CALUDE_fraction_power_cube_l1459_145931


namespace NUMINAMATH_CALUDE_unique_K_value_l1459_145983

theorem unique_K_value : ∃! K : ℕ, 
  (∃ Z : ℕ, 1000 < Z ∧ Z < 8000 ∧ K > 2 ∧ Z = K * K^2) ∧ 
  (∃ a b : ℕ, K^3 = a^2 ∧ K^3 = b^3) ∧
  K = 16 :=
sorry

end NUMINAMATH_CALUDE_unique_K_value_l1459_145983


namespace NUMINAMATH_CALUDE_connie_marbles_l1459_145945

theorem connie_marbles (marbles_given : ℕ) (marbles_left : ℕ) : 
  marbles_given = 183 → marbles_left = 593 → marbles_given + marbles_left = 776 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l1459_145945


namespace NUMINAMATH_CALUDE_library_biography_increase_l1459_145922

theorem library_biography_increase (B : ℝ) (h1 : B > 0) : 
  let original_biographies := 0.20 * B
  let new_biographies := (7 / 9) * B
  let percentage_increase := (new_biographies / original_biographies - 1) * 100
  percentage_increase = 3500 / 9 := by sorry

end NUMINAMATH_CALUDE_library_biography_increase_l1459_145922


namespace NUMINAMATH_CALUDE_last_digit_padic_fermat_l1459_145925

/-- Represents a p-adic integer with a non-zero last digit -/
structure PAdic (p : ℕ) where
  digits : ℕ → ℕ
  last_nonzero : digits 0 ≠ 0
  bound : ∀ n, digits n < p

/-- The last digit of a p-adic number -/
def last_digit {p : ℕ} (a : PAdic p) : ℕ := a.digits 0

/-- Exponentiation for p-adic numbers -/
def padic_pow {p : ℕ} (a : PAdic p) (n : ℕ) : PAdic p :=
  sorry

/-- Subtraction for p-adic numbers -/
def padic_sub {p : ℕ} (a b : PAdic p) : PAdic p :=
  sorry

theorem last_digit_padic_fermat (p : ℕ) (hp : Prime p) (a : PAdic p) :
  last_digit (padic_sub (padic_pow a (p - 1)) (PAdic.mk (λ _ => 1) sorry sorry)) = 0 :=
sorry

end NUMINAMATH_CALUDE_last_digit_padic_fermat_l1459_145925


namespace NUMINAMATH_CALUDE_data_set_range_is_67_l1459_145978

-- Define a structure for our data set
structure DataSet where
  points : List ℝ
  min_value : ℝ
  max_value : ℝ
  h_min : min_value ∈ points
  h_max : max_value ∈ points
  h_lower_bound : ∀ x ∈ points, min_value ≤ x
  h_upper_bound : ∀ x ∈ points, x ≤ max_value

-- Define the range of a data set
def range (d : DataSet) : ℝ := d.max_value - d.min_value

-- Theorem statement
theorem data_set_range_is_67 (d : DataSet) 
  (h_min : d.min_value = 31)
  (h_max : d.max_value = 98) : 
  range d = 67 := by
  sorry

end NUMINAMATH_CALUDE_data_set_range_is_67_l1459_145978


namespace NUMINAMATH_CALUDE_correct_calculation_l1459_145989

theorem correct_calculation (x : ℝ) (h : x ≠ 0) : (x^2 + x) / x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1459_145989


namespace NUMINAMATH_CALUDE_bus_stop_time_l1459_145905

/-- Calculates the stop time of a bus given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) :
  speed_without_stops = 54 →
  speed_with_stops = 45 →
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 10 := by
  sorry

#check bus_stop_time

end NUMINAMATH_CALUDE_bus_stop_time_l1459_145905


namespace NUMINAMATH_CALUDE_zero_point_of_odd_function_l1459_145903

/-- A function f is odd if f(-x) = -f(x) for all x. -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem zero_point_of_odd_function (f : ℝ → ℝ) (x₀ : ℝ) :
  IsOdd f →
  f x₀ + Real.exp x₀ = 0 →
  Real.exp (-x₀) * f (-x₀) - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_point_of_odd_function_l1459_145903


namespace NUMINAMATH_CALUDE_negative_two_cubed_equality_l1459_145939

theorem negative_two_cubed_equality : (-2)^3 = -2^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_equality_l1459_145939


namespace NUMINAMATH_CALUDE_green_marble_probability_l1459_145933

/-- Represents a bag of marbles with specific colors and quantities -/
structure MarbleBag where
  color1 : String
  count1 : ℕ
  color2 : String
  count2 : ℕ

/-- Calculates the probability of drawing a specific color from a bag -/
def drawProbability (bag : MarbleBag) (color : String) : ℚ :=
  if color == bag.color1 then
    bag.count1 / (bag.count1 + bag.count2)
  else if color == bag.color2 then
    bag.count2 / (bag.count1 + bag.count2)
  else
    0

/-- The main theorem stating the probability of drawing a green marble -/
theorem green_marble_probability
  (bagX : MarbleBag)
  (bagY : MarbleBag)
  (bagZ : MarbleBag)
  (hX : bagX = ⟨"white", 5, "black", 5⟩)
  (hY : bagY = ⟨"green", 4, "red", 6⟩)
  (hZ : bagZ = ⟨"green", 3, "purple", 7⟩) :
  let probWhiteX := drawProbability bagX "white"
  let probGreenY := drawProbability bagY "green"
  let probBlackX := drawProbability bagX "black"
  let probGreenZ := drawProbability bagZ "green"
  probWhiteX * probGreenY + probBlackX * probGreenZ = 7 / 20 := by
  sorry


end NUMINAMATH_CALUDE_green_marble_probability_l1459_145933


namespace NUMINAMATH_CALUDE_line_parallel_plane_perpendicular_implies_perpendicular_l1459_145941

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Parallel relation between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Perpendicular relation between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Perpendicular relation between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop := sorry

/-- Theorem: If a line is parallel to a plane and another line is perpendicular to the same plane, 
    then the two lines are perpendicular to each other -/
theorem line_parallel_plane_perpendicular_implies_perpendicular 
  (m n : Line3D) (α : Plane3D) 
  (h1 : parallel m α) 
  (h2 : perpendicular_line_plane n α) : 
  perpendicular_lines m n := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_plane_perpendicular_implies_perpendicular_l1459_145941


namespace NUMINAMATH_CALUDE_travelers_meeting_l1459_145948

/-- The problem of two travelers meeting --/
theorem travelers_meeting
  (total_distance : ℝ)
  (travel_time : ℝ)
  (shook_speed : ℝ)
  (h_total_distance : total_distance = 490)
  (h_travel_time : travel_time = 7)
  (h_shook_speed : shook_speed = 37)
  : ∃ (beta_speed : ℝ),
    beta_speed = 33 ∧
    total_distance = shook_speed * travel_time + beta_speed * travel_time :=
by
  sorry

#check travelers_meeting

end NUMINAMATH_CALUDE_travelers_meeting_l1459_145948


namespace NUMINAMATH_CALUDE_distinct_sums_theorem_l1459_145963

theorem distinct_sums_theorem (k n : ℕ) (a b c : Fin n → ℝ) :
  k ≥ 3 →
  n > Nat.choose k 3 →
  (∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ b i ≠ b j ∧ c i ≠ c j) →
  ∃ S : Finset ℝ,
    S.card ≥ k + 1 ∧
    (∀ i : Fin n, (a i + b i) ∈ S ∧ (a i + c i) ∈ S ∧ (b i + c i) ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_distinct_sums_theorem_l1459_145963


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_three_l1459_145908

theorem sqrt_expression_equals_sqrt_three :
  Real.sqrt 48 - 6 * Real.sqrt (1/3) - Real.sqrt 18 / Real.sqrt 6 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_three_l1459_145908


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_l1459_145997

/-- A function that checks if all digits of a natural number are even -/
def allDigitsEven (n : ℕ) : Prop := sorry

/-- A function that returns the largest positive integer less than 10000 
    with all even digits that is a multiple of 5 -/
noncomputable def largestEvenDigitMultipleOf5 : ℕ := sorry

/-- Theorem stating that 8860 is the largest positive integer less than 10000 
    with all even digits that is a multiple of 5 -/
theorem largest_even_digit_multiple_of_5 : 
  largestEvenDigitMultipleOf5 = 8860 ∧ 
  allDigitsEven 8860 ∧ 
  8860 < 10000 ∧ 
  8860 % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_l1459_145997


namespace NUMINAMATH_CALUDE_line_through_circle_center_l1459_145991

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 1 = 0

-- Define the line l
def line_l (x y m : ℝ) : Prop := x + m*y + 1 = 0

-- Define the center of the circle
def center_C : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem line_through_circle_center (m : ℝ) :
  line_l (center_C.1) (center_C.2) m → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l1459_145991


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1459_145926

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := -3
  let n : ℕ := 7
  let S := (a * (r^n - 1)) / (r - 1)
  ((-3)^6 = 729) → ((-3)^7 = -2187) → S = 547 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1459_145926


namespace NUMINAMATH_CALUDE_max_value_S_l1459_145936

/-- The maximum value of S given the conditions -/
theorem max_value_S (a b : ℝ) (ha : a > 0) (hb : b > 0) (hline : 2 * a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 
    2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≥ 2 * Real.sqrt (x * y) - 4 * x^2 - y^2) →
  2 * Real.sqrt (a * b) - 4 * a^2 - b^2 = (Real.sqrt 2 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_S_l1459_145936


namespace NUMINAMATH_CALUDE_compare_expressions_inequality_proof_l1459_145947

-- Part 1
theorem compare_expressions (x : ℝ) : (x + 7) * (x + 8) > (x + 6) * (x + 9) := by
  sorry

-- Part 2
theorem inequality_proof (a b c d : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : 0 < c) (h4 : c < d) :
  a * d + c < b * c + d := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_inequality_proof_l1459_145947


namespace NUMINAMATH_CALUDE_log_identity_l1459_145954

theorem log_identity : Real.log 16 / Real.log 4 - (Real.log 3 / Real.log 2) * (Real.log 2 / Real.log 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l1459_145954


namespace NUMINAMATH_CALUDE_complex_function_property_l1459_145995

/-- A function g on complex numbers defined by g(z) = (c+di)z, where c and d are real numbers. -/
def g (c d : ℝ) (z : ℂ) : ℂ := (c + d * Complex.I) * z

/-- The theorem stating that if g(z) = (c+di)z where c and d are real numbers, 
    and for all complex z, |g(z) - z| = |g(z)|, and |c+di| = 7, then d^2 = 195/4. -/
theorem complex_function_property (c d : ℝ) : 
  (∀ z : ℂ, Complex.abs (g c d z - z) = Complex.abs (g c d z)) → 
  Complex.abs (c + d * Complex.I) = 7 → 
  d^2 = 195/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_function_property_l1459_145995


namespace NUMINAMATH_CALUDE_two_p_plus_q_l1459_145967

theorem two_p_plus_q (p q : ℚ) (h : p / q = 3 / 5) : 2 * p + q = (11 / 5) * q := by
  sorry

end NUMINAMATH_CALUDE_two_p_plus_q_l1459_145967


namespace NUMINAMATH_CALUDE_q_polynomial_form_l1459_145998

theorem q_polynomial_form (x : ℝ) (q : ℝ → ℝ) 
  (h : ∀ x, q x + (x^6 + 4*x^4 + 8*x^2 + 7*x) = 12*x^4 + 30*x^3 + 40*x^2 + 10*x + 2) :
  q x = -x^6 + 8*x^4 + 30*x^3 + 32*x^2 + 3*x + 2 := by
sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l1459_145998


namespace NUMINAMATH_CALUDE_first_group_size_l1459_145980

/-- The number of men in the first group -/
def M : ℕ := sorry

/-- The number of acres that can be reaped by M men in 15 days -/
def acres_first_group : ℕ := 120

/-- The number of days it takes M men to reap 120 acres -/
def days_first_group : ℕ := 15

/-- The number of men in the second group -/
def men_second_group : ℕ := 20

/-- The number of acres that can be reaped by 20 men in 30 days -/
def acres_second_group : ℕ := 480

/-- The number of days it takes 20 men to reap 480 acres -/
def days_second_group : ℕ := 30

theorem first_group_size :
  M = 10 :=
sorry

end NUMINAMATH_CALUDE_first_group_size_l1459_145980


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1459_145990

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  8 * x^4 + 12 * y^4 + 18 * z^4 + 25 / (x * y * z) ≥ 30 :=
by sorry

theorem min_value_attained :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  8 * x^4 + 12 * y^4 + 18 * z^4 + 25 / (x * y * z) = 30 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1459_145990


namespace NUMINAMATH_CALUDE_correct_non_attacking_placements_non_attacking_placements_positive_l1459_145920

/-- Represents a chess piece type -/
inductive ChessPiece
  | Rook
  | King
  | Bishop
  | Knight
  | Queen

/-- Represents the dimensions of a chessboard -/
def BoardSize : Nat := 8

/-- Calculates the number of ways to place two pieces of the same type on a chessboard without attacking each other -/
def nonAttackingPlacements (piece : ChessPiece) : Nat :=
  match piece with
  | ChessPiece.Rook => 1568
  | ChessPiece.King => 1806
  | ChessPiece.Bishop => 1736
  | ChessPiece.Knight => 1848
  | ChessPiece.Queen => 1288

/-- Theorem stating the correct number of non-attacking placements for each piece type -/
theorem correct_non_attacking_placements :
  (nonAttackingPlacements ChessPiece.Rook = 1568) ∧
  (nonAttackingPlacements ChessPiece.King = 1806) ∧
  (nonAttackingPlacements ChessPiece.Bishop = 1736) ∧
  (nonAttackingPlacements ChessPiece.Knight = 1848) ∧
  (nonAttackingPlacements ChessPiece.Queen = 1288) := by
  sorry

/-- Theorem stating that the number of non-attacking placements is always positive -/
theorem non_attacking_placements_positive (piece : ChessPiece) :
  nonAttackingPlacements piece > 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_non_attacking_placements_non_attacking_placements_positive_l1459_145920


namespace NUMINAMATH_CALUDE_largest_divisor_of_four_consecutive_naturals_l1459_145912

theorem largest_divisor_of_four_consecutive_naturals :
  ∀ n : ℕ, ∃ k : ℕ, k * 120 = n * (n + 1) * (n + 2) * (n + 3) ∧
  ∀ m : ℕ, m > 120 → ¬(∀ n : ℕ, ∃ k : ℕ, k * m = n * (n + 1) * (n + 2) * (n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_four_consecutive_naturals_l1459_145912


namespace NUMINAMATH_CALUDE_expression_values_l1459_145985

theorem expression_values (m n : ℕ) (h : m * n ≠ 1) :
  let expr := (m^2 + m*n + n^2) / (m*n - 1)
  expr ∈ ({0, 4, 7} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_expression_values_l1459_145985


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1459_145907

theorem partial_fraction_decomposition (x : ℝ) (A B : ℚ) : 
  (5 * x - 3) / ((x - 3) * (x + 6)) = A / (x - 3) + B / (x + 6) ↔ 
  A = 4/3 ∧ B = 11/3 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1459_145907


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1459_145988

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x : ℝ, x < 0 → x^2019 < x^2018)) ↔ 
  (∃ x : ℝ, x < 0 ∧ x^2019 ≥ x^2018) := by
sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1459_145988


namespace NUMINAMATH_CALUDE_range_of_a_l1459_145984

/-- Given functions f and g, prove the range of a -/
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, |2*x₁ - a| + |2*x₁ + 3| = |2*x₂ - 3| + 2) →
  (a ≥ -1 ∨ a ≤ -5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1459_145984


namespace NUMINAMATH_CALUDE_more_birds_than_nests_l1459_145943

/-- Given 6 birds and 3 nests, prove that there are 3 more birds than nests. -/
theorem more_birds_than_nests (birds : ℕ) (nests : ℕ) 
  (h1 : birds = 6) (h2 : nests = 3) : birds - nests = 3 := by
  sorry

end NUMINAMATH_CALUDE_more_birds_than_nests_l1459_145943


namespace NUMINAMATH_CALUDE_train_length_l1459_145924

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (crossing_time : ℝ) : 
  speed_kmph = 18 → crossing_time = 5 → 
  (speed_kmph * 1000 / 3600) * crossing_time = 25 := by sorry

end NUMINAMATH_CALUDE_train_length_l1459_145924


namespace NUMINAMATH_CALUDE_combination_square_28_l1459_145958

theorem combination_square_28 (n : ℕ) : (n.choose 2 = 28) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_combination_square_28_l1459_145958


namespace NUMINAMATH_CALUDE_complex_magnitude_l1459_145928

theorem complex_magnitude (z : ℂ) : z = 5 / (2 + Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1459_145928


namespace NUMINAMATH_CALUDE_max_area_rhombus_l1459_145968

/-- Given a rhombus OABC in a rectangular coordinate system xOy with the following properties:
  - The diagonals intersect at point M(x₀, y₀)
  - The hyperbola y = k/x (x > 0) passes through points C and M
  - 2 ≤ x₀ ≤ 4
  Prove that the maximum area of rhombus OABC is 24√2 -/
theorem max_area_rhombus (x₀ y₀ k : ℝ) (hx₀ : 2 ≤ x₀ ∧ x₀ ≤ 4) (hk : k > 0) 
  (h_hyperbola : y₀ = k / x₀) : 
  (∃ (S : ℝ), S = 24 * Real.sqrt 2 ∧ 
    ∀ (A : ℝ), A ≤ S ∧ 
    (∃ (x₁ y₁ : ℝ), 2 ≤ x₁ ∧ x₁ ≤ 4 ∧ 
      y₁ = k / x₁ ∧ 
      A = (3 * Real.sqrt 2 / 2) * x₁^2)) := by
  sorry

end NUMINAMATH_CALUDE_max_area_rhombus_l1459_145968


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l1459_145950

theorem divisibility_of_sum_of_squares (p x y z : ℕ) : 
  Prime p → 
  0 < x → x < y → y < z → z < p → 
  (x^3 % p = y^3 % p) → (y^3 % p = z^3 % p) →
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l1459_145950


namespace NUMINAMATH_CALUDE_function_decomposition_into_symmetric_parts_l1459_145956

/-- A function is symmetric about the y-axis if f(x) = f(-x) for all x ∈ ℝ -/
def SymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

/-- A function is symmetric about the vertical line x = a if f(x) = f(2a - x) for all x ∈ ℝ -/
def SymmetricAboutVerticalLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f x = f (2 * a - x)

/-- Main theorem: Any function on ℝ can be represented as the sum of two symmetric functions -/
theorem function_decomposition_into_symmetric_parts (f : ℝ → ℝ) :
  ∃ (f₁ f₂ : ℝ → ℝ) (a : ℝ),
    (∀ x : ℝ, f x = f₁ x + f₂ x) ∧
    SymmetricAboutYAxis f₁ ∧
    a > 0 ∧
    SymmetricAboutVerticalLine f₂ a :=
  sorry

end NUMINAMATH_CALUDE_function_decomposition_into_symmetric_parts_l1459_145956


namespace NUMINAMATH_CALUDE_zongzi_pricing_and_purchase_l1459_145993

-- Define the total number of zongzi and total cost
def total_zongzi : ℕ := 1100
def total_cost : ℚ := 3000

-- Define the price ratio between type A and B
def price_ratio : ℚ := 1.2

-- Define the new total number of zongzi and budget
def new_total_zongzi : ℕ := 2600
def new_budget : ℚ := 7000

-- Define the unit prices of type A and B zongzi
def unit_price_B : ℚ := 2.5
def unit_price_A : ℚ := 3

-- Define the maximum number of type A zongzi in the second scenario
def max_type_A : ℕ := 1000

theorem zongzi_pricing_and_purchase :
  -- The cost of purchasing type A is the same as type B
  (total_cost / 2) / unit_price_A = (total_cost / 2) / unit_price_B ∧
  -- The unit price of type A is 1.2 times the unit price of type B
  unit_price_A = price_ratio * unit_price_B ∧
  -- The total number of zongzi purchased is 1100
  (total_cost / 2) / unit_price_A + (total_cost / 2) / unit_price_B = total_zongzi ∧
  -- The maximum number of type A zongzi in the second scenario is 1000
  max_type_A * unit_price_A + (new_total_zongzi - max_type_A) * unit_price_B ≤ new_budget ∧
  ∀ n : ℕ, n > max_type_A → n * unit_price_A + (new_total_zongzi - n) * unit_price_B > new_budget :=
by sorry

end NUMINAMATH_CALUDE_zongzi_pricing_and_purchase_l1459_145993


namespace NUMINAMATH_CALUDE_soda_price_ratio_l1459_145910

/-- Represents the volume and price of a soda brand relative to Brand Y -/
structure SodaBrand where
  volume : ℚ  -- Relative volume compared to Brand Y
  price : ℚ   -- Relative price compared to Brand Y

/-- Calculates the unit price of a soda brand -/
def unitPrice (brand : SodaBrand) : ℚ :=
  brand.price / brand.volume

theorem soda_price_ratio :
  let brand_x : SodaBrand := { volume := 13/10, price := 17/20 }
  let brand_z : SodaBrand := { volume := 14/10, price := 11/10 }
  (unitPrice brand_z) / (unitPrice brand_x) = 13/11 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l1459_145910


namespace NUMINAMATH_CALUDE_telescope_visual_range_increase_l1459_145952

theorem telescope_visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 90)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_increase_l1459_145952


namespace NUMINAMATH_CALUDE_max_reflections_is_nine_l1459_145994

/-- Represents the angle between two mirrors in degrees -/
def mirror_angle : ℝ := 10

/-- Represents the increase in angle of incidence after each reflection in degrees -/
def angle_increase : ℝ := 10

/-- Represents the maximum angle at which reflection is possible in degrees -/
def max_reflection_angle : ℝ := 90

/-- Calculates the angle of incidence after a given number of reflections -/
def angle_after_reflections (n : ℕ) : ℝ := n * angle_increase

/-- Determines if reflection is possible after a given number of reflections -/
def is_reflection_possible (n : ℕ) : Prop :=
  angle_after_reflections n ≤ max_reflection_angle

/-- The maximum number of reflections possible -/
def max_reflections : ℕ := 9

/-- Theorem stating that the maximum number of reflections is 9 -/
theorem max_reflections_is_nine :
  (∀ n : ℕ, is_reflection_possible n → n ≤ max_reflections) ∧
  is_reflection_possible max_reflections ∧
  ¬is_reflection_possible (max_reflections + 1) :=
sorry

end NUMINAMATH_CALUDE_max_reflections_is_nine_l1459_145994


namespace NUMINAMATH_CALUDE_gcf_lcm_360_210_l1459_145979

theorem gcf_lcm_360_210 : 
  (Nat.gcd 360 210 = 30) ∧ (Nat.lcm 360 210 = 2520) := by
sorry

end NUMINAMATH_CALUDE_gcf_lcm_360_210_l1459_145979


namespace NUMINAMATH_CALUDE_playground_boys_count_l1459_145981

theorem playground_boys_count (total_children girls : ℕ) 
  (h1 : total_children = 117) 
  (h2 : girls = 77) : 
  total_children - girls = 40 := by
sorry

end NUMINAMATH_CALUDE_playground_boys_count_l1459_145981


namespace NUMINAMATH_CALUDE_twelve_digit_divisibility_l1459_145957

theorem twelve_digit_divisibility (n : ℕ) (h : 100000 ≤ n ∧ n < 1000000) :
  ∃ k : ℕ, 1000001 * n + n = 1000001 * k := by
  sorry

end NUMINAMATH_CALUDE_twelve_digit_divisibility_l1459_145957


namespace NUMINAMATH_CALUDE_solve_system_l1459_145915

theorem solve_system (x y : ℚ) : 
  (1 / 3 - 1 / 4 = 1 / x) → (x + y = 10) → (x = 12 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1459_145915


namespace NUMINAMATH_CALUDE_H_surjective_l1459_145951

-- Define the function H
def H (x : ℝ) : ℝ := 2 * |2 * x + 3| - 3 * |x - 2|

-- Theorem statement
theorem H_surjective : Function.Surjective H := by sorry

end NUMINAMATH_CALUDE_H_surjective_l1459_145951


namespace NUMINAMATH_CALUDE_servings_per_jar_l1459_145930

/-- The number of servings of peanut butter consumed per day -/
def servings_per_day : ℕ := 2

/-- The number of days the peanut butter should last -/
def days : ℕ := 30

/-- The number of jars needed to last for the given number of days -/
def jars : ℕ := 4

/-- Theorem stating that each jar contains 15 servings of peanut butter -/
theorem servings_per_jar : 
  (servings_per_day * days) / jars = 15 := by sorry

end NUMINAMATH_CALUDE_servings_per_jar_l1459_145930


namespace NUMINAMATH_CALUDE_first_supply_cost_l1459_145974

theorem first_supply_cost (total_budget : ℕ) (remaining_budget : ℕ) (second_supply_cost : ℕ) :
  total_budget = 56 →
  remaining_budget = 19 →
  second_supply_cost = 24 →
  total_budget - remaining_budget - second_supply_cost = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_first_supply_cost_l1459_145974


namespace NUMINAMATH_CALUDE_triangle_area_problem_l1459_145909

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * a * b * Real.sin C

theorem triangle_area_problem (a b c : ℝ) (A B C : ℝ) 
  (h1 : c^2 = (a-b)^2 + 6)
  (h2 : C = π/3) :
  triangle_area a b c A B C = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l1459_145909


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l1459_145975

theorem cookie_jar_problem :
  ∃ C : ℕ, (C - 1 = (C + 5) / 2) ∧ (C = 7) :=
by sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l1459_145975


namespace NUMINAMATH_CALUDE_reciprocal_problem_l1459_145901

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 4) : 200 * (1 / x) = 400 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l1459_145901


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l1459_145906

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 5 ∧ b > 5 ∧
    n = 1 * a + 5 ∧
    n = 5 * b + 1

theorem smallest_dual_base_representation :
  (∀ m : ℕ, m < 31 → ¬ is_valid_representation m) ∧
  is_valid_representation 31 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l1459_145906


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1459_145934

theorem unique_integer_solution : ∃! (d e f : ℕ+), 
  let x : ℝ := Real.sqrt ((Real.sqrt 77 / 2) + (5 / 2))
  x^100 = 3*x^98 + 18*x^96 + 13*x^94 - x^50 + d*x^46 + e*x^44 + f*x^40 ∧ 
  d + e + f = 86 := by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1459_145934


namespace NUMINAMATH_CALUDE_fractional_expression_transformation_l1459_145937

theorem fractional_expression_transformation (x : ℝ) :
  let A : ℝ → ℝ := λ x => x^2 - 2*x
  x / (x + 2) = A x / (x^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_fractional_expression_transformation_l1459_145937


namespace NUMINAMATH_CALUDE_frying_time_correct_l1459_145960

/-- Calculates the minimum time required to fry n pancakes -/
def min_frying_time (n : ℕ) : ℕ :=
  if n ≤ 2 then
    4
  else if n % 2 = 0 then
    2 * n
  else
    2 * (n - 1) + 2

theorem frying_time_correct :
  (min_frying_time 3 = 6) ∧ (min_frying_time 2016 = 4032) := by
  sorry

#eval min_frying_time 3
#eval min_frying_time 2016

end NUMINAMATH_CALUDE_frying_time_correct_l1459_145960


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1459_145902

/-- A quadratic function f(x) = ax^2 + 2x + c with range [2, +∞) -/
def QuadraticFunction (a c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + 2 * x + c

/-- The range of the quadratic function is [2, +∞) -/
def HasRange (f : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, y ≥ 2 → ∃ x : ℝ, f x = y

theorem min_value_of_reciprocal_sum (a c : ℝ) :
  a > 0 →
  c > 0 →
  HasRange (QuadraticFunction a c) →
  (∀ x : ℝ, 1 / a + 9 / c ≥ 4) ∧ (∃ x : ℝ, 1 / a + 9 / c = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1459_145902


namespace NUMINAMATH_CALUDE_rain_probability_l1459_145976

theorem rain_probability (p_friday p_monday : ℝ) 
  (h1 : p_friday = 0.3)
  (h2 : p_monday = 0.2)
  (h3 : 0 ≤ p_friday ∧ p_friday ≤ 1)
  (h4 : 0 ≤ p_monday ∧ p_monday ≤ 1) :
  1 - (1 - p_friday) * (1 - p_monday) = 0.44 := by
sorry

end NUMINAMATH_CALUDE_rain_probability_l1459_145976


namespace NUMINAMATH_CALUDE_phones_sold_is_four_l1459_145966

/-- Calculates the total number of cell phones sold given the initial and final counts
    of Samsung phones and iPhones, as well as the number of damaged/defective phones. -/
def total_phones_sold (initial_samsung : ℕ) (final_samsung : ℕ) (initial_iphone : ℕ) 
                      (final_iphone : ℕ) (damaged_samsung : ℕ) (defective_iphone : ℕ) : ℕ :=
  (initial_samsung - final_samsung - damaged_samsung) + 
  (initial_iphone - final_iphone - defective_iphone)

/-- Theorem stating that the total number of cell phones sold is 4 given the specific
    initial and final counts, and the number of damaged/defective phones. -/
theorem phones_sold_is_four : 
  total_phones_sold 14 10 8 5 2 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_phones_sold_is_four_l1459_145966


namespace NUMINAMATH_CALUDE_piggy_bank_savings_l1459_145955

theorem piggy_bank_savings (x y : ℕ) : 
  x + y = 290 →  -- Total number of coins
  2 * (y / 4) = x / 3 →  -- Relationship between coin values
  2 * y + x = 406  -- Total amount saved
  := by sorry

end NUMINAMATH_CALUDE_piggy_bank_savings_l1459_145955


namespace NUMINAMATH_CALUDE_right_triangle_sine_roots_l1459_145962

theorem right_triangle_sine_roots (A B C : Real) (p q : Real) :
  0 < A ∧ A < Real.pi / 2 →
  0 < B ∧ B < Real.pi / 2 →
  C = Real.pi / 2 →
  A + B + C = Real.pi →
  (∀ x, x^2 + p*x + q = 0 ↔ x = Real.sin A ∨ x = Real.sin B) →
  (p^2 - 2*q = 1 ∧ -Real.sqrt 2 ≤ p ∧ p < -1 ∧ 0 < q ∧ q ≤ 1/2) ∧
  (∀ x, x^2 + p*x + q = 0 → (x = Real.sin A ∨ x = Real.sin B)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sine_roots_l1459_145962


namespace NUMINAMATH_CALUDE_sin_3phi_l1459_145940

theorem sin_3phi (φ : ℝ) (h : Complex.exp (φ * Complex.I) = (1 + 3 * Complex.I) / (2 * Real.sqrt 2)) :
  Real.sin (3 * φ) = 15 / (16 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_3phi_l1459_145940


namespace NUMINAMATH_CALUDE_expected_heads_3000_tosses_l1459_145900

/-- A coin toss experiment with a fair coin -/
structure CoinTossExperiment where
  numTosses : ℕ
  probHeads : ℝ
  probHeads_eq : probHeads = 0.5

/-- The expected frequency of heads in a coin toss experiment -/
def expectedHeads (e : CoinTossExperiment) : ℝ :=
  e.numTosses * e.probHeads

/-- Theorem: The expected frequency of heads for 3000 tosses of a fair coin is 1500 -/
theorem expected_heads_3000_tosses (e : CoinTossExperiment) 
    (h : e.numTosses = 3000) : expectedHeads e = 1500 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_3000_tosses_l1459_145900


namespace NUMINAMATH_CALUDE_total_comics_in_box_l1459_145927

-- Define the problem parameters
def pages_per_comic : ℕ := 25
def found_pages : ℕ := 150
def untorn_comics : ℕ := 5

-- State the theorem
theorem total_comics_in_box : 
  (found_pages / pages_per_comic) + untorn_comics = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_comics_in_box_l1459_145927


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1459_145938

theorem condition_necessary_not_sufficient (x y : ℝ) :
  (x + y > 3 → (x > 1 ∨ y > 2)) ∧
  ¬((x > 1 ∨ y > 2) → x + y > 3) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1459_145938


namespace NUMINAMATH_CALUDE_cyclic_iff_perpendicular_l1459_145986

-- Define the basic structures
structure Point := (x : ℝ) (y : ℝ)

structure Quadrilateral :=
  (A B C D : Point)

-- Define the properties
def is_convex (q : Quadrilateral) : Prop := sorry

def are_perpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

def is_intersection (p : Point) (p1 p2 p3 p4 : Point) : Prop := sorry

def is_midpoint (m : Point) (p1 p2 : Point) : Prop := sorry

def is_cyclic (q : Quadrilateral) : Prop := sorry

-- Main theorem
theorem cyclic_iff_perpendicular (q : Quadrilateral) (P M : Point) :
  is_convex q →
  are_perpendicular q.A q.C q.B q.D →
  is_intersection P q.A q.C q.B q.D →
  is_midpoint M q.A q.B →
  (is_cyclic q ↔ are_perpendicular P M q.D q.C) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_iff_perpendicular_l1459_145986


namespace NUMINAMATH_CALUDE_no_real_solutions_for_equation_l1459_145982

theorem no_real_solutions_for_equation :
  ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 1/a + 1/b = 1/(a+b) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_equation_l1459_145982


namespace NUMINAMATH_CALUDE_ratio_of_part_to_whole_l1459_145935

theorem ratio_of_part_to_whole (N : ℝ) : 
  (1 / 1) * (1 / 3) * (2 / 5) * N = 10 →
  (40 / 100) * N = 120 →
  (10 : ℝ) / ((1 / 3) * (2 / 5) * N) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_part_to_whole_l1459_145935


namespace NUMINAMATH_CALUDE_larger_interior_angle_measure_l1459_145949

/-- A circular monument consisting of congruent isosceles trapezoids. -/
structure CircularMonument where
  /-- The number of trapezoids in the monument. -/
  num_trapezoids : ℕ
  /-- The measure of the larger interior angle of each trapezoid in degrees. -/
  larger_interior_angle : ℝ

/-- The properties of the circular monument. -/
def monument_properties (m : CircularMonument) : Prop :=
  m.num_trapezoids = 12 ∧
  m.larger_interior_angle > 0 ∧
  m.larger_interior_angle < 180

/-- Theorem stating the measure of the larger interior angle in the monument. -/
theorem larger_interior_angle_measure (m : CircularMonument) 
  (h : monument_properties m) : m.larger_interior_angle = 97.5 := by
  sorry

#check larger_interior_angle_measure

end NUMINAMATH_CALUDE_larger_interior_angle_measure_l1459_145949


namespace NUMINAMATH_CALUDE_solution_in_interval_l1459_145918

theorem solution_in_interval (x₀ : ℝ) : 
  (Real.log x₀ + x₀ - 3 = 0) → (2 < x₀ ∧ x₀ < 2.5) := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l1459_145918


namespace NUMINAMATH_CALUDE_all_chameleons_red_l1459_145999

/-- Represents the color of a chameleon -/
inductive Color
  | Yellow
  | Green
  | Red

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  yellow : Nat
  green : Nat
  red : Nat

/-- The initial state of chameleons on the island -/
def initialState : ChameleonState :=
  { yellow := 7, green := 10, red := 17 }

/-- The total number of chameleons on the island -/
def totalChameleons : Nat := 34

/-- Calculates the difference between red and green chameleons -/
def delta (state : ChameleonState) : Int :=
  state.red - state.green

/-- Represents a single interaction between two chameleons -/
def interaction (c1 c2 : Color) : Color :=
  match c1, c2 with
  | Color.Yellow, Color.Green => Color.Red
  | Color.Yellow, Color.Red => Color.Green
  | Color.Green, Color.Red => Color.Yellow
  | Color.Green, Color.Yellow => Color.Red
  | Color.Red, Color.Yellow => Color.Green
  | Color.Red, Color.Green => Color.Yellow
  | _, _ => c1  -- Same color, no change

/-- The theorem to be proved -/
theorem all_chameleons_red (state : ChameleonState) : 
  state.yellow + state.green + state.red = totalChameleons →
  delta state ≠ 0 →
  ∃ (finalState : ChameleonState), 
    finalState.yellow = 0 ∧ 
    finalState.green = 0 ∧ 
    finalState.red = totalChameleons :=
by
  sorry

end NUMINAMATH_CALUDE_all_chameleons_red_l1459_145999


namespace NUMINAMATH_CALUDE_min_value_sum_products_l1459_145977

theorem min_value_sum_products (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = a * b * c) (h2 : a + b + c = a^3) :
  ∀ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = x * y * z ∧ x + y + z = x^3 →
  x * y + y * z + z * x ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_products_l1459_145977


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1459_145913

/-- Given a boat that travels 20 km downstream in 2 hours and 20 km upstream in 5 hours,
    prove that its speed in still water is 7 km/h. -/
theorem boat_speed_in_still_water :
  ∀ (downstream_speed upstream_speed : ℝ),
  downstream_speed = 20 / 2 →
  upstream_speed = 20 / 5 →
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = downstream_speed ∧
    boat_speed - stream_speed = upstream_speed ∧
    boat_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1459_145913


namespace NUMINAMATH_CALUDE_remaining_pencils_l1459_145942

/-- The number of pencils remaining in a drawer after some are taken. -/
def pencils_remaining (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Theorem stating that 12 pencils remain in the drawer. -/
theorem remaining_pencils :
  pencils_remaining 34 22 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pencils_l1459_145942


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1459_145969

/-- 
Given an arithmetic sequence with:
- First term a₁ = -48
- Common difference d = 5
- Last term aₙ = 72

Prove that the sequence has 25 terms.
-/
theorem arithmetic_sequence_length : 
  let a₁ : ℤ := -48  -- First term
  let d : ℤ := 5     -- Common difference
  let aₙ : ℤ := 72   -- Last term
  ∃ n : ℕ, n = 25 ∧ aₙ = a₁ + (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1459_145969


namespace NUMINAMATH_CALUDE_natural_numbers_less_than_10_l1459_145959

theorem natural_numbers_less_than_10 : 
  {n : ℕ | n < 10} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

end NUMINAMATH_CALUDE_natural_numbers_less_than_10_l1459_145959


namespace NUMINAMATH_CALUDE_jack_hand_in_amount_l1459_145996

/-- Represents the number of bills of each denomination in the till -/
structure TillContents where
  hundreds : Nat
  fifties : Nat
  twenties : Nat
  tens : Nat
  fives : Nat
  ones : Nat

/-- Calculates the total value of the bills in the till -/
def totalValue (t : TillContents) : Nat :=
  100 * t.hundreds + 50 * t.fifties + 20 * t.twenties + 10 * t.tens + 5 * t.fives + t.ones

/-- Calculates the amount to be handed in to the main office -/
def amountToHandIn (t : TillContents) (amountToLeave : Nat) : Nat :=
  totalValue t - amountToLeave

/-- Theorem stating that given Jack's till contents and the amount to leave,
    the amount to hand in is $142 -/
theorem jack_hand_in_amount :
  let jacksTill : TillContents := {
    hundreds := 2,
    fifties := 1,
    twenties := 5,
    tens := 3,
    fives := 7,
    ones := 27
  }
  let amountToLeave := 300
  amountToHandIn jacksTill amountToLeave = 142 := by
  sorry


end NUMINAMATH_CALUDE_jack_hand_in_amount_l1459_145996


namespace NUMINAMATH_CALUDE_delta_calculation_l1459_145917

-- Define the operation Δ
def delta (a b : ℝ) : ℝ := a^3 - b^2

-- State the theorem
theorem delta_calculation :
  delta (3^(delta 5 14)) (4^(delta 4 6)) = -4^56 := by
  sorry

end NUMINAMATH_CALUDE_delta_calculation_l1459_145917


namespace NUMINAMATH_CALUDE_traffic_class_multiple_l1459_145970

theorem traffic_class_multiple (drunk_drivers : ℕ) (total_students : ℕ) (M : ℕ) : 
  drunk_drivers = 6 →
  total_students = 45 →
  total_students = drunk_drivers + (M * drunk_drivers - 3) →
  M = 7 := by
sorry

end NUMINAMATH_CALUDE_traffic_class_multiple_l1459_145970


namespace NUMINAMATH_CALUDE_max_difference_l1459_145923

theorem max_difference (a b : ℝ) (ha : -5 ≤ a ∧ a ≤ 10) (hb : -5 ≤ b ∧ b ≤ 10) :
  ∃ (x y : ℝ), -5 ≤ x ∧ x ≤ 10 ∧ -5 ≤ y ∧ y ≤ 10 ∧ x - y = 15 ∧ ∀ (c d : ℝ), -5 ≤ c ∧ c ≤ 10 ∧ -5 ≤ d ∧ d ≤ 10 → c - d ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_l1459_145923


namespace NUMINAMATH_CALUDE_equation_is_parabola_l1459_145973

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents the equation |y-3| = √((x+1)² + (y-1)²) -/
def conicEquation (p : Point2D) : Prop :=
  |p.y - 3| = Real.sqrt ((p.x + 1)^2 + (p.y - 1)^2)

/-- Defines a parabola as a set of points satisfying a quadratic equation in x or y -/
def isParabola (S : Set Point2D) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧
    (∀ p ∈ S, p.x = a * p.y^2 + b * p.y + c ∨ p.y = a * p.x^2 + b * p.x + c) ∧
    (∀ p, p ∈ S ↔ conicEquation p)

/-- Theorem stating that the given equation represents a parabola -/
theorem equation_is_parabola :
  ∃ S : Set Point2D, isParabola S :=
sorry

end NUMINAMATH_CALUDE_equation_is_parabola_l1459_145973
