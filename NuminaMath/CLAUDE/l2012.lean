import Mathlib

namespace NUMINAMATH_CALUDE_negative_two_minus_six_l2012_201277

theorem negative_two_minus_six : -2 - 6 = -8 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_minus_six_l2012_201277


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2012_201274

theorem solution_set_of_inequality (x : ℝ) :
  (((1 : ℝ) / Real.pi) ^ (-x + 1) > ((1 : ℝ) / Real.pi) ^ (x^2 - x)) ↔ (x > 1 ∨ x < -1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2012_201274


namespace NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_nine_l2012_201204

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem no_two_digit_primes_with_digit_sum_nine :
  ∀ n : ℕ, is_two_digit n → digit_sum n = 9 → ¬ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_nine_l2012_201204


namespace NUMINAMATH_CALUDE_fifteenth_triangular_number_l2012_201228

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem fifteenth_triangular_number : triangular_number 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_triangular_number_l2012_201228


namespace NUMINAMATH_CALUDE_inradius_bounds_l2012_201208

theorem inradius_bounds (a b c r : ℝ) :
  a > 0 → b > 0 → c > 0 →
  c^2 = a^2 + b^2 →
  r = (a + b - c) / 2 →
  r < c / 4 ∧ r < min a b / 2 := by
  sorry

end NUMINAMATH_CALUDE_inradius_bounds_l2012_201208


namespace NUMINAMATH_CALUDE_jake_has_eleven_apples_l2012_201299

-- Define the number of peaches and apples for Steven
def steven_peaches : ℕ := 9
def steven_apples : ℕ := 8

-- Define Jake's peaches and apples in relation to Steven's
def jake_peaches : ℕ := steven_peaches - 13
def jake_apples : ℕ := steven_apples + 3

-- Theorem to prove
theorem jake_has_eleven_apples : jake_apples = 11 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_eleven_apples_l2012_201299


namespace NUMINAMATH_CALUDE_existence_of_integers_l2012_201219

theorem existence_of_integers : ∃ (x y : ℤ), x * y = 4747 ∧ x - y = -54 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l2012_201219


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l2012_201259

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(30.3 : ℝ)⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l2012_201259


namespace NUMINAMATH_CALUDE_nine_integer_chords_l2012_201278

/-- Represents a circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distance_to_p : ℝ

/-- Counts the number of integer-length chords through P -/
def count_integer_chords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem -/
theorem nine_integer_chords :
  let c := CircleWithPoint.mk 20 12
  count_integer_chords c = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_integer_chords_l2012_201278


namespace NUMINAMATH_CALUDE_triangle_side_length_l2012_201288

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2, b = 3, and angle C is twice angle A, then the length of side c is √10. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a = 2 ∧ 
  b = 3 ∧ 
  C = 2 * A ∧  -- Angle C is twice angle A
  a / Real.sin A = b / Real.sin B ∧  -- Sine theorem
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C  -- Cosine theorem
  → c = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2012_201288


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2012_201290

open Set

-- Define the universal set U as the set of integers
def U : Set ℤ := univ

-- Define set M
def M : Set ℤ := {-1, 0, 1}

-- Define set N
def N : Set ℤ := {0, 1, 3}

-- State the theorem
theorem complement_M_intersect_N :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2012_201290


namespace NUMINAMATH_CALUDE_lottery_probabilities_l2012_201210

/-- Represents the total number of lottery tickets -/
def total_tickets : ℕ := 12

/-- Represents the number of winning tickets -/
def winning_tickets : ℕ := 2

/-- Represents the number of people -/
def num_people : ℕ := 4

/-- Represents the probability of giving 2 winning tickets to different people -/
def prob_different_people : ℚ := 9/11

/-- Represents the probability of giving 1 winning ticket to A and 1 to B -/
def prob_A_and_B : ℚ := 3/22

/-- Theorem stating the probabilities for the lottery ticket distribution -/
theorem lottery_probabilities :
  (prob_different_people = 9/11) ∧ (prob_A_and_B = 3/22) :=
sorry

end NUMINAMATH_CALUDE_lottery_probabilities_l2012_201210


namespace NUMINAMATH_CALUDE_chord_length_range_l2012_201265

/-- The chord length intercepted by the line y = x + t on the circle x + y² = 8 -/
def chordLength (t : ℝ) : ℝ := sorry

theorem chord_length_range (t : ℝ) :
  (∀ x y : ℝ, y = x + t ∧ x + y^2 = 8 → chordLength t ≥ 4 * Real.sqrt 2 / 3) →
  t ∈ Set.Icc (-(8 * Real.sqrt 2 / 3)) (8 * Real.sqrt 2 / 3) :=
sorry

end NUMINAMATH_CALUDE_chord_length_range_l2012_201265


namespace NUMINAMATH_CALUDE_roots_sum_angle_l2012_201272

theorem roots_sum_angle (a : ℝ) (α β : ℝ) : 
  a > 2 → 
  α ∈ Set.Ioo (-π/2) (π/2) →
  β ∈ Set.Ioo (-π/2) (π/2) →
  (Real.tan α)^2 + 3*a*(Real.tan α) + 3*a + 1 = 0 →
  (Real.tan β)^2 + 3*a*(Real.tan β) + 3*a + 1 = 0 →
  α + β = π/4 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_angle_l2012_201272


namespace NUMINAMATH_CALUDE_no_super_squarish_numbers_l2012_201297

-- Define a super-squarish number
def is_super_squarish (n : ℕ) : Prop :=
  -- Seven-digit number
  1000000 ≤ n ∧ n < 10000000 ∧
  -- No digit is zero
  ∀ d, (n / 10^d) % 10 ≠ 0 ∧
  -- Perfect square
  ∃ y, n = y^2 ∧
  -- First two digits are a perfect square
  ∃ a, (n / 100000)^2 = a ∧
  -- Next three digits are a perfect square
  ∃ b, ((n / 1000) % 1000)^2 = b ∧
  -- Last two digits are a perfect square
  ∃ c, (n % 100)^2 = c

-- Theorem statement
theorem no_super_squarish_numbers : ¬∃ n : ℕ, is_super_squarish n := by
  sorry

end NUMINAMATH_CALUDE_no_super_squarish_numbers_l2012_201297


namespace NUMINAMATH_CALUDE_median_sum_bounds_l2012_201291

/-- The sum of the medians of a triangle is less than its perimeter and greater than its semiperimeter -/
theorem median_sum_bounds (a b c ma mb mc : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hma : ma = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2))
  (hmb : mb = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2))
  (hmc : mc = (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)) :
  (a + b + c) / 2 < ma + mb + mc ∧ ma + mb + mc < a + b + c := by
  sorry

end NUMINAMATH_CALUDE_median_sum_bounds_l2012_201291


namespace NUMINAMATH_CALUDE_factorization_of_5x_squared_minus_5_l2012_201216

theorem factorization_of_5x_squared_minus_5 (x : ℝ) : 5 * x^2 - 5 = 5 * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_5x_squared_minus_5_l2012_201216


namespace NUMINAMATH_CALUDE_unique_absolute_value_complex_root_l2012_201292

theorem unique_absolute_value_complex_root : ∃! r : ℝ, 
  (∃ z : ℂ, z^2 - 6*z + 20 = 0 ∧ Complex.abs z = r) ∧ r ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_absolute_value_complex_root_l2012_201292


namespace NUMINAMATH_CALUDE_min_value_a2_b2_l2012_201268

/-- Given that (ax^2 + b/x)^6 has a coefficient of 20 for x^3, 
    the minimum value of a^2 + b^2 is 2 -/
theorem min_value_a2_b2 (a b : ℝ) : 
  (∃ c : ℝ, c = 20 ∧ 
   c = (Nat.choose 6 3 : ℝ) * a^3 * b^3) → 
  ∀ x y : ℝ, x^2 + y^2 ≥ 2 ∧ (x^2 + y^2 = 2 → x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a2_b2_l2012_201268


namespace NUMINAMATH_CALUDE_hybrid_car_journey_length_l2012_201200

theorem hybrid_car_journey_length :
  ∀ (d : ℝ),
  d > 60 →
  (60 : ℝ) / d + (d - 60) / (0.04 * (d - 60)) = 50 →
  d = 120 := by
  sorry

end NUMINAMATH_CALUDE_hybrid_car_journey_length_l2012_201200


namespace NUMINAMATH_CALUDE_same_commission_list_price_is_65_l2012_201244

-- Define the list price
def list_price : ℝ := 65

-- Define Alice's selling price
def alice_selling_price : ℝ := list_price - 15

-- Define Bob's selling price
def bob_selling_price : ℝ := list_price - 25

-- Define Alice's commission rate
def alice_commission_rate : ℝ := 0.12

-- Define Bob's commission rate
def bob_commission_rate : ℝ := 0.15

-- Theorem stating that Alice and Bob get the same commission
theorem same_commission :
  alice_commission_rate * alice_selling_price = bob_commission_rate * bob_selling_price :=
by sorry

-- Main theorem proving that the list price is 65
theorem list_price_is_65 : list_price = 65 :=
by sorry

end NUMINAMATH_CALUDE_same_commission_list_price_is_65_l2012_201244


namespace NUMINAMATH_CALUDE_quadratic_polynomial_discriminant_l2012_201217

/-- Given a quadratic polynomial P(x) = ax² + bx + c where a ≠ 0,
    if P(x) = x - 2 has exactly one root and
    P(x) = 1 - x/2 has exactly one root,
    then the discriminant of P(x) is -1/2 -/
theorem quadratic_polynomial_discriminant
  (a b c : ℝ) (ha : a ≠ 0)
  (h1 : ∃! x, a * x^2 + b * x + c = x - 2)
  (h2 : ∃! x, a * x^2 + b * x + c = 1 - x / 2) :
  b^2 - 4*a*c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_discriminant_l2012_201217


namespace NUMINAMATH_CALUDE_marble_problem_l2012_201249

/-- The number of marbles in the jar after adjustment -/
def final_marbles : ℕ := 195

/-- Proves that the final number of marbles is 195 given the conditions -/
theorem marble_problem (ben : ℕ) (leo : ℕ) (tim : ℕ) 
  (h1 : ben = 56)
  (h2 : leo = ben + 20)
  (h3 : tim = leo - 15)
  (h4 : ∃ k : ℤ, -5 ≤ k ∧ k ≤ 5 ∧ (ben + leo + tim + k) % 5 = 0) :
  final_marbles = ben + leo + tim + 2 :=
sorry

end NUMINAMATH_CALUDE_marble_problem_l2012_201249


namespace NUMINAMATH_CALUDE_acute_angle_relation_l2012_201294

theorem acute_angle_relation (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = (1/2) * Real.sin (α + β)) : α < β := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_relation_l2012_201294


namespace NUMINAMATH_CALUDE_velocity_dividing_trapezoid_area_l2012_201298

/-- 
Given a trapezoidal velocity-time graph with bases V and U, 
this theorem proves that the velocity W that divides the area 
under the curve in the ratio 1:k is given by W = √((V^2 + kU^2) / (k + 1)).
-/
theorem velocity_dividing_trapezoid_area 
  (V U : ℝ) (k : ℝ) (hk : k > 0) :
  let W := Real.sqrt ((V^2 + k * U^2) / (k + 1))
  ∃ (h : ℝ), 
    h * (V - W) = (1 / (k + 1)) * ((1 / 2) * h * (V + U)) ∧
    h * (W - U) = (k / (k + 1)) * ((1 / 2) * h * (V + U)) :=
by sorry

end NUMINAMATH_CALUDE_velocity_dividing_trapezoid_area_l2012_201298


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2012_201206

theorem inequality_system_solution (p : ℝ) :
  (19 * p < 10 ∧ p > 1/2) ↔ (1/2 < p ∧ p < 10/19) := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2012_201206


namespace NUMINAMATH_CALUDE_field_trip_arrangements_l2012_201264

/-- The number of grades --/
def num_grades : ℕ := 6

/-- The number of museums --/
def num_museums : ℕ := 6

/-- The number of grades that choose Museum A --/
def grades_choosing_a : ℕ := 2

/-- The number of ways to choose exactly two grades to visit Museum A --/
def ways_to_choose_a : ℕ := Nat.choose num_grades grades_choosing_a

/-- The number of museums excluding Museum A --/
def remaining_museums : ℕ := num_museums - 1

/-- The number of grades not choosing Museum A --/
def grades_not_choosing_a : ℕ := num_grades - grades_choosing_a

/-- The total number of ways to arrange the field trip --/
def total_arrangements : ℕ := ways_to_choose_a * (remaining_museums ^ grades_not_choosing_a)

theorem field_trip_arrangements :
  total_arrangements = Nat.choose num_grades grades_choosing_a * (remaining_museums ^ grades_not_choosing_a) :=
by sorry

end NUMINAMATH_CALUDE_field_trip_arrangements_l2012_201264


namespace NUMINAMATH_CALUDE_exact_selection_probability_l2012_201242

def num_forks : ℕ := 8
def num_spoons : ℕ := 8
def num_knives : ℕ := 8
def total_pieces : ℕ := num_forks + num_spoons + num_knives
def selected_pieces : ℕ := 6

def probability_exact_selection : ℚ :=
  (Nat.choose num_forks 2 * Nat.choose num_spoons 2 * Nat.choose num_knives 2) /
  Nat.choose total_pieces selected_pieces

theorem exact_selection_probability :
  probability_exact_selection = 2744 / 16825 := by
  sorry

#eval probability_exact_selection

end NUMINAMATH_CALUDE_exact_selection_probability_l2012_201242


namespace NUMINAMATH_CALUDE_charlie_fruit_picking_l2012_201255

theorem charlie_fruit_picking : 
  let golden_delicious : ℚ := 0.17
  let macintosh : ℚ := 0.17
  let cortland : ℚ := 0.33
  golden_delicious + macintosh + cortland = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_charlie_fruit_picking_l2012_201255


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l2012_201235

theorem existence_of_special_sequence :
  ∃ (a : Fin 2013 → ℕ), 
    (∀ i j : Fin 2013, i ≠ j → a i ≠ a j) ∧ 
    (∀ k m : Fin 2013, k < m → (a m + a k) % (a m - a k) = 0) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l2012_201235


namespace NUMINAMATH_CALUDE_arcsin_double_angle_l2012_201225

theorem arcsin_double_angle (x : ℝ) (θ : ℝ) 
  (h1 : x ∈ Set.Icc (-1) 1) 
  (h2 : Real.arcsin x = θ) 
  (h3 : θ ∈ Set.Icc (-Real.pi/2) (-Real.pi/4)) :
  Real.arcsin (2 * x * Real.sqrt (1 - x^2)) = -(Real.pi + 2*θ) := by
  sorry

end NUMINAMATH_CALUDE_arcsin_double_angle_l2012_201225


namespace NUMINAMATH_CALUDE_rectangle_area_in_isosceles_triangle_l2012_201276

/-- The area of a rectangle inscribed in an isosceles triangle -/
theorem rectangle_area_in_isosceles_triangle 
  (b h x : ℝ) 
  (hb : b > 0) 
  (hh : h > 0) 
  (hx : x > 0) 
  (hx_bound : x < h/2) : 
  let rectangle_area := x * (b/2 - b*x/h)
  ∃ (rectangle_base : ℝ), 
    rectangle_base > 0 ∧ 
    rectangle_base = b * (h/2 - x) / h ∧
    rectangle_area = x * rectangle_base :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_in_isosceles_triangle_l2012_201276


namespace NUMINAMATH_CALUDE_max_distance_point_to_circle_l2012_201252

/-- The maximum distance from a point to a circle -/
theorem max_distance_point_to_circle :
  let circle := {(x, y) : ℝ × ℝ | (x - 3)^2 + (y - 4)^2 = 25}
  let point := (2, 3)
  (⨆ p ∈ circle, Real.sqrt ((point.1 - p.1)^2 + (point.2 - p.2)^2)) = Real.sqrt 2 + 5 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_point_to_circle_l2012_201252


namespace NUMINAMATH_CALUDE_boxes_given_to_brother_l2012_201254

/-- Proves the number of boxes Ned gave to his little brother -/
theorem boxes_given_to_brother 
  (total_boxes : ℝ) 
  (pieces_per_box : ℝ) 
  (pieces_left : ℕ) 
  (h1 : total_boxes = 14.0)
  (h2 : pieces_per_box = 6.0)
  (h3 : pieces_left = 42) :
  (total_boxes * pieces_per_box - pieces_left) / pieces_per_box = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_boxes_given_to_brother_l2012_201254


namespace NUMINAMATH_CALUDE_g_increasing_on_neg_l2012_201215

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions on f
variable (h1 : ∀ x y, x < y → f x < f y)  -- f is increasing
variable (h2 : ∀ x, f x < 0)  -- f(x) < 0 for all x

-- Define the function g
def g (x : ℝ) : ℝ := x^2 * f x

-- State the theorem
theorem g_increasing_on_neg : 
  ∀ x y, x < y ∧ y < 0 → g f x < g f y :=
sorry

end NUMINAMATH_CALUDE_g_increasing_on_neg_l2012_201215


namespace NUMINAMATH_CALUDE_fraction_equality_l2012_201214

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 4) 
  (h2 : s / u = 7 / 8) : 
  (2 * p * s - 3 * q * u) / (5 * q * u - 4 * p * s) = -13 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2012_201214


namespace NUMINAMATH_CALUDE_football_field_lap_time_l2012_201281

-- Define the field dimensions
def field_length : ℝ := 100
def field_width : ℝ := 50

-- Define the number of laps and obstacles
def num_laps : ℕ := 6
def num_obstacles : ℕ := 2

-- Define the additional distance per obstacle
def obstacle_distance : ℝ := 20

-- Define the average speed of the player
def average_speed : ℝ := 4

-- Theorem to prove
theorem football_field_lap_time :
  let perimeter : ℝ := 2 * (field_length + field_width)
  let total_obstacle_distance : ℝ := num_obstacles * obstacle_distance
  let lap_distance : ℝ := perimeter + total_obstacle_distance
  let total_distance : ℝ := num_laps * lap_distance
  let time_taken : ℝ := total_distance / average_speed
  time_taken = 510 := by sorry

end NUMINAMATH_CALUDE_football_field_lap_time_l2012_201281


namespace NUMINAMATH_CALUDE_predictor_accuracy_two_thirds_l2012_201213

/-- Represents a match between two teams -/
structure Match where
  team_a_win_prob : ℝ
  team_b_win_prob : ℝ
  (prob_sum_one : team_a_win_prob + team_b_win_prob = 1)

/-- Represents a predictor who chooses winners with the same probability as the team's chance of winning -/
def predictor_correct_prob (m : Match) : ℝ :=
  m.team_a_win_prob * m.team_a_win_prob + m.team_b_win_prob * m.team_b_win_prob

/-- Theorem stating that for a match where one team has 2/3 probability of winning,
    the probability of the predictor correctly choosing the winner is 5/9 -/
theorem predictor_accuracy_two_thirds :
  ∀ m : Match, m.team_a_win_prob = 2/3 → predictor_correct_prob m = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_predictor_accuracy_two_thirds_l2012_201213


namespace NUMINAMATH_CALUDE_area_equality_l2012_201233

-- Define the types for points and quadrilaterals
variable (Point : Type) [AddCommGroup Point] [Module ℝ Point]
variable (Quadrilateral : Type)

-- Define the necessary functions
variable (is_cyclic : Quadrilateral → Prop)
variable (midpoint : Point → Point → Point)
variable (orthocenter : Point → Point → Point → Point)
variable (area : Quadrilateral → ℝ)

-- Define the theorem
theorem area_equality 
  (A B C D E F G H W X Y Z : Point)
  (quad_ABCD quad_WXYZ : Quadrilateral) :
  is_cyclic quad_ABCD →
  E = midpoint A B →
  F = midpoint B C →
  G = midpoint C D →
  H = midpoint D A →
  W = orthocenter A H E →
  X = orthocenter B E F →
  Y = orthocenter C F G →
  Z = orthocenter D G H →
  area quad_ABCD = area quad_WXYZ :=
by sorry

end NUMINAMATH_CALUDE_area_equality_l2012_201233


namespace NUMINAMATH_CALUDE_book_pages_calculation_l2012_201245

theorem book_pages_calculation (first_day_percent : ℝ) (second_day_percent : ℝ) 
  (third_day_pages : ℕ) :
  first_day_percent = 0.1 →
  second_day_percent = 0.25 →
  (first_day_percent + second_day_percent + (third_day_pages : ℝ) / (240 : ℝ)) = 0.5 →
  third_day_pages = 30 →
  (240 : ℕ) = 240 :=
by sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l2012_201245


namespace NUMINAMATH_CALUDE_group_dynamics_index_l2012_201236

theorem group_dynamics_index (n : ℕ) (female_count : ℕ) : 
  n = 25 →
  female_count ≤ n →
  (n - female_count : ℚ) / n - (n - (n - female_count) : ℚ) / n = 9 / 25 →
  female_count = 8 := by
sorry

end NUMINAMATH_CALUDE_group_dynamics_index_l2012_201236


namespace NUMINAMATH_CALUDE_time_to_cook_one_potato_l2012_201241

theorem time_to_cook_one_potato 
  (total_potatoes : ℕ) 
  (cooked_potatoes : ℕ) 
  (time_for_remaining : ℕ) : 
  total_potatoes = 13 → 
  cooked_potatoes = 5 → 
  time_for_remaining = 48 → 
  (time_for_remaining / (total_potatoes - cooked_potatoes) : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_time_to_cook_one_potato_l2012_201241


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l2012_201273

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 10)
  (h2 : average_speed = 35) :
  let speed_second_hour := 2 * average_speed - speed_first_hour
  speed_second_hour = 60 := by
  sorry

#check car_speed_second_hour

end NUMINAMATH_CALUDE_car_speed_second_hour_l2012_201273


namespace NUMINAMATH_CALUDE_line_through_circle_centers_l2012_201223

/-- Given two circles in polar coordinates, C1: ρ = 2cos θ and C2: ρ = 2sin θ,
    the polar equation of the line passing through their centers is θ = π/4 -/
theorem line_through_circle_centers (θ : Real) :
  let c1 : Real → Real := fun θ => 2 * Real.cos θ
  let c2 : Real → Real := fun θ => 2 * Real.sin θ
  ∃ (ρ : Real), (ρ * Real.cos (π/4) = 1 ∧ ρ * Real.sin (π/4) = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_through_circle_centers_l2012_201223


namespace NUMINAMATH_CALUDE_product_125_sum_31_l2012_201246

theorem product_125_sum_31 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → a * b * c = 125 → a + b + c = 31 := by
  sorry

end NUMINAMATH_CALUDE_product_125_sum_31_l2012_201246


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l2012_201270

def geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ := a * r^(n - 1)

theorem fourth_term_of_geometric_sequence :
  ∀ (r : ℕ),
  (geometric_sequence 5 r 1 > 0) →
  (geometric_sequence 5 r 5 = 1280) →
  (geometric_sequence 5 r 4 = 320) := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l2012_201270


namespace NUMINAMATH_CALUDE_find_some_number_l2012_201282

theorem find_some_number (x : ℝ) (some_number : ℝ) 
  (eq1 : x + some_number = 4) (eq2 : x = 3) : some_number = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_some_number_l2012_201282


namespace NUMINAMATH_CALUDE_union_of_sets_l2012_201275

def set_A : Set ℝ := {x | x^2 - x = 0}
def set_B : Set ℝ := {x | x^2 + x = 0}

theorem union_of_sets : set_A ∪ set_B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l2012_201275


namespace NUMINAMATH_CALUDE_students_in_either_not_both_l2012_201261

/-- The number of students taking both geometry and statistics -/
def both_subjects : ℕ := 18

/-- The total number of students taking geometry -/
def geometry_total : ℕ := 35

/-- The number of students taking only statistics -/
def only_statistics : ℕ := 16

/-- Theorem: The number of students taking geometry or statistics but not both is 33 -/
theorem students_in_either_not_both : 
  (geometry_total - both_subjects) + only_statistics = 33 := by
  sorry

end NUMINAMATH_CALUDE_students_in_either_not_both_l2012_201261


namespace NUMINAMATH_CALUDE_set_operations_l2012_201260

def A : Set ℕ := {x | x > 0 ∧ x < 11}
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6, 7}

theorem set_operations :
  (A ∩ C = {3, 4, 5, 6, 7}) ∧
  ((A \ B) = {5, 6, 7, 8, 9, 10}) ∧
  ((A \ (B ∪ C)) = {8, 9, 10}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l2012_201260


namespace NUMINAMATH_CALUDE_fibonacci_identities_l2012_201203

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_identities (n : ℕ) : 
  (fib (2*n + 1) * fib (2*n - 1) = fib (2*n)^2 + 1) ∧ 
  (fib (2*n + 1)^2 + fib (2*n - 1)^2 + 1 = 3 * fib (2*n + 1) * fib (2*n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_identities_l2012_201203


namespace NUMINAMATH_CALUDE_sequence_property_l2012_201248

/-- Represents the number of items in the nth row of the sequence -/
def num_items (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the sum of items in the nth row of the sequence -/
def sum_items (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The row number we're interested in -/
def target_row : ℕ := 1005

/-- The target value we're trying to match -/
def target_value : ℕ := 2009^2

theorem sequence_property :
  num_items target_row = 2009 ∧ sum_items target_row = target_value := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l2012_201248


namespace NUMINAMATH_CALUDE_bookmarks_end_of_march_l2012_201287

/-- Represents the number of pages bookmarked on each day of the week -/
def weekly_bookmarks : Fin 7 → ℕ
| 0 => 25  -- Monday
| 1 => 30  -- Tuesday
| 2 => 35  -- Wednesday
| 3 => 40  -- Thursday
| 4 => 45  -- Friday
| 5 => 50  -- Saturday
| _ => 55  -- Sunday

/-- The current number of bookmarked pages -/
def current_bookmarks : ℕ := 400

/-- The number of days in March -/
def march_days : ℕ := 31

/-- March starts on a Monday (represented by 0) -/
def march_start : Fin 7 := 0

/-- Calculates the total number of bookmarked pages at the end of March -/
def total_bookmarks_end_of_march : ℕ :=
  current_bookmarks +
  (march_days / 7 * (Finset.sum Finset.univ weekly_bookmarks)) +
  (Finset.sum (Finset.range (march_days % 7)) (λ i => weekly_bookmarks ((i + march_start) % 7)))

/-- Theorem stating that the total number of bookmarked pages at the end of March is 1610 -/
theorem bookmarks_end_of_march :
  total_bookmarks_end_of_march = 1610 := by sorry

end NUMINAMATH_CALUDE_bookmarks_end_of_march_l2012_201287


namespace NUMINAMATH_CALUDE_smallest_n_for_perfect_square_product_l2012_201251

/-- The set of integers from 70 to 70 + n, inclusive -/
def numberSet (n : ℕ) : Set ℤ :=
  {x | 70 ≤ x ∧ x ≤ 70 + n}

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (x : ℤ) : Prop :=
  ∃ y : ℤ, x = y * y

/-- Predicate to check if there exist two different numbers in the set whose product is a perfect square -/
def hasPerfectSquareProduct (n : ℕ) : Prop :=
  ∃ a b : ℤ, a ∈ numberSet n ∧ b ∈ numberSet n ∧ a ≠ b ∧ isPerfectSquare (a * b)

theorem smallest_n_for_perfect_square_product : 
  (∀ m : ℕ, m < 28 → ¬hasPerfectSquareProduct m) ∧ hasPerfectSquareProduct 28 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_perfect_square_product_l2012_201251


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l2012_201257

/-- Given a principal amount where the simple interest for 2 years at 5% per annum is $50,
    prove that the compound interest for the same principal, rate, and time is $51.25. -/
theorem compound_interest_calculation (P : ℝ) : 
  P * 0.05 * 2 = 50 → P * (1 + 0.05)^2 - P = 51.25 := by sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l2012_201257


namespace NUMINAMATH_CALUDE_distinct_configurations_eq_seven_l2012_201212

/-- The group of 2D rotations and flips for a 2x3 rectangle -/
inductive SymmetryGroup
| identity
| rotation180
| flipVertical
| flipHorizontal

/-- A configuration of red and yellow cubes in a 2x3 rectangle -/
def Configuration := Fin 6 → Bool

/-- The number of elements in the symmetry group -/
def symmetryGroupSize : ℕ := 4

/-- The total number of configurations -/
def totalConfigurations : ℕ := Nat.choose 6 3

/-- Function to count fixed points for each symmetry operation -/
noncomputable def fixedPoints (g : SymmetryGroup) : ℕ :=
  match g with
  | SymmetryGroup.identity => totalConfigurations
  | _ => 3  -- For rotation180, flipVertical, and flipHorizontal

/-- The sum of fixed points for all symmetry operations -/
noncomputable def totalFixedPoints : ℕ :=
  (fixedPoints SymmetryGroup.identity) +
  (fixedPoints SymmetryGroup.rotation180) +
  (fixedPoints SymmetryGroup.flipVertical) +
  (fixedPoints SymmetryGroup.flipHorizontal)

/-- The number of distinct configurations -/
noncomputable def distinctConfigurations : ℕ :=
  totalFixedPoints / symmetryGroupSize

theorem distinct_configurations_eq_seven :
  distinctConfigurations = 7 := by sorry

end NUMINAMATH_CALUDE_distinct_configurations_eq_seven_l2012_201212


namespace NUMINAMATH_CALUDE_specific_hexahedron_volume_l2012_201221

/-- A regular hexahedron with specific dimensions -/
structure RegularHexahedron where
  -- Base edge length
  ab : ℝ
  -- Top edge length
  a₁b₁ : ℝ
  -- Height
  aa₁ : ℝ
  -- Regularity conditions
  ab_positive : 0 < ab
  a₁b₁_positive : 0 < a₁b₁
  aa₁_positive : 0 < aa₁

/-- The volume of a regular hexahedron -/
def volume (h : RegularHexahedron) : ℝ :=
  -- Definition of volume calculation
  sorry

/-- Theorem stating the volume of the specific hexahedron -/
theorem specific_hexahedron_volume :
  ∃ (h : RegularHexahedron),
    h.ab = 2 ∧
    h.a₁b₁ = 3 ∧
    h.aa₁ = Real.sqrt 10 ∧
    volume h = (57 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_hexahedron_volume_l2012_201221


namespace NUMINAMATH_CALUDE_investment_rate_proof_l2012_201250

-- Define the given values
def total_investment : ℚ := 12000
def first_investment : ℚ := 5000
def first_rate : ℚ := 3 / 100
def second_investment : ℚ := 3000
def second_return : ℚ := 90
def desired_income : ℚ := 480

-- Define the theorem
theorem investment_rate_proof :
  let remaining_investment := total_investment - first_investment - second_investment
  let known_income := first_investment * first_rate + second_return
  let required_income := desired_income - known_income
  let rate := required_income / remaining_investment
  rate = 6 / 100 := by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l2012_201250


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2012_201243

/-- Given an arithmetic sequence {aₙ}, prove that its common difference is 2 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  (h_sum : a 1 + a 5 = 10)  -- Given condition
  (h_S4 : (a 1 + a 2 + a 3 + a 4) = 16)  -- Given condition for S₄
  : a 2 - a 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2012_201243


namespace NUMINAMATH_CALUDE_units_digit_27_pow_23_l2012_201266

def units_digit (n : ℕ) : ℕ := n % 10

def units_digit_power (base : ℕ) (exp : ℕ) : ℕ :=
  units_digit ((units_digit base)^exp)

theorem units_digit_27_pow_23 :
  units_digit (27^23) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_27_pow_23_l2012_201266


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2012_201220

theorem complex_equation_solution :
  ∀ (a b : ℝ), (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2012_201220


namespace NUMINAMATH_CALUDE_root_implies_sum_l2012_201295

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem root_implies_sum (a b : ℤ) : 
  (∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) →
  b - a = 1 →
  a + b = -3 := by sorry

end NUMINAMATH_CALUDE_root_implies_sum_l2012_201295


namespace NUMINAMATH_CALUDE_expense_representation_l2012_201207

-- Define a type for financial transactions
inductive Transaction
| Income : ℕ → Transaction
| Expense : ℕ → Transaction

-- Define a function to represent transactions numerically
def represent : Transaction → ℤ
| Transaction.Income n => n
| Transaction.Expense n => -n

-- State the theorem
theorem expense_representation (amount : ℕ) :
  represent (Transaction.Income amount) = amount →
  represent (Transaction.Expense amount) = -amount :=
by
  sorry

end NUMINAMATH_CALUDE_expense_representation_l2012_201207


namespace NUMINAMATH_CALUDE_inequality_proof_l2012_201211

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum : x + 2*y + 3*z = 11/12) :
  6*(3*x*y + 4*x*z + 2*y*z) + 6*x + 3*y + 4*z + 72*x*y*z ≤ 107/18 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2012_201211


namespace NUMINAMATH_CALUDE_polynomial_decomposition_l2012_201256

theorem polynomial_decomposition (x y : ℝ) :
  x^7 + x^6*y + x^5*y^2 + x^4*y^3 + x^3*y^4 + x^2*y^5 + x*y^6 + y^7 = (x + y)*(x^2 + y^2)*(x^4 + y^4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_decomposition_l2012_201256


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2012_201230

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the circle: x^2 + y^2 - 4x = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x = 0

/-- The circle defined by the equation -/
def given_circle : Circle :=
  { center := (2, 0),
    radius := 2 }

/-- Theorem: The given equation defines a circle with center (2, 0) and radius 2 -/
theorem circle_center_and_radius :
  ∀ x y : ℝ, circle_equation x y ↔ 
    (x - given_circle.center.1)^2 + (y - given_circle.center.2)^2 = given_circle.radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l2012_201230


namespace NUMINAMATH_CALUDE_shop_owner_gain_l2012_201205

/-- Calculates the overall percentage gain for a shop owner based on purchase and sale data -/
def overall_percentage_gain (
  notebook_purchase_qty : ℕ) (notebook_purchase_price : ℚ)
  (notebook_sale_qty : ℕ) (notebook_sale_price : ℚ)
  (pen_purchase_qty : ℕ) (pen_purchase_price : ℚ)
  (pen_sale_qty : ℕ) (pen_sale_price : ℚ)
  (bowl_purchase_qty : ℕ) (bowl_purchase_price : ℚ)
  (bowl_sale_qty : ℕ) (bowl_sale_price : ℚ) : ℚ :=
  let total_cost := notebook_purchase_qty * notebook_purchase_price +
                    pen_purchase_qty * pen_purchase_price +
                    bowl_purchase_qty * bowl_purchase_price
  let total_sale := notebook_sale_qty * notebook_sale_price +
                    pen_sale_qty * pen_sale_price +
                    bowl_sale_qty * bowl_sale_price
  let gain := total_sale - total_cost
  (gain / total_cost) * 100

/-- The overall percentage gain for the shop owner is approximately 16.01% -/
theorem shop_owner_gain :
  let gain := overall_percentage_gain 150 25 140 30 90 15 80 20 114 13 108 17
  ∃ ε > 0, |gain - 16.01| < ε :=
by sorry

end NUMINAMATH_CALUDE_shop_owner_gain_l2012_201205


namespace NUMINAMATH_CALUDE_shirt_price_l2012_201240

/-- Given the sales of shoes and shirts, prove the price of each shirt -/
theorem shirt_price (num_shoes : ℕ) (shoe_price : ℚ) (num_shirts : ℕ) (total_earnings_per_person : ℚ) :
  num_shoes = 6 →
  shoe_price = 3 →
  num_shirts = 18 →
  total_earnings_per_person = 27 →
  ∃ (shirt_price : ℚ), 
    (↑num_shoes * shoe_price + ↑num_shirts * shirt_price) / 2 = total_earnings_per_person ∧
    shirt_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_l2012_201240


namespace NUMINAMATH_CALUDE_circle_properties_l2012_201267

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 8*x + 18*y + 98 = -y^2 - 6*x

-- Define the center and radius
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- State the theorem
theorem circle_properties :
  ∃ a b r : ℝ,
    is_center_radius a b r ∧
    a = -7 ∧
    b = -9 ∧
    r = 4 * Real.sqrt 2 ∧
    a + b + r = -16 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l2012_201267


namespace NUMINAMATH_CALUDE_production_line_b_l2012_201229

def total_production : ℕ := 5000

def sampling_ratio : List ℕ := [1, 2, 2]

theorem production_line_b (a b c : ℕ) : 
  a + b + c = total_production →
  [a, b, c] = sampling_ratio.map (λ x => x * (total_production / sampling_ratio.sum)) →
  b = 2000 := by
  sorry

end NUMINAMATH_CALUDE_production_line_b_l2012_201229


namespace NUMINAMATH_CALUDE_point_Q_in_first_quadrant_l2012_201279

-- Define the conditions for point P
def fourth_quadrant (a b : ℝ) : Prop := a > 0 ∧ b < 0

-- Define the condition |a| > |b|
def magnitude_condition (a b : ℝ) : Prop := abs a > abs b

-- Define what it means for a point to be in the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem point_Q_in_first_quadrant (a b : ℝ) 
  (h1 : fourth_quadrant a b) (h2 : magnitude_condition a b) : 
  first_quadrant (a + b) (a - b) := by
  sorry

end NUMINAMATH_CALUDE_point_Q_in_first_quadrant_l2012_201279


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l2012_201234

theorem overtime_hours_calculation (regular_rate overtime_rate total_pay : ℚ) 
  (h1 : regular_rate = 3)
  (h2 : overtime_rate = 2 * regular_rate)
  (h3 : total_pay = 186) : 
  (total_pay - 40 * regular_rate) / overtime_rate = 11 := by
  sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l2012_201234


namespace NUMINAMATH_CALUDE_rectangle_side_length_l2012_201283

/-- Given two rectangles A and B, with sides (a, b) and (c, d) respectively,
    where the ratio of corresponding sides is 3/4 and rectangle B has sides 4 and 8,
    prove that the side a of rectangle A is 3. -/
theorem rectangle_side_length (a b c d : ℝ) : 
  a / c = 3 / 4 →
  b / d = 3 / 4 →
  c = 4 →
  d = 8 →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l2012_201283


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l2012_201262

/-- Proof that for a rectangle with length to width ratio of 5:2 and perimeter 42 cm, 
    the area A can be expressed as (10/29)d^2, where d is the diagonal of the rectangle. -/
theorem rectangle_area_diagonal (length width : ℝ) (d : ℝ) : 
  length / width = 5 / 2 →
  2 * (length + width) = 42 →
  d^2 = length^2 + width^2 →
  length * width = (10/29) * d^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l2012_201262


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2012_201280

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line
def line (x y a : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - a)

-- Define the condition for F being outside the circle with diameter CD
def F_outside_circle (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - 1) * (x₂ - 1) + y₁ * y₂ > 0

theorem parabola_line_intersection (a : ℝ) :
  a < 0 →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line x₁ y₁ a ∧ line x₂ y₂ a ∧
    F_outside_circle x₁ y₁ x₂ y₂) ↔
  -3 < a ∧ a < -2 * Real.sqrt 5 + 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2012_201280


namespace NUMINAMATH_CALUDE_minibus_seats_l2012_201222

/-- The number of seats in a minibus given specific seating arrangements -/
theorem minibus_seats (total_children : ℕ) (three_child_seats : ℕ) : 
  total_children = 19 →
  three_child_seats = 5 →
  (∃ (two_child_seats : ℕ), 
    total_children = three_child_seats * 3 + two_child_seats * 2 ∧
    three_child_seats + two_child_seats = 7) := by
  sorry

end NUMINAMATH_CALUDE_minibus_seats_l2012_201222


namespace NUMINAMATH_CALUDE_max_yellow_apples_max_total_apples_l2012_201286

/-- Represents the number of apples of each color in the basket -/
structure Basket :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Represents the number of apples taken from the basket -/
structure ApplesTaken :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Checks if the condition for stopping is met -/
def stopCondition (taken : ApplesTaken) : Prop :=
  taken.green < taken.yellow ∧ taken.yellow < taken.red

/-- The initial state of the basket -/
def initialBasket : Basket :=
  { green := 11, yellow := 14, red := 19 }

/-- Theorem stating the maximum number of yellow apples that can be taken -/
theorem max_yellow_apples (taken : ApplesTaken) 
  (h : taken.yellow ≤ initialBasket.yellow) 
  (h_stop : ¬stopCondition taken) : 
  taken.yellow ≤ 14 :=
sorry

/-- Theorem stating the maximum total number of apples that can be taken -/
theorem max_total_apples (taken : ApplesTaken) 
  (h_green : taken.green ≤ initialBasket.green)
  (h_yellow : taken.yellow ≤ initialBasket.yellow)
  (h_red : taken.red ≤ initialBasket.red)
  (h_stop : ¬stopCondition taken) :
  taken.green + taken.yellow + taken.red ≤ 42 :=
sorry

end NUMINAMATH_CALUDE_max_yellow_apples_max_total_apples_l2012_201286


namespace NUMINAMATH_CALUDE_john_house_wall_planks_l2012_201224

/-- The number of planks John uses for the house wall -/
def num_planks : ℕ := 32 / 2

/-- Each plank needs 2 nails -/
def nails_per_plank : ℕ := 2

/-- The total number of nails needed for the wall -/
def total_nails : ℕ := 32

theorem john_house_wall_planks : num_planks = 16 := by
  sorry

end NUMINAMATH_CALUDE_john_house_wall_planks_l2012_201224


namespace NUMINAMATH_CALUDE_bus_speed_and_interval_l2012_201209

/-- The speed of buses and interval between departures in a traffic scenario --/
theorem bus_speed_and_interval (a b c : ℝ) (hc : c > b) (hb : b > 0) (ha : a > 0) :
  ∃ (x t : ℝ),
    (a + x) * b = t * x ∧
    (x - a) * c = t * x ∧
    x = a * (c + b) / (c - b) ∧
    t = 2 * b * c / (b + c) := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_and_interval_l2012_201209


namespace NUMINAMATH_CALUDE_find_divisor_l2012_201284

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 16698) (h2 : quotient = 89) (h3 : remainder = 14) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 187 :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l2012_201284


namespace NUMINAMATH_CALUDE_even_polynomial_iff_product_with_negation_l2012_201227

/-- A polynomial over the complex numbers -/
def ComplexPolynomial := ℂ → ℂ

/-- Definition of an even polynomial -/
def IsEvenPolynomial (P : ComplexPolynomial) : Prop :=
  ∀ z : ℂ, P z = P (-z)

/-- The main theorem -/
theorem even_polynomial_iff_product_with_negation (P : ComplexPolynomial) :
  IsEvenPolynomial P ↔ ∃ Q : ComplexPolynomial, ∀ z : ℂ, P z = Q z * Q (-z) := by
  sorry

end NUMINAMATH_CALUDE_even_polynomial_iff_product_with_negation_l2012_201227


namespace NUMINAMATH_CALUDE_dmitry_black_socks_l2012_201247

/-- Proves that Dmitry bought 22 pairs of black socks -/
theorem dmitry_black_socks :
  let initial_blue : ℕ := 10
  let initial_black : ℕ := 22
  let initial_white : ℕ := 12
  let bought_black : ℕ := x
  let total_initial : ℕ := initial_blue + initial_black + initial_white
  let total_after : ℕ := total_initial + bought_black
  let black_after : ℕ := initial_black + bought_black
  (black_after : ℚ) / (total_after : ℚ) = 2 / 3 →
  x = 22 := by
sorry

end NUMINAMATH_CALUDE_dmitry_black_socks_l2012_201247


namespace NUMINAMATH_CALUDE_boat_speed_l2012_201271

/-- The speed of a boat in still water, given its speeds with and against a stream -/
theorem boat_speed (along_stream : ℝ) (against_stream : ℝ) 
  (h1 : along_stream = 11) 
  (h2 : against_stream = 3) : ℝ :=
by
  -- The speed of the boat in still water is 7 km/hr
  sorry

#check boat_speed

end NUMINAMATH_CALUDE_boat_speed_l2012_201271


namespace NUMINAMATH_CALUDE_unique_positive_number_l2012_201238

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l2012_201238


namespace NUMINAMATH_CALUDE_range_of_a_l2012_201232

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + 1 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → a^2 - 3*a - x + 1 ≤ 0

-- Define the theorem
theorem range_of_a :
  ∃ a : ℝ, (¬(p a ∧ q a) ∧ ¬(¬(q a))) ∧ a ∈ Set.Icc 1 2 ∧ a ≠ 2 ∧
  (∀ b : ℝ, (¬(p b ∧ q b) ∧ ¬(¬(q b))) → b ∈ Set.Icc 1 2 ∧ b < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2012_201232


namespace NUMINAMATH_CALUDE_chosen_number_l2012_201231

theorem chosen_number (x : ℝ) : (x / 4) - 175 = 10 → x = 740 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_l2012_201231


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_one_third_l2012_201239

theorem fraction_zero_implies_x_one_third (x : ℝ) :
  (3*x - 1) / (x^2 + 1) = 0 → x = 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_one_third_l2012_201239


namespace NUMINAMATH_CALUDE_first_step_error_l2012_201293

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  (x - 1) / 2 + 1 = (2 * x + 1) / 3

-- Define the incorrect step 1 result
def incorrect_step1 (x : ℝ) : Prop :=
  3 * (x - 1) + 2 = 2 * x + 1

-- Define the correct step 1 result
def correct_step1 (x : ℝ) : Prop :=
  3 * (x - 1) + 6 = 2 * x + 1

-- Theorem stating that the first step is erroneous
theorem first_step_error :
  ∃ x : ℝ, original_equation x ∧ ¬(incorrect_step1 x ↔ correct_step1 x) :=
sorry

end NUMINAMATH_CALUDE_first_step_error_l2012_201293


namespace NUMINAMATH_CALUDE_parabola_properties_l2012_201289

/-- Parabola represented by y = x^2 + bx - 2 -/
structure Parabola where
  b : ℝ

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about a specific parabola and its properties -/
theorem parabola_properties (p : Parabola) 
  (h1 : p.b = 4) -- Derived from the condition that the parabola passes through (1, 3)
  (A : Point) 
  (hA : A.x = 0 ∧ A.y = -2) -- A is the y-axis intersection point
  (B : Point) 
  (hB : B.x = -2 ∧ B.y = -6) -- B is the vertex of the parabola
  (k : ℝ) 
  (hk : k^2 + p.b * k - 2 = 0) -- k is the x-coordinate of x-axis intersection
  : 
  (1/2 * |A.y| * |B.x| = 2) ∧ 
  ((4*k^4 + 3*k^2 + 12*k - 6) / (k^8 + 2*k^6 + k^5 - 2*k^3 + 8*k^2 + 16) = 1/107) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2012_201289


namespace NUMINAMATH_CALUDE_fraction_decomposition_l2012_201201

theorem fraction_decomposition (x : ℚ) (A B : ℚ) 
  (h : x ≠ -5 ∧ x ≠ 2/3) : 
  (7 * x - 13) / (3 * x^2 + 13 * x - 10) = A / (x + 5) + B / (3 * x - 2) → 
  A = 48/17 ∧ B = -25/17 :=
by sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l2012_201201


namespace NUMINAMATH_CALUDE_dave_initial_apps_l2012_201237

/-- Represents the number of apps on Dave's phone at different stages -/
structure AppCount where
  initial : ℕ
  afterAdding : ℕ
  afterDeleting : ℕ
  final : ℕ

/-- Represents the number of apps added and deleted -/
structure AppChanges where
  added : ℕ
  deleted : ℕ

/-- The theorem stating Dave's initial app count based on the given conditions -/
theorem dave_initial_apps (ac : AppCount) (ch : AppChanges) : 
  ch.added = 89 ∧ 
  ac.afterDeleting = 24 ∧ 
  ch.added = ch.deleted + 3 ∧
  ac.afterAdding = ac.initial + ch.added ∧
  ac.afterDeleting = ac.afterAdding - ch.deleted ∧
  ac.final = ac.afterDeleting + (ch.added - ch.deleted) →
  ac.initial = 21 := by
  sorry


end NUMINAMATH_CALUDE_dave_initial_apps_l2012_201237


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2012_201226

/-- Given vectors a and b, if a + 2b is parallel to ma + b, then m = 1/2 -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
  (ha : a = (2, 3)) 
  (hb : b = (1, 2)) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a + 2 • b = k • (m • a + b)) : 
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2012_201226


namespace NUMINAMATH_CALUDE_derivative_odd_implies_a_eq_neg_one_l2012_201258

/-- Given a real number a and a function f(x) = e^x - ae^(-x), 
    if the derivative of f is an odd function, then a = -1. -/
theorem derivative_odd_implies_a_eq_neg_one (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.exp x - a * Real.exp (-x)
  let f' : ℝ → ℝ := λ x ↦ Real.exp x + a * Real.exp (-x)
  (∀ x, f' x = -f' (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_odd_implies_a_eq_neg_one_l2012_201258


namespace NUMINAMATH_CALUDE_quadratic_intersection_point_l2012_201218

theorem quadratic_intersection_point 
  (a b c d : ℝ) 
  (h1 : d ≠ 0) 
  (h2 : b ≠ 0) : 
  let f1 := fun x : ℝ => a * x^2 + b * x + c
  let f2 := fun x : ℝ => a * x^2 - b * x + c + d
  ∃! p : ℝ × ℝ, 
    f1 p.1 = f2 p.1 ∧ 
    p = (d / (2 * b), a * (d^2 / (4 * b^2)) + d / 2 + c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_point_l2012_201218


namespace NUMINAMATH_CALUDE_minimum_draws_for_divisible_by_3_or_5_l2012_201202

theorem minimum_draws_for_divisible_by_3_or_5 (n : ℕ) (hn : n = 90) :
  let divisible_by_3_or_5 (k : ℕ) := k % 3 = 0 ∨ k % 5 = 0
  let count_divisible := (Finset.range n).filter divisible_by_3_or_5 |>.card
  49 = n - count_divisible + 1 :=
by sorry

end NUMINAMATH_CALUDE_minimum_draws_for_divisible_by_3_or_5_l2012_201202


namespace NUMINAMATH_CALUDE_basketball_shot_probability_l2012_201285

theorem basketball_shot_probability (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < 1) (hbb : b < 1) (hcb : c < 1) (sum_prob : a + b + c = 1) (expected_value : 3*a + 2*b = 2) :
  (2/a + 1/(3*b)) ≥ 16/3 := by
sorry

end NUMINAMATH_CALUDE_basketball_shot_probability_l2012_201285


namespace NUMINAMATH_CALUDE_first_group_size_is_eight_l2012_201253

/-- The number of men in the first group that can complete a work in 18 days, working 8 hours a day -/
def first_group_size : ℕ := sorry

/-- The number of hours worked per day by both groups -/
def hours_per_day : ℕ := 8

/-- The number of days the first group takes to complete the work -/
def days_first_group : ℕ := 18

/-- The number of men in the second group -/
def second_group_size : ℕ := 12

/-- The number of days the second group takes to complete the work -/
def days_second_group : ℕ := 12

/-- The total amount of work done is constant and equal for both groups -/
axiom work_done_equal : first_group_size * hours_per_day * days_first_group = second_group_size * hours_per_day * days_second_group

theorem first_group_size_is_eight : first_group_size = 8 := by sorry

end NUMINAMATH_CALUDE_first_group_size_is_eight_l2012_201253


namespace NUMINAMATH_CALUDE_complex_number_location_l2012_201263

theorem complex_number_location :
  ∀ (z : ℂ), (z * Complex.I = 1 - 2 * Complex.I) →
  (z = -2 - Complex.I ∧ z.re < 0 ∧ z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2012_201263


namespace NUMINAMATH_CALUDE_h_shape_perimeter_is_44_l2012_201269

/-- The perimeter of a rectangle with length l and width w -/
def rectanglePerimeter (l w : ℝ) : ℝ := 2 * l + 2 * w

/-- The perimeter of an H shape formed by two vertical rectangles and one horizontal rectangle -/
def hShapePerimeter (v_length v_width h_length h_width : ℝ) : ℝ :=
  2 * (rectanglePerimeter v_length v_width) + 
  (rectanglePerimeter h_length h_width) - 
  2 * (2 * h_width)

theorem h_shape_perimeter_is_44 : 
  hShapePerimeter 6 3 6 2 = 44 := by sorry

end NUMINAMATH_CALUDE_h_shape_perimeter_is_44_l2012_201269


namespace NUMINAMATH_CALUDE_gcd_cube_plus_five_cube_l2012_201296

theorem gcd_cube_plus_five_cube (n : ℕ) (h : n > 2^5) : Nat.gcd (n^3 + 5^3) (n + 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_five_cube_l2012_201296
