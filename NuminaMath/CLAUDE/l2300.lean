import Mathlib

namespace NUMINAMATH_CALUDE_three_numbers_equation_l2300_230040

theorem three_numbers_equation (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : x^2 - y^2 = y*z) (eq2 : y^2 - z^2 = x*z) :
  x^2 - z^2 = x*y :=
by
  sorry

end NUMINAMATH_CALUDE_three_numbers_equation_l2300_230040


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2300_230067

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 11 = 24)
  (h_a4 : a 4 = 3) :
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2300_230067


namespace NUMINAMATH_CALUDE_min_value_expression_l2300_230093

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + 2 * q) + (5 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) ≥ 4 ∧
  ((5 * r) / (3 * p + 2 * q) + (5 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) = 4 ↔ 3 * p = 2 * q ∧ 2 * q = 3 * r) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2300_230093


namespace NUMINAMATH_CALUDE_student_ticket_price_l2300_230045

/-- Calculates the price of a student ticket given the total number of tickets sold,
    the total amount collected, the price of an adult ticket, and the number of student tickets sold. -/
theorem student_ticket_price
  (total_tickets : ℕ)
  (total_amount : ℚ)
  (adult_price : ℚ)
  (student_tickets : ℕ)
  (h1 : total_tickets = 59)
  (h2 : total_amount = 222.5)
  (h3 : adult_price = 4)
  (h4 : student_tickets = 9) :
  (total_amount - (adult_price * (total_tickets - student_tickets))) / student_tickets = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_student_ticket_price_l2300_230045


namespace NUMINAMATH_CALUDE_line_points_determine_m_l2300_230086

-- Define the points on the line
def point1 : ℝ × ℝ := (7, 10)
def point2 : ℝ → ℝ × ℝ := λ m ↦ (-3, m)
def point3 : ℝ × ℝ := (-11, 5)

-- Define the condition that the points are collinear
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

-- Theorem statement
theorem line_points_determine_m :
  collinear point1 (point2 m) point3 → m = 65 / 9 := by
  sorry

end NUMINAMATH_CALUDE_line_points_determine_m_l2300_230086


namespace NUMINAMATH_CALUDE_sphere_volume_rectangular_solid_l2300_230009

theorem sphere_volume_rectangular_solid (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b = 1 →
  b * c = 2 →
  a * c = 2 →
  (4 / 3) * Real.pi * ((a^2 + b^2 + c^2).sqrt / 2)^3 = Real.pi * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_rectangular_solid_l2300_230009


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l2300_230085

theorem absolute_value_inequality_solution :
  {x : ℝ | |2 - x| ≤ 1} = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l2300_230085


namespace NUMINAMATH_CALUDE_m_range_l2300_230053

-- Define the sets A and B
def A : Set ℝ := {x | x^2 < 16}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- State the theorem
theorem m_range (m : ℝ) : A ∩ B m = A → m ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_m_range_l2300_230053


namespace NUMINAMATH_CALUDE_sum_of_squares_l2300_230059

theorem sum_of_squares (a b : ℕ+) (h : a.val^2 + 2*a.val*b.val - 3*b.val^2 - 41 = 0) : 
  a.val^2 + b.val^2 = 221 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2300_230059


namespace NUMINAMATH_CALUDE_system_solution_exists_l2300_230012

theorem system_solution_exists (x y z : ℝ) : 
  (x * y = 8 - 3 * x - 2 * y) →
  (y * z = 8 - 2 * y - 3 * z) →
  (x * z = 35 - 5 * x - 3 * z) →
  ∃ (x : ℝ), x = 8 := by
sorry

end NUMINAMATH_CALUDE_system_solution_exists_l2300_230012


namespace NUMINAMATH_CALUDE_impossible_to_break_record_duke_game_impossible_l2300_230032

/-- Represents the constraints and conditions for Duke's basketball game --/
structure GameConstraints where
  old_record : ℕ
  points_to_tie : ℕ
  points_to_break : ℕ
  free_throws : ℕ
  regular_baskets : ℕ
  normal_three_pointers : ℕ
  max_attempts : ℕ

/-- Calculates the total points scored based on the number of each type of shot --/
def total_points (ft reg tp : ℕ) : ℕ :=
  ft + 2 * reg + 3 * tp

/-- Theorem stating that it's impossible to break the record under the given constraints --/
theorem impossible_to_break_record (gc : GameConstraints) : 
  ¬∃ (tp : ℕ), 
    total_points gc.free_throws gc.regular_baskets tp = gc.old_record + gc.points_to_tie + gc.points_to_break ∧
    gc.free_throws + gc.regular_baskets + tp ≤ gc.max_attempts :=
by
  sorry

/-- The specific game constraints for Duke's final game --/
def duke_game : GameConstraints :=
  { old_record := 257
  , points_to_tie := 17
  , points_to_break := 5
  , free_throws := 5
  , regular_baskets := 4
  , normal_three_pointers := 2
  , max_attempts := 10
  }

/-- Theorem applying the impossibility proof to Duke's specific game --/
theorem duke_game_impossible : 
  ¬∃ (tp : ℕ), 
    total_points duke_game.free_throws duke_game.regular_baskets tp = 
      duke_game.old_record + duke_game.points_to_tie + duke_game.points_to_break ∧
    duke_game.free_throws + duke_game.regular_baskets + tp ≤ duke_game.max_attempts :=
by
  apply impossible_to_break_record duke_game

end NUMINAMATH_CALUDE_impossible_to_break_record_duke_game_impossible_l2300_230032


namespace NUMINAMATH_CALUDE_intersection_point_expression_value_l2300_230033

/-- Given a point P(a,b) at the intersection of y=x-2 and y=1/x,
    prove that (a-a²/(a+b)) ÷ (a²b²/(a²-b²)) equals 2 -/
theorem intersection_point_expression_value (a b : ℝ) 
  (h1 : b = a - 2)
  (h2 : b = 1 / a)
  (h3 : a ≠ 0)
  (h4 : b ≠ 0)
  (h5 : a ≠ b)
  (h6 : a + b ≠ 0) :
  (a - a^2 / (a + b)) / (a^2 * b^2 / (a^2 - b^2)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_expression_value_l2300_230033


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l2300_230046

theorem rectangle_side_ratio (a b c d : ℝ) (h : a / c = b / d ∧ a / c = 4 / 5) :
  a / c = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l2300_230046


namespace NUMINAMATH_CALUDE_inequality_solution_set_empty_implies_k_range_l2300_230020

theorem inequality_solution_set_empty_implies_k_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 2 * |x - 1| + 6 * k ≥ 0) → 
  k ≥ (1 + Real.sqrt 7) / 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_empty_implies_k_range_l2300_230020


namespace NUMINAMATH_CALUDE_trapezoid_area_equality_l2300_230073

/-- Given a square ABCD with side length a, and a trapezoid EBCF inside it with BE = CF = x,
    if the area of EBCF equals the area of ABCD minus twice the area of a rectangle JKHG 
    inside the square, then x = a/2 -/
theorem trapezoid_area_equality (a : ℝ) (x : ℝ) :
  (∃ (y z : ℝ), y + z = a ∧ x * a = a^2 - 2 * y * z) →
  x = a / 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_equality_l2300_230073


namespace NUMINAMATH_CALUDE_brothers_age_equation_l2300_230090

theorem brothers_age_equation (x : ℝ) (h1 : x > 0) : 
  (x - 6) + (2*x - 6) = 15 :=
by
  sorry

#check brothers_age_equation

end NUMINAMATH_CALUDE_brothers_age_equation_l2300_230090


namespace NUMINAMATH_CALUDE_combination_lock_code_l2300_230005

theorem combination_lock_code (x y : ℕ) : 
  x ≤ 9 ∧ y ≤ 9 ∧ x ≠ 0 → 
  (x + y + x * y = 10 * x + y) ↔ 
  (y = 9 ∧ x ∈ Finset.range 10 \ {0}) :=
sorry

end NUMINAMATH_CALUDE_combination_lock_code_l2300_230005


namespace NUMINAMATH_CALUDE_simplify_expression_l2300_230089

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^10 = 1342177280 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2300_230089


namespace NUMINAMATH_CALUDE_distribute_four_balls_l2300_230028

/-- The number of ways to distribute n distinguishable balls into 2 indistinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ :=
  (n + 1)

/-- The number of ways to distribute 4 distinguishable balls into 2 indistinguishable boxes is 8 -/
theorem distribute_four_balls : distribute_balls 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_balls_l2300_230028


namespace NUMINAMATH_CALUDE_sum_of_squares_l2300_230054

theorem sum_of_squares (a b c x y z : ℝ) 
  (h1 : x/a + y/b + z/c = 4)
  (h2 : a/x + b/y + c/z = 2) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2300_230054


namespace NUMINAMATH_CALUDE_select_teachers_eq_140_l2300_230052

/-- The number of ways to select 6 out of 10 teachers, where two specific teachers cannot be selected together -/
def select_teachers : ℕ :=
  let total_teachers : ℕ := 10
  let teachers_to_invite : ℕ := 6
  let remaining_teachers : ℕ := 8  -- Excluding A and B
  let case1 : ℕ := 2 * Nat.choose remaining_teachers (teachers_to_invite - 1)
  let case2 : ℕ := Nat.choose remaining_teachers teachers_to_invite
  case1 + case2

theorem select_teachers_eq_140 : select_teachers = 140 := by
  sorry

end NUMINAMATH_CALUDE_select_teachers_eq_140_l2300_230052


namespace NUMINAMATH_CALUDE_christinas_total_distance_l2300_230061

/-- The total distance Christina walks in a week given her routine -/
def christinas_weekly_distance (school_distance : ℕ) (friend_distance : ℕ) : ℕ :=
  (school_distance * 2 * 4) + (school_distance * 2 + friend_distance * 2)

/-- Theorem stating that Christina's total distance for the week is 74km -/
theorem christinas_total_distance :
  christinas_weekly_distance 7 2 = 74 := by
  sorry

end NUMINAMATH_CALUDE_christinas_total_distance_l2300_230061


namespace NUMINAMATH_CALUDE_complex_number_location_l2300_230038

theorem complex_number_location (Z : ℂ) : Z = Complex.I :=
  by
  -- Define Z
  have h1 : Z = (Real.sqrt 2 - Complex.I ^ 3) / (1 - Real.sqrt 2 * Complex.I) := by sorry
  
  -- Define properties of complex numbers
  have h2 : Complex.I ^ 2 = -1 := by sorry
  have h3 : Complex.I ^ 3 = -Complex.I := by sorry
  
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2300_230038


namespace NUMINAMATH_CALUDE_base_10_to_base_8_conversion_l2300_230037

theorem base_10_to_base_8_conversion : 
  (2 * 8^3 + 0 * 8^2 + 0 * 8^1 + 0 * 8^0 : ℕ) = 1024 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_8_conversion_l2300_230037


namespace NUMINAMATH_CALUDE_baby_sea_turtles_on_sand_l2300_230069

theorem baby_sea_turtles_on_sand (total : ℕ) (swept_fraction : ℚ) (remaining : ℕ) : 
  total = 42 → 
  swept_fraction = 1/3 → 
  remaining = total - (total * swept_fraction).floor → 
  remaining = 28 := by
sorry

end NUMINAMATH_CALUDE_baby_sea_turtles_on_sand_l2300_230069


namespace NUMINAMATH_CALUDE_segment_division_l2300_230014

theorem segment_division (AB : ℝ) (n : ℕ) (h : n > 1) :
  ∃ E : ℝ, (E = AB / (n^2 + 1) ∨ E = AB / (n^2 - 1)) ∧ 0 ≤ E ∧ E ≤ AB :=
sorry

end NUMINAMATH_CALUDE_segment_division_l2300_230014


namespace NUMINAMATH_CALUDE_adjacent_probability_l2300_230058

/-- The number of students in the arrangement -/
def total_students : ℕ := 9

/-- The number of rows in the seating arrangement -/
def rows : ℕ := 3

/-- The number of columns in the seating arrangement -/
def columns : ℕ := 3

/-- The number of ways to arrange n students -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of adjacent pairs in a row or column -/
def adjacent_pairs_per_line : ℕ := 2

/-- The number of ways to arrange two specific students in an adjacent pair -/
def ways_to_arrange_pair : ℕ := 2

/-- The probability of two specific students being adjacent in a 3x3 grid -/
theorem adjacent_probability :
  (((rows * adjacent_pairs_per_line + columns * adjacent_pairs_per_line) * ways_to_arrange_pair * 
    (arrangements (total_students - 2))) : ℚ) / 
  (arrangements total_students) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_probability_l2300_230058


namespace NUMINAMATH_CALUDE_cat_weight_l2300_230013

theorem cat_weight (num_puppies num_cats : ℕ) (puppy_weight : ℝ) (weight_difference : ℝ) :
  num_puppies = 4 →
  num_cats = 14 →
  puppy_weight = 7.5 →
  weight_difference = 5 →
  puppy_weight + weight_difference = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_cat_weight_l2300_230013


namespace NUMINAMATH_CALUDE_expression_value_l2300_230065

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (2 * x - 3 * y) - f (x + y) = -2 * x + 8 * y

/-- The main theorem stating that the given expression is always equal to 4 -/
theorem expression_value (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ t, f 4*t ≠ f 3*t → (f 5*t - f t) / (f 4*t - f 3*t) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2300_230065


namespace NUMINAMATH_CALUDE_hat_guessing_strategy_exists_l2300_230097

/-- Represents a strategy for guessing hat numbers -/
def Strategy := (ι : Fin 2023 → ℕ) → Fin 2023 → ℕ

/-- Theorem stating that there exists a winning strategy for the hat guessing game -/
theorem hat_guessing_strategy_exists :
  ∃ (s : Strategy),
    ∀ (ι : Fin 2023 → ℕ),
      (∀ i, 1 ≤ ι i ∧ ι i ≤ 2023) →
      ∃ i, s ι i = ι i :=
sorry

end NUMINAMATH_CALUDE_hat_guessing_strategy_exists_l2300_230097


namespace NUMINAMATH_CALUDE_flight_time_sum_l2300_230035

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

def Time.toMinutes (t : Time) : ℕ := t.hours * 60 + t.minutes

theorem flight_time_sum (departure : Time) (arrival : Time) (layover : ℕ) 
  (h m : ℕ) (hm : 0 < m ∧ m < 60) :
  departure.hours = 15 ∧ departure.minutes = 45 →
  arrival.hours = 20 ∧ arrival.minutes = 2 →
  layover = 25 →
  arrival.toMinutes - departure.toMinutes - layover = h * 60 + m →
  h + m = 55 := by
sorry

end NUMINAMATH_CALUDE_flight_time_sum_l2300_230035


namespace NUMINAMATH_CALUDE_train_speed_problem_train_speed_solution_l2300_230015

theorem train_speed_problem (train1_length train2_length : ℝ) 
  (train1_speed time_to_clear : ℝ) : ℝ :=
  let total_length := train1_length + train2_length
  let total_length_km := total_length / 1000
  let time_to_clear_hours := time_to_clear / 3600
  let relative_speed := total_length_km / time_to_clear_hours
  relative_speed - train1_speed

theorem train_speed_solution :
  train_speed_problem 140 280 42 20.99832013438925 = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_train_speed_solution_l2300_230015


namespace NUMINAMATH_CALUDE_not_all_regular_pentagons_congruent_l2300_230081

-- Define a regular pentagon
structure RegularPentagon where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

-- Define congruence for regular pentagons
def congruent (p1 p2 : RegularPentagon) : Prop :=
  p1.sideLength = p2.sideLength

-- Theorem statement
theorem not_all_regular_pentagons_congruent :
  ∃ (p1 p2 : RegularPentagon), ¬(congruent p1 p2) := by
  sorry

end NUMINAMATH_CALUDE_not_all_regular_pentagons_congruent_l2300_230081


namespace NUMINAMATH_CALUDE_min_MN_length_l2300_230026

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the circle containing point P -/
def circle_P (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

/-- Theorem statement -/
theorem min_MN_length (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ellipse_C 0 1 a b) -- vertex at (0,1)
  (h4 : (a^2 - b^2) / a^2 = 3/4) -- eccentricity is √3/2
  : ∃ (x_p y_p x_m y_n : ℝ),
    circle_P x_p y_p ∧
    (∀ (x_a y_a x_b y_b : ℝ),
      ellipse_C x_a y_a a b →
      ellipse_C x_b y_b a b →
      (y_n - y_a) * (x_p - x_a) = (y_p - y_a) * (x_m - x_a) →
      (y_n - y_b) * (x_p - x_b) = (y_p - y_b) * (x_m - x_b) →
      x_m = 0 ∧ y_n = 0) →
    (x_m - 0)^2 + (0 - y_n)^2 ≥ (5/4)^2 :=
by sorry

end NUMINAMATH_CALUDE_min_MN_length_l2300_230026


namespace NUMINAMATH_CALUDE_probability_same_length_segments_l2300_230047

def regular_pentagon_segments : Finset ℕ := sorry

theorem probability_same_length_segments :
  let S := regular_pentagon_segments
  let total_segments := S.card
  let same_type_segments := (total_segments / 2) - 1
  (same_type_segments : ℚ) / ((total_segments - 1) : ℚ) = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_same_length_segments_l2300_230047


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2300_230050

/-- The eccentricity of a hyperbola with the given conditions is between 1 and 2 -/
theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ (x y e : ℝ),
    x^2 / a^2 - y^2 / b^2 = 1 ∧
    x ≥ a ∧
    ∃ (f1 f2 : ℝ × ℝ),
      (∀ (p : ℝ × ℝ), p.1^2 / a^2 - p.2^2 / b^2 = 1 →
        |p.1 - f1.1| - |p.1 - f2.1| = 2 * a * e) ∧
      |x - f1.1| = 3 * |x - f2.1| →
    1 < e ∧ e ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2300_230050


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2300_230072

theorem solution_set_inequality (x : ℝ) : (x - 2)^2 ≤ 2*x + 11 ↔ x ∈ Set.Icc (-1) 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2300_230072


namespace NUMINAMATH_CALUDE_average_weight_decrease_l2300_230062

theorem average_weight_decrease (n : ℕ) (old_weight new_weight : ℝ) :
  n = 6 →
  old_weight = 80 →
  new_weight = 62 →
  (old_weight - new_weight) / n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l2300_230062


namespace NUMINAMATH_CALUDE_two_integer_solutions_l2300_230017

/-- The function f(x) = x^2 + bx + 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

/-- The condition on b -/
def valid_b (b : ℝ) : Prop :=
  abs b > 2 ∧ ∀ a : ℤ, a ≠ 0 ∧ a ≠ 1 ∧ a ≠ -1 → b ≠ a + 1/a

/-- The main theorem -/
theorem two_integer_solutions (b : ℝ) (hb : valid_b b) :
  ∃! n : ℕ, n = 2 ∧ ∃ s : Finset ℤ, s.card = n ∧
    ∀ x : ℤ, x ∈ s ↔ f b (f b x + x) < 0 :=
sorry

end NUMINAMATH_CALUDE_two_integer_solutions_l2300_230017


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2300_230088

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, -1/2 < x ∧ x < 1/3 ↔ a * x^2 + b * x + 20 > 0) →
  a = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2300_230088


namespace NUMINAMATH_CALUDE_vector_problem_l2300_230031

/-- Given two collinear vectors a and b in ℝ², with b = (1, -2) and a ⋅ b = -10,
    prove that a = (-2, 4) and |a + c| = 5 where c = (6, -7) -/
theorem vector_problem (a b c : ℝ × ℝ) : 
  (∃ (k : ℝ), a = k • b) →  -- a and b are collinear
  b = (1, -2) → 
  a.1 * b.1 + a.2 * b.2 = -10 →  -- dot product
  c = (6, -7) → 
  a = (-2, 4) ∧ 
  Real.sqrt ((a.1 + c.1)^2 + (a.2 + c.2)^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_vector_problem_l2300_230031


namespace NUMINAMATH_CALUDE_gym_down_payment_down_payment_is_50_l2300_230043

/-- Calculates the down payment for a gym membership -/
theorem gym_down_payment (monthly_fee : ℕ) (total_payment : ℕ) : ℕ :=
  let months : ℕ := 3 * 12
  let total_monthly_payments : ℕ := months * monthly_fee
  total_payment - total_monthly_payments

/-- Proves that the down payment for the gym membership is $50 -/
theorem down_payment_is_50 :
  gym_down_payment 12 482 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gym_down_payment_down_payment_is_50_l2300_230043


namespace NUMINAMATH_CALUDE_ellipse_equation_l2300_230007

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (2 * a * b = 4) →
  (a^2 - b^2 = 3) →
  (a = 2 ∧ b = 1) := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2300_230007


namespace NUMINAMATH_CALUDE_g_1993_of_4_l2300_230030

-- Define the function g
def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

-- Define the recursive function gn
def gn : ℕ → ℚ → ℚ
  | 0, x => x
  | n + 1, x => g (gn n x)

-- Theorem statement
theorem g_1993_of_4 : gn 1993 4 = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_g_1993_of_4_l2300_230030


namespace NUMINAMATH_CALUDE_yellow_shirt_pairs_l2300_230003

theorem yellow_shirt_pairs (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_students = 60 →
  yellow_students = 84 →
  total_students = 144 →
  total_pairs = 72 →
  blue_blue_pairs = 25 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 37 :=
by sorry

end NUMINAMATH_CALUDE_yellow_shirt_pairs_l2300_230003


namespace NUMINAMATH_CALUDE_percentage_not_sold_l2300_230022

def initial_stock : ℕ := 620
def monday_sales : ℕ := 50
def tuesday_sales : ℕ := 82
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

theorem percentage_not_sold (initial_stock monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ) :
  (initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) / initial_stock * 100 =
  (620 - (50 + 82 + 60 + 48 + 40)) / 620 * 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_not_sold_l2300_230022


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2300_230018

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The fifth term of a geometric sequence. -/
def FifthTerm (a : ℕ → ℝ) : ℝ := a 5

theorem geometric_sequence_fifth_term
    (a : ℕ → ℝ)
    (h_pos : ∀ n, a n > 0)
    (h_geom : IsGeometricSequence a)
    (h_prod : a 1 * a 3 = 16)
    (h_sum : a 3 + a 4 = 24) :
  FifthTerm a = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2300_230018


namespace NUMINAMATH_CALUDE_triangle_collinearity_l2300_230049

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the orthocenter H
variable (H : ℝ × ℝ)

-- Define points M and N
variable (M N : ℝ × ℝ)

-- Define the circumcenter O of triangle HMN
variable (O : ℝ × ℝ)

-- Define point D
variable (D : ℝ × ℝ)

-- Define the conditions
def is_acute_triangle (A B C : ℝ × ℝ) : Prop := sorry

def angle_A_greater_than_60 (A B C : ℝ × ℝ) : Prop := sorry

def is_orthocenter (H A B C : ℝ × ℝ) : Prop := sorry

def on_side (M A B : ℝ × ℝ) : Prop := sorry

def angle_equals_60 (H M B : ℝ × ℝ) : Prop := sorry

def is_circumcenter (O H M N : ℝ × ℝ) : Prop := sorry

def forms_equilateral_triangle (D B C : ℝ × ℝ) : Prop := sorry

def same_side_as_A (D A B C : ℝ × ℝ) : Prop := sorry

def are_collinear (H O D : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem triangle_collinearity 
  (h_acute : is_acute_triangle A B C)
  (h_angle_A : angle_A_greater_than_60 A B C)
  (h_orthocenter : is_orthocenter H A B C)
  (h_M_on_AB : on_side M A B)
  (h_N_on_AC : on_side N A C)
  (h_angle_HMB : angle_equals_60 H M B)
  (h_angle_HNC : angle_equals_60 H N C)
  (h_circumcenter : is_circumcenter O H M N)
  (h_equilateral : forms_equilateral_triangle D B C)
  (h_same_side : same_side_as_A D A B C) :
  are_collinear H O D :=
sorry

end NUMINAMATH_CALUDE_triangle_collinearity_l2300_230049


namespace NUMINAMATH_CALUDE_probability_three_heads_seven_tosses_prove_probability_three_heads_seven_tosses_l2300_230051

/-- The probability of getting exactly 3 heads in 7 fair coin tosses -/
theorem probability_three_heads_seven_tosses : ℚ :=
  35 / 128

/-- Prove that the probability of getting exactly 3 heads in 7 fair coin tosses is 35/128 -/
theorem prove_probability_three_heads_seven_tosses :
  probability_three_heads_seven_tosses = 35 / 128 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_heads_seven_tosses_prove_probability_three_heads_seven_tosses_l2300_230051


namespace NUMINAMATH_CALUDE_sum_first_three_terms_l2300_230074

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  /-- The common difference between consecutive terms -/
  d : ℝ
  /-- The first term of the sequence -/
  a : ℝ
  /-- The eighth term of the sequence is 20 -/
  eighth_term : a + 7 * d = 20
  /-- The common difference is 2 -/
  diff_is_two : d = 2

/-- The sum of the first three terms of the arithmetic sequence is 24 -/
theorem sum_first_three_terms (seq : ArithmeticSequence) :
  seq.a + (seq.a + seq.d) + (seq.a + 2 * seq.d) = 24 := by
  sorry


end NUMINAMATH_CALUDE_sum_first_three_terms_l2300_230074


namespace NUMINAMATH_CALUDE_sticker_pages_l2300_230078

theorem sticker_pages (stickers_per_page : ℕ) (total_stickers : ℕ) (pages : ℕ) : 
  stickers_per_page = 10 →
  total_stickers = 220 →
  pages * stickers_per_page = total_stickers →
  pages = 22 := by
sorry

end NUMINAMATH_CALUDE_sticker_pages_l2300_230078


namespace NUMINAMATH_CALUDE_box_surface_area_l2300_230055

/-- Calculates the surface area of the interior of a box formed by removing squares from corners of a rectangular sheet --/
def interior_surface_area (sheet_length : ℕ) (sheet_width : ℕ) (corner_size : ℕ) : ℕ :=
  let original_area := sheet_length * sheet_width
  let corner_area := corner_size * corner_size
  let total_removed_area := 4 * corner_area
  original_area - total_removed_area

/-- Theorem stating that the surface area of the interior of the box is 1379 square units --/
theorem box_surface_area :
  interior_surface_area 35 45 7 = 1379 :=
by sorry

end NUMINAMATH_CALUDE_box_surface_area_l2300_230055


namespace NUMINAMATH_CALUDE_prob_two_dice_show_two_is_15_64_l2300_230023

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of at least one of two fair n-sided dice showing a specific number -/
def prob_at_least_one (n : ℕ) : ℚ :=
  1 - (n - 1)^2 / n^2

/-- The probability of at least one of two fair 8-sided dice showing a 2 -/
def prob_two_dice_show_two : ℚ := prob_at_least_one num_sides

theorem prob_two_dice_show_two_is_15_64 : 
  prob_two_dice_show_two = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_dice_show_two_is_15_64_l2300_230023


namespace NUMINAMATH_CALUDE_median_salary_is_40000_l2300_230025

/-- Represents a position in the company with its title, number of employees, and salary. -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company. -/
def positions : List Position := [
  ⟨"President", 1, 160000⟩,
  ⟨"Vice-President", 4, 105000⟩,
  ⟨"Director", 15, 80000⟩,
  ⟨"Associate Director", 10, 55000⟩,
  ⟨"Senior Manager", 20, 40000⟩,
  ⟨"Administrative Specialist", 50, 28000⟩
]

/-- The total number of employees in the company. -/
def totalEmployees : Nat := 100

/-- Calculates the median salary of the employees. -/
def medianSalary (pos : List Position) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median salary is $40,000. -/
theorem median_salary_is_40000 :
  medianSalary positions totalEmployees = 40000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_40000_l2300_230025


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l2300_230079

theorem second_term_of_geometric_series :
  ∀ (a : ℝ) (r : ℝ) (S : ℝ),
    r = (1 : ℝ) / 4 →
    S = 16 →
    S = a / (1 - r) →
    a * r = 3 :=
by sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l2300_230079


namespace NUMINAMATH_CALUDE_customer_difference_l2300_230084

theorem customer_difference (initial : ℕ) (remaining : ℕ) 
  (h1 : initial = 19) (h2 : remaining = 4) : 
  initial - remaining = 15 := by
  sorry

end NUMINAMATH_CALUDE_customer_difference_l2300_230084


namespace NUMINAMATH_CALUDE_least_with_twelve_factors_l2300_230087

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- A function that returns true if n is the least positive integer with exactly k factors -/
def is_least_with_factors (n k : ℕ+) : Prop :=
  (num_factors n = k) ∧ ∀ m : ℕ+, m < n → num_factors m ≠ k

theorem least_with_twelve_factors :
  is_least_with_factors 72 12 := by sorry

end NUMINAMATH_CALUDE_least_with_twelve_factors_l2300_230087


namespace NUMINAMATH_CALUDE_nested_inequality_l2300_230016

/-- A function is ascendant if it preserves order -/
def Ascendant (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem nested_inequality (f g φ : ℝ → ℝ)
    (hf : Ascendant f) (hg : Ascendant g) (hφ : Ascendant φ)
    (h : ∀ x, f x ≤ g x ∧ g x ≤ φ x) :
    ∀ x, f (f x) ≤ g (g x) ∧ g (g x) ≤ φ (φ x) := by
  sorry

end NUMINAMATH_CALUDE_nested_inequality_l2300_230016


namespace NUMINAMATH_CALUDE_figure_circumference_value_l2300_230021

/-- The circumference of a figure formed by one large semicircular arc and 8 identical small semicircular arcs -/
def figure_circumference (d : ℝ) (π : ℝ) : ℝ :=
  π * d

/-- Theorem stating that the circumference of the described figure is 75.36 -/
theorem figure_circumference_value :
  let d : ℝ := 24
  let π : ℝ := 3.14
  figure_circumference d π = 75.36 := by sorry

end NUMINAMATH_CALUDE_figure_circumference_value_l2300_230021


namespace NUMINAMATH_CALUDE_system_solution_l2300_230094

theorem system_solution : 
  ∀ (x y : ℝ), x > 0 ∧ y > 0 → 
  (x - 3 * Real.sqrt (x * y) - 2 * Real.sqrt (x / y) + 6 = 0 ∧ 
   x^2 * y^2 + x^4 = 82) → 
  ((x = 3 ∧ y = 1/3) ∨ (x = Real.rpow 33 (1/4) ∧ y = 4 / Real.rpow 33 (1/4))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2300_230094


namespace NUMINAMATH_CALUDE_least_multiplier_for_72_to_be_multiple_of_112_l2300_230019

theorem least_multiplier_for_72_to_be_multiple_of_112 :
  (∃ n : ℕ+, (72 * n : ℕ) % 112 = 0 ∧ ∀ m : ℕ+, m < n → (72 * m : ℕ) % 112 ≠ 0) ∧
  (∃ n : ℕ+, n = 14 ∧ (72 * n : ℕ) % 112 = 0 ∧ ∀ m : ℕ+, m < n → (72 * m : ℕ) % 112 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_least_multiplier_for_72_to_be_multiple_of_112_l2300_230019


namespace NUMINAMATH_CALUDE_parabola_translation_l2300_230070

def parabola1 (x : ℝ) := -(x - 1)^2 + 3
def parabola2 (x : ℝ) := -x^2

def translation (f : ℝ → ℝ) (h k : ℝ) : ℝ → ℝ :=
  λ x => f (x + h) + k

theorem parabola_translation :
  ∃ h k : ℝ, (∀ x : ℝ, translation parabola1 h k x = parabola2 x) ∧ h = 1 ∧ k = -3 :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_l2300_230070


namespace NUMINAMATH_CALUDE_tangency_condition_min_area_triangle_l2300_230002

/-- The curve C: x^2 + y^2 - 2x - 2y + 1 = 0 -/
def curve (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- The line l: bx + ay = ab -/
def line (a b x y : ℝ) : Prop :=
  b*x + a*y = a*b

/-- The line l is tangent to the curve C -/
def is_tangent (a b : ℝ) : Prop :=
  ∃ x y, curve x y ∧ line a b x y

theorem tangency_condition (a b : ℝ) (ha : a > 2) (hb : b > 2) (h_tangent : is_tangent a b) :
  (a - 2) * (b - 2) = 2 :=
sorry

theorem min_area_triangle (a b : ℝ) (ha : a > 2) (hb : b > 2) (h_tangent : is_tangent a b) :
  ∃ area : ℝ, area = 3 + 2 * Real.sqrt 2 ∧ 
  ∀ a' b', a' > 2 → b' > 2 → is_tangent a' b' → (1/2 * a' * b' ≥ area) :=
sorry

end NUMINAMATH_CALUDE_tangency_condition_min_area_triangle_l2300_230002


namespace NUMINAMATH_CALUDE_lg_100_is_proposition_l2300_230024

/-- A proposition is a declarative sentence that can be judged to be true or false. -/
def IsProposition (s : String) : Prop := 
  ∃ (truthValue : Bool), (∀ (evaluation : String → Bool), evaluation s = truthValue)

/-- The statement "lg 100 = 2" -/
def statement : String := "lg 100 = 2"

/-- Theorem: The statement "lg 100 = 2" is a proposition -/
theorem lg_100_is_proposition : IsProposition statement := by
  sorry

end NUMINAMATH_CALUDE_lg_100_is_proposition_l2300_230024


namespace NUMINAMATH_CALUDE_wedge_volume_l2300_230029

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (angle : ℝ) : 
  d = 20 →
  angle = 60 →
  (1 / 6) * d^3 * π = 667 * π := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l2300_230029


namespace NUMINAMATH_CALUDE_most_accurate_reading_l2300_230036

def scale_reading : ℝ → Prop :=
  λ x => 3.25 < x ∧ x < 3.5

def closer_to_3_3 (x : ℝ) : Prop :=
  |x - 3.3| < |x - 3.375|

def options : Set ℝ :=
  {3.05, 3.15, 3.25, 3.3, 3.6}

theorem most_accurate_reading (x : ℝ) 
  (h1 : scale_reading x) 
  (h2 : closer_to_3_3 x) : 
  ∀ y ∈ options, |x - 3.3| ≤ |x - y| :=
by sorry

end NUMINAMATH_CALUDE_most_accurate_reading_l2300_230036


namespace NUMINAMATH_CALUDE_inequality_condition_l2300_230011

theorem inequality_condition (a x : ℝ) : x^3 + 13*a^2*x > 5*a*x^2 + 9*a^3 ↔ x > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l2300_230011


namespace NUMINAMATH_CALUDE_tetrahedron_altitude_impossibility_l2300_230082

theorem tetrahedron_altitude_impossibility : ∀ (S₁ S₂ S₃ S₄ : ℝ),
  S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 →
  ¬ ∃ (h₁ h₂ h₃ h₄ : ℝ),
    h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ h₄ > 0 ∧
    (S₁ * h₁ = S₂ * h₂) ∧ (S₁ * h₁ = S₃ * h₃) ∧ (S₁ * h₁ = S₄ * h₄) ∧
    h₁ = 4 ∧ h₂ = 25 * Real.sqrt 3 / 3 ∧ h₃ = 25 * Real.sqrt 3 / 3 ∧ h₄ = 25 * Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_tetrahedron_altitude_impossibility_l2300_230082


namespace NUMINAMATH_CALUDE_inequality_condition_not_sufficient_l2300_230071

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) → 0 ≤ a ∧ a < 4 :=
sorry

theorem not_sufficient : 
  ∃ a : ℝ, 0 ≤ a ∧ a < 4 ∧ ∃ x : ℝ, a * x^2 - a * x + 1 ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_not_sufficient_l2300_230071


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2300_230057

theorem point_in_second_quadrant (x y : ℝ) : 
  x < 0 ∧ y > 0 →  -- point is in the second quadrant
  |y| = 4 →        -- 4 units away from x-axis
  |x| = 7 →        -- 7 units away from y-axis
  x = -7 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2300_230057


namespace NUMINAMATH_CALUDE_negative_300_coterminal_with_60_l2300_230044

/-- Two angles are coterminal if their difference is a multiple of 360° -/
def coterminal (a b : ℝ) : Prop := ∃ k : ℤ, a - b = 360 * k

/-- The theorem states that -300° is coterminal with 60° -/
theorem negative_300_coterminal_with_60 : coterminal (-300 : ℝ) 60 := by
  sorry

end NUMINAMATH_CALUDE_negative_300_coterminal_with_60_l2300_230044


namespace NUMINAMATH_CALUDE_q_coordinates_l2300_230096

/-- Triangle ABC with points G on AC and H on AB -/
structure Triangle (A B C G H : ℝ × ℝ) : Prop where
  g_on_ac : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ G = t • C + (1 - t) • A
  h_on_ab : ∃ s : ℝ, 0 < s ∧ s < 1 ∧ H = s • B + (1 - s) • A
  ag_gc_ratio : (G.1 - A.1) / (C.1 - G.1) = 3 / 2 ∧ (G.2 - A.2) / (C.2 - G.2) = 3 / 2
  ah_hb_ratio : (H.1 - A.1) / (B.1 - H.1) = 2 / 3 ∧ (H.2 - A.2) / (B.2 - H.2) = 2 / 3

/-- Q is the intersection of BG and CH -/
def Q (A B C G H : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: Coordinates of Q in terms of A, B, and C -/
theorem q_coordinates (A B C G H : ℝ × ℝ) (tri : Triangle A B C G H) :
  ∃ (u v w : ℝ), u + v + w = 1 ∧ 
    Q A B C G H = (u • A.1 + v • B.1 + w • C.1, u • A.2 + v • B.2 + w • C.2) ∧
    u = 5/13 ∧ v = 11/26 ∧ w = 3/13 :=
  sorry

end NUMINAMATH_CALUDE_q_coordinates_l2300_230096


namespace NUMINAMATH_CALUDE_downstream_speed_l2300_230060

/-- Calculates the downstream speed of a rower given upstream and still water speeds -/
theorem downstream_speed (upstream_speed still_water_speed : ℝ) :
  upstream_speed = 20 →
  still_water_speed = 24 →
  still_water_speed + (still_water_speed - upstream_speed) = 28 := by
  sorry


end NUMINAMATH_CALUDE_downstream_speed_l2300_230060


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l2300_230034

theorem solution_implies_a_value (a : ℝ) : (2 * 2 - a = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l2300_230034


namespace NUMINAMATH_CALUDE_johns_donation_l2300_230004

/-- Calculates the size of a donation that increases the average contribution by 75% to $100 when added to 10 existing contributions. -/
theorem johns_donation (initial_contributions : ℕ) (increase_percentage : ℚ) (new_average : ℚ) : 
  initial_contributions = 10 → 
  increase_percentage = 75 / 100 → 
  new_average = 100 → 
  (11 : ℚ) * new_average - initial_contributions * (new_average / (1 + increase_percentage)) = 3700 / 7 := by
  sorry

#eval (3700 : ℚ) / 7

end NUMINAMATH_CALUDE_johns_donation_l2300_230004


namespace NUMINAMATH_CALUDE_factorization_equality_l2300_230080

theorem factorization_equality (a b : ℝ) : a * b^2 - 2 * a * b + a = a * (b - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2300_230080


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l2300_230063

def systematicSampling (totalProducts : Nat) (sampleSize : Nat) (firstSample : Nat) : List Nat :=
  let interval := totalProducts / sampleSize
  List.range sampleSize |>.map (fun i => firstSample + i * interval)

theorem systematic_sampling_result :
  systematicSampling 60 5 5 = [5, 17, 29, 41, 53] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l2300_230063


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_product_l2300_230066

theorem root_sum_reciprocal_product (p q r s : ℂ) : 
  (p^4 + 6*p^3 + 13*p^2 + 7*p + 3 = 0) →
  (q^4 + 6*q^3 + 13*q^2 + 7*q + 3 = 0) →
  (r^4 + 6*r^3 + 13*r^2 + 7*r + 3 = 0) →
  (s^4 + 6*s^3 + 13*s^2 + 7*s + 3 = 0) →
  (1 / (p*q*r) + 1 / (p*q*s) + 1 / (p*r*s) + 1 / (q*r*s) = -2) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_product_l2300_230066


namespace NUMINAMATH_CALUDE_problem_solution_l2300_230091

theorem problem_solution (p q r s : ℕ+) 
  (h1 : p^3 = q^2) 
  (h2 : r^4 = s^3) 
  (h3 : r - p = 17) : 
  s - q = 73 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2300_230091


namespace NUMINAMATH_CALUDE_original_number_proof_l2300_230095

theorem original_number_proof : ∃ N : ℕ, 
  (N > 30) ∧ 
  (N - 30) % 87 = 0 ∧ 
  (∀ M : ℕ, M > 30 ∧ (M - 30) % 87 = 0 → M ≥ N) :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l2300_230095


namespace NUMINAMATH_CALUDE_angle_system_solution_l2300_230039

theorem angle_system_solution (k : ℤ) :
  let x : ℝ := π/3 + k*π
  let y : ℝ := k*π
  (x - y = π/3) ∧ (Real.tan x - Real.tan y = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_angle_system_solution_l2300_230039


namespace NUMINAMATH_CALUDE_sector_arc_length_l2300_230001

/-- Given a circular sector with area 10 cm² and central angle 2 radians,
    the arc length of the sector is 2√10 cm. -/
theorem sector_arc_length (S : ℝ) (α : ℝ) (l : ℝ) :
  S = 10 →  -- Area of the sector
  α = 2 →   -- Central angle in radians
  l = 2 * Real.sqrt 10 -- Arc length
  := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2300_230001


namespace NUMINAMATH_CALUDE_book_collection_problem_l2300_230083

/-- The number of books in either Jessica's or Tina's collection, but not both -/
def unique_books (shared : ℕ) (jessica_total : ℕ) (tina_unique : ℕ) : ℕ :=
  (jessica_total - shared) + tina_unique

theorem book_collection_problem :
  unique_books 12 22 10 = 20 := by
sorry

end NUMINAMATH_CALUDE_book_collection_problem_l2300_230083


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l2300_230075

theorem reciprocal_inequality (a b : ℝ) :
  (∀ a b, b < a ∧ a < 0 → 1/b > 1/a) ∧
  (∃ a b, 1/b > 1/a ∧ ¬(b < a ∧ a < 0)) :=
sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l2300_230075


namespace NUMINAMATH_CALUDE_checkerboard_squares_l2300_230077

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  row : Nat
  col : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 10

/-- Checks if a square is valid on the board -/
def isValidSquare (s : Square) : Bool :=
  s.size > 0 && s.size <= boardSize && s.row + s.size <= boardSize && s.col + s.size <= boardSize

/-- Counts the number of black squares in a given square -/
def countBlackSquares (s : Square) : Nat :=
  sorry

/-- Counts the number of valid squares with at least 6 black squares -/
def countValidSquares : Nat :=
  sorry

theorem checkerboard_squares : countValidSquares = 155 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_squares_l2300_230077


namespace NUMINAMATH_CALUDE_calculate_F_l2300_230098

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 5*a + 6
def F (a b c : ℝ) : ℝ := b^2 + a*c + 1

-- State the theorem
theorem calculate_F : F 3 (f 3) (f 5) = 19 := by
  sorry

end NUMINAMATH_CALUDE_calculate_F_l2300_230098


namespace NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l2300_230006

theorem least_n_with_gcd_conditions : 
  ∃ (n : ℕ), n > 1000 ∧ 
  Nat.gcd 63 (n + 120) = 21 ∧ 
  Nat.gcd (n + 63) 120 = 60 ∧
  (∀ m : ℕ, m > 1000 ∧ m < n → 
    Nat.gcd 63 (m + 120) ≠ 21 ∨ 
    Nat.gcd (m + 63) 120 ≠ 60) ∧
  n = 1917 :=
by sorry

end NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l2300_230006


namespace NUMINAMATH_CALUDE_a_2n_is_perfect_square_l2300_230056

/-- Define a function that counts the number of natural numbers with a given digit sum,
    where each digit can only be 1, 3, or 4 -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n, a(2n) is a perfect square -/
theorem a_2n_is_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k ^ 2 := by sorry

end NUMINAMATH_CALUDE_a_2n_is_perfect_square_l2300_230056


namespace NUMINAMATH_CALUDE_infinite_parallel_lines_l2300_230027

/-- A plane in 3D space -/
structure Plane3D where
  -- Define the plane (implementation details omitted)

/-- A point in 3D space -/
structure Point3D where
  -- Define the point (implementation details omitted)

/-- A line in 3D space -/
structure Line3D where
  -- Define the line (implementation details omitted)

/-- Predicate to check if a point is not on a plane -/
def notOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line is parallel to a plane -/
def isParallelToPlane (l : Line3D) (plane : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line passes through a point -/
def passesThroughPoint (l : Line3D) (p : Point3D) : Prop :=
  sorry

/-- The main theorem -/
theorem infinite_parallel_lines
  (plane : Plane3D) (p : Point3D) (h : notOnPlane p plane) :
  ∃ (s : Set Line3D), (∀ l ∈ s, isParallelToPlane l plane ∧ passesThroughPoint l p) ∧ Set.Infinite s :=
sorry

end NUMINAMATH_CALUDE_infinite_parallel_lines_l2300_230027


namespace NUMINAMATH_CALUDE_age_of_a_l2300_230000

/-- Given the ages of four people a, b, c, and d, prove that the age of a is 11 years. -/
theorem age_of_a (A B C D : ℕ) : 
  A + B + C + D = 76 →
  ∃ (k : ℕ), A - 3 = k ∧ B - 3 = 2 * k ∧ C - 3 = 3 * k →
  ∃ (m : ℕ), A - 5 = 3 * m ∧ D - 5 = 4 * m ∧ B - 5 = 5 * m →
  A = 11 := by
  sorry

end NUMINAMATH_CALUDE_age_of_a_l2300_230000


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l2300_230008

theorem integer_solutions_quadratic_equation :
  {(x, y) : ℤ × ℤ | x^2 + y^2 = x + y + 2} =
  {(2, 1), (2, 0), (-1, 1), (-1, 0)} := by sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l2300_230008


namespace NUMINAMATH_CALUDE_student_weight_l2300_230064

theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight - 5 = 2 * sister_weight)
  (h2 : student_weight + sister_weight = 110) :
  student_weight = 75 := by
sorry

end NUMINAMATH_CALUDE_student_weight_l2300_230064


namespace NUMINAMATH_CALUDE_finish_in_sixteen_days_l2300_230042

/-- Represents Jack's reading pattern and book information -/
structure ReadingPattern where
  totalPages : Nat
  weekdayPages : Nat
  weekendPages : Nat
  weekdaySkip : Nat
  weekendSkip : Nat

/-- Calculates the number of days it takes to read the book -/
def daysToFinish (pattern : ReadingPattern) : Nat :=
  sorry

/-- Theorem stating that it takes 16 days to finish the book with the given reading pattern -/
theorem finish_in_sixteen_days :
  daysToFinish { totalPages := 285
                , weekdayPages := 23
                , weekendPages := 35
                , weekdaySkip := 3
                , weekendSkip := 2 } = 16 := by
  sorry

end NUMINAMATH_CALUDE_finish_in_sixteen_days_l2300_230042


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2300_230048

open Real

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  3 * cos θ + 1 / sin θ + 2 * tan θ ≥ 3 * Real.rpow 6 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2300_230048


namespace NUMINAMATH_CALUDE_wedding_champagne_bottles_l2300_230041

/-- The number of wedding guests -/
def num_guests : ℕ := 120

/-- The number of glasses of champagne per guest -/
def glasses_per_guest : ℕ := 2

/-- The number of servings per bottle of champagne -/
def servings_per_bottle : ℕ := 6

/-- The number of bottles of champagne needed for the wedding toast -/
def bottles_needed : ℕ := (num_guests * glasses_per_guest) / servings_per_bottle

theorem wedding_champagne_bottles : bottles_needed = 40 := by
  sorry

end NUMINAMATH_CALUDE_wedding_champagne_bottles_l2300_230041


namespace NUMINAMATH_CALUDE_product_remainder_theorem_l2300_230010

def numbers : List Nat := [445876, 985420, 215546, 656452, 387295]

def remainder_sum_squares (nums : List Nat) : Nat :=
  (nums.map (λ n => (n^2) % 8)).sum

theorem product_remainder_theorem :
  (remainder_sum_squares numbers) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_theorem_l2300_230010


namespace NUMINAMATH_CALUDE_ellipse_and_line_equation_l2300_230099

-- Define the ellipse C₁
def C₁ (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the point M
def M (x y : ℝ) : Prop := C₁ 2 (Real.sqrt 3) x y ∧ C₂ x y ∧ x > 0 ∧ y > 0

-- Define the distance between M and F₂
def MF₂_distance (x y : ℝ) : Prop := (x - 1)^2 + y^2 = (5/3)^2

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := y = Real.sqrt 6 * (x - m)

-- Define the perpendicularity condition
def perpendicular_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_and_line_equation :
  ∃ (x y : ℝ),
    M x y ∧ MF₂_distance x y ∧
    (∀ (m : ℝ),
      (∃ (x₁ y₁ x₂ y₂ : ℝ),
        C₁ 2 (Real.sqrt 3) x₁ y₁ ∧ C₁ 2 (Real.sqrt 3) x₂ y₂ ∧
        line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧
        perpendicular_condition x₁ y₁ x₂ y₂) →
      m = Real.sqrt 2 ∨ m = -Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_equation_l2300_230099


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l2300_230068

-- Define the function y
def y (a b x : ℝ) : ℝ := a * x^2 + x - b

-- Part 1
theorem part_one (a : ℝ) :
  (∃! x, y a 1 x = 0) → (a = -1/4 ∨ a = 0) :=
sorry

-- Part 2
theorem part_two (a b x : ℝ) :
  y a b x < (a-1) * x^2 + (b+2) * x - 2*b ↔
    (b < 1 ∧ b < x ∧ x < 1) ∨
    (b > 1 ∧ 1 < x ∧ x < b) :=
sorry

-- Part 3
theorem part_three (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (∀ t > 0, ∃ x, y a b x > 0 ∧ -2-t < x ∧ x < -2+t) →
  (∃ m, m = 1/a - 1/b ∧ ∀ a' b', a' > 0 → b' > 1 →
    (∀ t > 0, ∃ x, y a' b' x > 0 ∧ -2-t < x ∧ x < -2+t) →
    1/a' - 1/b' ≤ m) ∧
  m = 1/2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l2300_230068


namespace NUMINAMATH_CALUDE_largest_increase_1993_l2300_230092

/-- Profit margin percentages for each year from 1990 to 1999 -/
def profitMargins : Fin 10 → ℝ
  | 0 => 10
  | 1 => 20
  | 2 => 30
  | 3 => 60
  | 4 => 70
  | 5 => 75
  | 6 => 80
  | 7 => 82
  | 8 => 86
  | 9 => 70

/-- Calculate the percentage increase between two years -/
def percentageIncrease (year1 year2 : Fin 10) : ℝ :=
  profitMargins year2 - profitMargins year1

/-- The year with the largest percentage increase -/
def yearWithLargestIncrease : Fin 10 :=
  3  -- Representing 1993 (index 3 corresponds to 1993)

/-- Theorem stating that 1993 (index 3) has the largest percentage increase -/
theorem largest_increase_1993 :
  ∀ (year : Fin 9), percentageIncrease year (year + 1) ≤ percentageIncrease 2 3 :=
sorry

end NUMINAMATH_CALUDE_largest_increase_1993_l2300_230092


namespace NUMINAMATH_CALUDE_five_two_difference_in_book_pages_l2300_230076

/-- Count occurrences of a digit in a range of numbers -/
def countDigit (d : Nat) (start finish : Nat) : Nat :=
  sorry

/-- The difference between occurrences of 5 and 2 in page numbers -/
def diffFiveTwo (totalPages : Nat) : Int :=
  (countDigit 5 1 totalPages : Int) - (countDigit 2 1 totalPages : Int)

/-- Theorem stating the difference between 5's and 2's in a 625-page book -/
theorem five_two_difference_in_book_pages : diffFiveTwo 625 = 20 := by
  sorry

end NUMINAMATH_CALUDE_five_two_difference_in_book_pages_l2300_230076
