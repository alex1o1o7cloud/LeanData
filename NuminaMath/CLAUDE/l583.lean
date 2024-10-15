import Mathlib

namespace NUMINAMATH_CALUDE_probability_same_group_is_one_fourth_l583_58335

def number_of_groups : ℕ := 4

def probability_same_group : ℚ :=
  (number_of_groups : ℚ) / ((number_of_groups : ℚ) * (number_of_groups : ℚ))

theorem probability_same_group_is_one_fourth :
  probability_same_group = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_group_is_one_fourth_l583_58335


namespace NUMINAMATH_CALUDE_max_students_before_third_wave_l583_58380

/-- The total number of students in the class -/
def total_students : ℕ := 35

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- The theorem to be proved -/
theorem max_students_before_third_wave :
  ∃ (a b c : ℕ),
    is_prime a ∧ is_prime b ∧ is_prime c ∧
    a + b + c = total_students ∧
    ∀ (x y z : ℕ),
      is_prime x ∧ is_prime y ∧ is_prime z ∧
      x + y + z = total_students →
      total_students - (a + b) ≥ total_students - (x + y) :=
sorry

end NUMINAMATH_CALUDE_max_students_before_third_wave_l583_58380


namespace NUMINAMATH_CALUDE_function_relationship_l583_58368

/-- Given that y-m is directly proportional to 3x+6, where m is a constant,
    and that when x=2, y=4 and when x=3, y=7,
    prove that the function relationship between y and x is y = 3x - 2 -/
theorem function_relationship (m : ℝ) (k : ℝ) :
  (∀ x y, y - m = k * (3 * x + 6)) →
  (4 - m = k * (3 * 2 + 6)) →
  (7 - m = k * (3 * 3 + 6)) →
  ∀ x y, y = 3 * x - 2 := by
sorry


end NUMINAMATH_CALUDE_function_relationship_l583_58368


namespace NUMINAMATH_CALUDE_percentage_problem_l583_58306

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 200) : 
  (1200 / x) * 100 = 120 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l583_58306


namespace NUMINAMATH_CALUDE_equation_roots_l583_58343

theorem equation_roots : ∀ (x : ℝ), x * (x - 3)^2 * (5 + x) = 0 ↔ x ∈ ({0, 3, -5} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_l583_58343


namespace NUMINAMATH_CALUDE_prime_condition_l583_58321

theorem prime_condition (p : ℕ) : 
  Prime p → Prime (p^4 - 3*p^2 + 9) → p = 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_condition_l583_58321


namespace NUMINAMATH_CALUDE_factor_expression_l583_58350

theorem factor_expression (x : ℝ) : 60 * x + 45 + 9 * x^2 = 3 * (3 * x + 5) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l583_58350


namespace NUMINAMATH_CALUDE_inequality_solution_set_l583_58372

theorem inequality_solution_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (x - a) * (x - a^2) < 0} = {x : ℝ | a^2 < x ∧ x < a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l583_58372


namespace NUMINAMATH_CALUDE_inequality_solution_set_l583_58327

theorem inequality_solution_set (x : ℝ) : 
  (x^2 + 8*x < 20) ↔ (-10 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l583_58327


namespace NUMINAMATH_CALUDE_expression_simplification_l583_58348

theorem expression_simplification (a b x y : ℝ) (h : b*x + a*y ≠ 0) :
  (b*x*(a^2*x^2 + 2*a^2*y^2 + b^2*y^2) + a*y*(a^2*x^2 + 2*b^2*x^2 + b^2*y^2)) / (b*x + a*y)
  = (a*x + b*y)^2 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l583_58348


namespace NUMINAMATH_CALUDE_jan_extra_miles_l583_58329

/-- Represents the driving scenario of Ian, Han, and Jan -/
structure DrivingScenario where
  ian_speed : ℝ
  ian_time : ℝ
  han_speed : ℝ
  han_time : ℝ
  jan_speed : ℝ
  jan_time : ℝ

/-- The conditions of the driving scenario -/
def scenario_conditions (s : DrivingScenario) : Prop :=
  s.han_speed = s.ian_speed + 10 ∧
  s.han_time = s.ian_time ∧
  s.jan_time = s.ian_time + 3 ∧
  s.jan_speed = s.ian_speed + 15 ∧
  s.han_speed * s.han_time = s.ian_speed * s.ian_time + 90

/-- The theorem to be proved -/
theorem jan_extra_miles (s : DrivingScenario) :
  scenario_conditions s →
  s.jan_speed * s.jan_time - s.ian_speed * s.ian_time = 210 := by
  sorry


end NUMINAMATH_CALUDE_jan_extra_miles_l583_58329


namespace NUMINAMATH_CALUDE_oil_leaked_before_is_6522_l583_58386

/-- The amount of oil leaked before engineers started to fix the pipe -/
def oil_leaked_before : ℕ := 11687 - 5165

/-- Theorem stating that the amount of oil leaked before engineers started to fix the pipe is 6522 liters -/
theorem oil_leaked_before_is_6522 : oil_leaked_before = 6522 := by
  sorry

end NUMINAMATH_CALUDE_oil_leaked_before_is_6522_l583_58386


namespace NUMINAMATH_CALUDE_class_size_l583_58332

theorem class_size (average_age : ℝ) (new_average : ℝ) (student_leave_age : ℝ) (teacher_age : ℝ)
  (h1 : average_age = 10)
  (h2 : new_average = 11)
  (h3 : student_leave_age = 11)
  (h4 : teacher_age = 41) :
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * average_age = (n : ℝ) * new_average - teacher_age + student_leave_age :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_l583_58332


namespace NUMINAMATH_CALUDE_num_non_officers_calculation_l583_58309

-- Define the problem parameters
def avg_salary_all : ℝ := 120
def avg_salary_officers : ℝ := 420
def avg_salary_non_officers : ℝ := 110
def num_officers : ℕ := 15

-- Define the theorem
theorem num_non_officers_calculation :
  ∃ (num_non_officers : ℕ),
    (num_officers : ℝ) * avg_salary_officers + (num_non_officers : ℝ) * avg_salary_non_officers =
    ((num_officers : ℝ) + (num_non_officers : ℝ)) * avg_salary_all ∧
    num_non_officers = 450 := by
  sorry

end NUMINAMATH_CALUDE_num_non_officers_calculation_l583_58309


namespace NUMINAMATH_CALUDE_three_diamonds_balance_six_dots_l583_58339

-- Define the symbols
variable (triangle diamond dot : ℕ)

-- Define the balance relation
def balances (left right : ℕ) : Prop := left = right

-- State the given conditions
axiom balance1 : balances (4 * triangle + 2 * diamond) (12 * dot)
axiom balance2 : balances (2 * triangle) (diamond + 2 * dot)

-- State the theorem to be proved
theorem three_diamonds_balance_six_dots : balances (3 * diamond) (6 * dot) := by
  sorry

end NUMINAMATH_CALUDE_three_diamonds_balance_six_dots_l583_58339


namespace NUMINAMATH_CALUDE_plane_perpendicular_criterion_l583_58377

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_criterion 
  (m : Line) (α β : Plane) :
  contains β m → perp m α → perpPlanes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_criterion_l583_58377


namespace NUMINAMATH_CALUDE_plywood_perimeter_difference_l583_58340

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its cutting --/
structure Plywood where
  length : ℝ
  width : ℝ
  num_pieces : ℕ

/-- Generates all possible ways to cut the plywood into congruent rectangles --/
def possible_cuts (p : Plywood) : List Rectangle :=
  sorry -- Implementation details omitted

/-- Finds the maximum perimeter among the possible cuts --/
def max_perimeter (cuts : List Rectangle) : ℝ :=
  sorry -- Implementation details omitted

/-- Finds the minimum perimeter among the possible cuts --/
def min_perimeter (cuts : List Rectangle) : ℝ :=
  sorry -- Implementation details omitted

theorem plywood_perimeter_difference :
  let p : Plywood := { length := 6, width := 9, num_pieces := 6 }
  let cuts := possible_cuts p
  max_perimeter cuts - min_perimeter cuts = 11 := by
  sorry

end NUMINAMATH_CALUDE_plywood_perimeter_difference_l583_58340


namespace NUMINAMATH_CALUDE_fred_cards_l583_58337

theorem fred_cards (initial_cards torn_cards bought_cards total_cards : ℕ) : 
  initial_cards = 18 →
  torn_cards = 8 →
  bought_cards = 40 →
  total_cards = 84 →
  total_cards = initial_cards - torn_cards + bought_cards + (total_cards - (initial_cards - torn_cards + bought_cards)) →
  total_cards - (initial_cards - torn_cards + bought_cards) = 34 := by
  sorry

end NUMINAMATH_CALUDE_fred_cards_l583_58337


namespace NUMINAMATH_CALUDE_conference_attendees_l583_58365

theorem conference_attendees (total : ℕ) (creators : ℕ) (editors : ℕ) (y : ℕ) :
  total = 200 →
  creators = 80 →
  editors = 65 →
  total = creators + editors - y + 3 * y →
  y ≤ 27 ∧ ∃ (y : ℕ), y = 27 ∧ total = creators + editors - y + 3 * y :=
by sorry

end NUMINAMATH_CALUDE_conference_attendees_l583_58365


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l583_58364

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  base_diff : ℝ
  midpoint_ratio : ℝ × ℝ
  equal_area_segment : ℝ

/-- The trapezoid satisfying the problem conditions -/
def problem_trapezoid : Trapezoid where
  base_diff := 120
  midpoint_ratio := (3, 4)
  equal_area_segment := x
  where x : ℝ := sorry  -- The actual value of x will be determined in the proof

/-- The theorem to be proved -/
theorem trapezoid_segment_length (t : Trapezoid) : 
  t = problem_trapezoid → ⌊(t.equal_area_segment^2) / 120⌋ = 270 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l583_58364


namespace NUMINAMATH_CALUDE_largest_n_for_product_2304_l583_58351

theorem largest_n_for_product_2304 :
  ∀ (d_a d_b : ℤ),
  ∃ (n : ℕ),
  (∀ k : ℕ, (1 + (k - 1) * d_a) * (3 + (k - 1) * d_b) = 2304 → k ≤ n) ∧
  (1 + (n - 1) * d_a) * (3 + (n - 1) * d_b) = 2304 ∧
  n = 20 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_product_2304_l583_58351


namespace NUMINAMATH_CALUDE_vat_volume_l583_58341

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The volume of juice in each glass (in pints) -/
def volume_per_glass : ℕ := 30

/-- Theorem: The total volume of orange juice in the vat is 150 pints -/
theorem vat_volume : num_glasses * volume_per_glass = 150 := by
  sorry

end NUMINAMATH_CALUDE_vat_volume_l583_58341


namespace NUMINAMATH_CALUDE_gray_eyed_black_haired_count_l583_58363

/-- Represents the characteristics of models in a modeling agency. -/
structure ModelAgency where
  total : Nat
  redHaired : Nat
  blackHaired : Nat
  greenEyed : Nat
  grayEyed : Nat
  greenEyedRedHaired : Nat

/-- Conditions for the modeling agency problem. -/
def modelingAgencyConditions : ModelAgency :=
  { total := 60
  , redHaired := 24  -- Derived from total - blackHaired
  , blackHaired := 36
  , greenEyed := 36  -- Derived from total - grayEyed
  , grayEyed := 24
  , greenEyedRedHaired := 22 }

/-- Theorem stating that the number of gray-eyed black-haired models is 10. -/
theorem gray_eyed_black_haired_count (agency : ModelAgency) 
  (h1 : agency = modelingAgencyConditions) : 
  agency.blackHaired + agency.greenEyedRedHaired - agency.greenEyed = 10 := by
  sorry

#check gray_eyed_black_haired_count

end NUMINAMATH_CALUDE_gray_eyed_black_haired_count_l583_58363


namespace NUMINAMATH_CALUDE_value_of_b_l583_58313

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l583_58313


namespace NUMINAMATH_CALUDE_empirical_regression_equation_l583_58389

/-- Data for 10 years of resident income and goods sales -/
def income : List Float := [32.2, 31.1, 32.9, 35.7, 37.1, 38.0, 39.0, 43.0, 44.6, 46.0]
def sales : List Float := [25.0, 30.0, 34.0, 37.0, 39.0, 41.0, 42.0, 44.0, 48.0, 51.0]

/-- Given statistics -/
def sum_x : Float := 379.6
def sum_y : Float := 391.0
def sum_x_squared : Float := 246.904
def sum_y_squared : Float := 568.9
def correlation_coefficient : Float := 0.95

/-- Mean values -/
def mean_x : Float := sum_x / 10
def mean_y : Float := sum_y / 10

/-- Regression coefficients -/
def b_hat : Float := correlation_coefficient * (sum_y_squared.sqrt / sum_x_squared.sqrt)
def a_hat : Float := mean_y - b_hat * mean_x

theorem empirical_regression_equation :
  (b_hat * 100).round / 100 = 1.44 ∧ 
  (a_hat * 100).round / 100 = -15.56 := by
  sorry

#check empirical_regression_equation

end NUMINAMATH_CALUDE_empirical_regression_equation_l583_58389


namespace NUMINAMATH_CALUDE_number_order_l583_58325

theorem number_order : 
  (1 * 4^3) < (8 * 9 + 5) ∧ (8 * 9 + 5) < (2 * 6^2 + 1 * 6 + 0) := by
  sorry

end NUMINAMATH_CALUDE_number_order_l583_58325


namespace NUMINAMATH_CALUDE_three_not_in_range_iff_c_gt_four_l583_58316

/-- The function g(x) = x^2 + 2x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + c

/-- 3 is not in the range of g(x) if and only if c > 4 -/
theorem three_not_in_range_iff_c_gt_four (c : ℝ) :
  (∀ x : ℝ, g c x ≠ 3) ↔ c > 4 := by
  sorry

end NUMINAMATH_CALUDE_three_not_in_range_iff_c_gt_four_l583_58316


namespace NUMINAMATH_CALUDE_figure_36_to_square_cut_and_rearrange_to_square_l583_58310

/-- Represents a figure made up of small squares --/
structure Figure where
  squares : ℕ

/-- Represents a square --/
structure Square where
  side_length : ℕ

/-- Function to check if a figure can be rearranged into a square --/
def can_form_square (f : Figure) : Prop :=
  ∃ (s : Square), s.side_length * s.side_length = f.squares

/-- Theorem stating that a figure with 36 squares can form a square --/
theorem figure_36_to_square :
  ∀ (f : Figure), f.squares = 36 → can_form_square f :=
by
  sorry

/-- Theorem stating that a figure with 36 squares can be cut into two pieces
    and rearranged to form a square --/
theorem cut_and_rearrange_to_square :
  ∀ (f : Figure), f.squares = 36 →
  ∃ (piece1 piece2 : Figure),
    piece1.squares + piece2.squares = f.squares ∧
    can_form_square (Figure.mk (piece1.squares + piece2.squares)) :=
by
  sorry

end NUMINAMATH_CALUDE_figure_36_to_square_cut_and_rearrange_to_square_l583_58310


namespace NUMINAMATH_CALUDE_equation_roots_imply_a_range_l583_58326

theorem equation_roots_imply_a_range :
  ∀ a : ℝ, (∃ x : ℝ, (2 - 2^(-|x - 3|))^2 = 3 + a) → -2 ≤ a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_imply_a_range_l583_58326


namespace NUMINAMATH_CALUDE_floor_sum_of_positive_reals_l583_58311

theorem floor_sum_of_positive_reals (u v w x : ℝ) 
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x)
  (h1 : u^2 + v^2 = 3005) (h2 : w^2 + x^2 = 3005)
  (h3 : u * w = 1729) (h4 : v * x = 1729) : 
  ⌊u + v + w + x⌋ = 155 := by
sorry

end NUMINAMATH_CALUDE_floor_sum_of_positive_reals_l583_58311


namespace NUMINAMATH_CALUDE_circle_line_distance_l583_58334

theorem circle_line_distance (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + 1 = 0) → 
  (∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 4) →
  (|a + 1| / Real.sqrt (a^2 + 1) = 1) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_circle_line_distance_l583_58334


namespace NUMINAMATH_CALUDE_milk_packs_per_set_l583_58362

/-- The number of packs in each set of milk -/
def packs_per_set : ℕ := sorry

/-- The cost of a set of milk packs in dollars -/
def cost_per_set : ℚ := 2.5

/-- The cost of an individual milk pack in dollars -/
def cost_per_pack : ℚ := 1.3

/-- The total savings from buying ten sets in dollars -/
def total_savings : ℚ := 1

theorem milk_packs_per_set :
  packs_per_set = 2 ∧
  10 * cost_per_set + total_savings = 10 * packs_per_set * cost_per_pack :=
sorry

end NUMINAMATH_CALUDE_milk_packs_per_set_l583_58362


namespace NUMINAMATH_CALUDE_perfect_square_condition_l583_58375

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that
    ax^2 + bx + c = (px + q)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_condition (m : ℝ) :
  is_perfect_square_trinomial 1 m 16 → m = 8 ∨ m = -8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l583_58375


namespace NUMINAMATH_CALUDE_circle_combined_value_l583_58315

/-- The combined value of circumference and area for a circle with radius 13 cm -/
theorem circle_combined_value :
  let r : ℝ := 13
  let π : ℝ := Real.pi
  let circumference : ℝ := 2 * π * r
  let area : ℝ := π * r^2
  abs ((circumference + area) - 612.6105) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_circle_combined_value_l583_58315


namespace NUMINAMATH_CALUDE_treasure_hunt_probability_l583_58317

def probability_gold : ℚ := 1 / 5
def probability_danger : ℚ := 1 / 10
def probability_neither : ℚ := 4 / 5
def total_caves : ℕ := 5
def gold_caves : ℕ := 2

theorem treasure_hunt_probability :
  (Nat.choose total_caves gold_caves : ℚ) *
  probability_gold ^ gold_caves *
  probability_neither ^ (total_caves - gold_caves) =
  128 / 625 := by sorry

end NUMINAMATH_CALUDE_treasure_hunt_probability_l583_58317


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l583_58381

/-- Given a right triangle with area 24 cm² and hypotenuse 10 cm, 
    prove that the radius of its inscribed circle is 2 cm. -/
theorem inscribed_circle_radius 
  (S : ℝ) 
  (c : ℝ) 
  (h1 : S = 24) 
  (h2 : c = 10) : 
  let a := Real.sqrt ((c^2 / 2) + Real.sqrt ((c^4 / 4) - S^2))
  let b := Real.sqrt ((c^2 / 2) - Real.sqrt ((c^4 / 4) - S^2))
  (a + b - c) / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l583_58381


namespace NUMINAMATH_CALUDE_negation_of_sum_squares_l583_58333

theorem negation_of_sum_squares (x y : ℝ) : -(x^2 + y^2) = -x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_sum_squares_l583_58333


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l583_58330

def trailing_zeros (n : ℕ) : ℕ := sorry

theorem product_trailing_zeros :
  trailing_zeros (50 * 720 * 125) = 5 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l583_58330


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l583_58366

theorem abs_sum_inequality (a b c d : ℝ) 
  (sum_pos : a + b + c + d > 0)
  (a_gt_c : a > c)
  (b_gt_d : b > d) :
  |a + b| > |c + d| := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l583_58366


namespace NUMINAMATH_CALUDE_final_time_and_sum_l583_58322

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time -/
def addTime (start : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

/-- Converts a time to 12-hour format -/
def to12HourFormat (t : Time) : Time :=
  sorry

/-- Theorem: Given the starting time and duration, prove the final time and sum -/
theorem final_time_and_sum 
  (start : Time)
  (durationHours durationMinutes durationSeconds : Nat) : 
  start.hours = 3 ∧ start.minutes = 0 ∧ start.seconds = 0 →
  durationHours = 313 ∧ durationMinutes = 45 ∧ durationSeconds = 56 →
  let finalTime := to12HourFormat (addTime start durationHours durationMinutes durationSeconds)
  finalTime.hours = 4 ∧ finalTime.minutes = 45 ∧ finalTime.seconds = 56 ∧
  finalTime.hours + finalTime.minutes + finalTime.seconds = 105 :=
by sorry

end NUMINAMATH_CALUDE_final_time_and_sum_l583_58322


namespace NUMINAMATH_CALUDE_max_handshakes_l583_58302

theorem max_handshakes (n : ℕ) (h : n = 25) : 
  (n * (n - 1)) / 2 = 300 := by
  sorry

#check max_handshakes

end NUMINAMATH_CALUDE_max_handshakes_l583_58302


namespace NUMINAMATH_CALUDE_horner_rule_V₂_l583_58353

def f (x : ℝ) : ℝ := 2*x^6 + 3*x^5 + 5*x^3 + 6*x^2 + 7*x + 8

def V₂ (x : ℝ) : ℝ := 2

def V₁ (x : ℝ) : ℝ := V₂ x * x + 3

def V₂_final (x : ℝ) : ℝ := V₁ x * x + 0

theorem horner_rule_V₂ : V₂_final 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_V₂_l583_58353


namespace NUMINAMATH_CALUDE_y_work_time_l583_58314

/-- The time it takes for y to complete the work alone, given the conditions -/
def time_y_alone (time_x time_yz time_xz : ℝ) : ℝ :=
  24

/-- Theorem stating that y takes 24 hours to complete the work alone -/
theorem y_work_time (time_x time_yz time_xz : ℝ) 
  (hx : time_x = 8) 
  (hyz : time_yz = 6) 
  (hxz : time_xz = 4) : 
  time_y_alone time_x time_yz time_xz = 24 := by
  sorry

#check y_work_time

end NUMINAMATH_CALUDE_y_work_time_l583_58314


namespace NUMINAMATH_CALUDE_prove_depletion_rate_l583_58349

-- Define the initial value of the machine
def initial_value : ℝ := 2500

-- Define the value of the machine after 2 years
def value_after_2_years : ℝ := 2256.25

-- Define the number of years
def years : ℝ := 2

-- Define the depletion rate
def depletion_rate : ℝ := 0.05

-- Theorem to prove that the given depletion rate is correct
theorem prove_depletion_rate : 
  value_after_2_years = initial_value * (1 - depletion_rate) ^ years := by
  sorry


end NUMINAMATH_CALUDE_prove_depletion_rate_l583_58349


namespace NUMINAMATH_CALUDE_bamboo_pole_problem_l583_58388

theorem bamboo_pole_problem (pole_length : ℝ) (point_distance : ℝ) 
  (h_pole_length : pole_length = 24)
  (h_point_distance : point_distance = 7) :
  ∃ (height : ℝ), height = 16 + 4 * Real.sqrt 2 ∨ height = 16 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_bamboo_pole_problem_l583_58388


namespace NUMINAMATH_CALUDE_eugene_toothpick_boxes_l583_58391

def toothpicks_per_card : ℕ := 64
def total_cards : ℕ := 52
def unused_cards : ℕ := 23
def toothpicks_per_box : ℕ := 550

theorem eugene_toothpick_boxes : 
  ∃ (boxes : ℕ), 
    boxes = (((total_cards - unused_cards) * toothpicks_per_card + toothpicks_per_box - 1) / toothpicks_per_box : ℕ) ∧ 
    boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_eugene_toothpick_boxes_l583_58391


namespace NUMINAMATH_CALUDE_math_books_same_box_probability_l583_58361

def total_textbooks : ℕ := 12
def math_textbooks : ℕ := 3
def box_capacities : List ℕ := [3, 4, 5]

def probability_all_math_in_same_box : ℚ :=
  3 / 44

theorem math_books_same_box_probability :
  probability_all_math_in_same_box = 3 / 44 :=
by sorry

end NUMINAMATH_CALUDE_math_books_same_box_probability_l583_58361


namespace NUMINAMATH_CALUDE_compound_has_three_oxygen_atoms_l583_58371

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  aluminium : Nat
  oxygen : Nat
  hydrogen : Nat

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) : Nat :=
  27 * c.aluminium + 16 * c.oxygen + c.hydrogen

/-- Theorem stating that the compound with 3 oxygen atoms satisfies the given conditions -/
theorem compound_has_three_oxygen_atoms :
  ∃ (c : Compound), c.aluminium = 1 ∧ c.hydrogen = 3 ∧ molecularWeight c = 78 ∧ c.oxygen = 3 := by
  sorry

#check compound_has_three_oxygen_atoms

end NUMINAMATH_CALUDE_compound_has_three_oxygen_atoms_l583_58371


namespace NUMINAMATH_CALUDE_katies_sister_candy_l583_58304

theorem katies_sister_candy (katie_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ) :
  katie_candy = 10 →
  eaten_candy = 9 →
  remaining_candy = 7 →
  ∃ sister_candy : ℕ, sister_candy = 6 ∧ katie_candy + sister_candy = eaten_candy + remaining_candy :=
by sorry

end NUMINAMATH_CALUDE_katies_sister_candy_l583_58304


namespace NUMINAMATH_CALUDE_binomial_expansion_terms_l583_58303

theorem binomial_expansion_terms (n : ℕ+) : 
  (Finset.range (2*n + 1)).card = 2*n + 1 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_terms_l583_58303


namespace NUMINAMATH_CALUDE_function_decreasing_interval_l583_58308

/-- Given a function f(x) = kx³ + 3(k-1)x² - k² + 1 where k > 0,
    and f(x) is decreasing in the interval (0,4),
    prove that k = 4. -/
theorem function_decreasing_interval (k : ℝ) (h1 : k > 0) : 
  (∀ x ∈ Set.Ioo 0 4, 
    (deriv (fun x => k*x^3 + 3*(k-1)*x^2 - k^2 + 1) x) < 0) → 
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_function_decreasing_interval_l583_58308


namespace NUMINAMATH_CALUDE_restaurant_problem_l583_58300

theorem restaurant_problem (initial_wings : ℕ) (additional_wings : ℕ) (wings_per_friend : ℕ) 
  (h1 : initial_wings = 9)
  (h2 : additional_wings = 7)
  (h3 : wings_per_friend = 4) :
  (initial_wings + additional_wings) / wings_per_friend = 4 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_problem_l583_58300


namespace NUMINAMATH_CALUDE_remaining_pie_portion_l583_58346

/-- Proves that given Carlos took 60% of a whole pie and Maria took one quarter of the remainder, 
    the portion of the whole pie left is 30%. -/
theorem remaining_pie_portion 
  (carlos_portion : Real) 
  (maria_portion : Real) 
  (h1 : carlos_portion = 0.6) 
  (h2 : maria_portion = 0.25 * (1 - carlos_portion)) : 
  1 - carlos_portion - maria_portion = 0.3 := by
sorry

end NUMINAMATH_CALUDE_remaining_pie_portion_l583_58346


namespace NUMINAMATH_CALUDE_vasya_wins_in_four_moves_l583_58354

-- Define a polynomial with integer coefficients
def IntPolynomial := ℤ → ℤ

-- Define a function that counts the number of integer solutions for P(x) = a
def countIntegerSolutions (P : IntPolynomial) (a : ℤ) : ℕ :=
  sorry

-- Theorem statement
theorem vasya_wins_in_four_moves :
  ∀ (P : IntPolynomial),
  ∃ (S : Finset ℤ),
  (Finset.card S ≤ 4) ∧
  ∃ (a b : ℤ),
  a ∈ S ∧ b ∈ S ∧ a ≠ b ∧
  countIntegerSolutions P a = countIntegerSolutions P b :=
sorry

end NUMINAMATH_CALUDE_vasya_wins_in_four_moves_l583_58354


namespace NUMINAMATH_CALUDE_percentage_problem_l583_58383

theorem percentage_problem (p : ℝ) (h1 : 0.25 * 660 = (p/100) * 1500 - 15) : p = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l583_58383


namespace NUMINAMATH_CALUDE_abc_perfect_cube_l583_58331

theorem abc_perfect_cube (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : ∃ (n : ℤ), (a / b : ℚ) + (b / c : ℚ) + (c / a : ℚ) = n) : 
  ∃ (k : ℤ), a * b * c = k^3 := by
  sorry

end NUMINAMATH_CALUDE_abc_perfect_cube_l583_58331


namespace NUMINAMATH_CALUDE_cost_per_book_is_5_l583_58345

/-- The cost to produce each book -/
def cost_per_book : ℝ := 5

/-- The selling price of each book -/
def selling_price : ℝ := 20

/-- The total profit -/
def total_profit : ℝ := 120

/-- The number of customers -/
def num_customers : ℕ := 4

/-- The number of books each customer buys -/
def books_per_customer : ℕ := 2

/-- The theorem stating the cost to make each book -/
theorem cost_per_book_is_5 : 
  cost_per_book = 5 :=
by sorry

end NUMINAMATH_CALUDE_cost_per_book_is_5_l583_58345


namespace NUMINAMATH_CALUDE_intersection_M_N_l583_58378

def M : Set ℝ := {-2, -1, 0, 1, 2}

def N : Set ℝ := {x | x < 0 ∨ x > 3}

theorem intersection_M_N : M ∩ N = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l583_58378


namespace NUMINAMATH_CALUDE_remainder_of_B_divided_by_9_l583_58338

theorem remainder_of_B_divided_by_9 (A B : ℕ) (h : B = A * 9 + 13) : B % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_B_divided_by_9_l583_58338


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l583_58390

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * s = 4 * Real.sqrt 241 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l583_58390


namespace NUMINAMATH_CALUDE_complementary_angles_imply_right_triangle_l583_58307

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the Triangle Angle Sum Theorem
axiom triangle_angle_sum (t : Triangle) : t.angle1 + t.angle2 + t.angle3 = 180

-- Define complementary angles
def complementary (a b : ℝ) : Prop := a + b = 90

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop := t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

-- Theorem statement
theorem complementary_angles_imply_right_triangle (t : Triangle) 
  (h : complementary t.angle1 t.angle2 ∨ complementary t.angle1 t.angle3 ∨ complementary t.angle2 t.angle3) : 
  is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_complementary_angles_imply_right_triangle_l583_58307


namespace NUMINAMATH_CALUDE_proposition_relationship_l583_58342

-- Define propositions as variables of type Prop
variable (A B C : Prop)

-- Define the relationships between propositions
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

def necessary_and_sufficient (P Q : Prop) : Prop :=
  (P ↔ Q)

def necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

-- State the theorem
theorem proposition_relationship :
  sufficient_not_necessary A B →
  necessary_and_sufficient B C →
  necessary_not_sufficient C A :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l583_58342


namespace NUMINAMATH_CALUDE_daniel_and_elsie_crackers_l583_58358

/-- The amount of crackers Matthew had initially -/
def initial_crackers : ℝ := 27.5

/-- The amount of crackers Ally ate -/
def ally_crackers : ℝ := 3.5

/-- The amount of crackers Bob ate -/
def bob_crackers : ℝ := 4

/-- The amount of crackers Clair ate -/
def clair_crackers : ℝ := 5.5

/-- The amount of crackers Matthew had left after giving to Ally, Bob, and Clair -/
def remaining_crackers : ℝ := 10.5

/-- The theorem stating that Daniel and Elsie ate 4 crackers combined -/
theorem daniel_and_elsie_crackers : 
  initial_crackers - (ally_crackers + bob_crackers + clair_crackers) - remaining_crackers = 4 := by
  sorry

end NUMINAMATH_CALUDE_daniel_and_elsie_crackers_l583_58358


namespace NUMINAMATH_CALUDE_eight_digit_divisibility_l583_58382

theorem eight_digit_divisibility (A B : ℕ) : 
  A < 10 → B < 10 → (757 * 10^5 + A * 10^4 + B * 10^3 + 384) % 357 = 0 → A = 5 := by
sorry

end NUMINAMATH_CALUDE_eight_digit_divisibility_l583_58382


namespace NUMINAMATH_CALUDE_factory_equation_holds_l583_58320

/-- Represents a factory's part processing scenario -/
def factory_scenario (x : ℝ) : Prop :=
  x > 0 ∧ 
  (100 / x) + (400 / (2 * x)) = 6

/-- Theorem stating the equation holds for the given scenario -/
theorem factory_equation_holds : 
  ∀ x : ℝ, x > 0 → (100 / x) + (400 / (2 * x)) = 6 ↔ factory_scenario x :=
by
  sorry

#check factory_equation_holds

end NUMINAMATH_CALUDE_factory_equation_holds_l583_58320


namespace NUMINAMATH_CALUDE_journey_duration_first_part_l583_58384

/-- Proves the duration of the first part of a journey given specific conditions -/
theorem journey_duration_first_part 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (speed_first_part : ℝ) 
  (speed_second_part : ℝ) 
  (h1 : total_distance = 240)
  (h2 : total_time = 5)
  (h3 : speed_first_part = 40)
  (h4 : speed_second_part = 60) :
  ∃ (t1 : ℝ), t1 = 3 ∧ 
    speed_first_part * t1 + speed_second_part * (total_time - t1) = total_distance :=
by sorry


end NUMINAMATH_CALUDE_journey_duration_first_part_l583_58384


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l583_58393

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x^2 + 1

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 5 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l583_58393


namespace NUMINAMATH_CALUDE_vector_computation_l583_58392

def a : Fin 2 → ℝ := ![2, 4]
def b : Fin 2 → ℝ := ![-1, 1]

theorem vector_computation : 
  (2 • a - b) = ![5, 7] := by sorry

end NUMINAMATH_CALUDE_vector_computation_l583_58392


namespace NUMINAMATH_CALUDE_binomial_1300_2_l583_58398

theorem binomial_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1300_2_l583_58398


namespace NUMINAMATH_CALUDE_fliers_remaining_l583_58357

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ)
  (h1 : total = 2500)
  (h2 : morning_fraction = 1 / 5)
  (h3 : afternoon_fraction = 1 / 4) :
  total - (morning_fraction * total).floor - (afternoon_fraction * (total - (morning_fraction * total).floor)).floor = 1500 :=
by sorry

end NUMINAMATH_CALUDE_fliers_remaining_l583_58357


namespace NUMINAMATH_CALUDE_cylinder_radius_l583_58367

/-- 
Given a right circular cylinder with height h and diagonal d (measured from the center of the
circular base to the top edge of the cylinder), this theorem proves that when h = 12 and d = 13,
the radius r of the cylinder is 5.
-/
theorem cylinder_radius (h d : ℝ) (h_pos : h > 0) (d_pos : d > 0) 
  (h_val : h = 12) (d_val : d = 13) : ∃ r : ℝ, r > 0 ∧ r = 5 ∧ r^2 + h^2 = d^2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_l583_58367


namespace NUMINAMATH_CALUDE_return_speed_calculation_l583_58387

/-- Proves that given a round trip between two cities 150 miles apart,
    where the outbound speed is 50 mph and the average round trip speed is 60 mph,
    the return speed is 75 mph. -/
theorem return_speed_calculation (distance : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) :
  distance = 150 →
  outbound_speed = 50 →
  average_speed = 60 →
  (2 * distance) / (distance / outbound_speed + distance / (2 * distance / average_speed - distance / outbound_speed)) = average_speed →
  2 * distance / average_speed - distance / outbound_speed = 75 := by
  sorry

end NUMINAMATH_CALUDE_return_speed_calculation_l583_58387


namespace NUMINAMATH_CALUDE_backyard_width_calculation_l583_58301

/-- Given a rectangular backyard with a rectangular shed, calculate the width of the backyard -/
theorem backyard_width_calculation 
  (backyard_length : ℝ) 
  (shed_length shed_width : ℝ) 
  (sod_area : ℝ) :
  backyard_length = 20 →
  shed_length = 3 →
  shed_width = 5 →
  sod_area = 245 →
  ∃ (backyard_width : ℝ), 
    sod_area = backyard_length * backyard_width - shed_length * shed_width ∧ 
    backyard_width = 13 :=
by sorry

end NUMINAMATH_CALUDE_backyard_width_calculation_l583_58301


namespace NUMINAMATH_CALUDE_vertical_line_slope_angle_l583_58312

-- Define the line x + 2 = 0
def vertical_line (x : ℝ) : Prop := x + 2 = 0

-- Define the slope angle of a line
def slope_angle (line : ℝ → Prop) : ℝ := sorry

-- Theorem: The slope angle of the line x + 2 = 0 is π/2
theorem vertical_line_slope_angle :
  slope_angle vertical_line = π / 2 := by sorry

end NUMINAMATH_CALUDE_vertical_line_slope_angle_l583_58312


namespace NUMINAMATH_CALUDE_square_difference_minus_product_l583_58379

theorem square_difference_minus_product (a b : ℝ) : (a - b)^2 - b * (b - 2*a) = a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_minus_product_l583_58379


namespace NUMINAMATH_CALUDE_binomial_expansion_simplification_l583_58352

theorem binomial_expansion_simplification (x : ℝ) : 
  (2*x+1)^5 - 5*(2*x+1)^4 + 10*(2*x+1)^3 - 10*(2*x+1)^2 + 5*(2*x+1) - 1 = 32*x^5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_simplification_l583_58352


namespace NUMINAMATH_CALUDE_fish_catching_ratio_l583_58324

/-- The number of fish Blaine caught -/
def blaine_fish : ℕ := 5

/-- The total number of fish caught by Keith and Blaine -/
def total_fish : ℕ := 15

/-- The number of fish Keith caught -/
def keith_fish : ℕ := total_fish - blaine_fish

/-- The ratio of fish Keith caught to fish Blaine caught -/
def fish_ratio : ℚ := keith_fish / blaine_fish

theorem fish_catching_ratio :
  fish_ratio = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_fish_catching_ratio_l583_58324


namespace NUMINAMATH_CALUDE_valid_paths_count_l583_58323

/-- Represents the grid dimensions -/
structure GridDimensions where
  width : Nat
  height : Nat

/-- Represents a forbidden vertical segment -/
structure ForbiddenSegment where
  x : Nat
  y_start : Nat
  y_end : Nat

/-- Calculates the number of valid paths on the grid -/
def countValidPaths (grid : GridDimensions) (forbidden : List ForbiddenSegment) : Nat :=
  sorry

/-- The main theorem stating the number of valid paths -/
theorem valid_paths_count :
  let grid := GridDimensions.mk 10 4
  let forbidden := [
    ForbiddenSegment.mk 5 1 3,
    ForbiddenSegment.mk 6 1 3
  ]
  countValidPaths grid forbidden = 329 := by
  sorry

end NUMINAMATH_CALUDE_valid_paths_count_l583_58323


namespace NUMINAMATH_CALUDE_class_size_proof_l583_58376

theorem class_size_proof (total : ℕ) 
  (h1 : (3 : ℚ) / 5 * total + (1 : ℚ) / 5 * total + 10 = total) : total = 50 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l583_58376


namespace NUMINAMATH_CALUDE_max_abs_sum_l583_58385

theorem max_abs_sum (a b c : ℝ) : 
  (∀ x : ℝ, |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) → 
  |a| + |b| + |c| ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_sum_l583_58385


namespace NUMINAMATH_CALUDE_sum_reciprocals_l583_58305

theorem sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a / b + b / c + c / a + b / a + c / b + a / c = 9) :
  a / b + b / c + c / a = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l583_58305


namespace NUMINAMATH_CALUDE_cookie_cutter_sides_l583_58359

/-- The number of sides on a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides on a square -/
def square_sides : ℕ := 4

/-- The number of sides on a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of triangle-shaped cookie cutters -/
def num_triangles : ℕ := 6

/-- The number of square-shaped cookie cutters -/
def num_squares : ℕ := 4

/-- The number of hexagon-shaped cookie cutters -/
def num_hexagons : ℕ := 2

/-- The total number of sides on all cookie cutters -/
def total_sides : ℕ := num_triangles * triangle_sides + num_squares * square_sides + num_hexagons * hexagon_sides

theorem cookie_cutter_sides : total_sides = 46 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cutter_sides_l583_58359


namespace NUMINAMATH_CALUDE_selection_methods_count_l583_58394

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of male athletes. -/
def num_males : ℕ := 4

/-- The number of female athletes. -/
def num_females : ℕ := 5

/-- The total number of athletes to be selected. -/
def num_selected : ℕ := 3

theorem selection_methods_count :
  (choose num_males 2 * choose num_females 1) + (choose num_males 1 * choose num_females 2) = 70 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l583_58394


namespace NUMINAMATH_CALUDE_complex_number_location_l583_58360

theorem complex_number_location :
  let z : ℂ := (1 : ℂ) / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l583_58360


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l583_58336

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem f_max_min_on_interval :
  let a := 1
  let b := 3
  ∃ (x_max x_min : ℝ), a ≤ x_max ∧ x_max ≤ b ∧ a ≤ x_min ∧ x_min ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    f x_max = 4 ∧ x_max = 3 ∧ f x_min = 0 ∧ x_min = 2 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l583_58336


namespace NUMINAMATH_CALUDE_circle_equation_from_center_and_chord_l583_58344

/-- The equation of a circle given its center and a chord. -/
theorem circle_equation_from_center_and_chord 
  (center_x center_y : ℝ) 
  (line1 : ℝ → ℝ → ℝ) (line2 : ℝ → ℝ → ℝ) (line3 : ℝ → ℝ → ℝ)
  (h1 : line1 center_x center_y = 0)
  (h2 : line2 center_x center_y = 0)
  (h3 : ∃ (A B : ℝ × ℝ), line3 A.1 A.2 = 0 ∧ line3 B.1 B.2 = 0)
  (h4 : ∀ (A B : ℝ × ℝ), line3 A.1 A.2 = 0 → line3 B.1 B.2 = 0 → 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36)
  (h5 : line1 x y = x - y - 1)
  (h6 : line2 x y = 2*x - y - 1)
  (h7 : line3 x y = 3*x + 4*y - 11) :
  ∀ (x y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_center_and_chord_l583_58344


namespace NUMINAMATH_CALUDE_sum_of_squares_of_consecutive_even_numbers_l583_58374

theorem sum_of_squares_of_consecutive_even_numbers : 
  ∃ (a b c d : ℕ), 
    (∃ (n : ℕ), a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6) ∧ 
    a + b + c + d = 36 → 
    a^2 + b^2 + c^2 + d^2 = 344 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_consecutive_even_numbers_l583_58374


namespace NUMINAMATH_CALUDE_carpet_cost_l583_58318

/-- The cost of carpeting a room with given dimensions and carpet specifications. -/
theorem carpet_cost (room_length room_width carpet_width carpet_cost : ℝ) :
  room_length = 13 ∧
  room_width = 9 ∧
  carpet_width = 0.75 ∧
  carpet_cost = 12 →
  room_length * room_width * carpet_cost = 1404 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cost_l583_58318


namespace NUMINAMATH_CALUDE_binomial_100_97_l583_58328

theorem binomial_100_97 : Nat.choose 100 97 = 161700 := by
  sorry

end NUMINAMATH_CALUDE_binomial_100_97_l583_58328


namespace NUMINAMATH_CALUDE_smallest_number_proof_smallest_number_l583_58397

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_proof (N : ℕ) (x : ℕ) : Prop :=
  N - x = 746 ∧
  is_divisible_by (N - x) 8 ∧
  is_divisible_by (N - x) 14 ∧
  is_divisible_by (N - x) 26 ∧
  is_divisible_by (N - x) 28 ∧
  ∀ M : ℕ, M < N → ¬(∃ y : ℕ, smallest_number_proof M y)

theorem smallest_number : smallest_number_proof 1474 728 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_proof_smallest_number_l583_58397


namespace NUMINAMATH_CALUDE_not_right_triangle_11_12_15_l583_58369

/-- A function that checks if three numbers can form a right triangle -/
def isRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- Theorem stating that 11, 12, and 15 cannot form a right triangle -/
theorem not_right_triangle_11_12_15 : ¬ isRightTriangle 11 12 15 := by
  sorry

#check not_right_triangle_11_12_15

end NUMINAMATH_CALUDE_not_right_triangle_11_12_15_l583_58369


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l583_58373

theorem exam_pass_percentage 
  (failed_hindi : Real) 
  (failed_english : Real) 
  (failed_both : Real) 
  (h1 : failed_hindi = 34)
  (h2 : failed_english = 44)
  (h3 : failed_both = 22) :
  100 - (failed_hindi + failed_english - failed_both) = 44 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_percentage_l583_58373


namespace NUMINAMATH_CALUDE_weighted_average_salary_l583_58395

/-- Represents the categories of employees in the departmental store -/
inductive EmployeeCategory
  | Manager
  | Associate
  | LeadCashier
  | SalesRepresentative

/-- Returns the number of employees for a given category -/
def employeeCount (category : EmployeeCategory) : Nat :=
  match category with
  | .Manager => 9
  | .Associate => 18
  | .LeadCashier => 6
  | .SalesRepresentative => 45

/-- Returns the average salary for a given category -/
def averageSalary (category : EmployeeCategory) : Nat :=
  match category with
  | .Manager => 4500
  | .Associate => 3500
  | .LeadCashier => 3000
  | .SalesRepresentative => 2500

/-- Calculates the total salary for all employees -/
def totalSalary : Nat :=
  (employeeCount .Manager * averageSalary .Manager) +
  (employeeCount .Associate * averageSalary .Associate) +
  (employeeCount .LeadCashier * averageSalary .LeadCashier) +
  (employeeCount .SalesRepresentative * averageSalary .SalesRepresentative)

/-- Calculates the total number of employees -/
def totalEmployees : Nat :=
  employeeCount .Manager +
  employeeCount .Associate +
  employeeCount .LeadCashier +
  employeeCount .SalesRepresentative

/-- Theorem stating that the weighted average salary is $3000 -/
theorem weighted_average_salary :
  totalSalary / totalEmployees = 3000 := by
  sorry


end NUMINAMATH_CALUDE_weighted_average_salary_l583_58395


namespace NUMINAMATH_CALUDE_tea_mixture_price_l583_58319

/-- Given two types of tea with different prices per kg, calculate the price per kg of their mixture when mixed in equal quantities. -/
theorem tea_mixture_price (price_a price_b : ℚ) (h1 : price_a = 65) (h2 : price_b = 70) :
  (price_a + price_b) / 2 = 67.5 := by
  sorry

#check tea_mixture_price

end NUMINAMATH_CALUDE_tea_mixture_price_l583_58319


namespace NUMINAMATH_CALUDE_digit_sum_in_t_shape_l583_58356

theorem digit_sum_in_t_shape : 
  ∀ (a b c d e f g : ℕ),
  a ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  b ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  c ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  d ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  e ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  f ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g →
  a + b + c = 23 →
  d + e + f + g = 12 →
  b = e →
  a + b + c + d + f + g = 29 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_in_t_shape_l583_58356


namespace NUMINAMATH_CALUDE_completing_square_transform_l583_58370

theorem completing_square_transform (x : ℝ) : 
  (x^2 - 2*x = 9) ↔ ((x - 1)^2 = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_completing_square_transform_l583_58370


namespace NUMINAMATH_CALUDE_secret_spreading_day_l583_58396

/-- The number of new students who learn the secret on day n -/
def new_students (n : ℕ) : ℕ := 3^n

/-- The total number of students who know the secret after n days -/
def total_students (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when 3280 students know the secret -/
theorem secret_spreading_day : 
  ∃ n : ℕ, total_students n = 3280 ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_secret_spreading_day_l583_58396


namespace NUMINAMATH_CALUDE_y_is_odd_square_l583_58399

def x : ℕ → ℤ
| 0 => 0
| 1 => 1
| (n + 2) => 3 * x (n + 1) - 2 * x n

def y (n : ℕ) : ℤ := x n ^ 2 + 2 ^ (n + 2)

theorem y_is_odd_square (n : ℕ) (h : n > 0) :
  ∃ k : ℤ, Odd k ∧ y n = k ^ 2 := by sorry

end NUMINAMATH_CALUDE_y_is_odd_square_l583_58399


namespace NUMINAMATH_CALUDE_unique_solution_triple_l583_58347

theorem unique_solution_triple (x y z : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x^2 - y = (z - 1)^2 ∧
  y^2 - z = (x - 1)^2 ∧
  z^2 - x = (y - 1)^2 →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_triple_l583_58347


namespace NUMINAMATH_CALUDE_point_transformation_l583_58355

def rotate_180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (c d : ℝ) :
  let (x₁, y₁) := rotate_180 c d 2 3
  let (x₂, y₂) := reflect_y_eq_x x₁ y₁
  (x₂ = 2 ∧ y₂ = -1) → d - c = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l583_58355
