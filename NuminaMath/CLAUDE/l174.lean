import Mathlib

namespace NUMINAMATH_CALUDE_total_plants_l174_17410

def garden_problem (basil oregano thyme rosemary : ℕ) : Prop :=
  oregano = 2 * basil + 2 ∧
  thyme = 3 * basil - 3 ∧
  rosemary = (basil + thyme) / 2 ∧
  basil = 5 ∧
  basil + oregano + thyme + rosemary ≤ 50

theorem total_plants (basil oregano thyme rosemary : ℕ) :
  garden_problem basil oregano thyme rosemary →
  basil + oregano + thyme + rosemary = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_total_plants_l174_17410


namespace NUMINAMATH_CALUDE_peach_difference_l174_17479

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 14

/-- The number of peaches Jill has -/
def jill_peaches : ℕ := 5

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - 6

/-- Jake has more peaches than Jill -/
axiom jake_more_than_jill : jake_peaches > jill_peaches

theorem peach_difference : jake_peaches - jill_peaches = 3 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l174_17479


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l174_17463

theorem arithmetic_mean_problem (a₁ a₂ a₃ a₄ a₅ a₆ A : ℝ) 
  (h1 : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = A)
  (h2 : (a₁ + a₂ + a₃ + a₄) / 4 = A + 10)
  (h3 : (a₃ + a₄ + a₅ + a₆) / 4 = A - 7) :
  (a₁ + a₂ + a₅ + a₆) / 4 = A - 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l174_17463


namespace NUMINAMATH_CALUDE_roots_always_real_l174_17468

/-- Given real numbers a, b, and c, the discriminant of the quadratic equation
    resulting from 1/(x+a) + 1/(x+b) + 1/(x+c) = 3/x is non-negative. -/
theorem roots_always_real (a b c : ℝ) : 
  2 * (a^2 * (b - c)^2 + b^2 * (c - a)^2 + c^2 * (a - b)^2) ≥ 0 := by
  sorry

#check roots_always_real

end NUMINAMATH_CALUDE_roots_always_real_l174_17468


namespace NUMINAMATH_CALUDE_largest_digit_sum_l174_17452

def is_digit (n : ℕ) : Prop := n < 10

theorem largest_digit_sum (a b c z : ℕ) : 
  is_digit a → is_digit b → is_digit c →
  (100 * a + 10 * b + c : ℚ) / 1000 = 1 / z →
  0 < z → z ≤ 15 →
  ∀ a' b' c' z',
    is_digit a' → is_digit b' → is_digit c' →
    (100 * a' + 10 * b' + c' : ℚ) / 1000 = 1 / z' →
    0 < z' → z' ≤ 15 →
    a + b + c ≥ a' + b' + c' →
  a + b + c = 8 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l174_17452


namespace NUMINAMATH_CALUDE_louisa_average_speed_l174_17498

/-- Proves that given the conditions of Louisa's travel, her average speed was 50 miles per hour -/
theorem louisa_average_speed :
  ∀ (v : ℝ), 
    v > 0 →
    350 / v - 200 / v = 3 →
    v = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_louisa_average_speed_l174_17498


namespace NUMINAMATH_CALUDE_not_right_triangle_l174_17444

theorem not_right_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 3 * C) 
  (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l174_17444


namespace NUMINAMATH_CALUDE_notebook_cost_is_50_l174_17475

def mean_expenditure : ℝ := 500
def num_days : ℕ := 7
def other_days_expenditure : List ℝ := [450, 600, 400, 500, 550, 300]
def pen_cost : ℝ := 30
def earphone_cost : ℝ := 620

def total_week_expenditure : ℝ := mean_expenditure * num_days
def other_days_total : ℝ := other_days_expenditure.sum
def friday_expenditure : ℝ := total_week_expenditure - other_days_total

theorem notebook_cost_is_50 :
  friday_expenditure - (pen_cost + earphone_cost) = 50 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_is_50_l174_17475


namespace NUMINAMATH_CALUDE_euler_formula_second_quadrant_l174_17453

/-- Prove that e^(i(2π/3)) lies in the second quadrant of the complex plane -/
theorem euler_formula_second_quadrant : 
  let z : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 3))
  z.re < 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_euler_formula_second_quadrant_l174_17453


namespace NUMINAMATH_CALUDE_graham_younger_than_mark_l174_17487

/-- Represents a person with a birth year and month -/
structure Person where
  birthYear : ℕ
  birthMonth : ℕ
  deriving Repr

def currentYear : ℕ := 2021
def currentMonth : ℕ := 2

def Mark : Person := { birthYear := 1976, birthMonth := 1 }

def JaniceAge : ℕ := 21

/-- Calculates the age of a person in years -/
def age (p : Person) : ℕ :=
  if currentMonth >= p.birthMonth then
    currentYear - p.birthYear
  else
    currentYear - p.birthYear - 1

/-- Calculates Graham's age based on Janice's age -/
def GrahamAge : ℕ := 2 * JaniceAge

theorem graham_younger_than_mark :
  age Mark - GrahamAge = 3 := by
  sorry

end NUMINAMATH_CALUDE_graham_younger_than_mark_l174_17487


namespace NUMINAMATH_CALUDE_sqrt_12_sqrt_2_div_sqrt_3_minus_2sin45_equals_1_l174_17455

theorem sqrt_12_sqrt_2_div_sqrt_3_minus_2sin45_equals_1 :
  let sqrt_12 := 2 * Real.sqrt 3
  let sin_45 := Real.sqrt 2 / 2
  (sqrt_12 * Real.sqrt 2) / Real.sqrt 3 - 2 * sin_45 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_sqrt_2_div_sqrt_3_minus_2sin45_equals_1_l174_17455


namespace NUMINAMATH_CALUDE_line_slope_angle_l174_17465

theorem line_slope_angle : 
  let x : ℝ → ℝ := λ t => 3 + t * Real.sin (π / 6)
  let y : ℝ → ℝ := λ t => -t * Real.cos (π / 6)
  (∃ m : ℝ, ∀ t₁ t₂ : ℝ, t₁ ≠ t₂ → 
    (y t₂ - y t₁) / (x t₂ - x t₁) = m ∧ 
    Real.arctan m = 2 * π / 3) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_angle_l174_17465


namespace NUMINAMATH_CALUDE_luncheon_invitees_l174_17415

theorem luncheon_invitees (no_shows : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) :
  no_shows = 10 →
  table_capacity = 7 →
  tables_needed = 2 →
  no_shows + (tables_needed * table_capacity) = 24 := by
sorry

end NUMINAMATH_CALUDE_luncheon_invitees_l174_17415


namespace NUMINAMATH_CALUDE_least_three_digit_product_12_l174_17467

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 126 ≤ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_product_12_l174_17467


namespace NUMINAMATH_CALUDE_initial_workers_count_l174_17448

/-- The time it takes one person to complete the task -/
def total_time : ℕ := 40

/-- The time the initial group works -/
def initial_work_time : ℕ := 4

/-- The number of additional people joining -/
def additional_workers : ℕ := 2

/-- The time the expanded group works -/
def expanded_work_time : ℕ := 8

/-- Proves that the initial number of workers is 2 -/
theorem initial_workers_count : 
  ∃ (x : ℕ), 
    (initial_work_time * x + expanded_work_time * (x + additional_workers)) / total_time = 1 ∧ 
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_workers_count_l174_17448


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l174_17481

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + y^2 = 4*x*y) : 
  1/x + 1/y = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l174_17481


namespace NUMINAMATH_CALUDE_min_F_beautiful_pair_l174_17486

def is_beautiful_pair (p q : ℕ) : Prop :=
  ∃ x y : ℕ,
    1 ≤ x ∧ x ≤ 4 ∧
    1 ≤ y ∧ y ≤ 5 ∧
    p = 21 * x + y ∧
    q = 52 + y ∧
    (10 * y + x + 6 * y) % 13 = 0

def F (p q : ℕ) : ℕ :=
  let tens_p := p / 10
  let units_p := p % 10
  let tens_q := q / 10
  let units_q := q % 10
  10 * tens_p + units_q +
  10 * tens_p + units_p +
  10 * units_p + units_q +
  10 * units_p + tens_q

theorem min_F_beautiful_pair :
  ∀ p q : ℕ,
    is_beautiful_pair p q →
    F p q ≥ 156 :=
sorry

end NUMINAMATH_CALUDE_min_F_beautiful_pair_l174_17486


namespace NUMINAMATH_CALUDE_square_of_negative_three_x_squared_y_l174_17437

theorem square_of_negative_three_x_squared_y (x y : ℝ) :
  (-3 * x^2 * y)^2 = 9 * x^4 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_three_x_squared_y_l174_17437


namespace NUMINAMATH_CALUDE_positive_real_inequality_l174_17476

theorem positive_real_inequality (x : ℝ) (hx : x > 0) :
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l174_17476


namespace NUMINAMATH_CALUDE_greatest_power_of_four_dividing_16_factorial_l174_17431

theorem greatest_power_of_four_dividing_16_factorial :
  (∃ k : ℕ+, k.val = 7 ∧ 
   ∀ m : ℕ+, (4 ^ m.val ∣ Nat.factorial 16) → m.val ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_four_dividing_16_factorial_l174_17431


namespace NUMINAMATH_CALUDE_max_value_theorem_l174_17458

theorem max_value_theorem (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_eq : x^2 - 3*x*y + 4*y^2 = 9) :
  x^2 + 3*x*y + 4*y^2 ≤ 63 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 - 3*x₀*y₀ + 4*y₀^2 = 9 ∧ x₀^2 + 3*x₀*y₀ + 4*y₀^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l174_17458


namespace NUMINAMATH_CALUDE_track_length_is_360_l174_17464

/-- Represents a circular running track with two runners -/
structure RunningTrack where
  length : ℝ
  sally_first_meeting : ℝ
  john_second_meeting : ℝ

/-- Theorem stating that given the conditions, the track length is 360 meters -/
theorem track_length_is_360 (track : RunningTrack) 
  (h1 : track.sally_first_meeting = 90)
  (h2 : track.john_second_meeting = 200)
  (h3 : track.sally_first_meeting > 0)
  (h4 : track.john_second_meeting > 0)
  (h5 : track.length > 0) :
  track.length = 360 := by
  sorry

#check track_length_is_360

end NUMINAMATH_CALUDE_track_length_is_360_l174_17464


namespace NUMINAMATH_CALUDE_minute_hand_rotation_1_to_3_20_l174_17499

/-- The number of radians a clock's minute hand turns through in a given time interval -/
def minute_hand_rotation (start_hour start_minute end_hour end_minute : ℕ) : ℝ :=
  sorry

/-- The number of radians a clock's minute hand turns through from 1:00 to 3:20 -/
theorem minute_hand_rotation_1_to_3_20 :
  minute_hand_rotation 1 0 3 20 = -14/3 * π :=
sorry

end NUMINAMATH_CALUDE_minute_hand_rotation_1_to_3_20_l174_17499


namespace NUMINAMATH_CALUDE_equation_solution_l174_17488

theorem equation_solution : ∃ x : ℝ, (24 - 5 = 3 + x) ∧ (x = 16) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l174_17488


namespace NUMINAMATH_CALUDE_sum_of_fractions_l174_17403

theorem sum_of_fractions : 
  (5 : ℚ) / 13 + (9 : ℚ) / 11 = (172 : ℚ) / 143 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l174_17403


namespace NUMINAMATH_CALUDE_kite_to_square_area_ratio_l174_17484

/-- The ratio of the area of a kite formed by the diagonals of four central
    smaller squares to the area of a large square --/
theorem kite_to_square_area_ratio :
  let large_side : ℝ := 60
  let small_side : ℝ := 10
  let large_area : ℝ := large_side ^ 2
  let kite_diagonal1 : ℝ := 2 * small_side
  let kite_diagonal2 : ℝ := 2 * small_side * Real.sqrt 2
  let kite_area : ℝ := (1 / 2) * kite_diagonal1 * kite_diagonal2
  kite_area / large_area = 100 * Real.sqrt 2 / 3600 := by
sorry

end NUMINAMATH_CALUDE_kite_to_square_area_ratio_l174_17484


namespace NUMINAMATH_CALUDE_ellipse_equivalence_l174_17401

/-- Given ellipse equation -/
def given_ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36

/-- New ellipse equation -/
def new_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

/-- Foci of an ellipse -/
def has_same_foci (e1 e2 : (ℝ → ℝ → Prop)) : Prop := sorry

theorem ellipse_equivalence :
  has_same_foci given_ellipse new_ellipse ∧ new_ellipse (-3) 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_equivalence_l174_17401


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l174_17472

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 37 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 37) = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l174_17472


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l174_17492

-- Define the concept of opposite
def opposite (x : ℤ) : ℤ := -x

-- Theorem statement
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l174_17492


namespace NUMINAMATH_CALUDE_line_segment_both_symmetric_l174_17470

-- Define the shapes
inductive Shape
| EquilateralTriangle
| IsoscelesTriangle
| Parallelogram
| LineSegment

-- Define symmetry properties
def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => true
  | Shape.LineSegment => true
  | _ => false

def isAxiallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.IsoscelesTriangle => true
  | Shape.LineSegment => true
  | _ => false

-- Theorem statement
theorem line_segment_both_symmetric :
  ∀ s : Shape, (isCentrallySymmetric s ∧ isAxiallySymmetric s) ↔ s = Shape.LineSegment :=
by sorry

end NUMINAMATH_CALUDE_line_segment_both_symmetric_l174_17470


namespace NUMINAMATH_CALUDE_x_fourth_equals_one_l174_17496

theorem x_fourth_equals_one (x : ℝ) 
  (h : Real.sqrt (1 - x^2) + Real.sqrt (1 + x^2) = Real.sqrt 2) : 
  x^4 = 1 := by
sorry

end NUMINAMATH_CALUDE_x_fourth_equals_one_l174_17496


namespace NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l174_17405

/-- The volume of a cylinder formed by rotating a square about its horizontal line of symmetry. -/
theorem volume_cylinder_from_square_rotation (side_length : ℝ) (h_positive : side_length > 0) :
  let radius : ℝ := side_length / 2
  let height : ℝ := side_length
  let volume : ℝ := π * radius ^ 2 * height
  side_length = 10 → volume = 250 * π := by sorry

end NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l174_17405


namespace NUMINAMATH_CALUDE_garden_furniture_cost_l174_17439

/-- The combined cost of a garden table and bench, given their price relationship -/
theorem garden_furniture_cost (bench_price : ℕ) (table_price : ℕ) : 
  bench_price = 150 → 
  table_price = 2 * bench_price → 
  bench_price + table_price = 450 := by
  sorry

end NUMINAMATH_CALUDE_garden_furniture_cost_l174_17439


namespace NUMINAMATH_CALUDE_triangle_area_l174_17434

/-- Given a triangle with perimeter 28 and inradius 2.5, prove that its area is 35 -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
    (h1 : perimeter = 28) 
    (h2 : inradius = 2.5) 
    (h3 : area = inradius * (perimeter / 2)) : area = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l174_17434


namespace NUMINAMATH_CALUDE_find_number_l174_17446

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_find_number_l174_17446


namespace NUMINAMATH_CALUDE_third_divisor_is_three_l174_17433

def smallest_number : ℕ := 1011
def diminished_number : ℕ := smallest_number - 3

theorem third_divisor_is_three :
  ∃ (x : ℕ), x ≠ 12 ∧ x ≠ 16 ∧ x ≠ 21 ∧ x ≠ 28 ∧
  diminished_number % 12 = 0 ∧
  diminished_number % 16 = 0 ∧
  diminished_number % x = 0 ∧
  diminished_number % 21 = 0 ∧
  diminished_number % 28 = 0 ∧
  x = 3 :=
by sorry

end NUMINAMATH_CALUDE_third_divisor_is_three_l174_17433


namespace NUMINAMATH_CALUDE_hollow_square_students_l174_17418

/-- Represents a hollow square formation of students -/
structure HollowSquare where
  outer_layer : Nat
  inner_layer : Nat

/-- Calculates the total number of students in a hollow square formation -/
def total_students (hs : HollowSquare) : Nat :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that a hollow square with 52 in the outer layer and 28 in the inner layer has 160 students total -/
theorem hollow_square_students :
  let hs : HollowSquare := { outer_layer := 52, inner_layer := 28 }
  total_students hs = 160 := by
  sorry

end NUMINAMATH_CALUDE_hollow_square_students_l174_17418


namespace NUMINAMATH_CALUDE_f_2013_equals_2_l174_17473

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def satisfies_recurrence (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) = f x + 2 * f 2

theorem f_2013_equals_2 (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : satisfies_recurrence f)
  (h3 : f (-1) = 2) :
  f 2013 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2013_equals_2_l174_17473


namespace NUMINAMATH_CALUDE_root_equality_implies_b_equals_four_l174_17483

theorem root_equality_implies_b_equals_four
  (a b c : ℕ)
  (a_gt_one : a > 1)
  (b_gt_one : b > 1)
  (c_gt_one : c > 1)
  (h : ∀ (N : ℝ), N ≠ 1 → N^(1/a + 1/(a*b) + 1/(a*b*c) + 1/(a*b*c^2)) = N^(49/60)) :
  b = 4 :=
sorry

end NUMINAMATH_CALUDE_root_equality_implies_b_equals_four_l174_17483


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l174_17491

theorem least_k_for_inequality (k : ℤ) : 
  (∀ m : ℤ, m < k → (0.00010101 * (10 : ℝ)^m ≤ 100)) ∧ 
  (0.00010101 * (10 : ℝ)^k > 100) → 
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l174_17491


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l174_17466

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 5 < x)
  (h2 : 7 < x ∧ x < 18)
  (h3 : 2 < x ∧ x < 13)
  (h4 : 9 < x ∧ x < 12)
  (h5 : x + 1 < 13) :
  ∃ (y : ℤ), x < y ∧ (∀ (z : ℤ), x < z → y ≤ z) ∧ y = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l174_17466


namespace NUMINAMATH_CALUDE_cinema_seats_l174_17477

theorem cinema_seats (rows : ℕ) (seats_per_row : ℕ) (h1 : rows = 21) (h2 : seats_per_row = 26) :
  rows * seats_per_row = 546 := by
  sorry

end NUMINAMATH_CALUDE_cinema_seats_l174_17477


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l174_17436

theorem smallest_angle_solution (x : ℝ) : 
  (0 < x) → 
  (∀ y : ℝ, 0 < y → 
    Real.tan (8 * π / 180 * y) = (Real.cos (π / 180 * y) - Real.sin (π / 180 * y)) / (Real.cos (π / 180 * y) + Real.sin (π / 180 * y)) → 
    x ≤ y) → 
  Real.tan (8 * π / 180 * x) = (Real.cos (π / 180 * x) - Real.sin (π / 180 * x)) / (Real.cos (π / 180 * x) + Real.sin (π / 180 * x)) → 
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l174_17436


namespace NUMINAMATH_CALUDE_james_nickels_l174_17489

/-- Represents the number of nickels in James' jar -/
def n : ℕ := sorry

/-- Represents the number of quarters in James' jar -/
def q : ℕ := sorry

/-- The total value in cents -/
def total_cents : ℕ := 685

/-- Theorem stating the number of nickels in James' jar -/
theorem james_nickels : 
  (5 * n + 25 * q = total_cents) ∧ 
  (n = q + 11) → 
  n = 32 := by sorry

end NUMINAMATH_CALUDE_james_nickels_l174_17489


namespace NUMINAMATH_CALUDE_mathematics_partition_ways_l174_17419

/-- Represents the word "MATHEMATICS" -/
def word : String := "MATHEMATICS"

/-- The positions of vowels in the word -/
def vowel_positions : List Nat := [2, 5, 7, 9]

/-- The number of vowels in the word -/
def num_vowels : Nat := vowel_positions.length

/-- A function to calculate the number of partition ways -/
def num_partition_ways : Nat := 4 * 3 * 3

/-- Theorem stating that the number of ways to partition the word "MATHEMATICS" 
    such that each part contains at least one vowel is 36 -/
theorem mathematics_partition_ways :
  num_partition_ways = 36 := by sorry

end NUMINAMATH_CALUDE_mathematics_partition_ways_l174_17419


namespace NUMINAMATH_CALUDE_work_completion_theorem_l174_17494

/-- Represents the number of men originally employed -/
def original_men : ℕ := 17

/-- Represents the number of days originally required to finish the work -/
def original_days : ℕ := 8

/-- Represents the number of additional men who joined -/
def additional_men : ℕ := 10

/-- Represents the number of days saved after additional men joined -/
def days_saved : ℕ := 3

/-- Theorem stating that the given conditions lead to the correct number of original men -/
theorem work_completion_theorem :
  (original_men * original_days = (original_men + additional_men) * (original_days - days_saved)) ∧
  (original_men ≥ 1) ∧
  (∀ m : ℕ, m < original_men →
    m * original_days ≠ (m + additional_men) * (original_days - days_saved)) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l174_17494


namespace NUMINAMATH_CALUDE_correct_calculation_l174_17480

theorem correct_calculation : ∃! x : ℤ, (2 - 3 = x ∧ x = -1) ∧
  ¬((-3)^2 = -9) ∧
  ¬(-3^2 = -6) ∧
  ¬(-3 - (-2) = -5) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l174_17480


namespace NUMINAMATH_CALUDE_smallest_number_properties_l174_17450

/-- The smallest number that is divisible by 18 and 30 and is a perfect square -/
def smallest_number : ℕ := 900

/-- Predicate to check if a number is divisible by both 18 and 30 -/
def divisible_by_18_and_30 (n : ℕ) : Prop := n % 18 = 0 ∧ n % 30 = 0

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem smallest_number_properties :
  divisible_by_18_and_30 smallest_number ∧
  is_perfect_square smallest_number ∧
  ∀ n : ℕ, n < smallest_number → ¬(divisible_by_18_and_30 n ∧ is_perfect_square n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_properties_l174_17450


namespace NUMINAMATH_CALUDE_electric_vehicle_analysis_l174_17438

-- Define the variables
variable (x : ℝ) -- Number of vehicles a skilled worker can install per month
variable (y : ℝ) -- Number of vehicles a new worker can install per month
variable (m : ℝ) -- Average cost per kilometer of the electric vehicle
variable (a : ℝ) -- Annual mileage

-- Define the theorem
theorem electric_vehicle_analysis :
  -- Part 1: Installation capacity
  (2 * x + y = 10 ∧ x + 3 * y = 10) →
  (x = 4 ∧ y = 2) ∧
  -- Part 2: Cost per kilometer
  (200 / m = 4 * (200 / (m + 0.6))) →
  m = 0.2 ∧
  -- Part 3: Annual cost comparison
  (0.2 * a + 6400 < 0.8 * a + 4000) →
  a > 4000 :=
by sorry

end NUMINAMATH_CALUDE_electric_vehicle_analysis_l174_17438


namespace NUMINAMATH_CALUDE_derivative_equals_function_implies_zero_at_two_l174_17430

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem derivative_equals_function_implies_zero_at_two 
  (h : ∀ x, deriv f x = f x) : 
  deriv f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_equals_function_implies_zero_at_two_l174_17430


namespace NUMINAMATH_CALUDE_simplify_expression_l174_17428

theorem simplify_expression (x : ℝ) : (3*x)^5 + (5*x)*(x^4) - 7*x^5 = 241*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l174_17428


namespace NUMINAMATH_CALUDE_sheep_purchase_l174_17420

/-- Calculates the number of sheep Mary needs to buy to have 69 fewer sheep than Bob -/
theorem sheep_purchase (mary_initial : ℕ) (bob_multiplier : ℕ) (bob_additional : ℕ) (target_difference : ℕ) : 
  mary_initial = 300 →
  bob_multiplier = 2 →
  bob_additional = 35 →
  target_difference = 69 →
  (mary_initial + (bob_multiplier * mary_initial + bob_additional - target_difference - mary_initial)) = 566 :=
by sorry

end NUMINAMATH_CALUDE_sheep_purchase_l174_17420


namespace NUMINAMATH_CALUDE_min_value_ab_l174_17443

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) : 
  a * b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = a₀ + b₀ + 3 ∧ a₀ * b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l174_17443


namespace NUMINAMATH_CALUDE_fraction_equality_l174_17445

theorem fraction_equality : (1625^2 - 1618^2) / (1632^2 - 1611^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l174_17445


namespace NUMINAMATH_CALUDE_rook_placement_count_l174_17413

theorem rook_placement_count (n k : ℕ) (h1 : n = 8) (h2 : k = 6) :
  (Nat.choose n k)^2 * Nat.factorial k = 564480 := by
  sorry

end NUMINAMATH_CALUDE_rook_placement_count_l174_17413


namespace NUMINAMATH_CALUDE_boys_running_speed_l174_17412

theorem boys_running_speed (side_length : ℝ) (time : ℝ) (speed : ℝ) : 
  side_length = 55 →
  time = 88 →
  speed = (4 * side_length / time) * 3.6 →
  speed = 9 := by sorry

end NUMINAMATH_CALUDE_boys_running_speed_l174_17412


namespace NUMINAMATH_CALUDE_student_number_proof_l174_17414

theorem student_number_proof :
  ∃! (a b c : ℕ),
    0 < a ∧ a < 10 ∧
    0 ≤ b ∧ b < 10 ∧
    0 ≤ c ∧ c < 10 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    2 * (100 * a + 10 * b + c) = (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b) ∧
    100 * a + 10 * b + c = 198 :=
by sorry

end NUMINAMATH_CALUDE_student_number_proof_l174_17414


namespace NUMINAMATH_CALUDE_cost_of_pens_l174_17497

/-- Given a pack of 150 pens costs $45, prove that the cost of 3600 pens is $1080 -/
theorem cost_of_pens (pack_size : ℕ) (pack_cost : ℝ) (total_pens : ℕ) :
  pack_size = 150 →
  pack_cost = 45 →
  total_pens = 3600 →
  (total_pens : ℝ) * (pack_cost / pack_size) = 1080 := by
sorry

end NUMINAMATH_CALUDE_cost_of_pens_l174_17497


namespace NUMINAMATH_CALUDE_employee_pay_l174_17482

/-- Given two employees X and Y, proves that Y's weekly pay is 150 units -/
theorem employee_pay (total_pay x y : ℝ) : 
  total_pay = x + y → 
  x = 1.2 * y → 
  total_pay = 330 → 
  y = 150 := by sorry

end NUMINAMATH_CALUDE_employee_pay_l174_17482


namespace NUMINAMATH_CALUDE_circle_equation_correct_l174_17474

-- Define the center and radius of the circle
def center : ℝ × ℝ := (-2, 3)
def radius : ℝ := 2

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 3)^2 = 4

-- Theorem stating that the given equation represents the circle with the specified center and radius
theorem circle_equation_correct :
  ∀ x y : ℝ, circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l174_17474


namespace NUMINAMATH_CALUDE_max_salary_theorem_l174_17471

/-- Represents a basketball team in a semipro league. -/
structure BasketballTeam where
  num_players : ℕ
  min_salary : ℕ
  max_total_salary : ℕ

/-- Calculates the maximum possible salary for a single player in a basketball team. -/
def max_single_player_salary (team : BasketballTeam) : ℕ :=
  team.max_total_salary - (team.num_players - 1) * team.min_salary

/-- Theorem stating the maximum possible salary for a single player in the given conditions. -/
theorem max_salary_theorem (team : BasketballTeam) 
    (h1 : team.num_players = 21)
    (h2 : team.min_salary = 20000)
    (h3 : team.max_total_salary = 900000) : 
  max_single_player_salary team = 500000 := by
  sorry

#eval max_single_player_salary { num_players := 21, min_salary := 20000, max_total_salary := 900000 }

end NUMINAMATH_CALUDE_max_salary_theorem_l174_17471


namespace NUMINAMATH_CALUDE_triangle_angle_C_l174_17454

/-- Given a triangle with angle A = 30°, side a = 1, and side b = √2,
    prove that the angle C is either 105° or 15°. -/
theorem triangle_angle_C (A : Real) (a b : Real) :
  A = 30 * π / 180 →
  a = 1 →
  b = Real.sqrt 2 →
  ∃ (C : Real), (C = 105 * π / 180 ∨ C = 15 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l174_17454


namespace NUMINAMATH_CALUDE_gcd_1729_867_l174_17409

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_867_l174_17409


namespace NUMINAMATH_CALUDE_four_valid_dimensions_l174_17408

/-- The number of valid floor dimensions -/
def valid_floor_dimensions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    a ≥ 5 ∧ b > a ∧ (a - 6) * (b - 6) = 36
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- The theorem stating that there are exactly 4 valid floor dimensions -/
theorem four_valid_dimensions : valid_floor_dimensions = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_valid_dimensions_l174_17408


namespace NUMINAMATH_CALUDE_absolute_value_integral_l174_17404

theorem absolute_value_integral : ∫ x in (-1)..2, |x| = 5/2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_integral_l174_17404


namespace NUMINAMATH_CALUDE_max_perimeter_triangle_l174_17493

/-- Given a triangle with two sides of length 7 and 9 units, and the third side of length x units
    (where x is an integer), the maximum perimeter of the triangle is 31 units. -/
theorem max_perimeter_triangle (x : ℤ) : 
  (7 : ℝ) + 9 > x ∧ (7 : ℝ) + x > 9 ∧ (9 : ℝ) + x > 7 → 
  x > 0 →
  (∀ y : ℤ, ((7 : ℝ) + 9 > y ∧ (7 : ℝ) + y > 9 ∧ (9 : ℝ) + y > 7 → y ≤ x)) →
  (7 : ℝ) + 9 + x = 31 := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_triangle_l174_17493


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l174_17407

theorem sum_of_specific_numbers : 3 + 33 + 333 + 33.3 = 402.3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l174_17407


namespace NUMINAMATH_CALUDE_total_unique_plants_l174_17442

-- Define the sets X, Y, Z as finite sets
variable (X Y Z : Finset ℕ)

-- Define the cardinalities of the sets and their intersections
axiom card_X : X.card = 700
axiom card_Y : Y.card = 600
axiom card_Z : Z.card = 400
axiom card_X_inter_Y : (X ∩ Y).card = 100
axiom card_X_inter_Z : (X ∩ Z).card = 200
axiom card_Y_inter_Z : (Y ∩ Z).card = 50
axiom card_X_inter_Y_inter_Z : (X ∩ Y ∩ Z).card = 25

-- Theorem statement
theorem total_unique_plants : (X ∪ Y ∪ Z).card = 1375 :=
sorry

end NUMINAMATH_CALUDE_total_unique_plants_l174_17442


namespace NUMINAMATH_CALUDE_trigonometric_identities_l174_17462

theorem trigonometric_identities (α : Real) (h : Real.tan α = 2) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -1/3 ∧
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l174_17462


namespace NUMINAMATH_CALUDE_line_inclination_l174_17460

/-- Given a line with equation y = √3x + 2, its angle of inclination is π/3 -/
theorem line_inclination (x y : ℝ) :
  y = Real.sqrt 3 * x + 2 → 
  ∃ θ : ℝ, θ ∈ Set.Icc 0 π ∧ θ = π / 3 ∧ Real.tan θ = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_line_inclination_l174_17460


namespace NUMINAMATH_CALUDE_jane_hiking_distance_l174_17417

/-- The distance between two points given a specific path --/
theorem jane_hiking_distance (A B D : ℝ × ℝ) : 
  (A.1 = B.1 ∧ A.2 + 3 = B.2) →  -- AB is 3 units northward
  (Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 8) →  -- BD is 8 units long
  (D.1 - B.1 = D.2 - B.2) →  -- 45 degree angle (isosceles right triangle)
  Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = Real.sqrt (73 + 24 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_jane_hiking_distance_l174_17417


namespace NUMINAMATH_CALUDE_train_speed_l174_17457

/-- The speed of a train given the time to pass an electric pole and a platform -/
theorem train_speed (pole_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  pole_time = 15 →
  platform_length = 380 →
  platform_time = 52.99696024318054 →
  ∃ (speed : ℝ), abs (speed - 36.0037908) < 0.0000001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l174_17457


namespace NUMINAMATH_CALUDE_advanced_vowel_soup_sequences_l174_17425

/-- The number of vowels in the alphabet soup -/
def num_vowels : ℕ := 5

/-- The number of consonants in the alphabet soup -/
def num_consonants : ℕ := 2

/-- The number of times each vowel appears -/
def vowel_occurrences : ℕ := 7

/-- The number of times each consonant appears -/
def consonant_occurrences : ℕ := 3

/-- The length of each sequence -/
def sequence_length : ℕ := 7

/-- The number of valid sequences in the Advanced Vowel Soup -/
theorem advanced_vowel_soup_sequences : 
  (num_vowels + num_consonants)^sequence_length - 
  num_vowels^sequence_length - 
  num_consonants^sequence_length = 745290 := by
  sorry

end NUMINAMATH_CALUDE_advanced_vowel_soup_sequences_l174_17425


namespace NUMINAMATH_CALUDE_cubes_not_touching_foil_l174_17406

/-- Represents a rectangular prism with inner and outer dimensions -/
structure RectangularPrism where
  inner_length : ℕ
  inner_width : ℕ
  inner_height : ℕ
  outer_width : ℕ

/-- Creates a RectangularPrism with the given constraints -/
def create_prism (outer_width : ℕ) : RectangularPrism :=
  { inner_length := (outer_width - 2) / 2,
    inner_width := outer_width - 2,
    inner_height := (outer_width - 2) / 2,
    outer_width := outer_width }

/-- Calculates the number of cubes not touching tin foil -/
def inner_cubes (prism : RectangularPrism) : ℕ :=
  prism.inner_length * prism.inner_width * prism.inner_height

/-- Theorem stating the number of cubes not touching tin foil -/
theorem cubes_not_touching_foil :
  inner_cubes (create_prism 10) = 128 := by
  sorry

#eval inner_cubes (create_prism 10)

end NUMINAMATH_CALUDE_cubes_not_touching_foil_l174_17406


namespace NUMINAMATH_CALUDE_pencil_packing_problem_l174_17451

theorem pencil_packing_problem :
  ∃ (a k m : ℤ),
    200 ≤ a ∧ a ≤ 300 ∧
    a % 10 = 7 ∧
    a % 12 = 9 ∧
    a = 60 * m + 57 ∧
    (a = 237 ∨ a = 297) :=
by sorry

end NUMINAMATH_CALUDE_pencil_packing_problem_l174_17451


namespace NUMINAMATH_CALUDE_shaded_area_recursive_square_division_l174_17449

theorem shaded_area_recursive_square_division (r : ℝ) (h1 : r = 1/16) (h2 : 0 < r) (h3 : r < 1) :
  (1/4) * (1 / (1 - r)) = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_recursive_square_division_l174_17449


namespace NUMINAMATH_CALUDE_inequality_proof_l174_17495

theorem inequality_proof (x y a b : ℝ) (hx : x > 0) (hy : y > 0) :
  ((a * x + b * y) / (x + y))^2 ≤ (a^2 * x + b^2 * y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l174_17495


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l174_17432

theorem sum_of_reciprocals_positive (a b c d : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1) (hd : |d| > 1)
  (h_eq : a * b * (c + d) + d * c * (a + b) + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l174_17432


namespace NUMINAMATH_CALUDE_factor_count_l174_17435

def n : ℕ := 2^2 * 3^2 * 7^2

def is_factor (d : ℕ) : Prop := d ∣ n

def is_even (d : ℕ) : Prop := d % 2 = 0

def is_odd (d : ℕ) : Prop := d % 2 = 1

theorem factor_count :
  (∃ (even_factors : Finset ℕ) (odd_factors : Finset ℕ),
    (∀ d ∈ even_factors, is_factor d ∧ is_even d) ∧
    (∀ d ∈ odd_factors, is_factor d ∧ is_odd d) ∧
    (Finset.card even_factors = 18) ∧
    (Finset.card odd_factors = 9) ∧
    (∀ d : ℕ, is_factor d → (d ∈ even_factors ∨ d ∈ odd_factors))) :=
by sorry

end NUMINAMATH_CALUDE_factor_count_l174_17435


namespace NUMINAMATH_CALUDE_solve_equation_l174_17441

theorem solve_equation (x : ℚ) : 15 * x = 165 ↔ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l174_17441


namespace NUMINAMATH_CALUDE_probability_of_selection_l174_17456

def total_students : ℕ := 10
def students_per_teacher : ℕ := 4

theorem probability_of_selection (total_students : ℕ) (students_per_teacher : ℕ) :
  total_students = 10 → students_per_teacher = 4 →
  (1 : ℚ) - (1 - students_per_teacher / total_students) ^ 2 = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_l174_17456


namespace NUMINAMATH_CALUDE_tan_beta_value_l174_17424

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l174_17424


namespace NUMINAMATH_CALUDE_max_value_function_l174_17440

theorem max_value_function (a : ℝ) (h : a > 0) :
  ∃ (max : ℝ), ∀ (x : ℝ), x > 0 → a > 2*x → x*(a - 2*x) ≤ max ∧
  ∃ (x₀ : ℝ), x₀ > 0 ∧ a > 2*x₀ ∧ x₀*(a - 2*x₀) = max :=
by sorry

end NUMINAMATH_CALUDE_max_value_function_l174_17440


namespace NUMINAMATH_CALUDE_smallest_perfect_cube_multiplier_l174_17478

def y : ℕ := 2^3^3^4^4^5^5^6^6^7^7^8^8^9

theorem smallest_perfect_cube_multiplier :
  (∃ k : ℕ, k > 0 ∧ ∃ n : ℕ, k * y = n^3) ∧
  (∀ k : ℕ, k > 0 → (∃ n : ℕ, k * y = n^3) → k ≥ 1500) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_cube_multiplier_l174_17478


namespace NUMINAMATH_CALUDE_bowl_game_points_l174_17411

/-- The total points scored by Noa and Phillip in a bowl game. -/
def total_points (noa_points phillip_points : ℕ) : ℕ := noa_points + phillip_points

/-- Theorem stating that given Noa's score and Phillip scoring twice as much,
    the total points scored by Noa and Phillip is 90. -/
theorem bowl_game_points :
  let noa_points : ℕ := 30
  let phillip_points : ℕ := 2 * noa_points
  total_points noa_points phillip_points = 90 := by
  sorry

end NUMINAMATH_CALUDE_bowl_game_points_l174_17411


namespace NUMINAMATH_CALUDE_archer_fish_count_l174_17459

/-- The total number of fish Archer caught in a day -/
def total_fish (first_round : ℕ) (second_round_increase : ℕ) (third_round_percentage : ℕ) : ℕ :=
  let second_round := first_round + second_round_increase
  let third_round := second_round + (third_round_percentage * second_round) / 100
  first_round + second_round + third_round

/-- Theorem stating that Archer caught 60 fish in total -/
theorem archer_fish_count : total_fish 8 12 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_archer_fish_count_l174_17459


namespace NUMINAMATH_CALUDE_power_mod_thousand_l174_17485

theorem power_mod_thousand : 7^27 % 1000 = 543 := by sorry

end NUMINAMATH_CALUDE_power_mod_thousand_l174_17485


namespace NUMINAMATH_CALUDE_f_max_at_neg_two_l174_17423

def f (x : ℝ) : ℝ := x^3 - 12*x

theorem f_max_at_neg_two :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≤ f m :=
sorry

end NUMINAMATH_CALUDE_f_max_at_neg_two_l174_17423


namespace NUMINAMATH_CALUDE_ellipse_from_hyperbola_vertices_l174_17400

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, 
    the equation of the ellipse whose foci are the vertices of the hyperbola 
    is x²/16 + y²/12 = 1 -/
theorem ellipse_from_hyperbola_vertices (x y : ℝ) :
  let hyperbola := (x^2 / 4 - y^2 / 12 = 1)
  let ellipse := (x^2 / 16 + y^2 / 12 = 1)
  let hyperbola_vertex := 2
  let hyperbola_focus := 4
  hyperbola → ellipse := by sorry

end NUMINAMATH_CALUDE_ellipse_from_hyperbola_vertices_l174_17400


namespace NUMINAMATH_CALUDE_total_spent_is_14_l174_17469

/-- The cost of one set of barrettes in dollars -/
def barrette_cost : ℕ := 3

/-- The cost of one comb in dollars -/
def comb_cost : ℕ := 1

/-- The number of barrette sets Kristine buys -/
def kristine_barrettes : ℕ := 1

/-- The number of combs Kristine buys -/
def kristine_combs : ℕ := 1

/-- The number of barrette sets Crystal buys -/
def crystal_barrettes : ℕ := 3

/-- The number of combs Crystal buys -/
def crystal_combs : ℕ := 1

/-- The total amount spent by both Kristine and Crystal -/
def total_spent : ℕ := 
  (kristine_barrettes * barrette_cost + kristine_combs * comb_cost) + 
  (crystal_barrettes * barrette_cost + crystal_combs * comb_cost)

theorem total_spent_is_14 : total_spent = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_14_l174_17469


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l174_17447

/-- Given a point M with polar coordinates (6, 11π/6), 
    the Cartesian coordinates of the point symmetric to M 
    with respect to the y-axis are (-3√3, -3) -/
theorem symmetric_point_coordinates : 
  let r : ℝ := 6
  let θ : ℝ := 11 * π / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (- x, y) = (-3 * Real.sqrt 3, -3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l174_17447


namespace NUMINAMATH_CALUDE_no_extremum_l174_17421

open Real

/-- A function satisfying the given differential equation and initial condition -/
def SolutionFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x > 0 → x * (deriv f x) + f x = exp x / x) ∧ f 1 = exp 1

/-- The main theorem stating that the function has no maximum or minimum -/
theorem no_extremum (f : ℝ → ℝ) (hf : SolutionFunction f) :
    (∀ x, x > 0 → ¬ IsLocalMax f x) ∧ (∀ x, x > 0 → ¬ IsLocalMin f x) := by
  sorry


end NUMINAMATH_CALUDE_no_extremum_l174_17421


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l174_17402

theorem vector_magnitude_proof (a b : ℝ × ℝ × ℝ) :
  a = (1, 1, 0) ∧ b = (-1, 0, 2) →
  ‖(2 • a) - b‖ = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l174_17402


namespace NUMINAMATH_CALUDE_pushup_comparison_l174_17427

theorem pushup_comparison (zachary david emily : ℕ) 
  (h1 : zachary = 51)
  (h2 : david = 44)
  (h3 : emily = 37) :
  zachary = (david + emily) - 30 :=
by sorry

end NUMINAMATH_CALUDE_pushup_comparison_l174_17427


namespace NUMINAMATH_CALUDE_max_rectangle_area_l174_17422

def is_valid_rectangle (l w : ℕ) : Prop :=
  l + w = 20 ∧ l ≥ w + 3

def rectangle_area (l w : ℕ) : ℕ :=
  l * w

theorem max_rectangle_area :
  ∃ (l w : ℕ), is_valid_rectangle l w ∧
    rectangle_area l w = 91 ∧
    ∀ (l' w' : ℕ), is_valid_rectangle l' w' →
      rectangle_area l' w' ≤ 91 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l174_17422


namespace NUMINAMATH_CALUDE_lacsap_hospital_staff_product_l174_17490

/-- Represents the Lacsap Hospital staff composition -/
structure HospitalStaff where
  doctors_excluding_emily : ℕ
  nurses_excluding_robert : ℕ
  emily_is_doctor : Bool
  robert_is_nurse : Bool

/-- Calculates the total number of doctors -/
def total_doctors (staff : HospitalStaff) : ℕ :=
  staff.doctors_excluding_emily + (if staff.emily_is_doctor then 1 else 0)

/-- Calculates the total number of nurses -/
def total_nurses (staff : HospitalStaff) : ℕ :=
  staff.nurses_excluding_robert + (if staff.robert_is_nurse then 1 else 0)

/-- Calculates the number of doctors excluding Robert -/
def doctors_excluding_robert (staff : HospitalStaff) : ℕ :=
  total_doctors staff

/-- Calculates the number of nurses excluding Robert -/
def nurses_excluding_robert (staff : HospitalStaff) : ℕ :=
  staff.nurses_excluding_robert

theorem lacsap_hospital_staff_product :
  ∀ (staff : HospitalStaff),
    staff.doctors_excluding_emily = 5 →
    staff.nurses_excluding_robert = 3 →
    staff.emily_is_doctor = true →
    staff.robert_is_nurse = true →
    (doctors_excluding_robert staff) * (nurses_excluding_robert staff) = 12 := by
  sorry

end NUMINAMATH_CALUDE_lacsap_hospital_staff_product_l174_17490


namespace NUMINAMATH_CALUDE_library_shelves_l174_17461

theorem library_shelves (books_per_shelf : ℕ) (total_books : ℕ) (h1 : books_per_shelf = 8) (h2 : total_books = 113920) :
  total_books / books_per_shelf = 14240 := by
  sorry

end NUMINAMATH_CALUDE_library_shelves_l174_17461


namespace NUMINAMATH_CALUDE_line_cannot_contain_point_l174_17426

theorem line_cannot_contain_point (m b : ℝ) (h : m * b < 0) :
  ¬∃ x y : ℝ, x = -2022 ∧ y = 0 ∧ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_line_cannot_contain_point_l174_17426


namespace NUMINAMATH_CALUDE_parabola_vertex_l174_17416

-- Define the parabola function
def f (x : ℝ) : ℝ := 3 * (x + 4)^2 - 9

-- State the theorem
theorem parabola_vertex :
  ∃ (x y : ℝ), (∀ t : ℝ, f t ≥ f x) ∧ f x = y ∧ x = -4 ∧ y = -9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l174_17416


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l174_17429

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - 2*x + a ≤ 0) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l174_17429
