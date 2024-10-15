import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_f_gt_5_empty_solution_set_condition_l1043_104339

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x + 2|

-- Theorem for the solution of f(x) > 5
theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -2 ∨ x > 4/3} :=
sorry

-- Theorem for the range of a where f(x) < a has no solution
theorem empty_solution_set_condition (a : ℝ) :
  ({x : ℝ | f x < a} = ∅) ↔ (a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_5_empty_solution_set_condition_l1043_104339


namespace NUMINAMATH_CALUDE_mass_of_six_moles_l1043_104378

/-- Given a compound with a molecular weight of 444 g/mol, 
    the mass of 6 moles of this compound is 2664 g. -/
theorem mass_of_six_moles (molecular_weight : ℝ) (h : molecular_weight = 444) : 
  6 * molecular_weight = 2664 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_six_moles_l1043_104378


namespace NUMINAMATH_CALUDE_rope_sections_l1043_104343

/-- Given a rope of 50 feet, prove that after using 1/5 for art and giving half of the remainder
    to a friend, the number of 2-foot sections that can be cut from the remaining rope is 10. -/
theorem rope_sections (total_rope : ℝ) (art_fraction : ℝ) (friend_fraction : ℝ) (section_length : ℝ) :
  total_rope = 50 ∧
  art_fraction = 1/5 ∧
  friend_fraction = 1/2 ∧
  section_length = 2 →
  (total_rope - art_fraction * total_rope) * (1 - friend_fraction) / section_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_rope_sections_l1043_104343


namespace NUMINAMATH_CALUDE_maria_green_towels_l1043_104379

/-- The number of green towels Maria bought -/
def green_towels : ℕ := 35

/-- The number of white towels Maria bought -/
def white_towels : ℕ := 21

/-- The number of towels Maria gave to her mother -/
def towels_given : ℕ := 34

/-- The number of towels Maria ended up with -/
def towels_left : ℕ := 22

/-- Theorem stating that the number of green towels Maria bought is 35 -/
theorem maria_green_towels :
  green_towels = 35 ∧
  green_towels + white_towels - towels_given = towels_left :=
by sorry

end NUMINAMATH_CALUDE_maria_green_towels_l1043_104379


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l1043_104347

def f (x : ℝ) := x^3 - 3*x^2 - 9*x + 2

theorem f_max_min_on_interval :
  ∃ (max min : ℝ), max = 2 ∧ min = -25 ∧
  (∀ x ∈ Set.Icc 0 4, f x ≤ max ∧ f x ≥ min) ∧
  (∃ x₁ ∈ Set.Icc 0 4, f x₁ = max) ∧
  (∃ x₂ ∈ Set.Icc 0 4, f x₂ = min) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l1043_104347


namespace NUMINAMATH_CALUDE_condition_D_iff_right_triangle_l1043_104317

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- Definition of a right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

/-- The condition a² = b² - c² -/
def condition_D (t : Triangle) : Prop :=
  t.a^2 = t.b^2 - t.c^2

/-- Theorem stating that condition D is equivalent to the triangle being a right triangle -/
theorem condition_D_iff_right_triangle (t : Triangle) :
  condition_D t ↔ is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_condition_D_iff_right_triangle_l1043_104317


namespace NUMINAMATH_CALUDE_student_age_problem_l1043_104333

theorem student_age_problem (num_students : ℕ) (teacher_age : ℕ) 
  (h1 : num_students = 20)
  (h2 : teacher_age = 42)
  (h3 : ∀ (student_avg : ℝ), 
    (num_students * student_avg + teacher_age) / (num_students + 1) = student_avg + 1) :
  ∃ (student_avg : ℝ), student_avg = 21 := by
sorry

end NUMINAMATH_CALUDE_student_age_problem_l1043_104333


namespace NUMINAMATH_CALUDE_triangle_count_properties_l1043_104341

/-- Function that counts the number of congruent integer-sided triangles with perimeter n -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating the properties of function f for specific values -/
theorem triangle_count_properties (h : ∀ n : ℕ, n ≥ 3 → f n = f n) :
  (f 1999 > f 1996) ∧ (f 2000 = f 1997) := by sorry

end NUMINAMATH_CALUDE_triangle_count_properties_l1043_104341


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_M_l1043_104368

-- Define the sum of divisors function
def sumOfDivisors (n : ℕ) : ℕ := sorry

-- Define M as the sum of divisors of 300
def M : ℕ := sumOfDivisors 300

-- Define a function to get the largest prime factor of a number
def largestPrimeFactor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem largest_prime_factor_of_M :
  largestPrimeFactor M = 31 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_M_l1043_104368


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_l1043_104340

/-- A circle with diameter endpoints at (0,0) and (10,0) -/
def circle_with_diameter (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = 25

/-- The y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- The intersection point of the circle and y-axis -/
def intersection_point (y : ℝ) : Prop :=
  circle_with_diameter 0 y ∧ y_axis 0

theorem circle_y_axis_intersection :
  ∃ y : ℝ, intersection_point y ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_l1043_104340


namespace NUMINAMATH_CALUDE_sarah_boxes_count_l1043_104377

def total_apples : ℕ := 49
def apples_per_box : ℕ := 7

theorem sarah_boxes_count :
  total_apples / apples_per_box = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_sarah_boxes_count_l1043_104377


namespace NUMINAMATH_CALUDE_exists_disjoint_graphs_l1043_104312

open Set

/-- The graph of a function f: [0, 1] → ℝ -/
def Graph (f : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ∈ Icc 0 1 ∧ p.2 = f p.1}

/-- The graph of the translated function f(x-a) -/
def GraphTranslated (f : ℝ → ℝ) (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ∈ Icc a (a+1) ∧ p.2 = f (p.1 - a)}

theorem exists_disjoint_graphs :
  ∀ a ∈ Ioo 0 1, ∃ f : ℝ → ℝ,
    Continuous f ∧
    f 0 = 0 ∧ f 1 = 0 ∧
    (Graph f) ∩ (GraphTranslated f a) = ∅ :=
sorry

end NUMINAMATH_CALUDE_exists_disjoint_graphs_l1043_104312


namespace NUMINAMATH_CALUDE_no_real_solutions_l1043_104323

theorem no_real_solutions : ∀ x : ℝ, x^2 ≠ 4 → x ≠ 2 → x ≠ -2 → 
  (8*x)/(x^2 - 4) ≠ (3*x)/(x - 2) - 4/(x + 2) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1043_104323


namespace NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l1043_104374

/-- A rectangular solid with volume 512 cm³, surface area 448 cm², and dimensions in geometric progression has a total edge length of 112 cm. -/
theorem rectangular_solid_edge_sum : 
  ∀ (a b c : ℝ),
    a > 0 → b > 0 → c > 0 →
    a * b * c = 512 →
    2 * (a * b + b * c + a * c) = 448 →
    ∃ (r : ℝ), r > 0 ∧ (a = b / r ∧ c = b * r) →
    4 * (a + b + c) = 112 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l1043_104374


namespace NUMINAMATH_CALUDE_max_value_quadratic_function_l1043_104348

theorem max_value_quadratic_function (p : ℝ) (hp : p > 0) :
  (∃ x ∈ Set.Icc (0 : ℝ) (4 / p), -1 / (2 * p) * x^2 + x > 1) ↔ 2 < p ∧ p < 1 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_function_l1043_104348


namespace NUMINAMATH_CALUDE_tom_tickets_left_l1043_104354

/-- The number of tickets Tom has left after winning some and spending some -/
def tickets_left (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (spent_tickets : ℕ) : ℕ :=
  whack_a_mole_tickets + skee_ball_tickets - spent_tickets

/-- Theorem stating that Tom has 50 tickets left -/
theorem tom_tickets_left : tickets_left 32 25 7 = 50 := by
  sorry

end NUMINAMATH_CALUDE_tom_tickets_left_l1043_104354


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l1043_104332

/-- Given an arithmetic sequence, prove that under certain conditions, 
    the maximum sum occurs at the 8th term -/
theorem arithmetic_sequence_max_sum 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2) 
  (h_15 : S 15 > 0) 
  (h_16 : S 16 < 0) : 
  ∃ (n : ℕ), ∀ (m : ℕ), S m ≤ S n ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l1043_104332


namespace NUMINAMATH_CALUDE_smallest_five_digit_negative_congruent_to_5_mod_17_l1043_104308

theorem smallest_five_digit_negative_congruent_to_5_mod_17 : 
  ∀ n : ℤ, -99999 ≤ n ∧ n < -9999 ∧ n ≡ 5 [ZMOD 17] → n ≥ -10013 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_negative_congruent_to_5_mod_17_l1043_104308


namespace NUMINAMATH_CALUDE_greg_bike_ride_l1043_104315

/-- Proves that Greg wants to ride 8 blocks given the conditions of the problem -/
theorem greg_bike_ride (rotations_per_block : ℕ) (rotations_so_far : ℕ) (rotations_needed : ℕ) :
  rotations_per_block = 200 →
  rotations_so_far = 600 →
  rotations_needed = 1000 →
  (rotations_so_far + rotations_needed) / rotations_per_block = 8 := by
  sorry

end NUMINAMATH_CALUDE_greg_bike_ride_l1043_104315


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l1043_104349

theorem sum_of_squares_and_products : (3 + 5)^2 + (3^2 + 5^2 + 3*5) = 113 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l1043_104349


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1043_104371

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := (1 + 2*I) / (a - I)
  (∃ (b : ℝ), z = b*I) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1043_104371


namespace NUMINAMATH_CALUDE_expression_evaluation_l1043_104351

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1043_104351


namespace NUMINAMATH_CALUDE_sqrt_combinability_with_sqrt_3_l1043_104376

theorem sqrt_combinability_with_sqrt_3 :
  ∃! x : ℝ, (x = Real.sqrt 32 ∨ x = -Real.sqrt 27 ∨ x = Real.sqrt 12 ∨ x = Real.sqrt (1/3)) ∧
  (∃ y : ℝ, x = y ∧ y ≠ 0 ∧ ∀ a b : ℝ, (y = a * Real.sqrt 3 + b → a = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_combinability_with_sqrt_3_l1043_104376


namespace NUMINAMATH_CALUDE_lune_area_l1043_104389

/-- The area of the region inside a semicircle of diameter 2, outside a semicircle of diameter 4,
    and outside an inscribed square with side length 2 is equal to -π + 2. -/
theorem lune_area (π : ℝ) (h : π > 0) : 
  let small_semicircle_area := (1/2) * π * (2/2)^2
  let large_semicircle_area := (1/2) * π * (4/2)^2
  let square_area := 2^2
  let sector_area := (1/4) * large_semicircle_area
  small_semicircle_area - sector_area - square_area = -π + 2 := by
  sorry

end NUMINAMATH_CALUDE_lune_area_l1043_104389


namespace NUMINAMATH_CALUDE_unique_toy_value_l1043_104325

theorem unique_toy_value (total_toys : ℕ) (total_worth : ℕ) (common_value : ℕ) (common_count : ℕ) :
  total_toys = common_count + 1 →
  total_worth = common_value * common_count + (total_worth - common_value * common_count) →
  common_count = 8 →
  total_toys = 9 →
  total_worth = 52 →
  common_value = 5 →
  total_worth - common_value * common_count = 12 :=
by sorry

end NUMINAMATH_CALUDE_unique_toy_value_l1043_104325


namespace NUMINAMATH_CALUDE_keith_and_jason_books_l1043_104394

/-- The number of books Keith and Jason have together -/
def total_books (keith_books jason_books : ℕ) : ℕ :=
  keith_books + jason_books

/-- Theorem: Keith and Jason have 41 books together -/
theorem keith_and_jason_books :
  total_books 20 21 = 41 := by
  sorry

end NUMINAMATH_CALUDE_keith_and_jason_books_l1043_104394


namespace NUMINAMATH_CALUDE_mikes_training_hours_l1043_104399

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a week number -/
inductive Week
  | First
  | Second

/-- Returns true if the given day is a weekday, false otherwise -/
def isWeekday (d : Day) : Bool :=
  match d with
  | Day.Saturday | Day.Sunday => false
  | _ => true

/-- Returns the maximum training hours for a given day and week -/
def maxTrainingHours (d : Day) (w : Week) : Nat :=
  match w with
  | Week.First => if isWeekday d then 2 else 1
  | Week.Second => if isWeekday d then 3 else 2

/-- Returns true if the given day is a rest day, false otherwise -/
def isRestDay (dayNumber : Nat) : Bool :=
  dayNumber % 5 == 0

/-- Calculates the total training hours for Mike over two weeks -/
def totalTrainingHours : Nat :=
  let firstWeekHours := 12  -- 5 weekdays * 2 hours + 2 weekend days * 1 hour
  let secondWeekHours := 16 -- 4 weekdays * 3 hours + 2 weekend days * 2 hours (1 rest day)
  firstWeekHours + secondWeekHours

/-- Theorem stating that Mike's total training hours over two weeks is 28 -/
theorem mikes_training_hours : totalTrainingHours = 28 := by
  sorry

#eval totalTrainingHours  -- This should output 28

end NUMINAMATH_CALUDE_mikes_training_hours_l1043_104399


namespace NUMINAMATH_CALUDE_min_sum_sequence_l1043_104319

theorem min_sum_sequence (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 → D > 0 →
  (∃ r : ℚ, C - B = B - A ∧ C / B = r ∧ D / C = r) →
  C / B = 7 / 3 →
  A + B + C + D ≥ 76 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_sequence_l1043_104319


namespace NUMINAMATH_CALUDE_total_money_found_l1043_104326

-- Define the value of each coin type in cents
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def penny_value : ℕ := 1

-- Define the number of each coin type found
def quarters_found : ℕ := 10
def dimes_found : ℕ := 3
def nickels_found : ℕ := 3
def pennies_found : ℕ := 5

-- Theorem to prove
theorem total_money_found :
  (quarters_found * quarter_value +
   dimes_found * dime_value +
   nickels_found * nickel_value +
   pennies_found * penny_value) = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_money_found_l1043_104326


namespace NUMINAMATH_CALUDE_det_AB_eq_one_l1043_104383

open Matrix

variable {n : ℕ}

theorem det_AB_eq_one
  (A B : Matrix (Fin n) (Fin n) ℝ)
  (hA : IsUnit A)
  (hB : IsUnit B)
  (h : (A + B⁻¹)⁻¹ = A⁻¹ + B) :
  det (A * B) = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_AB_eq_one_l1043_104383


namespace NUMINAMATH_CALUDE_x_eq_3_is_linear_l1043_104336

/-- Definition of a linear equation with one variable -/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation x = 3 -/
def f : ℝ → ℝ := λ x ↦ x - 3

/-- Theorem: x = 3 is a linear equation with one variable -/
theorem x_eq_3_is_linear : is_linear_equation_one_var f := by
  sorry


end NUMINAMATH_CALUDE_x_eq_3_is_linear_l1043_104336


namespace NUMINAMATH_CALUDE_jane_crayons_l1043_104338

theorem jane_crayons (initial_crayons : ℕ) (eaten_crayons : ℕ) : 
  initial_crayons = 87 → eaten_crayons = 7 → initial_crayons - eaten_crayons = 80 := by
  sorry

end NUMINAMATH_CALUDE_jane_crayons_l1043_104338


namespace NUMINAMATH_CALUDE_min_value_expression_l1043_104331

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2*a + b = a*b) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = x*y → 1/(a-1) + 2/(b-2) ≤ 1/(x-1) + 2/(y-2) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1043_104331


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1043_104310

/-- For a quadratic equation px^2 - 20x + 4 = 0, where p is nonzero,
    the equation has only one solution if and only if p = 25. -/
theorem unique_solution_quadratic (p : ℝ) (hp : p ≠ 0) :
  (∃! x, p * x^2 - 20 * x + 4 = 0) ↔ p = 25 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1043_104310


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_8_5_l1043_104353

theorem smallest_four_digit_mod_8_5 : ∃ (n : ℕ), 
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 8 = 5) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 8 = 5 → m ≥ n) ∧
  (n = 1005) := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_8_5_l1043_104353


namespace NUMINAMATH_CALUDE_overlapping_area_is_half_unit_l1043_104386

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p4.y) + p2.x * (p3.y - p1.y) + p3.x * (p4.y - p2.y) + p4.x * (p1.y - p3.y))

/-- The main theorem stating that the overlapping area is 0.5 square units -/
theorem overlapping_area_is_half_unit : 
  let t1p1 : Point := ⟨0, 0⟩
  let t1p2 : Point := ⟨6, 2⟩
  let t1p3 : Point := ⟨2, 6⟩
  let t2p1 : Point := ⟨6, 6⟩
  let t2p2 : Point := ⟨0, 2⟩
  let t2p3 : Point := ⟨2, 0⟩
  let ip1 : Point := ⟨2, 2⟩
  let ip2 : Point := ⟨4, 2⟩
  let ip3 : Point := ⟨3, 3⟩
  let ip4 : Point := ⟨2, 3⟩
  quadrilateralArea ip1 ip2 ip3 ip4 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_area_is_half_unit_l1043_104386


namespace NUMINAMATH_CALUDE_banana_permutations_eq_60_l1043_104390

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_eq_60_l1043_104390


namespace NUMINAMATH_CALUDE_uncle_payment_ratio_l1043_104392

/-- Represents the cost structure and payment for James' singing lessons -/
structure LessonPayment where
  total_lessons : ℕ
  free_lessons : ℕ
  full_price_lessons : ℕ
  half_price_lessons : ℕ
  lesson_cost : ℕ
  james_payment : ℕ

/-- Calculates the total cost of lessons -/
def total_cost (l : LessonPayment) : ℕ :=
  l.lesson_cost * (l.full_price_lessons + l.half_price_lessons)

/-- Calculates the amount paid by James' uncle -/
def uncle_payment (l : LessonPayment) : ℕ :=
  total_cost l - l.james_payment

/-- Theorem stating the ratio of uncle's payment to total cost is 1:2 -/
theorem uncle_payment_ratio (l : LessonPayment) 
  (h1 : l.total_lessons = 20)
  (h2 : l.free_lessons = 1)
  (h3 : l.full_price_lessons = 10)
  (h4 : l.half_price_lessons = 4)
  (h5 : l.lesson_cost = 5)
  (h6 : l.james_payment = 35) :
  2 * uncle_payment l = total_cost l := by
  sorry

#check uncle_payment_ratio

end NUMINAMATH_CALUDE_uncle_payment_ratio_l1043_104392


namespace NUMINAMATH_CALUDE_opposite_solutions_system_l1043_104306

theorem opposite_solutions_system (x y m : ℝ) : 
  x - 2*y = -3 → 
  2*x + 3*y = m - 1 → 
  x = -y → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_system_l1043_104306


namespace NUMINAMATH_CALUDE_min_max_f_on_interval_l1043_104375

-- Define the function f(x) = x³ - 12x
def f (x : ℝ) : ℝ := x^3 - 12*x

-- State the theorem
theorem min_max_f_on_interval :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc (-3) 3, f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc (-3) 3, f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc (-3) 3, f x₂ = max) ∧
    min = -16 ∧ max = 16 := by
  sorry


end NUMINAMATH_CALUDE_min_max_f_on_interval_l1043_104375


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_primes_l1043_104330

def is_divisible_by_primes (n : ℕ) : Prop :=
  ∃ k : ℕ, k * (2213 * 3323 * 6121) = (n / 2).factorial * 2^(n / 2)

theorem smallest_n_divisible_by_primes :
  (∀ m : ℕ, m < 12242 → ¬(is_divisible_by_primes m)) ∧
  (is_divisible_by_primes 12242) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_primes_l1043_104330


namespace NUMINAMATH_CALUDE_japanese_selectors_l1043_104307

theorem japanese_selectors (j c f : ℕ) : 
  j = 3 * c →
  c = f + 15 →
  j + c + f = 165 →
  j = 108 := by
sorry

end NUMINAMATH_CALUDE_japanese_selectors_l1043_104307


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1043_104321

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and lines
variable (perp_plane_line : Plane → Line → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (m n : Line)
  (h1 : perp_plane_line α m)
  (h2 : perp_plane_line β n)
  (h3 : perp_plane α β) :
  perp_line m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1043_104321


namespace NUMINAMATH_CALUDE_fish_count_l1043_104372

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 11

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 21 := by sorry

end NUMINAMATH_CALUDE_fish_count_l1043_104372


namespace NUMINAMATH_CALUDE_constant_remainder_l1043_104397

-- Define the polynomials
def f (b : ℚ) (x : ℚ) : ℚ := 12 * x^3 - 9 * x^2 + b * x + 8
def g (x : ℚ) : ℚ := 3 * x^2 - 4 * x + 2

-- Define the remainder function
def remainder (b : ℚ) (x : ℚ) : ℚ := f b x - g x * ((4 * x + 0) : ℚ)

-- Theorem statement
theorem constant_remainder :
  ∃ (c : ℚ), ∀ (x : ℚ), remainder (-4/3) x = c :=
sorry

end NUMINAMATH_CALUDE_constant_remainder_l1043_104397


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1043_104335

theorem quadrilateral_area (rectangle_area shaded_triangles_area : ℝ) 
  (h1 : rectangle_area = 24)
  (h2 : shaded_triangles_area = 7.5) :
  rectangle_area - shaded_triangles_area = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1043_104335


namespace NUMINAMATH_CALUDE_mixed_strategy_optimal_mixed_strategy_optimal_at_60_l1043_104311

/-- Represents the cost function for purchasing heaters from a store -/
structure StoreCost where
  typeA : ℝ  -- Cost per unit of Type A heater (including shipping)
  typeB : ℝ  -- Cost per unit of Type B heater (including shipping)

/-- Calculates the total cost for a store given the number of Type A heaters -/
def totalCost (store : StoreCost) (x : ℝ) : ℝ :=
  store.typeA * x + store.typeB * (100 - x)

/-- Store A's cost structure -/
def storeA : StoreCost := { typeA := 110, typeB := 210 }

/-- Store B's cost structure -/
def storeB : StoreCost := { typeA := 120, typeB := 202 }

/-- Cost function for buying Type A from Store A and Type B from Store B -/
def mixedCost (x : ℝ) : ℝ := storeA.typeA * x + storeB.typeB * (100 - x)

/-- Theorem: The mixed purchasing strategy is always the most cost-effective -/
theorem mixed_strategy_optimal (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 100) : 
  mixedCost x ≤ min (totalCost storeA x) (totalCost storeB x) := by
  sorry

/-- Corollary: When x = 60, the mixed strategy is more cost-effective than buying from a single store -/
theorem mixed_strategy_optimal_at_60 : 
  mixedCost 60 < min (totalCost storeA 60) (totalCost storeB 60) := by
  sorry

end NUMINAMATH_CALUDE_mixed_strategy_optimal_mixed_strategy_optimal_at_60_l1043_104311


namespace NUMINAMATH_CALUDE_painting_job_completion_time_l1043_104388

/-- Represents the time in hours it takes to paint a wall -/
structure PaintingTime where
  hours : ℚ
  is_positive : 0 < hours

/-- Represents a painter's rate in terms of wall painted per hour -/
def painting_rate (time : PaintingTime) : ℚ :=
  1 / time.hours

theorem painting_job_completion_time 
  (gina_time : PaintingTime)
  (tom_time : PaintingTime)
  (joint_work_time : ℚ)
  (h_gina : gina_time.hours = 3)
  (h_tom : tom_time.hours = 5)
  (h_joint : joint_work_time = 2)
  : ∃ (t : ℚ), t = 20/3 ∧ 
    (painting_rate gina_time + painting_rate tom_time) * joint_work_time + 
    painting_rate tom_time * (t - joint_work_time) = 1 :=
sorry

end NUMINAMATH_CALUDE_painting_job_completion_time_l1043_104388


namespace NUMINAMATH_CALUDE_cow_population_characteristics_l1043_104370

/-- Represents the number of cows in each category --/
structure CowPopulation where
  total : ℕ
  male : ℕ
  female : ℕ
  transgender : ℕ

/-- Represents the characteristics of cows in each category --/
structure CowCharacteristics where
  hornedMalePercentage : ℚ
  spottedFemalePercentage : ℚ
  uniquePatternTransgenderPercentage : ℚ

/-- Theorem stating the relation between spotted females and the sum of horned males and uniquely patterned transgender cows --/
theorem cow_population_characteristics 
  (pop : CowPopulation)
  (char : CowCharacteristics)
  (h1 : pop.total = 450)
  (h2 : pop.male = 3 * pop.female / 2)
  (h3 : pop.female = 2 * pop.transgender)
  (h4 : pop.total = pop.male + pop.female + pop.transgender)
  (h5 : char.hornedMalePercentage = 3/5)
  (h6 : char.spottedFemalePercentage = 1/2)
  (h7 : char.uniquePatternTransgenderPercentage = 7/10) :
  ↑(pop.female * 1) * char.spottedFemalePercentage = 
  ↑(pop.male * 1) * char.hornedMalePercentage + ↑(pop.transgender * 1) * char.uniquePatternTransgenderPercentage - 112 :=
sorry

end NUMINAMATH_CALUDE_cow_population_characteristics_l1043_104370


namespace NUMINAMATH_CALUDE_unique_arithmetic_grid_solution_l1043_104384

/-- Represents a 5x5 grid of integers -/
def Grid := Matrix (Fin 5) (Fin 5) Int

/-- Checks if a sequence of 5 integers forms an arithmetic progression -/
def isArithmeticSequence (seq : Fin 5 → Int) : Prop :=
  ∃ d : Int, ∀ i : Fin 5, i.val < 4 → seq (i + 1) - seq i = d

/-- The initial grid with given values -/
def initialGrid : Grid :=
  fun i j => if i = 0 ∧ j = 0 then 2
             else if i = 0 ∧ j = 4 then 14
             else if i = 1 ∧ j = 1 then 8
             else if i = 2 ∧ j = 1 then 11
             else if i = 2 ∧ j = 2 then 16
             else if i = 4 ∧ j = 0 then 10
             else 0  -- placeholder for unknown values

/-- Theorem stating the existence and uniqueness of the solution -/
theorem unique_arithmetic_grid_solution :
  ∃! g : Grid,
    (∀ i j, initialGrid i j ≠ 0 → g i j = initialGrid i j) ∧
    (∀ i, isArithmeticSequence (fun j => g i j)) ∧
    (∀ j, isArithmeticSequence (fun i => g i j)) := by
  sorry

end NUMINAMATH_CALUDE_unique_arithmetic_grid_solution_l1043_104384


namespace NUMINAMATH_CALUDE_checkerboard_partition_l1043_104396

theorem checkerboard_partition (n : ℕ) : 
  n % 5 = 0 → n % 7 = 0 → n ≤ 200 → n % 6 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_checkerboard_partition_l1043_104396


namespace NUMINAMATH_CALUDE_cindy_calculation_l1043_104356

theorem cindy_calculation (x : ℚ) : 
  ((x - 5) * 3 / 7 = 10) → ((3 * x - 5) / 7 = 80 / 7) := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l1043_104356


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_l1043_104318

/-- Given a cubic polynomial Q with specific values at 0, 1, and -1, prove that Q(2) + Q(-2) = 20m -/
theorem cubic_polynomial_sum (m : ℝ) (Q : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, Q x = a * x^3 + b * x^2 + c * x + 2 * m) →
  Q 0 = 2 * m →
  Q 1 = 3 * m →
  Q (-1) = 5 * m →
  Q 2 + Q (-2) = 20 * m :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_l1043_104318


namespace NUMINAMATH_CALUDE_base_prime_rep_945_l1043_104352

def base_prime_representation (n : ℕ) : List ℕ :=
  sorry

theorem base_prime_rep_945 : base_prime_representation 945 = [3, 1, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_rep_945_l1043_104352


namespace NUMINAMATH_CALUDE_initial_concentrated_kola_percentage_l1043_104366

/-- Proves that the initial percentage of concentrated kola in a 340-liter solution is 5% -/
theorem initial_concentrated_kola_percentage
  (initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_concentrated_kola : ℝ)
  (new_volume : ℝ)
  (new_sugar_percentage : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percentage = 88 / 100)
  (h3 : added_sugar = 3.2)
  (h4 : added_water = 10)
  (h5 : added_concentrated_kola = 6.8)
  (h6 : new_volume = initial_volume + added_sugar + added_water + added_concentrated_kola)
  (h7 : new_sugar_percentage = 7.5 / 100) :
  ∃ (initial_concentrated_kola_percentage : ℝ),
    initial_concentrated_kola_percentage = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_initial_concentrated_kola_percentage_l1043_104366


namespace NUMINAMATH_CALUDE_red_face_probability_l1043_104387

def large_cube_edge : ℕ := 6
def small_cube_edge : ℕ := 1

def total_small_cubes : ℕ := large_cube_edge ^ 3

def corner_cubes : ℕ := 8
def edge_cubes : ℕ := 4 * 12
def face_cubes : ℕ := 4 * 6

def red_faced_cubes : ℕ := corner_cubes + edge_cubes + face_cubes

theorem red_face_probability :
  (red_faced_cubes : ℚ) / total_small_cubes = 10 / 27 := by sorry

end NUMINAMATH_CALUDE_red_face_probability_l1043_104387


namespace NUMINAMATH_CALUDE_storks_birds_difference_l1043_104327

theorem storks_birds_difference : 
  let initial_birds : ℕ := 2
  let initial_storks : ℕ := 6
  let additional_birds : ℕ := 3
  let final_birds : ℕ := initial_birds + additional_birds
  initial_storks - final_birds = 1 := by sorry

end NUMINAMATH_CALUDE_storks_birds_difference_l1043_104327


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l1043_104322

def is_in_second_quadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi

def is_in_first_or_third_quadrant (α : Real) : Prop :=
  ∃ n : Int, (2 * n * Real.pi + Real.pi / 4 < α ∧ α < 2 * n * Real.pi + Real.pi / 2) ∨
             ((2 * n + 1) * Real.pi + Real.pi / 4 < α ∧ α < (2 * n + 1) * Real.pi + Real.pi / 2)

theorem half_angle_quadrant (α : Real) :
  is_in_second_quadrant α → is_in_first_or_third_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l1043_104322


namespace NUMINAMATH_CALUDE_success_permutations_l1043_104324

/-- The number of distinct permutations of a word with repeated letters -/
def permutationsWithRepetition (totalLetters : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repetitions.map Nat.factorial).prod

/-- The word "SUCCESS" has 7 letters with 'S' appearing 3 times, 'C' appearing 2 times, 
    and 'U' and 'E' appearing once each -/
def successWord : (ℕ × List ℕ) :=
  (7, [3, 2, 1, 1])

theorem success_permutations :
  permutationsWithRepetition successWord.1 successWord.2 = 420 := by
  sorry

end NUMINAMATH_CALUDE_success_permutations_l1043_104324


namespace NUMINAMATH_CALUDE_tangent_intersection_for_specific_circles_l1043_104357

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Finds the x-coordinate of the intersection point between the common tangent line
    of two circles and the x-axis -/
def tangentIntersectionX (c1 c2 : Circle) : ℝ :=
  sorry

theorem tangent_intersection_for_specific_circles :
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (12, 0), radius := 5 }
  tangentIntersectionX c1 c2 = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_intersection_for_specific_circles_l1043_104357


namespace NUMINAMATH_CALUDE_factorization_3x_squared_minus_27_l1043_104344

theorem factorization_3x_squared_minus_27 (x : ℝ) : 3 * x^2 - 27 = 3 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3x_squared_minus_27_l1043_104344


namespace NUMINAMATH_CALUDE_new_hires_count_l1043_104395

theorem new_hires_count (initial_workers : ℕ) (initial_men_ratio : ℚ) (final_women_percentage : ℚ) : 
  initial_workers = 90 →
  initial_men_ratio = 2/3 →
  final_women_percentage = 40/100 →
  ∃ (new_hires : ℕ), 
    (initial_workers * (1 - initial_men_ratio) + new_hires) / (initial_workers + new_hires) = final_women_percentage ∧
    new_hires = 10 := by
  sorry

end NUMINAMATH_CALUDE_new_hires_count_l1043_104395


namespace NUMINAMATH_CALUDE_symmetry_of_point_l1043_104364

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry of a point with respect to the origin -/
def symmetrical_to_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetry_of_point :
  let P : Point := ⟨3, 2⟩
  let P' : Point := symmetrical_to_origin P
  P'.x = -3 ∧ P'.y = -2 := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l1043_104364


namespace NUMINAMATH_CALUDE_valid_colorings_6x6_l1043_104363

/-- Recursive function for the number of valid colorings of an nxn grid -/
def f : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 1
| n + 2 => n * (n + 1) * f (n + 1) + (n * (n + 1)^2 / 2) * f n

/-- The size of the grid -/
def grid_size : ℕ := 6

/-- The number of red squares required in each row and column -/
def red_squares_per_line : ℕ := 2

/-- Theorem stating the number of valid colorings for a 6x6 grid -/
theorem valid_colorings_6x6 : f grid_size = 67950 := by
  sorry

end NUMINAMATH_CALUDE_valid_colorings_6x6_l1043_104363


namespace NUMINAMATH_CALUDE_inequality_solution_l1043_104369

theorem inequality_solution (x : ℝ) : 
  (x - 2) / (x - 1) > (4 * x - 1) / (3 * x + 8) ↔ 
  (x > -3 ∧ x < -2) ∨ (x > -8/3 ∧ x < 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1043_104369


namespace NUMINAMATH_CALUDE_division_problem_l1043_104300

theorem division_problem (a b q : ℕ) 
  (h1 : a - b = 1390) 
  (h2 : a = 1650) 
  (h3 : a = b * q + 15) : q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1043_104300


namespace NUMINAMATH_CALUDE_find_number_l1043_104367

theorem find_number : ∃! x : ℚ, (x + 305) / 16 = 31 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1043_104367


namespace NUMINAMATH_CALUDE_four_solutions_l1043_104355

-- Define the two equations
def equation1 (x y : ℝ) : Prop := (x - y + 3) * (4 * x + y - 5) = 0
def equation2 (x y : ℝ) : Prop := (x + y - 3) * (3 * x - 4 * y + 6) = 0

-- Define a solution as a pair of real numbers satisfying both equations
def is_solution (p : ℝ × ℝ) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

-- State the theorem
theorem four_solutions :
  ∃ (s : Finset (ℝ × ℝ)), s.card = 4 ∧ (∀ p ∈ s, is_solution p) ∧
  (∀ p : ℝ × ℝ, is_solution p → p ∈ s) :=
sorry

end NUMINAMATH_CALUDE_four_solutions_l1043_104355


namespace NUMINAMATH_CALUDE_reading_time_difference_l1043_104304

/-- The difference in reading time between two people reading the same book -/
theorem reading_time_difference (xanthia_rate molly_rate book_pages : ℕ) : 
  xanthia_rate = 150 → 
  molly_rate = 75 → 
  book_pages = 300 → 
  (book_pages / molly_rate - book_pages / xanthia_rate) * 60 = 120 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l1043_104304


namespace NUMINAMATH_CALUDE_extrema_not_necessarily_unique_l1043_104313

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define what it means for a point to be an extremum
def IsExtremum (f : RealFunction) (x : ℝ) (a b : ℝ) : Prop :=
  ∀ y ∈ Set.Icc a b, f x ≥ f y ∨ f x ≤ f y

-- Theorem statement
theorem extrema_not_necessarily_unique :
  ∃ (f : RealFunction) (a b x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧ a < x₁ ∧ x₁ < b ∧ a < x₂ ∧ x₂ < b ∧
    IsExtremum f x₁ a b ∧ IsExtremum f x₂ a b :=
sorry

end NUMINAMATH_CALUDE_extrema_not_necessarily_unique_l1043_104313


namespace NUMINAMATH_CALUDE_projection_problem_l1043_104365

/-- Given that the projection of [2, 5] onto w is [2/5, -1/5],
    prove that the projection of [3, 2] onto w is [8/5, -4/5] -/
theorem projection_problem (w : ℝ × ℝ) :
  let v₁ : ℝ × ℝ := (2, 5)
  let v₂ : ℝ × ℝ := (3, 2)
  let proj₁ : ℝ × ℝ := (2/5, -1/5)
  (∃ (k : ℝ), w = k • proj₁) →
  (v₁ • w / (w • w)) • w = proj₁ →
  (v₂ • w / (w • w)) • w = (8/5, -4/5) := by
sorry

end NUMINAMATH_CALUDE_projection_problem_l1043_104365


namespace NUMINAMATH_CALUDE_stratified_sampling_low_income_l1043_104358

/-- Represents the number of households sampled from a given group -/
def sampleSize (totalSize : ℕ) (groupSize : ℕ) (sampledHighIncome : ℕ) (totalHighIncome : ℕ) : ℕ :=
  (sampledHighIncome * groupSize) / totalHighIncome

theorem stratified_sampling_low_income 
  (totalHouseholds : ℕ) 
  (highIncomeHouseholds : ℕ) 
  (lowIncomeHouseholds : ℕ) 
  (sampledHighIncome : ℕ) :
  totalHouseholds = 500 →
  highIncomeHouseholds = 125 →
  lowIncomeHouseholds = 95 →
  sampledHighIncome = 25 →
  sampleSize totalHouseholds lowIncomeHouseholds sampledHighIncome highIncomeHouseholds = 19 := by
  sorry

#check stratified_sampling_low_income

end NUMINAMATH_CALUDE_stratified_sampling_low_income_l1043_104358


namespace NUMINAMATH_CALUDE_lattice_triangle_area_bound_l1043_104361

/-- A 3D lattice point is represented as a triple of integers -/
def LatticePoint3D := ℤ × ℤ × ℤ

/-- A triangle in 3D space is represented by its three vertices -/
structure Triangle3D where
  v1 : LatticePoint3D
  v2 : LatticePoint3D
  v3 : LatticePoint3D

/-- The area of a triangle -/
noncomputable def area (t : Triangle3D) : ℝ := sorry

/-- Theorem: The area of a triangle with vertices at 3D lattice points is at least 1/2 -/
theorem lattice_triangle_area_bound (t : Triangle3D) : area t ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_lattice_triangle_area_bound_l1043_104361


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l1043_104302

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 distinguishable balls into 3 indistinguishable boxes is 301 -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 301 := by sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l1043_104302


namespace NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l1043_104316

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 
  ∃ (m : ℕ), m = 6 ∧ 
  (∃ (k : ℤ), n^4 - n^2 = m * k) ∧ 
  (∀ (d : ℕ), d > m → ¬∃ (j : ℤ), ∀ (n : ℤ), n^4 - n^2 = d * j) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l1043_104316


namespace NUMINAMATH_CALUDE_driving_time_to_airport_l1043_104320

-- Define time in minutes since midnight
def flight_time : ℕ := 20 * 60
def check_in_buffer : ℕ := 2 * 60
def house_departure_time : ℕ := 17 * 60
def parking_and_terminal_time : ℕ := 15

-- Theorem statement
theorem driving_time_to_airport :
  let check_in_time := flight_time - check_in_buffer
  let airport_arrival_time := check_in_time - parking_and_terminal_time
  airport_arrival_time - house_departure_time = 45 := by
sorry

end NUMINAMATH_CALUDE_driving_time_to_airport_l1043_104320


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l1043_104362

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^3 * a^6 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l1043_104362


namespace NUMINAMATH_CALUDE_larry_stickers_l1043_104345

/-- The number of stickers Larry starts with -/
def initial_stickers : ℕ := 93

/-- The number of stickers Larry loses -/
def lost_stickers : ℕ := 6

/-- The number of stickers Larry ends with -/
def final_stickers : ℕ := initial_stickers - lost_stickers

theorem larry_stickers : final_stickers = 87 := by
  sorry

end NUMINAMATH_CALUDE_larry_stickers_l1043_104345


namespace NUMINAMATH_CALUDE_cat_food_percentage_l1043_104328

/-- Proves that given 7 dogs and 4 cats, where all dogs receive equal amounts of food,
    all cats receive equal amounts of food, and the total food for all cats equals
    the food for one dog, the percentage of total food that one cat receives is 1/32. -/
theorem cat_food_percentage :
  ∀ (dog_food cat_food : ℚ),
  dog_food > 0 →
  cat_food > 0 →
  4 * cat_food = dog_food →
  (cat_food / (7 * dog_food + 4 * cat_food)) = 1 / 32 :=
by
  sorry

end NUMINAMATH_CALUDE_cat_food_percentage_l1043_104328


namespace NUMINAMATH_CALUDE_largest_angle_in_hexagon_l1043_104391

/-- Theorem: In a hexagon ABCDEF with given angle conditions, the largest angle measures 304°. -/
theorem largest_angle_in_hexagon (A B C D E F : ℝ) : 
  A = 100 →
  B = 120 →
  C = D →
  F = 3 * C + 10 →
  A + B + C + D + E + F = 720 →
  max A (max B (max C (max D (max E F)))) = 304 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_hexagon_l1043_104391


namespace NUMINAMATH_CALUDE_exponent_law_multiplication_l1043_104337

theorem exponent_law_multiplication (y : ℝ) (n : ℤ) (h : y ≠ 0) :
  y * y^n = y^(n + 1) := by sorry

end NUMINAMATH_CALUDE_exponent_law_multiplication_l1043_104337


namespace NUMINAMATH_CALUDE_marcos_strawberries_weight_l1043_104350

theorem marcos_strawberries_weight (total_weight dad_weight : ℕ) 
  (h1 : total_weight = 23)
  (h2 : dad_weight = 9) :
  total_weight - dad_weight = 14 := by
  sorry

end NUMINAMATH_CALUDE_marcos_strawberries_weight_l1043_104350


namespace NUMINAMATH_CALUDE_cave_depth_remaining_l1043_104373

theorem cave_depth_remaining (total_depth : ℕ) (traveled_distance : ℕ) 
  (h1 : total_depth = 1218)
  (h2 : traveled_distance = 849) :
  total_depth - traveled_distance = 369 := by
sorry

end NUMINAMATH_CALUDE_cave_depth_remaining_l1043_104373


namespace NUMINAMATH_CALUDE_multiply_102_98_l1043_104329

theorem multiply_102_98 : 102 * 98 = 9996 := by
  sorry

end NUMINAMATH_CALUDE_multiply_102_98_l1043_104329


namespace NUMINAMATH_CALUDE_root_location_l1043_104301

theorem root_location (a b : ℝ) (n : ℤ) : 
  (2 : ℝ)^a = 3 → 
  (3 : ℝ)^b = 2 → 
  (∃ x_b : ℝ, x_b ∈ Set.Ioo (n : ℝ) (n + 1) ∧ a^x_b + x_b - b = 0) → 
  n = -1 := by
sorry

end NUMINAMATH_CALUDE_root_location_l1043_104301


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l1043_104342

/-- Represents the outcome of tossing three coins -/
inductive CoinToss
  | HHH
  | HHT
  | HTH
  | HTT
  | THH
  | THT
  | TTH
  | TTT

/-- The event of getting no more than one heads -/
def noMoreThanOneHeads (t : CoinToss) : Prop :=
  t = CoinToss.HTT ∨ t = CoinToss.THT ∨ t = CoinToss.TTH ∨ t = CoinToss.TTT

/-- The event of getting at least two heads -/
def atLeastTwoHeads (t : CoinToss) : Prop :=
  t = CoinToss.HHH ∨ t = CoinToss.HHT ∨ t = CoinToss.HTH ∨ t = CoinToss.THH

/-- Theorem stating that the two events are mutually exclusive -/
theorem mutually_exclusive_events :
  ∀ t : CoinToss, ¬(noMoreThanOneHeads t ∧ atLeastTwoHeads t) :=
by
  sorry


end NUMINAMATH_CALUDE_mutually_exclusive_events_l1043_104342


namespace NUMINAMATH_CALUDE_emily_pastry_production_l1043_104385

/-- Emily's pastry production problem -/
theorem emily_pastry_production (p h : ℕ) : 
  p = 3 * h →
  h = 1 →
  (p - 3) * (h + 3) - p * h = 3 := by
  sorry

end NUMINAMATH_CALUDE_emily_pastry_production_l1043_104385


namespace NUMINAMATH_CALUDE_triangle_angles_from_exterior_l1043_104382

theorem triangle_angles_from_exterior (A B C : ℝ) : 
  A + B + C = 180 →
  (180 - B) / (180 - C) = 12 / 7 →
  (180 - B) - (180 - C) = 50 →
  (A = 10 ∧ B = 60 ∧ C = 110) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_from_exterior_l1043_104382


namespace NUMINAMATH_CALUDE_solve_equation_l1043_104303

theorem solve_equation (x : ℚ) : (3 * x + 4) / 7 = 15 → x = 101 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1043_104303


namespace NUMINAMATH_CALUDE_mocktail_lime_cost_l1043_104314

/-- Represents the cost of limes in dollars for a given number of limes -/
def lime_cost (num_limes : ℕ) : ℚ :=
  (num_limes : ℚ) / 3

/-- Calculates the number of limes needed for a given number of days -/
def limes_needed (days : ℕ) : ℕ :=
  (days + 1) / 2

theorem mocktail_lime_cost : lime_cost (limes_needed 30) = 5 := by
  sorry

end NUMINAMATH_CALUDE_mocktail_lime_cost_l1043_104314


namespace NUMINAMATH_CALUDE_existence_of_odd_powers_representation_l1043_104305

theorem existence_of_odd_powers_representation (m : ℤ) :
  ∃ (a b k : ℤ), 
    Odd a ∧ 
    Odd b ∧ 
    k ≥ 0 ∧ 
    2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_odd_powers_representation_l1043_104305


namespace NUMINAMATH_CALUDE_focal_chord_circle_tangent_to_directrix_l1043_104309

-- Define a parabola
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ := (0, p)
  vertex : ℝ × ℝ := (0, 0)
  directrix : ℝ → ℝ := fun x ↦ -p

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the focal chord circle
def focal_chord_circle (parab : Parabola) : Circle :=
  { center := parab.focus
  , radius := parab.p }

-- Theorem statement
theorem focal_chord_circle_tangent_to_directrix (parab : Parabola) :
  let circle := focal_chord_circle parab
  let lowest_point := (circle.center.1, circle.center.2 - circle.radius)
  lowest_point.2 = 0 ∧ parab.directrix lowest_point.1 = -parab.p :=
sorry

end NUMINAMATH_CALUDE_focal_chord_circle_tangent_to_directrix_l1043_104309


namespace NUMINAMATH_CALUDE_cube_vertex_to_plane_distance_l1043_104398

/-- The distance from the closest vertex of a cube to a plane, given specific conditions --/
theorem cube_vertex_to_plane_distance (s : ℝ) (h₁ h₂ h₃ : ℝ) : 
  s = 8 ∧ h₁ = 8 ∧ h₂ = 9 ∧ h₃ = 10 → 
  ∃ (a b c d : ℝ), 
    a^2 + b^2 + c^2 = 1 ∧
    s * a + d = h₁ ∧
    s * b + d = h₂ ∧
    s * c + d = h₃ ∧
    d = (27 - Real.sqrt 186) / 3 := by
  sorry

#check cube_vertex_to_plane_distance

end NUMINAMATH_CALUDE_cube_vertex_to_plane_distance_l1043_104398


namespace NUMINAMATH_CALUDE_max_expression_l1043_104380

/-- A permutation of the digits 1 to 9 -/
def Digits := Fin 9 → Fin 9

/-- Check if a permutation is valid (bijective) -/
def is_valid_permutation (p : Digits) : Prop :=
  Function.Bijective p

/-- Convert three consecutive digits in a permutation to a number -/
def to_number (p : Digits) (start : Fin 9) : ℕ :=
  100 * (p start).val + 10 * (p (start + 1)).val + (p (start + 2)).val

/-- The expression to be maximized -/
def expression (p : Digits) : ℤ :=
  (to_number p 0 : ℤ) + (to_number p 3 : ℤ) - (to_number p 6 : ℤ)

/-- The main theorem -/
theorem max_expression :
  ∃ (p : Digits), is_valid_permutation p ∧ 
    (∀ (q : Digits), is_valid_permutation q → expression q ≤ expression p) ∧
    expression p = 1716 := by sorry

end NUMINAMATH_CALUDE_max_expression_l1043_104380


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l1043_104381

/-- A trinomial is a perfect square if it can be expressed as (x - a)^2 for some real number a -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (x - k)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, IsPerfectSquareTrinomial 1 (-m) 16 → m = 8 ∨ m = -8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l1043_104381


namespace NUMINAMATH_CALUDE_dime_nickel_difference_l1043_104393

/-- Proves that given 70 cents total and 2 nickels, the number of dimes exceeds the number of nickels by 4 -/
theorem dime_nickel_difference :
  ∀ (total_cents : ℕ) (num_nickels : ℕ) (nickel_value : ℕ) (dime_value : ℕ),
    total_cents = 70 →
    num_nickels = 2 →
    nickel_value = 5 →
    dime_value = 10 →
    ∃ (num_dimes : ℕ),
      num_dimes * dime_value + num_nickels * nickel_value = total_cents ∧
      num_dimes = num_nickels + 4 := by
  sorry

end NUMINAMATH_CALUDE_dime_nickel_difference_l1043_104393


namespace NUMINAMATH_CALUDE_ps_length_l1043_104360

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0
  pq_length : Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 15
  qr_length : Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 20

-- Define points S and T
def S (P R : ℝ × ℝ) : ℝ × ℝ := sorry
def T (Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the theorem
theorem ps_length (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  let S := S P R
  let T := T Q R
  (S.1 - P.1) * (T.1 - S.1) + (S.2 - P.2) * (T.2 - S.2) = 0 →
  Real.sqrt ((T.1 - S.1)^2 + (T.2 - S.2)^2) = 12 →
  Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ps_length_l1043_104360


namespace NUMINAMATH_CALUDE_saly_needs_ten_eggs_l1043_104359

/-- The number of eggs needed by various individuals and produced by the farm --/
structure EggNeeds where
  ben_weekly : ℕ  -- Ben's weekly egg needs
  ked_weekly : ℕ  -- Ked's weekly egg needs
  monthly_total : ℕ  -- Total eggs produced by the farm in a month
  weeks_in_month : ℕ  -- Number of weeks in a month

/-- Calculates Saly's weekly egg needs based on the given conditions --/
def saly_weekly_needs (e : EggNeeds) : ℕ :=
  (e.monthly_total - (e.ben_weekly + e.ked_weekly) * e.weeks_in_month) / e.weeks_in_month

/-- Theorem stating that Saly needs 10 eggs per week given the conditions --/
theorem saly_needs_ten_eggs (e : EggNeeds) 
  (h1 : e.ben_weekly = 14)
  (h2 : e.ked_weekly = e.ben_weekly / 2)
  (h3 : e.monthly_total = 124)
  (h4 : e.weeks_in_month = 4) : 
  saly_weekly_needs e = 10 := by
  sorry

end NUMINAMATH_CALUDE_saly_needs_ten_eggs_l1043_104359


namespace NUMINAMATH_CALUDE_prob_at_least_one_girl_pair_value_l1043_104334

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 4

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The total number of ways to pair 8 people -/
def total_pairings : ℕ := (total_people.factorial) / (2^num_pairs * num_pairs.factorial)

/-- The number of ways to pair boys with girls (no all-girl pairs) -/
def boy_girl_pairings : ℕ := num_boys.factorial

/-- The probability of at least one pair consisting of two girls -/
def prob_at_least_one_girl_pair : ℚ := 1 - (boy_girl_pairings : ℚ) / total_pairings

theorem prob_at_least_one_girl_pair_value :
  prob_at_least_one_girl_pair = 27 / 35 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_girl_pair_value_l1043_104334


namespace NUMINAMATH_CALUDE_distance_inequality_l1043_104346

theorem distance_inequality (x b : ℝ) (h1 : b > 0) (h2 : |x - 3| + |x - 5| < b) : b > 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_inequality_l1043_104346
