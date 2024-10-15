import Mathlib

namespace NUMINAMATH_CALUDE_solution_satisfies_inequalities_inequalities_imply_solution_solution_is_correct_l1934_193428

-- Define the system of inequalities
def inequality1 (x : ℝ) : Prop := x + 2 < 3 + 2*x
def inequality2 (x : ℝ) : Prop := 4*x - 3 < 3*x - 1
def inequality3 (x : ℝ) : Prop := 8 + 5*x ≥ 6*x + 7

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

-- Theorem stating that the solution set satisfies all inequalities
theorem solution_satisfies_inequalities :
  ∀ x ∈ solution_set, inequality1 x ∧ inequality2 x ∧ inequality3 x :=
sorry

-- Theorem stating that any real number satisfying all inequalities is in the solution set
theorem inequalities_imply_solution :
  ∀ x : ℝ, inequality1 x ∧ inequality2 x ∧ inequality3 x → x ∈ solution_set :=
sorry

-- Main theorem: The solution set is exactly (-1, 1]
theorem solution_is_correct :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality1 x ∧ inequality2 x ∧ inequality3 x :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_inequalities_inequalities_imply_solution_solution_is_correct_l1934_193428


namespace NUMINAMATH_CALUDE_number_999_in_column_C_l1934_193413

/-- Represents the columns in which numbers are arranged --/
inductive Column
  | A | B | C | D | E | F | G

/-- Determines the column for a given positive integer greater than 1 --/
def column_for_number (n : ℕ) : Column :=
  sorry

/-- The main theorem stating that 999 is in column C --/
theorem number_999_in_column_C : column_for_number 999 = Column.C := by
  sorry

end NUMINAMATH_CALUDE_number_999_in_column_C_l1934_193413


namespace NUMINAMATH_CALUDE_max_value_fraction_l1934_193462

theorem max_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 2 / 2 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * b + b * c) / (a^2 + b^2 + c^2) = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1934_193462


namespace NUMINAMATH_CALUDE_circle_area_decrease_l1934_193458

theorem circle_area_decrease (r : ℝ) (h : r > 0) :
  let new_r := r / 2
  let original_area := π * r^2
  let new_area := π * new_r^2
  (original_area - new_area) / original_area = 3/4 := by sorry

end NUMINAMATH_CALUDE_circle_area_decrease_l1934_193458


namespace NUMINAMATH_CALUDE_find_a_over_b_l1934_193404

theorem find_a_over_b (a b c d e f : ℝ) 
  (h1 : a * b * c / (d * e * f) = 0.1875)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) :
  a / b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_find_a_over_b_l1934_193404


namespace NUMINAMATH_CALUDE_train_vs_airplanes_capacity_difference_l1934_193429

/-- The number of passengers a single train car can carry -/
def train_car_capacity : ℕ := 60

/-- The number of passengers a 747 airplane can carry -/
def airplane_capacity : ℕ := 366

/-- The number of cars in the train -/
def train_cars : ℕ := 16

/-- The number of airplanes being compared -/
def num_airplanes : ℕ := 2

/-- Theorem stating the difference in passenger capacity between the train and the airplanes -/
theorem train_vs_airplanes_capacity_difference :
  train_cars * train_car_capacity - num_airplanes * airplane_capacity = 228 := by
  sorry

end NUMINAMATH_CALUDE_train_vs_airplanes_capacity_difference_l1934_193429


namespace NUMINAMATH_CALUDE_remainder_x_power_10_minus_1_div_x_plus_1_l1934_193459

theorem remainder_x_power_10_minus_1_div_x_plus_1 (x : ℝ) : 
  (x^10 - 1) % (x + 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_x_power_10_minus_1_div_x_plus_1_l1934_193459


namespace NUMINAMATH_CALUDE_cricket_count_l1934_193438

theorem cricket_count (initial : Real) (additional : Real) :
  initial = 7.0 → additional = 11.0 → initial + additional = 18.0 := by
  sorry

end NUMINAMATH_CALUDE_cricket_count_l1934_193438


namespace NUMINAMATH_CALUDE_vegetarian_count_l1934_193430

theorem vegetarian_count (non_veg_only : ℕ) (both : ℕ) (total_veg : ℕ) 
  (h1 : non_veg_only = 9)
  (h2 : both = 12)
  (h3 : total_veg = 28) :
  total_veg - both = 16 := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_count_l1934_193430


namespace NUMINAMATH_CALUDE_digit_equation_sum_l1934_193455

/-- Represents a base-10 digit -/
def Digit := Fin 10

/-- Checks if all digits in a natural number are the same -/
def allDigitsSame (n : ℕ) : Prop :=
  ∃ d : Digit, n = d.val * 100 + d.val * 10 + d.val

/-- The main theorem -/
theorem digit_equation_sum :
  ∀ (Y E M L : Digit),
    Y ≠ E → Y ≠ M → Y ≠ L → E ≠ M → E ≠ L → M ≠ L →
    (Y.val * 10 + E.val) * (M.val * 10 + E.val) = L.val * 100 + L.val * 10 + L.val →
    E.val + M.val + L.val + Y.val = 15 := by
  sorry


end NUMINAMATH_CALUDE_digit_equation_sum_l1934_193455


namespace NUMINAMATH_CALUDE_semi_circle_perimeter_l1934_193488

/-- The perimeter of a semi-circle with radius 38.50946843518593 cm is 198.03029487037186 cm. -/
theorem semi_circle_perimeter :
  let r : ℝ := 38.50946843518593
  let π : ℝ := Real.pi
  let perimeter : ℝ := π * r + 2 * r
  perimeter = 198.03029487037186 := by sorry

end NUMINAMATH_CALUDE_semi_circle_perimeter_l1934_193488


namespace NUMINAMATH_CALUDE_intersection_implies_t_equals_two_l1934_193478

theorem intersection_implies_t_equals_two (t : ℝ) : 
  let M : Set ℝ := {1, t^2}
  let N : Set ℝ := {-2, t+2}
  (M ∩ N).Nonempty → t = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_t_equals_two_l1934_193478


namespace NUMINAMATH_CALUDE_faith_works_five_days_l1934_193415

/-- Faith's work schedule and earnings --/
def faith_work_schedule (hourly_rate : ℚ) (regular_hours : ℕ) (overtime_hours : ℕ) (weekly_earnings : ℚ) : Prop :=
  ∃ (days_worked : ℕ),
    (hourly_rate * regular_hours + hourly_rate * 1.5 * overtime_hours) * days_worked = weekly_earnings ∧
    days_worked ≤ 7

theorem faith_works_five_days :
  faith_work_schedule 13.5 8 2 675 →
  ∃ (days_worked : ℕ), days_worked = 5 := by
  sorry

end NUMINAMATH_CALUDE_faith_works_five_days_l1934_193415


namespace NUMINAMATH_CALUDE_min_days_to_solve_100_problems_l1934_193431

/-- The number of problems solved on day n -/
def problems_solved (n : ℕ) : ℕ := 3^(n-1)

/-- The total number of problems solved up to day n -/
def total_problems (n : ℕ) : ℕ := (3^n - 1) / 2

theorem min_days_to_solve_100_problems :
  ∀ n : ℕ, n > 0 → (total_problems n ≥ 100 ↔ n ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_min_days_to_solve_100_problems_l1934_193431


namespace NUMINAMATH_CALUDE_cube_sum_equality_l1934_193406

theorem cube_sum_equality (a b : ℝ) (h : a + b = 4) : a^3 + 12*a*b + b^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l1934_193406


namespace NUMINAMATH_CALUDE_x_depends_on_m_and_n_l1934_193443

theorem x_depends_on_m_and_n (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃ (a b : ℝ → ℝ → ℝ), ∀ (x : ℝ),
    (x = a m n * m + b m n * n) →
    ((x + m)^3 - (x + n)^3 = (m - n)^3) →
    (a m n ≠ 1 ∨ b m n ≠ 1) ∧
    (a m n ≠ -1 ∨ b m n ≠ 1) ∧
    (a m n ≠ 1 ∨ b m n ≠ -1) ∧
    (a m n ≠ -1 ∨ b m n ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_x_depends_on_m_and_n_l1934_193443


namespace NUMINAMATH_CALUDE_trailing_zeros_factorial_100_l1934_193464

-- Define a function to count trailing zeros in factorial
def trailingZerosInFactorial (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

-- Theorem statement
theorem trailing_zeros_factorial_100 : trailingZerosInFactorial 100 = 24 := by
  sorry


end NUMINAMATH_CALUDE_trailing_zeros_factorial_100_l1934_193464


namespace NUMINAMATH_CALUDE_right_trapezoid_base_difference_l1934_193450

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- The length of the longer leg -/
  longer_leg : ℝ
  /-- The measure of the largest angle in degrees -/
  largest_angle : ℝ
  /-- The length of the longer base -/
  longer_base : ℝ
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- The longer leg is positive -/
  longer_leg_pos : longer_leg > 0
  /-- The largest angle is between 90° and 180° -/
  largest_angle_range : 90 < largest_angle ∧ largest_angle < 180
  /-- The longer base is longer than the shorter base -/
  base_order : longer_base > shorter_base

/-- The theorem stating the difference between bases of the specific right trapezoid -/
theorem right_trapezoid_base_difference (t : RightTrapezoid) 
    (h1 : t.longer_leg = 12)
    (h2 : t.largest_angle = 120) :
    t.longer_base - t.shorter_base = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_base_difference_l1934_193450


namespace NUMINAMATH_CALUDE_number_problem_l1934_193434

theorem number_problem (x : ℝ) : 0.6667 * x + 0.75 = 1.6667 → x = 1.375 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1934_193434


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1934_193441

/-- Given an equilateral triangle and an isosceles triangle sharing a side,
    prove that the base of the isosceles triangle is 25 units long. -/
theorem isosceles_triangle_base_length
  (equilateral_perimeter : ℝ)
  (isosceles_perimeter : ℝ)
  (h_equilateral : equilateral_perimeter = 60)
  (h_isosceles : isosceles_perimeter = 65)
  (h_shared_side : equilateral_perimeter / 3 = (isosceles_perimeter - isosceles_base) / 2) :
  isosceles_base = 25 :=
by
  sorry

#check isosceles_triangle_base_length

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1934_193441


namespace NUMINAMATH_CALUDE_students_not_in_same_row_or_column_l1934_193466

/-- Represents a student's position in a classroom --/
structure Position where
  row : ℕ
  column : ℕ

/-- Defines the seating arrangement for students A and B --/
def seating_arrangement : (Position × Position) :=
  (⟨3, 6⟩, ⟨6, 3⟩)

/-- Theorem stating that students A and B are not in the same row or column --/
theorem students_not_in_same_row_or_column :
  let (student_a, student_b) := seating_arrangement
  (student_a.row ≠ student_b.row) ∧ (student_a.column ≠ student_b.column) := by
  sorry

#check students_not_in_same_row_or_column

end NUMINAMATH_CALUDE_students_not_in_same_row_or_column_l1934_193466


namespace NUMINAMATH_CALUDE_bhupathi_amount_l1934_193414

theorem bhupathi_amount (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : A + B = 1210) (h4 : (4/15) * A = (2/5) * B) : B = 484 := by
  sorry

end NUMINAMATH_CALUDE_bhupathi_amount_l1934_193414


namespace NUMINAMATH_CALUDE_circle_passes_through_P_with_center_C_l1934_193492

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 29

-- Define the center point
def center : ℝ × ℝ := (3, 0)

-- Define the point P
def point_P : ℝ × ℝ := (-2, 2)

-- Theorem statement
theorem circle_passes_through_P_with_center_C :
  circle_equation point_P.1 point_P.2 ∧
  ∀ (x y : ℝ), circle_equation x y → 
    (x - center.1)^2 + (y - center.2)^2 = 
    (point_P.1 - center.1)^2 + (point_P.2 - center.2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_P_with_center_C_l1934_193492


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l1934_193449

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 70 / 100)
  (h3 : candidate_a_votes = 333200) :
  (total_votes - (candidate_a_votes / candidate_a_percentage)) / total_votes = 15 / 100 :=
by sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l1934_193449


namespace NUMINAMATH_CALUDE_compute_expression_l1934_193468

theorem compute_expression : 12 + 4 * (5 - 10)^3 = -488 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1934_193468


namespace NUMINAMATH_CALUDE_number_of_students_l1934_193477

theorem number_of_students (possible_outcomes : ℕ) (total_results : ℕ) : 
  possible_outcomes = 3 → total_results = 59049 → 
  ∃ n : ℕ, possible_outcomes ^ n = total_results ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_l1934_193477


namespace NUMINAMATH_CALUDE_divisibility_of_consecutive_numbers_l1934_193426

theorem divisibility_of_consecutive_numbers (n : ℕ) 
  (h1 : ∀ p : ℕ, Prime p → p ∣ n → p^2 ∣ n)
  (h2 : ∀ p : ℕ, Prime p → p ∣ (n + 1) → p^2 ∣ (n + 1))
  (h3 : ∀ p : ℕ, Prime p → p ∣ (n + 2) → p^2 ∣ (n + 2)) :
  ∃ p : ℕ, Prime p ∧ p^3 ∣ n :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_consecutive_numbers_l1934_193426


namespace NUMINAMATH_CALUDE_existence_of_unique_representation_sets_l1934_193494

-- Define the property of being an infinite set of non-negative integers
def IsInfiniteNonNegSet (S : Set ℕ) : Prop :=
  Set.Infinite S ∧ ∀ x ∈ S, x ≥ 0

-- Define the property that every non-negative integer has a unique representation
def HasUniqueRepresentation (A B : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ n = a + b

-- The main theorem
theorem existence_of_unique_representation_sets :
  ∃ A B : Set ℕ, IsInfiniteNonNegSet A ∧ IsInfiniteNonNegSet B ∧ HasUniqueRepresentation A B :=
sorry

end NUMINAMATH_CALUDE_existence_of_unique_representation_sets_l1934_193494


namespace NUMINAMATH_CALUDE_painted_cube_problem_l1934_193401

theorem painted_cube_problem (n : ℕ) : 
  n > 0 →  -- Ensure n is positive
  (2 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 6 → 
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_problem_l1934_193401


namespace NUMINAMATH_CALUDE_total_cards_l1934_193411

/-- The number of cards each person has -/
structure Cards where
  janet : ℕ
  brenda : ℕ
  mara : ℕ

/-- The conditions of the problem -/
def problem_conditions (c : Cards) : Prop :=
  c.janet = c.brenda + 9 ∧
  c.mara = 2 * c.janet ∧
  c.mara = 150 - 40

/-- The theorem to prove -/
theorem total_cards (c : Cards) (h : problem_conditions c) : 
  c.janet + c.brenda + c.mara = 211 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_l1934_193411


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l1934_193474

theorem right_triangle_leg_length (square_side : ℝ) (triangle_leg : ℝ) : 
  square_side = 1 →
  4 * (1/2 * triangle_leg * triangle_leg) = square_side * square_side →
  triangle_leg = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l1934_193474


namespace NUMINAMATH_CALUDE_increase_by_fraction_l1934_193456

theorem increase_by_fraction (initial : ℝ) (increase : ℚ) (result : ℝ) :
  initial = 120 →
  increase = 5/6 →
  result = initial * (1 + increase) →
  result = 220 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_fraction_l1934_193456


namespace NUMINAMATH_CALUDE_divide_plot_with_fences_l1934_193446

/-- Represents a rectangular plot with length and width -/
structure Plot where
  length : ℝ
  width : ℝ

/-- Represents a fence with a length -/
structure Fence where
  length : ℝ

/-- Represents a section of the plot -/
structure Section where
  area : ℝ

theorem divide_plot_with_fences (p : Plot) (f : Fence) :
  p.length = 80 →
  p.width = 50 →
  ∃ (sections : Finset Section),
    sections.card = 5 ∧
    (∀ s ∈ sections, s.area = (p.length * p.width) / 5) ∧
    f.length = 40 := by
  sorry

end NUMINAMATH_CALUDE_divide_plot_with_fences_l1934_193446


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1934_193475

/-- A quadratic function f(x) = ax² + bx + c -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties :
  ∃ (a b c : ℝ),
    (∀ x : ℝ, quadratic_function a b c (-1) = 0 ∧
              quadratic_function a b c (x + 1) - quadratic_function a b c x = 2 * x) →
    (∀ x : ℝ, quadratic_function a b c x = x^2 - x - 2) ∧
    (∀ x : ℝ, quadratic_function a b c x ≥ 0) ∧
    (∀ x : ℝ, quadratic_function a b c (x - 4) = quadratic_function a b c (2 - x)) ∧
    (∀ x : ℝ, 0 ≤ quadratic_function a b c x - x ∧
              quadratic_function a b c x - x ≤ (1/2) * (x - 1)^2) ∧
    a = 1/4 ∧ b = 1/2 ∧ c = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1934_193475


namespace NUMINAMATH_CALUDE_gcd_g_x_eq_one_l1934_193470

def g (x : ℤ) : ℤ := (3*x+4)*(9*x+5)*(17*x+11)*(x+17)

theorem gcd_g_x_eq_one (x : ℤ) (h : ∃ k : ℤ, x = 7263 * k) :
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_eq_one_l1934_193470


namespace NUMINAMATH_CALUDE_binary_101110110101_mod_8_l1934_193421

/-- Given a binary number represented as a list of bits (least significant bit first),
    calculate its value modulo 8. -/
def binary_mod_8 (bits : List Bool) : Nat :=
  match bits.take 3 with
  | [b0, b1, b2] => (if b0 then 1 else 0) + (if b1 then 2 else 0) + (if b2 then 4 else 0)
  | _ => 0

theorem binary_101110110101_mod_8 :
  binary_mod_8 [true, false, true, false, true, true, false, true, true, false, true, true] = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101110110101_mod_8_l1934_193421


namespace NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l1934_193416

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x ≤ x - a}
def B : Set ℝ := {x | 4*x - x^2 - 3 ≥ 0}

-- State the theorem
theorem union_equality_iff_a_in_range : 
  ∀ a : ℝ, (A a ∪ B = B) ↔ a ∈ Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l1934_193416


namespace NUMINAMATH_CALUDE_right_triangle_check_other_sets_not_right_triangle_l1934_193418

theorem right_triangle_check (a b c : ℝ) : 
  (a = 5 ∧ b = 12 ∧ c = 13) → a^2 + b^2 = c^2 :=
by sorry

theorem other_sets_not_right_triangle :
  ¬(∃ a b c : ℝ, 
    ((a = Real.sqrt 3 ∧ b = Real.sqrt 4 ∧ c = Real.sqrt 5) ∨
     (a = 4 ∧ b = 9 ∧ c = Real.sqrt 13) ∨
     (a = 0.8 ∧ b = 0.15 ∧ c = 0.17)) ∧
    a^2 + b^2 = c^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_check_other_sets_not_right_triangle_l1934_193418


namespace NUMINAMATH_CALUDE_sunflower_seed_contest_l1934_193424

theorem sunflower_seed_contest (player1 player2 player3 player4 total : ℕ) :
  player1 = 78 →
  player2 = 53 →
  player3 = player2 + 30 →
  player4 = 2 * player3 →
  total = player1 + player2 + player3 + player4 →
  total = 380 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_seed_contest_l1934_193424


namespace NUMINAMATH_CALUDE_m_greater_than_n_l1934_193452

theorem m_greater_than_n (a b m n : ℝ) 
  (ha : 0 < a) (hb : 0 < b)
  (h : m^2 * n^2 > a^2 * m^2 + b^2 * n^2) : 
  Real.sqrt (m^2 + n^2) > a + b := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l1934_193452


namespace NUMINAMATH_CALUDE_season_games_count_l1934_193405

/-- Represents a sports league with the given structure -/
structure SportsLeague where
  num_divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of games in a complete season -/
def total_games (league : SportsLeague) : Nat :=
  let total_teams := league.num_divisions * league.teams_per_division
  let intra_division_total := league.num_divisions * (league.teams_per_division * (league.teams_per_division - 1) / 2) * league.intra_division_games
  let inter_division_total := (total_teams * (total_teams - league.teams_per_division) / 2) * league.inter_division_games
  intra_division_total + inter_division_total

/-- The theorem to be proved -/
theorem season_games_count : 
  let league := SportsLeague.mk 3 6 3 2
  total_games league = 351 := by
  sorry

end NUMINAMATH_CALUDE_season_games_count_l1934_193405


namespace NUMINAMATH_CALUDE_valid_numbers_l1934_193448

def is_valid_number (n : ℕ) : Prop :=
  500 < n ∧ n < 2500 ∧ n % 180 = 0 ∧ n % 75 = 0

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n = 900 ∨ n = 1800 :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l1934_193448


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_condition_for_f_geq_g_l1934_193433

-- Define f(x) and g(x)
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |2*x - 1|
def g : ℝ → ℝ := λ x => 2

-- Theorem 1: Solution set when a = 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ g x} = {x : ℝ | x ≤ 2/3} := by sorry

-- Theorem 2: Condition for f(x) ≥ g(x) to always hold
theorem condition_for_f_geq_g (a : ℝ) :
  (∀ x : ℝ, f a x ≥ g x) ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_condition_for_f_geq_g_l1934_193433


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1934_193487

open Real

theorem function_inequality_implies_a_bound (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, x^2 * exp x > 3 * exp x + a) →
  a < exp 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1934_193487


namespace NUMINAMATH_CALUDE_average_weight_problem_l1934_193412

theorem average_weight_problem (total_boys : Nat) (group_a_boys : Nat) (group_b_boys : Nat)
  (group_b_avg_weight : ℝ) (total_avg_weight : ℝ) :
  total_boys = 34 →
  group_a_boys = 26 →
  group_b_boys = 8 →
  group_b_avg_weight = 45.15 →
  total_avg_weight = 49.05 →
  let group_a_avg_weight := (total_boys * total_avg_weight - group_b_boys * group_b_avg_weight) / group_a_boys
  group_a_avg_weight = 50.25 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1934_193412


namespace NUMINAMATH_CALUDE_mashed_potatoes_tomatoes_difference_l1934_193465

/-- The number of students who suggested mashed potatoes -/
def mashed_potatoes : ℕ := 144

/-- The number of students who suggested bacon -/
def bacon : ℕ := 467

/-- The number of students who suggested tomatoes -/
def tomatoes : ℕ := 79

/-- The difference between the number of students who suggested mashed potatoes and tomatoes -/
def difference : ℕ := mashed_potatoes - tomatoes

theorem mashed_potatoes_tomatoes_difference : difference = 65 := by sorry

end NUMINAMATH_CALUDE_mashed_potatoes_tomatoes_difference_l1934_193465


namespace NUMINAMATH_CALUDE_train_crossing_time_l1934_193444

/-- Prove that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 200 ∧ train_speed_kmh = 144 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1934_193444


namespace NUMINAMATH_CALUDE_hillary_stop_distance_l1934_193453

/-- Proves that Hillary stops 2900 feet short of the summit given the climbing conditions --/
theorem hillary_stop_distance (summit_distance : ℝ) (hillary_rate : ℝ) (eddy_rate : ℝ) 
  (hillary_descent_rate : ℝ) (climb_time : ℝ) 
  (h1 : summit_distance = 4700)
  (h2 : hillary_rate = 800)
  (h3 : eddy_rate = 500)
  (h4 : hillary_descent_rate = 1000)
  (h5 : climb_time = 6) :
  ∃ x : ℝ, x = 2900 ∧ 
  (summit_distance - x) + (eddy_rate * climb_time) + x = 
  summit_distance + (hillary_rate * climb_time - (summit_distance - x)) :=
by sorry

end NUMINAMATH_CALUDE_hillary_stop_distance_l1934_193453


namespace NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l1934_193407

theorem polar_to_cartesian_conversion :
  let r : ℝ := 2
  let θ : ℝ := π / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (Real.sqrt 3, 1) := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l1934_193407


namespace NUMINAMATH_CALUDE_afternoon_and_evening_emails_l1934_193436

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 5

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 8

/-- Theorem: The sum of emails Jack received in the afternoon and evening is 13 -/
theorem afternoon_and_evening_emails :
  afternoon_emails + evening_emails = 13 := by sorry

end NUMINAMATH_CALUDE_afternoon_and_evening_emails_l1934_193436


namespace NUMINAMATH_CALUDE_exists_polynomial_with_negative_coeff_positive_powers_l1934_193481

theorem exists_polynomial_with_negative_coeff_positive_powers :
  ∃ (P : Polynomial ℝ), 
    (∃ (i : ℕ), (P.coeff i) < 0) ∧ 
    (∀ (n : ℕ), n > 1 → ∀ (j : ℕ), ((P ^ n).coeff j) > 0) := by
  sorry

end NUMINAMATH_CALUDE_exists_polynomial_with_negative_coeff_positive_powers_l1934_193481


namespace NUMINAMATH_CALUDE_problem_statement_l1934_193432

theorem problem_statement (a b : ℝ) : (2*a + b)^2 + |b - 2| = 0 → (-a - b)^2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1934_193432


namespace NUMINAMATH_CALUDE_max_value_trigonometric_function_l1934_193499

theorem max_value_trigonometric_function :
  ∀ θ : ℝ, 0 < θ → θ < π / 2 →
  (∀ φ : ℝ, 0 < φ → φ < π / 2 →
    (1 / Real.sin θ - 1) * (1 / Real.cos θ - 1) ≥ (1 / Real.sin φ - 1) * (1 / Real.cos φ - 1)) →
  (1 / Real.sin θ - 1) * (1 / Real.cos θ - 1) = 3 - 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_max_value_trigonometric_function_l1934_193499


namespace NUMINAMATH_CALUDE_class_composition_l1934_193427

theorem class_composition (total_students : ℕ) (total_planes : ℕ) (girls_planes : ℕ) (boys_planes : ℕ) 
  (h1 : total_students = 21)
  (h2 : total_planes = 69)
  (h3 : girls_planes = 2)
  (h4 : boys_planes = 5) :
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧
    boys * boys_planes + girls * girls_planes = total_planes ∧
    boys = 9 ∧
    girls = 12 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_l1934_193427


namespace NUMINAMATH_CALUDE_base_8_246_to_base_10_l1934_193410

def base_8_to_10 (a b c : ℕ) : ℕ := a * 8^2 + b * 8^1 + c * 8^0

theorem base_8_246_to_base_10 : base_8_to_10 2 4 6 = 166 := by
  sorry

end NUMINAMATH_CALUDE_base_8_246_to_base_10_l1934_193410


namespace NUMINAMATH_CALUDE_max_remainder_eleven_l1934_193451

theorem max_remainder_eleven (A B C : ℕ) (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h2 : A = 11 * B + C) : C ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_remainder_eleven_l1934_193451


namespace NUMINAMATH_CALUDE_bobby_sarah_fish_ratio_l1934_193457

/-- The number of fish in each person's aquarium and their relationships -/
structure FishCounts where
  billy : ℕ
  tony : ℕ
  sarah : ℕ
  bobby : ℕ
  billy_count : billy = 10
  tony_count : tony = 3 * billy
  sarah_count : sarah = tony + 5
  total_count : billy + tony + sarah + bobby = 145

/-- The ratio of fish in Bobby's aquarium to Sarah's aquarium -/
def fish_ratio (fc : FishCounts) : ℚ :=
  fc.bobby / fc.sarah

/-- Theorem stating that the ratio of fish in Bobby's aquarium to Sarah's aquarium is 2:1 -/
theorem bobby_sarah_fish_ratio (fc : FishCounts) : fish_ratio fc = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_bobby_sarah_fish_ratio_l1934_193457


namespace NUMINAMATH_CALUDE_rational_operations_l1934_193493

-- Define the new operation
def star (x y : ℚ) : ℚ := x + y - x * y

-- Theorem statement
theorem rational_operations :
  -- Unit elements
  (∀ a : ℚ, a + 0 = a) ∧
  (∀ a : ℚ, a * 1 = a) ∧
  -- Inverse element of 3 under addition
  (3 + (-3) = 0) ∧
  -- 0 has no multiplicative inverse
  (∀ x : ℚ, x ≠ 0 → ∃ y : ℚ, x * y = 1) ∧
  -- Properties of the new operation
  (∀ x : ℚ, star x 0 = x) ∧
  (∀ m : ℚ, m ≠ 1 → star m (m / (m - 1)) = 0) :=
by sorry

end NUMINAMATH_CALUDE_rational_operations_l1934_193493


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l1934_193498

/-- A quadrilateral inscribed in a circle with given side lengths --/
structure InscribedQuadrilateral where
  radius : ℝ
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- The theorem stating the relationship between the sides of the inscribed quadrilateral --/
theorem inscribed_quadrilateral_fourth_side 
  (q : InscribedQuadrilateral) 
  (h1 : q.radius = 100 * Real.sqrt 3)
  (h2 : q.side1 = 100)
  (h3 : q.side2 = 150)
  (h4 : q.side3 = 200) :
  q.side4^2 = 35800 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l1934_193498


namespace NUMINAMATH_CALUDE_unique_coin_configuration_l1934_193495

/-- Represents the different types of coins -/
inductive CoinType
  | Penny
  | Nickel
  | Dime

/-- The value of each coin type in cents -/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10

/-- A configuration of coins -/
structure CoinConfiguration where
  pennies : Nat
  nickels : Nat
  dimes : Nat

/-- The total number of coins in a configuration -/
def CoinConfiguration.totalCoins (c : CoinConfiguration) : Nat :=
  c.pennies + c.nickels + c.dimes

/-- The total value of coins in a configuration in cents -/
def CoinConfiguration.totalValue (c : CoinConfiguration) : Nat :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime

/-- Theorem: There is a unique coin configuration with 8 coins, 53 cents total value,
    and at least one of each coin type, which must have exactly 3 nickels -/
theorem unique_coin_configuration :
  ∃! c : CoinConfiguration,
    c.totalCoins = 8 ∧
    c.totalValue = 53 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    c.nickels = 3 := by
  sorry


end NUMINAMATH_CALUDE_unique_coin_configuration_l1934_193495


namespace NUMINAMATH_CALUDE_lines_parallel_in_plane_l1934_193480

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contained_in : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (coplanar : Line → Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_in_plane 
  (m n : Line) (α β : Plane) :
  m ≠ n →  -- m and n are distinct
  α ≠ β →  -- α and β are distinct
  contained_in m α →
  parallel n α →
  coplanar m n β →
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_in_plane_l1934_193480


namespace NUMINAMATH_CALUDE_square_binomial_coefficient_l1934_193490

theorem square_binomial_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 20 * x + 16 = (r * x + s)^2) → 
  a = 25 / 4 :=
by sorry

end NUMINAMATH_CALUDE_square_binomial_coefficient_l1934_193490


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l1934_193463

theorem cone_sphere_ratio (r : ℝ) (h : ℝ) (R : ℝ) : 
  R = 2 * r →
  (1/3) * (4/3) * Real.pi * r^3 = (1/3) * Real.pi * R^2 * h →
  h / R = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l1934_193463


namespace NUMINAMATH_CALUDE_equidistant_points_on_line_in_quadrants_I_and_IV_l1934_193408

/-- The line equation 4x + 3y = 12 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y = 12

/-- A point (x, y) is equidistant from coordinate axes if |x| = |y| -/
def equidistant_from_axes (x y : ℝ) : Prop := abs x = abs y

/-- Quadrant I: x > 0 and y > 0 -/
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Quadrant IV: x > 0 and y < 0 -/
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The main theorem: points on the line 4x + 3y = 12 that are equidistant 
    from coordinate axes exist only in quadrants I and IV -/
theorem equidistant_points_on_line_in_quadrants_I_and_IV :
  ∀ x y : ℝ, line_equation x y → equidistant_from_axes x y →
  (in_quadrant_I x y ∨ in_quadrant_IV x y) ∧
  ¬(∃ x y : ℝ, line_equation x y ∧ equidistant_from_axes x y ∧ 
    ¬(in_quadrant_I x y ∨ in_quadrant_IV x y)) :=
sorry

end NUMINAMATH_CALUDE_equidistant_points_on_line_in_quadrants_I_and_IV_l1934_193408


namespace NUMINAMATH_CALUDE_bees_count_l1934_193471

theorem bees_count (first_day_count : ℕ) (second_day_multiplier : ℕ) : first_day_count = 144 → second_day_multiplier = 3 → first_day_count * second_day_multiplier = 432 := by
  sorry

end NUMINAMATH_CALUDE_bees_count_l1934_193471


namespace NUMINAMATH_CALUDE_sum_difference_multiples_l1934_193467

theorem sum_difference_multiples (m n : ℕ+) : 
  (∃ x : ℕ+, m = 101 * x) → 
  (∃ y : ℕ+, n = 63 * y) → 
  m + n = 2018 → 
  m - n = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_difference_multiples_l1934_193467


namespace NUMINAMATH_CALUDE_coin_game_probability_l1934_193479

/-- Represents a player in the coin game -/
inductive Player := | Abby | Bernardo | Carl | Debra

/-- Represents a ball color in the game -/
inductive BallColor := | Green | Red | Blue

/-- The number of rounds in the game -/
def numRounds : Nat := 5

/-- The number of coins each player starts with -/
def initialCoins : Nat := 5

/-- The number of balls of each color in the urn -/
def ballCounts : Fin 3 → Nat
  | 0 => 2  -- Green
  | 1 => 2  -- Red
  | 2 => 1  -- Blue

/-- Represents the state of the game after each round -/
structure GameState where
  coins : Player → Nat
  round : Nat

/-- Represents a single round of the game -/
def gameRound (state : GameState) : GameState := sorry

/-- The probability of a specific outcome in a single round -/
def roundProbability (outcome : Player → BallColor) : Rat := sorry

/-- The probability of returning to the initial state after all rounds -/
def finalProbability : Rat := sorry

/-- The main theorem stating the probability of each player having 5 coins at the end -/
theorem coin_game_probability : finalProbability = 64 / 15625 := sorry

end NUMINAMATH_CALUDE_coin_game_probability_l1934_193479


namespace NUMINAMATH_CALUDE_complex_roots_theorem_l1934_193447

theorem complex_roots_theorem (p q r : ℂ) 
  (sum_eq : p + q + r = -1)
  (sum_prod_eq : p * q + p * r + q * r = -1)
  (prod_eq : p * q * r = -1) :
  (({p, q, r} : Set ℂ) = {-1, Complex.I, -Complex.I}) := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_theorem_l1934_193447


namespace NUMINAMATH_CALUDE_smallest_leftover_four_boxes_l1934_193489

/-- The number of kids among whom the Snackies are distributed -/
def num_kids : ℕ := 8

/-- The number of Snackies left over when one box is divided among the kids -/
def leftover_one_box : ℕ := 5

/-- The number of boxes used in the final distribution -/
def num_boxes : ℕ := 4

/-- Represents the number of Snackies in one box -/
def snackies_per_box : ℕ := num_kids * leftover_one_box + leftover_one_box

theorem smallest_leftover_four_boxes :
  ∃ (leftover : ℕ), leftover < num_kids ∧
  ∃ (pieces_per_kid : ℕ),
    num_boxes * snackies_per_box = num_kids * pieces_per_kid + leftover ∧
    ∀ (smaller_leftover : ℕ),
      smaller_leftover < leftover →
      ¬∃ (alt_pieces_per_kid : ℕ),
        num_boxes * snackies_per_box = num_kids * alt_pieces_per_kid + smaller_leftover :=
by sorry

end NUMINAMATH_CALUDE_smallest_leftover_four_boxes_l1934_193489


namespace NUMINAMATH_CALUDE_valid_attachments_count_l1934_193417

/-- Represents a square in our figure -/
structure Square

/-- Represents the cross-shaped figure -/
structure CrossFigure where
  center : Square
  extensions : Fin 4 → Square

/-- Represents a position where an extra square can be attached -/
inductive AttachmentPosition
  | TopOfExtension (i : Fin 4)
  | Other

/-- Represents the resulting figure after attaching an extra square -/
structure ResultingFigure where
  base : CrossFigure
  extraSquare : Square
  position : AttachmentPosition

/-- Predicate to check if a resulting figure can be folded into a topless square pyramid -/
def canFoldIntoPyramid (fig : ResultingFigure) : Prop :=
  match fig.position with
  | AttachmentPosition.TopOfExtension _ => True
  | AttachmentPosition.Other => False

/-- The main theorem to prove -/
theorem valid_attachments_count :
  ∃ (validPositions : Finset AttachmentPosition),
    (∀ pos, pos ∈ validPositions ↔ ∃ fig : ResultingFigure, fig.position = pos ∧ canFoldIntoPyramid fig) ∧
    Finset.card validPositions = 4 := by
  sorry

end NUMINAMATH_CALUDE_valid_attachments_count_l1934_193417


namespace NUMINAMATH_CALUDE_greatest_x_lcm_l1934_193483

theorem greatest_x_lcm (x : ℕ) : 
  (Nat.lcm x (Nat.lcm 10 14) = 70) → x ≤ 70 ∧ ∃ y : ℕ, y = 70 ∧ Nat.lcm y (Nat.lcm 10 14) = 70 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_lcm_l1934_193483


namespace NUMINAMATH_CALUDE_angle_ENG_is_45_degrees_l1934_193484

-- Define the rectangle EFGH
structure Rectangle where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

-- Define the properties of the rectangle
def is_valid_rectangle (rect : Rectangle) : Prop :=
  rect.E.1 = 0 ∧ rect.E.2 = 0 ∧
  rect.F.1 = 8 ∧ rect.F.2 = 0 ∧
  rect.G.1 = 8 ∧ rect.G.2 = 4 ∧
  rect.H.1 = 0 ∧ rect.H.2 = 4

-- Define point N on side EF
def N : ℝ × ℝ := (4, 0)

-- Define the property that triangle ENG is isosceles
def is_isosceles_ENG (rect : Rectangle) : Prop :=
  let EN := ((N.1 - rect.E.1)^2 + (N.2 - rect.E.2)^2).sqrt
  let NG := ((rect.G.1 - N.1)^2 + (rect.G.2 - N.2)^2).sqrt
  EN = NG

-- Theorem statement
theorem angle_ENG_is_45_degrees (rect : Rectangle) 
  (h1 : is_valid_rectangle rect) 
  (h2 : is_isosceles_ENG rect) : 
  Real.arctan 1 = 45 * (π / 180) :=
sorry

end NUMINAMATH_CALUDE_angle_ENG_is_45_degrees_l1934_193484


namespace NUMINAMATH_CALUDE_unique_element_in_A_not_in_B_l1934_193423

-- Define the sets A and B
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {2, 4, 6}

-- State the theorem
theorem unique_element_in_A_not_in_B :
  ∀ x : ℕ, x ∈ A ∧ x ∉ B → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_element_in_A_not_in_B_l1934_193423


namespace NUMINAMATH_CALUDE_part1_part2_l1934_193482

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the set A
def A (a b c : ℝ) : Set ℝ := {x | f a b c x = x}

-- Part 1
theorem part1 (a b c : ℝ) :
  A a b c = {1, 2} → f a b c 0 = 2 →
  ∃ (M m : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x ≤ M ∧ m ≤ f a b c x) ∧ M = 10 ∧ m = 1 :=
sorry

-- Part 2
theorem part2 (a b c : ℝ) :
  A a b c = {2} → a ≥ 1 →
  ∃ (g : ℝ → ℝ), (∀ a' ≥ 1, g a' ≥ 63/4) ∧
    (∃ (M m : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a b c x ≤ M ∧ m ≤ f a b c x) ∧ g a = M + m) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l1934_193482


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1934_193425

/-- Given an ellipse and a circle satisfying certain conditions, prove that the eccentricity of the ellipse is 1/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (x y : ℝ), (x - 2)^2 + y^2 = 1 ∧ 
   ((x^2 / a^2 + y^2 / b^2 = 1 ∧ x^2 + y^2 = a^2) ∨ 
    (x^2 / a^2 + y^2 / b^2 = 1 ∧ x^2 - c^2 = a^2 - c^2 ∧ c^2 = a^2 - b^2))) →
  (a^2 - b^2) / a = 1/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1934_193425


namespace NUMINAMATH_CALUDE_origin_fixed_under_dilation_l1934_193403

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square defined by its four vertices -/
structure Square where
  s : Point
  t : Point
  u : Point
  v : Point

/-- Defines a dilation transformation -/
def dilation (center : Point) (k : ℝ) (p : Point) : Point :=
  { x := center.x + k * (p.x - center.x)
  , y := center.y + k * (p.y - center.y) }

theorem origin_fixed_under_dilation (original : Square) (dilated : Square) :
  original.s = Point.mk 3 3 ∧
  original.t = Point.mk 7 3 ∧
  original.u = Point.mk 7 7 ∧
  original.v = Point.mk 3 7 ∧
  dilated.s = Point.mk 6 6 ∧
  dilated.t = Point.mk 12 6 ∧
  dilated.u = Point.mk 12 12 ∧
  dilated.v = Point.mk 6 12 →
  ∃ (k : ℝ), ∀ (p : Point),
    dilation (Point.mk 0 0) k original.s = dilated.s ∧
    dilation (Point.mk 0 0) k original.t = dilated.t ∧
    dilation (Point.mk 0 0) k original.u = dilated.u ∧
    dilation (Point.mk 0 0) k original.v = dilated.v :=
by sorry

end NUMINAMATH_CALUDE_origin_fixed_under_dilation_l1934_193403


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1934_193476

-- Define the parabola function
def f (x : ℝ) : ℝ := 2 * x^2

-- Define the slope of the line parallel to 4x - y + 3 = 0
def m : ℝ := 4

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the parabola
    y₀ = f x₀ ∧
    -- The slope of the tangent line at (x₀, y₀) is m
    (deriv f) x₀ = m ∧
    -- The equation of the tangent line is 4x - y - 2 = 0
    ∀ (x y : ℝ), y - y₀ = m * (x - x₀) ↔ 4 * x - y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1934_193476


namespace NUMINAMATH_CALUDE_three_digit_not_mult_4_or_6_eq_600_l1934_193461

/-- The number of three-digit numbers that are multiples of neither 4 nor 6 -/
def three_digit_not_mult_4_or_6 : ℕ :=
  let three_digit_count := 999 - 100 + 1
  let mult_4_count := (996 / 4) - (100 / 4) + 1
  let mult_6_count := (996 / 6) - (102 / 6) + 1
  let mult_12_count := (996 / 12) - (108 / 12) + 1
  three_digit_count - (mult_4_count + mult_6_count - mult_12_count)

theorem three_digit_not_mult_4_or_6_eq_600 :
  three_digit_not_mult_4_or_6 = 600 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_not_mult_4_or_6_eq_600_l1934_193461


namespace NUMINAMATH_CALUDE_model_airplane_competition_l1934_193496

/-- Represents a model airplane -/
structure ModelAirplane where
  speed : ℝ
  flightTime : ℝ

/-- Theorem about model airplane competition -/
theorem model_airplane_competition 
  (m h c : ℝ) 
  (model1 model2 : ModelAirplane) 
  (h_positive : h > 0)
  (m_positive : m > 0)
  (c_positive : c > 0)
  (time_diff : model2.flightTime = model1.flightTime + m)
  (headwind_distance : 
    (model1.speed - c) * model1.flightTime = 
    (model2.speed - c) * model2.flightTime + h) :
  (h > c * m → 
    model1.speed * model1.flightTime > 
    model2.speed * model2.flightTime) ∧
  (h < c * m → 
    model1.speed * model1.flightTime < 
    model2.speed * model2.flightTime) ∧
  (h = c * m → 
    model1.speed * model1.flightTime = 
    model2.speed * model2.flightTime) := by
  sorry

end NUMINAMATH_CALUDE_model_airplane_competition_l1934_193496


namespace NUMINAMATH_CALUDE_cistern_filling_time_l1934_193439

theorem cistern_filling_time (p q : ℝ) (h1 : p > 0) (h2 : q > 0) : 
  p = 1 / 12 → q = 1 / 15 → 
  let combined_rate := p + q
  let filled_portion := 4 * combined_rate
  let remaining_portion := 1 - filled_portion
  remaining_portion / q = 6 := by sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l1934_193439


namespace NUMINAMATH_CALUDE_original_number_property_l1934_193469

theorem original_number_property (k : ℕ) : ∃ (N : ℕ), N = 23 * k + 22 ∧ (N + 1) % 23 = 0 := by
  sorry

end NUMINAMATH_CALUDE_original_number_property_l1934_193469


namespace NUMINAMATH_CALUDE_gcd_1987_1463_l1934_193435

theorem gcd_1987_1463 : Nat.gcd 1987 1463 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1987_1463_l1934_193435


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l1934_193460

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 2

-- State the theorem
theorem f_sum_symmetric (a b m : ℝ) : 
  f a b (-2) = m → f a b 2 + f a b (-2) = -4 := by
sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l1934_193460


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1934_193454

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1934_193454


namespace NUMINAMATH_CALUDE_scientific_notation_12000_l1934_193422

theorem scientific_notation_12000 : 
  12000 = 1.2 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_12000_l1934_193422


namespace NUMINAMATH_CALUDE_point_coordinates_l1934_193486

theorem point_coordinates (m n : ℝ) : (m + 3)^2 + Real.sqrt (4 - n) = 0 → m = -3 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1934_193486


namespace NUMINAMATH_CALUDE_husband_catch_up_time_and_distance_l1934_193491

-- Define the problem parameters
def yolanda_initial_speed : ℝ := 20
def yolanda_second_speed : ℝ := 22
def yolanda_final_speed : ℝ := 18
def yolanda_first_distance : ℝ := 5
def yolanda_second_distance : ℝ := 8
def yolanda_third_distance : ℝ := 7
def yolanda_stop_time : ℝ := 12
def husband_speed : ℝ := 40
def husband_delay : ℝ := 15
def route_difference : ℝ := 10

-- Define the theorem
theorem husband_catch_up_time_and_distance : 
  let yolanda_total_distance := yolanda_first_distance + yolanda_second_distance + yolanda_third_distance
  let husband_distance := yolanda_total_distance - route_difference
  let yolanda_travel_time := yolanda_first_distance / yolanda_initial_speed * 60 + 
                             yolanda_second_distance / yolanda_second_speed * 60 + 
                             yolanda_third_distance / yolanda_final_speed * 60 + 
                             yolanda_stop_time
  let husband_travel_time := husband_distance / husband_speed * 60
  husband_distance = 10 ∧ husband_travel_time + husband_delay = 30 := by
    sorry


end NUMINAMATH_CALUDE_husband_catch_up_time_and_distance_l1934_193491


namespace NUMINAMATH_CALUDE_original_denominator_proof_l1934_193437

theorem original_denominator_proof (d : ℚ) : 
  (3 : ℚ) / d ≠ (1 : ℚ) / 3 ∧ (3 + 7 : ℚ) / (d + 7) = (1 : ℚ) / 3 → d = 23 := by
  sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l1934_193437


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l1934_193400

/-- Represents a rectangle on a grid --/
structure Rectangle where
  width : ℕ
  height : ℕ
  is_not_square : width ≠ height

/-- Represents the configuration of rectangles in the problem --/
structure Configuration where
  abcd : Rectangle
  qrsc : Rectangle
  ap : ℕ
  qr : ℕ
  bp : ℕ
  br : ℕ
  sc : ℕ

/-- The main theorem statement --/
theorem shaded_area_theorem (config : Configuration) :
  config.abcd.width * config.abcd.height = 35 →
  config.ap < config.qr →
  (config.abcd.width * config.abcd.height - 
   (config.bp * config.br + config.ap * config.sc) = 24) ∨
  (config.abcd.width * config.abcd.height - 
   (config.bp * config.br + config.ap * config.sc) = 26) := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l1934_193400


namespace NUMINAMATH_CALUDE_only_valid_k_values_l1934_193419

/-- Represents a line in the form y = kx + b --/
structure Line where
  k : ℤ
  b : ℤ

/-- Represents a parabola in the form y = a(x - c)² --/
structure Parabola where
  a : ℤ
  c : ℤ

/-- Checks if a given k value satisfies all conditions --/
def is_valid_k (k : ℤ) : Prop :=
  ∃ (b : ℤ) (a c : ℤ),
    -- Line passes through (-1, 2020)
    2020 = -k + b ∧
    -- Parabola vertex is on the line
    c = -1 - 2020 / k ∧
    -- a is an integer
    a = k^2 / (2020 + k) ∧
    -- k is negative
    k < 0

/-- The main theorem stating that only -404 and -1010 are valid k values --/
theorem only_valid_k_values :
  ∀ k : ℤ, is_valid_k k ↔ k = -404 ∨ k = -1010 := by sorry

end NUMINAMATH_CALUDE_only_valid_k_values_l1934_193419


namespace NUMINAMATH_CALUDE_annual_savings_l1934_193402

/-- Given monthly income and expenses, calculate annual savings --/
theorem annual_savings (monthly_income monthly_expenses : ℕ) : 
  monthly_income = 5000 → 
  monthly_expenses = 4600 → 
  (monthly_income - monthly_expenses) * 12 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_annual_savings_l1934_193402


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1934_193420

theorem absolute_value_inequality (x : ℝ) :
  (1 < |x - 1| ∧ |x - 1| < 4) ↔ ((-3 < x ∧ x < 0) ∨ (2 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1934_193420


namespace NUMINAMATH_CALUDE_not_less_than_negative_double_l1934_193409

theorem not_less_than_negative_double {x y : ℝ} (h : x < y) : ¬(-2 * x < -2 * y) := by
  sorry

end NUMINAMATH_CALUDE_not_less_than_negative_double_l1934_193409


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1934_193445

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ, 2^x * 3^y + 1 = 7^z →
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 2)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1934_193445


namespace NUMINAMATH_CALUDE_ceramic_firing_probabilities_l1934_193473

/-- Represents the probability of success for a craft in each firing process -/
structure CraftProbabilities where
  first : Float
  second : Float

/-- Calculates the probability of exactly one success out of three independent events -/
def probExactlyOne (p1 p2 p3 : Float) : Float :=
  p1 * (1 - p2) * (1 - p3) + (1 - p1) * p2 * (1 - p3) + (1 - p1) * (1 - p2) * p3

/-- Calculates the expected value of a binomial distribution -/
def binomialExpectedValue (n : Nat) (p : Float) : Float :=
  n.toFloat * p

/-- Theorem about ceramic firing probabilities -/
theorem ceramic_firing_probabilities
  (craftA craftB craftC : CraftProbabilities)
  (h1 : craftA.first = 0.5)
  (h2 : craftB.first = 0.6)
  (h3 : craftC.first = 0.4)
  (h4 : craftA.second = 0.6)
  (h5 : craftB.second = 0.5)
  (h6 : craftC.second = 0.75) :
  (probExactlyOne craftA.first craftB.first craftC.first = 0.38) ∧
  (binomialExpectedValue 3 (craftA.first * craftA.second) = 0.9) := by
  sorry


end NUMINAMATH_CALUDE_ceramic_firing_probabilities_l1934_193473


namespace NUMINAMATH_CALUDE_first_divisor_problem_l1934_193472

theorem first_divisor_problem (x : ℚ) : 
  ((377 / x) / 29) * (1/4) / 2 = 0.125 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l1934_193472


namespace NUMINAMATH_CALUDE_expression_evaluation_l1934_193485

theorem expression_evaluation :
  let x : ℝ := 2
  (x + 1) * (x - 1) + x * (3 - x) = 5 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1934_193485


namespace NUMINAMATH_CALUDE_exponential_monotonicity_l1934_193497

theorem exponential_monotonicity (a b : ℝ) : a < b → (2 : ℝ) ^ a < (2 : ℝ) ^ b := by
  sorry

end NUMINAMATH_CALUDE_exponential_monotonicity_l1934_193497


namespace NUMINAMATH_CALUDE_slope_of_line_parallel_lines_solution_l1934_193440

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope of a line in the form ax + by + c = 0 is -a/b -/
theorem slope_of_line (a b c : ℝ) (hb : b ≠ 0) :
  ∀ x y, a * x + b * y + c = 0 ↔ y = (-a/b) * x + (-c/b) :=
  sorry

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y, a * x + y - 4 = 0 ↔ x + (a + 3/2) * y + 2 = 0) → a = 1/2 :=
  sorry

end NUMINAMATH_CALUDE_slope_of_line_parallel_lines_solution_l1934_193440


namespace NUMINAMATH_CALUDE_sum_of_squares_l1934_193442

theorem sum_of_squares (a b c : ℝ) 
  (h1 : a * b + b * c + a * c = 116) 
  (h2 : a + b + c = 22) : 
  a^2 + b^2 + c^2 = 252 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1934_193442
