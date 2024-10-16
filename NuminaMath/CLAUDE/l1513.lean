import Mathlib

namespace NUMINAMATH_CALUDE_area_at_stage_8_l1513_151324

/-- The side length of each square -/
def squareSide : ℕ := 4

/-- The area of each square -/
def squareArea : ℕ := squareSide * squareSide

/-- The number of squares at a given stage -/
def numSquaresAtStage (stage : ℕ) : ℕ := stage

/-- The total area of the rectangle at a given stage -/
def totalAreaAtStage (stage : ℕ) : ℕ := numSquaresAtStage stage * squareArea

/-- The theorem stating that the area of the rectangle at Stage 8 is 128 square inches -/
theorem area_at_stage_8 : totalAreaAtStage 8 = 128 := by sorry

end NUMINAMATH_CALUDE_area_at_stage_8_l1513_151324


namespace NUMINAMATH_CALUDE_fraction_equality_l1513_151365

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / (a + 35 : ℚ) = 7/8 → a = 245 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1513_151365


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l1513_151325

theorem fraction_subtraction_equality : (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l1513_151325


namespace NUMINAMATH_CALUDE_sum_special_numbers_largest_odd_two_digit_correct_smallest_even_three_digit_correct_l1513_151363

/-- The largest odd number less than 100 -/
def largest_odd_two_digit : ℕ :=
  99

/-- The smallest even number greater than or equal to 100 -/
def smallest_even_three_digit : ℕ :=
  100

/-- Theorem stating the sum of the largest odd two-digit number
    and the smallest even three-digit number -/
theorem sum_special_numbers :
  largest_odd_two_digit + smallest_even_three_digit = 199 := by
  sorry

/-- Proof that largest_odd_two_digit is indeed the largest odd number less than 100 -/
theorem largest_odd_two_digit_correct :
  largest_odd_two_digit < 100 ∧
  largest_odd_two_digit % 2 = 1 ∧
  ∀ n : ℕ, n < 100 → n % 2 = 1 → n ≤ largest_odd_two_digit := by
  sorry

/-- Proof that smallest_even_three_digit is indeed the smallest even number ≥ 100 -/
theorem smallest_even_three_digit_correct :
  smallest_even_three_digit ≥ 100 ∧
  smallest_even_three_digit % 2 = 0 ∧
  ∀ n : ℕ, n ≥ 100 → n % 2 = 0 → n ≥ smallest_even_three_digit := by
  sorry

end NUMINAMATH_CALUDE_sum_special_numbers_largest_odd_two_digit_correct_smallest_even_three_digit_correct_l1513_151363


namespace NUMINAMATH_CALUDE_set_relationship_l1513_151381

-- Define the sets M, N, and P
def M : Set ℝ := {x | (x + 3) / (x - 1) ≤ 0 ∧ x ≠ 1}
def N : Set ℝ := {x | |x + 1| ≤ 2}
def P : Set ℝ := {x | (1/2 : ℝ)^(x^2 + 2*x - 3) ≥ 1}

-- State the theorem
theorem set_relationship : M ⊆ N ∧ N = P := by sorry

end NUMINAMATH_CALUDE_set_relationship_l1513_151381


namespace NUMINAMATH_CALUDE_matias_grade_size_l1513_151392

/-- Given a student's rank from best and worst in a group, calculate the total number of students -/
def totalStudents (rankBest : ℕ) (rankWorst : ℕ) : ℕ :=
  (rankBest - 1) + (rankWorst - 1) + 1

/-- Theorem: In a group where a student is both the 75th best and 75th worst, there are 149 students -/
theorem matias_grade_size :
  totalStudents 75 75 = 149 := by
  sorry

#eval totalStudents 75 75

end NUMINAMATH_CALUDE_matias_grade_size_l1513_151392


namespace NUMINAMATH_CALUDE_inequality_proof_l1513_151353

theorem inequality_proof (x y : ℝ) : 
  -1/2 ≤ (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ∧ 
  (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ≤ 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1513_151353


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l1513_151377

/-- A digit in base d is a natural number less than d. -/
def Digit (d : ℕ) := { n : ℕ // n < d }

/-- The value of a two-digit number AB in base d. -/
def TwoDigitValue (d : ℕ) (A B : Digit d) : ℕ := A.val * d + B.val

theorem digit_difference_in_base_d (d : ℕ) (A B : Digit d) 
  (h_d : d > 7)
  (h_sum : TwoDigitValue d A B + TwoDigitValue d A A = 1 * d * d + 7 * d + 2) :
  A.val - B.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l1513_151377


namespace NUMINAMATH_CALUDE_height_comparison_l1513_151327

theorem height_comparison (a b : ℝ) (h : a = 0.8 * b) : b = 1.25 * a := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l1513_151327


namespace NUMINAMATH_CALUDE_evening_screen_time_l1513_151320

-- Define the total recommended screen time in hours
def total_screen_time_hours : ℕ := 2

-- Define the screen time already used in minutes
def morning_screen_time : ℕ := 45

-- Define the function to calculate remaining screen time
def remaining_screen_time (total_hours : ℕ) (used_minutes : ℕ) : ℕ :=
  total_hours * 60 - used_minutes

-- Theorem statement
theorem evening_screen_time :
  remaining_screen_time total_screen_time_hours morning_screen_time = 75 := by
  sorry

end NUMINAMATH_CALUDE_evening_screen_time_l1513_151320


namespace NUMINAMATH_CALUDE_f_shape_perimeter_l1513_151364

/-- The perimeter of a shape formed by two rectangles arranged in an F shape -/
def f_perimeter (h1 w1 h2 w2 overlap_h overlap_w : ℝ) : ℝ :=
  2 * (h1 + w1) + 2 * (h2 + w2) - 2 * overlap_w

/-- Theorem: The perimeter of the F shape is 18 inches -/
theorem f_shape_perimeter :
  f_perimeter 5 3 1 5 1 3 = 18 := by
  sorry

#eval f_perimeter 5 3 1 5 1 3

end NUMINAMATH_CALUDE_f_shape_perimeter_l1513_151364


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l1513_151329

theorem number_puzzle_solution (A B C : ℤ) 
  (sum_eq : A + B = 44)
  (ratio_eq : 5 * A = 6 * B)
  (diff_eq : C = 2 * (A - B)) :
  A = 24 ∧ B = 20 ∧ C = 8 := by
sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l1513_151329


namespace NUMINAMATH_CALUDE_tape_overlap_l1513_151321

theorem tape_overlap (tape_length : ℕ) (total_length : ℕ) (h1 : tape_length = 275) (h2 : total_length = 512) :
  2 * tape_length - total_length = 38 := by
  sorry

end NUMINAMATH_CALUDE_tape_overlap_l1513_151321


namespace NUMINAMATH_CALUDE_company_production_l1513_151351

/-- The number of bottles a case can hold -/
def bottles_per_case : ℕ := 13

/-- The number of cases required for one-day production -/
def cases_per_day : ℕ := 5000

/-- The total number of bottles produced in one day -/
def bottles_per_day : ℕ := bottles_per_case * cases_per_day

/-- Theorem stating that the company produces 65,000 bottles per day -/
theorem company_production : bottles_per_day = 65000 := by
  sorry

end NUMINAMATH_CALUDE_company_production_l1513_151351


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l1513_151360

theorem rectangle_area_equals_perimeter (x : ℝ) : 
  (4 * x) * (x + 4) = 2 * (4 * x) + 2 * (x + 4) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l1513_151360


namespace NUMINAMATH_CALUDE_trig_inequality_l1513_151335

theorem trig_inequality : 
  let a := Real.sin (Real.cos (2016 * π / 180))
  let b := Real.sin (Real.sin (2016 * π / 180))
  let c := Real.cos (Real.sin (2016 * π / 180))
  let d := Real.cos (Real.cos (2016 * π / 180))
  c > d ∧ d > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_trig_inequality_l1513_151335


namespace NUMINAMATH_CALUDE_about_set_S_l1513_151315

def S : Set ℤ := {x | ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2}

theorem about_set_S :
  (∀ x ∈ S, ¬(3 ∣ x)) ∧ (∃ x ∈ S, 11 ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_about_set_S_l1513_151315


namespace NUMINAMATH_CALUDE_hash_seven_three_l1513_151334

/-- The # operation on real numbers -/
noncomputable def hash (x y : ℝ) : ℝ :=
  sorry

/-- The first condition: x # 0 = x -/
axiom hash_zero (x : ℝ) : hash x 0 = x

/-- The second condition: x # y = y # x -/
axiom hash_comm (x y : ℝ) : hash x y = hash y x

/-- The third condition: (x + 1) # y = (x # y) + 2y + 1 -/
axiom hash_succ (x y : ℝ) : hash (x + 1) y = hash x y + 2 * y + 1

/-- The main theorem: 7 # 3 = 52 -/
theorem hash_seven_three : hash 7 3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_hash_seven_three_l1513_151334


namespace NUMINAMATH_CALUDE_larger_number_is_eight_l1513_151318

theorem larger_number_is_eight (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_eight_l1513_151318


namespace NUMINAMATH_CALUDE_union_of_sets_l1513_151313

theorem union_of_sets (A B : Set ℕ) (m : ℕ) : 
  A = {1, 2, 4} → 
  B = {m, 4, 7} → 
  A ∩ B = {1, 4} → 
  A ∪ B = {1, 2, 4, 7} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1513_151313


namespace NUMINAMATH_CALUDE_circle_theorem_l1513_151311

structure Circle where
  center : Point
  radius : ℝ

structure Angle where
  vertex : Point
  ray1 : Point
  ray2 : Point

def parallel (l1 l2 : Line) : Prop := sorry

def diameter (c : Circle) (l : Line) : Prop := sorry

def inscribed_angle (c : Circle) (a : Angle) : Prop := sorry

def angle_measure (a : Angle) : ℝ := sorry

theorem circle_theorem (c : Circle) (F B D C A : Point) 
  (FB DC AB FD : Line) (AFB ABF BCD : Angle) :
  diameter c FB →
  parallel FB DC →
  parallel AB FD →
  angle_measure AFB / angle_measure ABF = 3 / 4 →
  inscribed_angle c BCD →
  angle_measure BCD = 330 / 7 := by sorry

end NUMINAMATH_CALUDE_circle_theorem_l1513_151311


namespace NUMINAMATH_CALUDE_dog_age_is_twelve_l1513_151340

def cat_age : ℕ := 8

def rabbit_age (cat_age : ℕ) : ℕ := cat_age / 2

def dog_age (rabbit_age : ℕ) : ℕ := 3 * rabbit_age

theorem dog_age_is_twelve : dog_age (rabbit_age cat_age) = 12 := by
  sorry

end NUMINAMATH_CALUDE_dog_age_is_twelve_l1513_151340


namespace NUMINAMATH_CALUDE_sum_of_squares_l1513_151319

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 12)
  (eq2 : y^2 + 5*z = -15)
  (eq3 : z^2 + 7*x = -21) :
  x^2 + y^2 + z^2 = 83/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1513_151319


namespace NUMINAMATH_CALUDE_sum_of_y_values_l1513_151369

/-- Given 5 experiments with x and y values, prove the sum of y values -/
theorem sum_of_y_values
  (x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ : ℝ)
  (sum_x : x₁ + x₂ + x₃ + x₄ + x₅ = 150)
  (regression_eq : ∀ x y, y = 0.67 * x + 54.9) :
  y₁ + y₂ + y₃ + y₄ + y₅ = 375 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l1513_151369


namespace NUMINAMATH_CALUDE_root_property_l1513_151366

theorem root_property (a : ℝ) : a^2 + a - 1 = 0 → a^2 + a + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l1513_151366


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l1513_151301

theorem fuel_tank_capacity (C : ℝ) 
  (h1 : 0.12 * 122 + 0.16 * (C - 122) = 30) : C = 218 := by
  sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l1513_151301


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_34020_l1513_151317

def largest_perfect_square_factor (n : ℕ) : ℕ := 
  sorry

theorem largest_perfect_square_factor_34020 :
  largest_perfect_square_factor 34020 = 324 := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_34020_l1513_151317


namespace NUMINAMATH_CALUDE_smallest_k_for_f_l1513_151300

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -x^3 + 2*x^2 - x
  else if x ≥ 1 then Real.log x
  else 0  -- This case is not specified in the original problem, but we need to cover all reals

theorem smallest_k_for_f (k : ℝ) : 
  (∀ t > 0, f t < k * t) ↔ k > Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_f_l1513_151300


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l1513_151378

theorem divisibility_by_eleven (n : ℕ) (h : Odd n) : ∃ k : ℤ, (10 : ℤ)^n + 1 = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l1513_151378


namespace NUMINAMATH_CALUDE_problem_statement_l1513_151328

theorem problem_statement (x y : ℝ) (h1 : x + y = 4) (h2 : x * y = -2) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 440 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1513_151328


namespace NUMINAMATH_CALUDE_cube_root_of_four_sixth_powers_l1513_151333

theorem cube_root_of_four_sixth_powers (x : ℝ) :
  x = (4^6 + 4^6 + 4^6 + 4^6)^(1/3) → x = 16 * (4^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_four_sixth_powers_l1513_151333


namespace NUMINAMATH_CALUDE_cube_root_equivalence_l1513_151308

theorem cube_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x^2 * x^(1/2))^(1/3) = x^(5/6) := by sorry

end NUMINAMATH_CALUDE_cube_root_equivalence_l1513_151308


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l1513_151359

theorem complex_magnitude_example : Complex.abs (-3 + (8/5) * Complex.I) = 17/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l1513_151359


namespace NUMINAMATH_CALUDE_expression_simplification_value_at_three_value_at_four_l1513_151338

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  (1 - 1 / (x - 1)) / ((x^2 - 4) / (x^2 - 2*x + 1)) = (x - 1) / (x + 2) := by
  sorry

theorem value_at_three :
  (1 - 1 / (3 - 1)) / ((3^2 - 4) / (3^2 - 2*3 + 1)) = 2 / 5 := by
  sorry

theorem value_at_four :
  (1 - 1 / (4 - 1)) / ((4^2 - 4) / (4^2 - 2*4 + 1)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_value_at_three_value_at_four_l1513_151338


namespace NUMINAMATH_CALUDE_hazel_lemonade_cups_l1513_151358

/-- The number of cups of lemonade Hazel sold to kids on bikes -/
def cups_sold_to_kids : ℕ := 18

/-- The number of cups of lemonade Hazel made -/
def total_cups : ℕ := 56

theorem hazel_lemonade_cups : 
  total_cups = 56 ∧
  (total_cups / 2 : ℕ) + cups_sold_to_kids + (cups_sold_to_kids / 2 : ℕ) + 1 = total_cups :=
by sorry


end NUMINAMATH_CALUDE_hazel_lemonade_cups_l1513_151358


namespace NUMINAMATH_CALUDE_solution_g_less_than_6_range_of_a_l1513_151302

-- Define the functions f and g
def f (a x : ℝ) : ℝ := -|x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1| + |2*x + 4|

-- Theorem for the solution of g(x) < 6
theorem solution_g_less_than_6 : 
  {x : ℝ | g x < 6} = Set.Ioo (-9/4) (3/4) := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x₁, ∃ x₂, -g x₁ = f a x₂} = Set.Ici (-5) := by sorry

end NUMINAMATH_CALUDE_solution_g_less_than_6_range_of_a_l1513_151302


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1513_151393

theorem solution_set_inequality (x : ℝ) : 
  (2*x - 1) / (3*x + 1) > 1 ↔ -2 < x ∧ x < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1513_151393


namespace NUMINAMATH_CALUDE_magazine_subscription_cost_l1513_151343

theorem magazine_subscription_cost (reduction_percentage : ℝ) (reduction_amount : ℝ) (original_cost : ℝ) : 
  reduction_percentage = 0.30 → 
  reduction_amount = 588 → 
  reduction_percentage * original_cost = reduction_amount → 
  original_cost = 1960 := by
  sorry

end NUMINAMATH_CALUDE_magazine_subscription_cost_l1513_151343


namespace NUMINAMATH_CALUDE_product_of_integers_l1513_151330

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 18)
  (diff_squares_eq : x^2 - y^2 = 36) :
  x * y = 80 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l1513_151330


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1513_151344

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem isosceles_triangle (t : Triangle) 
  (h : 2 * Real.cos t.A * Real.cos t.B = 1 - Real.cos t.C) : 
  t.A = t.B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1513_151344


namespace NUMINAMATH_CALUDE_zsigmondy_prime_l1513_151382

theorem zsigmondy_prime (n : ℕ+) (p : ℕ) (k : ℕ) :
  3^(n : ℕ) - 2^(n : ℕ) = p^k → Nat.Prime p → Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_zsigmondy_prime_l1513_151382


namespace NUMINAMATH_CALUDE_min_distance_to_plane_l1513_151314

theorem min_distance_to_plane (x y z : ℝ) :
  x + 2*y + 3*z = 1 →
  x^2 + y^2 + z^2 ≥ 1/14 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_plane_l1513_151314


namespace NUMINAMATH_CALUDE_parabola_shift_l1513_151396

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := p.b - 2 * p.a * h,
    c := p.c + p.a * h^2 + p.b * h }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a,
    b := p.b,
    c := p.c + v }

/-- The original parabola y = x^2 + 2 -/
def original : Parabola :=
  { a := 1,
    b := 0,
    c := 2 }

theorem parabola_shift :
  let p1 := shift_horizontal original 1
  let p2 := shift_vertical p1 (-1)
  p2 = { a := 1, b := 2, c := 1 } :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l1513_151396


namespace NUMINAMATH_CALUDE_veranda_area_l1513_151388

/-- The area of a veranda surrounding a rectangular room. -/
theorem veranda_area (room_length room_width veranda_length_side veranda_width_side : ℝ)
  (h1 : room_length = 19)
  (h2 : room_width = 12)
  (h3 : veranda_length_side = 2.5)
  (h4 : veranda_width_side = 3) :
  (room_length + 2 * veranda_length_side) * (room_width + 2 * veranda_width_side) - 
  room_length * room_width = 204 := by
  sorry

end NUMINAMATH_CALUDE_veranda_area_l1513_151388


namespace NUMINAMATH_CALUDE_accountant_total_amount_l1513_151385

/-- Calculates the total amount given to the accountant for festival allowance --/
def festival_allowance_total (staff_count : ℕ) (daily_rate : ℕ) (days : ℕ) (petty_cash : ℕ) : ℕ :=
  staff_count * daily_rate * days + petty_cash

/-- Theorem stating the total amount given to the accountant --/
theorem accountant_total_amount :
  festival_allowance_total 20 100 30 1000 = 61000 := by
  sorry

end NUMINAMATH_CALUDE_accountant_total_amount_l1513_151385


namespace NUMINAMATH_CALUDE_f_extrema_half_f_extrema_sum_gt_zero_l1513_151316

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + a * x) - 2 * x / (x + 2)

-- Theorem for part (1)
theorem f_extrema_half :
  let a : ℝ := 1/2
  ∃ (min_val : ℝ), (∀ x, x > -2 → f a x ≥ min_val) ∧
                   (∃ x, x > -2 ∧ f a x = min_val) ∧
                   min_val = Real.log 2 - 1 ∧
                   (∀ M, ∃ x, x > -2 ∧ f a x > M) :=
sorry

-- Theorem for part (2)
theorem f_extrema_sum_gt_zero (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : 1/2 < a ∧ a < 1) 
  (hx₁ : x₁ > -1/a ∧ (∀ y, y > -1/a → f a y ≤ f a x₁))
  (hx₂ : x₂ > -1/a ∧ (∀ y, y > -1/a → f a y ≤ f a x₂))
  (hd : x₁ ≠ x₂) :
  f a x₁ + f a x₂ > f a 0 :=
sorry

end NUMINAMATH_CALUDE_f_extrema_half_f_extrema_sum_gt_zero_l1513_151316


namespace NUMINAMATH_CALUDE_blue_corduroy_glasses_count_l1513_151384

theorem blue_corduroy_glasses_count (total_students : ℕ) 
  (blue_shirt_percent : ℚ) (corduroy_percent : ℚ) (glasses_percent : ℚ) :
  total_students = 1500 →
  blue_shirt_percent = 35 / 100 →
  corduroy_percent = 20 / 100 →
  glasses_percent = 15 / 100 →
  ⌊total_students * blue_shirt_percent * corduroy_percent * glasses_percent⌋ = 15 := by
sorry

end NUMINAMATH_CALUDE_blue_corduroy_glasses_count_l1513_151384


namespace NUMINAMATH_CALUDE_village_population_l1513_151356

theorem village_population (P : ℝ) : 
  (P > 0) →
  (0.8 * (0.9 * P) = 4500) →
  P = 6250 := by
sorry

end NUMINAMATH_CALUDE_village_population_l1513_151356


namespace NUMINAMATH_CALUDE_calculate_expression_l1513_151349

theorem calculate_expression : 
  3⁻¹ + (27 : ℝ) ^ (1/3) - (5 - Real.sqrt 5)^0 + |Real.sqrt 3 - 1/3| = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1513_151349


namespace NUMINAMATH_CALUDE_triangle_side_length_l1513_151398

/-- Given a triangle ABC with angle A = 30°, angle B = 105°, and side a = 4,
    prove that the length of side c is 4√2. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/6 → B = 7*π/12 → a = 4 → 
  A + B + C = π → 
  a / Real.sin A = b / Real.sin B → 
  b / Real.sin B = c / Real.sin C → 
  c = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1513_151398


namespace NUMINAMATH_CALUDE_oliver_ferris_wheel_rides_l1513_151362

/-- The number of times Oliver rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride (ferris wheel or bumper car) -/
def ticket_cost : ℕ := 3

/-- The total number of tickets Oliver used -/
def total_tickets : ℕ := 30

/-- The number of times Oliver rode the ferris wheel -/
def ferris_wheel_rides : ℕ := (total_tickets - bumper_rides * ticket_cost) / ticket_cost

theorem oliver_ferris_wheel_rides :
  ferris_wheel_rides = 7 := by sorry

end NUMINAMATH_CALUDE_oliver_ferris_wheel_rides_l1513_151362


namespace NUMINAMATH_CALUDE_intersection_segment_length_l1513_151383

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 1 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ line_l A.1 A.2 ∧
  curve_C B.1 B.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_segment_length (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l1513_151383


namespace NUMINAMATH_CALUDE_friday_five_times_in_june_l1513_151304

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- Represents a month with its dates -/
structure Month :=
  (dates : List Date)
  (numDays : Nat)

def May : Month := sorry
def June : Month := sorry

/-- Counts the occurrences of a specific day of the week in a month -/
def countDayInMonth (d : DayOfWeek) (m : Month) : Nat := sorry

/-- Checks if a month has exactly five occurrences of a specific day of the week -/
def hasFiveOccurrences (d : DayOfWeek) (m : Month) : Prop :=
  countDayInMonth d m = 5

theorem friday_five_times_in_june 
  (h1 : hasFiveOccurrences DayOfWeek.Tuesday May)
  (h2 : May.numDays = 31)
  (h3 : June.numDays = 31) :
  hasFiveOccurrences DayOfWeek.Friday June := by
  sorry

end NUMINAMATH_CALUDE_friday_five_times_in_june_l1513_151304


namespace NUMINAMATH_CALUDE_amanda_notebooks_problem_l1513_151326

theorem amanda_notebooks_problem (initial_notebooks ordered_notebooks loss_percentage : ℕ) 
  (h1 : initial_notebooks = 65)
  (h2 : ordered_notebooks = 23)
  (h3 : loss_percentage = 15) : 
  initial_notebooks + ordered_notebooks - (((initial_notebooks + ordered_notebooks) * loss_percentage) / 100) = 75 := by
  sorry

end NUMINAMATH_CALUDE_amanda_notebooks_problem_l1513_151326


namespace NUMINAMATH_CALUDE_fourth_power_sum_geq_four_times_product_l1513_151341

theorem fourth_power_sum_geq_four_times_product (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a^4 + b^4 + c^4 + d^4 ≥ 4 * a * b * c * d := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_geq_four_times_product_l1513_151341


namespace NUMINAMATH_CALUDE_negative_sum_l1513_151346

theorem negative_sum (a b c : ℝ) 
  (ha : 1 < a ∧ a < 2) 
  (hb : 0 < b ∧ b < 1) 
  (hc : -2 < c ∧ c < -1) : 
  c + b < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_l1513_151346


namespace NUMINAMATH_CALUDE_inscribed_cube_side_length_is_sqrt6_div_2_l1513_151305

/-- Represents a pyramid with a regular hexagonal base and equilateral triangle lateral faces -/
structure HexagonalPyramid where
  base_side_length : ℝ
  lateral_face_is_equilateral : Bool

/-- Represents a cube inscribed in a hexagonal pyramid -/
structure InscribedCube where
  pyramid : HexagonalPyramid
  bottom_covers_base : Bool
  top_touches_midpoints : Bool

/-- Calculates the side length of an inscribed cube in a hexagonal pyramid -/
def inscribed_cube_side_length (cube : InscribedCube) : ℝ :=
  sorry

/-- Theorem stating that the side length of the inscribed cube is √6/2 -/
theorem inscribed_cube_side_length_is_sqrt6_div_2 
  (cube : InscribedCube) 
  (h1 : cube.pyramid.base_side_length = 2)
  (h2 : cube.pyramid.lateral_face_is_equilateral = true)
  (h3 : cube.bottom_covers_base = true)
  (h4 : cube.top_touches_midpoints = true) :
  inscribed_cube_side_length cube = Real.sqrt 6 / 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_side_length_is_sqrt6_div_2_l1513_151305


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1513_151367

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude_base_relation : ℝ → ℝ → Prop) 
  (base : ℝ) :
  area = 242 ∧ 
  altitude_base_relation base (2 * base) ∧
  area = base * (2 * base) →
  base = 11 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1513_151367


namespace NUMINAMATH_CALUDE_swimming_distance_l1513_151323

theorem swimming_distance (x : ℝ) 
  (h1 : x > 0)
  (h2 : (4 * x) / (5 * x) = 4 / 5)
  (h3 : (4 * x - 200) / (5 * x + 100) = 5 / 8) :
  4 * x = 1200 ∧ 5 * x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_swimming_distance_l1513_151323


namespace NUMINAMATH_CALUDE_bob_earnings_l1513_151310

def regular_rate : ℕ := 5
def overtime_rate : ℕ := 6
def regular_hours : ℕ := 40
def first_week_hours : ℕ := 44
def second_week_hours : ℕ := 48

def calculate_earnings (hours_worked : ℕ) : ℕ :=
  regular_rate * regular_hours + 
  overtime_rate * (hours_worked - regular_hours)

theorem bob_earnings : 
  calculate_earnings first_week_hours + calculate_earnings second_week_hours = 472 := by
  sorry

end NUMINAMATH_CALUDE_bob_earnings_l1513_151310


namespace NUMINAMATH_CALUDE_arithmetic_contains_geometric_l1513_151386

/-- An arithmetic sequence of positive real numbers -/
def arithmetic_sequence (a₀ d : ℝ) (n : ℕ) : ℝ := a₀ + n • d

/-- A geometric sequence of real numbers -/
def geometric_sequence (b₀ q : ℝ) (n : ℕ) : ℝ := b₀ * q^n

/-- Theorem: If an infinite arithmetic sequence of positive real numbers contains two different
    powers of an integer greater than 1, then it contains an infinite geometric sequence -/
theorem arithmetic_contains_geometric
  (a₀ d : ℝ) (a : ℕ) (h_a : a > 1) 
  (h_pos : ∀ n, arithmetic_sequence a₀ d n > 0)
  (m n : ℕ) (h_mn : m ≠ n)
  (h_power_m : ∃ k₁, arithmetic_sequence a₀ d k₁ = a^m)
  (h_power_n : ∃ k₂, arithmetic_sequence a₀ d k₂ = a^n) :
  ∃ b₀ q : ℝ, ∀ k, ∃ l, arithmetic_sequence a₀ d l = geometric_sequence b₀ q k :=
sorry

end NUMINAMATH_CALUDE_arithmetic_contains_geometric_l1513_151386


namespace NUMINAMATH_CALUDE_returning_players_l1513_151375

theorem returning_players (new_players : ℕ) (group_size : ℕ) (total_groups : ℕ) : 
  new_players = 48 → group_size = 6 → total_groups = 9 → 
  (total_groups * group_size) - new_players = 6 := by
  sorry

end NUMINAMATH_CALUDE_returning_players_l1513_151375


namespace NUMINAMATH_CALUDE_jamies_coins_value_l1513_151350

/-- Proves that given 30 coins of nickels and dimes, if swapping their values
    results in a 90-cent increase, then the total value is $1.80. -/
theorem jamies_coins_value :
  ∀ (n d : ℕ),
  n + d = 30 →
  (10 * n + 5 * d) - (5 * n + 10 * d) = 90 →
  5 * n + 10 * d = 180 := by
sorry

end NUMINAMATH_CALUDE_jamies_coins_value_l1513_151350


namespace NUMINAMATH_CALUDE_polyhedron_edge_length_bound_l1513_151307

/-- A polyhedron is represented as a set of points in ℝ³. -/
def Polyhedron : Type := Set (ℝ × ℝ × ℝ)

/-- The edges of a polyhedron. -/
def edges (P : Polyhedron) : Set (Set (ℝ × ℝ × ℝ)) := sorry

/-- The length of an edge. -/
def edgeLength (e : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The sum of all edge lengths in a polyhedron. -/
def sumEdgeLengths (P : Polyhedron) : ℝ := sorry

/-- The distance between two points in ℝ³. -/
def distance (p q : ℝ × ℝ × ℝ) : ℝ := sorry

/-- The maximum distance between any two points in a polyhedron. -/
def maxDistance (P : Polyhedron) : ℝ := sorry

/-- Theorem: The sum of edge lengths is at least 3 times the maximum distance. -/
theorem polyhedron_edge_length_bound (P : Polyhedron) :
  sumEdgeLengths P ≥ 3 * maxDistance P := by sorry

end NUMINAMATH_CALUDE_polyhedron_edge_length_bound_l1513_151307


namespace NUMINAMATH_CALUDE_joyces_property_size_l1513_151347

theorem joyces_property_size (new_property_size old_property_size pond_size suitable_land : ℝ) : 
  new_property_size = 10 * old_property_size →
  pond_size = 1 →
  suitable_land = 19 →
  new_property_size = suitable_land + pond_size →
  old_property_size = 2 := by
sorry

end NUMINAMATH_CALUDE_joyces_property_size_l1513_151347


namespace NUMINAMATH_CALUDE_consecutive_odd_power_sum_divisibility_l1513_151372

theorem consecutive_odd_power_sum_divisibility (p q m n : ℕ) : 
  Odd p → Odd q → p = q + 2 → Odd m → Odd n → m > 0 → n > 0 → 
  ∃ k : ℤ, p^m + q^n = k * (p + q) :=
sorry

end NUMINAMATH_CALUDE_consecutive_odd_power_sum_divisibility_l1513_151372


namespace NUMINAMATH_CALUDE_m_eq_two_iff_z_on_y_eq_x_l1513_151309

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := 1 + ((-1 + m) * Complex.I)

-- Define the condition for a point to lie on the line y = x
def lies_on_y_eq_x (z : ℂ) : Prop := z.im = z.re

-- State the theorem
theorem m_eq_two_iff_z_on_y_eq_x :
  ∀ m : ℝ, (m = 2) ↔ lies_on_y_eq_x (z m) :=
by sorry

end NUMINAMATH_CALUDE_m_eq_two_iff_z_on_y_eq_x_l1513_151309


namespace NUMINAMATH_CALUDE_absolute_value_sum_greater_than_one_l1513_151312

theorem absolute_value_sum_greater_than_one (x y : ℝ) :
  y ≤ -2 → abs x + abs y > 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_greater_than_one_l1513_151312


namespace NUMINAMATH_CALUDE_problem_solution_l1513_151352

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 5*x + 5/x + 1/x^2 = 40)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 11 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1513_151352


namespace NUMINAMATH_CALUDE_cut_triangular_prism_has_27_edges_l1513_151387

/-- Represents a triangular prism with corners cut off -/
structure CutTriangularPrism where
  /-- The number of vertices in the original triangular prism -/
  original_vertices : Nat
  /-- The number of edges in the original triangular prism -/
  original_edges : Nat
  /-- The number of new edges created by each corner cut -/
  new_edges_per_cut : Nat
  /-- Assertion that the cuts remove each corner entirely -/
  corners_removed : Prop
  /-- Assertion that the cuts do not intersect elsewhere on the prism -/
  cuts_dont_intersect : Prop

/-- The number of edges in a triangular prism with corners cut off -/
def num_edges_after_cuts (prism : CutTriangularPrism) : Nat :=
  prism.original_edges + prism.original_vertices * prism.new_edges_per_cut

/-- Theorem stating that a triangular prism with corners cut off has 27 edges -/
theorem cut_triangular_prism_has_27_edges (prism : CutTriangularPrism)
  (h1 : prism.original_vertices = 6)
  (h2 : prism.original_edges = 9)
  (h3 : prism.new_edges_per_cut = 3)
  (h4 : prism.corners_removed)
  (h5 : prism.cuts_dont_intersect) :
  num_edges_after_cuts prism = 27 := by
  sorry


end NUMINAMATH_CALUDE_cut_triangular_prism_has_27_edges_l1513_151387


namespace NUMINAMATH_CALUDE_problem_statement_l1513_151379

theorem problem_statement (a b x y : ℝ) 
  (h1 : a*x + b*y = 5)
  (h2 : a*x^2 + b*y^2 = 9)
  (h3 : a*x^3 + b*y^3 = 20)
  (h4 : a*x^4 + b*y^4 = 48)
  (h5 : x + y = -15)
  (h6 : x^2 + y^2 = 55) :
  a*x^5 + b*y^5 = -1065 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1513_151379


namespace NUMINAMATH_CALUDE_min_n_for_60n_divisible_by_4_and_8_l1513_151331

theorem min_n_for_60n_divisible_by_4_and_8 : 
  ∃ (n : ℕ), n > 0 ∧ 
    (∀ (m : ℕ), m > 0 → (4 ∣ 60 * m) ∧ (8 ∣ 60 * m) → n ≤ m) ∧
    (4 ∣ 60 * n) ∧ (8 ∣ 60 * n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_min_n_for_60n_divisible_by_4_and_8_l1513_151331


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l1513_151376

/-- Represents the number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 3 indistinguishable boxes is 5 -/
theorem five_balls_three_boxes : distributeBalls 5 3 = 5 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l1513_151376


namespace NUMINAMATH_CALUDE_water_distribution_l1513_151361

theorem water_distribution (total_water : ℕ) (eight_oz_glasses : ℕ) (four_oz_glasses : ℕ) 
  (h1 : total_water = 122)
  (h2 : eight_oz_glasses = 4)
  (h3 : four_oz_glasses = 15) : 
  (total_water - (8 * eight_oz_glasses + 4 * four_oz_glasses)) / 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_water_distribution_l1513_151361


namespace NUMINAMATH_CALUDE_complex_sum_problem_l1513_151370

theorem complex_sum_problem (a b c d e f : ℝ) : 
  b = 3 → 
  e = -a - c → 
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = 2 * Complex.I → 
  d + f = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l1513_151370


namespace NUMINAMATH_CALUDE_consecutive_digits_pattern_l1513_151399

def consecutive_digits (n : Nat) : Nat :=
  if n = 0 then 0 else
  let rec aux (k : Nat) (acc : Nat) : Nat :=
    if k = 0 then acc else aux (k - 1) (acc * 10 + k)
  aux n 0

def reverse_consecutive_digits (n : Nat) : Nat :=
  if n = 0 then 0 else
  let rec aux (k : Nat) (acc : Nat) : Nat :=
    if k = 0 then acc else aux (k - 1) (acc * 10 + (10 - k))
  aux n 0

theorem consecutive_digits_pattern (n : Nat) (h : n > 0 ∧ n ≤ 9) :
  consecutive_digits n * 8 + n = reverse_consecutive_digits n := by
  sorry

end NUMINAMATH_CALUDE_consecutive_digits_pattern_l1513_151399


namespace NUMINAMATH_CALUDE_xy_value_l1513_151374

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1513_151374


namespace NUMINAMATH_CALUDE_tax_center_revenue_l1513_151395

/-- Calculates the total revenue for a tax center based on the number and types of returns sold --/
theorem tax_center_revenue (federal_price state_price quarterly_price : ℕ)
                           (federal_sold state_sold quarterly_sold : ℕ) :
  federal_price = 50 →
  state_price = 30 →
  quarterly_price = 80 →
  federal_sold = 60 →
  state_sold = 20 →
  quarterly_sold = 10 →
  federal_price * federal_sold + state_price * state_sold + quarterly_price * quarterly_sold = 4400 :=
by sorry

end NUMINAMATH_CALUDE_tax_center_revenue_l1513_151395


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l1513_151397

/-- Calculates the upstream speed of a person given their downstream speed and the stream speed. -/
def upstreamSpeed (downstreamSpeed streamSpeed : ℝ) : ℝ :=
  downstreamSpeed - 2 * streamSpeed

/-- Theorem: Given a downstream speed of 12 km/h and a stream speed of 2 km/h, the upstream speed is 8 km/h. -/
theorem upstream_speed_calculation :
  upstreamSpeed 12 2 = 8 := by
  sorry

#eval upstreamSpeed 12 2

end NUMINAMATH_CALUDE_upstream_speed_calculation_l1513_151397


namespace NUMINAMATH_CALUDE_square_symmetry_count_l1513_151389

/-- Represents the symmetry operations on a square -/
inductive SquareSymmetry
| reflect : SquareSymmetry
| rotate : SquareSymmetry

/-- Represents a sequence of symmetry operations -/
def SymmetrySequence := List SquareSymmetry

/-- Checks if a sequence of symmetry operations results in the identity transformation -/
def is_identity (seq : SymmetrySequence) : Prop :=
  sorry

/-- Counts the number of valid symmetry sequences of a given length -/
def count_valid_sequences (n : Nat) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem square_symmetry_count :
  count_valid_sequences 2016 % 100000 = 20000 :=
sorry

end NUMINAMATH_CALUDE_square_symmetry_count_l1513_151389


namespace NUMINAMATH_CALUDE_trig_identity_l1513_151306

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 
  1 / (Real.cos (10 * π / 180) * Real.cos (20 * π / 180)) := by sorry

end NUMINAMATH_CALUDE_trig_identity_l1513_151306


namespace NUMINAMATH_CALUDE_dot_product_properties_l1513_151332

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem dot_product_properties 
  (a b : ℝ × ℝ) 
  (h1 : Real.sqrt (a.1^2 + a.2^2) = 10)
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 12)
  (h3 : angle_between a b = 2 * π / 3) : 
  (a.1 * b.1 + a.2 * b.2 = -60) ∧ 
  (3 * a.1 * (1/5 * b.1) + 3 * a.2 * (1/5 * b.2) = -36) ∧
  ((3 * b.1 - 2 * a.1) * (4 * a.1 + b.1) + (3 * b.2 - 2 * a.2) * (4 * a.2 + b.2) = -968) := by
  sorry

end NUMINAMATH_CALUDE_dot_product_properties_l1513_151332


namespace NUMINAMATH_CALUDE_basketball_handshakes_l1513_151390

/-- The number of handshakes in a basketball game with specific conditions -/
theorem basketball_handshakes :
  let team_size : ℕ := 6
  let num_teams : ℕ := 2
  let num_referees : ℕ := 3
  let opposing_team_handshakes := team_size * team_size
  let same_team_handshakes := num_teams * (team_size * (team_size - 1) / 2)
  let player_referee_handshakes := (num_teams * team_size) * num_referees
  opposing_team_handshakes + same_team_handshakes + player_referee_handshakes = 102 :=
by sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l1513_151390


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1513_151337

theorem min_distance_to_line (x y : ℝ) (h : 2 * x - y - 5 = 0) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧
  ∀ (x' y' : ℝ), 2 * x' - y' - 5 = 0 → Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1513_151337


namespace NUMINAMATH_CALUDE_equation_equivalence_l1513_151345

theorem equation_equivalence (a b c : ℕ) 
  (ha : 0 < a ∧ a ≤ 10) 
  (hb : 0 < b ∧ b ≤ 10) 
  (hc : 0 < c ∧ c ≤ 10) : 
  (10 * a + b) * (10 * a + c) = 100 * a^2 + 100 * a + 11 * b * c ↔ b + 11 * c = 10 * a :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1513_151345


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1513_151348

theorem quadratic_transformation (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (h k s : ℝ) (hs : s ≠ 0), ∀ x : ℝ,
    a * x^2 + b * x + c = s^2 * ((x - h)^2 + k) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1513_151348


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1513_151368

theorem circle_tangent_to_line (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = m^2 → x + y ≠ m) ∨ 
  (∃ x y : ℝ, x^2 + y^2 = m^2 ∧ x + y = m ∧ 
    (∀ x' y' : ℝ, x'^2 + y'^2 = m^2 → x' + y' = m → (x', y') = (x, y))) ↔ 
  m = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1513_151368


namespace NUMINAMATH_CALUDE_no_winning_strategy_for_tony_l1513_151322

/-- Represents the state of the Ring Mafia game -/
structure GameState where
  total_counters : ℕ
  mafia_counters : ℕ
  town_counters : ℕ

/-- Defines a valid initial state for the Ring Mafia game -/
def valid_initial_state (state : GameState) : Prop :=
  state.total_counters ≥ 3 ∧
  state.total_counters % 2 = 1 ∧
  state.mafia_counters = (state.total_counters - 1) / 3 ∧
  state.town_counters = 2 * (state.total_counters - 1) / 3 ∧
  state.mafia_counters + state.town_counters = state.total_counters

/-- Defines a winning state for Tony -/
def tony_wins (state : GameState) : Prop :=
  state.town_counters > 0 ∧ state.mafia_counters = 0

/-- Represents a strategy for Tony -/
def TonyStrategy := GameState → Set ℕ

/-- Defines the concept of a winning strategy for Tony -/
def winning_strategy (strategy : TonyStrategy) : Prop :=
  ∀ (initial_state : GameState),
    valid_initial_state initial_state →
    ∃ (final_state : GameState),
      tony_wins final_state

/-- The main theorem: Tony does not have a winning strategy -/
theorem no_winning_strategy_for_tony :
  ¬∃ (strategy : TonyStrategy), winning_strategy strategy :=
sorry

end NUMINAMATH_CALUDE_no_winning_strategy_for_tony_l1513_151322


namespace NUMINAMATH_CALUDE_xy_value_l1513_151357

theorem xy_value (x y : ℝ) : y = Real.sqrt (x - 1/2) + Real.sqrt (1/2 - x) - 6 → x * y = -3 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1513_151357


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l1513_151380

theorem fraction_inequality_solution_set : 
  {x : ℝ | x / (x + 1) < 0} = Set.Ioo (-1) 0 := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l1513_151380


namespace NUMINAMATH_CALUDE_angle_identity_l1513_151342

theorem angle_identity (α : Real) (h1 : 0 ≤ α) (h2 : α < 2 * Real.pi) 
  (h3 : ∃ (x y : Real), x = Real.sin (215 * Real.pi / 180) ∧ 
                        y = Real.cos (215 * Real.pi / 180) ∧ 
                        x = Real.sin α ∧ 
                        y = Real.cos α) : 
  α = 235 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_identity_l1513_151342


namespace NUMINAMATH_CALUDE_map_segment_to_yards_l1513_151371

/-- Converts a length in inches on a map to yards in reality, given a scale --/
def map_length_to_yards (map_length : ℚ) (scale : ℚ) : ℚ :=
  (map_length * scale) / 3

/-- The scale of the map (feet per inch) --/
def map_scale : ℚ := 500

/-- The length of the line segment on the map (in inches) --/
def line_segment_length : ℚ := 6.25

/-- Theorem: The 6.25-inch line segment on the map represents 1041 2/3 yards in reality --/
theorem map_segment_to_yards :
  map_length_to_yards line_segment_length map_scale = 1041 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_map_segment_to_yards_l1513_151371


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1513_151303

def first_ten_integers : Finset ℕ := Finset.range 10

theorem least_common_multiple_first_ten : ∃ n : ℕ, n > 0 ∧ (∀ i ∈ first_ten_integers, i ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ i ∈ first_ten_integers, i ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1513_151303


namespace NUMINAMATH_CALUDE_expression_equals_sum_l1513_151355

theorem expression_equals_sum (a b c : ℝ) (ha : a = 14) (hb : b = 19) (hc : c = 23) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

#eval (14 : ℝ) + 19 + 23

end NUMINAMATH_CALUDE_expression_equals_sum_l1513_151355


namespace NUMINAMATH_CALUDE_gcd_18_30_is_6_and_even_l1513_151336

theorem gcd_18_30_is_6_and_even : 
  Nat.gcd 18 30 = 6 ∧ Even 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_is_6_and_even_l1513_151336


namespace NUMINAMATH_CALUDE_number_problem_l1513_151354

theorem number_problem (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 320) : 
  x * y = 64 ∧ x^3 + y^3 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1513_151354


namespace NUMINAMATH_CALUDE_total_area_equals_total_frequency_l1513_151339

/-- A frequency distribution histogram -/
structure FrequencyHistogram where
  /-- The list of frequencies for each bin -/
  frequencies : List ℝ
  /-- All frequencies are non-negative -/
  all_nonneg : ∀ f ∈ frequencies, f ≥ 0

/-- The total frequency of a histogram -/
def totalFrequency (h : FrequencyHistogram) : ℝ :=
  h.frequencies.sum

/-- The total area of small rectangles in a histogram -/
def totalArea (h : FrequencyHistogram) : ℝ :=
  h.frequencies.sum

/-- Theorem: The total area of small rectangles in a frequency distribution histogram
    is equal to the total frequency -/
theorem total_area_equals_total_frequency (h : FrequencyHistogram) :
  totalArea h = totalFrequency h := by
  sorry


end NUMINAMATH_CALUDE_total_area_equals_total_frequency_l1513_151339


namespace NUMINAMATH_CALUDE_construction_costs_l1513_151394

/-- Calculate the total construction costs for a house project. -/
theorem construction_costs
  (land_cost_per_sqm : ℝ)
  (brick_cost_per_1000 : ℝ)
  (tile_cost_per_tile : ℝ)
  (land_area : ℝ)
  (brick_count : ℝ)
  (tile_count : ℝ)
  (h1 : land_cost_per_sqm = 50)
  (h2 : brick_cost_per_1000 = 100)
  (h3 : tile_cost_per_tile = 10)
  (h4 : land_area = 2000)
  (h5 : brick_count = 10000)
  (h6 : tile_count = 500) :
  land_cost_per_sqm * land_area +
  brick_cost_per_1000 * (brick_count / 1000) +
  tile_cost_per_tile * tile_count = 106000 := by
  sorry


end NUMINAMATH_CALUDE_construction_costs_l1513_151394


namespace NUMINAMATH_CALUDE_rosys_age_l1513_151373

/-- Proves that Rosy's current age is 8 years, given the conditions about David's age --/
theorem rosys_age (rosy_age : ℕ) : 
  (rosy_age + 12 + 4 = 2 * (rosy_age + 4)) → rosy_age = 8 := by
  sorry

#check rosys_age

end NUMINAMATH_CALUDE_rosys_age_l1513_151373


namespace NUMINAMATH_CALUDE_problem_solution_l1513_151391

/-- The set of integers of the form m^k for integers m, k ≥ 2 -/
def S : Set ℕ := {n : ℕ | ∃ m k : ℕ, m ≥ 2 ∧ k ≥ 2 ∧ n = m^k}

/-- The number of ways to write n as the sum of distinct elements of S -/
def f (n : ℕ) : ℕ := sorry

/-- The set of integers for which f(n) = 3 -/
def T : Set ℕ := {n : ℕ | f n = 3}

theorem problem_solution :
  (f 30 = 0) ∧
  (∀ n : ℕ, n ≥ 31 → f n ≥ 1) ∧
  (T.Finite ∧ T.Nonempty) ∧
  (∃ m : ℕ, m ∈ T ∧ ∀ n ∈ T, n ≤ m) ∧
  (∃ m : ℕ, m ∈ T ∧ ∀ n ∈ T, n ≤ m ∧ m = 111) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1513_151391
