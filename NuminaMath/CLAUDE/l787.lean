import Mathlib

namespace NUMINAMATH_CALUDE_sum_ac_l787_78761

theorem sum_ac (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 42)
  (h2 : b + d = 5) : 
  a + c = 42 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_ac_l787_78761


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l787_78770

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : α > 0) (h2 : α < Real.pi / 2) 
  (h3 : Real.cos (2 * α) + Real.cos α ^ 2 = 0) : 
  Real.tan (α + Real.pi / 4) = -3 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l787_78770


namespace NUMINAMATH_CALUDE_probability_of_valid_assignment_l787_78778

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

def valid_assignment (al bill cal : ℕ) : Prop :=
  1 ≤ al ∧ al ≤ 12 ∧
  1 ≤ bill ∧ bill ≤ 12 ∧
  1 ≤ cal ∧ cal ≤ 12 ∧
  is_multiple al bill ∧
  is_multiple bill cal

def total_assignments : ℕ := 12 * 12 * 12

def count_valid_assignments : ℕ := sorry

theorem probability_of_valid_assignment :
  (count_valid_assignments : ℚ) / total_assignments = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_of_valid_assignment_l787_78778


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_exist_l787_78747

theorem infinitely_many_pairs_exist : 
  ∀ n : ℕ, ∃ a b : ℕ+, 
    a.val > n ∧ 
    b.val > n ∧ 
    (a.val * b.val) ∣ (a.val^2 + b.val^2 + a.val + b.val + 1) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_exist_l787_78747


namespace NUMINAMATH_CALUDE_log_inequality_l787_78787

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x^2) < x^2 / (1 + x^2) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l787_78787


namespace NUMINAMATH_CALUDE_chocolate_squares_l787_78781

theorem chocolate_squares (jenny_squares mike_squares : ℕ) : 
  jenny_squares = 65 → 
  jenny_squares = 3 * mike_squares + 5 → 
  mike_squares = 20 := by
sorry

end NUMINAMATH_CALUDE_chocolate_squares_l787_78781


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l787_78782

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l787_78782


namespace NUMINAMATH_CALUDE_lewis_speed_l787_78769

/-- Proves that Lewis's speed is 80 mph given the problem conditions -/
theorem lewis_speed (john_speed : ℝ) (total_distance : ℝ) (meeting_distance : ℝ) :
  john_speed = 40 ∧ 
  total_distance = 240 ∧ 
  meeting_distance = 160 →
  (total_distance + (total_distance - meeting_distance)) / (meeting_distance / john_speed) = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_lewis_speed_l787_78769


namespace NUMINAMATH_CALUDE_dispatch_plans_eq_28_l787_78791

/-- Given a set of athletes with the following properties:
  * There are 9 athletes in total
  * 5 athletes can play basketball
  * 6 athletes can play soccer
This function calculates the number of ways to select one athlete for basketball
and one for soccer. -/
def dispatch_plans (total : Nat) (basketball : Nat) (soccer : Nat) : Nat :=
  sorry

/-- Theorem stating that the number of dispatch plans for the given conditions is 28. -/
theorem dispatch_plans_eq_28 : dispatch_plans 9 5 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_dispatch_plans_eq_28_l787_78791


namespace NUMINAMATH_CALUDE_discount_problem_l787_78788

/-- Calculate the original price given the discounted price and discount rate -/
def originalPrice (discountedPrice : ℚ) (discountRate : ℚ) : ℚ :=
  discountedPrice / (1 - discountRate)

/-- The problem statement -/
theorem discount_problem (item1_discounted : ℚ) (item1_rate : ℚ)
                         (item2_discounted : ℚ) (item2_rate : ℚ)
                         (item3_discounted : ℚ) (item3_rate : ℚ) :
  item1_discounted = 4400 →
  item1_rate = 56 / 100 →
  item2_discounted = 3900 →
  item2_rate = 35 / 100 →
  item3_discounted = 2400 →
  item3_rate = 20 / 100 →
  originalPrice item1_discounted item1_rate +
  originalPrice item2_discounted item2_rate +
  originalPrice item3_discounted item3_rate = 19000 := by
  sorry

end NUMINAMATH_CALUDE_discount_problem_l787_78788


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l787_78701

/-- Given that M(-3, 2) is the midpoint of AB and A(-8, 5) is one endpoint,
    prove that the sum of coordinates of B is 1. -/
theorem midpoint_coordinate_sum :
  let M : ℝ × ℝ := (-3, 2)
  let A : ℝ × ℝ := (-8, 5)
  ∀ B : ℝ × ℝ,
  (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) →
  B.1 + B.2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l787_78701


namespace NUMINAMATH_CALUDE_inverse_88_mod_89_l787_78767

theorem inverse_88_mod_89 : ∃ x : ℕ, x ≤ 88 ∧ (88 * x) % 89 = 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_inverse_88_mod_89_l787_78767


namespace NUMINAMATH_CALUDE_rita_jackets_l787_78719

def problem (num_dresses num_pants jacket_cost dress_cost pants_cost transport_cost initial_amount remaining_amount : ℕ) : Prop :=
  let total_spent := initial_amount - remaining_amount
  let dress_pants_cost := num_dresses * dress_cost + num_pants * pants_cost
  let jacket_total_cost := total_spent - dress_pants_cost - transport_cost
  jacket_total_cost / jacket_cost = 4

theorem rita_jackets : 
  problem 5 3 30 20 12 5 400 139 := by sorry

end NUMINAMATH_CALUDE_rita_jackets_l787_78719


namespace NUMINAMATH_CALUDE_three_digit_sum_l787_78744

theorem three_digit_sum (a b c : Nat) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  (1730 + a) % 9 = 0 →
  (1730 + b) % 11 = 0 →
  (1730 + c) % 6 = 0 →
  a + b + c = 19 := by
sorry

end NUMINAMATH_CALUDE_three_digit_sum_l787_78744


namespace NUMINAMATH_CALUDE_grasshopper_jumps_l787_78720

theorem grasshopper_jumps : ∃ (x y : ℕ), 80 * x - 50 * y = 170 ∧ x + y ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_jumps_l787_78720


namespace NUMINAMATH_CALUDE_salary_percentage_decrease_l787_78754

theorem salary_percentage_decrease 
  (x : ℝ) -- Original salary
  (h1 : x * 1.15 = 575) -- 15% increase condition
  (h2 : x * (1 - y / 100) = 560) -- y% decrease condition
  : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_salary_percentage_decrease_l787_78754


namespace NUMINAMATH_CALUDE_triangle_inequality_l787_78797

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  let s := (a + b + c) / 2
  (a * b) / (s - c) + (b * c) / (s - a) + (c * a) / (s - b) ≥ 4 * s := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l787_78797


namespace NUMINAMATH_CALUDE_tom_running_distance_l787_78756

def base_twelve_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 12^3 + ((n / 100) % 10) * 12^2 + ((n / 10) % 10) * 12^1 + (n % 10)

def average_per_week (total : ℕ) (weeks : ℕ) : ℚ :=
  (total : ℚ) / (weeks : ℚ)

theorem tom_running_distance :
  let base_twelve_distance : ℕ := 3847
  let decimal_distance : ℕ := base_twelve_to_decimal base_twelve_distance
  let weeks : ℕ := 4
  decimal_distance = 6391 ∧ average_per_week decimal_distance weeks = 1597.75 := by
  sorry

end NUMINAMATH_CALUDE_tom_running_distance_l787_78756


namespace NUMINAMATH_CALUDE_problem_statement_l787_78779

theorem problem_statement (a b : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (x + a) * (x + b) * (x + 10) = 0 ∧
    (y + a) * (y + b) * (y + 10) = 0 ∧
    (z + a) * (z + b) * (z + 10) = 0 ∧
    x ≠ -4 ∧ y ≠ -4 ∧ z ≠ -4) →
  (∃! w : ℝ, (w + 2*a) * (w + 5) * (w + 8) = 0 ∧ 
    w ≠ -b ∧ w ≠ -10) →
  100 * a + b = 258 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l787_78779


namespace NUMINAMATH_CALUDE_range_of_a_l787_78766

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 ≥ a) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) → 
  a ∈ Set.Iic (-2) ∪ {1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l787_78766


namespace NUMINAMATH_CALUDE_tangent_point_exists_l787_78773

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_point_exists : ∃ (x₀ y₀ : ℝ), 
  f x₀ = y₀ ∧ 
  f' x₀ = 4 ∧ 
  x₀ = -1 ∧ 
  y₀ = -4 :=
sorry

end NUMINAMATH_CALUDE_tangent_point_exists_l787_78773


namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l787_78712

/-- A calendrical system where leap years occur every three years -/
structure ModifiedCalendar where
  leapYearInterval : ℕ
  leapYearInterval_eq : leapYearInterval = 3

/-- The number of years in the period we're considering -/
def periodLength : ℕ := 100

/-- The maximum number of leap years in the given period -/
def maxLeapYears (c : ModifiedCalendar) : ℕ :=
  periodLength / c.leapYearInterval

/-- Theorem stating that the maximum number of leap years in a 100-year period is 33 -/
theorem max_leap_years_in_period (c : ModifiedCalendar) :
  maxLeapYears c = 33 := by
  sorry

#check max_leap_years_in_period

end NUMINAMATH_CALUDE_max_leap_years_in_period_l787_78712


namespace NUMINAMATH_CALUDE_triangles_with_fixed_vertex_l787_78757

theorem triangles_with_fixed_vertex (n : ℕ) (h : n = 9) :
  Nat.choose (n - 1) 2 = 28 :=
sorry

end NUMINAMATH_CALUDE_triangles_with_fixed_vertex_l787_78757


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l787_78703

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific conditions,
    the man's speed against the current is 16 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 21 2.5 = 16 := by
  sorry

#eval speed_against_current 21 2.5

end NUMINAMATH_CALUDE_mans_speed_against_current_l787_78703


namespace NUMINAMATH_CALUDE_quadratic_sequence_formula_l787_78700

theorem quadratic_sequence_formula (a : ℕ → ℚ) (α β : ℚ) :
  (∀ n : ℕ, a n * α^2 - a (n + 1) * α + 1 = 0) →
  (∀ n : ℕ, a n * β^2 - a (n + 1) * β + 1 = 0) →
  (6 * α - 2 * α * β + 6 * β = 3) →
  (a 1 = 7 / 6) →
  (∀ n : ℕ, a n = (1 / 2)^n + 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sequence_formula_l787_78700


namespace NUMINAMATH_CALUDE_min_value_F_l787_78758

/-- The function F as defined in the problem -/
def F (m n : ℝ) : ℝ := (m - n)^2 + (m^2 - n + 1)^2

/-- Theorem stating that the minimum value of F(m,n) is 9/32 -/
theorem min_value_F :
  ∀ m n : ℝ, F m n ≥ 9/32 := by
  sorry

end NUMINAMATH_CALUDE_min_value_F_l787_78758


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l787_78786

theorem inserted_numbers_sum : ∃! (a b : ℝ), 
  0 < a ∧ 0 < b ∧ 
  4 < a ∧ a < b ∧ b < 16 ∧ 
  (∃ r : ℝ, 0 < r ∧ a = 4 * r ∧ b = 4 * r^2) ∧
  (∃ d : ℝ, b = a + d ∧ 16 = b + d) ∧
  a + b = 24 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l787_78786


namespace NUMINAMATH_CALUDE_expression_evaluation_l787_78789

theorem expression_evaluation (x y : ℝ) 
  (h : (3*x + 1)^2 + |y - 3| = 0) : 
  (x + 2*y) * (x - 2*y) + (x + 2*y)^2 - x * (2*x + 3*y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l787_78789


namespace NUMINAMATH_CALUDE_larger_tv_diagonal_l787_78735

theorem larger_tv_diagonal (d : ℝ) : d > 0 →
  (d^2 / 2) = (17^2 / 2) + 143.5 →
  d = 24 := by
sorry

end NUMINAMATH_CALUDE_larger_tv_diagonal_l787_78735


namespace NUMINAMATH_CALUDE_dart_points_ratio_l787_78763

/-- Prove that the ratio of the points of the third dart to the points of the bullseye is 1:2 -/
theorem dart_points_ratio :
  let bullseye_points : ℕ := 50
  let missed_points : ℕ := 0
  let total_score : ℕ := 75
  let third_dart_points : ℕ := total_score - bullseye_points - missed_points
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ a * bullseye_points = b * third_dart_points ∧ a = 1 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_dart_points_ratio_l787_78763


namespace NUMINAMATH_CALUDE_expression_value_l787_78776

theorem expression_value : (5^8 - 3^7) * (1^6 + (-1)^5)^11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l787_78776


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l787_78713

theorem min_value_sum_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2*x + y - 3 = 0) : 
  2/x + 1/y ≥ 3 ∧ (2/x + 1/y = 3 ↔ x = 1 ∧ y = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l787_78713


namespace NUMINAMATH_CALUDE_newspaper_sale_percentage_l787_78717

/-- Represents the problem of calculating the percentage of newspapers John sells. -/
theorem newspaper_sale_percentage
  (total_newspapers : ℕ)
  (selling_price : ℚ)
  (discount_percentage : ℚ)
  (profit : ℚ)
  (h1 : total_newspapers = 500)
  (h2 : selling_price = 2)
  (h3 : discount_percentage = 75 / 100)
  (h4 : profit = 550)
  : (selling_price * (1 - discount_percentage) * total_newspapers + profit) / (selling_price * total_newspapers) = 4 / 5 :=
sorry

end NUMINAMATH_CALUDE_newspaper_sale_percentage_l787_78717


namespace NUMINAMATH_CALUDE_vector_operation_proof_l787_78745

def vector_a : Fin 2 → ℝ := ![2, 4]
def vector_b : Fin 2 → ℝ := ![-1, 1]

theorem vector_operation_proof :
  (2 • vector_a - vector_b) = ![5, 7] := by sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l787_78745


namespace NUMINAMATH_CALUDE_dice_roll_probability_l787_78725

/-- The probability of rolling a number other than 1 on a standard die -/
def prob_not_one : ℚ := 5 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability that (a-1)(b-1)(c-1)(d-1) ≠ 0 when four standard dice are tossed -/
def prob_product_nonzero : ℚ := prob_not_one ^ num_dice

theorem dice_roll_probability :
  prob_product_nonzero = 625 / 1296 := by sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l787_78725


namespace NUMINAMATH_CALUDE_carousel_attendance_l787_78729

/-- The number of children attending a carousel, given:
  * 4 clowns also attend
  * The candy seller initially had 700 candies
  * Each clown and child receives 20 candies
  * The candy seller has 20 candies left after selling
-/
def num_children : ℕ := 30

theorem carousel_attendance : num_children = 30 := by
  sorry

end NUMINAMATH_CALUDE_carousel_attendance_l787_78729


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l787_78711

theorem matrix_multiplication_result :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -1; 5, 7]
  let B : Matrix (Fin 2) (Fin 3) ℤ := !![2, 1, 4; 1, 0, -2]
  A * B = !![5, 3, 14; 17, 5, 6] := by
sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l787_78711


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_and_ellipse_l787_78795

-- Define the equation
def equation (y z : ℝ) : Prop :=
  z^4 - 6*y^4 = 3*z^2 - 8

-- Define what it means for the equation to represent a hyperbola
def represents_hyperbola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
    ∀ (y z : ℝ), eq y z ↔ a*y^2 + b*z^2 + c*y*z + d*y + e*z + f = 0

-- Define what it means for the equation to represent an ellipse
def represents_ellipse (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d e f : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (y z : ℝ), eq y z ↔ a*y^2 + b*z^2 + c*y*z + d*y + e*z + f = 0

-- Theorem statement
theorem equation_represents_hyperbola_and_ellipse :
  represents_hyperbola equation ∧ represents_ellipse equation :=
sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_and_ellipse_l787_78795


namespace NUMINAMATH_CALUDE_min_value_cube_sum_squared_l787_78738

theorem min_value_cube_sum_squared (a b c : ℝ) :
  (∃ (α β γ : ℤ), α ∈ ({-1, 1} : Set ℤ) ∧ β ∈ ({-1, 1} : Set ℤ) ∧ γ ∈ ({-1, 1} : Set ℤ) ∧ a * α + b * β + c * γ = 0) →
  ((a^3 + b^3 + c^3) / (a * b * c))^2 ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_cube_sum_squared_l787_78738


namespace NUMINAMATH_CALUDE_right_triangle_area_l787_78716

/-- A right-angled triangle with specific properties -/
structure RightTriangle where
  -- The legs of the triangle
  a : ℝ
  b : ℝ
  -- The hypotenuse of the triangle
  c : ℝ
  -- Conditions
  right_angle : a^2 + b^2 = c^2
  perimeter : a + b + c = 2 + Real.sqrt 6
  hypotenuse : c = 2
  median : (a + b) / 2 = 1

/-- The area of a right-angled triangle with the given properties is 1/2 -/
theorem right_triangle_area (t : RightTriangle) : (t.a * t.b) / 2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l787_78716


namespace NUMINAMATH_CALUDE_range_of_a_l787_78723

open Set

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - 6*x + 8 > 0

def sufficient_not_necessary (P Q : Set ℝ) : Prop :=
  P ⊂ Q ∧ P ≠ Q

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (sufficient_not_necessary {x | p x a} {x | q x}) →
  (a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l787_78723


namespace NUMINAMATH_CALUDE_functions_strictly_greater_iff_no_leq_l787_78790

-- Define the functions f and g with domain ℝ
variable (f g : ℝ → ℝ)

-- State the theorem
theorem functions_strictly_greater_iff_no_leq :
  (∀ x : ℝ, f x > g x) ↔ ¬∃ x : ℝ, f x ≤ g x := by sorry

end NUMINAMATH_CALUDE_functions_strictly_greater_iff_no_leq_l787_78790


namespace NUMINAMATH_CALUDE_m_minus_n_values_l787_78798

theorem m_minus_n_values (m n : ℤ) 
  (hm : |m| = 5)
  (hn : |n| = 7)
  (hmn_neg : m + n < 0) :
  m - n = 12 ∨ m - n = 2 := by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_values_l787_78798


namespace NUMINAMATH_CALUDE_lowest_unique_score_l787_78715

/-- The scoring function for the modified AHSME -/
def score (c w : ℕ) : ℤ := 30 + 4 * c - 2 * w

/-- Predicate to check if a score uniquely determines c and w -/
def uniquely_determines (s : ℤ) : Prop :=
  ∃! (c w : ℕ), score c w = s ∧ c + w ≤ 30

theorem lowest_unique_score : 
  (∀ s : ℤ, 100 < s → s < 116 → ¬ uniquely_determines s) ∧
  uniquely_determines 116 := by
  sorry

end NUMINAMATH_CALUDE_lowest_unique_score_l787_78715


namespace NUMINAMATH_CALUDE_dataset_mode_l787_78792

def dataset : List Nat := [3, 1, 3, 0, 3, 2, 1, 2]

def mode (l : List Nat) : Nat :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem dataset_mode :
  mode dataset = 3 := by
  sorry

end NUMINAMATH_CALUDE_dataset_mode_l787_78792


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l787_78793

theorem pencil_buyers_difference (price : ℕ) 
  (h1 : price > 0)
  (h2 : 234 % price = 0)
  (h3 : 325 % price = 0) :
  325 / price - 234 / price = 7 := by
  sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l787_78793


namespace NUMINAMATH_CALUDE_octopus_equality_month_l787_78724

theorem octopus_equality_month : 
  (∀ k : ℕ, k < 4 → 3^(k + 1) ≠ 15 * 5^k) ∧ 
  3^(4 + 1) = 15 * 5^4 := by
  sorry

end NUMINAMATH_CALUDE_octopus_equality_month_l787_78724


namespace NUMINAMATH_CALUDE_boat_license_count_l787_78708

def boat_license_options : ℕ :=
  let letter_options := 3  -- A, M, or S
  let digit_options := 10  -- 0 to 9
  let number_of_digits := 6
  letter_options * digit_options ^ number_of_digits

theorem boat_license_count : boat_license_options = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_count_l787_78708


namespace NUMINAMATH_CALUDE_simplify_expression_l787_78799

theorem simplify_expression (x : ℝ) (h : x^2 ≥ 49) :
  (7 - Real.sqrt (x^2 - 49))^2 = x^2 - 14 * Real.sqrt (x^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l787_78799


namespace NUMINAMATH_CALUDE_max_quarters_kevin_l787_78796

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- Represents the total amount of money Kevin has in dollars -/
def total_money : ℚ := 4.85

/-- 
Given that Kevin has $4.85 in U.S. coins and twice as many nickels as quarters,
prove that the maximum number of quarters he could have is 13.
-/
theorem max_quarters_kevin : 
  ∃ (q : ℕ), 
    q ≤ 13 ∧ 
    q * quarter_value + 2 * q * nickel_value ≤ total_money ∧
    ∀ (n : ℕ), n * quarter_value + 2 * n * nickel_value ≤ total_money → n ≤ q :=
sorry

end NUMINAMATH_CALUDE_max_quarters_kevin_l787_78796


namespace NUMINAMATH_CALUDE_tourist_cookie_problem_l787_78759

theorem tourist_cookie_problem :
  ∃ (n : ℕ) (k : ℕ+), 
    (2 * n ≡ 1 [MOD k]) ∧ 
    (3 * n ≡ 13 [MOD k]) → 
    k = 23 := by
  sorry

end NUMINAMATH_CALUDE_tourist_cookie_problem_l787_78759


namespace NUMINAMATH_CALUDE_total_seeds_eaten_l787_78750

def player1_seeds : ℕ := 78
def player2_seeds : ℕ := 53
def extra_seeds : ℕ := 30

def player3_seeds : ℕ := player2_seeds + extra_seeds

theorem total_seeds_eaten :
  player1_seeds + player2_seeds + player3_seeds = 214 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_eaten_l787_78750


namespace NUMINAMATH_CALUDE_sum_of_first_n_integers_second_difference_constant_sum_formula_l787_78783

def f (n : ℕ) : ℕ := (List.range n).sum + n

theorem sum_of_first_n_integers (n : ℕ) : 
  f n = n * (n + 1) / 2 :=
by sorry

theorem second_difference_constant (n : ℕ) : 
  f (n + 2) - 2 * f (n + 1) + f n = 1 :=
by sorry

theorem sum_formula (n : ℕ) : 
  (List.range n).sum + n = n * (n + 1) / 2 :=
by
  have h1 := sum_of_first_n_integers n
  have h2 := second_difference_constant n
  sorry

end NUMINAMATH_CALUDE_sum_of_first_n_integers_second_difference_constant_sum_formula_l787_78783


namespace NUMINAMATH_CALUDE_max_value_squared_l787_78752

theorem max_value_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y)
  (h : x^3 + 2013*y = y^3 + 2013*x) :
  ∃ (M : ℝ), M = (Real.sqrt 3 + 1) * x + 2 * y ∧
    ∀ (N : ℝ), N = (Real.sqrt 3 + 1) * x + 2 * y → N^2 ≤ 16104 :=
by sorry

end NUMINAMATH_CALUDE_max_value_squared_l787_78752


namespace NUMINAMATH_CALUDE_gcf_of_180_250_300_l787_78737

theorem gcf_of_180_250_300 : Nat.gcd 180 (Nat.gcd 250 300) = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_250_300_l787_78737


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l787_78784

/-- 
A four-digit number is a natural number between 1000 and 9999, inclusive.
-/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- 
Given a four-digit number, this function returns the three-digit number 
obtained by removing its leftmost digit.
-/
def RemoveLeftmostDigit (n : ℕ) : ℕ := n % 1000

/-- 
Theorem: 3500 is the only four-digit number N such that the three-digit number 
obtained by removing its leftmost digit is one-seventh of N.
-/
theorem unique_four_digit_number : 
  ∀ N : ℕ, FourDigitNumber N → 
    (RemoveLeftmostDigit N = N / 7 ↔ N = 3500) :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l787_78784


namespace NUMINAMATH_CALUDE_sin_cos_theorem_l787_78772

theorem sin_cos_theorem (θ : ℝ) (z : ℂ) : 
  z = (Real.sin θ - 2 * Real.cos θ) + (Real.sin θ + 2 * Real.cos θ) * Complex.I →
  z.re = 0 →
  z.im ≠ 0 →
  Real.sin θ * Real.cos θ = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_theorem_l787_78772


namespace NUMINAMATH_CALUDE_find_y_l787_78743

theorem find_y (x : ℕ) (y : ℕ) (h1 : 2^x - 2^(x-2) = 3 * 2^y) (h2 : x = 12) : y = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l787_78743


namespace NUMINAMATH_CALUDE_birds_in_tree_l787_78748

/-- Given 179 initial birds in a tree and 38 additional birds joining them,
    the total number of birds in the tree is 217. -/
theorem birds_in_tree (initial_birds additional_birds : ℕ) 
  (h1 : initial_birds = 179)
  (h2 : additional_birds = 38) :
  initial_birds + additional_birds = 217 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l787_78748


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l787_78727

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 50,
    where one side of the equilateral triangle is a side of the isosceles triangle,
    the base of the isosceles triangle is 10 units long. -/
theorem isosceles_triangle_base_length : ℝ → ℝ → ℝ → Prop :=
  fun equilateral_perimeter isosceles_perimeter isosceles_base =>
    equilateral_perimeter = 60 →
    isosceles_perimeter = 50 →
    let equilateral_side := equilateral_perimeter / 3
    let isosceles_side := equilateral_side
    isosceles_perimeter = 2 * isosceles_side + isosceles_base →
    isosceles_base = 10

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof :
  isosceles_triangle_base_length 60 50 10 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l787_78727


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_of_squares_l787_78714

theorem polynomial_equality_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x : ℝ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_of_squares_l787_78714


namespace NUMINAMATH_CALUDE_no_valid_sum_of_consecutive_integers_l787_78794

def sum_of_consecutive_integers (k : ℕ) : ℕ := 150 * k + 11175

def given_integers : List ℕ := [1625999850, 2344293800, 3578726150, 4691196050, 5815552000]

theorem no_valid_sum_of_consecutive_integers : 
  ∀ n ∈ given_integers, ¬ ∃ k : ℕ, sum_of_consecutive_integers k = n :=
by sorry

end NUMINAMATH_CALUDE_no_valid_sum_of_consecutive_integers_l787_78794


namespace NUMINAMATH_CALUDE_race_solution_l787_78734

/-- Race between A and B from M to N and back -/
structure Race where
  distance : ℝ  -- Distance between M and N
  time_A : ℝ    -- Time taken by A
  time_B : ℝ    -- Time taken by B

/-- Conditions of the race -/
def race_conditions (r : Race) : Prop :=
  -- A reaches N sooner than B
  r.time_A < r.time_B
  -- A meets B 100 meters before N on the way back
  ∧ ∃ t : ℝ, t < r.time_A ∧ t * (r.distance / r.time_A) = (2 * r.distance - 100)
  -- A arrives at M 4 minutes earlier than B
  ∧ r.time_B = r.time_A + 4
  -- If A turns around at M, they meet B at 1/5 of the M to N distance
  ∧ ∃ t : ℝ, t < r.time_A ∧ t * (r.distance / r.time_A) = (1/5) * r.distance

/-- The theorem to be proved -/
theorem race_solution :
  ∃ r : Race, race_conditions r ∧ r.distance = 1000 ∧ r.time_A = 18 ∧ r.time_B = 22 := by
  sorry

end NUMINAMATH_CALUDE_race_solution_l787_78734


namespace NUMINAMATH_CALUDE_lcm_1806_1230_l787_78731

theorem lcm_1806_1230 : Nat.lcm 1806 1230 = 247230 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1806_1230_l787_78731


namespace NUMINAMATH_CALUDE_factorization_am2_minus_an2_l787_78718

theorem factorization_am2_minus_an2 (a m n : ℝ) : a * m^2 - a * n^2 = a * (m + n) * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_am2_minus_an2_l787_78718


namespace NUMINAMATH_CALUDE_appropriate_word_count_l787_78755

-- Define the presentation parameters
def min_duration : ℕ := 40
def max_duration : ℕ := 50
def speech_rate : ℕ := 160

-- Define the range of appropriate word counts
def min_words : ℕ := min_duration * speech_rate
def max_words : ℕ := max_duration * speech_rate

-- Theorem statement
theorem appropriate_word_count (word_count : ℕ) :
  (min_words ≤ word_count ∧ word_count ≤ max_words) ↔
  (word_count ≥ 6400 ∧ word_count ≤ 8000) :=
by sorry

end NUMINAMATH_CALUDE_appropriate_word_count_l787_78755


namespace NUMINAMATH_CALUDE_gcf_of_48_160_120_l787_78765

theorem gcf_of_48_160_120 : Nat.gcd 48 (Nat.gcd 160 120) = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_48_160_120_l787_78765


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l787_78760

theorem angle_in_second_quadrant (α : Real) (x : Real) :
  -- α is in the second quadrant
  π / 2 < α ∧ α < π →
  -- p(x, √5) is on the terminal side of α
  ∃ (r : Real), r > 0 ∧ x = r * Real.cos α ∧ Real.sqrt 5 = r * Real.sin α →
  -- cos α = (√2/4)x
  Real.cos α = (Real.sqrt 2 / 4) * x →
  -- x = -√3
  x = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l787_78760


namespace NUMINAMATH_CALUDE_annes_speed_l787_78742

/-- Given a distance of 6 miles traveled in 3 hours, prove that the speed is 2 miles per hour. -/
theorem annes_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 6 → time = 3 → speed = distance / time → speed = 2 := by sorry

end NUMINAMATH_CALUDE_annes_speed_l787_78742


namespace NUMINAMATH_CALUDE_distance_specific_point_to_line_l787_78707

/-- The distance from a point to a line in 3D space -/
def distance_point_to_line (point : ℝ × ℝ × ℝ) (line_point : ℝ × ℝ × ℝ) (line_direction : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem: The distance from (2, -1, 4) to the line (4, 3, 9) + t(1, -1, 3) is 65/11 -/
theorem distance_specific_point_to_line :
  let point : ℝ × ℝ × ℝ := (2, -1, 4)
  let line_point : ℝ × ℝ × ℝ := (4, 3, 9)
  let line_direction : ℝ × ℝ × ℝ := (1, -1, 3)
  distance_point_to_line point line_point line_direction = 65 / 11 :=
by sorry

end NUMINAMATH_CALUDE_distance_specific_point_to_line_l787_78707


namespace NUMINAMATH_CALUDE_minimum_distance_point_to_curve_l787_78774

open Real

theorem minimum_distance_point_to_curve (t m : ℝ) : 
  (∃ (P : ℝ × ℝ), 
    P.2 = exp P.1 ∧ 
    (∀ (Q : ℝ × ℝ), Q.2 = exp Q.1 → (t - P.1)^2 + P.2^2 ≤ (t - Q.1)^2 + Q.2^2) ∧
    (t - P.1)^2 + P.2^2 = 12) →
  t = 3 + log 3 / 2 := by
sorry


end NUMINAMATH_CALUDE_minimum_distance_point_to_curve_l787_78774


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_43_l787_78706

theorem smallest_four_digit_divisible_by_43 : 
  ∃ n : ℕ, 
    (n ≥ 1000 ∧ n < 10000) ∧ 
    n % 43 = 0 ∧
    (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) → m % 43 = 0 → m ≥ n) ∧
    n = 1032 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_43_l787_78706


namespace NUMINAMATH_CALUDE_seating_arrangements_mod_1000_l787_78751

/-- Represents a seating arrangement of ambassadors and advisors. -/
structure SeatingArrangement where
  ambassador_seats : Finset (Fin 6)
  advisor_seats : Finset (Fin 12)

/-- The set of all valid seating arrangements. -/
def validArrangements : Finset SeatingArrangement :=
  sorry

/-- The number of valid seating arrangements. -/
def N : ℕ := Finset.card validArrangements

/-- Theorem stating that the number of valid seating arrangements
    is congruent to 520 modulo 1000. -/
theorem seating_arrangements_mod_1000 :
  N % 1000 = 520 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_mod_1000_l787_78751


namespace NUMINAMATH_CALUDE_sqrt_simplification_l787_78721

theorem sqrt_simplification : 
  Real.sqrt 45 - 2 * Real.sqrt 5 + Real.sqrt 360 / Real.sqrt 2 = Real.sqrt 245 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l787_78721


namespace NUMINAMATH_CALUDE_min_value_f_inequality_abc_l787_78704

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - 2)

-- Theorem for the minimum value of f(x)
theorem min_value_f : ∀ x : ℝ, f x ≥ 3 := by sorry

-- Theorem for the inequality
theorem inequality_abc (a b c m : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hm : a + b + c = m) :
  b^2 / a + c^2 / b + a^2 / c ≥ 3 := by sorry

end NUMINAMATH_CALUDE_min_value_f_inequality_abc_l787_78704


namespace NUMINAMATH_CALUDE_existence_of_abc_l787_78705

theorem existence_of_abc (n : ℕ) (A : Finset ℕ) :
  A ⊆ Finset.range (5^n + 1) →
  A.card = 4*n + 2 →
  ∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a < b ∧ b < c ∧ c + 2*a > 3*b :=
sorry

end NUMINAMATH_CALUDE_existence_of_abc_l787_78705


namespace NUMINAMATH_CALUDE_triangle_area_angle_l787_78762

theorem triangle_area_angle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (a^2 + b^2 - c^2) / (4 * Real.sqrt 3)
  ∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π ∧
    S = 1/2 * a * b * Real.sin C ∧
    C = π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_angle_l787_78762


namespace NUMINAMATH_CALUDE_projection_a_on_b_l787_78728

def a : ℝ × ℝ := (-1, 3)
def b : ℝ × ℝ := (3, -4)

theorem projection_a_on_b :
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -3 := by sorry

end NUMINAMATH_CALUDE_projection_a_on_b_l787_78728


namespace NUMINAMATH_CALUDE_odd_function_sum_l787_78710

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_sum (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : HasPeriod (fun x ↦ f (2 * x + 1)) 5) (h3 : f 1 = 5) :
  f 2009 + f 2010 = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_sum_l787_78710


namespace NUMINAMATH_CALUDE_number_and_square_sum_l787_78764

theorem number_and_square_sum (x : ℝ) : x + x^2 = 132 → x = 11 ∨ x = -12 := by
  sorry

end NUMINAMATH_CALUDE_number_and_square_sum_l787_78764


namespace NUMINAMATH_CALUDE_park_to_circus_route_comparison_l787_78702

/-- Represents a circular tram line with three stops -/
structure TramLine where
  circumference : ℝ
  park_to_zoo : ℝ
  circus_to_zoo : ℝ

/-- The properties of the tram line as described in the problem -/
def tram_line_properties (t : TramLine) : Prop :=
  t.park_to_zoo > 0 ∧
  t.circus_to_zoo > 0 ∧
  t.circumference > 0 ∧
  t.park_to_zoo < t.circumference ∧
  t.circus_to_zoo < t.circumference ∧
  3 * t.park_to_zoo = t.circumference - t.park_to_zoo ∧
  2 * (t.circumference - t.park_to_zoo - t.circus_to_zoo) = t.circus_to_zoo

theorem park_to_circus_route_comparison (t : TramLine) 
  (h : tram_line_properties t) : 
  (t.circumference - t.park_to_zoo - t.circus_to_zoo) * 11 = t.park_to_zoo + t.circus_to_zoo :=
sorry

end NUMINAMATH_CALUDE_park_to_circus_route_comparison_l787_78702


namespace NUMINAMATH_CALUDE_no_two_common_tangents_l787_78739

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Number of common tangents between two circles --/
def commonTangents (c1 c2 : Circle) : ℕ := sorry

theorem no_two_common_tangents (c1 c2 : Circle) (h : c1.radius ≠ c2.radius) :
  commonTangents c1 c2 ≠ 2 := by sorry

end NUMINAMATH_CALUDE_no_two_common_tangents_l787_78739


namespace NUMINAMATH_CALUDE_polygon_150_sides_diagonals_l787_78780

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 150 sides has 11025 diagonals -/
theorem polygon_150_sides_diagonals : num_diagonals 150 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_polygon_150_sides_diagonals_l787_78780


namespace NUMINAMATH_CALUDE_ab_and_c_values_l787_78771

theorem ab_and_c_values (a b c : ℤ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) 
  (h3 : a + b + c = 10) : 
  a * b = 10 ∧ (c = 3 ∨ c = 17) := by
sorry

end NUMINAMATH_CALUDE_ab_and_c_values_l787_78771


namespace NUMINAMATH_CALUDE_min_pieces_same_color_l787_78749

theorem min_pieces_same_color (total_pieces : ℕ) (pieces_per_color : ℕ) (h1 : total_pieces = 60) (h2 : pieces_per_color = 15) :
  ∃ (min_pieces : ℕ), 
    (∀ (n : ℕ), n < min_pieces → ∃ (selection : Finset ℕ), selection.card = n ∧ 
      ∀ (i j : ℕ), i ∈ selection → j ∈ selection → i ≠ j → (i / pieces_per_color) ≠ (j / pieces_per_color)) ∧
    (∃ (selection : Finset ℕ), selection.card = min_pieces ∧ 
      ∃ (i j : ℕ), i ∈ selection ∧ j ∈ selection ∧ i ≠ j ∧ (i / pieces_per_color) = (j / pieces_per_color)) ∧
    min_pieces = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_pieces_same_color_l787_78749


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l787_78740

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l787_78740


namespace NUMINAMATH_CALUDE_polynomial_value_l787_78746

theorem polynomial_value (x : ℝ) (h : x^2 - 2*x + 6 = 9) : 2*x^2 - 4*x + 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l787_78746


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l787_78709

theorem tangent_sum_simplification :
  (Real.tan (30 * π / 180) + Real.tan (40 * π / 180) + Real.tan (50 * π / 180) + Real.tan (60 * π / 180)) / Real.cos (20 * π / 180) = 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l787_78709


namespace NUMINAMATH_CALUDE_min_abs_z_l787_78785

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 7) + Complex.abs (z - 6*I) = 15) :
  ∃ (w : ℂ), Complex.abs w = 14/5 ∧ ∀ (v : ℂ), Complex.abs (v - 7) + Complex.abs (v - 6*I) = 15 → Complex.abs v ≥ Complex.abs w :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_l787_78785


namespace NUMINAMATH_CALUDE_jill_final_llama_count_l787_78722

/-- Represents the number of llamas Jill has after all operations -/
def final_llama_count (single_calf_llamas twin_calf_llamas traded_calves new_adults : ℕ) : ℕ :=
  let initial_llamas := single_calf_llamas + twin_calf_llamas
  let total_calves := single_calf_llamas + 2 * twin_calf_llamas
  let remaining_calves := total_calves - traded_calves
  let total_before_sale := initial_llamas + remaining_calves + new_adults
  total_before_sale - (total_before_sale / 3)

/-- Theorem stating that Jill ends up with 18 llamas given the initial conditions -/
theorem jill_final_llama_count :
  final_llama_count 9 5 8 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_jill_final_llama_count_l787_78722


namespace NUMINAMATH_CALUDE_floor_product_equation_l787_78753

theorem floor_product_equation (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 29 ↔ x ≥ 5.8 ∧ x < 6 :=
sorry

end NUMINAMATH_CALUDE_floor_product_equation_l787_78753


namespace NUMINAMATH_CALUDE_class_average_problem_l787_78736

theorem class_average_problem (total_students : ℕ) 
  (high_score_students : ℕ) (zero_score_students : ℕ) 
  (high_score : ℕ) (class_average : ℕ) :
  total_students = 25 →
  high_score_students = 3 →
  zero_score_students = 5 →
  high_score = 95 →
  class_average = 42 →
  let remaining_students := total_students - (high_score_students + zero_score_students)
  let total_score := total_students * class_average
  let high_score_total := high_score_students * high_score
  let remaining_score := total_score - high_score_total
  remaining_score / remaining_students = 45 := by
sorry

end NUMINAMATH_CALUDE_class_average_problem_l787_78736


namespace NUMINAMATH_CALUDE_reflection_of_P_is_correct_l787_78777

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflection_of_P_is_correct : 
  let P : Point := { x := 2, y := -3 }
  reflectXAxis P = { x := 2, y := 3 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_P_is_correct_l787_78777


namespace NUMINAMATH_CALUDE_h_ratio_theorem_l787_78732

/-- Sum of even integers from 2 to n, inclusive, for even n -/
def h (n : ℕ) : ℚ :=
  if n % 2 = 0 then (n / 2) * (n + 2) / 4 else 0

theorem h_ratio_theorem (m k n : ℕ) (h_even : Even n) :
  h (m * n) / h (k * n) = (m : ℚ) / k * (m / k + 1) := by
  sorry

end NUMINAMATH_CALUDE_h_ratio_theorem_l787_78732


namespace NUMINAMATH_CALUDE_total_sum_lent_total_sum_lent_proof_l787_78733

/-- Proves that the total sum lent is 2795 rupees given the problem conditions -/
theorem total_sum_lent : ℕ → Prop := fun total_sum =>
  ∃ (first_part second_part : ℕ),
    -- The sum is divided into two parts
    total_sum = first_part + second_part ∧
    -- Interest on first part for 8 years at 3% per annum equals interest on second part for 3 years at 5% per annum
    (first_part * 3 * 8) = (second_part * 5 * 3) ∧
    -- The second part is Rs. 1720
    second_part = 1720 ∧
    -- The total sum lent is 2795 rupees
    total_sum = 2795

/-- The proof of the theorem -/
theorem total_sum_lent_proof : total_sum_lent 2795 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_lent_total_sum_lent_proof_l787_78733


namespace NUMINAMATH_CALUDE_tangent_r_values_l787_78730

-- Define the curves C and C1
def C (x y : ℝ) : Prop := (x - 0)^2 + (y - 2)^2 = 4

def C1 (x y r : ℝ) : Prop := ∃ α, x = 3 + r * Real.cos α ∧ y = -2 + r * Real.sin α

-- Define the tangency condition
def are_tangent (r : ℝ) : Prop :=
  ∃ x y, C x y ∧ C1 x y r

-- Theorem statement
theorem tangent_r_values :
  ∀ r : ℝ, are_tangent r ↔ r = 3 ∨ r = -3 ∨ r = 7 ∨ r = -7 :=
sorry

end NUMINAMATH_CALUDE_tangent_r_values_l787_78730


namespace NUMINAMATH_CALUDE_diamond_ratio_equals_five_thirds_l787_78726

-- Define the diamond operation
def diamond (n m : ℤ) : ℤ := n^2 * m^3

-- Theorem statement
theorem diamond_ratio_equals_five_thirds :
  (diamond 3 5) / (diamond 5 3) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_diamond_ratio_equals_five_thirds_l787_78726


namespace NUMINAMATH_CALUDE_equation_solutions_l787_78741

def is_solution (x y : ℤ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 1987

def solution_count : ℕ := 5

theorem equation_solutions :
  (∃! (s : Finset (ℤ × ℤ)), s.card = solution_count ∧
    ∀ (p : ℤ × ℤ), p ∈ s ↔ is_solution p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l787_78741


namespace NUMINAMATH_CALUDE_carnival_earnings_value_l787_78775

/-- The total earnings from two ring toss games at a carnival -/
def carnival_earnings : ℕ :=
  let game1_period1 := 88
  let game1_rate1 := 761
  let game1_period2 := 20
  let game1_rate2 := 487
  let game2_period1 := 66
  let game2_rate1 := 569
  let game2_period2 := 15
  let game2_rate2 := 932
  let game1_earnings := game1_period1 * game1_rate1 + game1_period2 * game1_rate2
  let game2_earnings := game2_period1 * game2_rate1 + game2_period2 * game2_rate2
  game1_earnings + game2_earnings

theorem carnival_earnings_value : carnival_earnings = 128242 := by
  sorry

end NUMINAMATH_CALUDE_carnival_earnings_value_l787_78775


namespace NUMINAMATH_CALUDE_fraction_equality_l787_78768

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4*x + y) / (x - 4*y) = 3) : 
  (x + 4*y) / (4*x - y) = 9/53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l787_78768
