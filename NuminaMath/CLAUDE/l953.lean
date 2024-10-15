import Mathlib

namespace NUMINAMATH_CALUDE_equation_roots_l953_95349

theorem equation_roots (m n : ℝ) (hm : m ≠ 0) 
  (h : 2 * m * (-3)^2 - n * (-3) + 2 = 0) : 
  ∃ (x y : ℝ), 2 * m * x^2 + n * x + 2 = 0 ∧ 2 * m * y^2 + n * y + 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_equation_roots_l953_95349


namespace NUMINAMATH_CALUDE_largest_integer_solution_l953_95314

theorem largest_integer_solution (x : ℤ) : (∀ y : ℤ, -y + 3 > 1 → y ≤ x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l953_95314


namespace NUMINAMATH_CALUDE_value_two_std_dev_below_mean_l953_95368

/-- Given a normal distribution with mean 14.5 and standard deviation 1.7,
    the value that is exactly 2 standard deviations less than the mean is 11.1 -/
theorem value_two_std_dev_below_mean :
  let μ : ℝ := 14.5  -- mean
  let σ : ℝ := 1.7   -- standard deviation
  μ - 2 * σ = 11.1 := by
  sorry

end NUMINAMATH_CALUDE_value_two_std_dev_below_mean_l953_95368


namespace NUMINAMATH_CALUDE_remainder_equality_l953_95391

def r (n : ℕ) : ℕ := n % 6

theorem remainder_equality (n : ℕ) : 
  r (2 * n + 3) = r (5 * n + 6) ↔ ∃ k : ℤ, n = 2 * k - 1 := by sorry

end NUMINAMATH_CALUDE_remainder_equality_l953_95391


namespace NUMINAMATH_CALUDE_probability_adjacent_circular_probability_two_adjacent_in_six_l953_95375

def num_people : ℕ := 6

def total_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

def adjacent_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 2)

theorem probability_adjacent_circular (n : ℕ) (h : n ≥ 3) :
  (adjacent_arrangements n : ℚ) / (total_arrangements n : ℚ) = 2 / (n - 1 : ℚ) :=
sorry

theorem probability_two_adjacent_in_six :
  (adjacent_arrangements num_people : ℚ) / (total_arrangements num_people : ℚ) = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_adjacent_circular_probability_two_adjacent_in_six_l953_95375


namespace NUMINAMATH_CALUDE_f_max_value_l953_95318

/-- The quadratic function f(x) = -9x^2 + 27x + 15 -/
def f (x : ℝ) : ℝ := -9 * x^2 + 27 * x + 15

/-- The maximum value of f(x) is 35.25 -/
theorem f_max_value : ∃ (M : ℝ), M = 35.25 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l953_95318


namespace NUMINAMATH_CALUDE_triangular_array_coin_sum_l953_95356

/-- The sum of the first n odd numbers -/
def triangular_sum (n : ℕ) : ℕ := n^2

/-- The sum of the digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_coin_sum :
  ∃ (n : ℕ), triangular_sum n = 3081 ∧ sum_of_digits n = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_coin_sum_l953_95356


namespace NUMINAMATH_CALUDE_number_of_possible_sums_l953_95306

/-- The set of chips in Bag A -/
def bagA : Finset ℕ := {1, 4, 5}

/-- The set of chips in Bag B -/
def bagB : Finset ℕ := {2, 4, 6}

/-- The set of all possible sums when drawing one chip from each bag -/
def possibleSums : Finset ℕ := (bagA.product bagB).image (fun p => p.1 + p.2)

theorem number_of_possible_sums : Finset.card possibleSums = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_possible_sums_l953_95306


namespace NUMINAMATH_CALUDE_hyperbola_equation_l953_95364

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    one of its asymptotes is perpendicular to the line l: x - 2y - 5 = 0,
    and one of its foci lies on line l,
    prove that the equation of the hyperbola is x²/5 - y²/20 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_asymptote : ∃ (m : ℝ), m * (1/2) = -1 ∧ m = b/a)
  (h_focus : ∃ (x y : ℝ), x - 2*y - 5 = 0 ∧ x^2/a^2 - y^2/b^2 = 1 ∧ x^2 - (a^2 + b^2) = 0) :
  a^2 = 5 ∧ b^2 = 20 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l953_95364


namespace NUMINAMATH_CALUDE_sum_of_digits_seven_pow_nineteen_l953_95393

/-- The sum of the tens digit and the ones digit of 7^19 is 7 -/
theorem sum_of_digits_seven_pow_nineteen : 
  (((7^19) / 10) % 10) + ((7^19) % 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_seven_pow_nineteen_l953_95393


namespace NUMINAMATH_CALUDE_prop_2_correct_prop_4_correct_prop_1_not_necessarily_true_prop_3_not_necessarily_true_l953_95303

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Proposition 2
theorem prop_2_correct 
  (a b : Line) (α : Plane) 
  (h1 : parallel_line_plane a α) 
  (h2 : perpendicular_line_plane b α) : 
  perpendicular_lines a b :=
sorry

-- Proposition 4
theorem prop_4_correct 
  (a : Line) (α β : Plane) 
  (h1 : perpendicular_line_plane a α) 
  (h2 : parallel_line_plane a β) : 
  perpendicular_planes α β :=
sorry

-- Proposition 1 is not necessarily true
theorem prop_1_not_necessarily_true :
  ¬ ∀ (a b : Line) (α β : Plane),
    parallel_line_plane a α → parallel_line_plane b β → parallel_lines a b :=
sorry

-- Proposition 3 is not necessarily true
theorem prop_3_not_necessarily_true :
  ¬ ∀ (a b : Line) (α : Plane),
    parallel_lines a b → parallel_line_plane b α → parallel_line_plane a α :=
sorry

end NUMINAMATH_CALUDE_prop_2_correct_prop_4_correct_prop_1_not_necessarily_true_prop_3_not_necessarily_true_l953_95303


namespace NUMINAMATH_CALUDE_concert_attendees_l953_95398

theorem concert_attendees :
  let num_buses : ℕ := 8
  let students_per_bus : ℕ := 45
  let chaperones_per_bus : List ℕ := [2, 3, 4, 5, 3, 4, 2, 6]
  let total_students : ℕ := num_buses * students_per_bus
  let total_chaperones : ℕ := chaperones_per_bus.sum
  let total_attendees : ℕ := total_students + total_chaperones
  total_attendees = 389 := by
  sorry


end NUMINAMATH_CALUDE_concert_attendees_l953_95398


namespace NUMINAMATH_CALUDE_four_digit_number_property_l953_95370

theorem four_digit_number_property : ∃ (a b c d : ℕ), 
  (a ≥ 1 ∧ a ≤ 9) ∧ 
  (b ≥ 0 ∧ b ≤ 9) ∧ 
  (c ≥ 0 ∧ c ≤ 9) ∧ 
  (d ≥ 0 ∧ d ≤ 9) ∧ 
  (a * 1000 + b * 100 + c * 10 + d ≥ 1000) ∧
  (a * 1000 + b * 100 + c * 10 + d ≤ 9999) ∧
  ((a + b + c + d) * (a * b * c * d) = 3990) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_property_l953_95370


namespace NUMINAMATH_CALUDE_problem_solution_l953_95320

theorem problem_solution (x y : ℝ) : 
  (65 / 100 : ℝ) * 900 = (40 / 100 : ℝ) * x → 
  (35 / 100 : ℝ) * 1200 = (25 / 100 : ℝ) * y → 
  x + y = 3142.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l953_95320


namespace NUMINAMATH_CALUDE_cost_reduction_equation_l953_95326

theorem cost_reduction_equation (x : ℝ) : 
  (∀ (total_reduction : ℝ), total_reduction = 0.36 → 
    ((1 - x) ^ 2 = 1 - total_reduction)) ↔ 
  ((1 - x) ^ 2 = 1 - 0.36) :=
sorry

end NUMINAMATH_CALUDE_cost_reduction_equation_l953_95326


namespace NUMINAMATH_CALUDE_line_vector_proof_l953_95317

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 0 = (1, 5, 9)) →
  (line_vector 1 = (6, 0, 4)) →
  (line_vector 4 = (21, -15, -11)) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_proof_l953_95317


namespace NUMINAMATH_CALUDE_simple_interest_problem_l953_95392

theorem simple_interest_problem (interest : ℚ) (rate : ℚ) (time : ℚ) (principal : ℚ) : 
  interest = 750 →
  rate = 6 / 100 →
  time = 5 →
  principal * rate * time = interest →
  principal = 2500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l953_95392


namespace NUMINAMATH_CALUDE_third_day_income_l953_95345

def cab_driver_income (day1 day2 day4 day5 : ℕ) (average : ℚ) : Prop :=
  ∃ day3 : ℕ,
    (day1 + day2 + day3 + day4 + day5 : ℚ) / 5 = average ∧
    day3 = 60

theorem third_day_income :
  cab_driver_income 45 50 65 70 58 :=
sorry

end NUMINAMATH_CALUDE_third_day_income_l953_95345


namespace NUMINAMATH_CALUDE_people_per_car_l953_95373

/-- Given 3.0 cars and 189 people going to the zoo, prove that there are 63 people in each car. -/
theorem people_per_car (total_cars : Float) (total_people : Nat) : 
  total_cars = 3.0 → total_people = 189 → (total_people.toFloat / total_cars).round = 63 := by
  sorry

end NUMINAMATH_CALUDE_people_per_car_l953_95373


namespace NUMINAMATH_CALUDE_count_even_four_digit_is_784_l953_95307

/-- Count of even integers between 3000 and 6000 with four different digits -/
def count_even_four_digit : ℕ := sorry

/-- An integer is between 3000 and 6000 -/
def is_between_3000_and_6000 (n : ℕ) : Prop :=
  3000 < n ∧ n < 6000

/-- An integer has four different digits -/
def has_four_different_digits (n : ℕ) : Prop := sorry

/-- Theorem stating that the count of even integers between 3000 and 6000
    with four different digits is 784 -/
theorem count_even_four_digit_is_784 :
  count_even_four_digit = 784 := by sorry

end NUMINAMATH_CALUDE_count_even_four_digit_is_784_l953_95307


namespace NUMINAMATH_CALUDE_max_value_of_f_l953_95361

/-- The function we're maximizing -/
def f (t : ℤ) : ℚ := (3^t - 2*t) * t / 9^t

/-- The theorem stating the maximum value of the function -/
theorem max_value_of_f :
  ∃ (max : ℚ), max = 1/8 ∧ ∀ (t : ℤ), f t ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l953_95361


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l953_95386

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio of 5:2,
    its diagonal length is 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 72) →
  (length / width = 5 / 2) →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l953_95386


namespace NUMINAMATH_CALUDE_max_daily_net_income_l953_95371

/-- Represents the daily rental fee for each electric car -/
def x : ℕ → ℕ := fun n => n

/-- Represents the daily net income from renting out electric cars -/
def y : ℕ → ℤ
| n =>
  if 60 ≤ n ∧ n ≤ 90 then
    750 * n - 1700
  else if 90 < n ∧ n ≤ 300 then
    -3 * n * n + 1020 * n - 1700
  else
    0

/-- The theorem stating the maximum daily net income and the corresponding rental fee -/
theorem max_daily_net_income :
  ∃ (n : ℕ), 60 ≤ n ∧ n ≤ 300 ∧ y n = 85000 ∧ n = 170 ∧
  ∀ (m : ℕ), 60 ≤ m ∧ m ≤ 300 → y m ≤ y n :=
sorry

end NUMINAMATH_CALUDE_max_daily_net_income_l953_95371


namespace NUMINAMATH_CALUDE_aristocrat_problem_l953_95397

theorem aristocrat_problem (total_people : ℕ) 
  (men_payment women_payment total_spent : ℚ) 
  (women_fraction : ℚ) :
  total_people = 3552 →
  men_payment = 45 →
  women_payment = 60 →
  women_fraction = 1 / 12 →
  total_spent = 17760 →
  ∃ (men_fraction : ℚ),
    men_fraction * men_payment * (total_people - (women_fraction⁻¹ * women_fraction * total_people)) + 
    women_fraction * women_payment * (women_fraction⁻¹ * women_fraction * total_people) = total_spent ∧
    men_fraction = 1 / 9 :=
by sorry

end NUMINAMATH_CALUDE_aristocrat_problem_l953_95397


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_l953_95372

/-- A trinomial ax^2 + bx + c is a perfect square if there exists a real number r
    such that ax^2 + bx + c = (√a * x + r)^2 for all x. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (Real.sqrt a * x + r)^2

theorem perfect_square_trinomial_m (m : ℝ) :
  IsPerfectSquareTrinomial 1 (-m) 16 → m = 8 ∨ m = -8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_l953_95372


namespace NUMINAMATH_CALUDE_triangle_arithmetic_angle_sequence_side_relation_l953_95336

open Real

/-- Given a triangle ABC with sides a, b, c and angles A, B, C (in radians),
    where A, B, C form an arithmetic sequence, 
    prove that 1/(a+b) + 1/(b+c) = 3/(a+b+c) -/
theorem triangle_arithmetic_angle_sequence_side_relation 
  (a b c A B C : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_angles : 0 < A ∧ 0 < B ∧ 0 < C)
  (h_sum_angles : A + B + C = π)
  (h_arithmetic_seq : ∃ d : ℝ, B = A + d ∧ C = B + d) :
  1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_angle_sequence_side_relation_l953_95336


namespace NUMINAMATH_CALUDE_locus_general_case_locus_special_case_l953_95332

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop := sorry

-- Define a point inside a triangle
def InsideTriangle (S : ℝ × ℝ) (P Q R : ℝ × ℝ) : Prop := sorry

-- Define a segment on a side of a triangle
def SegmentOnSide (A B : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop := sorry

-- Define the area of a triangle
def AreaTriangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define a line segment parallel to another line segment
def ParallelSegment (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the locus of points
def Locus (S : ℝ × ℝ) (P Q R : ℝ × ℝ) (A B C D E F : ℝ × ℝ) (S₀ : ℝ × ℝ) : Prop :=
  InsideTriangle S P Q R ∧
  AreaTriangle S A B + AreaTriangle S C D + AreaTriangle S E F =
  AreaTriangle S₀ A B + AreaTriangle S₀ C D + AreaTriangle S₀ E F

-- Theorem for the general case
theorem locus_general_case 
  (P Q R : ℝ × ℝ) 
  (A B C D E F : ℝ × ℝ) 
  (S₀ : ℝ × ℝ) 
  (h1 : Triangle P Q R)
  (h2 : SegmentOnSide A B P Q)
  (h3 : SegmentOnSide C D Q R)
  (h4 : SegmentOnSide E F R P)
  (h5 : InsideTriangle S₀ P Q R) :
  ∃ D' E' : ℝ × ℝ, 
    ParallelSegment D' E' C D ∧ 
    (∀ S : ℝ × ℝ, Locus S P Q R A B C D E F S₀ ↔ 
      (S = S₀ ∨ ParallelSegment S S₀ D' E')) :=
sorry

-- Theorem for the special case
theorem locus_special_case
  (P Q R : ℝ × ℝ) 
  (A B C D E F : ℝ × ℝ) 
  (S₀ : ℝ × ℝ) 
  (h1 : Triangle P Q R)
  (h2 : SegmentOnSide A B P Q)
  (h3 : SegmentOnSide C D Q R)
  (h4 : SegmentOnSide E F R P)
  (h5 : InsideTriangle S₀ P Q R)
  (h6 : ∃ k : ℝ, k > 0 ∧ 
    ‖A - B‖ / ‖P - Q‖ = k ∧ 
    ‖C - D‖ / ‖Q - R‖ = k ∧ 
    ‖E - F‖ / ‖R - P‖ = k) :
  ∀ S : ℝ × ℝ, InsideTriangle S P Q R → Locus S P Q R A B C D E F S₀ :=
sorry

end NUMINAMATH_CALUDE_locus_general_case_locus_special_case_l953_95332


namespace NUMINAMATH_CALUDE_repeating_decimal_sqrt_pairs_l953_95381

def is_valid_pair (a b : Nat) : Prop :=
  a ≤ 9 ∧ b ≤ 9 ∧ (b * b = 9 * a)

theorem repeating_decimal_sqrt_pairs :
  ∀ a b : Nat, is_valid_pair a b ↔ 
    (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 9 ∧ b = 9) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sqrt_pairs_l953_95381


namespace NUMINAMATH_CALUDE_simplify_fraction_l953_95312

theorem simplify_fraction (b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ -1/2) :
  1 - 1 / (1 + b / (1 + b)) = b / (1 + 2*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l953_95312


namespace NUMINAMATH_CALUDE_compute_expression_l953_95347

theorem compute_expression : 6^3 - 5*7 + 2^4 = 197 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l953_95347


namespace NUMINAMATH_CALUDE_homework_completion_l953_95300

theorem homework_completion (total : ℕ) (math : ℕ) (korean : ℕ) 
  (h1 : total = 48) 
  (h2 : math = 37) 
  (h3 : korean = 42) 
  (h4 : math + korean - total ≥ 0) : 
  math + korean - total = 31 := by
  sorry

end NUMINAMATH_CALUDE_homework_completion_l953_95300


namespace NUMINAMATH_CALUDE_discount_is_25_percent_l953_95367

-- Define the cost of one photocopy
def cost_per_copy : ℚ := 2 / 100

-- Define the number of copies for each person
def copies_per_person : ℕ := 80

-- Define the total number of copies in the combined order
def total_copies : ℕ := 2 * copies_per_person

-- Define the savings per person
def savings_per_person : ℚ := 40 / 100

-- Define the total savings
def total_savings : ℚ := 2 * savings_per_person

-- Define the total cost without discount
def total_cost_without_discount : ℚ := total_copies * cost_per_copy

-- Define the total cost with discount
def total_cost_with_discount : ℚ := total_cost_without_discount - total_savings

-- Define the discount percentage
def discount_percentage : ℚ := (total_savings / total_cost_without_discount) * 100

-- Theorem statement
theorem discount_is_25_percent : discount_percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_discount_is_25_percent_l953_95367


namespace NUMINAMATH_CALUDE_blackboard_numbers_l953_95311

theorem blackboard_numbers (a b : ℕ) : 
  (¬ ∃ a b : ℕ, 13 * a + 11 * b = 86) ∧ 
  (∃ a b : ℕ, 13 * a + 11 * b = 2015) := by
  sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l953_95311


namespace NUMINAMATH_CALUDE_initial_persons_count_l953_95330

/-- The number of persons initially in the group -/
def n : ℕ := sorry

/-- The average weight increase when a new person replaces one person -/
def average_increase : ℚ := 5/2

/-- The weight difference between the new person and the replaced person -/
def weight_difference : ℕ := 20

/-- Theorem stating that the initial number of persons is 8 -/
theorem initial_persons_count : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_persons_count_l953_95330


namespace NUMINAMATH_CALUDE_k_range_l953_95310

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := x^2 - x > 2

-- Define what it means for p to be sufficient but not necessary for q
def sufficient_not_necessary (k : ℝ) : Prop :=
  (∀ x, p x k → q x) ∧ (∃ x, q x ∧ ¬p x k)

-- Theorem statement
theorem k_range :
  ∀ k : ℝ, sufficient_not_necessary k ↔ k > 2 :=
sorry

end NUMINAMATH_CALUDE_k_range_l953_95310


namespace NUMINAMATH_CALUDE_complex_equation_proof_l953_95379

theorem complex_equation_proof (z : ℂ) (h : z = 1 - Complex.I) : z^2 - 2*z + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l953_95379


namespace NUMINAMATH_CALUDE_roden_gold_fish_l953_95382

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := 22 - 7

/-- Theorem stating that Roden bought 15 gold fish -/
theorem roden_gold_fish : gold_fish = 15 := by
  sorry

end NUMINAMATH_CALUDE_roden_gold_fish_l953_95382


namespace NUMINAMATH_CALUDE_sandwich_combinations_l953_95325

theorem sandwich_combinations (num_meats num_cheeses : ℕ) : 
  num_meats = 12 → num_cheeses = 11 → 
  (num_meats * num_cheeses) + (num_meats * (num_cheeses.choose 2)) = 792 := by
  sorry

#check sandwich_combinations

end NUMINAMATH_CALUDE_sandwich_combinations_l953_95325


namespace NUMINAMATH_CALUDE_m_plus_one_value_l953_95348

theorem m_plus_one_value (m n : ℕ) 
  (h1 : m * n = 121) 
  (h2 : (m + 1) * (n + 1) = 1000) : 
  m + 1 = 879 - n := by
sorry

end NUMINAMATH_CALUDE_m_plus_one_value_l953_95348


namespace NUMINAMATH_CALUDE_floor_of_e_equals_two_l953_95301

-- Define e as the base of natural logarithms
noncomputable def e : ℝ := Real.exp 1

-- Theorem statement
theorem floor_of_e_equals_two : ⌊e⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_e_equals_two_l953_95301


namespace NUMINAMATH_CALUDE_parallelogram_area_l953_95304

/-- Given two 2D vectors u and z, this theorem proves that the area of the parallelogram
    formed by u and z + u is 3. -/
theorem parallelogram_area (u z : Fin 2 → ℝ) (hu : u = ![4, -1]) (hz : z = ![9, -3]) :
  let z' := z + u
  abs (u 0 * z' 1 - u 1 * z' 0) = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l953_95304


namespace NUMINAMATH_CALUDE_min_value_of_f_l953_95346

-- Define the function f
def f (x : ℝ) : ℝ := 3*x - 4*x^3

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧
  (∀ y ∈ Set.Icc 0 1, f y ≥ f x) ∧
  f x = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l953_95346


namespace NUMINAMATH_CALUDE_loanYears_correct_l953_95363

/-- Calculates the number of years for which the first part of a loan is lent, given the following conditions:
  * The total sum is 2704
  * The second part of the loan is 1664
  * The interest rate for the first part is 3% per annum
  * The interest rate for the second part is 5% per annum
  * The interest period for the second part is 3 years
  * The interest on the first part equals the interest on the second part
-/
def loanYears : ℕ :=
  let totalSum : ℕ := 2704
  let secondPart : ℕ := 1664
  let firstPartRate : ℚ := 3 / 100
  let secondPartRate : ℚ := 5 / 100
  let secondPartPeriod : ℕ := 3
  let firstPart : ℕ := totalSum - secondPart
  8

theorem loanYears_correct : loanYears = 8 := by sorry

end NUMINAMATH_CALUDE_loanYears_correct_l953_95363


namespace NUMINAMATH_CALUDE_second_planner_cheaper_at_31_l953_95354

/-- Represents the cost function for an event planner -/
structure PlannerCost where
  initial_fee : ℕ
  per_guest : ℕ

/-- Calculates the total cost for a given number of guests -/
def total_cost (p : PlannerCost) (guests : ℕ) : ℕ :=
  p.initial_fee + p.per_guest * guests

/-- First planner's pricing structure -/
def planner1 : PlannerCost := ⟨150, 20⟩

/-- Second planner's pricing structure -/
def planner2 : PlannerCost := ⟨300, 15⟩

/-- Theorem stating that 31 is the minimum number of guests for which the second planner is cheaper -/
theorem second_planner_cheaper_at_31 :
  (∀ g : ℕ, g < 31 → total_cost planner1 g ≤ total_cost planner2 g) ∧
  (∀ g : ℕ, g ≥ 31 → total_cost planner2 g < total_cost planner1 g) :=
sorry

end NUMINAMATH_CALUDE_second_planner_cheaper_at_31_l953_95354


namespace NUMINAMATH_CALUDE_black_block_is_t_shaped_l953_95359

/-- Represents the shape of a block --/
inductive BlockShape
  | L
  | T
  | S
  | I

/-- Represents a block in the rectangular prism --/
structure Block where
  shape : BlockShape
  visible : Bool
  inLowestLayer : Bool

/-- Represents the rectangular prism --/
structure RectangularPrism where
  blocks : Fin 4 → Block
  threeFullyVisible : ∃ (a b c : Fin 4), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (blocks a).visible ∧ (blocks b).visible ∧ (blocks c).visible
  onePartiallyVisible : ∃ (d : Fin 4), ¬(blocks d).visible
  blackBlockInLowestLayer : ∃ (d : Fin 4), ¬(blocks d).visible ∧ (blocks d).inLowestLayer

/-- The main theorem --/
theorem black_block_is_t_shaped (prism : RectangularPrism) : 
  ∃ (d : Fin 4), ¬(prism.blocks d).visible ∧ (prism.blocks d).shape = BlockShape.T := by
  sorry

end NUMINAMATH_CALUDE_black_block_is_t_shaped_l953_95359


namespace NUMINAMATH_CALUDE_exam_score_calculation_l953_95329

theorem exam_score_calculation (total_questions : ℕ) (answered_questions : ℕ) (correct_answers : ℕ) (raw_score : ℚ) :
  total_questions = 85 →
  answered_questions = 82 →
  correct_answers = 70 →
  raw_score = 67 →
  ∃ (points_per_correct : ℚ),
    points_per_correct * correct_answers - (answered_questions - correct_answers) * (1/4 : ℚ) = raw_score ∧
    points_per_correct = 1 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l953_95329


namespace NUMINAMATH_CALUDE_homework_time_calculation_l953_95351

theorem homework_time_calculation (jacob_time greg_time patrick_time : ℕ) : 
  jacob_time = 18 →
  greg_time = jacob_time - 6 →
  patrick_time = 2 * greg_time - 4 →
  jacob_time + greg_time + patrick_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_homework_time_calculation_l953_95351


namespace NUMINAMATH_CALUDE_friendly_iff_ge_seven_l953_95390

def is_friendly (n : ℕ) : Prop :=
  n ≥ 2 ∧
  ∃ (A : Fin n → Set (Fin n)),
    (∀ i, i ∉ A i) ∧
    (∀ i j, i ≠ j → (i ∈ A j ↔ j ∉ A i)) ∧
    (∀ i j, (A i ∩ A j).Nonempty)

theorem friendly_iff_ge_seven :
  ∀ n : ℕ, is_friendly n ↔ n ≥ 7 := by sorry

end NUMINAMATH_CALUDE_friendly_iff_ge_seven_l953_95390


namespace NUMINAMATH_CALUDE_real_roots_quadratic_equation_l953_95394

theorem real_roots_quadratic_equation (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x + 1 = 0) → m ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_equation_l953_95394


namespace NUMINAMATH_CALUDE_basket_probability_l953_95352

def binomial_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem basket_probability : 
  let n : ℕ := 6
  let k : ℕ := 2
  let p : ℝ := 2/3
  binomial_probability n k p = 20/243 := by sorry

end NUMINAMATH_CALUDE_basket_probability_l953_95352


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l953_95350

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

theorem intersection_equals_open_interval : A ∩ B = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l953_95350


namespace NUMINAMATH_CALUDE_bronson_leaf_collection_l953_95343

theorem bronson_leaf_collection (thursday_leaves : ℕ) (yellow_leaves : ℕ) 
  (h1 : thursday_leaves = 12)
  (h2 : yellow_leaves = 15)
  (h3 : yellow_leaves = (3 / 5 : ℚ) * (thursday_leaves + friday_leaves)) :
  friday_leaves = 13 := by
  sorry

end NUMINAMATH_CALUDE_bronson_leaf_collection_l953_95343


namespace NUMINAMATH_CALUDE_exists_term_between_zero_and_one_l953_95322

/-- An infinite sequence satisfying a_{n+2} = |a_{n+1} - a_n| -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1) - a n|

/-- Theorem: For any special sequence, there exists a term between 0 and 1 -/
theorem exists_term_between_zero_and_one (a : ℕ → ℝ) (h : SpecialSequence a) :
    ∃ k : ℕ, 0 ≤ a k ∧ a k < 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_term_between_zero_and_one_l953_95322


namespace NUMINAMATH_CALUDE_money_problem_l953_95369

theorem money_problem (a b : ℝ) : 
  (4 * a - b = 40) ∧ (6 * a + b = 110) → a = 15 ∧ b = 20 := by
sorry

end NUMINAMATH_CALUDE_money_problem_l953_95369


namespace NUMINAMATH_CALUDE_probability_in_given_scenario_l953_95328

/-- Represents the probability of drawing a genuine product after drawing a defective one -/
def probability_genuine_after_defective (total : ℕ) (genuine : ℕ) (defective : ℕ) : ℚ :=
  if total = genuine + defective ∧ defective > 0 then
    genuine / (total - 1)
  else
    0

/-- The main theorem about the probability in the given scenario -/
theorem probability_in_given_scenario :
  probability_genuine_after_defective 7 4 3 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_given_scenario_l953_95328


namespace NUMINAMATH_CALUDE_sixteen_fifth_equals_four_tenth_l953_95353

theorem sixteen_fifth_equals_four_tenth : 16^5 = 4^10 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_fifth_equals_four_tenth_l953_95353


namespace NUMINAMATH_CALUDE_workshop_workers_count_l953_95388

/-- Proves that the total number of workers in a workshop is 24 given specific salary conditions -/
theorem workshop_workers_count : ∀ (W : ℕ) (N : ℕ),
  (W : ℚ) * 8000 = (8 : ℚ) * 12000 + (N : ℚ) * 6000 →  -- total salary equation
  W = 8 + N →                                          -- total workers equation
  W = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l953_95388


namespace NUMINAMATH_CALUDE_fraction_equality_l953_95360

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = -3/7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l953_95360


namespace NUMINAMATH_CALUDE_polynomial_factorization_l953_95365

theorem polynomial_factorization (x : ℝ) : 
  x^6 + 2*x^4 - x^2 - 2 = (x - 1) * (x + 1) * (x^2 + 1) * (x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l953_95365


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l953_95395

theorem quadratic_equation_solution :
  let x₁ : ℝ := (3 + Real.sqrt 15) / 3
  let x₂ : ℝ := (3 - Real.sqrt 15) / 3
  (3 * x₁^2 - 6 * x₁ - 2 = 0) ∧ (3 * x₂^2 - 6 * x₂ - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l953_95395


namespace NUMINAMATH_CALUDE_function_inequality_l953_95399

theorem function_inequality (a : ℝ) : 
  let f (x : ℝ) := (1/3) * x^3 - Real.log (x + 1)
  let g (x : ℝ) := x^2 - 2 * a * x
  (∃ (x₁ : ℝ) (x₂ : ℝ), x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 1 2 ∧ 
    (deriv f x₁) ≥ g x₂) →
  a ≥ (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l953_95399


namespace NUMINAMATH_CALUDE_wilson_cola_purchase_wilson_cola_purchase_correct_l953_95333

theorem wilson_cola_purchase (hamburger_cost : ℕ) (total_cost : ℕ) (discount : ℕ) (cola_cost : ℕ) : ℕ :=
  let hamburgers := 2
  let hamburger_total := hamburgers * hamburger_cost
  let discounted_hamburger_cost := hamburger_total - discount
  let cola_total := total_cost - discounted_hamburger_cost
  cola_total / cola_cost

#check wilson_cola_purchase 5 12 4 2

theorem wilson_cola_purchase_correct : wilson_cola_purchase 5 12 4 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_wilson_cola_purchase_wilson_cola_purchase_correct_l953_95333


namespace NUMINAMATH_CALUDE_committee_selection_l953_95327

theorem committee_selection (n : ℕ) (h : Nat.choose n 3 = 20) : Nat.choose n 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l953_95327


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l953_95305

theorem unique_x_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
  13 ∣ x →
  x = 728 :=
sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l953_95305


namespace NUMINAMATH_CALUDE_exponent_calculation_l953_95377

theorem exponent_calculation : (1 / ((-5^2)^4)) * ((-5)^9) = -5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l953_95377


namespace NUMINAMATH_CALUDE_division_remainder_problem_l953_95358

theorem division_remainder_problem (a b : ℕ) (h1 : a - b = 1365) (h2 : a = 1634) 
  (h3 : ∃ (q : ℕ), q = 6 ∧ a = q * b + (a % b) ∧ a % b < b) : a % b = 20 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l953_95358


namespace NUMINAMATH_CALUDE_jimmy_drinks_eight_times_per_day_l953_95334

/-- The number of times Jimmy drinks water per day -/
def times_per_day : ℕ :=
  let ounces_per_drink : ℚ := 8
  let gallons_for_five_days : ℚ := 5/2
  let ounces_per_gallon : ℚ := 1 / 0.0078125
  let days : ℕ := 5
  let total_ounces : ℚ := gallons_for_five_days * ounces_per_gallon
  let ounces_per_day : ℚ := total_ounces / days
  (ounces_per_day / ounces_per_drink).num.toNat

theorem jimmy_drinks_eight_times_per_day : times_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_drinks_eight_times_per_day_l953_95334


namespace NUMINAMATH_CALUDE_percentage_of_x_l953_95331

theorem percentage_of_x (x y : ℝ) (P : ℝ) : 
  (P / 100) * x = (20 / 100) * y →
  x / y = 2 →
  P = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_x_l953_95331


namespace NUMINAMATH_CALUDE_square_condition_l953_95378

def is_square (x : ℕ) : Prop := ∃ t : ℕ, x = t^2

def floor_div (n m : ℕ) : ℕ := n / m

def expression (n : ℕ) : ℕ :=
  let k := Nat.log2 n
  (List.range (k+1)).foldl (λ acc i => acc * floor_div n (2^i)) 1 + 2 * 4^(k / 2)

theorem square_condition (n : ℕ) : 
  n > 0 → 
  (∃ k : ℕ, 2^k ≤ n ∧ n < 2^(k+1)) → 
  is_square (expression n) → 
  n = 2 ∨ n = 4 := by
sorry

#eval expression 2  -- Expected: 4 (which is 2^2)
#eval expression 4  -- Expected: 16 (which is 4^2)

end NUMINAMATH_CALUDE_square_condition_l953_95378


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l953_95319

/-- Given that the solution set of ax² + bx + c > 0 is {x | 1 < x < 2},
    prove that the solution set of cx² + bx + a < 0 is {x | x < 1/2 or x > 1} -/
theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, ax^2 + b*x + c > 0 ↔ 1 < x ∧ x < 2) :
  ∀ x : ℝ, c*x^2 + b*x + a < 0 ↔ x < 1/2 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l953_95319


namespace NUMINAMATH_CALUDE_lcm_36_75_l953_95308

theorem lcm_36_75 : Nat.lcm 36 75 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_75_l953_95308


namespace NUMINAMATH_CALUDE_probability_at_least_one_head_and_three_l953_95387

def coin_flip : Nat := 2
def die_sides : Nat := 8

def coin_success : Nat := 3  -- number of successful coin flip outcomes (HH, HT, TH)
def die_success : Nat := 1   -- number of successful die roll outcomes (3)

def total_outcomes : Nat := coin_flip^2 * die_sides
def successful_outcomes : Nat := coin_success * die_success

theorem probability_at_least_one_head_and_three :
  (successful_outcomes : ℚ) / total_outcomes = 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_head_and_three_l953_95387


namespace NUMINAMATH_CALUDE_meeting_democrat_ratio_l953_95339

/-- Given a meeting with participants, prove the ratio of male democrats to total male participants -/
theorem meeting_democrat_ratio 
  (total_participants : ℕ) 
  (female_democrats : ℕ) 
  (h_total : total_participants = 780)
  (h_female_dem : female_democrats = 130)
  (h_half_female : female_democrats * 2 ≤ total_participants)
  (h_third_dem : 3 * female_democrats * 2 = total_participants)
  : (total_participants - female_democrats * 2 - female_democrats) / 
    (total_participants - female_democrats * 2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_meeting_democrat_ratio_l953_95339


namespace NUMINAMATH_CALUDE_rectangular_hall_area_l953_95357

/-- Calculates the area of a rectangular hall given its length and breadth ratio. -/
def hall_area (length : ℝ) (breadth_ratio : ℝ) : ℝ :=
  length * (breadth_ratio * length)

/-- Theorem: The area of a rectangular hall with length 60 meters and breadth
    two-thirds of its length is 2400 square meters. -/
theorem rectangular_hall_area :
  hall_area 60 (2/3) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_hall_area_l953_95357


namespace NUMINAMATH_CALUDE_sandwich_combinations_l953_95337

theorem sandwich_combinations (meat_types : ℕ) (cheese_types : ℕ) (condiment_types : ℕ) :
  meat_types = 12 →
  cheese_types = 11 →
  condiment_types = 5 →
  (meat_types * Nat.choose cheese_types 2 * (condiment_types + 1)) = 3960 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l953_95337


namespace NUMINAMATH_CALUDE_factorization_proof_l953_95340

theorem factorization_proof (a b : ℝ) : -a^3 + 12*a^2*b - 36*a*b^2 = -a*(a-6*b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l953_95340


namespace NUMINAMATH_CALUDE_max_area_square_with_perimeter_32_l953_95383

/-- The maximum area of a square with a perimeter of 32 meters is 64 square meters. -/
theorem max_area_square_with_perimeter_32 :
  let perimeter : ℝ := 32
  let side_length : ℝ := perimeter / 4
  let area : ℝ := side_length ^ 2
  area = 64 := by sorry

end NUMINAMATH_CALUDE_max_area_square_with_perimeter_32_l953_95383


namespace NUMINAMATH_CALUDE_square_not_always_positive_l953_95313

theorem square_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l953_95313


namespace NUMINAMATH_CALUDE_hotel_bill_problem_l953_95323

theorem hotel_bill_problem (total_bill : ℕ) (equal_share : ℕ) (extra_payment : ℕ) (num_paying_80 : ℕ) :
  (num_paying_80 = 7) →
  (80 * num_paying_80 + 160 = total_bill) →
  (equal_share + 70 = 160) →
  (total_bill / equal_share = 8) :=
by
  sorry

#check hotel_bill_problem

end NUMINAMATH_CALUDE_hotel_bill_problem_l953_95323


namespace NUMINAMATH_CALUDE_board_ratio_l953_95355

theorem board_ratio (total_length shorter_length : ℝ) 
  (h1 : total_length = 6)
  (h2 : shorter_length = 2)
  (h3 : shorter_length < total_length) :
  (total_length - shorter_length) / shorter_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_board_ratio_l953_95355


namespace NUMINAMATH_CALUDE_seventh_term_ratio_l953_95396

/-- Two arithmetic sequences with sums of first n terms R_n and U_n -/
def R_n (n : ℕ) : ℚ := sorry
def U_n (n : ℕ) : ℚ := sorry

/-- The ratio condition for all n -/
axiom ratio_condition (n : ℕ) : R_n n / U_n n = (3 * n + 5 : ℚ) / (2 * n + 13 : ℚ)

/-- The 7th term of each sequence -/
def r_7 : ℚ := sorry
def s_7 : ℚ := sorry

/-- The main theorem -/
theorem seventh_term_ratio : r_7 / s_7 = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_seventh_term_ratio_l953_95396


namespace NUMINAMATH_CALUDE_birthday_party_ratio_l953_95362

theorem birthday_party_ratio (total_guests : ℕ) (men : ℕ) (stayed : ℕ) : 
  total_guests = 60 →
  men = 15 →
  stayed = 50 →
  (total_guests / 2 : ℕ) + men + (total_guests - (total_guests / 2 + men)) = total_guests →
  (total_guests - stayed - 5 : ℕ) / men = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_birthday_party_ratio_l953_95362


namespace NUMINAMATH_CALUDE_fraction_sum_integer_l953_95344

theorem fraction_sum_integer (n : ℕ) (hn : n > 0) 
  (h_sum : ∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) : 
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_l953_95344


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l953_95316

/-- Simple interest calculation -/
theorem simple_interest_rate_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 1000) 
  (h2 : interest = 400) 
  (h3 : time = 4) : 
  (interest * 100) / (principal * time) = 10 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_percent_l953_95316


namespace NUMINAMATH_CALUDE_theater_attendance_l953_95385

theorem theater_attendance
  (adult_ticket_price : ℕ)
  (child_ticket_price : ℕ)
  (total_revenue : ℕ)
  (num_children : ℕ)
  (h1 : adult_ticket_price = 8)
  (h2 : child_ticket_price = 1)
  (h3 : total_revenue = 50)
  (h4 : num_children = 18) :
  adult_ticket_price * (total_revenue - child_ticket_price * num_children) / adult_ticket_price + num_children = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_attendance_l953_95385


namespace NUMINAMATH_CALUDE_area_of_special_points_triangle_l953_95341

/-- A triangle with side lengths 18, 24, and 30 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_sides : a = 18 ∧ b = 24 ∧ c = 30

/-- The incenter of a triangle -/
def incenter (t : RightTriangle) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle -/
def circumcenter (t : RightTriangle) : ℝ × ℝ := sorry

/-- The centroid of a triangle -/
def centroid (t : RightTriangle) : ℝ × ℝ := sorry

/-- The area of a triangle given its vertices -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of the triangle formed by the incenter, circumcenter, and centroid of a 18-24-30 right triangle is 6 -/
theorem area_of_special_points_triangle (t : RightTriangle) : 
  triangleArea (incenter t) (circumcenter t) (centroid t) = 6 := by sorry

end NUMINAMATH_CALUDE_area_of_special_points_triangle_l953_95341


namespace NUMINAMATH_CALUDE_smallest_congruent_difference_l953_95338

/-- The smallest positive four-digit integer congruent to 7 (mod 13) -/
def p : ℕ := sorry

/-- The smallest positive five-digit integer congruent to 7 (mod 13) -/
def q : ℕ := sorry

theorem smallest_congruent_difference : q - p = 8996 := by sorry

end NUMINAMATH_CALUDE_smallest_congruent_difference_l953_95338


namespace NUMINAMATH_CALUDE_minimum_amount_is_1000_l953_95376

/-- The minimum amount of the sell to get a discount -/
def minimum_amount_for_discount (
  item_count : ℕ) 
  (item_cost : ℚ) 
  (discounted_total : ℚ) 
  (discount_rate : ℚ) : ℚ :=
  item_count * item_cost - (item_count * item_cost - discounted_total) / discount_rate

/-- Theorem stating the minimum amount for discount is $1000 -/
theorem minimum_amount_is_1000 : 
  minimum_amount_for_discount 7 200 1360 (1/10) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_minimum_amount_is_1000_l953_95376


namespace NUMINAMATH_CALUDE_random_selection_result_l953_95389

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Represents a position in the random number table --/
structure TablePosition where
  row : Nat
  column : Nat

/-- Function to select numbers from the random number table --/
def selectNumbers (table : RandomNumberTable) (start : TablePosition) (count : Nat) (maxNumber : Nat) : List Nat :=
  sorry

/-- The theorem to prove --/
theorem random_selection_result (table : RandomNumberTable) (studentCount : Nat) (selectionCount : Nat) (startPosition : TablePosition) :
  studentCount = 247 →
  selectionCount = 4 →
  startPosition = ⟨4, 9⟩ →
  selectNumbers table startPosition selectionCount studentCount = [050, 121, 014, 218] :=
sorry

end NUMINAMATH_CALUDE_random_selection_result_l953_95389


namespace NUMINAMATH_CALUDE_next_number_with_property_l953_95384

/-- A function that splits a four-digit number into its hundreds and tens-ones parts -/
def split_number (n : ℕ) : ℕ × ℕ :=
  (n / 100, n % 100)

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The property we're looking for in the number -/
def has_property (n : ℕ) : Prop :=
  let (a, b) := split_number n
  is_perfect_square (a * b)

theorem next_number_with_property :
  ∀ n : ℕ, 1818 < n → n < 1832 → ¬(has_property n) ∧ has_property 1832 := by
  sorry

#check next_number_with_property

end NUMINAMATH_CALUDE_next_number_with_property_l953_95384


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l953_95335

theorem divisibility_equivalence (m n : ℕ+) :
  (6 * m.val ∣ (2 * m.val + 3)^n.val + 1) ↔ (4 * m.val ∣ 3^n.val + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l953_95335


namespace NUMINAMATH_CALUDE_function_max_value_l953_95380

open Real

theorem function_max_value (a : ℝ) (h1 : a > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.log x - a * x
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≤ -4) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f x = -4) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_max_value_l953_95380


namespace NUMINAMATH_CALUDE_union_of_sets_l953_95315

theorem union_of_sets : 
  let A : Set Int := {-2, 0}
  let B : Set Int := {-2, 3}
  A ∪ B = {-2, 0, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l953_95315


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l953_95342

/-- Two lines in the form ax + by + c = 0 are parallel if and only if their slopes are equal -/
def are_parallel (a1 b1 a2 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

/-- The first line equation: x + ay + 3 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 3 = 0

/-- The second line equation: (a-2)x + 3y + a = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y + a = 0

theorem parallel_lines_condition (a : ℝ) : 
  are_parallel 1 a (a - 2) 3 ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l953_95342


namespace NUMINAMATH_CALUDE_bricklayer_solution_l953_95321

/-- Represents the problem of two bricklayers building a wall -/
structure BricklayerProblem where
  -- Total number of bricks in the wall
  total_bricks : ℕ
  -- Time taken by the first bricklayer alone (in hours)
  time_first : ℕ
  -- Time taken by the second bricklayer alone (in hours)
  time_second : ℕ
  -- Reduction in combined output (in bricks per hour)
  output_reduction : ℕ
  -- Time taken when working together (in hours)
  time_together : ℕ

/-- The theorem stating the solution to the bricklayer problem -/
theorem bricklayer_solution (problem : BricklayerProblem) :
  problem.time_first = 8 →
  problem.time_second = 12 →
  problem.output_reduction = 15 →
  problem.time_together = 6 →
  problem.total_bricks = 360 := by
  sorry

#check bricklayer_solution

end NUMINAMATH_CALUDE_bricklayer_solution_l953_95321


namespace NUMINAMATH_CALUDE_books_added_to_bin_l953_95302

/-- Proves the number of books added to a bargain bin -/
theorem books_added_to_bin (initial books_sold final : ℕ) 
  (h1 : initial = 4)
  (h2 : books_sold = 3)
  (h3 : final = 11) :
  final - (initial - books_sold) = 10 := by
  sorry

end NUMINAMATH_CALUDE_books_added_to_bin_l953_95302


namespace NUMINAMATH_CALUDE_sin_negative_120_degrees_l953_95324

theorem sin_negative_120_degrees : Real.sin (-(2 * π / 3)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_120_degrees_l953_95324


namespace NUMINAMATH_CALUDE_max_subsets_l953_95366

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the property for subsets A₁, A₂, ..., Aₖ
def valid_subsets (A : Finset (Finset ℕ)) : Prop :=
  ∀ X ∈ A, X ⊆ S ∧ X.card = 5 ∧ ∀ Y ∈ A, X ≠ Y → (X ∩ Y).card ≤ 2

-- Theorem statement
theorem max_subsets :
  ∀ A : Finset (Finset ℕ), valid_subsets A → A.card ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_subsets_l953_95366


namespace NUMINAMATH_CALUDE_negation_existence_real_l953_95374

theorem negation_existence_real : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_existence_real_l953_95374


namespace NUMINAMATH_CALUDE_proper_subsets_without_two_eq_l953_95309

def S : Set ℕ := {1, 2, 3, 4}

def proper_subsets_without_two : Set (Set ℕ) :=
  {A | A ⊂ S ∧ 2 ∉ A}

theorem proper_subsets_without_two_eq :
  proper_subsets_without_two = {∅, {1}, {3}, {4}, {1, 3}, {1, 4}, {3, 4}, {1, 3, 4}} := by
  sorry

end NUMINAMATH_CALUDE_proper_subsets_without_two_eq_l953_95309
