import Mathlib

namespace NUMINAMATH_CALUDE_field_ratio_l3980_398095

theorem field_ratio (field_length field_width pond_side : ℝ) : 
  field_length = 16 →
  field_length = field_width * (field_length / field_width) →
  pond_side = 4 →
  pond_side^2 = (1/8) * (field_length * field_width) →
  field_length / field_width = 2 := by
sorry

end NUMINAMATH_CALUDE_field_ratio_l3980_398095


namespace NUMINAMATH_CALUDE_pony_price_is_20_l3980_398061

/-- The regular price of fox jeans in dollars -/
def fox_price : ℝ := 15

/-- The regular price of pony jeans in dollars -/
def pony_price : ℝ := 20

/-- The number of fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings in dollars -/
def total_savings : ℝ := 9

/-- The sum of the two discount rates as a percentage -/
def total_discount_rate : ℝ := 22

/-- The discount rate on pony jeans as a percentage -/
def pony_discount_rate : ℝ := 18

/-- Theorem stating that the regular price of pony jeans is $20 given the conditions -/
theorem pony_price_is_20 : 
  fox_price * fox_quantity * (total_discount_rate - pony_discount_rate) / 100 +
  pony_price * pony_quantity * pony_discount_rate / 100 = total_savings :=
by sorry

end NUMINAMATH_CALUDE_pony_price_is_20_l3980_398061


namespace NUMINAMATH_CALUDE_basketball_chess_fans_l3980_398051

/-- The number of students who like basketball or chess given the following conditions:
  * 40% of students like basketball
  * 10% of students like chess
  * 250 students were interviewed
-/
theorem basketball_chess_fans (total_students : ℕ) (basketball_percent : ℚ) (chess_percent : ℚ) :
  total_students = 250 →
  basketball_percent = 40 / 100 →
  chess_percent = 10 / 100 →
  (basketball_percent + chess_percent) * total_students = 125 := by
sorry

end NUMINAMATH_CALUDE_basketball_chess_fans_l3980_398051


namespace NUMINAMATH_CALUDE_square_area_error_l3980_398079

theorem square_area_error (actual_side : ℝ) (h : actual_side > 0) :
  let measured_side := actual_side * 1.1
  let actual_area := actual_side ^ 2
  let calculated_area := measured_side ^ 2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.21 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l3980_398079


namespace NUMINAMATH_CALUDE_intersection_A_notB_C_subset_A_implies_a_range_l3980_398004

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- Define the complement of B in ℝ
def notB : Set ℝ := {x | ¬(x ∈ B)}

-- Theorem for part (I)
theorem intersection_A_notB : A ∩ notB = {x : ℝ | -3 < x ∧ x ≤ 2} := by sorry

-- Theorem for part (II)
theorem C_subset_A_implies_a_range (a : ℝ) (h : a ≠ 0) :
  C a ⊆ A → (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a ≤ 4/3) := by sorry

end NUMINAMATH_CALUDE_intersection_A_notB_C_subset_A_implies_a_range_l3980_398004


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3980_398073

theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3980_398073


namespace NUMINAMATH_CALUDE_circle_graph_fractions_l3980_398083

/-- Represents the fractions of a circle graph split into three colors -/
structure CircleGraph :=
  (black : ℚ)
  (gray : ℚ)
  (white : ℚ)

/-- The conditions of the circle graph -/
def valid_circle_graph (g : CircleGraph) : Prop :=
  g.black = 2 * g.gray ∧
  g.white = g.gray / 2 ∧
  g.black + g.gray + g.white = 1

/-- The theorem to prove -/
theorem circle_graph_fractions :
  ∃ (g : CircleGraph), valid_circle_graph g ∧
    g.black = 4/7 ∧ g.gray = 2/7 ∧ g.white = 1/7 :=
sorry

end NUMINAMATH_CALUDE_circle_graph_fractions_l3980_398083


namespace NUMINAMATH_CALUDE_matrix_not_invertible_sum_fractions_l3980_398021

theorem matrix_not_invertible_sum_fractions (a b c : ℝ) :
  let M := !![a, b, c; b, c, a; c, a, b]
  ¬(IsUnit (Matrix.det M)) →
  (a / (b + c) + b / (a + c) + c / (a + b) = -3) ∨
  (a / (b + c) + b / (a + c) + c / (a + b) = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_sum_fractions_l3980_398021


namespace NUMINAMATH_CALUDE_smaller_number_problem_l3980_398015

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 12) (h3 : x > y) :
  y = 14 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3980_398015


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l3980_398093

/-- Given five collinear points A, B, C, D, E in that order, with specific distances between them,
    prove that the minimum sum of squared distances from these points to any point P on AD is 237. -/
theorem min_sum_squared_distances (A B C D E P : ℝ) : 
  (A < B) → (B < C) → (C < D) → (D < E) →  -- Points are collinear and in order
  (B - A = 1) → (C - B = 1) → (D - C = 3) → (E - D = 12) →  -- Given distances
  (A ≤ P) → (P ≤ D) →  -- P is on segment AD
  ∃ (m : ℝ), ∀ (Q : ℝ), (A ≤ Q) → (Q ≤ D) → 
    (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2 ≥ m ∧ 
    m = 237 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l3980_398093


namespace NUMINAMATH_CALUDE_set_membership_l3980_398007

def M : Set ℤ := {a | ∃ b c : ℤ, a = b^2 - c^2}

theorem set_membership : (8 ∈ M) ∧ (9 ∈ M) ∧ (10 ∉ M) := by sorry

end NUMINAMATH_CALUDE_set_membership_l3980_398007


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3980_398055

/-- Given that m + 2n - 1 = 0, prove that the line mx + 3y + n = 0 passes through the point (1/2, -1/6) -/
theorem line_passes_through_point (m n : ℝ) (h : m + 2 * n - 1 = 0) :
  m * (1/2 : ℝ) + 3 * (-1/6 : ℝ) + n = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3980_398055


namespace NUMINAMATH_CALUDE_exists_x0_implies_a_value_l3980_398070

noncomputable section

def f (a x : ℝ) : ℝ := x + Real.exp (x - a)

def g (a x : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem exists_x0_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 3) → a = -1 - Real.log 2 := by
  sorry

end

end NUMINAMATH_CALUDE_exists_x0_implies_a_value_l3980_398070


namespace NUMINAMATH_CALUDE_factorization_proof_l3980_398001

theorem factorization_proof (m n : ℝ) : m^2 - m*n = m*(m - n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3980_398001


namespace NUMINAMATH_CALUDE_stacy_extra_berries_l3980_398037

/-- The number of berries each person has -/
structure BerryCount where
  stacy : ℕ
  steve : ℕ
  skylar : ℕ

/-- The given conditions for the berry problem -/
def berry_conditions (b : BerryCount) : Prop :=
  b.stacy > 3 * b.steve ∧
  2 * b.steve = b.skylar ∧
  b.skylar = 20 ∧
  b.stacy = 32

/-- The theorem to prove -/
theorem stacy_extra_berries (b : BerryCount) (h : berry_conditions b) :
  b.stacy - 3 * b.steve = 2 := by
  sorry

end NUMINAMATH_CALUDE_stacy_extra_berries_l3980_398037


namespace NUMINAMATH_CALUDE_adjacent_combinations_l3980_398098

def number_of_people : ℕ := 9
def number_of_friends : ℕ := 8
def adjacent_positions : ℕ := 2

theorem adjacent_combinations :
  Nat.choose number_of_friends adjacent_positions = 28 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_combinations_l3980_398098


namespace NUMINAMATH_CALUDE_coeff_x_squared_is_thirteen_l3980_398086

/-- The coefficient of x^2 in the expansion of (1-x)^3(2x^2+1)^5 -/
def coeff_x_squared : ℕ :=
  (Nat.choose 5 4) * 2 + 3 * (Nat.choose 5 5)

/-- Theorem stating that the coefficient of x^2 in the expansion of (1-x)^3(2x^2+1)^5 is 13 -/
theorem coeff_x_squared_is_thirteen : coeff_x_squared = 13 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x_squared_is_thirteen_l3980_398086


namespace NUMINAMATH_CALUDE_guitar_center_discount_l3980_398028

/-- The discount offered by Guitar Center for a guitar with a suggested retail price of $1000,
    given that Guitar Center has a $100 shipping fee, Sweetwater has a 10% discount with free shipping,
    and the difference in final price between the two stores is $50. -/
theorem guitar_center_discount (suggested_price : ℕ) (gc_shipping : ℕ) (sw_discount_percent : ℕ) (price_difference : ℕ) :
  suggested_price = 1000 →
  gc_shipping = 100 →
  sw_discount_percent = 10 →
  price_difference = 50 →
  ∃ (gc_discount : ℕ), gc_discount = 150 :=
by sorry

end NUMINAMATH_CALUDE_guitar_center_discount_l3980_398028


namespace NUMINAMATH_CALUDE_pizza_toppings_l3980_398047

/-- Given a pizza with the following properties:
  * It has 16 slices in total
  * Every slice has at least one topping
  * There are three toppings: cheese, chicken, and olives
  * 8 slices have cheese
  * 12 slices have chicken
  * 6 slices have olives
  This theorem proves that exactly 5 slices have all three toppings. -/
theorem pizza_toppings (total_slices : ℕ) (cheese_slices : ℕ) (chicken_slices : ℕ) (olive_slices : ℕ)
    (h_total : total_slices = 16)
    (h_cheese : cheese_slices = 8)
    (h_chicken : chicken_slices = 12)
    (h_olives : olive_slices = 6)
    (h_at_least_one : ∀ slice, slice ∈ Finset.range total_slices →
      (slice ∈ Finset.range cheese_slices ∨
       slice ∈ Finset.range chicken_slices ∨
       slice ∈ Finset.range olive_slices)) :
    ∃ all_toppings : ℕ, all_toppings = 5 ∧
      (∀ slice, slice ∈ Finset.range total_slices →
        (slice ∈ Finset.range cheese_slices ∧
         slice ∈ Finset.range chicken_slices ∧
         slice ∈ Finset.range olive_slices) ↔
        slice ∈ Finset.range all_toppings) := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l3980_398047


namespace NUMINAMATH_CALUDE_sara_survey_sara_survey_result_l3980_398090

theorem sara_survey (total : ℕ) 
  (belief_rate : ℚ) 
  (zika_rate : ℚ) 
  (zika_believers : ℕ) : Prop :=
  belief_rate = 753/1000 →
  zika_rate = 602/1000 →
  zika_believers = 37 →
  ∃ (believers : ℕ),
    (believers : ℚ) = zika_believers / zika_rate ∧
    (total : ℚ) = (believers : ℚ) / belief_rate ∧
    total = 81

theorem sara_survey_result : 
  ∃ (total : ℕ), sara_survey total (753/1000) (602/1000) 37 :=
sorry

end NUMINAMATH_CALUDE_sara_survey_sara_survey_result_l3980_398090


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l3980_398057

theorem shaded_area_fraction (length width : ℕ) (quarter_shaded_fraction : ℚ) (unshaded_squares : ℕ) :
  length = 15 →
  width = 20 →
  quarter_shaded_fraction = 1/4 →
  unshaded_squares = 9 →
  (quarter_shaded_fraction * (1/4 * (length * width)) - unshaded_squares) / (length * width) = 13/400 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l3980_398057


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l3980_398065

theorem solution_satisfies_equation :
  let x : ℝ := 1
  let y : ℝ := -1
  x - 2 * y = 3 := by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l3980_398065


namespace NUMINAMATH_CALUDE_product_difference_squares_l3980_398072

theorem product_difference_squares : (3 + Real.sqrt 7) * (3 - Real.sqrt 7) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_squares_l3980_398072


namespace NUMINAMATH_CALUDE_rectangle_area_l3980_398026

/-- A rectangle with specific properties -/
structure Rectangle where
  width : ℝ
  length : ℝ
  length_exceed_twice_width : length = 2 * width + 25
  perimeter_650 : 2 * (length + width) = 650

/-- The area of a rectangle with the given properties is 22500 -/
theorem rectangle_area (r : Rectangle) : r.length * r.width = 22500 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3980_398026


namespace NUMINAMATH_CALUDE_product_of_primes_l3980_398099

theorem product_of_primes : 3^2 * 5 * 7^2 * 11 = 24255 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_l3980_398099


namespace NUMINAMATH_CALUDE_simplify_fraction_l3980_398084

theorem simplify_fraction (a : ℝ) (h : a ≠ 3) :
  1 / (a - 3) - 6 / (a^2 - 9) = 1 / (a + 3) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3980_398084


namespace NUMINAMATH_CALUDE_cereal_box_ratio_l3980_398092

/-- Theorem: Cereal Box Ratio
Given 3 boxes of cereal where:
- The first box contains 14 ounces
- The total amount in all boxes is 33 ounces
- The second box contains 5 ounces less than the third box
Then the ratio of cereal in the second box to the first box is 1:2
-/
theorem cereal_box_ratio (box1 box2 box3 : ℝ) : 
  box1 = 14 →
  box1 + box2 + box3 = 33 →
  box2 = box3 - 5 →
  box2 / box1 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cereal_box_ratio_l3980_398092


namespace NUMINAMATH_CALUDE_probability_product_multiple_of_four_l3980_398082

def range_start : ℕ := 5
def range_end : ℕ := 25

def is_in_range (n : ℕ) : Prop := range_start ≤ n ∧ n ≤ range_end

def count_in_range : ℕ := range_end - range_start + 1

def count_multiples_of_four : ℕ := (range_end / 4) - ((range_start - 1) / 4)

def total_combinations : ℕ := count_in_range * (count_in_range - 1) / 2

def favorable_combinations : ℕ := count_multiples_of_four * (count_multiples_of_four - 1) / 2

theorem probability_product_multiple_of_four :
  (favorable_combinations : ℚ) / total_combinations = 1 / 21 := by sorry

end NUMINAMATH_CALUDE_probability_product_multiple_of_four_l3980_398082


namespace NUMINAMATH_CALUDE_min_value_of_f_l3980_398034

open Real

-- Define the function f
def f (a b c d e x : ℝ) : ℝ := |x - a| + |x - b| + |x - c| + |x - d| + |x - e|

-- State the theorem
theorem min_value_of_f (a b c d e : ℝ) (h : a < b ∧ b < c ∧ c < d ∧ d < e) :
  ∃ (m : ℝ), (∀ x, f a b c d e x ≥ m) ∧ m = e + d - b - a := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3980_398034


namespace NUMINAMATH_CALUDE_rows_remain_ascending_l3980_398031

/-- Represents a rectangular table of numbers -/
def Table (m n : ℕ) := Fin m → Fin n → ℝ

/-- Checks if a row is in ascending order -/
def isRowAscending (t : Table m n) (i : Fin m) : Prop :=
  ∀ j k : Fin n, j < k → t i j ≤ t i k

/-- Checks if a column is in ascending order -/
def isColumnAscending (t : Table m n) (j : Fin n) : Prop :=
  ∀ i k : Fin m, i < k → t i j ≤ t k j

/-- Sorts a row in ascending order -/
def sortRow (t : Table m n) (i : Fin m) : Table m n :=
  sorry

/-- Sorts a column in ascending order -/
def sortColumn (t : Table m n) (j : Fin n) : Table m n :=
  sorry

/-- Sorts all rows in ascending order -/
def sortAllRows (t : Table m n) : Table m n :=
  sorry

/-- Sorts all columns in ascending order -/
def sortAllColumns (t : Table m n) : Table m n :=
  sorry

/-- Main theorem: After sorting rows and then columns, rows remain in ascending order -/
theorem rows_remain_ascending (m n : ℕ) (t : Table m n) :
  ∀ i : Fin m, isRowAscending (sortAllColumns (sortAllRows t)) i :=
sorry

end NUMINAMATH_CALUDE_rows_remain_ascending_l3980_398031


namespace NUMINAMATH_CALUDE_eight_power_division_l3980_398038

theorem eight_power_division (x : ℕ) (y : ℕ) (z : ℕ) :
  x^15 / (x^2)^3 = x^9 :=
by sorry

end NUMINAMATH_CALUDE_eight_power_division_l3980_398038


namespace NUMINAMATH_CALUDE_find_n_l3980_398068

theorem find_n : ∃ n : ℚ, 1/2 + 2/3 + 3/4 + n/12 = 2 → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3980_398068


namespace NUMINAMATH_CALUDE_razorback_tshirt_profit_l3980_398076

/-- The Razorback T-shirt Shop problem -/
theorem razorback_tshirt_profit :
  let profit_per_shirt : ℕ := 9
  let shirts_sold : ℕ := 245
  let total_profit : ℕ := profit_per_shirt * shirts_sold
  total_profit = 2205 := by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_profit_l3980_398076


namespace NUMINAMATH_CALUDE_trisection_points_on_circle_implies_equilateral_l3980_398060

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The trisection points of a triangle's sides. -/
def trisectionPoints (t : Triangle) : List (ℝ × ℝ) :=
  sorry

/-- Predicate to check if a list of points lie on a circle. -/
def lieOnCircle (points : List (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a triangle is equilateral. -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- Theorem: If the trisection points of a triangle's sides lie on a circle,
    then the triangle is equilateral. -/
theorem trisection_points_on_circle_implies_equilateral (t : Triangle) :
  lieOnCircle (trisectionPoints t) → isEquilateral t :=
sorry

end NUMINAMATH_CALUDE_trisection_points_on_circle_implies_equilateral_l3980_398060


namespace NUMINAMATH_CALUDE_four_digit_perfect_square_l3980_398014

theorem four_digit_perfect_square : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧  -- 4-digit number
  (∃ m : ℕ, n = m^2) ∧      -- perfect square
  (n / 100 = n % 100 + 1)   -- first two digits are one more than last two digits
  := by
  use 8281
  sorry

end NUMINAMATH_CALUDE_four_digit_perfect_square_l3980_398014


namespace NUMINAMATH_CALUDE_four_steps_on_number_line_l3980_398041

/-- Given a number line with equally spaced markings where the distance from 0 to 25
    is covered in 7 steps, prove that the number reached after 4 steps from 0 is 100/7. -/
theorem four_steps_on_number_line :
  ∀ (step_length : ℚ),
  step_length * 7 = 25 →
  4 * step_length = 100 / 7 := by
sorry

end NUMINAMATH_CALUDE_four_steps_on_number_line_l3980_398041


namespace NUMINAMATH_CALUDE_train_platform_length_l3980_398006

/-- The length of a train platform problem -/
theorem train_platform_length 
  (train_length : ℝ) 
  (platform1_length : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (h1 : train_length = 270)
  (h2 : platform1_length = 120)
  (h3 : time1 = 15)
  (h4 : time2 = 20) :
  ∃ (platform2_length : ℝ),
    platform2_length = 250 ∧ 
    (train_length + platform1_length) / time1 = 
    (train_length + platform2_length) / time2 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_length_l3980_398006


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l3980_398020

theorem two_digit_number_problem (M N : ℕ) (h1 : M < 10) (h2 : N < 10) (h3 : N > M) :
  let x := 10 * N + M
  let y := 10 * M + N
  (x + y = 11 * (x - y)) → (M = 4 ∧ N = 5) := by
sorry


end NUMINAMATH_CALUDE_two_digit_number_problem_l3980_398020


namespace NUMINAMATH_CALUDE_seventeen_stations_tickets_l3980_398058

/-- The number of unique, non-directional tickets needed for travel between any two stations -/
def num_tickets (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

/-- Theorem: For 17 stations, the number of unique, non-directional tickets is 68 -/
theorem seventeen_stations_tickets :
  num_tickets 17 = 68 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_stations_tickets_l3980_398058


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3980_398010

theorem quadratic_inequality_solution (x : ℝ) : 
  (3 * x^2 - 2 * x - 8 ≤ 0) ↔ (-4/3 ≤ x ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3980_398010


namespace NUMINAMATH_CALUDE_womens_tennis_handshakes_l3980_398022

theorem womens_tennis_handshakes (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 2) : 
  (n * k * (n * k - k)) / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_womens_tennis_handshakes_l3980_398022


namespace NUMINAMATH_CALUDE_newberg_airport_passengers_l3980_398044

theorem newberg_airport_passengers (on_time late : ℕ) 
  (h1 : on_time = 14507) 
  (h2 : late = 213) : 
  on_time + late = 14720 := by sorry

end NUMINAMATH_CALUDE_newberg_airport_passengers_l3980_398044


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l3980_398045

def hyperbola_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + y^2 = 1

def imaginary_axis_twice_real_axis (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a = b ∧
  ∀ x y : ℝ, hyperbola_equation m x y ↔ (x/a)^2 - (y/b)^2 = 1

theorem hyperbola_m_value :
  ∀ m : ℝ, imaginary_axis_twice_real_axis m → m = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l3980_398045


namespace NUMINAMATH_CALUDE_expression_value_l3980_398002

theorem expression_value : 
  (0.02^2 + 0.52^2 + 0.035^2) / (0.002^2 + 0.052^2 + 0.0035^2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3980_398002


namespace NUMINAMATH_CALUDE_log_sum_equality_l3980_398071

theorem log_sum_equality : 10^(Real.log 3 / Real.log 10) + Real.log 25 / Real.log 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l3980_398071


namespace NUMINAMATH_CALUDE_perfect_square_property_l3980_398054

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + a n

theorem perfect_square_property (n : ℕ) (h : n > 0) : 
  ∃ m : ℤ, 2 * ((a (2 * n))^2 - 1) = m^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_property_l3980_398054


namespace NUMINAMATH_CALUDE_min_value_theorem_l3980_398019

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let line := {(x, y) : ℝ × ℝ | 2 * a * x - b * y + 2 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + 2*x - 4*y + 1 = 0}
  let chord_length := 4
  (∃ (p q : ℝ × ℝ), p ∈ line ∧ q ∈ line ∧ p ∈ circle ∧ q ∈ circle ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) →
  (∀ c d : ℝ, c > 0 → d > 0 → 2 * c - d + 2 = 0 → 1/c + 1/d ≥ 1/a + 1/b) ∧
  1/a + 1/b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3980_398019


namespace NUMINAMATH_CALUDE_equation_solution_l3980_398025

theorem equation_solution : ∃ x : ℝ, 0.4 * x + (0.6 * 0.8) = 0.56 ∧ x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3980_398025


namespace NUMINAMATH_CALUDE_intersection_range_l3980_398066

-- Define the points M and N
def M : ℝ × ℝ := (1, 0)
def N : ℝ × ℝ := (-1, 0)

-- Define the line equation
def line (x y b : ℝ) : Prop := 2 * x + y = b

-- Define the line segment MN
def on_segment (x y : ℝ) : Prop :=
  x ≥ -1 ∧ x ≤ 1 ∧ y = 0

-- Theorem statement
theorem intersection_range :
  ∀ b : ℝ,
  (∃ x y : ℝ, line x y b ∧ on_segment x y) ↔
  b ≥ -2 ∧ b ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l3980_398066


namespace NUMINAMATH_CALUDE_sin_240_l3980_398040

-- Define the cofunction identity
axiom cofunction_identity (α : Real) : Real.sin (180 + α) = -Real.sin α

-- Define the special angle value
axiom sin_60 : Real.sin 60 = Real.sqrt 3 / 2

-- State the theorem to be proved
theorem sin_240 : Real.sin 240 = -(Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_240_l3980_398040


namespace NUMINAMATH_CALUDE_product_of_linear_terms_l3980_398032

theorem product_of_linear_terms (x y : ℝ) : (-2 * x) * (3 * y) = -6 * x * y := by
  sorry

end NUMINAMATH_CALUDE_product_of_linear_terms_l3980_398032


namespace NUMINAMATH_CALUDE_exterior_angle_triangle_l3980_398088

theorem exterior_angle_triangle (α β γ : ℝ) : 
  0 < α ∧ 0 < β ∧ 0 < γ →  -- angles are positive
  α + β + γ = 180 →  -- sum of angles in a triangle is 180°
  α + β = 148 →  -- exterior angle
  β = 58 →  -- one interior angle
  γ = 90  -- prove that the other interior angle is 90°
  := by sorry

end NUMINAMATH_CALUDE_exterior_angle_triangle_l3980_398088


namespace NUMINAMATH_CALUDE_cubic_sum_zero_l3980_398049

theorem cubic_sum_zero (a b c : ℝ) : 
  a + b + c = 0 → a^3 + a^2*c - a*b*c + b^2*c + b^3 = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_zero_l3980_398049


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_differences_l3980_398081

theorem greatest_common_divisor_of_differences (a b c : ℕ) (h : a < b ∧ b < c) :
  ∃ d : ℕ, d > 0 ∧ 
    (∃ (r : ℕ), a % d = r ∧ b % d = r ∧ c % d = r) ∧
    (∀ k : ℕ, k > d → ¬(∃ (s : ℕ), a % k = s ∧ b % k = s ∧ c % k = s)) →
  (Nat.gcd (b - a) (c - b) = 10) →
  (a = 20 ∧ b = 40 ∧ c = 90) →
  (∃ d : ℕ, d = 10 ∧ d > 0 ∧ 
    (∃ (r : ℕ), a % d = r ∧ b % d = r ∧ c % d = r) ∧
    (∀ k : ℕ, k > d → ¬(∃ (s : ℕ), a % k = s ∧ b % k = s ∧ c % k = s))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_differences_l3980_398081


namespace NUMINAMATH_CALUDE_exists_square_composition_function_l3980_398003

theorem exists_square_composition_function :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_square_composition_function_l3980_398003


namespace NUMINAMATH_CALUDE_line_equation_sum_l3980_398029

/-- Given two points on a line, proves that m + b = 7 where y = mx + b is the equation of the line. -/
theorem line_equation_sum (x₁ y₁ x₂ y₂ m b : ℚ) : 
  x₁ = 1 → y₁ = 7 → x₂ = -2 → y₂ = -1 →
  m = (y₂ - y₁) / (x₂ - x₁) →
  y₁ = m * x₁ + b →
  m + b = 7 := by
sorry

end NUMINAMATH_CALUDE_line_equation_sum_l3980_398029


namespace NUMINAMATH_CALUDE_linear_function_solution_l3980_398069

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Define the domain and range constraints
def domain_constraint (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 4
def range_constraint (y : ℝ) : Prop := 3 ≤ y ∧ y ≤ 6

-- Theorem statement
theorem linear_function_solution (k b : ℝ) :
  (∀ x, domain_constraint x → range_constraint (linear_function k b x)) →
  ((k = 1 ∧ b = 2) ∨ (k = -1 ∧ b = 7)) :=
sorry

end NUMINAMATH_CALUDE_linear_function_solution_l3980_398069


namespace NUMINAMATH_CALUDE_fence_poles_for_given_plot_l3980_398075

/-- Calculates the number of fence poles needed to enclose a rectangular plot -/
def fence_poles (length width pole_distance : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  (perimeter + pole_distance - 1) / pole_distance

/-- Theorem stating the number of fence poles needed for the given plot -/
theorem fence_poles_for_given_plot :
  fence_poles 250 150 7 = 115 := by
  sorry

end NUMINAMATH_CALUDE_fence_poles_for_given_plot_l3980_398075


namespace NUMINAMATH_CALUDE_fraction_relation_l3980_398030

theorem fraction_relation (p r s u : ℝ) 
  (h1 : p / r = 8)
  (h2 : s / r = 5)
  (h3 : s / u = 1 / 3) :
  u / p = 15 / 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l3980_398030


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3980_398017

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 2 ∧ x₂ = 2 - Real.sqrt 2 ∧
  (x₁^2 - 4*x₁ + 2 = 0) ∧ (x₂^2 - 4*x₂ + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3980_398017


namespace NUMINAMATH_CALUDE_fraction_equality_l3980_398091

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + (a + 6 * b) / (b + 6 * a) = 2) : a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3980_398091


namespace NUMINAMATH_CALUDE_sum_square_of_sum_and_diff_l3980_398011

theorem sum_square_of_sum_and_diff (x y : ℝ) 
  (sum_eq : x + y = 60) 
  (diff_eq : x - y = 10) : 
  (x + y)^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_sum_square_of_sum_and_diff_l3980_398011


namespace NUMINAMATH_CALUDE_power_of_two_equals_quadratic_plus_linear_plus_one_l3980_398097

theorem power_of_two_equals_quadratic_plus_linear_plus_one
  (x y : ℕ) (h : 2^x = y^2 + y + 1) : x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equals_quadratic_plus_linear_plus_one_l3980_398097


namespace NUMINAMATH_CALUDE_shopkeeper_loss_theorem_l3980_398064

/-- Calculates the loss percent for a shopkeeper given profit margin and theft percentage -/
def shopkeeper_loss_percent (profit_margin : ℝ) (theft_percentage : ℝ) : ℝ :=
  let selling_price := 1 + profit_margin
  let remaining_goods := 1 - theft_percentage
  let actual_revenue := selling_price * remaining_goods
  let actual_profit := actual_revenue - remaining_goods
  let net_loss := theft_percentage - actual_profit
  net_loss * 100

/-- Theorem stating that a shopkeeper with 10% profit margin and 20% theft has a 12% loss -/
theorem shopkeeper_loss_theorem :
  shopkeeper_loss_percent 0.1 0.2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_theorem_l3980_398064


namespace NUMINAMATH_CALUDE_handshake_problem_l3980_398080

theorem handshake_problem (a b : ℕ) : 
  a + b = 20 →
  (a.choose 2) + (b.choose 2) = 106 →
  a * b = 84 :=
by sorry

end NUMINAMATH_CALUDE_handshake_problem_l3980_398080


namespace NUMINAMATH_CALUDE_arithmetic_equation_l3980_398012

theorem arithmetic_equation : 4 * (8 - 3) - 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l3980_398012


namespace NUMINAMATH_CALUDE_total_savings_ten_sets_l3980_398042

-- Define the cost of 2 packs
def cost_two_packs : ℚ := 2.5

-- Define the cost of an individual pack
def cost_individual : ℚ := 1.3

-- Define the number of sets
def num_sets : ℕ := 10

-- Theorem statement
theorem total_savings_ten_sets : 
  let cost_per_pack := cost_two_packs / 2
  let savings_per_pack := cost_individual - cost_per_pack
  let total_packs := num_sets * 2
  savings_per_pack * total_packs = 1 := by sorry

end NUMINAMATH_CALUDE_total_savings_ten_sets_l3980_398042


namespace NUMINAMATH_CALUDE_f_has_one_zero_max_ab_value_l3980_398096

noncomputable def f (a b x : ℝ) : ℝ := Real.log (a * x + b) + Real.exp (x - 1)

theorem f_has_one_zero :
  ∃! x, f (-1) 1 x = 0 :=
sorry

theorem max_ab_value (a b : ℝ) (h : a ≠ 0) :
  (∀ x, f a b x ≤ Real.exp (x - 1) + x + 1) →
  a * b ≤ (1 / 2) * Real.exp 3 :=
sorry

end NUMINAMATH_CALUDE_f_has_one_zero_max_ab_value_l3980_398096


namespace NUMINAMATH_CALUDE_lady_bird_flour_theorem_l3980_398062

/-- The amount of flour needed for a given number of guests at Lady Bird's Junior League club meeting -/
def flour_needed (guests : ℕ) : ℚ :=
  let biscuits_per_guest : ℕ := 2
  let biscuits_per_batch : ℕ := 9
  let flour_per_batch : ℚ := 5 / 4
  let total_biscuits : ℕ := guests * biscuits_per_guest
  let batches : ℕ := (total_biscuits + biscuits_per_batch - 1) / biscuits_per_batch
  (batches : ℚ) * flour_per_batch

/-- Theorem stating that Lady Bird needs 5 cups of flour for 18 guests -/
theorem lady_bird_flour_theorem :
  flour_needed 18 = 5 := by
  sorry

end NUMINAMATH_CALUDE_lady_bird_flour_theorem_l3980_398062


namespace NUMINAMATH_CALUDE_compare_expressions_l3980_398018

theorem compare_expressions (m x : ℝ) : x^2 - x + 1 > -2*m^2 - 2*m*x := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l3980_398018


namespace NUMINAMATH_CALUDE_brahmagupta_formula_l3980_398052

/-- Represents a convex quadrilateral ABCD with side lengths a, b, c, d and diagonal lengths m, n -/
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  m : ℝ
  n : ℝ
  A : ℝ
  C : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  d_pos : d > 0
  m_pos : m > 0
  n_pos : n > 0

/-- The Brahmagupta's formula for a convex quadrilateral -/
theorem brahmagupta_formula (q : ConvexQuadrilateral) :
  q.m^2 * q.n^2 = q.a^2 * q.c^2 + q.b^2 * q.d^2 - 2 * q.a * q.b * q.c * q.d * Real.cos (q.A + q.C) :=
by sorry

end NUMINAMATH_CALUDE_brahmagupta_formula_l3980_398052


namespace NUMINAMATH_CALUDE_uniform_transform_l3980_398016

/-- A uniform random number between 0 and 1 -/
def uniform_random_01 : Set ℝ := Set.Icc 0 1

/-- The transformation function -/
def transform (x : ℝ) : ℝ := x * 5 - 2

/-- The set of numbers between -2 and 3 -/
def target_set : Set ℝ := Set.Icc (-2) 3

theorem uniform_transform :
  ∀ (a₁ : ℝ), a₁ ∈ uniform_random_01 → transform a₁ ∈ target_set :=
sorry

end NUMINAMATH_CALUDE_uniform_transform_l3980_398016


namespace NUMINAMATH_CALUDE_basketball_shot_probability_l3980_398008

theorem basketball_shot_probability :
  let p_at_least_one : ℝ := 0.9333333333333333
  let p_free_throw : ℝ := 4/5
  let p_high_school : ℝ := 1/2
  let p_pro : ℝ := 1/3
  (1 - (1 - p_free_throw) * (1 - p_high_school) * (1 - p_pro) = p_at_least_one) :=
by sorry

end NUMINAMATH_CALUDE_basketball_shot_probability_l3980_398008


namespace NUMINAMATH_CALUDE_sum_of_roots_l3980_398053

theorem sum_of_roots (a b : ℝ) : 
  a ≠ b → 
  let M : Set ℝ := {a^2 - 4*a, -1}
  let N : Set ℝ := {b^2 - 4*b + 1, -2}
  ∃ f : ℝ → ℝ, (∀ x ∈ M, f x = x ∧ f x ∈ N) →
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3980_398053


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_15_12_l3980_398074

theorem half_abs_diff_squares_15_12 : (1/2 : ℝ) * |15^2 - 12^2| = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_15_12_l3980_398074


namespace NUMINAMATH_CALUDE_sandra_beignets_l3980_398087

/-- The number of beignets Sandra eats in 16 weeks -/
def total_beignets : ℕ := 336

/-- The number of weeks -/
def num_weeks : ℕ := 16

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of beignets Sandra eats every morning -/
def beignets_per_morning : ℕ := total_beignets / (num_weeks * days_per_week)

theorem sandra_beignets : beignets_per_morning = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandra_beignets_l3980_398087


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3980_398059

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁ + 1) * (x₁ - 1) = 2 * x₁ + 3 ∧ 
  (x₂ + 1) * (x₂ - 1) = 2 * x₂ + 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3980_398059


namespace NUMINAMATH_CALUDE_subtraction_multiplication_equality_l3980_398056

theorem subtraction_multiplication_equality : 
  ((2000000000000 - 1111111111111) * 2) = 1777777777778 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_equality_l3980_398056


namespace NUMINAMATH_CALUDE_danielle_spending_l3980_398027

/-- Represents the cost and yield of supplies for making popsicles. -/
structure PopsicleSupplies where
  mold_cost : ℕ
  stick_pack_cost : ℕ
  stick_pack_size : ℕ
  juice_bottle_cost : ℕ
  popsicles_per_bottle : ℕ
  remaining_sticks : ℕ

/-- Calculates the total cost of supplies for making popsicles. -/
def total_cost (supplies : PopsicleSupplies) : ℕ :=
  supplies.mold_cost + supplies.stick_pack_cost +
  (supplies.stick_pack_size - supplies.remaining_sticks) / supplies.popsicles_per_bottle * supplies.juice_bottle_cost

/-- Theorem stating that Danielle's total spending on supplies equals $10. -/
theorem danielle_spending (supplies : PopsicleSupplies)
  (h1 : supplies.mold_cost = 3)
  (h2 : supplies.stick_pack_cost = 1)
  (h3 : supplies.stick_pack_size = 100)
  (h4 : supplies.juice_bottle_cost = 2)
  (h5 : supplies.popsicles_per_bottle = 20)
  (h6 : supplies.remaining_sticks = 40) :
  total_cost supplies = 10 := by
    sorry

end NUMINAMATH_CALUDE_danielle_spending_l3980_398027


namespace NUMINAMATH_CALUDE_area_ratio_is_five_sevenths_l3980_398013

-- Define the points
variable (A B C D O P X Y : ℝ × ℝ)

-- Define the lengths
def AD : ℝ := 13
def AO : ℝ := 13
def OB : ℝ := 13
def BC : ℝ := 13
def AB : ℝ := 15
def DO : ℝ := 15
def OC : ℝ := 15

-- Define the conditions
axiom triangle_dao_isosceles : AO = AD
axiom triangle_aob_isosceles : AO = OB
axiom triangle_obc_isosceles : OB = BC
axiom p_on_ab : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B
axiom op_perpendicular_ab : (P.1 - O.1) * (B.1 - A.1) + (P.2 - O.2) * (B.2 - A.2) = 0
axiom x_midpoint_ad : X = ((A.1 + D.1) / 2, (A.2 + D.2) / 2)
axiom y_midpoint_bc : Y = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the areas of trapezoids
def area_ABYX : ℝ := sorry
def area_XYCD : ℝ := sorry

-- State the theorem
theorem area_ratio_is_five_sevenths :
  area_ABYX / area_XYCD = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_area_ratio_is_five_sevenths_l3980_398013


namespace NUMINAMATH_CALUDE_egg_production_increase_l3980_398085

/-- The number of eggs produced last year -/
def last_year_production : ℕ := 1416

/-- The number of eggs produced this year -/
def this_year_production : ℕ := 4636

/-- The increase in egg production -/
def production_increase : ℕ := this_year_production - last_year_production

theorem egg_production_increase :
  production_increase = 3220 := by sorry

end NUMINAMATH_CALUDE_egg_production_increase_l3980_398085


namespace NUMINAMATH_CALUDE_transportation_budget_degrees_l3980_398089

theorem transportation_budget_degrees (salaries research_and_development utilities equipment supplies : ℝ)
  (h1 : salaries = 60)
  (h2 : research_and_development = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : supplies = 2)
  (h6 : salaries + research_and_development + utilities + equipment + supplies < 100) :
  let transportation := 100 - (salaries + research_and_development + utilities + equipment + supplies)
  (transportation / 100) * 360 = 72 := by
sorry

end NUMINAMATH_CALUDE_transportation_budget_degrees_l3980_398089


namespace NUMINAMATH_CALUDE_constant_term_quadratic_l3980_398094

theorem constant_term_quadratic (x : ℝ) : 
  (2 * x^2 = x + 4) → 
  (∃ a b : ℝ, 2 * x^2 - x - 4 = a * x^2 + b * x + (-4)) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_quadratic_l3980_398094


namespace NUMINAMATH_CALUDE_calculation_proof_l3980_398046

theorem calculation_proof : 8500 + 45 * 2 - 500 / 25 + 100 = 8670 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3980_398046


namespace NUMINAMATH_CALUDE_sum_interior_angles_180_l3980_398005

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define interior angles of a triangle
def interior_angles (t : Triangle) : Fin 3 → ℝ := sorry

-- Theorem: The sum of interior angles of any triangle is 180°
theorem sum_interior_angles_180 (t : Triangle) : 
  (interior_angles t 0) + (interior_angles t 1) + (interior_angles t 2) = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_180_l3980_398005


namespace NUMINAMATH_CALUDE_sphere_volume_l3980_398078

theorem sphere_volume (prism_length prism_width prism_height : ℝ) 
  (sphere_volume : ℝ → ℝ) (L : ℝ) :
  prism_length = 4 →
  prism_width = 2 →
  prism_height = 1 →
  (∀ r : ℝ, sphere_volume r = (4 / 3) * π * r^3) →
  (∃ r : ℝ, 4 * π * r^2 = 2 * (prism_length * prism_width + 
    prism_length * prism_height + prism_width * prism_height)) →
  (∃ r : ℝ, sphere_volume r = L * Real.sqrt 2 / Real.sqrt π) →
  L = 14 * Real.sqrt 14 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_l3980_398078


namespace NUMINAMATH_CALUDE_gcd_1189_264_l3980_398009

theorem gcd_1189_264 : Nat.gcd 1189 264 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1189_264_l3980_398009


namespace NUMINAMATH_CALUDE_pen_purchasing_plans_l3980_398067

theorem pen_purchasing_plans :
  ∃! (solutions : List (ℕ × ℕ)), 
    solutions.length = 3 ∧
    (∀ (x y : ℕ), (x, y) ∈ solutions ↔ 
      x > 0 ∧ y > 0 ∧ 15 * x + 10 * y = 105) :=
by sorry

end NUMINAMATH_CALUDE_pen_purchasing_plans_l3980_398067


namespace NUMINAMATH_CALUDE_company_french_speakers_l3980_398023

theorem company_french_speakers 
  (total_employees : ℝ) 
  (total_employees_positive : 0 < total_employees) :
  let men_percentage : ℝ := 65 / 100
  let women_percentage : ℝ := 1 - men_percentage
  let men_french_speakers_percentage : ℝ := 60 / 100
  let women_non_french_speakers_percentage : ℝ := 97.14285714285714 / 100
  let men_count : ℝ := men_percentage * total_employees
  let women_count : ℝ := women_percentage * total_employees
  let men_french_speakers : ℝ := men_french_speakers_percentage * men_count
  let women_french_speakers : ℝ := (1 - women_non_french_speakers_percentage) * women_count
  let total_french_speakers : ℝ := men_french_speakers + women_french_speakers
  let french_speakers_percentage : ℝ := total_french_speakers / total_employees * 100
  french_speakers_percentage = 40 := by
sorry


end NUMINAMATH_CALUDE_company_french_speakers_l3980_398023


namespace NUMINAMATH_CALUDE_find_number_B_l3980_398043

/-- Given that A = 5 and A = 2.8B - 0.6, prove that B = 2 -/
theorem find_number_B (A B : ℝ) (h1 : A = 5) (h2 : A = 2.8 * B - 0.6) : B = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_number_B_l3980_398043


namespace NUMINAMATH_CALUDE_a_minus_b_equals_1790_l3980_398050

/-- Prove that A - B = 1790 given the definitions of A and B -/
theorem a_minus_b_equals_1790 :
  let A := 1 * 1000 + 16 * 100 + 28 * 10
  let B := 355 + 3 * 245
  A - B = 1790 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_1790_l3980_398050


namespace NUMINAMATH_CALUDE_equation_solution_l3980_398063

theorem equation_solution : ∃! x : ℚ, (10 - 2*x)^2 = 4*x^2 ∧ x = 5/2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3980_398063


namespace NUMINAMATH_CALUDE_jess_walked_five_blocks_l3980_398036

/-- The number of blocks Jess has already walked -/
def blocks_walked (total_blocks remaining_blocks : ℕ) : ℕ :=
  total_blocks - remaining_blocks

/-- Proof that Jess has walked 5 blocks -/
theorem jess_walked_five_blocks :
  blocks_walked 25 20 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jess_walked_five_blocks_l3980_398036


namespace NUMINAMATH_CALUDE_student_selection_probability_l3980_398039

theorem student_selection_probability (n : ℕ) : 
  (4 : ℝ) ≥ 0 ∧ (n : ℝ) ≥ 0 →
  (((n + 4) * (n + 3) / 2 - 6) / ((n + 4) * (n + 3) / 2) = 5 / 6) →
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_student_selection_probability_l3980_398039


namespace NUMINAMATH_CALUDE_binomial_coefficient_x7_l3980_398077

theorem binomial_coefficient_x7 (a : ℝ) : 
  (Nat.choose 10 7 : ℝ) * a^3 = 15 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x7_l3980_398077


namespace NUMINAMATH_CALUDE_xy_equals_nine_x_div_y_equals_thirtysix_l3980_398035

theorem xy_equals_nine_x_div_y_equals_thirtysix (x y : ℝ) 
  (h1 : x * y = 9)
  (h2 : x / y = 36)
  (hx : x > 0)
  (hy : y > 0) :
  y = 1/2 := by
sorry

end NUMINAMATH_CALUDE_xy_equals_nine_x_div_y_equals_thirtysix_l3980_398035


namespace NUMINAMATH_CALUDE_inverse_not_always_true_l3980_398000

theorem inverse_not_always_true :
  ¬(∀ (a b m : ℝ), (a < b → a * m^2 < b * m^2)) :=
sorry

end NUMINAMATH_CALUDE_inverse_not_always_true_l3980_398000


namespace NUMINAMATH_CALUDE_math_representative_selection_l3980_398033

theorem math_representative_selection (boys girls : ℕ) 
  (h_boys : boys = 36) 
  (h_girls : girls = 28) : 
  (boys + girls : ℕ) = 64 := by
  sorry

end NUMINAMATH_CALUDE_math_representative_selection_l3980_398033


namespace NUMINAMATH_CALUDE_problem_solution_l3980_398048

theorem problem_solution (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3980_398048


namespace NUMINAMATH_CALUDE_max_cake_boxes_in_carton_l3980_398024

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the carton dimensions -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- Represents the cake box dimensions -/
def cakeBoxDimensions : BoxDimensions :=
  { length := 8, width := 7, height := 5 }

/-- Theorem stating the maximum number of cake boxes that can fit in the carton -/
theorem max_cake_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume cakeBoxDimensions) = 225 := by
  sorry

end NUMINAMATH_CALUDE_max_cake_boxes_in_carton_l3980_398024
