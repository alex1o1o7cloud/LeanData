import Mathlib

namespace NUMINAMATH_CALUDE_alik_collection_l1581_158146

theorem alik_collection (badges bracelets : ℕ) (n : ℚ) : 
  badges > bracelets →
  badges + n * bracelets = 100 →
  n * badges + bracelets = 101 →
  ((badges = 34 ∧ bracelets = 33) ∨ (badges = 66 ∧ bracelets = 33)) :=
by sorry

end NUMINAMATH_CALUDE_alik_collection_l1581_158146


namespace NUMINAMATH_CALUDE_tetromino_properties_l1581_158130

-- Define a tetromino as a shape formed from 4 squares
structure Tetromino :=
  (squares : Finset (ℤ × ℤ))
  (size : squares.card = 4)

-- Define rotation equivalence
def rotationEquivalent (t1 t2 : Tetromino) : Prop := sorry

-- Define the set of distinct tetrominos
def distinctTetrominos : Finset Tetromino := sorry

-- Define a tiling of a rectangle
def tiling (w h : ℕ) (pieces : Finset Tetromino) : Prop := sorry

theorem tetromino_properties :
  -- There are exactly 7 distinct tetrominos
  distinctTetrominos.card = 7 ∧
  -- It is impossible to tile a 4 × 7 rectangle with one of each distinct tetromino
  ¬ tiling 4 7 distinctTetrominos := by sorry

end NUMINAMATH_CALUDE_tetromino_properties_l1581_158130


namespace NUMINAMATH_CALUDE_stratified_sampling_teachers_l1581_158121

theorem stratified_sampling_teachers :
  let total_teachers : ℕ := 150
  let senior_teachers : ℕ := 45
  let intermediate_teachers : ℕ := 90
  let junior_teachers : ℕ := 15
  let sample_size : ℕ := 30
  let sample_senior : ℕ := 9
  let sample_intermediate : ℕ := 18
  let sample_junior : ℕ := 3
  
  (total_teachers = senior_teachers + intermediate_teachers + junior_teachers) →
  (sample_size = sample_senior + sample_intermediate + sample_junior) →
  (sample_senior : ℚ) / senior_teachers = (sample_intermediate : ℚ) / intermediate_teachers →
  (sample_senior : ℚ) / senior_teachers = (sample_junior : ℚ) / junior_teachers →
  (sample_size : ℚ) / total_teachers = (sample_senior : ℚ) / senior_teachers :=
by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_teachers_l1581_158121


namespace NUMINAMATH_CALUDE_tennis_balls_first_set_l1581_158108

theorem tennis_balls_first_set :
  ∀ (total_balls first_set second_set : ℕ),
    total_balls = 175 →
    second_set = 75 →
    first_set + second_set = total_balls →
    (2 : ℚ) / 5 * first_set + (1 : ℚ) / 3 * second_set + 110 = total_balls →
    first_set = 100 := by
  sorry

end NUMINAMATH_CALUDE_tennis_balls_first_set_l1581_158108


namespace NUMINAMATH_CALUDE_y_intercepts_of_curve_l1581_158147

/-- The y-intercepts of the curve 3x + 5y^2 = 25 are (0, √5) and (0, -√5) -/
theorem y_intercepts_of_curve (x y : ℝ) :
  3*x + 5*y^2 = 25 ∧ x = 0 ↔ y = Real.sqrt 5 ∨ y = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_y_intercepts_of_curve_l1581_158147


namespace NUMINAMATH_CALUDE_triangle_vector_relation_l1581_158179

theorem triangle_vector_relation (A B C : ℝ × ℝ) (a b : ℝ × ℝ) :
  (B.1 - C.1, B.2 - C.2) = a →
  (C.1 - A.1, C.2 - A.2) = b →
  (A.1 - B.1, A.2 - B.2) = (b.1 - a.1, b.2 - a.2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_relation_l1581_158179


namespace NUMINAMATH_CALUDE_video_votes_l1581_158160

theorem video_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 140 ∧ like_percentage = 70 / 100 → 
  ∃ (total_votes : ℕ), 
    (like_percentage : ℚ) * total_votes - (1 - like_percentage) * total_votes = score ∧
    total_votes = 350 := by
  sorry

end NUMINAMATH_CALUDE_video_votes_l1581_158160


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_including_11_l1581_158188

theorem unique_number_with_three_prime_divisors_including_11 :
  ∀ (x n : ℕ), 
    x = 9^n - 1 →
    (∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
    11 ∣ x →
    x = 59048 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_including_11_l1581_158188


namespace NUMINAMATH_CALUDE_complex_real_condition_l1581_158132

theorem complex_real_condition (a : ℝ) : 
  let Z : ℂ := (a - 5) / (a^2 + 4*a - 5) + (a^2 + 2*a - 15) * I
  Z.im = 0 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1581_158132


namespace NUMINAMATH_CALUDE_soda_cost_l1581_158137

theorem soda_cost (burger_cost soda_cost : ℕ) : 
  (4 * burger_cost + 3 * soda_cost = 440) →
  (3 * burger_cost + 2 * soda_cost = 310) →
  soda_cost = 80 := by
sorry

end NUMINAMATH_CALUDE_soda_cost_l1581_158137


namespace NUMINAMATH_CALUDE_factorization_x4_minus_81_l1581_158152

theorem factorization_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_81_l1581_158152


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1581_158124

/-- The speed of a boat in still water, given downstream travel information and current speed. -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (travel_time_minutes : ℝ) :
  current_speed = 8 →
  downstream_distance = 36.67 →
  travel_time_minutes = 44 →
  ∃ (boat_speed : ℝ), boat_speed = 42 ∧ 
    (boat_speed + current_speed) * (travel_time_minutes / 60) = downstream_distance :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1581_158124


namespace NUMINAMATH_CALUDE_inequality_solution_l1581_158171

theorem inequality_solution (x : ℝ) : 
  x ≠ 2 → (x * (x + 1)) / ((x - 2)^2) ≥ 8 ↔ x ∈ Set.Ici 1 ∩ Set.Iio 2 ∪ Set.Ioi (32/7) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1581_158171


namespace NUMINAMATH_CALUDE_van_transport_l1581_158113

theorem van_transport (students_per_van : ℕ) (num_boys : ℕ) (num_girls : ℕ) 
  (h1 : students_per_van = 28)
  (h2 : num_boys = 60)
  (h3 : num_girls = 80) :
  (num_boys + num_girls) / students_per_van = 5 := by
  sorry

#check van_transport

end NUMINAMATH_CALUDE_van_transport_l1581_158113


namespace NUMINAMATH_CALUDE_arcade_tickets_l1581_158143

theorem arcade_tickets (initial_tickets spent_tickets additional_tickets : ℕ) :
  initial_tickets ≥ spent_tickets →
  initial_tickets - spent_tickets + additional_tickets =
    initial_tickets + additional_tickets - spent_tickets :=
by sorry

end NUMINAMATH_CALUDE_arcade_tickets_l1581_158143


namespace NUMINAMATH_CALUDE_complex_number_equality_l1581_158193

theorem complex_number_equality : Complex.I * (1 - Complex.I)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1581_158193


namespace NUMINAMATH_CALUDE_angle_relationship_l1581_158195

theorem angle_relationship (larger_angle smaller_angle : ℝ) : 
  larger_angle = 99 ∧ smaller_angle = 81 → larger_angle - smaller_angle = 18 := by
  sorry

end NUMINAMATH_CALUDE_angle_relationship_l1581_158195


namespace NUMINAMATH_CALUDE_total_fruits_picked_l1581_158166

theorem total_fruits_picked (joan_oranges sara_oranges carlos_oranges 
                             alyssa_pears ben_pears vanessa_pears 
                             tim_apples linda_apples : ℕ) 
                            (h1 : joan_oranges = 37)
                            (h2 : sara_oranges = 10)
                            (h3 : carlos_oranges = 25)
                            (h4 : alyssa_pears = 30)
                            (h5 : ben_pears = 40)
                            (h6 : vanessa_pears = 20)
                            (h7 : tim_apples = 15)
                            (h8 : linda_apples = 10) :
  joan_oranges + sara_oranges + carlos_oranges + 
  alyssa_pears + ben_pears + vanessa_pears + 
  tim_apples + linda_apples = 187 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_picked_l1581_158166


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_350_l1581_158100

theorem closest_perfect_square_to_350 :
  ∀ n : ℕ, n ≠ 19 → (n ^ 2 : ℤ) ≠ 361 → |350 - (19 ^ 2 : ℤ)| ≤ |350 - (n ^ 2 : ℤ)| :=
by sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_350_l1581_158100


namespace NUMINAMATH_CALUDE_diamond_three_four_l1581_158190

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := a^2 + 2*a*b

-- Define the ◇ operation
def diamond (a b : ℝ) : ℝ := 4*a + 6*b - (oplus a b)

-- Theorem statement
theorem diamond_three_four : diamond 3 4 = 3 := by sorry

end NUMINAMATH_CALUDE_diamond_three_four_l1581_158190


namespace NUMINAMATH_CALUDE_base_h_solution_l1581_158158

/-- Represents a digit in base h --/
def Digit (h : ℕ) := {d : ℕ // d < h}

/-- Converts a natural number to its representation in base h --/
def toBaseH (n h : ℕ) : List (Digit h) :=
  sorry

/-- Performs addition in base h --/
def addBaseH (a b : List (Digit h)) : List (Digit h) :=
  sorry

/-- The given addition problem --/
def additionProblem (h : ℕ) : Prop :=
  let a := toBaseH 5342 h
  let b := toBaseH 6421 h
  let result := toBaseH 14263 h
  addBaseH a b = result

theorem base_h_solution :
  ∃ h : ℕ, h > 0 ∧ additionProblem h ∧ h = 8 :=
sorry

end NUMINAMATH_CALUDE_base_h_solution_l1581_158158


namespace NUMINAMATH_CALUDE_fourth_transaction_is_37_l1581_158128

/-- Represents the balance of class funds after a series of transactions -/
def class_funds (initial_balance : Int) (transactions : List Int) : Int :=
  initial_balance + transactions.sum

/-- Theorem: Given the initial balance and three transactions, 
    the fourth transaction must be 37 to reach the final balance of 82 -/
theorem fourth_transaction_is_37 
  (initial_balance : Int)
  (transaction1 transaction2 transaction3 : Int)
  (h1 : initial_balance = 0)
  (h2 : transaction1 = 230)
  (h3 : transaction2 = -75)
  (h4 : transaction3 = -110) :
  class_funds initial_balance [transaction1, transaction2, transaction3, 37] = 82 := by
  sorry

#eval class_funds 0 [230, -75, -110, 37]

end NUMINAMATH_CALUDE_fourth_transaction_is_37_l1581_158128


namespace NUMINAMATH_CALUDE_luncheon_attendance_l1581_158134

theorem luncheon_attendance (invited : ℕ) (table_capacity : ℕ) (tables_used : ℕ) 
  (h1 : invited = 24) 
  (h2 : table_capacity = 7) 
  (h3 : tables_used = 2) : 
  invited - (table_capacity * tables_used) = 10 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_attendance_l1581_158134


namespace NUMINAMATH_CALUDE_planes_parallel_if_common_perpendicular_l1581_158101

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_common_perpendicular 
  (a b : Plane) (m : Line) : 
  a ≠ b → 
  perpendicular m a → 
  perpendicular m b → 
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_common_perpendicular_l1581_158101


namespace NUMINAMATH_CALUDE_monday_sales_calculation_l1581_158144

def total_stock : ℕ := 1300
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135
def unsold_percentage : ℚ := 69.07692307692308

theorem monday_sales_calculation :
  ∃ (monday_sales : ℕ),
    monday_sales = total_stock - tuesday_sales - wednesday_sales - thursday_sales - friday_sales -
      (unsold_percentage / 100 * total_stock).floor ∧
    monday_sales = 75 := by
  sorry

end NUMINAMATH_CALUDE_monday_sales_calculation_l1581_158144


namespace NUMINAMATH_CALUDE_z_power_2017_l1581_158123

theorem z_power_2017 (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) : 
  z^2017 = Complex.I := by
sorry

end NUMINAMATH_CALUDE_z_power_2017_l1581_158123


namespace NUMINAMATH_CALUDE_nonnegative_integer_solutions_l1581_158119

theorem nonnegative_integer_solutions : 
  {(x, y) : ℕ × ℕ | 3 * x^2 + 2 * 9^y = x * (4^(y + 1) - 1)} = {(3, 1), (2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_integer_solutions_l1581_158119


namespace NUMINAMATH_CALUDE_multiple_of_8_in_second_column_thousand_in_second_column_l1581_158140

/-- Represents the column number in the arrangement -/
inductive Column
| First
| Second
| Third
| Fourth
| Fifth

/-- Represents the row type in the arrangement -/
inductive RowType
| Odd
| Even

/-- Function to determine the column of a given integer in the arrangement -/
def column_of_integer (n : ℕ) : Column :=
  sorry

/-- Function to determine the row type of a given integer in the arrangement -/
def row_type_of_integer (n : ℕ) : RowType :=
  sorry

/-- Theorem stating that any multiple of 8 appears in the second column -/
theorem multiple_of_8_in_second_column (n : ℕ) (h : 8 ∣ n) : column_of_integer n = Column.Second :=
  sorry

/-- Corollary: 1000 appears in the second column -/
theorem thousand_in_second_column : column_of_integer 1000 = Column.Second :=
  sorry

end NUMINAMATH_CALUDE_multiple_of_8_in_second_column_thousand_in_second_column_l1581_158140


namespace NUMINAMATH_CALUDE_megan_zoo_pictures_l1581_158126

/-- Represents the number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := sorry

/-- Represents the total number of pictures Megan took -/
def total_pictures : ℕ := zoo_pictures + 18

/-- Represents the number of pictures remaining after deletion -/
def remaining_pictures : ℕ := total_pictures - 31

theorem megan_zoo_pictures : 
  zoo_pictures = 15 ∧ 
  total_pictures = zoo_pictures + 18 ∧ 
  remaining_pictures = 2 :=
sorry

end NUMINAMATH_CALUDE_megan_zoo_pictures_l1581_158126


namespace NUMINAMATH_CALUDE_max_profit_at_max_price_max_profit_value_mall_sale_max_profit_l1581_158149

/-- Represents the shopping mall's clothing sale scenario -/
structure ClothingSale where
  cost : ℝ
  sales_function : ℝ → ℝ
  profit_function : ℝ → ℝ
  min_price : ℝ
  max_price : ℝ

/-- The specific clothing sale scenario as described in the problem -/
def mall_sale : ClothingSale :=
  { cost := 60
  , sales_function := λ x => -x + 120
  , profit_function := λ x => (x - 60) * (-x + 120)
  , min_price := 60
  , max_price := 84
  }

/-- Theorem stating that the maximum profit is achieved at the highest allowed price -/
theorem max_profit_at_max_price (sale : ClothingSale) :
  ∀ x ∈ Set.Icc sale.min_price sale.max_price,
    sale.profit_function x ≤ sale.profit_function sale.max_price :=
sorry

/-- Theorem stating that the maximum profit is 864 dollars -/
theorem max_profit_value (sale : ClothingSale) :
  sale.profit_function sale.max_price = 864 :=
sorry

/-- Main theorem combining the above results -/
theorem mall_sale_max_profit :
  ∃ x ∈ Set.Icc mall_sale.min_price mall_sale.max_price,
    mall_sale.profit_function x = 864 ∧
    ∀ y ∈ Set.Icc mall_sale.min_price mall_sale.max_price,
      mall_sale.profit_function y ≤ 864 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_max_price_max_profit_value_mall_sale_max_profit_l1581_158149


namespace NUMINAMATH_CALUDE_third_generation_tail_length_l1581_158172

/-- The tail length of a generation of kittens -/
def tail_length (n : ℕ) : ℝ :=
  if n = 0 then 16
  else tail_length (n - 1) * 1.25

/-- The theorem stating that the third generation's tail length is 25 cm -/
theorem third_generation_tail_length :
  tail_length 2 = 25 := by sorry

end NUMINAMATH_CALUDE_third_generation_tail_length_l1581_158172


namespace NUMINAMATH_CALUDE_nick_quarters_count_l1581_158145

-- Define the total number of quarters
def total_quarters : ℕ := 35

-- Define the fraction of state quarters
def state_quarter_fraction : ℚ := 2 / 5

-- Define the fraction of Pennsylvania quarters among state quarters
def pennsylvania_quarter_fraction : ℚ := 1 / 2

-- Define the number of Pennsylvania quarters
def pennsylvania_quarters : ℕ := 7

-- Theorem statement
theorem nick_quarters_count :
  (pennsylvania_quarter_fraction * state_quarter_fraction * total_quarters : ℚ) = pennsylvania_quarters :=
by sorry

end NUMINAMATH_CALUDE_nick_quarters_count_l1581_158145


namespace NUMINAMATH_CALUDE_root_difference_of_arithmetic_progression_l1581_158189

-- Define the polynomial coefficients
def a : ℝ := 81
def b : ℝ := -171
def c : ℝ := 107
def d : ℝ := -18

-- Define the polynomial
def p (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem root_difference_of_arithmetic_progression :
  ∃ (r₁ r₂ r₃ : ℝ),
    -- The roots satisfy the polynomial equation
    p r₁ = 0 ∧ p r₂ = 0 ∧ p r₃ = 0 ∧
    -- The roots are in arithmetic progression
    r₂ - r₁ = r₃ - r₂ ∧
    -- The difference between the largest and smallest roots is approximately 1.66
    abs (max r₁ (max r₂ r₃) - min r₁ (min r₂ r₃) - 1.66) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_root_difference_of_arithmetic_progression_l1581_158189


namespace NUMINAMATH_CALUDE_triangle_side_value_l1581_158127

theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) : 
  (b^2 - c^2 + 2*a = 0) →
  (Real.tan C / Real.tan B = 3) →
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l1581_158127


namespace NUMINAMATH_CALUDE_largest_share_example_l1581_158105

/-- Represents the profit distribution for partners in a business --/
structure ProfitDistribution where
  ratios : List Nat
  total_profit : ℚ

/-- Calculates the largest share of profit given a profit distribution --/
def largest_share (pd : ProfitDistribution) : ℚ :=
  let total_parts := pd.ratios.sum
  let value_per_part := pd.total_profit / total_parts
  value_per_part * pd.ratios.maximum.getD 0

/-- Theorem stating that for the given profit distribution, the largest share is $11,333.35 --/
theorem largest_share_example : 
  let pd : ProfitDistribution := { 
    ratios := [2, 3, 4, 1, 5],
    total_profit := 34000
  }
  largest_share pd = 11333.35 := by
  sorry

#eval largest_share { ratios := [2, 3, 4, 1, 5], total_profit := 34000 }

end NUMINAMATH_CALUDE_largest_share_example_l1581_158105


namespace NUMINAMATH_CALUDE_governor_addresses_l1581_158156

theorem governor_addresses (sandoval hawkins sloan : ℕ) : 
  hawkins = sandoval / 2 →
  sloan = sandoval + 10 →
  sandoval + hawkins + sloan = 40 →
  sandoval = 12 := by
sorry

end NUMINAMATH_CALUDE_governor_addresses_l1581_158156


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1581_158154

theorem stratified_sampling_theorem (first_grade second_grade third_grade total_selected : ℕ) 
  (h1 : first_grade = 120)
  (h2 : second_grade = 180)
  (h3 : third_grade = 150)
  (h4 : total_selected = 90) :
  let total_students := first_grade + second_grade + third_grade
  let sampling_ratio := total_selected / total_students
  (sampling_ratio * first_grade : ℕ) = 24 ∧
  (sampling_ratio * second_grade : ℕ) = 36 ∧
  (sampling_ratio * third_grade : ℕ) = 30 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1581_158154


namespace NUMINAMATH_CALUDE_bug_meeting_point_l1581_158107

/-- Triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point on the perimeter of a triangle -/
structure PerimeterPoint where
  distanceFromP : ℝ

/-- Theorem: In a triangle PQR with side lengths PQ=7, QR=8, and PR=9, 
    if two bugs start simultaneously from P and crawl along the perimeter 
    in opposite directions at the same speed, meeting at point S, 
    then the length of QS is 5. -/
theorem bug_meeting_point (t : Triangle) (s : PerimeterPoint) : 
  t.a = 7 ∧ t.b = 8 ∧ t.c = 9 → s.distanceFromP = 12 → t.a + s.distanceFromP - 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bug_meeting_point_l1581_158107


namespace NUMINAMATH_CALUDE_special_equation_result_l1581_158136

/-- If y is a real number satisfying y + 1/y = 3, then y^13 - 5y^9 + y^5 = 0 -/
theorem special_equation_result (y : ℝ) (h : y + 1/y = 3) : y^13 - 5*y^9 + y^5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_equation_result_l1581_158136


namespace NUMINAMATH_CALUDE_unique_intersection_l1581_158184

/-- The value of m for which the line x = m intersects the parabola x = -3y² - 4y + 7 at exactly one point -/
def intersection_point : ℚ := 25 / 3

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

theorem unique_intersection :
  ∀ m : ℝ, (∃! y : ℝ, parabola y = m) ↔ m = intersection_point := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l1581_158184


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1581_158186

/-- The base length of an isosceles triangle with specific conditions -/
theorem isosceles_triangle_base_length 
  (equilateral_perimeter : ℝ) 
  (isosceles_perimeter : ℝ) 
  (h_equilateral : equilateral_perimeter = 60) 
  (h_isosceles : isosceles_perimeter = 45) 
  (h_shared_side : equilateral_perimeter / 3 = (isosceles_perimeter - isosceles_base) / 2) : 
  isosceles_base = 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1581_158186


namespace NUMINAMATH_CALUDE_least_coins_ten_coins_coins_in_wallet_l1581_158157

theorem least_coins (n : ℕ) : (n % 7 = 3 ∧ n % 4 = 2) → n ≥ 10 :=
by sorry

theorem ten_coins : (10 % 7 = 3) ∧ (10 % 4 = 2) :=
by sorry

theorem coins_in_wallet : ∃ (n : ℕ), n % 7 = 3 ∧ n % 4 = 2 ∧ ∀ (m : ℕ), (m % 7 = 3 ∧ m % 4 = 2) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_coins_ten_coins_coins_in_wallet_l1581_158157


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l1581_158161

/-- For an infinite geometric series with first term a and sum S, 
    the common ratio r can be calculated. -/
theorem infinite_geometric_series_ratio 
  (a : ℝ) (S : ℝ) (h1 : a = 400) (h2 : S = 2500) :
  let r := 1 - a / S
  r = 21 / 25 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l1581_158161


namespace NUMINAMATH_CALUDE_expansion_equality_l1581_158103

-- Define a positive integer n
variable (n : ℕ+)

-- Define the condition that the coefficient of x^3 is the same in both expansions
def coefficient_equality (n : ℕ+) : Prop :=
  (Nat.choose (2 * n) 3) = 2 * (Nat.choose n 1)

-- Theorem statement
theorem expansion_equality (n : ℕ+) (h : coefficient_equality n) :
  n = 2 ∧ 
  ∀ k : ℕ, k ≤ n → 2 * (Nat.choose n k) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_expansion_equality_l1581_158103


namespace NUMINAMATH_CALUDE_range_of_a_l1581_158122

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) → a > -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1581_158122


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l1581_158185

/-- Calculates the distance traveled downstream by a boat given its speed in still water,
    the stream speed, and the time taken. -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat with a speed of 25 km/hr in still water, traveling downstream
    in a stream with a speed of 5 km/hr for 4 hours, travels a distance of 120 km. -/
theorem boat_downstream_distance :
  distance_downstream 25 5 4 = 120 := by
  sorry

#eval distance_downstream 25 5 4

end NUMINAMATH_CALUDE_boat_downstream_distance_l1581_158185


namespace NUMINAMATH_CALUDE_sphere_radius_from_hole_l1581_158180

/-- Given a sphere intersecting a plane, if the resulting circular hole has a diameter of 30 cm
    and a depth of 10 cm, then the radius of the sphere is 16.25 cm. -/
theorem sphere_radius_from_hole (r : ℝ) (h : r > 0) :
  (∃ x : ℝ, x > 0 ∧ x^2 + 15^2 = (x + 10)^2 ∧ r^2 = x^2 + 15^2) →
  r = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_hole_l1581_158180


namespace NUMINAMATH_CALUDE_ben_time_to_school_l1581_158151

/-- Represents the walking parameters of a person -/
structure WalkingParams where
  steps_per_minute : ℕ
  step_length : ℕ
  time_to_school : ℕ

/-- Calculates the time it takes for a person to walk to school given their walking parameters and the distance to school -/
def time_to_school (params : WalkingParams) (distance : ℕ) : ℚ :=
  distance / (params.steps_per_minute * params.step_length)

theorem ben_time_to_school 
  (amy : WalkingParams)
  (ben : WalkingParams)
  (h1 : amy.steps_per_minute = 80)
  (h2 : amy.step_length = 70)
  (h3 : amy.time_to_school = 20)
  (h4 : ben.steps_per_minute = 120)
  (h5 : ben.step_length = 50) :
  time_to_school ben (amy.steps_per_minute * amy.step_length * amy.time_to_school) = 56/3 := by
  sorry

end NUMINAMATH_CALUDE_ben_time_to_school_l1581_158151


namespace NUMINAMATH_CALUDE_average_of_numbers_l1581_158173

def numbers : List ℝ := [12, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : 
  (numbers.sum / numbers.length : ℝ) = 125830.7 := by sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1581_158173


namespace NUMINAMATH_CALUDE_kids_at_camp_l1581_158170

theorem kids_at_camp (total : ℕ) (home : ℕ) (difference : ℕ) : 
  total = home + (home + difference) → 
  home = 668278 → 
  difference = 150780 → 
  home + difference = 409529 :=
by sorry

end NUMINAMATH_CALUDE_kids_at_camp_l1581_158170


namespace NUMINAMATH_CALUDE_tan_90_degrees_undefined_l1581_158162

theorem tan_90_degrees_undefined :
  let θ : Real := 90 * Real.pi / 180  -- Convert 90 degrees to radians
  ∀ (tan sin cos : Real → Real),
    (∀ α, tan α = sin α / cos α) →    -- Definition of tangent
    sin θ = 1 →                       -- Given: sin 90° = 1
    cos θ = 0 →                       -- Given: cos 90° = 0
    ¬∃ (x : Real), tan θ = x          -- tan 90° is undefined
  := by sorry

end NUMINAMATH_CALUDE_tan_90_degrees_undefined_l1581_158162


namespace NUMINAMATH_CALUDE_area_of_inscribed_square_l1581_158148

/-- A right triangle with an inscribed square -/
structure RightTriangleWithInscribedSquare where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side CD -/
  cd : ℝ
  /-- Side length of the inscribed square BCFE -/
  x : ℝ
  /-- The inscribed square's side is perpendicular to both legs of the right triangle -/
  perpendicular : True
  /-- The inscribed square touches both legs of the right triangle -/
  touches_legs : True

/-- Theorem: Area of inscribed square in right triangle -/
theorem area_of_inscribed_square 
  (triangle : RightTriangleWithInscribedSquare) 
  (h1 : triangle.ab = 36)
  (h2 : triangle.cd = 64) :
  triangle.x^2 = 2304 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribed_square_l1581_158148


namespace NUMINAMATH_CALUDE_ceiling_negative_fraction_squared_l1581_158175

theorem ceiling_negative_fraction_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_fraction_squared_l1581_158175


namespace NUMINAMATH_CALUDE_cylinder_sphere_volume_ratio_l1581_158112

/-- The ratio of the volume of a cylinder inscribed in a sphere to the volume of the sphere,
    where the cylinder's height is 4/3 of the sphere's radius. -/
theorem cylinder_sphere_volume_ratio (R : ℝ) (h : R > 0) :
  let sphere_volume := (4 / 3) * Real.pi * R^3
  let cylinder_height := (4 / 3) * R
  let cylinder_radius := Real.sqrt ((5 / 9) * R^2)
  let cylinder_volume := Real.pi * cylinder_radius^2 * cylinder_height
  cylinder_volume / sphere_volume = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_sphere_volume_ratio_l1581_158112


namespace NUMINAMATH_CALUDE_symmetry_sum_l1581_158181

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

theorem symmetry_sum (m n : ℝ) : 
  symmetric_wrt_origin (m, 1) (-2, n) → m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l1581_158181


namespace NUMINAMATH_CALUDE_f_properties_l1581_158110

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 else x + 6/x - 6

theorem f_properties :
  (f (f (-2)) = -1/2) ∧
  (∀ x, f x ≥ 2 * Real.sqrt 6 - 6) ∧
  (∃ x, f x = 2 * Real.sqrt 6 - 6) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1581_158110


namespace NUMINAMATH_CALUDE_carries_payment_is_94_l1581_158117

/-- The amount Carrie pays for clothes at the mall -/
def carries_payment (num_shirts num_pants num_jackets shirt_cost pant_cost jacket_cost : ℕ) : ℕ :=
  let total_cost := num_shirts * shirt_cost + num_pants * pant_cost + num_jackets * jacket_cost
  total_cost / 2

/-- Theorem: Carrie pays $94 for the clothes -/
theorem carries_payment_is_94 : carries_payment 4 2 2 8 18 60 = 94 := by
  sorry

#eval carries_payment 4 2 2 8 18 60

end NUMINAMATH_CALUDE_carries_payment_is_94_l1581_158117


namespace NUMINAMATH_CALUDE_triangle_ratio_l1581_158194

/-- Given an acute triangle ABC and a point D inside it, 
    if ∠ADB = ∠ACB + 90° and AC · BD = AD · BC, 
    then (AB · CD) / (AC · BD) = √2 -/
theorem triangle_ratio (A B C D : ℝ × ℝ) : 
  let triangle_is_acute : Bool := sorry
  let D_inside_triangle : Bool := sorry
  let angle_ADB : ℝ := sorry
  let angle_ACB : ℝ := sorry
  let AC : ℝ := sorry
  let BD : ℝ := sorry
  let AD : ℝ := sorry
  let BC : ℝ := sorry
  let AB : ℝ := sorry
  let CD : ℝ := sorry
  triangle_is_acute ∧ 
  D_inside_triangle ∧
  angle_ADB = angle_ACB + π/2 ∧ 
  AC * BD = AD * BC →
  (AB * CD) / (AC * BD) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l1581_158194


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l1581_158178

theorem max_product_under_constraint :
  ∀ x y : ℕ, 27 * x + 35 * y ≤ 1000 →
  x * y ≤ 252 ∧ ∃ a b : ℕ, 27 * a + 35 * b ≤ 1000 ∧ a * b = 252 := by
sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l1581_158178


namespace NUMINAMATH_CALUDE_symmetry_condition_l1581_158116

/-- Given a curve y = (ax + b) / (cx - d) where a, b, c, and d are nonzero real numbers,
    if y = x and y = -x are axes of symmetry, then d + b = 0 -/
theorem symmetry_condition (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x : ℝ, x = (a * x + b) / (c * x - d)) →
  (∀ x : ℝ, x = (a * (-x) + b) / (c * (-x) - d)) →
  d + b = 0 := by
  sorry


end NUMINAMATH_CALUDE_symmetry_condition_l1581_158116


namespace NUMINAMATH_CALUDE_supervisors_average_salary_l1581_158129

/-- Given the following conditions in a factory:
  1. The average monthly salary of laborers and supervisors combined is 1250.
  2. There are 6 supervisors.
  3. There are 42 laborers.
  4. The average monthly salary of the laborers is 950.
  Prove that the average monthly salary of the supervisors is 3350. -/
theorem supervisors_average_salary
  (total_average : ℚ)
  (num_supervisors : ℕ)
  (num_laborers : ℕ)
  (laborers_average : ℚ)
  (h1 : total_average = 1250)
  (h2 : num_supervisors = 6)
  (h3 : num_laborers = 42)
  (h4 : laborers_average = 950) :
  (total_average * (num_supervisors + num_laborers) - laborers_average * num_laborers) / num_supervisors = 3350 := by
  sorry


end NUMINAMATH_CALUDE_supervisors_average_salary_l1581_158129


namespace NUMINAMATH_CALUDE_peanuts_added_l1581_158125

theorem peanuts_added (initial_peanuts final_peanuts : ℕ) 
  (h1 : initial_peanuts = 10)
  (h2 : final_peanuts = 18) :
  final_peanuts - initial_peanuts = 8 := by
sorry

end NUMINAMATH_CALUDE_peanuts_added_l1581_158125


namespace NUMINAMATH_CALUDE_production_average_proof_l1581_158192

/-- Calculates the new average daily production after adding a new day's production -/
def newAverageProduction (n : ℕ) (oldAverage : ℚ) (newProduction : ℚ) : ℚ :=
  ((n : ℚ) * oldAverage + newProduction) / ((n : ℚ) + 1)

theorem production_average_proof :
  let n : ℕ := 4
  let oldAverage : ℚ := 50
  let newProduction : ℚ := 90
  newAverageProduction n oldAverage newProduction = 58 := by
sorry

end NUMINAMATH_CALUDE_production_average_proof_l1581_158192


namespace NUMINAMATH_CALUDE_hospital_age_l1581_158199

/-- Proves that the hospital's current age is 40 years, given Grant's current age and the relationship between their ages in 5 years. -/
theorem hospital_age (grant_current_age : ℕ) (hospital_age : ℕ) : 
  grant_current_age = 25 →
  grant_current_age + 5 = 2 / 3 * (hospital_age + 5) →
  hospital_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_hospital_age_l1581_158199


namespace NUMINAMATH_CALUDE_tan_equality_implies_75_l1581_158155

theorem tan_equality_implies_75 (n : ℤ) (h1 : -90 < n) (h2 : n < 90) :
  Real.tan (n • π / 180) = Real.tan (255 • π / 180) → n = 75 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_implies_75_l1581_158155


namespace NUMINAMATH_CALUDE_complex_calculation_l1581_158114

theorem complex_calculation (a b : ℂ) (ha : a = 3 + 2*Complex.I) (hb : b = 2 - 2*Complex.I) :
  3*a - 4*b = 1 + 14*Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_calculation_l1581_158114


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1581_158142

theorem largest_integer_satisfying_inequality :
  ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), 3 * (m^2007 : ℝ) < 3^4015 → m ≤ n) ∧
  (3 * ((n : ℝ)^2007) < 3^4015) :=
sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1581_158142


namespace NUMINAMATH_CALUDE_slope_range_for_intersecting_line_l1581_158102

/-- The range of possible slopes for a line passing through a given point and intersecting a line segment -/
theorem slope_range_for_intersecting_line (M P Q : ℝ × ℝ) :
  M = (-1, 2) →
  P = (-4, -1) →
  Q = (3, 0) →
  let slope_range := {k : ℝ | k ≤ -1/2 ∨ k ≥ 1}
  ∀ k : ℝ,
    (∃ (x y : ℝ), (k * (x - M.1) = y - M.2 ∧
      ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
        x = P.1 + t * (Q.1 - P.1) ∧
        y = P.2 + t * (Q.2 - P.2))) ↔
    k ∈ slope_range :=
by sorry

end NUMINAMATH_CALUDE_slope_range_for_intersecting_line_l1581_158102


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1581_158159

theorem sum_of_reciprocals_of_roots (a b c d : ℝ) (z₁ z₂ z₃ z₄ : ℂ) : 
  z₁^4 + a*z₁^3 + b*z₁^2 + c*z₁ + d = 0 ∧
  z₂^4 + a*z₂^3 + b*z₂^2 + c*z₂ + d = 0 ∧
  z₃^4 + a*z₃^3 + b*z₃^2 + c*z₃ + d = 0 ∧
  z₄^4 + a*z₄^3 + b*z₄^2 + c*z₄ + d = 0 ∧
  Complex.abs z₁ = 1 ∧ Complex.abs z₂ = 1 ∧ Complex.abs z₃ = 1 ∧ Complex.abs z₄ = 1 →
  1/z₁ + 1/z₂ + 1/z₃ + 1/z₄ = -a := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1581_158159


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l1581_158153

theorem quadratic_root_transformation (p q r : ℝ) (u v : ℝ) :
  (p * u^2 + q * u + r = 0) ∧ (p * v^2 + q * v + r = 0) →
  ((q * u + p)^2 - p * (q * u + p) + q * r = 0) ∧ ((q * v + p)^2 - p * (q * v + p) + q * r = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l1581_158153


namespace NUMINAMATH_CALUDE_relay_race_sarah_speed_l1581_158120

/-- Relay race problem -/
theorem relay_race_sarah_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (sadie_speed : ℝ) 
  (sadie_time : ℝ) 
  (ariana_speed : ℝ) 
  (ariana_time : ℝ) 
  (h1 : total_distance = 17) 
  (h2 : total_time = 4.5) 
  (h3 : sadie_speed = 3) 
  (h4 : sadie_time = 2) 
  (h5 : ariana_speed = 6) 
  (h6 : ariana_time = 0.5) : 
  (total_distance - (sadie_speed * sadie_time + ariana_speed * ariana_time)) / 
  (total_time - sadie_time - ariana_time) = 4 := by
  sorry


end NUMINAMATH_CALUDE_relay_race_sarah_speed_l1581_158120


namespace NUMINAMATH_CALUDE_complex_real_condition_l1581_158164

theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := (m + 2*I) / (3 - 4*I)
  (∃ (x : ℝ), z = x) → m = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1581_158164


namespace NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l1581_158150

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle for rotational symmetry in degrees -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_17gon_symmetry_sum :
  ∀ p : RegularPolygon 17,
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 649 / 17 := by
  sorry

end NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l1581_158150


namespace NUMINAMATH_CALUDE_sarah_cans_yesterday_l1581_158139

theorem sarah_cans_yesterday (sarah_yesterday : ℕ) 
  (h1 : sarah_yesterday + (sarah_yesterday + 30) = 40 + 70 + 20) : 
  sarah_yesterday = 50 := by
  sorry

end NUMINAMATH_CALUDE_sarah_cans_yesterday_l1581_158139


namespace NUMINAMATH_CALUDE_trivia_team_members_l1581_158133

theorem trivia_team_members (absent_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  absent_members = 6 → points_per_member = 3 → total_points = 27 → 
  ∃ (total_members : ℕ), total_members = 15 ∧ 
  points_per_member * (total_members - absent_members) = total_points :=
by
  sorry

end NUMINAMATH_CALUDE_trivia_team_members_l1581_158133


namespace NUMINAMATH_CALUDE_abs_fraction_inequality_l1581_158183

theorem abs_fraction_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_abs_fraction_inequality_l1581_158183


namespace NUMINAMATH_CALUDE_product_of_roots_l1581_158174

theorem product_of_roots (x₁ x₂ : ℝ) 
  (h1 : x₁^2 - 2*x₁ = 2) 
  (h2 : x₂^2 - 2*x₂ = 2) 
  (h3 : x₁ ≠ x₂) : 
  x₁ * x₂ = -2 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l1581_158174


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1581_158167

/-- Given a quadratic function f(x) = -x^2 + 4x + a on the interval [0, 1] 
    with a minimum value of -2, prove that its maximum value is 1. -/
theorem quadratic_max_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = -x^2 + 4*x + a) →
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f x ≤ f y) →
  (∃ x ∈ Set.Icc 0 1, f x = -2) →
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f y ≤ f x) →
  (∃ x ∈ Set.Icc 0 1, f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1581_158167


namespace NUMINAMATH_CALUDE_complex_second_quadrant_second_quadrant_implies_neg_real_pos_imag_l1581_158141

/-- A complex number is in the second quadrant if its real part is negative and its imaginary part is positive -/
theorem complex_second_quadrant (z : ℂ) :
  (z.re < 0 ∧ z.im > 0) ↔ (z.arg > Real.pi / 2 ∧ z.arg < Real.pi) :=
by sorry

/-- If a complex number is in the second quadrant, then its real part is negative and its imaginary part is positive -/
theorem second_quadrant_implies_neg_real_pos_imag (z : ℂ) 
  (h : z.arg > Real.pi / 2 ∧ z.arg < Real.pi) : 
  z.re < 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_second_quadrant_second_quadrant_implies_neg_real_pos_imag_l1581_158141


namespace NUMINAMATH_CALUDE_two_thousand_fourteen_between_powers_of_ten_l1581_158169

theorem two_thousand_fourteen_between_powers_of_ten : 10^3 < 2014 ∧ 2014 < 10^4 := by
  sorry

end NUMINAMATH_CALUDE_two_thousand_fourteen_between_powers_of_ten_l1581_158169


namespace NUMINAMATH_CALUDE_logical_equivalence_l1581_158138

theorem logical_equivalence (S X Y : Prop) :
  (S → ¬X ∧ ¬Y) ↔ (X ∨ Y → ¬S) := by
  sorry

end NUMINAMATH_CALUDE_logical_equivalence_l1581_158138


namespace NUMINAMATH_CALUDE_partner_profit_percentage_l1581_158168

theorem partner_profit_percentage (total_profit : ℝ) (majority_owner_percentage : ℝ) 
  (combined_amount : ℝ) (num_partners : ℕ) :
  total_profit = 80000 →
  majority_owner_percentage = 0.25 →
  combined_amount = 50000 →
  num_partners = 4 →
  let remaining_profit := total_profit * (1 - majority_owner_percentage)
  let partner_share := (combined_amount - total_profit * majority_owner_percentage) / 2
  (partner_share / remaining_profit) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_partner_profit_percentage_l1581_158168


namespace NUMINAMATH_CALUDE_square_minus_two_x_plus_one_l1581_158191

theorem square_minus_two_x_plus_one (x : ℝ) : x = Real.sqrt 3 + 1 → x^2 - 2*x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_two_x_plus_one_l1581_158191


namespace NUMINAMATH_CALUDE_mango_distribution_l1581_158177

/-- Given 560 mangoes, if half are sold and the remainder is distributed evenly among 8 neighbors,
    each neighbor receives 35 mangoes. -/
theorem mango_distribution (total_mangoes : ℕ) (neighbors : ℕ) 
    (h1 : total_mangoes = 560) 
    (h2 : neighbors = 8) : 
  (total_mangoes / 2) / neighbors = 35 := by
  sorry

end NUMINAMATH_CALUDE_mango_distribution_l1581_158177


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l1581_158197

/-- Given vectors a and b in ℝ², if a + b is perpendicular to b, then the second component of a is 9. -/
theorem vector_perpendicular_condition (m : ℝ) : 
  let a : ℝ × ℝ := (5, m)
  let b : ℝ × ℝ := (2, -2)
  (a.1 + b.1, a.2 + b.2) • b = 0 → m = 9 := by
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l1581_158197


namespace NUMINAMATH_CALUDE_adjacent_sum_negative_total_sum_positive_l1581_158165

theorem adjacent_sum_negative_total_sum_positive :
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℝ),
    (a₁ + a₂ < 0) ∧
    (a₂ + a₃ < 0) ∧
    (a₃ + a₄ < 0) ∧
    (a₄ + a₅ < 0) ∧
    (a₅ + a₁ < 0) ∧
    (a₁ + a₂ + a₃ + a₄ + a₅ > 0) :=
  sorry

end NUMINAMATH_CALUDE_adjacent_sum_negative_total_sum_positive_l1581_158165


namespace NUMINAMATH_CALUDE_sock_pair_count_l1581_158111

/-- The number of ways to choose a pair of socks of different colors -/
def different_color_sock_pairs (black white blue : ℕ) : ℕ :=
  black * white + black * blue + white * blue

/-- Theorem: There are 107 ways to choose a pair of socks of different colors
    from a drawer containing 5 black socks, 6 white socks, and 7 blue socks -/
theorem sock_pair_count :
  different_color_sock_pairs 5 6 7 = 107 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l1581_158111


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l1581_158135

theorem rectangular_plot_length_difference (breadth : ℝ) (x : ℝ) : 
  breadth > 0 →
  x > 0 →
  breadth + x = 60 →
  4 * breadth + 2 * x = 200 →
  x = 20 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l1581_158135


namespace NUMINAMATH_CALUDE_fraction_power_equality_l1581_158196

theorem fraction_power_equality (x y : ℚ) 
  (hx : x = 5/6) (hy : y = 6/5) : 
  (1/3 : ℚ) * x^7 * y^8 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l1581_158196


namespace NUMINAMATH_CALUDE_initial_cash_calculation_l1581_158115

-- Define the initial cash as a real number
variable (X : ℝ)

-- Define the constants from the problem
def raw_materials : ℝ := 500
def machinery : ℝ := 400
def sales_tax : ℝ := 0.05
def exchange_rate : ℝ := 1.2
def labor_cost_rate : ℝ := 0.1
def inflation_rate : ℝ := 0.02
def years : ℕ := 2
def remaining_amount : ℝ := 900

-- State the theorem
theorem initial_cash_calculation :
  remaining_amount = (X - ((1 + sales_tax) * (raw_materials + machinery) * exchange_rate + labor_cost_rate * X)) / (1 + inflation_rate) ^ years :=
by sorry

end NUMINAMATH_CALUDE_initial_cash_calculation_l1581_158115


namespace NUMINAMATH_CALUDE_a_plus_b_equals_one_l1581_158131

-- Define the universe U as the real numbers
def U : Type := ℝ

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | (x^2 + a*x + b)*(x - 1) = 0}

-- Define the theorem
theorem a_plus_b_equals_one (a b : ℝ) (B : Set ℝ) :
  (∃ (B : Set ℝ), (A a b ∩ B = {1, 2}) ∧ (A a b ∩ (Set.univ \ B) = {3})) →
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_one_l1581_158131


namespace NUMINAMATH_CALUDE_square_difference_identity_l1581_158104

theorem square_difference_identity : (50 + 12)^2 - (12^2 + 50^2) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l1581_158104


namespace NUMINAMATH_CALUDE_kitty_window_cleaning_time_l1581_158176

/-- Represents the weekly cleaning time for various tasks in minutes -/
structure CleaningTime where
  pickup : ℕ
  vacuum : ℕ
  dust : ℕ
  window : ℕ

/-- Calculates the total cleaning time for a given number of weeks -/
def totalCleaningTime (ct : CleaningTime) (weeks : ℕ) : ℕ :=
  (ct.pickup + ct.vacuum + ct.dust + ct.window) * weeks

/-- The main theorem about Kitty's cleaning time -/
theorem kitty_window_cleaning_time :
  ∀ (ct : CleaningTime),
    ct.pickup = 5 →
    ct.vacuum = 20 →
    ct.dust = 10 →
    totalCleaningTime ct 4 = 200 →
    ct.window = 15 :=
by sorry

end NUMINAMATH_CALUDE_kitty_window_cleaning_time_l1581_158176


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1581_158106

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  (1 - 3 * (x - 1) < 8 - x) ∧ ((x - 3) / 2 + 3 ≥ x)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  -2 < x ∧ x ≤ 3

-- Theorem statement
theorem inequality_system_solution :
  ∀ x : ℝ, inequality_system x ↔ solution_set x := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1581_158106


namespace NUMINAMATH_CALUDE_total_distance_two_wheels_l1581_158109

/-- The total distance covered by two wheels with different radii -/
theorem total_distance_two_wheels 
  (r1 r2 N : ℝ) 
  (h_positive : r1 > 0 ∧ r2 > 0 ∧ N > 0) : 
  let wheel1_revolutions : ℝ := 1500
  let wheel2_revolutions : ℝ := N * wheel1_revolutions
  let distance_wheel1 : ℝ := 2 * Real.pi * r1 * wheel1_revolutions
  let distance_wheel2 : ℝ := 2 * Real.pi * r2 * wheel2_revolutions
  let total_distance : ℝ := distance_wheel1 + distance_wheel2
  total_distance = 3000 * Real.pi * (r1 + N * r2) :=
by sorry

end NUMINAMATH_CALUDE_total_distance_two_wheels_l1581_158109


namespace NUMINAMATH_CALUDE_arc_length_calculation_l1581_158198

theorem arc_length_calculation (r θ : Real) (h1 : r = 2) (h2 : θ = π/3) :
  r * θ = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_calculation_l1581_158198


namespace NUMINAMATH_CALUDE_kelly_snacks_weight_l1581_158118

theorem kelly_snacks_weight (peanuts raisins total : Real) : 
  peanuts = 0.1 → raisins = 0.4 → total = peanuts + raisins → total = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_kelly_snacks_weight_l1581_158118


namespace NUMINAMATH_CALUDE_test_questions_count_l1581_158163

theorem test_questions_count : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (5 * n = 45) ∧ 
  (32 > 0.70 * 45) ∧ 
  (32 < 0.77 * 45) := by
  sorry

end NUMINAMATH_CALUDE_test_questions_count_l1581_158163


namespace NUMINAMATH_CALUDE_max_contribution_l1581_158187

theorem max_contribution (n : ℕ) (total : ℝ) (min_contrib : ℝ) (h1 : n = 15) (h2 : total = 30) (h3 : min_contrib = 1) :
  let max_single := total - (n - 1) * min_contrib
  max_single = 16 := by
sorry

end NUMINAMATH_CALUDE_max_contribution_l1581_158187


namespace NUMINAMATH_CALUDE_school_female_students_l1581_158182

theorem school_female_students 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (sample_difference : ℕ) : 
  total_students = 1600 → 
  sample_size = 200 → 
  sample_difference = 10 →
  (∃ (female_students : ℕ), 
    female_students = 760 ∧ 
    female_students + (total_students - female_students) = total_students ∧
    (female_students : ℚ) / (total_students - female_students) = 
      ((sample_size / 2 - sample_difference / 2) : ℚ) / (sample_size / 2 + sample_difference / 2)) :=
by sorry

end NUMINAMATH_CALUDE_school_female_students_l1581_158182
