import Mathlib

namespace NUMINAMATH_CALUDE_ratio_equality_l1141_114160

theorem ratio_equality (a b : ℝ) (h1 : 3 * a = 4 * b) (h2 : a * b ≠ 0) :
  (a / 4) / (b / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1141_114160


namespace NUMINAMATH_CALUDE_student_d_not_top_student_l1141_114135

/-- Represents a student's rankings in three consecutive exams -/
structure StudentRankings :=
  (r1 r2 r3 : ℕ)

/-- Calculates the mode of three numbers -/
def mode (a b c : ℕ) : ℕ := sorry

/-- Calculates the variance of three numbers -/
def variance (a b c : ℕ) : ℚ := sorry

/-- Determines if a student is a top student based on their rankings -/
def is_top_student (s : StudentRankings) : Prop :=
  s.r1 ≤ 3 ∧ s.r2 ≤ 3 ∧ s.r3 ≤ 3

theorem student_d_not_top_student (s : StudentRankings) :
  mode s.r1 s.r2 s.r3 = 2 ∧ variance s.r1 s.r2 s.r3 > 1 →
  ¬(is_top_student s) := by sorry

end NUMINAMATH_CALUDE_student_d_not_top_student_l1141_114135


namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l1141_114186

theorem inverse_proportion_inequality (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁ > x₂ → x₂ > 0 → y₁ = -3 / x₁ → y₂ = -3 / x₂ → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l1141_114186


namespace NUMINAMATH_CALUDE_local_face_value_diff_problem_l1141_114167

/-- The difference between the local value and face value of a digit in a number -/
def local_face_value_diff (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (10 ^ position) - digit

theorem local_face_value_diff_problem : 
  local_face_value_diff 3 3 - local_face_value_diff 7 1 = 2934 := by
  sorry

end NUMINAMATH_CALUDE_local_face_value_diff_problem_l1141_114167


namespace NUMINAMATH_CALUDE_decimal_to_binary_38_l1141_114115

theorem decimal_to_binary_38 : 
  (38 : ℕ).digits 2 = [0, 1, 1, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_decimal_to_binary_38_l1141_114115


namespace NUMINAMATH_CALUDE_square_side_length_l1141_114126

theorem square_side_length (side : ℝ) : 
  (5 * side) * (side / 2) = 160 → side = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1141_114126


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1141_114112

theorem sqrt_x_minus_one_real (x : ℝ) : x ≥ 1 ↔ ∃ y : ℝ, y ^ 2 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1141_114112


namespace NUMINAMATH_CALUDE_nancy_total_games_l1141_114183

/-- The total number of games Nancy will attend over three months -/
def total_games (this_month : ℕ) (last_month : ℕ) (next_month : ℕ) : ℕ :=
  this_month + last_month + next_month

/-- Theorem stating that Nancy will attend 24 games in total -/
theorem nancy_total_games : 
  total_games 9 8 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_nancy_total_games_l1141_114183


namespace NUMINAMATH_CALUDE_perimeter_plus_area_sum_l1141_114124

/-- A parallelogram with integer coordinates -/
structure IntegerParallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ
  is_parallelogram : v1.1 + v3.1 = v2.1 + v4.1 ∧ v1.2 + v3.2 = v2.2 + v4.2

/-- Calculate the perimeter of a parallelogram -/
def perimeter (p : IntegerParallelogram) : ℝ :=
  2 * (dist p.v1 p.v2 + dist p.v2 p.v3)

/-- Calculate the area of a parallelogram -/
def area (p : IntegerParallelogram) : ℝ :=
  abs ((p.v2.1 - p.v1.1) * (p.v3.2 - p.v1.2) - (p.v3.1 - p.v1.1) * (p.v2.2 - p.v1.2))

/-- The sum of perimeter and area for the specific parallelogram -/
theorem perimeter_plus_area_sum (p : IntegerParallelogram) 
  (h1 : p.v1 = (2, 3)) 
  (h2 : p.v2 = (5, 7)) 
  (h3 : p.v3 = (0, -1)) : 
  perimeter p + area p = 10 + 12 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_plus_area_sum_l1141_114124


namespace NUMINAMATH_CALUDE_gorilla_exhibit_percentage_is_80_l1141_114157

-- Define the given parameters
def visitors_per_hour : ℕ := 50
def open_hours : ℕ := 8
def gorilla_exhibit_visitors : ℕ := 320

-- Define the total number of visitors
def total_visitors : ℕ := visitors_per_hour * open_hours

-- Define the percentage of visitors going to the gorilla exhibit
def gorilla_exhibit_percentage : ℚ := (gorilla_exhibit_visitors : ℚ) / (total_visitors : ℚ) * 100

-- Theorem statement
theorem gorilla_exhibit_percentage_is_80 : 
  gorilla_exhibit_percentage = 80 := by sorry

end NUMINAMATH_CALUDE_gorilla_exhibit_percentage_is_80_l1141_114157


namespace NUMINAMATH_CALUDE_point_symmetric_second_quadrant_l1141_114130

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Checks if a point is symmetric about the y-axis -/
def isSymmetricAboutYAxis (p : Point) : Prop :=
  p.x < 0

/-- The main theorem -/
theorem point_symmetric_second_quadrant (a : ℝ) :
  let A : Point := ⟨a - 1, 2 * a - 4⟩
  isInSecondQuadrant A ∧ isSymmetricAboutYAxis A → a > 2 :=
by sorry

end NUMINAMATH_CALUDE_point_symmetric_second_quadrant_l1141_114130


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1141_114171

/-- Triangle EFG with inscribed rectangle ABCD -/
structure InscribedRectangle where
  /-- Length of side EG of triangle EFG -/
  eg : ℝ
  /-- Height of altitude from F to EG -/
  altitude : ℝ
  /-- Length of side AD of rectangle ABCD -/
  ad : ℝ
  /-- Length of side AB of rectangle ABCD -/
  ab : ℝ
  /-- AD is on EG -/
  ad_on_eg : ad ≤ eg
  /-- AB is one-third of AD -/
  ab_is_third_of_ad : ab = ad / 3
  /-- EG is 15 inches -/
  eg_length : eg = 15
  /-- Altitude is 10 inches -/
  altitude_length : altitude = 10

/-- The area of the inscribed rectangle ABCD is 100/3 square inches -/
theorem inscribed_rectangle_area (rect : InscribedRectangle) : 
  rect.ad * rect.ab = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1141_114171


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1141_114123

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1141_114123


namespace NUMINAMATH_CALUDE_total_money_l1141_114104

/-- Given three people A, B, and C with some money, prove their total amount. -/
theorem total_money (A B C : ℕ) 
  (hAC : A + C = 250)
  (hBC : B + C = 450)
  (hC : C = 100) : 
  A + B + C = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l1141_114104


namespace NUMINAMATH_CALUDE_tiles_per_row_l1141_114172

-- Define the area of the room in square feet
def room_area : ℝ := 324

-- Define the side length of a tile in inches
def tile_side : ℝ := 9

-- Define the conversion factor from feet to inches
def feet_to_inches : ℝ := 12

-- Theorem statement
theorem tiles_per_row : 
  ⌊(feet_to_inches * Real.sqrt room_area) / tile_side⌋ = 24 := by
  sorry

end NUMINAMATH_CALUDE_tiles_per_row_l1141_114172


namespace NUMINAMATH_CALUDE_odd_sum_count_l1141_114138

def card_set : Finset ℕ := {1, 2, 3, 4}

def is_sum_odd (pair : ℕ × ℕ) : Bool :=
  (pair.1 + pair.2) % 2 = 1

def odd_sum_pairs : Finset (ℕ × ℕ) :=
  (card_set.product card_set).filter (λ pair => pair.1 < pair.2 ∧ is_sum_odd pair)

theorem odd_sum_count : odd_sum_pairs.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_count_l1141_114138


namespace NUMINAMATH_CALUDE_books_left_to_read_l1141_114142

theorem books_left_to_read (total_books assigned_books : ℕ) 
  (mcgregor_finished floyd_finished : ℕ) 
  (h1 : assigned_books = 89)
  (h2 : mcgregor_finished = 34)
  (h3 : floyd_finished = 32)
  (h4 : total_books = assigned_books - (mcgregor_finished + floyd_finished)) :
  total_books = 23 := by
  sorry

end NUMINAMATH_CALUDE_books_left_to_read_l1141_114142


namespace NUMINAMATH_CALUDE_sum_of_other_x_coordinates_l1141_114178

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- The property that two points are opposite vertices of a rectangle --/
def are_opposite_vertices (p1 p2 : ℝ × ℝ) (r : Rectangle) : Prop :=
  (r.v1 = p1 ∧ r.v3 = p2) ∨ (r.v1 = p2 ∧ r.v3 = p1) ∨
  (r.v2 = p1 ∧ r.v4 = p2) ∨ (r.v2 = p2 ∧ r.v4 = p1)

/-- The theorem to be proved --/
theorem sum_of_other_x_coordinates (r : Rectangle) :
  are_opposite_vertices (2, 12) (8, 3) r →
  (r.v1.1 + r.v2.1 + r.v3.1 + r.v4.1) - (2 + 8) = 10 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_other_x_coordinates_l1141_114178


namespace NUMINAMATH_CALUDE_jelly_beans_distribution_l1141_114103

theorem jelly_beans_distribution (initial_beans : ℕ) (remaining_beans : ℕ) 
  (h1 : initial_beans = 8000)
  (h2 : remaining_beans = 1600) :
  ∃ (x : ℕ), 
    x = 400 ∧ 
    initial_beans - remaining_beans = 6 * (2 * x) + 4 * x :=
by sorry

end NUMINAMATH_CALUDE_jelly_beans_distribution_l1141_114103


namespace NUMINAMATH_CALUDE_genevieve_thermoses_l1141_114174

/-- Proves the number of thermoses Genevieve drank given the conditions -/
theorem genevieve_thermoses (total_coffee : ℚ) (num_thermoses : ℕ) (genevieve_consumption : ℚ) : 
  total_coffee = 4.5 ∧ num_thermoses = 18 ∧ genevieve_consumption = 6 →
  (genevieve_consumption / (total_coffee * 8 / num_thermoses) : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_genevieve_thermoses_l1141_114174


namespace NUMINAMATH_CALUDE_bathtub_fill_time_with_open_drain_l1141_114187

/-- Represents the time it takes to fill a bathtub with the drain open. -/
def fill_time_with_open_drain (fill_time drain_time : ℚ) : ℚ :=
  (fill_time * drain_time) / (drain_time - fill_time)

/-- Theorem stating that a bathtub taking 10 minutes to fill and 12 minutes to drain
    will take 60 minutes to fill with the drain open. -/
theorem bathtub_fill_time_with_open_drain :
  fill_time_with_open_drain 10 12 = 60 := by
  sorry

#eval fill_time_with_open_drain 10 12

end NUMINAMATH_CALUDE_bathtub_fill_time_with_open_drain_l1141_114187


namespace NUMINAMATH_CALUDE_range_of_a_l1141_114177

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x > -1, x^2 / (x + 1) ≥ a

def q (a : ℝ) : Prop := ∃ x : ℝ, a * x^2 - a * x + 1 = 0

-- State the theorem
theorem range_of_a :
  ∃ a : ℝ, (¬(p a) ∧ ¬(q a)) ∧ (p a ∨ q a) ↔ (a = 0 ∨ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1141_114177


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1141_114173

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 560) 
  (h2 : a*b + b*c + c*a = 8) : 
  a + b + c = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1141_114173


namespace NUMINAMATH_CALUDE_paperClips_in_two_cases_l1141_114162

/-- The number of paper clips in 2 cases -/
def paperClipsIn2Cases (c b : ℕ) : ℕ := 2 * c * b * 300

/-- Theorem: The number of paper clips in 2 cases is 2c * b * 300 -/
theorem paperClips_in_two_cases (c b : ℕ) : 
  paperClipsIn2Cases c b = 2 * c * b * 300 := by
  sorry

end NUMINAMATH_CALUDE_paperClips_in_two_cases_l1141_114162


namespace NUMINAMATH_CALUDE_rectangle_perimeter_equals_20_l1141_114156

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle with width w and length l -/
structure Rectangle where
  w : ℝ
  l : ℝ

/-- Area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := 
  sorry

/-- Area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ :=
  r.w * r.l

/-- Perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.w + r.l)

theorem rectangle_perimeter_equals_20 (t : Triangle) (r : Rectangle) :
  t.a = 6 ∧ t.b = 8 ∧ t.c = 10 ∧ r.w = 4 ∧ Triangle.area t = Rectangle.area r →
  Rectangle.perimeter r = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_equals_20_l1141_114156


namespace NUMINAMATH_CALUDE_percentage_of_1000_l1141_114176

theorem percentage_of_1000 (x : ℝ) (h : x = 66.2) : 
  (x / 1000) * 100 = 6.62 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_1000_l1141_114176


namespace NUMINAMATH_CALUDE_tangent_identity_l1141_114144

theorem tangent_identity (β : ℝ) : 
  Real.tan (6 * β) - Real.tan (4 * β) - Real.tan (2 * β) = 
  Real.tan (6 * β) * Real.tan (4 * β) * Real.tan (2 * β) := by
  sorry

end NUMINAMATH_CALUDE_tangent_identity_l1141_114144


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l1141_114185

/-- Theorem: Theater Ticket Sales --/
theorem theater_ticket_sales
  (orchestra_price : ℕ)
  (balcony_price : ℕ)
  (total_revenue : ℕ)
  (balcony_excess : ℕ)
  (h1 : orchestra_price = 12)
  (h2 : balcony_price = 8)
  (h3 : total_revenue = 3320)
  (h4 : balcony_excess = 190)
  : ∃ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets * orchestra_price + balcony_tickets * balcony_price = total_revenue ∧
    balcony_tickets = orchestra_tickets + balcony_excess ∧
    orchestra_tickets + balcony_tickets = 370 :=
by
  sorry


end NUMINAMATH_CALUDE_theater_ticket_sales_l1141_114185


namespace NUMINAMATH_CALUDE_fraction_problem_l1141_114149

theorem fraction_problem (f : ℚ) : f * 76 = 76 - 19 → f = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1141_114149


namespace NUMINAMATH_CALUDE_divisibility_by_five_l1141_114194

theorem divisibility_by_five (x y : ℕ+) (h1 : 2 * x ^ 2 - 1 = y ^ 15) (h2 : x > 1) :
  5 ∣ x.val :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l1141_114194


namespace NUMINAMATH_CALUDE_tan_105_degrees_l1141_114190

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l1141_114190


namespace NUMINAMATH_CALUDE_base_8_23456_equals_10030_l1141_114152

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ (digits.length - 1 - i))) 0

theorem base_8_23456_equals_10030 :
  base_8_to_10 [2, 3, 4, 5, 6] = 10030 := by
  sorry

end NUMINAMATH_CALUDE_base_8_23456_equals_10030_l1141_114152


namespace NUMINAMATH_CALUDE_digits_of_power_product_l1141_114196

theorem digits_of_power_product : 
  (Nat.log 10 (2^15 * 5^10) + 1 : ℕ) = 12 := by sorry

end NUMINAMATH_CALUDE_digits_of_power_product_l1141_114196


namespace NUMINAMATH_CALUDE_worker_speed_comparison_l1141_114195

/-- Given two workers A and B, this theorem proves that A is 3 times faster than B
    under the specified conditions. -/
theorem worker_speed_comparison 
  (work_rate_A : ℝ) 
  (work_rate_B : ℝ) 
  (total_work : ℝ) 
  (h1 : work_rate_A + work_rate_B = total_work / 24)
  (h2 : work_rate_A = total_work / 32) :
  work_rate_A = 3 * work_rate_B :=
sorry

end NUMINAMATH_CALUDE_worker_speed_comparison_l1141_114195


namespace NUMINAMATH_CALUDE_octopus_ink_conversion_l1141_114129

/-- Converts a three-digit number from base 8 to base 10 -/
def base8ToBase10 (hundreds : Nat) (tens : Nat) (units : Nat) : Nat :=
  hundreds * 8^2 + tens * 8^1 + units * 8^0

/-- The octopus ink problem -/
theorem octopus_ink_conversion :
  base8ToBase10 2 7 6 = 190 := by
  sorry

end NUMINAMATH_CALUDE_octopus_ink_conversion_l1141_114129


namespace NUMINAMATH_CALUDE_midpoint_product_zero_l1141_114128

/-- Given that C = (4, 3) is the midpoint of line segment AB where A = (2, 6) and B = (x, y), prove that xy = 0 -/
theorem midpoint_product_zero (x y : ℝ) : 
  (4 : ℝ) = (2 + x) / 2 → 
  (3 : ℝ) = (6 + y) / 2 → 
  x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_product_zero_l1141_114128


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l1141_114100

theorem quadratic_inequality_empty_solution (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + a * x + 2 ≥ 0) → -4 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l1141_114100


namespace NUMINAMATH_CALUDE_triangular_prism_has_nine_edges_l1141_114125

/-- The number of sides in the base polygon of a triangular prism -/
def triangular_prism_base_sides : ℕ := 3

/-- The number of edges in a prism given the number of sides in its base polygon -/
def prism_edges (n : ℕ) : ℕ := 3 * n

/-- Theorem: A triangular prism has 9 edges -/
theorem triangular_prism_has_nine_edges :
  prism_edges triangular_prism_base_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_has_nine_edges_l1141_114125


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l1141_114179

theorem quadratic_roots_product (c d : ℝ) : 
  (3 * c^2 + 9 * c - 21 = 0) → 
  (3 * d^2 + 9 * d - 21 = 0) → 
  (3 * c - 4) * (6 * d - 8) = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l1141_114179


namespace NUMINAMATH_CALUDE_quadratic_behavior_l1141_114193

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 6*x - 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -2*x + 6

-- Theorem statement
theorem quadratic_behavior (x : ℝ) : x > 5 → f x < 0 ∧ f' x < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_behavior_l1141_114193


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1141_114199

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the set of four coins -/
structure FourCoins :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)

/-- The total number of possible outcomes when flipping four coins -/
def totalOutcomes : ℕ := 16

/-- The number of favorable outcomes (penny heads, nickel heads, dime tails) -/
def favorableOutcomes : ℕ := 2

/-- The probability of the desired outcome -/
def desiredProbability : ℚ := 1 / 8

theorem coin_flip_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = desiredProbability := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1141_114199


namespace NUMINAMATH_CALUDE_pencil_distribution_l1141_114184

theorem pencil_distribution (total_pencils : ℕ) (num_friends : ℕ) (pencils_per_friend : ℕ) :
  total_pencils = 24 →
  num_friends = 3 →
  total_pencils = num_friends * pencils_per_friend →
  pencils_per_friend = 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1141_114184


namespace NUMINAMATH_CALUDE_cost_per_minute_is_twelve_cents_l1141_114110

/-- Calculates the cost per minute for a phone service -/
def costPerMinute (monthlyFee : ℚ) (totalBill : ℚ) (minutesUsed : ℕ) : ℚ :=
  (totalBill - monthlyFee) / minutesUsed

/-- Proof that the cost per minute is $0.12 given the specified conditions -/
theorem cost_per_minute_is_twelve_cents :
  let monthlyFee : ℚ := 2
  let totalBill : ℚ := 23.36
  let minutesUsed : ℕ := 178
  costPerMinute monthlyFee totalBill minutesUsed = 0.12 := by
  sorry

#eval costPerMinute 2 23.36 178

end NUMINAMATH_CALUDE_cost_per_minute_is_twelve_cents_l1141_114110


namespace NUMINAMATH_CALUDE_prob_both_selected_l1141_114136

theorem prob_both_selected (prob_x prob_y prob_both : ℚ) : 
  prob_x = 1/5 → prob_y = 2/7 → prob_both = prob_x * prob_y → prob_both = 2/35 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_selected_l1141_114136


namespace NUMINAMATH_CALUDE_distinct_sentences_count_l1141_114181

/-- Represents the number of variations for each phrase -/
def phrase_variations : Fin 4 → ℕ
  | 0 => 3  -- Phrase I
  | 1 => 2  -- Phrase II
  | 2 => 1  -- Phrase III (mandatory)
  | 3 => 2  -- Phrase IV

/-- Calculates the total number of combinations -/
def total_combinations : ℕ := 
  (phrase_variations 0) * (phrase_variations 1) * (phrase_variations 2) * (phrase_variations 3)

/-- The number of distinct meaningful sentences -/
def distinct_sentences : ℕ := total_combinations - 1

theorem distinct_sentences_count : distinct_sentences = 23 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sentences_count_l1141_114181


namespace NUMINAMATH_CALUDE_count_valid_sums_l1141_114114

/-- The number of valid ways to sum to 5750 using 5's, 55's, and 555's -/
def num_valid_sums : ℕ := 124

/-- Predicate for a valid sum configuration -/
def is_valid_sum (a b c : ℕ) : Prop :=
  a + 11 * b + 111 * c = 1150

/-- The length of the original string of 5's -/
def string_length (a b c : ℕ) : ℕ :=
  a + 2 * b + 3 * c

/-- Theorem stating that there are exactly 124 valid string lengths -/
theorem count_valid_sums :
  (∃ (S : Finset ℕ), S.card = num_valid_sums ∧
    (∀ n, n ∈ S ↔ ∃ a b c, is_valid_sum a b c ∧ string_length a b c = n)) :=
sorry

end NUMINAMATH_CALUDE_count_valid_sums_l1141_114114


namespace NUMINAMATH_CALUDE_repeating_decimal_problem_l1141_114113

/-- Represents a repeating decimal with a single digit followed by 25 -/
def RepeatingDecimal (d : Nat) : ℚ :=
  (d * 100 + 25 : ℚ) / 999

/-- The main theorem -/
theorem repeating_decimal_problem (n : ℕ) (d : Nat) 
    (h_n_pos : n > 0)
    (h_d_digit : d < 10)
    (h_eq : (n : ℚ) / 810 = RepeatingDecimal d) :
    n = 750 ∧ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_problem_l1141_114113


namespace NUMINAMATH_CALUDE_age_ratio_l1141_114137

/-- Represents the ages of Albert, Mary, and Betty -/
structure Ages where
  albert : ℕ
  mary : ℕ
  betty : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.albert = 4 * ages.betty ∧
  ages.mary = ages.albert - 10 ∧
  ages.betty = 5

/-- The theorem to prove -/
theorem age_ratio (ages : Ages) 
  (h : satisfiesConditions ages) : 
  ages.albert / ages.mary = 2 := by
sorry


end NUMINAMATH_CALUDE_age_ratio_l1141_114137


namespace NUMINAMATH_CALUDE_regions_in_circle_l1141_114165

/-- The number of regions created by radii and concentric circles in a circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem stating that 16 radii and 10 concentric circles create 176 regions -/
theorem regions_in_circle (r : ℕ) (c : ℕ) 
  (h1 : r = 16) (h2 : c = 10) : 
  num_regions r c = 176 := by
  sorry

end NUMINAMATH_CALUDE_regions_in_circle_l1141_114165


namespace NUMINAMATH_CALUDE_cube_surface_area_from_diagonal_l1141_114161

/-- The surface area of a cube with space diagonal length 6 is 72 -/
theorem cube_surface_area_from_diagonal (d : ℝ) (h : d = 6) : 
  6 * (d / Real.sqrt 3) ^ 2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_diagonal_l1141_114161


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1141_114191

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  a 1 = 1 →
  (∀ n, S n = (a 1 - a (n + 1) * (a 2 / a 1)^n) / (1 - a 2 / a 1)) →
  1 / a 1 - 1 / a 2 = 2 / a 3 →
  S 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1141_114191


namespace NUMINAMATH_CALUDE_soap_brand_usage_l1141_114158

theorem soap_brand_usage (total : ℕ) (neither : ℕ) (only_e : ℕ) (both : ℕ) :
  total = 200 →
  neither = 80 →
  only_e = 60 →
  total = neither + only_e + both + 3 * both →
  both = 15 := by
sorry

end NUMINAMATH_CALUDE_soap_brand_usage_l1141_114158


namespace NUMINAMATH_CALUDE_transcendental_equation_solution_l1141_114148

-- Define the variables
variable (n : ℝ)
variable (x : ℝ)
variable (y : ℝ)

-- State the theorem
theorem transcendental_equation_solution (hx : x = 3) (hy : y = 27) 
  (h : Real.exp (n / (2 * Real.sqrt (Real.pi + x))) = y) :
  n ^ (n / (2 * Real.sqrt (Real.pi + 3))) = Real.exp 27 := by
  sorry


end NUMINAMATH_CALUDE_transcendental_equation_solution_l1141_114148


namespace NUMINAMATH_CALUDE_polygon_is_decagon_iff_seven_diagonals_l1141_114170

/-- A polygon is a decagon if and only if 7 diagonals can be drawn from a single vertex. -/
theorem polygon_is_decagon_iff_seven_diagonals (n : ℕ) : 
  n = 10 ↔ n - 3 = 7 :=
sorry

end NUMINAMATH_CALUDE_polygon_is_decagon_iff_seven_diagonals_l1141_114170


namespace NUMINAMATH_CALUDE_train_crossing_time_l1141_114147

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 240 →
  train_speed_kmh = 216 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1141_114147


namespace NUMINAMATH_CALUDE_two_times_three_plus_two_l1141_114189

theorem two_times_three_plus_two :
  (2 : ℕ) * 3 + 2 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_two_times_three_plus_two_l1141_114189


namespace NUMINAMATH_CALUDE_even_sum_probability_l1141_114134

/-- Represents a wheel with sections --/
structure Wheel where
  totalSections : Nat
  evenSections : Nat
  oddSections : Nat
  zeroSections : Nat

/-- First wheel configuration --/
def wheel1 : Wheel := {
  totalSections := 6
  evenSections := 2
  oddSections := 3
  zeroSections := 1
}

/-- Second wheel configuration --/
def wheel2 : Wheel := {
  totalSections := 4
  evenSections := 2
  oddSections := 2
  zeroSections := 0
}

/-- Calculate the probability of getting an even sum when spinning two wheels --/
def probabilityEvenSum (w1 w2 : Wheel) : Real :=
  sorry

/-- Theorem: The probability of getting an even sum when spinning the two given wheels is 1/2 --/
theorem even_sum_probability :
  probabilityEvenSum wheel1 wheel2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_probability_l1141_114134


namespace NUMINAMATH_CALUDE_pentagon_area_in_16_sided_polygon_l1141_114151

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sideLength : ℝ

/-- Represents a pentagon in a regular polygon -/
structure Pentagon (n : ℕ) where
  polygon : RegularPolygon n
  vertices : Fin 5 → Fin n

/-- Calculates the area of a pentagon in a regular polygon -/
def pentagonArea (n : ℕ) (p : Pentagon n) : ℝ := sorry

theorem pentagon_area_in_16_sided_polygon :
  ∀ (p : Pentagon 16),
    p.polygon.sideLength = 3 →
    (∀ i : Fin 5, (p.vertices i + 4) % 16 = p.vertices ((i + 1) % 5)) →
    pentagonArea 16 p = 198 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_in_16_sided_polygon_l1141_114151


namespace NUMINAMATH_CALUDE_juanico_age_30_years_from_now_l1141_114197

/-- Juanico's age 30 years from now, given the conditions in the problem -/
def juanico_future_age (gladys_current_age : ℕ) (juanico_current_age : ℕ) : ℕ :=
  juanico_current_age + 30

/-- The theorem stating Juanico's age 30 years from now -/
theorem juanico_age_30_years_from_now :
  ∀ (gladys_current_age : ℕ) (juanico_current_age : ℕ),
    gladys_current_age + 10 = 40 →
    juanico_current_age = gladys_current_age / 2 - 4 →
    juanico_future_age gladys_current_age juanico_current_age = 41 :=
by
  sorry

#check juanico_age_30_years_from_now

end NUMINAMATH_CALUDE_juanico_age_30_years_from_now_l1141_114197


namespace NUMINAMATH_CALUDE_vector_linear_combination_l1141_114168

/-- Given vectors a, b, and c in R², prove that c can be expressed as a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) (hb : b = (1, -1)) (hc : c = (-1, 2)) :
  c = (1/2 : ℝ) • a - (3/2 : ℝ) • b :=
sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l1141_114168


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1141_114143

theorem no_positive_integer_solutions :
  ¬∃ (a b : ℕ+), 4 * (a^2 + a) = b^2 + b := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1141_114143


namespace NUMINAMATH_CALUDE_sector_arc_length_l1141_114150

/-- The length of an arc in a circular sector -/
def arcLength (radius : ℝ) (centralAngle : ℝ) : ℝ := radius * centralAngle

theorem sector_arc_length :
  let radius : ℝ := 16
  let centralAngle : ℝ := 2
  arcLength radius centralAngle = 32 := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l1141_114150


namespace NUMINAMATH_CALUDE_books_sold_l1141_114111

theorem books_sold (initial_books remaining_books : ℕ) 
  (h1 : initial_books = 136) 
  (h2 : remaining_books = 27) : 
  initial_books - remaining_books = 109 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l1141_114111


namespace NUMINAMATH_CALUDE_min_value_of_a_l1141_114116

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l1141_114116


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_leg_length_l1141_114106

/-- Proves that in an isosceles right triangle with a hypotenuse of length 8.485281374238571, the length of one leg is 6. -/
theorem isosceles_right_triangle_leg_length : 
  ∀ (a : ℝ), 
    (a > 0) →  -- Ensure positive length
    (a * Real.sqrt 2 = 8.485281374238571) →  -- Hypotenuse length condition
    (a = 6) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_leg_length_l1141_114106


namespace NUMINAMATH_CALUDE_no_integer_distance_point_l1141_114107

theorem no_integer_distance_point (x y : ℕ) (hx : Odd x) (hy : Odd y) :
  ¬ ∃ (a d : ℝ), 0 < a ∧ a < x ∧ 0 < d ∧ d < y ∧
    (∃ (w x y z : ℕ), 
      a^2 + d^2 = (w : ℝ)^2 ∧
      (x - a)^2 + d^2 = (x : ℝ)^2 ∧
      a^2 + (y - d)^2 = (y : ℝ)^2 ∧
      (x - a)^2 + (y - d)^2 = (z : ℝ)^2) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_distance_point_l1141_114107


namespace NUMINAMATH_CALUDE_multiply_by_eleven_l1141_114163

theorem multiply_by_eleven (x : ℝ) : 11 * x = 103.95 → x = 9.45 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_eleven_l1141_114163


namespace NUMINAMATH_CALUDE_player_A_winning_strategy_l1141_114105

-- Define the game state
structure GameState where
  board : ℕ

-- Define the possible moves for player A
inductive MoveA where 
  | half : MoveA
  | quarter : MoveA
  | triple : MoveA

-- Define the possible moves for player B
inductive MoveB where
  | increment : MoveB
  | decrement : MoveB

-- Define the game step for player A
def stepA (state : GameState) (move : MoveA) : GameState :=
  match move with
  | MoveA.half => 
      if state.board % 2 = 0 then { board := state.board / 2 } else state
  | MoveA.quarter => 
      if state.board % 4 = 0 then { board := state.board / 4 } else state
  | MoveA.triple => { board := state.board * 3 }

-- Define the game step for player B
def stepB (state : GameState) (move : MoveB) : GameState :=
  match move with
  | MoveB.increment => { board := state.board + 1 }
  | MoveB.decrement => 
      if state.board > 1 then { board := state.board - 1 } else state

-- Define the winning condition
def isWinningState (state : GameState) : Prop :=
  state.board = 3

-- Theorem statement
theorem player_A_winning_strategy (n : ℕ) (h : n > 0) : 
  ∃ (strategy : ℕ → MoveA), 
    ∀ (player_B_moves : ℕ → MoveB),
      ∃ (k : ℕ), isWinningState (
        (stepB (stepA { board := n } (strategy 0)) (player_B_moves 0))
      ) ∨ 
      isWinningState (
        (List.foldl 
          (λ state i => stepB (stepA state (strategy i)) (player_B_moves i))
          { board := n }
          (List.range k)
        )
      ) := by
  sorry

end NUMINAMATH_CALUDE_player_A_winning_strategy_l1141_114105


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1141_114146

theorem complex_equation_solution :
  ∃ z : ℂ, (5 : ℂ) - 2 * Complex.I * z = 1 + 5 * Complex.I * z ∧ z = -((4 : ℂ) * Complex.I / 7) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1141_114146


namespace NUMINAMATH_CALUDE_chess_and_go_purchase_l1141_114140

theorem chess_and_go_purchase (m : ℕ) : 
  (m + (120 - m) = 120) →
  (m ≥ 2 * (120 - m)) →
  (30 * m + 25 * (120 - m) ≤ 3500) →
  (80 ≤ m ∧ m ≤ 100) :=
by sorry

end NUMINAMATH_CALUDE_chess_and_go_purchase_l1141_114140


namespace NUMINAMATH_CALUDE_intersection_M_N_l1141_114131

def M : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1141_114131


namespace NUMINAMATH_CALUDE_integer_equation_solution_l1141_114198

theorem integer_equation_solution (x y : ℤ) (h : x^2 + 2 = 3*x + 75*y) :
  ∃ t : ℤ, x = 75*t + 1 ∨ x = 75*t + 2 ∨ x = 75*t + 26 ∨ x = 75*t - 23 :=
sorry

end NUMINAMATH_CALUDE_integer_equation_solution_l1141_114198


namespace NUMINAMATH_CALUDE_train_length_calculation_l1141_114182

/-- The length of a train given specific conditions -/
theorem train_length_calculation (crossing_time : Real) (man_speed : Real) (train_speed : Real) :
  let relative_speed := (train_speed - man_speed) * (5 / 18)
  let train_length := relative_speed * crossing_time
  crossing_time = 35.99712023038157 ∧ 
  man_speed = 3 ∧ 
  train_speed = 63 →
  ∃ ε > 0, |train_length - 600| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1141_114182


namespace NUMINAMATH_CALUDE_bob_and_alice_heights_l1141_114108

/-- The problem statement about Bob and Alice's heights --/
theorem bob_and_alice_heights :
  ∀ (initial_height : ℝ) (bob_growth_percent : ℝ) (alice_growth_ratio : ℝ) (bob_final_height : ℝ),
  initial_height > 0 →
  bob_growth_percent = 0.25 →
  alice_growth_ratio = 1/3 →
  bob_final_height = 75 →
  bob_final_height = initial_height * (1 + bob_growth_percent) →
  let bob_growth_inches := initial_height * bob_growth_percent
  let alice_growth_inches := bob_growth_inches * alice_growth_ratio
  let alice_final_height := initial_height + alice_growth_inches
  alice_final_height = 65 := by
sorry


end NUMINAMATH_CALUDE_bob_and_alice_heights_l1141_114108


namespace NUMINAMATH_CALUDE_max_cut_length_30x30_225_l1141_114132

/-- Represents a square board with side length and number of parts it's cut into -/
structure Board where
  side_length : ℕ
  num_parts : ℕ

/-- Calculates the maximum total length of cuts for a given board -/
def max_cut_length (b : Board) : ℕ :=
  sorry

/-- The theorem stating the maximum cut length for a 30x30 board cut into 225 parts -/
theorem max_cut_length_30x30_225 :
  let b : Board := { side_length := 30, num_parts := 225 }
  max_cut_length b = 1065 :=
sorry

end NUMINAMATH_CALUDE_max_cut_length_30x30_225_l1141_114132


namespace NUMINAMATH_CALUDE_not_in_sample_l1141_114155

/-- Represents a systematic sampling problem -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  known_seats : Finset ℕ
  h_total : total_students = 60
  h_sample : sample_size = 5
  h_known : known_seats = {3, 15, 45, 53}

/-- The interval between sampled items in systematic sampling -/
def sample_interval (s : SystematicSampling) : ℕ :=
  s.total_students / s.sample_size

/-- Checks if a given seat number could be in the sample -/
def could_be_in_sample (s : SystematicSampling) (seat : ℕ) : Prop :=
  ∃ k, 0 < k ∧ k ≤ s.sample_size ∧ seat = k * (sample_interval s)

/-- The main theorem stating that 37 cannot be the remaining seat in the sample -/
theorem not_in_sample (s : SystematicSampling) : ¬(could_be_in_sample s 37) := by
  sorry

end NUMINAMATH_CALUDE_not_in_sample_l1141_114155


namespace NUMINAMATH_CALUDE_range_of_a_l1141_114122

def A (a : ℝ) := {x : ℝ | 1 ≤ x ∧ x ≤ a}
def B (a : ℝ) := {y : ℝ | ∃ x ∈ A a, y = 5 * x - 6}
def C (a : ℝ) := {m : ℝ | ∃ x ∈ A a, m = x^2}

theorem range_of_a (a : ℝ) :
  (B a ∩ C a = C a) ↔ (2 ≤ a ∧ a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1141_114122


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1141_114145

def a : Fin 2 → ℝ := ![3, 2]
def b (n : ℝ) : Fin 2 → ℝ := ![2, n]

theorem perpendicular_vectors (n : ℝ) : 
  (∀ i : Fin 2, (a i) * (b n i) = 0) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1141_114145


namespace NUMINAMATH_CALUDE_bearings_count_proof_l1141_114159

/-- The number of machines -/
def num_machines : ℕ := 10

/-- The normal cost per ball bearing in cents -/
def normal_cost : ℕ := 100

/-- The sale price per ball bearing in cents -/
def sale_price : ℕ := 75

/-- The additional discount rate for bulk purchase -/
def bulk_discount : ℚ := 1/5

/-- The amount saved in cents by buying during the sale -/
def amount_saved : ℕ := 12000

/-- The number of ball bearings per machine -/
def bearings_per_machine : ℕ := 30

theorem bearings_count_proof :
  ∃ (x : ℕ),
    x = bearings_per_machine ∧
    (num_machines * normal_cost * x) -
    (num_machines * sale_price * x * (1 - bulk_discount)) =
    amount_saved :=
sorry

end NUMINAMATH_CALUDE_bearings_count_proof_l1141_114159


namespace NUMINAMATH_CALUDE_minimum_nickels_needed_l1141_114109

def book_cost : ℚ := 46.25

def five_dollar_bills : ℕ := 4
def one_dollar_bills : ℕ := 5
def quarters : ℕ := 10

def nickel_value : ℚ := 0.05

theorem minimum_nickels_needed :
  ∃ n : ℕ,
    (n : ℚ) * nickel_value +
    (five_dollar_bills : ℚ) * 5 +
    (one_dollar_bills : ℚ) * 1 +
    (quarters : ℚ) * 0.25 ≥ book_cost ∧
    ∀ m : ℕ, m < n →
      (m : ℚ) * nickel_value +
      (five_dollar_bills : ℚ) * 5 +
      (one_dollar_bills : ℚ) * 1 +
      (quarters : ℚ) * 0.25 < book_cost :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_nickels_needed_l1141_114109


namespace NUMINAMATH_CALUDE_f_composition_eq_one_fourth_l1141_114188

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_eq_one_fourth :
  f (f (1/9)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_eq_one_fourth_l1141_114188


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1141_114133

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + m + 6 = 0

-- Define the condition for intersection with y-axis
def intersects_y_axis (m : ℝ) : Prop :=
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ circle_equation 0 y1 m ∧ circle_equation 0 y2 m

-- Define the condition for points being on the same side of the origin
def same_side_of_origin (y1 y2 : ℝ) : Prop :=
  (y1 > 0 ∧ y2 > 0) ∨ (y1 < 0 ∧ y2 < 0)

-- Main theorem
theorem circle_intersection_range (m : ℝ) :
  (intersects_y_axis m ∧ 
   ∀ y1 y2 : ℝ, circle_equation 0 y1 m → circle_equation 0 y2 m → same_side_of_origin y1 y2) →
  -6 < m ∧ m < -5 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1141_114133


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1141_114169

/-- Two planar vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- Given two parallel planar vectors (3, 1) and (x, -3), x equals -9 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (3, 1) (x, -3) → x = -9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1141_114169


namespace NUMINAMATH_CALUDE_smallest_population_satisfying_conditions_l1141_114141

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem smallest_population_satisfying_conditions :
  ∃ (n : ℕ),
    (is_perfect_square n) ∧
    (is_perfect_square (n + 100)) ∧
    (∃ k : ℕ, n + 50 = k * k + 1) ∧
    (n % 3 = 0) ∧
    (∀ m : ℕ, m < n →
      ¬(is_perfect_square m ∧
        is_perfect_square (m + 100) ∧
        (∃ k : ℕ, m + 50 = k * k + 1) ∧
        (m % 3 = 0))) ∧
    n = 576 :=
by sorry

end NUMINAMATH_CALUDE_smallest_population_satisfying_conditions_l1141_114141


namespace NUMINAMATH_CALUDE_safe_gold_rows_l1141_114120

/-- The number of gold bars per row in the safe. -/
def gold_bars_per_row : ℕ := 20

/-- The total worth of all gold bars in the safe, in dollars. -/
def total_worth : ℕ := 1600000

/-- The number of rows of gold bars in the safe. -/
def num_rows : ℕ := total_worth / (gold_bars_per_row * (total_worth / gold_bars_per_row))

theorem safe_gold_rows : num_rows = 1 := by
  sorry

end NUMINAMATH_CALUDE_safe_gold_rows_l1141_114120


namespace NUMINAMATH_CALUDE_power_of_power_l1141_114164

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1141_114164


namespace NUMINAMATH_CALUDE_free_throw_contest_l1141_114117

theorem free_throw_contest (x : ℕ) : 
  x + 3*x + 6*x = 80 → x = 8 := by sorry

end NUMINAMATH_CALUDE_free_throw_contest_l1141_114117


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1141_114154

/-- The total surface area of a rectangular solid. -/
def totalSurfaceArea (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 9 meters, 
    width 8 meters, and depth 5 meters is 314 square meters. -/
theorem rectangular_solid_surface_area :
  totalSurfaceArea 9 8 5 = 314 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1141_114154


namespace NUMINAMATH_CALUDE_max_value_constraint_l1141_114192

theorem max_value_constraint (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) :
  ∃ (max : ℝ), (∀ x' y' : ℝ, 4 * x'^2 + y'^2 + x' * y' = 1 → 2 * x' + y' ≤ max) ∧
                max = 2 * Real.sqrt 10 / 5 :=
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1141_114192


namespace NUMINAMATH_CALUDE_inequality_proof_l1141_114101

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1141_114101


namespace NUMINAMATH_CALUDE_hemisphere_on_cone_surface_area_l1141_114102

/-- The total surface area of a solid consisting of a hemisphere on top of a cone,
    where the area of the hemisphere's base is 144π and the height of the cone is
    twice the radius of the hemisphere. -/
theorem hemisphere_on_cone_surface_area :
  ∀ (r : ℝ),
  r > 0 →
  π * r^2 = 144 * π →
  let hemisphere_area := 2 * π * r^2
  let cone_height := 2 * r
  let cone_slant_height := Real.sqrt (r^2 + cone_height^2)
  let cone_area := π * r * cone_slant_height
  hemisphere_area + cone_area = 288 * π + 144 * Real.sqrt 5 * π :=
by
  sorry

end NUMINAMATH_CALUDE_hemisphere_on_cone_surface_area_l1141_114102


namespace NUMINAMATH_CALUDE_function_properties_l1141_114139

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x ↦ sorry

-- State the main theorem
theorem function_properties (h : ∀ x : ℝ, 3 * f (2 - x) - 2 * f x = x^2 - 2*x) :
  (∀ x : ℝ, f x = x^2 - 2*x) ∧
  (∀ a : ℝ, a > 1 → ∀ x : ℝ, f x + a > 0) ∧
  (∀ x : ℝ, f x + 1 > 0 ↔ x ≠ 1) ∧
  (∀ a : ℝ, a < 1 → ∀ x : ℝ, f x + a > 0 ↔ x > 1 + Real.sqrt (1 - a) ∨ x < 1 - Real.sqrt (1 - a)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1141_114139


namespace NUMINAMATH_CALUDE_hostel_stay_duration_l1141_114180

/-- Cost structure for a student youth hostel stay -/
structure CostStructure where
  first_week_rate : ℝ
  additional_day_rate : ℝ

/-- Calculate the number of days stayed given the cost structure and total cost -/
def days_stayed (cs : CostStructure) (total_cost : ℝ) : ℕ :=
  sorry

/-- Theorem stating that for the given cost structure and total cost, the stay is 23 days -/
theorem hostel_stay_duration :
  let cs : CostStructure := { first_week_rate := 18, additional_day_rate := 14 }
  let total_cost : ℝ := 350
  days_stayed cs total_cost = 23 := by
  sorry

end NUMINAMATH_CALUDE_hostel_stay_duration_l1141_114180


namespace NUMINAMATH_CALUDE_pencil_cost_proof_l1141_114119

/-- The cost of 4 pencils and 5 pens in dollars -/
def total_cost_1 : ℚ := 2

/-- The cost of 3 pencils and 4 pens in dollars -/
def total_cost_2 : ℚ := 79/50

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := 1/10

theorem pencil_cost_proof :
  ∃ (pen_cost : ℚ),
    4 * pencil_cost + 5 * pen_cost = total_cost_1 ∧
    3 * pencil_cost + 4 * pen_cost = total_cost_2 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_proof_l1141_114119


namespace NUMINAMATH_CALUDE_correct_cases_delivered_l1141_114121

/-- The number of tins in each case -/
def tins_per_case : ℕ := 24

/-- The percentage of undamaged tins -/
def undamaged_percentage : ℚ := 95/100

/-- The number of undamaged tins left -/
def undamaged_tins : ℕ := 342

/-- The number of cases delivered -/
def cases_delivered : ℕ := 15

theorem correct_cases_delivered :
  cases_delivered * tins_per_case * undamaged_percentage = undamaged_tins := by
  sorry

end NUMINAMATH_CALUDE_correct_cases_delivered_l1141_114121


namespace NUMINAMATH_CALUDE_f_of_3_equals_11_f_equiv_l1141_114118

-- Define the function f
def f (t : ℝ) : ℝ := t^2 + 2

-- State the theorem
theorem f_of_3_equals_11 : f 3 = 11 := by
  sorry

-- Define the original function property
axiom f_property (x : ℝ) (hx : x ≠ 0) : f (x - 1/x) = x^2 + 1/x^2

-- Prove the equivalence of the two function definitions
theorem f_equiv (x : ℝ) (hx : x ≠ 0) : f (x - 1/x) = x^2 + 1/x^2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_11_f_equiv_l1141_114118


namespace NUMINAMATH_CALUDE_election_result_l1141_114166

/-- Represents the number of votes for each candidate in an election --/
structure ElectionResults where
  total_votes : Nat
  john_votes : Nat
  james_percentage : Rat
  john_votes_le_total : john_votes ≤ total_votes

/-- Calculates the difference in votes between the third candidate and John --/
def vote_difference (e : ElectionResults) : Int :=
  e.total_votes - e.john_votes - 
  Nat.floor (e.james_percentage * (e.total_votes - e.john_votes : Rat)) - e.john_votes

/-- Theorem stating the vote difference for the given election scenario --/
theorem election_result : 
  ∀ (e : ElectionResults), 
    e.total_votes = 1150 ∧ 
    e.john_votes = 150 ∧ 
    e.james_percentage = 7/10 → 
    vote_difference e = 150 := by
  sorry


end NUMINAMATH_CALUDE_election_result_l1141_114166


namespace NUMINAMATH_CALUDE_number_puzzle_l1141_114175

theorem number_puzzle (x y : ℝ) : x = 95 → (x / 5 + y = 42) → y = 23 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1141_114175


namespace NUMINAMATH_CALUDE_square_impossibility_l1141_114153

theorem square_impossibility (n : ℕ) : n^2 = 24 → False := by
  sorry

end NUMINAMATH_CALUDE_square_impossibility_l1141_114153


namespace NUMINAMATH_CALUDE_square_of_105_l1141_114127

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_square_of_105_l1141_114127
