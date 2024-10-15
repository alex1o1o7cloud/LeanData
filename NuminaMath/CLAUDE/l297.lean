import Mathlib

namespace NUMINAMATH_CALUDE_smallest_c_l297_29737

/-- A square with side length c -/
structure Square (c : ℝ) where
  side : c > 0

/-- A coloring of points on a square -/
def Coloring (c : ℝ) := Square c → Bool

/-- The distance between two points on a square -/
def distance (c : ℝ) (p q : Square c) : ℝ := sorry

/-- There exist two points of the same color with distance at least √5 -/
def hasMonochromaticPair (c : ℝ) (coloring : Coloring c) : Prop :=
  ∃ (p q : Square c), coloring p = coloring q ∧ distance c p q ≥ Real.sqrt 5

/-- The smallest possible value of c satisfying the condition -/
theorem smallest_c : 
  (∀ c : ℝ, c ≥ Real.sqrt 10 / 2 → ∀ coloring : Coloring c, hasMonochromaticPair c coloring) ∧
  (∀ c : ℝ, c < Real.sqrt 10 / 2 → ∃ coloring : Coloring c, ¬hasMonochromaticPair c coloring) :=
sorry

end NUMINAMATH_CALUDE_smallest_c_l297_29737


namespace NUMINAMATH_CALUDE_age_range_count_l297_29770

/-- Calculates the number of integer ages within one standard deviation of the average age -/
def count_ages_within_std_dev (average_age : ℕ) (std_dev : ℕ) : ℕ :=
  (average_age + std_dev) - (average_age - std_dev) + 1

/-- Proves that given an average age of 31 and a standard deviation of 9, 
    the number of integer ages within one standard deviation of the average is 19 -/
theorem age_range_count : count_ages_within_std_dev 31 9 = 19 := by
  sorry

end NUMINAMATH_CALUDE_age_range_count_l297_29770


namespace NUMINAMATH_CALUDE_quadratic_as_binomial_square_l297_29785

theorem quadratic_as_binomial_square (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 - 24*x + c = (a*x + b)^2) → c = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_as_binomial_square_l297_29785


namespace NUMINAMATH_CALUDE_triangle_side_length_l297_29773

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  a = 8 → b = 7 → B = Real.pi / 3 → (c = 3 ∨ c = 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l297_29773


namespace NUMINAMATH_CALUDE_area_ratio_squares_l297_29739

/-- Given squares A, B, and C with specified properties, prove the ratio of areas of A to C -/
theorem area_ratio_squares (sideA sideB sideC : ℝ) : 
  sideA * 4 = 16 →  -- Perimeter of A is 16
  sideB * 4 = 40 →  -- Perimeter of B is 40
  sideC = 1.5 * sideA →  -- Side of C is 1.5 times side of A
  (sideA ^ 2) / (sideC ^ 2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_squares_l297_29739


namespace NUMINAMATH_CALUDE_largest_four_digit_number_l297_29797

def digits : Finset Nat := {5, 1, 6, 2, 4}

def is_valid_number (n : Nat) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ 
  (Finset.card (Finset.filter (λ d => d ∈ digits) (Finset.image (λ i => (n / 10^i) % 10) {0,1,2,3})) = 4)

theorem largest_four_digit_number :
  ∀ n : Nat, is_valid_number n → n ≤ 6542 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_number_l297_29797


namespace NUMINAMATH_CALUDE_book_sale_price_l297_29744

def book_sale (total_books : ℕ) (unsold_books : ℕ) (total_amount : ℚ) : Prop :=
  let sold_books := total_books - unsold_books
  let price_per_book := total_amount / sold_books
  (2 : ℚ) / 3 * total_books = sold_books ∧
  unsold_books = 36 ∧
  total_amount = 252 ∧
  price_per_book = (7 : ℚ) / 2

theorem book_sale_price :
  ∃ (total_books : ℕ) (unsold_books : ℕ) (total_amount : ℚ),
    book_sale total_books unsold_books total_amount :=
by
  sorry

end NUMINAMATH_CALUDE_book_sale_price_l297_29744


namespace NUMINAMATH_CALUDE_log_ratio_squared_l297_29779

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1)
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log x)
  (h2 : x * y = 27) :
  ((Real.log x - Real.log y) / Real.log 3) ^ 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l297_29779


namespace NUMINAMATH_CALUDE_image_of_two_three_l297_29743

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * p.2, p.1 + p.2)

-- State the theorem
theorem image_of_two_three :
  f (2, 3) = (6, 5) := by
  sorry

end NUMINAMATH_CALUDE_image_of_two_three_l297_29743


namespace NUMINAMATH_CALUDE_carol_trivia_score_l297_29731

/-- Carol's trivia game score calculation -/
theorem carol_trivia_score (first_round : ℤ) (second_round : ℤ) (last_round : ℤ) (total : ℤ) : 
  second_round = 6 →
  last_round = -16 →
  total = 7 →
  first_round + second_round + last_round = total →
  first_round = 17 := by
sorry

end NUMINAMATH_CALUDE_carol_trivia_score_l297_29731


namespace NUMINAMATH_CALUDE_triangle_perimeter_l297_29750

/-- Given a triangle with inradius 5.0 cm and area 105 cm², its perimeter is 42 cm. -/
theorem triangle_perimeter (inradius : ℝ) (area : ℝ) (perimeter : ℝ) : 
  inradius = 5.0 → area = 105 → area = inradius * (perimeter / 2) → perimeter = 42 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l297_29750


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l297_29759

-- Define sets A and B
def A : Set ℝ := {x | 2 * x - 1 > 0}
def B : Set ℝ := {x | |x| < 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l297_29759


namespace NUMINAMATH_CALUDE_not_all_tangents_equal_l297_29722

/-- A convex quadrilateral where the tangent of one angle is m -/
structure ConvexQuadrilateral (m : ℝ) where
  angles : Fin 4 → ℝ
  sum_360 : angles 0 + angles 1 + angles 2 + angles 3 = 360
  all_positive : ∀ i, 0 < angles i
  all_less_180 : ∀ i, angles i < 180
  one_tangent_m : ∃ i, Real.tan (angles i) = m

/-- Theorem stating that it's impossible for all angles to have tangent m -/
theorem not_all_tangents_equal (m : ℝ) (q : ConvexQuadrilateral m) :
  ¬(∀ i, Real.tan (q.angles i) = m) :=
sorry

end NUMINAMATH_CALUDE_not_all_tangents_equal_l297_29722


namespace NUMINAMATH_CALUDE_total_readers_l297_29767

/-- The number of eBook readers Anna bought -/
def anna_readers : ℕ := 50

/-- The difference between Anna's and John's initial number of eBook readers -/
def reader_difference : ℕ := 15

/-- The number of eBook readers John lost -/
def john_lost : ℕ := 3

/-- Theorem: The total number of eBook readers John and Anna have is 82 -/
theorem total_readers : 
  anna_readers + (anna_readers - reader_difference - john_lost) = 82 := by
sorry

end NUMINAMATH_CALUDE_total_readers_l297_29767


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_b_l297_29704

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_b : 
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (b n) (b (n + 1)) ≤ k ∧ 
  ∃ (m : ℕ), Nat.gcd (b m) (b (m + 1)) = k :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_b_l297_29704


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l297_29730

theorem students_taking_one_subject (both : ℕ) (math : ℕ) (only_science : ℕ)
  (h1 : both = 15)
  (h2 : math = 30)
  (h3 : only_science = 18) :
  math - both + only_science = 33 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l297_29730


namespace NUMINAMATH_CALUDE_cost_of_six_lollipops_l297_29752

/-- The cost of 6 giant lollipops with discounts and promotions -/
theorem cost_of_six_lollipops (regular_price : ℝ) (discount_rate : ℝ) : 
  regular_price = 2.4 / 2 →
  discount_rate = 0.1 →
  6 * regular_price * (1 - discount_rate) = 6.48 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_six_lollipops_l297_29752


namespace NUMINAMATH_CALUDE_smallest_x_multiple_of_53_l297_29707

theorem smallest_x_multiple_of_53 : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ (3*y)^2 + 3*43*3*y + 43^2)) ∧
  (53 ∣ (3*x)^2 + 3*43*3*x + 43^2) ∧
  x = 21 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_multiple_of_53_l297_29707


namespace NUMINAMATH_CALUDE_ball_max_height_l297_29783

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 55

-- State the theorem
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 135 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l297_29783


namespace NUMINAMATH_CALUDE_append_digit_twice_divisible_by_three_l297_29721

theorem append_digit_twice_divisible_by_three (N d : ℕ) 
  (hN : N % 3 ≠ 0) (hd : d % 3 ≠ 0) (hd_last : d < 10) :
  ∃ k, N * 100 + d * 10 + d = 3 * k :=
sorry

end NUMINAMATH_CALUDE_append_digit_twice_divisible_by_three_l297_29721


namespace NUMINAMATH_CALUDE_inequality_proof_l297_29708

theorem inequality_proof (x : ℝ) (h1 : (3/2 : ℝ) ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l297_29708


namespace NUMINAMATH_CALUDE_trivia_team_score_l297_29701

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_score : ℕ) :
  total_members = 5 →
  absent_members = 2 →
  total_score = 18 →
  ∃ (points_per_member : ℕ),
    points_per_member * (total_members - absent_members) = total_score ∧
    points_per_member = 6 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_score_l297_29701


namespace NUMINAMATH_CALUDE_ship_food_supply_l297_29738

/-- Calculates the remaining food supply on a ship after a specific consumption pattern. -/
theorem ship_food_supply (initial_supply : ℝ) : 
  initial_supply = 400 →
  (initial_supply - 2/5 * initial_supply) - 3/5 * (initial_supply - 2/5 * initial_supply) = 96 := by
  sorry

#check ship_food_supply

end NUMINAMATH_CALUDE_ship_food_supply_l297_29738


namespace NUMINAMATH_CALUDE_modified_triangular_array_100th_row_sum_l297_29705

/-- Sum of numbers in the nth row of the modified triangular array -/
def row_sum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * row_sum (n - 1) + 2

theorem modified_triangular_array_100th_row_sum :
  row_sum 100 = 2^100 - 2 :=
sorry

end NUMINAMATH_CALUDE_modified_triangular_array_100th_row_sum_l297_29705


namespace NUMINAMATH_CALUDE_mark_current_age_l297_29775

/-- Mark's current age -/
def mark_age : ℕ := 28

/-- Aaron's current age -/
def aaron_age : ℕ := 11

/-- Theorem stating that Mark's current age is 28, given the conditions about their ages -/
theorem mark_current_age :
  (mark_age - 3 = 3 * (aaron_age - 3) + 1) ∧
  (mark_age + 4 = 2 * (aaron_age + 4) + 2) →
  mark_age = 28 := by
  sorry


end NUMINAMATH_CALUDE_mark_current_age_l297_29775


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_three_to_m_l297_29788

def m : ℕ := 2021^2 + 3^2021

theorem units_digit_of_m_squared_plus_three_to_m (m : ℕ := 2021^2 + 3^2021) :
  (m^2 + 3^m) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_three_to_m_l297_29788


namespace NUMINAMATH_CALUDE_power_plus_sum_l297_29709

theorem power_plus_sum : 10^2 + 10 + 1 = 111 := by
  sorry

end NUMINAMATH_CALUDE_power_plus_sum_l297_29709


namespace NUMINAMATH_CALUDE_mason_daily_water_l297_29768

/-- The number of cups of water Theo drinks per day -/
def theo_daily : ℕ := 8

/-- The number of cups of water Roxy drinks per day -/
def roxy_daily : ℕ := 9

/-- The total number of cups of water the siblings drink in one week -/
def total_weekly : ℕ := 168

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Proves that Mason drinks 7 cups of water every day -/
theorem mason_daily_water : ℕ := by
  sorry

end NUMINAMATH_CALUDE_mason_daily_water_l297_29768


namespace NUMINAMATH_CALUDE_triangle_side_range_l297_29716

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_side_range (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c)
  (h2 : t.b = Real.sqrt 3) :
  Real.sqrt 3 < 2 * t.a + t.c ∧ 2 * t.a + t.c ≤ 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_range_l297_29716


namespace NUMINAMATH_CALUDE_marcus_baseball_cards_l297_29748

/-- The number of baseball cards Carter has -/
def carterCards : ℕ := 152

/-- The number of additional cards Marcus has compared to Carter -/
def marcusExtraCards : ℕ := 58

/-- The number of baseball cards Marcus has -/
def marcusCards : ℕ := carterCards + marcusExtraCards

theorem marcus_baseball_cards : marcusCards = 210 := by
  sorry

end NUMINAMATH_CALUDE_marcus_baseball_cards_l297_29748


namespace NUMINAMATH_CALUDE_sin_2beta_value_l297_29777

theorem sin_2beta_value (α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.cos (2 * α + β) - 2 * Real.cos (α + β) * Real.cos α = 3/5) :
  Real.sin (2 * β) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2beta_value_l297_29777


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l297_29712

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -6) : 
  x^2 + y^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l297_29712


namespace NUMINAMATH_CALUDE_triangle_area_l297_29778

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  A = π / 6 →  -- 30°
  C = π / 4 →  -- 45°
  a = 2 →
  B + C + A = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l297_29778


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l297_29734

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l297_29734


namespace NUMINAMATH_CALUDE_cube_root_and_seventh_root_sum_l297_29782

theorem cube_root_and_seventh_root_sum (m n : ℤ) 
  (hm : m ^ 3 = 61629875)
  (hn : n ^ 7 = 170859375) :
  100 * m + n = 39515 := by
sorry

end NUMINAMATH_CALUDE_cube_root_and_seventh_root_sum_l297_29782


namespace NUMINAMATH_CALUDE_polynomial_coefficient_product_l297_29706

theorem polynomial_coefficient_product (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a * (a₁ + a₃) = 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_product_l297_29706


namespace NUMINAMATH_CALUDE_route_b_saves_six_hours_l297_29718

-- Define the time for each route (one way)
def route_a_time : ℕ := 5
def route_b_time : ℕ := 2

-- Define the function to calculate round trip time
def round_trip_time (one_way_time : ℕ) : ℕ := 2 * one_way_time

-- Define the function to calculate time saved
def time_saved (longer_route : ℕ) (shorter_route : ℕ) : ℕ :=
  round_trip_time longer_route - round_trip_time shorter_route

-- Theorem statement
theorem route_b_saves_six_hours :
  time_saved route_a_time route_b_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_route_b_saves_six_hours_l297_29718


namespace NUMINAMATH_CALUDE_next_draw_highest_probability_l297_29711

/-- The probability of drawing a specific number in a lottery draw -/
def draw_probability : ℚ := 5 / 90

/-- The probability of not drawing a specific number in a lottery draw -/
def not_draw_probability : ℚ := 1 - draw_probability

/-- The probability of drawing a specific number in the n-th future draw -/
def future_draw_probability (n : ℕ) : ℚ :=
  (not_draw_probability ^ (n - 1)) * draw_probability

theorem next_draw_highest_probability :
  ∀ n : ℕ, n > 1 → draw_probability > future_draw_probability n :=
sorry

end NUMINAMATH_CALUDE_next_draw_highest_probability_l297_29711


namespace NUMINAMATH_CALUDE_polynomial_sum_l297_29781

theorem polynomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l297_29781


namespace NUMINAMATH_CALUDE_power_sum_equality_l297_29760

theorem power_sum_equality : (-2)^23 + 2^(2^4 + 5^2 - 7^2) = -8388607.99609375 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l297_29760


namespace NUMINAMATH_CALUDE_ticket_sales_problem_l297_29719

/-- Proves that the total number of tickets sold is 42 given the conditions of the ticket sales problem. -/
theorem ticket_sales_problem (adult_price child_price total_sales child_tickets : ℕ)
  (h1 : adult_price = 5)
  (h2 : child_price = 3)
  (h3 : total_sales = 178)
  (h4 : child_tickets = 16) :
  ∃ (adult_tickets : ℕ), adult_price * adult_tickets + child_price * child_tickets = total_sales ∧
                          adult_tickets + child_tickets = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_problem_l297_29719


namespace NUMINAMATH_CALUDE_problem_solution_l297_29755

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/6 = 3*y) : x = 108 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l297_29755


namespace NUMINAMATH_CALUDE_savings_comparison_l297_29732

theorem savings_comparison (S : ℝ) (h1 : S > 0) : 
  let last_year_savings := 0.06 * S
  let this_year_salary := 1.1 * S
  let this_year_savings := 0.09 * this_year_salary
  (this_year_savings / last_year_savings) * 100 = 165 := by
sorry

end NUMINAMATH_CALUDE_savings_comparison_l297_29732


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l297_29724

-- Define the polynomial g(x)
def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem sum_of_coefficients (a b c d : ℝ) : 
  g a b c d (1 + I) = 0 → g a b c d (3*I) = 0 → a + b + c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l297_29724


namespace NUMINAMATH_CALUDE_almond_butter_cookie_cost_difference_l297_29710

/-- The cost difference per batch between almond butter cookies and peanut butter cookies -/
def cost_difference (peanut_butter_price : ℝ) (almond_butter_multiplier : ℝ) (jar_fraction : ℝ) (sugar_price_difference : ℝ) : ℝ :=
  (almond_butter_multiplier * peanut_butter_price * jar_fraction - peanut_butter_price * jar_fraction) + sugar_price_difference

/-- Theorem: The cost difference per batch between almond butter cookies and peanut butter cookies is $3.50 -/
theorem almond_butter_cookie_cost_difference :
  cost_difference 3 3 (1/2) 0.5 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_almond_butter_cookie_cost_difference_l297_29710


namespace NUMINAMATH_CALUDE_hyperbola_equation_l297_29758

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  h_asymptote : ∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt 3 * x
  h_focus_on_directrix : ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ x = -6

/-- The theorem stating the specific equation of the hyperbola -/
theorem hyperbola_equation (h : Hyperbola) : 
  h.a^2 = 9 ∧ h.b^2 = 27 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l297_29758


namespace NUMINAMATH_CALUDE_power_of_two_problem_l297_29798

theorem power_of_two_problem (k : ℕ) (N : ℕ) :
  2^k = N → 2^(2*k + 2) = 64 → N = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_problem_l297_29798


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_l297_29793

theorem quadratic_equations_common_root (a b c x : ℝ) 
  (h1 : a * c ≠ 0) (h2 : a ≠ c) 
  (hM : a * x^2 + b * x + c = 0) 
  (hN : c * x^2 + b * x + a = 0) : 
  x = 1 ∨ x = -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_l297_29793


namespace NUMINAMATH_CALUDE_bear_population_difference_l297_29766

theorem bear_population_difference :
  ∀ (white black brown : ℕ),
    black = 2 * white →
    black = 60 →
    white + black + brown = 190 →
    brown - black = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_bear_population_difference_l297_29766


namespace NUMINAMATH_CALUDE_find_y_l297_29791

theorem find_y (x y : ℤ) (h1 : x + y = 280) (h2 : x - y = 200) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l297_29791


namespace NUMINAMATH_CALUDE_keychain_arrangement_theorem_l297_29780

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

def number_of_adjacent_arrangements (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem keychain_arrangement_theorem :
  let total_arrangements := number_of_arrangements 5
  let adjacent_arrangements := number_of_adjacent_arrangements 5
  total_arrangements - adjacent_arrangements = 72 := by
  sorry

end NUMINAMATH_CALUDE_keychain_arrangement_theorem_l297_29780


namespace NUMINAMATH_CALUDE_triangle_altitude_and_area_l297_29762

/-- Triangle with sides a, b, c and altitude h from the vertex opposite side b --/
structure Triangle (a b c : ℝ) where
  h : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_altitude_and_area 
  (t : Triangle 11 13 16) : t.h = 168 / 13 ∧ (1 / 2 : ℝ) * 13 * t.h = 84 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_area_l297_29762


namespace NUMINAMATH_CALUDE_binomial_equation_solution_l297_29728

theorem binomial_equation_solution (x : ℕ) : 
  (Nat.choose 10 (2*x) - Nat.choose 10 (x+1) = 0) → (x = 1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_binomial_equation_solution_l297_29728


namespace NUMINAMATH_CALUDE_highest_probability_l297_29723

-- Define the sample space
variable (Ω : Type*)

-- Define the events A, B, and C
variable (A B C : Set Ω)

-- Define a probability measure
variable (P : Set Ω → ℝ)

-- State the theorem
theorem highest_probability 
  (h_subset1 : C ⊆ B) 
  (h_subset2 : B ⊆ A) 
  (h_prob : ∀ X : Set Ω, 0 ≤ P X ∧ P X ≤ 1) 
  (h_monotone : ∀ X Y : Set Ω, X ⊆ Y → P X ≤ P Y) : 
  P A ≥ P B ∧ P A ≥ P C :=
by sorry

end NUMINAMATH_CALUDE_highest_probability_l297_29723


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l297_29717

theorem purely_imaginary_complex_number (a : ℝ) : 
  (((a^2 - 3*a + 2) : ℂ) + (a - 1)*I).re = 0 ∧ (((a^2 - 3*a + 2) : ℂ) + (a - 1)*I).im ≠ 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l297_29717


namespace NUMINAMATH_CALUDE_expression_evaluation_l297_29764

theorem expression_evaluation : 
  Real.sqrt 2 * (2 ^ (3/2)) + 15 / 5 * 3 - Real.sqrt 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l297_29764


namespace NUMINAMATH_CALUDE_power_function_property_l297_29784

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 4 / f 2 = 3) : 
  f (1/2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_power_function_property_l297_29784


namespace NUMINAMATH_CALUDE_total_sundaes_l297_29726

def num_flavors : ℕ := 8

def sundae_combinations (n : ℕ) : ℕ := Nat.choose num_flavors n

theorem total_sundaes : 
  sundae_combinations 1 + sundae_combinations 2 + sundae_combinations 3 = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_sundaes_l297_29726


namespace NUMINAMATH_CALUDE_fraction_equivalence_l297_29713

theorem fraction_equivalence : 
  (14 / 10 : ℚ) = 7 / 5 ∧ 
  (1 + 2 / 5 : ℚ) = 7 / 5 ∧ 
  (1 + 7 / 25 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 2 / 10 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 14 / 70 : ℚ) ≠ 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l297_29713


namespace NUMINAMATH_CALUDE_inequality_solution_l297_29729

-- Define the inequality function
def f (x : ℝ) := x * (x - 2)

-- Define the solution set
def solution_set : Set ℝ := {x | x < 0 ∨ x > 2}

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l297_29729


namespace NUMINAMATH_CALUDE_max_equidistant_circles_l297_29733

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Four points in a 2D plane -/
def FourPoints := Fin 4 → Point

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if four points lie on the same circle -/
def on_same_circle (points : FourPoints) : Prop := sorry

/-- Predicate to check if a circle is equidistant from all four points -/
def equidistant_circle (c : Circle) (points : FourPoints) : Prop := sorry

/-- The main theorem -/
theorem max_equidistant_circles (points : FourPoints) 
  (h1 : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k))
  (h2 : ¬on_same_circle points) :
  (∃ (circles : Finset Circle), 
    (∀ c ∈ circles, equidistant_circle c points) ∧ 
    circles.card = 7 ∧
    (∀ circles' : Finset Circle, 
      (∀ c ∈ circles', equidistant_circle c points) → 
      circles'.card ≤ 7)) := by sorry

end NUMINAMATH_CALUDE_max_equidistant_circles_l297_29733


namespace NUMINAMATH_CALUDE_yellow_square_ratio_l297_29787

/-- Represents a square banner with a symmetric cross -/
structure Banner where
  side : ℝ
  cross_area_ratio : ℝ
  yellow_area_ratio : ℝ

/-- The banner satisfies the problem conditions -/
def valid_banner (b : Banner) : Prop :=
  b.side > 0 ∧
  b.cross_area_ratio = 0.25 ∧
  b.yellow_area_ratio > 0 ∧
  b.yellow_area_ratio < b.cross_area_ratio

theorem yellow_square_ratio (b : Banner) (h : valid_banner b) :
  b.yellow_area_ratio = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_yellow_square_ratio_l297_29787


namespace NUMINAMATH_CALUDE_original_bill_amount_l297_29725

theorem original_bill_amount (new_bill : ℝ) (increase_percent : ℝ) 
  (h1 : new_bill = 78)
  (h2 : increase_percent = 30) : 
  ∃ (original_bill : ℝ), 
    original_bill * (1 + increase_percent / 100) = new_bill ∧ 
    original_bill = 60 := by
  sorry

end NUMINAMATH_CALUDE_original_bill_amount_l297_29725


namespace NUMINAMATH_CALUDE_smallest_balanced_number_l297_29747

/-- A function that returns true if a number is a three-digit number with distinct non-zero digits -/
def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10) ∧
  (n / 100 ≠ 0) ∧
  ((n / 10) % 10 ≠ 0) ∧
  (n % 10 ≠ 0)

/-- A function that calculates the sum of all two-digit numbers formed from the digits of a three-digit number -/
def sum_of_two_digit_numbers (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b)

/-- The main theorem stating that 132 is the smallest balanced number -/
theorem smallest_balanced_number :
  is_valid_number 132 ∧
  132 = sum_of_two_digit_numbers 132 ∧
  ∀ n < 132, is_valid_number n → n ≠ sum_of_two_digit_numbers n :=
sorry

end NUMINAMATH_CALUDE_smallest_balanced_number_l297_29747


namespace NUMINAMATH_CALUDE_angle_between_v_and_w_l297_29769

/-- The angle between two vectors in ℝ³ -/
def angle (v w : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Vector 1 -/
def v : ℝ × ℝ × ℝ := (3, -2, 2)

/-- Vector 2 -/
def w : ℝ × ℝ × ℝ := (2, 2, -1)

/-- Theorem: The angle between vectors v and w is 90° -/
theorem angle_between_v_and_w : angle v w = 90 := by sorry

end NUMINAMATH_CALUDE_angle_between_v_and_w_l297_29769


namespace NUMINAMATH_CALUDE_h_sqrt_two_equals_zero_min_a_plus_b_h_not_arbitrary_quadratic_l297_29740

-- Define the functions f, g, l, and h
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x
def g (b : ℝ) (x : ℝ) : ℝ := x + b
def l (x : ℝ) : ℝ := 2*x^2 + 3*x - 1

-- Define the property of h being generated by f and g
def is_generated_by_f_and_g (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (m n : ℝ), ∀ x, h x = m * (f a x) + n * (g b x)

-- Define h as a quadratic function
def h (a b m n : ℝ) (x : ℝ) : ℝ := m * (f a x) + n * (g b x)

-- Theorem 1
theorem h_sqrt_two_equals_zero (a b m n : ℝ) :
  a = 1 → b = 2 → (∀ x, h a b m n x = h a b m n (-x)) → 
  h a b m n (Real.sqrt 2) = 0 := by sorry

-- Theorem 2
theorem min_a_plus_b (a b m n : ℝ) :
  b > 0 → is_generated_by_f_and_g (h a b m n) a b →
  (∃ m' n', ∀ x, h a b m n x = m' * g b x + n' * l x) →
  a + b ≥ 3/2 + Real.sqrt 2 := by sorry

-- Theorem 3
theorem h_not_arbitrary_quadratic (a b : ℝ) :
  ¬ ∀ (p q r : ℝ), ∃ (m n : ℝ), ∀ x, h a b m n x = p*x^2 + q*x + r := by sorry

end NUMINAMATH_CALUDE_h_sqrt_two_equals_zero_min_a_plus_b_h_not_arbitrary_quadratic_l297_29740


namespace NUMINAMATH_CALUDE_A_final_value_l297_29772

def update_A (initial_A : Int) : Int :=
  -initial_A + 5

theorem A_final_value (initial_A : Int) (h : initial_A = 15) :
  update_A initial_A = -10 := by
  sorry

end NUMINAMATH_CALUDE_A_final_value_l297_29772


namespace NUMINAMATH_CALUDE_pitcher_problem_l297_29703

theorem pitcher_problem (C : ℝ) (h : C > 0) :
  let juice_volume := C / 2
  let num_cups := 8
  let cup_volume := juice_volume / num_cups
  (cup_volume / C) * 100 = 6.25 := by sorry

end NUMINAMATH_CALUDE_pitcher_problem_l297_29703


namespace NUMINAMATH_CALUDE_parallel_plane_count_l297_29795

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane

/-- Enum representing the possible number of parallel planes -/
inductive ParallelPlaneCount
  | Zero
  | One
  | Infinite

/-- Function to determine the number of parallel planes -/
def countParallelPlanes (l1 l2 : Line3D) : ParallelPlaneCount :=
  sorry

/-- Theorem stating that the number of parallel planes is either zero, one, or infinite -/
theorem parallel_plane_count (l1 l2 : Line3D) :
  ∃ (count : ParallelPlaneCount), countParallelPlanes l1 l2 = count :=
sorry

end NUMINAMATH_CALUDE_parallel_plane_count_l297_29795


namespace NUMINAMATH_CALUDE_ordered_pair_solution_l297_29786

theorem ordered_pair_solution (a b : ℤ) :
  Real.sqrt (9 - 8 * Real.sin (50 * π / 180)) = a + b * (1 / Real.sin (50 * π / 180)) →
  a = 3 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_ordered_pair_solution_l297_29786


namespace NUMINAMATH_CALUDE_sandwich_cost_l297_29774

/-- The cost of a sandwich given the total cost of sandwiches and sodas -/
theorem sandwich_cost (total_cost : ℚ) (num_sandwiches : ℕ) (num_sodas : ℕ) (soda_cost : ℚ) : 
  total_cost = 8.36 ∧ 
  num_sandwiches = 2 ∧ 
  num_sodas = 4 ∧ 
  soda_cost = 0.87 → 
  (total_cost - num_sodas * soda_cost) / num_sandwiches = 2.44 := by
sorry

end NUMINAMATH_CALUDE_sandwich_cost_l297_29774


namespace NUMINAMATH_CALUDE_max_good_word_length_l297_29753

/-- An alphabet is a finite set of letters. -/
def Alphabet (n : ℕ) := Fin n

/-- A word is a finite sequence of letters where consecutive letters are different. -/
def Word (α : Type) := List α

/-- A good word is one where it's impossible to delete all but four letters to obtain aabb. -/
def isGoodWord {α : Type} (w : Word α) : Prop :=
  ∀ (a b : α), a ≠ b → ¬∃ (i j k l : ℕ), i < j ∧ j < k ∧ k < l ∧
    w.get? i = some a ∧ w.get? j = some a ∧ w.get? k = some b ∧ w.get? l = some b

/-- The maximum length of a good word in an alphabet with n > 1 letters is 2n + 1. -/
theorem max_good_word_length {n : ℕ} (h : n > 1) :
  ∃ (w : Word (Alphabet n)), isGoodWord w ∧ w.length = 2 * n + 1 ∧
  ∀ (w' : Word (Alphabet n)), isGoodWord w' → w'.length ≤ 2 * n + 1 :=
sorry

end NUMINAMATH_CALUDE_max_good_word_length_l297_29753


namespace NUMINAMATH_CALUDE_sculpture_and_base_height_l297_29794

/-- The total height of a sculpture and its base -/
def total_height (sculpture_height_m : ℝ) (base_height_cm : ℝ) : ℝ :=
  sculpture_height_m * 100 + base_height_cm

/-- Theorem stating that a 0.88m sculpture on a 20cm base is 108cm tall -/
theorem sculpture_and_base_height : 
  total_height 0.88 20 = 108 := by sorry

end NUMINAMATH_CALUDE_sculpture_and_base_height_l297_29794


namespace NUMINAMATH_CALUDE_division_result_l297_29746

theorem division_result : (-0.91) / (-0.13) = 7 := by sorry

end NUMINAMATH_CALUDE_division_result_l297_29746


namespace NUMINAMATH_CALUDE_ruth_gave_two_sandwiches_to_brother_l297_29745

/-- The number of sandwiches Ruth prepared -/
def total_sandwiches : ℕ := 10

/-- The number of sandwiches Ruth ate -/
def ruth_ate : ℕ := 1

/-- The number of sandwiches the first cousin ate -/
def first_cousin_ate : ℕ := 2

/-- The number of other cousins -/
def other_cousins : ℕ := 2

/-- The number of sandwiches each other cousin ate -/
def each_other_cousin_ate : ℕ := 1

/-- The number of sandwiches left -/
def sandwiches_left : ℕ := 3

/-- The number of sandwiches Ruth gave to her brother -/
def sandwiches_to_brother : ℕ := total_sandwiches - (ruth_ate + first_cousin_ate + other_cousins * each_other_cousin_ate + sandwiches_left)

theorem ruth_gave_two_sandwiches_to_brother : sandwiches_to_brother = 2 := by
  sorry

end NUMINAMATH_CALUDE_ruth_gave_two_sandwiches_to_brother_l297_29745


namespace NUMINAMATH_CALUDE_percentage_decrease_proof_l297_29749

def original_price : ℝ := 250
def new_price : ℝ := 200

theorem percentage_decrease_proof :
  (original_price - new_price) / original_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_proof_l297_29749


namespace NUMINAMATH_CALUDE_interleave_sequences_count_l297_29735

def interleave_sequences (n₁ n₂ n₃ : ℕ) : ℕ :=
  Nat.factorial (n₁ + n₂ + n₃) / (Nat.factorial n₁ * Nat.factorial n₂ * Nat.factorial n₃)

theorem interleave_sequences_count (n₁ n₂ n₃ : ℕ) :
  interleave_sequences n₁ n₂ n₃ = 
    Nat.choose (n₁ + n₂ + n₃) n₁ * Nat.choose (n₂ + n₃) n₂ :=
by sorry

end NUMINAMATH_CALUDE_interleave_sequences_count_l297_29735


namespace NUMINAMATH_CALUDE_range_of_f_l297_29700

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x - 4)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range (fun (x : ℝ) => f x) = {y : ℝ | y < -27 ∨ y > -27} :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l297_29700


namespace NUMINAMATH_CALUDE_defective_units_shipped_percentage_l297_29741

/-- The percentage of units with Type A defects in the first stage -/
def type_a_defect_rate : ℝ := 0.07

/-- The percentage of units with Type B defects in the second stage -/
def type_b_defect_rate : ℝ := 0.08

/-- The percentage of Type A defects that are reworked and repaired -/
def type_a_repair_rate : ℝ := 0.4

/-- The percentage of Type B defects that are reworked and repaired -/
def type_b_repair_rate : ℝ := 0.3

/-- The percentage of remaining Type A defects that are shipped for sale -/
def type_a_ship_rate : ℝ := 0.03

/-- The percentage of remaining Type B defects that are shipped for sale -/
def type_b_ship_rate : ℝ := 0.06

/-- The theorem stating the percentage of units produced that are defective (Type A or B) and shipped for sale -/
theorem defective_units_shipped_percentage :
  (type_a_defect_rate * (1 - type_a_repair_rate) * type_a_ship_rate +
   type_b_defect_rate * (1 - type_b_repair_rate) * type_b_ship_rate) * 100 =
  0.462 := by sorry

end NUMINAMATH_CALUDE_defective_units_shipped_percentage_l297_29741


namespace NUMINAMATH_CALUDE_total_rainfall_l297_29799

theorem total_rainfall (monday tuesday wednesday : ℚ) 
  (h1 : monday = 0.16666666666666666)
  (h2 : tuesday = 0.4166666666666667)
  (h3 : wednesday = 0.08333333333333333) :
  monday + tuesday + wednesday = 0.6666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_l297_29799


namespace NUMINAMATH_CALUDE_unique_fraction_exists_l297_29754

def is_relatively_prime (x y : ℕ+) : Prop := Nat.gcd x.val y.val = 1

theorem unique_fraction_exists : ∃! (x y : ℕ+), 
  is_relatively_prime x y ∧ 
  (x.val + 1 : ℚ) / (y.val + 1) = 1.2 * (x.val : ℚ) / y.val := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_exists_l297_29754


namespace NUMINAMATH_CALUDE_egg_price_calculation_l297_29776

theorem egg_price_calculation (num_eggs : ℕ) (num_chickens : ℕ) (price_per_chicken : ℚ) (total_spent : ℚ) :
  num_eggs = 20 →
  num_chickens = 6 →
  price_per_chicken = 8 →
  total_spent = 88 →
  (total_spent - (num_chickens * price_per_chicken)) / num_eggs = 2 :=
by sorry

end NUMINAMATH_CALUDE_egg_price_calculation_l297_29776


namespace NUMINAMATH_CALUDE_min_cans_correct_l297_29756

/-- The number of ounces in one can of soda -/
def ounces_per_can : ℕ := 12

/-- The number of ounces in a gallon -/
def ounces_per_gallon : ℕ := 128

/-- The minimum number of cans needed to provide at least a gallon of soda -/
def min_cans : ℕ := 11

/-- Theorem stating that min_cans is the minimum number of cans needed to provide at least a gallon of soda -/
theorem min_cans_correct : 
  (∀ n : ℕ, n * ounces_per_can ≥ ounces_per_gallon → n ≥ min_cans) ∧ 
  (min_cans * ounces_per_can ≥ ounces_per_gallon) :=
sorry

end NUMINAMATH_CALUDE_min_cans_correct_l297_29756


namespace NUMINAMATH_CALUDE_min_horses_oxen_solution_l297_29761

theorem min_horses_oxen_solution :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧
    344 * x - 265 * y = 33 ∧
    ∀ (x' y' : ℕ), x' > 0 → y' > 0 → 344 * x' - 265 * y' = 33 → x' ≥ x ∧ y' ≥ y :=
by
  -- The proof would go here
  sorry

#check min_horses_oxen_solution

end NUMINAMATH_CALUDE_min_horses_oxen_solution_l297_29761


namespace NUMINAMATH_CALUDE_total_cost_calculation_l297_29796

/-- The total cost of buying jerseys and basketballs -/
def total_cost (m n : ℝ) : ℝ := 8 * m + 5 * n

/-- Theorem: The total cost of buying 8 jerseys at m yuan each and 5 basketballs at n yuan each is 8m + 5n yuan -/
theorem total_cost_calculation (m n : ℝ) : 
  total_cost m n = 8 * m + 5 * n := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l297_29796


namespace NUMINAMATH_CALUDE_inequality_implication_l297_29763

theorem inequality_implication (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x - Real.log y > y - Real.log x → x - y > 1 / x - 1 / y := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l297_29763


namespace NUMINAMATH_CALUDE_line_through_two_points_line_with_special_intercepts_l297_29702

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Part 1
theorem line_through_two_points :
  ∃ (l : Line), pointOnLine l ⟨4, 1⟩ ∧ pointOnLine l ⟨-1, 6⟩ →
  l.slope * 1 + l.intercept = 5 :=
sorry

-- Part 2
theorem line_with_special_intercepts :
  ∃ (l : Line), pointOnLine l ⟨4, 1⟩ ∧ 
  (l.intercept = 2 * (- l.intercept / l.slope)) →
  (l.slope = 1/4 ∧ l.intercept = 0) ∨ (l.slope = -2 ∧ l.intercept = 9) :=
sorry

end NUMINAMATH_CALUDE_line_through_two_points_line_with_special_intercepts_l297_29702


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l297_29720

theorem solve_exponential_equation (x : ℝ) : 
  (12 : ℝ)^x * 6^4 / 432 = 432 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l297_29720


namespace NUMINAMATH_CALUDE_A_gives_B_150m_start_l297_29736

-- Define the speeds of runners A, B, and C
variable (Va Vb Vc : ℝ)

-- Define the conditions
def A_gives_C_300m_start : Prop := Va / Vc = 1000 / 700
def B_gives_C_176_47m_start : Prop := Vb / Vc = 1000 / 823.53

-- Define the theorem
theorem A_gives_B_150m_start 
  (h1 : A_gives_C_300m_start Va Vc) 
  (h2 : B_gives_C_176_47m_start Vb Vc) : 
  Va / Vb = 1000 / 850 := by sorry

end NUMINAMATH_CALUDE_A_gives_B_150m_start_l297_29736


namespace NUMINAMATH_CALUDE_jennifers_spending_l297_29751

theorem jennifers_spending (total : ℚ) (sandwich_frac : ℚ) (museum_frac : ℚ) (leftover : ℚ)
  (h1 : total = 150)
  (h2 : sandwich_frac = 1/5)
  (h3 : museum_frac = 1/6)
  (h4 : leftover = 20) :
  let spent_on_sandwich := total * sandwich_frac
  let spent_on_museum := total * museum_frac
  let total_spent := total - leftover
  let spent_on_book := total_spent - spent_on_sandwich - spent_on_museum
  spent_on_book / total = 1/2 := by
sorry

end NUMINAMATH_CALUDE_jennifers_spending_l297_29751


namespace NUMINAMATH_CALUDE_ancient_chinese_math_problem_l297_29790

/-- Represents the cost of animals in taels of silver -/
structure AnimalCost where
  cow : ℝ
  sheep : ℝ

/-- The total cost of a group of animals -/
def totalCost (c : AnimalCost) (numCows numSheep : ℕ) : ℝ :=
  c.cow * (numCows : ℝ) + c.sheep * (numSheep : ℝ)

/-- The theorem representing the ancient Chinese mathematical problem -/
theorem ancient_chinese_math_problem (c : AnimalCost) : 
  totalCost c 5 2 = 19 ∧ totalCost c 2 3 = 12 ↔ 
  (5 * c.cow + 2 * c.sheep = 19 ∧ 2 * c.cow + 3 * c.sheep = 12) := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_math_problem_l297_29790


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l297_29765

theorem simplify_complex_fraction (a b : ℝ) 
  (h1 : a ≠ b) (h2 : a ≠ -b) (h3 : a ≠ 2*b) : 
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l297_29765


namespace NUMINAMATH_CALUDE_inequality_equivalence_l297_29714

theorem inequality_equivalence (x : ℝ) : 
  |((x^2 + 2*x - 3) / 4)| ≤ 3 ↔ -5 ≤ x ∧ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l297_29714


namespace NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_abs_a_positive_l297_29742

theorem a_positive_sufficient_not_necessary_for_abs_a_positive :
  (∃ a : ℝ, a > 0 → abs a > 0) ∧ 
  (∃ a : ℝ, abs a > 0 ∧ ¬(a > 0)) :=
by sorry

end NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_abs_a_positive_l297_29742


namespace NUMINAMATH_CALUDE_arithmetic_arrangement_proof_l297_29715

theorem arithmetic_arrangement_proof :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∧
  ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_arrangement_proof_l297_29715


namespace NUMINAMATH_CALUDE_largest_n_value_l297_29757

/-- Represents a number in a given base -/
structure BaseRepresentation (base : ℕ) where
  digits : List ℕ
  valid : ∀ d ∈ digits, d < base

/-- The value of n in base 10 given its representation in another base -/
def toBase10 (base : ℕ) (repr : BaseRepresentation base) : ℕ :=
  repr.digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem largest_n_value (n : ℕ) 
  (base8_repr : BaseRepresentation 8)
  (base12_repr : BaseRepresentation 12)
  (h1 : toBase10 8 base8_repr = n)
  (h2 : toBase10 12 base12_repr = n)
  (h3 : base8_repr.digits.length = 3)
  (h4 : base12_repr.digits.length = 3)
  (h5 : base8_repr.digits.reverse = base12_repr.digits) :
  n ≤ 509 := by
  sorry

#check largest_n_value

end NUMINAMATH_CALUDE_largest_n_value_l297_29757


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_eight_l297_29792

theorem greatest_three_digit_divisible_by_eight :
  ∃ n : ℕ, n = 992 ∧ 
  n ≥ 100 ∧ n < 1000 ∧
  n % 8 = 0 ∧
  ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 8 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_eight_l297_29792


namespace NUMINAMATH_CALUDE_blue_candy_probability_l297_29727

def green_candies : ℕ := 5
def blue_candies : ℕ := 3
def red_candies : ℕ := 4

def total_candies : ℕ := green_candies + blue_candies + red_candies

theorem blue_candy_probability :
  (blue_candies : ℚ) / total_candies = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_blue_candy_probability_l297_29727


namespace NUMINAMATH_CALUDE_probability_exact_hits_l297_29771

def probability_single_hit : ℝ := 0.7
def total_shots : ℕ := 5
def desired_hits : ℕ := 2

theorem probability_exact_hits :
  let p := probability_single_hit
  let n := total_shots
  let k := desired_hits
  let q := 1 - p
  (Nat.choose n k : ℝ) * p ^ k * q ^ (n - k) = 0.1323 := by sorry

end NUMINAMATH_CALUDE_probability_exact_hits_l297_29771


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l297_29789

theorem quadratic_rewrite (x : ℝ) :
  ∃ (a b c : ℤ), 16 * x^2 - 40 * x + 24 = (a * x + b : ℝ)^2 + c ∧ a * b = -20 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l297_29789
