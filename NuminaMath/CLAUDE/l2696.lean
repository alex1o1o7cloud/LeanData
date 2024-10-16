import Mathlib

namespace NUMINAMATH_CALUDE_max_red_balls_l2696_269663

theorem max_red_balls (n : ℕ) : 
  (∃ y : ℕ, 
    n = 90 + 9 * y ∧
    (89 + 8 * y : ℚ) / (90 + 9 * y) ≥ 92 / 100 ∧
    ∀ m > n, (∃ z : ℕ, m = 90 + 9 * z) → 
      (89 + 8 * z : ℚ) / (90 + 9 * z) < 92 / 100) →
  n = 288 :=
sorry

end NUMINAMATH_CALUDE_max_red_balls_l2696_269663


namespace NUMINAMATH_CALUDE_sqrt_product_equals_six_l2696_269693

theorem sqrt_product_equals_six : Real.sqrt 8 * Real.sqrt (9/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_six_l2696_269693


namespace NUMINAMATH_CALUDE_apples_eaten_by_two_children_l2696_269619

/-- Proves that given 5 children who each collected 15 apples, if one child sold 7 apples
    and they had 60 apples left when they got home, then two children ate a total of 8 apples. -/
theorem apples_eaten_by_two_children
  (num_children : Nat)
  (apples_per_child : Nat)
  (apples_sold : Nat)
  (apples_left : Nat)
  (h1 : num_children = 5)
  (h2 : apples_per_child = 15)
  (h3 : apples_sold = 7)
  (h4 : apples_left = 60) :
  ∃ (eaten_by_two : Nat), eaten_by_two = 8 ∧
    num_children * apples_per_child = apples_left + apples_sold + eaten_by_two :=
by sorry


end NUMINAMATH_CALUDE_apples_eaten_by_two_children_l2696_269619


namespace NUMINAMATH_CALUDE_percentage_failed_both_l2696_269621

/-- The percentage of students who failed in Hindi -/
def failed_hindi : ℝ := 25

/-- The percentage of students who failed in English -/
def failed_english : ℝ := 35

/-- The percentage of students who passed in both subjects -/
def passed_both : ℝ := 80

/-- The theorem stating the percentage of students who failed in both subjects -/
theorem percentage_failed_both :
  100 - passed_both = failed_hindi + failed_english - 40 := by sorry

end NUMINAMATH_CALUDE_percentage_failed_both_l2696_269621


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2696_269647

/-- An isosceles triangle with two sides of 7 cm each and a perimeter of 23 cm has a base of 9 cm. -/
theorem isosceles_triangle_base_length : 
  ∀ (base : ℝ), 
    base > 0 → 
    7 + 7 + base = 23 → 
    base = 9 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2696_269647


namespace NUMINAMATH_CALUDE_sixth_term_term_1994_l2696_269679

-- Define the sequence
def a (n : ℕ+) : ℕ := n * (n + 1)

-- Theorem for the 6th term
theorem sixth_term : a 6 = 42 := by sorry

-- Theorem for the 1994th term
theorem term_1994 : a 1994 = 3978030 := by sorry

end NUMINAMATH_CALUDE_sixth_term_term_1994_l2696_269679


namespace NUMINAMATH_CALUDE_problem_solution_l2696_269697

theorem problem_solution (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 - a*b - b*c - c*d - d + 2/5 = 0) : 
  a = 1/5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2696_269697


namespace NUMINAMATH_CALUDE_product_difference_l2696_269653

theorem product_difference (A B : ℝ) 
  (h1 : (A + 2) * B = A * B + 60)
  (h2 : A * (B - 3) = A * B - 24) :
  (A + 2) * (B - 3) - A * B = 30 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_l2696_269653


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2696_269602

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log x}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x^2)}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2696_269602


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2696_269613

theorem inequality_and_equality_condition 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) 
  (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) 
  (hb₁ : b₁ > 0) (hb₂ : b₂ > 0) (hb₃ : b₃ > 0) : 
  (a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂)^2 ≥ 
    4 * (a₁ * a₂ + a₂ * a₃ + a₃ * a₁) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁) ∧ 
  ((a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂)^2 = 
    4 * (a₁ * a₂ + a₂ * a₃ + a₃ * a₁) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁) ↔ 
    a₁ * b₂ = a₂ * b₁ ∧ a₂ * b₃ = a₃ * b₂ ∧ a₃ * b₁ = a₁ * b₃) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2696_269613


namespace NUMINAMATH_CALUDE_total_full_price_tickets_is_16525_l2696_269643

/-- Represents the ticket sales data for a play over three weeks -/
structure PlayTicketSales where
  total_tickets : ℕ
  week1_tickets : ℕ
  week2_tickets : ℕ
  week3_tickets : ℕ
  week2_full_price_ratio : ℕ
  week3_full_price_ratio : ℕ

/-- Calculates the total number of full-price tickets sold during the play's run -/
def total_full_price_tickets (sales : PlayTicketSales) : ℕ :=
  let week2_full_price := sales.week2_tickets * sales.week2_full_price_ratio / (sales.week2_full_price_ratio + 1)
  let week3_full_price := sales.week3_tickets * sales.week3_full_price_ratio / (sales.week3_full_price_ratio + 1)
  week2_full_price + week3_full_price

/-- Theorem stating that given the specific ticket sales data, the total number of full-price tickets is 16525 -/
theorem total_full_price_tickets_is_16525 (sales : PlayTicketSales) 
  (h1 : sales.total_tickets = 25200)
  (h2 : sales.week1_tickets = 5400)
  (h3 : sales.week2_tickets = 7200)
  (h4 : sales.week3_tickets = 13400)
  (h5 : sales.week2_full_price_ratio = 2)
  (h6 : sales.week3_full_price_ratio = 7) :
  total_full_price_tickets sales = 16525 := by
  sorry


end NUMINAMATH_CALUDE_total_full_price_tickets_is_16525_l2696_269643


namespace NUMINAMATH_CALUDE_total_peppers_calculation_l2696_269626

/-- The amount of green peppers bought by Dale's Vegetarian Restaurant in pounds -/
def green_peppers : ℝ := 2.8333333333333335

/-- The amount of red peppers bought by Dale's Vegetarian Restaurant in pounds -/
def red_peppers : ℝ := 2.8333333333333335

/-- The total amount of peppers bought by Dale's Vegetarian Restaurant in pounds -/
def total_peppers : ℝ := green_peppers + red_peppers

theorem total_peppers_calculation :
  total_peppers = 5.666666666666667 := by sorry

end NUMINAMATH_CALUDE_total_peppers_calculation_l2696_269626


namespace NUMINAMATH_CALUDE_students_not_coming_to_class_l2696_269608

theorem students_not_coming_to_class (pieces_per_student : ℕ) 
  (total_pieces_last_monday : ℕ) (total_pieces_this_monday : ℕ) :
  pieces_per_student = 4 →
  total_pieces_last_monday = 40 →
  total_pieces_this_monday = 28 →
  total_pieces_last_monday / pieces_per_student - 
  total_pieces_this_monday / pieces_per_student = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_students_not_coming_to_class_l2696_269608


namespace NUMINAMATH_CALUDE_probability_expired_20_2_l2696_269695

/-- The probability of selecting an expired bottle from a set of bottles -/
def probability_expired (total : ℕ) (expired : ℕ) : ℚ :=
  (expired : ℚ) / (total : ℚ)

/-- Theorem: The probability of selecting an expired bottle from 20 bottles, where 2 are expired, is 1/10 -/
theorem probability_expired_20_2 :
  probability_expired 20 2 = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_expired_20_2_l2696_269695


namespace NUMINAMATH_CALUDE_solution_in_interval_l2696_269673

theorem solution_in_interval (x : ℝ) : 
  (Real.log x + x = 2) → (1.5 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l2696_269673


namespace NUMINAMATH_CALUDE_continuous_fraction_value_l2696_269699

theorem continuous_fraction_value :
  ∃ x : ℝ, x = 1 + 1 / (2 + 1 / x) ∧ x = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continuous_fraction_value_l2696_269699


namespace NUMINAMATH_CALUDE_debate_team_boys_l2696_269652

theorem debate_team_boys (total : ℕ) (girls : ℕ) (groups : ℕ) :
  total % 9 = 0 →
  total / 9 = groups →
  girls = 46 →
  groups = 8 →
  total - girls = 26 :=
by sorry

end NUMINAMATH_CALUDE_debate_team_boys_l2696_269652


namespace NUMINAMATH_CALUDE_opposite_of_fraction_l2696_269641

theorem opposite_of_fraction (n : ℕ) (h : n ≠ 0) : 
  (-(1 : ℚ) / n) = -((1 : ℚ) / n) := by sorry

end NUMINAMATH_CALUDE_opposite_of_fraction_l2696_269641


namespace NUMINAMATH_CALUDE_olivia_chocolate_sales_l2696_269603

/-- The amount of money Olivia made selling chocolate bars -/
def olivia_money (total_bars : ℕ) (unsold_bars : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

theorem olivia_chocolate_sales : olivia_money 7 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_olivia_chocolate_sales_l2696_269603


namespace NUMINAMATH_CALUDE_simple_interest_time_period_l2696_269664

/-- Proves that given a principal of 6000, if increasing the interest rate by 2%
    results in 360 more interest over the same time period, then the time period is 3 years. -/
theorem simple_interest_time_period 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : principal = 6000)
  (h2 : principal * (rate + 2) / 100 * time = principal * rate / 100 * time + 360) :
  time = 3 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_time_period_l2696_269664


namespace NUMINAMATH_CALUDE_pairing_theorem_l2696_269690

/-- Represents the pairing of boys and girls in the school event. -/
structure Pairing where
  boys : ℕ
  girls : ℕ
  first_pairing : ℕ := 3
  pairing_increment : ℕ := 2

/-- The relationship between boys and girls in the pairing. -/
def pairing_relationship (p : Pairing) : Prop :=
  p.boys = (p.girls - 1) / 2

/-- Theorem stating the relationship between boys and girls in the pairing. -/
theorem pairing_theorem (p : Pairing) : pairing_relationship p := by
  sorry

#check pairing_theorem

end NUMINAMATH_CALUDE_pairing_theorem_l2696_269690


namespace NUMINAMATH_CALUDE_math_problems_l2696_269646

theorem math_problems :
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a*b ∧ a*b > b^2) ∧
  (∀ a b c d : ℝ, c > d ∧ a > b → a - d > b - c) ∧
  (∀ a b c : ℝ, b < a ∧ a < 0 ∧ c < 0 → c/a > c/b) ∧
  (∀ a b c : ℝ, a > 0 ∧ b > c ∧ c > 0 → (c+a)/(b+a) > c/b) :=
by sorry

end NUMINAMATH_CALUDE_math_problems_l2696_269646


namespace NUMINAMATH_CALUDE_unique_square_friendly_l2696_269658

/-- A number is a perfect square if it's equal to some integer squared. -/
def IsPerfectSquare (n : ℤ) : Prop := ∃ k : ℤ, n = k^2

/-- An integer c is square-friendly if for all integers m, m^2 + 18m + c is a perfect square. -/
def IsSquareFriendly (c : ℤ) : Prop :=
  ∀ m : ℤ, IsPerfectSquare (m^2 + 18*m + c)

/-- Theorem: 81 is the only square-friendly integer. -/
theorem unique_square_friendly : ∃! c : ℤ, IsSquareFriendly c ∧ c = 81 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_friendly_l2696_269658


namespace NUMINAMATH_CALUDE_train_length_calculation_l2696_269683

/-- Calculates the length of two trains given their speeds and overtaking time -/
theorem train_length_calculation (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 72) :
  let v_rel := v_fast - v_slow
  let d := v_rel * t * (1000 / 3600)
  let L := d / 2
  L = 100 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2696_269683


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2696_269680

theorem point_in_fourth_quadrant (α : Real) (h : -π/2 < α ∧ α < 0) :
  let P : ℝ × ℝ := (Real.tan α, Real.cos α)
  P.1 < 0 ∧ P.2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2696_269680


namespace NUMINAMATH_CALUDE_number_is_composite_l2696_269630

theorem number_is_composite (n : ℕ) (h : n = 2^1000) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 10^n + 1 = a * b :=
sorry

end NUMINAMATH_CALUDE_number_is_composite_l2696_269630


namespace NUMINAMATH_CALUDE_probability_eliminate_six_eq_seven_twentysix_l2696_269685

/-- Represents a team in the tournament -/
structure Team :=
  (players : ℕ)

/-- Represents the tournament structure -/
structure Tournament :=
  (teamA : Team)
  (teamB : Team)

/-- Calculates the binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Calculates the probability of one team eliminating exactly 6 players before winning -/
def probability_eliminate_six (t : Tournament) : ℚ :=
  if t.teamA.players = 7 ∧ t.teamB.players = 7 then
    (binomial 12 6 : ℚ) / (2 * (binomial 13 7 : ℚ))
  else
    0

/-- Theorem stating the probability of eliminating 6 players before winning -/
theorem probability_eliminate_six_eq_seven_twentysix (t : Tournament) :
  t.teamA.players = 7 ∧ t.teamB.players = 7 →
  probability_eliminate_six t = 7 / 26 :=
sorry

end NUMINAMATH_CALUDE_probability_eliminate_six_eq_seven_twentysix_l2696_269685


namespace NUMINAMATH_CALUDE_joe_video_game_spending_l2696_269684

/-- Joe's video game spending problem -/
theorem joe_video_game_spending
  (initial_money : ℕ)
  (selling_price : ℕ)
  (months : ℕ)
  (h1 : initial_money = 240)
  (h2 : selling_price = 30)
  (h3 : months = 12)
  : ∃ (monthly_spending : ℕ),
    monthly_spending = 50 ∧
    initial_money = months * monthly_spending - months * selling_price :=
by sorry

end NUMINAMATH_CALUDE_joe_video_game_spending_l2696_269684


namespace NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l2696_269681

-- Define the equations
def equation1 (x : ℝ) : Prop := (2*x + 9) / (3 - x) = (4*x - 7) / (x - 3)
def equation2 (x : ℝ) : Prop := (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1

-- Theorem for equation1
theorem equation1_solution :
  ∃! x : ℝ, equation1 x ∧ x ≠ 3 := by sorry

-- Theorem for equation2
theorem equation2_no_solution :
  ¬∃ x : ℝ, equation2 x ∧ x ≠ 1 ∧ x ≠ -1 := by sorry

end NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l2696_269681


namespace NUMINAMATH_CALUDE_geometric_progression_value_l2696_269687

def is_geometric_progression (a b c : ℝ) : Prop :=
  b ^ 2 = a * c

theorem geometric_progression_value :
  ∃ x : ℝ, is_geometric_progression (30 + x) (70 + x) (150 + x) ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_value_l2696_269687


namespace NUMINAMATH_CALUDE_table_problem_l2696_269674

theorem table_problem :
  (∀ x : ℤ, (2 * x - 7 : ℤ) = -5 ↔ x = 1) ∧
  (∀ x : ℤ, (-3 * x - 1 : ℤ) = 5 ↔ x = -2) ∧
  (∀ x : ℤ, (3 * x + 2 : ℤ) - (-2 * x + 5) = 7 ↔ x = 2) ∧
  (∀ m n : ℤ, (∀ x : ℤ, (m * (x + 1) + n : ℤ) - (m * x + n) = -4) →
              (m * 3 + n : ℤ) = -5 →
              (m * 7 + n : ℤ) = -21) :=
by sorry

end NUMINAMATH_CALUDE_table_problem_l2696_269674


namespace NUMINAMATH_CALUDE_equation_solution_l2696_269659

theorem equation_solution : 
  ∃ n : ℚ, (3 - n) / (n + 2) + (3 * n - 9) / (3 - n) = 2 ∧ n = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2696_269659


namespace NUMINAMATH_CALUDE_quadratic_inequality_min_value_l2696_269692

theorem quadratic_inequality_min_value (a b : ℝ) (h1 : a > b)
  (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (h3 : ∃ x0 : ℝ, a * x0^2 + 2 * x0 + b = 0) :
  (∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2) ∧
  (∃ x : ℝ, (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_min_value_l2696_269692


namespace NUMINAMATH_CALUDE_absolute_value_sum_l2696_269654

theorem absolute_value_sum (a b : ℝ) : a^2 + b^2 > 1 → |a| + |b| > 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l2696_269654


namespace NUMINAMATH_CALUDE_no_perfect_square_exists_l2696_269657

theorem no_perfect_square_exists (a : ℕ) : 
  (∃ k : ℕ, ((a^2 - 3)^3 + 1)^a - 1 = k^2) → False ∧
  (∃ k : ℕ, ((a^2 - 3)^3 + 1)^(a+1) - 1 = k^2) → False :=
by sorry

end NUMINAMATH_CALUDE_no_perfect_square_exists_l2696_269657


namespace NUMINAMATH_CALUDE_length_in_cube4_is_4root3_l2696_269655

/-- The length of the portion of the line segment from (0,0,0) to (5,5,11) 
    contained in the cube with edge length 4, which extends from (0,0,5) to (4,4,9) -/
def lengthInCube4 : ℝ := sorry

/-- The coordinates of the entry point of the line segment into the cube with edge length 4 -/
def entryPoint : Fin 3 → ℝ
| 0 => 0
| 1 => 0
| 2 => 5

/-- The coordinates of the exit point of the line segment from the cube with edge length 4 -/
def exitPoint : Fin 3 → ℝ
| 0 => 4
| 1 => 4
| 2 => 9

theorem length_in_cube4_is_4root3 : lengthInCube4 = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_length_in_cube4_is_4root3_l2696_269655


namespace NUMINAMATH_CALUDE_tshirt_original_price_l2696_269605

/-- Proves that the original price of a t-shirt is $20 given the conditions of the problem -/
theorem tshirt_original_price 
  (num_friends : ℕ) 
  (discount_percent : ℚ) 
  (total_spent : ℚ) : 
  num_friends = 4 → 
  discount_percent = 1/2 → 
  total_spent = 40 → 
  (total_spent / num_friends) / (1 - discount_percent) = 20 := by
sorry

end NUMINAMATH_CALUDE_tshirt_original_price_l2696_269605


namespace NUMINAMATH_CALUDE_flagpole_height_l2696_269622

-- Define the given conditions
def flagpoleShadowLength : ℝ := 45
def buildingShadowLength : ℝ := 65
def buildingHeight : ℝ := 26

-- Define the theorem
theorem flagpole_height :
  ∃ (h : ℝ), h / flagpoleShadowLength = buildingHeight / buildingShadowLength ∧ h = 18 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_height_l2696_269622


namespace NUMINAMATH_CALUDE_triangle_tan_A_l2696_269662

theorem triangle_tan_A (A B C : ℝ) (AB BC : ℝ) 
  (h_angle : A = π/3)
  (h_AB : AB = 20)
  (h_BC : BC = 21) : 
  Real.tan A = (21 * Real.sqrt 3) / (2 * Real.sqrt (421 - 1323/4)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_tan_A_l2696_269662


namespace NUMINAMATH_CALUDE_new_person_weight_example_l2696_269612

/-- Calculates the weight of a new person given the initial number of persons,
    the average weight increase, and the weight of the replaced person. -/
def new_person_weight (initial_count : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * avg_increase

theorem new_person_weight_example :
  new_person_weight 7 6.2 76 = 119.4 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_example_l2696_269612


namespace NUMINAMATH_CALUDE_slower_time_is_692_l2696_269688

/-- The number of stories in the building -/
def num_stories : ℕ := 50

/-- The time Lola takes to run up one story (in seconds) -/
def lola_time_per_story : ℕ := 12

/-- The time the elevator takes to go up one story (in seconds) -/
def elevator_time_per_story : ℕ := 10

/-- The time the elevator stops on each floor (in seconds) -/
def elevator_stop_time : ℕ := 4

/-- The number of floors where the elevator stops -/
def num_elevator_stops : ℕ := num_stories - 2

/-- The time Lola takes to reach the top floor -/
def lola_total_time : ℕ := num_stories * lola_time_per_story

/-- The time Tara takes to reach the top floor -/
def tara_total_time : ℕ := num_stories * elevator_time_per_story + num_elevator_stops * elevator_stop_time

theorem slower_time_is_692 : max lola_total_time tara_total_time = 692 := by sorry

end NUMINAMATH_CALUDE_slower_time_is_692_l2696_269688


namespace NUMINAMATH_CALUDE_sin_squared_50_over_1_plus_sin_10_l2696_269669

theorem sin_squared_50_over_1_plus_sin_10 :
  (Real.sin (50 * π / 180))^2 / (1 + Real.sin (10 * π / 180)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_50_over_1_plus_sin_10_l2696_269669


namespace NUMINAMATH_CALUDE_twenty_three_in_base_two_l2696_269636

theorem twenty_three_in_base_two : 
  (23 : ℕ) = 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 := by
  sorry

end NUMINAMATH_CALUDE_twenty_three_in_base_two_l2696_269636


namespace NUMINAMATH_CALUDE_marie_erasers_l2696_269631

/-- Given that Marie starts with 95 erasers and loses 42, prove that she ends with 53 erasers. -/
theorem marie_erasers : 
  let initial_erasers : ℕ := 95
  let lost_erasers : ℕ := 42
  initial_erasers - lost_erasers = 53 := by sorry

end NUMINAMATH_CALUDE_marie_erasers_l2696_269631


namespace NUMINAMATH_CALUDE_score_not_above_average_l2696_269651

structure ClassData where
  participants : ℕ
  mean : ℝ
  median : ℝ
  mode : ℝ
  variance : ℝ
  excellenceRate : ℝ

def class901 : ClassData :=
  { participants := 40
  , mean := 75
  , median := 78
  , mode := 77
  , variance := 158
  , excellenceRate := 0.2 }

def class902 : ClassData :=
  { participants := 45
  , mean := 75
  , median := 76
  , mode := 74
  , variance := 122
  , excellenceRate := 0.2 }

theorem score_not_above_average (score : ℝ) :
  score = 77 → ¬(score > class902.mean) := by
  sorry

end NUMINAMATH_CALUDE_score_not_above_average_l2696_269651


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2696_269615

theorem arithmetic_sequence_third_term
  (a : ℤ) (d : ℤ) -- First term and common difference
  (h1 : a + 14 * d = 14) -- 15th term is 14
  (h2 : a + 15 * d = 17) -- 16th term is 17
  : a + 2 * d = -22 := -- 3rd term is -22
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2696_269615


namespace NUMINAMATH_CALUDE_work_completion_time_l2696_269644

/-- Given that Ravi can do a piece of work in 15 days and Prakash can do it in 30 days,
    prove that they will finish it together in 10 days. -/
theorem work_completion_time (ravi_time prakash_time : ℝ) (h1 : ravi_time = 15) (h2 : prakash_time = 30) :
  1 / (1 / ravi_time + 1 / prakash_time) = 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2696_269644


namespace NUMINAMATH_CALUDE_fraction_simplification_l2696_269617

theorem fraction_simplification : (4 / 252 : ℚ) + (17 / 36 : ℚ) = 41 / 84 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2696_269617


namespace NUMINAMATH_CALUDE_smallest_a_value_l2696_269638

theorem smallest_a_value (a b : ℕ) (r₁ r₂ r₃ : ℕ+) : 
  (∀ x : ℝ, x^3 - a*x^2 + b*x - 2310 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  r₁ * r₂ * r₃ = 2310 →
  a = r₁ + r₂ + r₃ →
  28 ≤ a :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2696_269638


namespace NUMINAMATH_CALUDE_total_wash_time_l2696_269689

def wash_time_normal : ℕ := 4 + 7 + 4 + 9

def wash_time_suv : ℕ := 2 * wash_time_normal

def wash_time_minivan : ℕ := (3 * wash_time_normal) / 2

def num_normal_cars : ℕ := 3

def num_suvs : ℕ := 2

def num_minivans : ℕ := 1

def break_time : ℕ := 5

def total_vehicles : ℕ := num_normal_cars + num_suvs + num_minivans

theorem total_wash_time : 
  num_normal_cars * wash_time_normal + 
  num_suvs * wash_time_suv + 
  num_minivans * wash_time_minivan + 
  (total_vehicles - 1) * break_time = 229 := by
  sorry

end NUMINAMATH_CALUDE_total_wash_time_l2696_269689


namespace NUMINAMATH_CALUDE_tree_growth_relation_l2696_269637

/-- The height of a tree after a number of months -/
def tree_height (initial_height growth_rate : ℝ) (months : ℝ) : ℝ :=
  initial_height + growth_rate * months

/-- Theorem: The height of the tree after x months is 80 + 2x -/
theorem tree_growth_relation (x : ℝ) :
  tree_height 80 2 x = 80 + 2 * x := by sorry

end NUMINAMATH_CALUDE_tree_growth_relation_l2696_269637


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2696_269625

theorem triangle_abc_properties (A B C : Real) (a b c : Real) (S : Real) :
  -- Given conditions
  (3 * Real.cos B * Real.cos C + 2 = 3 * Real.sin B * Real.sin C + 2 * Real.cos (2 * A)) →
  (S = 5 * Real.sqrt 3) →
  (b = 5) →
  -- Triangle inequality and positive side lengths
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Angle sum in a triangle
  (A + B + C = Real.pi) →
  -- Area formula
  (S = 1/2 * b * c * Real.sin A) →
  -- Conclusions
  (A = Real.pi / 3 ∧ Real.sin B * Real.sin C = 5 / 7) := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l2696_269625


namespace NUMINAMATH_CALUDE_simplify_fraction_l2696_269666

theorem simplify_fraction : 20 * (9 / 14) * (1 / 18) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2696_269666


namespace NUMINAMATH_CALUDE_product_xy_l2696_269635

theorem product_xy (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = 190 / 9 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_l2696_269635


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l2696_269616

theorem cube_plus_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 3) : 
  r^3 + 1/r^3 = 0 := by sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l2696_269616


namespace NUMINAMATH_CALUDE_sin_45_degrees_l2696_269607

theorem sin_45_degrees : Real.sin (π / 4) = 1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l2696_269607


namespace NUMINAMATH_CALUDE_may_day_travelers_l2696_269624

def scientific_notation (n : ℕ) (c : ℝ) (e : ℤ) : Prop :=
  (1 ≤ c) ∧ (c < 10) ∧ (n = c * (10 : ℝ) ^ e)

theorem may_day_travelers :
  scientific_notation 213000000 2.13 8 :=
by sorry

end NUMINAMATH_CALUDE_may_day_travelers_l2696_269624


namespace NUMINAMATH_CALUDE_expression_evaluation_l2696_269696

theorem expression_evaluation (a b : ℝ) 
  (h : (a + 1/2)^2 + |b - 2| = 0) : 
  5*(3*a^2*b - a*b^2) - (a*b^2 + 3*a^2*b) = 18 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2696_269696


namespace NUMINAMATH_CALUDE_complex_magnitude_l2696_269606

/-- Given a complex number z = (3+i)/(1+2i), prove that its magnitude |z| is equal to √2 -/
theorem complex_magnitude (z : ℂ) : z = (3 + I) / (1 + 2*I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2696_269606


namespace NUMINAMATH_CALUDE_total_spent_on_cards_l2696_269649

-- Define the prices and tax rates
def football_card_price : ℝ := 2.73
def football_card_tax_rate : ℝ := 0.05
def football_card_quantity : ℕ := 2

def pokemon_card_price : ℝ := 4.01
def pokemon_card_tax_rate : ℝ := 0.08

def baseball_card_original_price : ℝ := 10
def baseball_card_discount_rate : ℝ := 0.10
def baseball_card_tax_rate : ℝ := 0.06

-- Calculate the total cost
def total_cost : ℝ :=
  -- Football cards
  (football_card_price * football_card_quantity) * (1 + football_card_tax_rate) +
  -- Pokemon cards
  pokemon_card_price * (1 + pokemon_card_tax_rate) +
  -- Baseball cards
  (baseball_card_original_price * (1 - baseball_card_discount_rate)) * (1 + baseball_card_tax_rate)

-- Theorem statement
theorem total_spent_on_cards :
  total_cost = 19.6038 := by sorry

end NUMINAMATH_CALUDE_total_spent_on_cards_l2696_269649


namespace NUMINAMATH_CALUDE_work_days_calculation_l2696_269677

/-- Represents the number of days worked by each person -/
structure WorkDays where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the daily wages of each person -/
structure DailyWages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The main theorem statement -/
theorem work_days_calculation 
  (work_days : WorkDays)
  (daily_wages : DailyWages)
  (total_earning : ℕ)
  (h1 : work_days.a = 6)
  (h2 : work_days.c = 4)
  (h3 : daily_wages.a * 4 = daily_wages.b * 3)
  (h4 : daily_wages.b * 5 = daily_wages.c * 4)
  (h5 : daily_wages.c = 95)
  (h6 : work_days.a * daily_wages.a + work_days.b * daily_wages.b + work_days.c * daily_wages.c = total_earning)
  (h7 : total_earning = 1406)
  : work_days.b = 9 := by
  sorry


end NUMINAMATH_CALUDE_work_days_calculation_l2696_269677


namespace NUMINAMATH_CALUDE_percentage_calculation_l2696_269633

theorem percentage_calculation (x y : ℝ) (h : x = 875.3 ∧ y = 318.65) : 
  (y / x) * 100 = 36.4 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2696_269633


namespace NUMINAMATH_CALUDE_molecular_weight_AlPO4_l2696_269609

/-- The atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of Phosphorus in g/mol -/
def P_weight : ℝ := 30.97

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The number of Oxygen atoms in AlPO4 -/
def O_count : ℕ := 4

/-- The molecular weight of AlPO4 in g/mol -/
def AlPO4_weight : ℝ := Al_weight + P_weight + O_count * O_weight

/-- The number of moles of AlPO4 -/
def moles : ℕ := 4

/-- Theorem stating the molecular weight of 4 moles of AlPO4 -/
theorem molecular_weight_AlPO4 : moles * AlPO4_weight = 487.80 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_AlPO4_l2696_269609


namespace NUMINAMATH_CALUDE_student_takehome_pay_l2696_269618

/-- Calculates the take-home pay for a well-performing student at a fast-food chain --/
def takehomePay (baseSalary bonus taxRate : ℚ) : ℚ :=
  let totalEarnings := baseSalary + bonus
  let incomeTax := totalEarnings * taxRate
  totalEarnings - incomeTax

/-- Theorem: The take-home pay for a well-performing student is 26,100 rubles --/
theorem student_takehome_pay :
  takehomePay 25000 5000 (13/100) = 26100 := by
  sorry

#eval takehomePay 25000 5000 (13/100)

end NUMINAMATH_CALUDE_student_takehome_pay_l2696_269618


namespace NUMINAMATH_CALUDE_prob_sum_seven_l2696_269628

/-- A type representing the possible outcomes of a single dice throw -/
inductive DiceOutcome
  | one
  | two
  | three
  | four
  | five
  | six

/-- The type of outcomes when throwing a dice twice -/
def TwoThrows := DiceOutcome × DiceOutcome

/-- The sum of points for a pair of dice outcomes -/
def sum_points (throw : TwoThrows) : Nat :=
  match throw with
  | (DiceOutcome.one, b) => 1 + DiceOutcome.toNat b
  | (DiceOutcome.two, b) => 2 + DiceOutcome.toNat b
  | (DiceOutcome.three, b) => 3 + DiceOutcome.toNat b
  | (DiceOutcome.four, b) => 4 + DiceOutcome.toNat b
  | (DiceOutcome.five, b) => 5 + DiceOutcome.toNat b
  | (DiceOutcome.six, b) => 6 + DiceOutcome.toNat b
where
  DiceOutcome.toNat : DiceOutcome → Nat
    | DiceOutcome.one => 1
    | DiceOutcome.two => 2
    | DiceOutcome.three => 3
    | DiceOutcome.four => 4
    | DiceOutcome.five => 5
    | DiceOutcome.six => 6

/-- The set of all possible outcomes when throwing a dice twice -/
def all_outcomes : Finset TwoThrows := sorry

/-- The set of outcomes where the sum of points is 7 -/
def sum_seven_outcomes : Finset TwoThrows := sorry

theorem prob_sum_seven : 
  (Finset.card sum_seven_outcomes : ℚ) / (Finset.card all_outcomes : ℚ) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_sum_seven_l2696_269628


namespace NUMINAMATH_CALUDE_new_capacity_is_250_l2696_269694

/-- Calculates the new combined total lifting capacity after improvements -/
def new_total_capacity (initial_clean_jerk : ℝ) (initial_snatch : ℝ) : ℝ :=
  (2 * initial_clean_jerk) + (initial_snatch + 0.8 * initial_snatch)

/-- Theorem stating that given the initial capacities and improvements, 
    the new total capacity is 250 kg -/
theorem new_capacity_is_250 :
  new_total_capacity 80 50 = 250 := by
  sorry

end NUMINAMATH_CALUDE_new_capacity_is_250_l2696_269694


namespace NUMINAMATH_CALUDE_sum_product_equality_l2696_269661

theorem sum_product_equality (x y z : ℝ) 
  (hx : |x| ≠ 1/Real.sqrt 3) 
  (hy : |y| ≠ 1/Real.sqrt 3) 
  (hz : |z| ≠ 1/Real.sqrt 3) 
  (h : x + y + z = x * y * z) : 
  (3*x - x^3)/(1-3*x^2) + (3*y - y^3)/(1-3*y^2) + (3*z - z^3)/(1-3*z^2) = 
  (3*x - x^3)/(1-3*x^2) * (3*y - y^3)/(1-3*y^2) * (3*z - z^3)/(1-3*z^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l2696_269661


namespace NUMINAMATH_CALUDE_circle_equation_alternatives_l2696_269682

/-- A circle with center on the y-axis, radius 5, passing through (3, -4) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_y_axis : center.1 = 0
  radius_is_5 : radius = 5
  passes_through_point : (center.1 - 3)^2 + (center.2 - (-4))^2 = radius^2

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_equation_alternatives (c : Circle) :
  (∀ x y, circle_equation c x y ↔ x^2 + y^2 = 25) ∨
  (∀ x y, circle_equation c x y ↔ x^2 + (y + 8)^2 = 25) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_alternatives_l2696_269682


namespace NUMINAMATH_CALUDE_initial_people_is_ten_l2696_269686

/-- Represents the job completion scenario with given conditions -/
structure JobCompletion where
  initialDays : ℕ
  initialWorkDone : ℚ
  daysBeforeFiring : ℕ
  peopleFired : ℕ
  remainingDays : ℕ

/-- Calculates the initial number of people hired -/
def initialPeopleHired (job : JobCompletion) : ℕ :=
  sorry

/-- Theorem stating that the initial number of people hired is 10 -/
theorem initial_people_is_ten (job : JobCompletion) 
  (h1 : job.initialDays = 100)
  (h2 : job.initialWorkDone = 1/4)
  (h3 : job.daysBeforeFiring = 20)
  (h4 : job.peopleFired = 2)
  (h5 : job.remainingDays = 75) :
  initialPeopleHired job = 10 :=
sorry

end NUMINAMATH_CALUDE_initial_people_is_ten_l2696_269686


namespace NUMINAMATH_CALUDE_smallest_positive_root_of_f_l2696_269656

open Real

theorem smallest_positive_root_of_f (f : ℝ → ℝ) :
  (∀ x, f x = sin x + 2 * cos x + 3 * tan x) →
  (∃ x ∈ Set.Ioo 3 4, f x = 0) ∧
  (∀ x ∈ Set.Ioo 0 3, f x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_root_of_f_l2696_269656


namespace NUMINAMATH_CALUDE_dice_probability_l2696_269604

def number_of_dice : ℕ := 8
def probability_even : ℚ := 1/2
def probability_odd : ℚ := 1/2

theorem dice_probability :
  (number_of_dice.choose (number_of_dice / 2)) * 
  (probability_even ^ (number_of_dice / 2)) * 
  (probability_odd ^ (number_of_dice / 2)) = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l2696_269604


namespace NUMINAMATH_CALUDE_symmetry_and_periodicity_l2696_269600

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define even function property
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem symmetry_and_periodicity 
  (h1 : is_even (fun x ↦ f (x - 1)))
  (h2 : is_even (fun x ↦ f (x - 2))) :
  (∀ x, f (-x - 2) = f x) ∧ 
  (∀ x, f (x + 2) = f x) ∧
  (∀ x, f' (-x + 4) = f' x) :=
sorry

end NUMINAMATH_CALUDE_symmetry_and_periodicity_l2696_269600


namespace NUMINAMATH_CALUDE_probability_is_three_fourths_l2696_269642

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The probability that a point (x, y) satisfies x + 2y < 4 when randomly and uniformly chosen from the given square --/
def probabilityLessThan4 (s : Square) : ℝ :=
  sorry

/-- The square with vertices (0, 0), (0, 3), (3, 3), and (3, 0) --/
def givenSquare : Square :=
  { bottomLeft := (0, 0), topRight := (3, 3) }

theorem probability_is_three_fourths :
  probabilityLessThan4 givenSquare = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_three_fourths_l2696_269642


namespace NUMINAMATH_CALUDE_agent_commission_l2696_269645

def commission_rate : ℝ := 0.025
def sales : ℝ := 840

theorem agent_commission :
  sales * commission_rate = 21 := by sorry

end NUMINAMATH_CALUDE_agent_commission_l2696_269645


namespace NUMINAMATH_CALUDE_angle_inequality_l2696_269648

theorem angle_inequality (θ : Real) : 
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^3 * Real.cos θ - x^2 * (1 - x) + (1 - x)^3 * Real.sin θ > 0) ↔
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) := by
sorry

end NUMINAMATH_CALUDE_angle_inequality_l2696_269648


namespace NUMINAMATH_CALUDE_line_segments_in_proportion_l2696_269614

theorem line_segments_in_proportion (a b c d : ℝ) : 
  a = 5 ∧ b = 15 ∧ c = 3 ∧ d = 9 → a * d = b * c := by
  sorry

end NUMINAMATH_CALUDE_line_segments_in_proportion_l2696_269614


namespace NUMINAMATH_CALUDE_perpendicular_preservation_l2696_269634

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_preservation 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : perpendicular m α) 
  (h4 : parallel_lines m n) 
  (h5 : parallel_planes α β) : 
  perpendicular n β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_preservation_l2696_269634


namespace NUMINAMATH_CALUDE_division_equals_500_l2696_269671

theorem division_equals_500 : (35 : ℝ) / 0.07 = 500 := by
  sorry

end NUMINAMATH_CALUDE_division_equals_500_l2696_269671


namespace NUMINAMATH_CALUDE_measure_one_kg_cereal_l2696_269623

/-- Represents a balance scale that may be inaccurate -/
structure BalanceScale where
  isBalanced : (ℝ → ℝ → Prop)

/-- Represents a bag of cereal -/
def CerealBag : Type := ℝ

/-- Represents a correct 1 kg weight -/
def CorrectWeight : ℝ := 1

/-- Function to measure cereal using the balance scale and correct weight -/
def measureCereal (scale : BalanceScale) (bag : CerealBag) (weight : ℝ) : Prop :=
  ∃ (amount : ℝ), 
    scale.isBalanced amount weight ∧ 
    scale.isBalanced amount amount ∧ 
    amount = weight

/-- Theorem stating that it's possible to measure 1 kg of cereal -/
theorem measure_one_kg_cereal 
  (scale : BalanceScale) 
  (bag : CerealBag) : 
  measureCereal scale bag CorrectWeight := by
  sorry


end NUMINAMATH_CALUDE_measure_one_kg_cereal_l2696_269623


namespace NUMINAMATH_CALUDE_slope_of_line_l2696_269650

/-- The slope of the line 6x + 10y = 30 is -3/5 -/
theorem slope_of_line (x y : ℝ) : 6 * x + 10 * y = 30 → (y - 3 = (-3 / 5) * (x - 0)) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2696_269650


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2696_269698

theorem condition_sufficient_not_necessary (a b : ℝ) :
  (∀ a b, (a - b)^3 * b^2 > 0 → a > b) ∧
  (∃ a b, a > b ∧ (a - b)^3 * b^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2696_269698


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l2696_269627

def f (x : ℝ) := 2 * abs (x + 1) + abs (x - 2)

theorem min_value_and_inequality :
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ y : ℝ, f y = m) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 →
    b^2 / a + c^2 / b + a^2 / c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l2696_269627


namespace NUMINAMATH_CALUDE_probability_three_quarters_l2696_269667

/-- A diamond-shaped checkerboard formed by an 8x8 grid -/
structure DiamondCheckerboard where
  total_squares : ℕ
  squares_per_vertex : ℕ
  num_vertices : ℕ

/-- The probability that a randomly chosen unit square does not touch a vertex of the diamond -/
def probability_not_touching_vertex (board : DiamondCheckerboard) : ℚ :=
  1 - (board.squares_per_vertex * board.num_vertices : ℚ) / board.total_squares

/-- Theorem stating that the probability of not touching a vertex is 3/4 -/
theorem probability_three_quarters (board : DiamondCheckerboard) 
  (h1 : board.total_squares = 64)
  (h2 : board.squares_per_vertex = 4)
  (h3 : board.num_vertices = 4) : 
  probability_not_touching_vertex board = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_quarters_l2696_269667


namespace NUMINAMATH_CALUDE_beverage_selection_probabilities_l2696_269632

def total_cups : ℕ := 5
def type_a_cups : ℕ := 3
def type_b_cups : ℕ := 2
def cups_to_select : ℕ := 3

def probability_all_correct : ℚ := 1 / 10
def probability_at_least_two_correct : ℚ := 7 / 10

theorem beverage_selection_probabilities :
  (total_cups = type_a_cups + type_b_cups) →
  (cups_to_select = type_a_cups) →
  (probability_all_correct = 1 / (Nat.choose total_cups cups_to_select)) ∧
  (probability_at_least_two_correct = 
    (Nat.choose type_a_cups cups_to_select + 
     Nat.choose type_a_cups (cups_to_select - 1) * Nat.choose type_b_cups 1) / 
    (Nat.choose total_cups cups_to_select)) := by
  sorry

end NUMINAMATH_CALUDE_beverage_selection_probabilities_l2696_269632


namespace NUMINAMATH_CALUDE_ashley_champagne_bottles_l2696_269670

/-- The number of bottles of champagne needed for a wedding toast --/
def bottles_needed (guests : ℕ) (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) : ℕ :=
  (guests * glasses_per_guest) / servings_per_bottle

/-- Theorem: Ashley needs 40 bottles of champagne for her wedding toast --/
theorem ashley_champagne_bottles : 
  bottles_needed 120 2 6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ashley_champagne_bottles_l2696_269670


namespace NUMINAMATH_CALUDE_intersection_range_intersection_length_l2696_269676

-- Define the hyperbola and line
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the intersection condition
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  (k > -Real.sqrt 2 ∧ k < -1) ∨ (k > -1 ∧ k < 1) ∨ (k > 1 ∧ k < Real.sqrt 2)

-- Define the midpoint condition
def midpoint_condition (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂ ∧
    (x₁ + x₂) / 2 = Real.sqrt 2

-- Theorem 1: Range of k for two distinct intersections
theorem intersection_range :
  ∀ k : ℝ, intersects_at_two_points k ↔ k_range k := by sorry

-- Theorem 2: Length of AB when midpoint x-coordinate is √2
theorem intersection_length :
  ∀ k : ℝ, midpoint_condition k → 
    ∃ x₁ y₁ x₂ y₂ : ℝ, hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂ ∧
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6 := by sorry

end NUMINAMATH_CALUDE_intersection_range_intersection_length_l2696_269676


namespace NUMINAMATH_CALUDE_equation_solution_l2696_269620

theorem equation_solution : 
  ∃! x : ℝ, x > 0 ∧ 7.61 * Real.log 3 / Real.log 2 + 2 * Real.log x / Real.log 4 = x^(Real.log 16 / Real.log 9 / (Real.log x / Real.log 3)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2696_269620


namespace NUMINAMATH_CALUDE_apples_given_to_larry_l2696_269675

/-- Given that Joyce starts with 75 apples and ends up with 23 apples,
    prove that she gave 52 apples to Larry. -/
theorem apples_given_to_larry (initial : ℕ) (final : ℕ) (given : ℕ) :
  initial = 75 →
  final = 23 →
  given = initial - final →
  given = 52 := by sorry

end NUMINAMATH_CALUDE_apples_given_to_larry_l2696_269675


namespace NUMINAMATH_CALUDE_spade_example_l2696_269665

-- Define the spade operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_example : spade 3 (spade 5 8) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spade_example_l2696_269665


namespace NUMINAMATH_CALUDE_expression_evaluation_l2696_269610

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := 1
  3 * x^2 + 2 * x * y - 4 * y^2 - 2 * (3 * y^2 + x * y - x^2) = -35/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2696_269610


namespace NUMINAMATH_CALUDE_wheel_spinner_probability_wheel_spinner_probability_proof_l2696_269691

theorem wheel_spinner_probability : Real → Real → Real → Real → Prop :=
  fun prob_E prob_F prob_G prob_H =>
    prob_E = 1/2 →
    prob_F = 1/4 →
    prob_G = 2 * prob_H →
    prob_E + prob_F + prob_G + prob_H = 1 →
    prob_G = 1/6

-- The proof is omitted
theorem wheel_spinner_probability_proof : wheel_spinner_probability (1/2) (1/4) (1/6) (1/12) := by
  sorry

end NUMINAMATH_CALUDE_wheel_spinner_probability_wheel_spinner_probability_proof_l2696_269691


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2696_269678

theorem initial_money_calculation (initial_amount : ℚ) : 
  (initial_amount * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = 500) → 
  initial_amount = 1250 := by
sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l2696_269678


namespace NUMINAMATH_CALUDE_power_function_through_point_l2696_269660

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 27 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2696_269660


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_one_l2696_269611

theorem sum_of_roots_equals_one :
  let f : ℝ → ℝ := λ x ↦ (x + 3) * (x - 4) - 20
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_one_l2696_269611


namespace NUMINAMATH_CALUDE_union_of_sets_l2696_269629

def A (a b : ℝ) : Set ℝ := {5, b/a, a-b}
def B (a b : ℝ) : Set ℝ := {b, a+b, -1}

theorem union_of_sets (a b : ℝ) (h : A a b ∩ B a b = {2, -1}) :
  A a b ∪ B a b = {-1, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2696_269629


namespace NUMINAMATH_CALUDE_ball_bounce_count_l2696_269668

/-- The number of bounces required for a ball dropped from 16 feet to reach a height less than 2 feet,
    when it bounces back up two-thirds the distance it just fell. -/
theorem ball_bounce_count : ∃ k : ℕ, 
  (∀ n < k, 16 * (2/3)^n ≥ 2) ∧ 
  16 * (2/3)^k < 2 ∧
  k = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_count_l2696_269668


namespace NUMINAMATH_CALUDE_niles_collection_l2696_269601

/-- The total amount collected by Niles from the book club -/
def total_collected (num_members : ℕ) (snack_fee : ℕ) (num_hardcover : ℕ) (hardcover_price : ℕ) (num_paperback : ℕ) (paperback_price : ℕ) : ℕ :=
  num_members * (snack_fee + num_hardcover * hardcover_price + num_paperback * paperback_price)

/-- Theorem stating the total amount collected by Niles -/
theorem niles_collection : total_collected 6 150 6 30 6 12 = 2412 := by
  sorry

end NUMINAMATH_CALUDE_niles_collection_l2696_269601


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2696_269672

theorem quadratic_equation_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 3*x + k = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁*x₂ + 2*x₁ + 2*x₂ = 1) →
  k = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2696_269672


namespace NUMINAMATH_CALUDE_arithmetic_sum_special_case_l2696_269640

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ d n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sum_special_case (k : ℕ) :
  arithmetic_sum (k^2 - k + 1) 1 (2*k + 1) = (2*k + 1) * (k^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_special_case_l2696_269640


namespace NUMINAMATH_CALUDE_f_one_geq_25_l2696_269639

def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem f_one_geq_25 (m : ℝ) (h : ∀ x ≥ -2, Monotone (f m)) :
  f m 1 ≥ 25 := by sorry

end NUMINAMATH_CALUDE_f_one_geq_25_l2696_269639
