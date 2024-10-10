import Mathlib

namespace trigonometric_inequality_l3887_388736

theorem trigonometric_inequality (a b α β : ℝ) 
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : 0 ≤ b ∧ b ≤ 1) 
  (h3 : 0 ≤ α ∧ α ≤ Real.pi / 2) 
  (h4 : 0 ≤ β ∧ β ≤ Real.pi / 2) 
  (h5 : a * b * Real.cos (α - β) ≤ Real.sqrt ((1 - a^2) * (1 - b^2))) :
  a * Real.cos α + b * Real.sin β ≤ 1 + a * b * Real.sin (β - α) := by
  sorry

end trigonometric_inequality_l3887_388736


namespace smallest_n_l3887_388704

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_n : ∃ (n : ℕ), n > 0 ∧ 
  is_factor 25 (n * 2^5 * 6^2 * 7^3) ∧ 
  is_factor 27 (n * 2^5 * 6^2 * 7^3) ∧
  (∀ (m : ℕ), m > 0 → 
    is_factor 25 (m * 2^5 * 6^2 * 7^3) → 
    is_factor 27 (m * 2^5 * 6^2 * 7^3) → 
    m ≥ n) ∧
  n = 75 := by
  sorry

end smallest_n_l3887_388704


namespace algebraic_identities_l3887_388730

theorem algebraic_identities :
  -- Part 1
  (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3) = 6 ∧
  -- Part 2
  (Real.sqrt 5 - Real.sqrt 6)^2 - (Real.sqrt 5 + Real.sqrt 6)^2 = -4 * Real.sqrt 30 ∧
  -- Part 3
  (2 * Real.sqrt (3/2) - Real.sqrt (1/2)) * (1/2 * Real.sqrt 8 + Real.sqrt (2/3)) = 5/3 * Real.sqrt 3 + 1 :=
by sorry

end algebraic_identities_l3887_388730


namespace four_special_numbers_exist_l3887_388757

theorem four_special_numbers_exist : ∃ (a b c d : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (¬(2 ∣ a) ∧ ¬(3 ∣ a) ∧ ¬(4 ∣ a)) ∧
  (¬(2 ∣ b) ∧ ¬(3 ∣ b) ∧ ¬(4 ∣ b)) ∧
  (¬(2 ∣ c) ∧ ¬(3 ∣ c) ∧ ¬(4 ∣ c)) ∧
  (¬(2 ∣ d) ∧ ¬(3 ∣ d) ∧ ¬(4 ∣ d)) ∧
  (2 ∣ (a + b)) ∧ (2 ∣ (a + c)) ∧ (2 ∣ (a + d)) ∧
  (2 ∣ (b + c)) ∧ (2 ∣ (b + d)) ∧ (2 ∣ (c + d)) ∧
  (3 ∣ (a + b + c)) ∧ (3 ∣ (a + b + d)) ∧
  (3 ∣ (a + c + d)) ∧ (3 ∣ (b + c + d)) ∧
  (4 ∣ (a + b + c + d)) := by
  sorry

#check four_special_numbers_exist

end four_special_numbers_exist_l3887_388757


namespace tank_problem_solution_l3887_388732

def tank_problem (initial_capacity : ℝ) (initial_loss_rate : ℝ) (initial_loss_time : ℝ)
  (second_loss_time : ℝ) (fill_rate : ℝ) (fill_time : ℝ) (final_missing : ℝ)
  (second_loss_rate : ℝ) : Prop :=
  let remaining_after_first_loss := initial_capacity - initial_loss_rate * initial_loss_time
  let remaining_after_second_loss := remaining_after_first_loss - second_loss_rate * second_loss_time
  let final_amount := remaining_after_second_loss + fill_rate * fill_time
  final_amount = initial_capacity - final_missing

theorem tank_problem_solution :
  tank_problem 350000 32000 5 10 40000 3 140000 10000 := by
  sorry

end tank_problem_solution_l3887_388732


namespace cos_double_angle_proof_l3887_388797

theorem cos_double_angle_proof (α : ℝ) (a : ℝ × ℝ) : 
  a = (Real.cos α, (1 : ℝ) / 2) → 
  Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = Real.sqrt 2 / 2 → 
  Real.cos (2 * α) = -(1 : ℝ) / 2 := by
sorry

end cos_double_angle_proof_l3887_388797


namespace max_p_value_l3887_388761

/-- The maximum value of p for two rectangular boxes with given conditions -/
theorem max_p_value (m n p : ℕ+) (h1 : m ≤ n) (h2 : n ≤ p)
  (h3 : 2 * (m * n * p) = (m + 2) * (n + 2) * (p + 2)) : 
  p ≤ 130 := by
sorry

end max_p_value_l3887_388761


namespace supremum_of_expression_is_zero_l3887_388714

open Real

theorem supremum_of_expression_is_zero :
  ∀ ε > 0, ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  x * y * z * (x + y + z) / (x + y + z)^3 < ε :=
by sorry

end supremum_of_expression_is_zero_l3887_388714


namespace triangle_inequality_l3887_388744

theorem triangle_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hSum : A + B + C = π) : 
  Real.tan (A/2)^2 + Real.tan (B/2)^2 + Real.tan (C/2)^2 + 
  8 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) ≥ 2 := by
  sorry

end triangle_inequality_l3887_388744


namespace travel_time_A_l3887_388721

/-- The time it takes for A to travel 60 miles given the conditions -/
theorem travel_time_A (y : ℝ) 
  (h1 : y > 0) -- B's speed is positive
  (h2 : (60 / y) - (60 / (y + 2)) = 3/4) -- Time difference equation
  : 60 / (y + 2) = 30/7 := by
sorry

end travel_time_A_l3887_388721


namespace penny_halfDollar_same_probability_l3887_388747

/-- Represents the outcome of a single coin flip -/
inductive CoinSide
| Heads
| Tails

/-- Represents the outcome of flipping six different coins -/
structure SixCoinFlip :=
  (penny : CoinSide)
  (nickel : CoinSide)
  (dime : CoinSide)
  (quarter : CoinSide)
  (halfDollar : CoinSide)
  (dollar : CoinSide)

/-- The set of all possible outcomes when flipping six coins -/
def allOutcomes : Finset SixCoinFlip := sorry

/-- The set of outcomes where the penny and half-dollar show the same side -/
def sameOutcomes : Finset SixCoinFlip := sorry

/-- The probability of an event occurring is the number of favorable outcomes
    divided by the total number of possible outcomes -/
def probability (event : Finset SixCoinFlip) : Rat :=
  (event.card : Rat) / (allOutcomes.card : Rat)

theorem penny_halfDollar_same_probability :
  probability sameOutcomes = 1/2 := by sorry

end penny_halfDollar_same_probability_l3887_388747


namespace school_furniture_prices_l3887_388779

/-- The price of a table in yuan -/
def table_price : ℕ := 36

/-- The price of a chair in yuan -/
def chair_price : ℕ := 9

/-- The total cost of 2 tables and 3 chairs in yuan -/
def total_cost : ℕ := 99

theorem school_furniture_prices :
  (2 * table_price + 3 * chair_price = total_cost) ∧
  (table_price = 4 * chair_price) ∧
  (table_price = 36) ∧
  (chair_price = 9) := by
  sorry

end school_furniture_prices_l3887_388779


namespace age_sum_problem_l3887_388771

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 162 → a + b + c = 20 := by
  sorry

end age_sum_problem_l3887_388771


namespace garage_wheel_count_l3887_388716

/-- Calculates the total number of wheels in a garage given the quantities of various vehicles --/
def total_wheels (bicycles cars tricycles single_axle_trailers double_axle_trailers eighteen_wheelers : ℕ) : ℕ :=
  bicycles * 2 + cars * 4 + tricycles * 3 + single_axle_trailers * 2 + double_axle_trailers * 4 + eighteen_wheelers * 18

/-- Proves that the total number of wheels in the garage is 97 --/
theorem garage_wheel_count :
  total_wheels 5 12 3 2 2 1 = 97 := by
  sorry

end garage_wheel_count_l3887_388716


namespace willson_work_hours_l3887_388715

theorem willson_work_hours : 
  let monday : ℚ := 3/4
  let tuesday : ℚ := 1/2
  let wednesday : ℚ := 2/3
  let thursday : ℚ := 5/6
  let friday : ℚ := 75/60
  monday + tuesday + wednesday + thursday + friday = 4 := by
sorry

end willson_work_hours_l3887_388715


namespace investment_proof_l3887_388748

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof of the investment problem -/
theorem investment_proof :
  let initial_investment : ℝ := 400
  let interest_rate : ℝ := 0.12
  let time_period : ℕ := 5
  let final_balance : ℝ := 705.03
  compound_interest initial_investment interest_rate time_period = final_balance :=
by
  sorry


end investment_proof_l3887_388748


namespace terms_before_one_l3887_388758

/-- An arithmetic sequence with first term 100 and common difference -3 -/
def arithmeticSequence : ℕ → ℤ := λ n => 100 - 3 * (n - 1)

/-- The position of 1 in the sequence -/
def positionOfOne : ℕ := 34

theorem terms_before_one :
  (∀ k < positionOfOne, arithmeticSequence k > 1) ∧
  arithmeticSequence positionOfOne = 1 ∧
  positionOfOne - 1 = 33 := by sorry

end terms_before_one_l3887_388758


namespace fourth_person_height_l3887_388767

def height_problem (h₁ h₂ h₃ h₄ : ℝ) : Prop :=
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ ∧
  h₂ - h₁ = 2 ∧
  h₃ - h₂ = 2 ∧
  h₄ - h₃ = 6 ∧
  (h₁ + h₂ + h₃ + h₄) / 4 = 76

theorem fourth_person_height 
  (h₁ h₂ h₃ h₄ : ℝ) 
  (h : height_problem h₁ h₂ h₃ h₄) : 
  h₄ = 82 := by
  sorry

end fourth_person_height_l3887_388767


namespace two_tangents_iff_m_gt_two_l3887_388775

/-- The circle equation: x^2 + y^2 + mx + 1 = 0 -/
def circle_equation (x y m : ℝ) : Prop := x^2 + y^2 + m*x + 1 = 0

/-- The point A -/
def point_A : ℝ × ℝ := (1, 0)

/-- Condition for two tangents to be drawn from a point to a circle -/
def two_tangents_condition (m : ℝ) : Prop :=
  let center := (-m/2, 0)
  let radius_squared := m^2/4 - 1
  let distance_squared := (point_A.1 - center.1)^2 + (point_A.2 - center.2)^2
  distance_squared > radius_squared ∧ radius_squared > 0

theorem two_tangents_iff_m_gt_two :
  ∀ m : ℝ, two_tangents_condition m ↔ m > 2 :=
sorry

end two_tangents_iff_m_gt_two_l3887_388775


namespace smallest_x_with_remainders_l3887_388759

theorem smallest_x_with_remainders : ∃! x : ℕ+, 
  (x : ℤ) % 3 = 2 ∧ 
  (x : ℤ) % 4 = 3 ∧ 
  (x : ℤ) % 5 = 4 ∧
  ∀ y : ℕ+, 
    (y : ℤ) % 3 = 2 → 
    (y : ℤ) % 4 = 3 → 
    (y : ℤ) % 5 = 4 → 
    x ≤ y :=
by
  -- Proof goes here
  sorry

end smallest_x_with_remainders_l3887_388759


namespace distribute_6_balls_4_boxes_l3887_388705

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 9 ways to distribute 6 indistinguishable balls into 4 indistinguishable boxes -/
theorem distribute_6_balls_4_boxes : distribute_balls 6 4 = 9 := by sorry

end distribute_6_balls_4_boxes_l3887_388705


namespace sum_greater_than_product_l3887_388756

theorem sum_greater_than_product (a b : ℕ+) : a + b > a * b ↔ a = 1 ∨ b = 1 := by
  sorry

end sum_greater_than_product_l3887_388756


namespace x_intercept_of_line_l3887_388737

/-- The x-intercept of the line 5y - 7x = 35 is (-5, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  5 * y - 7 * x = 35 → y = 0 → x = -5 := by
  sorry

end x_intercept_of_line_l3887_388737


namespace smallest_sum_of_sequence_l3887_388738

/-- Given positive integers A, B, C, and integer D, where A, B, C form an arithmetic sequence,
    B, C, D form a geometric sequence, and C/B = 7/3, the smallest possible value of A + B + C + D is 76. -/
theorem smallest_sum_of_sequence (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →
  (∃ r : ℤ, C - B = B - A) →  -- arithmetic sequence condition
  (∃ q : ℚ, C = B * q ∧ D = C * q) →  -- geometric sequence condition
  C = (7 * B) / 3 →
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 →
    (∃ r : ℤ, C' - B' = B' - A') →
    (∃ q : ℚ, C' = B' * q ∧ D' = C' * q) →
    C' = (7 * B') / 3 →
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 76 := by
sorry

end smallest_sum_of_sequence_l3887_388738


namespace simplify_expression_1_simplify_expression_2_l3887_388729

-- First expression
theorem simplify_expression_1 (x : ℝ) : 2 * x - 3 * (x - 1) = 3 - x := by sorry

-- Second expression
theorem simplify_expression_2 (a b : ℝ) : 
  6 * (a * b^2 - a^2 * b) - 2 * (3 * a^2 * b + a * b^2) = 4 * a * b^2 - 12 * a^2 * b := by sorry

end simplify_expression_1_simplify_expression_2_l3887_388729


namespace cupcake_distribution_l3887_388726

theorem cupcake_distribution (total_cupcakes : ℕ) (num_children : ℕ) 
  (h1 : total_cupcakes = 96) 
  (h2 : num_children = 8) 
  (h3 : total_cupcakes % num_children = 0) : 
  total_cupcakes / num_children = 12 := by
  sorry

#check cupcake_distribution

end cupcake_distribution_l3887_388726


namespace businessmen_neither_coffee_nor_tea_l3887_388766

theorem businessmen_neither_coffee_nor_tea
  (total : ℕ)
  (coffee : ℕ)
  (tea : ℕ)
  (both : ℕ)
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 12)
  (h4 : both = 6) :
  total - (coffee + tea - both) = 9 :=
by sorry

end businessmen_neither_coffee_nor_tea_l3887_388766


namespace prism_volume_l3887_388799

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (side_area front_area bottom_area : ℝ) 
  (h_side : side_area = 20)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 8) :
  ∃ (x y z : ℝ), x * y = side_area ∧ y * z = front_area ∧ x * z = bottom_area ∧ 
  x * y * z = 20 * Real.sqrt 4.8 :=
by sorry

end prism_volume_l3887_388799


namespace nina_shells_to_liam_l3887_388796

theorem nina_shells_to_liam (oliver liam nina : ℕ) 
  (h1 : liam = 3 * oliver) 
  (h2 : nina = 4 * liam) 
  (h3 : oliver > 0) : 
  (nina - (oliver + liam + nina) / 3) / nina = 7 / 36 := by
  sorry

end nina_shells_to_liam_l3887_388796


namespace arithmetic_sequence_m_value_l3887_388731

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_m_value
  (a : ℕ → ℝ)
  (m : ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_roots : (a 2)^2 + m * (a 2) - 8 = 0 ∧ (a 8)^2 + m * (a 8) - 8 = 0)
  (h_sum : a 4 + a 6 = (a 5)^2 + 1) :
  m = -2 := by
sorry

end arithmetic_sequence_m_value_l3887_388731


namespace red_star_selection_probability_l3887_388720

/-- The probability of selecting a specific book from a set of books -/
def probability_of_selection (total_books : ℕ) (target_books : ℕ) : ℚ :=
  target_books / total_books

/-- Theorem: The probability of selecting "The Red Star Shines Over China" from 4 books is 1/4 -/
theorem red_star_selection_probability :
  probability_of_selection 4 1 = 1 / 4 := by
  sorry

end red_star_selection_probability_l3887_388720


namespace unique_solution_trigonometric_equation_l3887_388784

theorem unique_solution_trigonometric_equation :
  ∃! (n : ℕ), n > 0 ∧ Real.sin (π / (3 * n)) + Real.cos (π / (3 * n)) = Real.sqrt (2 * n) / 3 :=
by sorry

end unique_solution_trigonometric_equation_l3887_388784


namespace ceiling_floor_sum_l3887_388707

theorem ceiling_floor_sum : ⌈(7:ℚ)/3⌉ + ⌊-(7:ℚ)/3⌋ = 0 := by sorry

end ceiling_floor_sum_l3887_388707


namespace mark_amy_age_difference_mark_amy_age_difference_proof_l3887_388712

theorem mark_amy_age_difference : ℕ → Prop :=
  fun age_difference =>
    ∃ (mark_current_age amy_current_age : ℕ),
      amy_current_age = 15 ∧
      mark_current_age + 5 = 27 ∧
      mark_current_age - amy_current_age = age_difference ∧
      age_difference = 7

-- The proof is omitted
theorem mark_amy_age_difference_proof : mark_amy_age_difference 7 := by
  sorry

end mark_amy_age_difference_mark_amy_age_difference_proof_l3887_388712


namespace women_average_age_is_23_l3887_388741

/-- The average age of two women given the conditions of the problem -/
def average_age_of_women (initial_men_count : ℕ) 
                         (age_increase : ℕ) 
                         (replaced_man1_age : ℕ) 
                         (replaced_man2_age : ℕ) : ℚ :=
  let total_age_increase := initial_men_count * age_increase
  let total_women_age := total_age_increase + replaced_man1_age + replaced_man2_age
  total_women_age / 2

/-- Theorem stating that the average age of the women is 23 years -/
theorem women_average_age_is_23 : 
  average_age_of_women 8 2 20 10 = 23 := by
  sorry

end women_average_age_is_23_l3887_388741


namespace percentage_votes_against_l3887_388752

/-- Given a total number of votes and the difference between votes in favor and against,
    calculate the percentage of votes against the proposal. -/
theorem percentage_votes_against (total_votes : ℕ) (favor_minus_against : ℕ) 
    (h1 : total_votes = 340)
    (h2 : favor_minus_against = 68) : 
    (total_votes - favor_minus_against) / 2 / total_votes * 100 = 40 := by
  sorry

#check percentage_votes_against

end percentage_votes_against_l3887_388752


namespace smallest_common_multiple_of_8_and_6_l3887_388706

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ+, (∀ m : ℕ+, 8 ∣ m ∧ 6 ∣ m → n ≤ m) ∧ 8 ∣ n ∧ 6 ∣ n := by
  sorry

end smallest_common_multiple_of_8_and_6_l3887_388706


namespace polynomial_factor_theorem_l3887_388764

theorem polynomial_factor_theorem (c : ℚ) : 
  (∀ x : ℚ, (x + 7) ∣ (c * x^3 + 19 * x^2 - c * x - 49)) → c = 21/8 := by
  sorry

end polynomial_factor_theorem_l3887_388764


namespace train_journey_time_l3887_388778

theorem train_journey_time 
  (distance : ℝ) 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) :
  speed1 = 48 →
  speed2 = 60 →
  time2 = 2/3 →
  distance = speed1 * time1 →
  distance = speed2 * time2 →
  time1 = 5/6 :=
by sorry

end train_journey_time_l3887_388778


namespace cases_in_1995_l3887_388788

/-- Calculates the number of cases in a given year assuming a linear decrease --/
def casesInYear (initialYear initialCases finalYear finalCases targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let targetYearsSinceStart := targetYear - initialYear
  let decrease := (targetYearsSinceStart * totalDecrease) / totalYears
  initialCases - decrease

/-- Theorem stating that the number of cases in 1995 is 263,125 --/
theorem cases_in_1995 : 
  casesInYear 1970 700000 2010 1000 1995 = 263125 := by
  sorry

#eval casesInYear 1970 700000 2010 1000 1995

end cases_in_1995_l3887_388788


namespace arrangement_schemes_l3887_388735

theorem arrangement_schemes (teachers students : ℕ) (h1 : teachers = 2) (h2 : students = 4) : 
  (teachers.choose 1) * (students.choose 2) = 12 := by
  sorry

end arrangement_schemes_l3887_388735


namespace root_sum_theorem_l3887_388770

theorem root_sum_theorem (x : ℝ) (a b c d : ℝ) : 
  (1/x + 1/(x+4) - 1/(x+6) - 1/(x+10) + 1/(x+12) + 1/(x+16) - 1/(x+18) - 1/(x+20) = 0) →
  (∃ (sign1 sign2 : Bool), x = -a + (-1)^(sign1.toNat : ℕ) * Real.sqrt (b + (-1)^(sign2.toNat : ℕ) * c * Real.sqrt d)) →
  a + b + c + d = 27 := by
  sorry

end root_sum_theorem_l3887_388770


namespace fraction_enlargement_l3887_388745

theorem fraction_enlargement (x y : ℝ) (h : 3 * x - y ≠ 0) :
  (2 * (3 * x) * (3 * y)) / (3 * (3 * x) - 3 * y) = 3 * ((2 * x * y) / (3 * x - y)) :=
by sorry

end fraction_enlargement_l3887_388745


namespace smallest_common_multiple_lcm_14_10_smallest_number_of_students_l3887_388724

theorem smallest_common_multiple (n : ℕ) : n > 0 ∧ 14 ∣ n ∧ 10 ∣ n → n ≥ 70 := by
  sorry

theorem lcm_14_10 : Nat.lcm 14 10 = 70 := by
  sorry

theorem smallest_number_of_students : ∃ (n : ℕ), n > 0 ∧ 14 ∣ n ∧ 10 ∣ n ∧ ∀ (m : ℕ), (m > 0 ∧ 14 ∣ m ∧ 10 ∣ m) → n ≤ m := by
  sorry

end smallest_common_multiple_lcm_14_10_smallest_number_of_students_l3887_388724


namespace max_odd_digits_on_board_l3887_388768

/-- A function that counts the number of odd digits in a natural number -/
def countOddDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 10 digits -/
def hasTenDigits (n : ℕ) : Prop := sorry

theorem max_odd_digits_on_board (a b : ℕ) (h1 : hasTenDigits a) (h2 : hasTenDigits b) :
  countOddDigits a + countOddDigits b + countOddDigits (a + b) ≤ 30 ∧
  ∃ (a' b' : ℕ), hasTenDigits a' ∧ hasTenDigits b' ∧
    countOddDigits a' + countOddDigits b' + countOddDigits (a' + b') = 30 :=
sorry

end max_odd_digits_on_board_l3887_388768


namespace rhombus_always_symmetrical_triangle_not_always_symmetrical_parallelogram_not_always_symmetrical_trapezoid_not_always_symmetrical_l3887_388790

-- Define the basic shapes
inductive Shape
  | Triangle
  | Parallelogram
  | Rhombus
  | Trapezoid

-- Define a property for symmetry
def isSymmetrical (s : Shape) : Prop :=
  match s with
  | Shape.Rhombus => True
  | _ => false

-- Theorem stating that only Rhombus is always symmetrical
theorem rhombus_always_symmetrical :
  ∀ (s : Shape), isSymmetrical s ↔ s = Shape.Rhombus :=
by sorry

-- Additional theorems to show that other shapes are not always symmetrical
theorem triangle_not_always_symmetrical :
  ∃ (t : Shape), t = Shape.Triangle ∧ ¬(isSymmetrical t) :=
by sorry

theorem parallelogram_not_always_symmetrical :
  ∃ (p : Shape), p = Shape.Parallelogram ∧ ¬(isSymmetrical p) :=
by sorry

theorem trapezoid_not_always_symmetrical :
  ∃ (t : Shape), t = Shape.Trapezoid ∧ ¬(isSymmetrical t) :=
by sorry

end rhombus_always_symmetrical_triangle_not_always_symmetrical_parallelogram_not_always_symmetrical_trapezoid_not_always_symmetrical_l3887_388790


namespace train_speed_l3887_388751

/-- Calculate the speed of a train given its length, platform length, and time to cross -/
theorem train_speed (train_length platform_length : ℝ) (time : ℝ) : 
  train_length = 250 →
  platform_length = 520 →
  time = 50.395968322534195 →
  ∃ (speed : ℝ), abs (speed - 54.99) < 0.01 ∧ 
    speed = (train_length + platform_length) / time * 3.6 := by
  sorry

end train_speed_l3887_388751


namespace gcd_18_30_l3887_388740

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l3887_388740


namespace distance_between_points_l3887_388701

/-- The distance between two points in 3D space is the square root of the sum of the squares of the differences of their coordinates. -/
theorem distance_between_points (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) :
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2) = Real.sqrt 185 ↔
  x₁ = -2 ∧ y₁ = 4 ∧ z₁ = 1 ∧ x₂ = 3 ∧ y₂ = -8 ∧ z₂ = 5 := by
  sorry

end distance_between_points_l3887_388701


namespace min_value_problem_l3887_388787

theorem min_value_problem (a b m n : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_sum : a + b = 1) (h_prod : m * n = 2) :
  (a * m + b * n) * (b * m + a * n) ≥ 2 := by
  sorry

end min_value_problem_l3887_388787


namespace hundredth_odd_and_following_even_l3887_388781

theorem hundredth_odd_and_following_even :
  (∃ n : ℕ, n = 100 ∧ 2 * n - 1 = 199) ∧
  (∃ m : ℕ, m = 200 ∧ m = 199 + 1 ∧ Even m) := by
  sorry

end hundredth_odd_and_following_even_l3887_388781


namespace sum_faces_edges_vertices_l3887_388739

/-- A rectangular prism is a three-dimensional shape with specific properties. -/
structure RectangularPrism where
  -- We don't need to define the specific properties here

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (rp : RectangularPrism) : ℕ := 8

/-- The sum of faces, edges, and vertices in a rectangular prism is 26 -/
theorem sum_faces_edges_vertices (rp : RectangularPrism) :
  num_faces rp + num_edges rp + num_vertices rp = 26 := by
  sorry

end sum_faces_edges_vertices_l3887_388739


namespace m_one_sufficient_not_necessary_l3887_388763

def z1 (m : ℝ) : ℂ := Complex.mk (m^2 + m + 1) (m^2 + m - 4)
def z2 : ℂ := Complex.mk 3 (-2)

theorem m_one_sufficient_not_necessary :
  (∃ m : ℝ, z1 m = z2 ∧ m ≠ 1) ∧ (z1 1 = z2) := by sorry

end m_one_sufficient_not_necessary_l3887_388763


namespace perfect_square_condition_l3887_388765

theorem perfect_square_condition (x : ℝ) :
  (∃ a : ℤ, 4 * x^5 - 7 = a^2) ∧ 
  (∃ b : ℤ, 4 * x^13 - 7 = b^2) → 
  x = 2 :=
by sorry

end perfect_square_condition_l3887_388765


namespace optimal_rental_plan_l3887_388750

/-- Represents the capacity and cost of different car types -/
structure CarType where
  capacity : ℕ
  cost : ℕ

/-- Represents a rental plan -/
structure RentalPlan where
  type_a : ℕ
  type_b : ℕ

/-- Calculates the total capacity of a rental plan -/
def total_capacity (plan : RentalPlan) (a : CarType) (b : CarType) : ℕ :=
  plan.type_a * a.capacity + plan.type_b * b.capacity

/-- Calculates the total cost of a rental plan -/
def total_cost (plan : RentalPlan) (a : CarType) (b : CarType) : ℕ :=
  plan.type_a * a.cost + plan.type_b * b.cost

/-- Checks if a rental plan is valid for the given total goods -/
def is_valid_plan (plan : RentalPlan) (a : CarType) (b : CarType) (total_goods : ℕ) : Prop :=
  total_capacity plan a b = total_goods

/-- Theorem: The optimal rental plan for transporting 27 tons of goods is 1 type A car and 6 type B cars, with a total cost of 820 yuan -/
theorem optimal_rental_plan :
  ∃ (a b : CarType) (optimal_plan : RentalPlan),
    -- Given conditions
    (2 * a.capacity + 3 * b.capacity = 18) ∧
    (a.capacity + 2 * b.capacity = 11) ∧
    (a.cost = 100) ∧
    (b.cost = 120) ∧
    -- Optimal plan
    (optimal_plan.type_a = 1) ∧
    (optimal_plan.type_b = 6) ∧
    -- Plan is valid
    (is_valid_plan optimal_plan a b 27) ∧
    -- Plan is optimal (minimum cost)
    (∀ (plan : RentalPlan),
      is_valid_plan plan a b 27 →
      total_cost optimal_plan a b ≤ total_cost plan a b) ∧
    -- Total cost is 820 yuan
    (total_cost optimal_plan a b = 820) :=
  sorry

end optimal_rental_plan_l3887_388750


namespace existence_of_special_number_l3887_388798

theorem existence_of_special_number (P : Finset Nat) (h_prime : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : Nat,
    (∀ p ∈ P, ∃ a b : Nat, x = a^p + b^p) ∧
    (∀ p : Nat, Nat.Prime p → p ∉ P → ¬∃ a b : Nat, x = a^p + b^p) := by
  sorry

end existence_of_special_number_l3887_388798


namespace syllogism_form_is_correct_l3887_388709

-- Define deductive reasoning
structure DeductiveReasoning where
  general_to_specific : Bool
  syllogism_form : Bool
  conclusion_correctness : Bool
  conclusion_depends_on_premises : Bool

-- Define the correct properties of deductive reasoning
def correct_deductive_reasoning : DeductiveReasoning :=
  { general_to_specific := true,
    syllogism_form := true,
    conclusion_correctness := false,
    conclusion_depends_on_premises := true }

-- Theorem to prove
theorem syllogism_form_is_correct (dr : DeductiveReasoning) :
  dr = correct_deductive_reasoning → dr.syllogism_form = true :=
by sorry

end syllogism_form_is_correct_l3887_388709


namespace difference_max_min_change_l3887_388760

def initial_yes : ℝ := 40
def initial_no : ℝ := 30
def initial_maybe : ℝ := 30
def final_yes : ℝ := 60
def final_no : ℝ := 20
def final_maybe : ℝ := 20

def min_change : ℝ := 20
def max_change : ℝ := 40

theorem difference_max_min_change :
  max_change - min_change = 20 :=
sorry

end difference_max_min_change_l3887_388760


namespace butter_mixture_profit_percentage_l3887_388773

/-- Calculates the profit percentage for a mixture of butter sold at a certain price -/
theorem butter_mixture_profit_percentage
  (weight1 : ℝ) (price1 : ℝ) (weight2 : ℝ) (price2 : ℝ) (selling_price : ℝ)
  (h1 : weight1 = 34)
  (h2 : price1 = 150)
  (h3 : weight2 = 36)
  (h4 : price2 = 125)
  (h5 : selling_price = 192) :
  let total_cost := weight1 * price1 + weight2 * price2
  let total_weight := weight1 + weight2
  let cost_price_per_kg := total_cost / total_weight
  let profit_percentage := (selling_price - cost_price_per_kg) / cost_price_per_kg * 100
  ∃ ε > 0, abs (profit_percentage - 40) < ε :=
by sorry


end butter_mixture_profit_percentage_l3887_388773


namespace total_value_of_coins_l3887_388794

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a half dollar in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The number of quarters found -/
def num_quarters : ℕ := 14

/-- The number of dimes found -/
def num_dimes : ℕ := 7

/-- The number of nickels found -/
def num_nickels : ℕ := 9

/-- The number of pennies found -/
def num_pennies : ℕ := 13

/-- The number of half dollars found -/
def num_half_dollars : ℕ := 4

/-- The total value of the coins found -/
theorem total_value_of_coins : 
  (num_quarters : ℚ) * quarter_value + 
  (num_dimes : ℚ) * dime_value + 
  (num_nickels : ℚ) * nickel_value + 
  (num_pennies : ℚ) * penny_value + 
  (num_half_dollars : ℚ) * half_dollar_value = 6.78 := by sorry

end total_value_of_coins_l3887_388794


namespace line_equation_l3887_388772

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line has equal intercepts on x and y axes -/
def Line.has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

theorem line_equation (l : Line) :
  l.has_equal_intercepts ∧ l.contains 1 2 →
  (l.a = 2 ∧ l.b = -1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -3) :=
sorry

end line_equation_l3887_388772


namespace multiple_of_1897_l3887_388753

theorem multiple_of_1897 (n : ℕ) : 1897 ∣ (2903^n - 803^n - 464^n + 261^n) := by
  sorry

end multiple_of_1897_l3887_388753


namespace hyperbola_asymptotes_l3887_388708

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    and eccentricity √3, its asymptotes have the equation y = ±√2x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt 3
  let c := e * a
  (c^2 = a^2 + b^2) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x) :=
by sorry

end hyperbola_asymptotes_l3887_388708


namespace rhombus_area_l3887_388780

/-- Represents a rhombus with diagonals 2a and 2b, and an acute angle θ. -/
structure Rhombus where
  a : ℕ+
  b : ℕ+
  θ : ℝ
  acute_angle : 0 < θ ∧ θ < π / 2

/-- The area of a rhombus is 2ab, where a and b are half the lengths of its diagonals. -/
theorem rhombus_area (r : Rhombus) : Real.sqrt ((2 * r.a) ^ 2 + (2 * r.b) ^ 2) / 2 * Real.sqrt ((2 * r.a) ^ 2 + (2 * r.b) ^ 2) * Real.sin r.θ / 2 = 2 * r.a * r.b := by
  sorry

end rhombus_area_l3887_388780


namespace sum_remainder_mod_seven_l3887_388742

theorem sum_remainder_mod_seven : (9^5 + 8^6 + 7^7) % 7 = 5 := by
  sorry

end sum_remainder_mod_seven_l3887_388742


namespace rectangle_configuration_l3887_388713

/-- The side length of square S2 in the given rectangle configuration. -/
def side_length_S2 : ℕ := 1300

/-- The side length of squares S1 and S3 in the given rectangle configuration. -/
def side_length_S1_S3 : ℕ := side_length_S2 + 50

/-- The width of the entire rectangle. -/
def total_width : ℕ := 4000

/-- The height of the entire rectangle. -/
def total_height : ℕ := 2500

/-- The theorem stating that the given configuration satisfies all conditions. -/
theorem rectangle_configuration :
  side_length_S1_S3 + side_length_S2 + side_length_S1_S3 = total_width ∧
  ∃ (r : ℕ), 2 * r + side_length_S2 = total_height :=
by sorry

end rectangle_configuration_l3887_388713


namespace beth_class_size_l3887_388774

/-- The number of students in Beth's class over three years -/
def final_class_size (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Theorem stating the final class size given the initial conditions -/
theorem beth_class_size :
  final_class_size 150 30 15 = 165 := by
  sorry

end beth_class_size_l3887_388774


namespace combined_monthly_profit_is_90_l3887_388734

/-- Represents a book with its purchase price, sale price, and months held before sale -/
structure Book where
  purchase_price : ℕ
  sale_price : ℕ
  months_held : ℕ

/-- Calculates the monthly profit for a single book -/
def monthly_profit (book : Book) : ℚ :=
  (book.sale_price - book.purchase_price : ℚ) / book.months_held

/-- Calculates the combined monthly rate of profit for a list of books -/
def combined_monthly_profit (books : List Book) : ℚ :=
  books.map monthly_profit |>.sum

theorem combined_monthly_profit_is_90 (books : List Book) : combined_monthly_profit books = 90 :=
  by
  have h1 : books = [
    { purchase_price := 50, sale_price := 90, months_held := 1 },
    { purchase_price := 120, sale_price := 150, months_held := 2 },
    { purchase_price := 75, sale_price := 110, months_held := 0 }
  ] := by sorry
  rw [h1]
  simp [combined_monthly_profit, monthly_profit]
  -- The proof goes here
  sorry

end combined_monthly_profit_is_90_l3887_388734


namespace floor_ceiling_sum_seven_l3887_388792

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℤ) + (⌈x⌉ : ℤ) = 7 ↔ 3 < x ∧ x < 4 := by
  sorry

end floor_ceiling_sum_seven_l3887_388792


namespace identity_function_only_solution_l3887_388725

theorem identity_function_only_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x + y) = x + f (f y)) → (∀ x : ℝ, f x = x) :=
by sorry

end identity_function_only_solution_l3887_388725


namespace sum_of_coefficients_l3887_388733

theorem sum_of_coefficients : ∃ (a b d c : ℕ+), 
  (∀ (a' b' d' c' : ℕ+), 
    (a' * Real.sqrt 3 + b' * Real.sqrt 11 + d' * Real.sqrt 2) / c' = 
    Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 + Real.sqrt 2 + 1 / Real.sqrt 2 →
    c ≤ c') ∧
  (a * Real.sqrt 3 + b * Real.sqrt 11 + d * Real.sqrt 2) / c = 
    Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 + Real.sqrt 2 + 1 / Real.sqrt 2 ∧
  a + b + d + c = 325 := by
sorry

end sum_of_coefficients_l3887_388733


namespace least_bench_sections_thirteen_is_least_l3887_388719

theorem least_bench_sections (M : ℕ) : M > 0 ∧ 5 * M = 13 * M → M ≥ 13 := by
  sorry

theorem thirteen_is_least : ∃ M : ℕ, M > 0 ∧ 5 * M = 13 * M ∧ M = 13 := by
  sorry

end least_bench_sections_thirteen_is_least_l3887_388719


namespace candy_distribution_l3887_388793

theorem candy_distribution (total_candy : ℕ) (pieces_per_student : ℕ) (num_students : ℕ) :
  total_candy = 344 →
  pieces_per_student = 8 →
  total_candy = num_students * pieces_per_student →
  num_students = 43 := by
sorry

end candy_distribution_l3887_388793


namespace dot_product_sum_l3887_388717

/-- Given vectors in ℝ², prove that the dot product of (a + b) and c equals 6 -/
theorem dot_product_sum (a b c : ℝ × ℝ) (ha : a = (1, -2)) (hb : b = (3, 4)) (hc : c = (2, -1)) :
  ((a.1 + b.1, a.2 + b.2) • c) = 6 := by
  sorry

end dot_product_sum_l3887_388717


namespace like_terms_power_l3887_388777

theorem like_terms_power (m n : ℕ) : 
  (∃ (x y : ℝ), 2 * x^(m-1) * y^2 = -2 * x^2 * y^n) → 
  (-m : ℤ)^n = 9 := by
sorry

end like_terms_power_l3887_388777


namespace tournament_teams_count_l3887_388722

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  n : ℕ  -- number of teams
  winner_points : ℕ
  last_place_points : ℕ
  winner_points_eq : winner_points = 26
  last_place_points_eq : last_place_points = 20

/-- Theorem stating that under the given conditions, the number of teams must be 12 -/
theorem tournament_teams_count (t : FootballTournament) : t.n = 12 := by
  sorry

end tournament_teams_count_l3887_388722


namespace r_value_when_n_is_3_l3887_388727

theorem r_value_when_n_is_3 : 
  ∀ (n s r : ℕ), 
    s = 2^n - 1 → 
    r = 3^s - s → 
    n = 3 → 
    r = 2180 := by
  sorry

end r_value_when_n_is_3_l3887_388727


namespace popsicle_problem_l3887_388782

/-- Popsicle Making Problem -/
theorem popsicle_problem (total_money : ℕ) (mold_cost : ℕ) (stick_pack_cost : ℕ) 
  (juice_cost : ℕ) (total_sticks : ℕ) (remaining_sticks : ℕ) :
  total_money = 10 →
  mold_cost = 3 →
  stick_pack_cost = 1 →
  juice_cost = 2 →
  total_sticks = 100 →
  remaining_sticks = 40 →
  (total_money - mold_cost - stick_pack_cost) / juice_cost * 
    ((total_sticks - remaining_sticks) / ((total_money - mold_cost - stick_pack_cost) / juice_cost)) = 20 :=
by
  sorry

end popsicle_problem_l3887_388782


namespace money_spent_on_baseball_gear_l3887_388703

def initial_amount : ℕ := 67
def amount_left : ℕ := 34

theorem money_spent_on_baseball_gear :
  initial_amount - amount_left = 33 := by sorry

end money_spent_on_baseball_gear_l3887_388703


namespace circle_contains_three_points_l3887_388746

/-- Represents a point in 2D space -/
structure Point where
  x : Real
  y : Real

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : Real

/-- Theorem: Given 51 points randomly placed on a unit square, 
    there always exists a circle of radius 1/7 that contains at least 3 of these points -/
theorem circle_contains_three_points 
  (points : Finset Point) 
  (h_count : points.card = 51) 
  (h_in_square : ∀ p ∈ points, 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1) :
  ∃ c : Circle, c.radius = 1/7 ∧ (points.filter (λ p => (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2)).card ≥ 3 :=
sorry

end circle_contains_three_points_l3887_388746


namespace line_symmetry_l3887_388791

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - y + 3 = 0
def line2 (x y : ℝ) : Prop := y = x + 2
def symmetric_line (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (f g h : ℝ → ℝ → Prop) : Prop :=
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    f x₁ y₁ → h x₂ y₂ → 
    ∃ (x_mid y_mid : ℝ), g x_mid y_mid ∧
    (x₂ - x_mid = x_mid - x₁) ∧ (y₂ - y_mid = y_mid - y₁)

-- Theorem statement
theorem line_symmetry : symmetric_wrt line1 line2 symmetric_line :=
sorry

end line_symmetry_l3887_388791


namespace fraction_simplification_and_result_l3887_388776

theorem fraction_simplification_and_result (a : ℤ) (h : a = 2018) : 
  (a + 1 : ℚ) / a - a / (a + 1) = (2 * a + 1 : ℚ) / (a * (a + 1)) ∧ 
  2 * a + 1 = 4037 := by
  sorry

end fraction_simplification_and_result_l3887_388776


namespace decagon_diagonals_l3887_388785

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l3887_388785


namespace coupon_probability_l3887_388710

theorem coupon_probability (n m k : ℕ) (hn : n = 17) (hm : m = 9) (hk : k = 6) :
  (Nat.choose k k * Nat.choose (n - k) (m - k)) / Nat.choose n m = 3 / 442 := by
  sorry

end coupon_probability_l3887_388710


namespace litter_collection_weight_l3887_388783

theorem litter_collection_weight 
  (gina_bags : ℕ) 
  (neighborhood_multiplier : ℕ) 
  (bag_weight : ℕ) 
  (h1 : gina_bags = 8)
  (h2 : neighborhood_multiplier = 120)
  (h3 : bag_weight = 6) : 
  (gina_bags + gina_bags * neighborhood_multiplier) * bag_weight = 5808 := by
  sorry

end litter_collection_weight_l3887_388783


namespace smallest_solution_abs_equation_l3887_388728

theorem smallest_solution_abs_equation :
  let f := fun x : ℝ => x * |x| - (3 * x - 2)
  ∃ x₀ : ℝ, f x₀ = 0 ∧ ∀ x : ℝ, f x = 0 → x₀ ≤ x ∧ x₀ = (-3 - Real.sqrt 17) / 2 :=
by sorry

end smallest_solution_abs_equation_l3887_388728


namespace a_range_theorem_l3887_388718

-- Define the sequence a_n
def a_n (a n : ℝ) : ℝ := a * n^2 + n + 5

-- State the theorem
theorem a_range_theorem (a : ℝ) :
  (∀ n : ℕ, a_n a n < a_n a (n + 1) ∧ n ≤ 3) ∧
  (∀ n : ℕ, a_n a n > a_n a (n + 1) ∧ n ≥ 8) →
  -1/7 < a ∧ a < -1/17 :=
sorry

end a_range_theorem_l3887_388718


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l3887_388755

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (X : ℕ → ℕ) k * (Y : ℕ → ℕ) (8 - k)) =
  56 * (X : ℕ → ℕ) 3 * (Y : ℕ → ℕ) 5 + (Finset.range 9).sum (fun k => 
    if k ≠ 3 
    then Nat.choose 8 k * (X : ℕ → ℕ) k * (Y : ℕ → ℕ) (8 - k)
    else 0) :=
by sorry

#check coefficient_x3y5_in_expansion_of_x_plus_y_8

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l3887_388755


namespace sum_smallest_largest_angle_l3887_388786

/-- A hexagon with angles in arithmetic progression -/
structure ArithmeticHexagon where
  /-- The smallest angle of the hexagon -/
  a : ℝ
  /-- The common difference between consecutive angles -/
  n : ℝ
  /-- The angles are non-negative -/
  a_nonneg : 0 ≤ a
  n_nonneg : 0 ≤ n
  /-- The sum of all angles in a hexagon is 720° -/
  sum_angles : a + (a + n) + (a + 2*n) + (a + 3*n) + (a + 4*n) + (a + 5*n) = 720

/-- The sum of the smallest and largest angles in an arithmetic hexagon is 240° -/
theorem sum_smallest_largest_angle (h : ArithmeticHexagon) : h.a + (h.a + 5*h.n) = 240 := by
  sorry


end sum_smallest_largest_angle_l3887_388786


namespace selling_price_equal_profit_loss_l3887_388700

/-- Proves that the selling price yielding the same profit as the loss is 54,
    given the cost price and a known selling price that results in a loss. -/
theorem selling_price_equal_profit_loss
  (cost_price : ℝ)
  (loss_price : ℝ)
  (h1 : cost_price = 47)
  (h2 : loss_price = 40)
  : ∃ (selling_price : ℝ),
    selling_price - cost_price = cost_price - loss_price ∧
    selling_price = 54 :=
by
  sorry

#check selling_price_equal_profit_loss

end selling_price_equal_profit_loss_l3887_388700


namespace integral_circle_area_l3887_388754

theorem integral_circle_area (f : ℝ → ℝ) (a b r : ℝ) (h : ∀ x ∈ Set.Icc a b, f x = Real.sqrt (r^2 - x^2)) :
  (∫ x in a..b, f x) = (π * r^2) / 2 :=
sorry

end integral_circle_area_l3887_388754


namespace technicians_in_exchange_group_and_expectation_l3887_388711

/-- Represents the distribution of job certificates --/
structure JobCertificates where
  junior : Nat
  intermediate : Nat
  senior : Nat
  technician : Nat
  seniorTechnician : Nat

/-- The total number of apprentices --/
def totalApprentices : Nat := 200

/-- The distribution of job certificates --/
def certificateDistribution : JobCertificates :=
  { junior := 20
  , intermediate := 60
  , senior := 60
  , technician := 40
  , seniorTechnician := 20 }

/-- The number of people selected for the exchange group --/
def exchangeGroupSize : Nat := 10

/-- The number of people chosen as representatives to speak --/
def speakersSize : Nat := 3

/-- Theorem stating the number of technicians in the exchange group and the expected number of technicians among speakers --/
theorem technicians_in_exchange_group_and_expectation :
  let totalTechnicians := certificateDistribution.technician + certificateDistribution.seniorTechnician
  let techniciansInExchangeGroup := (totalTechnicians * exchangeGroupSize) / totalApprentices
  let expectationOfTechnicians : Rat := 9 / 10
  techniciansInExchangeGroup = 3 ∧ 
  expectationOfTechnicians = (0 * (7 / 24 : Rat) + 1 * (21 / 40 : Rat) + 2 * (7 / 40 : Rat) + 3 * (1 / 120 : Rat)) := by
  sorry

end technicians_in_exchange_group_and_expectation_l3887_388711


namespace min_area_intersecting_hyperbolas_l3887_388702

/-- A set in ℝ² is convex if for any two points in the set, 
    the line segment connecting them is also in the set -/
def is_convex (S : Set (ℝ × ℝ)) : Prop :=
  ∀ (p q : ℝ × ℝ), p ∈ S → q ∈ S → 
    ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 → (1 - t) • p + t • q ∈ S

/-- A set intersects a hyperbola if there exists a point in the set 
    that satisfies the hyperbola equation -/
def intersects_hyperbola (S : Set (ℝ × ℝ)) (k : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) ∈ S ∧ x * y = k

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem: The minimum area of a convex set intersecting 
    both branches of xy = 1 and xy = -1 is 4 -/
theorem min_area_intersecting_hyperbolas :
  (∃ (S : Set (ℝ × ℝ)), 
    is_convex S ∧ 
    intersects_hyperbola S 1 ∧ 
    intersects_hyperbola S (-1)) →
  (∀ (S : Set (ℝ × ℝ)), 
    is_convex S → 
    intersects_hyperbola S 1 → 
    intersects_hyperbola S (-1) → 
    area S ≥ 4) ∧
  (∃ (S : Set (ℝ × ℝ)), 
    is_convex S ∧ 
    intersects_hyperbola S 1 ∧ 
    intersects_hyperbola S (-1) ∧ 
    area S = 4) :=
by sorry

end min_area_intersecting_hyperbolas_l3887_388702


namespace theater_ticket_profit_l3887_388743

/-- Calculates the total profit from ticket sales given the ticket prices and quantities sold. -/
theorem theater_ticket_profit
  (adult_price : ℕ)
  (kid_price : ℕ)
  (total_tickets : ℕ)
  (kid_tickets : ℕ)
  (h1 : adult_price = 6)
  (h2 : kid_price = 2)
  (h3 : total_tickets = 175)
  (h4 : kid_tickets = 75) :
  (total_tickets - kid_tickets) * adult_price + kid_tickets * kid_price = 750 :=
by sorry


end theater_ticket_profit_l3887_388743


namespace apples_left_over_l3887_388789

theorem apples_left_over (greg_sarah_apples susan_apples mark_apples : ℕ) : 
  greg_sarah_apples = 18 →
  susan_apples = 2 * (greg_sarah_apples / 2) →
  mark_apples = susan_apples - 5 →
  (greg_sarah_apples + susan_apples + mark_apples) - 40 = 9 :=
by sorry

end apples_left_over_l3887_388789


namespace reciprocal_sum_pairs_l3887_388723

theorem reciprocal_sum_pairs : 
  (Finset.filter 
    (fun p : ℕ × ℕ => 
      p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 6)
    (Finset.product (Finset.range 100) (Finset.range 100))).card = 9 :=
by sorry

end reciprocal_sum_pairs_l3887_388723


namespace equation_system_solutions_equation_system_unique_solutions_l3887_388795

/-- The system of equations has four solutions: (1, 1, 1) and three cyclic permutations of another triple -/
theorem equation_system_solutions :
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧
  (∀ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x^2 - y = (z - 1)^2 ∧
    y^2 - z = (x - 1)^2 ∧
    z^2 - x = (y - 1)^2 →
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨
     (x = a ∧ y = b ∧ z = c) ∨
     (x = b ∧ y = c ∧ z = a) ∨
     (x = c ∧ y = a ∧ z = b))) :=
by sorry

/-- The system of equations has exactly four solutions -/
theorem equation_system_unique_solutions :
  ∃! (s : Finset (ℝ × ℝ × ℝ)), s.card = 4 ∧
  (∀ (x y z : ℝ), (x, y, z) ∈ s ↔
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x^2 - y = (z - 1)^2 ∧
    y^2 - z = (x - 1)^2 ∧
    z^2 - x = (y - 1)^2) :=
by sorry

end equation_system_solutions_equation_system_unique_solutions_l3887_388795


namespace carpenter_woodblocks_needed_l3887_388769

/-- Calculates the total number of woodblocks needed by a carpenter to build a house. -/
theorem carpenter_woodblocks_needed 
  (initial_logs : ℕ) 
  (woodblocks_per_log : ℕ) 
  (additional_logs_needed : ℕ) : 
  (initial_logs + additional_logs_needed) * woodblocks_per_log = 80 :=
by
  sorry

#check carpenter_woodblocks_needed 8 5 8

end carpenter_woodblocks_needed_l3887_388769


namespace square_area_relationship_l3887_388749

/-- Given a square with side length a+b, prove that the relationship between 
    the areas of three squares formed within it can be expressed as a^2 + b^2 = c^2. -/
theorem square_area_relationship (a b c : ℝ) : 
  (∃ (total_area : ℝ), total_area = (a + b)^2) → 
  a^2 + b^2 = c^2 := by
  sorry

end square_area_relationship_l3887_388749


namespace quadratic_rewrite_l3887_388762

theorem quadratic_rewrite (d e f : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 40 * x - 72 = (d * x + e)^2 + f) →
  d * e = -20 := by
sorry

end quadratic_rewrite_l3887_388762
