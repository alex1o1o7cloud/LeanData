import Mathlib

namespace initial_water_calculation_l897_89776

/-- The amount of water initially poured into the pool -/
def initial_amount : ℝ := 1

/-- The amount of water added later -/
def added_amount : ℝ := 8.8

/-- The total amount of water in the pool -/
def total_amount : ℝ := 9.8

/-- Theorem stating that the initial amount plus the added amount equals the total amount -/
theorem initial_water_calculation :
  initial_amount + added_amount = total_amount := by
  sorry

end initial_water_calculation_l897_89776


namespace complementary_event_l897_89772

-- Define the sample space
def SampleSpace := List Bool

-- Define the event of missing both times
def MissBoth (outcome : SampleSpace) : Prop := outcome = [false, false]

-- Define the event of at least one hit
def AtLeastOneHit (outcome : SampleSpace) : Prop := outcome ≠ [false, false]

-- Theorem statement
theorem complementary_event : 
  ∀ (outcome : SampleSpace), MissBoth outcome ↔ ¬(AtLeastOneHit outcome) :=
sorry

end complementary_event_l897_89772


namespace josiah_cookie_spending_l897_89746

/-- The number of days in March -/
def days_in_march : ℕ := 31

/-- The number of cookies Josiah buys each day -/
def cookies_per_day : ℕ := 2

/-- The cost of each cookie in dollars -/
def cost_per_cookie : ℕ := 16

/-- Josiah's total spending on cookies in March -/
def total_spending : ℕ := days_in_march * cookies_per_day * cost_per_cookie

/-- Theorem stating that Josiah's total spending on cookies in March is 992 dollars -/
theorem josiah_cookie_spending : total_spending = 992 := by
  sorry

end josiah_cookie_spending_l897_89746


namespace thirty_divides_p_squared_minus_one_l897_89792

theorem thirty_divides_p_squared_minus_one (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  30 ∣ (p^2 - 1) ↔ p = 5 := by
  sorry

end thirty_divides_p_squared_minus_one_l897_89792


namespace tax_calculation_l897_89734

/-- Calculates the annual income before tax given tax rates and differential savings -/
def annual_income_before_tax (original_rate new_rate : ℚ) (differential_savings : ℚ) : ℚ :=
  differential_savings / (original_rate - new_rate)

/-- Theorem stating that given the specified tax rates and differential savings, 
    the annual income before tax is $34,500 -/
theorem tax_calculation (original_rate new_rate differential_savings : ℚ) 
  (h1 : original_rate = 42 / 100)
  (h2 : new_rate = 28 / 100)
  (h3 : differential_savings = 4830) :
  annual_income_before_tax original_rate new_rate differential_savings = 34500 := by
  sorry

#eval annual_income_before_tax (42/100) (28/100) 4830

end tax_calculation_l897_89734


namespace angle_difference_l897_89775

theorem angle_difference (α β : Real) 
  (h1 : 3 * Real.sin α - Real.cos α = 0)
  (h2 : 7 * Real.sin β + Real.cos β = 0)
  (h3 : 0 < α) (h4 : α < Real.pi / 2)
  (h5 : Real.pi / 2 < β) (h6 : β < Real.pi) :
  2 * α - β = -3 * Real.pi / 4 := by
  sorry

end angle_difference_l897_89775


namespace wage_increase_l897_89700

theorem wage_increase (original_wage new_wage : ℝ) (increase_percentage : ℝ) :
  new_wage = original_wage * (1 + increase_percentage / 100) →
  increase_percentage = 30 →
  new_wage = 78 →
  original_wage = 60 := by
sorry

end wage_increase_l897_89700


namespace ava_remaining_distance_l897_89780

/-- The remaining distance for Ava to finish the race -/
def remaining_distance (race_length : ℕ) (distance_covered : ℕ) : ℕ :=
  race_length - distance_covered

/-- Proof that Ava's remaining distance is 167 meters -/
theorem ava_remaining_distance :
  remaining_distance 1000 833 = 167 := by
  sorry

end ava_remaining_distance_l897_89780


namespace problem_statement_l897_89764

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
sorry

end problem_statement_l897_89764


namespace tom_initial_investment_l897_89703

/-- Represents the initial investment of Tom in rupees -/
def tom_investment : ℕ := 30000

/-- Represents Jose's investment in rupees -/
def jose_investment : ℕ := 45000

/-- Represents the total profit after one year in rupees -/
def total_profit : ℕ := 63000

/-- Represents Jose's share of the profit in rupees -/
def jose_profit : ℕ := 35000

/-- Represents the number of months Tom invested -/
def tom_months : ℕ := 12

/-- Represents the number of months Jose invested -/
def jose_months : ℕ := 10

theorem tom_initial_investment :
  tom_investment * tom_months * jose_profit = jose_investment * jose_months * (total_profit - jose_profit) :=
sorry

end tom_initial_investment_l897_89703


namespace gear_rotation_l897_89759

theorem gear_rotation (teeth_A teeth_B turns_A : ℕ) (h1 : teeth_A = 6) (h2 : teeth_B = 8) (h3 : turns_A = 12) :
  teeth_A * turns_A = teeth_B * (teeth_A * turns_A / teeth_B) :=
sorry

end gear_rotation_l897_89759


namespace circle_radius_is_five_l897_89729

/-- A rectangle with length 10 and width 6 -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (length_eq : length = 10)
  (width_eq : width = 6)

/-- A circle passing through two vertices of the rectangle and tangent to the opposite side -/
structure CircleTangentToRectangle (rect : Rectangle) :=
  (radius : ℝ)
  (passes_through_vertices : Bool)
  (tangent_to_opposite_side : Bool)

/-- The theorem stating that the radius of the circle is 5 -/
theorem circle_radius_is_five (rect : Rectangle) (circle : CircleTangentToRectangle rect) :
  circle.radius = 5 := by
  sorry

end circle_radius_is_five_l897_89729


namespace right_triangle_third_side_l897_89716

theorem right_triangle_third_side 
  (a b : ℝ) 
  (h1 : Real.sqrt (a - 3) + |b - 4| = 0) : 
  ∃ c : ℝ, (c = 5 ∨ c = Real.sqrt 7) ∧ 
    ((a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)) := by
  sorry

end right_triangle_third_side_l897_89716


namespace circumradius_of_specific_trapezoid_l897_89798

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  longBase : ℝ
  shortBase : ℝ
  lateralSide : ℝ

/-- The radius of the circumscribed circle of an isosceles trapezoid -/
def circumradius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The radius of the circumscribed circle of the given isosceles trapezoid is 5√2 -/
theorem circumradius_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := ⟨14, 2, 10⟩
  circumradius t = 5 * Real.sqrt 2 := by
  sorry

end circumradius_of_specific_trapezoid_l897_89798


namespace range_of_m_l897_89763

-- Define the conditions
def p (x : ℝ) : Prop := (x + 2) * (x - 10) ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → p x) →
  (∃ x, p x ∧ ¬q x m) →
  (0 < m ∧ m ≤ 3) :=
sorry

end range_of_m_l897_89763


namespace resultant_calculation_l897_89752

theorem resultant_calculation : 
  let original : ℕ := 13
  let doubled := 2 * original
  let added_seven := doubled + 7
  let trebled := 3 * added_seven
  trebled = 99 := by sorry

end resultant_calculation_l897_89752


namespace find_special_number_l897_89745

/-- A number is a perfect square if it's equal to some integer squared. -/
def is_perfect_square (n : ℤ) : Prop := ∃ k : ℤ, n = k^2

/-- The problem statement -/
theorem find_special_number : 
  ∃ m : ℕ+, 
    is_perfect_square (m.val + 100) ∧ 
    is_perfect_square (m.val + 168) ∧ 
    m.val = 156 := by
  sorry

end find_special_number_l897_89745


namespace line_invariant_under_transformation_l897_89777

def transformation (a b : ℝ) (x y : ℝ) : ℝ × ℝ :=
  (-x + a*y, b*x + 3*y)

theorem line_invariant_under_transformation (a b : ℝ) :
  (∀ x y : ℝ, 2*x - y = 3 → 
    let (x', y') := transformation a b x y
    2*x' - y' = 3) →
  a = 1 ∧ b = -4 := by
sorry

end line_invariant_under_transformation_l897_89777


namespace inequality_solution_set_l897_89742

theorem inequality_solution_set (t : ℝ) (a : ℝ) : 
  (∀ x : ℝ, (tx^2 - 6*x + t^2 < 0) ↔ (x < a ∨ x > 1)) → a = -3 :=
by sorry

end inequality_solution_set_l897_89742


namespace prob_b_is_three_fourths_l897_89768

/-- The probability that either A or B solves a problem, given their individual probabilities -/
def prob_either_solves (prob_a prob_b : ℝ) : ℝ :=
  prob_a + prob_b - prob_a * prob_b

/-- Theorem stating that if A's probability is 2/3 and the probability of either A or B solving
    is 0.9166666666666666, then B's probability is 3/4 -/
theorem prob_b_is_three_fourths (prob_a prob_b : ℝ) 
    (h1 : prob_a = 2/3)
    (h2 : prob_either_solves prob_a prob_b = 0.9166666666666666) :
    prob_b = 3/4 := by
  sorry


end prob_b_is_three_fourths_l897_89768


namespace no_solution_l897_89744

/-- Product of digits of a natural number in base ten -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- The main theorem: no natural number satisfies the given equation -/
theorem no_solution :
  ∀ x : ℕ, productOfDigits x ≠ x^2 - 10*x - 22 := by
  sorry

end no_solution_l897_89744


namespace sum_reciprocal_inequality_l897_89712

theorem sum_reciprocal_inequality (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) 
  (h_sum_squares : a^2 + b^2 + c^2 = 12) : 
  1/(a-1) + 1/(b-1) + 1/(c-1) ≥ 3 := by
  sorry

end sum_reciprocal_inequality_l897_89712


namespace annes_cleaning_time_l897_89783

/-- Represents the time it takes Anne to clean the house individually -/
def annes_individual_time (bruce_rate anne_rate : ℚ) : ℚ :=
  1 / anne_rate

/-- The condition that Bruce and Anne can clean the house in 4 hours together -/
def condition1 (bruce_rate anne_rate : ℚ) : Prop :=
  bruce_rate + anne_rate = 1 / 4

/-- The condition that Bruce and Anne with Anne's doubled speed can clean the house in 3 hours -/
def condition2 (bruce_rate anne_rate : ℚ) : Prop :=
  bruce_rate + 2 * anne_rate = 1 / 3

theorem annes_cleaning_time 
  (bruce_rate anne_rate : ℚ) 
  (h1 : condition1 bruce_rate anne_rate) 
  (h2 : condition2 bruce_rate anne_rate) :
  annes_individual_time bruce_rate anne_rate = 12 := by
sorry

end annes_cleaning_time_l897_89783


namespace tan_alpha_3_expression_equals_2_l897_89778

theorem tan_alpha_3_expression_equals_2 (α : Real) (h : Real.tan α = 3) :
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 2 := by
  sorry

end tan_alpha_3_expression_equals_2_l897_89778


namespace max_value_location_l897_89707

theorem max_value_location (f : ℝ → ℝ) (a b : ℝ) (h : a < b) :
  Differentiable ℝ f → ∃ x ∈ Set.Icc a b,
    (∀ y ∈ Set.Icc a b, f y ≤ f x) ∧
    (x = a ∨ x = b ∨ deriv f x = 0) :=
sorry

end max_value_location_l897_89707


namespace arithmetic_sequence_common_difference_l897_89726

/-- An arithmetic sequence with first term 2 and the sum of the second and fourth terms equal to the sixth term has a common difference of 2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 1 = 2)  -- First term is 2
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)  -- Definition of arithmetic sequence
  (h3 : a 2 + a 4 = a 6)  -- Sum of second and fourth terms equals sixth term
  : d = 2 := by
  sorry

end arithmetic_sequence_common_difference_l897_89726


namespace books_read_during_travel_l897_89722

theorem books_read_during_travel (total_distance : ℕ) (reading_rate : ℕ) (books_finished : ℕ) : 
  total_distance = 6760 → reading_rate = 450 → books_finished = total_distance / reading_rate → books_finished = 15 := by
  sorry

end books_read_during_travel_l897_89722


namespace exam_question_count_l897_89711

/-- Represents the scoring system and results of an examination. -/
structure ExamResult where
  correct_score : ℕ  -- Score for each correct answer
  wrong_penalty : ℕ  -- Penalty for each wrong answer
  total_score : ℤ    -- Total score achieved
  correct_count : ℕ  -- Number of correctly answered questions
  total_count : ℕ    -- Total number of questions attempted

/-- Theorem stating the relationship between exam parameters and the total number of questions attempted. -/
theorem exam_question_count 
  (exam : ExamResult) 
  (h1 : exam.correct_score = 4)
  (h2 : exam.wrong_penalty = 1)
  (h3 : exam.total_score = 130)
  (h4 : exam.correct_count = 42) :
  exam.total_count = 80 := by
  sorry

#check exam_question_count

end exam_question_count_l897_89711


namespace exhibition_planes_l897_89767

/-- The number of wings on a commercial plane -/
def wings_per_plane : ℕ := 2

/-- The total number of wings counted -/
def total_wings : ℕ := 50

/-- The number of planes in the exhibition -/
def num_planes : ℕ := total_wings / wings_per_plane

theorem exhibition_planes : num_planes = 25 := by
  sorry

end exhibition_planes_l897_89767


namespace cube_sum_magnitude_l897_89733

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 20) :
  Complex.abs (w^3 + z^3) = 56 := by
  sorry

end cube_sum_magnitude_l897_89733


namespace shaded_area_in_squares_l897_89732

/-- The area of the shaded region in a specific geometric configuration -/
theorem shaded_area_in_squares : 
  let small_square_side : ℝ := 4
  let large_square_side : ℝ := 12
  let rectangle_width : ℝ := 2
  let rectangle_height : ℝ := 4
  let total_width : ℝ := small_square_side + large_square_side
  let triangle_height : ℝ := (small_square_side * small_square_side) / total_width
  let triangle_area : ℝ := (1 / 2) * triangle_height * small_square_side
  let small_square_area : ℝ := small_square_side * small_square_side
  let shaded_area : ℝ := small_square_area - triangle_area
  shaded_area = 14 := by
    sorry

end shaded_area_in_squares_l897_89732


namespace integral_x_plus_cos_2x_over_symmetric_interval_l897_89740

theorem integral_x_plus_cos_2x_over_symmetric_interval : 
  ∫ x in (-π/2)..(π/2), (x + Real.cos (2*x)) = 0 := by
  sorry

end integral_x_plus_cos_2x_over_symmetric_interval_l897_89740


namespace sum_of_diagonals_is_190_l897_89754

/-- A hexagon inscribed in a circle -/
structure InscribedHexagon where
  -- Sides of the hexagon
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  -- Conditions on the sides
  h1 : side1 = 20
  h2 : side3 = 30
  h3 : side2 = 50
  h4 : side4 = 50
  h5 : side5 = 50
  h6 : side6 = 50

/-- The sum of diagonals from one vertex in the inscribed hexagon -/
def sumOfDiagonals (h : InscribedHexagon) : ℝ := sorry

/-- Theorem: The sum of diagonals from one vertex in the specified hexagon is 190 -/
theorem sum_of_diagonals_is_190 (h : InscribedHexagon) : sumOfDiagonals h = 190 := by
  sorry

end sum_of_diagonals_is_190_l897_89754


namespace logarithm_and_exponential_equalities_l897_89758

theorem logarithm_and_exponential_equalities :
  (Real.log 9 / Real.log 6 + 2 * Real.log 2 / Real.log 6 = 2) ∧
  (Real.exp 0 + Real.sqrt ((1 - Real.sqrt 2)^2) - 8^(1/6) = 1 + Real.sqrt 5 - Real.sqrt 2 - 2^(1/3)) := by
  sorry

end logarithm_and_exponential_equalities_l897_89758


namespace arithmetic_equality_l897_89799

theorem arithmetic_equality : 3 * 9 + 4 * 10 + 11 * 3 + 3 * 8 = 124 := by
  sorry

end arithmetic_equality_l897_89799


namespace circle_symmetry_line_coefficient_product_l897_89750

/-- Given a circle and a line, prove that the product of the line's coefficients is non-positive -/
theorem circle_symmetry_line_coefficient_product (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0 →
    ∃ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' + 1 = 0 ∧
      2*a*x - b*y + 2 = 2*a*x' - b*y' + 2) →
  a * b ≤ 0 :=
sorry

end circle_symmetry_line_coefficient_product_l897_89750


namespace emily_sixth_score_l897_89728

def emily_scores : List ℕ := [94, 97, 88, 90, 102]
def target_mean : ℚ := 95
def num_quizzes : ℕ := 6

theorem emily_sixth_score (sixth_score : ℕ) : 
  sixth_score = 99 →
  (emily_scores.sum + sixth_score) / num_quizzes = target_mean := by
sorry

end emily_sixth_score_l897_89728


namespace power_of_power_three_l897_89748

theorem power_of_power_three : (3^3)^2 = 729 := by
  sorry

end power_of_power_three_l897_89748


namespace perpendicular_bisector_c_value_l897_89779

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The perpendicular bisector of a line segment -/
def isPerpBisector (c : ℝ) (p1 p2 : Point) : Prop :=
  let midpoint : Point := ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩
  (midpoint.x + midpoint.y = c) ∧ 
  (c - p1.x - p1.y = p2.x + p2.y - c)

/-- The theorem statement -/
theorem perpendicular_bisector_c_value :
  ∀ c : ℝ, isPerpBisector c ⟨2, 5⟩ ⟨8, 11⟩ → c = 13 := by
  sorry


end perpendicular_bisector_c_value_l897_89779


namespace walk_distance_l897_89782

theorem walk_distance (x y : ℝ) : 
  x > 0 → y > 0 → 
  (x^2 + y^2 - x*y = 9) → 
  x = 3 :=
by sorry

end walk_distance_l897_89782


namespace sum_of_multiples_is_odd_l897_89731

theorem sum_of_multiples_is_odd (c d : ℤ) 
  (hc : ∃ m : ℤ, c = 6 * m) 
  (hd : ∃ n : ℤ, d = 9 * n) : 
  Odd (c + d) := by
  sorry

end sum_of_multiples_is_odd_l897_89731


namespace ellipse_theorem_l897_89724

-- Define the ellipse
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition that a > b > 0
def size_condition (a b : ℝ) : Prop := a > b ∧ b > 0

-- Define the angle condition
def angle_condition (PF1F2_angle : ℝ) : Prop := Real.sin PF1F2_angle = 1/3

-- Main theorem
theorem ellipse_theorem (a b : ℝ) (h1 : size_condition a b) 
  (h2 : ∃ P F1 F2 : ℝ × ℝ, 
    ellipse a b (F2.1) (P.2) ∧ 
    angle_condition (Real.arcsin (1/3))) : 
  a = Real.sqrt 2 * b := by
  sorry

end ellipse_theorem_l897_89724


namespace bicycle_problem_solution_l897_89709

/-- Represents the bicycle sales and inventory problem over three days -/
def bicycle_problem (S1 S2 S3 B1 B2 B3 P1 P2 P3 C1 C2 C3 : ℕ) : Prop :=
  let sale_profit1 := S1 * P1
  let sale_profit2 := S2 * P2
  let sale_profit3 := S3 * P3
  let repair_cost1 := B1 * C1
  let repair_cost2 := B2 * C2
  let repair_cost3 := B3 * C3
  let net_profit1 := sale_profit1 - repair_cost1
  let net_profit2 := sale_profit2 - repair_cost2
  let net_profit3 := sale_profit3 - repair_cost3
  let total_net_profit := net_profit1 + net_profit2 + net_profit3
  let net_increase := (B1 - S1) + (B2 - S2) + (B3 - S3)
  (S1 = 10 ∧ S2 = 12 ∧ S3 = 9 ∧
   B1 = 15 ∧ B2 = 8 ∧ B3 = 11 ∧
   P1 = 250 ∧ P2 = 275 ∧ P3 = 260 ∧
   C1 = 100 ∧ C2 = 110 ∧ C3 = 120) →
  (total_net_profit = 4440 ∧ net_increase = 3)

theorem bicycle_problem_solution :
  ∀ S1 S2 S3 B1 B2 B3 P1 P2 P3 C1 C2 C3 : ℕ,
  bicycle_problem S1 S2 S3 B1 B2 B3 P1 P2 P3 C1 C2 C3 :=
sorry

end bicycle_problem_solution_l897_89709


namespace no_valid_day_for_statements_l897_89773

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents whether a statement is true or false on a given day -/
def Statement := Day → Prop

/-- The statement "I lied yesterday" -/
def LiedYesterday : Statement := fun d => 
  match d with
  | Day.Monday => false     -- Sunday's statement
  | Day.Tuesday => false    -- Monday's statement
  | Day.Wednesday => false  -- Tuesday's statement
  | Day.Thursday => false   -- Wednesday's statement
  | Day.Friday => false     -- Thursday's statement
  | Day.Saturday => false   -- Friday's statement
  | Day.Sunday => false     -- Saturday's statement

/-- The statement "I will lie tomorrow" -/
def WillLieTomorrow : Statement := fun d =>
  match d with
  | Day.Monday => false     -- Tuesday's statement
  | Day.Tuesday => false    -- Wednesday's statement
  | Day.Wednesday => false  -- Thursday's statement
  | Day.Thursday => false   -- Friday's statement
  | Day.Friday => false     -- Saturday's statement
  | Day.Saturday => false   -- Sunday's statement
  | Day.Sunday => false     -- Monday's statement

/-- Theorem stating that there is no day where both statements can be made without contradiction -/
theorem no_valid_day_for_statements : ¬∃ (d : Day), LiedYesterday d ∧ WillLieTomorrow d := by
  sorry


end no_valid_day_for_statements_l897_89773


namespace greatest_integer_quadratic_inequality_l897_89757

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 13*n + 40 ≤ 0 ∧ n = 8 ∧ ∀ (m : ℤ), m^2 - 13*m + 40 ≤ 0 → m ≤ 8 :=
by sorry

end greatest_integer_quadratic_inequality_l897_89757


namespace unique_m_value_l897_89787

/-- Given function f(x) = |x-a| + m|x+a| -/
def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

theorem unique_m_value (m a : ℝ) 
  (h1 : 0 < m) (h2 : m < 1)
  (h3 : ∀ x : ℝ, f x a m ≥ 2)
  (h4 : a ≤ -5 ∨ a ≥ 5) :
  m = 1/5 := by sorry

end unique_m_value_l897_89787


namespace simplify_expression_l897_89727

theorem simplify_expression (x y : ℝ) : (5 - 2*x) - (8 - 6*x + 3*y) = -3 + 4*x - 3*y := by
  sorry

end simplify_expression_l897_89727


namespace cosine_sine_equality_l897_89765

theorem cosine_sine_equality (α : ℝ) : 
  3.3998 * (Real.cos α)^4 - 4 * (Real.cos α)^3 - 8 * (Real.cos α)^2 + 3 * Real.cos α + 1 = 
  -2 * Real.sin (7 * α / 2) * Real.sin (α / 2) := by
  sorry

end cosine_sine_equality_l897_89765


namespace least_number_with_remainder_five_forty_five_satisfies_least_number_is_545_l897_89715

theorem least_number_with_remainder (n : ℕ) : 
  (n % 12 = 5 ∧ n % 15 = 5 ∧ n % 20 = 5 ∧ n % 54 = 5) → n ≥ 545 := by
  sorry

theorem five_forty_five_satisfies :
  545 % 12 = 5 ∧ 545 % 15 = 5 ∧ 545 % 20 = 5 ∧ 545 % 54 = 5 := by
  sorry

theorem least_number_is_545 : 
  ∃! n : ℕ, (n % 12 = 5 ∧ n % 15 = 5 ∧ n % 20 = 5 ∧ n % 54 = 5) ∧
  ∀ m : ℕ, (m % 12 = 5 ∧ m % 15 = 5 ∧ m % 20 = 5 ∧ m % 54 = 5) → m ≥ n := by
  sorry

end least_number_with_remainder_five_forty_five_satisfies_least_number_is_545_l897_89715


namespace min_value_sum_of_powers_l897_89755

theorem min_value_sum_of_powers (a b x y : ℝ) (n : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) (hxy : x + y = 1) :
  a / x^n + b / y^n ≥ (a^(1/(n+1:ℝ)) + b^(1/(n+1:ℝ)))^(n+1) :=
sorry

end min_value_sum_of_powers_l897_89755


namespace savings_duration_l897_89713

/-- Proves that saving $34 daily for a total of $12,410 results in 365 days of savings -/
theorem savings_duration (daily_savings : ℕ) (total_savings : ℕ) (days : ℕ) :
  daily_savings = 34 →
  total_savings = 12410 →
  total_savings = daily_savings * days →
  days = 365 := by
sorry

end savings_duration_l897_89713


namespace contractor_labor_problem_l897_89706

theorem contractor_labor_problem (planned_days : ℕ) (absent_workers : ℕ) (actual_days : ℕ) 
  (h1 : planned_days = 9)
  (h2 : absent_workers = 6)
  (h3 : actual_days = 15) :
  ∃ (original_workers : ℕ), 
    original_workers * planned_days = (original_workers - absent_workers) * actual_days ∧ 
    original_workers = 15 := by
  sorry


end contractor_labor_problem_l897_89706


namespace seven_balls_four_boxes_l897_89702

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) (min_per_box : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 20 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes
    with at least one ball in each box -/
theorem seven_balls_four_boxes : distribute_balls 7 4 1 = 20 := by
  sorry

end seven_balls_four_boxes_l897_89702


namespace inscribed_sphere_volume_l897_89774

/-- The volume of a sphere inscribed in a cube with edge length 8 feet -/
theorem inscribed_sphere_volume :
  let cube_edge : ℝ := 8
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = (256 / 3) * Real.pi := by sorry

end inscribed_sphere_volume_l897_89774


namespace trajectory_and_intersection_l897_89781

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  2 * Real.sqrt ((x - 1)^2 + y^2) = 2 * (x + 1)

-- Define the trajectory equation
def trajectory_equation (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  y^2 = 4 * x

-- Define the intersection points M and N
def intersection_points (m : ℝ) (M N : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  y₁ = x₁ + m ∧ y₂ = x₂ + m ∧
  trajectory_equation M ∧ trajectory_equation N ∧
  m ≠ 0

-- Define the perpendicularity condition
def perpendicular (O M N : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem trajectory_and_intersection :
  (∀ P, P_condition P → trajectory_equation P) ∧
  (∀ m M N, intersection_points m M N → perpendicular (0, 0) M N → m = -4) := by
  sorry

end trajectory_and_intersection_l897_89781


namespace second_day_speed_l897_89736

/-- Given a journey with specific conditions, prove the speed on the second day -/
theorem second_day_speed 
  (distance : ℝ) 
  (first_day_speed : ℝ) 
  (normal_time : ℝ) 
  (first_day_delay : ℝ) 
  (second_day_early : ℝ) 
  (h1 : distance = 60) 
  (h2 : first_day_speed = 10) 
  (h3 : normal_time = distance / first_day_speed) 
  (h4 : first_day_delay = 2) 
  (h5 : second_day_early = 1) : 
  distance / (normal_time - second_day_early) = 12 := by
sorry

end second_day_speed_l897_89736


namespace hasan_plates_removal_l897_89704

/-- The weight of each plate in ounces -/
def plate_weight : ℕ := 10

/-- The weight limit for each box in pounds -/
def box_weight_limit : ℕ := 20

/-- The number of plates initially packed in the box -/
def initial_plates : ℕ := 38

/-- The number of ounces in a pound -/
def ounces_per_pound : ℕ := 16

/-- The number of plates Hasan needs to remove from the box -/
def plates_to_remove : ℕ := 6

theorem hasan_plates_removal :
  plates_to_remove = 
    (initial_plates * plate_weight - box_weight_limit * ounces_per_pound) / plate_weight :=
by sorry

end hasan_plates_removal_l897_89704


namespace zack_classroom_count_l897_89788

/-- The number of students in each classroom -/
structure ClassroomCounts where
  tina : ℕ
  maura : ℕ
  zack : ℕ

/-- The conditions of the problem -/
def classroom_problem (c : ClassroomCounts) : Prop :=
  c.tina = c.maura ∧
  c.zack = (c.tina + c.maura) / 2 ∧
  c.tina + c.maura + c.zack = 69

/-- The theorem to prove -/
theorem zack_classroom_count (c : ClassroomCounts) : 
  classroom_problem c → c.zack = 23 := by
  sorry

end zack_classroom_count_l897_89788


namespace smallest_prime_factor_of_7047_l897_89730

theorem smallest_prime_factor_of_7047 : Nat.minFac 7047 = 3 := by
  sorry

end smallest_prime_factor_of_7047_l897_89730


namespace unique_solution_quadratic_inequality_l897_89738

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + 2*a*x + 4*a| ≤ 4) ↔ a = 2 := by sorry

end unique_solution_quadratic_inequality_l897_89738


namespace probability_of_zero_in_one_over_99999_l897_89721

def decimal_expansion (n : ℕ) : List ℕ := 
  if n = 99999 then [0, 0, 0, 0, 1] else sorry

theorem probability_of_zero_in_one_over_99999 : 
  let expansion := decimal_expansion 99999
  let total_digits := expansion.length
  let zero_count := (expansion.filter (· = 0)).length
  (zero_count : ℚ) / total_digits = 4 / 5 := by sorry

end probability_of_zero_in_one_over_99999_l897_89721


namespace percentage_of_c_grades_l897_89717

def grading_scale : List (String × (Int × Int)) :=
  [("A", (95, 100)), ("B", (88, 94)), ("C", (78, 87)), ("D", (65, 77)), ("F", (0, 64))]

def scores : List Int :=
  [94, 65, 59, 99, 82, 89, 90, 68, 79, 62, 85, 81, 64, 83, 91]

def is_grade_c (score : Int) : Bool :=
  78 ≤ score ∧ score ≤ 87

def count_grade_c (scores : List Int) : Nat :=
  (scores.filter is_grade_c).length

theorem percentage_of_c_grades :
  (count_grade_c scores : Rat) / (scores.length : Rat) * 100 = 100/3 := by
  sorry

end percentage_of_c_grades_l897_89717


namespace tangent_line_inclination_l897_89786

/-- The angle of inclination of the tangent line to y = x^3 - 2x + 4 at (1, 3) is 45°. -/
theorem tangent_line_inclination (f : ℝ → ℝ) (x₀ y₀ : ℝ) :
  f x = x^3 - 2*x + 4 →
  x₀ = 1 →
  y₀ = 3 →
  f x₀ = y₀ →
  HasDerivAt f (3*x₀^2 - 2) x₀ →
  (Real.arctan (3*x₀^2 - 2)) * (180 / Real.pi) = 45 := by
  sorry

end tangent_line_inclination_l897_89786


namespace impossible_to_gather_all_stones_l897_89739

/-- Represents the number of stones in each pile -/
structure PileState :=
  (pile1 : Nat) (pile2 : Nat) (pile3 : Nat)

/-- Represents a valid move -/
inductive Move
  | move12 : Move  -- Move from pile 1 and 2 to 3
  | move13 : Move  -- Move from pile 1 and 3 to 2
  | move23 : Move  -- Move from pile 2 and 3 to 1

/-- Apply a move to a PileState -/
def applyMove (state : PileState) (move : Move) : PileState :=
  match move with
  | Move.move12 => PileState.mk (state.pile1 - 1) (state.pile2 - 1) (state.pile3 + 2)
  | Move.move13 => PileState.mk (state.pile1 - 1) (state.pile2 + 2) (state.pile3 - 1)
  | Move.move23 => PileState.mk (state.pile1 + 2) (state.pile2 - 1) (state.pile3 - 1)

/-- Check if all stones are in one pile -/
def isAllInOnePile (state : PileState) : Prop :=
  (state.pile1 = 0 ∧ state.pile2 = 0) ∨
  (state.pile1 = 0 ∧ state.pile3 = 0) ∨
  (state.pile2 = 0 ∧ state.pile3 = 0)

/-- Initial state of the piles -/
def initialState : PileState := PileState.mk 20 1 9

/-- Theorem stating it's impossible to gather all stones in one pile -/
theorem impossible_to_gather_all_stones :
  ¬∃ (moves : List Move), isAllInOnePile (moves.foldl applyMove initialState) :=
sorry

end impossible_to_gather_all_stones_l897_89739


namespace distance_table_1_to_3_l897_89718

/-- Calculates the distance between the first and third table in a relay race. -/
def distance_between_tables_1_and_3 (race_length : ℕ) (num_tables : ℕ) : ℕ :=
  2 * (race_length / num_tables)

/-- Proves that in a 1200-meter race with 6 equally spaced tables, 
    the distance between the first and third table is 400 meters. -/
theorem distance_table_1_to_3 : 
  distance_between_tables_1_and_3 1200 6 = 400 := by
  sorry

end distance_table_1_to_3_l897_89718


namespace fish_weight_sum_l897_89789

/-- The total weight of fish caught by Ali, Peter, and Joey -/
def total_fish_weight (ali_weight peter_weight joey_weight : ℝ) : ℝ :=
  ali_weight + peter_weight + joey_weight

/-- Theorem: The total weight of fish caught by Ali, Peter, and Joey is 25 kg -/
theorem fish_weight_sum :
  ∀ (peter_weight : ℝ),
  peter_weight > 0 →
  let ali_weight := 2 * peter_weight
  let joey_weight := peter_weight + 1
  ali_weight = 12 →
  total_fish_weight ali_weight peter_weight joey_weight = 25 := by
sorry

end fish_weight_sum_l897_89789


namespace vector_not_parallel_implies_x_value_l897_89701

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (4, x)

-- Define the condition that vectors are not parallel
def not_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 ≠ v.2 * w.1

-- Theorem statement
theorem vector_not_parallel_implies_x_value :
  ∃ x : ℝ, not_parallel a (b x) → x = 2 := by
  sorry

end vector_not_parallel_implies_x_value_l897_89701


namespace ring_ratio_l897_89741

def ring_problem (first_ring_cost second_ring_cost selling_price out_of_pocket : ℚ) : Prop :=
  first_ring_cost = 10000 ∧
  second_ring_cost = 2 * first_ring_cost ∧
  first_ring_cost + second_ring_cost - selling_price = out_of_pocket ∧
  out_of_pocket = 25000

theorem ring_ratio (first_ring_cost second_ring_cost selling_price out_of_pocket : ℚ) 
  (h : ring_problem first_ring_cost second_ring_cost selling_price out_of_pocket) :
  selling_price / first_ring_cost = 1 / 2 := by
  sorry

end ring_ratio_l897_89741


namespace b_25_mod_55_l897_89743

/-- Definition of b_n as a function that concatenates integers from 5 to n+4 -/
def b (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that b_25 mod 55 = 39 -/
theorem b_25_mod_55 : b 25 % 55 = 39 := by
  sorry

end b_25_mod_55_l897_89743


namespace amanda_pay_l897_89791

def hourly_rate : ℝ := 50
def hours_worked : ℝ := 10
def withholding_percentage : ℝ := 0.20

def daily_pay : ℝ := hourly_rate * hours_worked
def withheld_amount : ℝ := daily_pay * withholding_percentage
def final_pay : ℝ := daily_pay - withheld_amount

theorem amanda_pay : final_pay = 400 := by
  sorry

end amanda_pay_l897_89791


namespace geometric_sequence_properties_l897_89737

-- Define the geometric sequence
def geometric_sequence (n : ℕ) : ℚ :=
  (-1/2) ^ (n - 1)

-- Define the sum of the first n terms
def geometric_sum (n : ℕ) : ℚ :=
  (2/3) * (1 - (-1/2)^n)

-- Theorem statement
theorem geometric_sequence_properties :
  (geometric_sequence 3 = 1/4) ∧
  (∀ n : ℕ, geometric_sequence n = (-1/2)^(n-1)) ∧
  (∀ n : ℕ, geometric_sum n = (2/3) * (1 - (-1/2)^n)) :=
by sorry

end geometric_sequence_properties_l897_89737


namespace joan_lost_balloons_l897_89785

/-- Given that Joan initially had 8 orange balloons and now has 6,
    prove that she lost 2 balloons. -/
theorem joan_lost_balloons (initial : ℕ) (current : ℕ) (h1 : initial = 8) (h2 : current = 6) :
  initial - current = 2 := by
  sorry

end joan_lost_balloons_l897_89785


namespace prove_birds_and_storks_l897_89761

def birds_and_storks_problem : Prop :=
  let initial_birds : ℕ := 3
  let initial_storks : ℕ := 4
  let birds_arrived : ℕ := 2
  let birds_left : ℕ := 1
  let storks_arrived : ℕ := 3
  let final_birds : ℕ := initial_birds + birds_arrived - birds_left
  let final_storks : ℕ := initial_storks + storks_arrived
  (final_birds : Int) - (final_storks : Int) = -3

theorem prove_birds_and_storks : birds_and_storks_problem := by
  sorry

end prove_birds_and_storks_l897_89761


namespace scientific_notation_14800_l897_89749

theorem scientific_notation_14800 :
  14800 = 1.48 * (10 : ℝ)^4 := by sorry

end scientific_notation_14800_l897_89749


namespace smallest_lpm_l897_89760

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number with repeating digits -/
def TwoDigitRepeating (d : Digit) := 10 * d.val + d.val

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Digit
  tens : Digit
  ones : Digit

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds.val + 10 * n.tens.val + n.ones.val

/-- Checks if a natural number is a valid result of the multiplication -/
def isValidResult (l : Digit) (result : ThreeDigitNumber) : Prop :=
  (TwoDigitRepeating l) * l.val = result.toNat ∧
  result.hundreds = l

theorem smallest_lpm :
  ∃ (result : ThreeDigitNumber),
    (∃ (l : Digit), isValidResult l result) ∧
    (∀ (other : ThreeDigitNumber),
      (∃ (l : Digit), isValidResult l other) →
      result.toNat ≤ other.toNat) ∧
    result.toNat = 275 := by
  sorry

end smallest_lpm_l897_89760


namespace convex_ngon_diagonal_intersections_l897_89795

/-- The number of intersection points of diagonals in a convex n-gon -/
def diagonalIntersections (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) / 24

/-- Theorem: The number of intersection points for diagonals of a convex n-gon,
    where no three diagonals intersect at a single point, is equal to n(n-1)(n-2)(n-3)/24 -/
theorem convex_ngon_diagonal_intersections (n : ℕ) (h1 : n ≥ 4) :
  diagonalIntersections n = (n.choose 4) := by
  sorry

end convex_ngon_diagonal_intersections_l897_89795


namespace fourteenSidedPolygonArea_l897_89762

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a polygon defined by a list of vertices -/
structure Polygon where
  vertices : List Point

/-- Calculates the area of a polygon given its vertices -/
def calculatePolygonArea (p : Polygon) : ℝ := sorry

/-- The fourteen-sided polygon from the problem -/
def fourteenSidedPolygon : Polygon :=
  { vertices := [
      { x := 1, y := 2 }, { x := 2, y := 2 }, { x := 3, y := 3 }, { x := 3, y := 4 },
      { x := 4, y := 5 }, { x := 5, y := 5 }, { x := 6, y := 5 }, { x := 6, y := 4 },
      { x := 5, y := 3 }, { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 3, y := 1 },
      { x := 2, y := 1 }, { x := 1, y := 1 }
    ]
  }

/-- Theorem stating that the area of the fourteen-sided polygon is 14 square centimeters -/
theorem fourteenSidedPolygonArea :
  calculatePolygonArea fourteenSidedPolygon = 14 := by sorry

end fourteenSidedPolygonArea_l897_89762


namespace kids_at_reunion_l897_89710

theorem kids_at_reunion (adults : ℕ) (tables : ℕ) (people_per_table : ℕ) 
  (h1 : adults = 123)
  (h2 : tables = 14)
  (h3 : people_per_table = 12) :
  tables * people_per_table - adults = 45 :=
by sorry

end kids_at_reunion_l897_89710


namespace banner_coverage_count_l897_89720

/-- A banner is a 2x5 grid with one 1x1 square removed from one of the four corners -/
def Banner : Type := Unit

/-- The grid table dimensions -/
def grid_width : Nat := 18
def grid_height : Nat := 9

/-- The number of banners used to cover the grid -/
def num_banners : Nat := 18

/-- The number of squares in each banner -/
def squares_per_banner : Nat := 9

/-- The number of pairs of banners -/
def num_pairs : Nat := 9

theorem banner_coverage_count : 
  (2 ^ num_pairs : Nat) + (2 ^ num_pairs : Nat) = 1024 := by sorry

end banner_coverage_count_l897_89720


namespace distribute_teachers_count_l897_89723

/-- The number of ways to distribute teachers to schools -/
def distribute_teachers : ℕ :=
  let chinese_teachers := 2
  let math_teachers := 4
  let total_teachers := chinese_teachers + math_teachers
  let schools := 2
  let teachers_per_school := 3
  let ways_to_choose_math := Nat.choose math_teachers (teachers_per_school - 1)
  ways_to_choose_math * schools

/-- Theorem stating that the number of ways to distribute teachers is 12 -/
theorem distribute_teachers_count : distribute_teachers = 12 := by
  sorry

end distribute_teachers_count_l897_89723


namespace xiaoqiang_father_annual_income_l897_89793

def monthly_salary : ℕ := 4380
def months_in_year : ℕ := 12

theorem xiaoqiang_father_annual_income :
  monthly_salary * months_in_year = 52560 := by sorry

end xiaoqiang_father_annual_income_l897_89793


namespace blanch_breakfast_slices_l897_89794

/-- The number of pizza slices Blanch ate for breakfast -/
def breakfast_slices : ℕ := sorry

/-- The total number of pizza slices Blanch started with -/
def total_slices : ℕ := 15

/-- The number of pizza slices Blanch ate for lunch -/
def lunch_slices : ℕ := 2

/-- The number of pizza slices Blanch ate as a snack -/
def snack_slices : ℕ := 2

/-- The number of pizza slices Blanch ate for dinner -/
def dinner_slices : ℕ := 5

/-- The number of pizza slices left at the end -/
def leftover_slices : ℕ := 2

/-- Theorem stating that Blanch ate 4 slices for breakfast -/
theorem blanch_breakfast_slices : 
  breakfast_slices = total_slices - (lunch_slices + snack_slices + dinner_slices + leftover_slices) :=
by sorry

end blanch_breakfast_slices_l897_89794


namespace square_of_complex_number_l897_89784

theorem square_of_complex_number :
  let i : ℂ := Complex.I
  (5 - 3 * i)^2 = 16 - 30 * i :=
by sorry

end square_of_complex_number_l897_89784


namespace car_uphill_speed_l897_89747

/-- Given a car's travel information, prove that its uphill speed is 80 km/hour. -/
theorem car_uphill_speed
  (downhill_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (downhill_time : ℝ)
  (uphill_time : ℝ)
  (h1 : downhill_speed = 50)
  (h2 : total_time = 15)
  (h3 : total_distance = 650)
  (h4 : downhill_time = 5)
  (h5 : uphill_time = 5)
  : ∃ (uphill_speed : ℝ), uphill_speed = 80 := by
  sorry

end car_uphill_speed_l897_89747


namespace lowest_unique_score_above_90_l897_89796

/-- Represents the scoring system for the modified AHSME exam -/
def score (c w : ℕ) : ℕ := 35 + 4 * c - w

/-- The total number of questions in the exam -/
def total_questions : ℕ := 35

theorem lowest_unique_score_above_90 :
  ∀ s : ℕ,
  s > 90 →
  (∃! (c w : ℕ), c + w ≤ total_questions ∧ score c w = s) →
  (∀ s' : ℕ, 90 < s' ∧ s' < s → ¬∃! (c w : ℕ), c + w ≤ total_questions ∧ score c w = s') →
  s = 91 :=
sorry

end lowest_unique_score_above_90_l897_89796


namespace smallest_divisor_of_720_two_divides_720_smallest_positive_divisor_of_720_is_two_l897_89790

theorem smallest_divisor_of_720 : 
  ∀ n : ℕ, n > 0 → n ∣ 720 → n ≥ 2 :=
by
  sorry

theorem two_divides_720 : 2 ∣ 720 :=
by
  sorry

theorem smallest_positive_divisor_of_720_is_two : 
  ∃ (d : ℕ), d > 0 ∧ d ∣ 720 ∧ ∀ n : ℕ, n > 0 → n ∣ 720 → n ≥ d :=
by
  sorry

end smallest_divisor_of_720_two_divides_720_smallest_positive_divisor_of_720_is_two_l897_89790


namespace root_range_implies_m_range_l897_89771

theorem root_range_implies_m_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < -1 ∧ x₂ > 1 ∧ 
   x₁^2 + (m^2 - 1)*x₁ + m - 2 = 0 ∧
   x₂^2 + (m^2 - 1)*x₂ + m - 2 = 0) →
  -2 < m ∧ m < 0 := by
sorry

end root_range_implies_m_range_l897_89771


namespace saline_drip_rate_l897_89725

/-- Proves that the saline drip makes 20 drops per minute given the treatment conditions -/
theorem saline_drip_rate (treatment_duration : ℕ) (drops_per_ml : ℚ) (total_volume : ℚ) :
  treatment_duration = 2 * 60 →  -- 2 hours in minutes
  drops_per_ml = 100 / 5 →       -- 100 drops per 5 ml
  total_volume = 120 →           -- 120 ml total volume
  (total_volume * drops_per_ml) / treatment_duration = 20 := by
  sorry

#check saline_drip_rate

end saline_drip_rate_l897_89725


namespace max_sum_nonnegative_l897_89735

theorem max_sum_nonnegative (a b c d : ℝ) (h : a + b + c + d = 0) :
  max a b + max a c + max a d + max b c + max b d + max c d ≥ 0 := by
  sorry

end max_sum_nonnegative_l897_89735


namespace line_quadrants_l897_89770

/-- Given a line ax + by + c = 0 where ab < 0 and bc < 0, 
    the line passes through the first, second, and third quadrants -/
theorem line_quadrants (a b c : ℝ) (hab : a * b < 0) (hbc : b * c < 0) :
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (x1 > 0 ∧ y1 > 0) ∧  -- First quadrant
    (x2 < 0 ∧ y2 > 0) ∧  -- Second quadrant
    (x3 < 0 ∧ y3 < 0) ∧  -- Third quadrant
    (a * x1 + b * y1 + c = 0) ∧
    (a * x2 + b * y2 + c = 0) ∧
    (a * x3 + b * y3 + c = 0) :=
by sorry

end line_quadrants_l897_89770


namespace power_function_value_l897_89766

theorem power_function_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = x ^ a) →
  f 2 = Real.sqrt 2 / 2 →
  f 4 = 1 / 2 := by
  sorry

end power_function_value_l897_89766


namespace cube_square_third_prime_times_fourth_prime_l897_89719

def third_smallest_prime : Nat := 5

def fourth_smallest_prime : Nat := 7

theorem cube_square_third_prime_times_fourth_prime :
  (third_smallest_prime ^ 2) ^ 3 * fourth_smallest_prime = 109375 := by
  sorry

end cube_square_third_prime_times_fourth_prime_l897_89719


namespace sequence_general_term_l897_89769

theorem sequence_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + n) :
  ∀ n, a n = 2 * n :=
sorry

end sequence_general_term_l897_89769


namespace team_combinations_l897_89708

-- Define the number of people and team size
def total_people : ℕ := 7
def team_size : ℕ := 4

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem team_combinations : combination total_people team_size = 35 := by
  sorry

end team_combinations_l897_89708


namespace product_and_reciprocal_sum_l897_89753

theorem product_and_reciprocal_sum (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ x * y = 16 ∧ 1 / x = 3 * (1 / y) → x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end product_and_reciprocal_sum_l897_89753


namespace trig_identities_l897_89705

theorem trig_identities :
  (Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4) ∧
  (Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4) ∧
  (Real.sin (18 * π / 180) = (-1 + Real.sqrt 5) / 4) ∧
  (Real.cos (18 * π / 180) = Real.sqrt (10 + 2 * Real.sqrt 5) / 4) := by
  sorry

end trig_identities_l897_89705


namespace jason_football_games_l897_89797

/-- Given the number of football games Jason attended this month and last month,
    and the total number of games he plans to attend,
    prove that the number of games he plans to attend next month is 16. -/
theorem jason_football_games (this_month last_month total : ℕ) 
    (h1 : this_month = 11)
    (h2 : last_month = 17)
    (h3 : total = 44) :
    total - (this_month + last_month) = 16 := by
  sorry

end jason_football_games_l897_89797


namespace parallel_tangents_l897_89751

/-- A homogeneous differential equation y' = φ(y/x) -/
noncomputable def homogeneous_de (φ : ℝ → ℝ) (x y : ℝ) : ℝ := φ (y / x)

/-- The slope of the tangent line at a point (x, y) -/
noncomputable def tangent_slope (φ : ℝ → ℝ) (x y : ℝ) : ℝ := homogeneous_de φ x y

theorem parallel_tangents (φ : ℝ → ℝ) (x y x₁ y₁ : ℝ) (hx : x ≠ 0) (hx₁ : x₁ ≠ 0) 
  (h_corresp : y / x = y₁ / x₁) :
  tangent_slope φ x y = tangent_slope φ x₁ y₁ := by
  sorry

end parallel_tangents_l897_89751


namespace estimate_battery_usage_l897_89756

/-- Estimates the total number of batteries used by a class based on a sample. -/
theorem estimate_battery_usage
  (sample_size : ℕ)
  (sample_total : ℕ)
  (class_size : ℕ)
  (h1 : sample_size = 6)
  (h2 : sample_total = 168)
  (h3 : class_size = 45) :
  (sample_total / sample_size) * class_size = 1260 :=
by sorry

end estimate_battery_usage_l897_89756


namespace evie_shell_collection_l897_89714

/-- The number of shells Evie collects per day -/
def shells_per_day : ℕ := 10

/-- The number of shells Evie gives to her brother -/
def shells_given : ℕ := 2

/-- The number of shells Evie has left after giving some to her brother -/
def shells_left : ℕ := 58

/-- The number of days Evie collected shells -/
def collection_days : ℕ := 6

theorem evie_shell_collection :
  shells_per_day * collection_days - shells_given = shells_left :=
by sorry

end evie_shell_collection_l897_89714
