import Mathlib

namespace solve_for_x_l3647_364718

theorem solve_for_x (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : y = 1) : x = 8 := by
  sorry

end solve_for_x_l3647_364718


namespace smallest_square_addition_l3647_364751

theorem smallest_square_addition (n : ℕ) (h : n = 2020) : 
  ∃ k : ℕ, k = 1 ∧ 
  (∃ m : ℕ, (n - 1) * n * (n + 1) * (n + 2) + k = m^2) ∧
  (∀ j : ℕ, j < k → ¬∃ m : ℕ, (n - 1) * n * (n + 1) * (n + 2) + j = m^2) :=
by sorry

#check smallest_square_addition

end smallest_square_addition_l3647_364751


namespace solution_set_quadratic_inequality_l3647_364752

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 5*x + 6 ≤ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by sorry

end solution_set_quadratic_inequality_l3647_364752


namespace prob_at_least_one_white_l3647_364717

/-- The number of red balls in the bag -/
def num_red : ℕ := 3

/-- The number of white balls in the bag -/
def num_white : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red + num_white

/-- The number of balls drawn -/
def num_drawn : ℕ := 2

/-- The probability of drawing at least one white ball when selecting two balls -/
theorem prob_at_least_one_white :
  (1 : ℚ) - (num_red.choose num_drawn : ℚ) / (total_balls.choose num_drawn : ℚ) = 7 / 10 := by
  sorry

end prob_at_least_one_white_l3647_364717


namespace arrangement_count_l3647_364772

def number_of_arrangements (total_people : ℕ) (selected_people : ℕ) 
  (meeting_a_participants : ℕ) (meeting_b_participants : ℕ) (meeting_c_participants : ℕ) : ℕ :=
  Nat.choose total_people selected_people * 
  Nat.choose selected_people meeting_a_participants * 
  Nat.choose (selected_people - meeting_a_participants) meeting_b_participants

theorem arrangement_count : 
  number_of_arrangements 10 4 2 1 1 = 2520 := by
  sorry

end arrangement_count_l3647_364772


namespace house_rent_expenditure_l3647_364780

-- Define the given parameters
def total_income : ℝ := 1000
def petrol_percentage : ℝ := 0.30
def house_rent_percentage : ℝ := 0.10
def petrol_expenditure : ℝ := 300

-- Define the theorem
theorem house_rent_expenditure :
  let remaining_income := total_income - petrol_expenditure
  remaining_income * house_rent_percentage = 70 := by
  sorry

end house_rent_expenditure_l3647_364780


namespace cricket_team_right_handed_players_l3647_364788

/-- Represents the composition of a cricket team -/
structure CricketTeam where
  total_players : ℕ
  throwers : ℕ
  hitters : ℕ
  runners : ℕ
  left_handed_hitters : ℕ
  left_handed_runners : ℕ

/-- Calculates the total number of right-handed players in a cricket team -/
def right_handed_players (team : CricketTeam) : ℕ :=
  team.throwers + (team.hitters - team.left_handed_hitters) + (team.runners - team.left_handed_runners)

/-- Theorem stating the total number of right-handed players in the given cricket team -/
theorem cricket_team_right_handed_players :
  ∃ (team : CricketTeam),
    team.total_players = 300 ∧
    team.throwers = 165 ∧
    team.hitters = team.runners ∧
    team.hitters + team.runners = team.total_players - team.throwers ∧
    team.left_handed_hitters * 5 = team.hitters * 2 ∧
    team.left_handed_runners * 7 = team.runners * 3 ∧
    right_handed_players team = 243 :=
  sorry


end cricket_team_right_handed_players_l3647_364788


namespace sqrt_seven_to_sixth_power_l3647_364726

theorem sqrt_seven_to_sixth_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_sixth_power_l3647_364726


namespace tangent_line_at_point_one_l3647_364737

-- Define the function
def f (x : ℝ) : ℝ := x^2 + x - 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 2*x + 1

-- Theorem statement
theorem tangent_line_at_point_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let k : ℝ := f' x₀
  ∀ x y : ℝ, (k * (x - x₀) = y - y₀) ↔ (3*x - y - 2 = 0) :=
sorry

end tangent_line_at_point_one_l3647_364737


namespace complex_equation_solution_l3647_364715

theorem complex_equation_solution : 
  ∃ (z : ℂ), (5 : ℂ) + 2 * Complex.I * z = (1 : ℂ) - 6 * Complex.I * z ∧ z = Complex.I / 2 := by
  sorry

end complex_equation_solution_l3647_364715


namespace q_share_is_7200_l3647_364792

/-- Calculates the share of profit for a partner in a business partnership. -/
def calculateShareOfProfit (investment1 : ℕ) (investment2 : ℕ) (totalProfit : ℕ) : ℕ :=
  let totalInvestment := investment1 + investment2
  (investment2 * totalProfit) / totalInvestment

/-- Theorem stating that Q's share of the profit is 7200 given the specified investments and total profit. -/
theorem q_share_is_7200 :
  calculateShareOfProfit 54000 36000 18000 = 7200 := by
  sorry

#eval calculateShareOfProfit 54000 36000 18000

end q_share_is_7200_l3647_364792


namespace vector_dot_product_equality_l3647_364776

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-1, 3)
def C : ℝ × ℝ := (2, 1)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def vector_BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_equality : 
  dot_product vector_AB (2 • vector_AC + vector_BC) = -14 := by sorry

end vector_dot_product_equality_l3647_364776


namespace parakeets_per_cage_l3647_364778

/-- Given a pet store with bird cages, prove the number of parakeets in each cage -/
theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (h1 : num_cages = 9)
  (h2 : parrots_per_cage = 2)
  (h3 : total_birds = 72) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 6 := by
  sorry

end parakeets_per_cage_l3647_364778


namespace triangle_side_length_l3647_364734

/-- An equilateral triangle with a point inside and perpendiculars to its sides. -/
structure TriangleWithPoint where
  /-- Side length of the equilateral triangle -/
  side_length : ℝ
  /-- Distance from the point to side AB -/
  dist_to_AB : ℝ
  /-- Distance from the point to side BC -/
  dist_to_BC : ℝ
  /-- Distance from the point to side CA -/
  dist_to_CA : ℝ
  /-- The triangle is equilateral -/
  equilateral : side_length > 0
  /-- The point is inside the triangle -/
  point_inside : dist_to_AB > 0 ∧ dist_to_BC > 0 ∧ dist_to_CA > 0

/-- Theorem: If the perpendicular distances are 2, 2√2, and 4, then the side length is 4√3 + (4√6)/3 -/
theorem triangle_side_length (t : TriangleWithPoint) 
  (h1 : t.dist_to_AB = 2) 
  (h2 : t.dist_to_BC = 2 * Real.sqrt 2) 
  (h3 : t.dist_to_CA = 4) : 
  t.side_length = 4 * Real.sqrt 3 + (4 * Real.sqrt 6) / 3 := by
  sorry

end triangle_side_length_l3647_364734


namespace fourth_cat_weight_proof_l3647_364705

/-- The weight of the fourth cat given the weights of three cats and the average weight of all four cats -/
def fourth_cat_weight (weight1 weight2 weight3 average_weight : ℝ) : ℝ :=
  4 * average_weight - (weight1 + weight2 + weight3)

/-- Theorem stating that given the specific weights of three cats and the average weight of all four cats, the weight of the fourth cat is 9.3 pounds -/
theorem fourth_cat_weight_proof :
  fourth_cat_weight 12 12 14.7 12 = 9.3 := by
  sorry

end fourth_cat_weight_proof_l3647_364705


namespace second_to_first_l3647_364764

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate for a point being in the second quadrant -/
def inSecondQuadrant (p : Point) : Prop := p.x < 0 ∧ p.y > 0

/-- Predicate for a point being in the first quadrant -/
def inFirstQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y > 0

/-- Theorem: If A(m,n) is in the second quadrant, then B(-m,|n|) is in the first quadrant -/
theorem second_to_first (m n : ℝ) :
  inSecondQuadrant ⟨m, n⟩ → inFirstQuadrant ⟨-m, |n|⟩ := by
  sorry

end second_to_first_l3647_364764


namespace parallel_lines_corresponding_angles_l3647_364774

/-- Represents a line in a plane -/
structure Line where
  -- Add necessary fields here
  
/-- Represents an angle in a plane -/
structure Angle where
  -- Add necessary fields here

/-- Represents a plane -/
structure Plane where
  -- Add necessary fields here

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  sorry

/-- A line intersects two other lines -/
def intersects (l : Line) (l1 l2 : Line) : Prop :=
  sorry

/-- Two angles are corresponding angles -/
def corresponding_angles (a1 a2 : Angle) (l1 l2 l : Line) : Prop :=
  sorry

/-- Two angles are equal -/
def angles_equal (a1 a2 : Angle) : Prop :=
  sorry

/-- Two angles are supplementary -/
def angles_supplementary (a1 a2 : Angle) : Prop :=
  sorry

/-- Main theorem: If two lines are parallel and intersected by a transversal,
    then the corresponding angles are either equal or supplementary -/
theorem parallel_lines_corresponding_angles 
  (p : Plane) (l1 l2 l : Line) (a1 a2 : Angle) :
  parallel l1 l2 → intersects l l1 l2 → corresponding_angles a1 a2 l1 l2 l →
  angles_equal a1 a2 ∨ angles_supplementary a1 a2 :=
sorry

end parallel_lines_corresponding_angles_l3647_364774


namespace tower_height_l3647_364744

/-- The height of a tower given specific angle measurements -/
theorem tower_height (angle1 angle2 : Real) (distance : Real) (height : Real) : 
  angle1 = Real.pi / 6 →  -- 30 degrees in radians
  angle2 = Real.pi / 4 →  -- 45 degrees in radians
  distance = 20 → 
  Real.tan angle1 = height / (height + distance) →
  height = 10 * (Real.sqrt 3 + 1) :=
by sorry

end tower_height_l3647_364744


namespace three_digit_sum_property_l3647_364794

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_sum_property (n : ℕ) :
  100 ≤ n ∧ n < 1000 ∧ 
  digit_sum n = 3 * digit_sum (n - 75) →
  n = 189 ∨ n = 675 := by
sorry

end three_digit_sum_property_l3647_364794


namespace probability_between_R_and_S_l3647_364786

/-- Given a line segment PQ with points R and S, where PQ = 4PR and PQ = 8QR,
    the probability that a randomly selected point on PQ lies between R and S is 5/8. -/
theorem probability_between_R_and_S (P Q R S : Real) (h1 : Q - P = 4 * (R - P)) (h2 : Q - P = 8 * (Q - R)) :
  (S - R) / (Q - P) = 5 / 8 := by
  sorry

end probability_between_R_and_S_l3647_364786


namespace gretchen_earnings_l3647_364741

/-- Gretchen's caricature business --/
def caricature_problem (price_per_drawing : ℚ) (saturday_sales : ℕ) (sunday_sales : ℕ) : ℚ :=
  (saturday_sales + sunday_sales : ℚ) * price_per_drawing

/-- Theorem stating the total money Gretchen made --/
theorem gretchen_earnings :
  caricature_problem 20 24 16 = 800 := by
  sorry

end gretchen_earnings_l3647_364741


namespace linda_travel_distance_l3647_364754

/-- Represents the travel data for one day --/
structure DayTravel where
  totalTime : ℕ
  timePerMile : ℕ

/-- Calculates the distance traveled in a day --/
def distanceTraveled (day : DayTravel) : ℚ :=
  day.totalTime / day.timePerMile

/-- Represents Linda's travel data over three days --/
structure ThreeDayTravel where
  day1 : DayTravel
  day2 : DayTravel
  day3 : DayTravel

/-- The main theorem to prove --/
theorem linda_travel_distance 
  (travel : ThreeDayTravel)
  (time_condition : travel.day1.totalTime = 60 ∧ 
                    travel.day2.totalTime = 75 ∧ 
                    travel.day3.totalTime = 90)
  (time_increase : travel.day2.timePerMile = travel.day1.timePerMile + 3 ∧
                   travel.day3.timePerMile = travel.day2.timePerMile + 3)
  (integer_distance : ∀ d : DayTravel, d ∈ [travel.day1, travel.day2, travel.day3] → 
                      (distanceTraveled d).den = 1)
  (integer_time : ∀ d : DayTravel, d ∈ [travel.day1, travel.day2, travel.day3] → 
                  d.timePerMile > 0) :
  (distanceTraveled travel.day1 + distanceTraveled travel.day2 + distanceTraveled travel.day3 : ℚ) = 15 := by
  sorry

end linda_travel_distance_l3647_364754


namespace expression_factorization_l3647_364732

theorem expression_factorization (x : ℝ) :
  (20 * x^3 + 100 * x - 10) - (-3 * x^3 + 5 * x - 15) = 5 * (23 * x^3 + 19 * x + 1) := by
  sorry

end expression_factorization_l3647_364732


namespace compound_interest_rate_l3647_364795

/-- Given an initial amount P at compound interest that sums to 17640 after 2 years
    and 22050 after 3 years, the annual interest rate is 25%. -/
theorem compound_interest_rate (P : ℝ) : 
  P * (1 + 0.25)^2 = 17640 ∧ P * (1 + 0.25)^3 = 22050 → 0.25 = 0.25 := by
  sorry

end compound_interest_rate_l3647_364795


namespace red_team_score_l3647_364775

theorem red_team_score (chuck_team_score : ℕ) (score_difference : ℕ) :
  chuck_team_score = 95 →
  score_difference = 19 →
  chuck_team_score - score_difference = 76 :=
by
  sorry

end red_team_score_l3647_364775


namespace roots_form_triangle_l3647_364777

/-- The roots of the equation (x-1)(x^2-2x+m) = 0 can form a triangle if and only if 3/4 < m ≤ 1 -/
theorem roots_form_triangle (m : ℝ) : 
  (∃ a b c : ℝ, 
    (a - 1) * (a^2 - 2*a + m) = 0 ∧
    (b - 1) * (b^2 - 2*b + m) = 0 ∧
    (c - 1) * (c^2 - 2*c + m) = 0 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ c + a > b) ↔
  (3/4 < m ∧ m ≤ 1) :=
by sorry

end roots_form_triangle_l3647_364777


namespace diamond_ratio_sixteen_two_over_two_sixteen_l3647_364700

-- Define the diamond operation
def diamond (n m : ℕ) : ℕ := n^4 * m^2

-- State the theorem
theorem diamond_ratio_sixteen_two_over_two_sixteen : 
  (diamond 16 2) / (diamond 2 16) = 64 := by sorry

end diamond_ratio_sixteen_two_over_two_sixteen_l3647_364700


namespace arithmetic_sequence_zero_term_l3647_364779

/-- An arithmetic sequence with common difference d ≠ 0 -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h : d ≠ 0  -- d is non-zero
  seq : ∀ n : ℕ, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- The theorem statement -/
theorem arithmetic_sequence_zero_term
  (seq : ArithmeticSequence)
  (h : seq.a 3 + seq.a 9 = seq.a 10 - seq.a 8) :
  ∃! n : ℕ, seq.a n = 0 ∧ n = 5 := by
  sorry

end arithmetic_sequence_zero_term_l3647_364779


namespace emails_morning_evening_l3647_364701

def morning_emails : ℕ := 3
def evening_emails : ℕ := 8

theorem emails_morning_evening : 
  morning_emails + evening_emails = 11 :=
by sorry

end emails_morning_evening_l3647_364701


namespace identity_function_characterization_l3647_364755

theorem identity_function_characterization (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y)
  (h_one : f 1 = 1)
  (h_additive : ∀ x y, f (x + y) = f x + f y) :
  ∀ x, f x = x :=
sorry

end identity_function_characterization_l3647_364755


namespace isosceles_right_triangle_intercept_l3647_364708

/-- Given a line that intersects a circle centered at the origin, 
    prove that the line forms an isosceles right triangle with the origin 
    if and only if the absolute value of its y-intercept equals √2. -/
theorem isosceles_right_triangle_intercept (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    A.1 - A.2 + a = 0 ∧ 
    B.1 - B.2 + a = 0 ∧ 
    A.1^2 + A.2^2 = 2 ∧ 
    B.1^2 + B.2^2 = 2 ∧ 
    (A.1 - 0)^2 + (A.2 - 0)^2 = (B.1 - 0)^2 + (B.2 - 0)^2 ∧ 
    (A.1 - 0) * (B.1 - 0) + (A.2 - 0) * (B.2 - 0) = 0) ↔ 
  |a| = Real.sqrt 2 :=
sorry

end isosceles_right_triangle_intercept_l3647_364708


namespace probability_of_white_ball_is_five_eighths_l3647_364767

/-- Represents the color of a ball -/
inductive Color
| White
| NonWhite

/-- Represents a bag of balls -/
def Bag := List Color

/-- The number of balls initially in the bag -/
def initialBallCount : Nat := 3

/-- Generates all possible initial configurations of the bag -/
def allPossibleInitialBags : List Bag :=
  sorry

/-- Adds a white ball to a bag -/
def addWhiteBall (bag : Bag) : Bag :=
  sorry

/-- Calculates the probability of drawing a white ball from a bag -/
def probabilityOfWhite (bag : Bag) : Rat :=
  sorry

/-- Calculates the average probability across all possible scenarios -/
def averageProbability (bags : List Bag) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem probability_of_white_ball_is_five_eighths :
  averageProbability (allPossibleInitialBags.map addWhiteBall) = 5/8 :=
  sorry

end probability_of_white_ball_is_five_eighths_l3647_364767


namespace sum_A_B_equals_24_l3647_364727

theorem sum_A_B_equals_24 (A B : ℚ) (h1 : (1 : ℚ) / 6 * (1 : ℚ) / 3 = 1 / (A * 3))
  (h2 : (1 : ℚ) / 6 * (1 : ℚ) / 3 = 1 / B) : A + B = 24 := by
  sorry

end sum_A_B_equals_24_l3647_364727


namespace box_volume_l3647_364716

/-- Given a rectangular box with face areas 30, 18, and 45 square centimeters, 
    its volume is 90√3 cubic centimeters. -/
theorem box_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 18) 
  (h3 : c * a = 45) : 
  a * b * c = 90 * Real.sqrt 3 := by
  sorry

end box_volume_l3647_364716


namespace area_at_stage_8_l3647_364753

/-- The side length of each square in inches -/
def square_side : ℝ := 4

/-- The number of squares at a given stage -/
def num_squares (stage : ℕ) : ℕ := stage

/-- The area of the rectangle at a given stage in square inches -/
def rectangle_area (stage : ℕ) : ℝ :=
  (num_squares stage) * (square_side ^ 2)

/-- Theorem: The area of the rectangle at Stage 8 is 128 square inches -/
theorem area_at_stage_8 : rectangle_area 8 = 128 := by
  sorry

end area_at_stage_8_l3647_364753


namespace expression_simplification_l3647_364729

theorem expression_simplification (m : ℝ) (h1 : m ≠ 2) (h2 : m ≠ -3) :
  (m - (4*m - 9) / (m - 2)) / ((m^2 - 9) / (m - 2)) = (m - 3) / (m + 3) := by
  sorry

end expression_simplification_l3647_364729


namespace largest_product_bound_l3647_364739

theorem largest_product_bound (a : Fin 1985 → Fin 1985) (h : Function.Bijective a) :
  (Finset.range 1985).sup (λ k => (k + 1) * a (k + 1)) ≥ 993^2 := by
  sorry

end largest_product_bound_l3647_364739


namespace mountain_climbing_equivalence_l3647_364731

/-- Given the elevations of two mountains and the number of times one is climbed,
    calculate how many times the other mountain needs to be climbed to cover the same distance. -/
theorem mountain_climbing_equivalence 
  (hugo_elevation : ℕ) 
  (elevation_difference : ℕ) 
  (hugo_climbs : ℕ) : 
  hugo_elevation = 10000 →
  elevation_difference = 2500 →
  hugo_climbs = 3 →
  (hugo_elevation * hugo_climbs) / (hugo_elevation - elevation_difference) = 4 := by
  sorry

end mountain_climbing_equivalence_l3647_364731


namespace vector_simplification_l3647_364789

variable {V : Type*} [AddCommGroup V]

variable (A B C M O : V)

theorem vector_simplification :
  (B - A) + (B - M) + (O - B) + (C - B) + (M - O) = C - A :=
by sorry

end vector_simplification_l3647_364789


namespace triangle_abc_properties_l3647_364770

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Area of triangle ABC
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 15 ∧
  -- Relationship between b and c
  b - c = 2 ∧
  -- Given cosine of A
  Real.cos A = -(1/4) →
  -- Conclusions
  a = 8 ∧
  Real.sin C = Real.sqrt 15 / 8 ∧
  Real.cos (2 * A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 := by
sorry


end triangle_abc_properties_l3647_364770


namespace expand_expression_l3647_364762

theorem expand_expression (x y z : ℝ) : 
  (x + 5) * (3 * y + 2 * z + 15) = 3 * x * y + 2 * x * z + 15 * x + 15 * y + 10 * z + 75 := by
  sorry

end expand_expression_l3647_364762


namespace candy_box_problem_l3647_364707

theorem candy_box_problem (milk_chocolate dark_chocolate milk_almond : ℕ) 
  (h1 : milk_chocolate = 25)
  (h2 : dark_chocolate = 25)
  (h3 : milk_almond = 25)
  (h4 : ∀ chocolate_type, chocolate_type = milk_chocolate ∨ 
                          chocolate_type = dark_chocolate ∨ 
                          chocolate_type = milk_almond ∨ 
                          chocolate_type = white_chocolate →
        chocolate_type = (milk_chocolate + dark_chocolate + milk_almond + white_chocolate) / 4) :
  white_chocolate = 25 :=
by
  sorry

end candy_box_problem_l3647_364707


namespace garden_planting_area_l3647_364760

def garden_length : ℝ := 18
def garden_width : ℝ := 14
def pond_length : ℝ := 4
def pond_width : ℝ := 2
def flower_bed_base : ℝ := 3
def flower_bed_height : ℝ := 2

theorem garden_planting_area :
  garden_length * garden_width - (pond_length * pond_width + 1/2 * flower_bed_base * flower_bed_height) = 241 := by
  sorry

end garden_planting_area_l3647_364760


namespace range_of_a_for_local_max_l3647_364769

noncomputable def f (a b x : ℝ) : ℝ := Real.log x + a * x^2 + b * x

theorem range_of_a_for_local_max (a b : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f a b x ≤ f a b 1) →
  a < 1/2 :=
by sorry

end range_of_a_for_local_max_l3647_364769


namespace x_less_equal_two_l3647_364742

theorem x_less_equal_two (x : ℝ) (h : Real.sqrt ((x - 2)^2) = 2 - x) : x ≤ 2 := by
  sorry

end x_less_equal_two_l3647_364742


namespace function_passes_through_point_l3647_364704

theorem function_passes_through_point (a : ℝ) (h : 0 < a ∧ a < 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x - 1)
  f 1 = 2 := by sorry

end function_passes_through_point_l3647_364704


namespace right_angle_in_triangle_l3647_364743

theorem right_angle_in_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (A > 0 ∧ B > 0 ∧ C > 0) →
  (A + B + C = Real.pi) →
  -- Side lengths are positive
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Given conditions
  (Real.sin B = Real.sin (2 * A)) →
  (c = 2 * a) →
  -- Conclusion
  C = Real.pi / 2 := by
sorry

end right_angle_in_triangle_l3647_364743


namespace sum_of_binary_digits_sum_l3647_364771

/-- The sum of binary digits of a positive integer -/
def s (n : ℕ+) : ℕ := sorry

/-- The sum of s(n) for n from 1 to 2^k -/
def S (k : ℕ+) : ℕ :=
  Finset.sum (Finset.range (2^k.val)) (fun i => s ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem: S(k) = 2^(k-1) * k + 1 for all positive integers k -/
theorem sum_of_binary_digits_sum (k : ℕ+) : S k = 2^(k.val - 1) * k.val + 1 := by
  sorry

end sum_of_binary_digits_sum_l3647_364771


namespace units_digit_of_G_1000_l3647_364763

-- Define G_n
def G (n : ℕ) : ℕ := 3^(3^n) + 1

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_G_1000 : unitsDigit (G 1000) = 4 := by
  sorry

end units_digit_of_G_1000_l3647_364763


namespace total_chocolate_bars_l3647_364758

/-- Represents the number of chocolate bars in a large box -/
def chocolateBarsInLargeBox (smallBoxes : ℕ) (barsPerSmallBox : ℕ) : ℕ :=
  smallBoxes * barsPerSmallBox

/-- Proves that the total number of chocolate bars in the large box is 500 -/
theorem total_chocolate_bars :
  chocolateBarsInLargeBox 20 25 = 500 := by
  sorry

end total_chocolate_bars_l3647_364758


namespace horner_v2_at_2_l3647_364735

def horner_polynomial (x : ℝ) : ℝ := 2*x^7 + x^6 + x^4 + x^2 + 1

def horner_v2 (x : ℝ) : ℝ := 
  let v0 := 2
  let v1 := 2*x + 1
  v1 * x

theorem horner_v2_at_2 : horner_v2 2 = 10 := by
  sorry

#eval horner_v2 2

end horner_v2_at_2_l3647_364735


namespace black_haired_girls_count_l3647_364723

theorem black_haired_girls_count (initial_total : ℕ) (added_blonde : ℕ) (initial_blonde : ℕ) : 
  initial_total = 80 → 
  added_blonde = 10 → 
  initial_blonde = 30 → 
  initial_total + added_blonde - (initial_blonde + added_blonde) = 50 := by
sorry

end black_haired_girls_count_l3647_364723


namespace sixteen_integer_lengths_l3647_364749

/-- Represents a right triangle with integer leg lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths possible for line segments
    drawn from a vertex to points on the hypotenuse -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

/-- The main theorem stating that for a right triangle with legs 24 and 25,
    there are exactly 16 distinct integer lengths possible -/
theorem sixteen_integer_lengths :
  ∃ (t : RightTriangle), t.de = 24 ∧ t.ef = 25 ∧ countIntegerLengths t = 16 := by
  sorry

end sixteen_integer_lengths_l3647_364749


namespace tangent_and_intersection_l3647_364799

-- Define the curve C
def C (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 - 9 * x^2 + 4

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := -12 * x + 8

-- Theorem statement
theorem tangent_and_intersection :
  -- The tangent line at x = 1 has the equation y = -12x + 8
  (∀ x, tangent_line x = -12 * x + 8) ∧
  -- The tangent line touches the curve at x = 1
  (C 1 = tangent_line 1) ∧
  -- The tangent line is indeed tangent to the curve at x = 1
  (deriv C 1 = -12) ∧
  -- The tangent line intersects the curve at two additional points
  (C (-2) = tangent_line (-2)) ∧
  (C (2/3) = tangent_line (2/3)) :=
sorry

end tangent_and_intersection_l3647_364799


namespace compound_interest_calculation_l3647_364750

/-- Compound interest calculation --/
theorem compound_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℕ) 
  (h1 : principal = 5000)
  (h2 : rate = 0.1)
  (h3 : time = 2) :
  principal * (1 + rate) ^ time = 6050 := by
  sorry

end compound_interest_calculation_l3647_364750


namespace students_present_l3647_364766

theorem students_present (total : ℕ) (absent_percent : ℚ) (present : ℕ) : 
  total = 50 → 
  absent_percent = 1/10 → 
  present = total - (total * (absent_percent : ℚ)).num / (absent_percent : ℚ).den → 
  present = 45 := by sorry

end students_present_l3647_364766


namespace car_speed_problem_l3647_364702

/-- The speed of Car B in km/h -/
def speed_B : ℝ := 35

/-- The time it takes Car A to catch up with Car B when traveling at 50 km/h -/
def time_50 : ℝ := 6

/-- The time it takes Car A to catch up with Car B when traveling at 80 km/h -/
def time_80 : ℝ := 2

/-- The speed of Car A in the first scenario (km/h) -/
def speed_A1 : ℝ := 50

/-- The speed of Car A in the second scenario (km/h) -/
def speed_A2 : ℝ := 80

theorem car_speed_problem :
  (speed_A1 * time_50 - speed_B * time_50 = speed_A2 * time_80 - speed_B * time_80) ∧
  speed_B = 35 := by sorry

end car_speed_problem_l3647_364702


namespace geometric_sum_n_eq_1_l3647_364738

theorem geometric_sum_n_eq_1 (a : ℝ) (h : a ≠ 1) :
  1 + a = (1 - a^3) / (1 - a) := by
sorry

end geometric_sum_n_eq_1_l3647_364738


namespace simplified_tax_for_leonid_business_l3647_364740

-- Define the types of tax regimes
inductive TaxRegime
  | UnifiedAgricultural
  | Simplified
  | General
  | Patent

-- Define the characteristics of a business
structure Business where
  isAgricultural : Bool
  isSmall : Bool
  hasComplexAccounting : Bool
  isNewEntrepreneur : Bool

-- Define the function to determine the appropriate tax regime
def appropriateTaxRegime (b : Business) : TaxRegime :=
  if b.isAgricultural then TaxRegime.UnifiedAgricultural
  else if b.isSmall && b.isNewEntrepreneur && !b.hasComplexAccounting then TaxRegime.Simplified
  else if !b.isSmall || b.hasComplexAccounting then TaxRegime.General
  else TaxRegime.Patent

-- Theorem statement
theorem simplified_tax_for_leonid_business :
  let leonidBusiness : Business := {
    isAgricultural := false,
    isSmall := true,
    hasComplexAccounting := false,
    isNewEntrepreneur := true
  }
  appropriateTaxRegime leonidBusiness = TaxRegime.Simplified :=
by sorry


end simplified_tax_for_leonid_business_l3647_364740


namespace count_integers_satisfying_inequality_l3647_364787

theorem count_integers_satisfying_inequality : 
  ∃ (S : Finset Int), (∀ n : Int, n ∈ S ↔ (n - 3) * (n + 5) < 0) ∧ Finset.card S = 7 :=
sorry

end count_integers_satisfying_inequality_l3647_364787


namespace base5_to_octal_polynomial_evaluation_l3647_364785

-- Define the base-5 number 1234₅
def base5_number : ℕ := 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4 * 5^0

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

-- Theorem 1: Converting base-5 to octal
theorem base5_to_octal : 
  (base5_number : ℕ).digits 8 = [3, 0, 2] := by sorry

-- Theorem 2: Evaluating the polynomial at x = 3
theorem polynomial_evaluation :
  f 3 = 21324 := by sorry

end base5_to_octal_polynomial_evaluation_l3647_364785


namespace not_all_prime_in_sequence_l3647_364747

-- Define the recursive sequence
def x (n : ℕ) (x₀ a b : ℕ) : ℕ :=
  match n with
  | 0 => x₀
  | n + 1 => x n x₀ a b * a + b

-- Theorem statement
theorem not_all_prime_in_sequence (x₀ a b : ℕ) :
  ∃ n : ℕ, ¬ Nat.Prime (x n x₀ a b) :=
by sorry

end not_all_prime_in_sequence_l3647_364747


namespace drum_size_correct_l3647_364730

/-- Represents the size of the drum in gallons -/
def D : ℝ := 54.99

/-- Represents the amount of 100% antifreeze used in gallons -/
def pure_antifreeze : ℝ := 6.11

/-- Represents the percentage of antifreeze in the final mixture -/
def final_mixture_percent : ℝ := 0.20

/-- Represents the percentage of antifreeze in the initial diluted mixture -/
def initial_diluted_percent : ℝ := 0.10

/-- Theorem stating that the given conditions result in the correct drum size -/
theorem drum_size_correct : 
  pure_antifreeze + (D - pure_antifreeze) * initial_diluted_percent = D * final_mixture_percent := by
  sorry

#check drum_size_correct

end drum_size_correct_l3647_364730


namespace smallest_candy_count_l3647_364759

theorem smallest_candy_count : ∃ (n : ℕ), 
  n = 127 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < n → ¬((m + 6) % 7 = 0 ∧ (m - 7) % 4 = 0)) ∧
  (n + 6) % 7 = 0 ∧ 
  (n - 7) % 4 = 0 :=
sorry

end smallest_candy_count_l3647_364759


namespace boat_journey_time_l3647_364719

/-- Calculates the total journey time for a boat traveling upstream and downstream in a river -/
theorem boat_journey_time 
  (river_speed : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_speed = 2) 
  (h2 : boat_speed = 6) 
  (h3 : distance = 48) : 
  (distance / (boat_speed - river_speed)) + (distance / (boat_speed + river_speed)) = 18 :=
by sorry

end boat_journey_time_l3647_364719


namespace joe_fish_compared_to_sam_l3647_364784

theorem joe_fish_compared_to_sam (harry_fish joe_fish sam_fish : ℕ) 
  (harry_joe_ratio : harry_fish = 4 * joe_fish)
  (joe_sam_ratio : ∃ x : ℕ, joe_fish = x * sam_fish)
  (sam_fish_count : sam_fish = 7)
  (harry_fish_count : harry_fish = 224) :
  ∃ x : ℕ, joe_fish = 8 * sam_fish := by
  sorry

end joe_fish_compared_to_sam_l3647_364784


namespace power_two_2017_mod_7_l3647_364725

theorem power_two_2017_mod_7 : 2^2017 % 7 = 2 := by
  sorry

end power_two_2017_mod_7_l3647_364725


namespace some_number_value_l3647_364793

theorem some_number_value : 
  ∀ some_number : ℝ, 
  (some_number * 3.6) / (0.04 * 0.1 * 0.007) = 990.0000000000001 → 
  some_number = 7.7 := by
sorry

end some_number_value_l3647_364793


namespace special_matrix_determinant_l3647_364713

/-- The determinant of a special n × n matrix A with elements a_{ij} = |i - j| -/
theorem special_matrix_determinant (n : ℕ) (hn : n > 0) :
  let A : Matrix (Fin n) (Fin n) ℤ := λ i j => |i.val - j.val|
  Matrix.det A = (-1 : ℤ)^(n-1) * (n - 1) * 2^(n-2) := by
  sorry

end special_matrix_determinant_l3647_364713


namespace rectangle_longest_side_l3647_364765

/-- Given a rectangle with perimeter 240 feet and area equal to eight times its perimeter,
    the length of its longest side is 101 feet. -/
theorem rectangle_longest_side : ∀ l w : ℝ,
  (2 * l + 2 * w = 240) →  -- perimeter is 240 feet
  (l * w = 8 * 240) →      -- area is 8 times the perimeter
  (l ≥ 0 ∧ w ≥ 0) →        -- length and width are non-negative
  (max l w = 101) :=       -- the longest side is 101 feet
by sorry

end rectangle_longest_side_l3647_364765


namespace smallest_math_club_size_l3647_364703

theorem smallest_math_club_size : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (d : ℕ), d > 0 ∧ 
    (40 * n < 100 * d) ∧ 
    (100 * d < 50 * n)) ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    ¬(∃ (k : ℕ), k > 0 ∧ 
      (40 * m < 100 * k) ∧ 
      (100 * k < 50 * m))) ∧
  n = 7 := by
sorry

end smallest_math_club_size_l3647_364703


namespace canvas_bag_lower_carbon_l3647_364745

/-- The number of shopping trips required for a canvas bag to be the lower-carbon solution -/
def shopping_trips_for_lower_carbon (canvas_co2_pounds : ℕ) (plastic_co2_ounces : ℕ) (bags_per_trip : ℕ) : ℕ :=
  let canvas_co2_ounces : ℕ := canvas_co2_pounds * 16
  let plastic_co2_per_trip : ℕ := plastic_co2_ounces * bags_per_trip
  canvas_co2_ounces / plastic_co2_per_trip

/-- Theorem stating the number of shopping trips required for the canvas bag to be lower-carbon -/
theorem canvas_bag_lower_carbon :
  shopping_trips_for_lower_carbon 600 4 8 = 300 := by
  sorry

end canvas_bag_lower_carbon_l3647_364745


namespace lines_parallel_perpendicular_l3647_364798

/-- Two lines l₁ and l₂ in the plane --/
structure Lines (m : ℝ) where
  l₁ : ℝ → ℝ → ℝ := λ x y => 2*x + (m+1)*y + 4
  l₂ : ℝ → ℝ → ℝ := λ x y => m*x + 3*y - 6

/-- The lines are parallel --/
def parallel (m : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 2 = k * m ∧ m + 1 = k * 3 ∧ 4 ≠ k * (-6)

/-- The lines are perpendicular --/
def perpendicular (m : ℝ) : Prop :=
  2 * m + 3 * (m + 1) = 0

/-- Main theorem --/
theorem lines_parallel_perpendicular (m : ℝ) :
  (parallel m ↔ m = 2) ∧ (perpendicular m ↔ m = -3/5) := by
  sorry

end lines_parallel_perpendicular_l3647_364798


namespace gardener_tree_probability_l3647_364709

theorem gardener_tree_probability (n_pine n_cedar n_fir : ℕ) 
  (h_pine : n_pine = 2)
  (h_cedar : n_cedar = 3)
  (h_fir : n_fir = 4) :
  let total_trees := n_pine + n_cedar + n_fir
  let non_fir_trees := n_pine + n_cedar
  let slots := non_fir_trees + 1
  let favorable_arrangements := Nat.choose slots n_fir
  let total_arrangements := Nat.choose total_trees n_fir
  let p := favorable_arrangements
  let q := total_arrangements
  p + q = 47 := by sorry

end gardener_tree_probability_l3647_364709


namespace center_sum_is_six_l3647_364711

/-- A circle in a shifted coordinate system -/
structure ShiftedCircle where
  -- The equation of the circle in the shifted system
  equation : (x y : ℝ) → Prop := fun x y => (x - 1)^2 + (y + 2)^2 = 4*x + 12*y + 6
  -- The shift of the coordinate system
  shift : ℝ × ℝ := (1, -2)

/-- The center of a circle in the standard coordinate system -/
def standardCenter (c : ShiftedCircle) : ℝ × ℝ := sorry

theorem center_sum_is_six (c : ShiftedCircle) : 
  let (h, k) := standardCenter c
  h + k = 6 := by sorry

end center_sum_is_six_l3647_364711


namespace transistors_in_2010_l3647_364797

/-- Moore's law tripling factor -/
def tripling_factor : ℕ := 3

/-- Years between tripling events -/
def years_per_tripling : ℕ := 3

/-- Initial number of transistors in 1995 -/
def initial_transistors : ℕ := 500000

/-- Years between 1995 and 2010 -/
def years_elapsed : ℕ := 15

/-- Number of tripling events in the given time period -/
def num_triplings : ℕ := years_elapsed / years_per_tripling

/-- Calculates the number of transistors after a given number of tripling events -/
def transistors_after_triplings (initial : ℕ) (triplings : ℕ) : ℕ :=
  initial * tripling_factor ^ triplings

/-- Theorem: The number of transistors in 2010 is 121,500,000 -/
theorem transistors_in_2010 :
  transistors_after_triplings initial_transistors num_triplings = 121500000 := by
  sorry

end transistors_in_2010_l3647_364797


namespace quadratic_max_value_l3647_364746

/-- The quadratic function f(x) = -x^2 + 2x has a maximum value of 1. -/
theorem quadratic_max_value (x : ℝ) : 
  (∀ y : ℝ, -y^2 + 2*y ≤ 1) ∧ (∃ z : ℝ, -z^2 + 2*z = 1) := by
  sorry

end quadratic_max_value_l3647_364746


namespace remaining_children_fed_theorem_l3647_364773

/-- Represents the capacity of a meal in terms of adults and children -/
structure MealCapacity where
  adults : ℕ
  children : ℕ

/-- Calculates the number of children that can be fed with the remaining food -/
def remainingChildrenFed (capacity : MealCapacity) (adultsEaten : ℕ) : ℕ :=
  let remainingAdults := capacity.adults - adultsEaten
  (remainingAdults * capacity.children) / capacity.adults

/-- Theorem stating that given a meal for 70 adults or 90 children, 
    if 42 adults have eaten, the remaining food can feed 36 children -/
theorem remaining_children_fed_theorem (capacity : MealCapacity) 
  (h1 : capacity.adults = 70)
  (h2 : capacity.children = 90)
  (h3 : adultsEaten = 42) :
  remainingChildrenFed capacity adultsEaten = 36 := by
  sorry

#eval remainingChildrenFed { adults := 70, children := 90 } 42

end remaining_children_fed_theorem_l3647_364773


namespace solution_set_part_i_range_of_a_part_ii_l3647_364706

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x + |2*x - 3| > 0} = {x : ℝ | x < 2/3 ∨ x > 2} :=
sorry

-- Part II
theorem range_of_a_part_ii :
  {a : ℝ | ∀ x, f a x < |x - 3|} = {a : ℝ | 1 < a ∧ a < 5} :=
sorry

end solution_set_part_i_range_of_a_part_ii_l3647_364706


namespace composition_difference_constant_l3647_364722

/-- Given two functions f and g, prove that their composition difference is constant. -/
theorem composition_difference_constant (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 4 * x - 5) 
  (hg : ∀ x, g x = x / 4 + 1) : 
  ∀ x, f (g x) - g (f x) = 1/4 := by
sorry

end composition_difference_constant_l3647_364722


namespace building_block_length_l3647_364710

theorem building_block_length 
  (box_height box_width box_length : ℝ)
  (block_height block_width : ℝ)
  (num_blocks : ℕ) :
  box_height = 8 →
  box_width = 10 →
  box_length = 12 →
  block_height = 3 →
  block_width = 2 →
  num_blocks = 40 →
  ∃ (block_length : ℝ),
    box_height * box_width * box_length = 
    num_blocks * block_height * block_width * block_length ∧
    block_length = 4 :=
by sorry

end building_block_length_l3647_364710


namespace problem_statement_l3647_364790

theorem problem_statement (a b : ℤ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 5 * b = 10 := by
  sorry

end problem_statement_l3647_364790


namespace pq_length_l3647_364768

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  -- PQR is a right-angled triangle
  (Q.1 - P.1) * (R.2 - P.2) = (R.1 - P.1) * (Q.2 - P.2) ∧
  -- Angle PQR is 45°
  (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) * Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) / Real.sqrt 2 ∧
  -- PR = 10
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 100

-- Theorem statement
theorem pq_length (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 50 := by
  sorry

end pq_length_l3647_364768


namespace empty_quadratic_set_l3647_364757

theorem empty_quadratic_set (a : ℝ) :
  ({x : ℝ | a * x^2 - 2 * a * x + 1 < 0} = ∅) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

end empty_quadratic_set_l3647_364757


namespace sum_of_specific_arithmetic_progression_l3647_364796

/-- Sum of an arithmetic progression -/
def sum_arithmetic_progression (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Theorem: The sum of the first 20 terms of an arithmetic progression
    with first term 30 and common difference -3 is equal to 30 -/
theorem sum_of_specific_arithmetic_progression :
  sum_arithmetic_progression 30 (-3) 20 = 30 := by
sorry

end sum_of_specific_arithmetic_progression_l3647_364796


namespace sally_pens_taken_home_l3647_364748

def total_pens : ℕ := 5230
def num_students : ℕ := 89
def pens_per_student : ℕ := 58

def pens_distributed : ℕ := num_students * pens_per_student
def pens_remaining : ℕ := total_pens - pens_distributed
def pens_in_locker : ℕ := pens_remaining / 2
def pens_taken_home : ℕ := pens_remaining - pens_in_locker

theorem sally_pens_taken_home : pens_taken_home = 34 := by
  sorry

end sally_pens_taken_home_l3647_364748


namespace train_passing_platform_l3647_364721

/-- Given a train of length 250 meters passing a pole in 10 seconds,
    prove that it takes 60 seconds to pass a platform of length 1250 meters. -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (pole_passing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 250)
  (h2 : pole_passing_time = 10)
  (h3 : platform_length = 1250) :
  (train_length + platform_length) / (train_length / pole_passing_time) = 60 := by
  sorry

#check train_passing_platform

end train_passing_platform_l3647_364721


namespace ice_cream_arrangement_count_l3647_364791

theorem ice_cream_arrangement_count : Nat.factorial 5 = 120 := by
  sorry

end ice_cream_arrangement_count_l3647_364791


namespace polar_coordinates_of_point_l3647_364714

theorem polar_coordinates_of_point (x y : ℝ) (h : (x, y) = (-2, -2 * Real.sqrt 3)) :
  ∃ (ρ θ : ℝ), ρ = 4 ∧ θ = (4 * π) / 3 ∧
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ := by
  sorry

end polar_coordinates_of_point_l3647_364714


namespace champagne_discount_percentage_l3647_364736

-- Define the problem parameters
def hot_tub_capacity : ℝ := 40
def bottle_capacity : ℝ := 1
def quarts_per_gallon : ℝ := 4
def original_price_per_bottle : ℝ := 50
def total_spent_after_discount : ℝ := 6400

-- Define the theorem
theorem champagne_discount_percentage :
  let total_quarts : ℝ := hot_tub_capacity * quarts_per_gallon
  let total_bottles : ℝ := total_quarts / bottle_capacity
  let full_price : ℝ := total_bottles * original_price_per_bottle
  let discount_amount : ℝ := full_price - total_spent_after_discount
  let discount_percentage : ℝ := (discount_amount / full_price) * 100
  discount_percentage = 20 := by
  sorry


end champagne_discount_percentage_l3647_364736


namespace largest_certain_divisor_l3647_364712

/-- The set of numbers on the eight-sided die -/
def dieNumbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

/-- The product of seven numbers from the die -/
def Q (s : Finset ℕ) : ℕ :=
  if s.card = 7 ∧ s ⊆ dieNumbers then s.prod id else 0

/-- The theorem stating that 48 is the largest number certain to divide Q -/
theorem largest_certain_divisor :
  ∀ n : ℕ, (∀ s : Finset ℕ, s.card = 7 ∧ s ⊆ dieNumbers → n ∣ Q s) → n ≤ 48 :=
sorry

end largest_certain_divisor_l3647_364712


namespace kens_climbing_pace_l3647_364782

/-- Climbing problem -/
theorem kens_climbing_pace 
  (sari_head_start : ℝ)  -- Time Sari starts before Ken (in hours)
  (sari_initial_lead : ℝ)  -- Sari's lead when Ken starts (in meters)
  (ken_climbing_time : ℝ)  -- Time Ken spends climbing (in hours)
  (final_distance : ℝ)  -- Distance Ken is ahead of Sari at the summit (in meters)
  (h1 : sari_head_start = 2)
  (h2 : sari_initial_lead = 700)
  (h3 : ken_climbing_time = 5)
  (h4 : final_distance = 50) :
  (sari_initial_lead + final_distance) / ken_climbing_time + 
  (sari_initial_lead / sari_head_start) = 500 :=
sorry

end kens_climbing_pace_l3647_364782


namespace largest_power_of_two_dividing_difference_l3647_364761

theorem largest_power_of_two_dividing_difference : ∃ (k : ℕ), k = 13 ∧ 
  (∀ (n : ℕ), 2^n ∣ (10^10 - 2^10) → n ≤ k) ∧ 
  (2^k ∣ (10^10 - 2^10)) := by
  sorry

end largest_power_of_two_dividing_difference_l3647_364761


namespace necklace_profit_calculation_l3647_364724

/-- Calculates the profit for a single necklace. -/
def profit_per_necklace (charm_count : ℕ) (charm_cost : ℕ) (selling_price : ℕ) : ℕ :=
  selling_price - charm_count * charm_cost

/-- Calculates the total profit for a specific necklace type. -/
def total_profit_for_type (necklace_count : ℕ) (charm_count : ℕ) (charm_cost : ℕ) (selling_price : ℕ) : ℕ :=
  necklace_count * profit_per_necklace charm_count charm_cost selling_price

theorem necklace_profit_calculation :
  let type_a_profit := total_profit_for_type 45 8 10 125
  let type_b_profit := total_profit_for_type 35 12 18 280
  let type_c_profit := total_profit_for_type 25 15 12 350
  type_a_profit + type_b_profit + type_c_profit = 8515 := by
  sorry

end necklace_profit_calculation_l3647_364724


namespace division_multiplication_problem_l3647_364728

theorem division_multiplication_problem : 377 / 13 / 29 * (1 / 4) / 2 = 0.125 := by
  sorry

end division_multiplication_problem_l3647_364728


namespace table_length_is_77_l3647_364783

/-- Represents the dimensions of a rectangular table. -/
structure TableDimensions where
  length : ℕ
  width : ℕ

/-- Represents the dimensions of a paper sheet. -/
structure SheetDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the length of a table given its width and the dimensions of the paper sheets used to cover it. -/
def calculateTableLength (tableWidth : ℕ) (sheet : SheetDimensions) : ℕ :=
  sheet.length + (tableWidth - sheet.width)

/-- Theorem stating that for a table of width 80 cm covered with 5x8 cm sheets,
    where each sheet is placed 1 cm higher and 1 cm to the right of the previous one,
    the length of the table is 77 cm. -/
theorem table_length_is_77 :
  let tableWidth : ℕ := 80
  let sheet : SheetDimensions := ⟨5, 8⟩
  calculateTableLength tableWidth sheet = 77 := by
  sorry

#check table_length_is_77

end table_length_is_77_l3647_364783


namespace smallest_n_divisible_by_2022_l3647_364733

theorem smallest_n_divisible_by_2022 :
  ∃ (n : ℕ), n > 1 ∧ n^7 - 1 % 2022 = 0 ∧
  ∀ (m : ℕ), m > 1 ∧ m < n → m^7 - 1 % 2022 ≠ 0 :=
by sorry

end smallest_n_divisible_by_2022_l3647_364733


namespace power_sum_integer_l3647_364781

theorem power_sum_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m :=
sorry

end power_sum_integer_l3647_364781


namespace problem_solution_l3647_364756

def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1) 1, x^2 - x - m < 0}

def A (a : ℝ) : Set ℝ := {x | (x - 3*a) * (x - a - 2) < 0}

theorem problem_solution :
  (B = Set.Ioi 2) ∧
  ({a : ℝ | A a ⊆ B ∧ A a ≠ B} = Set.Ici (2/3)) := by sorry

end problem_solution_l3647_364756


namespace angle_B_when_A_is_pi_sixth_sin_A_plus_sin_C_range_l3647_364720

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a = t.b * Real.tan t.A ∧
  t.B > Real.pi / 2

-- Theorem 1
theorem angle_B_when_A_is_pi_sixth (t : Triangle) 
  (h : is_valid_triangle t) (h_A : t.A = Real.pi / 6) : 
  t.B = 2 * Real.pi / 3 := 
sorry

-- Theorem 2
theorem sin_A_plus_sin_C_range (t : Triangle) 
  (h : is_valid_triangle t) : 
  Real.sqrt 2 / 2 < Real.sin t.A + Real.sin t.C ∧ 
  Real.sin t.A + Real.sin t.C ≤ 9 / 8 := 
sorry

end angle_B_when_A_is_pi_sixth_sin_A_plus_sin_C_range_l3647_364720
