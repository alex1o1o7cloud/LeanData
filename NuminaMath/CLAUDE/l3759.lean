import Mathlib

namespace number_problem_l3759_375963

theorem number_problem (x : ℝ) (n : ℝ) (h1 : x > 0) (h2 : x / 50 + x / n = 0.06 * x) : n = 25 := by
  sorry

end number_problem_l3759_375963


namespace chess_tournament_games_l3759_375940

/-- The number of unique games in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group with 7 players, where each player plays every other player once,
    the total number of games played is 21. -/
theorem chess_tournament_games :
  num_games 7 = 21 := by sorry

end chess_tournament_games_l3759_375940


namespace intersection_empty_implies_k_leq_one_l3759_375973

-- Define the sets P and Q
def P (k : ℝ) : Set ℝ := {y | y = k}
def Q (a : ℝ) : Set ℝ := {y | ∃ x : ℝ, y = a^x + 1}

-- State the theorem
theorem intersection_empty_implies_k_leq_one
  (k : ℝ) (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  (P k ∩ Q a = ∅) → k ≤ 1 := by
  sorry

end intersection_empty_implies_k_leq_one_l3759_375973


namespace min_distinct_values_l3759_375927

theorem min_distinct_values (n : ℕ) (mode_count : ℕ) (total_count : ℕ) :
  n = 2017 →
  mode_count = 11 →
  total_count = n →
  (∃ (distinct_values : ℕ), 
    distinct_values ≥ 202 ∧
    ∀ (m : ℕ), m < 202 → 
      ¬(∃ (list : List ℕ),
        list.length = total_count ∧
        (∃ (mode : ℕ), list.count mode = mode_count ∧
          ∀ (x : ℕ), x ≠ mode → list.count x < mode_count) ∧
        list.toFinset.card = m)) :=
sorry

end min_distinct_values_l3759_375927


namespace expression_simplification_l3759_375976

/-- Given nonzero real numbers a, b, c, and a constant real number θ,
    define x, y, z as specified, and prove that x^2 + y^2 + z^2 - xyz = 4 -/
theorem expression_simplification 
  (a b c : ℝ) (θ : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (x : ℝ := b / c + c / b + Real.sin θ)
  (y : ℝ := a / c + c / a + Real.cos θ)
  (z : ℝ := a / b + b / a + Real.tan θ) :
  x^2 + y^2 + z^2 - x*y*z = 4 := by
  sorry

end expression_simplification_l3759_375976


namespace delivery_time_problem_l3759_375936

/-- Calculates the time needed to deliver all cars -/
def delivery_time (coal_cars iron_cars wood_cars : ℕ) 
                  (coal_deposit iron_deposit wood_deposit : ℕ) 
                  (time_between_stations : ℕ) : ℕ :=
  let coal_stations := (coal_cars + coal_deposit - 1) / coal_deposit
  let iron_stations := (iron_cars + iron_deposit - 1) / iron_deposit
  let wood_stations := (wood_cars + wood_deposit - 1) / wood_deposit
  let max_stations := max coal_stations (max iron_stations wood_stations)
  max_stations * time_between_stations

/-- Proves that the delivery time for the given problem is 100 minutes -/
theorem delivery_time_problem : 
  delivery_time 6 12 2 2 3 1 25 = 100 := by
  sorry

end delivery_time_problem_l3759_375936


namespace breath_holding_difference_l3759_375929

/-- 
Given that:
- Kelly held her breath for 3 minutes
- Brittany held her breath for 20 seconds less than Kelly
- Buffy held her breath for 120 seconds

Prove that Buffy held her breath for 40 seconds less than Brittany
-/
theorem breath_holding_difference : 
  let kelly_time := 3 * 60 -- Kelly's time in seconds
  let brittany_time := kelly_time - 20 -- Brittany's time in seconds
  let buffy_time := 120 -- Buffy's time in seconds
  brittany_time - buffy_time = 40 := by
  sorry

end breath_holding_difference_l3759_375929


namespace negation_equivalence_l3759_375985

theorem negation_equivalence :
  (¬ ∃ x : ℝ, 2 * x - 3 > 1) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) := by
  sorry

end negation_equivalence_l3759_375985


namespace inequality_solution_set_l3759_375957

theorem inequality_solution_set : 
  {x : ℝ | x + 7 > -2*x + 1} = {x : ℝ | x > -2} := by sorry

end inequality_solution_set_l3759_375957


namespace boys_in_class_l3759_375969

theorem boys_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 56) (h2 : ratio_girls = 4) (h3 : ratio_boys = 3) : 
  (total * ratio_boys) / (ratio_girls + ratio_boys) = 24 := by
sorry

end boys_in_class_l3759_375969


namespace mans_swimming_speed_l3759_375998

/-- Proves that a man's swimming speed in still water is 1.5 km/h given the conditions -/
theorem mans_swimming_speed 
  (stream_speed : ℝ) 
  (upstream_time downstream_time : ℝ) 
  (h1 : stream_speed = 0.5)
  (h2 : upstream_time = 2 * downstream_time) : 
  ∃ (still_water_speed : ℝ), still_water_speed = 1.5 :=
by
  sorry

#check mans_swimming_speed

end mans_swimming_speed_l3759_375998


namespace power_mod_eleven_l3759_375905

theorem power_mod_eleven : 5^2023 % 11 = 4 := by
  sorry

end power_mod_eleven_l3759_375905


namespace union_of_A_and_B_l3759_375925

def A : Set ℝ := {x | x * (x + 1) ≤ 0}
def B : Set ℝ := {x | -1 < x ∧ x < 1}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end union_of_A_and_B_l3759_375925


namespace entrance_exam_questions_entrance_exam_questions_is_70_l3759_375904

/-- Proves that the total number of questions in an entrance exam is 70,
    given the specified scoring system and student performance. -/
theorem entrance_exam_questions : ℕ :=
  let correct_marks : ℕ := 3
  let wrong_marks : ℤ := -1
  let total_score : ℤ := 38
  let correct_answers : ℕ := 27
  let total_questions : ℕ := 70
  
  have h1 : (correct_answers : ℤ) * correct_marks + 
            (total_questions - correct_answers : ℤ) * wrong_marks = total_score := by sorry
  
  total_questions

/-- The proof that the number of questions in the entrance exam is 70. -/
theorem entrance_exam_questions_is_70 : entrance_exam_questions = 70 := by sorry

end entrance_exam_questions_entrance_exam_questions_is_70_l3759_375904


namespace y0_minus_one_is_perfect_square_l3759_375967

theorem y0_minus_one_is_perfect_square 
  (x y : ℕ → ℕ) 
  (h : ∀ n, (x n : ℝ) + Real.sqrt 2 * (y n) = Real.sqrt 2 * (3 + 2 * Real.sqrt 2) ^ (2 ^ n)) : 
  ∃ k : ℕ, y 0 - 1 = k ^ 2 := by
sorry

end y0_minus_one_is_perfect_square_l3759_375967


namespace correct_multiplier_problem_solution_l3759_375935

theorem correct_multiplier (number_to_multiply : ℕ) (mistaken_multiplier : ℕ) (difference : ℕ) : ℕ :=
  let correct_multiplier := (mistaken_multiplier * number_to_multiply + difference) / number_to_multiply
  correct_multiplier

theorem problem_solution :
  correct_multiplier 135 34 1215 = 43 := by
  sorry

end correct_multiplier_problem_solution_l3759_375935


namespace largest_derivative_at_one_l3759_375943

/-- The derivative of 2x+1 at x=1 is greater than the derivatives of -x², 1/x, and √x at x=1 -/
theorem largest_derivative_at_one :
  let f₁ (x : ℝ) := -x^2
  let f₂ (x : ℝ) := 1/x
  let f₃ (x : ℝ) := 2*x + 1
  let f₄ (x : ℝ) := Real.sqrt x
  (deriv f₃ 1 > deriv f₁ 1) ∧ 
  (deriv f₃ 1 > deriv f₂ 1) ∧ 
  (deriv f₃ 1 > deriv f₄ 1) :=
by sorry

end largest_derivative_at_one_l3759_375943


namespace equation_solutions_l3759_375953

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁ + 3)^2 = 16 ∧ (2 * x₂ + 3)^2 = 16 ∧ x₁ = 1/2 ∧ x₂ = -7/2) ∧
  (∃ y₁ y₂ : ℝ, y₁^2 - 4*y₁ - 3 = 0 ∧ y₂^2 - 4*y₂ - 3 = 0 ∧ y₁ = 2 + Real.sqrt 7 ∧ y₂ = 2 - Real.sqrt 7) :=
by sorry

end equation_solutions_l3759_375953


namespace least_number_divisible_by_all_l3759_375958

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 6) % 24 = 0 ∧ (n + 6) % 32 = 0 ∧ (n + 6) % 36 = 0 ∧ (n + 6) % 54 = 0

theorem least_number_divisible_by_all : 
  is_divisible_by_all 858 ∧ ∀ m : ℕ, m < 858 → ¬is_divisible_by_all m :=
by sorry

end least_number_divisible_by_all_l3759_375958


namespace no_obtuse_angles_l3759_375956

-- Define an isosceles triangle with two 70-degree angles
structure IsoscelesTriangle70 where
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  is_isosceles : angle_a = angle_b
  angles_70 : angle_a = 70 ∧ angle_b = 70
  sum_180 : angle_a + angle_b + angle_c = 180

-- Define what an obtuse angle is
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Theorem statement
theorem no_obtuse_angles (t : IsoscelesTriangle70) :
  ¬ (is_obtuse t.angle_a ∨ is_obtuse t.angle_b ∨ is_obtuse t.angle_c) :=
by sorry

end no_obtuse_angles_l3759_375956


namespace cubic_polynomial_sum_zero_l3759_375964

-- Define a cubic polynomial
def cubic_polynomial (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem cubic_polynomial_sum_zero 
  (a b c d : ℝ) 
  (h1 : cubic_polynomial a b c d 0 = 2 * d)
  (h2 : cubic_polynomial a b c d 1 = 3 * d)
  (h3 : cubic_polynomial a b c d (-1) = 5 * d) :
  cubic_polynomial a b c d 3 + cubic_polynomial a b c d (-3) = 0 := by
sorry


end cubic_polynomial_sum_zero_l3759_375964


namespace wedding_chairs_l3759_375983

theorem wedding_chairs (rows : ℕ) (chairs_per_row : ℕ) (extra_chairs : ℕ) : 
  rows = 7 → chairs_per_row = 12 → extra_chairs = 11 → 
  rows * chairs_per_row + extra_chairs = 95 := by
sorry

end wedding_chairs_l3759_375983


namespace employee_reduction_percentage_l3759_375991

def original_employees : ℝ := 227
def reduced_employees : ℝ := 195

theorem employee_reduction_percentage : 
  let difference := original_employees - reduced_employees
  let percentage := (difference / original_employees) * 100
  abs (percentage - 14.1) < 0.1 := by
  sorry

end employee_reduction_percentage_l3759_375991


namespace intersection_point_sum_l3759_375975

/-- Two lines in a plane -/
structure TwoLines where
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ

/-- Points P, Q, and T for the given lines -/
structure LinePoints (l : TwoLines) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  T : ℝ × ℝ
  h_P : l.line1 P.1 = P.2 ∧ P.2 = 0
  h_Q : l.line1 Q.1 = Q.2 ∧ Q.1 = 0
  h_T : l.line1 T.1 = T.2 ∧ l.line2 T.1 = T.2

/-- The theorem statement -/
theorem intersection_point_sum (l : TwoLines) (pts : LinePoints l) 
  (h_line1 : ∀ x, l.line1 x = -2/3 * x + 8)
  (h_line2 : ∀ x, l.line2 x = 3/2 * x - 9)
  (h_area : (pts.P.1 * pts.Q.2) / 2 = 2 * ((pts.P.1 - pts.T.1) * pts.T.2) / 2) :
  pts.T.1 + pts.T.2 = 138/13 := by sorry

end intersection_point_sum_l3759_375975


namespace cube_root_of_eight_l3759_375951

theorem cube_root_of_eight : ∃ x : ℝ, x^3 = 8 ∧ x = 2 := by sorry

end cube_root_of_eight_l3759_375951


namespace smallest_w_proof_l3759_375903

/-- The product of 1452 and the smallest positive integer w that results in a number 
    with 3^3 and 13^3 as factors -/
def smallest_w : ℕ := 19773

theorem smallest_w_proof :
  ∀ w : ℕ, w > 0 →
  (∃ k : ℕ, 1452 * w = k * 3^3 * 13^3) →
  w ≥ smallest_w :=
by sorry

end smallest_w_proof_l3759_375903


namespace linear_system_solution_l3759_375918

/-- Solution to a system of linear equations -/
theorem linear_system_solution (a b c h : ℝ) :
  let x := (h - b) * (h - c) / ((a - b) * (a - c))
  let y := (h - a) * (h - c) / ((b - a) * (b - c))
  let z := (h - a) * (h - b) / ((c - a) * (c - b))
  x + y + z = 1 ∧
  a * x + b * y + c * z = h ∧
  a^2 * x + b^2 * y + c^2 * z = h^2 := by
  sorry

end linear_system_solution_l3759_375918


namespace bag_contains_sixty_balls_l3759_375913

/-- The number of white balls in the bag -/
def white_balls : ℕ := 22

/-- The number of green balls in the bag -/
def green_balls : ℕ := 10

/-- The number of yellow balls in the bag -/
def yellow_balls : ℕ := 7

/-- The number of red balls in the bag -/
def red_balls : ℕ := 15

/-- The number of purple balls in the bag -/
def purple_balls : ℕ := 6

/-- The probability of choosing a ball that is neither red nor purple -/
def prob_not_red_or_purple : ℚ := 65/100

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + green_balls + yellow_balls + red_balls + purple_balls

theorem bag_contains_sixty_balls : total_balls = 60 := by
  sorry

end bag_contains_sixty_balls_l3759_375913


namespace determinant_zero_implies_ratio_four_l3759_375910

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_zero_implies_ratio_four (θ : ℝ) : 
  determinant (Real.sin θ) 2 (Real.cos θ) 3 = 0 → 
  (3 * Real.sin θ + 2 * Real.cos θ) / (3 * Real.sin θ - Real.cos θ) = 4 :=
by sorry

end determinant_zero_implies_ratio_four_l3759_375910


namespace cell_phone_plan_comparison_l3759_375902

/-- Cellular phone plan comparison -/
theorem cell_phone_plan_comparison (F : ℝ) : 
  (∀ (minutes : ℝ), 
    F + max (minutes - 500) 0 * 0.35 = 
    75 + max (minutes - 1000) 0 * 0.45 → minutes = 2500) →
  F = 50 := by
sorry

end cell_phone_plan_comparison_l3759_375902


namespace max_score_2079_score_2079_eq_30_unique_max_score_l3759_375944

def score (x : ℕ) : ℕ :=
  (if x % 3 = 0 then 3 else 0) +
  (if x % 5 = 0 then 5 else 0) +
  (if x % 7 = 0 then 7 else 0) +
  (if x % 9 = 0 then 9 else 0) +
  (if x % 11 = 0 then 11 else 0)

theorem max_score_2079 :
  ∀ x : ℕ, 2017 ≤ x → x ≤ 2117 → score x ≤ score 2079 :=
by sorry

theorem score_2079_eq_30 : score 2079 = 30 :=
by sorry

theorem unique_max_score :
  ∀ x : ℕ, 2017 ≤ x → x ≤ 2117 → x ≠ 2079 → score x < score 2079 :=
by sorry

end max_score_2079_score_2079_eq_30_unique_max_score_l3759_375944


namespace different_signs_larger_negative_l3759_375950

theorem different_signs_larger_negative (a b : ℝ) : 
  a + b < 0 → a * b < 0 → 
  ((a < 0 ∧ b > 0 ∧ abs a > abs b) ∨ (a > 0 ∧ b < 0 ∧ abs b > abs a)) := by
  sorry

end different_signs_larger_negative_l3759_375950


namespace condition_analysis_l3759_375945

theorem condition_analysis (x : ℝ) :
  (∀ x, -1 < x ∧ x < 3 → x^2 - 2*x < 8) ∧
  (∃ x, x^2 - 2*x < 8 ∧ ¬(-1 < x ∧ x < 3)) :=
by sorry

end condition_analysis_l3759_375945


namespace baseball_card_value_decrease_l3759_375942

theorem baseball_card_value_decrease (x : ℝ) : 
  (1 - x / 100) * (1 - 30 / 100) = 1 - 44.00000000000001 / 100 → x = 20 := by
  sorry

end baseball_card_value_decrease_l3759_375942


namespace probability_on_2x_is_one_twelfth_l3759_375914

/-- A die is a finite set of numbers from 1 to 6 -/
def Die : Finset ℕ := Finset.range 6

/-- The probability space of rolling a die twice -/
def DieRollSpace : Finset (ℕ × ℕ) := Die.product Die

/-- The event where (x, y) falls on y = 2x -/
def EventOn2x : Finset (ℕ × ℕ) := DieRollSpace.filter (fun (x, y) => y = 2 * x)

/-- The probability of the event -/
def ProbabilityOn2x : ℚ := (EventOn2x.card : ℚ) / (DieRollSpace.card : ℚ)

theorem probability_on_2x_is_one_twelfth : ProbabilityOn2x = 1 / 12 := by
  sorry

end probability_on_2x_is_one_twelfth_l3759_375914


namespace loan_amount_calculation_l3759_375960

/-- Proves that the amount lent is 1000 given the specified conditions -/
theorem loan_amount_calculation (P : ℝ) : 
  (P * 0.115 * 3 - P * 0.10 * 3 = 45) → P = 1000 := by
  sorry

end loan_amount_calculation_l3759_375960


namespace M_intersect_N_equals_open_zero_one_l3759_375948

def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem M_intersect_N_equals_open_zero_one : M ∩ N = Set.Ioo 0 1 := by sorry

end M_intersect_N_equals_open_zero_one_l3759_375948


namespace max_value_fraction_l3759_375900

theorem max_value_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * (a + b + c) = b * c) :
  a / (b + c) ≤ (Real.sqrt 2 - 1) / 2 ∧
  ∃ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * (a + b + c) = b * c ∧ a / (b + c) = (Real.sqrt 2 - 1) / 2 :=
by sorry

end max_value_fraction_l3759_375900


namespace kgood_existence_l3759_375941

def IsKGood (k : ℕ) (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

theorem kgood_existence (k : ℕ) :
  (k ≥ 2 → ∃ f : ℕ+ → ℕ+, IsKGood k f) ∧
  (k = 1 → ¬∃ f : ℕ+ → ℕ+, IsKGood k f) :=
sorry

end kgood_existence_l3759_375941


namespace probability_at_least_one_woman_l3759_375977

def num_men : ℕ := 8
def num_women : ℕ := 4
def num_selected : ℕ := 4

theorem probability_at_least_one_woman :
  let total_people := num_men + num_women
  let prob_all_men := (num_men.choose num_selected : ℚ) / (total_people.choose num_selected : ℚ)
  (1 : ℚ) - prob_all_men = 85 / 99 := by sorry

end probability_at_least_one_woman_l3759_375977


namespace diamond_three_four_l3759_375911

/-- The diamond operation defined for real numbers -/
def diamond (a b : ℝ) : ℝ := 4*a + 5*b - a^2*b

/-- Theorem stating that 3 ⋄ 4 = -4 -/
theorem diamond_three_four : diamond 3 4 = -4 := by
  sorry

end diamond_three_four_l3759_375911


namespace hawks_score_l3759_375981

/-- Calculates the total score for a team given the number of touchdowns and points per touchdown -/
def totalScore (touchdowns : ℕ) (pointsPerTouchdown : ℕ) : ℕ :=
  touchdowns * pointsPerTouchdown

/-- Theorem: If a team scores 3 touchdowns, and each touchdown is worth 7 points, then the team's total score is 21 points -/
theorem hawks_score :
  totalScore 3 7 = 21 := by
  sorry

end hawks_score_l3759_375981


namespace sqrt_equation_solution_l3759_375919

theorem sqrt_equation_solution (x : ℝ) : (5 - 1/x)^(1/4) = -3 → x = -1/76 := by
  sorry

end sqrt_equation_solution_l3759_375919


namespace parabola_line_intersection_l3759_375920

/-- Given a parabola y = ax² - a (a ≠ 0) intersecting a line y = kx at points
    with sum of x-coordinates less than 0, prove that the line y = ax + k
    passes through the first and fourth quadrants. -/
theorem parabola_line_intersection (a k : ℝ) (ha : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, a * x₁^2 - a = k * x₁ ∧
               a * x₂^2 - a = k * x₂ ∧
               x₁ + x₂ < 0) →
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = a * x + k) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ y = a * x + k) :=
by sorry

end parabola_line_intersection_l3759_375920


namespace inscribed_squares_max_distance_l3759_375955

def inner_square_perimeter : ℝ := 20
def outer_square_perimeter : ℝ := 28

theorem inscribed_squares_max_distance :
  let inner_side := inner_square_perimeter / 4
  let outer_side := outer_square_perimeter / 4
  ∃ (x y : ℝ),
    x + y = outer_side ∧
    x^2 + y^2 = inner_side^2 ∧
    Real.sqrt (x^2 + (x + y)^2) = Real.sqrt 65 :=
by sorry

end inscribed_squares_max_distance_l3759_375955


namespace six_coin_flip_probability_six_coin_flip_probability_is_one_thirtysecond_l3759_375966

theorem six_coin_flip_probability : ℝ :=
  let n : ℕ := 6  -- number of coins
  let p : ℝ := 1 / 2  -- probability of heads for a fair coin
  let total_outcomes : ℕ := 2^n
  let favorable_outcomes : ℕ := 2  -- all heads or all tails
  favorable_outcomes / total_outcomes

theorem six_coin_flip_probability_is_one_thirtysecond : 
  six_coin_flip_probability = 1 / 32 := by
  sorry

end six_coin_flip_probability_six_coin_flip_probability_is_one_thirtysecond_l3759_375966


namespace mans_age_twice_students_l3759_375959

/-- Proves that it takes 2 years for a man's age to be twice his student's age -/
theorem mans_age_twice_students (student_age : ℕ) (age_difference : ℕ) : 
  student_age = 24 → age_difference = 26 → 
  ∃ (years : ℕ), (student_age + years) * 2 = (student_age + age_difference + years) ∧ years = 2 :=
by sorry

end mans_age_twice_students_l3759_375959


namespace chef_potato_problem_chef_leftover_potatoes_l3759_375970

/-- A chef's potato problem -/
theorem chef_potato_problem (total_potatoes : ℕ) 
  (fries_needed : ℕ) (fries_per_potato : ℕ) 
  (cubes_needed : ℕ) (cubes_per_potato : ℕ) : ℕ :=
  let potatoes_for_fries := (fries_needed + fries_per_potato - 1) / fries_per_potato
  let potatoes_for_cubes := (cubes_needed + cubes_per_potato - 1) / cubes_per_potato
  let potatoes_used := potatoes_for_fries + potatoes_for_cubes
  total_potatoes - potatoes_used

/-- The chef will have 17 potatoes leftover -/
theorem chef_leftover_potatoes : 
  chef_potato_problem 30 200 25 50 10 = 17 := by
  sorry

end chef_potato_problem_chef_leftover_potatoes_l3759_375970


namespace polly_lunch_time_l3759_375928

/-- Represents the cooking time for a week -/
structure CookingTime where
  breakfast : ℕ  -- Time spent on breakfast daily
  dinner_short : ℕ  -- Time spent on dinner for short days
  dinner_long : ℕ  -- Time spent on dinner for long days
  short_days : ℕ  -- Number of days with short dinner time
  total : ℕ  -- Total cooking time for the week

/-- Calculates the time spent on lunch given the cooking time for other meals -/
def lunch_time (c : CookingTime) : ℕ :=
  c.total - (7 * c.breakfast + c.short_days * c.dinner_short + (7 - c.short_days) * c.dinner_long)

/-- Theorem stating that Polly spends 35 minutes cooking lunch -/
theorem polly_lunch_time :
  ∃ (c : CookingTime),
    c.breakfast = 20 ∧
    c.dinner_short = 10 ∧
    c.dinner_long = 30 ∧
    c.short_days = 4 ∧
    c.total = 305 ∧
    lunch_time c = 35 := by
  sorry

end polly_lunch_time_l3759_375928


namespace grandview_soccer_league_members_l3759_375906

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := 2 * sock_cost

/-- The total cost for one member's gear for both home and away games -/
def member_cost : ℕ := 2 * (sock_cost + tshirt_cost + cap_cost)

/-- The total cost for all members' gear -/
def total_cost : ℕ := 4410

/-- The number of members in the Grandview Soccer League -/
def num_members : ℕ := 70

theorem grandview_soccer_league_members :
  num_members * member_cost = total_cost :=
sorry

end grandview_soccer_league_members_l3759_375906


namespace ways_to_go_home_via_library_l3759_375999

theorem ways_to_go_home_via_library (school_to_library : ℕ) (library_to_home : ℕ) : 
  school_to_library = 2 → library_to_home = 3 → school_to_library * library_to_home = 6 := by
  sorry

end ways_to_go_home_via_library_l3759_375999


namespace lcm_gcd_product_l3759_375982

theorem lcm_gcd_product (a b : ℕ) (h1 : a = 15) (h2 : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 135 := by
  sorry

end lcm_gcd_product_l3759_375982


namespace remainder_theorem_polynomial_remainder_l3759_375930

def f (x : ℝ) : ℝ := 8*x^5 - 10*x^4 + 3*x^3 + 5*x^2 - 7*x - 35

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := by sorry

theorem polynomial_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, f x = (x - 5) * q x + 19180 := by
  have h := remainder_theorem f 5
  sorry

end remainder_theorem_polynomial_remainder_l3759_375930


namespace max_sum_squares_triangle_sides_l3759_375968

theorem max_sum_squares_triangle_sides (a : ℝ) (α : ℝ) 
  (h_a_pos : a > 0) (h_α_acute : 0 < α ∧ α < π / 2) :
  ∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ 
    b^2 + c^2 = a^2 / (1 - Real.cos α) ∧
    ∀ (b' c' : ℝ), b' > 0 → c' > 0 → 
      b'^2 + a^2 = c'^2 + 2 * a * b' * Real.cos α →
      b'^2 + c'^2 ≤ a^2 / (1 - Real.cos α) := by
sorry


end max_sum_squares_triangle_sides_l3759_375968


namespace projection_matrix_values_l3759_375947

/-- A projection matrix P satisfies P² = P -/
def IsProjectionMatrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific matrix form given in the problem -/
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, 15/34],
    ![c, 25/34]]

theorem projection_matrix_values :
  ∀ a c : ℚ, IsProjectionMatrix (P a c) ↔ a = 9/34 ∧ c = 15/34 := by
  sorry

end projection_matrix_values_l3759_375947


namespace line_equation_l3759_375909

/-- A line L in R² -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (a * X + b * Y + c = 0)

/-- Point in R² -/
structure Point where
  x : ℝ
  y : ℝ

def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def intersects_at (l1 l2 : Line) (x : ℝ) : Prop :=
  ∃ y : ℝ, l1.a * x + l1.b * y + l1.c = 0 ∧ l2.a * x + l2.b * y + l2.c = 0

theorem line_equation (l : Line) (p : Point) (l2 l3 : Line) :
  passes_through l { x := 1, y := 5 } →
  perpendicular l { a := 2, b := -5, c := 3, eq := sorry } →
  intersects_at l { a := 3, b := 1, c := -1, eq := sorry } (-1) →
  l = { a := 5, b := 2, c := -15, eq := sorry } :=
by sorry

end line_equation_l3759_375909


namespace fraction_comparison_and_absolute_value_inequality_l3759_375931

theorem fraction_comparison_and_absolute_value_inequality :
  (-3 : ℚ) / 7 < (-2 : ℚ) / 5 ∧
  ∃ (a b : ℚ), |a + b| ≠ |a| + |b| :=
by sorry

end fraction_comparison_and_absolute_value_inequality_l3759_375931


namespace vacation_cost_l3759_375988

theorem vacation_cost (C : ℝ) :
  (C / 4 - C / 5 = 50) → C = 1000 := by
  sorry

end vacation_cost_l3759_375988


namespace sabina_college_loan_l3759_375962

theorem sabina_college_loan (college_cost savings grant_percentage : ℝ) : 
  college_cost = 30000 →
  savings = 10000 →
  grant_percentage = 0.4 →
  let remainder := college_cost - savings
  let grant_amount := grant_percentage * remainder
  let loan_amount := remainder - grant_amount
  loan_amount = 12000 := by
sorry

end sabina_college_loan_l3759_375962


namespace fish_eater_birds_count_l3759_375901

theorem fish_eater_birds_count (day1 day2 day3 : ℕ) : 
  day2 = 2 * day1 →
  day3 = day2 - 200 →
  day1 + day2 + day3 = 1300 →
  day1 = 300 := by
sorry

end fish_eater_birds_count_l3759_375901


namespace complex_equation_solution_l3759_375961

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 + Complex.I → z = 1 - 2 * Complex.I := by
  sorry

end complex_equation_solution_l3759_375961


namespace min_value_expression_l3759_375989

theorem min_value_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  (1 / m + 2 / n) ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 1 ∧ 1 / m₀ + 2 / n₀ = 8 :=
sorry

end min_value_expression_l3759_375989


namespace man_to_boy_work_ratio_l3759_375924

/-- The daily work done by a man -/
def M : ℝ := sorry

/-- The daily work done by a boy -/
def B : ℝ := sorry

/-- The total amount of work to be done -/
def total_work : ℝ := sorry

/-- The first condition: 12 men and 16 boys can do the work in 5 days -/
axiom condition1 : 5 * (12 * M + 16 * B) = total_work

/-- The second condition: 13 men and 24 boys can do the work in 4 days -/
axiom condition2 : 4 * (13 * M + 24 * B) = total_work

/-- The theorem stating that the ratio of daily work done by a man to that of a boy is 2:1 -/
theorem man_to_boy_work_ratio : M / B = 2 := by sorry

end man_to_boy_work_ratio_l3759_375924


namespace popcorn_buckets_needed_l3759_375917

/-- The number of popcorn buckets needed by a movie theater -/
theorem popcorn_buckets_needed (packages : ℕ) (buckets_per_package : ℕ) 
  (h1 : packages = 54)
  (h2 : buckets_per_package = 8) :
  packages * buckets_per_package = 432 := by
  sorry

end popcorn_buckets_needed_l3759_375917


namespace food_product_shelf_life_l3759_375916

/-- Represents the shelf life function of a food product -/
noncomputable def shelf_life (k b x : ℝ) : ℝ := Real.exp (k * x + b)

/-- Theorem stating the shelf life at 30°C and the maximum temperature for 80 hours shelf life -/
theorem food_product_shelf_life 
  (k b : ℝ) 
  (h1 : shelf_life k b 0 = 160) 
  (h2 : shelf_life k b 20 = 40) : 
  shelf_life k b 30 = 20 ∧ 
  ∀ x, shelf_life k b x ≥ 80 → x ≤ 10 := by
sorry


end food_product_shelf_life_l3759_375916


namespace probability_two_green_marbles_l3759_375986

def red_marbles : ℕ := 3
def green_marbles : ℕ := 4
def white_marbles : ℕ := 13

def total_marbles : ℕ := red_marbles + green_marbles + white_marbles

theorem probability_two_green_marbles :
  (green_marbles / total_marbles) * ((green_marbles - 1) / (total_marbles - 1)) = 3 / 95 := by
  sorry

end probability_two_green_marbles_l3759_375986


namespace commodity_price_increase_l3759_375922

/-- The annual price increase of commodity Y -/
def y : ℝ := sorry

/-- The year we're interested in -/
def target_year : ℝ := 1999.18

/-- The reference year -/
def reference_year : ℝ := 2001

/-- The price of commodity X in the reference year -/
def price_x_reference : ℝ := 5.20

/-- The price of commodity Y in the reference year -/
def price_y_reference : ℝ := 7.30

/-- The annual price increase of commodity X -/
def x_increase : ℝ := 0.45

/-- The price difference between X and Y in the target year -/
def price_difference : ℝ := 0.90

/-- The number of years between the target year and the reference year -/
def years_difference : ℝ := reference_year - target_year

theorem commodity_price_increase : 
  abs (y - 0.021) < 0.001 :=
by
  sorry

end commodity_price_increase_l3759_375922


namespace sufficient_not_necessary_condition_l3759_375933

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x = 0 → x^2 - 2*x = 0) ∧ ¬(x^2 - 2*x = 0 → x = 0) := by
  sorry

end sufficient_not_necessary_condition_l3759_375933


namespace negative_four_cubed_equality_l3759_375954

theorem negative_four_cubed_equality : (-4)^3 = -(4^3) := by sorry

end negative_four_cubed_equality_l3759_375954


namespace lawn_width_is_60_l3759_375934

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  roadWidth : ℝ
  costPerSqm : ℝ
  totalCost : ℝ

/-- Calculates the total area of the roads -/
def roadArea (l : LawnWithRoads) : ℝ :=
  l.roadWidth * l.length + l.roadWidth * l.width - l.roadWidth * l.roadWidth

/-- Theorem: Given the specifications, the width of the lawn is 60 meters -/
theorem lawn_width_is_60 (l : LawnWithRoads) 
    (h1 : l.length = 90)
    (h2 : l.roadWidth = 10)
    (h3 : l.costPerSqm = 3)
    (h4 : l.totalCost = 4200)
    (h5 : l.totalCost = l.costPerSqm * roadArea l) : 
  l.width = 60 := by
  sorry

end lawn_width_is_60_l3759_375934


namespace remainder_98_pow_50_mod_100_l3759_375937

theorem remainder_98_pow_50_mod_100 : 98^50 % 100 = 24 := by
  sorry

end remainder_98_pow_50_mod_100_l3759_375937


namespace complex_equation_sum_l3759_375995

theorem complex_equation_sum (a b : ℝ) :
  (Complex.I * (1 + a * Complex.I) = 1 + b * Complex.I) → a + b = 0 := by
  sorry

end complex_equation_sum_l3759_375995


namespace max_length_sum_l3759_375972

/-- Length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer -/
def length (n : ℕ) : ℕ := sorry

theorem max_length_sum :
  ∃ (x y : ℕ),
    x > 1 ∧
    y > 1 ∧
    x + 3 * y < 5000 ∧
    length x + length y = 20 ∧
    ∀ (a b : ℕ),
      a > 1 →
      b > 1 →
      a + 3 * b < 5000 →
      length a + length b ≤ 20 := by
  sorry

end max_length_sum_l3759_375972


namespace latest_departure_time_l3759_375921

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : Int :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

theorem latest_departure_time 
  (flight_time : Time)
  (check_in_time : Nat)
  (drive_time : Nat)
  (park_walk_time : Nat)
  (h1 : flight_time = ⟨20, 0⟩)  -- 8:00 pm
  (h2 : check_in_time = 120)    -- 2 hours
  (h3 : drive_time = 45)        -- 45 minutes
  (h4 : park_walk_time = 15)    -- 15 minutes
  : 
  let latest_departure := Time.mk 17 0  -- 5:00 pm
  timeDifferenceInMinutes flight_time latest_departure = 
    check_in_time + drive_time + park_walk_time :=
by sorry

end latest_departure_time_l3759_375921


namespace ellipse_k_value_l3759_375932

-- Define the ellipse equation
def ellipse_equation (k : ℝ) (x y : ℝ) : Prop :=
  4 * x^2 + k * y^2 = 4

-- Define the focus point
def focus : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem ellipse_k_value :
  ∃ (k : ℝ), 
    (∀ (x y : ℝ), ellipse_equation k x y → 
      ∃ (c : ℝ), c^2 = (4/k) - 1 ∧ c = 1) →
    k = 2 :=
sorry

end ellipse_k_value_l3759_375932


namespace calculation_proof_l3759_375907

theorem calculation_proof : (1/2)⁻¹ + Real.sqrt 12 - 4 * Real.sin (60 * π / 180) = 2 := by
  sorry

end calculation_proof_l3759_375907


namespace atmosphere_depth_for_specific_peak_l3759_375923

/-- Represents a cone-shaped peak on an alien planet -/
structure ConePeak where
  height : ℝ
  atmosphereVolumeFraction : ℝ

/-- Calculates the depth of the atmosphere at the base of a cone-shaped peak -/
def atmosphereDepth (peak : ConePeak) : ℝ :=
  peak.height * (1 - (peak.atmosphereVolumeFraction)^(1/3))

/-- Theorem stating the depth of the atmosphere for a specific cone-shaped peak -/
theorem atmosphere_depth_for_specific_peak :
  let peak : ConePeak := { height := 5000, atmosphereVolumeFraction := 4/5 }
  atmosphereDepth peak = 340 := by
  sorry

end atmosphere_depth_for_specific_peak_l3759_375923


namespace smallest_quotient_smallest_quotient_achievable_l3759_375952

def card_set : Set ℤ := {-5, -4, 0, 4, 6}

theorem smallest_quotient (a b : ℤ) (ha : a ∈ card_set) (hb : b ∈ card_set) (hab : a ≠ b) (hb_nonzero : b ≠ 0) :
  (a : ℚ) / b ≥ -3/2 :=
sorry

theorem smallest_quotient_achievable :
  ∃ (a b : ℤ), a ∈ card_set ∧ b ∈ card_set ∧ a ≠ b ∧ b ≠ 0 ∧ (a : ℚ) / b = -3/2 :=
sorry

end smallest_quotient_smallest_quotient_achievable_l3759_375952


namespace rotation_sum_110_l3759_375992

/-- A structure representing a triangle in a 2D coordinate plane -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The rotation parameters -/
structure RotationParams where
  n : ℝ
  u : ℝ
  v : ℝ

/-- Predicate to check if a rotation transforms one triangle to another -/
def rotates (t1 t2 : Triangle) (r : RotationParams) : Prop :=
  sorry  -- Definition of rotation transformation

theorem rotation_sum_110 (DEF D'E'F' : Triangle) (r : RotationParams) :
  DEF.D = (0, 0) →
  DEF.E = (0, 10) →
  DEF.F = (20, 0) →
  D'E'F'.D = (30, 20) →
  D'E'F'.E = (40, 20) →
  D'E'F'.F = (30, 6) →
  0 < r.n →
  r.n < 180 →
  rotates DEF D'E'F' r →
  r.n + r.u + r.v = 110 := by
  sorry

#check rotation_sum_110

end rotation_sum_110_l3759_375992


namespace projectile_distance_l3759_375980

theorem projectile_distance (v1 v2 t : ℝ) (h1 : v1 = 470) (h2 : v2 = 500) (h3 : t = 90 / 60) :
  v1 * t + v2 * t = 1455 :=
by sorry

end projectile_distance_l3759_375980


namespace soap_survey_ratio_l3759_375965

/-- Represents the survey results of household soap usage -/
structure SoapSurvey where
  total : ℕ
  neither : ℕ
  onlyE : ℕ
  both : ℕ
  onlyB : ℕ

/-- The ratio of households using only brand B to those using both brands -/
def brandBRatio (survey : SoapSurvey) : ℚ :=
  survey.onlyB / survey.both

/-- The survey satisfies the given conditions -/
def validSurvey (survey : SoapSurvey) : Prop :=
  survey.total = 200 ∧
  survey.neither = 80 ∧
  survey.onlyE = 60 ∧
  survey.both = 40 ∧
  survey.total = survey.neither + survey.onlyE + survey.onlyB + survey.both

theorem soap_survey_ratio (survey : SoapSurvey) (h : validSurvey survey) :
  brandBRatio survey = 1/2 := by
  sorry

end soap_survey_ratio_l3759_375965


namespace equation_system_equivalent_quadratic_l3759_375990

theorem equation_system_equivalent_quadratic (x y : ℝ) :
  (3 * x^2 + 4 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 4 = 0) →
  4 * y^2 + 29 * y + 6 = 0 :=
by sorry

end equation_system_equivalent_quadratic_l3759_375990


namespace cos_80_cos_20_plus_sin_80_sin_20_l3759_375938

theorem cos_80_cos_20_plus_sin_80_sin_20 : Real.cos (80 * π / 180) * Real.cos (20 * π / 180) + Real.sin (80 * π / 180) * Real.sin (20 * π / 180) = 1 / 2 := by
  sorry

end cos_80_cos_20_plus_sin_80_sin_20_l3759_375938


namespace flower_planting_cost_l3759_375987

/-- The cost of planting and maintaining flowers with given items -/
theorem flower_planting_cost (flower_cost : ℚ) (h1 : flower_cost = 9) : ∃ total_cost : ℚ,
  let clay_pot_cost := flower_cost + 20
  let soil_cost := flower_cost - 2
  let fertilizer_cost := flower_cost * (1 + 1/2)
  let tools_cost := clay_pot_cost * (1 - 1/4)
  total_cost = flower_cost + clay_pot_cost + soil_cost + fertilizer_cost + tools_cost ∧ 
  total_cost = 80.25 := by
  sorry

end flower_planting_cost_l3759_375987


namespace tangent_line_problem_l3759_375915

theorem tangent_line_problem (f : ℝ → ℝ) (h : ∀ x y, x = 2 ∧ f x = y → 2*x + y - 3 = 0) :
  f 2 + deriv f 2 = -3 := by
  sorry

end tangent_line_problem_l3759_375915


namespace arithmetic_seq_nth_term_l3759_375926

/-- Arithmetic sequence with first term 3 and common difference 2 -/
def arithmeticSeq (n : ℕ) : ℝ := 3 + 2 * (n - 1)

/-- Theorem: If the nth term of the arithmetic sequence is 25, then n is 12 -/
theorem arithmetic_seq_nth_term (n : ℕ) :
  arithmeticSeq n = 25 → n = 12 := by
  sorry

end arithmetic_seq_nth_term_l3759_375926


namespace parametric_to_ordinary_equation_l3759_375979

theorem parametric_to_ordinary_equation (t : ℝ) :
  let x := Real.exp t + Real.exp (-t)
  let y := 2 * (Real.exp t - Real.exp (-t))
  (x^2 / 4) - (y^2 / 16) = 1 ∧ x ≥ 2 := by
  sorry

end parametric_to_ordinary_equation_l3759_375979


namespace top_price_calculation_l3759_375994

def shorts_price : ℝ := 7
def shoes_price : ℝ := 10
def hats_price : ℝ := 6
def socks_price : ℝ := 2

def shorts_quantity : ℕ := 5
def shoes_quantity : ℕ := 2
def hats_quantity : ℕ := 3
def socks_quantity : ℕ := 6
def tops_quantity : ℕ := 4

def total_spent : ℝ := 102

theorem top_price_calculation :
  let other_items_cost := shorts_price * shorts_quantity + shoes_price * shoes_quantity +
                          hats_price * hats_quantity + socks_price * socks_quantity
  let tops_total_cost := total_spent - other_items_cost
  tops_total_cost / tops_quantity = 4.25 := by sorry

end top_price_calculation_l3759_375994


namespace tetrahedron_inscribed_circumscribed_inequality_l3759_375997

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The inscribed sphere of a tetrahedron -/
def inscribedSphere (t : Tetrahedron) : Sphere := sorry

/-- The circumscribed sphere of a tetrahedron -/
def circumscribedSphere (t : Tetrahedron) : Sphere := sorry

/-- The intersection of the planes of the remaining faces -/
def planesIntersection (t : Tetrahedron) : Point3D := sorry

/-- The intersection of a line segment with a sphere -/
def lineIntersectSphere (p1 p2 : Point3D) (s : Sphere) : Point3D := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

theorem tetrahedron_inscribed_circumscribed_inequality (t : Tetrahedron) :
  let I := (inscribedSphere t).center
  let J := planesIntersection t
  let K := lineIntersectSphere I J (circumscribedSphere t)
  distance I K > distance J K := by sorry

end tetrahedron_inscribed_circumscribed_inequality_l3759_375997


namespace ellipse_line_slope_l3759_375996

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = 1/2

/-- A line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  h_slope_pos : 0 < slope

/-- Radii of incircles of triangles formed by points on the ellipse and foci -/
structure IncircleRadii where
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  h_radii_rel : r₁ + r₃ = 2 * r₂

/-- The main theorem statement -/
theorem ellipse_line_slope (E : Ellipse) (l : Line) (R : IncircleRadii) :
  l.slope = 2 * Real.sqrt 2 := by sorry

end ellipse_line_slope_l3759_375996


namespace original_earnings_before_raise_l3759_375908

theorem original_earnings_before_raise (new_earnings : ℝ) (increase_percentage : ℝ) :
  new_earnings = 75 ∧ increase_percentage = 0.25 →
  ∃ original_earnings : ℝ,
    original_earnings * (1 + increase_percentage) = new_earnings ∧
    original_earnings = 60 :=
by sorry

end original_earnings_before_raise_l3759_375908


namespace intersection_quadratic_equations_l3759_375993

theorem intersection_quadratic_equations (p q : ℝ) : 
  let M := {x : ℝ | x^2 - p*x + 6 = 0}
  let N := {x : ℝ | x^2 + 6*x - q = 0}
  (M ∩ N = {2}) → p + q = 21 := by
sorry

end intersection_quadratic_equations_l3759_375993


namespace special_triangle_is_equilateral_l3759_375984

/-- A triangle with sides in geometric progression and angles in arithmetic progression -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  q : ℝ
  -- Angles of the triangle
  α : ℝ
  δ : ℝ
  -- Side lengths form a geometric progression
  side_gp : q > 0
  -- Angles form an arithmetic progression
  angle_ap : True
  -- Sum of angles is 180 degrees
  angle_sum : α - δ + α + (α + δ) = 180

/-- The theorem stating that a SpecialTriangle must be equilateral -/
theorem special_triangle_is_equilateral (t : SpecialTriangle) : t.q = 1 := by
  sorry

end special_triangle_is_equilateral_l3759_375984


namespace power_multiply_l3759_375912

theorem power_multiply (x : ℝ) : x^2 * x^3 = x^5 := by sorry

end power_multiply_l3759_375912


namespace container_production_l3759_375971

/-- Container production problem -/
theorem container_production
  (december : ℕ)
  (h_dec_nov : december = (110 * november) / 100)
  (h_nov_oct : november = (105 * october) / 100)
  (h_oct_sep : october = (120 * september) / 100)
  (h_december : december = 11088) :
  november = 10080 ∧ october = 9600 ∧ september = 8000 := by
  sorry

end container_production_l3759_375971


namespace flowerbed_count_l3759_375978

theorem flowerbed_count (total_seeds : ℕ) (seeds_per_bed : ℕ) (h1 : total_seeds = 32) (h2 : seeds_per_bed = 4) :
  total_seeds / seeds_per_bed = 8 := by
  sorry

end flowerbed_count_l3759_375978


namespace min_value_theorem_l3759_375949

theorem min_value_theorem (x : ℝ) (h : x > 5) : x + 1 / (x - 5) ≥ 7 ∧ ∃ y > 5, y + 1 / (y - 5) = 7 := by
  sorry

end min_value_theorem_l3759_375949


namespace sample_size_calculation_l3759_375946

theorem sample_size_calculation (total_population : ℕ) (sampling_rate : ℚ) :
  total_population = 2000 →
  sampling_rate = 1/10 →
  (total_population : ℚ) * sampling_rate = 200 := by
  sorry

end sample_size_calculation_l3759_375946


namespace garden_area_increase_l3759_375974

/-- Given a rectangular garden with length 60 feet and width 20 feet,
    prove that changing it to a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase :
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_perimeter := 2 * (rect_length + rect_width)
  let square_side := rect_perimeter / 4
  let rect_area := rect_length * rect_width
  let square_area := square_side * square_side
  square_area - rect_area = 400 := by sorry

end garden_area_increase_l3759_375974


namespace unique_three_digit_divisible_by_seven_l3759_375939

theorem unique_three_digit_divisible_by_seven :
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit number
    n % 10 = 5 ∧          -- units digit is 5
    n / 100 = 6 ∧         -- hundreds digit is 6
    n % 7 = 0             -- divisible by 7
  := by sorry

end unique_three_digit_divisible_by_seven_l3759_375939
