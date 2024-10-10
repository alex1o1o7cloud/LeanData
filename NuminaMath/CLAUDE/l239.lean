import Mathlib

namespace chosen_number_proof_l239_23982

theorem chosen_number_proof :
  ∀ x : ℝ, (x / 6 - 15 = 5) → x = 120 := by
  sorry

end chosen_number_proof_l239_23982


namespace min_value_of_function_min_value_achievable_l239_23942

theorem min_value_of_function (x : ℝ) (h : x > 0) : x^2 + 2/x ≥ 3 := by sorry

theorem min_value_achievable : ∃ x > 0, x^2 + 2/x = 3 := by sorry

end min_value_of_function_min_value_achievable_l239_23942


namespace quadratic_rewrite_product_l239_23987

/-- Given a quadratic equation 9x^2 - 30x - 42 that can be rewritten as (ax + b)^2 + c
    where a, b, and c are integers, prove that ab = -15 -/
theorem quadratic_rewrite_product (a b c : ℤ) : 
  (∀ x, 9*x^2 - 30*x - 42 = (a*x + b)^2 + c) → a*b = -15 := by
  sorry

end quadratic_rewrite_product_l239_23987


namespace carpenter_table_difference_carpenter_table_difference_proof_l239_23967

theorem carpenter_table_difference : ℕ → ℕ → ℕ → Prop :=
  fun this_month total difference =>
    this_month = 10 →
    total = 17 →
    difference = this_month - (total - this_month) →
    difference = 3

-- The proof is omitted
theorem carpenter_table_difference_proof : carpenter_table_difference 10 17 3 := by
  sorry

end carpenter_table_difference_carpenter_table_difference_proof_l239_23967


namespace investment_partnership_problem_l239_23952

/-- Investment partnership problem -/
theorem investment_partnership_problem 
  (a b c d : ℝ) -- Investments of partners A, B, C, and D
  (total_profit : ℝ) -- Total profit
  (ha : a = 3 * b) -- A invests 3 times as much as B
  (hb : b = (2/3) * c) -- B invests two-thirds of what C invests
  (hd : d = (1/2) * a) -- D invests half as much as A
  (hp : total_profit = 19900) -- Total profit is Rs.19900
  : b * total_profit / (a + b + c + d) = 2842.86 := by
  sorry

end investment_partnership_problem_l239_23952


namespace equation_real_root_implies_m_value_l239_23923

theorem equation_real_root_implies_m_value (x m : ℝ) (i : ℂ) :
  (∃ x : ℝ, x^2 + (1 - 2*i)*x + 3*m - i = 0) →
  m = 1/12 := by
  sorry

end equation_real_root_implies_m_value_l239_23923


namespace complex_fraction_evaluation_l239_23910

theorem complex_fraction_evaluation :
  2 - (1 / (2 + (1 / (2 - (1 / 3))))) = 21 / 13 := by
  sorry

end complex_fraction_evaluation_l239_23910


namespace scientific_notation_35000_l239_23988

theorem scientific_notation_35000 :
  35000 = 3.5 * (10 ^ 4) := by sorry

end scientific_notation_35000_l239_23988


namespace investment_interest_calculation_l239_23996

theorem investment_interest_calculation (total_investment : ℝ) (first_investment : ℝ) 
  (first_rate : ℝ) (second_rate : ℝ) (h1 : total_investment = 10000) 
  (h2 : first_investment = 6000) (h3 : first_rate = 0.09) (h4 : second_rate = 0.11) : 
  first_investment * first_rate + (total_investment - first_investment) * second_rate = 980 :=
by
  sorry

end investment_interest_calculation_l239_23996


namespace rational_sum_squares_l239_23925

theorem rational_sum_squares (a b c : ℚ) :
  1 / (b - c)^2 + 1 / (c - a)^2 + 1 / (a - b)^2 = (1 / (a - b) + 1 / (b - c) + 1 / (c - a))^2 :=
by sorry

end rational_sum_squares_l239_23925


namespace negation_of_proposition_l239_23922

theorem negation_of_proposition :
  (∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) ↔ ¬(∃ x : ℝ, 2^x - 2*x - 2 < 0) :=
by sorry

end negation_of_proposition_l239_23922


namespace frood_game_theorem_l239_23980

/-- Sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Points earned from eating n froods -/
def eating_points (n : ℕ) : ℕ := 12 * n

/-- The least number of froods for which dropping them earns more points than eating them -/
def least_froods : ℕ := 24

theorem frood_game_theorem :
  least_froods = 24 ∧
  (∀ n : ℕ, n < least_froods → triangular_number n ≤ eating_points n) ∧
  triangular_number least_froods > eating_points least_froods :=
by sorry

end frood_game_theorem_l239_23980


namespace probability_jack_queen_king_hearts_l239_23902

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (face_cards_per_suit : Nat)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { cards := 52,
    suits := 4,
    face_cards_per_suit := 3 }

/-- The probability of drawing a specific set of cards from a deck -/
def probability (d : Deck) (favorable_outcomes : Nat) : ℚ :=
  favorable_outcomes / d.cards

theorem probability_jack_queen_king_hearts (d : Deck := standard_deck) :
  probability d d.face_cards_per_suit = 3 / 52 := by
  sorry

#eval probability standard_deck standard_deck.face_cards_per_suit

end probability_jack_queen_king_hearts_l239_23902


namespace ratio_to_nine_l239_23935

/-- Given a ratio of 5:1 and a number 9, prove that the number x which satisfies this ratio is 45. -/
theorem ratio_to_nine : ∃ x : ℚ, (5 : ℚ) / 1 = x / 9 ∧ x = 45 := by
  sorry

end ratio_to_nine_l239_23935


namespace peter_and_susan_money_l239_23971

/-- The total amount of money Peter and Susan have together -/
def total_money (peter_amount susan_amount : ℚ) : ℚ :=
  peter_amount + susan_amount

/-- Theorem stating that Peter and Susan have 0.65 dollars altogether -/
theorem peter_and_susan_money :
  total_money (2/5) (1/4) = 13/20 := by
  sorry

end peter_and_susan_money_l239_23971


namespace cube_sqrt_16_equals_8_times_8_l239_23965

theorem cube_sqrt_16_equals_8_times_8 : 
  (8 : ℝ) * 8 = (Real.sqrt 16)^3 := by sorry

end cube_sqrt_16_equals_8_times_8_l239_23965


namespace total_visitors_is_440_l239_23911

/-- Represents the survey results from a modern art museum --/
structure SurveyResults where
  totalVisitors : ℕ
  notEnjoyedNotUnderstood : ℕ
  enjoyedAndUnderstood : ℕ
  visitorsBelowFortyRatio : ℚ
  visitorFortyAndAboveRatio : ℚ
  expertRatio : ℚ
  nonExpertRatio : ℚ
  enjoyedAndUnderstoodRatio : ℚ
  fortyAndAboveEnjoyedRatio : ℚ

/-- Theorem stating the total number of visitors based on survey conditions --/
theorem total_visitors_is_440 (survey : SurveyResults) :
  survey.totalVisitors = 440 ∧
  survey.notEnjoyedNotUnderstood = 110 ∧
  survey.enjoyedAndUnderstood = survey.totalVisitors - survey.notEnjoyedNotUnderstood ∧
  survey.visitorsBelowFortyRatio = 2 * survey.visitorFortyAndAboveRatio ∧
  survey.expertRatio = 3/5 ∧
  survey.nonExpertRatio = 2/5 ∧
  survey.enjoyedAndUnderstoodRatio = 3/4 ∧
  survey.fortyAndAboveEnjoyedRatio = 3/5 :=
by sorry

end total_visitors_is_440_l239_23911


namespace right_triangle_from_parabolas_l239_23912

theorem right_triangle_from_parabolas (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hac : a ≠ c)
  (h_intersect : ∃ x₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + 2*a*x₀ + b^2 = 0 ∧ x₀^2 + 2*c*x₀ - b^2 = 0) :
  a^2 = b^2 + c^2 := by
sorry

end right_triangle_from_parabolas_l239_23912


namespace sqrt_two_irrational_l239_23977

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop :=
  ∃ (n d : ℤ), d ≠ 0 ∧ x = n / d

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- Theorem statement
theorem sqrt_two_irrational : IsIrrational (Real.sqrt 2) := by
  sorry

end sqrt_two_irrational_l239_23977


namespace right_triangle_with_special_sides_l239_23915

theorem right_triangle_with_special_sides : ∃ (a b c : ℕ), 
  (a * a + b * b = c * c) ∧ 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  ((a % 4 = 0 ∨ b % 4 = 0) ∧ 
   (a % 3 = 0 ∨ b % 3 = 0) ∧ 
   (a % 5 = 0 ∨ b % 5 = 0 ∨ c % 5 = 0)) :=
by sorry

end right_triangle_with_special_sides_l239_23915


namespace total_books_l239_23957

theorem total_books (joan_books tom_books sarah_books alex_books : ℕ) 
  (h1 : joan_books = 10)
  (h2 : tom_books = 38)
  (h3 : sarah_books = 25)
  (h4 : alex_books = 45) :
  joan_books + tom_books + sarah_books + alex_books = 118 := by
  sorry

end total_books_l239_23957


namespace algebraic_expression_transformation_l239_23907

theorem algebraic_expression_transformation (a b : ℝ) :
  (∀ x, x^2 - 6*x + b = (x - a)^2 - 1) → b - a = 5 := by
  sorry

end algebraic_expression_transformation_l239_23907


namespace complementary_angle_of_30_28_l239_23936

/-- Represents an angle in degrees and minutes -/
structure DegreeMinute where
  degree : ℕ
  minute : ℕ

/-- Converts DegreeMinute to a rational number -/
def DegreeMinute.toRational (dm : DegreeMinute) : ℚ :=
  dm.degree + dm.minute / 60

/-- Theorem: The complementary angle of 30°28' is 59°32' -/
theorem complementary_angle_of_30_28 :
  let angle1 : DegreeMinute := ⟨30, 28⟩
  let complement : DegreeMinute := ⟨59, 32⟩
  DegreeMinute.toRational angle1 + DegreeMinute.toRational complement = 90 := by
  sorry


end complementary_angle_of_30_28_l239_23936


namespace brians_largest_integer_l239_23920

theorem brians_largest_integer (x : ℤ) : 
  (∀ y : ℤ, 10 ≤ 8*y - 70 ∧ 8*y - 70 ≤ 99 → y ≤ x) ↔ x = 21 :=
by sorry

end brians_largest_integer_l239_23920


namespace spears_from_log_l239_23978

/-- The number of spears Marcy can make from a sapling -/
def spears_from_sapling : ℕ := 3

/-- The total number of spears Marcy can make from 6 saplings and a log -/
def total_spears : ℕ := 27

/-- The number of saplings used -/
def num_saplings : ℕ := 6

/-- Theorem: Marcy can make 9 spears from a single log -/
theorem spears_from_log : 
  ∃ (L : ℕ), L = total_spears - (num_saplings * spears_from_sapling) ∧ L = 9 :=
by sorry

end spears_from_log_l239_23978


namespace max_draws_at_23_l239_23903

/-- Represents a lottery draw as a list of distinct integers -/
def LotteryDraw := List Nat

/-- The number of numbers drawn in each lottery draw -/
def drawSize : Nat := 5

/-- The maximum number that can be drawn -/
def maxNumber : Nat := 90

/-- Function to calculate the number of possible draws for a given second smallest number -/
def countDraws (secondSmallest : Nat) : Nat :=
  (secondSmallest - 1) * (maxNumber - secondSmallest) * (maxNumber - secondSmallest - 1) * (maxNumber - secondSmallest - 2)

theorem max_draws_at_23 :
  ∀ m, m ≠ 23 → countDraws 23 ≥ countDraws m :=
sorry

end max_draws_at_23_l239_23903


namespace definite_integral_equals_six_ln_five_l239_23906

theorem definite_integral_equals_six_ln_five :
  ∫ x in (π / 4)..(Real.arccos (1 / Real.sqrt 26)),
    36 / ((6 - Real.tan x) * Real.sin (2 * x)) = 6 * Real.log 5 := by
  sorry

end definite_integral_equals_six_ln_five_l239_23906


namespace product_expansion_l239_23997

theorem product_expansion (x : ℝ) : 
  (x^2 + 3*x - 4) * (2*x^2 - x + 5) = 2*x^4 + 5*x^3 - 6*x^2 + 19*x - 20 := by
  sorry

end product_expansion_l239_23997


namespace function_inequality_l239_23956

theorem function_inequality (f : ℝ → ℝ) :
  (∀ x ≥ 1, f x ≤ x) →
  (∀ x ≥ 1, f (2 * x) / Real.sqrt 2 ≤ f x) →
  (∀ x ≥ 1, f x < Real.sqrt (2 * x)) := by
  sorry

end function_inequality_l239_23956


namespace no_four_digit_number_equals_46_10X_plus_Y_l239_23963

theorem no_four_digit_number_equals_46_10X_plus_Y :
  ¬ ∃ (X Y : ℕ) (a b c d : ℕ),
    (a = 4 ∨ a = 6 ∨ a = X ∨ a = Y) ∧
    (b = 4 ∨ b = 6 ∨ b = X ∨ b = Y) ∧
    (c = 4 ∨ c = 6 ∨ c = X ∨ c = Y) ∧
    (d = 4 ∨ d = 6 ∨ d = X ∨ d = Y) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    1000 ≤ 1000 * a + 100 * b + 10 * c + d ∧
    1000 * a + 100 * b + 10 * c + d < 10000 ∧
    1000 * a + 100 * b + 10 * c + d = 46 * (10 * X + Y) :=
by sorry

end no_four_digit_number_equals_46_10X_plus_Y_l239_23963


namespace rachel_pool_fill_time_l239_23984

/-- Represents the time (in hours) required to fill a pool -/
def fill_time (pool_capacity : ℕ) (num_hoses : ℕ) (flow_rate : ℕ) : ℕ :=
  let total_flow_per_hour := num_hoses * flow_rate * 60
  (pool_capacity + total_flow_per_hour - 1) / total_flow_per_hour

/-- Proves that it takes 33 hours to fill Rachel's pool -/
theorem rachel_pool_fill_time :
  fill_time 30000 5 3 = 33 := by
  sorry

end rachel_pool_fill_time_l239_23984


namespace arccos_cos_eq_x_div_3_l239_23938

theorem arccos_cos_eq_x_div_3 (x : ℝ) :
  -Real.pi ≤ x ∧ x ≤ 2 * Real.pi →
  (Real.arccos (Real.cos x) = x / 3 ↔ x = 0 ∨ x = 3 * Real.pi / 2 ∨ x = -3 * Real.pi / 2) :=
by sorry

end arccos_cos_eq_x_div_3_l239_23938


namespace parabola_point_to_directrix_distance_l239_23916

/-- Proves that for a parabola y² = 2px containing the point (1, √5), 
    the distance from this point to the directrix is 9/4 -/
theorem parabola_point_to_directrix_distance :
  ∀ (p : ℝ), 
  (5 : ℝ) = 2 * p →  -- Condition from y² = 2px with (1, √5)
  (1 : ℝ) + p / 2 = 9 / 4 := by
  sorry

end parabola_point_to_directrix_distance_l239_23916


namespace wallet_cost_l239_23948

theorem wallet_cost (wallet_cost purse_cost : ℝ) : 
  purse_cost = 4 * wallet_cost - 3 →
  wallet_cost + purse_cost = 107 →
  wallet_cost = 22 := by
sorry

end wallet_cost_l239_23948


namespace inequality_proof_l239_23945

theorem inequality_proof (a b : ℝ) (h : 4 * b + a = 1) : a^2 + 4 * b^2 ≥ 1/5 := by
  sorry

end inequality_proof_l239_23945


namespace jeans_pricing_l239_23930

theorem jeans_pricing (cost : ℝ) (cost_positive : cost > 0) :
  let retailer_price := cost * (1 + 0.4)
  let customer_price := retailer_price * (1 + 0.15)
  (customer_price - cost) / cost = 0.61 := by
  sorry

end jeans_pricing_l239_23930


namespace share_of_y_l239_23976

def total_amount : ℕ := 690
def ratio_x : ℕ := 5
def ratio_y : ℕ := 7
def ratio_z : ℕ := 11

theorem share_of_y : 
  (total_amount * ratio_y) / (ratio_x + ratio_y + ratio_z) = 210 := by
  sorry

end share_of_y_l239_23976


namespace parabola_ellipse_focus_coincide_l239_23931

/-- The value of 'a' for a parabola y^2 = ax whose focus coincides with 
    the left focus of the ellipse x^2/6 + y^2/2 = 1 -/
theorem parabola_ellipse_focus_coincide : ∃ (a : ℝ), 
  (∀ (x y : ℝ), y^2 = a*x → x^2/6 + y^2/2 = 1 → 
    (x = -2 ∧ y = 0)) → a = -8 := by
  sorry

end parabola_ellipse_focus_coincide_l239_23931


namespace aquarium_fish_count_l239_23944

theorem aquarium_fish_count (total : ℕ) : 
  (total : ℚ) / 3 = 60 ∧  -- One third of fish are blue
  (total : ℚ) / 4 ≤ (total : ℚ) / 3 ∧  -- One fourth of fish are yellow
  (total : ℚ) - ((total : ℚ) / 3 + (total : ℚ) / 4) = 45 ∧  -- The rest are red
  (60 : ℚ) / 2 = 30 ∧  -- 50% of blue fish have spots
  30 * (100 : ℚ) / 60 = 50 ∧  -- Verify 50% of blue fish have spots
  9 * (100 : ℚ) / 45 = 20  -- Verify 20% of red fish have spots
  → total = 140 := by
sorry

end aquarium_fish_count_l239_23944


namespace min_value_of_function_l239_23943

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 ∧ ∃ y > 1, y + 1 / (y - 1) = 3 :=
by sorry

end min_value_of_function_l239_23943


namespace overtime_pay_is_3_20_l239_23981

/-- Calculates the overtime pay rate given the following conditions:
  * Regular week has 5 working days
  * Regular working hours per day is 8
  * Regular pay rate is 2.40 rupees per hour
  * Total earnings in 4 weeks is 432 rupees
  * Total hours worked in 4 weeks is 175
-/
def overtime_pay_rate (
  regular_days_per_week : ℕ)
  (regular_hours_per_day : ℕ)
  (regular_pay_rate : ℚ)
  (total_earnings : ℚ)
  (total_hours : ℕ) : ℚ :=
by
  sorry

/-- Theorem stating that the overtime pay rate is 3.20 rupees per hour -/
theorem overtime_pay_is_3_20 :
  overtime_pay_rate 5 8 (240/100) 432 175 = 320/100 :=
by
  sorry

end overtime_pay_is_3_20_l239_23981


namespace count_triangles_l239_23926

/-- A point in the plane with coordinates that are multiples of 3 -/
structure Point :=
  (x : ℤ)
  (y : ℤ)
  (x_multiple : 3 ∣ x)
  (y_multiple : 3 ∣ y)

/-- The equation 47x + y = 2353 -/
def satisfies_equation (p : Point) : Prop :=
  47 * p.x + p.y = 2353

/-- The area of triangle OPQ where O is the origin -/
def triangle_area (p q : Point) : ℚ :=
  (p.x * q.y - q.x * p.y : ℚ) / 2

/-- The main theorem -/
theorem count_triangles :
  ∃ (triangle_set : Finset (Point × Point)),
    (∀ (p q : Point), (p, q) ∈ triangle_set →
      p ≠ q ∧
      satisfies_equation p ∧
      satisfies_equation q ∧
      (triangle_area p q).num ≠ 0 ∧
      (triangle_area p q).den = 1) ∧
    triangle_set.card = 64 ∧
    ∀ (p q : Point),
      p ≠ q →
      satisfies_equation p →
      satisfies_equation q →
      (triangle_area p q).num ≠ 0 →
      (triangle_area p q).den = 1 →
      (p, q) ∈ triangle_set :=
sorry

end count_triangles_l239_23926


namespace phi_value_l239_23924

theorem phi_value (φ : Real) (h1 : 0 < φ ∧ φ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (15 * π / 180) = Real.cos φ - Real.sin φ) : 
  φ = 30 * π / 180 := by
sorry

end phi_value_l239_23924


namespace chessboard_border_covering_l239_23939

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Number of ways to cover a 2xn rectangle with 1x2 dominos -/
def cover_2xn (n : ℕ) : ℕ := fib (n + 1)

/-- Number of ways to cover the 2-unit wide border of an 8x8 chessboard with 1x2 dominos -/
def cover_chessboard_border : ℕ :=
  let f9 := cover_2xn 8
  let f10 := cover_2xn 9
  let f11 := cover_2xn 10
  2 + 2 * f11^2 * f9^2 + 12 * f11 * f10^2 * f9 + 2 * f10^4

theorem chessboard_border_covering :
  cover_chessboard_border = 146458404 := by sorry

end chessboard_border_covering_l239_23939


namespace tan_alpha_plus_pi_fourth_l239_23909

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : α > Real.pi / 2) 
  (h2 : α < Real.pi) 
  (h3 : Real.sin α = 5 / 13) : 
  Real.tan (α + Real.pi / 4) = 7 / 17 := by
sorry

end tan_alpha_plus_pi_fourth_l239_23909


namespace recruitment_probabilities_l239_23918

/-- Represents the recruitment scenario -/
structure RecruitmentScenario where
  totalQuestions : Nat
  drawnQuestions : Nat
  knownQuestions : Nat
  minCorrect : Nat

/-- Calculates the probability of proceeding to the interview stage -/
def probabilityToInterview (scenario : RecruitmentScenario) : Rat :=
  sorry

/-- Represents the probability distribution of correctly answerable questions -/
structure ProbabilityDistribution where
  p0 : Rat
  p1 : Rat
  p2 : Rat
  p3 : Rat

/-- Calculates the probability distribution of correctly answerable questions -/
def probabilityDistribution (scenario : RecruitmentScenario) : ProbabilityDistribution :=
  sorry

theorem recruitment_probabilities 
  (scenario : RecruitmentScenario)
  (h1 : scenario.totalQuestions = 10)
  (h2 : scenario.drawnQuestions = 3)
  (h3 : scenario.knownQuestions = 6)
  (h4 : scenario.minCorrect = 2) :
  probabilityToInterview scenario = 2/3 ∧
  let dist := probabilityDistribution scenario
  dist.p0 = 1/30 ∧ dist.p1 = 3/10 ∧ dist.p2 = 1/2 ∧ dist.p3 = 1/6 :=
sorry

end recruitment_probabilities_l239_23918


namespace difference_of_squares_consecutive_evens_l239_23913

def consecutive_even_integers (a b c : ℤ) : Prop :=
  b = a + 2 ∧ c = b + 2

theorem difference_of_squares_consecutive_evens (a b c : ℤ) :
  consecutive_even_integers a b c →
  a + b + c = 1992 →
  c^2 - a^2 = 5312 :=
by sorry

end difference_of_squares_consecutive_evens_l239_23913


namespace manuscript_pages_count_l239_23954

/-- The cost structure and revision information for a manuscript typing service. -/
structure ManuscriptTyping where
  first_time_cost : ℕ
  revision_cost : ℕ
  pages_revised_once : ℕ
  pages_revised_twice : ℕ
  total_cost : ℕ

/-- Calculates the total number of pages in a manuscript given the typing costs and revision information. -/
def total_pages (m : ManuscriptTyping) : ℕ :=
  (m.total_cost - (m.pages_revised_once * (m.first_time_cost + m.revision_cost) + 
   m.pages_revised_twice * (m.first_time_cost + 2 * m.revision_cost))) / m.first_time_cost + 
   m.pages_revised_once + m.pages_revised_twice

/-- Theorem stating that for the given manuscript typing scenario, the total number of pages is 100. -/
theorem manuscript_pages_count (m : ManuscriptTyping) 
  (h1 : m.first_time_cost = 6)
  (h2 : m.revision_cost = 4)
  (h3 : m.pages_revised_once = 35)
  (h4 : m.pages_revised_twice = 15)
  (h5 : m.total_cost = 860) :
  total_pages m = 100 := by
  sorry

end manuscript_pages_count_l239_23954


namespace area_between_circles_l239_23908

theorem area_between_circles (r : ℝ) (R : ℝ) : 
  r = 3 →                   -- radius of smaller circle
  R = 3 * r →               -- radius of larger circle is three times the smaller
  π * R^2 - π * r^2 = 72*π  -- area between circles is 72π
  := by sorry

end area_between_circles_l239_23908


namespace pictures_per_album_l239_23953

theorem pictures_per_album 
  (total_pictures : ℕ) 
  (phone_pictures camera_pictures : ℕ) 
  (num_albums : ℕ) 
  (h1 : total_pictures = phone_pictures + camera_pictures)
  (h2 : phone_pictures = 5)
  (h3 : camera_pictures = 35)
  (h4 : num_albums = 8)
  (h5 : total_pictures % num_albums = 0) :
  total_pictures / num_albums = 5 := by
sorry

end pictures_per_album_l239_23953


namespace oranges_bought_l239_23968

/-- Represents the fruit shopping scenario over a week -/
structure FruitShopping where
  apples : ℕ
  oranges : ℕ
  total_fruits : apples + oranges = 5
  total_cost : ℕ
  cost_is_whole_dollars : total_cost % 100 = 0
  cost_calculation : total_cost = 30 * apples + 45 * oranges + 20

/-- Theorem stating that the number of oranges bought is 2 -/
theorem oranges_bought (shop : FruitShopping) : shop.oranges = 2 := by
  sorry

end oranges_bought_l239_23968


namespace optimal_purchase_l239_23934

/-- Represents the cost and quantity of soccer balls and basketballs --/
structure BallPurchase where
  soccer_price : ℝ
  basketball_price : ℝ
  soccer_quantity : ℕ
  basketball_quantity : ℕ

/-- Defines the conditions of the ball purchase problem --/
def valid_purchase (p : BallPurchase) : Prop :=
  p.soccer_price + 3 * p.basketball_price = 275 ∧
  3 * p.soccer_price + 2 * p.basketball_price = 300 ∧
  p.soccer_quantity + p.basketball_quantity = 80 ∧
  p.soccer_quantity ≤ 3 * p.basketball_quantity

/-- Calculates the total cost of a ball purchase --/
def total_cost (p : BallPurchase) : ℝ :=
  p.soccer_price * p.soccer_quantity + p.basketball_price * p.basketball_quantity

/-- Theorem stating the most cost-effective purchase plan --/
theorem optimal_purchase :
  ∃ (p : BallPurchase),
    valid_purchase p ∧
    p.soccer_price = 50 ∧
    p.basketball_price = 75 ∧
    p.soccer_quantity = 60 ∧
    p.basketball_quantity = 20 ∧
    (∀ (q : BallPurchase), valid_purchase q → total_cost p ≤ total_cost q) :=
  sorry

end optimal_purchase_l239_23934


namespace watermelon_seeds_theorem_l239_23993

/-- Calculates the total number of seeds in three watermelons -/
def total_seeds (slices1 slices2 slices3 seeds_per_slice1 seeds_per_slice2 seeds_per_slice3 : ℕ) : ℕ :=
  slices1 * seeds_per_slice1 + slices2 * seeds_per_slice2 + slices3 * seeds_per_slice3

/-- Proves that the total number of seeds in the given watermelons is 6800 -/
theorem watermelon_seeds_theorem :
  total_seeds 40 30 50 60 80 40 = 6800 := by
  sorry

#eval total_seeds 40 30 50 60 80 40

end watermelon_seeds_theorem_l239_23993


namespace right_triangle_leg_length_l239_23914

/-- Proves that in a right triangle with an area of 800 square feet and one leg of 40 feet, 
    the length of the other leg is also 40 feet. -/
theorem right_triangle_leg_length 
  (area : ℝ) 
  (base : ℝ) 
  (h : area = 800) 
  (b : base = 40) : 
  (2 * area) / base = 40 := by
sorry

end right_triangle_leg_length_l239_23914


namespace survey_respondents_l239_23986

theorem survey_respondents (preferred_x : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) : 
  preferred_x = 150 → ratio_x = 5 → ratio_y = 1 → 
  ∃ (total : ℕ), total = preferred_x + (preferred_x * ratio_y) / ratio_x ∧ total = 180 := by
  sorry

end survey_respondents_l239_23986


namespace min_value_expression_l239_23940

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (8 * z) / (3 * x + 2 * y) + (8 * x) / (2 * y + 3 * z) + y / (x + z) ≥ 4.5 := by
  sorry

end min_value_expression_l239_23940


namespace problem_solution_l239_23947

def A : Set ℝ := {x | |x - 2| < 3}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

theorem problem_solution :
  (∀ x, x ∈ (A ∩ (Set.univ \ B 3)) ↔ 3 ≤ x ∧ x < 5) ∧
  (A ∩ B 8 = {x | -1 < x ∧ x < 4}) := by
  sorry

end problem_solution_l239_23947


namespace vector_conclusions_l239_23946

-- Define the vector space
variable {V : Type*} [AddCommGroup V]

-- Define the vectors
variable (O D E M : V)

-- Define the given equation
axiom given_equation : D - O + (E - O) = M - O

-- Theorem to prove the three correct conclusions
theorem vector_conclusions :
  (M - O + (D - O) = E - O) ∧
  (M - O - (E - O) = D - O) ∧
  ((O - D) + (O - E) = O - M) := by
  sorry

end vector_conclusions_l239_23946


namespace four_point_segment_ratio_l239_23955

/-- Given four distinct points on a plane with segment lengths a, a, a, a, 2a, and b,
    prove that b = a√3 -/
theorem four_point_segment_ratio (a b : ℝ) :
  ∃ (A B C D : ℝ × ℝ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    ({dist A B, dist A C, dist A D, dist B C, dist B D, dist C D} : Finset ℝ) =
      {a, a, a, a, 2*a, b} →
    b = a * Real.sqrt 3 :=
by sorry

end four_point_segment_ratio_l239_23955


namespace abs_sum_lt_abs_diff_when_product_negative_l239_23964

theorem abs_sum_lt_abs_diff_when_product_negative (a b : ℝ) : 
  a * b < 0 → |a + b| < |a - b| := by
sorry

end abs_sum_lt_abs_diff_when_product_negative_l239_23964


namespace tetrahedron_subdivision_existence_l239_23992

theorem tetrahedron_subdivision_existence :
  ∃ (n : ℕ), (1 / 2 : ℝ) ^ n < (1 / 100 : ℝ) := by sorry

end tetrahedron_subdivision_existence_l239_23992


namespace product_mod_seven_l239_23917

theorem product_mod_seven : (2031 * 2032 * 2033 * 2034) % 7 = 3 := by
  sorry

end product_mod_seven_l239_23917


namespace min_value_of_sum_fractions_l239_23974

theorem min_value_of_sum_fractions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) / c + (a + c) / b + (b + c) / a ≥ 6 ∧
  ((a + b) / c + (a + c) / b + (b + c) / a = 6 ↔ a = b ∧ b = c) :=
sorry

end min_value_of_sum_fractions_l239_23974


namespace eight_ninths_position_l239_23969

/-- Represents a fraction as a pair of natural numbers -/
def Fraction := ℕ × ℕ

/-- The sequence of fractions as described in the problem -/
def fraction_sequence : ℕ → Fraction := sorry

/-- The sum of numerator and denominator of a fraction -/
def sum_of_parts (f : Fraction) : ℕ := f.1 + f.2

/-- The position of a fraction in the sequence -/
def position_in_sequence (f : Fraction) : ℕ := sorry

/-- The main theorem: 8/9 is at position 128 in the sequence -/
theorem eight_ninths_position :
  position_in_sequence (8, 9) = 128 := by sorry

end eight_ninths_position_l239_23969


namespace problem_statement_l239_23989

theorem problem_statement (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4*b^2 = 1/(a*b) + 3) :
  (ab ≤ 1) ∧ (b > a → 1/a^3 - 1/b^3 > 3*(1/a - 1/b)) := by
  sorry

end problem_statement_l239_23989


namespace smallest_gcd_bc_l239_23900

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 240) (h2 : Nat.gcd a c = 1001) :
  ∃ (b' c' : ℕ+), Nat.gcd b'.val c'.val = 1 ∧ 
    ∀ (b'' c'' : ℕ+), Nat.gcd a b''.val = 240 → Nat.gcd a c''.val = 1001 → 
      Nat.gcd b''.val c''.val ≥ 1 :=
sorry

end smallest_gcd_bc_l239_23900


namespace charlie_widget_production_l239_23990

/-- Charlie's widget production problem -/
theorem charlie_widget_production 
  (w t : ℕ) -- w: widgets per hour, t: hours worked on Thursday
  (h1 : w = 3 * t) -- Condition: w = 3t
  : w * t - (w + 6) * (t - 3) = 3 * t + 18 := by
  sorry


end charlie_widget_production_l239_23990


namespace cubic_equation_roots_l239_23991

theorem cubic_equation_roots : ∃ (x₁ x₂ x₃ : ℝ),
  (x₁ = 3 ∧ x₂ = -3 ∧ x₃ = 5) ∧
  (∀ x : ℝ, x^3 - 5*x^2 - 9*x + 45 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
  (x₁ = -x₂) := by
  sorry

end cubic_equation_roots_l239_23991


namespace lines_intersect_implies_planes_intersect_l239_23975

-- Define the space
variable (S : Type*) [NormedAddCommGroup S] [InnerProductSpace ℝ S] [CompleteSpace S]

-- Define lines and planes
def Line (S : Type*) [NormedAddCommGroup S] := Set S
def Plane (S : Type*) [NormedAddCommGroup S] := Set S

-- Define the subset relation
def IsSubset {S : Type*} (A B : Set S) := A ⊆ B

-- Define intersection for lines and planes
def Intersect {S : Type*} (A B : Set S) := ∃ x, x ∈ A ∧ x ∈ B

-- Theorem statement
theorem lines_intersect_implies_planes_intersect
  (m n : Line S) (α β : Plane S)
  (hm : m ≠ n) (hα : α ≠ β)
  (hmα : IsSubset m α) (hnβ : IsSubset n β)
  (hmn : Intersect m n) :
  Intersect α β :=
sorry

end lines_intersect_implies_planes_intersect_l239_23975


namespace winner_percentage_approx_62_l239_23962

/-- Represents an election with two candidates -/
structure Election :=
  (total_votes : ℕ)
  (winner_votes : ℕ)
  (margin : ℕ)

/-- Calculates the percentage of votes for the winner -/
def winner_percentage (e : Election) : ℚ :=
  (e.winner_votes : ℚ) / (e.total_votes : ℚ) * 100

/-- Theorem stating the winner's percentage in the given election -/
theorem winner_percentage_approx_62 (e : Election) 
  (h1 : e.winner_votes = 837)
  (h2 : e.margin = 324)
  (h3 : e.total_votes = e.winner_votes + (e.winner_votes - e.margin)) :
  ∃ (p : ℚ), abs (winner_percentage e - p) < 1 ∧ p = 62 := by
  sorry

#eval winner_percentage { total_votes := 1350, winner_votes := 837, margin := 324 }

end winner_percentage_approx_62_l239_23962


namespace complex_equation_solution_l239_23921

theorem complex_equation_solution :
  ∀ z : ℂ, z + 5 - 6*I = 3 + 4*I → z = -2 + 10*I :=
by
  sorry

end complex_equation_solution_l239_23921


namespace expression_nonnegative_iff_l239_23941

/-- The expression (x^2-4x+4)/(9-x^3) is nonnegative if and only if x ≤ 3 -/
theorem expression_nonnegative_iff (x : ℝ) : (x^2 - 4*x + 4) / (9 - x^3) ≥ 0 ↔ x ≤ 3 := by
  sorry

end expression_nonnegative_iff_l239_23941


namespace box_area_product_equals_volume_squared_l239_23959

/-- Given a rectangular box with dimensions x, y, and z, 
    prove that the product of the areas of its three pairs of opposite faces 
    is equal to the square of its volume. -/
theorem box_area_product_equals_volume_squared 
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y) * (y * z) * (z * x) = (x * y * z)^2 := by
  sorry

end box_area_product_equals_volume_squared_l239_23959


namespace three_digit_segment_sum_l239_23901

/-- Represents the number of horizontal and vertical segments for a digit --/
structure DigitSegments where
  horizontal : Nat
  vertical : Nat

/-- The set of all digits and their corresponding segment counts --/
def digit_segments : Fin 10 → DigitSegments := fun d =>
  match d with
  | 0 => ⟨2, 4⟩
  | 1 => ⟨0, 2⟩
  | 2 => ⟨2, 3⟩
  | 3 => ⟨3, 3⟩
  | 4 => ⟨1, 3⟩
  | 5 => ⟨2, 2⟩
  | 6 => ⟨1, 3⟩
  | 7 => ⟨1, 2⟩
  | 8 => ⟨3, 4⟩
  | 9 => ⟨2, 3⟩

theorem three_digit_segment_sum :
  ∃ (a b c : Fin 10),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (digit_segments a).horizontal + (digit_segments b).horizontal + (digit_segments c).horizontal = 5 ∧
    (digit_segments a).vertical + (digit_segments b).vertical + (digit_segments c).vertical = 10 ∧
    a.val + b.val + c.val = 9 :=
by sorry


end three_digit_segment_sum_l239_23901


namespace cube_volume_from_surface_area_l239_23972

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 54 →
  volume = (surface_area / 6) * (surface_area / 6).sqrt →
  volume = 27 := by
  sorry

end cube_volume_from_surface_area_l239_23972


namespace enrollment_ways_count_l239_23979

/-- The number of elective courses -/
def num_courses : ℕ := 4

/-- The number of students -/
def num_students : ℕ := 3

/-- The number of courses each student must choose -/
def courses_per_student : ℕ := 2

/-- The number of different ways each course can have students enrolled -/
def num_enrollment_ways : ℕ := 114

theorem enrollment_ways_count :
  (num_courses = 4) →
  (num_students = 3) →
  (courses_per_student = 2) →
  (num_enrollment_ways = 114) := by
  sorry

end enrollment_ways_count_l239_23979


namespace division_sum_equals_111_l239_23994

theorem division_sum_equals_111 : (111 / 3) + (222 / 6) + (333 / 9) = 111 := by
  sorry

end division_sum_equals_111_l239_23994


namespace sum_plus_difference_l239_23960

theorem sum_plus_difference (a b c : ℝ) (h : c = a + b + 5.1) : c = 48.9 :=
  by sorry

#check sum_plus_difference 20.2 33.8 48.9

end sum_plus_difference_l239_23960


namespace work_trip_speed_l239_23966

/-- Proves that given a round trip of 3 hours, where the return journey takes 1.2 hours at 120 km/h,
    and the journey to work takes 1.8 hours, the average speed to work is 80 km/h. -/
theorem work_trip_speed (total_time : ℝ) (return_time : ℝ) (return_speed : ℝ) (to_work_time : ℝ)
    (h1 : total_time = 3)
    (h2 : return_time = 1.2)
    (h3 : return_speed = 120)
    (h4 : to_work_time = 1.8)
    (h5 : total_time = return_time + to_work_time) :
    (return_speed * return_time) / to_work_time = 80 := by
  sorry

end work_trip_speed_l239_23966


namespace guise_hot_dog_consumption_l239_23933

/-- Proves that given the conditions of Guise's hot dog consumption, the daily increase was 2 hot dogs. -/
theorem guise_hot_dog_consumption (monday_consumption : ℕ) (total_by_wednesday : ℕ) (daily_increase : ℕ) : 
  monday_consumption = 10 →
  total_by_wednesday = 36 →
  total_by_wednesday = monday_consumption + (monday_consumption + daily_increase) + (monday_consumption + 2 * daily_increase) →
  daily_increase = 2 := by
  sorry

end guise_hot_dog_consumption_l239_23933


namespace track_length_l239_23905

/-- The length of a circular track given specific running conditions -/
theorem track_length : 
  ∀ (L : ℝ),
  (∃ (d₁ d₂ : ℝ),
    d₁ = 100 ∧
    d₂ = 100 ∧
    d₁ + (L / 2 - d₁) = L / 2 ∧
    (L - d₁) + (L / 2 - d₁ + d₂) = L) →
  L = 200 :=
by sorry

end track_length_l239_23905


namespace simple_interest_double_l239_23970

/-- The factor by which a sum of money increases under simple interest -/
def simple_interest_factor (rate : ℝ) (time : ℝ) : ℝ :=
  1 + rate * time

theorem simple_interest_double :
  simple_interest_factor 0.1 10 = 2 := by
  sorry

end simple_interest_double_l239_23970


namespace rice_mixture_price_l239_23998

/-- Represents the price of rice in Rupees per kilogram -/
@[ext] structure RicePrice where
  price : ℝ

/-- Represents a mixture of two types of rice -/
structure RiceMixture where
  price1 : RicePrice
  price2 : RicePrice
  ratio : ℝ
  mixtureCost : ℝ

/-- The theorem statement -/
theorem rice_mixture_price (mix : RiceMixture) 
  (h1 : mix.price1.price = 16)
  (h2 : mix.ratio = 3)
  (h3 : mix.mixtureCost = 18) :
  mix.price2.price = 24 := by
  sorry

end rice_mixture_price_l239_23998


namespace brent_initial_lollipops_l239_23958

/-- The number of lollipops Brent initially received -/
def initial_lollipops : ℕ := sorry

/-- The number of Kit-Kat bars Brent received -/
def kit_kat : ℕ := 5

/-- The number of Hershey kisses Brent received -/
def hershey_kisses : ℕ := 3 * kit_kat

/-- The number of boxes of Nerds Brent received -/
def nerds : ℕ := 8

/-- The number of Baby Ruths Brent had -/
def baby_ruths : ℕ := 10

/-- The number of Reese's Peanut Butter Cups Brent had -/
def reeses_cups : ℕ := baby_ruths / 2

/-- The number of lollipops Brent gave to his sister -/
def lollipops_given : ℕ := 5

/-- The total number of candy pieces Brent had after giving away lollipops -/
def remaining_candy : ℕ := 49

theorem brent_initial_lollipops :
  initial_lollipops = 11 :=
by sorry

end brent_initial_lollipops_l239_23958


namespace exposed_sides_is_30_l239_23929

/-- Represents a regular polygon -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the arrangement of polygons -/
structure PolygonArrangement :=
  (triangle : RegularPolygon)
  (square : RegularPolygon)
  (pentagon : RegularPolygon)
  (hexagon : RegularPolygon)
  (heptagon : RegularPolygon)
  (octagon : RegularPolygon)
  (nonagon : RegularPolygon)

/-- Calculates the number of exposed sides in the arrangement -/
def exposedSides (arrangement : PolygonArrangement) : ℕ :=
  arrangement.triangle.sides +
  arrangement.square.sides +
  arrangement.pentagon.sides +
  arrangement.hexagon.sides +
  arrangement.heptagon.sides +
  arrangement.octagon.sides +
  arrangement.nonagon.sides -
  12 -- Subtracting the shared sides

/-- The specific arrangement described in the problem -/
def specificArrangement : PolygonArrangement :=
  { triangle := ⟨3⟩
  , square := ⟨4⟩
  , pentagon := ⟨5⟩
  , hexagon := ⟨6⟩
  , heptagon := ⟨7⟩
  , octagon := ⟨8⟩
  , nonagon := ⟨9⟩ }

/-- Theorem stating that the number of exposed sides in the specific arrangement is 30 -/
theorem exposed_sides_is_30 : exposedSides specificArrangement = 30 := by
  sorry

end exposed_sides_is_30_l239_23929


namespace y_order_l239_23928

/-- Quadratic function f(x) = -x² + 4x - 5 -/
def f (x : ℝ) : ℝ := -x^2 + 4*x - 5

/-- Given three points on the graph of f -/
def A : ℝ × ℝ := (-4, f (-4))
def B : ℝ × ℝ := (-3, f (-3))
def C : ℝ × ℝ := (1, f 1)

/-- y-coordinates of the points -/
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

theorem y_order : y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end y_order_l239_23928


namespace donalds_apples_l239_23904

theorem donalds_apples (marin_apples total_apples : ℕ) 
  (h1 : marin_apples = 9)
  (h2 : total_apples = 11) :
  total_apples - marin_apples = 2 := by
  sorry

end donalds_apples_l239_23904


namespace line_circle_min_value_l239_23950

/-- Given a line ax + by + 1 = 0 that divides a circle into two equal areas, 
    prove that the minimum value of 1/(2a) + 2/b is 8 -/
theorem line_circle_min_value (a b : ℝ) : 
  a > 0 → 
  b > 0 → 
  (∀ x y : ℝ, a * x + b * y + 1 = 0 → (x + 4)^2 + (y + 1)^2 = 16 → 
    (∃ k : ℝ, k > 0 ∧ k * ((x + 4)^2 + (y + 1)^2) = 16 ∧ 
    k * (a * x + b * y + 1) = 0)) → 
  (∀ x y : ℝ, (1 / (2 * a) + 2 / b) ≥ 8) ∧ 
  (∃ x y : ℝ, 1 / (2 * a) + 2 / b = 8) := by
  sorry

end line_circle_min_value_l239_23950


namespace four_digit_number_l239_23932

/-- Represents a 6x6 grid of numbers -/
def Grid := Matrix (Fin 6) (Fin 6) Nat

/-- Check if a number is within the range 1 to 6 -/
def inRange (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 6

/-- Check if a list of numbers contains no duplicates -/
def noDuplicates (l : List Nat) : Prop := l.Nodup

/-- Check if a number is prime -/
def isPrime (n : Nat) : Prop := Nat.Prime n

/-- Check if a number is composite -/
def isComposite (n : Nat) : Prop := ¬(isPrime n) ∧ n > 1

/-- Theorem: Under the given conditions, the four-digit number is 4123 -/
theorem four_digit_number (g : Grid) 
  (range_check : ∀ i j, inRange (g i j))
  (row_unique : ∀ i, noDuplicates (List.ofFn (λ j => g i j)))
  (col_unique : ∀ j, noDuplicates (List.ofFn (λ i => g i j)))
  (rect_unique : ∀ i j, noDuplicates [g i j, g i (j+1), g i (j+2), g (i+1) j, g (i+1) (j+1), g (i+1) (j+2)])
  (circle_sum : ∀ i j, isComposite (g i j + g (i+1) j) → ∀ k l, (k, l) ≠ (i, j) → isPrime (g k l + g (k+1) l))
  : ∃ i j k l, g i j = 4 ∧ g k j = 1 ∧ g k l = 2 ∧ g i l = 3 := by
  sorry

end four_digit_number_l239_23932


namespace scientific_notation_of_104000_l239_23999

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

/-- The given number from the problem -/
def givenNumber : ℝ := 104000

theorem scientific_notation_of_104000 :
  toScientificNotation givenNumber = ScientificNotation.mk 1.04 5 (by norm_num) := by sorry

end scientific_notation_of_104000_l239_23999


namespace pancake_fundraiser_l239_23973

/-- The civic league's pancake breakfast fundraiser --/
theorem pancake_fundraiser 
  (pancake_price : ℝ) 
  (bacon_price : ℝ) 
  (pancake_stacks : ℕ) 
  (bacon_slices : ℕ) 
  (h1 : pancake_price = 4)
  (h2 : bacon_price = 2)
  (h3 : pancake_stacks = 60)
  (h4 : bacon_slices = 90) :
  pancake_price * (pancake_stacks : ℝ) + bacon_price * (bacon_slices : ℝ) = 420 :=
by sorry

end pancake_fundraiser_l239_23973


namespace students_using_red_color_l239_23927

theorem students_using_red_color 
  (total_students : ℕ) 
  (green_users : ℕ) 
  (both_colors : ℕ) 
  (h1 : total_students = 70) 
  (h2 : green_users = 52) 
  (h3 : both_colors = 38) : 
  total_students + both_colors - green_users = 56 := by
  sorry

end students_using_red_color_l239_23927


namespace least_three_digit_multiple_of_13_l239_23995

theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), 
  n = 104 ∧ 
  13 ∣ n ∧ 
  100 ≤ n ∧ 
  n < 1000 ∧
  ∀ m : ℕ, (13 ∣ m ∧ 100 ≤ m ∧ m < 1000) → n ≤ m :=
by sorry

end least_three_digit_multiple_of_13_l239_23995


namespace shopping_expenditure_l239_23951

/-- Represents the percentage spent on clothing -/
def clothing_percentage : ℝ := sorry

/-- Represents the percentage spent on food -/
def food_percentage : ℝ := 20

/-- Represents the percentage spent on other items -/
def other_percentage : ℝ := 30

/-- Represents the tax rate on clothing -/
def clothing_tax_rate : ℝ := 4

/-- Represents the tax rate on other items -/
def other_tax_rate : ℝ := 8

/-- Represents the total tax rate as a percentage of pre-tax spending -/
def total_tax_rate : ℝ := 4.4

theorem shopping_expenditure :
  clothing_percentage + food_percentage + other_percentage = 100 ∧
  clothing_percentage * clothing_tax_rate / 100 + other_percentage * other_tax_rate / 100 = total_tax_rate ∧
  clothing_percentage = 50 := by sorry

end shopping_expenditure_l239_23951


namespace left_placement_equals_100a_plus_b_l239_23985

/-- A single-digit number is a natural number from 0 to 9 -/
def SingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- A two-digit number is a natural number from 10 to 99 -/
def TwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The three-digit number formed by placing a to the left of b -/
def LeftPlacement (a b : ℕ) : ℕ := 100 * a + b

theorem left_placement_equals_100a_plus_b (a b : ℕ) 
  (ha : SingleDigit a) (hb : TwoDigit b) : 
  LeftPlacement a b = 100 * a + b := by
  sorry

end left_placement_equals_100a_plus_b_l239_23985


namespace regression_line_at_12_l239_23937

def regression_line (x_mean y_mean slope : ℝ) (x : ℝ) : ℝ :=
  slope * (x - x_mean) + y_mean

theorem regression_line_at_12 
  (x_mean : ℝ) 
  (y_mean : ℝ) 
  (slope : ℝ) 
  (h1 : x_mean = 10) 
  (h2 : y_mean = 4) 
  (h3 : slope = 0.6) :
  regression_line x_mean y_mean slope 12 = 5.2 := by
  sorry

end regression_line_at_12_l239_23937


namespace eliminate_denominators_l239_23983

theorem eliminate_denominators (x : ℝ) : 
  (x / 2 - 1 = (x - 1) / 3) ↔ (3 * x - 6 = 2 * (x - 1)) := by
  sorry

end eliminate_denominators_l239_23983


namespace euler_conjecture_counterexample_l239_23949

theorem euler_conjecture_counterexample : 133^5 + 110^5 + 84^5 + 27^5 = 144^5 := by
  sorry

end euler_conjecture_counterexample_l239_23949


namespace sum_unit_digit_not_two_l239_23961

theorem sum_unit_digit_not_two (n : ℕ) : (n * (n + 1) / 2) % 10 ≠ 2 := by
  sorry

end sum_unit_digit_not_two_l239_23961


namespace circle_center_coord_sum_l239_23919

theorem circle_center_coord_sum (x y : ℝ) :
  x^2 + y^2 = 4*x - 6*y + 9 →
  ∃ (center_x center_y : ℝ), center_x + center_y = -1 ∧
    ∀ (point_x point_y : ℝ),
      (point_x - center_x)^2 + (point_y - center_y)^2 = (x - center_x)^2 + (y - center_y)^2 := by
  sorry

end circle_center_coord_sum_l239_23919
