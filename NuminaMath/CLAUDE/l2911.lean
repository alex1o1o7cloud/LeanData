import Mathlib

namespace commercial_break_length_is_47_l2911_291131

/-- Calculates the total length of a commercial break given the following conditions:
    - Three commercials of 5, 6, and 7 minutes
    - Eleven 2-minute commercials
    - Two of the 2-minute commercials overlap with a 3-minute interruption and restart after
-/
def commercial_break_length : ℕ :=
  let long_commercials := 5 + 6 + 7
  let short_commercials := 11 * 2
  let interruption := 3
  let restarted_commercials := 2 * 2
  long_commercials + short_commercials + interruption + restarted_commercials

/-- Theorem stating that the commercial break length is 47 minutes -/
theorem commercial_break_length_is_47 : commercial_break_length = 47 := by
  sorry

end commercial_break_length_is_47_l2911_291131


namespace total_votes_is_102000_l2911_291184

/-- The number of votes that switched from the first to the second candidate -/
def votes_switched_to_second : ℕ := 16000

/-- The number of votes that switched from the first to the third candidate -/
def votes_switched_to_third : ℕ := 8000

/-- The ratio of votes between the winner and the second place in the second round -/
def winner_ratio : ℕ := 5

/-- Represents the election results -/
structure ElectionResult where
  first_round_votes : ℕ
  second_round_first : ℕ
  second_round_second : ℕ
  second_round_third : ℕ

/-- Checks if the election result satisfies all conditions -/
def is_valid_result (result : ElectionResult) : Prop :=
  -- First round: all candidates have equal votes
  result.first_round_votes * 3 = result.second_round_first + result.second_round_second + result.second_round_third
  -- Vote transfers in second round
  ∧ result.second_round_first = result.first_round_votes - votes_switched_to_second - votes_switched_to_third
  ∧ result.second_round_second = result.first_round_votes + votes_switched_to_second
  ∧ result.second_round_third = result.first_round_votes + votes_switched_to_third
  -- Winner has 5 times as many votes as the second place
  ∧ (result.second_round_second = winner_ratio * result.second_round_first
     ∨ result.second_round_second = winner_ratio * result.second_round_third
     ∨ result.second_round_third = winner_ratio * result.second_round_first
     ∨ result.second_round_third = winner_ratio * result.second_round_second)

/-- The main theorem: prove that the total number of votes is 102000 -/
theorem total_votes_is_102000 :
  ∃ (result : ElectionResult), is_valid_result result ∧ result.first_round_votes * 3 = 102000 :=
sorry

end total_votes_is_102000_l2911_291184


namespace certain_value_proof_l2911_291172

theorem certain_value_proof (N : ℝ) (h : 0.4 * N = 420) : (1/4) * (1/3) * (2/5) * N = 35 := by
  sorry

end certain_value_proof_l2911_291172


namespace length_of_A_l2911_291153

def A : ℝ × ℝ := (0, 9)
def B : ℝ × ℝ := (0, 12)
def C : ℝ × ℝ := (2, 8)

def on_line_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

def intersect_at (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))

theorem length_of_A'B' :
  ∀ A' B' : ℝ × ℝ,
  on_line_y_eq_x A' →
  on_line_y_eq_x B' →
  intersect_at A A' C →
  intersect_at B B' C →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2 * Real.sqrt 2 := by
  sorry

end length_of_A_l2911_291153


namespace largest_triangle_area_21_points_l2911_291143

/-- A configuration of points where every three adjacent points form an equilateral triangle --/
structure TriangleConfiguration where
  num_points : ℕ
  small_triangle_area : ℝ

/-- The area of the largest triangle formed by the configuration --/
def largest_triangle_area (config : TriangleConfiguration) : ℝ :=
  sorry

/-- Theorem stating that for a configuration of 21 points with unit area small triangles,
    the largest triangle has an area of 13 --/
theorem largest_triangle_area_21_points :
  let config : TriangleConfiguration := { num_points := 21, small_triangle_area := 1 }
  largest_triangle_area config = 13 := by
  sorry

end largest_triangle_area_21_points_l2911_291143


namespace intersection_of_A_and_B_l2911_291192

-- Define sets A and B
def A : Set ℝ := {x | (x - 1) * (x - 3) < 0}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l2911_291192


namespace line_equation_from_slope_and_intercept_l2911_291179

/-- Given a line with slope 4 and x-intercept 2, its equation is 4x - y - 8 = 0 -/
theorem line_equation_from_slope_and_intercept :
  ∀ (f : ℝ → ℝ), 
    (∀ x y, f y = 4 * (x - 2)) →  -- slope is 4, x-intercept is 2
    (f 0 = -8) →                  -- y-intercept is -8
    ∀ x, 4 * x - f x - 8 = 0 :=
by
  sorry

end line_equation_from_slope_and_intercept_l2911_291179


namespace wage_increase_l2911_291118

theorem wage_increase (original_wage : ℝ) (increase_percentage : ℝ) 
  (h1 : original_wage = 60)
  (h2 : increase_percentage = 20) : 
  original_wage * (1 + increase_percentage / 100) = 72 := by
  sorry

end wage_increase_l2911_291118


namespace fifth_term_of_sequence_l2911_291111

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fifth_term_of_sequence (x y : ℝ) :
  ∀ a : ℕ → ℝ, arithmetic_sequence a →
  a 1 = x - y → a 2 = x → a 3 = x + y → a 4 = x + 2*y →
  a 5 = x + 3*y := by
sorry

end fifth_term_of_sequence_l2911_291111


namespace fourth_pentagon_dots_l2911_291175

/-- Represents the number of dots in a pentagon at a given position in the sequence -/
def dots_in_pentagon (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else dots_in_pentagon (n - 1) + 5 * (n - 1)

/-- The main theorem stating that the fourth pentagon contains 31 dots -/
theorem fourth_pentagon_dots :
  dots_in_pentagon 4 = 31 := by
  sorry

#eval dots_in_pentagon 4

end fourth_pentagon_dots_l2911_291175


namespace simplify_expression_l2911_291103

theorem simplify_expression : (8 * 10^7) / (4 * 10^2) = 200000 := by
  sorry

end simplify_expression_l2911_291103


namespace non_working_games_l2911_291130

theorem non_working_games (total_games : ℕ) (price_per_game : ℕ) (total_earnings : ℕ) : 
  total_games = 15 → price_per_game = 5 → total_earnings = 30 → 
  total_games - (total_earnings / price_per_game) = 9 := by
sorry

end non_working_games_l2911_291130


namespace group_difference_theorem_l2911_291181

theorem group_difference_theorem :
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  A - B = 4397 := by sorry

end group_difference_theorem_l2911_291181


namespace third_month_sale_l2911_291199

/-- Calculates the missing sale amount given the other sales and the required average -/
def missing_sale (sale1 sale2 sale4 sale5 sale6 required_average : ℕ) : ℕ :=
  6 * required_average - (sale1 + sale2 + sale4 + sale5 + sale6)

/-- Proves that the missing sale in the third month is 10555 -/
theorem third_month_sale : missing_sale 2500 6500 7230 7000 11915 7500 = 10555 := by
  sorry

end third_month_sale_l2911_291199


namespace reinforcement_is_1900_l2911_291163

/-- Calculates the reinforcement size given initial garrison size, provision duration, and remaining provision duration after reinforcement --/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) (days_before_reinforcement : ℕ) (remaining_duration : ℕ) : ℕ :=
  let provisions_left := initial_garrison * (initial_duration - days_before_reinforcement)
  (provisions_left / remaining_duration) - initial_garrison

/-- The reinforcement size for the given problem --/
def problem_reinforcement : ℕ := calculate_reinforcement 2000 54 15 20

/-- Theorem stating that the reinforcement size for the given problem is 1900 --/
theorem reinforcement_is_1900 : problem_reinforcement = 1900 := by
  sorry

#eval problem_reinforcement

end reinforcement_is_1900_l2911_291163


namespace max_dimes_possible_l2911_291147

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- The total amount in cents -/
def total_amount : ℕ := 550

/-- Theorem stating the maximum number of dimes possible -/
theorem max_dimes_possible (quarters nickels dimes : ℕ) 
  (h1 : quarters = nickels)
  (h2 : dimes ≥ 3 * quarters)
  (h3 : quarters * coin_value "quarter" + 
        nickels * coin_value "nickel" + 
        dimes * coin_value "dime" = total_amount) :
  dimes ≤ 28 :=
sorry

end max_dimes_possible_l2911_291147


namespace root_ratio_to_power_l2911_291148

theorem root_ratio_to_power (x : ℝ) (h : x > 0) :
  (x^(1/3)) / (x^(1/5)) = x^(2/15) :=
by sorry

end root_ratio_to_power_l2911_291148


namespace simplify_expression_l2911_291157

theorem simplify_expression (a b : ℝ) : (30*a + 70*b) + (15*a + 45*b) - (12*a + 60*b) = 33*a + 55*b := by
  sorry

end simplify_expression_l2911_291157


namespace smallest_number_divisible_l2911_291139

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < 257 → ¬(((m + 7) % 8 = 0) ∧ ((m + 7) % 11 = 0) ∧ ((m + 7) % 24 = 0))) ∧
  ((257 + 7) % 8 = 0) ∧ ((257 + 7) % 11 = 0) ∧ ((257 + 7) % 24 = 0) := by
  sorry

end smallest_number_divisible_l2911_291139


namespace circle_diameter_from_area_l2911_291170

theorem circle_diameter_from_area :
  ∀ (A : ℝ) (d : ℝ),
    A = 225 * Real.pi →
    d = 2 * Real.sqrt (A / Real.pi) →
    d = 30 :=
by
  sorry

end circle_diameter_from_area_l2911_291170


namespace traffic_light_change_probability_l2911_291115

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total duration of a traffic light cycle -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed in a given interval -/
def changeObservationWindow (cycle : TrafficLightCycle) (interval : ℕ) : ℕ :=
  3 * interval  -- There are 3 color changes in a cycle

/-- Calculates the probability of observing a color change during a given interval -/
def probabilityOfChange (cycle : TrafficLightCycle) (interval : ℕ) : ℚ :=
  (changeObservationWindow cycle interval : ℚ) / (cycleDuration cycle : ℚ)

theorem traffic_light_change_probability :
  ∀ (cycle : TrafficLightCycle),
    cycle.green = 45 →
    cycle.yellow = 5 →
    cycle.red = 40 →
    probabilityOfChange cycle 4 = 2 / 15 := by
  sorry

end traffic_light_change_probability_l2911_291115


namespace coefficient_x3y3_l2911_291150

/-- The coefficient of x³y³ in the expansion of (x+2y)(x+y)⁵ is 30 -/
theorem coefficient_x3y3 : Int :=
  30

#check coefficient_x3y3

end coefficient_x3y3_l2911_291150


namespace power_multiplication_l2911_291168

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l2911_291168


namespace evaluate_expression_l2911_291128

theorem evaluate_expression : (2 : ℕ) ^ (3 ^ 2) + 3 ^ (2 ^ 3) = 7073 := by
  sorry

end evaluate_expression_l2911_291128


namespace slope_range_theorem_l2911_291176

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line t
def line_t (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the condition for O being outside the circle with diameter PQ
def O_outside_circle (P Q : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  x₁ * x₂ + y₁ * y₂ > 0

theorem slope_range_theorem (k : ℝ) :
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧
    C₁ P.1 P.2 ∧ C₁ Q.1 Q.2 ∧
    line_t k P.1 P.2 ∧ line_t k Q.1 Q.2 ∧
    O_outside_circle P Q) →
  k ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 3 / 2) ∪ Set.Ioo (Real.sqrt 3 / 2) 2 :=
by sorry

end slope_range_theorem_l2911_291176


namespace partial_fraction_decomposition_l2911_291183

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ),
    (∀ (x : ℝ), x ≠ 0 →
      (x^3 - 2*x^2 + x - 5) / (x^4 + x^2) = A / x^2 + (B*x + C) / (x^2 + 1)) ↔
    (A = -5 ∧ B = 1 ∧ C = 3) :=
by sorry

end partial_fraction_decomposition_l2911_291183


namespace tangent_lines_parallel_to_given_line_l2911_291198

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

/-- The slope of the line parallel to 4x - y = 1 -/
def m : ℝ := 4

theorem tangent_lines_parallel_to_given_line :
  ∃ (a b : ℝ), 
    (f' a = m) ∧ 
    (b = f a) ∧ 
    ((4*x - y = 0) ∨ (4*x - y - 4 = 0)) ∧
    (∀ x y : ℝ, y - b = m * (x - a) → y = f x) :=
sorry

end tangent_lines_parallel_to_given_line_l2911_291198


namespace no_2014_ambiguous_numbers_l2911_291177

/-- A positive integer k is 2014-ambiguous if both x^2 + kx + 2014 and x^2 + kx - 2014 have two integer roots -/
def is_2014_ambiguous (k : ℕ+) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℤ,
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    x₁^2 + k * x₁ + 2014 = 0 ∧
    x₂^2 + k * x₂ + 2014 = 0 ∧
    y₁^2 + k * y₁ - 2014 = 0 ∧
    y₂^2 + k * y₂ - 2014 = 0

theorem no_2014_ambiguous_numbers : ¬∃ k : ℕ+, is_2014_ambiguous k := by
  sorry

end no_2014_ambiguous_numbers_l2911_291177


namespace three_digit_numbers_with_repeated_digits_l2911_291186

/-- The number of digits available (0 to 9) -/
def num_digits : ℕ := 10

/-- The number of digits in the numbers we're considering -/
def num_places : ℕ := 3

/-- The total number of possible three-digit numbers -/
def total_numbers : ℕ := 900

/-- The number of three-digit numbers without repeated digits -/
def non_repeating_numbers : ℕ := 9 * 9 * 8

theorem three_digit_numbers_with_repeated_digits : 
  total_numbers - non_repeating_numbers = 252 := by
  sorry

end three_digit_numbers_with_repeated_digits_l2911_291186


namespace average_difference_l2911_291154

theorem average_difference (a b c : ℝ) 
  (hab : (a + b) / 2 = 40) 
  (hbc : (b + c) / 2 = 60) : 
  c - a = 40 := by
sorry

end average_difference_l2911_291154


namespace trolley_theorem_l2911_291162

def trolley_problem (X : ℕ) : Prop :=
  let initial_passengers := 10
  let second_stop_off := 3
  let second_stop_on := 2 * initial_passengers
  let third_stop_off := 18
  let third_stop_on := 2
  let fourth_stop_off := 5
  let fourth_stop_on := X
  let final_passengers := 
    initial_passengers - second_stop_off + second_stop_on - 
    third_stop_off + third_stop_on - fourth_stop_off + fourth_stop_on
  final_passengers = 6 + X

theorem trolley_theorem (X : ℕ) : 
  trolley_problem X :=
sorry

end trolley_theorem_l2911_291162


namespace min_cans_required_l2911_291119

def can_capacity : ℕ := 10
def tank_capacity : ℕ := 140

theorem min_cans_required : 
  ∃ n : ℕ, n * can_capacity ≥ tank_capacity ∧ 
  ∀ m : ℕ, m * can_capacity ≥ tank_capacity → m ≥ n :=
by sorry

end min_cans_required_l2911_291119


namespace inequality_proof_l2911_291138

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (b * c / a) + (a * c / b) + (a * b / c) > a + b + c := by
  sorry

end inequality_proof_l2911_291138


namespace other_root_of_quadratic_l2911_291101

theorem other_root_of_quadratic (p : ℝ) : 
  (2 : ℝ)^2 + 4*2 - p = 0 → 
  ∃ (x : ℝ), x^2 + 4*x - p = 0 ∧ x = -6 := by
sorry

end other_root_of_quadratic_l2911_291101


namespace inequality_implies_k_range_l2911_291174

theorem inequality_implies_k_range (k : ℝ) : 
  (∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0) → k ≥ 1 := by
  sorry

end inequality_implies_k_range_l2911_291174


namespace consecutive_integers_coprime_l2911_291156

theorem consecutive_integers_coprime (n : ℤ) : 
  ∃ k ∈ Finset.range 10, ∀ m ∈ Finset.range 10, m ≠ k → Int.gcd (n + k) (n + m) = 1 := by
  sorry

end consecutive_integers_coprime_l2911_291156


namespace weight_difference_l2911_291133

theorem weight_difference (w_a w_b w_c w_d w_e : ℝ) : 
  (w_a + w_b + w_c) / 3 = 50 →
  (w_a + w_b + w_c + w_d) / 4 = 53 →
  (w_b + w_c + w_d + w_e) / 4 = 51 →
  w_a = 73 →
  w_e > w_d →
  w_e - w_d = 3 := by
sorry

end weight_difference_l2911_291133


namespace simplify_expression_1_simplify_expression_2_l2911_291167

-- Define the variables
variable (a b c x : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 :
  a^3*b - 2*b^2*c + 5*a^3*b - 3*a^3*b + 2*c*b^2 = 3*a^3*b := by sorry

-- Theorem for the second expression
theorem simplify_expression_2 :
  (2*x^2 - 1/2 + 3*x) - 4*(x - x^2 + 1/2) = 6*x^2 - x - 5/2 := by sorry

end simplify_expression_1_simplify_expression_2_l2911_291167


namespace max_product_theorem_l2911_291194

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 10000 ∧ a < 100000 ∧ b ≥ 10000 ∧ b < 100000 ∧
  (∀ d : ℕ, d < 10 → (d.digits 10).count d + (a.digits 10).count d + (b.digits 10).count d = 1)

def max_product : ℕ := 96420 * 87531

theorem max_product_theorem :
  ∀ a b : ℕ, is_valid_pair a b → a * b ≤ max_product :=
by sorry

end max_product_theorem_l2911_291194


namespace square_value_preserving_shifted_square_value_preserving_l2911_291191

-- Define a "value-preserving" interval
def is_value_preserving (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y)

-- Theorem for f(x) = x^2
theorem square_value_preserving :
  ∀ a b : ℝ, is_value_preserving (fun x => x^2) a b ↔ a = 0 ∧ b = 1 := by sorry

-- Theorem for g(x) = x^2 + m
theorem shifted_square_value_preserving :
  ∀ m : ℝ, m ≠ 0 →
  (∃ a b : ℝ, is_value_preserving (fun x => x^2 + m) a b) ↔
  (m ∈ Set.Icc (-1) (-3/4) ∪ Set.Ioo 0 (1/4)) := by sorry

end square_value_preserving_shifted_square_value_preserving_l2911_291191


namespace triangle_inequality_l2911_291120

theorem triangle_inequality (a b c : ℝ) (h_area : (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) = 1/4) (h_circumradius : (a * b * c) / (4 * (1/4)) = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c < 1/a + 1/b + 1/c := by
  sorry

end triangle_inequality_l2911_291120


namespace sequence_e_is_perfect_cube_l2911_291173

def sequence_a (n : ℕ) : ℕ := n

def sequence_b (n : ℕ) : ℕ :=
  if sequence_a n % 3 ≠ 0 then sequence_a n else 0

def sequence_c (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_b

def sequence_d (n : ℕ) : ℕ :=
  if sequence_c n % 3 ≠ 0 then sequence_c n else 0

def sequence_e (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_d

theorem sequence_e_is_perfect_cube (n : ℕ) :
  sequence_e n = ((n + 2) / 3)^3 := by sorry

end sequence_e_is_perfect_cube_l2911_291173


namespace cookie_sales_revenue_l2911_291127

theorem cookie_sales_revenue : 
  let chocolate_cookies : ℕ := 220
  let chocolate_price : ℕ := 1
  let vanilla_cookies : ℕ := 70
  let vanilla_price : ℕ := 2
  chocolate_cookies * chocolate_price + vanilla_cookies * vanilla_price = 360 := by
sorry

end cookie_sales_revenue_l2911_291127


namespace mike_ride_length_l2911_291125

/-- Represents the taxi fare structure -/
structure TaxiFare where
  startFee : ℝ
  perMileFee : ℝ
  bridgeToll : ℝ

/-- Calculates the total fare for a given ride -/
def calculateFare (fare : TaxiFare) (miles : ℝ) : ℝ :=
  fare.startFee + fare.perMileFee * miles + fare.bridgeToll

theorem mike_ride_length :
  let mikeFare : TaxiFare := { startFee := 2.5, perMileFee := 0.25, bridgeToll := 0 }
  let annieFare : TaxiFare := { startFee := 2.5, perMileFee := 0.25, bridgeToll := 5 }
  let annieMiles : ℝ := 26
  ∃ mikeMiles : ℝ, mikeMiles = 36 ∧ 
    calculateFare mikeFare mikeMiles = calculateFare annieFare annieMiles :=
by sorry

end mike_ride_length_l2911_291125


namespace six_balls_two_boxes_l2911_291145

/-- The number of ways to distribute n indistinguishable balls into 2 distinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ := n + 1

/-- Theorem: There are 7 ways to distribute 6 indistinguishable balls into 2 distinguishable boxes -/
theorem six_balls_two_boxes : distribute_balls 6 = 7 := by
  sorry

end six_balls_two_boxes_l2911_291145


namespace lynne_book_purchase_l2911_291189

/-- The number of books about the solar system Lynne bought -/
def solar_system_books : ℕ := 2

/-- The total amount Lynne spent -/
def total_spent : ℕ := 75

/-- The number of books about cats Lynne bought -/
def cat_books : ℕ := 7

/-- The number of magazines Lynne bought -/
def magazines : ℕ := 3

/-- The cost of each book -/
def book_cost : ℕ := 7

/-- The cost of each magazine -/
def magazine_cost : ℕ := 4

theorem lynne_book_purchase :
  cat_books * book_cost + solar_system_books * book_cost + magazines * magazine_cost = total_spent :=
by sorry

end lynne_book_purchase_l2911_291189


namespace percentage_owning_only_cats_l2911_291187

/-- The percentage of students owning only cats in a survey. -/
theorem percentage_owning_only_cats
  (total_students : ℕ)
  (cat_owners : ℕ)
  (dog_owners : ℕ)
  (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 120)
  (h3 : dog_owners = 200)
  (h4 : both_owners = 40) :
  (cat_owners - both_owners) / total_students * 100 = 16 :=
by sorry

end percentage_owning_only_cats_l2911_291187


namespace f_composition_of_three_l2911_291166

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_composition_of_three : f (f (f (f 3))) = 8 := by
  sorry

end f_composition_of_three_l2911_291166


namespace language_courses_enrollment_l2911_291136

theorem language_courses_enrollment (total : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ)
  (french_german : ℕ) (german_spanish : ℕ) (spanish_french : ℕ) (all_three : ℕ) :
  total = 180 →
  french = 60 →
  german = 50 →
  spanish = 35 →
  french_german = 20 →
  german_spanish = 15 →
  spanish_french = 10 →
  all_three = 5 →
  total - (french + german + spanish - french_german - german_spanish - spanish_french + all_three) = 80 := by
sorry

end language_courses_enrollment_l2911_291136


namespace initial_workers_count_l2911_291126

/-- The work rate of workers (depth dug per worker per hour) -/
def work_rate : ℝ := sorry

/-- The initial number of workers -/
def initial_workers : ℕ := sorry

/-- The depth of the hole in meters -/
def hole_depth : ℝ := 30

theorem initial_workers_count : initial_workers = 45 := by
  have h1 : initial_workers * 8 * work_rate = hole_depth := sorry
  have h2 : (initial_workers + 15) * 6 * work_rate = hole_depth := sorry
  sorry


end initial_workers_count_l2911_291126


namespace quadratic_one_solution_l2911_291160

theorem quadratic_one_solution (m : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + m = 0) ↔ m = 49/12 := by
  sorry

end quadratic_one_solution_l2911_291160


namespace minimum_growth_rate_for_doubling_output_l2911_291106

theorem minimum_growth_rate_for_doubling_output :
  let r : ℝ := Real.sqrt 2 - 1
  ∀ x : ℝ, (1 + x)^2 ≥ 2 → x ≥ r :=
by sorry

end minimum_growth_rate_for_doubling_output_l2911_291106


namespace slope_range_l2911_291117

theorem slope_range (m : ℝ) : ((8 - m) / (m - 5) > 1) → (5 < m ∧ m < 13/2) := by
  sorry

end slope_range_l2911_291117


namespace work_completion_time_l2911_291142

theorem work_completion_time (a_total_days b_remaining_days : ℚ) 
  (h1 : a_total_days = 15)
  (h2 : b_remaining_days = 10) : 
  let a_work_days : ℚ := 5
  let a_work_fraction : ℚ := a_work_days / a_total_days
  let b_work_fraction : ℚ := 1 - a_work_fraction
  b_remaining_days / b_work_fraction = 15 := by sorry

end work_completion_time_l2911_291142


namespace two_colored_line_exists_l2911_291165

-- Define the color type
inductive Color
| Red
| Blue
| Green
| Yellow

-- Define the grid
def Grid := ℤ × ℤ → Color

-- Define the property that vertices of any 1x1 square are painted in different colors
def ValidColoring (g : Grid) : Prop :=
  ∀ x y : ℤ, 
    g (x, y) ≠ g (x + 1, y) ∧
    g (x, y) ≠ g (x, y + 1) ∧
    g (x, y) ≠ g (x + 1, y + 1) ∧
    g (x + 1, y) ≠ g (x, y + 1) ∧
    g (x + 1, y) ≠ g (x + 1, y + 1) ∧
    g (x, y + 1) ≠ g (x + 1, y + 1)

-- Define a line in the grid
def Line := ℤ → ℤ × ℤ

-- Define the property that a line has nodes painted in exactly two colors
def TwoColoredLine (g : Grid) (l : Line) : Prop :=
  ∃ c1 c2 : Color, c1 ≠ c2 ∧ ∀ z : ℤ, g (l z) = c1 ∨ g (l z) = c2

-- The main theorem
theorem two_colored_line_exists (g : Grid) (h : ValidColoring g) : 
  ∃ l : Line, TwoColoredLine g l := by
  sorry

end two_colored_line_exists_l2911_291165


namespace hyperbola_foci_distance_l2911_291155

/-- Given a hyperbola with equation 4x^2 - y^2 + 64 = 0, 
    if a point P on this hyperbola is at distance 1 from one focus,
    then it is at distance 17 from the other focus. -/
theorem hyperbola_foci_distance (x y : ℝ) (P : ℝ × ℝ) :
  P.1 = x ∧ P.2 = y →  -- P is the point (x, y)
  4 * x^2 - y^2 + 64 = 0 →  -- P is on the hyperbola
  (∃ F₁ : ℝ × ℝ, (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = 1) →  -- Distance to one focus is 1
  (∃ F₂ : ℝ × ℝ, (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = 17^2) :=  -- Distance to other focus is 17
by sorry

end hyperbola_foci_distance_l2911_291155


namespace box_length_proof_l2911_291182

theorem box_length_proof (x : ℕ) (cube_side : ℕ) : 
  (x * 48 * 12 = 80 * cube_side^3) → 
  (x % cube_side = 0) → 
  (48 % cube_side = 0) → 
  (12 % cube_side = 0) → 
  x = 240 := by
sorry

end box_length_proof_l2911_291182


namespace katies_soccer_game_granola_boxes_l2911_291185

/-- Given the number of kids, granola bars per kid, and bars per box, 
    calculate the number of boxes needed. -/
def boxes_needed (num_kids : ℕ) (bars_per_kid : ℕ) (bars_per_box : ℕ) : ℕ :=
  (num_kids * bars_per_kid + bars_per_box - 1) / bars_per_box

/-- Prove that for Katie's soccer game scenario, 5 boxes are needed. -/
theorem katies_soccer_game_granola_boxes : 
  boxes_needed 30 2 12 = 5 := by
  sorry

end katies_soccer_game_granola_boxes_l2911_291185


namespace f_negative_expression_l2911_291109

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the function f for x > 0
def f_positive (x : ℝ) : ℝ :=
  x^3 + x + 1

-- Theorem statement
theorem f_negative_expression 
  (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x < 0, f x = -x^3 - x + 1 := by
sorry

end f_negative_expression_l2911_291109


namespace play_role_assignment_l2911_291129

theorem play_role_assignment (men : ℕ) (women : ℕ) : men = 7 ∧ women = 5 →
  (men * women * (Nat.choose (men + women - 2) 4)) = 7350 :=
by sorry

end play_role_assignment_l2911_291129


namespace power_relation_l2911_291124

theorem power_relation (m n : ℤ) : 
  (3 : ℝ) ^ m = (1 : ℝ) / 27 → 
  ((1 : ℝ) / 2) ^ n = 16 → 
  (m : ℝ) ^ n = 1 / 81 := by
sorry

end power_relation_l2911_291124


namespace neighbor_house_height_l2911_291197

/-- Given three houses where one is 80 feet tall, another is 70 feet tall,
    and the 80-foot house is 3 feet shorter than the average height of all three houses,
    prove that the height of the third house must be 99 feet. -/
theorem neighbor_house_height (h1 h2 h3 : ℝ) : 
  h1 = 80 → h2 = 70 → h1 = (h1 + h2 + h3) / 3 - 3 → h3 = 99 := by
  sorry

end neighbor_house_height_l2911_291197


namespace exponent_equation_solution_l2911_291141

theorem exponent_equation_solution :
  ∃ x : ℝ, (4 : ℝ)^x * (4 : ℝ)^x * (4 : ℝ)^x * (4 : ℝ)^x = (256 : ℝ)^4 ∧ x = 4 := by
  sorry

end exponent_equation_solution_l2911_291141


namespace square_of_negative_triple_l2911_291151

theorem square_of_negative_triple (a : ℝ) : (-3 * a)^2 = 9 * a^2 := by sorry

end square_of_negative_triple_l2911_291151


namespace polynomial_symmetry_l2911_291104

/-- Given a polynomial function g(x) = ax^7 + bx^3 + dx^2 + cx - 8,
    prove that if g(-7) = 3 and d = 0, then g(7) = -19 -/
theorem polynomial_symmetry (a b c d : ℝ) :
  let g := λ x : ℝ => a * x^7 + b * x^3 + d * x^2 + c * x - 8
  (g (-7) = 3) → (d = 0) → (g 7 = -19) := by
  sorry

end polynomial_symmetry_l2911_291104


namespace system_solution_l2911_291196

theorem system_solution : 
  ∀ x y : ℝ, 
  (x^3 + y^3 = 19 ∧ x^2 + y^2 + 5*x + 5*y + x*y = 12) ↔ 
  ((x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3)) := by
  sorry

end system_solution_l2911_291196


namespace prime_sum_product_l2911_291169

theorem prime_sum_product (x₁ x₂ x₃ : ℕ) 
  (h_prime₁ : Nat.Prime x₁) 
  (h_prime₂ : Nat.Prime x₂) 
  (h_prime₃ : Nat.Prime x₃) 
  (h_sum : x₁ + x₂ + x₃ = 68) 
  (h_sum_prod : x₁*x₂ + x₁*x₃ + x₂*x₃ = 1121) : 
  x₁ * x₂ * x₃ = 1978 := by
sorry

end prime_sum_product_l2911_291169


namespace power_two_305_mod_9_l2911_291137

theorem power_two_305_mod_9 : 2^305 % 9 = 5 := by sorry

end power_two_305_mod_9_l2911_291137


namespace six_partitions_into_three_or_fewer_l2911_291135

/-- The number of ways to partition n indistinguishable objects into k or fewer non-empty parts -/
def partitions_into_k_or_fewer (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to partition 6 indistinguishable objects into 3 or fewer non-empty parts -/
theorem six_partitions_into_three_or_fewer : partitions_into_k_or_fewer 6 3 = 6 := by
  sorry

end six_partitions_into_three_or_fewer_l2911_291135


namespace evaluate_expression_l2911_291144

theorem evaluate_expression : 
  2011 * 20122012 * 201320132013 - 2013 * 20112011 * 201220122012 = 
  -2 * 2012 * 2013 * 10001 * 100010001 := by sorry

end evaluate_expression_l2911_291144


namespace unique_pair_power_sum_l2911_291161

theorem unique_pair_power_sum : 
  ∃! (a b : ℕ), ∀ (n : ℕ), ∃ (c : ℕ), a^n + b^n = c^(n+1) :=
by
  -- The proof goes here
  sorry

end unique_pair_power_sum_l2911_291161


namespace equation_with_operations_l2911_291102

theorem equation_with_operations : ∃ (op1 op2 op3 : ℕ → ℕ → ℕ), 
  op1 6 (op2 3 (op3 4 2)) = 24 :=
sorry

end equation_with_operations_l2911_291102


namespace lcm_1640_1020_l2911_291110

theorem lcm_1640_1020 : Nat.lcm 1640 1020 = 83640 := by
  sorry

end lcm_1640_1020_l2911_291110


namespace unique_number_divisible_by_24_with_cube_root_between_9_and_9_1_l2911_291116

theorem unique_number_divisible_by_24_with_cube_root_between_9_and_9_1 :
  ∃! (n : ℕ), n > 0 ∧ 24 ∣ n ∧ 9 < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < 9.1 :=
by
  -- The proof goes here
  sorry

end unique_number_divisible_by_24_with_cube_root_between_9_and_9_1_l2911_291116


namespace phillips_remaining_money_l2911_291146

/-- Calculates the remaining money after purchases --/
def remaining_money (initial : ℕ) (orange_cost apple_cost candy_cost : ℕ) : ℕ :=
  initial - (orange_cost + apple_cost + candy_cost)

/-- Theorem stating that given the specific amounts, the remaining money is $50 --/
theorem phillips_remaining_money :
  remaining_money 95 14 25 6 = 50 := by
  sorry

end phillips_remaining_money_l2911_291146


namespace sum_of_valid_a_values_l2911_291149

theorem sum_of_valid_a_values : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, (∀ y : ℝ, ¬(y - 1 ≥ (2*y - 1)/3 ∧ -1/2*(y - a) > 0)) ∧ 
              (∃ x : ℝ, x < 0 ∧ a/(x + 1) + 1 = (x + a)/(x - 1))) ∧
  (∀ a : ℤ, (∀ y : ℝ, ¬(y - 1 ≥ (2*y - 1)/3 ∧ -1/2*(y - a) > 0)) ∧ 
             (∃ x : ℝ, x < 0 ∧ a/(x + 1) + 1 = (x + a)/(x - 1)) → a ∈ S) ∧
  (Finset.sum S (λ a => a) = 3) := by
sorry

end sum_of_valid_a_values_l2911_291149


namespace sin_120_degrees_l2911_291188

theorem sin_120_degrees : Real.sin (2 * π / 3) = 1 / 2 := by sorry

end sin_120_degrees_l2911_291188


namespace specific_rectangle_burning_time_l2911_291159

/-- Represents a rectangular structure made of toothpicks -/
structure ToothpickRectangle where
  rows : ℕ
  columns : ℕ
  toothpicks : ℕ

/-- Represents the burning properties of toothpicks -/
structure BurningProperties where
  burn_time_per_toothpick : ℕ
  start_corners : ℕ

/-- Calculates the total burning time for a toothpick rectangle -/
def total_burning_time (rect : ToothpickRectangle) (props : BurningProperties) : ℕ :=
  sorry

/-- Theorem statement for the burning time of the specific rectangle -/
theorem specific_rectangle_burning_time :
  let rect := ToothpickRectangle.mk 3 5 38
  let props := BurningProperties.mk 10 2
  total_burning_time rect props = 65 :=
by sorry

end specific_rectangle_burning_time_l2911_291159


namespace milk_butterfat_percentage_l2911_291195

theorem milk_butterfat_percentage : 
  ∀ (initial_volume initial_percentage added_volume final_volume final_percentage : ℝ),
  initial_volume > 0 →
  added_volume > 0 →
  initial_volume + added_volume = final_volume →
  initial_volume * initial_percentage + added_volume * (added_percentage / 100) = final_volume * final_percentage →
  initial_volume = 8 →
  initial_percentage = 0.4 →
  added_volume = 16 →
  final_volume = 24 →
  final_percentage = 0.2 →
  ∃ added_percentage : ℝ, added_percentage = 10 :=
by
  sorry

#check milk_butterfat_percentage

end milk_butterfat_percentage_l2911_291195


namespace isosceles_triangle_min_ratio_l2911_291108

/-- The minimum value of (2a + b) / a for an isosceles triangle with two equal sides of length a and base of length b is 3 -/
theorem isosceles_triangle_min_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a ≥ b) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x ≥ y → (2 * x + y) / x ≥ (2 * a + b) / a ∧ (2 * a + b) / a = 3 :=
sorry

end isosceles_triangle_min_ratio_l2911_291108


namespace smallest_solution_of_equation_l2911_291114

theorem smallest_solution_of_equation (y : ℝ) : 
  (3 * y^2 + 36 * y - 90 = y * (y + 18)) → y ≥ -15 := by
  sorry

end smallest_solution_of_equation_l2911_291114


namespace max_distance_from_origin_dog_max_distance_l2911_291152

/-- The maximum distance a point on a circle can be from the origin,
    given the circle's center coordinates and radius. -/
theorem max_distance_from_origin (x y r : ℝ) : 
  let center_distance := Real.sqrt (x^2 + y^2)
  let max_distance := center_distance + r
  ∀ p : ℝ × ℝ, (p.1 - x)^2 + (p.2 - y)^2 = r^2 → 
    p.1^2 + p.2^2 ≤ max_distance^2 :=
by
  sorry

/-- The specific case for the dog problem -/
theorem dog_max_distance : 
  let x : ℝ := 6
  let y : ℝ := 8
  let r : ℝ := 15
  let center_distance := Real.sqrt (x^2 + y^2)
  let max_distance := center_distance + r
  max_distance = 25 :=
by
  sorry

end max_distance_from_origin_dog_max_distance_l2911_291152


namespace pascals_remaining_distance_l2911_291193

/-- Proves that Pascal's remaining cycling distance is 256 miles -/
theorem pascals_remaining_distance (current_speed : ℝ) (reduced_speed : ℝ) (increased_speed : ℝ)
  (h1 : current_speed = 8)
  (h2 : reduced_speed = current_speed - 4)
  (h3 : increased_speed = current_speed * 1.5)
  (h4 : ∃ (t : ℝ), current_speed * t = reduced_speed * (t + 16))
  (h5 : ∃ (t : ℝ), increased_speed * t = reduced_speed * (t + 16)) :
  ∃ (distance : ℝ), distance = 256 ∧ 
    (∃ (t : ℝ), distance = current_speed * t ∧
                distance = reduced_speed * (t + 16) ∧
                distance = increased_speed * (t - 16)) :=
sorry

end pascals_remaining_distance_l2911_291193


namespace sum_seven_consecutive_integers_multiple_of_seven_l2911_291134

theorem sum_seven_consecutive_integers_multiple_of_seven (n : ℕ+) :
  ∃ k : ℕ, n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * k := by
  sorry

end sum_seven_consecutive_integers_multiple_of_seven_l2911_291134


namespace quadratic_function_properties_l2911_291190

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x, f x = a * x^2 + b * x + c) ∧
    (∀ x, f x ≥ f (-1)) ∧
    f (-1) = -4 ∧
    f (-2) = 5

theorem quadratic_function_properties (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  (∀ x, f x = 9 * (x + 1)^2 - 4) ∧
  f 0 = 5 ∧
  f (-5/3) = 0 ∧
  f (-1/3) = 0 :=
sorry

end quadratic_function_properties_l2911_291190


namespace arithmetic_sequence_sum_l2911_291180

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 7 = 2) →
  (a 3)^2 - 2*(a 3) - 3 = 0 →
  (a 7)^2 - 2*(a 7) - 3 = 0 →
  a 1 + a 9 = 2 :=
by sorry

end arithmetic_sequence_sum_l2911_291180


namespace beijing_spirit_max_l2911_291107

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- The equation: Patriotism × Innovation × Inclusiveness + Integrity = Beijing Spirit -/
def equation (patriotism innovation inclusiveness integrity : Digit) (beijingSpirit : Nat) :=
  (patriotism.val * innovation.val * inclusiveness.val + integrity.val = beijingSpirit)

/-- All digits are different -/
def all_different (patriotism nation creation new inclusiveness tolerance integrity virtue : Digit) :=
  patriotism ≠ nation ∧ patriotism ≠ creation ∧ patriotism ≠ new ∧ patriotism ≠ inclusiveness ∧
  patriotism ≠ tolerance ∧ patriotism ≠ integrity ∧ patriotism ≠ virtue ∧
  nation ≠ creation ∧ nation ≠ new ∧ nation ≠ inclusiveness ∧ nation ≠ tolerance ∧
  nation ≠ integrity ∧ nation ≠ virtue ∧ creation ≠ new ∧ creation ≠ inclusiveness ∧
  creation ≠ tolerance ∧ creation ≠ integrity ∧ creation ≠ virtue ∧ new ≠ inclusiveness ∧
  new ≠ tolerance ∧ new ≠ integrity ∧ new ≠ virtue ∧ inclusiveness ≠ tolerance ∧
  inclusiveness ≠ integrity ∧ inclusiveness ≠ virtue ∧ tolerance ≠ integrity ∧ tolerance ≠ virtue ∧
  integrity ≠ virtue

theorem beijing_spirit_max (patriotism nation creation new inclusiveness tolerance integrity virtue : Digit) :
  all_different patriotism nation creation new inclusiveness tolerance integrity virtue →
  equation patriotism creation inclusiveness integrity 9898 →
  integrity.val = 98 := by
  sorry

end beijing_spirit_max_l2911_291107


namespace distance_for_given_point_l2911_291132

/-- The distance between a point and its symmetric point about the x-axis --/
def distance_to_symmetric_point (x y : ℝ) : ℝ := 2 * |y|

/-- Theorem: The distance between (2, -3) and its symmetric point about the x-axis is 6 --/
theorem distance_for_given_point : distance_to_symmetric_point 2 (-3) = 6 := by
  sorry

end distance_for_given_point_l2911_291132


namespace card_distribution_convergence_l2911_291100

/-- Represents a person in the circular arrangement -/
structure Person where
  id : Nat
  cards : Nat

/-- Represents the state of the card distribution -/
structure CardState where
  people : List Person
  total_cards : Nat

/-- Defines a valid move in the card game -/
def valid_move (state : CardState) (giver : Nat) : Prop :=
  ∃ (p : Person), p ∈ state.people ∧ p.id = giver ∧ p.cards ≥ 2

/-- Defines the result of a move -/
def move_result (state : CardState) (giver : Nat) : CardState :=
  sorry

/-- Defines a sequence of moves -/
def move_sequence (initial : CardState) : List Nat → CardState
  | [] => initial
  | (m :: ms) => move_result (move_sequence initial ms) m

/-- The main theorem to be proved -/
theorem card_distribution_convergence 
  (n : Nat) 
  (h : n > 1) :
  ∃ (initial : CardState) (moves : List Nat),
    (initial.people.length = n) ∧ 
    (initial.total_cards = n - 1) ∧
    (∀ (p : Person), p ∈ (move_sequence initial moves).people → p.cards ≤ 1) :=
  sorry

end card_distribution_convergence_l2911_291100


namespace fibonacci_matrix_power_fibonacci_determinant_l2911_291121

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n+2 => fibonacci (n+1) + fibonacci n

def fibonacci_matrix (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ := 
  ![![fibonacci (n+1), fibonacci n],
    ![fibonacci n, fibonacci (n-1)]]

theorem fibonacci_matrix_power (n : ℕ) :
  (Matrix.of ![![1, 1], ![1, 0]] : Matrix (Fin 2) (Fin 2) ℕ) ^ n = fibonacci_matrix n := by
  sorry

theorem fibonacci_determinant (n : ℕ) :
  fibonacci (n+1) * fibonacci (n-1) - fibonacci n ^ 2 = (-1 : ℤ) ^ n := by
  sorry

end fibonacci_matrix_power_fibonacci_determinant_l2911_291121


namespace first_puncture_time_l2911_291112

/-- Given a tyre with two punctures, this theorem proves the time it takes
    for the first puncture alone to flatten the tyre. -/
theorem first_puncture_time
  (second_puncture_time : ℝ)
  (both_punctures_time : ℝ)
  (h1 : second_puncture_time = 6)
  (h2 : both_punctures_time = 336 / 60)
  (h3 : both_punctures_time > 0) :
  ∃ (first_puncture_time : ℝ),
    first_puncture_time > 0 ∧
    1 / first_puncture_time + 1 / second_puncture_time = 1 / both_punctures_time ∧
    first_puncture_time = 84 := by
  sorry

end first_puncture_time_l2911_291112


namespace eighth_term_of_sequence_l2911_291171

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem eighth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 4/3) (h₃ : a₃ = 7/6) :
  arithmetic_sequence a₁ ((a₂ - a₁) : ℚ) 8 = 19/3 := by
sorry

end eighth_term_of_sequence_l2911_291171


namespace bits_required_for_ABC12_l2911_291178

-- Define the hexadecimal number ABC12₁₆
def hex_number : ℕ := 0xABC12

-- Theorem stating that the number of bits required to represent ABC12₁₆ is 20
theorem bits_required_for_ABC12 :
  (Nat.log 2 hex_number).succ = 20 := by sorry

end bits_required_for_ABC12_l2911_291178


namespace tangent_line_to_x_ln_x_l2911_291140

/-- A line y = 2x + m is tangent to the curve y = x ln x if and only if m = -e -/
theorem tangent_line_to_x_ln_x (m : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (2 * x₀ + m = x₀ * Real.log x₀) ∧ 
    (2 = Real.log x₀ + 1)) ↔ 
  m = -Real.exp 1 := by
sorry

end tangent_line_to_x_ln_x_l2911_291140


namespace max_value_of_a_max_value_is_negative_two_l2911_291113

theorem max_value_of_a : ∀ a : ℝ, 
  (∀ x : ℝ, x < a → x^2 - x - 6 > 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 > 0 ∧ x ≥ a) →
  a ≤ -2 :=
by sorry

theorem max_value_is_negative_two : 
  ∃ a : ℝ, a = -2 ∧
  (∀ x : ℝ, x < a → x^2 - x - 6 > 0) ∧
  (∃ x : ℝ, x^2 - x - 6 > 0 ∧ x ≥ a) ∧
  (∀ b : ℝ, b > a →
    (∃ x : ℝ, x < b ∧ x^2 - x - 6 ≤ 0) ∨
    (∀ x : ℝ, x^2 - x - 6 > 0 → x < b)) :=
by sorry

end max_value_of_a_max_value_is_negative_two_l2911_291113


namespace new_members_weight_combined_weight_proof_l2911_291164

/-- Calculates the combined weight of new members in a group replacement scenario. -/
theorem new_members_weight (group_size : ℕ) (original_avg : ℝ) (new_avg : ℝ)
  (replaced_weights : List ℝ) : ℝ :=
  let total_original := group_size * original_avg
  let total_replaced := replaced_weights.sum
  let remaining_weight := total_original - total_replaced
  let new_total := group_size * new_avg
  new_total - remaining_weight

/-- Proves that the combined weight of new members is 238 kg in the given scenario. -/
theorem combined_weight_proof :
  new_members_weight 8 70 76 [50, 65, 75] = 238 := by
  sorry

end new_members_weight_combined_weight_proof_l2911_291164


namespace arithmetic_sequence_property_l2911_291158

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumCondition a) : 
  3 * a 9 - a 11 = 48 := by
  sorry

end arithmetic_sequence_property_l2911_291158


namespace hyperbola_asymptote_slope_l2911_291123

theorem hyperbola_asymptote_slope (x y m : ℝ) : 
  (((x^2 : ℝ) / 49) - ((y^2 : ℝ) / 36) = 1) →
  (∃ (k : ℝ), y = k * m * x ∧ y = -k * m * x) →
  (m > 0) →
  (m = 6/7) := by
sorry

end hyperbola_asymptote_slope_l2911_291123


namespace simplify_and_evaluate_l2911_291105

theorem simplify_and_evaluate : (Real.sqrt 2 + 1)^2 - 2*(Real.sqrt 2 + 1) = 1 := by
  sorry

end simplify_and_evaluate_l2911_291105


namespace probability_sum_six_l2911_291122

-- Define a die as having 6 faces
def die : ℕ := 6

-- Define the favorable outcomes (combinations that sum to 6)
def favorable_outcomes : ℕ := 5

-- Define the total number of possible outcomes
def total_outcomes : ℕ := die * die

-- State the theorem
theorem probability_sum_six (d : ℕ) (h : d = die) : 
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 := by
  sorry

end probability_sum_six_l2911_291122
