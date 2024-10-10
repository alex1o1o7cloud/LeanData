import Mathlib

namespace incorrect_proportion_l3190_319002

theorem incorrect_proportion (a b m n : ℝ) (h : a * b = m * n) :
  ¬(m / a = n / b) := by
  sorry

end incorrect_proportion_l3190_319002


namespace distance_between_chord_endpoints_l3190_319076

/-- In a circle with radius R, given two mutually perpendicular chords MN and PQ,
    where NQ = m, the distance between points M and P is √(4R² - m²). -/
theorem distance_between_chord_endpoints (R m : ℝ) (R_pos : R > 0) (m_pos : m > 0) :
  ∃ (M P : ℝ × ℝ),
    (∃ (N Q : ℝ × ℝ),
      (∀ (X : ℝ × ℝ), (X.1 - 0)^2 + (X.2 - 0)^2 = R^2 → 
        ((M.1 - N.1) * (P.1 - Q.1) + (M.2 - N.2) * (P.2 - Q.2) = 0) ∧
        ((N.1 - Q.1)^2 + (N.2 - Q.2)^2 = m^2)) →
      ((M.1 - P.1)^2 + (M.2 - P.2)^2 = 4 * R^2 - m^2)) :=
sorry

end distance_between_chord_endpoints_l3190_319076


namespace smallest_integer_in_set_l3190_319089

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def is_consecutive_odd_set (s : Set ℤ) : Prop :=
  ∃ a b : ℤ, a ≤ b ∧ s = {x | a ≤ x ∧ x ≤ b ∧ is_odd x ∧ ∀ y, a ≤ y ∧ y < x → is_odd y}

def median (s : Set ℤ) : ℤ := sorry

theorem smallest_integer_in_set (s : Set ℤ) :
  is_consecutive_odd_set s ∧ median s = 153 ∧ (∃ x ∈ s, ∀ y ∈ s, y ≤ x) ∧ 167 ∈ s →
  (∃ z ∈ s, ∀ w ∈ s, z ≤ w) ∧ 139 ∈ s :=
by sorry

end smallest_integer_in_set_l3190_319089


namespace quadratic_real_roots_l3190_319062

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x + k^2 - 1 = 0) ↔ k ∈ Set.Icc (-2*Real.sqrt 3/3) (2*Real.sqrt 3/3) :=
by sorry

end quadratic_real_roots_l3190_319062


namespace power_of_three_divides_a_l3190_319054

def a : ℕ → ℤ
  | 0 => 3
  | n + 1 => (3 * a n ^ 2 + 1) / 2 - a n

theorem power_of_three_divides_a (k : ℕ) : 
  (3 ^ (k + 1) : ℤ) ∣ a (3 ^ k) := by sorry

end power_of_three_divides_a_l3190_319054


namespace problem_statement_l3190_319013

theorem problem_statement (x y z : ℝ) 
  (h1 : (1/x) + (2/y) + (3/z) = 0)
  (h2 : (1/x) - (6/y) - (5/z) = 0) :
  (x/y) + (y/z) + (z/x) = -1 := by
  sorry

end problem_statement_l3190_319013


namespace loggerhead_turtle_eggs_per_nest_l3190_319097

/-- The average number of eggs per nest for loggerhead turtles -/
def average_eggs_per_nest (total_eggs : ℕ) (total_nests : ℕ) : ℚ :=
  total_eggs / total_nests

/-- Theorem: The average number of eggs per nest is 150 -/
theorem loggerhead_turtle_eggs_per_nest :
  average_eggs_per_nest 3000000 20000 = 150 := by
  sorry

end loggerhead_turtle_eggs_per_nest_l3190_319097


namespace stratified_sampling_size_l3190_319085

theorem stratified_sampling_size (high_school_students junior_high_students : ℕ) 
  (high_school_sample : ℕ) (total_sample : ℕ) : 
  high_school_students = 3500 →
  junior_high_students = 1500 →
  high_school_sample = 70 →
  (high_school_sample : ℚ) / high_school_students = 
    (total_sample : ℚ) / (high_school_students + junior_high_students) →
  total_sample = 100 := by
sorry

end stratified_sampling_size_l3190_319085


namespace brett_red_marbles_l3190_319035

/-- The number of red marbles Brett has -/
def red_marbles : ℕ := sorry

/-- The number of blue marbles Brett has -/
def blue_marbles : ℕ := sorry

/-- Brett has 24 more blue marbles than red marbles -/
axiom more_blue : blue_marbles = red_marbles + 24

/-- Brett has 5 times as many blue marbles as red marbles -/
axiom five_times : blue_marbles = 5 * red_marbles

theorem brett_red_marbles : red_marbles = 6 := by sorry

end brett_red_marbles_l3190_319035


namespace optimal_portfolio_l3190_319086

/-- Represents an investment project with maximum profit and loss percentages -/
structure Project where
  max_profit : Real
  max_loss : Real

/-- Represents an investment portfolio with amounts invested in two projects -/
structure Portfolio where
  amount_a : Real
  amount_b : Real

def project_a : Project := { max_profit := 1.0, max_loss := 0.3 }
def project_b : Project := { max_profit := 0.5, max_loss := 0.1 }

def total_investment_limit : Real := 100000
def max_allowed_loss : Real := 18000

def portfolio_loss (p : Portfolio) : Real :=
  p.amount_a * project_a.max_loss + p.amount_b * project_b.max_loss

def portfolio_profit (p : Portfolio) : Real :=
  p.amount_a * project_a.max_profit + p.amount_b * project_b.max_profit

def is_valid_portfolio (p : Portfolio) : Prop :=
  p.amount_a ≥ 0 ∧ p.amount_b ≥ 0 ∧
  p.amount_a + p.amount_b ≤ total_investment_limit ∧
  portfolio_loss p ≤ max_allowed_loss

theorem optimal_portfolio :
  ∃ (p : Portfolio), is_valid_portfolio p ∧
    ∀ (q : Portfolio), is_valid_portfolio q → portfolio_profit q ≤ portfolio_profit p :=
  sorry

end optimal_portfolio_l3190_319086


namespace b_value_in_discriminant_l3190_319077

/-- For a quadratic equation ax^2 + bx + c = 0, 
    the discriminant is defined as b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation x^2 - 2x - 3 = 0 -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

theorem b_value_in_discriminant :
  ∃ (a b c : ℝ), 
    (∀ x, quadratic_equation x ↔ a*x^2 + b*x + c = 0) ∧
    b = -2 :=
sorry

end b_value_in_discriminant_l3190_319077


namespace smallest_prime_divisor_of_sum_l3190_319056

theorem smallest_prime_divisor_of_sum (p : Nat) : 
  Prime p ∧ p ∣ (2^14 + 7^12) ∧ ∀ q, Prime q → q ∣ (2^14 + 7^12) → p ≤ q → p = 5 := by
  sorry

end smallest_prime_divisor_of_sum_l3190_319056


namespace unique_number_exists_l3190_319046

theorem unique_number_exists : ∃! N : ℕ, 
  (∃ Q : ℕ, N = 11 * Q) ∧ 
  (N / 11 + N + 11 = 71) := by
sorry

end unique_number_exists_l3190_319046


namespace sequence_difference_l3190_319084

theorem sequence_difference (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, n > 0 → S n = n^2 + 2*n) →
  (∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1)) →
  a 4 - a 2 = 4 := by
sorry

end sequence_difference_l3190_319084


namespace inequality_of_distinct_positives_l3190_319063

theorem inequality_of_distinct_positives (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_distinct_ab : a ≠ b) (h_distinct_ac : a ≠ c) (h_distinct_bc : b ≠ c) :
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 := by
  sorry

end inequality_of_distinct_positives_l3190_319063


namespace remainder_congruence_l3190_319079

theorem remainder_congruence (x : ℤ) 
  (h1 : (2 + x) % 8 = 9 % 8)
  (h2 : (3 + x) % 27 = 4 % 27)
  (h3 : (11 + x) % 1331 = 49 % 1331) :
  x % 198 = 1 := by
  sorry

end remainder_congruence_l3190_319079


namespace range_of_k_l3190_319022

theorem range_of_k (k : ℝ) : 
  (k ≠ 0) → 
  (k^2 * 1^2 - 6*k*1 + 8 ≥ 0) → 
  ((k ≥ 4) ∨ (k ≤ 2)) := by
  sorry

end range_of_k_l3190_319022


namespace basketball_team_size_l3190_319067

theorem basketball_team_size 
  (total_score : ℕ) 
  (min_score : ℕ) 
  (max_score : ℕ) 
  (h1 : total_score = 100) 
  (h2 : min_score = 7) 
  (h3 : max_score = 23) :
  ∃ (team_size : ℕ), 
    team_size * min_score ≤ total_score ∧ 
    total_score ≤ (team_size - 1) * min_score + max_score ∧
    team_size = 12 :=
by
  sorry

end basketball_team_size_l3190_319067


namespace integer_roots_of_polynomial_l3190_319012

/-- A polynomial with integer coefficients of the form x^3 + b₂x^2 + b₁x + 18 = 0 -/
def IntPolynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ :=
  x^3 + b₂ * x^2 + b₁ * x + 18

/-- The set of all possible integer roots of the polynomial -/
def PossibleRoots : Set ℤ :=
  {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  ∀ x : ℤ, IntPolynomial b₂ b₁ x = 0 → x ∈ PossibleRoots :=
sorry

end integer_roots_of_polynomial_l3190_319012


namespace caravan_hens_count_l3190_319047

/-- A caravan with hens, goats, camels, and keepers. -/
structure Caravan where
  hens : ℕ
  goats : ℕ
  camels : ℕ
  keepers : ℕ

/-- Calculate the total number of feet in the caravan. -/
def totalFeet (c : Caravan) : ℕ :=
  2 * c.hens + 4 * c.goats + 4 * c.camels + 2 * c.keepers

/-- Calculate the total number of heads in the caravan. -/
def totalHeads (c : Caravan) : ℕ :=
  c.hens + c.goats + c.camels + c.keepers

/-- The main theorem stating the number of hens in the caravan. -/
theorem caravan_hens_count : ∃ (c : Caravan), 
  c.goats = 45 ∧ 
  c.camels = 8 ∧ 
  c.keepers = 15 ∧ 
  totalFeet c = totalHeads c + 224 ∧ 
  c.hens = 50 := by
  sorry


end caravan_hens_count_l3190_319047


namespace polygon_with_540_degree_sum_is_pentagon_l3190_319096

theorem polygon_with_540_degree_sum_is_pentagon (n : ℕ) (h : n ≥ 3) :
  (n - 2) * 180 = 540 → n = 5 := by
  sorry

end polygon_with_540_degree_sum_is_pentagon_l3190_319096


namespace mark_piggy_bank_problem_l3190_319036

/-- Given a total amount of money and a total number of bills (one and two dollar bills only),
    calculate the number of one dollar bills. -/
def one_dollar_bills (total_money : ℕ) (total_bills : ℕ) : ℕ :=
  total_bills - (total_money - total_bills)

/-- Theorem stating that given 87 dollars in total and 58 bills,
    the number of one dollar bills is 29. -/
theorem mark_piggy_bank_problem :
  one_dollar_bills 87 58 = 29 := by
  sorry

end mark_piggy_bank_problem_l3190_319036


namespace co_molecular_weight_l3190_319019

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (carbon_count oxygen_count : ℕ) : ℝ :=
  carbon_count * carbon_weight + oxygen_count * oxygen_weight

/-- Theorem: The molecular weight of CO is 28.01 g/mol -/
theorem co_molecular_weight :
  molecular_weight 1 1 = 28.01 := by sorry

end co_molecular_weight_l3190_319019


namespace double_price_increase_rate_l3190_319016

/-- The rate of price increase that, when applied twice, doubles the original price -/
theorem double_price_increase_rate : 
  ∃ x : ℝ, (1 + x) * (1 + x) = 2 ∧ x > 0 :=
by sorry

end double_price_increase_rate_l3190_319016


namespace annual_output_scientific_notation_l3190_319037

/-- The annual output of the photovoltaic power station in kWh -/
def annual_output : ℝ := 448000

/-- The scientific notation representation of the annual output -/
def scientific_notation : ℝ := 4.48 * (10 ^ 5)

theorem annual_output_scientific_notation : annual_output = scientific_notation := by
  sorry

end annual_output_scientific_notation_l3190_319037


namespace distance_to_soccer_is_12_l3190_319055

-- Define the distances and costs
def distance_to_grocery : ℝ := 8
def distance_to_school : ℝ := 6
def miles_per_gallon : ℝ := 25
def cost_per_gallon : ℝ := 2.5
def total_gas_cost : ℝ := 5

-- Define the unknown distance to soccer practice
def distance_to_soccer : ℝ → ℝ := λ x => x

-- Define the total distance driven
def total_distance (x : ℝ) : ℝ :=
  distance_to_grocery + distance_to_school + distance_to_soccer x + 2 * distance_to_soccer x

-- Theorem stating that the distance to soccer practice is 12 miles
theorem distance_to_soccer_is_12 :
  ∃ x : ℝ, distance_to_soccer x = 12 ∧ 
    total_distance x = (total_gas_cost / cost_per_gallon) * miles_per_gallon := by
  sorry

end distance_to_soccer_is_12_l3190_319055


namespace island_ratio_l3190_319065

theorem island_ratio (centipedes humans sheep : ℕ) : 
  centipedes = 2 * humans →
  centipedes = 100 →
  sheep + humans = 75 →
  sheep.gcd humans = 25 →
  (sheep / 25 : ℚ) / (humans / 25 : ℚ) = 1 / 2 :=
by sorry

end island_ratio_l3190_319065


namespace simplify_fraction_l3190_319092

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = (5 * Real.sqrt 2) / 28 := by
  sorry

end simplify_fraction_l3190_319092


namespace number_equation_l3190_319099

theorem number_equation : ∃ x : ℝ, x * (37 - 15) - 25 = 327 :=
by
  sorry

end number_equation_l3190_319099


namespace sin_2alpha_equals_one_minus_p_squared_l3190_319005

theorem sin_2alpha_equals_one_minus_p_squared (α : ℝ) (p : ℝ) 
  (h : Real.sin α - Real.cos α = p) : 
  Real.sin (2 * α) = 1 - p^2 := by
  sorry

end sin_2alpha_equals_one_minus_p_squared_l3190_319005


namespace sequence_expression_l3190_319023

theorem sequence_expression (a : ℕ → ℕ) :
  a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) - 2 * a n = 2^n) →
  ∀ n : ℕ, n ≥ 1 → a n = n * 2^(n - 1) :=
by sorry

end sequence_expression_l3190_319023


namespace equal_selection_probability_l3190_319045

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents the probability of an individual being selected in a sampling method -/
def selectionProbability (method : SamplingMethod) (individual : ℕ) : ℝ := sorry

/-- The theorem stating that all three sampling methods have equal selection probability for all individuals -/
theorem equal_selection_probability (population : Finset ℕ) :
  ∀ (method : SamplingMethod) (i j : ℕ), i ∈ population → j ∈ population →
    selectionProbability method i = selectionProbability method j :=
  sorry

end equal_selection_probability_l3190_319045


namespace total_football_games_l3190_319059

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := 9

/-- Theorem: The total number of football games Joan went to is 13 -/
theorem total_football_games : games_this_year + games_last_year = 13 := by
  sorry

end total_football_games_l3190_319059


namespace q_minimized_at_2_l3190_319066

/-- The quadratic function q in terms of x -/
def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 6

/-- The value of x that minimizes q -/
def minimizing_x : ℝ := 2

theorem q_minimized_at_2 :
  ∀ x : ℝ, q x ≥ q minimizing_x :=
sorry

end q_minimized_at_2_l3190_319066


namespace tan_equality_in_range_l3190_319091

theorem tan_equality_in_range : ∃ (n : ℤ), -150 < n ∧ n < 150 ∧ Real.tan (n * π / 180) = Real.tan (286 * π / 180) ∧ n = -74 := by
  sorry

end tan_equality_in_range_l3190_319091


namespace quadratic_real_roots_l3190_319071

theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + (a+1)^2 = 0) ↔ (a ∈ Set.Icc (-2) (-2/3) ∧ a ≠ -1) :=
by sorry

end quadratic_real_roots_l3190_319071


namespace painting_area_is_1836_l3190_319051

/-- The area of a rectangular painting within a frame -/
def painting_area (frame_width outer_length outer_width : ℝ) : ℝ :=
  (outer_length - 2 * frame_width) * (outer_width - 2 * frame_width)

/-- Theorem: The area of the painting is 1836 cm² -/
theorem painting_area_is_1836 :
  painting_area 8 70 50 = 1836 := by
  sorry

end painting_area_is_1836_l3190_319051


namespace kindergarten_sample_size_l3190_319042

/-- Represents a kindergarten with students and a height measurement sample -/
structure Kindergarten where
  total_students : ℕ
  sample_size : ℕ

/-- Defines the sample size of a kindergarten height measurement -/
def sample_size (k : Kindergarten) : ℕ := k.sample_size

/-- Theorem: The sample size of the kindergarten height measurement is 31 -/
theorem kindergarten_sample_size :
  ∀ (k : Kindergarten),
  k.total_students = 310 →
  k.sample_size = 31 →
  sample_size k = 31 := by
  sorry

end kindergarten_sample_size_l3190_319042


namespace least_multiple_remainder_l3190_319053

theorem least_multiple_remainder (m : ℕ) : 
  (m % 23 = 0) → 
  (m % 1821 = 710) → 
  (m = 3024) → 
  (m % 24 = 0) := by
sorry

end least_multiple_remainder_l3190_319053


namespace eleven_pictures_left_to_color_l3190_319034

/-- The number of pictures left to color given two coloring books and some already colored pictures. -/
def pictures_left_to_color (book1_pictures book2_pictures colored_pictures : ℕ) : ℕ :=
  book1_pictures + book2_pictures - colored_pictures

/-- Theorem stating that given the specific numbers in the problem, 11 pictures are left to color. -/
theorem eleven_pictures_left_to_color :
  pictures_left_to_color 23 32 44 = 11 := by
  sorry

end eleven_pictures_left_to_color_l3190_319034


namespace factorization_problems_l3190_319010

theorem factorization_problems :
  (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = (2*x + 3*y) * (2*x - 3*y)) ∧
  (∀ a b : ℝ, -16 * a^2 + 25 * b^2 = (5*b + 4*a) * (5*b - 4*a)) ∧
  (∀ x y : ℝ, x^3 * y - x * y^3 = x * y * (x + y) * (x - y)) :=
by sorry

end factorization_problems_l3190_319010


namespace arithmetic_sequences_equal_sum_l3190_319024

/-- Sum of first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ d n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequences_equal_sum :
  ∃! (n : ℕ), n > 0 ∧ arithmetic_sum 5 5 n = arithmetic_sum 22 3 n :=
by
  sorry

end arithmetic_sequences_equal_sum_l3190_319024


namespace polynomial_equality_l3190_319026

theorem polynomial_equality (a b : ℝ) : 
  (∀ x : ℝ, (x - 2) * (x + 3) = x^2 + a*x + b) → (a = 1 ∧ b = -6) := by
  sorry

end polynomial_equality_l3190_319026


namespace journey_speed_calculation_l3190_319015

/-- Given a journey with the following properties:
  * Total distance is 112 km
  * Total time is 5 hours
  * The first half is traveled at 21 km/hr
  Prove that the speed for the second half is 24 km/hr -/
theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ)
  (h1 : total_distance = 112)
  (h2 : total_time = 5)
  (h3 : first_half_speed = 21)
  : (2 * total_distance) / (2 * total_time - total_distance / first_half_speed) = 24 :=
by sorry

end journey_speed_calculation_l3190_319015


namespace largest_multiple_of_15_less_than_neg_150_l3190_319094

theorem largest_multiple_of_15_less_than_neg_150 :
  ∀ n : ℤ, n * 15 < -150 → n * 15 ≤ -165 :=
by
  sorry

end largest_multiple_of_15_less_than_neg_150_l3190_319094


namespace product_18396_9999_l3190_319068

theorem product_18396_9999 : 18396 * 9999 = 183962604 := by sorry

end product_18396_9999_l3190_319068


namespace gummy_bear_distribution_l3190_319040

theorem gummy_bear_distribution (initial_candies : ℕ) (num_siblings : ℕ) (josh_eat : ℕ) (leftover : ℕ) :
  initial_candies = 100 →
  num_siblings = 3 →
  josh_eat = 16 →
  leftover = 19 →
  ∃ (sibling_candies : ℕ),
    sibling_candies * num_siblings + 2 * (josh_eat + leftover) = initial_candies ∧
    sibling_candies = 10 :=
by sorry

end gummy_bear_distribution_l3190_319040


namespace inscribed_rectangles_area_sum_l3190_319038

/-- A structure representing a rectangle --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- A structure representing two inscribed rectangles sharing a common vertex --/
structure InscribedRectangles where
  outer : Rectangle
  common_vertex : ℝ  -- Position of K on AB, 0 ≤ common_vertex ≤ outer.width

/-- Calculate the area of a rectangle --/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculate the sum of areas of the two inscribed rectangles --/
def InscribedRectangles.sumOfAreas (ir : InscribedRectangles) : ℝ :=
  ir.common_vertex * ir.outer.height

/-- Theorem stating that the sum of areas of inscribed rectangles equals the area of the outer rectangle --/
theorem inscribed_rectangles_area_sum (ir : InscribedRectangles) :
  ir.sumOfAreas = ir.outer.area := by sorry

end inscribed_rectangles_area_sum_l3190_319038


namespace max_distinct_sums_diffs_is_64_l3190_319028

/-- Given a set of five natural numbers including 100, 200, and 400,
    this function returns the maximum number of distinct non-zero natural numbers
    that can be obtained by performing addition and subtraction operations,
    where each number is used at most once in each expression
    and at least two numbers are used. -/
def max_distinct_sums_diffs (a b : ℕ) : ℕ :=
  64

/-- Theorem stating that the maximum number of distinct non-zero natural numbers
    obtainable from the given set of numbers under the specified conditions is 64. -/
theorem max_distinct_sums_diffs_is_64 (a b : ℕ) :
  max_distinct_sums_diffs a b = 64 := by
  sorry

end max_distinct_sums_diffs_is_64_l3190_319028


namespace power_product_equals_128_l3190_319044

theorem power_product_equals_128 (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end power_product_equals_128_l3190_319044


namespace only_thirteen_fourths_between_three_and_four_l3190_319049

theorem only_thirteen_fourths_between_three_and_four :
  let numbers : List ℚ := [5/2, 11/4, 11/5, 13/4, 13/5]
  ∀ x ∈ numbers, (3 < x ∧ x < 4) ↔ x = 13/4 := by
  sorry

end only_thirteen_fourths_between_three_and_four_l3190_319049


namespace solve_system_l3190_319000

theorem solve_system (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 18) 
  (eq2 : 2 * x + y = 21) : 
  y = 27 / 11 := by
  sorry

end solve_system_l3190_319000


namespace vessel_base_length_l3190_319090

/-- The length of the base of a vessel given specific conditions -/
theorem vessel_base_length : ∀ (breadth rise cube_edge : ℝ),
  breadth = 30 →
  rise = 15 →
  cube_edge = 30 →
  (cube_edge ^ 3) = breadth * rise * 60 :=
by
  sorry

end vessel_base_length_l3190_319090


namespace largest_base_5_to_base_7_l3190_319075

/-- The largest four-digit number in base-5 -/
def m : ℕ := 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

/-- Conversion of a natural number to its base-7 representation -/
def to_base_7 (n : ℕ) : List ℕ :=
  sorry

theorem largest_base_5_to_base_7 :
  to_base_7 m = [1, 5, 5, 1] :=
sorry

end largest_base_5_to_base_7_l3190_319075


namespace fifth_largest_divisor_of_2014000000_l3190_319032

theorem fifth_largest_divisor_of_2014000000 :
  ∃ (d : ℕ), d ∣ 2014000000 ∧
  (∀ (x : ℕ), x ∣ 2014000000 → x ≠ 2014000000 → x ≠ 1007000000 → x ≠ 503500000 → x ≠ 251750000 → x ≤ d) ∧
  d = 125875000 :=
by sorry

end fifth_largest_divisor_of_2014000000_l3190_319032


namespace arithmetic_progression_rth_term_l3190_319008

/-- Given an arithmetic progression where the sum of n terms is 5n + 4n^2,
    prove that the r-th term is 8r + 1 -/
theorem arithmetic_progression_rth_term (n : ℕ) (r : ℕ) :
  (∀ n, ∃ S : ℕ → ℕ, S n = 5*n + 4*n^2) →
  ∃ a : ℕ → ℕ, a r = 8*r + 1 :=
by sorry

end arithmetic_progression_rth_term_l3190_319008


namespace N_composite_and_three_factors_l3190_319087

def N (n : ℕ) : ℤ := n^4 - 90*n^2 - 91*n - 90

theorem N_composite_and_three_factors (n : ℕ) (h : n > 10) :
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N n = a * b) ∧
  (∃ (x y z : ℕ), x > 1 ∧ y > 1 ∧ z > 1 ∧ N n = x * y * z) :=
sorry

end N_composite_and_three_factors_l3190_319087


namespace sequence_properties_l3190_319001

/-- Sequence a_n with given properties -/
def sequence_a (n : ℕ) : ℝ :=
  sorry

/-- Sum of first n terms of sequence a_n -/
def S (n : ℕ) : ℝ :=
  sorry

/-- Sum of first n terms of sequence a_n / 2^n -/
def T (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating the properties of the sequence and its sums -/
theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_a n / S n = 2 / (n + 1)) ∧
  sequence_a 1 = 1 →
  (∀ n : ℕ, n ≥ 1 → sequence_a n = n) ∧
  (∀ n : ℕ, n ≥ 1 → T n = 2 - (n + 2) * (1/2)^n) :=
by
  sorry

end sequence_properties_l3190_319001


namespace smallest_integer_with_remainders_l3190_319027

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 3 = 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 8 = 1) ∧ 
  (n % 7 = 2) ∧ 
  (∀ m : ℕ, m > 1 → m % 3 = 1 → m % 5 = 1 → m % 8 = 1 → m % 7 = 2 → m ≥ n) ∧
  n = 481 := by
  sorry

end smallest_integer_with_remainders_l3190_319027


namespace intersection_of_sets_l3190_319098

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define set A
def SetA (x y m : ℝ) : Prop := (m + 3) * x + (m - 2) * y - 1 - 2 * m = 0

-- Define set B (tangent lines to the circle)
def SetB (x y : ℝ) : Prop := ∃ (a b : ℝ), Circle a b ∧ (x - a) * a + (y - b) * b = 0

-- Define the intersection set
def IntersectionSet (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem intersection_of_sets :
  ∀ (x y : ℝ), (∃ (m : ℝ), SetA x y m) ∧ SetB x y ↔ IntersectionSet x y :=
sorry

end intersection_of_sets_l3190_319098


namespace rectangular_plot_width_l3190_319078

theorem rectangular_plot_width
  (length : ℝ)
  (num_poles : ℕ)
  (pole_spacing : ℝ)
  (h1 : length = 90)
  (h2 : num_poles = 14)
  (h3 : pole_spacing = 20)
  : ∃ width : ℝ, width = 40 ∧ 2 * (length + width) = (num_poles - 1 : ℝ) * pole_spacing :=
by sorry

end rectangular_plot_width_l3190_319078


namespace inequalities_theorem_l3190_319073

variables (a b c x y z : ℝ)

def M : ℝ := a * x + b * y + c * z
def N : ℝ := a * z + b * y + c * x
def P : ℝ := a * y + b * z + c * x
def Q : ℝ := a * z + b * x + c * y

theorem inequalities_theorem (h1 : a > b) (h2 : b > c) (h3 : x > y) (h4 : y > z) :
  M a b c x y z > P a b c x y z ∧ 
  P a b c x y z > N a b c x y z ∧ 
  M a b c x y z > Q a b c x y z ∧ 
  Q a b c x y z > N a b c x y z :=
by sorry

end inequalities_theorem_l3190_319073


namespace celine_erasers_l3190_319003

/-- The number of erasers collected by each person -/
structure EraserCollection where
  gabriel : ℕ
  celine : ℕ
  julian : ℕ
  erica : ℕ

/-- The conditions of the eraser collection problem -/
def EraserProblem (ec : EraserCollection) : Prop :=
  ec.celine = 2 * ec.gabriel ∧
  ec.julian = 2 * ec.celine ∧
  ec.erica = 3 * ec.julian ∧
  ec.gabriel + ec.celine + ec.julian + ec.erica = 151

theorem celine_erasers (ec : EraserCollection) (h : EraserProblem ec) : ec.celine = 16 := by
  sorry

end celine_erasers_l3190_319003


namespace exterior_angle_measure_l3190_319070

/-- The degree measure of an interior angle of a regular n-gon -/
def interior_angle (n : ℕ) : ℚ := 180 * (n - 2) / n

theorem exterior_angle_measure :
  let square_angle : ℚ := 90
  let heptagon_angle : ℚ := interior_angle 7
  let exterior_angle : ℚ := 360 - heptagon_angle - square_angle
  exterior_angle = 990 / 7 := by sorry

end exterior_angle_measure_l3190_319070


namespace fraction_sum_rounded_l3190_319093

theorem fraction_sum_rounded : 
  let sum := (3 : ℚ) / 20 + 7 / 200 + 8 / 2000 + 3 / 20000
  round (sum * 10000) / 10000 = (1892 : ℚ) / 10000 := by
  sorry

end fraction_sum_rounded_l3190_319093


namespace jonessa_take_home_pay_l3190_319004

/-- Given Jonessa's pay and tax rate, calculate her take-home pay -/
theorem jonessa_take_home_pay (total_pay : ℝ) (tax_rate : ℝ) 
  (h1 : total_pay = 500)
  (h2 : tax_rate = 0.1) : 
  total_pay * (1 - tax_rate) = 450 := by
sorry

end jonessa_take_home_pay_l3190_319004


namespace closest_point_l3190_319030

/-- The vector v as a function of t -/
def v (t : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 1 + 5*t
  | 1 => -2 + 4*t
  | 2 => -4 - 2*t

/-- The vector a -/
def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3
  | 1 => 2
  | 2 => 6

/-- The direction vector of v -/
def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 4
  | 2 => -2

/-- Theorem: The value of t that minimizes the distance between v and a is 2/15 -/
theorem closest_point : 
  (∀ t : ℝ, (v t - a) • direction = 0 → t = 2/15) ∧ 
  (v (2/15) - a) • direction = 0 := by
  sorry

end closest_point_l3190_319030


namespace trigonometric_identities_l3190_319018

theorem trigonometric_identities :
  (Real.cos (75 * π / 180))^2 = (2 - Real.sqrt 3) / 4 ∧
  Real.tan (1 * π / 180) + Real.tan (44 * π / 180) + Real.tan (1 * π / 180) * Real.tan (44 * π / 180) = 1 := by
  sorry

end trigonometric_identities_l3190_319018


namespace solution_set_f_range_of_k_l3190_319060

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem 1: Solution set of f(x) > 2
theorem solution_set_f (x : ℝ) : f x > 2 ↔ x < -2/3 ∨ x > 0 := by sorry

-- Theorem 2: Range of k for |k - 1| < g(x)
theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, |k - 1| < g x) → -3 < k ∧ k < 5 := by sorry

end solution_set_f_range_of_k_l3190_319060


namespace power_division_equals_integer_l3190_319069

theorem power_division_equals_integer : 3^18 / 27^2 = 531441 := by
  sorry

end power_division_equals_integer_l3190_319069


namespace specific_trapezoid_area_l3190_319050

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculates the area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 36,
    diagonal_length := 48,
    longer_base := 60
  }
  area t = 1105.92 := by
  sorry

end specific_trapezoid_area_l3190_319050


namespace matrix_not_invertible_l3190_319009

def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2 + x, 9],
    ![4 - x, 10]]

theorem matrix_not_invertible (x : ℝ) :
  ¬(IsUnit (A x).det) ↔ x = 16 / 19 := by
  sorry

end matrix_not_invertible_l3190_319009


namespace original_number_of_people_l3190_319041

theorem original_number_of_people (x : ℕ) : 
  (3 * x / 4 : ℚ) - (3 * x / 20 : ℚ) = 16 → x = 27 := by
  sorry

end original_number_of_people_l3190_319041


namespace log_base_conversion_l3190_319011

theorem log_base_conversion (a : ℝ) (h : Real.log 16 / Real.log 14 = a) :
  Real.log 14 / Real.log 8 = 4 / (3 * a) := by
  sorry

end log_base_conversion_l3190_319011


namespace manufacturing_quality_probability_l3190_319017

theorem manufacturing_quality_probability 
  (defect_rate1 : ℝ) 
  (defect_rate2 : ℝ) 
  (h1 : defect_rate1 = 0.03) 
  (h2 : defect_rate2 = 0.05) 
  (independent : True) -- Representing the independence of processes
  : (1 - defect_rate1) * (1 - defect_rate2) = 0.9215 := by
  sorry

end manufacturing_quality_probability_l3190_319017


namespace system_has_solution_l3190_319057

/-- Given a system of equations {sin x + a = b x, cos x = b} where a and b are real numbers,
    and the equation sin x + a = b x has exactly two solutions,
    prove that the system has at least one solution. -/
theorem system_has_solution (a b : ℝ) 
    (h : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
         (∀ x, Real.sin x + a = b * x ↔ x = x₁ ∨ x = x₂)) :
  ∃ x, Real.sin x + a = b * x ∧ Real.cos x = b := by
  sorry

end system_has_solution_l3190_319057


namespace binomial_inequality_l3190_319080

theorem binomial_inequality (n : ℕ) : 2 ≤ (1 + 1 / n)^n ∧ (1 + 1 / n)^n < 3 := by
  sorry

end binomial_inequality_l3190_319080


namespace derivative_f_at_one_l3190_319061

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 + 3*(x - 1)

theorem derivative_f_at_one :
  deriv f 1 = 3 := by sorry

end derivative_f_at_one_l3190_319061


namespace annual_car_insurance_cost_l3190_319025

/-- Theorem: If a person spends 40000 dollars on car insurance over a decade,
    then their annual car insurance cost is 4000 dollars. -/
theorem annual_car_insurance_cost (total_cost : ℕ) (years : ℕ) (annual_cost : ℕ) :
  total_cost = 40000 →
  years = 10 →
  annual_cost = total_cost / years →
  annual_cost = 4000 := by
  sorry

end annual_car_insurance_cost_l3190_319025


namespace trajectory_of_point_P_l3190_319083

/-- The trajectory of point P given the symmetry of points A and B and the product of slopes condition -/
theorem trajectory_of_point_P (x y : ℝ) : 
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (1, -1)
  let P : ℝ × ℝ := (x, y)
  let slope_AP := (y - A.2) / (x - A.1)
  let slope_BP := (y - B.2) / (x - B.1)
  x ≠ 1 ∧ x ≠ -1 →
  slope_AP * slope_BP = 1/3 →
  3 * y^2 - x^2 = 2 :=
sorry

end trajectory_of_point_P_l3190_319083


namespace ellipse_eccentricity_l3190_319029

/-- Given an ellipse with the following properties:
    1. The chord passing through the focus and perpendicular to the major axis has a length of √2
    2. The distance from the focus to the corresponding directrix is 1
    This theorem states that the eccentricity of the ellipse is √2/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * b^2 / a = Real.sqrt 2) (h4 : a^2 / c - c = 1) : 
  c / a = Real.sqrt 2 / 2 := by
  sorry

end ellipse_eccentricity_l3190_319029


namespace geometric_sequence_problem_l3190_319033

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a)
  (h_eq : a 2 * a 4 * a 5 = a 3 * a 6)
  (h_prod : a 9 * a 10 = -8) :
  a 7 = -2 := by
sorry

end geometric_sequence_problem_l3190_319033


namespace soap_cost_per_pound_l3190_319048

theorem soap_cost_per_pound 
  (num_bars : ℕ) 
  (weight_per_bar : ℝ) 
  (total_cost : ℝ) 
  (h1 : num_bars = 20)
  (h2 : weight_per_bar = 1.5)
  (h3 : total_cost = 15) : 
  total_cost / (num_bars * weight_per_bar) = 0.5 := by
sorry

end soap_cost_per_pound_l3190_319048


namespace jeans_price_calculation_l3190_319064

/-- The price of jeans after discount and tax -/
def jeans_final_price (socks_price t_shirt_price jeans_price : ℝ)
  (jeans_discount t_shirt_discount tax_rate : ℝ) : ℝ :=
  let jeans_discounted := jeans_price * (1 - jeans_discount)
  let taxable_amount := jeans_discounted + t_shirt_price * (1 - t_shirt_discount)
  jeans_discounted * (1 + tax_rate)

/-- The problem statement -/
theorem jeans_price_calculation :
  let socks_price := 5
  let t_shirt_price := socks_price + 10
  let jeans_price := 2 * t_shirt_price
  let jeans_discount := 0.15
  let t_shirt_discount := 0.10
  let tax_rate := 0.08
  jeans_final_price socks_price t_shirt_price jeans_price
    jeans_discount t_shirt_discount tax_rate = 27.54 := by
  sorry

end jeans_price_calculation_l3190_319064


namespace correct_ways_select_four_correct_ways_select_five_l3190_319031

/-- Number of distinct red balls -/
def num_red_balls : ℕ := 4

/-- Number of distinct white balls -/
def num_white_balls : ℕ := 7

/-- Score for selecting a red ball -/
def red_score : ℕ := 2

/-- Score for selecting a white ball -/
def white_score : ℕ := 1

/-- The number of ways to select 4 balls such that the number of red balls
    is not less than the number of white balls -/
def ways_select_four : ℕ := 115

/-- The number of ways to select 5 balls such that the total score
    is at least 7 points -/
def ways_select_five : ℕ := 301

/-- Theorem stating the correct number of ways to select 4 balls -/
theorem correct_ways_select_four :
  ways_select_four = Nat.choose num_red_balls 4 +
    Nat.choose num_red_balls 3 * Nat.choose num_white_balls 1 +
    Nat.choose num_red_balls 2 * Nat.choose num_white_balls 2 := by sorry

/-- Theorem stating the correct number of ways to select 5 balls -/
theorem correct_ways_select_five :
  ways_select_five = Nat.choose num_red_balls 2 * Nat.choose num_white_balls 3 +
    Nat.choose num_red_balls 3 * Nat.choose num_white_balls 2 +
    Nat.choose num_red_balls 4 * Nat.choose num_white_balls 1 := by sorry

end correct_ways_select_four_correct_ways_select_five_l3190_319031


namespace gcd_special_numbers_l3190_319081

theorem gcd_special_numbers : Nat.gcd 33333333 777777777 = 2 := by
  sorry

end gcd_special_numbers_l3190_319081


namespace narration_per_disc_l3190_319095

/-- Represents the duration of the narration in minutes -/
def narration_duration : ℕ := 6 * 60 + 45

/-- Represents the capacity of each disc in minutes -/
def disc_capacity : ℕ := 75

/-- Calculates the minimum number of discs needed -/
def min_discs : ℕ := (narration_duration + disc_capacity - 1) / disc_capacity

/-- Theorem stating the duration of narration on each disc -/
theorem narration_per_disc :
  (narration_duration : ℚ) / min_discs = 67.5 := by sorry

end narration_per_disc_l3190_319095


namespace cos_graph_transformation_l3190_319007

theorem cos_graph_transformation (x : ℝ) : 
  let f (x : ℝ) := Real.cos ((1/2 : ℝ) * x - π/6)
  let g (x : ℝ) := f (x + π/3)
  let h (x : ℝ) := g (2 * x)
  h x = Real.cos x := by sorry

end cos_graph_transformation_l3190_319007


namespace function_properties_l3190_319043

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - 4| + |x - a|

-- State the theorem
theorem function_properties (a : ℝ) 
  (h1 : ∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m ∧ ∃ (y : ℝ), f a y = m) 
  (h2 : ∀ (x : ℝ), f a x ≥ a) :
  (a = 2) ∧ 
  (∀ (x : ℝ), f 2 x ≤ 5 ↔ 1/2 ≤ x ∧ x ≤ 11/2) := by
sorry

end function_properties_l3190_319043


namespace negation_of_universal_proposition_l3190_319039

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) := by
  sorry

end negation_of_universal_proposition_l3190_319039


namespace inequality_solution_existence_condition_l3190_319014

-- Define the functions f and g
def f (a x : ℝ) := |2 * x + a| - |2 * x + 3|
def g (x : ℝ) := |x - 1| - 3

-- Theorem for the first part of the problem
theorem inequality_solution (x : ℝ) :
  |g x| < 2 ↔ -4 < x ∧ x < 6 := by sorry

-- Theorem for the second part of the problem
theorem existence_condition (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) ↔ 0 ≤ a ∧ a ≤ 6 := by sorry

end inequality_solution_existence_condition_l3190_319014


namespace unique_solutions_for_exponential_equation_l3190_319072

theorem unique_solutions_for_exponential_equation :
  ∀ x n : ℕ+, 3 * 2^(x : ℕ) + 4 = (n : ℕ)^2 ↔ (x = 2 ∧ n = 4) ∨ (x = 5 ∧ n = 10) ∨ (x = 6 ∧ n = 14) :=
by sorry

end unique_solutions_for_exponential_equation_l3190_319072


namespace polynomial_division_theorem_l3190_319074

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 + 2*x^2 + 8 = (x - 1)*(x^5 + x^4 + x^3 + x^2 + 3*x + 3) + 11 := by
  sorry

end polynomial_division_theorem_l3190_319074


namespace paul_weed_eating_money_l3190_319088

/-- The amount of money Paul made weed eating -/
def weed_eating_money (mowing_money weekly_spending weeks_lasted : ℕ) : ℕ :=
  weekly_spending * weeks_lasted - mowing_money

/-- Theorem stating that Paul made $28 weed eating -/
theorem paul_weed_eating_money :
  weed_eating_money 44 9 8 = 28 := by
  sorry

end paul_weed_eating_money_l3190_319088


namespace chewing_gum_cost_l3190_319006

/-- Proves that the cost of each pack of chewing gum is $1, given the initial amount,
    purchases, and remaining amount. -/
theorem chewing_gum_cost
  (initial_amount : ℝ)
  (num_gum_packs : ℕ)
  (num_chocolate_bars : ℕ)
  (chocolate_bar_price : ℝ)
  (num_candy_canes : ℕ)
  (candy_cane_price : ℝ)
  (remaining_amount : ℝ)
  (h1 : initial_amount = 10)
  (h2 : num_gum_packs = 3)
  (h3 : num_chocolate_bars = 5)
  (h4 : chocolate_bar_price = 1)
  (h5 : num_candy_canes = 2)
  (h6 : candy_cane_price = 0.5)
  (h7 : remaining_amount = 1) :
  (initial_amount - remaining_amount
    - (num_chocolate_bars * chocolate_bar_price + num_candy_canes * candy_cane_price))
  / num_gum_packs = 1 := by
sorry


end chewing_gum_cost_l3190_319006


namespace smallest_addition_for_divisibility_l3190_319058

theorem smallest_addition_for_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((627 + y) % 510 = 0 ∧ (627 + y) % 4590 = 0 ∧ (627 + y) % 105 = 0)) ∧
  ((627 + x) % 510 = 0 ∧ (627 + x) % 4590 = 0 ∧ (627 + x) % 105 = 0) ∧
  x = 31503 := by
  sorry

end smallest_addition_for_divisibility_l3190_319058


namespace curve_is_ellipse_l3190_319052

/-- Given real numbers a and b where ab ≠ 0, the curve bx² + ay² = ab represents an ellipse. -/
theorem curve_is_ellipse (a b : ℝ) (h : a * b ≠ 0) :
  ∃ (A B : ℝ), A > 0 ∧ B > 0 ∧
  ∀ (x y : ℝ), b * x^2 + a * y^2 = a * b ↔ x^2 / A^2 + y^2 / B^2 = 1 :=
sorry

end curve_is_ellipse_l3190_319052


namespace kim_total_water_consumption_l3190_319021

/-- The amount of water Kim drinks from various sources -/
def kim_water_consumption (quart_to_ounce : Real) (bottle_quarts : Real) (can_ounces : Real) 
  (shared_bottle_ounces : Real) (jake_fraction : Real) : Real :=
  let bottle_ounces := bottle_quarts * quart_to_ounce
  let kim_shared_fraction := 1 - jake_fraction
  bottle_ounces + can_ounces + (kim_shared_fraction * shared_bottle_ounces)

/-- Theorem stating that Kim's total water consumption is 79.2 ounces -/
theorem kim_total_water_consumption :
  kim_water_consumption 32 1.5 12 32 (2/5) = 79.2 := by
  sorry

end kim_total_water_consumption_l3190_319021


namespace teacher_instructions_l3190_319020

theorem teacher_instructions (x : ℤ) : 4 * (3 * (x + 3) - 2) = 4 * (3 * x + 7) := by
  sorry

end teacher_instructions_l3190_319020


namespace root_equation_q_value_l3190_319082

theorem root_equation_q_value (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 16/3 := by sorry

end root_equation_q_value_l3190_319082
