import Mathlib

namespace ellipse_k_range_l1897_189734

-- Define the equation
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (3 + k) + y^2 / (2 - k) = 1 ∧
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ k ∈ (Set.Ioo (-3) (-1/2) ∪ Set.Ioo (-1/2) 2) :=
by sorry

end ellipse_k_range_l1897_189734


namespace chloe_profit_l1897_189797

/-- Calculates the profit from selling chocolate-dipped strawberries -/
def calculate_profit (buy_price_per_dozen : ℕ) (sell_price_per_half_dozen : ℕ) (dozens_sold : ℕ) : ℕ :=
  let cost := buy_price_per_dozen * dozens_sold
  let revenue := sell_price_per_half_dozen * 2 * dozens_sold
  revenue - cost

/-- Proves that Chloe's profit is $500 given the specified conditions -/
theorem chloe_profit :
  calculate_profit 50 30 50 = 500 := by
  sorry

end chloe_profit_l1897_189797


namespace highway_length_is_105_l1897_189793

/-- The length of a highway where two cars meet after traveling from opposite ends -/
def highway_length (speed1 speed2 time : ℝ) : ℝ :=
  speed1 * time + speed2 * time

/-- Theorem: The highway length is 105 miles given the specific conditions -/
theorem highway_length_is_105 :
  highway_length 15 20 3 = 105 := by
  sorry

end highway_length_is_105_l1897_189793


namespace classroom_position_representation_l1897_189779

/-- Represents a position in a classroom -/
structure ClassroomPosition where
  column : ℕ
  row : ℕ

/-- Given that (1, 2) represents the 1st column and 2nd row -/
def given_position : ClassroomPosition := ⟨1, 2⟩

/-- The position we want to prove represents the 2nd column and 3rd row -/
def target_position : ClassroomPosition := ⟨2, 3⟩

/-- Theorem stating that if (1, 2) represents the 1st column and 2nd row,
    then (2, 3) represents the 2nd column and 3rd row -/
theorem classroom_position_representation :
  (given_position.column = 1 ∧ given_position.row = 2) →
  (target_position.column = 2 ∧ target_position.row = 3) :=
by sorry

end classroom_position_representation_l1897_189779


namespace average_income_p_q_l1897_189745

theorem average_income_p_q (p q r : ℕ) : 
  (q + r) / 2 = 6250 →
  (p + r) / 2 = 5200 →
  p = 4000 →
  (p + q) / 2 = 5050 :=
by
  sorry

#check average_income_p_q

end average_income_p_q_l1897_189745


namespace inequality_solution_l1897_189780

def solution_set : Set ℝ := {x | x < -2 ∨ (x ≥ 0 ∧ x < 2)}

theorem inequality_solution :
  {x : ℝ | x / (x^2 - 4) ≥ 0} = solution_set :=
by sorry

end inequality_solution_l1897_189780


namespace hash_difference_l1897_189798

/-- Custom operation # defined as x#y = xy + 2x -/
def hash (x y : ℤ) : ℤ := x * y + 2 * x

/-- Theorem stating that (5#3) - (3#5) = 4 -/
theorem hash_difference : hash 5 3 - hash 3 5 = 4 := by
  sorry

end hash_difference_l1897_189798


namespace min_correct_answers_for_min_score_l1897_189749

/-- Represents the scoring system and conditions of the AMC 12 test -/
structure AMC12Test where
  total_questions : Nat
  attempted_questions : Nat
  correct_points : Int
  incorrect_points : Int
  unanswered_points : Int
  min_score : Int

/-- Calculates the score based on the number of correct answers -/
def calculate_score (test : AMC12Test) (correct_answers : Nat) : Int :=
  let incorrect_answers := test.attempted_questions - correct_answers
  let unanswered_questions := test.total_questions - test.attempted_questions
  correct_answers * test.correct_points +
  incorrect_answers * test.incorrect_points +
  unanswered_questions * test.unanswered_points

/-- Theorem stating the minimum number of correct answers needed to achieve the minimum score -/
theorem min_correct_answers_for_min_score (test : AMC12Test)
  (h1 : test.total_questions = 35)
  (h2 : test.attempted_questions = 30)
  (h3 : test.correct_points = 7)
  (h4 : test.incorrect_points = -1)
  (h5 : test.unanswered_points = 2)
  (h6 : test.min_score = 150) :
  ∃ (n : Nat), n = 20 ∧ 
    (∀ (m : Nat), m < n → calculate_score test m < test.min_score) ∧
    calculate_score test n ≥ test.min_score :=
  sorry

end min_correct_answers_for_min_score_l1897_189749


namespace min_common_edges_l1897_189748

/-- Represents a closed route on a grid -/
def ClosedRoute (n : ℕ) := List (Fin n × Fin n)

/-- The number of edges in an n×n grid -/
def gridEdges (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- The number of edges traversed by a closed route visiting all vertices once -/
def routeEdges (n : ℕ) : ℕ := n * n

theorem min_common_edges (route1 route2 : ClosedRoute 8) :
  (∀ v : Fin 8 × Fin 8, v ∈ route1.toFinset ∧ v ∈ route2.toFinset) →
  (∀ v : Fin 8 × Fin 8, (route1.count v = 1 ∧ route2.count v = 1)) →
  (∃ m : ℕ, m = 16 ∧ m = routeEdges 8 + routeEdges 8 - gridEdges 7) := by
  sorry

end min_common_edges_l1897_189748


namespace inscribed_ngon_iff_three_or_four_l1897_189755

/-- An ellipse that is not a circle -/
structure NonCircularEllipse where
  -- Add necessary fields to define a non-circular ellipse
  is_not_circle : Bool

/-- A regular n-gon -/
structure RegularNGon (n : ℕ) where
  -- Add necessary fields to define a regular n-gon
  vertices : Fin n → ℝ × ℝ

/-- Predicate to check if a regular n-gon is inscribed in an ellipse -/
def is_inscribed (E : NonCircularEllipse) (n : ℕ) (polygon : RegularNGon n) : Prop :=
  sorry

/-- Theorem: A regular n-gon can be inscribed in a non-circular ellipse if and only if n = 3 or n = 4 -/
theorem inscribed_ngon_iff_three_or_four (E : NonCircularEllipse) (n : ℕ) :
    (∃ (polygon : RegularNGon n), is_inscribed E n polygon) ↔ (n = 3 ∨ n = 4) := by
  sorry

end inscribed_ngon_iff_three_or_four_l1897_189755


namespace arithmetic_sequence_problem_l1897_189733

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions, prove that 2a_10 - a_12 = 24 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 := by
  sorry

end arithmetic_sequence_problem_l1897_189733


namespace frank_has_three_cookies_l1897_189728

/-- Given the number of cookies Millie has -/
def millies_cookies : ℕ := 4

/-- Mike's cookies in terms of Millie's -/
def mikes_cookies (m : ℕ) : ℕ := 3 * m

/-- Frank's cookies in terms of Mike's -/
def franks_cookies (m : ℕ) : ℕ := m / 2 - 3

/-- Theorem stating that Frank has 3 cookies given the conditions -/
theorem frank_has_three_cookies :
  franks_cookies (mikes_cookies millies_cookies) = 3 := by
  sorry

end frank_has_three_cookies_l1897_189728


namespace min_value_x_over_y_l1897_189767

theorem min_value_x_over_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + y = 2) :
  ∀ x' y' : ℝ, x' > 0 → y' > 0 → 1/x' + y' = 2 → x/y ≤ x'/y' ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + y₀ = 2 ∧ x₀/y₀ = 1 :=
by sorry

end min_value_x_over_y_l1897_189767


namespace triangle_area_l1897_189786

theorem triangle_area (A B C : Real) (a b c : Real) :
  A = π / 3 →
  a = Real.sqrt 3 →
  c = 1 →
  (∃ S : Real, S = (Real.sqrt 3) / 2 ∧ S = (1 / 2) * a * c) :=
by sorry

end triangle_area_l1897_189786


namespace marcus_paintings_l1897_189708

def paintings_per_day (day : Nat) : Nat :=
  match day with
  | 1 => 2
  | 2 => min 8 (2 * paintings_per_day 1)
  | 3 => min 8 (paintings_per_day 2 + (paintings_per_day 2 * 3) / 4)
  | 4 => min 8 (paintings_per_day 3 + (paintings_per_day 3 * 1) / 2)
  | 5 => min 8 (paintings_per_day 4 + (paintings_per_day 4 * 1) / 4)
  | _ => 0

def total_paintings : Nat :=
  paintings_per_day 1 + paintings_per_day 2 + paintings_per_day 3 + paintings_per_day 4 + paintings_per_day 5

theorem marcus_paintings :
  total_paintings = 29 :=
by sorry

end marcus_paintings_l1897_189708


namespace sally_net_earnings_l1897_189720

def calculate_net_earnings (
  first_month_income : ℝ)
  (first_month_expenses : ℝ)
  (side_hustle : ℝ)
  (income_raise_percentage : ℝ)
  (expense_increase_percentage : ℝ) : ℝ :=
  let first_month := first_month_income + side_hustle - first_month_expenses
  let second_month_income := first_month_income * (1 + income_raise_percentage)
  let second_month_expenses := first_month_expenses * (1 + expense_increase_percentage)
  let second_month := second_month_income + side_hustle - second_month_expenses
  first_month + second_month

theorem sally_net_earnings :
  calculate_net_earnings 1000 200 150 0.1 0.15 = 1970 := by
  sorry

end sally_net_earnings_l1897_189720


namespace equation_solutions_l1897_189770

theorem equation_solutions : 
  {x : ℝ | Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6} = {2, -1} := by
  sorry

end equation_solutions_l1897_189770


namespace hundredths_place_is_zero_l1897_189776

def number : ℚ := 317.502

theorem hundredths_place_is_zero : 
  (number * 100 % 10).floor = 0 := by sorry

end hundredths_place_is_zero_l1897_189776


namespace n_balanced_max_size_l1897_189742

/-- A set S is n-balanced if:
    1) Any subset of 3 elements contains at least 2 that are connected
    2) Any subset of n elements contains at least 2 that are not connected -/
def IsNBalanced (n : ℕ) (S : Set α) (connected : α → α → Prop) : Prop :=
  n ≠ 0 ∧
  (∀ (a b c : α), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
    connected a b ∨ connected b c ∨ connected a c) ∧
  (∀ (T : Set α), T ⊆ S → T.ncard = n →
    ∃ (x y : α), x ∈ T ∧ y ∈ T ∧ x ≠ y ∧ ¬connected x y)

theorem n_balanced_max_size (n : ℕ) (S : Set α) (connected : α → α → Prop) :
  IsNBalanced n S connected → S.ncard ≤ (n - 1) * (n + 2) / 2 :=
sorry

end n_balanced_max_size_l1897_189742


namespace draw_red_black_red_prob_value_l1897_189729

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (hTotal : total_cards = 52)
  (hRed : red_cards = 26)
  (hBlack : black_cards = 26)
  (hSum : red_cards + black_cards = total_cards)

/-- The probability of drawing a red card first, then a black card, and then a red card -/
def draw_red_black_red_prob (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards *
  (d.black_cards : ℚ) / (d.total_cards - 1) *
  ((d.red_cards - 1) : ℚ) / (d.total_cards - 2)

/-- Theorem stating the probability is 13/102 -/
theorem draw_red_black_red_prob_value (d : Deck) :
  draw_red_black_red_prob d = 13 / 102 := by
  sorry

end draw_red_black_red_prob_value_l1897_189729


namespace symmetric_difference_of_A_and_B_l1897_189713

-- Define the sets A and B
def A : Set ℝ := {y | y > 0}
def B : Set ℝ := {y | y ≤ 2}

-- Define the symmetric difference operation
def symmetricDifference (M N : Set ℝ) : Set ℝ :=
  (M \ N) ∪ (N \ M)

-- State the theorem
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {y | y ≤ 0 ∨ y > 2} := by sorry

end symmetric_difference_of_A_and_B_l1897_189713


namespace sequence_properties_l1897_189799

def S (n : ℕ) : ℤ := n^2 - 9*n

def a (n : ℕ) : ℤ := 2*n - 10

theorem sequence_properties :
  (∀ n : ℕ, S (n+1) - S n = a (n+1)) ∧
  (∃! k : ℕ, k > 0 ∧ 5 < a k ∧ a k < 8 ∧ k = 8) := by
  sorry

end sequence_properties_l1897_189799


namespace cube_net_in_square_l1897_189715

-- Define a square
structure Square where
  side_length : ℝ

-- Define a cube net
structure CubeNet where
  faces : Finset (Square)
  face_count : Nat

-- Define the problem
theorem cube_net_in_square :
  ∃ (large_square : Square) (cube_net : CubeNet),
    large_square.side_length = 3 ∧
    cube_net.face_count = 6 ∧
    ∀ (face : Square), face ∈ cube_net.faces → face.side_length = 1 ∧
    -- The condition that the cube net fits within the large square
    -- is represented by this placeholder
    (cube_net_fits_in_square : Prop) :=
by
  sorry

end cube_net_in_square_l1897_189715


namespace elephant_arrangements_l1897_189775

theorem elephant_arrangements (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 2) :
  (n! / k!) = 20160 := by
  sorry

end elephant_arrangements_l1897_189775


namespace inscribed_square_area_l1897_189732

theorem inscribed_square_area (triangle_area : ℝ) (square1_area : ℝ) (square2_fraction : ℝ) :
  square1_area = 441 →
  square2_fraction = 4 / 9 →
  triangle_area = 2 * square1_area →
  square2_fraction * triangle_area = 392 := by
  sorry

end inscribed_square_area_l1897_189732


namespace water_bottle_distribution_l1897_189781

theorem water_bottle_distribution (initial_bottles : ℕ) (drunk_bottles : ℕ) (num_friends : ℕ) : 
  initial_bottles = 120 → 
  drunk_bottles = 15 → 
  num_friends = 5 → 
  (initial_bottles - drunk_bottles) / (num_friends + 1) = 17 := by
sorry

end water_bottle_distribution_l1897_189781


namespace least_x_1894x_divisible_by_3_l1897_189774

theorem least_x_1894x_divisible_by_3 :
  ∃ x : ℕ, x < 10 ∧ (1894 * x) % 3 = 0 ∧ ∀ y : ℕ, y < x → y < 10 → (1894 * y) % 3 ≠ 0 :=
by sorry

end least_x_1894x_divisible_by_3_l1897_189774


namespace solution_set_inequality_l1897_189771

theorem solution_set_inequality (x : ℝ) :
  {x : ℝ | |x + 1| - |x - 5| < 4} = Set.Iio 4 :=
sorry

end solution_set_inequality_l1897_189771


namespace r_amount_calculation_l1897_189792

def total_amount : ℝ := 5000

theorem r_amount_calculation (p_amount q_amount r_amount : ℝ) 
  (h1 : p_amount + q_amount + r_amount = total_amount)
  (h2 : r_amount = (2/3) * (p_amount + q_amount)) :
  r_amount = 2000 := by
  sorry

end r_amount_calculation_l1897_189792


namespace sum_inequality_l1897_189747

theorem sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x * y + y * z + z * x = 1) : 
  (1 + x^2 * y^2) / (x + y)^2 + (1 + y^2 * z^2) / (y + z)^2 + (1 + z^2 * x^2) / (z + x)^2 ≥ 5/2 := by
  sorry

end sum_inequality_l1897_189747


namespace count_increasing_digit_numbers_eq_502_l1897_189778

def is_increasing_digits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get i < (n.digits 10).get j

def count_increasing_digit_numbers : ℕ :=
  (Finset.range 8).sum (λ k => Nat.choose 9 (k + 2))

theorem count_increasing_digit_numbers_eq_502 :
  count_increasing_digit_numbers = 502 :=
sorry

end count_increasing_digit_numbers_eq_502_l1897_189778


namespace heptagon_diagonals_l1897_189711

/-- The number of diagonals in a convex n-gon --/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Proof that a convex heptagon has 14 diagonals --/
theorem heptagon_diagonals : num_diagonals 7 = 14 := by
  sorry

end heptagon_diagonals_l1897_189711


namespace boys_less_than_four_sevenths_l1897_189790

/-- Represents a class of students with two hiking trips -/
structure HikingClass where
  boys : ℕ
  girls : ℕ
  boys_trip1 : ℕ
  girls_trip1 : ℕ
  boys_trip2 : ℕ
  girls_trip2 : ℕ

/-- The conditions of the hiking trips -/
def validHikingClass (c : HikingClass) : Prop :=
  c.boys_trip1 < (2 * (c.boys_trip1 + c.girls_trip1)) / 5 ∧
  c.boys_trip2 < (2 * (c.boys_trip2 + c.girls_trip2)) / 5 ∧
  c.boys_trip1 + c.boys_trip2 ≥ c.boys ∧
  c.girls_trip1 ≤ c.girls ∧
  c.girls_trip2 ≤ c.girls

/-- The main theorem to prove -/
theorem boys_less_than_four_sevenths (c : HikingClass) 
  (h : validHikingClass c) : 
  c.boys < (4 * (c.boys + c.girls)) / 7 := by
  sorry

end boys_less_than_four_sevenths_l1897_189790


namespace arc_length_cardioid_l1897_189743

/-- The arc length of the curve ρ = 1 - sin φ from -π/2 to -π/6 is 2 -/
theorem arc_length_cardioid (φ : ℝ) : 
  let ρ : ℝ → ℝ := λ φ => 1 - Real.sin φ
  let L : ℝ := ∫ φ in Set.Icc (-Real.pi/2) (-Real.pi/6), 
    Real.sqrt ((ρ φ)^2 + (- Real.cos φ)^2)
  L = 2 := by sorry

end arc_length_cardioid_l1897_189743


namespace pizza_slices_left_l1897_189785

theorem pizza_slices_left (total_slices : ℕ) (fraction_eaten : ℚ) : 
  total_slices = 16 → fraction_eaten = 3/4 → total_slices - (total_slices * fraction_eaten).floor = 4 := by
  sorry

end pizza_slices_left_l1897_189785


namespace eggs_next_month_l1897_189746

/-- Calculates the number of eggs after the next purchase given initial conditions --/
def eggs_after_next_purchase (eggs_left_previous : ℕ) (eggs_after_purchase : ℕ) (eggs_consumed : ℕ) : ℕ :=
  let eggs_bought := eggs_after_purchase - eggs_left_previous
  let eggs_remaining := eggs_after_purchase - eggs_consumed
  eggs_remaining + eggs_bought

/-- Theorem stating that given the initial conditions, the number of eggs after the next purchase will be 41 --/
theorem eggs_next_month (eggs_left_previous : ℕ) (eggs_after_purchase : ℕ) (eggs_consumed : ℕ)
  (h1 : eggs_left_previous = 27)
  (h2 : eggs_after_purchase = 58)
  (h3 : eggs_consumed = 48) :
  eggs_after_next_purchase eggs_left_previous eggs_after_purchase eggs_consumed = 41 :=
by sorry

end eggs_next_month_l1897_189746


namespace smallest_square_containing_circle_l1897_189784

theorem smallest_square_containing_circle (r : ℝ) (h : r = 4) :
  (2 * r) ^ 2 = 64 := by sorry

end smallest_square_containing_circle_l1897_189784


namespace mean_of_additional_numbers_l1897_189796

theorem mean_of_additional_numbers
  (original_count : Nat)
  (original_mean : ℝ)
  (new_count : Nat)
  (new_mean : ℝ)
  (h1 : original_count = 7)
  (h2 : original_mean = 72)
  (h3 : new_count = 9)
  (h4 : new_mean = 80) :
  let x_plus_y := new_count * new_mean - original_count * original_mean
  let mean_x_y := x_plus_y / 2
  mean_x_y = 108 := by
  sorry

end mean_of_additional_numbers_l1897_189796


namespace base3_to_base10_conversion_l1897_189722

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

/-- The base 3 representation of the number -/
def base3Digits : List Nat := [2, 0, 2, 0, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Digits = 182 := by
  sorry

#eval base3ToBase10 base3Digits

end base3_to_base10_conversion_l1897_189722


namespace total_mass_on_boat_total_mass_is_183_l1897_189707

/-- Calculates the total mass of two individuals on a boat given specific conditions. -/
theorem total_mass_on_boat (boat_length boat_breadth initial_sinking_depth : ℝ) 
  (mass_second_person : ℝ) (water_density : ℝ) : ℝ :=
  let volume_displaced := boat_length * boat_breadth * initial_sinking_depth
  let mass_first_person := volume_displaced * water_density
  mass_first_person + mass_second_person

/-- Proves that the total mass of two individuals on a boat is 183 kg under given conditions. -/
theorem total_mass_is_183 :
  total_mass_on_boat 3 2 0.018 75 1000 = 183 := by
  sorry

end total_mass_on_boat_total_mass_is_183_l1897_189707


namespace divisible_by_five_l1897_189752

theorem divisible_by_five (B : Nat) : 
  B < 10 → (476 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := by
  sorry

end divisible_by_five_l1897_189752


namespace race_win_probability_l1897_189737

/-- Represents the weather conditions -/
inductive Weather
  | Sunny
  | Rainy

/-- Represents a car in the race -/
structure Car where
  winProbability : ℚ

/-- Represents Car E with weather-dependent win probability -/
structure CarE where
  sunnyProbability : ℚ
  rainyProbability : ℚ

/-- Calculate the total win probability for the given cars and Car E under specified weather -/
def totalWinProbability (cars : List Car) (carE : CarE) (weather : Weather) : ℚ :=
  let carEProb := match weather with
    | Weather.Sunny => carE.sunnyProbability
    | Weather.Rainy => carE.rainyProbability
  (cars.map (·.winProbability)).sum + carEProb

/-- The main theorem statement -/
theorem race_win_probability 
  (carA carB carC carD : Car)
  (carE : CarE) :
  carA.winProbability = 1/8 →
  carB.winProbability = 1/12 →
  carC.winProbability = 1/6 →
  carD.winProbability = 1/10 →
  carE.sunnyProbability = 1/20 →
  carE.rainyProbability = 1/15 →
  totalWinProbability [carA, carB, carC, carD] carE Weather.Sunny = 21/40 ∧
  totalWinProbability [carA, carB, carC, carD] carE Weather.Rainy = 13/24 :=
by sorry


end race_win_probability_l1897_189737


namespace people_left_of_kolya_l1897_189706

/-- Given a line of people with the following conditions:
  * There are 12 people to the right of Kolya
  * There are 20 people to the left of Sasha
  * There are 8 people to the right of Sasha
  Then there are 16 people to the left of Kolya -/
theorem people_left_of_kolya 
  (total : ℕ) 
  (kolya_right : ℕ) 
  (sasha_left : ℕ) 
  (sasha_right : ℕ) 
  (h1 : kolya_right = 12)
  (h2 : sasha_left = 20)
  (h3 : sasha_right = 8)
  (h4 : total = sasha_left + sasha_right + 1) : 
  total - kolya_right - 1 = 16 := by
sorry

end people_left_of_kolya_l1897_189706


namespace maxwell_brad_meeting_time_l1897_189773

/-- Proves that Maxwell walks for 10 hours before meeting Brad -/
theorem maxwell_brad_meeting_time :
  ∀ (distance : ℝ) (maxwell_speed : ℝ) (brad_speed : ℝ) (head_start : ℝ),
    distance = 94 →
    maxwell_speed = 4 →
    brad_speed = 6 →
    head_start = 1 →
    ∃ (t : ℝ),
      t > 0 ∧
      maxwell_speed * (t + head_start) + brad_speed * t = distance ∧
      t + head_start = 10 :=
by
  sorry

end maxwell_brad_meeting_time_l1897_189773


namespace susan_hourly_rate_l1897_189716

/-- Susan's vacation and pay structure -/
structure VacationPay where
  work_days_per_week : ℕ
  paid_vacation_days : ℕ
  hours_per_day : ℕ
  missed_pay : ℕ
  vacation_weeks : ℕ

/-- Calculate Susan's hourly pay rate -/
def hourly_pay_rate (v : VacationPay) : ℚ :=
  let total_vacation_days := v.vacation_weeks * v.work_days_per_week
  let unpaid_vacation_days := total_vacation_days - v.paid_vacation_days
  let daily_pay := v.missed_pay / unpaid_vacation_days
  daily_pay / v.hours_per_day

/-- Theorem: Susan's hourly pay rate is $15 -/
theorem susan_hourly_rate :
  let v : VacationPay := {
    work_days_per_week := 5,
    paid_vacation_days := 6,
    hours_per_day := 8,
    missed_pay := 480,
    vacation_weeks := 2
  }
  hourly_pay_rate v = 15 := by sorry

end susan_hourly_rate_l1897_189716


namespace system_solution_l1897_189703

theorem system_solution : ∃ (x y z : ℝ), 
  (x + y + z = 15) ∧ 
  (x^2 + y^2 + z^2 = 81) ∧ 
  (x*y + x*z = 3*y*z) ∧ 
  ((x = 6 ∧ y = 3 ∧ z = 6) ∨ (x = 6 ∧ y = 6 ∧ z = 3)) := by
  sorry

end system_solution_l1897_189703


namespace tetrahedron_volume_l1897_189751

/-- Tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem: The volume of tetrahedron PQRS with given edge lengths is 99/2 -/
theorem tetrahedron_volume :
  ∀ t : Tetrahedron,
  t.PQ = 6 ∧
  t.PR = 4 ∧
  t.PS = 5 ∧
  t.QR = 5 ∧
  t.QS = 4 ∧
  t.RS = 15 / 5 * Real.sqrt 2 →
  volume t = 99 / 2 :=
sorry

end tetrahedron_volume_l1897_189751


namespace logarithm_problem_l1897_189791

theorem logarithm_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1)
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
  (h2 : x * y^2 = 729) :
  (Real.log (x / y) / Real.log 3)^2 = (206 - 90 * Real.sqrt 5) / 4 := by
sorry

end logarithm_problem_l1897_189791


namespace pentagon_segment_parallel_and_length_l1897_189717

/-- Given a pentagon ABCDE with points P, Q, R, S on its sides and points M, N on PR and QS respectively,
    satisfying specific ratios, prove that MN is parallel to AE and its length is AE / ((k₁ + 1)(k₂ + 1)). -/
theorem pentagon_segment_parallel_and_length 
  (A B C D E P Q R S M N : ℝ × ℝ) (k₁ k₂ : ℝ) :
  -- Pentagon ABCDE
  -- Points P, Q, R, S on sides AB, BC, CD, DE respectively
  (P.1 - A.1) / (B.1 - P.1) = k₁ ∧ 
  (P.2 - A.2) / (B.2 - P.2) = k₁ ∧
  (Q.1 - B.1) / (C.1 - Q.1) = k₂ ∧
  (Q.2 - B.2) / (C.2 - Q.2) = k₂ ∧
  (R.1 - D.1) / (C.1 - R.1) = k₁ ∧
  (R.2 - D.2) / (C.2 - R.2) = k₁ ∧
  (S.1 - E.1) / (D.1 - S.1) = k₂ ∧
  (S.2 - E.2) / (D.2 - S.2) = k₂ ∧
  -- Points M and N on PR and QS respectively
  (M.1 - P.1) / (R.1 - M.1) = k₂ ∧
  (M.2 - P.2) / (R.2 - M.2) = k₂ ∧
  (N.1 - S.1) / (Q.1 - N.1) = k₁ ∧
  (N.2 - S.2) / (Q.2 - N.2) = k₁ →
  -- MN is parallel to AE
  (N.2 - M.2) / (N.1 - M.1) = (E.2 - A.2) / (E.1 - A.1) ∧
  -- Length of MN
  Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) = 
    Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) / ((k₁ + 1) * (k₂ + 1)) := by
  sorry

end pentagon_segment_parallel_and_length_l1897_189717


namespace rupert_candles_l1897_189795

/-- Given that Peter has 10 candles on his cake and Rupert is 3.5 times older than Peter,
    prove that Rupert's cake will have 35 candles. -/
theorem rupert_candles (peter_candles : ℕ) (age_ratio : ℚ) : ℕ :=
  by
  -- Define Peter's candles
  have h1 : peter_candles = 10 := by sorry
  -- Define the age ratio between Rupert and Peter
  have h2 : age_ratio = 3.5 := by sorry
  -- Calculate Rupert's candles
  have h3 : ↑peter_candles * age_ratio = 35 := by sorry
  -- Prove that Rupert's candles equal 35
  exact 35

end rupert_candles_l1897_189795


namespace cauchy_inequality_and_minimum_value_l1897_189772

theorem cauchy_inequality_and_minimum_value (a b x y : ℝ) :
  (a^2 + b^2) * (x^2 + y^2) ≥ (a*x + b*y)^2 ∧
  (x^2 + y^2 = 2 ∧ |x| ≠ |y| → ∃ (min : ℝ), min = 50/9 ∧ ∀ z, z = 1/(9*x^2) + 9/y^2 → z ≥ min) :=
by sorry

end cauchy_inequality_and_minimum_value_l1897_189772


namespace total_crayons_l1897_189764

/-- The total number of crayons given the specified box counts and colors -/
theorem total_crayons : 
  let orange_boxes : ℕ := 6
  let orange_per_box : ℕ := 8
  let blue_boxes : ℕ := 7
  let blue_per_box : ℕ := 5
  let red_boxes : ℕ := 1
  let red_per_box : ℕ := 11
  orange_boxes * orange_per_box + blue_boxes * blue_per_box + red_boxes * red_per_box = 94 :=
by sorry

end total_crayons_l1897_189764


namespace number_above_196_l1897_189719

/-- Represents the number of elements in the k-th row of the array -/
def elementsInRow (k : ℕ) : ℕ := 2 * k - 1

/-- Represents the sum of elements up to and including the k-th row -/
def sumUpToRow (k : ℕ) : ℕ := k^2

/-- Represents the first element in the k-th row -/
def firstElementInRow (k : ℕ) : ℕ := (k - 1)^2 + 1

/-- Represents the last element in the k-th row -/
def lastElementInRow (k : ℕ) : ℕ := k^2

/-- The theorem to be proved -/
theorem number_above_196 :
  ∃ (k : ℕ), 
    sumUpToRow k ≥ 196 ∧
    sumUpToRow (k-1) < 196 ∧
    lastElementInRow (k-1) = 169 := by
  sorry

end number_above_196_l1897_189719


namespace super_soup_stores_l1897_189763

/-- The number of stores Super Soup had at the end of 2020 -/
def final_stores : ℕ :=
  let initial_stores : ℕ := 23
  let opened_2019 : ℕ := 5
  let closed_2019 : ℕ := 2
  let opened_2020 : ℕ := 10
  let closed_2020 : ℕ := 6
  initial_stores + (opened_2019 - closed_2019) + (opened_2020 - closed_2020)

/-- Theorem stating that the final number of stores is 30 -/
theorem super_soup_stores : final_stores = 30 := by
  sorry

end super_soup_stores_l1897_189763


namespace solution_line_correct_l1897_189736

/-- Given two lines in the plane -/
def line1 : ℝ → ℝ → Prop := λ x y => 4*x + 3*y - 1 = 0
def line2 : ℝ → ℝ → Prop := λ x y => x + 2*y + 1 = 0

/-- The line to which our solution should be perpendicular -/
def perp_line : ℝ → ℝ → Prop := λ x y => x - 2*y - 1 = 0

/-- The proposed solution line -/
def solution_line : ℝ → ℝ → Prop := λ x y => 2*x + y - 1 = 0

/-- The intersection point of line1 and line2 -/
def intersection_point : ℝ × ℝ := (1, -1)

theorem solution_line_correct :
  (∀ x y, line1 x y ∧ line2 x y → solution_line x y) ∧
  (∀ m₁ m₂, (∀ x y, perp_line x y ↔ y = m₁ * x + m₂) →
            (∀ x y, solution_line x y ↔ y = (-1/m₁) * x + m₂) →
            m₁ * (-1/m₁) = -1) :=
sorry

end solution_line_correct_l1897_189736


namespace weight_calculation_l1897_189731

theorem weight_calculation (a b c : ℝ) (h : (a + b + c + (a + b) + (b + c) + (c + a) + (a + b + c)) / 7 = 95.42857142857143) : 
  a + b + c = 222.66666666666666 := by
  sorry

end weight_calculation_l1897_189731


namespace largest_number_l1897_189727

theorem largest_number (S : Set ℝ := {-5, 0, 3, 1/3}) : 
  ∃ m ∈ S, ∀ x ∈ S, x ≤ m ∧ m = 3 :=
by sorry

end largest_number_l1897_189727


namespace specific_pyramid_volume_l1897_189768

/-- A pyramid with an isosceles triangular base and equal lateral edges -/
structure IsoscelesPyramid where
  base_length : ℝ
  base_height : ℝ
  lateral_edge : ℝ

/-- The volume of an isosceles pyramid -/
def volume (p : IsoscelesPyramid) : ℝ := sorry

/-- Theorem: The volume of a specific isosceles pyramid is 108 -/
theorem specific_pyramid_volume :
  let p : IsoscelesPyramid := {
    base_length := 6,
    base_height := 9,
    lateral_edge := 13
  }
  volume p = 108 := by sorry

end specific_pyramid_volume_l1897_189768


namespace multiple_of_five_last_digit_l1897_189724

def is_multiple_of_five (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

def last_digit (n : ℕ) : ℕ := n % 10

def five_digit_number (d : ℕ) : ℕ := 45670 + d

theorem multiple_of_five_last_digit (d : ℕ) (h : d < 10) : 
  is_multiple_of_five (five_digit_number d) ↔ (d = 0 ∨ d = 5) :=
sorry

end multiple_of_five_last_digit_l1897_189724


namespace liz_team_final_deficit_l1897_189718

/-- Calculates the final deficit for Liz's team given the initial deficit and points scored in the final quarter -/
def final_deficit (initial_deficit : ℕ) (liz_free_throws : ℕ) (liz_three_pointers : ℕ) (liz_jump_shots : ℕ) (other_team_points : ℕ) : ℕ :=
  let liz_points := liz_free_throws + 3 * liz_three_pointers + 2 * liz_jump_shots
  let point_difference := liz_points - other_team_points
  initial_deficit - point_difference

theorem liz_team_final_deficit :
  final_deficit 20 5 3 4 10 = 8 := by
  sorry

end liz_team_final_deficit_l1897_189718


namespace set_operations_l1897_189753

def U : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def A : Finset Nat := {1, 3, 5, 8, 9}
def B : Finset Nat := {2, 5, 6, 8, 10}

theorem set_operations :
  (A ∪ B = {1, 2, 3, 5, 6, 8, 9, 10}) ∧
  ((U \ A) ∩ (U \ B) = {4, 7}) ∧
  (A \ B = {1, 3, 9}) :=
by sorry

end set_operations_l1897_189753


namespace min_ratio_rectangle_l1897_189769

theorem min_ratio_rectangle (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ k : ℝ, k > 0 → ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = k * a * b ∧ x + y = k * (a + b)) →
  (∃ k₀ : ℝ, k₀ > 0 ∧
    (∀ k : ℝ, k > 0 → (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = k * a * b ∧ x + y = k * (a + b)) → k ≥ k₀) ∧
    k₀ = 4 * a * b / ((a + b) ^ 2)) :=
sorry

end min_ratio_rectangle_l1897_189769


namespace star_equation_solution_l1897_189750

/-- The star operation defined as a * b = ab + 2b - a -/
def star (a b : ℝ) : ℝ := a * b + 2 * b - a

/-- Theorem stating that if 3 * x = 27 under the star operation, then x = 6 -/
theorem star_equation_solution :
  ∀ x : ℝ, star 3 x = 27 → x = 6 := by
  sorry

end star_equation_solution_l1897_189750


namespace room_problem_l1897_189789

theorem room_problem (boys girls : ℕ) : 
  boys = 3 * girls ∧ 
  (boys - 4) = 5 * (girls - 4) →
  boys + girls = 32 :=
by sorry

end room_problem_l1897_189789


namespace rectangle_area_divisible_by_12_l1897_189766

theorem rectangle_area_divisible_by_12 (x y z : ℤ) 
  (h : x^2 + y^2 = z^2) : 
  ∃ k : ℤ, x * y = 12 * k := by
sorry

end rectangle_area_divisible_by_12_l1897_189766


namespace solution_a_amount_l1897_189740

/-- Proves that the amount of Solution A used is 100 milliliters -/
theorem solution_a_amount (solution_a : ℝ) (solution_b : ℝ) : solution_a = 100 :=
  by
  -- Solution B is 500 milliliters more than Solution A
  have h1 : solution_b = solution_a + 500 := by sorry
  
  -- Solution A is 16% alcohol
  have h2 : solution_a * 0.16 = solution_a * (16 / 100) := by sorry
  
  -- Solution B is 10% alcohol
  have h3 : solution_b * 0.10 = solution_b * (10 / 100) := by sorry
  
  -- The resulting mixture has 76 milliliters of pure alcohol
  have h4 : solution_a * (16 / 100) + solution_b * (10 / 100) = 76 := by sorry
  
  sorry -- Skip the proof


end solution_a_amount_l1897_189740


namespace george_marbles_l1897_189710

/-- The number of red marbles in George's collection --/
def red_marbles (total : ℕ) (white : ℕ) (yellow : ℕ) (green : ℕ) : ℕ :=
  total - (white + yellow + green)

/-- Theorem stating the number of red marbles in George's collection --/
theorem george_marbles :
  let total := 50
  let white := total / 2
  let yellow := 12
  let green := yellow - yellow / 2
  red_marbles total white yellow green = 7 := by
  sorry

end george_marbles_l1897_189710


namespace compound_hydrogen_atoms_l1897_189714

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight oxygenWeight hydrogenWeight : ℝ) : ℝ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

theorem compound_hydrogen_atoms :
  ∀ (c : Compound),
    c.carbon = 4 →
    c.oxygen = 2 →
    molecularWeight c 12.01 16.00 1.008 = 88 →
    c.hydrogen = 8 := by
  sorry

end compound_hydrogen_atoms_l1897_189714


namespace point_C_coordinates_l1897_189759

/-- Given points O, A, B, and C in a 2D plane, prove that C has specific coordinates. -/
theorem point_C_coordinates (O A B C : ℝ × ℝ) : 
  (A.1 - O.1 = -3 ∧ A.2 - O.2 = 1) →  -- OA = (-3, 1)
  (B.1 - O.1 = 0 ∧ B.2 - O.2 = 5) →   -- OB = (0, 5)
  (∃ k : ℝ, C.1 - A.1 = k * (B.1 - O.1) ∧ C.2 - A.2 = k * (B.2 - O.2)) →  -- AC parallel to OB
  ((C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 0) →  -- BC perpendicular to AB
  C = (-3, 29/4) := by
sorry

end point_C_coordinates_l1897_189759


namespace complex_arithmetic_l1897_189794

/-- Given complex numbers B, Q, R, and T, prove that 2(B - Q + R + T) = 18 + 10i -/
theorem complex_arithmetic (B Q R T : ℂ) 
  (hB : B = 3 + 2*I)
  (hQ : Q = -5)
  (hR : R = -2*I)
  (hT : T = 1 + 5*I) :
  2 * (B - Q + R + T) = 18 + 10*I :=
by sorry

end complex_arithmetic_l1897_189794


namespace target_miss_probability_l1897_189788

theorem target_miss_probability (p_I p_II p_III : ℝ) 
  (h_I : p_I = 0.35) 
  (h_II : p_II = 0.30) 
  (h_III : p_III = 0.25) : 
  1 - (p_I + p_II + p_III) = 0.10 := by
  sorry

end target_miss_probability_l1897_189788


namespace painting_price_l1897_189704

theorem painting_price (num_paintings : ℕ) (num_toys : ℕ) (toy_price : ℝ) 
  (painting_discount : ℝ) (toy_discount : ℝ) (total_loss : ℝ) :
  num_paintings = 10 →
  num_toys = 8 →
  toy_price = 20 →
  painting_discount = 0.1 →
  toy_discount = 0.15 →
  total_loss = 64 →
  ∃ (painting_price : ℝ),
    painting_price * num_paintings + toy_price * num_toys -
    (painting_price * (1 - painting_discount) * num_paintings + 
     toy_price * (1 - toy_discount) * num_toys) = total_loss ∧
    painting_price = 40 := by
  sorry

end painting_price_l1897_189704


namespace cube_root_of_three_twos_to_seven_l1897_189783

theorem cube_root_of_three_twos_to_seven (x : ℝ) :
  x = (2^7 + 2^7 + 2^7)^(1/3) → x = 4 * (2^(1/3)) := by
  sorry

end cube_root_of_three_twos_to_seven_l1897_189783


namespace donna_marcia_total_pencils_l1897_189754

/-- The number of pencils bought by Cindi -/
def cindi_pencils : ℕ := 60

/-- The number of pencils bought by Marcia -/
def marcia_pencils : ℕ := 2 * cindi_pencils

/-- The number of pencils bought by Donna -/
def donna_pencils : ℕ := 3 * marcia_pencils

/-- The total number of pencils bought by Donna and Marcia -/
def total_pencils : ℕ := donna_pencils + marcia_pencils

theorem donna_marcia_total_pencils :
  total_pencils = 480 := by
  sorry

end donna_marcia_total_pencils_l1897_189754


namespace max_pieces_from_cake_l1897_189721

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 21

/-- The size of the small cake pieces in inches -/
def small_piece_size : ℕ := 3

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := (large_cake_size * large_cake_size) / (small_piece_size * small_piece_size)

theorem max_pieces_from_cake : max_pieces = 49 := by
  sorry

end max_pieces_from_cake_l1897_189721


namespace field_trip_students_l1897_189739

/-- Given a field trip with buses, prove the number of students. -/
theorem field_trip_students (seats_per_bus : ℕ) (num_buses : ℕ) : 
  seats_per_bus = 3 → num_buses = 3 → seats_per_bus * num_buses = 9 := by
  sorry

#check field_trip_students

end field_trip_students_l1897_189739


namespace part_one_part_two_l1897_189782

-- Define the propositions p and q
def p (a x : ℝ) : Prop := a < x ∧ x < 3 * a

def q (x : ℝ) : Prop := x^2 - 5*x + 6 < 0

-- Part 1
theorem part_one (x : ℝ) : 
  p 1 x ∧ q x → 2 < x ∧ x < 3 :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  (a > 0) → (∀ x : ℝ, q x → p a x) → 1 ≤ a ∧ a ≤ 2 :=
sorry

end part_one_part_two_l1897_189782


namespace common_factor_proof_l1897_189758

theorem common_factor_proof (m : ℝ) : ∃ (k₁ k₂ : ℝ), 
  m^2 - 4 = (m - 2) * k₁ ∧ m^2 - 4*m + 4 = (m - 2) * k₂ := by
  sorry

end common_factor_proof_l1897_189758


namespace stream_speed_l1897_189762

/-- Proves that the speed of a stream is 135/14 km/h given the conditions of a boat's travel --/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) 
  (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : downstream_distance = 150)
  (h2 : upstream_distance = 75)
  (h3 : downstream_time = 5)
  (h4 : upstream_time = 7) : 
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧
    stream_speed = 135 / 14 := by
  sorry

end stream_speed_l1897_189762


namespace batsman_average_increase_l1897_189760

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the increase in average between two BatsmanPerformance instances -/
def averageIncrease (before after : BatsmanPerformance) : ℚ :=
  after.average - before.average

theorem batsman_average_increase
  (performance16 : BatsmanPerformance)
  (performance17 : BatsmanPerformance)
  (h1 : performance17.innings = performance16.innings + 1)
  (h2 : performance17.innings = 17)
  (h3 : performance17.totalRuns = performance16.totalRuns + 82)
  (h4 : performance17.average = 34)
  : averageIncrease performance16 performance17 = 3 := by
  sorry

end batsman_average_increase_l1897_189760


namespace other_sales_percentage_l1897_189761

theorem other_sales_percentage (pen_sales pencil_sales notebook_sales : ℝ) 
  (h_pen : pen_sales = 20)
  (h_pencil : pencil_sales = 15)
  (h_notebook : notebook_sales = 30)
  (h_total : pen_sales + pencil_sales + notebook_sales + 100 - (pen_sales + pencil_sales + notebook_sales) = 100) :
  100 - (pen_sales + pencil_sales + notebook_sales) = 35 := by
  sorry

end other_sales_percentage_l1897_189761


namespace min_value_fraction_sum_l1897_189700

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  1 / (a + 1) + 1 / (b + 2) ≥ 1 / 2 := by
  sorry

end min_value_fraction_sum_l1897_189700


namespace calculation_difference_l1897_189735

/-- The correct calculation of 12 - (3 × 2) + 4 -/
def H : Int := 12 - (3 * 2) + 4

/-- The incorrect calculation of 12 - 3 × 2 + 4 (ignoring parentheses) -/
def P : Int := 12 - 3 * 2 + 4

/-- The difference between the correct and incorrect calculations -/
def difference : Int := H - P

/-- Theorem stating that the difference between the correct and incorrect calculations is -12 -/
theorem calculation_difference : difference = -12 := by
  sorry

end calculation_difference_l1897_189735


namespace f_greater_than_one_max_a_for_derivative_inequality_l1897_189702

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - (1/2) * a * x^2

theorem f_greater_than_one (x : ℝ) (hx : x > 0) : f 2 x > 1 := by sorry

theorem max_a_for_derivative_inequality :
  ∃ (a : ℕ), a = 2 ∧
  (∀ (x : ℝ), x > 0 → deriv (f a) x ≥ x^2 * log x) ∧
  (∀ (b : ℕ), b > 2 → ∃ (x : ℝ), x > 0 ∧ deriv (f b) x < x^2 * log x) := by sorry

end f_greater_than_one_max_a_for_derivative_inequality_l1897_189702


namespace extended_triangle_properties_l1897_189744

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumcircle of a triangle
def circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Define the inradius of a triangle
def inradius (t : Triangle) : ℝ := sorry

-- Define the extended triangle DEF
def extendedTriangle (t : Triangle) : Triangle := sorry

-- Theorem statement
theorem extended_triangle_properties (t : Triangle) :
  let t' := extendedTriangle t
  perimeter t' ≥ perimeter t ∧ inradius t' > inradius t := by sorry

end extended_triangle_properties_l1897_189744


namespace count_squares_five_by_five_l1897_189756

/-- Represents a square grid with a dot in the center -/
structure CenteredGrid (n : ℕ) :=
  (size : ℕ)
  (has_center_dot : size % 2 = 1)

/-- Counts the number of squares in a grid that contain the center dot -/
def count_squares_with_dot (grid : CenteredGrid 5) : ℕ :=
  let center := grid.size / 2
  let count_for_size (k : ℕ) : ℕ :=
    if k ≤ grid.size
    then (min (center + 1) (grid.size - k + 1))^2
    else 0
  (List.range grid.size).map count_for_size |> List.sum

/-- The main theorem to prove -/
theorem count_squares_five_by_five :
  ∀ (grid : CenteredGrid 5), count_squares_with_dot grid = 19 :=
sorry

end count_squares_five_by_five_l1897_189756


namespace plane_perp_from_line_relations_l1897_189730

/-- Two lines are parallel -/
def parallel_lines (m n : Line) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (m : Line) (α : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def perp_planes (α β : Plane) : Prop := sorry

/-- Main theorem: If m is parallel to n, m is contained in α, and n is perpendicular to β, then α is perpendicular to β -/
theorem plane_perp_from_line_relations 
  (m n : Line) (α β : Plane) 
  (h1 : parallel_lines m n) 
  (h2 : line_in_plane m α) 
  (h3 : line_perp_plane n β) : 
  perp_planes α β := by sorry

end plane_perp_from_line_relations_l1897_189730


namespace three_propositions_true_l1897_189741

-- Define reciprocals
def are_reciprocals (x y : ℝ) : Prop := x * y = 1

-- Define triangle area and congruence
def triangle_area (t : Set ℝ × Set ℝ × Set ℝ) : ℝ := sorry
def triangle_congruent (t1 t2 : Set ℝ × Set ℝ × Set ℝ) : Prop := sorry

-- Define quadratic equation solution existence
def has_real_solutions (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0

theorem three_propositions_true : 
  (∀ x y : ℝ, are_reciprocals x y → x * y = 1) ∧ 
  (∃ t1 t2 : Set ℝ × Set ℝ × Set ℝ, triangle_area t1 = triangle_area t2 ∧ ¬ triangle_congruent t1 t2) ∧
  (∀ m : ℝ, ¬ has_real_solutions m → m > 1) := by
  sorry

end three_propositions_true_l1897_189741


namespace fraction_problem_l1897_189757

theorem fraction_problem (p q x y : ℚ) : 
  p / q = 4 / 5 →
  x / y + (2 * q - p) / (2 * q + p) = 1 →
  x / y = 4 / 7 := by
sorry

end fraction_problem_l1897_189757


namespace pq_length_is_ten_l1897_189709

/-- A trapezoid PQRS with specific properties -/
structure Trapezoid where
  /-- The length of side RS -/
  rs : ℝ
  /-- The tangent of angle S -/
  tan_s : ℝ
  /-- The tangent of angle Q -/
  tan_q : ℝ
  /-- PQ is parallel to RS -/
  pq_parallel_rs : True
  /-- PR is perpendicular to RS -/
  pr_perpendicular_rs : True
  /-- RS has length 15 -/
  rs_length : rs = 15
  /-- tan S = 2 -/
  tan_s_value : tan_s = 2
  /-- tan Q = 3 -/
  tan_q_value : tan_q = 3

/-- The length of PQ in the trapezoid -/
def pq_length (t : Trapezoid) : ℝ := 10

/-- Theorem stating that the length of PQ is 10 -/
theorem pq_length_is_ten (t : Trapezoid) : pq_length t = 10 := by sorry

end pq_length_is_ten_l1897_189709


namespace max_tan_A_in_triangle_l1897_189723

/-- Given a triangle ABC where sin A + 2sin B cos C = 0, the maximum value of tan A is 1/√3 -/
theorem max_tan_A_in_triangle (A B C : Real) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) (h5 : Real.sin A + 2 * Real.sin B * Real.cos C = 0) : 
  (∀ A' B' C' : Real, 0 < A' ∧ A' < π → 0 < B' ∧ B' < π → 0 < C' ∧ C' < π → 
   A' + B' + C' = π → Real.sin A' + 2 * Real.sin B' * Real.cos C' = 0 → 
   Real.tan A' ≤ Real.tan A) → Real.tan A = 1 / Real.sqrt 3 :=
by sorry

end max_tan_A_in_triangle_l1897_189723


namespace z_plus_inv_z_ellipse_l1897_189712

theorem z_plus_inv_z_ellipse (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), (z + z⁻¹ = x + y * I) →
  (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end z_plus_inv_z_ellipse_l1897_189712


namespace prom_color_assignment_l1897_189725

-- Define the colors
inductive Color
| White
| Red
| Blue

-- Define a person's outfit
structure Outfit :=
  (dress : Color)
  (shoes : Color)

-- Define the problem statement
theorem prom_color_assignment :
  ∀ (tamara valya lida : Outfit),
    -- Only Tamara's dress and shoes were the same color
    (tamara.dress = tamara.shoes) ∧
    (valya.dress ≠ valya.shoes ∨ lida.dress ≠ lida.shoes) →
    -- Valya was in white shoes
    (valya.shoes = Color.White) →
    -- Neither Lida's dress nor her shoes were red
    (lida.dress ≠ Color.Red ∧ lida.shoes ≠ Color.Red) →
    -- All colors are used exactly once for dresses
    (tamara.dress ≠ valya.dress ∧ tamara.dress ≠ lida.dress ∧ valya.dress ≠ lida.dress) →
    -- All colors are used exactly once for shoes
    (tamara.shoes ≠ valya.shoes ∧ tamara.shoes ≠ lida.shoes ∧ valya.shoes ≠ lida.shoes) →
    -- The only valid assignment is:
    (tamara = ⟨Color.Red, Color.Red⟩ ∧
     valya = ⟨Color.Blue, Color.White⟩ ∧
     lida = ⟨Color.White, Color.Blue⟩) :=
by sorry

end prom_color_assignment_l1897_189725


namespace cookie_count_l1897_189726

/-- Given a set of bags filled with cookies, where each bag contains 3 cookies
    and there are 25 bags in total, the total number of cookies is 75. -/
theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) : 
  bags = 25 → cookies_per_bag = 3 → bags * cookies_per_bag = 75 := by
  sorry

end cookie_count_l1897_189726


namespace hyperbola_real_axis_length_l1897_189701

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := 2 * x^2 - y^2 = 8

/-- The length of the real axis of the hyperbola -/
def real_axis_length : ℝ := 4

/-- Theorem: The length of the real axis of the hyperbola 2x^2 - y^2 = 8 is 4 -/
theorem hyperbola_real_axis_length :
  ∀ x y : ℝ, hyperbola_eq x y → real_axis_length = 4 := by
  sorry

end hyperbola_real_axis_length_l1897_189701


namespace cubic_repeated_roots_l1897_189777

theorem cubic_repeated_roots (p q : ℝ) :
  (∃ (x : ℝ), (x^3 + p*x + q = 0 ∧ ∃ (y : ℝ), y ≠ x ∧ y^3 + p*y + q = 0 ∧ (∀ (z : ℝ), z^3 + p*z + q = 0 → z = x ∨ z = y))) ↔
  (q^2 / 4 + p^3 / 27 = 0) := by
sorry

end cubic_repeated_roots_l1897_189777


namespace complex_equation_solution_l1897_189765

theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I)^2 / z = 1 + Complex.I) : 
  z = -1 - Complex.I := by
sorry

end complex_equation_solution_l1897_189765


namespace campers_rowing_morning_l1897_189787

theorem campers_rowing_morning (total_rowing : ℕ) (afternoon_rowing : ℕ) 
  (h1 : total_rowing = 34) 
  (h2 : afternoon_rowing = 21) : 
  total_rowing - afternoon_rowing = 13 := by
  sorry

end campers_rowing_morning_l1897_189787


namespace characterization_of_satisfying_polynomials_l1897_189738

/-- A polynomial that satisfies the given functional equation. -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (x - 1) * P (x + 1) - (x + 2) * P x = 0

/-- The theorem stating the form of polynomials satisfying the functional equation. -/
theorem characterization_of_satisfying_polynomials :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P →
  ∃ a : ℝ, ∀ x : ℝ, P x = a * (x^3 - x) :=
sorry

end characterization_of_satisfying_polynomials_l1897_189738


namespace age_difference_l1897_189705

/-- The difference in total ages of (A, B) and (B, C) given C is 20 years younger than A -/
theorem age_difference (A B C : ℕ) (h : C = A - 20) : 
  (A + B) - (B + C) = 20 := by
  sorry

end age_difference_l1897_189705
