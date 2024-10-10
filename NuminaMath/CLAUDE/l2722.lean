import Mathlib

namespace quadratic_roots_condition_necessary_not_sufficient_condition_l2722_272220

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - 2*a*x + 2*a^2 - a - 6

-- Define the proposition p
def has_real_roots (a : ℝ) : Prop := ∃ x : ℝ, quadratic a x = 0

-- Define the proposition q
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 3

theorem quadratic_roots_condition (a : ℝ) :
  ¬(has_real_roots a) ↔ (a < -2 ∨ a > 3) := by sorry

theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ a : ℝ, q m a → has_real_roots a) ∧
  (∃ a : ℝ, has_real_roots a ∧ ¬(q m a)) →
  -1 ≤ m ∧ m < 0 := by sorry

end quadratic_roots_condition_necessary_not_sufficient_condition_l2722_272220


namespace f_value_at_2013_l2722_272224

theorem f_value_at_2013 (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x^3 + b * Real.sin x + 9) →
  f (-2013) = 7 →
  f 2013 = 11 := by
sorry

end f_value_at_2013_l2722_272224


namespace fraction_invariance_l2722_272292

theorem fraction_invariance (x y : ℝ) : 
  (2 * x) / (3 * x - y) = (2 * (3 * x)) / (3 * (3 * x) - (3 * y)) :=
by sorry

end fraction_invariance_l2722_272292


namespace min_difference_of_composite_functions_l2722_272287

open Real

theorem min_difference_of_composite_functions :
  let f : ℝ → ℝ := λ x ↦ Real.exp (3 * x - 1)
  let g : ℝ → ℝ := λ x ↦ 1 / 3 + Real.log x
  ∃ (min_diff : ℝ), min_diff = (2 + Real.log 3) / 3 ∧
    ∀ m n : ℝ, f m = g n → n - m ≥ min_diff :=
by sorry

end min_difference_of_composite_functions_l2722_272287


namespace remainder_3_pow_2023_mod_17_l2722_272231

theorem remainder_3_pow_2023_mod_17 : 3^2023 % 17 = 7 := by
  sorry

end remainder_3_pow_2023_mod_17_l2722_272231


namespace sum_of_solutions_is_zero_l2722_272225

theorem sum_of_solutions_is_zero (x : ℝ) :
  ((-12 * x) / (x^2 - 1) = (3 * x) / (x + 1) - 9 / (x - 1)) →
  (∃ y : ℝ, (-12 * y) / (y^2 - 1) = (3 * y) / (y + 1) - 9 / (y - 1) ∧ y ≠ x) →
  x + y = 0 :=
by sorry

end sum_of_solutions_is_zero_l2722_272225


namespace sues_necklace_beads_l2722_272201

/-- The number of beads in Sue's necklace -/
def total_beads (purple blue green : ℕ) : ℕ := purple + blue + green

/-- Theorem stating the total number of beads in Sue's necklace -/
theorem sues_necklace_beads : 
  ∀ (purple blue green : ℕ),
  purple = 7 →
  blue = 2 * purple →
  green = blue + 11 →
  total_beads purple blue green = 46 := by
sorry

end sues_necklace_beads_l2722_272201


namespace complex_product_equals_24_plus_18i_l2722_272208

/-- Complex number multiplication -/
def complex_mult (a b c d : ℤ) : ℤ × ℤ :=
  (a * c - b * d, a * d + b * c)

/-- The imaginary unit i -/
def i : ℤ × ℤ := (0, 1)

theorem complex_product_equals_24_plus_18i : 
  complex_mult 3 (-4) 0 6 = (24, 18) := by sorry

end complex_product_equals_24_plus_18i_l2722_272208


namespace two_digit_number_difference_l2722_272248

theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → x ≠ 0 → 
  (10 * x + y) - (10 * y + x) = 72 →
  x - y = 8 := by sorry

end two_digit_number_difference_l2722_272248


namespace dream_car_cost_proof_l2722_272209

/-- Calculates the cost of a dream car given monthly earnings, savings, and total earnings before purchase. -/
def dream_car_cost (monthly_earnings : ℕ) (monthly_savings : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings / monthly_earnings) * monthly_savings

/-- Proves that the cost of the dream car is £45,000 given the specified conditions. -/
theorem dream_car_cost_proof :
  dream_car_cost 4000 500 360000 = 45000 := by
  sorry

end dream_car_cost_proof_l2722_272209


namespace cube_root_x_plus_3y_equals_3_l2722_272226

theorem cube_root_x_plus_3y_equals_3 (x y : ℝ) 
  (h : y = Real.sqrt (3 - x) + Real.sqrt (x - 3) + 8) :
  (x + 3 * y) ^ (1/3 : ℝ) = 3 := by
  sorry

end cube_root_x_plus_3y_equals_3_l2722_272226


namespace range_of_a_l2722_272253

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 8*x - 20 > 0 → x^2 - 2*x + 1 - a^2 > 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x + 1 - a^2 > 0 ∧ x^2 - 8*x - 20 ≤ 0) ∧ 
  a > 0 
  ↔ 0 < a ∧ a ≤ 3 := by sorry

end range_of_a_l2722_272253


namespace sixth_term_of_geometric_sequence_l2722_272211

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

-- Theorem statement
theorem sixth_term_of_geometric_sequence 
  (a₁ a₂ : ℝ) 
  (h₁ : a₁ = 5) 
  (h₂ : a₂ = 15) : 
  geometric_sequence a₁ (a₂ / a₁) 6 = 1215 := by
sorry

end sixth_term_of_geometric_sequence_l2722_272211


namespace multiply_by_seven_equals_98_l2722_272274

theorem multiply_by_seven_equals_98 (x : ℝ) : x * 7 = 98 ↔ x = 14 := by
  sorry

end multiply_by_seven_equals_98_l2722_272274


namespace positive_root_implies_m_value_l2722_272286

theorem positive_root_implies_m_value 
  (h : ∃ (x : ℝ), x > 0 ∧ (6 - x) / (x - 3) - (2 * m) / (x - 3) = 0) : 
  m = 3/2 := by
  sorry

end positive_root_implies_m_value_l2722_272286


namespace sum_congruence_modulo_9_l2722_272202

theorem sum_congruence_modulo_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end sum_congruence_modulo_9_l2722_272202


namespace quadratic_inequality_implies_range_l2722_272218

theorem quadratic_inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0) →
  -1 < a ∧ a ≤ 0 :=
by sorry

end quadratic_inequality_implies_range_l2722_272218


namespace sequence_properties_l2722_272217

def sequence_a (n : ℕ+) : ℚ := (1/2) ^ (n.val - 2)

def sum_S (n : ℕ+) : ℚ := 4 * (1 - (1/2) ^ n.val)

theorem sequence_properties :
  ∀ (n : ℕ+),
  (∀ (m : ℕ+), sum_S (m + 1) = (1/2) * sum_S m + 2) →
  sequence_a 1 = 2 →
  sequence_a 2 = 1 →
  (∀ (k : ℕ+), sequence_a k = (1/2) ^ (k.val - 2)) ∧
  (∀ (t : ℕ+), (∀ (n : ℕ+), (sequence_a t * sum_S (n + 1) - 1) / (sequence_a t * sequence_a (n + 1) - 1) < 1/2) ↔ (t = 3 ∨ t = 4)) ∧
  (∀ (m n k : ℕ+), m ≠ n → n ≠ k → m ≠ k → sequence_a m + sequence_a n ≠ sequence_a k) :=
by sorry

end sequence_properties_l2722_272217


namespace simplify_and_rationalize_l2722_272262

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 2) * (Real.sqrt 9 / Real.sqrt 13) * (Real.sqrt 22 / Real.sqrt 7) = 
  (3 * Real.sqrt 20020) / 182 := by sorry

end simplify_and_rationalize_l2722_272262


namespace complex_sum_magnitude_l2722_272297

theorem complex_sum_magnitude (a b c : ℂ) :
  Complex.abs a = 2 →
  Complex.abs b = 2 →
  Complex.abs c = 2 →
  a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = 0 →
  Complex.abs (a + b + c) = 6 + 2 * Real.sqrt 6 ∨
  Complex.abs (a + b + c) = 6 - 2 * Real.sqrt 6 := by
  sorry

end complex_sum_magnitude_l2722_272297


namespace polynomial_value_l2722_272234

theorem polynomial_value (x y : ℝ) (h : x - y = 1) :
  x^4 - x*y^3 - x^3*y - 3*x^2*y + 3*x*y^2 + y^4 = 1 := by
  sorry

end polynomial_value_l2722_272234


namespace buddy_cards_thursday_l2722_272267

def baseball_cards_problem (initial_cards : ℕ) (bought_wednesday : ℕ) : ℕ :=
  let tuesday_cards := initial_cards / 2
  let wednesday_cards := tuesday_cards + bought_wednesday
  let thursday_bought := tuesday_cards / 3
  wednesday_cards + thursday_bought

theorem buddy_cards_thursday (initial_cards : ℕ) (bought_wednesday : ℕ) 
  (h1 : initial_cards = 30) (h2 : bought_wednesday = 12) : 
  baseball_cards_problem initial_cards bought_wednesday = 32 := by
  sorry

end buddy_cards_thursday_l2722_272267


namespace art_gallery_theorem_l2722_272222

theorem art_gallery_theorem (total_pieces : ℕ) : 
  (total_pieces : ℚ) * (1 / 3) * (1 / 6) + 
  (total_pieces : ℚ) * (2 / 3) * (2 / 3) = 800 →
  total_pieces = 1800 := by
  sorry

end art_gallery_theorem_l2722_272222


namespace express_train_meetings_l2722_272254

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- The problem statement -/
theorem express_train_meetings :
  let travelTime : Nat := 210 -- 3 hours and 30 minutes in minutes
  let departureInterval : Nat := 60 -- 1 hour in minutes
  let firstDeparture : Time := ⟨6, 0⟩ -- 6:00 AM
  let expressDeparture : Time := ⟨9, 0⟩ -- 9:00 AM
  let expressArrival : Time := ⟨12, 30⟩ -- 12:30 PM (9:00 AM + 3h30m)
  
  (timeDifference firstDeparture expressDeparture / departureInterval + 1) -
  (timeDifference firstDeparture expressArrival / departureInterval + 1) = 6 := by
  sorry

end express_train_meetings_l2722_272254


namespace function_inequality_l2722_272260

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, (x - 1) * (deriv f x) ≤ 0)
variable (h2 : ∀ x, f (x + 1) = f (-x + 1))

-- Define the theorem
theorem function_inequality (x₁ x₂ : ℝ) (h3 : |x₁ - 1| < |x₂ - 1|) : 
  f (2 - x₁) ≥ f (2 - x₂) := by sorry

end function_inequality_l2722_272260


namespace unique_solution_for_exponential_equation_l2722_272296

theorem unique_solution_for_exponential_equation :
  ∀ n p : ℕ+,
    Nat.Prime p →
    3^(p : ℕ) - n * p = n + p →
    n = 6 ∧ p = 3 := by
  sorry

end unique_solution_for_exponential_equation_l2722_272296


namespace dessert_coffee_probability_l2722_272271

theorem dessert_coffee_probability
  (p_dessert_and_coffee : ℝ)
  (p_no_dessert : ℝ)
  (h1 : p_dessert_and_coffee = 0.6)
  (h2 : p_no_dessert = 0.2500000000000001) :
  p_dessert_and_coffee + (1 - p_no_dessert - p_dessert_and_coffee) = 0.75 :=
by sorry

end dessert_coffee_probability_l2722_272271


namespace sum_row_10_pascal_l2722_272289

/-- Sum of numbers in a row of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

/-- Row 10 of Pascal's Triangle -/
def row_10 : ℕ := 10

theorem sum_row_10_pascal : pascal_row_sum row_10 = 1024 := by
  sorry

end sum_row_10_pascal_l2722_272289


namespace consecutive_integers_sum_l2722_272227

theorem consecutive_integers_sum (x : ℤ) : 
  x * (x + 1) * (x + 2) = 336 → x + (x + 1) + (x + 2) = 21 := by
sorry

end consecutive_integers_sum_l2722_272227


namespace rotate90_matches_optionC_l2722_272206

-- Define the plane
def Plane : Type := ℝ × ℝ

-- Define the X-like shape
def XLikeShape : Type := Set Plane

-- Define rotation function
def rotate90Clockwise (shape : XLikeShape) : XLikeShape := sorry

-- Define the original shape
def originalShape : XLikeShape := sorry

-- Define the shape in option C
def optionCShape : XLikeShape := sorry

-- Theorem statement
theorem rotate90_matches_optionC : 
  rotate90Clockwise originalShape = optionCShape := by sorry

end rotate90_matches_optionC_l2722_272206


namespace events_related_95_percent_confidence_l2722_272293

-- Define the confidence level
def confidence_level : ℝ := 0.95

-- Define the critical value for 95% confidence
def critical_value : ℝ := 3.841

-- Define the relation between events A and B
def events_related (K : ℝ) : Prop := K^2 > critical_value

-- Theorem statement
theorem events_related_95_percent_confidence (K : ℝ) :
  events_related K ↔ K^2 > critical_value :=
sorry

end events_related_95_percent_confidence_l2722_272293


namespace partial_fraction_decomposition_l2722_272246

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 7 → x ≠ -2 →
  (5 * x - 4) / (x^2 - 5*x - 14) = (31/9) / (x - 7) + (14/9) / (x + 2) := by
sorry

end partial_fraction_decomposition_l2722_272246


namespace tan_double_angle_l2722_272268

/-- Given an angle θ with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (-1, 2), prove that tan 2θ = 4/3 -/
theorem tan_double_angle (θ : ℝ) : 
  (∃ (x y : ℝ), x = -1 ∧ y = 2 ∧ Real.tan θ = y / x) → 
  Real.tan (2 * θ) = 4 / 3 := by
  sorry

end tan_double_angle_l2722_272268


namespace complement_A_intersect_B_a_greater_than_one_l2722_272259

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the first part of the problem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x | 7 < x ∧ x < 10} := by sorry

-- Theorem for the second part of the problem
theorem a_greater_than_one (h : A ∩ C a ≠ ∅) : a > 1 := by sorry

end complement_A_intersect_B_a_greater_than_one_l2722_272259


namespace stone_game_loser_l2722_272299

/-- Represents a pile of stones -/
structure Pile :=
  (count : Nat)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)
  (currentPlayer : Nat)

/-- Defines a valid move in the game -/
def validMove (state : GameState) : Prop :=
  ∃ (p : Pile) (n : Nat), p ∈ state.piles ∧ 1 ≤ n ∧ n < p.count

/-- The initial game state -/
def initialState : GameState :=
  { piles := [⟨6⟩, ⟨8⟩, ⟨8⟩, ⟨9⟩], currentPlayer := 1 }

/-- The number of players -/
def numPlayers : Nat := 5

/-- The losing player -/
def losingPlayer : Nat := 3

theorem stone_game_loser :
  ¬∃ (moves : Nat), 
    (moves + initialState.piles.length = (initialState.piles.map Pile.count).sum) ∧
    (moves % numPlayers + 1 = losingPlayer) ∧
    (∀ (state : GameState), state.piles.length ≤ moves + initialState.piles.length → validMove state) :=
sorry

end stone_game_loser_l2722_272299


namespace sum_of_two_numbers_l2722_272285

theorem sum_of_two_numbers (A B : ℝ) : 
  A - B = 8 → (A + B) / 4 = 6 → A = 16 → A + B = 24 := by sorry

end sum_of_two_numbers_l2722_272285


namespace cube_volume_l2722_272213

theorem cube_volume (edge_sum : ℝ) (h : edge_sum = 96) : 
  let edge_length := edge_sum / 12
  let volume := edge_length ^ 3
  volume = 512 := by sorry

end cube_volume_l2722_272213


namespace max_rectangles_6x6_grid_l2722_272295

/-- Counts the number of rectangles in a right triangle grid of size n x n -/
def count_rectangles (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

/-- The maximum number of rectangles in a 6x6 right triangle grid is 126 -/
theorem max_rectangles_6x6_grid :
  count_rectangles 6 = 126 := by sorry

end max_rectangles_6x6_grid_l2722_272295


namespace primes_up_to_100_l2722_272273

theorem primes_up_to_100 : 
  {p : ℕ | Nat.Prime p ∧ 2 ≤ p ∧ p ≤ 100} = 
  {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97} := by
  sorry

end primes_up_to_100_l2722_272273


namespace triangle_angle_A_l2722_272275

theorem triangle_angle_A (a b : ℝ) (B : Real) (A : Real) : 
  a = 4 → 
  b = 4 * Real.sqrt 3 → 
  B = 60 * π / 180 →
  (a / Real.sin A = b / Real.sin B) →
  A = 30 * π / 180 := by
sorry

end triangle_angle_A_l2722_272275


namespace age_equality_l2722_272277

/-- Proves that the number of years after which grandfather's age equals the sum of Xiaoming and father's ages is 14, given their current ages. -/
theorem age_equality (grandfather_age father_age xiaoming_age : ℕ) 
  (h1 : grandfather_age = 60)
  (h2 : father_age = 35)
  (h3 : xiaoming_age = 11) : 
  ∃ (years : ℕ), grandfather_age + years = (father_age + years) + (xiaoming_age + years) ∧ years = 14 := by
  sorry

end age_equality_l2722_272277


namespace multiple_y_solutions_l2722_272214

theorem multiple_y_solutions : ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧
  (∃ (x₁ : ℝ), x₁^2 + y₁^2 - 10 = 0 ∧ x₁^2 - x₁*y₁ - 3*y₁ + 12 = 0) ∧
  (∃ (x₂ : ℝ), x₂^2 + y₂^2 - 10 = 0 ∧ x₂^2 - x₂*y₂ - 3*y₂ + 12 = 0) :=
by sorry

end multiple_y_solutions_l2722_272214


namespace linear_system_solution_existence_l2722_272219

theorem linear_system_solution_existence
  (a b c d : ℤ)
  (h_nonzero : a * d - b * c ≠ 0)
  (b₁ b₂ : ℤ)
  (h_b₁ : ∃ k : ℤ, b₁ = (a * d - b * c) * k)
  (h_b₂ : ∃ q : ℤ, b₂ = (a * d - b * c) * q) :
  ∃ x y : ℤ, a * x + b * y = b₁ ∧ c * x + d * y = b₂ :=
sorry

end linear_system_solution_existence_l2722_272219


namespace square_difference_133_l2722_272261

theorem square_difference_133 : 
  ∃ (a b c d : ℕ), 
    a * a - b * b = 133 ∧ 
    c * c - d * d = 133 ∧ 
    a > b ∧ c > d ∧ 
    (a ≠ c ∨ b ≠ d) := by
  sorry

end square_difference_133_l2722_272261


namespace cassy_jars_proof_l2722_272203

def initial_jars (boxes_type1 boxes_type2 jars_per_box1 jars_per_box2 leftover_jars : ℕ) : ℕ :=
  boxes_type1 * jars_per_box1 + boxes_type2 * jars_per_box2 + leftover_jars

theorem cassy_jars_proof :
  initial_jars 10 30 12 10 80 = 500 := by
  sorry

end cassy_jars_proof_l2722_272203


namespace optimal_reading_distribution_l2722_272239

theorem optimal_reading_distribution 
  (total_time : ℕ) 
  (disc_capacity : ℕ) 
  (max_unused_space : ℕ) 
  (h1 : total_time = 630) 
  (h2 : disc_capacity = 80) 
  (h3 : max_unused_space = 4) :
  ∃ (num_discs : ℕ), 
    num_discs > 0 ∧ 
    num_discs * (disc_capacity - max_unused_space) ≥ total_time ∧
    (num_discs - 1) * disc_capacity < total_time ∧
    total_time / num_discs = 70 :=
sorry

end optimal_reading_distribution_l2722_272239


namespace polynomial_simplification_l2722_272298

theorem polynomial_simplification (x : ℝ) :
  (2*x^6 + 3*x^5 + x^4 + 3*x^3 + 2*x + 15) - (x^6 + 4*x^5 + 2*x^3 - x^2 + 5) =
  x^6 - x^5 + x^4 + x^3 + x^2 + 2*x + 10 := by
sorry

end polynomial_simplification_l2722_272298


namespace prob_bus_251_theorem_l2722_272266

/-- Represents the bus schedule system with two routes -/
structure BusSchedule where
  interval_152 : ℕ
  interval_251 : ℕ

/-- The probability of getting on bus No. 251 given a bus schedule -/
def prob_bus_251 (schedule : BusSchedule) : ℚ :=
  5 / 14

/-- Theorem stating the probability of getting on bus No. 251 -/
theorem prob_bus_251_theorem (schedule : BusSchedule) 
  (h1 : schedule.interval_152 = 5)
  (h2 : schedule.interval_251 = 7) :
  prob_bus_251 schedule = 5 / 14 := by
  sorry

#eval prob_bus_251 ⟨5, 7⟩

end prob_bus_251_theorem_l2722_272266


namespace line_segment_proportion_l2722_272247

theorem line_segment_proportion (a b c d : ℝ) :
  a = 1 →
  b = 2 →
  c = 3 →
  (a / b = c / d) →
  d = 6 := by
sorry

end line_segment_proportion_l2722_272247


namespace reduce_tiles_to_less_than_five_l2722_272233

/-- Represents the operation of removing prime-numbered tiles and renumbering --/
def remove_primes_and_renumber (n : ℕ) : ℕ := sorry

/-- Counts the number of operations needed to reduce the set to fewer than 5 tiles --/
def count_operations (initial_count : ℕ) : ℕ := sorry

/-- Theorem stating that 5 operations are needed to reduce 50 tiles to fewer than 5 --/
theorem reduce_tiles_to_less_than_five :
  count_operations 50 = 5 ∧ remove_primes_and_renumber (remove_primes_and_renumber (remove_primes_and_renumber (remove_primes_and_renumber (remove_primes_and_renumber 50)))) < 5 := by
  sorry

end reduce_tiles_to_less_than_five_l2722_272233


namespace arithmetic_sequence_properties_l2722_272221

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_a2 : a 2 = 5)
  (h_sum : a 6 + a 8 = 30) :
  d = 2 ∧
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, 1 / ((a n)^2 - 1) = (1/4) * (1/n - 1/(n+1))) :=
sorry

end arithmetic_sequence_properties_l2722_272221


namespace parabola_and_line_properties_l2722_272263

/-- Parabola with vertex at origin and directrix x = -1 -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  vertex_at_origin : equation 0 0
  directrix : ℝ → Prop
  directrix_eq : ∀ x, directrix x ↔ x = -1

/-- Line passing through two points on the parabola -/
structure IntersectingLine (p : Parabola) where
  equation : ℝ → ℝ → Prop
  passes_through_focus : equation 1 0
  intersects_parabola : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    equation x₁ y₁ ∧ p.equation x₁ y₁ ∧
    equation x₂ y₂ ∧ p.equation x₂ y₂ ∧
    x₁ ≠ x₂
  midpoint_x_coord : ℝ
  midpoint_condition : ∀ (x₁ y₁ x₂ y₂ : ℝ),
    equation x₁ y₁ ∧ p.equation x₁ y₁ ∧
    equation x₂ y₂ ∧ p.equation x₂ y₂ ∧
    x₁ ≠ x₂ →
    (x₁ + x₂) / 2 = midpoint_x_coord

/-- Main theorem about the parabola and intersecting line -/
theorem parabola_and_line_properties (p : Parabola) (l : IntersectingLine p) 
    (h_midpoint : l.midpoint_x_coord = 2) :
  (∀ x y, p.equation x y ↔ y^2 = 4*x) ∧
  (∀ x y, l.equation x y ↔ (y = Real.sqrt 2 * x - Real.sqrt 2 ∨ 
                            y = -Real.sqrt 2 * x + Real.sqrt 2)) := by
  sorry

end parabola_and_line_properties_l2722_272263


namespace fifty_cows_fifty_bags_l2722_272280

/-- The number of bags of husk eaten by a group of cows over a fixed period -/
def bagsEaten (numCows : ℕ) (daysPerBag : ℕ) (totalDays : ℕ) : ℕ :=
  numCows * (totalDays / daysPerBag)

/-- Theorem: 50 cows eat 50 bags of husk in 50 days -/
theorem fifty_cows_fifty_bags :
  bagsEaten 50 50 50 = 50 := by
  sorry

end fifty_cows_fifty_bags_l2722_272280


namespace exists_multiple_of_three_l2722_272228

def CircleNumbers (n : ℕ) := Fin n → ℕ

def ValidCircle (nums : CircleNumbers 99) : Prop :=
  ∀ i : Fin 99, 
    (nums i - nums (i + 1) = 1) ∨ 
    (nums i - nums (i + 1) = 2) ∨ 
    (nums i / nums (i + 1) = 2)

theorem exists_multiple_of_three (nums : CircleNumbers 99) 
  (h : ValidCircle nums) : 
  ∃ i : Fin 99, 3 ∣ nums i :=
sorry

end exists_multiple_of_three_l2722_272228


namespace clara_has_68_stickers_l2722_272255

/-- Calculates the number of stickers Clara has left after a series of transactions -/
def claras_stickers : ℕ :=
  let initial := 100
  let after_boy := initial - 10
  let after_teacher := after_boy + 50
  let after_classmates := after_teacher - 20
  let after_exchange := after_classmates - 15 + 30
  let to_friends := after_exchange / 2
  after_exchange - to_friends

/-- Proves that Clara ends up with 68 stickers -/
theorem clara_has_68_stickers : claras_stickers = 68 := by
  sorry

#eval claras_stickers

end clara_has_68_stickers_l2722_272255


namespace isosceles_triangle_theorem_congruent_triangles_theorem_supplementary_angles_not_always_equal_supplements_of_equal_angles_are_equal_proposition_c_is_false_l2722_272282

-- Define the basic geometric concepts
def Triangle : Type := sorry
def Angle : Type := sorry
def Line : Type := sorry

-- Define the properties and relations
def equal_sides (t : Triangle) (s1 s2 : Nat) : Prop := sorry
def equal_angles (t : Triangle) (a1 a2 : Nat) : Prop := sorry
def congruent (t1 t2 : Triangle) : Prop := sorry
def corresponding_sides_equal (t1 t2 : Triangle) : Prop := sorry
def supplementary (a1 a2 : Angle) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def supplement_of (a1 a2 : Angle) : Prop := sorry

-- Theorem statements
theorem isosceles_triangle_theorem (t : Triangle) (s1 s2 a1 a2 : Nat) :
  equal_sides t s1 s2 → equal_angles t a1 a2 := sorry

theorem congruent_triangles_theorem (t1 t2 : Triangle) :
  congruent t1 t2 → corresponding_sides_equal t1 t2 := sorry

theorem supplementary_angles_not_always_equal :
  ∃ (a1 a2 : Angle), supplementary a1 a2 ∧ a1 ≠ a2 := sorry

theorem supplements_of_equal_angles_are_equal (a1 a2 a3 a4 : Angle) :
  a1 = a2 → supplement_of a1 a3 → supplement_of a2 a4 → a3 = a4 := sorry

-- The main theorem proving that proposition C is false while others are true
theorem proposition_c_is_false :
  (∀ (t : Triangle) (s1 s2 a1 a2 : Nat), equal_sides t s1 s2 → equal_angles t a1 a2) ∧
  (∀ (t1 t2 : Triangle), congruent t1 t2 → corresponding_sides_equal t1 t2) ∧
  (∃ (a1 a2 : Angle) (l1 l2 : Line), supplementary a1 a2 ∧ a1 ≠ a2 ∧ parallel l1 l2) ∧
  (∀ (a1 a2 a3 a4 : Angle), a1 = a2 → supplement_of a1 a3 → supplement_of a2 a4 → a3 = a4) :=
sorry

end isosceles_triangle_theorem_congruent_triangles_theorem_supplementary_angles_not_always_equal_supplements_of_equal_angles_are_equal_proposition_c_is_false_l2722_272282


namespace exists_valid_31_min_students_smallest_total_l2722_272215

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- The ratios between grades are correct --/
def valid_ratios (gc : GradeCount) : Prop :=
  4 * gc.ninth = 3 * gc.eleventh ∧ 6 * gc.tenth = 5 * gc.eleventh

/-- The total number of students --/
def total_students (gc : GradeCount) : ℕ :=
  gc.ninth + gc.tenth + gc.eleventh

/-- There exists a valid configuration with 31 students --/
theorem exists_valid_31 : ∃ gc : GradeCount, valid_ratios gc ∧ total_students gc = 31 := by
  sorry

/-- Any valid configuration has at least 31 students --/
theorem min_students (gc : GradeCount) (h : valid_ratios gc) : total_students gc ≥ 31 := by
  sorry

/-- The smallest possible number of students is 31 --/
theorem smallest_total : (∃ gc : GradeCount, valid_ratios gc ∧ total_students gc = 31) ∧
  (∀ gc : GradeCount, valid_ratios gc → total_students gc ≥ 31) := by
  sorry

end exists_valid_31_min_students_smallest_total_l2722_272215


namespace simplify_trigonometric_expression_l2722_272272

theorem simplify_trigonometric_expression (α : Real) 
  (h1 : π < α ∧ α < 3*π/2) : 
  Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) - 
  Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) = 
  -2 * Real.tan α := by
  sorry

end simplify_trigonometric_expression_l2722_272272


namespace coefficient_x3y4_in_binomial_expansion_l2722_272269

theorem coefficient_x3y4_in_binomial_expansion :
  (Finset.range 8).sum (fun k => (Nat.choose 7 k) * (X : ℕ → ℕ) k * (Y : ℕ → ℕ) (7 - k)) =
  35 * (X : ℕ → ℕ) 3 * (Y : ℕ → ℕ) 4 + 
  (Finset.range 8).sum (fun k => if k ≠ 3 then (Nat.choose 7 k) * (X : ℕ → ℕ) k * (Y : ℕ → ℕ) (7 - k) else 0) :=
by sorry

end coefficient_x3y4_in_binomial_expansion_l2722_272269


namespace log_sum_adjacent_terms_l2722_272237

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem log_sum_adjacent_terms 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_a5 : a 5 = 10) : 
  Real.log (a 4) + Real.log (a 6) = 2 := by
sorry

end log_sum_adjacent_terms_l2722_272237


namespace mildred_initial_oranges_l2722_272238

/-- The number of oranges Mildred's father gave her -/
def oranges_from_father : ℕ := 2

/-- The total number of oranges Mildred has after receiving oranges from her father -/
def total_oranges : ℕ := 79

/-- The number of oranges Mildred initially collected -/
def initial_oranges : ℕ := total_oranges - oranges_from_father

theorem mildred_initial_oranges :
  initial_oranges = 77 :=
by sorry

end mildred_initial_oranges_l2722_272238


namespace f_odd_and_periodic_l2722_272257

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_odd_and_periodic (f : ℝ → ℝ) 
  (h1 : ∀ x, f (10 + x) = f (10 - x))
  (h2 : ∀ x, f (20 - x) = -f (20 + x)) :
  is_odd f ∧ is_periodic f 40 := by
  sorry

end f_odd_and_periodic_l2722_272257


namespace one_circle_exists_l2722_272264

def circle_equation (a x y : ℝ) : Prop :=
  x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1 = 0

def is_circle (a : ℝ) : Prop :=
  ∃ (center_x center_y radius : ℝ), 
    radius > 0 ∧
    ∀ (x y : ℝ), circle_equation a x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2

def a_set : Set ℝ := {-2, 0, 1, 3/4}

theorem one_circle_exists :
  ∃! (a : ℝ), a ∈ a_set ∧ is_circle a :=
sorry

end one_circle_exists_l2722_272264


namespace basketball_playoff_condition_l2722_272230

/-- A basketball team's playoff qualification condition -/
theorem basketball_playoff_condition (x : ℕ) : 
  (∀ (game : ℕ), game ≤ 32 → (game = 32 - x ∨ game = x)) →  -- Each game is either won or lost
  (2 * x + (32 - x) ≥ 48) →                                  -- Points condition
  (x ≤ 32) →                                                 -- Cannot win more games than played
  (2 * x + (32 - x) ≥ 48) :=                                 -- Conclusion: same as second hypothesis
by sorry

end basketball_playoff_condition_l2722_272230


namespace increasing_shift_l2722_272232

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Theorem statement
theorem increasing_shift (h : IncreasingOn f (-2) 3) :
  IncreasingOn (fun x => f (x + 5)) (-7) (-2) :=
sorry

end increasing_shift_l2722_272232


namespace final_price_calculation_final_price_is_841_32_l2722_272290

/-- Calculates the final price of a TV and sound system after discounts and tax --/
theorem final_price_calculation (tv_price sound_price : ℝ) 
  (tv_discount1 tv_discount2 sound_discount tax_rate : ℝ) : ℝ :=
  let tv_after_discounts := tv_price * (1 - tv_discount1) * (1 - tv_discount2)
  let sound_after_discount := sound_price * (1 - sound_discount)
  let total_before_tax := tv_after_discounts + sound_after_discount
  let tax_amount := total_before_tax * tax_rate
  let final_price := total_before_tax + tax_amount
  final_price

/-- Theorem stating that the final price is $841.32 given the specific conditions --/
theorem final_price_is_841_32 : 
  final_price_calculation 600 400 0.1 0.15 0.2 0.08 = 841.32 := by
  sorry

end final_price_calculation_final_price_is_841_32_l2722_272290


namespace equation_solution_l2722_272256

theorem equation_solution : ∃ x : ℝ, (x + 2) / (2 * x - 1) = 1 ∧ x = 3 := by
  sorry

end equation_solution_l2722_272256


namespace f_neg_two_eq_neg_nine_l2722_272229

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_neg_two_eq_neg_nine
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x ∈ Set.Icc 1 5, f x = x^3 + 1) :
  f (-2) = -9 := by sorry

end f_neg_two_eq_neg_nine_l2722_272229


namespace vector_a_start_point_l2722_272216

/-- The endpoint of vector a -/
def B : ℝ × ℝ := (1, 0)

/-- Vector b -/
def b : ℝ × ℝ := (-3, -4)

/-- Vector c -/
def c : ℝ × ℝ := (1, 1)

/-- Vector a in terms of b and c -/
def a : ℝ × ℝ := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)

/-- The starting point of vector a -/
def start_point : ℝ × ℝ := (B.1 - a.1, B.2 - a.2)

theorem vector_a_start_point : start_point = (12, 14) := by
  sorry

end vector_a_start_point_l2722_272216


namespace trigonometric_identity_l2722_272212

theorem trigonometric_identity : 
  let cos_45 : ℝ := Real.sqrt 2 / 2
  let tan_30 : ℝ := Real.sqrt 3 / 3
  let sin_60 : ℝ := Real.sqrt 3 / 2
  cos_45^2 + tan_30 * sin_60 = 1 := by
sorry

end trigonometric_identity_l2722_272212


namespace derivative_of_f_l2722_272251

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) * Real.log x

theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = -2 * Real.sin (2 * x) * Real.log x + Real.cos (2 * x) / x :=
by sorry

end derivative_of_f_l2722_272251


namespace new_circle_externally_tangent_l2722_272281

/-- Given circle equation -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- Center of the new circle -/
def new_center : ℝ × ℝ := (2, -2)

/-- Equation of the new circle -/
def new_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 2)^2 = 9

/-- Theorem stating that the new circle is externally tangent to the given circle -/
theorem new_circle_externally_tangent :
  ∃ (x y : ℝ), given_circle x y ∧ new_circle x y ∧
  (∀ (x' y' : ℝ), given_circle x' y' ∧ new_circle x' y' → (x, y) = (x', y')) :=
sorry

end new_circle_externally_tangent_l2722_272281


namespace arith_progression_poly_j_value_l2722_272244

/-- A polynomial of degree 4 with four distinct real zeros in arithmetic progression -/
structure ArithProgressionPoly where
  j : ℝ
  k : ℝ
  zeros : Fin 4 → ℝ
  distinct : ∀ (i j : Fin 4), i ≠ j → zeros i ≠ zeros j
  arith_prog : ∃ (a d : ℝ), ∀ (i : Fin 4), zeros i = a + i.val * d
  is_zero : ∀ (i : Fin 4), zeros i ^ 4 + j * (zeros i ^ 2) + k * zeros i + 225 = 0

theorem arith_progression_poly_j_value (p : ArithProgressionPoly) : p.j = -50 := by
  sorry

end arith_progression_poly_j_value_l2722_272244


namespace parabola_vertex_l2722_272284

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -3 * x^2 + 6 * x + 1

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex -/
def vertex_y : ℝ := 4

/-- Theorem: The vertex of the parabola y = -3x^2 + 6x + 1 is (1, 4) -/
theorem parabola_vertex :
  (∀ x : ℝ, parabola x ≤ vertex_y) ∧
  parabola vertex_x = vertex_y :=
sorry

end parabola_vertex_l2722_272284


namespace impossibility_of_sequence_conditions_l2722_272223

def is_valid_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  (∀ n : ℕ, a (n + 3) = a n) ∧
  (∀ n : ℕ, a n * a (n + 3) - a (n + 1) * a (n + 2) = c)

theorem impossibility_of_sequence_conditions : 
  ¬∃ (a : ℕ → ℝ) (c : ℝ), is_valid_sequence a c ∧ a 1 = 2 ∧ c = 2 :=
sorry

end impossibility_of_sequence_conditions_l2722_272223


namespace oil_price_reduction_l2722_272245

/-- Represents the price reduction percentage of oil -/
def price_reduction : ℚ := 30 / 100

/-- Represents the additional amount of oil that can be bought after the price reduction -/
def additional_oil : ℚ := 9

/-- Represents the fixed amount spent on oil -/
def fixed_amount : ℚ := 900

/-- Represents the price increase percentage of oil compared to rice -/
def price_increase : ℚ := 50 / 100

/-- Represents the prime number that divides the reduced oil price -/
def prime_divisor : ℕ := 5

theorem oil_price_reduction (original_price reduced_price rice_price : ℚ) : 
  reduced_price = original_price * (1 - price_reduction) →
  fixed_amount / original_price - fixed_amount / reduced_price = additional_oil →
  ∃ (n : ℕ), reduced_price = n * prime_divisor →
  original_price = rice_price * (1 + price_increase) →
  original_price = 857142 / 20000 ∧ 
  reduced_price = 30 ∧
  rice_price = 571428 / 20000 := by
  sorry

#eval 857142 / 20000  -- Outputs 42.8571
#eval 571428 / 20000  -- Outputs 28.5714

end oil_price_reduction_l2722_272245


namespace train_vs_airplane_capacity_difference_l2722_272294

/-- The passenger capacity of a single train car -/
def train_car_capacity : ℕ := 60

/-- The passenger capacity of a 747 airplane -/
def airplane_capacity : ℕ := 366

/-- The number of cars in the train -/
def train_cars : ℕ := 16

/-- The number of airplanes -/
def num_airplanes : ℕ := 2

/-- The theorem stating the difference in passenger capacity -/
theorem train_vs_airplane_capacity_difference :
  train_cars * train_car_capacity - num_airplanes * airplane_capacity = 228 := by
  sorry

end train_vs_airplane_capacity_difference_l2722_272294


namespace tangent_slope_at_2_l2722_272265

-- Define the function f(x) = x^2 + 3x
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 2*x + 3

-- Theorem statement
theorem tangent_slope_at_2 :
  f' 2 = 7 := by sorry

end tangent_slope_at_2_l2722_272265


namespace point_on_line_l2722_272252

/-- Given a line with slope 2 and y-intercept 2, the y-coordinate of a point on this line
    with x-coordinate 498 is 998. -/
theorem point_on_line (line : ℝ → ℝ) (x y : ℝ) : 
  (∀ t, line t = 2 * t + 2) →  -- Condition 1 and 2: slope is 2, y-intercept is 2
  x = 498 →                    -- Condition 4: x-coordinate is 498
  y = line x →                 -- Condition 3: the point (x, y) is on the line
  y = 998 := by                -- Question: prove y = 998
sorry


end point_on_line_l2722_272252


namespace prob_two_white_balls_prob_one_white_one_black_l2722_272207

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 4

/-- Represents the number of black balls in the bag -/
def black_balls : ℕ := 2

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- Calculates the probability of an event given the number of favorable outcomes and total outcomes -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- Theorem stating the probability of drawing two white balls -/
theorem prob_two_white_balls : 
  probability (white_balls.choose 2) (total_balls.choose 2) = 2 / 5 := by sorry

/-- Theorem stating the probability of drawing one white ball and one black ball -/
theorem prob_one_white_one_black : 
  probability (white_balls * black_balls) (total_balls.choose 2) = 8 / 15 := by sorry

end prob_two_white_balls_prob_one_white_one_black_l2722_272207


namespace ellipse_parabola_shared_focus_eccentricity_l2722_272205

/-- The eccentricity of an ellipse sharing a focus with a parabola -/
theorem ellipse_parabola_shared_focus_eccentricity 
  (p a b : ℝ) 
  (hp : p > 0) 
  (hab : a > b) 
  (hb : b > 0) : 
  ∃ (x y : ℝ), 
    x^2 = 2*p*y ∧ 
    y^2/a^2 + x^2/b^2 = 1 ∧ 
    (∃ (t : ℝ), x = 2*p*t ∧ y = p*t^2) → 
    Real.sqrt 2 - 1 = Real.sqrt (1 - b^2/a^2) := by
  sorry

#check ellipse_parabola_shared_focus_eccentricity

end ellipse_parabola_shared_focus_eccentricity_l2722_272205


namespace triangle_base_and_area_l2722_272288

theorem triangle_base_and_area (height : ℝ) (base : ℝ) (h_height : height = 12) 
  (h_ratio : height = 2 / 3 * base) : base = 18 ∧ height * base / 2 = 108 := by
  sorry

end triangle_base_and_area_l2722_272288


namespace profit_calculation_l2722_272236

-- Define the buying and selling rates
def buy_rate : ℚ := 5 / 6
def sell_rate : ℚ := 4 / 8

-- Define the target profit
def target_profit : ℚ := 120

-- Define the number of disks to be sold
def disks_to_sell : ℕ := 150

-- Theorem statement
theorem profit_calculation :
  (disks_to_sell : ℚ) * (1 / sell_rate - 1 / buy_rate) = target_profit := by
  sorry

end profit_calculation_l2722_272236


namespace max_area_rectangular_enclosure_l2722_272279

/-- Given a rectangular area with perimeter P (excluding one side) and length L twice the width W,
    the maximum area A is (P/4)^2 square units. -/
theorem max_area_rectangular_enclosure (P : ℝ) (h : P > 0) :
  let W := P / 4
  let L := 2 * W
  let A := L * W
  A = (P / 4) ^ 2 := by
  sorry

#check max_area_rectangular_enclosure

end max_area_rectangular_enclosure_l2722_272279


namespace complex_linear_combination_l2722_272276

theorem complex_linear_combination :
  let x : ℂ := 3 + 2*I
  let y : ℂ := 2 - 3*I
  3*x + 4*y = 17 - 6*I :=
by sorry

end complex_linear_combination_l2722_272276


namespace division_problem_l2722_272291

theorem division_problem (x y z total : ℚ) : 
  x / y = 5 / 7 →
  x / z = 5 / 11 →
  y = 150 →
  total = x + y + z →
  total = 493 := by
sorry

end division_problem_l2722_272291


namespace max_value_of_f_l2722_272258

/-- Given that f(x) = 2sin(x) - cos(x) reaches its maximum value when x = θ, prove that sin(θ) = 2√5/5 -/
theorem max_value_of_f (θ : ℝ) : 
  (∀ x, 2 * Real.sin x - Real.cos x ≤ 2 * Real.sin θ - Real.cos θ) →
  Real.sin θ = 2 * Real.sqrt 5 / 5 :=
by sorry

end max_value_of_f_l2722_272258


namespace symmetric_axis_of_translated_sine_l2722_272250

theorem symmetric_axis_of_translated_sine (f g : ℝ → ℝ) :
  (∀ x, f x = Real.sin (2 * x - π / 6)) →
  (∀ x, g x = f (x - π / 4)) →
  (∀ x, g x = Real.sin (2 * x - 2 * π / 3)) →
  (π / 12 : ℝ) ∈ {x | ∀ y, g (x + y) = g (x - y)} :=
by sorry

end symmetric_axis_of_translated_sine_l2722_272250


namespace minimum_time_to_fill_buckets_l2722_272243

def bucket_times : List Nat := [2, 4, 5, 7, 9]

def total_time (times : List Nat) : Nat :=
  (times.enum.map (fun (i, t) => t * (times.length - i))).sum

theorem minimum_time_to_fill_buckets :
  total_time bucket_times = 55 := by
  sorry

end minimum_time_to_fill_buckets_l2722_272243


namespace james_total_toys_l2722_272204

/-- The number of toy cars James buys to maximize his discount -/
def num_cars : ℕ := 26

/-- The number of toy soldiers James buys -/
def num_soldiers : ℕ := 2 * num_cars

/-- The total number of toys James buys -/
def total_toys : ℕ := num_cars + num_soldiers

theorem james_total_toys :
  (num_soldiers = 2 * num_cars) ∧ 
  (num_cars > 25) ∧
  (∀ n : ℕ, n > num_cars → n > 25) →
  total_toys = 78 := by
  sorry

end james_total_toys_l2722_272204


namespace union_condition_intersection_condition_l2722_272210

-- Define sets A and B
def A : Set ℝ := {x | 2 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < 3 * a}

-- Theorem 1
theorem union_condition (a : ℝ) : A ∪ B a = {x | 2 < x ∧ x < 6} → a = 2 := by
  sorry

-- Theorem 2
theorem intersection_condition (a : ℝ) : (A ∩ B a).Nonempty → 2/3 < a ∧ a < 4 := by
  sorry

end union_condition_intersection_condition_l2722_272210


namespace carrots_rows_planted_l2722_272249

/-- Calculates the number of rows of carrots planted given the planting conditions -/
theorem carrots_rows_planted (plants_per_row : ℕ) (planting_time : ℕ) (planting_rate : ℕ) : 
  plants_per_row > 0 →
  planting_time * planting_rate / plants_per_row = 400 :=
by
  intro h
  sorry

#check carrots_rows_planted 300 20 6000

end carrots_rows_planted_l2722_272249


namespace fruit_store_total_weight_l2722_272242

/-- Given a store with apples and pears, where the weight of pears is three times
    that of apples, calculate the total weight of apples and pears. -/
theorem fruit_store_total_weight (apple_weight : ℕ) (pear_weight : ℕ) : 
  apple_weight = 3200 →
  pear_weight = 3 * apple_weight →
  apple_weight + pear_weight = 12800 := by
sorry

end fruit_store_total_weight_l2722_272242


namespace original_class_strength_l2722_272283

/-- Proves that the original strength of an adult class is 17 students given the conditions. -/
theorem original_class_strength (original_average : ℝ) (new_students : ℕ) (new_average : ℝ) (average_decrease : ℝ) :
  original_average = 40 →
  new_students = 17 →
  new_average = 32 →
  average_decrease = 4 →
  ∃ (x : ℕ), x = 17 ∧ 
    (x : ℝ) * original_average + (new_students : ℝ) * new_average = 
      ((x : ℝ) + (new_students : ℝ)) * (original_average - average_decrease) := by
  sorry

#check original_class_strength

end original_class_strength_l2722_272283


namespace tangent_line_implies_a_b_values_min_a_value_min_a_value_achieved_l2722_272200

noncomputable def f (a b x : ℝ) : ℝ := b * x / Real.log x - a * x

theorem tangent_line_implies_a_b_values (a b : ℝ) :
  (∀ x y : ℝ, y = f a b x → 3 * x + 4 * y - Real.exp 2 = 0) →
  a = 1 ∧ b = 1 := by sorry

theorem min_a_value (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (Real.exp 1) (Real.exp 2) →
    x₂ ∈ Set.Icc (Real.exp 1) (Real.exp 2) →
    f 1 1 x₁ ≤ (deriv (f 1 1)) x₂ + a) →
  a ≥ 1/2 - 1/(4 * Real.exp 2) := by sorry

theorem min_a_value_achieved (a : ℝ) :
  a = 1/2 - 1/(4 * Real.exp 2) →
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧
    x₂ ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧
    f 1 1 x₁ ≤ (deriv (f 1 1)) x₂ + a := by sorry

end tangent_line_implies_a_b_values_min_a_value_min_a_value_achieved_l2722_272200


namespace angle_terminal_side_point_l2722_272240

/-- If the terminal side of angle α passes through point (-2, 1), then 1/(sin 2α) = -5/4 -/
theorem angle_terminal_side_point (α : ℝ) : 
  (Real.cos α = -2 / Real.sqrt 5 ∧ Real.sin α = 1 / Real.sqrt 5) → 
  1 / Real.sin (2 * α) = -5/4 := by
  sorry

end angle_terminal_side_point_l2722_272240


namespace movie_theater_adult_price_l2722_272270

/-- Proves that the adult ticket price is $6.75 given the conditions of the movie theater problem -/
theorem movie_theater_adult_price :
  let children_price : ℚ := 9/2
  let num_children : ℕ := 48
  let child_adult_diff : ℕ := 20
  let total_receipts : ℚ := 405
  let num_adults : ℕ := num_children - child_adult_diff
  let adult_price : ℚ := (total_receipts - children_price * num_children) / num_adults
  adult_price = 27/4 := by
sorry

end movie_theater_adult_price_l2722_272270


namespace triangle_side_sum_l2722_272278

/-- Given real numbers x, y, and z, if 1/|x^2+2yz|, 1/|y^2+2zx|, and 1/|z^2+2xy| 
    form the sides of a non-degenerate triangle, then xy + yz + zx = 0 -/
theorem triangle_side_sum (x y z : ℝ) 
  (h1 : 1 / |x^2 + 2*y*z| + 1 / |y^2 + 2*z*x| > 1 / |z^2 + 2*x*y|)
  (h2 : 1 / |y^2 + 2*z*x| + 1 / |z^2 + 2*x*y| > 1 / |x^2 + 2*y*z|)
  (h3 : 1 / |z^2 + 2*x*y| + 1 / |x^2 + 2*y*z| > 1 / |y^2 + 2*z*x|)
  (h4 : |x^2 + 2*y*z| ≠ 0)
  (h5 : |y^2 + 2*z*x| ≠ 0)
  (h6 : |z^2 + 2*x*y| ≠ 0) :
  x*y + y*z + z*x = 0 := by
sorry

end triangle_side_sum_l2722_272278


namespace officer_selection_count_l2722_272235

def total_members : ℕ := 25
def num_officers : ℕ := 3

-- Define a structure to represent a pair of members
structure MemberPair :=
  (member1 : ℕ)
  (member2 : ℕ)

-- Define the two special pairs
def pair1 : MemberPair := ⟨1, 2⟩  -- Rachel and Simon
def pair2 : MemberPair := ⟨3, 4⟩  -- Penelope and Quentin

-- Function to calculate the number of ways to choose officers
def count_officer_choices (total : ℕ) (officers : ℕ) (pair1 pair2 : MemberPair) : ℕ := 
  sorry

-- Theorem statement
theorem officer_selection_count :
  count_officer_choices total_members num_officers pair1 pair2 = 8072 :=
sorry

end officer_selection_count_l2722_272235


namespace figure_area_theorem_l2722_272241

theorem figure_area_theorem (x : ℝ) :
  let square1_area := (3 * x)^2
  let square2_area := (7 * x)^2
  let triangle_area := (1 / 2) * (3 * x) * (7 * x)
  square1_area + square2_area + triangle_area = 1300 →
  x = Real.sqrt (2600 / 137) := by
sorry

end figure_area_theorem_l2722_272241
