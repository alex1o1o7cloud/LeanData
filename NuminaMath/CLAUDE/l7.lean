import Mathlib

namespace cost_effective_flower_purchase_l7_791

/-- Represents the cost-effective flower purchasing problem --/
theorem cost_effective_flower_purchase
  (total_flowers : ℕ)
  (carnation_price lily_price : ℚ)
  (h_total : total_flowers = 300)
  (h_carnation_price : carnation_price = 5)
  (h_lily_price : lily_price = 10)
  : ∃ (carnations lilies : ℕ),
    carnations + lilies = total_flowers ∧
    carnations ≤ 2 * lilies ∧
    ∀ (c l : ℕ),
      c + l = total_flowers →
      c ≤ 2 * l →
      carnation_price * carnations + lily_price * lilies ≤
      carnation_price * c + lily_price * l ∧
    carnations = 200 ∧
    lilies = 100 := by
  sorry

end cost_effective_flower_purchase_l7_791


namespace poker_night_cards_l7_711

theorem poker_night_cards (half_decks full_decks thrown_away remaining : ℕ) : 
  half_decks = 3 →
  full_decks = 3 →
  thrown_away = 34 →
  remaining = 200 →
  ∃ (cards_per_full_deck cards_per_half_deck : ℕ),
    cards_per_half_deck = cards_per_full_deck / 2 ∧
    remaining + thrown_away = half_decks * cards_per_half_deck + full_decks * cards_per_full_deck ∧
    cards_per_full_deck = 52 :=
by sorry

end poker_night_cards_l7_711


namespace min_beta_delta_sum_l7_749

open Complex

/-- A complex function g satisfying certain conditions -/
def g (β δ : ℂ) : ℂ → ℂ := λ z ↦ (3 + 2*I)*z^3 + β*z + δ

/-- The theorem stating the minimum value of |β| + |δ| -/
theorem min_beta_delta_sum (β δ : ℂ) :
  (g β δ 1).im = 0 →
  (g β δ (-I)).im = -Real.pi →
  ∃ (min : ℝ), min = Real.sqrt (Real.pi^2 + 2*Real.pi + 2) + 2 ∧
    ∀ (β' δ' : ℂ), (g β' δ' 1).im = 0 → (g β' δ' (-I)).im = -Real.pi →
      Complex.abs β' + Complex.abs δ' ≥ min :=
sorry


end min_beta_delta_sum_l7_749


namespace vector_problem_l7_789

/-- Given two vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![2, -2]

/-- Define vector c as a linear combination of a and b -/
def c : Fin 2 → ℝ := λ i ↦ 4 * a i + b i

/-- The dot product of two vectors -/
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

theorem vector_problem :
  (dot_product b c • a = 0) ∧
  (dot_product a (a + (5/2 • b)) = 0) := by
  sorry

end vector_problem_l7_789


namespace cookery_club_committee_probability_l7_716

def total_members : ℕ := 24
def num_boys : ℕ := 14
def num_girls : ℕ := 10
def committee_size : ℕ := 5

theorem cookery_club_committee_probability :
  let total_committees := Nat.choose total_members committee_size
  let committees_with_fewer_than_two_girls := 
    Nat.choose num_boys committee_size + 
    (num_girls * Nat.choose num_boys (committee_size - 1))
  let committees_with_at_least_two_girls := 
    total_committees - committees_with_fewer_than_two_girls
  (committees_with_at_least_two_girls : ℚ) / total_committees = 2541 / 3542 := by
  sorry

end cookery_club_committee_probability_l7_716


namespace factor_proof_l7_705

theorem factor_proof :
  (∃ n : ℕ, 24 = 4 * n) ∧ (∃ m : ℕ, 162 = 9 * m) := by
  sorry

end factor_proof_l7_705


namespace river_width_calculation_l7_735

/-- The width of a river given the length of an existing bridge and the additional length needed to cross it. -/
def river_width (existing_bridge_length additional_length : ℕ) : ℕ :=
  existing_bridge_length + additional_length

/-- Theorem: The width of the river is equal to the sum of the existing bridge length and the additional length needed. -/
theorem river_width_calculation (existing_bridge_length additional_length : ℕ) :
  river_width existing_bridge_length additional_length = existing_bridge_length + additional_length :=
by
  sorry

/-- The width of the specific river in the problem. -/
def specific_river_width : ℕ := river_width 295 192

#eval specific_river_width

end river_width_calculation_l7_735


namespace calvin_haircut_goal_l7_765

/-- Calculate the percentage of progress towards a goal -/
def progressPercentage (completed : ℕ) (total : ℕ) : ℚ :=
  (completed : ℚ) / (total : ℚ) * 100

/-- Calvin's haircut goal problem -/
theorem calvin_haircut_goal :
  let total_haircuts : ℕ := 10
  let completed_haircuts : ℕ := 8
  progressPercentage completed_haircuts total_haircuts = 80 := by
  sorry

end calvin_haircut_goal_l7_765


namespace final_chicken_count_l7_720

def chicken_count (initial : ℕ) (second_factor : ℕ) (second_subtract : ℕ) (dog_eat : ℕ) (final_factor : ℕ) (final_subtract : ℕ) : ℕ :=
  let after_second := initial + (second_factor * initial - second_subtract)
  let after_dog := after_second - dog_eat
  let final_addition := final_factor * (final_factor * after_dog - final_subtract)
  after_dog + final_addition

theorem final_chicken_count :
  chicken_count 12 3 8 2 2 10 = 246 := by
  sorry

end final_chicken_count_l7_720


namespace expression_evaluation_l7_730

theorem expression_evaluation (x y z : ℤ) (hx : x = -2) (hy : y = -4) (hz : z = 3) :
  (5 * (x - y)^2 - x * z^2) / (z - y) = 38 / 7 := by
  sorry

end expression_evaluation_l7_730


namespace flora_milk_consumption_l7_714

/-- Calculates the total amount of milk Flora needs to drink based on the given conditions -/
def total_milk_gallons (weeks : ℕ) (flora_estimate : ℕ) (brother_additional : ℕ) : ℕ :=
  let days := weeks * 7
  let daily_amount := flora_estimate + brother_additional
  days * daily_amount

/-- Theorem stating that the total amount of milk Flora needs to drink is 105 gallons -/
theorem flora_milk_consumption :
  total_milk_gallons 3 3 2 = 105 := by
  sorry

end flora_milk_consumption_l7_714


namespace factor_implies_c_value_l7_726

theorem factor_implies_c_value (c : ℝ) : 
  (∀ x : ℝ, (x - 5) * ((c/100) * x^2 + (23/100) * x - (c/20) + 11/20) = 
             c * x^3 + 23 * x^2 - 5 * c * x + 55) → 
  c = -6.3 := by sorry

end factor_implies_c_value_l7_726


namespace correct_last_digit_prob_l7_700

/-- The number of possible digits for each position in the password -/
def num_digits : ℕ := 10

/-- The probability of guessing the correct digit on the first attempt -/
def first_attempt_prob : ℚ := 1 / num_digits

/-- The probability of guessing the correct digit on the second attempt, given the first attempt was incorrect -/
def second_attempt_prob : ℚ := 1 / (num_digits - 1)

/-- The probability of guessing the correct last digit within 2 attempts -/
def two_attempt_prob : ℚ := first_attempt_prob + (1 - first_attempt_prob) * second_attempt_prob

theorem correct_last_digit_prob :
  two_attempt_prob = 1 / 5 := by
  sorry

end correct_last_digit_prob_l7_700


namespace inscribed_octagon_area_l7_778

/-- The area of a regular octagon inscribed in a circle with area 400π square units is 800√2 square units. -/
theorem inscribed_octagon_area (circle_area : ℝ) (octagon_area : ℝ) :
  circle_area = 400 * Real.pi →
  octagon_area = 8 * (1 / 2 * (20^2) * Real.sin (π / 4)) →
  octagon_area = 800 * Real.sqrt 2 := by
  sorry

end inscribed_octagon_area_l7_778


namespace certain_number_proof_l7_708

theorem certain_number_proof : ∃ x : ℕ, (3 * 16) + (3 * 17) + (3 * 20) + x = 170 ∧ x = 11 := by
  sorry

end certain_number_proof_l7_708


namespace min_correct_responses_l7_723

def score (correct : ℕ) : ℤ :=
  8 * (correct : ℤ) - 20

theorem min_correct_responses : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → score m + 10 < 120) ∧ 
  (score n + 10 ≥ 120) ∧
  n = 17 :=
sorry

end min_correct_responses_l7_723


namespace points_earned_in_level_l7_784

/-- Calculates the points earned in a video game level -/
theorem points_earned_in_level 
  (points_per_enemy : ℕ) 
  (total_enemies : ℕ) 
  (enemies_not_destroyed : ℕ) : 
  points_per_enemy = 9 →
  total_enemies = 11 →
  enemies_not_destroyed = 3 →
  (total_enemies - enemies_not_destroyed) * points_per_enemy = 72 :=
by
  sorry

end points_earned_in_level_l7_784


namespace ebook_reader_difference_l7_790

theorem ebook_reader_difference (anna_count john_original_count : ℕ) : 
  anna_count = 50 →
  john_original_count < anna_count →
  john_original_count + anna_count = 82 + 3 →
  anna_count - john_original_count = 15 := by
sorry

end ebook_reader_difference_l7_790


namespace largest_power_dividing_factorial_l7_788

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_power_dividing_factorial : 
  ∃ (k : ℕ), k = 287 ∧ 
  (∀ (m : ℕ), 1729^m ∣ factorial 1729 → m ≤ k) ∧
  (1729^k ∣ factorial 1729) :=
by sorry

end largest_power_dividing_factorial_l7_788


namespace sum_geq_abs_sum_div_3_l7_706

theorem sum_geq_abs_sum_div_3 (a b c : ℝ) 
  (hab : a + b ≥ 0) (hbc : b + c ≥ 0) (hca : c + a ≥ 0) : 
  a + b + c ≥ (|a| + |b| + |c|) / 3 := by
  sorry

end sum_geq_abs_sum_div_3_l7_706


namespace gcd_problem_l7_797

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) :
  Int.gcd (4 * b ^ 2 + 35 * b + 72) (3 * b + 8) = 2 := by
  sorry

end gcd_problem_l7_797


namespace quadratic_minimum_l7_798

theorem quadratic_minimum (x : ℝ) (h : x ≥ 0) : x^2 + 13*x + 4 ≥ 4 ∧ ∃ y ≥ 0, y^2 + 13*y + 4 = 4 := by
  sorry

end quadratic_minimum_l7_798


namespace direct_proportion_decreasing_l7_799

theorem direct_proportion_decreasing (k x₁ x₂ y₁ y₂ : ℝ) :
  k < 0 →
  x₁ < x₂ →
  y₁ = k * x₁ →
  y₂ = k * x₂ →
  y₁ > y₂ := by
  sorry

end direct_proportion_decreasing_l7_799


namespace binomial_expansion_coefficient_l7_721

/-- The coefficient of x^n in the expansion of (x^2 + a/x)^m -/
def coeff (a : ℝ) (m n : ℕ) : ℝ := sorry

theorem binomial_expansion_coefficient (a : ℝ) :
  coeff a 5 7 = -15 → a = -3 := by sorry

end binomial_expansion_coefficient_l7_721


namespace min_value_x_plus_3y_l7_703

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 4 * y = x * y) :
  x + 3 * y ≥ 25 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 3 * x + 4 * y = x * y ∧ x + 3 * y = 25 := by
  sorry

end min_value_x_plus_3y_l7_703


namespace train_speed_calculation_l7_746

-- Define the distance in meters
def distance : ℝ := 200

-- Define the time in seconds (as a variable)
variable (p : ℝ)

-- Define the speed conversion factor from m/s to km/h
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_calculation (p : ℝ) (h : p > 0) :
  (distance / p) * conversion_factor = 720 / p := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l7_746


namespace f_negation_property_l7_757

theorem f_negation_property (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = 2 * Real.sin x + x^3 + 1) →
  f a = 3 →
  f (-a) = -1 := by
sorry

end f_negation_property_l7_757


namespace circle_and_tangent_line_l7_725

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a line is tangent to a circle -/
def Line.tangentTo (l : Line) (c : Circle) : Prop :=
  (l.a * c.center.1 + l.b * c.center.2 + l.c)^2 = 
    (l.a^2 + l.b^2) * c.radius^2

theorem circle_and_tangent_line 
  (c : Circle) 
  (l : Line) : 
  c.contains (0, 0) → 
  c.contains (4, 0) → 
  c.contains (0, 2) → 
  l.a = 2 → 
  l.b = -1 → 
  l.c = 2 → 
  l.tangentTo c → 
  c = { center := (2, 1), radius := Real.sqrt 5 } ∧ 
  l = { a := 2, b := -1, c := 2 } := by
  sorry

end circle_and_tangent_line_l7_725


namespace min_value_reciprocal_sum_l7_710

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 1 / b ≥ 4 := by
  sorry

end min_value_reciprocal_sum_l7_710


namespace total_pears_picked_l7_737

theorem total_pears_picked (jason_pears keith_pears mike_pears : ℕ)
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_pears = 12) :
  jason_pears + keith_pears + mike_pears = 105 := by
  sorry

end total_pears_picked_l7_737


namespace percentage_sum_l7_748

theorem percentage_sum : 
  (20 / 100 * 40) + (25 / 100 * 60) = 23 := by
  sorry

end percentage_sum_l7_748


namespace largest_number_with_constraints_l7_794

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 2

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_constraints :
  ∀ n : ℕ, 
    is_valid_number n ∧ 
    digit_sum n = 20 →
    n ≤ 44444 :=
by sorry

end largest_number_with_constraints_l7_794


namespace chicken_egg_problem_l7_793

theorem chicken_egg_problem (initial_eggs : ℕ) (used_eggs : ℕ) (eggs_per_chicken : ℕ) (final_eggs : ℕ) : 
  initial_eggs = 10 →
  used_eggs = 5 →
  eggs_per_chicken = 3 →
  final_eggs = 11 →
  (final_eggs - (initial_eggs - used_eggs)) / eggs_per_chicken = 2 :=
by sorry

end chicken_egg_problem_l7_793


namespace right_triangle_common_factor_l7_763

theorem right_triangle_common_factor (d : ℝ) (h_pos : d > 0) : 
  (2 * d = 45 ∨ 4 * d = 45 ∨ 5 * d = 45) ∧ 
  (2 * d)^2 + (4 * d)^2 = (5 * d)^2 → 
  d = 9 := by sorry

end right_triangle_common_factor_l7_763


namespace bead_arrangement_probability_l7_724

/-- The number of red beads -/
def num_red : ℕ := 4

/-- The number of white beads -/
def num_white : ℕ := 2

/-- The number of green beads -/
def num_green : ℕ := 1

/-- The total number of beads -/
def total_beads : ℕ := num_red + num_white + num_green

/-- A function that calculates the probability of arranging the beads
    such that no two neighboring beads are the same color -/
def prob_no_adjacent_same_color : ℚ :=
  2 / 15

/-- Theorem stating that the probability of arranging the beads
    such that no two neighboring beads are the same color is 2/15 -/
theorem bead_arrangement_probability :
  prob_no_adjacent_same_color = 2 / 15 := by
  sorry

end bead_arrangement_probability_l7_724


namespace probability_two_aces_l7_769

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of aces in a standard deck -/
def AcesInDeck : ℕ := 4

/-- Represents the total number of cards in the mixed deck -/
def TotalCards : ℕ := 2 * StandardDeck

/-- Represents the total number of aces in the mixed deck -/
def TotalAces : ℕ := 2 * AcesInDeck

/-- The probability of drawing two aces consecutively from a mixed deck of 104 cards -/
theorem probability_two_aces (StandardDeck AcesInDeck TotalCards TotalAces : ℕ) 
  (h1 : TotalCards = 2 * StandardDeck)
  (h2 : TotalAces = 2 * AcesInDeck)
  (h3 : StandardDeck = 52)
  (h4 : AcesInDeck = 4) :
  (TotalAces : ℚ) / TotalCards * (TotalAces - 1) / (TotalCards - 1) = 7 / 1339 := by
  sorry

end probability_two_aces_l7_769


namespace sqrt_sum_squares_eq_sum_l7_782

theorem sqrt_sum_squares_eq_sum (a b : ℝ) :
  Real.sqrt (a^2 + b^2) = a + b ↔ a * b = 0 ∧ a + b ≥ 0 := by
  sorry

end sqrt_sum_squares_eq_sum_l7_782


namespace min_n_greater_than_T10_plus_1013_l7_767

def T (n : ℕ) : ℚ := n + 1 - (1 / 2^n)

theorem min_n_greater_than_T10_plus_1013 :
  (∀ n : ℕ, n > T 10 + 1013 → n ≥ 1024) ∧
  (∃ n : ℕ, n > T 10 + 1013 ∧ n = 1024) :=
sorry

end min_n_greater_than_T10_plus_1013_l7_767


namespace system_solution_l7_792

theorem system_solution (x y z t : ℝ) : 
  (x * y * z = x + y + z ∧
   y * z * t = y + z + t ∧
   z * t * x = z + t + x ∧
   t * x * y = t + x + y) →
  ((x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0) ∨
   (x = Real.sqrt 3 ∧ y = Real.sqrt 3 ∧ z = Real.sqrt 3 ∧ t = Real.sqrt 3) ∨
   (x = -Real.sqrt 3 ∧ y = -Real.sqrt 3 ∧ z = -Real.sqrt 3 ∧ t = -Real.sqrt 3)) :=
by sorry

end system_solution_l7_792


namespace range_of_function_l7_776

theorem range_of_function (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) 
  (h4 : b = (1 + Real.sqrt 5) / 2 * a) : 
  ∃ (x : ℝ), (9 - 9 * Real.sqrt 5) / 32 < a * (b - 3/2) ∧ 
             a * (b - 3/2) < (Real.sqrt 5 - 2) / 2 := by
  sorry

end range_of_function_l7_776


namespace total_pokemon_cards_l7_771

/-- The number of cards in one dozen -/
def cards_per_dozen : ℕ := 12

/-- The number of dozens each person has -/
def dozens_per_person : ℕ := 9

/-- The number of people -/
def number_of_people : ℕ := 4

/-- Theorem: The total number of Pokemon cards owned by 4 people, each having 9 dozen cards, is equal to 432 -/
theorem total_pokemon_cards : 
  (cards_per_dozen * dozens_per_person * number_of_people) = 432 := by
  sorry

end total_pokemon_cards_l7_771


namespace intersection_property_l7_773

/-- The curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 2*ρ*Real.cos θ - 3 = 0

/-- The line l in polar coordinates -/
def line_l (ρ θ : ℝ) (m : ℝ) : Prop :=
  ρ * (Real.cos θ + Real.sin θ) = m

/-- The theorem statement -/
theorem intersection_property (m : ℝ) :
  m > 0 →
  ∃ (ρ_A ρ_M ρ_N : ℝ),
    line_l ρ_A (π/4) m ∧
    curve_C ρ_M (π/4) ∧
    curve_C ρ_N (π/4) ∧
    ρ_A * ρ_M * ρ_N = 6 →
  m = 2 * Real.sqrt 2 :=
by sorry

end intersection_property_l7_773


namespace cubic_equation_root_b_value_l7_760

theorem cubic_equation_root_b_value :
  ∀ (a b : ℚ),
  (∃ (x : ℝ), x = 2 + Real.sqrt 3 ∧ x^3 + a*x^2 + b*x + 10 = 0) →
  b = -39 :=
by sorry

end cubic_equation_root_b_value_l7_760


namespace problem_statement_l7_754

open Real

theorem problem_statement (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ - a)^2 + (log (x₀^2) - 2*a)^2 ≤ 4/5) → a = 1/5 := by
  sorry

end problem_statement_l7_754


namespace same_conclusion_from_true_and_false_l7_768

theorem same_conclusion_from_true_and_false :
  ∃ (A : Prop) (T F : Prop), T ∧ ¬F ∧ (T → A) ∧ (F → A) := by
  sorry

end same_conclusion_from_true_and_false_l7_768


namespace jeffreys_steps_calculation_l7_728

-- Define the number of steps for Andrew and Jeffrey
def andrews_steps : ℕ := 150
def jeffreys_steps : ℕ := 200

-- Define the ratio of Andrew's steps to Jeffrey's steps
def step_ratio : ℚ := 3 / 4

-- Theorem statement
theorem jeffreys_steps_calculation :
  andrews_steps * 4 = jeffreys_steps * 3 :=
by sorry

end jeffreys_steps_calculation_l7_728


namespace long_distance_bill_calculation_l7_739

-- Define the constants
def monthly_fee : ℚ := 2
def per_minute_rate : ℚ := 12 / 100
def minutes_used : ℕ := 178

-- Define the theorem
theorem long_distance_bill_calculation :
  monthly_fee + per_minute_rate * minutes_used = 23.36 := by
  sorry

end long_distance_bill_calculation_l7_739


namespace journey_speed_calculation_l7_796

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 400 ∧ 
  total_time = 30 ∧ 
  second_half_speed = 10 →
  (total_distance / 2) / (total_time - (total_distance / 2) / second_half_speed) = 20 := by
  sorry

end journey_speed_calculation_l7_796


namespace solution_set_equality_l7_756

def S : Set ℝ := {x : ℝ | |x - 1| + |x + 2| ≤ 4}

theorem solution_set_equality : S = Set.Icc (-5/2) (3/2) := by
  sorry

end solution_set_equality_l7_756


namespace percentage_problem_l7_717

theorem percentage_problem (x : ℝ) (h : x / 100 * 60 = 12) : 15 / 100 * x = 3 := by
  sorry

end percentage_problem_l7_717


namespace prime_sum_85_product_166_l7_775

theorem prime_sum_85_product_166 (p q : ℕ) (hp : Prime p) (hq : Prime q) (hsum : p + q = 85) :
  p * q = 166 := by
sorry

end prime_sum_85_product_166_l7_775


namespace student_union_selections_l7_761

/-- Represents the number of students in each grade of the student union -/
structure StudentUnion where
  freshmen : Nat
  sophomores : Nat
  juniors : Nat

/-- Calculates the number of ways to select one person as president -/
def selectPresident (su : StudentUnion) : Nat :=
  su.freshmen + su.sophomores + su.juniors

/-- Calculates the number of ways to select one person from each grade for the standing committee -/
def selectStandingCommittee (su : StudentUnion) : Nat :=
  su.freshmen * su.sophomores * su.juniors

/-- Calculates the number of ways to select two people from different grades for a city activity -/
def selectCityActivity (su : StudentUnion) : Nat :=
  su.freshmen * su.sophomores + su.sophomores * su.juniors + su.juniors * su.freshmen

theorem student_union_selections (su : StudentUnion) 
  (h1 : su.freshmen = 5) 
  (h2 : su.sophomores = 6) 
  (h3 : su.juniors = 4) : 
  selectPresident su = 15 ∧ 
  selectStandingCommittee su = 120 ∧ 
  selectCityActivity su = 74 := by
  sorry

#eval selectPresident ⟨5, 6, 4⟩
#eval selectStandingCommittee ⟨5, 6, 4⟩
#eval selectCityActivity ⟨5, 6, 4⟩

end student_union_selections_l7_761


namespace dodecahedron_interior_diagonals_l7_758

/-- A dodecahedron is a 3D figure with 20 vertices and 3 faces meeting at each vertex. -/
structure Dodecahedron where
  vertices : ℕ
  faces_per_vertex : ℕ
  h_vertices : vertices = 20
  h_faces_per_vertex : faces_per_vertex = 3

/-- The number of interior diagonals in a dodecahedron. -/
def interior_diagonals (d : Dodecahedron) : ℕ :=
  (d.vertices * (d.vertices - 1 - 2 * d.faces_per_vertex)) / 2

/-- Theorem: The number of interior diagonals in a dodecahedron is 160. -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l7_758


namespace exist_good_coloring_l7_774

/-- The set of colors --/
inductive Color
| red
| white

/-- The type of coloring functions --/
def Coloring := Fin 2017 → Color

/-- Checks if a sequence is an arithmetic progression --/
def isArithmeticSequence (s : Fin n → Fin 2017) : Prop :=
  ∃ a d : ℕ, ∀ i : Fin n, s i = a + i.val * d

/-- The main theorem --/
theorem exist_good_coloring (n : ℕ) (h : n ≥ 18) :
  ∃ f : Coloring, ∀ s : Fin n → Fin 2017, 
    isArithmeticSequence s → 
    ∃ i j : Fin n, f (s i) ≠ f (s j) :=
sorry

end exist_good_coloring_l7_774


namespace percentage_difference_l7_745

theorem percentage_difference : 
  (38 / 100 * 80) - (12 / 100 * 160) = 11.2 := by sorry

end percentage_difference_l7_745


namespace max_value_theorem_l7_718

theorem max_value_theorem (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  ∃ (max : ℝ), max = 1 + Real.sqrt 3 ∧ 
  ∀ (z : ℝ), z = (y + x) / x → z ≤ max := by
sorry

end max_value_theorem_l7_718


namespace custodian_jugs_theorem_l7_734

/-- The number of jugs needed to provide water for students -/
def jugs_needed (jug_capacity : ℕ) (num_students : ℕ) (cups_per_student : ℕ) : ℕ :=
  (num_students * cups_per_student + jug_capacity - 1) / jug_capacity

/-- Theorem: Given the conditions, 50 jugs are needed -/
theorem custodian_jugs_theorem :
  jugs_needed 40 200 10 = 50 := by
  sorry

end custodian_jugs_theorem_l7_734


namespace divisor_problem_l7_753

theorem divisor_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 17698 →
  quotient = 89 →
  remainder = 14 →
  ∃ (divisor : ℕ), 
    dividend = divisor * quotient + remainder ∧
    divisor = 198 := by
  sorry

end divisor_problem_l7_753


namespace corn_height_after_three_weeks_l7_781

def corn_growth (first_week_growth : ℝ) (second_week_multiplier : ℝ) (third_week_multiplier : ℝ) : ℝ :=
  let second_week_growth := first_week_growth * second_week_multiplier
  let third_week_growth := second_week_growth * third_week_multiplier
  first_week_growth + second_week_growth + third_week_growth

theorem corn_height_after_three_weeks :
  corn_growth 2 2 4 = 22 := by
  sorry

end corn_height_after_three_weeks_l7_781


namespace hannah_total_cost_l7_759

/-- The total cost of Hannah's purchase of sweatshirts and T-shirts -/
def total_cost (num_sweatshirts num_tshirts sweatshirt_price tshirt_price : ℕ) : ℕ :=
  num_sweatshirts * sweatshirt_price + num_tshirts * tshirt_price

/-- Theorem stating that Hannah's total cost is $65 -/
theorem hannah_total_cost :
  total_cost 3 2 15 10 = 65 := by
  sorry

end hannah_total_cost_l7_759


namespace square_garden_perimeter_l7_772

theorem square_garden_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 450 →
  area = side^2 →
  perimeter = 4 * side →
  perimeter = 60 * Real.sqrt 2 := by
sorry

end square_garden_perimeter_l7_772


namespace matrix_power_four_l7_742

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, 1; 1, 1]

theorem matrix_power_four :
  A ^ 4 = !![34, 21; 21, 13] := by sorry

end matrix_power_four_l7_742


namespace insects_in_laboratory_l7_751

/-- The number of insects in a laboratory given the total number of insect legs and legs per insect. -/
def number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Theorem stating that there are 9 insects in the laboratory given the conditions. -/
theorem insects_in_laboratory : number_of_insects 54 6 = 9 := by
  sorry

end insects_in_laboratory_l7_751


namespace gmat_scores_l7_719

theorem gmat_scores (u v : ℝ) (h1 : u > v) (h2 : u - v = (u + v) / 2) : v / u = 1 / 3 := by
  sorry

end gmat_scores_l7_719


namespace garden_area_l7_755

theorem garden_area (length width : ℝ) (h1 : length = 350) (h2 : width = 50) :
  (length * width) / 10000 = 1.75 := by
  sorry

end garden_area_l7_755


namespace heartsuit_ratio_l7_733

def heartsuit (n m : ℝ) : ℝ := n^2 * m^3

theorem heartsuit_ratio : (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 := by
  sorry

end heartsuit_ratio_l7_733


namespace janes_cans_l7_787

theorem janes_cans (total_seeds : ℕ) (seeds_per_can : ℕ) (h1 : total_seeds = 54) (h2 : seeds_per_can = 6) :
  total_seeds / seeds_per_can = 9 :=
by sorry

end janes_cans_l7_787


namespace sticker_packs_total_cost_l7_713

/-- Calculates the total cost of sticker packs bought over three days --/
def total_cost (monday_packs : ℕ) (monday_price : ℚ) (monday_discount : ℚ)
                (tuesday_packs : ℕ) (tuesday_price : ℚ) (tuesday_tax : ℚ)
                (wednesday_packs : ℕ) (wednesday_price : ℚ) (wednesday_discount : ℚ) (wednesday_tax : ℚ) : ℚ :=
  let monday_cost := (monday_packs : ℚ) * monday_price * (1 - monday_discount)
  let tuesday_cost := (tuesday_packs : ℚ) * tuesday_price * (1 + tuesday_tax)
  let wednesday_cost := (wednesday_packs : ℚ) * wednesday_price * (1 - wednesday_discount) * (1 + wednesday_tax)
  monday_cost + tuesday_cost + wednesday_cost

/-- Theorem stating the total cost of sticker packs over three days --/
theorem sticker_packs_total_cost :
  total_cost 15 (5/2) (1/10) 25 3 (1/20) 30 (7/2) (3/20) (2/25) = 20889/100 :=
by sorry

end sticker_packs_total_cost_l7_713


namespace profit_2004_l7_722

/-- Represents the profit of a company over years -/
def CompanyProfit (initialProfit : ℝ) (growthRate : ℝ) (year : ℕ) : ℝ :=
  initialProfit * (1 + growthRate) ^ (year - 2002)

/-- Theorem stating the profit in 2004 given initial conditions -/
theorem profit_2004 (initialProfit growthRate : ℝ) :
  initialProfit = 10 →
  CompanyProfit initialProfit growthRate 2004 = 1000 * (1 + growthRate)^2 := by
  sorry

#check profit_2004

end profit_2004_l7_722


namespace arrangement_counts_l7_704

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- Calculates the number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of arrangements where the three girls must stand together -/
def arrangements_girls_together : ℕ := 
  permutations num_girls num_girls * permutations (num_boys + 1) (num_boys + 1)

/-- The number of arrangements where no two girls are next to each other -/
def arrangements_girls_apart : ℕ := 
  permutations num_boys num_boys * permutations (num_boys + 1) num_girls

/-- The number of arrangements where there are exactly three people between person A and person B -/
def arrangements_three_between : ℕ := 
  permutations 2 2 * permutations (total_people - 2) 3 * permutations 5 5

/-- The number of arrangements where persons A and B are adjacent, but neither is next to person C -/
def arrangements_ab_adjacent_not_c : ℕ := 
  permutations 2 2 * permutations (total_people - 3) (total_people - 3) * permutations 5 2

theorem arrangement_counts :
  arrangements_girls_together = 720 ∧
  arrangements_girls_apart = 1440 ∧
  arrangements_three_between = 720 ∧
  arrangements_ab_adjacent_not_c = 960 := by sorry

end arrangement_counts_l7_704


namespace rectangular_prism_sum_l7_795

/-- A rectangular prism is a three-dimensional shape with rectangular faces. -/
structure RectangularPrism where
  edges : Nat
  corners : Nat
  faces : Nat

/-- The properties of a standard rectangular prism. -/
def standardPrism : RectangularPrism :=
  { edges := 12
  , corners := 8
  , faces := 6 }

/-- The sum of edges, corners, and faces of a rectangular prism is 26. -/
theorem rectangular_prism_sum :
  standardPrism.edges + standardPrism.corners + standardPrism.faces = 26 := by
  sorry

end rectangular_prism_sum_l7_795


namespace repeating_decimal_23_value_l7_715

/-- The value of the infinite repeating decimal 0.overline{23} -/
def repeating_decimal_23 : ℚ := 23 / 99

/-- Theorem stating that the infinite repeating decimal 0.overline{23} is equal to 23/99 -/
theorem repeating_decimal_23_value : 
  repeating_decimal_23 = 23 / 99 := by sorry

end repeating_decimal_23_value_l7_715


namespace find_divisor_l7_712

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 132 →
  quotient = 8 →
  remainder = 4 →
  dividend = divisor * quotient + remainder →
  divisor = 16 := by
  sorry

end find_divisor_l7_712


namespace clothing_store_loss_l7_707

/-- Proves that selling two sets of clothes at 168 yuan each, with one set having a 20% profit
    and the other having a 20% loss, results in a total loss of 14 yuan. -/
theorem clothing_store_loss (selling_price : ℝ) (profit_percentage : ℝ) (loss_percentage : ℝ) :
  selling_price = 168 →
  profit_percentage = 0.2 →
  loss_percentage = 0.2 →
  let profit_cost := selling_price / (1 + profit_percentage)
  let loss_cost := selling_price / (1 - loss_percentage)
  (2 * selling_price) - (profit_cost + loss_cost) = -14 := by
sorry

end clothing_store_loss_l7_707


namespace triangle_centers_l7_785

/-- Triangle XYZ with side lengths x, y, z -/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Incenter coordinates (a, b, c) -/
structure Incenter where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_one : a + b + c = 1

/-- Centroid coordinates (p, q, r) -/
structure Centroid where
  p : ℝ
  q : ℝ
  r : ℝ
  sum_one : p + q + r = 1

/-- The theorem to be proved -/
theorem triangle_centers (t : Triangle) (i : Incenter) (c : Centroid) :
  t.x = 13 ∧ t.y = 15 ∧ t.z = 6 →
  i.a = 13/34 ∧ i.b = 15/34 ∧ i.c = 6/34 ∧
  c.p = 1/3 ∧ c.q = 1/3 ∧ c.r = 1/3 := by
  sorry

end triangle_centers_l7_785


namespace book_price_problem_l7_741

theorem book_price_problem (cost_price : ℝ) : 
  (110 / 100 * cost_price = 1100) → 
  (80 / 100 * cost_price = 800) := by
  sorry

end book_price_problem_l7_741


namespace parkway_elementary_soccer_l7_740

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) (boys_soccer_percent : ℚ) :
  total_students = 470 →
  boys = 300 →
  soccer_players = 250 →
  boys_soccer_percent = 86 / 100 →
  (total_students - boys) - (soccer_players - (boys_soccer_percent * soccer_players).floor) = 135 := by
sorry

end parkway_elementary_soccer_l7_740


namespace winter_carnival_participants_l7_786

theorem winter_carnival_participants (total_students : ℕ) (total_participants : ℕ) 
  (girls : ℕ) (boys : ℕ) (h1 : total_students = 1500) 
  (h2 : total_participants = 900) (h3 : girls + boys = total_students) 
  (h4 : (3 * girls / 4 : ℚ) + (2 * boys / 3 : ℚ) = total_participants) : 
  3 * girls / 4 = 900 := by
  sorry

end winter_carnival_participants_l7_786


namespace solution_set_when_a_is_one_range_of_a_l7_701

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x < 6} = Set.Ioo (-3) 3 := by sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ (m n : ℝ), m > 0 → n > 0 → m + n = 1 → 
    ∃ x₀ : ℝ, 1/m + 1/n ≥ f a x₀} = Set.Icc (-5) 3 := by sorry

end solution_set_when_a_is_one_range_of_a_l7_701


namespace four_numbers_lcm_l7_702

theorem four_numbers_lcm (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a + b + c + d = 2020 →
  Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 202 →
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 2424 := by
sorry

end four_numbers_lcm_l7_702


namespace binomial_expansion_difference_l7_744

theorem binomial_expansion_difference : 
  3^7 + (Nat.choose 7 2) * 3^5 + (Nat.choose 7 4) * 3^3 + (Nat.choose 7 6) * 3 -
  ((Nat.choose 7 1) * 3^6 + (Nat.choose 7 3) * 3^4 + (Nat.choose 7 5) * 3^2 + 1) = 128 := by
  sorry

end binomial_expansion_difference_l7_744


namespace apple_trees_count_l7_780

/-- The number of apple trees in the orchard -/
def num_apple_trees : ℕ := 30

/-- The yield of apples per apple tree in kg -/
def apple_yield : ℕ := 150

/-- The number of peach trees in the orchard -/
def num_peach_trees : ℕ := 45

/-- The average yield of peaches per peach tree in kg -/
def peach_yield : ℕ := 65

/-- The total mass of fruit harvested in kg -/
def total_harvest : ℕ := 7425

/-- Theorem stating that the number of apple trees is correct given the conditions -/
theorem apple_trees_count :
  num_apple_trees * apple_yield + num_peach_trees * peach_yield = total_harvest :=
by sorry


end apple_trees_count_l7_780


namespace tailor_cut_difference_l7_732

theorem tailor_cut_difference (dress_outer dress_middle dress_inner pants_outer pants_inner : ℝ) 
  (h1 : dress_outer = 0.75)
  (h2 : dress_middle = 0.60)
  (h3 : dress_inner = 0.55)
  (h4 : pants_outer = 0.50)
  (h5 : pants_inner = 0.45) :
  (dress_outer + dress_middle + dress_inner) - (pants_outer + pants_inner) = 0.95 := by
  sorry

end tailor_cut_difference_l7_732


namespace smallest_k_for_largest_three_digit_prime_l7_750

theorem smallest_k_for_largest_three_digit_prime (p k : ℕ) : 
  p = 997 →  -- p is the largest 3-digit prime
  k > 0 →    -- k is positive
  (∀ m : ℕ, m > 0 ∧ m < k → ¬(10 ∣ (p^2 - m))) →  -- k is the smallest such positive integer
  (10 ∣ (p^2 - k)) →  -- p^2 - k is divisible by 10
  k = 9 :=  -- k equals 9
by sorry

end smallest_k_for_largest_three_digit_prime_l7_750


namespace distance_between_points_l7_747

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3.5, -2)
  let p2 : ℝ × ℝ := (7.5, 5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 65 := by
sorry

end distance_between_points_l7_747


namespace rearrangement_theorem_l7_727

/-- Represents the number of people in the line -/
def n : ℕ := 8

/-- Represents the number of people moved to the front -/
def k : ℕ := 3

/-- Calculates the number of ways to rearrange people in a line
    under the given conditions -/
def rearrangement_count (n k : ℕ) : ℕ :=
  (n - k - 1) * (n - k) * (n - k + 1)

/-- The theorem stating that the number of rearrangements is 210 -/
theorem rearrangement_theorem :
  rearrangement_count n k = 210 := by sorry

end rearrangement_theorem_l7_727


namespace first_storm_rate_l7_736

/-- Represents the rainfall data for a week with two rainstorms -/
structure RainfallData where
  firstStormRate : ℝ
  secondStormRate : ℝ
  totalRainTime : ℝ
  totalRainfall : ℝ
  firstStormDuration : ℝ

/-- Theorem stating that given the rainfall conditions, the first storm's rate was 30 mm/hour -/
theorem first_storm_rate (data : RainfallData)
    (h1 : data.secondStormRate = 15)
    (h2 : data.totalRainTime = 45)
    (h3 : data.totalRainfall = 975)
    (h4 : data.firstStormDuration = 20) :
    data.firstStormRate = 30 := by
  sorry

#check first_storm_rate

end first_storm_rate_l7_736


namespace binomial_prime_divisors_l7_766

theorem binomial_prime_divisors (k : ℕ+) :
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → (Nat.choose n k.val).factors.card ≥ k.val := by
  sorry

end binomial_prime_divisors_l7_766


namespace triangle_area_function_l7_738

theorem triangle_area_function (A B C : ℝ) (a b c : ℝ) (x y : ℝ) :
  -- Given conditions
  A = π / 6 →
  a = 2 →
  0 < x →
  x < 5 * π / 6 →
  B = x →
  C = 5 * π / 6 - x →
  -- Area function
  y = 4 * Real.sin x * Real.sin (5 * π / 6 - x) →
  -- Prove
  0 < y ∧ y ≤ 2 + Real.sqrt 3 :=
by sorry

end triangle_area_function_l7_738


namespace bus_journey_speed_l7_729

/-- Given a bus journey with specific conditions, prove the average speed for the remaining distance -/
theorem bus_journey_speed (total_distance : ℝ) (total_time : ℝ) (partial_distance : ℝ) (partial_speed : ℝ)
  (h1 : total_distance = 250)
  (h2 : total_time = 6)
  (h3 : partial_distance = 220)
  (h4 : partial_speed = 40)
  (h5 : partial_distance / partial_speed + (total_distance - partial_distance) / (total_time - partial_distance / partial_speed) = total_time) :
  (total_distance - partial_distance) / (total_time - partial_distance / partial_speed) = 60 := by
  sorry

#check bus_journey_speed

end bus_journey_speed_l7_729


namespace cheapest_candle_combination_l7_777

/-- Represents a candle with its burning time and cost -/
structure Candle where
  burn_time : ℕ
  cost : ℕ

/-- Finds the minimum cost to measure exactly one minute using given candles -/
def min_cost_to_measure_one_minute (candles : List Candle) : ℕ :=
  sorry

/-- The problem statement -/
theorem cheapest_candle_combination :
  let big_candle : Candle := { burn_time := 16, cost := 16 }
  let small_candle : Candle := { burn_time := 7, cost := 7 }
  let candles : List Candle := [big_candle, small_candle]
  min_cost_to_measure_one_minute candles = 97 :=
sorry

end cheapest_candle_combination_l7_777


namespace odd_function_proof_l7_731

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define h in terms of f
def h (x : ℝ) : ℝ := f x - 9

-- State the theorem
theorem odd_function_proof :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  h 1 = 2 →               -- h(1) = 2
  f (-1) = -11 :=         -- Conclusion: f(-1) = -11
by
  sorry  -- Proof is omitted as per instructions

end odd_function_proof_l7_731


namespace parabola_equation_l7_709

/-- A parabola with vertex at the origin and axis of symmetry along a coordinate axis -/
structure Parabola where
  a : ℝ
  axis : Bool -- true for y-axis, false for x-axis

/-- The point (-4, -2) -/
def P : ℝ × ℝ := (-4, -2)

/-- Check if a point satisfies the parabola equation -/
def satisfiesEquation (p : Parabola) (point : ℝ × ℝ) : Prop :=
  if p.axis then
    point.2^2 = p.a * point.1
  else
    point.1^2 = p.a * point.2

theorem parabola_equation :
  ∃ (p1 p2 : Parabola),
    satisfiesEquation p1 P ∧
    satisfiesEquation p2 P ∧
    p1.axis = true ∧
    p2.axis = false ∧
    p1.a = -1 ∧
    p2.a = -8 :=
  sorry

end parabola_equation_l7_709


namespace angle_property_equivalence_l7_783

theorem angle_property_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by sorry

end angle_property_equivalence_l7_783


namespace smallest_number_satisfying_conditions_l7_752

theorem smallest_number_satisfying_conditions : ∃ n : ℕ,
  (n % 6 = 1) ∧ (n % 8 = 3) ∧ (n % 9 = 2) ∧
  (∀ m : ℕ, m < n → ¬((m % 6 = 1) ∧ (m % 8 = 3) ∧ (m % 9 = 2))) ∧
  n = 107 := by
  sorry

end smallest_number_satisfying_conditions_l7_752


namespace jakes_lawn_mowing_time_l7_743

/-- Jake's lawn mowing problem -/
theorem jakes_lawn_mowing_time
  (desired_hourly_rate : ℝ)
  (flower_planting_time : ℝ)
  (flower_planting_charge : ℝ)
  (lawn_mowing_pay : ℝ)
  (h1 : desired_hourly_rate = 20)
  (h2 : flower_planting_time = 2)
  (h3 : flower_planting_charge = 45)
  (h4 : lawn_mowing_pay = 15) :
  (flower_planting_charge + lawn_mowing_pay) / desired_hourly_rate - flower_planting_time = 1 := by
  sorry

#check jakes_lawn_mowing_time

end jakes_lawn_mowing_time_l7_743


namespace trigonometric_equation_solution_l7_762

theorem trigonometric_equation_solution (x : ℝ) :
  8.471 * (3 * Real.tan x - Real.tan x ^ 3) / (2 - 1 / Real.cos x ^ 2) = 
  (4 + 2 * Real.cos (6 * x / 5)) / (Real.cos (3 * x) + Real.cos x) ↔
  ∃ k : ℤ, x = 5 * π / 6 + 10 * π * k / 3 ∧ ¬∃ t : ℤ, k = 2 + 3 * t :=
by sorry

end trigonometric_equation_solution_l7_762


namespace mary_sugar_needed_l7_764

/-- Given a recipe that requires a certain amount of sugar and an amount already added,
    calculate the remaining amount of sugar needed. -/
def sugar_needed (recipe_requirement : ℕ) (already_added : ℕ) : ℕ :=
  recipe_requirement - already_added

/-- Prove that Mary needs to add 3 more cups of sugar. -/
theorem mary_sugar_needed : sugar_needed 7 4 = 3 := by
  sorry

end mary_sugar_needed_l7_764


namespace unique_base_number_l7_779

theorem unique_base_number : ∃! (x : ℕ), x < 6 ∧ x^23 % 6 = 4 := by
  sorry

end unique_base_number_l7_779


namespace exists_line_and_circle_through_origin_l7_770

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define a line passing through (0, -2)
def line_through_point (k : ℝ) (x y : ℝ) : Prop := y = k * x - 2

-- Define two points on the intersection of the line and the ellipse
def intersection_points (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  is_on_ellipse x₁ y₁ ∧ is_on_ellipse x₂ y₂ ∧
  line_through_point k x₁ y₁ ∧ line_through_point k x₂ y₂ ∧
  x₁ ≠ x₂

-- Define the condition for a circle with diameter AB passing through the origin
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- The main theorem
theorem exists_line_and_circle_through_origin :
  ∃ (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    intersection_points k x₁ y₁ x₂ y₂ ∧
    circle_through_origin x₁ y₁ x₂ y₂ :=
sorry

end exists_line_and_circle_through_origin_l7_770
