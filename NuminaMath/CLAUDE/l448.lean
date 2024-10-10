import Mathlib

namespace distance_sum_squares_l448_44884

theorem distance_sum_squares (z : ℂ) (h : Complex.abs (z - (3 - 3*I)) = 3) :
  let z' := 1 + I
  let z'' := 5 - 5*I  -- reflection of z' about 3 - 3i
  (Complex.abs (z - z'))^2 + (Complex.abs (z - z''))^2 = 101 := by
  sorry

end distance_sum_squares_l448_44884


namespace complex_fraction_equality_l448_44887

theorem complex_fraction_equality : 1 + 1 / (2 + 1 / (2 + 2)) = 13 / 9 := by
  sorry

end complex_fraction_equality_l448_44887


namespace equation_real_roots_range_l448_44897

-- Define the equation
def equation (x m : ℝ) : ℝ := 25 - |x + 1| - 4 * 5 - |x + 1| - m

-- Define the property of having real roots
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, equation x m = 0

-- Theorem statement
theorem equation_real_roots_range :
  ∀ m : ℝ, has_real_roots m ↔ m ∈ Set.Ioo (-3 : ℝ) 0 :=
sorry

end equation_real_roots_range_l448_44897


namespace x_plus_y_value_l448_44855

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 2) (hy : |y| = 5) (hxy : x < y) :
  x + y = 7 ∨ x + y = 3 := by
sorry

end x_plus_y_value_l448_44855


namespace hyperbola_asymptotes_separate_from_circle_l448_44837

/-- The hyperbola with equation x^2 - my^2 and eccentricity 3 has asymptotes that are separate from the circle (  )x^2 + y^2 = 7 -/
theorem hyperbola_asymptotes_separate_from_circle 
  (m : ℝ) 
  (hyperbola : ℝ → ℝ → Prop) 
  (circle : ℝ → ℝ → Prop) 
  (eccentricity : ℝ) :
  (∀ x y, hyperbola x y ↔ x^2 - m*y^2 = 1) →
  (∀ x y, circle x y ↔ x^2 + y^2 = 7) →
  eccentricity = 3 →
  ∃ d : ℝ, d > Real.sqrt 7 ∧ 
    (∀ x y, y = 2*Real.sqrt 2*x ∨ y = -2*Real.sqrt 2*x → 
      d ≤ Real.sqrt ((x - 3)^2 + y^2)) :=
by sorry


end hyperbola_asymptotes_separate_from_circle_l448_44837


namespace arithmetic_sequence_sum_l448_44827

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence, if a₁ + a₉ = 8, then a₂ + a₈ = 8 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 + a 9 = 8 → a 2 + a 8 = 8 := by sorry

end arithmetic_sequence_sum_l448_44827


namespace range_of_a_l448_44828

/-- The function f(x) = a^x - x - a has two zeros -/
def has_two_zeros (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

/-- The main theorem -/
theorem range_of_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  has_two_zeros (fun x => a^x - x - a) → a > 1 := by
  sorry

end range_of_a_l448_44828


namespace central_cell_value_l448_44831

/-- A 3x3 table of real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)

/-- The conditions for the table -/
def satisfies_conditions (t : Table) : Prop :=
  t.a * t.b * t.c = 10 ∧
  t.d * t.e * t.f = 10 ∧
  t.g * t.h * t.i = 10 ∧
  t.a * t.d * t.g = 10 ∧
  t.b * t.e * t.h = 10 ∧
  t.c * t.f * t.i = 10 ∧
  t.a * t.b * t.d * t.e = 3 ∧
  t.b * t.c * t.e * t.f = 3 ∧
  t.d * t.e * t.g * t.h = 3 ∧
  t.e * t.f * t.h * t.i = 3

/-- The theorem statement -/
theorem central_cell_value (t : Table) (h : satisfies_conditions t) : t.e = 0.00081 := by
  sorry

end central_cell_value_l448_44831


namespace worker_efficiency_l448_44886

/-- Given two workers p and q, where p can complete a work in 26 days,
    and p and q together can complete the same work in 16 days,
    prove that p is approximately 1.442% more efficient than q. -/
theorem worker_efficiency (p q : ℝ) (h1 : p > 0) (h2 : q > 0) 
  (h3 : p = 1 / 26) (h4 : p + q = 1 / 16) : 
  ∃ ε > 0, |((p - q) / q) * 100 - 1.442| < ε :=
sorry

end worker_efficiency_l448_44886


namespace original_number_is_point_three_l448_44878

theorem original_number_is_point_three : 
  ∃ x : ℝ, (10 * x = x + 2.7) ∧ (x = 0.3) := by
  sorry

end original_number_is_point_three_l448_44878


namespace quadratic_function_with_specific_properties_l448_44856

theorem quadratic_function_with_specific_properties :
  ∀ (a b x₁ x₂ : ℝ),
    a < 0 →
    b > 0 →
    x₁ ≠ x₂ →
    x₁^2 + a*x₁ + b = 0 →
    x₂^2 + a*x₂ + b = 0 →
    ((x₁ - (-2) = x₂ - x₁) ∨ (x₁ / (-2) = x₂ / x₁)) →
    (∀ x, x^2 + a*x + b = x^2 - 5*x + 4) :=
by sorry

end quadratic_function_with_specific_properties_l448_44856


namespace hotel_charges_l448_44814

theorem hotel_charges (G : ℝ) (h1 : G > 0) : 
  let R := 2 * G
  let P := R * (1 - 0.55)
  P = G * (1 - 0.1) := by
sorry

end hotel_charges_l448_44814


namespace cube_sum_given_sum_and_product_l448_44821

theorem cube_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) 
  (h2 : x * y = 12) : 
  x^3 + y^3 = 640 := by
sorry

end cube_sum_given_sum_and_product_l448_44821


namespace roots_of_polynomial_l448_44851

def p (x : ℂ) : ℂ := 5 * x^5 + 18 * x^3 - 45 * x^2 + 30 * x

theorem roots_of_polynomial :
  ∀ x : ℂ, p x = 0 ↔ x = 0 ∨ x = 1/5 ∨ x = Complex.I * Real.sqrt 3 ∨ x = -Complex.I * Real.sqrt 3 :=
by sorry

end roots_of_polynomial_l448_44851


namespace number_problem_l448_44892

theorem number_problem : ∃ x : ℝ, 0.50 * x = 0.30 * 50 + 13 ∧ x = 56 := by
  sorry

end number_problem_l448_44892


namespace arthur_walk_distance_l448_44817

/-- Represents the distance walked in a single direction -/
structure DirectionalWalk where
  blocks : ℕ
  direction : String

/-- Calculates the total distance walked given a list of directional walks and the length of each block in miles -/
def totalDistance (walks : List DirectionalWalk) (blockLength : ℚ) : ℚ :=
  (walks.map (·.blocks)).sum * blockLength

theorem arthur_walk_distance :
  let eastWalk : DirectionalWalk := ⟨8, "east"⟩
  let northWalk : DirectionalWalk := ⟨10, "north"⟩
  let westWalk : DirectionalWalk := ⟨3, "west"⟩
  let walks : List DirectionalWalk := [eastWalk, northWalk, westWalk]
  let blockLength : ℚ := 1/3
  totalDistance walks blockLength = 7 := by
  sorry

end arthur_walk_distance_l448_44817


namespace chess_game_draw_probability_l448_44874

theorem chess_game_draw_probability 
  (p_a_wins : ℝ) 
  (p_a_not_lose : ℝ) 
  (h1 : p_a_wins = 0.4) 
  (h2 : p_a_not_lose = 0.9) : 
  p_a_not_lose - p_a_wins = 0.5 := by
  sorry

end chess_game_draw_probability_l448_44874


namespace neon_signs_blink_together_l448_44838

theorem neon_signs_blink_together (a b c d : ℕ) 
  (ha : a = 7) (hb : b = 11) (hc : c = 13) (hd : d = 17) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 17017 := by
  sorry

end neon_signs_blink_together_l448_44838


namespace ben_win_probability_l448_44898

theorem ben_win_probability (p_lose : ℚ) (h1 : p_lose = 3/7) (h2 : p_lose + p_win = 1) : p_win = 4/7 := by
  sorry

end ben_win_probability_l448_44898


namespace sock_selection_theorem_l448_44885

theorem sock_selection_theorem :
  let n : ℕ := 7  -- Total number of socks
  let k : ℕ := 4  -- Number of socks to choose
  Nat.choose n k = 35 := by
  sorry

end sock_selection_theorem_l448_44885


namespace coin_problem_l448_44812

/-- Proves that Tom has 8 quarters given the conditions of the coin problem -/
theorem coin_problem (total_coins : ℕ) (total_value : ℚ) 
  (quarter_value nickel_value : ℚ) : 
  total_coins = 12 →
  total_value = 11/5 →
  quarter_value = 1/4 →
  nickel_value = 1/20 →
  ∃ (quarters nickels : ℕ),
    quarters + nickels = total_coins ∧
    quarter_value * quarters + nickel_value * nickels = total_value ∧
    quarters = 8 := by
  sorry

end coin_problem_l448_44812


namespace modular_congruence_solution_l448_44804

theorem modular_congruence_solution :
  ∀ m n : ℕ,
  0 ≤ m ∧ m ≤ 17 →
  0 ≤ n ∧ n ≤ 13 →
  m ≡ 98765 [MOD 18] →
  n ≡ 98765 [MOD 14] →
  m = 17 ∧ n = 9 := by
sorry

end modular_congruence_solution_l448_44804


namespace raw_materials_cost_l448_44893

/-- The total amount Kanul had --/
def total : ℝ := 5714.29

/-- The amount spent on machinery --/
def machinery : ℝ := 1000

/-- The percentage of total amount kept as cash --/
def cash_percentage : ℝ := 0.30

/-- The amount spent on raw materials --/
def raw_materials : ℝ := total - machinery - (cash_percentage * total)

/-- Theorem stating that the amount spent on raw materials is approximately $3000.00 --/
theorem raw_materials_cost : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |raw_materials - 3000| < ε := by
  sorry

end raw_materials_cost_l448_44893


namespace students_playing_sports_l448_44859

theorem students_playing_sports (A B : Finset ℕ) : 
  A.card = 7 → B.card = 8 → (A ∩ B).card = 3 → (A ∪ B).card = 12 := by
  sorry

end students_playing_sports_l448_44859


namespace existence_of_n_l448_44890

theorem existence_of_n (k : ℕ+) : ∃ n : ℤ, 
  Real.sqrt (n + 1981^k.val : ℝ) + Real.sqrt (n : ℝ) = (Real.sqrt 1982 + 1)^k.val := by
  sorry

end existence_of_n_l448_44890


namespace last_four_average_l448_44850

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 65 →
  (list.take 3).sum / 3 = 60 →
  (list.drop 3).sum / 4 = 68.75 := by
  sorry

end last_four_average_l448_44850


namespace distance_between_cities_l448_44839

/-- Represents the travel scenario of two cars between two cities -/
structure TravelScenario where
  v : ℝ  -- Speed of car A in km/min
  x : ℝ  -- Total travel time for car A in minutes
  d : ℝ  -- Distance between the two cities in km

/-- Conditions of the travel scenario -/
def travel_conditions (s : TravelScenario) : Prop :=
  -- Both cars travel the same distance in first 5 minutes
  -- Car B's speed reduces to 2/5 of original after 5 minutes
  -- Car B arrives 15 minutes after car A
  (5 * s.v - 25) / 2 = s.x - 5 + 15 ∧
  -- If failure occurred 4 km farther, B would arrive 10 minutes after A
  25 - 10 / s.v = 20 - 4 / s.v ∧
  -- Total distance is speed multiplied by time
  s.d = s.v * s.x

/-- The main theorem stating the distance between the cities -/
theorem distance_between_cities :
  ∀ s : TravelScenario, travel_conditions s → s.d = 18 :=
by sorry

end distance_between_cities_l448_44839


namespace min_decimal_digits_l448_44803

def fraction : ℚ := 987654321 / (2^30 * 5^6)

theorem min_decimal_digits (n : ℕ) : n = 30 ↔ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, fraction * 10^m ≠ k) ∧ 
  (∃ k : ℕ, fraction * 10^n = k) :=
sorry

end min_decimal_digits_l448_44803


namespace total_players_on_ground_l448_44879

theorem total_players_on_ground (cricket hockey football softball : ℕ) : 
  cricket = 15 → hockey = 12 → football = 13 → softball = 15 →
  cricket + hockey + football + softball = 55 := by
  sorry

end total_players_on_ground_l448_44879


namespace hockey_league_teams_l448_44867

/-- The number of teams in a hockey league --/
def num_teams : ℕ := 18

/-- The number of times each team faces every other team --/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season --/
def total_games : ℕ := 1530

/-- Theorem: Given the conditions, the number of teams in the league is 18 --/
theorem hockey_league_teams :
  (num_teams * (num_teams - 1) * games_per_pair) / 2 = total_games :=
sorry

end hockey_league_teams_l448_44867


namespace three_card_selections_standard_deck_l448_44843

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (red_suits : Nat)
  (black_suits : Nat)

/-- A standard deck of cards -/
def standard_deck : Deck :=
  { total_cards := 52
  , num_suits := 4
  , cards_per_suit := 13
  , red_suits := 2
  , black_suits := 2 }

/-- The number of ways to choose three different cards in a specific order -/
def three_card_selections (d : Deck) : Nat :=
  d.total_cards * (d.total_cards - 1) * (d.total_cards - 2)

/-- Theorem stating that the number of ways to choose three different cards
    in a specific order from a standard deck is 132600 -/
theorem three_card_selections_standard_deck :
  three_card_selections standard_deck = 132600 := by
  sorry

end three_card_selections_standard_deck_l448_44843


namespace repeating_decimal_simplest_form_sum_of_numerator_and_denominator_l448_44811

def repeating_decimal : ℚ := 24/99

theorem repeating_decimal_simplest_form : 
  repeating_decimal = 8/33 := by sorry

theorem sum_of_numerator_and_denominator : 
  (Nat.gcd 8 33 = 1) ∧ (8 + 33 = 41) := by sorry

end repeating_decimal_simplest_form_sum_of_numerator_and_denominator_l448_44811


namespace max_value_of_a_l448_44880

theorem max_value_of_a (a b c : ℝ) (sum_zero : a + b + c = 0) (sum_squares_six : a^2 + b^2 + c^2 = 6) :
  ∀ x : ℝ, (∃ y z : ℝ, x + y + z = 0 ∧ x^2 + y^2 + z^2 = 6) → x ≤ 2 :=
by sorry

end max_value_of_a_l448_44880


namespace geometric_series_sum_l448_44832

theorem geometric_series_sum (a b : ℝ) (h : ∑' n, a / b^n = 7) :
  ∑' n, a / (a + 2*b)^n = (7*(b-1)) / (9*b-8) := by
  sorry

end geometric_series_sum_l448_44832


namespace tan_sum_pi_fourth_l448_44866

theorem tan_sum_pi_fourth (θ : Real) (h : Real.tan θ = 1/3) : 
  Real.tan (θ + π/4) = 2 := by
  sorry

end tan_sum_pi_fourth_l448_44866


namespace heptagon_diagonals_l448_44800

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon is a polygon with 7 sides -/
def is_heptagon (n : ℕ) : Prop := n = 7

theorem heptagon_diagonals (n : ℕ) (h : is_heptagon n) : num_diagonals n = 14 := by
  sorry

end heptagon_diagonals_l448_44800


namespace initial_water_ratio_l448_44862

/-- Proves that the ratio of initial water to tank capacity is 1:2 given the specified conditions --/
theorem initial_water_ratio (tank_capacity : ℝ) (inflow_rate : ℝ) (outflow_rate1 : ℝ) (outflow_rate2 : ℝ) (fill_time : ℝ) :
  tank_capacity = 6000 →
  inflow_rate = 500 →
  outflow_rate1 = 250 →
  outflow_rate2 = 1000 / 6 →
  fill_time = 36 →
  (tank_capacity - (inflow_rate - outflow_rate1 - outflow_rate2) * fill_time) / tank_capacity = 1 / 2 := by
  sorry

end initial_water_ratio_l448_44862


namespace age_problem_l448_44819

theorem age_problem (parent_age son_age : ℕ) : 
  parent_age = 3 * son_age ∧ 
  parent_age + 5 = (5/2) * (son_age + 5) →
  parent_age = 45 ∧ son_age = 15 := by
sorry

end age_problem_l448_44819


namespace square_land_side_length_l448_44860

theorem square_land_side_length (area : ℝ) (h : area = Real.sqrt 625) :
  ∃ (side : ℝ), side * side = area ∧ side = 25 := by
  sorry

end square_land_side_length_l448_44860


namespace sales_volume_equation_l448_44825

def daily_sales_volume (x : ℝ) : ℝ := -x + 38

theorem sales_volume_equation :
  (∀ x y : ℝ, y = daily_sales_volume x → (x = 13 → y = 25) ∧ (x = 18 → y = 20)) ∧
  (daily_sales_volume 13 = 25) ∧
  (daily_sales_volume 18 = 20) :=
by sorry

end sales_volume_equation_l448_44825


namespace equilateral_triangle_cd_product_l448_44823

/-- Given an equilateral triangle with vertices at (0,0), (c,15), and (d,47),
    the product cd equals 1216√3/9 -/
theorem equilateral_triangle_cd_product (c d : ℝ) : 
  (∀ (z : ℂ), z ^ 3 = 1 ∧ z ≠ 1 → (c + 15 * I) * z = d + 47 * I) →
  c * d = 1216 * Real.sqrt 3 / 9 := by
  sorry

end equilateral_triangle_cd_product_l448_44823


namespace max_omega_for_increasing_g_l448_44871

/-- Given a function f and its translation g, proves that the maximum value of ω is 2 
    when g is increasing on [0, π/4] -/
theorem max_omega_for_increasing_g (ω : ℝ) (f g : ℝ → ℝ) : 
  ω > 0 → 
  (∀ x, f x = 2 * Real.sin (ω * x - π / 8)) →
  (∀ x, g x = f (x + π / (8 * ω))) →
  (∀ x ∈ Set.Icc 0 (π / 4), Monotone g) →
  ω ≤ 2 :=
sorry

end max_omega_for_increasing_g_l448_44871


namespace road_trip_duration_l448_44842

theorem road_trip_duration (family_size : ℕ) (water_per_person_per_hour : ℚ) 
  (total_water_bottles : ℕ) (h : ℕ) : 
  family_size = 4 → 
  water_per_person_per_hour = 1/2 → 
  total_water_bottles = 32 → 
  (2 * h : ℚ) * (family_size : ℚ) * water_per_person_per_hour = total_water_bottles → 
  h = 8 := by
sorry

end road_trip_duration_l448_44842


namespace drummer_tosses_six_sets_l448_44895

/-- Calculates the number of drum stick sets tossed to the audience after each show -/
def drumSticksTossedPerShow (setsPerShow : ℕ) (totalNights : ℕ) (totalSetsUsed : ℕ) : ℕ :=
  ((totalSetsUsed - setsPerShow * totalNights) / totalNights)

/-- Theorem: Given the conditions, the drummer tosses 6 sets of drum sticks after each show -/
theorem drummer_tosses_six_sets :
  drumSticksTossedPerShow 5 30 330 = 6 := by
  sorry

end drummer_tosses_six_sets_l448_44895


namespace triangle_area_l448_44820

/-- The area of a triangle with base 30 inches and height 18 inches is 270 square inches. -/
theorem triangle_area (base height : ℝ) (h1 : base = 30) (h2 : height = 18) :
  (1 / 2 : ℝ) * base * height = 270 :=
by sorry

end triangle_area_l448_44820


namespace inequality_solution_l448_44896

theorem inequality_solution (x : ℝ) : 
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + 2 * x)) ↔ 
  x ∈ Set.Icc (-12/7) (-3/4) ∧ x ≠ -3/4 :=
sorry

end inequality_solution_l448_44896


namespace average_text_messages_l448_44830

/-- Calculate the average number of text messages sent over 5 days -/
theorem average_text_messages 
  (day1 : ℕ) 
  (day2 : ℕ) 
  (day3_to_5 : ℕ) 
  (h1 : day1 = 220) 
  (h2 : day2 = day1 / 2) 
  (h3 : day3_to_5 = 50) :
  (day1 + day2 + 3 * day3_to_5) / 5 = 96 := by
  sorry

end average_text_messages_l448_44830


namespace difference_of_sums_l448_44833

/-- Sum of first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- Sum of first n even numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The number of odd numbers from 1 to 2011 -/
def oddCount : ℕ := 1006

/-- The number of even numbers from 2 to 2010 -/
def evenCount : ℕ := 1005

theorem difference_of_sums : sumOddNumbers oddCount - sumEvenNumbers evenCount = 1006 := by
  sorry

end difference_of_sums_l448_44833


namespace cos_2017_pi_thirds_l448_44876

theorem cos_2017_pi_thirds : Real.cos (2017 * Real.pi / 3) = 1 / 2 := by
  sorry

end cos_2017_pi_thirds_l448_44876


namespace car_rental_rate_proof_l448_44805

/-- The daily rate of the first car rental company -/
def first_company_rate : ℝ := 17.99

/-- The per-mile rate of the first car rental company -/
def first_company_per_mile : ℝ := 0.18

/-- The daily rate of City Rentals -/
def city_rentals_rate : ℝ := 18.95

/-- The per-mile rate of City Rentals -/
def city_rentals_per_mile : ℝ := 0.16

/-- The number of miles at which the cost is the same for both companies -/
def equal_cost_miles : ℝ := 48

theorem car_rental_rate_proof :
  first_company_rate + first_company_per_mile * equal_cost_miles =
  city_rentals_rate + city_rentals_per_mile * equal_cost_miles :=
by sorry

end car_rental_rate_proof_l448_44805


namespace exam_results_l448_44826

theorem exam_results (failed_hindi : ℝ) (failed_english : ℝ) (failed_both : ℝ)
  (h1 : failed_hindi = 20)
  (h2 : failed_english = 70)
  (h3 : failed_both = 10) :
  100 - (failed_hindi + failed_english - failed_both) = 20 := by
  sorry

end exam_results_l448_44826


namespace calories_per_dollar_difference_l448_44870

-- Define the given conditions
def burrito_count : ℕ := 10
def burrito_price : ℚ := 6
def burrito_calories : ℕ := 120
def burger_count : ℕ := 5
def burger_price : ℚ := 8
def burger_calories : ℕ := 400

-- Define the theorem
theorem calories_per_dollar_difference :
  (burger_count * burger_calories : ℚ) / burger_price -
  (burrito_count * burrito_calories : ℚ) / burrito_price = 50 := by
  sorry

end calories_per_dollar_difference_l448_44870


namespace exponential_function_point_l448_44877

theorem exponential_function_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (fun x => a^x) 2 = 9 → a = 3 := by
  sorry

end exponential_function_point_l448_44877


namespace sum_of_tenth_powers_l448_44801

theorem sum_of_tenth_powers (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -1) :
  a^10 + b^10 = 123 := by sorry

end sum_of_tenth_powers_l448_44801


namespace coefficient_x3y5_in_x_plus_y_8_l448_44882

theorem coefficient_x3y5_in_x_plus_y_8 :
  Finset.sum (Finset.range 9) (λ k => Nat.choose 8 k * (if k = 3 then 1 else 0)) = 56 := by
  sorry

end coefficient_x3y5_in_x_plus_y_8_l448_44882


namespace base_for_256_with_4_digits_l448_44872

theorem base_for_256_with_4_digits : ∃ (b : ℕ), b = 5 ∧ b^3 ≤ 256 ∧ 256 < b^4 ∧ ∀ (x : ℕ), x < b → (x^3 ≤ 256 → 256 ≥ x^4) := by
  sorry

end base_for_256_with_4_digits_l448_44872


namespace ripe_apples_theorem_l448_44852

-- Define the universe of discourse
variable (Basket : Type)
-- Define the property of being ripe
variable (isRipe : Basket → Prop)

-- Define the statement "All apples in this basket are ripe" is false
axiom not_all_ripe : ¬(∀ (apple : Basket), isRipe apple)

-- Theorem to prove
theorem ripe_apples_theorem :
  (∃ (apple : Basket), ¬(isRipe apple)) ∧
  (¬(∀ (apple : Basket), isRipe apple)) := by
  sorry

end ripe_apples_theorem_l448_44852


namespace lattice_points_on_quadratic_l448_44869

def is_lattice_point (x y : ℤ) : Prop :=
  y = (x^2 / 10) - (x / 10) + 9 / 5 ∧ y ≤ abs x

theorem lattice_points_on_quadratic :
  ∀ x y : ℤ, is_lattice_point x y ↔ 
    (x = 2 ∧ y = 2) ∨ 
    (x = 4 ∧ y = 3) ∨ 
    (x = 7 ∧ y = 6) ∨ 
    (x = 9 ∧ y = 9) ∨ 
    (x = -6 ∧ y = 6) ∨ 
    (x = -3 ∧ y = 3) :=
by sorry


end lattice_points_on_quadratic_l448_44869


namespace triangle_problem_l448_44863

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : (t.a + t.c) / (t.a + t.b) = (t.b - t.a) / t.c)
  (h2 : t.b = Real.sqrt 7)
  (h3 : Real.sin t.C = 2 * Real.sin t.A) :
  t.B = 2 * π / 3 ∧ min t.a (min t.b t.c) = 1 := by
  sorry


end triangle_problem_l448_44863


namespace mixture_problem_l448_44810

/-- Proves that the initial amount of liquid A is 16 liters given the conditions of the mixture problem -/
theorem mixture_problem (x : ℝ) : 
  x > 0 ∧ 
  (4*x) / x = 4 / 1 ∧ 
  (4*x - 8) / (x + 8) = 2 / 3 → 
  4*x = 16 := by
sorry

end mixture_problem_l448_44810


namespace probability_at_least_three_aces_l448_44834

/-- The number of cards in a standard deck without jokers -/
def deck_size : ℕ := 52

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The number of cards drawn -/
def draw_size : ℕ := 5

/-- The probability of drawing at least 3 Aces when randomly selecting 5 cards from a standard 52-card deck (without jokers) -/
theorem probability_at_least_three_aces :
  (Nat.choose num_aces 3 * Nat.choose (deck_size - num_aces) 2 +
   Nat.choose num_aces 4 * Nat.choose (deck_size - num_aces) 1) /
  Nat.choose deck_size draw_size =
  (Nat.choose 4 3 * Nat.choose 48 2 + Nat.choose 4 4 * Nat.choose 48 1) /
  Nat.choose 52 5 := by
  sorry

end probability_at_least_three_aces_l448_44834


namespace sum_of_arguments_l448_44864

def complex_pow_eq (z : ℂ) : Prop := z^5 = -32 * Complex.I

theorem sum_of_arguments (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h₁ : complex_pow_eq z₁) (h₂ : complex_pow_eq z₂) (h₃ : complex_pow_eq z₃) 
  (h₄ : complex_pow_eq z₄) (h₅ : complex_pow_eq z₅) 
  (distinct : z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₁ ≠ z₅ ∧ 
              z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₂ ≠ z₅ ∧ 
              z₃ ≠ z₄ ∧ z₃ ≠ z₅ ∧ 
              z₄ ≠ z₅) :
  Complex.arg z₁ + Complex.arg z₂ + Complex.arg z₃ + Complex.arg z₄ + Complex.arg z₅ = 
  990 * (π / 180) := by
  sorry

end sum_of_arguments_l448_44864


namespace atMostOneHead_atLeastTwoHeads_mutually_exclusive_atMostOneHead_atLeastTwoHeads_cover_all_l448_44889

-- Define the sample space for tossing two coins
inductive CoinToss
  | HH -- Two heads
  | HT -- Head then tail
  | TH -- Tail then head
  | TT -- Two tails

-- Define the events
def atMostOneHead (outcome : CoinToss) : Prop :=
  outcome = CoinToss.HT ∨ outcome = CoinToss.TH ∨ outcome = CoinToss.TT

def atLeastTwoHeads (outcome : CoinToss) : Prop :=
  outcome = CoinToss.HH

-- Theorem: The events are mutually exclusive
theorem atMostOneHead_atLeastTwoHeads_mutually_exclusive :
  ∀ (outcome : CoinToss), ¬(atMostOneHead outcome ∧ atLeastTwoHeads outcome) :=
by
  sorry

-- Theorem: The events cover all possible outcomes
theorem atMostOneHead_atLeastTwoHeads_cover_all :
  ∀ (outcome : CoinToss), atMostOneHead outcome ∨ atLeastTwoHeads outcome :=
by
  sorry

end atMostOneHead_atLeastTwoHeads_mutually_exclusive_atMostOneHead_atLeastTwoHeads_cover_all_l448_44889


namespace tournament_scheduling_correct_l448_44875

/-- Represents a team in the tournament --/
inductive Team (n : ℕ) where
  | num : Fin n → Team n
  | inf : Team n

/-- A match between two teams --/
structure Match (n : ℕ) where
  team1 : Team n
  team2 : Team n

/-- A round in the tournament --/
def Round (n : ℕ) := List (Match n)

/-- Generate the next round based on the current round --/
def nextRound (n : ℕ) (current : Round n) : Round n :=
  sorry

/-- Check if a round is valid (each team plays exactly once) --/
def isValidRound (n : ℕ) (round : Round n) : Prop :=
  sorry

/-- Check if two teams have played against each other --/
def havePlayedAgainst (n : ℕ) (team1 team2 : Team n) (rounds : List (Round n)) : Prop :=
  sorry

/-- The main theorem: tournament scheduling is correct --/
theorem tournament_scheduling_correct (n : ℕ) (h : n > 1) :
  ∃ (rounds : List (Round n)),
    (rounds.length = n - 1) ∧
    (∀ r ∈ rounds, isValidRound n r) ∧
    (∀ t1 t2 : Team n, t1 ≠ t2 → havePlayedAgainst n t1 t2 rounds) :=
  sorry

end tournament_scheduling_correct_l448_44875


namespace problem_statement_l448_44848

theorem problem_statement (a : ℝ) 
  (A : Set ℝ) (hA : A = {0, 2, a^2})
  (B : Set ℝ) (hB : B = {1, a})
  (hUnion : A ∪ B = {0, 1, 2, 4}) : a = 2 := by
  sorry

end problem_statement_l448_44848


namespace triangle_properties_l448_44815

-- Define the points in the complex plane
def A : ℂ := 1
def B : ℂ := -Complex.I
def C : ℂ := -1 + 2 * Complex.I

-- Define the vectors
def AB : ℂ := B - A
def AC : ℂ := C - A
def BC : ℂ := C - B

-- Theorem statement
theorem triangle_properties :
  (AB.re = -1 ∧ AB.im = -1) ∧
  (AC.re = -2 ∧ AC.im = 2) ∧
  (BC.re = -1 ∧ BC.im = 3) ∧
  (AB.re * AC.re + AB.im * AC.im = 0) := by
  sorry

-- The last condition (AB.re * AC.re + AB.im * AC.im = 0) checks if AB and AC are perpendicular,
-- which implies that the triangle is right-angled.

end triangle_properties_l448_44815


namespace graphing_calculator_theorem_l448_44836

/-- Represents the number of students who brought graphing calculators -/
def graphing_calculator_count : ℕ := 10

/-- Represents the total number of boys in the class -/
def total_boys : ℕ := 20

/-- Represents the total number of girls in the class -/
def total_girls : ℕ := 18

/-- Represents the number of students who brought scientific calculators -/
def scientific_calculator_count : ℕ := 30

/-- Represents the number of girls who brought scientific calculators -/
def girls_with_scientific_calculators : ℕ := 15

theorem graphing_calculator_theorem :
  graphing_calculator_count = 10 ∧
  total_boys + total_girls = scientific_calculator_count + graphing_calculator_count :=
by sorry

end graphing_calculator_theorem_l448_44836


namespace difference_of_squares_factorization_l448_44847

theorem difference_of_squares_factorization (a b p q : ℝ) : 
  (∃ x y, -a^2 + 9 = (x + y) * (x - y)) ∧ 
  (¬∃ x y, -a^2 - b^2 = (x + y) * (x - y)) ∧ 
  (¬∃ x y, p^2 - (-q^2) = (x + y) * (x - y)) ∧ 
  (¬∃ x y, a^2 - b^3 = (x + y) * (x - y)) :=
by sorry

end difference_of_squares_factorization_l448_44847


namespace mrs_hilt_carnival_tickets_cost_l448_44846

/-- Represents the cost and quantity of carnival tickets --/
structure CarnivalTickets where
  kids_usual_cost : ℚ
  kids_usual_quantity : ℕ
  adults_usual_cost : ℚ
  adults_usual_quantity : ℕ
  kids_deal_cost : ℚ
  kids_deal_quantity : ℕ
  adults_deal_cost : ℚ
  adults_deal_quantity : ℕ
  kids_bought : ℕ
  adults_bought : ℕ

/-- Calculates the total cost of carnival tickets --/
def total_cost (tickets : CarnivalTickets) : ℚ :=
  let kids_deal_used := tickets.kids_bought / tickets.kids_deal_quantity
  let kids_usual_used := tickets.kids_bought % tickets.kids_deal_quantity / tickets.kids_usual_quantity
  let adults_deal_used := tickets.adults_bought / tickets.adults_deal_quantity
  let adults_usual_used := tickets.adults_bought % tickets.adults_deal_quantity / tickets.adults_usual_quantity
  kids_deal_used * tickets.kids_deal_cost +
  kids_usual_used * tickets.kids_usual_cost +
  adults_deal_used * tickets.adults_deal_cost +
  adults_usual_used * tickets.adults_usual_cost

/-- Theorem: The total cost of Mrs. Hilt's carnival tickets is $15 --/
theorem mrs_hilt_carnival_tickets_cost :
  let tickets : CarnivalTickets := {
    kids_usual_cost := 1/4,
    kids_usual_quantity := 4,
    adults_usual_cost := 2/3,
    adults_usual_quantity := 3,
    kids_deal_cost := 4,
    kids_deal_quantity := 20,
    adults_deal_cost := 8,
    adults_deal_quantity := 15,
    kids_bought := 24,
    adults_bought := 18
  }
  total_cost tickets = 15 := by sorry


end mrs_hilt_carnival_tickets_cost_l448_44846


namespace triangular_arrangement_rows_l448_44849

/-- The number of cans in a triangular arrangement with n rows -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The proposition to be proved -/
theorem triangular_arrangement_rows : 
  ∃ (n : ℕ), triangular_sum n = 480 - 15 ∧ n = 30 := by
  sorry

end triangular_arrangement_rows_l448_44849


namespace eighteenth_power_digits_l448_44883

/-- The function that returns the list of digits in the decimal representation of a natural number -/
def digits (n : ℕ) : List ℕ :=
  sorry

/-- Theorem stating that 18 is the positive integer whose sixth power's decimal representation
    consists of the digits 0, 1, 2, 2, 2, 3, 4, 4 -/
theorem eighteenth_power_digits :
  ∃! (n : ℕ), n > 0 ∧ digits (n^6) = [3, 4, 0, 1, 2, 2, 2, 4] ∧ n = 18 :=
sorry

end eighteenth_power_digits_l448_44883


namespace horner_v2_equals_22_l448_44854

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => x * acc + a) 0

/-- The polynomial f(x) = x^6 + 6x^4 + 9x^2 + 208 -/
def f (x : ℝ) : ℝ := x^6 + 6*x^4 + 9*x^2 + 208

/-- The coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [1, 0, 6, 0, 9, 0, 208]

/-- Theorem: v₂ = 22 when evaluating f(x) at x = -4 using Horner's method -/
theorem horner_v2_equals_22 :
  let x := -4
  let v₀ := 208
  let v₁ := x * v₀ + 0
  let v₂ := x * v₁ + 9
  v₂ = 22 := by sorry

end horner_v2_equals_22_l448_44854


namespace library_repacking_l448_44829

theorem library_repacking (total_books : Nat) (initial_boxes : Nat) (new_box_size : Nat) 
  (h1 : total_books = 1870)
  (h2 : initial_boxes = 55)
  (h3 : new_box_size = 36) :
  total_books % new_box_size = 34 := by
  sorry

end library_repacking_l448_44829


namespace election_vote_difference_l448_44868

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 7600 → 
  candidate_percentage = 35/100 → 
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 2280 := by
sorry

end election_vote_difference_l448_44868


namespace smallest_gcd_qr_l448_44844

theorem smallest_gcd_qr (p q r : ℕ+) 
  (h1 : Nat.gcd p q = 540)
  (h2 : Nat.gcd p r = 1080) :
  ∃ (m : ℕ+), 
    (∀ (q' r' : ℕ+), Nat.gcd p q' = 540 → Nat.gcd p r' = 1080 → m ≤ Nat.gcd q' r') ∧
    Nat.gcd q r = m :=
  sorry

end smallest_gcd_qr_l448_44844


namespace fourth_term_of_geometric_sequence_l448_44894

def geometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem fourth_term_of_geometric_sequence (a : ℕ → ℝ) :
  geometricSequence a → a 1 = 2 → a 2 = 4 → a 4 = 16 := by
  sorry

end fourth_term_of_geometric_sequence_l448_44894


namespace fencing_cost_proof_l448_44818

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length breadth cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Proves that the total cost of fencing the given rectangular plot is 5300 -/
theorem fencing_cost_proof (length breadth cost_per_meter : ℝ) 
  (h1 : length = 64)
  (h2 : breadth = length - 28)
  (h3 : cost_per_meter = 26.5) :
  total_fencing_cost length breadth cost_per_meter = 5300 := by
  sorry

#eval total_fencing_cost 64 36 26.5

end fencing_cost_proof_l448_44818


namespace circle_and_line_properties_l448_44822

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (t : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ∧ t - y + 1 = 0

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  y = (5/12) * x + 43/12

theorem circle_and_line_properties :
  -- Circle C passes through (0,2) and (2,-2)
  circle_C 0 2 ∧ circle_C 2 (-2) ∧
  -- Line m passes through (1,4)
  line_m 1 4 ∧
  -- The chord length of the intersection between circle C and line m is 6
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_m x₁ y₁ ∧ line_m x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) :=
by
  sorry

#check circle_and_line_properties

end circle_and_line_properties_l448_44822


namespace ron_multiplication_mistake_l448_44888

theorem ron_multiplication_mistake (a b : ℕ) : 
  10 ≤ a ∧ a < 100 →  -- a is a two-digit number
  b < 10 →            -- b is a single-digit number
  a * (b + 10) = 190 →
  a * b = 0 := by
sorry

end ron_multiplication_mistake_l448_44888


namespace point_A_x_range_l448_44835

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 6 = 0

-- Define a point on the circle
def point_on_circle (x y : ℝ) : Prop := circle_M x y

-- Define a point on the line
def point_on_line (x y : ℝ) : Prop := line_l x y

-- Define the angle between three points
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem point_A_x_range :
  ∀ (A B C : ℝ × ℝ),
    point_on_line A.1 A.2 →
    point_on_circle B.1 B.2 →
    point_on_circle C.1 C.2 →
    angle A B C = 60 →
    1 ≤ A.1 ∧ A.1 ≤ 5 :=
by sorry

end point_A_x_range_l448_44835


namespace sum_u_v_l448_44857

theorem sum_u_v (u v : ℚ) (h1 : 5 * u - 6 * v = 19) (h2 : 3 * u + 5 * v = -1) : 
  u + v = 27 / 43 := by
  sorry

end sum_u_v_l448_44857


namespace angle_triple_complement_l448_44865

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end angle_triple_complement_l448_44865


namespace income_expenditure_ratio_l448_44891

def income : ℕ := 21000
def savings : ℕ := 3000
def expenditure : ℕ := income - savings

def ratio_income_expenditure : ℚ := income / expenditure

theorem income_expenditure_ratio :
  ratio_income_expenditure = 7 / 6 := by sorry

end income_expenditure_ratio_l448_44891


namespace curve_is_hyperbola_l448_44841

/-- The curve defined by r = 1 / (1 - sin θ) is a hyperbola -/
theorem curve_is_hyperbola (θ : ℝ) (r : ℝ) :
  r = 1 / (1 - Real.sin θ) → ∃ (a b c d e f : ℝ), 
    a ≠ 0 ∧ c ≠ 0 ∧ a * c < 0 ∧
    ∀ (x y : ℝ), a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0 := by
  sorry

end curve_is_hyperbola_l448_44841


namespace convex_quadrilaterals_count_l448_44816

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of distinct points on the circle -/
def num_points : ℕ := 12

/-- The number of vertices in a quadrilateral -/
def vertices_per_quadrilateral : ℕ := 4

theorem convex_quadrilaterals_count :
  binomial num_points vertices_per_quadrilateral = 495 := by
  sorry

end convex_quadrilaterals_count_l448_44816


namespace tan_11_25_decomposition_l448_44809

theorem tan_11_25_decomposition :
  ∃ (a b c d : ℕ+), 
    (Real.tan (11.25 * Real.pi / 180) = Real.sqrt a - Real.sqrt b + Real.sqrt c - d) ∧
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (a + b + c + d = 4) := by
  sorry

end tan_11_25_decomposition_l448_44809


namespace alcohol_solution_proof_l448_44845

/-- Proves that adding 1.8 litres of pure alcohol to a 6-litre solution
    that is 35% alcohol results in a 50% alcohol solution -/
theorem alcohol_solution_proof :
  let initial_volume : ℝ := 6
  let initial_percentage : ℝ := 0.35
  let target_percentage : ℝ := 0.5
  let added_alcohol : ℝ := 1.8
  let final_volume := initial_volume + added_alcohol
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  (final_alcohol / final_volume) = target_percentage := by sorry

end alcohol_solution_proof_l448_44845


namespace one_shot_each_probability_l448_44858

def yao_rate : ℝ := 0.8
def mcgrady_rate : ℝ := 0.7

theorem one_shot_each_probability :
  let yao_one_shot := 2 * yao_rate * (1 - yao_rate)
  let mcgrady_one_shot := 2 * mcgrady_rate * (1 - mcgrady_rate)
  yao_one_shot * mcgrady_one_shot = 0.1344 := by
sorry

end one_shot_each_probability_l448_44858


namespace quadratic_equation_completing_square_l448_44899

theorem quadratic_equation_completing_square (x : ℝ) : 
  x^2 - 4*x + 3 = 0 → (x - 2)^2 = 1 := by
  sorry

end quadratic_equation_completing_square_l448_44899


namespace canoe_upstream_speed_l448_44853

/-- 
Given a canoe that rows downstream at 12 km/hr and a stream with a speed of 4.5 km/hr,
prove that the speed of the canoe when rowing upstream is 3 km/hr.
-/
theorem canoe_upstream_speed : 
  ∀ (downstream_speed stream_speed : ℝ),
  downstream_speed = 12 →
  stream_speed = 4.5 →
  (downstream_speed - 2 * stream_speed) = 3 :=
by sorry

end canoe_upstream_speed_l448_44853


namespace range_of_m_l448_44840

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = x - 2) →
  (∀ x, g x = x^2 - 2*m*x + 4) →
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 4 5, g x₁ = f x₂) →
  m ∈ Set.Icc (5/4) (Real.sqrt 2) :=
by sorry

end range_of_m_l448_44840


namespace largest_integer_satisfying_inequality_l448_44802

theorem largest_integer_satisfying_inequality : 
  ∀ n : ℕ+, n^200 < 3^500 ↔ n ≤ 15 :=
by
  sorry

end largest_integer_satisfying_inequality_l448_44802


namespace eight_friends_lineup_l448_44806

theorem eight_friends_lineup (n : ℕ) (h : n = 8) : Nat.factorial n = 40320 := by
  sorry

end eight_friends_lineup_l448_44806


namespace monotonic_decreasing_interval_f_monotonic_decreasing_on_open_interval_l448_44808

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 2) ↔ (f_derivative x < 0) :=
by sorry

-- Main theorem
theorem f_monotonic_decreasing_on_open_interval :
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 2 → f y < f x :=
by sorry

end monotonic_decreasing_interval_f_monotonic_decreasing_on_open_interval_l448_44808


namespace complex_magnitude_problem_l448_44813

theorem complex_magnitude_problem (z : ℂ) (h : (1 + 2*I)*z = -1 + 3*I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end complex_magnitude_problem_l448_44813


namespace distinct_colorings_l448_44824

/-- The number of disks in the circle -/
def n : ℕ := 8

/-- The number of blue disks -/
def blue : ℕ := 4

/-- The number of red disks -/
def red : ℕ := 3

/-- The number of green disks -/
def green : ℕ := 1

/-- The number of rotational symmetries -/
def rotations : ℕ := 4

/-- The number of reflection symmetries -/
def reflections : ℕ := 4

/-- The total number of symmetries -/
def total_symmetries : ℕ := rotations + reflections + 1

/-- The number of colorings fixed by identity -/
def fixed_by_identity : ℕ := (n.choose blue) * ((n - blue).choose red)

/-- The number of colorings fixed by each reflection -/
def fixed_by_reflection : ℕ := 6

/-- The number of colorings fixed by each rotation (other than identity) -/
def fixed_by_rotation : ℕ := 0

/-- Theorem: The number of distinct colorings is 38 -/
theorem distinct_colorings : 
  (fixed_by_identity + reflections * fixed_by_reflection + (rotations - 1) * fixed_by_rotation) / total_symmetries = 38 := by
  sorry

end distinct_colorings_l448_44824


namespace complex_equation_solution_l448_44807

theorem complex_equation_solution (c d x : ℂ) (i : ℂ) : 
  c * d = x - 5 * i → 
  Complex.abs c = 3 →
  Complex.abs d = Real.sqrt 50 →
  x = 5 * Real.sqrt 17 :=
by sorry

end complex_equation_solution_l448_44807


namespace remaining_calculation_l448_44861

def calculate_remaining (income : ℝ) : ℝ :=
  let after_rent := income * (1 - 0.15)
  let after_education := after_rent * (1 - 0.15)
  let after_misc := after_education * (1 - 0.10)
  let after_medical := after_misc * (1 - 0.15)
  after_medical

theorem remaining_calculation (income : ℝ) :
  income = 10037.77 →
  calculate_remaining income = 5547.999951125 := by
  sorry

end remaining_calculation_l448_44861


namespace distinguishable_triangles_count_l448_44881

/-- The number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- The number of corner triangles in the large triangle -/
def num_corners : ℕ := 3

/-- The total number of small triangles in the large triangle -/
def total_triangles : ℕ := num_corners + 1

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of distinguishable large triangles -/
def num_distinguishable_triangles : ℕ :=
  (num_colors + 
   num_colors * (num_colors - 1) + 
   choose num_colors num_corners) * num_colors

theorem distinguishable_triangles_count :
  num_distinguishable_triangles = 960 :=
sorry

end distinguishable_triangles_count_l448_44881


namespace dance_partners_exist_l448_44873

-- Define the types for boys and girls
variable {Boy Girl : Type}

-- Define the dance relation
variable (danced : Boy → Girl → Prop)

-- Define the conditions
variable (h1 : ∀ b : Boy, ∃ g : Girl, ¬danced b g)
variable (h2 : ∀ g : Girl, ∃ b : Boy, danced b g)

-- State the theorem
theorem dance_partners_exist :
  ∃ (f f' : Boy) (g g' : Girl), f ≠ f' ∧ g ≠ g' ∧ danced f g ∧ danced f' g' :=
sorry

end dance_partners_exist_l448_44873
