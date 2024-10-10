import Mathlib

namespace correct_road_determination_l624_62428

/-- Represents the two tribes on the island -/
inductive Tribe
| TruthTeller
| Liar

/-- Represents the possible roads -/
inductive Road
| ToVillage
| AwayFromVillage

/-- Represents the possible answers to a question -/
inductive Answer
| Yes
| No

/-- The actual state of the road -/
def actual_road : Road := sorry

/-- The tribe of the islander being asked -/
def islander_tribe : Tribe := sorry

/-- Function that determines how a member of a given tribe would answer a direct question about the road -/
def direct_answer (t : Tribe) (r : Road) : Answer := sorry

/-- Function that determines how an islander would answer the traveler's question -/
def islander_answer (t : Tribe) (r : Road) : Answer := sorry

/-- The traveler's interpretation of the islander's answer -/
def traveler_interpretation (a : Answer) : Road := sorry

theorem correct_road_determination :
  traveler_interpretation (islander_answer islander_tribe actual_road) = actual_road := by sorry

end correct_road_determination_l624_62428


namespace special_shape_perimeter_l624_62462

/-- A shape with right angles, a base of 12 feet, 10 congruent sides of 2 feet each, and an area of 132 square feet -/
structure SpecialShape where
  base : ℝ
  congruent_side : ℝ
  num_congruent_sides : ℕ
  area : ℝ
  base_eq : base = 12
  congruent_side_eq : congruent_side = 2
  num_congruent_sides_eq : num_congruent_sides = 10
  area_eq : area = 132

/-- The perimeter of the SpecialShape is 54 feet -/
theorem special_shape_perimeter (s : SpecialShape) : 
  s.base + s.congruent_side * s.num_congruent_sides + 4 + 6 = 54 := by
  sorry

end special_shape_perimeter_l624_62462


namespace triangle_properties_l624_62478

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (a + b) / Real.sin (A + B) = (a - c) / (Real.sin A - Real.sin B) →
  b = 3 →
  Real.cos A = Real.sqrt 6 / 3 →
  B = π / 3 ∧
  (1 / 2) * a * b * Real.sin C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 2 :=
by sorry

end triangle_properties_l624_62478


namespace laptop_sale_price_l624_62434

def original_price : ℝ := 500
def first_discount : ℝ := 0.10
def second_discount : ℝ := 0.20
def delivery_fee : ℝ := 30

theorem laptop_sale_price :
  let price_after_first_discount := original_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  let final_price := price_after_second_discount + delivery_fee
  final_price = 390 := by sorry

end laptop_sale_price_l624_62434


namespace value_of_b_minus_d_squared_l624_62496

theorem value_of_b_minus_d_squared 
  (a b c d : ℤ) 
  (eq1 : a - b - c + d = 13) 
  (eq2 : a + b - c - d = 3) : 
  (b - d)^2 = 25 := by
  sorry

end value_of_b_minus_d_squared_l624_62496


namespace aladdin_travel_l624_62499

/-- A continuous function that takes all values in [0,1) -/
def equator_travel (φ : ℝ → ℝ) : Prop :=
  Continuous φ ∧ ∀ y : ℝ, 0 ≤ y ∧ y < 1 → ∃ t : ℝ, φ t = y

/-- The maximum difference between any two values of φ is at least 1 -/
theorem aladdin_travel (φ : ℝ → ℝ) (h : equator_travel φ) :
  ∃ t₁ t₂ : ℝ, |φ t₁ - φ t₂| ≥ 1 :=
sorry

end aladdin_travel_l624_62499


namespace ms_delmont_class_size_l624_62421

/-- Proves the number of students in Ms. Delmont's class given the cupcake distribution -/
theorem ms_delmont_class_size 
  (total_cupcakes : ℕ)
  (adults_received : ℕ)
  (mrs_donnelly_class : ℕ)
  (leftover_cupcakes : ℕ)
  (h1 : total_cupcakes = 40)
  (h2 : adults_received = 4)
  (h3 : mrs_donnelly_class = 16)
  (h4 : leftover_cupcakes = 2) :
  total_cupcakes - adults_received - mrs_donnelly_class - leftover_cupcakes = 18 := by
  sorry

#check ms_delmont_class_size

end ms_delmont_class_size_l624_62421


namespace max_tan_B_in_triangle_l624_62416

/-- In a triangle ABC, given that 3a*cos(C) + b = 0, the maximum value of tan(B) is 3/4 -/
theorem max_tan_B_in_triangle (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  3 * a * Real.cos C + b = 0 →
  ∀ (a' b' c' : ℝ) (A' B' C' : ℝ),
    a' > 0 → b' > 0 → c' > 0 →
    A' > 0 → B' > 0 → C' > 0 →
    A' + B' + C' = Real.pi →
    3 * a' * Real.cos C' + b' = 0 →
    Real.tan B ≤ Real.tan B' →
  Real.tan B ≤ 3/4 :=
by sorry

end max_tan_B_in_triangle_l624_62416


namespace gardener_mowing_time_l624_62411

theorem gardener_mowing_time (B : ℝ) (together : ℝ) (h1 : B = 5) (h2 : together = 1.875) :
  ∃ A : ℝ, A = 3 ∧ 1 / A + 1 / B = 1 / together :=
by sorry

end gardener_mowing_time_l624_62411


namespace triangle_perimeter_l624_62452

/-- Proves that a triangle with inradius 2.5 cm and area 50 cm² has a perimeter of 40 cm -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 50 → A = r * (p / 2) → p = 40 := by sorry

end triangle_perimeter_l624_62452


namespace power_function_through_fixed_point_l624_62412

-- Define the fixed point
def P : ℝ × ℝ := (4, 2)

-- Define the power function
def f (x : ℝ) : ℝ := x^(1/2)

-- State the theorem
theorem power_function_through_fixed_point :
  f P.1 = P.2 ∧ ∀ x > 0, f x = Real.sqrt x := by sorry

end power_function_through_fixed_point_l624_62412


namespace x_2000_value_l624_62430

def sequence_property (x : ℕ → ℝ) :=
  ∀ n, x (n + 1) + x (n + 2) + x (n + 3) = 20

theorem x_2000_value (x : ℕ → ℝ) 
  (h1 : sequence_property x) 
  (h2 : x 4 = 9) 
  (h3 : x 12 = 7) : 
  x 2000 = 4 := by
sorry

end x_2000_value_l624_62430


namespace tom_speed_l624_62476

theorem tom_speed (karen_speed : ℝ) (karen_delay : ℝ) (win_distance : ℝ) (tom_distance : ℝ) :
  karen_speed = 60 ∧ 
  karen_delay = 4 / 60 ∧ 
  win_distance = 4 ∧ 
  tom_distance = 24 → 
  ∃ (tom_speed : ℝ), tom_speed = 60 :=
by sorry

end tom_speed_l624_62476


namespace subtract_negative_term_l624_62488

theorem subtract_negative_term (a : ℝ) :
  (4 * a^2 - 3 * a + 7) - (-6 * a) = 4 * a^2 + 3 * a + 7 := by
  sorry

end subtract_negative_term_l624_62488


namespace last_to_first_points_l624_62489

/-- Represents a chess tournament with initial and final states -/
structure ChessTournament where
  initial_players : ℕ
  disqualified_players : ℕ
  points_per_win : ℚ
  points_per_draw : ℚ
  points_per_loss : ℚ

/-- Calculates the total number of games in a round-robin tournament -/
def total_games (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating that a player who goes from last to first must have 4 points after disqualification -/
theorem last_to_first_points (t : ChessTournament) 
  (h1 : t.initial_players = 10)
  (h2 : t.disqualified_players = 2)
  (h3 : t.points_per_win = 1)
  (h4 : t.points_per_draw = 1/2)
  (h5 : t.points_per_loss = 0) :
  ∃ (initial_points final_points : ℚ),
    initial_points < (total_games t.initial_players : ℚ) / t.initial_players ∧
    final_points > (total_games (t.initial_players - t.disqualified_players) : ℚ) / (t.initial_players - t.disqualified_players) ∧
    final_points = 4 :=
  sorry

end last_to_first_points_l624_62489


namespace unique_three_digit_number_l624_62436

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  n % 5 = 3 ∧ 
  n % 7 = 4 ∧ 
  n % 4 = 2 := by
sorry

end unique_three_digit_number_l624_62436


namespace mean_median_difference_l624_62482

def class_size : ℕ := 40

def score_distribution : List (ℕ × ℚ) := [
  (60, 15/100),
  (75, 35/100),
  (82, 10/100),
  (88, 20/100),
  (92, 20/100)
]

def mean_score : ℚ :=
  (score_distribution.map (λ (score, percentage) => score * (percentage * class_size))).sum / class_size

def median_score : ℕ := 75

theorem mean_median_difference : 
  ⌊mean_score - median_score⌋ = 4 :=
sorry

end mean_median_difference_l624_62482


namespace find_transmitter_probability_l624_62431

/-- The number of possible government vehicle license plates starting with 79 -/
def total_vehicles : ℕ := 900

/-- The number of vehicles police can inspect per hour -/
def inspection_rate : ℕ := 6

/-- The search time in hours -/
def search_time : ℕ := 3

/-- The probability of finding the transmitter within the given search time -/
theorem find_transmitter_probability :
  (inspection_rate * search_time : ℚ) / total_vehicles = 1 / 50 := by
  sorry

end find_transmitter_probability_l624_62431


namespace irrational_among_given_numbers_l624_62467

theorem irrational_among_given_numbers : 
  (∃ (q : ℚ), (1 : ℝ) / 2 = ↑q) ∧ 
  (∃ (q : ℚ), (1 : ℝ) / 3 = ↑q) ∧ 
  (∃ (q : ℚ), Real.sqrt 4 = ↑q) ∧ 
  (∀ (q : ℚ), Real.sqrt 5 ≠ ↑q) := by
  sorry

end irrational_among_given_numbers_l624_62467


namespace product_mod_thousand_l624_62480

theorem product_mod_thousand : (1234 * 5678) % 1000 = 652 := by
  sorry

end product_mod_thousand_l624_62480


namespace parallelogram_area_l624_62466

/-- The area of a parallelogram with a 150-degree angle and consecutive sides of 10 and 20 units --/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 20) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin θ = 100 * Real.sqrt 3 := by
  sorry

#check parallelogram_area

end parallelogram_area_l624_62466


namespace divisible_by_72_sum_of_digits_l624_62483

theorem divisible_by_72_sum_of_digits (A B : ℕ) : 
  A < 10 → B < 10 → 
  (100000 * A + 44610 + B) % 72 = 0 → 
  A + B = 12 :=
by sorry

end divisible_by_72_sum_of_digits_l624_62483


namespace unique_function_property_l624_62494

def last_digit (n : ℕ) : ℕ := n % 10

def is_constant_one (f : ℕ → ℕ) : Prop :=
  ∀ n, f n = 1

theorem unique_function_property (f : ℕ → ℕ) 
  (h1 : ∀ x y, f (x * y) = f x * f y)
  (h2 : f 30 = 1)
  (h3 : ∀ n, last_digit n = 7 → f n = 1) :
  is_constant_one f :=
sorry

end unique_function_property_l624_62494


namespace circle_tangent_chord_relation_l624_62451

/-- Given a circle O with radius r, prove the relationship between x and y -/
theorem circle_tangent_chord_relation (r : ℝ) (x y : ℝ) : y^2 = x^3 / (2*r - x) :=
  sorry

end circle_tangent_chord_relation_l624_62451


namespace f_properties_l624_62405

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 1 / (2^x - 1)

theorem f_properties (a : ℝ) :
  (∀ x : ℝ, f a x ≠ 0 ↔ x ≠ 0) ∧
  (∀ x : ℝ, f a (-x) = -(f a x) ↔ a = 1/2) ∧
  (a = 1/2 → ∀ x : ℝ, x ≠ 0 → x^3 * f a x > 0) :=
sorry

end f_properties_l624_62405


namespace similar_triangles_problem_l624_62410

/-- Triangle similarity -/
structure SimilarTriangles (G H I J K L : ℝ × ℝ) : Prop where
  similar : True  -- Placeholder for similarity condition

/-- Angle measure in degrees -/
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem similar_triangles_problem 
  (G H I J K L : ℝ × ℝ) 
  (sim : SimilarTriangles G H I J K L)
  (gh_length : dist G H = 8)
  (hi_length : dist H I = 16)
  (kl_length : dist K L = 24)
  (ghi_angle : angle_measure G H I = 30) :
  dist J K = 12 ∧ angle_measure J K L = 30 := by
  sorry

end similar_triangles_problem_l624_62410


namespace complement_A_intersect_B_l624_62484

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2} := by sorry

end complement_A_intersect_B_l624_62484


namespace sin_inequality_l624_62440

theorem sin_inequality (θ : Real) (h : 0 < θ ∧ θ < Real.pi) :
  Real.sin θ + (1/2) * Real.sin (2*θ) + (1/3) * Real.sin (3*θ) > 0 := by
  sorry

end sin_inequality_l624_62440


namespace investment_solution_l624_62401

def investment_problem (amount_A : ℝ) : Prop :=
  let yield_A : ℝ := 0.30
  let yield_B : ℝ := 0.50
  let amount_B : ℝ := 200
  (amount_A * (1 + yield_A)) = (amount_B * (1 + yield_B) + 90)

theorem investment_solution : 
  ∃ (amount_A : ℝ), investment_problem amount_A ∧ amount_A = 300 := by
  sorry

end investment_solution_l624_62401


namespace sams_nickels_l624_62459

/-- Given Sam's initial nickels and his dad's gift of nickels, calculate Sam's total nickels -/
theorem sams_nickels (initial_nickels dad_gift_nickels : ℕ) :
  initial_nickels = 24 → dad_gift_nickels = 39 →
  initial_nickels + dad_gift_nickels = 63 := by sorry

end sams_nickels_l624_62459


namespace arithmetic_calculations_l624_62441

theorem arithmetic_calculations : 
  (1 - (-5) * ((-1)^2) - 4 / ((-1/2)^2) = -11) ∧ 
  ((-2)^3 * (1/8) + (2/3 - 1/2 - 1/4) / (1/12) = -2) := by
  sorry

end arithmetic_calculations_l624_62441


namespace fourth_power_nested_root_l624_62479

theorem fourth_power_nested_root : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := by
  sorry

end fourth_power_nested_root_l624_62479


namespace one_ta_grading_time_l624_62415

/-- The number of initial teaching assistants -/
def N : ℕ := 5

/-- The time it takes N teaching assistants to grade all homework -/
def initial_time : ℕ := 5

/-- The time it takes N+1 teaching assistants to grade all homework -/
def new_time : ℕ := 4

/-- The total work required to grade all homework -/
def total_work : ℕ := N * initial_time

theorem one_ta_grading_time :
  (total_work : ℚ) = 20 :=
by sorry

end one_ta_grading_time_l624_62415


namespace wades_total_spend_l624_62413

/-- Wade's purchase at a rest stop -/
def wades_purchase : ℕ → ℕ → ℕ → ℕ → ℕ := fun num_sandwiches sandwich_price num_drinks drink_price =>
  num_sandwiches * sandwich_price + num_drinks * drink_price

theorem wades_total_spend :
  wades_purchase 3 6 2 4 = 26 := by
  sorry

end wades_total_spend_l624_62413


namespace specific_l_shape_perimeter_l624_62470

/-- Represents an L-shaped region formed by congruent squares -/
structure LShapedRegion where
  squareCount : Nat
  topRowCount : Nat
  bottomRowCount : Nat
  totalArea : ℝ

/-- Calculates the perimeter of an L-shaped region -/
def calculatePerimeter (region : LShapedRegion) : ℝ :=
  sorry

/-- Theorem: The perimeter of the specific L-shaped region is 91 cm -/
theorem specific_l_shape_perimeter :
  let region : LShapedRegion := {
    squareCount := 8,
    topRowCount := 3,
    bottomRowCount := 5,
    totalArea := 392
  }
  calculatePerimeter region = 91 := by
  sorry

end specific_l_shape_perimeter_l624_62470


namespace solve_equation_l624_62464

theorem solve_equation (x : ℝ) :
  (1 / 7 : ℝ) + 7 / x = 15 / x + (1 / 15 : ℝ) → x = 105 := by
  sorry

end solve_equation_l624_62464


namespace lemonade_stand_revenue_l624_62406

theorem lemonade_stand_revenue 
  (total_cups : ℝ) 
  (small_cup_price : ℝ) 
  (h1 : small_cup_price > 0) : 
  let small_cups := (3 / 5) * total_cups
  let large_cups := (2 / 5) * total_cups
  let large_cup_price := (1 / 6) * small_cup_price
  let small_revenue := small_cups * small_cup_price
  let large_revenue := large_cups * large_cup_price
  let total_revenue := small_revenue + large_revenue
  (large_revenue / total_revenue) = (1 / 10) := by
sorry

end lemonade_stand_revenue_l624_62406


namespace prime_with_integer_roots_range_l624_62486

theorem prime_with_integer_roots_range (p : ℕ) : 
  Nat.Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 500*p = 0 ∧ y^2 + p*y - 500*p = 0) → 
  1 < p ∧ p ≤ 10 := by
  sorry

end prime_with_integer_roots_range_l624_62486


namespace expression_evaluation_l624_62493

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  5 * x^(y + 1) + 6 * y^(x + 1) = 2751 := by
  sorry

end expression_evaluation_l624_62493


namespace square_plus_one_l624_62485

theorem square_plus_one (a : ℝ) (h : a^2 + 2*a - 2 = 0) : (a + 1)^2 = 3 := by
  sorry

end square_plus_one_l624_62485


namespace octagon_diagonals_l624_62443

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: An octagon has 20 internal diagonals -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l624_62443


namespace largest_prime_factor_of_10001_l624_62445

theorem largest_prime_factor_of_10001 : ∃ p : ℕ, 
  p.Prime ∧ p ∣ 10001 ∧ ∀ q : ℕ, q.Prime → q ∣ 10001 → q ≤ p :=
by sorry

end largest_prime_factor_of_10001_l624_62445


namespace andrew_kept_490_stickers_l624_62447

/-- The number of stickers Andrew bought -/
def total_stickers : ℕ := 1500

/-- The number of stickers Daniel received -/
def daniel_stickers : ℕ := 250

/-- The number of stickers Fred received -/
def fred_stickers : ℕ := daniel_stickers + 120

/-- The number of stickers Emily received -/
def emily_stickers : ℕ := (daniel_stickers + fred_stickers) / 2

/-- The number of stickers Gina received -/
def gina_stickers : ℕ := 80

/-- The number of stickers Andrew kept -/
def andrew_stickers : ℕ := total_stickers - (daniel_stickers + fred_stickers + emily_stickers + gina_stickers)

theorem andrew_kept_490_stickers : andrew_stickers = 490 := by
  sorry

end andrew_kept_490_stickers_l624_62447


namespace stone_99_is_11_l624_62472

/-- Represents the counting pattern for 12 stones -/
def stone_count (n : ℕ) : ℕ :=
  let cycle := 22  -- The pattern repeats every 22 counts
  let within_cycle := n % cycle
  if within_cycle ≤ 12
  then within_cycle
  else 13 - (within_cycle - 11)

/-- The theorem stating that the 99th count corresponds to the 11th stone -/
theorem stone_99_is_11 : stone_count 99 = 11 := by
  sorry

#eval stone_count 99  -- This should output 11

end stone_99_is_11_l624_62472


namespace ninth_term_of_sequence_l624_62404

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r ^ (n - 1)

theorem ninth_term_of_sequence :
  let a₁ : ℚ := 5
  let r : ℚ := 3/2
  geometric_sequence a₁ r 9 = 32805/256 := by
sorry

end ninth_term_of_sequence_l624_62404


namespace anderson_trousers_count_l624_62492

theorem anderson_trousers_count :
  let total_clothing : ℕ := 934
  let shirts : ℕ := 589
  let trousers : ℕ := total_clothing - shirts
  trousers = 345 := by sorry

end anderson_trousers_count_l624_62492


namespace candy_cost_l624_62458

theorem candy_cost (packs : ℕ) (paid : ℕ) (change : ℕ) (h1 : packs = 3) (h2 : paid = 20) (h3 : change = 11) :
  (paid - change) / packs = 3 := by
  sorry

end candy_cost_l624_62458


namespace amy_work_hours_l624_62491

theorem amy_work_hours (hourly_wage : ℝ) (tips : ℝ) (total_earnings : ℝ) : 
  hourly_wage = 2 → tips = 9 → total_earnings = 23 → 
  ∃ h : ℝ, h * hourly_wage + tips = total_earnings ∧ h = 7 := by
sorry

end amy_work_hours_l624_62491


namespace zilla_savings_calculation_l624_62423

def monthly_savings (total_earnings : ℝ) (rent_amount : ℝ) : ℝ :=
  let after_tax := total_earnings * 0.9
  let rent_percent := rent_amount / after_tax
  let groceries := after_tax * 0.3
  let entertainment := after_tax * 0.2
  let transportation := after_tax * 0.12
  let total_expenses := rent_amount + groceries + entertainment + transportation
  let remaining := after_tax - total_expenses
  remaining * 0.15

theorem zilla_savings_calculation (total_earnings : ℝ) (h1 : total_earnings > 0) 
  (h2 : monthly_savings total_earnings 133 = 77.52) : 
  ∃ (e : ℝ), e = total_earnings ∧ monthly_savings e 133 = 77.52 :=
by
  sorry

#eval monthly_savings 1900 133

end zilla_savings_calculation_l624_62423


namespace video_game_spending_is_correct_l624_62498

def total_allowance : ℚ := 50

def movie_fraction : ℚ := 1/4
def burger_fraction : ℚ := 1/5
def ice_cream_fraction : ℚ := 1/10
def music_fraction : ℚ := 2/5

def video_game_spending : ℚ := total_allowance - (movie_fraction * total_allowance + burger_fraction * total_allowance + ice_cream_fraction * total_allowance + music_fraction * total_allowance)

theorem video_game_spending_is_correct : video_game_spending = 5/2 := by
  sorry

end video_game_spending_is_correct_l624_62498


namespace cosine_identity_l624_62439

theorem cosine_identity (n : Real) : 
  (Real.cos (30 * π / 180 - n * π / 180)) / (Real.cos (n * π / 180)) = 
  (1 / 2) * (Real.sqrt 3 + Real.tan (n * π / 180)) := by
  sorry

end cosine_identity_l624_62439


namespace ellipse_eccentricity_l624_62469

/-- An ellipse passing through (2,3) with foci at (-2,0) and (2,0) has eccentricity 1/2 -/
theorem ellipse_eccentricity : ∀ (e : ℝ), 
  (∃ (a b : ℝ), 
    (2/a)^2 + (3/b)^2 = 1 ∧  -- ellipse passes through (2,3)
    a > b ∧ b > 0 ∧         -- standard form constraints
    4 = a^2 - b^2) →        -- distance between foci is 4
  e = 1/2 := by
sorry

end ellipse_eccentricity_l624_62469


namespace crofton_orchestra_max_members_l624_62426

theorem crofton_orchestra_max_members :
  ∀ n : ℕ,
  (25 * n < 1000) →
  (25 * n % 24 = 5) →
  (∀ m : ℕ, (25 * m < 1000) ∧ (25 * m % 24 = 5) → m ≤ n) →
  25 * n = 725 :=
by
  sorry

end crofton_orchestra_max_members_l624_62426


namespace book_weight_l624_62403

theorem book_weight (total_weight : ℝ) (num_books : ℕ) (h1 : total_weight = 42) (h2 : num_books = 14) :
  total_weight / num_books = 3 := by
sorry

end book_weight_l624_62403


namespace tomatoes_left_l624_62487

theorem tomatoes_left (initial : ℕ) (picked_yesterday : ℕ) (picked_today : ℕ) 
  (h1 : initial = 171)
  (h2 : picked_yesterday = 134)
  (h3 : picked_today = 30) : 
  initial - picked_yesterday - picked_today = 7 := by
  sorry

end tomatoes_left_l624_62487


namespace stream_speed_l624_62418

/-- Proves that the speed of the stream is 8 km/hr, given the conditions of the boat's travel --/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 10 →
  downstream_distance = 54 →
  downstream_time = 3 →
  (boat_speed + (downstream_distance / downstream_time - boat_speed)) = 18 :=
by
  sorry

end stream_speed_l624_62418


namespace prom_attendance_l624_62437

/-- The number of students who attended the prom on their own -/
def solo_students : ℕ := 3

/-- The number of couples who came to the prom -/
def couples : ℕ := 60

/-- The total number of students who attended the prom -/
def total_students : ℕ := solo_students + 2 * couples

/-- Theorem: The total number of students who attended the prom is 123 -/
theorem prom_attendance : total_students = 123 := by
  sorry

end prom_attendance_l624_62437


namespace stella_toilet_paper_l624_62409

/-- The number of packs of toilet paper Stella needs to buy for 4 weeks -/
def toilet_paper_packs (bathrooms : ℕ) (rolls_per_bathroom_per_day : ℕ) 
  (days_per_week : ℕ) (rolls_per_pack : ℕ) (weeks : ℕ) : ℕ :=
  (bathrooms * rolls_per_bathroom_per_day * days_per_week * weeks) / rolls_per_pack

/-- Stella's toilet paper restocking problem -/
theorem stella_toilet_paper : 
  toilet_paper_packs 6 1 7 12 4 = 14 := by
  sorry

end stella_toilet_paper_l624_62409


namespace square_roots_calculation_l624_62497

theorem square_roots_calculation : (Real.sqrt 3 + Real.sqrt 2)^2 * (5 - 2 * Real.sqrt 6) = 1 := by
  sorry

end square_roots_calculation_l624_62497


namespace triangle_side_range_l624_62495

theorem triangle_side_range (A B C : Real) (AB AC BC : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Given equation
  (Real.sqrt 3 * Real.sin B - Real.cos B) * (Real.sqrt 3 * Real.sin C - Real.cos C) = 4 * Real.cos B * Real.cos C →
  -- Sum of two sides
  AB + AC = 4 →
  -- Triangle inequality
  AB > 0 ∧ AC > 0 ∧ BC > 0 →
  -- BC satisfies the triangle inequality
  BC < AB + AC ∧ AB < BC + AC ∧ AC < AB + BC →
  -- Conclusion: Range of BC
  2 ≤ BC ∧ BC < 4 := by
sorry

end triangle_side_range_l624_62495


namespace a_squared_b_plus_ab_squared_l624_62417

theorem a_squared_b_plus_ab_squared (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a * b = 6) : 
  a^2 * b + a * b^2 = 30 := by
sorry

end a_squared_b_plus_ab_squared_l624_62417


namespace francis_muffins_l624_62473

/-- The cost of breakfast for Francis and Kiera -/
def breakfast_cost : ℕ → ℕ
| m => 2 * m + 2 * 3 + 2 * 2 + 3

/-- The theorem stating that Francis had 2 muffins -/
theorem francis_muffins : 
  ∃ m : ℕ, breakfast_cost m = 17 ∧ m = 2 :=
by sorry

end francis_muffins_l624_62473


namespace only_statement3_correct_l624_62400

-- Define the propositions
variable (p q : Prop)

-- Define the four statements
def statement1 : Prop := ∀ (p q : Prop), (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)
def statement2 : Prop := ∀ (p q : Prop), (¬(p ∧ q) → p ∨ q) ∧ ¬(p ∨ q → ¬(p ∧ q))
def statement3 : Prop := ∀ (p q : Prop), (p ∨ q → ¬(¬p)) ∧ ¬(¬(¬p) → p ∨ q)
def statement4 : Prop := ∀ (p q : Prop), (¬p → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → ¬p)

-- Theorem stating that only the third statement is correct
theorem only_statement3_correct :
  ¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4 :=
sorry

end only_statement3_correct_l624_62400


namespace no_perfect_squares_l624_62432

/-- Represents a 100-digit number with a repeating pattern -/
def RepeatingNumber (pattern : ℕ) : ℕ :=
  -- Implementation details omitted for simplicity
  sorry

/-- N₁ is a 100-digit number consisting of all 3's -/
def N1 : ℕ := RepeatingNumber 3

/-- N₂ is a 100-digit number consisting of all 6's -/
def N2 : ℕ := RepeatingNumber 6

/-- N₃ is a 100-digit number with repeating pattern 15 -/
def N3 : ℕ := RepeatingNumber 15

/-- N₄ is a 100-digit number with repeating pattern 21 -/
def N4 : ℕ := RepeatingNumber 21

/-- N₅ is a 100-digit number with repeating pattern 27 -/
def N5 : ℕ := RepeatingNumber 27

/-- A number is a perfect square if there exists an integer whose square equals the number -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_perfect_squares : ¬(is_perfect_square N1 ∨ is_perfect_square N2 ∨ 
                               is_perfect_square N3 ∨ is_perfect_square N4 ∨ 
                               is_perfect_square N5) := by
  sorry

end no_perfect_squares_l624_62432


namespace sector_area_90_degrees_l624_62450

/-- The area of a sector with radius 2 and central angle 90° is π. -/
theorem sector_area_90_degrees : 
  let r : ℝ := 2
  let angle : ℝ := 90
  let sector_area := (angle / 360) * π * r^2
  sector_area = π := by sorry

end sector_area_90_degrees_l624_62450


namespace smallest_valid_integers_difference_l624_62454

def is_valid (n : ℕ) : Prop :=
  n > 2 ∧ ∀ k : ℕ, 2 ≤ k ∧ k ≤ 12 → n % k = 2

theorem smallest_valid_integers_difference :
  ∃ n m : ℕ, is_valid n ∧ is_valid m ∧
  (∀ x : ℕ, is_valid x → n ≤ x) ∧
  (∀ x : ℕ, is_valid x ∧ x ≠ n → m ≤ x) ∧
  m - n = 13860 :=
sorry

end smallest_valid_integers_difference_l624_62454


namespace max_runs_in_match_l624_62463

/-- Represents the maximum number of runs that can be scored in a single delivery -/
def max_runs_per_delivery : ℕ := 6

/-- Represents the number of deliveries in an over -/
def deliveries_per_over : ℕ := 6

/-- Represents the total number of overs in the match -/
def total_overs : ℕ := 35

/-- Represents the maximum number of consecutive boundaries allowed in an over -/
def max_consecutive_boundaries : ℕ := 3

/-- Calculates the maximum runs that can be scored in a single over -/
def max_runs_per_over : ℕ :=
  max_consecutive_boundaries * max_runs_per_delivery + 
  (deliveries_per_over - max_consecutive_boundaries)

/-- Theorem: The maximum number of runs a batsman can score in the given match is 735 -/
theorem max_runs_in_match : 
  total_overs * max_runs_per_over = 735 := by
  sorry

end max_runs_in_match_l624_62463


namespace min_workers_for_profit_l624_62429

/-- Represents the problem of finding the minimum number of workers needed for profit --/
theorem min_workers_for_profit (
  maintenance_cost : ℝ)
  (worker_hourly_wage : ℝ)
  (widgets_per_hour : ℝ)
  (widget_price : ℝ)
  (work_hours : ℝ)
  (h1 : maintenance_cost = 600)
  (h2 : worker_hourly_wage = 20)
  (h3 : widgets_per_hour = 6)
  (h4 : widget_price = 3.5)
  (h5 : work_hours = 8)
  : ∃ n : ℕ, n = 76 ∧ ∀ m : ℕ, m < n → maintenance_cost + m * worker_hourly_wage * work_hours ≥ m * widgets_per_hour * widget_price * work_hours :=
sorry

end min_workers_for_profit_l624_62429


namespace car_sales_profit_loss_percentage_l624_62420

/-- Calculates the overall profit or loss percentage for two car sales --/
theorem car_sales_profit_loss_percentage 
  (selling_price : ℝ) 
  (gain_percentage : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : selling_price > 0) 
  (h2 : gain_percentage > 0) 
  (h3 : loss_percentage > 0) 
  (h4 : gain_percentage = loss_percentage) : 
  ∃ (loss_percent : ℝ), 
    loss_percent > 0 ∧ 
    loss_percent < gain_percentage ∧
    loss_percent = (2 * selling_price - (selling_price / (1 + gain_percentage / 100) + selling_price / (1 - loss_percentage / 100))) / 
                   (selling_price / (1 + gain_percentage / 100) + selling_price / (1 - loss_percentage / 100)) * 100 := by
  sorry

end car_sales_profit_loss_percentage_l624_62420


namespace square_area_decrease_l624_62460

theorem square_area_decrease (initial_area : ℝ) (side_decrease_percent : ℝ) 
  (h1 : initial_area = 50)
  (h2 : side_decrease_percent = 25) : 
  let new_side := (1 - side_decrease_percent / 100) * Real.sqrt initial_area
  let new_area := new_side * Real.sqrt initial_area
  let percent_decrease := (initial_area - new_area) / initial_area * 100
  percent_decrease = 43.75 := by
sorry

end square_area_decrease_l624_62460


namespace transport_percentage_l624_62419

/-- Calculate the percentage of income spent on public transport -/
theorem transport_percentage (income : ℝ) (remaining : ℝ) 
  (h1 : income = 2000)
  (h2 : remaining = 1900) :
  (income - remaining) / income * 100 = 5 := by
sorry

end transport_percentage_l624_62419


namespace euro_equation_solution_l624_62424

def euro (x y : ℝ) : ℝ := 2 * x * y

theorem euro_equation_solution (x : ℝ) : 
  euro 9 (euro 4 x) = 720 → x = 5 := by
  sorry

end euro_equation_solution_l624_62424


namespace smallest_n_is_eight_l624_62446

/-- A geometric sequence (a_n) with given conditions -/
def geometric_sequence (x : ℝ) (a : ℕ → ℝ) : Prop :=
  x > 0 ∧
  a 1 = Real.exp x ∧
  a 2 = x ∧
  a 3 = Real.log x ∧
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = r * a n

/-- The smallest n for which a_n = 2x is 8 -/
theorem smallest_n_is_eight (x : ℝ) (a : ℕ → ℝ) 
  (h : geometric_sequence x a) : 
  (∃ n : ℕ, n ≥ 1 ∧ a n = 2 * x) ∧ 
  (∀ m : ℕ, m ≥ 1 ∧ m < 8 → a m ≠ 2 * x) ∧ 
  a 8 = 2 * x := by
  sorry

end smallest_n_is_eight_l624_62446


namespace jake_arrives_later_l624_62481

/-- Represents the building with elevators and stairs --/
structure Building where
  floors : ℕ
  steps_per_floor : ℕ
  elevator_b_time : ℕ

/-- Represents a person descending the building --/
structure Person where
  steps_per_second : ℕ

def time_to_descend (b : Building) (p : Person) : ℕ :=
  let total_steps := b.steps_per_floor * (b.floors - 1)
  (total_steps + p.steps_per_second - 1) / p.steps_per_second

theorem jake_arrives_later (b : Building) (jake : Person) :
  b.floors = 12 →
  b.steps_per_floor = 25 →
  b.elevator_b_time = 90 →
  jake.steps_per_second = 3 →
  time_to_descend b jake - b.elevator_b_time = 2 := by
  sorry

#eval time_to_descend { floors := 12, steps_per_floor := 25, elevator_b_time := 90 } { steps_per_second := 3 }

end jake_arrives_later_l624_62481


namespace sum_of_inscribed_angles_is_180_l624_62402

/-- A regular pentagon inscribed in a circle -/
structure RegularPentagonInCircle where
  /-- The circle in which the pentagon is inscribed -/
  circle : Real
  /-- The regular pentagon inscribed in the circle -/
  pentagon : Real
  /-- The sides of the pentagon divide the circle into five equal arcs -/
  equal_arcs : pentagon = 5

/-- The sum of angles inscribed in the five arcs cut off by the sides of a regular pentagon inscribed in a circle -/
def sum_of_inscribed_angles (p : RegularPentagonInCircle) : Real :=
  sorry

/-- Theorem: The sum of angles inscribed in the five arcs cut off by the sides of a regular pentagon inscribed in a circle is 180° -/
theorem sum_of_inscribed_angles_is_180 (p : RegularPentagonInCircle) :
  sum_of_inscribed_angles p = 180 := by
  sorry

end sum_of_inscribed_angles_is_180_l624_62402


namespace sarah_copies_pages_l624_62461

theorem sarah_copies_pages (people meeting_size copies_per_person contract_pages : ℕ) 
  (h1 : meeting_size = 15)
  (h2 : copies_per_person = 5)
  (h3 : contract_pages = 35) :
  people = meeting_size * copies_per_person * contract_pages :=
by sorry

end sarah_copies_pages_l624_62461


namespace tower_heights_count_l624_62457

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : Nat
  width : Nat
  height : Nat

/-- Represents the possible orientations of a brick -/
inductive BrickOrientation
  | Length
  | Width
  | Height

/-- Calculates the number of different tower heights achievable -/
def calculateTowerHeights (brickDimensions : BrickDimensions) (totalBricks : Nat) : Nat :=
  sorry

/-- Theorem stating the number of different tower heights achievable -/
theorem tower_heights_count (brickDimensions : BrickDimensions) 
  (h1 : brickDimensions.length = 3)
  (h2 : brickDimensions.width = 12)
  (h3 : brickDimensions.height = 20)
  (h4 : totalBricks = 100) :
  calculateTowerHeights brickDimensions totalBricks = 187 := by
    sorry

end tower_heights_count_l624_62457


namespace hyperbola_equation_l624_62442

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the eccentricity of the hyperbola
def hyperbola_eccentricity : ℝ := 2

-- Define the standard form of a hyperbola
def is_standard_hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Theorem statement
theorem hyperbola_equation : 
  ∃ (a b : ℝ), a^2 = 4 ∧ b^2 = 12 ∧ 
  (∀ (x y : ℝ), is_standard_hyperbola a b x y) ∧
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ 
   c = hyperbola_eccentricity * a ∧
   c^2 = 25 - 9) := by sorry

end hyperbola_equation_l624_62442


namespace ratio_y_to_x_l624_62456

theorem ratio_y_to_x (x y z : ℝ) : 
  (0.6 * (x - y) = 0.4 * (x + y) + 0.3 * (x - 3 * z)) →
  (∃ k : ℤ, z = k * y) →
  (z = 7 * y) →
  (y = 5 * x / 7) →
  y / x = 5 / 7 := by
  sorry

end ratio_y_to_x_l624_62456


namespace tamara_height_is_62_l624_62427

/-- Calculates Tamara's height given Kim's height and the age difference effect -/
def tamaraHeight (kimHeight : ℝ) (ageDifference : ℕ) : ℝ :=
  2 * kimHeight - 4

/-- Calculates Gavin's height given Kim's height -/
def gavinHeight (kimHeight : ℝ) : ℝ :=
  2 * kimHeight + 6

/-- The combined height of all three people -/
def combinedHeight : ℝ := 200

/-- The age difference between Tamara and Kim -/
def ageDifference : ℕ := 5

/-- The change in height ratio per year of age difference -/
def ratioChangePerYear : ℝ := 0.2

theorem tamara_height_is_62 :
  ∃ (kimHeight : ℝ),
    tamaraHeight kimHeight ageDifference +
    kimHeight +
    gavinHeight kimHeight = combinedHeight ∧
    tamaraHeight kimHeight ageDifference = 62 := by
  sorry

end tamara_height_is_62_l624_62427


namespace sum_x_y_equals_thirteen_l624_62438

theorem sum_x_y_equals_thirteen 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 16)
  (h2 : (16:ℝ)^(x+y) / (4:ℝ)^(7*y) = 1024)
  : x + y = 13 := by
  sorry

end sum_x_y_equals_thirteen_l624_62438


namespace correct_delivery_probability_l624_62444

def num_packages : ℕ := 5
def num_correct : ℕ := 3

theorem correct_delivery_probability :
  (num_packages.choose num_correct * (num_correct.factorial * (num_packages - num_correct).factorial)) /
  num_packages.factorial = 1 / 12 := by
sorry

end correct_delivery_probability_l624_62444


namespace area_of_S3_l624_62474

/-- Given a square S1 with area 16, S2 is constructed by connecting the midpoints of S1's sides,
    and S3 is constructed by connecting the midpoints of S2's sides. -/
def nested_squares (S1 S2 S3 : Real) : Prop :=
  S1 = 16 ∧ S2 = S1 / 2 ∧ S3 = S2 / 2

theorem area_of_S3 (S1 S2 S3 : Real) (h : nested_squares S1 S2 S3) : S3 = 4 := by
  sorry

end area_of_S3_l624_62474


namespace triangle_abc_properties_l624_62449

theorem triangle_abc_properties (a b : ℝ) (cosB : ℝ) (S : ℝ) :
  a = 5 →
  b = 6 →
  cosB = -4/5 →
  S = 15 * Real.sqrt 7 / 4 →
  ∃ (A R c : ℝ),
    (A = π/6 ∧ R = 5) ∧
    (c = 4 ∨ c = Real.sqrt 106) :=
by sorry

end triangle_abc_properties_l624_62449


namespace janice_stairs_l624_62475

/-- The number of flights of stairs to reach Janice's office -/
def flights_per_staircase : ℕ := 3

/-- The number of times Janice goes up the stairs in a day -/
def times_up : ℕ := 5

/-- The number of times Janice goes down the stairs in a day -/
def times_down : ℕ := 3

/-- The total number of flights Janice walks up in a day -/
def flights_up : ℕ := flights_per_staircase * times_up

/-- The total number of flights Janice walks down in a day -/
def flights_down : ℕ := flights_per_staircase * times_down

/-- The total number of flights Janice walks in a day -/
def total_flights : ℕ := flights_up + flights_down

theorem janice_stairs : total_flights = 24 := by
  sorry

end janice_stairs_l624_62475


namespace factor_polynomial_l624_62490

theorem factor_polynomial (x : ℝ) : 80 * x^5 - 250 * x^9 = -10 * x^5 * (25 * x^4 - 8) := by
  sorry

end factor_polynomial_l624_62490


namespace value_of_expression_l624_62414

theorem value_of_expression (a b : ℝ) (h : 2 * a - b = -1) : 
  4 * a - 2 * b + 1 = 1 := by
sorry

end value_of_expression_l624_62414


namespace angle_between_perpendicular_lines_in_dihedral_l624_62468

-- Define the dihedral angle
def dihedral_angle (α l β : Line3) : ℝ := sorry

-- Define perpendicularity between a line and a plane
def perpendicular (m : Line3) (α : Plane3) : Prop := sorry

-- Define the angle between two lines
def angle_between_lines (m n : Line3) : ℝ := sorry

-- Main theorem
theorem angle_between_perpendicular_lines_in_dihedral 
  (α l β : Line3) (m n : Line3) :
  dihedral_angle α l β = 60 →
  m ≠ n →
  perpendicular m α →
  perpendicular n β →
  angle_between_lines m n = 60 :=
sorry

end angle_between_perpendicular_lines_in_dihedral_l624_62468


namespace charlie_snowball_count_l624_62455

theorem charlie_snowball_count (lucy_snowballs : ℕ) (charlie_extra : ℕ) 
  (h1 : lucy_snowballs = 19)
  (h2 : charlie_extra = 31) : 
  lucy_snowballs + charlie_extra = 50 := by
  sorry

end charlie_snowball_count_l624_62455


namespace athletes_arrangement_l624_62448

/-- The number of ways to arrange athletes from three teams in a row -/
def arrange_athletes (team_a : ℕ) (team_b : ℕ) (team_c : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial team_a) * (Nat.factorial team_b) * (Nat.factorial team_c)

/-- Theorem: The number of ways to arrange 10 athletes from 3 teams (with 4, 3, and 3 athletes respectively) in a row, where athletes from the same team must sit together, is 5184 -/
theorem athletes_arrangement :
  arrange_athletes 4 3 3 = 5184 :=
by sorry

end athletes_arrangement_l624_62448


namespace left_of_kolya_l624_62477

/-- The number of people in a line-up -/
def total_people : ℕ := 29

/-- The number of people to the right of Kolya -/
def right_of_kolya : ℕ := 12

/-- The number of people to the left of Sasha -/
def left_of_sasha : ℕ := 20

/-- The number of people to the right of Sasha -/
def right_of_sasha : ℕ := 8

/-- Theorem: The number of people to the left of Kolya is 16 -/
theorem left_of_kolya : total_people - right_of_kolya - 1 = 16 := by
  sorry

end left_of_kolya_l624_62477


namespace abs_neg_reciprocal_2023_l624_62465

theorem abs_neg_reciprocal_2023 : |-1 / 2023| = 1 / 2023 := by
  sorry

end abs_neg_reciprocal_2023_l624_62465


namespace smallest_modulus_z_l624_62422

theorem smallest_modulus_z (z : ℂ) (h : 3 * Complex.abs (z - 8) + 2 * Complex.abs (z - Complex.I * 7) = 26) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ 3 * Complex.abs (w - 8) + 2 * Complex.abs (w - Complex.I * 7) = 26 ∧ Complex.abs w = 7 :=
sorry

end smallest_modulus_z_l624_62422


namespace absolute_value_square_sum_zero_l624_62453

theorem absolute_value_square_sum_zero (x y : ℝ) :
  |x + 2| + (y - 1)^2 = 0 → x = -2 ∧ y = 1 := by
  sorry

end absolute_value_square_sum_zero_l624_62453


namespace minimize_sum_distances_l624_62435

/-- The point that minimizes the sum of distances from two fixed points on a line --/
theorem minimize_sum_distances (A B C : ℝ × ℝ) : 
  A = (3, 6) → 
  B = (6, 2) → 
  C.2 = 0 → 
  (∀ k : ℝ, C = (k, 0) → 
    Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) + 
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) ≥ 
    Real.sqrt ((6.75 - A.1)^2 + (0 - A.2)^2) + 
    Real.sqrt ((6.75 - B.1)^2 + (0 - B.2)^2)) :=
by sorry

end minimize_sum_distances_l624_62435


namespace distance_to_concert_l624_62408

/-- The distance to a concert given the distance driven before and after a gas stop -/
theorem distance_to_concert (distance_before_gas : ℕ) (distance_after_gas : ℕ) :
  distance_before_gas = 32 →
  distance_after_gas = 46 →
  distance_before_gas + distance_after_gas = 78 := by
  sorry


end distance_to_concert_l624_62408


namespace sum_of_digits_greatest_prime_divisor_32767_l624_62471

def greatest_prime_divisor (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_greatest_prime_divisor_32767 :
  sum_of_digits (greatest_prime_divisor 32767) = 7 := by sorry

end sum_of_digits_greatest_prime_divisor_32767_l624_62471


namespace intersection_chord_length_l624_62433

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := x^2 = 8*y
def line (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points
def intersection_points (M N : ℝ × ℝ) : Prop :=
  parabola M.1 M.2 ∧ line M.1 M.2 ∧
  parabola N.1 N.2 ∧ line N.1 N.2 ∧
  M ≠ N

-- Theorem statement
theorem intersection_chord_length :
  ∀ M N : ℝ × ℝ, intersection_points M N →
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 16 :=
sorry

end intersection_chord_length_l624_62433


namespace jack_evening_emails_l624_62425

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 9

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 10

/-- The difference between morning and evening emails -/
def morning_evening_difference : ℕ := 2

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := morning_emails - morning_evening_difference

theorem jack_evening_emails :
  evening_emails = 7 :=
sorry

end jack_evening_emails_l624_62425


namespace parallelogram_base_l624_62407

/-- The base of a parallelogram given its area and height -/
theorem parallelogram_base (area height base : ℝ) (h1 : area = 648) (h2 : height = 18) 
    (h3 : area = base * height) : base = 36 := by
  sorry

end parallelogram_base_l624_62407
