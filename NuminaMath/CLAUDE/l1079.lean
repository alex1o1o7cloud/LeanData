import Mathlib

namespace NUMINAMATH_CALUDE_fraction_equality_l1079_107943

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 4 * y) = 3) : 
  (2 * x + 4 * y) / (4 * x - 2 * y) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1079_107943


namespace NUMINAMATH_CALUDE_divisibility_implication_l1079_107926

theorem divisibility_implication (m n : ℤ) : 
  (11 ∣ (5 * m + 3 * n)) → (11 ∣ (9 * m + n)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l1079_107926


namespace NUMINAMATH_CALUDE_ride_time_is_36_seconds_l1079_107934

/-- Represents the escalator problem with given conditions -/
structure EscalatorProblem where
  theo_walk_time_non_operating : ℝ
  theo_walk_time_operating : ℝ
  escalator_efficiency : ℝ
  theo_walk_time_non_operating_eq : theo_walk_time_non_operating = 80
  theo_walk_time_operating_eq : theo_walk_time_operating = 30
  escalator_efficiency_eq : escalator_efficiency = 0.75

/-- Calculates the time it takes Theo to ride down the operating escalator while standing still -/
def ride_time (problem : EscalatorProblem) : ℝ :=
  problem.theo_walk_time_non_operating * problem.escalator_efficiency

/-- Theorem stating that the ride time for Theo is 36 seconds -/
theorem ride_time_is_36_seconds (problem : EscalatorProblem) :
  ride_time problem = 36 := by
  sorry

#eval ride_time { theo_walk_time_non_operating := 80,
                  theo_walk_time_operating := 30,
                  escalator_efficiency := 0.75,
                  theo_walk_time_non_operating_eq := rfl,
                  theo_walk_time_operating_eq := rfl,
                  escalator_efficiency_eq := rfl }

end NUMINAMATH_CALUDE_ride_time_is_36_seconds_l1079_107934


namespace NUMINAMATH_CALUDE_brandy_trail_mix_peanuts_l1079_107988

/-- The weight of peanuts in Brandy's trail mix -/
def weight_of_peanuts (total_weight chocolate_weight raisin_weight : ℚ) : ℚ :=
  total_weight - (chocolate_weight + raisin_weight)

/-- Theorem stating that the weight of peanuts in Brandy's trail mix is correct -/
theorem brandy_trail_mix_peanuts :
  weight_of_peanuts 0.4166666666666667 0.16666666666666666 0.08333333333333333 = 0.1666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_brandy_trail_mix_peanuts_l1079_107988


namespace NUMINAMATH_CALUDE_last_three_digits_of_3_to_1000_l1079_107998

theorem last_three_digits_of_3_to_1000 (h : 3^200 ≡ 1 [ZMOD 500]) :
  3^1000 ≡ 1 [ZMOD 1000] :=
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_3_to_1000_l1079_107998


namespace NUMINAMATH_CALUDE_line_circle_intersections_l1079_107997

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 4 * x + 9 * y = 7

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the number of intersection points
def num_intersections : ℕ := 2

-- Theorem statement
theorem line_circle_intersections :
  ∃ (p q : ℝ × ℝ), 
    p ≠ q ∧ 
    line_eq p.1 p.2 ∧ circle_eq p.1 p.2 ∧
    line_eq q.1 q.2 ∧ circle_eq q.1 q.2 ∧
    (∀ (r : ℝ × ℝ), line_eq r.1 r.2 ∧ circle_eq r.1 r.2 → r = p ∨ r = q) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersections_l1079_107997


namespace NUMINAMATH_CALUDE_hands_closest_and_farthest_l1079_107995

/-- Represents a time between 6:30 and 6:35 -/
inductive ClockTime
  | t630
  | t631
  | t632
  | t633
  | t634
  | t635

/-- Calculates the angle between hour and minute hands for a given time -/
def angleBetweenHands (t : ClockTime) : ℝ :=
  match t with
  | ClockTime.t630 => 15
  | ClockTime.t631 => 9.5
  | ClockTime.t632 => 4
  | ClockTime.t633 => 1.5
  | ClockTime.t634 => 7
  | ClockTime.t635 => 12.5

theorem hands_closest_and_farthest :
  (∀ t : ClockTime, angleBetweenHands ClockTime.t633 ≤ angleBetweenHands t) ∧
  (∀ t : ClockTime, angleBetweenHands t ≤ angleBetweenHands ClockTime.t630) :=
by sorry


end NUMINAMATH_CALUDE_hands_closest_and_farthest_l1079_107995


namespace NUMINAMATH_CALUDE_largest_circle_tangent_to_line_l1079_107947

/-- The largest circle with center (0,2) that is tangent to the line mx - y - 3m - 1 = 0 -/
theorem largest_circle_tangent_to_line (m : ℝ) :
  ∃! (r : ℝ), r > 0 ∧
    (∀ (x y : ℝ), x^2 + (y - 2)^2 = r^2 →
      ∃ (x₀ y₀ : ℝ), x₀^2 + (y₀ - 2)^2 = r^2 ∧
        m * x₀ - y₀ - 3 * m - 1 = 0) ∧
    (∀ (r' : ℝ), r' > r →
      ¬∃ (x y : ℝ), x^2 + (y - 2)^2 = r'^2 ∧
        m * x - y - 3 * m - 1 = 0) ∧
    r^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_largest_circle_tangent_to_line_l1079_107947


namespace NUMINAMATH_CALUDE_envelope_counting_time_l1079_107968

/-- Represents the time in seconds to count a given number of envelopes -/
def count_time (envelopes : ℕ) : ℕ :=
  10 * ((100 - envelopes) / 10)

theorem envelope_counting_time :
  (count_time 60 = 40) ∧ (count_time 90 = 10) :=
sorry

end NUMINAMATH_CALUDE_envelope_counting_time_l1079_107968


namespace NUMINAMATH_CALUDE_prob_red_ball_l1079_107921

/-- The probability of drawing a red ball from a bag containing 1 red ball and 2 yellow balls is 1/3. -/
theorem prob_red_ball (num_red : ℕ) (num_yellow : ℕ) (h1 : num_red = 1) (h2 : num_yellow = 2) :
  (num_red : ℚ) / (num_red + num_yellow) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_ball_l1079_107921


namespace NUMINAMATH_CALUDE_smallest_n_for_factorization_l1079_107932

theorem smallest_n_for_factorization : 
  ∀ n : ℤ, 
  (∃ A B : ℤ, ∀ x : ℝ, 3 * x^2 + n * x + 72 = (3 * x + A) * (x + B)) → 
  n ≥ 35 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_factorization_l1079_107932


namespace NUMINAMATH_CALUDE_cyros_population_growth_l1079_107939

/-- The number of years it takes for the population to meet or exceed the island's capacity -/
def years_to_capacity (island_size : ℕ) (land_per_person : ℕ) (initial_population : ℕ) (doubling_period : ℕ) : ℕ :=
  sorry

theorem cyros_population_growth :
  years_to_capacity 32000 2 500 30 = 150 :=
sorry

end NUMINAMATH_CALUDE_cyros_population_growth_l1079_107939


namespace NUMINAMATH_CALUDE_acceptable_outfits_l1079_107954

/-- The number of shirts, pants, and hats available -/
def num_items : ℕ := 8

/-- The number of colors available for each item -/
def num_colors : ℕ := 8

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_items^3

/-- The number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- The number of outfits where shirt and pants are the same color but hat is different -/
def shirt_pants_same : ℕ := num_colors * (num_colors - 1)

/-- Theorem stating the number of acceptable outfit combinations -/
theorem acceptable_outfits : 
  total_combinations - same_color_outfits - shirt_pants_same = 448 := by
  sorry

end NUMINAMATH_CALUDE_acceptable_outfits_l1079_107954


namespace NUMINAMATH_CALUDE_power_division_equality_l1079_107945

theorem power_division_equality : (2 ^ 24) / (8 ^ 3) = 32768 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l1079_107945


namespace NUMINAMATH_CALUDE_tileB_smallest_unique_p_l1079_107979

/-- Represents a rectangular tile with four labeled sides -/
structure Tile where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The set of all tiles -/
def tiles : Finset Tile := sorry

/-- Tile A -/
def tileA : Tile := { p := 5, q := 2, r := 8, s := 11 }

/-- Tile B -/
def tileB : Tile := { p := 2, q := 1, r := 4, s := 7 }

/-- Tile C -/
def tileC : Tile := { p := 4, q := 9, r := 6, s := 3 }

/-- Tile D -/
def tileD : Tile := { p := 10, q := 6, r := 5, s := 9 }

/-- Tile E -/
def tileE : Tile := { p := 11, q := 3, r := 7, s := 0 }

/-- Function to check if a value is unique among all tiles -/
def isUnique (t : Tile) (f : Tile → ℤ) : Prop :=
  ∀ t' ∈ tiles, t' ≠ t → f t' ≠ f t

/-- Theorem: Tile B has the smallest unique p value -/
theorem tileB_smallest_unique_p :
  isUnique tileB Tile.p ∧ 
  ∀ t ∈ tiles, isUnique t Tile.p → tileB.p ≤ t.p :=
sorry

end NUMINAMATH_CALUDE_tileB_smallest_unique_p_l1079_107979


namespace NUMINAMATH_CALUDE_distance_between_points_l1079_107935

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-3.5, -4.5)
  let p2 : ℝ × ℝ := (3.5, 2.5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1079_107935


namespace NUMINAMATH_CALUDE_five_letter_words_same_ends_l1079_107991

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The number of letters that can vary in the word -/
def variable_letters : ℕ := word_length - 2

theorem five_letter_words_same_ends : 
  alphabet_size ^ variable_letters = 456976 := by
  sorry


end NUMINAMATH_CALUDE_five_letter_words_same_ends_l1079_107991


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1079_107969

theorem quadratic_transformation (p q r : ℝ) :
  (∀ x, p * x^2 + q * x + r = 7 * (x - 5)^2 + 14) →
  ∃ m k, ∀ x, 5 * p * x^2 + 5 * q * x + 5 * r = m * (x - 5)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1079_107969


namespace NUMINAMATH_CALUDE_members_playing_two_sports_l1079_107931

theorem members_playing_two_sports
  (total_members : ℕ)
  (badminton_players : ℕ)
  (tennis_players : ℕ)
  (soccer_players : ℕ)
  (no_sport_players : ℕ)
  (badminton_tennis : ℕ)
  (badminton_soccer : ℕ)
  (tennis_soccer : ℕ)
  (h1 : total_members = 60)
  (h2 : badminton_players = 25)
  (h3 : tennis_players = 32)
  (h4 : soccer_players = 14)
  (h5 : no_sport_players = 5)
  (h6 : badminton_tennis = 10)
  (h7 : badminton_soccer = 8)
  (h8 : tennis_soccer = 6)
  (h9 : badminton_tennis + badminton_soccer + tennis_soccer ≤ badminton_players + tennis_players + soccer_players) :
  badminton_tennis + badminton_soccer + tennis_soccer = 24 :=
by sorry

end NUMINAMATH_CALUDE_members_playing_two_sports_l1079_107931


namespace NUMINAMATH_CALUDE_magnitude_of_OP_l1079_107960

/-- Given vectors OA and OB, and the relation between AP and AB, prove the magnitude of OP --/
theorem magnitude_of_OP (OA OB OP : ℝ × ℝ) : 
  OA = (1, 2) → 
  OB = (-2, -1) → 
  2 * (OP - OA) = OB - OA → 
  Real.sqrt ((OP.1)^2 + (OP.2)^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_OP_l1079_107960


namespace NUMINAMATH_CALUDE_martha_initial_pantry_bottles_l1079_107975

/-- The number of bottles of juice Martha initially had in the pantry -/
def initial_pantry_bottles : ℕ := sorry

/-- The number of bottles of juice Martha initially had in the refrigerator -/
def initial_fridge_bottles : ℕ := 4

/-- The number of bottles of juice Martha bought during the week -/
def bought_bottles : ℕ := 5

/-- The number of bottles of juice Martha and her family drank during the week -/
def drunk_bottles : ℕ := 3

/-- The number of bottles of juice left at the end of the week -/
def remaining_bottles : ℕ := 10

theorem martha_initial_pantry_bottles :
  initial_pantry_bottles = 4 :=
by sorry

end NUMINAMATH_CALUDE_martha_initial_pantry_bottles_l1079_107975


namespace NUMINAMATH_CALUDE_min_a1_value_l1079_107987

/-- An arithmetic sequence with positive integer terms -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_a1_value (a : ℕ → ℕ) (h_arith : ArithmeticSequence a) (h_a9 : a 9 = 2023) :
  (∀ n : ℕ, a n > 0) → a 1 ≥ 7 ∧ ∃ a' : ℕ → ℕ, ArithmeticSequence a' ∧ a' 9 = 2023 ∧ a' 1 = 7 :=
sorry

end NUMINAMATH_CALUDE_min_a1_value_l1079_107987


namespace NUMINAMATH_CALUDE_work_completion_time_l1079_107904

-- Define work rates as fractions of work completed per hour
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 12
def work_rate_BC : ℚ := 1 / 3

-- Define the time taken for A and C together
def time_AC : ℚ := 2

theorem work_completion_time :
  let work_rate_C : ℚ := work_rate_BC - work_rate_B
  let work_rate_AC : ℚ := work_rate_A + work_rate_C
  time_AC = 1 / work_rate_AC :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1079_107904


namespace NUMINAMATH_CALUDE_robin_gum_count_l1079_107957

theorem robin_gum_count (initial_gum : Real) (additional_gum : Real) : 
  initial_gum = 18.5 → additional_gum = 44.25 → initial_gum + additional_gum = 62.75 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l1079_107957


namespace NUMINAMATH_CALUDE_simplify_expression_l1079_107971

theorem simplify_expression (x y : ℝ) :
  5 * x^4 + 3 * x^2 * y - 4 - 3 * x^2 * y - 3 * x^4 - 1 = 2 * x^4 - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1079_107971


namespace NUMINAMATH_CALUDE_car_rental_cost_per_km_l1079_107990

theorem car_rental_cost_per_km (samuel_fixed_cost carrey_fixed_cost carrey_per_km distance : ℝ) 
  (h1 : samuel_fixed_cost = 24)
  (h2 : carrey_fixed_cost = 20)
  (h3 : carrey_per_km = 0.25)
  (h4 : distance = 44.44444444444444)
  (h5 : ∃ samuel_per_km : ℝ, samuel_fixed_cost + samuel_per_km * distance = carrey_fixed_cost + carrey_per_km * distance) :
  ∃ samuel_per_km : ℝ, samuel_per_km = 0.16 := by
sorry


end NUMINAMATH_CALUDE_car_rental_cost_per_km_l1079_107990


namespace NUMINAMATH_CALUDE_worker_payment_schedule_l1079_107924

/-- Represents the worker payment schedule problem -/
theorem worker_payment_schedule 
  (daily_wage : ℕ) 
  (daily_return : ℕ) 
  (days_not_worked : ℕ) : 
  daily_wage = 100 → 
  daily_return = 25 → 
  days_not_worked = 24 → 
  ∃ (days_worked : ℕ), 
    daily_wage * days_worked = daily_return * days_not_worked ∧ 
    days_worked + days_not_worked = 30 := by
  sorry

end NUMINAMATH_CALUDE_worker_payment_schedule_l1079_107924


namespace NUMINAMATH_CALUDE_score_ordering_l1079_107900

-- Define the set of people
inductive Person : Type
| K : Person  -- Kaleana
| Q : Person  -- Quay
| M : Person  -- Marty
| S : Person  -- Shana

-- Define a function to represent the score of each person
variable (score : Person → ℕ)

-- Define the conditions
axiom quay_thought : score Person.Q = score Person.K
axiom marty_thought : score Person.M > score Person.K
axiom shana_thought : score Person.S < score Person.K

-- Define the theorem to prove
theorem score_ordering :
  score Person.S < score Person.Q ∧ score Person.Q < score Person.M :=
sorry

end NUMINAMATH_CALUDE_score_ordering_l1079_107900


namespace NUMINAMATH_CALUDE_remainder_not_always_power_of_four_l1079_107938

theorem remainder_not_always_power_of_four :
  ∃ n : ℕ, n ≥ 2 ∧ ∃ k : ℕ, (2^(2^n) : ℕ) % (2^n - 1) = k ∧ ¬∃ m : ℕ, k = 4^m := by
  sorry

end NUMINAMATH_CALUDE_remainder_not_always_power_of_four_l1079_107938


namespace NUMINAMATH_CALUDE_melissa_driving_hours_l1079_107992

/-- The number of hours Melissa spends driving in a year -/
def driving_hours_per_year (trips_per_month : ℕ) (hours_per_trip : ℕ) : ℕ :=
  trips_per_month * 12 * hours_per_trip

/-- Proof that Melissa spends 72 hours driving in a year -/
theorem melissa_driving_hours :
  driving_hours_per_year 2 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_melissa_driving_hours_l1079_107992


namespace NUMINAMATH_CALUDE_students_per_section_after_changes_l1079_107996

theorem students_per_section_after_changes 
  (initial_students_per_section : ℕ)
  (new_sections : ℕ)
  (total_sections_after : ℕ)
  (new_students : ℕ)
  (h1 : initial_students_per_section = 24)
  (h2 : new_sections = 3)
  (h3 : total_sections_after = 16)
  (h4 : new_students = 24) :
  (initial_students_per_section * (total_sections_after - new_sections) + new_students) / total_sections_after = 21 :=
by sorry

end NUMINAMATH_CALUDE_students_per_section_after_changes_l1079_107996


namespace NUMINAMATH_CALUDE_abs_increasing_on_unit_interval_l1079_107948

-- Define the function f(x) = |x|
def f (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_increasing_on_unit_interval : 
  ∀ x y : ℝ, 0 < x → x < y → y < 1 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_abs_increasing_on_unit_interval_l1079_107948


namespace NUMINAMATH_CALUDE_scientific_notation_of_11580000_l1079_107961

theorem scientific_notation_of_11580000 :
  (11580000 : ℝ) = 1.158 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_11580000_l1079_107961


namespace NUMINAMATH_CALUDE_water_pump_problem_l1079_107920

theorem water_pump_problem (t₁ t₂ t_combined : ℝ) 
  (h₁ : t₂ = 6)
  (h₂ : t_combined = 3.6)
  (h₃ : 1 / t₁ + 1 / t₂ = 1 / t_combined) :
  t₁ = 9 := by
sorry

end NUMINAMATH_CALUDE_water_pump_problem_l1079_107920


namespace NUMINAMATH_CALUDE_seattle_seahawks_field_goals_l1079_107941

theorem seattle_seahawks_field_goals :
  ∀ (total_score touchdown_score field_goal_score touchdown_count : ℕ),
    total_score = 37 →
    touchdown_score = 7 →
    field_goal_score = 3 →
    touchdown_count = 4 →
    ∃ (field_goal_count : ℕ),
      total_score = touchdown_count * touchdown_score + field_goal_count * field_goal_score ∧
      field_goal_count = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_seattle_seahawks_field_goals_l1079_107941


namespace NUMINAMATH_CALUDE_projection_squared_magnitude_l1079_107918

-- Define the 3D Cartesian coordinate system
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define point A
def A : Point3D := ⟨3, 7, -4⟩

-- Define point B as the projection of A onto the xOz plane
def B : Point3D := ⟨A.x, 0, A.z⟩

-- Define the squared magnitude of a vector
def squaredMagnitude (p : Point3D) : ℝ :=
  p.x^2 + p.y^2 + p.z^2

-- Theorem statement
theorem projection_squared_magnitude :
  squaredMagnitude B = 25 := by sorry

end NUMINAMATH_CALUDE_projection_squared_magnitude_l1079_107918


namespace NUMINAMATH_CALUDE_faye_money_left_l1079_107914

def initial_money : ℕ := 20
def mother_multiplier : ℕ := 2
def cupcake_price : ℚ := 3/2
def cupcake_quantity : ℕ := 10
def cookie_box_price : ℕ := 3
def cookie_box_quantity : ℕ := 5

theorem faye_money_left :
  let total_money := initial_money + mother_multiplier * initial_money
  let spent_money := cupcake_price * cupcake_quantity + cookie_box_price * cookie_box_quantity
  total_money - spent_money = 30 := by
sorry

end NUMINAMATH_CALUDE_faye_money_left_l1079_107914


namespace NUMINAMATH_CALUDE_min_value_inequality_l1079_107912

theorem min_value_inequality (r s t : ℝ) (h1 : 1 ≤ r) (h2 : r ≤ s) (h3 : s ≤ t) (h4 : t ≤ 4) :
  (r - 1)^2 + (s/r - 1)^2 + (t/s - 1)^2 + (4/t - 1)^2 ≥ 4 * (Real.sqrt 2 - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1079_107912


namespace NUMINAMATH_CALUDE_johns_umbrella_cost_l1079_107994

/-- The total cost of John's umbrellas -/
def total_cost (house_umbrellas car_umbrellas unit_cost : ℕ) : ℕ :=
  (house_umbrellas + car_umbrellas) * unit_cost

/-- Proof that John's total umbrella cost is $24 -/
theorem johns_umbrella_cost :
  total_cost 2 1 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_johns_umbrella_cost_l1079_107994


namespace NUMINAMATH_CALUDE_solve_equation_l1079_107946

theorem solve_equation :
  let y := 45 / (8 - 3/7)
  y = 315/53 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l1079_107946


namespace NUMINAMATH_CALUDE_neighbor_birth_year_l1079_107909

def is_valid_year (year : ℕ) : Prop :=
  year ≥ 1000 ∧ year ≤ 9999

def first_two_digits (year : ℕ) : ℕ :=
  year / 100

def last_two_digits (year : ℕ) : ℕ :=
  year % 100

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def diff_of_digits (n : ℕ) : ℕ :=
  (n / 10) - (n % 10)

theorem neighbor_birth_year :
  ∀ year : ℕ, is_valid_year year →
    (sum_of_digits (first_two_digits year) = diff_of_digits (last_two_digits year)) →
    year = 1890 :=
by sorry

end NUMINAMATH_CALUDE_neighbor_birth_year_l1079_107909


namespace NUMINAMATH_CALUDE_max_acute_angles_convex_polygon_l1079_107940

-- Define a convex polygon
structure ConvexPolygon where
  n : ℕ  -- number of sides
  convex : Bool  -- property of being convex

-- Define the theorem
theorem max_acute_angles_convex_polygon (p : ConvexPolygon) : 
  p.convex = true →  -- the polygon is convex
  (∃ (sum_exterior_angles : ℝ), sum_exterior_angles = 360) →  -- sum of exterior angles is 360°
  (∀ (i : ℕ) (interior_angle exterior_angle : ℝ), 
    i < p.n → interior_angle + exterior_angle = 180) →  -- interior and exterior angles are supplementary
  (∃ (max_acute : ℕ), max_acute = 3 ∧ 
    ∀ (acute_count : ℕ), acute_count ≤ max_acute) :=
by sorry

end NUMINAMATH_CALUDE_max_acute_angles_convex_polygon_l1079_107940


namespace NUMINAMATH_CALUDE_probability_not_red_blue_purple_l1079_107908

def total_balls : ℕ := 240
def white_balls : ℕ := 60
def green_balls : ℕ := 70
def yellow_balls : ℕ := 45
def red_balls : ℕ := 35
def blue_balls : ℕ := 20
def purple_balls : ℕ := 10

theorem probability_not_red_blue_purple :
  let favorable_outcomes := total_balls - (red_balls + blue_balls + purple_balls)
  (favorable_outcomes : ℚ) / total_balls = 35 / 48 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_red_blue_purple_l1079_107908


namespace NUMINAMATH_CALUDE_chess_team_boys_count_l1079_107933

theorem chess_team_boys_count :
  ∀ (total_members : ℕ) (total_attendees : ℕ) (boys : ℕ) (girls : ℕ),
  total_members = 30 →
  total_attendees = 20 →
  total_members = boys + girls →
  total_attendees = boys + (girls / 3) →
  boys = 15 := by
sorry

end NUMINAMATH_CALUDE_chess_team_boys_count_l1079_107933


namespace NUMINAMATH_CALUDE_painted_cubes_count_l1079_107919

/-- A cube-based construction with 5 layers -/
structure CubeConstruction where
  middle_layer : Nat
  other_layers : Nat
  unpainted_cubes : Nat

/-- The number of cubes with at least one face painted in the construction -/
def painted_cubes (c : CubeConstruction) : Nat :=
  c.middle_layer + 4 * c.other_layers - c.unpainted_cubes

/-- Theorem: In the given cube construction, 104 cubes have at least one face painted -/
theorem painted_cubes_count (c : CubeConstruction) 
  (h1 : c.middle_layer = 16)
  (h2 : c.other_layers = 24)
  (h3 : c.unpainted_cubes = 8) : 
  painted_cubes c = 104 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l1079_107919


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1079_107911

/-- An isosceles triangle with two sides of length 6 and one side of length 2 has a perimeter of 14. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 6 ∧ b = 6 ∧ c = 2 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  a + b + c = 14 :=
by
  sorry

#check isosceles_triangle_perimeter

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1079_107911


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l1079_107922

theorem pencil_pen_cost (x y : ℚ) : 
  (7 * x + 6 * y = 46.8) → 
  (3 * x + 5 * y = 32.2) → 
  (x = 2.4 ∧ y = 5) := by
sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l1079_107922


namespace NUMINAMATH_CALUDE_virus_growth_time_l1079_107986

/-- Represents the memory occupied by the virus in KB -/
def virus_memory (t : ℕ) : ℕ := 2 * 2^t

/-- Converts MB to KB -/
def mb_to_kb (mb : ℕ) : ℕ := mb * 2^10

/-- The theorem stating that the virus will occupy 64MB after 45 minutes -/
theorem virus_growth_time : ∃ (t : ℕ), t * 3 = 45 ∧ virus_memory t = mb_to_kb 64 := by
  sorry

end NUMINAMATH_CALUDE_virus_growth_time_l1079_107986


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1079_107905

theorem complex_equation_solution (z : ℂ) (h : z * (3 - I) = 1 - I) : z = 2/5 - 1/5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1079_107905


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1079_107929

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1079_107929


namespace NUMINAMATH_CALUDE_greening_task_equation_l1079_107927

theorem greening_task_equation (x : ℝ) 
  (h1 : x > 0) -- x must be positive
  (h2 : 600000 / (x * 1000) - 600000 / ((1 + 0.25) * x * 1000) = 30) : 
  60 * (1 + 0.25) / x - 60 / x = 30 := by
  sorry

end NUMINAMATH_CALUDE_greening_task_equation_l1079_107927


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1079_107944

theorem floor_equation_solution (x : ℝ) (h1 : x > 0) (h2 : x * ⌊x⌋ = 90) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1079_107944


namespace NUMINAMATH_CALUDE_jellybean_problem_l1079_107982

/-- The number of jellybeans initially in the jar -/
def initial_jellybeans : ℕ := 90

/-- The number of jellybeans Samantha took -/
def samantha_took : ℕ := 24

/-- The number of jellybeans Shelby ate -/
def shelby_ate : ℕ := 12

/-- The final number of jellybeans in the jar -/
def final_jellybeans : ℕ := 72

theorem jellybean_problem :
  initial_jellybeans - samantha_took - shelby_ate +
  ((samantha_took + shelby_ate) / 2) = final_jellybeans :=
by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l1079_107982


namespace NUMINAMATH_CALUDE_weight_loss_in_april_l1079_107958

/-- Given Michael's weight loss plan:
  * total_weight: Total weight Michael wants to lose
  * march_loss: Weight lost in March
  * may_loss: Weight to lose in May
  * april_loss: Weight lost in April

  This theorem proves that the weight lost in April is equal to
  the total weight minus the weight lost in March and the weight to lose in May. -/
theorem weight_loss_in_april 
  (total_weight march_loss may_loss april_loss : ℕ) : 
  april_loss = total_weight - march_loss - may_loss := by
  sorry

#check weight_loss_in_april

end NUMINAMATH_CALUDE_weight_loss_in_april_l1079_107958


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1079_107964

/-- A parabola with vertex at the origin and focus on the x-axis. -/
structure Parabola where
  /-- The x-coordinate of the focus -/
  p : ℝ
  /-- The parabola passes through this point -/
  point : ℝ × ℝ

/-- The distance from the focus to the directrix for a parabola -/
def focusDirectrixDistance (c : Parabola) : ℝ :=
  c.p

theorem parabola_focus_directrix_distance 
  (c : Parabola) 
  (h1 : c.point = (1, 3)) : 
  focusDirectrixDistance c = 9/2 := by
  sorry

#check parabola_focus_directrix_distance

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1079_107964


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1079_107903

/-- A triangle with side lengths a, b, and c is isosceles if at least two of its sides are equal -/
def IsIsosceles (a b c : ℝ) : Prop := a = b ∨ b = c ∨ a = c

/-- The perimeter of a triangle with side lengths a, b, and c -/
def Perimeter (a b c : ℝ) : ℝ := a + b + c

theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  IsIsosceles a b c →
  (a = 5 ∧ b = 8) ∨ (a = 8 ∧ b = 5) ∨ (b = 5 ∧ c = 8) ∨ (b = 8 ∧ c = 5) ∨ (a = 5 ∧ c = 8) ∨ (a = 8 ∧ c = 5) →
  Perimeter a b c = 18 ∨ Perimeter a b c = 21 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1079_107903


namespace NUMINAMATH_CALUDE_positive_real_equality_l1079_107970

theorem positive_real_equality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a^b = b^a) (h2 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_equality_l1079_107970


namespace NUMINAMATH_CALUDE_g_is_even_f_periodic_4_l1079_107974

-- Define the real-valued function f
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem 1: g is an even function
theorem g_is_even : ∀ x : ℝ, g f x = g f (-x) := by sorry

-- Theorem 2: f is periodic with period 4 if it's odd and f(x+2) is odd
theorem f_periodic_4 (h1 : ∀ x : ℝ, f (-x) = -f x) 
                     (h2 : ∀ x : ℝ, f (-(x+2)) = -f (x+2)) : 
  ∀ x : ℝ, f (x + 4) = f x := by sorry

end NUMINAMATH_CALUDE_g_is_even_f_periodic_4_l1079_107974


namespace NUMINAMATH_CALUDE_bamboo_pole_problem_l1079_107966

/-- 
Given a bamboo pole of height 10 feet, if the top part when bent to the ground 
reaches a point 3 feet from the base, then the length of the broken part is 109/20 feet.
-/
theorem bamboo_pole_problem (h : ℝ) (x : ℝ) (y : ℝ) :
  h = 10 ∧ 
  x + y = h ∧ 
  x^2 + 3^2 = y^2 →
  y = 109/20 := by
sorry

end NUMINAMATH_CALUDE_bamboo_pole_problem_l1079_107966


namespace NUMINAMATH_CALUDE_birch_count_is_87_l1079_107913

def is_valid_tree_arrangement (total_trees : ℕ) (birch_count : ℕ) : Prop :=
  ∃ (lime_count : ℕ),
    -- Total number of trees is 130
    total_trees = 130 ∧
    -- Sum of birches and limes is the total number of trees
    birch_count + lime_count = total_trees ∧
    -- There is at least one birch and one lime
    birch_count > 0 ∧ lime_count > 0 ∧
    -- The number of limes is equal to the number of groups of two birches plus one lime
    lime_count = (birch_count - 1) / 2 ∧
    -- There is exactly one group of three consecutive birches
    (birch_count - 1) % 2 = 1

theorem birch_count_is_87 :
  ∃ (birch_count : ℕ), is_valid_tree_arrangement 130 birch_count ∧ birch_count = 87 :=
sorry

end NUMINAMATH_CALUDE_birch_count_is_87_l1079_107913


namespace NUMINAMATH_CALUDE_exists_unique_box_l1079_107906

/-- Represents a rectangular box with square base -/
structure Box where
  x : ℝ  -- side length of square base
  h : ℝ  -- height of box

/-- Calculates the surface area of the box -/
def surfaceArea (b : Box) : ℝ := 2 * b.x^2 + 4 * b.x * b.h

/-- Calculates the volume of the box -/
def volume (b : Box) : ℝ := b.x^2 * b.h

/-- Theorem stating the existence of a box meeting the given conditions -/
theorem exists_unique_box :
  ∃! b : Box,
    b.h = 2 * b.x + 2 ∧
    surfaceArea b ≥ 150 ∧
    volume b = 100 ∧
    b.x > 0 ∧
    b.h > 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_unique_box_l1079_107906


namespace NUMINAMATH_CALUDE_inequality_proof_l1079_107942

theorem inequality_proof (n : ℕ) (h : n > 1) : (4^n : ℚ) / (n + 1) < (2*n).factorial / (n.factorial ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1079_107942


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l1079_107977

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, 
    (10000 ≤ m ∧ m < 100000) →
    (∃ x : ℕ, m = x^2) →
    (∃ y : ℕ, m = y^3) →
    n ≤ m) ∧                  -- least such number
  n = 15625 := by
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l1079_107977


namespace NUMINAMATH_CALUDE_parallel_lines_intersection_l1079_107981

theorem parallel_lines_intersection (n : ℕ) : 
  (10 - 1) * (n - 1) = 1260 → n = 141 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_intersection_l1079_107981


namespace NUMINAMATH_CALUDE_circle_intersection_m_range_l1079_107980

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*m*y + m + 6 = 0

-- Define the condition that intersections are on the same side of the origin
def intersections_same_side (m : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, y₁ * y₂ > 0 ∧ circle_equation 0 y₁ m ∧ circle_equation 0 y₂ m

-- Theorem statement
theorem circle_intersection_m_range :
  ∀ m : ℝ, intersections_same_side m → (m > 2 ∨ (-6 < m ∧ m < -2)) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_m_range_l1079_107980


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1079_107959

-- Define the repeating decimal 0.4̅36̅
def repeating_decimal : ℚ := 0.4 + (36 / 990)

-- Theorem statement
theorem repeating_decimal_equals_fraction : repeating_decimal = 24 / 55 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1079_107959


namespace NUMINAMATH_CALUDE_distance_comparisons_l1079_107976

-- Define the driving conditions
def joseph_speed1 : ℝ := 48
def joseph_time1 : ℝ := 2.5
def joseph_speed2 : ℝ := 60
def joseph_time2 : ℝ := 1.5

def kyle_speed1 : ℝ := 70
def kyle_time1 : ℝ := 2
def kyle_speed2 : ℝ := 63
def kyle_time2 : ℝ := 2.5

def emily_speed : ℝ := 65
def emily_time : ℝ := 3

-- Define the distances driven
def joseph_distance : ℝ := joseph_speed1 * joseph_time1 + joseph_speed2 * joseph_time2
def kyle_distance : ℝ := kyle_speed1 * kyle_time1 + kyle_speed2 * kyle_time2
def emily_distance : ℝ := emily_speed * emily_time

-- Theorem to prove the distance comparisons
theorem distance_comparisons :
  (joseph_distance = 210) ∧
  (kyle_distance = 297.5) ∧
  (emily_distance = 195) ∧
  (joseph_distance - kyle_distance = -87.5) ∧
  (emily_distance - joseph_distance = -15) ∧
  (emily_distance - kyle_distance = -102.5) :=
by sorry

end NUMINAMATH_CALUDE_distance_comparisons_l1079_107976


namespace NUMINAMATH_CALUDE_zoo_visitors_ratio_l1079_107956

theorem zoo_visitors_ratio :
  let friday_visitors : ℕ := 1250
  let saturday_visitors : ℕ := 3750
  (saturday_visitors : ℚ) / (friday_visitors : ℚ) = 3 := by
sorry

end NUMINAMATH_CALUDE_zoo_visitors_ratio_l1079_107956


namespace NUMINAMATH_CALUDE_maurice_age_proof_l1079_107978

/-- Ron's current age -/
def ron_current_age : ℕ := 43

/-- Maurice's current age -/
def maurice_current_age : ℕ := 7

/-- Theorem stating that Maurice's current age is 7 years -/
theorem maurice_age_proof :
  (ron_current_age + 5 = 4 * (maurice_current_age + 5)) →
  maurice_current_age = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_maurice_age_proof_l1079_107978


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1079_107916

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1079_107916


namespace NUMINAMATH_CALUDE_zero_in_interval_l1079_107923

def f (x : ℝ) : ℝ := x^3 + x - 4

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 := by
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1079_107923


namespace NUMINAMATH_CALUDE_not_eight_sum_l1079_107985

theorem not_eight_sum (a b c : ℕ) (h : 2^a * 3^b * 4^c = 192) : a + b + c ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_not_eight_sum_l1079_107985


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1079_107930

def U : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def A : Set ℤ := {-2, 0, 2, 4}
def B : Set ℤ := {-2, 0, 4, 6, 8}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {6, 8} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1079_107930


namespace NUMINAMATH_CALUDE_gcd_54000_36000_l1079_107983

theorem gcd_54000_36000 : Nat.gcd 54000 36000 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_54000_36000_l1079_107983


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l1079_107902

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 - a - 2013 = 0) → 
  (b^2 - b - 2013 = 0) → 
  (a ≠ b) →
  (a^2 + 2*a + 3*b - 2 = 2014) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l1079_107902


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1079_107993

theorem polynomial_remainder (x : ℝ) : (x^4 + x + 2) % (x - 3) = 86 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1079_107993


namespace NUMINAMATH_CALUDE_lcm_gcd_product_15_45_l1079_107953

theorem lcm_gcd_product_15_45 : Nat.lcm 15 45 * Nat.gcd 15 45 = 675 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_15_45_l1079_107953


namespace NUMINAMATH_CALUDE_ducks_theorem_l1079_107984

def ducks_remaining (initial : ℕ) : ℕ :=
  let after_first := initial - (initial / 4)
  let after_second := after_first - (after_first / 6)
  after_second - (after_second * 3 / 10)

theorem ducks_theorem : ducks_remaining 320 = 140 := by
  sorry

end NUMINAMATH_CALUDE_ducks_theorem_l1079_107984


namespace NUMINAMATH_CALUDE_rotated_line_equation_l1079_107989

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a line 90 degrees counterclockwise around a given point --/
def rotateLine90 (l : Line) (p : Point) : Line :=
  sorry

/-- The initial line l₀ --/
def l₀ : Line :=
  { slope := 1, intercept := 1 }

/-- The point P around which the line is rotated --/
def P : Point :=
  { x := 3, y := 1 }

/-- The rotated line l --/
def l : Line :=
  rotateLine90 l₀ P

theorem rotated_line_equation :
  l.slope * 3 + l.intercept = 1 ∧ l.slope = -1 ∧ l.intercept = 4 :=
sorry

end NUMINAMATH_CALUDE_rotated_line_equation_l1079_107989


namespace NUMINAMATH_CALUDE_seventh_term_is_13_4_l1079_107950

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference of the sequence
  d : ℝ
  -- The sum of the first four terms is 14
  sum_first_four : a + (a + d) + (a + 2*d) + (a + 3*d) = 14
  -- The fifth term is 9
  fifth_term : a + 4*d = 9

/-- The seventh term of the arithmetic sequence is 13.4 -/
theorem seventh_term_is_13_4 (seq : ArithmeticSequence) :
  seq.a + 6*seq.d = 13.4 := by
  sorry


end NUMINAMATH_CALUDE_seventh_term_is_13_4_l1079_107950


namespace NUMINAMATH_CALUDE_constant_k_equality_l1079_107951

theorem constant_k_equality (k : ℝ) : 
  (∀ x : ℝ, -x^2 - (k + 9)*x - 8 = -(x - 2)*(x - 4)) → k = -15 := by
  sorry

end NUMINAMATH_CALUDE_constant_k_equality_l1079_107951


namespace NUMINAMATH_CALUDE_quad_pair_sum_l1079_107907

/-- Two distinct quadratic polynomials with specific properties -/
structure QuadraticPair where
  f : ℝ → ℝ
  g : ℝ → ℝ
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  hf : f = fun x ↦ x^2 + p*x + q
  hg : g = fun x ↦ x^2 + r*x + s
  distinct : f ≠ g
  vertex_root : g (-p/2) = 0 ∧ f (-r/2) = 0
  same_min : ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₁, f x₁ = m) ∧ (∀ x, g x ≥ m) ∧ (∃ x₂, g x₂ = m)
  intersection : f 50 = -200 ∧ g 50 = -200

/-- The sum of coefficients p and r is -200 -/
theorem quad_pair_sum (qp : QuadraticPair) : qp.p + qp.r = -200 := by
  sorry

end NUMINAMATH_CALUDE_quad_pair_sum_l1079_107907


namespace NUMINAMATH_CALUDE_tenth_term_is_18_l1079_107901

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 2 = 2 ∧ 
  a 3 = 4

/-- The 10th term of the arithmetic sequence is 18 -/
theorem tenth_term_is_18 (a : ℕ → ℝ) (h : arithmetic_sequence a) : 
  a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_18_l1079_107901


namespace NUMINAMATH_CALUDE_school_students_problem_l1079_107937

theorem school_students_problem (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 150 →
  total = boys + girls →
  girls = (boys : ℚ) / 100 * total →
  boys = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_school_students_problem_l1079_107937


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1079_107973

-- Define a random variable following normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define probability function
noncomputable def probability {α : Type} (event : Set α) : ℝ := sorry

-- Theorem statement
theorem normal_distribution_probability 
  (ξ : normal_distribution 1 σ) 
  (h1 : probability {x | x < 1} = 1/2) 
  (h2 : probability {x | x > 2} = p) :
  probability {x | 0 < x ∧ x < 1} = 1/2 - p :=
sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1079_107973


namespace NUMINAMATH_CALUDE_line_points_k_value_l1079_107965

/-- Given a line with equation x = 2y + 5, if (m, n) and (m + 1, n + k) are two points on this line, then k = 1/2 -/
theorem line_points_k_value (m n k : ℝ) : 
  (m = 2 * n + 5) →  -- (m, n) is on the line
  (m + 1 = 2 * (n + k) + 5) →  -- (m + 1, n + k) is on the line
  k = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_line_points_k_value_l1079_107965


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l1079_107917

theorem equation_has_real_roots (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) + 2 * x :=
sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l1079_107917


namespace NUMINAMATH_CALUDE_min_value_of_f_l1079_107928

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + Real.exp (-x)

theorem min_value_of_f (a : ℝ) :
  (∃ k : ℝ, k * f a 0 + 1 = 0 ∧ k * (a - 1) = -1) →
  (∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x) ∧
  (∃ x : ℝ, f a x = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1079_107928


namespace NUMINAMATH_CALUDE_vector_magnitude_l1079_107962

def a (t : ℝ) : ℝ × ℝ := (2, t)
def b : ℝ × ℝ := (-1, 2)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_magnitude (t : ℝ) :
  parallel (a t) b →
  ‖(a t - b)‖ = 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1079_107962


namespace NUMINAMATH_CALUDE_triangle_midpoint_line_sum_l1079_107910

/-- Given a triangle ABC with vertices A(0,6), B(0,0), C(10,0), and D the midpoint of AB,
    the sum of the slope and y-intercept of line CD is 27/10 -/
theorem triangle_midpoint_line_sum (A B C D : ℝ × ℝ) : 
  A = (0, 6) → 
  B = (0, 0) → 
  C = (10, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := D.2
  m + b = 27 / 10 := by sorry

end NUMINAMATH_CALUDE_triangle_midpoint_line_sum_l1079_107910


namespace NUMINAMATH_CALUDE_BC_length_l1079_107999

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A, B, C, B', C', and D
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def B' : ℝ × ℝ := sorry
def C' : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry

-- Define the conditions
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω
axiom C_on_ω : C ∈ ω
axiom BC_is_diameter : sorry -- BC is a diameter of ω
axiom B'C'_parallel_BC : sorry -- B'C' is parallel to BC
axiom B'C'_tangent_ω : sorry -- B'C' is tangent to ω at D
axiom B'D_length : dist B' D = 4
axiom C'D_length : dist C' D = 6

-- Define the theorem
theorem BC_length : dist B C = 24/5 := by sorry

end NUMINAMATH_CALUDE_BC_length_l1079_107999


namespace NUMINAMATH_CALUDE_sum_in_base6_l1079_107952

/-- Converts a number from base 6 to base 10 -/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a number from base 10 to base 6 -/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec go (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else go (m / 6) ((m % 6) :: acc)
  go n []

/-- The main theorem to prove -/
theorem sum_in_base6 :
  let a := toBase10 [4, 4, 4]
  let b := toBase10 [6, 6]
  let c := toBase10 [4]
  toBase6 (a + b + c) = [6, 0, 2] := by sorry

end NUMINAMATH_CALUDE_sum_in_base6_l1079_107952


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1079_107963

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / (x - 5) ≥ 3 ↔ x ∈ Set.Ioo 5 6 ∪ {6} := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1079_107963


namespace NUMINAMATH_CALUDE_base_prime_representation_of_540_l1079_107949

/-- Base prime representation of a natural number -/
def BasePrimeRepresentation (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list represents a valid base prime representation -/
def IsValidBasePrimeRepresentation (l : List ℕ) : Prop :=
  sorry

theorem base_prime_representation_of_540 :
  let representation := [1, 3, 1]
  540 = 2^1 * 3^3 * 5^1 →
  IsValidBasePrimeRepresentation representation ∧
  BasePrimeRepresentation 540 = representation :=
by sorry

end NUMINAMATH_CALUDE_base_prime_representation_of_540_l1079_107949


namespace NUMINAMATH_CALUDE_max_value_xy_l1079_107972

theorem max_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 1) :
  xy ≤ 1/12 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 1 ∧ x₀*y₀ = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xy_l1079_107972


namespace NUMINAMATH_CALUDE_complex_magnitude_l1079_107936

theorem complex_magnitude (i : ℂ) (h : i * i = -1) :
  Complex.abs (i + 2 * i^2 + 3 * i^3) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1079_107936


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l1079_107967

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- State the theorem
theorem root_exists_in_interval :
  (∀ x ∈ Set.Icc 0 0.5, ContinuousAt f x) →
  f 0 < 0 →
  f 0.5 > 0 →
  ∃ x ∈ Set.Ioo 0 0.5, f x = 0 := by
  sorry

#check root_exists_in_interval

end NUMINAMATH_CALUDE_root_exists_in_interval_l1079_107967


namespace NUMINAMATH_CALUDE_tim_sugar_cookies_l1079_107925

/-- Represents the number of sugar cookies Tim baked given the total number of cookies and the ratio of cookie types. -/
def sugar_cookies (total : ℕ) (choc_ratio sugar_ratio pb_ratio : ℕ) : ℕ :=
  (sugar_ratio * total) / (choc_ratio + sugar_ratio + pb_ratio)

/-- Theorem stating that Tim baked 15 sugar cookies given the problem conditions. -/
theorem tim_sugar_cookies :
  sugar_cookies 30 2 5 3 = 15 := by
sorry

end NUMINAMATH_CALUDE_tim_sugar_cookies_l1079_107925


namespace NUMINAMATH_CALUDE_stating_production_constraint_equations_l1079_107955

/-- Represents the daily production capacity for type A toys -/
def type_A_production : ℕ := 200

/-- Represents the daily production capacity for type B toys -/
def type_B_production : ℕ := 100

/-- Represents the number of type A parts required for one complete toy -/
def type_A_parts_per_toy : ℕ := 1

/-- Represents the number of type B parts required for one complete toy -/
def type_B_parts_per_toy : ℕ := 2

/-- Represents the total number of production days -/
def total_days : ℕ := 30

/-- 
Theorem stating that the given system of equations correctly represents 
the production constraints for maximizing toy assembly within 30 days
-/
theorem production_constraint_equations 
  (x y : ℕ) : 
  (x + y = total_days ∧ 
   type_A_production * type_A_parts_per_toy * x = type_B_production * y) ↔ 
  (x + y = 30 ∧ 400 * x = 100 * y) :=
sorry

end NUMINAMATH_CALUDE_stating_production_constraint_equations_l1079_107955


namespace NUMINAMATH_CALUDE_infinitely_many_coprime_binomial_quotients_l1079_107915

/-- Given positive integers k, l, and m, there exist infinitely many positive integers n
    such that (n choose k) / m is a positive integer coprime with m. -/
theorem infinitely_many_coprime_binomial_quotients
  (k l m : ℕ+) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S,
    ∃ (q : ℕ+), Nat.choose n k.val = q.val * m.val ∧ Nat.Coprime q.val m.val :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_coprime_binomial_quotients_l1079_107915
