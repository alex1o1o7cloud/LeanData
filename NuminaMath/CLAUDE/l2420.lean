import Mathlib

namespace complex_fraction_simplification_l2420_242036

theorem complex_fraction_simplification :
  ∀ (z : ℂ), z = (3 : ℂ) + 8 * I →
  (1 / ((1 : ℂ) - 4 * I)) * z = 2 + (3 / 17) * I :=
by
  sorry

end complex_fraction_simplification_l2420_242036


namespace x_squared_plus_reciprocal_l2420_242090

theorem x_squared_plus_reciprocal (x : ℝ) (h : 54 = x^4 + 1/x^4) :
  x^2 + 1/x^2 = Real.sqrt 56 := by
  sorry

end x_squared_plus_reciprocal_l2420_242090


namespace empty_solution_set_implies_a_range_l2420_242043

theorem empty_solution_set_implies_a_range 
  (h : ∀ x : ℝ, ¬(|x + 3| + |x - 1| < a^2 - 3*a)) : 
  a ∈ Set.Icc (-1 : ℝ) 4 := by
  sorry

end empty_solution_set_implies_a_range_l2420_242043


namespace binomial_square_proof_l2420_242040

theorem binomial_square_proof :
  ∃ (r s : ℚ), (r * x + s)^2 = (81/16 : ℚ) * x^2 + 18 * x + 16 :=
by
  sorry

end binomial_square_proof_l2420_242040


namespace total_grains_in_gray_parts_l2420_242060

/-- Represents a circle with grains -/
structure GrainCircle where
  total : ℕ
  intersection : ℕ

/-- Calculates the number of grains in the non-intersecting part of a circle -/
def nonIntersectingGrains (circle : GrainCircle) : ℕ :=
  circle.total - circle.intersection

/-- The main theorem -/
theorem total_grains_in_gray_parts 
  (circle1 circle2 : GrainCircle)
  (h1 : circle1.total = 87)
  (h2 : circle2.total = 110)
  (h3 : circle1.intersection = 68)
  (h4 : circle2.intersection = 68) :
  nonIntersectingGrains circle1 + nonIntersectingGrains circle2 = 61 := by
  sorry

#eval nonIntersectingGrains { total := 87, intersection := 68 } +
      nonIntersectingGrains { total := 110, intersection := 68 }

end total_grains_in_gray_parts_l2420_242060


namespace sum_of_four_with_common_divisors_l2420_242076

theorem sum_of_four_with_common_divisors (n : ℕ) (h : n > 31) :
  ∃ (a b c d : ℕ), 
    n = a + b + c + d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (∃ (k₁ : ℕ), k₁ > 1 ∧ k₁ ∣ a ∧ k₁ ∣ b) ∧
    (∃ (k₂ : ℕ), k₂ > 1 ∧ k₂ ∣ a ∧ k₂ ∣ c) ∧
    (∃ (k₃ : ℕ), k₃ > 1 ∧ k₃ ∣ a ∧ k₃ ∣ d) ∧
    (∃ (k₄ : ℕ), k₄ > 1 ∧ k₄ ∣ b ∧ k₄ ∣ c) ∧
    (∃ (k₅ : ℕ), k₅ > 1 ∧ k₅ ∣ b ∧ k₅ ∣ d) ∧
    (∃ (k₆ : ℕ), k₆ > 1 ∧ k₆ ∣ c ∧ k₆ ∣ d) :=
by sorry

end sum_of_four_with_common_divisors_l2420_242076


namespace point_distance_theorem_l2420_242072

theorem point_distance_theorem (x y : ℝ) (h1 : x > 1) :
  y = 12 ∧ (x - 1)^2 + (y - 6)^2 = 10^2 →
  x^2 + y^2 = 15^2 := by
sorry

end point_distance_theorem_l2420_242072


namespace race_heartbeats_l2420_242024

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Proves that the total number of heartbeats during the race is 27000 -/
theorem race_heartbeats :
  let heart_rate : ℕ := 180  -- heartbeats per minute
  let pace : ℕ := 3          -- minutes per kilometer
  let distance : ℕ := 50     -- kilometers
  total_heartbeats heart_rate pace distance = 27000 := by
  sorry


end race_heartbeats_l2420_242024


namespace largest_non_sum_36_composite_l2420_242007

/-- A function that checks if a number is composite -/
def is_composite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a number can be expressed as the sum of a positive integral multiple of 36 and a positive composite integer -/
def is_sum_of_multiple_36_and_composite (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ is_composite m ∧ n = 36 * k + m

/-- The theorem stating that 145 is the largest positive integer that cannot be expressed as the sum of a positive integral multiple of 36 and a positive composite integer -/
theorem largest_non_sum_36_composite : 
  (∀ n : ℕ, n > 145 → is_sum_of_multiple_36_and_composite n) ∧
  ¬is_sum_of_multiple_36_and_composite 145 :=
sorry

end largest_non_sum_36_composite_l2420_242007


namespace smallest_digit_divisible_by_nine_l2420_242081

theorem smallest_digit_divisible_by_nine :
  ∃ (d : ℕ), d < 10 ∧ 
    (∀ (x : ℕ), x < d → ¬(528000 + x * 100 + 46) % 9 = 0) ∧
    (528000 + d * 100 + 46) % 9 = 0 :=
by
  -- The proof goes here
  sorry

end smallest_digit_divisible_by_nine_l2420_242081


namespace circle_center_sum_l2420_242050

/-- Given a circle with equation x^2 + y^2 = 6x + 18y - 63, 
    prove that the sum of the coordinates of its center is 12. -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 18*y - 63 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 6*h - 18*k + 63)) →
  h + k = 12 := by
sorry

end circle_center_sum_l2420_242050


namespace subset_count_with_nonempty_intersection_l2420_242029

theorem subset_count_with_nonempty_intersection :
  let A : Finset ℕ := Finset.range 10
  let B : Finset ℕ := {1, 2, 3, 4}
  (Finset.filter (fun C => (C ∩ B).Nonempty) (Finset.powerset A)).card = 960 := by
  sorry

end subset_count_with_nonempty_intersection_l2420_242029


namespace intersection_probability_l2420_242033

-- Define the probability measure q
variable (q : Set ℝ → ℝ)

-- Define events g and h
variable (g h : Set ℝ)

-- Define the conditions
variable (hg : q g = 0.30)
variable (hh : q h = 0.9)
variable (hgh : q (g ∩ h) / q h = 1 / 3)
variable (hhg : q (g ∩ h) / q g = 1 / 3)

-- The theorem to prove
theorem intersection_probability : q (g ∩ h) = 0.3 := by
  sorry

end intersection_probability_l2420_242033


namespace triangle_tiling_exists_quadrilateral_tiling_exists_hexagon_tiling_exists_l2420_242020

/-- A polygon in the plane -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- A tiling of the plane using a given polygon -/
def Tiling (p : Polygon) := 
  List (ℝ × ℝ) → Prop

/-- Predicate for a centrally symmetric hexagon -/
def IsCentrallySymmetricHexagon (p : Polygon) : Prop :=
  p.vertices.length = 6 ∧ 
  ∃ center : ℝ × ℝ, ∀ v ∈ p.vertices, 
    ∃ v' ∈ p.vertices, v' = (2 * center.1 - v.1, 2 * center.2 - v.2)

/-- Theorem stating that any triangle can tile the plane -/
theorem triangle_tiling_exists (t : Polygon) (h : t.vertices.length = 3) :
  ∃ tiling : Tiling t, True :=
sorry

/-- Theorem stating that any quadrilateral can tile the plane -/
theorem quadrilateral_tiling_exists (q : Polygon) (h : q.vertices.length = 4) :
  ∃ tiling : Tiling q, True :=
sorry

/-- Theorem stating that any centrally symmetric hexagon can tile the plane -/
theorem hexagon_tiling_exists (h : Polygon) (symmetric : IsCentrallySymmetricHexagon h) :
  ∃ tiling : Tiling h, True :=
sorry

end triangle_tiling_exists_quadrilateral_tiling_exists_hexagon_tiling_exists_l2420_242020


namespace train_crossing_time_l2420_242026

theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 330 →
  train_speed = 25 →
  man_speed = 2 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 44 := by
  sorry

end train_crossing_time_l2420_242026


namespace annie_aaron_visibility_time_l2420_242098

/-- The time (in minutes) Annie can see Aaron given their speeds and distances -/
theorem annie_aaron_visibility_time : 
  let annie_speed : ℝ := 10  -- Annie's speed in miles per hour
  let aaron_speed : ℝ := 6   -- Aaron's speed in miles per hour
  let initial_distance : ℝ := 1/4  -- Initial distance between Annie and Aaron in miles
  let final_distance : ℝ := 1/4   -- Final distance between Annie and Aaron in miles
  let relative_speed : ℝ := annie_speed - aaron_speed
  let time_hours : ℝ := (initial_distance + final_distance) / relative_speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 7.5
  := by sorry


end annie_aaron_visibility_time_l2420_242098


namespace inequality_solution_l2420_242088

theorem inequality_solution (a x : ℝ) : 
  a * x^2 - 2 ≥ 2 * x - a * x ↔ 
  (a = 0 ∧ x ≤ -1) ∨
  (a > 0 ∧ (x ≥ 2/a ∨ x ≤ -1)) ∨
  (-2 < a ∧ a < 0 ∧ 2/a ≤ x ∧ x ≤ -1) ∨
  (a = -2 ∧ x = -1) ∨
  (a < -2 ∧ -1 ≤ x ∧ x ≤ 2/a) := by
sorry

end inequality_solution_l2420_242088


namespace profit_calculation_l2420_242080

theorem profit_calculation :
  let selling_price : ℝ := 84
  let profit_percentage : ℝ := 0.4
  let loss_percentage : ℝ := 0.2
  let cost_price_profit_item : ℝ := selling_price / (1 + profit_percentage)
  let cost_price_loss_item : ℝ := selling_price / (1 - loss_percentage)
  let total_cost : ℝ := cost_price_profit_item + cost_price_loss_item
  let total_revenue : ℝ := 2 * selling_price
  total_revenue - total_cost = 3 := by
sorry

end profit_calculation_l2420_242080


namespace range_of_m_l2420_242013

/-- The range of m satisfying the given conditions -/
def M : Set ℝ := { m | ∀ x ∈ Set.Icc 0 1, 2 * m - 1 < x * (m^2 - 1) }

/-- Theorem stating that M is equal to the open interval (-∞, 0) -/
theorem range_of_m : M = Set.Ioi 0 := by sorry

end range_of_m_l2420_242013


namespace normal_distribution_probability_l2420_242031

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- The probability that a normal random variable is less than a given value -/
def prob_less_than (ξ : NormalRandomVariable) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is between two given values -/
def prob_between (ξ : NormalRandomVariable) (a b : ℝ) : ℝ := sorry

/-- Theorem: For a normal random variable ξ with mean 40, 
    if P(ξ < 30) = 0.2, then P(30 < ξ < 50) = 0.6 -/
theorem normal_distribution_probability 
  (ξ : NormalRandomVariable) 
  (h_mean : ξ.μ = 40) 
  (h_prob : prob_less_than ξ 30 = 0.2) : 
  prob_between ξ 30 50 = 0.6 := by sorry

end normal_distribution_probability_l2420_242031


namespace scientific_notation_of_216000_l2420_242015

theorem scientific_notation_of_216000 :
  (216000 : ℝ) = 2.16 * (10 ^ 5) := by
  sorry

end scientific_notation_of_216000_l2420_242015


namespace fraction_addition_simplification_l2420_242045

theorem fraction_addition_simplification :
  (1 : ℚ) / 462 + 23 / 42 = 127 / 231 := by sorry

end fraction_addition_simplification_l2420_242045


namespace modular_inverse_17_mod_800_l2420_242038

theorem modular_inverse_17_mod_800 : ∃ x : ℕ, x < 800 ∧ (17 * x) % 800 = 1 :=
by
  use 753
  sorry

end modular_inverse_17_mod_800_l2420_242038


namespace buffer_solution_calculation_l2420_242058

theorem buffer_solution_calculation (initial_volume_A initial_volume_B total_volume_needed : ℚ) :
  initial_volume_A = 0.05 →
  initial_volume_B = 0.03 →
  initial_volume_A + initial_volume_B = 0.08 →
  total_volume_needed = 0.64 →
  (total_volume_needed * (initial_volume_B / (initial_volume_A + initial_volume_B))) = 0.24 := by
sorry

end buffer_solution_calculation_l2420_242058


namespace constant_term_expansion_l2420_242041

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the constant term
def constantTerm (a b : ℕ) : ℕ :=
  binomial 8 3 * (5 ^ 5) * (2 ^ 3)

-- Theorem statement
theorem constant_term_expansion :
  constantTerm 5 2 = 1400000 := by sorry

end constant_term_expansion_l2420_242041


namespace coal_transport_trucks_l2420_242016

/-- The number of trucks needed to transport a given amount of coal -/
def trucks_needed (total_coal : ℕ) (truck_capacity : ℕ) : ℕ :=
  (total_coal + truck_capacity - 1) / truck_capacity

/-- Proof that 19 trucks are needed to transport 47,500 kg of coal when each truck can carry 2,500 kg -/
theorem coal_transport_trucks : trucks_needed 47500 2500 = 19 := by
  sorry

end coal_transport_trucks_l2420_242016


namespace ratio_first_term_to_common_difference_l2420_242046

/-- An arithmetic progression where the sum of the first twenty terms
    is five times the sum of the first ten terms -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference
  sum_condition : (20 * a + 190 * d) = 5 * (10 * a + 45 * d)

/-- The ratio of the first term to the common difference is -7/6 -/
theorem ratio_first_term_to_common_difference
  (ap : ArithmeticProgression) : ap.a / ap.d = -7 / 6 := by
  sorry

end ratio_first_term_to_common_difference_l2420_242046


namespace chocolate_topping_proof_l2420_242070

/-- Proves that adding 220 ounces of pure chocolate to the initial mixture 
    results in a 75% chocolate topping -/
theorem chocolate_topping_proof 
  (initial_total : ℝ) 
  (initial_chocolate : ℝ) 
  (initial_other : ℝ) 
  (target_percentage : ℝ) 
  (h1 : initial_total = 220)
  (h2 : initial_chocolate = 110)
  (h3 : initial_other = 110)
  (h4 : initial_total = initial_chocolate + initial_other)
  (h5 : target_percentage = 0.75) : 
  let added_chocolate : ℝ := 220
  let final_chocolate : ℝ := initial_chocolate + added_chocolate
  let final_total : ℝ := initial_total + added_chocolate
  final_chocolate / final_total = target_percentage :=
by sorry

end chocolate_topping_proof_l2420_242070


namespace parallel_segments_length_l2420_242083

/-- Given a quadrilateral ABYZ where AB is parallel to YZ, this theorem proves
    that if AZ = 54, BQ = 18, and QY = 36, then QZ = 36. -/
theorem parallel_segments_length (A B Y Z Q : ℝ × ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ B - A = k • (Z - Y)) →  -- AB parallel to YZ
  dist A Z = 54 →
  dist B Q = 18 →
  dist Q Y = 36 →
  dist Q Z = 36 := by
sorry


end parallel_segments_length_l2420_242083


namespace absolute_value_equation_solution_l2420_242044

theorem absolute_value_equation_solution :
  ∃! y : ℝ, (|y - 4| + 3 * y = 14) :=
by
  -- The unique solution is y = 4.5
  use 4.5
  sorry

end absolute_value_equation_solution_l2420_242044


namespace honey_percentage_l2420_242011

theorem honey_percentage (initial_honey : ℝ) (final_honey : ℝ) (repetitions : ℕ) 
  (h_initial : initial_honey = 1250)
  (h_final : final_honey = 512)
  (h_repetitions : repetitions = 4) :
  ∃ (percentage : ℝ), 
    percentage = 0.2 ∧ 
    final_honey = initial_honey * (1 - percentage) ^ repetitions :=
by sorry

end honey_percentage_l2420_242011


namespace lcm_of_ratio_two_three_l2420_242052

/-- Given two numbers a and b in the ratio 2:3, where a = 40 and b = 60, prove that their LCM is 60. -/
theorem lcm_of_ratio_two_three (a b : ℕ) (h1 : a = 40) (h2 : b = 60) (h3 : 3 * a = 2 * b) :
  Nat.lcm a b = 60 := by
  sorry

end lcm_of_ratio_two_three_l2420_242052


namespace max_value_x_cubed_over_y_fourth_l2420_242078

theorem max_value_x_cubed_over_y_fourth (x y : ℝ) 
  (h1 : 3 ≤ x * y^2 ∧ x * y^2 ≤ 8) 
  (h2 : 4 ≤ x^2 / y ∧ x^2 / y ≤ 9) : 
  x^3 / y^4 ≤ 27 := by
sorry

end max_value_x_cubed_over_y_fourth_l2420_242078


namespace floor_neg_seven_fourths_l2420_242082

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end floor_neg_seven_fourths_l2420_242082


namespace weather_ratings_theorem_l2420_242099

-- Define the weather observation
structure WeatherObservation :=
  (morning : Bool)
  (afternoon : Bool)
  (evening : Bool)

-- Define the rating system for each child
def firstChildRating (w : WeatherObservation) : Bool :=
  ¬(w.morning ∨ w.afternoon ∨ w.evening)

def secondChildRating (w : WeatherObservation) : Bool :=
  ¬w.morning ∨ ¬w.afternoon ∨ ¬w.evening

-- Define the combined rating
def combinedRating (w : WeatherObservation) : Bool × Bool :=
  (firstChildRating w, secondChildRating w)

-- Define the set of all possible weather observations
def allWeatherObservations : Set WeatherObservation :=
  {w | w.morning = true ∨ w.morning = false ∧
       w.afternoon = true ∨ w.afternoon = false ∧
       w.evening = true ∨ w.evening = false}

-- Theorem statement
theorem weather_ratings_theorem :
  {(true, true), (true, false), (false, true), (false, false)} =
  {r | ∃ w ∈ allWeatherObservations, combinedRating w = r} :=
by sorry

end weather_ratings_theorem_l2420_242099


namespace fewer_vip_tickets_l2420_242077

/-- Represents the number of tickets sold in a snooker tournament -/
structure TicketSales where
  vip : ℕ
  general : ℕ

/-- The ticket prices and sales data for the snooker tournament -/
def snookerTournament : TicketSales → Prop := fun ts =>
  ts.vip + ts.general = 320 ∧
  40 * ts.vip + 10 * ts.general = 7500

theorem fewer_vip_tickets (ts : TicketSales) 
  (h : snookerTournament ts) : ts.general - ts.vip = 34 := by
  sorry

end fewer_vip_tickets_l2420_242077


namespace cost_difference_l2420_242079

/-- Represents the pricing policy of the store -/
def pencil_cost (quantity : ℕ) : ℚ :=
  if quantity < 40 then 4 else (7/2)

/-- Calculate the total cost for a given quantity of pencils -/
def total_cost (quantity : ℕ) : ℚ :=
  (pencil_cost quantity) * quantity

/-- The number of pencils Joy bought -/
def joy_pencils : ℕ := 30

/-- The number of pencils Colleen bought -/
def colleen_pencils : ℕ := 50

/-- Theorem stating the difference in cost between Colleen's and Joy's purchases -/
theorem cost_difference : 
  total_cost colleen_pencils - total_cost joy_pencils = 55 := by
  sorry

end cost_difference_l2420_242079


namespace translated_linear_function_range_l2420_242084

theorem translated_linear_function_range (x : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x + 2
  f x > 0 → x > -2 := by
sorry

end translated_linear_function_range_l2420_242084


namespace car_travel_problem_l2420_242064

/-- Represents a car's travel information -/
structure CarTravel where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem statement -/
theorem car_travel_problem 
  (p : CarTravel) 
  (q : CarTravel) 
  (h1 : p.time = 3) 
  (h2 : p.speed = 60) 
  (h3 : q.speed = 3 * p.speed) 
  (h4 : q.distance = p.distance / 2) 
  (h5 : p.distance = p.speed * p.time) 
  (h6 : q.distance = q.speed * q.time) : 
  q.time = 0.5 := by
sorry

end car_travel_problem_l2420_242064


namespace airplane_cost_l2420_242037

def initial_amount : ℚ := 5.00
def change_received : ℚ := 0.72

theorem airplane_cost : initial_amount - change_received = 4.28 := by
  sorry

end airplane_cost_l2420_242037


namespace rectangular_box_volume_l2420_242019

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 40)
  (area2 : w * h = 10)
  (area3 : l * h = 8) :
  l * w * h = 40 * Real.sqrt 2 := by
sorry

end rectangular_box_volume_l2420_242019


namespace min_value_of_sum_squares_l2420_242056

theorem min_value_of_sum_squares (x y z : ℝ) (h : x + y + z = 2) :
  x^2 + 2*y^2 + z^2 ≥ 4/3 ∧ 
  ∃ (a b c : ℝ), a + b + c = 2 ∧ a^2 + 2*b^2 + c^2 = 4/3 :=
by sorry

end min_value_of_sum_squares_l2420_242056


namespace square_minus_self_divisible_by_two_l2420_242096

theorem square_minus_self_divisible_by_two (n : ℕ) : 
  2 ∣ (n^2 - n) := by sorry

end square_minus_self_divisible_by_two_l2420_242096


namespace inequality_system_solution_range_l2420_242091

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x - a + 1 ≥ 0 ∧ 3 - 2*x > 0))) → 
  -1 < a ∧ a ≤ 0 := by
sorry

end inequality_system_solution_range_l2420_242091


namespace total_gas_consumption_is_18_gallons_l2420_242054

/-- Represents the number of cuts for a lawn in a given month. -/
structure MonthlyCuts where
  regular : Nat  -- Number of cuts in regular months
  peak : Nat     -- Number of cuts in peak months

/-- Represents the gas consumption pattern for a lawn. -/
structure GasConsumption where
  gallons : Nat  -- Number of gallons consumed
  frequency : Nat  -- Frequency of consumption (every nth cut)

/-- Calculates the total number of cuts for a lawn over the season. -/
def totalCuts (cuts : MonthlyCuts) : Nat :=
  4 * cuts.regular + 4 * cuts.peak

/-- Calculates the gas consumed for a lawn over the season. -/
def gasConsumed (cuts : Nat) (consumption : GasConsumption) : Nat :=
  (cuts / consumption.frequency) * consumption.gallons

/-- Theorem stating that the total gas consumption is 18 gallons. -/
theorem total_gas_consumption_is_18_gallons 
  (large_lawn_cuts : MonthlyCuts)
  (small_lawn_cuts : MonthlyCuts)
  (large_lawn_gas : GasConsumption)
  (small_lawn_gas : GasConsumption)
  (h1 : large_lawn_cuts = { regular := 1, peak := 3 })
  (h2 : small_lawn_cuts = { regular := 2, peak := 2 })
  (h3 : large_lawn_gas = { gallons := 2, frequency := 3 })
  (h4 : small_lawn_gas = { gallons := 1, frequency := 2 })
  : gasConsumed (totalCuts large_lawn_cuts) large_lawn_gas + 
    gasConsumed (totalCuts small_lawn_cuts) small_lawn_gas = 18 := by
  sorry

end total_gas_consumption_is_18_gallons_l2420_242054


namespace algebraic_simplification_l2420_242048

theorem algebraic_simplification (x y : ℝ) :
  (18 * x^3 * y) * (8 * x * y^2) * (1 / (6 * x * y)^2) = 4 * x * y := by
  sorry

end algebraic_simplification_l2420_242048


namespace sweets_distribution_l2420_242032

theorem sweets_distribution (num_children : ℕ) (sweets_per_child : ℕ) (remaining_fraction : ℚ) :
  num_children = 48 →
  sweets_per_child = 4 →
  remaining_fraction = 1/3 →
  (num_children * sweets_per_child) / (1 - remaining_fraction) = 288 := by
  sorry

end sweets_distribution_l2420_242032


namespace inverse_proportion_relationship_l2420_242049

theorem inverse_proportion_relationship (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ < x₂ → x₂ < 0 → y₁ = 2 / x₁ → y₂ = 2 / x₂ → y₂ < y₁ ∧ y₁ < 0 := by
  sorry

end inverse_proportion_relationship_l2420_242049


namespace video_game_lives_l2420_242057

theorem video_game_lives (initial_players : ℕ) (players_quit : ℕ) (total_lives : ℕ) :
  initial_players = 10 →
  players_quit = 7 →
  total_lives = 24 →
  (total_lives / (initial_players - players_quit) : ℚ) = 8 :=
by
  sorry

end video_game_lives_l2420_242057


namespace exist_unequal_triangles_with_equal_angles_and_two_sides_l2420_242063

-- Define two triangles
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)

-- Define the conditions for our triangles
def triangles_satisfy_conditions (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ ∧
  ((t1.a = t2.a ∧ t1.b = t2.b) ∨ (t1.a = t2.a ∧ t1.c = t2.c) ∨ (t1.b = t2.b ∧ t1.c = t2.c))

-- Define triangle inequality
def triangles_not_congruent (t1 t2 : Triangle) : Prop :=
  t1.a ≠ t2.a ∨ t1.b ≠ t2.b ∨ t1.c ≠ t2.c

-- Theorem statement
theorem exist_unequal_triangles_with_equal_angles_and_two_sides :
  ∃ (t1 t2 : Triangle), triangles_satisfy_conditions t1 t2 ∧ triangles_not_congruent t1 t2 :=
sorry

end exist_unequal_triangles_with_equal_angles_and_two_sides_l2420_242063


namespace cistern_width_is_six_l2420_242065

/-- Represents a rectangular cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the total wet surface area of the cistern --/
def wetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem: Given the dimensions and wet surface area, the width of the cistern is 6 meters --/
theorem cistern_width_is_six (c : Cistern) 
    (h_length : c.length = 8)
    (h_depth : c.depth = 1.25)
    (h_area : wetSurfaceArea c = 83) : 
  c.width = 6 := by
  sorry

end cistern_width_is_six_l2420_242065


namespace locus_is_circle_l2420_242085

/-- An isosceles triangle with side length s and base b -/
structure IsoscelesTriangle where
  s : ℝ
  b : ℝ
  s_pos : 0 < s
  b_pos : 0 < b
  triangle_ineq : b < 2 * s

/-- The locus of points P such that the sum of distances from P to the vertices equals a -/
def Locus (t : IsoscelesTriangle) (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧
    Real.sqrt (x^2 + y^2) +
    Real.sqrt ((x - t.b)^2 + y^2) +
    Real.sqrt ((x - t.b/2)^2 + (y - Real.sqrt (t.s^2 - (t.b/2)^2))^2) = a}

/-- The theorem stating that the locus is a circle if and only if a > 2s + b -/
theorem locus_is_circle (t : IsoscelesTriangle) (a : ℝ) :
  (∃ (c : ℝ × ℝ) (r : ℝ), Locus t a = {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}) ↔
  a > 2 * t.s + t.b := by
  sorry

end locus_is_circle_l2420_242085


namespace problem_statement_l2420_242042

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a / 2 * x^2

def l (k : ℤ) (x : ℝ) : ℝ := (k - 2 : ℝ) * x - k + 1

theorem problem_statement :
  (∀ a : ℝ, (∃ x₀ : ℝ, x₀ ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ f a x₀ > 0) →
    a < 2 / Real.exp 1) ∧
  (∃ k : ℤ, k = 4 ∧
    ∀ k' : ℤ, (∀ x : ℝ, x > 1 → f 0 x > l k' x) → k' ≤ k) :=
by sorry

end problem_statement_l2420_242042


namespace lost_to_initial_ratio_l2420_242062

/-- Represents the number of black socks Andy initially had -/
def initial_black_socks : ℕ := 6

/-- Represents the number of white socks Andy initially had -/
def initial_white_socks : ℕ := 4 * initial_black_socks

/-- Represents the number of white socks Andy has after losing some -/
def remaining_white_socks : ℕ := initial_black_socks + 6

/-- Represents the number of white socks Andy lost -/
def lost_white_socks : ℕ := initial_white_socks - remaining_white_socks

/-- Theorem stating that the ratio of lost white socks to initial white socks is 1/2 -/
theorem lost_to_initial_ratio :
  (lost_white_socks : ℚ) / initial_white_socks = 1 / 2 := by sorry

end lost_to_initial_ratio_l2420_242062


namespace crayons_count_l2420_242095

/-- The number of rows of crayons --/
def num_rows : ℕ := 7

/-- The number of crayons in each row --/
def crayons_per_row : ℕ := 30

/-- The total number of crayons --/
def total_crayons : ℕ := num_rows * crayons_per_row

theorem crayons_count : total_crayons = 210 := by
  sorry

end crayons_count_l2420_242095


namespace problem_solution_roots_product_l2420_242073

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.log x
def g (m : ℝ) (x : ℝ) : ℝ := x + m

-- Define the function F
def F (m : ℝ) (x : ℝ) : ℝ := f x - g m x

theorem problem_solution (m : ℝ) :
  (∀ x > 0, f x ≤ g m x) ↔ m ≥ -1 :=
sorry

theorem roots_product (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  F m x₁ = 0 →
  F m x₂ = 0 →
  x₁ * x₂ < 1 :=
sorry

end problem_solution_roots_product_l2420_242073


namespace rectangular_plot_length_difference_l2420_242053

/-- Proves that for a rectangular plot with given conditions, the length is 20 metres more than the breadth -/
theorem rectangular_plot_length_difference (length width : ℝ) : 
  length = 60 ∧ 
  (2 * length + 2 * width) * 26.5 = 5300 →
  length - width = 20 := by
  sorry

end rectangular_plot_length_difference_l2420_242053


namespace bleacher_exercise_calories_l2420_242005

/-- Given the number of round trips, stairs one way, and total calories burned,
    calculate the number of calories burned per stair. -/
def calories_per_stair (round_trips : ℕ) (stairs_one_way : ℕ) (total_calories : ℕ) : ℚ :=
  total_calories / (2 * round_trips * stairs_one_way)

/-- Theorem stating that under the given conditions, each stair burns 2 calories. -/
theorem bleacher_exercise_calories :
  calories_per_stair 40 32 5120 = 2 := by
  sorry

end bleacher_exercise_calories_l2420_242005


namespace oil_price_reduction_l2420_242089

/-- Calculates the percentage reduction in oil price given the original amount, additional amount, total cost, and reduced price. -/
theorem oil_price_reduction 
  (X : ℝ)              -- Original amount of oil in kg
  (additional : ℝ)     -- Additional amount of oil in kg
  (total_cost : ℝ)     -- Total cost in Rs
  (reduced_price : ℝ)  -- Reduced price per kg in Rs
  (h1 : additional = 5)
  (h2 : total_cost = 600)
  (h3 : reduced_price = 30)
  (h4 : X + additional = total_cost / reduced_price)
  (h5 : X = total_cost / (total_cost / X))
  : (1 - reduced_price / (total_cost / X)) * 100 = 25 := by
  sorry


end oil_price_reduction_l2420_242089


namespace units_digit_factorial_sum_plus_7_l2420_242009

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_plus_7 :
  units_digit (factorial_sum 25 + 7) = 0 := by sorry

end units_digit_factorial_sum_plus_7_l2420_242009


namespace divisibility_implies_divisibility_l2420_242092

theorem divisibility_implies_divisibility (a b m n : ℕ) 
  (ha : a > 1) (hcoprime : Nat.Coprime a b) :
  (((a^m + 1) ∣ (a^n + 1)) → (m ∣ n)) ∧
  (((a^m + b^m) ∣ (a^n + b^n)) → (m ∣ n)) := by
  sorry

end divisibility_implies_divisibility_l2420_242092


namespace equation_solution_l2420_242047

theorem equation_solution : 
  ∃ n : ℚ, (3 - n) / (n + 2) + (3 * n - 9) / (3 - n) = 2 ∧ n = -7/6 := by
  sorry

end equation_solution_l2420_242047


namespace product_sum_theorem_l2420_242027

theorem product_sum_theorem (a b c d : ℝ) 
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 6)
  (eq3 : a + c + d = 15)
  (eq4 : b + c + d = 10) :
  a * b + c * d = 408 / 9 := by
sorry

end product_sum_theorem_l2420_242027


namespace solution_existence_l2420_242087

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The system of equations has a solution -/
def has_solution (K : ℤ) : Prop :=
  ∃ (x y : ℝ), (2 * (floor x) + y = 3/2) ∧ ((floor x - x)^2 - 2 * (floor y) = K)

/-- The theorem stating the conditions for the existence of a solution -/
theorem solution_existence (K : ℤ) :
  has_solution K ↔ ∃ (M : ℤ), K = 4*M - 2 ∧ has_solution (4*M - 2) :=
sorry

end solution_existence_l2420_242087


namespace stating_valid_orderings_count_l2420_242010

/-- 
Given a positive integer n, this function returns the number of ways to order 
integers from 1 to n, where except for the first integer, every integer differs 
by 1 from some integer to its left.
-/
def validOrderings (n : ℕ) : ℕ :=
  2^(n-1)

/-- 
Theorem stating that the number of valid orderings of integers from 1 to n 
is equal to 2^(n-1), where a valid ordering is one in which, except for the 
first integer, every integer differs by 1 from some integer to its left.
-/
theorem valid_orderings_count (n : ℕ) (h : n > 0) : 
  validOrderings n = 2^(n-1) := by
  sorry

end stating_valid_orderings_count_l2420_242010


namespace complex_equation_solutions_l2420_242039

theorem complex_equation_solutions :
  ∃! (s : Finset ℂ), s.card = 4 ∧
  (∀ c ∈ s, ∃ u v w : ℂ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    ∀ z : ℂ, (z - u) * (z - v) * (z - w) = (z - c*u) * (z - c*v) * (z - c*w)) ∧
  (∀ c : ℂ, c ∉ s →
    ¬∃ u v w : ℂ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    ∀ z : ℂ, (z - u) * (z - v) * (z - w) = (z - c*u) * (z - c*v) * (z - c*w)) :=
by
  sorry

end complex_equation_solutions_l2420_242039


namespace systems_solutions_l2420_242023

-- Define the systems of equations
def system1 (x y : ℝ) : Prop :=
  y = 2 * x - 3 ∧ 3 * x + 2 * y = 8

def system2 (x y : ℝ) : Prop :=
  x + 2 * y = 3 ∧ 2 * x - 4 * y = -10

-- State the theorem
theorem systems_solutions :
  (∃ x y : ℝ, system1 x y ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℝ, system2 x y ∧ x = -1 ∧ y = 2) :=
sorry

end systems_solutions_l2420_242023


namespace weight_of_new_person_l2420_242034

/-- The weight of the new person when the average weight of a group increases -/
def new_person_weight (num_people : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + num_people * avg_increase

/-- Theorem stating the weight of the new person under given conditions -/
theorem weight_of_new_person :
  new_person_weight 12 4 65 = 113 := by
  sorry

end weight_of_new_person_l2420_242034


namespace geometric_sequence_ratio_l2420_242014

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem -/
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_prod : a 7 * a 11 = 6)
  (h_sum : a 4 + a 14 = 5) :
  a 20 / a 10 = 2/3 ∨ a 20 / a 10 = 3/2 :=
sorry

end geometric_sequence_ratio_l2420_242014


namespace angle_equality_l2420_242075

theorem angle_equality (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.cos α + Real.cos (2*β) - Real.cos (α + β) = 3/2 →
  α = π/3 ∧ β = π/3 := by
  sorry

end angle_equality_l2420_242075


namespace min_selling_price_A_l2420_242012

/-- Represents the water purifier problem with given conditions -/
structure WaterPurifierProblem where
  total_units : ℕ
  price_A : ℕ
  price_B : ℕ
  total_cost : ℕ
  units_A : ℕ
  units_B : ℕ
  min_total_profit : ℕ

/-- The specific instance of the water purifier problem -/
def problem : WaterPurifierProblem := {
  total_units := 160,
  price_A := 150,
  price_B := 350,
  total_cost := 36000,
  units_A := 100,
  units_B := 60,
  min_total_profit := 11000
}

/-- Theorem stating the minimum selling price for model A -/
theorem min_selling_price_A (p : WaterPurifierProblem) : 
  p.total_units = p.units_A + p.units_B →
  p.total_cost = p.price_A * p.units_A + p.price_B * p.units_B →
  ∀ selling_price_A : ℕ, 
    (selling_price_A - p.price_A) * p.units_A + 
    (2 * (selling_price_A - p.price_A)) * p.units_B ≥ p.min_total_profit →
    selling_price_A ≥ 200 := by
  sorry

#check min_selling_price_A problem

end min_selling_price_A_l2420_242012


namespace solution_relationship_l2420_242018

theorem solution_relationship (x y : ℝ) : 
  2 * x + y = 7 → x - y = 5 → x + 2 * y = 2 := by
  sorry

end solution_relationship_l2420_242018


namespace line_slope_intercept_sum_l2420_242002

/-- Given a line passing through points (1,3) and (3,11), 
    prove that the sum of its slope and y-intercept equals 3. -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
  (3 : ℝ) = m * (1 : ℝ) + b → 
  (11 : ℝ) = m * (3 : ℝ) + b → 
  m + b = 3 := by
sorry

end line_slope_intercept_sum_l2420_242002


namespace not_p_or_not_q_true_l2420_242030

theorem not_p_or_not_q_true (p q : Prop) (h : ¬(p ∧ q)) : (¬p) ∨ (¬q) := by
  sorry

end not_p_or_not_q_true_l2420_242030


namespace remainder_problem_l2420_242003

theorem remainder_problem : 123456789012 % 240 = 132 := by
  sorry

end remainder_problem_l2420_242003


namespace puzzle_completion_percentage_l2420_242028

theorem puzzle_completion_percentage (total_pieces : ℕ) 
  (day1_percentage day2_percentage : ℚ) (pieces_left : ℕ) : 
  total_pieces = 1000 →
  day1_percentage = 1/10 →
  day2_percentage = 1/5 →
  pieces_left = 504 →
  let pieces_after_day1 := total_pieces - (total_pieces * day1_percentage).num
  let pieces_after_day2 := pieces_after_day1 - (pieces_after_day1 * day2_percentage).num
  let pieces_completed_day3 := pieces_after_day2 - pieces_left
  (pieces_completed_day3 : ℚ) / pieces_after_day2 = 3/10 := by sorry

end puzzle_completion_percentage_l2420_242028


namespace correct_delivery_probability_l2420_242008

def num_houses : ℕ := 5

def num_correct_deliveries : ℕ := 3

def probability_correct_deliveries : ℚ :=
  (num_houses.choose num_correct_deliveries : ℚ) / (num_houses.factorial : ℚ)

theorem correct_delivery_probability :
  probability_correct_deliveries = 1 / 12 := by
  sorry

end correct_delivery_probability_l2420_242008


namespace shells_equation_initial_shells_value_l2420_242071

/-- The number of shells Lucy initially put in her bucket -/
def initial_shells : ℕ := sorry

/-- The number of additional shells Lucy found -/
def additional_shells : ℕ := 21

/-- The total number of shells Lucy has now -/
def total_shells : ℕ := 89

/-- Theorem stating that the initial number of shells plus the additional shells equals the total shells -/
theorem shells_equation : initial_shells + additional_shells = total_shells := by sorry

/-- Theorem proving that the initial number of shells is 68 -/
theorem initial_shells_value : initial_shells = 68 := by sorry

end shells_equation_initial_shells_value_l2420_242071


namespace cid_car_wash_count_l2420_242035

theorem cid_car_wash_count :
  let oil_change_price : ℕ := 20
  let repair_price : ℕ := 30
  let car_wash_price : ℕ := 5
  let oil_change_count : ℕ := 5
  let repair_count : ℕ := 10
  let total_earnings : ℕ := 475
  let car_wash_count : ℕ := (total_earnings - (oil_change_price * oil_change_count + repair_price * repair_count)) / car_wash_price
  car_wash_count = 15 :=
by sorry

end cid_car_wash_count_l2420_242035


namespace bob_corn_harvest_l2420_242004

/-- Calculates the number of bushels of corn harvested given the number of rows,
    corn stalks per row, and corn stalks per bushel. -/
def corn_harvest (rows : ℕ) (stalks_per_row : ℕ) (stalks_per_bushel : ℕ) : ℕ :=
  (rows * stalks_per_row) / stalks_per_bushel

/-- Proves that Bob will harvest 50 bushels of corn given the specified conditions. -/
theorem bob_corn_harvest :
  corn_harvest 5 80 8 = 50 := by
  sorry

end bob_corn_harvest_l2420_242004


namespace cubic_sum_over_product_l2420_242001

theorem cubic_sum_over_product (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a + b + c + d = 0) : 
  (a^3 + b^3 + c^3 + d^3) / (a * b * c * d) = -3 := by
  sorry

end cubic_sum_over_product_l2420_242001


namespace negation_of_universal_proposition_l2420_242067

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by sorry

end negation_of_universal_proposition_l2420_242067


namespace polynomial_remainder_l2420_242022

theorem polynomial_remainder (x : ℝ) : 
  (5*x^8 - x^7 + 3*x^6 - 5*x^4 + 6*x^3 - 7) % (3*x - 6) = 1305 := by
  sorry

end polynomial_remainder_l2420_242022


namespace percentage_decrease_after_increase_l2420_242059

theorem percentage_decrease_after_increase (x : ℝ) (hx : x > 0) :
  let y := x * 1.6
  y * (1 - 0.375) = x :=
by sorry

end percentage_decrease_after_increase_l2420_242059


namespace fermat_divisibility_l2420_242025

/-- Fermat number -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Theorem: For all natural numbers n, F_n divides 2^F_n - 2 -/
theorem fermat_divisibility (n : ℕ) : (F n) ∣ (2^(F n) - 2) := by
  sorry

end fermat_divisibility_l2420_242025


namespace probability_for_2x3x4_prism_l2420_242051

/-- Represents a rectangular prism with dimensions a, b, and c. -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The probability that a plane determined by three randomly selected distinct vertices
    of a rectangular prism contains points inside the prism. -/
def probability_plane_intersects_interior (prism : RectangularPrism) : ℚ :=
  4/7

/-- Theorem stating that for a 2x3x4 rectangular prism, the probability of a plane
    determined by three randomly selected distinct vertices containing points inside
    the prism is 4/7. -/
theorem probability_for_2x3x4_prism :
  let prism : RectangularPrism := ⟨2, 3, 4, by norm_num, by norm_num, by norm_num⟩
  probability_plane_intersects_interior prism = 4/7 := by
  sorry


end probability_for_2x3x4_prism_l2420_242051


namespace cubic_sum_theorem_l2420_242066

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_condition : a + b + c = 1)
  (product_sum_condition : a * b + a * c + b * c = -4)
  (product_condition : a * b * c = -4) : 
  a^3 + b^3 + c^3 = 1 := by sorry

end cubic_sum_theorem_l2420_242066


namespace square_division_impossibility_l2420_242017

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane --/
structure Square where
  side : ℝ
  center : Point

/-- Represents a division of a square --/
structure SquareDivision where
  square : Square
  point1 : Point
  point2 : Point

/-- Checks if a point is inside a square --/
def is_inside (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) < s.side / 2 ∧ abs (p.y - s.center.y) < s.side / 2

/-- The theorem stating the impossibility of the division --/
theorem square_division_impossibility (s : Square) : 
  ¬ ∃ (d : SquareDivision), 
    (is_inside d.point1 s) ∧ 
    (is_inside d.point2 s) ∧ 
    (∃ (areas : List ℝ), areas.length = 9 ∧ (∀ a ∈ areas, a > 0) ∧ areas.sum = s.side ^ 2) :=
sorry

end square_division_impossibility_l2420_242017


namespace like_terms_imply_zero_power_l2420_242000

theorem like_terms_imply_zero_power (n : ℕ) : 
  (∃ x y, -x^(2*n-1) * y = 3 * x^8 * y) → (2*n - 9)^2013 = 0 := by
  sorry

end like_terms_imply_zero_power_l2420_242000


namespace distance_implies_abs_x_l2420_242093

theorem distance_implies_abs_x (x : ℝ) :
  |((3 + x) - (3 - x))| = 8 → |x| = 4 := by
  sorry

end distance_implies_abs_x_l2420_242093


namespace unique_modular_congruence_l2420_242055

theorem unique_modular_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4702 [ZMOD 8] := by
  sorry

end unique_modular_congruence_l2420_242055


namespace emilys_spending_l2420_242061

/-- Emily's spending problem -/
theorem emilys_spending (X : ℝ) : 
  X + 2 * X + 3 * X = 120 → X = 20 := by
  sorry

end emilys_spending_l2420_242061


namespace product_w_z_l2420_242097

/-- A parallelogram with side lengths defined in terms of w and z -/
structure Parallelogram (w z : ℝ) :=
  (ef : ℝ)
  (fg : ℝ)
  (gh : ℝ)
  (he : ℝ)
  (ef_eq : ef = 50)
  (fg_eq : fg = 4 * z^2)
  (gh_eq : gh = 3 * w + 6)
  (he_eq : he = 32)
  (opposite_sides_equal : ef = gh ∧ fg = he)

/-- The product of w and z in the given parallelogram is 88√2/3 -/
theorem product_w_z (w z : ℝ) (p : Parallelogram w z) : w * z = 88 * Real.sqrt 2 / 3 := by
  sorry

end product_w_z_l2420_242097


namespace coles_return_speed_coles_return_speed_is_120_l2420_242021

/-- Calculates the average speed of the return trip given the conditions of Cole's journey --/
theorem coles_return_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work : ℝ) : ℝ :=
  let distance_to_work := speed_to_work * time_to_work
  let time_to_return := total_time - time_to_work
  distance_to_work / time_to_return

/-- Proves that Cole's average speed driving back home is 120 km/h --/
theorem coles_return_speed_is_120 :
  coles_return_speed 80 2 (72 / 60) = 120 := by
  sorry

end coles_return_speed_coles_return_speed_is_120_l2420_242021


namespace employee_pay_calculation_l2420_242094

/-- Given two employees with a total pay of 570 and one paid 150% of the other,
    prove that the lower-paid employee receives 228. -/
theorem employee_pay_calculation (total_pay : ℝ) (ratio : ℝ) :
  total_pay = 570 →
  ratio = 1.5 →
  ∃ (low_pay : ℝ), low_pay * (1 + ratio) = total_pay ∧ low_pay = 228 := by
  sorry

end employee_pay_calculation_l2420_242094


namespace travel_time_ratio_l2420_242086

/-- Proves that the ratio of the time taken to travel a fixed distance at a given speed
    to the time taken to travel the same distance in a given time is equal to a specific ratio. -/
theorem travel_time_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 360 ∧ original_time = 6 ∧ new_speed = 40 →
  (distance / new_speed) / original_time = 3 / 2 := by
  sorry

end travel_time_ratio_l2420_242086


namespace factorization_proof_l2420_242006

theorem factorization_proof :
  ∀ x : ℝ,
  (x^2 - x - 6 = (x + 2) * (x - 3)) ∧
  ¬(x^2 - 1 = x * (x - 1/x)) ∧
  ¬(7 * x^2 * y^5 = x * y * 7 * x * y^4) ∧
  ¬(x^2 + 4*x + 4 = x * (x + 4) + 4) :=
by
  sorry

end factorization_proof_l2420_242006


namespace solution_set_equivalence_k_range_l2420_242074

noncomputable section

-- Define the function f
def f (k x : ℝ) : ℝ := (k * x) / (x^2 + 3 * k)

-- Define the conditions
variable (k : ℝ)
variable (h_k_pos : k > 0)

-- Part 1
theorem solution_set_equivalence :
  (∃ m : ℝ, ∀ x : ℝ, f k x > m ↔ x < -3 ∨ x > -2) →
  ∃ m : ℝ, ∀ x : ℝ, 5 * m * x^2 + (k / 2) * x + 3 > 0 ↔ -1 < x ∧ x < 3/2 :=
sorry

-- Part 2
theorem k_range :
  (∃ x : ℝ, x > 3 ∧ f k x > 1) →
  k > 12 :=
sorry

end

end solution_set_equivalence_k_range_l2420_242074


namespace complement_equal_l2420_242069

/-- The complement of an angle is the angle that, when added to the original angle, results in a right angle (90 degrees). -/
def complement (α : ℝ) : ℝ := 90 - α

/-- For any angle, its complement is equal to itself. -/
theorem complement_equal (α : ℝ) : complement α = complement α := by sorry

end complement_equal_l2420_242069


namespace counterexample_exists_l2420_242068

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ Nat.Prime (n + 3) := by
  sorry

end counterexample_exists_l2420_242068
