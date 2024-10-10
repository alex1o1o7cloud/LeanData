import Mathlib

namespace arrangement_theorem_l1198_119803

/-- The number of people standing in a row -/
def n : ℕ := 7

/-- Calculate the number of ways person A and B can stand next to each other -/
def adjacent_AB : ℕ := sorry

/-- Calculate the number of ways person A and B can stand not next to each other -/
def not_adjacent_AB : ℕ := sorry

/-- Calculate the number of ways person A, B, and C can stand so that no two of them are next to each other -/
def no_two_adjacent_ABC : ℕ := sorry

/-- Calculate the number of ways person A, B, and C can stand so that at most two of them are not next to each other -/
def at_most_two_not_adjacent_ABC : ℕ := sorry

theorem arrangement_theorem :
  adjacent_AB = 1440 ∧
  not_adjacent_AB = 3600 ∧
  no_two_adjacent_ABC = 1440 ∧
  at_most_two_not_adjacent_ABC = 4320 := by sorry

end arrangement_theorem_l1198_119803


namespace complex_magnitude_constraint_l1198_119806

theorem complex_magnitude_constraint (a : ℝ) :
  let z : ℂ := 1 + a * I
  (Complex.abs z < 2) → (-Real.sqrt 3 < a ∧ a < Real.sqrt 3) := by
sorry

end complex_magnitude_constraint_l1198_119806


namespace prime_between_30_and_40_with_specific_remainder_l1198_119828

theorem prime_between_30_and_40_with_specific_remainder : 
  {n : ℕ | 30 ≤ n ∧ n ≤ 40 ∧ Prime n ∧ 1 ≤ n % 7 ∧ n % 7 ≤ 6} = {31, 37} := by
  sorry

end prime_between_30_and_40_with_specific_remainder_l1198_119828


namespace fraction_equality_l1198_119848

theorem fraction_equality (p q r u v w : ℝ) 
  (h_positive : p > 0 ∧ q > 0 ∧ r > 0 ∧ u > 0 ∧ v > 0 ∧ w > 0)
  (h_sum_squares1 : p^2 + q^2 + r^2 = 49)
  (h_sum_squares2 : u^2 + v^2 + w^2 = 64)
  (h_dot_product : p*u + q*v + r*w = 56)
  (h_p_2q : p = 2*q) :
  (p + q + r) / (u + v + w) = 7/8 := by
  sorry

end fraction_equality_l1198_119848


namespace temperature_matches_data_temperature_decreases_with_altitude_constant_temperature_change_rate_l1198_119824

-- Define the relationship between altitude and temperature
def temperature (h : ℝ) : ℝ := 20 - 6 * h

-- Define the set of data points from the table
def data_points : List (ℝ × ℝ) := [
  (0, 20), (1, 14), (2, 8), (3, 2), (4, -4), (5, -10)
]

-- Theorem stating that the temperature function matches the data points
theorem temperature_matches_data : ∀ (point : ℝ × ℝ), 
  point ∈ data_points → temperature point.1 = point.2 := by
  sorry

-- Theorem stating that the temperature decreases as altitude increases
theorem temperature_decreases_with_altitude : 
  ∀ (h1 h2 : ℝ), h1 < h2 → temperature h1 > temperature h2 := by
  sorry

-- Theorem stating that the rate of temperature change is constant
theorem constant_temperature_change_rate : 
  ∀ (h1 h2 : ℝ), h1 ≠ h2 → (temperature h2 - temperature h1) / (h2 - h1) = -6 := by
  sorry

end temperature_matches_data_temperature_decreases_with_altitude_constant_temperature_change_rate_l1198_119824


namespace smallest_tangent_circle_l1198_119883

-- Define the given line
def given_line (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the given circle
def given_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 2

-- Define the solution circle
def solution_circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- Define a general circle with center (a, b) and radius r
def general_circle (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem smallest_tangent_circle :
  ∃! (a b r : ℝ),
    (∀ x y, general_circle x y a b r → 
      (∃ x₀ y₀, given_line x₀ y₀ ∧ general_circle x₀ y₀ a b r) ∧
      (∃ x₁ y₁, given_circle x₁ y₁ ∧ general_circle x₁ y₁ a b r)) ∧
    (∀ a' b' r', 
      (∀ x y, general_circle x y a' b' r' → 
        (∃ x₀ y₀, given_line x₀ y₀ ∧ general_circle x₀ y₀ a' b' r') ∧
        (∃ x₁ y₁, given_circle x₁ y₁ ∧ general_circle x₁ y₁ a' b' r')) →
      r ≤ r') ∧
    (∀ x y, general_circle x y a b r ↔ solution_circle x y) :=
sorry

end smallest_tangent_circle_l1198_119883


namespace projection_onto_yOz_l1198_119819

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the projection of a point onto the yOz plane -/
def projectToYOZ (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

/-- Theorem stating that the projection of (1, -2, 3) onto the yOz plane is (-1, -2, 3) -/
theorem projection_onto_yOz :
  let p := Point3D.mk 1 (-2) 3
  projectToYOZ p = Point3D.mk (-1) (-2) 3 := by
  sorry

end projection_onto_yOz_l1198_119819


namespace equation_root_implies_a_value_l1198_119809

theorem equation_root_implies_a_value (x a : ℝ) : 
  ((x - 2) / (x + 4) = a / (x + 4)) → (∃ x, (x - 2) / (x + 4) = a / (x + 4)) → a = -6 :=
by sorry

end equation_root_implies_a_value_l1198_119809


namespace point_in_third_quadrant_l1198_119865

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Definition of the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If A(m,n) is in the first quadrant, then B(-m,-n) is in the third quadrant -/
theorem point_in_third_quadrant
  (A : Point)
  (hA : isInFirstQuadrant A) :
  isInThirdQuadrant (Point.mk (-A.x) (-A.y)) :=
by sorry

end point_in_third_quadrant_l1198_119865


namespace cousins_distribution_l1198_119888

/-- The number of ways to distribute n indistinguishable objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 4 rooms -/
def num_rooms : ℕ := 4

/-- There are 5 cousins -/
def num_cousins : ℕ := 5

/-- The number of ways to distribute the cousins into the rooms -/
def num_distributions : ℕ := distribute num_cousins num_rooms

theorem cousins_distribution : num_distributions = 51 := by sorry

end cousins_distribution_l1198_119888


namespace cubic_root_values_l1198_119859

theorem cubic_root_values (m : ℝ) : 
  (-1 : ℂ)^3 - (m^2 - m + 7)*(-1 : ℂ) - (3*m^2 - 3*m - 6) = 0 ↔ m = -2 ∨ m = 3 := by
  sorry

end cubic_root_values_l1198_119859


namespace yoongi_age_yoongi_age_when_namjoon_is_six_l1198_119825

theorem yoongi_age (namjoon_age : ℕ) (age_difference : ℕ) : ℕ :=
  namjoon_age - age_difference

theorem yoongi_age_when_namjoon_is_six :
  yoongi_age 6 3 = 3 := by
  sorry

end yoongi_age_yoongi_age_when_namjoon_is_six_l1198_119825


namespace straight_A_students_after_increase_l1198_119816

/-- The number of straight-A students after new students join, given the initial conditions -/
theorem straight_A_students_after_increase 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (percentage_increase : ℚ) : ℕ :=
by
  -- Assume the initial conditions
  have h1 : initial_students = 25 := by sorry
  have h2 : new_students = 7 := by sorry
  have h3 : percentage_increase = 1/10 := by sorry

  -- Define the total number of students after new students join
  let total_students : ℕ := initial_students + new_students

  -- Define the function to calculate the number of straight-A students
  let calc_straight_A (x : ℕ) (y : ℕ) : Prop :=
    (x : ℚ) / initial_students + percentage_increase = ((x + y) : ℚ) / total_students

  -- Prove that there are 16 straight-A students after the increase
  have h4 : ∃ (x y : ℕ), calc_straight_A x y ∧ x + y = 16 := by sorry

  -- Conclude the theorem
  exact 16

end straight_A_students_after_increase_l1198_119816


namespace nickel_probability_is_one_fourth_l1198_119881

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℚ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℚ
  | Coin.Dime => 500
  | Coin.Nickel => 250
  | Coin.Penny => 100

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℚ :=
  totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℚ :=
  coinCount Coin.Dime + coinCount Coin.Nickel + coinCount Coin.Penny

/-- The probability of choosing a nickel from the jar -/
def nickelProbability : ℚ :=
  coinCount Coin.Nickel / totalCoins

theorem nickel_probability_is_one_fourth :
  nickelProbability = 1/4 := by
  sorry

end nickel_probability_is_one_fourth_l1198_119881


namespace function_range_lower_bound_l1198_119887

theorem function_range_lower_bound (n : ℕ) (f : ℤ → Fin n) 
  (h : ∀ (x y : ℤ), |x - y| ∈ ({2, 3, 5} : Set ℤ) → f x ≠ f y) : 
  n ≥ 4 := by
  sorry

end function_range_lower_bound_l1198_119887


namespace green_area_percentage_l1198_119872

/-- Represents a square flag with a symmetric pattern -/
structure SymmetricFlag where
  side_length : ℝ
  cross_area_percentage : ℝ
  green_area_percentage : ℝ

/-- The flag satisfies the problem conditions -/
def valid_flag (flag : SymmetricFlag) : Prop :=
  flag.cross_area_percentage = 25 ∧
  flag.green_area_percentage > 0 ∧
  flag.green_area_percentage < flag.cross_area_percentage

/-- The theorem to be proved -/
theorem green_area_percentage (flag : SymmetricFlag) :
  valid_flag flag → flag.green_area_percentage = 4 := by
  sorry

end green_area_percentage_l1198_119872


namespace min_b_value_l1198_119817

theorem min_b_value (a b : ℤ) (h1 : 6 < a ∧ a < 17) (h2 : b < 29) 
  (h3 : (16 : ℚ) / b - (7 : ℚ) / 28 = 15/4) : 4 ≤ b := by
  sorry

end min_b_value_l1198_119817


namespace base12_addition_l1198_119846

/-- Converts a base 12 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 12 + d) 0

/-- Converts a decimal number to its base 12 representation -/
def toBase12 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 12) ((m % 12) :: acc)
  aux n []

/-- The sum of 1704₁₂ and 259₁₂ in base 12 is equal to 1961₁₂ -/
theorem base12_addition :
  toBase12 (toDecimal [1, 7, 0, 4] + toDecimal [2, 5, 9]) = [1, 9, 6, 1] :=
by sorry

end base12_addition_l1198_119846


namespace linda_age_l1198_119868

/-- Given that Linda's age is 3 more than 2 times Jane's age, and in 5 years
    the sum of their ages will be 28, prove that Linda's current age is 13. -/
theorem linda_age (j : ℕ) : 
  (j + 5) + ((2 * j + 3) + 5) = 28 → 2 * j + 3 = 13 := by
  sorry

end linda_age_l1198_119868


namespace floor_plus_self_eq_thirteen_thirds_l1198_119818

theorem floor_plus_self_eq_thirteen_thirds (x : ℝ) : 
  (⌊x⌋ : ℝ) + x = 13/3 → x = 7/3 := by
  sorry

end floor_plus_self_eq_thirteen_thirds_l1198_119818


namespace joan_attended_games_l1198_119863

theorem joan_attended_games (total_games missed_games : ℕ) 
  (h1 : total_games = 864) 
  (h2 : missed_games = 469) : 
  total_games - missed_games = 395 := by
  sorry

end joan_attended_games_l1198_119863


namespace initial_truck_distance_l1198_119864

/-- 
Given two trucks on opposite sides of a highway, where:
- Driver A starts driving at 90 km/h
- Driver B starts 1 hour later at 80 km/h
- When they meet, Driver A has driven 140 km farther than Driver B

This theorem proves that the initial distance between the trucks is 940 km.
-/
theorem initial_truck_distance :
  ∀ (t : ℝ) (d_a d_b : ℝ),
  d_a = 90 * (t + 1) →
  d_b = 80 * t →
  d_a = d_b + 140 →
  d_a + d_b = 940 :=
by
  sorry

end initial_truck_distance_l1198_119864


namespace chess_game_probabilities_l1198_119840

/-- The probability of winning a single game -/
def prob_win : ℝ := 0.4

/-- The probability of not losing a single game -/
def prob_not_lose : ℝ := 0.9

/-- The probability of a draw in a single game -/
def prob_draw : ℝ := prob_not_lose - prob_win

/-- The probability of winning at least one game out of two independent games -/
def prob_win_at_least_one : ℝ := 1 - (1 - prob_win) ^ 2

theorem chess_game_probabilities :
  prob_draw = 0.5 ∧ prob_win_at_least_one = 0.64 := by
  sorry

end chess_game_probabilities_l1198_119840


namespace min_ab_value_l1198_119878

theorem min_ab_value (a b : ℕ+) 
  (h1 : ¬ (7 ∣ (a * b * (a + b))))
  (h2 : (7 ∣ ((a + b)^7 - a^7 - b^7))) :
  ∀ x y : ℕ+, 
    (¬ (7 ∣ (x * y * (x + y)))) → 
    ((7 ∣ ((x + y)^7 - x^7 - y^7))) → 
    a * b ≤ x * y :=
by sorry

end min_ab_value_l1198_119878


namespace solution_satisfies_equation_l1198_119852

theorem solution_satisfies_equation :
  ∃ (x y : ℝ), x ≥ 0 ∧ y > 0 ∧
  Real.sqrt (9 + x) + Real.sqrt (9 - x) + Real.sqrt y = 5 * Real.sqrt 3 ∧
  x = 0 ∧ y = 111 - 30 * Real.sqrt 3 := by
  sorry

end solution_satisfies_equation_l1198_119852


namespace integer_quotient_characterization_l1198_119879

def solution_set : Set (ℤ × ℤ) :=
  {(1, 2), (1, 3), (2, 1), (3, 1), (2, 5), (3, 5), (5, 2), (5, 3), (2, 2)}

theorem integer_quotient_characterization (m n : ℤ) :
  (∃ k : ℤ, (n^3 + 1) = k * (m * n - 1)) ↔ (m, n) ∈ solution_set :=
sorry

end integer_quotient_characterization_l1198_119879


namespace no_three_digit_sum_product_l1198_119873

theorem no_three_digit_sum_product : ∀ a b c : ℕ, 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  ¬(a + b + c = 100 * a + 10 * b + c - a * b * c) :=
by sorry


end no_three_digit_sum_product_l1198_119873


namespace no_married_triple_possible_all_married_triples_possible_l1198_119856

/-- Represents a person in the alien race -/
structure Person where
  gender : Fin 3
  likes : Fin 3 → Finset (Fin n)

/-- Represents a married triple -/
structure MarriedTriple where
  male : Fin n
  female : Fin n
  emale : Fin n

/-- The set of all persons in the colony -/
def Colony (n : ℕ) : Finset Person := sorry

/-- Predicate to check if a person likes another person -/
def likes (p1 p2 : Person) : Prop := sorry

/-- Predicate to check if a triple is a valid married triple -/
def isMarriedTriple (t : MarriedTriple) (c : Colony n) : Prop := sorry

theorem no_married_triple_possible 
  (n : ℕ) 
  (k : ℕ) 
  (h1 : Even n) 
  (h2 : k ≥ n / 2) : 
  ∃ (c : Colony n), ∀ (t : MarriedTriple), ¬isMarriedTriple t c := by sorry

theorem all_married_triples_possible 
  (n : ℕ) 
  (k : ℕ) 
  (h : k ≥ 3 * n / 4) : 
  ∃ (c : Colony n), ∃ (triples : Finset MarriedTriple), 
    (∀ t ∈ triples, isMarriedTriple t c) ∧ 
    (triples.card = n) := by sorry

end no_married_triple_possible_all_married_triples_possible_l1198_119856


namespace alcohol_concentration_problem_l1198_119866

/-- Proves that the initial concentration of alcohol in the second vessel is 55% --/
theorem alcohol_concentration_problem (vessel1_capacity : ℝ) (vessel1_concentration : ℝ)
  (vessel2_capacity : ℝ) (total_liquid : ℝ) (final_vessel_capacity : ℝ)
  (final_concentration : ℝ) :
  vessel1_capacity = 2 →
  vessel1_concentration = 20 →
  vessel2_capacity = 6 →
  total_liquid = 8 →
  final_vessel_capacity = 10 →
  final_concentration = 37 →
  ∃ vessel2_concentration : ℝ,
    vessel2_concentration = 55 ∧
    vessel1_capacity * (vessel1_concentration / 100) +
    vessel2_capacity * (vessel2_concentration / 100) =
    final_vessel_capacity * (final_concentration / 100) :=
by sorry


end alcohol_concentration_problem_l1198_119866


namespace product_three_consecutive_integers_div_by_6_l1198_119861

theorem product_three_consecutive_integers_div_by_6 (n : ℤ) :
  ∃ k : ℤ, n * (n + 1) * (n + 2) = 6 * k := by sorry

end product_three_consecutive_integers_div_by_6_l1198_119861


namespace sine_graph_shift_l1198_119810

theorem sine_graph_shift (x : ℝ) :
  3 * Real.sin (2 * (x + π/8)) = 3 * Real.sin (2 * x + π/4) :=
by sorry

end sine_graph_shift_l1198_119810


namespace equation_solution_l1198_119832

theorem equation_solution : ∃! x : ℝ, (16 : ℝ)^(x - 1) / (8 : ℝ)^(x - 1) = (64 : ℝ)^(x + 2) ∧ x = -13/5 := by
  sorry

end equation_solution_l1198_119832


namespace product_of_powers_equals_hundred_l1198_119833

theorem product_of_powers_equals_hundred : 
  (10 ^ 0.6) * (10 ^ 0.2) * (10 ^ 0.1) * (10 ^ 0.3) * (10 ^ 0.7) * (10 ^ 0.1) = 100 := by
sorry

end product_of_powers_equals_hundred_l1198_119833


namespace characterize_function_l1198_119801

/-- A strictly increasing function from positive integers to positive integers -/
def StrictlyIncreasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m < n → f m < f n

/-- The main theorem about the structure of functions satisfying f(f(n)) = f(n)^2 -/
theorem characterize_function (f : ℕ+ → ℕ+) 
  (h_incr : StrictlyIncreasing f) 
  (h_eq : ∀ n : ℕ+, f (f n) = (f n)^2) :
  ∃ c : ℕ+, 
    (∀ n : ℕ+, n ≥ 2 → f n = c * n) ∧
    (f 1 = 1 ∨ f 1 = c) :=
sorry

end characterize_function_l1198_119801


namespace cottage_build_time_l1198_119815

/-- Represents the time (in days) it takes to build a cottage given the number of builders -/
def build_time (num_builders : ℕ) : ℚ := sorry

theorem cottage_build_time :
  build_time 3 = 8 →
  build_time 6 = 4 :=
by sorry

end cottage_build_time_l1198_119815


namespace rich_walking_distance_l1198_119842

def house_to_sidewalk : ℕ := 20
def sidewalk_to_road_end : ℕ := 200

def total_distance : ℕ :=
  let to_road_end := house_to_sidewalk + sidewalk_to_road_end
  let to_intersection := to_road_end + 2 * to_road_end
  let to_route_end := to_intersection + to_intersection / 2
  2 * to_route_end

theorem rich_walking_distance :
  total_distance = 1980 := by
  sorry

end rich_walking_distance_l1198_119842


namespace arithmetic_sequence_property_l1198_119875

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₂ + a₇ = 6,
    prove that 3a₄ + a₆ = 12 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ)
    (h_arithmetic : is_arithmetic_sequence a)
    (h_sum : a 2 + a 7 = 6) :
  3 * a 4 + a 6 = 12 := by
  sorry

end arithmetic_sequence_property_l1198_119875


namespace inequality_proof_l1198_119836

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end inequality_proof_l1198_119836


namespace circle_center_l1198_119831

/-- Given a circle with polar equation ρ = 2cos(θ), its center in the Cartesian coordinate system is at (1,0) -/
theorem circle_center (ρ θ : ℝ) : ρ = 2 * Real.cos θ → ∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ (x - 1)^2 + y^2 = 1 :=
by sorry

end circle_center_l1198_119831


namespace chromium_percentage_in_new_alloy_l1198_119849

/-- Calculates the percentage of chromium in a new alloy formed by combining two alloys -/
theorem chromium_percentage_in_new_alloy 
  (weight1 : ℝ) (percentage1 : ℝ) 
  (weight2 : ℝ) (percentage2 : ℝ) :
  weight1 = 10 →
  weight2 = 30 →
  percentage1 = 12 →
  percentage2 = 8 →
  (weight1 * percentage1 / 100 + weight2 * percentage2 / 100) / (weight1 + weight2) * 100 = 9 :=
by sorry

end chromium_percentage_in_new_alloy_l1198_119849


namespace prob_last_roll_is_15th_l1198_119812

/-- The number of sides on the die -/
def n : ℕ := 20

/-- The total number of rolls -/
def total_rolls : ℕ := 15

/-- The number of non-repeating rolls -/
def non_repeating_rolls : ℕ := 13

/-- Probability of getting a specific sequence of rolls on a n-sided die,
    where the first 'non_repeating_rolls' are different from their predecessors,
    and the last roll is the same as its predecessor -/
def prob_sequence (n : ℕ) (total_rolls : ℕ) (non_repeating_rolls : ℕ) : ℚ :=
  (n - 1 : ℚ)^non_repeating_rolls / n^(total_rolls - 1)

theorem prob_last_roll_is_15th :
  prob_sequence n total_rolls non_repeating_rolls = 19^13 / 20^14 := by
  sorry

end prob_last_roll_is_15th_l1198_119812


namespace triple_equation_solution_l1198_119891

theorem triple_equation_solution (a b c : ℝ) :
  (a * (b^2 + c) = c * (c + a * b)) ∧
  (b * (c^2 + a) = a * (a + b * c)) ∧
  (c * (a^2 + b) = b * (b + c * a)) →
  (∃ x : ℝ, a = x ∧ b = x ∧ c = x) ∨
  (b = 0 ∧ c = 0) :=
sorry

end triple_equation_solution_l1198_119891


namespace function_properties_l1198_119841

open Real

theorem function_properties (f : ℝ → ℝ) (k : ℝ) (hf : Differentiable ℝ f) 
  (h0 : f 0 = -1) (hk : k > 1) (hf' : ∀ x, deriv f x > k) :
  (f (1/k) > 1/k - 1) ∧ 
  (f (1/(k-1)) > 1/(k-1)) ∧ 
  (f (1/k) < f (1/(k-1))) := by
sorry

end function_properties_l1198_119841


namespace reflection_line_sum_l1198_119858

/-- Given a line y = mx + b, if the reflection of point (2,2) across this line is (10,6), then m + b = 14 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), (x, y) = (10, 6) ∧ 
    (x - 2, y - 2) = (2 * (m * (x + 2) / 2 + b - y) / (1 + m^2), 
                      2 * (m * (y + 2) / 2 - (x + 2) / 2 + b) / (1 + m^2))) →
  m + b = 14 := by
sorry


end reflection_line_sum_l1198_119858


namespace prime_square_mod_twelve_l1198_119850

theorem prime_square_mod_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) :
  p ^ 2 % 12 = 1 := by
sorry

end prime_square_mod_twelve_l1198_119850


namespace five_student_committees_from_eight_l1198_119821

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The number of 5-student committees from 8 students is 56 -/
theorem five_student_committees_from_eight (n k : ℕ) (hn : n = 8) (hk : k = 5) :
  binomial n k = 56 := by
  sorry

end five_student_committees_from_eight_l1198_119821


namespace mat_weavers_problem_l1198_119877

/-- Given that 4 mat-weavers can weave 4 mats in 4 days, prove that 12 mat-weavers
    are needed to weave 36 mats in 12 days. -/
theorem mat_weavers_problem (weavers_group1 mats_group1 days_group1 : ℕ)
                             (mats_group2 days_group2 : ℕ) :
  weavers_group1 = 4 →
  mats_group1 = 4 →
  days_group1 = 4 →
  mats_group2 = 36 →
  days_group2 = 12 →
  (weavers_group1 * mats_group2 * days_group1 = mats_group1 * days_group2 * 12) :=
by sorry

end mat_weavers_problem_l1198_119877


namespace quadratic_inequality_solution_l1198_119867

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 50*x + 575 ≤ 25) ↔ (25 - 5*Real.sqrt 3 ≤ x ∧ x ≤ 25 + 5*Real.sqrt 3) :=
by sorry

end quadratic_inequality_solution_l1198_119867


namespace club_advantage_l1198_119880

/-- Represents a fitness club with a monthly subscription cost -/
structure FitnessClub where
  name : String
  monthlyCost : ℕ

/-- Represents an attendance pattern -/
inductive AttendancePattern
  | Regular
  | MoodBased

/-- Calculates the yearly cost for a given club and attendance pattern -/
def yearlyCost (club : FitnessClub) (pattern : AttendancePattern) : ℕ :=
  match pattern with
  | AttendancePattern.Regular => club.monthlyCost * 12
  | AttendancePattern.MoodBased => 
      if club.name = "Beta" then club.monthlyCost * 8 else club.monthlyCost * 12

/-- Calculates the number of visits per year for a given attendance pattern -/
def yearlyVisits (pattern : AttendancePattern) : ℕ :=
  match pattern with
  | AttendancePattern.Regular => 96
  | AttendancePattern.MoodBased => 56

/-- Calculates the cost per visit for a given club and attendance pattern -/
def costPerVisit (club : FitnessClub) (pattern : AttendancePattern) : ℚ :=
  (yearlyCost club pattern : ℚ) / (yearlyVisits pattern : ℚ)

theorem club_advantage :
  let alpha : FitnessClub := { name := "Alpha", monthlyCost := 999 }
  let beta : FitnessClub := { name := "Beta", monthlyCost := 1299 }
  (costPerVisit alpha AttendancePattern.Regular < costPerVisit beta AttendancePattern.Regular) ∧
  (costPerVisit beta AttendancePattern.MoodBased < costPerVisit alpha AttendancePattern.MoodBased) := by
  sorry

end club_advantage_l1198_119880


namespace exist_three_similar_numbers_l1198_119838

/-- A function that repeats a given 3-digit number to form a 1995-digit number -/
def repeat_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is a 1995-digit number -/
def is_1995_digit (n : ℕ) : Prop := sorry

/-- Predicate to check if two numbers use the same set of digits -/
def same_digit_set (a b : ℕ) : Prop := sorry

/-- Predicate to check if a number contains the digit 0 -/
def contains_zero (n : ℕ) : Prop := sorry

theorem exist_three_similar_numbers :
  ∃ (A B C : ℕ),
    is_1995_digit A ∧
    is_1995_digit B ∧
    is_1995_digit C ∧
    same_digit_set A B ∧
    same_digit_set B C ∧
    ¬contains_zero A ∧
    ¬contains_zero B ∧
    ¬contains_zero C ∧
    A + B = C :=
  sorry

end exist_three_similar_numbers_l1198_119838


namespace min_value_theorem_l1198_119889

theorem min_value_theorem (a b c d : ℝ) 
  (hb : b ≠ 0) 
  (hd : d ≠ -1) 
  (h1 : (a^2 - Real.log a) / b = (c - 1) / (d + 1))
  (h2 : (a^2 - Real.log a) / b = 1) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x y : ℝ), (x^2 - Real.log x) / y = (c - 1) / (d + 1) → 
    (x - c)^2 + (y - d)^2 ≥ min :=
by sorry

end min_value_theorem_l1198_119889


namespace average_speed_calculation_l1198_119826

/-- Given a distance of 100 kilometers traveled in 1.25 hours,
    prove that the average speed is 80 kilometers per hour. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 100 →
  time = 1.25 →
  speed = distance / time →
  speed = 80 :=
by sorry

end average_speed_calculation_l1198_119826


namespace non_shaded_perimeter_l1198_119827

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter (large : Rectangle) (shaded : Rectangle) :
  large.width = 12 ∧
  large.height = 10 ∧
  shaded.width = 5 ∧
  shaded.height = 11 ∧
  area shaded = 55 →
  perimeter large + perimeter shaded - 2 * (shaded.width + shaded.height) = 48 := by
  sorry

end non_shaded_perimeter_l1198_119827


namespace total_pay_theorem_l1198_119897

/-- Calculates the total pay for a worker given regular and overtime hours --/
def totalPay (regularRate : ℕ) (regularHours : ℕ) (overtimeHours : ℕ) : ℕ :=
  let regularPay := regularRate * regularHours
  let overtimeRate := 2 * regularRate
  let overtimePay := overtimeRate * overtimeHours
  regularPay + overtimePay

/-- Theorem stating that the total pay for the given conditions is $186 --/
theorem total_pay_theorem :
  totalPay 3 40 11 = 186 := by
  sorry

end total_pay_theorem_l1198_119897


namespace limit_implies_a_and_b_limit_implies_a_range_l1198_119808

-- Problem 1
theorem limit_implies_a_and_b (a b : ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |2*n^2 / (n+2) - n*a - b| < ε) →
  a = 2 ∧ b = 4 := by sorry

-- Problem 2
theorem limit_implies_a_range (a : ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |3^n / (3^(n+1) + (a+1)^n) - 1/3| < ε) →
  -4 < a ∧ a < 2 := by sorry

end limit_implies_a_and_b_limit_implies_a_range_l1198_119808


namespace trapezoid_segment_length_l1198_119854

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  area_ratio : ℝ
  total_length : ℝ
  h : AB > 0
  i : CD > 0
  j : area_ratio = 5 / 2
  k : AB + CD = total_length

/-- Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC to ADC is 5:2,
    and AB + CD = 280, then AB = 200 -/
theorem trapezoid_segment_length (t : Trapezoid) (h : t.total_length = 280) : t.AB = 200 := by
  sorry

end trapezoid_segment_length_l1198_119854


namespace expression_evaluation_l1198_119800

theorem expression_evaluation (x y k : ℤ) 
  (hx : x = 7) (hy : y = 3) (hk : k = 10) : 
  (x - y) * (x + y) + k = 50 := by
  sorry

end expression_evaluation_l1198_119800


namespace shaded_area_ratio_l1198_119862

/-- Given a line segment AB of length 3r with point C on AB such that AC = r and CB = 2r,
    and semi-circles constructed on AB, AC, and CB, prove that the ratio of the shaded area
    to the area of a circle with radius equal to the radius of the semi-circle on CB is 2:1. -/
theorem shaded_area_ratio (r : ℝ) (h : r > 0) : 
  let total_area := π * (3 * r)^2 / 2
  let small_semicircle_area := π * r^2 / 2
  let medium_semicircle_area := π * (2 * r)^2 / 2
  let shaded_area := total_area - (small_semicircle_area + medium_semicircle_area)
  let circle_area := π * r^2
  shaded_area / circle_area = 2 := by
  sorry

end shaded_area_ratio_l1198_119862


namespace complement_of_A_in_U_l1198_119885

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {x | ∃ n : Int, x = 2 / (n - 1) ∧ x ∈ U}

theorem complement_of_A_in_U :
  (U \ A) = {0} := by sorry

end complement_of_A_in_U_l1198_119885


namespace sector_central_angle_l1198_119894

/-- Proves that a circular sector with radius 4 cm and area 4 cm² has a central angle of 1/4 radians -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (θ : ℝ) : 
  r = 4 → area = 4 → area = 1/2 * r^2 * θ → θ = 1/4 := by sorry

end sector_central_angle_l1198_119894


namespace factorial_ratio_sum_l1198_119860

theorem factorial_ratio_sum (p q : ℕ) : 
  p < 10 → q < 10 → p > 0 → q > 0 → (840 : ℕ) = p! / q! → p + q = 10 := by
  sorry

end factorial_ratio_sum_l1198_119860


namespace lily_family_vacation_suitcases_l1198_119890

/-- The number of suitcases Lily's family brings on vacation -/
def family_suitcases (num_siblings : ℕ) (suitcases_per_sibling : ℕ) (parent_suitcases : ℕ) : ℕ :=
  num_siblings * suitcases_per_sibling + parent_suitcases

/-- Theorem stating the total number of suitcases Lily's family brings on vacation -/
theorem lily_family_vacation_suitcases :
  family_suitcases 4 2 6 = 14 := by
  sorry

#eval family_suitcases 4 2 6

end lily_family_vacation_suitcases_l1198_119890


namespace smallest_value_w_cube_plus_z_cube_l1198_119823

theorem smallest_value_w_cube_plus_z_cube (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 8) :
  Complex.abs (w^3 + z^3) = 20 := by
  sorry

end smallest_value_w_cube_plus_z_cube_l1198_119823


namespace student_arrangement_count_l1198_119830

/-- The number of ways to arrange n elements -/
def factorial (n : ℕ) : ℕ := (List.range n).foldr (· * ·) 1

/-- The number of ways to arrange 5 students in a line -/
def totalArrangements : ℕ := factorial 5

/-- The number of ways to arrange 3 students together and 2 separately -/
def restrictedArrangements : ℕ := factorial 3 * factorial 3

/-- The number of valid arrangements where 3 students are not next to each other -/
def validArrangements : ℕ := totalArrangements - restrictedArrangements

theorem student_arrangement_count :
  validArrangements = 84 :=
sorry

end student_arrangement_count_l1198_119830


namespace flower_pots_distance_l1198_119882

/-- Given 8 equally spaced points on a line, if the distance between
    the first and fifth points is 100, then the distance between
    the first and eighth points is 175. -/
theorem flower_pots_distance (points : Fin 8 → ℝ) 
    (equally_spaced : ∀ i j k : Fin 8, i.val < j.val → j.val < k.val → 
      points k - points j = points j - points i)
    (dist_1_5 : points 4 - points 0 = 100) :
    points 7 - points 0 = 175 := by
  sorry


end flower_pots_distance_l1198_119882


namespace sin_three_pi_over_four_l1198_119822

theorem sin_three_pi_over_four : Real.sin (3 * π / 4) = Real.sqrt 2 / 2 := by
  sorry

end sin_three_pi_over_four_l1198_119822


namespace average_monthly_bill_l1198_119835

theorem average_monthly_bill (first_four_months_avg : ℝ) (last_two_months_avg : ℝ) :
  first_four_months_avg = 30 →
  last_two_months_avg = 24 →
  (4 * first_four_months_avg + 2 * last_two_months_avg) / 6 = 28 :=
by sorry

end average_monthly_bill_l1198_119835


namespace coordinates_sum_of_X_l1198_119813

def X : ℝ × ℝ := sorry
def Y : ℝ × ℝ := (3, 1)
def Z : ℝ × ℝ := (-1, 5)

theorem coordinates_sum_of_X :
  (X.1 + X.2 = 4) ∧
  (‖Z - X‖ / ‖Y - X‖ = 1/2) ∧
  (‖Y - Z‖ / ‖Y - X‖ = 1/2) :=
by sorry

end coordinates_sum_of_X_l1198_119813


namespace geometric_series_sum_quarter_five_terms_l1198_119844

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_quarter_five_terms :
  geometric_series_sum (1/4) (1/4) 5 = 341/1024 := by
  sorry

end geometric_series_sum_quarter_five_terms_l1198_119844


namespace lines_coincide_l1198_119895

/-- If three lines y = kx + m, y = mx + n, and y = nx + k have a common point, then k = m = n -/
theorem lines_coincide (k m n : ℝ) (x y : ℝ) 
  (h1 : y = k * x + m)
  (h2 : y = m * x + n)
  (h3 : y = n * x + k) :
  k = m ∧ m = n := by
  sorry

end lines_coincide_l1198_119895


namespace perpendicular_lines_coefficient_l1198_119896

/-- Two lines in the form y = mx + b are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product (m₁ m₂ : ℝ) : 
  m₁ * m₂ = -1 ↔ (∃ (b₁ b₂ : ℝ), ∀ (x y : ℝ), y = m₁ * x + b₁ ↔ y = -1/m₂ * x + b₂)

/-- Given two lines ax + y + 2 = 0 and 3x - y - 2 = 0 that are perpendicular, prove that a = 2/3 -/
theorem perpendicular_lines_coefficient (a : ℝ) :
  (∀ (x y : ℝ), y = -a * x - 2 ↔ y = 3 * x - 2) →
  a = 2/3 := by
sorry

end perpendicular_lines_coefficient_l1198_119896


namespace probability_rain_given_strong_winds_l1198_119829

theorem probability_rain_given_strong_winds 
  (p_strong_winds : ℝ) 
  (p_rain : ℝ) 
  (p_both : ℝ) 
  (h1 : p_strong_winds = 0.4) 
  (h2 : p_rain = 0.5) 
  (h3 : p_both = 0.3) : 
  p_both / p_strong_winds = 3/4 := by
  sorry

end probability_rain_given_strong_winds_l1198_119829


namespace toothpicks_at_250_l1198_119893

/-- Calculates the number of toothpicks at a given stage -/
def toothpicks (stage : ℕ) : ℕ :=
  if stage = 0 then 0
  else if stage % 50 = 0 then 2 * toothpicks (stage - 1)
  else if stage = 1 then 5
  else toothpicks (stage - 1) + 5

/-- The number of toothpicks at the 250th stage is 15350 -/
theorem toothpicks_at_250 : toothpicks 250 = 15350 := by
  sorry

#eval toothpicks 250  -- This line is optional, for verification purposes

end toothpicks_at_250_l1198_119893


namespace blue_eyed_blonde_proportion_l1198_119845

/-- Proves that if the proportion of blondes among blue-eyed people is greater
    than the proportion of blondes among all people, then the proportion of
    blue-eyed people among blondes is greater than the proportion of blue-eyed
    people among all people. -/
theorem blue_eyed_blonde_proportion
  (l : ℕ) -- total number of people
  (g : ℕ) -- number of blue-eyed people
  (b : ℕ) -- number of blond-haired people
  (a : ℕ) -- number of people who are both blue-eyed and blond-haired
  (hl : l > 0)
  (hg : g > 0)
  (hb : b > 0)
  (ha : a > 0)
  (h_subset : a ≤ g ∧ a ≤ b ∧ g ≤ l ∧ b ≤ l)
  (h_proportion : (a : ℚ) / g > (b : ℚ) / l) :
  (a : ℚ) / b > (g : ℚ) / l :=
sorry

end blue_eyed_blonde_proportion_l1198_119845


namespace parking_cost_average_l1198_119802

/-- Parking cost structure and calculation -/
theorem parking_cost_average (base_cost : ℝ) (base_hours : ℝ) (additional_cost : ℝ) (total_hours : ℝ) : 
  base_cost = 20 →
  base_hours = 2 →
  additional_cost = 1.75 →
  total_hours = 9 →
  (base_cost + (total_hours - base_hours) * additional_cost) / total_hours = 3.58 := by
sorry

end parking_cost_average_l1198_119802


namespace average_of_five_numbers_l1198_119811

theorem average_of_five_numbers 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : (x₁ + x₂) / 2 = 12) 
  (h₂ : (x₃ + x₄ + x₅) / 3 = 7) : 
  (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 9 := by
  sorry

end average_of_five_numbers_l1198_119811


namespace vectors_are_collinear_l1198_119839

/-- Two 2D vectors are collinear if their cross product is zero -/
def are_collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem vectors_are_collinear : 
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (6, -4)
  are_collinear a b :=
by sorry

end vectors_are_collinear_l1198_119839


namespace michaels_pets_l1198_119820

theorem michaels_pets (total_pets : ℕ) 
  (h1 : (total_pets : ℝ) * 0.25 = total_pets * 0.5 + 9) 
  (h2 : (total_pets : ℝ) * 0.25 + total_pets * 0.5 + 9 = total_pets) : 
  total_pets = 36 := by
  sorry

end michaels_pets_l1198_119820


namespace distinct_arrangements_l1198_119892

def word_length : ℕ := 6
def freq_letter1 : ℕ := 1
def freq_letter2 : ℕ := 2
def freq_letter3 : ℕ := 3

theorem distinct_arrangements :
  (word_length.factorial) / (freq_letter1.factorial * freq_letter2.factorial * freq_letter3.factorial) = 60 := by
  sorry

end distinct_arrangements_l1198_119892


namespace west_movement_representation_l1198_119804

/-- Represents the direction of movement on an east-west road -/
inductive Direction
| East
| West

/-- Represents a movement on the road with a direction and distance -/
structure Movement where
  direction : Direction
  distance : ℝ

/-- Converts a movement to its numerical representation -/
def movementToNumber (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.distance
  | Direction.West => -m.distance

/-- The theorem stating that moving west by 7m should be denoted as -7m -/
theorem west_movement_representation :
  let eastMovement := Movement.mk Direction.East 3
  let westMovement := Movement.mk Direction.West 7
  movementToNumber eastMovement = 3 →
  movementToNumber westMovement = -7 :=
by sorry

end west_movement_representation_l1198_119804


namespace sqrt_seven_to_sixth_l1198_119886

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_sixth_l1198_119886


namespace min_area_rectangle_l1198_119876

theorem min_area_rectangle (l w : ℕ) : 
  l > 0 ∧ w > 0 ∧ 2 * (l + w) = 120 → l * w ≥ 59 := by
  sorry

end min_area_rectangle_l1198_119876


namespace oscar_height_l1198_119884

/-- Represents the heights of four brothers -/
structure BrothersHeights where
  tobias : ℝ
  victor : ℝ
  peter : ℝ
  oscar : ℝ

/-- The conditions of the problem -/
def heightConditions (h : BrothersHeights) : Prop :=
  h.tobias = 184 ∧
  h.victor - h.tobias = h.tobias - h.peter ∧
  h.peter - h.oscar = h.victor - h.tobias ∧
  (h.tobias + h.victor + h.peter + h.oscar) / 4 = 178

/-- The theorem to prove -/
theorem oscar_height (h : BrothersHeights) :
  heightConditions h → h.oscar = 160 := by
  sorry

end oscar_height_l1198_119884


namespace isosceles_triangle_angle_l1198_119807

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  isosceles : dist A B = dist B C

-- Define the angles
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem isosceles_triangle_angle (A B C O : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : angle A B C = 80) 
  (h3 : angle O A C = 10) 
  (h4 : angle O C A = 30) : 
  angle A O B = 70 := by sorry

end isosceles_triangle_angle_l1198_119807


namespace negative_square_range_l1198_119843

theorem negative_square_range (x : ℝ) (h : -1 < x ∧ x < 0) : -1 < -x^2 ∧ -x^2 < 0 := by
  sorry

end negative_square_range_l1198_119843


namespace circumcircle_radius_obtuse_triangle_consecutive_sides_l1198_119847

/-- The radius of the circumcircle of an obtuse triangle with consecutive integer sides --/
theorem circumcircle_radius_obtuse_triangle_consecutive_sides : 
  ∀ (a b c : ℕ) (R : ℝ),
    a + 1 = b → b + 1 = c →  -- Consecutive integer sides
    a < b ∧ b < c →          -- Ordered sides
    a^2 + b^2 < c^2 →        -- Obtuse triangle condition
    R = (8 * Real.sqrt 15) / 15 →  -- Radius of circumcircle
    2 * R * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = c := by
  sorry


end circumcircle_radius_obtuse_triangle_consecutive_sides_l1198_119847


namespace dany_sheep_count_l1198_119871

/-- Represents the number of bushels eaten by sheep and chickens on Dany's farm -/
def farm_bushels (num_sheep : ℕ) : ℕ :=
  2 * num_sheep + 3

/-- Theorem stating that Dany has 16 sheep on his farm -/
theorem dany_sheep_count : ∃ (num_sheep : ℕ), farm_bushels num_sheep = 35 ∧ num_sheep = 16 := by
  sorry

end dany_sheep_count_l1198_119871


namespace constant_term_expansion_l1198_119869

theorem constant_term_expansion (x : ℝ) : 
  (fun r : ℕ => (-1)^r * (Nat.choose 6 r) * x^(6 - 2*r)) 3 = -20 := by sorry

end constant_term_expansion_l1198_119869


namespace cary_walk_distance_l1198_119851

/-- The number of calories Cary burns per mile walked -/
def calories_per_mile : ℝ := 150

/-- The number of calories in the candy bar Cary eats -/
def candy_bar_calories : ℝ := 200

/-- Cary's net calorie deficit -/
def net_calorie_deficit : ℝ := 250

/-- The number of miles Cary walked round-trip -/
def miles_walked : ℝ := 3

theorem cary_walk_distance :
  miles_walked * calories_per_mile - candy_bar_calories = net_calorie_deficit :=
by sorry

end cary_walk_distance_l1198_119851


namespace function_lower_bound_l1198_119814

theorem function_lower_bound (a b : ℝ) (h : a + b = 4) :
  ∀ x : ℝ, |x + a^2| + |x - b^2| ≥ 8 := by
  sorry

end function_lower_bound_l1198_119814


namespace arithmetic_sequence_fifth_term_l1198_119853

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The fifth term of an arithmetic sequence is 5, given the sum of the first and ninth terms is 10 -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 1 + a 9 = 10) :
  a 5 = 5 := by
sorry

end arithmetic_sequence_fifth_term_l1198_119853


namespace zain_coin_count_l1198_119870

/-- Represents the number of coins Emerie has -/
structure EmerieCoinCount where
  quarters : Nat
  dimes : Nat
  nickels : Nat

/-- Calculates the total number of coins Zain has -/
def zainTotalCoins (emerie : EmerieCoinCount) : Nat :=
  (emerie.quarters + 10) + (emerie.dimes + 10) + (emerie.nickels + 10)

/-- Theorem: Given Emerie's coin counts, prove that Zain has 48 coins -/
theorem zain_coin_count (emerie : EmerieCoinCount)
  (hq : emerie.quarters = 6)
  (hd : emerie.dimes = 7)
  (hn : emerie.nickels = 5) :
  zainTotalCoins emerie = 48 := by
  sorry


end zain_coin_count_l1198_119870


namespace function_characterization_l1198_119834

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the property that the function must satisfy
def SatisfiesEquation (f : RealFunction) : Prop :=
  ∀ x y : ℝ, f ((x - y)^2) = x^2 - 2*y*f x + (f y)^2

-- State the theorem
theorem function_characterization :
  ∀ f : RealFunction, SatisfiesEquation f ↔ (∀ x : ℝ, f x = x ∨ f x = x + 1) :=
by sorry

end function_characterization_l1198_119834


namespace intercepts_satisfy_equation_intercepts_unique_l1198_119874

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-intercept of the line -/
def y_intercept : ℝ := -5

/-- Theorem stating that the x-intercept and y-intercept satisfy the line equation -/
theorem intercepts_satisfy_equation : 
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept := by
  sorry

/-- Theorem stating that the x-intercept and y-intercept are unique -/
theorem intercepts_unique :
  ∀ x y : ℝ, line_equation x 0 → x = x_intercept ∧
  ∀ x y : ℝ, line_equation 0 y → y = y_intercept := by
  sorry

end intercepts_satisfy_equation_intercepts_unique_l1198_119874


namespace analysis_method_seeks_sufficient_condition_l1198_119837

/-- The type of condition sought in the analysis method for proving inequalities -/
inductive ConditionType
  | Necessary
  | Sufficient
  | NecessaryAndSufficient
  | NecessaryOrSufficient

/-- The analysis method for proving inequalities -/
structure AnalysisMethod where
  /-- The type of condition sought by the method -/
  condition_type : ConditionType

/-- Theorem: The analysis method for proving inequalities primarily seeks sufficient conditions -/
theorem analysis_method_seeks_sufficient_condition :
  ∀ (method : AnalysisMethod), method.condition_type = ConditionType.Sufficient :=
by
  sorry

end analysis_method_seeks_sufficient_condition_l1198_119837


namespace pumpkin_pie_pieces_l1198_119805

/-- The number of pieces a pumpkin pie is cut into -/
def pumpkin_pieces : ℕ := sorry

/-- The number of pieces a custard pie is cut into -/
def custard_pieces : ℕ := 6

/-- The price of a pumpkin pie slice in dollars -/
def pumpkin_price : ℕ := 5

/-- The price of a custard pie slice in dollars -/
def custard_price : ℕ := 6

/-- The number of pumpkin pies sold -/
def pumpkin_pies_sold : ℕ := 4

/-- The number of custard pies sold -/
def custard_pies_sold : ℕ := 5

/-- The total revenue in dollars -/
def total_revenue : ℕ := 340

theorem pumpkin_pie_pieces : 
  pumpkin_pieces * pumpkin_price * pumpkin_pies_sold + 
  custard_pieces * custard_price * custard_pies_sold = total_revenue → 
  pumpkin_pieces = 8 := by sorry

end pumpkin_pie_pieces_l1198_119805


namespace positive_integer_solutions_count_l1198_119855

theorem positive_integer_solutions_count : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ p.1 > 0 ∧ p.2 > 0 ∧ (4 : ℚ) / p.1 + (2 : ℚ) / p.2 = 1) ∧
    s.card = 4 := by
  sorry

end positive_integer_solutions_count_l1198_119855


namespace smallest_with_9_odd_18_even_factors_l1198_119898

/-- The number of odd factors of an integer -/
def num_odd_factors (n : ℕ) : ℕ := sorry

/-- The number of even factors of an integer -/
def num_even_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest integer with exactly 9 odd factors and 18 even factors is 900 -/
theorem smallest_with_9_odd_18_even_factors :
  ∀ n : ℕ, num_odd_factors n = 9 ∧ num_even_factors n = 18 → n ≥ 900 ∧
  ∃ m : ℕ, m = 900 ∧ num_odd_factors m = 9 ∧ num_even_factors m = 18 :=
sorry

end smallest_with_9_odd_18_even_factors_l1198_119898


namespace correct_bottles_per_pack_l1198_119899

/-- The number of bottles in each pack of soda -/
def bottles_per_pack : ℕ := 6

/-- The number of packs Rebecca bought -/
def packs_bought : ℕ := 3

/-- The number of bottles Rebecca drinks per day -/
def bottles_per_day : ℚ := 1/2

/-- The number of days in the given period -/
def days : ℕ := 28

/-- The number of bottles remaining after the given period -/
def bottles_remaining : ℕ := 4

/-- Theorem stating that the number of bottles in each pack is correct -/
theorem correct_bottles_per_pack :
  bottles_per_pack * packs_bought - (bottles_per_day * days).floor = bottles_remaining :=
sorry

end correct_bottles_per_pack_l1198_119899


namespace a_on_diameter_bck_l1198_119857

/-- Given a triangle ABC with vertices A(a,b), B(0,0), C(c,0), and K as the intersection
    of the bisector of the exterior angle at C and the interior angle at B,
    prove that A lies on the line passing through K and the circumcenter of triangle BCK. -/
theorem a_on_diameter_bck (a b c : ℝ) : 
  let A : ℝ × ℝ := (a, b)
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (c, 0)
  let u : ℝ := Real.sqrt (a^2 + b^2)
  let w : ℝ := Real.sqrt ((a - c)^2 + b^2)
  let K : ℝ × ℝ := (c * (a + u) / (c + u - w), b * c / (c + u - w))
  let O : ℝ × ℝ := (c / 2, c * (a - c + w) * (c + u + w) / (2 * b * (c + u - w)))
  (∃ t : ℝ, A = (1 - t) • K + t • O) := by
    sorry

end a_on_diameter_bck_l1198_119857
