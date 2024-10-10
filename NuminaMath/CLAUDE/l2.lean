import Mathlib

namespace distance_between_students_l2_235

/-- The distance between two students after 4 hours, given they start from the same point
    and walk in opposite directions with speeds of 6 km/hr and 9 km/hr respectively. -/
theorem distance_between_students (speed1 speed2 time : ℝ) 
  (h1 : speed1 = 6)
  (h2 : speed2 = 9)
  (h3 : time = 4) :
  speed1 * time + speed2 * time = 60 :=
by sorry

end distance_between_students_l2_235


namespace trip_time_calculation_l2_242

/-- Given a driving time and a traffic time that is twice the driving time, 
    calculate the total trip time. -/
def total_trip_time (driving_time : ℝ) : ℝ :=
  driving_time + 2 * driving_time

theorem trip_time_calculation :
  total_trip_time 5 = 15 := by
  sorry

end trip_time_calculation_l2_242


namespace number_with_given_quotient_and_remainder_l2_259

theorem number_with_given_quotient_and_remainder : 
  ∀ N : ℕ, (N / 7 = 12 ∧ N % 7 = 5) → N = 89 :=
by sorry

end number_with_given_quotient_and_remainder_l2_259


namespace parallel_lines_k_value_l2_238

/-- Two lines are parallel if their slopes are equal or if they are both vertical -/
def parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 = 0 ∧ a2 = 0) ∨ (a1 ≠ 0 ∧ a2 ≠ 0 ∧ b1 / a1 = b2 / a2)

/-- The statement of the problem -/
theorem parallel_lines_k_value (k : ℝ) :
  parallel (k - 3) (4 - k) 1 (k - 3) (-1) 1 → k = 3 := by
  sorry

end parallel_lines_k_value_l2_238


namespace cubic_equation_solution_l2_285

theorem cubic_equation_solution :
  ∃ (x : ℝ), x + x^3 = 10 ∧ x = 2 := by
  sorry

end cubic_equation_solution_l2_285


namespace cone_cross_section_area_l2_247

/-- Represents a cone with height and base radius -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a cross-section of a cone -/
structure CrossSection where
  distanceFromCenter : ℝ

/-- Calculates the area of a cross-section passing through the vertex of a cone -/
def crossSectionArea (c : Cone) (cs : CrossSection) : ℝ :=
  sorry

theorem cone_cross_section_area 
  (c : Cone) 
  (cs : CrossSection) 
  (h1 : c.height = 20) 
  (h2 : c.baseRadius = 25) 
  (h3 : cs.distanceFromCenter = 12) : 
  crossSectionArea c cs = 500 := by
  sorry

end cone_cross_section_area_l2_247


namespace circle_points_theorem_l2_213

/-- The number of points on the circumference of the circle -/
def n : ℕ := 9

/-- Represents the property that no three points are collinear along any line passing through the circle's center -/
def no_three_collinear (points : Fin n → ℝ × ℝ) : Prop := sorry

/-- The number of different triangles that can be formed -/
def num_triangles : ℕ := Nat.choose n 3

/-- The number of distinct straight lines that can be drawn -/
def num_lines : ℕ := Nat.choose n 2

/-- Main theorem stating the number of triangles and lines -/
theorem circle_points_theorem (points : Fin n → ℝ × ℝ) 
  (h : no_three_collinear points) : 
  num_triangles = 84 ∧ num_lines = 36 := by sorry

end circle_points_theorem_l2_213


namespace inequality_proof_l2_226

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
by sorry

end inequality_proof_l2_226


namespace polygon_sides_l2_289

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →                           -- n is at least 3 for a polygon
  ((n - 2) * 180 = 2 * 360) →         -- sum of interior angles = twice sum of exterior angles
  n = 6 := by
sorry

end polygon_sides_l2_289


namespace least_four_digit_multiple_l2_233

theorem least_four_digit_multiple : ∀ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) → 
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 7 = 0) → 
  1008 ≤ n :=
by sorry

end least_four_digit_multiple_l2_233


namespace fraction_product_equality_l2_288

theorem fraction_product_equality : (2 : ℚ) / 8 * (6 : ℚ) / 9 = (1 : ℚ) / 6 := by sorry

end fraction_product_equality_l2_288


namespace rainbow_bead_arrangement_probability_l2_219

def num_beads : ℕ := 7

def num_permutations (n : ℕ) : ℕ := Nat.factorial n

def probability_specific_arrangement (n : ℕ) : ℚ :=
  1 / (num_permutations n)

theorem rainbow_bead_arrangement_probability :
  probability_specific_arrangement num_beads = 1 / 5040 := by
  sorry

end rainbow_bead_arrangement_probability_l2_219


namespace cheapest_plan_b_l2_261

/-- Represents the cost of a cell phone plan in cents -/
def PlanCost (flatFee minutes : ℕ) (perMinute : ℚ) : ℚ :=
  (flatFee : ℚ) * 100 + perMinute * minutes

theorem cheapest_plan_b (minutes : ℕ) : 
  (minutes ≥ 834) ↔ 
  (PlanCost 25 minutes 6 < PlanCost 0 minutes 12 ∧ 
   PlanCost 25 minutes 6 < PlanCost 0 minutes 9) :=
sorry

end cheapest_plan_b_l2_261


namespace playground_slide_total_l2_215

theorem playground_slide_total (boys_first_10min : ℕ) (boys_next_5min : ℕ) (boys_last_20min : ℕ)
  (h1 : boys_first_10min = 22)
  (h2 : boys_next_5min = 13)
  (h3 : boys_last_20min = 35) :
  boys_first_10min + boys_next_5min + boys_last_20min = 70 :=
by sorry

end playground_slide_total_l2_215


namespace cube_sum_minus_product_eq_2003_l2_200

theorem cube_sum_minus_product_eq_2003 :
  ∀ x y z : ℤ, x^3 + y^3 + z^3 - 3*x*y*z = 2003 ↔ 
  ((x = 668 ∧ y = 668 ∧ z = 667) ∨ 
   (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
   (x = 667 ∧ y = 668 ∧ z = 668)) :=
by sorry

end cube_sum_minus_product_eq_2003_l2_200


namespace equation_solution_l2_272

theorem equation_solution (x y z w : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
  (eq : 1/x + 1/y = 1/z + w) : 
  z = x*y / (x + y - w*x*y) := by
sorry

end equation_solution_l2_272


namespace rectangle_problem_l2_236

theorem rectangle_problem (num_rectangles : ℕ) (area_large : ℝ) 
  (h1 : num_rectangles = 6)
  (h2 : area_large = 6000) :
  ∃ (x : ℝ), 
    x > 0 ∧ 
    (num_rectangles : ℝ) * (2/5 * x) * x = area_large ∧ 
    x = 50 := by
  sorry

end rectangle_problem_l2_236


namespace product_from_hcf_lcm_l2_207

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 128) :
  a * b = 2560 := by
  sorry

end product_from_hcf_lcm_l2_207


namespace inequality_not_preserved_after_subtraction_of_squares_l2_212

theorem inequality_not_preserved_after_subtraction_of_squares : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧ (a - a^2) ≤ (b - b^2) := by
  sorry

end inequality_not_preserved_after_subtraction_of_squares_l2_212


namespace modulus_of_z_l2_211

open Complex

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : abs z = 2 := by
  sorry

end modulus_of_z_l2_211


namespace round_robin_tournament_teams_l2_283

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 36 games, there are 9 teams -/
theorem round_robin_tournament_teams :
  ∃ (n : ℕ), n > 0 ∧ num_games n = 36 → n = 9 :=
by
  sorry

end round_robin_tournament_teams_l2_283


namespace vector_projection_l2_243

theorem vector_projection (a b : ℝ × ℝ) 
  (h1 : (a.1 + b.1, a.2 + b.2) • (2*a.1 - b.1, 2*a.2 - b.2) = -12)
  (h2 : Real.sqrt (a.1^2 + a.2^2) = 2)
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 4) :
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2) = -2 := by
sorry

end vector_projection_l2_243


namespace max_value_of_f_l2_220

open Real

theorem max_value_of_f (φ : ℝ) :
  (⨆ x, cos (x + 2*φ) + 2*sin φ * sin (x + φ)) = 1 := by sorry

end max_value_of_f_l2_220


namespace square_area_from_vertices_l2_292

/-- The area of a square with adjacent vertices at (1,3) and (-2,7) is 25 -/
theorem square_area_from_vertices :
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (-2, 7)
  let distance := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let area := distance^2
  area = 25 := by sorry

end square_area_from_vertices_l2_292


namespace larger_number_problem_l2_280

theorem larger_number_problem (L S : ℕ) (hL : L > S) :
  L - S = 1390 → L = 6 * S + 15 → L = 1665 := by
  sorry

end larger_number_problem_l2_280


namespace sqrt_18_minus_sqrt_6_over_sqrt_2_between_1_and_2_l2_267

theorem sqrt_18_minus_sqrt_6_over_sqrt_2_between_1_and_2 :
  1 < (Real.sqrt 18 - Real.sqrt 6) / Real.sqrt 2 ∧
  (Real.sqrt 18 - Real.sqrt 6) / Real.sqrt 2 < 2 := by
  sorry

end sqrt_18_minus_sqrt_6_over_sqrt_2_between_1_and_2_l2_267


namespace product_equality_equal_S_not_imply_equal_Q_l2_241

-- Define a structure for a triangle divided by cevians
structure CevianTriangle where
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  Q₁ : ℝ
  Q₂ : ℝ
  Q₃ : ℝ
  S_positive : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0
  Q_positive : Q₁ > 0 ∧ Q₂ > 0 ∧ Q₃ > 0

-- Theorem 1: Product of S areas equals product of Q areas
theorem product_equality (t : CevianTriangle) : t.S₁ * t.S₂ * t.S₃ = t.Q₁ * t.Q₂ * t.Q₃ := by
  sorry

-- Theorem 2: Equal S areas do not necessarily imply equal Q areas
theorem equal_S_not_imply_equal_Q :
  ∃ t : CevianTriangle, (t.S₁ = t.S₂ ∧ t.S₂ = t.S₃) ∧ (t.Q₁ ≠ t.Q₂ ∨ t.Q₂ ≠ t.Q₃ ∨ t.Q₁ ≠ t.Q₃) := by
  sorry

end product_equality_equal_S_not_imply_equal_Q_l2_241


namespace distinct_prime_factors_of_30_factorial_l2_284

theorem distinct_prime_factors_of_30_factorial (n : ℕ) :
  n = 30 →
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card =
  (Finset.filter (λ p => p.Prime ∧ p ∣ n!) (Finset.range (n + 1))).card :=
sorry

end distinct_prime_factors_of_30_factorial_l2_284


namespace pokemon_card_solution_l2_296

def pokemon_card_problem (initial_cards : ℕ) : Prop :=
  let after_trade := initial_cards - 5 + 3
  let after_giving := after_trade - 9
  let final_cards := after_giving + 2
  final_cards = 4

theorem pokemon_card_solution :
  pokemon_card_problem 13 := by sorry

end pokemon_card_solution_l2_296


namespace roots_of_equation_l2_227

def f (x : ℝ) : ℝ := x * (x - 3)^2 * (5 + x) * (x^2 - 1)

theorem roots_of_equation :
  {x : ℝ | f x = 0} = {0, 3, -5, 1, -1} := by sorry

end roots_of_equation_l2_227


namespace cube_root_equal_self_l2_208

theorem cube_root_equal_self (a : ℝ) : a^(1/3) = a → a = 1 ∨ a = 0 ∨ a = -1 := by
  sorry

end cube_root_equal_self_l2_208


namespace asymptotic_necessary_not_sufficient_l2_249

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the asymptotic line equation
def asymptotic_line (a b x y : ℝ) : Prop := y = (b/a) * x ∨ y = -(b/a) * x

-- Theorem stating that the asymptotic line is a necessary but not sufficient condition for the hyperbola
theorem asymptotic_necessary_not_sufficient (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) :
  (∀ x y, hyperbola a b x y → asymptotic_line a b x y) ∧
  (∃ x y, asymptotic_line a b x y ∧ ¬hyperbola a b x y) :=
sorry

end asymptotic_necessary_not_sufficient_l2_249


namespace teddy_bear_cost_teddy_bear_cost_proof_l2_260

theorem teddy_bear_cost (initial_toys : ℕ) (initial_toy_cost : ℕ) 
  (teddy_bears : ℕ) (total_cost : ℕ) : ℕ :=
  let remaining_cost := total_cost - initial_toys * initial_toy_cost
  remaining_cost / teddy_bears

theorem teddy_bear_cost_proof :
  teddy_bear_cost 28 10 20 580 = 15 := by
  sorry

end teddy_bear_cost_teddy_bear_cost_proof_l2_260


namespace perpendicular_vectors_magnitude_l2_282

theorem perpendicular_vectors_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![x, 2]
  (a 0 * b 0 + a 1 * b 1 = 0) →  -- perpendicular condition
  ‖a + 2 • b‖ = Real.sqrt 34 := by
  sorry

end perpendicular_vectors_magnitude_l2_282


namespace max_t_geq_pi_l2_257

noncomputable def f (x : ℝ) : ℝ := Real.sin x / x

theorem max_t_geq_pi (t : ℝ) (h : ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < t → f x₁ > f x₂) :
  t ≥ π :=
sorry

end max_t_geq_pi_l2_257


namespace inequality_proof_l2_299

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^4 + b^4 + c^4 - 2*(a^2*b^2 + a^2*c^2 + b^2*c^2) + a^2*b*c + b^2*a*c + c^2*a*b ≥ 0 := by
  sorry

end inequality_proof_l2_299


namespace team_formation_ways_l2_264

/-- The number of ways to choose 2 players from a group of 5 players -/
def choose_teams (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- There are 5 friends in total -/
def total_players : ℕ := 5

/-- The size of the smaller team -/
def team_size : ℕ := 2

theorem team_formation_ways :
  choose_teams total_players team_size = 10 := by
  sorry

end team_formation_ways_l2_264


namespace solve_equation_l2_298

theorem solve_equation (x : ℝ) : 3 * x + 12 = (1/3) * (6 * x + 36) → x = 0 := by
  sorry

end solve_equation_l2_298


namespace buses_passed_count_l2_223

/-- Represents the frequency of bus departures in hours -/
structure BusSchedule where
  austin_to_san_antonio : ℕ
  san_antonio_to_austin : ℕ

/-- Represents the journey details -/
structure JourneyDetails where
  trip_duration : ℕ
  same_highway : Bool

/-- Calculates the number of buses passed during the journey -/
def buses_passed (schedule : BusSchedule) (journey : JourneyDetails) : ℕ :=
  sorry

theorem buses_passed_count 
  (schedule : BusSchedule)
  (journey : JourneyDetails)
  (h1 : schedule.austin_to_san_antonio = 2)
  (h2 : schedule.san_antonio_to_austin = 3)
  (h3 : journey.trip_duration = 8)
  (h4 : journey.same_highway = true) :
  buses_passed schedule journey = 4 :=
sorry

end buses_passed_count_l2_223


namespace multiple_with_ones_and_zeros_multiple_with_only_ones_l2_297

def a (k : ℕ) : ℕ := (10^k - 1) / 9

theorem multiple_with_ones_and_zeros (n : ℤ) :
  ∃ k l : ℕ, k < l ∧ n ∣ (a l - a k) :=
sorry

theorem multiple_with_only_ones (n : ℤ) (h_odd : Odd n) (h_not_div_5 : ¬(5 ∣ n)) :
  ∃ d : ℕ, n ∣ (10^d - 1) :=
sorry

end multiple_with_ones_and_zeros_multiple_with_only_ones_l2_297


namespace quadratic_equation_solution_l2_293

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 - 2*x₁ - 8 = 0) ∧ 
  (x₂^2 - 2*x₂ - 8 = 0) ∧ 
  x₁ = 4 ∧ 
  x₂ = -2 :=
by
  sorry

#check quadratic_equation_solution

end quadratic_equation_solution_l2_293


namespace unique_integer_root_difference_l2_217

/-- Given an integer n, A = √(n² + 24), and B = √(n² - 9),
    prove that n = 5 is the only value for which A - B is an integer. -/
theorem unique_integer_root_difference (n : ℤ) : 
  (∃ m : ℤ, Real.sqrt (n^2 + 24) - Real.sqrt (n^2 - 9) = m) ↔ n = 5 :=
by sorry

end unique_integer_root_difference_l2_217


namespace ducks_at_north_pond_l2_279

/-- The number of ducks at North Pond given the specified conditions -/
theorem ducks_at_north_pond :
  let mallard_lake_michigan : ℕ := 100
  let pintail_lake_michigan : ℕ := 75
  let mallard_north_pond : ℕ := 2 * mallard_lake_michigan + 6
  let pintail_north_pond : ℕ := 4 * mallard_lake_michigan
  mallard_north_pond + pintail_north_pond = 606 :=
by sorry


end ducks_at_north_pond_l2_279


namespace koschei_coins_l2_248

theorem koschei_coins : ∃! n : ℕ, 300 ≤ n ∧ n ≤ 400 ∧ n % 10 = 7 ∧ n % 12 = 9 ∧ n = 357 := by
  sorry

end koschei_coins_l2_248


namespace special_item_identification_l2_245

/-- Represents the result of a yes/no question -/
inductive Answer
| Yes
| No

/-- Converts an Answer to a natural number (0 for Yes, 1 for No) -/
def answerToNat (a : Answer) : Nat :=
  match a with
  | Answer.Yes => 0
  | Answer.No => 1

/-- Represents the set of items -/
def Items : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7}

/-- The function to determine the special item based on three answers -/
def determineSpecialItem (a₁ a₂ a₃ : Answer) : Nat :=
  answerToNat a₁ + 2 * answerToNat a₂ + 4 * answerToNat a₃

theorem special_item_identification :
  ∀ (special : Nat),
  special ∈ Items →
  ∃ (a₁ a₂ a₃ : Answer),
  determineSpecialItem a₁ a₂ a₃ = special ∧
  ∀ (other : Nat),
  other ∈ Items →
  other ≠ special →
  determineSpecialItem a₁ a₂ a₃ ≠ other :=
sorry

end special_item_identification_l2_245


namespace good_number_characterization_l2_250

def isGoodNumber (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), Function.Bijective a ∧
    ∀ k : Fin n, ∃ m : ℕ, (k.val + 1 + (a k).val + 1 : ℕ) = m ^ 2

def notGoodNumbers : Set ℕ := {1, 2, 4, 6, 7, 9, 11}

theorem good_number_characterization (n : ℕ) :
  n ≠ 0 → (isGoodNumber n ↔ n ∉ notGoodNumbers) :=
by sorry

end good_number_characterization_l2_250


namespace minimum_in_interval_implies_a_range_l2_287

open Real

/-- The function f(x) = x³ - 2ax + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*a*x + a

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a

theorem minimum_in_interval_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Ioo 0 1, IsLocalMin (f a) x) →
  (∀ x ∈ Set.Ioo 0 1, ¬IsLocalMax (f a) x) →
  a ∈ Set.Ioo 0 (3/2) :=
by sorry

end minimum_in_interval_implies_a_range_l2_287


namespace inheritance_sum_l2_256

theorem inheritance_sum (n : ℕ) : n = 36 → (n * (n + 1)) / 2 = 666 := by
  sorry

end inheritance_sum_l2_256


namespace gcf_of_2550_and_7140_l2_266

theorem gcf_of_2550_and_7140 : Nat.gcd 2550 7140 = 510 := by
  sorry

end gcf_of_2550_and_7140_l2_266


namespace initial_water_amount_l2_255

/-- Proves that the initial amount of water in the tank was 100 L given the conditions of the rainstorm. -/
theorem initial_water_amount (flow_rate : ℝ) (duration : ℝ) (total_after : ℝ) : 
  flow_rate = 2 → duration = 90 → total_after = 280 → 
  total_after - (flow_rate * duration) = 100 := by
sorry

end initial_water_amount_l2_255


namespace garden_perimeter_l2_230

/-- The perimeter of a rectangular garden with length 205 m and breadth 95 m is 600 m. -/
theorem garden_perimeter : 
  ∀ (perimeter length breadth : ℕ), 
    length = 205 → 
    breadth = 95 → 
    perimeter = 2 * (length + breadth) → 
    perimeter = 600 := by
  sorry

end garden_perimeter_l2_230


namespace log_powers_sum_l2_240

theorem log_powers_sum (a b : ℝ) (ha : a = Real.log 25) (hb : b = Real.log 49) :
  (5 : ℝ) ^ (a / b) + (7 : ℝ) ^ (b / a) = 12 := by
  sorry

end log_powers_sum_l2_240


namespace school_referendum_l2_205

theorem school_referendum (U : Finset ℕ) (A B : Finset ℕ) : 
  Finset.card U = 198 →
  Finset.card A = 149 →
  Finset.card B = 119 →
  Finset.card (U \ (A ∪ B)) = 29 →
  Finset.card (A ∩ B) = 99 := by
  sorry

end school_referendum_l2_205


namespace log_sum_exists_base_l2_244

theorem log_sum_exists_base : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, x > 0 → Real.log x / Real.log 2 + Real.log x / Real.log 3 = Real.log x / Real.log a := by
  sorry

end log_sum_exists_base_l2_244


namespace fractional_equation_solution_range_l2_270

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ (m / (x - 2) + 1 = x / (2 - x))) → 
  (m ≤ 2 ∧ m ≠ -2) :=
by sorry

end fractional_equation_solution_range_l2_270


namespace largest_of_seven_consecutive_integers_l2_281

theorem largest_of_seven_consecutive_integers (a : ℕ) 
  (h1 : a > 0)
  (h2 : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) = 77) :
  a + 6 = 14 := by
  sorry

end largest_of_seven_consecutive_integers_l2_281


namespace max_value_abc_l2_210

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  2*a*b*Real.sqrt 2 + 2*a*c + 2*b*c ≤ 1 / Real.sqrt 2 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a'^2 + b'^2 + c'^2 = 1 ∧
  2*a'*b'*Real.sqrt 2 + 2*a'*c' + 2*b'*c' = 1 / Real.sqrt 2 := by
  sorry

end max_value_abc_l2_210


namespace train_passing_time_l2_203

theorem train_passing_time (fast_length slow_length : ℝ) (time_slow_observes : ℝ) :
  fast_length = 150 →
  slow_length = 200 →
  time_slow_observes = 6 →
  ∃ time_fast_observes : ℝ,
    time_fast_observes = 8 ∧
    fast_length / time_slow_observes = slow_length / time_fast_observes :=
by sorry

end train_passing_time_l2_203


namespace adult_office_visit_cost_l2_291

/-- Represents the cost of an adult's office visit -/
def adult_cost : ℝ := sorry

/-- Represents the number of adult patients seen per hour -/
def adults_per_hour : ℕ := 4

/-- Represents the number of child patients seen per hour -/
def children_per_hour : ℕ := 3

/-- Represents the cost of a child's office visit -/
def child_cost : ℝ := 25

/-- Represents the number of hours worked in a day -/
def hours_per_day : ℕ := 8

/-- Represents the total income for a day -/
def daily_income : ℝ := 2200

theorem adult_office_visit_cost :
  adult_cost * (adults_per_hour * hours_per_day : ℝ) +
  child_cost * (children_per_hour * hours_per_day : ℝ) =
  daily_income ∧ adult_cost = 50 := by sorry

end adult_office_visit_cost_l2_291


namespace periodic_function_proof_l2_231

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a * f b

theorem periodic_function_proof (f : ℝ → ℝ) (c : ℝ) 
    (h1 : FunctionalEquation f) 
    (h2 : c > 0) 
    (h3 : f (c / 2) = 0) :
    ∀ x : ℝ, f (x + 2 * c) = f x := by
  sorry

end periodic_function_proof_l2_231


namespace median_inequality_exists_l2_216

/-- A dataset is represented as a list of real numbers -/
def Dataset := List ℝ

/-- The median of a dataset -/
def median (d : Dataset) : ℝ := sorry

/-- Count of values in a dataset less than a given value -/
def count_less_than (d : Dataset) (x : ℝ) : ℕ := sorry

/-- Count of values in a dataset greater than a given value -/
def count_greater_than (d : Dataset) (x : ℝ) : ℕ := sorry

/-- Theorem: There exists a dataset where the number of values greater than 
    the median is not equal to the number of values less than the median -/
theorem median_inequality_exists : 
  ∃ (d : Dataset), count_greater_than d (median d) ≠ count_less_than d (median d) := by
  sorry

end median_inequality_exists_l2_216


namespace simplify_expression_l2_274

theorem simplify_expression (x : ℝ) : (3*x)^4 - (4*x^2)*(2*x^3) + 5*x^4 = 86*x^4 - 8*x^5 := by
  sorry

end simplify_expression_l2_274


namespace only_108_117_207_satisfy_l2_273

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Calculates the sum of digits of a three-digit number -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- Increases a three-digit number by 3 -/
def increaseByThree (n : ThreeDigitNumber) : ThreeDigitNumber :=
  let newOnes := (n.ones + 3) % 10
  let carryTens := (n.ones + 3) / 10
  let newTens := (n.tens + carryTens) % 10
  let carryHundreds := (n.tens + carryTens) / 10
  let newHundreds := n.hundreds + carryHundreds
  ⟨newHundreds, newTens, newOnes, sorry, sorry, sorry⟩

/-- Checks if a three-digit number satisfies the condition -/
def satisfiesCondition (n : ThreeDigitNumber) : Prop :=
  digitSum (increaseByThree n) = 3 * digitSum n

/-- The main theorem stating that only 108, 117, and 207 satisfy the condition -/
theorem only_108_117_207_satisfy :
  ∀ n : ThreeDigitNumber, satisfiesCondition n ↔ 
    (n.hundreds = 1 ∧ n.tens = 0 ∧ n.ones = 8) ∨
    (n.hundreds = 1 ∧ n.tens = 1 ∧ n.ones = 7) ∨
    (n.hundreds = 2 ∧ n.tens = 0 ∧ n.ones = 7) :=
  sorry

end only_108_117_207_satisfy_l2_273


namespace remaining_distance_to_cave_end_l2_278

theorem remaining_distance_to_cave_end (total_depth : ℕ) (traveled_distance : ℕ) 
  (h1 : total_depth = 1218)
  (h2 : traveled_distance = 849) :
  total_depth - traveled_distance = 369 := by
  sorry

end remaining_distance_to_cave_end_l2_278


namespace mountain_has_three_sections_l2_237

/-- Given a mountain with eagles, calculate the number of sections. -/
def mountain_sections (eagles_per_section : ℕ) (total_eagles : ℕ) : ℕ :=
  total_eagles / eagles_per_section

/-- Theorem: The mountain has 3 sections given the specified conditions. -/
theorem mountain_has_three_sections :
  let eagles_per_section := 6
  let total_eagles := 18
  mountain_sections eagles_per_section total_eagles = 3 := by
  sorry

end mountain_has_three_sections_l2_237


namespace eight_bead_bracelet_arrangements_l2_214

-- Define the number of beads
def n : ℕ := 8

-- Define the function to calculate the number of distinct arrangements
def bracelet_arrangements (m : ℕ) : ℕ :=
  (Nat.factorial m) / (m * 2)

-- Theorem statement
theorem eight_bead_bracelet_arrangements :
  bracelet_arrangements n = 2520 := by
  sorry

end eight_bead_bracelet_arrangements_l2_214


namespace positive_real_inequality_l2_228

theorem positive_real_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_prod : a * b * c * d = 1) : 
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d := by
  sorry

end positive_real_inequality_l2_228


namespace no_prime_divisible_by_35_l2_204

/-- A number is prime if it has exactly two distinct positive divisors -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 → m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_35 :
  ¬∃ p : ℕ, isPrime p ∧ 35 ∣ p :=
by sorry

end no_prime_divisible_by_35_l2_204


namespace stick_cutting_l2_224

theorem stick_cutting (n : ℕ) : (1 : ℝ) / 2^n = 1 / 64 → n = 6 := by
  sorry

end stick_cutting_l2_224


namespace max_z_value_l2_258

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x*y + y*z + z*x = 3) :
  z ≤ 13/3 :=
by sorry

end max_z_value_l2_258


namespace correct_average_l2_202

theorem correct_average (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 18 ∧ wrong_num = 26 ∧ correct_num = 36 →
  (n * initial_avg - wrong_num + correct_num) / n = 19 := by
  sorry

end correct_average_l2_202


namespace hiking_rate_ratio_l2_268

/-- Prove the ratio of hiking rates for a mountain trip -/
theorem hiking_rate_ratio 
  (rate_up : ℝ) 
  (time_up : ℝ) 
  (distance_down : ℝ) 
  (rate_up_is_4 : rate_up = 4)
  (time_up_is_2 : time_up = 2)
  (distance_down_is_12 : distance_down = 12)
  (time_equal : time_up = distance_down / (distance_down / time_up * rate_up)) :
  distance_down / (time_up * rate_up) / rate_up = 3 / 2 := by
  sorry


end hiking_rate_ratio_l2_268


namespace squirrel_walnuts_l2_201

/-- The number of walnuts left in the squirrels' burrow after their gathering and eating activities. -/
def walnuts_left (initial : ℕ) (boy_gathered : ℕ) (boy_dropped : ℕ) (girl_brought : ℕ) (girl_ate : ℕ) : ℕ :=
  initial + (boy_gathered - boy_dropped) + girl_brought - girl_ate

/-- Theorem stating that given the specific conditions of the problem, the number of walnuts left is 20. -/
theorem squirrel_walnuts : walnuts_left 12 6 1 5 2 = 20 := by
  sorry

end squirrel_walnuts_l2_201


namespace solve_consecutive_integer_sets_l2_222

/-- A set of consecutive integers -/
structure ConsecutiveIntegerSet where
  start : ℤ
  size : ℕ

/-- The sum of elements in a ConsecutiveIntegerSet -/
def sum_of_set (s : ConsecutiveIntegerSet) : ℤ :=
  (s.size : ℤ) * (2 * s.start + s.size - 1) / 2

/-- The greatest element in a ConsecutiveIntegerSet -/
def greatest_element (s : ConsecutiveIntegerSet) : ℤ :=
  s.start + s.size - 1

theorem solve_consecutive_integer_sets :
  ∃ (m : ℕ) (a b : ConsecutiveIntegerSet),
    m > 0 ∧
    a.size = m ∧
    b.size = 2 * m ∧
    sum_of_set a = 2 * m ∧
    sum_of_set b = m ∧
    |greatest_element a - greatest_element b| = 99 →
    m = 201 := by
  sorry

end solve_consecutive_integer_sets_l2_222


namespace symmetric_origin_correct_symmetric_point_correct_l2_276

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry with respect to the origin
def symmetricOrigin (p : Point2D) : Point2D :=
  { x := -p.x, y := -p.y }

-- Define symmetry with respect to another point
def symmetricPoint (p : Point2D) (k : Point2D) : Point2D :=
  { x := 2 * k.x - p.x, y := 2 * k.y - p.y }

-- Theorem for symmetry with respect to the origin
theorem symmetric_origin_correct (m : Point2D) :
  symmetricOrigin m = { x := -m.x, y := -m.y } := by sorry

-- Theorem for symmetry with respect to another point
theorem symmetric_point_correct (m k : Point2D) :
  symmetricPoint m k = { x := 2 * k.x - m.x, y := 2 * k.y - m.y } := by sorry

end symmetric_origin_correct_symmetric_point_correct_l2_276


namespace three_fourths_cubed_l2_265

theorem three_fourths_cubed : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end three_fourths_cubed_l2_265


namespace angle_sum_l2_252

theorem angle_sum (θ φ : Real) (h1 : 4 * (Real.cos θ)^2 + 3 * (Real.cos φ)^2 = 1)
  (h2 : 4 * Real.cos (2 * θ) + 3 * Real.sin (2 * φ) = 0)
  (h3 : 0 < θ ∧ θ < Real.pi / 2) (h4 : 0 < φ ∧ φ < Real.pi / 2) :
  θ + 3 * φ = Real.pi / 2 :=
by sorry

end angle_sum_l2_252


namespace train_speed_l2_277

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 240 ∧ 
  bridge_length = 150 ∧ 
  crossing_time = 20 →
  (train_length + bridge_length) / crossing_time * 3.6 = 70.2 := by
  sorry

#check train_speed

end train_speed_l2_277


namespace farm_has_two_fields_l2_209

/-- Represents a corn field -/
structure CornField where
  rows : ℕ
  cobs_per_row : ℕ

/-- Calculates the total number of corn cobs in a field -/
def total_cobs (field : CornField) : ℕ :=
  field.rows * field.cobs_per_row

/-- Represents the farm's corn production -/
structure FarmProduction where
  field1 : CornField
  field2 : CornField
  total_cobs : ℕ

/-- Theorem: The farm is growing corn in 2 fields -/
theorem farm_has_two_fields (farm : FarmProduction) : 
  farm.field1.rows = 13 ∧ 
  farm.field2.rows = 16 ∧ 
  farm.field1.cobs_per_row = 4 ∧ 
  farm.field2.cobs_per_row = 4 ∧ 
  farm.total_cobs = 116 → 
  2 = (if total_cobs farm.field1 + total_cobs farm.field2 = farm.total_cobs then 2 else 1) :=
by sorry


end farm_has_two_fields_l2_209


namespace count_positive_area_triangles_l2_221

/-- A point in the 4x4 grid -/
structure GridPoint where
  x : Fin 4
  y : Fin 4

/-- A triangle formed by three points in the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Predicate to check if three points are collinear -/
def collinear (p1 p2 p3 : GridPoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Predicate to check if a triangle has positive area -/
def positiveArea (t : GridTriangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all triangles with positive area in the 4x4 grid -/
def PositiveAreaTriangles : Finset GridTriangle :=
  sorry

theorem count_positive_area_triangles :
  Finset.card PositiveAreaTriangles = 520 :=
sorry

end count_positive_area_triangles_l2_221


namespace selene_and_tanya_spending_l2_225

/-- Represents the prices of items in the school canteen -/
structure CanteenPrices where
  sandwich : ℕ
  hamburger : ℕ
  hotdog : ℕ
  fruitJuice : ℕ

/-- Represents Selene's purchase -/
structure SelenePurchase where
  sandwiches : ℕ
  fruitJuice : ℕ

/-- Represents Tanya's purchase -/
structure TanyaPurchase where
  hamburgers : ℕ
  fruitJuice : ℕ

/-- Calculates the total spending of Selene and Tanya -/
def totalSpending (prices : CanteenPrices) (selene : SelenePurchase) (tanya : TanyaPurchase) : ℕ :=
  prices.sandwich * selene.sandwiches + prices.fruitJuice * selene.fruitJuice +
  prices.hamburger * tanya.hamburgers + prices.fruitJuice * tanya.fruitJuice

/-- Theorem stating that Selene and Tanya spend $16 in total -/
theorem selene_and_tanya_spending :
  ∀ (prices : CanteenPrices) (selene : SelenePurchase) (tanya : TanyaPurchase),
    prices.sandwich = 2 →
    prices.hamburger = 2 →
    prices.hotdog = 1 →
    prices.fruitJuice = 2 →
    selene.sandwiches = 3 →
    selene.fruitJuice = 1 →
    tanya.hamburgers = 2 →
    tanya.fruitJuice = 2 →
    totalSpending prices selene tanya = 16 := by
  sorry

end selene_and_tanya_spending_l2_225


namespace complete_square_sum_l2_290

theorem complete_square_sum (a b c : ℤ) : 
  (∀ x : ℚ, 25 * x^2 + 30 * x - 45 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 62 := by
  sorry

end complete_square_sum_l2_290


namespace simplify_sqrt_expression_l2_295

theorem simplify_sqrt_expression :
  (Real.sqrt 500 / Real.sqrt 180) + (Real.sqrt 128 / Real.sqrt 32) = 11 / 3 := by
  sorry

end simplify_sqrt_expression_l2_295


namespace polynomial_factorization_l2_239

theorem polynomial_factorization (a : ℝ) : a^2 - a = a * (a - 1) := by
  sorry

end polynomial_factorization_l2_239


namespace average_price_of_rackets_l2_254

/-- The average price of a pair of rackets given total sales and number of pairs sold -/
theorem average_price_of_rackets (total_sales : ℝ) (num_pairs : ℕ) (h1 : total_sales = 490) (h2 : num_pairs = 50) :
  total_sales / num_pairs = 9.80 := by
  sorry

end average_price_of_rackets_l2_254


namespace x_squared_gt_one_necessary_not_sufficient_l2_271

theorem x_squared_gt_one_necessary_not_sufficient (x : ℝ) :
  (∀ x, x > 1 → x^2 > 1) ∧ (∃ x, x^2 > 1 ∧ x ≤ 1) := by
  sorry

end x_squared_gt_one_necessary_not_sufficient_l2_271


namespace toy_store_shelves_l2_234

def shelves_required (initial_stock new_shipment bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

theorem toy_store_shelves :
  shelves_required 4 10 7 = 2 := by
  sorry

end toy_store_shelves_l2_234


namespace condition_necessary_not_sufficient_l2_294

theorem condition_necessary_not_sufficient :
  (∀ a b : ℝ, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  (∃ a b : ℝ, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) :=
by sorry

end condition_necessary_not_sufficient_l2_294


namespace crates_delivered_is_twelve_l2_218

/-- The number of crates of apples delivered to a factory --/
def crates_delivered (apples_per_crate : ℕ) (rotten_apples : ℕ) (apples_per_box : ℕ) (boxes_filled : ℕ) : ℕ :=
  (boxes_filled * apples_per_box + rotten_apples) / apples_per_crate

/-- Theorem stating that the number of crates delivered is 12 --/
theorem crates_delivered_is_twelve :
  crates_delivered 42 4 10 50 = 12 := by
  sorry

end crates_delivered_is_twelve_l2_218


namespace guarantee_target_color_count_l2_206

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  purple : Nat

/-- The initial ball counts in the box -/
def initialBalls : BallCounts :=
  { red := 30, green := 24, yellow := 16, blue := 14, white := 12, purple := 4 }

/-- The minimum number of balls of a single color we want to guarantee -/
def targetCount : Nat := 12

/-- The number of balls we claim will guarantee the target count -/
def claimedDrawCount : Nat := 60

/-- Theorem stating that drawing the claimed number of balls guarantees
    at least the target count of a single color -/
theorem guarantee_target_color_count :
  ∀ (drawn : Nat),
    drawn ≥ claimedDrawCount →
    ∃ (color : Fin 6),
      (match color with
       | 0 => initialBalls.red
       | 1 => initialBalls.green
       | 2 => initialBalls.yellow
       | 3 => initialBalls.blue
       | 4 => initialBalls.white
       | 5 => initialBalls.purple) -
      (claimedDrawCount - drawn) ≥ targetCount :=
by sorry

end guarantee_target_color_count_l2_206


namespace exterior_angle_is_60_l2_251

/-- An isosceles triangle with one angle opposite an equal side being 30 degrees -/
structure IsoscelesTriangle30 where
  /-- The measure of one of the angles opposite an equal side -/
  angle_opposite_equal_side : ℝ
  /-- The measure of the largest angle in the triangle -/
  largest_angle : ℝ
  /-- The fact that the triangle is isosceles with one angle being 30 degrees -/
  is_isosceles_30 : angle_opposite_equal_side = 30

/-- The measure of the exterior angle adjacent to the largest angle in the triangle -/
def exterior_angle (t : IsoscelesTriangle30) : ℝ := 180 - t.largest_angle

/-- Theorem: The measure of the exterior angle adjacent to the largest angle is 60 degrees -/
theorem exterior_angle_is_60 (t : IsoscelesTriangle30) : exterior_angle t = 60 := by
  sorry

end exterior_angle_is_60_l2_251


namespace min_value_abc_l2_269

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 27) :
  54 ≤ 3 * a + 6 * b + 9 * c ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 27 ∧ 3 * a₀ + 6 * b₀ + 9 * c₀ = 54 :=
by sorry

end min_value_abc_l2_269


namespace abs_diff_eq_one_point_one_l2_229

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ :=
  x - Int.floor x

/-- Theorem: Given the conditions, |x - y| = 1.1 -/
theorem abs_diff_eq_one_point_one (x y : ℝ) 
  (h1 : floor x + frac y = 3.7)
  (h2 : frac x + floor y = 4.6) : 
  |x - y| = 1.1 := by
sorry

end abs_diff_eq_one_point_one_l2_229


namespace smallest_positive_angle_same_terminal_side_l2_263

/-- Given an angle α = 2012°, this theorem states that the smallest positive angle 
    with the same terminal side as α is 212°. -/
theorem smallest_positive_angle_same_terminal_side (α : Real) : 
  α = 2012 → ∃ (θ : Real), θ = 212 ∧ 
  θ > 0 ∧ 
  θ < 360 ∧
  ∃ (k : ℤ), α = θ + 360 * k := by
  sorry

end smallest_positive_angle_same_terminal_side_l2_263


namespace classroom_desks_l2_253

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Calculates the number of desks needed given the number of students and students per desk -/
def calculateDesks (students : ℕ) (studentsPerDesk : ℕ) : ℕ := sorry

theorem classroom_desks :
  let studentsBase6 : ℕ := 305
  let studentsPerDesk : ℕ := 3
  let studentsBase10 : ℕ := base6ToBase10 studentsBase6
  calculateDesks studentsBase10 studentsPerDesk = 38 := by sorry

end classroom_desks_l2_253


namespace tan_75_degrees_l2_246

theorem tan_75_degrees : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end tan_75_degrees_l2_246


namespace rectangles_in_4x4_grid_l2_232

/-- The number of rows in the grid -/
def n : ℕ := 4

/-- The number of columns in the grid -/
def m : ℕ := 4

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of rectangles in an n × m grid -/
def num_rectangles (n m : ℕ) : ℕ := choose_two n * choose_two m

theorem rectangles_in_4x4_grid : 
  num_rectangles n m = 36 :=
sorry

end rectangles_in_4x4_grid_l2_232


namespace scientific_notation_3120000_l2_286

theorem scientific_notation_3120000 :
  3120000 = 3.12 * (10 ^ 6) := by
  sorry

end scientific_notation_3120000_l2_286


namespace average_age_of_extreme_new_employees_is_30_l2_275

/-- Represents a company with employees and their ages -/
structure Company where
  initialEmployees : ℕ
  group1Size : ℕ
  group1AvgAge : ℕ
  group2Size : ℕ
  group2AvgAge : ℕ
  group3Size : ℕ
  group3AvgAge : ℕ
  newEmployees : ℕ
  newEmployeesTotalAge : ℕ
  ageDifference : ℕ

/-- Calculates the average age of the youngest and oldest new employees -/
def averageAgeOfExtremeNewEmployees (c : Company) : ℚ :=
  let totalAge := c.group1Size * c.group1AvgAge + c.group2Size * c.group2AvgAge + c.group3Size * c.group3AvgAge
  let totalEmployees := c.initialEmployees + c.newEmployees
  let x := (c.newEmployeesTotalAge - (c.newEmployees - 1) * (c.ageDifference / 2)) / c.newEmployees
  (x + x + c.ageDifference) / 2

/-- Theorem stating that for the given company configuration, 
    the average age of the youngest and oldest new employees is 30 -/
theorem average_age_of_extreme_new_employees_is_30 :
  let c : Company := {
    initialEmployees := 50,
    group1Size := 20,
    group1AvgAge := 30,
    group2Size := 20,
    group2AvgAge := 40,
    group3Size := 10,
    group3AvgAge := 50,
    newEmployees := 5,
    newEmployeesTotalAge := 150,
    ageDifference := 20
  }
  averageAgeOfExtremeNewEmployees c = 30 := by
  sorry

end average_age_of_extreme_new_employees_is_30_l2_275


namespace reflection_line_l2_262

-- Define a Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define the triangle vertices
def P : Point := ⟨-2, 3⟩
def Q : Point := ⟨3, 7⟩
def R : Point := ⟨5, 1⟩

-- Define the reflected triangle vertices
def P' : Point := ⟨-6, 3⟩
def Q' : Point := ⟨-9, 7⟩
def R' : Point := ⟨-11, 1⟩

-- Define the line of reflection
def line_of_reflection (x : ℝ) : Prop :=
  (P.x + P'.x) / 2 = x ∧
  (Q.x + Q'.x) / 2 = x ∧
  (R.x + R'.x) / 2 = x

-- Theorem statement
theorem reflection_line : line_of_reflection (-3) := by
  sorry

end reflection_line_l2_262
