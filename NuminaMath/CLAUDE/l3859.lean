import Mathlib

namespace min_pizzas_to_break_even_l3859_385930

def car_cost : ℕ := 6000
def bag_cost : ℕ := 200
def earning_per_pizza : ℕ := 12
def gas_cost_per_delivery : ℕ := 4

theorem min_pizzas_to_break_even :
  let total_cost := car_cost + bag_cost
  let net_earning_per_pizza := earning_per_pizza - gas_cost_per_delivery
  (∀ n : ℕ, n * net_earning_per_pizza < total_cost → n < 775) ∧
  775 * net_earning_per_pizza ≥ total_cost :=
sorry

end min_pizzas_to_break_even_l3859_385930


namespace baseball_cards_difference_l3859_385992

theorem baseball_cards_difference (jorge matias carlos : ℕ) : 
  jorge = matias → 
  carlos = 20 → 
  jorge + matias + carlos = 48 → 
  carlos - matias = 6 := by
sorry

end baseball_cards_difference_l3859_385992


namespace bicycle_speed_calculation_l3859_385964

theorem bicycle_speed_calculation (distance : ℝ) (speed_difference : ℝ) (time_ratio : ℝ) :
  distance = 10 ∧ 
  speed_difference = 45 ∧ 
  time_ratio = 4 →
  ∃ x : ℝ, x = 15 ∧ 
    distance / x = time_ratio * (distance / (x + speed_difference)) :=
by sorry

end bicycle_speed_calculation_l3859_385964


namespace total_weight_is_20_2_l3859_385939

-- Define the capacities of the jugs
def jug1_capacity : ℝ := 2
def jug2_capacity : ℝ := 3
def jug3_capacity : ℝ := 4

-- Define the fill percentages
def jug1_fill_percent : ℝ := 0.7
def jug2_fill_percent : ℝ := 0.6
def jug3_fill_percent : ℝ := 0.5

-- Define the sand densities
def jug1_density : ℝ := 5
def jug2_density : ℝ := 4
def jug3_density : ℝ := 3

-- Calculate the weight of sand in each jug
def jug1_weight : ℝ := jug1_capacity * jug1_fill_percent * jug1_density
def jug2_weight : ℝ := jug2_capacity * jug2_fill_percent * jug2_density
def jug3_weight : ℝ := jug3_capacity * jug3_fill_percent * jug3_density

-- Total weight of sand in all jugs
def total_weight : ℝ := jug1_weight + jug2_weight + jug3_weight

theorem total_weight_is_20_2 : total_weight = 20.2 := by
  sorry

end total_weight_is_20_2_l3859_385939


namespace unique_solution_condition_l3859_385956

/-- The equation (x + 3) / (mx - 2) = x + 1 has exactly one solution if and only if m = -8 ± 2√15 -/
theorem unique_solution_condition (m : ℝ) : 
  (∃! x : ℝ, (x + 3) / (m * x - 2) = x + 1) ↔ 
  (m = -8 + 2 * Real.sqrt 15 ∨ m = -8 - 2 * Real.sqrt 15) :=
sorry

end unique_solution_condition_l3859_385956


namespace train_crossing_time_l3859_385986

/-- Proves that a train 175 meters long, traveling at 180 km/hr, will take 3.5 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) 
    (h1 : train_length = 175)
    (h2 : train_speed_kmh = 180) : 
  let train_speed_ms : Real := train_speed_kmh * (1000 / 3600)
  let crossing_time : Real := train_length / train_speed_ms
  crossing_time = 3.5 := by
    sorry

#check train_crossing_time

end train_crossing_time_l3859_385986


namespace point_on_y_axis_l3859_385946

/-- A point P with coordinates (2m, m+8) lies on the y-axis if and only if its coordinates are (0, 8) -/
theorem point_on_y_axis (m : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (2*m, m+8) ∧ P.1 = 0) ↔ (∃ (P : ℝ × ℝ), P = (0, 8)) :=
by sorry

end point_on_y_axis_l3859_385946


namespace fry_all_cutlets_in_15_minutes_l3859_385943

/-- Represents a cutlet that needs to be fried -/
structure Cutlet where
  sides_fried : Fin 2 → Bool
  deriving Repr

/-- Represents the state of frying cutlets -/
structure FryingState where
  time : ℕ
  cutlets : Fin 3 → Cutlet
  pan : Fin 2 → Option (Fin 3)
  deriving Repr

/-- Checks if all cutlets are fully fried -/
def all_fried (state : FryingState) : Prop :=
  ∀ i : Fin 3, (state.cutlets i).sides_fried 0 ∧ (state.cutlets i).sides_fried 1

/-- Represents a valid frying step -/
def valid_step (before after : FryingState) : Prop :=
  after.time = before.time + 5 ∧
  (∀ i : Fin 3, 
    (after.cutlets i).sides_fried 0 = (before.cutlets i).sides_fried 0 ∨
    (after.cutlets i).sides_fried 1 = (before.cutlets i).sides_fried 1) ∧
  (∀ i : Fin 2, after.pan i ≠ none → 
    (∃ j : Fin 3, after.pan i = some j ∧ 
      ((before.cutlets j).sides_fried 0 ≠ (after.cutlets j).sides_fried 0 ∨
       (before.cutlets j).sides_fried 1 ≠ (after.cutlets j).sides_fried 1)))

/-- The initial state of frying -/
def initial_state : FryingState := {
  time := 0,
  cutlets := λ _ ↦ { sides_fried := λ _ ↦ false },
  pan := λ _ ↦ none
}

/-- Theorem stating that it's possible to fry all cutlets in 15 minutes -/
theorem fry_all_cutlets_in_15_minutes : 
  ∃ (final_state : FryingState), 
    final_state.time ≤ 15 ∧ 
    all_fried final_state ∧
    ∃ (step1 step2 : FryingState), 
      valid_step initial_state step1 ∧
      valid_step step1 step2 ∧
      valid_step step2 final_state :=
sorry

end fry_all_cutlets_in_15_minutes_l3859_385943


namespace flea_treatment_result_l3859_385929

/-- The number of fleas on a dog after a series of treatments -/
def fleas_after_treatments (initial_fleas : ℕ) (num_treatments : ℕ) : ℕ :=
  initial_fleas / (2^num_treatments)

/-- Theorem: If a dog undergoes four flea treatments, where each treatment halves the number of fleas,
    and the initial number of fleas is 210 more than the final number, then the final number of fleas is 14. -/
theorem flea_treatment_result :
  ∀ F : ℕ,
  (F + 210 = fleas_after_treatments (F + 210) 4) →
  F = 14 :=
by sorry

end flea_treatment_result_l3859_385929


namespace multiply_586645_by_9999_l3859_385933

theorem multiply_586645_by_9999 : 586645 * 9999 = 5865864355 := by
  sorry

end multiply_586645_by_9999_l3859_385933


namespace base_five_3214_equals_434_l3859_385935

def base_five_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem base_five_3214_equals_434 :
  base_five_to_ten [4, 1, 2, 3] = 434 := by
  sorry

end base_five_3214_equals_434_l3859_385935


namespace inequality_problem_l3859_385997

theorem inequality_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y ≤ 4) :
  1 / (x * y) ≥ 1 / 4 := by
  sorry

end inequality_problem_l3859_385997


namespace division_remainder_problem_l3859_385942

theorem division_remainder_problem :
  let dividend : ℕ := 12401
  let divisor : ℕ := 163
  let quotient : ℕ := 76
  dividend = quotient * divisor + 13 :=
by sorry

end division_remainder_problem_l3859_385942


namespace right_triangle_hypotenuse_l3859_385982

theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : a = 90) (h2 : b = 120) 
  (h3 : c^2 = a^2 + b^2) : c = 150 := by
  sorry

end right_triangle_hypotenuse_l3859_385982


namespace g_properties_l3859_385981

def f (n : ℕ) : ℕ := (Nat.factorial n)^2

def g (x : ℕ+) : ℚ := (f (x + 1) : ℚ) / (f x : ℚ)

theorem g_properties :
  (g 1 = 4) ∧
  (g 2 = 9) ∧
  (g 3 = 16) ∧
  (∀ ε > 0, ∃ N : ℕ+, ∀ x ≥ N, g x > ε) :=
sorry

end g_properties_l3859_385981


namespace line_not_in_second_quadrant_l3859_385922

-- Define the line
def line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem line_not_in_second_quadrant :
  ¬ ∃ (x y : ℝ), line x y ∧ second_quadrant x y :=
sorry

end line_not_in_second_quadrant_l3859_385922


namespace average_speed_two_hours_l3859_385909

/-- Calculates the average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) (h1 : speed1 = 90) (h2 : speed2 = 80) :
  (speed1 + speed2) / 2 = 85 := by
  sorry

end average_speed_two_hours_l3859_385909


namespace geometry_class_eligibility_l3859_385925

def minimum_score (s1 s2 s3 s4 : ℝ) : ℝ :=
  let required_average := 85
  let total_required := 5 * required_average
  let current_sum := s1 + s2 + s3 + s4
  total_required - current_sum

theorem geometry_class_eligibility 
  (s1 s2 s3 s4 : ℝ) 
  (h1 : s1 = 86) 
  (h2 : s2 = 82) 
  (h3 : s3 = 80) 
  (h4 : s4 = 84) : 
  minimum_score s1 s2 s3 s4 = 93 := by
  sorry

#eval minimum_score 86 82 80 84

end geometry_class_eligibility_l3859_385925


namespace odd_function_extension_l3859_385921

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_pos : ∀ x > 0, f x = x^2 + 1) :
  ∀ x < 0, f x = -x^2 - 1 := by
sorry

end odd_function_extension_l3859_385921


namespace odd_function_property_l3859_385902

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h_odd : IsOdd f)
    (h_slope : ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 1)
    (m : ℝ) (h_m : f m > m) : m > 0 := by
  sorry

end odd_function_property_l3859_385902


namespace solve_equation_l3859_385904

theorem solve_equation (x : ℝ) (h : Real.sqrt ((2 / x) + 3) = 2) : x = 2 := by
  sorry

end solve_equation_l3859_385904


namespace cos_squared_plus_half_sin_double_l3859_385994

theorem cos_squared_plus_half_sin_double (θ : ℝ) :
  3 * Real.cos (π / 2 - θ) + Real.cos (π + θ) = 0 →
  Real.cos θ ^ 2 + (1 / 2) * Real.sin (2 * θ) = 6 / 5 := by
  sorry

end cos_squared_plus_half_sin_double_l3859_385994


namespace original_number_proof_l3859_385973

theorem original_number_proof : ∃ x : ℝ, x / 12.75 = 16 ∧ x = 204 := by
  sorry

end original_number_proof_l3859_385973


namespace sin_390_degrees_l3859_385980

theorem sin_390_degrees (h1 : ∀ θ, Real.sin (θ + 2 * Real.pi) = Real.sin θ) 
                        (h2 : Real.sin (Real.pi / 6) = 1 / 2) : 
  Real.sin (13 * Real.pi / 6) = 1 / 2 := by
  sorry

end sin_390_degrees_l3859_385980


namespace sqrt_x_minus_3_real_l3859_385996

theorem sqrt_x_minus_3_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by sorry

end sqrt_x_minus_3_real_l3859_385996


namespace trig_expression_equals_eight_thirds_l3859_385990

theorem trig_expression_equals_eight_thirds :
  let sin30 : ℝ := 1/2
  let cos30 : ℝ := Real.sqrt 3 / 2
  (cos30^2 - sin30^2) / (cos30^2 * sin30^2) = 8/3 := by
  sorry

end trig_expression_equals_eight_thirds_l3859_385990


namespace messenger_catches_up_l3859_385907

/-- Represents the scenario of a messenger catching up to Ilya Muromets --/
def catchUpScenario (ilyaSpeed : ℝ) : Prop :=
  let messengerSpeed := 2 * ilyaSpeed
  let horseSpeed := 5 * messengerSpeed
  let initialDelay := 10 -- seconds
  let ilyaDistance := ilyaSpeed * initialDelay
  let horseDistance := horseSpeed * initialDelay
  let totalDistance := ilyaDistance + horseDistance
  let relativeSpeed := messengerSpeed - ilyaSpeed
  let catchUpTime := totalDistance / relativeSpeed
  catchUpTime = 110

/-- Theorem stating that under the given conditions, 
    the messenger catches up to Ilya Muromets in 110 seconds --/
theorem messenger_catches_up (ilyaSpeed : ℝ) (ilyaSpeed_pos : 0 < ilyaSpeed) :
  catchUpScenario ilyaSpeed := by
  sorry

#check messenger_catches_up

end messenger_catches_up_l3859_385907


namespace probability_of_same_color_l3859_385911

/-- Represents a 12-sided die with colored sides -/
structure ColoredDie :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (total_sides : ℕ)
  (side_sum : red + green + blue + yellow = total_sides)

/-- Calculates the probability of two identical dice showing the same color -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.red^2 + d.green^2 + d.blue^2 + d.yellow^2) / d.total_sides^2

/-- The specific 12-sided die described in the problem -/
def problem_die : ColoredDie :=
  { red := 3
    green := 4
    blue := 2
    yellow := 3
    total_sides := 12
    side_sum := by rfl }

theorem probability_of_same_color :
  same_color_probability problem_die = 19 / 72 := by
  sorry

end probability_of_same_color_l3859_385911


namespace parabola_equation_l3859_385906

/-- Given a parabola y^2 = 2px with a point P(2, y_0) on it, and the distance from P to the directrix is 4,
    prove that p = 4 and the standard equation of the parabola is y^2 = 8x. -/
theorem parabola_equation (p : ℝ) (y_0 : ℝ) (h1 : p > 0) (h2 : y_0^2 = 2*p*2) (h3 : p/2 + 2 = 4) :
  p = 4 ∧ ∀ x y, y^2 = 8*x ↔ y^2 = 2*p*x := by sorry

end parabola_equation_l3859_385906


namespace employment_percentage_l3859_385917

theorem employment_percentage (population : ℝ) 
  (h1 : population > 0)
  (h2 : (80 : ℝ) / 100 * population = employed_males)
  (h3 : (1 : ℝ) / 3 * total_employed = employed_females)
  (h4 : employed_males + employed_females = total_employed) :
  total_employed / population = (60 : ℝ) / 100 := by
sorry

end employment_percentage_l3859_385917


namespace remaining_tickets_proof_l3859_385989

def tickets_to_be_sold (total_tickets jude_tickets : ℕ) : ℕ :=
  let andrea_tickets := 6 * jude_tickets
  let sandra_tickets := 3 * jude_tickets + 10
  total_tickets - (jude_tickets + andrea_tickets + sandra_tickets)

theorem remaining_tickets_proof (total_tickets jude_tickets : ℕ) 
  (h1 : total_tickets = 300) 
  (h2 : jude_tickets = 24) : 
  tickets_to_be_sold total_tickets jude_tickets = 50 := by
  sorry

end remaining_tickets_proof_l3859_385989


namespace min_a_value_l3859_385959

theorem min_a_value (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 := by
  sorry

end min_a_value_l3859_385959


namespace number_problem_l3859_385963

theorem number_problem : ∃ x : ℝ, x = 580 ∧ 0.2 * x = 0.3 * 120 + 80 := by
  sorry

end number_problem_l3859_385963


namespace sin_240_degrees_l3859_385926

theorem sin_240_degrees : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l3859_385926


namespace equal_expressions_l3859_385979

theorem equal_expressions : (-2)^3 = -2^3 := by
  sorry

end equal_expressions_l3859_385979


namespace billy_decoration_rate_l3859_385991

/-- The number of eggs Mia can decorate per hour -/
def mia_rate : ℕ := 24

/-- The total number of eggs to be decorated -/
def total_eggs : ℕ := 170

/-- The time taken by Mia and Billy together to decorate all eggs (in hours) -/
def total_time : ℕ := 5

/-- Billy's decoration rate (in eggs per hour) -/
def billy_rate : ℕ := total_eggs / total_time - mia_rate

theorem billy_decoration_rate :
  billy_rate = 10 := by sorry

end billy_decoration_rate_l3859_385991


namespace tree_height_problem_l3859_385962

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 20 →  -- One tree is 20 feet taller than the other
  h₂ / h₁ = 5 / 7 →  -- The heights are in the ratio 5:7
  h₁ = 70 :=  -- The height of the taller tree is 70 feet
by sorry

end tree_height_problem_l3859_385962


namespace percentage_calculation_l3859_385978

theorem percentage_calculation : (168 / 100 * 1265) / 6 = 354.2 := by
  sorry

end percentage_calculation_l3859_385978


namespace partnership_profit_l3859_385903

/-- Represents a partnership between two individuals -/
structure Partnership where
  investmentA : ℕ
  investmentB : ℕ
  periodA : ℕ
  periodB : ℕ

/-- Calculates the total profit of a partnership -/
def totalProfit (p : Partnership) (profitB : ℕ) : ℕ :=
  7 * profitB

theorem partnership_profit (p : Partnership) (h1 : p.investmentA = 3 * p.investmentB)
    (h2 : p.periodA = 2 * p.periodB) (h3 : 4000 = p.investmentB * p.periodB) :
  totalProfit p 4000 = 28000 := by
  sorry

#eval totalProfit ⟨30000, 10000, 10, 5⟩ 4000

end partnership_profit_l3859_385903


namespace inscribed_circles_area_l3859_385967

theorem inscribed_circles_area (R : ℝ) (d : ℝ) : 
  R = 10 ∧ d = 6 → 
  let h := R - d / 2
  let r := R - d / 2
  2 * Real.pi * r^2 = 98 * Real.pi :=
by
  sorry

end inscribed_circles_area_l3859_385967


namespace revenue_difference_l3859_385913

/-- The revenue generated by a single jersey -/
def jersey_revenue : ℕ := 210

/-- The revenue generated by a single t-shirt -/
def tshirt_revenue : ℕ := 240

/-- The number of t-shirts sold -/
def tshirts_sold : ℕ := 177

/-- The number of jerseys sold -/
def jerseys_sold : ℕ := 23

/-- The difference in revenue between t-shirts and jerseys -/
theorem revenue_difference : 
  tshirt_revenue * tshirts_sold - jersey_revenue * jerseys_sold = 37650 := by
  sorry

end revenue_difference_l3859_385913


namespace arctan_less_arcsin_iff_l3859_385970

theorem arctan_less_arcsin_iff (x : ℝ) : Real.arctan x < Real.arcsin x ↔ -1 < x ∧ x ≤ 0 := by
  sorry

end arctan_less_arcsin_iff_l3859_385970


namespace arithmetic_sequence_sum_l3859_385923

/-- Given an arithmetic sequence, if the sum of the first n terms is P 
    and the sum of the first 2n terms is q, then the sum of the first 3n terms is 3(2P - q). -/
theorem arithmetic_sequence_sum (n : ℕ) (P q : ℝ) :
  (∃ (a d : ℝ), P = n / 2 * (2 * a + (n - 1) * d) ∧ q = n * (2 * a + (2 * n - 1) * d)) →
  (∃ (S_3n : ℝ), S_3n = 3 * (2 * P - q)) :=
sorry


end arithmetic_sequence_sum_l3859_385923


namespace sand_box_fill_time_l3859_385993

/-- The time required to fill a rectangular box with sand -/
theorem sand_box_fill_time
  (length width height : ℝ)
  (fill_rate : ℝ)
  (h_length : length = 7)
  (h_width : width = 6)
  (h_height : height = 2)
  (h_fill_rate : fill_rate = 4)
  : (length * width * height) / fill_rate = 21 := by
  sorry

end sand_box_fill_time_l3859_385993


namespace total_money_l3859_385949

def sam_money : ℕ := 38
def erica_money : ℕ := 53

theorem total_money : sam_money + erica_money = 91 := by
  sorry

end total_money_l3859_385949


namespace intersection_of_P_and_M_l3859_385944

-- Define the sets P and M
def P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def M : Set ℝ := {x | x^2 ≤ 9}

-- State the theorem
theorem intersection_of_P_and_M : P ∩ M = {x | 0 ≤ x ∧ x < 3} := by sorry

end intersection_of_P_and_M_l3859_385944


namespace map_scale_l3859_385958

/-- Given a map scale where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale (map_cm : ℝ) (real_km : ℝ) (h : map_cm = 15 ∧ real_km = 90) :
  (20 : ℝ) * (real_km / map_cm) = 120 := by
  sorry

end map_scale_l3859_385958


namespace orange_juice_problem_l3859_385952

theorem orange_juice_problem (jug_volume : ℚ) (portion_drunk : ℚ) :
  jug_volume = 2/7 →
  portion_drunk = 5/8 →
  portion_drunk * jug_volume = 5/28 := by
  sorry

end orange_juice_problem_l3859_385952


namespace front_parking_spaces_l3859_385941

theorem front_parking_spaces (back_spaces : ℕ) (total_parked : ℕ) (available_spaces : ℕ)
  (h1 : back_spaces = 38)
  (h2 : total_parked = 39)
  (h3 : available_spaces = 32)
  (h4 : back_spaces / 2 + available_spaces + total_parked = back_spaces + front_spaces) :
  front_spaces = 33 := by
  sorry

end front_parking_spaces_l3859_385941


namespace tim_balloon_count_l3859_385953

theorem tim_balloon_count (dan_balloons : ℕ) (tim_multiplier : ℕ) (h1 : dan_balloons = 29) (h2 : tim_multiplier = 7) : 
  dan_balloons * tim_multiplier = 203 := by
  sorry

end tim_balloon_count_l3859_385953


namespace sum_of_radii_l3859_385951

/-- A circle with center C(r, r) is tangent to the positive x-axis and y-axis,
    and externally tangent to a circle centered at (4,0) with radius 2. -/
def CircleTangency (r : ℝ) : Prop :=
  r > 0 ∧ (r - 4)^2 + r^2 = (r + 2)^2

/-- The sum of all possible radii of the circle with center C is 12. -/
theorem sum_of_radii :
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ CircleTangency r₁ ∧ CircleTangency r₂ ∧ r₁ + r₂ = 12 :=
sorry

end sum_of_radii_l3859_385951


namespace non_shaded_perimeter_l3859_385969

/-- Given a rectangle with dimensions 15 inches by 10 inches and area 150 square inches,
    containing a shaded rectangle with area 110 square inches,
    the perimeter of the non-shaded region is 26 inches. -/
theorem non_shaded_perimeter (large_width large_height : ℝ)
                              (large_area shaded_area : ℝ)
                              (non_shaded_width non_shaded_height : ℝ) :
  large_width = 15 →
  large_height = 10 →
  large_area = 150 →
  shaded_area = 110 →
  large_area = large_width * large_height →
  non_shaded_width * non_shaded_height = large_area - shaded_area →
  non_shaded_width ≤ large_width →
  non_shaded_height ≤ large_height →
  2 * (non_shaded_width + non_shaded_height) = 26 :=
by sorry

end non_shaded_perimeter_l3859_385969


namespace hyperbola_condition_equivalence_l3859_385977

/-- The equation represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 1) + y^2 / (k + 2) = 1 ∧ (k - 1) * (k + 2) < 0

/-- The condition 0 < k < 1 -/
def condition (k : ℝ) : Prop := 0 < k ∧ k < 1

theorem hyperbola_condition_equivalence :
  ∀ k : ℝ, is_hyperbola k ↔ condition k := by sorry

end hyperbola_condition_equivalence_l3859_385977


namespace distinct_dice_designs_count_l3859_385919

/-- Represents a dice design -/
structure DiceDesign where
  -- We don't need to explicitly define the structure,
  -- as we're only concerned with the count of distinct designs

/-- The number of ways to choose 2 numbers from 4 -/
def choose_two_from_four : Nat := 6

/-- The number of ways to arrange the chosen numbers on opposite faces -/
def arrangement_ways : Nat := 2

/-- The number of ways to color three pairs of opposite faces -/
def coloring_ways : Nat := 8

/-- The total number of distinct dice designs -/
def distinct_dice_designs : Nat := 
  (choose_two_from_four * arrangement_ways / 2) * coloring_ways

theorem distinct_dice_designs_count :
  distinct_dice_designs = 48 := by
  sorry

end distinct_dice_designs_count_l3859_385919


namespace distribute_five_books_three_students_l3859_385940

/-- The number of ways to distribute n different books among k students,
    with each student receiving at least one book -/
def distribute_books (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 different books among 3 students,
    with each student receiving at least one book, is 150 -/
theorem distribute_five_books_three_students :
  distribute_books 5 3 = 150 := by sorry

end distribute_five_books_three_students_l3859_385940


namespace regular_polygon_interior_angle_sum_l3859_385947

theorem regular_polygon_interior_angle_sum 
  (n : ℕ) 
  (h_exterior : (360 : ℝ) / n = 45) : 
  (n - 2 : ℝ) * 180 = 1080 := by
  sorry

end regular_polygon_interior_angle_sum_l3859_385947


namespace dans_remaining_money_is_14_02_l3859_385910

/-- Calculates the remaining money after Dan's shopping trip -/
def dans_remaining_money (initial_money : ℚ) (candy_price : ℚ) (candy_count : ℕ) 
  (toy_price : ℚ) (toy_discount : ℚ) (sales_tax : ℚ) : ℚ :=
  let candy_total := candy_price * candy_count
  let discounted_toy := toy_price * (1 - toy_discount)
  let subtotal := candy_total + discounted_toy
  let total_with_tax := subtotal * (1 + sales_tax)
  initial_money - total_with_tax

/-- Theorem stating that Dan's remaining money after shopping is $14.02 -/
theorem dans_remaining_money_is_14_02 :
  dans_remaining_money 45 4 4 15 0.1 0.05 = 14.02 := by
  sorry

#eval dans_remaining_money 45 4 4 15 0.1 0.05

end dans_remaining_money_is_14_02_l3859_385910


namespace sum_of_coefficients_l3859_385915

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (k : ℚ), k * (a.val * Real.sqrt 6 + b.val * Real.sqrt 8) / c.val = 
    Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) →
  (∀ (x y z : ℕ+), (∃ (l : ℚ), l * (x.val * Real.sqrt 6 + y.val * Real.sqrt 8) / z.val = 
    Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) → 
    c.val ≤ z.val) →
  a.val + b.val + c.val = 192 := by
sorry

end sum_of_coefficients_l3859_385915


namespace marble_distribution_l3859_385900

theorem marble_distribution (x : ℚ) 
  (h1 : (5 * x + 2) + 2 * x + 4 * x = 88) : x = 86 / 11 := by
  sorry

end marble_distribution_l3859_385900


namespace point_reflection_origin_l3859_385918

/-- Given a point P(4, -3) in the Cartesian coordinate system,
    its coordinates with respect to the origin are (-4, 3). -/
theorem point_reflection_origin : 
  let P : ℝ × ℝ := (4, -3)
  let P_reflected : ℝ × ℝ := (-4, 3)
  P_reflected = (-(P.1), -(P.2)) :=
by sorry

end point_reflection_origin_l3859_385918


namespace master_bedroom_size_l3859_385920

theorem master_bedroom_size 
  (master_bath : ℝ) 
  (new_room : ℝ) 
  (h1 : master_bath = 150) 
  (h2 : new_room = 918) 
  (h3 : new_room = 2 * (master_bedroom + master_bath)) : 
  master_bedroom = 309 :=
by
  sorry

end master_bedroom_size_l3859_385920


namespace permutations_of_four_distinct_elements_l3859_385983

theorem permutations_of_four_distinct_elements : 
  Nat.factorial 4 = 24 := by
  sorry

end permutations_of_four_distinct_elements_l3859_385983


namespace binomial_60_3_l3859_385955

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l3859_385955


namespace sabrina_leaves_l3859_385984

/-- The number of basil leaves Sabrina needs -/
def basil : ℕ := 12

/-- The number of sage leaves Sabrina needs -/
def sage : ℕ := basil / 2

/-- The number of verbena leaves Sabrina needs -/
def verbena : ℕ := sage + 5

/-- The total number of leaves Sabrina needs -/
def total : ℕ := basil + sage + verbena

theorem sabrina_leaves : total = 29 := by
  sorry

end sabrina_leaves_l3859_385984


namespace chameleons_multiple_colors_l3859_385901

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The initial state of chameleons on the island -/
def initial_state : ChameleonState :=
  { red := 155, blue := 49, green := 96 }

/-- Defines the color change rule for chameleons -/
def color_change_rule (state : ChameleonState) : ChameleonState → Prop :=
  λ new_state =>
    (new_state.red + new_state.blue + new_state.green = state.red + state.blue + state.green) ∧
    (new_state.red - new_state.blue) % 3 = (state.red - state.blue) % 3 ∧
    (new_state.blue - new_state.green) % 3 = (state.blue - state.green) % 3 ∧
    (new_state.red - new_state.green) % 3 = (state.red - state.green) % 3

/-- Theorem stating that it's impossible for all chameleons to be the same color -/
theorem chameleons_multiple_colors (final_state : ChameleonState) :
  color_change_rule initial_state final_state →
  ¬(final_state.red = 0 ∧ final_state.blue = 0) ∧
  ¬(final_state.red = 0 ∧ final_state.green = 0) ∧
  ¬(final_state.blue = 0 ∧ final_state.green = 0) :=
by sorry

end chameleons_multiple_colors_l3859_385901


namespace exists_all_strawberry_day_l3859_385995

-- Define the type for our matrix
def WorkSchedule := Matrix (Fin 7) (Fin 16) Bool

-- Define the conditions
def first_day_all_mine (schedule : WorkSchedule) : Prop :=
  ∀ i : Fin 7, schedule i 0 = false

def at_least_three_different (schedule : WorkSchedule) : Prop :=
  ∀ j k : Fin 16, j ≠ k → 
    (∃ (s : Finset (Fin 7)), s.card ≥ 3 ∧ 
      (∀ i ∈ s, schedule i j ≠ schedule i k))

-- The main theorem
theorem exists_all_strawberry_day (schedule : WorkSchedule) 
  (h1 : first_day_all_mine schedule)
  (h2 : at_least_three_different schedule) : 
  ∃ j : Fin 16, ∀ i : Fin 7, schedule i j = true :=
sorry

end exists_all_strawberry_day_l3859_385995


namespace lisa_age_l3859_385998

theorem lisa_age :
  ∀ (L N : ℕ),
  L = N + 8 →
  L - 2 = 3 * (N - 2) →
  L = 14 :=
by sorry

end lisa_age_l3859_385998


namespace quadratic_properties_l3859_385912

-- Define the quadratic function
def quadratic (x b c : ℝ) : ℝ := x^2 + b*x + c

-- Define the points
def point_A : ℝ × ℝ := (2, 0)
def point_B (n y1 : ℝ) : ℝ × ℝ := (3*n - 4, y1)
def point_C (n y2 : ℝ) : ℝ × ℝ := (5*n + 6, y2)

theorem quadratic_properties (b c n y1 y2 : ℝ) 
  (h1 : quadratic 2 b c = 0)  -- A(2,0) is on the curve
  (h2 : quadratic (3*n - 4) b c = y1)  -- B is on the curve
  (h3 : quadratic (5*n + 6) b c = y2)  -- C is on the curve
  (h4 : ∀ x, quadratic x b c ≥ quadratic 2 b c)  -- A is the vertex
  (h5 : n < -5) :  -- Given condition
  -- 1) The function can be expressed as y = x^2 - 4x + 4
  (∀ x, quadratic x b c = x^2 - 4*x + 4) ∧
  -- 2) If y1 = y2, then b+c < -38
  (y1 = y2 → b + c < -38) ∧
  -- 3) If c > 0, then y1 < y2
  (c > 0 → y1 < y2) := by
  sorry

end quadratic_properties_l3859_385912


namespace car_A_original_speed_l3859_385966

/-- Represents the speed and position of a car --/
structure Car where
  speed : ℝ
  position : ℝ

/-- Represents the scenario of two cars meeting --/
structure MeetingScenario where
  carA : Car
  carB : Car
  meetingTime : ℝ
  meetingPosition : ℝ

/-- The original scenario where cars meet at point C --/
def originalScenario : MeetingScenario := sorry

/-- Scenario where car B increases speed by 5 km/h --/
def scenarioBFaster : MeetingScenario := sorry

/-- Scenario where car A increases speed by 5 km/h --/
def scenarioAFaster : MeetingScenario := sorry

theorem car_A_original_speed :
  ∃ (s : ℝ),
    (originalScenario.carA.speed = s) ∧
    (originalScenario.meetingTime = 6) ∧
    (scenarioBFaster.carB.speed = originalScenario.carB.speed + 5) ∧
    (scenarioBFaster.meetingPosition = originalScenario.meetingPosition - 12) ∧
    (scenarioAFaster.carA.speed = originalScenario.carA.speed + 5) ∧
    (scenarioAFaster.meetingPosition = originalScenario.meetingPosition + 16) ∧
    (s = 30) := by
  sorry

end car_A_original_speed_l3859_385966


namespace fraction_sum_equality_l3859_385965

theorem fraction_sum_equality (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_sum : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 
  1 / (b - c)^2 + 1 / (c - a)^2 + 1 / (a - b)^2 := by
  sorry

end fraction_sum_equality_l3859_385965


namespace triangle_properties_l3859_385916

theorem triangle_properties (a b c : ℝ) 
  (h : |a - Real.sqrt 7| + Real.sqrt (b - 5) + (c - 4 * Real.sqrt 2)^2 = 0) :
  a = Real.sqrt 7 ∧ 
  b = 5 ∧ 
  c = 4 * Real.sqrt 2 ∧
  a^2 + b^2 = c^2 ∧
  a + b > c ∧
  b + c > a ∧
  c + a > b := by
sorry

end triangle_properties_l3859_385916


namespace factorial_1200_trailing_zeroes_l3859_385928

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- Theorem: 1200! has 298 trailing zeroes -/
theorem factorial_1200_trailing_zeroes :
  trailingZeroes 1200 = 298 := by
  sorry

end factorial_1200_trailing_zeroes_l3859_385928


namespace exists_valid_path_2020_l3859_385971

/-- Represents a square grid with diagonals drawn in each cell. -/
structure DiagonalGrid (n : ℕ) where
  size : n > 0

/-- Represents a path on the diagonal grid. -/
structure DiagonalPath (n : ℕ) where
  grid : DiagonalGrid n
  is_closed : Bool
  visits_all_cells : Bool
  no_repeated_diagonals : Bool

/-- Theorem stating the existence of a valid path in a 2020x2020 grid. -/
theorem exists_valid_path_2020 :
  ∃ (path : DiagonalPath 2020),
    path.is_closed ∧
    path.visits_all_cells ∧
    path.no_repeated_diagonals :=
  sorry

end exists_valid_path_2020_l3859_385971


namespace exists_nonnegative_product_polynomial_l3859_385938

theorem exists_nonnegative_product_polynomial (f : Polynomial ℝ) 
  (h_no_nonneg_root : ∀ x : ℝ, x ≥ 0 → f.eval x ≠ 0) :
  ∃ h : Polynomial ℝ, ∀ i : ℕ, (f * h).coeff i ≥ 0 := by
  sorry

end exists_nonnegative_product_polynomial_l3859_385938


namespace circular_seating_arrangement_l3859_385905

/-- Given a circular seating arrangement where:
    - There is equal spacing between all positions
    - The 6th person is directly opposite the 16th person
    - One position is reserved for a teacher
    Prove that the total number of students (excluding the teacher) is 20. -/
theorem circular_seating_arrangement (n : ℕ) 
  (h1 : ∃ (teacher_pos : ℕ), teacher_pos ≤ n + 1) 
  (h2 : (6 + n/2) % (n + 1) = 16 % (n + 1)) : n = 20 := by
  sorry

end circular_seating_arrangement_l3859_385905


namespace water_heater_capacity_l3859_385985

/-- Represents a water heater with given parameters -/
structure WaterHeater where
  initialCapacity : ℝ
  addRate : ℝ → ℝ
  dischargeRate : ℝ → ℝ
  maxPersonUsage : ℝ

/-- Calculates the water volume as a function of time -/
def waterVolume (heater : WaterHeater) (t : ℝ) : ℝ :=
  heater.initialCapacity + heater.addRate t - heater.dischargeRate t

/-- Theorem: The given water heater can supply at least 4 people for continuous showers -/
theorem water_heater_capacity (heater : WaterHeater) 
  (h1 : heater.initialCapacity = 200)
  (h2 : ∀ t, heater.addRate t = 2 * t^2)
  (h3 : ∀ t, heater.dischargeRate t = 34 * t)
  (h4 : heater.maxPersonUsage = 60) :
  ∃ n : ℕ, n ≥ 4 ∧ 
    (∃ t : ℝ, t > 0 ∧ 
      heater.dischargeRate t / heater.maxPersonUsage ≥ n ∧
      ∀ s, 0 ≤ s ∧ s ≤ t → waterVolume heater s ≥ 0) :=
by sorry

end water_heater_capacity_l3859_385985


namespace angle_division_l3859_385987

theorem angle_division (α : ℝ) (n : ℕ) (h1 : α = 19) (h2 : n = 19) :
  α / n = 1 := by
  sorry

end angle_division_l3859_385987


namespace john_vacation_money_l3859_385972

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ :=
  sorry

/-- Calculates the remaining money after buying a ticket -/
def remainingMoney (savings : ℕ) (ticketCost : ℕ) : ℕ :=
  savings - ticketCost

theorem john_vacation_money :
  let savings := base8ToBase10 5555
  let ticketCost := 1200
  remainingMoney savings ticketCost = 1725 := by
  sorry

end john_vacation_money_l3859_385972


namespace triangle_side_length_l3859_385931

theorem triangle_side_length (a b x : ℝ) : 
  a = 2 → 
  b = 6 → 
  x^2 - 10*x + 21 = 0 → 
  x > 0 → 
  a + x > b → 
  b + x > a → 
  a + b > x → 
  x = 7 := by sorry

end triangle_side_length_l3859_385931


namespace games_last_month_l3859_385960

def games_this_month : ℕ := 9
def games_next_month : ℕ := 7
def total_games : ℕ := 24

theorem games_last_month : total_games - (games_this_month + games_next_month) = 8 := by
  sorry

end games_last_month_l3859_385960


namespace binomial_probability_two_successes_l3859_385932

/-- A random variable X follows a binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ

/-- The probability mass function of a binomial distribution -/
def probability_mass_function (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

/-- Theorem: For a binomial distribution B(4, 1/2), P(X=2) = 3/8 -/
theorem binomial_probability_two_successes :
  ∀ (X : BinomialDistribution 4 (1/2)),
  probability_mass_function 4 (1/2) 2 = 3/8 := by
  sorry

end binomial_probability_two_successes_l3859_385932


namespace camp_cedar_counselors_l3859_385999

/-- The number of counselors needed at Camp Cedar --/
def counselors_needed (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  (num_boys / 6) + (num_girls / 10)

/-- Theorem: Camp Cedar needs 26 counselors --/
theorem camp_cedar_counselors :
  let num_boys : ℕ := 48
  let num_girls : ℕ := 4 * num_boys - 12
  counselors_needed num_boys num_girls = 26 := by
  sorry


end camp_cedar_counselors_l3859_385999


namespace divisors_of_2744_l3859_385950

-- Define 2744 as the number we're interested in
def n : ℕ := 2744

-- Define the function that counts the number of positive divisors
def count_divisors (m : ℕ) : ℕ := (Finset.filter (· ∣ m) (Finset.range (m + 1))).card

-- State the theorem
theorem divisors_of_2744 : count_divisors n = 16 := by sorry

end divisors_of_2744_l3859_385950


namespace cyclic_inequality_l3859_385954

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (x^3 / (x^2 + y)) + (y^3 / (y^2 + z)) + (z^3 / (z^2 + x)) ≥ 3/2 := by
  sorry

end cyclic_inequality_l3859_385954


namespace certain_number_proof_l3859_385936

theorem certain_number_proof : ∃ x : ℝ, 45 * 12 = 0.60 * x ∧ x = 900 := by
  sorry

end certain_number_proof_l3859_385936


namespace sphere_volume_in_cube_l3859_385945

/-- Given a cube with edge length a and two congruent spheres inscribed in opposite trihedral angles
    that touch each other, this theorem states the volume of each sphere. -/
theorem sphere_volume_in_cube (a : ℝ) (a_pos : 0 < a) : 
  ∃ (r : ℝ), r = (3 * a - a * Real.sqrt 3) / 4 ∧ 
              (4 / 3 : ℝ) * Real.pi * r^3 = (4 / 3 : ℝ) * Real.pi * ((3 * a - a * Real.sqrt 3) / 4)^3 := by
  sorry

end sphere_volume_in_cube_l3859_385945


namespace no_integer_pairs_with_square_diff_30_l3859_385948

theorem no_integer_pairs_with_square_diff_30 :
  ¬∃ (m n : ℕ), m ≥ n ∧ m * m - n * n = 30 := by
  sorry

end no_integer_pairs_with_square_diff_30_l3859_385948


namespace hyperbola_a_value_l3859_385976

theorem hyperbola_a_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (b / a = 2) →
  (a^2 + b^2 = 20) →
  a = 1 := by
sorry

end hyperbola_a_value_l3859_385976


namespace planes_parallel_to_line_are_parallel_planes_parallel_to_plane_are_parallel_l3859_385975

-- Define a type for planes
variable (Plane : Type)

-- Define a type for lines
variable (Line : Type)

-- Define a relation for parallelism between planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define a relation for parallelism between a plane and a line
variable (parallel_plane_line : Plane → Line → Prop)

-- Define a relation for parallelism between a plane and another plane
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Theorem 1: Two planes parallel to the same line are parallel
theorem planes_parallel_to_line_are_parallel 
  (P Q : Plane) (L : Line) 
  (h1 : parallel_plane_line P L) 
  (h2 : parallel_plane_line Q L) : 
  parallel_planes P Q :=
sorry

-- Theorem 2: Two planes parallel to the same plane are parallel
theorem planes_parallel_to_plane_are_parallel 
  (P Q R : Plane) 
  (h1 : parallel_plane_plane P R) 
  (h2 : parallel_plane_plane Q R) : 
  parallel_planes P Q :=
sorry

end planes_parallel_to_line_are_parallel_planes_parallel_to_plane_are_parallel_l3859_385975


namespace symmetry_implies_sum_l3859_385974

/-- Two points are symmetric about the y-axis if their x-coordinates are opposite and y-coordinates are equal -/
def symmetric_about_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = -x2 ∧ y1 = y2

theorem symmetry_implies_sum (a b : ℝ) :
  symmetric_about_y_axis (a + 1) 3 (-2) (b + 2) →
  a + b = 2 := by
  sorry

end symmetry_implies_sum_l3859_385974


namespace dress_designs_count_l3859_385914

/-- The number of available fabric colors -/
def num_colors : ℕ := 3

/-- The number of available patterns -/
def num_patterns : ℕ := 4

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_patterns

/-- Theorem stating that the total number of possible dress designs is 12 -/
theorem dress_designs_count : total_designs = 12 := by
  sorry

end dress_designs_count_l3859_385914


namespace fish_purchase_total_l3859_385968

theorem fish_purchase_total (yesterday_fish : ℕ) (yesterday_cost : ℕ) (today_extra_cost : ℕ) : 
  yesterday_fish = 10 →
  yesterday_cost = 3000 →
  today_extra_cost = 6000 →
  ∃ (today_fish : ℕ), 
    (yesterday_fish + today_fish = 40 ∧ 
     yesterday_cost + today_extra_cost = (yesterday_cost / yesterday_fish) * (yesterday_fish + today_fish)) := by
  sorry

#check fish_purchase_total

end fish_purchase_total_l3859_385968


namespace prob_odd_sum_is_half_l3859_385908

/-- Represents a wheel with numbers from 1 to n -/
def Wheel (n : ℕ) := Finset (Fin n)

/-- The probability of selecting an odd number from a wheel -/
def prob_odd (w : Wheel n) : ℚ :=
  (w.filter (λ x => x.val % 2 = 1)).card / w.card

/-- The probability of selecting an even number from a wheel -/
def prob_even (w : Wheel n) : ℚ :=
  (w.filter (λ x => x.val % 2 = 0)).card / w.card

/-- The first wheel with numbers 1 to 5 -/
def wheel1 : Wheel 5 := Finset.univ

/-- The second wheel with numbers 1 to 4 -/
def wheel2 : Wheel 4 := Finset.univ

theorem prob_odd_sum_is_half :
  prob_odd wheel1 * prob_even wheel2 + prob_even wheel1 * prob_odd wheel2 = 1/2 := by
  sorry

end prob_odd_sum_is_half_l3859_385908


namespace trigonometric_problem_l3859_385988

open Real

theorem trigonometric_problem (α β : ℝ) 
  (h1 : sin α = (4 * Real.sqrt 3) / 7)
  (h2 : cos (β - α) = 13 / 14)
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π / 2) : 
  tan (2 * α) = -(8 * Real.sqrt 3) / 47 ∧ cos β = 1 / 2 := by
  sorry

end trigonometric_problem_l3859_385988


namespace janet_needs_775_l3859_385937

/-- The amount of additional money Janet needs to rent an apartment -/
def additional_money_needed (savings : ℕ) (monthly_rent : ℕ) (advance_months : ℕ) (deposit : ℕ) : ℕ :=
  (monthly_rent * advance_months + deposit) - savings

/-- Proof that Janet needs $775 more to rent the apartment -/
theorem janet_needs_775 : 
  additional_money_needed 2225 1250 2 500 = 775 := by
  sorry

end janet_needs_775_l3859_385937


namespace sufficient_not_necessary_l3859_385961

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end sufficient_not_necessary_l3859_385961


namespace equation_one_solutions_equation_two_solution_l3859_385934

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  4 * (x - 1)^2 = 25 ↔ x = 7/2 ∨ x = -3/2 :=
sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  1/3 * (x + 2)^3 - 9 = 0 ↔ x = 1 :=
sorry

end equation_one_solutions_equation_two_solution_l3859_385934


namespace binomial_expansion_sum_l3859_385924

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 233 := by
sorry

end binomial_expansion_sum_l3859_385924


namespace other_side_length_l3859_385927

/-- Represents a right triangle with given side lengths -/
structure RightTriangle where
  hypotenuse : ℝ
  side1 : ℝ
  side2 : ℝ
  hypotenuse_positive : hypotenuse > 0
  side1_positive : side1 > 0
  side2_positive : side2 > 0
  pythagorean : hypotenuse^2 = side1^2 + side2^2

/-- The length of the other side in a right triangle with hypotenuse 10 and one side 6 is 8 -/
theorem other_side_length (t : RightTriangle) (h1 : t.hypotenuse = 10) (h2 : t.side1 = 6) :
  t.side2 = 8 := by
  sorry

end other_side_length_l3859_385927


namespace subtract_product_equality_l3859_385957

theorem subtract_product_equality : 7899665 - 12 * 3 * 2 = 7899593 := by
  sorry

end subtract_product_equality_l3859_385957
