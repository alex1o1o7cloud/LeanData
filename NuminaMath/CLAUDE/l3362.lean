import Mathlib

namespace speed_equivalence_l3362_336238

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in m/s -/
def given_speed_mps : ℝ := 15.001199999999999

/-- The calculated speed in km/h -/
def calculated_speed_kmph : ℝ := 54.004319999999996

/-- Theorem stating that the calculated speed in km/h is equivalent to the given speed in m/s -/
theorem speed_equivalence : calculated_speed_kmph = given_speed_mps * mps_to_kmph := by
  sorry

#check speed_equivalence

end speed_equivalence_l3362_336238


namespace triangular_front_view_solids_l3362_336262

/-- Enumeration of possible solids --/
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

/-- Definition of a solid having a triangular front view --/
def hasTriangularFrontView (s : Solid) : Prop :=
  match s with
  | Solid.TriangularPyramid => True
  | Solid.SquarePyramid => True
  | Solid.TriangularPrism => True  -- Assuming it can be laid on its side
  | Solid.Cone => True
  | _ => False

/-- Theorem stating that a solid with a triangular front view must be one of the specified solids --/
theorem triangular_front_view_solids (s : Solid) :
  hasTriangularFrontView s →
  s = Solid.TriangularPyramid ∨
  s = Solid.SquarePyramid ∨
  s = Solid.TriangularPrism ∨
  s = Solid.Cone :=
by
  sorry

end triangular_front_view_solids_l3362_336262


namespace hexagon_triangles_l3362_336222

/-- The number of triangles that can be formed from a regular hexagon and its center -/
def num_triangles_hexagon : ℕ :=
  let total_points : ℕ := 7
  let total_combinations : ℕ := Nat.choose total_points 3
  let invalid_triangles : ℕ := 3
  total_combinations - invalid_triangles

theorem hexagon_triangles :
  num_triangles_hexagon = 32 := by
  sorry

end hexagon_triangles_l3362_336222


namespace sequence1_correct_sequence2_correct_sequence3_correct_l3362_336236

-- Sequence 1
def sequence1 (n : ℕ) : ℤ := (-1)^n * (6*n - 5)

-- Sequence 2
def sequence2 (n : ℕ) : ℚ := 8/9 * (1 - 1/10^n)

-- Sequence 3
def sequence3 (n : ℕ) : ℚ := (-1)^n * (2^n - 3) / 2^n

theorem sequence1_correct (n : ℕ) : 
  sequence1 1 = -1 ∧ sequence1 2 = 7 ∧ sequence1 3 = -13 ∧ sequence1 4 = 19 := by sorry

theorem sequence2_correct (n : ℕ) : 
  sequence2 1 = 0.8 ∧ sequence2 2 = 0.88 ∧ sequence2 3 = 0.888 := by sorry

theorem sequence3_correct (n : ℕ) : 
  sequence3 1 = -1/2 ∧ sequence3 2 = 1/4 ∧ sequence3 3 = -5/8 ∧ 
  sequence3 4 = 13/16 ∧ sequence3 5 = -29/32 ∧ sequence3 6 = 61/64 := by sorry

end sequence1_correct_sequence2_correct_sequence3_correct_l3362_336236


namespace total_bricks_used_l3362_336241

/-- The number of brick walls -/
def total_walls : ℕ := 10

/-- The number of walls of the first type -/
def first_type_walls : ℕ := 5

/-- The number of walls of the second type -/
def second_type_walls : ℕ := 5

/-- The number of bricks in a single row for the first type of wall -/
def first_type_bricks_per_row : ℕ := 60

/-- The number of rows in the first type of wall -/
def first_type_rows : ℕ := 100

/-- The number of bricks in a single row for the second type of wall -/
def second_type_bricks_per_row : ℕ := 80

/-- The number of rows in the second type of wall -/
def second_type_rows : ℕ := 120

/-- Theorem: The total number of bricks used for all ten walls is 78000 -/
theorem total_bricks_used : 
  first_type_walls * first_type_bricks_per_row * first_type_rows +
  second_type_walls * second_type_bricks_per_row * second_type_rows = 78000 :=
by
  sorry

end total_bricks_used_l3362_336241


namespace palindrome_square_base_l3362_336228

theorem palindrome_square_base (r : ℕ) (x : ℕ) (n : ℕ) : 
  r > 3 →
  (∃ (p : ℕ), x = p * r^3 + p * r^2 + 2*p * r + 2*p) →
  (∃ (a b c : ℕ), x^2 = a * r^6 + b * r^5 + c * r^4 + c * r^3 + c * r^2 + b * r + a) →
  r = 3 * n^2 ∧ n > 1 :=
by sorry

end palindrome_square_base_l3362_336228


namespace complex_equation_solution_l3362_336226

theorem complex_equation_solution (m : ℝ) (i : ℂ) : 
  i * i = -1 → (m + 2 * i) * (2 - i) = 4 + 3 * i → m = 1 := by
  sorry

end complex_equation_solution_l3362_336226


namespace inequality_equivalence_l3362_336285

theorem inequality_equivalence (x : ℝ) : 
  (-2 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 2) ↔ 
  (4 - Real.sqrt 21 < x ∧ x < 4 + Real.sqrt 21) := by sorry

end inequality_equivalence_l3362_336285


namespace complex_simplification_l3362_336280

theorem complex_simplification : 
  3 * (4 - 2 * Complex.I) + 2 * Complex.I * (3 + Complex.I) = (10 : ℂ) := by
  sorry

end complex_simplification_l3362_336280


namespace price_reduction_achieves_desired_profit_l3362_336284

/-- Represents the profit and sales scenario for black pork zongzi --/
structure ZongziSales where
  initialProfit : ℝ  -- Initial profit per box
  initialQuantity : ℝ  -- Initial quantity sold
  priceElasticity : ℝ  -- Additional boxes sold per dollar of price reduction
  priceReduction : ℝ  -- Amount of price reduction per box
  desiredTotalProfit : ℝ  -- Desired total profit

/-- Calculates the new profit per box after price reduction --/
def newProfitPerBox (s : ZongziSales) : ℝ :=
  s.initialProfit - s.priceReduction

/-- Calculates the new quantity sold after price reduction --/
def newQuantitySold (s : ZongziSales) : ℝ :=
  s.initialQuantity + s.priceElasticity * s.priceReduction

/-- Calculates the total profit after price reduction --/
def totalProfit (s : ZongziSales) : ℝ :=
  newProfitPerBox s * newQuantitySold s

/-- Theorem stating that a price reduction of 15 achieves the desired total profit --/
theorem price_reduction_achieves_desired_profit (s : ZongziSales)
  (h1 : s.initialProfit = 50)
  (h2 : s.initialQuantity = 50)
  (h3 : s.priceElasticity = 2)
  (h4 : s.priceReduction = 15)
  (h5 : s.desiredTotalProfit = 2800) :
  totalProfit s = s.desiredTotalProfit := by
  sorry

#eval totalProfit { initialProfit := 50, initialQuantity := 50, priceElasticity := 2, priceReduction := 15, desiredTotalProfit := 2800 }

end price_reduction_achieves_desired_profit_l3362_336284


namespace exponent_difference_l3362_336227

theorem exponent_difference (a m n : ℝ) (h1 : a^m = 12) (h2 : a^n = 3) : a^(m-n) = 4 := by
  sorry

end exponent_difference_l3362_336227


namespace third_to_second_ratio_l3362_336273

/-- Represents the number of questions solved in each hour -/
structure HourlyQuestions where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Verifies if the given hourly questions satisfy the problem conditions -/
def satisfiesConditions (q : HourlyQuestions) : Prop :=
  q.third = 132 ∧
  q.third = 3 * q.first ∧
  q.first + q.second + q.third = 242

/-- Theorem stating that if the conditions are satisfied, the ratio of third to second hour questions is 2:1 -/
theorem third_to_second_ratio (q : HourlyQuestions) 
  (h : satisfiesConditions q) : q.third = 2 * q.second :=
by
  sorry

#check third_to_second_ratio

end third_to_second_ratio_l3362_336273


namespace distribute_five_prizes_to_three_students_l3362_336291

/-- The number of ways to distribute n different prizes to k students,
    with each student receiving at least one prize -/
def distribute_prizes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 different prizes to 3 students,
    with each student receiving at least one prize, is 150 -/
theorem distribute_five_prizes_to_three_students :
  distribute_prizes 5 3 = 150 := by sorry

end distribute_five_prizes_to_three_students_l3362_336291


namespace lcm_e_n_l3362_336268

theorem lcm_e_n (e n : ℕ) (h1 : e > 0) (h2 : n ≥ 100 ∧ n ≤ 999) 
  (h3 : ¬(3 ∣ n)) (h4 : ¬(2 ∣ e)) (h5 : n = 230) : 
  Nat.lcm e n = 230 := by
  sorry

end lcm_e_n_l3362_336268


namespace unique_modular_congruence_l3362_336218

theorem unique_modular_congruence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ -300 ≡ n [ZMOD 31] ∧ n = 10 := by
  sorry

end unique_modular_congruence_l3362_336218


namespace alice_prob_after_three_turns_l3362_336272

/-- Represents the possessor of the ball -/
inductive Possessor : Type
| Alice : Possessor
| Bob : Possessor

/-- The game state after a turn -/
structure GameState :=
  (possessor : Possessor)

/-- The probability of Alice having the ball after one turn, given the current possessor -/
def aliceProbAfterOneTurn (current : Possessor) : ℚ :=
  match current with
  | Possessor.Alice => 2/3
  | Possessor.Bob => 3/5

/-- The probability of Alice having the ball after three turns, given Alice starts -/
def aliceProbAfterThreeTurns : ℚ := 7/45

theorem alice_prob_after_three_turns :
  aliceProbAfterThreeTurns = 7/45 :=
sorry

end alice_prob_after_three_turns_l3362_336272


namespace jessica_scores_mean_l3362_336249

def jessica_scores : List ℝ := [87, 94, 85, 92, 90, 88]

theorem jessica_scores_mean :
  (jessica_scores.sum / jessica_scores.length : ℝ) = 89.3333333333333 := by
  sorry

end jessica_scores_mean_l3362_336249


namespace circle_center_and_radius_l3362_336250

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y - 2 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 1)

-- Define the radius of the circle
def radius : ℝ := 2

-- Theorem stating that the given center and radius satisfy the circle equation
theorem circle_center_and_radius :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry


end circle_center_and_radius_l3362_336250


namespace pastry_sale_revenue_l3362_336214

/-- Calculates the total money made from selling discounted pastries -/
theorem pastry_sale_revenue 
  (original_cupcake_price original_cookie_price : ℚ)
  (cupcakes_sold cookies_sold : ℕ)
  (h1 : original_cupcake_price = 3)
  (h2 : original_cookie_price = 2)
  (h3 : cupcakes_sold = 16)
  (h4 : cookies_sold = 8) :
  (cupcakes_sold : ℚ) * (original_cupcake_price / 2) + 
  (cookies_sold : ℚ) * (original_cookie_price / 2) = 32 := by
sorry


end pastry_sale_revenue_l3362_336214


namespace initial_water_amount_l3362_336269

/-- Given a container with alcohol and water, prove the initial amount of water -/
theorem initial_water_amount (initial_alcohol : ℝ) (added_water : ℝ) (ratio_alcohol : ℝ) (ratio_water : ℝ) :
  initial_alcohol = 4 →
  added_water = 2.666666666666667 →
  ratio_alcohol = 3 →
  ratio_water = 5 →
  ratio_alcohol / ratio_water = initial_alcohol / (initial_alcohol + added_water + x) →
  x = 4 :=
by sorry

end initial_water_amount_l3362_336269


namespace solution_equation1_solution_equation2_l3362_336203

-- Define the equations
def equation1 (x : ℝ) : Prop := 7*x - 20 = 2*(3 - 3*x)
def equation2 (x : ℝ) : Prop := (2*x - 3)/5 = (3*x - 1)/2 + 1

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 2 := by sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -1 := by sorry

end solution_equation1_solution_equation2_l3362_336203


namespace parabola_equation_l3362_336254

/-- A parabola with vertex at the origin, axis of symmetry along a coordinate axis, 
    and passing through the point (√3, -2√3) has the equation y² = 4√3x or x² = -√3/2y -/
theorem parabola_equation (x y : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ ((y^2 = 2*p*x ∧ x = Real.sqrt 3 ∧ y = -2*Real.sqrt 3) ∨ 
                       (x^2 = -2*p*y ∧ x = Real.sqrt 3 ∧ y = -2*Real.sqrt 3))) → 
  (y^2 = 4*Real.sqrt 3*x ∨ x^2 = -(Real.sqrt 3/2)*y) :=
by sorry

end parabola_equation_l3362_336254


namespace water_flow_fraction_l3362_336200

/-- Given a water flow problem with the following conditions:
  * The original flow rate is 5 gallons per minute
  * The reduced flow rate is 2 gallons per minute
  * The reduced flow rate is 1 gallon per minute less than a fraction of the original flow rate
  Prove that the fraction of the original flow rate is 3/5 -/
theorem water_flow_fraction (original_rate reduced_rate : ℚ) 
  (h1 : original_rate = 5)
  (h2 : reduced_rate = 2) :
  ∃ f : ℚ, f * original_rate - 1 = reduced_rate ∧ f = 3/5 := by
  sorry

end water_flow_fraction_l3362_336200


namespace nth_decimal_35_36_l3362_336208

/-- The fraction 35/36 as a real number -/
def f : ℚ := 35 / 36

/-- Predicate to check if the nth decimal digit of a rational number is 2 -/
def is_nth_decimal_2 (q : ℚ) (n : ℕ) : Prop :=
  (q * 10^n - ⌊q * 10^n⌋) * 10 ≥ 2 ∧ (q * 10^n - ⌊q * 10^n⌋) * 10 < 3

/-- Theorem stating that the nth decimal digit of 35/36 is 2 if and only if n ≥ 2 -/
theorem nth_decimal_35_36 (n : ℕ) : is_nth_decimal_2 f n ↔ n ≥ 2 := by
  sorry

end nth_decimal_35_36_l3362_336208


namespace count_integers_with_7_or_8_eq_386_l3362_336265

/-- The number of digits in base 9 that do not include 7 or 8 -/
def base7_digits : ℕ := 7

/-- The number of digits we consider in base 9 -/
def num_digits : ℕ := 3

/-- The total number of integers we consider -/
def total_integers : ℕ := 729

/-- The function that calculates the number of integers in base 9 
    from 1 to 729 that contain at least one digit 7 or 8 -/
def count_integers_with_7_or_8 : ℕ := total_integers - base7_digits ^ num_digits

theorem count_integers_with_7_or_8_eq_386 : 
  count_integers_with_7_or_8 = 386 := by sorry

end count_integers_with_7_or_8_eq_386_l3362_336265


namespace right_triangle_matchsticks_l3362_336232

theorem right_triangle_matchsticks (a b c : ℕ) : 
  a = 6 ∧ b = 8 ∧ c^2 = a^2 + b^2 → a + b + c = 24 :=
by
  sorry

end right_triangle_matchsticks_l3362_336232


namespace trajectory_of_C_l3362_336202

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  let A := (-2, 0)
  let B := (2, 0)
  let perimeter := dist A C + dist B C + dist A B
  perimeter = 10

-- Define the equation of the trajectory
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 5 = 1 ∧ y ≠ 0

-- Theorem statement
theorem trajectory_of_C :
  ∀ C : ℝ × ℝ, triangle_ABC C → 
  ∃ x y : ℝ, C = (x, y) ∧ trajectory_equation x y :=
sorry

end trajectory_of_C_l3362_336202


namespace function_composition_problem_l3362_336204

theorem function_composition_problem (k b : ℝ) (f : ℝ → ℝ) :
  (k < 0) →
  (∀ x, f x = k * x + b) →
  (∀ x, f (f x) = 4 * x + 1) →
  (∀ x, f x = -2 * x - 1) :=
by sorry

end function_composition_problem_l3362_336204


namespace lowest_temp_is_harbin_l3362_336239

def harbin_temp : ℤ := -20
def beijing_temp : ℤ := -10
def hangzhou_temp : ℤ := 0
def jinhua_temp : ℤ := 2

def city_temps : List ℤ := [harbin_temp, beijing_temp, hangzhou_temp, jinhua_temp]

theorem lowest_temp_is_harbin :
  List.minimum city_temps = some harbin_temp := by
  sorry

end lowest_temp_is_harbin_l3362_336239


namespace mean_of_playground_counts_l3362_336287

def playground_counts : List ℕ := [6, 12, 1, 12, 7, 3, 8]

theorem mean_of_playground_counts :
  (playground_counts.sum : ℚ) / playground_counts.length = 7 := by
  sorry

end mean_of_playground_counts_l3362_336287


namespace plot_length_is_65_l3362_336245

/-- Represents a rectangular plot with its dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ

/-- The length of the plot is 30 meters more than its breadth. -/
def lengthCondition (plot : RectangularPlot) : Prop :=
  plot.length = plot.breadth + 30

/-- The cost of fencing the plot at the given rate equals the total fencing cost. -/
def fencingCostCondition (plot : RectangularPlot) : Prop :=
  plot.fencingCostPerMeter * (2 * plot.length + 2 * plot.breadth) = plot.totalFencingCost

/-- The main theorem stating that under the given conditions, the length of the plot is 65 meters. -/
theorem plot_length_is_65 (plot : RectangularPlot) 
    (h1 : lengthCondition plot) 
    (h2 : fencingCostCondition plot) 
    (h3 : plot.fencingCostPerMeter = 26.5) 
    (h4 : plot.totalFencingCost = 5300) : 
  plot.length = 65 := by
  sorry

end plot_length_is_65_l3362_336245


namespace system_solution_ratio_l3362_336276

theorem system_solution_ratio (x y c d : ℝ) : 
  x ≠ 0 → y ≠ 0 → d ≠ 0 →
  8 * x - 6 * y = c →
  10 * y - 15 * x = d →
  c / d = -8 / 15 := by
sorry

end system_solution_ratio_l3362_336276


namespace trig_expression_equals_half_l3362_336258

/-- Proves that the given trigonometric expression equals 1/2 --/
theorem trig_expression_equals_half : 
  (Real.sin (70 * π / 180) * Real.sin (20 * π / 180)) / 
  (Real.cos (155 * π / 180)^2 - Real.sin (155 * π / 180)^2) = 1/2 := by
  sorry

end trig_expression_equals_half_l3362_336258


namespace parabola_tangent_line_l3362_336297

/-- A parabola is tangent to a line if and only if their intersection equation has exactly one solution -/
axiom tangent_condition (a : ℝ) : 
  (∃! x, a * x^2 + 4 = 2 * x + 1) ↔ (∃ x, a * x^2 + 4 = 2 * x + 1 ∧ ∀ y, a * y^2 + 4 = 2 * y + 1 → y = x)

/-- The main theorem: if a parabola y = ax^2 + 4 is tangent to the line y = 2x + 1, then a = 1/3 -/
theorem parabola_tangent_line (a : ℝ) : 
  (∃! x, a * x^2 + 4 = 2 * x + 1) → a = 1/3 := by
sorry


end parabola_tangent_line_l3362_336297


namespace max_stations_is_ten_exists_valid_network_with_ten_stations_max_stations_is_exactly_ten_l3362_336261

/-- Represents a surveillance network of stations -/
structure SurveillanceNetwork where
  stations : Finset ℕ
  connections : Finset (ℕ × ℕ)

/-- Checks if a station can communicate with all others directly or through one intermediary -/
def canCommunicateWithAll (net : SurveillanceNetwork) (s : ℕ) : Prop :=
  ∀ t ∈ net.stations, s ≠ t →
    (s, t) ∈ net.connections ∨ ∃ u ∈ net.stations, (s, u) ∈ net.connections ∧ (u, t) ∈ net.connections

/-- Checks if a station has at most three direct connections -/
def hasAtMostThreeConnections (net : SurveillanceNetwork) (s : ℕ) : Prop :=
  (net.connections.filter (λ p => p.1 = s ∨ p.2 = s)).card ≤ 3

/-- A valid surveillance network satisfies all conditions -/
def isValidNetwork (net : SurveillanceNetwork) : Prop :=
  ∀ s ∈ net.stations, canCommunicateWithAll net s ∧ hasAtMostThreeConnections net s

/-- The maximum number of stations in a valid surveillance network is 10 -/
theorem max_stations_is_ten :
  ∀ net : SurveillanceNetwork, isValidNetwork net → net.stations.card ≤ 10 :=
sorry

/-- There exists a valid surveillance network with 10 stations -/
theorem exists_valid_network_with_ten_stations :
  ∃ net : SurveillanceNetwork, isValidNetwork net ∧ net.stations.card = 10 :=
sorry

/-- The maximum number of stations in a valid surveillance network is exactly 10 -/
theorem max_stations_is_exactly_ten :
  (∃ net : SurveillanceNetwork, isValidNetwork net ∧ net.stations.card = 10) ∧
  (∀ net : SurveillanceNetwork, isValidNetwork net → net.stations.card ≤ 10) :=
sorry

end max_stations_is_ten_exists_valid_network_with_ten_stations_max_stations_is_exactly_ten_l3362_336261


namespace real_axis_length_of_hyperbola_l3362_336246

/-- The length of the real axis of a hyperbola given by the equation 2x^2 - y^2 = 8 -/
def real_axis_length : ℝ := 4

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := 2 * x^2 - y^2 = 8

theorem real_axis_length_of_hyperbola :
  ∀ x y : ℝ, hyperbola_equation x y → real_axis_length = 4 := by
  sorry

end real_axis_length_of_hyperbola_l3362_336246


namespace overall_loss_percentage_l3362_336299

/-- Calculate the overall loss percentage for three items given their cost and selling prices -/
theorem overall_loss_percentage
  (cp_radio : ℚ) (cp_speaker : ℚ) (cp_headphones : ℚ)
  (sp_radio : ℚ) (sp_speaker : ℚ) (sp_headphones : ℚ)
  (h1 : cp_radio = 1500)
  (h2 : cp_speaker = 2500)
  (h3 : cp_headphones = 800)
  (h4 : sp_radio = 1275)
  (h5 : sp_speaker = 2300)
  (h6 : sp_headphones = 700) :
  let total_cp := cp_radio + cp_speaker + cp_headphones
  let total_sp := sp_radio + sp_speaker + sp_headphones
  let loss := total_cp - total_sp
  let loss_percentage := (loss / total_cp) * 100
  abs (loss_percentage - 10.94) < 0.01 := by
  sorry

end overall_loss_percentage_l3362_336299


namespace line_intercept_sum_l3362_336290

/-- Given a line 3x + 5y + c = 0, if the sum of its x-intercept and y-intercept is 55/4, then c = 825/32 -/
theorem line_intercept_sum (c : ℚ) : 
  (∃ x y : ℚ, 3 * x + 5 * y + c = 0 ∧ x + y = 55 / 4) → c = 825 / 32 := by
  sorry

end line_intercept_sum_l3362_336290


namespace closest_multiple_of_12_to_1987_is_correct_l3362_336293

def is_multiple_of_12 (n : ℤ) : Prop := n % 12 = 0

def closest_multiple_of_12_to_1987 : ℤ := 1984

theorem closest_multiple_of_12_to_1987_is_correct :
  is_multiple_of_12 closest_multiple_of_12_to_1987 ∧
  ∀ m : ℤ, is_multiple_of_12 m →
    |m - 1987| ≥ |closest_multiple_of_12_to_1987 - 1987| :=
by sorry

end closest_multiple_of_12_to_1987_is_correct_l3362_336293


namespace g_of_x_plus_3_l3362_336263

/-- Given a function g(x) = x^2 - x, prove that g(x+3) = x^2 + 5x + 6 -/
theorem g_of_x_plus_3 (x : ℝ) : 
  let g := fun (x : ℝ) => x^2 - x
  g (x + 3) = x^2 + 5*x + 6 := by
  sorry

end g_of_x_plus_3_l3362_336263


namespace trajectory_of_midpoint_l3362_336279

/-- Given a point P on the circle x^2 + y^2 = 16, and M being the midpoint of the perpendicular
    line segment from P to the x-axis, the trajectory of M satisfies the equation x^2/4 + y^2/16 = 1. -/
theorem trajectory_of_midpoint (x₀ y₀ x y : ℝ) : 
  x₀^2 + y₀^2 = 16 →  -- P is on the circle
  x₀ = 2*x →  -- M is the midpoint of PD (x-coordinate)
  y₀ = y →  -- M is the midpoint of PD (y-coordinate)
  x^2/4 + y^2/16 = 1 := by
sorry

end trajectory_of_midpoint_l3362_336279


namespace geometric_sequence_sum_l3362_336271

/-- Given a geometric sequence {a_n} where a_1 + a_2 = 30 and a_4 + a_5 = 120, 
    then a_7 + a_8 = 480. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence
  a 1 + a 2 = 30 →                          -- a_1 + a_2 = 30
  a 4 + a 5 = 120 →                         -- a_4 + a_5 = 120
  a 7 + a 8 = 480 :=                        -- a_7 + a_8 = 480
by
  sorry

end geometric_sequence_sum_l3362_336271


namespace noras_oranges_l3362_336240

/-- The total number of oranges Nora picked from three trees -/
def total_oranges (tree1 tree2 tree3 : ℕ) : ℕ := tree1 + tree2 + tree3

/-- Theorem stating that the total number of oranges Nora picked is 260 -/
theorem noras_oranges : total_oranges 80 60 120 = 260 := by
  sorry

end noras_oranges_l3362_336240


namespace smallest_integer_gcd_18_is_6_l3362_336207

theorem smallest_integer_gcd_18_is_6 : 
  ∃ (n : ℕ), n > 100 ∧ Nat.gcd n 18 = 6 ∧ ∀ m, m > 100 ∧ m < n → Nat.gcd m 18 ≠ 6 :=
by
  -- The proof goes here
  sorry

end smallest_integer_gcd_18_is_6_l3362_336207


namespace league_games_l3362_336281

theorem league_games (n : ℕ) (h : n = 14) : (n * (n - 1)) / 2 = 91 := by
  sorry

end league_games_l3362_336281


namespace arithmetic_sequence_problem_l3362_336235

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)) →  -- Definition of sum for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →                      -- Definition of arithmetic sequence
  S 3 = 6 →                                                 -- Given condition
  5 * a 1 + a 7 = 12 :=                                     -- Conclusion to prove
by sorry

end arithmetic_sequence_problem_l3362_336235


namespace hyperbola_center_l3362_336230

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 c : ℝ × ℝ) : 
  f1 = (3, -2) → f2 = (-1, 6) → c = (1, 2) → 
  c.1 = (f1.1 + f2.1) / 2 ∧ c.2 = (f1.2 + f2.2) / 2 := by
  sorry

end hyperbola_center_l3362_336230


namespace real_roots_quadratic_range_l3362_336223

theorem real_roots_quadratic_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + x - 1 = 0) ↔ (m ≥ -1/4 ∧ m ≠ 0) :=
by sorry

end real_roots_quadratic_range_l3362_336223


namespace cone_surface_area_l3362_336244

/-- Given a cone with slant height 4 and cross-sectional area π, 
    its total surface area is 12π. -/
theorem cone_surface_area (s : ℝ) (a : ℝ) (h1 : s = 4) (h2 : a = π) :
  let r := Real.sqrt (a / π)
  let lateral_area := π * r * s
  let base_area := a
  lateral_area + base_area = 12 * π := by
  sorry

end cone_surface_area_l3362_336244


namespace sin_2theta_value_l3362_336270

theorem sin_2theta_value (θ : ℝ) (h : Real.cos (π/4 - θ) = 1/2) : 
  Real.sin (2*θ) = -1/2 := by
  sorry

end sin_2theta_value_l3362_336270


namespace problem1_l3362_336277

theorem problem1 (a b : ℝ) (ha : a = -Real.sqrt 2) (hb : b = Real.sqrt 6) :
  (a + b) * (a - b) + b * (a + 2 * b) - (a + b)^2 = 2 * Real.sqrt 3 := by
  sorry

end problem1_l3362_336277


namespace ab_inequality_relationship_l3362_336206

theorem ab_inequality_relationship (a b : ℝ) :
  (∀ a b, a < b ∧ b < 0 → a * b * (a - b) < 0) ∧
  (∃ a b, a * b * (a - b) < 0 ∧ ¬(a < b ∧ b < 0)) :=
by sorry

end ab_inequality_relationship_l3362_336206


namespace min_reciprocal_sum_l3362_336264

theorem min_reciprocal_sum (x y z : ℝ) (hpos_x : 0 < x) (hpos_y : 0 < y) (hpos_z : 0 < z)
  (hsum : x + y + z = 2) (hx : x = 2 * y) :
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 ∧ a = 2 * b →
  1 / x + 1 / y + 1 / z ≤ 1 / a + 1 / b + 1 / c ∧
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 ∧ a = 2 * b ∧
  1 / x + 1 / y + 1 / z = 1 / a + 1 / b + 1 / c ∧
  1 / x + 1 / y + 1 / z = 5 := by
  sorry

end min_reciprocal_sum_l3362_336264


namespace regular_polygon_sides_l3362_336234

/-- Given a regular polygon inscribed in a circle, if the central angle corresponding to one side is 72°, then the polygon has 5 sides. -/
theorem regular_polygon_sides (n : ℕ) (central_angle : ℝ) : 
  n ≥ 3 → 
  central_angle = 72 → 
  (360 : ℝ) / n = central_angle → 
  n = 5 := by sorry

end regular_polygon_sides_l3362_336234


namespace intersection_difference_l3362_336260

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6
def parabola2 (x : ℝ) : ℝ := -2 * x^2 + 2 * x + 6

theorem intersection_difference :
  ∃ (a c : ℝ),
    (∀ x : ℝ, parabola1 x = parabola2 x → x = a ∨ x = c) ∧
    c ≥ a ∧
    c - a = 8/5 := by
  sorry

end intersection_difference_l3362_336260


namespace jacket_pricing_l3362_336296

theorem jacket_pricing (x : ℝ) : 
  let marked_price : ℝ := 300
  let discount_rate : ℝ := 0.7
  let profit : ℝ := 20
  (marked_price * discount_rate - x = profit) ↔ 
  (300 * 0.7 - x = 20) :=
by sorry

end jacket_pricing_l3362_336296


namespace cube_decomposition_91_l3362_336252

/-- Decomposition of a cube into consecutive odd numbers -/
def cube_decomposition (n : ℕ+) : List ℕ :=
  sorry

/-- The smallest number in the decomposition of m³ -/
def smallest_in_decomposition (m : ℕ+) : ℕ :=
  sorry

/-- Theorem: If the smallest number in the decomposition of m³ is 91, then m = 10 -/
theorem cube_decomposition_91 (m : ℕ+) :
  smallest_in_decomposition m = 91 → m = 10 := by
  sorry

end cube_decomposition_91_l3362_336252


namespace jane_reading_pages_l3362_336295

/-- Calculates the number of pages Jane reads in a week -/
def pages_read_in_week (morning_pages : ℕ) (evening_pages : ℕ) (days_in_week : ℕ) : ℕ :=
  (morning_pages + evening_pages) * days_in_week

theorem jane_reading_pages : pages_read_in_week 5 10 7 = 105 := by
  sorry

end jane_reading_pages_l3362_336295


namespace right_angled_triangle_isosceles_triangle_isosceles_perimeter_l3362_336286

/-- Definition of the triangle ABC with side lengths based on the quadratic equation -/
def Triangle (k : ℝ) : Prop :=
  ∃ (a b : ℝ),
    a^2 - (2*k + 3)*a + k^2 + 3*k + 2 = 0 ∧
    b^2 - (2*k + 3)*b + k^2 + 3*k + 2 = 0 ∧
    a ≠ b

/-- The length of side BC is 5 -/
def BC_length (k : ℝ) : ℝ := 5

/-- Theorem: If ABC is a right-angled triangle with BC as the hypotenuse, then k = 2 -/
theorem right_angled_triangle (k : ℝ) :
  Triangle k → (∃ (a b : ℝ), a^2 + b^2 = (BC_length k)^2) → k = 2 :=
sorry

/-- Theorem: If ABC is an isosceles triangle, then k = 3 or k = 4 -/
theorem isosceles_triangle (k : ℝ) :
  Triangle k → (∃ (a b : ℝ), (a = b ∧ a ≠ BC_length k) ∨ (a = BC_length k ∧ b ≠ BC_length k) ∨ (b = BC_length k ∧ a ≠ BC_length k)) →
  k = 3 ∨ k = 4 :=
sorry

/-- Theorem: If ABC is an isosceles triangle, then its perimeter is 14 or 16 -/
theorem isosceles_perimeter (k : ℝ) :
  Triangle k → (∃ (a b : ℝ), (a = b ∧ a ≠ BC_length k) ∨ (a = BC_length k ∧ b ≠ BC_length k) ∨ (b = BC_length k ∧ a ≠ BC_length k)) →
  (∃ (p : ℝ), p = a + b + BC_length k ∧ (p = 14 ∨ p = 16)) :=
sorry

end right_angled_triangle_isosceles_triangle_isosceles_perimeter_l3362_336286


namespace room_length_calculation_l3362_336209

/-- Given a rectangular room with width 4 meters and a floor paving cost resulting in a total cost of 18700, prove that the length of the room is 5.5 meters. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (length : ℝ) : 
  width = 4 →
  cost_per_sqm = 850 →
  total_cost = 18700 →
  total_cost = cost_per_sqm * (length * width) →
  length = 5.5 := by
  sorry

end room_length_calculation_l3362_336209


namespace second_platform_length_l3362_336216

/-- The length of the second platform given train and first platform details -/
theorem second_platform_length
  (train_length : ℝ)
  (first_platform_length : ℝ)
  (time_first_platform : ℝ)
  (time_second_platform : ℝ)
  (h1 : train_length = 150)
  (h2 : first_platform_length = 150)
  (h3 : time_first_platform = 15)
  (h4 : time_second_platform = 20) :
  (time_second_platform * (train_length + first_platform_length) / time_first_platform) - train_length = 250 :=
by sorry

end second_platform_length_l3362_336216


namespace set_operations_l3362_336267

def A : Set ℕ := {6, 8, 10, 12}
def B : Set ℕ := {1, 6, 8}

theorem set_operations :
  (A ∪ B = {1, 6, 8, 10, 12}) ∧
  (𝒫(A ∩ B) = {∅, {6}, {8}, {6, 8}}) := by
  sorry

end set_operations_l3362_336267


namespace pond_ducks_l3362_336266

/-- The number of ducks in the pond -/
def num_ducks : ℕ := 3

/-- The total number of bread pieces thrown in the pond -/
def total_bread : ℕ := 100

/-- The number of bread pieces left in the water -/
def left_bread : ℕ := 30

/-- The number of bread pieces eaten by the second duck -/
def second_duck_bread : ℕ := 13

/-- The number of bread pieces eaten by the third duck -/
def third_duck_bread : ℕ := 7

/-- Theorem stating that the number of ducks in the pond is 3 -/
theorem pond_ducks : 
  (total_bread / 2 + second_duck_bread + third_duck_bread = total_bread - left_bread) → 
  num_ducks = 3 := by
  sorry


end pond_ducks_l3362_336266


namespace sum_x_coordinates_on_parabola_l3362_336233

/-- The parabola equation y = x² - 2x + 1 -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem: For any two points P(x₁, 1) and Q(x₂, 1) on the parabola y = x² - 2x + 1,
    the sum of their x-coordinates (x₁ + x₂) is equal to 2. -/
theorem sum_x_coordinates_on_parabola (x₁ x₂ : ℝ) 
    (h₁ : parabola x₁ = 1) 
    (h₂ : parabola x₂ = 1) : 
  x₁ + x₂ = 2 := by
  sorry

end sum_x_coordinates_on_parabola_l3362_336233


namespace perfect_square_divisibility_l3362_336288

theorem perfect_square_divisibility (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : 
  ∃ k : ℕ, a = k^2 :=
sorry

end perfect_square_divisibility_l3362_336288


namespace greatest_common_divisor_with_digit_sum_l3362_336225

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem greatest_common_divisor_with_digit_sum : 
  ∃ (n : ℕ), 
    n ∣ (6905 - 4665) ∧ 
    sum_of_digits n = 4 ∧ 
    ∀ (m : ℕ), m ∣ (6905 - 4665) ∧ sum_of_digits m = 4 → m ≤ n :=
by sorry

end greatest_common_divisor_with_digit_sum_l3362_336225


namespace total_customers_l3362_336213

def customers_in_line (people_in_front : ℕ) : ℕ := people_in_front + 1

theorem total_customers (people_in_front : ℕ) : 
  people_in_front = 8 → customers_in_line people_in_front = 9 := by
  sorry

end total_customers_l3362_336213


namespace complex_fraction_equality_l3362_336289

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) - (x - y) / (x + y) = 3) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 59525 / 30964 := by
  sorry

end complex_fraction_equality_l3362_336289


namespace potato_rows_l3362_336275

theorem potato_rows (seeds_per_row : ℕ) (total_potatoes : ℕ) (h1 : seeds_per_row = 9) (h2 : total_potatoes = 54) :
  total_potatoes / seeds_per_row = 6 := by
  sorry

end potato_rows_l3362_336275


namespace adjacent_knights_probability_l3362_336294

def n : ℕ := 20  -- Total number of knights
def k : ℕ := 4   -- Number of knights chosen

-- Probability that at least two of the four chosen knights were sitting next to each other
def adjacent_probability : ℚ :=
  1 - (Nat.choose (n - k) (k - 1) : ℚ) / (Nat.choose n k : ℚ)

theorem adjacent_knights_probability :
  adjacent_probability = 66 / 75 :=
sorry

end adjacent_knights_probability_l3362_336294


namespace isosceles_triangle_base_angle_l3362_336229

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only define the angles, as that's all we need for this problem
  vertex_angle : ℝ
  base_angle : ℝ
  -- The sum of angles in a triangle is 180°
  angle_sum : vertex_angle + 2 * base_angle = 180
  -- In an isosceles triangle, the base angles are equal

-- Define our specific isosceles triangle with one 40° angle
def triangle_with_40_degree_angle (t : IsoscelesTriangle) : Prop :=
  t.vertex_angle = 40 ∨ t.base_angle = 40

-- Theorem to prove
theorem isosceles_triangle_base_angle 
  (t : IsoscelesTriangle) 
  (h : triangle_with_40_degree_angle t) : 
  t.base_angle = 40 ∨ t.base_angle = 70 := by
  sorry


end isosceles_triangle_base_angle_l3362_336229


namespace s_not_lowest_avg_l3362_336274

-- Define the set of runners
inductive Runner : Type
  | P | Q | R | S | T

-- Define a type for race results
def RaceResult := List Runner

-- Define the first race result
def firstRace : RaceResult := sorry

-- Define the second race result
def secondRace : RaceResult := [Runner.R, Runner.P, Runner.T, Runner.Q, Runner.S]

-- Function to calculate the position of a runner in a race
def position (runner : Runner) (race : RaceResult) : Nat := sorry

-- Function to calculate the average position of a runner across two races
def avgPosition (runner : Runner) (race1 race2 : RaceResult) : Rat :=
  (position runner race1 + position runner race2) / 2

-- Theorem stating that S cannot have the lowest average position
theorem s_not_lowest_avg :
  ∀ (r : Runner), r ≠ Runner.S →
    avgPosition Runner.S firstRace secondRace ≥ avgPosition r firstRace secondRace :=
  sorry

end s_not_lowest_avg_l3362_336274


namespace indeterminate_roots_l3362_336282

theorem indeterminate_roots (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_equal_roots : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ 
    ∀ y : ℝ, a * y^2 + b * y + c = 0 → y = x) :
  ¬∃ (root_nature : Prop), 
    (∀ x : ℝ, (a + 1) * x^2 + (b + 2) * x + (c + 1) = 0 ↔ root_nature) :=
sorry

end indeterminate_roots_l3362_336282


namespace integer_solutions_count_l3362_336221

theorem integer_solutions_count :
  let f : ℤ → ℤ → ℤ := λ x y => 6 * y^2 + 3 * x * y + x + 2 * y - 72
  ∃! s : Finset (ℤ × ℤ), (∀ (x y : ℤ), (x, y) ∈ s ↔ f x y = 0) ∧ Finset.card s = 4 :=
by sorry

end integer_solutions_count_l3362_336221


namespace solve_quadratic_equation_l3362_336278

theorem solve_quadratic_equation (x : ℝ) : 3 * (x + 1)^2 = 27 → x = 2 ∨ x = -4 := by
  sorry

end solve_quadratic_equation_l3362_336278


namespace hyperbola_distance_l3362_336215

/-- Given a hyperbola with equation x²/25 - y²/9 = 1, prove that |ON| = 4 --/
theorem hyperbola_distance (M F₁ F₂ N O : ℝ × ℝ) : 
  (∀ x y, (x^2 / 25) - (y^2 / 9) = 1 → (x, y) = M) →  -- M is on the hyperbola
  (M.1 < 0) →  -- M is on the left branch
  ‖M - F₂‖ = 18 →  -- Distance from M to F₂ is 18
  N = (M + F₂) / 2 →  -- N is the midpoint of MF₂
  O = (0, 0) →  -- O is the origin
  ‖O - N‖ = 4 := by
  sorry

end hyperbola_distance_l3362_336215


namespace tan_ratio_problem_l3362_336237

theorem tan_ratio_problem (x : ℝ) (h : Real.tan (x + π/4) = 2) : 
  Real.tan x / Real.tan (2*x) = 4/9 := by
  sorry

end tan_ratio_problem_l3362_336237


namespace smallest_solution_equation_l3362_336248

theorem smallest_solution_equation (x : ℝ) :
  (3 * x) / (x - 2) + (2 * x^2 - 28) / x = 11 →
  x ≥ ((-1 : ℝ) - Real.sqrt 17) / 2 ∧
  (3 * (((-1 : ℝ) - Real.sqrt 17) / 2)) / (((-1 : ℝ) - Real.sqrt 17) / 2 - 2) +
  (2 * (((-1 : ℝ) - Real.sqrt 17) / 2)^2 - 28) / (((-1 : ℝ) - Real.sqrt 17) / 2) = 11 :=
by sorry

end smallest_solution_equation_l3362_336248


namespace inclination_angle_range_l3362_336220

/-- Given a line with equation x*sin(α) + y + 2 = 0, 
    the range of the inclination angle α is [0, π/4] ∪ [3π/4, π) -/
theorem inclination_angle_range (x y : ℝ) (α : ℝ) :
  (x * Real.sin α + y + 2 = 0) →
  α ∈ Set.Icc 0 (Real.pi / 4) ∪ Set.Ico (3 * Real.pi / 4) Real.pi :=
by sorry

end inclination_angle_range_l3362_336220


namespace base_c_is_seven_l3362_336257

theorem base_c_is_seven (c : ℕ) (h : c > 1) : 
  (3 * c + 2)^2 = c^3 + 2 * c^2 + 6 * c + 4 → c = 7 := by
  sorry

end base_c_is_seven_l3362_336257


namespace only_constant_one_is_divisor_respecting_l3362_336212

-- Define the number of positive divisors function
def d (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range n)).card + 1

-- Define divisor-respecting property
def divisor_respecting (F : ℕ → ℕ) : Prop :=
  (∀ m n : ℕ, d (F (m * n)) = d (F m) * d (F n)) ∧
  (∀ n : ℕ, d (F n) ≤ d n)

-- Theorem statement
theorem only_constant_one_is_divisor_respecting :
  ∀ F : ℕ → ℕ, divisor_respecting F → ∀ x : ℕ, F x = 1 :=
by sorry

end only_constant_one_is_divisor_respecting_l3362_336212


namespace ellipse_focal_distance_l3362_336205

/-- Given an ellipse with equation x^2 + 9y^2 = 144, the distance between its foci is 16√2 -/
theorem ellipse_focal_distance : 
  ∀ (x y : ℝ), x^2 + 9*y^2 = 144 → 
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (16 * Real.sqrt 2)^2 := by
  sorry


end ellipse_focal_distance_l3362_336205


namespace trees_planted_by_two_classes_l3362_336243

theorem trees_planted_by_two_classes 
  (trees_A : ℕ) 
  (trees_B : ℕ) 
  (h1 : trees_A = 8) 
  (h2 : trees_B = 7) : 
  trees_A + trees_B = 15 := by
sorry

end trees_planted_by_two_classes_l3362_336243


namespace sum_of_coefficients_l3362_336231

theorem sum_of_coefficients (x : ℝ) : 
  (fun x => (x - 2)^6 - (x - 1)^7 + (3*x - 2)^8) 1 = 2 := by
  sorry

end sum_of_coefficients_l3362_336231


namespace right_triangle_side_length_l3362_336255

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2)  -- Pythagorean theorem (right triangle condition)
  (h2 : a = 3)            -- One non-hypotenuse side length
  (h3 : c = 5)            -- Hypotenuse length
  : b = 4 := by           -- Conclusion: other non-hypotenuse side length
sorry

end right_triangle_side_length_l3362_336255


namespace smallest_base_for_square_property_l3362_336217

theorem smallest_base_for_square_property : ∃ (b x y : ℕ), 
  (b ≥ 2) ∧ 
  (x < b) ∧ 
  (y < b) ∧ 
  (x ≠ 0) ∧ 
  (y ≠ 0) ∧ 
  ((x * b + x)^2 = y * b^3 + y * b^2 + y * b + y) ∧
  (∀ b' x' y' : ℕ, 
    (b' ≥ 2) ∧ 
    (x' < b') ∧ 
    (y' < b') ∧ 
    (x' ≠ 0) ∧ 
    (y' ≠ 0) ∧ 
    ((x' * b' + x')^2 = y' * b'^3 + y' * b'^2 + y' * b' + y') →
    (b ≤ b')) ∧
  (b = 7) ∧ 
  (x = 5) ∧ 
  (y = 4) := by
sorry

end smallest_base_for_square_property_l3362_336217


namespace sin_300_degrees_l3362_336253

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l3362_336253


namespace sin_eq_cos_necessary_not_sufficient_l3362_336292

open Real

theorem sin_eq_cos_necessary_not_sufficient :
  (∃ α, sin α = cos α ∧ ¬(∃ k : ℤ, α = π / 4 + 2 * k * π)) ∧
  (∀ α, (∃ k : ℤ, α = π / 4 + 2 * k * π) → sin α = cos α) :=
by sorry

end sin_eq_cos_necessary_not_sufficient_l3362_336292


namespace max_b_value_l3362_336283

/-- The function f(x) = ax^3 + bx^2 - a^2x -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 - a^2 * x

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x - a^2

theorem max_b_value (a b : ℝ) (x₁ x₂ : ℝ) (ha : a > 0) (hx : x₁ ≠ x₂)
  (hextreme : f' a b x₁ = 0 ∧ f' a b x₂ = 0)
  (hsum : abs x₁ + abs x₂ = 2 * Real.sqrt 2) :
  b ≤ 4 * Real.sqrt 6 ∧ ∃ b₀, b₀ = 4 * Real.sqrt 6 := by
  sorry

end max_b_value_l3362_336283


namespace square_diagonal_l3362_336251

/-- The diagonal of a square with perimeter 800 cm is 200√2 cm. -/
theorem square_diagonal (perimeter : ℝ) (side : ℝ) (diagonal : ℝ) : 
  perimeter = 800 →
  side = perimeter / 4 →
  diagonal = side * Real.sqrt 2 →
  diagonal = 200 * Real.sqrt 2 :=
by sorry

end square_diagonal_l3362_336251


namespace correct_propositions_l3362_336219

-- Define the proposition from statement ②
def proposition_2 : Prop := 
  (∃ x : ℝ, x^2 + 1 > 3*x) ↔ ¬(∀ x : ℝ, x^2 + 1 ≤ 3*x)

-- Define the proposition from statement ③
def proposition_3 : Prop :=
  (∃ x : ℝ, x^2 - 3*x - 4 = 0 ∧ x ≠ 4) ∧ 
  (∀ x : ℝ, x = 4 → x^2 - 3*x - 4 = 0)

theorem correct_propositions : proposition_2 ∧ proposition_3 := by
  sorry

end correct_propositions_l3362_336219


namespace blacksmith_iron_calculation_l3362_336298

/-- The amount of iron needed for one horseshoe in kilograms -/
def iron_per_horseshoe : ℕ := 2

/-- The number of horseshoes needed for one horse -/
def horseshoes_per_horse : ℕ := 4

/-- The number of farms -/
def num_farms : ℕ := 2

/-- The number of horses in each farm -/
def horses_per_farm : ℕ := 2

/-- The number of stables -/
def num_stables : ℕ := 2

/-- The number of horses in each stable -/
def horses_per_stable : ℕ := 5

/-- The number of horses at the riding school -/
def riding_school_horses : ℕ := 36

/-- The total amount of iron the blacksmith had initially in kilograms -/
def initial_iron : ℕ := 400

theorem blacksmith_iron_calculation : 
  initial_iron = 
    (num_farms * horses_per_farm + num_stables * horses_per_stable + riding_school_horses) * 
    horseshoes_per_horse * iron_per_horseshoe :=
by sorry

end blacksmith_iron_calculation_l3362_336298


namespace total_treats_is_275_l3362_336201

/-- The total number of treats Mary, John, and Sue have -/
def total_treats (chewing_gums chocolate_bars lollipops cookies other_candies : ℕ) : ℕ :=
  chewing_gums + chocolate_bars + lollipops + cookies + other_candies

/-- Theorem stating that the total number of treats is 275 -/
theorem total_treats_is_275 :
  total_treats 60 55 70 50 40 = 275 := by
  sorry

end total_treats_is_275_l3362_336201


namespace college_running_survey_l3362_336259

/-- Represents the sample data for running mileage --/
structure SampleData where
  male_0_30 : ℕ
  male_30_60 : ℕ
  male_60_90 : ℕ
  male_90_plus : ℕ
  female_0_30 : ℕ
  female_30_60 : ℕ
  female_60_90 : ℕ
  female_90_plus : ℕ

/-- Theorem representing the problem and its solution --/
theorem college_running_survey (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
    (sample : SampleData) :
    total_students = 1000 →
    male_students = 640 →
    female_students = 360 →
    sample.male_30_60 = 12 →
    sample.male_60_90 = 10 →
    sample.male_90_plus = 5 →
    sample.female_0_30 = 6 →
    sample.female_30_60 = 6 →
    sample.female_60_90 = 4 →
    sample.female_90_plus = 2 →
    (∃ (a : ℕ),
      sample.male_0_30 = a ∧
      a = 5 ∧
      ((a + 12 + 10 + 5 : ℚ) / (6 + 6 + 4 + 2) = 640 / 360) ∧
      (a * 1000 / (a + 12 + 10 + 5 + 6 + 6 + 4 + 2) = 100)) ∧
    (∃ (X : Fin 4 → ℚ),
      X 1 = 1/7 ∧ X 2 = 4/7 ∧ X 3 = 2/7 ∧
      (X 1 + X 2 + X 3 = 1) ∧
      (1 * X 1 + 2 * X 2 + 3 * X 3 = 15/7)) := by
  sorry


end college_running_survey_l3362_336259


namespace quadratic_equation_roots_l3362_336210

/-- Given a quadratic equation x^2 + (a+1)x + 4 = 0 with roots x₁ and x₂, where x₁ = 1 + √3i and a ∈ ℝ,
    prove that a = -3 and the distance between the points corresponding to x₁ and x₂ in the complex plane is 2√3. -/
theorem quadratic_equation_roots (a : ℝ) (x₁ x₂ : ℂ) : 
  x₁^2 + (a+1)*x₁ + 4 = 0 ∧ 
  x₂^2 + (a+1)*x₂ + 4 = 0 ∧
  x₁ = 1 + Complex.I * Real.sqrt 3 →
  a = -3 ∧ 
  Complex.abs (x₁ - x₂) = Real.sqrt 12 :=
by sorry

end quadratic_equation_roots_l3362_336210


namespace problem_solution_l3362_336211

theorem problem_solution : 
  (Real.sqrt 32 + 4 * Real.sqrt (1/2) - Real.sqrt 18 = 3 * Real.sqrt 2) ∧ 
  ((7 - 4 * Real.sqrt 3) * (7 + 4 * Real.sqrt 3) - (Real.sqrt 3 - 1)^2 + (1/3)⁻¹ = 2 * Real.sqrt 3) := by
  sorry

end problem_solution_l3362_336211


namespace vicente_spent_2475_l3362_336224

/-- Calculates the total amount spent by Vicente on rice and meat --/
def total_spent (rice_kg : ℕ) (rice_price : ℚ) (rice_discount : ℚ)
                (meat_lbs : ℕ) (meat_price : ℚ) (meat_tax : ℚ) : ℚ :=
  let rice_cost := rice_kg * rice_price * (1 - rice_discount)
  let meat_cost := meat_lbs * meat_price * (1 + meat_tax)
  rice_cost + meat_cost

/-- Theorem stating that Vicente's total spent is $24.75 --/
theorem vicente_spent_2475 :
  total_spent 5 2 (1/10) 3 5 (1/20) = 2475/100 := by
  sorry

end vicente_spent_2475_l3362_336224


namespace f_min_value_f_max_value_tangent_line_equation_l3362_336256

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem for the minimum value
theorem f_min_value : ∃ x₀ ∈ Set.Icc (-2 : ℝ) 1, ∀ x ∈ Set.Icc (-2 : ℝ) 1, f x₀ ≤ f x ∧ f x₀ = -2 := by sorry

-- Theorem for the maximum value
theorem f_max_value : ∃ x₀ ∈ Set.Icc (-2 : ℝ) 1, ∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ f x₀ ∧ f x₀ = 2 := by sorry

-- Theorem for the tangent line equation
theorem tangent_line_equation : 
  let P : ℝ × ℝ := (2, -6)
  let tangent_line (x y : ℝ) : Prop := 24 * x - y - 54 = 0
  ∀ x y : ℝ, tangent_line x y ↔ (y - f P.1 = (3 * P.1^2 - 3) * (x - P.1)) := by sorry

end f_min_value_f_max_value_tangent_line_equation_l3362_336256


namespace odd_function_property_l3362_336242

-- Define the domain D
def D : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the properties of the function f
def is_odd_function_on_D (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ D, f (-x) = -f x

-- State the theorem
theorem odd_function_property
  (f : ℝ → ℝ)
  (h_odd : is_odd_function_on_D f)
  (h_pos : ∀ x > 0, f x = x^2 - x) :
  ∀ x < 0, f x = -x^2 - x :=
sorry

end odd_function_property_l3362_336242


namespace function_minimum_implies_a_range_l3362_336247

theorem function_minimum_implies_a_range :
  ∀ (a : ℝ),
  (∀ (x : ℝ), (a * (Real.cos x)^2 - 3) * Real.sin x ≥ -3) →
  (∃ (x : ℝ), (a * (Real.cos x)^2 - 3) * Real.sin x = -3) →
  a ∈ Set.Icc (-3/2) 12 :=
by sorry

end function_minimum_implies_a_range_l3362_336247
