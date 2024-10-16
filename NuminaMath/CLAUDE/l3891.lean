import Mathlib

namespace NUMINAMATH_CALUDE_cos_2A_value_l3891_389190

theorem cos_2A_value (A : Real) (h1 : 0 < A ∧ A < π / 2) 
  (h2 : 3 * Real.cos A - 8 * Real.tan A = 0) : 
  Real.cos (2 * A) = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_cos_2A_value_l3891_389190


namespace NUMINAMATH_CALUDE_urn_problem_solution_l3891_389178

/-- The number of blue balls in the second urn -/
def N : ℕ := 144

/-- The probability of drawing two balls of the same color -/
def same_color_probability : ℚ := 29/50

theorem urn_problem_solution :
  let urn1_green : ℕ := 4
  let urn1_blue : ℕ := 6
  let urn2_green : ℕ := 16
  let urn1_total : ℕ := urn1_green + urn1_blue
  let urn2_total : ℕ := urn2_green + N
  let same_green : ℕ := urn1_green * urn2_green
  let same_blue : ℕ := urn1_blue * N
  let total_outcomes : ℕ := urn1_total * urn2_total
  (same_green + same_blue : ℚ) / total_outcomes = same_color_probability :=
by sorry

end NUMINAMATH_CALUDE_urn_problem_solution_l3891_389178


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l3891_389145

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the foci of the hyperbola
def foci (F₁ F₂ : ℝ × ℝ) : Prop := ∃ c : ℝ, c > 0 ∧ F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop := is_on_hyperbola P.1 P.2

-- Define the right angle condition
def right_angle (F₁ P F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 =
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2

-- State the theorem
theorem hyperbola_triangle_area
  (F₁ F₂ P : ℝ × ℝ)
  (h_foci : foci F₁ F₂)
  (h_on_hyperbola : point_on_hyperbola P)
  (h_right_angle : right_angle F₁ P F₂) :
  (abs ((F₁.1 - P.1) * (F₂.2 - P.2) - (F₁.2 - P.2) * (F₂.1 - P.1))) / 2 = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l3891_389145


namespace NUMINAMATH_CALUDE_solve_equation_l3891_389109

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 4 / 3 → x = -27 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3891_389109


namespace NUMINAMATH_CALUDE_sine_increasing_for_acute_angles_l3891_389160

theorem sine_increasing_for_acute_angles (α β : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < Real.pi / 2) : 
  Real.sin α < Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_sine_increasing_for_acute_angles_l3891_389160


namespace NUMINAMATH_CALUDE_min_value_expression_l3891_389169

theorem min_value_expression (n : ℕ+) : 
  (3 * n : ℝ) / 4 + 32 / (n^2 : ℝ) ≥ 5 ∧ 
  (∃ n : ℕ+, (3 * n : ℝ) / 4 + 32 / (n^2 : ℝ) = 5) := by
  sorry

#check min_value_expression

end NUMINAMATH_CALUDE_min_value_expression_l3891_389169


namespace NUMINAMATH_CALUDE_negative_negative_one_plus_abs_negative_one_equals_two_l3891_389193

theorem negative_negative_one_plus_abs_negative_one_equals_two : 
  -(-1) + |-1| = 2 := by sorry

end NUMINAMATH_CALUDE_negative_negative_one_plus_abs_negative_one_equals_two_l3891_389193


namespace NUMINAMATH_CALUDE_power_3_2048_mod_11_l3891_389125

theorem power_3_2048_mod_11 : 3^2048 ≡ 5 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_power_3_2048_mod_11_l3891_389125


namespace NUMINAMATH_CALUDE_inequality_solution_sum_l3891_389114

/-- Given an inequality ax^2 - 3x + 2 > 0 with solution set {x | x < 1 or x > b}, prove a + b = 3 -/
theorem inequality_solution_sum (a b : ℝ) : 
  (∀ x, ax^2 - 3*x + 2 > 0 ↔ (x < 1 ∨ x > b)) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_sum_l3891_389114


namespace NUMINAMATH_CALUDE_barbell_cost_l3891_389106

theorem barbell_cost (number_of_barbells : ℕ) (amount_paid : ℕ) (change_received : ℕ) :
  number_of_barbells = 3 ∧ amount_paid = 850 ∧ change_received = 40 →
  (amount_paid - change_received) / number_of_barbells = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_barbell_cost_l3891_389106


namespace NUMINAMATH_CALUDE_algebraic_simplification_l3891_389165

theorem algebraic_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 / a) * (a * b / 4) = b / 2 ∧
  -6 * a * b / ((3 * b^2) / (2 * a)) = -4 * a^2 / b := by sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l3891_389165


namespace NUMINAMATH_CALUDE_letters_written_in_ten_hours_l3891_389140

/-- The number of letters Nathan can write in one hour -/
def nathanRate : ℕ := 25

/-- The number of letters Jacob can write in one hour -/
def jacobRate : ℕ := 2 * nathanRate

/-- The number of hours they write together -/
def totalHours : ℕ := 10

/-- The total number of letters Jacob and Nathan can write together in the given time -/
def totalLetters : ℕ := (nathanRate + jacobRate) * totalHours

theorem letters_written_in_ten_hours : totalLetters = 750 := by
  sorry

end NUMINAMATH_CALUDE_letters_written_in_ten_hours_l3891_389140


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l3891_389175

theorem binomial_coefficient_equation_solution : 
  ∃! n : ℕ, (Nat.choose 25 n) + (Nat.choose 25 12) = (Nat.choose 26 13) ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l3891_389175


namespace NUMINAMATH_CALUDE_chicken_pizza_menu_combinations_l3891_389139

theorem chicken_pizza_menu_combinations : 
  let chicken_types : ℕ := 4
  let pizza_types : ℕ := 3
  let same_chicken_diff_pizza := chicken_types * (pizza_types * (pizza_types - 1))
  let same_pizza_diff_chicken := pizza_types * (chicken_types * (chicken_types - 1))
  same_chicken_diff_pizza + same_pizza_diff_chicken = 60 :=
by sorry

end NUMINAMATH_CALUDE_chicken_pizza_menu_combinations_l3891_389139


namespace NUMINAMATH_CALUDE_river_crossing_drift_l3891_389186

/-- Given a river crossing scenario, calculate the drift of the boat. -/
theorem river_crossing_drift (river_width : ℝ) (boat_speed : ℝ) (crossing_time : ℝ) 
  (h1 : river_width = 400)
  (h2 : boat_speed = 10)
  (h3 : crossing_time = 50) :
  boat_speed * crossing_time - river_width = 100 := by
  sorry

#check river_crossing_drift

end NUMINAMATH_CALUDE_river_crossing_drift_l3891_389186


namespace NUMINAMATH_CALUDE_line_of_intersection_canonical_equation_l3891_389113

/-- Given two planes in 3D space, this theorem states that their line of intersection 
    can be represented by a specific canonical equation. -/
theorem line_of_intersection_canonical_equation 
  (plane1 : x + 5*y + 2*z = 5) 
  (plane2 : 2*x - 5*y - z = -5) :
  ∃ (t : ℝ), x = 5*t ∧ y = 5*t + 1 ∧ z = -15*t :=
by sorry

end NUMINAMATH_CALUDE_line_of_intersection_canonical_equation_l3891_389113


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l3891_389182

/-- Given that the equation m/(x-2) + 1 = x/(2-x) has a non-negative solution for x,
    prove that the range of values for m is m ≤ 2 and m ≠ -2. -/
theorem fractional_equation_solution_range (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ m / (x - 2) + 1 = x / (2 - x)) →
  m ≤ 2 ∧ m ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l3891_389182


namespace NUMINAMATH_CALUDE_min_matchsticks_removal_theorem_l3891_389107

/-- Represents a configuration of matchsticks forming triangles -/
structure MatchstickConfiguration where
  total_matchsticks : ℕ
  total_triangles : ℕ

/-- Represents the minimum number of matchsticks to remove -/
def min_matchsticks_to_remove (config : MatchstickConfiguration) : ℕ := sorry

/-- The theorem to be proved -/
theorem min_matchsticks_removal_theorem (config : MatchstickConfiguration) 
  (h1 : config.total_matchsticks = 42)
  (h2 : config.total_triangles = 38) :
  min_matchsticks_to_remove config ≥ 12 := by sorry

end NUMINAMATH_CALUDE_min_matchsticks_removal_theorem_l3891_389107


namespace NUMINAMATH_CALUDE_point_in_region_l3891_389118

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (a, a + 1)

-- Define the inequality that represents the region
def in_region (x y a : ℝ) : Prop := x + a * y - 3 > 0

-- Theorem statement
theorem point_in_region (a : ℝ) :
  in_region (P a).1 (P a).2 a ↔ a < -3 ∨ a > 1 :=
sorry

end NUMINAMATH_CALUDE_point_in_region_l3891_389118


namespace NUMINAMATH_CALUDE_ball_radius_is_10_ball_surface_area_is_400pi_l3891_389170

/-- Represents a spherical ball floating on water that leaves a circular hole in ice --/
structure FloatingBall where
  /-- The radius of the circular hole left in the ice --/
  holeRadius : ℝ
  /-- The depth of the hole left in the ice --/
  holeDepth : ℝ
  /-- The radius of the ball --/
  ballRadius : ℝ

/-- The properties of the floating ball problem --/
def floatingBallProblem : FloatingBall where
  holeRadius := 6
  holeDepth := 2
  ballRadius := 10

/-- Theorem stating that the radius of the ball is 10 cm --/
theorem ball_radius_is_10 (ball : FloatingBall) :
  ball.holeRadius = 6 ∧ ball.holeDepth = 2 → ball.ballRadius = 10 := by sorry

/-- Theorem stating that the surface area of the ball is 400π cm² --/
theorem ball_surface_area_is_400pi (ball : FloatingBall) :
  ball.ballRadius = 10 → 4 * Real.pi * ball.ballRadius ^ 2 = 400 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ball_radius_is_10_ball_surface_area_is_400pi_l3891_389170


namespace NUMINAMATH_CALUDE_find_b_value_l3891_389126

theorem find_b_value (a b : ℝ) (h1 : 2 * a + 1 = 1) (h2 : b - a = 1) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l3891_389126


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_through_point_l3891_389171

/-- A line in the plane can be represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if its coordinates satisfy the line equation -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem y_intercept_of_parallel_line_through_point 
  (l1 : Line) (p : Point) :
  l1.slope = 3 →
  parallel l1 { slope := 3, yIntercept := -2 } →
  pointOnLine p l1 →
  p.x = 5 →
  p.y = 7 →
  l1.yIntercept = -8 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_through_point_l3891_389171


namespace NUMINAMATH_CALUDE_first_month_sales_l3891_389196

def sales_month_2 : ℕ := 5744
def sales_month_3 : ℕ := 5864
def sales_month_4 : ℕ := 6122
def sales_month_5 : ℕ := 6588
def sales_month_6 : ℕ := 4916
def average_sale : ℕ := 5750

theorem first_month_sales :
  sales_month_2 + sales_month_3 + sales_month_4 + sales_month_5 + sales_month_6 + 5266 = 6 * average_sale :=
by sorry

end NUMINAMATH_CALUDE_first_month_sales_l3891_389196


namespace NUMINAMATH_CALUDE_F_6_indeterminate_l3891_389121

theorem F_6_indeterminate (F : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (F k → F (k + 1)))
  (h2 : ¬ F 7) :
  (F 6 ∨ ¬ F 6) :=
sorry

end NUMINAMATH_CALUDE_F_6_indeterminate_l3891_389121


namespace NUMINAMATH_CALUDE_optimal_road_network_l3891_389172

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a configuration of 4 observation stations -/
structure Configuration where
  stations : Fin 4 → Point
  valid : ∀ i, 0 ≤ (stations i).x ∧ (stations i).x ≤ 10 ∧ 0 ≤ (stations i).y ∧ (stations i).y ≤ 10

/-- Represents a network of roads -/
structure RoadNetwork where
  horizontal : List ℝ  -- y-coordinates of horizontal roads
  vertical : List ℝ    -- x-coordinates of vertical roads

/-- Checks if a road network connects all stations to both top and bottom edges -/
def connects (c : Configuration) (n : RoadNetwork) : Prop :=
  ∀ i, ∃ h v,
    h ∈ n.horizontal ∧ v ∈ n.vertical ∧
    ((c.stations i).x = v ∨ (c.stations i).y = h)

/-- Calculates the total length of a road network -/
def networkLength (n : RoadNetwork) : ℝ :=
  (n.horizontal.length * 10 : ℝ) + (n.vertical.sum : ℝ)

/-- The main theorem to be proved -/
theorem optimal_road_network :
  (∀ c : Configuration, ∃ n : RoadNetwork, connects c n ∧ networkLength n ≤ 25) ∧
  (∀ ε > 0, ∃ c : Configuration, ∀ n : RoadNetwork, connects c n → networkLength n > 25 - ε) :=
sorry

end NUMINAMATH_CALUDE_optimal_road_network_l3891_389172


namespace NUMINAMATH_CALUDE_sunglasses_cap_probability_l3891_389180

theorem sunglasses_cap_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (prob_cap_and_sunglasses : ℚ) :
  total_sunglasses = 50 →
  total_caps = 35 →
  prob_cap_and_sunglasses = 2/5 →
  (prob_cap_and_sunglasses * total_caps : ℚ) / total_sunglasses = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_cap_probability_l3891_389180


namespace NUMINAMATH_CALUDE_hiker_final_distance_l3891_389111

-- Define the hiker's movements
def east_distance : ℝ := 15
def south_distance : ℝ := 20
def west_distance : ℝ := 15
def north_distance : ℝ := 5

-- Define the net horizontal and vertical movements
def net_horizontal : ℝ := east_distance - west_distance
def net_vertical : ℝ := south_distance - north_distance

-- Theorem to prove
theorem hiker_final_distance :
  Real.sqrt (net_horizontal ^ 2 + net_vertical ^ 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_hiker_final_distance_l3891_389111


namespace NUMINAMATH_CALUDE_ivy_coverage_l3891_389104

/-- The amount of ivy Cary strips each day, in feet -/
def daily_strip : ℕ := 6

/-- The amount of ivy that grows back each night, in feet -/
def nightly_growth : ℕ := 2

/-- The number of days it takes Cary to strip all the ivy -/
def days_to_strip : ℕ := 10

/-- The net amount of ivy stripped per day, in feet -/
def net_strip_per_day : ℕ := daily_strip - nightly_growth

/-- The total amount of ivy covering the tree, in feet -/
def total_ivy : ℕ := net_strip_per_day * days_to_strip

theorem ivy_coverage : total_ivy = 40 := by
  sorry

end NUMINAMATH_CALUDE_ivy_coverage_l3891_389104


namespace NUMINAMATH_CALUDE_fraction_equation_implies_value_l3891_389134

theorem fraction_equation_implies_value (x : ℝ) : 
  (3 / (x - 3) + 5 / (2 * x - 6) = 11 / 2) → (2 * x - 6 = 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_implies_value_l3891_389134


namespace NUMINAMATH_CALUDE_company_capital_expenditure_l3891_389159

theorem company_capital_expenditure (C : ℚ) (C_pos : C > 0) :
  let raw_material_cost : ℚ := C / 4
  let remaining_after_raw : ℚ := C - raw_material_cost
  let machinery_cost : ℚ := remaining_after_raw / 10
  C - raw_material_cost - machinery_cost = (27 / 40) * C := by
  sorry

end NUMINAMATH_CALUDE_company_capital_expenditure_l3891_389159


namespace NUMINAMATH_CALUDE_crayons_count_l3891_389151

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 41

/-- The number of crayons Sam added to the drawer -/
def added_crayons : ℕ := 12

/-- The total number of crayons after Sam's addition -/
def total_crayons : ℕ := initial_crayons + added_crayons

theorem crayons_count : total_crayons = 53 := by
  sorry

end NUMINAMATH_CALUDE_crayons_count_l3891_389151


namespace NUMINAMATH_CALUDE_marie_sold_925_reading_materials_l3891_389123

/-- The total number of reading materials Marie sold -/
def total_reading_materials (magazines newspapers books pamphlets : ℕ) : ℕ :=
  magazines + newspapers + books + pamphlets

/-- Theorem stating that Marie sold 925 reading materials -/
theorem marie_sold_925_reading_materials :
  total_reading_materials 425 275 150 75 = 925 := by
  sorry

end NUMINAMATH_CALUDE_marie_sold_925_reading_materials_l3891_389123


namespace NUMINAMATH_CALUDE_encoded_value_of_CBD_l3891_389173

/-- Represents the encoding of a base-5 digit --/
inductive Encoding : Type
| A | B | C | D | E

/-- Converts an Encoding to its corresponding base-5 digit --/
def encoding_to_digit (e : Encoding) : Nat :=
  match e with
  | Encoding.A => 2
  | Encoding.B => 3
  | Encoding.C => 4
  | Encoding.D => 0
  | Encoding.E => 1

/-- Converts a sequence of Encodings to its base-10 value --/
def encode_to_base10 (seq : List Encoding) : Nat :=
  seq.enum.foldr (fun (i, e) acc => acc + (encoding_to_digit e) * (5 ^ i)) 0

theorem encoded_value_of_CBD :
  let encoded_seq : List Encoding := [Encoding.C, Encoding.B, Encoding.D]
  encode_to_base10 encoded_seq = 115 :=
by sorry


end NUMINAMATH_CALUDE_encoded_value_of_CBD_l3891_389173


namespace NUMINAMATH_CALUDE_polynomial_determination_l3891_389135

/-- Given a polynomial Q(x) = x^4 - 2x^3 + 3x^2 + kx + m, where k and m are constants,
    prove that if Q(0) = 16 and Q(1) = 2, then Q(x) = x^4 - 2x^3 + 3x^2 - 16x + 16 -/
theorem polynomial_determination (k m : ℝ) : 
  let Q := fun (x : ℝ) => x^4 - 2*x^3 + 3*x^2 + k*x + m
  (Q 0 = 16) → (Q 1 = 2) → 
  (∀ x, Q x = x^4 - 2*x^3 + 3*x^2 - 16*x + 16) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_determination_l3891_389135


namespace NUMINAMATH_CALUDE_calculation_proof_l3891_389154

theorem calculation_proof : 0.54 - (1/8 : ℚ) + 0.46 - (7/8 : ℚ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3891_389154


namespace NUMINAMATH_CALUDE_equation_is_pair_of_straight_lines_l3891_389101

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := 3 * x^2 - 12 * y^2 = 0

/-- Definition of a pair of straight lines -/
def is_pair_of_straight_lines (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∧ c ≠ 0 ∧
    ∀ x y, f x y ↔ (a * x + b * y = 0) ∨ (c * x + d * y = 0)

/-- Theorem stating that the equation represents a pair of straight lines -/
theorem equation_is_pair_of_straight_lines :
  is_pair_of_straight_lines equation :=
sorry

end NUMINAMATH_CALUDE_equation_is_pair_of_straight_lines_l3891_389101


namespace NUMINAMATH_CALUDE_product_ab_equals_twelve_l3891_389147

-- Define the set A
def A (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

-- Define the complement of A with respect to ℝ
def complement_A : Set ℝ := {x | x < 3 ∨ x > 4}

-- Theorem statement
theorem product_ab_equals_twelve (a b : ℝ) : 
  A a b ∪ complement_A = Set.univ → a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_equals_twelve_l3891_389147


namespace NUMINAMATH_CALUDE_cab_delay_l3891_389102

theorem cab_delay (usual_time : ℝ) (speed_ratio : ℝ) (delay : ℝ) : 
  usual_time = 60 →
  speed_ratio = 5/6 →
  delay = usual_time * (1 / speed_ratio - 1) →
  delay = 12 := by
sorry

end NUMINAMATH_CALUDE_cab_delay_l3891_389102


namespace NUMINAMATH_CALUDE_f_surjective_and_unique_l3891_389167

def f (x y : ℕ) : ℕ := (x + y - 1) * (x + y - 2) / 2 + y

theorem f_surjective_and_unique :
  ∀ n : ℕ, ∃! (x y : ℕ), f x y = n :=
by sorry

end NUMINAMATH_CALUDE_f_surjective_and_unique_l3891_389167


namespace NUMINAMATH_CALUDE_y_axis_intersection_uniqueness_l3891_389150

theorem y_axis_intersection_uniqueness (f : ℝ → ℝ) : 
  ∃! y, f 0 = y :=
sorry

end NUMINAMATH_CALUDE_y_axis_intersection_uniqueness_l3891_389150


namespace NUMINAMATH_CALUDE_proposition_q_false_l3891_389157

open Real

theorem proposition_q_false (p q : Prop) 
  (hp : ¬ (∃ x : ℝ, (1/10)^(x-3) ≤ cos 2))
  (hpq : ¬((¬p) ∧ q)) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_proposition_q_false_l3891_389157


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l3891_389199

/-- Represents the taxi fare structure -/
structure TaxiFare where
  startupFee : ℝ
  ratePerMile : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.startupFee + tf.ratePerMile * distance

theorem taxi_fare_calculation (tf : TaxiFare) :
  tf.startupFee = 30 ∧ totalFare tf 60 = 150 → totalFare tf 90 = 210 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l3891_389199


namespace NUMINAMATH_CALUDE_investment_ratio_a_to_b_l3891_389161

/-- Given the investment ratios and profit distribution, prove the ratio of investments between A and B -/
theorem investment_ratio_a_to_b :
  ∀ (a b c total_investment total_profit : ℚ),
  -- A and C invested in ratio 3:2
  a / c = 3 / 2 →
  -- Total investment
  total_investment = a + b + c →
  -- Total profit
  total_profit = 60000 →
  -- C's profit
  c / total_investment * total_profit = 20000 →
  -- Prove that A:B = 3:1
  a / b = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_investment_ratio_a_to_b_l3891_389161


namespace NUMINAMATH_CALUDE_parabola_ellipse_tangency_l3891_389143

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/6 + y^2/4 = 1

-- Define the latus rectum of the parabola
def latus_rectum (p : ℝ) (y : ℝ) : Prop := y = -p/2

-- Theorem statement
theorem parabola_ellipse_tangency :
  ∃ (p : ℝ), ∃ (x y : ℝ),
    parabola p x y ∧
    ellipse x y ∧
    latus_rectum p y ∧
    p = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_ellipse_tangency_l3891_389143


namespace NUMINAMATH_CALUDE_train_crossing_time_l3891_389136

/-- Given a train traveling at 72 kmph that passes a man on a 260-meter platform in 17 seconds,
    the time taken for the train to cross the entire platform is 30 seconds. -/
theorem train_crossing_time (train_speed_kmph : ℝ) (man_crossing_time : ℝ) (platform_length : ℝ) :
  train_speed_kmph = 72 →
  man_crossing_time = 17 →
  platform_length = 260 →
  (platform_length + train_speed_kmph * 1000 / 3600 * man_crossing_time) / (train_speed_kmph * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3891_389136


namespace NUMINAMATH_CALUDE_tank_capacity_l3891_389195

theorem tank_capacity (x : ℝ) (h : 0.5 * x = 75) : x = 150 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l3891_389195


namespace NUMINAMATH_CALUDE_max_area_rectangular_garden_l3891_389198

/-- The maximum area of a rectangular garden enclosed by a fence of length 36m is 81 m² -/
theorem max_area_rectangular_garden : 
  ∀ x y : ℝ, x > 0 → y > 0 → 2*(x + y) = 36 → x*y ≤ 81 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*(x + y) = 36 ∧ x*y = 81 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_garden_l3891_389198


namespace NUMINAMATH_CALUDE_total_carrots_eq_101_l3891_389117

/-- The number of carrots grown by Joan -/
def joan_carrots : ℕ := 29

/-- The number of watermelons grown by Joan -/
def joan_watermelons : ℕ := 14

/-- The number of carrots grown by Jessica -/
def jessica_carrots : ℕ := 11

/-- The number of cantaloupes grown by Jessica -/
def jessica_cantaloupes : ℕ := 9

/-- The number of carrots grown by Michael -/
def michael_carrots : ℕ := 37

/-- The number of carrots grown by Taylor -/
def taylor_carrots : ℕ := 24

/-- The number of cantaloupes grown by Taylor -/
def taylor_cantaloupes : ℕ := 3

/-- The total number of carrots grown by all -/
def total_carrots : ℕ := joan_carrots + jessica_carrots + michael_carrots + taylor_carrots

theorem total_carrots_eq_101 : total_carrots = 101 :=
by sorry

end NUMINAMATH_CALUDE_total_carrots_eq_101_l3891_389117


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3891_389122

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - x < 0) → (-1 < x ∧ x < 1) ∧
  ∃ y : ℝ, -1 < y ∧ y < 1 ∧ ¬(y^2 - y < 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3891_389122


namespace NUMINAMATH_CALUDE_car_distance_covered_l3891_389115

/-- Prove that a car traveling at 195 km/h for 3 1/5 hours covers a distance of 624 km. -/
theorem car_distance_covered (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 195 → time = 3 + 1 / 5 → distance = speed * time → distance = 624 := by
sorry

end NUMINAMATH_CALUDE_car_distance_covered_l3891_389115


namespace NUMINAMATH_CALUDE_fraction_simplification_l3891_389130

theorem fraction_simplification : (8 + 4) / (8 - 4) = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3891_389130


namespace NUMINAMATH_CALUDE_entertainment_budget_percentage_l3891_389116

/-- Proves that given a budget of $1000, with 30% spent on food, 15% on accommodation,
    $300 on coursework materials, the remaining percentage spent on entertainment is 25%. -/
theorem entertainment_budget_percentage
  (total_budget : ℝ)
  (food_percentage : ℝ)
  (accommodation_percentage : ℝ)
  (coursework_materials : ℝ)
  (h1 : total_budget = 1000)
  (h2 : food_percentage = 30)
  (h3 : accommodation_percentage = 15)
  (h4 : coursework_materials = 300) :
  (total_budget - (food_percentage / 100 * total_budget + 
   accommodation_percentage / 100 * total_budget + coursework_materials)) / 
   total_budget * 100 = 25 := by
  sorry

#check entertainment_budget_percentage

end NUMINAMATH_CALUDE_entertainment_budget_percentage_l3891_389116


namespace NUMINAMATH_CALUDE_simplify_expression_l3891_389108

theorem simplify_expression (x : ℝ) : 5*x + 6 - x + 12 = 4*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3891_389108


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3891_389163

theorem consecutive_even_numbers_sum (n : ℤ) : 
  (∃ (a b c d : ℤ), 
    a = n ∧ 
    b = n + 2 ∧ 
    c = n + 4 ∧ 
    d = n + 6 ∧ 
    a + b + c + d = 52) → 
  n + 4 = 14 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3891_389163


namespace NUMINAMATH_CALUDE_optimal_transport_solution_l3891_389188

/-- Represents the optimal solution for transporting cargo -/
structure CargoTransport where
  large_trucks : ℕ
  small_trucks : ℕ
  total_fuel : ℕ

/-- Finds the optimal cargo transport solution -/
def find_optimal_transport (total_cargo : ℕ) (large_capacity : ℕ) (small_capacity : ℕ)
  (large_fuel : ℕ) (small_fuel : ℕ) : CargoTransport :=
  sorry

/-- Theorem stating the optimal solution for the given problem -/
theorem optimal_transport_solution :
  let total_cargo : ℕ := 89
  let large_capacity : ℕ := 7
  let small_capacity : ℕ := 4
  let large_fuel : ℕ := 14
  let small_fuel : ℕ := 9
  let solution := find_optimal_transport total_cargo large_capacity small_capacity large_fuel small_fuel
  solution.total_fuel = 181 ∧
  solution.large_trucks * large_capacity + solution.small_trucks * small_capacity ≥ total_cargo :=
by sorry

end NUMINAMATH_CALUDE_optimal_transport_solution_l3891_389188


namespace NUMINAMATH_CALUDE_five_fridays_september_implies_five_mondays_october_l3891_389144

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month -/
structure Month where
  days : Nat
  first_day : DayOfWeek

/-- Given a day, returns the next day of the week -/
def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the occurrences of a specific day in a month -/
def count_day_occurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: If September has five Fridays, October has five Mondays -/
theorem five_fridays_september_implies_five_mondays_october 
  (september : Month) 
  (october : Month) :
  september.days = 30 →
  october.days = 31 →
  count_day_occurrences september DayOfWeek.Friday = 5 →
  count_day_occurrences october DayOfWeek.Monday = 5 :=
  sorry

end NUMINAMATH_CALUDE_five_fridays_september_implies_five_mondays_october_l3891_389144


namespace NUMINAMATH_CALUDE_sum_product_bounds_l3891_389149

theorem sum_product_bounds (a b c k : ℝ) (h : a + b + c = k) (h_nonzero : k ≠ 0) :
  -2/3 * k^2 ≤ a*b + a*c + b*c ∧ a*b + a*c + b*c ≤ k^2/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_bounds_l3891_389149


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l3891_389119

/-- Given 6 people in an elevator with an average weight of 154 lbs, 
    if a 7th person enters and the new average weight becomes 151 lbs, 
    then the weight of the 7th person is 133 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
    (final_people : ℕ) (final_avg_weight : ℝ) : 
    initial_people = 6 → 
    initial_avg_weight = 154 → 
    final_people = 7 → 
    final_avg_weight = 151 → 
    (initial_people * initial_avg_weight + 
      (final_people - initial_people) * 
      ((final_people * final_avg_weight) - (initial_people * initial_avg_weight))) / 
      (final_people - initial_people) = 133 := by
  sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l3891_389119


namespace NUMINAMATH_CALUDE_boys_pass_percentage_l3891_389124

/-- Proves that 28% of boys passed the examination given the problem conditions -/
theorem boys_pass_percentage (total_candidates : ℕ) (girls : ℕ) (girls_pass_rate : ℚ) (total_fail_rate : ℚ) :
  total_candidates = 2000 →
  girls = 900 →
  girls_pass_rate = 32 / 100 →
  total_fail_rate = 702 / 1000 →
  let boys := total_candidates - girls
  let total_pass := total_candidates * (1 - total_fail_rate)
  let girls_pass := girls * girls_pass_rate
  let boys_pass := total_pass - girls_pass
  (boys_pass / boys : ℚ) = 28 / 100 := by sorry

end NUMINAMATH_CALUDE_boys_pass_percentage_l3891_389124


namespace NUMINAMATH_CALUDE_davids_math_marks_l3891_389197

theorem davids_math_marks
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (num_subjects : ℕ)
  (h1 : english_marks = 70)
  (h2 : physics_marks = 80)
  (h3 : chemistry_marks = 63)
  (h4 : biology_marks = 65)
  (h5 : average_marks = 68.2)
  (h6 : num_subjects = 5) :
  ∃ math_marks : ℕ,
    math_marks = 63 ∧
    (english_marks + physics_marks + chemistry_marks + biology_marks + math_marks : ℚ) / num_subjects = average_marks :=
by sorry

end NUMINAMATH_CALUDE_davids_math_marks_l3891_389197


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l3891_389176

theorem lcm_gcd_product (a b : ℕ) (ha : a = 30) (hb : b = 75) :
  Nat.lcm a b * Nat.gcd a b = a * b := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l3891_389176


namespace NUMINAMATH_CALUDE_complex_product_theorem_l3891_389142

theorem complex_product_theorem : 
  let z1 : ℂ := -1 + 2*I
  let z2 : ℂ := 2 + I
  z1 * z2 = -4 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l3891_389142


namespace NUMINAMATH_CALUDE_sequence_sum_l3891_389177

/-- Given a sequence {a_n} where a_1 = 1 and S_n = n^2 * a_n for all positive integers n,
    prove that S_n = 2n / (n+1) for all positive integers n. -/
theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) :
  a 1 = 1 →
  (∀ n : ℕ, n > 0 → S n = n^2 * a n) →
  ∀ n : ℕ, n > 0 → S n = 2 * n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3891_389177


namespace NUMINAMATH_CALUDE_onion_rings_cost_l3891_389156

/-- Proves that the cost of onion rings is $2 given the costs of other items and payment details --/
theorem onion_rings_cost (hamburger_cost smoothie_cost total_paid change : ℕ) :
  hamburger_cost = 4 →
  smoothie_cost = 3 →
  total_paid = 20 →
  change = 11 →
  total_paid - change - hamburger_cost - smoothie_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_onion_rings_cost_l3891_389156


namespace NUMINAMATH_CALUDE_sqrt_12_same_type_as_sqrt_3_l3891_389191

def is_same_type (a b : ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ a = k * b

theorem sqrt_12_same_type_as_sqrt_3 :
  let options := [Real.sqrt 8, Real.sqrt 12, Real.sqrt 18, Real.sqrt 6]
  ∃ (x : ℝ), x ∈ options ∧ is_same_type x (Real.sqrt 3) ∧
    ∀ (y : ℝ), y ∈ options → y ≠ x → ¬(is_same_type y (Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_same_type_as_sqrt_3_l3891_389191


namespace NUMINAMATH_CALUDE_edward_games_from_friend_l3891_389181

/-- The number of games Edward bought from his friend -/
def games_from_friend : ℕ := sorry

/-- The number of games Edward bought at the garage sale -/
def games_from_garage_sale : ℕ := 14

/-- The number of games that didn't work -/
def non_working_games : ℕ := 31

/-- The number of good games Edward ended up with -/
def good_games : ℕ := 24

theorem edward_games_from_friend :
  games_from_friend = 41 :=
by
  have h1 : games_from_friend + games_from_garage_sale - non_working_games = good_games := by sorry
  sorry

end NUMINAMATH_CALUDE_edward_games_from_friend_l3891_389181


namespace NUMINAMATH_CALUDE_base_3_to_decimal_l3891_389168

/-- Converts a list of digits in base k to its decimal representation -/
def to_decimal (digits : List Nat) (k : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * k^i) 0

/-- The base-3 representation of a number -/
def base_3_number : List Nat := [1, 0, 2]

/-- Theorem stating that the base-3 number (102)₃ is equal to 11 in decimal -/
theorem base_3_to_decimal :
  to_decimal base_3_number 3 = 11 := by sorry

end NUMINAMATH_CALUDE_base_3_to_decimal_l3891_389168


namespace NUMINAMATH_CALUDE_smallest_n_for_given_mean_l3891_389155

theorem smallest_n_for_given_mean : ∃ (n : ℕ) (m : ℕ),
  n > 0 ∧
  m ∈ Finset.range (n + 1) ∧
  (Finset.sum (Finset.range (n + 1) \ {m}) id) / ((n : ℚ) - 1) = 439 / 13 ∧
  ∀ (k : ℕ) (j : ℕ), k > 0 ∧ k < n →
    j ∈ Finset.range (k + 1) →
    (Finset.sum (Finset.range (k + 1) \ {j}) id) / ((k : ℚ) - 1) ≠ 439 / 13 ∧
  n = 68 ∧
  m = 45 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_given_mean_l3891_389155


namespace NUMINAMATH_CALUDE_kaleb_sold_games_l3891_389166

theorem kaleb_sold_games (initial_games : ℕ) (games_per_box : ℕ) (boxes_used : ℕ) 
  (h1 : initial_games = 76)
  (h2 : games_per_box = 5)
  (h3 : boxes_used = 6) :
  initial_games - (games_per_box * boxes_used) = 46 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_sold_games_l3891_389166


namespace NUMINAMATH_CALUDE_sum_zero_implies_opposites_l3891_389129

theorem sum_zero_implies_opposites (a b : ℝ) : a + b = 0 → a = -b := by sorry

end NUMINAMATH_CALUDE_sum_zero_implies_opposites_l3891_389129


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3891_389103

/-- If the roots of the quadratic equation 5x^2 + 4x + k are (-4 ± i√379) / 10, then k = 19.75 -/
theorem quadratic_root_problem (k : ℝ) : 
  (∀ x : ℂ, 5 * x^2 + 4 * x + k = 0 ↔ x = (-4 + ℍ * Real.sqrt 379) / 10 ∨ x = (-4 - ℍ * Real.sqrt 379) / 10) →
  k = 19.75 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3891_389103


namespace NUMINAMATH_CALUDE_maria_backpack_sheets_l3891_389187

/-- The number of sheets of paper in Maria's desk -/
def sheets_in_desk : ℕ := 50

/-- The total number of sheets of paper Maria has -/
def total_sheets : ℕ := 91

/-- The number of sheets of paper in Maria's backpack -/
def sheets_in_backpack : ℕ := total_sheets - sheets_in_desk

theorem maria_backpack_sheets : sheets_in_backpack = 41 := by
  sorry

end NUMINAMATH_CALUDE_maria_backpack_sheets_l3891_389187


namespace NUMINAMATH_CALUDE_perpendicular_parallel_planes_l3891_389183

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_parallel_planes
  (m n : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : parallel_lines m n)
  (h3 : parallel_planes α β) :
  perpendicular n β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_planes_l3891_389183


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l3891_389174

-- Define the quadratic function
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, p a b c x = a * (x - 9.5)^2 + p a b c 9.5) →
  p a b c 0 = -8 →
  p a b c 20 = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l3891_389174


namespace NUMINAMATH_CALUDE_same_color_sock_pairs_l3891_389152

def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem same_color_sock_pairs (white black red : ℕ) 
  (h_white : white = 5) 
  (h_black : black = 4) 
  (h_red : red = 3) : 
  (choose white 2) + (choose black 2) + (choose red 2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_same_color_sock_pairs_l3891_389152


namespace NUMINAMATH_CALUDE_smallest_gcd_l3891_389158

theorem smallest_gcd (m n p : ℕ+) (h1 : Nat.gcd m n = 180) (h2 : Nat.gcd m p = 240) :
  ∃ (n' p' : ℕ+), Nat.gcd m n' = 180 ∧ Nat.gcd m p' = 240 ∧ Nat.gcd n' p' = 60 ∧
  ∀ (n'' p'' : ℕ+), Nat.gcd m n'' = 180 → Nat.gcd m p'' = 240 → Nat.gcd n'' p'' ≥ 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_l3891_389158


namespace NUMINAMATH_CALUDE_max_gcd_of_sum_1155_l3891_389153

theorem max_gcd_of_sum_1155 :
  ∃ (a b : ℕ+), a + b = 1155 ∧
  ∀ (c d : ℕ+), c + d = 1155 → Nat.gcd c d ≤ Nat.gcd a b ∧
  Nat.gcd a b = 105 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_sum_1155_l3891_389153


namespace NUMINAMATH_CALUDE_parallelogram_base_l3891_389146

/-- Given a parallelogram with area 78.88 cm² and height 8 cm, its base is 9.86 cm -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 78.88 ∧ height = 8 ∧ area = base * height → base = 9.86 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l3891_389146


namespace NUMINAMATH_CALUDE_water_in_bucket_l3891_389120

/-- 
Given a bucket with an initial amount of water and an additional amount added,
calculate the total amount of water in the bucket.
-/
theorem water_in_bucket (initial : ℝ) (added : ℝ) :
  initial = 3 → added = 6.8 → initial + added = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_water_in_bucket_l3891_389120


namespace NUMINAMATH_CALUDE_longest_segment_is_BD_l3891_389185

/-- Given a triangle ABC, returns true if AC > AB > BC -/
def triangleInequalityOrder (angleA angleB angleC : ℝ) : Prop :=
  angleA > angleB ∧ angleB > angleC

theorem longest_segment_is_BD 
  (angleABD angleADB angleCBD angleBDC : ℝ)
  (h1 : angleABD = 50)
  (h2 : angleADB = 45)
  (h3 : angleCBD = 70)
  (h4 : angleBDC = 65)
  (h5 : triangleInequalityOrder (180 - angleABD - angleADB) angleABD angleADB)
  (h6 : triangleInequalityOrder angleCBD angleBDC (180 - angleCBD - angleBDC)) :
  ∃ (lengthAB lengthBC lengthCD lengthAD lengthBD : ℝ),
    lengthAD < lengthAB ∧ 
    lengthAB < lengthBC ∧ 
    lengthBC < lengthCD ∧ 
    lengthCD < lengthBD :=
by sorry

end NUMINAMATH_CALUDE_longest_segment_is_BD_l3891_389185


namespace NUMINAMATH_CALUDE_range_of_m_satisfies_conditions_l3891_389184

/-- Given two functions f and g, prove that the range of m satisfies the given conditions -/
theorem range_of_m_satisfies_conditions (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = x^2 + m) →
  (∀ x, g x = 2^x - m) →
  (∀ x₁ ∈ Set.Icc (-1) 2, ∃ x₂ ∈ Set.Icc 0 3, f x₁ = g x₂) →
  m ∈ Set.Icc (1/2) 2 := by
  sorry

#check range_of_m_satisfies_conditions

end NUMINAMATH_CALUDE_range_of_m_satisfies_conditions_l3891_389184


namespace NUMINAMATH_CALUDE_hidden_faces_sum_l3891_389194

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The number of dice -/
def num_dice : ℕ := 4

/-- The visible numbers on the stacked dice -/
def visible_numbers : List ℕ := [1, 2, 2, 3, 3, 4, 5, 6]

/-- The sum of visible numbers -/
def visible_sum : ℕ := visible_numbers.sum

theorem hidden_faces_sum :
  (num_dice * die_sum) - visible_sum = 58 := by
  sorry

end NUMINAMATH_CALUDE_hidden_faces_sum_l3891_389194


namespace NUMINAMATH_CALUDE_remaining_money_l3891_389112

def initialAmount : ℚ := 7.10
def spentOnSweets : ℚ := 1.05
def givenToFriend : ℚ := 1.00
def numberOfFriends : ℕ := 2

theorem remaining_money :
  initialAmount - (spentOnSweets + givenToFriend * numberOfFriends) = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l3891_389112


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l3891_389105

theorem complex_modulus_theorem (ω : ℂ) (h : ω = 8 + I) : 
  Complex.abs (ω^2 - 4*ω + 13) = 4 * Real.sqrt 130 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l3891_389105


namespace NUMINAMATH_CALUDE_amount_ratio_l3891_389138

/-- Prove that the ratio of A's amount to B's amount is 1:3 given the conditions -/
theorem amount_ratio (total amount_B amount_C : ℚ) (h1 : total = 1440)
  (h2 : amount_B = 270) (h3 : amount_B = (1/4) * amount_C) :
  ∃ amount_A : ℚ, amount_A + amount_B + amount_C = total ∧ amount_A = (1/3) * amount_B := by
  sorry

end NUMINAMATH_CALUDE_amount_ratio_l3891_389138


namespace NUMINAMATH_CALUDE_interview_probability_l3891_389148

/-- The total number of students in at least one club -/
def total_students : ℕ := 30

/-- The number of students in the Robotics club -/
def robotics_students : ℕ := 22

/-- The number of students in the Drama club -/
def drama_students : ℕ := 19

/-- The probability of selecting two students who are not both from the same single club -/
theorem interview_probability : 
  (Nat.choose total_students 2 - (Nat.choose (robotics_students + drama_students - total_students) 2 + 
   Nat.choose (drama_students - (robotics_students + drama_students - total_students)) 2)) / 
  Nat.choose total_students 2 = 352 / 435 := by sorry

end NUMINAMATH_CALUDE_interview_probability_l3891_389148


namespace NUMINAMATH_CALUDE_min_value_a_plus_3b_l3891_389110

theorem min_value_a_plus_3b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a * b - 3 = a + 3 * b) :
  ∀ x y, x > 0 → y > 0 → 3 * x * y - 3 = x + 3 * y → a + 3 * b ≤ x + 3 * y :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_3b_l3891_389110


namespace NUMINAMATH_CALUDE_fourth_root_81_times_cube_root_27_times_sqrt_9_equals_27_l3891_389141

theorem fourth_root_81_times_cube_root_27_times_sqrt_9_equals_27 :
  (81 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_81_times_cube_root_27_times_sqrt_9_equals_27_l3891_389141


namespace NUMINAMATH_CALUDE_vip_tickets_count_l3891_389164

theorem vip_tickets_count (initial_savings : ℕ) (vip_ticket_cost : ℕ) (regular_ticket_cost : ℕ) (regular_tickets_count : ℕ) (remaining_money : ℕ) : 
  initial_savings = 500 →
  vip_ticket_cost = 100 →
  regular_ticket_cost = 50 →
  regular_tickets_count = 3 →
  remaining_money = 150 →
  ∃ vip_tickets_count : ℕ, 
    vip_tickets_count * vip_ticket_cost + regular_tickets_count * regular_ticket_cost = initial_savings - remaining_money ∧
    vip_tickets_count = 2 :=
by sorry

end NUMINAMATH_CALUDE_vip_tickets_count_l3891_389164


namespace NUMINAMATH_CALUDE_convention_handshakes_eq_990_l3891_389189

/-- The number of handshakes at the Annual Mischief Convention -/
def convention_handshakes : ℕ :=
  let total_gremlins : ℕ := 30
  let total_imps : ℕ := 20
  let unfriendly_gremlins : ℕ := 10
  let friendly_gremlins : ℕ := total_gremlins - unfriendly_gremlins

  let gremlin_handshakes : ℕ := 
    (friendly_gremlins * (friendly_gremlins - 1)) / 2 + 
    unfriendly_gremlins * friendly_gremlins

  let imp_gremlin_handshakes : ℕ := total_imps * total_gremlins

  gremlin_handshakes + imp_gremlin_handshakes

theorem convention_handshakes_eq_990 : convention_handshakes = 990 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_eq_990_l3891_389189


namespace NUMINAMATH_CALUDE_intersection_line_slope_is_one_third_l3891_389127

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 5 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 16*y + 24 = 0

/-- The slope of the line passing through the intersection points of two circles -/
def intersectionLineSlope (c1 c2 : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem intersection_line_slope_is_one_third :
  intersectionLineSlope circle1 circle2 = 1/3 := by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_is_one_third_l3891_389127


namespace NUMINAMATH_CALUDE_jackson_charity_collection_l3891_389137

-- Define the working days in a week
def working_days : ℕ := 5

-- Define the amount collected on Monday and Tuesday
def monday_collection : ℕ := 300
def tuesday_collection : ℕ := 40

-- Define the average collection per 4 houses
def avg_collection_per_4_houses : ℕ := 10

-- Define the number of houses visited on each remaining day
def houses_per_day : ℕ := 88

-- Define the goal for the week
def weekly_goal : ℕ := 1000

-- Theorem statement
theorem jackson_charity_collection :
  monday_collection + tuesday_collection +
  (working_days - 2) * (houses_per_day / 4 * avg_collection_per_4_houses) =
  weekly_goal := by sorry

end NUMINAMATH_CALUDE_jackson_charity_collection_l3891_389137


namespace NUMINAMATH_CALUDE_lens_circumference_approx_l3891_389162

-- Define π as a constant (approximation)
def π : ℝ := 3.14159

-- Define the diameter of the lens
def d : ℝ := 10

-- Define the circumference calculation function
def circumference (diameter : ℝ) : ℝ := π * diameter

-- Theorem statement
theorem lens_circumference_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |circumference d - 31.42| < ε :=
sorry

end NUMINAMATH_CALUDE_lens_circumference_approx_l3891_389162


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_intersection_equals_B_iff_m_leq_1_l3891_389132

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | 1 - m ≤ x ∧ x ≤ 3 * m - 1}

-- Theorem for part 1
theorem intersection_and_union_when_m_is_3 :
  (A ∩ B 3 = {x : ℝ | -2 ≤ x ∧ x ≤ 2}) ∧
  (A ∪ B 3 = {x : ℝ | -3 ≤ x ∧ x ≤ 8}) := by
  sorry

-- Theorem for part 2
theorem intersection_equals_B_iff_m_leq_1 (m : ℝ) :
  A ∩ B m = B m ↔ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_intersection_equals_B_iff_m_leq_1_l3891_389132


namespace NUMINAMATH_CALUDE_y_divisibility_l3891_389133

def y : ℕ := 32 + 48 + 64 + 96 + 200 + 224 + 1600

theorem y_divisibility :
  (∃ k : ℕ, y = 4 * k) ∧
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 16 * k) ∧
  ¬(∃ k : ℕ, y = 32 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l3891_389133


namespace NUMINAMATH_CALUDE_slope_intercept_sum_horizontal_line_l3891_389128

/-- Given two points with the same y-coordinate and different x-coordinates,
    the sum of the slope and y-intercept of the line containing both points is 20. -/
theorem slope_intercept_sum_horizontal_line (C D : ℝ × ℝ) :
  C.2 = 20 →
  D.2 = 20 →
  C.1 ≠ D.1 →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := C.2 - m * C.1
  m + b = 20 := by
  sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_horizontal_line_l3891_389128


namespace NUMINAMATH_CALUDE_negative_three_a_cubed_squared_l3891_389100

theorem negative_three_a_cubed_squared (a : ℝ) : (-3 * a^3)^2 = 9 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_a_cubed_squared_l3891_389100


namespace NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l3891_389131

/-- Given a hemisphere with base area 144π and a cylindrical extension of height 10 units
    with the same radius as the hemisphere, the total surface area of the combined object is 672π. -/
theorem hemisphere_cylinder_surface_area (r : ℝ) (h : ℝ) :
  r^2 * Real.pi = 144 * Real.pi →
  h = 10 →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h + Real.pi * r^2 = 672 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l3891_389131


namespace NUMINAMATH_CALUDE_factors_of_72_l3891_389179

theorem factors_of_72 : Finset.card (Nat.divisors 72) = 12 := by sorry

end NUMINAMATH_CALUDE_factors_of_72_l3891_389179


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l3891_389192

/-- The minimum distance from a point on the circle x^2 + y^2 - 2x - 2y = 0 to the line x + y - 8 = 0 is 2√2. -/
theorem min_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | (p.1^2 + p.2^2 - 2*p.1 - 2*p.2) = 0}
  let line := {p : ℝ × ℝ | p.1 + p.2 - 8 = 0}
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧
    ∀ (p : ℝ × ℝ), p ∈ circle →
      ∀ (q : ℝ × ℝ), q ∈ line →
        d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l3891_389192
