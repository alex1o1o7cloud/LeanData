import Mathlib

namespace polynomial_identity_l1482_148258

theorem polynomial_identity (a b c : ℝ) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - 2 * a * b * c := by
  sorry

end polynomial_identity_l1482_148258


namespace strawberry_jelly_sales_l1482_148248

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ

/-- Defines the relationships between jelly sales and proves the number of strawberry jelly jars sold -/
theorem strawberry_jelly_sales (sales : JellySales) : 
  sales.grape = 2 * sales.strawberry ∧ 
  sales.raspberry = 2 * sales.plum ∧
  sales.raspberry = sales.grape / 3 ∧
  sales.plum = 6 →
  sales.strawberry = 18 := by
sorry

end strawberry_jelly_sales_l1482_148248


namespace seating_theorem_l1482_148234

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  groups : Nat
  seats_per_group : Nat
  extra_pair : Nat
  total_seats : Nat
  max_customers : Nat

/-- Checks if a seating arrangement is valid --/
def is_valid_arrangement (arr : SeatingArrangement) : Prop :=
  arr.groups * arr.seats_per_group + arr.extra_pair = arr.total_seats ∧
  arr.max_customers ≤ arr.total_seats

/-- Checks if pairs can always be seated adjacently --/
def can_seat_pairs (arr : SeatingArrangement) : Prop :=
  ∀ n : Nat, n ≤ arr.max_customers → 
    (n / 2) * 2 ≤ arr.groups * 2 + arr.extra_pair

theorem seating_theorem (arr : SeatingArrangement) 
  (h1 : arr.groups = 7)
  (h2 : arr.seats_per_group = 3)
  (h3 : arr.extra_pair = 2)
  (h4 : arr.total_seats = 23)
  (h5 : arr.max_customers = 16)
  : is_valid_arrangement arr ∧ can_seat_pairs arr := by
  sorry

#check seating_theorem

end seating_theorem_l1482_148234


namespace three_number_difference_l1482_148209

theorem three_number_difference (x y : ℝ) (h : (23 + x + y) / 3 = 31) :
  max (max 23 x) y - min (min 23 x) y ≥ 17 :=
sorry

end three_number_difference_l1482_148209


namespace hamburgers_for_lunch_l1482_148206

theorem hamburgers_for_lunch (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 9 → additional = 3 → total = initial + additional → total = 12 := by
  sorry

end hamburgers_for_lunch_l1482_148206


namespace joes_trip_expenses_l1482_148224

/-- Joe's trip expenses problem -/
theorem joes_trip_expenses (initial_savings : ℕ) (flight_cost : ℕ) (hotel_cost : ℕ) (remaining : ℕ) 
  (h1 : initial_savings = 6000)
  (h2 : flight_cost = 1200)
  (h3 : hotel_cost = 800)
  (h4 : remaining = 1000) :
  initial_savings - flight_cost - hotel_cost - remaining = 3000 := by
  sorry

end joes_trip_expenses_l1482_148224


namespace husband_additional_payment_l1482_148207

def medical_procedure_1 : ℚ := 128
def medical_procedure_2 : ℚ := 256
def medical_procedure_3 : ℚ := 64
def house_help_salary : ℚ := 160
def tax_rate : ℚ := 0.05

def total_medical_expenses : ℚ := medical_procedure_1 + medical_procedure_2 + medical_procedure_3
def couple_medical_contribution : ℚ := total_medical_expenses / 2
def house_help_medical_contribution : ℚ := min (total_medical_expenses / 2) house_help_salary
def tax_deduction : ℚ := house_help_salary * tax_rate
def total_couple_expense : ℚ := couple_medical_contribution + (total_medical_expenses / 2 - house_help_medical_contribution) + tax_deduction
def husband_paid : ℚ := couple_medical_contribution

theorem husband_additional_payment (
  split_equally : total_couple_expense / 2 < husband_paid
) : husband_paid - total_couple_expense / 2 = 76 := by sorry

end husband_additional_payment_l1482_148207


namespace brick_tower_heights_l1482_148276

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of distinct tower heights achievable -/
def distinctTowerHeights (numBricks : ℕ) (dimensions : BrickDimensions) : ℕ :=
  sorry

/-- Theorem stating the number of distinct tower heights for the given problem -/
theorem brick_tower_heights :
  let dimensions : BrickDimensions := ⟨3, 11, 17⟩
  distinctTowerHeights 62 dimensions = 435 := by
  sorry

end brick_tower_heights_l1482_148276


namespace prize_interval_l1482_148242

/-- Proves that the interval between prizes is 400 given the conditions of the tournament. -/
theorem prize_interval (total_prize : ℕ) (first_prize : ℕ) (interval : ℕ) : 
  total_prize = 4800 → 
  first_prize = 2000 → 
  total_prize = first_prize + (first_prize - interval) + (first_prize - 2 * interval) →
  interval = 400 := by
  sorry

#check prize_interval

end prize_interval_l1482_148242


namespace max_value_expression_max_value_achieved_l1482_148235

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  (a + 3*b + 5*c) * (a + b/3 + c/5) ≤ 9/5 := by
  sorry

theorem max_value_achieved (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  ∃ a₀ b₀ c₀ : ℝ, 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀ + b₀ + c₀ = 1 ∧
    (a₀ + 3*b₀ + 5*c₀) * (a₀ + b₀/3 + c₀/5) = 9/5 := by
  sorry

end max_value_expression_max_value_achieved_l1482_148235


namespace gcd_98_63_l1482_148247

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l1482_148247


namespace line_y_intercept_l1482_148238

/-- Given a line passing through the points (2, -3) and (6, 5), its y-intercept is -7 -/
theorem line_y_intercept : 
  ∀ (f : ℝ → ℝ), 
  (f 2 = -3) → 
  (f 6 = 5) → 
  (∀ x y, f x = y ↔ ∃ m b, y = m * x + b) →
  (∃ b, f 0 = b) →
  f 0 = -7 := by
sorry

end line_y_intercept_l1482_148238


namespace dalmatians_right_spot_count_l1482_148296

/-- The number of Dalmatians with a spot on the right ear -/
def dalmatians_with_right_spot (total : ℕ) (left_only : ℕ) (right_only : ℕ) (no_spots : ℕ) : ℕ :=
  total - left_only - no_spots

/-- Theorem stating the number of Dalmatians with a spot on the right ear -/
theorem dalmatians_right_spot_count :
  dalmatians_with_right_spot 101 29 17 22 = 50 := by
  sorry

#eval dalmatians_with_right_spot 101 29 17 22

end dalmatians_right_spot_count_l1482_148296


namespace fraction_equality_l1482_148283

theorem fraction_equality (w x y : ℚ) 
  (h1 : w / y = 2 / 3)
  (h2 : (x + y) / y = 3) :
  w / x = 1 / 3 := by
sorry

end fraction_equality_l1482_148283


namespace unique_difference_of_squares_1979_l1482_148210

theorem unique_difference_of_squares_1979 : 
  ∃! (x y : ℕ), 1979 = x^2 - y^2 :=
by
  sorry

end unique_difference_of_squares_1979_l1482_148210


namespace trig_sum_equality_l1482_148287

theorem trig_sum_equality : 
  Real.sin (2 * π / 3) ^ 2 + Real.cos π + Real.tan (π / 4) - 
  Real.cos (-11 * π / 6) ^ 2 + Real.sin (-7 * π / 6) = 1 / 2 := by
  sorry

end trig_sum_equality_l1482_148287


namespace spotlight_detection_l1482_148280

/-- Represents the spotlight's properties -/
structure Spotlight where
  illumination_length : ℝ  -- Length of illuminated segment in km
  rotation_period : ℝ      -- Time for one complete rotation in minutes

/-- Represents the boat's properties -/
structure Boat where
  speed : ℝ  -- Speed in km/min

/-- Determines if a boat can approach undetected given a spotlight -/
def can_approach_undetected (s : Spotlight) (b : Boat) : Prop :=
  b.speed ≥ 48.6 / 60  -- Convert 48.6 km/h to km/min

theorem spotlight_detection (s : Spotlight) (b : Boat) :
  s.illumination_length = 1 ∧ s.rotation_period = 1 →
  (b.speed < 800 / 1000 → ¬can_approach_undetected s b) ∧
  (b.speed ≥ 48.6 / 60 → can_approach_undetected s b) := by
  sorry

#check spotlight_detection

end spotlight_detection_l1482_148280


namespace nickel_quarter_problem_l1482_148294

theorem nickel_quarter_problem :
  ∀ (n : ℕ),
    (n : ℚ) * 0.05 + (n : ℚ) * 0.25 = 12 →
    n = 40 :=
by
  sorry

end nickel_quarter_problem_l1482_148294


namespace actual_average_height_l1482_148256

-- Define the problem parameters
def totalStudents : ℕ := 50
def initialAverage : ℚ := 175
def incorrectHeights : List ℚ := [162, 150, 155]
def actualHeights : List ℚ := [142, 135, 145]

-- Define the theorem
theorem actual_average_height :
  let totalInitialHeight : ℚ := initialAverage * totalStudents
  let heightDifference : ℚ := (List.sum incorrectHeights) - (List.sum actualHeights)
  let correctedTotalHeight : ℚ := totalInitialHeight - heightDifference
  let actualAverage : ℚ := correctedTotalHeight / totalStudents
  actualAverage = 174.1 := by
  sorry

end actual_average_height_l1482_148256


namespace well_digging_rate_l1482_148233

/-- Calculates the rate per cubic meter for digging a cylindrical well -/
theorem well_digging_rate (depth : ℝ) (diameter : ℝ) (total_cost : ℝ) : 
  depth = 14 →
  diameter = 3 →
  total_cost = 1583.3626974092558 →
  ∃ (rate : ℝ), abs (rate - 15.993) < 0.001 ∧ 
    rate = total_cost / (Real.pi * (diameter / 2)^2 * depth) := by
  sorry

end well_digging_rate_l1482_148233


namespace tan_product_values_l1482_148277

theorem tan_product_values (a b : Real) 
  (h : 7 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b - 1) = 0) : 
  (Real.tan (a/2) * Real.tan (b/2) = Real.sqrt ((-7 + Real.sqrt 133) / 3) ∨
   Real.tan (a/2) * Real.tan (b/2) = -Real.sqrt ((-7 + Real.sqrt 133) / 3) ∨
   Real.tan (a/2) * Real.tan (b/2) = Real.sqrt ((-7 - Real.sqrt 133) / 3) ∨
   Real.tan (a/2) * Real.tan (b/2) = -Real.sqrt ((-7 - Real.sqrt 133) / 3)) := by
  sorry

end tan_product_values_l1482_148277


namespace gcd_lcm_properties_l1482_148205

theorem gcd_lcm_properties (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 100) : 
  (a * b = 2000) ∧ 
  (Nat.lcm (10 * a) (10 * b) = 10 * Nat.lcm a b) ∧ 
  ((10 * a) * (10 * b) = 100 * (a * b)) := by
  sorry

end gcd_lcm_properties_l1482_148205


namespace symmetry_probability_l1482_148200

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- The size of the square grid -/
def gridSize : Nat := 11

/-- The center point of the grid -/
def centerPoint : GridPoint := ⟨gridSize / 2 + 1, gridSize / 2 + 1⟩

/-- The total number of points in the grid -/
def totalPoints : Nat := gridSize * gridSize

/-- The number of points excluding the center point -/
def remainingPoints : Nat := totalPoints - 1

/-- Checks if a point forms a line of symmetry with the center point -/
def isSymmetryPoint (p : GridPoint) : Bool :=
  p.x = centerPoint.x ∨ 
  p.y = centerPoint.y ∨ 
  p.x - centerPoint.x = p.y - centerPoint.y ∨
  p.x - centerPoint.x = centerPoint.y - p.y

/-- The number of points that form lines of symmetry -/
def symmetryPoints : Nat := 4 * (gridSize - 1)

/-- The probability theorem -/
theorem symmetry_probability : 
  (symmetryPoints : ℚ) / remainingPoints = 1 / 3 := by sorry

end symmetry_probability_l1482_148200


namespace rectangular_box_dimensions_l1482_148260

theorem rectangular_box_dimensions (X Y Z : ℝ) 
  (h1 : X * Y = 40)
  (h2 : X * Z = 72)
  (h3 : Y * Z = 90) :
  X + Y + Z = 18 := by
sorry

end rectangular_box_dimensions_l1482_148260


namespace sum_terms_increase_l1482_148298

def sum_terms (k : ℕ) : ℕ := 2^(k-1) + 1

theorem sum_terms_increase (k : ℕ) (h : k ≥ 2) : 
  sum_terms (k+1) - sum_terms k = 2^(k-1) := by
  sorry

end sum_terms_increase_l1482_148298


namespace ladder_problem_l1482_148273

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end ladder_problem_l1482_148273


namespace expected_total_score_l1482_148219

/-- The number of students participating in the contest -/
def num_students : ℕ := 10

/-- The number of shooting opportunities for each student -/
def shots_per_student : ℕ := 2

/-- The probability of scoring a goal -/
def goal_probability : ℝ := 0.6

/-- The scoring system -/
def score (goals : ℕ) : ℝ :=
  match goals with
  | 0 => 0
  | 1 => 5
  | _ => 10

/-- The expected score for a single student -/
def expected_score_per_student : ℝ :=
  (score 0) * (1 - goal_probability)^2 +
  (score 1) * 2 * goal_probability * (1 - goal_probability) +
  (score 2) * goal_probability^2

/-- Theorem: The expected total score for all students is 60 -/
theorem expected_total_score :
  num_students * expected_score_per_student = 60 := by
  sorry

end expected_total_score_l1482_148219


namespace maria_total_money_l1482_148214

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The number of dimes Maria has initially -/
def initial_dimes : ℕ := 4

/-- The number of quarters Maria has initially -/
def initial_quarters : ℕ := 4

/-- The number of nickels Maria has initially -/
def initial_nickels : ℕ := 7

/-- The number of quarters Maria's mom gives her -/
def additional_quarters : ℕ := 5

/-- The total amount of money Maria has after receiving the additional quarters -/
theorem maria_total_money :
  (initial_dimes * dime_value +
   initial_quarters * quarter_value +
   initial_nickels * nickel_value +
   additional_quarters * quarter_value) = 3 :=
by sorry

end maria_total_money_l1482_148214


namespace kfc_chicken_legs_l1482_148257

/-- Given the number of thighs, wings, and platters, calculate the number of legs baked. -/
def chicken_legs_baked (thighs wings platters : ℕ) : ℕ :=
  let thighs_per_platter := thighs / platters
  thighs_per_platter * platters

/-- Theorem stating that 144 chicken legs were baked given the problem conditions. -/
theorem kfc_chicken_legs :
  let thighs := 144
  let wings := 224
  let platters := 16
  chicken_legs_baked thighs wings platters = 144 := by
  sorry

#eval chicken_legs_baked 144 224 16

end kfc_chicken_legs_l1482_148257


namespace range_of_c_l1482_148220

-- Define the sets corresponding to propositions p and q
def p (c : ℝ) : Set ℝ := {x | 1 - c < x ∧ x < 1 + c ∧ c > 0}
def q : Set ℝ := {x | (x - 3)^2 < 16}

-- Define the property that p is a sufficient but not necessary condition for q
def sufficient_not_necessary (c : ℝ) : Prop :=
  p c ⊂ q ∧ p c ≠ q

-- State the theorem
theorem range_of_c :
  ∀ c : ℝ, sufficient_not_necessary c ↔ 0 < c ∧ c ≤ 6 := by sorry

end range_of_c_l1482_148220


namespace cone_base_radius_l1482_148269

/-- Given a cone formed from a circular sector with central angle 120° and radius 4,
    the radius of the base circle of the cone is 4/3. -/
theorem cone_base_radius (θ : Real) (R : Real) (r : Real) : 
  θ = 120 → R = 4 → 2 * π * r = (θ / 360) * 2 * π * R → r = 4/3 := by
  sorry

end cone_base_radius_l1482_148269


namespace problem_solution_l1482_148225

theorem problem_solution : 
  (Real.sqrt 27 + Real.sqrt 3 - 2 * Real.sqrt 12 = 0) ∧
  ((3 + 2 * Real.sqrt 2) * (3 - 2 * Real.sqrt 2) - Real.sqrt 54 / Real.sqrt 6 = -2) :=
by sorry

end problem_solution_l1482_148225


namespace sqrt_neg_four_squared_l1482_148213

theorem sqrt_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by sorry

end sqrt_neg_four_squared_l1482_148213


namespace chamber_boundary_area_l1482_148222

/-- The area of the boundary of a chamber formed by three intersecting pipes -/
theorem chamber_boundary_area (pipe_circumference : ℝ) (h1 : pipe_circumference = 4) :
  let pipe_diameter := pipe_circumference / Real.pi
  let cross_section_area := Real.pi * (pipe_diameter / 2) ^ 2
  let chamber_boundary_area := 2 * (1 / 4) * Real.pi * pipe_diameter ^ 2
  chamber_boundary_area = 8 / Real.pi := by
  sorry

end chamber_boundary_area_l1482_148222


namespace symmetric_circle_equation_l1482_148295

/-- The equation of a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to find the symmetric point of a point with respect to a line -/
def symmetricPoint (p : ℝ × ℝ) (l : Line) : ℝ × ℝ := sorry

/-- Function to find the symmetric circle -/
def symmetricCircle (c : Circle) (l : Line) : Circle := sorry

theorem symmetric_circle_equation (c : Circle) (l : Line) : 
  c.center = (1, 0) ∧ c.radius = Real.sqrt 2 ∧ 
  l = { a := 2, b := -1, c := 3 } →
  let c' := symmetricCircle c l
  c'.center = (-3, 2) ∧ c'.radius = Real.sqrt 2 := by sorry

end symmetric_circle_equation_l1482_148295


namespace digit_101_of_7_26_l1482_148237

def decimal_expansion (n d : ℕ) : ℕ → ℕ
  | 0 => (10 * n / d) % 10
  | k + 1 => decimal_expansion (10 * (n % d)) d k

theorem digit_101_of_7_26 : decimal_expansion 7 26 100 = 6 := by
  sorry

end digit_101_of_7_26_l1482_148237


namespace expression_value_l1482_148253

theorem expression_value : 
  (3^2016 + 3^2014 + 3^2012) / (3^2016 - 3^2014 + 3^2012) = 91/73 := by
  sorry

end expression_value_l1482_148253


namespace shortest_side_length_l1482_148250

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the side divided by the point of tangency -/
  side : ℝ
  /-- The length of the shorter segment of the divided side -/
  segment1 : ℝ
  /-- The length of the longer segment of the divided side -/
  segment2 : ℝ
  /-- The condition that the segments add up to the side length -/
  side_condition : side = segment1 + segment2

/-- The theorem stating the length of the shortest side -/
theorem shortest_side_length (t : InscribedCircleTriangle)
  (h1 : t.r = 3)
  (h2 : t.segment1 = 5)
  (h3 : t.segment2 = 9) :
  ∃ (shortest_side : ℝ), shortest_side = 12 ∧ 
  (∀ (other_side : ℝ), other_side ≥ shortest_side) :=
sorry

end shortest_side_length_l1482_148250


namespace last_three_digits_of_5_to_9000_l1482_148299

theorem last_three_digits_of_5_to_9000 (h : 5^300 ≡ 1 [ZMOD 125]) :
  5^9000 ≡ 1 [ZMOD 1000] := by
  sorry

end last_three_digits_of_5_to_9000_l1482_148299


namespace center_of_mass_distance_to_line_l1482_148281

/-- Two material points in a plane -/
structure MaterialPoint where
  position : ℝ × ℝ
  mass : ℝ

/-- A line in a plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Distance from a point to a line -/
def distanceToLine (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Center of mass of two material points -/
def centerOfMass (p1 p2 : MaterialPoint) : ℝ × ℝ := sorry

theorem center_of_mass_distance_to_line 
  (P Q : MaterialPoint) (MN : Line) 
  (a b : ℝ) 
  (h1 : distanceToLine P.position MN = a) 
  (h2 : distanceToLine Q.position MN = b) :
  let Z := centerOfMass P Q
  distanceToLine Z MN = (P.mass * a + Q.mass * b) / (P.mass + Q.mass) := by
  sorry

end center_of_mass_distance_to_line_l1482_148281


namespace girls_more_likely_separated_l1482_148240

/-- The probability of two girls being separated when randomly seated among three boys on a 5-seat bench is greater than the probability of them sitting together. -/
theorem girls_more_likely_separated (n : ℕ) (h : n = 5) :
  let total_arrangements := Nat.choose n 2
  let adjacent_arrangements := n - 1
  (total_arrangements - adjacent_arrangements : ℚ) / total_arrangements > adjacent_arrangements / total_arrangements :=
by
  sorry

end girls_more_likely_separated_l1482_148240


namespace pinky_bought_36_apples_l1482_148284

/-- The number of apples Danny the Duck bought -/
def danny_apples : ℕ := 73

/-- The total number of apples Pinky the Pig and Danny the Duck have -/
def total_apples : ℕ := 109

/-- The number of apples Pinky the Pig bought -/
def pinky_apples : ℕ := total_apples - danny_apples

theorem pinky_bought_36_apples : pinky_apples = 36 := by
  sorry

end pinky_bought_36_apples_l1482_148284


namespace equation_solution_l1482_148272

theorem equation_solution :
  ∃! x : ℚ, 6 * (3 * x - 1) + 7 = -3 * (2 - 5 * x) - 4 :=
by
  use -11/3
  constructor
  · -- Proof that -11/3 satisfies the equation
    sorry
  · -- Proof of uniqueness
    sorry

#check equation_solution

end equation_solution_l1482_148272


namespace vector_equality_transitive_l1482_148285

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equality_transitive (a b c : V) :
  a = b → b = c → a = c := by sorry

end vector_equality_transitive_l1482_148285


namespace exponent_calculation_l1482_148229

theorem exponent_calculation (a n m k : ℝ) 
  (h1 : a^n = 2) 
  (h2 : a^m = 3) 
  (h3 : a^k = 4) : 
  a^(2*n + m - 2*k) = 3/4 := by
sorry

end exponent_calculation_l1482_148229


namespace robin_gum_packages_l1482_148267

theorem robin_gum_packages (pieces_per_package : ℕ) (total_pieces : ℕ) (h1 : pieces_per_package = 18) (h2 : total_pieces = 486) :
  total_pieces / pieces_per_package = 27 := by
  sorry

end robin_gum_packages_l1482_148267


namespace second_smallest_divisible_sum_of_digits_l1482_148261

def isDivisibleByAllLessThan8 (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k ∧ k < 8 → n % k = 0

def isSecondSmallestDivisible (n : ℕ) : Prop :=
  isDivisibleByAllLessThan8 n ∧
  ∃ m : ℕ, m < n ∧ isDivisibleByAllLessThan8 m ∧
  ∀ k : ℕ, k < n ∧ isDivisibleByAllLessThan8 k → k ≤ m

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem second_smallest_divisible_sum_of_digits :
  ∃ N : ℕ, isSecondSmallestDivisible N ∧ sumOfDigits N = 12 :=
sorry

end second_smallest_divisible_sum_of_digits_l1482_148261


namespace measure_17kg_cranberries_l1482_148254

/-- Represents a two-pan scale -/
structure TwoPanScale :=
  (leftPan : ℝ)
  (rightPan : ℝ)

/-- Represents the state of the cranberry measurement process -/
structure CranberryMeasurement :=
  (totalAmount : ℝ)
  (weightAmount : ℝ)
  (scale : TwoPanScale)
  (weighingsUsed : ℕ)

/-- Definition of a valid weighing operation -/
def validWeighing (m : CranberryMeasurement) : Prop :=
  m.scale.leftPan = m.scale.rightPan ∧ m.weighingsUsed ≤ 2

/-- The main theorem to prove -/
theorem measure_17kg_cranberries :
  ∃ (m : CranberryMeasurement),
    m.totalAmount = 22 ∧
    m.weightAmount = 2 ∧
    validWeighing m ∧
    ∃ (amount : ℝ), amount = 17 ∧ amount ≤ m.totalAmount :=
sorry

end measure_17kg_cranberries_l1482_148254


namespace front_view_of_given_stack_map_l1482_148218

/-- Represents a stack map as a list of lists of natural numbers -/
def StackMap := List (List Nat)

/-- Calculates the front view of a stack map -/
def frontView (sm : StackMap) : List Nat :=
  let columns := sm.map List.length
  List.map (fun col => List.foldl Nat.max 0 (List.map (fun row => row.getD col 0) sm)) (List.range (columns.foldl Nat.max 0))

/-- The given stack map -/
def givenStackMap : StackMap := [[4, 1], [1, 2, 4], [3, 1]]

theorem front_view_of_given_stack_map :
  frontView givenStackMap = [4, 2, 4] := by sorry

end front_view_of_given_stack_map_l1482_148218


namespace factorial_equation_sum_of_digits_l1482_148265

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- The theorem statement -/
theorem factorial_equation_sum_of_digits :
  ∃ (n : ℕ), n > 0 ∧ 
  (factorial (n + 1) + 2 * factorial (n + 2) = factorial n * 871) ∧
  (sumOfDigits n = 10) := by
  sorry

end factorial_equation_sum_of_digits_l1482_148265


namespace greatest_divisor_four_consecutive_integers_l1482_148291

theorem greatest_divisor_four_consecutive_integers :
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (n : ℕ), n > 0 → (k ∣ n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (m : ℕ), m > k → ∃ (n : ℕ), n > 0 ∧ ¬(m ∣ n * (n + 1) * (n + 2) * (n + 3))) :=
by
  use 24
  sorry

end greatest_divisor_four_consecutive_integers_l1482_148291


namespace square_of_1003_l1482_148204

theorem square_of_1003 : (1003 : ℕ)^2 = 1006009 := by
  sorry

end square_of_1003_l1482_148204


namespace triangle_circle_intersection_l1482_148286

theorem triangle_circle_intersection (DE DF EY FY : ℕ) (EF : ℝ) : 
  DE = 65 →
  DF = 104 →
  EY + FY = EF →
  FY * EF = 39 * 169 →
  EF = 169 :=
by sorry

end triangle_circle_intersection_l1482_148286


namespace abs_cube_plus_cube_equals_two_cube_l1482_148275

theorem abs_cube_plus_cube_equals_two_cube (x : ℝ) : |x^3| + x^3 = 2*x^3 := by
  sorry

end abs_cube_plus_cube_equals_two_cube_l1482_148275


namespace card_sum_problem_l1482_148227

theorem card_sum_problem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
  sorry

end card_sum_problem_l1482_148227


namespace linear_function_properties_l1482_148226

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem linear_function_properties (k b : ℝ) (hk : k < 0) (hb : b > 0) :
  -- 1. The graph passes through the first, second, and fourth quadrants
  (∃ x y, x > 0 ∧ y > 0 ∧ y = linear_function k b x) ∧
  (∃ x y, x < 0 ∧ y > 0 ∧ y = linear_function k b x) ∧
  (∃ x y, x > 0 ∧ y < 0 ∧ y = linear_function k b x) ∧
  -- 2. y decreases as x increases
  (∀ x₁ x₂, x₁ < x₂ → linear_function k b x₁ > linear_function k b x₂) ∧
  -- 3. The graph intersects the y-axis at the point (0, b)
  (linear_function k b 0 = b) ∧
  -- 4. When x > -b/k, y < 0
  (∀ x, x > -b/k → linear_function k b x < 0) :=
by
  sorry

end linear_function_properties_l1482_148226


namespace arithmetic_sequence_nth_term_l1482_148230

theorem arithmetic_sequence_nth_term (x : ℝ) (n : ℕ) : 
  (2*x - 3 = (5*x - 11) - (3*x - 8)) → 
  (5*x - 11 = (3*x + 1) - (3*x - 8)) → 
  (1 + 4*n = 2009) → 
  n = 502 := by sorry

end arithmetic_sequence_nth_term_l1482_148230


namespace perfume_price_with_tax_l1482_148241

/-- Calculates the total price including tax given the original price and tax rate. -/
def totalPriceWithTax (originalPrice taxRate : ℝ) : ℝ :=
  originalPrice * (1 + taxRate)

/-- Theorem stating that for a product with an original price of $92 and a tax rate of 7.5%,
    the total price including tax is $98.90. -/
theorem perfume_price_with_tax :
  totalPriceWithTax 92 0.075 = 98.90 := by
  sorry

end perfume_price_with_tax_l1482_148241


namespace fraction_subtraction_and_division_l1482_148244

theorem fraction_subtraction_and_division :
  (5/6 - 1/3) / (2/9) = 9/4 := by
sorry

end fraction_subtraction_and_division_l1482_148244


namespace ellipse_focal_length_l1482_148221

-- Define the ellipse parameters
def a : ℝ := 3
def b_squared : ℝ := 8

-- Define the focal length
def focal_length : ℝ := 2

-- Theorem statement
theorem ellipse_focal_length :
  focal_length = 2 * Real.sqrt (a^2 - b_squared) :=
by sorry

end ellipse_focal_length_l1482_148221


namespace smallest_class_size_l1482_148266

theorem smallest_class_size (N : ℕ) (G : ℕ) : N = 7 ↔ 
  (N > 0 ∧ G > 0 ∧ (25 : ℚ) / 100 < (G : ℚ) / N ∧ (G : ℚ) / N < (30 : ℚ) / 100) ∧
  ∀ (M : ℕ) (H : ℕ), M < N → ¬(M > 0 ∧ H > 0 ∧ (25 : ℚ) / 100 < (H : ℚ) / M ∧ (H : ℚ) / M < (30 : ℚ) / 100) :=
sorry

end smallest_class_size_l1482_148266


namespace consecutive_negative_integers_product_sum_l1482_148297

theorem consecutive_negative_integers_product_sum (n : ℤ) : 
  n < 0 ∧ n > -50 ∧ n * (n + 1) = 2400 → n + (n + 1) = -97 := by
  sorry

end consecutive_negative_integers_product_sum_l1482_148297


namespace nunzio_pizza_consumption_l1482_148236

/-- Calculates the number of whole pizzas eaten given daily pieces, days, and pieces per pizza -/
def pizzas_eaten (daily_pieces : ℕ) (days : ℕ) (pieces_per_pizza : ℕ) : ℕ :=
  (daily_pieces * days) / pieces_per_pizza

theorem nunzio_pizza_consumption :
  pizzas_eaten 3 72 8 = 27 := by
  sorry

end nunzio_pizza_consumption_l1482_148236


namespace sqrt_sum_equals_2sqrt7_l1482_148203

theorem sqrt_sum_equals_2sqrt7 :
  Real.sqrt (10 - 2 * Real.sqrt 21) + Real.sqrt (10 + 2 * Real.sqrt 21) = 2 * Real.sqrt 7 := by
  sorry

end sqrt_sum_equals_2sqrt7_l1482_148203


namespace cannot_form_desired_rectangle_l1482_148249

-- Define the tile sizes
def tile_size_1 : ℕ := 3
def tile_size_2 : ℕ := 4

-- Define the initial rectangles
def rect1_width : ℕ := 2
def rect1_height : ℕ := 6
def rect2_width : ℕ := 7
def rect2_height : ℕ := 8

-- Define the desired rectangle
def desired_width : ℕ := 12
def desired_height : ℕ := 5

-- Theorem statement
theorem cannot_form_desired_rectangle :
  ∀ (removed_tile1 removed_tile2 : ℕ),
  (removed_tile1 = tile_size_1 ∨ removed_tile1 = tile_size_2) →
  (removed_tile2 = tile_size_1 ∨ removed_tile2 = tile_size_2) →
  (rect1_width * rect1_height + rect2_width * rect2_height - removed_tile1 - removed_tile2) >
  (desired_width * desired_height) :=
by sorry

end cannot_form_desired_rectangle_l1482_148249


namespace arithmetic_sequence_2019th_term_l1482_148292

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_2019th_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_5 : (a 1 + a 2 + a 3 + a 4 + a 5) = 15)
  (h_6th_term : a 6 = 6) :
  a 2019 = 2019 := by
sorry

end arithmetic_sequence_2019th_term_l1482_148292


namespace sin_cos_value_l1482_148270

theorem sin_cos_value (α : Real) (h : 1 / Real.sin α + 1 / Real.cos α = Real.sqrt 3) :
  Real.sin α * Real.cos α = -1/3 := by
  sorry

end sin_cos_value_l1482_148270


namespace fifth_term_is_thirteen_l1482_148245

/-- An arithmetic sequence with the first term 1 and common difference 3 -/
def arithmeticSeq : ℕ → ℤ
  | 0 => 1
  | n+1 => arithmeticSeq n + 3

/-- The theorem stating that the 5th term of the sequence is 13 -/
theorem fifth_term_is_thirteen : arithmeticSeq 4 = 13 := by
  sorry

#eval arithmeticSeq 4  -- This will evaluate to 13

end fifth_term_is_thirteen_l1482_148245


namespace travel_time_proof_l1482_148201

def speed1 : ℝ := 6
def speed2 : ℝ := 12
def speed3 : ℝ := 18
def total_distance : ℝ := 1.8 -- 1800 meters converted to kilometers

theorem travel_time_proof :
  let d := total_distance / 3
  let time1 := d / speed1
  let time2 := d / speed2
  let time3 := d / speed3
  let total_time := (time1 + time2 + time3) * 60
  total_time = 11 := by
sorry

end travel_time_proof_l1482_148201


namespace cubic_roots_sum_of_squares_reciprocal_l1482_148289

theorem cubic_roots_sum_of_squares_reciprocal :
  ∀ a b c : ℝ,
  (a^3 - 6*a^2 + 11*a - 6 = 0) →
  (b^3 - 6*b^2 + 11*b - 6 = 0) →
  (c^3 - 6*c^2 + 11*c - 6 = 0) →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by sorry

end cubic_roots_sum_of_squares_reciprocal_l1482_148289


namespace binary_1101_is_13_l1482_148232

def binary_to_decimal (b : List Bool) : ℕ :=
  List.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0 b

theorem binary_1101_is_13 :
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end binary_1101_is_13_l1482_148232


namespace geometric_sequence_first_term_l1482_148246

/-- A geometric sequence with positive common ratio -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_eq : a 1 * a 9 = 2 * a 52)
  (h_a2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 := by
  sorry

end geometric_sequence_first_term_l1482_148246


namespace cube_volume_and_diagonal_l1482_148259

/-- Given a cube with surface area 150 square centimeters, prove its volume and space diagonal. -/
theorem cube_volume_and_diagonal (s : ℝ) (h : 6 * s^2 = 150) : 
  s^3 = 125 ∧ s * Real.sqrt 3 = 5 * Real.sqrt 3 := by
  sorry

end cube_volume_and_diagonal_l1482_148259


namespace area_ratio_of_triangles_l1482_148243

theorem area_ratio_of_triangles : 
  let mnp_sides : Fin 3 → ℝ := ![7, 24, 25]
  let qrs_sides : Fin 3 → ℝ := ![9, 12, 15]
  let mnp_area := (mnp_sides 0 * mnp_sides 1) / 2
  let qrs_area := (qrs_sides 0 * qrs_sides 1) / 2
  mnp_area / qrs_area = 14 / 9 := by
sorry

end area_ratio_of_triangles_l1482_148243


namespace octal_subtraction_3456_1234_l1482_148252

/-- Represents a number in base 8 --/
def OctalNumber := List Nat

/-- Converts an octal number to its decimal representation --/
def octal_to_decimal (n : OctalNumber) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

/-- Subtracts two octal numbers --/
def octal_subtract (a b : OctalNumber) : OctalNumber :=
  sorry -- Actual implementation would go here

theorem octal_subtraction_3456_1234 :
  let a : OctalNumber := [6, 5, 4, 3]  -- 3456 in base 8
  let b : OctalNumber := [4, 3, 2, 1]  -- 1234 in base 8
  let result : OctalNumber := [2, 2, 2, 2]  -- 2222 in base 8
  octal_subtract a b = result := by
  sorry

end octal_subtraction_3456_1234_l1482_148252


namespace baylor_freelance_earnings_l1482_148293

theorem baylor_freelance_earnings (initial_amount : ℝ) : initial_amount = 4000 → 
  let first_payment := initial_amount / 2
  let second_payment := first_payment + (2/5 * first_payment)
  let third_payment := 2 * (first_payment + second_payment)
  initial_amount + first_payment + second_payment + third_payment = 18400 := by
sorry

end baylor_freelance_earnings_l1482_148293


namespace ciphertext_solution_l1482_148231

theorem ciphertext_solution :
  ∃! (x₁ x₂ x₃ x₄ : ℕ),
    x₁ ≤ 25 ∧ x₂ ≤ 25 ∧ x₃ ≤ 25 ∧ x₄ ≤ 25 ∧
    (x₁ + 2*x₂) % 26 = 9 ∧
    (3*x₂) % 26 = 16 ∧
    (x₃ + 2*x₄) % 26 = 23 ∧
    (3*x₄) % 26 = 12 ∧
    x₁ = 7 ∧ x₂ = 14 ∧ x₃ = 15 ∧ x₄ = 4 :=
by sorry

end ciphertext_solution_l1482_148231


namespace don_raise_is_880_l1482_148282

/-- Calculates Don's raise given the conditions of the problem -/
def calculate_don_raise (wife_raise : ℚ) (salary_difference : ℚ) : ℚ :=
  let wife_salary := wife_raise / 0.08
  let don_salary := (wife_salary + salary_difference + wife_raise) / 1.08
  0.08 * don_salary

/-- Theorem stating that Don's raise is 880 given the problem conditions -/
theorem don_raise_is_880 :
  calculate_don_raise 840 540 = 880 := by
  sorry

end don_raise_is_880_l1482_148282


namespace distance_to_origin_l1482_148255

open Complex

theorem distance_to_origin : let z : ℂ := (1 - I) * (1 + I) / I
  abs z = 2 := by sorry

end distance_to_origin_l1482_148255


namespace petya_lives_in_sixth_entrance_l1482_148279

/-- Represents the layout of the houses -/
structure HouseLayout where
  num_entrances : ℕ
  petya_entrance : ℕ
  vasya_entrance : ℕ

/-- Calculates the distance between two entrances -/
def distance (layout : HouseLayout) (entrance1 entrance2 : ℕ) : ℝ :=
  sorry

/-- Represents the shortest path around Petya's house -/
def shortest_path (layout : HouseLayout) (side : Bool) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem petya_lives_in_sixth_entrance (layout : HouseLayout) :
  layout.vasya_entrance = 4 →
  shortest_path layout true = shortest_path layout false →
  layout.petya_entrance = 6 :=
sorry

end petya_lives_in_sixth_entrance_l1482_148279


namespace multiple_properties_l1482_148278

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 5 * k) 
  (hb : ∃ m : ℤ, b = 10 * m) : 
  (∃ n : ℤ, b = 5 * n) ∧ 
  (∃ p : ℤ, a - b = 5 * p) ∧ 
  (∃ q : ℤ, a + b = 5 * q) :=
by sorry

end multiple_properties_l1482_148278


namespace coordinate_conditions_l1482_148211

theorem coordinate_conditions (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁ = 4 * π / 5 ∧ y₁ = -π / 5)
  (h₂ : x₂ = 12 * π / 5 ∧ y₂ = -3 * π / 5)
  (h₃ : x₃ = 4 * π / 3 ∧ y₃ = -π / 3) :
  (x₁ + 4 * y₁ = 0 ∧ x₁ + 3 * y₁ < π ∧ π - x₁ - 3 * y₁ ≠ 1 ∧ 3 * x₁ + 5 * y₁ > 0) ∧
  (x₂ + 4 * y₂ = 0 ∧ x₂ + 3 * y₂ < π ∧ π - x₂ - 3 * y₂ ≠ 1 ∧ 3 * x₂ + 5 * y₂ > 0) ∧
  (x₃ + 4 * y₃ = 0 ∧ x₃ + 3 * y₃ < π ∧ π - x₃ - 3 * y₃ ≠ 1 ∧ 3 * x₃ + 5 * y₃ > 0) :=
by
  sorry

end coordinate_conditions_l1482_148211


namespace matrix_equation_solution_l1482_148217

theorem matrix_equation_solution : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -4; 3, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-16, -6; 7, 2]
  let M : Matrix (Fin 2) (Fin 2) ℤ := !![5, -7; -2, 3]
  M * A = B := by sorry

end matrix_equation_solution_l1482_148217


namespace sum_2016_l1482_148271

/-- An arithmetic sequence with its first term and sum property -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  s : ℕ → ℤ  -- The sum sequence
  first_term : a 1 = -2016
  sum_property : s 20 / 20 - s 18 / 18 = 2
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n : ℕ, s n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

/-- The sum of the first 2016 terms is -2016 -/
theorem sum_2016 (seq : ArithmeticSequence) : seq.s 2016 = -2016 := by
  sorry

end sum_2016_l1482_148271


namespace hexagonal_pyramid_vertices_l1482_148202

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add necessary fields

/-- A pyramid with a regular polygon base -/
structure Pyramid (n : ℕ) where
  base : RegularPolygon n

/-- The number of vertices in a pyramid -/
def Pyramid.numVertices (p : Pyramid n) : ℕ := sorry

/-- Theorem: A pyramid with a base that is a regular polygon with six equal angles has 7 vertices -/
theorem hexagonal_pyramid_vertices :
  ∀ (p : Pyramid 6), p.numVertices = 7 := by sorry

end hexagonal_pyramid_vertices_l1482_148202


namespace power_addition_l1482_148262

theorem power_addition (x y : ℝ) (a b : ℝ) 
  (h1 : (4 : ℝ) ^ x = a) 
  (h2 : (4 : ℝ) ^ y = b) : 
  (4 : ℝ) ^ (x + y) = a * b := by
  sorry

end power_addition_l1482_148262


namespace rectangular_field_area_l1482_148228

theorem rectangular_field_area (m : ℝ) : ∃ m : ℝ, (3*m + 8)*(m - 3) = 100 ∧ m > 0 := by
  sorry

end rectangular_field_area_l1482_148228


namespace solve_equation_l1482_148263

theorem solve_equation (t x : ℝ) : 2*t + 2*x - t - 3*x + 4*x + 2*t = 30 → x = 4 := by
  sorry

end solve_equation_l1482_148263


namespace function_equation_solution_l1482_148268

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 1 → f x + f (1 / (1 - x)) = x) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f x = (1 / 2) * (x + 1 - 1 / x - 1 / (1 - x))) ∧
  f 1 = -f 0 := by
  sorry

end function_equation_solution_l1482_148268


namespace lee_earnings_l1482_148251

/-- Represents Lee's lawn care services and earnings --/
structure LawnCareServices where
  mowing_price : ℕ
  trimming_price : ℕ
  weed_removal_price : ℕ
  mowed_lawns : ℕ
  trimmed_lawns : ℕ
  weed_removed_lawns : ℕ
  mowing_tips : ℕ
  trimming_tips : ℕ
  weed_removal_tips : ℕ

/-- Calculates the total earnings from Lee's lawn care services --/
def total_earnings (s : LawnCareServices) : ℕ :=
  s.mowing_price * s.mowed_lawns +
  s.trimming_price * s.trimmed_lawns +
  s.weed_removal_price * s.weed_removed_lawns +
  s.mowing_tips +
  s.trimming_tips +
  s.weed_removal_tips

/-- Theorem stating that Lee's total earnings for the week were $747 --/
theorem lee_earnings : 
  let s : LawnCareServices := {
    mowing_price := 33,
    trimming_price := 15,
    weed_removal_price := 10,
    mowed_lawns := 16,
    trimmed_lawns := 8,
    weed_removed_lawns := 5,
    mowing_tips := 3 * 10,
    trimming_tips := 2 * 7,
    weed_removal_tips := 1 * 5
  }
  total_earnings s = 747 := by
  sorry


end lee_earnings_l1482_148251


namespace golf_strokes_over_par_l1482_148208

/-- Given a golfer who plays 9 rounds with an average of 4 strokes per hole,
    and a par value of 3 per hole, prove that the golfer will be 9 strokes over par. -/
theorem golf_strokes_over_par (rounds : ℕ) (avg_strokes_per_hole : ℕ) (par_value_per_hole : ℕ)
  (h1 : rounds = 9)
  (h2 : avg_strokes_per_hole = 4)
  (h3 : par_value_per_hole = 3) :
  rounds * avg_strokes_per_hole - rounds * par_value_per_hole = 9 :=
by sorry

end golf_strokes_over_par_l1482_148208


namespace problem_1_problem_2_l1482_148216

-- Problem 1
theorem problem_1 : -20 - (-8) + (-4) = -16 := by sorry

-- Problem 2
theorem problem_2 : -1^3 * (-2)^2 / (4/3) + |5-8| = 0 := by sorry

end problem_1_problem_2_l1482_148216


namespace total_canoes_by_april_l1482_148274

def canoes_in_month (n : ℕ) : ℕ :=
  2 * (3 ^ (n - 1))

theorem total_canoes_by_april : 
  canoes_in_month 1 + canoes_in_month 2 + canoes_in_month 3 + canoes_in_month 4 = 80 := by
  sorry

end total_canoes_by_april_l1482_148274


namespace rhombus_perimeter_l1482_148290

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  let side_length := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side_length = 8 * Real.sqrt 41 := by
  sorry

end rhombus_perimeter_l1482_148290


namespace smallest_multiple_of_6_and_15_l1482_148215

theorem smallest_multiple_of_6_and_15 : 
  ∃ (a : ℕ), a > 0 ∧ 6 ∣ a ∧ 15 ∣ a ∧ ∀ (b : ℕ), b > 0 ∧ 6 ∣ b ∧ 15 ∣ b → a ≤ b :=
by
  -- The proof goes here
  sorry

end smallest_multiple_of_6_and_15_l1482_148215


namespace f_negative_pi_third_l1482_148264

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (2^x + 1) + a * x + Real.cos (2 * x)

theorem f_negative_pi_third (a : ℝ) : 
  f a (π / 3) = 2 → f a (-π / 3) = -2 := by
  sorry

end f_negative_pi_third_l1482_148264


namespace min_type_A_costumes_l1482_148288

-- Define the cost of type B costumes
def cost_B : ℝ := 120

-- Define the cost of type A costumes
def cost_A : ℝ := cost_B + 30

-- Define the total number of costumes
def total_costumes : ℕ := 20

-- Define the minimum total cost
def min_total_cost : ℝ := 2800

-- Theorem statement
theorem min_type_A_costumes :
  ∀ m : ℕ,
  (m : ℝ) * cost_A + (total_costumes - m : ℝ) * cost_B ≥ min_total_cost →
  m ≥ 14 :=
by sorry

end min_type_A_costumes_l1482_148288


namespace y_percent_of_x_in_terms_of_z_l1482_148212

theorem y_percent_of_x_in_terms_of_z (x y z : ℝ) 
  (h1 : 0.7 * (x - y) = 0.3 * (x + y))
  (h2 : 0.6 * (x + z) = 0.4 * (y - z)) :
  y = 0.4 * x := by
  sorry

end y_percent_of_x_in_terms_of_z_l1482_148212


namespace max_profit_theorem_l1482_148223

/-- Represents the sales data for a week -/
structure WeekData where
  modelA : ℕ
  modelB : ℕ
  revenue : ℕ

/-- Represents the appliance models -/
inductive Model
| A
| B

def purchase_price (m : Model) : ℕ :=
  match m with
  | Model.A => 180
  | Model.B => 160

def selling_price (m : Model) : ℕ :=
  match m with
  | Model.A => 240
  | Model.B => 200

def profit (m : Model) : ℕ :=
  selling_price m - purchase_price m

def total_units : ℕ := 35

def max_budget : ℕ := 6000

def profit_goal : ℕ := 1750

def week1_data : WeekData := ⟨3, 2, 1120⟩

def week2_data : WeekData := ⟨4, 3, 1560⟩

/-- Calculates the total profit for a given number of units of each model -/
def total_profit (units_A units_B : ℕ) : ℕ :=
  units_A * profit Model.A + units_B * profit Model.B

/-- Calculates the total purchase cost for a given number of units of each model -/
def total_cost (units_A units_B : ℕ) : ℕ :=
  units_A * purchase_price Model.A + units_B * purchase_price Model.B

theorem max_profit_theorem :
  ∃ (units_A units_B : ℕ),
    units_A + units_B = total_units ∧
    total_cost units_A units_B ≤ max_budget ∧
    total_profit units_A units_B > profit_goal ∧
    total_profit units_A units_B = 1800 ∧
    ∀ (x y : ℕ), x + y = total_units → total_cost x y ≤ max_budget →
      total_profit x y ≤ total_profit units_A units_B :=
by sorry

end max_profit_theorem_l1482_148223


namespace range_of_f_l1482_148239

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 else 2^x

theorem range_of_f :
  Set.range f = Set.Ici 1 := by sorry

end range_of_f_l1482_148239
