import Mathlib

namespace inscribed_circle_radius_quarter_sector_l971_97175

/-- The radius of an inscribed circle in a quarter circle sector -/
theorem inscribed_circle_radius_quarter_sector (r : ℝ) (h : r = 5) :
  let inscribed_radius := r * (Real.sqrt 2 - 1)
  inscribed_radius = 5 * Real.sqrt 2 - 5 := by
sorry

end inscribed_circle_radius_quarter_sector_l971_97175


namespace ratio_to_percent_l971_97108

theorem ratio_to_percent (a b : ℕ) (h : a = 2 ∧ b = 3) : (a : ℚ) / (a + b : ℚ) * 100 = 40 := by
  sorry

end ratio_to_percent_l971_97108


namespace units_digit_of_7_pow_2023_l971_97150

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of raising 7 to the power of 2023 -/
def power : ℕ := 7^2023

/-- Theorem: The units digit of 7^2023 is 3 -/
theorem units_digit_of_7_pow_2023 : unitsDigit power = 3 := by sorry

end units_digit_of_7_pow_2023_l971_97150


namespace vector_opposite_directions_x_value_l971_97168

/-- Two vectors are in opposite directions if their dot product is negative -/
def opposite_directions (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 < 0

theorem vector_opposite_directions_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (4, x)
  opposite_directions a b → x = -2 := by
sorry

end vector_opposite_directions_x_value_l971_97168


namespace negation_of_forall_square_ge_self_l971_97146

theorem negation_of_forall_square_ge_self :
  (¬ ∀ x : ℕ, x^2 ≥ x) ↔ (∃ x : ℕ, x^2 < x) := by sorry

end negation_of_forall_square_ge_self_l971_97146


namespace geometric_sequence_roots_l971_97126

theorem geometric_sequence_roots (m n : ℝ) : 
  (∃ a b c d : ℝ, 
    (a^2 - m*a + 2) * (a^2 - n*a + 2) = 0 ∧
    (b^2 - m*b + 2) * (b^2 - n*b + 2) = 0 ∧
    (c^2 - m*c + 2) * (c^2 - n*c + 2) = 0 ∧
    (d^2 - m*d + 2) * (d^2 - n*d + 2) = 0 ∧
    a = (1/2) ∧
    (∃ r : ℝ, b = a*r ∧ c = b*r ∧ d = c*r)) →
  |m - n| = (3/2) :=
by sorry

end geometric_sequence_roots_l971_97126


namespace perpendicular_sufficient_condition_perpendicular_not_necessary_condition_l971_97152

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- Define the given conditions
variable (m n : Line)
variable (α β : Plane)
variable (h_diff_lines : m ≠ n)
variable (h_diff_planes : α ≠ β)

-- State the theorem
theorem perpendicular_sufficient_condition 
  (h_parallel : parallel α β) 
  (h_perp : perpendicular n β) : 
  perpendicular n α :=
sorry

-- State that the condition is not necessary
theorem perpendicular_not_necessary_condition :
  ¬(∀ (n : Line) (α β : Plane), 
    perpendicular n α → 
    (parallel α β ∧ perpendicular n β)) :=
sorry

end perpendicular_sufficient_condition_perpendicular_not_necessary_condition_l971_97152


namespace volume_of_inscribed_sphere_l971_97187

/-- The volume of a sphere inscribed in a cube with edge length 8 inches -/
theorem volume_of_inscribed_sphere (π : ℝ) :
  let cube_edge : ℝ := 8
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  sphere_volume = (256 / 3) * π :=
by sorry

end volume_of_inscribed_sphere_l971_97187


namespace factorial_equation_solution_l971_97123

theorem factorial_equation_solution : ∃ k : ℕ, (4 * 3 * 2 * 1) * (2 * 1) = 2 * k * (3 * 2 * 1) ∧ k = 4 := by
  sorry

end factorial_equation_solution_l971_97123


namespace cube_root_two_not_expressible_l971_97176

theorem cube_root_two_not_expressible : ¬ ∃ (p q r : ℚ), (2 : ℝ)^(1/3) = p + q * (r^(1/2)) := by
  sorry

end cube_root_two_not_expressible_l971_97176


namespace contrapositive_equivalence_l971_97113

theorem contrapositive_equivalence : 
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ↔ 
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 1) := by
sorry

end contrapositive_equivalence_l971_97113


namespace point_coordinates_l971_97153

/-- A point in the two-dimensional plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the two-dimensional plane -/
def fourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance between a point and the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance between a point and the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: Coordinates of a point in the fourth quadrant with given distances to axes -/
theorem point_coordinates (p : Point) 
  (h1 : fourthQuadrant p) 
  (h2 : distanceToXAxis p = 2) 
  (h3 : distanceToYAxis p = 3) : 
  p = Point.mk 3 (-2) := by
  sorry

end point_coordinates_l971_97153


namespace cost_of_cakes_l971_97115

/-- The cost of cakes problem -/
theorem cost_of_cakes (num_cakes : ℕ) (johns_share : ℚ) (cost_per_cake : ℚ) 
  (h1 : num_cakes = 3)
  (h2 : johns_share = 18)
  (h3 : johns_share * 2 = num_cakes * cost_per_cake) :
  cost_per_cake = 12 := by
  sorry

end cost_of_cakes_l971_97115


namespace workshop_attendance_prove_workshop_attendance_l971_97137

theorem workshop_attendance : ℕ → Prop :=
  fun total_scientists =>
    ∃ (wolf_laureates nobel_laureates wolf_and_nobel non_wolf_non_nobel : ℕ),
      wolf_laureates = 31 ∧
      wolf_and_nobel = 14 ∧
      nobel_laureates = 25 ∧
      nobel_laureates - wolf_and_nobel = non_wolf_non_nobel + 3 ∧
      total_scientists = wolf_laureates + (nobel_laureates - wolf_and_nobel) + non_wolf_non_nobel ∧
      total_scientists = 50

theorem prove_workshop_attendance : workshop_attendance 50 := by
  sorry

end workshop_attendance_prove_workshop_attendance_l971_97137


namespace quadratic_solution_property_l971_97195

theorem quadratic_solution_property (p q : ℝ) : 
  (3 * p^2 + 7 * p - 6 = 0) → 
  (3 * q^2 + 7 * q - 6 = 0) → 
  (p - 2) * (q - 2) = 6 := by
sorry

end quadratic_solution_property_l971_97195


namespace polynomial_division_quotient_l971_97183

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 3 * X^4 - 5 * X^3 + 6 * X^2 - 8 * X + 3
  let divisor : Polynomial ℚ := X^2 + X + 1
  let quotient : Polynomial ℚ := 3 * X^2 - 8 * X
  (dividend / divisor) = quotient := by sorry

end polynomial_division_quotient_l971_97183


namespace inverse_proportion_quadrants_l971_97107

/-- An inverse proportion function passing through a specific point -/
structure InverseProportionFunction where
  k : ℝ
  a : ℝ
  point_condition : k / (3 * a) = a

/-- The quadrants where the graph of an inverse proportion function lies -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- The set of quadrants where the graph lies -/
def graph_quadrants (f : InverseProportionFunction) : Set Quadrant :=
  {Quadrant.I, Quadrant.III}

/-- Theorem: The graph of the inverse proportion function lies in Quadrants I and III -/
theorem inverse_proportion_quadrants (f : InverseProportionFunction) :
  graph_quadrants f = {Quadrant.I, Quadrant.III} := by
  sorry

end inverse_proportion_quadrants_l971_97107


namespace train_platform_passing_time_l971_97157

/-- Calculates the time for a train to pass a platform -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (time_to_pass_point : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_to_pass_point = 120)
  (h3 : platform_length = 1000) :
  (train_length + platform_length) / (train_length / time_to_pass_point) = 220 := by
  sorry

#check train_platform_passing_time

end train_platform_passing_time_l971_97157


namespace min_value_sum_reciprocals_l971_97133

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry

end min_value_sum_reciprocals_l971_97133


namespace slope_value_l971_97197

/-- The slope of a line passing through a focus of the ellipse x^2 + 2y^2 = 3 
    and intersecting it at two points with distance 2 apart. -/
def slope_through_focus (k : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ),
    -- A and B are on the ellipse
    A.1^2 + 2*A.2^2 = 3 ∧ B.1^2 + 2*B.2^2 = 3 ∧
    -- The line passes through a focus
    ∃ (x : ℝ), x^2 = 3/2 ∧ (A.2 - 0) = k * (A.1 - x) ∧ (B.2 - 0) = k * (B.1 - x) ∧
    -- The distance between A and B is 2
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4

/-- The theorem stating the absolute value of the slope -/
theorem slope_value : ∀ k : ℝ, slope_through_focus k → k^2 = 2 + Real.sqrt 3 := by
  sorry

end slope_value_l971_97197


namespace a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l971_97105

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  ∀ a : ℝ,
  (∀ a : ℝ, a = 1 → |a| = 1) ∧
  (∃ a : ℝ, |a| = 1 ∧ a ≠ 1) :=
by sorry

end a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l971_97105


namespace rectangular_box_volume_l971_97140

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 24)
  (area2 : w * h = 16)
  (area3 : l * h = 6) :
  l * w * h = 48 := by
sorry

end rectangular_box_volume_l971_97140


namespace m_eq_two_iff_parallel_l971_97149

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The lines l1 and l2 with parameter m -/
def l1 (m : ℝ) : Line := ⟨2, -m, -1⟩
def l2 (m : ℝ) : Line := ⟨m-1, -1, 1⟩

/-- The theorem stating that m=2 is a necessary and sufficient condition for l1 ∥ l2 -/
theorem m_eq_two_iff_parallel :
  ∀ m : ℝ, parallel (l1 m) (l2 m) ↔ m = 2 :=
sorry

end m_eq_two_iff_parallel_l971_97149


namespace equation_solutions_l971_97199

theorem equation_solutions : 
  ∀ (m n : ℕ), 3 * 2^m + 1 = n^2 ↔ (m = 0 ∧ n = 2) ∨ (m = 3 ∧ n = 5) ∨ (m = 4 ∧ n = 7) :=
by sorry

end equation_solutions_l971_97199


namespace perpendicular_lines_l971_97151

-- Define the slopes of the lines
def m1 : ℚ := 3 / 4
def m2 : ℚ := -3 / 4
def m3 : ℚ := -3 / 4
def m4 : ℚ := -4 / 3
def m5 : ℚ := 12 / 5

-- Define the condition for perpendicularity
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines :
  (are_perpendicular m1 m4) ∧
  (¬ are_perpendicular m1 m2) ∧
  (¬ are_perpendicular m1 m3) ∧
  (¬ are_perpendicular m1 m5) ∧
  (¬ are_perpendicular m2 m3) ∧
  (¬ are_perpendicular m2 m4) ∧
  (¬ are_perpendicular m2 m5) ∧
  (¬ are_perpendicular m3 m4) ∧
  (¬ are_perpendicular m3 m5) ∧
  (¬ are_perpendicular m4 m5) :=
by sorry

end perpendicular_lines_l971_97151


namespace strawberry_price_difference_l971_97109

/-- Proves that the difference in price per pint between the regular price and the sale price is $2 --/
theorem strawberry_price_difference
  (pints_sold : ℕ)
  (sale_revenue : ℚ)
  (revenue_difference : ℚ)
  (h1 : pints_sold = 54)
  (h2 : sale_revenue = 216)
  (h3 : revenue_difference = 108)
  : (sale_revenue + revenue_difference) / pints_sold - sale_revenue / pints_sold = 2 := by
  sorry

#check strawberry_price_difference

end strawberry_price_difference_l971_97109


namespace great_circle_bisects_angle_l971_97177

/-- A point on a sphere -/
structure SpherePoint where
  -- Add necessary fields

/-- A great circle on a sphere -/
structure GreatCircle where
  -- Add necessary fields

/-- The North Pole -/
def NorthPole : SpherePoint :=
  sorry

/-- The equator -/
def Equator : GreatCircle :=
  sorry

/-- Check if a point is on a great circle -/
def isOnGreatCircle (p : SpherePoint) (gc : GreatCircle) : Prop :=
  sorry

/-- Check if two points are equidistant from a third point -/
def areEquidistant (p1 p2 p3 : SpherePoint) : Prop :=
  sorry

/-- Check if a point is on the equator -/
def isOnEquator (p : SpherePoint) : Prop :=
  sorry

/-- The great circle through two points -/
def greatCircleThrough (p1 p2 : SpherePoint) : GreatCircle :=
  sorry

/-- Check if a great circle bisects an angle in a spherical triangle -/
def bisectsAngle (gc : GreatCircle) (p1 p2 p3 : SpherePoint) : Prop :=
  sorry

/-- Main theorem -/
theorem great_circle_bisects_angle (A B C : SpherePoint) :
  isOnGreatCircle A (greatCircleThrough NorthPole B) →
  isOnGreatCircle B (greatCircleThrough NorthPole A) →
  areEquidistant A B NorthPole →
  isOnEquator C →
  bisectsAngle (greatCircleThrough C NorthPole) A C B :=
by
  sorry

end great_circle_bisects_angle_l971_97177


namespace inverse_composition_equals_target_l971_97167

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 4

-- Define the inverse function f⁻¹
noncomputable def f_inv (x : ℝ) : ℝ := (x + 4) / 3

-- Theorem statement
theorem inverse_composition_equals_target : f_inv (f_inv 13) = 29 / 9 := by
  sorry

end inverse_composition_equals_target_l971_97167


namespace ac_lt_zero_sufficient_not_necessary_l971_97120

theorem ac_lt_zero_sufficient_not_necessary (a b c : ℝ) (h : c < b ∧ b < a) :
  (∀ x y z : ℝ, x < y ∧ y < z ∧ x*z < 0 → z*y > z*x) ∧
  (∃ x y z : ℝ, x < y ∧ y < z ∧ z*y > z*x ∧ x*z ≥ 0) :=
by sorry

end ac_lt_zero_sufficient_not_necessary_l971_97120


namespace order_combination_savings_l971_97136

/-- Calculates the discount percentage based on the number of photocopies -/
def discount_percentage (n : ℕ) : ℚ :=
  if n ≤ 50 then 0
  else if n ≤ 100 then 1/10
  else if n ≤ 200 then 1/4
  else 7/20

/-- Calculates the discounted cost for a given number of photocopies -/
def discounted_cost (n : ℕ) : ℚ :=
  let base_cost : ℚ := (n : ℚ) * 2/100
  base_cost * (1 - discount_percentage n)

/-- Theorem: The savings from combining orders is $0.225 -/
theorem order_combination_savings :
  discounted_cost 75 + discounted_cost 105 - discounted_cost 180 = 9/40 := by
  sorry

end order_combination_savings_l971_97136


namespace solution_difference_l971_97111

theorem solution_difference (r s : ℝ) : 
  (r - 5) * (r + 5) = 25 * r - 125 →
  (s - 5) * (s + 5) = 25 * s - 125 →
  r ≠ s →
  r > s →
  r - s = 15 := by
sorry

end solution_difference_l971_97111


namespace ratio_equality_solutions_l971_97172

theorem ratio_equality_solutions :
  {x : ℝ | (x + 3) / (3 * x + 3) = (3 * x + 4) / (6 * x + 4)} = {0, 1/3} := by
  sorry

end ratio_equality_solutions_l971_97172


namespace sum_of_coefficients_is_neg_105_l971_97154

/-- A quadratic function with given properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  has_minimum_20 : ∃ x, a * x^2 + b * x + c = 20 ∧ ∀ y, a * y^2 + b * y + c ≥ 20
  root_at_3 : a * 3^2 + b * 3 + c = 0
  root_at_7 : a * 7^2 + b * 7 + c = 0

/-- The sum of coefficients of a quadratic function with given properties is -105 -/
theorem sum_of_coefficients_is_neg_105 (f : QuadraticFunction) : 
  f.a + f.b + f.c = -105 := by
  sorry

end sum_of_coefficients_is_neg_105_l971_97154


namespace add_1876_minutes_to_6am_l971_97117

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem add_1876_minutes_to_6am (start : Time) 
  (h_start : start.hours = 6 ∧ start.minutes = 0) :
  addMinutes start 1876 = Time.mk 13 16 sorry sorry :=
sorry

end add_1876_minutes_to_6am_l971_97117


namespace consecutive_integers_fourth_power_sum_l971_97138

theorem consecutive_integers_fourth_power_sum (a b c : ℤ) : 
  (b = a + 1) →
  (c = b + 1) →
  (a^2 + b^2 + c^2 = 12246) →
  (a^4 + b^4 + c^4 = 50380802) :=
by sorry

end consecutive_integers_fourth_power_sum_l971_97138


namespace no_unbounded_phine_sequence_l971_97196

/-- A phine sequence is a sequence of positive real numbers satisfying
    a_{n+2} = (a_{n+1} + a_{n-1}) / a_n for all n ≥ 2 -/
def IsPhine (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ n ≥ 2, a (n + 2) = (a (n + 1) + a (n - 1)) / a n)

/-- There does not exist an unbounded phine sequence -/
theorem no_unbounded_phine_sequence :
  ¬ ∃ a : ℕ → ℝ, IsPhine a ∧ ∀ r : ℝ, ∃ n : ℕ, a n > r :=
sorry

end no_unbounded_phine_sequence_l971_97196


namespace parakeets_per_cage_l971_97134

theorem parakeets_per_cage (num_cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ) :
  num_cages = 6 →
  parrots_per_cage = 6 →
  total_birds = 48 →
  (total_birds - num_cages * parrots_per_cage) / num_cages = 2 := by
  sorry

end parakeets_per_cage_l971_97134


namespace elevatorProblem_l971_97128

-- Define the type of elevator move
inductive Move
| Up7 : Move
| Up10 : Move
| Down7 : Move
| Down10 : Move

-- Function to apply a move to a floor number
def applyMove (floor : ℕ) (move : Move) : ℕ :=
  match move with
  | Move.Up7 => floor + 7
  | Move.Up10 => floor + 10
  | Move.Down7 => if floor ≥ 7 then floor - 7 else floor
  | Move.Down10 => if floor ≥ 10 then floor - 10 else floor

-- Function to check if a floor is visited in a sequence of moves
def isVisited (startFloor : ℕ) (moves : List Move) (targetFloor : ℕ) : Prop :=
  targetFloor ∈ List.scanl applyMove startFloor moves

-- Theorem stating the existence of a valid sequence of moves
theorem elevatorProblem : 
  ∃ (moves : List Move), 
    moves.length ≤ 10 ∧ 
    isVisited 1 moves 13 ∧ 
    isVisited 1 moves 16 ∧ 
    isVisited 1 moves 24 :=
by
  sorry


end elevatorProblem_l971_97128


namespace fruit_vendor_lemons_sold_l971_97127

/-- Proves that a fruit vendor who sold 5 dozens of avocados and a total of 90 fruits sold 2.5 dozens of lemons -/
theorem fruit_vendor_lemons_sold (total_fruits : ℕ) (avocado_dozens : ℕ) (lemon_dozens : ℚ) : 
  total_fruits = 90 → avocado_dozens = 5 → lemon_dozens = 2.5 → 
  total_fruits = 12 * avocado_dozens + 12 * lemon_dozens := by
  sorry

#check fruit_vendor_lemons_sold

end fruit_vendor_lemons_sold_l971_97127


namespace square_perimeter_contradiction_l971_97112

theorem square_perimeter_contradiction (perimeter : ℝ) (side_length : ℝ) : 
  perimeter = 4 → side_length = 2 → perimeter ≠ 4 * side_length :=
by
  sorry

#check square_perimeter_contradiction

end square_perimeter_contradiction_l971_97112


namespace vector_subtraction_l971_97125

/-- Given two vectors a and b in ℝ², prove that a - 2*b equals (7, 3) -/
theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (3, 5)) (h2 : b = (-2, 1)) :
  a - 2 • b = (7, 3) := by
  sorry

end vector_subtraction_l971_97125


namespace point_in_fourth_quadrant_l971_97116

theorem point_in_fourth_quadrant :
  let P : ℝ × ℝ := (Real.tan (549 * π / 180), Real.cos (549 * π / 180))
  (P.1 > 0) ∧ (P.2 < 0) :=
by sorry

end point_in_fourth_quadrant_l971_97116


namespace rod_cutting_l971_97130

theorem rod_cutting (rod_length : Real) (num_pieces : Nat) (piece_length_cm : Real) : 
  rod_length = 29.75 ∧ num_pieces = 35 → piece_length_cm = 85 := by
  sorry

end rod_cutting_l971_97130


namespace hyperbola_eccentricity_l971_97188

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let f (x y : ℝ) := x^2 / a^2 - y^2 / b^2 - 1
  ∃ (x y : ℝ), f x y = 0 ∧ x > 0 ∧ x * c = 0 ∧ 2 * c = y * a / b ∧ c^2 = a^2 + b^2
  → c / a = 1 + Real.sqrt 2 := by
  sorry

end hyperbola_eccentricity_l971_97188


namespace min_value_of_f_l971_97158

def f (x : ℝ) := x^2 + 14*x + 3

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -7 :=
sorry

end min_value_of_f_l971_97158


namespace water_formed_moles_l971_97171

/-- Represents a chemical compound -/
structure Compound where
  name : String
  moles : ℚ

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Compound
  products : List Compound

def naoh : Compound := ⟨"NaOH", 2⟩
def h2so4 : Compound := ⟨"H2SO4", 2⟩

def balanced_reaction : Reaction := {
  reactants := [⟨"NaOH", 2⟩, ⟨"H2SO4", 1⟩],
  products := [⟨"Na2SO4", 1⟩, ⟨"H2O", 2⟩]
}

/-- Calculates the moles of a product formed in a reaction -/
def moles_formed (reaction : Reaction) (product : Compound) (limiting_reactant : Compound) : ℚ :=
  sorry

theorem water_formed_moles :
  moles_formed balanced_reaction ⟨"H2O", 0⟩ naoh = 2 := by
  sorry

end water_formed_moles_l971_97171


namespace trapezium_side_length_l971_97114

/-- Proves that given a trapezium with specified dimensions, the length of the unknown parallel side is 28 cm. -/
theorem trapezium_side_length 
  (known_side : ℝ)
  (height : ℝ)
  (area : ℝ)
  (h1 : known_side = 20)
  (h2 : height = 21)
  (h3 : area = 504)
  (h4 : area = (1/2) * (known_side + unknown_side) * height) :
  unknown_side = 28 :=
by
  sorry

end trapezium_side_length_l971_97114


namespace card_probability_ratio_l971_97100

def num_cards : ℕ := 40
def num_numbers : ℕ := 10
def cards_per_number : ℕ := 4
def cards_drawn : ℕ := 4

def p : ℚ := num_numbers / (num_cards.choose cards_drawn)
def q : ℚ := (num_numbers * (num_numbers - 1) * (cards_per_number.choose 3) * (cards_per_number.choose 1)) / (num_cards.choose cards_drawn)

theorem card_probability_ratio : q / p = 144 := by
  sorry

end card_probability_ratio_l971_97100


namespace largest_root_equation_l971_97164

theorem largest_root_equation (a b c d : ℝ) 
  (h1 : a + d = 2022)
  (h2 : b + c = 2022)
  (h3 : a ≠ c) :
  ∃ x : ℝ, (x - a) * (x - b) = (x - c) * (x - d) ∧ 
    x = 1011 ∧
    ∀ y : ℝ, (y - a) * (y - b) = (y - c) * (y - d) → y ≤ 1011 :=
by sorry

end largest_root_equation_l971_97164


namespace inscribed_triangle_area_l971_97147

/-- The area of a right-angled triangle inscribed in a circle of radius 100, 
    with acute angles α and β satisfying tan α = 4 tan β, is equal to 8000. -/
theorem inscribed_triangle_area (α β : Real) (h1 : α > 0) (h2 : β > 0) (h3 : α + β = Real.pi / 2) 
  (h4 : Real.tan α = 4 * Real.tan β) : 
  let r : Real := 100
  let area := r^2 * Real.sin α * Real.sin β
  area = 8000 := by
sorry

end inscribed_triangle_area_l971_97147


namespace x_value_l971_97142

theorem x_value : ∀ (x y z w : ℤ), 
  x = y + 5 →
  y = z + 10 →
  z = w + 20 →
  w = 80 →
  x = 115 := by
sorry

end x_value_l971_97142


namespace six_digit_concatenation_divisibility_l971_97141

theorem six_digit_concatenation_divisibility :
  ∀ a b : ℕ,
    100000 ≤ a ∧ a < 1000000 →
    100000 ≤ b ∧ b < 1000000 →
    (∃ k : ℕ, 1000000 * a + b = k * a * b) →
    (a = 166667 ∧ b = 333334) := by
  sorry

end six_digit_concatenation_divisibility_l971_97141


namespace square_roots_problem_l971_97106

theorem square_roots_problem (n : ℝ) (a : ℝ) : 
  n > 0 ∧ (a - 7)^2 = n ∧ (2*a + 1)^2 = n → n = 25 := by
  sorry

end square_roots_problem_l971_97106


namespace quadratic_decreasing_iff_a_in_range_l971_97131

/-- A quadratic function f(x) = ax^2 + 2(a-3)x + 1 is decreasing on [-2, +∞) if and only if a ∈ [-3, 0] -/
theorem quadratic_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, -2 ≤ x ∧ x < y → (a*x^2 + 2*(a-3)*x + 1) > (a*y^2 + 2*(a-3)*y + 1)) ↔ 
  -3 ≤ a ∧ a ≤ 0 := by
  sorry

end quadratic_decreasing_iff_a_in_range_l971_97131


namespace total_palm_trees_l971_97104

theorem total_palm_trees (forest_trees : ℕ) (desert_reduction : ℚ) (river_trees : ℕ)
  (h1 : forest_trees = 5000)
  (h2 : desert_reduction = 3 / 5)
  (h3 : river_trees = 1200) :
  forest_trees + (forest_trees - desert_reduction * forest_trees) + river_trees = 8200 :=
by sorry

end total_palm_trees_l971_97104


namespace subtraction_of_reciprocals_l971_97129

theorem subtraction_of_reciprocals (p q : ℝ) : 
  3 / p = 6 → 3 / q = 15 → p - q = 3 / 10 := by
  sorry

end subtraction_of_reciprocals_l971_97129


namespace cubic_polynomial_satisfies_conditions_l971_97102

def q (x : ℝ) : ℝ := -x^3 + 4*x^2 - 7*x - 4

theorem cubic_polynomial_satisfies_conditions :
  q 1 = -8 ∧ q 2 = -10 ∧ q 3 = -16 ∧ q 4 = -32 := by
  sorry

end cubic_polynomial_satisfies_conditions_l971_97102


namespace apps_added_l971_97185

theorem apps_added (initial_apps final_apps : ℕ) (h1 : initial_apps = 17) (h2 : final_apps = 18) :
  final_apps - initial_apps = 1 := by
  sorry

end apps_added_l971_97185


namespace inequality_proof_l971_97156

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 3) :
  Real.sqrt (1 + a) + Real.sqrt (1 + b) ≤ Real.sqrt 10 := by
  sorry

end inequality_proof_l971_97156


namespace arithmetic_progression_duality_l971_97173

theorem arithmetic_progression_duality 
  (x y z k p n : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hk : k > 0) (hp : p > 0) (hn : n > 0)
  (h_arith : ∃ (a d : ℝ), x = a + d * (k - 1) ∧ 
                          y = a + d * (p - 1) ∧ 
                          z = a + d * (n - 1)) :
  ∃ (a' d' : ℝ), 
    (k = a' + d' * (x - 1) ∧
     p = a' + d' * (y - 1) ∧
     n = a' + d' * (z - 1)) ∧
    (∃ (d : ℝ), d * d' = 1) := by
  sorry

end arithmetic_progression_duality_l971_97173


namespace sum_of_absolute_coefficients_l971_97132

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| = 81 :=
by sorry

end sum_of_absolute_coefficients_l971_97132


namespace exists_non_squareable_number_l971_97184

/-- A complication is adding a single digit to a number. -/
def Complication := Nat → Nat

/-- Apply a sequence of complications to a number. -/
def applyComplications (n : Nat) (complications : List Complication) : Nat :=
  complications.foldl (fun acc c => c acc) n

/-- Check if a number is a perfect square. -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem exists_non_squareable_number : 
  ∃ n : Nat, ∀ complications : List Complication, 
    complications.length ≤ 100 → 
    ¬(isPerfectSquare (applyComplications n complications)) := by
  sorry

end exists_non_squareable_number_l971_97184


namespace total_cost_is_1100_l971_97179

def piano_cost : ℝ := 500
def num_lessons : ℕ := 20
def lesson_cost : ℝ := 40
def discount_rate : ℝ := 0.25

def total_cost : ℝ := 
  piano_cost + (1 - discount_rate) * (num_lessons : ℝ) * lesson_cost

theorem total_cost_is_1100 : total_cost = 1100 := by sorry

end total_cost_is_1100_l971_97179


namespace percentage_of_sikh_boys_l971_97148

theorem percentage_of_sikh_boys (total_boys : ℕ) (muslim_percentage hindu_percentage : ℚ) 
  (other_boys : ℕ) (h1 : total_boys = 850) (h2 : muslim_percentage = 40/100) 
  (h3 : hindu_percentage = 28/100) (h4 : other_boys = 187) : 
  (total_boys - (muslim_percentage * total_boys + hindu_percentage * total_boys + other_boys)) / total_boys = 1/10 := by
  sorry

end percentage_of_sikh_boys_l971_97148


namespace eighth_term_ratio_l971_97163

/-- Two arithmetic sequences U and V with their partial sums Un and Vn -/
def arithmetic_sequences (U V : ℕ → ℚ) : Prop :=
  ∃ (u f v g : ℚ), ∀ n,
    U n = u + (n - 1) * f ∧
    V n = v + (n - 1) * g

/-- Partial sum of the first n terms of an arithmetic sequence -/
def partial_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem eighth_term_ratio
  (U V : ℕ → ℚ)
  (h_arith : arithmetic_sequences U V)
  (h_ratio : ∀ n, partial_sum U n / partial_sum V n = (5 * n + 5) / (3 * n + 9)) :
  U 8 / V 8 = 5 / 6 := by
  sorry

end eighth_term_ratio_l971_97163


namespace sum_of_distinct_prime_factors_l971_97135

def n : ℕ := 240360

theorem sum_of_distinct_prime_factors :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (n + 1))) id) = 62 :=
sorry

end sum_of_distinct_prime_factors_l971_97135


namespace vegetable_planting_methods_l971_97161

theorem vegetable_planting_methods (n m : ℕ) (hn : n = 5) (hm : m = 4) :
  (n.choose m) * (m.factorial) = 120 := by
  sorry

end vegetable_planting_methods_l971_97161


namespace f_deriv_l971_97192

/-- The function f(x) = 2x + 3 -/
def f (x : ℝ) : ℝ := 2 * x + 3

/-- Theorem: The derivative of f(x) = 2x + 3 is equal to 2 -/
theorem f_deriv : deriv f = λ _ => 2 := by sorry

end f_deriv_l971_97192


namespace simplify_expression_proof_l971_97124

noncomputable def simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : ℝ :=
  (a / b * (b - 4 * a^6 / b^3)^(1/3) - a^2 * (b / a^6 - 4 / b^3)^(1/3) + 2 / (a * b) * (a^3 * b^4 - 4 * a^9)^(1/3)) / ((b^2 - 2 * a^3)^(1/3) / b^2)

theorem simplify_expression_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  simplify_expression a b ha hb = (a + b) * (b^2 + 2 * a^3)^(1/3) := by
  sorry

end simplify_expression_proof_l971_97124


namespace fifth_root_unity_product_l971_97159

/-- Given a complex number z that is a fifth root of unity, 
    prove that the product (1 - z)(1 - z^2)(1 - z^3)(1 - z^4) equals 5 -/
theorem fifth_root_unity_product (z : ℂ) 
  (h : z = Complex.exp (2 * Real.pi * I / 5)) : 
  (1 - z) * (1 - z^2) * (1 - z^3) * (1 - z^4) = 5 := by
  sorry

end fifth_root_unity_product_l971_97159


namespace i_cubed_plus_i_squared_in_third_quadrant_l971_97181

def i : ℂ := Complex.I

-- Define the quadrants of the complex plane
def first_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im > 0
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0
def fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

theorem i_cubed_plus_i_squared_in_third_quadrant :
  third_quadrant (i^3 + i^2) :=
by
  sorry

end i_cubed_plus_i_squared_in_third_quadrant_l971_97181


namespace walking_distance_l971_97170

/-- 
Given a person who walks for time t hours:
- At 12 km/hr, they cover a distance of 12t km
- At 20 km/hr, they cover a distance of 20t km
- The difference between these distances is 30 km

Prove that the actual distance travelled at 12 km/hr is 45 km
-/
theorem walking_distance (t : ℝ) 
  (h1 : 20 * t = 12 * t + 30) : 12 * t = 45 := by sorry

end walking_distance_l971_97170


namespace no_rational_roots_l971_97119

theorem no_rational_roots : ∀ (p q : ℤ), q ≠ 0 → 3 * (p / q)^4 - 2 * (p / q)^3 - 8 * (p / q)^2 + (p / q) + 1 ≠ 0 := by
  sorry

end no_rational_roots_l971_97119


namespace edward_total_money_l971_97198

-- Define the variables
def dollars_per_lawn : ℕ := 8
def lawns_mowed : ℕ := 5
def initial_savings : ℕ := 7

-- Define the theorem
theorem edward_total_money :
  dollars_per_lawn * lawns_mowed + initial_savings = 47 :=
by
  sorry

end edward_total_money_l971_97198


namespace cargo_arrival_time_l971_97190

/-- Calculates the time between leaving the port and arriving at the warehouse -/
def timeBetweenPortAndWarehouse (navigationTime : ℕ) (customsTime : ℕ) (departureTime : ℕ) (expectedArrival : ℕ) : ℕ :=
  departureTime - (navigationTime + customsTime + expectedArrival)

/-- Theorem stating that the cargo arrives at the warehouse 1 day after leaving the port -/
theorem cargo_arrival_time :
  ∀ (navigationTime customsTime departureTime expectedArrival : ℕ),
    navigationTime = 21 →
    customsTime = 4 →
    departureTime = 30 →
    expectedArrival = 2 →
    timeBetweenPortAndWarehouse navigationTime customsTime departureTime expectedArrival = 1 := by
  sorry

#eval timeBetweenPortAndWarehouse 21 4 30 2

end cargo_arrival_time_l971_97190


namespace function_shift_l971_97186

theorem function_shift (f : ℝ → ℝ) :
  (∀ x, f (x - 1) = x^2 + 4*x - 5) →
  (∀ x, f (x + 1) = x^2 + 8*x + 7) :=
by
  sorry

end function_shift_l971_97186


namespace distance_focus_to_asymptotes_l971_97189

/-- The distance from the focus of the parabola x^2 = 8y to the asymptotes of the hyperbola x^2 - y^2/9 = 1 is √10 / 5 -/
theorem distance_focus_to_asymptotes :
  let parabola := {p : ℝ × ℝ | p.1^2 = 8 * p.2}
  let hyperbola := {p : ℝ × ℝ | p.1^2 - p.2^2 / 9 = 1}
  let focus : ℝ × ℝ := (0, 2)
  let asymptote (x : ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 ∨ p.2 = -3 * p.1}
  let distance (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) := 
    Real.sqrt (10) / 5
  ∀ p ∈ parabola, p.1^2 = 8 * p.2 →
  ∀ h ∈ hyperbola, h.1^2 - h.2^2 / 9 = 1 →
  distance focus (asymptote 0) = Real.sqrt 10 / 5 :=
by sorry

end distance_focus_to_asymptotes_l971_97189


namespace cut_cube_total_count_cube_cutting_problem_l971_97182

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  /-- The number of smaller cubes along each edge of the original cube -/
  edge_count : ℕ
  /-- The number of smaller cubes painted on exactly two faces -/
  two_face_painted : ℕ

/-- Theorem stating that a cube cut into smaller cubes with 12 two-face painted cubes results in 27 total cubes -/
theorem cut_cube_total_count (c : CutCube) (h1 : c.two_face_painted = 12) : 
  c.edge_count ^ 3 = 27 := by
  sorry

/-- Main theorem proving the solution to the original problem -/
theorem cube_cutting_problem : 
  ∃ (c : CutCube), c.two_face_painted = 12 ∧ c.edge_count ^ 3 = 27 := by
  sorry

end cut_cube_total_count_cube_cutting_problem_l971_97182


namespace unique_solution_quadratic_l971_97103

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
  sorry

end unique_solution_quadratic_l971_97103


namespace shifted_quadratic_equation_solutions_l971_97155

/-- Given an equation a(x+m)²+b=0 with solutions x₁=-2 and x₂=1, 
    prove that a(x+m+2)²+b=0 has solutions x₁=-4 and x₂=-1 -/
theorem shifted_quadratic_equation_solutions 
  (a m b : ℝ) 
  (ha : a ≠ 0) 
  (h1 : a * ((-2 : ℝ) + m)^2 + b = 0) 
  (h2 : a * ((1 : ℝ) + m)^2 + b = 0) :
  a * ((-4 : ℝ) + m + 2)^2 + b = 0 ∧ a * ((-1 : ℝ) + m + 2)^2 + b = 0 :=
sorry

end shifted_quadratic_equation_solutions_l971_97155


namespace parabola_intersection_theorem_l971_97139

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the perpendicular bisector of a line with slope m
def perpendicular_bisector (m : ℝ) (x y : ℝ) : Prop := x = -1/m * y + (2*m^2 + 3)

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

theorem parabola_intersection_theorem (m : ℝ) :
  ∃ (xA yA xB yB xC yC xD yD : ℝ),
    -- A and B are on the parabola and the line through focus
    parabola xA yA ∧ parabola xB yB ∧
    line_through_focus m xA yA ∧ line_through_focus m xB yB ∧
    -- C and D are on the parabola and the perpendicular bisector
    parabola xC yC ∧ parabola xD yD ∧
    perpendicular_bisector m xC yC ∧ perpendicular_bisector m xD yD ∧
    -- AC is perpendicular to AD
    dot_product (xC - xA) (yC - yA) (xD - xA) (yD - yA) = 0 →
    m = 1 ∨ m = -1 :=
sorry

end parabola_intersection_theorem_l971_97139


namespace julia_birth_year_l971_97110

/-- Given that Wayne is 37 years old in 2021, Peter is 3 years older than Wayne,
    and Julia is 2 years older than Peter, prove that Julia was born in 1979. -/
theorem julia_birth_year (wayne_age : ℕ) (peter_age_diff : ℕ) (julia_age_diff : ℕ) :
  wayne_age = 37 →
  peter_age_diff = 3 →
  julia_age_diff = 2 →
  2021 - wayne_age - peter_age_diff - julia_age_diff = 1979 := by
  sorry

end julia_birth_year_l971_97110


namespace count_triangles_in_polygon_l971_97160

/-- The number of triangles in a regular n-sided polygon (n ≥ 6) whose sides are formed by diagonals
    and whose vertices are vertices of the polygon -/
def triangles_in_polygon (n : ℕ) : ℕ :=
  n * (n - 4) * (n - 5) / 6

/-- Theorem stating the number of triangles in a regular n-sided polygon (n ≥ 6) whose sides are formed
    by diagonals and whose vertices are vertices of the polygon -/
theorem count_triangles_in_polygon (n : ℕ) (h : n ≥ 6) :
  triangles_in_polygon n = n * (n - 4) * (n - 5) / 6 := by
  sorry

end count_triangles_in_polygon_l971_97160


namespace check_to_new_balance_ratio_l971_97166

def initial_balance : ℚ := 150
def check_amount : ℚ := 50

def new_balance : ℚ := initial_balance + check_amount

theorem check_to_new_balance_ratio :
  check_amount / new_balance = 1 / 4 := by sorry

end check_to_new_balance_ratio_l971_97166


namespace sqrt_sum_equals_nineteen_fifteenths_l971_97118

theorem sqrt_sum_equals_nineteen_fifteenths (w x z : ℝ) 
  (hw : w = 4) (hx : x = 9) (hz : z = 25) : 
  Real.sqrt (w / x) + Real.sqrt (x / z) = 19 / 15 := by
  sorry

end sqrt_sum_equals_nineteen_fifteenths_l971_97118


namespace basketball_free_throws_l971_97145

theorem basketball_free_throws (total_players : Nat) (goalkeepers : Nat) : 
  total_players = 18 → goalkeepers = 2 → 
  (total_players - goalkeepers) * goalkeepers = 34 := by
  sorry

end basketball_free_throws_l971_97145


namespace unique_two_digit_integer_l971_97165

theorem unique_two_digit_integer (u : ℕ) : 
  (u ≥ 10 ∧ u ≤ 99) →  -- u is a two-digit positive integer
  (13 * u) % 100 = 52 →  -- when multiplied by 13, the last two digits are 52
  u = 4 := by
sorry

end unique_two_digit_integer_l971_97165


namespace tv_selection_theorem_l971_97169

/-- The number of televisions of type A -/
def typeA : ℕ := 3

/-- The number of televisions of type B -/
def typeB : ℕ := 3

/-- The number of televisions of type C -/
def typeC : ℕ := 4

/-- The total number of televisions -/
def totalTVs : ℕ := typeA + typeB + typeC

/-- The number of televisions to be selected -/
def selectCount : ℕ := 3

/-- Calculates the number of ways to select r items from n items -/
def combination (n r : ℕ) : ℕ :=
  Nat.choose n r

/-- The theorem to be proved -/
theorem tv_selection_theorem : 
  combination totalTVs selectCount - 
  (combination typeA selectCount + combination typeB selectCount + combination typeC selectCount) = 114 := by
  sorry

end tv_selection_theorem_l971_97169


namespace complex_sum_theorem_l971_97162

theorem complex_sum_theorem (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^3 = 1) (hω2 : ω ≠ 1)
  (h : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 := by
  sorry

end complex_sum_theorem_l971_97162


namespace min_value_expression_l971_97121

theorem min_value_expression (x y : ℝ) (h : x^2 + y^2 ≤ 1) :
  ∃ (m : ℝ), (∀ (a b : ℝ), a^2 + b^2 ≤ 1 → |2*a + b - 2| + |6 - a - 3*b| ≥ m) ∧
             (∃ (c d : ℝ), c^2 + d^2 ≤ 1 ∧ |2*c + d - 2| + |6 - c - 3*d| = m) ∧
             m = 3 := by
  sorry

end min_value_expression_l971_97121


namespace rock_climbing_participants_number_of_rock_climbing_participants_l971_97191

/- Define the total number of students in the school -/
def total_students : ℕ := 800

/- Define the percentage of students who went on the camping trip -/
def camping_percentage : ℚ := 25 / 100

/- Define the percentage of camping students who took more than $100 -/
def more_than_100_percentage : ℚ := 15 / 100

/- Define the percentage of camping students who took exactly $100 -/
def exactly_100_percentage : ℚ := 30 / 100

/- Define the percentage of camping students who took between $50 and $100 -/
def between_50_and_100_percentage : ℚ := 40 / 100

/- Define the percentage of students with more than $100 who participated in rock climbing -/
def rock_climbing_participation_percentage : ℚ := 50 / 100

/- Theorem stating the number of students who participated in rock climbing -/
theorem rock_climbing_participants : ℕ := by
  sorry

/- Main theorem to prove -/
theorem number_of_rock_climbing_participants : rock_climbing_participants = 15 := by
  sorry

end rock_climbing_participants_number_of_rock_climbing_participants_l971_97191


namespace rectangular_prism_edge_pairs_l971_97193

/-- A rectangular prism -/
structure RectangularPrism where
  edges : Finset Edge
  faces : Finset Face

/-- An edge of a rectangular prism -/
structure Edge where
  -- Add necessary fields

/-- A face of a rectangular prism -/
structure Face where
  -- Add necessary fields

/-- Two edges are parallel -/
def parallel (e1 e2 : Edge) : Prop := sorry

/-- Two edges are perpendicular -/
def perpendicular (e1 e2 : Edge) : Prop := sorry

/-- The set of pairs of parallel edges in a rectangular prism -/
def parallelEdgePairs (rp : RectangularPrism) : Finset (Edge × Edge) :=
  sorry

/-- The set of pairs of perpendicular edges in a rectangular prism -/
def perpendicularEdgePairs (rp : RectangularPrism) : Finset (Edge × Edge) :=
  sorry

theorem rectangular_prism_edge_pairs (rp : RectangularPrism) :
  (Finset.card (parallelEdgePairs rp) = 8) ∧
  (Finset.card (perpendicularEdgePairs rp) = 20) := by
  sorry

end rectangular_prism_edge_pairs_l971_97193


namespace prob_same_color_specific_l971_97101

/-- Probability of drawing two marbles of the same color -/
def prob_same_color (red white blue green : ℕ) : ℚ :=
  let total := red + white + blue + green
  let prob_red := (red * (red - 1)) / (total * (total - 1))
  let prob_white := (white * (white - 1)) / (total * (total - 1))
  let prob_blue := (blue * (blue - 1)) / (total * (total - 1))
  let prob_green := (green * (green - 1)) / (total * (total - 1))
  prob_red + prob_white + prob_blue + prob_green

theorem prob_same_color_specific : prob_same_color 5 6 7 3 = 7 / 30 := by
  sorry

#eval prob_same_color 5 6 7 3

end prob_same_color_specific_l971_97101


namespace zachary_pushups_l971_97122

/-- Given that Zachary did 14 crunches and a total of 67 push-ups and crunches,
    prove that Zachary did 53 push-ups. -/
theorem zachary_pushups :
  ∀ (zachary_pushups zachary_crunches : ℕ),
    zachary_crunches = 14 →
    zachary_pushups + zachary_crunches = 67 →
    zachary_pushups = 53 :=
by
  sorry

end zachary_pushups_l971_97122


namespace dandelion_counts_l971_97144

/-- Represents the state of dandelions in the meadow on a given day -/
structure DandelionState where
  yellow : ℕ
  white : ℕ

/-- The lifecycle of dandelions -/
def dandelionLifecycle : Prop :=
  ∀ d : DandelionState, d.yellow = d.white

/-- Yesterday's dandelion state -/
def yesterday : DandelionState :=
  { yellow := 20, white := 14 }

/-- Today's dandelion state -/
def today : DandelionState :=
  { yellow := 15, white := 11 }

/-- Theorem: Given the dandelion lifecycle and the counts for yesterday and today,
    the number of yellow dandelions the day before yesterday was 25, and
    the number of white dandelions tomorrow will be 9. -/
theorem dandelion_counts
  (h : dandelionLifecycle)
  (hy : yesterday.yellow = 20 ∧ yesterday.white = 14)
  (ht : today.yellow = 15 ∧ today.white = 11) :
  (yesterday.white + today.white = 25) ∧
  (yesterday.yellow - today.white = 9) :=
by sorry

end dandelion_counts_l971_97144


namespace probability_of_double_is_two_ninths_l971_97180

/-- Represents a domino tile with two squares -/
structure Domino :=
  (first : Nat)
  (second : Nat)

/-- The set of all possible dominos with integers from 0 to 7 -/
def dominoSet : Finset Domino :=
  sorry

/-- Predicate to check if a domino is a double -/
def isDouble (d : Domino) : Bool :=
  d.first = d.second

/-- The probability of selecting a double from the domino set -/
def probabilityOfDouble : ℚ :=
  sorry

/-- Theorem stating that the probability of selecting a double is 2/9 -/
theorem probability_of_double_is_two_ninths :
  probabilityOfDouble = 2 / 9 := by
  sorry

end probability_of_double_is_two_ninths_l971_97180


namespace six_digit_divisible_by_eleven_l971_97143

theorem six_digit_divisible_by_eleven (d : Nat) : 
  d < 10 → (67890 * 10 + d) % 11 = 0 ↔ d = 9 := by
  sorry

end six_digit_divisible_by_eleven_l971_97143


namespace soccer_penalty_kicks_l971_97174

/-- Calculates the total number of penalty kicks in a soccer team drill -/
theorem soccer_penalty_kicks (total_players : ℕ) (goalies : ℕ) : 
  total_players = 25 → goalies = 4 → total_players * (goalies - 1) = 96 := by
  sorry

end soccer_penalty_kicks_l971_97174


namespace inequality_proof_l971_97194

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1/3) * (a + b + c)^2 := by
  sorry

end inequality_proof_l971_97194


namespace min_lines_for_100_squares_l971_97178

/-- The number of squares formed by n lines when n is odd -/
def squares_odd (n : ℕ) : ℕ := ((n - 3) * (n - 1) * (n + 1)) / 24

/-- The number of squares formed by n lines when n is even -/
def squares_even (n : ℕ) : ℕ := ((n - 2) * n * (n - 1)) / 24

/-- The maximum number of squares that can be formed by n lines -/
def max_squares (n : ℕ) : ℕ :=
  if n % 2 = 0 then squares_even n else squares_odd n

/-- Predicate indicating whether it's possible to form exactly k squares with n lines -/
def can_form_squares (n k : ℕ) : Prop :=
  k ≤ max_squares n ∧ k > max_squares (n - 1)

theorem min_lines_for_100_squares :
  ∃ n : ℕ, can_form_squares n 100 ∧ ∀ m : ℕ, m < n → ¬can_form_squares m 100 :=
sorry

end min_lines_for_100_squares_l971_97178
