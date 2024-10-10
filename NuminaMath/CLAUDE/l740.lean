import Mathlib

namespace temperature_change_over_700_years_l740_74097

/-- Calculates the total temperature change over 700 years and converts it to Fahrenheit. -/
theorem temperature_change_over_700_years :
  let rate1 : ℝ := 3  -- rate for first 300 years (units per century)
  let rate2 : ℝ := 5  -- rate for next 200 years (units per century)
  let rate3 : ℝ := 2  -- rate for last 200 years (units per century)
  let period1 : ℝ := 3  -- first period in centuries
  let period2 : ℝ := 2  -- second period in centuries
  let period3 : ℝ := 2  -- third period in centuries
  let total_change_celsius : ℝ := rate1 * period1 + rate2 * period2 + rate3 * period3
  let total_change_fahrenheit : ℝ := total_change_celsius * (9/5) + 32
  total_change_celsius = 23 ∧ total_change_fahrenheit = 73.4 := by
  sorry

end temperature_change_over_700_years_l740_74097


namespace factorization_sum_l740_74045

theorem factorization_sum (a b : ℤ) : 
  (∀ x : ℝ, 24 * x^2 - 50 * x - 84 = (6 * x + a) * (4 * x + b)) → 
  a + 2 * b = -17 := by
sorry

end factorization_sum_l740_74045


namespace specific_triangle_area_l740_74006

/-- RightTriangle represents a right triangle with specific properties -/
structure RightTriangle where
  AB : ℝ  -- Length of hypotenuse
  median_CA : ℝ → ℝ  -- Equation of median to side CA
  median_CB : ℝ → ℝ  -- Equation of median to side CB

/-- Calculate the area of the right triangle -/
def triangle_area (t : RightTriangle) : ℝ := sorry

/-- Theorem stating the area of the specific right triangle -/
theorem specific_triangle_area :
  let t : RightTriangle := {
    AB := 60,
    median_CA := λ x => x + 3,
    median_CB := λ x => 2 * x + 4
  }
  triangle_area t = 400 := by sorry

end specific_triangle_area_l740_74006


namespace lee_weight_l740_74060

/-- Given Anna's and Lee's weights satisfying certain conditions, prove Lee's weight is 144 pounds. -/
theorem lee_weight (anna lee : ℝ) 
  (h1 : anna + lee = 240)
  (h2 : lee - anna = lee / 3) : 
  lee = 144 := by sorry

end lee_weight_l740_74060


namespace white_dogs_count_l740_74085

theorem white_dogs_count (total brown black : ℕ) 
  (h_total : total = 45)
  (h_brown : brown = 20)
  (h_black : black = 15) :
  total - (brown + black) = 10 := by
  sorry

end white_dogs_count_l740_74085


namespace study_group_formation_l740_74005

def number_of_ways (n : ℕ) (g2 : ℕ) (g3 : ℕ) : ℕ :=
  (Nat.choose n 3 * Nat.choose (n - 3) 3) / 2 *
  ((Nat.choose (n - 6) 2 * Nat.choose (n - 8) 2 * Nat.choose (n - 10) 2) / 6)

theorem study_group_formation :
  number_of_ways 12 3 2 = 138600 := by
  sorry

end study_group_formation_l740_74005


namespace product_zero_implies_factor_zero_l740_74027

theorem product_zero_implies_factor_zero (a b : ℝ) : a * b = 0 → a = 0 ∨ b = 0 := by
  sorry

end product_zero_implies_factor_zero_l740_74027


namespace sequences_sum_product_l740_74032

/-- Two sequences satisfying the given conditions -/
def Sequences (α β γ : ℕ) (a b : ℕ → ℕ) : Prop :=
  (α < γ) ∧ 
  (α * γ = β^2 + 1) ∧
  (a 0 = 1) ∧ 
  (b 0 = 1) ∧
  (∀ n, a (n + 1) = α * a n + β * b n) ∧
  (∀ n, b (n + 1) = β * a n + γ * b n)

/-- The main theorem to be proved -/
theorem sequences_sum_product (α β γ : ℕ) (a b : ℕ → ℕ) 
  (h : Sequences α β γ a b) :
  ∀ m n : ℕ, a (m + n) + b (m + n) = a m * a n + b m * b n :=
sorry

end sequences_sum_product_l740_74032


namespace mn_pq_ratio_l740_74054

-- Define the points on a real line
variable (A B C M N P Q : ℝ)

-- Define the conditions
variable (h1 : A ≤ B ∧ B ≤ C)  -- B is on line segment AC
variable (h2 : M = (A + B) / 2)  -- M is midpoint of AB
variable (h3 : N = (A + C) / 2)  -- N is midpoint of AC
variable (h4 : P = (N + A) / 2)  -- P is midpoint of NA
variable (h5 : Q = (M + A) / 2)  -- Q is midpoint of MA

-- State the theorem
theorem mn_pq_ratio :
  |N - M| / |P - Q| = 2 :=
sorry

end mn_pq_ratio_l740_74054


namespace positive_numbers_inequality_l740_74021

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) := by
  sorry

end positive_numbers_inequality_l740_74021


namespace count_with_zero_eq_952_l740_74026

/-- A function that checks if a positive integer contains the digit 0 in its base-ten representation -/
def containsZero (n : ℕ+) : Bool :=
  sorry

/-- The count of positive integers less than or equal to 2500 that contain the digit 0 -/
def countWithZero : ℕ :=
  sorry

/-- Theorem stating that the count of positive integers less than or equal to 2500 
    containing the digit 0 is 952 -/
theorem count_with_zero_eq_952 : countWithZero = 952 := by
  sorry

end count_with_zero_eq_952_l740_74026


namespace star_equation_solution_l740_74064

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- Theorem statement
theorem star_equation_solution (h : ℝ) :
  star 8 h = 11 → h = 6 := by
  sorry

end star_equation_solution_l740_74064


namespace power_minus_ten_over_nine_equals_ten_l740_74075

theorem power_minus_ten_over_nine_equals_ten : (10^2 - 10) / 9 = 10 := by
  sorry

end power_minus_ten_over_nine_equals_ten_l740_74075


namespace sum_first_five_eq_l740_74037

/-- A geometric progression with given fourth and fifth terms -/
structure GeometricProgression where
  b₄ : ℚ  -- Fourth term
  b₅ : ℚ  -- Fifth term
  h₄ : b₄ = 1 / 25
  h₅ : b₅ = 1 / 125

/-- The sum of the first five terms of a geometric progression -/
def sum_first_five (gp : GeometricProgression) : ℚ :=
  -- Definition of the sum (to be proved)
  781 / 125

/-- Theorem stating that the sum of the first five terms is 781/125 -/
theorem sum_first_five_eq (gp : GeometricProgression) :
  sum_first_five gp = 781 / 125 := by
  sorry

#eval sum_first_five ⟨1/25, 1/125, rfl, rfl⟩

end sum_first_five_eq_l740_74037


namespace paint_ratio_circular_signs_l740_74080

theorem paint_ratio_circular_signs (d : ℝ) (h : d > 0) : 
  let D := 7 * d
  (π * (D / 2)^2) / (π * (d / 2)^2) = 49 := by sorry

end paint_ratio_circular_signs_l740_74080


namespace integer_sqrt_pair_l740_74093

theorem integer_sqrt_pair : ∃! (x y : ℕ), 
  ((x = 88209 ∧ y = 90288) ∨
   (x = 82098 ∧ y = 89028) ∨
   (x = 28098 ∧ y = 89082) ∨
   (x = 90882 ∧ y = 28809)) ∧
  ∃ (z : ℕ), z^2 = x^2 + y^2 := by
sorry

end integer_sqrt_pair_l740_74093


namespace water_displacement_cubed_l740_74000

/-- Given a cylindrical tank and a partially submerged cube, calculate the volume of water displaced cubed. -/
theorem water_displacement_cubed (tank_radius : ℝ) (cube_side : ℝ) (h : tank_radius = 3 ∧ cube_side = 6) : 
  let submerged_height := cube_side / 2
  let tank_diameter := 2 * tank_radius
  let inscribed_square_side := tank_diameter / Real.sqrt 2
  let intersection_area := inscribed_square_side ^ 2
  let displaced_volume := intersection_area * submerged_height
  displaced_volume ^ 3 = 157464 := by
  sorry

end water_displacement_cubed_l740_74000


namespace minimum_box_cost_greenville_box_cost_l740_74069

/-- The minimum amount spent on boxes for packaging a fine arts collection -/
theorem minimum_box_cost (box_length box_width box_height : ℝ) 
  (box_cost : ℝ) (collection_volume : ℝ) : ℝ :=
  let box_volume := box_length * box_width * box_height
  let num_boxes := collection_volume / box_volume
  num_boxes * box_cost

/-- The specific case for Greenville State University -/
theorem greenville_box_cost : 
  minimum_box_cost 20 20 12 0.40 2400000 = 200 := by
  sorry

end minimum_box_cost_greenville_box_cost_l740_74069


namespace ellipse_equation_hyperbola_equation_l740_74041

-- Define the ellipse properties
def ellipse_axis_sum : ℝ := 18
def ellipse_focal_length : ℝ := 6

-- Define the reference ellipse for the hyperbola
def reference_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the point Q
def point_Q : ℝ × ℝ := (2, 1)

-- Theorem for the ellipse equation
theorem ellipse_equation (x y : ℝ) :
  (x^2 / 25 + y^2 / 16 = 1) ∨ (x^2 / 16 + y^2 / 25 = 1) :=
sorry

-- Theorem for the hyperbola equation
theorem hyperbola_equation (x y : ℝ) :
  x^2 / 2 - y^2 = 1 :=
sorry

end ellipse_equation_hyperbola_equation_l740_74041


namespace race_car_probability_l740_74096

/-- A circular racetrack with a given length -/
structure CircularTrack where
  length : ℝ
  length_positive : length > 0

/-- A race car on the circular track -/
structure RaceCar where
  track : CircularTrack
  start_position : ℝ
  travel_distance : ℝ

/-- The probability of the car ending within a certain distance of a specific point -/
def end_probability (car : RaceCar) (target : ℝ) (range : ℝ) : ℝ :=
  sorry

/-- Theorem stating the probability for the specific problem -/
theorem race_car_probability (track : CircularTrack) 
  (h1 : track.length = 3)
  (car : RaceCar)
  (h2 : car.track = track)
  (h3 : car.travel_distance = 0.5)
  (target : ℝ)
  (h4 : target = 2.5) :
  end_probability car target 0.5 = 1/3 := by
  sorry

end race_car_probability_l740_74096


namespace infinite_series_sum_l740_74099

theorem infinite_series_sum (a b : ℝ) 
  (h : (a / (b + 1)) / (1 - 1 / (b + 1)) = 3) : 
  (a / (a + 2*b)) / (1 - 1 / (a + 2*b)) = 3*(b + 1) / (5*b + 2) := by
  sorry

end infinite_series_sum_l740_74099


namespace language_study_difference_l740_74055

theorem language_study_difference (total : ℕ) (german_min german_max russian_min russian_max : ℕ) :
  total = 2500 →
  german_min = 1750 →
  german_max = 1875 →
  russian_min = 1000 →
  russian_max = 1125 →
  let m := german_min + russian_min - total
  let M := german_max + russian_max - total
  M - m = 250 := by sorry

end language_study_difference_l740_74055


namespace pie_chart_angle_l740_74017

theorem pie_chart_angle (percentage : ℝ) (angle : ℝ) :
  percentage = 0.15 →
  angle = percentage * 360 →
  angle = 54 := by
  sorry

end pie_chart_angle_l740_74017


namespace value_of_T_l740_74044

theorem value_of_T : ∃ T : ℚ, (1/2 : ℚ) * (1/7 : ℚ) * T = (1/3 : ℚ) * (1/5 : ℚ) * 60 ∧ T = 56 := by
  sorry

end value_of_T_l740_74044


namespace polynomial_value_at_zero_l740_74053

theorem polynomial_value_at_zero (p : Polynomial ℝ) : 
  (Polynomial.degree p = 7) →
  (∀ n : Nat, n ≤ 7 → p.eval (3^n) = (3^n)⁻¹) →
  p.eval 0 = 19682 / 6561 := by
sorry

end polynomial_value_at_zero_l740_74053


namespace long_tennis_players_l740_74040

theorem long_tennis_players (total : ℕ) (football : ℕ) (both : ℕ) (neither : ℕ) :
  total = 36 →
  football = 26 →
  both = 17 →
  neither = 7 →
  ∃ (long_tennis : ℕ), long_tennis = 20 ∧ 
    total = football + long_tennis - both + neither :=
by sorry

end long_tennis_players_l740_74040


namespace farmer_apples_l740_74068

theorem farmer_apples (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 924 → given_away = 639 → remaining = initial - given_away → remaining = 285 := by
  sorry

end farmer_apples_l740_74068


namespace spinner_final_direction_l740_74074

-- Define the possible directions
inductive Direction
| North
| East
| South
| West

-- Define a function to calculate the final direction
def finalDirection (initialDir : Direction) (clockwiseRev : ℚ) (counterClockwiseRev : ℚ) : Direction :=
  sorry

-- Theorem statement
theorem spinner_final_direction :
  finalDirection Direction.North (7/2) (21/4) = Direction.East :=
sorry

end spinner_final_direction_l740_74074


namespace goals_theorem_l740_74043

/-- The total number of goals scored in the league against Barca -/
def total_goals : ℕ := 300

/-- The number of players who scored goals -/
def num_players : ℕ := 2

/-- The number of goals scored by each player -/
def goals_per_player : ℕ := 30

/-- The percentage of total goals scored by the two players -/
def percentage : ℚ := 1/5

theorem goals_theorem (h1 : num_players * goals_per_player = (percentage * total_goals).num) :
  total_goals = 300 := by
  sorry

end goals_theorem_l740_74043


namespace base12_addition_l740_74003

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

/-- Represents a number in base 12 as a list of digits --/
def Base12 := List Digit12

/-- Convert a base 12 number to its decimal representation --/
def toDecimal (n : Base12) : Nat := sorry

/-- Convert a decimal number to its base 12 representation --/
def fromDecimal (n : Nat) : Base12 := sorry

/-- Addition operation for base 12 numbers --/
def addBase12 (a b : Base12) : Base12 := sorry

/-- The main theorem --/
theorem base12_addition :
  let n1 : Base12 := [Digit12.D5, Digit12.A, Digit12.D3]
  let n2 : Base12 := [Digit12.D2, Digit12.B, Digit12.D8]
  addBase12 n1 n2 = [Digit12.D8, Digit12.D9, Digit12.D6] := by sorry

end base12_addition_l740_74003


namespace ellipse_properties_l740_74079

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_minor_axis : b = Real.sqrt 3
  h_eccentricity : a / Real.sqrt (a^2 - b^2) = 2

/-- The standard form of the ellipse -/
def standard_form (e : Ellipse) : Prop :=
  e.a^2 = 4 ∧ e.b^2 = 3

/-- The maximum area of triangle F₁AB -/
def max_triangle_area (e : Ellipse) : ℝ := 3

/-- Main theorem stating the properties of the ellipse -/
theorem ellipse_properties (e : Ellipse) :
  standard_form e ∧ max_triangle_area e = 3 := by sorry

end ellipse_properties_l740_74079


namespace rectangle_perimeter_l740_74056

theorem rectangle_perimeter (area : ℝ) (length : ℝ) (h1 : area = 192) (h2 : length = 24) :
  2 * (length + area / length) = 64 := by
  sorry

end rectangle_perimeter_l740_74056


namespace fifth_row_solution_l740_74001

/-- Represents the possible values in the grid -/
inductive GridValue
  | Two
  | Zero
  | One
  | Five
  | Blank

/-- Represents a 5x5 grid -/
def Grid := Fin 5 → Fin 5 → GridValue

/-- Check if a grid satisfies the row constraint -/
def satisfiesRowConstraint (g : Grid) : Prop :=
  ∀ row, ∃! i, g row i = GridValue.Two ∧
         ∃! i, g row i = GridValue.Zero ∧
         ∃! i, g row i = GridValue.One ∧
         ∃! i, g row i = GridValue.Five

/-- Check if a grid satisfies the column constraint -/
def satisfiesColumnConstraint (g : Grid) : Prop :=
  ∀ col, ∃! i, g i col = GridValue.Two ∧
         ∃! i, g i col = GridValue.Zero ∧
         ∃! i, g i col = GridValue.One ∧
         ∃! i, g i col = GridValue.Five

/-- Check if a grid satisfies the diagonal constraint -/
def satisfiesDiagonalConstraint (g : Grid) : Prop :=
  ∀ i j, i < 4 → j < 4 →
    (g i j ≠ GridValue.Blank → g (i+1) (j+1) ≠ g i j) ∧
    (g i (j+1) ≠ GridValue.Blank → g (i+1) j ≠ g i (j+1))

/-- The main theorem stating the solution for the fifth row -/
theorem fifth_row_solution (g : Grid) 
  (hrow : satisfiesRowConstraint g)
  (hcol : satisfiesColumnConstraint g)
  (hdiag : satisfiesDiagonalConstraint g) :
  g 4 0 = GridValue.One ∧
  g 4 1 = GridValue.Five ∧
  g 4 2 = GridValue.Blank ∧
  g 4 3 = GridValue.Blank ∧
  g 4 4 = GridValue.Two :=
sorry

end fifth_row_solution_l740_74001


namespace sticker_cost_theorem_l740_74051

/-- Calculates the total cost of buying stickers on two days -/
def total_sticker_cost (day1_packs : ℕ) (day1_price : ℚ) (day1_discount : ℚ)
                       (day2_packs : ℕ) (day2_price : ℚ) (day2_tax : ℚ) : ℚ :=
  let day1_cost := day1_packs * day1_price * (1 - day1_discount)
  let day2_cost := day2_packs * day2_price * (1 + day2_tax)
  day1_cost + day2_cost

/-- Theorem stating the total cost of buying stickers on two days -/
theorem sticker_cost_theorem :
  total_sticker_cost 15 (5/2) (1/10) 25 3 (1/20) = 225/2 := by
  sorry

end sticker_cost_theorem_l740_74051


namespace vector_properties_l740_74039

noncomputable section

def a (x : ℝ) : ℝ × ℝ := (Real.cos (3/2 * x), Real.sin (3/2 * x))
def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), Real.sin (x/2))

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def vector_sum (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def f (m x : ℝ) : ℝ := m * vector_magnitude (vector_sum (a x) (b x)) - dot_product (a x) (b x)

theorem vector_properties (m : ℝ) :
  (dot_product (a (π/4)) (b (π/4)) = Real.sqrt 2 / 2) ∧
  (vector_magnitude (vector_sum (a (π/4)) (b (π/4))) = Real.sqrt (2 + Real.sqrt 2)) ∧
  (∀ x ∈ Set.Icc 0 π, 
    (m > 2 → f m x ≤ 2*m - 3) ∧
    (0 ≤ m ∧ m ≤ 2 → f m x ≤ m^2/2 - 1) ∧
    (m < 0 → f m x ≤ -1)) :=
by sorry

end vector_properties_l740_74039


namespace business_partnership_problem_l740_74071

/-- A business partnership problem -/
theorem business_partnership_problem 
  (a_investment : ℕ) 
  (total_duration : ℕ) 
  (b_join_time : ℕ) 
  (profit_ratio_a : ℕ) 
  (profit_ratio_b : ℕ) 
  (h1 : a_investment = 3500)
  (h2 : total_duration = 12)
  (h3 : b_join_time = 8)
  (h4 : profit_ratio_a = 2)
  (h5 : profit_ratio_b = 3) : 
  ∃ (b_investment : ℕ), 
    b_investment = 1575 ∧ 
    (a_investment * total_duration) / (b_investment * (total_duration - b_join_time)) = 
    profit_ratio_a / profit_ratio_b :=
sorry

end business_partnership_problem_l740_74071


namespace unique_integer_prime_expressions_l740_74023

theorem unique_integer_prime_expressions : ∃! n : ℤ, 
  Nat.Prime (Int.natAbs (n^3 - 4*n^2 + 3*n - 35)) ∧ 
  Nat.Prime (Int.natAbs (n^2 + 4*n + 8)) ∧ 
  n = 5 := by sorry

end unique_integer_prime_expressions_l740_74023


namespace base8_47_equals_39_l740_74016

/-- Converts a two-digit base-8 number to base-10 --/
def base8_to_base10 (tens : Nat) (ones : Nat) : Nat :=
  tens * 8 + ones

/-- The base-8 number 47 is equal to 39 in base-10 --/
theorem base8_47_equals_39 : base8_to_base10 4 7 = 39 := by
  sorry

end base8_47_equals_39_l740_74016


namespace hyperbola_vertices_distance_l740_74083

/-- Given a hyperbola with equation (y-4)^2/32 - (x+3)^2/18 = 1,
    the distance between its vertices is 8√2. -/
theorem hyperbola_vertices_distance :
  let k : ℝ := 4
  let h : ℝ := -3
  let a_squared : ℝ := 32
  let b_squared : ℝ := 18
  let hyperbola_eq := fun (x y : ℝ) => (y - k)^2 / a_squared - (x - h)^2 / b_squared = 1
  let vertices_distance := 2 * Real.sqrt a_squared
  hyperbola_eq x y → vertices_distance = 8 * Real.sqrt 2 :=
by sorry

end hyperbola_vertices_distance_l740_74083


namespace positive_divisors_of_90_l740_74081

theorem positive_divisors_of_90 : Finset.card (Nat.divisors 90) = 12 := by
  sorry

end positive_divisors_of_90_l740_74081


namespace container_volume_ratio_l740_74030

theorem container_volume_ratio : 
  ∀ (A B : ℝ), A > 0 → B > 0 →
  (3/5 * A + 1/4 * B = 4/5 * B) →
  A / B = 11/12 := by
  sorry

end container_volume_ratio_l740_74030


namespace train_length_l740_74008

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 4 → speed * time * (1000 / 3600) = 100 := by
  sorry

end train_length_l740_74008


namespace tan_product_seventh_pi_l740_74042

theorem tan_product_seventh_pi : 
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7) = Real.sqrt 7 := by
  sorry

end tan_product_seventh_pi_l740_74042


namespace arnolds_mileage_calculation_l740_74002

/-- Calculates the total monthly driving mileage for Arnold given his car efficiencies and gas spending --/
def arnolds_mileage (efficiency1 efficiency2 efficiency3 : ℚ) (gas_price : ℚ) (monthly_spend : ℚ) : ℚ :=
  let total_cars := 3
  let inverse_efficiency := (1 / efficiency1 + 1 / efficiency2 + 1 / efficiency3) / total_cars
  monthly_spend / (gas_price * inverse_efficiency)

/-- Theorem stating that Arnold's total monthly driving mileage is (56 * 450) / 43 miles --/
theorem arnolds_mileage_calculation :
  let efficiency1 := 50
  let efficiency2 := 10
  let efficiency3 := 15
  let gas_price := 2
  let monthly_spend := 56
  arnolds_mileage efficiency1 efficiency2 efficiency3 gas_price monthly_spend = 56 * 450 / 43 := by
  sorry

end arnolds_mileage_calculation_l740_74002


namespace average_first_twelve_even_numbers_l740_74089

-- Define the first 12 even numbers
def firstTwelveEvenNumbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- Theorem to prove
theorem average_first_twelve_even_numbers :
  (List.sum firstTwelveEvenNumbers) / (List.length firstTwelveEvenNumbers) = 13 := by
  sorry


end average_first_twelve_even_numbers_l740_74089


namespace sequence_properties_l740_74024

/-- Sequence a_n is a first-degree function of n -/
def a (n : ℕ) : ℝ := 2 * n + 1

/-- Sequence b_n is composed of a_2, a_4, a_6, a_8, ... -/
def b (n : ℕ) : ℝ := a (2 * n)

theorem sequence_properties :
  (a 1 = 3) ∧ 
  (a 10 = 21) ∧ 
  (∀ n : ℕ, a_2009 = 4019) ∧
  (∀ n : ℕ, b n = 4 * n + 1) :=
sorry

end sequence_properties_l740_74024


namespace subtract_problem_l740_74090

theorem subtract_problem (x : ℤ) : x - 46 = 15 → x - 29 = 32 := by
  sorry

end subtract_problem_l740_74090


namespace complex_modulus_theorem_l740_74092

theorem complex_modulus_theorem : 
  let i : ℂ := Complex.I
  let T : ℂ := (1 + i)^19 - (1 - i)^19
  Complex.abs T = 512 := by sorry

end complex_modulus_theorem_l740_74092


namespace machine_production_l740_74063

/-- The number of shirts produced by a machine given its production rate and working time -/
def shirts_produced (rate : ℕ) (time : ℕ) : ℕ := rate * time

/-- Theorem: A machine producing 2 shirts per minute for 6 minutes makes 12 shirts -/
theorem machine_production : shirts_produced 2 6 = 12 := by
  sorry

end machine_production_l740_74063


namespace round_trip_time_l740_74052

/-- Calculates the total time for a round trip between two towns given the speeds and initial travel time. -/
theorem round_trip_time (speed_to_b : ℝ) (speed_to_a : ℝ) (time_to_b : ℝ) : 
  speed_to_b > 0 → speed_to_a > 0 → time_to_b > 0 →
  speed_to_b = 100 → speed_to_a = 150 → time_to_b = 3 →
  time_to_b + (speed_to_b * time_to_b) / speed_to_a = 5 := by
  sorry

#check round_trip_time

end round_trip_time_l740_74052


namespace jason_total_games_l740_74088

/-- The number of football games Jason attended or plans to attend each month from January to July -/
def games_per_month : List Nat := [11, 17, 16, 20, 14, 14, 14]

/-- The total number of games Jason will have attended by the end of July -/
def total_games : Nat := games_per_month.sum

theorem jason_total_games : total_games = 106 := by
  sorry

end jason_total_games_l740_74088


namespace isosceles_triangle_exists_l740_74077

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A coloring of the vertices of a polygon with two colors -/
def Coloring (n : ℕ) := Fin n → Bool

/-- An isosceles triangle in a regular polygon -/
def IsIsoscelesTriangle (p : RegularPolygon n) (v1 v2 v3 : Fin n) : Prop :=
  let d12 := dist (p.vertices v1) (p.vertices v2)
  let d23 := dist (p.vertices v2) (p.vertices v3)
  let d31 := dist (p.vertices v3) (p.vertices v1)
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- The main theorem -/
theorem isosceles_triangle_exists (p : RegularPolygon 13) (c : Coloring 13) :
  ∃ (v1 v2 v3 : Fin 13), c v1 = c v2 ∧ c v2 = c v3 ∧ IsIsoscelesTriangle p v1 v2 v3 := by
  sorry


end isosceles_triangle_exists_l740_74077


namespace age_difference_l740_74058

theorem age_difference (x y z : ℕ) (h1 : z = x - 19) : x + y - (y + z) = 19 := by
  sorry

end age_difference_l740_74058


namespace slope_of_line_l740_74028

theorem slope_of_line (x y : ℝ) (h : x/3 + y/2 = 1) : 
  ∃ m b : ℝ, y = m*x + b ∧ m = -2/3 := by
  sorry

end slope_of_line_l740_74028


namespace imaginary_part_of_one_plus_i_squared_l740_74048

theorem imaginary_part_of_one_plus_i_squared (z : ℂ) : z = 1 + Complex.I → Complex.im (z^2) = 2 := by
  sorry

end imaginary_part_of_one_plus_i_squared_l740_74048


namespace smallest_a_value_l740_74011

def rectangle_vertices : List (ℝ × ℝ) := [(34, 0), (41, 0), (34, 9), (41, 9)]

def line_equation (a : ℝ) (x : ℝ) : ℝ := a * x

def divides_rectangle (a : ℝ) : Prop :=
  ∃ (area1 area2 : ℝ), area1 = 2 * area2 ∧
  area1 + area2 = 63 ∧
  (∃ (x1 y1 x2 y2 : ℝ),
    ((x1, y1) ∈ rectangle_vertices ∨ (x1 ∈ Set.Icc 34 41 ∧ y1 = line_equation a x1)) ∧
    ((x2, y2) ∈ rectangle_vertices ∨ (x2 ∈ Set.Icc 34 41 ∧ y2 = line_equation a x2)) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2))

theorem smallest_a_value :
  ∀ ε > 0, divides_rectangle (0.08 + ε) → divides_rectangle 0.08 ∧ ¬divides_rectangle (0.08 - ε) := by
  sorry

end smallest_a_value_l740_74011


namespace deceased_member_income_l740_74076

theorem deceased_member_income
  (initial_members : ℕ)
  (initial_average : ℚ)
  (final_members : ℕ)
  (final_average : ℚ)
  (h1 : initial_members = 4)
  (h2 : initial_average = 735)
  (h3 : final_members = 3)
  (h4 : final_average = 590)
  : (initial_members : ℚ) * initial_average - (final_members : ℚ) * final_average = 1170 :=
by sorry

end deceased_member_income_l740_74076


namespace larger_number_problem_l740_74038

theorem larger_number_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 4 * y = 6 * x) (h4 : x + y = 50) : y = 30 := by
  sorry

end larger_number_problem_l740_74038


namespace multiples_between_2000_and_3000_l740_74067

def count_multiples (lower upper lcm : ℕ) : ℕ :=
  (upper / lcm) - ((lower - 1) / lcm)

theorem multiples_between_2000_and_3000 : count_multiples 2000 3000 72 = 14 := by
  sorry

end multiples_between_2000_and_3000_l740_74067


namespace hyperbola_eccentricity_l740_74084

/-- Given a hyperbola with the equation y²/a² - x²/b² = l, where a > 0 and b > 0,
    if the point (1, 2) lies on the hyperbola, then its eccentricity e is greater than √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  4/a^2 - 1/b^2 = 1 → ∃ e : ℝ, e > Real.sqrt 5 / 2 ∧ e^2 = (a^2 + b^2)/a^2 := by
  sorry

end hyperbola_eccentricity_l740_74084


namespace matrix_equation_solution_l740_74091

def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, 2, 3; 0, 1, 4; 5, 0, 1]

theorem matrix_equation_solution :
  ∃ (p q r : ℤ), 
    B^3 + p • B^2 + q • B + r • (1 : Matrix (Fin 3) (Fin 3) ℤ) = (0 : Matrix (Fin 3) (Fin 3) ℤ) ∧
    p = -41 ∧ q = -80 ∧ r = -460 := by
  sorry

end matrix_equation_solution_l740_74091


namespace max_baggies_l740_74004

def chocolate_chip_cookies : ℕ := 2
def oatmeal_cookies : ℕ := 16
def cookies_per_bag : ℕ := 3

theorem max_baggies : 
  (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_bag = 6 := by
  sorry

end max_baggies_l740_74004


namespace cranberry_juice_ounces_l740_74046

/-- Given a can of cranberry juice that sells for 84 cents with a cost of 7 cents per ounce,
    prove that the can contains 12 ounces of juice. -/
theorem cranberry_juice_ounces (total_cost : ℕ) (cost_per_ounce : ℕ) (h1 : total_cost = 84) (h2 : cost_per_ounce = 7) :
  total_cost / cost_per_ounce = 12 := by
sorry

end cranberry_juice_ounces_l740_74046


namespace difference_of_squares_2018_ways_l740_74087

theorem difference_of_squares_2018_ways :
  ∃ (n : ℕ), n = 5^(2 * 2018) ∧
  (∃! (ways : Finset (ℕ × ℕ)), ways.card = 2018 ∧
    ∀ (a b : ℕ), (a, b) ∈ ways ↔ n = a^2 - b^2) :=
by sorry

end difference_of_squares_2018_ways_l740_74087


namespace total_fish_is_36_l740_74012

/-- The total number of fish caught by Carla, Kyle, and Tasha -/
def total_fish (carla_fish kyle_fish : ℕ) : ℕ :=
  carla_fish + kyle_fish + kyle_fish

/-- Theorem: Given the conditions, the total number of fish caught is 36 -/
theorem total_fish_is_36 (carla_fish kyle_fish : ℕ) 
  (h1 : carla_fish = 8)
  (h2 : kyle_fish = 14) :
  total_fish carla_fish kyle_fish = 36 := by
  sorry

end total_fish_is_36_l740_74012


namespace simplify_expression_l740_74086

theorem simplify_expression (a : ℝ) : (-a^2)^3 * 3*a = -3*a^7 := by
  sorry

end simplify_expression_l740_74086


namespace sequence_equality_l740_74033

theorem sequence_equality (n : ℕ+) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

end sequence_equality_l740_74033


namespace alpha_value_l740_74034

theorem alpha_value (α : ℝ) :
  (6 * Real.sqrt 3) / (3 * Real.sqrt 2 + 2 * Real.sqrt 3) = 3 * Real.sqrt α - 6 →
  α = 6 :=
by sorry

end alpha_value_l740_74034


namespace center_is_five_l740_74036

/-- Represents a 3x3 array of integers -/
def Array3x3 := Fin 3 → Fin 3 → ℕ

/-- Checks if two positions in the array are adjacent -/
def is_adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Checks if the array contains all numbers from 1 to 9 -/
def contains_all_numbers (a : Array3x3) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, a i j = n + 1

/-- Checks if consecutive numbers are adjacent in the array -/
def consecutive_adjacent (a : Array3x3) : Prop :=
  ∀ n : Fin 8, ∃ i₁ j₁ i₂ j₂ : Fin 3,
    a i₁ j₁ = n + 1 ∧ a i₂ j₂ = n + 2 ∧ is_adjacent (i₁, j₁) (i₂, j₂)

/-- The sum of corner numbers is 20 -/
def corner_sum_20 (a : Array3x3) : Prop :=
  a 0 0 + a 0 2 + a 2 0 + a 2 2 = 20

/-- The product of top-left and bottom-right corner numbers is 9 -/
def corner_product_9 (a : Array3x3) : Prop :=
  a 0 0 * a 2 2 = 9

theorem center_is_five (a : Array3x3)
  (h1 : contains_all_numbers a)
  (h2 : consecutive_adjacent a)
  (h3 : corner_sum_20 a)
  (h4 : corner_product_9 a) :
  a 1 1 = 5 :=
sorry

end center_is_five_l740_74036


namespace yoque_monthly_payment_l740_74025

/-- Calculates the monthly payment for a loan with interest -/
def monthly_payment (principal : ℚ) (months : ℕ) (interest_rate : ℚ) : ℚ :=
  (principal * (1 + interest_rate)) / months

/-- Proves that the monthly payment is $15 given the problem conditions -/
theorem yoque_monthly_payment :
  let principal : ℚ := 150
  let months : ℕ := 11
  let interest_rate : ℚ := 1/10
  monthly_payment principal months interest_rate = 15 := by
sorry

end yoque_monthly_payment_l740_74025


namespace sample_correlation_strength_theorem_l740_74082

/-- Sample correlation coefficient -/
def sample_correlation_coefficient (data : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Strength of linear relationship -/
def linear_relationship_strength (r : ℝ) : ℝ :=
  sorry

theorem sample_correlation_strength_theorem (data : Set (ℝ × ℝ)) :
  let r := sample_correlation_coefficient data
  ∀ s : ℝ, s ∈ Set.Icc (-1 : ℝ) 1 →
    linear_relationship_strength r = linear_relationship_strength (abs r) ∧
    (abs r > abs s → linear_relationship_strength r > linear_relationship_strength s) :=
  sorry

end sample_correlation_strength_theorem_l740_74082


namespace negation_of_existence_inequality_l740_74050

open Set Real

theorem negation_of_existence_inequality :
  (¬ ∃ x : ℝ, x^2 - 5*x - 6 < 0) ↔ (∀ x : ℝ, x^2 - 5*x - 6 ≥ 0) := by
  sorry

end negation_of_existence_inequality_l740_74050


namespace choose_four_from_ten_l740_74070

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by
  sorry

end choose_four_from_ten_l740_74070


namespace inhomogeneous_system_solution_l740_74014

variable (α₁ α₂ α₃ : ℝ)

def X_gen : Fin 4 → ℝ := fun i =>
  match i with
  | 0 => -7 * α₁ + 8 * α₂ - 9 * α₃ + 4
  | 1 => α₁
  | 2 => α₂
  | 3 => α₃

theorem inhomogeneous_system_solution :
  X_gen α₁ α₂ α₃ 0 + 7 * X_gen α₁ α₂ α₃ 1 - 8 * X_gen α₁ α₂ α₃ 2 + 9 * X_gen α₁ α₂ α₃ 3 = 4 := by
  sorry

end inhomogeneous_system_solution_l740_74014


namespace quarter_piles_count_l740_74098

/-- Represents the number of coins in each pile -/
def coins_per_pile : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the number of piles of dimes -/
def dime_piles : ℕ := 6

/-- Represents the number of piles of nickels -/
def nickel_piles : ℕ := 9

/-- Represents the number of piles of pennies -/
def penny_piles : ℕ := 5

/-- Represents the total value of all coins in cents -/
def total_value : ℕ := 2100

/-- Theorem stating that the number of piles of quarters is 4 -/
theorem quarter_piles_count : 
  ∃ (quarter_piles : ℕ), 
    quarter_piles * coins_per_pile * quarter_value + 
    dime_piles * coins_per_pile * dime_value + 
    nickel_piles * coins_per_pile * nickel_value + 
    penny_piles * coins_per_pile * penny_value = total_value ∧ 
    quarter_piles = 4 := by
  sorry

end quarter_piles_count_l740_74098


namespace exists_negative_fraction_abs_lt_four_l740_74013

theorem exists_negative_fraction_abs_lt_four : ∃ (a b : ℤ), b ≠ 0 ∧ (a / b : ℚ) < 0 ∧ |a / b| < 4 := by
  sorry

end exists_negative_fraction_abs_lt_four_l740_74013


namespace basketball_win_rate_l740_74057

theorem basketball_win_rate (games_won_first_half : ℕ) (total_games : ℕ) (desired_win_rate : ℚ) (games_to_win : ℕ) : 
  games_won_first_half = 30 →
  total_games = 80 →
  desired_win_rate = 3/4 →
  games_to_win = 30 →
  (games_won_first_half + games_to_win : ℚ) / total_games = desired_win_rate :=
by
  sorry

#check basketball_win_rate

end basketball_win_rate_l740_74057


namespace number_puzzle_l740_74061

theorem number_puzzle (x : ℚ) : (x / 4) * 12 = 9 → x = 3 := by
  sorry

end number_puzzle_l740_74061


namespace distance_between_points_l740_74066

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 17)
  let p2 : ℝ × ℝ := (10, 3)
  let distance := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  distance = 2 * Real.sqrt 65 := by
  sorry

end distance_between_points_l740_74066


namespace problem_solution_l740_74022

/-- Represents the color of a ball -/
inductive Color
| White
| Black

/-- Represents a pair of drawn balls -/
structure DrawnBalls :=
  (first second : Color)

/-- The sample space of all possible outcomes when drawing two balls without replacement -/
def Ω : Finset DrawnBalls := sorry

/-- Event A: drawing two balls of the same color -/
def A : Set DrawnBalls := {db | db.first = db.second}

/-- Event B: the first ball drawn is white -/
def B : Set DrawnBalls := {db | db.first = Color.White}

/-- Event C: the second ball drawn is white -/
def C : Set DrawnBalls := {db | db.second = Color.White}

/-- Event D: drawing two balls of different colors -/
def D : Set DrawnBalls := {db | db.first ≠ db.second}

/-- The probability measure on the sample space -/
noncomputable def P : Set DrawnBalls → ℝ := sorry

theorem problem_solution :
  (P B = 1/2) ∧
  (P (A ∩ B) = P A * P B) ∧
  (A ∪ D = Set.univ) ∧ (A ∩ D = ∅) :=
sorry

end problem_solution_l740_74022


namespace store_optimal_plan_l740_74062

/-- Represents the types of soccer balls -/
inductive BallType
| A
| B

/-- Represents the store's inventory and pricing -/
structure Store where
  cost_price : BallType → ℕ
  selling_price : BallType → ℕ
  budget : ℕ

/-- Represents the purchase plan -/
structure PurchasePlan where
  num_A : ℕ
  num_B : ℕ

def Store.is_valid (s : Store) : Prop :=
  s.cost_price BallType.A = s.cost_price BallType.B + 40 ∧
  480 / s.cost_price BallType.A = 240 / s.cost_price BallType.B ∧
  s.budget = 4000 ∧
  s.selling_price BallType.A = 100 ∧
  s.selling_price BallType.B = 55

def PurchasePlan.is_valid (p : PurchasePlan) (s : Store) : Prop :=
  p.num_A ≥ p.num_B ∧
  p.num_A * s.cost_price BallType.A + p.num_B * s.cost_price BallType.B ≤ s.budget

def PurchasePlan.profit (p : PurchasePlan) (s : Store) : ℤ :=
  (s.selling_price BallType.A - s.cost_price BallType.A) * p.num_A +
  (s.selling_price BallType.B - s.cost_price BallType.B) * p.num_B

theorem store_optimal_plan (s : Store) (h : s.is_valid) :
  ∃ (p : PurchasePlan), 
    p.is_valid s ∧ 
    s.cost_price BallType.A = 80 ∧ 
    s.cost_price BallType.B = 40 ∧
    p.num_A = 34 ∧ 
    p.num_B = 32 ∧
    ∀ (p' : PurchasePlan), p'.is_valid s → p.profit s ≥ p'.profit s :=
sorry

end store_optimal_plan_l740_74062


namespace line_contains_point_l740_74009

theorem line_contains_point (m : ℚ) : 
  (2 * m - 3 * (-1) = 5 * 3 + 1) ↔ (m = 13 / 2) := by sorry

end line_contains_point_l740_74009


namespace percentage_relationship_l740_74049

theorem percentage_relationship (a b c : ℝ) (h1 : 0.06 * a = 10) (h2 : c = b / a) :
  ∃ p : ℝ, p * b = 6 ∧ p * 100 = 3.6 := by
  sorry

end percentage_relationship_l740_74049


namespace fraction_equality_l740_74059

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (4 * a + b) / (a - 4 * b) = 3) : 
  (a + 4 * b) / (4 * a - b) = 9 / 53 := by
sorry

end fraction_equality_l740_74059


namespace condition_sufficient_not_necessary_l740_74015

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, x > 2 → x^2 - 2*x > 0) ∧
  (∃ x : ℝ, x^2 - 2*x > 0 ∧ ¬(x > 2)) := by
  sorry

end condition_sufficient_not_necessary_l740_74015


namespace plan_a_monthly_fee_is_9_l740_74029

/-- Represents the cost per text message for Plan A -/
def plan_a_cost_per_text : ℚ := 25 / 100

/-- Represents the cost per text message for the other plan -/
def other_plan_cost_per_text : ℚ := 40 / 100

/-- Represents the number of text messages at which both plans cost the same -/
def equal_cost_messages : ℕ := 60

/-- The monthly fee for Plan A makes both plans cost the same at 60 messages -/
theorem plan_a_monthly_fee_is_9 :
  ∃ (monthly_fee : ℚ),
    plan_a_cost_per_text * equal_cost_messages + monthly_fee =
    other_plan_cost_per_text * equal_cost_messages ∧
    monthly_fee = 9 := by
  sorry

end plan_a_monthly_fee_is_9_l740_74029


namespace consecutive_even_numbers_sum_l740_74047

theorem consecutive_even_numbers_sum (x : ℤ) : 
  (x % 2 = 0) →  -- x is even
  (x + (x + 2) + (x + 4) = x + 18) →  -- sum condition
  (x + 4 = 10)  -- largest number is 10
  := by sorry

end consecutive_even_numbers_sum_l740_74047


namespace exists_m_divisible_by_1988_l740_74035

-- Define the function f
def f (x : ℤ) : ℤ := 3 * x + 2

-- Define the k-fold composition of f
def f_comp (k : ℕ) : ℤ → ℤ :=
  match k with
  | 0 => id
  | n + 1 => f ∘ (f_comp n)

-- State the theorem
theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ+, (1988 : ℤ) ∣ (f_comp 100 m.val) :=
sorry

end exists_m_divisible_by_1988_l740_74035


namespace sqrt_2_times_sqrt_24_between_6_and_7_l740_74010

theorem sqrt_2_times_sqrt_24_between_6_and_7 : 6 < Real.sqrt 2 * Real.sqrt 24 ∧ Real.sqrt 2 * Real.sqrt 24 < 7 := by
  sorry

end sqrt_2_times_sqrt_24_between_6_and_7_l740_74010


namespace star_seven_five_l740_74031

/-- The star operation for positive integers -/
def star (a b : ℕ+) : ℚ :=
  (a * b - (a - b)) / (a + b)

/-- Theorem stating that 7 ★ 5 = 11/4 -/
theorem star_seven_five : star 7 5 = 11 / 4 := by
  sorry

end star_seven_five_l740_74031


namespace terese_tuesday_run_l740_74065

-- Define the days Terese runs
inductive RunDay
| monday
| tuesday
| wednesday
| thursday

-- Define a function that returns the distance run on each day
def distance_run (day : RunDay) : Real :=
  match day with
  | RunDay.monday => 4.2
  | RunDay.wednesday => 3.6
  | RunDay.thursday => 4.4
  | RunDay.tuesday => 3.8  -- This is what we want to prove

-- Define the average distance
def average_distance : Real := 4

-- Define the number of days Terese runs
def num_run_days : Nat := 4

-- Theorem statement
theorem terese_tuesday_run :
  (distance_run RunDay.monday +
   distance_run RunDay.tuesday +
   distance_run RunDay.wednesday +
   distance_run RunDay.thursday) / num_run_days = average_distance :=
by
  sorry


end terese_tuesday_run_l740_74065


namespace circus_ticket_cost_l740_74019

/-- The cost of an adult ticket to the circus -/
def adult_ticket_cost : ℕ := 2

/-- The number of people in Mary's group -/
def total_people : ℕ := 4

/-- The number of children in Mary's group -/
def num_children : ℕ := 3

/-- The cost of a child's ticket -/
def child_ticket_cost : ℕ := 1

/-- The total amount Mary paid -/
def total_paid : ℕ := 5

theorem circus_ticket_cost :
  adult_ticket_cost = total_paid - (num_children * child_ticket_cost) :=
by sorry

end circus_ticket_cost_l740_74019


namespace no_positive_integer_solution_l740_74072

theorem no_positive_integer_solution (p : ℕ) (x y : ℕ) (hp : p > 3) (hp_prime : Nat.Prime p) (hx : p ∣ x) :
  ¬(x^2 - 1 = y^p) := by
  sorry

end no_positive_integer_solution_l740_74072


namespace equation_comparison_l740_74007

theorem equation_comparison : 
  (abs (-2))^3 = abs 2^3 ∧ 
  (-2)^2 = 2^2 ∧ 
  (-2)^3 = -(2^3) ∧ 
  (-2)^4 ≠ -(2^4) := by
sorry

end equation_comparison_l740_74007


namespace bisecting_line_min_value_bisecting_line_min_value_achievable_l740_74018

/-- A line that bisects the circumference of a circle --/
structure BisectingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  bisects : ∀ (x y : ℝ), a * x + 2 * b * y - 2 = 0 → 
    (x^2 + y^2 - 4*x - 2*y - 8 = 0 → 
      ∃ (d : ℝ), d > 0 ∧ (x - 2)^2 + (y - 1)^2 = d^2)

/-- The theorem stating the minimum value of 1/a + 2/b --/
theorem bisecting_line_min_value (l : BisectingLine) :
  (1 / l.a + 2 / l.b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

/-- The theorem stating that the minimum value is achievable --/
theorem bisecting_line_min_value_achievable :
  ∃ (l : BisectingLine), 1 / l.a + 2 / l.b = 3 + 2 * Real.sqrt 2 := by
  sorry

end bisecting_line_min_value_bisecting_line_min_value_achievable_l740_74018


namespace product_of_squares_l740_74078

theorem product_of_squares (a b : ℝ) 
  (h1 : a^2 - b^2 = 10) 
  (h2 : a^4 + b^4 = 228) : 
  a * b = 8 := by
sorry

end product_of_squares_l740_74078


namespace number_of_arrangements_l740_74094

/-- Represents a checkout lane with two checkout points -/
structure CheckoutLane :=
  (point1 : Bool)
  (point2 : Bool)

/-- Represents the arrangement of checkout lanes -/
def Arrangement := List CheckoutLane

/-- The total number of checkout lanes -/
def totalLanes : Nat := 6

/-- The number of lanes to be selected -/
def selectedLanes : Nat := 3

/-- Checks if the lanes in an arrangement are non-adjacent -/
def areNonAdjacent (arr : Arrangement) : Bool :=
  sorry

/-- Checks if at least one checkout point is open in each lane -/
def hasOpenPoint (lane : CheckoutLane) : Bool :=
  lane.point1 || lane.point2

/-- Counts the number of valid arrangements -/
def countArrangements : Nat :=
  sorry

/-- The main theorem stating the number of valid arrangements -/
theorem number_of_arrangements :
  countArrangements = 108 :=
sorry

end number_of_arrangements_l740_74094


namespace magazine_selection_count_l740_74073

def total_magazines : ℕ := 8
def literature_magazines : ℕ := 3
def math_magazines : ℕ := 5
def magazines_to_select : ℕ := 3

theorem magazine_selection_count :
  (Nat.choose math_magazines magazines_to_select) +
  (Nat.choose math_magazines (magazines_to_select - 1)) +
  (Nat.choose math_magazines (magazines_to_select - 2)) +
  (if literature_magazines ≥ magazines_to_select then 1 else 0) = 26 := by
  sorry

end magazine_selection_count_l740_74073


namespace bouncing_ball_distance_l740_74020

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The bouncing ball theorem -/
theorem bouncing_ball_distance :
  totalDistance 160 (3/4) 4 = 816.25 := by sorry

end bouncing_ball_distance_l740_74020


namespace expression_evaluation_l740_74095

theorem expression_evaluation :
  let a : ℚ := -1/3
  let expr := (3 - a) / (2*a - 4) / (a + 2 - 5/(a - 2))
  expr = 3/16 := by sorry

end expression_evaluation_l740_74095
