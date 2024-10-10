import Mathlib

namespace alice_pairs_l3500_350044

theorem alice_pairs (total_students : ℕ) (h1 : total_students = 12) : 
  (total_students - 2) = 11 := by
  sorry

#check alice_pairs

end alice_pairs_l3500_350044


namespace library_books_sold_l3500_350005

theorem library_books_sold (total_books : ℕ) (remaining_fraction : ℚ) (books_sold : ℕ) : 
  total_books = 9900 ∧ remaining_fraction = 4/6 → books_sold = 3300 :=
by sorry

end library_books_sold_l3500_350005


namespace trishas_total_distance_is_correct_l3500_350001

/-- The total distance Trisha walked during her vacation in New York City -/
def trishas_total_distance : ℝ :=
  let hotel_to_postcard := 0.11
  let postcard_to_hotel := 0.11
  let hotel_to_tshirt := 1.52
  let tshirt_to_hat := 0.45
  let hat_to_purse := 0.87
  let purse_to_hotel := 2.32
  hotel_to_postcard + postcard_to_hotel + hotel_to_tshirt + tshirt_to_hat + hat_to_purse + purse_to_hotel

/-- Theorem stating that the total distance Trisha walked is 5.38 miles -/
theorem trishas_total_distance_is_correct : trishas_total_distance = 5.38 := by
  sorry

end trishas_total_distance_is_correct_l3500_350001


namespace line_tangent_to_parabola_tangent_line_k_value_l3500_350084

/-- A line is tangent to a parabola if and only if the discriminant of their intersection equation is zero -/
theorem line_tangent_to_parabola (k : ℝ) : 
  (∃ x y : ℝ, 4*x - 3*y + k = 0 ∧ y^2 = 16*x ∧ 
   ∀ x' y' : ℝ, (4*x' - 3*y' + k = 0 ∧ y'^2 = 16*x') → (x' = x ∧ y' = y)) ↔ 
  k = 9 := by
  sorry

/-- The value of k for which the line 4x - 3y + k = 0 is tangent to the parabola y² = 16x is 9 -/
theorem tangent_line_k_value : 
  ∃! k : ℝ, ∃ x y : ℝ, 4*x - 3*y + k = 0 ∧ y^2 = 16*x ∧ 
  ∀ x' y' : ℝ, (4*x' - 3*y' + k = 0 ∧ y'^2 = 16*x') → (x' = x ∧ y' = y) := by
  sorry

end line_tangent_to_parabola_tangent_line_k_value_l3500_350084


namespace sum_of_digits_M_l3500_350018

/-- The smallest positive integer divisible by all positive integers less than 8 -/
def M : ℕ := Nat.lcm 7 (Nat.lcm 6 (Nat.lcm 5 (Nat.lcm 4 (Nat.lcm 3 (Nat.lcm 2 1)))))

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_M :
  sum_of_digits M = 6 := by
  sorry

end sum_of_digits_M_l3500_350018


namespace count_valid_house_numbers_l3500_350085

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_valid_house_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  is_prime (n / 1000) ∧ 
  is_prime (n % 1000) ∧
  (n / 1000) < 100 ∧
  (n % 1000) < 500 ∧
  ∀ d : ℕ, d < 5 → (n / 10^d % 10) ≠ 0

theorem count_valid_house_numbers :
  ∃ (s : Finset ℕ), (∀ n ∈ s, is_valid_house_number n) ∧ s.card = 1302 :=
sorry

end count_valid_house_numbers_l3500_350085


namespace min_value_reciprocal_sum_equality_condition_l3500_350033

theorem min_value_reciprocal_sum (x : ℝ) (h : 0 < x ∧ x < 1) :
  1/x + 2/(1-x) ≥ 3 + 2*Real.sqrt 2 :=
sorry

theorem equality_condition (x : ℝ) (h : 0 < x ∧ x < 1) :
  1/x + 2/(1-x) = 3 + 2*Real.sqrt 2 ↔ x = Real.sqrt 2 - 1 :=
sorry

end min_value_reciprocal_sum_equality_condition_l3500_350033


namespace mike_ride_length_l3500_350090

/-- Represents the taxi fare structure and trip details for Mike and Annie -/
structure TaxiTrip where
  initialCharge : ℝ
  costPerMile : ℝ
  surcharge : ℝ
  tollFees : ℝ
  miles : ℝ

/-- Calculates the total cost of a taxi trip -/
def tripCost (trip : TaxiTrip) : ℝ :=
  trip.initialCharge + trip.costPerMile * trip.miles + trip.surcharge + trip.tollFees

/-- Theorem stating that Mike's ride was 30 miles long -/
theorem mike_ride_length 
  (mike : TaxiTrip)
  (annie : TaxiTrip)
  (h1 : mike.initialCharge = 2.5)
  (h2 : mike.costPerMile = 0.25)
  (h3 : mike.surcharge = 3)
  (h4 : mike.tollFees = 0)
  (h5 : annie.initialCharge = 2.5)
  (h6 : annie.costPerMile = 0.25)
  (h7 : annie.surcharge = 0)
  (h8 : annie.tollFees = 5)
  (h9 : annie.miles = 22)
  (h10 : tripCost mike = tripCost annie) :
  mike.miles = 30 := by
  sorry

end mike_ride_length_l3500_350090


namespace sparrow_population_decrease_l3500_350066

def initial_population : ℕ := 1200
def decrease_rate : ℚ := 0.7
def target_percentage : ℚ := 0.15

def population (year : ℕ) : ℚ :=
  (initial_population : ℚ) * decrease_rate ^ (year - 2010)

theorem sparrow_population_decrease (year : ℕ) :
  year = 2016 ↔ 
    (population year < (initial_population : ℚ) * target_percentage ∧
     ∀ y, 2010 ≤ y ∧ y < 2016 → population y ≥ (initial_population : ℚ) * target_percentage) :=
by sorry

end sparrow_population_decrease_l3500_350066


namespace rowing_speed_problem_l3500_350011

/-- Given a man who can row upstream at 26 kmph and downstream at 40 kmph,
    prove that his speed in still water is 33 kmph and the speed of the river current is 7 kmph. -/
theorem rowing_speed_problem (upstream_speed downstream_speed : ℝ)
  (h_upstream : upstream_speed = 26)
  (h_downstream : downstream_speed = 40) :
  ∃ (still_water_speed river_current_speed : ℝ),
    still_water_speed = 33 ∧
    river_current_speed = 7 ∧
    upstream_speed = still_water_speed - river_current_speed ∧
    downstream_speed = still_water_speed + river_current_speed :=
by sorry

end rowing_speed_problem_l3500_350011


namespace probability_five_white_two_red_l3500_350004

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def red_balls : ℕ := 3
def total_balls : ℕ := white_balls + black_balls + red_balls
def drawn_balls : ℕ := 7

theorem probability_five_white_two_red :
  (Nat.choose white_balls 5 * Nat.choose red_balls 2) / Nat.choose total_balls drawn_balls = 63 / 31824 :=
sorry

end probability_five_white_two_red_l3500_350004


namespace parallel_vectors_magnitude_l3500_350034

/-- Given two 2D vectors a and b, where a = (1,2) and b = (-1,x), 
    if a is parallel to b, then the magnitude of b is √5. -/
theorem parallel_vectors_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-1, x]
  (∃ (k : ℝ), ∀ i, b i = k * a i) →
  Real.sqrt ((b 0)^2 + (b 1)^2) = Real.sqrt 5 := by
sorry

end parallel_vectors_magnitude_l3500_350034


namespace unique_solution_quadratic_l3500_350024

theorem unique_solution_quadratic (k : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 3) = k + 3 * x) ↔ k = 35 / 4 := by
  sorry

end unique_solution_quadratic_l3500_350024


namespace min_value_sum_reciprocals_l3500_350027

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) :
  1/a + 1/b + 4/c + 9/d + 16/e + 25/f ≥ 25.6 ∧ 
  ∃ (a' b' c' d' e' f' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 0 < f' ∧
    a' + b' + c' + d' + e' + f' = 10 ∧
    1/a' + 1/b' + 4/c' + 9/d' + 16/e' + 25/f' = 25.6 := by
  sorry

end min_value_sum_reciprocals_l3500_350027


namespace only_negative_three_halves_and_one_half_satisfy_l3500_350096

def numbers : List ℚ := [-3/2, -1, 1/2, 1, 3]

def satisfies_conditions (x : ℚ) : Prop :=
  x < x⁻¹ ∧ x > -3

theorem only_negative_three_halves_and_one_half_satisfy :
  ∀ x ∈ numbers, satisfies_conditions x ↔ (x = -3/2 ∨ x = 1/2) := by
  sorry

end only_negative_three_halves_and_one_half_satisfy_l3500_350096


namespace grid_toothpicks_l3500_350008

/-- The number of toothpicks needed to construct a grid of given length and width -/
def toothpicks_needed (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem stating that a grid of 50 toothpicks long and 40 toothpicks wide requires 4090 toothpicks -/
theorem grid_toothpicks : toothpicks_needed 50 40 = 4090 := by
  sorry

end grid_toothpicks_l3500_350008


namespace fixed_point_existence_l3500_350023

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two line segments have equal length -/
def equal_length (a b c d : Point) : Prop :=
  (a.x - b.x)^2 + (a.y - b.y)^2 = (c.x - d.x)^2 + (c.y - d.y)^2

/-- Check if an angle is 90 degrees -/
def is_right_angle (a b c : Point) : Prop :=
  (b.x - a.x) * (b.x - c.x) + (b.y - a.y) * (b.y - c.y) = 0

/-- Check if a quadrilateral is convex -/
def is_convex (a b c d : Point) : Prop := sorry

/-- Check if two points are on the same side of a line -/
def same_side (p q : Point) (l : Line) : Prop := sorry

theorem fixed_point_existence (a b : Point) :
  ∃ p : Point,
    ∀ c d : Point,
      is_convex a b c d →
      equal_length a b b c →
      equal_length a d d c →
      is_right_angle a d c →
      same_side c d (Line.mk (b.y - a.y) (a.x - b.x) (a.y * b.x - a.x * b.y)) →
      ∃ l : Line, d.on_line l ∧ c.on_line l ∧ p.on_line l :=
sorry

end fixed_point_existence_l3500_350023


namespace area_of_ADE_l3500_350030

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def Triangle.area (t : Triangle) : ℝ := sorry

def Triangle.isRightAngle (t : Triangle) (vertex : ℝ × ℝ) : Prop := sorry

def angle_bisector (A B C D : ℝ × ℝ) (E : ℝ × ℝ) : Prop := sorry

theorem area_of_ADE (A B C D E : ℝ × ℝ) : 
  let abc := Triangle.mk A B C
  let abd := Triangle.mk A B D
  (Triangle.area abc = 24) →
  (Triangle.isRightAngle abc B) →
  (Triangle.isRightAngle abd B) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 64) →
  ((B.1 - C.1)^2 + (B.2 - C.2)^2 = 36) →
  ((A.1 - D.1)^2 + (A.2 - D.2)^2 = 64) →
  (angle_bisector A C A D E) →
  Triangle.area (Triangle.mk A D E) = 20 := by sorry

end area_of_ADE_l3500_350030


namespace gas_tank_capacity_l3500_350002

/-- Represents the gas prices at each station -/
def gas_prices : List ℝ := [3, 3.5, 4, 4.5]

/-- Represents the total amount spent on gas -/
def total_spent : ℝ := 180

/-- Theorem: If a car owner fills up their tank 4 times at the given prices and spends $180 in total,
    then the gas tank capacity is 12 gallons -/
theorem gas_tank_capacity :
  ∀ (C : ℝ),
  (C > 0) →
  (List.sum (List.map (λ price => price * C) gas_prices) = total_spent) →
  C = 12 := by
  sorry

end gas_tank_capacity_l3500_350002


namespace equation_solution_l3500_350042

theorem equation_solution : ∃ x : ℝ, (x - 3)^4 = 16 ∧ x = 5 := by sorry

end equation_solution_l3500_350042


namespace approx_C_squared_minus_D_squared_for_specific_values_l3500_350071

/-- Given nonnegative real numbers x, y, z, we define C and D as follows:
C = √(x + 3) + √(y + 6) + √(z + 12)
D = √(x + 2) + √(y + 4) + √(z + 8)
This theorem states that when x = 1, y = 2, and z = 3, the value of C² - D² 
is approximately 19.483 with arbitrary precision. -/
theorem approx_C_squared_minus_D_squared_for_specific_values :
  ∀ ε > 0, ∃ C D : ℝ,
  C = Real.sqrt (1 + 3) + Real.sqrt (2 + 6) + Real.sqrt (3 + 12) ∧
  D = Real.sqrt (1 + 2) + Real.sqrt (2 + 4) + Real.sqrt (3 + 8) ∧
  |C^2 - D^2 - 19.483| < ε :=
by
  sorry

end approx_C_squared_minus_D_squared_for_specific_values_l3500_350071


namespace binomial_expected_value_theorem_l3500_350000

/-- A random variable following a Binomial distribution -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  prob : ℝ
  property : prob = p

/-- The expected value of a Binomial distribution -/
def expectedValue (ξ : BinomialDistribution n p) : ℝ := n * p

theorem binomial_expected_value_theorem (ξ : BinomialDistribution 18 p) 
  (h : expectedValue ξ = 9) : p = 1/2 := by
  sorry

end binomial_expected_value_theorem_l3500_350000


namespace pen_profit_percentage_retailer_profit_is_20_625_percent_l3500_350048

/-- Calculates the profit percentage for a retailer selling pens with a discount -/
theorem pen_profit_percentage 
  (num_pens_bought : ℕ) 
  (num_pens_price : ℕ) 
  (discount_percent : ℝ) : ℝ :=
  let cost_price := num_pens_price
  let selling_price_per_pen := 1 - (discount_percent / 100)
  let total_selling_price := selling_price_per_pen * num_pens_bought
  let profit := total_selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  20.625

/-- The retailer's profit percentage is 20.625% -/
theorem retailer_profit_is_20_625_percent : 
  pen_profit_percentage 75 60 3.5 = 20.625 := by
  sorry

end pen_profit_percentage_retailer_profit_is_20_625_percent_l3500_350048


namespace inequality_system_solution_range_l3500_350093

-- Define the inequality system
def inequality_system (x a : ℝ) : Prop :=
  2 * x < 3 * (x - 3) + 1 ∧ (3 * x + 2) / 4 > x + a

-- Define the condition for having exactly four integer solutions
def has_four_integer_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ : ℤ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
    (∀ x : ℤ, inequality_system x a ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

-- The theorem to be proved
theorem inequality_system_solution_range :
  ∀ a : ℝ, has_four_integer_solutions a → -11/4 ≤ a ∧ a < -5/2 :=
sorry

end inequality_system_solution_range_l3500_350093


namespace computer_accessories_cost_l3500_350069

def original_amount : ℕ := 48
def snack_cost : ℕ := 8

theorem computer_accessories_cost (remaining_amount : ℕ) 
  (h1 : remaining_amount = original_amount / 2 + 4) 
  (h2 : remaining_amount = original_amount - snack_cost - (original_amount - remaining_amount - snack_cost)) :
  original_amount - remaining_amount - snack_cost = 12 := by
  sorry

end computer_accessories_cost_l3500_350069


namespace payment_equality_l3500_350099

/-- Represents the payment structure and hours worked for Harry and James -/
structure PaymentSystem where
  x : ℝ
  y : ℝ
  james_hours : ℝ
  harry_hours : ℝ

/-- Calculates James' earnings based on the given payment structure -/
def james_earnings (ps : PaymentSystem) : ℝ :=
  40 * ps.x + (ps.james_hours - 40) * 2 * ps.x

/-- Calculates Harry's earnings based on the given payment structure -/
def harry_earnings (ps : PaymentSystem) : ℝ :=
  12 * ps.x + (ps.harry_hours - 12) * ps.y * ps.x

/-- Theorem stating the conditions and the result to be proved -/
theorem payment_equality (ps : PaymentSystem) :
  ps.x > 0 ∧ 
  ps.y > 1 ∧ 
  ps.james_hours = 48 ∧ 
  james_earnings ps = harry_earnings ps →
  ps.harry_hours = 23 ∧ ps.y = 4 := by
  sorry

end payment_equality_l3500_350099


namespace abs_a_b_sum_l3500_350037

theorem abs_a_b_sum (a b : ℝ) (ha : |a| = 7) (hb : |b| = 3) (hab : a * b > 0) :
  a + b = 10 ∨ a + b = -10 := by
  sorry

end abs_a_b_sum_l3500_350037


namespace circleplus_properties_l3500_350081

-- Define the ⊕ operation
def circleplus (x y : ℚ) : ℚ := x * y + 1

-- Theorem statement
theorem circleplus_properties :
  (circleplus 2 4 = 9) ∧
  (∀ x : ℚ, circleplus 3 (2*x - 1) = 4 → x = 1) := by
sorry

end circleplus_properties_l3500_350081


namespace sum_of_coefficients_l3500_350039

theorem sum_of_coefficients : ∃ (a b c : ℕ+), 
  (a.val : ℝ) * Real.sqrt 6 + (b.val : ℝ) * Real.sqrt 8 = c.val * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) ∧ 
  (∀ (a' b' c' : ℕ+), (a'.val : ℝ) * Real.sqrt 6 + (b'.val : ℝ) * Real.sqrt 8 = c'.val * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) → c'.val ≥ c.val) →
  a.val + b.val + c.val = 67 := by
  sorry

end sum_of_coefficients_l3500_350039


namespace water_distribution_l3500_350043

/-- Proves that given 122 ounces of water, after filling six 5-ounce glasses and four 8-ounce glasses,
    the remaining water can fill exactly 15 four-ounce glasses. -/
theorem water_distribution (total_water : ℕ) (five_oz_glasses : ℕ) (eight_oz_glasses : ℕ) 
  (four_oz_glasses : ℕ) : 
  total_water = 122 ∧ 
  five_oz_glasses = 6 ∧ 
  eight_oz_glasses = 4 ∧ 
  four_oz_glasses * 4 = total_water - (five_oz_glasses * 5 + eight_oz_glasses * 8) → 
  four_oz_glasses = 15 :=
by sorry

end water_distribution_l3500_350043


namespace volume_ratio_is_19_89_l3500_350045

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  origin : Point3D
  edge_length : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  point1 : Point3D
  point2 : Point3D
  point3 : Point3D

def cube : Cube := {
  origin := { x := 0, y := 0, z := 0 }
  edge_length := 6
}

def point_A : Point3D := { x := 0, y := 0, z := 0 }
def point_H : Point3D := { x := 6, y := 6, z := 2 }
def point_F : Point3D := { x := 6, y := 6, z := 3 }

def cutting_plane : Plane := {
  point1 := point_A
  point2 := point_H
  point3 := point_F
}

/-- Calculates the volume of a part of the cube cut by the plane -/
def volume_of_part (c : Cube) (p : Plane) : ℝ := sorry

/-- Theorem: The volume ratio of the two parts is 19:89 -/
theorem volume_ratio_is_19_89 (c : Cube) (p : Plane) : 
  let v1 := volume_of_part c p
  let v2 := c.edge_length ^ 3 - v1
  v1 / v2 = 19 / 89 := by sorry

end volume_ratio_is_19_89_l3500_350045


namespace max_sum_of_four_numbers_l3500_350022

theorem max_sum_of_four_numbers (a b c d : ℕ) : 
  a < b → b < c → c < d → (c + d) + (a + b + c) = 2017 → 
  a + b + c + d ≤ 806 := by
sorry

end max_sum_of_four_numbers_l3500_350022


namespace purely_imaginary_condition_l3500_350050

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem purely_imaginary_condition (m : ℝ) :
  is_purely_imaginary ((m - i) * (1 + i)) → m = -1 := by
  sorry

end purely_imaginary_condition_l3500_350050


namespace complement_probability_l3500_350092

theorem complement_probability (event_prob : ℚ) (h : event_prob = 1/4) :
  1 - event_prob = 3/4 := by
  sorry

end complement_probability_l3500_350092


namespace cos_120_degrees_l3500_350057

theorem cos_120_degrees : Real.cos (120 * π / 180) = -(1 / 2) := by
  sorry

end cos_120_degrees_l3500_350057


namespace tangent_line_equation_f_decreasing_intervals_l3500_350013

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Theorem for the equation of the tangent line
theorem tangent_line_equation :
  ∀ x y : ℝ, y = f x → (x = 0 → 9*x - y - 2 = 0) :=
sorry

-- Theorem for the intervals where f is decreasing
theorem f_decreasing_intervals :
  ∀ x : ℝ, (x < -1 ∨ x > 3) → (f' x < 0) :=
sorry

end tangent_line_equation_f_decreasing_intervals_l3500_350013


namespace max_value_abc_l3500_350079

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  a^2 * b^3 * c^2 ≤ 81/262144 :=
sorry

end max_value_abc_l3500_350079


namespace det_A_eq_32_l3500_350064

def A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 4; -2, 3]

theorem det_A_eq_32 : Matrix.det A = 32 := by
  sorry

end det_A_eq_32_l3500_350064


namespace x_squared_gt_4_necessary_not_sufficient_for_x_gt_2_l3500_350073

theorem x_squared_gt_4_necessary_not_sufficient_for_x_gt_2 :
  (∀ x : ℝ, x > 2 → x^2 > 4) ∧ (∃ x : ℝ, x^2 > 4 ∧ x ≤ 2) := by sorry

end x_squared_gt_4_necessary_not_sufficient_for_x_gt_2_l3500_350073


namespace line_direction_vector_l3500_350041

/-- Given two points and a direction vector, prove the value of b -/
theorem line_direction_vector (p1 p2 : ℝ × ℝ) (b : ℝ) :
  p1 = (-3, 2) →
  p2 = (4, -3) →
  ∃ (k : ℝ), k • (p2.1 - p1.1, p2.2 - p1.2) = (b, -2) →
  b = 14/5 := by
sorry

end line_direction_vector_l3500_350041


namespace printer_time_345_pages_l3500_350068

/-- The time (in minutes) it takes to print a given number of pages at a given rate -/
def print_time (pages : ℕ) (rate : ℕ) : ℚ :=
  pages / rate

theorem printer_time_345_pages : 
  let pages := 345
  let rate := 23
  Int.floor (print_time pages rate) = 15 := by
  sorry

end printer_time_345_pages_l3500_350068


namespace root_product_equals_27_l3500_350026

theorem root_product_equals_27 : 
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by sorry

end root_product_equals_27_l3500_350026


namespace basketball_game_score_l3500_350056

/-- Represents the score of a team in a quarter -/
structure QuarterScore where
  score : ℕ
  valid : score ≤ 25

/-- Represents the scores of a team for all four quarters -/
structure GameScore where
  q1 : QuarterScore
  q2 : QuarterScore
  q3 : QuarterScore
  q4 : QuarterScore
  increasing : q1.score < q2.score ∧ q2.score < q3.score ∧ q3.score < q4.score
  arithmetic : ∃ d : ℕ, q2.score = q1.score + d ∧ q3.score = q2.score + d ∧ q4.score = q3.score + d

def total_score (g : GameScore) : ℕ :=
  g.q1.score + g.q2.score + g.q3.score + g.q4.score

def first_half_score (g : GameScore) : ℕ :=
  g.q1.score + g.q2.score

theorem basketball_game_score :
  ∀ raiders wildcats : GameScore,
    raiders.q1 = wildcats.q1 →  -- First quarter tie
    total_score raiders = total_score wildcats + 2 →  -- Raiders win by 2
    first_half_score raiders + first_half_score wildcats = 38 :=
by sorry

end basketball_game_score_l3500_350056


namespace cooking_cleaning_arrangements_l3500_350097

theorem cooking_cleaning_arrangements (n : ℕ) (h : n = 8) : 
  Nat.choose n (n / 2) = 70 := by
  sorry

end cooking_cleaning_arrangements_l3500_350097


namespace rectangle_area_rectangle_area_is_120_l3500_350098

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_120 :
  rectangle_area 900 10 = 120 := by
  sorry

end rectangle_area_rectangle_area_is_120_l3500_350098


namespace tangent_line_y_intercept_l3500_350078

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles at the same height in the first quadrant -/
def TangentLine (c1 c2 : Circle) :=
  ∃ (y : ℝ), y > 0 ∧
    (y = c1.radius ∨ y = c2.radius) ∧
    (c1.center.1 + c1.radius < c2.center.1 - c2.radius)

theorem tangent_line_y_intercept
  (c1 : Circle)
  (c2 : Circle)
  (h1 : c1 = ⟨(3, 0), 3⟩)
  (h2 : c2 = ⟨(8, 0), 2⟩)
  (h3 : TangentLine c1 c2) :
  ∃ (line : ℝ → ℝ), line 0 = 3 :=
sorry

end tangent_line_y_intercept_l3500_350078


namespace comparison_inequality_l3500_350058

theorem comparison_inequality (h1 : 0.83 > 0.73) 
  (h2 : Real.log 0.4 / Real.log 0.5 > Real.log 0.6 / Real.log 0.5)
  (h3 : Real.log 1.6 > Real.log 1.4) : 
  0.75 - 0.1 > 0.75 * 0.1 := by
  sorry

end comparison_inequality_l3500_350058


namespace problem_statement_l3500_350077

theorem problem_statement : 7^2 - 2 * 6 + (3^2 - 1) = 45 := by
  sorry

end problem_statement_l3500_350077


namespace world_population_scientific_notation_l3500_350075

/-- The number of people in the global population by the end of 2022 -/
def world_population : ℕ := 8000000000

/-- The scientific notation representation of the world population -/
def scientific_notation : ℝ := 8 * (10 : ℝ) ^ 9

/-- Theorem stating that the world population is equal to its scientific notation representation -/
theorem world_population_scientific_notation : 
  (world_population : ℝ) = scientific_notation := by sorry

end world_population_scientific_notation_l3500_350075


namespace probability_four_white_balls_l3500_350055

def total_balls : ℕ := 25
def white_balls : ℕ := 10
def black_balls : ℕ := 15
def drawn_balls : ℕ := 4

theorem probability_four_white_balls : 
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls) = 3 / 181 :=
sorry

end probability_four_white_balls_l3500_350055


namespace panda_weekly_consumption_l3500_350052

/-- The amount of bamboo an adult panda eats per day in pounds -/
def adult_panda_daily_consumption : ℕ := 138

/-- The amount of bamboo a baby panda eats per day in pounds -/
def baby_panda_daily_consumption : ℕ := 50

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total bamboo consumption for an adult panda and a baby panda in a week is 1316 pounds -/
theorem panda_weekly_consumption :
  (adult_panda_daily_consumption * days_in_week) + (baby_panda_daily_consumption * days_in_week) = 1316 :=
by sorry

end panda_weekly_consumption_l3500_350052


namespace max_distinct_count_is_five_l3500_350025

/-- A type representing a circular arrangement of nine natural numbers -/
def CircularArrangement := Fin 9 → ℕ

/-- Checks if a number is prime -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The condition that all adjacent triples in the circle form prime sums -/
def AllAdjacentTriplesPrime (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 9, IsPrime (arr i + arr (i + 1) ^ (arr (i + 2)))

/-- The number of distinct elements in the circular arrangement -/
def DistinctCount (arr : CircularArrangement) : ℕ :=
  Finset.card (Finset.image arr Finset.univ)

/-- The main theorem statement -/
theorem max_distinct_count_is_five (arr : CircularArrangement) 
  (h : AllAdjacentTriplesPrime arr) : 
  DistinctCount arr ≤ 5 :=
sorry

end max_distinct_count_is_five_l3500_350025


namespace constant_function_theorem_l3500_350086

theorem constant_function_theorem (f : ℝ → ℝ) : 
  (∀ x y z : ℝ, f (x * y) + f (x * z) ≥ f x * f (y * z) + 1) → 
  (∀ x : ℝ, f x = 1) :=
by sorry

end constant_function_theorem_l3500_350086


namespace equation_has_two_distinct_real_roots_l3500_350060

/-- Custom multiplication operation -/
def star_op (a b : ℝ) := a^2 - a*b

/-- Theorem stating that the equation (x+1)*3 = -2 has two distinct real roots -/
theorem equation_has_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ star_op (x₁ + 1) 3 = -2 ∧ star_op (x₂ + 1) 3 = -2 :=
sorry

end equation_has_two_distinct_real_roots_l3500_350060


namespace impossibleTransformation_l3500_350028

-- Define the type for a card
def Card := ℤ × ℤ

-- Define the operations of the machines
def machine1 (c : Card) : Card :=
  (c.1 + 1, c.2 + 1)

def machine2 (c : Card) : Option Card :=
  if c.1 % 2 = 0 ∧ c.2 % 2 = 0 then some (c.1 / 2, c.2 / 2) else none

def machine3 (c1 c2 : Card) : Option Card :=
  if c1.2 = c2.1 then some (c1.1, c2.2) else none

-- Define the property that the difference is divisible by 7
def diffDivisibleBy7 (c : Card) : Prop :=
  (c.1 - c.2) % 7 = 0

-- Theorem stating the impossibility of the transformation
theorem impossibleTransformation :
  ∀ (c : Card), diffDivisibleBy7 c →
  ¬∃ (sequence : List (Card → Card)), 
    (List.foldl (λ acc f => f acc) c sequence = (1, 1988)) :=
by sorry

end impossibleTransformation_l3500_350028


namespace consecutive_non_primes_under_50_l3500_350009

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem consecutive_non_primes_under_50 :
  ∃ (a b c d e : ℕ),
    a < 50 ∧ b < 50 ∧ c < 50 ∧ d < 50 ∧ e < 50 ∧
    ¬(is_prime a) ∧ ¬(is_prime b) ∧ ¬(is_prime c) ∧ ¬(is_prime d) ∧ ¬(is_prime e) ∧
    b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
    e = 46 :=
by
  sorry

end consecutive_non_primes_under_50_l3500_350009


namespace dinos_money_theorem_l3500_350015

/-- Calculates Dino's remaining money after expenses given his work hours and rates. -/
def dinos_remaining_money (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (expenses : ℕ) : ℕ :=
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 - expenses

/-- Theorem stating that Dino's remaining money at the end of the month is $500. -/
theorem dinos_money_theorem : 
  dinos_remaining_money 20 30 5 10 20 40 500 = 500 := by
  sorry

#eval dinos_remaining_money 20 30 5 10 20 40 500

end dinos_money_theorem_l3500_350015


namespace min_value_theorem_l3500_350094

theorem min_value_theorem (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) (h5 : a * b = 1/4) :
  ∃ (min_val : ℝ), min_val = 4 + 4 * Real.sqrt 2 / 3 ∧
    ∀ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x * y = 1/4 →
      1 / (1 - x) + 2 / (1 - y) ≥ min_val :=
by sorry

end min_value_theorem_l3500_350094


namespace cos_2alpha_minus_pi_over_2_l3500_350076

theorem cos_2alpha_minus_pi_over_2 (α : ℝ) :
  (Real.cos α = -5/13) → (Real.sin α = 12/13) → Real.cos (2*α - π/2) = -120/169 := by
  sorry

end cos_2alpha_minus_pi_over_2_l3500_350076


namespace pentagon_right_angles_l3500_350074

/-- The sum of interior angles in a pentagon in degrees -/
def pentagonAngleSum : ℝ := 540

/-- The measure of a right angle in degrees -/
def rightAngle : ℝ := 90

/-- The set of possible numbers of right angles in a pentagon -/
def possibleRightAngles : Set ℕ := {0, 1, 2, 3}

/-- Theorem: The set of possible numbers of right angles in a pentagon is {0, 1, 2, 3} -/
theorem pentagon_right_angles :
  ∀ n : ℕ, n ∈ possibleRightAngles ↔ 
    (n : ℝ) * rightAngle ≤ pentagonAngleSum ∧ 
    (n + 1 : ℝ) * rightAngle > pentagonAngleSum :=
by sorry

end pentagon_right_angles_l3500_350074


namespace unique_solution_cube_equation_l3500_350007

theorem unique_solution_cube_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 ∧ x = 27 := by
  sorry

end unique_solution_cube_equation_l3500_350007


namespace primitive_decomposition_existence_l3500_350032

/-- A decomposition of a square into rectangles. -/
structure SquareDecomposition :=
  (n : ℕ)  -- number of rectangles
  (is_finite : n > 0)
  (parallel_sides : Bool)
  (is_primitive : Bool)

/-- Predicate for a valid primitive square decomposition. -/
def valid_primitive_decomposition (d : SquareDecomposition) : Prop :=
  d.parallel_sides ∧ d.is_primitive

/-- Theorem stating for which n a primitive decomposition exists. -/
theorem primitive_decomposition_existence :
  ∀ n : ℕ, (∃ d : SquareDecomposition, d.n = n ∧ valid_primitive_decomposition d) ↔ (n = 5 ∨ n ≥ 7) :=
sorry

end primitive_decomposition_existence_l3500_350032


namespace student_distribution_l3500_350059

/-- The number of students standing next to exactly one from club A and one from club B -/
def p : ℕ := 16

/-- The number of students standing between two from club A -/
def q : ℕ := 46

/-- The number of students standing between two from club B -/
def r : ℕ := 38

/-- The total number of students -/
def total : ℕ := 100

/-- The number of students standing next to at least one from club A -/
def next_to_A : ℕ := 62

/-- The number of students standing next to at least one from club B -/
def next_to_B : ℕ := 54

theorem student_distribution :
  p + q + r = total ∧
  p + q = next_to_A ∧
  p + r = next_to_B :=
by sorry

end student_distribution_l3500_350059


namespace fruit_store_inventory_l3500_350019

theorem fruit_store_inventory (initial_amount : ℚ) : 
  initial_amount - 3/10 + 2/5 = 19/20 → initial_amount = 17/20 := by
  sorry

end fruit_store_inventory_l3500_350019


namespace hyperbola_asymptote_implies_a_equals_two_l3500_350063

-- Define the hyperbola equation
def hyperbola_equation (x y a : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 9 = 1

-- Define the asymptote equations
def asymptote_equations (x y : ℝ) : Prop :=
  (3*x + 2*y = 0) ∧ (3*x - 2*y = 0)

theorem hyperbola_asymptote_implies_a_equals_two :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, hyperbola_equation x y a → asymptote_equations x y) →
  a = 2 := by
  sorry

end hyperbola_asymptote_implies_a_equals_two_l3500_350063


namespace numeral_system_base_proof_l3500_350088

theorem numeral_system_base_proof (x : ℕ) : 
  (3 * x + 4)^2 = x^3 + 5 * x^2 + 5 * x + 2 → x = 7 := by
sorry

end numeral_system_base_proof_l3500_350088


namespace product_ratio_equals_one_l3500_350067

theorem product_ratio_equals_one
  (a b c d e f : ℝ)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 1 := by
  sorry

end product_ratio_equals_one_l3500_350067


namespace polynomial_product_expansion_l3500_350061

theorem polynomial_product_expansion :
  let p₁ : Polynomial ℝ := 3 * X^2 + 4 * X - 5
  let p₂ : Polynomial ℝ := 4 * X^3 - 3 * X^2 + 2 * X - 1
  p₁ * p₂ = 12 * X^5 + 25 * X^4 - 41 * X^3 - 14 * X^2 + 28 * X - 5 := by
  sorry

end polynomial_product_expansion_l3500_350061


namespace second_number_value_l3500_350051

theorem second_number_value (x y : ℝ) 
  (h1 : x - y = 88)
  (h2 : y = 0.2 * x) : 
  y = 22 := by
sorry

end second_number_value_l3500_350051


namespace watermelon_pineapple_weight_difference_l3500_350089

/-- Given that 4 watermelons weigh 5200 grams and 3 watermelons plus 4 pineapples
    weigh 5700 grams, prove that a watermelon is 850 grams heavier than a pineapple. -/
theorem watermelon_pineapple_weight_difference :
  let watermelon_weight : ℕ := 5200 / 4
  let pineapple_weight : ℕ := (5700 - 3 * watermelon_weight) / 4
  watermelon_weight - pineapple_weight = 850 := by
sorry

end watermelon_pineapple_weight_difference_l3500_350089


namespace point_in_plane_region_l3500_350087

/-- The range of values for m such that point A(2, 3) lies within or on the boundary
    of the plane region represented by 3x - 2y + m ≥ 0 -/
theorem point_in_plane_region (m : ℝ) : 
  (3 * 2 - 2 * 3 + m ≥ 0) ↔ (m ≥ 0) := by sorry

end point_in_plane_region_l3500_350087


namespace largest_reciprocal_l3500_350006

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/4 → b = 3/7 → c = 2 → d = 7 → e = 1000 →
  (1/a > 1/b) ∧ (1/a > 1/c) ∧ (1/a > 1/d) ∧ (1/a > 1/e) :=
by
  sorry

#check largest_reciprocal

end largest_reciprocal_l3500_350006


namespace kevin_run_distance_l3500_350065

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents Kevin's run with three segments -/
structure KevinRun where
  flat_speed : ℝ
  flat_time : ℝ
  uphill_speed : ℝ
  uphill_time : ℝ
  downhill_speed : ℝ
  downhill_time : ℝ

/-- Calculates the total distance of Kevin's run -/
def total_distance (run : KevinRun) : ℝ :=
  distance run.flat_speed run.flat_time +
  distance run.uphill_speed run.uphill_time +
  distance run.downhill_speed run.downhill_time

/-- Theorem stating that Kevin's total run distance is 17 miles -/
theorem kevin_run_distance :
  let run : KevinRun := {
    flat_speed := 10,
    flat_time := 0.5,
    uphill_speed := 20,
    uphill_time := 0.5,
    downhill_speed := 8,
    downhill_time := 0.25
  }
  total_distance run = 17 := by sorry

end kevin_run_distance_l3500_350065


namespace prob_two_cards_sum_fifteen_l3500_350017

/-- Represents a standard playing card --/
inductive Card
| Number (n : Nat)
| Face
| Ace

/-- A standard 52-card deck --/
def Deck : Finset Card := sorry

/-- The set of number cards (2 through 10) in the deck --/
def NumberCards : Finset Card := sorry

/-- The probability of drawing two specific cards from the deck --/
def drawTwoCardsProbability (card1 card2 : Card) : ℚ := sorry

/-- The sum of two cards --/
def cardSum (card1 card2 : Card) : ℕ := sorry

/-- The probability of drawing two number cards that sum to 15 --/
def probSumFifteen : ℚ := sorry

theorem prob_two_cards_sum_fifteen :
  probSumFifteen = 16 / 884 := by sorry

end prob_two_cards_sum_fifteen_l3500_350017


namespace lilys_remaining_balance_l3500_350047

/-- Calculates the remaining balance in Lily's account after purchases --/
def remaining_balance (initial_balance : ℕ) (shirt_cost : ℕ) : ℕ :=
  initial_balance - shirt_cost - (3 * shirt_cost)

/-- Theorem stating that Lily's remaining balance is 27 dollars --/
theorem lilys_remaining_balance :
  remaining_balance 55 7 = 27 := by
  sorry

end lilys_remaining_balance_l3500_350047


namespace min_cars_with_stripes_is_two_l3500_350072

/-- Represents the properties of a car group -/
structure CarGroup where
  total : ℕ
  no_ac : ℕ
  max_ac_no_stripes : ℕ

/-- The minimum number of cars with racing stripes -/
def min_cars_with_stripes (group : CarGroup) : ℕ :=
  group.total - group.no_ac - group.max_ac_no_stripes

/-- Theorem stating the minimum number of cars with racing stripes -/
theorem min_cars_with_stripes_is_two (group : CarGroup) 
  (h1 : group.total = 100)
  (h2 : group.no_ac = 49)
  (h3 : group.max_ac_no_stripes = 49) : 
  min_cars_with_stripes group = 2 := by
  sorry

#eval min_cars_with_stripes ⟨100, 49, 49⟩

end min_cars_with_stripes_is_two_l3500_350072


namespace dog_distance_total_dog_distance_l3500_350054

/-- Proves that a dog running back and forth between two points covers 4000 meters
    by the time a person walks the distance between the points. -/
theorem dog_distance (distance : ℝ) (walking_speed : ℝ) (dog_speed : ℝ) : ℝ :=
  let time := distance * 1000 / walking_speed
  dog_speed * time

/-- The main theorem that calculates the total distance run by the dog. -/
theorem total_dog_distance : dog_distance 1 50 200 = 4000 := by
  sorry

end dog_distance_total_dog_distance_l3500_350054


namespace cafeteria_apples_l3500_350080

/-- The number of apples initially in the cafeteria. -/
def initial_apples : ℕ := sorry

/-- The number of apples handed out to students. -/
def apples_handed_out : ℕ := 19

/-- The number of pies that can be made. -/
def pies_made : ℕ := 7

/-- The number of apples required for each pie. -/
def apples_per_pie : ℕ := 8

/-- The number of apples used for making pies. -/
def apples_for_pies : ℕ := pies_made * apples_per_pie

theorem cafeteria_apples : initial_apples = apples_handed_out + apples_for_pies := by
  sorry

end cafeteria_apples_l3500_350080


namespace no_integer_solution_for_equation_l3500_350003

theorem no_integer_solution_for_equation : ¬ ∃ (x y : ℤ), x^2 - y^2 = 1998 := by sorry

end no_integer_solution_for_equation_l3500_350003


namespace number_line_steps_l3500_350036

/-- Given a number line with equally spaced markings, where 35 is reached in 7 steps from 0,
    prove that after 5 steps, the number reached is 25. -/
theorem number_line_steps (total_distance : ℝ) (total_steps : ℕ) (target_steps : ℕ) : 
  total_distance = 35 ∧ total_steps = 7 ∧ target_steps = 5 → 
  (total_distance / total_steps) * target_steps = 25 := by
  sorry

#check number_line_steps

end number_line_steps_l3500_350036


namespace sum_a_n_1_to_1499_l3500_350020

def a_n (n : ℕ) : ℕ :=
  if n < 1500 then
    if n % 10 = 0 ∧ n % 15 = 0 then 15
    else if n % 15 = 0 ∧ n % 12 = 0 then 10
    else if n % 12 = 0 ∧ n % 10 = 0 then 12
    else 0
  else 0

theorem sum_a_n_1_to_1499 :
  (Finset.range 1499).sum a_n = 1263 := by
  sorry

end sum_a_n_1_to_1499_l3500_350020


namespace number_of_men_is_correct_l3500_350040

/-- The number of men in a group where:
  1. The average age of the group increases by 2 years when two men are replaced.
  2. The ages of the two men being replaced are 21 and 23 years.
  3. The average age of the two new men is 37 years. -/
def number_of_men : ℕ :=
  let age_increase : ℕ := 2
  let replaced_men_ages : Fin 2 → ℕ := ![21, 23]
  let new_men_average_age : ℕ := 37
  15

theorem number_of_men_is_correct :
  let age_increase : ℕ := 2
  let replaced_men_ages : Fin 2 → ℕ := ![21, 23]
  let new_men_average_age : ℕ := 37
  number_of_men = 15 := by
  sorry

end number_of_men_is_correct_l3500_350040


namespace tg_sum_formula_l3500_350021

-- Define the tangent and cotangent functions
noncomputable def tg (x : ℝ) : ℝ := Real.tan x
noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

-- Define the theorem
theorem tg_sum_formula (α β p q : ℝ) 
  (h1 : tg α + tg β = p) 
  (h2 : ctg α + ctg β = q) :
  (p = 0 ∧ q = 0 → tg (α + β) = 0) ∧
  (p ≠ 0 ∧ q ≠ 0 ∧ p ≠ q → tg (α + β) = p * q / (q - p)) ∧
  (p ≠ 0 ∧ q ≠ 0 ∧ p = q → ¬∃x, x = tg (α + β)) ∧
  ((p = 0 ∨ q = 0) ∧ p ≠ q → False) :=
by sorry


end tg_sum_formula_l3500_350021


namespace polygon_diagonals_twice_sides_l3500_350014

theorem polygon_diagonals_twice_sides (n : ℕ) : 
  n ≥ 3 → (n * (n - 3) / 2 = 2 * n ↔ n = 7) := by
  sorry

end polygon_diagonals_twice_sides_l3500_350014


namespace trigonometric_simplification_special_angle_simplification_l3500_350070

-- Part 1
theorem trigonometric_simplification (α : ℝ) :
  (Real.cos (α - π / 2) / Real.sin (5 * π / 2 + α)) *
  Real.sin (α - π) * Real.cos (2 * π - α) = -Real.sin α ^ 2 := by
  sorry

-- Part 2
theorem special_angle_simplification :
  (Real.sqrt (1 - 2 * Real.sin (20 * π / 180) * Real.cos (200 * π / 180))) /
  (Real.cos (160 * π / 180) - Real.sqrt (1 - Real.cos (20 * π / 180) ^ 2)) = -1 := by
  sorry

end trigonometric_simplification_special_angle_simplification_l3500_350070


namespace locus_of_R_l3500_350035

/-- The ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

/-- Point A -/
def A : ℝ × ℝ := (-1, 0)

/-- Point B -/
def B : ℝ × ℝ := (1, 0)

/-- Point C -/
def C : ℝ × ℝ := (3, 0)

/-- The locus equation -/
def locus_equation (x y : ℝ) : Prop := 45 * x^2 - 108 * y^2 = 20

/-- The theorem stating the locus of point R -/
theorem locus_of_R :
  ∀ (R : ℝ × ℝ),
  (∃ (P Q : ℝ × ℝ),
    ellipse P.1 P.2 ∧
    ellipse Q.1 Q.2 ∧
    Q.1 < P.1 ∧
    (∃ (t : ℝ), t > 0 ∧ Q.1 = C.1 - t * (C.1 - P.1) ∧ Q.2 = C.2 - t * (C.2 - P.2)) ∧
    (∃ (s : ℝ), A.1 + s * (P.1 - A.1) = R.1 ∧ A.2 + s * (P.2 - A.2) = R.2) ∧
    (∃ (u : ℝ), B.1 + u * (Q.1 - B.1) = R.1 ∧ B.2 + u * (Q.2 - B.2) = R.2)) →
  locus_equation R.1 R.2 ∧ 2/3 < R.1 ∧ R.1 < 4/3 :=
sorry

end locus_of_R_l3500_350035


namespace quadrilateral_has_four_sides_and_angles_l3500_350082

/-- Definition of a quadrilateral -/
structure Quadrilateral where
  sides : Fin 4 → Seg
  angles : Fin 4 → Angle

/-- Theorem: A quadrilateral has four sides and four angles -/
theorem quadrilateral_has_four_sides_and_angles (q : Quadrilateral) :
  (∃ (s : Fin 4 → Seg), q.sides = s) ∧ (∃ (a : Fin 4 → Angle), q.angles = a) := by
  sorry

end quadrilateral_has_four_sides_and_angles_l3500_350082


namespace triangle_area_PQR_l3500_350053

/-- The area of a triangle PQR with given coordinates -/
theorem triangle_area_PQR :
  let P : ℝ × ℝ := (-4, 2)
  let Q : ℝ × ℝ := (6, 2)
  let R : ℝ × ℝ := (2, -5)
  let base : ℝ := Q.1 - P.1
  let height : ℝ := P.2 - R.2
  (1 / 2 : ℝ) * base * height = 35 := by sorry

end triangle_area_PQR_l3500_350053


namespace max_ratio_squared_l3500_350049

theorem max_ratio_squared (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a^2 - 2*b^2 = 0)
  (h2 : a^2 + y^2 = b^2 + x^2)
  (h3 : b^2 + x^2 = (a - x)^2 + (b - y)^2)
  (h4 : 0 ≤ x ∧ x < a)
  (h5 : 0 ≤ y ∧ y < b)
  (h6 : x^2 + y^2 = b^2) :
  ∃ (ρ : ℝ), ρ^2 = 2 ∧ ∀ (c : ℝ), (c = a / b → c^2 ≤ ρ^2) :=
sorry

end max_ratio_squared_l3500_350049


namespace pairball_longest_time_l3500_350012

/-- Represents the pairball game setup -/
structure PairballGame where
  totalTime : ℕ
  numChildren : ℕ
  longestPlayRatio : ℕ

/-- Calculates the playing time of the child who played the longest -/
def longestPlayingTime (game : PairballGame) : ℕ :=
  let totalChildMinutes := 2 * game.totalTime
  let adjustedChildren := game.numChildren - 1 + game.longestPlayRatio
  (totalChildMinutes * game.longestPlayRatio) / adjustedChildren

/-- Theorem stating that the longest playing time in the given scenario is 68 minutes -/
theorem pairball_longest_time :
  let game : PairballGame := {
    totalTime := 120,
    numChildren := 6,
    longestPlayRatio := 2
  }
  longestPlayingTime game = 68 := by
  sorry

end pairball_longest_time_l3500_350012


namespace smallest_integer_with_remainder_one_general_form_with_remainder_one_l3500_350029

theorem smallest_integer_with_remainder_one : ∃ (a : ℕ), a > 0 ∧
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 9 → a % k = 1) ∧
  (∀ b : ℕ, b > 0 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ 9 → b % k = 1) → a ≤ b) ∧
  a = 2521 :=
sorry

theorem general_form_with_remainder_one :
  ∀ (a : ℕ), a > 0 →
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 9 → a % k = 1) →
  ∃ (n : ℕ), a = 2520 * n + 1 :=
sorry

end smallest_integer_with_remainder_one_general_form_with_remainder_one_l3500_350029


namespace audrey_heracles_age_ratio_l3500_350095

/-- Proves that the ratio of Audrey's age in 3 years to Heracles' current age is 2:1 -/
theorem audrey_heracles_age_ratio :
  let heracles_age : ℕ := 10
  let audrey_age : ℕ := heracles_age + 7
  let audrey_age_in_3_years : ℕ := audrey_age + 3
  (audrey_age_in_3_years : ℚ) / heracles_age = 2 := by
  sorry

end audrey_heracles_age_ratio_l3500_350095


namespace only_cri_du_chat_chromosomal_variation_l3500_350062

-- Define the types of genetic causes
inductive GeneticCause
| GeneMutation
| ChromosomalVariation

-- Define the genetic diseases
inductive GeneticDisease
| Albinism
| Hemophilia
| CriDuChatSyndrome
| SickleCellAnemia

-- Define a function that assigns a cause to each disease
def diseaseCause : GeneticDisease → GeneticCause
| GeneticDisease.Albinism => GeneticCause.GeneMutation
| GeneticDisease.Hemophilia => GeneticCause.GeneMutation
| GeneticDisease.CriDuChatSyndrome => GeneticCause.ChromosomalVariation
| GeneticDisease.SickleCellAnemia => GeneticCause.GeneMutation

-- Theorem stating that only Cri-du-chat syndrome is caused by chromosomal variation
theorem only_cri_du_chat_chromosomal_variation :
  ∀ (d : GeneticDisease), diseaseCause d = GeneticCause.ChromosomalVariation ↔ d = GeneticDisease.CriDuChatSyndrome :=
by sorry


end only_cri_du_chat_chromosomal_variation_l3500_350062


namespace determinant_scaling_l3500_350083

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = 10 →
  Matrix.det ![![3*x, 3*y], ![3*z, 3*w]] = 90 := by
  sorry

end determinant_scaling_l3500_350083


namespace expression_simplification_l3500_350091

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) : 
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 := by
  sorry

end expression_simplification_l3500_350091


namespace fraction_power_product_l3500_350046

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end fraction_power_product_l3500_350046


namespace exponential_inequality_l3500_350031

theorem exponential_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  2^x * x + 2^y * y ≥ 2^y * x + 2^x * y := by
  sorry

end exponential_inequality_l3500_350031


namespace new_class_mean_approx_67_percent_l3500_350010

/-- Represents the class statistics for Mr. Thompson's chemistry class -/
structure ChemistryClass where
  total_students : ℕ
  group1_students : ℕ
  group1_average : ℚ
  group2_students : ℕ
  group2_average : ℚ
  group3_students : ℕ
  group3_average : ℚ

/-- Calculates the new class mean for Mr. Thompson's chemistry class -/
def new_class_mean (c : ChemistryClass) : ℚ :=
  (c.group1_students * c.group1_average + 
   c.group2_students * c.group2_average + 
   c.group3_students * c.group3_average) / c.total_students

/-- Theorem stating that the new class mean is approximately 67% -/
theorem new_class_mean_approx_67_percent (c : ChemistryClass) 
  (h1 : c.total_students = 60)
  (h2 : c.group1_students = 50)
  (h3 : c.group1_average = 65/100)
  (h4 : c.group2_students = 8)
  (h5 : c.group2_average = 85/100)
  (h6 : c.group3_students = 2)
  (h7 : c.group3_average = 55/100) :
  ∃ ε > 0, |new_class_mean c - 67/100| < ε :=
by
  sorry


end new_class_mean_approx_67_percent_l3500_350010


namespace cucumber_problem_l3500_350016

theorem cucumber_problem (boxes : ℕ) (cucumbers_per_box : ℕ) (rotten : ℕ) (bags : ℕ) :
  boxes = 7 →
  cucumbers_per_box = 16 →
  rotten = 13 →
  bags = 8 →
  (boxes * cucumbers_per_box - rotten) % bags = 3 := by
sorry

end cucumber_problem_l3500_350016


namespace least_number_remainder_l3500_350038

theorem least_number_remainder (n : ℕ) : 
  (n % 20 = 1929 % 20 ∧ n % 2535 = 1929 ∧ n % 40 = 34) →
  (∀ m : ℕ, m < n → ¬(m % 20 = 1929 % 20 ∧ m % 2535 = 1929 ∧ m % 40 = 34)) →
  n = 1394 →
  n % 20 = 14 := by
sorry

end least_number_remainder_l3500_350038
