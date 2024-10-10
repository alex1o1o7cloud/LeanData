import Mathlib

namespace two_digit_number_problem_l210_21081

theorem two_digit_number_problem (n m : ℕ) : 
  10 ≤ m ∧ m < n ∧ n ≤ 99 →  -- n and m are 2-digit numbers, n > m
  n - m = 58 →  -- difference is 58
  n^2 % 100 = m^2 % 100 →  -- last two digits of squares are the same
  m = 21 := by
sorry

end two_digit_number_problem_l210_21081


namespace round_table_seating_l210_21095

/-- The number of unique circular arrangements of n distinct objects -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of people to be seated around the round table -/
def numberOfPeople : ℕ := 8

theorem round_table_seating :
  circularArrangements numberOfPeople = 5040 := by
  sorry

end round_table_seating_l210_21095


namespace white_paint_calculation_l210_21028

theorem white_paint_calculation (total_paint blue_paint : ℕ) 
  (h1 : total_paint = 6689)
  (h2 : blue_paint = 6029) :
  total_paint - blue_paint = 660 := by
  sorry

end white_paint_calculation_l210_21028


namespace tony_running_speed_l210_21010

/-- The distance to the store in miles -/
def distance : ℝ := 4

/-- Tony's walking speed in miles per hour -/
def walking_speed : ℝ := 2

/-- The average time Tony spends to get to the store in minutes -/
def average_time : ℝ := 56

/-- Tony's running speed in miles per hour -/
def running_speed : ℝ := 10

theorem tony_running_speed :
  let time_walking := (distance / walking_speed) * 60
  let time_running := (distance / running_speed) * 60
  (time_walking + 2 * time_running) / 3 = average_time :=
by sorry

end tony_running_speed_l210_21010


namespace soccer_league_games_l210_21042

theorem soccer_league_games (n : ℕ) (h : n = 12) : (n * (n - 1)) / 2 = 66 := by
  sorry

end soccer_league_games_l210_21042


namespace quadrilateral_area_theorem_l210_21048

-- Define the quadrilateral PQRS
def P (k a : ℤ) : ℤ × ℤ := (k, a)
def Q (k a : ℤ) : ℤ × ℤ := (a, k)
def R (k a : ℤ) : ℤ × ℤ := (-k, -a)
def S (k a : ℤ) : ℤ × ℤ := (-a, -k)

-- Define the area function for PQRS
def area_PQRS (k a : ℤ) : ℤ := 2 * |k - a| * |k + a|

-- Theorem statement
theorem quadrilateral_area_theorem (k a : ℤ) 
  (h1 : k > a) (h2 : a > 0) (h3 : area_PQRS k a = 32) : 
  k + a = 8 := by sorry

end quadrilateral_area_theorem_l210_21048


namespace problem_solution_l210_21069

theorem problem_solution : (1 / ((-8^2)^4)) * (-8)^9 = -8 := by sorry

end problem_solution_l210_21069


namespace problem_solution_l210_21075

theorem problem_solution (x y : ℚ) : 
  x = 51 → x^3*y - 2*x^2*y + x*y = 51000 → y = 2/5 := by
  sorry

end problem_solution_l210_21075


namespace store_items_cost_price_l210_21035

/-- The cost price of an item given its profit and loss prices -/
def costPrice (profitPrice lossPrice : ℚ) : ℚ := (profitPrice + lossPrice) / 2

/-- The combined cost price of three items -/
def combinedCostPrice (cpA cpB cpC : ℚ) : ℚ := cpA + cpB + cpC

theorem store_items_cost_price : 
  let cpA := costPrice 110 70
  let cpB := costPrice 90 30
  let cpC := costPrice 150 50
  combinedCostPrice cpA cpB cpC = 250 := by
sorry

#eval costPrice 110 70 -- Expected output: 90
#eval costPrice 90 30  -- Expected output: 60
#eval costPrice 150 50 -- Expected output: 100
#eval combinedCostPrice (costPrice 110 70) (costPrice 90 30) (costPrice 150 50) -- Expected output: 250

end store_items_cost_price_l210_21035


namespace perpendicular_tangents_intersection_l210_21099

/-- The value of a for which the tangents to C₁ and C₂ at their intersection point are perpendicular -/
theorem perpendicular_tangents_intersection (a : ℝ) : 
  a > 0 → 
  ∃ (x y : ℝ), 
    (y = a * x^3 + 1) ∧ 
    (x^2 + y^2 = 5/2) ∧ 
    (∃ (m₁ m₂ : ℝ), 
      (m₁ = 3 * a * x^2) ∧ 
      (m₂ = -x / y) ∧ 
      (m₁ * m₂ = -1)) →
  a = 4 := by
  sorry

end perpendicular_tangents_intersection_l210_21099


namespace inequality_group_C_equality_group_A_equality_group_B_equality_group_D_l210_21068

theorem inequality_group_C (a b : ℝ) : ∃ a b, 3 * (a + b) ≠ 3 * a + b :=
sorry

theorem equality_group_A (a b : ℝ) : a + b = b + a :=
sorry

theorem equality_group_B (a : ℝ) : 3 * a = a + a + a :=
sorry

theorem equality_group_D (a : ℝ) : a ^ 3 = a * a * a :=
sorry

end inequality_group_C_equality_group_A_equality_group_B_equality_group_D_l210_21068


namespace base_10_to_base_4_123_l210_21002

/-- Converts a natural number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list of digits is a valid base 4 representation -/
def isValidBase4 (digits : List ℕ) : Prop :=
  sorry

theorem base_10_to_base_4_123 :
  let base4Repr := toBase4 123
  isValidBase4 base4Repr ∧ base4Repr = [1, 3, 2, 3] :=
by sorry

end base_10_to_base_4_123_l210_21002


namespace necessary_not_sufficient_condition_l210_21054

def f (a b x : ℝ) : ℝ := (x + a) * abs (x + b)

theorem necessary_not_sufficient_condition :
  (∀ x, f a b x = -f a b (-x)) → a = b ∧
  ∃ a b, a = b ∧ ∃ x, f a b x ≠ -f a b (-x) :=
by sorry

end necessary_not_sufficient_condition_l210_21054


namespace dave_car_count_l210_21090

theorem dave_car_count (store1 store2 store3 store4 store5 : ℕ) 
  (h1 : store2 = 14)
  (h2 : store3 = 14)
  (h3 : store4 = 21)
  (h4 : store5 = 25)
  (h5 : (store1 + store2 + store3 + store4 + store5) / 5 = 208/10) :
  store1 = 30 := by
sorry

end dave_car_count_l210_21090


namespace honey_harvest_calculation_l210_21011

/-- The amount of honey harvested last year -/
def last_year_harvest : ℕ := 8564 - 6085

/-- The increase in honey harvest this year -/
def harvest_increase : ℕ := 6085

/-- The total amount of honey harvested this year -/
def this_year_harvest : ℕ := 8564

theorem honey_harvest_calculation :
  last_year_harvest = 2479 :=
by sorry

end honey_harvest_calculation_l210_21011


namespace circle_center_sum_l210_21036

theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) →  -- Circle equation
  (∀ a b : ℝ, (a - x)^2 + (b - y)^2 ≤ (x - a)^2 + (y - b)^2) →  -- Definition of center
  x + y = -1 := by
sorry

end circle_center_sum_l210_21036


namespace play_attendance_l210_21006

theorem play_attendance (total_people : ℕ) (adult_price child_price : ℕ) (total_receipts : ℕ) :
  total_people = 610 →
  adult_price = 2 →
  child_price = 1 →
  total_receipts = 960 →
  ∃ (adults children : ℕ),
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_receipts ∧
    adults = 350 :=
by sorry

end play_attendance_l210_21006


namespace inequality_solution_set_l210_21084

theorem inequality_solution_set (x : ℝ) : 
  8 * x^2 + 6 * x > 10 ↔ x < -1 ∨ x > 5/4 := by sorry

end inequality_solution_set_l210_21084


namespace triangle_altitude_circumradius_l210_21097

/-- For any triangle with sides a, b, c, altitude ha from vertex A to side a,
    and circumradius R, the equation ha = bc / (2R) holds. -/
theorem triangle_altitude_circumradius (a b c ha R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ ha > 0 ∧ R > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_altitude : ha = (2 * (a.sqrt * b.sqrt * c.sqrt + (a + b + c) / 2)) / a)
  (h_circumradius : R = (a * b * c) / (4 * (a.sqrt * b.sqrt * c.sqrt + (a + b + c) / 2))) :
  ha = b * c / (2 * R) := by
sorry

end triangle_altitude_circumradius_l210_21097


namespace product_cde_value_l210_21029

theorem product_cde_value (a b c d e f : ℝ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 0.6666666666666666) :
  c * d * e = 750 := by
  sorry

end product_cde_value_l210_21029


namespace apple_difference_l210_21015

/-- Proves that the difference between green and red apples after delivery is 140 -/
theorem apple_difference (initial_green : ℕ) (initial_red_difference : ℕ) (delivered_green : ℕ) : 
  initial_green = 32 →
  initial_red_difference = 200 →
  delivered_green = 340 →
  (initial_green + delivered_green) - (initial_green + initial_red_difference) = 140 := by
sorry

end apple_difference_l210_21015


namespace binary_product_l210_21078

-- Define the binary numbers
def binary1 : Nat := 0b11011
def binary2 : Nat := 0b111
def binary3 : Nat := 0b101

-- Define the result
def result : Nat := 0b1110110001

-- Theorem statement
theorem binary_product :
  binary1 * binary2 * binary3 = result := by
  sorry

end binary_product_l210_21078


namespace set_intersection_problem_l210_21038

def A : Set ℕ := {1, 6, 8, 10}
def B : Set ℕ := {2, 4, 8, 10}

theorem set_intersection_problem : A ∩ B = {8, 10} := by
  sorry

end set_intersection_problem_l210_21038


namespace digit_sum_is_seventeen_l210_21030

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 100

/-- Represents a three-digit number -/
def ThreeDigitNumber := Fin 1000

/-- The equation (AB) * (CD) = GGG -/
def satisfiesEquation (A B C D G : Digit) : Prop :=
  ∃ (AB CD : TwoDigitNumber) (GGG : ThreeDigitNumber),
    AB.val = 10 * A.val + B.val ∧
    CD.val = 10 * C.val + D.val ∧
    GGG.val = 100 * G.val + 10 * G.val + G.val ∧
    AB.val * CD.val = GGG.val

/-- All digits are distinct -/
def allDistinct (A B C D G : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ G ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ G ∧
  C ≠ D ∧ C ≠ G ∧
  D ≠ G

theorem digit_sum_is_seventeen :
  ∃ (A B C D G : Digit),
    satisfiesEquation A B C D G ∧
    allDistinct A B C D G ∧
    A.val + B.val + C.val + D.val + G.val = 17 :=
by sorry

end digit_sum_is_seventeen_l210_21030


namespace foci_coordinates_l210_21065

/-- Definition of a hyperbola with equation x^2 - y^2/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- Definition of the distance from center to focus for this hyperbola -/
def c : ℝ := 2

/-- The coordinates of the foci of the hyperbola x^2 - y^2/3 = 1 are (±2, 0) -/
theorem foci_coordinates :
  ∀ x y : ℝ, hyperbola x y → (x = c ∨ x = -c) ∧ y = 0 :=
sorry

end foci_coordinates_l210_21065


namespace cube_plane_angle_l210_21000

/-- Given a cube with a plane passing through a side of its base, dividing the volume
    in the ratio m:n (where m ≤ n), the angle α between this plane and the base of
    the cube is given by α = arctan(2m / (m + n)). -/
theorem cube_plane_angle (m n : ℝ) (h : 0 < m ∧ m ≤ n) : 
  ∃ (α : ℝ), α = Real.arctan (2 * m / (m + n)) ∧
  ∃ (V₁ V₂ : ℝ), V₁ / V₂ = m / n ∧
  V₁ = (1/2) * (Real.tan α) ∧
  V₂ = 1 - (1/2) * (Real.tan α) := by
sorry

end cube_plane_angle_l210_21000


namespace monica_students_l210_21040

/-- The number of students Monica sees each day -/
def total_students : ℕ :=
  let first_class := 20
  let second_third_classes := 25 + 25
  let fourth_class := first_class / 2
  let fifth_sixth_classes := 28 + 28
  first_class + second_third_classes + fourth_class + fifth_sixth_classes

/-- Monica sees 136 students each day -/
theorem monica_students : total_students = 136 := by
  sorry

end monica_students_l210_21040


namespace not_integer_fraction_l210_21024

theorem not_integer_fraction (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  ¬ ∃ (n : ℤ), (a^2 + b^2) / (a^2 - b^2) = n := by
  sorry

end not_integer_fraction_l210_21024


namespace perpendicular_vector_scalar_l210_21060

/-- Given vectors a and b in ℝ², if a + t*b is perpendicular to a, then t = -5/8 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (t : ℝ) 
  (h1 : a = (1, 2))
  (h2 : b = (2, 3))
  (h3 : (a.1 + t * b.1, a.2 + t * b.2) • a = 0) :
  t = -5/8 := by
  sorry

end perpendicular_vector_scalar_l210_21060


namespace eight_point_five_million_scientific_notation_l210_21007

theorem eight_point_five_million_scientific_notation :
  (8.5 * 1000000 : ℝ) = 8.5 * (10 ^ 6) :=
by sorry

end eight_point_five_million_scientific_notation_l210_21007


namespace cubic_root_product_l210_21023

theorem cubic_root_product : ∃ (z₁ z₂ : ℂ),
  z₁^3 = -27 ∧ z₂^3 = -27 ∧ 
  (∃ (a₁ b₁ a₂ b₂ : ℝ), z₁ = a₁ + b₁ * I ∧ z₂ = a₂ + b₂ * I ∧ a₁ > 0 ∧ a₂ > 0) ∧
  z₁ * z₂ = 9 := by
sorry

end cubic_root_product_l210_21023


namespace max_triangle_area_l210_21082

/-- The maximum area of a triangle ABC with side lengths satisfying the given constraints is 1 -/
theorem max_triangle_area (AB BC CA : ℝ) (h1 : 0 ≤ AB ∧ AB ≤ 1) (h2 : 1 ≤ BC ∧ BC ≤ 2) (h3 : 2 ≤ CA ∧ CA ≤ 3) :
  (∃ (S : ℝ), S = Real.sqrt ((AB + BC + CA) / 2 * ((AB + BC + CA) / 2 - AB) * ((AB + BC + CA) / 2 - BC) * ((AB + BC + CA) / 2 - CA))) →
  (∀ (area : ℝ), area ≤ 1) :=
by sorry

end max_triangle_area_l210_21082


namespace between_negative_two_and_zero_l210_21047

def numbers : Set ℝ := {3, 1, -3, -1}

theorem between_negative_two_and_zero :
  ∃ x ∈ numbers, -2 < x ∧ x < 0 :=
by
  sorry

end between_negative_two_and_zero_l210_21047


namespace value_of_expression_l210_21027

def smallest_positive_integer : ℕ := 1

def largest_negative_integer : ℤ := -1

def smallest_absolute_rational : ℚ := 0

def rational_at_distance_4 : Set ℚ := {d : ℚ | d = 4 ∨ d = -4}

theorem value_of_expression (a b : ℤ) (c d : ℚ) :
  a = smallest_positive_integer ∧
  b = largest_negative_integer ∧
  c = smallest_absolute_rational ∧
  d ∈ rational_at_distance_4 →
  a - b - c + d = -2 ∨ a - b - c + d = 6 := by
  sorry

end value_of_expression_l210_21027


namespace P_equals_F_l210_21087

-- Define the sets P and F
def P : Set ℝ := {y | ∃ x, y = x^2 + 1}
def F : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem P_equals_F : P = F := by sorry

end P_equals_F_l210_21087


namespace kim_total_points_l210_21039

/-- Represents the points awarded for each round in the contest -/
structure RoundPoints where
  easy : Nat
  average : Nat
  hard : Nat

/-- Represents the number of correct answers in each round -/
structure CorrectAnswers where
  easy : Nat
  average : Nat
  hard : Nat

/-- Calculates the total points given the round points and correct answers -/
def calculateTotalPoints (points : RoundPoints) (answers : CorrectAnswers) : Nat :=
  points.easy * answers.easy + points.average * answers.average + points.hard * answers.hard

/-- Theorem: Given the contest conditions, Kim's total points are 38 -/
theorem kim_total_points :
  let points : RoundPoints := ⟨2, 3, 5⟩
  let answers : CorrectAnswers := ⟨6, 2, 4⟩
  calculateTotalPoints points answers = 38 := by
  sorry


end kim_total_points_l210_21039


namespace constant_term_implies_a_l210_21098

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The constant term in the expansion of (ax^2 + 1/√x)^5 -/
def constantTerm (a : ℝ) : ℝ := a * (binomial 5 4)

theorem constant_term_implies_a (a : ℝ) :
  constantTerm a = -10 → a = -2 := by sorry

end constant_term_implies_a_l210_21098


namespace max_value_of_operation_l210_21051

theorem max_value_of_operation : ∃ (n : ℤ), 
  10 ≤ n ∧ n ≤ 99 ∧ 
  (250 - 3*n)^2 = 4 ∧
  ∀ (m : ℤ), 10 ≤ m ∧ m ≤ 99 → (250 - 3*m)^2 ≤ 4 :=
by sorry

end max_value_of_operation_l210_21051


namespace arithmetic_mean_of_fractions_l210_21094

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((3 / 8 : ℚ) + (5 / 9 : ℚ)) = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l210_21094


namespace sqrt_inequality_l210_21072

theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end sqrt_inequality_l210_21072


namespace candy_difference_l210_21033

-- Define the initial variables
def candy_given : ℝ := 6.25
def candy_left : ℝ := 4.75

-- Define the theorem
theorem candy_difference : candy_given - candy_left = 1.50 := by
  sorry

end candy_difference_l210_21033


namespace parabola_circle_tangent_l210_21061

/-- Given a parabola and a circle, if the parabola's axis is tangent to the circle, then p = 2 -/
theorem parabola_circle_tangent (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Parabola equation
  (∀ x y : ℝ, x^2 + y^2 - 6*x - 7 = 0) →  -- Circle equation
  (∀ x y : ℝ, x = -p/2) →  -- Parabola's axis equation
  (abs (-p/2 + 3) = 4) →  -- Tangency condition (distance from circle center to axis equals radius)
  p = 2 := by
sorry

end parabola_circle_tangent_l210_21061


namespace factorial_ratio_equals_15120_l210_21083

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_ratio_equals_15120 : 
  factorial 10 / (factorial 5 * factorial 2) = 15120 := by sorry

end factorial_ratio_equals_15120_l210_21083


namespace smallest_b_value_l210_21079

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 4) 
  (h2 : Nat.gcd ((a^3 - b^3) / (a - b)) (a * b) = 4) : 
  b = 2 ∧ ∀ (c : ℕ+), c < b → ¬(∃ (d : ℕ+), d - c = 4 ∧ 
    Nat.gcd ((d^3 - c^3) / (d - c)) (d * c) = 4) :=
by sorry

end smallest_b_value_l210_21079


namespace geometry_propositions_l210_21085

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) := by
  sorry

end geometry_propositions_l210_21085


namespace soccer_team_penalty_kicks_l210_21044

/-- Calculates the total number of penalty kicks in a soccer team training exercise. -/
def total_penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  (total_players - goalies) * goalies

/-- Theorem: In a soccer team with 24 players, including 4 goalies, 
    where each player shoots once at each goalie, the total number of penalty kicks is 92. -/
theorem soccer_team_penalty_kicks :
  total_penalty_kicks 24 4 = 92 := by
  sorry


end soccer_team_penalty_kicks_l210_21044


namespace log_2_base_10_bounds_l210_21062

theorem log_2_base_10_bounds : ∃ (log_2_base_10 : ℝ),
  (10 : ℝ) ^ 3 = 1000 ∧
  (10 : ℝ) ^ 4 = 10000 ∧
  (2 : ℝ) ^ 10 = 1024 ∧
  (2 : ℝ) ^ 11 = 2048 ∧
  (2 : ℝ) ^ 12 = 4096 ∧
  (2 : ℝ) ^ 13 = 8192 ∧
  (∀ x > 0, (10 : ℝ) ^ (log_2_base_10 * Real.log x) = x) ∧
  3 / 10 < log_2_base_10 ∧
  log_2_base_10 < 4 / 13 :=
by sorry

end log_2_base_10_bounds_l210_21062


namespace product_of_real_parts_of_complex_solutions_l210_21018

theorem product_of_real_parts_of_complex_solutions : ∃ (z₁ z₂ : ℂ),
  (z₁^2 + 2*z₁ = Complex.I) ∧
  (z₂^2 + 2*z₂ = Complex.I) ∧
  (z₁ ≠ z₂) ∧
  (Complex.re z₁ * Complex.re z₂ = (1 - Real.sqrt 2) / 2) := by
  sorry

end product_of_real_parts_of_complex_solutions_l210_21018


namespace arithmetic_sequence_sum_l210_21049

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 2 = 3 →
  a 5 + a 7 = 10 →
  a 1 + a 10 = 9.5 := by sorry

end arithmetic_sequence_sum_l210_21049


namespace uncoverable_iff_odd_specified_boards_uncoverable_l210_21077

/-- Represents a board configuration -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)
  (missing : ℕ)

/-- Calculates the number of coverable squares on a board -/
def coverableSquares (b : Board) : ℕ :=
  b.rows * b.cols - b.missing

/-- Determines if a board can be completely covered by dominoes -/
def canBeCovered (b : Board) : Prop :=
  coverableSquares b % 2 = 0

/-- Theorem: A board cannot be covered iff the number of coverable squares is odd -/
theorem uncoverable_iff_odd (b : Board) :
  ¬(canBeCovered b) ↔ coverableSquares b % 2 = 1 :=
sorry

/-- Examples of board configurations -/
def board_7x3 : Board := ⟨7, 3, 0⟩
def board_6x4_unpainted : Board := ⟨6, 4, 1⟩
def board_5x7 : Board := ⟨5, 7, 0⟩
def board_8x8_missing : Board := ⟨8, 8, 1⟩

/-- Theorem: The specified boards cannot be covered -/
theorem specified_boards_uncoverable :
  (¬(canBeCovered board_7x3)) ∧
  (¬(canBeCovered board_6x4_unpainted)) ∧
  (¬(canBeCovered board_5x7)) ∧
  (¬(canBeCovered board_8x8_missing)) :=
sorry

end uncoverable_iff_odd_specified_boards_uncoverable_l210_21077


namespace inverse_function_sum_l210_21016

/-- Given a function g and constants a, b, c, d, k satisfying certain conditions,
    prove that a + d = 0 -/
theorem inverse_function_sum (a b c d k : ℝ) :
  (∀ x, (k * (a * x + b)) / (k * (c * x + d)) = 
        ((k * (a * ((k * (a * x + b)) / (k * (c * x + d))) + b)) / 
         (k * (c * ((k * (a * x + b)) / (k * (c * x + d))) + d)))) →
  (a * b * c * d * k ≠ 0) →
  (a + k * c = 0) →
  (a + d = 0) := by
sorry


end inverse_function_sum_l210_21016


namespace negation_of_exists_negation_of_proposition_l210_21055

theorem negation_of_exists (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ ∀ n, ¬ p n :=
by sorry

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end negation_of_exists_negation_of_proposition_l210_21055


namespace smallest_sum_X_plus_c_l210_21074

theorem smallest_sum_X_plus_c : ∀ (X c : ℕ),
  X < 5 → 
  X > 0 →
  c > 6 →
  (31 * X = 4 * c + 4) →
  ∀ (Y d : ℕ), Y < 5 → Y > 0 → d > 6 → (31 * Y = 4 * d + 4) →
  X + c ≤ Y + d :=
by
  sorry

end smallest_sum_X_plus_c_l210_21074


namespace largest_factorial_with_100_zeros_l210_21046

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- The largest positive integer n such that n! ends with exactly 100 zeros -/
theorem largest_factorial_with_100_zeros : 
  (∀ m : ℕ, m > 409 → trailingZeros m > 100) ∧ 
  trailingZeros 409 = 100 :=
sorry

end largest_factorial_with_100_zeros_l210_21046


namespace circle_equation_l210_21021

theorem circle_equation (x y : ℝ) : 
  (∃ (C : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1^2 + p.2^2 = 16)) ∧
    ((-4, 0) ∈ C) ∧
    ((x, y) ∈ C)) →
  x^2 + y^2 = 16 :=
by
  sorry

end circle_equation_l210_21021


namespace fourth_power_sum_l210_21031

theorem fourth_power_sum (α β γ : ℂ) 
  (h1 : α + β + γ = 1)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 9) :
  α^4 + β^4 + γ^4 = 56 := by
  sorry

end fourth_power_sum_l210_21031


namespace charity_event_arrangements_l210_21089

/-- The number of ways to arrange volunteers for a 3-day charity event -/
def charity_arrangements (total_volunteers : ℕ) (day1_needed : ℕ) (day2_needed : ℕ) (day3_needed : ℕ) : ℕ :=
  Nat.choose total_volunteers day1_needed *
  Nat.choose (total_volunteers - day1_needed) day2_needed *
  Nat.choose (total_volunteers - day1_needed - day2_needed) day3_needed

/-- Theorem stating that the number of arrangements for the given conditions is 60 -/
theorem charity_event_arrangements :
  charity_arrangements 6 1 2 3 = 60 := by
  sorry

end charity_event_arrangements_l210_21089


namespace dog_does_not_catch_hare_l210_21009

/-- Represents the chase scenario between a dog and a hare -/
structure ChaseScenario where
  dog_speed : ℝ
  hare_speed : ℝ
  initial_distance : ℝ
  bushes_distance : ℝ

/-- Determines if the dog catches the hare before it reaches the bushes -/
def dog_catches_hare (scenario : ChaseScenario) : Prop :=
  let relative_speed := scenario.dog_speed - scenario.hare_speed
  let catch_time := scenario.initial_distance / relative_speed
  let hare_distance := scenario.hare_speed * catch_time
  hare_distance < scenario.bushes_distance

/-- The theorem stating that the dog does not catch the hare -/
theorem dog_does_not_catch_hare (scenario : ChaseScenario)
  (h1 : scenario.dog_speed = 17)
  (h2 : scenario.hare_speed = 14)
  (h3 : scenario.initial_distance = 150)
  (h4 : scenario.bushes_distance = 520) :
  ¬(dog_catches_hare scenario) := by
  sorry

#check dog_does_not_catch_hare

end dog_does_not_catch_hare_l210_21009


namespace correct_hourly_wage_l210_21005

/-- The hourly wage for a manufacturing plant worker --/
def hourly_wage : ℝ :=
  12.50

/-- The piece rate per widget --/
def piece_rate : ℝ :=
  0.16

/-- The number of widgets produced in a week --/
def widgets_per_week : ℕ :=
  1000

/-- The number of hours worked in a week --/
def hours_per_week : ℕ :=
  40

/-- The total earnings for a week --/
def total_earnings : ℝ :=
  660

theorem correct_hourly_wage :
  hourly_wage * hours_per_week + piece_rate * widgets_per_week = total_earnings :=
by sorry

end correct_hourly_wage_l210_21005


namespace hyperbola_eccentricity_l210_21071

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is the line x - 2y = 0,
    then its eccentricity is √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ x - 2*y = 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 5 / 2 := by
  sorry

end hyperbola_eccentricity_l210_21071


namespace unique_b_value_l210_21093

theorem unique_b_value : ∃! b : ℚ, ∀ x : ℚ, 5 * (3 * x - b) = 3 * (5 * x - 9) :=
by
  -- The proof goes here
  sorry

end unique_b_value_l210_21093


namespace calculator_result_l210_21001

def special_key (x : ℚ) : ℚ := 1 / (1 - x)

theorem calculator_result :
  (Nat.iterate special_key 100 5 : ℚ) = -1/4 := by
  sorry

end calculator_result_l210_21001


namespace alice_work_problem_l210_21017

/-- Alice's work problem -/
theorem alice_work_problem (total_days : ℕ) (daily_wage : ℕ) (daily_loss : ℕ) (total_earnings : ℤ) :
  total_days = 20 →
  daily_wage = 80 →
  daily_loss = 40 →
  total_earnings = 880 →
  ∃ (days_not_worked : ℕ),
    days_not_worked = 6 ∧
    days_not_worked ≤ total_days ∧
    (daily_wage * (total_days - days_not_worked) : ℤ) - (daily_loss * days_not_worked : ℤ) = total_earnings :=
by sorry

end alice_work_problem_l210_21017


namespace largest_integer_with_four_digit_square_base8_l210_21019

/-- The largest integer whose square has exactly 4 digits in base 8 -/
def M : ℕ := 31

/-- Conversion of M to base 8 -/
def M_base8 : ℕ := 37

theorem largest_integer_with_four_digit_square_base8 :
  (∀ n : ℕ, n > M → ¬(8^3 ≤ n^2 ∧ n^2 < 8^4)) ∧
  (8^3 ≤ M^2 ∧ M^2 < 8^4) ∧
  M_base8 = M := by sorry

end largest_integer_with_four_digit_square_base8_l210_21019


namespace marble_theorem_l210_21056

def marble_problem (wolfgang ludo michael shania gabriel : ℕ) : Prop :=
  wolfgang = 16 ∧
  ludo = wolfgang + wolfgang / 4 ∧
  michael = 2 * (wolfgang + ludo) / 3 ∧
  shania = 2 * ludo ∧
  gabriel = wolfgang + ludo + michael + shania - 1 ∧
  (wolfgang + ludo + michael + shania + gabriel) / 5 = 39

theorem marble_theorem : ∃ wolfgang ludo michael shania gabriel : ℕ,
  marble_problem wolfgang ludo michael shania gabriel := by
  sorry

end marble_theorem_l210_21056


namespace count_zeros_up_to_2500_l210_21091

/-- A function that returns true if a natural number contains the digit 0 in its decimal representation -/
def containsZero (n : ℕ) : Bool :=
  sorry

/-- The count of numbers less than or equal to 2500 that contain the digit 0 -/
def countZeros : ℕ := (List.range 2501).filter containsZero |>.length

/-- Theorem stating that the count of numbers less than or equal to 2500 containing 0 is 591 -/
theorem count_zeros_up_to_2500 : countZeros = 591 := by
  sorry

end count_zeros_up_to_2500_l210_21091


namespace third_month_relation_l210_21043

def freelancer_earnings 
  (first_month : ℕ) 
  (second_month : ℕ) 
  (third_month : ℕ) 
  (total : ℕ) : Prop :=
  first_month = 350 ∧
  second_month = 2 * first_month + 50 ∧
  total = first_month + second_month + third_month ∧
  total = 5500

theorem third_month_relation 
  (first_month second_month third_month total : ℕ) :
  freelancer_earnings first_month second_month third_month total →
  third_month = 4 * (first_month + second_month) :=
by
  sorry

end third_month_relation_l210_21043


namespace right_triangle_enlargement_l210_21014

theorem right_triangle_enlargement (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : c^2 = a^2 + b^2) : 
  (5*c)^2 = (5*a)^2 + (5*b)^2 := by
sorry

end right_triangle_enlargement_l210_21014


namespace simplify_fraction_1_l210_21067

theorem simplify_fraction_1 (a : ℝ) (h : a ≠ 1 ∧ a ≠ -2) :
  (a^2 - 3*a + 2) / (a^2 + a - 2) = (a - 2) / (a + 2) := by
sorry

end simplify_fraction_1_l210_21067


namespace geometric_sequence_ratio_l210_21012

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 5 * a 11 = 3 →
  a 3 + a 13 = 4 →
  a 15 / a 5 = 1/3 ∨ a 15 / a 5 = 3 := by
sorry

end geometric_sequence_ratio_l210_21012


namespace length_OB_is_sqrt_13_l210_21025

-- Define the point A
def A : ℝ × ℝ × ℝ := (1, 2, 3)

-- Define the projection B of A onto the yOz plane
def B : ℝ × ℝ × ℝ := (0, A.2.1, A.2.2)

-- Define the origin O
def O : ℝ × ℝ × ℝ := (0, 0, 0)

-- Theorem to prove
theorem length_OB_is_sqrt_13 : 
  Real.sqrt ((B.1 - O.1)^2 + (B.2.1 - O.2.1)^2 + (B.2.2 - O.2.2)^2) = Real.sqrt 13 := by
  sorry

end length_OB_is_sqrt_13_l210_21025


namespace cos_alpha_plus_pi_third_l210_21076

theorem cos_alpha_plus_pi_third (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 2 * Real.sin β - Real.cos α = 1)
  (h3 : Real.sin α + 2 * Real.cos β = Real.sqrt 3) :
  Real.cos (α + π/3) = -1/4 := by
sorry

end cos_alpha_plus_pi_third_l210_21076


namespace max_value_of_function_l210_21013

theorem max_value_of_function (f : ℝ → ℝ) (h : f = λ x => x + Real.sqrt 2 * Real.cos x) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.pi / 2 → f y ≤ f x ∧
  f x = Real.pi / 4 + 1 := by
sorry

end max_value_of_function_l210_21013


namespace sqrt_meaningful_range_l210_21037

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 5) → x ≥ 5 := by
sorry

end sqrt_meaningful_range_l210_21037


namespace total_questions_in_three_hours_l210_21096

/-- The number of questions Bob creates in the first hour -/
def first_hour_questions : ℕ := 13

/-- Calculates the number of questions created in the second hour -/
def second_hour_questions : ℕ := 2 * first_hour_questions

/-- Calculates the number of questions created in the third hour -/
def third_hour_questions : ℕ := 2 * second_hour_questions

/-- Theorem: The total number of questions Bob creates in three hours is 91 -/
theorem total_questions_in_three_hours :
  first_hour_questions + second_hour_questions + third_hour_questions = 91 := by
  sorry

end total_questions_in_three_hours_l210_21096


namespace two_queens_or_at_least_one_jack_probability_l210_21064

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Jacks in a standard deck -/
def num_jacks : ℕ := 4

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The probability of drawing either two Queens or at least 1 Jack from a standard deck when selecting 2 cards randomly -/
def prob_two_queens_or_at_least_one_jack : ℚ := 2 / 13

theorem two_queens_or_at_least_one_jack_probability :
  prob_two_queens_or_at_least_one_jack = 2 / 13 := by
  sorry

end two_queens_or_at_least_one_jack_probability_l210_21064


namespace football_game_attendance_l210_21034

/-- Represents the number of adults attending the football game -/
def num_adults : ℕ := sorry

/-- Represents the number of children attending the football game -/
def num_children : ℕ := sorry

/-- The price of an adult ticket in cents -/
def adult_price : ℕ := 60

/-- The price of a child ticket in cents -/
def child_price : ℕ := 25

/-- The total number of attendees -/
def total_attendance : ℕ := 280

/-- The total money collected in cents -/
def total_money : ℕ := 14000

theorem football_game_attendance :
  (num_adults + num_children = total_attendance) ∧
  (num_adults * adult_price + num_children * child_price = total_money) →
  num_adults = 200 := by sorry

end football_game_attendance_l210_21034


namespace line_passes_through_quadrants_l210_21004

-- Define the line ax + by + c = 0
def line (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Define the quadrants
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- State the theorem
theorem line_passes_through_quadrants (a b c : ℝ) 
  (h1 : a * c < 0) (h2 : b * c < 0) :
  ∃ (x1 y1 x2 y2 x4 y4 : ℝ),
    line a b c x1 y1 ∧ first_quadrant x1 y1 ∧
    line a b c x2 y2 ∧ second_quadrant x2 y2 ∧
    line a b c x4 y4 ∧ fourth_quadrant x4 y4 :=
sorry

end line_passes_through_quadrants_l210_21004


namespace perfect_square_factors_count_l210_21053

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def count_perfect_square_factors (a b c : ℕ) : ℕ :=
  (a + 1) * (b + 1) * (c + 1)

theorem perfect_square_factors_count :
  count_perfect_square_factors 6 7 9 = 560 :=
by sorry

end perfect_square_factors_count_l210_21053


namespace math_problem_proof_l210_21057

theorem math_problem_proof (first_answer : ℕ) (second_answer : ℕ) (third_answer : ℕ) : 
  first_answer = 600 →
  second_answer = 2 * first_answer →
  first_answer + second_answer + third_answer = 3200 →
  first_answer + second_answer - third_answer = 400 := by
  sorry

end math_problem_proof_l210_21057


namespace complex_expression_evaluation_l210_21086

theorem complex_expression_evaluation : 
  (((3.2 - 1.7) / 0.003) / ((29 / 35 - 3 / 7) * 4 / 0.2) - 
   ((1 + 13 / 20 - 1.5) * 1.5) / ((2.44 + (1 + 14 / 25)) * (1 / 8))) / (62 + 1 / 20) + 
  (1.364 / 0.124) = 12 := by
  sorry

end complex_expression_evaluation_l210_21086


namespace intersection_properties_l210_21008

/-- Given a line y = a intersecting two curves, prove properties of intersection points -/
theorem intersection_properties (a : ℝ) (x₁ x₂ x₃ : ℝ) : 
  (∀ x, x/Real.exp x = a ↔ x = x₁ ∨ x = x₂) →  -- y = x/e^x intersects y = a at x₁ and x₂
  (∀ x, Real.log x/x = a ↔ x = x₂ ∨ x = x₃) →  -- y = ln(x)/x intersects y = a at x₂ and x₃
  0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ →                 -- order of x₁, x₂, x₃
  (x₂ = a * Real.exp x₂ ∧                      -- Statement A
   x₃ = Real.exp x₂ ∧                          -- Statement C
   x₁ + x₃ > 2 * x₂)                           -- Statement D
:= by sorry

end intersection_properties_l210_21008


namespace smallest_base_for_145_l210_21088

theorem smallest_base_for_145 :
  ∃ (b : ℕ), b = 12 ∧ 
  (∀ (n : ℕ), n^2 ≤ 145 ∧ 145 < n^3 → b ≤ n) :=
by sorry

end smallest_base_for_145_l210_21088


namespace joan_initial_balloons_l210_21020

/-- The number of balloons Joan lost -/
def lost_balloons : ℕ := 2

/-- The number of balloons Joan currently has -/
def current_balloons : ℕ := 6

/-- The initial number of balloons Joan had -/
def initial_balloons : ℕ := current_balloons + lost_balloons

theorem joan_initial_balloons : initial_balloons = 8 := by
  sorry

end joan_initial_balloons_l210_21020


namespace distance_calculation_l210_21066

/-- The distance between Cara's and Don's homes -/
def distance_between_homes : ℝ := 45

/-- Cara's walking speed in km/h -/
def cara_speed : ℝ := 6

/-- Don's walking speed in km/h -/
def don_speed : ℝ := 5

/-- The distance Cara walks before meeting Don in km -/
def cara_distance : ℝ := 30

/-- The time difference between Cara's and Don's start in hours -/
def time_difference : ℝ := 2

theorem distance_calculation :
  distance_between_homes = cara_distance + don_speed * (cara_distance / cara_speed - time_difference) :=
sorry

end distance_calculation_l210_21066


namespace triangle_area_proof_l210_21003

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define the side lengths
def SideLength (A B C : ℝ × ℝ) (a b c : ℝ) : Prop := 
  Triangle A B C ∧ 
  (a = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) ∧
  (b = Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) ∧
  (c = Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))

-- Define the angle C
def AngleC (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the area of the triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_proof 
  (A B C : ℝ × ℝ) (a b c : ℝ) 
  (h1 : SideLength A B C a b c)
  (h2 : b = 1)
  (h3 : c = Real.sqrt 3)
  (h4 : AngleC A B C = 2 * Real.pi / 3) :
  TriangleArea A B C = Real.sqrt 3 / 4 := by sorry

end triangle_area_proof_l210_21003


namespace largest_primary_divisor_l210_21059

/-- A positive integer is prime if it has exactly two positive divisors. -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A positive integer is a primary divisor if for every positive divisor d,
    at least one of d - 1 or d + 1 is prime. -/
def IsPrimaryDivisor (n : ℕ) : Prop :=
  n > 0 ∧ ∀ d : ℕ, d ∣ n → IsPrime (d - 1) ∨ IsPrime (d + 1)

/-- 48 is the largest primary divisor number. -/
theorem largest_primary_divisor : ∀ n : ℕ, IsPrimaryDivisor n → n ≤ 48 :=
  sorry

#check largest_primary_divisor

end largest_primary_divisor_l210_21059


namespace clock_movement_theorem_l210_21045

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ

/-- Represents a clock -/
structure Clock where
  startTime : Time
  degreeMoved : ℝ
  degreesPer12Hours : ℝ

/-- Calculates the ending time given a clock -/
def endingTime (c : Clock) : Time :=
  sorry

/-- The theorem to prove -/
theorem clock_movement_theorem (c : Clock) : 
  c.startTime = ⟨12, 0⟩ →
  c.degreeMoved = 74.99999999999999 →
  c.degreesPer12Hours = 360 →
  endingTime c = ⟨14, 30⟩ :=
sorry

end clock_movement_theorem_l210_21045


namespace angle_CAG_measure_l210_21063

-- Define the points
variable (A B C F G : ℝ × ℝ)

-- Define the properties of the configuration
def is_equilateral (A B C : ℝ × ℝ) : Prop := sorry

def is_rectangle (B C F G : ℝ × ℝ) : Prop := sorry

def shared_side (A B C F G : ℝ × ℝ) : Prop := sorry

def longer_side (B C F G : ℝ × ℝ) : Prop := sorry

-- Define the angle measure function
def angle_measure (P Q R : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_CAG_measure 
  (h1 : is_equilateral A B C)
  (h2 : is_rectangle B C F G)
  (h3 : shared_side A B C F G)
  (h4 : longer_side B C F G) :
  angle_measure C A G = 15 := by sorry

end angle_CAG_measure_l210_21063


namespace simplify_square_roots_l210_21026

theorem simplify_square_roots : Real.sqrt 81 - Real.sqrt 144 = -3 := by
  sorry

end simplify_square_roots_l210_21026


namespace simplify_and_evaluate_l210_21050

theorem simplify_and_evaluate (x : ℝ) (h : x ≠ 1) : 
  (2 / (x - 1) + 1) / ((x + 1) / (x^2 - 2*x + 1)) = x - 1 :=
by sorry

end simplify_and_evaluate_l210_21050


namespace third_rectangle_area_l210_21032

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: Given three rectangles forming a larger rectangle without gaps or overlaps,
    where two rectangles have dimensions 3 cm × 8 cm and 2 cm × 5 cm,
    the area of the third rectangle must be 4 cm². -/
theorem third_rectangle_area :
  ∀ (r1 r2 r3 : Rectangle),
  r1.length = 3 ∧ r1.width = 8 ∧
  r2.length = 2 ∧ r2.width = 5 →
  r1.area + r2.area + r3.area = (r1.area + r2.area) →
  r3.area = 4 := by
  sorry

#check third_rectangle_area

end third_rectangle_area_l210_21032


namespace equation_solutions_l210_21041

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (3 * x₁^2 - 4 * x₁ = 2 * x₁ ∧ x₁ = 0) ∧
                (3 * x₂^2 - 4 * x₂ = 2 * x₂ ∧ x₂ = 2)) ∧
  (∃ y₁ y₂ : ℝ, (y₁ * (y₁ + 8) = 16 ∧ y₁ = -4 + 4 * Real.sqrt 2) ∧
                (y₂ * (y₂ + 8) = 16 ∧ y₂ = -4 - 4 * Real.sqrt 2)) :=
by sorry

end equation_solutions_l210_21041


namespace square_formation_theorem_l210_21022

def sum_of_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

def can_form_square (n : ℕ) : Prop :=
  sum_of_natural_numbers n % 4 = 0

def min_breaks_for_square (n : ℕ) : ℕ :=
  let total := sum_of_natural_numbers n
  let remainder := total % 4
  if remainder = 0 then 0
  else if remainder = 1 || remainder = 3 then 1
  else 2

theorem square_formation_theorem :
  (min_breaks_for_square 12 = 2) ∧
  (can_form_square 15 = true) := by sorry

end square_formation_theorem_l210_21022


namespace perfect_matching_exists_l210_21073

/-- Represents a polygon with unit area -/
structure UnitPolygon where
  -- Add necessary fields here
  area : ℝ
  area_eq_one : area = 1

/-- Represents a square sheet of side length 2019 cut into 2019² unit polygons -/
structure Sheet where
  side_length : ℕ
  side_length_eq_2019 : side_length = 2019
  polygons : Finset UnitPolygon
  polygon_count : polygons.card = side_length * side_length

/-- Represents the intersection between two polygons from different sheets -/
def intersects (p1 p2 : UnitPolygon) : Prop :=
  sorry

/-- The main theorem -/
theorem perfect_matching_exists (sheet1 sheet2 : Sheet) : ∃ (matching : Finset (UnitPolygon × UnitPolygon)), 
  matching.card = 2019 * 2019 ∧ 
  (∀ (p1 p2 : UnitPolygon), (p1, p2) ∈ matching → p1 ∈ sheet1.polygons ∧ p2 ∈ sheet2.polygons ∧ intersects p1 p2) ∧
  (∀ p1 ∈ sheet1.polygons, ∃! p2, (p1, p2) ∈ matching) ∧
  (∀ p2 ∈ sheet2.polygons, ∃! p1, (p1, p2) ∈ matching) :=
sorry

end perfect_matching_exists_l210_21073


namespace three_lines_intersection_angles_l210_21080

-- Define a structure for a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define a structure for an intersection point
structure IntersectionPoint where
  point : ℝ × ℝ

-- Define a function to calculate the angle between two lines
def angleBetweenLines (l1 l2 : Line) : ℝ := sorry

-- Theorem statement
theorem three_lines_intersection_angles 
  (l1 l2 l3 : Line) 
  (p : IntersectionPoint) 
  (h1 : l1.point1 = p.point ∨ l1.point2 = p.point)
  (h2 : l2.point1 = p.point ∨ l2.point2 = p.point)
  (h3 : l3.point1 = p.point ∨ l3.point2 = p.point) :
  angleBetweenLines l1 l2 = 120 ∧ 
  angleBetweenLines l2 l3 = 120 ∧ 
  angleBetweenLines l3 l1 = 120 := by sorry

end three_lines_intersection_angles_l210_21080


namespace largest_subset_sine_inequality_l210_21052

theorem largest_subset_sine_inequality :
  ∀ y ∈ Set.Icc 0 Real.pi, ∀ x ∈ Set.Icc 0 Real.pi,
  Real.sin (x + y) ≤ Real.sin x + Real.sin y :=
by sorry

end largest_subset_sine_inequality_l210_21052


namespace N_subset_M_l210_21092

-- Define the sets M and N
def M : Set ℝ := {x | x < 9}
def N : Set ℝ := {x | x^2 < 9}

-- State the theorem
theorem N_subset_M : N ⊆ M := by
  sorry

end N_subset_M_l210_21092


namespace walking_problem_l210_21070

/-- The problem of two people walking towards each other on a road -/
theorem walking_problem (total_distance : ℝ) (yolanda_speed : ℝ) (bob_speed : ℝ) 
  (head_start : ℝ) :
  total_distance = 40 ∧ 
  yolanda_speed = 2 ∧ 
  bob_speed = 4 ∧ 
  head_start = 1 →
  ∃ (meeting_time : ℝ),
    meeting_time > 0 ∧
    head_start * yolanda_speed + meeting_time * yolanda_speed + meeting_time * bob_speed = total_distance ∧
    meeting_time * bob_speed = 25 + 1/3 :=
by sorry

end walking_problem_l210_21070


namespace complex_sum_theorem_l210_21058

theorem complex_sum_theorem (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^3 = 1 →
  ω ≠ 1 →
  (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω →
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 3 := by
  sorry

end complex_sum_theorem_l210_21058
