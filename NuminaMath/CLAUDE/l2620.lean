import Mathlib

namespace total_repair_cost_is_4850_l2620_262010

-- Define the repair costs
def engine_labor_rate : ℕ := 75
def engine_labor_hours : ℕ := 16
def engine_part_cost : ℕ := 1200

def brake_labor_rate : ℕ := 85
def brake_labor_hours : ℕ := 10
def brake_part_cost : ℕ := 800

def tire_labor_rate : ℕ := 50
def tire_labor_hours : ℕ := 4
def tire_part_cost : ℕ := 600

-- Define the total repair cost function
def total_repair_cost : ℕ :=
  (engine_labor_rate * engine_labor_hours + engine_part_cost) +
  (brake_labor_rate * brake_labor_hours + brake_part_cost) +
  (tire_labor_rate * tire_labor_hours + tire_part_cost)

-- Theorem statement
theorem total_repair_cost_is_4850 : total_repair_cost = 4850 := by
  sorry

end total_repair_cost_is_4850_l2620_262010


namespace power_sum_prime_l2620_262075

theorem power_sum_prime (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → (2^p + 3^p = a^n) → n = 1 := by
  sorry

end power_sum_prime_l2620_262075


namespace lcm_of_6_and_15_l2620_262081

theorem lcm_of_6_and_15 : Nat.lcm 6 15 = 30 := by
  sorry

end lcm_of_6_and_15_l2620_262081


namespace abc_product_l2620_262045

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 168) (h2 : b * (c + a) = 153) (h3 : c * (a + b) = 147) :
  a * b * c = 720 := by
sorry

end abc_product_l2620_262045


namespace floor_product_equation_l2620_262007

theorem floor_product_equation : ∃! (x : ℝ), x > 0 ∧ x * ⌊x⌋ = 50 ∧ |x - (50 / 7)| < 0.000001 := by
  sorry

end floor_product_equation_l2620_262007


namespace two_p_plus_q_l2620_262080

theorem two_p_plus_q (p q : ℚ) (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 := by
  sorry

end two_p_plus_q_l2620_262080


namespace total_players_count_l2620_262033

/-- The number of players who play kabadi -/
def kabadi_players : ℕ := 10

/-- The number of players who play kho kho only -/
def kho_kho_only_players : ℕ := 40

/-- The number of players who play both games -/
def both_games_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := kabadi_players + kho_kho_only_players - both_games_players

theorem total_players_count : total_players = 45 := by sorry

end total_players_count_l2620_262033


namespace paint_combinations_l2620_262087

theorem paint_combinations (num_colors num_methods : ℕ) :
  num_colors = 5 → num_methods = 4 → num_colors * num_methods = 20 := by
  sorry

end paint_combinations_l2620_262087


namespace conference_theorem_l2620_262046

/-- A graph with vertices labeled 1 to n, where edges are colored either red or blue -/
structure ColoredGraph (n : ℕ) where
  edge_color : Fin n → Fin n → Bool

/-- Predicate to check if a subgraph of 4 vertices satisfies the given conditions -/
def valid_subgraph (G : ColoredGraph n) (a b c d : Fin n) : Prop :=
  let edges := [G.edge_color a b, G.edge_color a c, G.edge_color a d, 
                G.edge_color b c, G.edge_color b d, G.edge_color c d]
  let red_count := (edges.filter id).length
  let blue_count := (edges.filter not).length
  (red_count + blue_count) % 2 = 0 ∧ 
  red_count > 0 ∧ 
  (blue_count = 0 ∨ blue_count ≥ red_count)

/-- Theorem statement -/
theorem conference_theorem :
  ∃ (G : ColoredGraph 2017),
    (∀ (a b c d : Fin 2017), valid_subgraph G a b c d) →
    ∃ (S : Finset (Fin 2017)),
      S.card = 673 ∧
      ∀ (x y : Fin 2017), x ∈ S → y ∈ S → x ≠ y → G.edge_color x y = true :=
by sorry

end conference_theorem_l2620_262046


namespace choir_robe_cost_l2620_262036

/-- Calculates the cost of buying additional robes for a school choir. -/
theorem choir_robe_cost (total_robes : ℕ) (existing_robes : ℕ) (cost_per_robe : ℕ) : 
  total_robes = 30 → existing_robes = 12 → cost_per_robe = 2 → 
  (total_robes - existing_robes) * cost_per_robe = 36 :=
by
  sorry

#check choir_robe_cost

end choir_robe_cost_l2620_262036


namespace digit_multiplication_puzzle_l2620_262097

def is_single_digit (n : ℕ) : Prop := n < 10

def is_five_digit_number (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def number_from_digits (a b c d e : ℕ) : ℕ := a * 10000 + b * 1000 + c * 100 + d * 10 + e

theorem digit_multiplication_puzzle :
  ∀ (a b c d e : ℕ),
    is_single_digit a ∧
    is_single_digit b ∧
    is_single_digit c ∧
    is_single_digit d ∧
    is_single_digit e ∧
    is_five_digit_number (number_from_digits a b c d e) ∧
    4 * (number_from_digits a b c d e) = number_from_digits e d c b a →
    a = 2 ∧ b = 1 ∧ c = 9 ∧ d = 7 ∧ e = 8 :=
by sorry

end digit_multiplication_puzzle_l2620_262097


namespace roots_sum_of_sixth_powers_l2620_262001

theorem roots_sum_of_sixth_powers (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 7 + 1 = 0 →
  s^2 - 2*s*Real.sqrt 7 + 1 = 0 →
  r^6 + s^6 = 389374 := by
  sorry

end roots_sum_of_sixth_powers_l2620_262001


namespace boat_distance_proof_l2620_262025

/-- The distance covered by a boat traveling downstream -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

theorem boat_distance_proof (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ)
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 5)
  (h3 : time = 8) :
  distance_downstream boat_speed stream_speed time = 168 := by
  sorry

#eval distance_downstream 16 5 8

end boat_distance_proof_l2620_262025


namespace narcissistic_numbers_l2620_262067

theorem narcissistic_numbers : 
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ 
    n = (n / 100)^3 + ((n % 100) / 10)^3 + (n % 10)^3} = 
  {153, 370, 371, 407} := by
sorry

end narcissistic_numbers_l2620_262067


namespace arithmetic_sequence_property_l2620_262050

/-- An arithmetic sequence is a sequence where the difference between 
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 :=
sorry

end arithmetic_sequence_property_l2620_262050


namespace base_b_sum_product_l2620_262095

/-- Given a base b, this function converts a number from base b to decimal --/
def toDecimal (b : ℕ) (n : ℕ) : ℕ := sorry

/-- Given a base b, this function converts a decimal number to base b --/
def fromDecimal (b : ℕ) (n : ℕ) : ℕ := sorry

/-- Theorem stating the relationship between the given product and sum in base b --/
theorem base_b_sum_product (b : ℕ) : 
  (toDecimal b 14) * (toDecimal b 17) * (toDecimal b 18) = toDecimal b 6274 →
  (toDecimal b 14) + (toDecimal b 17) + (toDecimal b 18) = 49 := by
  sorry

end base_b_sum_product_l2620_262095


namespace exists_common_divisor_l2620_262059

/-- A function from positive integers to integers greater than or equal to 2 -/
def PositiveIntegerFunction := ℕ+ → ℕ

/-- The property that f(m+n) divides f(m) + f(n) for all positive integers m and n -/
def HasDivisibilityProperty (f : PositiveIntegerFunction) : Prop :=
  ∀ m n : ℕ+, (f (m + n) : ℤ) ∣ (f m + f n : ℤ)

/-- The main theorem -/
theorem exists_common_divisor
  (f : PositiveIntegerFunction)
  (h1 : ∀ n : ℕ+, f n ≥ 2)
  (h2 : HasDivisibilityProperty f) :
  ∃ c : ℕ+, c > 1 ∧ ∀ n : ℕ+, (c : ℤ) ∣ (f n : ℤ) :=
sorry

end exists_common_divisor_l2620_262059


namespace geometric_sequence_sum_l2620_262018

/-- A geometric sequence is a sequence where the ratio of any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem states that for a geometric sequence satisfying certain conditions, 
    the sum of specific terms equals 3. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h1 : a 1 + a 3 = 8) 
    (h2 : a 5 + a 7 = 4) : 
  a 9 + a 11 + a 13 + a 15 = 3 := by
  sorry

end geometric_sequence_sum_l2620_262018


namespace composite_function_difference_l2620_262037

theorem composite_function_difference (A B : ℝ) (h : A ≠ B) :
  let f := λ x : ℝ => A * x + B
  let g := λ x : ℝ => B * x + A
  (∀ x, f (g x) - g (f x) = 2 * (B - A)) →
  A + B = -2 := by
sorry

end composite_function_difference_l2620_262037


namespace pedal_triangle_area_l2620_262012

/-- Given a triangle with area S and circumradius R, and a point at distance d from the circumcenter,
    S₁ is the area of the triangle formed by the feet of the perpendiculars from this point
    to the sides of the original triangle. -/
theorem pedal_triangle_area (S R d S₁ : ℝ) (h_pos_S : S > 0) (h_pos_R : R > 0) :
  S₁ = (S / 4) * |1 - (d^2 / R^2)| := by
  sorry

end pedal_triangle_area_l2620_262012


namespace complex_arithmetic_simplification_l2620_262084

theorem complex_arithmetic_simplification :
  ((6 - 3 * Complex.I) - (2 + 4 * Complex.I)) * (2 * Complex.I) = 14 + 8 * Complex.I :=
by sorry

end complex_arithmetic_simplification_l2620_262084


namespace original_number_l2620_262027

theorem original_number (x : ℝ) : 
  (x * 10) * 0.001 = 0.375 → x = 37.5 := by
sorry

end original_number_l2620_262027


namespace hannah_restaurant_bill_hannah_restaurant_bill_proof_l2620_262029

/-- The total amount Hannah spent on the entree and dessert is $23, 
    given that the entree costs $14 and it is $5 more than the dessert. -/
theorem hannah_restaurant_bill : ℕ → ℕ → ℕ → Prop :=
  fun entree_cost dessert_cost total_cost =>
    (entree_cost = 14) →
    (entree_cost = dessert_cost + 5) →
    (total_cost = entree_cost + dessert_cost) →
    (total_cost = 23)

/-- Proof of hannah_restaurant_bill -/
theorem hannah_restaurant_bill_proof : hannah_restaurant_bill 14 9 23 := by
  sorry

end hannah_restaurant_bill_hannah_restaurant_bill_proof_l2620_262029


namespace sphere_surface_area_from_cube_l2620_262044

theorem sphere_surface_area_from_cube (a : ℝ) (h : a > 0) :
  ∃ (cube_edge : ℝ) (sphere_radius : ℝ),
    cube_edge > 0 ∧
    sphere_radius > 0 ∧
    (6 * cube_edge ^ 2 = a) ∧
    (cube_edge * Real.sqrt 3 = 2 * sphere_radius) ∧
    (4 * π * sphere_radius ^ 2 = π / 2 * a) :=
by sorry

end sphere_surface_area_from_cube_l2620_262044


namespace percentage_difference_l2620_262083

theorem percentage_difference : (0.55 * 40) - (4/5 * 25) = 2 := by
  sorry

end percentage_difference_l2620_262083


namespace larrys_to_keiths_score_ratio_l2620_262031

/-- Given that Keith scored 3 points, Danny scored 5 more marks than Larry,
    and the total amount of marks scored by the three students is 26,
    prove that the ratio of Larry's score to Keith's score is 3:1 -/
theorem larrys_to_keiths_score_ratio (keith_score larry_score danny_score : ℕ) : 
  keith_score = 3 →
  danny_score = larry_score + 5 →
  keith_score + larry_score + danny_score = 26 →
  larry_score / keith_score = 3 / 1 := by
sorry

end larrys_to_keiths_score_ratio_l2620_262031


namespace water_speed_calculation_l2620_262061

/-- A person's swimming speed in still water (in km/h) -/
def still_water_speed : ℝ := 4

/-- The time taken to swim against the current (in hours) -/
def time_against_current : ℝ := 3

/-- The distance swum against the current (in km) -/
def distance_against_current : ℝ := 6

/-- The speed of the water (in km/h) -/
def water_speed : ℝ := 2

theorem water_speed_calculation :
  (distance_against_current = (still_water_speed - water_speed) * time_against_current) →
  water_speed = 2 := by
sorry

end water_speed_calculation_l2620_262061


namespace cereal_eating_time_l2620_262066

/-- The time taken for three people to eat a certain amount of cereal together -/
def time_to_eat (fat_rate thin_rate medium_rate total_cereal : ℚ) : ℚ :=
  total_cereal / (fat_rate + thin_rate + medium_rate)

theorem cereal_eating_time :
  let fat_rate : ℚ := 1 / 15
  let thin_rate : ℚ := 1 / 35
  let medium_rate : ℚ := 1 / 25
  let total_cereal : ℚ := 5
  time_to_eat fat_rate thin_rate medium_rate total_cereal = 2625 / 71 :=
by sorry

end cereal_eating_time_l2620_262066


namespace escalator_walking_rate_l2620_262060

/-- Calculates the walking rate of a person on an escalator -/
theorem escalator_walking_rate 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (time_taken : ℝ) 
  (h1 : escalator_speed = 15)
  (h2 : escalator_length = 180)
  (h3 : time_taken = 10) :
  (escalator_length / time_taken) - escalator_speed = 3 := by
  sorry

end escalator_walking_rate_l2620_262060


namespace cylinder_water_properties_l2620_262086

/-- Represents a cylindrical tank lying on its side -/
structure HorizontalCylinder where
  radius : ℝ
  height : ℝ

/-- Represents the water level in the tank -/
def WaterLevel : ℝ := 3

/-- The volume of water in the cylindrical tank -/
def waterVolume (c : HorizontalCylinder) (h : ℝ) : ℝ := sorry

/-- The submerged surface area of the cylindrical side of the tank -/
def submergedSurfaceArea (c : HorizontalCylinder) (h : ℝ) : ℝ := sorry

theorem cylinder_water_properties :
  let c : HorizontalCylinder := { radius := 5, height := 10 }
  (waterVolume c WaterLevel = 290.7 * Real.pi - 40 * Real.sqrt 6) ∧
  (submergedSurfaceArea c WaterLevel = 91.5) := by sorry

end cylinder_water_properties_l2620_262086


namespace sum_of_squares_l2620_262088

theorem sum_of_squares (x y : ℚ) (h1 : x + 2*y = 20) (h2 : 3*x + y = 19) : x^2 + y^2 = 401/5 := by
  sorry

end sum_of_squares_l2620_262088


namespace increasing_derivative_relation_l2620_262072

open Set
open Function
open Real

-- Define the interval (a, b)
variable (a b : ℝ) (hab : a < b)

-- Define a real-valued function on the interval (a, b)
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on (a, b)
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Define the derivative of f
variable (f' : ℝ → ℝ)
variable (hf' : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)

-- State the theorem
theorem increasing_derivative_relation :
  (∀ x ∈ Ioo a b, f' x > 0 → IsIncreasing f a b) ∧
  ∃ f : ℝ → ℝ, IsIncreasing f a b ∧ ¬(∀ x ∈ Ioo a b, f' x > 0) :=
sorry

end increasing_derivative_relation_l2620_262072


namespace cubic_polynomials_l2620_262020

-- Define the polynomials A and B
def A (x : ℝ) : ℝ := 5 * x^3 - 6 * x^2 + 10
def B (x e f : ℝ) : ℝ := x^2 + e * x + f

-- Define the alternative form of A
def A_alt (x a b c d : ℝ) : ℝ := a * (x - 1)^3 + b * (x - 1)^2 + c * (x - 1) + d

-- State the theorem
theorem cubic_polynomials (a b c d e f : ℝ) (hf : f ≠ 0) (he : e ≠ 0) :
  (∀ x, A x = A_alt x a b c d) →
  (∀ x, ∃ k₁ k₂ k₃, A x + B x e f = k₁ * x^3 + k₂ * x^2 + k₃ * x + (10 + f)) →
  (a + b + c = 17) ∧
  (∃ x₀, ∀ x, B x e f = 0 ↔ x = x₀) →
  (f = -10 ∧ a + b + c = 17 ∧ e^2 = 4 * f) := by
sorry

end cubic_polynomials_l2620_262020


namespace male_students_count_l2620_262000

/-- Represents the number of students in a grade. -/
def total_students : ℕ := 800

/-- Represents the size of the stratified sample. -/
def sample_size : ℕ := 20

/-- Represents the number of female students in the sample. -/
def females_in_sample : ℕ := 8

/-- Calculates the number of male students in the entire grade based on stratified sampling. -/
def male_students_in_grade : ℕ := 
  (total_students * (sample_size - females_in_sample)) / sample_size

/-- Theorem stating that the number of male students in the grade is 480. -/
theorem male_students_count : male_students_in_grade = 480 := by sorry

end male_students_count_l2620_262000


namespace simplify_square_root_l2620_262004

theorem simplify_square_root : 
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by sorry

end simplify_square_root_l2620_262004


namespace john_travel_money_l2620_262049

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after buying a ticket -/
def remainingMoney (savings : ℕ) (ticketCost : ℕ) : ℕ :=
  base8ToBase10 savings - ticketCost

theorem john_travel_money :
  remainingMoney 5555 1200 = 1725 := by sorry

end john_travel_money_l2620_262049


namespace problem_solution_l2620_262065

theorem problem_solution (a b : ℚ) 
  (h1 : 5 + a = 3 - b) 
  (h2 : 3 + b = 8 + a) : 
  5 - a = 17 / 2 := by
sorry

end problem_solution_l2620_262065


namespace fred_initial_cards_l2620_262034

theorem fred_initial_cards (cards_bought cards_left : ℕ) : 
  cards_bought = 3 → cards_left = 2 → cards_bought + cards_left = 5 :=
by sorry

end fred_initial_cards_l2620_262034


namespace system_solution_l2620_262082

theorem system_solution : ∃ (x y z : ℤ), 
  (7*x + 3*y = 2*z + 1) ∧ 
  (4*x - 5*y = 3*z - 30) ∧ 
  (x + 2*y = 5*z + 15) ∧ 
  (x = -1) ∧ (y = 2) ∧ (z = 7) := by
  sorry

#check system_solution

end system_solution_l2620_262082


namespace triangle_angle_relation_l2620_262079

/-- Given a triangle ABC with B > A, prove that C₁ - C₂ = B - A,
    where C₁ and C₂ are parts of angle C divided by the altitude,
    and C₂ is adjacent to side a. -/
theorem triangle_angle_relation (A B C C₁ C₂ : Real) : 
  B > A → 
  C = C₁ + C₂ → 
  A + B + C = Real.pi → 
  C₁ - C₂ = B - A := by
  sorry

end triangle_angle_relation_l2620_262079


namespace talia_age_in_seven_years_l2620_262028

/-- Proves Talia's age in seven years given the conditions of the problem -/
theorem talia_age_in_seven_years :
  ∀ (talia_age mom_age dad_age : ℕ),
    mom_age = 3 * talia_age →
    dad_age + 3 = mom_age →
    dad_age = 36 →
    talia_age + 7 = 20 := by
  sorry

end talia_age_in_seven_years_l2620_262028


namespace restaurant_table_difference_l2620_262006

/-- Represents the number of tables and seating capacity in a restaurant --/
structure Restaurant where
  new_tables : ℕ
  original_tables : ℕ
  new_table_capacity : ℕ
  original_table_capacity : ℕ

/-- Calculates the total number of tables in the restaurant --/
def Restaurant.total_tables (r : Restaurant) : ℕ :=
  r.new_tables + r.original_tables

/-- Calculates the total seating capacity of the restaurant --/
def Restaurant.total_capacity (r : Restaurant) : ℕ :=
  r.new_tables * r.new_table_capacity + r.original_tables * r.original_table_capacity

/-- Theorem stating the difference between new and original tables --/
theorem restaurant_table_difference (r : Restaurant) 
  (h1 : r.total_tables = 40)
  (h2 : r.total_capacity = 212)
  (h3 : r.new_table_capacity = 6)
  (h4 : r.original_table_capacity = 4) :
  r.new_tables - r.original_tables = 12 := by
  sorry


end restaurant_table_difference_l2620_262006


namespace intersection_of_A_and_B_l2620_262052

def A : Set ℤ := {1, 3}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {3} := by
  sorry

end intersection_of_A_and_B_l2620_262052


namespace circle_symmetry_l2620_262039

-- Define the line l: x + y = 0
def line_l (x y : ℝ) : Prop := x + y = 0

-- Define circle C: (x-2)^2 + (y-1)^2 = 4
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Define circle C': (x+1)^2 + (y+2)^2 = 4
def circle_C' (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 4

-- Function to reflect a point (x, y) across the line l
def reflect_point (x y : ℝ) : ℝ × ℝ := (-y, -x)

-- Theorem stating that C' is symmetric to C with respect to l
theorem circle_symmetry :
  ∀ x y : ℝ, circle_C x y ↔ circle_C' (reflect_point x y).1 (reflect_point x y).2 :=
sorry

end circle_symmetry_l2620_262039


namespace ellipse_equation_l2620_262015

/-- An ellipse with center at the origin, a focus on a coordinate axis,
    eccentricity √3/2, and passing through (2,0) -/
structure Ellipse where
  -- The focus is either on the x-axis or y-axis
  focus_on_axis : Bool
  -- The equation of the ellipse in the form x²/a² + y²/b² = 1
  a : ℝ
  b : ℝ
  -- Conditions
  center_origin : a > 0 ∧ b > 0
  passes_through_2_0 : (2 : ℝ)^2 / a^2 + 0^2 / b^2 = 1
  eccentricity : Real.sqrt (1 - b^2 / a^2) = Real.sqrt 3 / 2

/-- The equation of the ellipse is either x²/4 + y² = 1 or x²/4 + y²/16 = 1 -/
theorem ellipse_equation (e : Ellipse) :
  (e.a^2 = 4 ∧ e.b^2 = 1) ∨ (e.a^2 = 16 ∧ e.b^2 = 4) := by
  sorry

end ellipse_equation_l2620_262015


namespace product_digit_count_l2620_262090

def x : ℕ := 3659893456789325678
def y : ℕ := 342973489379256

theorem product_digit_count :
  (String.length (toString (x * y))) = 34 := by
  sorry

end product_digit_count_l2620_262090


namespace fraction_equality_l2620_262023

theorem fraction_equality (a b c d : ℚ) 
  (h1 : b / a = 1 / 2)
  (h2 : d / c = 1 / 2)
  (h3 : a ≠ c) :
  (2 * b - d) / (2 * a - c) = 1 / 2 := by
  sorry

end fraction_equality_l2620_262023


namespace tan_5040_degrees_equals_zero_l2620_262096

theorem tan_5040_degrees_equals_zero : Real.tan (5040 * π / 180) = 0 := by
  sorry

end tan_5040_degrees_equals_zero_l2620_262096


namespace product_of_integers_l2620_262035

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 20)
  (diff_squares_eq : x^2 - y^2 = 40) :
  x * y = 99 := by
  sorry

end product_of_integers_l2620_262035


namespace problem_solution_l2620_262009

theorem problem_solution :
  ∀ m n : ℕ+,
  (m : ℝ)^2 - (n : ℝ) = 32 →
  (∃ x : ℝ, x = (m + n^(1/2))^(1/5) + (m - n^(1/2))^(1/5) ∧ x^5 - 10*x^3 + 20*x - 40 = 0) →
  (m : ℕ) + n = 388 := by
sorry

end problem_solution_l2620_262009


namespace sum_of_specific_terms_l2620_262094

/-- An arithmetic sequence with a_5 = 15 -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ a 5 = 15

/-- Theorem: In an arithmetic sequence where a_5 = 15, the sum of a_2, a_4, a_6, and a_8 is 60 -/
theorem sum_of_specific_terms (a : ℕ → ℝ) (h : arithmeticSequence a) :
  a 2 + a 4 + a 6 + a 8 = 60 := by
  sorry

end sum_of_specific_terms_l2620_262094


namespace product_of_numbers_with_given_hcf_lcm_l2620_262021

theorem product_of_numbers_with_given_hcf_lcm :
  ∀ (a b : ℕ+),
  Nat.gcd a b = 33 →
  Nat.lcm a b = 2574 →
  a * b = 84942 :=
by sorry

end product_of_numbers_with_given_hcf_lcm_l2620_262021


namespace smallest_b_factorization_l2620_262091

/-- The smallest positive integer b for which x^2 + bx + 2304 factors into a product of two polynomials with integer coefficients -/
def smallest_factorizable_b : ℕ := 96

/-- Predicate to check if a polynomial factors with integer coefficients -/
def factors_with_integer_coeffs (a b c : ℤ) : Prop :=
  ∃ (p q : ℤ), ∀ (x : ℤ), a * x^2 + b * x + c = (x + p) * (x + q)

theorem smallest_b_factorization :
  (factors_with_integer_coeffs 1 smallest_factorizable_b 2304) ∧
  (∀ b : ℕ, b < smallest_factorizable_b →
    ¬(factors_with_integer_coeffs 1 b 2304)) :=
by sorry

end smallest_b_factorization_l2620_262091


namespace student_arrangements_l2620_262070

/-- Represents a student with a unique height -/
structure Student :=
  (height : ℕ)

/-- The set of 7 students with different heights -/
def Students : Finset Student :=
  sorry

/-- Predicate for a valid arrangement in a row -/
def ValidRowArrangement (arrangement : List Student) : Prop :=
  sorry

/-- Predicate for a valid arrangement in two rows and three columns -/
def Valid2x3Arrangement (arrangement : List (List Student)) : Prop :=
  sorry

theorem student_arrangements :
  (∃ (arrangements : Finset (List Student)),
    (∀ arr ∈ arrangements, ValidRowArrangement arr) ∧
    Finset.card arrangements = 20) ∧
  (∃ (arrangements : Finset (List (List Student))),
    (∀ arr ∈ arrangements, Valid2x3Arrangement arr) ∧
    Finset.card arrangements = 630) :=
  sorry

end student_arrangements_l2620_262070


namespace workers_problem_l2620_262032

/-- Given a number of workers that can complete a job in 25 days, 
    and adding 10 workers reduces the time to 15 days, 
    prove that the original number of workers is 15. -/
theorem workers_problem (W : ℕ) : 
  W * 25 = (W + 10) * 15 → W = 15 := by sorry

end workers_problem_l2620_262032


namespace ap_triangle_centroid_incenter_parallel_l2620_262053

/-- A triangle with sides in arithmetic progression -/
structure APTriangle where
  a : ℝ
  b : ℝ
  hab : a ≠ b
  hab_pos : 0 < a ∧ 0 < b

/-- The centroid of a triangle -/
def centroid (t : APTriangle) : ℝ × ℝ := sorry

/-- The incenter of a triangle -/
def incenter (t : APTriangle) : ℝ × ℝ := sorry

/-- Two lines are parallel -/
def parallel (l1 l2 : ℝ × ℝ → ℝ × ℝ → Prop) : Prop := sorry

/-- The line passing through two points -/
def line_through (p1 p2 : ℝ × ℝ) : ℝ × ℝ → ℝ × ℝ → Prop := sorry

/-- The side AB of the triangle -/
def side_AB (t : APTriangle) : ℝ × ℝ → ℝ × ℝ → Prop := sorry

theorem ap_triangle_centroid_incenter_parallel (t : APTriangle) :
  parallel (line_through (centroid t) (incenter t)) (side_AB t) := by
  sorry

end ap_triangle_centroid_incenter_parallel_l2620_262053


namespace circles_have_three_common_tangents_l2620_262073

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 9

-- Define the centers and radii
def center1 : ℝ × ℝ := (-1, -2)
def center2 : ℝ × ℝ := (2, 2)
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem statement
theorem circles_have_three_common_tangents :
  ∃! (n : ℕ), n = 3 ∧ 
  (∃ (tangents : Finset (ℝ → ℝ)), tangents.card = n ∧ 
    (∀ f ∈ tangents, ∀ x y : ℝ, 
      (circle1 x y → (y = f x ∨ y = -f x)) ∧ 
      (circle2 x y → (y = f x ∨ y = -f x)))) := by sorry

end circles_have_three_common_tangents_l2620_262073


namespace complex_power_30_150_deg_l2620_262062

theorem complex_power_30_150_deg : (Complex.exp (Complex.I * Real.pi * (5/6)))^30 = -1 := by
  sorry

end complex_power_30_150_deg_l2620_262062


namespace polynomial_positive_intervals_l2620_262077

/-- The polynomial (x+1)(x-1)(x-3) is positive if and only if x is in the interval (-1, 1) or (3, ∞) -/
theorem polynomial_positive_intervals (x : ℝ) : 
  (x + 1) * (x - 1) * (x - 3) > 0 ↔ (x > -1 ∧ x < 1) ∨ x > 3 := by
  sorry

end polynomial_positive_intervals_l2620_262077


namespace function_symmetry_l2620_262003

/-- The function f(x) defined as √3 sin(2x) + 2 cos²x is symmetric about the line x = π/6 -/
theorem function_symmetry (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2
  f (π / 6 + x) = f (π / 6 - x) :=
by sorry

end function_symmetry_l2620_262003


namespace existence_of_special_integers_l2620_262014

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  ((7^7 : ℕ) ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) ∧
  (a.val = 18 ∧ b.val = 1) := by
sorry

end existence_of_special_integers_l2620_262014


namespace smallest_divisible_by_10_11_18_l2620_262011

theorem smallest_divisible_by_10_11_18 : ∃ n : ℕ+, (∀ m : ℕ+, 10 ∣ m ∧ 11 ∣ m ∧ 18 ∣ m → n ≤ m) ∧ 10 ∣ n ∧ 11 ∣ n ∧ 18 ∣ n :=
  by sorry

end smallest_divisible_by_10_11_18_l2620_262011


namespace square_area_subtraction_l2620_262024

theorem square_area_subtraction (s : ℝ) (x : ℝ) : 
  s = 4 → s^2 + s - x = 4 → x = 16 := by sorry

end square_area_subtraction_l2620_262024


namespace p_iff_between_two_and_three_l2620_262093

def p (x : ℝ) : Prop := x^2 - 5*x + 6 < 0

theorem p_iff_between_two_and_three :
  ∀ x : ℝ, p x ↔ 2 < x ∧ x < 3 :=
by sorry

end p_iff_between_two_and_three_l2620_262093


namespace vector_representation_l2620_262058

-- Define points A, B, and Q in a 2D plane
variable (A B Q : ℝ × ℝ)

-- Define the ratio condition
def ratio_condition (A B Q : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ Q.1 - A.1 = 7*k ∧ Q.1 - B.1 = 2*k ∧
              Q.2 - A.2 = 7*k ∧ Q.2 - B.2 = 2*k

-- Define vector addition and scalar multiplication
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def scalar_mul (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)

-- Theorem statement
theorem vector_representation (A B Q : ℝ × ℝ) 
  (h : ratio_condition A B Q) :
  Q = vec_add (scalar_mul (-2/5) A) B :=
sorry

end vector_representation_l2620_262058


namespace solve_for_y_l2620_262038

theorem solve_for_y (x y : ℝ) (h1 : x + 2 * y = 20) (h2 : x = 10) : y = 5 := by
  sorry

end solve_for_y_l2620_262038


namespace committee_formation_count_l2620_262055

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of people in the club -/
def total_people : ℕ := 12

/-- The number of board members -/
def board_members : ℕ := 3

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- The number of regular members (non-board members) -/
def regular_members : ℕ := total_people - board_members

theorem committee_formation_count :
  choose total_people committee_size - choose regular_members committee_size = 666 := by
  sorry

end committee_formation_count_l2620_262055


namespace net_population_increase_per_day_l2620_262022

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℚ := 5

/-- Represents the death rate in people per two seconds -/
def death_rate : ℚ := 3

/-- Calculates the net population increase per second -/
def net_increase_per_second : ℚ := (birth_rate - death_rate) / 2

/-- Theorem stating the net population increase over one day -/
theorem net_population_increase_per_day :
  (net_increase_per_second * seconds_per_day : ℚ) = 86400 := by
  sorry

end net_population_increase_per_day_l2620_262022


namespace disjoint_sets_cardinality_relation_l2620_262076

theorem disjoint_sets_cardinality_relation (a b : ℕ+) (A B : Finset ℤ) :
  Disjoint A B →
  (∀ i : ℤ, i ∈ A ∪ B → (i + a) ∈ A ∨ (i - b) ∈ B) →
  a * A.card = b * B.card := by
  sorry

end disjoint_sets_cardinality_relation_l2620_262076


namespace characterization_of_f_l2620_262078

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the conditions
def NonNegative (f : RealFunction) : Prop :=
  ∀ x : ℝ, f x ≥ 0

def SatisfiesEquation (f : RealFunction) : Prop :=
  ∀ a b c d : ℝ, a * b + b * c + c * d = 0 →
    f (a - b) + f (c - d) = f a + f (b + c) + f d

-- Main theorem
theorem characterization_of_f (f : RealFunction)
  (h1 : NonNegative f)
  (h2 : SatisfiesEquation f) :
  ∃ c : ℝ, c ≥ 0 ∧ ∀ x : ℝ, f x = c * x^2 :=
sorry

end characterization_of_f_l2620_262078


namespace book_profit_percentage_l2620_262085

/-- Given a book with cost price $1800, if selling it for $90 more than the initial
    selling price would result in a 15% profit, then the initial profit percentage is 10% -/
theorem book_profit_percentage (cost_price : ℝ) (additional_price : ℝ) 
  (higher_profit_percentage : ℝ) (initial_selling_price : ℝ) :
  cost_price = 1800 →
  additional_price = 90 →
  higher_profit_percentage = 15 →
  initial_selling_price + additional_price = cost_price * (1 + higher_profit_percentage / 100) →
  (initial_selling_price - cost_price) / cost_price * 100 = 10 := by
  sorry

end book_profit_percentage_l2620_262085


namespace xyz_value_l2620_262030

theorem xyz_value (x y z : ℝ) 
  (eq1 : (x + y + z) * (x * y + x * z + y * z) = 35)
  (eq2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 23 / 3 := by sorry

end xyz_value_l2620_262030


namespace solve_for_a_l2620_262051

theorem solve_for_a (a : ℝ) :
  (∀ x, |2*x - a| + a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) →
  a = 2 :=
by sorry

end solve_for_a_l2620_262051


namespace unique_solution_quadratic_inequality_l2620_262016

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x, -3 ≤ x^2 - 2*a*x + a ∧ x^2 - 2*a*x + a ≤ -2) → (a = 2 ∨ a = -1) :=
by sorry

end unique_solution_quadratic_inequality_l2620_262016


namespace world_book_day_purchase_l2620_262071

theorem world_book_day_purchase (planned_spending : ℝ) (price_reduction : ℝ) (additional_books_ratio : ℝ) :
  planned_spending = 180 →
  price_reduction = 9 →
  additional_books_ratio = 1/4 →
  ∃ (planned_books actual_books : ℝ),
    planned_books > 0 ∧
    actual_books = planned_books * (1 + additional_books_ratio) ∧
    planned_spending / planned_books - planned_spending / actual_books = price_reduction ∧
    actual_books = 5 := by
  sorry

end world_book_day_purchase_l2620_262071


namespace max_sum_of_square_roots_l2620_262002

theorem max_sum_of_square_roots (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 2013) : 
  Real.sqrt (3 * a + 12) + Real.sqrt (3 * b + 12) + Real.sqrt (3 * c + 12) ≤ 135 := by
sorry

end max_sum_of_square_roots_l2620_262002


namespace cost_of_dozen_pens_l2620_262017

/-- Given the cost of 3 pens and 5 pencils, and the cost ratio of pen to pencil,
    calculate the cost of one dozen pens -/
theorem cost_of_dozen_pens (total_cost : ℕ) (ratio_pen_pencil : ℕ) :
  total_cost = 260 →
  ratio_pen_pencil = 5 →
  ∃ (pen_cost : ℕ) (pencil_cost : ℕ),
    3 * pen_cost + 5 * pencil_cost = total_cost ∧
    pen_cost = ratio_pen_pencil * pencil_cost ∧
    12 * pen_cost = 780 :=
by sorry

end cost_of_dozen_pens_l2620_262017


namespace excellent_students_problem_l2620_262008

theorem excellent_students_problem (B₁ B₂ B₃ : Finset ℕ) 
  (h_total : (B₁ ∪ B₂ ∪ B₃).card = 100)
  (h_math : B₁.card = 70)
  (h_phys : B₂.card = 65)
  (h_chem : B₃.card = 75)
  (h_math_phys : (B₁ ∩ B₂).card = 40)
  (h_math_chem : (B₁ ∩ B₃).card = 45)
  (h_all : (B₁ ∩ B₂ ∩ B₃).card = 25) :
  ((B₂ ∩ B₃) \ B₁).card = 25 := by
  sorry

end excellent_students_problem_l2620_262008


namespace cartesian_product_eq_expected_set_l2620_262005

-- Define the set of possible x and y values
def X : Set ℕ := {1, 2}
def Y : Set ℕ := {1, 2}

-- Define the Cartesian product set
def cartesianProduct : Set (ℕ × ℕ) := {p | p.1 ∈ X ∧ p.2 ∈ Y}

-- Define the expected result set
def expectedSet : Set (ℕ × ℕ) := {(1, 1), (1, 2), (2, 1), (2, 2)}

-- Theorem stating that the Cartesian product is equal to the expected set
theorem cartesian_product_eq_expected_set : cartesianProduct = expectedSet := by
  sorry

end cartesian_product_eq_expected_set_l2620_262005


namespace stuart_reward_points_l2620_262043

/-- Represents the reward points earned per $25 spent at the Gauss Store. -/
def reward_points_per_unit : ℕ := 5

/-- Represents the amount Stuart spends at the Gauss Store in dollars. -/
def stuart_spend : ℕ := 200

/-- Represents the dollar amount that earns one unit of reward points. -/
def dollars_per_unit : ℕ := 25

/-- Calculates the number of reward points earned based on the amount spent. -/
def calculate_reward_points (spend : ℕ) : ℕ :=
  (spend / dollars_per_unit) * reward_points_per_unit

/-- Theorem stating that Stuart earns 40 reward points when spending $200 at the Gauss Store. -/
theorem stuart_reward_points : 
  calculate_reward_points stuart_spend = 40 := by
  sorry

end stuart_reward_points_l2620_262043


namespace quadratic_inequality_theorem_l2620_262056

theorem quadratic_inequality_theorem (c : ℝ) : 
  (∀ (a b : ℝ), (c^2 - 2*a*c + b) * (c^2 + 2*a*c + b) ≥ a^2 - 2*a^2 + b) ↔ 
  (c = 1/2 ∨ c = -1/2) :=
sorry

end quadratic_inequality_theorem_l2620_262056


namespace coefficient_of_x_cubed_l2620_262040

def p (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 2
def q (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem coefficient_of_x_cubed (x : ℝ) : 
  ∃ (a b c d e : ℝ), p x * q x = a * x^5 + b * x^4 - 25 * x^3 + c * x^2 + d * x + e :=
sorry

end coefficient_of_x_cubed_l2620_262040


namespace not_perfect_square_if_last_two_digits_odd_l2620_262047

-- Define a function to get the last two digits of an integer
def lastTwoDigits (n : ℤ) : ℤ × ℤ :=
  let d₁ := n % 10
  let d₂ := (n / 10) % 10
  (d₂, d₁)

-- Define a predicate for an integer being odd
def isOdd (n : ℤ) : Prop := n % 2 ≠ 0

-- Theorem statement
theorem not_perfect_square_if_last_two_digits_odd (n : ℤ) :
  let (d₂, d₁) := lastTwoDigits n
  isOdd d₂ ∧ isOdd d₁ → ¬∃ (m : ℤ), n = m ^ 2 :=
by sorry

end not_perfect_square_if_last_two_digits_odd_l2620_262047


namespace six_people_arrangement_l2620_262054

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

def permutations (n : ℕ) (k : ℕ) : ℕ := 
  if k > n then 0
  else factorial n / factorial (n - k)

theorem six_people_arrangement : 
  let total_arrangements := permutations 6 6
  let a_head_b_tail := permutations 4 4
  let a_head_b_not_tail := permutations 4 1 * permutations 4 4
  let a_not_head_b_tail := permutations 4 1 * permutations 4 4
  total_arrangements - a_head_b_tail - a_head_b_not_tail - a_not_head_b_tail = 504 := by
  sorry

end six_people_arrangement_l2620_262054


namespace product_expansion_l2620_262092

theorem product_expansion (a b c d : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 5 * x + 4) * (7 - 2 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = 18 := by
  sorry

end product_expansion_l2620_262092


namespace a_range_l2620_262064

def f (a x : ℝ) := x^2 - 2*a*x + 7

theorem a_range (a : ℝ) : 
  (∀ x y, 1 ≤ x ∧ x < y → f a x < f a y) → 
  a ≤ 1 := by
sorry

end a_range_l2620_262064


namespace marble_selection_theorem_l2620_262098

theorem marble_selection_theorem (total_marbles special_marbles marbles_to_choose : ℕ) 
  (h1 : total_marbles = 18)
  (h2 : special_marbles = 6)
  (h3 : marbles_to_choose = 4) :
  (Nat.choose special_marbles 2) * (Nat.choose (total_marbles - special_marbles) 2) = 990 := by
  sorry

end marble_selection_theorem_l2620_262098


namespace geometric_sequence_property_l2620_262068

/-- A sequence where the ratio of consecutive terms is constant -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence with a_4 = 2, prove that a_2 * a_6 = 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : GeometricSequence a) (h_a4 : a 4 = 2) : 
    a 2 * a 6 = 4 := by
  sorry

end geometric_sequence_property_l2620_262068


namespace subtracted_value_l2620_262089

theorem subtracted_value (chosen_number : ℕ) (final_answer : ℕ) : 
  chosen_number = 848 → final_answer = 6 → 
  ∃ x : ℚ, (chosen_number / 8 : ℚ) - x = final_answer ∧ x = 100 := by
sorry

end subtracted_value_l2620_262089


namespace triangle_angle_range_l2620_262019

theorem triangle_angle_range (A B C : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Triangle conditions
  2 * Real.sin A + Real.sin B = Real.sqrt 3 * Real.sin C →  -- Given equation
  π / 6 ≤ A ∧ A ≤ π / 2 := by  -- Conclusion to prove
sorry

end triangle_angle_range_l2620_262019


namespace unique_solution_is_76_l2620_262069

/-- The cubic equation in question -/
def cubic_equation (p x : ℝ) : ℝ := 5*x^3 - 5*(p+1)*x^2 + (71*p-1)*x + 1 - 66*p

/-- A function that checks if a number is a natural number -/
def is_natural (x : ℝ) : Prop := ∃ n : ℕ, x = n

/-- The main theorem stating that p = 76 is the unique solution -/
theorem unique_solution_is_76 :
  ∃! p : ℝ, p = 76 ∧ 
    ∃ x y : ℝ, x ≠ y ∧ 
      is_natural x ∧ is_natural y ∧
      cubic_equation p x = 0 ∧ 
      cubic_equation p y = 0 :=
sorry

end unique_solution_is_76_l2620_262069


namespace gcd_13n_plus_4_7n_plus_2_max_2_l2620_262074

theorem gcd_13n_plus_4_7n_plus_2_max_2 :
  (∀ n : ℕ+, Nat.gcd (13 * n + 4) (7 * n + 2) ≤ 2) ∧
  (∃ n : ℕ+, Nat.gcd (13 * n + 4) (7 * n + 2) = 2) := by
  sorry

end gcd_13n_plus_4_7n_plus_2_max_2_l2620_262074


namespace jellybean_difference_l2620_262013

theorem jellybean_difference (black green orange : ℕ) : 
  black = 8 →
  green = black + 2 →
  black + green + orange = 27 →
  green - orange = 1 :=
by sorry

end jellybean_difference_l2620_262013


namespace find_x_l2620_262057

def A : Set ℝ := {0, 2, 3}
def B (x : ℝ) : Set ℝ := {x + 1, x^2 + 4}

theorem find_x : ∃ x : ℝ, A ∩ B x = {3} → x = 2 := by
  sorry

end find_x_l2620_262057


namespace water_in_tank_after_40_days_l2620_262063

/-- Calculates the final amount of water in a tank given initial conditions and events. -/
def finalWaterAmount (initialWater : ℝ) (evaporationRate : ℝ) (daysBeforeAddition : ℕ) 
  (addedWater : ℝ) (remainingDays : ℕ) : ℝ :=
  let waterAfterFirstEvaporation := initialWater - evaporationRate * daysBeforeAddition
  let waterAfterAddition := waterAfterFirstEvaporation + addedWater
  waterAfterAddition - evaporationRate * remainingDays

/-- The final amount of water in the tank is 520 liters. -/
theorem water_in_tank_after_40_days :
  finalWaterAmount 500 2 15 100 25 = 520 := by
  sorry

end water_in_tank_after_40_days_l2620_262063


namespace inclination_angle_range_l2620_262099

theorem inclination_angle_range (α : Real) (h : α ∈ Set.Icc (π / 6) (2 * π / 3)) :
  let θ := Real.arctan (2 * Real.cos α)
  θ ∈ Set.Icc 0 (π / 3) ∪ Set.Ico (3 * π / 4) π := by
  sorry

end inclination_angle_range_l2620_262099


namespace purely_imaginary_z_l2620_262042

theorem purely_imaginary_z (α : ℝ) :
  let z : ℂ := Complex.mk (Real.sin α) (-(1 - Real.cos α))
  z.re = 0 → ∃ k : ℤ, α = (2 * k + 1) * Real.pi :=
by sorry

end purely_imaginary_z_l2620_262042


namespace van_speed_problem_l2620_262026

theorem van_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 288 →
  original_time = 6 →
  new_time_factor = 3 / 2 →
  let new_time := original_time * new_time_factor
  let new_speed := distance / new_time
  new_speed = 32 := by
sorry

end van_speed_problem_l2620_262026


namespace least_number_of_sweets_l2620_262041

theorem least_number_of_sweets (s : ℕ) : s > 0 ∧ 
  s % 6 = 5 ∧ 
  s % 8 = 3 ∧ 
  s % 9 = 6 ∧ 
  s % 11 = 10 ∧ 
  (∀ t : ℕ, t > 0 → t % 6 = 5 → t % 8 = 3 → t % 9 = 6 → t % 11 = 10 → s ≤ t) → 
  s = 2095 := by
  sorry

end least_number_of_sweets_l2620_262041


namespace new_year_markup_verify_new_year_markup_l2620_262048

/-- Calculates the New Year season markup percentage given other price adjustments and final profit -/
theorem new_year_markup (initial_markup : ℝ) (february_discount : ℝ) (final_profit : ℝ) : ℝ :=
  let new_year_markup := 
    ((1 + final_profit) / ((1 + initial_markup) * (1 - february_discount)) - 1) * 100
  by
    -- The proof would go here
    sorry

/-- Verifies that the New Year markup is 25% given the problem conditions -/
theorem verify_new_year_markup : 
  new_year_markup 0.20 0.09 0.365 = 25 :=
by
  -- The proof would go here
  sorry

end new_year_markup_verify_new_year_markup_l2620_262048
