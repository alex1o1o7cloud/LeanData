import Mathlib

namespace cherries_purchase_l3539_353997

theorem cherries_purchase (cost_per_kg : ℝ) (short_amount : ℝ) (money_on_hand : ℝ) 
  (h1 : cost_per_kg = 8)
  (h2 : short_amount = 400)
  (h3 : money_on_hand = 1600) :
  (money_on_hand + short_amount) / cost_per_kg = 250 := by
  sorry

end cherries_purchase_l3539_353997


namespace xyz_sum_lower_bound_l3539_353962

theorem xyz_sum_lower_bound (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : x * y * z + x * y + y * z + z * x = 4) : 
  x + y + z ≥ 3 := by
sorry

end xyz_sum_lower_bound_l3539_353962


namespace robbery_trial_l3539_353990

theorem robbery_trial (A B C : Prop) 
  (h1 : (¬A ∨ B) → C)
  (h2 : ¬A → ¬C) : 
  A ∧ C ∧ (B ∨ ¬B) := by
sorry

end robbery_trial_l3539_353990


namespace inequality_solution_l3539_353977

theorem inequality_solution (x : ℝ) : (1 + x) / 3 < x / 2 ↔ x > 2 := by
  sorry

end inequality_solution_l3539_353977


namespace xy_difference_squared_l3539_353905

theorem xy_difference_squared (x y b c : ℝ) 
  (h1 : x * y = c^2) 
  (h2 : 1 / x^2 + 1 / y^2 = b * c) : 
  (x - y)^2 = b * c^4 - 2 * c^2 := by
sorry

end xy_difference_squared_l3539_353905


namespace max_containers_proof_l3539_353938

def oatmeal_cookies : ℕ := 50
def chocolate_chip_cookies : ℕ := 75
def sugar_cookies : ℕ := 36

theorem max_containers_proof :
  let gcd := Nat.gcd oatmeal_cookies (Nat.gcd chocolate_chip_cookies sugar_cookies)
  (sugar_cookies / gcd) = 7 ∧ 
  (oatmeal_cookies / gcd) ≥ 7 ∧ 
  (chocolate_chip_cookies / gcd) ≥ 7 :=
by sorry

end max_containers_proof_l3539_353938


namespace circle_diameter_l3539_353909

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = π * r^2 → A = 196 * π → d = 2 * r → d = 28 := by
  sorry

end circle_diameter_l3539_353909


namespace min_value_6x_5y_l3539_353948

theorem min_value_6x_5y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (2 * x + y) + 3 / (x + y) = 2) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 1 / (2 * x' + y') + 3 / (x' + y') = 2 →
    6 * x + 5 * y ≤ 6 * x' + 5 * y') ∧
  6 * x + 5 * y = (13 + 4 * Real.sqrt 3) / 2 := by
sorry

end min_value_6x_5y_l3539_353948


namespace volunteer_distribution_l3539_353902

theorem volunteer_distribution (n : ℕ) (k : ℕ) (m : ℕ) : n = 5 ∧ k = 3 ∧ m = 3 →
  (Nat.choose n 2 * Nat.choose (n - 2) 2 * Nat.choose (n - 4) 1) / 2 * Nat.factorial k = 90 := by
  sorry

end volunteer_distribution_l3539_353902


namespace sum_of_decimals_l3539_353929

theorem sum_of_decimals : 5.623 + 4.76 = 10.383 := by
  sorry

end sum_of_decimals_l3539_353929


namespace farm_animals_l3539_353921

theorem farm_animals (chickens buffalos : ℕ) : 
  chickens + buffalos = 9 →
  2 * chickens + 4 * buffalos = 26 →
  chickens = 5 := by
sorry

end farm_animals_l3539_353921


namespace bob_daily_earnings_l3539_353946

/-- Proves that Bob makes $4 per day given the conditions of the problem -/
theorem bob_daily_earnings (sally_earnings : ℝ) (total_savings : ℝ) (days_in_year : ℕ) :
  sally_earnings = 6 →
  total_savings = 1825 →
  days_in_year = 365 →
  ∃ (bob_earnings : ℝ),
    bob_earnings = 4 ∧
    (sally_earnings / 2 + bob_earnings / 2) * days_in_year = total_savings :=
by sorry

end bob_daily_earnings_l3539_353946


namespace linear_function_property_l3539_353914

theorem linear_function_property (a : ℝ) :
  (∃ y : ℝ, y = a * 3 + (1 - a) ∧ y = 7) →
  (∃ y : ℝ, y = a * 8 + (1 - a) ∧ y = 22) :=
by sorry

end linear_function_property_l3539_353914


namespace theater_ticket_price_l3539_353976

/-- Proves that the cost of an orchestra seat is $12 given the conditions of the theater problem -/
theorem theater_ticket_price
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (balcony_price : ℕ)
  (ticket_difference : ℕ)
  (h1 : total_tickets = 355)
  (h2 : total_revenue = 3320)
  (h3 : balcony_price = 8)
  (h4 : ticket_difference = 115) :
  ∃ (orchestra_price : ℕ),
    orchestra_price = 12 ∧
    ∃ (orchestra_tickets : ℕ),
      orchestra_tickets + (orchestra_tickets + ticket_difference) = total_tickets ∧
      orchestra_price * orchestra_tickets + balcony_price * (orchestra_tickets + ticket_difference) = total_revenue :=
by
  sorry

end theater_ticket_price_l3539_353976


namespace base_conversion_sum_rounded_to_28_l3539_353943

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : Rat) : Int :=
  (q + 1/2).floor

theorem base_conversion_sum_rounded_to_28 :
  let a := to_base_10 [4, 5, 2] 8  -- 254 in base 8
  let b := to_base_10 [3, 1] 4     -- 13 in base 4
  let c := to_base_10 [2, 3, 1] 5  -- 132 in base 5
  let d := to_base_10 [2, 3] 4     -- 32 in base 4
  round_to_nearest ((a / b : Rat) + (c / d : Rat)) = 28 := by
  sorry

#eval round_to_nearest ((172 / 7 : Rat) + (42 / 14 : Rat))

end base_conversion_sum_rounded_to_28_l3539_353943


namespace range_of_m_l3539_353963

theorem range_of_m (x : ℝ) :
  (∀ x, (1/3 < x ∧ x < 1/2) → (m - 1 < x ∧ x < m + 1)) →
  (-1/2 ≤ m ∧ m ≤ 4/3) :=
by sorry

end range_of_m_l3539_353963


namespace parabola_circle_theorem_l3539_353931

/-- Given a parabola y = ax^2 + bx + c (a ≠ 0) intersecting the x-axis at points A and B,
    the equation of the circle with AB as diameter is ax^2 + bx + c + ay^2 = 0. -/
theorem parabola_circle_theorem (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ →
  ∀ x y : ℝ, a * x^2 + b * x + c + a * y^2 = 0 ↔ 
    ∃ t : ℝ, x = (1 - t) * x₁ + t * x₂ ∧ 
             y^2 = t * (1 - t) * (x₂ - x₁)^2 :=
by sorry

end parabola_circle_theorem_l3539_353931


namespace paper_strip_to_squares_l3539_353942

/-- Represents a strip of paper with given width and length -/
structure PaperStrip where
  width : ℝ
  length : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Function to calculate the area of a square -/
def squareArea (s : Square) : ℝ := s.side * s.side

/-- Theorem stating that a paper strip of width 1 cm and length 4 cm
    can be transformed into squares of areas 1 sq cm and 2 sq cm -/
theorem paper_strip_to_squares 
  (strip : PaperStrip) 
  (h_width : strip.width = 1) 
  (h_length : strip.length = 4) :
  ∃ (s1 s2 : Square), 
    squareArea s1 = 1 ∧ 
    squareArea s2 = 2 := by
  sorry


end paper_strip_to_squares_l3539_353942


namespace shadow_problem_l3539_353973

/-- Given a cube with edge length 2 cm and a light source y cm above an upper vertex
    casting a shadow with area 98 sq cm (excluding the area beneath the cube),
    prove that the greatest integer not exceeding 1000y is 8100. -/
theorem shadow_problem (y : ℝ) : 
  (2 : ℝ) > 0 ∧ 
  y > 0 ∧ 
  (y / (Real.sqrt 102 - 2) = 1) ∧ 
  (98 : ℝ) = (Real.sqrt 102)^2 - 2^2 →
  Int.floor (1000 * y) = 8100 := by
sorry

end shadow_problem_l3539_353973


namespace smallest_spend_l3539_353982

/-- Represents a gift set with its composition and price -/
structure GiftSet where
  chocolates : ℕ
  caramels : ℕ
  price : ℕ

/-- The first type of gift set -/
def gift1 : GiftSet := { chocolates := 3, caramels := 15, price := 350 }

/-- The second type of gift set -/
def gift2 : GiftSet := { chocolates := 20, caramels := 5, price := 500 }

/-- Calculates the total cost of buying gift sets -/
def totalCost (m n : ℕ) : ℕ := m * gift1.price + n * gift2.price

/-- Calculates the total number of chocolate candies -/
def totalChocolates (m n : ℕ) : ℕ := m * gift1.chocolates + n * gift2.chocolates

/-- Calculates the total number of caramel candies -/
def totalCaramels (m n : ℕ) : ℕ := m * gift1.caramels + n * gift2.caramels

/-- Theorem stating the smallest non-zero amount Eugene needs to spend -/
theorem smallest_spend : 
  ∃ m n : ℕ, m + n > 0 ∧ 
    totalChocolates m n = totalCaramels m n ∧
    totalCost m n = 3750 ∧
    ∀ k l : ℕ, k + l > 0 → 
      totalChocolates k l = totalCaramels k l → 
      totalCost k l ≥ 3750 := by sorry

end smallest_spend_l3539_353982


namespace cube_rotation_theorem_l3539_353971

/-- Represents a cube with numbers on its faces -/
structure Cube where
  left : ℕ
  right : ℕ
  front : ℕ
  back : ℕ
  top : ℕ
  bottom : ℕ

/-- Represents the state of the cube after rotations -/
structure CubeState where
  bottom : ℕ
  front : ℕ
  right : ℕ

/-- Rotates the cube from left to right -/
def rotateLeftRight (c : Cube) : Cube := sorry

/-- Rotates the cube from front to back -/
def rotateFrontBack (c : Cube) : Cube := sorry

/-- Applies multiple rotations to the cube -/
def applyRotations (c : Cube) (leftRightRotations frontBackRotations : ℕ) : Cube := sorry

/-- Theorem stating the final state of the cube after rotations -/
theorem cube_rotation_theorem (c : Cube) 
  (h1 : c.left + c.right = 50)
  (h2 : c.front + c.back = 50)
  (h3 : c.top + c.bottom = 50) :
  let finalCube := applyRotations c 97 98
  CubeState.mk finalCube.bottom finalCube.front finalCube.right = CubeState.mk 13 35 11 := by sorry

end cube_rotation_theorem_l3539_353971


namespace condition_relationship_l3539_353989

theorem condition_relationship (θ : ℝ) (a : ℝ) : 
  ¬(∀ θ a, (Real.sqrt (1 + Real.sin θ) = a) ↔ (Real.sin (θ/2) + Real.cos (θ/2) = a)) :=
by sorry

end condition_relationship_l3539_353989


namespace octagon_area_in_square_l3539_353932

/-- Given a square with side length s, the area of the octagon formed by connecting each vertex
    to the midpoints of the opposite two sides is s^2/6. -/
theorem octagon_area_in_square (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let octagon_area := square_area / 6
  octagon_area = square_area / 6 := by
sorry

end octagon_area_in_square_l3539_353932


namespace set_7_24_25_is_pythagorean_triple_l3539_353998

/-- A Pythagorean triple is a set of three positive integers a, b, and c that satisfy a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop := a * a + b * b = c * c

/-- The set (7, 24, 25) is a Pythagorean triple -/
theorem set_7_24_25_is_pythagorean_triple : is_pythagorean_triple 7 24 25 := by
  sorry

end set_7_24_25_is_pythagorean_triple_l3539_353998


namespace quadrilateral_symmetry_theorem_l3539_353908

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a symmetry operation
def symmetryOperation (Q : Quadrilateral) : Quadrilateral := sorry

-- Define a cyclic quadrilateral
def isCyclic (Q : Quadrilateral) : Prop := sorry

-- Define a permissible quadrilateral
def isPermissible (Q : Quadrilateral) : Prop := sorry

-- Define equality of quadrilaterals
def equalQuadrilaterals (Q1 Q2 : Quadrilateral) : Prop := sorry

-- Define the application of n symmetry operations
def applyNOperations (Q : Quadrilateral) (n : ℕ) : Quadrilateral := sorry

theorem quadrilateral_symmetry_theorem (Q : Quadrilateral) :
  (isCyclic Q → equalQuadrilaterals Q (applyNOperations Q 3)) ∧
  (isPermissible Q → equalQuadrilaterals Q (applyNOperations Q 6)) := by
  sorry

end quadrilateral_symmetry_theorem_l3539_353908


namespace eggs_per_set_l3539_353920

theorem eggs_per_set (total_eggs : ℕ) (num_sets : ℕ) (h1 : total_eggs = 108) (h2 : num_sets = 9) :
  total_eggs / num_sets = 12 := by
sorry

end eggs_per_set_l3539_353920


namespace power_sum_equality_l3539_353911

theorem power_sum_equality : 2^345 + 9^8 / 9^5 = 2^345 + 729 := by sorry

end power_sum_equality_l3539_353911


namespace pear_weighs_130_l3539_353926

/-- The weight of an apple in grams -/
def apple_weight : ℝ := sorry

/-- The weight of a pear in grams -/
def pear_weight : ℝ := sorry

/-- The weight of a banana in grams -/
def banana_weight : ℝ := sorry

/-- The first condition: one apple, three pears, and two bananas weigh 920 grams -/
axiom condition1 : apple_weight + 3 * pear_weight + 2 * banana_weight = 920

/-- The second condition: two apples, four bananas, and five pears weigh 1,710 grams -/
axiom condition2 : 2 * apple_weight + 4 * banana_weight + 5 * pear_weight = 1710

/-- Theorem stating that a pear weighs 130 grams -/
theorem pear_weighs_130 : pear_weight = 130 := by sorry

end pear_weighs_130_l3539_353926


namespace employee_count_l3539_353978

/-- The number of employees in an organization (excluding the manager) -/
def num_employees : ℕ := 20

/-- The average monthly salary of employees (excluding the manager) -/
def avg_salary : ℕ := 1600

/-- The increase in average salary when the manager's salary is added -/
def salary_increase : ℕ := 100

/-- The manager's monthly salary -/
def manager_salary : ℕ := 3700

/-- Theorem stating the number of employees given the salary conditions -/
theorem employee_count :
  (num_employees * avg_salary + manager_salary) / (num_employees + 1) =
  avg_salary + salary_increase :=
by sorry

end employee_count_l3539_353978


namespace class_size_problem_l3539_353907

theorem class_size_problem (average_age : ℝ) (teacher_age : ℝ) (new_average : ℝ) :
  average_age = 10 →
  teacher_age = 26 →
  new_average = average_age + 1 →
  ∃ n : ℕ, (n : ℝ) * average_age + teacher_age = (n + 1 : ℝ) * new_average ∧ n = 15 :=
by sorry

end class_size_problem_l3539_353907


namespace celine_initial_amount_l3539_353970

/-- The price of a laptop in dollars -/
def laptop_price : ℕ := 600

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 400

/-- The number of laptops Celine bought -/
def laptops_bought : ℕ := 2

/-- The number of smartphones Celine bought -/
def smartphones_bought : ℕ := 4

/-- The amount of change Celine received in dollars -/
def change_received : ℕ := 200

/-- Celine's initial amount of money in dollars -/
def initial_amount : ℕ := laptop_price * laptops_bought + smartphone_price * smartphones_bought + change_received

theorem celine_initial_amount : initial_amount = 3000 := by
  sorry

end celine_initial_amount_l3539_353970


namespace arithmetic_sequence_sum_l3539_353915

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3) ^ 2 - 6 * (a 3) - 1 = 0 →
  (a 15) ^ 2 - 6 * (a 15) - 1 = 0 →
  (a 7) + (a 8) + (a 9) + (a 10) + (a 11) = 15 :=
by sorry

end arithmetic_sequence_sum_l3539_353915


namespace geometric_sequence_third_term_l3539_353923

/-- A geometric sequence {a_n} with a_1 = 1 and a_5 = 9 has a_3 = 3 -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 5 / a 1)^(1/4)) →  -- Geometric sequence condition
  a 1 = 1 →
  a 5 = 9 →
  a 3 = 3 := by
sorry

end geometric_sequence_third_term_l3539_353923


namespace smallest_number_with_properties_l3539_353906

def digit_sum (n : ℕ) : ℕ := sorry

def is_smallest_with_properties (n : ℕ) : Prop :=
  (n % 5 = 0) ∧
  (digit_sum n = 100) ∧
  (∀ m : ℕ, m < n → (m % 5 ≠ 0 ∨ digit_sum m ≠ 100))

theorem smallest_number_with_properties :
  is_smallest_with_properties 599999999995 := by sorry

end smallest_number_with_properties_l3539_353906


namespace genevieve_cherry_shortage_l3539_353917

/-- The amount Genevieve was short when buying cherries -/
def amount_short (cost_per_kg : ℕ) (amount_had : ℕ) (kg_bought : ℕ) : ℕ :=
  cost_per_kg * kg_bought - amount_had

/-- Proof that Genevieve was short $400 -/
theorem genevieve_cherry_shortage : amount_short 8 1600 250 = 400 := by
  sorry

end genevieve_cherry_shortage_l3539_353917


namespace prime_factors_and_recalculation_l3539_353901

def original_number : ℕ := 546

theorem prime_factors_and_recalculation (n : ℕ) (h : n = original_number) :
  (∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ n ∧ largest ∣ n ∧
    (∀ p : ℕ, p.Prime → p ∣ n → smallest ≤ p) ∧
    (∀ p : ℕ, p.Prime → p ∣ n → p ≤ largest) ∧
    smallest + largest = 15) ∧
  (∃ (factors : List ℕ),
    (∀ p ∈ factors, p.Prime ∧ p ∣ n) ∧
    (∀ p : ℕ, p.Prime → p ∣ n → p ∈ factors) ∧
    (List.prod (List.map (· * 2) factors) = 8736)) :=
by sorry

end prime_factors_and_recalculation_l3539_353901


namespace tournament_committee_count_l3539_353975

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The minimum number of female members in each team -/
def min_females : ℕ := 2

/-- The number of members selected for the committee by the host team -/
def host_committee_size : ℕ := 3

/-- The number of members selected for the committee by non-host teams -/
def non_host_committee_size : ℕ := 2

/-- The total number of members in the tournament committee -/
def total_committee_size : ℕ := 10

/-- The number of possible tournament committees -/
def num_committees : ℕ := 1296540

theorem tournament_committee_count :
  (num_teams > 0) →
  (team_size ≥ host_committee_size) →
  (team_size ≥ non_host_committee_size) →
  (min_females ≥ non_host_committee_size) →
  (min_females < host_committee_size) →
  (num_teams * non_host_committee_size + host_committee_size = total_committee_size) →
  (num_committees = (num_teams - 1) * (Nat.choose team_size host_committee_size) * 
    (Nat.choose team_size non_host_committee_size)^(num_teams - 2) * 
    (Nat.choose min_females non_host_committee_size)) :=
by sorry

#check tournament_committee_count

end tournament_committee_count_l3539_353975


namespace pq_length_l3539_353994

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

-- Define the intersection points
def intersection_points (P Q : ℝ × ℝ) : Prop :=
  parabola P.1 P.2 ∧ parabola Q.1 Q.2 ∧ line P.1 P.2 ∧ line Q.1 Q.2

-- Theorem statement
theorem pq_length (P Q : ℝ × ℝ) :
  intersection_points P Q →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 16/3 := by sorry

end pq_length_l3539_353994


namespace phillips_cucumbers_l3539_353983

/-- Proves that Phillip has 8 cucumbers given the pickle-making conditions --/
theorem phillips_cucumbers :
  ∀ (jars : ℕ) (initial_vinegar : ℕ) (pickles_per_cucumber : ℕ) (pickles_per_jar : ℕ)
    (vinegar_per_jar : ℕ) (remaining_vinegar : ℕ),
  jars = 4 →
  initial_vinegar = 100 →
  pickles_per_cucumber = 6 →
  pickles_per_jar = 12 →
  vinegar_per_jar = 10 →
  remaining_vinegar = 60 →
  ∃ (cucumbers : ℕ),
    cucumbers = 8 ∧
    cucumbers * pickles_per_cucumber = jars * pickles_per_jar ∧
    initial_vinegar - remaining_vinegar = jars * vinegar_per_jar :=
by
  sorry


end phillips_cucumbers_l3539_353983


namespace shirt_sale_price_l3539_353995

theorem shirt_sale_price (original_price : ℝ) (original_price_pos : original_price > 0) :
  let first_sale_price := original_price * (1 - 0.5)
  let final_price := first_sale_price * (1 - 0.1)
  final_price / original_price = 0.45 := by
sorry

end shirt_sale_price_l3539_353995


namespace roberts_balls_l3539_353936

theorem roberts_balls (robert_initial : ℕ) (tim_initial : ℕ) : 
  robert_initial = 25 → 
  tim_initial = 40 → 
  robert_initial + tim_initial / 2 = 45 := by
  sorry

end roberts_balls_l3539_353936


namespace arithmetic_equalities_l3539_353957

theorem arithmetic_equalities :
  (-16 - (-12) - 24 + 18 = -10) ∧
  (0.125 + 1/4 + (-2 - 1/8) + (-0.25) = -2) ∧
  ((-1/12 - 1/36 + 1/6) * (-36) = -2) ∧
  ((-2 + 3) * 3 - (-2)^3 / 4 = 5) := by
  sorry

end arithmetic_equalities_l3539_353957


namespace two_number_problem_l3539_353974

theorem two_number_problem (x y : ℚ) 
  (sum_eq : x + y = 40)
  (double_subtract : 2 * y - 4 * x = 12) :
  |y - x| = 52 / 3 := by
sorry

end two_number_problem_l3539_353974


namespace coconut_trees_per_square_meter_l3539_353919

/-- Represents the coconut farm scenario -/
structure CoconutFarm where
  size : ℝ
  treesPerSquareMeter : ℝ
  coconutsPerTree : ℕ
  harvestFrequency : ℕ
  pricePerCoconut : ℝ
  earningsAfterSixMonths : ℝ

/-- Theorem stating the number of coconut trees per square meter -/
theorem coconut_trees_per_square_meter (farm : CoconutFarm)
  (h1 : farm.size = 20)
  (h2 : farm.coconutsPerTree = 6)
  (h3 : farm.harvestFrequency = 3)
  (h4 : farm.pricePerCoconut = 0.5)
  (h5 : farm.earningsAfterSixMonths = 240) :
  farm.treesPerSquareMeter = 2 := by
  sorry


end coconut_trees_per_square_meter_l3539_353919


namespace max_leap_years_in_period_l3539_353940

/-- A calendrical system where leap years occur every five years -/
structure CalendarSystem where
  leap_year_interval : ℕ
  leap_year_interval_eq : leap_year_interval = 5

/-- The number of years in the period we're considering -/
def period : ℕ := 200

/-- The maximum number of leap years in the given period -/
def max_leap_years (c : CalendarSystem) : ℕ := period / c.leap_year_interval

/-- Theorem: The maximum number of leap years in a 200-year period is 40 -/
theorem max_leap_years_in_period (c : CalendarSystem) : max_leap_years c = 40 := by
  sorry

end max_leap_years_in_period_l3539_353940


namespace function_not_in_second_quadrant_l3539_353985

/-- The function f(x) = a^x + b does not pass through the second quadrant when a > 1 and b < -1 -/
theorem function_not_in_second_quadrant (a b : ℝ) (ha : a > 1) (hb : b < -1) :
  ∀ x : ℝ, x < 0 → a^x + b ≤ 0 := by sorry

end function_not_in_second_quadrant_l3539_353985


namespace complex_product_square_l3539_353969

/-- Given complex numbers Q, E, and D, prove that (Q * E * D)² equals 8400 + 8000i -/
theorem complex_product_square (Q E D : ℂ) 
  (hQ : Q = 7 + 3*I) 
  (hE : E = 1 + I) 
  (hD : D = 7 - 3*I) : 
  (Q * E * D)^2 = 8400 + 8000*I := by
  sorry

end complex_product_square_l3539_353969


namespace systematic_sampling_removal_l3539_353945

theorem systematic_sampling_removal (total_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 1252) 
  (h2 : sample_size = 50) : 
  ∃ (removed : ℕ), removed = total_students % sample_size ∧ removed = 2 := by
  sorry

end systematic_sampling_removal_l3539_353945


namespace hotel_charge_difference_l3539_353996

/-- The charge for a single room at different hotels -/
structure HotelCharges where
  G : ℝ  -- Charge at hotel G
  R : ℝ  -- Charge at hotel R
  P : ℝ  -- Charge at hotel P

/-- The conditions given in the problem -/
def problem_conditions (h : HotelCharges) : Prop :=
  h.P = 0.9 * h.G ∧ 
  h.R = 1.125 * h.G

/-- The theorem stating the percentage difference between charges at hotel P and R -/
theorem hotel_charge_difference (h : HotelCharges) 
  (hcond : problem_conditions h) : 
  (h.R - h.P) / h.R = 0.2 := by
  sorry


end hotel_charge_difference_l3539_353996


namespace centroid_construction_condition_l3539_353991

/-- A function that checks if a number is divisible by all prime factors of another number -/
def isDivisibleByAllPrimeFactors (m n : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → p ∣ n → p ∣ m

/-- The main theorem stating the condition for constructing the centroid -/
theorem centroid_construction_condition (n m : ℕ) (h : n ≥ 3) :
  (∃ (construction : Unit), True) ↔ (2 ∣ m ∧ isDivisibleByAllPrimeFactors m n) :=
sorry

end centroid_construction_condition_l3539_353991


namespace program_output_is_44_l3539_353954

/-- The output value of the program -/
def program_output : ℕ := 44

/-- Theorem stating that the program output is 44 -/
theorem program_output_is_44 : program_output = 44 := by
  sorry

end program_output_is_44_l3539_353954


namespace min_triangles_to_cover_l3539_353941

/-- The minimum number of small equilateral triangles needed to cover a large equilateral triangle -/
theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) : 
  small_side = 2 →
  large_side = 16 →
  (large_side / small_side) ^ 2 = 64 :=
by sorry

end min_triangles_to_cover_l3539_353941


namespace smallest_n_satisfying_conditions_l3539_353904

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), n > 0 ∧ n = 6 ∧
  (∃ (p : ℕ), p < n ∧ Prime p ∧ Odd p ∧ (n^2 - n + 4) % p = 0) ∧
  (∃ (q : ℕ), q < n ∧ Prime q ∧ (n^2 - n + 4) % q ≠ 0) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n →
    ¬((∃ (p : ℕ), p < m ∧ Prime p ∧ Odd p ∧ (m^2 - m + 4) % p = 0) ∧
      (∃ (q : ℕ), q < m ∧ Prime q ∧ (m^2 - m + 4) % q ≠ 0))) :=
by sorry

end smallest_n_satisfying_conditions_l3539_353904


namespace exist_similar_triangles_same_color_l3539_353912

-- Define a color type
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
def colorFunction : Point → Color := sorry

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Define similarity between triangles
def areSimilar (t1 t2 : Triangle) (ratio : ℝ) : Prop := sorry

-- Define the main theorem
theorem exist_similar_triangles_same_color :
  ∃ (t1 t2 : Triangle) (color : Color),
    areSimilar t1 t2 1995 ∧
    colorFunction t1.a = color ∧
    colorFunction t1.b = color ∧
    colorFunction t1.c = color ∧
    colorFunction t2.a = color ∧
    colorFunction t2.b = color ∧
    colorFunction t2.c = color := by
  sorry

end exist_similar_triangles_same_color_l3539_353912


namespace quadratic_function_minimum_l3539_353992

/-- A quadratic function that takes values 6, 5, and 5 for three consecutive natural values. -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 6 ∧ f (n + 1) = 5 ∧ f (n + 2) = 5

/-- The theorem stating that the minimum value of the quadratic function is 5. -/
theorem quadratic_function_minimum (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 5 :=
by sorry

end quadratic_function_minimum_l3539_353992


namespace pythagorean_triple_identification_l3539_353984

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_identification :
  is_pythagorean_triple 3 4 5 ∧
  is_pythagorean_triple 5 12 13 ∧
  ¬(is_pythagorean_triple 4 5 6) ∧
  is_pythagorean_triple 8 15 17 :=
by sorry

end pythagorean_triple_identification_l3539_353984


namespace geometry_theorem_l3539_353960

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line → Prop)

-- Theorem statement
theorem geometry_theorem 
  (α β : Plane) (m n l : Line) : 
  (∀ m n α β, perpendicular m n → perpendicular_plane_line α m → perpendicular_plane_line β n → perpendicular_planes α β) ∧
  (∀ m α β, contained_in m α → parallel_planes α β → parallel_line_plane m β) ∧
  (∀ α β m l, intersection α β l → parallel_line_plane m α → parallel_line_plane m β → parallel_lines m l) ∧
  ¬(∀ m n α β, perpendicular m n → perpendicular_plane_line α m → parallel_line_plane n β → perpendicular_planes α β) :=
by sorry

end geometry_theorem_l3539_353960


namespace fraction_sum_equality_l3539_353988

theorem fraction_sum_equality : 
  (1 : ℚ) / 15 + (2 : ℚ) / 25 + (3 : ℚ) / 35 + (4 : ℚ) / 45 = (506 : ℚ) / 1575 := by
  sorry

end fraction_sum_equality_l3539_353988


namespace cos_pi_twelve_squared_identity_l3539_353967

theorem cos_pi_twelve_squared_identity : 2 * (Real.cos (π / 12))^2 - 1 = Real.sqrt 3 / 2 := by
  sorry

end cos_pi_twelve_squared_identity_l3539_353967


namespace set_cardinality_relation_l3539_353959

theorem set_cardinality_relation (a b : ℕ+) (A B : Finset ℕ+) :
  (A ∩ B = ∅) →
  (∀ i ∈ A ∪ B, (i + a) ∈ A ∨ (i - b) ∈ B) →
  a * A.card = b * B.card :=
sorry

end set_cardinality_relation_l3539_353959


namespace compound_propositions_l3539_353903

-- Define the propositions p and q
def p : Prop := ∃ x : ℝ, x > 2 ∧ x > 1
def q : Prop := ∀ a b : ℝ, a > b → (1 / a) < (1 / b)

-- Define that p is sufficient but not necessary for x > 1
axiom p_sufficient : p → ∃ x : ℝ, x > 1
axiom p_not_necessary : ∃ x : ℝ, x > 1 ∧ ¬(x > 2)

-- Theorem stating that p ∧ ¬q is true, while other compounds are false
theorem compound_propositions :
  (p ∧ ¬q) ∧ ¬(p ∧ q) ∧ ¬(¬p ∨ q) ∧ ¬(¬p ∧ ¬q) :=
sorry

end compound_propositions_l3539_353903


namespace geometric_sequence_a6_l3539_353979

/-- Given a geometric sequence {a_n} where a_4 = 7 and a_8 = 63, prove that a_6 = 21 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_a4 : a 4 = 7) (h_a8 : a 8 = 63) : a 6 = 21 := by
  sorry

end geometric_sequence_a6_l3539_353979


namespace geometric_sequence_sum_l3539_353956

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  a 2 + a 3 = 4 →               -- given condition
  a 1 + a 4 = 6 :=              -- conclusion to prove
by
  sorry

end geometric_sequence_sum_l3539_353956


namespace line_through_point_parallel_to_line_line_equation_proof_l3539_353935

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point2D.liesOn (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line2D.isParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (given_point : Point2D) 
  (given_line : Line2D) 
  (result_line : Line2D) : Prop :=
  (given_point.liesOn result_line) ∧ 
  (result_line.isParallel given_line) →
  (result_line.a = 2 ∧ result_line.b = -1 ∧ result_line.c = 4)

#check line_through_point_parallel_to_line 
  (Point2D.mk 0 4) 
  (Line2D.mk 2 (-1) (-3)) 
  (Line2D.mk 2 (-1) 4)

theorem line_equation_proof : 
  line_through_point_parallel_to_line 
    (Point2D.mk 0 4) 
    (Line2D.mk 2 (-1) (-3)) 
    (Line2D.mk 2 (-1) 4) := by
  sorry

end line_through_point_parallel_to_line_line_equation_proof_l3539_353935


namespace rectangle_geometric_mean_l3539_353928

/-- Given a rectangle with side lengths a and b, where b is the geometric mean
    of a and the perimeter, prove that b = a + a√3 -/
theorem rectangle_geometric_mean (a b : ℝ) (h_pos : 0 < a) :
  b^2 = a * (2*a + 2*b) → b = a + a * Real.sqrt 3 :=
sorry

end rectangle_geometric_mean_l3539_353928


namespace vector_difference_magnitude_l3539_353965

def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 6]

theorem vector_difference_magnitude 
  (h_parallel : ∃ (k : ℝ), ∀ i, a i = k * b x i) :
  ‖a - b x‖ = 2 * Real.sqrt 5 := by
  sorry

end vector_difference_magnitude_l3539_353965


namespace f_decreasing_range_f_less_than_g_range_l3539_353951

open Real

noncomputable def f (a x : ℝ) : ℝ := log x - a^2 * x^2 + a * x

noncomputable def g (a x : ℝ) : ℝ := (3*a + 1) * x - (a^2 + a) * x^2

theorem f_decreasing_range (a : ℝ) (h : a ≠ 0) :
  (∀ x ≥ 1, ∀ y ≥ x, f a x ≥ f a y) ↔ a ≥ 1 :=
sorry

theorem f_less_than_g_range (a : ℝ) (h : a ≠ 0) :
  (∀ x > 1, f a x < g a x) ↔ -1 < a ∧ a ≤ 0 :=
sorry

end f_decreasing_range_f_less_than_g_range_l3539_353951


namespace cos_two_theta_value_l3539_353952

theorem cos_two_theta_value (θ : Real) 
  (h : Real.exp (Real.log 2 * (-5/2 + 2 * Real.cos θ)) + 1 = Real.exp (Real.log 2 * (3/4 + Real.cos θ))) : 
  Real.cos (2 * θ) = 17/8 := by
sorry

end cos_two_theta_value_l3539_353952


namespace part_one_part_two_l3539_353981

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | -x^2 + 2*x + m > 0}

-- Part 1
theorem part_one : A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Part 2
theorem part_two : ∃ m : ℝ, m = 8 ∧ A ∩ B m = {x | -1 < x ∧ x < 4} := by sorry

end part_one_part_two_l3539_353981


namespace rachels_homework_l3539_353910

/-- 
Given that Rachel has 5 pages of math homework and 3 more pages of math homework 
than reading homework, prove that she has 2 pages of reading homework.
-/
theorem rachels_homework (math_pages reading_pages : ℕ) : 
  math_pages = 5 → 
  math_pages = reading_pages + 3 → 
  reading_pages = 2 := by
sorry

end rachels_homework_l3539_353910


namespace four_digit_divisible_by_45_l3539_353930

theorem four_digit_divisible_by_45 : ∃ (a b : ℕ), 
  a < 10 ∧ b < 10 ∧ 
  (1000 * a + 520 + b) % 45 = 0 ∧
  (∃ (c d : ℕ), c < 10 ∧ d < 10 ∧ c ≠ a ∧ d ≠ b ∧ (1000 * c + 520 + d) % 45 = 0) := by
  sorry

end four_digit_divisible_by_45_l3539_353930


namespace p_sufficient_not_necessary_for_q_l3539_353900

-- Define propositions p and q
def p (x y : ℝ) : Prop := x > 0 ∧ y > 0
def q (x y : ℝ) : Prop := x * y > 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q : 
  (∀ x y : ℝ, p x y → q x y) ∧ 
  (∃ x y : ℝ, q x y ∧ ¬(p x y)) :=
sorry

end p_sufficient_not_necessary_for_q_l3539_353900


namespace jam_solution_l3539_353964

/-- Represents the amount and consumption rate of jam for a person -/
structure JamConsumption where
  amount : ℝ
  rate : ℝ

/-- The problem of determining jam consumption for Ponchik and Syropchik -/
def jam_problem (ponchik : JamConsumption) (syropchik : JamConsumption) : Prop :=
  -- Total amount of jam
  ponchik.amount + syropchik.amount = 100 ∧
  -- Same time to consume their own supplies
  ponchik.amount / ponchik.rate = syropchik.amount / syropchik.rate ∧
  -- Ponchik's consumption time if he had Syropchik's amount
  syropchik.amount / ponchik.rate = 45 ∧
  -- Syropchik's consumption time if he had Ponchik's amount
  ponchik.amount / syropchik.rate = 20

/-- The solution to the jam consumption problem -/
theorem jam_solution :
  ∃ (ponchik syropchik : JamConsumption),
    jam_problem ponchik syropchik ∧
    ponchik.amount = 40 ∧
    ponchik.rate = 4/3 ∧
    syropchik.amount = 60 ∧
    syropchik.rate = 2 :=
by sorry

end jam_solution_l3539_353964


namespace fifth_power_sum_l3539_353949

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 21)
  (h4 : a * x^4 + b * y^4 = 55) :
  a * x^5 + b * y^5 = -131 := by
  sorry

end fifth_power_sum_l3539_353949


namespace conditional_prob_B_given_A_l3539_353987

-- Define the sample space for a six-sided die
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define event A: odd numbers
def A : Finset Nat := {1, 3, 5}

-- Define event B: getting 3 points
def B : Finset Nat := {3}

-- Define the probability measure
def P (S : Finset Nat) : ℚ := (S.card : ℚ) / (Ω.card : ℚ)

-- Theorem: P(B|A) = 1/3
theorem conditional_prob_B_given_A : 
  P (A ∩ B) / P A = 1 / 3 := by sorry

end conditional_prob_B_given_A_l3539_353987


namespace caitlin_age_l3539_353961

/-- Proves that Caitlin's age is 13 years given the ages of Aunt Anna and Brianna -/
theorem caitlin_age (anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ)
  (h1 : anna_age = 60)
  (h2 : brianna_age = anna_age / 3)
  (h3 : caitlin_age = brianna_age - 7) :
  caitlin_age = 13 := by
sorry

end caitlin_age_l3539_353961


namespace doctor_nurse_ratio_l3539_353968

theorem doctor_nurse_ratio (total : ℕ) (nurses : ℕ) (h1 : total = 200) (h2 : nurses = 120) :
  (total - nurses) / nurses = 2 / 3 := by
sorry

end doctor_nurse_ratio_l3539_353968


namespace exists_excursion_with_frequent_participants_l3539_353924

/-- Represents an excursion --/
structure Excursion where
  participants : Finset Nat
  deriving Inhabited

/-- The problem statement --/
theorem exists_excursion_with_frequent_participants
  (n : Nat) -- number of excursions
  (excursions : Finset Excursion)
  (h1 : excursions.card = n) -- there are n excursions
  (h2 : ∀ e ∈ excursions, e.participants.card ≥ 4) -- each excursion has at least 4 participants
  (h3 : ∀ e ∈ excursions, e.participants.card ≤ 20) -- each excursion has at most 20 participants
  : ∃ e ∈ excursions, ∀ s ∈ e.participants,
    (excursions.filter (λ ex : Excursion => s ∈ ex.participants)).card ≥ n / 17 :=
sorry

end exists_excursion_with_frequent_participants_l3539_353924


namespace minimum_time_for_given_problem_l3539_353999

/-- Represents the problem of replacing shades in chandeliers --/
structure ChandelierProblem where
  num_chandeliers : ℕ
  shades_per_chandelier : ℕ
  time_per_shade : ℕ
  num_electricians : ℕ

/-- Calculates the minimum time required to replace all shades --/
def minimum_replacement_time (p : ChandelierProblem) : ℕ :=
  let total_shades := p.num_chandeliers * p.shades_per_chandelier
  let total_work_time := total_shades * p.time_per_shade
  (total_work_time + p.num_electricians - 1) / p.num_electricians

/-- Theorem stating the minimum time for the given problem --/
theorem minimum_time_for_given_problem :
  let p : ChandelierProblem := {
    num_chandeliers := 60,
    shades_per_chandelier := 4,
    time_per_shade := 5,
    num_electricians := 48
  }
  minimum_replacement_time p = 25 := by sorry


end minimum_time_for_given_problem_l3539_353999


namespace frustum_volume_l3539_353939

/-- The volume of a frustum formed by cutting a triangular pyramid parallel to its base --/
theorem frustum_volume (base_edge : ℝ) (altitude : ℝ) (small_base_edge : ℝ) (small_altitude : ℝ) :
  base_edge = 18 →
  altitude = 9 →
  small_base_edge = 9 →
  small_altitude = 3 →
  ∃ (v : ℝ), v = 212.625 * Real.sqrt 3 ∧ v = 
    ((1/3 * (Real.sqrt 3 / 4) * base_edge^2 * altitude) - 
     (1/3 * (Real.sqrt 3 / 4) * small_base_edge^2 * small_altitude)) :=
by sorry

end frustum_volume_l3539_353939


namespace zero_rational_others_irrational_l3539_353972

-- Define rational numbers
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- State the theorem
theorem zero_rational_others_irrational :
  IsRational 0 ∧ ¬IsRational (-Real.pi) ∧ ¬IsRational (Real.sqrt 3) ∧ ¬IsRational (Real.sqrt 2) := by
  sorry

end zero_rational_others_irrational_l3539_353972


namespace exactly_two_late_probability_l3539_353913

/-- The probability of a worker being late on any given day -/
def p_late : ℚ := 1 / 40

/-- The probability of a worker being on time on any given day -/
def p_on_time : ℚ := 1 - p_late

/-- The number of workers considered -/
def n_workers : ℕ := 3

/-- The number of workers that need to be late -/
def n_late : ℕ := 2

theorem exactly_two_late_probability :
  (n_workers.choose n_late : ℚ) * p_late ^ n_late * p_on_time ^ (n_workers - n_late) = 117 / 64000 :=
sorry

end exactly_two_late_probability_l3539_353913


namespace line_tangent_to_parabola_l3539_353944

/-- The line 4x + 3y + k = 0 is tangent to the parabola y^2 = 16x if and only if k = 9 -/
theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, 4 * x + 3 * y + k = 0 → y^2 = 16 * x) ↔ k = 9 := by
  sorry

end line_tangent_to_parabola_l3539_353944


namespace joe_lift_weight_l3539_353993

theorem joe_lift_weight (first_lift second_lift : ℕ) 
  (total_weight : first_lift + second_lift = 900)
  (lift_relation : 2 * first_lift = second_lift + 300) :
  first_lift = 400 := by
  sorry

end joe_lift_weight_l3539_353993


namespace complex_modulus_sqrt_two_l3539_353955

theorem complex_modulus_sqrt_two (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_sqrt_two_l3539_353955


namespace rhombus_area_l3539_353918

/-- Given a rhombus with perimeter 48 and sum of diagonals 26, its area is 25 -/
theorem rhombus_area (perimeter : ℝ) (diagonal_sum : ℝ) (area : ℝ) : 
  perimeter = 48 → diagonal_sum = 26 → area = 25 := by
  sorry

end rhombus_area_l3539_353918


namespace lattice_points_on_hyperbola_l3539_353922

theorem lattice_points_on_hyperbola : 
  ∃! (s : Finset (ℤ × ℤ)), s.card = 4 ∧ ∀ (x y : ℤ), (x, y) ∈ s ↔ x^2 - y^2 = 77 := by
  sorry

end lattice_points_on_hyperbola_l3539_353922


namespace power_product_equality_l3539_353980

theorem power_product_equality (a : ℝ) : a^2 * (-a)^2 = a^4 := by
  sorry

end power_product_equality_l3539_353980


namespace extra_time_with_speed_decrease_l3539_353958

/-- Given a 20% decrease in speed and an original travel time of 40 minutes,
    prove that the extra time taken to cover the same distance is 10 minutes. -/
theorem extra_time_with_speed_decrease (original_speed : ℝ) (original_time : ℝ) 
  (h1 : original_time = 40) 
  (h2 : original_speed > 0) : 
  let decreased_speed := 0.8 * original_speed
  let new_time := (original_speed * original_time) / decreased_speed
  new_time - original_time = 10 := by
  sorry

end extra_time_with_speed_decrease_l3539_353958


namespace simplify_expressions_l3539_353953

variable (a b x y : ℝ)

theorem simplify_expressions :
  (2 * a - (a + b) = a - b) ∧
  ((x^2 - 2*y^2) - 2*(3*y^2 - 2*x^2) = 5*x^2 - 8*y^2) := by
  sorry

end simplify_expressions_l3539_353953


namespace expression_evaluation_l3539_353966

theorem expression_evaluation : 1 + 3 + 5 + 7 - (2 + 4 + 6) + 3^2 + 5^2 = 38 := by
  sorry

end expression_evaluation_l3539_353966


namespace expression_value_at_three_l3539_353927

theorem expression_value_at_three :
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x - 5)
  f 3 = 5 := by
sorry

end expression_value_at_three_l3539_353927


namespace problem_statement_l3539_353947

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  -- Statement 1
  (a^2 - b^2 = 1 → a - b < 1) ∧
  -- Statement 2
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1/b - 1/a = 1 ∧ a - b ≥ 1) ∧
  -- Statement 3
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ |Real.sqrt a - Real.sqrt b| = 1 ∧ |a - b| ≥ 1) ∧
  -- Statement 4
  (|a^3 - b^3| = 1 → |a - b| < 1) :=
by sorry

end problem_statement_l3539_353947


namespace ellipse_properties_l3539_353986

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = -8 * x

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  y = x - 2

theorem ellipse_properties (a b c : ℝ) :
  a > b ∧ b > 0 ∧
  ellipse a b (-Real.sqrt 3) 1 ∧
  c = 2 ∧
  b^2 = a^2 - 4 ∧
  (∃ x y, ellipse a b x y ∧ parabola x y) →
  (a^2 = 6 ∧
   (∀ x y, ellipse a b x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
   (∃ x₁ y₁ x₂ y₂, 
      ellipse a b x₁ y₁ ∧ 
      ellipse a b x₂ y₂ ∧
      line_l x₁ y₁ ∧
      line_l x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = 6)) :=
by sorry

end ellipse_properties_l3539_353986


namespace quadratic_equation_solution_l3539_353933

theorem quadratic_equation_solution (m : ℝ) 
  (x₁ x₂ : ℝ) -- Two real roots
  (h1 : x₁^2 - m*x₁ + 2*m - 1 = 0) -- x₁ satisfies the equation
  (h2 : x₂^2 - m*x₂ + 2*m - 1 = 0) -- x₂ satisfies the equation
  (h3 : x₁^2 + x₂^2 = 7) -- Given condition
  : m = -1 := by
  sorry

end quadratic_equation_solution_l3539_353933


namespace xy_value_l3539_353950

theorem xy_value (x y : ℝ) (h : (|x| - 1)^2 + (2*y + 1)^2 = 0) : x * y = 1/2 := by
  sorry

end xy_value_l3539_353950


namespace hohyeon_taller_than_seulgi_l3539_353916

/-- Seulgi's height in centimeters -/
def seulgi_height : ℕ := 159

/-- Hohyeon's height in centimeters -/
def hohyeon_height : ℕ := 162

/-- Theorem stating that Hohyeon is taller than Seulgi -/
theorem hohyeon_taller_than_seulgi : hohyeon_height > seulgi_height := by
  sorry

end hohyeon_taller_than_seulgi_l3539_353916


namespace volume_range_l3539_353934

/-- Pyramid S-ABCD with square base ABCD and isosceles right triangle side face SAD -/
structure Pyramid where
  /-- Side length of the square base ABCD -/
  base_side : ℝ
  /-- Length of SC -/
  sc_length : ℝ
  /-- The base ABCD is a square with side length 2 -/
  base_side_eq_two : base_side = 2
  /-- The side face SAD is an isosceles right triangle with SD as the hypotenuse -/
  sad_isosceles_right : True  -- This condition is implied by the structure
  /-- 2√2 ≤ SC ≤ 4 -/
  sc_range : 2 * Real.sqrt 2 ≤ sc_length ∧ sc_length ≤ 4

/-- Volume of the pyramid -/
def volume (p : Pyramid) : ℝ := sorry

/-- Theorem stating the range of the pyramid's volume -/
theorem volume_range (p : Pyramid) : 
  (4 * Real.sqrt 3) / 3 ≤ volume p ∧ volume p ≤ 8 / 3 := by
  sorry

end volume_range_l3539_353934


namespace binary_conversion_l3539_353925

-- Define the binary number
def binary_num : List Bool := [true, true, false, false, true, true]

-- Function to convert binary to decimal
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

-- Function to convert decimal to base 5
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

-- Theorem statement
theorem binary_conversion :
  binary_to_decimal binary_num = 51 ∧
  decimal_to_base5 (binary_to_decimal binary_num) = [2, 0, 1] := by
  sorry

end binary_conversion_l3539_353925


namespace sum_interior_angles_polygon_l3539_353937

/-- The sum of interior angles of an n-sided polygon -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- The number of triangles formed by drawing diagonals from one vertex -/
def num_triangles (n : ℕ) : ℕ := n - 2

/-- The number of diagonals drawn from one vertex -/
def num_diagonals (n : ℕ) : ℕ := n - 3

theorem sum_interior_angles_polygon (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (num_triangles n) * 180 :=
by sorry


end sum_interior_angles_polygon_l3539_353937
