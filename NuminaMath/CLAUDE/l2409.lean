import Mathlib

namespace inequality_of_four_variables_l2409_240982

theorem inequality_of_four_variables (a b c d : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := by
  sorry

end inequality_of_four_variables_l2409_240982


namespace race_head_start_l2409_240921

theorem race_head_start (L : ℝ) (vₐ vᵦ : ℝ) (h : vₐ = (17 / 14) * vᵦ) :
  let x := (3 / 17) * L
  L / vₐ = (L - x) / vᵦ :=
by sorry

end race_head_start_l2409_240921


namespace intersection_of_M_and_N_l2409_240995

def M : Set ℝ := {x | x + 2 ≥ 0}
def N : Set ℝ := {x | x - 1 < 0}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l2409_240995


namespace min_bottles_for_27_people_min_bottles_sufficient_l2409_240953

/-- The minimum number of bottles needed to be purchased for a given number of people,
    given that 3 empty bottles can be exchanged for 1 full bottle -/
def min_bottles_to_purchase (num_people : ℕ) : ℕ :=
  (2 * num_people + 2) / 3

/-- Proof that for 27 people, the minimum number of bottles to purchase is 18 -/
theorem min_bottles_for_27_people :
  min_bottles_to_purchase 27 = 18 := by
  sorry

/-- Proof that the calculated minimum number of bottles is sufficient for all people -/
theorem min_bottles_sufficient (num_people : ℕ) :
  min_bottles_to_purchase num_people + (min_bottles_to_purchase num_people) / 2 ≥ num_people := by
  sorry

end min_bottles_for_27_people_min_bottles_sufficient_l2409_240953


namespace number_of_divisors_2310_l2409_240946

/-- The number of positive divisors of 2310 is 32. -/
theorem number_of_divisors_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end number_of_divisors_2310_l2409_240946


namespace ellen_croissants_l2409_240959

/-- The price of a can of cola in pence -/
def cola_price : ℕ := sorry

/-- The price of a croissant in pence -/
def croissant_price : ℕ := sorry

/-- The total amount of money Ellen has in pence -/
def total_money : ℕ := sorry

/-- Assumption that Ellen can spend all her money on 6 cans of cola and 7 croissants -/
axiom combination1 : 6 * cola_price + 7 * croissant_price = total_money

/-- Assumption that Ellen can spend all her money on 8 cans of cola and 4 croissants -/
axiom combination2 : 8 * cola_price + 4 * croissant_price = total_money

/-- Theorem stating that Ellen can buy 16 croissants if she decides to buy only croissants -/
theorem ellen_croissants : total_money / croissant_price = 16 := by sorry

end ellen_croissants_l2409_240959


namespace curve_description_l2409_240909

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def right_half (x y : ℝ) : Prop := x = Real.sqrt (1 - y^2)

def lower_half (x y : ℝ) : Prop := y = -Real.sqrt (1 - x^2)

def curve_equation (x y : ℝ) : Prop :=
  (x - Real.sqrt (1 - y^2)) * (y + Real.sqrt (1 - x^2)) = 0

theorem curve_description (x y : ℝ) :
  unit_circle x y ∧ (right_half x y ∨ lower_half x y) ↔ curve_equation x y :=
sorry

end curve_description_l2409_240909


namespace large_hexagon_area_l2409_240908

/-- Represents a regular hexagon -/
structure RegularHexagon where
  area : ℝ

/-- The large regular hexagon containing smaller hexagons -/
def large_hexagon : RegularHexagon := sorry

/-- One of the smaller regular hexagons -/
def small_hexagon : RegularHexagon := sorry

/-- The number of small hexagons in the large hexagon -/
def num_small_hexagons : ℕ := 7

/-- The number of small hexagons in the shaded area -/
def num_shaded_hexagons : ℕ := 6

/-- The area of the shaded part (6 small hexagons) -/
def shaded_area : ℝ := 180

theorem large_hexagon_area : large_hexagon.area = 270 := by sorry

end large_hexagon_area_l2409_240908


namespace one_is_optimal_l2409_240903

/-- Represents the number of teams that chose a particular number -/
def TeamChoices := ℕ → ℕ

/-- Calculates the score based on the game rules -/
def score (N : ℕ) (choices : TeamChoices) : ℕ :=
  if choices N > N then N else 0

/-- Theorem stating that 1 is the optimal choice -/
theorem one_is_optimal :
  ∀ (N : ℕ) (choices : TeamChoices),
    0 ≤ N ∧ N ≤ 20 →
    score 1 choices ≥ score N choices :=
sorry

end one_is_optimal_l2409_240903


namespace unique_solution_l2409_240964

/-- For every positive integer n, there exists a positive integer c_n
    such that a^n + b^n = c_n^(n+1) -/
def satisfies_condition (a b : ℕ+) : Prop :=
  ∀ n : ℕ+, ∃ c_n : ℕ+, (a : ℕ)^(n : ℕ) + (b : ℕ)^(n : ℕ) = (c_n : ℕ)^((n : ℕ) + 1)

/-- The only pair of positive integers (a,b) satisfying the condition is (2,2) -/
theorem unique_solution :
  ∀ a b : ℕ+, satisfies_condition a b ↔ a = 2 ∧ b = 2 := by
  sorry

end unique_solution_l2409_240964


namespace arithmetic_sequence_common_difference_l2409_240955

/-- An arithmetic sequence with the given properties has a common difference of 2 -/
theorem arithmetic_sequence_common_difference : ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) > a n) →  -- increasing sequence
  (a 1 + a 2 + a 3 = 12) →  -- sum of first three terms
  ((a 3)^2 = a 2 * (a 4 + 1)) →  -- geometric sequence condition
  (∃ d : ℝ, ∀ n, a (n + 1) - a n = d) →  -- arithmetic sequence
  (∃ d : ℝ, (∀ n, a (n + 1) - a n = d) ∧ d = 2) :=
by sorry

end arithmetic_sequence_common_difference_l2409_240955


namespace number_of_girls_l2409_240966

/-- Given a group of children with specific characteristics, prove the number of girls. -/
theorem number_of_girls (total_children happy_children sad_children neutral_children boys happy_boys sad_girls neutral_boys : ℕ) 
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neutral_children = 20)
  (h5 : boys = 19)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neutral_boys = 7)
  (h9 : happy_children + sad_children + neutral_children = total_children) :
  total_children - boys = 41 := by
  sorry

end number_of_girls_l2409_240966


namespace existence_of_epsilon_and_u_l2409_240961

theorem existence_of_epsilon_and_u (n : ℕ+) :
  ∃ (ε : ℝ), 0 < ε ∧ ε < (1 : ℝ) / 2014 ∧
  ∀ (a : Fin n → ℝ), (∀ i, 0 < a i) →
  ∃ (u : ℝ), u > 0 ∧ ∀ i, ε < u * (a i) - ⌊u * (a i)⌋ ∧ u * (a i) - ⌊u * (a i)⌋ < (1 : ℝ) / 2014 :=
sorry

end existence_of_epsilon_and_u_l2409_240961


namespace total_orange_balloons_l2409_240994

-- Define the initial number of orange balloons
def initial_orange_balloons : ℝ := 9.0

-- Define the number of orange balloons found
def found_orange_balloons : ℝ := 2.0

-- Theorem to prove
theorem total_orange_balloons :
  initial_orange_balloons + found_orange_balloons = 11.0 := by
  sorry

end total_orange_balloons_l2409_240994


namespace fixed_point_of_exponential_function_l2409_240960

theorem fixed_point_of_exponential_function (a : ℝ) :
  let f : ℝ → ℝ := λ x => a^(x - 3) + 3
  f 3 = 4 := by sorry

end fixed_point_of_exponential_function_l2409_240960


namespace tangent_triangle_area_l2409_240932

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a triangle formed by three points -/
structure Triangle where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- Given three mutually externally tangent circles with radii 1, 2, and 3,
    returns the triangle formed by their points of tangency -/
def tangentTriangle (c1 c2 c3 : Circle) : Triangle := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Three mutually externally tangent circles with radii 1, 2, and 3 -/
def circle1 : Circle := { center := (0, 0), radius := 1 }
def circle2 : Circle := { center := (3, 0), radius := 2 }
def circle3 : Circle := { center := (0, 4), radius := 3 }

theorem tangent_triangle_area :
  triangleArea (tangentTriangle circle1 circle2 circle3) = 4/5 := by
  sorry

end tangent_triangle_area_l2409_240932


namespace special_property_implies_interval_l2409_240977

/-- A positive integer n < 1000 has the property that 1/n is a repeating decimal
    of period 3 and 1/(n+6) is a repeating decimal of period 2 -/
def has_special_property (n : ℕ) : Prop :=
  n > 0 ∧ n < 1000 ∧
  ∃ (a b c : ℕ), (1 : ℚ) / n = (a * 100 + b * 10 + c : ℚ) / 999 ∧
  ∃ (x y : ℕ), (1 : ℚ) / (n + 6) = (x * 10 + y : ℚ) / 99

theorem special_property_implies_interval :
  ∀ n : ℕ, has_special_property n → n ∈ Set.Icc 1 250 :=
by
  sorry

end special_property_implies_interval_l2409_240977


namespace trigonometric_equation_solution_l2409_240976

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sin (2025 * x))^4 + (Real.cos (2016 * x))^2019 * (Real.cos (2025 * x))^2018 = 1 ↔ 
  (∃ n : ℤ, x = π / 4050 + π * n / 2025) ∨ (∃ k : ℤ, x = π * k / 9) :=
sorry

end trigonometric_equation_solution_l2409_240976


namespace zoo_pictures_l2409_240967

/-- Represents the number of pictures Debby took at the zoo -/
def Z : ℕ := sorry

/-- The total number of pictures Debby initially took -/
def total_initial : ℕ := Z + 12

/-- The number of pictures Debby deleted -/
def deleted : ℕ := 14

/-- The number of pictures Debby has remaining -/
def remaining : ℕ := 22

theorem zoo_pictures : Z = 24 :=
  sorry

end zoo_pictures_l2409_240967


namespace part_dimensions_l2409_240919

/-- Given a base dimension with upper and lower tolerances, 
    prove the maximum and minimum allowable dimensions. -/
theorem part_dimensions 
  (base : ℝ) 
  (upper_tolerance : ℝ) 
  (lower_tolerance : ℝ) 
  (h_base : base = 7) 
  (h_upper : upper_tolerance = 0.05) 
  (h_lower : lower_tolerance = 0.02) : 
  (base + upper_tolerance = 7.05) ∧ (base - lower_tolerance = 6.98) := by
  sorry

end part_dimensions_l2409_240919


namespace smallest_three_way_sum_of_squares_l2409_240947

/-- A function that returns true if a number can be expressed as the sum of two squares -/
def isSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^2 + b^2 = n

/-- A function that counts the number of ways a number can be expressed as the sum of two squares -/
def countSumOfTwoSquares (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card

/-- The theorem stating that 325 is the smallest number that can be expressed as the sum of two squares in three distinct ways -/
theorem smallest_three_way_sum_of_squares :
  (∀ m : ℕ, m < 325 → countSumOfTwoSquares m < 3) ∧
  countSumOfTwoSquares 325 = 3 := by
  sorry

end smallest_three_way_sum_of_squares_l2409_240947


namespace school_field_trip_cost_l2409_240974

/-- Calculates the total cost for a school field trip to a farm -/
theorem school_field_trip_cost (num_students : ℕ) (num_adults : ℕ) 
  (student_fee : ℕ) (adult_fee : ℕ) : 
  num_students = 35 → num_adults = 4 → student_fee = 5 → adult_fee = 6 →
  num_students * student_fee + num_adults * adult_fee = 199 :=
by sorry

end school_field_trip_cost_l2409_240974


namespace problem_solution_l2409_240981

open Set

def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def M (a : ℝ) : Set ℝ := {x | x ≤ a + 6}

theorem problem_solution (a : ℝ) (h : A ⊆ M a) :
  ((𝒰 \ B) ∩ A = {x | -3 ≤ x ∧ x ≤ 2}) ∧ (a ≥ -3) :=
by sorry

end problem_solution_l2409_240981


namespace max_similar_triangles_five_points_l2409_240970

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle formed by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- The set of all triangles formed by choosing 3 points from a set of 5 points -/
def allTriangles (points : Finset Point) : Finset Triangle := sorry

/-- The set of all similar triangles from a set of triangles -/
def similarTriangles (triangles : Finset Triangle) : Finset (Finset Triangle) := sorry

/-- The theorem stating that the maximum number of similar triangles from 5 points is 4 -/
theorem max_similar_triangles_five_points (points : Finset Point) :
  points.card = 5 →
  (similarTriangles (allTriangles points)).sup (λ s => s.card) ≤ 4 :=
sorry

end max_similar_triangles_five_points_l2409_240970


namespace cost_of_lettuce_cost_of_lettuce_is_one_dollar_l2409_240965

/-- The cost of the head of lettuce in Lauren's grocery purchase --/
theorem cost_of_lettuce : ℝ := by
  -- Define the known costs
  let meat_cost : ℝ := 2 * 3.5
  let buns_cost : ℝ := 1.5
  let tomato_cost : ℝ := 1.5 * 2
  let pickles_cost : ℝ := 2.5 - 1

  -- Define the total bill and change
  let total_paid : ℝ := 20
  let change : ℝ := 6

  -- Define the actual spent amount
  let actual_spent : ℝ := total_paid - change

  -- Define the sum of known costs
  let known_costs : ℝ := meat_cost + buns_cost + tomato_cost + pickles_cost

  -- The cost of lettuce is the difference between actual spent and known costs
  have lettuce_cost : ℝ := actual_spent - known_costs

  -- Prove that the cost of lettuce is 1.00
  sorry

/-- The cost of the head of lettuce is $1.00 --/
theorem cost_of_lettuce_is_one_dollar : cost_of_lettuce = 1 := by sorry

end cost_of_lettuce_cost_of_lettuce_is_one_dollar_l2409_240965


namespace brothers_age_sum_l2409_240900

/-- Two brothers with an age difference of 4 years -/
structure Brothers where
  older_age : ℕ
  younger_age : ℕ
  age_difference : older_age = younger_age + 4

/-- The sum of the brothers' ages -/
def age_sum (b : Brothers) : ℕ := b.older_age + b.younger_age

/-- Theorem: The sum of the ages of two brothers who are 4 years apart,
    where the older one is 16 years old, is 28 years. -/
theorem brothers_age_sum :
  ∀ (b : Brothers), b.older_age = 16 → age_sum b = 28 := by
  sorry

end brothers_age_sum_l2409_240900


namespace parabola_vertex_l2409_240993

/-- The vertex of a parabola given by y^2 - 4y + 2x + 7 = 0 is (-3/2, 2) -/
theorem parabola_vertex :
  let f : ℝ → ℝ → ℝ := λ x y => y^2 - 4*y + 2*x + 7
  ∃! (vx vy : ℝ), (∀ x y, f x y = 0 → (x - vx)^2 ≤ (x + 3/2)^2 ∧ y = vy) ∧ vx = -3/2 ∧ vy = 2 :=
sorry

end parabola_vertex_l2409_240993


namespace square_sum_plus_sum_squares_l2409_240997

theorem square_sum_plus_sum_squares : (6 + 10)^2 + (6^2 + 10^2) = 392 := by
  sorry

end square_sum_plus_sum_squares_l2409_240997


namespace subtract_seven_percent_l2409_240988

theorem subtract_seven_percent (a : ℝ) : a - 0.07 * a = 0.93 * a := by
  sorry

end subtract_seven_percent_l2409_240988


namespace tangent_slope_implies_a_l2409_240915

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a (a : ℝ) :
  curve a (-1) = a + 2 →
  curve_derivative a (-1) = 8 →
  a = -6 := by sorry

end tangent_slope_implies_a_l2409_240915


namespace age_problem_l2409_240948

theorem age_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 28 →
  (a + c) / 2 = 29 →
  b = 26 := by
sorry

end age_problem_l2409_240948


namespace abs_m_minus_n_equals_five_l2409_240923

theorem abs_m_minus_n_equals_five (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : 
  |m - n| = 5 := by
sorry

end abs_m_minus_n_equals_five_l2409_240923


namespace simplify_calculations_l2409_240972

theorem simplify_calculations :
  (3.5 * 10.1 = 35.35) ∧
  (0.58 * 98 = 56.84) ∧
  (3.6 * 6.91 + 6.4 * 6.91 = 69.1) ∧
  ((19.1 - (1.64 + 2.36)) / 2.5 = 6.04) := by
  sorry

end simplify_calculations_l2409_240972


namespace average_velocity_first_30_seconds_l2409_240984

-- Define the velocity function
def v (t : ℝ) : ℝ := t^2 - 3*t + 8

-- Define the time interval
def t_start : ℝ := 0
def t_end : ℝ := 30

-- Theorem statement
theorem average_velocity_first_30_seconds :
  (∫ t in t_start..t_end, v t) / (t_end - t_start) = 263 := by
  sorry

end average_velocity_first_30_seconds_l2409_240984


namespace museum_ticket_price_l2409_240945

theorem museum_ticket_price 
  (group_size : ℕ) 
  (total_paid : ℚ) 
  (tax_rate : ℚ) 
  (h1 : group_size = 25) 
  (h2 : total_paid = 945) 
  (h3 : tax_rate = 5 / 100) : 
  ∃ (face_value : ℚ), 
    face_value = 36 ∧ 
    total_paid = group_size * face_value * (1 + tax_rate) := by
  sorry

end museum_ticket_price_l2409_240945


namespace partition_M_theorem_l2409_240998

/-- The set M containing elements from 1 to 12 -/
def M : Finset ℕ := Finset.range 12

/-- Predicate to check if a set is a valid partition of M -/
def is_valid_partition (A B C : Finset ℕ) : Prop :=
  A ∪ B ∪ C = M ∧ A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
  A.card = 4 ∧ B.card = 4 ∧ C.card = 4

/-- Predicate to check if C satisfies the ordering condition -/
def C_ordered (C : Finset ℕ) : Prop :=
  ∃ c₁ c₂ c₃ c₄, C = {c₁, c₂, c₃, c₄} ∧ c₁ < c₂ ∧ c₂ < c₃ ∧ c₃ < c₄

/-- Predicate to check if A, B, and C satisfy the sum condition -/
def sum_condition (A B C : Finset ℕ) : Prop :=
  ∃ a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ c₁ c₂ c₃ c₄,
    A = {a₁, a₂, a₃, a₄} ∧ B = {b₁, b₂, b₃, b₄} ∧ C = {c₁, c₂, c₃, c₄} ∧
    a₁ + b₁ = c₁ ∧ a₂ + b₂ = c₂ ∧ a₃ + b₃ = c₃ ∧ a₄ + b₄ = c₄

theorem partition_M_theorem :
  ∀ A B C : Finset ℕ,
    is_valid_partition A B C →
    C_ordered C →
    sum_condition A B C →
    C = {8, 9, 10, 12} ∨ C = {7, 9, 11, 12} ∨ C = {6, 10, 11, 12} :=
sorry

end partition_M_theorem_l2409_240998


namespace right_triangle_arm_square_l2409_240914

/-- In a right triangle with hypotenuse c and arms a and b, where c = a + 2,
    the square of b is equal to 4a + 4. -/
theorem right_triangle_arm_square (a c : ℝ) (h1 : c = a + 2) :
  ∃ b : ℝ, b^2 = 4*a + 4 ∧ a^2 + b^2 = c^2 := by sorry

end right_triangle_arm_square_l2409_240914


namespace cos_alpha_for_point_l2409_240985

/-- Given a point P(4, -3) on the terminal side of angle α, prove that cos(α) = 4/5 -/
theorem cos_alpha_for_point (α : Real) : 
  (∃ (P : Real × Real), P = (4, -3) ∧ P.1 = 4 * Real.cos α ∧ P.2 = 4 * Real.sin α) →
  Real.cos α = 4/5 := by
sorry

end cos_alpha_for_point_l2409_240985


namespace negation_of_existence_l2409_240930

theorem negation_of_existence (l : ℝ) : 
  (¬ ∃ x : ℝ, x + l ≥ 0) ↔ (∀ x : ℝ, x + l < 0) :=
sorry

end negation_of_existence_l2409_240930


namespace inequality_proof_l2409_240924

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (b / (a - b) > c / (a - c)) ∧
  (a / (a + b) < (a + c) / (a + b + c)) ∧
  (1 / (a - b) + 1 / (b - c) ≥ 4 / (a - c)) := by
  sorry

end inequality_proof_l2409_240924


namespace vector_u_satisfies_equation_l2409_240983

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 0, 2]

theorem vector_u_satisfies_equation :
  let u : Matrix (Fin 2) (Fin 1) ℝ := !![5/273; 8/21]
  (B^5 + B^3 + B) * u = !![5; 16] := by
  sorry

end vector_u_satisfies_equation_l2409_240983


namespace polygon_exterior_angles_l2409_240954

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n : ℝ) * exterior_angle = 360 → exterior_angle = 30 → n = 12 := by
  sorry

end polygon_exterior_angles_l2409_240954


namespace exactly_two_successes_probability_l2409_240942

/-- The probability of success in a single trial -/
def p : ℚ := 3/5

/-- The number of trials -/
def n : ℕ := 5

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The binomial probability formula -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1-p)^(n-k)

/-- The main theorem: probability of exactly 2 successes in 5 trials with p = 3/5 is 144/625 -/
theorem exactly_two_successes_probability :
  binomial_probability n k p = 144/625 := by
  sorry

end exactly_two_successes_probability_l2409_240942


namespace price_reduction_correct_l2409_240904

/-- The final price of a medication after two price reductions -/
def final_price (m : ℝ) (x : ℝ) : ℝ := m * (1 - x)^2

/-- Theorem stating that the final price after two reductions is correct -/
theorem price_reduction_correct (m : ℝ) (x : ℝ) (y : ℝ) 
  (hm : m > 0) (hx : 0 ≤ x ∧ x < 1) :
  y = final_price m x ↔ y = m * (1 - x)^2 := by sorry

end price_reduction_correct_l2409_240904


namespace f_402_equals_zero_l2409_240956

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom period_condition : ∀ x, f (x + 4) - f x = 2 * f 2
axiom symmetry_condition : ∀ x, f (2 - x) = f x

-- Theorem to prove
theorem f_402_equals_zero : f 402 = 0 := by
  sorry

end f_402_equals_zero_l2409_240956


namespace white_sox_games_lost_l2409_240933

theorem white_sox_games_lost (total_games won_games : ℕ) 
  (h1 : total_games = 162)
  (h2 : won_games = 99)
  (h3 : won_games = lost_games + 36) : lost_games = 63 :=
by
  sorry

end white_sox_games_lost_l2409_240933


namespace parallel_vectors_a_value_l2409_240999

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_a_value :
  let m : ℝ × ℝ := (2, 1)
  let n : ℝ × ℝ := (4, a)
  ∀ a : ℝ, are_parallel m n → a = 2 := by
sorry

end parallel_vectors_a_value_l2409_240999


namespace x_y_relation_l2409_240928

theorem x_y_relation (Q : ℝ) (x y : ℝ) (hx : x = Real.sqrt (Q/2 + Real.sqrt (Q/2)))
  (hy : y = Real.sqrt (Q/2 - Real.sqrt (Q/2))) :
  (x^6 + y^6) / 40 = 10 := by
  sorry

end x_y_relation_l2409_240928


namespace legs_of_special_triangle_l2409_240935

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of one leg of the triangle -/
  leg1 : ℝ
  /-- The length of the other leg of the triangle -/
  leg2 : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The distance from the center of the inscribed circle to one end of the hypotenuse -/
  dist1 : ℝ
  /-- The distance from the center of the inscribed circle to the other end of the hypotenuse -/
  dist2 : ℝ
  /-- The leg1 is positive -/
  leg1_pos : 0 < leg1
  /-- The leg2 is positive -/
  leg2_pos : 0 < leg2
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The dist1 is positive -/
  dist1_pos : 0 < dist1
  /-- The dist2 is positive -/
  dist2_pos : 0 < dist2
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : leg1^2 + leg2^2 = (dist1 + dist2)^2
  /-- The radius is related to the legs and distances as per the properties of an inscribed circle -/
  inscribed_circle : radius = (leg1 + leg2 - dist1 - dist2) / 2

/-- 
If the center of the inscribed circle in a right triangle is at distances √5 and √10 
from the ends of the hypotenuse, then the legs of the triangle are 3 and 4.
-/
theorem legs_of_special_triangle (t : RightTriangleWithInscribedCircle) 
  (h1 : t.dist1 = Real.sqrt 5) (h2 : t.dist2 = Real.sqrt 10) : 
  (t.leg1 = 3 ∧ t.leg2 = 4) ∨ (t.leg1 = 4 ∧ t.leg2 = 3) := by
  sorry

end legs_of_special_triangle_l2409_240935


namespace unique_valid_number_l2409_240989

def is_valid_number (n : ℕ) : Prop :=
  n % 25 = 0 ∧ n % 35 = 0 ∧
  (∃ (a b c : ℕ), a * n ≤ 1050 ∧ b * n ≤ 1050 ∧ c * n ≤ 1050 ∧
   a < b ∧ b < c ∧
   ∀ (x : ℕ), x * n ≤ 1050 → x = a ∨ x = b ∨ x = c)

theorem unique_valid_number : 
  is_valid_number 350 ∧ ∀ (m : ℕ), is_valid_number m → m = 350 :=
sorry

end unique_valid_number_l2409_240989


namespace ella_video_game_spending_l2409_240962

/-- Proves that Ella spent $100 on video games last year given her current salary and spending habits -/
theorem ella_video_game_spending (new_salary : ℝ) (raise_percentage : ℝ) (video_game_percentage : ℝ) :
  new_salary = 275 →
  raise_percentage = 0.1 →
  video_game_percentage = 0.4 →
  (new_salary / (1 + raise_percentage)) * video_game_percentage = 100 := by
  sorry

end ella_video_game_spending_l2409_240962


namespace intersection_sum_l2409_240996

theorem intersection_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + 7 ∧ y = 4 * x + b → x = 8 ∧ y = 11) → 
  b + m = -20.5 := by
sorry

end intersection_sum_l2409_240996


namespace smallest_odd_integer_triangle_perimeter_l2409_240980

/-- A function that checks if three numbers form a valid triangle --/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that generates three consecutive odd integers --/
def consecutive_odd_integers (n : ℕ) : (ℕ × ℕ × ℕ) :=
  (2*n + 1, 2*n + 3, 2*n + 5)

/-- The theorem stating the smallest possible perimeter of a triangle with consecutive odd integer side lengths --/
theorem smallest_odd_integer_triangle_perimeter :
  ∃ (n : ℕ), 
    let (a, b, c) := consecutive_odd_integers n
    is_valid_triangle a b c ∧
    ∀ (m : ℕ), m < n → ¬(is_valid_triangle (2*m + 1) (2*m + 3) (2*m + 5)) ∧
    a + b + c = 15 :=
sorry

end smallest_odd_integer_triangle_perimeter_l2409_240980


namespace g_derivative_at_midpoint_sign_l2409_240901

/-- The function g(x) defined as x + a * ln(x) - k * x^2 --/
noncomputable def g (a k x : ℝ) : ℝ := x + a * Real.log x - k * x^2

/-- The derivative of g(x) --/
noncomputable def g' (a k x : ℝ) : ℝ := 1 + a / x - 2 * k * x

theorem g_derivative_at_midpoint_sign (a k x₁ x₂ : ℝ) 
  (hk : k > 0) 
  (hx : x₁ ≠ x₂) 
  (hz₁ : g a k x₁ = 0) 
  (hz₂ : g a k x₂ = 0) :
  (a > 0 → g' a k ((x₁ + x₂) / 2) < 0) ∧
  (a < 0 → g' a k ((x₁ + x₂) / 2) > 0) :=
by sorry

end g_derivative_at_midpoint_sign_l2409_240901


namespace sand_pile_volume_l2409_240936

/-- The volume of a cylindrical pile of sand -/
theorem sand_pile_volume :
  ∀ (r h d : ℝ),
  d = 8 →                -- diameter is 8 feet
  r = d / 2 →            -- radius is half the diameter
  h = 2 * r →            -- height is twice the radius
  π * r^2 * h = 128 * π  -- volume is 128π cubic feet
  := by sorry

end sand_pile_volume_l2409_240936


namespace exists_permutation_1984_divisible_by_7_l2409_240978

/-- A permutation of the digits of 1984 -/
def Permutation1984 : Type :=
  { p : Nat // p ∈ ({1498, 1849, 1948, 1984, 1894, 1489, 9148} : Set Nat) }

/-- Theorem: For any positive integer N, there exists a permutation of 1984's digits
    that when added to N, is divisible by 7 -/
theorem exists_permutation_1984_divisible_by_7 (N : Nat) :
  ∃ (p : Permutation1984), 7 ∣ (N + p.val) := by
  sorry

end exists_permutation_1984_divisible_by_7_l2409_240978


namespace number_of_men_in_first_group_l2409_240949

-- Define the number of men in the first group
def M : ℕ := sorry

-- Define the given conditions
def hours_per_day_group1 : ℕ := 10
def earnings_per_week_group1 : ℕ := 1000
def men_group2 : ℕ := 9
def hours_per_day_group2 : ℕ := 6
def earnings_per_week_group2 : ℕ := 1350
def days_per_week : ℕ := 7

-- Theorem to prove
theorem number_of_men_in_first_group :
  (M * hours_per_day_group1 * days_per_week) / earnings_per_week_group1 =
  (men_group2 * hours_per_day_group2 * days_per_week) / earnings_per_week_group2 →
  M = 4 := by
  sorry

end number_of_men_in_first_group_l2409_240949


namespace stock_price_decrease_l2409_240968

theorem stock_price_decrease (x : ℝ) (h : x > 0) :
  let increase_factor := 1.3
  let decrease_factor := 1 - 1 / increase_factor
  x = (1 - decrease_factor) * (increase_factor * x) :=
by sorry

end stock_price_decrease_l2409_240968


namespace stephanies_speed_l2409_240939

/-- Given a distance of 15 miles and a time of 3 hours, prove that the speed is 5 miles per hour. -/
theorem stephanies_speed (distance : ℝ) (time : ℝ) (h1 : distance = 15) (h2 : time = 3) :
  distance / time = 5 := by
  sorry

end stephanies_speed_l2409_240939


namespace sector_angle_l2409_240911

/-- Given a circular sector with area 1 and radius 1, prove that its central angle in radians is 2 -/
theorem sector_angle (area : ℝ) (radius : ℝ) (angle : ℝ) 
  (h_area : area = 1) 
  (h_radius : radius = 1) 
  (h_sector : area = 1/2 * radius^2 * angle) : angle = 2 := by
  sorry

end sector_angle_l2409_240911


namespace rectangle_to_cylinder_surface_area_l2409_240952

/-- The surface area of a cylinder formed by rolling a rectangle -/
def cylinderSurfaceArea (length width : Real) : Set Real :=
  let baseArea1 := Real.pi * (length / (2 * Real.pi))^2
  let baseArea2 := Real.pi * (width / (2 * Real.pi))^2
  let lateralArea := length * width
  {lateralArea + 2 * baseArea1, lateralArea + 2 * baseArea2}

theorem rectangle_to_cylinder_surface_area :
  cylinderSurfaceArea (4 * Real.pi) (8 * Real.pi) = {32 * Real.pi^2 + 8 * Real.pi, 32 * Real.pi^2 + 32 * Real.pi} := by
  sorry

#check rectangle_to_cylinder_surface_area

end rectangle_to_cylinder_surface_area_l2409_240952


namespace smallest_a1_l2409_240951

theorem smallest_a1 (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_rec : ∀ n > 1, a n = 7 * a (n - 1) - 2 * n) :
  (∀ a₁ : ℝ, (∀ n, a n > 0) → (∀ n > 1, a n = 7 * a (n - 1) - 2 * n) → a₁ ≥ a 1) →
  a 1 = 13 / 18 := by
sorry

end smallest_a1_l2409_240951


namespace bus_wheel_radius_proof_l2409_240934

/-- The speed of the bus in km/h -/
def bus_speed : ℝ := 66

/-- The revolutions per minute of the wheel -/
def wheel_rpm : ℝ := 125.11373976342128

/-- The radius of the wheel in centimeters -/
def wheel_radius : ℝ := 140.007

/-- Theorem stating that given the bus speed and wheel rpm, the wheel radius is approximately 140.007 cm -/
theorem bus_wheel_radius_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |wheel_radius - (bus_speed * 100000 / (60 * wheel_rpm * 2 * Real.pi))| < ε :=
sorry

end bus_wheel_radius_proof_l2409_240934


namespace cone_height_calculation_l2409_240912

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents a cone with a given base radius and height -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Theorem: Given three spheres and a cone touching externally on a flat surface,
    the height of the cone is 28 -/
theorem cone_height_calculation (s₁ s₂ s₃ : Sphere) (c : Cone) :
  s₁.radius = 20 →
  s₂.radius = 40 →
  s₃.radius = 40 →
  c.baseRadius = 21 →
  (∃ (arrangement : ℝ → ℝ → ℝ), 
    arrangement s₁.radius s₂.radius = arrangement s₁.radius s₃.radius ∧
    arrangement s₂.radius s₃.radius = s₂.radius + s₃.radius ∧
    arrangement s₁.radius s₂.radius = Real.sqrt ((s₁.radius + s₂.radius)^2 - (s₂.radius - s₁.radius)^2)) →
  c.height = 28 := by
  sorry


end cone_height_calculation_l2409_240912


namespace intersection_line_canonical_equations_l2409_240973

/-- Given two planes in 3D space, this theorem proves that the canonical equations
    of the line formed by their intersection have a specific form. -/
theorem intersection_line_canonical_equations
  (plane1 : ℝ → ℝ → ℝ → Prop)
  (plane2 : ℝ → ℝ → ℝ → Prop)
  (h1 : ∀ x y z, plane1 x y z ↔ 4*x + y - 3*z + 2 = 0)
  (h2 : ∀ x y z, plane2 x y z ↔ 2*x - y + z - 8 = 0) :
  ∃ (line : ℝ → ℝ → ℝ → Prop),
    (∀ x y z, line x y z ↔ (x - 1) / (-2) = (y + 6) / (-10) ∧ (y + 6) / (-10) = z / (-6)) ∧
    (∀ x y z, line x y z ↔ plane1 x y z ∧ plane2 x y z) :=
by sorry

end intersection_line_canonical_equations_l2409_240973


namespace invisible_square_exists_l2409_240922

/-- A point with integer coordinates is invisible if the gcd of its coordinates is greater than 1 -/
def invisible (p q : ℤ) : Prop := Nat.gcd p.natAbs q.natAbs > 1

/-- There exists a square with side length n*k where all integer coordinate points are invisible -/
theorem invisible_square_exists (n : ℕ) : ∃ k : ℕ, k ≥ 2 ∧ 
  ∀ p q : ℤ, 0 ≤ p ∧ p ≤ n * k ∧ 0 ≤ q ∧ q ≤ n * k → invisible p q :=
sorry

end invisible_square_exists_l2409_240922


namespace small_bottles_initial_count_small_bottles_initial_count_proof_l2409_240943

theorem small_bottles_initial_count : ℝ → Prop :=
  fun S =>
    let big_bottles : ℝ := 12000
    let small_bottles_remaining_ratio : ℝ := 0.85
    let big_bottles_remaining_ratio : ℝ := 0.82
    let total_remaining : ℝ := 14090
    S * small_bottles_remaining_ratio + big_bottles * big_bottles_remaining_ratio = total_remaining →
    S = 5000

-- The proof goes here
theorem small_bottles_initial_count_proof : small_bottles_initial_count 5000 := by
  sorry

end small_bottles_initial_count_small_bottles_initial_count_proof_l2409_240943


namespace quadratic_inequality_range_l2409_240913

theorem quadratic_inequality_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x + 1 ≤ 0) → (m ≥ 2 ∨ m ≤ -2) := by
  sorry

end quadratic_inequality_range_l2409_240913


namespace quadratic_real_solutions_range_l2409_240906

theorem quadratic_real_solutions_range (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 :=
by sorry

end quadratic_real_solutions_range_l2409_240906


namespace binomial_18_10_l2409_240925

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 8008) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 43758 := by
  sorry

end binomial_18_10_l2409_240925


namespace plane_division_l2409_240937

/-- Represents a line on a plane -/
structure Line

/-- Represents a point on a plane -/
structure Point

/-- λ(P) represents the number of lines passing through a point P -/
def lambda (P : Point) (lines : Finset Line) : ℕ := sorry

/-- The set of all intersection points of the given lines -/
def intersectionPoints (lines : Finset Line) : Finset Point := sorry

/-- Theorem: For n lines on a plane, the total number of regions formed is 1+n+∑(λ(P)-1),
    and the number of unbounded regions is 2n -/
theorem plane_division (n : ℕ) (lines : Finset Line) 
  (h : lines.card = n) :
  (∃ (regions unboundedRegions : ℕ),
    regions = 1 + n + (intersectionPoints lines).sum (λ P => lambda P lines - 1) ∧
    unboundedRegions = 2 * n) :=
  sorry

end plane_division_l2409_240937


namespace product_zero_implications_l2409_240963

theorem product_zero_implications (a b c : ℝ) : 
  (((a * b * c = 0) → (a = 0 ∨ b = 0 ∨ c = 0)) ∧
   ((a = 0 ∨ b = 0 ∨ c = 0) → (a * b * c = 0)) ∧
   ((a * b * c ≠ 0) → (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)) ∧
   ((a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → (a * b * c ≠ 0))) :=
by sorry

end product_zero_implications_l2409_240963


namespace missing_number_proof_l2409_240916

theorem missing_number_proof (x : ℝ) : (4 + x) + (8 - 3 - 1) = 11 → x = 3 := by
  sorry

end missing_number_proof_l2409_240916


namespace not_prime_a_l2409_240910

theorem not_prime_a (a b : ℕ+) (h : ∃ k : ℤ, (5 * a^4 + a^2 : ℤ) = k * (b^4 + 3 * b^2 + 4)) : 
  ¬ Nat.Prime a.val := by
  sorry

end not_prime_a_l2409_240910


namespace students_between_positions_l2409_240938

theorem students_between_positions (n : ℕ) (h : n = 9) : 
  (n - 2) - (3 + 1) = 4 :=
by
  sorry

end students_between_positions_l2409_240938


namespace sum_pairwise_reciprocal_sums_geq_three_halves_l2409_240905

theorem sum_pairwise_reciprocal_sums_geq_three_halves 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 := by
  sorry

end sum_pairwise_reciprocal_sums_geq_three_halves_l2409_240905


namespace max_piles_l2409_240958

/-- Represents a configuration of stone piles -/
structure StonePiles :=
  (piles : List Nat)
  (total_stones : Nat)
  (h_total : piles.sum = total_stones)
  (h_factor : ∀ (p q : Nat), p ∈ piles → q ∈ piles → p < 2 * q)

/-- Defines a valid split operation on stone piles -/
def split (sp : StonePiles) (i : Nat) (n : Nat) : Option StonePiles :=
  sorry

/-- Theorem: The maximum number of piles that can be formed is 30 -/
theorem max_piles (sp : StonePiles) (h_initial : sp.total_stones = 660) :
  (∀ sp' : StonePiles, ∃ (i j : Nat), split sp i j = some sp') →
  sp.piles.length ≤ 30 :=
sorry

end max_piles_l2409_240958


namespace hundred_chickens_problem_l2409_240992

theorem hundred_chickens_problem :
  ∀ x y z : ℕ,
  x + y + z = 100 →
  5 * x + 3 * y + (z / 3 : ℚ) = 100 →
  z = 81 →
  x = 8 ∧ y = 11 := by
sorry

end hundred_chickens_problem_l2409_240992


namespace line_equation_l2409_240991

-- Define the circle C
def Circle (x y : ℝ) : Prop := x^2 + (y-1)^2 = 5

-- Define the line l
def Line (m x y : ℝ) : Prop := m*x - y + 1 - m = 0

-- Define the condition that P(1,1) satisfies 2⃗AP = ⃗PB
def PointCondition (xa ya xb yb : ℝ) : Prop :=
  2*(1 - xa, 1 - ya) = (xb - 1, yb - 1)

theorem line_equation :
  ∀ (m : ℝ) (xa ya xb yb : ℝ),
    Circle xa ya → Circle xb yb →  -- A and B are on the circle
    Line m xa ya → Line m xb yb →  -- A and B are on the line
    PointCondition xa ya xb yb →   -- P(1,1) satisfies 2⃗AP = ⃗PB
    (m = 1 ∨ m = -1) :=             -- The slope of the line is either 1 or -1
by sorry

end line_equation_l2409_240991


namespace geometric_arithmetic_sequence_common_difference_l2409_240902

/-- Given a geometric sequence {a_n} where a_1+1, a_3+4, a_5+7 form an arithmetic sequence,
    the common difference of this arithmetic sequence is 3. -/
theorem geometric_arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)
  (h_arithmetic : ∃ d : ℝ, (a 3 + 4) - (a 1 + 1) = d ∧ (a 5 + 7) - (a 3 + 4) = d) :
  ∃ d : ℝ, d = 3 ∧ (a 3 + 4) - (a 1 + 1) = d ∧ (a 5 + 7) - (a 3 + 4) = d :=
by sorry

end geometric_arithmetic_sequence_common_difference_l2409_240902


namespace min_m_plus_n_l2409_240957

/-- The sum of interior angles of a regular n-gon -/
def interior_angle_sum (n : ℕ) : ℕ := 180 * (n - 2)

/-- The sum of interior angles of m regular n-gons -/
def total_interior_angle_sum (m n : ℕ) : ℕ := m * interior_angle_sum n

/-- Predicate to check if the sum of interior angles is divisible by 27 -/
def is_divisible_by_27 (m n : ℕ) : Prop :=
  (total_interior_angle_sum m n) % 27 = 0

/-- The main theorem stating the minimum value of m + n -/
theorem min_m_plus_n :
  ∃ (m₀ n₀ : ℕ), is_divisible_by_27 m₀ n₀ ∧
    ∀ (m n : ℕ), is_divisible_by_27 m n → m₀ + n₀ ≤ m + n :=
sorry

end min_m_plus_n_l2409_240957


namespace dot_product_AP_BP_l2409_240926

/-- The dot product of vectors AP and BP, where P is a point on a specific ellipse satisfying certain conditions. -/
theorem dot_product_AP_BP : ∃ (x y : ℝ), 
  (x^2 / 12 + y^2 / 16 = 1) ∧ 
  (((x - 0)^2 + (y - (-2))^2).sqrt - ((x - 0)^2 + (y - 2)^2).sqrt = 2) →
  (x * x + (y + 2) * (y - 2) = 9) := by
sorry

end dot_product_AP_BP_l2409_240926


namespace tims_bill_denomination_l2409_240920

theorem tims_bill_denomination :
  let unknown_bills : ℕ := 13
  let five_dollar_bills : ℕ := 11
  let one_dollar_bills : ℕ := 17
  let total_amount : ℕ := 128
  let min_bills_used : ℕ := 16
  
  ∃ (x : ℕ),
    x * unknown_bills + 5 * five_dollar_bills + one_dollar_bills = total_amount ∧
    unknown_bills + five_dollar_bills + one_dollar_bills ≥ min_bills_used ∧
    x = 4 :=
by sorry

end tims_bill_denomination_l2409_240920


namespace fraction_equality_implies_value_l2409_240986

theorem fraction_equality_implies_value (a : ℝ) (x : ℝ) :
  (a - 2) / x = 1 / (2 * a + 7) → x = 2 * a^2 + 3 * a - 14 := by
  sorry

end fraction_equality_implies_value_l2409_240986


namespace polygon_interior_angles_sum_l2409_240950

theorem polygon_interior_angles_sum (n : ℕ) : n ≥ 3 →
  (2 * n - 2) * 180 = 2160 → n = 7 := by
  sorry

end polygon_interior_angles_sum_l2409_240950


namespace probability_at_least_five_consecutive_heads_l2409_240931

def num_flips : ℕ := 8
def min_consecutive_heads : ℕ := 5

def favorable_outcomes : ℕ := 10
def total_outcomes : ℕ := 2^num_flips

theorem probability_at_least_five_consecutive_heads :
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 128 := by sorry

end probability_at_least_five_consecutive_heads_l2409_240931


namespace imaginary_part_of_pure_imaginary_z_l2409_240927

def i : ℂ := Complex.I

theorem imaginary_part_of_pure_imaginary_z (a : ℝ) :
  let z : ℂ := a + 15 / (3 - 4 * i)
  (z.re = 0) → z.im = 12/5 := by
  sorry

end imaginary_part_of_pure_imaginary_z_l2409_240927


namespace range_of_even_power_function_l2409_240969

theorem range_of_even_power_function (k : ℕ) (hk : Even k) (hk_pos : k > 0) :
  Set.range (fun x : ℝ => x ^ k) = Set.Ici (0 : ℝ) := by
  sorry

end range_of_even_power_function_l2409_240969


namespace sarah_money_l2409_240917

/-- Given that Bridge and Sarah have 300 cents in total, and Bridge has 50 cents more than Sarah,
    prove that Sarah has 125 cents. -/
theorem sarah_money : 
  ∀ (sarah_cents bridge_cents : ℕ), 
    sarah_cents + bridge_cents = 300 →
    bridge_cents = sarah_cents + 50 →
    sarah_cents = 125 := by
  sorry

end sarah_money_l2409_240917


namespace fraction_equality_implies_numerator_equality_l2409_240929

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end fraction_equality_implies_numerator_equality_l2409_240929


namespace total_cars_in_group_l2409_240990

/-- Given a group of cars with specific properties, we prove that the total number of cars is 137. -/
theorem total_cars_in_group (total : ℕ) 
  (no_ac : ℕ) 
  (with_stripes : ℕ) 
  (ac_no_stripes : ℕ) 
  (h1 : no_ac = 37)
  (h2 : with_stripes ≥ 51)
  (h3 : ac_no_stripes = 49)
  (h4 : total = no_ac + with_stripes + ac_no_stripes) :
  total = 137 := by
  sorry

end total_cars_in_group_l2409_240990


namespace average_speed_two_hours_l2409_240940

/-- The average speed of a car given its distances traveled in two consecutive hours -/
theorem average_speed_two_hours (d1 d2 : ℝ) : 
  d1 = 90 → d2 = 40 → (d1 + d2) / 2 = 65 := by
  sorry

end average_speed_two_hours_l2409_240940


namespace not_sufficient_not_necessary_l2409_240987

theorem not_sufficient_not_necessary (a b : ℝ) :
  ¬(∀ a b : ℝ, 0 < a * b ∧ a * b < 1 → b < 1 / a) ∧
  ¬(∀ a b : ℝ, b < 1 / a → 0 < a * b ∧ a * b < 1) := by
  sorry

end not_sufficient_not_necessary_l2409_240987


namespace fixed_point_on_circle_l2409_240918

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line L passing through M(0, -1/3)
def line (x y k : ℝ) : Prop := y = k*x - 1/3

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y

-- Define the intersection points A and B
def intersection_points (x1 y1 x2 y2 k : ℝ) : Prop :=
  point_on_ellipse x1 y1 ∧ point_on_ellipse x2 y2 ∧
  line x1 y1 k ∧ line x2 y2 k

-- Define the circle with diameter AB
def circle_AB (x y x1 y1 x2 y2 : ℝ) : Prop :=
  (x - (x1 + x2)/2)^2 + (y - (y1 + y2)/2)^2 = ((x1 - x2)^2 + (y1 - y2)^2)/4

-- Theorem statement
theorem fixed_point_on_circle (k : ℝ) :
  ∀ x1 y1 x2 y2,
  intersection_points x1 y1 x2 y2 k →
  circle_AB 0 1 x1 y1 x2 y2 :=
sorry

end fixed_point_on_circle_l2409_240918


namespace largest_integral_solution_l2409_240907

theorem largest_integral_solution : ∃ x : ℤ, (1 : ℚ) / 4 < x / 6 ∧ x / 6 < 7 / 9 ∧ ∀ y : ℤ, (1 : ℚ) / 4 < y / 6 ∧ y / 6 < 7 / 9 → y ≤ x := by
  sorry

end largest_integral_solution_l2409_240907


namespace grid_division_theorem_l2409_240971

/-- Represents a grid division into squares and corners -/
structure GridDivision where
  squares : ℕ  -- number of 2x2 squares
  corners : ℕ  -- number of 3-cell corners

/-- Checks if a grid division is valid for a 7x14 grid -/
def is_valid_division (d : GridDivision) : Prop :=
  4 * d.squares + 3 * d.corners = 7 * 14

theorem grid_division_theorem :
  -- Part a: There exists a valid division where squares = corners
  (∃ d : GridDivision, is_valid_division d ∧ d.squares = d.corners) ∧
  -- Part b: There does not exist a valid division where squares > corners
  (¬ ∃ d : GridDivision, is_valid_division d ∧ d.squares > d.corners) := by
  sorry

end grid_division_theorem_l2409_240971


namespace tower_configurations_mod_1000_l2409_240944

/-- Recursively calculates the number of valid tower configurations for m cubes -/
def tower_configurations (m : ℕ) : ℕ :=
  match m with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | n + 1 => (n + 2) * tower_configurations n

/-- Represents the conditions for building towers with 9 cubes -/
def valid_tower_conditions (n : ℕ) : Prop :=
  n ≤ 9 ∧ 
  ∀ k : ℕ, k ≤ n → ∃ cube : ℕ, cube = k ∧
  ∀ i j : ℕ, i < j → j - i ≤ 3

/-- The main theorem stating that the number of different towers is congruent to 200 modulo 1000 -/
theorem tower_configurations_mod_1000 :
  valid_tower_conditions 9 →
  tower_configurations 9 % 1000 = 200 := by
  sorry


end tower_configurations_mod_1000_l2409_240944


namespace rectangle_rotation_volume_l2409_240941

/-- The volume of a solid formed by rotating a rectangle around one of its sides -/
theorem rectangle_rotation_volume (length width : ℝ) (h_length : length = 6) (h_width : width = 4) :
  ∃ (volume : ℝ), (volume = 96 * Real.pi ∨ volume = 144 * Real.pi) ∧
  (∃ (axis : ℝ), (axis = length ∨ axis = width) ∧
    volume = Real.pi * (axis / 2) ^ 2 * (if axis = length then width else length)) := by
  sorry

end rectangle_rotation_volume_l2409_240941


namespace mowing_area_calculation_l2409_240979

/-- Given that 3 mowers can mow 3 hectares in 3 days, 
    this theorem proves that 5 mowers can mow 25/3 hectares in 5 days. -/
theorem mowing_area_calculation 
  (mowers_initial : ℕ) 
  (days_initial : ℕ) 
  (area_initial : ℚ) 
  (mowers_final : ℕ) 
  (days_final : ℕ) 
  (h1 : mowers_initial = 3) 
  (h2 : days_initial = 3) 
  (h3 : area_initial = 3) 
  (h4 : mowers_final = 5) 
  (h5 : days_final = 5) :
  (area_initial * mowers_final * days_final) / (mowers_initial * days_initial) = 25 / 3 := by
  sorry

#check mowing_area_calculation

end mowing_area_calculation_l2409_240979


namespace mario_garden_flowers_l2409_240975

/-- The number of flowers on Mario's first hibiscus plant -/
def first_plant : ℕ := 2

/-- The number of flowers on Mario's second hibiscus plant -/
def second_plant : ℕ := 2 * first_plant

/-- The number of flowers on Mario's third hibiscus plant -/
def third_plant : ℕ := 4 * second_plant

/-- The total number of flowers in Mario's garden -/
def total_flowers : ℕ := first_plant + second_plant + third_plant

theorem mario_garden_flowers : total_flowers = 22 := by
  sorry

end mario_garden_flowers_l2409_240975
