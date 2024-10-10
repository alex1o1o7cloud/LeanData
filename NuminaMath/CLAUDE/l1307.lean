import Mathlib

namespace max_value_of_expression_l1307_130700

theorem max_value_of_expression (y : ℝ) :
  (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) ≤ 15 ∧
  ∃ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) = 15 := by
  sorry

end max_value_of_expression_l1307_130700


namespace pencil_price_after_discount_l1307_130702

/-- Given a pencil with an original cost and a discount, calculate the final price. -/
def final_price (original_cost discount : ℚ) : ℚ :=
  original_cost - discount

/-- Theorem stating that for a pencil with an original cost of 4 dollars and a discount of 0.63 dollars, the final price is 3.37 dollars. -/
theorem pencil_price_after_discount :
  final_price 4 0.63 = 3.37 := by
  sorry

end pencil_price_after_discount_l1307_130702


namespace min_value_expression_l1307_130703

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2)) / (a * b * c) ≥ 343 ∧
  ((1^2 + 5*1 + 2) * (1^2 + 5*1 + 2) * (1^2 + 5*1 + 2)) / (1 * 1 * 1) = 343 :=
by sorry

end min_value_expression_l1307_130703


namespace spinner_probability_l1307_130763

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_C = 1/6 → p_A + p_B + p_C + p_D = 1 → p_D = 1/4 := by
  sorry

end spinner_probability_l1307_130763


namespace equation_solutions_l1307_130712

theorem equation_solutions : 
  {x : ℝ | Real.sqrt (6 * x - 5) + 12 / Real.sqrt (6 * x - 5) = 8} = {41/6, 3/2} := by
sorry

end equation_solutions_l1307_130712


namespace fourth_square_area_l1307_130740

-- Define the triangles and their properties
def triangle_PQR (PQ PR QR : ℝ) : Prop :=
  PQ^2 + PR^2 = QR^2 ∧ PQ = 5 ∧ PR = 7

def triangle_PRS (PR PS RS : ℝ) : Prop :=
  PR^2 + PS^2 = RS^2 ∧ PS = 8 ∧ PR = 7

-- Theorem statement
theorem fourth_square_area 
  (PQ PR QR PS RS : ℝ) 
  (h1 : triangle_PQR PQ PR QR) 
  (h2 : triangle_PRS PR PS RS) : 
  RS^2 = 113 := by
sorry

end fourth_square_area_l1307_130740


namespace david_profit_l1307_130781

/-- Represents the discount percentage based on the number of sacks bought -/
def discount_percentage (num_sacks : ℕ) : ℚ :=
  if num_sacks ≤ 10 then 2/100
  else if num_sacks ≤ 20 then 4/100
  else 5/100

/-- Calculates the total cost of buying sacks with discount -/
def total_cost (num_sacks : ℕ) (price_per_sack : ℚ) : ℚ :=
  num_sacks * price_per_sack * (1 - discount_percentage num_sacks)

/-- Calculates the total selling price for a given number of days and price per kg -/
def selling_price (kg_per_day : ℚ) (price_per_kg : ℚ) (num_days : ℕ) : ℚ :=
  kg_per_day * price_per_kg * num_days

/-- Calculates the total selling price for the week -/
def total_selling_price (kg_per_day : ℚ) : ℚ :=
  selling_price kg_per_day 1.20 3 +
  selling_price kg_per_day 1.30 2 +
  selling_price kg_per_day 1.25 2

/-- Calculates the profit after tax -/
def profit_after_tax (total_selling : ℚ) (total_cost : ℚ) (tax_rate : ℚ) : ℚ :=
  total_selling * (1 - tax_rate) - total_cost

/-- Theorem stating David's profit for the week -/
theorem david_profit :
  let num_sacks : ℕ := 25
  let price_per_sack : ℚ := 50
  let sack_weight : ℚ := 50
  let total_kg : ℚ := num_sacks * sack_weight
  let kg_per_day : ℚ := total_kg / 7
  let tax_rate : ℚ := 12/100
  profit_after_tax
    (total_selling_price kg_per_day)
    (total_cost num_sacks price_per_sack)
    tax_rate = 179.62 := by
  sorry


end david_profit_l1307_130781


namespace all_setC_are_polyhedra_setC_consists_entirely_of_polyhedra_l1307_130776

-- Define the type for geometric bodies
inductive GeometricBody
  | TriangularPrism
  | QuadrangularPyramid
  | Cube
  | HexagonalPyramid
  | Sphere
  | Cone
  | Frustum
  | Hemisphere

-- Define a predicate for polyhedra
def isPolyhedron : GeometricBody → Prop
  | GeometricBody.TriangularPrism => True
  | GeometricBody.QuadrangularPyramid => True
  | GeometricBody.Cube => True
  | GeometricBody.HexagonalPyramid => True
  | _ => False

-- Define the set of geometric bodies in option C
def setC : List GeometricBody :=
  [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid,
   GeometricBody.Cube, GeometricBody.HexagonalPyramid]

-- Theorem: All elements in setC are polyhedra
theorem all_setC_are_polyhedra : ∀ x ∈ setC, isPolyhedron x := by
  sorry

-- Main theorem: setC consists entirely of polyhedra
theorem setC_consists_entirely_of_polyhedra : 
  (∀ x ∈ setC, isPolyhedron x) ∧ (setC ≠ []) := by
  sorry

end all_setC_are_polyhedra_setC_consists_entirely_of_polyhedra_l1307_130776


namespace polynomial_division_result_l1307_130748

variables {a p x : ℝ}

theorem polynomial_division_result :
  (p^8 * x^4 - 81 * a^12) / (p^6 * x^3 - 3 * a^3 * p^4 * x^2 + 9 * a^6 * p^2 * x - 27 * a^9) = p^2 * x + 3 * a^3 :=
by sorry

end polynomial_division_result_l1307_130748


namespace max_triangle_area_l1307_130750

theorem max_triangle_area (a b c : ℝ) (ha : 0 < a ∧ a ≤ 1) (hb : 1 ≤ b ∧ b ≤ 2) (hc : 2 ≤ c ∧ c ≤ 3)
  (htri : a + b > c ∧ a + c > b ∧ b + c > a) :
  ∃ (area : ℝ), area ≤ 1 ∧ ∀ (other_area : ℝ), 
    (∃ (x y z : ℝ), 0 < x ∧ x ≤ 1 ∧ 1 ≤ y ∧ y ≤ 2 ∧ 2 ≤ z ∧ z ≤ 3 ∧ 
      x + y > z ∧ x + z > y ∧ y + z > x ∧
      other_area = (x + y + z) * (- x + y + z) * (x - y + z) * (x + y - z) / (4 * (x + y + z))) →
    other_area ≤ area :=
by sorry

end max_triangle_area_l1307_130750


namespace max_salary_baseball_team_l1307_130782

/-- Represents the maximum salary for a single player in a baseball team under given constraints -/
def max_player_salary (num_players : ℕ) (min_salary : ℕ) (total_budget : ℕ) : ℕ :=
  total_budget - (num_players - 1) * min_salary

/-- Theorem stating the maximum possible salary for a single player under given constraints -/
theorem max_salary_baseball_team :
  max_player_salary 18 20000 600000 = 260000 :=
by sorry

end max_salary_baseball_team_l1307_130782


namespace cord_length_proof_l1307_130736

theorem cord_length_proof (n : ℕ) (longest shortest : ℝ) : 
  n = 19 → 
  longest = 8 → 
  shortest = 2 → 
  (n : ℝ) * (longest / 2 + shortest) = 114 :=
by
  sorry

end cord_length_proof_l1307_130736


namespace ant_probability_l1307_130701

/-- Represents a vertex of a cube -/
inductive Vertex : Type
| A | B | C | D | E | F | G | H

/-- Represents the movement of an ant from one vertex to another -/
def Move : Type := Vertex → Vertex

/-- The set of all possible moves for all 8 ants -/
def AllMoves : Type := Fin 8 → Move

/-- Checks if a move is valid (i.e., to an adjacent vertex) -/
def isValidMove (m : Move) : Prop := sorry

/-- Checks if a set of moves results in no two ants on the same vertex -/
def noCollisions (moves : AllMoves) : Prop := sorry

/-- The total number of possible movement combinations -/
def totalMoves : ℕ := 3^8

/-- The number of valid movement combinations where no two ants collide -/
def validMoves : ℕ := 240

/-- The probability of no two ants arriving at the same vertex -/
theorem ant_probability : 
  (validMoves : ℚ) / totalMoves = 240 / 6561 := by sorry

end ant_probability_l1307_130701


namespace fraction_increase_l1307_130731

theorem fraction_increase (x y : ℝ) (h : 2*x ≠ 3*y) : 
  (5*x * 5*y) / (2*(5*x) - 3*(5*y)) = 5 * (x*y / (2*x - 3*y)) := by
  sorry

end fraction_increase_l1307_130731


namespace average_price_per_book_l1307_130793

theorem average_price_per_book (books1 : ℕ) (price1 : ℕ) (books2 : ℕ) (price2 : ℕ) 
  (h1 : books1 = 55) (h2 : price1 = 1500) (h3 : books2 = 60) (h4 : price2 = 340) :
  (price1 + price2) / (books1 + books2) = 16 := by
  sorry

end average_price_per_book_l1307_130793


namespace f_composition_and_domain_l1307_130734

def f (x : ℝ) : ℝ := x + 5

theorem f_composition_and_domain :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f x ∈ Set.Icc 2 7) →
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, ∀ y ∈ Set.Icc (-3 : ℝ) 2, x < y → f x < f y) →
  (∀ x ∈ Set.Icc (-8 : ℝ) (-3), f (f x) = x + 10) ∧
  (∀ x, f (f x) ∈ Set.Icc 2 7 ↔ x ∈ Set.Icc (-8 : ℝ) (-3)) := by
  sorry

#check f_composition_and_domain

end f_composition_and_domain_l1307_130734


namespace awards_distribution_l1307_130752

/-- The number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- The condition that each student receives at least one award -/
def at_least_one_award (distribution : List ℕ) : Prop :=
  sorry

theorem awards_distribution :
  ∃ (d : List ℕ),
    d.length = 4 ∧
    d.sum = 6 ∧
    at_least_one_award d ∧
    distribute_awards 6 4 = 1560 :=
  sorry

end awards_distribution_l1307_130752


namespace exists_line_with_F_as_incenter_l1307_130724

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The right focus F -/
def F : ℝ × ℝ := (1, 0)

/-- The upper vertex M -/
def M : ℝ × ℝ := (0, 1)

/-- The line l -/
def line_l (x y : ℝ) : Prop := x + (2 - Real.sqrt 6) * y + 6 - 3 * Real.sqrt 6 = 0

/-- P and Q are points on both the ellipse and line l -/
def P_Q_on_ellipse_and_line (P Q : ℝ × ℝ) : Prop :=
  ellipse P.1 P.2 ∧ ellipse Q.1 Q.2 ∧ line_l P.1 P.2 ∧ line_l Q.1 Q.2

/-- F is the incenter of triangle MPQ -/
def F_is_incenter (P Q : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
    (F.1 - P.1)^2 + (F.2 - P.2)^2 = r^2 ∧
    (F.1 - Q.1)^2 + (F.2 - Q.2)^2 = r^2 ∧
    (F.1 - M.1)^2 + (F.2 - M.2)^2 = r^2

/-- The main theorem -/
theorem exists_line_with_F_as_incenter :
  ∃ (P Q : ℝ × ℝ), P_Q_on_ellipse_and_line P Q ∧ F_is_incenter P Q := by sorry

end exists_line_with_F_as_incenter_l1307_130724


namespace function_equation_solution_l1307_130711

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2) - f (y^2) + 2*x + 1 = f (x + y) * f (x - y)) : 
  (∀ x : ℝ, f x = x + 1) ∨ (∀ x : ℝ, f x = -x - 1) := by
  sorry

end function_equation_solution_l1307_130711


namespace clothes_batch_size_l1307_130784

/-- Proves that the number of sets of clothes in a batch is 30, given the production rates of two workers and their time difference. -/
theorem clothes_batch_size :
  let wang_rate : ℚ := 3  -- Wang's production rate (sets per day)
  let li_rate : ℚ := 5    -- Li's production rate (sets per day)
  let time_diff : ℚ := 4  -- Time difference in days
  let batch_size : ℚ := (wang_rate * li_rate * time_diff) / (li_rate - wang_rate)
  batch_size = 30 := by
  sorry


end clothes_batch_size_l1307_130784


namespace smallest_AAB_l1307_130749

theorem smallest_AAB : ∃ (A B : ℕ),
  A ≠ B ∧
  A ∈ Finset.range 10 ∧
  B ∈ Finset.range 10 ∧
  A ≠ 0 ∧
  (10 * A + B) = (110 * A + B) / 8 ∧
  ∀ (A' B' : ℕ),
    A' ≠ B' →
    A' ∈ Finset.range 10 →
    B' ∈ Finset.range 10 →
    A' ≠ 0 →
    (10 * A' + B') = (110 * A' + B') / 8 →
    110 * A + B ≤ 110 * A' + B' ∧
    110 * A + B = 773 :=
by sorry

end smallest_AAB_l1307_130749


namespace select_with_abc_must_select_with_one_abc_select_with_at_most_two_abc_l1307_130765

-- Define the total number of people
def total_people : ℕ := 12

-- Define the number of people to be selected
def select_count : ℕ := 5

-- Define the number of special people (A, B, C)
def special_people : ℕ := 3

-- Theorem 1: When A, B, and C must be chosen
theorem select_with_abc_must (n : ℕ) (k : ℕ) (s : ℕ) : 
  n = total_people ∧ k = select_count ∧ s = special_people →
  Nat.choose (n - s) (k - s) = 36 :=
sorry

-- Theorem 2: When only one among A, B, and C is chosen
theorem select_with_one_abc (n : ℕ) (k : ℕ) (s : ℕ) :
  n = total_people ∧ k = select_count ∧ s = special_people →
  Nat.choose s 1 * Nat.choose (n - s) (k - 1) = 378 :=
sorry

-- Theorem 3: When at most two among A, B, and C are chosen
theorem select_with_at_most_two_abc (n : ℕ) (k : ℕ) (s : ℕ) :
  n = total_people ∧ k = select_count ∧ s = special_people →
  Nat.choose n k - Nat.choose (n - s) (k - s) = 756 :=
sorry

end select_with_abc_must_select_with_one_abc_select_with_at_most_two_abc_l1307_130765


namespace opposite_of_four_l1307_130742

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- The opposite of 4 is -4. -/
theorem opposite_of_four : opposite 4 = -4 := by sorry

end opposite_of_four_l1307_130742


namespace gcf_of_48_180_98_l1307_130747

theorem gcf_of_48_180_98 : Nat.gcd 48 (Nat.gcd 180 98) = 2 := by
  sorry

end gcf_of_48_180_98_l1307_130747


namespace intersection_of_P_and_Q_l1307_130795

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | |x - 1| < 1}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = Set.Ioo 0 2 := by sorry

end intersection_of_P_and_Q_l1307_130795


namespace halfway_between_one_third_and_one_fifth_l1307_130774

theorem halfway_between_one_third_and_one_fifth : 
  (1 / 3 + 1 / 5) / 2 = 4 / 15 := by
  sorry

end halfway_between_one_third_and_one_fifth_l1307_130774


namespace binomial_expansion_arithmetic_sequence_l1307_130739

theorem binomial_expansion_arithmetic_sequence (n : ℕ) : 
  (∃ d : ℚ, 1 + d = n / 2 ∧ n / 2 + d = n * (n - 1) / 8) → n = 8 := by
  sorry

end binomial_expansion_arithmetic_sequence_l1307_130739


namespace no_prime_solution_l1307_130704

/-- Convert a number from base p to decimal --/
def baseP_to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.foldr (fun d acc => d + p * acc) 0

/-- The equation that needs to be satisfied --/
def equation (p : Nat) : Prop :=
  baseP_to_decimal [1, 0, 1, 3] p +
  baseP_to_decimal [2, 0, 7] p +
  baseP_to_decimal [2, 1, 4] p +
  baseP_to_decimal [1, 0, 0] p +
  baseP_to_decimal [1, 0] p =
  baseP_to_decimal [3, 2, 1] p +
  baseP_to_decimal [4, 0, 3] p +
  baseP_to_decimal [2, 1, 0] p

theorem no_prime_solution :
  ∀ p, Nat.Prime p → ¬(equation p) := by
  sorry

end no_prime_solution_l1307_130704


namespace contrapositive_real_roots_l1307_130751

theorem contrapositive_real_roots :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔
  (∀ m : ℝ, (¬∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
by sorry

end contrapositive_real_roots_l1307_130751


namespace geometric_sequence_first_term_l1307_130743

/-- A geometric sequence is a sequence where the ratio between successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Nat.factorial n)

theorem geometric_sequence_first_term :
  ∀ a : ℕ → ℝ,
  IsGeometricSequence a →
  a 4 = factorial 6 →
  a 6 = factorial 7 →
  a 1 = (720 : ℝ) * Real.sqrt 7 / 49 := by
  sorry

end geometric_sequence_first_term_l1307_130743


namespace complement_of_A_in_U_l1307_130759

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define the complement of A in U
def complementA : Set ℝ := {x | x < -1 ∨ x > 3}

-- Theorem statement
theorem complement_of_A_in_U : Set.compl A = complementA := by sorry

end complement_of_A_in_U_l1307_130759


namespace sin_15_degrees_l1307_130792

theorem sin_15_degrees : Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end sin_15_degrees_l1307_130792


namespace congruence_problem_l1307_130732

theorem congruence_problem (x : ℤ) : 
  (5 * x + 9) % 20 = 3 → (3 * x + 14) % 20 = 2 := by
  sorry

end congruence_problem_l1307_130732


namespace tank_dimension_l1307_130764

theorem tank_dimension (x : ℝ) : 
  x > 0 ∧ 
  (2 * (x * 5 + x * 2 + 5 * 2)) * 20 = 1240 → 
  x = 3 :=
by sorry

end tank_dimension_l1307_130764


namespace andrews_snacks_l1307_130766

theorem andrews_snacks (num_friends : ℕ) (sandwiches_per_friend : ℕ) (cheese_crackers_per_friend : ℕ) (cookies_per_friend : ℕ) 
  (h1 : num_friends = 7)
  (h2 : sandwiches_per_friend = 5)
  (h3 : cheese_crackers_per_friend = 4)
  (h4 : cookies_per_friend = 3) :
  num_friends * sandwiches_per_friend + 
  num_friends * cheese_crackers_per_friend + 
  num_friends * cookies_per_friend = 84 := by
  sorry

end andrews_snacks_l1307_130766


namespace taxi_fare_for_100_miles_l1307_130761

/-- Represents the taxi fare system -/
structure TaxiFare where
  fixedCharge : ℝ
  fixedDistance : ℝ
  proportionalRate : ℝ

/-- Calculates the fare for a given distance -/
def calculateFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.fixedCharge + tf.proportionalRate * (distance - tf.fixedDistance)

theorem taxi_fare_for_100_miles 
  (tf : TaxiFare)
  (h1 : tf.fixedCharge = 20)
  (h2 : tf.fixedDistance = 10)
  (h3 : calculateFare tf 80 = 160) :
  calculateFare tf 100 = 200 := by
  sorry


end taxi_fare_for_100_miles_l1307_130761


namespace ellipse_hyperbola_coinciding_foci_l1307_130798

/-- Given an ellipse and a hyperbola with coinciding foci, prove that b² of the ellipse equals 14.76 -/
theorem ellipse_hyperbola_coinciding_foci (b : ℝ) : 
  (∀ x y : ℝ, x^2/25 + y^2/b^2 = 1 → x^2/100 - y^2/64 = 1/16 → 
   ∃ c : ℝ, c^2 = 25 - b^2 ∧ c^2 = 10.25) → 
  b^2 = 14.76 := by
sorry

end ellipse_hyperbola_coinciding_foci_l1307_130798


namespace reciprocal_equation_l1307_130723

theorem reciprocal_equation (x : ℝ) : 
  (3 + 1 / (2 - x) = 2 * (1 / (2 - x))) → x = 5/3 :=
by sorry

end reciprocal_equation_l1307_130723


namespace compound_interest_rate_l1307_130786

theorem compound_interest_rate (P r : ℝ) (h1 : P * (1 + r / 100) ^ 2 = 3650) (h2 : P * (1 + r / 100) ^ 3 = 4015) : r = 10 := by
  sorry

end compound_interest_rate_l1307_130786


namespace find_incorrect_value_l1307_130799

/-- Represents the problem of finding the incorrect value in a mean calculation --/
theorem find_incorrect_value (n : ℕ) (initial_mean correct_mean correct_value : ℚ) :
  n = 30 ∧ 
  initial_mean = 180 ∧ 
  correct_mean = 180 + 2/3 ∧ 
  correct_value = 155 →
  ∃ incorrect_value : ℚ,
    incorrect_value = 175 ∧
    n * initial_mean = (n - 1) * correct_mean + incorrect_value ∧
    n * correct_mean = (n - 1) * correct_mean + correct_value :=
by sorry

end find_incorrect_value_l1307_130799


namespace luncheon_tables_l1307_130797

theorem luncheon_tables (invited : ℕ) (no_show : ℕ) (per_table : ℕ) 
  (h1 : invited = 18) 
  (h2 : no_show = 12) 
  (h3 : per_table = 3) : 
  (invited - no_show) / per_table = 2 := by
  sorry

end luncheon_tables_l1307_130797


namespace least_value_property_l1307_130755

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_three_digit : hundreds ≥ 1 ∧ hundreds ≤ 9

/-- The value of a 3-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The sum of digits of a 3-digit number -/
def ThreeDigitNumber.digit_sum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.units

/-- Predicate for the difference between hundreds and tens being 8 -/
def digit_difference_eight (n : ThreeDigitNumber) : Prop :=
  n.tens - n.hundreds = 8 ∨ n.hundreds - n.tens = 8

theorem least_value_property (k : ThreeDigitNumber) 
  (h : digit_difference_eight k) :
  ∃ (min_k : ThreeDigitNumber), 
    digit_difference_eight min_k ∧
    ∀ (k' : ThreeDigitNumber), digit_difference_eight k' → 
      min_k.value ≤ k'.value ∧
      min_k.value = 19 * min_k.digit_sum :=
  sorry

end least_value_property_l1307_130755


namespace triangle_area_change_l1307_130718

theorem triangle_area_change (base height : ℝ) (base_new height_new area area_new : ℝ) 
  (h1 : base_new = 1.4 * base) 
  (h2 : height_new = 0.6 * height) 
  (h3 : area = (base * height) / 2) 
  (h4 : area_new = (base_new * height_new) / 2) : 
  area_new = 0.42 * area := by
sorry

end triangle_area_change_l1307_130718


namespace sin_difference_monotone_increasing_l1307_130769

/-- The function f(x) = sin(2x - π/3) - sin(2x) is monotonically increasing on [π/12, 7π/12] -/
theorem sin_difference_monotone_increasing :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x - π / 3) - Real.sin (2 * x)
  ∀ x y, π / 12 ≤ x ∧ x < y ∧ y ≤ 7 * π / 12 → f x < f y := by
  sorry

end sin_difference_monotone_increasing_l1307_130769


namespace triangle_max_perimeter_l1307_130773

theorem triangle_max_perimeter :
  ∃ (a b : ℕ), 
    a > 0 ∧ 
    b > 0 ∧ 
    b = 4 * a ∧ 
    a + b > 18 ∧ 
    a + 18 > b ∧ 
    b + 18 > a ∧
    ∀ (x y : ℕ), 
      x > 0 → 
      y > 0 → 
      y = 4 * x → 
      x + y > 18 → 
      x + 18 > y → 
      y + 18 > x → 
      a + b + 18 ≥ x + y + 18 ∧
    a + b + 18 = 43 :=
by sorry

end triangle_max_perimeter_l1307_130773


namespace vector_expression_l1307_130758

/-- Given vectors a, b, and c in ℝ², prove that c = a - 2b --/
theorem vector_expression (a b c : ℝ × ℝ) :
  a = (3, -2) → b = (-2, 1) → c = (7, -4) → c = a - 2 • b := by
  sorry

end vector_expression_l1307_130758


namespace triangle_inradius_l1307_130787

/-- Given a triangle with perimeter 48 cm and area 60 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h1 : p = 48) 
  (h2 : A = 60) 
  (h3 : A = r * p / 2) : 
  r = 2.5 := by
  sorry

end triangle_inradius_l1307_130787


namespace solution_value_l1307_130796

theorem solution_value (x a : ℝ) (h : 2 * 2 + a = 3) : a = -1 := by
  sorry

end solution_value_l1307_130796


namespace value_of_M_l1307_130789

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.35 * 1500) ∧ (M = 2100) := by sorry

end value_of_M_l1307_130789


namespace intersection_length_l1307_130730

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 5
def circle_O₂ (x y m : ℝ) : Prop := (x + m)^2 + y^2 = 20

-- Define the intersection points
structure IntersectionPoints (m : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : circle_O₁ A.1 A.2
  h₂ : circle_O₂ A.1 A.2 m
  h₃ : circle_O₁ B.1 B.2
  h₄ : circle_O₂ B.1 B.2 m

-- Define the perpendicular tangents condition
def perpendicular_tangents (m : ℝ) (A : ℝ × ℝ) : Prop :=
  circle_O₁ A.1 A.2 ∧ circle_O₂ A.1 A.2 m ∧
  (∃ t₁ t₂ : ℝ × ℝ, 
    (t₁.1 * t₂.1 + t₁.2 * t₂.2 = 0) ∧  -- Perpendicular condition
    (t₁.1 * A.1 + t₁.2 * A.2 = 0) ∧    -- Tangent to O₁
    (t₂.1 * (A.1 + m) + t₂.2 * A.2 = 0)) -- Tangent to O₂

-- Theorem statement
theorem intersection_length (m : ℝ) (points : IntersectionPoints m) :
  perpendicular_tangents m points.A →
  Real.sqrt ((points.A.1 - points.B.1)^2 + (points.A.2 - points.B.2)^2) = 4 :=
sorry

end intersection_length_l1307_130730


namespace trevor_eggs_left_l1307_130741

/-- Represents the number of eggs laid by each chicken and the number of eggs dropped --/
structure EggCollection where
  gertrude : ℕ
  blanche : ℕ
  nancy : ℕ
  martha : ℕ
  dropped : ℕ

/-- Calculates the number of eggs Trevor has left --/
def eggsLeft (collection : EggCollection) : ℕ :=
  collection.gertrude + collection.blanche + collection.nancy + collection.martha - collection.dropped

/-- Theorem stating that Trevor has 9 eggs left --/
theorem trevor_eggs_left :
  ∃ (collection : EggCollection),
    collection.gertrude = 4 ∧
    collection.blanche = 3 ∧
    collection.nancy = 2 ∧
    collection.martha = 2 ∧
    collection.dropped = 2 ∧
    eggsLeft collection = 9 := by
  sorry

end trevor_eggs_left_l1307_130741


namespace least_valid_k_l1307_130708

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_valid_k (k : ℤ) : Prop :=
  (0.00010101 * (10 : ℝ) ^ k > 100) ∧
  (sum_of_digits k.natAbs ≤ 15)

def exists_valid_m : Prop :=
  ∃ m : ℤ, 0.000515151 * (10 : ℝ) ^ m ≤ 500

theorem least_valid_k :
  is_valid_k 7 ∧ exists_valid_m ∧
  ∀ k : ℤ, k < 7 → ¬(is_valid_k k) :=
sorry

end least_valid_k_l1307_130708


namespace fraction_sum_equals_123_128th_l1307_130783

theorem fraction_sum_equals_123_128th : 
  (4 : ℚ) / 4 + 7 / 8 + 12 / 16 + 19 / 32 + 28 / 64 + 39 / 128 - 3 = 123 / 128 := by
  sorry

end fraction_sum_equals_123_128th_l1307_130783


namespace no_five_digit_flippy_divisible_by_11_l1307_130709

/-- A flippy number is a number whose digits alternate between two distinct digits. -/
def is_flippy (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≠ y ∧ x < 10 ∧ y < 10 ∧
  (n = x * 10000 + y * 1000 + x * 100 + y * 10 + x ∨
   n = y * 10000 + x * 1000 + y * 100 + x * 10 + y)

/-- A number is five digits long if it's between 10000 and 99999, inclusive. -/
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

theorem no_five_digit_flippy_divisible_by_11 :
  ¬∃ n : ℕ, is_flippy n ∧ is_five_digit n ∧ n % 11 = 0 :=
sorry

end no_five_digit_flippy_divisible_by_11_l1307_130709


namespace ratio_w_to_y_l1307_130744

/-- Given ratios between w, x, y, and z, prove the ratio of w to y -/
theorem ratio_w_to_y 
  (h1 : ∃ (k : ℚ), w = (5/2) * k ∧ x = k) 
  (h2 : ∃ (m : ℚ), y = 4 * m ∧ z = m) 
  (h3 : ∃ (n : ℚ), z = (1/8) * n ∧ x = n) : 
  w = 5 * y := by sorry

end ratio_w_to_y_l1307_130744


namespace max_eggs_per_basket_l1307_130721

def red_eggs : ℕ := 15
def blue_eggs : ℕ := 30
def min_eggs_per_basket : ℕ := 3

def is_valid_distribution (eggs_per_basket : ℕ) : Prop :=
  eggs_per_basket ≥ min_eggs_per_basket ∧
  red_eggs % eggs_per_basket = 0 ∧
  blue_eggs % eggs_per_basket = 0

theorem max_eggs_per_basket :
  ∃ (max : ℕ), is_valid_distribution max ∧
    ∀ (n : ℕ), is_valid_distribution n → n ≤ max :=
by sorry

end max_eggs_per_basket_l1307_130721


namespace characterize_solutions_l1307_130725

/-- The functional equation satisfied by f and g -/
def functional_equation (f g : ℕ → ℕ) : Prop :=
  ∀ n, f n + f (n + g n) = f (n + 1)

/-- The trivial solution where f is identically zero -/
def trivial_solution (f g : ℕ → ℕ) : Prop :=
  ∀ n, f n = 0

/-- The non-trivial solution family -/
def non_trivial_solution (f g : ℕ → ℕ) : Prop :=
  ∃ n₀ c : ℕ,
    (∀ n < n₀, f n = 0) ∧
    (∀ n ≥ n₀, f n = c * 2^(n - n₀)) ∧
    (∀ n < n₀ - 1, g n < n₀ - n) ∧
    (g (n₀ - 1) = 1) ∧
    (∀ n ≥ n₀, g n = 0)

/-- The main theorem characterizing all solutions to the functional equation -/
theorem characterize_solutions (f g : ℕ → ℕ) :
  functional_equation f g → (trivial_solution f g ∨ non_trivial_solution f g) :=
sorry

end characterize_solutions_l1307_130725


namespace tangent_line_minimum_b_l1307_130706

/-- Given a > 0 and y = 2x + b is tangent to y = 2a ln x, the minimum value of b is -2 -/
theorem tangent_line_minimum_b (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, (2 * x + b = 2 * a * Real.log x) ∧ 
             (∀ y : ℝ, y ≠ x → 2 * y + b > 2 * a * Real.log y)) → 
  (∀ c : ℝ, (∃ x : ℝ, (2 * x + c = 2 * a * Real.log x) ∧ 
                       (∀ y : ℝ, y ≠ x → 2 * y + c > 2 * a * Real.log y)) → 
            c ≥ -2) :=
sorry

end tangent_line_minimum_b_l1307_130706


namespace max_k_value_l1307_130760

theorem max_k_value (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (∀ k : ℝ, (4 / (a - b) + 1 / (b - c) + k / (c - a) ≥ 0) → k ≤ 9) ∧ 
  (∃ k : ℝ, k = 9 ∧ 4 / (a - b) + 1 / (b - c) + k / (c - a) ≥ 0) :=
sorry

end max_k_value_l1307_130760


namespace lemonade_problem_l1307_130753

theorem lemonade_problem (x : ℝ) :
  x > 0 ∧
  (x + (x / 8 + 2) = (3 / 2 * x) - (x / 8 + 2)) →
  x + (3 / 2 * x) = 40 := by
sorry

end lemonade_problem_l1307_130753


namespace abc_sum_sqrt_l1307_130745

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 5 := by
  sorry

end abc_sum_sqrt_l1307_130745


namespace liam_bills_cost_liam_bills_proof_l1307_130719

/-- Calculates the cost of Liam's bills given his savings and remaining money. -/
theorem liam_bills_cost (monthly_savings : ℕ) (savings_duration_months : ℕ) (money_left : ℕ) : ℕ :=
  let total_savings := monthly_savings * savings_duration_months
  total_savings - money_left

/-- Proves that Liam's bills cost $3,500 given the problem conditions. -/
theorem liam_bills_proof :
  liam_bills_cost 500 24 8500 = 3500 := by
  sorry

end liam_bills_cost_liam_bills_proof_l1307_130719


namespace sphere_radius_from_intersection_l1307_130785

theorem sphere_radius_from_intersection (r h : ℝ) : 
  r > 0 → h > 0 → r^2 + h^2 = (r + h)^2 → r = 12 → h = 8 → r + h = 13 := by
  sorry

end sphere_radius_from_intersection_l1307_130785


namespace probability_score_at_most_seven_l1307_130716

/-- The probability of scoring at most 7 points when drawing 4 balls from a bag containing 4 red balls (1 point each) and 3 black balls (3 points each) -/
theorem probability_score_at_most_seven (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) (red_score : ℕ) (black_score : ℕ) :
  total_balls = red_balls + black_balls →
  red_balls = 4 →
  black_balls = 3 →
  drawn_balls = 4 →
  red_score = 1 →
  black_score = 3 →
  (Nat.choose total_balls drawn_balls : ℚ) * (13 : ℚ) / (35 : ℚ) = 
    (Nat.choose red_balls drawn_balls : ℚ) + 
    (Nat.choose red_balls (drawn_balls - 1) : ℚ) * (Nat.choose black_balls 1 : ℚ) :=
by sorry

#check probability_score_at_most_seven

end probability_score_at_most_seven_l1307_130716


namespace sqrt_sum_squares_is_integer_l1307_130772

theorem sqrt_sum_squares_is_integer : ∃ (z : ℕ), z * z = 25530 * 25530 + 29464 * 29464 := by
  sorry

end sqrt_sum_squares_is_integer_l1307_130772


namespace nail_fractions_l1307_130705

theorem nail_fractions (fraction_4d fraction_total : ℝ) 
  (h1 : fraction_4d = 0.5)
  (h2 : fraction_total = 0.75) : 
  fraction_total - fraction_4d = 0.25 := by
  sorry

end nail_fractions_l1307_130705


namespace sqrt_difference_inequality_l1307_130775

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end sqrt_difference_inequality_l1307_130775


namespace p_20_equals_657_l1307_130722

/-- A polynomial p(x) = 3x^2 + kx + 117 where k is a constant such that p(1) = p(10) -/
def p (k : ℚ) (x : ℚ) : ℚ := 3 * x^2 + k * x + 117

/-- The theorem stating that for the polynomial p(x) with the given properties, p(20) = 657 -/
theorem p_20_equals_657 :
  ∃ k : ℚ, (p k 1 = p k 10) ∧ (p k 20 = 657) := by sorry

end p_20_equals_657_l1307_130722


namespace expand_expression_l1307_130713

theorem expand_expression (x y z : ℝ) : 
  (x + 12) * (3 * y + 4 * z + 15) = 3 * x * y + 4 * x * z + 15 * x + 36 * y + 48 * z + 180 := by
  sorry

end expand_expression_l1307_130713


namespace simplify_fraction_l1307_130791

theorem simplify_fraction : (333 : ℚ) / 9999 * 99 = 37 / 101 := by
  sorry

end simplify_fraction_l1307_130791


namespace equality_holds_iff_l1307_130738

theorem equality_holds_iff (α : ℝ) : 
  Real.sqrt (1 + Real.sin (2 * α)) = Real.sin α + Real.cos α ↔ 
  -π/4 < α ∧ α < 3*π/4 :=
sorry

end equality_holds_iff_l1307_130738


namespace cards_kept_away_is_two_l1307_130768

/-- The number of cards in a standard deck of playing cards. -/
def standard_deck_size : ℕ := 52

/-- The number of cards used for playing. -/
def cards_used : ℕ := 50

/-- The number of cards kept away. -/
def cards_kept_away : ℕ := standard_deck_size - cards_used

theorem cards_kept_away_is_two : cards_kept_away = 2 := by
  sorry

end cards_kept_away_is_two_l1307_130768


namespace punch_bowl_problem_l1307_130726

/-- The capacity of the punch bowl in gallons -/
def bowl_capacity : ℝ := 16

/-- The amount of punch Sally drinks in gallons -/
def sally_drinks : ℝ := 2

/-- The amount of punch Mark adds to completely fill the bowl after Sally drinks -/
def final_addition : ℝ := 12

/-- The amount of punch Mark adds after his cousin drinks -/
def mark_addition : ℝ := 12

theorem punch_bowl_problem :
  ∃ (initial_amount : ℝ),
    initial_amount ≥ 0 ∧
    initial_amount ≤ bowl_capacity ∧
    (initial_amount / 2 + mark_addition - sally_drinks + final_addition = bowl_capacity) :=
by
  sorry

#check punch_bowl_problem

end punch_bowl_problem_l1307_130726


namespace father_age_proof_l1307_130771

/-- The age of the father -/
def father_age : ℕ := 48

/-- The age of the son -/
def son_age : ℕ := 75 - father_age

/-- The time difference between when the father was the son's current age and now -/
def time_difference : ℕ := father_age - son_age

theorem father_age_proof :
  (father_age + son_age = 75) ∧
  (father_age = 8 * (son_age - time_difference)) ∧
  (father_age - time_difference = son_age) →
  father_age = 48 :=
by sorry

end father_age_proof_l1307_130771


namespace expected_value_of_coins_l1307_130733

/-- Represents a coin with its value in cents and probability of heads -/
structure Coin where
  value : ℝ
  prob_heads : ℝ

/-- The set of coins being flipped -/
def coins : Finset Coin := sorry

/-- The expected value of a single coin -/
def expected_value (c : Coin) : ℝ := c.value * c.prob_heads

/-- The total expected value of all coins -/
def total_expected_value : ℝ := (coins.sum expected_value)

/-- Theorem stating the expected value of coins coming up heads -/
theorem expected_value_of_coins : 
  (coins.card = 5) → 
  (∃ c ∈ coins, c.value = 1 ∧ c.prob_heads = 1/2) →
  (∃ c ∈ coins, c.value = 5 ∧ c.prob_heads = 1/2) →
  (∃ c ∈ coins, c.value = 10 ∧ c.prob_heads = 1/2) →
  (∃ c ∈ coins, c.value = 25 ∧ c.prob_heads = 1/2) →
  (∃ c ∈ coins, c.value = 50 ∧ c.prob_heads = 3/4) →
  total_expected_value = 58 :=
by sorry

end expected_value_of_coins_l1307_130733


namespace continuity_from_g_and_h_continuous_l1307_130720

open Function Set Filter Topology

/-- Given a function f: ℝ → ℝ, if g(x) = f(x) + f(2x) and h(x) = f(x) + f(4x) are continuous,
    then f is continuous. -/
theorem continuity_from_g_and_h_continuous
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h : ℝ → ℝ)
  (hg : g = λ x => f x + f (2 * x))
  (hh : h = λ x => f x + f (4 * x))
  (hg_cont : Continuous g)
  (hh_cont : Continuous h) :
  Continuous f :=
sorry

end continuity_from_g_and_h_continuous_l1307_130720


namespace total_notes_count_l1307_130746

def total_amount : ℕ := 10350
def note_50_value : ℕ := 50
def note_500_value : ℕ := 500
def num_50_notes : ℕ := 17

theorem total_notes_count : 
  ∃ (num_500_notes : ℕ), 
    num_50_notes * note_50_value + num_500_notes * note_500_value = total_amount ∧
    num_50_notes + num_500_notes = 36 := by
  sorry

end total_notes_count_l1307_130746


namespace luis_red_socks_l1307_130735

/-- The number of pairs of red socks Luis bought -/
def red_socks : ℕ := sorry

/-- The number of pairs of blue socks Luis bought -/
def blue_socks : ℕ := 6

/-- The cost of each pair of red socks in dollars -/
def red_sock_cost : ℕ := 3

/-- The cost of each pair of blue socks in dollars -/
def blue_sock_cost : ℕ := 5

/-- The total amount Luis spent in dollars -/
def total_spent : ℕ := 42

/-- Theorem stating that Luis bought 4 pairs of red socks -/
theorem luis_red_socks : 
  red_socks * red_sock_cost + blue_socks * blue_sock_cost = total_spent → 
  red_socks = 4 := by sorry

end luis_red_socks_l1307_130735


namespace ball_max_height_l1307_130728

/-- The height function of a ball traveling along a parabolic path -/
def h (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 20

/-- The maximum height reached by the ball -/
def max_height : ℝ := 145

theorem ball_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ max_height :=
sorry

end ball_max_height_l1307_130728


namespace sum_of_absolute_coefficients_l1307_130790

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 729 := by
sorry

end sum_of_absolute_coefficients_l1307_130790


namespace simplify_expression_l1307_130777

theorem simplify_expression : (5 + 7 + 8) / 3 - 2 / 3 = 6 := by sorry

end simplify_expression_l1307_130777


namespace factorial_sum_equals_36018_l1307_130710

theorem factorial_sum_equals_36018 : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 3 * Nat.factorial 3 + Nat.factorial 5 = 36018 := by
  sorry

end factorial_sum_equals_36018_l1307_130710


namespace equal_cake_distribution_l1307_130794

theorem equal_cake_distribution (total_cakes : ℕ) (num_children : ℕ) (cakes_per_child : ℕ) : 
  total_cakes = 18 → 
  num_children = 3 → 
  total_cakes = num_children * cakes_per_child →
  cakes_per_child = 6 := by
sorry

end equal_cake_distribution_l1307_130794


namespace initial_amount_calculation_l1307_130757

/-- Given a person receives additional money and the difference between their
initial amount and the received amount is known, calculate their initial amount. -/
theorem initial_amount_calculation (received : ℕ) (difference : ℕ) : 
  received = 13 → difference = 11 → received + difference = 24 := by
  sorry

end initial_amount_calculation_l1307_130757


namespace parabola_properties_l1307_130779

-- Define the parabola C
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def Focus : ℝ × ℝ := (1, 0)

-- Define the directrix of the parabola
def Directrix (x : ℝ) : Prop := x = -1

-- Define a point on the parabola
def PointOnParabola (p : ℝ × ℝ) : Prop := Parabola p.1 p.2

-- Define a line passing through two points
def Line (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

-- Theorem statement
theorem parabola_properties :
  ∀ (M N : ℝ × ℝ),
  Directrix M.1 ∧ Directrix N.1 →
  M.2 * N.2 = -4 →
  ∃ (A B : ℝ × ℝ) (F : ℝ × ℝ → ℝ × ℝ),
    PointOnParabola A ∧ PointOnParabola B ∧
    Line (0, 0) M A.1 A.2 ∧
    Line (0, 0) N B.1 B.2 ∧
    (∀ (x y : ℝ), Line A B x y → Line A B (F A).1 (F A).2) :=
sorry

end parabola_properties_l1307_130779


namespace range_of_a_l1307_130756

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x + Real.exp x - 1 / Real.exp x

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a^2) ≤ 0) : -1 ≤ a ∧ a ≤ 1/2 := by
  sorry

end range_of_a_l1307_130756


namespace fair_coin_three_heads_probability_l1307_130714

theorem fair_coin_three_heads_probability :
  let p_head : ℚ := 1/2  -- Probability of getting heads on a single flip
  let p_three_heads : ℚ := p_head * p_head * p_head  -- Probability of getting heads on all three flips
  p_three_heads = 1/8 := by
  sorry

end fair_coin_three_heads_probability_l1307_130714


namespace min_average_of_four_integers_l1307_130717

theorem min_average_of_four_integers (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  d = 90 ∧                 -- Largest is 90
  a ≥ 29 →                 -- Smallest is at least 29
  (a + b + c + d) / 4 ≥ 45 :=
sorry

end min_average_of_four_integers_l1307_130717


namespace inequality_proof_l1307_130727

theorem inequality_proof (x y z t : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : t ≥ 0)
  (h5 : x * y * z = 2) (h6 : y + z + t = 2 * Real.sqrt 2) :
  2 * x^2 + y^2 + z^2 + t^2 ≥ 6 := by
  sorry

end inequality_proof_l1307_130727


namespace incorrect_inequality_l1307_130762

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬(-3 * a > -3 * b) := by
  sorry

end incorrect_inequality_l1307_130762


namespace fraction_transformation_l1307_130729

theorem fraction_transformation (a b c d x : ℚ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a + x) / (b + x) = c / d) : 
  x = (a * d - b * c) / (c - d) := by
sorry

end fraction_transformation_l1307_130729


namespace impossibleToGetAllPlus_l1307_130770

/-- Represents a 4x4 grid of signs -/
def Grid := Matrix (Fin 4) (Fin 4) Bool

/-- Flips all signs in a given row -/
def flipRow (g : Grid) (row : Fin 4) : Grid := sorry

/-- Flips all signs in a given column -/
def flipColumn (g : Grid) (col : Fin 4) : Grid := sorry

/-- The initial grid configuration -/
def initialGrid : Grid := 
  ![![true,  false, true,  true],
    ![true,  true,  true,  true],
    ![true,  true,  true,  true],
    ![true,  false, true,  true]]

/-- Checks if all cells in the grid are true ("+") -/
def allPlus (g : Grid) : Prop := ∀ i j, g i j = true

/-- Represents a sequence of row and column flipping operations -/
inductive FlipSequence : Type
  | empty : FlipSequence
  | flipRow : FlipSequence → Fin 4 → FlipSequence
  | flipColumn : FlipSequence → Fin 4 → FlipSequence

/-- Applies a sequence of flipping operations to a grid -/
def applyFlips : Grid → FlipSequence → Grid
  | g, FlipSequence.empty => g
  | g, FlipSequence.flipRow s i => applyFlips (flipRow g i) s
  | g, FlipSequence.flipColumn s j => applyFlips (flipColumn g j) s

theorem impossibleToGetAllPlus : 
  ¬∃ (s : FlipSequence), allPlus (applyFlips initialGrid s) := by
  sorry

end impossibleToGetAllPlus_l1307_130770


namespace arithmetic_mean_difference_l1307_130778

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 26) : 
  r - p = 32 := by
sorry

end arithmetic_mean_difference_l1307_130778


namespace nine_sided_figure_perimeter_l1307_130737

/-- The perimeter of a regular polygon with n sides of length s is n * s -/
def perimeter (n : ℕ) (s : ℝ) : ℝ := n * s

theorem nine_sided_figure_perimeter :
  let n : ℕ := 9
  let s : ℝ := 2
  perimeter n s = 18 := by sorry

end nine_sided_figure_perimeter_l1307_130737


namespace solution_equality_l1307_130754

theorem solution_equality (a b c d : ℝ) 
  (eq1 : a - Real.sqrt (1 - b^2) + Real.sqrt (1 - c^2) = d)
  (eq2 : b - Real.sqrt (1 - c^2) + Real.sqrt (1 - d^2) = a)
  (eq3 : c - Real.sqrt (1 - d^2) + Real.sqrt (1 - a^2) = b)
  (eq4 : d - Real.sqrt (1 - a^2) + Real.sqrt (1 - b^2) = c)
  (nonneg1 : 1 - a^2 ≥ 0)
  (nonneg2 : 1 - b^2 ≥ 0)
  (nonneg3 : 1 - c^2 ≥ 0)
  (nonneg4 : 1 - d^2 ≥ 0) :
  a = b ∧ b = c ∧ c = d := by
  sorry

end solution_equality_l1307_130754


namespace dianes_honey_harvest_l1307_130767

/-- Diane's honey harvest calculation -/
theorem dianes_honey_harvest 
  (last_year_harvest : ℕ) 
  (harvest_increase : ℕ) 
  (h1 : last_year_harvest = 2479)
  (h2 : harvest_increase = 6085) : 
  last_year_harvest + harvest_increase = 8564 := by
  sorry

end dianes_honey_harvest_l1307_130767


namespace xy_squared_change_l1307_130707

/-- Theorem: Change in xy^2 when x increases by 20% and y decreases by 30% --/
theorem xy_squared_change (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let x' := 1.2 * x
  let y' := 0.7 * y
  1 - (x' * y' * y') / (x * y * y) = 0.412 := by
  sorry

end xy_squared_change_l1307_130707


namespace flower_town_coin_impossibility_l1307_130788

/-- Represents the number of inhabitants in Flower Town -/
def num_inhabitants : ℕ := 1990

/-- Represents the number of coins each inhabitant must give -/
def coins_per_inhabitant : ℕ := 10

/-- Represents a meeting between two inhabitants -/
structure Meeting where
  giver : Fin num_inhabitants
  receiver : Fin num_inhabitants
  giver_gives_10 : Bool

/-- The main theorem stating the impossibility of the scenario -/
theorem flower_town_coin_impossibility :
  ¬ ∃ (meetings : List Meeting),
    (∀ i : Fin num_inhabitants, 
      (meetings.filter (λ m => m.giver = i ∨ m.receiver = i)).length = coins_per_inhabitant) ∧
    (∀ m : Meeting, m ∈ meetings → m.giver ≠ m.receiver) :=
by
  sorry


end flower_town_coin_impossibility_l1307_130788


namespace problem_1_l1307_130715

theorem problem_1 (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -1) : 
  2 / (a + 1) - (a - 2) / (a^2 - 1) / ((a^2 - 2*a) / (a^2 - 2*a + 1)) = 1 / a := by
  sorry

end problem_1_l1307_130715


namespace triangle_side_length_l1307_130780

theorem triangle_side_length (a c area : ℝ) (ha : a = 1) (hc : c = 7) (harea : area = 5) :
  let h := 2 * area / c
  let b := Real.sqrt ((a^2 + h^2) : ℝ)
  b = Real.sqrt 149 / 7 := by
sorry

end triangle_side_length_l1307_130780
