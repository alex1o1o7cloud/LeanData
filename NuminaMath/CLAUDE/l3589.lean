import Mathlib

namespace circle_center_coordinates_l3589_358921

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 --/
def CircleEquation (h k r : ℝ) : ℝ × ℝ → Prop :=
  λ p => (p.1 - h)^2 + (p.2 - k)^2 = r^2

/-- The center of a circle given by its equation --/
def CircleCenter (eq : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

theorem circle_center_coordinates :
  CircleCenter (CircleEquation 1 2 1) = (1, 2) := by sorry

end circle_center_coordinates_l3589_358921


namespace equation_solution_l3589_358928

theorem equation_solution :
  ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0 :=
by
  -- The proof would go here
  sorry

end equation_solution_l3589_358928


namespace cricketer_wickets_after_match_l3589_358967

/-- Represents a cricketer's bowling statistics -/
structure CricketerStats where
  wickets : ℕ
  runs : ℕ
  average : ℚ

/-- Calculates the new average after a match -/
def newAverage (stats : CricketerStats) (newWickets : ℕ) (newRuns : ℕ) : ℚ :=
  (stats.runs + newRuns) / (stats.wickets + newWickets)

/-- Theorem: A cricketer with given stats takes 5 wickets for 26 runs, decreasing average by 0.4 -/
theorem cricketer_wickets_after_match 
  (stats : CricketerStats)
  (h1 : stats.average = 12.4)
  (h2 : newAverage stats 5 26 = stats.average - 0.4) :
  stats.wickets + 5 = 90 := by
sorry

end cricketer_wickets_after_match_l3589_358967


namespace brick_width_calculation_l3589_358958

theorem brick_width_calculation (courtyard_length courtyard_width : ℝ)
                                (brick_length : ℝ)
                                (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 16 →
  brick_length = 0.2 →
  total_bricks = 20000 →
  ∃ (brick_width : ℝ),
    brick_width = 0.1 ∧
    courtyard_length * courtyard_width * 10000 = total_bricks * brick_length * brick_width :=
by
  sorry

end brick_width_calculation_l3589_358958


namespace inequality_not_always_true_l3589_358925

theorem inequality_not_always_true (x y : ℝ) (h : x > 1 ∧ 1 > y) :
  ¬ (∀ x y : ℝ, x > 1 ∧ 1 > y → x - 1 > 1 - y) :=
by sorry

end inequality_not_always_true_l3589_358925


namespace sum_of_polynomials_l3589_358906

/-- The polynomial p(x) -/
def p (x : ℝ) : ℝ := -4*x^2 + 2*x - 5

/-- The polynomial q(x) -/
def q (x : ℝ) : ℝ := -6*x^2 + 4*x - 9

/-- The polynomial r(x) -/
def r (x : ℝ) : ℝ := 6*x^2 + 6*x + 2

/-- The polynomial s(x) -/
def s (x : ℝ) : ℝ := 3*x^2 - 2*x + 1

/-- The sum of polynomials p(x), q(x), r(x), and s(x) is equal to -x^2 + 10x - 11 -/
theorem sum_of_polynomials (x : ℝ) : p x + q x + r x + s x = -x^2 + 10*x - 11 := by
  sorry

end sum_of_polynomials_l3589_358906


namespace chandra_akiko_ratio_l3589_358970

/-- Represents the points scored by each player in the basketball game -/
structure GameScores where
  chandra : ℕ
  akiko : ℕ
  michiko : ℕ
  bailey : ℕ

/-- The conditions of the basketball game -/
def gameConditions (s : GameScores) : Prop :=
  s.akiko = s.michiko + 4 ∧
  s.michiko * 2 = s.bailey ∧
  s.bailey = 14 ∧
  s.chandra + s.akiko + s.michiko + s.bailey = 54

/-- The theorem stating the ratio of Chandra's points to Akiko's points -/
theorem chandra_akiko_ratio (s : GameScores) : 
  gameConditions s → s.chandra * 1 = s.akiko * 2 := by
  sorry

#check chandra_akiko_ratio

end chandra_akiko_ratio_l3589_358970


namespace inequality_proof_l3589_358956

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) :
  x^2 * y^2 + |x^2 - y^2| ≤ π/2 := by
  sorry

end inequality_proof_l3589_358956


namespace special_divisor_form_l3589_358984

/-- A function that checks if a number is of the form a^r + 1 --/
def isOfForm (d : ℕ) : Prop :=
  ∃ (a r : ℕ), a > 0 ∧ r > 1 ∧ d = a^r + 1

/-- The main theorem --/
theorem special_divisor_form (n : ℕ) :
  n > 1 ∧ (∀ d : ℕ, 1 < d ∧ d ∣ n → isOfForm d) →
  n = 10 ∨ ∃ a : ℕ, n = a^2 + 1 :=
sorry

end special_divisor_form_l3589_358984


namespace final_amoeba_is_blue_l3589_358964

/-- Represents the color of an amoeba -/
inductive AmoebaCop
  | Red
  | Blue
  | Yellow

/-- Represents the state of the puddle -/
structure PuddleState where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Determines if a number is odd -/
def isOdd (n : Nat) : Bool :=
  n % 2 = 1

/-- The initial state of the puddle -/
def initialState : PuddleState :=
  { red := 26, blue := 31, yellow := 16 }

/-- Determines the color of the final amoeba based on the initial state -/
def finalAmoeba (state : PuddleState) : AmoebaCop :=
  if isOdd (state.red - state.blue) ∧ 
     isOdd (state.blue - state.yellow) ∧ 
     ¬isOdd (state.red - state.yellow)
  then AmoebaCop.Blue
  else if isOdd (state.red - state.blue) ∧ 
          isOdd (state.red - state.yellow) ∧ 
          ¬isOdd (state.blue - state.yellow)
  then AmoebaCop.Red
  else AmoebaCop.Yellow

theorem final_amoeba_is_blue :
  finalAmoeba initialState = AmoebaCop.Blue :=
by
  sorry


end final_amoeba_is_blue_l3589_358964


namespace roots_not_analytically_determinable_l3589_358922

/-- The polynomial equation whose roots we want to determine -/
def f (x : ℝ) : ℝ := (x - 2) * (x + 5)^3 * (5 - x) - 8

/-- Theorem stating that the roots of the polynomial equation cannot be determined analytically -/
theorem roots_not_analytically_determinable :
  ¬ ∃ (roots : Set ℝ), ∀ (x : ℝ), x ∈ roots ↔ f x = 0 ∧ 
  ∃ (formula : ℝ → ℝ), ∀ (x : ℝ), x ∈ roots → ∃ (n : ℕ), formula x = x ∧ 
  (∀ (y : ℝ), formula y = y → y ∈ roots) :=
sorry

end roots_not_analytically_determinable_l3589_358922


namespace car_a_speed_car_a_speed_is_58_l3589_358920

/-- Proves that the speed of Car A is 58 miles per hour given the initial conditions -/
theorem car_a_speed (initial_distance : ℝ) (time : ℝ) (speed_b : ℝ) : ℝ :=
  let distance_b := speed_b * time
  let total_distance := distance_b + initial_distance + 8
  total_distance / time

#check car_a_speed 24 4 50 = 58

/-- Theorem stating that the speed of Car A is indeed 58 miles per hour -/
theorem car_a_speed_is_58 :
  car_a_speed 24 4 50 = 58 := by sorry

end car_a_speed_car_a_speed_is_58_l3589_358920


namespace range_of_a_min_value_of_g_l3589_358990

-- Define the quadratic function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + a

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * f a x - a^2 * (x + 1) - 2*x

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧
   f a x₁ = x₁ ∧ f a x₂ = x₂) →
  0 < a ∧ a < 3 - 2 * Real.sqrt 2 :=
sorry

-- Theorem for the minimum value of g
theorem min_value_of_g (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, 
    (a < 1 → g a x ≥ a - 2) ∧
    (a ≥ 1 → g a x ≥ -1/a)) ∧
  (∃ x ∈ Set.Icc 0 1, 
    (a < 1 → g a x = a - 2) ∧
    (a ≥ 1 → g a x = -1/a)) :=
sorry

end range_of_a_min_value_of_g_l3589_358990


namespace sum_of_digits_l3589_358940

/-- 
Given a three-digit number ABC, where:
- ABC is an integer between 100 and 999 (inclusive)
- ABC = 17 * 28 + 9

Prove that the sum of its digits A, B, and C is 17.
-/
theorem sum_of_digits (ABC : ℕ) (h1 : 100 ≤ ABC) (h2 : ABC ≤ 999) (h3 : ABC = 17 * 28 + 9) :
  (ABC / 100) + ((ABC / 10) % 10) + (ABC % 10) = 17 := by
  sorry

end sum_of_digits_l3589_358940


namespace min_value_theorem_l3589_358934

theorem min_value_theorem (x y : ℝ) (hx : x > 1) (hy : y > 1) (h_sum : x + 2*y = 5) :
  1/(x-1) + 1/(y-1) ≥ 3/2 + Real.sqrt 2 := by
  sorry

end min_value_theorem_l3589_358934


namespace basketball_only_count_l3589_358923

theorem basketball_only_count (total students_basketball students_table_tennis students_neither : ℕ) :
  total = 30 ∧
  students_basketball = 15 ∧
  students_table_tennis = 10 ∧
  students_neither = 8 →
  ∃ (students_both : ℕ),
    students_basketball - students_both = 12 ∧
    students_both + (students_basketball - students_both) + (students_table_tennis - students_both) + students_neither = total :=
by sorry

end basketball_only_count_l3589_358923


namespace log_positive_iff_greater_than_one_l3589_358907

theorem log_positive_iff_greater_than_one (x : ℝ) : x > 1 ↔ Real.log x > 0 := by
  sorry

end log_positive_iff_greater_than_one_l3589_358907


namespace total_pages_in_collection_l3589_358943

/-- Represents a book in the reader's collection -/
structure Book where
  chapterPages : List Nat
  additionalPages : Nat

/-- The reader's book collection -/
def bookCollection : List Book := [
  { chapterPages := [22, 34, 18, 46, 30, 38], additionalPages := 14 },  -- Science
  { chapterPages := [24, 32, 40, 20], additionalPages := 13 },          -- History
  { chapterPages := [12, 28, 16, 22, 18, 26, 20], additionalPages := 8 }, -- Literature
  { chapterPages := [48, 52, 36, 62, 24], additionalPages := 18 },      -- Art
  { chapterPages := [16, 28, 44], additionalPages := 28 }               -- Mathematics
]

/-- Calculate the total pages in a book -/
def totalPagesInBook (book : Book) : Nat :=
  (book.chapterPages.sum) + book.additionalPages

/-- Calculate the total pages in the collection -/
def totalPagesInCollection (collection : List Book) : Nat :=
  collection.map totalPagesInBook |>.sum

/-- Theorem: The total number of pages in the reader's collection is 837 -/
theorem total_pages_in_collection :
  totalPagesInCollection bookCollection = 837 := by
  sorry


end total_pages_in_collection_l3589_358943


namespace intersection_when_a_zero_subset_condition_l3589_358945

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- Theorem 1: When a = 0, A ∩ B = {x | 0 < x < 1}
theorem intersection_when_a_zero :
  A 0 ∩ B = {x | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: A ⊆ B if and only if 1 ≤ a ≤ 2
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ 1 ≤ a ∧ a ≤ 2 := by sorry

end intersection_when_a_zero_subset_condition_l3589_358945


namespace percentage_calculation_l3589_358929

theorem percentage_calculation (x : ℝ) (h : 0.2 * x = 300) : 1.2 * x = 1800 := by
  sorry

end percentage_calculation_l3589_358929


namespace sequence_general_term_l3589_358911

/-- The sequence a_n defined by a_1 = 2 and a_{n+1} = 2a_n for n ≥ 1 has the general term a_n = 2^n -/
theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, a (n + 1) = 2 * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^n :=
sorry

end sequence_general_term_l3589_358911


namespace polynomial_value_at_two_l3589_358982

theorem polynomial_value_at_two :
  let f : ℝ → ℝ := fun x ↦ x^2 - 3*x + 2
  f 2 = 0 := by
sorry

end polynomial_value_at_two_l3589_358982


namespace initial_deposit_l3589_358988

theorem initial_deposit (P R : ℝ) : 
  P + (P * R * 3) / 100 = 9200 →
  P + (P * (R + 0.5) * 3) / 100 = 9320 →
  P = 8000 := by
sorry

end initial_deposit_l3589_358988


namespace function_inequality_implies_parameter_bound_l3589_358916

/-- Given a function f(x) = (x+a)e^x that satisfies f(x) ≥ (1/6)x^3 - x - 2 for all x ∈ ℝ,
    prove that a ≥ -2. -/
theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∀ x : ℝ, (x + a) * Real.exp x ≥ (1/6) * x^3 - x - 2) →
  a ≥ -2 :=
by sorry

end function_inequality_implies_parameter_bound_l3589_358916


namespace deane_gas_cost_l3589_358931

/-- Calculates the total cost of gas for Mr. Deane --/
def total_gas_cost (rollback : ℝ) (current_price : ℝ) (liters_today : ℝ) (liters_friday : ℝ) : ℝ :=
  let price_friday := current_price - rollback
  let cost_today := current_price * liters_today
  let cost_friday := price_friday * liters_friday
  cost_today + cost_friday

/-- Proves that Mr. Deane's total gas cost is $39 --/
theorem deane_gas_cost :
  let rollback : ℝ := 0.4
  let current_price : ℝ := 1.4
  let liters_today : ℝ := 10
  let liters_friday : ℝ := 25
  total_gas_cost rollback current_price liters_today liters_friday = 39 := by
  sorry

#eval total_gas_cost 0.4 1.4 10 25

end deane_gas_cost_l3589_358931


namespace average_price_is_18_l3589_358974

/-- The average price per book given two book purchases -/
def average_price_per_book (books1 books2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / (books1 + books2)

/-- Theorem stating that the average price per book is 18 for the given purchases -/
theorem average_price_is_18 :
  average_price_per_book 65 50 1150 920 = 18 := by
  sorry

end average_price_is_18_l3589_358974


namespace smallest_circle_area_l3589_358991

/-- The smallest area of a circle passing through two given points -/
theorem smallest_circle_area (x₁ y₁ x₂ y₂ : ℝ) : 
  let d := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  let r := d / 2
  let A := π * r^2
  x₁ = -3 ∧ y₁ = -2 ∧ x₂ = 2 ∧ y₂ = 4 →
  A = (61 * π) / 4 :=
by sorry

end smallest_circle_area_l3589_358991


namespace largest_interesting_is_max_l3589_358947

/-- A natural number is interesting if all its digits, except for the first and last,
    are less than the arithmetic mean of their two neighboring digits. -/
def is_interesting (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

/-- The largest interesting number -/
def largest_interesting : ℕ := 96433469

theorem largest_interesting_is_max :
  is_interesting largest_interesting ∧
  ∀ n : ℕ, is_interesting n → n ≤ largest_interesting :=
sorry

end largest_interesting_is_max_l3589_358947


namespace cannot_visit_all_friends_l3589_358994

-- Define the building structure
structure Building where
  num_floors : ℕ
  start_floor : ℕ
  friend_floors : List ℕ
  elevator_moves : List ℕ

-- Define the problem specifics
def problem : Building :=
  { num_floors := 14
  , start_floor := 1
  , friend_floors := [12, 14]
  , elevator_moves := [3, 7]
  }

-- Define a single elevator trip
def elevator_trip (current : ℤ) (move : ℤ) : ℤ :=
  current + move

-- Define if a floor is reachable within given moves
def is_reachable (building : Building) (target : ℕ) (max_moves : ℕ) : Prop :=
  ∃ (moves : List ℤ),
    moves.length ≤ max_moves ∧
    moves.all (λ m => m.natAbs ∈ building.elevator_moves) ∧
    (moves.foldl elevator_trip building.start_floor : ℤ) = target

-- Theorem statement
theorem cannot_visit_all_friends :
  ¬∃ (moves : List ℤ),
    moves.length ≤ 6 ∧
    moves.all (λ m => m.natAbs ∈ problem.elevator_moves) ∧
    (∀ floor ∈ problem.friend_floors,
      ∃ (submoves : List ℤ),
        submoves ⊆ moves ∧
        (submoves.foldl elevator_trip problem.start_floor : ℤ) = floor) :=
sorry

end cannot_visit_all_friends_l3589_358994


namespace max_sum_of_abs_on_unit_sphere_l3589_358957

theorem max_sum_of_abs_on_unit_sphere :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |x| + |y| + |z| ≤ M) ∧
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 1 ∧ |x| + |y| + |z| = M) := by
  sorry

end max_sum_of_abs_on_unit_sphere_l3589_358957


namespace olivia_initial_money_l3589_358996

/-- The amount of money Olivia and Nigel spent on tickets -/
def ticket_cost : ℕ := 6 * 28

/-- The amount of money Nigel had initially -/
def nigel_money : ℕ := 139

/-- The amount of money Olivia and Nigel have left after buying tickets -/
def remaining_money : ℕ := 83

/-- The amount of money Olivia had initially -/
def olivia_money : ℕ := (ticket_cost + remaining_money) - nigel_money

theorem olivia_initial_money : olivia_money = 112 := by
  sorry

end olivia_initial_money_l3589_358996


namespace complex_product_pure_imaginary_l3589_358979

theorem complex_product_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := 1 - 2 * Complex.I
  let z₂ : ℂ := a + Complex.I
  (∃ (b : ℝ), z₁ * z₂ = b * Complex.I ∧ b ≠ 0) → a = -2 := by
  sorry

end complex_product_pure_imaginary_l3589_358979


namespace count_D_two_eq_30_l3589_358950

/-- The number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- The count of positive integers n ≤ 127 for which D(n) = 2 -/
def count_D_two : ℕ := sorry

theorem count_D_two_eq_30 : count_D_two = 30 := by sorry

end count_D_two_eq_30_l3589_358950


namespace sin_150_degrees_l3589_358917

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l3589_358917


namespace condition_necessary_not_sufficient_l3589_358909

open Set Real

theorem condition_necessary_not_sufficient :
  let A : Set ℝ := {x | 0 < x ∧ x < 3}
  let B : Set ℝ := {x | log (x - 2) < 0}
  B ⊂ A ∧ B ≠ A := by
  sorry

end condition_necessary_not_sufficient_l3589_358909


namespace hyperbola_a_value_l3589_358954

/-- The value of a for a hyperbola with given properties -/
def hyperbola_a : ℝ → Prop := λ a =>
  a > 0 ∧
  ∃ (x y : ℝ), x^2 / a^2 - y^2 / 4 = 1 ∧
  (y = 2 * x / a ∨ y = -2 * x / a) ∧
  x = 2 ∧ y = 1

/-- Theorem: The value of a for the given hyperbola is 4 -/
theorem hyperbola_a_value : ∃ (a : ℝ), hyperbola_a a ∧ a = 4 := by
  sorry

end hyperbola_a_value_l3589_358954


namespace alice_painted_cuboids_l3589_358951

/-- The number of cuboids Alice painted -/
def num_cuboids : ℕ := 6

/-- The number of faces on each cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of faces painted -/
def total_faces_painted : ℕ := 36

theorem alice_painted_cuboids :
  num_cuboids * faces_per_cuboid = total_faces_painted :=
by sorry

end alice_painted_cuboids_l3589_358951


namespace inequality_solution_l3589_358913

theorem inequality_solution (m : ℝ) (x : ℝ) :
  (m * x - 2 ≥ 3 * x - 4 * m) ↔
  (m > 3 ∧ x ≥ (2 - 4*m) / (m - 3)) ∨
  (m < 3 ∧ x ≤ (2 - 4*m) / (m - 3)) ∨
  (m = 3) := by
  sorry

end inequality_solution_l3589_358913


namespace solution_set_implies_sum_l3589_358955

theorem solution_set_implies_sum (a b : ℝ) : 
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) → 
  a + b = -1 := by
sorry

end solution_set_implies_sum_l3589_358955


namespace no_fermat_solutions_with_constraints_l3589_358993

theorem no_fermat_solutions_with_constraints (n : ℕ) (hn : n > 1) :
  ¬∃ (x y z : ℕ), x^n + y^n = z^n ∧ x ≤ n ∧ y ≤ n := by
  sorry

end no_fermat_solutions_with_constraints_l3589_358993


namespace number_operations_l3589_358995

theorem number_operations (x : ℝ) : ((x + 5) * 5 - 5) / 5 = 5 ↔ x = 1 := by
  sorry

end number_operations_l3589_358995


namespace circle_and_line_intersection_l3589_358971

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

-- Define the line l
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Define the points E and F
def E : ℝ × ℝ := (1, -3)
def F : ℝ × ℝ := (0, 4)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- Theorem statement
theorem circle_and_line_intersection :
  ∃ (A B : ℝ × ℝ) (C2 : ℝ → ℝ → Prop),
    (∀ x y, C1 x y ∧ l x y ↔ (x, y) = A ∨ (x, y) = B) ∧
    (C2 E.1 E.2 ∧ C2 F.1 F.2) ∧
    (∃ D E F, ∀ x y, C2 x y ↔ x^2 + y^2 + D*x + E*y + F = 0) ∧
    (∃ k, ∀ x y, (C1 x y ∧ C2 x y) → (∃ c, x + k*y = c ∧ ∀ x' y', parallel_line x' y' → ∃ c', x' + k*y' = c')) →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 / 5 ∧
    (∀ x y, C2 x y ↔ x^2 + y^2 + 6*x - 16 = 0) :=
by sorry

end circle_and_line_intersection_l3589_358971


namespace proportional_function_two_quadrants_l3589_358936

/-- A proportional function passing through two quadrants -/
theorem proportional_function_two_quadrants (m : ℝ) : 
  let f : ℝ → ℝ := λ x => (m + 3) * x^(m^2 + m - 5)
  m = 2 → (∃ x y, f x = y ∧ x > 0 ∧ y > 0) ∧ 
          (∃ x y, f x = y ∧ x < 0 ∧ y < 0) ∧
          (∀ x y, f x = y → (x ≥ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≤ 0)) :=
by sorry


end proportional_function_two_quadrants_l3589_358936


namespace jack_morning_emails_l3589_358939

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 8

/-- The total number of emails Jack received in the morning and evening -/
def total_morning_evening : ℕ := 11

/-- Theorem stating that Jack received 3 emails in the morning -/
theorem jack_morning_emails :
  morning_emails = 3 :=
by sorry

end jack_morning_emails_l3589_358939


namespace jungkook_english_score_l3589_358901

/-- Jungkook's average score in Korean, math, and science -/
def initial_average : ℝ := 92

/-- The increase in average score after taking the English test -/
def average_increase : ℝ := 2

/-- The number of subjects before taking the English test -/
def initial_subjects : ℕ := 3

/-- The number of subjects after taking the English test -/
def total_subjects : ℕ := 4

/-- Jungkook's English score -/
def english_score : ℝ := total_subjects * (initial_average + average_increase) - initial_subjects * initial_average

theorem jungkook_english_score : english_score = 100 := by
  sorry

end jungkook_english_score_l3589_358901


namespace expression_evaluation_l3589_358985

theorem expression_evaluation : 86 + (144 / 12) + (15 * 13) - 300 - (480 / 8) = -67 := by
  sorry

end expression_evaluation_l3589_358985


namespace museum_paintings_ratio_l3589_358900

theorem museum_paintings_ratio (total_paintings portraits : ℕ) 
  (h1 : total_paintings = 80)
  (h2 : portraits = 16)
  (h3 : ∃ k : ℕ, total_paintings - portraits = k * portraits) :
  (total_paintings - portraits) / portraits = 4 := by
  sorry

end museum_paintings_ratio_l3589_358900


namespace temperature_function_properties_l3589_358960

-- Define the temperature function
def T (t : ℝ) : ℝ := t^3 - 3*t + 60

-- Define the theorem
theorem temperature_function_properties :
  -- Conditions
  (T (-4) = 8) ∧
  (T 0 = 60) ∧
  (T 1 = 58) ∧
  (deriv T (-4) = deriv T 4) ∧
  -- Conclusions
  (∀ t ∈ Set.Icc (-2) 2, T t ≤ 62) ∧
  (T (-1) = 62) ∧
  (T 2 = 62) :=
by sorry

end temperature_function_properties_l3589_358960


namespace horatio_sonnets_l3589_358986

def sonnet_lines : ℕ := 16
def sonnets_read : ℕ := 9
def unread_lines : ℕ := 126

theorem horatio_sonnets :
  ∃ (total_sonnets : ℕ),
    total_sonnets * sonnet_lines = sonnets_read * sonnet_lines + unread_lines ∧
    total_sonnets = 16 := by
  sorry

end horatio_sonnets_l3589_358986


namespace ratio_value_l3589_358905

theorem ratio_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 2 * c) 
  (h3 : c = 5 * d) 
  (h4 : b ≠ 0) 
  (h5 : d ≠ 0) : 
  (a * c) / (b * d) = 20 := by
sorry

end ratio_value_l3589_358905


namespace greatest_b_value_l3589_358948

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 15 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 8*5 - 15 ≥ 0) := by
  sorry

end greatest_b_value_l3589_358948


namespace fraction_equality_l3589_358935

theorem fraction_equality (x : ℚ) : (1/5)^35 * x^18 = 1/(2*(10)^35) → x = 1/4 := by
  sorry

end fraction_equality_l3589_358935


namespace angle_ratio_equality_l3589_358975

-- Define a triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define a point P inside the triangle
def PointInside (t : Triangle) (P : Point) : Prop := sorry

-- Define angle measure
def AngleMeasure (A B C : Point) : ℝ := sorry

-- Theorem statement
theorem angle_ratio_equality (t : Triangle) (P : Point) (x : ℝ) 
  (h_inside : PointInside t P)
  (h_ratio_AB_AC : AngleMeasure t.A P t.B / AngleMeasure t.A P t.C = x)
  (h_ratio_CA_CB : AngleMeasure t.C P t.A / AngleMeasure t.C P t.B = x)
  (h_ratio_BC_BA : AngleMeasure t.B P t.C / AngleMeasure t.B P t.A = x) :
  x = 1 := by
  sorry

end angle_ratio_equality_l3589_358975


namespace solution_set_contains_two_and_zero_l3589_358959

/-- The solution set of the inequality (1+k²)x ≤ k⁴+4 with respect to x -/
def M (k : ℝ) : Set ℝ :=
  {x : ℝ | (1 + k^2) * x ≤ k^4 + 4}

/-- For any real constant k, both 2 and 0 are in the solution set M -/
theorem solution_set_contains_two_and_zero :
  ∀ k : ℝ, (2 ∈ M k) ∧ (0 ∈ M k) :=
by sorry

end solution_set_contains_two_and_zero_l3589_358959


namespace percentage_of_male_employees_l3589_358992

theorem percentage_of_male_employees (total_employees : ℕ) 
  (males_below_50 : ℕ) (h1 : total_employees = 1800) 
  (h2 : males_below_50 = 756) : 
  (males_below_50 : ℝ) / (0.7 * total_employees) = 0.6 := by
  sorry

end percentage_of_male_employees_l3589_358992


namespace polynomial_equation_properties_l3589_358965

theorem polynomial_equation_properties (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (2*x + 1)^4 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4) →
  (a₀ = 1 ∧ a₃ = -32 ∧ a₄ = 16 ∧ a₁ + a₂ + a₃ + a₄ = 0) :=
by sorry

end polynomial_equation_properties_l3589_358965


namespace roots_of_quadratic_l3589_358918

-- Define the variables and conditions
variable (x y : ℝ)
variable (h1 : 2 * x + 3 * y = 18)
variable (h2 : x * y = 8)

-- Define the quadratic polynomial
def f (t : ℝ) := t^2 - 18*t + 8

-- State the theorem
theorem roots_of_quadratic :
  f x = 0 ∧ f y = 0 := by sorry

end roots_of_quadratic_l3589_358918


namespace quotient_of_composites_l3589_358946

def first_five_even_composites : List Nat := [4, 6, 8, 10, 12]
def next_five_odd_composites : List Nat := [9, 15, 21, 25, 27]

def product_even : Nat := first_five_even_composites.prod
def product_odd : Nat := next_five_odd_composites.prod

theorem quotient_of_composites :
  (product_even : ℚ) / (product_odd : ℚ) = 512 / 28525 := by
  sorry

end quotient_of_composites_l3589_358946


namespace shrink_ray_effect_l3589_358966

/-- Represents the shrink ray's effect on volume -/
def shrink_factor : ℝ := 0.5

/-- The number of coffee cups -/
def num_cups : ℕ := 5

/-- The initial volume of coffee in each cup (in ounces) -/
def initial_volume : ℝ := 8

/-- Calculates the total volume of coffee after shrinking -/
def total_volume_after_shrink : ℝ := num_cups * (initial_volume * shrink_factor)

theorem shrink_ray_effect :
  total_volume_after_shrink = 20 := by sorry

end shrink_ray_effect_l3589_358966


namespace g_100_eq_520_l3589_358953

/-- The sum of greatest common divisors function -/
def g (n : ℕ+) : ℕ := (Finset.range n.val.succ).sum (fun k => Nat.gcd k n.val)

/-- The theorem stating that g(100) = 520 -/
theorem g_100_eq_520 : g 100 = 520 := by sorry

end g_100_eq_520_l3589_358953


namespace tire_circumference_l3589_358912

/-- The circumference of a tire given its rotation speed and the car's velocity -/
theorem tire_circumference (revolutions_per_minute : ℝ) (car_speed_kmh : ℝ) :
  revolutions_per_minute = 400 →
  car_speed_kmh = 48 →
  (car_speed_kmh * 1000 / 60) / revolutions_per_minute = 2 :=
by sorry

end tire_circumference_l3589_358912


namespace remainder_theorem_l3589_358981

theorem remainder_theorem : (43^43 + 43) % 44 = 42 := by
  sorry

end remainder_theorem_l3589_358981


namespace reflection_line_sum_l3589_358944

/-- Given a line y = mx + b passing through (0, 3) and reflecting (2, -4) to (-4, 8), prove that m + b = 3.5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (3 = m * 0 + b) →  -- Line passes through (0, 3)
  (let midpoint_x := (2 + (-4)) / 2
   let midpoint_y := (-4 + 8) / 2
   (midpoint_y - 3) = m * (midpoint_x - 0)) →  -- Midpoint lies on the line
  (8 - (-4)) / (-4 - 2) = -1 / m →  -- Perpendicular slopes
  m + b = 3.5 := by
sorry

end reflection_line_sum_l3589_358944


namespace third_quarter_gdp_l3589_358997

/-- Represents the GDP growth over quarters -/
def gdp_growth (initial_gdp : ℝ) (growth_rate : ℝ) (quarters : ℕ) : ℝ :=
  initial_gdp * (1 + growth_rate) ^ quarters

theorem third_quarter_gdp 
  (initial_gdp : ℝ) 
  (growth_rate : ℝ) :
  gdp_growth initial_gdp growth_rate 2 = initial_gdp * (1 + growth_rate)^2 :=
by sorry

end third_quarter_gdp_l3589_358997


namespace kids_played_monday_l3589_358963

theorem kids_played_monday (total : ℕ) (tuesday : ℕ) (h1 : total = 16) (h2 : tuesday = 14) :
  total - tuesday = 2 := by
  sorry

end kids_played_monday_l3589_358963


namespace arithmetic_sequence_nth_term_l3589_358919

/-- Given an arithmetic sequence {a_n} with a₁ = 1, d = 2, and aₙ = 19, prove that n = 10 -/
theorem arithmetic_sequence_nth_term (a : ℕ → ℝ) (n : ℕ) : 
  (∀ k, a (k + 1) - a k = 2) →  -- common difference is 2
  a 1 = 1 →                     -- first term is 1
  a n = 19 →                    -- n-th term is 19
  n = 10 := by
sorry

end arithmetic_sequence_nth_term_l3589_358919


namespace greatest_difference_of_units_digit_l3589_358968

theorem greatest_difference_of_units_digit (x : ℕ) : 
  (x < 10) →
  (637 * 10 + x) % 3 = 0 →
  ∃ y z, y < 10 ∧ z < 10 ∧ 
         (637 * 10 + y) % 3 = 0 ∧ 
         (637 * 10 + z) % 3 = 0 ∧ 
         y - z ≤ 6 ∧
         ∀ w, w < 10 → (637 * 10 + w) % 3 = 0 → y - w ≤ 6 ∧ w - z ≤ 6 :=
by sorry

end greatest_difference_of_units_digit_l3589_358968


namespace smallest_multiple_of_one_to_five_l3589_358927

theorem smallest_multiple_of_one_to_five : ∃ (n : ℕ), n > 0 ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → i ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → i ∣ m) → n ≤ m) ∧ n = 60 := by
  sorry

end smallest_multiple_of_one_to_five_l3589_358927


namespace max_distance_between_points_l3589_358969

/-- Given vector OA = (1, -1) and |OA| = |OB|, the maximum value of |AB| is 2√2. -/
theorem max_distance_between_points (OA OB : ℝ × ℝ) : 
  OA = (1, -1) → 
  Real.sqrt ((OA.1 ^ 2) + (OA.2 ^ 2)) = Real.sqrt ((OB.1 ^ 2) + (OB.2 ^ 2)) →
  (∃ (AB : ℝ × ℝ), AB = OB - OA ∧ 
    Real.sqrt ((AB.1 ^ 2) + (AB.2 ^ 2)) ≤ 2 * Real.sqrt 2 ∧
    ∃ (OB' : ℝ × ℝ), Real.sqrt ((OB'.1 ^ 2) + (OB'.2 ^ 2)) = Real.sqrt ((OA.1 ^ 2) + (OA.2 ^ 2)) ∧
      let AB' := OB' - OA
      Real.sqrt ((AB'.1 ^ 2) + (AB'.2 ^ 2)) = 2 * Real.sqrt 2) :=
by sorry

end max_distance_between_points_l3589_358969


namespace miles_owns_seventeen_instruments_l3589_358942

/-- Represents the number of musical instruments Miles owns --/
structure MilesInstruments where
  fingers : ℕ
  hands : ℕ
  heads : ℕ
  trumpets : ℕ
  guitars : ℕ
  trombones : ℕ
  frenchHorns : ℕ

/-- The total number of musical instruments Miles owns --/
def totalInstruments (m : MilesInstruments) : ℕ :=
  m.trumpets + m.guitars + m.trombones + m.frenchHorns

/-- Theorem stating that Miles owns 17 musical instruments --/
theorem miles_owns_seventeen_instruments (m : MilesInstruments)
  (h1 : m.fingers = 10)
  (h2 : m.hands = 2)
  (h3 : m.heads = 1)
  (h4 : m.trumpets = m.fingers - 3)
  (h5 : m.guitars = m.hands + 2)
  (h6 : m.trombones = m.heads + 2)
  (h7 : m.frenchHorns = m.guitars - 1) :
  totalInstruments m = 17 := by
  sorry

#check miles_owns_seventeen_instruments

end miles_owns_seventeen_instruments_l3589_358942


namespace emily_beads_count_l3589_358973

theorem emily_beads_count (necklaces : ℕ) (beads_per_necklace : ℕ) 
  (h1 : necklaces = 11) (h2 : beads_per_necklace = 28) : 
  necklaces * beads_per_necklace = 308 := by
  sorry

end emily_beads_count_l3589_358973


namespace gross_revenue_increase_l3589_358930

theorem gross_revenue_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_reduction_rate : ℝ)
  (quantity_increase_rate : ℝ)
  (h1 : price_reduction_rate = 0.2)
  (h2 : quantity_increase_rate = 0.6)
  : (((1 - price_reduction_rate) * (1 + quantity_increase_rate) - 1) * 100 : ℝ) = 28 := by
  sorry

end gross_revenue_increase_l3589_358930


namespace carpet_dimensions_l3589_358938

/-- Represents a rectangular room --/
structure Room where
  length : ℝ
  width : ℝ

/-- Represents a rectangular carpet --/
structure Carpet where
  length : ℝ
  width : ℝ

/-- Checks if a carpet fits in a room such that each corner touches a different wall --/
def fits_in_room (c : Carpet) (r : Room) : Prop :=
  ∃ (α b : ℝ),
    α + (c.length / c.width) * b = r.length ∧
    (c.length / c.width) * α + b = r.width ∧
    c.width^2 = α^2 + b^2

/-- The main theorem to prove --/
theorem carpet_dimensions :
  ∃ (c : Carpet),
    fits_in_room c { length := 38, width := 55 } ∧
    fits_in_room c { length := 50, width := 55 } ∧
    c.length = 50 ∧
    c.width = 25 := by
  sorry

end carpet_dimensions_l3589_358938


namespace quadratic_point_m_value_l3589_358904

theorem quadratic_point_m_value (a m : ℝ) : 
  a > 0 → 
  m ≠ 0 → 
  3 = -a * m^2 + 2 * a * m + 3 → 
  m = 2 := by
sorry

end quadratic_point_m_value_l3589_358904


namespace desired_circle_satisfies_conditions_l3589_358976

/-- The equation of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

/-- The equation of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- The equation of the line on which the center of the desired circle lies -/
def centerLine (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

/-- The equation of the desired circle -/
def desiredCircle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 11 = 0

/-- Theorem stating that the desired circle satisfies all conditions -/
theorem desired_circle_satisfies_conditions :
  ∀ (x y : ℝ),
    (circle1 x y ∧ circle2 x y → desiredCircle x y) ∧
    (∃ (h k : ℝ), centerLine h k ∧ 
      ∀ (x y : ℝ), desiredCircle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 11)) :=
sorry

end desired_circle_satisfies_conditions_l3589_358976


namespace rectangle_perimeter_and_area_l3589_358999

theorem rectangle_perimeter_and_area :
  ∀ (length width perimeter area : ℝ),
    length = 10 →
    width = length - 3 →
    perimeter = 2 * (length + width) →
    area = length * width →
    perimeter = 34 ∧ area = 70 := by
  sorry

end rectangle_perimeter_and_area_l3589_358999


namespace smallest_white_buttons_l3589_358926

theorem smallest_white_buttons (n : ℕ) (h1 : n % 10 = 0) : 
  (n / 2 : ℚ) + (n / 5 : ℚ) + 8 ≤ n → 
  (∃ m : ℕ, m ≥ 1 ∧ (n : ℚ) - ((n / 2 : ℚ) + (n / 5 : ℚ) + 8) = m) →
  (∃ k : ℕ, k ≥ 1 ∧ (30 : ℚ) - ((30 / 2 : ℚ) + (30 / 5 : ℚ) + 8) = k) :=
by sorry

end smallest_white_buttons_l3589_358926


namespace exponential_equation_solution_l3589_358941

theorem exponential_equation_solution : 
  ∃ x : ℝ, (3 : ℝ)^x * 9^x = 27^(x - 20) ∧ x = 20 := by
  sorry

end exponential_equation_solution_l3589_358941


namespace parabola_properties_l3589_358987

/-- Given a parabola with equation x² = 8y, this theorem proves the equation of its directrix
    and the coordinates of its focus. -/
theorem parabola_properties (x y : ℝ) :
  x^2 = 8*y →
  (∃ (directrix : ℝ → Prop) (focus : ℝ × ℝ),
    directrix = λ y' => y' = -2 ∧
    focus = (0, 2)) :=
by sorry

end parabola_properties_l3589_358987


namespace system_equation_solution_range_l3589_358924

theorem system_equation_solution_range (x y m : ℝ) : 
  (3 * x + y = m - 1) → 
  (x - 3 * y = 2 * m) → 
  (x + 2 * y ≥ 0) → 
  (m ≤ -1) := by
sorry

end system_equation_solution_range_l3589_358924


namespace constant_distance_special_points_min_distance_to_origin_euclidean_vs_orthogonal_distance_l3589_358977

-- Define orthogonal distance
def orthogonal_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Proposition 1
theorem constant_distance_special_points :
  ∀ α : ℝ, orthogonal_distance 2 3 (Real.sin α ^ 2) (Real.cos α ^ 2) = 4 :=
sorry

-- Proposition 2 (negation)
theorem min_distance_to_origin :
  ∃ x y : ℝ, x - y + 1 = 0 ∧ |x| + |y| < 1 :=
sorry

-- Proposition 3
theorem euclidean_vs_orthogonal_distance :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ (Real.sqrt 2 / 2) * (|x₁ - x₂| + |y₁ - y₂|) :=
sorry

end constant_distance_special_points_min_distance_to_origin_euclidean_vs_orthogonal_distance_l3589_358977


namespace sum_of_real_solutions_l3589_358978

theorem sum_of_real_solutions (b : ℝ) (h : b > 0) :
  ∃ x : ℝ, x ≥ 0 ∧ Real.sqrt (b - Real.sqrt (b + 2*x)) = x ∧
  (∀ y : ℝ, y ≥ 0 ∧ Real.sqrt (b - Real.sqrt (b + 2*y)) = y → y = x) ∧
  x = Real.sqrt (b - 1) - 1 :=
sorry

end sum_of_real_solutions_l3589_358978


namespace initial_bedbug_count_l3589_358902

/-- The number of bedbugs after n days, given an initial population -/
def bedbug_population (initial : ℕ) (days : ℕ) : ℕ :=
  initial * (3 ^ days)

/-- Theorem: If the number of bedbugs triples every day and there are 810 bedbugs after four days, 
    then the initial number of bedbugs was 30. -/
theorem initial_bedbug_count : bedbug_population 30 4 = 810 := by
  sorry

#check initial_bedbug_count

end initial_bedbug_count_l3589_358902


namespace quadratic_form_k_value_l3589_358932

theorem quadratic_form_k_value (a h k : ℚ) :
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) →
  k = -49/4 := by
  sorry

end quadratic_form_k_value_l3589_358932


namespace student_pairs_l3589_358961

theorem student_pairs (n : ℕ) (h : n = 12) : (n * (n - 1)) / 2 = 66 := by
  sorry

end student_pairs_l3589_358961


namespace residue_mod_12_l3589_358910

theorem residue_mod_12 : (172 * 15 - 13 * 8 + 6) % 12 = 10 := by sorry

end residue_mod_12_l3589_358910


namespace monday_count_l3589_358989

/-- The number of Mondays on which it rained -/
def num_mondays : ℕ := sorry

/-- The rainfall on each Monday in centimeters -/
def rainfall_per_monday : ℚ := 3/2

/-- The number of Tuesdays on which it rained -/
def num_tuesdays : ℕ := 9

/-- The rainfall on each Tuesday in centimeters -/
def rainfall_per_tuesday : ℚ := 5/2

/-- The difference in total rainfall between Tuesdays and Mondays in centimeters -/
def rainfall_difference : ℚ := 12

theorem monday_count : 
  num_mondays * rainfall_per_monday + rainfall_difference = 
  num_tuesdays * rainfall_per_tuesday ∧ num_mondays = 7 := by sorry

end monday_count_l3589_358989


namespace perfect_square_factors_of_10080_l3589_358937

/-- Given that 10080 = 2^4 * 3^2 * 5 * 7, this function counts the number of positive integer factors of 10080 that are perfect squares. -/
def count_perfect_square_factors : ℕ :=
  let prime_factorization : List (ℕ × ℕ) := [(2, 4), (3, 2), (5, 1), (7, 1)]
  -- Function implementation
  sorry

/-- The number of positive integer factors of 10080 that are perfect squares is 6. -/
theorem perfect_square_factors_of_10080 : count_perfect_square_factors = 6 := by
  sorry

end perfect_square_factors_of_10080_l3589_358937


namespace rotate_point_A_about_C_l3589_358998

-- Define the rotation function
def rotate90ClockwiseAboutPoint (p center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (cx, cy) := center
  (cx + (y - cy), cy - (x - cx))

-- Theorem statement
theorem rotate_point_A_about_C :
  let A : ℝ × ℝ := (-3, 2)
  let C : ℝ × ℝ := (-2, 2)
  rotate90ClockwiseAboutPoint A C = (-2, 3) := by sorry

end rotate_point_A_about_C_l3589_358998


namespace smallest_prime_after_five_nonprimes_l3589_358962

/-- A function that returns true if a natural number is prime, false otherwise -/
def is_prime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nth_prime (n : ℕ) : ℕ := sorry

/-- A function that returns true if there are at least five consecutive nonprime numbers before n, false otherwise -/
def five_consecutive_nonprimes_before (n : ℕ) : Prop := sorry

theorem smallest_prime_after_five_nonprimes : 
  ∃ (n : ℕ), is_prime n ∧ five_consecutive_nonprimes_before n ∧ 
  ∀ (m : ℕ), m < n → ¬(is_prime m ∧ five_consecutive_nonprimes_before m) :=
sorry

end smallest_prime_after_five_nonprimes_l3589_358962


namespace f_derivative_l3589_358983

noncomputable def f (x : ℝ) : ℝ := x * Real.cos (2 * x)

theorem f_derivative : 
  deriv f = fun x => Real.cos (2 * x) - 2 * x * Real.sin (2 * x) := by
  sorry

end f_derivative_l3589_358983


namespace green_hats_count_l3589_358949

theorem green_hats_count (total_hats : ℕ) (blue_cost green_cost total_cost : ℚ) :
  total_hats = 85 →
  blue_cost = 6 →
  green_cost = 7 →
  total_cost = 540 →
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_cost * blue_hats + green_cost * green_hats = total_cost ∧
    green_hats = 30 :=
by
  sorry

end green_hats_count_l3589_358949


namespace special_numbers_exist_l3589_358972

theorem special_numbers_exist : ∃ (a b c d e : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
  (¬(3 ∣ a) ∧ ¬(4 ∣ a) ∧ ¬(5 ∣ a)) ∧
  (¬(3 ∣ b) ∧ ¬(4 ∣ b) ∧ ¬(5 ∣ b)) ∧
  (¬(3 ∣ c) ∧ ¬(4 ∣ c) ∧ ¬(5 ∣ c)) ∧
  (¬(3 ∣ d) ∧ ¬(4 ∣ d) ∧ ¬(5 ∣ d)) ∧
  (¬(3 ∣ e) ∧ ¬(4 ∣ e) ∧ ¬(5 ∣ e)) ∧
  (3 ∣ (a + b + c)) ∧ (3 ∣ (a + b + d)) ∧ (3 ∣ (a + b + e)) ∧
  (3 ∣ (a + c + d)) ∧ (3 ∣ (a + c + e)) ∧ (3 ∣ (a + d + e)) ∧
  (3 ∣ (b + c + d)) ∧ (3 ∣ (b + c + e)) ∧ (3 ∣ (b + d + e)) ∧
  (3 ∣ (c + d + e)) ∧
  (4 ∣ (a + b + c + d)) ∧ (4 ∣ (a + b + c + e)) ∧
  (4 ∣ (a + b + d + e)) ∧ (4 ∣ (a + c + d + e)) ∧
  (4 ∣ (b + c + d + e)) ∧
  (5 ∣ (a + b + c + d + e)) := by
sorry

end special_numbers_exist_l3589_358972


namespace problem_statement_l3589_358903

theorem problem_statement (a : ℝ) (h : (a + 1/a)^3 = 4) :
  a^4 + 1/a^4 = -158/81 := by sorry

end problem_statement_l3589_358903


namespace elena_earnings_l3589_358915

def charging_sequence : List Nat := [3, 4, 5, 6, 7]

def calculate_earnings (hours : Nat) : Nat :=
  let complete_cycles := hours / 5
  let remaining_hours := hours % 5
  let cycle_earnings := charging_sequence.sum * complete_cycles
  let remaining_earnings := (charging_sequence.take remaining_hours).sum
  cycle_earnings + remaining_earnings

theorem elena_earnings :
  calculate_earnings 47 = 232 := by
  sorry

end elena_earnings_l3589_358915


namespace decreasing_interval_of_f_l3589_358933

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f(x)
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem statement
theorem decreasing_interval_of_f :
  ∀ x : ℝ, (x ∈ Set.Ioo (-1) 1) ↔ (f_deriv x < 0) :=
sorry

end decreasing_interval_of_f_l3589_358933


namespace house_ratio_l3589_358980

theorem house_ratio (houses_one_side : ℕ) (total_houses : ℕ) : 
  houses_one_side = 40 → 
  total_houses = 160 → 
  (total_houses - houses_one_side) / houses_one_side = 3 := by
sorry

end house_ratio_l3589_358980


namespace quadrilateral_inequality_l3589_358914

theorem quadrilateral_inequality (a b c : ℝ) : 
  (a > 0) →  -- EF has positive length
  (b > 0) →  -- EG has positive length
  (c > 0) →  -- EH has positive length
  (a < b) →  -- F is between E and G
  (b < c) →  -- G is between E and H
  (2 * b > c) →  -- Condition for positive area after rotation
  (a < c / 3) :=
sorry

end quadrilateral_inequality_l3589_358914


namespace f_properties_l3589_358952

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * abs (x + a) - (1/2) * Real.log x

theorem f_properties :
  (∀ x > 0, ∀ a : ℝ,
    (a = 0 → (∀ y > (1/2), f a y > f a x) ∧ (∀ z ∈ Set.Ioo 0 (1/2), f a z < f a x)) ∧
    (a < 0 →
      (a < -2 → ∃ x₁ x₂, x₁ = (-a - Real.sqrt (a^2 - 4)) / 4 ∧
                         x₂ = (-a + Real.sqrt (a^2 - 4)) / 4 ∧
                         (∀ y ≠ x₁, f a y ≥ f a x₁) ∧
                         (∀ y ≠ x₂, f a y ≤ f a x₂)) ∧
      (-2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2 → ∀ y > 0, f a y ≠ f a x) ∧
      (-Real.sqrt 2 / 2 < a ∧ a < 0 →
        ∃ x₃, x₃ = (-a + Real.sqrt (a^2 + 4)) / 4 ∧
               (∀ y ≠ x₃, f a y ≥ f a x₃)))) :=
by sorry

end f_properties_l3589_358952


namespace brothers_money_l3589_358908

theorem brothers_money (a₁ a₂ a₃ a₄ : ℚ) :
  a₁ + a₂ + a₃ + a₄ = 48 ∧
  a₁ + 3 = a₂ - 3 ∧
  a₁ + 3 = 3 * a₃ ∧
  a₁ + 3 = a₄ / 3 →
  a₁ = 6 ∧ a₂ = 12 ∧ a₃ = 3 ∧ a₄ = 27 := by
sorry

end brothers_money_l3589_358908
