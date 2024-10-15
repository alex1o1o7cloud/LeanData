import Mathlib

namespace NUMINAMATH_CALUDE_function_inequality_l657_65766

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h1 : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) : 
  Real.exp a * f 0 < f a := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l657_65766


namespace NUMINAMATH_CALUDE_solve_equation_l657_65720

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 5 / 3 → x = -27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l657_65720


namespace NUMINAMATH_CALUDE_cube_root_eight_over_sqrt_two_equals_sqrt_two_l657_65727

theorem cube_root_eight_over_sqrt_two_equals_sqrt_two : 
  (8 : ℝ)^(1/3) / (2 : ℝ)^(1/2) = (2 : ℝ)^(1/2) := by sorry

end NUMINAMATH_CALUDE_cube_root_eight_over_sqrt_two_equals_sqrt_two_l657_65727


namespace NUMINAMATH_CALUDE_cube_volume_to_surface_area_l657_65721

theorem cube_volume_to_surface_area :
  ∀ (s : ℝ), s > 0 → s^3 = 729 → 6 * s^2 = 486 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_to_surface_area_l657_65721


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l657_65734

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 + 2*a*x + a > 0) ↔ a > -1/3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l657_65734


namespace NUMINAMATH_CALUDE_theater_ticket_cost_l657_65743

/-- The cost of theater tickets for a group -/
def theater_cost (adult_price : ℚ) : ℚ :=
  let child_price := adult_price / 2
  let total_price := 10 * adult_price + 8 * child_price
  total_price * (1 - 1/10)  -- 10% discount applied

theorem theater_ticket_cost :
  ∃ (adult_price : ℚ),
    8 * adult_price + 7 * (adult_price / 2) = 42 ∧
    theater_cost adult_price = 46 :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_cost_l657_65743


namespace NUMINAMATH_CALUDE_bakery_sugar_amount_l657_65785

/-- Given the ratios of ingredients in a bakery storage room, 
    prove the amount of sugar stored. -/
theorem bakery_sugar_amount 
  (sugar flour baking_soda : ℕ) 
  (h1 : sugar = flour)  -- sugar to flour ratio is 5:5, which simplifies to 1:1
  (h2 : flour = 10 * baking_soda)  -- flour to baking soda ratio is 10:1
  (h3 : flour = 8 * (baking_soda + 60))  -- if 60 more pounds of baking soda, ratio would be 8:1
  : sugar = 2400 := by
  sorry

end NUMINAMATH_CALUDE_bakery_sugar_amount_l657_65785


namespace NUMINAMATH_CALUDE_inequality_proof_l657_65716

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l657_65716


namespace NUMINAMATH_CALUDE_exam_score_problem_l657_65708

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 60)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 110) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 34 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l657_65708


namespace NUMINAMATH_CALUDE_candy_store_spending_l657_65794

/-- Proves that given a weekly allowance of $2.25, after spending 3/5 of it at the arcade
    and 1/3 of the remainder at the toy store, the amount left for the candy store is $0.60. -/
theorem candy_store_spending (weekly_allowance : ℚ) (h1 : weekly_allowance = 2.25) :
  let arcade_spending := (3 / 5) * weekly_allowance
  let remaining_after_arcade := weekly_allowance - arcade_spending
  let toy_store_spending := (1 / 3) * remaining_after_arcade
  let candy_store_spending := remaining_after_arcade - toy_store_spending
  candy_store_spending = 0.60 := by
  sorry


end NUMINAMATH_CALUDE_candy_store_spending_l657_65794


namespace NUMINAMATH_CALUDE_megan_homework_problems_l657_65706

/-- The total number of homework problems Megan had -/
def total_problems (finished : ℕ) (pages_left : ℕ) (problems_per_page : ℕ) : ℕ :=
  finished + pages_left * problems_per_page

/-- Proof that Megan had 40 homework problems in total -/
theorem megan_homework_problems :
  total_problems 26 2 7 = 40 := by
  sorry

end NUMINAMATH_CALUDE_megan_homework_problems_l657_65706


namespace NUMINAMATH_CALUDE_water_depth_in_tank_l657_65792

/-- Represents a horizontally placed cylindrical water tank -/
structure CylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the possible depths of water in a cylindrical tank -/
def water_depths (tank : CylindricalTank) (water_surface_area : ℝ) : Set ℝ :=
  sorry

/-- Theorem stating the depths of water in the given cylindrical tank -/
theorem water_depth_in_tank (tank : CylindricalTank) 
  (h1 : tank.length = 12)
  (h2 : tank.diameter = 8)
  (h3 : water_surface_area = 48) :
  water_depths tank water_surface_area = {4 - 2 * Real.sqrt 3, 4 + 2 * Real.sqrt 3} := by
  sorry

end NUMINAMATH_CALUDE_water_depth_in_tank_l657_65792


namespace NUMINAMATH_CALUDE_prime_power_sum_existence_l657_65770

theorem prime_power_sum_existence (p : Finset Nat) (h_prime : ∀ q ∈ p, Nat.Prime q) :
  ∃ x : Nat,
    (∃ a b m n : Nat, (m ∈ p) ∧ (n ∈ p) ∧ (x = a^m + b^n)) ∧
    (∀ q : Nat, Nat.Prime q → (∃ c d : Nat, x = c^q + d^q) → q ∈ p) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_existence_l657_65770


namespace NUMINAMATH_CALUDE_turtle_marathon_time_l657_65753

/-- The time taken by a turtle to complete a marathon -/
theorem turtle_marathon_time (turtle_speed : ℝ) (marathon_distance : ℝ) :
  turtle_speed = 15 →
  marathon_distance = 42195 →
  ∃ (days hours minutes : ℕ),
    (days = 1 ∧ hours = 22 ∧ minutes = 53) ∧
    (days * 24 * 60 + hours * 60 + minutes : ℝ) * turtle_speed = marathon_distance :=
by sorry

end NUMINAMATH_CALUDE_turtle_marathon_time_l657_65753


namespace NUMINAMATH_CALUDE_project_cans_total_l657_65710

theorem project_cans_total (martha_cans : ℕ) (diego_extra : ℕ) (additional_cans : ℕ) : 
  martha_cans = 90 →
  diego_extra = 10 →
  additional_cans = 5 →
  martha_cans + (martha_cans / 2 + diego_extra) + additional_cans = 150 :=
by sorry

end NUMINAMATH_CALUDE_project_cans_total_l657_65710


namespace NUMINAMATH_CALUDE_stick_triangle_area_l657_65757

/-- Given three sticks of length 24, one of which is broken into two parts,
    if these parts form a right triangle with the other two sticks,
    then the area of this triangle is 216 square centimeters. -/
theorem stick_triangle_area : ∀ a : ℝ,
  0 < a →
  a < 24 →
  a^2 + 24^2 = (48 - a)^2 →
  (1/2) * a * 24 = 216 := by
  sorry

end NUMINAMATH_CALUDE_stick_triangle_area_l657_65757


namespace NUMINAMATH_CALUDE_park_tree_count_l657_65703

def park_trees (initial_maple : ℕ) (initial_poplar : ℕ) (oak : ℕ) : ℕ :=
  let planted_maple := 3 * initial_poplar
  let total_maple := initial_maple + planted_maple
  let planted_poplar := 3 * initial_poplar
  let total_poplar := initial_poplar + planted_poplar
  total_maple + total_poplar + oak

theorem park_tree_count :
  park_trees 2 5 4 = 32 := by sorry

end NUMINAMATH_CALUDE_park_tree_count_l657_65703


namespace NUMINAMATH_CALUDE_max_dimes_count_l657_65738

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.1

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The total amount of money Sasha has in dollars -/
def total_money : ℚ := 3.5

/-- Theorem: Given $3.50 in coins and an equal number of dimes and pennies, 
    the maximum number of dimes possible is 31 -/
theorem max_dimes_count : 
  ∃ (d : ℕ), d ≤ 31 ∧ 
  ∀ (n : ℕ), n * (dime_value + penny_value) ≤ total_money → n ≤ d :=
sorry

end NUMINAMATH_CALUDE_max_dimes_count_l657_65738


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l657_65779

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3| < 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l657_65779


namespace NUMINAMATH_CALUDE_new_function_not_transformation_of_original_l657_65737

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the new quadratic function
def new_function (x : ℝ) : ℝ := x^2

-- Define a general quadratic function
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Theorem statement
theorem new_function_not_transformation_of_original :
  ¬∃ (h k : ℝ), ∀ x, new_function x = original_function (x - h) + k ∧
  ¬∃ (h k : ℝ), ∀ x, new_function x = original_function (-(x - h)) + k :=
by sorry

end NUMINAMATH_CALUDE_new_function_not_transformation_of_original_l657_65737


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_minus_three_l657_65739

theorem one_third_of_seven_times_nine_minus_three (x : ℚ) : 
  x = (1 / 3 : ℚ) * (7 * 9) - 3 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_minus_three_l657_65739


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l657_65764

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {0, 1, 2}
def N : Set Int := {0, 1, 2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l657_65764


namespace NUMINAMATH_CALUDE_f_value_at_four_l657_65733

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^a else |x - 2|

theorem f_value_at_four (a : ℝ) :
  (f a (-2) = f a 2) → f a 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_four_l657_65733


namespace NUMINAMATH_CALUDE_exists_city_that_reaches_all_l657_65756

-- Define the type for cities
variable {City : Type}

-- Define the "can reach" relation
variable (canReach : City → City → Prop)

-- Define the properties of the "can reach" relation
variable (h_reflexive : ∀ x : City, canReach x x)
variable (h_transitive : ∀ x y z : City, canReach x y → canReach y z → canReach x z)

-- Define the condition that for any two cities, there's a city that can reach both
variable (h_common_reachable : ∀ x y : City, ∃ z : City, canReach z x ∧ canReach z y)

-- State the theorem
theorem exists_city_that_reaches_all [Finite City] :
  ∃ c : City, ∀ x : City, canReach c x :=
sorry

end NUMINAMATH_CALUDE_exists_city_that_reaches_all_l657_65756


namespace NUMINAMATH_CALUDE_two_digit_primes_with_digit_sum_10_l657_65712

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 10

theorem two_digit_primes_with_digit_sum_10 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_conditions n) ∧ S.card = 4 :=
sorry

end NUMINAMATH_CALUDE_two_digit_primes_with_digit_sum_10_l657_65712


namespace NUMINAMATH_CALUDE_difference_3003_l657_65732

/-- The number of terms in each sequence -/
def n : ℕ := 3003

/-- The sum of the first n odd numbers -/
def sum_odd (n : ℕ) : ℕ := n * n

/-- The sum of the first n even numbers starting from 2 -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even numbers (starting from 2) 
    and the sum of the first n odd numbers -/
def difference (n : ℕ) : ℤ := sum_even n - sum_odd n

theorem difference_3003 : difference n = 7999 := by sorry

end NUMINAMATH_CALUDE_difference_3003_l657_65732


namespace NUMINAMATH_CALUDE_boxwood_charge_theorem_l657_65789

/-- Calculates the total charge for trimming and shaping boxwoods -/
def total_charge (num_boxwoods : ℕ) (num_shaped : ℕ) (trim_cost : ℚ) (shape_cost : ℚ) : ℚ :=
  (num_boxwoods * trim_cost) + (num_shaped * shape_cost)

/-- Proves that the total charge for trimming 30 boxwoods and shaping 4 of them is $210.00 -/
theorem boxwood_charge_theorem :
  total_charge 30 4 5 15 = 210 := by
  sorry

#eval total_charge 30 4 5 15

end NUMINAMATH_CALUDE_boxwood_charge_theorem_l657_65789


namespace NUMINAMATH_CALUDE_inequality_solution_set_l657_65755

theorem inequality_solution_set (x : ℝ) : 
  x ≠ 3 → ((x^2 - 1) / ((x - 3)^2) ≥ 0 ↔ x ∈ Set.Iic (-1) ∪ Set.Ici 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l657_65755


namespace NUMINAMATH_CALUDE_square_sum_of_integers_l657_65795

theorem square_sum_of_integers (x y z : ℤ) 
  (eq1 : x^2*y + y^2*z + z^2*x = 2186)
  (eq2 : x*y^2 + y*z^2 + z*x^2 = 2188) :
  x^2 + y^2 + z^2 = 245 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_integers_l657_65795


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l657_65731

theorem fraction_sum_equality : (18 : ℚ) / 45 - 2 / 9 + 1 / 6 = 31 / 90 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l657_65731


namespace NUMINAMATH_CALUDE_problem_solution_l657_65754

-- Define the set M
def M : Set ℝ := {m | ∃ x ∈ Set.Icc (-1 : ℝ) 1, m = x^2 - x}

-- Define the set N
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x - (2 - a)) < 0}

-- Theorem statement
theorem problem_solution :
  (M = Set.Icc (-1/4 : ℝ) 2) ∧
  (∀ a : ℝ, N a ⊆ M ↔ 0 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l657_65754


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l657_65736

theorem polynomial_evaluation : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l657_65736


namespace NUMINAMATH_CALUDE_lance_penny_savings_l657_65778

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Lance's penny savings problem -/
theorem lance_penny_savings :
  arithmetic_sum 5 2 20 = 480 := by
  sorry

end NUMINAMATH_CALUDE_lance_penny_savings_l657_65778


namespace NUMINAMATH_CALUDE_distance_in_scientific_notation_l657_65707

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem distance_in_scientific_notation :
  let distance : ℝ := 38000
  let scientific_form := toScientificNotation distance
  scientific_form.coefficient = 3.8 ∧ scientific_form.exponent = 4 :=
sorry

end NUMINAMATH_CALUDE_distance_in_scientific_notation_l657_65707


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l657_65700

-- System (1)
theorem system_one_solution (x y : ℚ) :
  (4 * x - 3 * y = 1 ∧ 3 * x - 2 * y = -1) ↔ (x = -5 ∧ y = 7) :=
sorry

-- System (2)
theorem system_two_solution (x y : ℚ) :
  ((y + 1) / 4 = (x + 2) / 3 ∧ 2 * x - 3 * y = 1) ↔ (x = -3 ∧ y = -7/3) :=
sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l657_65700


namespace NUMINAMATH_CALUDE_cashier_bills_l657_65791

theorem cashier_bills (total_bills : ℕ) (total_value : ℕ) : 
  total_bills = 126 → total_value = 840 → ∃ (five_dollar_bills ten_dollar_bills : ℕ),
    five_dollar_bills + ten_dollar_bills = total_bills ∧
    5 * five_dollar_bills + 10 * ten_dollar_bills = total_value ∧
    five_dollar_bills = 84 := by
  sorry

end NUMINAMATH_CALUDE_cashier_bills_l657_65791


namespace NUMINAMATH_CALUDE_original_people_count_l657_65718

/-- The original number of people in the room. -/
def original_people : ℕ := 36

/-- The fraction of people who left initially. -/
def fraction_left : ℚ := 1 / 3

/-- The fraction of remaining people who started dancing. -/
def fraction_dancing : ℚ := 1 / 4

/-- The number of people who were not dancing. -/
def non_dancing_people : ℕ := 18

theorem original_people_count :
  (original_people : ℚ) * (1 - fraction_left) * (1 - fraction_dancing) = non_dancing_people := by
  sorry

end NUMINAMATH_CALUDE_original_people_count_l657_65718


namespace NUMINAMATH_CALUDE_union_S_T_l657_65726

-- Define the sets S and T
def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

-- State the theorem
theorem union_S_T : S ∪ T = {x : ℝ | x ≥ -4} := by sorry

end NUMINAMATH_CALUDE_union_S_T_l657_65726


namespace NUMINAMATH_CALUDE_max_k_inequality_l657_65742

theorem max_k_inequality (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∃ (k : ℝ), ∀ m, 0 < m → m < 1/2 → 1/m + 2/(1-2*m) ≥ k) ∧ 
  (∀ k, (∀ m, 0 < m → m < 1/2 → 1/m + 2/(1-2*m) ≥ k) → k ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_max_k_inequality_l657_65742


namespace NUMINAMATH_CALUDE_douglas_weight_is_52_l657_65751

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- The difference in weight between Anne and Douglas in pounds -/
def weight_difference : ℕ := 15

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := anne_weight - weight_difference

/-- Theorem stating Douglas's weight -/
theorem douglas_weight_is_52 : douglas_weight = 52 := by
  sorry

end NUMINAMATH_CALUDE_douglas_weight_is_52_l657_65751


namespace NUMINAMATH_CALUDE_linear_function_m_value_l657_65713

/-- A linear function y = mx + m^2 passing through (0, 4) with positive slope -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := m * x + m^2

theorem linear_function_m_value :
  ∀ m : ℝ,
  m ≠ 0 →
  linear_function m 0 = 4 →
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function m x₁ < linear_function m x₂) →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_m_value_l657_65713


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l657_65771

theorem arithmetic_sequence_sum (a : ℝ) :
  (a + 6 * 2 = 20) →  -- seventh term is 20
  (a + 2 + a = 18)    -- sum of first two terms is 18
:= by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l657_65771


namespace NUMINAMATH_CALUDE_system_solution_ratio_l657_65735

theorem system_solution_ratio (k x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + 2*k*y + 4*z = 0 →
  4*x + k*y - 3*z = 0 →
  3*x + 5*y - 4*z = 0 →
  x^2 * z / y^3 = 125 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l657_65735


namespace NUMINAMATH_CALUDE_red_in_B_equals_black_in_C_l657_65783

/-- Represents a playing card color -/
inductive CardColor
| Red
| Black

/-- Represents a box where cards are placed -/
inductive Box
| A
| B
| C

/-- Represents the state of the card distribution -/
structure CardDistribution where
  cardsInA : ℕ
  redInB : ℕ
  blackInB : ℕ
  redInC : ℕ
  blackInC : ℕ

/-- The card distribution process -/
def distributeCards : CardDistribution → CardColor → CardDistribution
| d, CardColor.Red => { d with
    cardsInA := d.cardsInA + 1,
    redInB := d.redInB + 1 }
| d, CardColor.Black => { d with
    cardsInA := d.cardsInA + 1,
    blackInC := d.blackInC + 1 }

/-- The theorem stating that the number of red cards in B equals the number of black cards in C -/
theorem red_in_B_equals_black_in_C (finalDist : CardDistribution)
  (h : finalDist.cardsInA = 52) :
  finalDist.redInB = finalDist.blackInC := by
  sorry

end NUMINAMATH_CALUDE_red_in_B_equals_black_in_C_l657_65783


namespace NUMINAMATH_CALUDE_competition_results_l657_65790

def team_a_scores : List ℝ := [7, 8, 9, 7, 10, 10, 9, 10, 10, 10]
def team_b_scores : List ℝ := [10, 8, 7, 9, 8, 10, 10, 9, 10, 9]
def team_a_variance : ℝ := 1.4

def median (l : List ℝ) : ℝ := sorry
def mode (l : List ℝ) : ℝ := sorry
def average (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem competition_results :
  (median team_a_scores = 9.5) ∧
  (mode team_b_scores = 10) ∧
  (average team_b_scores = 9) ∧
  (variance team_b_scores = 1) ∧
  (variance team_b_scores < team_a_variance) :=
by sorry

end NUMINAMATH_CALUDE_competition_results_l657_65790


namespace NUMINAMATH_CALUDE_league_games_count_l657_65728

/-- The number of games played in a season for a league with a given number of teams and games per pair of teams. -/
def games_in_season (num_teams : ℕ) (games_per_pair : ℕ) : ℕ :=
  (num_teams * (num_teams - 1) / 2) * games_per_pair

/-- Theorem: In a league with 20 teams where each pair of teams plays 10 games, 
    the total number of games in the season is 1900. -/
theorem league_games_count : games_in_season 20 10 = 1900 := by
  sorry

#eval games_in_season 20 10

end NUMINAMATH_CALUDE_league_games_count_l657_65728


namespace NUMINAMATH_CALUDE_bill_and_harry_combined_nuts_l657_65714

def sue_nuts : ℕ := 48

theorem bill_and_harry_combined_nuts :
  let harry_nuts := 2 * sue_nuts
  let bill_nuts := 6 * harry_nuts
  bill_nuts + harry_nuts = 672 := by
  sorry

end NUMINAMATH_CALUDE_bill_and_harry_combined_nuts_l657_65714


namespace NUMINAMATH_CALUDE_product_sum_relation_l657_65784

/-- Given single-digit integers P and Q where 39P × Q3 = 32951, prove that P + Q = 15 -/
theorem product_sum_relation (P Q : ℕ) : 
  P < 10 → Q < 10 → 39 * P * 10 + P * 3 + Q * 300 + Q * 30 + Q * 3 = 32951 → P + Q = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l657_65784


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l657_65749

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2016 = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l657_65749


namespace NUMINAMATH_CALUDE_helicopter_rental_hours_per_day_l657_65722

/-- Given the total cost, hourly rate, and number of days for renting a helicopter,
    calculate the number of hours rented per day. -/
theorem helicopter_rental_hours_per_day 
  (total_cost : ℝ) 
  (hourly_rate : ℝ) 
  (num_days : ℝ) 
  (h1 : total_cost = 450)
  (h2 : hourly_rate = 75)
  (h3 : num_days = 3)
  (h4 : hourly_rate > 0)
  (h5 : num_days > 0) :
  total_cost / (hourly_rate * num_days) = 2 := by
  sorry

#check helicopter_rental_hours_per_day

end NUMINAMATH_CALUDE_helicopter_rental_hours_per_day_l657_65722


namespace NUMINAMATH_CALUDE_square_perimeter_l657_65763

theorem square_perimeter (a : ℝ) : 
  a > 0 → 
  let l := a * Real.sqrt 2
  let d := l / 2 + l / 4 + l / 8 + l / 16
  d = 15 * Real.sqrt 2 → 
  4 * a = 64 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l657_65763


namespace NUMINAMATH_CALUDE_root_sum_product_l657_65724

theorem root_sum_product (a b : ℝ) : 
  (a^4 - 4*a^2 - a - 1 = 0) → 
  (b^4 - 4*b^2 - b - 1 = 0) → 
  (a + b) * (a * b + 1) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l657_65724


namespace NUMINAMATH_CALUDE_dessert_preference_l657_65777

theorem dessert_preference (total : ℕ) (apple : ℕ) (chocolate : ℕ) (neither : ℕ) : 
  total = 50 → apple = 22 → chocolate = 20 → neither = 17 →
  apple + chocolate - (total - neither) = 9 := by
sorry

end NUMINAMATH_CALUDE_dessert_preference_l657_65777


namespace NUMINAMATH_CALUDE_fraction_evaluation_l657_65775

theorem fraction_evaluation : 
  (10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l657_65775


namespace NUMINAMATH_CALUDE_opposite_sides_of_line_l657_65762

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in the form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to evaluate the line equation at a point
def evaluateLine (l : Line2D) (p : Point2D) : ℝ :=
  l.a * p.x + l.b * p.y + l.c

-- Define the specific line and points
def line : Line2D := { a := -3, b := 1, c := 2 }
def origin : Point2D := { x := 0, y := 0 }
def point : Point2D := { x := 2, y := 1 }

-- Theorem statement
theorem opposite_sides_of_line :
  evaluateLine line origin * evaluateLine line point < 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_of_line_l657_65762


namespace NUMINAMATH_CALUDE_union_of_sets_l657_65744

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 4}
  A ∪ B = {1, 2, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l657_65744


namespace NUMINAMATH_CALUDE_gold_silver_ratio_l657_65747

/-- Proves that the ratio of gold to silver bought is 2:1 given the specified conditions --/
theorem gold_silver_ratio :
  let silver_amount : ℝ := 1.5
  let silver_price_per_ounce : ℝ := 20
  let gold_price_multiplier : ℝ := 50
  let total_spent : ℝ := 3030
  let gold_price_per_ounce := silver_price_per_ounce * gold_price_multiplier
  let silver_cost := silver_amount * silver_price_per_ounce
  let gold_cost := total_spent - silver_cost
  let gold_amount := gold_cost / gold_price_per_ounce
  gold_amount / silver_amount = 2 := by
sorry


end NUMINAMATH_CALUDE_gold_silver_ratio_l657_65747


namespace NUMINAMATH_CALUDE_ring_arrangement_correct_l657_65719

/-- The number of ways to arrange 5 rings out of 9 on 5 fingers -/
def ring_arrangements (total_rings : ℕ) (arranged_rings : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings arranged_rings * Nat.factorial arranged_rings * Nat.choose (total_rings - 1) (fingers - 1)

/-- The correct number of arrangements for 9 rings, 5 arranged, on 5 fingers -/
def correct_arrangement : ℕ := 1905120

/-- Theorem stating that the number of arrangements is correct -/
theorem ring_arrangement_correct :
  ring_arrangements 9 5 5 = correct_arrangement := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangement_correct_l657_65719


namespace NUMINAMATH_CALUDE_initial_ratio_new_ratio_partners_count_l657_65786

/-- Represents the number of partners in a firm -/
def partners : ℕ := 18

/-- Represents the number of associates in a firm -/
def associates : ℕ := (63 * partners) / 2

/-- The ratio of partners to associates is 2:63 -/
theorem initial_ratio : partners * 63 = associates * 2 := by sorry

/-- Adding 45 associates changes the ratio to 1:34 -/
theorem new_ratio : partners * 34 = (associates + 45) * 1 := by sorry

/-- The number of partners in the firm is 18 -/
theorem partners_count : partners = 18 := by sorry

end NUMINAMATH_CALUDE_initial_ratio_new_ratio_partners_count_l657_65786


namespace NUMINAMATH_CALUDE_portfolio_worth_calculation_l657_65760

/-- Calculates the final portfolio worth after two years given the initial investment,
    growth rates, and transactions. -/
def calculate_portfolio_worth (initial_investment : ℝ) 
                              (year1_growth_rate : ℝ) 
                              (year1_addition : ℝ) 
                              (year1_withdrawal : ℝ)
                              (year2_growth_rate1 : ℝ)
                              (year2_decline_rate : ℝ) : ℝ :=
  sorry

/-- Theorem stating that given the specified conditions, 
    the final portfolio worth is approximately $115.59 -/
theorem portfolio_worth_calculation :
  let initial_investment : ℝ := 80
  let year1_growth_rate : ℝ := 0.15
  let year1_addition : ℝ := 28
  let year1_withdrawal : ℝ := 10
  let year2_growth_rate1 : ℝ := 0.10
  let year2_decline_rate : ℝ := 0.04
  
  abs (calculate_portfolio_worth initial_investment 
                                 year1_growth_rate
                                 year1_addition
                                 year1_withdrawal
                                 year2_growth_rate1
                                 year2_decline_rate - 115.59) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_portfolio_worth_calculation_l657_65760


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l657_65767

def scores : List ℝ := [90, 93.5, 87, 96, 92, 89.5]

theorem arithmetic_mean_of_scores :
  (scores.sum / scores.length : ℝ) = 91.333333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l657_65767


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l657_65752

theorem tan_alpha_plus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  Real.tan (α + π/4) = 3/22 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l657_65752


namespace NUMINAMATH_CALUDE_meal_cost_l657_65780

theorem meal_cost (adults children : ℕ) (total_bill : ℚ) :
  adults = 2 →
  children = 5 →
  total_bill = 21 →
  (total_bill / (adults + children : ℚ)) = 3 := by
sorry

end NUMINAMATH_CALUDE_meal_cost_l657_65780


namespace NUMINAMATH_CALUDE_count_seven_digit_phone_numbers_l657_65730

/-- The number of different seven-digit phone numbers where the first digit cannot be zero -/
def seven_digit_phone_numbers : ℕ :=
  9 * 10^6

/-- Theorem stating that the number of different seven-digit phone numbers
    where the first digit cannot be zero is equal to 9 × 10^6 -/
theorem count_seven_digit_phone_numbers :
  seven_digit_phone_numbers = 9 * 10^6 := by
  sorry

end NUMINAMATH_CALUDE_count_seven_digit_phone_numbers_l657_65730


namespace NUMINAMATH_CALUDE_compound_interest_rate_problem_l657_65705

theorem compound_interest_rate_problem (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 17640) 
  (h2 : P * (1 + r)^3 = 18522) : 
  (1 + r)^3 / (1 + r)^2 = 18522 / 17640 := by sorry

end NUMINAMATH_CALUDE_compound_interest_rate_problem_l657_65705


namespace NUMINAMATH_CALUDE_marks_change_factor_l657_65798

theorem marks_change_factor (n : ℕ) (initial_avg final_avg : ℝ) (h_n : n = 12) (h_initial : initial_avg = 50) (h_final : final_avg = 100) :
  (final_avg * n) / (initial_avg * n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_marks_change_factor_l657_65798


namespace NUMINAMATH_CALUDE_inequality_proof_l657_65774

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1 / a + 1 / b + 1 / c = 1) :
  Real.sqrt (a * b + c) + Real.sqrt (b * c + a) + Real.sqrt (c * a + b) ≥
  Real.sqrt (a * b * c) + Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l657_65774


namespace NUMINAMATH_CALUDE_equation_solution_l657_65745

theorem equation_solution : 
  ∃ x : ℚ, x ≠ -4 ∧ (7 * x / (x + 4) - 4 / (x + 4) = 2 / (x + 4)) ∧ x = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l657_65745


namespace NUMINAMATH_CALUDE_polygon_with_five_triangles_is_heptagon_l657_65793

/-- The number of triangles formed by diagonals from one vertex in an n-sided polygon -/
def triangles_from_diagonals (n : ℕ) : ℕ := n - 2

theorem polygon_with_five_triangles_is_heptagon (n : ℕ) :
  (n ≥ 3) → (triangles_from_diagonals n = 5) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_five_triangles_is_heptagon_l657_65793


namespace NUMINAMATH_CALUDE_jolene_washed_five_cars_l657_65797

/-- The number of cars Jolene washed to raise money for a bicycle -/
def cars_washed (families : ℕ) (babysitting_rate : ℕ) (car_wash_rate : ℕ) (total_raised : ℕ) : ℕ :=
  (total_raised - families * babysitting_rate) / car_wash_rate

/-- Theorem: Jolene washed 5 cars given the problem conditions -/
theorem jolene_washed_five_cars :
  cars_washed 4 30 12 180 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jolene_washed_five_cars_l657_65797


namespace NUMINAMATH_CALUDE_container_capacity_l657_65704

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 9 = 0.75 * C) : C = 20 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l657_65704


namespace NUMINAMATH_CALUDE_license_plate_count_l657_65711

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits available -/
def num_digits : ℕ := 10

/-- A valid license plate configuration -/
structure LicensePlate where
  first : Fin num_letters
  second : Fin (num_letters + num_digits - 2)
  third : Fin num_letters
  fourth : Fin num_digits

/-- The total number of valid license plates -/
def total_license_plates : ℕ := num_letters * (num_letters + num_digits - 2) * num_letters * num_digits

theorem license_plate_count :
  total_license_plates = 236600 := by
  sorry

#eval total_license_plates

end NUMINAMATH_CALUDE_license_plate_count_l657_65711


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l657_65796

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all x. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_condition (k : ℝ) :
  IsPerfectSquareTrinomial 4 (-2 * k) 1 → k = 2 ∨ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l657_65796


namespace NUMINAMATH_CALUDE_problem_solution_l657_65759

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 15) :
  a / (a + b) + b / (b + c) + c / (c + a) = -12 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l657_65759


namespace NUMINAMATH_CALUDE_adding_five_increases_value_l657_65765

theorem adding_five_increases_value (x : ℝ) : x + 5 > x := by
  sorry

end NUMINAMATH_CALUDE_adding_five_increases_value_l657_65765


namespace NUMINAMATH_CALUDE_petya_bonus_points_l657_65769

def calculate_bonus (score : ℕ) : ℕ :=
  if score < 1000 then
    (score * 20) / 100
  else if score < 2000 then
    200 + ((score - 1000) * 30) / 100
  else
    200 + 300 + ((score - 2000) * 50) / 100

theorem petya_bonus_points : calculate_bonus 2370 = 685 := by
  sorry

end NUMINAMATH_CALUDE_petya_bonus_points_l657_65769


namespace NUMINAMATH_CALUDE_total_weight_theorem_l657_65761

/-- The weight of the orange ring in ounces -/
def orange_ring_oz : ℚ := 1 / 12

/-- The weight of the purple ring in ounces -/
def purple_ring_oz : ℚ := 1 / 3

/-- The weight of the white ring in ounces -/
def white_ring_oz : ℚ := 5 / 12

/-- The weight of the blue ring in ounces -/
def blue_ring_oz : ℚ := 1 / 4

/-- The weight of the green ring in ounces -/
def green_ring_oz : ℚ := 1 / 6

/-- The weight of the red ring in ounces -/
def red_ring_oz : ℚ := 1 / 10

/-- The conversion factor from ounces to grams -/
def oz_to_g : ℚ := 28.3495

/-- The total weight of all rings in grams -/
def total_weight_g : ℚ :=
  (orange_ring_oz + purple_ring_oz + white_ring_oz + blue_ring_oz + green_ring_oz + red_ring_oz) * oz_to_g

theorem total_weight_theorem :
  total_weight_g = 38.271825 := by sorry

end NUMINAMATH_CALUDE_total_weight_theorem_l657_65761


namespace NUMINAMATH_CALUDE_possible_values_of_a_plus_b_l657_65772

theorem possible_values_of_a_plus_b (a b : ℝ) 
  (h1 : |a| = 5) 
  (h2 : |b| = 1) 
  (h3 : a - b < 0) : 
  a + b = -6 ∨ a + b = -4 := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_plus_b_l657_65772


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l657_65750

/-- 
Given a point P in a Cartesian coordinate system, this theorem states that 
its coordinates with respect to the origin are the negatives of its original coordinates.
-/
theorem point_coordinates_wrt_origin (x y : ℝ) : 
  let P : ℝ × ℝ := (x, y)
  let P_wrt_origin : ℝ × ℝ := (-x, -y)
  P_wrt_origin = (-(P.1), -(P.2)) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l657_65750


namespace NUMINAMATH_CALUDE_tank_circumference_l657_65725

/-- Given two right circular cylindrical tanks C and B, prove that the circumference of tank B is 10 meters. -/
theorem tank_circumference (h_C h_B r_C r_B : ℝ) : 
  h_C = 10 →  -- Height of tank C
  h_B = 8 →   -- Height of tank B
  2 * Real.pi * r_C = 8 →  -- Circumference of tank C
  (Real.pi * r_C^2 * h_C) = 0.8 * (Real.pi * r_B^2 * h_B) →  -- Volume relation
  2 * Real.pi * r_B = 10  -- Circumference of tank B
:= by sorry

end NUMINAMATH_CALUDE_tank_circumference_l657_65725


namespace NUMINAMATH_CALUDE_xoxoxox_probability_l657_65701

/-- The probability of arranging 4 X tiles and 3 O tiles in the specific order XOXOXOX -/
theorem xoxoxox_probability (n : ℕ) (x o : ℕ) (h1 : n = 7) (h2 : x = 4) (h3 : o = 3) :
  (1 : ℚ) / (n.choose x) = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_xoxoxox_probability_l657_65701


namespace NUMINAMATH_CALUDE_dinner_time_calculation_l657_65729

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, sorry⟩

theorem dinner_time_calculation (start : Time) (commute grocery drycleaning groomer cooking : ℕ) :
  commute = 30 →
  grocery = 30 →
  drycleaning = 10 →
  groomer = 20 →
  cooking = 90 →
  start = ⟨16, 0, sorry⟩ →
  addMinutes start (commute + grocery + drycleaning + groomer + cooking) = ⟨19, 0, sorry⟩ := by
  sorry


end NUMINAMATH_CALUDE_dinner_time_calculation_l657_65729


namespace NUMINAMATH_CALUDE_female_fraction_is_19_52_l657_65788

/-- Represents the chess club membership --/
structure ChessClub where
  males_last_year : ℕ
  total_increase_rate : ℚ
  male_increase_rate : ℚ
  female_increase_rate : ℚ

/-- Calculates the fraction of female participants in the chess club this year --/
def female_fraction (club : ChessClub) : ℚ :=
  sorry

/-- Theorem stating that the fraction of female participants is 19/52 --/
theorem female_fraction_is_19_52 (club : ChessClub) 
  (h1 : club.males_last_year = 30)
  (h2 : club.total_increase_rate = 15/100)
  (h3 : club.male_increase_rate = 10/100)
  (h4 : club.female_increase_rate = 25/100) :
  female_fraction club = 19/52 := by
  sorry

end NUMINAMATH_CALUDE_female_fraction_is_19_52_l657_65788


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l657_65748

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l657_65748


namespace NUMINAMATH_CALUDE_product_327_8_and_7_8_l657_65787

/-- Convert a number from base 8 to base 10 -/
def base8To10 (n : ℕ) : ℕ := sorry

/-- Convert a number from base 10 to base 8 -/
def base10To8 (n : ℕ) : ℕ := sorry

/-- Multiply two numbers in base 8 -/
def multiplyBase8 (a b : ℕ) : ℕ :=
  base10To8 (base8To10 a * base8To10 b)

theorem product_327_8_and_7_8 :
  multiplyBase8 327 7 = 2741 := by sorry

end NUMINAMATH_CALUDE_product_327_8_and_7_8_l657_65787


namespace NUMINAMATH_CALUDE_diophantine_fraction_equality_l657_65702

theorem diophantine_fraction_equality : ∃ (A B : ℤ), 
  A = 500 ∧ B = -501 ∧ (A : ℚ) / 999 + (B : ℚ) / 1001 = 1 / 999999 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_fraction_equality_l657_65702


namespace NUMINAMATH_CALUDE_negp_sufficient_not_necessary_for_negq_l657_65773

def p (x : ℝ) : Prop := x < -1 ∨ x > 1

def q (x : ℝ) : Prop := x < -2 ∨ x > 1

theorem negp_sufficient_not_necessary_for_negq :
  (∀ x, ¬(p x) → ¬(q x)) ∧ ¬(∀ x, ¬(q x) → ¬(p x)) := by sorry

end NUMINAMATH_CALUDE_negp_sufficient_not_necessary_for_negq_l657_65773


namespace NUMINAMATH_CALUDE_infinitely_many_2024_endings_l657_65768

/-- The sequence (x_n) defined by the given recurrence relation -/
def x : ℕ → ℕ
  | 0 => 0
  | 1 => 2024
  | (n + 2) => x (n + 1) + x n

/-- The set of natural numbers n where x_n ends with 2024 -/
def ends_with_2024 : Set ℕ := {n | x n % 10000 = 2024}

/-- The main theorem stating that there are infinitely many terms in the sequence ending with 2024 -/
theorem infinitely_many_2024_endings : Set.Infinite ends_with_2024 := by
  sorry


end NUMINAMATH_CALUDE_infinitely_many_2024_endings_l657_65768


namespace NUMINAMATH_CALUDE_dice_sum_theorem_l657_65715

/-- Represents a single die -/
structure Die where
  opposite_sum : ℕ
  opposite_sum_is_seven : opposite_sum = 7

/-- Represents a set of 7 dice -/
structure DiceSet where
  dice : Fin 7 → Die
  all_dice_have_opposite_sum_seven : ∀ i, (dice i).opposite_sum = 7

/-- The sum of numbers on the upward faces of a set of dice -/
def upward_sum (d : DiceSet) : ℕ := sorry

/-- The sum of numbers on the downward faces of a set of dice -/
def downward_sum (d : DiceSet) : ℕ := sorry

/-- The probability of getting a specific sum on the upward faces -/
noncomputable def prob_upward_sum (d : DiceSet) (sum : ℕ) : ℝ := sorry

/-- The probability of getting a specific sum on the downward faces -/
noncomputable def prob_downward_sum (d : DiceSet) (sum : ℕ) : ℝ := sorry

theorem dice_sum_theorem (d : DiceSet) (a : ℕ) 
  (h1 : a ≠ 10)
  (h2 : prob_upward_sum d 10 = prob_downward_sum d a) :
  a = 39 := by sorry

end NUMINAMATH_CALUDE_dice_sum_theorem_l657_65715


namespace NUMINAMATH_CALUDE_unique_solution_iff_c_equals_three_l657_65740

theorem unique_solution_iff_c_equals_three :
  ∀ c : ℝ, (∃! (x y : ℝ), (2 * |x + 7| + |y - 4| = c) ∧ (|x + 4| + 2 * |y - 7| = c)) ↔ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_c_equals_three_l657_65740


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l657_65758

theorem cubic_polynomial_property (p q r : ℝ) : 
  let Q : ℝ → ℝ := λ x => x^3 + p*x^2 + q*x + r
  let mean_zeros := -p / 3
  let product_zeros := -r
  let sum_coefficients := 1 + p + q + r
  (mean_zeros = product_zeros ∧ product_zeros = sum_coefficients ∧ r = 3) →
  q = -16 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l657_65758


namespace NUMINAMATH_CALUDE_total_wheels_at_park_l657_65741

-- Define the number of regular bikes
def regular_bikes : ℕ := 7

-- Define the number of children's bikes
def children_bikes : ℕ := 11

-- Define the number of wheels on a regular bike
def regular_bike_wheels : ℕ := 2

-- Define the number of wheels on a children's bike
def children_bike_wheels : ℕ := 4

-- Theorem: The total number of wheels Naomi saw at the park is 58
theorem total_wheels_at_park : 
  regular_bikes * regular_bike_wheels + children_bikes * children_bike_wheels = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_at_park_l657_65741


namespace NUMINAMATH_CALUDE_doubled_roots_ratio_l657_65782

theorem doubled_roots_ratio (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : ∃ x₁ x₂ : ℝ, (x₁^2 + a*x₁ + b = 0 ∧ x₂^2 + a*x₂ + b = 0) ∧ 
                    ((2*x₁)^2 + b*(2*x₁) + c = 0 ∧ (2*x₂)^2 + b*(2*x₂) + c = 0)) :
  a / c = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_doubled_roots_ratio_l657_65782


namespace NUMINAMATH_CALUDE_fraction_simplification_l657_65776

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l657_65776


namespace NUMINAMATH_CALUDE_trig_identity_l657_65799

theorem trig_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.cos (x + y) =
  Real.sin x ^ 2 + Real.sin y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l657_65799


namespace NUMINAMATH_CALUDE_vanishing_function_l657_65781

theorem vanishing_function (g : ℝ → ℝ) (h₁ : Continuous (deriv g)) 
  (h₂ : g 0 = 0) (h₃ : ∀ x, |deriv g x| ≤ |g x|) : 
  ∀ x, g x = 0 := by
  sorry

end NUMINAMATH_CALUDE_vanishing_function_l657_65781


namespace NUMINAMATH_CALUDE_solve_for_m_l657_65717

theorem solve_for_m (n : ℝ) : 
  ∃ m : ℝ, 21 * (m + n) + 21 = 21 * (-m + n) + 21 ∧ m = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l657_65717


namespace NUMINAMATH_CALUDE_square_area_error_l657_65709

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.01)
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.0201 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l657_65709


namespace NUMINAMATH_CALUDE_inequality_solution_set_l657_65723

theorem inequality_solution_set (x : ℝ) : 
  (2 * x^2 - x ≤ 0) ↔ (0 ≤ x ∧ x ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l657_65723


namespace NUMINAMATH_CALUDE_ellipse_intersection_k_values_l657_65746

noncomputable section

def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def Line (k m : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + m}

def Eccentricity (a b : ℝ) := Real.sqrt (1 - (b^2 / a^2))

def Parallelogram (A B C D : ℝ × ℝ) :=
  (B.1 - A.1 = D.1 - C.1 ∧ B.2 - A.2 = D.2 - C.2) ∧
  (C.1 - A.1 = D.1 - B.1 ∧ C.2 - A.2 = D.2 - B.2)

theorem ellipse_intersection_k_values
  (a b : ℝ)
  (h_ab : a > b ∧ b > 0)
  (h_A : (2, 0) ∈ Ellipse a b)
  (h_e : Eccentricity a b = Real.sqrt 3 / 2)
  (k : ℝ)
  (M N : ℝ × ℝ)
  (h_MN : M ∈ Ellipse a b ∧ N ∈ Ellipse a b)
  (h_MN_line : M ∈ Line k (Real.sqrt 3) ∧ N ∈ Line k (Real.sqrt 3))
  (P : ℝ × ℝ)
  (h_P : P.1 = 3)
  (h_parallelogram : Parallelogram (2, 0) P M N) :
  k = Real.sqrt 3 / 2 ∨ k = Real.sqrt 11 / 2 ∨ k = -Real.sqrt 11 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_k_values_l657_65746
