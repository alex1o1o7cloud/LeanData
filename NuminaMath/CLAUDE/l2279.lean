import Mathlib

namespace min_value_of_function_l2279_227984

theorem min_value_of_function (x : ℝ) (h : x > 5/4) :
  ∃ (y : ℝ), y = 4*x - 1 + 1/(4*x - 5) ∧ y ≥ 6 ∧ (∃ (x₀ : ℝ), x₀ > 5/4 ∧ 4*x₀ - 1 + 1/(4*x₀ - 5) = 6) :=
by sorry

end min_value_of_function_l2279_227984


namespace jills_shopping_breakdown_l2279_227921

/-- Represents the shopping breakdown and tax calculation for Jill's purchase --/
theorem jills_shopping_breakdown (T : ℝ) (x : ℝ) 
  (h1 : T > 0) -- Total amount spent is positive
  (h2 : x ≥ 0 ∧ x ≤ 1) -- Percentage spent on other items is between 0 and 100%
  (h3 : 0.5 + 0.2 + x = 1) -- Total percentage spent is 100%
  (h4 : 0.02 * T + 0.1 * x * T = 0.05 * T) -- Tax equation
  : x = 0.3 := by
  sorry

end jills_shopping_breakdown_l2279_227921


namespace circle_area_tripled_l2279_227992

theorem circle_area_tripled (n : ℝ) (r : ℝ) (h_pos : r > 0) : 
  π * (r + n)^2 = 3 * π * r^2 → r = n * (Real.sqrt 3 - 1) / 2 := by
sorry

end circle_area_tripled_l2279_227992


namespace x_value_when_y_is_negative_four_l2279_227965

theorem x_value_when_y_is_negative_four :
  ∀ x y : ℝ, 16 * (3 : ℝ)^x = 7^(y + 4) → y = -4 → x = -4 * (Real.log 2 / Real.log 3) := by
  sorry

end x_value_when_y_is_negative_four_l2279_227965


namespace cindys_age_l2279_227944

/-- Given the ages of siblings, prove Cindy's age -/
theorem cindys_age (cindy jan marcia greg : ℕ) 
  (h1 : jan = cindy + 2)
  (h2 : marcia = 2 * jan)
  (h3 : greg = marcia + 2)
  (h4 : greg = 16) :
  cindy = 5 := by
  sorry

end cindys_age_l2279_227944


namespace clothing_store_profit_model_l2279_227922

/-- Represents the clothing store's sales and profit model -/
structure ClothingStore where
  originalCost : ℝ
  originalPrice : ℝ
  originalSales : ℝ
  salesIncrease : ℝ
  priceReduction : ℝ

/-- Calculate daily sales after price reduction -/
def dailySales (store : ClothingStore) : ℝ :=
  store.originalSales + store.salesIncrease * store.priceReduction

/-- Calculate profit per piece after price reduction -/
def profitPerPiece (store : ClothingStore) : ℝ :=
  store.originalPrice - store.originalCost - store.priceReduction

/-- Calculate total daily profit -/
def dailyProfit (store : ClothingStore) : ℝ :=
  dailySales store * profitPerPiece store

/-- The main theorem about the clothing store's profit model -/
theorem clothing_store_profit_model (store : ClothingStore) 
  (h1 : store.originalCost = 80)
  (h2 : store.originalPrice = 120)
  (h3 : store.originalSales = 20)
  (h4 : store.salesIncrease = 2) :
  (∀ x, dailySales { store with priceReduction := x } = 20 + 2 * x) ∧
  (∀ x, profitPerPiece { store with priceReduction := x } = 40 - x) ∧
  (dailyProfit { store with priceReduction := 20 } = 1200) ∧
  (∀ x, dailyProfit { store with priceReduction := x } ≠ 2000) := by
  sorry


end clothing_store_profit_model_l2279_227922


namespace f_odd_and_decreasing_l2279_227999

-- Define the function f(x) = -x^3
def f (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end f_odd_and_decreasing_l2279_227999


namespace deductive_reasoning_example_l2279_227927

-- Define the property of conducting electricity
def conducts_electricity (x : Type) : Prop := sorry

-- Define the concept of metal
def is_metal (x : Type) : Prop := sorry

-- Define deductive reasoning
def is_deductive_reasoning (premise1 premise2 conclusion : Prop) : Prop := sorry

-- Define the specific metals
def gold : Type := sorry
def silver : Type := sorry
def copper : Type := sorry

-- Theorem statement
theorem deductive_reasoning_example :
  let premise1 : Prop := ∀ x, is_metal x → conducts_electricity x
  let premise2 : Prop := is_metal gold ∧ is_metal silver ∧ is_metal copper
  let conclusion : Prop := conducts_electricity gold ∧ conducts_electricity silver ∧ conducts_electricity copper
  is_deductive_reasoning premise1 premise2 conclusion := by sorry

end deductive_reasoning_example_l2279_227927


namespace mady_balls_after_2023_steps_l2279_227910

/-- Converts a natural number to its septenary (base 7) representation -/
def to_septenary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Sums the digits in a list of natural numbers -/
def sum_digits (l : List ℕ) : ℕ :=
  l.sum

/-- Represents Mady's ball placement process -/
def mady_process (steps : ℕ) : ℕ :=
  sum_digits (to_septenary steps)

theorem mady_balls_after_2023_steps :
  mady_process 2023 = 13 := by sorry

end mady_balls_after_2023_steps_l2279_227910


namespace ticket_identification_operations_l2279_227952

/-- The maximum ticket number --/
def max_ticket : Nat := 30

/-- The number of operations needed to identify all ticket numbers --/
def num_operations : Nat := 5

/-- Function to calculate the number of binary digits needed to represent a number --/
def binary_digits (n : Nat) : Nat :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem ticket_identification_operations :
  binary_digits max_ticket = num_operations :=
by sorry

end ticket_identification_operations_l2279_227952


namespace cats_remaining_after_sale_l2279_227986

/-- The number of cats remaining after a sale at a pet store. -/
theorem cats_remaining_after_sale (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 19 → house = 45 → sold = 56 → siamese + house - sold = 8 := by
  sorry

end cats_remaining_after_sale_l2279_227986


namespace grocery_solution_l2279_227902

/-- Represents the grocery shopping problem --/
def grocery_problem (mustard_oil_price : ℝ) (mustard_oil_amount : ℝ) 
  (pasta_price : ℝ) (sauce_price : ℝ) (sauce_amount : ℝ) 
  (initial_money : ℝ) (remaining_money : ℝ) : Prop :=
  ∃ (pasta_amount : ℝ),
    mustard_oil_price * mustard_oil_amount + 
    pasta_price * pasta_amount + 
    sauce_price * sauce_amount = 
    initial_money - remaining_money ∧
    pasta_amount = 3

/-- Theorem stating the solution to the grocery problem --/
theorem grocery_solution : 
  grocery_problem 13 2 4 5 1 50 7 := by
  sorry

end grocery_solution_l2279_227902


namespace age_problem_l2279_227931

theorem age_problem (age_older age_younger : ℕ) : 
  age_older = age_younger + 2 →
  age_older + age_younger = 74 →
  age_older = 38 := by
sorry

end age_problem_l2279_227931


namespace unique_three_digit_square_l2279_227977

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define a function to get the first two digits of a three-digit number
def first_two_digits (n : ℕ) : ℕ :=
  n / 10

-- Define a function to get the last digit of a three-digit number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the main theorem
theorem unique_three_digit_square : ∃! n : ℕ, 
  100 ≤ n ∧ n ≤ 999 ∧ 
  is_perfect_square n ∧ 
  is_perfect_square (first_two_digits n / last_digit n) ∧
  n = 361 :=
sorry

end unique_three_digit_square_l2279_227977


namespace residue_mod_13_l2279_227932

theorem residue_mod_13 : (156 + 3 * 52 + 4 * 182 + 6 * 26) % 13 = 0 := by
  sorry

end residue_mod_13_l2279_227932


namespace systematic_sampling_methods_systematic_sampling_characterization_l2279_227912

/-- Represents a sampling method -/
inductive SamplingMethod
| Method1
| Method2
| Method3
| Method4

/-- Predicate to determine if a sampling method is systematic -/
def is_systematic (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.Method1 => true
  | SamplingMethod.Method2 => true
  | SamplingMethod.Method3 => false
  | SamplingMethod.Method4 => true

/-- Theorem stating which sampling methods are systematic -/
theorem systematic_sampling_methods :
  (is_systematic SamplingMethod.Method1) ∧
  (is_systematic SamplingMethod.Method2) ∧
  (¬is_systematic SamplingMethod.Method3) ∧
  (is_systematic SamplingMethod.Method4) :=
by sorry

/-- Characterization of systematic sampling -/
theorem systematic_sampling_characterization (method : SamplingMethod) :
  is_systematic method ↔ 
    (∃ (rule : Prop), 
      (rule ↔ method = SamplingMethod.Method1 ∨ 
               method = SamplingMethod.Method2 ∨ 
               method = SamplingMethod.Method4) ∧
      (rule → ∃ (interval : Nat), interval > 0)) :=
by sorry

end systematic_sampling_methods_systematic_sampling_characterization_l2279_227912


namespace no_statement_implies_p_and_not_q_l2279_227937

theorem no_statement_implies_p_and_not_q (p q : Prop) : 
  ¬((p → q) → (p ∧ ¬q)) ∧ 
  ¬((p ∨ ¬q) → (p ∧ ¬q)) ∧ 
  ¬((¬p ∧ q) → (p ∧ ¬q)) ∧ 
  ¬((¬p ∨ q) → (p ∧ ¬q)) := by
  sorry

end no_statement_implies_p_and_not_q_l2279_227937


namespace average_birds_seen_l2279_227942

def marcus_birds : ℕ := 7
def humphrey_birds : ℕ := 11
def darrel_birds : ℕ := 9
def total_watchers : ℕ := 3

theorem average_birds_seen :
  (marcus_birds + humphrey_birds + darrel_birds) / total_watchers = 9 := by
  sorry

end average_birds_seen_l2279_227942


namespace sqrt_equation_solution_l2279_227923

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (25 - Real.sqrt n) = 3 → n = 256 := by
  sorry

end sqrt_equation_solution_l2279_227923


namespace election_majority_l2279_227918

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 900 →
  winning_percentage = 70 / 100 →
  (total_votes : ℚ) * winning_percentage - (total_votes : ℚ) * (1 - winning_percentage) = 360 := by
sorry

end election_majority_l2279_227918


namespace quadratic_distinct_roots_l2279_227957

theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) ↔ c < 1/4 := by
  sorry

end quadratic_distinct_roots_l2279_227957


namespace arithmetic_sequence_ninth_term_l2279_227945

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 5 + a 7 = 16) 
  (h_third : a 3 = 4) : 
  a 9 = 12 := by
sorry

end arithmetic_sequence_ninth_term_l2279_227945


namespace determinant_2x2_l2279_227909

open Matrix

theorem determinant_2x2 (a b c d : ℝ) : 
  det ![![a, c], ![b, d]] = a * d - b * c := by
  sorry

end determinant_2x2_l2279_227909


namespace passing_methods_after_six_passes_l2279_227967

/-- The number of ways the ball can be passed back to player A after n passes -/
def passing_methods (n : ℕ) : ℕ :=
  if n < 2 then 0
  else if n = 2 then 2
  else 2^(n-1) - passing_methods (n-1)

/-- The theorem stating that there are 22 different passing methods after 6 passes -/
theorem passing_methods_after_six_passes :
  passing_methods 6 = 22 := by
  sorry

end passing_methods_after_six_passes_l2279_227967


namespace inequality_proof_l2279_227975

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 6) :
  (1 / (a * (1 + b))) + (1 / (b * (1 + c))) + (1 / (c * (1 + a))) ≥ 1 / 2 := by
  sorry

end inequality_proof_l2279_227975


namespace trigonometric_equation_solution_l2279_227998

theorem trigonometric_equation_solution (x : ℝ) :
  (Real.sin x)^3 + 6 * (Real.cos x)^3 + (1 / Real.sqrt 2) * Real.sin (2 * x) * Real.sin (x + π / 4) = 0 →
  ∃ n : ℤ, x = -Real.arctan 2 + n * π :=
by sorry

end trigonometric_equation_solution_l2279_227998


namespace infinite_solutions_condition_l2279_227953

theorem infinite_solutions_condition (a : ℝ) :
  (∀ x : ℝ, 3 * (2 * x - a) = 2 * (3 * x + 12)) ↔ a = -8 := by
  sorry

end infinite_solutions_condition_l2279_227953


namespace mechanics_billing_problem_l2279_227976

/-- A mechanic's billing problem -/
theorem mechanics_billing_problem 
  (total_bill : ℝ) 
  (parts_cost : ℝ) 
  (job_duration : ℝ) 
  (h1 : total_bill = 450)
  (h2 : parts_cost = 225)
  (h3 : job_duration = 5) :
  (total_bill - parts_cost) / job_duration = 45 := by
sorry

end mechanics_billing_problem_l2279_227976


namespace interior_angle_regular_pentagon_l2279_227900

/-- The measure of one interior angle of a regular pentagon is 108 degrees. -/
theorem interior_angle_regular_pentagon : ℝ :=
  let n : ℕ := 5  -- number of sides in a pentagon
  let S : ℝ := 180 * (n - 2)  -- sum of interior angles formula
  let angle_measure : ℝ := S / n  -- measure of one interior angle
  108

/-- Proof of the theorem -/
lemma proof_interior_angle_regular_pentagon : interior_angle_regular_pentagon = 108 := by
  sorry

end interior_angle_regular_pentagon_l2279_227900


namespace p_sufficient_not_necessary_for_q_l2279_227947

-- Define propositions p and q
def p (x : ℝ) : Prop := 1 < x ∧ x < 2
def q (x : ℝ) : Prop := x > 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by
  sorry

end p_sufficient_not_necessary_for_q_l2279_227947


namespace square_circle_union_area_l2279_227982

theorem square_circle_union_area (s : Real) (r : Real) : 
  s = 12 → r = 12 → (s ^ 2 + π * r ^ 2 - s ^ 2) = 144 * π := by
  sorry

end square_circle_union_area_l2279_227982


namespace nut_boxes_problem_l2279_227906

theorem nut_boxes_problem (first second third : ℕ) : 
  (second = (11 * first) / 10) →
  (second = (13 * third) / 10) →
  (first = third + 80) →
  (first = 520 ∧ second = 572 ∧ third = 440) :=
by sorry

end nut_boxes_problem_l2279_227906


namespace one_eighth_of_2_36_l2279_227970

theorem one_eighth_of_2_36 (y : ℤ) : (1 / 8 : ℚ) * (2 ^ 36) = 2 ^ y → y = 33 := by
  sorry

end one_eighth_of_2_36_l2279_227970


namespace one_third_of_ten_y_minus_three_l2279_227904

theorem one_third_of_ten_y_minus_three (y : ℝ) : (1/3) * (10*y - 3) = (10*y)/3 - 1 := by
  sorry

end one_third_of_ten_y_minus_three_l2279_227904


namespace max_value_of_expression_l2279_227907

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 9 → (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ (a - b)^2 + (b - c)^2 + (c - a)^2) →
  (a - b)^2 + (b - c)^2 + (c - a)^2 = 27 :=
sorry

end max_value_of_expression_l2279_227907


namespace inequality_proof_l2279_227960

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (2 * x^2) / (y + z) + (2 * y^2) / (z + x) + (2 * z^2) / (x + y) ≥ x + y + z := by
  sorry

end inequality_proof_l2279_227960


namespace square_diff_squared_l2279_227915

theorem square_diff_squared : (7^2 - 5^2)^2 = 576 := by sorry

end square_diff_squared_l2279_227915


namespace misread_addition_l2279_227916

/-- Given a two-digit number XY where Y = 9, if 57 + X6 = 123, then XY = 69 -/
theorem misread_addition (X Y : Nat) : Y = 9 → 57 + (10 * X + 6) = 123 → 10 * X + Y = 69 := by
  sorry

end misread_addition_l2279_227916


namespace beautiful_point_coordinates_l2279_227950

/-- A point (x, y) is "beautiful" if x + y = x * y -/
def is_beautiful_point (x y : ℝ) : Prop := x + y = x * y

/-- The distance of a point (x, y) from the y-axis is |x| -/
def distance_from_y_axis (x : ℝ) : ℝ := |x|

theorem beautiful_point_coordinates :
  ∀ x y : ℝ, is_beautiful_point x y → distance_from_y_axis x = 2 →
  ((x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2/3)) :=
by sorry

end beautiful_point_coordinates_l2279_227950


namespace sum_of_20th_and_30th_triangular_l2279_227958

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of the 20th and 30th triangular numbers is 675 -/
theorem sum_of_20th_and_30th_triangular : triangular_number 20 + triangular_number 30 = 675 := by
  sorry

end sum_of_20th_and_30th_triangular_l2279_227958


namespace book_arrangements_count_l2279_227995

/-- The number of ways to arrange 8 different books (3 math, 3 foreign language, 2 literature)
    such that all math books are together and all foreign language books are together. -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 8
  let math_books : ℕ := 3
  let foreign_books : ℕ := 3
  let literature_books : ℕ := 2
  sorry

/-- Theorem stating that the number of book arrangements is 864. -/
theorem book_arrangements_count : book_arrangements = 864 := by
  sorry

end book_arrangements_count_l2279_227995


namespace sock_selection_theorem_l2279_227988

/-- Represents the number of socks of a given color -/
def num_socks : Fin 3 → Nat
  | 0 => 5  -- white
  | 1 => 5  -- brown
  | 2 => 3  -- blue

/-- Calculates the number of socks in odd positions for a given color -/
def odd_positions (color : Fin 3) : Nat :=
  (num_socks color + 1) / 2

/-- Calculates the number of socks in even positions for a given color -/
def even_positions (color : Fin 3) : Nat :=
  num_socks color / 2

/-- Calculates the number of ways to select a pair of socks of different colors from either odd or even positions -/
def select_pair_ways : Nat :=
  let white := 0
  let brown := 1
  let blue := 2
  (odd_positions white * odd_positions brown + even_positions white * even_positions brown) +
  (odd_positions brown * odd_positions blue + even_positions brown * even_positions blue) +
  (odd_positions white * odd_positions blue + even_positions white * even_positions blue)

/-- The main theorem stating that the number of ways to select a pair of socks is 29 -/
theorem sock_selection_theorem : select_pair_ways = 29 := by
  sorry

end sock_selection_theorem_l2279_227988


namespace smallest_base_representation_l2279_227991

/-- Given two bases a and b greater than 2, this function returns the base-10 
    representation of 21 in base a and 12 in base b. -/
def baseRepresentation (a b : ℕ) : ℕ := 2 * a + 1

/-- The smallest base-10 integer that can be represented as 21₍ₐ₎ in one base 
    and 12₍ᵦ₎ in another base, where a and b are any bases larger than 2. -/
def smallestInteger : ℕ := 7

theorem smallest_base_representation :
  ∀ a b : ℕ, a > 2 → b > 2 → 
  (baseRepresentation a b = baseRepresentation b a) → 
  (baseRepresentation a b ≥ smallestInteger) :=
by sorry

end smallest_base_representation_l2279_227991


namespace sphere_diameter_from_cylinder_l2279_227943

noncomputable def cylinder_volume (d h : ℝ) : ℝ := Real.pi * (d / 2)^2 * h

noncomputable def sphere_volume (d : ℝ) : ℝ := (4 / 3) * Real.pi * (d / 2)^3

theorem sphere_diameter_from_cylinder (cylinder_diameter cylinder_height : ℝ) :
  let total_volume := cylinder_volume cylinder_diameter cylinder_height
  let sphere_count := 9
  let individual_sphere_volume := total_volume / sphere_count
  let sphere_diameter := (6 * individual_sphere_volume / Real.pi)^(1/3)
  cylinder_diameter = 16 ∧ cylinder_height = 12 →
  sphere_diameter = 8 := by
  sorry

#check sphere_diameter_from_cylinder

end sphere_diameter_from_cylinder_l2279_227943


namespace arithmetic_sequence_reciprocal_S_general_term_formula_l2279_227963

def sequence_a (n : ℕ) : ℚ := sorry

def sum_S (n : ℕ) : ℚ := sorry

axiom a_1 : sequence_a 1 = 3

axiom relation_a_S (n : ℕ) : n ≥ 2 → 2 * sequence_a n = sum_S n * sum_S (n - 1)

theorem arithmetic_sequence_reciprocal_S :
  ∀ n : ℕ, n ≥ 2 → (1 / sum_S n - 1 / sum_S (n - 1) = -1 / 2) :=
sorry

theorem general_term_formula :
  ∀ n : ℕ, n ≥ 2 →
    sequence_a n = 18 / ((8 - 3 * n) * (5 - 3 * n)) :=
sorry

end arithmetic_sequence_reciprocal_S_general_term_formula_l2279_227963


namespace pen_selling_problem_l2279_227920

/-- Proves that given the conditions of the pen selling problem, the initial number of pens purchased is 30 -/
theorem pen_selling_problem (n : ℕ) (P : ℝ) (h1 : P > 0) :
  (∃ (S : ℝ), S > 0 ∧ 20 * S = P ∧ n * (2/3 * S) = P) →
  n = 30 := by
  sorry

end pen_selling_problem_l2279_227920


namespace shopkeeper_profit_percent_l2279_227924

theorem shopkeeper_profit_percent (initial_value : ℝ) (theft_percent : ℝ) (overall_loss_percent : ℝ) :
  theft_percent = 20 →
  overall_loss_percent = 12 →
  initial_value > 0 →
  let remaining_value := initial_value * (1 - theft_percent / 100)
  let selling_price := initial_value * (1 - overall_loss_percent / 100)
  let profit := selling_price - remaining_value
  let profit_percent := (profit / remaining_value) * 100
  profit_percent = 10 := by
sorry

end shopkeeper_profit_percent_l2279_227924


namespace f_is_even_and_decreasing_l2279_227951

-- Define the function f(x) = -x²
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end f_is_even_and_decreasing_l2279_227951


namespace M_intersect_N_eq_N_l2279_227971

def M : Set ℝ := {x | |x| ≥ 3}

def N : Set ℝ := {y | ∃ x ∈ M, y = x^2}

theorem M_intersect_N_eq_N : M ∩ N = N := by sorry

end M_intersect_N_eq_N_l2279_227971


namespace opposite_of_negative_2023_l2279_227987

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end opposite_of_negative_2023_l2279_227987


namespace consecutive_product_and_fourth_power_properties_l2279_227974

theorem consecutive_product_and_fourth_power_properties (c d m n : ℕ) : 
  (c * (c + 1) ≠ d * (d + 2)) ∧ 
  (m^4 + (m + 1)^4 ≠ n^2 + (n + 1)^2) := by
  sorry

end consecutive_product_and_fourth_power_properties_l2279_227974


namespace food_allocation_difference_l2279_227990

/-- Proves that the difference in food allocation between soldiers on the first and second sides is 2 pounds -/
theorem food_allocation_difference (
  soldiers_first : ℕ)
  (soldiers_second : ℕ)
  (food_per_soldier_first : ℝ)
  (total_food : ℝ)
  (h1 : soldiers_first = 4000)
  (h2 : soldiers_second = soldiers_first - 500)
  (h3 : food_per_soldier_first = 10)
  (h4 : total_food = 68000)
  (h5 : total_food = soldiers_first * food_per_soldier_first + 
    soldiers_second * (food_per_soldier_first - (food_per_soldier_first - food_per_soldier_second)))
  : food_per_soldier_first - food_per_soldier_second = 2 := by
  sorry

end food_allocation_difference_l2279_227990


namespace toms_robot_collection_l2279_227956

/-- Represents the number of robots of each type for a person -/
structure RobotCollection where
  animal : ℕ
  humanoid : ℕ
  vehicle : ℕ

/-- Given the conditions of the problem, prove that Tom's robot collection matches the expected values -/
theorem toms_robot_collection (michael : RobotCollection) (tom : RobotCollection) : 
  michael.animal = 8 ∧ 
  michael.humanoid = 12 ∧ 
  michael.vehicle = 20 ∧
  tom.animal = 2 * michael.animal ∧
  tom.humanoid = (3 : ℕ) / 2 * michael.humanoid ∧
  michael.vehicle = (5 : ℕ) / 4 * tom.vehicle →
  tom.animal = 16 ∧ tom.humanoid = 18 ∧ tom.vehicle = 16 := by
  sorry

end toms_robot_collection_l2279_227956


namespace quadratic_solutions_fractional_solution_l2279_227980

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x + 1 = 0

-- Define the fractional equation
def fractional_equation (x : ℝ) : Prop := (4*x)/(x-2) - 1 = 3/(2-x)

-- Theorem for the quadratic equation solutions
theorem quadratic_solutions :
  ∃ x1 x2 : ℝ, 
    quadratic_equation x1 ∧ 
    quadratic_equation x2 ∧ 
    x1 = (-3 + Real.sqrt 5) / 2 ∧ 
    x2 = (-3 - Real.sqrt 5) / 2 :=
sorry

-- Theorem for the fractional equation solution
theorem fractional_solution :
  ∃ x : ℝ, fractional_equation x ∧ x = -5/3 :=
sorry

end quadratic_solutions_fractional_solution_l2279_227980


namespace area_of_curve_l2279_227972

/-- The curve defined by x^2 + y^2 = |x| + 2|y| -/
def curve (x y : ℝ) : Prop := x^2 + y^2 = |x| + 2 * |y|

/-- The area enclosed by the curve -/
noncomputable def enclosed_area : ℝ := sorry

theorem area_of_curve : enclosed_area = (5 * π) / 4 := by sorry

end area_of_curve_l2279_227972


namespace brick_wall_pattern_l2279_227934

/-- Represents a brick wall with a given number of rows and bricks -/
structure BrickWall where
  rows : ℕ
  total_bricks : ℕ
  bottom_row_bricks : ℕ

/-- Calculates the number of bricks in a given row -/
def bricks_in_row (wall : BrickWall) (row : ℕ) : ℕ :=
  wall.bottom_row_bricks - (row - 1)

theorem brick_wall_pattern (wall : BrickWall) 
  (h1 : wall.rows = 5)
  (h2 : wall.total_bricks = 50)
  (h3 : wall.bottom_row_bricks = 8) :
  ∀ row : ℕ, 1 < row → row ≤ wall.rows → 
    bricks_in_row wall row = bricks_in_row wall (row - 1) - 1 :=
by sorry

end brick_wall_pattern_l2279_227934


namespace cistern_fill_fraction_l2279_227962

/-- Represents the time in minutes it takes to fill a portion of the cistern -/
def fill_time : ℝ := 35

/-- Represents the fraction of the cistern filled in the given time -/
def fraction_filled : ℝ := 1

/-- Proves that the fraction of the cistern filled is 1 given the conditions -/
theorem cistern_fill_fraction :
  (fill_time = 35) → fraction_filled = 1 := by
  sorry

end cistern_fill_fraction_l2279_227962


namespace zoey_holidays_per_month_l2279_227938

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total number of holidays Zoey took in a year -/
def total_holidays : ℕ := 24

/-- Zoey took holidays every month for an entire year -/
axiom holidays_every_month : ∀ (month : ℕ), month ≤ months_in_year → ∃ (holidays : ℕ), holidays > 0

/-- The number of holidays Zoey took each month -/
def holidays_per_month : ℚ := total_holidays / months_in_year

theorem zoey_holidays_per_month : holidays_per_month = 2 := by sorry

end zoey_holidays_per_month_l2279_227938


namespace gcd_lcm_product_l2279_227996

theorem gcd_lcm_product (a b : ℕ) (ha : a = 180) (hb : b = 225) :
  (Nat.gcd a b) * (Nat.lcm a b) = 40500 := by
  sorry

end gcd_lcm_product_l2279_227996


namespace simultaneous_equations_solution_l2279_227936

theorem simultaneous_equations_solution (k : ℝ) :
  (k ≠ 1) ↔ (∃ x y : ℝ, (y = k * x + 2) ∧ (y = (3 * k - 2) * x + 5)) :=
by sorry

end simultaneous_equations_solution_l2279_227936


namespace fraction_expressions_l2279_227954

theorem fraction_expressions (x z : ℚ) (h : x / z = 5 / 6) :
  ((x + 3 * z) / z = 23 / 6) ∧
  (z / (x - z) = -6) ∧
  ((2 * x + z) / z = 8 / 3) ∧
  (3 * x / (4 * z) = 5 / 8) ∧
  ((x - 2 * z) / z = -7 / 6) := by
  sorry

end fraction_expressions_l2279_227954


namespace mork_tax_rate_l2279_227908

theorem mork_tax_rate (mork_income : ℝ) (mork_rate : ℝ) : 
  mork_income > 0 →
  mork_rate * mork_income + 0.2 * (4 * mork_income) = 0.25 * (5 * mork_income) →
  mork_rate = 0.45 := by
sorry

end mork_tax_rate_l2279_227908


namespace oil_purchase_calculation_l2279_227994

theorem oil_purchase_calculation (tank_capacity : ℕ) (tanks_needed : ℕ) (total_oil : ℕ) : 
  tank_capacity = 32 → tanks_needed = 23 → total_oil = tank_capacity * tanks_needed → total_oil = 736 :=
by sorry

end oil_purchase_calculation_l2279_227994


namespace trajectory_of_M_l2279_227946

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define point A
def point_A : ℝ × ℝ := (1, 0)

-- Define a point Q on the circle
def point_Q (x y : ℝ) : Prop := circle_C x y

-- Define point M as the intersection of the perpendicular bisector of AQ and CQ
def point_M (x y : ℝ) : Prop :=
  ∃ (qx qy : ℝ), point_Q qx qy ∧
  (x - 1)^2 + y^2 = (x - qx)^2 + (y - qy)^2 ∧
  (x + qx = 1) ∧ (y + qy = 0)

-- Theorem statement
theorem trajectory_of_M :
  ∀ (x y : ℝ), point_M x y → x^2/4 + y^2/3 = 1 :=
sorry

end trajectory_of_M_l2279_227946


namespace train_time_difference_l2279_227911

def distance : ℝ := 425.80645161290323
def speed_slow : ℝ := 44
def speed_fast : ℝ := 75

theorem train_time_difference :
  (distance / speed_slow) - (distance / speed_fast) = 4 := by
  sorry

end train_time_difference_l2279_227911


namespace gcd_6363_1923_l2279_227926

theorem gcd_6363_1923 : Nat.gcd 6363 1923 = 3 := by
  sorry

end gcd_6363_1923_l2279_227926


namespace geometric_sequence_middle_term_l2279_227929

theorem geometric_sequence_middle_term (χ : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ -1 * r = χ ∧ χ * r = -4) → χ = 2 ∨ χ = -2 := by
  sorry

end geometric_sequence_middle_term_l2279_227929


namespace triangle_angle_and_parameter_l2279_227968

/-- Given a triangle ABC where tan A and tan B are real roots of a quadratic equation,
    prove that angle C is 60° and find the value of p. -/
theorem triangle_angle_and_parameter
  (A B C : ℝ) (p : ℝ) (AB AC : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (h_roots : ∃ (x y : ℝ), x^2 + Real.sqrt 3 * p * x - p + 1 = 0 ∧
                          y^2 + Real.sqrt 3 * p * y - p + 1 = 0 ∧
                          x = Real.tan A ∧ y = Real.tan B)
  (h_AB : AB = 3)
  (h_AC : AC = Real.sqrt 6) :
  C = Real.pi / 3 ∧ p = -1 - Real.sqrt 3 := by
  sorry

end triangle_angle_and_parameter_l2279_227968


namespace function_inequality_implies_a_range_l2279_227983

theorem function_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * 9^x - 3^x + a^2 - a - 3 > 0) → 
  (a > 2 ∨ a < -1) := by
  sorry

end function_inequality_implies_a_range_l2279_227983


namespace james_streaming_income_l2279_227966

/-- James' streaming income calculation --/
theorem james_streaming_income 
  (initial_subscribers : ℕ) 
  (gifted_subscribers : ℕ) 
  (income_per_subscriber : ℕ) : ℕ :=
  by
  have total_subscribers : ℕ := initial_subscribers + gifted_subscribers
  have monthly_income : ℕ := total_subscribers * income_per_subscriber
  exact monthly_income

#check james_streaming_income 150 50 9

end james_streaming_income_l2279_227966


namespace remainder_4n_squared_mod_13_l2279_227901

theorem remainder_4n_squared_mod_13 (n : ℤ) (h : n % 13 = 7) : (4 * n^2) % 13 = 1 := by
  sorry

end remainder_4n_squared_mod_13_l2279_227901


namespace park_outer_boundary_diameter_l2279_227981

/-- The diameter of the outer boundary of a circular park structure -/
def outer_boundary_diameter (pond_diameter : ℝ) (picnic_width : ℝ) (track_width : ℝ) : ℝ :=
  pond_diameter + 2 * (picnic_width + track_width)

/-- Theorem stating the diameter of the outer boundary of the cycling track -/
theorem park_outer_boundary_diameter :
  outer_boundary_diameter 16 10 4 = 44 := by
  sorry

end park_outer_boundary_diameter_l2279_227981


namespace triangle_side_count_l2279_227919

/-- The number of integer values for the third side of a triangle with sides 15 and 40 -/
def triangleSideCount : ℕ := by
  sorry

theorem triangle_side_count :
  triangleSideCount = 29 := by
  sorry

end triangle_side_count_l2279_227919


namespace health_risk_factors_l2279_227930

theorem health_risk_factors (total_population : ℝ) 
  (prob_single : ℝ) (prob_pair : ℝ) (prob_all_given_two : ℝ) :
  prob_single = 0.08 →
  prob_pair = 0.15 →
  prob_all_given_two = 1/4 →
  ∃ (prob_none_given_not_one : ℝ),
    prob_none_given_not_one = 26/57 := by
  sorry

end health_risk_factors_l2279_227930


namespace quadratic_is_square_of_binomial_l2279_227955

theorem quadratic_is_square_of_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 12*x + a = (3*x + b)^2) → a = 4 := by
  sorry

end quadratic_is_square_of_binomial_l2279_227955


namespace problem_solid_surface_area_l2279_227925

/-- Represents a solid formed by unit cubes -/
structure CubeSolid where
  base_length : ℕ
  base_width : ℕ
  top_length : ℕ
  top_width : ℕ
  base_height : ℕ := 1
  top_height : ℕ := 1

/-- Calculates the surface area of the CubeSolid -/
def surface_area (solid : CubeSolid) : ℕ :=
  let base_area := solid.base_length * solid.base_width
  let top_area := solid.top_length * solid.top_width
  let exposed_base := 2 * base_area
  let exposed_sides := 2 * (solid.base_length * solid.base_height + solid.base_width * solid.base_height)
  let exposed_top := base_area - top_area + top_area
  let exposed_top_sides := 2 * (solid.top_length * solid.top_height + solid.top_width * solid.top_height)
  exposed_base + exposed_sides + exposed_top + exposed_top_sides

/-- The specific solid described in the problem -/
def problem_solid : CubeSolid := {
  base_length := 4
  base_width := 2
  top_length := 2
  top_width := 2
}

theorem problem_solid_surface_area :
  surface_area problem_solid = 36 := by
  sorry

end problem_solid_surface_area_l2279_227925


namespace equation_solution_l2279_227961

theorem equation_solution : ∃ x : ℝ, (x - 1) / 2 = 1 - (x + 2) / 3 ∧ x = 1 := by
  sorry

end equation_solution_l2279_227961


namespace no_prime_roots_for_quadratic_l2279_227941

theorem no_prime_roots_for_quadratic : 
  ¬ ∃ (k : ℤ), ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    p ≠ q ∧
    (p : ℤ) + q = 107 ∧ 
    (p : ℤ) * q = k ∧
    p^2 - 107*p + k = 0 ∧ 
    q^2 - 107*q + k = 0 :=
sorry

end no_prime_roots_for_quadratic_l2279_227941


namespace mod_eleven_problem_l2279_227935

theorem mod_eleven_problem : ∃ n : ℕ, 0 ≤ n ∧ n < 11 ∧ 1234 % 11 = n := by
  sorry

end mod_eleven_problem_l2279_227935


namespace strawberry_area_l2279_227949

theorem strawberry_area (garden_size : ℝ) (fruit_ratio : ℝ) (strawberry_ratio : ℝ) : 
  garden_size = 64 → 
  fruit_ratio = 1/2 → 
  strawberry_ratio = 1/4 → 
  garden_size * fruit_ratio * strawberry_ratio = 8 := by
sorry

end strawberry_area_l2279_227949


namespace cos_sum_17th_roots_unity_l2279_227948

theorem cos_sum_17th_roots_unity :
  Real.cos (2 * Real.pi / 17) + Real.cos (8 * Real.pi / 17) + Real.cos (14 * Real.pi / 17) = (Real.sqrt 17 - 1) / 4 := by
  sorry

end cos_sum_17th_roots_unity_l2279_227948


namespace rectangle_area_ratio_l2279_227973

structure Rectangle where
  width : ℝ
  height : ℝ

def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r2.width = r1.height / r2.height

theorem rectangle_area_ratio 
  (ABCD EFGH : Rectangle) 
  (h1 : similar ABCD EFGH) 
  (h2 : ∃ (K : ℝ), K > 0 ∧ K < ABCD.width ∧ (ABCD.width - K) / K = 2 / 3) : 
  (ABCD.width * ABCD.height) / (EFGH.width * EFGH.height) = 9 / 4 := by
  sorry

end rectangle_area_ratio_l2279_227973


namespace smallest_positive_integer_ending_in_6_divisible_by_5_l2279_227969

theorem smallest_positive_integer_ending_in_6_divisible_by_5 :
  ∃ (n : ℕ), n > 0 ∧ n % 10 = 6 ∧ n % 5 = 0 ∧
  ∀ (m : ℕ), m > 0 → m % 10 = 6 → m % 5 = 0 → m ≥ n :=
by
  use 46
  sorry

end smallest_positive_integer_ending_in_6_divisible_by_5_l2279_227969


namespace quadratic_function_properties_l2279_227940

/-- Given a quadratic function y = x^2 - 2px - p with two distinct roots, 
    prove properties about p and the roots. -/
theorem quadratic_function_properties (p : ℝ) 
  (h_distinct : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 2*p*x₁ - p = 0 ∧ x₂^2 - 2*p*x₂ - p = 0) :
  (∃ (x₁ x₂ : ℝ), 2*p*x₁ + x₂^2 + 3*p > 0) ∧
  (∃ (max_p : ℝ), max_p = 9/16 ∧ 
    ∀ (q : ℝ), (∃ (x₁ x₂ : ℝ), x₁^2 - 2*q*x₁ - q = 0 ∧ x₂^2 - 2*q*x₂ - q = 0 ∧ |x₁ - x₂| ≤ |2*q - 3|) 
    → q ≤ max_p) := by
  sorry

end quadratic_function_properties_l2279_227940


namespace square_card_arrangement_l2279_227914

theorem square_card_arrangement (perimeter_cards : ℕ) (h : perimeter_cards = 240) : 
  ∃ (side_length : ℕ), 
    4 * side_length - 4 = perimeter_cards ∧ 
    side_length * side_length = 3721 := by
  sorry

end square_card_arrangement_l2279_227914


namespace alloy_ratio_theorem_l2279_227993

/-- Represents an alloy with zinc and copper -/
structure Alloy where
  zinc : ℚ
  copper : ℚ

/-- The first alloy with zinc:copper ratio of 1:2 -/
def alloy1 : Alloy := { zinc := 1, copper := 2 }

/-- The second alloy with zinc:copper ratio of 2:3 -/
def alloy2 : Alloy := { zinc := 2, copper := 3 }

/-- The desired third alloy with zinc:copper ratio of 17:27 -/
def alloy3 : Alloy := { zinc := 17, copper := 27 }

/-- Theorem stating the ratio of alloys needed to create the third alloy -/
theorem alloy_ratio_theorem :
  ∃ (x y : ℚ),
    x > 0 ∧ y > 0 ∧
    (x * alloy1.zinc + y * alloy2.zinc) / (x * alloy1.copper + y * alloy2.copper) = alloy3.zinc / alloy3.copper ∧
    x / y = 9 / 35 := by
  sorry


end alloy_ratio_theorem_l2279_227993


namespace alyssa_final_money_l2279_227989

def weekly_allowance : ℕ := 8
def movie_spending : ℕ := weekly_allowance / 2
def car_wash_earnings : ℕ := 8

theorem alyssa_final_money :
  weekly_allowance - movie_spending + car_wash_earnings = 12 :=
by sorry

end alyssa_final_money_l2279_227989


namespace combined_savings_equal_individual_savings_problem_specific_savings_l2279_227928

/-- Represents the store's window offer -/
structure WindowOffer where
  normalPrice : ℕ
  freeWindowsPer : ℕ

/-- Calculates the cost of purchasing windows under the offer -/
def costUnderOffer (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let paidWindows := windowsNeeded - (windowsNeeded / (offer.freeWindowsPer + 1))
  paidWindows * offer.normalPrice

/-- Calculates the savings for a given number of windows -/
def savings (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  windowsNeeded * offer.normalPrice - costUnderOffer offer windowsNeeded

/-- Theorem: The combined savings equal the sum of individual savings -/
theorem combined_savings_equal_individual_savings 
  (offer : WindowOffer) 
  (daveWindows : ℕ) 
  (dougWindows : ℕ) : 
  savings offer (daveWindows + dougWindows) = 
    savings offer daveWindows + savings offer dougWindows :=
by sorry

/-- The specific offer in the problem -/
def storeOffer : WindowOffer := { normalPrice := 100, freeWindowsPer := 3 }

/-- Theorem: For Dave (9 windows) and Doug (6 windows), 
    the combined savings equal the sum of their individual savings -/
theorem problem_specific_savings : 
  savings storeOffer (9 + 6) = savings storeOffer 9 + savings storeOffer 6 :=
by sorry

end combined_savings_equal_individual_savings_problem_specific_savings_l2279_227928


namespace negative_sixty_four_to_four_thirds_l2279_227903

theorem negative_sixty_four_to_four_thirds (x : ℝ) : x = (-64)^(4/3) → x = 256 := by
  sorry

end negative_sixty_four_to_four_thirds_l2279_227903


namespace polynomial_division_remainder_l2279_227979

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (X^5 - 1) * (X^3 - 1) = (X^3 + X^2 + X + 1) * q + (2*X + 2) := by
  sorry

end polynomial_division_remainder_l2279_227979


namespace sam_study_time_l2279_227997

theorem sam_study_time (total_hours : ℕ) (science_minutes : ℕ) (literature_minutes : ℕ) 
  (h1 : total_hours = 3)
  (h2 : science_minutes = 60)
  (h3 : literature_minutes = 40) :
  total_hours * 60 - (science_minutes + literature_minutes) = 80 :=
by sorry

end sam_study_time_l2279_227997


namespace wall_height_calculation_l2279_227959

/-- Given a brick and wall with specified dimensions, prove the height of the wall --/
theorem wall_height_calculation (brick_length brick_width brick_height : Real)
  (wall_length wall_width : Real) (num_bricks : Nat) :
  brick_length = 0.20 →
  brick_width = 0.10 →
  brick_height = 0.075 →
  wall_length = 25 →
  wall_width = 0.75 →
  num_bricks = 25000 →
  ∃ (wall_height : Real),
    wall_height = 2 ∧
    num_bricks * (brick_length * brick_width * brick_height) = wall_length * wall_width * wall_height :=
by sorry

end wall_height_calculation_l2279_227959


namespace game_preparation_time_l2279_227905

/-- The time taken to prepare all games is 150 minutes, given that each game takes 10 minutes to prepare and Andrew prepared 15 games. -/
theorem game_preparation_time : 
  let time_per_game : ℕ := 10
  let total_games : ℕ := 15
  let total_time := time_per_game * total_games
  total_time = 150 := by sorry

end game_preparation_time_l2279_227905


namespace dog_human_age_difference_l2279_227933

/-- The ratio of dog years to human years -/
def dogYearRatio : ℕ := 7

/-- Calculates the age difference in dog years between a dog and a human of the same age in human years -/
def ageDifferenceInDogYears (humanAge : ℕ) : ℕ :=
  humanAge * dogYearRatio - humanAge

/-- Theorem stating that for a 3-year-old human and their 3-year-old dog, 
    the dog will be 18 years older in dog years -/
theorem dog_human_age_difference : ageDifferenceInDogYears 3 = 18 := by
  sorry

end dog_human_age_difference_l2279_227933


namespace fractional_equation_root_l2279_227978

theorem fractional_equation_root (m : ℝ) : 
  (∃ x : ℝ, x ≠ 2 ∧ x ≠ 2 ∧ (3 / (x - 2) + 1 = m / (4 - 2*x))) → m = -6 :=
by sorry

end fractional_equation_root_l2279_227978


namespace equation_value_l2279_227964

theorem equation_value (x y z w : ℝ) 
  (h1 : 4 * x * z + y * w = 3)
  (h2 : (2 * x + y) * (2 * z + w) = 15) :
  x * w + y * z = 6 := by
sorry

end equation_value_l2279_227964


namespace student_weight_l2279_227913

theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight + sister_weight = 132)
  (h2 : student_weight - 6 = 2 * sister_weight) : 
  student_weight = 90 := by
  sorry

end student_weight_l2279_227913


namespace cable_cost_per_roommate_l2279_227917

/-- Represents the cost structure of the cable program --/
structure CableProgram where
  tier1_channels : Nat
  tier1_cost : ℚ
  tier2_channels : Nat
  tier2_cost : ℚ
  tier3_channels : Nat
  tier3_cost_ratio : ℚ
  tier4_channels : Nat
  tier4_cost_ratio : ℚ

/-- Calculates the total cost of the cable program --/
def total_cost (program : CableProgram) : ℚ :=
  program.tier1_cost +
  program.tier2_cost +
  (program.tier2_cost * program.tier3_cost_ratio * program.tier3_channels / program.tier2_channels) +
  (program.tier2_cost * program.tier3_cost_ratio * (1 + program.tier4_cost_ratio) * program.tier4_channels / program.tier2_channels)

/-- Theorem stating that each roommate pays $81.25 --/
theorem cable_cost_per_roommate (program : CableProgram)
  (h1 : program.tier1_channels = 100)
  (h2 : program.tier1_cost = 100)
  (h3 : program.tier2_channels = 100)
  (h4 : program.tier2_cost = 75)
  (h5 : program.tier3_channels = 150)
  (h6 : program.tier3_cost_ratio = 1/2)
  (h7 : program.tier4_channels = 200)
  (h8 : program.tier4_cost_ratio = 1/4)
  (h9 : program.tier2_channels = 100) :
  (total_cost program) / 4 = 325 / 4 := by
  sorry

#eval (325 : ℚ) / 4

end cable_cost_per_roommate_l2279_227917


namespace no_brownies_left_l2279_227985

/-- Represents the number of brownies left after consumption --/
def brownies_left (total : ℚ) (tina_lunch : ℚ) (tina_dinner : ℚ) (husband : ℚ) (guests : ℚ) (daughter : ℚ) : ℚ :=
  total - (5 * (tina_lunch + tina_dinner) + 5 * husband + 2 * guests + 3 * daughter)

/-- Theorem stating that no brownies are left after consumption --/
theorem no_brownies_left : 
  brownies_left 24 1.5 0.5 0.75 2.5 2 = 0 := by
  sorry

#eval brownies_left 24 1.5 0.5 0.75 2.5 2

end no_brownies_left_l2279_227985


namespace waiter_customers_waiter_customers_proof_l2279_227939

theorem waiter_customers : ℕ → Prop :=
  fun initial_customers =>
    initial_customers - 14 + 36 = 41 →
    initial_customers = 19

-- The proof of the theorem
theorem waiter_customers_proof : ∃ x : ℕ, waiter_customers x :=
  sorry

end waiter_customers_waiter_customers_proof_l2279_227939
