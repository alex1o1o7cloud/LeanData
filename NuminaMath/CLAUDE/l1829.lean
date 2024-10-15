import Mathlib

namespace NUMINAMATH_CALUDE_diana_operations_l1829_182906

theorem diana_operations (x : ℝ) : 
  (((x + 3) * 3 - 3) / 3 + 3 = 12) → x = 7 := by
sorry

end NUMINAMATH_CALUDE_diana_operations_l1829_182906


namespace NUMINAMATH_CALUDE_four_solutions_to_simultaneous_equations_l1829_182905

theorem four_solutions_to_simultaneous_equations :
  ∃! (s : Finset (ℝ × ℝ)), (∀ (p : ℝ × ℝ), p ∈ s ↔ p.1^2 - p.2 = 2022 ∧ p.2^2 - p.1 = 2022) ∧ s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_solutions_to_simultaneous_equations_l1829_182905


namespace NUMINAMATH_CALUDE_smallest_a_for_distinct_roots_in_unit_interval_l1829_182963

theorem smallest_a_for_distinct_roots_in_unit_interval :
  ∃ (b c : ℤ), 
    (∃ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧ 
      5 * x^2 - b * x + c = 0 ∧ 5 * y^2 - b * y + c = 0) ∧
    (∀ (a : ℕ), a < 5 → 
      ¬∃ (b c : ℤ), ∃ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧ 
        a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_distinct_roots_in_unit_interval_l1829_182963


namespace NUMINAMATH_CALUDE_circumscribed_circle_condition_l1829_182933

/-- Two lines forming a quadrilateral with coordinate axes that has a circumscribed circle -/
def has_circumscribed_circle (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    ((a + 2) * x + (1 - a) * y - 3 = 0) ∧
    ((a - 1) * x + (2 * a + 3) * y + 2 = 0) ∧
    (x ≥ 0 ∧ y ≥ 0)

/-- Theorem stating the condition for the quadrilateral to have a circumscribed circle -/
theorem circumscribed_circle_condition (a : ℝ) :
  has_circumscribed_circle a → (a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_condition_l1829_182933


namespace NUMINAMATH_CALUDE_average_age_of_women_l1829_182946

/-- The average age of four women given the following conditions:
    - There are 15 men initially.
    - The average age of 15 men is 40 years.
    - Four men of ages 26, 32, 41, and 39 years are replaced by four women.
    - The new average age increases by 2.9 years after the replacement. -/
theorem average_age_of_women (
  initial_men : ℕ)
  (initial_avg_age : ℝ)
  (replaced_men_ages : Fin 4 → ℝ)
  (new_avg_increase : ℝ)
  (h1 : initial_men = 15)
  (h2 : initial_avg_age = 40)
  (h3 : replaced_men_ages = ![26, 32, 41, 39])
  (h4 : new_avg_increase = 2.9)
  : (initial_men * initial_avg_age + 4 * new_avg_increase * initial_men - (replaced_men_ages 0 + replaced_men_ages 1 + replaced_men_ages 2 + replaced_men_ages 3)) / 4 = 45.375 := by
  sorry


end NUMINAMATH_CALUDE_average_age_of_women_l1829_182946


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1829_182974

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The roots of the quadratic equation 3x^2 - 2x - 6 = 0 -/
def are_roots (x y : ℝ) : Prop :=
  3 * x^2 - 2 * x - 6 = 0 ∧ 3 * y^2 - 2 * y - 6 = 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  are_roots (a 1) (a 10) →
  a 4 * a 7 = -2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1829_182974


namespace NUMINAMATH_CALUDE_union_M_N_intersect_N_complement_M_l1829_182911

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x)}

-- Theorem for M ∪ N
theorem union_M_N : M ∪ N = {x | x ≤ 2} := by sorry

-- Theorem for N ∩ (∁ᵤM)
theorem intersect_N_complement_M : N ∩ (U \ M) = {x | x < -2} := by sorry

end NUMINAMATH_CALUDE_union_M_N_intersect_N_complement_M_l1829_182911


namespace NUMINAMATH_CALUDE_line_circle_intersection_and_dot_product_l1829_182966

-- Define the line l passing through A(0, 1) with slope k
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1}

-- Define the circle C: (x-2)^2+(y-3)^2=1
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

-- Define the point A
def point_A : ℝ × ℝ := (0, 1)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem line_circle_intersection_and_dot_product
  (k : ℝ) (M N : ℝ × ℝ) :
  (M ∈ line_l k ∧ M ∈ circle_C ∧ N ∈ line_l k ∧ N ∈ circle_C) →
  ((4 - Real.sqrt 7) / 3 < k ∧ k < (4 + Real.sqrt 7) / 3) ∧
  (dot_product (M.1 - point_A.1, M.2 - point_A.2) (N.1 - point_A.1, N.2 - point_A.2) = 7) ∧
  (dot_product (M.1 - origin.1, M.2 - origin.2) (N.1 - origin.1, N.2 - origin.2) = 12 →
    k = 1 ∧ line_l k = {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_and_dot_product_l1829_182966


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1829_182976

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = b) → ¬(a^2 - b^2 = 0)) ↔ (a^2 - b^2 = 0 → a = b) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1829_182976


namespace NUMINAMATH_CALUDE_abcd_imag_zero_l1829_182902

open Complex

-- Define the condition for angles being equal and oppositely oriented
def anglesEqualOpposite (a b c d : ℂ) : Prop :=
  ∃ θ : ℝ, b / a = exp (θ * I) ∧ d / c = exp (-θ * I)

theorem abcd_imag_zero (a b c d : ℂ) 
  (h : anglesEqualOpposite a b c d) : 
  (a * b * c * d).im = 0 := by
  sorry

end NUMINAMATH_CALUDE_abcd_imag_zero_l1829_182902


namespace NUMINAMATH_CALUDE_collins_savings_l1829_182924

def cans_per_dollar : ℚ := 4

def cans_at_home : ℕ := 12
def cans_at_grandparents : ℕ := 3 * cans_at_home
def cans_from_neighbor : ℕ := 46
def cans_from_office : ℕ := 250

def total_cans : ℕ := cans_at_home + cans_at_grandparents + cans_from_neighbor + cans_from_office

def total_money : ℚ := (total_cans : ℚ) / cans_per_dollar

def savings_amount : ℚ := total_money / 2

theorem collins_savings : savings_amount = 43 := by sorry

end NUMINAMATH_CALUDE_collins_savings_l1829_182924


namespace NUMINAMATH_CALUDE_kim_shoe_pairs_l1829_182912

/-- The number of shoes Kim has -/
def total_shoes : ℕ := 18

/-- The probability of selecting two shoes of the same color -/
def probability : ℚ := 58823529411764705 / 1000000000000000000

/-- The number of pairs of shoes Kim has -/
def num_pairs : ℕ := total_shoes / 2

theorem kim_shoe_pairs :
  (probability = 1 / (total_shoes - 1)) → num_pairs = 9 := by
  sorry

end NUMINAMATH_CALUDE_kim_shoe_pairs_l1829_182912


namespace NUMINAMATH_CALUDE_point_symmetric_to_origin_l1829_182969

theorem point_symmetric_to_origin (a : ℝ) : 
  let P : ℝ × ℝ := (2 - a, 3 * a + 6)
  (|2 - a| = |3 * a + 6|) → 
  (∃ (x y : ℝ), (x = -3 ∧ y = -3) ∨ (x = -6 ∧ y = 6)) ∧ 
  ((-(2 - a), -(3 * a + 6)) = (x, y)) :=
by sorry

end NUMINAMATH_CALUDE_point_symmetric_to_origin_l1829_182969


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1829_182941

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℕ+, (x ≤ 4 → a * x.val + 4 ≥ 0) ∧ (x > 4 → a * x.val + 4 < 0)) → 
  -1 ≤ a ∧ a < -4/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1829_182941


namespace NUMINAMATH_CALUDE_visible_red_bus_length_l1829_182983

/-- Proves that the visible length of a red bus from a yellow bus is 6 feet, given specific length relationships between red, orange, and yellow buses. -/
theorem visible_red_bus_length 
  (red_bus_length : ℝ)
  (orange_car_length : ℝ)
  (yellow_bus_length : ℝ)
  (h1 : red_bus_length = 4 * orange_car_length)
  (h2 : yellow_bus_length = 3.5 * orange_car_length)
  (h3 : red_bus_length = 48) :
  red_bus_length - yellow_bus_length = 6 := by
  sorry

#check visible_red_bus_length

end NUMINAMATH_CALUDE_visible_red_bus_length_l1829_182983


namespace NUMINAMATH_CALUDE_equation_solution_l1829_182939

theorem equation_solution : ∃! x : ℝ, (4 : ℝ) ^ (x + 6) = 64 ^ x :=
  have h : (4 : ℝ) ^ (3 + 6) = 64 ^ 3 := by sorry
  ⟨3, h, λ y hy => by sorry⟩

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l1829_182939


namespace NUMINAMATH_CALUDE_tank_weight_l1829_182973

/-- Given a tank with the following properties:
  * When four-fifths full, it weighs p kilograms
  * When two-thirds full, it weighs q kilograms
  * The empty tank and other contents weigh r kilograms
  Prove that the total weight of the tank when completely full is (5/2)p + (3/2)q -/
theorem tank_weight (p q r : ℝ) : 
  (∃ (x y : ℝ), x + (4/5) * y = p ∧ x + (2/3) * y = q ∧ x = r) →
  (∃ (z : ℝ), z = (5/2) * p + (3/2) * q ∧ 
    (∀ (x y : ℝ), x + (4/5) * y = p ∧ x + (2/3) * y = q → x + y = z)) :=
by sorry

end NUMINAMATH_CALUDE_tank_weight_l1829_182973


namespace NUMINAMATH_CALUDE_saras_baking_days_l1829_182964

/-- Proves the number of weekdays Sara makes cakes given the problem conditions -/
theorem saras_baking_days (cakes_per_day : ℕ) (price_per_cake : ℕ) (total_collected : ℕ) 
  (h1 : cakes_per_day = 4)
  (h2 : price_per_cake = 8)
  (h3 : total_collected = 640) :
  total_collected / price_per_cake / cakes_per_day = 20 := by
  sorry

end NUMINAMATH_CALUDE_saras_baking_days_l1829_182964


namespace NUMINAMATH_CALUDE_valid_numbers_characterization_l1829_182930

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (10 * n) % 7 = 0 ∧
  (n / 1000 * 10000 + (n % 1000)) % 7 = 0 ∧
  (n / 100 * 1000 + (n % 100) + (n / 1000 * 10000)) % 7 = 0 ∧
  (n / 10 * 100 + (n % 10) + (n / 100 * 10000)) % 7 = 0 ∧
  (n * 10 + (n / 1000)) % 7 = 0

theorem valid_numbers_characterization :
  {n : ℕ | is_valid_number n} = {7000, 7007, 7070, 7077, 7700, 7707, 7770, 7777} := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_characterization_l1829_182930


namespace NUMINAMATH_CALUDE_negation_of_all_men_honest_l1829_182961

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a man and being honest
variable (man : U → Prop)
variable (honest : U → Prop)

-- State the theorem
theorem negation_of_all_men_honest :
  (¬ ∀ x, man x → honest x) ↔ (∃ x, man x ∧ ¬ honest x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_men_honest_l1829_182961


namespace NUMINAMATH_CALUDE_gamma_delta_sum_l1829_182915

theorem gamma_delta_sum : 
  ∃ (γ δ : ℝ), ∀ x : ℝ, (x - γ) / (x + δ) = (x^2 - 90*x + 1980) / (x^2 + 60*x - 3240) → 
  γ + δ = 140 := by
  sorry

end NUMINAMATH_CALUDE_gamma_delta_sum_l1829_182915


namespace NUMINAMATH_CALUDE_second_number_approximation_l1829_182909

theorem second_number_approximation (x y z : ℝ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 7 / 9)
  (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0) : 
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 1 ∧ y = 40 + ε :=
sorry

end NUMINAMATH_CALUDE_second_number_approximation_l1829_182909


namespace NUMINAMATH_CALUDE_inequality_solution_l1829_182913

theorem inequality_solution : 
  {x : ℝ | 5*x > 4*x + 2} = {x : ℝ | x > 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1829_182913


namespace NUMINAMATH_CALUDE_convex_pentagon_probability_l1829_182989

/-- The number of points on the circle -/
def n : ℕ := 7

/-- The number of chords to be selected -/
def k : ℕ := 5

/-- The total number of chords possible with n points -/
def total_chords : ℕ := n.choose 2

/-- The total number of ways to select k chords from total_chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def favorable_outcomes : ℕ := n.choose k

/-- The probability of k randomly selected chords from n points on a circle forming a convex polygon -/
def probability : ℚ := favorable_outcomes / total_selections

theorem convex_pentagon_probability :
  n = 7 ∧ k = 5 → probability = 1 / 969 := by
  sorry

end NUMINAMATH_CALUDE_convex_pentagon_probability_l1829_182989


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l1829_182959

theorem junk_mail_distribution (total_mail : ℕ) (total_houses : ℕ) 
  (h1 : total_mail = 48) (h2 : total_houses = 8) :
  total_mail / total_houses = 6 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l1829_182959


namespace NUMINAMATH_CALUDE_equal_angles_implies_rectangle_l1829_182900

-- Define a quadrilateral
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

-- Define the concept of equal angles in a quadrilateral
def has_four_equal_angles (q : Quadrilateral) : Prop := sorry

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem equal_angles_implies_rectangle (q : Quadrilateral) :
  has_four_equal_angles q → is_rectangle q := by sorry

end NUMINAMATH_CALUDE_equal_angles_implies_rectangle_l1829_182900


namespace NUMINAMATH_CALUDE_square_sum_equals_150_l1829_182957

theorem square_sum_equals_150 (u v : ℝ) 
  (h1 : u * (u + v) = 50) 
  (h2 : v * (u + v) = 100) : 
  (u + v)^2 = 150 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_150_l1829_182957


namespace NUMINAMATH_CALUDE_corn_height_after_three_weeks_l1829_182958

/-- The height of corn plants after three weeks of growth -/
def corn_height (initial_height week1_growth : ℕ) : ℕ :=
  let week2_growth := 2 * week1_growth
  let week3_growth := 4 * week2_growth
  initial_height + week1_growth + week2_growth + week3_growth

/-- Theorem stating that the corn height after three weeks is 22 inches -/
theorem corn_height_after_three_weeks :
  corn_height 0 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_corn_height_after_three_weeks_l1829_182958


namespace NUMINAMATH_CALUDE_fraction_sum_l1829_182922

theorem fraction_sum (p q : ℚ) (h : p / q = 4 / 5) : 
  1 / 7 + (2 * q - p) / (2 * q + p) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1829_182922


namespace NUMINAMATH_CALUDE_binomial_19_13_l1829_182950

theorem binomial_19_13 : Nat.choose 19 13 = 27132 := by
  -- Given conditions
  have h1 : Nat.choose 20 13 = 77520 := by sorry
  have h2 : Nat.choose 20 14 = 38760 := by sorry
  have h3 : Nat.choose 18 12 = 18564 := by sorry
  
  -- Proof
  sorry

end NUMINAMATH_CALUDE_binomial_19_13_l1829_182950


namespace NUMINAMATH_CALUDE_canoe_weight_proof_l1829_182978

def canoe_capacity : ℕ := 6
def person_weight : ℕ := 140

def total_weight_with_dog : ℕ :=
  let people_with_dog := (2 * canoe_capacity) / 3
  let total_people_weight := people_with_dog * person_weight
  let dog_weight := person_weight / 4
  total_people_weight + dog_weight

theorem canoe_weight_proof :
  total_weight_with_dog = 595 := by
  sorry

end NUMINAMATH_CALUDE_canoe_weight_proof_l1829_182978


namespace NUMINAMATH_CALUDE_expression_evaluation_l1829_182942

theorem expression_evaluation : -20 + 7 * (8 - 2 / 2) = 29 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1829_182942


namespace NUMINAMATH_CALUDE_product_equals_zero_l1829_182943

theorem product_equals_zero (a : ℤ) (h : a = -1) : (a - 3) * (a - 2) * (a - 1) * a = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l1829_182943


namespace NUMINAMATH_CALUDE_partnership_profit_share_l1829_182923

/-- 
Given:
- A, B, and C are in a partnership
- A invests 3 times as much as B
- B invests two-thirds of what C invests
- The total profit is 4400

Prove that B's share of the profit is 800
-/
theorem partnership_profit_share (c : ℝ) (total_profit : ℝ) 
  (h1 : c > 0)
  (h2 : total_profit = 4400) :
  let b := (2/3) * c
  let a := 3 * b
  let total_investment := a + b + c
  b / total_investment * total_profit = 800 := by
sorry

end NUMINAMATH_CALUDE_partnership_profit_share_l1829_182923


namespace NUMINAMATH_CALUDE_third_islander_statement_l1829_182990

-- Define the types of islanders
inductive IslanderType
| Knight
| Liar

-- Define the islanders
def A : IslanderType := IslanderType.Liar
def B : IslanderType := IslanderType.Knight
def C : IslanderType := IslanderType.Knight

-- Define the statements made by the islanders
def statement_A : Prop := ∀ x, x ≠ A → IslanderType.Liar = x
def statement_B : Prop := ∃! x, x ≠ B ∧ IslanderType.Knight = x

-- Theorem to prove
theorem third_islander_statement :
  (A = IslanderType.Liar) →
  (B = IslanderType.Knight) →
  (C = IslanderType.Knight) →
  statement_A →
  statement_B →
  (∃! x, x ≠ C ∧ IslanderType.Knight = x) :=
by sorry

end NUMINAMATH_CALUDE_third_islander_statement_l1829_182990


namespace NUMINAMATH_CALUDE_betty_age_l1829_182935

/-- Given the ages of Albert, Mary, Betty, and Charlie, prove Betty's age --/
theorem betty_age (albert mary betty charlie : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 14)
  (h4 : charlie = 3 * betty)
  (h5 : charlie = mary + 10) :
  betty = 7 := by
sorry

end NUMINAMATH_CALUDE_betty_age_l1829_182935


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1829_182955

theorem polynomial_simplification (s : ℝ) : (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 4) = -4 * s + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1829_182955


namespace NUMINAMATH_CALUDE_parrot_response_characterization_l1829_182927

def parrot_calc (n : ℤ) : ℚ :=
  (5 * n + 14) / 6 - 1

theorem parrot_response_characterization :
  ∀ n : ℤ, (∃ k : ℤ, parrot_calc n = k) ↔ ∃ m : ℤ, n = 6 * m + 2 :=
sorry

end NUMINAMATH_CALUDE_parrot_response_characterization_l1829_182927


namespace NUMINAMATH_CALUDE_mikes_games_l1829_182972

/-- Given Mike's earnings, expenses, and game cost, prove the number of games he can buy -/
theorem mikes_games (earnings : ℕ) (blade_cost : ℕ) (game_cost : ℕ) 
  (h1 : earnings = 101)
  (h2 : blade_cost = 47)
  (h3 : game_cost = 6) :
  (earnings - blade_cost) / game_cost = 9 := by
  sorry

end NUMINAMATH_CALUDE_mikes_games_l1829_182972


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l1829_182979

/-- Converts a list of bits (0s and 1s) to a natural number -/
def binaryToNat (bits : List Nat) : Nat :=
  bits.foldl (fun acc bit => 2 * acc + bit) 0

/-- The theorem to be proved -/
theorem binary_arithmetic_equality : 
  let a := binaryToNat [1, 1, 0, 1, 1]
  let b := binaryToNat [1, 0, 1, 0]
  let c := binaryToNat [1, 0, 0, 0, 1]
  let d := binaryToNat [1, 0, 1, 1]
  let e := binaryToNat [1, 1, 1, 0]
  let result := binaryToNat [0, 0, 1, 0, 0, 1]
  a + b - c + d - e = result := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l1829_182979


namespace NUMINAMATH_CALUDE_yu_chan_walking_distance_l1829_182937

def step_length : ℝ := 0.75
def walking_time : ℝ := 13
def steps_per_minute : ℝ := 70

theorem yu_chan_walking_distance : 
  step_length * walking_time * steps_per_minute = 682.5 := by
  sorry

end NUMINAMATH_CALUDE_yu_chan_walking_distance_l1829_182937


namespace NUMINAMATH_CALUDE_line_through_circle_center_l1829_182970

theorem line_through_circle_center (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y = 0 ∧ 
                 3*x + y + a = 0 ∧ 
                 x = -1 ∧ y = 2) → 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l1829_182970


namespace NUMINAMATH_CALUDE_sqrt_product_equals_thirty_l1829_182953

theorem sqrt_product_equals_thirty (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (5 * x) * Real.sqrt (30 * x) = 30) : 
  x = 1 / Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_thirty_l1829_182953


namespace NUMINAMATH_CALUDE_inequality_proof_l1829_182992

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 8 / (x * y) + y^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1829_182992


namespace NUMINAMATH_CALUDE_gcd_of_B_l1829_182948

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, n = 6 * x + 6}

theorem gcd_of_B : ∃ d : ℕ, d > 0 ∧ ∀ n ∈ B, d ∣ n ∧ ∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d :=
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_l1829_182948


namespace NUMINAMATH_CALUDE_sqrt_of_square_neg_l1829_182956

theorem sqrt_of_square_neg (a : ℝ) (h : a < 0) : Real.sqrt (a^2) = -a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_square_neg_l1829_182956


namespace NUMINAMATH_CALUDE_eight_elevenths_rounded_l1829_182986

-- Define a function to round a rational number to n decimal places
def round_to_decimal_places (q : ℚ) (n : ℕ) : ℚ :=
  (↑(⌊q * 10^n + 1/2⌋)) / 10^n

-- State the theorem
theorem eight_elevenths_rounded : round_to_decimal_places (8/11) 2 = 73/100 := by
  sorry

end NUMINAMATH_CALUDE_eight_elevenths_rounded_l1829_182986


namespace NUMINAMATH_CALUDE_jessica_expense_increase_l1829_182936

/-- Calculates the increase in Jessica's yearly expenses --/
def yearly_expense_increase (
  last_year_rent : ℕ)
  (last_year_food : ℕ)
  (last_year_insurance : ℕ)
  (rent_increase_percent : ℕ)
  (food_increase_percent : ℕ)
  (insurance_multiplier : ℕ) : ℕ :=
  let new_rent := last_year_rent + last_year_rent * rent_increase_percent / 100
  let new_food := last_year_food + last_year_food * food_increase_percent / 100
  let new_insurance := last_year_insurance * insurance_multiplier
  let last_year_total := last_year_rent + last_year_food + last_year_insurance
  let this_year_total := new_rent + new_food + new_insurance
  (this_year_total - last_year_total) * 12

theorem jessica_expense_increase :
  yearly_expense_increase 1000 200 100 30 50 3 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_jessica_expense_increase_l1829_182936


namespace NUMINAMATH_CALUDE_integer_and_mod_three_remainder_l1829_182932

theorem integer_and_mod_three_remainder (n : ℕ+) :
  ∃ k : ℤ, (n.val : ℝ)^3 + (3/2) * (n.val : ℝ)^2 + (1/2) * (n.val : ℝ) - 1 = (k : ℝ) ∧ k ≡ 2 [ZMOD 3] :=
sorry

end NUMINAMATH_CALUDE_integer_and_mod_three_remainder_l1829_182932


namespace NUMINAMATH_CALUDE_parabola_vertex_l1829_182981

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 5 * (x - 2)^2 + 6

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 6)

/-- Theorem: The vertex of the parabola y = 5(x-2)^2 + 6 is at the point (2, 6) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola vertex.1) ∧ 
  parabola vertex.1 = vertex.2 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1829_182981


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l1829_182993

def team_size : ℕ := 15
def starting_lineup_size : ℕ := 5
def preselected_players : ℕ := 3

theorem starting_lineup_combinations :
  Nat.choose (team_size - preselected_players) (starting_lineup_size - preselected_players) = 66 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l1829_182993


namespace NUMINAMATH_CALUDE_max_age_on_aubrey_eighth_birthday_l1829_182910

/-- Proves that Max's age on Aubrey's 8th birthday is 6 years -/
theorem max_age_on_aubrey_eighth_birthday 
  (max_birth : ℕ) -- Max's birth year
  (luka_birth : ℕ) -- Luka's birth year
  (aubrey_birth : ℕ) -- Aubrey's birth year
  (h1 : max_birth = luka_birth + 4) -- Max born when Luka turned 4
  (h2 : luka_birth = aubrey_birth - 2) -- Luka is 2 years older than Aubrey
  (h3 : aubrey_birth + 8 = max_birth + 6) -- Aubrey's 8th birthday is when Max is 6
  : (aubrey_birth + 8) - max_birth = 6 := by
sorry

end NUMINAMATH_CALUDE_max_age_on_aubrey_eighth_birthday_l1829_182910


namespace NUMINAMATH_CALUDE_max_switches_student_circle_l1829_182944

/-- 
Given n students with distinct heights arranged in a circle, 
where switches are allowed between a student and the one directly 
in front if the height difference is at least 2, the maximum number 
of possible switches before reaching a stable arrangement is ⁿC₃.
-/
theorem max_switches_student_circle (n : ℕ) : 
  ∃ (heights : Fin n → ℕ) (is_switch : Fin n → Fin n → Bool),
  (∀ i j, i ≠ j → heights i ≠ heights j) →
  (∀ i j, is_switch i j = true ↔ heights i > heights j + 1) →
  (∃ (switches : List (Fin n × Fin n)),
    (∀ (s : Fin n × Fin n), s ∈ switches → is_switch s.1 s.2 = true) ∧
    (∀ i j, is_switch i j = false) ∧
    switches.length = Nat.choose n 3) :=
by sorry

end NUMINAMATH_CALUDE_max_switches_student_circle_l1829_182944


namespace NUMINAMATH_CALUDE_fraction_comparison_l1829_182925

theorem fraction_comparison : 
  (14 / 10 : ℚ) = 7 / 5 ∧ 
  (1 + 2 / 5 : ℚ) = 7 / 5 ∧ 
  (1 + 4 / 20 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 3 / 15 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 2 / 6 : ℚ) ≠ 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1829_182925


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l1829_182901

/-- Two parabolas with mutually perpendicular axes -/
structure PerpendicularParabolas where
  -- First parabola: x = ay² + b
  a : ℝ
  b : ℝ
  -- Second parabola: y = cx² + d
  c : ℝ
  d : ℝ
  a_pos : 0 < a
  c_pos : 0 < c

/-- The four intersection points of two perpendicular parabolas -/
def intersectionPoints (p : PerpendicularParabolas) : Set (ℝ × ℝ) :=
  {point | point.1 = p.a * point.2^2 + p.b ∧ point.2 = p.c * point.1^2 + p.d}

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The theorem stating that the intersection points lie on a circle -/
theorem intersection_points_on_circle (p : PerpendicularParabolas) :
  ∃ (circle : Circle), ∀ point ∈ intersectionPoints p,
    (point.1 - circle.center.1)^2 + (point.2 - circle.center.2)^2 = circle.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l1829_182901


namespace NUMINAMATH_CALUDE_average_score_calculation_l1829_182904

theorem average_score_calculation (total_students : ℝ) (male_ratio : ℝ) 
  (male_avg_score : ℝ) (female_avg_score : ℝ) 
  (h1 : male_ratio = 0.4)
  (h2 : male_avg_score = 75)
  (h3 : female_avg_score = 80) :
  (male_ratio * male_avg_score + (1 - male_ratio) * female_avg_score) = 78 := by
  sorry

#check average_score_calculation

end NUMINAMATH_CALUDE_average_score_calculation_l1829_182904


namespace NUMINAMATH_CALUDE_therapy_charges_relation_l1829_182914

/-- A psychologist's charging scheme for therapy sessions. -/
structure TherapyCharges where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  first_hour_premium : firstHourCharge = additionalHourCharge + 30

/-- Calculate the total charge for a given number of therapy hours. -/
def totalCharge (charges : TherapyCharges) (hours : ℕ) : ℕ :=
  charges.firstHourCharge + (hours - 1) * charges.additionalHourCharge

/-- Theorem stating the relationship between charges for 5 hours and 3 hours of therapy. -/
theorem therapy_charges_relation (charges : TherapyCharges) :
  totalCharge charges 5 = 400 → totalCharge charges 3 = 252 := by
  sorry

#check therapy_charges_relation

end NUMINAMATH_CALUDE_therapy_charges_relation_l1829_182914


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1829_182968

/-- Given a line y = mx + b, if the reflection of point (2,2) across this line is (10,6), then m + b = 14 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The point (x, y) is on the line y = mx + b
    y = m * x + b ∧ 
    -- The point (x, y) is equidistant from (2,2) and (10,6)
    (x - 2)^2 + (y - 2)^2 = (x - 10)^2 + (y - 6)^2 ∧
    -- The line connecting (2,2) and (10,6) is perpendicular to y = mx + b
    (6 - 2) = -1 / m * (10 - 2)) →
  m + b = 14 := by
sorry


end NUMINAMATH_CALUDE_reflection_line_sum_l1829_182968


namespace NUMINAMATH_CALUDE_sacred_words_count_l1829_182988

-- Define the number of letters in the alien script
variable (n : ℕ)

-- Define the length of sacred words
variable (k : ℕ)

-- Condition that k is less than half of n
variable (h : k < n / 2)

-- Define a function to calculate the number of sacred k-words
def num_sacred_words (n k : ℕ) : ℕ :=
  n * Nat.choose (n - k - 1) (k - 1) * Nat.factorial k / k

-- Theorem statement
theorem sacred_words_count (n k : ℕ) (h : k < n / 2) :
  num_sacred_words n k = n * Nat.choose (n - k - 1) (k - 1) * Nat.factorial k / k :=
by sorry

-- Example for n = 10 and k = 4
example : num_sacred_words 10 4 = 600 :=
by sorry

end NUMINAMATH_CALUDE_sacred_words_count_l1829_182988


namespace NUMINAMATH_CALUDE_quadratic_equation_positive_roots_l1829_182999

theorem quadratic_equation_positive_roots (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m+2)*x + m + 5 = 0 → x > 0) ↔ -5 < m ∧ m ≤ -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_positive_roots_l1829_182999


namespace NUMINAMATH_CALUDE_average_home_runs_l1829_182952

theorem average_home_runs (players_5 players_7 players_9 players_11 players_13 : ℕ) 
  (h1 : players_5 = 3)
  (h2 : players_7 = 2)
  (h3 : players_9 = 1)
  (h4 : players_11 = 2)
  (h5 : players_13 = 1) :
  (5 * players_5 + 7 * players_7 + 9 * players_9 + 11 * players_11 + 13 * players_13) / 
  (players_5 + players_7 + players_9 + players_11 + players_13) = 73 / 9 :=
by sorry

end NUMINAMATH_CALUDE_average_home_runs_l1829_182952


namespace NUMINAMATH_CALUDE_car_speed_calculation_l1829_182960

/-- Proves that a car's speed is 52 miles per hour given specific conditions -/
theorem car_speed_calculation (fuel_efficiency : ℝ) (fuel_consumed : ℝ) (time : ℝ)
  (gallon_to_liter : ℝ) (km_to_mile : ℝ) :
  fuel_efficiency = 32 →
  fuel_consumed = 3.9 →
  time = 5.7 →
  gallon_to_liter = 3.8 →
  km_to_mile = 1.6 →
  (fuel_consumed * gallon_to_liter * fuel_efficiency) / (time * km_to_mile) = 52 := by
sorry

end NUMINAMATH_CALUDE_car_speed_calculation_l1829_182960


namespace NUMINAMATH_CALUDE_fraction_power_product_l1829_182949

theorem fraction_power_product : (1/3)^100 * 3^101 = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_power_product_l1829_182949


namespace NUMINAMATH_CALUDE_jump_rope_problem_l1829_182908

theorem jump_rope_problem (a : ℕ) : 
  let counts : List ℕ := [180, 182, 173, 175, a, 178, 176]
  (counts.sum / counts.length : ℚ) = 178 →
  a = 182 := by
sorry

end NUMINAMATH_CALUDE_jump_rope_problem_l1829_182908


namespace NUMINAMATH_CALUDE_function_value_theorem_l1829_182921

theorem function_value_theorem (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f (x + 1) = x) 
  (h2 : f a = 8) : 
  a = 9 := by
  sorry

end NUMINAMATH_CALUDE_function_value_theorem_l1829_182921


namespace NUMINAMATH_CALUDE_rental_cost_equation_l1829_182991

/-- The monthly cost of renting a car. -/
def R : ℝ := sorry

/-- The monthly cost of the new car. -/
def new_car_cost : ℝ := 30

/-- The number of months in a year. -/
def months_in_year : ℕ := 12

/-- The difference in total cost over a year. -/
def cost_difference : ℝ := 120

/-- Theorem stating the relationship between rental cost and new car cost. -/
theorem rental_cost_equation : 
  months_in_year * R - months_in_year * new_car_cost = cost_difference := by
  sorry

end NUMINAMATH_CALUDE_rental_cost_equation_l1829_182991


namespace NUMINAMATH_CALUDE_smallest_x_value_l1829_182980

theorem smallest_x_value (x : ℝ) : 
  ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20 → x ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1829_182980


namespace NUMINAMATH_CALUDE_bald_eagle_dive_time_l1829_182918

/-- The time it takes for the bald eagle to dive to the ground given the specified conditions -/
theorem bald_eagle_dive_time : 
  ∀ (v_eagle : ℝ) (v_falcon : ℝ) (t_falcon : ℝ) (distance : ℝ),
  v_eagle > 0 →
  v_falcon = 2 * v_eagle →
  t_falcon = 15 →
  distance > 0 →
  distance = v_eagle * (2 * t_falcon) →
  distance = v_falcon * t_falcon →
  2 * t_falcon = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_bald_eagle_dive_time_l1829_182918


namespace NUMINAMATH_CALUDE_house_amenities_l1829_182995

theorem house_amenities (total : ℕ) (garage : ℕ) (pool : ℕ) (neither : ℕ) :
  total = 65 → garage = 50 → pool = 40 → neither = 10 →
  ∃ both : ℕ, both = 35 ∧ garage + pool - both = total - neither :=
by sorry

end NUMINAMATH_CALUDE_house_amenities_l1829_182995


namespace NUMINAMATH_CALUDE_symmetric_roots_iff_b_eq_two_or_four_l1829_182996

/-- The polynomial in question -/
def P (b : ℝ) (z : ℂ) : ℂ :=
  z^5 - 8*z^4 + 12*b*z^3 - 4*(3*b^2 + 4*b - 4)*z^2 + 2*z + 2

/-- The roots of the polynomial form a symmetric pattern around the origin -/
def symmetric_roots (b : ℝ) : Prop :=
  ∃ (r : Finset ℂ), Finset.card r = 5 ∧ 
    (∀ z ∈ r, P b z = 0) ∧
    (∀ z ∈ r, -z ∈ r)

/-- The main theorem stating the condition for symmetric roots -/
theorem symmetric_roots_iff_b_eq_two_or_four :
  ∀ b : ℝ, symmetric_roots b ↔ b = 2 ∨ b = 4 := by sorry

end NUMINAMATH_CALUDE_symmetric_roots_iff_b_eq_two_or_four_l1829_182996


namespace NUMINAMATH_CALUDE_initially_tagged_fish_l1829_182919

-- Define the total number of fish in the pond
def total_fish : ℕ := 750

-- Define the number of fish in the second catch
def second_catch : ℕ := 50

-- Define the number of tagged fish in the second catch
def tagged_in_second_catch : ℕ := 2

-- Define the ratio of tagged fish in the second catch
def tagged_ratio : ℚ := tagged_in_second_catch / second_catch

-- Theorem: The number of fish initially caught and tagged is 30
theorem initially_tagged_fish : 
  ∃ (T : ℕ), T = 30 ∧ (T : ℚ) / total_fish = tagged_ratio :=
sorry

end NUMINAMATH_CALUDE_initially_tagged_fish_l1829_182919


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1829_182907

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  (a = b + 2) →  -- a is two years older than b
  (a + b + c = 27) →  -- The total of the ages of a, b, and c is 27
  (b = 10) →  -- b is 10 years old
  (b : ℚ) / c = 2 / 1 :=  -- The ratio of b's age to c's age is 2:1
by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1829_182907


namespace NUMINAMATH_CALUDE_kid_tickets_sold_l1829_182962

/-- Prove that the number of kid tickets sold is 75 -/
theorem kid_tickets_sold (total_tickets : ℕ) (total_profit : ℕ) 
  (adult_price kid_price : ℕ) (h1 : total_tickets = 175) 
  (h2 : total_profit = 750) (h3 : adult_price = 6) (h4 : kid_price = 2) : 
  ∃ (adult_tickets kid_tickets : ℕ), 
    adult_tickets + kid_tickets = total_tickets ∧ 
    adult_price * adult_tickets + kid_price * kid_tickets = total_profit ∧
    kid_tickets = 75 :=
sorry

end NUMINAMATH_CALUDE_kid_tickets_sold_l1829_182962


namespace NUMINAMATH_CALUDE_garden_area_l1829_182994

/-- A rectangular garden with width one-third of its length and perimeter 72 meters has an area of 243 square meters. -/
theorem garden_area (width length : ℝ) : 
  width > 0 ∧ 
  length > 0 ∧ 
  width = length / 3 ∧ 
  2 * (width + length) = 72 →
  width * length = 243 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l1829_182994


namespace NUMINAMATH_CALUDE_three_aligned_probability_l1829_182938

-- Define the grid
def Grid := Fin 3 × Fin 3

-- Define the number of markers
def num_markers : ℕ := 4

-- Define the total number of cells in the grid
def total_cells : ℕ := 9

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the total number of ways to place markers
def total_arrangements : ℕ := combination total_cells num_markers

-- Define the number of ways to align 3 markers in a row, column, or diagonal
def aligned_arrangements : ℕ := 48

-- The main theorem
theorem three_aligned_probability :
  (aligned_arrangements : ℚ) / total_arrangements = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_three_aligned_probability_l1829_182938


namespace NUMINAMATH_CALUDE_unique_root_condition_l1829_182916

/-- The equation ln(x+a) - 4(x+a)^2 + a = 0 has a unique root at x = 3 if and only if a = (3 ln 2 + 1) / 2 -/
theorem unique_root_condition (a : ℝ) : 
  (∃! x : ℝ, Real.log (x + a) - 4 * (x + a)^2 + a = 0 ∧ x = 3) ↔ 
  a = (3 * Real.log 2 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_unique_root_condition_l1829_182916


namespace NUMINAMATH_CALUDE_sphere_area_ratio_l1829_182984

theorem sphere_area_ratio (r₁ r₂ A₁ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : A₁ > 0) :
  let A₂ := A₁ * (r₂ / r₁)^2
  r₁ = 4 ∧ r₂ = 6 ∧ A₁ = 37 → A₂ = 83.25 := by
  sorry

end NUMINAMATH_CALUDE_sphere_area_ratio_l1829_182984


namespace NUMINAMATH_CALUDE_distinct_positive_numbers_properties_l1829_182951

theorem distinct_positive_numbers_properties (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  ((a - b)^2 + (b - c)^2 + (c - a)^2 > 0) ∧ 
  (a > b ∨ a < b ∨ a = b) ∧
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_distinct_positive_numbers_properties_l1829_182951


namespace NUMINAMATH_CALUDE_sales_solution_l1829_182947

def sales_problem (sale1 sale2 sale3 sale5 sale6 average : ℕ) : Prop :=
  let total_sales := 6 * average
  let known_sales := sale1 + sale2 + sale3 + sale5 + sale6
  total_sales - known_sales = 5730

theorem sales_solution :
  sales_problem 4000 6524 5689 6000 12557 7000 := by
  sorry

end NUMINAMATH_CALUDE_sales_solution_l1829_182947


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1829_182954

/-- Represents a systematic sampling of examination rooms -/
structure SystematicSampling where
  totalRooms : Nat
  sampleSize : Nat
  firstRoom : Nat
  interval : Nat

/-- Checks if a room number is part of the systematic sample -/
def isSelected (s : SystematicSampling) (room : Nat) : Prop :=
  ∃ k : Nat, room = s.firstRoom + k * s.interval ∧ room ≤ s.totalRooms

/-- The set of selected room numbers in a systematic sampling -/
def selectedRooms (s : SystematicSampling) : Set Nat :=
  {room | isSelected s room}

theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.totalRooms = 64)
  (h2 : s.sampleSize = 8)
  (h3 : s.firstRoom = 5)
  (h4 : isSelected s 21)
  (h5 : s.interval = s.totalRooms / s.sampleSize) :
  selectedRooms s = {5, 13, 21, 29, 37, 45, 53, 61} := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1829_182954


namespace NUMINAMATH_CALUDE_garage_roof_leak_l1829_182997

/-- The amount of water leaked from three holes in a garage roof over a 2-hour period -/
def water_leaked (largest_hole_rate : ℚ) (time_hours : ℚ) : ℚ :=
  let medium_hole_rate := largest_hole_rate / 2
  let smallest_hole_rate := medium_hole_rate / 3
  let time_minutes := time_hours * 60
  (largest_hole_rate + medium_hole_rate + smallest_hole_rate) * time_minutes

/-- Theorem stating the total amount of water leaked from three holes in a garage roof over a 2-hour period -/
theorem garage_roof_leak : water_leaked 3 2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_garage_roof_leak_l1829_182997


namespace NUMINAMATH_CALUDE_prob_at_least_three_matching_l1829_182940

/-- The number of sides on each die -/
def numSides : ℕ := 10

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of getting at least three matching dice out of five fair ten-sided dice -/
def probAtLeastThreeMatching : ℚ := 173 / 20000

/-- Theorem stating that the probability of at least three out of five fair ten-sided dice 
    showing the same value is 173/20000 -/
theorem prob_at_least_three_matching : 
  probAtLeastThreeMatching = 173 / 20000 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_matching_l1829_182940


namespace NUMINAMATH_CALUDE_decimal_multiplication_addition_l1829_182934

theorem decimal_multiplication_addition : (0.3 * 0.7) + (0.5 * 0.4) = 0.41 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_addition_l1829_182934


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1829_182998

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 2 = 2 →
  a 3 + a 4 = 10 →
  a 5 + a 6 = 18 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1829_182998


namespace NUMINAMATH_CALUDE_june_video_hours_l1829_182985

/-- Calculates the total video hours uploaded in a month with varying upload rates -/
def total_video_hours (days : ℕ) (initial_rate : ℕ) (doubled_rate : ℕ) : ℕ :=
  let half_days := days / 2
  (half_days * initial_rate) + (half_days * doubled_rate)

/-- Proves that the total video hours uploaded in June is 450 -/
theorem june_video_hours :
  total_video_hours 30 10 20 = 450 := by
  sorry

end NUMINAMATH_CALUDE_june_video_hours_l1829_182985


namespace NUMINAMATH_CALUDE_decreasing_cubic_condition_l1829_182975

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x

-- State the theorem
theorem decreasing_cubic_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_cubic_condition_l1829_182975


namespace NUMINAMATH_CALUDE_repeating_decimal_36_equals_4_11_l1829_182982

/-- Represents a repeating decimal with a repeating part of two digits -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (a * 10 + b) / 99

/-- The theorem states that 0.¯36 is equal to 4/11 -/
theorem repeating_decimal_36_equals_4_11 :
  RepeatingDecimal 3 6 = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_36_equals_4_11_l1829_182982


namespace NUMINAMATH_CALUDE_unique_multiplication_707_l1829_182917

theorem unique_multiplication_707 : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    ∃ (a b : ℕ), n = 100 * a + 70 + b ∧ 
    707 * n = 124432 := by
  sorry

end NUMINAMATH_CALUDE_unique_multiplication_707_l1829_182917


namespace NUMINAMATH_CALUDE_intersection_length_l1829_182967

/-- The length of the line segment formed by the intersection of y = x + 1 and x²/4 + y²/3 = 1 is 24/7 -/
theorem intersection_length :
  let l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2/4 + p.2^2/3 = 1}
  let A : Set (ℝ × ℝ) := l ∩ C
  ∃ p q : ℝ × ℝ, p ∈ A ∧ q ∈ A ∧ p ≠ q ∧ ‖p - q‖ = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_length_l1829_182967


namespace NUMINAMATH_CALUDE_marathon_distance_l1829_182945

theorem marathon_distance (marathon_miles : ℕ) (marathon_yards : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) :
  marathon_miles = 26 →
  marathon_yards = 395 →
  yards_per_mile = 1760 →
  num_marathons = 15 →
  (num_marathons * marathon_yards) % yards_per_mile = 645 := by
  sorry

#check marathon_distance

end NUMINAMATH_CALUDE_marathon_distance_l1829_182945


namespace NUMINAMATH_CALUDE_largest_factorial_as_consecutive_product_l1829_182928

theorem largest_factorial_as_consecutive_product : 
  ∀ n : ℕ, n > 0 → 
  (∃ k : ℕ, n! = (k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5)) → 
  n = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_factorial_as_consecutive_product_l1829_182928


namespace NUMINAMATH_CALUDE_cube_root_plus_abs_plus_power_equals_six_linear_function_through_two_points_l1829_182931

-- Problem 1
theorem cube_root_plus_abs_plus_power_equals_six :
  (8 : ℝ) ^ (1/3) + |(-5)| + (-1)^2023 = 6 := by sorry

-- Problem 2
theorem linear_function_through_two_points :
  ∀ (k b : ℝ), (∀ x y : ℝ, y = k * x + b) →
  (1 = k * 0 + b) →
  (5 = k * 2 + b) →
  (∀ x : ℝ, k * x + b = 2 * x + 1) := by sorry

end NUMINAMATH_CALUDE_cube_root_plus_abs_plus_power_equals_six_linear_function_through_two_points_l1829_182931


namespace NUMINAMATH_CALUDE_intersection_line_proof_l1829_182987

/-- Given two lines in the plane and a slope, prove that a certain line passes through their intersection point with the given slope. -/
theorem intersection_line_proof (x y : ℝ) : 
  (3 * x + 4 * y = 5) →  -- First line equation
  (3 * x - 4 * y = 13) →  -- Second line equation
  (∃ (x₀ y₀ : ℝ), (3 * x₀ + 4 * y₀ = 5) ∧ (3 * x₀ - 4 * y₀ = 13) ∧ (2 * x₀ - y₀ = 7)) ∧  -- Intersection point exists and satisfies all equations
  (∀ (x₁ y₁ : ℝ), (2 * x₁ - y₁ = 7) → (y₁ - y) / (x₁ - x) = 2 ∨ x₁ = x)  -- Slope of the line 2x - y - 7 = 0 is 2
  := by sorry

end NUMINAMATH_CALUDE_intersection_line_proof_l1829_182987


namespace NUMINAMATH_CALUDE_savings_amount_l1829_182903

/-- Represents the price of a single book -/
def book_price : ℝ := 45

/-- Represents the discount percentage for Promotion A -/
def promotion_a_discount : ℝ := 0.4

/-- Represents the fixed discount amount for Promotion B -/
def promotion_b_discount : ℝ := 15

/-- Represents the local tax rate -/
def tax_rate : ℝ := 0.08

/-- Calculates the total cost for Promotion A including tax -/
def total_cost_a : ℝ :=
  (book_price + book_price * (1 - promotion_a_discount)) * (1 + tax_rate)

/-- Calculates the total cost for Promotion B including tax -/
def total_cost_b : ℝ :=
  (book_price + (book_price - promotion_b_discount)) * (1 + tax_rate)

/-- Theorem stating the savings amount by choosing Promotion A over Promotion B -/
theorem savings_amount : 
  total_cost_b - total_cost_a = 3.24 := by sorry

end NUMINAMATH_CALUDE_savings_amount_l1829_182903


namespace NUMINAMATH_CALUDE_prob_at_least_one_unqualified_is_correct_l1829_182965

/-- The total number of products -/
def total_products : ℕ := 6

/-- The number of qualified products -/
def qualified_products : ℕ := 4

/-- The number of unqualified products -/
def unqualified_products : ℕ := 2

/-- The number of products randomly selected -/
def selected_products : ℕ := 2

/-- The probability of selecting at least one unqualified product -/
def prob_at_least_one_unqualified : ℚ := 3/5

theorem prob_at_least_one_unqualified_is_correct :
  (1 : ℚ) - (Nat.choose qualified_products selected_products : ℚ) / (Nat.choose total_products selected_products : ℚ) = prob_at_least_one_unqualified :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_unqualified_is_correct_l1829_182965


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l1829_182977

theorem retailer_profit_percentage 
  (cost : ℝ) 
  (discounted_price : ℝ) 
  (discount_rate : ℝ) : 
  cost = 80 → 
  discounted_price = 130 → 
  discount_rate = 0.2 → 
  ((discounted_price / (1 - discount_rate) - cost) / cost) * 100 = 103.125 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l1829_182977


namespace NUMINAMATH_CALUDE_unique_solution_l1829_182971

/-- The set of digits used in the equation -/
def Digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The sum of all digits from 0 to 9 -/
def DigitsSum : Nat := Finset.sum Digits id

/-- The set of digits used on the left side of the equation -/
def LeftDigits : Finset Nat := {0, 1, 2, 4, 5, 7, 8, 9}

/-- The sum of digits on the left side of the equation -/
def LeftSum : Nat := Finset.sum LeftDigits id

/-- The two-digit number on the right side of the equation -/
def RightNumber : Nat := 36

/-- The statement that the equation is a valid solution -/
theorem unique_solution :
  (LeftSum = RightNumber) ∧
  (Digits \ LeftDigits).card = 2 ∧
  (∀ (s : Finset Nat), s ⊂ Digits → s.card = 8 → Finset.sum s id ≠ RightNumber) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l1829_182971


namespace NUMINAMATH_CALUDE_remainder_17_power_77_mod_7_l1829_182926

theorem remainder_17_power_77_mod_7 : 17^77 % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_power_77_mod_7_l1829_182926


namespace NUMINAMATH_CALUDE_equation_solutions_l1829_182929

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 2*x - 15 = 0 ↔ x = 5 ∨ x = -3) ∧
  (∀ x : ℝ, (x - 1)^2 = 2*(x - 1) ↔ x = 1 ∨ x = 3) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1829_182929


namespace NUMINAMATH_CALUDE_pictures_deleted_l1829_182920

theorem pictures_deleted (zoo_pics : ℕ) (museum_pics : ℕ) (remaining_pics : ℕ) : 
  zoo_pics = 49 → museum_pics = 8 → remaining_pics = 19 →
  zoo_pics + museum_pics - remaining_pics = 38 := by
  sorry

end NUMINAMATH_CALUDE_pictures_deleted_l1829_182920
