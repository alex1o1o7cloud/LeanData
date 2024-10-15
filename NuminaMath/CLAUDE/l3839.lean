import Mathlib

namespace NUMINAMATH_CALUDE_handshake_problem_l3839_383959

theorem handshake_problem (a b : ℕ) : 
  a + b = 20 →
  (a.choose 2) + (b.choose 2) = 106 →
  a * b = 84 :=
by sorry

end NUMINAMATH_CALUDE_handshake_problem_l3839_383959


namespace NUMINAMATH_CALUDE_andy_problem_count_l3839_383996

/-- The number of problems Andy solves when he completes problems from 80 to 125 inclusive -/
def problems_solved : ℕ := 125 - 80 + 1

theorem andy_problem_count : problems_solved = 46 := by
  sorry

end NUMINAMATH_CALUDE_andy_problem_count_l3839_383996


namespace NUMINAMATH_CALUDE_solve_pencil_problem_l3839_383929

def pencil_problem (total_pencils : ℕ) (buy_price sell_price : ℚ) (desired_profit : ℚ) (pencils_to_sell : ℕ) : Prop :=
  let total_cost : ℚ := total_pencils * buy_price
  let revenue : ℚ := pencils_to_sell * sell_price
  let actual_profit : ℚ := revenue - total_cost
  actual_profit = desired_profit

theorem solve_pencil_problem :
  pencil_problem 2000 (15/100) (30/100) 180 1600 := by
  sorry

end NUMINAMATH_CALUDE_solve_pencil_problem_l3839_383929


namespace NUMINAMATH_CALUDE_pages_revised_twice_l3839_383953

/-- Represents the manuscript typing scenario -/
structure ManuscriptTyping where
  totalPages : Nat
  revisedOnce : Nat
  revisedTwice : Nat
  firstTypingCost : Nat
  revisionCost : Nat
  totalCost : Nat

/-- Calculates the total cost of typing and revising a manuscript -/
def calculateTotalCost (m : ManuscriptTyping) : Nat :=
  m.firstTypingCost * m.totalPages + 
  m.revisionCost * m.revisedOnce + 
  2 * m.revisionCost * m.revisedTwice

/-- Theorem stating that given the specified conditions, 30 pages were revised twice -/
theorem pages_revised_twice (m : ManuscriptTyping) 
  (h1 : m.totalPages = 100)
  (h2 : m.revisedOnce = 20)
  (h3 : m.firstTypingCost = 10)
  (h4 : m.revisionCost = 5)
  (h5 : m.totalCost = 1400)
  (h6 : calculateTotalCost m = m.totalCost) :
  m.revisedTwice = 30 := by
  sorry

end NUMINAMATH_CALUDE_pages_revised_twice_l3839_383953


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3839_383923

/-- Given a geometric sequence with common ratio 2 and sum of first 4 terms equal to 1,
    the sum of the first 8 terms is 17 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = 2 * a n) 
    (h2 : a 1 + a 2 + a 3 + a 4 = 1) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 17 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3839_383923


namespace NUMINAMATH_CALUDE_original_class_size_l3839_383937

theorem original_class_size (x : ℕ) : 
  (x * 40 + 12 * 32) / (x + 12) = 36 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_original_class_size_l3839_383937


namespace NUMINAMATH_CALUDE_fraction_inequality_l3839_383985

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3839_383985


namespace NUMINAMATH_CALUDE_proposition_truth_values_l3839_383988

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def is_solution (x : ℤ) : Prop := x^2 + x - 2 = 0

theorem proposition_truth_values :
  ((is_prime 3 ∨ is_even 3) = true) ∧
  ((is_prime 3 ∧ is_even 3) = false) ∧
  ((¬is_prime 3) = false) ∧
  ((is_solution (-2) ∨ is_solution 1) = true) ∧
  ((is_solution (-2) ∧ is_solution 1) = true) ∧
  ((¬is_solution (-2)) = false) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l3839_383988


namespace NUMINAMATH_CALUDE_dot_product_theorem_l3839_383992

def vector_dot_product (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2)

def vector_perpendicular (v w : ℝ × ℝ) : Prop :=
  vector_dot_product v w = 0

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem dot_product_theorem (x y : ℝ) : 
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (1, y)
  let c : ℝ × ℝ := (3, -6)
  vector_perpendicular a c → vector_parallel b c →
  vector_dot_product (a.1 + b.1, a.2 + b.2) c = 15 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_theorem_l3839_383992


namespace NUMINAMATH_CALUDE_distinct_weights_theorem_l3839_383958

/-- The number of distinct weights that can be measured with four weights on a two-pan balance scale. -/
def distinct_weights : ℕ := 40

/-- The number of weights available. -/
def num_weights : ℕ := 4

/-- The number of possible placements for each weight (left pan, right pan, or not used). -/
def placement_options : ℕ := 3

/-- Represents the two-pan balance scale. -/
structure BalanceScale :=
  (left_pan : Finset ℕ)
  (right_pan : Finset ℕ)

/-- Calculates the total number of possible configurations. -/
def total_configurations : ℕ := placement_options ^ num_weights

/-- Theorem stating the number of distinct weights that can be measured. -/
theorem distinct_weights_theorem :
  distinct_weights = (total_configurations - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_weights_theorem_l3839_383958


namespace NUMINAMATH_CALUDE_line_formation_ways_l3839_383955

/-- The number of ways to form a line by selecting r people out of n -/
def permutations (n : ℕ) (r : ℕ) : ℕ := (n.factorial) / ((n - r).factorial)

/-- The total number of people -/
def total_people : ℕ := 7

/-- The number of people to select -/
def selected_people : ℕ := 5

theorem line_formation_ways :
  permutations total_people selected_people = 2520 := by
  sorry

end NUMINAMATH_CALUDE_line_formation_ways_l3839_383955


namespace NUMINAMATH_CALUDE_a_investment_l3839_383957

/-- Represents the investment scenario and proves A's investment amount -/
theorem a_investment (a_time b_time : ℕ) (b_investment total_profit a_share : ℚ) :
  a_time = 12 →
  b_time = 6 →
  b_investment = 200 →
  total_profit = 100 →
  a_share = 75 →
  ∃ (a_investment : ℚ),
    a_investment * a_time / (a_investment * a_time + b_investment * b_time) * total_profit = a_share ∧
    a_investment = 300 := by
  sorry


end NUMINAMATH_CALUDE_a_investment_l3839_383957


namespace NUMINAMATH_CALUDE_system_solution_l3839_383965

theorem system_solution (x y z : ℝ) : 
  (x + y + z = 6 ∧ x*y + y*z + z*x = 11 ∧ x*y*z = 6) ↔ 
  ((x = 1 ∧ y = 2 ∧ z = 3) ∨
   (x = 1 ∧ y = 3 ∧ z = 2) ∨
   (x = 2 ∧ y = 1 ∧ z = 3) ∨
   (x = 2 ∧ y = 3 ∧ z = 1) ∨
   (x = 3 ∧ y = 1 ∧ z = 2) ∨
   (x = 3 ∧ y = 2 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3839_383965


namespace NUMINAMATH_CALUDE_linear_pair_angle_ratio_l3839_383934

theorem linear_pair_angle_ratio (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Both angles are positive
  a + b = 180 ∧    -- Angles form a linear pair (sum to 180°)
  a = 5 * b →      -- Angles are in ratio 5:1
  b = 30 :=        -- The smaller angle is 30°
by sorry

end NUMINAMATH_CALUDE_linear_pair_angle_ratio_l3839_383934


namespace NUMINAMATH_CALUDE_cracker_cost_is_350_l3839_383914

/-- The cost of a box of crackers in dollars -/
def cracker_cost : ℝ := sorry

/-- The total cost before discount in dollars -/
def total_cost_before_discount : ℝ := 5 + 4 * 2 + 3.5 + cracker_cost

/-- The discount rate as a decimal -/
def discount_rate : ℝ := 0.1

/-- The total cost after discount in dollars -/
def total_cost_after_discount : ℝ := total_cost_before_discount * (1 - discount_rate)

theorem cracker_cost_is_350 :
  cracker_cost = 3.5 ∧ total_cost_after_discount = 18 := by sorry

end NUMINAMATH_CALUDE_cracker_cost_is_350_l3839_383914


namespace NUMINAMATH_CALUDE_cooking_gear_final_cost_l3839_383913

def cookingGearCost (mitts apron utensils recipients discount tax : ℝ) : ℝ :=
  let knife := 2 * utensils
  let setPrice := mitts + apron + utensils + knife
  let discountedPrice := setPrice * (1 - discount)
  let totalBeforeTax := discountedPrice * recipients
  totalBeforeTax * (1 + tax)

theorem cooking_gear_final_cost :
  cookingGearCost 14 16 10 8 0.25 0.08 = 388.80 := by
  sorry

end NUMINAMATH_CALUDE_cooking_gear_final_cost_l3839_383913


namespace NUMINAMATH_CALUDE_min_rectangles_cover_l3839_383977

/-- A point in the unit square -/
structure Point where
  x : Real
  y : Real
  x_in_unit : 0 < x ∧ x < 1
  y_in_unit : 0 < y ∧ y < 1

/-- A rectangle with sides parallel to the unit square -/
structure Rectangle where
  left : Real
  right : Real
  bottom : Real
  top : Real
  valid : 0 ≤ left ∧ left < right ∧ right ≤ 1 ∧
          0 ≤ bottom ∧ bottom < top ∧ top ≤ 1

/-- The theorem statement -/
theorem min_rectangles_cover (n : Nat) (S : Finset Point) :
  S.card = n →
  ∃ (k : Nat) (R : Finset Rectangle),
    R.card = k ∧
    (∀ p ∈ S, ∀ r ∈ R, ¬(r.left < p.x ∧ p.x < r.right ∧ r.bottom < p.y ∧ p.y < r.top)) ∧
    (∀ x y : Real, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 →
      (∀ p ∈ S, p.x ≠ x ∨ p.y ≠ y) →
      ∃ r ∈ R, r.left < x ∧ x < r.right ∧ r.bottom < y ∧ y < r.top) ∧
    k = 2 * n + 2 ∧
    (∀ m : Nat, m < k →
      ¬∃ (R' : Finset Rectangle),
        R'.card = m ∧
        (∀ p ∈ S, ∀ r ∈ R', ¬(r.left < p.x ∧ p.x < r.right ∧ r.bottom < p.y ∧ p.y < r.top)) ∧
        (∀ x y : Real, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 →
          (∀ p ∈ S, p.x ≠ x ∨ p.y ≠ y) →
          ∃ r ∈ R', r.left < x ∧ x < r.right ∧ r.bottom < y ∧ y < r.top)) :=
by sorry

end NUMINAMATH_CALUDE_min_rectangles_cover_l3839_383977


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_36_56_l3839_383932

theorem gcf_lcm_sum_36_56 : Nat.gcd 36 56 + Nat.lcm 36 56 = 508 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_36_56_l3839_383932


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3839_383915

/-- Given a stratified sample where the ratio of product A to total production
    is 1/5 and 18 products of type A are sampled, prove the total sample size is 90. -/
theorem stratified_sample_size (sample_A : ℕ) (ratio_A : ℚ) (total_sample : ℕ) :
  sample_A = 18 →
  ratio_A = 1 / 5 →
  (sample_A : ℚ) / total_sample = ratio_A →
  total_sample = 90 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l3839_383915


namespace NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l3839_383990

/-- Given a quadratic equation x^2 + mx + (m^2 - 3m + 3) = 0, 
    if its roots are reciprocals of each other, then m = 2 -/
theorem reciprocal_roots_quadratic (m : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 
   x^2 + m*x + (m^2 - 3*m + 3) = 0 ∧
   y^2 + m*y + (m^2 - 3*m + 3) = 0 ∧
   x*y = 1) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l3839_383990


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3839_383919

/-- Proves that a train of given length and speed takes 30 seconds to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : Real)
  (train_speed_kmh : Real)
  (bridge_length : Real)
  (h1 : train_length = 140)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 235) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry


end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3839_383919


namespace NUMINAMATH_CALUDE_stairs_problem_l3839_383922

/-- Calculates the number of steps climbed given the number of flights, height per flight, and step height. -/
def steps_climbed (flights : ℕ) (flight_height : ℕ) (step_height : ℕ) : ℕ :=
  (flights * flight_height * 12) / step_height

/-- Theorem: Given 9 flights of stairs, with each flight being 10 feet, and each step being 18 inches, 
    the total number of steps climbed is 60. -/
theorem stairs_problem : steps_climbed 9 10 18 = 60 := by
  sorry

end NUMINAMATH_CALUDE_stairs_problem_l3839_383922


namespace NUMINAMATH_CALUDE_expression_evaluation_l3839_383987

theorem expression_evaluation : 72 + (150 / 25) + (16 * 19) - 250 - (450 / 9) = 82 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3839_383987


namespace NUMINAMATH_CALUDE_quarters_sale_amount_l3839_383984

/-- The amount received for selling quarters at a percentage of their face value -/
def amount_received (num_quarters : ℕ) (face_value : ℚ) (percentage : ℕ) : ℚ :=
  (num_quarters : ℚ) * face_value * ((percentage : ℚ) / 100)

/-- Theorem stating that selling 8 quarters with face value $0.25 at 500% yields $10 -/
theorem quarters_sale_amount : 
  amount_received 8 (1/4) 500 = 10 := by sorry

end NUMINAMATH_CALUDE_quarters_sale_amount_l3839_383984


namespace NUMINAMATH_CALUDE_certain_number_problem_l3839_383918

theorem certain_number_problem (x : ℝ) : (((x + 10) * 7) / 5) - 5 = 88 / 2 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3839_383918


namespace NUMINAMATH_CALUDE_g_difference_zero_l3839_383911

def sum_of_divisors (n : ℕ+) : ℕ := sorry

def f (n : ℕ+) : ℚ := (sum_of_divisors n : ℚ) / n

def g (n : ℕ+) : ℚ := f n + 1 / n

theorem g_difference_zero : g 512 - g 256 = 0 := by sorry

end NUMINAMATH_CALUDE_g_difference_zero_l3839_383911


namespace NUMINAMATH_CALUDE_triangle_area_with_given_base_height_l3839_383916

/-- The area of a triangle with base 12 cm and height 15 cm is 90 cm². -/
theorem triangle_area_with_given_base_height :
  let base : ℝ := 12
  let height : ℝ := 15
  let area : ℝ := (1 / 2) * base * height
  area = 90 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_with_given_base_height_l3839_383916


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3839_383933

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I : ℂ) * Complex.I = -1 →
  (∃ b : ℝ, (a + Complex.I) / (1 - Complex.I) = b * Complex.I) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3839_383933


namespace NUMINAMATH_CALUDE_inequality_proofs_l3839_383999

theorem inequality_proofs (a b : ℝ) :
  (a ≥ b ∧ b > 0) →
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b ∧
  (a > 0 ∧ b > 0 ∧ a + b = 10) →
  Real.sqrt (1 + 3 * a) + Real.sqrt (1 + 3 * b) ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l3839_383999


namespace NUMINAMATH_CALUDE_trig_combination_l3839_383982

theorem trig_combination (x : ℝ) : 
  Real.cos (3 * x) + Real.cos (5 * x) + Real.tan (2 * x) = 
  2 * Real.cos (4 * x) * Real.cos x + Real.sin (2 * x) / Real.cos (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_trig_combination_l3839_383982


namespace NUMINAMATH_CALUDE_average_theorem_l3839_383910

theorem average_theorem (x : ℝ) : 
  (x + 0.005) / 2 = 0.2025 → x = 0.400 := by
sorry

end NUMINAMATH_CALUDE_average_theorem_l3839_383910


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3839_383927

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 4*x - 1 = 0 ∧ a * y^2 - 4*y - 1 = 0) ↔ 
  (a > -4 ∧ a ≠ 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3839_383927


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3839_383945

/-- Given a geometric sequence {aₙ} with common ratio 2 and a₁ + a₃ = 5, prove that a₂ + a₄ = 10 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  a 1 + a 3 = 5 →               -- given condition
  a 2 + a 4 = 10 :=             -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3839_383945


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_neg_one_l3839_383902

theorem fraction_meaningful_iff_not_neg_one (a : ℝ) :
  (∃ (x : ℝ), x = 2 / (a + 1)) ↔ a ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_neg_one_l3839_383902


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3839_383981

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 2*x > 35) ↔ (x < -5 ∨ x > 7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3839_383981


namespace NUMINAMATH_CALUDE_sum_from_true_discount_and_simple_interest_l3839_383917

theorem sum_from_true_discount_and_simple_interest 
  (S : ℝ) 
  (D I : ℝ) 
  (h1 : D = 75) 
  (h2 : I = 85) 
  (h3 : D / I = (S - D) / S) : S = 637.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_from_true_discount_and_simple_interest_l3839_383917


namespace NUMINAMATH_CALUDE_sqrt_5_irrational_l3839_383964

-- Define the set of numbers
def number_set : Set ℝ := {0.618, 22/7, Real.sqrt 5, -3}

-- Define irrationality
def is_irrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Theorem statement
theorem sqrt_5_irrational : ∃ (x : ℝ), x ∈ number_set ∧ is_irrational x :=
sorry

end NUMINAMATH_CALUDE_sqrt_5_irrational_l3839_383964


namespace NUMINAMATH_CALUDE_victor_sticker_count_l3839_383926

/-- The number of stickers Victor has -/
def total_stickers (flower animal insect space : ℕ) : ℕ :=
  flower + animal + insect + space

theorem victor_sticker_count :
  ∀ (flower animal insect space : ℕ),
    flower = 12 →
    animal = 8 →
    insect = animal - 3 →
    space = flower + 7 →
    total_stickers flower animal insect space = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_victor_sticker_count_l3839_383926


namespace NUMINAMATH_CALUDE_temperature_difference_l3839_383952

theorem temperature_difference (highest lowest : ℤ) (h1 : highest = 8) (h2 : lowest = -1) :
  highest - lowest = 9 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l3839_383952


namespace NUMINAMATH_CALUDE_loop_statement_efficiency_l3839_383956

/-- Enum representing different types of algorithm statements -/
inductive AlgorithmStatement
  | InputOutput
  | Assignment
  | Conditional
  | Loop

/-- Definition of a program's capability to handle large computational problems -/
def CanHandleLargeProblems (statements : List AlgorithmStatement) : Prop :=
  statements.length > 0

/-- Definition of the primary reason for efficient handling of large problems -/
def PrimaryReasonForEfficiency (statement : AlgorithmStatement) (statements : List AlgorithmStatement) : Prop :=
  CanHandleLargeProblems statements ∧ statement ∈ statements

theorem loop_statement_efficiency :
  ∀ (statements : List AlgorithmStatement),
    CanHandleLargeProblems statements →
    AlgorithmStatement.InputOutput ∈ statements →
    AlgorithmStatement.Assignment ∈ statements →
    AlgorithmStatement.Conditional ∈ statements →
    AlgorithmStatement.Loop ∈ statements →
    PrimaryReasonForEfficiency AlgorithmStatement.Loop statements :=
by
  sorry

#check loop_statement_efficiency

end NUMINAMATH_CALUDE_loop_statement_efficiency_l3839_383956


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3839_383950

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = {x : ℝ | x < -2} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3839_383950


namespace NUMINAMATH_CALUDE_consecutive_sum_33_l3839_383960

theorem consecutive_sum_33 (m : ℕ) (h1 : m > 1) :
  (∃ a : ℕ, (Finset.range m).sum (λ i => a + i) = 33) ↔ m = 2 ∨ m = 3 ∨ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_33_l3839_383960


namespace NUMINAMATH_CALUDE_tens_digit_sum_factorials_l3839_383963

def factorial (n : ℕ) : ℕ := sorry

def tensDigit (n : ℕ) : ℕ := sorry

def sumFactorials (n : ℕ) : ℕ := sorry

theorem tens_digit_sum_factorials :
  tensDigit (sumFactorials 100) = 0 := by sorry

end NUMINAMATH_CALUDE_tens_digit_sum_factorials_l3839_383963


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3839_383912

theorem repeating_decimal_sum (x : ℚ) : 
  (∃ (n : ℕ), x = (457 : ℚ) / (10^n * 999)) → 
  (∃ (a b : ℕ), x = a / b ∧ a + b = 1456) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3839_383912


namespace NUMINAMATH_CALUDE_simplify_fraction_l3839_383904

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (45 * b^3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3839_383904


namespace NUMINAMATH_CALUDE_toms_sleep_deficit_l3839_383907

/-- Calculates the sleep deficit for a week given ideal and actual sleep hours -/
def sleep_deficit (ideal_hours : ℕ) (weeknight_hours : ℕ) (weekend_hours : ℕ) : ℕ :=
  let ideal_total := ideal_hours * 7
  let actual_total := weeknight_hours * 5 + weekend_hours * 2
  ideal_total - actual_total

/-- Proves that Tom's sleep deficit for a week is 19 hours -/
theorem toms_sleep_deficit : sleep_deficit 8 5 6 = 19 := by
  sorry

end NUMINAMATH_CALUDE_toms_sleep_deficit_l3839_383907


namespace NUMINAMATH_CALUDE_one_diagonal_implies_four_sides_l3839_383972

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  sides : ℕ
  sides_pos : sides > 0

/-- A diagonal in a polygon is a line segment that connects two non-adjacent vertices. -/
def has_one_diagonal (p : Polygon) : Prop :=
  ∃ (v : ℕ), v < p.sides ∧ (p.sides - v - 2 = 1)

/-- Theorem: A polygon with exactly one diagonal that can be drawn from one vertex has 4 sides. -/
theorem one_diagonal_implies_four_sides (p : Polygon) (h : has_one_diagonal p) : p.sides = 4 := by
  sorry

end NUMINAMATH_CALUDE_one_diagonal_implies_four_sides_l3839_383972


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3839_383980

theorem book_arrangement_theorem :
  let total_books : ℕ := 10
  let spanish_books : ℕ := 4
  let french_books : ℕ := 3
  let german_books : ℕ := 3
  let number_of_units : ℕ := 2 + german_books

  spanish_books + french_books + german_books = total_books →
  (number_of_units.factorial * spanish_books.factorial * french_books.factorial : ℕ) = 17280 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3839_383980


namespace NUMINAMATH_CALUDE_coloring_book_problem_l3839_383989

/-- The number of pictures colored given the initial count and remaining count --/
def pictures_colored (initial_count : ℕ) (remaining_count : ℕ) : ℕ :=
  initial_count - remaining_count

/-- Theorem stating that given two coloring books with 44 pictures each and 68 pictures left to color, 
    the number of pictures colored is 20 --/
theorem coloring_book_problem :
  let book1_count : ℕ := 44
  let book2_count : ℕ := 44
  let total_count : ℕ := book1_count + book2_count
  let remaining_count : ℕ := 68
  pictures_colored total_count remaining_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_problem_l3839_383989


namespace NUMINAMATH_CALUDE_colonization_theorem_l3839_383969

/-- Represents the number of different combinations of planets that can be colonized --/
def colonization_combinations (total_planets : ℕ) (earth_like : ℕ) (mars_like : ℕ) 
  (earth_effort : ℕ) (mars_effort : ℕ) (total_effort : ℕ) : ℕ :=
  (Finset.range (earth_like + 1)).sum (fun a =>
    if 2 * a ≤ total_effort ∧ (total_effort - 2 * a) % 2 = 0 ∧ (total_effort - 2 * a) / 2 ≤ mars_like
    then Nat.choose earth_like a * Nat.choose mars_like ((total_effort - 2 * a) / 2)
    else 0)

/-- The main theorem stating the number of colonization combinations --/
theorem colonization_theorem : 
  colonization_combinations 15 7 8 2 1 16 = 1141 := by sorry

end NUMINAMATH_CALUDE_colonization_theorem_l3839_383969


namespace NUMINAMATH_CALUDE_teacher_engineer_ratio_l3839_383974

theorem teacher_engineer_ratio 
  (t : ℕ) -- number of teachers
  (e : ℕ) -- number of engineers
  (h_total : t + e > 0) -- ensure total group size is positive
  (h_avg : (40 * t + 55 * e) / (t + e) = 45) -- overall average age is 45
  : t = 2 * e := by
sorry

end NUMINAMATH_CALUDE_teacher_engineer_ratio_l3839_383974


namespace NUMINAMATH_CALUDE_num_special_words_is_35280_l3839_383905

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of consonants in the English alphabet -/
def num_consonants : ℕ := 21

/-- The number of six-letter words that begin and end with the same vowel,
    alternate between vowels and consonants, and start with a vowel -/
def num_special_words : ℕ := num_vowels * num_consonants * (num_vowels - 1) * num_consonants * (num_vowels - 1)

/-- Theorem stating that the number of special words is 35280 -/
theorem num_special_words_is_35280 : num_special_words = 35280 := by sorry

end NUMINAMATH_CALUDE_num_special_words_is_35280_l3839_383905


namespace NUMINAMATH_CALUDE_croissants_for_breakfast_l3839_383975

theorem croissants_for_breakfast (total_items cakes pizzas : ℕ) 
  (h1 : total_items = 110)
  (h2 : cakes = 18)
  (h3 : pizzas = 30) :
  total_items - cakes - pizzas = 62 := by
  sorry

end NUMINAMATH_CALUDE_croissants_for_breakfast_l3839_383975


namespace NUMINAMATH_CALUDE_fractional_sides_eq_seven_l3839_383967

/-- A 3-dimensional polyhedron with fractional sides -/
structure Polyhedron where
  F : ℝ  -- number of fractional sides
  D : ℝ  -- number of diagonals
  h1 : D = 2 * F
  h2 : D = F * (F - 3) / 2

/-- The number of fractional sides in the polyhedron is 7 -/
theorem fractional_sides_eq_seven (P : Polyhedron) : P.F = 7 := by
  sorry

end NUMINAMATH_CALUDE_fractional_sides_eq_seven_l3839_383967


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3839_383908

def f (x : ℝ) : ℝ := 4 * x^5 + 13 * x^4 - 30 * x^3 + 8 * x^2

theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = (1 : ℝ) / 2 ∨ x = -2 + 2 * Real.sqrt 2 ∨ x = -2 - 2 * Real.sqrt 2) ∧
  (∃ ε > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < ε → f x / x^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3839_383908


namespace NUMINAMATH_CALUDE_exam_average_l3839_383973

theorem exam_average (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 15 →
  n2 = 10 →
  avg1 = 70 / 100 →
  avg2 = 95 / 100 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 80 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l3839_383973


namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l3839_383961

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Given a two-digit number, insert_zero inserts a 0 between its digits. -/
def insert_zero (n : ℕ) : ℕ := (n / 10) * 100 + (n % 10)

/-- The set of numbers that satisfy the condition in the problem. -/
def solution_set : Set ℕ := {80, 81, 82, 83, 84, 85, 86, 87, 88, 89}

/-- The main theorem that proves the solution to the problem. -/
theorem two_digit_number_theorem (n : ℕ) : 
  TwoDigitNumber n → (insert_zero n = n + 720) → n ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l3839_383961


namespace NUMINAMATH_CALUDE_no_rational_squares_l3839_383983

def sequence_a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => sequence_a n + 2 / sequence_a n

theorem no_rational_squares :
  ∀ n : ℕ, ∀ r : ℚ, sequence_a n ≠ r^2 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_squares_l3839_383983


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3839_383946

theorem trig_identity_proof : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 
  4 * Real.sin (10 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3839_383946


namespace NUMINAMATH_CALUDE_intersection_property_characterization_l3839_383930

/-- A function satisfying the property that the line through any two points
    on its graph intersects the y-axis at (0, pq) -/
def IntersectionProperty (f : ℝ → ℝ) : Prop :=
  ∀ p q : ℝ, p ≠ q →
    let m := (f q - f p) / (q - p)
    let b := f p - m * p
    b = p * q

/-- Theorem stating that functions satisfying the intersection property
    are of the form f(x) = x(c + x) for some constant c -/
theorem intersection_property_characterization (f : ℝ → ℝ) :
  IntersectionProperty f ↔ ∃ c : ℝ, ∀ x : ℝ, f x = x * (c + x) :=
sorry

end NUMINAMATH_CALUDE_intersection_property_characterization_l3839_383930


namespace NUMINAMATH_CALUDE_sara_lunch_cost_l3839_383994

/-- The cost of Sara's lunch given the prices of a hotdog and a salad -/
def lunch_cost (hotdog_price salad_price : ℚ) : ℚ :=
  hotdog_price + salad_price

/-- Theorem stating that Sara's lunch cost is $10.46 -/
theorem sara_lunch_cost :
  lunch_cost 5.36 5.10 = 10.46 := by
  sorry

end NUMINAMATH_CALUDE_sara_lunch_cost_l3839_383994


namespace NUMINAMATH_CALUDE_triangle_area_solution_l3839_383998

theorem triangle_area_solution (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * (2*x) * x = 50 → x = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_solution_l3839_383998


namespace NUMINAMATH_CALUDE_inequality_proof_l3839_383991

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  Real.sqrt (a * b^2 + a^2 * b) + Real.sqrt ((1 - a) * (1 - b)^2 + (1 - a)^2 * (1 - b)) < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3839_383991


namespace NUMINAMATH_CALUDE_fraction_simplification_l3839_383935

theorem fraction_simplification (x : ℝ) : (2*x + 3) / 4 + (4 - 2*x) / 3 = (-2*x + 25) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3839_383935


namespace NUMINAMATH_CALUDE_sixty_degrees_in_clerts_l3839_383978

/-- The number of clerts in a full circle on Venus -/
def venus_full_circle : ℚ := 800

/-- The number of degrees in a full circle on Earth -/
def earth_full_circle : ℚ := 360

/-- Converts degrees to clerts on Venus -/
def degrees_to_clerts (degrees : ℚ) : ℚ :=
  (degrees / earth_full_circle) * venus_full_circle

/-- Theorem: 60 degrees is equivalent to 133.3 (repeating) clerts on Venus -/
theorem sixty_degrees_in_clerts :
  degrees_to_clerts 60 = 133 + 1/3 := by sorry

end NUMINAMATH_CALUDE_sixty_degrees_in_clerts_l3839_383978


namespace NUMINAMATH_CALUDE_complex_cube_problem_l3839_383993

theorem complex_cube_problem :
  ∀ (x y : ℕ+) (c : ℤ),
    (x : ℂ) + y * Complex.I ≠ 1 + 6 * Complex.I →
    ((x : ℂ) + y * Complex.I) ^ 3 ≠ -107 + c * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_problem_l3839_383993


namespace NUMINAMATH_CALUDE_choir_performance_theorem_l3839_383924

/-- Represents the number of singers joining in each verse of a choir performance --/
structure ChoirPerformance where
  total : ℕ
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- Calculates the number of singers joining in the fifth verse --/
def fifthVerseSingers (c : ChoirPerformance) : ℕ :=
  c.total - (c.first + c.second + c.third + c.fourth)

/-- Theorem stating the number of singers joining in the fifth verse --/
theorem choir_performance_theorem (c : ChoirPerformance) 
  (h_total : c.total = 60)
  (h_first : c.first = c.total / 2)
  (h_second : c.second = (c.total - c.first) / 3)
  (h_third : c.third = (c.total - c.first - c.second) / 4)
  (h_fourth : c.fourth = (c.total - c.first - c.second - c.third) / 5) :
  fifthVerseSingers c = 12 := by
  sorry

#eval fifthVerseSingers { total := 60, first := 30, second := 10, third := 5, fourth := 3, fifth := 12 }

end NUMINAMATH_CALUDE_choir_performance_theorem_l3839_383924


namespace NUMINAMATH_CALUDE_jacks_hair_length_l3839_383948

/-- Given the relative lengths of Kate's, Emily's, Logan's, and Jack's hair, prove that Jack's hair is 39 inches long. -/
theorem jacks_hair_length (logan_hair emily_hair kate_hair jack_hair : ℝ) : 
  logan_hair = 20 →
  emily_hair = logan_hair + 6 →
  kate_hair = emily_hair / 2 →
  jack_hair = 3 * kate_hair →
  jack_hair = 39 :=
by
  sorry

#check jacks_hair_length

end NUMINAMATH_CALUDE_jacks_hair_length_l3839_383948


namespace NUMINAMATH_CALUDE_equation_roots_l3839_383925

theorem equation_roots :
  let f : ℝ → ℝ := λ x => (x^3 - 3*x^2 + x - 2)*(x^3 - x^2 - 4*x + 7) + 6*x^2 - 15*x + 18
  ∃ (a b c d e : ℝ), 
    (a = 1 ∧ b = 2 ∧ c = -2 ∧ d = 1 + Real.sqrt 2 ∧ e = 1 - Real.sqrt 2) ∧
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l3839_383925


namespace NUMINAMATH_CALUDE_max_true_statements_l3839_383962

theorem max_true_statements (x : ℝ) : 
  let statements := [
    (0 < x^2 ∧ x^2 < 2),
    (x^2 > 2),
    (-2 < x ∧ x < 0),
    (0 < x ∧ x < 2),
    (0 < x - x^2 ∧ x - x^2 < 2)
  ]
  (∀ (s : Finset (Fin 5)), s.card > 3 → ¬(∀ i ∈ s, statements[i]))
  ∧
  (∃ (s : Finset (Fin 5)), s.card = 3 ∧ (∀ i ∈ s, statements[i])) :=
by sorry

#check max_true_statements

end NUMINAMATH_CALUDE_max_true_statements_l3839_383962


namespace NUMINAMATH_CALUDE_marble_count_l3839_383920

theorem marble_count (yellow : ℕ) (blue : ℕ) (red : ℕ) : 
  yellow = 5 →
  blue * 4 = red * 3 →
  red = yellow + 3 →
  yellow + blue + red = 19 := by
sorry

end NUMINAMATH_CALUDE_marble_count_l3839_383920


namespace NUMINAMATH_CALUDE_line_mb_value_l3839_383979

/-- A line passing through (-1, -3) and intersecting the y-axis at y = -1 has mb = 2 -/
theorem line_mb_value (m b : ℝ) : 
  (∀ x y, y = m * x + b → (x = -1 ∧ y = -3) ∨ (x = 0 ∧ y = -1)) → 
  m * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_mb_value_l3839_383979


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3839_383997

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f := λ x : ℝ => a^(x - 1) + 4
  f 1 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3839_383997


namespace NUMINAMATH_CALUDE_intersecting_spheres_equal_volumes_l3839_383900

theorem intersecting_spheres_equal_volumes (r : ℝ) (d : ℝ) : 
  r = 1 → 
  0 < d ∧ d < 2 * r →
  (4 * π * r^3 / 3 - π * (r - d / 2)^2 * (2 * r + d / 2) / 3) * 2 = 4 * π * r^3 / 3 →
  d = 4 * Real.cos (4 * π / 9) :=
sorry

end NUMINAMATH_CALUDE_intersecting_spheres_equal_volumes_l3839_383900


namespace NUMINAMATH_CALUDE_existence_of_counterexample_l3839_383901

theorem existence_of_counterexample : ∃ m n : ℝ, m > n ∧ m^2 ≤ n^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_counterexample_l3839_383901


namespace NUMINAMATH_CALUDE_cats_awake_l3839_383936

theorem cats_awake (total : ℕ) (asleep : ℕ) (h1 : total = 98) (h2 : asleep = 92) :
  total - asleep = 6 := by
  sorry

end NUMINAMATH_CALUDE_cats_awake_l3839_383936


namespace NUMINAMATH_CALUDE_infinite_power_tower_eq_four_solution_l3839_383949

/-- Define the infinite power tower function --/
noncomputable def infinitePowerTower (x : ℝ) : ℝ :=
  Real.log x / Real.log 2

/-- Theorem: The solution to x^(x^(x^...)) = 4 is √2 --/
theorem infinite_power_tower_eq_four_solution :
  ∀ x : ℝ, x > 0 → infinitePowerTower x = 4 → x = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_infinite_power_tower_eq_four_solution_l3839_383949


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l3839_383909

def M : ℕ := 36 * 36 * 98 * 150

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors_M :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 62 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l3839_383909


namespace NUMINAMATH_CALUDE_coin_bill_combinations_l3839_383943

theorem coin_bill_combinations : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + 5 * p.2 = 207) (Finset.product (Finset.range 104) (Finset.range 42))).card :=
by
  sorry

end NUMINAMATH_CALUDE_coin_bill_combinations_l3839_383943


namespace NUMINAMATH_CALUDE_smallest_reachable_integer_l3839_383906

-- Define the sequence u_n
def u : ℕ → ℕ
  | 0 => 2010^2010
  | (n+1) => if u n % 2 = 1 then u n + 7 else u n / 2

-- Define the property of being reachable by the sequence
def Reachable (m : ℕ) : Prop := ∃ n, u n = m

-- State the theorem
theorem smallest_reachable_integer : 
  (∃ m, Reachable m) ∧ (∀ k, Reachable k → k ≥ 1) := by sorry

end NUMINAMATH_CALUDE_smallest_reachable_integer_l3839_383906


namespace NUMINAMATH_CALUDE_tan_double_angle_special_point_l3839_383939

theorem tan_double_angle_special_point (α : Real) :
  (∃ (x y : Real), x = 1 ∧ y = -2 ∧ x * Real.cos α = y * Real.sin α) →
  Real.tan (2 * α) = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_point_l3839_383939


namespace NUMINAMATH_CALUDE_jordons_machine_l3839_383954

theorem jordons_machine (x : ℝ) : 2 * x + 3 = 27 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_jordons_machine_l3839_383954


namespace NUMINAMATH_CALUDE_circle_radius_zero_l3839_383942

theorem circle_radius_zero (x y : ℝ) :
  x^2 - 4*x + y^2 - 6*y + 13 = 0 → (∃ r : ℝ, r = 0 ∧ (x - 2)^2 + (y - 3)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l3839_383942


namespace NUMINAMATH_CALUDE_value_set_of_x_l3839_383938

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- State the theorem
theorem value_set_of_x (x : ℝ) :
  (∀ a : ℝ, a ≠ 0 → f x ≥ (|a + 1| - |2*a - 1|) / |a|) →
  x ≤ -1 ∨ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_value_set_of_x_l3839_383938


namespace NUMINAMATH_CALUDE_tangent_line_equation_inequality_l3839_383940

-- Define the function f(x) = x ln(x+1)
noncomputable def f (x : ℝ) : ℝ := x * Real.log (x + 1)

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  let slope := Real.log 2 + 1 / 2
  let y_intercept := -1 / 2
  (fun x => slope * x + y_intercept) 1 = f 1 ∧
  HasDerivAt f slope 1 :=
sorry

-- Theorem for the inequality
theorem inequality (x : ℝ) (h : x > -1) :
  f x + (1/2) * x^3 ≥ x^2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_inequality_l3839_383940


namespace NUMINAMATH_CALUDE_sandwich_cost_is_two_l3839_383941

/-- Calculates the cost per sandwich given the prices and discounts for ingredients -/
def cost_per_sandwich (bread_price : ℚ) (meat_price : ℚ) (cheese_price : ℚ) 
  (meat_discount : ℚ) (cheese_discount : ℚ) (num_sandwiches : ℕ) : ℚ :=
  let total_cost := bread_price + 2 * meat_price + 2 * cheese_price - meat_discount - cheese_discount
  total_cost / num_sandwiches

/-- Proves that the cost per sandwich is $2.00 given the specified conditions -/
theorem sandwich_cost_is_two :
  cost_per_sandwich 4 5 4 1 1 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_is_two_l3839_383941


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l3839_383928

/-- The area of an equilateral triangle with altitude √15 is 5√3 -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = Real.sqrt 15) :
  let base := 2 * Real.sqrt 5
  let area := (1 / 2) * base * h
  area = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l3839_383928


namespace NUMINAMATH_CALUDE_last_two_digits_product_l3839_383921

def last_two_digits (n : ℤ) : ℤ × ℤ :=
  let tens := (n / 10) % 10
  let ones := n % 10
  (tens, ones)

def sum_last_two_digits (n : ℤ) : ℤ :=
  let (tens, ones) := last_two_digits n
  tens + ones

def product_last_two_digits (n : ℤ) : ℤ :=
  let (tens, ones) := last_two_digits n
  tens * ones

theorem last_two_digits_product (n : ℤ) :
  n % 5 = 0 → sum_last_two_digits n = 14 → product_last_two_digits n = 45 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l3839_383921


namespace NUMINAMATH_CALUDE_frustum_smaller_cone_height_l3839_383995

-- Define the frustum
structure Frustum where
  height : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

-- Define the theorem
theorem frustum_smaller_cone_height (f : Frustum) 
  (h1 : f.height = 18)
  (h2 : f.lower_base_area = 144 * Real.pi)
  (h3 : f.upper_base_area = 16 * Real.pi) :
  ∃ (smaller_cone_height : ℝ), smaller_cone_height = 9 := by
  sorry

end NUMINAMATH_CALUDE_frustum_smaller_cone_height_l3839_383995


namespace NUMINAMATH_CALUDE_cube_has_eight_vertices_l3839_383966

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- The number of vertices in a cube -/
def num_vertices (c : Cube) : ℕ := 8

/-- Theorem: A cube has 8 vertices -/
theorem cube_has_eight_vertices (c : Cube) : num_vertices c = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_has_eight_vertices_l3839_383966


namespace NUMINAMATH_CALUDE_find_number_l3839_383931

theorem find_number : ∃ x : ℝ, x = 50 ∧ (0.6 * x = 0.5 * 30 + 15) := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3839_383931


namespace NUMINAMATH_CALUDE_l_shaped_playground_area_l3839_383947

def large_rectangle_length : ℕ := 10
def large_rectangle_width : ℕ := 7
def small_rectangle_length : ℕ := 3
def small_rectangle_width : ℕ := 2
def num_small_rectangles : ℕ := 2

theorem l_shaped_playground_area :
  (large_rectangle_length * large_rectangle_width) -
  (num_small_rectangles * small_rectangle_length * small_rectangle_width) = 58 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_playground_area_l3839_383947


namespace NUMINAMATH_CALUDE_power_three_mod_eleven_l3839_383976

theorem power_three_mod_eleven : 3^2048 % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_eleven_l3839_383976


namespace NUMINAMATH_CALUDE_cos_sin_difference_l3839_383903

theorem cos_sin_difference (α β : Real) 
  (h : Real.cos (α + β) * Real.cos (α - β) = 1/3) : 
  Real.cos α ^ 2 - Real.sin β ^ 2 = 1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_sin_difference_l3839_383903


namespace NUMINAMATH_CALUDE_original_denominator_problem_l3839_383944

theorem original_denominator_problem (d : ℤ) : 
  (3 : ℚ) / d ≠ 0 →
  (9 : ℚ) / (d + 7) = (1 : ℚ) / 3 →
  d = 20 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l3839_383944


namespace NUMINAMATH_CALUDE_c_invests_after_eight_months_l3839_383986

/-- Represents the investment problem with three partners A, B, and C --/
structure InvestmentProblem where
  initial_investment : ℝ
  annual_gain : ℝ
  a_share : ℝ
  b_invest_time : ℕ
  c_invest_time : ℕ

/-- Calculates the time when C invests given the problem parameters --/
def calculate_c_invest_time (problem : InvestmentProblem) : ℕ :=
  let a_investment := problem.initial_investment * 12
  let b_investment := 2 * problem.initial_investment * (12 - problem.b_invest_time)
  let c_investment := 3 * problem.initial_investment * problem.c_invest_time
  let total_investment := a_investment + b_investment + c_investment
  problem.c_invest_time

/-- Theorem stating that C invests after 8 months --/
theorem c_invests_after_eight_months (problem : InvestmentProblem) 
  (h1 : problem.annual_gain = 21000)
  (h2 : problem.a_share = 7000)
  (h3 : problem.b_invest_time = 6) :
  calculate_c_invest_time problem = 8 := by
  sorry

#eval calculate_c_invest_time {
  initial_investment := 1000,
  annual_gain := 21000,
  a_share := 7000,
  b_invest_time := 6,
  c_invest_time := 8
}

end NUMINAMATH_CALUDE_c_invests_after_eight_months_l3839_383986


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l3839_383971

theorem complementary_angles_difference (x : ℝ) (h1 : 4 * x + x = 90) (h2 : x > 0) : |4 * x - x| = 54 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l3839_383971


namespace NUMINAMATH_CALUDE_gcd_plus_ten_l3839_383970

theorem gcd_plus_ten (a b : ℕ) (h : a = 8436 ∧ b = 156) :
  (Nat.gcd a b) + 10 = 22 := by
  sorry

end NUMINAMATH_CALUDE_gcd_plus_ten_l3839_383970


namespace NUMINAMATH_CALUDE_rank_of_Mn_l3839_383968

/-- Definition of the matrix Mn -/
def Mn (n : ℕ+) : Matrix (Fin (2*n+1)) (Fin (2*n+1)) ℤ :=
  Matrix.of fun i j =>
    if i = j then 0
    else if i > j then
      if i - j ≤ n then 1 else -1
    else
      if j - i ≤ n then -1 else 1

/-- The rank of Mn is 2n for any positive integer n -/
theorem rank_of_Mn (n : ℕ+) : Matrix.rank (Mn n) = 2*n := by sorry

end NUMINAMATH_CALUDE_rank_of_Mn_l3839_383968


namespace NUMINAMATH_CALUDE_village_population_problem_l3839_383951

theorem village_population_problem (X : ℝ) : 
  (X > 0) →
  (0.9 * X * 0.75 + 0.9 * X * 0.25 * 0.15 = 5265) →
  X = 7425 := by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l3839_383951
