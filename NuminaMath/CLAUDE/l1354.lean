import Mathlib

namespace NUMINAMATH_CALUDE_like_terms_imply_exponent_relation_l1354_135417

/-- Given that -25a^(2m)b and 7a^4b^(3-n) are like terms, prove that 2m - n = 2 -/
theorem like_terms_imply_exponent_relation (a b : ℝ) (m n : ℕ) 
  (h : ∃ (k : ℝ), -25 * a^(2*m) * b = k * (7 * a^4 * b^(3-n))) : 
  2 * m - n = 2 :=
sorry

end NUMINAMATH_CALUDE_like_terms_imply_exponent_relation_l1354_135417


namespace NUMINAMATH_CALUDE_linear_equation_exponent_values_l1354_135460

theorem linear_equation_exponent_values (m n : ℤ) : 
  (∀ x y : ℝ, ∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ a * x + b * y + c = 5 * x^(3*m-2*n) - 2 * y^(n-m) + 11) →
  m = 0 ∧ n = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_values_l1354_135460


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1354_135479

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≤ 1) ↔ (∃ x : ℝ, x^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1354_135479


namespace NUMINAMATH_CALUDE_matching_socks_probability_l1354_135406

def gray_socks : ℕ := 10
def white_socks : ℕ := 8
def blue_socks : ℕ := 6

def total_socks : ℕ := gray_socks + white_socks + blue_socks

theorem matching_socks_probability :
  (Nat.choose gray_socks 2 + Nat.choose white_socks 2 + Nat.choose blue_socks 2) /
  Nat.choose total_socks 2 = 22 / 69 :=
by sorry

end NUMINAMATH_CALUDE_matching_socks_probability_l1354_135406


namespace NUMINAMATH_CALUDE_last_part_distance_calculation_l1354_135458

/-- Calculates the distance of the last part of a trip given the total distance,
    first part distance, speeds for different parts, and average speed. -/
def last_part_distance (total_distance first_part_distance first_part_speed
                        average_speed last_part_speed : ℝ) : ℝ :=
  total_distance - first_part_distance

theorem last_part_distance_calculation (total_distance : ℝ) (first_part_distance : ℝ)
    (first_part_speed : ℝ) (average_speed : ℝ) (last_part_speed : ℝ)
    (h1 : total_distance = 100)
    (h2 : first_part_distance = 30)
    (h3 : first_part_speed = 60)
    (h4 : average_speed = 40)
    (h5 : last_part_speed = 35) :
  last_part_distance total_distance first_part_distance first_part_speed average_speed last_part_speed = 70 := by
sorry

end NUMINAMATH_CALUDE_last_part_distance_calculation_l1354_135458


namespace NUMINAMATH_CALUDE_complement_of_intersection_l1354_135456

def A : Set ℕ := {4, 5, 7}
def B : Set ℕ := {3, 4, 7, 8}
def U : Set ℕ := A ∪ B

theorem complement_of_intersection (A B : Set ℕ) (U : Set ℕ) (h : U = A ∪ B) :
  (A ∩ B)ᶜ = {3, 5, 8} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l1354_135456


namespace NUMINAMATH_CALUDE_guests_per_table_l1354_135478

theorem guests_per_table (tables : ℝ) (total_guests : ℕ) 
  (h1 : tables = 252.0) 
  (h2 : total_guests = 1008) : 
  (total_guests : ℝ) / tables = 4 := by
  sorry

end NUMINAMATH_CALUDE_guests_per_table_l1354_135478


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1354_135454

/-- Given a geometric sequence {a_n} where a₂a₆ + a₄² = π, prove that a₃a₅ = π/2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_eq : a 2 * a 6 + a 4 * a 4 = Real.pi) : a 3 * a 5 = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1354_135454


namespace NUMINAMATH_CALUDE_correct_reading_of_6005_l1354_135432

/-- Represents the correct way to read a number between 1000 and 9999 -/
def ReadNumber (n : ℕ) : String :=
  sorry

/-- The correct reading of 6005 -/
def Correct6005Reading : String :=
  ReadNumber 6005

/-- The incorrect reading of 6005 -/
def Incorrect6005Reading : String :=
  "six thousand zero zero five"

theorem correct_reading_of_6005 : 
  Correct6005Reading ≠ Incorrect6005Reading :=
sorry

end NUMINAMATH_CALUDE_correct_reading_of_6005_l1354_135432


namespace NUMINAMATH_CALUDE_min_distance_parallel_lines_l1354_135463

/-- The minimum distance between two parallel lines -/
theorem min_distance_parallel_lines : 
  let line1 := {(x, y) : ℝ × ℝ | 3 * x + 4 * y - 10 = 0}
  let line2 := {(x, y) : ℝ × ℝ | 6 * x + 8 * y + 5 = 0}
  ∃ d : ℝ, d = (5 : ℝ) / 2 ∧ 
    ∀ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ line1 → Q ∈ line2 → 
      d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_parallel_lines_l1354_135463


namespace NUMINAMATH_CALUDE_exists_sum_of_scores_with_two_ways_l1354_135412

/-- Represents a scoring configuration for the modified AMC test. -/
structure ScoringConfig where
  total_questions : ℕ
  correct_points : ℕ
  unanswered_points : ℕ
  incorrect_points : ℕ

/-- Represents an answer combination for the test. -/
structure AnswerCombination where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- Calculates the score for a given answer combination under a specific scoring config. -/
def calculate_score (config : ScoringConfig) (answers : AnswerCombination) : ℕ :=
  answers.correct * config.correct_points + answers.unanswered * config.unanswered_points + answers.incorrect * config.incorrect_points

/-- Checks if an answer combination is valid for a given total number of questions. -/
def is_valid_combination (total_questions : ℕ) (answers : AnswerCombination) : Prop :=
  answers.correct + answers.unanswered + answers.incorrect = total_questions

/-- Defines the specific scoring configuration for the problem. -/
def amc_scoring : ScoringConfig :=
  { total_questions := 20
  , correct_points := 7
  , unanswered_points := 3
  , incorrect_points := 0 }

/-- Theorem stating the existence of a sum of scores meeting the problem criteria. -/
theorem exists_sum_of_scores_with_two_ways :
  ∃ (sum : ℕ), 
    (∃ (scores : List ℕ),
      (∀ score ∈ scores, 
        score ≤ 140 ∧ 
        (∃ (ways : List AnswerCombination), 
          ways.length = 2 ∧
          (∀ way ∈ ways, 
            is_valid_combination amc_scoring.total_questions way ∧
            calculate_score amc_scoring way = score))) ∧
      sum = scores.sum) := by
  sorry


end NUMINAMATH_CALUDE_exists_sum_of_scores_with_two_ways_l1354_135412


namespace NUMINAMATH_CALUDE_seconds_in_3h45m_is_13500_l1354_135453

/-- Converts hours to minutes -/
def hours_to_minutes (h : ℕ) : ℕ := h * 60

/-- Converts minutes to seconds -/
def minutes_to_seconds (m : ℕ) : ℕ := m * 60

/-- The number of seconds in 3 hours and 45 minutes -/
def seconds_in_3h45m : ℕ := minutes_to_seconds (hours_to_minutes 3 + 45)

theorem seconds_in_3h45m_is_13500 : seconds_in_3h45m = 13500 := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_3h45m_is_13500_l1354_135453


namespace NUMINAMATH_CALUDE_total_molecular_weight_l1354_135498

/-- Atomic weight of Aluminium in g/mol -/
def Al : ℝ := 26.98

/-- Atomic weight of Oxygen in g/mol -/
def O : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def H : ℝ := 1.01

/-- Atomic weight of Sodium in g/mol -/
def Na : ℝ := 22.99

/-- Atomic weight of Chlorine in g/mol -/
def Cl : ℝ := 35.45

/-- Atomic weight of Calcium in g/mol -/
def Ca : ℝ := 40.08

/-- Atomic weight of Carbon in g/mol -/
def C : ℝ := 12.01

/-- Molecular weight of Aluminium hydroxide in g/mol -/
def Al_OH_3 : ℝ := Al + 3 * O + 3 * H

/-- Molecular weight of Sodium chloride in g/mol -/
def NaCl : ℝ := Na + Cl

/-- Molecular weight of Calcium carbonate in g/mol -/
def CaCO_3 : ℝ := Ca + C + 3 * O

/-- Total molecular weight of the given compounds in grams -/
def total_weight : ℝ := 4 * Al_OH_3 + 2 * NaCl + 3 * CaCO_3

theorem total_molecular_weight : total_weight = 729.19 := by
  sorry

end NUMINAMATH_CALUDE_total_molecular_weight_l1354_135498


namespace NUMINAMATH_CALUDE_seans_apples_l1354_135472

/-- Sean's apple problem -/
theorem seans_apples (initial_apples final_apples susans_apples : ℕ) :
  final_apples = initial_apples + susans_apples →
  susans_apples = 8 →
  final_apples = 17 →
  initial_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_seans_apples_l1354_135472


namespace NUMINAMATH_CALUDE_interest_rate_problem_l1354_135493

/-- Given a sum P at simple interest rate R for 3 years, if increasing the rate by 1%
    results in Rs. 78 more interest, then P = 2600. -/
theorem interest_rate_problem (P R : ℝ) (h : P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 78) :
  P = 2600 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l1354_135493


namespace NUMINAMATH_CALUDE_one_minus_repeating_third_eq_two_thirds_l1354_135488

/-- The repeating decimal 0.333... -/
def repeating_third : ℚ := 1/3

/-- Theorem stating that 1 minus the repeating decimal 0.333... equals 2/3 -/
theorem one_minus_repeating_third_eq_two_thirds :
  1 - repeating_third = 2/3 := by sorry

end NUMINAMATH_CALUDE_one_minus_repeating_third_eq_two_thirds_l1354_135488


namespace NUMINAMATH_CALUDE_point_2023_coordinates_l1354_135445

/-- The y-coordinate of the nth point in the sequence -/
def y_coord (n : ℕ) : ℤ :=
  match n % 4 with
  | 0 => 0
  | 1 => 1
  | 2 => 0
  | 3 => -1
  | _ => 0  -- This case is technically unreachable, but Lean requires it

/-- The sequence of points as described in the problem -/
def point_sequence (n : ℕ) : ℕ × ℤ :=
  (n, y_coord (n + 1))

theorem point_2023_coordinates :
  point_sequence 2022 = (2022, 0) := by sorry

end NUMINAMATH_CALUDE_point_2023_coordinates_l1354_135445


namespace NUMINAMATH_CALUDE_divisor_problem_l1354_135467

theorem divisor_problem (D : ℚ) : 
  (1280 + 720) / 125 = 7392 / D → D = 462 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1354_135467


namespace NUMINAMATH_CALUDE_divisors_of_n_squared_less_than_n_not_dividing_n_l1354_135455

def n : ℕ := 2^33 * 5^21

-- Function to count divisors of a number
def count_divisors (m : ℕ) : ℕ := sorry

-- Function to count divisors of m less than n
def count_divisors_less_than (m n : ℕ) : ℕ := sorry

theorem divisors_of_n_squared_less_than_n_not_dividing_n :
  count_divisors_less_than (n^2) n - count_divisors n = 692 := by sorry

end NUMINAMATH_CALUDE_divisors_of_n_squared_less_than_n_not_dividing_n_l1354_135455


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1354_135436

theorem polynomial_evaluation (a : ℝ) (h : a = 2) : (7*a^2 - 20*a + 5) * (3*a - 4) = -14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1354_135436


namespace NUMINAMATH_CALUDE_loss_fraction_for_apple_l1354_135457

/-- Calculates the fraction of loss given the cost price and selling price -/
def fractionOfLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice

/-- Theorem stating that for given cost price 17 and selling price 16, 
    the fraction of loss is 1/17 -/
theorem loss_fraction_for_apple : 
  fractionOfLoss 17 16 = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_loss_fraction_for_apple_l1354_135457


namespace NUMINAMATH_CALUDE_simultaneous_arrival_l1354_135409

/-- Represents a point on the shore of the circular lake -/
structure Pier where
  point : ℝ × ℝ

/-- Represents a boat with a starting position and speed -/
structure Boat where
  start : Pier
  speed : ℝ

/-- Represents the circular lake with four piers -/
structure Lake where
  k : Pier
  l : Pier
  p : Pier
  q : Pier

/-- Represents the collision point of two boats -/
def collision_point (b1 b2 : Boat) (dest1 dest2 : Pier) : ℝ × ℝ := sorry

/-- Time taken for a boat to reach its destination -/
def time_to_destination (b : Boat) (dest : Pier) : ℝ := sorry

/-- Main theorem: If boats collide when going to opposite piers,
    they will reach swapped destinations simultaneously -/
theorem simultaneous_arrival (lake : Lake) (boat : Boat) (rowboat : Boat) :
  let x := collision_point boat rowboat lake.p lake.q
  boat.start = lake.k →
  rowboat.start = lake.l →
  time_to_destination boat lake.q = time_to_destination rowboat lake.p := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_arrival_l1354_135409


namespace NUMINAMATH_CALUDE_soap_brand_ratio_l1354_135466

theorem soap_brand_ratio (total : ℕ) (neither : ℕ) (only_a : ℕ) (both : ℕ) 
  (h1 : total = 300)
  (h2 : neither = 80)
  (h3 : only_a = 60)
  (h4 : both = 40) :
  (total - neither - only_a - both) / both = 3 := by
  sorry

end NUMINAMATH_CALUDE_soap_brand_ratio_l1354_135466


namespace NUMINAMATH_CALUDE_pool_capacity_l1354_135416

theorem pool_capacity (C : ℝ) 
  (h1 : 0.55 * C + 300 = 0.85 * C) : C = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l1354_135416


namespace NUMINAMATH_CALUDE_quadratic_equation_real_solutions_l1354_135439

theorem quadratic_equation_real_solutions (x y z : ℝ) :
  (∃ z, 16 * z^2 + 4 * x * y * z + (y^2 - 3) = 0) ↔ x ≤ -2 ∨ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_solutions_l1354_135439


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1354_135431

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Represents a solution in the form (m ± √n)/p -/
structure QuadraticSolution where
  m : ℚ
  n : ℕ
  p : ℕ

/-- Check if three numbers are coprime -/
def are_coprime (m : ℚ) (n p : ℕ) : Prop :=
  Nat.gcd (Nat.gcd (Int.natAbs (Rat.num m)) n) p = 1

/-- The main theorem -/
theorem quadratic_solution_sum (eq : QuadraticEquation) (sol : QuadraticSolution) :
  eq.a = 3 ∧ eq.b = -7 ∧ eq.c = 3 ∧
  are_coprime sol.m sol.n sol.p ∧
  (∃ x : ℚ, x * (3 * x - 7) = -3 ∧ 
    (x = (sol.m + Real.sqrt sol.n) / sol.p ∨ 
     x = (sol.m - Real.sqrt sol.n) / sol.p)) →
  sol.m + sol.n + sol.p = 26 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1354_135431


namespace NUMINAMATH_CALUDE_point_set_classification_l1354_135489

-- Define the type for 2D points
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance squared between two points
def distanceSquared (p q : Point2D) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

-- Define the equation
def satisfiesEquation (X : Point2D) (A : List Point2D) (k : List ℝ) (c : ℝ) : Prop :=
  (List.zip k A).foldl (λ sum (kᵢ, Aᵢ) => sum + kᵢ * distanceSquared Aᵢ X) 0 = c

-- State the theorem
theorem point_set_classification 
  (A : List Point2D) (k : List ℝ) (c : ℝ) 
  (h_length : A.length = k.length) :
  (k.sum ≠ 0 → 
    (∃ center : Point2D, ∃ radius : ℝ, 
      ∀ X, satisfiesEquation X A k c ↔ distanceSquared center X = radius^2) ∨
    (∀ X, ¬satisfiesEquation X A k c)) ∧
  (k.sum = 0 → 
    (∃ a b d : ℝ, ∀ X, satisfiesEquation X A k c ↔ a * X.x + b * X.y = d) ∨
    (∀ X, ¬satisfiesEquation X A k c)) :=
sorry

end NUMINAMATH_CALUDE_point_set_classification_l1354_135489


namespace NUMINAMATH_CALUDE_parabola_line_intersection_slope_range_l1354_135440

theorem parabola_line_intersection_slope_range :
  ∀ k : ℝ,
  (∃ A B : ℝ × ℝ, A ≠ B ∧
    (A.2)^2 = 4 * A.1 ∧
    (B.2)^2 = 4 * B.1 ∧
    A.2 = k * (A.1 + 2) ∧
    B.2 = k * (B.1 + 2)) ↔
  (k ∈ Set.Ioo (- Real.sqrt 2 / 2) 0 ∪ Set.Ioo 0 (Real.sqrt 2 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_slope_range_l1354_135440


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1354_135426

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x ≤ 0} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1354_135426


namespace NUMINAMATH_CALUDE_cone_trajectory_length_l1354_135415

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cone -/
structure Cone where
  base_side_length : ℝ
  apex : Point3D
  base_center : Point3D

/-- The theorem statement -/
theorem cone_trajectory_length 
  (c : Cone) 
  (h_base_side : c.base_side_length = 2) 
  (M : Point3D) 
  (h_M_midpoint : M = Point3D.mk 0 0 ((c.apex.z - c.base_center.z) / 2)) 
  (A : Point3D) 
  (h_A_on_base : A.z = c.base_center.z ∧ (A.x - c.base_center.x)^2 + (A.y - c.base_center.y)^2 = 1) 
  (P : Point3D → Prop) 
  (h_P_on_base : ∀ p, P p → p.z = c.base_center.z ∧ (p.x - c.base_center.x)^2 + (p.y - c.base_center.y)^2 ≤ 1) 
  (h_AM_perp_MP : ∀ p, P p → (M.x - A.x) * (p.x - M.x) + (M.y - A.y) * (p.y - M.y) + (M.z - A.z) * (p.z - M.z) = 0) :
  (∃ l : ℝ, l = Real.sqrt 7 / 2 ∧ 
    ∀ ε > 0, ∃ δ > 0, ∀ p q, P p → P q → abs (p.x - q.x) < δ ∧ abs (p.y - q.y) < δ → 
      abs (Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2) - l) < ε) :=
by sorry

end NUMINAMATH_CALUDE_cone_trajectory_length_l1354_135415


namespace NUMINAMATH_CALUDE_tv_show_episodes_l1354_135474

/-- Given a TV show with the following properties:
  - There were 9 seasons before a new season was announced
  - The last (10th) season has 4 more episodes than the others
  - Each episode is 0.5 hours long
  - It takes 112 hours to watch all episodes after the last season finishes
  This theorem proves that each season (except the last) has 22 episodes. -/
theorem tv_show_episodes :
  let seasons_before : ℕ := 9
  let extra_episodes_last_season : ℕ := 4
  let episode_length : ℚ := 1/2
  let total_watch_time : ℕ := 112
  let episodes_per_season : ℕ := (2 * total_watch_time - 2 * extra_episodes_last_season) / (2 * seasons_before + 2)
  episodes_per_season = 22 := by
sorry

end NUMINAMATH_CALUDE_tv_show_episodes_l1354_135474


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1354_135497

theorem quadratic_form_sum (x : ℝ) : ∃ (d e : ℝ), 
  (∀ x, x^2 - 24*x + 50 = (x + d)^2 + e) ∧ d + e = -106 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1354_135497


namespace NUMINAMATH_CALUDE_sandys_puppies_l1354_135405

/-- Given that Sandy initially had 8 puppies and gave away 4,
    prove that she now has 4 puppies. -/
theorem sandys_puppies :
  let initial_puppies : ℕ := 8
  let puppies_given_away : ℕ := 4
  let remaining_puppies := initial_puppies - puppies_given_away
  remaining_puppies = 4 :=
by sorry

end NUMINAMATH_CALUDE_sandys_puppies_l1354_135405


namespace NUMINAMATH_CALUDE_cost_function_property_l1354_135477

/-- A function representing the cost with respect to some parameter b -/
def cost_function (f : ℝ → ℝ) : Prop :=
  ∀ b : ℝ, f (2 * b) = 16 * f b

/-- Theorem stating that if doubling the input results in a cost that is 1600% of the original,
    then f(2b) = 16f(b) for any value of b -/
theorem cost_function_property (f : ℝ → ℝ) (h : ∀ b : ℝ, f (2 * b) = 16 * f b) :
  cost_function f := by sorry

end NUMINAMATH_CALUDE_cost_function_property_l1354_135477


namespace NUMINAMATH_CALUDE_find_b_l1354_135418

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A (b : ℝ) : Set ℝ := {x | 1 ≤ x ∧ x < b}

-- Define the complement of A in U
def complement_A (b : ℝ) : Set ℝ := {x | x < 1 ∨ x ≥ 2}

-- Theorem statement
theorem find_b : ∃ b : ℝ, A b = Set.compl (complement_A b) := by sorry

end NUMINAMATH_CALUDE_find_b_l1354_135418


namespace NUMINAMATH_CALUDE_unicorn_rope_problem_l1354_135475

theorem unicorn_rope_problem (tower_radius : ℝ) (rope_length : ℝ) (rope_end_distance : ℝ)
  (a b c : ℕ) (h_radius : tower_radius = 10)
  (h_rope_length : rope_length = 25)
  (h_rope_end : rope_end_distance = 5)
  (h_c_prime : Nat.Prime c)
  (h_rope_touch : (a : ℝ) - Real.sqrt b = c * (rope_length - Real.sqrt ((tower_radius + rope_end_distance) ^ 2 + 5 ^ 2))) :
  a + b + c = 136 := by
sorry

end NUMINAMATH_CALUDE_unicorn_rope_problem_l1354_135475


namespace NUMINAMATH_CALUDE_peanut_butter_cost_l1354_135459

/-- The cost of the jar of peanut butter given the cost of bread, initial money, and money left over -/
theorem peanut_butter_cost
  (bread_cost : ℝ)
  (bread_quantity : ℕ)
  (initial_money : ℝ)
  (money_left : ℝ)
  (h1 : bread_cost = 2.25)
  (h2 : bread_quantity = 3)
  (h3 : initial_money = 14)
  (h4 : money_left = 5.25) :
  initial_money - money_left - bread_cost * bread_quantity = 2 :=
by sorry

end NUMINAMATH_CALUDE_peanut_butter_cost_l1354_135459


namespace NUMINAMATH_CALUDE_trig_expression_eval_l1354_135446

/-- Proves that the given trigonometric expression evaluates to -4√3 --/
theorem trig_expression_eval :
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) /
  (4 * (Real.cos (12 * π / 180))^2 * Real.sin (12 * π / 180) - 2 * Real.sin (12 * π / 180)) =
  -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_eval_l1354_135446


namespace NUMINAMATH_CALUDE_problem_statement_l1354_135499

theorem problem_statement (a b : ℚ) (h1 : a = 1/2) (h2 : b = 1/3) : 
  (a - b) / (1/78) = 13 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1354_135499


namespace NUMINAMATH_CALUDE_symmetric_rook_placements_8x8_l1354_135452

/-- Represents a chessboard configuration with rooks placed symmetrically --/
structure SymmetricRookPlacement where
  board_size : Nat
  num_rooks : Nat
  is_symmetric : Bool

/-- Counts the number of symmetric rook placements on a chessboard --/
def count_symmetric_rook_placements (config : SymmetricRookPlacement) : Nat :=
  sorry

/-- Theorem stating the number of symmetric rook placements for 8 rooks on an 8x8 chessboard --/
theorem symmetric_rook_placements_8x8 :
  count_symmetric_rook_placements ⟨8, 8, true⟩ = 139448 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_rook_placements_8x8_l1354_135452


namespace NUMINAMATH_CALUDE_monday_temperature_l1354_135480

theorem monday_temperature
  (avg_mon_to_thu : (mon + tue + wed + thu) / 4 = 48)
  (avg_tue_to_fri : (tue + wed + thu + 36) / 4 = 46)
  : mon = 44 := by
  sorry

end NUMINAMATH_CALUDE_monday_temperature_l1354_135480


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l1354_135485

-- Define the function f with domain [-1, 2]
def f : Set ℝ := Set.Icc (-1) 2

-- Define the function g(x) = f(2x-1)
def g (x : ℝ) : ℝ := 2 * x - 1

-- Theorem statement
theorem domain_of_composite_function :
  {x : ℝ | g x ∈ f} = Set.Icc 0 (3/2) := by sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l1354_135485


namespace NUMINAMATH_CALUDE_lily_hydrangea_plants_l1354_135490

/-- Prove that Lily buys 1 hydrangea plant per year -/
theorem lily_hydrangea_plants (start_year end_year : ℕ) (plant_cost total_spent : ℚ) : 
  start_year = 1989 →
  end_year = 2021 →
  plant_cost = 20 →
  total_spent = 640 →
  (total_spent / (end_year - start_year : ℚ)) / plant_cost = 1 := by
  sorry

end NUMINAMATH_CALUDE_lily_hydrangea_plants_l1354_135490


namespace NUMINAMATH_CALUDE_dust_retention_proof_l1354_135404

/-- The average dust retention of a ginkgo leaf in milligrams per year. -/
def ginkgo_retention : ℝ := 40

/-- The average dust retention of a locust leaf in milligrams per year. -/
def locust_retention : ℝ := 22

/-- The number of ginkgo leaves. -/
def num_ginkgo_leaves : ℕ := 50000

theorem dust_retention_proof :
  -- Condition 1: Ginkgo retention is 4mg less than twice locust retention
  ginkgo_retention = 2 * locust_retention - 4 ∧
  -- Condition 2: Total retention of ginkgo and locust is 62mg
  ginkgo_retention + locust_retention = 62 ∧
  -- Result 1: Ginkgo retention is 40mg
  ginkgo_retention = 40 ∧
  -- Result 2: Locust retention is 22mg
  locust_retention = 22 ∧
  -- Result 3: Total retention of 50,000 ginkgo leaves is 2kg
  (ginkgo_retention * num_ginkgo_leaves) / 1000000 = 2 :=
by sorry

end NUMINAMATH_CALUDE_dust_retention_proof_l1354_135404


namespace NUMINAMATH_CALUDE_find_b_value_l1354_135424

/-- The cube of a and the fourth root of b vary inversely -/
def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^(1/4) = k

theorem find_b_value (a b : ℝ) :
  inverse_relation a b →
  (3: ℝ)^3 * (256 : ℝ)^(1/4) = a^3 * b^(1/4) →
  a * b = 81 →
  b = 16 := by
sorry

end NUMINAMATH_CALUDE_find_b_value_l1354_135424


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1354_135402

theorem sum_with_radical_conjugate : 
  let x : ℝ := 15 - Real.sqrt 500
  let y : ℝ := 15 + Real.sqrt 500
  x + y = 30 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1354_135402


namespace NUMINAMATH_CALUDE_regular_polygon_interior_exterior_angle_relation_l1354_135419

theorem regular_polygon_interior_exterior_angle_relation (n : ℕ) :
  (n ≥ 3) →
  ((n - 2) * 180 : ℝ) = 2 * 360 →
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_exterior_angle_relation_l1354_135419


namespace NUMINAMATH_CALUDE_average_age_increase_l1354_135494

theorem average_age_increase (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 19 →
  student_avg_age = 20 →
  teacher_age = 40 →
  ((num_students : ℝ) * student_avg_age + teacher_age) / ((num_students : ℝ) + 1) - student_avg_age = 1 :=
by sorry

end NUMINAMATH_CALUDE_average_age_increase_l1354_135494


namespace NUMINAMATH_CALUDE_unique_divisor_of_18_l1354_135469

def divides (m n : ℕ) : Prop := ∃ k, n = m * k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem unique_divisor_of_18 : ∃! a : ℕ, 
  divides 3 a ∧ 
  divides a 18 ∧ 
  Even (sum_of_digits a) := by sorry

end NUMINAMATH_CALUDE_unique_divisor_of_18_l1354_135469


namespace NUMINAMATH_CALUDE_f_inequality_l1354_135441

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 1/x + 2 * Real.sin x

theorem f_inequality (x : ℝ) (hx : x > 0) :
  f (1 - x) > f x ↔ 0 < x ∧ x < 1/2 := by sorry

end NUMINAMATH_CALUDE_f_inequality_l1354_135441


namespace NUMINAMATH_CALUDE_hundred_with_five_twos_l1354_135462

theorem hundred_with_five_twos :
  ∃ (a b c d e : ℕ), a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2 ∧
  (a * b * c / d - e / d = 100) := by
  sorry

end NUMINAMATH_CALUDE_hundred_with_five_twos_l1354_135462


namespace NUMINAMATH_CALUDE_smallest_number_l1354_135481

theorem smallest_number (S : Set ℚ) (h : S = {-3, -1, 0, 1}) : 
  ∃ (m : ℚ), m ∈ S ∧ ∀ (x : ℚ), x ∈ S → m ≤ x ∧ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1354_135481


namespace NUMINAMATH_CALUDE_inequality_proof_l1354_135422

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum : x*y + y*z + z*x = 1) : 
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1354_135422


namespace NUMINAMATH_CALUDE_problem_solution_l1354_135443

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then x^2 - 4 else |x - 3| + a

theorem problem_solution (a : ℝ) :
  f a (f a (Real.sqrt 6)) = 3 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1354_135443


namespace NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l1354_135400

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_six_terms :
  let a : ℚ := 1/2
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 4095/6144 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l1354_135400


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1354_135407

theorem expression_simplification_and_evaluation (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4*a + 4) / (a + 1)) = (2 + a) / (2 - a) ∧
  (2 + 1) / (2 - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1354_135407


namespace NUMINAMATH_CALUDE_area_of_region_l1354_135410

-- Define the region
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 - 1 ≤ p.2 ∧ p.2 ≤ Real.sqrt (1 - p.1^2)}

-- State the theorem
theorem area_of_region :
  MeasureTheory.volume R = π / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l1354_135410


namespace NUMINAMATH_CALUDE_fraction_equality_l1354_135492

theorem fraction_equality (a b : ℚ) 
  (h1 : a = 1/2) 
  (h2 : b = 2/3) : 
  (6*a + 18*b) / (12*a + 6*b) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1354_135492


namespace NUMINAMATH_CALUDE_cookie_problem_l1354_135425

theorem cookie_problem (total_cookies : ℕ) (total_nuts : ℕ) (nuts_per_cookie : ℕ) 
  (h_total_cookies : total_cookies = 60)
  (h_total_nuts : total_nuts = 72)
  (h_nuts_per_cookie : nuts_per_cookie = 2)
  (h_quarter_nuts : (total_cookies / 4 : ℚ) = (total_cookies - (total_nuts / nuts_per_cookie) : ℕ)) :
  (((total_cookies - (total_cookies / 4) - (total_nuts / nuts_per_cookie - total_cookies / 4)) / total_cookies : ℚ) * 100 = 40) := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l1354_135425


namespace NUMINAMATH_CALUDE_heejin_drinks_most_l1354_135486

-- Define the drinking habits
def dongguk_frequency : ℕ := 5
def dongguk_amount : ℝ := 0.2

def yoonji_frequency : ℕ := 6
def yoonji_amount : ℝ := 0.3

def heejin_frequency : ℕ := 4
def heejin_amount : ℝ := 0.5  -- 500 ml = 0.5 L

-- Calculate total daily water intake for each person
def dongguk_total : ℝ := dongguk_frequency * dongguk_amount
def yoonji_total : ℝ := yoonji_frequency * yoonji_amount
def heejin_total : ℝ := heejin_frequency * heejin_amount

-- Theorem stating Heejin drinks the most water
theorem heejin_drinks_most : 
  heejin_total > dongguk_total ∧ heejin_total > yoonji_total :=
by sorry

end NUMINAMATH_CALUDE_heejin_drinks_most_l1354_135486


namespace NUMINAMATH_CALUDE_defective_product_probability_l1354_135403

theorem defective_product_probability 
  (total_products : ℕ) 
  (genuine_products : ℕ) 
  (defective_products : ℕ) 
  (h1 : total_products = genuine_products + defective_products)
  (h2 : genuine_products = 16)
  (h3 : defective_products = 4) :
  let prob_second_defective : ℚ := defective_products - 1 / (total_products - 1)
  prob_second_defective = 3 / 19 := by
sorry

end NUMINAMATH_CALUDE_defective_product_probability_l1354_135403


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1354_135471

/-- Given a quadratic equation ax^2 + 10x + c = 0 with exactly one solution,
    where a + c = 12 and a < c, prove that a = 6 - √11 and c = 6 + √11 -/
theorem unique_solution_quadratic (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) → 
  a + c = 12 → 
  a < c → 
  (a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1354_135471


namespace NUMINAMATH_CALUDE_product_remainder_l1354_135449

theorem product_remainder (a b c d e : ℕ) (h1 : a = 12457) (h2 : b = 12463) (h3 : c = 12469) (h4 : d = 12473) (h5 : e = 12479) :
  (a * b * c * d * e) % 18 = 3 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_l1354_135449


namespace NUMINAMATH_CALUDE_swimming_problem_l1354_135470

/-- Proves that Jamir swims 20 more meters per day than Sarah given the conditions of the swimming problem. -/
theorem swimming_problem (julien sarah jamir : ℕ) : 
  julien = 50 →  -- Julien swims 50 meters per day
  sarah = 2 * julien →  -- Sarah swims twice the distance Julien swims
  jamir > sarah →  -- Jamir swims some more meters per day than Sarah
  7 * (julien + sarah + jamir) = 1890 →  -- Combined distance for the week
  jamir - sarah = 20 := by  -- Jamir swims 20 more meters per day than Sarah
sorry

end NUMINAMATH_CALUDE_swimming_problem_l1354_135470


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1354_135401

def vector1 : Fin 3 → ℝ := ![1, 3, -2]
def vector2 : Fin 3 → ℝ := ![4, -2, 1]

theorem angle_between_vectors :
  let dot_product := (vector1 0) * (vector2 0) + (vector1 1) * (vector2 1) + (vector1 2) * (vector2 2)
  let magnitude1 := Real.sqrt ((vector1 0)^2 + (vector1 1)^2 + (vector1 2)^2)
  let magnitude2 := Real.sqrt ((vector2 0)^2 + (vector2 1)^2 + (vector2 2)^2)
  dot_product / (magnitude1 * magnitude2) = -2 / (7 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1354_135401


namespace NUMINAMATH_CALUDE_train_length_and_speed_l1354_135482

/-- Proves the length and speed of a train given its crossing times over two platforms. -/
theorem train_length_and_speed 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (platform1_length : ℝ) 
  (platform2_length : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (h1 : platform1_length = 90) 
  (h2 : platform2_length = 120) 
  (h3 : time1 = 12) 
  (h4 : time2 = 15) 
  (h5 : train_speed * time1 = train_length + platform1_length) 
  (h6 : train_speed * time2 = train_length + platform2_length) : 
  train_length = 30 ∧ train_speed = 10 := by
  sorry

#check train_length_and_speed

end NUMINAMATH_CALUDE_train_length_and_speed_l1354_135482


namespace NUMINAMATH_CALUDE_initial_girls_count_l1354_135408

theorem initial_girls_count (initial_total : ℕ) (initial_girls : ℕ) : 
  (initial_girls = initial_total / 2) →
  (initial_girls - 3) * 10 = 4 * (initial_total + 1) →
  (initial_girls - 4) * 20 = 7 * (initial_total + 2) →
  initial_girls = 17 := by
sorry

end NUMINAMATH_CALUDE_initial_girls_count_l1354_135408


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_one_l1354_135496

theorem arithmetic_expression_equals_one :
  2016 * 2014 - 2013 * 2015 + 2012 * 2015 - 2013 * 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_one_l1354_135496


namespace NUMINAMATH_CALUDE_roses_cut_l1354_135429

theorem roses_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) 
  (h1 : initial_roses = 13)
  (h2 : initial_orchids = 84)
  (h3 : final_roses = 14)
  (h4 : final_orchids = 91) :
  final_roses - initial_roses = 1 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l1354_135429


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l1354_135414

theorem quadratic_unique_solution (a b : ℝ) : 
  (∃! x, 16 * x^2 + a * x + b = 0) → 
  a^2 = 4 * b → 
  a = 0 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l1354_135414


namespace NUMINAMATH_CALUDE_seashells_given_theorem_l1354_135447

/-- The number of seashells Tim gave to Sara -/
def seashells_given_to_sara (initial_seashells final_seashells : ℕ) : ℕ :=
  initial_seashells - final_seashells

/-- Theorem stating that the number of seashells given to Sara is the difference between
    the initial and final counts of seashells Tim has -/
theorem seashells_given_theorem (initial_seashells final_seashells : ℕ) 
    (h : initial_seashells ≥ final_seashells) :
  seashells_given_to_sara initial_seashells final_seashells = initial_seashells - final_seashells :=
by
  sorry

#eval seashells_given_to_sara 679 507

end NUMINAMATH_CALUDE_seashells_given_theorem_l1354_135447


namespace NUMINAMATH_CALUDE_hyperbola_equation_for_given_parameters_l1354_135430

/-- Represents a hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ  -- Half of the real axis length
  e : ℝ  -- Eccentricity

/-- The equation of a hyperbola with given parameters -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / (h.a^2 * (h.e^2 - 1)) = 1

theorem hyperbola_equation_for_given_parameters :
  let h : Hyperbola := { a := 3, e := 5/3 }
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 / 9 - y^2 / 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_for_given_parameters_l1354_135430


namespace NUMINAMATH_CALUDE_inscribed_sphere_theorem_l1354_135465

/-- A truncated triangular pyramid with an inscribed sphere -/
structure TruncatedPyramid where
  /-- Height of the pyramid -/
  h : ℝ
  /-- Radius of the circle described around the first base -/
  R₁ : ℝ
  /-- Radius of the circle described around the second base -/
  R₂ : ℝ
  /-- Distance between the center of the first base circle and the point where the sphere touches it -/
  O₁T₁ : ℝ
  /-- Distance between the center of the second base circle and the point where the sphere touches it -/
  O₂T₂ : ℝ
  /-- All lengths are positive -/
  h_pos : 0 < h
  R₁_pos : 0 < R₁
  R₂_pos : 0 < R₂
  O₁T₁_pos : 0 < O₁T₁
  O₂T₂_pos : 0 < O₂T₂
  /-- The sphere touches the bases inside the circles -/
  O₁T₁_le_R₁ : O₁T₁ ≤ R₁
  O₂T₂_le_R₂ : O₂T₂ ≤ R₂

/-- The main theorem about the inscribed sphere in a truncated triangular pyramid -/
theorem inscribed_sphere_theorem (p : TruncatedPyramid) :
    p.R₁ * p.R₂ * p.h^2 = (p.R₁^2 - p.O₁T₁^2) * (p.R₂^2 - p.O₂T₂^2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_theorem_l1354_135465


namespace NUMINAMATH_CALUDE_min_a_for_inequality_solution_set_inequality_l1354_135448

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2) * abs (x - 2)

-- Theorem for part (1)
theorem min_a_for_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 1, f x ≤ a) ↔ a ≥ 4 := by sorry

-- Theorem for part (2)
theorem solution_set_inequality :
  {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4} ∪ {x : ℝ | -4 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_min_a_for_inequality_solution_set_inequality_l1354_135448


namespace NUMINAMATH_CALUDE_vector_at_negative_three_l1354_135438

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  -- The vector on the line at parameter t
  vector : ℝ → ℝ × ℝ

/-- The given parameterized line satisfying the problem conditions -/
def given_line : ParameterizedLine :=
  { vector := sorry }

theorem vector_at_negative_three :
  given_line.vector 1 = (4, 5) →
  given_line.vector 5 = (12, -11) →
  given_line.vector (-3) = (-4, 21) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_three_l1354_135438


namespace NUMINAMATH_CALUDE_marbles_remaining_l1354_135491

/-- Calculates the number of marbles remaining in a store after sales. -/
theorem marbles_remaining 
  (initial_marbles : ℕ) 
  (num_customers : ℕ) 
  (marbles_per_customer : ℕ) 
  (h1 : initial_marbles = 400)
  (h2 : num_customers = 20)
  (h3 : marbles_per_customer = 15) : 
  initial_marbles - num_customers * marbles_per_customer = 100 := by
sorry

end NUMINAMATH_CALUDE_marbles_remaining_l1354_135491


namespace NUMINAMATH_CALUDE_candy_bar_cost_l1354_135444

theorem candy_bar_cost (num_members : ℕ) (avg_sold_per_member : ℕ) (total_earnings : ℚ) :
  num_members = 20 →
  avg_sold_per_member = 8 →
  total_earnings = 80 →
  (total_earnings / (num_members * avg_sold_per_member : ℚ)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l1354_135444


namespace NUMINAMATH_CALUDE_population_growth_theorem_l1354_135421

/-- Calculates the population after a given number of years -/
def population_after_years (initial_population : ℕ) (birth_rate : ℚ) (death_rate : ℚ) (years : ℕ) : ℚ :=
  match years with
  | 0 => initial_population
  | n + 1 => 
    let prev_population := population_after_years initial_population birth_rate death_rate n
    prev_population + prev_population * birth_rate - prev_population * death_rate

/-- The population after 2 years is approximately 53045 -/
theorem population_growth_theorem : 
  let initial_population : ℕ := 50000
  let birth_rate : ℚ := 43 / 1000
  let death_rate : ℚ := 13 / 1000
  let years : ℕ := 2
  ⌊population_after_years initial_population birth_rate death_rate years⌋ = 53045 := by
  sorry


end NUMINAMATH_CALUDE_population_growth_theorem_l1354_135421


namespace NUMINAMATH_CALUDE_worker_a_completion_time_l1354_135411

/-- 
Given two workers a and b who can complete a work in 4 days together, 
and in 8/3 days when working simultaneously,
prove that worker a alone can complete the work in 8 days.
-/
theorem worker_a_completion_time 
  (total_time : ℝ) 
  (combined_time : ℝ) 
  (ha : total_time = 4) 
  (hb : combined_time = 8/3) : 
  ∃ (a_time : ℝ), a_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_worker_a_completion_time_l1354_135411


namespace NUMINAMATH_CALUDE_random_sampling_correct_l1354_135450

/-- Represents a random number table row -/
def RandomTableRow := List Nat

/-- Checks if a number is a valid bag number (000-799) -/
def isValidBagNumber (n : Nat) : Bool :=
  n >= 0 && n <= 799

/-- Extracts valid bag numbers from a list of numbers -/
def extractValidBagNumbers (numbers : List Nat) : List Nat :=
  numbers.filter isValidBagNumber

/-- Represents the given random number table row -/
def givenRow : RandomTableRow :=
  [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79]

/-- The expected result -/
def expectedResult : List Nat := [785, 567, 199, 507, 175]

theorem random_sampling_correct :
  let startIndex := 6  -- 7th column (0-based index)
  let relevantNumbers := givenRow.drop startIndex
  let validBagNumbers := extractValidBagNumbers relevantNumbers
  validBagNumbers.take 5 = expectedResult := by sorry

end NUMINAMATH_CALUDE_random_sampling_correct_l1354_135450


namespace NUMINAMATH_CALUDE_pencil_count_l1354_135423

theorem pencil_count (mitchell_pencils : ℕ) (difference : ℕ) : mitchell_pencils = 30 → difference = 6 →
  mitchell_pencils + (mitchell_pencils - difference) = 54 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l1354_135423


namespace NUMINAMATH_CALUDE_price_reduction_equation_l1354_135442

-- Define the original price
def original_price : ℝ := 200

-- Define the final price after reductions
def final_price : ℝ := 162

-- Define the average percentage reduction
variable (x : ℝ)

-- Theorem statement
theorem price_reduction_equation :
  original_price * (1 - x)^2 = final_price :=
sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l1354_135442


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l1354_135433

theorem product_of_sum_and_difference (x y : ℝ) : 
  x + y = 50 → x - y = 6 → x * y = 616 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l1354_135433


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1354_135420

theorem smallest_integer_with_remainders : ∃ N : ℕ, 
  N > 0 ∧
  N % 5 = 2 ∧
  N % 6 = 3 ∧
  N % 7 = 4 ∧
  N % 11 = 9 ∧
  (∀ M : ℕ, M > 0 ∧ M % 5 = 2 ∧ M % 6 = 3 ∧ M % 7 = 4 ∧ M % 11 = 9 → N ≤ M) ∧
  N = 207 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1354_135420


namespace NUMINAMATH_CALUDE_company_c_cheapest_l1354_135468

-- Define the pricing structures for each company
def company_a_cost (miles : ℝ) : ℝ :=
  2.10 + 0.40 * (miles * 5 - 1)

def company_b_cost (miles : ℝ) : ℝ :=
  3.00 + 0.50 * (miles * 4 - 1)

def company_c_cost (miles : ℝ) : ℝ :=
  1.50 * miles + 2.00

-- Theorem statement
theorem company_c_cheapest :
  let journey_length : ℝ := 8
  company_c_cost journey_length < company_a_cost journey_length ∧
  company_c_cost journey_length < company_b_cost journey_length :=
by sorry

end NUMINAMATH_CALUDE_company_c_cheapest_l1354_135468


namespace NUMINAMATH_CALUDE_linear_function_passes_through_points_l1354_135413

/-- A linear function y = kx - k passing through (-1, 4) also passes through (1, 0) -/
theorem linear_function_passes_through_points :
  ∃ k : ℝ, (k * (-1) - k = 4) ∧ (k * 1 - k = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_points_l1354_135413


namespace NUMINAMATH_CALUDE_amusement_park_total_cost_l1354_135437

/-- Represents the total cost for a group of children at an amusement park -/
def amusement_park_cost (num_children : ℕ) 
  (ferris_wheel_cost ferris_wheel_participants : ℕ)
  (roller_coaster_cost roller_coaster_participants : ℕ)
  (merry_go_round_cost : ℕ)
  (bumper_cars_cost bumper_cars_participants : ℕ)
  (haunted_house_cost haunted_house_participants : ℕ)
  (log_flume_cost log_flume_participants : ℕ)
  (ice_cream_cost ice_cream_participants : ℕ)
  (hot_dog_cost hot_dog_participants : ℕ)
  (pizza_cost pizza_participants : ℕ)
  (pretzel_cost pretzel_participants : ℕ)
  (cotton_candy_cost cotton_candy_participants : ℕ)
  (soda_cost soda_participants : ℕ) : ℕ :=
  ferris_wheel_cost * ferris_wheel_participants +
  roller_coaster_cost * roller_coaster_participants +
  merry_go_round_cost * num_children +
  bumper_cars_cost * bumper_cars_participants +
  haunted_house_cost * haunted_house_participants +
  log_flume_cost * log_flume_participants +
  ice_cream_cost * ice_cream_participants +
  hot_dog_cost * hot_dog_participants +
  pizza_cost * pizza_participants +
  pretzel_cost * pretzel_participants +
  cotton_candy_cost * cotton_candy_participants +
  soda_cost * soda_participants

/-- The total cost for the group of children at the amusement park is $286 -/
theorem amusement_park_total_cost : 
  amusement_park_cost 10 5 6 7 4 3 4 7 6 5 8 3 8 4 6 5 4 3 5 2 3 6 2 7 = 286 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_total_cost_l1354_135437


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l1354_135434

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^n + n
  let r : ℕ := 3^s - n^2
  r = 177138 := by sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l1354_135434


namespace NUMINAMATH_CALUDE_roof_length_width_difference_l1354_135427

-- Define the trapezoidal roof
structure TrapezoidalRoof where
  width : ℝ
  length : ℝ
  height : ℝ
  area : ℝ

-- Define the conditions of the problem
def roof_conditions (roof : TrapezoidalRoof) : Prop :=
  roof.length = 3 * roof.width ∧
  roof.height = 25 ∧
  roof.area = 675 ∧
  roof.area = (1 / 2) * (roof.width + roof.length) * roof.height

-- Theorem to prove
theorem roof_length_width_difference (roof : TrapezoidalRoof) 
  (h : roof_conditions roof) : roof.length - roof.width = 27 := by
  sorry


end NUMINAMATH_CALUDE_roof_length_width_difference_l1354_135427


namespace NUMINAMATH_CALUDE_distribution_count_7_l1354_135476

/-- The number of ways to distribute n distinct objects into 3 distinct containers
    labeled 1, 2, and 3, such that each container has at least as many objects as its label -/
def distribution_count (n : ℕ) : ℕ :=
  let ways_221 := (n.choose 2) * ((n - 2).choose 2)
  let ways_133 := (n.choose 1) * ((n - 1).choose 3)
  let ways_124 := (n.choose 1) * ((n - 1).choose 2)
  ways_221 + ways_133 + ways_124

/-- Theorem stating that there are 455 ways to distribute 7 distinct objects
    into 3 distinct containers with the given constraints -/
theorem distribution_count_7 : distribution_count 7 = 455 := by
  sorry

end NUMINAMATH_CALUDE_distribution_count_7_l1354_135476


namespace NUMINAMATH_CALUDE_oil_leaked_before_work_l1354_135484

def total_oil_leaked : ℕ := 11687
def oil_leaked_during_work : ℕ := 5165

theorem oil_leaked_before_work (total : ℕ) (during_work : ℕ) 
  (h1 : total = total_oil_leaked) 
  (h2 : during_work = oil_leaked_during_work) : 
  total - during_work = 6522 := by
  sorry

end NUMINAMATH_CALUDE_oil_leaked_before_work_l1354_135484


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l1354_135495

theorem min_value_fraction_sum (x y a b : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ a > 0 ∧ b > 0) 
  (h_sum : x + y = 1) : 
  (a / x + b / y) ≥ (Real.sqrt a + Real.sqrt b)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l1354_135495


namespace NUMINAMATH_CALUDE_acute_angles_are_in_first_quadrant_l1354_135461

/- Definition of acute angle -/
def is_acute_angle (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

/- Definition of angle in the first quadrant -/
def is_in_first_quadrant (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

/- Theorem stating that acute angles are angles in the first quadrant -/
theorem acute_angles_are_in_first_quadrant :
  ∀ θ : Real, is_acute_angle θ → is_in_first_quadrant θ := by
  sorry

#check acute_angles_are_in_first_quadrant

end NUMINAMATH_CALUDE_acute_angles_are_in_first_quadrant_l1354_135461


namespace NUMINAMATH_CALUDE_larger_number_of_sum_and_difference_l1354_135487

theorem larger_number_of_sum_and_difference (x y : ℝ) : 
  x + y = 40 → x - y = 4 → max x y = 22 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_sum_and_difference_l1354_135487


namespace NUMINAMATH_CALUDE_incorrect_statement_C_l1354_135483

theorem incorrect_statement_C : ¬ (∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_C_l1354_135483


namespace NUMINAMATH_CALUDE_sqrt_t6_plus_t4_l1354_135435

theorem sqrt_t6_plus_t4 (t : ℝ) : Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_t6_plus_t4_l1354_135435


namespace NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l1354_135473

theorem range_of_a_for_quadratic_inequality 
  (h : ∃ x ∈ Set.Icc 1 2, x^2 + a*x - 2 > 0) :
  a ∈ Set.Ioi (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l1354_135473


namespace NUMINAMATH_CALUDE_system_solution_l1354_135464

theorem system_solution :
  let x : ℚ := -7/3
  let y : ℚ := -1/9
  (4 * x - 3 * y = -9) ∧ (5 * x + 6 * y = -3) := by sorry

end NUMINAMATH_CALUDE_system_solution_l1354_135464


namespace NUMINAMATH_CALUDE_function_inequality_l1354_135451

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x, x * deriv f x ≥ 0) : 
  f (-1) + f 1 ≥ 2 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1354_135451


namespace NUMINAMATH_CALUDE_estimate_pi_l1354_135428

theorem estimate_pi (n : ℕ) (m : ℕ) (h1 : n = 120) (h2 : m = 34) :
  let π_estimate : ℚ := 4 * (m : ℚ) / (n : ℚ) + 2
  π_estimate = 47 / 15 := by
  sorry

end NUMINAMATH_CALUDE_estimate_pi_l1354_135428
