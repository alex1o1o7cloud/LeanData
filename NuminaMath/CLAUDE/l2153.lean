import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l2153_215306

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) : 
  x / (1 - x) + y / (1 - y) + z / (1 - z) ≥ 3 * (x * y * z)^(1/3) / (1 - (x * y * z)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2153_215306


namespace NUMINAMATH_CALUDE_john_light_bulbs_left_l2153_215364

/-- The number of light bulbs John has left after using some and giving away half of the remainder --/
def lightBulbsLeft (initial : ℕ) (used : ℕ) : ℕ :=
  let remaining := initial - used
  remaining - remaining / 2

/-- Theorem stating that John has 12 light bulbs left --/
theorem john_light_bulbs_left :
  lightBulbsLeft 40 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_john_light_bulbs_left_l2153_215364


namespace NUMINAMATH_CALUDE_intersection_M_N_l2153_215329

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2153_215329


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l2153_215374

/-- Given an arithmetic sequence where the first term is 4/7 and the seventeenth term is 5/6,
    the ninth term is equal to 59/84. -/
theorem ninth_term_of_arithmetic_sequence (a : ℕ → ℚ) 
  (h1 : a 1 = 4/7)
  (h17 : a 17 = 5/6)
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) :
  a 9 = 59/84 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l2153_215374


namespace NUMINAMATH_CALUDE_point_p_properties_l2153_215332

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

-- Define point Q
def Q : ℝ × ℝ := (4, 5)

theorem point_p_properties (a : ℝ) :
  -- Part 1
  (P a).2 = 0 → P a = (-12, 0) ∧
  -- Part 2
  (P a).1 = Q.1 → P a = (4, 8) ∧
  -- Part 3
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).1| = |(P a).2| → a^2022 + 2022 = 2023 :=
by sorry

end NUMINAMATH_CALUDE_point_p_properties_l2153_215332


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2153_215351

theorem arithmetic_sequence_sum : 
  ∀ (a d n : ℕ) (last : ℕ),
    a = 3 → d = 2 → last = 25 →
    last = a + (n - 1) * d →
    (n : ℝ) / 2 * (a + last) = 168 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2153_215351


namespace NUMINAMATH_CALUDE_ratio_evaluation_l2153_215340

theorem ratio_evaluation : (2^2023 * 3^2025) / 6^2024 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_evaluation_l2153_215340


namespace NUMINAMATH_CALUDE_gcd_of_specific_squares_l2153_215301

theorem gcd_of_specific_squares : Nat.gcd (130^2 + 240^2 + 350^2) (131^2 + 241^2 + 349^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_squares_l2153_215301


namespace NUMINAMATH_CALUDE_solve_candy_problem_l2153_215369

def candy_problem (megan_candy : ℕ) (mary_multiplier : ℕ) (mary_additional : ℕ) : Prop :=
  let mary_initial := mary_multiplier * megan_candy
  let mary_total := mary_initial + mary_additional
  megan_candy = 5 ∧ mary_multiplier = 3 ∧ mary_additional = 10 → mary_total = 25

theorem solve_candy_problem :
  candy_problem 5 3 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l2153_215369


namespace NUMINAMATH_CALUDE_equation_has_solution_in_interval_l2153_215395

theorem equation_has_solution_in_interval : 
  ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^3 = 2^x := by sorry

end NUMINAMATH_CALUDE_equation_has_solution_in_interval_l2153_215395


namespace NUMINAMATH_CALUDE_arcsin_sum_inequality_l2153_215388

theorem arcsin_sum_inequality (x y : ℝ) : 
  Real.arcsin x + Real.arcsin y > π / 2 ↔ 
  x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ x^2 + y^2 > 1 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sum_inequality_l2153_215388


namespace NUMINAMATH_CALUDE_square_mod_five_l2153_215354

theorem square_mod_five (n : ℤ) (h : n % 5 = 3) : (n^2) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_mod_five_l2153_215354


namespace NUMINAMATH_CALUDE_delegate_seating_probability_l2153_215383

/-- Represents the number of delegates -/
def num_delegates : ℕ := 8

/-- Represents the number of countries -/
def num_countries : ℕ := 4

/-- Represents the number of delegates per country -/
def delegates_per_country : ℕ := 2

/-- Represents the number of seats at the round table -/
def num_seats : ℕ := 8

/-- Calculates the total number of possible seating arrangements -/
def total_arrangements : ℕ := num_delegates.factorial / (delegates_per_country.factorial ^ num_countries)

/-- Calculates the number of favorable seating arrangements -/
def favorable_arrangements : ℕ := total_arrangements - 324

/-- The probability that each delegate sits next to at least one delegate from another country -/
def probability : ℚ := favorable_arrangements / total_arrangements

theorem delegate_seating_probability :
  probability = 131 / 140 := by sorry

end NUMINAMATH_CALUDE_delegate_seating_probability_l2153_215383


namespace NUMINAMATH_CALUDE_julie_leftover_money_l2153_215311

def bike_cost : ℕ := 2345
def initial_savings : ℕ := 1500
def lawns_to_mow : ℕ := 20
def lawn_pay : ℕ := 20
def newspapers_to_deliver : ℕ := 600
def newspaper_pay : ℚ := 40/100
def dogs_to_walk : ℕ := 24
def dog_walk_pay : ℕ := 15

theorem julie_leftover_money :
  let total_earnings := lawns_to_mow * lawn_pay + 
                        (newspapers_to_deliver : ℚ) * newspaper_pay + 
                        dogs_to_walk * dog_walk_pay
  let total_money := (initial_savings : ℚ) + total_earnings
  let leftover := total_money - bike_cost
  leftover = 155 := by sorry

end NUMINAMATH_CALUDE_julie_leftover_money_l2153_215311


namespace NUMINAMATH_CALUDE_part_one_part_two_l2153_215378

-- Define propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Part 1
theorem part_one (x : ℝ) (h1 : p x 1) (h2 : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Part 2
theorem part_two (a : ℝ) (h : a > 0)
  (h_suff : ∀ x, ¬(p x a) → ¬(q x))
  (h_not_nec : ∃ x, q x ∧ p x a) : 
  1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2153_215378


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2153_215355

theorem complex_division_simplification :
  let z : ℂ := (2 + 3*I) / I
  z = 3 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2153_215355


namespace NUMINAMATH_CALUDE_triangle_condition_l2153_215314

theorem triangle_condition (x y z : ℝ) : 
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 →
  (x + y > z ∧ x + z > y ∧ y + z > x) ↔ (x < 1 ∧ y < 1 ∧ z < 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_condition_l2153_215314


namespace NUMINAMATH_CALUDE_pharmacist_weights_exist_l2153_215366

theorem pharmacist_weights_exist : ∃ (w₁ w₂ w₃ : ℝ),
  w₁ < 90 ∧ w₂ < 90 ∧ w₃ < 90 ∧
  w₁ + w₂ + w₃ = 100 ∧
  w₁ + w₂ + (w₃ + 1) = 101 ∧
  w₂ + w₃ + (w₃ + 1) = 102 :=
by sorry

end NUMINAMATH_CALUDE_pharmacist_weights_exist_l2153_215366


namespace NUMINAMATH_CALUDE_hexalia_base_theorem_l2153_215331

/-- Converts a number from base s to base 10 -/
def toBase10 (digits : List Nat) (s : Nat) : Nat :=
  digits.foldr (fun d acc => d + s * acc) 0

/-- The base s used in Hexalia -/
def s : Nat :=
  sorry

/-- The cost of the computer in base s -/
def cost : List Nat :=
  [5, 3, 0]

/-- The amount paid in base s -/
def paid : List Nat :=
  [1, 2, 0, 0]

/-- The change received in base s -/
def change : List Nat :=
  [4, 5, 5]

/-- Theorem stating that the base s satisfies the transaction equation -/
theorem hexalia_base_theorem :
  toBase10 cost s + toBase10 change s = toBase10 paid s ∧ s = 10 :=
sorry

end NUMINAMATH_CALUDE_hexalia_base_theorem_l2153_215331


namespace NUMINAMATH_CALUDE_original_painting_width_l2153_215330

/-- Given a painting and its enlarged print, calculate the width of the original painting. -/
theorem original_painting_width
  (original_height : ℝ)
  (print_height : ℝ)
  (print_width : ℝ)
  (h1 : original_height = 10)
  (h2 : print_height = 25)
  (h3 : print_width = 37.5) :
  print_width / (print_height / original_height) = 15 :=
by sorry

#check original_painting_width

end NUMINAMATH_CALUDE_original_painting_width_l2153_215330


namespace NUMINAMATH_CALUDE_bobby_deadlift_increase_l2153_215316

/-- Represents Bobby's deadlift progression --/
structure DeadliftProgress where
  initial_weight : ℕ
  initial_age : ℕ
  final_age : ℕ
  percentage_increase : ℕ
  additional_weight : ℕ

/-- Calculates the average yearly increase in Bobby's deadlift --/
def average_yearly_increase (d : DeadliftProgress) : ℚ :=
  let final_weight := d.initial_weight * (d.percentage_increase : ℚ) / 100 + d.additional_weight
  let total_increase := final_weight - d.initial_weight
  let years := d.final_age - d.initial_age
  total_increase / years

/-- Theorem stating that Bobby's average yearly increase in deadlift is 110 pounds --/
theorem bobby_deadlift_increase :
  let bobby := DeadliftProgress.mk 300 13 18 250 100
  average_yearly_increase bobby = 110 := by
  sorry

end NUMINAMATH_CALUDE_bobby_deadlift_increase_l2153_215316


namespace NUMINAMATH_CALUDE_g_zero_at_three_l2153_215319

def g (x s : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + x^3 - 4 * x^2 + 5 * x + s

theorem g_zero_at_three (s : ℝ) : g 3 s = 0 ↔ s = -573 := by sorry

end NUMINAMATH_CALUDE_g_zero_at_three_l2153_215319


namespace NUMINAMATH_CALUDE_beka_flew_more_than_jackson_l2153_215303

/-- The difference in miles flown between Beka and Jackson -/
def miles_difference (beka_miles jackson_miles : ℕ) : ℕ :=
  beka_miles - jackson_miles

/-- Theorem stating that Beka flew 310 miles more than Jackson -/
theorem beka_flew_more_than_jackson :
  miles_difference 873 563 = 310 := by
  sorry

end NUMINAMATH_CALUDE_beka_flew_more_than_jackson_l2153_215303


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2153_215375

theorem smallest_integer_with_remainder_one : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 6 = 1) ∧ 
  (n % 7 = 1) ∧ 
  (n % 8 = 1) ∧ 
  (∀ m : ℕ, m > 1 → m % 6 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) ∧
  (n > 120) ∧ 
  (n < 209) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2153_215375


namespace NUMINAMATH_CALUDE_problem_solution_l2153_215309

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = -Real.sqrt (5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2153_215309


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l2153_215322

/-- Two vectors in ℝ² are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem collinear_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, m)
  collinear a b → m = -4 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l2153_215322


namespace NUMINAMATH_CALUDE_a_b_reciprocal_l2153_215350

theorem a_b_reciprocal (a b : ℚ) : 
  (-7/8) / (7/4 - 7/8 - 7/12) = a →
  (7/4 - 7/8 - 7/12) / (-7/8) = b →
  a = -1 / b :=
by
  sorry

end NUMINAMATH_CALUDE_a_b_reciprocal_l2153_215350


namespace NUMINAMATH_CALUDE_betty_savings_ratio_l2153_215346

theorem betty_savings_ratio (wallet_cost parents_gift grandparents_gift needed_more initial_savings : ℚ) :
  wallet_cost = 100 →
  parents_gift = 15 →
  grandparents_gift = 2 * parents_gift →
  needed_more = 5 →
  initial_savings + parents_gift + grandparents_gift = wallet_cost - needed_more →
  initial_savings / wallet_cost = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_betty_savings_ratio_l2153_215346


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l2153_215376

/-- Proves that mixing two varieties of rice in the ratio 1:2.4, where one costs 4.5 per kg 
    and the other costs 8.75 per kg, results in a mixture costing 7.50 per kg. -/
theorem rice_mixture_cost 
  (cost1 : ℝ) (cost2 : ℝ) (mixture_cost : ℝ) 
  (ratio1 : ℝ) (ratio2 : ℝ) :
  cost1 = 4.5 →
  cost2 = 8.75 →
  mixture_cost = 7.50 →
  ratio1 = 1 →
  ratio2 = 2.4 →
  (cost1 * ratio1 + cost2 * ratio2) / (ratio1 + ratio2) = mixture_cost :=
by sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_l2153_215376


namespace NUMINAMATH_CALUDE_F_is_second_from_left_l2153_215367

-- Define a structure for rectangles
structure Rectangle where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

-- Define the four rectangles
def F : Rectangle := ⟨7, 2, 5, 9⟩
def G : Rectangle := ⟨6, 9, 1, 3⟩
def H : Rectangle := ⟨2, 5, 7, 10⟩
def J : Rectangle := ⟨3, 1, 6, 8⟩

-- Define a function to check if two rectangles can connect
def canConnect (r1 r2 : Rectangle) : Prop :=
  (r1.a = r2.a) ∨ (r1.a = r2.b) ∨ (r1.a = r2.c) ∨ (r1.a = r2.d) ∨
  (r1.b = r2.a) ∨ (r1.b = r2.b) ∨ (r1.b = r2.c) ∨ (r1.b = r2.d) ∨
  (r1.c = r2.a) ∨ (r1.c = r2.b) ∨ (r1.c = r2.c) ∨ (r1.c = r2.d) ∨
  (r1.d = r2.a) ∨ (r1.d = r2.b) ∨ (r1.d = r2.c) ∨ (r1.d = r2.d)

-- Theorem stating that F is second from the left
theorem F_is_second_from_left :
  ∃ (left right : Rectangle), left ≠ F ∧ right ≠ F ∧
  canConnect left F ∧ canConnect F right ∧
  (∀ r : Rectangle, r ≠ F → r ≠ left → r ≠ right → ¬(canConnect left r ∧ canConnect r right)) :=
by
  sorry

end NUMINAMATH_CALUDE_F_is_second_from_left_l2153_215367


namespace NUMINAMATH_CALUDE_transistor_count_2010_l2153_215385

def initial_year : ℕ := 1985
def final_year : ℕ := 2010
def initial_transistors : ℕ := 500000
def doubling_period : ℕ := 2

def moores_law (t : ℕ) : ℕ := initial_transistors * 2^((t - initial_year) / doubling_period)

theorem transistor_count_2010 : moores_law final_year = 2048000000 := by
  sorry

end NUMINAMATH_CALUDE_transistor_count_2010_l2153_215385


namespace NUMINAMATH_CALUDE_germination_probability_l2153_215347

/-- The germination rate of the seeds -/
def germination_rate : ℝ := 0.7

/-- The number of seeds -/
def total_seeds : ℕ := 3

/-- The number of seeds we want to germinate -/
def target_germination : ℕ := 2

/-- The probability of exactly 2 out of 3 seeds germinating -/
def probability_2_out_of_3 : ℝ := 
  (Nat.choose total_seeds target_germination : ℝ) * 
  germination_rate ^ target_germination * 
  (1 - germination_rate) ^ (total_seeds - target_germination)

theorem germination_probability : 
  probability_2_out_of_3 = 0.441 := by sorry

end NUMINAMATH_CALUDE_germination_probability_l2153_215347


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2153_215318

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a ≥ 0 → ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ∧
  ¬(∃ x : ℝ, a * x^2 + x + 1 ≥ 0 → a ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2153_215318


namespace NUMINAMATH_CALUDE_taxi_charge_proof_l2153_215344

/-- Calculates the total charge for a taxi trip -/
def calculate_taxi_charge (initial_fee : ℚ) (charge_per_increment : ℚ) (increment_distance : ℚ) (trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / increment_distance).floor * charge_per_increment

/-- Proves that the total charge for a 3.6-mile trip is $4.95 -/
theorem taxi_charge_proof :
  let initial_fee : ℚ := 225/100
  let charge_per_increment : ℚ := 3/10
  let increment_distance : ℚ := 2/5
  let trip_distance : ℚ := 36/10
  calculate_taxi_charge initial_fee charge_per_increment increment_distance trip_distance = 495/100 := by
  sorry

#eval calculate_taxi_charge (225/100) (3/10) (2/5) (36/10)

end NUMINAMATH_CALUDE_taxi_charge_proof_l2153_215344


namespace NUMINAMATH_CALUDE_simplify_expression_l2153_215300

theorem simplify_expression : (1 / ((-5^4)^2)) * (-5)^9 = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2153_215300


namespace NUMINAMATH_CALUDE_odd_squares_sum_representation_l2153_215312

theorem odd_squares_sum_representation (k n : ℕ) (h : k ≠ n) :
  ((2 * k + 1)^2 + (2 * n + 1)^2) / 2 = (k + n + 1)^2 + (k - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_squares_sum_representation_l2153_215312


namespace NUMINAMATH_CALUDE_sqrt_of_nine_l2153_215337

theorem sqrt_of_nine : {x : ℝ | x ^ 2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_l2153_215337


namespace NUMINAMATH_CALUDE_max_lateral_surface_area_triangular_prism_l2153_215341

/-- Given a triangular prism with perimeter 12, prove that its maximum lateral surface area is 6 -/
theorem max_lateral_surface_area_triangular_prism :
  ∀ x y : ℝ, x > 0 → y > 0 → 6 * x + 3 * y = 12 →
  3 * x * y ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_lateral_surface_area_triangular_prism_l2153_215341


namespace NUMINAMATH_CALUDE_unique_square_sum_pair_l2153_215363

theorem unique_square_sum_pair : 
  ∃! (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    (∃ (m n : ℕ), 100 * a + b = m^2 ∧ 201 * a + b = n^2) ∧
    a = 17 ∧ b = 64 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_sum_pair_l2153_215363


namespace NUMINAMATH_CALUDE_max_points_top_four_l2153_215335

/-- Represents a tournament with 8 teams -/
structure Tournament :=
  (teams : Fin 8)
  (games : Fin 8 → Fin 8 → Nat)
  (points : Fin 8 → Nat)

/-- The scoring system for the tournament -/
def score (result : Nat) : Nat :=
  match result with
  | 0 => 3  -- win
  | 1 => 1  -- draw
  | _ => 0  -- loss

/-- The theorem stating the maximum possible points for the top four teams -/
theorem max_points_top_four (t : Tournament) : 
  ∃ (a b c d : Fin 8), 
    (∀ i : Fin 8, t.points i ≤ t.points a) ∧
    (t.points a = t.points b) ∧
    (t.points b = t.points c) ∧
    (t.points c = t.points d) ∧
    (t.points a ≤ 33) :=
sorry

end NUMINAMATH_CALUDE_max_points_top_four_l2153_215335


namespace NUMINAMATH_CALUDE_triangle_abc_degenerate_l2153_215324

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parabola defined by y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- A horizontal line defined by y = 2 -/
def HorizontalLine := {p : Point | p.y = 2}

/-- Theorem: The intersection of the horizontal line y = 2 and the parabola y^2 = 4x 
    results in a point that coincides with A(1,2), making triangle ABC degenerate -/
theorem triangle_abc_degenerate (A : Point) (h1 : A.x = 1) (h2 : A.y = 2) :
  ∃ (B : Point), B ∈ Parabola ∧ B ∈ HorizontalLine ∧ B = A :=
sorry

end NUMINAMATH_CALUDE_triangle_abc_degenerate_l2153_215324


namespace NUMINAMATH_CALUDE_constant_remainder_iff_a_eq_neg_seven_l2153_215382

/-- The dividend polynomial -/
def dividend (a : ℝ) (x : ℝ) : ℝ := 10 * x^3 - 7 * x^2 + a * x + 6

/-- The divisor polynomial -/
def divisor (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

/-- The remainder of the polynomial division -/
def remainder (a : ℝ) (x : ℝ) : ℝ := (a + 7) * x + 2

theorem constant_remainder_iff_a_eq_neg_seven :
  ∀ a : ℝ, (∀ x : ℝ, ∃ q : ℝ, dividend a x = divisor x * q + remainder a x) ↔ a = -7 := by
  sorry

end NUMINAMATH_CALUDE_constant_remainder_iff_a_eq_neg_seven_l2153_215382


namespace NUMINAMATH_CALUDE_bisection_next_point_l2153_215305

theorem bisection_next_point 
  (f : ℝ → ℝ) 
  (h_continuous : ContinuousOn f (Set.Icc 1 2))
  (h_f1 : f 1 < 0)
  (h_f1_5 : f 1.5 > 0) :
  (1 + 1.5) / 2 = 1.25 := by sorry

end NUMINAMATH_CALUDE_bisection_next_point_l2153_215305


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2153_215390

theorem cube_root_equation_solution (Q P : ℝ) 
  (h1 : (13 * Q + 6 * P + 1) ^ (1/3) - (13 * Q - 6 * P - 1) ^ (1/3) = 2 ^ (1/3))
  (h2 : Q > 0) : 
  Q = 7 := by
sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2153_215390


namespace NUMINAMATH_CALUDE_dennis_initial_money_dennis_initial_money_proof_l2153_215320

/-- Proves that Dennis's initial amount of money equals $50, given the conditions of his purchase and change received. -/
theorem dennis_initial_money : ℕ → Prop :=
  fun initial : ℕ =>
    let shirt_cost : ℕ := 27
    let change_bills : ℕ := 2 * 10
    let change_coins : ℕ := 3
    let total_change : ℕ := change_bills + change_coins
    initial = shirt_cost + total_change ∧ initial = 50

/-- The theorem holds for the specific case where Dennis's initial money is 50. -/
theorem dennis_initial_money_proof : dennis_initial_money 50 := by
  sorry

#check dennis_initial_money
#check dennis_initial_money_proof

end NUMINAMATH_CALUDE_dennis_initial_money_dennis_initial_money_proof_l2153_215320


namespace NUMINAMATH_CALUDE_linda_college_applications_l2153_215365

def number_of_colleges (hourly_rate : ℚ) (application_fee : ℚ) (hours_worked : ℚ) : ℚ :=
  (hourly_rate * hours_worked) / application_fee

theorem linda_college_applications : number_of_colleges 10 25 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_linda_college_applications_l2153_215365


namespace NUMINAMATH_CALUDE_base8_subtraction_l2153_215304

-- Define a function to convert base 8 numbers to natural numbers
def base8ToNat (x : ℕ) : ℕ := sorry

-- Define a function to convert natural numbers to base 8
def natToBase8 (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem base8_subtraction :
  natToBase8 (base8ToNat 546 - base8ToNat 321 - base8ToNat 105) = 120 := by sorry

end NUMINAMATH_CALUDE_base8_subtraction_l2153_215304


namespace NUMINAMATH_CALUDE_square_of_thirteen_x_l2153_215356

theorem square_of_thirteen_x (x : ℝ) : (13 * x)^2 = 169 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_thirteen_x_l2153_215356


namespace NUMINAMATH_CALUDE_min_speed_to_arrive_first_l2153_215371

/-- Proves the minimum speed required for Person B to arrive before Person A --/
theorem min_speed_to_arrive_first (distance : ℝ) (speed_a : ℝ) (delay : ℝ) 
  (h1 : distance = 220)
  (h2 : speed_a = 40)
  (h3 : delay = 0.5)
  (h4 : speed_a > 0) :
  ∃ (min_speed : ℝ), 
    (∀ (speed_b : ℝ), speed_b > min_speed → 
      distance / speed_b + delay < distance / speed_a) ∧
    min_speed = 44 := by
  sorry

end NUMINAMATH_CALUDE_min_speed_to_arrive_first_l2153_215371


namespace NUMINAMATH_CALUDE_range_of_t_l2153_215336

theorem range_of_t (x y a : ℝ) 
  (eq1 : x + 3 * y + a = 4)
  (eq2 : x - y - 3 * a = 0)
  (bounds : -1 ≤ a ∧ a ≤ 1) :
  let t := x + y
  1 ≤ t ∧ t ≤ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_t_l2153_215336


namespace NUMINAMATH_CALUDE_range_upper_bound_l2153_215393

theorem range_upper_bound (n : ℕ) : 
  (n ≥ 1) → 
  ((n - 1 : ℝ) / (2 * n)) = 0.4995 → 
  n = 1000 := by
sorry

end NUMINAMATH_CALUDE_range_upper_bound_l2153_215393


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l2153_215397

/-- Ellipse C: x²/9 + y²/8 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/9 + y^2/8 = 1

/-- Line l: x = my + 1 -/
def line_l (m x y : ℝ) : Prop := x = m*y + 1

/-- Point on ellipse C -/
structure PointOnC where
  x : ℝ
  y : ℝ
  on_C : ellipse_C x y

/-- Foci and vertices of ellipse C -/
structure EllipseCPoints where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Line l intersects ellipse C at points M and N -/
structure Intersection where
  m : ℝ
  M : PointOnC
  N : PointOnC
  M_on_l : line_l m M.x M.y
  N_on_l : line_l m N.x N.y
  y_conditions : M.y > 0 ∧ N.y < 0

/-- MA is perpendicular to NF₁ -/
def perpendicular (A M N F₁ : ℝ × ℝ) : Prop :=
  (M.2 - A.2) * (N.2 - F₁.2) = -(M.1 - A.1) * (N.1 - F₁.1)

/-- Theorem statement -/
theorem ellipse_intersection_theorem 
  (C : EllipseCPoints) 
  (I : Intersection) 
  (h_perp : perpendicular C.A (I.M.x, I.M.y) (I.N.x, I.N.y) C.F₁) :
  I.m = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l2153_215397


namespace NUMINAMATH_CALUDE_game_gameplay_hours_l2153_215308

theorem game_gameplay_hours (T : ℝ) (h1 : 0.2 * T + 30 = 50) : T = 100 := by
  sorry

end NUMINAMATH_CALUDE_game_gameplay_hours_l2153_215308


namespace NUMINAMATH_CALUDE_bells_lcm_l2153_215321

/-- The time interval (in minutes) between consecutive rings of the library bell -/
def library_interval : ℕ := 18

/-- The time interval (in minutes) between consecutive rings of the community center bell -/
def community_interval : ℕ := 24

/-- The time interval (in minutes) between consecutive rings of the restaurant bell -/
def restaurant_interval : ℕ := 30

/-- The theorem states that the least common multiple of the three bell intervals is 360 minutes -/
theorem bells_lcm :
  lcm (lcm library_interval community_interval) restaurant_interval = 360 :=
by sorry

end NUMINAMATH_CALUDE_bells_lcm_l2153_215321


namespace NUMINAMATH_CALUDE_business_profit_l2153_215342

def total_subscription : ℕ := 50000
def a_more_than_b : ℕ := 4000
def b_more_than_c : ℕ := 5000
def a_profit : ℕ := 29400

theorem business_profit :
  ∃ (c_subscription : ℕ),
    let b_subscription := c_subscription + b_more_than_c
    let a_subscription := b_subscription + a_more_than_b
    a_subscription + b_subscription + c_subscription = total_subscription →
    (a_profit * total_subscription) / a_subscription = 70000 :=
sorry

end NUMINAMATH_CALUDE_business_profit_l2153_215342


namespace NUMINAMATH_CALUDE_right_triangle_pythagorean_representation_l2153_215384

theorem right_triangle_pythagorean_representation
  (a b c : ℕ)
  (d : ℤ)
  (h_order : a < b ∧ b < c)
  (h_gcd : Nat.gcd (c - a) (c - b) = 1)
  (h_right_triangle : (a + d)^2 + (b + d)^2 = (c + d)^2) :
  ∃ l m : ℤ, (c : ℤ) + d = l^2 + m^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_pythagorean_representation_l2153_215384


namespace NUMINAMATH_CALUDE_quadratic_polynomials_sum_nonnegative_l2153_215317

theorem quadratic_polynomials_sum_nonnegative 
  (b c p q m₁ m₂ k₁ k₂ : ℝ) 
  (hf : ∀ x, x^2 + b*x + c = (x - m₁) * (x - m₂))
  (hg : ∀ x, x^2 + p*x + q = (x - k₁) * (x - k₂)) :
  (k₁^2 + b*k₁ + c) + (k₂^2 + b*k₂ + c) + 
  (m₁^2 + p*m₁ + q) + (m₂^2 + p*m₂ + q) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomials_sum_nonnegative_l2153_215317


namespace NUMINAMATH_CALUDE_absolute_value_quadratic_inequality_l2153_215389

theorem absolute_value_quadratic_inequality (x : ℝ) :
  |3 * x^2 - 5 * x - 2| < 5 ↔ x > -1/3 ∧ x < 1/3 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_quadratic_inequality_l2153_215389


namespace NUMINAMATH_CALUDE_pi_is_irrational_one_third_is_rational_sqrt_16_is_rational_finite_decimal_is_rational_pi_only_irrational_option_l2153_215345

theorem pi_is_irrational :
  ∀ (a b : ℚ), (a : ℝ) ≠ π ∧ (b : ℝ) ≠ π → Irrational π := by
  sorry

theorem one_third_is_rational : ∃ (a b : ℤ), b ≠ 0 ∧ (1 : ℝ) / 3 = (a : ℝ) / (b : ℝ) := by
  sorry

theorem sqrt_16_is_rational : ∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 16 = (a : ℝ) / (b : ℝ) := by
  sorry

theorem finite_decimal_is_rational : ∃ (a b : ℤ), b ≠ 0 ∧ 3.1415926 = (a : ℝ) / (b : ℝ) := by
  sorry

theorem pi_only_irrational_option : Irrational π := by
  sorry

end NUMINAMATH_CALUDE_pi_is_irrational_one_third_is_rational_sqrt_16_is_rational_finite_decimal_is_rational_pi_only_irrational_option_l2153_215345


namespace NUMINAMATH_CALUDE_students_neither_music_nor_art_l2153_215359

theorem students_neither_music_nor_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (both : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 50) 
  (h3 : art = 20) 
  (h4 : both = 10) : 
  total - (music + art - both) = 440 := by
  sorry

end NUMINAMATH_CALUDE_students_neither_music_nor_art_l2153_215359


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l2153_215302

theorem product_of_sums_equals_difference_of_powers : 
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l2153_215302


namespace NUMINAMATH_CALUDE_prob_receive_one_out_of_two_prob_receive_at_least_ten_l2153_215373

/-- The probability of receiving a red envelope for each recipient -/
def prob_receive : ℚ := 1 / 3

/-- The probability of not receiving a red envelope for each recipient -/
def prob_not_receive : ℚ := 2 / 3

/-- The number of recipients -/
def num_recipients : ℕ := 3

/-- The number of red envelopes sent in the first scenario -/
def num_envelopes_1 : ℕ := 2

/-- The number of red envelopes sent in the second scenario -/
def num_envelopes_2 : ℕ := 3

/-- The amounts in the red envelopes for the second scenario -/
def envelope_amounts : List ℚ := [5, 5, 10]

/-- Theorem 1: Probability of receiving exactly one envelope out of two -/
theorem prob_receive_one_out_of_two :
  let p := prob_receive
  let q := prob_not_receive
  p * q + q * p = 4 / 9 := by sorry

/-- Theorem 2: Probability of receiving at least 10 yuan out of three envelopes -/
theorem prob_receive_at_least_ten :
  let p := prob_receive
  let q := prob_not_receive
  p^2 * q + 2 * p^2 * q + p^3 = 11 / 27 := by sorry

end NUMINAMATH_CALUDE_prob_receive_one_out_of_two_prob_receive_at_least_ten_l2153_215373


namespace NUMINAMATH_CALUDE_inequality_proof_l2153_215327

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a + 1/b)^2 + (b + 1/c)^2 + (c + 1/a)^2 ≥ 3 * (a + b + c + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2153_215327


namespace NUMINAMATH_CALUDE_nancy_bottle_caps_l2153_215315

theorem nancy_bottle_caps (initial final found : ℕ) : 
  initial = 91 → final = 179 → found = final - initial :=
by sorry

end NUMINAMATH_CALUDE_nancy_bottle_caps_l2153_215315


namespace NUMINAMATH_CALUDE_andras_bela_numbers_l2153_215339

theorem andras_bela_numbers :
  ∀ (a b : ℕ+),
  (a = b + 1992 ∨ b = a + 1992) →
  a > 1992 →
  b > 3984 →
  a ≤ 5976 →
  (a + 1 > 5976) →
  (a = 5976 ∧ b = 7968) :=
by sorry

end NUMINAMATH_CALUDE_andras_bela_numbers_l2153_215339


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2153_215394

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2153_215394


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l2153_215326

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 20)
  (h2 : c + a = 22)
  (h3 : a + b = 24) :
  Real.sqrt (a * b * c * (a + b + c)) = 206.1 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l2153_215326


namespace NUMINAMATH_CALUDE_square_area_perimeter_ratio_l2153_215353

theorem square_area_perimeter_ratio : 
  ∀ (a b : ℝ), a > 0 → b > 0 → (a^2 / b^2 = 49 / 64) → (4*a / (4*b) = 7 / 8) := by
  sorry

end NUMINAMATH_CALUDE_square_area_perimeter_ratio_l2153_215353


namespace NUMINAMATH_CALUDE_last_three_digits_of_power_l2153_215391

theorem last_three_digits_of_power (N : ℕ) : 
  N = 2002^2001 → 2003^N ≡ 241 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_power_l2153_215391


namespace NUMINAMATH_CALUDE_max_triangle_side_length_l2153_215313

theorem max_triangle_side_length (a b c : ℕ) : 
  a < b ∧ b < c ∧                -- Three different side lengths
  a + b + c = 24 ∧               -- Perimeter is 24
  a + b > c ∧ a + c > b ∧ b + c > a →  -- Triangle inequality
  c ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_side_length_l2153_215313


namespace NUMINAMATH_CALUDE_quadratic_symmetry_axis_l2153_215357

theorem quadratic_symmetry_axis 
  (a b c : ℝ) 
  (ha : a ≠ 0)
  (h1 : a * (0 + 4)^2 + b * (0 + 4) + c = 0)
  (h2 : a * (0 - 1)^2 + b * (0 - 1) + c = 0) :
  -b / (2 * a) = 1.5 := by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_axis_l2153_215357


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2153_215349

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_properties (a₁ : ℝ) (d : ℝ) (h : d > 0) :
  (∀ n m : ℕ, n < m → arithmetic_sequence a₁ d n < arithmetic_sequence a₁ d m) ∧
  (∀ n m : ℕ, n < m → arithmetic_sequence a₁ d n + 3 * n * d < arithmetic_sequence a₁ d m + 3 * m * d) ∧
  (∃ a₁ d : ℝ, d > 0 ∧ ∃ n m : ℕ, n < m ∧ n * (arithmetic_sequence a₁ d n) ≥ m * (arithmetic_sequence a₁ d m)) ∧
  (∃ a₁ d : ℝ, d > 0 ∧ ∃ n m : ℕ, n < m ∧ arithmetic_sequence a₁ d n / n ≤ arithmetic_sequence a₁ d m / m) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2153_215349


namespace NUMINAMATH_CALUDE_container_water_percentage_l2153_215333

theorem container_water_percentage (capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) 
  (h1 : capacity = 120)
  (h2 : added_water = 54)
  (h3 : final_fraction = 3/4) :
  let initial_percentage := (final_fraction * capacity - added_water) / capacity * 100
  initial_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_container_water_percentage_l2153_215333


namespace NUMINAMATH_CALUDE_range_of_a_for_R_solution_set_l2153_215368

-- Define the quadratic function
def f (a x : ℝ) : ℝ := (a - 2) * x^2 + 4 * (a - 2) * x - 4

-- Define the property that the solution set is ℝ
def solution_set_is_R (a : ℝ) : Prop := ∀ x, f a x < 0

-- Theorem statement
theorem range_of_a_for_R_solution_set :
  {a : ℝ | solution_set_is_R a} = Set.Ioo 1 2 ∪ {2} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_R_solution_set_l2153_215368


namespace NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l2153_215310

theorem vector_subtraction_scalar_multiplication :
  (⟨3, -7⟩ : ℝ × ℝ) - 3 • (⟨2, -4⟩ : ℝ × ℝ) = (⟨-3, 5⟩ : ℝ × ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l2153_215310


namespace NUMINAMATH_CALUDE_tan_plus_four_sin_twenty_degrees_l2153_215379

theorem tan_plus_four_sin_twenty_degrees :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_four_sin_twenty_degrees_l2153_215379


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2153_215348

theorem unique_solution_for_equation : ∃! (m n : ℕ), m^m + (m*n)^n = 1984 ∧ m = 4 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2153_215348


namespace NUMINAMATH_CALUDE_velvet_for_hats_and_cloaks_l2153_215323

/-- The amount of velvet needed for hats and cloaks -/
def velvet_needed (hats_per_yard : ℚ) (yards_per_cloak : ℚ) (num_hats : ℚ) (num_cloaks : ℚ) : ℚ :=
  (num_hats / hats_per_yard) + (num_cloaks * yards_per_cloak)

/-- Theorem stating the total amount of velvet needed for 6 cloaks and 12 hats -/
theorem velvet_for_hats_and_cloaks :
  velvet_needed 4 3 12 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_velvet_for_hats_and_cloaks_l2153_215323


namespace NUMINAMATH_CALUDE_union_equals_B_l2153_215380

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x - a ≤ 0}

-- State the theorem
theorem union_equals_B (a : ℝ) : A ∪ B a = B a → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_B_l2153_215380


namespace NUMINAMATH_CALUDE_scientific_notation_120_million_l2153_215398

theorem scientific_notation_120_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 120000000 = a * (10 : ℝ) ^ n ∧ a = 1.2 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_120_million_l2153_215398


namespace NUMINAMATH_CALUDE_alpha_one_sufficient_not_necessary_l2153_215381

-- Define sets A and B
def A : Set ℝ := {x | 2 < x ∧ x < 3}
def B (α : ℝ) : Set ℝ := {x | (x + 2) * (x - α) < 0}

-- Statement to prove
theorem alpha_one_sufficient_not_necessary :
  (∀ x, x ∈ A ∩ B 1 → False) ∧
  (∃ α, α ≠ 1 ∧ ∀ x, x ∈ A ∩ B α → False) := by sorry

end NUMINAMATH_CALUDE_alpha_one_sufficient_not_necessary_l2153_215381


namespace NUMINAMATH_CALUDE_smallest_k_for_fifteen_digit_period_l2153_215387

/-- Represents a positive rational number with a decimal representation having a minimal period of 30 digits -/
def RationalWith30DigitPeriod : Type := { q : ℚ // q > 0 ∧ ∃ m : ℕ+, q = m / (10^30 - 1) }

/-- Given two positive rational numbers with 30-digit periods, returns true if their difference has a 15-digit period -/
def hasFifteenDigitPeriodDiff (a b : RationalWith30DigitPeriod) : Prop :=
  ∃ p : ℤ, (a.val - b.val : ℚ) = p / (10^15 - 1)

/-- Given two positive rational numbers with 30-digit periods and a natural number k,
    returns true if their sum with k times the second number has a 15-digit period -/
def hasFifteenDigitPeriodSum (a b : RationalWith30DigitPeriod) (k : ℕ) : Prop :=
  ∃ q : ℤ, (a.val + k * b.val : ℚ) = q / (10^15 - 1)

theorem smallest_k_for_fifteen_digit_period (a b : RationalWith30DigitPeriod)
  (h : hasFifteenDigitPeriodDiff a b) :
  (∀ k < 6, ¬hasFifteenDigitPeriodSum a b k) ∧ hasFifteenDigitPeriodSum a b 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_fifteen_digit_period_l2153_215387


namespace NUMINAMATH_CALUDE_pool_length_l2153_215399

theorem pool_length (width : ℝ) (depth : ℝ) (capacity : ℝ) (drain_rate : ℝ) (drain_time : ℝ) :
  width = 50 →
  depth = 10 →
  capacity = 0.8 →
  drain_rate = 60 →
  drain_time = 1000 →
  ∃ (length : ℝ), length = 150 ∧ capacity * width * length * depth = drain_rate * drain_time :=
by sorry

end NUMINAMATH_CALUDE_pool_length_l2153_215399


namespace NUMINAMATH_CALUDE_train_speed_l2153_215343

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 250) (h2 : crossing_time = 4) :
  train_length / crossing_time = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2153_215343


namespace NUMINAMATH_CALUDE_base_eight_sum_l2153_215396

theorem base_eight_sum (A B C : ℕ) : 
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  A < 8 ∧ B < 8 ∧ C < 8 ∧
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (A * 8^2 + B * 8 + C) + (B * 8^2 + C * 8 + A) + (C * 8^2 + A * 8 + B) = A * (8^3 + 8^2 + 8) →
  B + C = 7 :=
by sorry

end NUMINAMATH_CALUDE_base_eight_sum_l2153_215396


namespace NUMINAMATH_CALUDE_group_size_calculation_l2153_215360

theorem group_size_calculation (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 6.2 →
  old_weight = 76 →
  new_weight = 119.4 →
  (new_weight - old_weight) / average_increase = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l2153_215360


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2153_215325

theorem diophantine_equation_solutions :
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≤ 1980 ∧ 4 * p.1^3 - 3 * p.1 + 1 = 2 * p.2^2) ∧
    S.card ≥ 31 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2153_215325


namespace NUMINAMATH_CALUDE_farmer_land_calculation_l2153_215307

theorem farmer_land_calculation (total_land : ℝ) : 
  (0.9 * total_land * 0.2 + 0.9 * total_land * 0.7 + 630 = 0.9 * total_land) →
  total_land = 7000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_calculation_l2153_215307


namespace NUMINAMATH_CALUDE_two_valid_solutions_l2153_215328

def original_number : ℕ := 20192020

def is_valid (a b : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ (a * 1000000000 + original_number * 10 + b) % 72 = 0

theorem two_valid_solutions :
  ∃! (s : Set (ℕ × ℕ)), s = {(2, 0), (3, 8)} ∧ 
    ∀ (a b : ℕ), (a, b) ∈ s ↔ is_valid a b :=
sorry

end NUMINAMATH_CALUDE_two_valid_solutions_l2153_215328


namespace NUMINAMATH_CALUDE_tan_theta_value_l2153_215377

theorem tan_theta_value (θ : Real) 
  (h1 : 2 * Real.sin θ + Real.cos θ = Real.sqrt 2 / 3)
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.tan θ = -(90 + 5 * Real.sqrt 86) / 168 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l2153_215377


namespace NUMINAMATH_CALUDE_sum_difference_even_odd_l2153_215358

/-- Sum of the first n positive even integers -/
def sumFirstEvenIntegers (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- Sum of the first n positive odd integers -/
def sumFirstOddIntegers (n : ℕ) : ℕ := n * n

/-- The positive difference between the sum of the first 25 positive even integers
    and the sum of the first 20 positive odd integers is 250 -/
theorem sum_difference_even_odd : 
  (sumFirstEvenIntegers 25 : ℤ) - (sumFirstOddIntegers 20 : ℤ) = 250 := by sorry

end NUMINAMATH_CALUDE_sum_difference_even_odd_l2153_215358


namespace NUMINAMATH_CALUDE_square_ending_same_nonzero_digits_l2153_215386

theorem square_ending_same_nonzero_digits (n : ℕ) :
  (∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧ n^2 % 100 = d * 10 + d) →
  n^2 % 100 = 44 := by
sorry

end NUMINAMATH_CALUDE_square_ending_same_nonzero_digits_l2153_215386


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2153_215338

theorem circle_center_radius_sum (x y : ℝ) : 
  x^2 - 16*x + y^2 - 18*y = -81 → 
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧ a + b + r = 17 + Real.sqrt 145 := by
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2153_215338


namespace NUMINAMATH_CALUDE_shirt_pricing_l2153_215372

theorem shirt_pricing (total_shirts : Nat) (price_shirt1 price_shirt2 : ℝ) (min_avg_price_remaining : ℝ) :
  total_shirts = 5 →
  price_shirt1 = 30 →
  price_shirt2 = 20 →
  min_avg_price_remaining = 33.333333333333336 →
  (price_shirt1 + price_shirt2 + (total_shirts - 2) * min_avg_price_remaining) / total_shirts = 30 := by
  sorry

end NUMINAMATH_CALUDE_shirt_pricing_l2153_215372


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2153_215392

/-- Given a geometric sequence of positive integers with first term 3 and sixth term 729,
    prove that the seventh term is 2187. -/
theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 3 →                             -- First term is 3
  a 6 = 729 →                           -- Sixth term is 729
  (∀ n, a n > 0) →                      -- All terms are positive
  a 7 = 2187 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2153_215392


namespace NUMINAMATH_CALUDE_janice_spent_1940_l2153_215362

/-- Calculates the total amount spent by Janice given the prices and quantities of items purchased --/
def total_spent (juice_price : ℚ) (sandwich_price : ℚ) (pastry_price : ℚ) (salad_price : ℚ) : ℚ :=
  let discounted_salad_price := salad_price * (1 - 0.2)
  sandwich_price + juice_price + 2 * pastry_price + discounted_salad_price

/-- Theorem stating that Janice spent $19.40 given the conditions in the problem --/
theorem janice_spent_1940 :
  let juice_price : ℚ := 10 / 5
  let sandwich_price : ℚ := 6 / 2
  let pastry_price : ℚ := 4
  let salad_price : ℚ := 8
  total_spent juice_price sandwich_price pastry_price salad_price = 1940 / 100 := by
  sorry

#eval total_spent (10/5) (6/2) 4 8

end NUMINAMATH_CALUDE_janice_spent_1940_l2153_215362


namespace NUMINAMATH_CALUDE_hotel_room_charges_l2153_215370

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R * (1 - 0.25))
  (h2 : P = G * (1 - 0.10)) :
  R = G * 1.20 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l2153_215370


namespace NUMINAMATH_CALUDE_identity_function_divisibility_l2153_215352

def is_divisible (a b : ℕ) : Prop := ∃ k, b = a * k

theorem identity_function_divisibility :
  ∀ f : ℕ+ → ℕ+, 
  (∀ x y : ℕ+, is_divisible (x.val * f x + y.val * f y) ((x.val^2 + y.val^2)^2022)) → 
  (∀ x : ℕ+, f x = x) := by sorry

end NUMINAMATH_CALUDE_identity_function_divisibility_l2153_215352


namespace NUMINAMATH_CALUDE_variance_scaling_l2153_215334

-- Define a set of data points
def DataSet : Type := List ℝ

-- Define the variance function
noncomputable def variance (data : DataSet) : ℝ := sorry

-- Define a function to multiply each data point by a scalar
def scaleData (data : DataSet) (scalar : ℝ) : DataSet :=
  data.map (· * scalar)

-- Theorem statement
theorem variance_scaling (data : DataSet) (s : ℝ) :
  variance data = s^2 → variance (scaleData data 2) = 4 * s^2 := by
  sorry

end NUMINAMATH_CALUDE_variance_scaling_l2153_215334


namespace NUMINAMATH_CALUDE_evaluate_expression_l2153_215361

theorem evaluate_expression : (3^4 + 3^4 + 3^4) / (3^(-4) + 3^(-4)) = 9841.5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2153_215361
