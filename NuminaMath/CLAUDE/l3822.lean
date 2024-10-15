import Mathlib

namespace NUMINAMATH_CALUDE_flag_arrangement_theorem_l3822_382259

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def N : ℕ := 858

/-- The number of blue flags -/
def blue_flags : ℕ := 12

/-- The number of green flags -/
def green_flags : ℕ := 11

/-- The total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- The number of flagpoles -/
def flagpoles : ℕ := 2

theorem flag_arrangement_theorem :
  (∀ (arrangement : Fin total_flags → Fin flagpoles),
    (∀ pole : Fin flagpoles, ∃ flag : Fin total_flags, arrangement flag = pole) ∧
    (∀ i j : Fin total_flags, i.val + 1 = j.val →
      (i.val < green_flags ∧ j.val < green_flags → arrangement i ≠ arrangement j) ∧
      (i.val ≥ green_flags ∧ j.val ≥ green_flags → arrangement i ≠ arrangement j)) →
    Fintype.card {arrangement : Fin total_flags → Fin flagpoles //
      (∀ pole : Fin flagpoles, ∃ flag : Fin total_flags, arrangement flag = pole) ∧
      (∀ i j : Fin total_flags, i.val + 1 = j.val →
        (i.val < green_flags ∧ j.val < green_flags → arrangement i ≠ arrangement j) ∧
        (i.val ≥ green_flags ∧ j.val ≥ green_flags → arrangement i ≠ arrangement j))} = N) :=
by sorry

end NUMINAMATH_CALUDE_flag_arrangement_theorem_l3822_382259


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3822_382246

theorem cubic_equation_roots (x : ℝ) : 
  let r1 := 2 * Real.sin (2 * Real.pi / 9)
  let r2 := 2 * Real.sin (8 * Real.pi / 9)
  let r3 := 2 * Real.sin (14 * Real.pi / 9)
  (x - r1) * (x - r2) * (x - r3) = x^3 - 3*x + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3822_382246


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l3822_382240

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
def C₂ (x y : ℝ) : Prop := (x + 12)^2 + y^2 = 225

-- Define the shortest tangent line segment
def shortest_tangent (R S : ℝ × ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    R = (x₁, y₁) ∧ S = (x₂, y₂) ∧
    C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
    ∀ (T U : ℝ × ℝ),
      C₁ T.1 T.2 → C₂ U.1 U.2 →
      Real.sqrt ((T.1 - U.1)^2 + (T.2 - U.2)^2) ≥ 
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem shortest_tangent_length :
  ∀ (R S : ℝ × ℝ),
    shortest_tangent R S →
    Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 
    Real.sqrt (16 - (60/19)^2) + Real.sqrt (225 - (225/19)^2) := by
  sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l3822_382240


namespace NUMINAMATH_CALUDE_remainder_5462_div_9_l3822_382278

theorem remainder_5462_div_9 : 5462 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_5462_div_9_l3822_382278


namespace NUMINAMATH_CALUDE_tim_final_soda_cans_l3822_382285

/-- Calculates the final number of soda cans Tim has -/
def final_soda_cans (initial : ℕ) (taken : ℕ) : ℕ :=
  let remaining := initial - taken
  let bought := remaining / 2
  remaining + bought

/-- Proves that Tim ends up with 24 cans of soda given the initial conditions -/
theorem tim_final_soda_cans : final_soda_cans 22 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_tim_final_soda_cans_l3822_382285


namespace NUMINAMATH_CALUDE_max_product_a2_a6_l3822_382203

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the maximum value of a₂ * a₆ in an arithmetic sequence where a₄ = 2 -/
theorem max_product_a2_a6 (a : ℕ → ℝ) (h : ArithmeticSequence a) (h4 : a 4 = 2) :
  (∀ b c : ℝ, a 2 = b ∧ a 6 = c → b * c ≤ 4) ∧ (∃ b c : ℝ, a 2 = b ∧ a 6 = c ∧ b * c = 4) :=
sorry

end NUMINAMATH_CALUDE_max_product_a2_a6_l3822_382203


namespace NUMINAMATH_CALUDE_algae_coverage_day21_l3822_382248

-- Define the algae coverage function
def algaeCoverage (day : ℕ) : ℚ :=
  1 / 2^(24 - day)

-- State the theorem
theorem algae_coverage_day21 :
  algaeCoverage 24 = 1 ∧ (∀ d : ℕ, algaeCoverage (d + 1) = 2 * algaeCoverage d) →
  algaeCoverage 21 = 1/8 :=
by
  sorry

end NUMINAMATH_CALUDE_algae_coverage_day21_l3822_382248


namespace NUMINAMATH_CALUDE_f_composition_result_l3822_382269

-- Define the function f for complex numbers
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^2 + 1 else -z^2 - 1

-- State the theorem
theorem f_composition_result : f (f (f (f (2 + I)))) = 3589 - 1984 * I := by
  sorry

end NUMINAMATH_CALUDE_f_composition_result_l3822_382269


namespace NUMINAMATH_CALUDE_inequality_holds_l3822_382279

theorem inequality_holds (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3822_382279


namespace NUMINAMATH_CALUDE_fraction_problem_l3822_382272

theorem fraction_problem (numerator denominator : ℤ) (x : ℤ) : 
  denominator = numerator - 4 →
  denominator = 5 →
  numerator + x = 3 * denominator →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3822_382272


namespace NUMINAMATH_CALUDE_set_subset_relations_l3822_382242

theorem set_subset_relations : 
  ({1,2,3} : Set ℕ) ⊆ {1,2,3} ∧ (∅ : Set ℕ) ⊆ {1} := by sorry

end NUMINAMATH_CALUDE_set_subset_relations_l3822_382242


namespace NUMINAMATH_CALUDE_nested_expression_evaluation_l3822_382200

theorem nested_expression_evaluation : (2*(2*(2*(2*(2*(2*(3+2)+2)+2)+2)+2)+2)+2) = 446 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_evaluation_l3822_382200


namespace NUMINAMATH_CALUDE_largest_circle_at_A_l3822_382222

/-- Represents a pentagon with circles at its vertices -/
structure PentagonWithCircles where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  AE : ℝ
  radA : ℝ
  radB : ℝ
  radC : ℝ
  radD : ℝ
  radE : ℝ
  circle_contact : 
    AB = radA + radB ∧
    BC = radB + radC ∧
    CD = radC + radD ∧
    DE = radD + radE ∧
    AE = radE + radA

/-- The circle centered at A has the largest radius -/
theorem largest_circle_at_A (p : PentagonWithCircles) 
  (h1 : p.AB = 16) (h2 : p.BC = 14) (h3 : p.CD = 17) (h4 : p.DE = 13) (h5 : p.AE = 14) :
  p.radA = max p.radA (max p.radB (max p.radC (max p.radD p.radE))) := by
  sorry

end NUMINAMATH_CALUDE_largest_circle_at_A_l3822_382222


namespace NUMINAMATH_CALUDE_coefficient_x_squared_zero_l3822_382261

theorem coefficient_x_squared_zero (x : ℝ) (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x ≠ 0, f x = (a + 1/x) * (1 + x)^4 ∧ 
   (∃ c₀ c₁ c₃ c₄ : ℝ, ∀ x ≠ 0, f x = c₀ + c₁*x + 0*x^2 + c₃*x^3 + c₄*x^4)) ↔ 
  a = -2/3 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_zero_l3822_382261


namespace NUMINAMATH_CALUDE_initial_speed_proof_l3822_382226

theorem initial_speed_proof (total_distance : ℝ) (first_duration : ℝ) (second_speed : ℝ) (second_duration : ℝ) (remaining_distance : ℝ) :
  total_distance = 600 →
  first_duration = 3 →
  second_speed = 80 →
  second_duration = 4 →
  remaining_distance = 130 →
  ∃ initial_speed : ℝ,
    initial_speed * first_duration + second_speed * second_duration = total_distance - remaining_distance ∧
    initial_speed = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_proof_l3822_382226


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3822_382297

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, 0]
def b : Fin 2 → ℝ := ![2, 1]

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, v = t • w ∨ w = t • v

theorem problem_1 (k : ℝ) :
  collinear (k • a - b) (a + 2 • b) ↔ k = -1/2 := by sorry

theorem problem_2 (m : ℝ) (A B C : Fin 2 → ℝ) :
  (B - A = 2 • a + 3 • b) →
  (C - B = a + m • b) →
  collinear (B - A) (C - B) →
  m = 3/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3822_382297


namespace NUMINAMATH_CALUDE_multiples_of_3_or_5_not_6_l3822_382239

def count_multiples (n : ℕ) (max : ℕ) : ℕ :=
  (max / n)

theorem multiples_of_3_or_5_not_6 (max : ℕ) (h : max = 150) : 
  count_multiples 3 max + count_multiples 5 max - count_multiples 15 max - count_multiples 6 max = 45 := by
  sorry

#check multiples_of_3_or_5_not_6

end NUMINAMATH_CALUDE_multiples_of_3_or_5_not_6_l3822_382239


namespace NUMINAMATH_CALUDE_equation_solution_l3822_382230

theorem equation_solution : ∃! x : ℝ, (1 : ℝ) / (x + 3) = 3 / (x + 9) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3822_382230


namespace NUMINAMATH_CALUDE_divisors_of_180_l3822_382283

theorem divisors_of_180 : Nat.card (Nat.divisors 180) = 18 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_180_l3822_382283


namespace NUMINAMATH_CALUDE_set_a_contains_one_l3822_382276

theorem set_a_contains_one (a : ℝ) : 
  let A : Set ℝ := {a, a^2}
  1 ∈ A → a = -1 := by sorry

end NUMINAMATH_CALUDE_set_a_contains_one_l3822_382276


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_parabola_vertex_l3822_382237

/-- Given that a, b, c, and d form an arithmetic sequence, and (a, d) is the vertex of y = x^2 - 2x + 5, prove that b + c = 5 -/
theorem arithmetic_sequence_and_parabola_vertex (a b c d : ℝ) :
  (∃ k : ℝ, b = a + k ∧ c = a + 2*k ∧ d = a + 3*k) →  -- arithmetic sequence condition
  (a = 1 ∧ d = 4) →  -- vertex condition (derived from y = x^2 - 2x + 5)
  b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_parabola_vertex_l3822_382237


namespace NUMINAMATH_CALUDE_woody_savings_l3822_382211

/-- The amount of money Woody already has -/
def money_saved (console_cost weekly_allowance weeks_to_save : ℕ) : ℕ :=
  console_cost - weekly_allowance * weeks_to_save

theorem woody_savings : money_saved 282 24 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_woody_savings_l3822_382211


namespace NUMINAMATH_CALUDE_boat_round_trip_average_speed_l3822_382205

/-- The average speed of a boat on a round trip, given its upstream and downstream speeds -/
theorem boat_round_trip_average_speed (distance : ℝ) (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 6)
  (h2 : downstream_speed = 3)
  (h3 : distance > 0) :
  (2 * distance) / ((distance / upstream_speed) + (distance / downstream_speed)) = 4 := by
  sorry

#check boat_round_trip_average_speed

end NUMINAMATH_CALUDE_boat_round_trip_average_speed_l3822_382205


namespace NUMINAMATH_CALUDE_vexel_language_words_l3822_382250

def alphabet_size : ℕ := 26
def max_word_length : ℕ := 5

def words_with_z (n : ℕ) : ℕ :=
  alphabet_size^n - (alphabet_size - 1)^n

def total_words : ℕ :=
  (words_with_z 1) + (words_with_z 2) + (words_with_z 3) + (words_with_z 4) + (words_with_z 5)

theorem vexel_language_words :
  total_words = 2205115 :=
by sorry

end NUMINAMATH_CALUDE_vexel_language_words_l3822_382250


namespace NUMINAMATH_CALUDE_four_digit_sum_11990_l3822_382281

def is_valid_digit (d : ℕ) : Prop := d > 0 ∧ d < 10

def distinct_digits (a b c d : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def smallest_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d
def largest_number (a b c d : ℕ) : ℕ := 1000 * d + 100 * c + 10 * b + a

theorem four_digit_sum_11990 (a b c d : ℕ) :
  distinct_digits a b c d →
  (smallest_number a b c d + largest_number a b c d = 11990 ↔
   ((a = 1 ∧ b = 9 ∧ c = 9 ∧ d = 9) ∨ (a = 9 ∧ b = 9 ∧ c = 9 ∧ d = 1))) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_11990_l3822_382281


namespace NUMINAMATH_CALUDE_pascal_triangle_15th_row_5th_number_l3822_382286

theorem pascal_triangle_15th_row_5th_number :
  Nat.choose 15 4 = 1365 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_15th_row_5th_number_l3822_382286


namespace NUMINAMATH_CALUDE_tax_to_savings_ratio_l3822_382293

/-- Esperanza's monthly finances -/
def monthly_finances (rent food mortgage savings tax gross_salary : ℚ) : Prop :=
  rent = 600 ∧
  food = (3/5) * rent ∧
  mortgage = 3 * food ∧
  savings = 2000 ∧
  gross_salary = 4840 ∧
  tax = gross_salary - (rent + food + mortgage + savings)

/-- The ratio of tax to savings is 2:5 -/
theorem tax_to_savings_ratio 
  (rent food mortgage savings tax gross_salary : ℚ) 
  (h : monthly_finances rent food mortgage savings tax gross_salary) : 
  tax / savings = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_tax_to_savings_ratio_l3822_382293


namespace NUMINAMATH_CALUDE_card_pack_size_l3822_382295

theorem card_pack_size (prob_not_face : ℝ) (num_face_cards : ℕ) (h1 : prob_not_face = 0.6923076923076923) (h2 : num_face_cards = 12) : 
  ∃ n : ℕ, n = 39 ∧ (n - num_face_cards : ℝ) / n = prob_not_face := by
sorry

end NUMINAMATH_CALUDE_card_pack_size_l3822_382295


namespace NUMINAMATH_CALUDE_scientific_notation_of_280000_l3822_382244

theorem scientific_notation_of_280000 : 
  280000 = 2.8 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_280000_l3822_382244


namespace NUMINAMATH_CALUDE_range_of_f_on_large_interval_l3822_382207

/-- A function with period 1 --/
def periodic_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 1) = g x

/-- The function f defined as f(x) = x + g(x) --/
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := x + g x

/-- The range of a function on an interval --/
def range_on (f : ℝ → ℝ) (a b : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc a b, f x = y}

theorem range_of_f_on_large_interval
    (g : ℝ → ℝ)
    (h_periodic : periodic_function g)
    (h_range : range_on (f g) 3 4 = Set.Icc (-2) 5) :
    range_on (f g) (-10) 10 = Set.Icc (-15) 11 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_on_large_interval_l3822_382207


namespace NUMINAMATH_CALUDE_prime_sum_square_fourth_power_l3822_382216

theorem prime_sum_square_fourth_power :
  ∀ p q r : ℕ,
  Prime p → Prime q → Prime r →
  p + q^2 = r^4 →
  p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_square_fourth_power_l3822_382216


namespace NUMINAMATH_CALUDE_largest_multiple_of_12_with_5_hundreds_l3822_382257

theorem largest_multiple_of_12_with_5_hundreds : ∃! n : ℕ, 
  n % 12 = 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧
  (n / 100) % 10 = 5 ∧
  ∀ m : ℕ, m % 12 = 0 → 100 ≤ m → m < 1000 → (m / 100) % 10 = 5 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_12_with_5_hundreds_l3822_382257


namespace NUMINAMATH_CALUDE_linear_function_value_l3822_382243

/-- A linear function f(x) = px + q -/
def f (p q : ℝ) (x : ℝ) : ℝ := p * x + q

theorem linear_function_value (p q : ℝ) :
  f p q 3 = 5 → f p q 5 = 9 → f p q 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_value_l3822_382243


namespace NUMINAMATH_CALUDE_remaining_balls_for_given_n_l3822_382260

/-- Represents the remaining balls after the removal process -/
def RemainingBalls (n : ℕ) : Finset ℕ := sorry

/-- The rule for removing balls -/
def removalRule (n : ℕ) (i : ℕ) : Bool := sorry

/-- The recurrence relation for the remaining balls -/
def F (n : ℕ) (i : ℕ) : ℕ := sorry

theorem remaining_balls_for_given_n (n : ℕ) (h : n ≥ 56) :
  (RemainingBalls 56 = {10, 20, 29, 37, 56}) →
  (n = 57 → RemainingBalls n = {5, 16, 26, 35, 43}) ∧
  (n = 58 → RemainingBalls n = {11, 22, 32, 41, 49}) ∧
  (n = 59 → RemainingBalls n = {17, 28, 38, 47, 55}) ∧
  (n = 60 → RemainingBalls n = {11, 23, 34, 44, 53}) := by
  sorry

end NUMINAMATH_CALUDE_remaining_balls_for_given_n_l3822_382260


namespace NUMINAMATH_CALUDE_no_inscribed_sphere_l3822_382208

structure Polyhedron where
  faces : ℕ
  paintedFaces : ℕ
  convex : Bool
  noAdjacentPainted : Bool

def canInscribeSphere (p : Polyhedron) : Prop :=
  sorry

theorem no_inscribed_sphere (p : Polyhedron) 
  (h_convex : p.convex = true)
  (h_painted : p.paintedFaces > p.faces / 2)
  (h_noAdjacent : p.noAdjacentPainted = true) :
  ¬(canInscribeSphere p) := by
  sorry

end NUMINAMATH_CALUDE_no_inscribed_sphere_l3822_382208


namespace NUMINAMATH_CALUDE_investment_interest_rate_proof_l3822_382209

/-- Proves that for an investment of 7000 over 2 years, if the interest earned is 840 more than
    what would be earned at 12% p.a., then the interest rate is 18% p.a. -/
theorem investment_interest_rate_proof 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_diff : ℝ) 
  (base_rate : ℝ) 
  (h1 : principal = 7000)
  (h2 : time = 2)
  (h3 : interest_diff = 840)
  (h4 : base_rate = 12)
  (h5 : principal * (rate / 100) * time - principal * (base_rate / 100) * time = interest_diff) :
  rate = 18 := by
  sorry

#check investment_interest_rate_proof

end NUMINAMATH_CALUDE_investment_interest_rate_proof_l3822_382209


namespace NUMINAMATH_CALUDE_perimeter_of_square_d_l3822_382282

/-- Given two squares C and D, where C has a side length of 10 cm and D has an area
    that is half the area of C, the perimeter of D is 20√2 cm. -/
theorem perimeter_of_square_d (c d : Real) : 
  c = 10 →  -- side length of square C
  d ^ 2 = (c ^ 2) / 2 →  -- area of D is half the area of C
  4 * d = 20 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_perimeter_of_square_d_l3822_382282


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3822_382218

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The roots of the quadratic equation -/
def are_roots (a : ℕ → ℝ) : Prop :=
  3 * (a 1)^2 + 7 * (a 1) - 9 = 0 ∧ 3 * (a 10)^2 + 7 * (a 10) - 9 = 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a → are_roots a → a 4 * a 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3822_382218


namespace NUMINAMATH_CALUDE_cube_difference_l3822_382273

theorem cube_difference (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l3822_382273


namespace NUMINAMATH_CALUDE_roots_relationship_l3822_382292

/-- The polynomial h(x) = x^3 - 2x^2 - x + 2 -/
def h (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

/-- The polynomial j(x) = x^3 + bx^2 + cx + d -/
def j (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- The theorem stating the relationship between h and j -/
theorem roots_relationship (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, h x = 0 → j (x^3) b c d = 0) →
  b = 10 ∧ c = 31 ∧ d = 30 := by
sorry

end NUMINAMATH_CALUDE_roots_relationship_l3822_382292


namespace NUMINAMATH_CALUDE_product_remainder_l3822_382263

theorem product_remainder (a b m : ℕ) (ha : a = 1488) (hb : b = 1977) (hm : m = 500) :
  (a * b) % m = 276 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l3822_382263


namespace NUMINAMATH_CALUDE_pierre_ice_cream_scoops_l3822_382289

/-- 
Given:
- The cost of each scoop of ice cream
- The number of scoops Pierre's mom gets
- The total bill amount
Prove that Pierre gets 3 scoops of ice cream
-/
theorem pierre_ice_cream_scoops 
  (cost_per_scoop : ℕ) 
  (mom_scoops : ℕ) 
  (total_bill : ℕ) 
  (h1 : cost_per_scoop = 2)
  (h2 : mom_scoops = 4)
  (h3 : total_bill = 14) :
  ∃ (pierre_scoops : ℕ), 
    pierre_scoops = 3 ∧ 
    cost_per_scoop * (pierre_scoops + mom_scoops) = total_bill :=
by sorry

end NUMINAMATH_CALUDE_pierre_ice_cream_scoops_l3822_382289


namespace NUMINAMATH_CALUDE_network_connections_l3822_382224

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  (n * k) / 2 = 30 :=
by sorry

end NUMINAMATH_CALUDE_network_connections_l3822_382224


namespace NUMINAMATH_CALUDE_trig_identity_l3822_382291

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3822_382291


namespace NUMINAMATH_CALUDE_ratio_c_to_d_l3822_382270

theorem ratio_c_to_d (a b c d : ℝ) 
  (hab : a / b = 3 / 4)
  (hbc : b / c = 7 / 9)
  (had : a / d = 0.4166666666666667) :
  c / d = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_c_to_d_l3822_382270


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3822_382234

/-- Theorem: Train Speed Calculation
Given two trains starting from the same station, traveling along parallel tracks in the same direction,
with one train traveling at 35 mph, and the distance between them after 10 hours being 250 miles,
the speed of the first train is 60 mph. -/
theorem train_speed_calculation (speed_second_train : ℝ) (time : ℝ) (distance : ℝ) :
  speed_second_train = 35 →
  time = 10 →
  distance = 250 →
  ∃ (speed_first_train : ℝ),
    speed_first_train > 0 ∧
    distance = (speed_first_train - speed_second_train) * time ∧
    speed_first_train = 60 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3822_382234


namespace NUMINAMATH_CALUDE_tangent_line_slope_l3822_382298

/-- Given that the line y = kx is tangent to the curve y = x + exp(-x), prove that k = 1 - exp(1) -/
theorem tangent_line_slope (k : ℝ) : 
  (∃ x₀ : ℝ, k * x₀ = x₀ + Real.exp (-x₀) ∧ 
              k = 1 - Real.exp (-x₀)) → 
  k = 1 - Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l3822_382298


namespace NUMINAMATH_CALUDE_ellipse_m_value_l3822_382204

/-- The value of m for an ellipse with given properties -/
theorem ellipse_m_value (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, x^2 / 25 + y^2 / m^2 = 1) →
  (∃ c : ℝ, c = 4 ∧ c^2 = 25 - m^2) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l3822_382204


namespace NUMINAMATH_CALUDE_min_t_for_inequality_l3822_382223

theorem min_t_for_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (∀ x y, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) ≤ (1 / (2 * Real.sqrt 6)) * (2 * x + 3 * y)) ∧
  (∀ ε > 0, ∃ x y, x ≥ 0 ∧ y ≥ 0 ∧ Real.sqrt (x * y) > (1 / (2 * Real.sqrt 6) - ε) * (2 * x + 3 * y)) :=
sorry

end NUMINAMATH_CALUDE_min_t_for_inequality_l3822_382223


namespace NUMINAMATH_CALUDE_det_max_value_l3822_382251

open Real

-- Define the determinant function
noncomputable def det (θ : ℝ) : ℝ :=
  let a := 1 + sin θ
  let b := 1 + cos θ
  a * (1 - a^2) - b * (1 - a * b) + (1 - b * a)

-- State the theorem
theorem det_max_value :
  ∀ θ : ℝ, det θ ≤ -1 ∧ ∃ θ₀ : ℝ, det θ₀ = -1 :=
by sorry

end NUMINAMATH_CALUDE_det_max_value_l3822_382251


namespace NUMINAMATH_CALUDE_weight_of_six_moles_of_compound_l3822_382280

/-- The weight of a given number of moles of a compound -/
def weight (moles : ℝ) (molecular_weight : ℝ) : ℝ :=
  moles * molecular_weight

/-- Proof that the weight of 6 moles of a compound with molecular weight 1404 is 8424 -/
theorem weight_of_six_moles_of_compound (molecular_weight : ℝ) 
  (h : molecular_weight = 1404) : weight 6 molecular_weight = 8424 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_six_moles_of_compound_l3822_382280


namespace NUMINAMATH_CALUDE_lamps_turned_on_l3822_382266

theorem lamps_turned_on (total_lamps : ℕ) (statement1 statement2 statement3 statement4 : Prop) :
  total_lamps = 10 →
  (statement1 ↔ (∃ x : ℕ, x = 5 ∧ x = total_lamps - (total_lamps - x))) →
  (statement2 ↔ ¬statement1) →
  (statement3 ↔ (∃ y : ℕ, y = 3 ∧ y = total_lamps - (total_lamps - y))) →
  (statement4 ↔ ∃ z : ℕ, z = total_lamps - (total_lamps - z) ∧ 2 ∣ z) →
  (statement1 ∨ statement2 ∨ statement3 ∨ statement4) →
  (statement1 → ¬statement2 ∧ ¬statement3 ∧ ¬statement4) →
  (statement2 → ¬statement1 ∧ ¬statement3 ∧ ¬statement4) →
  (statement3 → ¬statement1 ∧ ¬statement2 ∧ ¬statement4) →
  (statement4 → ¬statement1 ∧ ¬statement2 ∧ ¬statement3) →
  ∃ (lamps_on : ℕ), lamps_on = 9 ∧ lamps_on = total_lamps - (total_lamps - lamps_on) :=
by sorry

end NUMINAMATH_CALUDE_lamps_turned_on_l3822_382266


namespace NUMINAMATH_CALUDE_parking_spaces_remaining_l3822_382219

theorem parking_spaces_remaining (total_spaces : ℕ) (caravan_spaces : ℕ) (parked_caravans : ℕ) : 
  total_spaces = 30 → caravan_spaces = 2 → parked_caravans = 3 → 
  total_spaces - (caravan_spaces * parked_caravans) = 24 := by
  sorry

end NUMINAMATH_CALUDE_parking_spaces_remaining_l3822_382219


namespace NUMINAMATH_CALUDE_rental_cost_equality_l3822_382217

/-- The daily rate charged by Safety Rent-a-Car in dollars -/
def safety_daily_rate : ℝ := 21.95

/-- The per-mile rate charged by Safety Rent-a-Car in dollars -/
def safety_mile_rate : ℝ := 0.19

/-- The daily rate charged by City Rentals in dollars -/
def city_daily_rate : ℝ := 18.95

/-- The per-mile rate charged by City Rentals in dollars -/
def city_mile_rate : ℝ := 0.21

/-- The mileage at which the cost is the same for both rental companies -/
def equal_cost_mileage : ℝ := 150

theorem rental_cost_equality :
  safety_daily_rate + safety_mile_rate * equal_cost_mileage =
  city_daily_rate + city_mile_rate * equal_cost_mileage :=
by sorry

end NUMINAMATH_CALUDE_rental_cost_equality_l3822_382217


namespace NUMINAMATH_CALUDE_alex_grocery_delivery_l3822_382290

/-- Represents the problem of calculating the total value of groceries Alex delivered --/
theorem alex_grocery_delivery 
  (savings : ℝ) 
  (car_cost : ℝ) 
  (trip_charge : ℝ) 
  (grocery_charge_percent : ℝ) 
  (num_trips : ℕ) 
  (h1 : savings = 14500) 
  (h2 : car_cost = 14600) 
  (h3 : trip_charge = 1.5) 
  (h4 : grocery_charge_percent = 0.05) 
  (h5 : num_trips = 40) 
  (h6 : savings + num_trips * trip_charge + grocery_charge_percent * (car_cost - savings - num_trips * trip_charge) / grocery_charge_percent ≥ car_cost) : 
  (car_cost - savings - num_trips * trip_charge) / grocery_charge_percent = 800 := by
  sorry

end NUMINAMATH_CALUDE_alex_grocery_delivery_l3822_382290


namespace NUMINAMATH_CALUDE_money_redistribution_l3822_382228

theorem money_redistribution (younger_money : ℝ) :
  let elder_money := 1.25 * younger_money
  let total_money := younger_money + elder_money
  let equal_share := total_money / 2
  let transfer_amount := equal_share - younger_money
  (transfer_amount / elder_money) = 0.1 := by
sorry

end NUMINAMATH_CALUDE_money_redistribution_l3822_382228


namespace NUMINAMATH_CALUDE_T_eight_three_l3822_382268

def T (a b : ℤ) : ℤ := 4*a + 5*b - 1

theorem T_eight_three : T 8 3 = 46 := by sorry

end NUMINAMATH_CALUDE_T_eight_three_l3822_382268


namespace NUMINAMATH_CALUDE_coin_array_problem_l3822_382212

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem coin_array_problem :
  ∃ N : ℕ, triangular_sum N = 3003 ∧ sum_of_digits N = 14 := by
  sorry

end NUMINAMATH_CALUDE_coin_array_problem_l3822_382212


namespace NUMINAMATH_CALUDE_inequality_implies_a_range_l3822_382249

open Real

theorem inequality_implies_a_range (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < π / 2 → 
    sqrt 2 * (2 * a + 3) * cos (θ - π / 4) + 6 / (sin θ + cos θ) - 2 * sin (2 * θ) < 3 * a + 6) →
  a > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_range_l3822_382249


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_32_l3822_382252

theorem smallest_positive_multiple_of_32 :
  ∀ n : ℕ, n > 0 → 32 * 1 ≤ 32 * n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_32_l3822_382252


namespace NUMINAMATH_CALUDE_least_non_factor_non_prime_l3822_382284

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem least_non_factor_non_prime : 
  ∃ (n : ℕ), n > 0 ∧ ¬(factorial 30 % n = 0) ∧ ¬(is_prime n) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → (factorial 30 % m = 0) ∨ (is_prime m)) ∧ n = 961 := by
  sorry

end NUMINAMATH_CALUDE_least_non_factor_non_prime_l3822_382284


namespace NUMINAMATH_CALUDE_units_digit_of_product_first_four_composites_l3822_382241

def first_four_composite_numbers : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_product_first_four_composites :
  units_digit (product_of_list first_four_composite_numbers) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_first_four_composites_l3822_382241


namespace NUMINAMATH_CALUDE_gcd_228_2010_l3822_382202

theorem gcd_228_2010 : Nat.gcd 228 2010 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_2010_l3822_382202


namespace NUMINAMATH_CALUDE_sequence_periodicity_implies_zero_l3822_382271

theorem sequence_periodicity_implies_zero (a b c d : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + b n)
  (h2 : ∀ n, b (n + 1) = b n + c n)
  (h3 : ∀ n, c (n + 1) = c n + d n)
  (h4 : ∀ n, d (n + 1) = d n + a n)
  (h5 : ∃ k m : ℕ, k ≥ 1 ∧ m ≥ 1 ∧ 
    a (k + m) = a m ∧ 
    b (k + m) = b m ∧ 
    c (k + m) = c m ∧ 
    d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_implies_zero_l3822_382271


namespace NUMINAMATH_CALUDE_min_lines_theorem_l3822_382225

/-- A plane -/
structure Plane where

/-- A point in a plane -/
structure Point (α : Plane) where

/-- A line in a plane -/
structure Line (α : Plane) where

/-- A ray in a plane -/
structure Ray (α : Plane) where

/-- Predicate for a line not passing through a point -/
def LineNotThroughPoint (α : Plane) (l : Line α) (P : Point α) : Prop :=
  sorry

/-- Predicate for a ray intersecting a line -/
def RayIntersectsLine (α : Plane) (r : Ray α) (l : Line α) : Prop :=
  sorry

/-- The minimum number of lines theorem -/
theorem min_lines_theorem (α : Plane) (P : Point α) (k : ℕ) (h : k > 0) :
  ∃ (n : ℕ),
    (∀ (m : ℕ),
      (∃ (lines : Fin m → Line α),
        (∀ i, LineNotThroughPoint α (lines i) P) ∧
        (∀ r : Ray α, ∃ (S : Finset (Fin m)), S.card ≥ k ∧ ∀ i ∈ S, RayIntersectsLine α r (lines i)))
      → m ≥ n) ∧
    (∃ (lines : Fin (2 * k + 1) → Line α),
      (∀ i, LineNotThroughPoint α (lines i) P) ∧
      (∀ r : Ray α, ∃ (S : Finset (Fin (2 * k + 1))), S.card ≥ k ∧ ∀ i ∈ S, RayIntersectsLine α r (lines i))) :=
  sorry

end NUMINAMATH_CALUDE_min_lines_theorem_l3822_382225


namespace NUMINAMATH_CALUDE_tan_product_equals_two_l3822_382287

theorem tan_product_equals_two :
  (∀ x y z : ℝ, x = 100 ∧ y = 35 ∧ z = 135 →
    Real.tan (z * π / 180) = -1 →
    (1 - Real.tan (x * π / 180)) * (1 - Real.tan (y * π / 180)) = 2) :=
by sorry

end NUMINAMATH_CALUDE_tan_product_equals_two_l3822_382287


namespace NUMINAMATH_CALUDE_expression_evaluation_l3822_382294

theorem expression_evaluation :
  let x : ℝ := 2
  (2 * (x^2 - 1) - 7*x - (2*x^2 - x + 3)) = -17 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3822_382294


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l3822_382253

theorem price_decrease_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 900)
  (h2 : new_price = 684) :
  (original_price - new_price) / original_price * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l3822_382253


namespace NUMINAMATH_CALUDE_perpendicular_line_to_cosine_tangent_l3822_382221

open Real

/-- The equation of a line perpendicular to the tangent of y = cos x at (π/3, 1/2) --/
theorem perpendicular_line_to_cosine_tangent :
  let f : ℝ → ℝ := fun x ↦ cos x
  let p : ℝ × ℝ := (π / 3, 1 / 2)
  let tangent_slope : ℝ := -sin (π / 3)
  let perpendicular_slope : ℝ := -1 / tangent_slope
  let line_equation : ℝ → ℝ → ℝ := fun x y ↦ 2 * x - sqrt 3 * y - 2 * π / 3 + sqrt 3 / 2
  (f (π / 3) = 1 / 2) →
  (perpendicular_slope = 2 / sqrt 3) →
  (∀ x y, line_equation x y = 0 ↔ y - p.2 = perpendicular_slope * (x - p.1)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_cosine_tangent_l3822_382221


namespace NUMINAMATH_CALUDE_min_value_theorem_l3822_382275

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : f a b 1 = 2) :
  ∃ (min_val : ℝ), min_val = 9 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → f a' b' 1 = 2 → 1/a' + 4/b' ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3822_382275


namespace NUMINAMATH_CALUDE_correct_factorization_l3822_382238

theorem correct_factorization (a : ℝ) : a^2 - 3*a - 4 = (a - 4) * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l3822_382238


namespace NUMINAMATH_CALUDE_pages_in_harrys_book_l3822_382265

/-- Given that Selena's book has x pages and Harry's book has y fewer pages than half the number
    of pages of Selena's book, prove that the number of pages in Harry's book is equal to (x/2) - y. -/
theorem pages_in_harrys_book (x y : ℕ) : ℕ :=
  x / 2 - y

#check pages_in_harrys_book

end NUMINAMATH_CALUDE_pages_in_harrys_book_l3822_382265


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3822_382247

theorem inequality_solution_set (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  let S := {x : ℝ | (x - a) * (x + a - 1) < 0}
  (0 ≤ a ∧ a < (1/2) → S = Set.Ioo a (1 - a)) ∧
  (a = (1/2) → S = ∅) ∧
  ((1/2) < a ∧ a ≤ 1 → S = Set.Ioo (1 - a) a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3822_382247


namespace NUMINAMATH_CALUDE_percentage_problem_l3822_382233

theorem percentage_problem : 
  let percentage : ℝ := 12
  let total : ℝ := 160
  let given_percentage : ℝ := 38
  let given_total : ℝ := 80
  let difference : ℝ := 11.2
  (given_percentage / 100) * given_total - (percentage / 100) * total = difference
  := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l3822_382233


namespace NUMINAMATH_CALUDE_angle_1303_equivalent_to_negative_137_l3822_382214

-- Define a function to reduce an angle to its equivalent angle between 0° and 360°
def reduce_angle (angle : Int) : Int :=
  angle % 360

-- Theorem statement
theorem angle_1303_equivalent_to_negative_137 :
  reduce_angle 1303 = reduce_angle (-137) :=
sorry

end NUMINAMATH_CALUDE_angle_1303_equivalent_to_negative_137_l3822_382214


namespace NUMINAMATH_CALUDE_minimum_value_curve_exponent_l3822_382227

theorem minimum_value_curve_exponent (m n : ℝ) (a : ℝ) : 
  m > 0 → n > 0 → m + n = 1 → 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (1/x) + (16/y) ≥ (1/m) + (16/n)) →
  ((m/5)^a = n/4) →
  a = 1/2 := by sorry

end NUMINAMATH_CALUDE_minimum_value_curve_exponent_l3822_382227


namespace NUMINAMATH_CALUDE_equation_solution_l3822_382254

theorem equation_solution : 
  ∃ (x : ℚ), x = -3/4 ∧ x/(x+1) = 3/x + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3822_382254


namespace NUMINAMATH_CALUDE_pizza_cost_l3822_382258

theorem pizza_cost (total_cost : ℝ) (num_pizzas : ℕ) (h1 : total_cost = 24) (h2 : num_pizzas = 3) :
  total_cost / num_pizzas = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_cost_l3822_382258


namespace NUMINAMATH_CALUDE_correct_emu_count_l3822_382236

/-- The number of emus in Farmer Brown's flock -/
def num_emus : ℕ := 20

/-- The number of heads per emu -/
def heads_per_emu : ℕ := 1

/-- The number of legs per emu -/
def legs_per_emu : ℕ := 2

/-- The total count of heads and legs in the flock -/
def total_count : ℕ := 60

/-- Theorem stating that the number of emus is correct given the conditions -/
theorem correct_emu_count : 
  num_emus * (heads_per_emu + legs_per_emu) = total_count :=
by sorry

end NUMINAMATH_CALUDE_correct_emu_count_l3822_382236


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_l3822_382232

/-- The equation of the curve in polar coordinates -/
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.sin θ)

/-- The equation of the curve in Cartesian coordinates -/
def cartesian_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) - y = 1

/-- The definition of a hyperbola in Cartesian coordinates -/
def is_hyperbola (f : ℝ × ℝ → ℝ) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
    ∀ x y, f (x, y) = a * x^2 + b * y^2 + c * x * y + d * x + e * y

theorem curve_is_hyperbola :
  ∃ f : ℝ × ℝ → ℝ, (∀ x y, f (x, y) = 0 ↔ cartesian_equation x y) ∧ is_hyperbola f :=
sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_l3822_382232


namespace NUMINAMATH_CALUDE_sum_of_powers_zero_l3822_382229

theorem sum_of_powers_zero : -(-1)^2006 - (-1)^2007 - 1^2008 - (-1)^2009 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_zero_l3822_382229


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3822_382201

theorem no_integer_solutions :
  ¬∃ (x y : ℤ), x^3 + 4*x^2 - 11*x + 30 = 8*y^3 + 24*y^2 + 18*y + 7 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3822_382201


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l3822_382267

/-- Caterer pricing structure -/
structure CatererPricing where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculate total cost for a caterer given number of people -/
def totalCost (pricing : CatererPricing) (people : ℕ) : ℕ :=
  pricing.basicFee + pricing.perPersonFee * people

/-- First caterer's pricing -/
def firstCaterer : CatererPricing :=
  { basicFee := 150, perPersonFee := 18 }

/-- Second caterer's pricing -/
def secondCaterer : CatererPricing :=
  { basicFee := 250, perPersonFee := 15 }

/-- Theorem: 34 is the least number of people for which the second caterer is less expensive -/
theorem second_caterer_cheaper_at_34 :
  (∀ n : ℕ, n < 34 → totalCost firstCaterer n ≤ totalCost secondCaterer n) ∧
  (totalCost secondCaterer 34 < totalCost firstCaterer 34) :=
by sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l3822_382267


namespace NUMINAMATH_CALUDE_diamond_solution_l3822_382256

def diamond (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem diamond_solution (a : ℝ) : diamond a 3 = 6 → a = 8 := by
  sorry

end NUMINAMATH_CALUDE_diamond_solution_l3822_382256


namespace NUMINAMATH_CALUDE_zander_sand_lorries_l3822_382213

/-- Represents the construction materials purchase scenario --/
structure ConstructionPurchase where
  total_payment : ℕ
  cement_bags : ℕ
  cement_price_per_bag : ℕ
  sand_tons_per_lorry : ℕ
  sand_price_per_ton : ℕ

/-- Calculates the number of lorries of sand purchased --/
def sand_lorries (purchase : ConstructionPurchase) : ℕ :=
  let cement_cost := purchase.cement_bags * purchase.cement_price_per_bag
  let sand_cost := purchase.total_payment - cement_cost
  let sand_price_per_lorry := purchase.sand_tons_per_lorry * purchase.sand_price_per_ton
  sand_cost / sand_price_per_lorry

/-- Theorem stating that for the given purchase scenario, the number of sand lorries is 20 --/
theorem zander_sand_lorries :
  let purchase := ConstructionPurchase.mk 13000 500 10 10 40
  sand_lorries purchase = 20 := by
  sorry

end NUMINAMATH_CALUDE_zander_sand_lorries_l3822_382213


namespace NUMINAMATH_CALUDE_guest_bathroom_towel_sets_l3822_382231

theorem guest_bathroom_towel_sets :
  let master_sets : ℕ := 4
  let guest_price : ℚ := 40
  let master_price : ℚ := 50
  let discount : ℚ := 20 / 100
  let total_spent : ℚ := 224
  let discounted_guest_price : ℚ := guest_price * (1 - discount)
  let discounted_master_price : ℚ := master_price * (1 - discount)
  ∃ guest_sets : ℕ,
    guest_sets * discounted_guest_price + master_sets * discounted_master_price = total_spent ∧
    guest_sets = 2 :=
by sorry

end NUMINAMATH_CALUDE_guest_bathroom_towel_sets_l3822_382231


namespace NUMINAMATH_CALUDE_square_with_arcs_area_l3822_382296

/-- The area of the regions inside a square with side length 3 cm, 
    but outside two quarter-circle arcs from adjacent corners. -/
theorem square_with_arcs_area : 
  let square_side : ℝ := 3
  let quarter_circle_area := (π * square_side^2) / 4
  let triangle_area := (square_side^2) / 2
  let arc_area := 2 * (quarter_circle_area - triangle_area)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |arc_area - 5.1| < ε :=
sorry

end NUMINAMATH_CALUDE_square_with_arcs_area_l3822_382296


namespace NUMINAMATH_CALUDE_inequality_proof_l3822_382277

theorem inequality_proof (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4 > x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3822_382277


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3822_382274

/-- An isosceles trapezoid with specific dimensions and inscribed circles. -/
structure IsoscelesTrapezoidWithCircles where
  /-- The length of side AB -/
  ab : ℝ
  /-- The length of sides BC and DA -/
  bc_da : ℝ
  /-- The length of side CD -/
  cd : ℝ
  /-- The radius of circles centered at A and B -/
  outer_radius_large : ℝ
  /-- The radius of circles centered at C and D -/
  outer_radius_small : ℝ
  /-- Constraint: AB = 8 -/
  ab_eq : ab = 8
  /-- Constraint: BC = DA = 6 -/
  bc_da_eq : bc_da = 6
  /-- Constraint: CD = 5 -/
  cd_eq : cd = 5
  /-- Constraint: Radius of circles at A and B is 4 -/
  outer_radius_large_eq : outer_radius_large = 4
  /-- Constraint: Radius of circles at C and D is 3 -/
  outer_radius_small_eq : outer_radius_small = 3

/-- The theorem stating the radius of the inscribed circle tangent to all four outer circles. -/
theorem inscribed_circle_radius (t : IsoscelesTrapezoidWithCircles) :
  ∃ r : ℝ, r = (-105 + 4 * Real.sqrt 141) / 13 ∧
    r > 0 ∧
    ∃ (x y : ℝ),
      x^2 + y^2 = (r + t.outer_radius_large)^2 ∧
      x^2 + (t.bc_da - y)^2 = (r + t.outer_radius_small)^2 ∧
      2*x = t.ab - 2*t.outer_radius_large :=
by sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_l3822_382274


namespace NUMINAMATH_CALUDE_overlapping_segments_length_l3822_382206

/-- Given a set of overlapping segments with known total length and span, 
    this theorem proves the length of each overlapping part. -/
theorem overlapping_segments_length 
  (total_length : ℝ) 
  (edge_to_edge : ℝ) 
  (num_overlaps : ℕ) 
  (h1 : total_length = 98) 
  (h2 : edge_to_edge = 83) 
  (h3 : num_overlaps = 6) :
  (total_length - edge_to_edge) / num_overlaps = 2.5 := by
  sorry

#check overlapping_segments_length

end NUMINAMATH_CALUDE_overlapping_segments_length_l3822_382206


namespace NUMINAMATH_CALUDE_count_triangles_in_dodecagon_l3822_382245

/-- The number of triangles that can be formed from the vertices of a dodecagon -/
def triangles_in_dodecagon : ℕ := 220

/-- The number of vertices in a dodecagon -/
def dodecagon_vertices : ℕ := 12

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- Theorem: The number of triangles that can be formed by selecting 3 vertices
    from a 12-vertex polygon is equal to 220 -/
theorem count_triangles_in_dodecagon :
  Nat.choose dodecagon_vertices triangle_vertices = triangles_in_dodecagon := by
  sorry

end NUMINAMATH_CALUDE_count_triangles_in_dodecagon_l3822_382245


namespace NUMINAMATH_CALUDE_tripodasaurus_count_l3822_382288

/-- A tripodasaurus is a creature with 3 legs and 1 head -/
structure Tripodasaurus where
  legs : Nat
  head : Nat

/-- A flock of tripodasauruses -/
structure Flock where
  count : Nat

/-- The total number of heads and legs in a flock -/
def totalHeadsAndLegs (f : Flock) : Nat :=
  f.count * (3 + 1)  -- 3 legs + 1 head per tripodasaurus

theorem tripodasaurus_count (f : Flock) :
  totalHeadsAndLegs f = 20 → f.count = 5 := by
  sorry

end NUMINAMATH_CALUDE_tripodasaurus_count_l3822_382288


namespace NUMINAMATH_CALUDE_least_pencils_l3822_382255

theorem least_pencils (p : ℕ) : p > 0 ∧ 
  p % 5 = 4 ∧ 
  p % 6 = 3 ∧ 
  p % 8 = 5 ∧ 
  (∀ q : ℕ, q > 0 ∧ q % 5 = 4 ∧ q % 6 = 3 ∧ q % 8 = 5 → p ≤ q) → 
  p = 69 := by
sorry

end NUMINAMATH_CALUDE_least_pencils_l3822_382255


namespace NUMINAMATH_CALUDE_N_subset_M_l3822_382299

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2 * p.1 + 1}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -(p.1^2)}

-- State the theorem
theorem N_subset_M : N ⊆ M := by sorry

end NUMINAMATH_CALUDE_N_subset_M_l3822_382299


namespace NUMINAMATH_CALUDE_symmetry_across_y_eq_neg_x_l3822_382262

/-- Given two lines in the xy-plane, this function checks if they are symmetrical across y = -x -/
def are_symmetrical_lines (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), line1 x y ↔ line2 y x

/-- The original line: √3x + y + 1 = 0 -/
def original_line (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y + 1 = 0

/-- The proposed symmetrical line: x + √3y - 1 = 0 -/
def symmetrical_line (x y : ℝ) : Prop :=
  x + Real.sqrt 3 * y - 1 = 0

/-- Theorem stating that the symmetrical_line is indeed symmetrical to the original_line across y = -x -/
theorem symmetry_across_y_eq_neg_x :
  are_symmetrical_lines original_line symmetrical_line :=
sorry

end NUMINAMATH_CALUDE_symmetry_across_y_eq_neg_x_l3822_382262


namespace NUMINAMATH_CALUDE_book_selection_theorem_l3822_382264

theorem book_selection_theorem (n m : ℕ) (h1 : n = 8) (h2 : m = 5) :
  (Nat.choose (n - 1) (m - 1)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l3822_382264


namespace NUMINAMATH_CALUDE_village_population_equality_l3822_382220

/-- The initial population of Village X -/
def Px : ℕ := sorry

/-- The yearly decrease in population of Village X -/
def decrease_x : ℕ := 1200

/-- The initial population of Village Y -/
def Py : ℕ := 42000

/-- The yearly increase in population of Village Y -/
def increase_y : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years : ℕ := 17

theorem village_population_equality :
  Px - years * decrease_x = Py + years * increase_y ∧ Px = 76000 := by sorry

end NUMINAMATH_CALUDE_village_population_equality_l3822_382220


namespace NUMINAMATH_CALUDE_opposite_of_negative_third_l3822_382215

theorem opposite_of_negative_third : 
  (fun x : ℚ => -x) (-1/3) = 1/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_third_l3822_382215


namespace NUMINAMATH_CALUDE_fish_in_tank_l3822_382210

theorem fish_in_tank (total : ℕ) (blue : ℕ) (spotted : ℕ) : 
  3 * blue = total →
  2 * spotted = blue →
  spotted = 5 →
  total = 30 := by
sorry

end NUMINAMATH_CALUDE_fish_in_tank_l3822_382210


namespace NUMINAMATH_CALUDE_line_through_center_line_bisecting_chord_l3822_382235

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 16

-- Define point P
def point_P : ℝ × ℝ := (2, 2)

-- Define the equation of line l passing through P and center of C
def line_l_through_center (x y : ℝ) : Prop := 2*x - y - 2 = 0

-- Define the equation of line l bisecting chord AB
def line_l_bisecting_chord (x y : ℝ) : Prop := x + 2*y - 6 = 0

-- Theorem 1: Line l passing through P and center of C
theorem line_through_center : 
  ∀ x y : ℝ, circle_C x y → line_l_through_center x y → 
  ∃ t : ℝ, x = 2 + t ∧ y = 2 + 2*t :=
sorry

-- Theorem 2: Line l passing through P and bisecting chord AB
theorem line_bisecting_chord :
  ∀ x y : ℝ, circle_C x y → line_l_bisecting_chord x y →
  ∃ t : ℝ, x = 2 + t ∧ y = 2 - t/2 :=
sorry

end NUMINAMATH_CALUDE_line_through_center_line_bisecting_chord_l3822_382235
