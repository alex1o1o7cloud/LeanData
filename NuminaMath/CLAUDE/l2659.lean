import Mathlib

namespace NUMINAMATH_CALUDE_bill_face_value_l2659_265956

/-- Calculates the face value of a bill given the true discount, time period, and annual interest rate. -/
def calculate_face_value (true_discount : ℚ) (time_months : ℚ) (annual_rate : ℚ) : ℚ :=
  (true_discount * (100 + annual_rate * (time_months / 12))) / (annual_rate * (time_months / 12))

/-- Theorem stating that given the specified conditions, the face value of the bill is 2520. -/
theorem bill_face_value :
  let true_discount : ℚ := 270
  let time_months : ℚ := 9
  let annual_rate : ℚ := 16
  calculate_face_value true_discount time_months annual_rate = 2520 :=
by sorry

end NUMINAMATH_CALUDE_bill_face_value_l2659_265956


namespace NUMINAMATH_CALUDE_tan_double_angle_l2659_265986

/-- Given an angle θ with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (-1, 2), prove that tan 2θ = 4/3 -/
theorem tan_double_angle (θ : ℝ) : 
  (∃ (x y : ℝ), x = -1 ∧ y = 2 ∧ Real.tan θ = y / x) → 
  Real.tan (2 * θ) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l2659_265986


namespace NUMINAMATH_CALUDE_pants_pricing_l2659_265958

theorem pants_pricing (S P : ℝ) 
  (h1 : S = P + 0.25 * S)
  (h2 : 14 = 0.8 * S - P) :
  P = 210 := by sorry

end NUMINAMATH_CALUDE_pants_pricing_l2659_265958


namespace NUMINAMATH_CALUDE_line_equation_l2659_265964

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a line has equal intercepts on both axes -/
def Line.equalIntercepts (l : Line) : Prop :=
  l.a * l.c = -l.b * l.c

theorem line_equation (P : Point) (l : Line) :
  P.x = 2 ∧ P.y = 1 ∧
  P.onLine l ∧
  l.perpendicular { a := 1, b := -1, c := 1 } ∧
  l.equalIntercepts →
  (l = { a := 1, b := 1, c := -3 } ∨ l = { a := 1, b := -2, c := 0 }) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2659_265964


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l2659_265933

theorem greatest_integer_b_for_all_real_domain : ∃ (b : ℤ), 
  (∀ (c : ℤ), c > b → ∃ (x : ℝ), x^2 + c*x + 5 = 0) ∧
  (∀ (x : ℝ), x^2 + b*x + 5 ≠ 0) ∧
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l2659_265933


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2659_265981

/-- The polynomial function f(x) = x^8 + 6x^7 + 13x^6 + 256x^5 - 684x^4 -/
def f (x : ℝ) : ℝ := x^8 + 6*x^7 + 13*x^6 + 256*x^5 - 684*x^4

/-- The theorem stating that f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2659_265981


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2659_265908

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2659_265908


namespace NUMINAMATH_CALUDE_car_distance_problem_l2659_265975

/-- Represents the distance traveled by a car given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem car_distance_problem (speed_x speed_y : ℝ) (initial_time : ℝ) :
  speed_x = 35 →
  speed_y = 65 →
  initial_time = 72 / 60 →
  ∃ t : ℝ, 
    distance speed_y t = distance speed_x initial_time + distance speed_x t ∧
    distance speed_x t = 49 := by
  sorry

#check car_distance_problem

end NUMINAMATH_CALUDE_car_distance_problem_l2659_265975


namespace NUMINAMATH_CALUDE_inequality_proof_l2659_265932

theorem inequality_proof (x y : ℝ) (a : ℝ) (hx : x > 0) (hy : y > 0) :
  x^(Real.sin a)^2 * y^(Real.cos a)^2 < x + y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2659_265932


namespace NUMINAMATH_CALUDE_boris_early_theorem_l2659_265979

/-- Represents the distance between two points -/
structure Distance where
  value : ℝ
  nonneg : 0 ≤ value

/-- Represents a speed (distance per unit time) -/
structure Speed where
  value : ℝ
  pos : 0 < value

/-- Represents a time duration -/
structure Time where
  value : ℝ
  nonneg : 0 ≤ value

/-- The scenario of Anna and Boris walking towards each other -/
structure WalkingScenario where
  d : Distance  -- distance between villages A and B
  v_A : Speed   -- Anna's speed
  v_B : Speed   -- Boris's speed
  t : Time      -- time they meet when starting simultaneously

variable (scenario : WalkingScenario)

/-- The distance Anna walks in the original scenario -/
def anna_distance : ℝ := scenario.v_A.value * scenario.t.value

/-- The distance Boris walks in the original scenario -/
def boris_distance : ℝ := scenario.v_B.value * scenario.t.value

/-- Condition: Anna and Boris meet when they start simultaneously -/
axiom meet_condition : anna_distance scenario + boris_distance scenario = scenario.d.value

/-- Condition: If Anna starts 30 minutes earlier, they meet 2 km closer to village B -/
axiom anna_early_condition : 
  scenario.v_A.value * (scenario.t.value + 0.5) + scenario.v_B.value * scenario.t.value 
  = scenario.d.value - 2

/-- Theorem: If Boris starts 30 minutes earlier, they meet 2 km closer to village A -/
theorem boris_early_theorem : 
  scenario.v_A.value * scenario.t.value + scenario.v_B.value * (scenario.t.value + 0.5) 
  = scenario.d.value + 2 := by
  sorry

end NUMINAMATH_CALUDE_boris_early_theorem_l2659_265979


namespace NUMINAMATH_CALUDE_juice_problem_l2659_265915

/-- Given the number of oranges per glass and the total number of oranges,
    calculate the number of glasses of juice. -/
def glasses_of_juice (oranges_per_glass : ℕ) (total_oranges : ℕ) : ℕ :=
  total_oranges / oranges_per_glass

theorem juice_problem :
  glasses_of_juice 2 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_juice_problem_l2659_265915


namespace NUMINAMATH_CALUDE_towel_area_decrease_l2659_265944

theorem towel_area_decrease : 
  ∀ (original_length original_width : ℝ),
  original_length > 0 → original_width > 0 →
  let new_length := original_length * 0.8
  let new_width := original_width * 0.9
  let original_area := original_length * original_width
  let new_area := new_length * new_width
  (original_area - new_area) / original_area = 0.28 := by
sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l2659_265944


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_range_l2659_265943

/-- A triangle with sides a, a+2, and a+4 is obtuse if and only if 2 < a < 6 -/
theorem obtuse_triangle_side_range (a : ℝ) : 
  (∃ (x y z : ℝ), x = a ∧ y = a + 2 ∧ z = a + 4 ∧ 
   x > 0 ∧ y > 0 ∧ z > 0 ∧
   x + y > z ∧ x + z > y ∧ y + z > x ∧
   z^2 > x^2 + y^2) ↔ 
  (2 < a ∧ a < 6) :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_range_l2659_265943


namespace NUMINAMATH_CALUDE_min_value_expression_l2659_265987

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + 2*b^2 + 2/(a + 2*b)^2 ≥ 2 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀^2 + 2*b₀^2 + 2/(a₀ + 2*b₀)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2659_265987


namespace NUMINAMATH_CALUDE_problem_statement_l2659_265934

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^9 + a^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2659_265934


namespace NUMINAMATH_CALUDE_profit_increase_l2659_265962

theorem profit_increase (march_profit : ℝ) (march_profit_pos : march_profit > 0) :
  let april_profit := march_profit * 1.35
  let may_profit := april_profit * 0.8
  let june_profit := may_profit * 1.5
  (june_profit - march_profit) / march_profit = 0.62 := by
  sorry

end NUMINAMATH_CALUDE_profit_increase_l2659_265962


namespace NUMINAMATH_CALUDE_estimate_red_balls_l2659_265930

/-- Represents the number of balls in the bag -/
def total_balls : ℕ := 10

/-- Represents the total number of draws -/
def total_draws : ℕ := 1000

/-- Represents the number of times a red ball was drawn -/
def red_draws : ℕ := 200

/-- The estimated number of red balls in the bag -/
def estimated_red_balls : ℚ := (red_draws : ℚ) / total_draws * total_balls

theorem estimate_red_balls :
  estimated_red_balls = 2 := by sorry

end NUMINAMATH_CALUDE_estimate_red_balls_l2659_265930


namespace NUMINAMATH_CALUDE_circle_radius_l2659_265954

theorem circle_radius (x y : ℝ) :
  2 * x^2 + 2 * y^2 - 10 = 2 * x + 4 * y →
  ∃ (center_x center_y : ℝ),
    (x - center_x)^2 + (y - center_y)^2 = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l2659_265954


namespace NUMINAMATH_CALUDE_inequality_range_l2659_265949

theorem inequality_range (a : ℚ) : 
  a^7 < a^5 ∧ a^5 < a^3 ∧ a^3 < a ∧ a < a^2 ∧ a^2 < a^4 ∧ a^4 < a^6 → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2659_265949


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2659_265992

theorem quadratic_one_solution_sum (b : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + b * x + 5 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + b * y + 5 * y + 12 = 0 → y = x) →
  (∃ c : ℝ, 3 * c^2 + (-b) * c + 5 * c + 12 = 0 ∧ 
   ∀ z : ℝ, 3 * z^2 + (-b) * z + 5 * z + 12 = 0 → z = c) →
  b + (-b) = -10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2659_265992


namespace NUMINAMATH_CALUDE_closest_integer_to_expression_l2659_265961

theorem closest_integer_to_expression : ∃ n : ℤ, 
  n = round ((3/2 : ℚ) * (4/9 : ℚ) + (7/2 : ℚ)) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_expression_l2659_265961


namespace NUMINAMATH_CALUDE_power_of_two_divisibility_l2659_265966

theorem power_of_two_divisibility (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ r : ℕ, n.val = 2^r :=
sorry

end NUMINAMATH_CALUDE_power_of_two_divisibility_l2659_265966


namespace NUMINAMATH_CALUDE_sphere_radius_l2659_265903

theorem sphere_radius (r_A : ℝ) : 
  let r_B : ℝ := 10
  (r_A^2 / r_B^2 = 16) → r_A = 40 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_l2659_265903


namespace NUMINAMATH_CALUDE_parallelogram_area_from_complex_equations_sum_pqrs_equals_102_l2659_265939

theorem parallelogram_area_from_complex_equations : ℂ → Prop :=
  fun i =>
  i * i = -1 →
  let eq1 := fun z : ℂ => z * z = 9 + 9 * Real.sqrt 7 * i
  let eq2 := fun z : ℂ => z * z = 5 + 5 * Real.sqrt 2 * i
  let solutions := {z : ℂ | eq1 z ∨ eq2 z}
  let parallelogram_area := Real.sqrt 96 * 2 - Real.sqrt 2 * 2
  (∃ (v1 v2 v3 v4 : ℂ), v1 ∈ solutions ∧ v2 ∈ solutions ∧ v3 ∈ solutions ∧ v4 ∈ solutions ∧
    (v1 - v2).im * (v3 - v4).re - (v1 - v2).re * (v3 - v4).im = parallelogram_area)

/-- The sum of p, q, r, and s is 102 -/
theorem sum_pqrs_equals_102 : 2 + 96 + 2 + 2 = 102 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_from_complex_equations_sum_pqrs_equals_102_l2659_265939


namespace NUMINAMATH_CALUDE_total_clothes_ironed_l2659_265988

/-- The time in minutes it takes Eliza to iron a blouse -/
def blouse_time : ℕ := 15

/-- The time in minutes it takes Eliza to iron a dress -/
def dress_time : ℕ := 20

/-- The time in hours Eliza spends ironing blouses -/
def blouse_hours : ℕ := 2

/-- The time in hours Eliza spends ironing dresses -/
def dress_hours : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating the total number of pieces of clothes Eliza ironed -/
theorem total_clothes_ironed : 
  (blouse_hours * minutes_per_hour / blouse_time) + 
  (dress_hours * minutes_per_hour / dress_time) = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_clothes_ironed_l2659_265988


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2659_265937

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 2015 = 10) →
  a 2 + a 1008 + a 2014 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2659_265937


namespace NUMINAMATH_CALUDE_cosine_sine_equivalence_l2659_265972

theorem cosine_sine_equivalence (θ : ℝ) : 
  Real.cos (3 * Real.pi / 2 - θ) = Real.sin (Real.pi + θ) ∧ 
  Real.cos (3 * Real.pi / 2 - θ) = Real.cos (Real.pi / 2 + θ) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_equivalence_l2659_265972


namespace NUMINAMATH_CALUDE_counterexample_exists_l2659_265960

theorem counterexample_exists : ∃ (x y z : ℝ), 
  x > 0 ∧ y > 0 ∧ x > y ∧ z ≠ 0 ∧ |x + z| ≤ |y + z| := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2659_265960


namespace NUMINAMATH_CALUDE_even_function_m_value_l2659_265976

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- Given f(x) = x^2 + (m+2)x + 3 is an even function, prove that m = -2 -/
theorem even_function_m_value (m : ℝ) :
  IsEven (fun x => x^2 + (m+2)*x + 3) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_m_value_l2659_265976


namespace NUMINAMATH_CALUDE_sum_of_first_six_primes_mod_seventh_prime_l2659_265911

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_of_first_six_primes_mod_seventh_prime : 
  (first_six_primes.sum) % seventh_prime = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_six_primes_mod_seventh_prime_l2659_265911


namespace NUMINAMATH_CALUDE_common_solution_range_l2659_265902

theorem common_solution_range (x y : ℝ) : 
  (∃ x, x^2 + y^2 - 11 = 0 ∧ x^2 - 4*y + 7 = 0) ↔ 7/4 ≤ y ∧ y ≤ Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_common_solution_range_l2659_265902


namespace NUMINAMATH_CALUDE_expectation_of_specific_distribution_l2659_265927

/-- The expected value of a random variable with a specific probability distribution -/
theorem expectation_of_specific_distribution (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_sum : x + y + x = 1) :
  let ξ : ℝ → ℝ := fun ω => 
    if ω < x then 1
    else if ω < x + y then 2
    else 3
  2 = ∫ ω in Set.Icc 0 1, ξ ω ∂volume :=
by sorry

end NUMINAMATH_CALUDE_expectation_of_specific_distribution_l2659_265927


namespace NUMINAMATH_CALUDE_solution_form_l2659_265928

theorem solution_form (a b c d : ℝ) 
  (h1 : a + b + c = d) 
  (h2 : 1/a + 1/b + 1/c = 1/d) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  (c = -a ∧ d = b) ∨ (c = -b ∧ d = a) := by
sorry

end NUMINAMATH_CALUDE_solution_form_l2659_265928


namespace NUMINAMATH_CALUDE_marble_probability_l2659_265955

/-- Given two boxes of marbles with the following properties:
  1. The total number of marbles in both boxes is 24.
  2. The probability of drawing a black marble from each box is 28/45.
  This theorem states that the probability of drawing a white marble from each box is 2/135. -/
theorem marble_probability (box_a box_b : Finset ℕ) 
  (h_total : box_a.card + box_b.card = 24)
  (h_black_prob : (box_a.filter (λ x => x = 1)).card / box_a.card * 
                  (box_b.filter (λ x => x = 1)).card / box_b.card = 28/45) :
  (box_a.filter (λ x => x = 0)).card / box_a.card * 
  (box_b.filter (λ x => x = 0)).card / box_b.card = 2/135 :=
sorry

end NUMINAMATH_CALUDE_marble_probability_l2659_265955


namespace NUMINAMATH_CALUDE_royal_family_theorem_l2659_265946

/-- Represents the royal family -/
structure RoyalFamily where
  king_age : ℕ
  queen_age : ℕ
  num_sons : ℕ
  num_daughters : ℕ
  children_total_age : ℕ

/-- The conditions of the problem -/
def royal_family_conditions (family : RoyalFamily) : Prop :=
  family.king_age = 35 ∧
  family.queen_age = 35 ∧
  family.num_sons = 3 ∧
  family.num_daughters ≥ 1 ∧
  family.children_total_age = 35 ∧
  family.num_sons + family.num_daughters ≤ 20

/-- The theorem to be proved -/
theorem royal_family_theorem (family : RoyalFamily) 
  (h : royal_family_conditions family) :
  family.num_sons + family.num_daughters = 7 ∨
  family.num_sons + family.num_daughters = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_royal_family_theorem_l2659_265946


namespace NUMINAMATH_CALUDE_fraction_pair_sum_equality_l2659_265996

theorem fraction_pair_sum_equality (n : ℕ) (h : n > 2009) :
  ∃ (a b c d : ℕ), a ≤ n ∧ b ≤ n ∧ c ≤ n ∧ d ≤ n ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (1 : ℚ) / (n + 1 - a) + (1 : ℚ) / (n + 1 - b) =
  (1 : ℚ) / (n + 1 - c) + (1 : ℚ) / (n + 1 - d) :=
by sorry

end NUMINAMATH_CALUDE_fraction_pair_sum_equality_l2659_265996


namespace NUMINAMATH_CALUDE_cos_difference_of_zeros_l2659_265901

open Real

theorem cos_difference_of_zeros (f g : ℝ → ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, f x = sin (2 * x - π / 3)) →
  (∀ x, g x = f x - 1 / 3) →
  g x₁ = 0 →
  g x₂ = 0 →
  x₁ ≠ x₂ →
  0 ≤ x₁ ∧ x₁ ≤ π →
  0 ≤ x₂ ∧ x₂ ≤ π →
  cos (x₁ - x₂) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_of_zeros_l2659_265901


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2659_265969

universe u

def I : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}

theorem intersection_complement_equality :
  M ∩ (I \ N) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2659_265969


namespace NUMINAMATH_CALUDE_line_through_points_l2659_265925

/-- Given a line x = 6y + 5 passing through points (m, n) and (m + 2, n + p), prove that p = 1/3 -/
theorem line_through_points (m n p : ℝ) : 
  (m = 6 * n + 5) → 
  (m + 2 = 6 * (n + p) + 5) → 
  p = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_line_through_points_l2659_265925


namespace NUMINAMATH_CALUDE_cosine_sum_theorem_l2659_265904

theorem cosine_sum_theorem (x m : Real) (h : Real.cos (x - Real.pi/6) = m) :
  Real.cos x + Real.cos (x - Real.pi/3) = Real.sqrt 3 * m := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_theorem_l2659_265904


namespace NUMINAMATH_CALUDE_largest_increase_2011_2012_l2659_265965

/-- Represents the number of students participating in AMC 12 for each year from 2010 to 2016 --/
def amc_participants : Fin 7 → ℕ
  | 0 => 120  -- 2010
  | 1 => 130  -- 2011
  | 2 => 150  -- 2012
  | 3 => 155  -- 2013
  | 4 => 160  -- 2014
  | 5 => 140  -- 2015
  | 6 => 150  -- 2016

/-- Calculates the percentage increase between two consecutive years --/
def percentage_increase (year : Fin 6) : ℚ :=
  (amc_participants (year.succ) - amc_participants year : ℚ) / amc_participants year * 100

/-- Theorem stating that the percentage increase between 2011 and 2012 is the largest --/
theorem largest_increase_2011_2012 :
  ∀ year : Fin 6, percentage_increase 1 ≥ percentage_increase year :=
by sorry

#eval percentage_increase 1  -- Should output the largest percentage increase

end NUMINAMATH_CALUDE_largest_increase_2011_2012_l2659_265965


namespace NUMINAMATH_CALUDE_angle_calculation_l2659_265916

/-- Given three angles, proves that if angle 1 and angle 2 are complementary, 
    angle 2 and angle 3 are supplementary, and angle 3 equals 18°, 
    then angle 1 equals 108°. -/
theorem angle_calculation (angle1 angle2 angle3 : ℝ) : 
  angle1 + angle2 = 90 →
  angle2 + angle3 = 180 →
  angle3 = 18 →
  angle1 = 108 := by
  sorry

end NUMINAMATH_CALUDE_angle_calculation_l2659_265916


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2659_265924

/-- Given a geometric sequence {a_n} with common ratio q > 1,
    if 2a₁, (3/2)a₂, and a₃ form an arithmetic sequence,
    then S₄/a₄ = 15/8, where S₄ is the sum of the first 4 terms. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence condition
  q > 1 →  -- Common ratio > 1
  (2 * a 1 - (3/2 * a 2) = (3/2 * a 2) - a 3) →  -- Arithmetic sequence condition
  (a 1 * (1 - q^4) / (1 - q)) / (a 1 * q^3) = 15/8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2659_265924


namespace NUMINAMATH_CALUDE_factorization_equality_l2659_265971

theorem factorization_equality (x : ℝ) : 75 * x^3 - 225 * x^10 = 75 * x^3 * (1 - 3 * x^7) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2659_265971


namespace NUMINAMATH_CALUDE_opposite_of_eight_l2659_265957

theorem opposite_of_eight : 
  -(8 : ℤ) = -8 := by sorry

end NUMINAMATH_CALUDE_opposite_of_eight_l2659_265957


namespace NUMINAMATH_CALUDE_symmetry_implies_phi_value_l2659_265912

theorem symmetry_implies_phi_value (φ : Real) :
  φ ∈ Set.Icc 0 Real.pi →
  (∀ x : Real, 3 * Real.cos (x + φ) - 1 = 3 * Real.cos ((2 * Real.pi / 3 - x) + φ) - 1) →
  φ = 2 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_symmetry_implies_phi_value_l2659_265912


namespace NUMINAMATH_CALUDE_parking_cost_theorem_l2659_265978

/-- Calculates the average hourly parking cost for a given duration -/
def averageHourlyCost (baseCost : ℚ) (baseHours : ℚ) (additionalHourlyRate : ℚ) (totalHours : ℚ) : ℚ :=
  let totalCost := baseCost + (totalHours - baseHours) * additionalHourlyRate
  totalCost / totalHours

/-- Proves that the average hourly cost for 9 hours of parking is $3.03 -/
theorem parking_cost_theorem :
  let baseCost : ℚ := 15
  let baseHours : ℚ := 2
  let additionalHourlyRate : ℚ := 1.75
  let totalHours : ℚ := 9
  averageHourlyCost baseCost baseHours additionalHourlyRate totalHours = 3.03 := by
  sorry

#eval averageHourlyCost 15 2 1.75 9

end NUMINAMATH_CALUDE_parking_cost_theorem_l2659_265978


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2659_265950

/-- The number of dots on each side of the square grid -/
def grid_size : ℕ := 5

/-- The number of different rectangles that can be formed in the grid -/
def num_rectangles : ℕ := (grid_size.choose 2) * (grid_size.choose 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100 -/
theorem rectangles_in_5x5_grid : num_rectangles = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2659_265950


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2659_265983

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 17) → (∃ m : ℤ, N = 13 * m + 4) :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2659_265983


namespace NUMINAMATH_CALUDE_clover_walking_distance_l2659_265973

/-- Clover's walking problem -/
theorem clover_walking_distance 
  (total_distance : ℝ) 
  (num_days : ℕ) 
  (walks_per_day : ℕ) :
  total_distance = 90 →
  num_days = 30 →
  walks_per_day = 2 →
  (total_distance / num_days) / walks_per_day = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_clover_walking_distance_l2659_265973


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2659_265906

theorem polynomial_evaluation (x : ℤ) (h : x = -2) : x^3 - x^2 + x - 1 = -15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2659_265906


namespace NUMINAMATH_CALUDE_kabadi_kho_kho_players_l2659_265931

theorem kabadi_kho_kho_players (total : ℕ) (kabadi : ℕ) (kho_kho_only : ℕ) 
  (h_total : total = 50)
  (h_kabadi : kabadi = 10)
  (h_kho_kho_only : kho_kho_only = 40) :
  total = kabadi + kho_kho_only - 0 :=
by sorry

end NUMINAMATH_CALUDE_kabadi_kho_kho_players_l2659_265931


namespace NUMINAMATH_CALUDE_firefighter_pay_theorem_l2659_265980

/-- Represents the firefighter's hourly pay in dollars -/
def hourly_pay : ℝ := 30

/-- Represents the number of work hours per week -/
def work_hours_per_week : ℝ := 48

/-- Represents the number of weeks in a month -/
def weeks_per_month : ℝ := 4

/-- Represents the monthly food expense in dollars -/
def food_expense : ℝ := 500

/-- Represents the monthly tax expense in dollars -/
def tax_expense : ℝ := 1000

/-- Represents the remaining money after expenses in dollars -/
def remaining_money : ℝ := 2340

theorem firefighter_pay_theorem :
  let monthly_pay := hourly_pay * work_hours_per_week * weeks_per_month
  let rent_expense := (1 / 3) * monthly_pay
  monthly_pay - rent_expense - food_expense - tax_expense = remaining_money :=
by sorry

end NUMINAMATH_CALUDE_firefighter_pay_theorem_l2659_265980


namespace NUMINAMATH_CALUDE_northwest_molded_handle_cost_l2659_265968

/-- Northwest Molded's handle production problem -/
theorem northwest_molded_handle_cost 
  (fixed_cost : ℝ) 
  (selling_price : ℝ) 
  (break_even_quantity : ℕ) 
  (h1 : fixed_cost = 7640)
  (h2 : selling_price = 4.60)
  (h3 : break_even_quantity = 1910) :
  ∃ (cost_per_handle : ℝ), 
    cost_per_handle = 0.60 ∧ 
    (selling_price * break_even_quantity : ℝ) = fixed_cost + (break_even_quantity : ℝ) * cost_per_handle :=
by sorry

end NUMINAMATH_CALUDE_northwest_molded_handle_cost_l2659_265968


namespace NUMINAMATH_CALUDE_newton_county_population_l2659_265900

theorem newton_county_population (num_cities : ℕ) (lower_bound upper_bound : ℝ) :
  num_cities = 20 →
  lower_bound = 4500 →
  upper_bound = 5000 →
  let avg_population := (lower_bound + upper_bound) / 2
  num_cities * avg_population = 95000 := by
  sorry

end NUMINAMATH_CALUDE_newton_county_population_l2659_265900


namespace NUMINAMATH_CALUDE_part1_part2_l2659_265905

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - 2*a^2

/-- Part 1: The range of a when f(x) > -9 always holds -/
theorem part1 (a : ℝ) : (∀ x, f a x > -9) → a ∈ Set.Ioo (-2) 2 := by sorry

/-- Part 2: Solving the inequality f(x) > 0 with respect to x -/
theorem part2 (a : ℝ) (x : ℝ) :
  (a > 0 → (f a x > 0 ↔ x ∈ Set.Iio (-a) ∪ Set.Ioi (2*a))) ∧
  (a = 0 → (f a x > 0 ↔ x ∈ Set.Iio 0 ∪ Set.Ioi 0)) ∧
  (a < 0 → (f a x > 0 ↔ x ∈ Set.Iio (2*a) ∪ Set.Ioi (-a))) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2659_265905


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_of_a_for_two_fixed_points_l2659_265922

/-- Definition of the function f(x) -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + (b - 1)

/-- A point x₀ is a fixed point of f if f(x₀) = x₀ -/
def is_fixed_point (a b x₀ : ℝ) : Prop := f a b x₀ = x₀

theorem fixed_points_for_specific_values :
  is_fixed_point 1 (-2) 3 ∧ is_fixed_point 1 (-2) (-1) :=
sorry

theorem range_of_a_for_two_fixed_points :
  (∀ b : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_fixed_point a b x₁ ∧ is_fixed_point a b x₂) →
  (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_of_a_for_two_fixed_points_l2659_265922


namespace NUMINAMATH_CALUDE_three_color_theorem_l2659_265959

/-- A complete graph with n vertices -/
def CompleteGraph (n : ℕ) := { v : Fin n // True }

/-- An edge in a complete graph connects any two distinct vertices -/
def Edge (n : ℕ) := { e : CompleteGraph n × CompleteGraph n // e.1 ≠ e.2 }

/-- A coloring assignment for vertices -/
def Coloring (n : ℕ) := CompleteGraph n → Fin 3

/-- A valid coloring ensures no two adjacent vertices have the same color -/
def ValidColoring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (e : Edge n), c e.1.1 ≠ c e.1.2

theorem three_color_theorem (n : ℕ) (h : n ≥ 2) :
  ∃ (c : Coloring n), ValidColoring n c :=
sorry

end NUMINAMATH_CALUDE_three_color_theorem_l2659_265959


namespace NUMINAMATH_CALUDE_express_c_in_terms_of_a_and_b_l2659_265967

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 3)
def c : ℝ × ℝ := (4, 1)

theorem express_c_in_terms_of_a_and_b :
  c = 2 • a - b := by sorry

end NUMINAMATH_CALUDE_express_c_in_terms_of_a_and_b_l2659_265967


namespace NUMINAMATH_CALUDE_hyperbola_condition_l2659_265989

theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (m^2 + 1) - y^2 / (m^2 - 3) = 1) → 
  (m < -Real.sqrt 3 ∨ m > Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l2659_265989


namespace NUMINAMATH_CALUDE_equation_with_geometric_progression_roots_l2659_265952

theorem equation_with_geometric_progression_roots : ∃ (x₁ x₂ x₃ x₄ : ℝ) (q : ℝ),
  (x₁ ≠ x₂) ∧ (x₁ ≠ x₃) ∧ (x₁ ≠ x₄) ∧ (x₂ ≠ x₃) ∧ (x₂ ≠ x₄) ∧ (x₃ ≠ x₄) ∧
  (q ≠ 1) ∧ (q > 0) ∧
  (x₂ = q * x₁) ∧ (x₃ = q * x₂) ∧ (x₄ = q * x₃) ∧
  (16 * x₁^4 - 170 * x₁^3 + 357 * x₁^2 - 170 * x₁ + 16 = 0) ∧
  (16 * x₂^4 - 170 * x₂^3 + 357 * x₂^2 - 170 * x₂ + 16 = 0) ∧
  (16 * x₃^4 - 170 * x₃^3 + 357 * x₃^2 - 170 * x₃ + 16 = 0) ∧
  (16 * x₄^4 - 170 * x₄^3 + 357 * x₄^2 - 170 * x₄ + 16 = 0) := by
sorry

end NUMINAMATH_CALUDE_equation_with_geometric_progression_roots_l2659_265952


namespace NUMINAMATH_CALUDE_cubic_function_property_l2659_265970

/-- Given a cubic function f(x) = ax³ - bx + 1 where a and b are real numbers,
    if f(-2) = -1, then f(2) = 3 -/
theorem cubic_function_property (a b : ℝ) :
  (fun x => a * x^3 - b * x + 1) (-2) = -1 →
  (fun x => a * x^3 - b * x + 1) 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2659_265970


namespace NUMINAMATH_CALUDE_eden_bears_count_eden_final_bears_count_l2659_265945

theorem eden_bears_count (initial_bears : ℕ) (favorite_bears : ℕ) (sisters : ℕ) (eden_initial_bears : ℕ) : ℕ :=
  let remaining_bears := initial_bears - favorite_bears
  let bears_per_sister := remaining_bears / sisters
  eden_initial_bears + bears_per_sister

theorem eden_final_bears_count :
  eden_bears_count 20 8 3 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_eden_bears_count_eden_final_bears_count_l2659_265945


namespace NUMINAMATH_CALUDE_simplify_expression_l2659_265999

theorem simplify_expression (x : ℝ) : 1 - (2 - (1 + (2 - (1 - x)))) = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2659_265999


namespace NUMINAMATH_CALUDE_impossible_time_reduction_l2659_265963

theorem impossible_time_reduction (initial_speed : ℝ) (time_reduction : ℝ) : 
  initial_speed = 60 → time_reduction = 1 → ¬ ∃ (new_speed : ℝ), 
    new_speed > 0 ∧ (1 / new_speed) * 60 = (1 / initial_speed) * 60 - time_reduction :=
by
  sorry

end NUMINAMATH_CALUDE_impossible_time_reduction_l2659_265963


namespace NUMINAMATH_CALUDE_sets_theorem_l2659_265948

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 5}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

-- State the theorem
theorem sets_theorem :
  (∀ x : ℝ, x ∈ A ∩ B (-4) ↔ 1/2 ≤ x ∧ x < 2) ∧
  (∀ x : ℝ, x ∈ A ∪ B (-4) ↔ -2 < x ∧ x ≤ 3) ∧
  (∀ a : ℝ, (B a ∩ (Aᶜ : Set ℝ) = B a) ↔ a ≥ -1/4) :=
sorry

end NUMINAMATH_CALUDE_sets_theorem_l2659_265948


namespace NUMINAMATH_CALUDE_perimeter_ratio_is_one_l2659_265920

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents the original paper --/
def original_paper : Rectangle := { width := 8, height := 12 }

/-- Represents one of the rectangles after folding and cutting --/
def folded_cut_rectangle : Rectangle := { width := 4, height := 6 }

/-- Theorem stating that the ratio of perimeters is 1 --/
theorem perimeter_ratio_is_one : 
  perimeter folded_cut_rectangle / perimeter folded_cut_rectangle = 1 := by sorry

end NUMINAMATH_CALUDE_perimeter_ratio_is_one_l2659_265920


namespace NUMINAMATH_CALUDE_product_purchase_discount_l2659_265917

/-- Proves that if a product is sold for $439.99999999999966 with a 10% profit, 
    and if buying it for x% less and selling at 30% profit would yield $28 more, 
    then x = 10%. -/
theorem product_purchase_discount (x : Real) : 
  (1.1 * (439.99999999999966 / 1.1) = 439.99999999999966) →
  (1.3 * (1 - x/100) * (439.99999999999966 / 1.1) = 439.99999999999966 + 28) →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_purchase_discount_l2659_265917


namespace NUMINAMATH_CALUDE_wendy_pastries_left_l2659_265953

/-- The number of pastries Wendy had left after the bake sale -/
def pastries_left (cupcakes cookies sold : ℕ) : ℕ :=
  cupcakes + cookies - sold

/-- Theorem stating that Wendy had 24 pastries left after the bake sale -/
theorem wendy_pastries_left : pastries_left 4 29 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_wendy_pastries_left_l2659_265953


namespace NUMINAMATH_CALUDE_cos_two_pi_seventh_inequality_l2659_265991

theorem cos_two_pi_seventh_inequality (a : ℝ) (h : a = Real.cos (2 * Real.pi / 7)) : 
  2^(a - 1/2) < 2 * a := by sorry

end NUMINAMATH_CALUDE_cos_two_pi_seventh_inequality_l2659_265991


namespace NUMINAMATH_CALUDE_ellipse_problem_l2659_265942

-- Define the ellipses and points
def C₁ (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1
def C₂ (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
theorem ellipse_problem (a b : ℝ) (A B H P M N : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (∃ x y, C₂ a b x y ∧ x^2 = 5 ∧ y = 0) ∧
  (∃ x₁ y₁ x₂ y₂, C₂ a b x₁ y₁ ∧ C₂ a b x₂ y₂ ∧ y₂ - y₁ = x₂ - x₁) ∧
  H = (2, -1) ∧
  C₂ a b P.1 P.2 ∧
  C₁ M.1 M.2 ∧
  C₁ N.1 N.2 ∧
  P.1 = M.1 + 2 * N.1 ∧
  P.2 = M.2 + 2 * N.2 →
  (a^2 = 10 ∧ b^2 = 5) ∧
  (M.2 / M.1 * N.2 / N.1 = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_problem_l2659_265942


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2659_265909

theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (total_length : ℝ) 
  (h1 : train_length = 130)
  (h2 : train_speed_kmh = 45)
  (h3 : total_length = 245) :
  let bridge_length := total_length - train_length
  let total_distance := total_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let crossing_time := total_distance / train_speed_ms
  crossing_time = 19.6 := by sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2659_265909


namespace NUMINAMATH_CALUDE_victor_deck_count_l2659_265977

theorem victor_deck_count (cost_per_deck : ℕ) (friend_deck_count : ℕ) (total_spent : ℕ) : ℕ :=
  let victor_deck_count := (total_spent - friend_deck_count * cost_per_deck) / cost_per_deck
  have h1 : cost_per_deck = 8 := by sorry
  have h2 : friend_deck_count = 2 := by sorry
  have h3 : total_spent = 64 := by sorry
  have h4 : victor_deck_count = 6 := by sorry
  victor_deck_count

#check victor_deck_count

end NUMINAMATH_CALUDE_victor_deck_count_l2659_265977


namespace NUMINAMATH_CALUDE_ajay_ride_distance_l2659_265993

/-- Ajay's riding speed in km/hour -/
def riding_speed : ℝ := 50

/-- Time taken for the ride in hours -/
def ride_time : ℝ := 18

/-- The distance Ajay can ride in the given time -/
def ride_distance : ℝ := riding_speed * ride_time

theorem ajay_ride_distance : ride_distance = 900 := by
  sorry

end NUMINAMATH_CALUDE_ajay_ride_distance_l2659_265993


namespace NUMINAMATH_CALUDE_diagonal_length_is_sqrt_457_l2659_265995

/-- An isosceles trapezoid with specific side lengths -/
structure IsoscelesTrapezoid :=
  (A B C D : ℝ × ℝ)
  (ab_length : dist A B = 24)
  (bc_length : dist B C = 13)
  (cd_length : dist C D = 12)
  (da_length : dist D A = 13)
  (isosceles : dist B C = dist D A)

/-- The length of the diagonal AC in the isosceles trapezoid -/
def diagonal_length (t : IsoscelesTrapezoid) : ℝ :=
  dist t.A t.C

/-- Theorem stating that the diagonal length is √457 -/
theorem diagonal_length_is_sqrt_457 (t : IsoscelesTrapezoid) :
  diagonal_length t = Real.sqrt 457 := by
  sorry


end NUMINAMATH_CALUDE_diagonal_length_is_sqrt_457_l2659_265995


namespace NUMINAMATH_CALUDE_sarah_won_thirty_games_l2659_265947

/-- Represents the outcome of Sarah's tic-tac-toe games -/
structure TicTacToeOutcome where
  total_games : ℕ
  tied_games : ℕ
  net_loss : ℤ
  win_reward : ℤ
  tie_reward : ℤ
  loss_penalty : ℤ

/-- Calculates the number of games Sarah won -/
def games_won (outcome : TicTacToeOutcome) : ℕ :=
  sorry

/-- Theorem stating that Sarah won 30 games given the specified conditions -/
theorem sarah_won_thirty_games (outcome : TicTacToeOutcome) 
  (h1 : outcome.total_games = 100)
  (h2 : outcome.tied_games = 40)
  (h3 : outcome.net_loss = 30)
  (h4 : outcome.win_reward = 1)
  (h5 : outcome.tie_reward = 0)
  (h6 : outcome.loss_penalty = 2) :
  games_won outcome = 30 :=
sorry

end NUMINAMATH_CALUDE_sarah_won_thirty_games_l2659_265947


namespace NUMINAMATH_CALUDE_jade_transactions_l2659_265998

theorem jade_transactions 
  (mabel_transactions : ℕ)
  (anthony_transactions : ℕ)
  (cal_transactions : ℕ)
  (jade_transactions : ℕ)
  (h1 : mabel_transactions = 90)
  (h2 : anthony_transactions = mabel_transactions + mabel_transactions / 10)
  (h3 : cal_transactions = anthony_transactions * 2 / 3)
  (h4 : jade_transactions = cal_transactions + 19) :
  jade_transactions = 85 := by
  sorry

end NUMINAMATH_CALUDE_jade_transactions_l2659_265998


namespace NUMINAMATH_CALUDE_product_negative_from_positive_sum_negative_quotient_l2659_265994

theorem product_negative_from_positive_sum_negative_quotient
  (a b : ℝ) (h_sum : a + b > 0) (h_quotient : a / b < 0) :
  a * b < 0 :=
by sorry

end NUMINAMATH_CALUDE_product_negative_from_positive_sum_negative_quotient_l2659_265994


namespace NUMINAMATH_CALUDE_perpendicular_planes_l2659_265914

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (not_parallel : Line → Line → Prop)
variable (line_not_parallel_plane : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (m n : Line) (α β γ : Plane)
  (different_lines : m ≠ n)
  (different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h1 : perpendicular m α)
  (h2 : not_parallel m n)
  (h3 : line_not_parallel_plane n β) :
  planes_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l2659_265914


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2659_265936

theorem arithmetic_computation : 2 + 5 * 3^2 - 4 + 6 * 2 / 3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2659_265936


namespace NUMINAMATH_CALUDE_equation_solution_l2659_265921

theorem equation_solution :
  ∃ x : ℚ, (2 * x / (x - 2) - 2 = 1 / (x * (x - 2))) ∧ (x = 1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2659_265921


namespace NUMINAMATH_CALUDE_tangent_line_problem_l2659_265918

theorem tangent_line_problem (a : ℝ) :
  (∃ (m : ℝ), 
    (∀ x y : ℝ, y = x^3 → (y - 0 = m * (x - 1) → (∀ t : ℝ, t ≠ x → t^3 > m * (t - 1)))) ∧
    (∀ x y : ℝ, y = a * x^2 + (15/4) * x - 9 → (y - 0 = m * (x - 1) → 
      (∀ t : ℝ, t ≠ x → a * t^2 + (15/4) * t - 9 ≠ m * (t - 1))))) →
  a = -1 ∨ a = -25/64 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l2659_265918


namespace NUMINAMATH_CALUDE_find_b_l2659_265984

theorem find_b (a b c : ℕ) 
  (h1 : 1 < a) (h2 : a < b) (h3 : b < c)
  (h4 : a + b + c = 111)
  (h5 : b^2 = a * c) :
  b = 36 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l2659_265984


namespace NUMINAMATH_CALUDE_inequality_proof_l2659_265913

theorem inequality_proof (x y : ℝ) (h : x ≠ y) : x^4 + y^4 > x^3*y + x*y^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2659_265913


namespace NUMINAMATH_CALUDE_min_value_expression_l2659_265929

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 5) / Real.sqrt (x - 4) ≥ 6 ∧ ∃ y : ℝ, y > 4 ∧ (y + 5) / Real.sqrt (y - 4) = 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2659_265929


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l2659_265982

/-- The parabola y = x^2 - 4 intersects the y-axis at the point (0, -4) -/
theorem parabola_y_axis_intersection :
  let f : ℝ → ℝ := fun x ↦ x^2 - 4
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = f p.1 ∧ p = (0, -4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l2659_265982


namespace NUMINAMATH_CALUDE_vessel_combination_theorem_l2659_265910

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℚ
  denominator : ℚ

/-- Represents a vessel containing a mixture of milk and water -/
structure Vessel where
  volume : ℚ
  milkWaterRatio : Ratio

/-- Combines the contents of two vessels -/
def combineVessels (v1 v2 : Vessel) : Ratio :=
  let totalMilk := v1.volume * v1.milkWaterRatio.numerator / (v1.milkWaterRatio.numerator + v1.milkWaterRatio.denominator) +
                   v2.volume * v2.milkWaterRatio.numerator / (v2.milkWaterRatio.numerator + v2.milkWaterRatio.denominator)
  let totalWater := v1.volume * v1.milkWaterRatio.denominator / (v1.milkWaterRatio.numerator + v1.milkWaterRatio.denominator) +
                    v2.volume * v2.milkWaterRatio.denominator / (v2.milkWaterRatio.numerator + v2.milkWaterRatio.denominator)
  { numerator := totalMilk, denominator := totalWater }

theorem vessel_combination_theorem :
  let v1 : Vessel := { volume := 3, milkWaterRatio := { numerator := 1, denominator := 2 } }
  let v2 : Vessel := { volume := 5, milkWaterRatio := { numerator := 3, denominator := 2 } }
  let combinedRatio := combineVessels v1 v2
  combinedRatio.numerator = combinedRatio.denominator :=
by
  sorry

end NUMINAMATH_CALUDE_vessel_combination_theorem_l2659_265910


namespace NUMINAMATH_CALUDE_unique_solution_l2659_265919

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem unique_solution :
  ∀ k : ℕ, (factorial (k / 2) * (k / 4) = 2016 + k^2) ↔ k = 12 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2659_265919


namespace NUMINAMATH_CALUDE_park_fencing_cost_l2659_265985

/-- The cost of fencing a rectangular park with sides in the ratio 3:2 and an area of 3750 sq m, at 80 paise per meter -/
theorem park_fencing_cost : 
  ∀ (length width : ℝ),
  length / width = 3 / 2 →
  length * width = 3750 →
  (2 * (length + width)) * (80 / 100) = 200 := by
sorry

end NUMINAMATH_CALUDE_park_fencing_cost_l2659_265985


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2659_265940

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ, are_parallel (1, m) (m, 4) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2659_265940


namespace NUMINAMATH_CALUDE_function_inequality_l2659_265926

def is_even_on (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, -a < x ∧ x < a → f x = f (-x)

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y < b → (f x ≤ f y ∨ f y ≤ f x)

theorem function_inequality (f : ℝ → ℝ) 
    (h1 : is_even_on f 6)
    (h2 : monotonic_on f 0 6)
    (h3 : f (-2) < f 1) :
  f 5 < f (-3) ∧ f (-3) < f (-1) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2659_265926


namespace NUMINAMATH_CALUDE_fraction_equality_implies_values_l2659_265923

theorem fraction_equality_implies_values (A B : ℚ) :
  (∀ x : ℚ, x ≠ 5 ∧ x ≠ -2 → A / (x - 5) + B / (x + 2) = (5 * x - 4) / (x^2 - 3*x - 10)) →
  A = 3 ∧ B = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_values_l2659_265923


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_chord_l2659_265951

/-- The line that intersects the unit circle -/
def intersecting_line (x y : ℝ) : Prop := x - y + 1 = 0

/-- The unit circle -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The perpendicular bisector of the chord -/
def perpendicular_bisector (x y : ℝ) : Prop := x + y = 0

/-- Theorem: The perpendicular bisector of the chord formed by the intersection
    of the line x - y + 1 = 0 and the unit circle x^2 + y^2 = 1 
    has the equation x + y = 0 -/
theorem perpendicular_bisector_of_chord :
  ∀ (x y : ℝ), 
  intersecting_line x y → unit_circle x y →
  perpendicular_bisector x y :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_chord_l2659_265951


namespace NUMINAMATH_CALUDE_initial_odometer_reading_l2659_265974

/-- Calculates the initial odometer reading before a trip -/
theorem initial_odometer_reading
  (odometer_at_lunch : ℝ)
  (distance_traveled : ℝ)
  (h1 : odometer_at_lunch = 372)
  (h2 : distance_traveled = 159.7) :
  odometer_at_lunch - distance_traveled = 212.3 := by
sorry

end NUMINAMATH_CALUDE_initial_odometer_reading_l2659_265974


namespace NUMINAMATH_CALUDE_fraction_ordering_l2659_265941

theorem fraction_ordering : 
  (21 : ℚ) / 17 < (18 : ℚ) / 13 ∧ (18 : ℚ) / 13 < (14 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2659_265941


namespace NUMINAMATH_CALUDE_snack_slices_theorem_l2659_265997

/-- Represents the household bread consumption scenario -/
structure HouseholdBread where
  members : ℕ
  breakfast_slices_per_member : ℕ
  slices_per_loaf : ℕ
  loaves_consumed : ℕ
  days_lasted : ℕ

/-- Calculate the number of slices each member consumes for snacks daily -/
def snack_slices_per_member_per_day (hb : HouseholdBread) : ℕ :=
  let total_slices := hb.loaves_consumed * hb.slices_per_loaf
  let breakfast_slices := hb.members * hb.breakfast_slices_per_member * hb.days_lasted
  let snack_slices := total_slices - breakfast_slices
  snack_slices / (hb.members * hb.days_lasted)

/-- Theorem stating that each member consumes 2 slices of bread for snacks daily -/
theorem snack_slices_theorem (hb : HouseholdBread) 
  (h1 : hb.members = 4)
  (h2 : hb.breakfast_slices_per_member = 3)
  (h3 : hb.slices_per_loaf = 12)
  (h4 : hb.loaves_consumed = 5)
  (h5 : hb.days_lasted = 3) :
  snack_slices_per_member_per_day hb = 2 := by
  sorry


end NUMINAMATH_CALUDE_snack_slices_theorem_l2659_265997


namespace NUMINAMATH_CALUDE_line_points_k_value_l2659_265990

/-- 
Given two points (m, n) and (m + 2, n + k) on the line x = 2y + 5,
prove that k = 0.
-/
theorem line_points_k_value (m n k : ℝ) : 
  (m = 2*n + 5) → 
  (m + 2 = 2*(n + k) + 5) → 
  k = 0 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l2659_265990


namespace NUMINAMATH_CALUDE_geometric_sequence_cars_below_threshold_l2659_265935

/- Define the sequence of ordinary cars -/
def a : ℕ → ℝ
  | 0 => 300  -- Initial value for 2020
  | n + 1 => 0.9 * a n + 8

/- Define the transformed sequence -/
def b (n : ℕ) : ℝ := a n - 80

/- Theorem statement -/
theorem geometric_sequence : ∀ n : ℕ, b (n + 1) = 0.9 * b n := by
  sorry

/- Additional theorem to show the year when cars are less than 1.5 million -/
theorem cars_below_threshold (n : ℕ) : a n < 150 → n ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_cars_below_threshold_l2659_265935


namespace NUMINAMATH_CALUDE_baseball_team_grouping_l2659_265907

theorem baseball_team_grouping (new_players returning_players num_groups : ℕ) 
  (h1 : new_players = 4)
  (h2 : returning_players = 6)
  (h3 : num_groups = 2) :
  (new_players + returning_players) / num_groups = 5 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_grouping_l2659_265907


namespace NUMINAMATH_CALUDE_initial_birds_count_l2659_265938

theorem initial_birds_count (B : ℕ) : 
  (B + 4 - 3 + 6 = 12) → B = 5 := by
sorry

end NUMINAMATH_CALUDE_initial_birds_count_l2659_265938
