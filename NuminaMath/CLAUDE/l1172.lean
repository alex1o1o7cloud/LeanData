import Mathlib

namespace NUMINAMATH_CALUDE_min_value_expression_l1172_117293

theorem min_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 9)) / x ≥ 6 / (2 * Real.sqrt 3 + Real.sqrt 6) ∧
  (∃ x₀ > 0, (x₀^2 + 3 - Real.sqrt (x₀^4 + 9)) / x₀ = 6 / (2 * Real.sqrt 3 + Real.sqrt 6)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1172_117293


namespace NUMINAMATH_CALUDE_correct_calculation_l1172_117250

theorem correct_calculation (x : ℤ) (h : x - 6 = 51) : 6 * x = 342 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1172_117250


namespace NUMINAMATH_CALUDE_points_difference_l1172_117208

theorem points_difference (zach_points ben_points : ℕ) 
  (h1 : zach_points = 42) 
  (h2 : ben_points = 21) : 
  zach_points - ben_points = 21 := by
sorry

end NUMINAMATH_CALUDE_points_difference_l1172_117208


namespace NUMINAMATH_CALUDE_constant_term_proof_l1172_117273

/-- The constant term in the expansion of (√x + 1/(3x))^10 -/
def constant_term : ℕ := 210

/-- The index of the term with the maximum coefficient -/
def max_coeff_index : ℕ := 6

theorem constant_term_proof (h : max_coeff_index = 6) : constant_term = 210 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_proof_l1172_117273


namespace NUMINAMATH_CALUDE_expected_remainder_mod_64_l1172_117285

/-- The expected value of (a + 2b + 4c + 8d + 16e + 32f) mod 64, where a, b, c, d, e, f 
    are independently and uniformly randomly selected integers from {1,2,...,100} -/
theorem expected_remainder_mod_64 : 
  let S := Finset.range 100
  let M (a b c d e f : ℕ) := a + 2*b + 4*c + 8*d + 16*e + 32*f
  (S.sum (λ a => S.sum (λ b => S.sum (λ c => S.sum (λ d => S.sum (λ e => 
    S.sum (λ f => (M a b c d e f) % 64))))))) / S.card^6 = 63/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_remainder_mod_64_l1172_117285


namespace NUMINAMATH_CALUDE_hash_example_l1172_117275

def hash (a b c d : ℝ) : ℝ := b^2 - 4*a*c + d^2

theorem hash_example : hash 2 3 1 4 = 17 := by sorry

end NUMINAMATH_CALUDE_hash_example_l1172_117275


namespace NUMINAMATH_CALUDE_last_digit_n_power_9999_minus_5555_l1172_117288

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_n_power_9999_minus_5555 (n : ℕ) : 
  last_digit (n^9999 - n^5555) = 0 :=
sorry

end NUMINAMATH_CALUDE_last_digit_n_power_9999_minus_5555_l1172_117288


namespace NUMINAMATH_CALUDE_probability_two_red_marbles_l1172_117249

/-- The probability of drawing two red marbles without replacement from a jar containing
    2 red marbles, 3 green marbles, and 10 white marbles is 1/105. -/
theorem probability_two_red_marbles :
  let red_marbles : ℕ := 2
  let green_marbles : ℕ := 3
  let white_marbles : ℕ := 10
  let total_marbles : ℕ := red_marbles + green_marbles + white_marbles
  (red_marbles : ℚ) / total_marbles * (red_marbles - 1) / (total_marbles - 1) = 1 / 105 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_marbles_l1172_117249


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l1172_117219

theorem simplify_radical_expression :
  Real.sqrt (13 + Real.sqrt 48) - Real.sqrt (5 - (2 * Real.sqrt 3 + 1)) + 2 * Real.sqrt (3 + (Real.sqrt 3 - 1)) = Real.sqrt 6 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l1172_117219


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l1172_117228

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_wrt_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_wrt_x_axis (a - 1, 5) (2, b - 1) →
  (a + b) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l1172_117228


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1172_117236

def A : Set ℕ := {0, 1, 2, 3, 4, 5}
def B : Set ℕ := {x | x^2 < 10}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1172_117236


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l1172_117259

theorem quadratic_root_implies_m (m : ℝ) : 
  (3^2 : ℝ) - m*3 - 6 = 0 → m = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l1172_117259


namespace NUMINAMATH_CALUDE_sum_of_coefficients_cubic_expansion_l1172_117278

theorem sum_of_coefficients_cubic_expansion :
  ∃ (a b c d e : ℝ), 
    (∀ x, 27 * x^3 + 64 = (a*x + b) * (c*x^2 + d*x + e)) ∧
    (a + b + c + d + e = 20) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_cubic_expansion_l1172_117278


namespace NUMINAMATH_CALUDE_zhaoqing_population_l1172_117234

theorem zhaoqing_population (total_population : ℝ) 
  (h1 : 3.06 = 0.8 * total_population - 0.18) 
  (h2 : 3.06 = agricultural_population) : 
  total_population = 4.05 := by
sorry

end NUMINAMATH_CALUDE_zhaoqing_population_l1172_117234


namespace NUMINAMATH_CALUDE_volunteer_event_arrangements_l1172_117290

/-- The number of ways to arrange volunteers for a 5-day event -/
def volunteer_arrangements (total_days : ℕ) (consecutive_days : ℕ) (total_people : ℕ) : ℕ :=
  (total_days - consecutive_days + 1) * (Nat.factorial (total_people - 1))

/-- Theorem: The number of arrangements for the volunteer event is 24 -/
theorem volunteer_event_arrangements :
  volunteer_arrangements 5 2 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_event_arrangements_l1172_117290


namespace NUMINAMATH_CALUDE_sequence_product_l1172_117274

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- The main theorem -/
theorem sequence_product (a b : ℕ → ℝ) :
  (∀ n, a n ≠ 0) →
  arithmetic_sequence a →
  2 * a 3 - (a 7)^2 + 2 * a 11 = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 6 * b 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l1172_117274


namespace NUMINAMATH_CALUDE_fixed_amount_more_economical_l1172_117279

theorem fixed_amount_more_economical (p₁ p₂ : ℝ) (h₁ : p₁ > 0) (h₂ : p₂ > 0) :
  2 / (1 / p₁ + 1 / p₂) ≤ (p₁ + p₂) / 2 := by
  sorry

#check fixed_amount_more_economical

end NUMINAMATH_CALUDE_fixed_amount_more_economical_l1172_117279


namespace NUMINAMATH_CALUDE_frog_edge_probability_l1172_117297

/-- Represents a position on the 4x4 grid -/
inductive Position
| Center : Position
| Edge : Position

/-- Represents the number of hops -/
def MaxHops : ℕ := 3

/-- The probability of reaching an edge from the center in n hops -/
def probability_reach_edge (n : ℕ) : ℚ :=
  sorry

/-- The 4x4 grid with wrapping and movement rules -/
structure Grid :=
  (size : ℕ := 4)
  (wrap : Bool := true)
  (diagonal_moves : Bool := false)

/-- Theorem: The probability of reaching an edge within 3 hops is 13/16 -/
theorem frog_edge_probability (g : Grid) : 
  probability_reach_edge MaxHops = 13/16 :=
sorry

end NUMINAMATH_CALUDE_frog_edge_probability_l1172_117297


namespace NUMINAMATH_CALUDE_constant_phi_is_cone_l1172_117262

/-- Spherical coordinates -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- A cone in 3D space -/
def Cone : Set SphericalCoord := sorry

/-- The shape described by φ = c in spherical coordinates -/
def ConstantPhiShape (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

/-- Theorem stating that the shape described by φ = c is a cone -/
theorem constant_phi_is_cone (c : ℝ) :
  ConstantPhiShape c = Cone := by sorry

end NUMINAMATH_CALUDE_constant_phi_is_cone_l1172_117262


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l1172_117223

theorem chess_game_draw_probability 
  (p_jian_win : ℝ) 
  (p_gu_not_win : ℝ) 
  (h1 : p_jian_win = 0.4) 
  (h2 : p_gu_not_win = 0.6) : 
  p_gu_not_win - p_jian_win = 0.2 := by
sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l1172_117223


namespace NUMINAMATH_CALUDE_sqrt_two_divided_by_sqrt_two_minus_one_l1172_117280

theorem sqrt_two_divided_by_sqrt_two_minus_one :
  Real.sqrt 2 / (Real.sqrt 2 - 1) = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_divided_by_sqrt_two_minus_one_l1172_117280


namespace NUMINAMATH_CALUDE_factorization_difference_l1172_117229

theorem factorization_difference (a b : ℤ) : 
  (∀ y : ℝ, 3 * y^2 - y - 18 = (3 * y + a) * (y + b)) → 
  a - b = -11 := by
sorry

end NUMINAMATH_CALUDE_factorization_difference_l1172_117229


namespace NUMINAMATH_CALUDE_function_extremum_l1172_117241

/-- The function f(x) with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem function_extremum (a b : ℝ) :
  f a b 1 = 10 ∧ f_deriv a b 1 = 0 →
  ∀ x, f a b x = x^3 + 4*x^2 - 11*x + 16 :=
by sorry

end NUMINAMATH_CALUDE_function_extremum_l1172_117241


namespace NUMINAMATH_CALUDE_weeks_to_save_shirt_l1172_117216

/-- Calculates the minimum number of whole weeks needed to save for a shirt -/
def min_weeks_to_save (shirt_cost : ℚ) (initial_savings : ℚ) (weekly_savings : ℚ) : ℕ :=
  Nat.ceil ((shirt_cost - initial_savings) / weekly_savings)

/-- Theorem stating that 34 weeks are needed to save for the shirt under given conditions -/
theorem weeks_to_save_shirt : min_weeks_to_save 15 5 0.3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_weeks_to_save_shirt_l1172_117216


namespace NUMINAMATH_CALUDE_quadratic_prime_roots_l1172_117220

theorem quadratic_prime_roots (k : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ 
   p + q = 99 ∧ p * q = k ∧
   ∀ x : ℝ, x^2 - 99*x + k = 0 ↔ (x = p ∨ x = q)) →
  k = 194 :=
sorry

end NUMINAMATH_CALUDE_quadratic_prime_roots_l1172_117220


namespace NUMINAMATH_CALUDE_term_without_x_in_special_expansion_l1172_117205

/-- Given a binomial expansion of (x³ + 1/x²)^n where n is such that only
    the coefficient of the sixth term is maximum, the term without x is 210 -/
theorem term_without_x_in_special_expansion :
  ∃ n : ℕ,
    (∀ k : ℕ, k ≠ 5 → Nat.choose n k ≤ Nat.choose n 5) ∧
    (∃ r : ℕ, Nat.choose n r = 210 ∧ 3 * n = 5 * r) :=
sorry

end NUMINAMATH_CALUDE_term_without_x_in_special_expansion_l1172_117205


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1172_117206

theorem simplify_trig_expression :
  (Real.sqrt (1 - 2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180))) /
  (Real.cos (40 * π / 180) - Real.sqrt (1 - Real.sin (50 * π / 180) ^ 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1172_117206


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1172_117272

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = 2 - Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1172_117272


namespace NUMINAMATH_CALUDE_tangent_slope_determines_point_l1172_117296

/-- Given a curve y = 2x^2 + 4x, prove that if the slope of the tangent line
    at point P is 16, then the coordinates of P are (3, 30). -/
theorem tangent_slope_determines_point :
  ∀ x y : ℝ,
  (y = 2 * x^2 + 4 * x) →  -- Curve equation
  ((4 * x + 4) = 16) →     -- Slope of tangent line is 16
  (x = 3 ∧ y = 30)         -- Coordinates of point P
  := by sorry

end NUMINAMATH_CALUDE_tangent_slope_determines_point_l1172_117296


namespace NUMINAMATH_CALUDE_rectangular_box_area_volume_relation_l1172_117221

/-- A rectangular box with dimensions x, y, and z -/
structure RectangularBox where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Properties of the rectangular box -/
def RectangularBox.properties (box : RectangularBox) : Prop :=
  let top_area := box.x * box.y
  let side_area := box.y * box.z
  let volume := box.x * box.y * box.z
  (side_area * volume ^ 2 = box.z ^ 3 * volume)

/-- Theorem: The product of the side area and square of volume equals z³V -/
theorem rectangular_box_area_volume_relation (box : RectangularBox) :
  box.properties :=
by
  sorry

#check rectangular_box_area_volume_relation

end NUMINAMATH_CALUDE_rectangular_box_area_volume_relation_l1172_117221


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_15_25_40_75_l1172_117289

theorem greatest_four_digit_divisible_by_15_25_40_75 : ∃ n : ℕ,
  n ≤ 9999 ∧
  n ≥ 1000 ∧
  n % 15 = 0 ∧
  n % 25 = 0 ∧
  n % 40 = 0 ∧
  n % 75 = 0 ∧
  ∀ m : ℕ, m ≤ 9999 ∧ m ≥ 1000 ∧ m % 15 = 0 ∧ m % 25 = 0 ∧ m % 40 = 0 ∧ m % 75 = 0 → m ≤ n :=
by
  -- Proof goes here
  sorry

#eval 9600 -- Expected output: 9600

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_15_25_40_75_l1172_117289


namespace NUMINAMATH_CALUDE_line_points_k_value_l1172_117227

theorem line_points_k_value (m n k : ℝ) : 
  (m = 2*n + 5) →                   -- First point (m, n) satisfies the line equation
  (m + 5 = 2*(n + k) + 5) →         -- Second point (m + 5, n + k) satisfies the line equation
  k = 5/2 := by                     -- Conclusion: k = 2.5
sorry


end NUMINAMATH_CALUDE_line_points_k_value_l1172_117227


namespace NUMINAMATH_CALUDE_bus_miss_time_l1172_117266

theorem bus_miss_time (usual_time : ℝ) (h : usual_time = 12) :
  let slower_time := (5 / 4) * usual_time
  slower_time - usual_time = 3 :=
by sorry

end NUMINAMATH_CALUDE_bus_miss_time_l1172_117266


namespace NUMINAMATH_CALUDE_octopus_puzzle_l1172_117231

structure Octopus where
  color : String
  legs : Nat
  statement : Bool

def isLying (o : Octopus) : Bool :=
  (o.legs = 7 ∧ ¬o.statement) ∨ (o.legs = 8 ∧ o.statement)

def totalLegs (os : List Octopus) : Nat :=
  os.foldl (fun acc o => acc + o.legs) 0

theorem octopus_puzzle :
  ∃ (green blue red : Octopus),
    [green, blue, red].all (fun o => o.legs = 7 ∨ o.legs = 8) ∧
    isLying green ∧
    ¬isLying blue ∧
    isLying red ∧
    green.statement = (totalLegs [green, blue, red] = 21) ∧
    blue.statement = ¬green.statement ∧
    red.statement = (¬green.statement ∧ ¬blue.statement) ∧
    green.legs = 7 ∧
    blue.legs = 8 ∧
    red.legs = 7 :=
  sorry

#check octopus_puzzle

end NUMINAMATH_CALUDE_octopus_puzzle_l1172_117231


namespace NUMINAMATH_CALUDE_existence_of_six_numbers_l1172_117267

theorem existence_of_six_numbers : ∃ (a b c d e f : ℕ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  (a + b + c + d + e + f : ℚ) / ((1 : ℚ)/a + 1/b + 1/c + 1/d + 1/e + 1/f) = 2012 :=
sorry

end NUMINAMATH_CALUDE_existence_of_six_numbers_l1172_117267


namespace NUMINAMATH_CALUDE_lucas_change_l1172_117246

/-- Represents the shopping scenario and calculates the change --/
def calculate_change (initial_amount : ℝ) 
  (avocado_costs : List ℝ) 
  (water_cost : ℝ) 
  (water_quantity : ℕ) 
  (apple_cost : ℝ) 
  (apple_quantity : ℕ) : ℝ :=
  let total_cost := (avocado_costs.sum + water_cost * water_quantity + apple_cost * apple_quantity)
  initial_amount - total_cost

/-- Theorem stating that Lucas brings home $6.75 in change --/
theorem lucas_change : 
  calculate_change 20 [1.50, 2.25, 3.00] 1.75 2 0.75 4 = 6.75 := by
  sorry

#eval calculate_change 20 [1.50, 2.25, 3.00] 1.75 2 0.75 4

end NUMINAMATH_CALUDE_lucas_change_l1172_117246


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1172_117210

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 4 + 2 * Complex.I) :
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1172_117210


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l1172_117212

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_overs : ℕ
  first_run_rate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_overs
  let runs_scored := game.first_run_rate * game.first_overs
  let runs_needed := game.target - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.total_overs = 50)
  (h2 : game.first_overs = 10)
  (h3 : game.first_run_rate = 4.8)
  (h4 : game.target = 282) :
  required_run_rate game = 5.85 := by
  sorry

#eval required_run_rate { total_overs := 50, first_overs := 10, first_run_rate := 4.8, target := 282 }

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l1172_117212


namespace NUMINAMATH_CALUDE_team_total_score_l1172_117277

/-- Represents a basketball player with their score -/
structure Player where
  name : String
  score : ℕ

/-- The school basketball team -/
def team : List Player := [
  { name := "Daniel", score := 7 },
  { name := "Ramon", score := 8 },
  { name := "Ian", score := 2 },
  { name := "Bernardo", score := 11 },
  { name := "Tiago", score := 6 },
  { name := "Pedro", score := 12 },
  { name := "Ed", score := 1 },
  { name := "André", score := 7 }
]

/-- The total score of the team is the sum of individual player scores -/
def totalScore (team : List Player) : ℕ :=
  team.map (·.score) |>.sum

/-- Theorem: The total score of the team is 54 -/
theorem team_total_score : totalScore team = 54 := by
  sorry

end NUMINAMATH_CALUDE_team_total_score_l1172_117277


namespace NUMINAMATH_CALUDE_factor_expression_l1172_117217

theorem factor_expression (x : ℝ) : 270 * x^3 - 90 * x^2 + 18 * x = 18 * x * (15 * x^2 - 5 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1172_117217


namespace NUMINAMATH_CALUDE_shaded_area_circles_l1172_117284

theorem shaded_area_circles (R : ℝ) (h : R = 10) : 
  let large_circle_area := π * R^2
  let small_circle_radius := R / 2
  let small_circle_area := π * small_circle_radius^2
  let shaded_area := large_circle_area - 2 * small_circle_area
  shaded_area = 50 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_circles_l1172_117284


namespace NUMINAMATH_CALUDE_max_value_inequality_l1172_117253

theorem max_value_inequality (k : ℝ) : 
  (∀ x : ℝ, |x^2 - 4*x + k| + |x - 3| ≤ 5) ∧ 
  (∃ x : ℝ, x = 3 ∧ |x^2 - 4*x + k| + |x - 3| = 5) ∧
  (∀ x : ℝ, x > 3 → |x^2 - 4*x + k| + |x - 3| > 5) →
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1172_117253


namespace NUMINAMATH_CALUDE_function_non_positive_on_interval_l1172_117222

theorem function_non_positive_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc 0 1, a^2 * x - 2*a + 1 ≤ 0) ↔ a ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_function_non_positive_on_interval_l1172_117222


namespace NUMINAMATH_CALUDE_ralph_received_eight_cards_l1172_117265

/-- The number of cards Ralph's father gave him -/
def cards_from_father (initial_cards final_cards : ℕ) : ℕ :=
  final_cards - initial_cards

/-- Proof that Ralph's father gave him 8 cards -/
theorem ralph_received_eight_cards :
  let initial_cards : ℕ := 4
  let final_cards : ℕ := 12
  cards_from_father initial_cards final_cards = 8 := by
  sorry

end NUMINAMATH_CALUDE_ralph_received_eight_cards_l1172_117265


namespace NUMINAMATH_CALUDE_present_age_of_B_present_age_of_B_exists_l1172_117281

/-- Proves that given the conditions in the problem, B's current age is 37 years. -/
theorem present_age_of_B : ℕ → ℕ → Prop :=
  fun a b =>
    (a + 10 = 2 * (b - 10)) →  -- A will be twice as old as B was 10 years ago, in 10 years
    (a = b + 7) →              -- A is now 7 years older than B
    (b = 37)                   -- B's current age is 37

/-- The theorem holds for some values of a and b. -/
theorem present_age_of_B_exists : ∃ a b, present_age_of_B a b :=
  sorry

end NUMINAMATH_CALUDE_present_age_of_B_present_age_of_B_exists_l1172_117281


namespace NUMINAMATH_CALUDE_prob_sum_three_l1172_117225

/-- Represents a ball with a number label -/
inductive Ball : Type
| one : Ball
| two : Ball

/-- Represents the result of two draws -/
structure TwoDraws where
  first : Ball
  second : Ball

/-- The set of all possible outcomes from two draws -/
def allOutcomes : Finset TwoDraws :=
  sorry

/-- The set of favorable outcomes (sum of drawn numbers is 3) -/
def favorableOutcomes : Finset TwoDraws :=
  sorry

/-- The probability of an event is the number of favorable outcomes
    divided by the total number of outcomes -/
def probability (event : Finset TwoDraws) : ℚ :=
  (event.card : ℚ) / (allOutcomes.card : ℚ)

/-- The main theorem: the probability of drawing two balls with sum 3 is 1/2 -/
theorem prob_sum_three : probability favorableOutcomes = 1/2 :=
  sorry

end NUMINAMATH_CALUDE_prob_sum_three_l1172_117225


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l1172_117230

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 4609 : ℤ) ≡ 2104 [ZMOD 12] ∧
  ∀ (y : ℕ), y > 0 → (y + 4609 : ℤ) ≡ 2104 [ZMOD 12] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l1172_117230


namespace NUMINAMATH_CALUDE_perfect_squares_divisibility_l1172_117239

theorem perfect_squares_divisibility (a b : ℕ+) 
  (h : ∃ S : Set (ℕ+ × ℕ+), Set.Infinite S ∧ 
    ∀ (m n : ℕ+), (m, n) ∈ S → 
      ∃ (k l : ℕ+), (m : ℕ)^2 + (a : ℕ) * (n : ℕ) + (b : ℕ) = (k : ℕ)^2 ∧
                    (n : ℕ)^2 + (a : ℕ) * (m : ℕ) + (b : ℕ) = (l : ℕ)^2) : 
  (a : ℕ) ∣ 2 * (b : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_divisibility_l1172_117239


namespace NUMINAMATH_CALUDE_consecutive_integer_products_sum_l1172_117207

theorem consecutive_integer_products_sum : 
  ∃ (a b c x y z w : ℕ), 
    (b = a + 1) ∧ 
    (c = b + 1) ∧ 
    (y = x + 1) ∧ 
    (z = y + 1) ∧ 
    (w = z + 1) ∧ 
    (a * b * c = 924) ∧ 
    (x * y * z * w = 924) ∧ 
    (a + b + c + x + y + z + w = 75) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integer_products_sum_l1172_117207


namespace NUMINAMATH_CALUDE_all_digits_satisfy_inequality_l1172_117226

theorem all_digits_satisfy_inequality :
  ∀ A : ℕ, A ≤ 9 → 27 * 10 * A + 2708 - 1203 > 1022 := by
  sorry

end NUMINAMATH_CALUDE_all_digits_satisfy_inequality_l1172_117226


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1172_117201

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 2 + Real.sqrt 10 ∧ x₂ = 2 - Real.sqrt 10) ∧ 
  (x₁^2 - 4*x₁ - 6 = 0 ∧ x₂^2 - 4*x₂ - 6 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1172_117201


namespace NUMINAMATH_CALUDE_magic_square_sum_l1172_117218

/-- Represents a 3x3 magic square with some known and unknown values -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  sum : ℕ
  sum_eq_row1 : sum = 30 + e + 18
  sum_eq_row2 : sum = 15 + c + d
  sum_eq_row3 : sum = a + 27 + b
  sum_eq_col1 : sum = 30 + 15 + a
  sum_eq_col2 : sum = e + c + 27
  sum_eq_col3 : sum = 18 + d + b
  sum_eq_diag1 : sum = 30 + c + b
  sum_eq_diag2 : sum = a + c + 18

theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 47 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l1172_117218


namespace NUMINAMATH_CALUDE_fifth_subject_score_l1172_117287

theorem fifth_subject_score (s1 s2 s3 s4 : ℕ) (avg : ℚ) :
  s1 = 50 →
  s2 = 60 →
  s3 = 70 →
  s4 = 80 →
  avg = 68 →
  (s1 + s2 + s3 + s4 : ℚ) / 4 + 80 / 5 = avg :=
by sorry

end NUMINAMATH_CALUDE_fifth_subject_score_l1172_117287


namespace NUMINAMATH_CALUDE_total_jumps_calculation_total_jumps_is_4411_l1172_117263

/-- Calculate the total number of jumps made by Rupert and Ronald throughout the week. -/
theorem total_jumps_calculation : ℕ := by
  -- Define the number of jumps for Ronald on Monday
  let ronald_monday : ℕ := 157

  -- Define Rupert's jumps on Monday relative to Ronald's
  let rupert_monday : ℕ := ronald_monday + 86

  -- Define Ronald's jumps on Tuesday
  let ronald_tuesday : ℕ := 193

  -- Define Rupert's jumps on Tuesday
  let rupert_tuesday : ℕ := rupert_monday - 35

  -- Define the constant decrease rate from Thursday to Sunday
  let daily_decrease : ℕ := 20

  -- Calculate total jumps
  let total_jumps : ℕ := 
    -- Monday
    ronald_monday + rupert_monday +
    -- Tuesday
    ronald_tuesday + rupert_tuesday +
    -- Wednesday (doubled from Tuesday)
    2 * ronald_tuesday + 2 * rupert_tuesday +
    -- Thursday to Sunday (4 days with constant decrease)
    (2 * ronald_tuesday - daily_decrease) + (2 * rupert_tuesday - daily_decrease) +
    (2 * ronald_tuesday - 2 * daily_decrease) + (2 * rupert_tuesday - 2 * daily_decrease) +
    (2 * ronald_tuesday - 3 * daily_decrease) + (2 * rupert_tuesday - 3 * daily_decrease) +
    (2 * ronald_tuesday - 4 * daily_decrease) + (2 * rupert_tuesday - 4 * daily_decrease)

  exact total_jumps

/-- Prove that the total number of jumps is 4411 -/
theorem total_jumps_is_4411 : total_jumps_calculation = 4411 := by
  sorry

end NUMINAMATH_CALUDE_total_jumps_calculation_total_jumps_is_4411_l1172_117263


namespace NUMINAMATH_CALUDE_factors_of_60_l1172_117202

-- Define the number of positive factors function
def num_positive_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

-- State the theorem
theorem factors_of_60 : num_positive_factors 60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_60_l1172_117202


namespace NUMINAMATH_CALUDE_sum_of_powers_l1172_117261

theorem sum_of_powers : -1^2008 + (-1)^2009 + 1^2010 - 1^2011 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1172_117261


namespace NUMINAMATH_CALUDE_first_discount_percentage_l1172_117255

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 32 →
  final_price = 18 →
  second_discount = 0.25 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (1 - first_discount) * (1 - second_discount) ∧
    first_discount = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l1172_117255


namespace NUMINAMATH_CALUDE_functional_equation_implies_linear_scaling_l1172_117213

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x^3 + y^3) = (x + y) * (f x^2 - f x * f y + f y^2)

/-- The main theorem to be proved -/
theorem functional_equation_implies_linear_scaling
  (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x, f (1996 * x) = 1996 * f x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_linear_scaling_l1172_117213


namespace NUMINAMATH_CALUDE_min_sphere_surface_area_l1172_117258

/-- Given a rectangular parallelepiped with volume 12, height 4, and all vertices on the surface of a sphere,
    prove that the minimum surface area of the sphere is 22π. -/
theorem min_sphere_surface_area (a b c : ℝ) (h_volume : a * b * c = 12) (h_height : c = 4)
  (h_on_sphere : ∃ (r : ℝ), a^2 + b^2 + c^2 = 4 * r^2) :
  ∃ (S : ℝ), S = 22 * Real.pi ∧ ∀ (r : ℝ), (a^2 + b^2 + c^2 = 4 * r^2) → 4 * Real.pi * r^2 ≥ S := by
  sorry

end NUMINAMATH_CALUDE_min_sphere_surface_area_l1172_117258


namespace NUMINAMATH_CALUDE_candles_remaining_l1172_117224

/-- Calculates the number of candles remaining after three people use them according to specific rules. -/
theorem candles_remaining (total : ℕ) (alyssa_fraction : ℚ) (chelsea_fraction : ℚ) (bianca_fraction : ℚ) : 
  total = 60 ∧ 
  alyssa_fraction = 1/2 ∧ 
  chelsea_fraction = 7/10 ∧ 
  bianca_fraction = 4/5 →
  ↑total - (alyssa_fraction * ↑total + 
    chelsea_fraction * (↑total - alyssa_fraction * ↑total) + 
    ⌊bianca_fraction * (↑total - alyssa_fraction * ↑total - chelsea_fraction * (↑total - alyssa_fraction * ↑total))⌋) = 2 := by
  sorry

#check candles_remaining

end NUMINAMATH_CALUDE_candles_remaining_l1172_117224


namespace NUMINAMATH_CALUDE_total_profit_is_4650_l1172_117254

/-- Given the capitals of three individuals P, Q, and R, and the profit share of R,
    calculate the total profit. -/
def calculate_total_profit (Cp Cq Cr R_share : ℚ) : ℚ :=
  let total_ratio := (10 : ℚ) / 4 + 10 / 6 + 1
  R_share * total_ratio / (1 : ℚ)

/-- Theorem stating that under given conditions, the total profit is 4650. -/
theorem total_profit_is_4650 (Cp Cq Cr : ℚ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) 
    (h3 : calculate_total_profit Cp Cq Cr 900 = 4650) : 
  calculate_total_profit Cp Cq Cr 900 = 4650 := by
  sorry

#eval calculate_total_profit 1 1 1 900

end NUMINAMATH_CALUDE_total_profit_is_4650_l1172_117254


namespace NUMINAMATH_CALUDE_tv_price_reduction_l1172_117270

theorem tv_price_reduction (x : ℝ) : 
  (1 - x / 100) * 1.80 = 1.44000000000000014 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_reduction_l1172_117270


namespace NUMINAMATH_CALUDE_line_quadrants_m_range_l1172_117299

theorem line_quadrants_m_range (m : ℝ) : 
  (∀ x y : ℝ, y = (m - 2) * x + m → 
    (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) → 
  0 < m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_line_quadrants_m_range_l1172_117299


namespace NUMINAMATH_CALUDE_frequency_of_fifth_group_l1172_117286

theorem frequency_of_fifth_group 
  (total_students : ℕ) 
  (group1 group2 group3 group4 : ℕ) 
  (h1 : total_students = 40)
  (h2 : group1 = 12)
  (h3 : group2 = 10)
  (h4 : group3 = 6)
  (h5 : group4 = 8) :
  total_students - (group1 + group2 + group3 + group4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_frequency_of_fifth_group_l1172_117286


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l1172_117247

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start ∧ k < start + count → ¬(is_prime k)

theorem smallest_prime_after_six_nonprimes :
  ∃ n : ℕ, 
    (consecutive_nonprimes (n - 6) 6) ∧ 
    (is_prime n) ∧ 
    (∀ m : ℕ, m < n → ¬(consecutive_nonprimes (m - 6) 6 ∧ is_prime m)) ∧
    n = 53 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l1172_117247


namespace NUMINAMATH_CALUDE_complement_of_union_l1172_117268

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {1, 3, 5}

theorem complement_of_union :
  (U \ (A ∪ B)) = {6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1172_117268


namespace NUMINAMATH_CALUDE_eulers_theorem_l1172_117252

/-- A convex polyhedron with f faces, p vertices, and a edges -/
structure ConvexPolyhedron where
  f : ℕ  -- number of faces
  p : ℕ  -- number of vertices
  a : ℕ  -- number of edges

/-- Euler's theorem for convex polyhedra -/
theorem eulers_theorem (poly : ConvexPolyhedron) : poly.f + poly.p - poly.a = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_theorem_l1172_117252


namespace NUMINAMATH_CALUDE_man_double_son_age_l1172_117291

/-- Represents the number of years until a man's age is twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  2

/-- Theorem stating that it takes 2 years for the man's age to be twice his son's age -/
theorem man_double_son_age (son_age : ℕ) (age_difference : ℕ) 
  (h1 : son_age = 24) 
  (h2 : age_difference = 26) : 
  years_until_double_age son_age age_difference = 2 := by
  sorry

#check man_double_son_age

end NUMINAMATH_CALUDE_man_double_son_age_l1172_117291


namespace NUMINAMATH_CALUDE_pi_minus_three_zero_plus_half_inverse_equals_three_l1172_117238

theorem pi_minus_three_zero_plus_half_inverse_equals_three :
  (Real.pi - 3) ^ (0 : ℕ) + (1 / 2) ^ (-1 : ℤ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_pi_minus_three_zero_plus_half_inverse_equals_three_l1172_117238


namespace NUMINAMATH_CALUDE_min_value_theorem_l1172_117276

open Real

theorem min_value_theorem (x y : ℝ) (hx : x > 1) (hy : y > 1) (h : x * y - 2 * x - y + 1 = 0) :
  ∃ (m : ℝ), m = 15 ∧ ∀ (a b : ℝ), a > 1 → b > 1 → a * b - 2 * a - b + 1 = 0 → (3/2) * a^2 + b^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1172_117276


namespace NUMINAMATH_CALUDE_sum_of_digits_of_special_palindrome_l1172_117245

/-- A function that checks if a natural number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) ∧ (n / 10 % 10 = n / 10 % 10)

/-- A function that calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  (n / 100) + (n / 10 % 10) + (n % 10)

/-- Theorem stating that if x is a three-digit palindrome and x + 50 is also a three-digit palindrome,
    then the sum of digits of x is 19 -/
theorem sum_of_digits_of_special_palindrome (x : ℕ) :
  isThreeDigitPalindrome x ∧ isThreeDigitPalindrome (x + 50) → sumOfDigits x = 19 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_special_palindrome_l1172_117245


namespace NUMINAMATH_CALUDE_crabs_count_proof_l1172_117271

/-- The number of crabs on the first day -/
def crabs_day1 : ℕ := 72

/-- The number of oysters on the first day -/
def oysters_day1 : ℕ := 50

/-- The total count of oysters and crabs over two days -/
def total_count : ℕ := 195

theorem crabs_count_proof :
  crabs_day1 = 72 ∧
  oysters_day1 = 50 ∧
  (oysters_day1 + crabs_day1 + oysters_day1 / 2 + crabs_day1 * 2 / 3 = total_count) :=
sorry

end NUMINAMATH_CALUDE_crabs_count_proof_l1172_117271


namespace NUMINAMATH_CALUDE_total_markers_l1172_117240

theorem total_markers :
  let red_markers : ℕ := 41
  let blue_markers : ℕ := 64
  let green_markers : ℕ := 35
  let black_markers : ℕ := 78
  let yellow_markers : ℕ := 102
  red_markers + blue_markers + green_markers + black_markers + yellow_markers = 320 :=
by
  sorry

end NUMINAMATH_CALUDE_total_markers_l1172_117240


namespace NUMINAMATH_CALUDE_not_right_triangle_not_triangle_l1172_117233

theorem not_right_triangle (a b c : ℝ) (ha : a = 1) (hb : b = 1) (hc : c = 2) :
  ¬(a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) :=
by sorry

theorem not_triangle (a b c : ℝ) (ha : a = 1) (hb : b = 1) (hc : c = 2) :
  ¬(a + b > c ∧ b + c > a ∧ c + a > b) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_not_triangle_l1172_117233


namespace NUMINAMATH_CALUDE_equality_of_expressions_l1172_117298

theorem equality_of_expressions (x : ℝ) (hx : x > 0) :
  x^(x+1) + x^(x+1) = 2*x^(x+1) ∧
  x^(x+1) + x^(x+1) ≠ x^(2*x+2) ∧
  x^(x+1) + x^(x+1) ≠ (2*x)^(x+1) ∧
  x^(x+1) + x^(x+1) ≠ (2*x)^(2*x+2) :=
by sorry

end NUMINAMATH_CALUDE_equality_of_expressions_l1172_117298


namespace NUMINAMATH_CALUDE_plane_equation_correct_l1172_117257

/-- A plane in 3D Cartesian coordinates with intercepts a, b, and c on the x, y, and z axes respectively. -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  h₁ : a ≠ 0
  h₂ : b ≠ 0
  h₃ : c ≠ 0

/-- The equation of a plane in 3D Cartesian coordinates with given intercepts. -/
def planeEquation (p : Plane3D) (x y z : ℝ) : Prop :=
  x / p.a + y / p.b + z / p.c = 1

/-- Theorem stating that the equation x/a + y/b + z/c = 1 represents a plane
    with intercepts a, b, and c on the x, y, and z axes respectively. -/
theorem plane_equation_correct (p : Plane3D) :
  ∀ x y z : ℝ, planeEquation p x y z ↔ 
    (x = p.a ∧ y = 0 ∧ z = 0) ∨
    (x = 0 ∧ y = p.b ∧ z = 0) ∨
    (x = 0 ∧ y = 0 ∧ z = p.c) :=
  sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l1172_117257


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1172_117248

/-- Transformation of a quadratic equation under a linear substitution -/
theorem quadratic_transformation (A B C D E F α β γ β' γ' : ℝ) :
  let Δ := A * C - B^2
  let x := λ x' y' : ℝ => α * x' + β * y' + γ
  let y := λ x' y' : ℝ => x' + β' * y' + γ'
  let original_eq := λ x y : ℝ => A * x^2 + 2 * B * x * y + C * y^2 + 2 * D * x + 2 * E * y + F
  ∃ a b : ℝ, 
    (Δ > 0 → ∀ x' y' : ℝ, original_eq (x x' y') (y x' y') = 0 ↔ x'^2 / a^2 + y'^2 / b^2 = 1) ∧
    (Δ < 0 → ∀ x' y' : ℝ, original_eq (x x' y') (y x' y') = 0 ↔ x'^2 / a^2 - y'^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1172_117248


namespace NUMINAMATH_CALUDE_smallest_n_for_gcd_lcm_condition_l1172_117243

theorem smallest_n_for_gcd_lcm_condition : ∃ (n : ℕ), 
  (∃ (a b : ℕ), Nat.gcd a b = 999 ∧ Nat.lcm a b = n.factorial) ∧ 
  (∀ (m : ℕ), m < n → ¬∃ (a b : ℕ), Nat.gcd a b = 999 ∧ Nat.lcm a b = m.factorial) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_gcd_lcm_condition_l1172_117243


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1172_117292

/-- Given an arithmetic sequence where the first term is 10/11 and the fifteenth term is 8/9,
    the eighth term is 89/99. -/
theorem arithmetic_sequence_eighth_term 
  (a : ℕ → ℚ)  -- a is the sequence
  (h1 : a 1 = 10 / 11)  -- first term is 10/11
  (h15 : a 15 = 8 / 9)  -- fifteenth term is 8/9
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- arithmetic sequence condition
  : a 8 = 89 / 99 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1172_117292


namespace NUMINAMATH_CALUDE_interview_probability_l1172_117242

theorem interview_probability (total_students : ℕ) (french_students : ℕ) (spanish_students : ℕ) (german_students : ℕ)
  (h1 : total_students = 30)
  (h2 : french_students = 22)
  (h3 : spanish_students = 25)
  (h4 : german_students = 5)
  (h5 : french_students ≤ spanish_students)
  (h6 : spanish_students ≤ total_students)
  (h7 : german_students ≤ total_students) :
  let non_french_spanish : ℕ := total_students - spanish_students
  let total_combinations : ℕ := total_students.choose 2
  let non_informative_combinations : ℕ := (non_french_spanish + (spanish_students - french_students)).choose 2
  (1 : ℚ) - (non_informative_combinations : ℚ) / (total_combinations : ℚ) = 407 / 435 := by
    sorry

end NUMINAMATH_CALUDE_interview_probability_l1172_117242


namespace NUMINAMATH_CALUDE_mary_initial_money_l1172_117295

/-- The amount of money Mary had before buying the pie -/
def initial_money : ℕ := sorry

/-- The cost of the pie -/
def pie_cost : ℕ := 6

/-- The amount of money Mary has after buying the pie -/
def remaining_money : ℕ := 52

theorem mary_initial_money : 
  initial_money = remaining_money + pie_cost := by sorry

end NUMINAMATH_CALUDE_mary_initial_money_l1172_117295


namespace NUMINAMATH_CALUDE_no_sum_of_squared_digits_greater_than_2008_l1172_117203

def sum_of_squared_digits (n : ℕ) : ℕ :=
  let digits := Nat.digits 10 n
  List.sum (List.map (λ d => d * d) digits)

theorem no_sum_of_squared_digits_greater_than_2008 :
  ∀ n : ℕ, n > 2008 → n ≠ sum_of_squared_digits n :=
sorry

end NUMINAMATH_CALUDE_no_sum_of_squared_digits_greater_than_2008_l1172_117203


namespace NUMINAMATH_CALUDE_triangle_angle_C_l1172_117214

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi

-- Define the conditions of the problem
def problem_conditions (t : Triangle) : Prop :=
  5 * Real.sin t.A + 3 * Real.cos t.B = 7 ∧
  3 * Real.sin t.B + 5 * Real.cos t.A = 3

-- Theorem statement
theorem triangle_angle_C (t : Triangle) :
  problem_conditions t → Real.sin t.C = 4/5 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l1172_117214


namespace NUMINAMATH_CALUDE_periodic_function_value_l1172_117209

/-- Given a function f(x) = a * sin(π * x + α) + b * cos(π * x + β) + 4,
    where a, b, α, β are non-zero real numbers, and f(2012) = 6,
    prove that f(2013) = 2 -/
theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  (f 2012 = 6) → (f 2013 = 2) := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l1172_117209


namespace NUMINAMATH_CALUDE_open_box_volume_is_5760_l1172_117244

/-- Calculate the volume of an open box formed by cutting squares from a rectangular sheet. -/
def openBoxVolume (sheetLength sheetWidth cutSize : ℝ) : ℝ :=
  (sheetLength - 2 * cutSize) * (sheetWidth - 2 * cutSize) * cutSize

/-- Theorem: The volume of the open box is 5760 m³ -/
theorem open_box_volume_is_5760 :
  openBoxVolume 52 36 8 = 5760 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_is_5760_l1172_117244


namespace NUMINAMATH_CALUDE_interior_angles_sum_increase_l1172_117211

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: If the sum of interior angles of a convex polygon with n sides is 2340°,
    then the sum of interior angles of a convex polygon with (n + 4) sides is 3060°. -/
theorem interior_angles_sum_increase (n : ℕ) :
  sum_interior_angles n = 2340 → sum_interior_angles (n + 4) = 3060 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_increase_l1172_117211


namespace NUMINAMATH_CALUDE_polyhedron_formula_l1172_117269

/-- Represents a convex polyhedron with specific face configuration -/
structure Polyhedron where
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  hexagons : ℕ
  T : ℕ
  P : ℕ
  H : ℕ
  faces_sum : faces = triangles + pentagons + hexagons
  faces_types : faces = 32 ∧ triangles = 10 ∧ pentagons = 8 ∧ hexagons = 14

/-- Calculates the number of edges in the polyhedron -/
def edges (poly : Polyhedron) : ℕ :=
  (3 * poly.triangles + 5 * poly.pentagons + 6 * poly.hexagons) / 2

/-- Calculates the number of vertices in the polyhedron using Euler's formula -/
def vertices (poly : Polyhedron) : ℕ :=
  edges poly - poly.faces + 2

/-- Theorem stating that for the given polyhedron, 100P + 10T + V = 249 -/
theorem polyhedron_formula (poly : Polyhedron) : 100 * poly.P + 10 * poly.T + vertices poly = 249 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_formula_l1172_117269


namespace NUMINAMATH_CALUDE_hill_climbing_speeds_l1172_117237

theorem hill_climbing_speeds (distance : ℝ) (ascending_time descending_time : ℝ) 
  (h1 : ascending_time = 3)
  (h2 : distance / ascending_time = 2.5)
  (h3 : (2 * distance) / (ascending_time + descending_time) = 3) :
  distance / descending_time = 3.75 := by sorry

end NUMINAMATH_CALUDE_hill_climbing_speeds_l1172_117237


namespace NUMINAMATH_CALUDE_selection_theorem_l1172_117235

def male_teachers : ℕ := 5
def female_teachers : ℕ := 3
def total_selection : ℕ := 3

def select_with_both_genders (m f s : ℕ) : ℕ :=
  Nat.choose m 2 * Nat.choose f 1 + Nat.choose m 1 * Nat.choose f 2

theorem selection_theorem :
  select_with_both_genders male_teachers female_teachers total_selection = 45 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l1172_117235


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_billion_l1172_117251

/-- Sum of digits function for a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := sorry

/-- The main theorem: sum of digits of all numbers from 1 to 1 billion -/
theorem sum_of_digits_up_to_billion :
  sumOfDigitsUpTo 1000000000 = 40500000001 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_billion_l1172_117251


namespace NUMINAMATH_CALUDE_orange_count_in_second_group_l1172_117282

def apple_cost : ℚ := 21/100

theorem orange_count_in_second_group 
  (first_group : 6 * apple_cost + 3 * orange_cost = 177/100)
  (second_group : 2 * apple_cost + x * orange_cost = 127/100)
  (orange_cost : ℚ) (x : ℚ) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_in_second_group_l1172_117282


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l1172_117283

theorem chord_length_concentric_circles (R r : ℝ) (h : R^2 - r^2 = 15) :
  ∃ c : ℝ, c = 2 * Real.sqrt 15 ∧ c^2 / 4 + r^2 = R^2 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l1172_117283


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1172_117294

def M (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def P (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, (M a) ∩ (P a) = {-3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1172_117294


namespace NUMINAMATH_CALUDE_sin_pi_half_equals_one_l1172_117215

theorem sin_pi_half_equals_one : 
  let f : ℝ → ℝ := fun x ↦ Real.sin (x / 2 + π / 4)
  f (π / 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_sin_pi_half_equals_one_l1172_117215


namespace NUMINAMATH_CALUDE_notebook_increase_correct_l1172_117260

/-- Calculates the increase in Jimin's notebook count -/
def notebook_increase (initial : ℕ) (father_bought : ℕ) (mother_bought : ℕ) : ℕ :=
  father_bought + mother_bought

theorem notebook_increase_correct (initial : ℕ) (father_bought : ℕ) (mother_bought : ℕ) :
  notebook_increase initial father_bought mother_bought = father_bought + mother_bought :=
by sorry

end NUMINAMATH_CALUDE_notebook_increase_correct_l1172_117260


namespace NUMINAMATH_CALUDE_both_not_land_l1172_117232

-- Define the propositions
variable (p q : Prop)

-- p represents "A lands within the designated area"
-- q represents "B lands within the designated area"

-- Theorem: "Both trainees did not land within the designated area" 
-- is equivalent to (¬p) ∧ (¬q)
theorem both_not_land (p q : Prop) : 
  (¬p ∧ ¬q) ↔ ¬(p ∨ q) :=
sorry

end NUMINAMATH_CALUDE_both_not_land_l1172_117232


namespace NUMINAMATH_CALUDE_prob_four_ones_in_five_rolls_l1172_117264

/-- The probability of rolling a 1 on a fair six-sided die -/
def prob_one : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a fair six-sided die -/
def prob_not_one : ℚ := 5 / 6

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of times we want to roll a 1 -/
def target_ones : ℕ := 4

/-- The probability of rolling exactly four 1s in five rolls of a fair six-sided die -/
theorem prob_four_ones_in_five_rolls : 
  (num_rolls.choose target_ones : ℚ) * prob_one ^ target_ones * prob_not_one ^ (num_rolls - target_ones) = 25 / 7776 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_ones_in_five_rolls_l1172_117264


namespace NUMINAMATH_CALUDE_inner_triangle_side_length_l1172_117256

theorem inner_triangle_side_length 
  (outer_side : ℝ) 
  (inner_side : ℝ) 
  (small_side : ℝ) 
  (h_outer : outer_side = 6) 
  (h_small : small_side = 1) 
  (h_parallel : inner_triangles_parallel_to_outer)
  (h_vertex_outer : inner_triangles_vertex_on_outer_side)
  (h_vertex_inner : inner_triangles_vertex_on_other_inner)
  (h_congruent : inner_triangles_congruent)
  : inner_side = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_side_length_l1172_117256


namespace NUMINAMATH_CALUDE_card_arrangement_possible_l1172_117204

def initial_sequence : List ℕ := [7, 8, 9, 4, 5, 6, 1, 2, 3]
def final_sequence : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def reverse_sublist (l : List α) (start finish : ℕ) : List α :=
  (l.take start) ++ (l.drop start |>.take (finish - start + 1) |>.reverse) ++ (l.drop (finish + 1))

def can_transform (l : List ℕ) : Prop :=
  ∃ (s1 f1 s2 f2 s3 f3 : ℕ),
    reverse_sublist (reverse_sublist (reverse_sublist l s1 f1) s2 f2) s3 f3 = final_sequence

theorem card_arrangement_possible :
  can_transform initial_sequence :=
sorry

end NUMINAMATH_CALUDE_card_arrangement_possible_l1172_117204


namespace NUMINAMATH_CALUDE_ohara_triple_49_64_l1172_117200

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b y : ℕ) : Prop :=
  Real.sqrt a + Real.sqrt b = y

/-- Theorem: If (49, 64, y) is an O'Hara triple, then y = 15 -/
theorem ohara_triple_49_64 (y : ℕ) :
  is_ohara_triple 49 64 y → y = 15 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_49_64_l1172_117200
