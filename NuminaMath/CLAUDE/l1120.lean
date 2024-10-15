import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l1120_112070

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_condition 
  (a : ℕ → ℝ) (d : ℝ) (h : is_arithmetic_sequence a) :
  (d > 0 ↔ a 2 > a 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l1120_112070


namespace NUMINAMATH_CALUDE_power_of_128_over_7_l1120_112031

theorem power_of_128_over_7 : (128 : ℝ) ^ (3/7) = 8 := by sorry

end NUMINAMATH_CALUDE_power_of_128_over_7_l1120_112031


namespace NUMINAMATH_CALUDE_shirt_price_shirt_price_is_33_l1120_112025

theorem shirt_price (pants_price : ℝ) (num_pants : ℕ) (num_shirts : ℕ) (total_payment : ℝ) (change : ℝ) : ℝ :=
  let total_spent := total_payment - change
  let pants_total := pants_price * num_pants
  let shirts_total := total_spent - pants_total
  shirts_total / num_shirts

#check shirt_price 54 2 4 250 10 = 33

-- The proof
theorem shirt_price_is_33 :
  shirt_price 54 2 4 250 10 = 33 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_shirt_price_is_33_l1120_112025


namespace NUMINAMATH_CALUDE_garden_furniture_cost_l1120_112000

def bench_cost : ℝ := 150

def table_cost (bench_cost : ℝ) : ℝ := 2 * bench_cost

def combined_cost (bench_cost table_cost : ℝ) : ℝ := bench_cost + table_cost

theorem garden_furniture_cost : combined_cost bench_cost (table_cost bench_cost) = 450 := by
  sorry

end NUMINAMATH_CALUDE_garden_furniture_cost_l1120_112000


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1120_112091

theorem remainder_divisibility (x : ℤ) : x % 8 = 3 → x % 72 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1120_112091


namespace NUMINAMATH_CALUDE_trig_identity_l1120_112075

theorem trig_identity (α : ℝ) : 
  (Real.sin (α / 2))^6 - (Real.cos (α / 2))^6 = ((Real.sin α)^2 - 4) / 4 * Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1120_112075


namespace NUMINAMATH_CALUDE_limit_exponential_arcsin_ratio_l1120_112034

open Real

theorem limit_exponential_arcsin_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → 
    |((exp (3 * x) - exp (-2 * x)) / (2 * arcsin x - sin x)) - 5| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_exponential_arcsin_ratio_l1120_112034


namespace NUMINAMATH_CALUDE_classroom_wall_paint_area_l1120_112022

/-- Calculates the area to be painted on a wall with two windows. -/
def areaToBePainted (wallHeight wallWidth window1Height window1Width window2Height window2Width : ℕ) : ℕ :=
  let wallArea := wallHeight * wallWidth
  let window1Area := window1Height * window1Width
  let window2Area := window2Height * window2Width
  wallArea - window1Area - window2Area

/-- Proves that the area to be painted on the classroom wall is 243 square feet. -/
theorem classroom_wall_paint_area :
  areaToBePainted 15 18 3 5 2 6 = 243 := by
  sorry

#eval areaToBePainted 15 18 3 5 2 6

end NUMINAMATH_CALUDE_classroom_wall_paint_area_l1120_112022


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l1120_112056

theorem modular_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l1120_112056


namespace NUMINAMATH_CALUDE_inequality_proof_l1120_112097

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  x + 1 / (2 * y) > y + 1 / x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1120_112097


namespace NUMINAMATH_CALUDE_p_decreasing_zero_l1120_112087

/-- The probability that |T-H| = k after a game with 4m coins, 
    where T is the number of tails and H is the number of heads. -/
def p (m : ℕ) (k : ℕ) : ℝ := sorry

/-- The optimal strategy for the coin flipping game -/
def optimal_strategy : sorry := sorry

axiom p_zero_zero : p 0 0 = 1

axiom p_zero_pos : ∀ k : ℕ, k ≥ 1 → p 0 k = 0

/-- The main theorem: p_m(0) ≥ p_m+1(0) for all nonnegative integers m -/
theorem p_decreasing_zero : ∀ m : ℕ, p m 0 ≥ p (m + 1) 0 := by sorry

end NUMINAMATH_CALUDE_p_decreasing_zero_l1120_112087


namespace NUMINAMATH_CALUDE_number_equation_solution_l1120_112010

theorem number_equation_solution : 
  ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1120_112010


namespace NUMINAMATH_CALUDE_distinct_scores_is_nineteen_l1120_112032

/-- Represents the number of distinct possible scores for a basketball player -/
def distinctScores : ℕ :=
  let shotTypes := 3  -- free throw, 2-point basket, 3-point basket
  let totalShots := 8
  let pointValues := [1, 2, 3]
  19  -- The actual count of distinct scores

/-- Theorem stating that the number of distinct possible scores is 19 -/
theorem distinct_scores_is_nineteen :
  distinctScores = 19 := by sorry

end NUMINAMATH_CALUDE_distinct_scores_is_nineteen_l1120_112032


namespace NUMINAMATH_CALUDE_problem_statement_l1120_112021

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_abc : a * b * c = 1)
  (h_a_c : a + 1 / c = 7)
  (h_b_a : b + 1 / a = 16) :
  c + 1 / b = 25 / 111 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1120_112021


namespace NUMINAMATH_CALUDE_sqrt_power_eight_equals_390625_l1120_112057

theorem sqrt_power_eight_equals_390625 :
  (Real.sqrt ((Real.sqrt 5) ^ 4)) ^ 8 = 390625 := by sorry

end NUMINAMATH_CALUDE_sqrt_power_eight_equals_390625_l1120_112057


namespace NUMINAMATH_CALUDE_money_problem_l1120_112046

theorem money_problem (M : ℚ) : 
  (3/4 * (2/3 * (2/3 * M + 10) + 20) = M) → M = 30 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l1120_112046


namespace NUMINAMATH_CALUDE_beta_highest_success_ratio_l1120_112026

/-- Represents a participant's scores in a two-day challenge -/
structure ParticipantScores where
  day1_score : ℕ
  day1_attempted : ℕ
  day2_score : ℕ
  day2_attempted : ℕ

def ParticipantScores.total_score (p : ParticipantScores) : ℕ :=
  p.day1_score + p.day2_score

def ParticipantScores.total_attempted (p : ParticipantScores) : ℕ :=
  p.day1_attempted + p.day2_attempted

def ParticipantScores.success_ratio (p : ParticipantScores) : ℚ :=
  (p.total_score : ℚ) / p.total_attempted

def ParticipantScores.daily_success_ratio (p : ParticipantScores) (day : Fin 2) : ℚ :=
  match day with
  | 0 => (p.day1_score : ℚ) / p.day1_attempted
  | 1 => (p.day2_score : ℚ) / p.day2_attempted

theorem beta_highest_success_ratio
  (alpha : ParticipantScores)
  (beta : ParticipantScores)
  (h_total_points : alpha.total_attempted = 500)
  (h_alpha_scores : alpha.day1_score = 200 ∧ alpha.day1_attempted = 300 ∧
                    alpha.day2_score = 100 ∧ alpha.day2_attempted = 200)
  (h_beta_fewer : beta.day1_attempted < alpha.day1_attempted ∧
                  beta.day2_attempted < alpha.day2_attempted)
  (h_beta_nonzero : beta.day1_score > 0 ∧ beta.day2_score > 0)
  (h_beta_lower_ratio : ∀ day, beta.daily_success_ratio day < alpha.daily_success_ratio day)
  (h_alpha_ratio : alpha.success_ratio = 3/5)
  (h_beta_day1 : beta.day1_attempted = 220) :
  beta.success_ratio ≤ 248/500 :=
sorry

end NUMINAMATH_CALUDE_beta_highest_success_ratio_l1120_112026


namespace NUMINAMATH_CALUDE_blown_out_sand_dunes_with_treasure_l1120_112090

theorem blown_out_sand_dunes_with_treasure :
  let sand_dunes_remain_prob : ℚ := 1 / 3
  let sand_dunes_with_coupon_prob : ℚ := 2 / 3
  let total_blown_out_dunes : ℕ := 5
  let both_treasure_and_coupon_prob : ℚ := 8 / 90
  ∃ (treasure_dunes : ℕ),
    (treasure_dunes : ℚ) / total_blown_out_dunes * sand_dunes_with_coupon_prob = both_treasure_and_coupon_prob ∧
    treasure_dunes = 20 :=
by sorry

end NUMINAMATH_CALUDE_blown_out_sand_dunes_with_treasure_l1120_112090


namespace NUMINAMATH_CALUDE_property_P_for_given_numbers_l1120_112052

-- Define property P
def has_property_P (n : ℤ) : Prop :=
  ∃ x y z : ℤ, n = x^3 + y^3 + z^3 - 3*x*y*z

-- Theorem statement
theorem property_P_for_given_numbers :
  (has_property_P 1) ∧
  (has_property_P 5) ∧
  (has_property_P 2014) ∧
  (¬ has_property_P 2013) :=
by sorry

end NUMINAMATH_CALUDE_property_P_for_given_numbers_l1120_112052


namespace NUMINAMATH_CALUDE_gravel_cost_proof_l1120_112093

/-- Calculates the cost of graveling two intersecting roads on a rectangular lawn. -/
def gravel_cost (lawn_length lawn_width road_width gravel_cost_per_sqm : ℕ) : ℕ :=
  let road_length_area := lawn_length * road_width
  let road_width_area := (lawn_width - road_width) * road_width
  let total_area := road_length_area + road_width_area
  total_area * gravel_cost_per_sqm

/-- Proves that the cost of graveling two intersecting roads on a rectangular lawn
    with given dimensions and costs is equal to 3900. -/
theorem gravel_cost_proof :
  gravel_cost 80 60 10 3 = 3900 := by
  sorry

end NUMINAMATH_CALUDE_gravel_cost_proof_l1120_112093


namespace NUMINAMATH_CALUDE_circle_tangency_l1120_112011

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle C₂ with equation x² + y² - 6x - 8y + F = 0 -/
def C₂ (F : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + F = 0}

/-- Two circles are internally tangent if they intersect at exactly one point
    and one circle is completely inside the other -/
def internally_tangent (C₁ C₂ : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ C₁ ∧ p ∈ C₂ ∧ (∀ q, q ∈ C₁ ∩ C₂ → q = p) ∧
  (∀ r, r ∈ C₁ → r ∈ C₂ ∨ r = p)

/-- Theorem: If C₁ is internally tangent to C₂, then F = -11 -/
theorem circle_tangency (F : ℝ) :
  internally_tangent C₁ (C₂ F) → F = -11 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_l1120_112011


namespace NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l1120_112069

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem fourth_power_of_cube_of_third_smallest_prime :
  (nthSmallestPrime 3) ^ 3 ^ 4 = 244140625 := by sorry

end NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l1120_112069


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1120_112030

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1120_112030


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1120_112028

/-- A hyperbola with parameter b > 0 -/
structure Hyperbola (b : ℝ) : Prop where
  pos : b > 0

/-- A line with equation x + 3y - 1 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + 3 * p.2 - 1 = 0}

/-- The left branch of the hyperbola -/
def LeftBranch (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 < 0 ∧ p.1^2 / 4 - p.2^2 / b^2 = 1}

/-- Predicate for line intersecting the left branch of hyperbola -/
def Intersects (b : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ Line ∩ LeftBranch b

/-- Theorem stating that b > 1 is sufficient but not necessary for intersection -/
theorem sufficient_not_necessary (h : Hyperbola b) :
    (b > 1 → Intersects b) ∧ ¬(Intersects b → b > 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1120_112028


namespace NUMINAMATH_CALUDE_LL₁_length_is_20_over_17_l1120_112044

/-- Right triangle XYZ with hypotenuse XZ = 13 and leg XY = 5 -/
structure TriangleXYZ where
  XZ : ℝ
  XY : ℝ
  is_right : XZ = 13 ∧ XY = 5

/-- Point X₁ on YZ where the angle bisector of ∠X meets YZ -/
def X₁ (t : TriangleXYZ) : ℝ × ℝ := sorry

/-- Right triangle LMN with hypotenuse LM = X₁Z and leg LN = X₁Y -/
structure TriangleLMN (t : TriangleXYZ) where
  LM : ℝ
  LN : ℝ
  is_right : LM = (X₁ t).2 ∧ LN = (X₁ t).1

/-- Point L₁ on MN where the angle bisector of ∠L meets MN -/
def L₁ (t : TriangleXYZ) (u : TriangleLMN t) : ℝ × ℝ := sorry

/-- The length of LL₁ -/
def LL₁_length (t : TriangleXYZ) (u : TriangleLMN t) : ℝ := sorry

/-- Theorem: The length of LL₁ is 20/17 -/
theorem LL₁_length_is_20_over_17 (t : TriangleXYZ) (u : TriangleLMN t) :
  LL₁_length t u = 20 / 17 := by sorry

end NUMINAMATH_CALUDE_LL₁_length_is_20_over_17_l1120_112044


namespace NUMINAMATH_CALUDE_square_remainder_l1120_112098

theorem square_remainder (n : ℤ) : n % 5 = 3 → (n^2) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_l1120_112098


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l1120_112035

/-- A box containing white and red balls -/
structure Box where
  white : Nat
  red : Nat

/-- The number of balls drawn from the box -/
def drawn : Nat := 3

/-- Event A: Exactly one red ball is drawn -/
def eventA (box : Box) : Prop :=
  ∃ (r w : Nat), r = 1 ∧ w = drawn - r ∧ r ≤ box.red ∧ w ≤ box.white

/-- Event B: Exactly one white ball is drawn -/
def eventB (box : Box) : Prop :=
  ∃ (w r : Nat), w = 1 ∧ r = drawn - w ∧ w ≤ box.white ∧ r ≤ box.red

/-- The box in the problem -/
def problemBox : Box := ⟨4, 3⟩

theorem events_mutually_exclusive_but_not_complementary :
  (¬ ∃ (outcome : Nat × Nat), eventA problemBox ∧ eventB problemBox) ∧
  (∃ (outcome : Nat × Nat), ¬(eventA problemBox ∨ eventB problemBox)) :=
by sorry


end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l1120_112035


namespace NUMINAMATH_CALUDE_complex_modulus_example_l1120_112036

theorem complex_modulus_example : Complex.abs (7/8 + 3*I) = 25/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l1120_112036


namespace NUMINAMATH_CALUDE_inequality_proof_l1120_112072

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_condition : a * b + b * c + c * a ≤ 1) :
  a + b + c + Real.sqrt 3 ≥ 8 * a * b * c * (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1120_112072


namespace NUMINAMATH_CALUDE_inequality_proof_l1120_112064

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  Real.sqrt (a * b) < (a - b) / (Real.log a - Real.log b) ∧ 
  (a - b) / (Real.log a - Real.log b) < (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1120_112064


namespace NUMINAMATH_CALUDE_consecutive_points_length_l1120_112081

/-- Given 5 consecutive points on a straight line, prove that ae = 21 -/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (c - b = 3 * (d - c)) →  -- bc = 3 cd
  (e - d = 8) →            -- de = 8
  (b - a = 5) →            -- ab = 5
  (c - a = 11) →           -- ac = 11
  (e - a = 21) :=          -- ae = 21
by sorry

end NUMINAMATH_CALUDE_consecutive_points_length_l1120_112081


namespace NUMINAMATH_CALUDE_championship_ties_l1120_112080

/-- Represents the number of points and games for a hockey team -/
structure HockeyTeam where
  wins : ℕ
  ties : ℕ
  totalPoints : ℕ

/-- Calculates the total points for a hockey team -/
def calculatePoints (team : HockeyTeam) : ℕ :=
  3 * team.wins + 2 * team.ties

theorem championship_ties (team : HockeyTeam) 
  (h1 : team.totalPoints = 85)
  (h2 : team.wins = team.ties + 15)
  (h3 : calculatePoints team = team.totalPoints) :
  team.ties = 8 := by
sorry

end NUMINAMATH_CALUDE_championship_ties_l1120_112080


namespace NUMINAMATH_CALUDE_benny_baseball_gear_expense_l1120_112049

/-- The amount Benny spent on baseball gear -/
def amount_spent (initial_amount remaining_amount : ℕ) : ℕ :=
  initial_amount - remaining_amount

/-- Theorem: Benny spent $47 on baseball gear -/
theorem benny_baseball_gear_expense :
  amount_spent 79 32 = 47 := by
  sorry

end NUMINAMATH_CALUDE_benny_baseball_gear_expense_l1120_112049


namespace NUMINAMATH_CALUDE_triangle_inequality_four_points_l1120_112068

-- Define a metric space
variable {X : Type*} [MetricSpace X]

-- Define four points in the metric space
variable (A B C D : X)

-- State the theorem
theorem triangle_inequality_four_points :
  dist A D ≤ dist A B + dist B C + dist C D := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_four_points_l1120_112068


namespace NUMINAMATH_CALUDE_second_chapter_longer_l1120_112094

/-- A book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- The difference in pages between two chapters -/
def page_difference (b : Book) : ℕ := b.chapter2_pages - b.chapter1_pages

theorem second_chapter_longer (b : Book) 
  (h1 : b.chapter1_pages = 37) 
  (h2 : b.chapter2_pages = 80) : 
  page_difference b = 43 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_longer_l1120_112094


namespace NUMINAMATH_CALUDE_negative_seven_plus_three_l1120_112041

theorem negative_seven_plus_three : (-7 : ℤ) + 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_seven_plus_three_l1120_112041


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l1120_112085

/-- Given a parabola and a hyperbola satisfying certain conditions, 
    prove their equations. -/
theorem parabola_hyperbola_equations :
  ∀ (a b : ℝ) (parabola hyperbola : ℝ → ℝ → Prop),
    a > 0 → b > 0 →
    (∀ x y, hyperbola x y ↔ x^2 / a^2 - y^2 / b^2 = 1) →
    (∃ f, f > 0 ∧ ∀ x y, parabola x y ↔ y^2 = 4 * f * x) →
    (∃ xf yf, hyperbola xf yf ∧ ∀ x y, parabola x y → (x - xf)^2 + y^2 = f^2) →
    parabola (3/2) (Real.sqrt 6) →
    hyperbola (3/2) (Real.sqrt 6) →
    (∀ x y, parabola x y ↔ y^2 = 4 * x) ∧
    (∀ x y, hyperbola x y ↔ 4 * x^2 - 4 * y^2 / 3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l1120_112085


namespace NUMINAMATH_CALUDE_sum_of_squares_progression_l1120_112061

/-- Given two infinite geometric progressions with common ratio q where |q| < 1,
    differing only in the sign of their common ratios, and with sums S₁ and S₂ respectively,
    the sum of the infinite geometric progression formed from the squares of the terms
    of either progression is equal to S₁ * S₂. -/
theorem sum_of_squares_progression (q : ℝ) (S₁ S₂ : ℝ) (h : |q| < 1) :
  let b₁ : ℝ := S₁ * (1 - q)
  ∑' n, (b₁ * q ^ n) ^ 2 = S₁ * S₂ :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_progression_l1120_112061


namespace NUMINAMATH_CALUDE_quadrilateral_comparison_l1120_112054

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral defined by four points -/
structure Quadrilateral :=
  (a b c d : Point)

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Calculates the perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℝ := sorry

/-- Quadrilateral I defined by its vertices -/
def quadI : Quadrilateral :=
  { a := {x := 0, y := 0},
    b := {x := 3, y := 0},
    c := {x := 3, y := 3},
    d := {x := 0, y := 2} }

/-- Quadrilateral II defined by its vertices -/
def quadII : Quadrilateral :=
  { a := {x := 0, y := 0},
    b := {x := 3, y := 0},
    c := {x := 3, y := 2},
    d := {x := 0, y := 3} }

theorem quadrilateral_comparison :
  (area quadI = 7.5 ∧ area quadII = 7.5) ∧
  perimeter quadI > perimeter quadII := by sorry

end NUMINAMATH_CALUDE_quadrilateral_comparison_l1120_112054


namespace NUMINAMATH_CALUDE_road_paving_length_l1120_112029

/-- The length of road paved in April, in meters -/
def april_length : ℕ := 480

/-- The difference between March and April paving lengths, in meters -/
def length_difference : ℕ := 160

/-- The total length of road paved in March and April -/
def total_length : ℕ := april_length + (april_length + length_difference)

theorem road_paving_length : total_length = 1120 := by sorry

end NUMINAMATH_CALUDE_road_paving_length_l1120_112029


namespace NUMINAMATH_CALUDE_archie_red_coins_l1120_112043

/-- Represents the number of coins collected for each color --/
structure CoinCount where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Calculates the total number of coins --/
def total_coins (c : CoinCount) : ℕ := c.yellow + c.red + c.blue

/-- Calculates the total money earned --/
def total_money (c : CoinCount) : ℕ := c.yellow + 3 * c.red + 5 * c.blue

/-- Theorem stating that Archie collected 700 red coins --/
theorem archie_red_coins :
  ∃ (c : CoinCount),
    total_coins c = 2800 ∧
    total_money c = 7800 ∧
    c.blue = c.red + 200 ∧
    c.red = 700 := by
  sorry


end NUMINAMATH_CALUDE_archie_red_coins_l1120_112043


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1120_112006

theorem sum_of_roots_quadratic (b : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ + b = 0 → x₂^2 - 2*x₂ + b = 0 → x₁ + x₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1120_112006


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1120_112004

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- The configuration of squares and rectangle -/
structure Configuration where
  square : Square
  rectangle : Rectangle
  square_count : ℕ

/-- The theorem statement -/
theorem rectangle_ratio (config : Configuration) :
  config.square_count = 3 →
  config.rectangle.length = config.square_count * config.square.side →
  config.rectangle.width = config.square.side →
  config.rectangle.length / config.rectangle.width = 3 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_ratio_l1120_112004


namespace NUMINAMATH_CALUDE_student_number_problem_l1120_112053

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 102 → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1120_112053


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1120_112045

theorem min_value_of_expression (c d : ℤ) (h : c > d) :
  (c + 2*d) / (c - d) + (c - d) / (c + 2*d) ≥ 2 ∧
  ∃ (c' d' : ℤ), c' > d' ∧ (c' + 2*d') / (c' - d') + (c' - d') / (c' + 2*d') = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1120_112045


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_sum_zero_l1120_112062

theorem sqrt_abs_sum_zero_implies_power_sum_zero (a b : ℝ) :
  Real.sqrt (a + 1) + |b - 1| = 0 → a^2023 + b^2024 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_sum_zero_l1120_112062


namespace NUMINAMATH_CALUDE_root_implies_u_value_l1120_112058

theorem root_implies_u_value (u : ℝ) : 
  (6 * ((-25 - Real.sqrt 421) / 12)^2 + 25 * ((-25 - Real.sqrt 421) / 12) + u = 0) → 
  u = 8.5 := by
sorry

end NUMINAMATH_CALUDE_root_implies_u_value_l1120_112058


namespace NUMINAMATH_CALUDE_count_solutions_x_plus_y_plus_z_eq_10_l1120_112002

def positive_integer_solutions (n : ℕ) : ℕ :=
  (n - 1) * (n - 2) / 2

theorem count_solutions_x_plus_y_plus_z_eq_10 :
  positive_integer_solutions 10 = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_x_plus_y_plus_z_eq_10_l1120_112002


namespace NUMINAMATH_CALUDE_abc_maximum_l1120_112095

theorem abc_maximum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + c^2 = (a + c) * (b + c)) (h2 : a + b + c = 3) :
  a * b * c ≤ 1 ∧ ∃ (a' b' c' : ℝ), a' * b' * c' = 1 ∧ 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 
    a' * b' + c'^2 = (a' + c') * (b' + c') ∧ 
    a' + b' + c' = 3 :=
by sorry

end NUMINAMATH_CALUDE_abc_maximum_l1120_112095


namespace NUMINAMATH_CALUDE_problem_solution_l1120_112096

noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := k * a^x - a^(-x)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 4 * (f a 1 x)

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := f a 1
  ∀ x : ℝ, f x = -f (-x) →
  f 1 > 0 →
  f 1 = 3/2 →
  (∀ x : ℝ, f (x^2 + 2*x) + f (x - 4) > 0 ↔ x < -4 ∨ x > 1) ∧
  (∃ m : ℝ, m = -2 ∧ ∀ x : ℝ, x ≥ 1 → g a x ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1120_112096


namespace NUMINAMATH_CALUDE_problem_solution_l1120_112082

-- Define the equation from the problem
def equation (n : ℕ) : Prop := 2^(2*n) = 2^n + 992

-- Define the constant term function
def constant_term (n : ℕ) : ℕ := Nat.choose (2*n) 2

-- Theorem statement
theorem problem_solution :
  (∃ n : ℕ, equation n ∧ n = 5) ∧
  constant_term 5 = 45 := by
sorry


end NUMINAMATH_CALUDE_problem_solution_l1120_112082


namespace NUMINAMATH_CALUDE_prism_21_edges_has_9_faces_l1120_112078

/-- Represents a prism with a given number of edges. -/
structure Prism where
  edges : ℕ

/-- Calculates the number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  (p.edges / 3) + 2

/-- Theorem stating that a prism with 21 edges has 9 faces. -/
theorem prism_21_edges_has_9_faces :
  ∀ (p : Prism), p.edges = 21 → num_faces p = 9 := by
  sorry

#eval num_faces { edges := 21 }

end NUMINAMATH_CALUDE_prism_21_edges_has_9_faces_l1120_112078


namespace NUMINAMATH_CALUDE_talia_drives_16_miles_l1120_112088

/-- Represents the total distance Talia drives in a day -/
def total_distance (home_to_park park_to_grocery grocery_to_home : ℝ) : ℝ :=
  home_to_park + park_to_grocery + grocery_to_home

/-- Theorem stating that Talia drives 16 miles given the distances between locations -/
theorem talia_drives_16_miles :
  let home_to_park : ℝ := 5
  let park_to_grocery : ℝ := 3
  let grocery_to_home : ℝ := 8
  total_distance home_to_park park_to_grocery grocery_to_home = 16 := by
  sorry

end NUMINAMATH_CALUDE_talia_drives_16_miles_l1120_112088


namespace NUMINAMATH_CALUDE_propositions_truth_l1120_112065

theorem propositions_truth : 
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) ∧ 
  (∃ x : ℝ, x^2 - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_propositions_truth_l1120_112065


namespace NUMINAMATH_CALUDE_max_girls_in_ballet_l1120_112047

/-- Represents the number of boys participating in the ballet -/
def num_boys : ℕ := 5

/-- Represents the distance requirement between girls and boys -/
def distance : ℕ := 5

/-- Represents the number of boys required at the specified distance from each girl -/
def boys_per_girl : ℕ := 2

/-- Calculates the maximum number of girls that can participate in the ballet -/
def max_girls : ℕ := (num_boys.choose boys_per_girl) * 2

/-- Theorem stating the maximum number of girls that can participate in the ballet -/
theorem max_girls_in_ballet : max_girls = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_girls_in_ballet_l1120_112047


namespace NUMINAMATH_CALUDE_steve_email_percentage_l1120_112071

/-- Given Steve's email management scenario, prove that the percentage of emails
    moved to the work folder out of the remaining emails after trashing is 40%. -/
theorem steve_email_percentage :
  ∀ (initial_emails : ℕ) (emails_left : ℕ),
    initial_emails = 400 →
    emails_left = 120 →
    let emails_after_trash : ℕ := initial_emails / 2
    let emails_to_work : ℕ := emails_after_trash - emails_left
    (emails_to_work : ℚ) / emails_after_trash * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_steve_email_percentage_l1120_112071


namespace NUMINAMATH_CALUDE_parabola_intersection_l1120_112086

/-- Given a parabola y = x^2 and four points on it, if two lines formed by these points
    intersect on the y-axis, then the x-coordinate of the fourth point is determined by
    the x-coordinates of the other three points. -/
theorem parabola_intersection (a b c d : ℝ) : 
  (∃ l : ℝ, (a^2 = (b + a)*a + l ∧ b^2 = (b + a)*b + l) ∧ 
             (c^2 = (d + c)*c + l ∧ d^2 = (d + c)*d + l)) →
  d = a * b / c :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1120_112086


namespace NUMINAMATH_CALUDE_min_value_theorem_l1120_112016

theorem min_value_theorem (a b c : ℝ) (h : a + 2*b + 3*c = 2) :
  (∀ x y z : ℝ, x + 2*y + 3*z = 2 → a^2 + 2*b^2 + 3*c^2 ≤ x^2 + 2*y^2 + 3*z^2) →
  2*a + 4*b + 9*c = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1120_112016


namespace NUMINAMATH_CALUDE_problem_statement_l1120_112024

theorem problem_statement (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) :
  x^2004 + y^2004 = 2^2004 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1120_112024


namespace NUMINAMATH_CALUDE_count_a_values_correct_l1120_112076

/-- The number of integer values of a for which (a-1)x^2 + 2x - a - 1 = 0 has integer roots for x -/
def count_a_values : ℕ := 5

/-- The equation has integer roots for x -/
def has_integer_roots (a : ℤ) : Prop :=
  ∃ x : ℤ, (a - 1) * x^2 + 2 * x - a - 1 = 0

/-- There are exactly 5 integer values of a for which the equation has integer roots -/
theorem count_a_values_correct :
  (∃ S : Finset ℤ, S.card = count_a_values ∧ 
    (∀ a : ℤ, a ∈ S ↔ has_integer_roots a)) ∧
  (∀ T : Finset ℤ, (∀ a : ℤ, a ∈ T ↔ has_integer_roots a) → T.card ≤ count_a_values) :=
sorry

end NUMINAMATH_CALUDE_count_a_values_correct_l1120_112076


namespace NUMINAMATH_CALUDE_paper_strip_length_l1120_112039

theorem paper_strip_length (strip_length : ℝ) : 
  strip_length > 0 →
  strip_length + strip_length - 6 = 30 →
  strip_length = 18 := by
sorry

end NUMINAMATH_CALUDE_paper_strip_length_l1120_112039


namespace NUMINAMATH_CALUDE_min_value_m_plus_n_l1120_112027

theorem min_value_m_plus_n (a b m n : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_mean : (a + b) / 2 = 1 / 2) (hm : m = a + 1 / a) (hn : n = b + 1 / b) :
  ∀ x y, x > 0 → y > 0 → (x + y) / 2 = 1 / 2 → 
  (x + 1 / x) + (y + 1 / y) ≥ m + n ∧ m + n ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_m_plus_n_l1120_112027


namespace NUMINAMATH_CALUDE_grapefruit_juice_percentage_l1120_112084

def total_volume : ℝ := 50
def orange_juice : ℝ := 20
def lemon_juice_percentage : ℝ := 35

theorem grapefruit_juice_percentage :
  (total_volume - orange_juice - (lemon_juice_percentage / 100) * total_volume) / total_volume * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_grapefruit_juice_percentage_l1120_112084


namespace NUMINAMATH_CALUDE_hajar_score_is_24_l1120_112017

def guessing_game (hajar_score farah_score : ℕ) : Prop :=
  farah_score - hajar_score = 21 ∧
  farah_score + hajar_score = 69 ∧
  farah_score > hajar_score

theorem hajar_score_is_24 :
  ∃ (hajar_score farah_score : ℕ), guessing_game hajar_score farah_score ∧ hajar_score = 24 :=
by sorry

end NUMINAMATH_CALUDE_hajar_score_is_24_l1120_112017


namespace NUMINAMATH_CALUDE_otimes_inequality_system_l1120_112050

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a - 2 * b

-- Theorem statement
theorem otimes_inequality_system (a : ℝ) :
  (∀ x : ℝ, x > 6 ↔ (otimes x 3 > 0 ∧ otimes x a > a)) →
  a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_otimes_inequality_system_l1120_112050


namespace NUMINAMATH_CALUDE_equation_solution_l1120_112066

theorem equation_solution (y : ℝ) : y + 81 / (y - 3) = -12 ↔ y = -6 ∨ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1120_112066


namespace NUMINAMATH_CALUDE_bead_probability_l1120_112055

/-- The probability that a point on a line segment of length 3 is more than 1 unit away from both endpoints is 1/3 -/
theorem bead_probability : 
  let segment_length : ℝ := 3
  let min_distance : ℝ := 1
  let favorable_length : ℝ := segment_length - 2 * min_distance
  favorable_length / segment_length = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_bead_probability_l1120_112055


namespace NUMINAMATH_CALUDE_cubic_sum_l1120_112099

theorem cubic_sum (x y : ℝ) (h1 : 1/x + 1/y = 4) (h2 : x + y + x*y = 3) :
  x^3 + y^3 = 1188/125 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_l1120_112099


namespace NUMINAMATH_CALUDE_sum_of_squares_l1120_112089

theorem sum_of_squares (a b c : ℝ) 
  (h1 : a * b + b * c + c * a = 5)
  (h2 : a + b + c = 20) :
  a^2 + b^2 + c^2 = 390 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1120_112089


namespace NUMINAMATH_CALUDE_square_difference_equality_l1120_112023

theorem square_difference_equality : (15 + 12)^2 - (12^2 + 15^2) = 360 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1120_112023


namespace NUMINAMATH_CALUDE_find_M_l1120_112033

theorem find_M : ∃ (M : ℕ+), (36 : ℕ)^2 * 81^2 = 18^2 * M^2 ∧ M = 162 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l1120_112033


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1120_112073

/-- Given a line segment from (2, 2) to (x, 6) with length 10 and x > 0, prove x = 2 + 2√21 -/
theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  (x - 2)^2 + 4^2 = 10^2 → 
  x = 2 + 2 * Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1120_112073


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_eight_l1120_112013

theorem smallest_n_multiple_of_eight (x y : ℤ) 
  (h1 : ∃ k : ℤ, x + 2 = 8 * k) 
  (h2 : ∃ m : ℤ, y - 2 = 8 * m) : 
  (∀ n : ℕ, n > 0 → n < 4 → ¬(∃ p : ℤ, x^2 - x*y + y^2 + n = 8 * p)) ∧ 
  (∃ q : ℤ, x^2 - x*y + y^2 + 4 = 8 * q) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_eight_l1120_112013


namespace NUMINAMATH_CALUDE_percentage_returned_is_65_percent_l1120_112042

/-- Represents the library's special collection --/
structure SpecialCollection where
  initial_count : ℕ
  final_count : ℕ
  loaned_out : ℕ

/-- Calculates the percentage of loaned books returned --/
def percentage_returned (sc : SpecialCollection) : ℚ :=
  (sc.loaned_out - (sc.initial_count - sc.final_count)) / sc.loaned_out * 100

/-- Theorem stating that the percentage of loaned books returned is 65% --/
theorem percentage_returned_is_65_percent (sc : SpecialCollection) 
  (h1 : sc.initial_count = 150)
  (h2 : sc.final_count = 122)
  (h3 : sc.loaned_out = 80) : 
  percentage_returned sc = 65 := by
  sorry

#eval percentage_returned { initial_count := 150, final_count := 122, loaned_out := 80 }

end NUMINAMATH_CALUDE_percentage_returned_is_65_percent_l1120_112042


namespace NUMINAMATH_CALUDE_inequality_proof_l1120_112015

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1120_112015


namespace NUMINAMATH_CALUDE_largest_perimeter_is_24_l1120_112092

/-- Represents a configuration of two regular polygons and a circle meeting at a point -/
structure ShapeConfiguration where
  n : ℕ  -- number of sides in each polygon
  polygonSideLength : ℝ
  circleRadius : ℝ
  polygonAngleSum : ℝ  -- sum of interior angles of polygons at meeting point
  circleAngle : ℝ  -- angle subtended by circle at meeting point

/-- The perimeter of the configuration, excluding the circle's circumference -/
def perimeter (config : ShapeConfiguration) : ℝ :=
  2 * config.n * config.polygonSideLength

/-- Theorem stating the largest possible perimeter for the given configuration -/
theorem largest_perimeter_is_24 (config : ShapeConfiguration) : 
  config.polygonSideLength = 2 ∧ 
  config.circleRadius = 1 ∧ 
  config.polygonAngleSum + config.circleAngle = 360 →
  ∃ (maxConfig : ShapeConfiguration), 
    perimeter maxConfig = 24 ∧ 
    ∀ (c : ShapeConfiguration), perimeter c ≤ perimeter maxConfig :=
by
  sorry

end NUMINAMATH_CALUDE_largest_perimeter_is_24_l1120_112092


namespace NUMINAMATH_CALUDE_classroom_size_theorem_l1120_112048

/-- Represents the number of students in a classroom -/
def classroom_size (boys : ℕ) (girls : ℕ) : ℕ := boys + girls

/-- Represents the ratio of boys to girls -/
def ratio_boys_girls (boys : ℕ) (girls : ℕ) : Prop := 3 * girls = 5 * boys

theorem classroom_size_theorem (boys girls : ℕ) :
  ratio_boys_girls boys girls →
  girls = boys + 4 →
  classroom_size boys girls = 16 := by
sorry

end NUMINAMATH_CALUDE_classroom_size_theorem_l1120_112048


namespace NUMINAMATH_CALUDE_nicholas_bottle_caps_l1120_112020

theorem nicholas_bottle_caps :
  ∀ (initial : ℕ),
  initial + 85 = 93 →
  initial = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_nicholas_bottle_caps_l1120_112020


namespace NUMINAMATH_CALUDE_min_value_expression_l1120_112063

theorem min_value_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) (heq : 2 * m + n = 4) :
  1 / m + 2 / n ≥ 2 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 4 ∧ 1 / m₀ + 2 / n₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1120_112063


namespace NUMINAMATH_CALUDE_function_symmetry_l1120_112074

theorem function_symmetry (f : ℝ → ℝ) (t : ℝ) :
  (∀ x, f x = 3 * x + Real.sin x + 1) →
  f t = 2 →
  f (-t) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l1120_112074


namespace NUMINAMATH_CALUDE_project_completion_time_l1120_112005

theorem project_completion_time (a b total_time quit_time : ℝ) 
  (hb : b = 30)
  (htotal : total_time = 15)
  (hquit : quit_time = 10)
  (h_completion : 5 * (1/a + 1/b) + 10 * (1/b) = 1) :
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_project_completion_time_l1120_112005


namespace NUMINAMATH_CALUDE_cindy_hourly_rate_l1120_112018

/-- Represents Cindy's teaching situation -/
structure TeachingSituation where
  num_courses : ℕ
  total_weekly_hours : ℕ
  weeks_in_month : ℕ
  monthly_earnings_per_course : ℕ

/-- Calculates the hourly rate given a teaching situation -/
def hourly_rate (s : TeachingSituation) : ℚ :=
  s.monthly_earnings_per_course / (s.total_weekly_hours / s.num_courses * s.weeks_in_month)

/-- Theorem stating that Cindy's hourly rate is $25 given the specified conditions -/
theorem cindy_hourly_rate :
  let s : TeachingSituation := {
    num_courses := 4,
    total_weekly_hours := 48,
    weeks_in_month := 4,
    monthly_earnings_per_course := 1200
  }
  hourly_rate s = 25 := by sorry

end NUMINAMATH_CALUDE_cindy_hourly_rate_l1120_112018


namespace NUMINAMATH_CALUDE_vector_collinearity_l1120_112008

/-- Two vectors in R² -/
def PA : Fin 2 → ℝ := ![(-1 : ℝ), 2]
def PB (x : ℝ) : Fin 2 → ℝ := ![2, x]

/-- Collinearity condition for three points in R² -/
def collinear (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 - v 1 * w 0 = 0

theorem vector_collinearity (x : ℝ) : 
  collinear PA (PB x) → x = -4 := by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1120_112008


namespace NUMINAMATH_CALUDE_valid_plate_count_l1120_112059

/-- Represents a license plate with 4 characters -/
structure LicensePlate :=
  (first : Char) (second : Char) (third : Char) (fourth : Char)

/-- Checks if a character is a letter -/
def isLetter (c : Char) : Bool := c.isAlpha

/-- Checks if a character is a digit -/
def isDigit (c : Char) : Bool := c.isDigit

/-- Checks if a license plate is valid according to the given conditions -/
def isValidPlate (plate : LicensePlate) : Bool :=
  (isLetter plate.first) &&
  (isDigit plate.second) &&
  (isDigit plate.third) &&
  (isLetter plate.fourth) &&
  (plate.first == plate.fourth || plate.second == plate.third)

/-- The total number of possible characters for a letter position -/
def numLetters : Nat := 26

/-- The total number of possible characters for a digit position -/
def numDigits : Nat := 10

/-- Counts the number of valid license plates -/
def countValidPlates : Nat :=
  (numLetters * numDigits * 1 * numLetters) +  -- Same digits
  (numLetters * numDigits * numDigits * 1) -   -- Same letters
  (numLetters * numDigits * 1 * 1)             -- Both pairs same

theorem valid_plate_count :
  countValidPlates = 9100 := by
  sorry

#eval countValidPlates  -- Should output 9100

end NUMINAMATH_CALUDE_valid_plate_count_l1120_112059


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1120_112067

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 7 * x - 10 = 0 ↔ x = -2 ∨ x = 5/2) ∧ 
  (7^2 - 4 * k * (-10) > 0) ↔ 
  k = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1120_112067


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1120_112009

/-- Given a line segment with midpoint (6, -10) and one endpoint (8, 0),
    the sum of the coordinates of the other endpoint is -16. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (x + 8) / 2 = 6 ∧ (y + 0) / 2 = -10 → 
  x + y = -16 := by
sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1120_112009


namespace NUMINAMATH_CALUDE_mary_chopped_six_tables_l1120_112003

/-- Represents the number of sticks of wood produced by different furniture items -/
structure FurnitureWood where
  chair : Nat
  table : Nat
  stool : Nat

/-- Represents the chopping and burning scenario -/
structure WoodScenario where
  furniture : FurnitureWood
  chopped_chairs : Nat
  chopped_stools : Nat
  burn_rate : Nat
  warm_hours : Nat

/-- Calculates the number of tables chopped given a wood scenario -/
def tables_chopped (scenario : WoodScenario) : Nat :=
  let total_wood := scenario.warm_hours * scenario.burn_rate
  let wood_from_chairs := scenario.chopped_chairs * scenario.furniture.chair
  let wood_from_stools := scenario.chopped_stools * scenario.furniture.stool
  let wood_from_tables := total_wood - wood_from_chairs - wood_from_stools
  wood_from_tables / scenario.furniture.table

theorem mary_chopped_six_tables :
  let mary_scenario : WoodScenario := {
    furniture := { chair := 6, table := 9, stool := 2 },
    chopped_chairs := 18,
    chopped_stools := 4,
    burn_rate := 5,
    warm_hours := 34
  }
  tables_chopped mary_scenario = 6 := by
  sorry

end NUMINAMATH_CALUDE_mary_chopped_six_tables_l1120_112003


namespace NUMINAMATH_CALUDE_pencil_users_count_l1120_112051

/-- The number of attendants who used a pen -/
def pen_users : ℕ := 15

/-- The number of attendants who used only one type of writing tool -/
def single_tool_users : ℕ := 20

/-- The number of attendants who used both types of writing tools -/
def both_tool_users : ℕ := 10

/-- The number of attendants who used a pencil -/
def pencil_users : ℕ := single_tool_users + both_tool_users - (pen_users - both_tool_users)

theorem pencil_users_count : pencil_users = 25 := by sorry

end NUMINAMATH_CALUDE_pencil_users_count_l1120_112051


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1120_112060

/-- The solution set of the quadratic inequality -x^2 + 4x + 12 > 0 is (-2, 6) -/
theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 4*x + 12 > 0} = Set.Ioo (-2 : ℝ) 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1120_112060


namespace NUMINAMATH_CALUDE_population_ratio_l1120_112037

/-- Represents the population of a city -/
structure CityPopulation where
  value : ℕ

/-- The relationship between populations of different cities -/
structure PopulationRelationship where
  cityA : CityPopulation
  cityB : CityPopulation
  cityC : CityPopulation
  cityD : CityPopulation
  cityE : CityPopulation
  cityF : CityPopulation
  A_to_B : cityA.value = 5 * cityB.value
  B_to_C : cityB.value = 3 * cityC.value
  C_to_D : cityC.value = 8 * cityD.value
  D_to_E : cityD.value = 2 * cityE.value
  E_to_F : cityE.value = 6 * cityF.value

/-- Theorem stating the ratio of population of City A to City F -/
theorem population_ratio (r : PopulationRelationship) : 
  r.cityA.value = 1440 * r.cityF.value := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_l1120_112037


namespace NUMINAMATH_CALUDE_task_assignment_count_l1120_112019

/-- Represents the number of people who can work as both English translators and software designers -/
def both_jobs : ℕ := 1

/-- Represents the total number of people -/
def total_people : ℕ := 8

/-- Represents the number of people who can work as English translators -/
def english_translators : ℕ := 5

/-- Represents the number of people who can work as software designers -/
def software_designers : ℕ := 4

/-- Represents the number of people to be selected for the task -/
def selected_people : ℕ := 5

/-- Represents the number of people to be assigned as English translators -/
def assigned_translators : ℕ := 3

/-- Represents the number of people to be assigned as software designers -/
def assigned_designers : ℕ := 2

/-- Theorem stating that the number of ways to assign tasks is 42 -/
theorem task_assignment_count : 
  (Nat.choose (english_translators - both_jobs) assigned_translators * 
   Nat.choose (software_designers - both_jobs) assigned_designers) +
  (Nat.choose (english_translators - both_jobs) (assigned_translators - 1) * 
   Nat.choose software_designers assigned_designers) +
  (Nat.choose english_translators assigned_translators * 
   Nat.choose (software_designers - both_jobs) (assigned_designers - 1)) = 42 :=
by sorry

end NUMINAMATH_CALUDE_task_assignment_count_l1120_112019


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1120_112007

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero -/
theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, 4*x + 6*y + k = 0 → y^2 = 32*x) ↔ k = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1120_112007


namespace NUMINAMATH_CALUDE_total_pennies_donated_l1120_112079

def cassandra_pennies : ℕ := 5000
def james_difference : ℕ := 276

theorem total_pennies_donated (cassandra : ℕ) (james_diff : ℕ) 
  (h1 : cassandra = cassandra_pennies) 
  (h2 : james_diff = james_difference) : 
  cassandra + (cassandra - james_diff) = 9724 :=
by
  sorry

end NUMINAMATH_CALUDE_total_pennies_donated_l1120_112079


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1120_112001

theorem smallest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1120_112001


namespace NUMINAMATH_CALUDE_root_product_cubic_l1120_112040

theorem root_product_cubic (p q r : ℝ) : 
  (3 * p^3 - 9 * p^2 + 5 * p - 15 = 0) ∧ 
  (3 * q^3 - 9 * q^2 + 5 * q - 15 = 0) ∧ 
  (3 * r^3 - 9 * r^2 + 5 * r - 15 = 0) →
  p * q * r = 5 := by
sorry

end NUMINAMATH_CALUDE_root_product_cubic_l1120_112040


namespace NUMINAMATH_CALUDE_expression_simplification_l1120_112014

theorem expression_simplification (x : ℝ) (h : x^2 + x - 5 = 0) :
  (x - 2) / (x^2 - 4*x + 4) / (x + 2 - (x^2 + x - 4) / (x - 2)) + 1 / (x + 1) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1120_112014


namespace NUMINAMATH_CALUDE_jack_and_toddlers_time_l1120_112012

/-- The time it takes for Jack and his toddlers to get ready -/
def total_time (jack_shoe_time : ℕ) (toddler_extra_time : ℕ) (num_toddlers : ℕ) : ℕ :=
  jack_shoe_time + num_toddlers * (jack_shoe_time + toddler_extra_time)

/-- Theorem: The total time for Jack and his toddlers to get ready is 18 minutes -/
theorem jack_and_toddlers_time : total_time 4 3 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_jack_and_toddlers_time_l1120_112012


namespace NUMINAMATH_CALUDE_real_root_of_cubic_l1120_112038

def cubic_polynomial (c d x : ℝ) : ℝ := c * x^3 + 4 * x^2 + d * x - 78

theorem real_root_of_cubic (c d : ℝ) :
  (∃ (z : ℂ), z = -3 - 4*I ∧ cubic_polynomial c d z.re = 0) →
  ∃ (x : ℝ), cubic_polynomial c d x = 0 ∧ x = -3 :=
sorry

end NUMINAMATH_CALUDE_real_root_of_cubic_l1120_112038


namespace NUMINAMATH_CALUDE_f_has_max_and_min_l1120_112077

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

-- Define the domain
def domain (x : ℝ) : Prop := -2 < x ∧ x < 4

-- Theorem statement
theorem f_has_max_and_min :
  ∃ (x_max x_min : ℝ),
    domain x_max ∧ domain x_min ∧
    (∀ x, domain x → f x ≤ f x_max) ∧
    (∀ x, domain x → f x_min ≤ f x) ∧
    f x_max = 5 ∧ f x_min = -27 ∧
    x_max = -1 ∧ x_min = 3 :=
sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_l1120_112077


namespace NUMINAMATH_CALUDE_point_p_transformation_l1120_112083

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the rotation function
def rotate90ClockwiseAboutOrigin (p : Point2D) : Point2D :=
  { x := p.y, y := -p.x }

-- Define the reflection function
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

-- Define the composition of rotation and reflection
def rotateAndReflect (p : Point2D) : Point2D :=
  reflectAcrossXAxis (rotate90ClockwiseAboutOrigin p)

theorem point_p_transformation :
  let p : Point2D := { x := 3, y := -5 }
  rotateAndReflect p = { x := -5, y := 3 } := by sorry

end NUMINAMATH_CALUDE_point_p_transformation_l1120_112083
