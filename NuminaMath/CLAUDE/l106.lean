import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_solution_correctness_l106_10602

/-- Solutions to the quadratic equation (p+1)x² - 2px + p - 2 = 0 --/
def quadratic_solutions (p : ℝ) : Set ℝ :=
  if p = -1 then
    {3/2}
  else if p > -2 then
    {(p + Real.sqrt (p+2)) / (p+1), (p - Real.sqrt (p+2)) / (p+1)}
  else if p = -2 then
    {2}
  else
    ∅

/-- The quadratic equation (p+1)x² - 2px + p - 2 = 0 --/
def quadratic_equation (p x : ℝ) : Prop :=
  (p+1) * x^2 - 2*p*x + p - 2 = 0

theorem quadratic_solution_correctness (p : ℝ) :
  ∀ x, x ∈ quadratic_solutions p ↔ quadratic_equation p x :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_correctness_l106_10602


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l106_10643

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a

theorem max_value_implies_a_equals_one :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 2, f a x ≤ 1) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 1) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l106_10643


namespace NUMINAMATH_CALUDE_discount_rate_inequality_l106_10651

/-- Represents the maximum discount rate that can be offered while ensuring a profit margin of at least 5% -/
def max_discount_rate (cost_price selling_price min_profit_margin : ℝ) : Prop :=
  ∃ x : ℝ, 
    0 ≤ x ∧ x ≤ 1 ∧
    selling_price * ((1 : ℝ) / 10) * x - cost_price ≥ min_profit_margin * cost_price

theorem discount_rate_inequality 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (min_profit_margin : ℝ) 
  (h1 : cost_price = 100)
  (h2 : selling_price = 150)
  (h3 : min_profit_margin = 0.05) :
  max_discount_rate cost_price selling_price min_profit_margin :=
sorry

end NUMINAMATH_CALUDE_discount_rate_inequality_l106_10651


namespace NUMINAMATH_CALUDE_solution_set_f_gt_2_range_m_common_points_l106_10634

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2*|x + 1|

-- Define the quadratic function g
def g (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Theorem 1: Solution set of f(x) > 2 when m = 5
theorem solution_set_f_gt_2 :
  {x : ℝ | f 5 x > 2} = Set.Ioo (-4/3 : ℝ) 0 := by sorry

-- Theorem 2: Range of m for which f and g always have common points
theorem range_m_common_points :
  {m : ℝ | ∀ y, ∃ x, f m x = y ∧ g x = y} = Set.Ici 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_2_range_m_common_points_l106_10634


namespace NUMINAMATH_CALUDE_circle_max_distance_l106_10623

/-- Given a circle with equation x^2 + y^2 + 4x - 2y - 4 = 0, 
    the maximum value of x^2 + y^2 is 14 + 6√5 -/
theorem circle_max_distance (x y : ℝ) : 
  x^2 + y^2 + 4*x - 2*y - 4 = 0 → 
  x^2 + y^2 ≤ 14 + 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_max_distance_l106_10623


namespace NUMINAMATH_CALUDE_negation_of_proposition_l106_10608

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x > 0 → ¬(x > 0)) ↔ ∃ x : ℝ, x ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l106_10608


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l106_10694

/-- The line passing through (-1, 0) and perpendicular to x + y = 0 has equation x - y + 1 = 0 -/
theorem perpendicular_line_equation : 
  let c : ℝ × ℝ := (-1, 0)
  let l₁ : Set (ℝ × ℝ) := {p | p.1 + p.2 = 0}
  let l₂ : Set (ℝ × ℝ) := {p | p.1 - p.2 + 1 = 0}
  (∀ p ∈ l₂, (p.1 - c.1) * (1 + 1) = -(p.2 - c.2)) ∧ 
  c ∈ l₂ :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l106_10694


namespace NUMINAMATH_CALUDE_f_at_6_l106_10653

/-- The polynomial f(x) = 3x^6 + 12x^5 + 8x^4 - 3.5x^3 + 7.2x^2 + 5x - 13 -/
def f (x : ℝ) : ℝ := 3*x^6 + 12*x^5 + 8*x^4 - 3.5*x^3 + 7.2*x^2 + 5*x - 13

/-- Theorem stating that f(6) = 243168.2 -/
theorem f_at_6 : f 6 = 243168.2 := by
  sorry

end NUMINAMATH_CALUDE_f_at_6_l106_10653


namespace NUMINAMATH_CALUDE_square_painting_size_l106_10674

/-- Given the total area of an art collection and the areas of non-square paintings,
    prove that the side length of each square painting is 6 feet. -/
theorem square_painting_size 
  (total_area : ℝ) 
  (num_square_paintings : ℕ) 
  (num_small_paintings : ℕ) 
  (small_painting_width small_painting_height : ℝ)
  (num_large_paintings : ℕ)
  (large_painting_width large_painting_height : ℝ) :
  total_area = 282 ∧ 
  num_square_paintings = 3 ∧
  num_small_paintings = 4 ∧
  small_painting_width = 2 ∧
  small_painting_height = 3 ∧
  num_large_paintings = 1 ∧
  large_painting_width = 10 ∧
  large_painting_height = 15 →
  ∃ (square_side : ℝ), 
    square_side = 6 ∧ 
    num_square_paintings * square_side^2 + 
    num_small_paintings * small_painting_width * small_painting_height +
    num_large_paintings * large_painting_width * large_painting_height = total_area :=
by sorry

end NUMINAMATH_CALUDE_square_painting_size_l106_10674


namespace NUMINAMATH_CALUDE_trash_can_count_l106_10633

theorem trash_can_count (x : ℕ) 
  (h1 : (x / 2 + 8) / 2 + x = 34) : x = 24 := by
  sorry

end NUMINAMATH_CALUDE_trash_can_count_l106_10633


namespace NUMINAMATH_CALUDE_tan_alpha_equals_two_tan_pi_fifth_l106_10603

theorem tan_alpha_equals_two_tan_pi_fifth (α : Real) 
  (h : Real.tan α = 2 * Real.tan (π / 5)) : 
  (Real.cos (α - 3 * π / 10)) / (Real.sin (α - π / 5)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_two_tan_pi_fifth_l106_10603


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l106_10650

theorem polar_coordinates_of_point (x y : ℝ) (ρ θ : ℝ) :
  x = 2 ∧ y = -2 →
  ρ = Real.sqrt (x^2 + y^2) ∧
  θ = -π/4 ∧
  x = ρ * Real.cos θ ∧
  y = ρ * Real.sin θ →
  ρ = 2 * Real.sqrt 2 ∧ θ = -π/4 := by
  sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l106_10650


namespace NUMINAMATH_CALUDE_sam_distance_l106_10679

/-- Proves that Sam drove 160 miles given the conditions of the problem -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ)
  (h1 : marguerite_distance = 120)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) :
  (marguerite_distance / marguerite_time) * sam_time = 160 :=
by sorry

end NUMINAMATH_CALUDE_sam_distance_l106_10679


namespace NUMINAMATH_CALUDE_no_real_solutions_ratio_equation_l106_10662

theorem no_real_solutions_ratio_equation :
  ∀ x : ℝ, (x + 3) / (2 * x + 5) ≠ (5 * x + 4) / (8 * x + 6) :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_ratio_equation_l106_10662


namespace NUMINAMATH_CALUDE_smallest_butterfly_count_l106_10600

theorem smallest_butterfly_count (n : ℕ) : n > 0 → (
  (∃ m k : ℕ, m > 0 ∧ k > 0 ∧ n * 44 = m * 17 ∧ n * 44 = k * 25) ∧
  (∃ t : ℕ, n * 44 + n * 17 + n * 25 = 60 * t) ∧
  (∀ x : ℕ, x > 0 ∧ x < n → 
    ¬(∃ y z : ℕ, y > 0 ∧ z > 0 ∧ x * 44 = y * 17 ∧ x * 44 = z * 25) ∨
    ¬(∃ s : ℕ, x * 44 + x * 17 + x * 25 = 60 * s))
) ↔ n = 425 := by
  sorry

end NUMINAMATH_CALUDE_smallest_butterfly_count_l106_10600


namespace NUMINAMATH_CALUDE_min_d_value_l106_10690

theorem min_d_value (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (horder : a < b ∧ b < c ∧ c < d)
  (hunique : ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a| + |x - b| + |x - c| + |x - d|) :
  d ≥ 999 ∧ ∃ (a' b' c' : ℕ), a' < b' ∧ b' < c' ∧ c' < 999 ∧
    ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a'| + |x - b'| + |x - c'| + |x - 999| :=
by
  sorry


end NUMINAMATH_CALUDE_min_d_value_l106_10690


namespace NUMINAMATH_CALUDE_dans_initial_money_l106_10606

/-- 
Given that Dan has some money, buys a candy bar for $1, and has $3 left afterwards,
prove that Dan's initial amount of money was $4.
-/
theorem dans_initial_money (money_left : ℕ) (candy_cost : ℕ) : 
  money_left = 3 → candy_cost = 1 → money_left + candy_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_dans_initial_money_l106_10606


namespace NUMINAMATH_CALUDE_cookies_per_pack_l106_10636

/-- Given that Candy baked four trays with 24 cookies each and divided them equally into eight packs,
    prove that the number of cookies in each pack is 12. -/
theorem cookies_per_pack :
  let num_trays : ℕ := 4
  let cookies_per_tray : ℕ := 24
  let num_packs : ℕ := 8
  let total_cookies : ℕ := num_trays * cookies_per_tray
  let cookies_per_pack : ℕ := total_cookies / num_packs
  cookies_per_pack = 12 := by sorry

end NUMINAMATH_CALUDE_cookies_per_pack_l106_10636


namespace NUMINAMATH_CALUDE_sum_four_digit_even_distinct_mod_1000_l106_10688

/-- A function that generates all four-digit positive integers with distinct even digits -/
def fourDigitEvenDistinct : List Nat := sorry

/-- The sum of all four-digit positive integers with distinct even digits -/
def sumFourDigitEvenDistinct : Nat := (fourDigitEvenDistinct.map id).sum

/-- Theorem: The sum of all four-digit positive integers with distinct even digits,
    when divided by 1000, leaves a remainder of 560 -/
theorem sum_four_digit_even_distinct_mod_1000 :
  sumFourDigitEvenDistinct % 1000 = 560 := by sorry

end NUMINAMATH_CALUDE_sum_four_digit_even_distinct_mod_1000_l106_10688


namespace NUMINAMATH_CALUDE_fans_attended_l106_10605

def stadium_capacity : ℕ := 60000
def seats_sold_percentage : ℚ := 75 / 100
def fans_stayed_home : ℕ := 5000

theorem fans_attended (capacity : ℕ) (sold_percentage : ℚ) (stayed_home : ℕ) 
  (h1 : capacity = stadium_capacity)
  (h2 : sold_percentage = seats_sold_percentage)
  (h3 : stayed_home = fans_stayed_home) :
  (capacity : ℚ) * sold_percentage - stayed_home = 40000 := by
  sorry

end NUMINAMATH_CALUDE_fans_attended_l106_10605


namespace NUMINAMATH_CALUDE_sequence_inequality_l106_10618

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℝ := 3 * n - 2 * n^2

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℝ := S n - S (n - 1)

theorem sequence_inequality (n : ℕ) (h : n ≥ 2) :
  n * (a 1) > S n ∧ S n > n * (a n) := by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l106_10618


namespace NUMINAMATH_CALUDE_log_216_equals_3log2_plus_3log3_l106_10616

theorem log_216_equals_3log2_plus_3log3 : Real.log 216 = 3 * Real.log 2 + 3 * Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_log_216_equals_3log2_plus_3log3_l106_10616


namespace NUMINAMATH_CALUDE_seven_mult_five_equals_34_l106_10673

/-- Custom multiplication operation -/
def custom_mult (A B : ℝ) : ℝ := (A + 2*B) * (A - B)

/-- Theorem stating that 7 * 5 = 34 under the custom multiplication -/
theorem seven_mult_five_equals_34 : custom_mult 7 5 = 34 := by
  sorry

end NUMINAMATH_CALUDE_seven_mult_five_equals_34_l106_10673


namespace NUMINAMATH_CALUDE_base_conversion_three_digits_l106_10691

theorem base_conversion_three_digits : 
  ∃ (b : ℕ), b > 1 ∧ b^2 ≤ 256 ∧ 256 < b^3 ∧ ∀ (x : ℕ), 1 < x ∧ x < b → (x^2 > 256 ∨ x^3 ≤ 256) :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_three_digits_l106_10691


namespace NUMINAMATH_CALUDE_shaded_area_is_600_l106_10659

-- Define the vertices of the rectangle
def rectangle_vertices : List (ℝ × ℝ) := [(0, 0), (40, 0), (40, 20), (0, 20)]

-- Define the vertices of the shaded polygon
def polygon_vertices : List (ℝ × ℝ) := [(0, 0), (20, 0), (40, 10), (40, 20), (10, 20)]

-- Function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem shaded_area_is_600 :
  polygon_area polygon_vertices = 600 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_600_l106_10659


namespace NUMINAMATH_CALUDE_linear_function_slope_l106_10697

/-- Given a linear function y = 2x - kx + 1 and two distinct points on its graph,
    if the product of differences is negative, then k > 2 -/
theorem linear_function_slope (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : 
  y₁ = 2*x₁ - k*x₁ + 1 →
  y₂ = 2*x₂ - k*x₂ + 1 →
  x₁ ≠ x₂ →
  (x₁ - x₂) * (y₁ - y₂) < 0 →
  k > 2 := by
sorry

end NUMINAMATH_CALUDE_linear_function_slope_l106_10697


namespace NUMINAMATH_CALUDE_larger_integer_value_l106_10614

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  max a b = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l106_10614


namespace NUMINAMATH_CALUDE_yellow_pairs_l106_10647

theorem yellow_pairs (total_students : ℕ) (blue_students : ℕ) (yellow_students : ℕ) 
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  total_students = 132 →
  blue_students = 57 →
  yellow_students = 75 →
  total_pairs = 66 →
  blue_blue_pairs = 23 →
  blue_students + yellow_students = total_students →
  2 * total_pairs = total_students →
  ∃ (yellow_yellow_pairs : ℕ),
    yellow_yellow_pairs = 32 ∧
    yellow_yellow_pairs + blue_blue_pairs + (total_pairs - yellow_yellow_pairs - blue_blue_pairs) = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_yellow_pairs_l106_10647


namespace NUMINAMATH_CALUDE_probability_of_specific_draw_l106_10641

def total_silverware : ℕ := 24
def forks : ℕ := 8
def spoons : ℕ := 8
def knives : ℕ := 8
def pieces_drawn : ℕ := 4

def favorable_outcomes : ℕ := forks * spoons * knives * (forks - 1 + spoons - 1 + knives - 1)
def total_outcomes : ℕ := Nat.choose total_silverware pieces_drawn

theorem probability_of_specific_draw :
  (favorable_outcomes : ℚ) / total_outcomes = 214 / 253 := by sorry

end NUMINAMATH_CALUDE_probability_of_specific_draw_l106_10641


namespace NUMINAMATH_CALUDE_emails_left_l106_10612

def process_emails (initial : ℕ) : ℕ :=
  let after_trash := initial / 2
  let after_work := after_trash - (after_trash * 2 / 5)
  let after_personal := after_work - (after_work / 4)
  let after_misc := after_personal - (after_personal / 10)
  let after_subfolder := after_misc - (after_misc * 3 / 10)
  after_subfolder - (after_subfolder / 5)

theorem emails_left (initial : ℕ) (h : initial = 600) : process_emails initial = 69 := by
  sorry

end NUMINAMATH_CALUDE_emails_left_l106_10612


namespace NUMINAMATH_CALUDE_riley_mistakes_l106_10648

theorem riley_mistakes (total_questions : Nat) (team_incorrect : Nat) (ofelia_bonus : Nat) :
  total_questions = 35 →
  team_incorrect = 17 →
  ofelia_bonus = 5 →
  ∃ (riley_mistakes : Nat),
    riley_mistakes + (35 - ((35 - riley_mistakes) / 2 + ofelia_bonus)) = team_incorrect ∧
    riley_mistakes = 3 :=
by sorry

end NUMINAMATH_CALUDE_riley_mistakes_l106_10648


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_for_nonempty_solution_l106_10617

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |2*x - 4|

-- Theorem 1: Solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = Set.Ioo (5/3) 3 := by sorry

-- Theorem 2: Range of t for non-empty solution set of f(x) > t^2 + 2t
theorem range_of_t_for_nonempty_solution :
  ∀ t : ℝ, (∃ x : ℝ, f x > t^2 + 2*t) ↔ t ∈ Set.Ioo (-3) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_for_nonempty_solution_l106_10617


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l106_10620

theorem diophantine_equation_solution :
  ∃ (a b c d : ℕ), 5 * a + 6 * b + 7 * c + 11 * d = 1999 ∧
  a = 389 ∧ b = 2 ∧ c = 1 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l106_10620


namespace NUMINAMATH_CALUDE_smallest_a_for_nonempty_solution_l106_10655

theorem smallest_a_for_nonempty_solution : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (∃ x : ℝ, 2*|x-3| + |x-4| < a^2 + a) ∧
  (∀ b : ℕ, (b > 0 ∧ b < a) → ¬∃ x : ℝ, 2*|x-3| + |x-4| < b^2 + b) ∧
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_for_nonempty_solution_l106_10655


namespace NUMINAMATH_CALUDE_theoretical_yield_NaNO3_l106_10670

/-- Theoretical yield of NaNO3 given initial conditions and overall yield -/
theorem theoretical_yield_NaNO3 (initial_NH4NO3 : ℝ) (initial_NaOH : ℝ) (percent_yield : ℝ) :
  initial_NH4NO3 = 2 →
  initial_NaOH = 2 →
  percent_yield = 0.85 →
  ∃ (theoretical_yield : ℝ),
    theoretical_yield = 289 ∧
    theoretical_yield = initial_NH4NO3 * 2 * 85 * percent_yield :=
by sorry

/-- Molar mass of NaNO3 in g/mol -/
def molar_mass_NaNO3 : ℝ := 85

/-- Theoretical yield in moles of NaNO3 -/
def theoretical_yield_moles (initial_NH4NO3 : ℝ) : ℝ := initial_NH4NO3 * 2

/-- Theoretical yield in grams of NaNO3 -/
def theoretical_yield_grams (theoretical_yield_moles : ℝ) : ℝ :=
  theoretical_yield_moles * molar_mass_NaNO3

/-- Actual yield in grams of NaNO3 considering percent yield -/
def actual_yield_grams (theoretical_yield_grams : ℝ) (percent_yield : ℝ) : ℝ :=
  theoretical_yield_grams * percent_yield

end NUMINAMATH_CALUDE_theoretical_yield_NaNO3_l106_10670


namespace NUMINAMATH_CALUDE_min_value_expression_l106_10638

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ≥ 11 / 16 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 2 ∧
    (4 / (x₀ + 2) + (3 * x₀ - 7) / (3 * y₀ + 4)) = 11 / 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l106_10638


namespace NUMINAMATH_CALUDE_special_tetrahedron_ratio_bounds_l106_10681

/-- Represents a tetrahedron with specific edge length properties -/
structure SpecialTetrahedron where
  -- The edge lengths
  a : ℝ
  b : ℝ
  -- Conditions on edge lengths
  h_positive : 0 < a ∧ 0 < b
  h_pa_eq_pb : true  -- Represents PA = PB = a
  h_pc_eq_sides : true  -- Represents PC = AB = BC = CA = b
  h_a_lt_b : a < b

/-- The ratio a/b in a special tetrahedron is bounded -/
theorem special_tetrahedron_ratio_bounds (t : SpecialTetrahedron) :
  Real.sqrt (2 - Real.sqrt 3) < t.a / t.b ∧ t.a / t.b < 1 := by
  sorry


end NUMINAMATH_CALUDE_special_tetrahedron_ratio_bounds_l106_10681


namespace NUMINAMATH_CALUDE_translation_result_l106_10631

def initial_point : ℝ × ℝ := (-4, 3)
def translation : ℝ × ℝ := (-2, -2)

def translate (p : ℝ × ℝ) (t : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + t.1, p.2 + t.2)

theorem translation_result :
  translate initial_point translation = (-6, 1) := by sorry

end NUMINAMATH_CALUDE_translation_result_l106_10631


namespace NUMINAMATH_CALUDE_reciprocal_of_golden_ratio_l106_10666

theorem reciprocal_of_golden_ratio (φ : ℝ) :
  φ = (Real.sqrt 5 + 1) / 2 →
  1 / φ = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_golden_ratio_l106_10666


namespace NUMINAMATH_CALUDE_system_of_equations_l106_10652

theorem system_of_equations (x y k : ℝ) 
  (eq1 : 3 * x + 4 * y = k + 2)
  (eq2 : 2 * x + y = 4)
  (eq3 : x + y = 2) : k = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l106_10652


namespace NUMINAMATH_CALUDE_first_chapter_has_13_pages_l106_10658

/-- Represents a book with chapters of increasing length -/
structure Book where
  num_chapters : ℕ
  total_pages : ℕ
  page_increase : ℕ

/-- Calculates the number of pages in the first chapter of a book -/
def first_chapter_pages (b : Book) : ℕ :=
  let x := (b.total_pages - (b.num_chapters * (b.num_chapters - 1) * b.page_increase / 2)) / b.num_chapters
  x

/-- Theorem stating that for a specific book, the first chapter has 13 pages -/
theorem first_chapter_has_13_pages :
  let b : Book := { num_chapters := 5, total_pages := 95, page_increase := 3 }
  first_chapter_pages b = 13 := by
  sorry


end NUMINAMATH_CALUDE_first_chapter_has_13_pages_l106_10658


namespace NUMINAMATH_CALUDE_min_money_for_city_l106_10624

/-- Represents the resources needed to build a city -/
structure CityResources where
  ore : ℕ
  wheat : ℕ

/-- Represents the market prices and exchange rates -/
structure MarketPrices where
  ore_price : ℕ
  wheat_bundle_price : ℕ
  wheat_bundle_size : ℕ
  wheat_to_ore_rate : ℕ

/-- The problem setup -/
def city_building_problem (work_days : ℕ) (daily_ore_production : ℕ) 
  (city_resources : CityResources) (market_prices : MarketPrices) : Prop :=
  ∃ (initial_money : ℕ),
    initial_money = 9 ∧
    work_days * daily_ore_production + 
    (market_prices.wheat_bundle_size - city_resources.wheat) = 
    city_resources.ore ∧
    initial_money + 
    (work_days * daily_ore_production - city_resources.ore) * market_prices.ore_price = 
    (market_prices.wheat_bundle_size / city_resources.wheat) * market_prices.wheat_bundle_price

/-- The theorem to be proved -/
theorem min_money_for_city : 
  city_building_problem 3 1 
    { ore := 3, wheat := 2 } 
    { ore_price := 3, 
      wheat_bundle_price := 12, 
      wheat_bundle_size := 3, 
      wheat_to_ore_rate := 1 } :=
by
  sorry


end NUMINAMATH_CALUDE_min_money_for_city_l106_10624


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l106_10610

/-- Given a stamp price of 28 cents and a budget of 3600 cents,
    the maximum number of stamps that can be purchased is 128. -/
theorem max_stamps_purchasable (stamp_price : ℕ) (budget : ℕ) :
  stamp_price = 28 → budget = 3600 → 
  (∃ (n : ℕ), n * stamp_price ≤ budget ∧ 
    ∀ (m : ℕ), m * stamp_price ≤ budget → m ≤ n) →
  (∃ (max_stamps : ℕ), max_stamps = 128) :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l106_10610


namespace NUMINAMATH_CALUDE_congruence_problem_l106_10680

def binomial_sum (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (λ k => Nat.choose n k * 2^k)

theorem congruence_problem (a b : ℤ) :
  a = binomial_sum 20 ∧ a ≡ b [ZMOD 10] → b = 2011 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l106_10680


namespace NUMINAMATH_CALUDE_carrots_theorem_l106_10675

/-- The number of carrots Sandy grew -/
def sandy_carrots : ℕ := 6

/-- The number of carrots Sam grew -/
def sam_carrots : ℕ := 3

/-- The total number of carrots grown -/
def total_carrots : ℕ := sandy_carrots + sam_carrots

theorem carrots_theorem : total_carrots = 9 := by sorry

end NUMINAMATH_CALUDE_carrots_theorem_l106_10675


namespace NUMINAMATH_CALUDE_first_five_terms_sum_l106_10692

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem first_five_terms_sum :
  let a : ℚ := 1
  let r : ℚ := 1/2
  let n : ℕ := 5
  geometric_series_sum a r n = 31/16 := by
sorry

end NUMINAMATH_CALUDE_first_five_terms_sum_l106_10692


namespace NUMINAMATH_CALUDE_school_governor_election_votes_l106_10657

theorem school_governor_election_votes (elvis_votes : ℕ) (elvis_percentage : ℚ) 
  (h1 : elvis_votes = 45)
  (h2 : elvis_percentage = 1/4)
  (h3 : elvis_votes = elvis_percentage * total_votes) :
  total_votes = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_school_governor_election_votes_l106_10657


namespace NUMINAMATH_CALUDE_reflect_h_twice_l106_10628

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflects a point across the line y = x - 2 -/
def reflect_y_eq_x_minus_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  let q := (p.1, p.2 + 2)  -- Translate up by 2
  let r := (q.2, q.1)      -- Reflect across y = x
  (r.1, r.2 - 2)           -- Translate down by 2

theorem reflect_h_twice (h : ℝ × ℝ) :
  h = (5, 3) →
  reflect_y_eq_x_minus_2 (reflect_x h) = (-1, 3) := by
  sorry

end NUMINAMATH_CALUDE_reflect_h_twice_l106_10628


namespace NUMINAMATH_CALUDE_initial_amount_proof_l106_10625

theorem initial_amount_proof (remaining_amount : ℝ) (spent_percentage : ℝ) (initial_amount : ℝ) : 
  remaining_amount = 3500 ∧ 
  spent_percentage = 30 ∧ 
  remaining_amount = initial_amount * (1 - spent_percentage / 100) → 
  initial_amount = 5000 := by
sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l106_10625


namespace NUMINAMATH_CALUDE_cubic_factorization_l106_10611

theorem cubic_factorization (x : ℝ) : x^3 - 4*x^2 + 4*x = x*(x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l106_10611


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l106_10604

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 2) / (3 * x^3))^2) = (Real.sqrt (x^12 + 5*x^6 + 4)) / (3 * x^3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l106_10604


namespace NUMINAMATH_CALUDE_malaria_parasite_length_l106_10642

theorem malaria_parasite_length : 0.0000015 = 1.5 * 10^(-6) := by
  sorry

end NUMINAMATH_CALUDE_malaria_parasite_length_l106_10642


namespace NUMINAMATH_CALUDE_profit_maximized_at_95_l106_10660

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ := -20 * x^2 + 200 * x + 4000

/-- Theorem stating that the profit is maximized at a selling price of 95 yuan -/
theorem profit_maximized_at_95 :
  let initial_purchase_price : ℝ := 80
  let initial_selling_price : ℝ := 90
  let initial_sales_volume : ℝ := 400
  let price_increase_rate : ℝ := 1
  let sales_decrease_rate : ℝ := 20
  ∃ (max_profit : ℝ), 
    (∀ x, profit_function x ≤ max_profit) ∧ 
    (profit_function 5 = max_profit) ∧
    (initial_selling_price + 5 = 95) := by
  sorry

#check profit_maximized_at_95

end NUMINAMATH_CALUDE_profit_maximized_at_95_l106_10660


namespace NUMINAMATH_CALUDE_calculate_marked_price_jobber_marked_price_l106_10669

/-- Calculate the marked price of an article given the original price, purchase discount,
    desired profit margin, and selling discount. -/
theorem calculate_marked_price (original_price : ℝ) (purchase_discount : ℝ) 
    (profit_margin : ℝ) (selling_discount : ℝ) : ℝ :=
  let purchase_price := original_price * (1 - purchase_discount)
  let desired_selling_price := purchase_price * (1 + profit_margin)
  desired_selling_price / (1 - selling_discount)

/-- The marked price of the article should be $50.00 -/
theorem jobber_marked_price : 
  calculate_marked_price 40 0.25 0.5 0.1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_calculate_marked_price_jobber_marked_price_l106_10669


namespace NUMINAMATH_CALUDE_function_property_l106_10686

/-- Given a function f(x) = (ln x - k - 1)x where k is a real number and x > 1,
    prove that if x₁ ≠ x₂ and f(x₁) = f(x₂), then x₁x₂ < e^(2k) -/
theorem function_property (k : ℝ) :
  let f : ℝ → ℝ := λ x => (Real.log x - k - 1) * x
  ∀ x₁ x₂, x₁ > 1 → x₂ > 1 → x₁ ≠ x₂ → f x₁ = f x₂ → x₁ * x₂ < Real.exp (2 * k) := by
  sorry


end NUMINAMATH_CALUDE_function_property_l106_10686


namespace NUMINAMATH_CALUDE_length_of_24_l106_10664

def length_of_integer (k : ℕ) : ℕ := sorry

theorem length_of_24 : 
  let k : ℕ := 24
  length_of_integer k = 4 := by sorry

end NUMINAMATH_CALUDE_length_of_24_l106_10664


namespace NUMINAMATH_CALUDE_f_not_even_l106_10637

def f (x : ℝ) := x^2 + x

theorem f_not_even : ¬(∀ x : ℝ, f x = f (-x)) := by
  sorry

end NUMINAMATH_CALUDE_f_not_even_l106_10637


namespace NUMINAMATH_CALUDE_solution_set_f_solution_set_g_l106_10677

-- Define the quadratic functions
def f (x : ℝ) : ℝ := x^2 - 3*x - 4
def g (x : ℝ) : ℝ := x^2 - x - 6

-- Define the solution sets
def S₁ : Set ℝ := {x | -1 < x ∧ x < 4}
def S₂ : Set ℝ := {x | x < -2 ∨ x > 3}

-- Theorem for the first inequality
theorem solution_set_f : {x : ℝ | f x < 0} = S₁ := by sorry

-- Theorem for the second inequality
theorem solution_set_g : {x : ℝ | g x > 0} = S₂ := by sorry

end NUMINAMATH_CALUDE_solution_set_f_solution_set_g_l106_10677


namespace NUMINAMATH_CALUDE_motion_equation_l106_10646

/-- Given V = gt + V₀ and S = (1/2)gt² + V₀t + kt³, 
    prove that t = (2S(V-V₀)) / (V² - V₀² + 2k(V-V₀)²) -/
theorem motion_equation (g k V V₀ S t : ℝ) 
  (hV : V = g * t + V₀)
  (hS : S = (1/2) * g * t^2 + V₀ * t + k * t^3) :
  t = (2 * S * (V - V₀)) / (V^2 - V₀^2 + 2 * k * (V - V₀)^2) :=
by sorry

end NUMINAMATH_CALUDE_motion_equation_l106_10646


namespace NUMINAMATH_CALUDE_problem_solution_l106_10663

theorem problem_solution (a b : ℝ) (h1 : 4 + a = 5 - b) (h2 : 5 + b = 8 + a) : 4 - a = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l106_10663


namespace NUMINAMATH_CALUDE_fine_payment_l106_10635

theorem fine_payment (F : ℚ) 
  (joe_payment : ℚ) (peter_payment : ℚ) (sam_payment : ℚ)
  (h1 : joe_payment = F / 4 + 7)
  (h2 : peter_payment = F / 3 - 7)
  (h3 : sam_payment = F / 2 - 12)
  (h4 : joe_payment + peter_payment + sam_payment = F) :
  sam_payment / F = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fine_payment_l106_10635


namespace NUMINAMATH_CALUDE_max_k_value_l106_10626

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8 = 0

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the condition for a point to be a valid center
def valid_center (k : ℝ) (x y : ℝ) : Prop :=
  line k x y ∧ ∃ (x' y' : ℝ), circle_C x' y' ∧ (x - x')^2 + (y - y')^2 = 1

-- Theorem statement
theorem max_k_value :
  (∃ k : ℝ, ∀ k' : ℝ, (∃ x y : ℝ, valid_center k' x y) → k' ≤ k) ∧
  (∃ x y : ℝ, valid_center (12/5) x y) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l106_10626


namespace NUMINAMATH_CALUDE_eight_students_pairing_l106_10668

theorem eight_students_pairing :
  (Nat.factorial 8) / ((Nat.factorial 4) * (2^4)) = 105 := by
  sorry

end NUMINAMATH_CALUDE_eight_students_pairing_l106_10668


namespace NUMINAMATH_CALUDE_equation_solution_l106_10601

theorem equation_solution :
  let f (x : ℝ) := 1 / (x + 8) + 1 / (x + 5) - 1 / (x + 11) - 1 / (x + 4)
  ∀ x : ℝ, f x = 0 ↔ x = (-3 + Real.sqrt 37) / 2 ∨ x = (-3 - Real.sqrt 37) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l106_10601


namespace NUMINAMATH_CALUDE_cubic_root_sum_of_squares_reciprocal_l106_10629

theorem cubic_root_sum_of_squares_reciprocal (a b c : ℝ) : 
  a^3 - 12*a^2 + 20*a - 3 = 0 → 
  b^3 - 12*b^2 + 20*b - 3 = 0 → 
  c^3 - 12*c^2 + 20*c - 3 = 0 → 
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 328/9 := by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_of_squares_reciprocal_l106_10629


namespace NUMINAMATH_CALUDE_correct_calculation_l106_10644

theorem correct_calculation (x y : ℝ) : 3 * x * y - 2 * y * x = x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l106_10644


namespace NUMINAMATH_CALUDE_frisbee_throwing_problem_l106_10649

/-- Frisbee throwing problem -/
theorem frisbee_throwing_problem 
  (bess_distance : ℝ) 
  (bess_throws : ℕ) 
  (holly_throws : ℕ) 
  (total_distance : ℝ) 
  (h1 : bess_distance = 20)
  (h2 : bess_throws = 4)
  (h3 : holly_throws = 5)
  (h4 : total_distance = 200)
  (h5 : bess_distance * bess_throws * 2 + holly_throws * holly_distance = total_distance) :
  holly_distance = 8 := by
  sorry


end NUMINAMATH_CALUDE_frisbee_throwing_problem_l106_10649


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l106_10622

-- Define the displacement function
def S (t : ℝ) : ℝ := 3 * t - t^2

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 3 - 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_2 : v 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l106_10622


namespace NUMINAMATH_CALUDE_min_value_of_f_l106_10632

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x ∈ Set.Icc (-1) 0, f x ≥ m) ∧ (∃ x ∈ Set.Icc (-1) 0, f x = m) ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l106_10632


namespace NUMINAMATH_CALUDE_council_vote_change_l106_10699

theorem council_vote_change (total : ℕ) (initial_for initial_against : ℕ) 
  (revote_for revote_against : ℕ) :
  total = 350 →
  initial_for + initial_against = total →
  initial_against > initial_for →
  revote_for + revote_against = total →
  revote_for > revote_against →
  (revote_for - revote_against) = 2 * (initial_against - initial_for) →
  revote_for = (10 * initial_against) / 9 →
  revote_for - initial_for = 66 := by
  sorry

end NUMINAMATH_CALUDE_council_vote_change_l106_10699


namespace NUMINAMATH_CALUDE_expansion_coefficient_l106_10693

/-- The coefficient of x^5 in the expansion of (2x-√x)^8 -/
def coefficient_x5 : ℕ := 112

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem expansion_coefficient :
  coefficient_x5 = (binomial 8 6) * 2^2 :=
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l106_10693


namespace NUMINAMATH_CALUDE_price_tags_offer_advantages_l106_10667

/-- Represents a product in a store -/
structure Product where
  name : String
  price : ℝ

/-- Represents a store with a collection of products -/
structure Store where
  products : List Product
  has_price_tags : Bool

/-- Represents the advantages of using price tags -/
structure PriceTagAdvantages where
  simplifies_purchase : Bool
  reduces_personnel_requirement : Bool
  provides_advertising : Bool
  increases_trust : Bool

/-- Theorem stating that attaching price tags to all products offers advantages -/
theorem price_tags_offer_advantages (store : Store) (h : store.has_price_tags = true) :
  ∃ (advantages : PriceTagAdvantages),
    advantages.simplifies_purchase ∧
    advantages.reduces_personnel_requirement ∧
    advantages.provides_advertising ∧
    advantages.increases_trust :=
  sorry

end NUMINAMATH_CALUDE_price_tags_offer_advantages_l106_10667


namespace NUMINAMATH_CALUDE_razorback_revenue_per_shirt_l106_10682

/-- Razorback t-shirt shop sales data -/
structure TShirtSales where
  total_shirts : ℕ
  game_shirts : ℕ
  game_revenue : ℕ

/-- Calculate the revenue per t-shirt -/
def revenue_per_shirt (sales : TShirtSales) : ℚ :=
  sales.game_revenue / sales.game_shirts

/-- Theorem: The revenue per t-shirt is $98 -/
theorem razorback_revenue_per_shirt :
  let sales : TShirtSales := {
    total_shirts := 163,
    game_shirts := 89,
    game_revenue := 8722
  }
  revenue_per_shirt sales = 98 := by
  sorry

end NUMINAMATH_CALUDE_razorback_revenue_per_shirt_l106_10682


namespace NUMINAMATH_CALUDE_round_trip_percentage_l106_10678

/-- Represents the distribution of passenger types and classes on a transatlantic ship crossing -/
structure PassengerDistribution where
  /-- Percentage of Type A passengers (round-trip with car) -/
  type_a_percent : ℝ
  /-- Percentage of round-trip passengers not taking cars -/
  no_car_percent : ℝ
  /-- Percentage of round-trip passengers in luxury class -/
  luxury_percent : ℝ
  /-- Percentage of round-trip passengers in economy class -/
  economy_percent : ℝ
  /-- Percentage of Type C passengers in economy class -/
  type_c_economy_percent : ℝ

/-- Theorem stating that given the passenger distribution, the percentage of round-trip passengers is 40% -/
theorem round_trip_percentage (pd : PassengerDistribution)
  (h1 : pd.type_a_percent = 0.2)
  (h2 : pd.no_car_percent = 0.5)
  (h3 : pd.luxury_percent = 0.3)
  (h4 : pd.economy_percent = 0.7)
  (h5 : pd.type_c_economy_percent = 0.4)
  : ℝ :=
  by sorry

end NUMINAMATH_CALUDE_round_trip_percentage_l106_10678


namespace NUMINAMATH_CALUDE_sin_675_degrees_l106_10672

theorem sin_675_degrees :
  Real.sin (675 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_675_degrees_l106_10672


namespace NUMINAMATH_CALUDE_expression_evaluation_l106_10627

theorem expression_evaluation :
  let f (x : ℝ) := (x^2 - 5*x + 6) / (x - 2)
  f 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l106_10627


namespace NUMINAMATH_CALUDE_investment_revenue_difference_l106_10621

def banks_investments : ℕ := 8
def banks_revenue_per_investment : ℕ := 500
def elizabeth_investments : ℕ := 5
def elizabeth_revenue_per_investment : ℕ := 900

theorem investment_revenue_difference :
  elizabeth_investments * elizabeth_revenue_per_investment - 
  banks_investments * banks_revenue_per_investment = 500 := by
sorry

end NUMINAMATH_CALUDE_investment_revenue_difference_l106_10621


namespace NUMINAMATH_CALUDE_angle_conversion_l106_10609

theorem angle_conversion (angle : Real) : ∃ (k : Int) (α : Real), 
  angle = k * (2 * Real.pi) + α ∧ 
  0 ≤ α ∧ 
  α < 2 * Real.pi ∧ 
  angle = -1125 * (Real.pi / 180) ∧ 
  angle = -8 * Real.pi + 7 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_conversion_l106_10609


namespace NUMINAMATH_CALUDE_non_mundane_primes_characterization_l106_10687

/-- A prime number is mundane if there exist positive integers a and b less than p/2 
    such that (ab - 1)/p is a positive integer. -/
def IsMundane (p : ℕ) : Prop :=
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a < p / 2 ∧ b < p / 2 ∧ 
  ∃ k : ℕ, 0 < k ∧ k * p = a * b - 1

/-- The set of non-mundane primes -/
def NonMundanePrimes : Set ℕ := {2, 3, 5, 7, 13}

/-- Theorem: A prime number is not mundane if and only if it belongs to the set {2, 3, 5, 7, 13} -/
theorem non_mundane_primes_characterization (p : ℕ) (hp : Nat.Prime p) : 
  ¬ IsMundane p ↔ p ∈ NonMundanePrimes := by
  sorry

end NUMINAMATH_CALUDE_non_mundane_primes_characterization_l106_10687


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l106_10698

/-- The line x + y = 2k is tangent to the circle x^2 + y^2 = 4k if and only if k = 2 -/
theorem line_tangent_to_circle (k : ℝ) : 
  (∀ x y : ℝ, x + y = 2 * k → x^2 + y^2 = 4 * k) ↔ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l106_10698


namespace NUMINAMATH_CALUDE_inclined_line_and_triangle_l106_10695

/-- A line passing through a point with a given angle of inclination -/
structure InclinedLine where
  point : ℝ × ℝ
  angle : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about an inclined line and the triangle it forms with the axes -/
theorem inclined_line_and_triangle (l : InclinedLine) 
    (h1 : l.point = (0, -2))
    (h2 : l.angle = Real.pi / 3) : 
  ∃ (eq : LineEquation) (area : ℝ),
    eq.a = Real.sqrt 3 ∧ 
    eq.b = -1 ∧ 
    eq.c = -2 ∧
    area = (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inclined_line_and_triangle_l106_10695


namespace NUMINAMATH_CALUDE_square_root_sum_implies_product_l106_10656

theorem square_root_sum_implies_product (x : ℝ) :
  (Real.sqrt (9 + x) + Real.sqrt (16 - x) = 8) →
  ((9 + x) * (16 - x) = 380.25) :=
by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_implies_product_l106_10656


namespace NUMINAMATH_CALUDE_roots_form_parallelogram_l106_10685

/-- The polynomial whose roots we're investigating -/
def f (b : ℝ) (z : ℂ) : ℂ := z^4 - 8*z^3 + 17*b*z^2 - 2*(3*b^2 + 5*b - 4)*z + 2

/-- Predicate to check if a set of complex numbers forms a parallelogram -/
def forms_parallelogram (s : Set ℂ) : Prop :=
  ∃ (z₁ z₂ z₃ z₄ : ℂ), s = {z₁, z₂, z₃, z₄} ∧ 
    z₁ + z₃ = z₂ + z₄ ∧ z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄

/-- The main theorem stating the condition for the roots to form a parallelogram -/
theorem roots_form_parallelogram (b : ℝ) :
  forms_parallelogram {z : ℂ | f b z = 0} ↔ b = 1 ∨ b = 5/2 :=
sorry

end NUMINAMATH_CALUDE_roots_form_parallelogram_l106_10685


namespace NUMINAMATH_CALUDE_coin_coverage_probability_l106_10639

theorem coin_coverage_probability (square_side : ℝ) (triangle_leg : ℝ) (diamond_side : ℝ) (coin_diameter : ℝ) :
  square_side = 10 →
  triangle_leg = 3 →
  diamond_side = 3 * Real.sqrt 2 →
  coin_diameter = 2 →
  let coin_radius : ℝ := coin_diameter / 2
  let landing_area : ℝ := (square_side - 2 * coin_radius) ^ 2
  let triangle_area : ℝ := 4 * (triangle_leg ^ 2 / 2 + π * coin_radius ^ 2 / 4 + triangle_leg * coin_radius)
  let diamond_area : ℝ := diamond_side ^ 2 + 4 * (π * coin_radius ^ 2 / 4 + diamond_side * coin_radius / Real.sqrt 2)
  let total_black_area : ℝ := triangle_area + diamond_area
  let probability : ℝ := total_black_area / landing_area
  probability = (1 / 225) * (900 + 300 * Real.sqrt 2 + π) := by
    sorry

end NUMINAMATH_CALUDE_coin_coverage_probability_l106_10639


namespace NUMINAMATH_CALUDE_roses_in_vase_l106_10615

-- Define the initial number of roses and orchids
def initial_roses : ℕ := 5
def initial_orchids : ℕ := 3

-- Define the current number of orchids
def current_orchids : ℕ := 2

-- Define the difference between roses and orchids
def rose_orchid_difference : ℕ := 10

-- Theorem to prove
theorem roses_in_vase :
  ∃ (current_roses : ℕ),
    current_roses = current_orchids + rose_orchid_difference ∧
    current_roses > initial_roses ∧
    current_roses = 12 :=
by sorry

end NUMINAMATH_CALUDE_roses_in_vase_l106_10615


namespace NUMINAMATH_CALUDE_sale_to_cost_ratio_l106_10607

/-- Given an article with a cost price, sale price, and profit, prove that if the ratio of profit to cost price is 2, then the ratio of sale price to cost price is 3. -/
theorem sale_to_cost_ratio (cost_price sale_price profit : ℝ) 
  (h_positive : cost_price > 0)
  (h_profit_ratio : profit / cost_price = 2)
  (h_profit_def : profit = sale_price - cost_price) :
  sale_price / cost_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_sale_to_cost_ratio_l106_10607


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l106_10671

theorem largest_n_divisibility : ∃ (n : ℕ), 
  (∀ m : ℕ, m > n → ¬((m + 20) ∣ (m^3 + 200))) ∧ 
  ((n + 20) ∣ (n^3 + 200)) ∧ 
  n = 7780 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l106_10671


namespace NUMINAMATH_CALUDE_knight_seating_probability_correct_l106_10683

/-- The probability of three knights seated at a round table with n chairs
    such that each knight has an empty chair on both sides. -/
def knight_seating_probability (n : ℕ) : ℚ :=
  if n ≥ 6 then
    (n - 4) * (n - 5) / ((n - 1) * (n - 2))
  else
    0

theorem knight_seating_probability_correct (n : ℕ) (h : n ≥ 6) :
  knight_seating_probability n =
    (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_knight_seating_probability_correct_l106_10683


namespace NUMINAMATH_CALUDE_point_distance_and_inequality_l106_10684

/-- The value of m for which the point P(m, 3) is at distance 4 from the line 4x-3y+1=0
    and satisfies the inequality 2x+y<3 -/
theorem point_distance_and_inequality (m : ℝ) : 
  (abs (4 * m - 3 * 3 + 1) / Real.sqrt (4^2 + (-3)^2) = 4) ∧ 
  (2 * m + 3 < 3) → 
  m = -3 := by sorry

end NUMINAMATH_CALUDE_point_distance_and_inequality_l106_10684


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l106_10696

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  m < -6 ∨ m > 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l106_10696


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l106_10619

/-- 
Given a quadratic equation ax² + bx + c = 0, 
this theorem proves that for the specific equation x² - 3x - 2 = 0, 
the coefficients a, b, and c are 1, -3, and -2 respectively.
-/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - 3*x - 2 = 0) ∧ 
    a = 1 ∧ b = -3 ∧ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l106_10619


namespace NUMINAMATH_CALUDE_square_sum_over_sum_ge_sqrt_product_l106_10613

theorem square_sum_over_sum_ge_sqrt_product (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + y^2) / (x + y) ≥ Real.sqrt (x * y) := by sorry

end NUMINAMATH_CALUDE_square_sum_over_sum_ge_sqrt_product_l106_10613


namespace NUMINAMATH_CALUDE_emmys_journey_l106_10689

theorem emmys_journey (total_length : ℚ) 
  (h1 : total_length / 4 + 30 + total_length / 6 = total_length) : 
  total_length = 360 / 7 := by
sorry

end NUMINAMATH_CALUDE_emmys_journey_l106_10689


namespace NUMINAMATH_CALUDE_max_winning_pieces_l106_10661

/-- Represents the game board -/
def Board := Fin 1000 → Option Nat

/-- The maximum number of pieces a player can place in one turn -/
def max_placement : Nat := 17

/-- Checks if a series of pieces is consecutive -/
def is_consecutive (b : Board) (start finish : Fin 1000) : Prop :=
  ∀ i : Fin 1000, start ≤ i ∧ i ≤ finish → b i.val ≠ none

/-- Represents a valid move by the first player -/
def valid_first_move (b1 b2 : Board) : Prop :=
  ∃ placed : Nat, placed ≤ max_placement ∧
    (∀ i : Fin 1000, b1 i = none → b2 i = none ∨ (∃ n : Nat, b2 i = some n)) ∧
    (∀ i : Fin 1000, b1 i ≠ none → b2 i = b1 i)

/-- Represents a valid move by the second player -/
def valid_second_move (b1 b2 : Board) : Prop :=
  ∃ start finish : Fin 1000, start ≤ finish ∧ is_consecutive b1 start finish ∧
    (∀ i : Fin 1000, (i < start ∨ finish < i) → b2 i = b1 i) ∧
    (∀ i : Fin 1000, start ≤ i ∧ i ≤ finish → b2 i = none)

/-- Checks if the first player has won -/
def first_player_wins (b : Board) (n : Nat) : Prop :=
  ∃ start finish : Fin 1000, start ≤ finish ∧ 
    is_consecutive b start finish ∧
    (∀ i : Fin 1000, i < start ∨ finish < i → b i = none) ∧
    (finish - start + 1 : Nat) = n

/-- The main theorem stating that 98 is the maximum number of pieces for which
    the first player can always win -/
theorem max_winning_pieces : 
  (∀ n : Nat, n ≤ 98 → 
    ∀ initial : Board, (∀ i : Fin 1000, initial i = none) → 
      ∃ strategy : Nat → Board → Board,
        ∀ opponent_strategy : Board → Board,
          ∃ final : Board, first_player_wins final n) ∧
  ¬(∀ n : Nat, n ≤ 99 → 
    ∀ initial : Board, (∀ i : Fin 1000, initial i = none) → 
      ∃ strategy : Nat → Board → Board,
        ∀ opponent_strategy : Board → Board,
          ∃ final : Board, first_player_wins final n) :=
sorry

end NUMINAMATH_CALUDE_max_winning_pieces_l106_10661


namespace NUMINAMATH_CALUDE_water_tank_capacity_l106_10630

theorem water_tank_capacity : ∀ C : ℝ, 
  (0.4 * C - 0.1 * C = 36) → C = 120 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l106_10630


namespace NUMINAMATH_CALUDE_scalar_cross_product_sum_l106_10645

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def cross_product : V → V → V := sorry

theorem scalar_cross_product_sum (a b c d : V) (h : a + b + c + d = 0) :
  ∃! k : ℝ, ∀ (a b c d : V), a + b + c + d = 0 →
    k • (cross_product c b) + cross_product b c + cross_product c a + 
    cross_product a d + cross_product d d = 0 ∧ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_scalar_cross_product_sum_l106_10645


namespace NUMINAMATH_CALUDE_unique_special_number_l106_10665

/-- A two-digit number satisfying specific divisibility properties -/
def SpecialNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ 
  2 ∣ n ∧
  3 ∣ (n + 1) ∧
  4 ∣ (n + 2) ∧
  5 ∣ (n + 3)

/-- Theorem stating that 62 is the unique two-digit number satisfying the given conditions -/
theorem unique_special_number : ∃! n, SpecialNumber n :=
  sorry

end NUMINAMATH_CALUDE_unique_special_number_l106_10665


namespace NUMINAMATH_CALUDE_mary_stickers_l106_10676

theorem mary_stickers (front_page : ℕ) (other_pages : ℕ) (pages : ℕ) (remaining : ℕ) :
  front_page = 3 →
  other_pages = 7 →
  pages = 6 →
  remaining = 44 →
  front_page + other_pages * pages + remaining = 89 := by
  sorry

end NUMINAMATH_CALUDE_mary_stickers_l106_10676


namespace NUMINAMATH_CALUDE_vector_v_satisfies_conditions_l106_10640

/-- Parametric equation of a line in 2D space -/
structure ParamLine2D where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Definition of line l -/
def line_l : ParamLine2D :=
  { x := λ t => 2 + 5*t,
    y := λ t => 3 + 2*t }

/-- Definition of line m -/
def line_m : ParamLine2D :=
  { x := λ s => -7 + 5*s,
    y := λ s => 9 + 2*s }

/-- Point A on line l -/
def point_A : Vector2D :=
  { x := line_l.x 0,
    y := line_l.y 0 }

/-- Point B on line m -/
def point_B : Vector2D :=
  { x := line_m.x 0,
    y := line_m.y 0 }

/-- Vector v that PA is projected onto -/
def vector_v : Vector2D :=
  { x := -2,
    y := 5 }

/-- Theorem: The vector v satisfies the given conditions -/
theorem vector_v_satisfies_conditions :
  vector_v.y - vector_v.x = 7 ∧
  (∃ (P : Vector2D), P.x = point_A.x ∧ P.y = point_A.y) ∧
  (∀ (B : Vector2D), B.x = line_m.x 0 ∧ B.y = line_m.y 0 →
    ∃ (k : ℝ), vector_v.x * k = 0 ∧ vector_v.y * k = 0) :=
by sorry

end NUMINAMATH_CALUDE_vector_v_satisfies_conditions_l106_10640


namespace NUMINAMATH_CALUDE_tom_lifting_capacity_l106_10654

def initial_capacity : ℝ := 80
def training_multiplier : ℝ := 2
def specialization_increase : ℝ := 1.1
def num_hands : ℕ := 2

theorem tom_lifting_capacity : 
  initial_capacity * training_multiplier * specialization_increase * num_hands = 352 := by
  sorry

end NUMINAMATH_CALUDE_tom_lifting_capacity_l106_10654
