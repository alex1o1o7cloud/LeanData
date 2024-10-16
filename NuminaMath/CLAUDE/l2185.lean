import Mathlib

namespace NUMINAMATH_CALUDE_intersection_product_l2185_218589

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 4

-- Define the line m in polar coordinates
def line_m (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 3) = 2

-- Define the ray l in polar coordinates
def ray_l (ρ θ : ℝ) : Prop := θ = 5 * Real.pi / 6 ∧ ρ ≥ 0

-- Theorem statement
theorem intersection_product :
  ∃ (ρ_A ρ_B : ℝ),
    (∃ (x_A y_A : ℝ), circle_C x_A y_A ∧ x_A = ρ_A * Real.cos (5 * Real.pi / 6) ∧ y_A = ρ_A * Real.sin (5 * Real.pi / 6)) ∧
    (line_m ρ_B (5 * Real.pi / 6)) ∧
    ρ_A * ρ_B = -3 + Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_intersection_product_l2185_218589


namespace NUMINAMATH_CALUDE_possible_sum_less_than_100_l2185_218578

/-- Represents a team in the tournament -/
structure Team :=
  (id : Nat)
  (score : Nat)

/-- Represents the tournament -/
structure Tournament :=
  (teams : List Team)
  (num_teams : Nat)
  (num_games : Nat)

/-- The scoring system for the tournament -/
def scoring_system (winner_rank : Nat) (loser_rank : Nat) : Nat :=
  if winner_rank ≤ 5 then 3 else 2

/-- The theorem stating that it's possible for the sum of scores to be less than 100 -/
theorem possible_sum_less_than_100 (t : Tournament) :
  t.num_teams = 10 →
  t.num_games = (t.num_teams * (t.num_teams - 1)) / 2 →
  ∃ (scores : List Nat), 
    scores.length = t.num_teams ∧ 
    scores.sum < 100 ∧
    (∀ (i j : Nat), i < j → j < t.num_teams → 
      ∃ (points : Nat), points ≤ (scoring_system (i + 1) (j + 1)) ∧
        (scores.get! i + scores.get! j = points)) :=
sorry

end NUMINAMATH_CALUDE_possible_sum_less_than_100_l2185_218578


namespace NUMINAMATH_CALUDE_page_number_added_twice_l2185_218575

theorem page_number_added_twice (n : ℕ) (h : ∃ k : ℕ, k ≤ n ∧ (n * (n + 1)) / 2 + k = 2900) : 
  ∃ k : ℕ, k ≤ n ∧ (n * (n + 1)) / 2 + k = 2900 ∧ k = 50 := by
  sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l2185_218575


namespace NUMINAMATH_CALUDE_simplify_expression_l2185_218513

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - x^3 - 2) / (3 * x^3))^2) =
  (Real.sqrt (x^12 - 2*x^9 + 6*x^6 - 2*x^3 + 4)) / (3 * x^3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2185_218513


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2185_218587

theorem consecutive_integers_sum (x : ℤ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2185_218587


namespace NUMINAMATH_CALUDE_simplify_expression_l2185_218546

theorem simplify_expression : 
  ∃ x : ℚ, (3/4 * 60) - (8/5 * 60) + x = 12 ∧ x = 63 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2185_218546


namespace NUMINAMATH_CALUDE_female_students_count_l2185_218554

theorem female_students_count (female : ℕ) (male : ℕ) : 
  male = 3 * female →
  female + male = 52 →
  female = 13 := by
sorry

end NUMINAMATH_CALUDE_female_students_count_l2185_218554


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2185_218560

theorem inequality_system_solution_set (a : ℝ) : 
  (∃ x : ℝ, a * x > -1 ∧ x + a > 0) ↔ a > -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2185_218560


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l2185_218544

def U : Set ℕ := {x | x < 6}
def P : Set ℕ := {2, 4}
def Q : Set ℕ := {1, 3, 4, 6}

theorem complement_intersection_problem :
  (U \ P) ∩ Q = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l2185_218544


namespace NUMINAMATH_CALUDE_no_nonneg_int_solutions_l2185_218500

theorem no_nonneg_int_solutions :
  ¬∃ (x₁ x₂ : ℕ), 96 * x₁ + 97 * x₂ = 1000 := by
  sorry

end NUMINAMATH_CALUDE_no_nonneg_int_solutions_l2185_218500


namespace NUMINAMATH_CALUDE_no_real_solutions_l2185_218526

theorem no_real_solutions : ¬∃ (x : ℝ), 3 * x^2 + 5 = |4 * x + 2| - 3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2185_218526


namespace NUMINAMATH_CALUDE_alyssas_turnips_l2185_218522

theorem alyssas_turnips (keith_turnips total_turnips : ℕ) 
  (h1 : keith_turnips = 6)
  (h2 : total_turnips = 15) :
  total_turnips - keith_turnips = 9 :=
by sorry

end NUMINAMATH_CALUDE_alyssas_turnips_l2185_218522


namespace NUMINAMATH_CALUDE_multiply_and_add_equality_l2185_218571

theorem multiply_and_add_equality : 45 * 52 + 48 * 45 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_equality_l2185_218571


namespace NUMINAMATH_CALUDE_ratio_difference_l2185_218590

theorem ratio_difference (a b c : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : ∃ (x : ℕ), a = 3 * x ∧ b = 5 * x ∧ c = 7 * x) (h3 : c = 70) :
  c - a = 40 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_l2185_218590


namespace NUMINAMATH_CALUDE_tg_sum_formula_l2185_218558

-- Define the tangent and cotangent functions
noncomputable def tg (x : ℝ) : ℝ := Real.tan x
noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

-- Define the theorem
theorem tg_sum_formula (α β p q : ℝ) 
  (h1 : tg α + tg β = p) 
  (h2 : ctg α + ctg β = q) :
  (p = 0 ∧ q = 0 → tg (α + β) = 0) ∧
  (p ≠ 0 ∧ q ≠ 0 ∧ p ≠ q → tg (α + β) = p * q / (q - p)) ∧
  (p ≠ 0 ∧ q ≠ 0 ∧ p = q → ¬∃x, x = tg (α + β)) ∧
  ((p = 0 ∨ q = 0) ∧ p ≠ q → False) :=
by sorry


end NUMINAMATH_CALUDE_tg_sum_formula_l2185_218558


namespace NUMINAMATH_CALUDE_purple_length_is_three_l2185_218507

/-- The length of the purple part of a pencil -/
def purple_length (total black blue : ℝ) : ℝ := total - black - blue

/-- Theorem stating that the length of the purple part of the pencil is 3 cm -/
theorem purple_length_is_three :
  let total := 6
  let black := 2
  let blue := 1
  purple_length total black blue = 3 := by
  sorry

end NUMINAMATH_CALUDE_purple_length_is_three_l2185_218507


namespace NUMINAMATH_CALUDE_solution_set_F_max_value_F_inequality_holds_l2185_218584

-- Define the function F(x) = |x + 2| - 3|x|
def F (x : ℝ) : ℝ := |x + 2| - 3 * |x|

-- Theorem 1: The solution set of F(x) ≥ 0 is {x | -1/2 ≤ x ≤ 1}
theorem solution_set_F : 
  {x : ℝ | F x ≥ 0} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: The maximum value of F(x) is 2
theorem max_value_F : 
  ∃ (x : ℝ), F x = 2 ∧ ∀ (y : ℝ), F y ≤ 2 := by sorry

-- Corollary: The inequality F(x) ≥ a holds for all a ∈ (-∞, 2]
theorem inequality_holds :
  ∀ (a : ℝ), a ≤ 2 → ∃ (x : ℝ), F x ≥ a := by sorry

end NUMINAMATH_CALUDE_solution_set_F_max_value_F_inequality_holds_l2185_218584


namespace NUMINAMATH_CALUDE_expression_evaluation_l2185_218508

theorem expression_evaluation : ((69 + 7 * 8) / 3) * 12 = 500 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2185_218508


namespace NUMINAMATH_CALUDE_equation_solution_l2185_218530

theorem equation_solution (x : ℝ) (h1 : x ≠ 6) (h2 : x ≠ 3/4) :
  (x^2 - 10*x + 24)/(x - 6) + (4*x^2 + 20*x - 24)/(4*x - 3) + 2*x = 5 ↔ x = 1/4 := by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2185_218530


namespace NUMINAMATH_CALUDE_binomial_coeff_not_coprime_l2185_218520

theorem binomial_coeff_not_coprime (k m n : ℕ) (h1 : 0 < k) (h2 : k < m) (h3 : m < n) :
  ¬(Nat.gcd (Nat.choose n k) (Nat.choose n m) = 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_coeff_not_coprime_l2185_218520


namespace NUMINAMATH_CALUDE_function_sum_equals_one_l2185_218592

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define an odd function
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Main theorem
theorem function_sum_equals_one
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_even : IsEven f)
  (h_odd : IsOdd g)
  (h_f0 : f 0 = 1)
  (h_fg : ∀ x, g x = f (x - 1)) :
  f 2011 + f 2012 + f 2013 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_sum_equals_one_l2185_218592


namespace NUMINAMATH_CALUDE_concentric_circles_radii_l2185_218562

theorem concentric_circles_radii 
  (r R : ℝ) 
  (h_positive : 0 < r ∧ 0 < R) 
  (h_order : r < R) 
  (h_min_distance : R - r = 2) 
  (h_max_distance : R + r = 16) : 
  r = 7 ∧ R = 9 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_l2185_218562


namespace NUMINAMATH_CALUDE_partitions_count_l2185_218569

/-- The number of partitions of a set with n+1 elements into n subsets -/
def num_partitions (n : ℕ) : ℕ := (2^n - 1) * n + 1

/-- Theorem stating the number of partitions of a set with n+1 elements into n subsets -/
theorem partitions_count (n : ℕ) (h : n > 0) :
  num_partitions n = (2^n - 1) * n + 1 :=
by sorry

end NUMINAMATH_CALUDE_partitions_count_l2185_218569


namespace NUMINAMATH_CALUDE_ping_pong_rackets_sold_l2185_218597

theorem ping_pong_rackets_sold (total_sales : ℝ) (avg_price : ℝ) (h1 : total_sales = 735) (h2 : avg_price = 9.8) :
  total_sales / avg_price = 75 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_rackets_sold_l2185_218597


namespace NUMINAMATH_CALUDE_payment_equality_l2185_218585

/-- Represents the payment structure and hours worked for Harry and James -/
structure PaymentSystem where
  x : ℝ
  y : ℝ
  james_hours : ℝ
  harry_hours : ℝ

/-- Calculates James' earnings based on the given payment structure -/
def james_earnings (ps : PaymentSystem) : ℝ :=
  40 * ps.x + (ps.james_hours - 40) * 2 * ps.x

/-- Calculates Harry's earnings based on the given payment structure -/
def harry_earnings (ps : PaymentSystem) : ℝ :=
  12 * ps.x + (ps.harry_hours - 12) * ps.y * ps.x

/-- Theorem stating the conditions and the result to be proved -/
theorem payment_equality (ps : PaymentSystem) :
  ps.x > 0 ∧ 
  ps.y > 1 ∧ 
  ps.james_hours = 48 ∧ 
  james_earnings ps = harry_earnings ps →
  ps.harry_hours = 23 ∧ ps.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_payment_equality_l2185_218585


namespace NUMINAMATH_CALUDE_max_digit_occurrence_l2185_218557

/-- Represents the range of apartment numbers on each floor -/
def apartment_range : Set ℕ := {n | 0 ≤ n ∧ n ≤ 35}

/-- Counts the occurrences of a digit in a given number -/
def count_digit (d : ℕ) (n : ℕ) : ℕ := sorry

/-- Counts the occurrences of a digit in a range of numbers -/
def count_digit_in_range (d : ℕ) (range : Set ℕ) : ℕ := sorry

/-- Counts the occurrences of a digit in the hundreds place for a floor -/
def count_digit_hundreds (d : ℕ) (floor : ℕ) : ℕ := sorry

/-- The main theorem stating that the maximum occurrence of any digit is 36 -/
theorem max_digit_occurrence :
  ∃ d : ℕ, d < 10 ∧
    (count_digit_in_range d apartment_range +
     count_digit_in_range d apartment_range +
     count_digit_in_range d apartment_range +
     count_digit_hundreds 1 1 +
     count_digit_hundreds 2 2 +
     count_digit_hundreds 3 3) = 36 ∧
    ∀ d' : ℕ, d' < 10 →
      (count_digit_in_range d' apartment_range +
       count_digit_in_range d' apartment_range +
       count_digit_in_range d' apartment_range +
       count_digit_hundreds 1 1 +
       count_digit_hundreds 2 2 +
       count_digit_hundreds 3 3) ≤ 36 := by
  sorry

#check max_digit_occurrence

end NUMINAMATH_CALUDE_max_digit_occurrence_l2185_218557


namespace NUMINAMATH_CALUDE_gum_pack_size_l2185_218521

theorem gum_pack_size (mint_gum orange_gum y : ℕ) : 
  mint_gum = 24 → 
  orange_gum = 36 → 
  (mint_gum - 2 * y) / orange_gum = mint_gum / (orange_gum + 4 * y) → 
  y = 3 := by
sorry

end NUMINAMATH_CALUDE_gum_pack_size_l2185_218521


namespace NUMINAMATH_CALUDE_martha_apples_l2185_218514

theorem martha_apples (tim harry martha : ℕ) 
  (h1 : martha = tim + 30)
  (h2 : harry = tim / 2)
  (h3 : harry = 19) : 
  martha = 68 := by sorry

end NUMINAMATH_CALUDE_martha_apples_l2185_218514


namespace NUMINAMATH_CALUDE_identity_proof_l2185_218567

theorem identity_proof (a b m n x y : ℝ) :
  (a^2 + b^2) * (m^2 + n^2) * (x^2 + y^2) = 
  (a*n*y - a*m*x - b*m*y + b*n*x)^2 + (a*m*y + a*n*x + b*m*x - b*n*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l2185_218567


namespace NUMINAMATH_CALUDE_quadrant_line_conditions_l2185_218583

/-- A line passing through the first, third, and fourth quadrants -/
structure QuadrantLine where
  k : ℝ
  b : ℝ
  first_quadrant : ∃ x y, x > 0 ∧ y > 0 ∧ y = k * x + b
  third_quadrant : ∃ x y, x < 0 ∧ y < 0 ∧ y = k * x + b
  fourth_quadrant : ∃ x y, x > 0 ∧ y < 0 ∧ y = k * x + b

/-- Theorem stating the conditions on k and b for a line passing through the first, third, and fourth quadrants -/
theorem quadrant_line_conditions (l : QuadrantLine) : l.k > 0 ∧ l.b < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_line_conditions_l2185_218583


namespace NUMINAMATH_CALUDE_handshake_count_l2185_218588

theorem handshake_count (n : ℕ) (h : n = 8) : n * (n - 1) / 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l2185_218588


namespace NUMINAMATH_CALUDE_perimeter_VWX_equals_5_plus_10_root_5_l2185_218552

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ

/-- Midpoints of edges in the right prism -/
structure Midpoints where
  v : ℝ × ℝ × ℝ
  w : ℝ × ℝ × ℝ
  x : ℝ × ℝ × ℝ

/-- Calculate the perimeter of triangle VWX in the right prism -/
def perimeterVWX (prism : RightPrism) (midpoints : Midpoints) : ℝ :=
  sorry

/-- Theorem stating the perimeter of triangle VWX -/
theorem perimeter_VWX_equals_5_plus_10_root_5 (prism : RightPrism) (midpoints : Midpoints) 
  (h1 : prism.height = 20)
  (h2 : prism.baseSideLength = 10)
  (h3 : midpoints.v = (5, 0, 0))
  (h4 : midpoints.w = (10, 5, 0))
  (h5 : midpoints.x = (5, 5, 10)) :
  perimeterVWX prism midpoints = 5 + 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_VWX_equals_5_plus_10_root_5_l2185_218552


namespace NUMINAMATH_CALUDE_power_twelve_minus_one_divisible_by_five_l2185_218553

theorem power_twelve_minus_one_divisible_by_five (a : ℤ) (h : ¬ 5 ∣ a) : 
  5 ∣ (a^12 - 1) := by
  sorry

end NUMINAMATH_CALUDE_power_twelve_minus_one_divisible_by_five_l2185_218553


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l2185_218548

def N : ℕ := 34 * 34 * 63 * 270

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry
def sum_of_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_of_odd_divisors N) * 14 = sum_of_even_divisors N := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l2185_218548


namespace NUMINAMATH_CALUDE_steve_sleeping_time_l2185_218576

theorem steve_sleeping_time (T : ℝ) (school_fraction : ℝ) (assignment_fraction : ℝ) (family_hours : ℝ) :
  T = 24 →
  school_fraction = 1 / 6 →
  assignment_fraction = 1 / 12 →
  family_hours = 10 →
  (T - (school_fraction * T + assignment_fraction * T + family_hours)) / T = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_steve_sleeping_time_l2185_218576


namespace NUMINAMATH_CALUDE_housing_units_without_cable_or_vcr_l2185_218517

theorem housing_units_without_cable_or_vcr 
  (total : ℝ) 
  (cable : ℝ) 
  (vcr : ℝ) 
  (both : ℝ) 
  (h1 : cable = (1 / 5) * total) 
  (h2 : vcr = (1 / 10) * total) 
  (h3 : both = (1 / 3) * cable) :
  (total - (cable + vcr - both)) / total = 23 / 30 := by
sorry

end NUMINAMATH_CALUDE_housing_units_without_cable_or_vcr_l2185_218517


namespace NUMINAMATH_CALUDE_problem_solution_l2185_218551

theorem problem_solution (a b m n : ℝ) : 
  (a = -(-(3 : ℝ))) → 
  (b = (-((1 : ℝ)/(2 : ℝ)))⁻¹) → 
  (|m - a| + |n + b| = 0) → 
  (m = 3 ∧ n = -2) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2185_218551


namespace NUMINAMATH_CALUDE_max_y_value_max_y_achieved_l2185_218504

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = 4) : y ≤ 7 := by
  sorry

theorem max_y_achieved : ∃ x y : ℤ, x * y + 3 * x + 2 * y = 4 ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_max_y_achieved_l2185_218504


namespace NUMINAMATH_CALUDE_infinitely_many_n_divides_2_pow_n_plus_2_l2185_218555

theorem infinitely_many_n_divides_2_pow_n_plus_2 :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, n > 0 ∧ n ∣ 2^n + 2 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_n_divides_2_pow_n_plus_2_l2185_218555


namespace NUMINAMATH_CALUDE_michaels_additional_money_michael_needs_additional_money_l2185_218580

/-- Calculates the additional money Michael needs to buy all items for Mother's Day. -/
theorem michaels_additional_money (michael_money : ℝ) 
  (cake_price discount_cake : ℝ) (bouquet_price tax_bouquet : ℝ) 
  (balloons_price : ℝ) (perfume_price_gbp discount_perfume gbp_to_usd : ℝ) 
  (album_price_eur tax_album eur_to_usd : ℝ) : ℝ :=
  let cake_cost := cake_price * (1 - discount_cake)
  let bouquet_cost := bouquet_price * (1 + tax_bouquet)
  let balloons_cost := balloons_price
  let perfume_cost := perfume_price_gbp * (1 - discount_perfume) * gbp_to_usd
  let album_cost := album_price_eur * (1 + tax_album) * eur_to_usd
  let total_cost := cake_cost + bouquet_cost + balloons_cost + perfume_cost + album_cost
  total_cost - michael_money

/-- Proves that Michael needs an additional $78.90 to buy all items. -/
theorem michael_needs_additional_money :
  michaels_additional_money 50 20 0.1 36 0.05 5 30 0.15 1.4 25 0.08 1.2 = 78.9 := by
  sorry

end NUMINAMATH_CALUDE_michaels_additional_money_michael_needs_additional_money_l2185_218580


namespace NUMINAMATH_CALUDE_article_percentage_gain_l2185_218533

/-- Calculates the percentage gain when selling an article --/
def percentage_gain (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating the percentage gain for the given problem --/
theorem article_percentage_gain :
  let cost_price : ℚ := 40
  let selling_price : ℚ := 350
  percentage_gain cost_price selling_price = 775 := by
  sorry

end NUMINAMATH_CALUDE_article_percentage_gain_l2185_218533


namespace NUMINAMATH_CALUDE_three_people_in_five_seats_l2185_218518

/-- The number of ways to arrange k objects from n distinct objects --/
def permutation (n k : ℕ) : ℕ := sorry

theorem three_people_in_five_seats :
  permutation 5 3 = 60 := by sorry

end NUMINAMATH_CALUDE_three_people_in_five_seats_l2185_218518


namespace NUMINAMATH_CALUDE_max_value_sum_of_fractions_l2185_218527

theorem max_value_sum_of_fractions (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + 2*b + 3*c = 1) : 
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2*y + 3*z = 1 ∧ 
    1/(2*x + y) + 1/(2*y + z) + 1/(2*z + x) > 1/(2*a + b) + 1/(2*b + c) + 1/(2*c + a)) ∨
  (1/(2*a + b) + 1/(2*b + c) + 1/(2*c + a) = 7) :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_of_fractions_l2185_218527


namespace NUMINAMATH_CALUDE_range_of_a_solution_set_l2185_218570

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem for part I
theorem range_of_a (a : ℝ) :
  (∃ x, f x < 2 * a - 1) ↔ a > 2 :=
sorry

-- Theorem for part II
theorem solution_set :
  {x : ℝ | f x ≥ x^2 - 2*x} = {x : ℝ | -1 ≤ x ∧ x ≤ 2 + Real.sqrt 3} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_solution_set_l2185_218570


namespace NUMINAMATH_CALUDE_yuri_total_puppies_l2185_218528

def puppies_week1 : ℕ := 20

def puppies_week2 : ℕ := (2 * puppies_week1) / 5

def puppies_week3 : ℕ := 2 * puppies_week2

def puppies_week4 : ℕ := puppies_week1 + 10

def total_puppies : ℕ := puppies_week1 + puppies_week2 + puppies_week3 + puppies_week4

theorem yuri_total_puppies : total_puppies = 74 := by
  sorry

end NUMINAMATH_CALUDE_yuri_total_puppies_l2185_218528


namespace NUMINAMATH_CALUDE_expression_evaluation_l2185_218516

theorem expression_evaluation (x : ℕ) (h : x = 3) :
  x + x * (x^x) + (x^(x^x)) = 7625597485071 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2185_218516


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l2185_218511

theorem quadratic_root_sum (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - (2*m - 2)*x + (m^2 - 2*m) = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + x₂ = 10 →
  m = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l2185_218511


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2185_218566

theorem unique_quadratic_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + (b + 2/b) * x + c = 0) ↔ 
  c = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2185_218566


namespace NUMINAMATH_CALUDE_intersection_point_on_graph_and_y_axis_l2185_218556

/-- The quadratic function f(x) = (x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- The point where f intersects the y-axis -/
def intersection_point : ℝ × ℝ := (0, 3)

/-- Theorem: The intersection_point lies on both the y-axis and the graph of f -/
theorem intersection_point_on_graph_and_y_axis :
  (intersection_point.1 = 0) ∧ 
  (intersection_point.2 = f intersection_point.1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_on_graph_and_y_axis_l2185_218556


namespace NUMINAMATH_CALUDE_minimum_packages_l2185_218506

theorem minimum_packages (p : ℕ) : p > 0 → (∃ N : ℕ, N = 19 * p ∧ N % 7 = 4 ∧ N % 11 = 1) → p ≥ 40 :=
by sorry

end NUMINAMATH_CALUDE_minimum_packages_l2185_218506


namespace NUMINAMATH_CALUDE_sprint_competition_races_l2185_218538

/-- The number of races needed to determine a champion in a sprint competition --/
def racesNeeded (totalSprinters : ℕ) (lanesPerRace : ℕ) (eliminatedPerRace : ℕ) : ℕ :=
  Nat.ceil ((totalSprinters - 1) / eliminatedPerRace)

/-- Theorem stating that 46 races are needed for the given conditions --/
theorem sprint_competition_races : 
  racesNeeded 320 8 7 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sprint_competition_races_l2185_218538


namespace NUMINAMATH_CALUDE_factorization_3mx_6my_factorization_1_25x_squared_l2185_218509

-- For the first expression
theorem factorization_3mx_6my (m x y : ℝ) : 
  3 * m * x - 6 * m * y = 3 * m * (x - 2 * y) := by sorry

-- For the second expression
theorem factorization_1_25x_squared (x : ℝ) :
  1 - 25 * x^2 = (1 + 5 * x) * (1 - 5 * x) := by sorry

end NUMINAMATH_CALUDE_factorization_3mx_6my_factorization_1_25x_squared_l2185_218509


namespace NUMINAMATH_CALUDE_division_problem_l2185_218595

theorem division_problem : (96 : ℚ) / ((8 : ℚ) / 4) = 48 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2185_218595


namespace NUMINAMATH_CALUDE_yellow_balloon_ratio_l2185_218531

theorem yellow_balloon_ratio (total_balloons : ℕ) (num_colors : ℕ) (anya_balloons : ℕ) : 
  total_balloons = 672 →
  num_colors = 4 →
  anya_balloons = 84 →
  (anya_balloons : ℚ) / (total_balloons / num_colors) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balloon_ratio_l2185_218531


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2185_218586

theorem unique_solution_trigonometric_equation :
  ∃! (n : ℕ+), Real.sin (π / (3 * n.val)) + Real.cos (π / (3 * n.val)) = Real.sqrt (n.val + 1) / 2 :=
by
  -- The unique solution is n = 7
  use 7
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2185_218586


namespace NUMINAMATH_CALUDE_blue_cube_faces_l2185_218519

theorem blue_cube_faces (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_cube_faces_l2185_218519


namespace NUMINAMATH_CALUDE_apples_on_ground_l2185_218572

/-- The number of apples that have fallen to the ground -/
def fallen_apples : ℕ := sorry

/-- The number of apples hanging on the tree -/
def hanging_apples : ℕ := 5

/-- The number of apples eaten by the dog -/
def eaten_apples : ℕ := 3

/-- The number of apples left after the dog eats -/
def remaining_apples : ℕ := 10

theorem apples_on_ground :
  fallen_apples = 13 :=
by sorry

end NUMINAMATH_CALUDE_apples_on_ground_l2185_218572


namespace NUMINAMATH_CALUDE_inequality_condition_l2185_218524

theorem inequality_condition (x y : ℝ) : (x > y ∧ 1 / x > 1 / y) ↔ x * y < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l2185_218524


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l2185_218577

theorem crazy_silly_school_books (movies : ℕ) (books : ℕ) : 
  movies = 14 → books = movies + 1 → books = 15 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l2185_218577


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2185_218541

theorem diophantine_equation_solution (m n x y : ℕ) : m ≥ 2 ∧ n ≥ 2 ∧ x^n + y^n = 3^m →
  ((x = 2 ∧ y = 1 ∧ n = 3 ∧ m = 2) ∨ (x = 1 ∧ y = 2 ∧ n = 3 ∧ m = 2)) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2185_218541


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2185_218535

theorem set_intersection_theorem (x : ℝ) :
  { x : ℝ | x ≥ -2 } ∩ ({ x : ℝ | x > 0 }ᶜ) = { x : ℝ | -2 ≤ x ∧ x ≤ 0 } := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2185_218535


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l2185_218582

theorem power_tower_mod_500 : 5^(5^(5^2)) ≡ 25 [ZMOD 500] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l2185_218582


namespace NUMINAMATH_CALUDE_sales_tax_calculation_l2185_218539

def total_cost : ℝ := 25
def tax_rate : ℝ := 0.05
def tax_free_cost : ℝ := 18.7

theorem sales_tax_calculation :
  ∃ (taxable_cost : ℝ),
    taxable_cost + tax_free_cost + taxable_cost * tax_rate = total_cost ∧
    taxable_cost * tax_rate = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_calculation_l2185_218539


namespace NUMINAMATH_CALUDE_percentage_men_correct_l2185_218565

/-- The percentage of men in a college class. -/
def percentage_men : ℝ := 40

theorem percentage_men_correct :
  let women_science_percentage : ℝ := 30
  let non_science_percentage : ℝ := 60
  let men_science_percentage : ℝ := 55.00000000000001
  let women_percentage : ℝ := 100 - percentage_men
  let science_percentage : ℝ := 100 - non_science_percentage
  (women_science_percentage / 100 * women_percentage + 
   men_science_percentage / 100 * percentage_men = science_percentage) ∧
  (percentage_men ≥ 0 ∧ percentage_men ≤ 100) :=
by sorry

end NUMINAMATH_CALUDE_percentage_men_correct_l2185_218565


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2185_218561

/-- Proves that the cost price is 1250 given the markup percentage and selling price -/
theorem cost_price_calculation (markup_percentage : ℝ) (selling_price : ℝ) : 
  markup_percentage = 60 →
  selling_price = 2000 →
  (100 + markup_percentage) / 100 * (selling_price / ((100 + markup_percentage) / 100)) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2185_218561


namespace NUMINAMATH_CALUDE_total_lamps_is_147_l2185_218525

/-- The number of lamps per room -/
def lamps_per_room : ℕ := 7

/-- The number of rooms in the hotel -/
def rooms : ℕ := 21

/-- The total number of lamps bought for the hotel -/
def total_lamps : ℕ := lamps_per_room * rooms

/-- Theorem stating that the total number of lamps bought is 147 -/
theorem total_lamps_is_147 : total_lamps = 147 := by sorry

end NUMINAMATH_CALUDE_total_lamps_is_147_l2185_218525


namespace NUMINAMATH_CALUDE_divisor_sum_condition_l2185_218515

def d (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_sum_condition (n : ℕ) : n ≥ 3 → (d (n - 1) + d n + d (n + 1) ≤ 8 ↔ n = 3 ∨ n = 4 ∨ n = 6) := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_condition_l2185_218515


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_real_l2185_218536

theorem sqrt_x_plus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_real_l2185_218536


namespace NUMINAMATH_CALUDE_min_cars_with_stripes_is_two_l2185_218593

/-- Represents the properties of a car group -/
structure CarGroup where
  total : ℕ
  no_ac : ℕ
  max_ac_no_stripes : ℕ

/-- The minimum number of cars with racing stripes -/
def min_cars_with_stripes (group : CarGroup) : ℕ :=
  group.total - group.no_ac - group.max_ac_no_stripes

/-- Theorem stating the minimum number of cars with racing stripes -/
theorem min_cars_with_stripes_is_two (group : CarGroup) 
  (h1 : group.total = 100)
  (h2 : group.no_ac = 49)
  (h3 : group.max_ac_no_stripes = 49) : 
  min_cars_with_stripes group = 2 := by
  sorry

#eval min_cars_with_stripes ⟨100, 49, 49⟩

end NUMINAMATH_CALUDE_min_cars_with_stripes_is_two_l2185_218593


namespace NUMINAMATH_CALUDE_base_b_cube_iff_six_l2185_218510

/-- Represents a number in base b --/
def base_b_number (b : ℕ) : ℕ := b^2 + 4*b + 4

/-- Checks if a natural number is a perfect cube --/
def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

/-- The main theorem: 144 in base b is a cube iff b = 6 --/
theorem base_b_cube_iff_six (b : ℕ) : 
  (b > 0) → (is_cube (base_b_number b) ↔ b = 6) := by
sorry

end NUMINAMATH_CALUDE_base_b_cube_iff_six_l2185_218510


namespace NUMINAMATH_CALUDE_quiz_percentage_correct_l2185_218545

theorem quiz_percentage_correct (x : ℕ) : 
  let total_questions : ℕ := 7 * x
  let missed_questions : ℕ := 2 * x
  let correct_questions : ℕ := total_questions - missed_questions
  let percentage_correct : ℚ := (correct_questions : ℚ) / (total_questions : ℚ) * 100
  percentage_correct = 500 / 7 :=
by sorry

end NUMINAMATH_CALUDE_quiz_percentage_correct_l2185_218545


namespace NUMINAMATH_CALUDE_congruence_solution_l2185_218550

theorem congruence_solution (x : ℤ) : 
  (10 * x + 3) % 18 = 7 % 18 → x % 9 = 4 % 9 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2185_218550


namespace NUMINAMATH_CALUDE_equation_roots_l2185_218523

theorem equation_roots : 
  let S := {x : ℝ | 0 < x ∧ x < 1 ∧ 8 * x * (2 * x^2 - 1) * (8 * x^4 - 8 * x^2 + 1) = 1}
  S = {Real.cos (π / 9), Real.cos (π / 3), Real.cos (2 * π / 7)} := by sorry

end NUMINAMATH_CALUDE_equation_roots_l2185_218523


namespace NUMINAMATH_CALUDE_trigonometric_simplification_special_angle_simplification_l2185_218599

-- Part 1
theorem trigonometric_simplification (α : ℝ) :
  (Real.cos (α - π / 2) / Real.sin (5 * π / 2 + α)) *
  Real.sin (α - π) * Real.cos (2 * π - α) = -Real.sin α ^ 2 := by
  sorry

-- Part 2
theorem special_angle_simplification :
  (Real.sqrt (1 - 2 * Real.sin (20 * π / 180) * Real.cos (200 * π / 180))) /
  (Real.cos (160 * π / 180) - Real.sqrt (1 - Real.cos (20 * π / 180) ^ 2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_special_angle_simplification_l2185_218599


namespace NUMINAMATH_CALUDE_child_weight_is_30_l2185_218537

/-- The weight of the child in pounds -/
def child_weight : ℝ := 30

/-- The weight of the dog in pounds -/
def dog_weight : ℝ := 0.3 * child_weight

/-- The weight of the father in pounds -/
def father_weight : ℝ := 180 - child_weight - dog_weight

theorem child_weight_is_30 :
  -- Combined weight is 180 pounds
  father_weight + child_weight + dog_weight = 180 ∧
  -- Father and child together weigh 162 pounds more than the dog
  father_weight + child_weight = dog_weight + 162 ∧
  -- Dog weighs 70% less than the child
  dog_weight = 0.3 * child_weight →
  child_weight = 30 := by sorry

end NUMINAMATH_CALUDE_child_weight_is_30_l2185_218537


namespace NUMINAMATH_CALUDE_max_valid_sequence_length_l2185_218534

/-- A sequence of natural numbers satisfying the given conditions -/
def ValidSequence (a : Fin k → ℕ) : Prop :=
  (∀ i j, i < j → a i < a j) ∧ 
  (∀ i, 1 ≤ a i ∧ a i ≤ 50) ∧
  (∀ i j, i ≠ j → ¬(7 ∣ (a i + a j)))

/-- The maximum length of a valid sequence -/
def MaxValidSequenceLength : ℕ := 23

theorem max_valid_sequence_length :
  (∃ (k : ℕ) (a : Fin k → ℕ), ValidSequence a ∧ k = MaxValidSequenceLength) ∧
  (∀ (k : ℕ) (a : Fin k → ℕ), ValidSequence a → k ≤ MaxValidSequenceLength) :=
sorry

end NUMINAMATH_CALUDE_max_valid_sequence_length_l2185_218534


namespace NUMINAMATH_CALUDE_house_cleaning_time_l2185_218574

def total_time (dawn_dish_time andy_laundry_time andy_vacuum_time dawn_window_time : ℝ) : ℝ :=
  dawn_dish_time + andy_laundry_time + andy_vacuum_time + dawn_window_time

theorem house_cleaning_time : ∃ (dawn_dish_time andy_laundry_time andy_vacuum_time dawn_window_time : ℝ),
  dawn_dish_time = 20 ∧
  andy_laundry_time = 2 * dawn_dish_time + 6 ∧
  andy_vacuum_time = Real.sqrt (andy_laundry_time - dawn_dish_time) ∧
  dawn_window_time = (1 / 4) * (andy_laundry_time + dawn_dish_time) ∧
  total_time dawn_dish_time andy_laundry_time andy_vacuum_time dawn_window_time = 87.6 := by
  sorry

end NUMINAMATH_CALUDE_house_cleaning_time_l2185_218574


namespace NUMINAMATH_CALUDE_train_length_problem_l2185_218596

theorem train_length_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) : 
  faster_speed = 46 →
  slower_speed = 36 →
  passing_time = 36 →
  ∃ (train_length : ℝ), 
    train_length = 50 ∧ 
    2 * train_length = (faster_speed - slower_speed) * (5/18) * passing_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_problem_l2185_218596


namespace NUMINAMATH_CALUDE_quadratic_roots_integer_l2185_218547

theorem quadratic_roots_integer (p : ℤ) :
  (∃ x y : ℤ, x ≠ y ∧ x^2 + p*x + p + 4 = 0 ∧ y^2 + p*y + p + 4 = 0) →
  p = 8 ∨ p = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_integer_l2185_218547


namespace NUMINAMATH_CALUDE_box_volume_conversion_l2185_218540

theorem box_volume_conversion (box_volume_cubic_feet : ℝ) :
  box_volume_cubic_feet = 216 →
  box_volume_cubic_feet / 27 = 8 :=
by sorry

end NUMINAMATH_CALUDE_box_volume_conversion_l2185_218540


namespace NUMINAMATH_CALUDE_parabola_normal_intersection_l2185_218579

/-- Given a parabola y = x^2, for any point (x₀, y₀) on the parabola,
    if the normal line at this point intersects the y-axis at (0, y₁),
    then y₁ - y₀ = 1/2 -/
theorem parabola_normal_intersection (x₀ y₀ y₁ : ℝ) : 
  y₀ = x₀^2 →  -- point (x₀, y₀) is on the parabola
  (∃ k : ℝ, k * (x - x₀) = y - y₀ ∧  -- equation of the normal line
            k = -(2 * x₀)⁻¹ ∧        -- slope of the normal line
            y₁ = k * (-x₀) + y₀) →   -- y₁ is the y-intercept of the normal line
  y₁ - y₀ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_normal_intersection_l2185_218579


namespace NUMINAMATH_CALUDE_slope_product_l2185_218568

/-- Given two lines with slopes m and n, where one line makes three times
    the angle with the horizontal as the other and has 3 times the slope,
    prove that mn = 9/4 -/
theorem slope_product (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : m = 3 * n)
  (h4 : Real.arctan m = 3 * Real.arctan n) : m * n = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_product_l2185_218568


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2185_218563

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 + i) / (1 - i) = (1 / 2 : ℂ) + (3 / 2 : ℂ) * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2185_218563


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l2185_218549

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l2185_218549


namespace NUMINAMATH_CALUDE_jerry_shelf_comparison_l2185_218501

theorem jerry_shelf_comparison : 
  ∀ (initial_action_figures initial_books added_action_figures : ℕ),
    initial_action_figures = 5 →
    initial_books = 9 →
    added_action_figures = 7 →
    (initial_action_figures + added_action_figures) - initial_books = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_comparison_l2185_218501


namespace NUMINAMATH_CALUDE_solution_satisfies_system_solution_is_unique_l2185_218512

/-- The solution to the system of equations 4x - 6y = -3 and 9x + 3y = 6.3 -/
def solution_pair : ℝ × ℝ := (0.436, 0.792)

/-- The first equation of the system -/
def equation1 (x y : ℝ) : Prop := 4 * x - 6 * y = -3

/-- The second equation of the system -/
def equation2 (x y : ℝ) : Prop := 9 * x + 3 * y = 6.3

/-- Theorem stating that the solution_pair satisfies both equations -/
theorem solution_satisfies_system : 
  let (x, y) := solution_pair
  equation1 x y ∧ equation2 x y :=
by sorry

/-- Theorem stating that the solution is unique -/
theorem solution_is_unique :
  ∀ (x y : ℝ), equation1 x y ∧ equation2 x y → (x, y) = solution_pair :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_solution_is_unique_l2185_218512


namespace NUMINAMATH_CALUDE_inequality_proof_l2185_218594

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  (1 / (1 + a*b) + 1 / (1 + b*c) + 1 / (1 + a*c) ≥ 3/2) ∧ 
  (1 / (1 + a*b) + 1 / (1 + b*c) + 1 / (1 + a*c) = 3/2 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2185_218594


namespace NUMINAMATH_CALUDE_infinitely_many_palindromes_in_x_seq_l2185_218591

/-- A sequence is defined as x_n = 2013 + 317n, where n ≥ 0. -/
def x_seq (n : ℕ) : ℕ := 2013 + 317 * n

/-- A number is palindromic if its decimal representation reads the same forwards and backwards. -/
def is_palindrome (n : ℕ) : Prop := sorry

/-- There are infinitely many palindromic numbers in the sequence x_n. -/
theorem infinitely_many_palindromes_in_x_seq :
  ∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ is_palindrome (x_seq n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_palindromes_in_x_seq_l2185_218591


namespace NUMINAMATH_CALUDE_amcb_paths_count_l2185_218542

/-- Represents the number of paths from one letter to the next -/
structure PathCount where
  a_to_m : Nat
  m_to_c : Nat
  c_to_b : Nat

/-- The configuration of the letter arrangement -/
structure LetterArrangement where
  central_a : Nat
  m_adjacent_to_a : Nat
  c_adjacent_to_m : Nat
  b_adjacent_to_c : Nat

/-- Calculates the total number of paths spelling "AMCB" -/
def total_paths (arrangement : LetterArrangement) : Nat :=
  arrangement.central_a * arrangement.m_adjacent_to_a * arrangement.c_adjacent_to_m * arrangement.b_adjacent_to_c

/-- The specific arrangement for this problem -/
def amcb_arrangement : LetterArrangement :=
  { central_a := 1
  , m_adjacent_to_a := 4
  , c_adjacent_to_m := 2
  , b_adjacent_to_c := 3 }

theorem amcb_paths_count :
  total_paths amcb_arrangement = 24 :=
sorry

end NUMINAMATH_CALUDE_amcb_paths_count_l2185_218542


namespace NUMINAMATH_CALUDE_find_m_l2185_218532

def A : Set ℕ := {1, 2, 3}
def B (m : ℕ) : Set ℕ := {2, m, 4}

theorem find_m : ∃ m : ℕ, (A ∩ B m = {2, 3}) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2185_218532


namespace NUMINAMATH_CALUDE_impossibleStar_l2185_218559

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the star vertices
variable (A1 A2 A3 A4 A5 : Point3D)

-- Define a function to check if a point is above a plane
def isAbovePlane (p : Point3D) (p1 p2 p3 : Point3D) : Prop := sorry

-- Define a function to check if a point is below a plane
def isBelowPlane (p : Point3D) (p1 p2 p3 : Point3D) : Prop := sorry

-- Define a function to check if two points are connected by a straight line
def areConnected (p1 p2 : Point3D) : Prop := sorry

-- Theorem statement
theorem impossibleStar (h1 : isAbovePlane A2 A1 A3 A5)
                       (h2 : isBelowPlane A4 A1 A3 A5)
                       (h3 : areConnected A2 A4) :
  False := sorry

end NUMINAMATH_CALUDE_impossibleStar_l2185_218559


namespace NUMINAMATH_CALUDE_computer_accessories_cost_l2185_218598

def original_amount : ℕ := 48
def snack_cost : ℕ := 8

theorem computer_accessories_cost (remaining_amount : ℕ) 
  (h1 : remaining_amount = original_amount / 2 + 4) 
  (h2 : remaining_amount = original_amount - snack_cost - (original_amount - remaining_amount - snack_cost)) :
  original_amount - remaining_amount - snack_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_computer_accessories_cost_l2185_218598


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2185_218505

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ↔ a ∈ Set.Ici 0 ∩ Set.Iio 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2185_218505


namespace NUMINAMATH_CALUDE_triangle_height_l2185_218502

/-- Given a triangle with area 615 m² and a side of 123 m, 
    the perpendicular height to that side is 10 m. -/
theorem triangle_height (A : ℝ) (b : ℝ) (h : ℝ) 
  (area_eq : A = 615) 
  (base_eq : b = 123) 
  (area_formula : A = (1/2) * b * h) : h = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l2185_218502


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2185_218503

theorem cyclic_sum_inequality (x y z : ℝ) (a b : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 1 → 
    x*y + y*z + z*x ≥ a*(y^2*z^2 + z^2*x^2 + x^2*y^2) + b*x*y*z) ↔ 
  (b = 9 - a ∧ 0 < a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2185_218503


namespace NUMINAMATH_CALUDE_product_of_numbers_l2185_218581

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 58) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2185_218581


namespace NUMINAMATH_CALUDE_lost_revenue_calculation_l2185_218564

/-- Represents the movie theater scenario --/
structure MovieTheater where
  capacity : ℕ
  ticketPrice : ℚ
  ticketsSold : ℕ

/-- Calculates the lost revenue for a movie theater --/
def lostRevenue (theater : MovieTheater) : ℚ :=
  (theater.capacity : ℚ) * theater.ticketPrice - (theater.ticketsSold : ℚ) * theater.ticketPrice

/-- Theorem stating the lost revenue for the given scenario --/
theorem lost_revenue_calculation (theater : MovieTheater) 
  (h1 : theater.capacity = 50)
  (h2 : theater.ticketPrice = 8)
  (h3 : theater.ticketsSold = 24) : 
  lostRevenue theater = 208 := by
  sorry

#eval lostRevenue { capacity := 50, ticketPrice := 8, ticketsSold := 24 }

end NUMINAMATH_CALUDE_lost_revenue_calculation_l2185_218564


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_squares_is_integer_l2185_218543

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_progression (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A sequence contains the squares of its first three terms. -/
def contains_first_three_squares (a : ℕ → ℚ) : Prop :=
  ∃ k₁ k₂ k₃ : ℕ, a k₁ = (a 1)^2 ∧ a k₂ = (a 2)^2 ∧ a k₃ = (a 3)^2

/-- If an arithmetic progression contains the squares of its first three terms,
    then all terms in the progression are integers. -/
theorem arithmetic_progression_with_squares_is_integer
  (a : ℕ → ℚ)
  (h₁ : is_arithmetic_progression a)
  (h₂ : contains_first_three_squares a) :
  ∀ n : ℕ, ∃ k : ℤ, a n = k :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_squares_is_integer_l2185_218543


namespace NUMINAMATH_CALUDE_min_legs_correct_l2185_218529

/-- The length of the circular track in meters -/
def track_length : ℕ := 660

/-- The length of each leg of the race in meters -/
def leg_length : ℕ := 150

/-- The minimum number of legs required for the relay race -/
def min_legs : ℕ := 22

/-- Theorem stating that the minimum number of legs is correct -/
theorem min_legs_correct :
  min_legs = Nat.lcm track_length leg_length / leg_length :=
by sorry

end NUMINAMATH_CALUDE_min_legs_correct_l2185_218529


namespace NUMINAMATH_CALUDE_currency_denomination_problem_l2185_218573

theorem currency_denomination_problem (total_notes : ℕ) (total_amount : ℕ) (amount_50 : ℕ) (d : ℕ) :
  total_notes = 85 →
  total_amount = 5000 →
  amount_50 = 3500 →
  (amount_50 / 50 + (total_notes - amount_50 / 50)) = total_notes →
  50 * (amount_50 / 50) + d * (total_notes - amount_50 / 50) = total_amount →
  d = 100 := by
sorry

end NUMINAMATH_CALUDE_currency_denomination_problem_l2185_218573
