import Mathlib

namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l454_45442

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l454_45442


namespace NUMINAMATH_CALUDE_ryan_chinese_learning_hours_l454_45411

/-- Given Ryan's daily Chinese learning hours and number of learning days, 
    calculate the total hours spent learning Chinese -/
def total_chinese_hours (daily_hours : ℕ) (days : ℕ) : ℕ :=
  daily_hours * days

/-- Theorem stating that Ryan spends 24 hours learning Chinese in 6 days -/
theorem ryan_chinese_learning_hours :
  total_chinese_hours 4 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ryan_chinese_learning_hours_l454_45411


namespace NUMINAMATH_CALUDE_sock_pairs_calculation_l454_45463

def calculate_sock_pairs (initial_socks thrown_away_socks new_socks : ℕ) : ℕ :=
  ((initial_socks - thrown_away_socks) + new_socks) / 2

theorem sock_pairs_calculation (initial_socks thrown_away_socks new_socks : ℕ) 
  (h1 : initial_socks ≥ thrown_away_socks) :
  calculate_sock_pairs initial_socks thrown_away_socks new_socks = 
  ((initial_socks - thrown_away_socks) + new_socks) / 2 := by
  sorry

#eval calculate_sock_pairs 28 4 36

end NUMINAMATH_CALUDE_sock_pairs_calculation_l454_45463


namespace NUMINAMATH_CALUDE_oil_leak_total_l454_45492

theorem oil_leak_total (before_repairs : ℕ) (during_repairs : ℕ) 
  (h1 : before_repairs = 6522) 
  (h2 : during_repairs = 5165) : 
  before_repairs + during_repairs = 11687 :=
by sorry

end NUMINAMATH_CALUDE_oil_leak_total_l454_45492


namespace NUMINAMATH_CALUDE_a_10_value_l454_45429

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_10_value
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 12 = 16)
  (h_7 : a 7 = 1) :
  a 10 = 15 := by
  sorry

end NUMINAMATH_CALUDE_a_10_value_l454_45429


namespace NUMINAMATH_CALUDE_smallest_positive_period_sin_l454_45496

/-- The smallest positive period of y = 5 * sin(3x + π/6) is 2π/3 --/
theorem smallest_positive_period_sin (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 5 * Real.sin (3 * x + π / 6)
  ∃ T : ℝ, T > 0 ∧ T = 2 * π / 3 ∧ (∀ t : ℝ, f (x + T) = f x) ∧
  (∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, f (x + S) = f x) → T ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_period_sin_l454_45496


namespace NUMINAMATH_CALUDE_larger_number_is_eight_l454_45418

theorem larger_number_is_eight (x y : ℕ) (h1 : x = 2 * y) (h2 : x * y = 40) (h3 : x + y = 14) : x = 8 := by
  sorry

#check larger_number_is_eight

end NUMINAMATH_CALUDE_larger_number_is_eight_l454_45418


namespace NUMINAMATH_CALUDE_factorization_equality_l454_45488

theorem factorization_equality (x : ℝ) : -4 * x^2 + 16 = 4 * (2 + x) * (2 - x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l454_45488


namespace NUMINAMATH_CALUDE_real_solutions_quadratic_l454_45490

theorem real_solutions_quadratic (x : ℝ) : 
  (∃ y : ℝ, 4 * y^2 + 4 * x * y + x + 6 = 0) ↔ x ≤ -2 ∨ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_real_solutions_quadratic_l454_45490


namespace NUMINAMATH_CALUDE_product_of_roots_l454_45437

theorem product_of_roots (x : ℝ) : 
  (24 * x^2 + 36 * x - 648 = 0) → 
  (∃ r₁ r₂ : ℝ, (24 * r₁^2 + 36 * r₁ - 648 = 0) ∧ 
                (24 * r₂^2 + 36 * r₂ - 648 = 0) ∧ 
                (r₁ * r₂ = -27)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l454_45437


namespace NUMINAMATH_CALUDE_prob_sum_le_10_is_25_72_l454_45476

/-- The number of possible outcomes when rolling three fair six-sided dice -/
def total_outcomes : ℕ := 6^3

/-- The number of favorable outcomes (sum ≤ 10) when rolling three fair six-sided dice -/
def favorable_outcomes : ℕ := 75

/-- The probability of rolling three fair six-sided dice and obtaining a sum less than or equal to 10 -/
def prob_sum_le_10 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_le_10_is_25_72 : prob_sum_le_10 = 25 / 72 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_le_10_is_25_72_l454_45476


namespace NUMINAMATH_CALUDE_comparison_theorem_l454_45452

theorem comparison_theorem :
  (-3/4 : ℚ) > -4/5 ∧ (3 : ℝ) > Real.rpow 9 (1/3) := by sorry

end NUMINAMATH_CALUDE_comparison_theorem_l454_45452


namespace NUMINAMATH_CALUDE_triangle_area_passes_through_1_2_passes_through_neg1_6_x_intercept_correct_y_intercept_correct_l454_45471

/-- A linear function passing through (1, 2) and (-1, 6) -/
def linear_function (x : ℝ) : ℝ := -2 * x + 4

/-- The x-intercept of the linear function -/
def x_intercept : ℝ := 2

/-- The y-intercept of the linear function -/
def y_intercept : ℝ := 4

/-- Theorem: The area of the triangle formed by the x-intercept, y-intercept, and origin is 4 -/
theorem triangle_area : (1/2 : ℝ) * x_intercept * y_intercept = 4 := by
  sorry

/-- The linear function passes through (1, 2) -/
theorem passes_through_1_2 : linear_function 1 = 2 := by
  sorry

/-- The linear function passes through (-1, 6) -/
theorem passes_through_neg1_6 : linear_function (-1) = 6 := by
  sorry

/-- The x-intercept is correct -/
theorem x_intercept_correct : linear_function x_intercept = 0 := by
  sorry

/-- The y-intercept is correct -/
theorem y_intercept_correct : linear_function 0 = y_intercept := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_passes_through_1_2_passes_through_neg1_6_x_intercept_correct_y_intercept_correct_l454_45471


namespace NUMINAMATH_CALUDE_sine_arithmetic_sequence_l454_45487

open Real

theorem sine_arithmetic_sequence (a : ℝ) : 
  0 < a ∧ a < 2 * π →
  (∃ r : ℝ, sin a + r = sin (2 * a) ∧ sin (2 * a) + r = sin (3 * a)) ↔ 
  a = π / 2 ∨ a = 3 * π / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_arithmetic_sequence_l454_45487


namespace NUMINAMATH_CALUDE_triangle_sqrt_inequality_l454_45406

theorem triangle_sqrt_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hac : a + c > b) :
  (Real.sqrt a + Real.sqrt b > Real.sqrt c) ∧ 
  (Real.sqrt b + Real.sqrt c > Real.sqrt a) ∧ 
  (Real.sqrt a + Real.sqrt c > Real.sqrt b) := by
sorry

end NUMINAMATH_CALUDE_triangle_sqrt_inequality_l454_45406


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_7_with_different_digits_l454_45436

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

theorem largest_four_digit_divisible_by_7_with_different_digits :
  ∃ (n : ℕ), is_four_digit n ∧ n % 7 = 0 ∧ has_different_digits n ∧
  ∀ (m : ℕ), is_four_digit m → m % 7 = 0 → has_different_digits m → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_7_with_different_digits_l454_45436


namespace NUMINAMATH_CALUDE_equation_solution_l454_45408

theorem equation_solution (x y z : ℕ) :
  (x : ℚ) + 1 / ((y : ℚ) + 1 / (z : ℚ)) = 10 / 7 →
  x = 1 ∧ y = 2 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l454_45408


namespace NUMINAMATH_CALUDE_max_discount_rate_l454_45422

theorem max_discount_rate (cost : ℝ) (original_price : ℝ) 
  (h1 : cost = 4) (h2 : original_price = 5) : 
  ∃ (max_discount : ℝ), 
    (∀ (discount : ℝ), discount ≤ max_discount → 
      (original_price * (1 - discount / 100) - cost) / cost ≥ 0.1) ∧
    (∀ (discount : ℝ), discount > max_discount → 
      (original_price * (1 - discount / 100) - cost) / cost < 0.1) ∧
    max_discount = 12 :=
sorry

end NUMINAMATH_CALUDE_max_discount_rate_l454_45422


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_l454_45427

theorem quadratic_always_nonnegative (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 2) * x + (1/4 : ℝ) ≥ 0) ↔ 1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_l454_45427


namespace NUMINAMATH_CALUDE_opposite_point_exists_l454_45420

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define diametrically opposite points
def DiametricallyOpposite (c : Circle) (p q : ℝ × ℝ) : Prop :=
  PointOnCircle c p ∧ PointOnCircle c q ∧
  (p.1 - c.center.1) = -(q.1 - c.center.1) ∧
  (p.2 - c.center.2) = -(q.2 - c.center.2)

-- Theorem statement
theorem opposite_point_exists (c : Circle) (A₁ : ℝ × ℝ) 
  (h : PointOnCircle c A₁) : 
  ∃ B₂ : ℝ × ℝ, DiametricallyOpposite c A₁ B₂ := by
  sorry

end NUMINAMATH_CALUDE_opposite_point_exists_l454_45420


namespace NUMINAMATH_CALUDE_laptop_repair_cost_laptop_repair_cost_proof_l454_45440

/-- The cost of a laptop repair given the following conditions:
  * Phone repair costs $11
  * Computer repair costs $18
  * 5 phone repairs, 2 laptop repairs, and 2 computer repairs were performed
  * Total earnings were $121
-/
theorem laptop_repair_cost : ℕ :=
  let phone_cost : ℕ := 11
  let computer_cost : ℕ := 18
  let phone_repairs : ℕ := 5
  let laptop_repairs : ℕ := 2
  let computer_repairs : ℕ := 2
  let total_earnings : ℕ := 121
  15

theorem laptop_repair_cost_proof :
  (let phone_cost : ℕ := 11
   let computer_cost : ℕ := 18
   let phone_repairs : ℕ := 5
   let laptop_repairs : ℕ := 2
   let computer_repairs : ℕ := 2
   let total_earnings : ℕ := 121
   laptop_repair_cost = 15) :=
by sorry

end NUMINAMATH_CALUDE_laptop_repair_cost_laptop_repair_cost_proof_l454_45440


namespace NUMINAMATH_CALUDE_store_discount_income_increase_l454_45419

theorem store_discount_income_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (discount_rate : ℝ) 
  (quantity_increase_rate : ℝ) 
  (h1 : discount_rate = 0.1)
  (h2 : quantity_increase_rate = 0.12)
  : (1 + quantity_increase_rate) * (1 - discount_rate) - 1 = 0.008 := by
  sorry

end NUMINAMATH_CALUDE_store_discount_income_increase_l454_45419


namespace NUMINAMATH_CALUDE_always_negative_monotone_decreasing_l454_45430

/-- The function f(x) = kx^2 - 2x + 4k -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 4 * k

/-- Theorem 1: f(x) is always less than zero on ℝ iff k < -1/2 -/
theorem always_negative (k : ℝ) : (∀ x : ℝ, f k x < 0) ↔ k < -1/2 := by sorry

/-- Theorem 2: f(x) is monotonically decreasing on [2, 4] iff k ≤ 1/4 -/
theorem monotone_decreasing (k : ℝ) : 
  (∀ x y : ℝ, 2 ≤ x ∧ x < y ∧ y ≤ 4 → f k x > f k y) ↔ k ≤ 1/4 := by sorry

end NUMINAMATH_CALUDE_always_negative_monotone_decreasing_l454_45430


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l454_45469

theorem fruit_seller_apples (initial_stock : ℕ) (remaining_stock : ℕ) 
  (sell_percentage : ℚ) (h1 : sell_percentage = 40 / 100) 
  (h2 : remaining_stock = 420) 
  (h3 : remaining_stock = initial_stock - (sell_percentage * initial_stock).floor) : 
  initial_stock = 700 := by
sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l454_45469


namespace NUMINAMATH_CALUDE_juan_marbles_count_l454_45449

/-- The number of marbles Connie has -/
def connie_marbles : ℕ := 39

/-- The number of additional marbles Juan has compared to Connie -/
def juan_additional_marbles : ℕ := 25

/-- The total number of marbles Juan has -/
def juan_marbles : ℕ := connie_marbles + juan_additional_marbles

theorem juan_marbles_count : juan_marbles = 64 := by sorry

end NUMINAMATH_CALUDE_juan_marbles_count_l454_45449


namespace NUMINAMATH_CALUDE_locus_is_S_l454_45460

/-- A point moving along a line with constant velocity -/
structure MovingPoint where
  line : Set ℝ × ℝ  -- Represents a line in 2D space
  velocity : ℝ

/-- The locus of lines XX' -/
def locus (X X' : MovingPoint) : Set (Set ℝ × ℝ) := sorry

/-- The specific set S that represents the correct locus -/
def S : Set (Set ℝ × ℝ) := sorry

/-- Theorem stating that the locus of lines XX' is the set S -/
theorem locus_is_S (X X' : MovingPoint) (h : X.velocity ≠ X'.velocity) :
  locus X X' = S := by sorry

end NUMINAMATH_CALUDE_locus_is_S_l454_45460


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l454_45416

/-- Given a geometric sequence {a_n} with common ratio q and S_n as the sum of its first n terms,
    if 3S_3 = a_4 - 2 and 3S_2 = a_3 - 2, then q = 4 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q))
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : 3 * S 3 = a 4 - 2)
  (h4 : 3 * S 2 = a 3 - 2) :
  q = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l454_45416


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l454_45439

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 110) :
  x^2 + y^2 = 1380 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l454_45439


namespace NUMINAMATH_CALUDE_concurrency_condition_l454_45428

/-- Triangle ABC with sides a, b, and c, where AD is an altitude, BE is an angle bisector, and CF is a median -/
structure Triangle :=
  (a b c : ℝ)
  (ad_is_altitude : Bool)
  (be_is_angle_bisector : Bool)
  (cf_is_median : Bool)

/-- The lines AD, BE, and CF are concurrent -/
def are_concurrent (t : Triangle) : Prop := sorry

/-- Theorem stating the condition for concurrency of AD, BE, and CF -/
theorem concurrency_condition (t : Triangle) : 
  are_concurrent t ↔ t.a^2 * (t.a - t.c) = (t.b^2 - t.c^2) * (t.a + t.c) :=
sorry

end NUMINAMATH_CALUDE_concurrency_condition_l454_45428


namespace NUMINAMATH_CALUDE_f_derivative_at_pi_half_l454_45486

noncomputable def f (x : ℝ) := Real.exp x * Real.cos x

theorem f_derivative_at_pi_half : 
  deriv f (Real.pi / 2) = -Real.exp (Real.pi / 2) := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_pi_half_l454_45486


namespace NUMINAMATH_CALUDE_exists_m_with_all_digits_l454_45497

/-- For any positive integer n, there exists a positive integer m such that
    the decimal representation of m * n contains all digits from 0 to 9. -/
theorem exists_m_with_all_digits (n : ℕ+) : ∃ m : ℕ+, ∀ d : Fin 10, ∃ k : ℕ,
  (m * n : ℕ) / 10^k % 10 = d.val :=
sorry

end NUMINAMATH_CALUDE_exists_m_with_all_digits_l454_45497


namespace NUMINAMATH_CALUDE_diophantine_equation_min_max_sum_l454_45458

theorem diophantine_equation_min_max_sum : 
  ∃ (p q : ℕ), 
    (∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ 6 * x + 7 * y = 2012 → x + y ≥ p) ∧
    (∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ 6 * x + 7 * y = 2012 → x + y ≤ q) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℕ), 
      x₁ > 0 ∧ y₁ > 0 ∧ 6 * x₁ + 7 * y₁ = 2012 ∧ x₁ + y₁ = p ∧
      x₂ > 0 ∧ y₂ > 0 ∧ 6 * x₂ + 7 * y₂ = 2012 ∧ x₂ + y₂ = q) ∧
    p + q = 623 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_min_max_sum_l454_45458


namespace NUMINAMATH_CALUDE_computer_table_price_l454_45466

theorem computer_table_price (cost_price : ℝ) (markup_percentage : ℝ) 
  (h1 : cost_price = 4090.9090909090905)
  (h2 : markup_percentage = 32) :
  cost_price * (1 + markup_percentage / 100) = 5400 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_price_l454_45466


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_three_sqrt_ten_l454_45453

theorem complex_magnitude_equals_three_sqrt_ten (x : ℝ) :
  x > 0 → Complex.abs (-3 + x * Complex.I) = 3 * Real.sqrt 10 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_three_sqrt_ten_l454_45453


namespace NUMINAMATH_CALUDE_sum_x_y_value_l454_45447

theorem sum_x_y_value (x y : ℚ) 
  (eq1 : 5 * x - 3 * y = 17)
  (eq2 : 3 * x + 5 * y = 1) : 
  x + y = 36 / 85 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_value_l454_45447


namespace NUMINAMATH_CALUDE_perpendicular_lines_not_both_perpendicular_to_plane_l454_45454

-- Define the plane α
variable (α : Set (ℝ × ℝ × ℝ))

-- Define lines a and b
variable (a b : Set (ℝ × ℝ × ℝ))

-- Define what it means for two lines to be perpendicular
def perpendicular (l1 l2 : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define what it means for a line to be perpendicular to a plane
def perpendicular_to_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- The theorem
theorem perpendicular_lines_not_both_perpendicular_to_plane :
  a ≠ b →
  perpendicular a b →
  ¬(perpendicular_to_plane a α ∧ perpendicular_to_plane b α) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_not_both_perpendicular_to_plane_l454_45454


namespace NUMINAMATH_CALUDE_triangle_angles_sum_l454_45498

theorem triangle_angles_sum (x y : ℕ+) : 
  (5 * x + 3 * y : ℕ) + (3 * x + 20 : ℕ) + (10 * y + 30 : ℕ) = 180 → x + y = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_sum_l454_45498


namespace NUMINAMATH_CALUDE_different_group_choices_l454_45425

theorem different_group_choices (n : Nat) (h : n = 3) : 
  n^2 - n = 6 := by
  sorry

#check different_group_choices

end NUMINAMATH_CALUDE_different_group_choices_l454_45425


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l454_45441

-- Define the function g
noncomputable def g : ℝ → ℤ
| x => if x > -3 then Int.ceil (2 / (x + 3))
       else if x < -3 then Int.floor (2 / (x + 3))
       else 0  -- arbitrary value for x = -3, as g is not defined there

-- Theorem statement
theorem zero_not_in_range_of_g : ∀ x : ℝ, g x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l454_45441


namespace NUMINAMATH_CALUDE_carys_savings_l454_45474

/-- Problem: Cary's Lawn Mowing Savings --/
theorem carys_savings (shoe_cost : ℕ) (saved : ℕ) (earnings_per_lawn : ℕ) (lawns_per_weekend : ℕ)
  (h1 : shoe_cost = 120)
  (h2 : saved = 30)
  (h3 : earnings_per_lawn = 5)
  (h4 : lawns_per_weekend = 3) :
  (shoe_cost - saved) / (lawns_per_weekend * earnings_per_lawn) = 6 := by
  sorry

end NUMINAMATH_CALUDE_carys_savings_l454_45474


namespace NUMINAMATH_CALUDE_clearview_soccer_league_members_l454_45459

/-- Represents the Clearview Soccer League --/
structure SoccerLeague where
  sockPrice : ℕ
  tshirtPriceIncrease : ℕ
  hatPrice : ℕ
  totalExpenditure : ℕ

/-- Calculates the number of members in the league --/
def calculateMembers (league : SoccerLeague) : ℕ :=
  let tshirtPrice := league.sockPrice + league.tshirtPriceIncrease
  let memberCost := 2 * (league.sockPrice + tshirtPrice + league.hatPrice)
  league.totalExpenditure / memberCost

/-- Theorem stating the number of members in the Clearview Soccer League --/
theorem clearview_soccer_league_members :
  let league := SoccerLeague.mk 3 7 2 3516
  calculateMembers league = 117 := by
  sorry

end NUMINAMATH_CALUDE_clearview_soccer_league_members_l454_45459


namespace NUMINAMATH_CALUDE_fraction_exponent_product_l454_45417

theorem fraction_exponent_product : (5 / 6 : ℚ)^2 * (2 / 3 : ℚ)^3 = 50 / 243 := by sorry

end NUMINAMATH_CALUDE_fraction_exponent_product_l454_45417


namespace NUMINAMATH_CALUDE_christinas_speed_l454_45413

/-- Prove Christina's speed given the problem conditions -/
theorem christinas_speed (initial_distance : ℝ) (jacks_speed : ℝ) (lindys_speed : ℝ) 
  (lindys_total_distance : ℝ) (h1 : initial_distance = 360) 
  (h2 : jacks_speed = 5) (h3 : lindys_speed = 12) (h4 : lindys_total_distance = 360) :
  ∃ (christinas_speed : ℝ), christinas_speed = 7 := by
  sorry


end NUMINAMATH_CALUDE_christinas_speed_l454_45413


namespace NUMINAMATH_CALUDE_christy_tanya_spending_ratio_l454_45402

/-- Represents the spending of Christy and Tanya at Target -/
structure TargetShopping where
  christy_spent : ℕ
  tanya_face_moisturizer_price : ℕ
  tanya_face_moisturizer_quantity : ℕ
  tanya_body_lotion_price : ℕ
  tanya_body_lotion_quantity : ℕ
  total_spent : ℕ

/-- Calculates Tanya's total spending -/
def tanya_total_spent (shopping : TargetShopping) : ℕ :=
  shopping.tanya_face_moisturizer_price * shopping.tanya_face_moisturizer_quantity +
  shopping.tanya_body_lotion_price * shopping.tanya_body_lotion_quantity

/-- Theorem stating the ratio of Christy's spending to Tanya's spending -/
theorem christy_tanya_spending_ratio (shopping : TargetShopping)
  (h1 : shopping.tanya_face_moisturizer_price = 50)
  (h2 : shopping.tanya_face_moisturizer_quantity = 2)
  (h3 : shopping.tanya_body_lotion_price = 60)
  (h4 : shopping.tanya_body_lotion_quantity = 4)
  (h5 : shopping.total_spent = 1020)
  (h6 : shopping.christy_spent + tanya_total_spent shopping = shopping.total_spent) :
  2 * tanya_total_spent shopping = shopping.christy_spent := by
  sorry

#check christy_tanya_spending_ratio

end NUMINAMATH_CALUDE_christy_tanya_spending_ratio_l454_45402


namespace NUMINAMATH_CALUDE_rachel_leah_age_difference_l454_45424

/-- Given that Rachel is 19 years old and the sum of Rachel and Leah's ages is 34,
    prove that Rachel is 4 years older than Leah. -/
theorem rachel_leah_age_difference :
  ∀ (rachel_age leah_age : ℕ),
  rachel_age = 19 →
  rachel_age + leah_age = 34 →
  rachel_age - leah_age = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_rachel_leah_age_difference_l454_45424


namespace NUMINAMATH_CALUDE_larger_number_proof_l454_45494

/-- Given two positive integers with HCF 23 and LCM factors 16 and 17, prove the larger is 391 -/
theorem larger_number_proof (a b : ℕ+) : 
  Nat.gcd a b = 23 → 
  Nat.lcm a b = 23 * 16 * 17 → 
  max a b = 391 := by sorry

end NUMINAMATH_CALUDE_larger_number_proof_l454_45494


namespace NUMINAMATH_CALUDE_complex_number_equality_l454_45470

theorem complex_number_equality : ∀ z : ℂ, z = 1 - 2*I → z = -I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l454_45470


namespace NUMINAMATH_CALUDE_unique_half_rectangle_l454_45489

/-- Given a rectangle R with dimensions a and b (a < b), prove that there exists exactly one rectangle
    with dimensions x and y such that x < b, y < b, its perimeter is half of R's, and its area is half of R's. -/
theorem unique_half_rectangle (a b : ℝ) (hab : a < b) :
  ∃! (x y : ℝ), x < b ∧ y < b ∧ 2 * (x + y) = a + b ∧ x * y = a * b / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_half_rectangle_l454_45489


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l454_45412

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = 2*c ∨ x = 3*d) : 
  c = 1/6 ∧ d = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l454_45412


namespace NUMINAMATH_CALUDE_extremum_condition_l454_45480

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f with respect to x -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_condition (a b : ℝ) : 
  (f a b 1 = 10) ∧ (f_derivative a b 1 = 0) → (a = 4 ∧ b = -11) :=
sorry

end NUMINAMATH_CALUDE_extremum_condition_l454_45480


namespace NUMINAMATH_CALUDE_distinct_sums_count_l454_45438

def S : Finset ℕ := {2, 5, 8, 11, 14, 17, 20}

def fourDistinctSum (s : Finset ℕ) : Finset ℕ :=
  (s.powerset.filter (λ t => t.card = 4)).image (λ t => t.sum id)

theorem distinct_sums_count : (fourDistinctSum S).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_count_l454_45438


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l454_45485

theorem quadratic_points_relationship :
  let f (x : ℝ) := (x - 2)^2 - 1
  let y₁ := f 4
  let y₂ := f (Real.sqrt 2)
  let y₃ := f (-2)
  y₃ > y₁ ∧ y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l454_45485


namespace NUMINAMATH_CALUDE_polynomial_division_l454_45423

theorem polynomial_division (x : ℝ) :
  x^5 - 17*x^3 + 8*x^2 - 9*x + 12 = (x - 3) * (x^4 + 3*x^3 - 8*x^2 - 16*x - 57) + (-159) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l454_45423


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l454_45464

/-- Given a rectangular prism with side areas 15, 10, and 6 (in square inches),
    where the dimension associated with the smallest area is the hypotenuse of a right triangle
    formed by the other two dimensions, prove that the volume of the prism is 30 cubic inches. -/
theorem rectangular_prism_volume (a b c : ℝ) 
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : a * c = 6)
  (h4 : c^2 = a^2 + b^2) : 
  a * b * c = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l454_45464


namespace NUMINAMATH_CALUDE_sector_central_angle_l454_45435

theorem sector_central_angle (r : ℝ) (A : ℝ) (θ : ℝ) : 
  r = 2 → A = 4 → A = (1/2) * r^2 * θ → θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l454_45435


namespace NUMINAMATH_CALUDE_charles_pictures_l454_45432

theorem charles_pictures (total_papers : ℕ) (today_pictures : ℕ) (yesterday_before_work : ℕ) (papers_left : ℕ) 
  (h1 : total_papers = 20)
  (h2 : today_pictures = 6)
  (h3 : yesterday_before_work = 6)
  (h4 : papers_left = 2) :
  total_papers - (today_pictures + yesterday_before_work) - papers_left = 6 := by
  sorry

end NUMINAMATH_CALUDE_charles_pictures_l454_45432


namespace NUMINAMATH_CALUDE_carpet_price_falls_below_8_at_945_l454_45456

def initial_price : ℝ := 10.00
def reduction_rate : ℝ := 0.9
def target_price : ℝ := 8.00

def price_after_n_reductions (n : ℕ) : ℝ :=
  initial_price * (reduction_rate ^ n)

theorem carpet_price_falls_below_8_at_945 :
  price_after_n_reductions 3 < target_price ∧
  price_after_n_reductions 2 ≥ target_price :=
by sorry

end NUMINAMATH_CALUDE_carpet_price_falls_below_8_at_945_l454_45456


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l454_45455

theorem arithmetic_mean_problem (x : ℝ) : (x + 1 = (5 + 7) / 2) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l454_45455


namespace NUMINAMATH_CALUDE_pascal_interior_sum_l454_45484

/-- Sum of interior numbers in the n-th row of Pascal's Triangle -/
def sumInteriorNumbers (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_interior_sum :
  (sumInteriorNumbers 5 = 14) →
  (sumInteriorNumbers 6 = 30) →
  (sumInteriorNumbers 8 = 126) :=
by
  sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_l454_45484


namespace NUMINAMATH_CALUDE_exists_solution_in_interval_l454_45426

theorem exists_solution_in_interval : 
  ∃ z : ℝ, -10 ≤ z ∧ z ≤ 10 ∧ Real.exp (2 * z) = (z - 2) / (z + 2) := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_in_interval_l454_45426


namespace NUMINAMATH_CALUDE_cupcakes_baked_and_iced_l454_45491

/-- Represents the number of cups of sugar in a bag -/
def sugar_per_bag : ℕ := 6

/-- Represents the number of bags of sugar bought -/
def bags_bought : ℕ := 2

/-- Represents the number of cups of sugar Lillian has at home -/
def sugar_at_home : ℕ := 3

/-- Represents the number of cups of sugar needed for batter per dozen cupcakes -/
def sugar_for_batter : ℕ := 1

/-- Represents the number of cups of sugar needed for frosting per dozen cupcakes -/
def sugar_for_frosting : ℕ := 2

/-- Theorem stating that Lillian can bake and ice 5 dozen cupcakes -/
theorem cupcakes_baked_and_iced : ℕ := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_baked_and_iced_l454_45491


namespace NUMINAMATH_CALUDE_bryan_pushups_l454_45457

/-- The number of push-ups Bryan did in total -/
def total_pushups (sets : ℕ) (pushups_per_set : ℕ) (reduction : ℕ) : ℕ :=
  (sets - 1) * pushups_per_set + (pushups_per_set - reduction)

/-- Proof that Bryan did 40 push-ups in total -/
theorem bryan_pushups :
  total_pushups 3 15 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bryan_pushups_l454_45457


namespace NUMINAMATH_CALUDE_sum_of_common_terms_l454_45410

/-- The sequence formed by common terms of {2n-1} and {3n-2} in ascending order -/
def a : ℕ → ℕ := sorry

/-- The sum of the first n terms of sequence a -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the first n terms of sequence a is 3n^2 - 2n -/
theorem sum_of_common_terms (n : ℕ) : S n = 3 * n^2 - 2 * n := by sorry

end NUMINAMATH_CALUDE_sum_of_common_terms_l454_45410


namespace NUMINAMATH_CALUDE_program_output_l454_45451

def program (a b : ℤ) : ℤ :=
  if a > b then a else b

theorem program_output : program 2 3 = 3 := by sorry

end NUMINAMATH_CALUDE_program_output_l454_45451


namespace NUMINAMATH_CALUDE_distribution_schemes_7_5_2_l454_45465

/-- The number of ways to distribute n identical items among k recipients,
    where two recipients must receive at least m items each. -/
def distribution_schemes (n k m : ℕ) : ℕ :=
  sorry

/-- The specific case for 7 items, 5 recipients, and 2 items minimum for two recipients -/
theorem distribution_schemes_7_5_2 :
  distribution_schemes 7 5 2 = 35 :=
sorry

end NUMINAMATH_CALUDE_distribution_schemes_7_5_2_l454_45465


namespace NUMINAMATH_CALUDE_sum_positive_from_inequality_l454_45404

theorem sum_positive_from_inequality (x y : ℝ) 
  (h : (3:ℝ)^x + (5:ℝ)^y > (3:ℝ)^(-y) + (5:ℝ)^(-x)) : 
  x + y > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_from_inequality_l454_45404


namespace NUMINAMATH_CALUDE_program_cost_calculation_l454_45479

/-- Calculates the total cost for running a computer program -/
theorem program_cost_calculation (program_time_seconds : ℝ) : 
  let milliseconds_per_second : ℝ := 1000
  let overhead_cost : ℝ := 1.07
  let cost_per_millisecond : ℝ := 0.023
  let tape_mounting_cost : ℝ := 5.35
  let program_time_milliseconds : ℝ := program_time_seconds * milliseconds_per_second
  let computer_time_cost : ℝ := program_time_milliseconds * cost_per_millisecond
  let total_cost : ℝ := overhead_cost + computer_time_cost + tape_mounting_cost
  program_time_seconds = 1.5 → total_cost = 40.92 := by
  sorry

end NUMINAMATH_CALUDE_program_cost_calculation_l454_45479


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l454_45462

theorem nested_sqrt_value :
  ∃ y : ℝ, y = Real.sqrt (3 + y) → y = (1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l454_45462


namespace NUMINAMATH_CALUDE_not_prime_29n_plus_11_l454_45415

theorem not_prime_29n_plus_11 (n : ℕ+) 
  (h1 : ∃ x : ℕ, 3 * n + 1 = x^2) 
  (h2 : ∃ y : ℕ, 10 * n + 1 = y^2) : 
  ¬ Nat.Prime (29 * n + 11) := by
sorry

end NUMINAMATH_CALUDE_not_prime_29n_plus_11_l454_45415


namespace NUMINAMATH_CALUDE_trigonometric_expression_l454_45448

theorem trigonometric_expression (α : Real) 
  (h : Real.sin (π/4 + α) = 1/2) : 
  (Real.sin (5*π/4 + α) / Real.cos (9*π/4 + α)) * Real.cos (7*π/4 - α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_l454_45448


namespace NUMINAMATH_CALUDE_system_solution_l454_45407

theorem system_solution (x y k : ℝ) : 
  x + 2*y = k - 1 → 
  2*x + y = 5*k + 4 → 
  x + y = 5 → 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l454_45407


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l454_45431

/-- The radius of a circle inscribed in a rhombus with given diagonals -/
theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 30) :
  let a := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  let r := (d1 * d2) / (8 * a)
  r = 105 / (2 * Real.sqrt 274) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l454_45431


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l454_45493

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^24 + 8^15) ∧ ∀ q, Nat.Prime q → q ∣ (3^24 + 8^15) → p ≤ q := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l454_45493


namespace NUMINAMATH_CALUDE_solution_part_i_solution_part_ii_l454_45421

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

-- Theorem for part (I)
theorem solution_part_i :
  {x : ℝ | f x ≤ 4} = {x : ℝ | -5/3 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part (II)
theorem solution_part_ii :
  {a : ℝ | ∀ x ∈ {x : ℝ | f x ≤ 4}, |x + 3| + |x + a| < x + 6} =
  {a : ℝ | -4/3 < a ∧ a < 2} := by sorry

end NUMINAMATH_CALUDE_solution_part_i_solution_part_ii_l454_45421


namespace NUMINAMATH_CALUDE_centroid_eq_circumcenter_implies_equilateral_l454_45475

/-- A triangle in a 2D Euclidean space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- A triangle is equilateral if all its sides have equal length -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- 
If the centroid of a triangle coincides with its circumcenter, 
then the triangle is equilateral
-/
theorem centroid_eq_circumcenter_implies_equilateral (t : Triangle) :
  centroid t = circumcenter t → is_equilateral t := by sorry

end NUMINAMATH_CALUDE_centroid_eq_circumcenter_implies_equilateral_l454_45475


namespace NUMINAMATH_CALUDE_joe_is_94_point_5_inches_tall_l454_45472

-- Define the heights of Sara, Joe, and Alex
variable (S J A : ℝ)

-- Define the conditions from the problem
def combined_height : ℝ → ℝ → ℝ → Prop :=
  λ s j a => s + j + a = 180

def joe_height : ℝ → ℝ → Prop :=
  λ s j => j = 2 * s + 6

def alex_height : ℝ → ℝ → Prop :=
  λ s a => a = s - 3

-- Theorem statement
theorem joe_is_94_point_5_inches_tall
  (h1 : combined_height S J A)
  (h2 : joe_height S J)
  (h3 : alex_height S A) :
  J = 94.5 :=
sorry

end NUMINAMATH_CALUDE_joe_is_94_point_5_inches_tall_l454_45472


namespace NUMINAMATH_CALUDE_no_rain_probability_l454_45477

theorem no_rain_probability (p : ℚ) (h : p = 2/3) : (1 - p)^4 = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l454_45477


namespace NUMINAMATH_CALUDE_set_intersection_equality_l454_45409

def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem set_intersection_equality : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 16} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l454_45409


namespace NUMINAMATH_CALUDE_significant_difference_l454_45499

/-- Represents the distribution of X, where X is the number of specified two mice
    assigned to the control group out of 40 total mice --/
def distribution_X : Fin 3 → ℚ
| 0 => 19/78
| 1 => 20/39
| 2 => 19/78

/-- The expectation of X --/
def E_X : ℚ := 1

/-- The median of the increase in body weight of all 40 mice --/
def median_weight : ℝ := 23.4

/-- The contingency table of mice counts below and above median --/
def contingency_table : Fin 2 → Fin 2 → ℕ
| 0, 0 => 6  -- Control group, below median
| 0, 1 => 14 -- Control group, above or equal to median
| 1, 0 => 14 -- Experimental group, below median
| 1, 1 => 6  -- Experimental group, above or equal to median

/-- The K² statistic --/
def K_squared : ℝ := 6.400

/-- The critical value for 95% confidence level --/
def critical_value_95 : ℝ := 3.841

/-- Theorem stating that the K² value is greater than the critical value,
    indicating a significant difference between groups --/
theorem significant_difference : K_squared > critical_value_95 := by sorry

end NUMINAMATH_CALUDE_significant_difference_l454_45499


namespace NUMINAMATH_CALUDE_root_transformation_l454_45467

/-- Given that s₁, s₂, and s₃ are the roots of x³ - 4x² + 9 = 0,
    prove that 3s₁, 3s₂, and 3s₃ are the roots of x³ - 12x² + 243 = 0 -/
theorem root_transformation (s₁ s₂ s₃ : ℂ) : 
  (s₁^3 - 4*s₁^2 + 9 = 0) ∧ 
  (s₂^3 - 4*s₂^2 + 9 = 0) ∧ 
  (s₃^3 - 4*s₃^2 + 9 = 0) → 
  ((3*s₁)^3 - 12*(3*s₁)^2 + 243 = 0) ∧ 
  ((3*s₂)^3 - 12*(3*s₂)^2 + 243 = 0) ∧ 
  ((3*s₃)^3 - 12*(3*s₃)^2 + 243 = 0) := by
sorry

end NUMINAMATH_CALUDE_root_transformation_l454_45467


namespace NUMINAMATH_CALUDE_gcd_3Sn_nplus1_le_1_l454_45468

def square_sum (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem gcd_3Sn_nplus1_le_1 (n : ℕ+) :
  Nat.gcd (3 * square_sum n) (n + 1) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_gcd_3Sn_nplus1_le_1_l454_45468


namespace NUMINAMATH_CALUDE_men_working_count_l454_45481

/-- Represents the amount of work done by one person in one hour -/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers -/
structure WorkGroup where
  size : ℕ
  work_rate : WorkRate
  days : ℕ
  hours_per_day : ℕ

/-- The total work done by a group is the product of their size, work rate, days, and hours per day -/
def total_work (group : WorkGroup) : ℝ :=
  group.size * group.work_rate.rate * group.days * group.hours_per_day

/-- Given the conditions of the problem, prove that the number of men working is 15 -/
theorem men_working_count (men_group women_group : WorkGroup) :
  men_group.days = 21 →
  men_group.hours_per_day = 8 →
  women_group.size = 21 →
  women_group.days = 60 →
  women_group.hours_per_day = 3 →
  3 * women_group.work_rate.rate = 2 * men_group.work_rate.rate →
  total_work men_group = total_work women_group →
  men_group.size = 15 := by
  sorry

end NUMINAMATH_CALUDE_men_working_count_l454_45481


namespace NUMINAMATH_CALUDE_certain_number_calculation_l454_45405

theorem certain_number_calculation : 
  ∃ x : ℝ, abs (3889 + 12.952 - 47.95000000000027 - x) < 0.0005 ∧ 
           abs (x - 3854.002) < 0.0005 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l454_45405


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l454_45450

-- Define the painting dimensions
def painting_width : ℝ := 20
def painting_height : ℝ := 30

-- Define the frame width variable
variable (x : ℝ)

-- Define the framed painting dimensions
def framed_width (x : ℝ) : ℝ := painting_width + 2 * x
def framed_height (x : ℝ) : ℝ := painting_height + 4 * x

-- State the theorem
theorem framed_painting_ratio :
  (∃ x > 0, framed_width x * framed_height x = 2 * painting_width * painting_height) →
  (min (framed_width x) (framed_height x)) / (max (framed_width x) (framed_height x)) = 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_framed_painting_ratio_l454_45450


namespace NUMINAMATH_CALUDE_number_ratio_l454_45400

theorem number_ratio (x y z : ℝ) (k : ℝ) : 
  y = 2 * x →
  z = k * y →
  (x + y + z) / 3 = 165 →
  y = 90 →
  z / y = 4 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l454_45400


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l454_45414

theorem diophantine_equation_solutions :
  let S : Set (ℤ × ℤ) := {(3995, 3993), (1, -1), (1999, 3996005), (3996005, 1997), (1997, -3996005), (-3996005, 1995)}
  ∀ x y : ℤ, (1996 * x + 1998 * y + 1 = x * y) ↔ (x, y) ∈ S :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l454_45414


namespace NUMINAMATH_CALUDE_no_equal_product_l454_45482

theorem no_equal_product (x y : ℕ) : x * (x + 1) ≠ 4 * y * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_equal_product_l454_45482


namespace NUMINAMATH_CALUDE_expand_and_simplify_l454_45478

theorem expand_and_simplify (x y : ℝ) :
  x * (x - 3 * y) + (2 * x - y)^2 = 5 * x^2 - 7 * x * y + y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l454_45478


namespace NUMINAMATH_CALUDE_tamika_always_wins_l454_45433

def tamika_set : Finset Nat := {6, 7, 8}
def carlos_set : Finset Nat := {2, 3, 5}

theorem tamika_always_wins :
  ∀ (t1 t2 : Nat) (c1 c2 : Nat),
    t1 ∈ tamika_set → t2 ∈ tamika_set → t1 ≠ t2 →
    c1 ∈ carlos_set → c2 ∈ carlos_set → c1 ≠ c2 →
    t1 * t2 > c1 * c2 := by
  sorry

#check tamika_always_wins

end NUMINAMATH_CALUDE_tamika_always_wins_l454_45433


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l454_45403

-- Define propositions P and Q
def P (a b : ℝ) : Prop := a > b ∧ b > 0
def Q (a b : ℝ) : Prop := a^2 > b^2

-- Theorem stating that P is sufficient but not necessary for Q
theorem P_sufficient_not_necessary_for_Q :
  (∀ a b : ℝ, P a b → Q a b) ∧
  ¬(∀ a b : ℝ, Q a b → P a b) :=
sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l454_45403


namespace NUMINAMATH_CALUDE_find_a_find_m_range_l454_45401

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + a

-- Statement 1
theorem find_a :
  (∀ x : ℝ, f 2 x < 5 ↔ -3/2 < x ∧ x < 1) →
  (∃! a : ℝ, ∀ x : ℝ, f a x < 5 ↔ -3/2 < x ∧ x < 1) :=
sorry

-- Statement 2
theorem find_m_range (m : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 5 → 2 * x^2 + x + 2 > m * x) →
  m < 5 :=
sorry

end NUMINAMATH_CALUDE_find_a_find_m_range_l454_45401


namespace NUMINAMATH_CALUDE_diesel_rates_indeterminable_l454_45473

/-- Represents the diesel purchase data for a company over 4 years -/
structure DieselPurchaseData where
  /-- The diesel rates for each of the 4 years (in dollars per gallon) -/
  rates : Fin 4 → ℝ
  /-- The amount spent on diesel each year (in dollars) -/
  annual_spend : ℝ
  /-- The mean cost of diesel over the 4-year period (in dollars per gallon) -/
  mean_cost : ℝ

/-- Theorem stating that given the conditions, the individual yearly rates cannot be uniquely determined -/
theorem diesel_rates_indeterminable (data : DieselPurchaseData) : 
  data.mean_cost = 1.52 → 
  (∀ (i j : Fin 4), i ≠ j → data.rates i ≠ data.rates j) →
  (∀ (i : Fin 4), data.annual_spend / data.rates i = data.annual_spend / data.rates 0) →
  ¬∃! (rates : Fin 4 → ℝ), rates = data.rates :=
sorry


end NUMINAMATH_CALUDE_diesel_rates_indeterminable_l454_45473


namespace NUMINAMATH_CALUDE_farm_legs_count_l454_45443

/-- The number of legs for a given animal type -/
def legs_per_animal (animal : String) : ℕ :=
  match animal with
  | "chicken" => 2
  | "sheep" => 4
  | _ => 0

/-- The total number of animals in the farm -/
def total_animals : ℕ := 20

/-- The number of sheep in the farm -/
def num_sheep : ℕ := 10

/-- The number of chickens in the farm -/
def num_chickens : ℕ := total_animals - num_sheep

theorem farm_legs_count : 
  (num_sheep * legs_per_animal "sheep") + (num_chickens * legs_per_animal "chicken") = 60 := by
  sorry

end NUMINAMATH_CALUDE_farm_legs_count_l454_45443


namespace NUMINAMATH_CALUDE_hanas_stamp_collection_value_l454_45445

theorem hanas_stamp_collection_value :
  ∀ (total_value : ℚ),
    (4 / 7 : ℚ) * total_value +  -- Amount sold at garage sale
    (1 / 3 : ℚ) * ((3 / 7 : ℚ) * total_value) = 28 →  -- Amount sold at auction
    total_value = 196 := by
  sorry

end NUMINAMATH_CALUDE_hanas_stamp_collection_value_l454_45445


namespace NUMINAMATH_CALUDE_range_of_a_l454_45444

/-- Line l: 3x + 4y + a = 0 -/
def line_l (a : ℝ) (x y : ℝ) : Prop := 3*x + 4*y + a = 0

/-- Circle C: (x-2)² + y² = 2 -/
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

/-- Point M is on line l -/
def M_on_line_l (a : ℝ) (M : ℝ × ℝ) : Prop := line_l a M.1 M.2

/-- Tangent condition: �angle PMQ = 90° -/
def tangent_condition (M : ℝ × ℝ) : Prop := 
  ∃ (P Q : ℝ × ℝ), circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧ 
    (M.1 - P.1) * (M.1 - Q.1) + (M.2 - P.2) * (M.2 - Q.2) = 0

/-- Main theorem -/
theorem range_of_a (a : ℝ) : 
  (∃ M : ℝ × ℝ, M_on_line_l a M ∧ tangent_condition M) → 
  -16 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l454_45444


namespace NUMINAMATH_CALUDE_system_solvability_l454_45446

-- Define the system of equations
def system_of_equations (x y z a b c : ℝ) : Prop :=
  (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c

-- Define the solvability condition
def solvability_condition (a b c : ℝ) : Prop :=
  a = 1 ∨ b = 1 ∨ c = 1 ∨ a + b + c + a * b * c = 0

-- Theorem statement
theorem system_solvability (a b c : ℝ) :
  (∃ x y z : ℝ, system_of_equations x y z a b c) ↔ solvability_condition a b c :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l454_45446


namespace NUMINAMATH_CALUDE_triangle_side_length_is_2_sqrt_3_l454_45495

/-- Represents a semicircle with its center and radius -/
structure Semicircle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an equilateral triangle with three touching semicircles -/
structure TriangleWithSemicircles where
  triangle : List (ℝ × ℝ)
  semicircles : List Semicircle

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (triangle : List (ℝ × ℝ)) : Prop := sorry

/-- Checks if semicircles touch each other and the triangle -/
def areTouchingSemicircles (t : TriangleWithSemicircles) : Prop := sorry

/-- Checks if the diameter of each semicircle lies along a side of the triangle -/
def semicirclesAlongSides (t : TriangleWithSemicircles) : Prop := sorry

/-- Calculates the side length of the triangle -/
noncomputable def triangleSideLength (t : TriangleWithSemicircles) : ℝ := sorry

theorem triangle_side_length_is_2_sqrt_3 (t : TriangleWithSemicircles) :
  isEquilateral t.triangle ∧
  (∀ s ∈ t.semicircles, s.radius = 1) ∧
  areTouchingSemicircles t ∧
  semicirclesAlongSides t →
  triangleSideLength t = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_is_2_sqrt_3_l454_45495


namespace NUMINAMATH_CALUDE_min_jellybeans_jellybeans_solution_l454_45434

theorem min_jellybeans (n : ℕ) : n ≥ 150 ∧ n % 15 = 14 → n ≥ 164 :=
by
  sorry

theorem jellybeans_solution : ∃ (n : ℕ), n = 164 ∧ n ≥ 150 ∧ n % 15 = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_min_jellybeans_jellybeans_solution_l454_45434


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_l454_45483

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

theorem smallest_five_digit_multiple : 
  (∀ k : ℕ, is_five_digit k ∧ 
            is_divisible_by k 2 ∧ 
            is_divisible_by k 3 ∧ 
            is_divisible_by k 5 ∧ 
            is_divisible_by k 7 ∧ 
            is_divisible_by k 11 
            → k ≥ 11550) ∧ 
  is_five_digit 11550 ∧ 
  is_divisible_by 11550 2 ∧ 
  is_divisible_by 11550 3 ∧ 
  is_divisible_by 11550 5 ∧ 
  is_divisible_by 11550 7 ∧ 
  is_divisible_by 11550 11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_l454_45483


namespace NUMINAMATH_CALUDE_emily_sixth_quiz_score_l454_45461

def emily_scores : List ℝ := [94, 97, 88, 91, 102]

theorem emily_sixth_quiz_score :
  let n : ℕ := emily_scores.length
  let sum : ℝ := emily_scores.sum
  let target_mean : ℝ := 95
  let target_sum : ℝ := target_mean * (n + 1)
  let sixth_score : ℝ := target_sum - sum
  sixth_score = 98 ∧ (sum + sixth_score) / (n + 1) = target_mean :=
by sorry

end NUMINAMATH_CALUDE_emily_sixth_quiz_score_l454_45461
