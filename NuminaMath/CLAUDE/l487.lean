import Mathlib

namespace NUMINAMATH_CALUDE_sarah_earnings_l487_48770

/-- Sarah's earnings for the week given her work hours and pay rates -/
theorem sarah_earnings : 
  let weekday_hours := 1.75 + 65/60 + 2.75 + 45/60
  let weekend_hours := 2
  let weekday_rate := 4
  let weekend_rate := 6
  (weekday_hours * weekday_rate + weekend_hours * weekend_rate : ℝ) = 37.33 := by
  sorry

end NUMINAMATH_CALUDE_sarah_earnings_l487_48770


namespace NUMINAMATH_CALUDE_special_collection_books_l487_48756

/-- The number of books in a special collection at the beginning of a month,
    given the number of books loaned, returned, and remaining at the end. -/
theorem special_collection_books
  (loaned : ℕ)
  (return_rate : ℚ)
  (end_count : ℕ)
  (h1 : loaned = 40)
  (h2 : return_rate = 7/10)
  (h3 : end_count = 63) :
  loaned * (1 - return_rate) + end_count = 47 :=
sorry

end NUMINAMATH_CALUDE_special_collection_books_l487_48756


namespace NUMINAMATH_CALUDE_bank_savings_exceed_two_dollars_l487_48711

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem bank_savings_exceed_two_dollars :
  let a : ℚ := 1/100  -- 1 cent in dollars
  let r : ℚ := 2      -- doubling each day
  (geometric_sum a r 8 > 2) ∧ (geometric_sum a r 7 ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_bank_savings_exceed_two_dollars_l487_48711


namespace NUMINAMATH_CALUDE_subset_implies_a_leq_4_l487_48767

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + 4 ≥ 0}

-- State the theorem
theorem subset_implies_a_leq_4 : ∀ a : ℝ, A ⊆ B a → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_leq_4_l487_48767


namespace NUMINAMATH_CALUDE_a_exp_a_inequality_l487_48734

theorem a_exp_a_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  a < Real.exp a - 1 ∧ Real.exp a - 1 < a ^ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_a_exp_a_inequality_l487_48734


namespace NUMINAMATH_CALUDE_fraction_meaningful_l487_48742

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 1)) ↔ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l487_48742


namespace NUMINAMATH_CALUDE_exists_increasing_sequence_with_gcd_property_l487_48722

theorem exists_increasing_sequence_with_gcd_property :
  ∃ (a : ℕ → ℕ), 
    (∀ n : ℕ, a n < a (n + 1)) ∧ 
    (∀ i j : ℕ, i ≠ j → Nat.gcd (i * a j) (j * a i) = Nat.gcd i j) := by
  sorry

end NUMINAMATH_CALUDE_exists_increasing_sequence_with_gcd_property_l487_48722


namespace NUMINAMATH_CALUDE_right_triangle_congruence_l487_48713

-- Define a right-angled triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define congruence for right-angled triangles
def congruent (t1 t2 : RightTriangle) : Prop :=
  t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2 ∧ t1.hypotenuse = t2.hypotenuse

-- Theorem: Two right-angled triangles with two equal legs are congruent
theorem right_triangle_congruence (t1 t2 : RightTriangle) 
  (h : t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2) : congruent t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_congruence_l487_48713


namespace NUMINAMATH_CALUDE_anne_heavier_than_douglas_l487_48775

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := 52

/-- The difference in weight between Anne and Douglas -/
def weight_difference : ℕ := anne_weight - douglas_weight

theorem anne_heavier_than_douglas : weight_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_anne_heavier_than_douglas_l487_48775


namespace NUMINAMATH_CALUDE_remainder_theorem_l487_48735

-- Define the polynomial and its divisor
def p (x : ℝ) : ℝ := 3*x^7 + 2*x^5 - 5*x^3 + x^2 - 9
def d (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the remainder
def r (x : ℝ) : ℝ := 14*x - 16

-- Theorem statement
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = d x * q x + r x :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l487_48735


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l487_48727

theorem max_value_sum_of_roots (x y z : ℝ) 
  (sum_eq_two : x + y + z = 2)
  (x_geq_neg_one : x ≥ -1)
  (y_geq_neg_two : y ≥ -2)
  (z_geq_neg_one : z ≥ -1) :
  ∃ (M : ℝ), M = 4 * Real.sqrt 3 ∧ 
  ∀ (a b c : ℝ), a + b + c = 2 → a ≥ -1 → b ≥ -2 → c ≥ -1 →
  Real.sqrt (3 * a^2 + 3) + Real.sqrt (3 * b^2 + 6) + Real.sqrt (3 * c^2 + 3) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l487_48727


namespace NUMINAMATH_CALUDE_alfred_storage_period_l487_48763

/-- Calculates the number of years Alfred stores maize -/
def years_storing_maize (
  monthly_storage : ℕ             -- tonnes stored per month
  ) (stolen : ℕ)                  -- tonnes stolen
  (donated : ℕ)                   -- tonnes donated
  (final_amount : ℕ)              -- final amount of maize in tonnes
  : ℕ :=
  (final_amount + stolen - donated) / (monthly_storage * 12)

/-- Theorem stating that Alfred stores maize for 2 years -/
theorem alfred_storage_period :
  years_storing_maize 1 5 8 27 = 2 := by
  sorry

end NUMINAMATH_CALUDE_alfred_storage_period_l487_48763


namespace NUMINAMATH_CALUDE_median_length_inequality_l487_48714

theorem median_length_inequality (a b c s_a : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Triangle sides are positive
  a + b > c ∧ b + c > a ∧ a + c > b ∧  -- Triangle inequality
  s_a > 0 ∧  -- Median length is positive
  s_a^2 = (b^2 + c^2) / 4 - a^2 / 16  -- Median length formula
  →
  s_a < (b + c) / 2 := by
sorry

end NUMINAMATH_CALUDE_median_length_inequality_l487_48714


namespace NUMINAMATH_CALUDE_valid_pairs_l487_48743

def is_valid_pair (A B : Nat) : Prop :=
  A ≠ B ∧
  A ≥ 10 ∧ A ≤ 99 ∧
  B ≥ 10 ∧ B ≤ 99 ∧
  A % 10 = B % 10 ∧
  A / 9 = B % 9 ∧
  B / 9 = A % 9

theorem valid_pairs : 
  (∀ A B : Nat, is_valid_pair A B → 
    ((A = 85 ∧ B = 75) ∨ (A = 25 ∧ B = 65) ∨ (A = 15 ∧ B = 55))) ∧
  (is_valid_pair 85 75 ∧ is_valid_pair 25 65 ∧ is_valid_pair 15 55) := by
  sorry

end NUMINAMATH_CALUDE_valid_pairs_l487_48743


namespace NUMINAMATH_CALUDE_blueberry_pies_l487_48768

theorem blueberry_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) :
  total_pies = 30 →
  apple_ratio = 2 →
  blueberry_ratio = 3 →
  cherry_ratio = 5 →
  blueberry_ratio * (total_pies / (apple_ratio + blueberry_ratio + cherry_ratio)) = 9 :=
by sorry

end NUMINAMATH_CALUDE_blueberry_pies_l487_48768


namespace NUMINAMATH_CALUDE_statement_A_statement_D_l487_48781

-- Statement A
theorem statement_A (a b c : ℝ) : 
  a / (c^2 + 1) > b / (c^2 + 1) → a > b :=
by sorry

-- Statement D
theorem statement_D (a b : ℝ) :
  -1 < 2*a + b ∧ 2*a + b < 1 ∧ -1 < a - b ∧ a - b < 2 →
  -3 < 4*a - b ∧ 4*a - b < 5 :=
by sorry

end NUMINAMATH_CALUDE_statement_A_statement_D_l487_48781


namespace NUMINAMATH_CALUDE_ring_toss_daily_income_l487_48772

theorem ring_toss_daily_income (total_income : ℕ) (num_days : ℕ) (daily_income : ℕ) : 
  total_income = 7560 → 
  num_days = 12 → 
  total_income = daily_income * num_days →
  daily_income = 630 := by
sorry

end NUMINAMATH_CALUDE_ring_toss_daily_income_l487_48772


namespace NUMINAMATH_CALUDE_adams_trivia_score_l487_48729

/-- Adam's trivia game score calculation -/
theorem adams_trivia_score :
  ∀ (first_half second_half points_per_question : ℕ),
    first_half = 8 →
    second_half = 2 →
    points_per_question = 8 →
    (first_half + second_half) * points_per_question = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_adams_trivia_score_l487_48729


namespace NUMINAMATH_CALUDE_quadratic_function_property_l487_48799

theorem quadratic_function_property (a m : ℝ) (h1 : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - x + a
  f m < 0 → f (m - 1) > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l487_48799


namespace NUMINAMATH_CALUDE_house_sale_profit_rate_l487_48786

/-- The profit rate calculation for a house sale with discount, price increase, and inflation -/
theorem house_sale_profit_rate 
  (list_price : ℝ) 
  (discount_rate : ℝ) 
  (price_increase_rate : ℝ) 
  (inflation_rate : ℝ) 
  (h1 : discount_rate = 0.05)
  (h2 : price_increase_rate = 0.60)
  (h3 : inflation_rate = 0.40) : 
  ∃ (profit_rate : ℝ), 
    abs (profit_rate - ((1 + price_increase_rate) / ((1 - discount_rate) * (1 + inflation_rate)) - 1)) < 0.001 ∧ 
    abs (profit_rate - 0.203) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_profit_rate_l487_48786


namespace NUMINAMATH_CALUDE_butterfat_mixture_l487_48796

theorem butterfat_mixture (x : ℝ) : 
  let initial_volume : ℝ := 8
  let initial_butterfat_percentage : ℝ := 50
  let added_volume : ℝ := 24
  let final_butterfat_percentage : ℝ := 20
  let final_volume : ℝ := initial_volume + added_volume
  let initial_butterfat : ℝ := initial_volume * (initial_butterfat_percentage / 100)
  let added_butterfat : ℝ := added_volume * (x / 100)
  let final_butterfat : ℝ := final_volume * (final_butterfat_percentage / 100)
  initial_butterfat + added_butterfat = final_butterfat → x = 10 := by
sorry

end NUMINAMATH_CALUDE_butterfat_mixture_l487_48796


namespace NUMINAMATH_CALUDE_matrix_N_property_l487_48755

theorem matrix_N_property (N : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ w : Fin 3 → ℝ, N.mulVec w = (3 : ℝ) • w) ↔
  N = ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]] :=
by sorry

end NUMINAMATH_CALUDE_matrix_N_property_l487_48755


namespace NUMINAMATH_CALUDE_subset_condition_l487_48792

theorem subset_condition (a : ℝ) : 
  let A := {x : ℝ | |x - (a+1)^2/2| ≤ (a-1)^2/2}
  let B := {x : ℝ | x^2 - 3*(a+1)*x + 2*(3*a+1) ≤ 0}
  (A ⊆ B) ↔ (1 ≤ a ∧ a ≤ 3) ∨ a = -1 := by sorry

end NUMINAMATH_CALUDE_subset_condition_l487_48792


namespace NUMINAMATH_CALUDE_fixed_points_of_specific_quadratic_min_value_of_ratio_sum_min_value_achieved_l487_48705

-- Define the quadratic function
def quadratic (m n t : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + t

-- Define what it means to be a fixed point
def is_fixed_point (m n t : ℝ) (x : ℝ) : Prop :=
  quadratic m n t x = x

-- Part 1: Fixed points of y = x^2 - x - 3
theorem fixed_points_of_specific_quadratic :
  {x : ℝ | is_fixed_point 1 (-1) (-3) x} = {-1, 3} := by sorry

-- Part 2: Minimum value of x1/x2 + x2/x1
theorem min_value_of_ratio_sum :
  ∀ a x₁ x₂ : ℝ,
    x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
    is_fixed_point 2 (-(2+a)) (a-1) x₁ →
    is_fixed_point 2 (-(2+a)) (a-1) x₂ →
    (x₁ / x₂ + x₂ / x₁) ≥ 6 := by sorry

-- The minimum value is achieved when a = 5
theorem min_value_achieved :
  ∃ a x₁ x₂ : ℝ,
    x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    is_fixed_point 2 (-(2+a)) (a-1) x₁ ∧
    is_fixed_point 2 (-(2+a)) (a-1) x₂ ∧
    x₁ / x₂ + x₂ / x₁ = 6 := by sorry

end NUMINAMATH_CALUDE_fixed_points_of_specific_quadratic_min_value_of_ratio_sum_min_value_achieved_l487_48705


namespace NUMINAMATH_CALUDE_complement_of_union_in_U_l487_48715

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union_in_U :
  (U \ (M ∪ N)) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_in_U_l487_48715


namespace NUMINAMATH_CALUDE_softball_team_ratio_l487_48736

theorem softball_team_ratio (total_players : ℕ) (more_women : ℕ) : 
  total_players = 15 → more_women = 5 → 
  ∃ (men women : ℕ), 
    men + women = total_players ∧ 
    women = men + more_women ∧ 
    men * 2 = women := by
  sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l487_48736


namespace NUMINAMATH_CALUDE_no_two_digit_factors_of_1806_l487_48762

theorem no_two_digit_factors_of_1806 : 
  ¬∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 1806 :=
by sorry

end NUMINAMATH_CALUDE_no_two_digit_factors_of_1806_l487_48762


namespace NUMINAMATH_CALUDE_red_shirt_percentage_l487_48710

theorem red_shirt_percentage (total_students : ℕ) (blue_percent : ℚ) (green_percent : ℚ) (other_colors : ℕ) 
  (h1 : total_students = 900)
  (h2 : blue_percent = 44 / 100)
  (h3 : green_percent = 10 / 100)
  (h4 : other_colors = 162) :
  (total_students - (blue_percent * total_students + green_percent * total_students + other_colors)) / total_students = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_red_shirt_percentage_l487_48710


namespace NUMINAMATH_CALUDE_line_equation_proof_l487_48769

theorem line_equation_proof (m b k : ℝ) : 
  (∃! k, ∀ y₁ y₂, y₁ = k^2 + 6*k + 5 ∧ y₂ = m*k + b → |y₁ - y₂| = 7) →
  (8 = 2*m + b) →
  (b ≠ 0) →
  (m = 10 ∧ b = -12) :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l487_48769


namespace NUMINAMATH_CALUDE_heater_purchase_comparison_l487_48784

/-- Represents the total cost of purchasing heaters from a store -/
structure HeaterPurchase where
  aPrice : ℝ  -- Price of A type heater
  bPrice : ℝ  -- Price of B type heater
  aShipping : ℝ  -- Shipping cost for A type heater
  bShipping : ℝ  -- Shipping cost for B type heater

/-- Calculate the total cost for a given number of A type heaters -/
def totalCost (p : HeaterPurchase) (x : ℝ) : ℝ :=
  (p.aPrice + p.aShipping) * x + (p.bPrice + p.bShipping) * (100 - x)

/-- Store A's pricing -/
def storeA : HeaterPurchase :=
  { aPrice := 100, bPrice := 200, aShipping := 10, bShipping := 10 }

/-- Store B's pricing -/
def storeB : HeaterPurchase :=
  { aPrice := 120, bPrice := 190, aShipping := 0, bShipping := 12 }

theorem heater_purchase_comparison :
  (∀ x, totalCost storeA x = -100 * x + 21000) ∧
  (∀ x, totalCost storeB x = -82 * x + 20200) ∧
  (totalCost storeA 60 < totalCost storeB 60) := by
  sorry

end NUMINAMATH_CALUDE_heater_purchase_comparison_l487_48784


namespace NUMINAMATH_CALUDE_income_ratio_proof_l487_48793

/-- Given two persons P1 and P2 with the following conditions:
    1. The ratio of their expenditures is 3:2
    2. Each saves 2200 at the end of the year
    3. The income of P1 is 5500
    Prove that the ratio of their incomes is 5:4 -/
theorem income_ratio_proof (income_P1 income_P2 expenditure_P1 expenditure_P2 : ℕ) : 
  income_P1 = 5500 →
  expenditure_P1 = income_P1 - 2200 →
  expenditure_P2 = income_P2 - 2200 →
  3 * expenditure_P2 = 2 * expenditure_P1 →
  5 * income_P2 = 4 * income_P1 := by
  sorry

#check income_ratio_proof

end NUMINAMATH_CALUDE_income_ratio_proof_l487_48793


namespace NUMINAMATH_CALUDE_quadratic_properties_is_vertex_l487_48797

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 4*x - 1

-- Theorem stating the properties of the quadratic function
theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f ((x + y) / 2) < (f x + f y) / 2) ∧ 
  (f 2 = -5) ∧ 
  (∀ x : ℝ, f x ≥ -5) := by
  sorry

-- Define the vertex of the quadratic function
def vertex : ℝ × ℝ := (2, -5)

-- Theorem stating that the defined point is indeed the vertex
theorem is_vertex : 
  ∀ x : ℝ, f x ≥ f vertex.1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_is_vertex_l487_48797


namespace NUMINAMATH_CALUDE_every_tomcat_has_thinner_queen_l487_48741

/-- Represents a cat in the exhibition -/
inductive Cat
| Tomcat : Cat
| Queen : Cat

/-- The total number of cats in the row -/
def total_cats : Nat := 29

/-- The number of tomcats in the row -/
def num_tomcats : Nat := 10

/-- The number of queens in the row -/
def num_queens : Nat := 19

/-- Represents the row of cats at the exhibition -/
def cat_row : Fin total_cats → Cat := sorry

/-- Predicate to check if a cat is fatter than another -/
def is_fatter (c1 c2 : Cat) : Prop := sorry

/-- Two cats are adjacent if their positions differ by 1 -/
def adjacent (i j : Fin total_cats) : Prop :=
  (i.val + 1 = j.val) ∨ (j.val + 1 = i.val)

/-- Each queen has a fatter tomcat next to her -/
axiom queen_has_fatter_tomcat :
  ∀ (i : Fin total_cats), cat_row i = Cat.Queen →
    ∃ (j : Fin total_cats), adjacent i j ∧ cat_row j = Cat.Tomcat ∧ is_fatter (cat_row j) (cat_row i)

/-- The main theorem to be proved -/
theorem every_tomcat_has_thinner_queen :
  ∀ (i : Fin total_cats), cat_row i = Cat.Tomcat →
    ∃ (j : Fin total_cats), adjacent i j ∧ cat_row j = Cat.Queen ∧ is_fatter (cat_row i) (cat_row j) := by
  sorry

end NUMINAMATH_CALUDE_every_tomcat_has_thinner_queen_l487_48741


namespace NUMINAMATH_CALUDE_car_speed_difference_l487_48764

/-- Prove that given two cars P and R traveling 300 miles, where car R's speed is 34.05124837953327 mph
    and car P takes 2 hours less than car R, the difference in their average speeds is 10 mph. -/
theorem car_speed_difference (distance : ℝ) (speed_R : ℝ) (time_difference : ℝ) :
  distance = 300 →
  speed_R = 34.05124837953327 →
  time_difference = 2 →
  let time_R := distance / speed_R
  let time_P := time_R - time_difference
  let speed_P := distance / time_P
  speed_P - speed_R = 10 := by sorry

end NUMINAMATH_CALUDE_car_speed_difference_l487_48764


namespace NUMINAMATH_CALUDE_smallest_integers_difference_smallest_integers_difference_is_27720_l487_48765

theorem smallest_integers_difference : ℕ → Prop :=
  fun d =>
    ∃ n₁ n₂ : ℕ,
      n₁ > 1 ∧ n₂ > 1 ∧
      n₁ < n₂ ∧
      (∀ k : ℕ, 2 ≤ k → k ≤ 11 → n₁ % k = 1) ∧
      (∀ k : ℕ, 2 ≤ k → k ≤ 11 → n₂ % k = 1) ∧
      (∀ m : ℕ, m > 1 → m < n₂ → m ≠ n₁ → ∃ k : ℕ, 2 ≤ k ∧ k ≤ 11 ∧ m % k ≠ 1) ∧
      d = n₂ - n₁

theorem smallest_integers_difference_is_27720 : smallest_integers_difference 27720 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integers_difference_smallest_integers_difference_is_27720_l487_48765


namespace NUMINAMATH_CALUDE_unique_integer_sum_property_l487_48798

theorem unique_integer_sum_property : ∃! (A : ℕ), A > 0 ∧ 
  ∃ (B : ℕ), B < 1000 ∧ 1000 * A + B = A * (A + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_sum_property_l487_48798


namespace NUMINAMATH_CALUDE_altitude_length_right_triangle_l487_48731

/-- Given a right triangle where the angle bisector divides the hypotenuse into segments
    of lengths p and q, the length of the altitude to the hypotenuse (m) is:
    m = (pq(p+q)) / (p^2 + q^2) -/
theorem altitude_length_right_triangle (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let m := (p * q * (p + q)) / (p^2 + q^2)
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 = b^2 + c^2 ∧
    (b / p = c / q) ∧
    m = (b * c) / a :=
by
  sorry

end NUMINAMATH_CALUDE_altitude_length_right_triangle_l487_48731


namespace NUMINAMATH_CALUDE_stamp_sale_value_l487_48703

def total_stamps : ℕ := 75
def stamps_of_one_kind : ℕ := 40
def value_type1 : ℚ := 5 / 100
def value_type2 : ℚ := 8 / 100

theorem stamp_sale_value :
  ∃ (type1_count type2_count : ℕ),
    type1_count + type2_count = total_stamps ∧
    (type1_count = stamps_of_one_kind ∨ type2_count = stamps_of_one_kind) ∧
    type1_count * value_type1 + type2_count * value_type2 = 48 / 10 := by
  sorry

end NUMINAMATH_CALUDE_stamp_sale_value_l487_48703


namespace NUMINAMATH_CALUDE_percentage_of_men_in_company_l487_48766

theorem percentage_of_men_in_company 
  (total_employees : ℝ) 
  (men : ℝ) 
  (women : ℝ) 
  (h1 : men + women = total_employees)
  (h2 : men * 0.5 + women * 0.1666666666666669 = total_employees * 0.4)
  (h3 : men > 0)
  (h4 : women > 0)
  (h5 : total_employees > 0) : 
  men / total_employees = 0.7 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_men_in_company_l487_48766


namespace NUMINAMATH_CALUDE_cell_chain_length_is_million_l487_48738

/-- The length of a cell chain in nanometers -/
def cell_chain_length (cell_diameter : ℕ) (num_cells : ℕ) : ℕ :=
  cell_diameter * num_cells

/-- Theorem: The length of a cell chain is 10⁶ nanometers -/
theorem cell_chain_length_is_million :
  cell_chain_length 500 2000 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_cell_chain_length_is_million_l487_48738


namespace NUMINAMATH_CALUDE_printer_time_calculation_l487_48730

-- Define the printer's specifications
def pages_to_print : ℕ := 300
def print_speed : ℕ := 25
def maintenance_interval : ℕ := 50
def maintenance_time : ℕ := 1

-- Define the function to calculate total printing time
def total_print_time (pages : ℕ) (speed : ℕ) (maint_interval : ℕ) (maint_time : ℕ) : ℕ :=
  let print_time := pages / speed
  let maintenance_breaks := pages / maint_interval
  print_time + maintenance_breaks * maint_time

-- Theorem statement
theorem printer_time_calculation :
  total_print_time pages_to_print print_speed maintenance_interval maintenance_time = 18 :=
by sorry

end NUMINAMATH_CALUDE_printer_time_calculation_l487_48730


namespace NUMINAMATH_CALUDE_square_cutting_l487_48788

theorem square_cutting (a b : ℕ+) : 
  4 * a ^ 2 + 3 * b ^ 2 + 10 * a * b = 144 ↔ a = 2 ∧ b = 4 :=
by sorry

end NUMINAMATH_CALUDE_square_cutting_l487_48788


namespace NUMINAMATH_CALUDE_teacher_selection_problem_l487_48750

/-- The number of ways to select k items from n items --/
def permutation (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

/-- The number of valid selections of teachers --/
def validSelections (totalTeachers maleTeachers femaleTeachers selectCount : ℕ) : ℕ :=
  permutation totalTeachers selectCount - 
  (permutation maleTeachers selectCount + permutation femaleTeachers selectCount)

theorem teacher_selection_problem :
  validSelections 9 5 4 3 = 420 := by
  sorry

end NUMINAMATH_CALUDE_teacher_selection_problem_l487_48750


namespace NUMINAMATH_CALUDE_complement_A_in_U_equals_union_l487_48740

-- Define the universal set U
def U : Set ℝ := {x | -3 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := {x | x ∈ U ∧ x ∉ A}

-- Theorem statement
theorem complement_A_in_U_equals_union : 
  complement_A_in_U = {x | (-3 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3)} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_equals_union_l487_48740


namespace NUMINAMATH_CALUDE_suresh_work_time_l487_48749

theorem suresh_work_time (S : ℝ) (h1 : S > 0) : 
  (∃ (ashutosh_time : ℝ), 
    ashutosh_time = 35 ∧ 
    (9 / S) + (14 / ashutosh_time) = 1) → 
  S = 15 := by
sorry

end NUMINAMATH_CALUDE_suresh_work_time_l487_48749


namespace NUMINAMATH_CALUDE_distance_to_concert_l487_48718

/-- The distance to a concert given the distance driven before and after a gas stop -/
theorem distance_to_concert (distance_before_gas : ℕ) (distance_after_gas : ℕ) :
  distance_before_gas = 32 →
  distance_after_gas = 46 →
  distance_before_gas + distance_after_gas = 78 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_concert_l487_48718


namespace NUMINAMATH_CALUDE_distance_to_complex_point_l487_48789

theorem distance_to_complex_point : ∃ (z : ℂ), z = 3 / (2 - Complex.I)^2 ∧ Complex.abs z = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_complex_point_l487_48789


namespace NUMINAMATH_CALUDE_greatest_k_for_inequality_l487_48795

theorem greatest_k_for_inequality : ∃! k : ℕ, 
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 → a * b * c = 1 → 
    (1 / a + 1 / b + 1 / c + k / (a + b + c + 1) ≥ 3 + k / 4)) ∧
  (∀ k' : ℕ, k' > k → 
    ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧
      1 / a + 1 / b + 1 / c + k' / (a + b + c + 1) < 3 + k' / 4) ∧
  k = 13 :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_for_inequality_l487_48795


namespace NUMINAMATH_CALUDE_infinite_product_equals_sqrt_two_l487_48745

/-- The nth term of the sequence in the exponent -/
def a (n : ℕ) : ℚ := (2^n - 1) / (3^n)

/-- The infinite product as a function -/
noncomputable def infiniteProduct : ℝ := Real.rpow 2 (∑' n, a n)

/-- The theorem stating that the infinite product equals √2 -/
theorem infinite_product_equals_sqrt_two : infiniteProduct = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_infinite_product_equals_sqrt_two_l487_48745


namespace NUMINAMATH_CALUDE_parallel_line_plane_l487_48746

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Define the theorem
theorem parallel_line_plane 
  (m n : Line) (α : Plane)
  (distinct_lines : m ≠ n)
  (m_parallel_n : parallel m n)
  (m_parallel_α : parallel_plane m α)
  (n_not_in_α : ¬ contained_in n α) :
  parallel_plane n α :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_l487_48746


namespace NUMINAMATH_CALUDE_sequence_equality_l487_48776

theorem sequence_equality (a : Fin 100 → ℝ)
  (h1 : ∀ n : Fin 98, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l487_48776


namespace NUMINAMATH_CALUDE_only_set_C_is_right_triangle_l487_48785

-- Define a function to check if three numbers satisfy the Pythagorean theorem
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Theorem statement
theorem only_set_C_is_right_triangle :
  (¬ isPythagoreanTriple 3 4 2) ∧
  (¬ isPythagoreanTriple 5 12 15) ∧
  (isPythagoreanTriple 8 15 17) ∧
  (¬ isPythagoreanTriple 9 16 25) :=
by sorry


end NUMINAMATH_CALUDE_only_set_C_is_right_triangle_l487_48785


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l487_48733

theorem complex_number_in_first_quadrant :
  let z : ℂ := (2 + Complex.I) / 3
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l487_48733


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l487_48760

theorem geometric_sequence_first_term 
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r^5 = Nat.factorial 9)  -- 6th term is 9!
  (h2 : a * r^8 = Nat.factorial 10) -- 9th term is 10!
  : a = (Nat.factorial 9) / (10 ^ (5/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l487_48760


namespace NUMINAMATH_CALUDE_parabola_directrix_l487_48709

/-- The equation of the directrix of the parabola y^2 = 6x is x = -3/2 -/
theorem parabola_directrix (x y : ℝ) : y^2 = 6*x → (∃ (k : ℝ), k = -3/2 ∧ x = k) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l487_48709


namespace NUMINAMATH_CALUDE_f_inequality_l487_48721

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem f_inequality : f (π/3) > f 1 ∧ f 1 > f (-π/4) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l487_48721


namespace NUMINAMATH_CALUDE_product_and_sum_of_factors_l487_48779

theorem product_and_sum_of_factors : ∃ a b : ℕ, 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8670 ∧ 
  a + b = 136 := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_of_factors_l487_48779


namespace NUMINAMATH_CALUDE_product_of_integers_with_given_lcm_and_gcd_l487_48704

theorem product_of_integers_with_given_lcm_and_gcd :
  ∀ x y : ℕ+, 
  x.val > 0 ∧ y.val > 0 →
  Nat.lcm x.val y.val = 60 →
  Nat.gcd x.val y.val = 5 →
  x.val * y.val = 300 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_with_given_lcm_and_gcd_l487_48704


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l487_48716

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) + a n = 4 * n - 58

theorem arithmetic_sequence_2015th_term (a : ℕ → ℤ) 
  (h : arithmetic_sequence a) : a 2015 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l487_48716


namespace NUMINAMATH_CALUDE_odd_function_with_period_4_l487_48790

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_with_period_4 (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 4) 
  (h_min_period : ∀ p, 0 < p → p < 4 → ¬ has_period f p) : 
  f 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_with_period_4_l487_48790


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_7_with_different_digits_l487_48782

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

theorem largest_four_digit_divisible_by_7_with_different_digits :
  ∃ (n : ℕ), is_four_digit n ∧ n % 7 = 0 ∧ has_different_digits n ∧
  ∀ (m : ℕ), is_four_digit m → m % 7 = 0 → has_different_digits m → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_7_with_different_digits_l487_48782


namespace NUMINAMATH_CALUDE_union_of_sets_l487_48739

theorem union_of_sets : 
  let A : Set ℕ := {1, 3}
  let B : Set ℕ := {3, 4}
  A ∪ B = {1, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l487_48739


namespace NUMINAMATH_CALUDE_bus_stop_timing_l487_48737

theorem bus_stop_timing (distance : ℝ) (speed1 speed2 : ℝ) (T : ℝ) : 
  distance = 9.999999999999993 →
  speed1 = 5 →
  speed2 = 6 →
  distance / speed1 * 60 - distance / speed2 * 60 = 2 * T →
  T = 10 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_timing_l487_48737


namespace NUMINAMATH_CALUDE_range_of_a_l487_48747

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := (1 + a)^2 + (1 - a)^2 < 4

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 ≥ 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) → (a ∈ Set.Icc (-2) (-1) ∪ Set.Icc 1 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l487_48747


namespace NUMINAMATH_CALUDE_complex_power_six_l487_48771

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Statement: (1 + i)^6 = -8i -/
theorem complex_power_six : (1 + i)^6 = -8 * i := by sorry

end NUMINAMATH_CALUDE_complex_power_six_l487_48771


namespace NUMINAMATH_CALUDE_exp_two_monotone_increasing_l487_48732

/-- The function f(x) = 2^x is monotonically increasing for all real numbers x. -/
theorem exp_two_monotone_increasing : ∀ x y : ℝ, x < y → (2 : ℝ) ^ x < (2 : ℝ) ^ y := by
  sorry

end NUMINAMATH_CALUDE_exp_two_monotone_increasing_l487_48732


namespace NUMINAMATH_CALUDE_max_five_sunday_months_correct_five_is_max_l487_48791

/-- Represents a year, which can be either common (365 days) or leap (366 days) -/
inductive Year
| Common
| Leap

/-- Represents a month in a year -/
structure Month where
  days : Nat
  h1 : days ≥ 28
  h2 : days ≤ 31

/-- The number of Sundays in a month -/
def sundays (m : Month) : Nat :=
  if m.days ≥ 35 then 5 else 4

/-- The maximum number of months with 5 Sundays in a year -/
def max_five_sunday_months (y : Year) : Nat :=
  match y with
  | Year.Common => 4
  | Year.Leap => 5

theorem max_five_sunday_months_correct (y : Year) :
  max_five_sunday_months y = 
    match y with
    | Year.Common => 4
    | Year.Leap => 5 :=
by
  sorry

theorem five_is_max (y : Year) :
  max_five_sunday_months y ≤ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_five_sunday_months_correct_five_is_max_l487_48791


namespace NUMINAMATH_CALUDE_keith_bought_cards_l487_48773

/-- The number of baseball cards Keith bought -/
def cards_bought : ℕ := sorry

/-- Fred's initial number of baseball cards -/
def initial_cards : ℕ := 40

/-- Fred's current number of baseball cards -/
def current_cards : ℕ := 18

/-- Theorem: The number of cards Keith bought is equal to the difference
    between Fred's initial and current number of cards -/
theorem keith_bought_cards : 
  cards_bought = initial_cards - current_cards := by sorry

end NUMINAMATH_CALUDE_keith_bought_cards_l487_48773


namespace NUMINAMATH_CALUDE_new_class_mean_l487_48794

theorem new_class_mean (total_students : ℕ) (initial_students : ℕ) (later_students : ℕ)
  (initial_mean : ℚ) (later_mean : ℚ) :
  total_students = initial_students + later_students →
  initial_students = 30 →
  later_students = 6 →
  initial_mean = 72 / 100 →
  later_mean = 78 / 100 →
  (initial_students * initial_mean + later_students * later_mean) / total_students = 73 / 100 :=
by sorry

end NUMINAMATH_CALUDE_new_class_mean_l487_48794


namespace NUMINAMATH_CALUDE_ship_grain_calculation_l487_48757

/-- The amount of grain (in tons) that spilled into the water -/
def spilled_grain : ℕ := 49952

/-- The amount of grain (in tons) that remained onboard -/
def remaining_grain : ℕ := 918

/-- The original amount of grain (in tons) on the ship -/
def original_grain : ℕ := spilled_grain + remaining_grain

theorem ship_grain_calculation : original_grain = 50870 := by
  sorry

end NUMINAMATH_CALUDE_ship_grain_calculation_l487_48757


namespace NUMINAMATH_CALUDE_correct_equation_is_fourth_l487_48777

theorem correct_equation_is_fourth : 
  ∃ (a b : ℝ), 
    (2*a + 3*b ≠ 5*a*b) ∧ 
    ((3*a^3)^2 ≠ 6*a^6) ∧ 
    (a^6 / a^2 ≠ a^3) ∧ 
    (a^2 * a^3 = a^5) := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_is_fourth_l487_48777


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l487_48706

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability of a normal random variable being less than a given value -/
noncomputable def prob_less_than (X : NormalRV) (x : ℝ) : ℝ := sorry

theorem normal_distribution_symmetry 
  (X : NormalRV) 
  (h : X.μ = 2) 
  (h2 : prob_less_than X 4 = 0.8) : 
  prob_less_than X 0 = 0.2 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l487_48706


namespace NUMINAMATH_CALUDE_locus_of_point_P_l487_48753

/-- The locus of point P given a line and specific conditions --/
theorem locus_of_point_P (x y m n : ℝ) :
  (m / 4 + n / 3 = 1) → -- M(m, n) is on the line l
  (x - m = -2 * x) →    -- Condition from AP = 2PB
  (y = 2 * n - 2 * y) → -- Condition from AP = 2PB
  (3 * x / 4 + y / 2 = 1) := by
sorry


end NUMINAMATH_CALUDE_locus_of_point_P_l487_48753


namespace NUMINAMATH_CALUDE_visited_none_count_l487_48701

/-- Represents the number of people who have visited a country or combination of countries. -/
structure VisitCount where
  total : Nat
  iceland : Nat
  norway : Nat
  sweden : Nat
  all_three : Nat
  iceland_norway : Nat
  iceland_sweden : Nat
  norway_sweden : Nat

/-- Calculates the number of people who have visited neither Iceland, Norway, nor Sweden. -/
def people_visited_none (vc : VisitCount) : Nat :=
  vc.total - (vc.iceland + vc.norway + vc.sweden - vc.iceland_norway - vc.iceland_sweden - vc.norway_sweden + vc.all_three)

/-- Theorem stating that given the conditions, 42 people have visited neither country. -/
theorem visited_none_count (vc : VisitCount) 
  (h_total : vc.total = 100)
  (h_iceland : vc.iceland = 45)
  (h_norway : vc.norway = 37)
  (h_sweden : vc.sweden = 21)
  (h_all_three : vc.all_three = 12)
  (h_iceland_norway : vc.iceland_norway = 20)
  (h_iceland_sweden : vc.iceland_sweden = 15)
  (h_norway_sweden : vc.norway_sweden = 10) :
  people_visited_none vc = 42 := by
  sorry

end NUMINAMATH_CALUDE_visited_none_count_l487_48701


namespace NUMINAMATH_CALUDE_root_values_l487_48752

theorem root_values (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * k^3 + b * k^2 + c * k + d = 0)
  (h2 : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = I ∨ k = -I :=
sorry

end NUMINAMATH_CALUDE_root_values_l487_48752


namespace NUMINAMATH_CALUDE_specific_tree_height_l487_48700

/-- Represents the height of a tree after a given number of years -/
def tree_height (initial_height : ℝ) (yearly_growth : ℝ) (years : ℝ) : ℝ :=
  initial_height + yearly_growth * years

/-- Theorem stating the height of a specific tree after n years -/
theorem specific_tree_height (n : ℝ) :
  tree_height 1.8 0.3 n = 0.3 * n + 1.8 := by
  sorry

end NUMINAMATH_CALUDE_specific_tree_height_l487_48700


namespace NUMINAMATH_CALUDE_function_composition_l487_48787

theorem function_composition (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2) :
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l487_48787


namespace NUMINAMATH_CALUDE_prime_divides_repunit_iff_l487_48759

/-- A number of the form 111...1 (consisting entirely of the digit '1') -/
def repunit (n : ℕ) : ℕ := (10^n - 1) / 9

/-- Theorem stating that a prime number p is a divisor of some repunit if and only if p ≠ 2 and p ≠ 5 -/
theorem prime_divides_repunit_iff (p : ℕ) (hp : Prime p) :
  (∃ n : ℕ, p ∣ repunit n) ↔ p ≠ 2 ∧ p ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_repunit_iff_l487_48759


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l487_48725

theorem greatest_integer_inequality :
  ∀ x : ℤ, (3 * x + 2 < 7 - 2 * x) → x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l487_48725


namespace NUMINAMATH_CALUDE_boris_candy_distribution_l487_48726

/-- Given the initial conditions of Boris's candy distribution, 
    prove that the final number of pieces in each bowl is 83. -/
theorem boris_candy_distribution (initial_candy : ℕ) 
  (daughter_eats : ℕ) (set_aside : ℕ) (num_bowls : ℕ) (take_away : ℕ) :
  initial_candy = 300 →
  daughter_eats = 25 →
  set_aside = 10 →
  num_bowls = 6 →
  take_away = 5 →
  let remaining := initial_candy - daughter_eats - set_aside
  let per_bowl := remaining / num_bowls
  let doubled := per_bowl * 2
  doubled - take_away = 83 := by
  sorry

end NUMINAMATH_CALUDE_boris_candy_distribution_l487_48726


namespace NUMINAMATH_CALUDE_base_8_to_10_reversal_exists_l487_48748

theorem base_8_to_10_reversal_exists : ∃ (a b c : Nat), 
  a < 8 ∧ b < 8 ∧ c < 8 ∧
  (512 * a + 64 * b + 8 * c + 6 : Nat) = 
  (1000 * 6 + 100 * c + 10 * b + a : Nat) :=
sorry

end NUMINAMATH_CALUDE_base_8_to_10_reversal_exists_l487_48748


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l487_48744

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Converts a Time to minutes since midnight -/
def timeToMinutes (t : Time) : Nat :=
  t.hours * 60 + t.minutes

/-- Converts minutes since midnight to Time -/
def minutesToTime (m : Nat) : Time :=
  { hours := m / 60, minutes := m % 60 }

theorem sunset_time_calculation (sunrise : Time) (daylight_length : Time) :
  let sunrise_minutes := timeToMinutes sunrise
  let daylight_minutes := timeToMinutes daylight_length
  let sunset_minutes := sunrise_minutes + daylight_minutes
  let sunset := minutesToTime sunset_minutes
  sunrise.hours = 7 ∧ sunrise.minutes = 15 ∧
  daylight_length.hours = 11 ∧ daylight_length.minutes = 36 →
  sunset.hours = 18 ∧ sunset.minutes = 51 := by
  sorry

end NUMINAMATH_CALUDE_sunset_time_calculation_l487_48744


namespace NUMINAMATH_CALUDE_inequality_proof_l487_48754

theorem inequality_proof (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (h : abs (Real.sqrt (a * b) - Real.sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l487_48754


namespace NUMINAMATH_CALUDE_parallelogram_base_l487_48717

/-- The base of a parallelogram given its area and height -/
theorem parallelogram_base (area height base : ℝ) (h1 : area = 648) (h2 : height = 18) 
    (h3 : area = base * height) : base = 36 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l487_48717


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l487_48720

def daily_incomes : List ℝ := [600, 250, 450, 400, 800]

theorem cab_driver_average_income :
  (daily_incomes.sum / daily_incomes.length : ℝ) = 500 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l487_48720


namespace NUMINAMATH_CALUDE_product_equals_expansion_l487_48761

-- Define the binomials
def binomial1 (x : ℝ) : ℝ := 4 * x + 3
def binomial2 (x : ℝ) : ℝ := 2 * x - 7

-- Define the product using the distributive property
def product (x : ℝ) : ℝ := binomial1 x * binomial2 x

-- Theorem stating that the product equals the expanded form
theorem product_equals_expansion (x : ℝ) : 
  product x = 8 * x^2 - 22 * x - 21 := by sorry

end NUMINAMATH_CALUDE_product_equals_expansion_l487_48761


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l487_48708

theorem smallest_k_for_inequality : ∃ k : ℕ, k = 8 ∧ 
  (∀ w x y z : ℝ, (w^2 + x^2 + y^2 + z^2)^3 ≤ k * (w^6 + x^6 + y^6 + z^6)) ∧
  (∀ k' : ℕ, k' < k → 
    ∃ w x y z : ℝ, (w^2 + x^2 + y^2 + z^2)^3 > k' * (w^6 + x^6 + y^6 + z^6)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l487_48708


namespace NUMINAMATH_CALUDE_convex_curves_length_area_difference_l487_48723

/-- A convex curve in a 2D plane -/
structure ConvexCurve where
  -- Add necessary fields and properties here
  length : ℝ
  area : ℝ

/-- The distance between two convex curves -/
def distance (K₁ K₂ : ConvexCurve) : ℝ := 
  sorry

theorem convex_curves_length_area_difference 
  (K₁ K₂ : ConvexCurve) (r : ℝ) (hr : distance K₁ K₂ ≤ r) :
  let L := max K₁.length K₂.length
  ∃ (L₁ L₂ S₁ S₂ : ℝ),
    L₁ = K₁.length ∧ 
    L₂ = K₂.length ∧
    S₁ = K₁.area ∧ 
    S₂ = K₂.area ∧
    |L₂ - L₁| ≤ 2 * Real.pi * r ∧
    |S₂ - S₁| ≤ L * r + Real.pi * r^2 := by
  sorry

end NUMINAMATH_CALUDE_convex_curves_length_area_difference_l487_48723


namespace NUMINAMATH_CALUDE_ribbon_length_difference_equals_side_length_specific_box_ribbon_difference_l487_48758

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the ribbon length for the first method -/
def ribbonLength1 (box : BoxDimensions) (bowLength : ℝ) : ℝ :=
  2 * box.length + 2 * box.width + 4 * box.height + bowLength

/-- Calculates the ribbon length for the second method -/
def ribbonLength2 (box : BoxDimensions) (bowLength : ℝ) : ℝ :=
  2 * box.length + 4 * box.width + 2 * box.height + bowLength

/-- The main theorem to prove -/
theorem ribbon_length_difference_equals_side_length 
  (box : BoxDimensions) (bowLength : ℝ) : 
  ribbonLength2 box bowLength - ribbonLength1 box bowLength = box.length :=
by
  sorry

/-- The specific case with given dimensions -/
theorem specific_box_ribbon_difference :
  let box : BoxDimensions := ⟨22, 22, 11⟩
  let bowLength : ℝ := 24
  ribbonLength2 box bowLength - ribbonLength1 box bowLength = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_difference_equals_side_length_specific_box_ribbon_difference_l487_48758


namespace NUMINAMATH_CALUDE_max_value_h_two_roots_range_max_positive_integer_a_l487_48702

noncomputable section

open Real

-- Define the functions
def f (x : ℝ) := exp x
def g (a b : ℝ) (x : ℝ) := (a / 2) * x + b
def h (a b : ℝ) (x : ℝ) := f x * g a b x

-- Statement 1
theorem max_value_h (a b : ℝ) :
  a = -4 → b = 1 - a / 2 →
  ∃ (M : ℝ), M = 2 * exp (1 / 2) ∧ ∀ x ∈ Set.Icc 0 1, h a b x ≤ M :=
sorry

-- Statement 2
theorem two_roots_range (b : ℝ) :
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ f x₁ = g 4 b x₁ ∧ f x₂ = g 4 b x₂) ↔
  b ∈ Set.Ioo (2 - 2 * log 2) 1 :=
sorry

-- Statement 3
theorem max_positive_integer_a :
  ∃ (a : ℕ), a = 14 ∧ ∀ x : ℝ, f x > g a (-15/2) x ∧
  ∀ n : ℕ, n > a → ∃ y : ℝ, f y ≤ g n (-15/2) y :=
sorry

end

end NUMINAMATH_CALUDE_max_value_h_two_roots_range_max_positive_integer_a_l487_48702


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l487_48780

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + a 2 = 7)
  (h_diff : a 1 - a 3 = -6) :
  a 5 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l487_48780


namespace NUMINAMATH_CALUDE_exists_positive_a_leq_inverse_l487_48751

theorem exists_positive_a_leq_inverse : ∃ a : ℝ, a > 0 ∧ a ≤ 1 / a := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_a_leq_inverse_l487_48751


namespace NUMINAMATH_CALUDE_compound_prop_evaluation_l487_48778

-- Define the propositions
variable (p q : Prop)

-- Define the truth values of p and q
axiom p_true : p
axiom q_false : ¬q

-- Define the compound propositions
def prop1 := p ∨ q
def prop2 := p ∧ q
def prop3 := ¬p ∧ q
def prop4 := ¬p ∨ ¬q

-- State the theorem
theorem compound_prop_evaluation :
  prop1 p q ∧ prop4 p q ∧ ¬(prop2 p q) ∧ ¬(prop3 p q) :=
sorry

end NUMINAMATH_CALUDE_compound_prop_evaluation_l487_48778


namespace NUMINAMATH_CALUDE_radical_product_simplification_l487_48728

theorem radical_product_simplification (m : ℝ) (h : m > 0) :
  Real.sqrt (50 * m) * Real.sqrt (5 * m) * Real.sqrt (45 * m) = 15 * m * Real.sqrt (10 * m) :=
by sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l487_48728


namespace NUMINAMATH_CALUDE_megan_folders_l487_48707

def number_of_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : ℕ :=
  (initial_files - deleted_files) / files_per_folder

theorem megan_folders :
  number_of_folders 93 21 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_megan_folders_l487_48707


namespace NUMINAMATH_CALUDE_cost_per_metre_l487_48712

/-- Given that John bought 9.25 m of cloth for $416.25, prove that the cost price per metre is $45. -/
theorem cost_per_metre (total_length : ℝ) (total_cost : ℝ) 
  (h1 : total_length = 9.25)
  (h2 : total_cost = 416.25) :
  total_cost / total_length = 45 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_metre_l487_48712


namespace NUMINAMATH_CALUDE_unique_half_rectangle_l487_48774

/-- Given a rectangle R with dimensions a and b (a < b), prove that there exists exactly one rectangle
    with dimensions x and y such that x < b, y < b, its perimeter is half of R's, and its area is half of R's. -/
theorem unique_half_rectangle (a b : ℝ) (hab : a < b) :
  ∃! (x y : ℝ), x < b ∧ y < b ∧ 2 * (x + y) = a + b ∧ x * y = a * b / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_half_rectangle_l487_48774


namespace NUMINAMATH_CALUDE_wrong_observation_value_l487_48719

theorem wrong_observation_value (n : ℕ) (initial_mean correct_value new_mean : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : correct_value = 60)
  (h4 : new_mean = 36.5) :
  ∃ wrong_value : ℝ,
    n * initial_mean - wrong_value + correct_value = n * new_mean ∧
    wrong_value = 35 := by
  sorry

end NUMINAMATH_CALUDE_wrong_observation_value_l487_48719


namespace NUMINAMATH_CALUDE_no_integer_solutions_l487_48724

theorem no_integer_solutions : 
  ¬ ∃ (m n : ℤ), m^3 + n^4 + 130*m*n = 35^3 ∧ m*n ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l487_48724


namespace NUMINAMATH_CALUDE_fred_dime_count_l487_48783

def final_dime_count (initial : ℕ) (borrowed : ℕ) (returned : ℕ) (given : ℕ) : ℕ :=
  initial - borrowed + returned + given

theorem fred_dime_count : final_dime_count 12 4 2 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fred_dime_count_l487_48783
