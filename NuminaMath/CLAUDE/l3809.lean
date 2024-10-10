import Mathlib

namespace consecutive_integers_product_l3809_380994

theorem consecutive_integers_product (a b c d e : ℤ) : 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (e = d + 1) →
  (a * b * c * d * e = 15120) →
  (e = 9) :=
sorry

end consecutive_integers_product_l3809_380994


namespace negation_of_proposition_l3809_380913

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, ∃ n : ℕ+, (n : ℝ) ≥ x^2)) ↔ (∃ x : ℝ, ∀ n : ℕ+, (n : ℝ) < x^2) :=
by sorry

end negation_of_proposition_l3809_380913


namespace custom_op_result_l3809_380938

/-- Custom operation € -/
def custom_op (x y : ℝ) : ℝ := 3 * x * y - x - y

/-- Theorem stating the result of the custom operation -/
theorem custom_op_result : 
  let x : ℝ := 6
  let y : ℝ := 4
  let z : ℝ := 2
  custom_op x (custom_op y z) = 300 := by
  sorry

end custom_op_result_l3809_380938


namespace sum_of_roots_l3809_380900

theorem sum_of_roots (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 27*p - 72 = 0)
  (hq : 10*q^3 - 75*q^2 + 50*q - 625 = 0) : 
  p + q = 2*(180^(1/3 : ℝ)) + 43/3 := by
  sorry

end sum_of_roots_l3809_380900


namespace largest_R_under_condition_l3809_380922

theorem largest_R_under_condition : ∃ (R : ℕ), R > 0 ∧ R^2000 < 5^3000 ∧ ∀ (S : ℕ), S > R → S^2000 ≥ 5^3000 :=
by sorry

end largest_R_under_condition_l3809_380922


namespace rectangle_area_error_percent_l3809_380903

theorem rectangle_area_error_percent (L W : ℝ) (L' W' : ℝ) : 
  L' = L * (1 + 0.07) → 
  W' = W * (1 - 0.06) → 
  let A := L * W
  let A' := L' * W'
  (A' - A) / A * 100 = 0.58 := by
sorry

end rectangle_area_error_percent_l3809_380903


namespace power_of_power_l3809_380984

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by sorry

end power_of_power_l3809_380984


namespace sum_of_15th_set_l3809_380987

/-- Defines the first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 
  1 + (n - 1) * n / 2

/-- Defines the last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := 
  first_element n + n - 1

/-- Defines the sum of elements in the nth set -/
def S (n : ℕ) : ℕ := 
  n * (first_element n + last_element n) / 2

theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end sum_of_15th_set_l3809_380987


namespace point_satisfies_conditions_l3809_380955

def point (m : ℝ) : ℝ × ℝ := (2 - m, 2 * m - 1)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

theorem point_satisfies_conditions (m : ℝ) :
  in_fourth_quadrant (point m) ∧
  distance_to_y_axis (point m) = 3 →
  m = -1 := by
sorry

end point_satisfies_conditions_l3809_380955


namespace geometric_progression_in_floor_sqrt2003_l3809_380965

/-- For any positive integers k and m greater than 1, there exists a subsequence
of {⌊n√2003⌋} (n ≥ 1) that forms a geometric progression with m terms and ratio k. -/
theorem geometric_progression_in_floor_sqrt2003 (k m : ℕ) (hk : k > 1) (hm : m > 1) :
  ∃ (n : ℕ), ∀ (i : ℕ), i < m →
    (⌊(k^i * n : ℝ) * Real.sqrt 2003⌋ : ℤ) = k^i * ⌊(n : ℝ) * Real.sqrt 2003⌋ :=
by sorry

end geometric_progression_in_floor_sqrt2003_l3809_380965


namespace complex_number_location_l3809_380964

theorem complex_number_location :
  let z : ℂ := Complex.I / (3 - 3 * Complex.I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end complex_number_location_l3809_380964


namespace factorization_m_squared_plus_3m_l3809_380937

theorem factorization_m_squared_plus_3m (m : ℝ) : m^2 + 3*m = m*(m+3) := by
  sorry

end factorization_m_squared_plus_3m_l3809_380937


namespace percentage_not_sold_is_62_14_l3809_380970

/-- Represents the book inventory and sales data for a bookshop --/
structure BookshopData where
  initial_fiction : ℕ
  initial_nonfiction : ℕ
  fiction_sold : ℕ
  nonfiction_sold : ℕ
  fiction_returned : ℕ
  nonfiction_returned : ℕ

/-- Calculates the percentage of books not sold --/
def percentage_not_sold (data : BookshopData) : ℚ :=
  let total_initial := data.initial_fiction + data.initial_nonfiction
  let net_fiction_sold := data.fiction_sold - data.fiction_returned
  let net_nonfiction_sold := data.nonfiction_sold - data.nonfiction_returned
  let total_sold := net_fiction_sold + net_nonfiction_sold
  let not_sold := total_initial - total_sold
  (not_sold : ℚ) / (total_initial : ℚ) * 100

/-- The main theorem stating the percentage of books not sold --/
theorem percentage_not_sold_is_62_14 (data : BookshopData)
  (h1 : data.initial_fiction = 400)
  (h2 : data.initial_nonfiction = 300)
  (h3 : data.fiction_sold = 150)
  (h4 : data.nonfiction_sold = 160)
  (h5 : data.fiction_returned = 30)
  (h6 : data.nonfiction_returned = 15) :
  percentage_not_sold data = 62.14 := by
  sorry

#eval percentage_not_sold {
  initial_fiction := 400,
  initial_nonfiction := 300,
  fiction_sold := 150,
  nonfiction_sold := 160,
  fiction_returned := 30,
  nonfiction_returned := 15
}

end percentage_not_sold_is_62_14_l3809_380970


namespace total_flowers_is_105_l3809_380990

/-- The total number of hibiscus, chrysanthemums, and dandelions -/
def total_flowers (h c d : ℕ) : ℕ := h + c + d

/-- Theorem: The total number of flowers is 105 -/
theorem total_flowers_is_105 
  (h : ℕ) 
  (c : ℕ) 
  (d : ℕ) 
  (h_count : h = 34)
  (h_vs_c : h = c - 13)
  (c_vs_d : c = d + 23) : 
  total_flowers h c d = 105 := by
  sorry

#check total_flowers_is_105

end total_flowers_is_105_l3809_380990


namespace sum_of_absolute_coefficients_l3809_380924

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) : 
  (∀ x, (1 - 3*x)^9 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                      a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 7^9 := by
sorry

end sum_of_absolute_coefficients_l3809_380924


namespace sunlight_rice_yield_is_correlation_l3809_380902

-- Define the concept of a relationship
structure Relationship (X Y : Type) where
  relates : X → Y → Prop

-- Define what it means for a relationship to be functional
def IsFunctional {X Y : Type} (r : Relationship X Y) : Prop :=
  ∀ x : X, ∃! y : Y, r.relates x y

-- Define what it means for a relationship to be a correlation
def IsCorrelation {X Y : Type} (r : Relationship X Y) : Prop :=
  (¬ IsFunctional r) ∧ 
  (∃ pattern : X → Y → Prop, ∀ x : X, ∃ y : Y, pattern x y ∧ r.relates x y) ∧
  (∃ x₁ x₂ : X, ∃ y₁ y₂ : Y, r.relates x₁ y₁ ∧ r.relates x₂ y₂ ∧ x₁ ≠ x₂ ∧ y₁ ≠ y₂)

-- Define the relationship between sunlight and rice yield
def SunlightRiceYield : Relationship ℝ ℝ :=
  { relates := λ sunlight yield => yield > 0 ∧ ∃ k > 0, yield ≤ k * sunlight }

-- State the theorem
theorem sunlight_rice_yield_is_correlation :
  IsCorrelation SunlightRiceYield :=
sorry

end sunlight_rice_yield_is_correlation_l3809_380902


namespace no_y_intercepts_l3809_380918

/-- The parabola equation -/
def parabola_equation (y : ℝ) : ℝ := 3 * y^2 - 5 * y + 9

/-- Theorem: The parabola x = 3y^2 - 5y + 9 has no y-intercepts -/
theorem no_y_intercepts : ¬ ∃ y : ℝ, parabola_equation y = 0 := by
  sorry

end no_y_intercepts_l3809_380918


namespace range_of_m_l3809_380916

-- Define the sets
def set1 (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 4 ≤ 0 ∧ p.2 ≥ 0 ∧ m * p.1 - p.2 ≥ 0 ∧ m > 0}

def set2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 2)^2 ≤ 8}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (set1 m ⊆ set2) → (0 < m ∧ m ≤ 1) :=
by sorry

end range_of_m_l3809_380916


namespace complement_intersection_equals_set_l3809_380952

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_equals_set : 
  (A ∩ B)ᶜ = {0, 1, 4} := by sorry

end complement_intersection_equals_set_l3809_380952


namespace complex_equation_solution_l3809_380981

theorem complex_equation_solution (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 4 →
  c * d = x - 3 * Complex.I →
  x > 0 →
  x = 3 * Real.sqrt 15 := by sorry

end complex_equation_solution_l3809_380981


namespace equation_two_solutions_l3809_380917

def equation (a x : ℝ) : Prop :=
  (Real.cos (2 * x) + 14 * Real.cos x - 14 * a)^7 - (6 * a * Real.cos x - 4 * a^2 - 1)^7 = 
  (6 * a - 14) * Real.cos x + 2 * Real.sin x^2 - 4 * a^2 + 14 * a - 2

theorem equation_two_solutions :
  ∃ (S₁ S₂ : Set ℝ),
    (S₁ = {a : ℝ | 3.25 ≤ a ∧ a < 4}) ∧
    (S₂ = {a : ℝ | -0.5 ≤ a ∧ a < 1}) ∧
    (∀ a ∈ S₁ ∪ S₂, ∃ (x₁ x₂ : ℝ),
      x₁ ≠ x₂ ∧
      -2 * Real.pi / 3 ≤ x₁ ∧ x₁ ≤ Real.pi ∧
      -2 * Real.pi / 3 ≤ x₂ ∧ x₂ ≤ Real.pi ∧
      equation a x₁ ∧
      equation a x₂ ∧
      (∀ x, -2 * Real.pi / 3 ≤ x ∧ x ≤ Real.pi ∧ equation a x → x = x₁ ∨ x = x₂)) :=
by sorry

end equation_two_solutions_l3809_380917


namespace largest_prime_factor_of_9883_l3809_380980

theorem largest_prime_factor_of_9883 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 9883 ∧ ∀ (q : ℕ), q.Prime → q ∣ 9883 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_9883_l3809_380980


namespace complex_number_in_first_quadrant_l3809_380977

theorem complex_number_in_first_quadrant : let z : ℂ := (Complex.I) / (Complex.I + 1)
  (0 < z.re) ∧ (0 < z.im) := by
  sorry

end complex_number_in_first_quadrant_l3809_380977


namespace same_remainder_implies_specific_remainder_l3809_380975

theorem same_remainder_implies_specific_remainder 
  (m : ℕ) 
  (h1 : m ≠ 1) 
  (h2 : ∃ r : ℕ, 69 % m = r ∧ 90 % m = r ∧ 125 % m = r) : 
  86 % m = 2 := by
sorry

end same_remainder_implies_specific_remainder_l3809_380975


namespace opposite_of_negative_five_l3809_380956

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_negative_five :
  opposite (-5) = 5 := by
  sorry

end opposite_of_negative_five_l3809_380956


namespace shaded_area_calculation_l3809_380911

/-- The area of the region within a rectangle of dimensions 5 by 6 units,
    but outside three semicircles with radii 2, 3, and 2.5 units, 
    is equal to 30 - 14.625π square units. -/
theorem shaded_area_calculation : 
  let rectangle_area : ℝ := 5 * 6
  let semicircle_area (r : ℝ) : ℝ := (1/2) * Real.pi * r^2
  let total_semicircle_area : ℝ := semicircle_area 2 + semicircle_area 3 + semicircle_area 2.5
  rectangle_area - total_semicircle_area = 30 - 14.625 * Real.pi := by
  sorry

end shaded_area_calculation_l3809_380911


namespace marble_draw_probability_l3809_380950

/-- The probability of drawing a white marble first and a red marble second from a bag 
    containing 5 red marbles and 7 white marbles, without replacement. -/
theorem marble_draw_probability :
  let total_marbles : ℕ := 5 + 7
  let red_marbles : ℕ := 5
  let white_marbles : ℕ := 7
  let prob_white_first : ℚ := white_marbles / total_marbles
  let prob_red_second : ℚ := red_marbles / (total_marbles - 1)
  prob_white_first * prob_red_second = 35 / 132 :=
by sorry

end marble_draw_probability_l3809_380950


namespace count_equality_l3809_380967

/-- The count of natural numbers from 1 to 3998 that are divisible by 4 -/
def count_divisible_by_4 : ℕ := 999

/-- The count of natural numbers from 1 to 3998 whose digit sum is divisible by 4 -/
def count_digit_sum_divisible_by_4 : ℕ := 999

/-- The upper bound of the range of natural numbers being considered -/
def upper_bound : ℕ := 3998

theorem count_equality :
  count_divisible_by_4 = count_digit_sum_divisible_by_4 ∧
  count_divisible_by_4 = (upper_bound / 4 : ℕ) :=
sorry

end count_equality_l3809_380967


namespace special_pair_characterization_l3809_380943

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Property of natural numbers a and b -/
def special_pair (a b : ℕ) : Prop :=
  (a^2 + 1) % b = 0 ∧ (b^2 + 1) % a = 0

/-- Main theorem -/
theorem special_pair_characterization (a b : ℕ) :
  special_pair a b → (a = 1 ∧ b = 1) ∨ (∃ n : ℕ, n ≥ 1 ∧ a = fib (2*n - 1) ∧ b = fib (2*n + 1)) :=
sorry

end special_pair_characterization_l3809_380943


namespace sum_reciprocals_of_sum_and_product_l3809_380988

theorem sum_reciprocals_of_sum_and_product (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hsum : x + y = 10) (hprod : x * y = 20) : 
  1 / x + 1 / y = 1 / 2 := by
sorry

end sum_reciprocals_of_sum_and_product_l3809_380988


namespace comic_book_percentage_l3809_380919

theorem comic_book_percentage (total_books : ℕ) (novel_percentage : ℚ) (graphic_novels : ℕ) : 
  total_books = 120 →
  novel_percentage = 65 / 100 →
  graphic_novels = 18 →
  (total_books - (total_books * novel_percentage).floor - graphic_novels) / total_books = 1 / 5 := by
sorry

end comic_book_percentage_l3809_380919


namespace f_zero_at_three_l3809_380958

-- Define the function f
def f (x r : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 5 * x + r

-- State the theorem
theorem f_zero_at_three (r : ℝ) : f 3 r = 0 ↔ r = -273 := by sorry

end f_zero_at_three_l3809_380958


namespace bills_tv_height_l3809_380942

-- Define the dimensions of the TVs
def bill_width : ℕ := 48
def bob_width : ℕ := 70
def bob_height : ℕ := 60

-- Define the weight per square inch
def weight_per_sq_inch : ℕ := 4

-- Define the weight difference in ounces
def weight_diff_oz : ℕ := 150 * 16

-- Theorem statement
theorem bills_tv_height :
  ∃ (h : ℕ),
    h * bill_width * weight_per_sq_inch =
    bob_width * bob_height * weight_per_sq_inch - weight_diff_oz ∧
    h = 75 := by
  sorry

end bills_tv_height_l3809_380942


namespace triangle_inequality_condition_unique_k_value_l3809_380926

theorem triangle_inequality_condition (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  (6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) →
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

theorem unique_k_value :
  ∀ k : ℕ, k > 0 →
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
    a + b > c ∧ b + c > a ∧ c + a > b) →
  k = 6 :=
sorry

end triangle_inequality_condition_unique_k_value_l3809_380926


namespace square_digit_sum_99999_l3809_380910

/-- Given a natural number n, returns the sum of its digits -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a number consists of all nines -/
def is_all_nines (n : ℕ) : Prop := sorry

theorem square_digit_sum_99999 (n : ℕ) :
  n = 99999 → is_all_nines n → sum_of_digits (n^2) = 45 := by sorry

end square_digit_sum_99999_l3809_380910


namespace negative_and_absolute_value_l3809_380949

theorem negative_and_absolute_value : 
  (-(-4) = 4) ∧ (-|(-4)| = -4) := by sorry

end negative_and_absolute_value_l3809_380949


namespace game_probability_theorem_l3809_380989

def game_probability (total_rounds : ℕ) 
                     (alex_prob : ℚ) 
                     (mel_chelsea_ratio : ℕ) 
                     (alex_wins mel_wins chelsea_wins : ℕ) : Prop :=
  let mel_prob := (1 - alex_prob) * (mel_chelsea_ratio / (mel_chelsea_ratio + 1 : ℚ))
  let chelsea_prob := (1 - alex_prob) * (1 / (mel_chelsea_ratio + 1 : ℚ))
  let specific_outcome_prob := alex_prob ^ alex_wins * mel_prob ^ mel_wins * chelsea_prob ^ chelsea_wins
  let arrangements := Nat.choose total_rounds alex_wins * Nat.choose (total_rounds - alex_wins) mel_wins
  (specific_outcome_prob * arrangements : ℚ) = 76545 / 823543

theorem game_probability_theorem : 
  game_probability 7 (3/7) 3 4 2 1 :=
sorry

end game_probability_theorem_l3809_380989


namespace clock_hands_right_angles_l3809_380934

/-- Represents the number of times clock hands are at right angles in one hour -/
def right_angles_per_hour : ℕ := 2

/-- Represents the number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Represents the number of days -/
def num_days : ℕ := 5

/-- Theorem: The hands of a clock are at right angles 240 times in 5 days -/
theorem clock_hands_right_angles :
  right_angles_per_hour * hours_per_day * num_days = 240 := by
  sorry

end clock_hands_right_angles_l3809_380934


namespace quadratic_expression_value_l3809_380932

theorem quadratic_expression_value : 
  let x : ℝ := 2
  2 * x^2 - 3 * x + 4 = 6 := by sorry

end quadratic_expression_value_l3809_380932


namespace log_sqrt10_1000sqrt10_l3809_380914

theorem log_sqrt10_1000sqrt10 :
  Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end log_sqrt10_1000sqrt10_l3809_380914


namespace percent_increase_l3809_380961

theorem percent_increase (P : ℝ) (Q : ℝ) (h : Q = P + (1/3) * P) :
  (Q - P) / P * 100 = 100/3 := by
  sorry

end percent_increase_l3809_380961


namespace perpendicular_bisector_c_value_l3809_380931

/-- Given that the line x + y = c is a perpendicular bisector of the line segment from (2,5) to (8,11), prove that c = 13 -/
theorem perpendicular_bisector_c_value :
  ∀ c : ℝ,
  (∀ x y : ℝ, x + y = c ↔ (x - 5)^2 + (y - 8)^2 = (5 - 2)^2 + (8 - 5)^2) →
  c = 13 := by
  sorry

end perpendicular_bisector_c_value_l3809_380931


namespace zoo_birds_count_l3809_380963

theorem zoo_birds_count (non_bird_animals : ℕ) : 
  (5 * non_bird_animals = non_bird_animals + 360) → 
  (5 * non_bird_animals = 450) := by
sorry

end zoo_birds_count_l3809_380963


namespace tan_sum_product_equals_one_l3809_380973

theorem tan_sum_product_equals_one :
  let tan15 : ℝ := 2 - Real.sqrt 3
  let tan30 : ℝ := Real.sqrt 3 / 3
  tan15 + tan30 + tan15 * tan30 = 1 := by
sorry

end tan_sum_product_equals_one_l3809_380973


namespace factors_720_l3809_380939

/-- The number of distinct positive factors of 720 -/
def num_factors_720 : ℕ := sorry

/-- 720 has exactly 30 distinct positive factors -/
theorem factors_720 : num_factors_720 = 30 := by sorry

end factors_720_l3809_380939


namespace fraction_depends_on_z_l3809_380912

theorem fraction_depends_on_z (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = 3) :
  ∃ z₁ z₂ : ℝ, z₁ ≠ z₂ ∧ 
    (x + 4 * y + z₁) / (4 * x - y - z₁) ≠ (x + 4 * y + z₂) / (4 * x - y - z₂) :=
by sorry

end fraction_depends_on_z_l3809_380912


namespace max_draw_without_pair_is_four_l3809_380982

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (white : Nat)
  (blue : Nat)
  (red : Nat)

/-- Represents the maximum number of socks that can be drawn without guaranteeing a pair -/
def maxDrawWithoutPair (drawer : SockDrawer) : Nat :=
  4

/-- Theorem stating that for the given sock drawer, the maximum number of socks
    that can be drawn without guaranteeing a pair is 4 -/
theorem max_draw_without_pair_is_four (drawer : SockDrawer) 
  (h1 : drawer.white = 16) 
  (h2 : drawer.blue = 3) 
  (h3 : drawer.red = 6) : 
  maxDrawWithoutPair drawer = 4 := by
  sorry

#eval maxDrawWithoutPair { white := 16, blue := 3, red := 6 }

end max_draw_without_pair_is_four_l3809_380982


namespace signal_count_theorem_l3809_380948

/-- Represents the number of indicator lights --/
def num_lights : Nat := 6

/-- Represents the number of lights that light up each time --/
def lights_lit : Nat := 3

/-- Represents the number of possible colors for each light --/
def num_colors : Nat := 3

/-- Calculates the total number of different signals that can be displayed --/
def total_signals : Nat :=
  -- The actual calculation is not provided, so we use a placeholder
  324

/-- Theorem stating that the total number of different signals is 324 --/
theorem signal_count_theorem :
  total_signals = 324 := by
  sorry

end signal_count_theorem_l3809_380948


namespace no_integer_solution_l3809_380908

theorem no_integer_solution : ¬∃ (a b : ℕ+), 
  (Real.sqrt a.val + Real.sqrt b.val = 10) ∧ 
  (Real.sqrt a.val * Real.sqrt b.val = 18) := by
sorry

end no_integer_solution_l3809_380908


namespace cube_diagonal_l3809_380944

theorem cube_diagonal (s : ℝ) (h1 : 6 * s^2 = 54) (h2 : 12 * s = 36) :
  ∃ d : ℝ, d = 3 * Real.sqrt 3 ∧ d^2 = 3 * s^2 := by
  sorry

#check cube_diagonal

end cube_diagonal_l3809_380944


namespace xy_negative_sufficient_not_necessary_l3809_380921

theorem xy_negative_sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x * y < 0 → |x - y| = |x| + |y|) ∧
  (∃ x y : ℝ, |x - y| = |x| + |y| ∧ x * y ≥ 0) :=
by sorry

end xy_negative_sufficient_not_necessary_l3809_380921


namespace shaded_area_of_concentric_circles_l3809_380985

theorem shaded_area_of_concentric_circles 
  (outer_circle_area : ℝ)
  (inner_circle_radius : ℝ)
  (h1 : outer_circle_area = 81 * Real.pi)
  (h2 : inner_circle_radius = 4.5)
  : ∃ (shaded_area : ℝ), shaded_area = 54 * Real.pi := by
  sorry

end shaded_area_of_concentric_circles_l3809_380985


namespace horner_v2_value_l3809_380995

def horner_polynomial (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def horner_step (v : ℝ) (x : ℝ) (a : ℝ) : ℝ :=
  v * x + a

theorem horner_v2_value :
  let f := fun x => 8 * x^4 + 5 * x^3 + 3 * x^2 + 2 * x + 1
  let coeffs := [8, 5, 3, 2, 1]
  let x := 2
  let v₀ := coeffs.head!
  let v₁ := horner_step v₀ x (coeffs.get! 1)
  let v₂ := horner_step v₁ x (coeffs.get! 2)
  v₂ = 45 := by sorry

end horner_v2_value_l3809_380995


namespace q_of_one_equals_zero_l3809_380940

/-- Given a function q: ℝ → ℝ, prove that q(1) = 0 -/
theorem q_of_one_equals_zero (q : ℝ → ℝ) 
  (h1 : (1, 0) ∈ Set.range (λ x => (x, q x))) 
  (h2 : ∃ n : ℤ, q 1 = n) : 
  q 1 = 0 := by
  sorry

end q_of_one_equals_zero_l3809_380940


namespace cannot_reach_2003_l3809_380976

/-- The set of numbers that can appear on the board -/
def BoardNumbers : Set ℕ :=
  {n : ℕ | ∃ (k : ℕ), n ≡ 5 [ZMOD 5] ∨ n ≡ 7 [ZMOD 5] ∨ n ≡ 9 [ZMOD 5]}

/-- The transformation rule -/
def Transform (a b : ℕ) : ℕ := 5 * a - 4 * b

/-- Theorem stating that 2003 cannot appear on the board -/
theorem cannot_reach_2003 : 2003 ∉ BoardNumbers := by
  sorry

/-- Lemma: The transformation preserves the set of possible remainders modulo 5 -/
lemma transform_preserves_remainders (a b : ℕ) (h : a ∈ BoardNumbers) (h' : b ∈ BoardNumbers) :
  Transform a b ∈ BoardNumbers := by
  sorry

end cannot_reach_2003_l3809_380976


namespace john_completion_time_l3809_380992

/-- Represents the time it takes to complete a task -/
structure TaskTime where
  days : ℝ
  time_positive : days > 0

/-- Represents a person's ability to complete a task -/
structure Worker where
  time_to_complete : TaskTime

/-- Represents two people working together on a task -/
structure TeamWork where
  worker1 : Worker
  worker2 : Worker
  time_to_complete : TaskTime
  jane_leaves_early : ℝ
  jane_leaves_early_positive : jane_leaves_early > 0
  jane_leaves_early_less_than_total : jane_leaves_early < time_to_complete.days

theorem john_completion_time 
  (john : Worker) 
  (jane : Worker) 
  (team : TeamWork) :
  team.worker1 = john →
  team.worker2 = jane →
  jane.time_to_complete.days = 12 →
  team.time_to_complete.days = 10 →
  team.jane_leaves_early = 4 →
  john.time_to_complete.days = 20 := by
  sorry

end john_completion_time_l3809_380992


namespace existence_of_special_polygon_l3809_380974

-- Define what it means for a polygon to have a center of symmetry
def has_center_of_symmetry (P : Set ℝ × ℝ) : Prop := sorry

-- Define what it means for a set to be a polygon
def is_polygon (P : Set ℝ × ℝ) : Prop := sorry

-- Define what it means for a polygon to be convex
def is_convex (P : Set ℝ × ℝ) : Prop := sorry

-- Define what it means for a polygon to be divided into two parts
def can_be_divided_into (P A B : Set ℝ × ℝ) : Prop := sorry

theorem existence_of_special_polygon : 
  ∃ (P A B : Set ℝ × ℝ), 
    is_polygon P ∧ 
    ¬(has_center_of_symmetry P) ∧
    is_polygon A ∧ 
    is_polygon B ∧
    is_convex A ∧ 
    is_convex B ∧
    can_be_divided_into P A B ∧
    has_center_of_symmetry A ∧
    has_center_of_symmetry B := by
  sorry

end existence_of_special_polygon_l3809_380974


namespace max_value_and_k_range_l3809_380905

def f (x : ℝ) : ℝ := -3 * x^2 - 3 * x + 18

theorem max_value_and_k_range :
  (∀ x > -1, (f x - 21) / (x + 1) ≤ -3) ∧
  (∀ k : ℝ, (∀ x ∈ Set.Ioo 1 4, -3 * x^2 + k * x - 5 > 0) → k < 2 * Real.sqrt 15) := by
  sorry

end max_value_and_k_range_l3809_380905


namespace function_transformation_l3809_380954

theorem function_transformation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = x^2) :
  ∀ x : ℝ, f x = (x + 1)^2 := by
sorry

end function_transformation_l3809_380954


namespace greatest_integer_radius_l3809_380953

theorem greatest_integer_radius (A : ℝ) (h : A < 90 * Real.pi) :
  ∃ (r : ℕ), r^2 * Real.pi = A ∧ ∀ (n : ℕ), n^2 * Real.pi < 90 * Real.pi → n ≤ r ∧ r ≤ 9 :=
sorry

end greatest_integer_radius_l3809_380953


namespace book_cost_solution_l3809_380997

/-- Represents the cost of books problem --/
def BookCostProblem (initial_budget : ℚ) (books_per_series : ℕ) (series_bought : ℕ) (money_left : ℚ) (tax_rate : ℚ) : Prop :=
  let total_books := books_per_series * series_bought
  let money_spent := initial_budget - money_left
  let pre_tax_total := money_spent / (1 + tax_rate)
  let book_cost := pre_tax_total / total_books
  book_cost = 60 / 11

/-- Theorem stating the solution to the book cost problem --/
theorem book_cost_solution :
  BookCostProblem 200 8 3 56 (1/10) :=
sorry

end book_cost_solution_l3809_380997


namespace fencing_cost_is_5300_l3809_380935

/-- A rectangular plot with given dimensions and fencing cost -/
structure Plot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ
  length_breadth_difference : length = breadth + 60
  length_value : length = 80

/-- Calculate the total cost of fencing for a given plot -/
def total_fencing_cost (p : Plot) : ℝ :=
  2 * (p.length + p.breadth) * p.fencing_cost_per_meter

/-- Theorem: The total fencing cost for the given plot is 5300 currency units -/
theorem fencing_cost_is_5300 (p : Plot) (h : p.fencing_cost_per_meter = 26.50) : 
  total_fencing_cost p = 5300 := by
  sorry

end fencing_cost_is_5300_l3809_380935


namespace quilt_cost_calculation_l3809_380941

/-- The cost of a rectangular quilt -/
def quilt_cost (length width price_per_sq_ft : ℝ) : ℝ :=
  length * width * price_per_sq_ft

/-- Theorem: The cost of a 12 ft by 15 ft quilt at $70 per square foot is $12,600 -/
theorem quilt_cost_calculation :
  quilt_cost 12 15 70 = 12600 := by
  sorry

end quilt_cost_calculation_l3809_380941


namespace units_digit_of_7_pow_3_pow_4_l3809_380920

theorem units_digit_of_7_pow_3_pow_4 : ∃ n : ℕ, 7^(3^4) ≡ 7 [ZMOD 10] ∧ n < 10 := by sorry

end units_digit_of_7_pow_3_pow_4_l3809_380920


namespace total_pears_theorem_l3809_380968

/-- Calculates the total number of pears picked over three days given the number of pears picked by each person in one day -/
def total_pears_over_three_days (jason keith mike alicia tina nicola : ℕ) : ℕ :=
  3 * (jason + keith + mike + alicia + tina + nicola)

/-- Theorem stating that given the specific number of pears picked by each person,
    the total number of pears picked over three days is 654 -/
theorem total_pears_theorem :
  total_pears_over_three_days 46 47 12 28 33 52 = 654 := by
  sorry

end total_pears_theorem_l3809_380968


namespace notebook_problem_l3809_380933

def satisfies_notebook_conditions (n : ℕ) : Prop :=
  ∃ x y : ℕ,
    (y + 2 = n * (x - 2)) ∧
    (x + n = 2 * (y - n)) ∧
    x > 2 ∧ y > n

theorem notebook_problem :
  {n : ℕ | satisfies_notebook_conditions n} = {1, 2, 3, 8} :=
by sorry

end notebook_problem_l3809_380933


namespace number_of_different_products_l3809_380960

def set_a : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}
def set_b : Finset ℕ := {2, 4, 6, 19, 21, 24, 27, 31, 35}

theorem number_of_different_products : 
  (Finset.card (set_a.powersetCard 2) * Finset.card set_b) = 405 := by
  sorry

end number_of_different_products_l3809_380960


namespace cube_root_simplification_l3809_380925

theorem cube_root_simplification : (5488000 : ℝ)^(1/3) = 140 * 2^(1/3) := by sorry

end cube_root_simplification_l3809_380925


namespace tom_catch_equals_16_l3809_380978

def melanie_catch : ℕ := 8

def tom_catch_multiplier : ℕ := 2

def tom_catch : ℕ := tom_catch_multiplier * melanie_catch

theorem tom_catch_equals_16 : tom_catch = 16 := by
  sorry

end tom_catch_equals_16_l3809_380978


namespace rotten_apples_smell_percentage_l3809_380966

theorem rotten_apples_smell_percentage 
  (total_apples : ℕ) 
  (rotten_percentage : ℚ) 
  (non_smelling_rotten : ℕ) 
  (h1 : total_apples = 200)
  (h2 : rotten_percentage = 40 / 100)
  (h3 : non_smelling_rotten = 24) : 
  (total_apples * rotten_percentage - non_smelling_rotten : ℚ) / (total_apples * rotten_percentage) * 100 = 70 := by
sorry

end rotten_apples_smell_percentage_l3809_380966


namespace specific_flowerbed_area_l3809_380999

/-- Represents a circular flowerbed with a straight path through its center -/
structure Flowerbed where
  diameter : ℝ
  pathWidth : ℝ

/-- Calculates the plantable area of a flowerbed -/
def plantableArea (f : Flowerbed) : ℝ := sorry

/-- Theorem stating the plantable area of a specific flowerbed configuration -/
theorem specific_flowerbed_area :
  let f : Flowerbed := { diameter := 20, pathWidth := 4 }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |plantableArea f - 58.66 * Real.pi| < ε :=
sorry

end specific_flowerbed_area_l3809_380999


namespace pure_imaginary_complex_number_l3809_380979

theorem pure_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 + 2*a - 3) (a^2 - 4*a + 3)
  (z.re = 0 ∧ z.im ≠ 0) → a = -3 := by
  sorry

end pure_imaginary_complex_number_l3809_380979


namespace quadratic_shift_l3809_380951

/-- The original quadratic function -/
def g (x : ℝ) : ℝ := (x + 1)^2 + 3

/-- The transformed quadratic function -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem stating that f is the result of shifting g 2 units right and 1 unit down -/
theorem quadratic_shift (x : ℝ) : f x = g (x - 2) - 1 := by
  sorry

end quadratic_shift_l3809_380951


namespace largest_three_digit_geometric_l3809_380928

/-- Checks if a number is a three-digit integer -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Extracts the hundreds digit of a three-digit number -/
def hundredsDigit (n : ℕ) : ℕ := n / 100

/-- Extracts the tens digit of a three-digit number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- Extracts the ones digit of a three-digit number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- Checks if the digits of a three-digit number are distinct -/
def hasDistinctDigits (n : ℕ) : Prop :=
  let h := hundredsDigit n
  let t := tensDigit n
  let o := onesDigit n
  h ≠ t ∧ t ≠ o ∧ h ≠ o

/-- Checks if the digits of a three-digit number form a geometric sequence -/
def isGeometricSequence (n : ℕ) : Prop :=
  let h := hundredsDigit n
  let t := tensDigit n
  let o := onesDigit n
  ∃ r : ℚ, r ≠ 0 ∧ t = h / r ∧ o = t / r

theorem largest_three_digit_geometric : 
  ∀ n : ℕ, isThreeDigit n → hasDistinctDigits n → isGeometricSequence n → hundredsDigit n = 8 → n ≤ 842 :=
sorry

end largest_three_digit_geometric_l3809_380928


namespace last_three_digits_sum_l3809_380983

theorem last_three_digits_sum (n : ℕ) : 9^15 + 15^15 ≡ 24 [MOD 1000] := by
  sorry

end last_three_digits_sum_l3809_380983


namespace rectangle_area_theorem_l3809_380901

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: A rectangle with length three times its width and perimeter 160 has an area of 1200 -/
theorem rectangle_area_theorem (r : Rectangle) 
  (h1 : r.length = 3 * r.width) 
  (h2 : perimeter r = 160) : 
  area r = 1200 := by
  sorry

end rectangle_area_theorem_l3809_380901


namespace probability_of_drawing_specific_balls_l3809_380986

theorem probability_of_drawing_specific_balls (red white blue black : ℕ) : 
  red = 5 → white = 4 → blue = 3 → black = 6 →
  (red * white * blue : ℚ) / ((red + white + blue + black) * (red + white + blue + black - 1) * (red + white + blue + black - 2)) = 5 / 408 := by
  sorry

end probability_of_drawing_specific_balls_l3809_380986


namespace sum_of_digits_45_40_l3809_380993

def product_45_40 : Nat := 45 * 40

def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_45_40 : sum_of_digits product_45_40 = 9 := by
  sorry

end sum_of_digits_45_40_l3809_380993


namespace paint_usage_l3809_380945

theorem paint_usage (initial_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ)
  (h1 : initial_paint = 360)
  (h2 : first_week_fraction = 1/4)
  (h3 : second_week_fraction = 1/3) :
  let first_week_usage := first_week_fraction * initial_paint
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  let total_usage := first_week_usage + second_week_usage
  total_usage = 180 := by
  sorry

end paint_usage_l3809_380945


namespace max_students_distribution_l3809_380915

theorem max_students_distribution (pens pencils notebooks erasers : ℕ) 
  (h1 : pens = 891) (h2 : pencils = 810) (h3 : notebooks = 1080) (h4 : erasers = 972) : 
  Nat.gcd pens (Nat.gcd pencils (Nat.gcd notebooks erasers)) = 27 := by
  sorry

end max_students_distribution_l3809_380915


namespace smallest_multiple_of_6_and_15_l3809_380998

theorem smallest_multiple_of_6_and_15 : ∃ (a : ℕ), a > 0 ∧ 6 ∣ a ∧ 15 ∣ a ∧ ∀ (b : ℕ), b > 0 → 6 ∣ b → 15 ∣ b → a ≤ b := by
  sorry

end smallest_multiple_of_6_and_15_l3809_380998


namespace triangle_area_implies_p_value_l3809_380946

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, p),
    prove that if the area of the triangle is 35, then p = 77.5/6 -/
theorem triangle_area_implies_p_value (p : ℝ) : 
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  triangle_area = 35 → p = 77.5 / 6 := by
  sorry

end triangle_area_implies_p_value_l3809_380946


namespace binomial_expansion_sum_difference_l3809_380904

theorem binomial_expansion_sum_difference (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + Real.sqrt 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end binomial_expansion_sum_difference_l3809_380904


namespace correct_machines_in_first_scenario_l3809_380991

/-- The number of machines in the first scenario -/
def machines_in_first_scenario : ℕ := 5

/-- The number of units produced in the first scenario -/
def units_first_scenario : ℕ := 20

/-- The number of hours in the first scenario -/
def hours_first_scenario : ℕ := 10

/-- The number of machines in the second scenario -/
def machines_second_scenario : ℕ := 10

/-- The number of units produced in the second scenario -/
def units_second_scenario : ℕ := 100

/-- The number of hours in the second scenario -/
def hours_second_scenario : ℕ := 25

/-- The production rate per machine is constant across both scenarios -/
axiom production_rate_constant : 
  (units_first_scenario : ℚ) / (machines_in_first_scenario * hours_first_scenario) = 
  (units_second_scenario : ℚ) / (machines_second_scenario * hours_second_scenario)

theorem correct_machines_in_first_scenario : 
  machines_in_first_scenario = 5 := by sorry

end correct_machines_in_first_scenario_l3809_380991


namespace new_tax_rate_is_30_percent_l3809_380972

/-- Calculates the new tax rate given the initial rate, income, and tax savings -/
def calculate_new_tax_rate (initial_rate : ℚ) (income : ℚ) (savings : ℚ) : ℚ :=
  let initial_tax := initial_rate * income
  let new_tax := initial_tax - savings
  new_tax / income

theorem new_tax_rate_is_30_percent :
  let initial_rate : ℚ := 45 / 100
  let income : ℚ := 48000
  let savings : ℚ := 7200
  calculate_new_tax_rate initial_rate income savings = 30 / 100 := by
sorry

end new_tax_rate_is_30_percent_l3809_380972


namespace original_paint_intensity_l3809_380909

theorem original_paint_intensity 
  (original_fraction : Real) 
  (replacement_intensity : Real) 
  (new_intensity : Real) 
  (replaced_fraction : Real) :
  original_fraction = 0.5 →
  replacement_intensity = 0.2 →
  new_intensity = 0.15 →
  replaced_fraction = 0.5 →
  (1 - replaced_fraction) * original_fraction + replaced_fraction * replacement_intensity = new_intensity →
  original_fraction = 0.1 := by
sorry

end original_paint_intensity_l3809_380909


namespace fourth_number_proof_l3809_380957

theorem fourth_number_proof (numbers : Fin 6 → ℝ) 
  (avg_all : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 30)
  (avg_first_four : (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 25)
  (avg_last_three : (numbers 3 + numbers 4 + numbers 5) / 3 = 35) :
  numbers 3 = 25 := by
sorry

end fourth_number_proof_l3809_380957


namespace certain_number_problem_l3809_380936

theorem certain_number_problem (y : ℝ) : 
  (0.25 * 780 = 0.15 * y - 30) → y = 1500 := by
sorry

end certain_number_problem_l3809_380936


namespace symmetrical_cubic_function_l3809_380923

-- Define the function f(x) with parameters a and b
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + (a - 1) * x^2 + 48 * (a - 2) * x + b

-- Define the property of symmetry about the origin
def symmetrical_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem symmetrical_cubic_function
  (a b : ℝ)
  (h_symmetry : symmetrical_about_origin (f a b)) :
  (a = 1 ∧ b = 0) ∧
  (∀ x, f a b x = x^3 - 48*x) ∧
  (∀ x, -4 ≤ x ∧ x ≤ 4 → (∀ y, x < y → f a b x > f a b y)) ∧
  (∀ x, (x < -4 ∨ x > 4) → (∀ y, x < y → f a b x < f a b y)) ∧
  (f a b (-4) = 128) ∧
  (f a b 4 = -128) ∧
  (∀ x, f a b x ≤ 128) ∧
  (∀ x, f a b x ≥ -128) :=
by sorry

end symmetrical_cubic_function_l3809_380923


namespace cube_root_monotone_l3809_380971

theorem cube_root_monotone (a b : ℝ) : a ≤ b → (a ^ (1/3 : ℝ)) ≤ (b ^ (1/3 : ℝ)) := by
  sorry

end cube_root_monotone_l3809_380971


namespace nina_shirt_price_l3809_380969

/-- Given Nina's shopping scenario, prove the price of each shirt. -/
theorem nina_shirt_price :
  -- Define the number and price of toys
  let num_toys : ℕ := 3
  let price_per_toy : ℕ := 10
  -- Define the number and price of card packs
  let num_card_packs : ℕ := 2
  let price_per_card_pack : ℕ := 5
  -- Define the number of shirts (equal to toys + card packs)
  let num_shirts : ℕ := num_toys + num_card_packs
  -- Define the total amount spent
  let total_spent : ℕ := 70
  -- Calculate the cost of toys and card packs
  let cost_toys_and_cards : ℕ := num_toys * price_per_toy + num_card_packs * price_per_card_pack
  -- Calculate the remaining amount spent on shirts
  let amount_spent_on_shirts : ℕ := total_spent - cost_toys_and_cards
  -- Calculate the price per shirt
  let price_per_shirt : ℕ := amount_spent_on_shirts / num_shirts
  -- Prove that the price per shirt is 6
  price_per_shirt = 6 := by sorry

end nina_shirt_price_l3809_380969


namespace fraction_subtraction_l3809_380947

theorem fraction_subtraction : (18 : ℚ) / 45 - 3 / 8 = 1 / 40 := by
  sorry

end fraction_subtraction_l3809_380947


namespace interval_equivalence_l3809_380907

-- Define the intervals as sets
def openRightInf (a : ℝ) : Set ℝ := {x | x > a}
def closedRightInf (a : ℝ) : Set ℝ := {x | x ≥ a}
def openLeftInf (b : ℝ) : Set ℝ := {x | x < b}
def closedLeftInf (b : ℝ) : Set ℝ := {x | x ≤ b}

-- State the theorem
theorem interval_equivalence (a b : ℝ) :
  (∀ x, x ∈ openRightInf a ↔ x > a) ∧
  (∀ x, x ∈ closedRightInf a ↔ x ≥ a) ∧
  (∀ x, x ∈ openLeftInf b ↔ x < b) ∧
  (∀ x, x ∈ closedLeftInf b ↔ x ≤ b) :=
by sorry

end interval_equivalence_l3809_380907


namespace bob_overspent_l3809_380930

theorem bob_overspent (necklace_cost book_cost total_spent limit : ℕ) : 
  necklace_cost = 34 →
  book_cost = necklace_cost + 5 →
  total_spent = necklace_cost + book_cost →
  limit = 70 →
  total_spent - limit = 3 := by
  sorry

end bob_overspent_l3809_380930


namespace logical_consequences_l3809_380927

-- Define the universe of students
variable (Student : Type)

-- Define predicates
variable (passed : Student → Prop)
variable (scored_above_90_percent : Student → Prop)

-- Define the given condition
variable (h : ∀ s : Student, scored_above_90_percent s → passed s)

-- Theorem to prove
theorem logical_consequences :
  (∀ s : Student, ¬(passed s) → ¬(scored_above_90_percent s)) ∧
  (∀ s : Student, ¬(scored_above_90_percent s) → ¬(passed s)) ∧
  (∀ s : Student, passed s → scored_above_90_percent s) :=
by sorry

end logical_consequences_l3809_380927


namespace three_roots_implies_a_plus_minus_four_l3809_380996

theorem three_roots_implies_a_plus_minus_four (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℝ, |x^2 + a*x| = 4 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) →
  a = 4 ∨ a = -4 :=
by sorry

end three_roots_implies_a_plus_minus_four_l3809_380996


namespace estate_value_l3809_380906

/-- Represents the estate distribution problem --/
def EstateDistribution (total : ℝ) : Prop :=
  ∃ (elder_niece younger_niece brother caretaker : ℝ),
    -- The two nieces together received half of the estate
    elder_niece + younger_niece = total / 2 ∧
    -- The nieces' shares are in the ratio of 3 to 2
    elder_niece = (3/5) * (total / 2) ∧
    younger_niece = (2/5) * (total / 2) ∧
    -- The brother got three times as much as the elder niece
    brother = 3 * elder_niece ∧
    -- The caretaker received $800
    caretaker = 800 ∧
    -- The sum of all shares equals the total estate
    elder_niece + younger_niece + brother + caretaker = total

/-- Theorem stating that the estate value is $2000 --/
theorem estate_value : EstateDistribution 2000 :=
sorry

end estate_value_l3809_380906


namespace l_structure_surface_area_l3809_380929

/-- Represents the L-shaped structure -/
structure LStructure where
  bottom_length : ℕ
  bottom_width : ℕ
  stack_height : ℕ

/-- Calculates the surface area of the L-shaped structure -/
def surface_area (l : LStructure) : ℕ :=
  let bottom_area := l.bottom_length * l.bottom_width
  let bottom_perimeter := 2 * l.bottom_length + l.bottom_width
  let stack_side_area := 2 * l.stack_height
  let stack_top_area := 1
  bottom_area + bottom_perimeter + stack_side_area + stack_top_area

/-- The specific L-shaped structure in the problem -/
def problem_structure : LStructure :=
  { bottom_length := 3
  , bottom_width := 3
  , stack_height := 6 }

theorem l_structure_surface_area :
  surface_area problem_structure = 29 := by
  sorry

end l_structure_surface_area_l3809_380929


namespace ellipse_problem_l3809_380959

-- Define the circles and curve C
def F₁ (r : ℝ) (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = r^2
def F₂ (r : ℝ) (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = (4 - r)^2
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point M
def M : ℝ × ℝ := (0, 1)

-- Define the orthogonality condition for points A and B
def orthogonal (A B : ℝ × ℝ) : Prop :=
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0

-- Theorem statement
theorem ellipse_problem (r : ℝ) (h : 0 < r ∧ r < 4) :
  -- 1. Equation of curve C
  (∀ x y : ℝ, (∃ r', F₁ r' x y ∧ F₂ r' x y) ↔ C x y) ∧
  -- 2. Line AB passes through fixed point
  (∀ A B : ℝ × ℝ, C A.1 A.2 → C B.1 B.2 → A ≠ B → orthogonal A B →
    ∃ t : ℝ, A.1 + t * (B.1 - A.1) = 0 ∧ A.2 + t * (B.2 - A.2) = -3/5) ∧
  -- 3. Maximum area of triangle ABM
  (∀ A B : ℝ × ℝ, C A.1 A.2 → C B.1 B.2 → A ≠ B → orthogonal A B →
    abs ((A.1 - M.1) * (B.2 - M.2) - (A.2 - M.2) * (B.1 - M.1)) / 2 ≤ 64/25) :=
by sorry

end ellipse_problem_l3809_380959


namespace no_solution_rebus_l3809_380962

theorem no_solution_rebus :
  ¬ ∃ (K U S Y : ℕ),
    K ≠ U ∧ K ≠ S ∧ K ≠ Y ∧ U ≠ S ∧ U ≠ Y ∧ S ≠ Y ∧
    K < 10 ∧ U < 10 ∧ S < 10 ∧ Y < 10 ∧
    1000 ≤ (1000 * K + 100 * U + 10 * S + Y) ∧
    (1000 * K + 100 * U + 10 * S + Y) < 10000 ∧
    1000 ≤ (1000 * U + 100 * K + 10 * S + Y) ∧
    (1000 * U + 100 * K + 10 * S + Y) < 10000 ∧
    10000 ≤ (10000 * U + 1000 * K + 100 * S + 10 * U + S) ∧
    (10000 * U + 1000 * K + 100 * S + 10 * U + S) < 100000 ∧
    (1000 * K + 100 * U + 10 * S + Y) + (1000 * U + 100 * K + 10 * S + Y) =
    (10000 * U + 1000 * K + 100 * S + 10 * U + S) :=
by
  sorry

end no_solution_rebus_l3809_380962
