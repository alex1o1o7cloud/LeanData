import Mathlib

namespace NUMINAMATH_GPT_find_F_neg_a_l2239_223906

-- Definitions of odd functions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Definition of F
def F (f g : ℝ → ℝ) (x : ℝ) := 3 * f x + 5 * g x + 2

theorem find_F_neg_a (f g : ℝ → ℝ) (a : ℝ)
  (hf : is_odd f) (hg : is_odd g) (hFa : F f g a = 3) : F f g (-a) = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_F_neg_a_l2239_223906


namespace NUMINAMATH_GPT_equation_of_line_l_equations_of_line_m_l2239_223910

-- Define the point P and condition for line l
def P := (2, (7 : ℚ)/4)
def l_slope : ℚ := 3 / 4

-- Define the given equation form and conditions for line l
def condition_l (x y : ℚ) : Prop := y - (7 / 4) = (3 / 4) * (x - 2)
def equation_l (x y : ℚ) : Prop := 3 * x - 4 * y = 5

theorem equation_of_line_l :
  ∀ x y : ℚ, condition_l x y → equation_l x y :=
sorry

-- Define the distance condition for line m
def equation_m (x y n : ℚ) : Prop := 3 * x - 4 * y + n = 0
def distance_condition_m (n : ℚ) : Prop := 
  |(-1 + n : ℚ)| / 5 = 3

theorem equations_of_line_m :
  ∃ n : ℚ, distance_condition_m n ∧ (equation_m 2 (7/4) n) ∨ 
            equation_m 2 (7/4) (-14) :=
sorry

end NUMINAMATH_GPT_equation_of_line_l_equations_of_line_m_l2239_223910


namespace NUMINAMATH_GPT_find_fraction_l2239_223924

noncomputable def fraction_of_eighths (N : ℝ) (a b : ℝ) : Prop :=
  (3/8) * N * (a/b) = 24

noncomputable def two_fifty_percent (N : ℝ) : Prop :=
  2.5 * N = 199.99999999999997

theorem find_fraction {N a b : ℝ} (h1 : fraction_of_eighths N a b) (h2 : two_fifty_percent N) :
  a/b = 4/5 :=
sorry

end NUMINAMATH_GPT_find_fraction_l2239_223924


namespace NUMINAMATH_GPT_third_term_is_five_l2239_223949

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Suppose S_n = n^2 for n ∈ ℕ*
axiom H1 : ∀ n : ℕ, n > 0 → S n = n * n

-- The relationship a_n = S_n - S_(n-1) for n ≥ 2
axiom H2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n - 1)

-- Prove that the third term is 5
theorem third_term_is_five : a 3 = 5 := by
  sorry

end NUMINAMATH_GPT_third_term_is_five_l2239_223949


namespace NUMINAMATH_GPT_am_gm_inequality_example_l2239_223911

theorem am_gm_inequality_example (x y : ℝ) (hx : x = 16) (hy : y = 64) : 
  (x + y) / 2 ≥ Real.sqrt (x * y) :=
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_am_gm_inequality_example_l2239_223911


namespace NUMINAMATH_GPT_rounding_bounds_l2239_223920

theorem rounding_bounds:
  ∃ (max min : ℕ), (∀ x : ℕ, (x >= 1305000) → (x < 1305000) -> false) ∧ 
  (max = 1304999) ∧ 
  (min = 1295000) :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_rounding_bounds_l2239_223920


namespace NUMINAMATH_GPT_no_solution_inequalities_l2239_223912

theorem no_solution_inequalities (a : ℝ) :
  (¬ ∃ x : ℝ, x > 1 ∧ x < a - 1) → a ≤ 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_no_solution_inequalities_l2239_223912


namespace NUMINAMATH_GPT_pears_for_twenty_apples_l2239_223966

-- Definitions based on given conditions
variables (a o p : ℕ) -- represent the number of apples, oranges, and pears respectively
variables (k1 k2 : ℕ) -- scaling factors 

-- Conditions as given
axiom ten_apples_five_oranges : 10 * a = 5 * o
axiom three_oranges_four_pears : 3 * o = 4 * p

-- Proving the number of pears Mia can buy for 20 apples
theorem pears_for_twenty_apples : 13 * p ≤ (20 * a) :=
by
  -- Actual proof would go here
  sorry

end NUMINAMATH_GPT_pears_for_twenty_apples_l2239_223966


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2239_223927

theorem minimum_value_of_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 3) : 
  a^2 + 8 * a * b + 32 * b^2 + 24 * b * c + 8 * c^2 ≥ 72 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2239_223927


namespace NUMINAMATH_GPT_least_number_to_add_l2239_223996

theorem least_number_to_add (n : ℕ) (sum_digits : ℕ) (next_multiple : ℕ) 
  (h1 : n = 51234) 
  (h2 : sum_digits = 5 + 1 + 2 + 3 + 4) 
  (h3 : next_multiple = 18) :
  ∃ k, (k = next_multiple - sum_digits) ∧ (n + k) % 9 = 0 :=
sorry

end NUMINAMATH_GPT_least_number_to_add_l2239_223996


namespace NUMINAMATH_GPT_total_price_increase_percentage_l2239_223997

theorem total_price_increase_percentage 
    (P : ℝ) 
    (h1 : P > 0) 
    (P_after_first_increase : ℝ := P * 1.2) 
    (P_after_second_increase : ℝ := P_after_first_increase * 1.15) :
    ((P_after_second_increase - P) / P) * 100 = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_price_increase_percentage_l2239_223997


namespace NUMINAMATH_GPT_find_n_positive_integer_l2239_223952

theorem find_n_positive_integer:
  ∀ n : ℕ, n > 0 → (∃ k : ℕ, 2^n + 12^n + 2011^n = k^2) ↔ n = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_n_positive_integer_l2239_223952


namespace NUMINAMATH_GPT_line_through_center_and_perpendicular_l2239_223957

theorem line_through_center_and_perpendicular 
(C : ℝ × ℝ) 
(HC : ∀ (x y : ℝ), x ^ 2 + (y - 1) ^ 2 = 4 → C = (0, 1))
(l : ℝ → ℝ)
(Hl : ∀ x y : ℝ, 3 * x + 2 * y + 1 = 0 → y = l x)
: ∃ k b : ℝ, (∀ x : ℝ, y = k * x + b ↔ 2 * x - 3 * y + 3 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_line_through_center_and_perpendicular_l2239_223957


namespace NUMINAMATH_GPT_elapsed_time_l2239_223980

variable (totalDistance : ℕ) (runningSpeed : ℕ) (distanceRemaining : ℕ)

theorem elapsed_time (h1 : totalDistance = 120) (h2 : runningSpeed = 4) (h3 : distanceRemaining = 20) :
  (totalDistance - distanceRemaining) / runningSpeed = 25 := by
sorry

end NUMINAMATH_GPT_elapsed_time_l2239_223980


namespace NUMINAMATH_GPT_find_missing_number_l2239_223973

theorem find_missing_number (square boxplus boxtimes boxminus : ℕ) :
  square = 423 / 47 ∧
  1448 = 282 * boxminus + (boxminus * 10 + boxtimes) ∧
  423 * (boxplus / 3) = 282 →
  square = 9 ∧
  boxminus = 5 ∧
  boxtimes = 8 ∧
  boxplus = 2 ∧
  9 = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_missing_number_l2239_223973


namespace NUMINAMATH_GPT_rent_percentage_l2239_223939

-- Define Elaine's earnings last year
def E : ℝ := sorry

-- Define last year's rent expenditure
def rentLastYear : ℝ := 0.20 * E

-- Define this year's earnings
def earningsThisYear : ℝ := 1.35 * E

-- Define this year's rent expenditure
def rentThisYear : ℝ := 0.30 * earningsThisYear

-- Prove the required percentage
theorem rent_percentage : ((rentThisYear / rentLastYear) * 100) = 202.5 := by
  sorry

end NUMINAMATH_GPT_rent_percentage_l2239_223939


namespace NUMINAMATH_GPT_convex_quadrilateral_diagonal_l2239_223916

theorem convex_quadrilateral_diagonal (P : ℝ) (d1 d2 : ℝ) (hP : P = 2004) (hd1 : d1 = 1001) :
  (d2 = 1 → False) ∧ 
  (d2 = 2 → True) ∧ 
  (d2 = 1001 → True) :=
by
  sorry

end NUMINAMATH_GPT_convex_quadrilateral_diagonal_l2239_223916


namespace NUMINAMATH_GPT_sum_of_possible_values_l2239_223988

theorem sum_of_possible_values (x y : ℝ)
  (h : x * y - (2 * x) / (y ^ 3) - (2 * y) / (x ^ 3) = 5) :
  ∃ s : ℝ, s = (x - 2) * (y - 2) ∧ (s = -3 ∨ s = 9) :=
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l2239_223988


namespace NUMINAMATH_GPT_union_of_sets_l2239_223936

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

theorem union_of_sets : A ∪ B = {-1, 0, 1, 2} := 
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l2239_223936


namespace NUMINAMATH_GPT_square_sides_product_l2239_223987

theorem square_sides_product (a : ℝ) : 
  (∃ s : ℝ, s = 5 ∧ (a = -3 + s ∨ a = -3 - s)) → (a = 2 ∨ a = -8) → -8 * 2 = -16 :=
by
  intro _ _
  exact rfl

end NUMINAMATH_GPT_square_sides_product_l2239_223987


namespace NUMINAMATH_GPT_product_of_inverses_l2239_223994

theorem product_of_inverses : 
  ((1 - 1 / (3^2)) * (1 - 1 / (5^2)) * (1 - 1 / (7^2)) * (1 - 1 / (11^2)) * (1 - 1 / (13^2)) * (1 - 1 / (17^2))) = 210 / 221 := 
by {
  sorry
}

end NUMINAMATH_GPT_product_of_inverses_l2239_223994


namespace NUMINAMATH_GPT_cost_for_23_days_l2239_223909

-- Define the cost structure
def costFirstWeek : ℕ → ℝ := λ days => if days <= 7 then days * 18 else 7 * 18
def costAdditionalDays : ℕ → ℝ := λ days => if days > 7 then (days - 7) * 14 else 0

-- Total cost equation
def totalCost (days : ℕ) : ℝ := costFirstWeek days + costAdditionalDays days

-- Declare the theorem to prove
theorem cost_for_23_days : totalCost 23 = 350 := by
  sorry

end NUMINAMATH_GPT_cost_for_23_days_l2239_223909


namespace NUMINAMATH_GPT_find_number_l2239_223930

theorem find_number (x : ℝ) (h : 0.6667 * x + 1 = 0.75 * x) : x = 12 :=
sorry

end NUMINAMATH_GPT_find_number_l2239_223930


namespace NUMINAMATH_GPT_arithmetic_progression_numbers_l2239_223922

theorem arithmetic_progression_numbers :
  ∃ (a d : ℚ), (3 * (2 * a - d) = 2 * (a + d)) ∧ ((a - d) * (a + d) = (a - 2)^2) ∧
  ((a = 5 ∧ d = 4 ∧ ∃ b c : ℚ, b = (a - d) ∧ c = (a + d) ∧ b = 1 ∧ c = 9) 
   ∨ (a = 5 / 4 ∧ d = 1 ∧ ∃ b c : ℚ, b = (a - d) ∧ c = (a + d) ∧ b = 1 / 4 ∧ c = 9 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_numbers_l2239_223922


namespace NUMINAMATH_GPT_truck_capacity_l2239_223975

theorem truck_capacity
  (x y : ℝ)
  (h1 : 2 * x + 3 * y = 15.5)
  (h2 : 5 * x + 6 * y = 35) :
  3 * x + 5 * y = 24.5 :=
sorry

end NUMINAMATH_GPT_truck_capacity_l2239_223975


namespace NUMINAMATH_GPT_area_of_rectangle_l2239_223976

theorem area_of_rectangle (l w : ℝ) (h_perimeter : 2 * (l + w) = 126) (h_difference : l - w = 37) : l * w = 650 :=
sorry

end NUMINAMATH_GPT_area_of_rectangle_l2239_223976


namespace NUMINAMATH_GPT_average_salary_of_employees_l2239_223925

theorem average_salary_of_employees (A : ℝ) 
  (h1 : (20 : ℝ) * A + 3400 = 21 * (A + 100)) : 
  A = 1300 := 
by 
  -- proof goes here 
  sorry

end NUMINAMATH_GPT_average_salary_of_employees_l2239_223925


namespace NUMINAMATH_GPT_circumscribed_sphere_radius_is_3_l2239_223977

noncomputable def radius_of_circumscribed_sphere (SA SB SC : ℝ) : ℝ :=
  let space_diagonal := Real.sqrt (SA^2 + SB^2 + SC^2)
  space_diagonal / 2

theorem circumscribed_sphere_radius_is_3 : radius_of_circumscribed_sphere 2 4 4 = 3 :=
by
  unfold radius_of_circumscribed_sphere
  simp
  apply sorry

end NUMINAMATH_GPT_circumscribed_sphere_radius_is_3_l2239_223977


namespace NUMINAMATH_GPT_wood_rope_length_equivalence_l2239_223995

variable (x y : ℝ)

theorem wood_rope_length_equivalence :
  (x - y = 4.5) ∧ (y = (1 / 2) * x + 1) :=
  sorry

end NUMINAMATH_GPT_wood_rope_length_equivalence_l2239_223995


namespace NUMINAMATH_GPT_lemon_count_l2239_223981

theorem lemon_count {total_fruits mangoes pears pawpaws : ℕ} (kiwi lemon : ℕ) :
  total_fruits = 58 ∧ 
  mangoes = 18 ∧ 
  pears = 10 ∧ 
  pawpaws = 12 ∧ 
  (kiwi = lemon) →
  lemon = 9 :=
by 
  sorry

end NUMINAMATH_GPT_lemon_count_l2239_223981


namespace NUMINAMATH_GPT_ending_number_of_multiples_l2239_223984

theorem ending_number_of_multiples (n : ℤ) (h : 991 = (n - 100) / 10 + 1) : n = 10000 :=
by
  sorry

end NUMINAMATH_GPT_ending_number_of_multiples_l2239_223984


namespace NUMINAMATH_GPT_supreme_sports_package_channels_l2239_223970

theorem supreme_sports_package_channels (c_start : ℕ) (c_removed1 : ℕ) (c_added1 : ℕ)
                                         (c_removed2 : ℕ) (c_added2 : ℕ)
                                         (c_final : ℕ)
                                         (net1 : ℕ) (net2 : ℕ) (c_mid : ℕ) :
  c_start = 150 →
  c_removed1 = 20 →
  c_added1 = 12 →
  c_removed2 = 10 →
  c_added2 = 8 →
  c_final = 147 →
  net1 = c_removed1 - c_added1 →
  net2 = c_removed2 - c_added2 →
  c_mid = c_start - net1 - net2 →
  c_final - c_mid = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_supreme_sports_package_channels_l2239_223970


namespace NUMINAMATH_GPT_discount_correct_l2239_223918

def normal_cost : ℝ := 80
def discount_rate : ℝ := 0.45
def discounted_cost : ℝ := normal_cost - (discount_rate * normal_cost)

theorem discount_correct : discounted_cost = 44 := by
  -- By computation, 0.45 * 80 = 36 and 80 - 36 = 44
  sorry

end NUMINAMATH_GPT_discount_correct_l2239_223918


namespace NUMINAMATH_GPT_gcd_9125_4277_l2239_223967

theorem gcd_9125_4277 : Nat.gcd 9125 4277 = 1 :=
by
  -- proof by Euclidean algorithm steps
  sorry

end NUMINAMATH_GPT_gcd_9125_4277_l2239_223967


namespace NUMINAMATH_GPT_odd_difference_even_odd_l2239_223983

theorem odd_difference_even_odd (a b : ℤ) (ha : a % 2 = 0) (hb : b % 2 = 1) : (a - b) % 2 = 1 :=
sorry

end NUMINAMATH_GPT_odd_difference_even_odd_l2239_223983


namespace NUMINAMATH_GPT_intersection_is_empty_l2239_223963

-- Define sets M and N
def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {0, 3, 4}

-- Define isolated elements for a set
def is_isolated (A : Set ℕ) (x : ℕ) : Prop :=
  x ∈ A ∧ (x - 1 ∉ A) ∧ (x + 1 ∉ A)

-- Define isolated sets
def isolated_set (A : Set ℕ) : Set ℕ :=
  {x | is_isolated A x}

-- Define isolated sets for M and N
def M' := isolated_set M
def N' := isolated_set N

-- The intersection of the isolated sets
theorem intersection_is_empty : M' ∩ N' = ∅ := 
  sorry

end NUMINAMATH_GPT_intersection_is_empty_l2239_223963


namespace NUMINAMATH_GPT_blue_pieces_correct_l2239_223974

def total_pieces : ℕ := 3409
def red_pieces : ℕ := 145
def blue_pieces : ℕ := total_pieces - red_pieces

theorem blue_pieces_correct : blue_pieces = 3264 := by
  sorry

end NUMINAMATH_GPT_blue_pieces_correct_l2239_223974


namespace NUMINAMATH_GPT_total_coins_constant_l2239_223989

-- Definitions based on the conditions
def stack1 := 12
def stack2 := 17
def stack3 := 23
def stack4 := 8

def totalCoins := stack1 + stack2 + stack3 + stack4 -- 60 coins
def is_divisor (x: ℕ) := x ∣ totalCoins

-- The theorem statement
theorem total_coins_constant {x: ℕ} (h: is_divisor x) : totalCoins = 60 :=
by
  -- skip the proof steps
  sorry

end NUMINAMATH_GPT_total_coins_constant_l2239_223989


namespace NUMINAMATH_GPT_largest_among_five_numbers_l2239_223940

theorem largest_among_five_numbers :
  max (max (max (max (12345 + 1 / 3579) 
                       (12345 - 1 / 3579))
                   (12345 ^ (1 / 3579)))
               (12345 / (1 / 3579)))
           12345.3579 = 12345 / (1 / 3579) := sorry

end NUMINAMATH_GPT_largest_among_five_numbers_l2239_223940


namespace NUMINAMATH_GPT_percentage_is_12_l2239_223903

variable (x : ℝ) (p : ℝ)

-- Given the conditions
def condition_1 : Prop := 0.25 * x = (p / 100) * 1500 - 15
def condition_2 : Prop := x = 660

-- We need to prove that the percentage p is 12
theorem percentage_is_12 (h1 : condition_1 x p) (h2 : condition_2 x) : p = 12 := by
  sorry

end NUMINAMATH_GPT_percentage_is_12_l2239_223903


namespace NUMINAMATH_GPT_order_of_fractions_l2239_223942

theorem order_of_fractions (a b c d : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (hpos_d : d > 0)
(hab : a > b) : (b / a) < (b + c) / (a + c) ∧ (b + c) / (a + c) < (a + d) / (b + d) ∧ (a + d) / (b + d) < (a / b) :=
by
  sorry

end NUMINAMATH_GPT_order_of_fractions_l2239_223942


namespace NUMINAMATH_GPT_spam_ratio_l2239_223901

theorem spam_ratio (total_emails important_emails promotional_fraction promotional_emails spam_emails : ℕ) 
  (h1 : total_emails = 400) 
  (h2 : important_emails = 180) 
  (h3 : promotional_fraction = 2/5) 
  (h4 : total_emails - important_emails = spam_emails + promotional_emails) 
  (h5 : promotional_emails = promotional_fraction * (total_emails - important_emails)) 
  : spam_emails / total_emails = 33 / 100 := 
by {
  sorry
}

end NUMINAMATH_GPT_spam_ratio_l2239_223901


namespace NUMINAMATH_GPT_jerry_reaches_3_at_some_time_l2239_223946

def jerry_reaches_3_probability (n : ℕ) (k : ℕ) : ℚ :=
  -- This function represents the probability that Jerry reaches 3 at some point during n coin tosses
  if n = 7 ∧ k = 3 then (21 / 64 : ℚ) else 0

theorem jerry_reaches_3_at_some_time :
  jerry_reaches_3_probability 7 3 = (21 / 64 : ℚ) :=
sorry

end NUMINAMATH_GPT_jerry_reaches_3_at_some_time_l2239_223946


namespace NUMINAMATH_GPT_ratio_boysGradeA_girlsGradeB_l2239_223986

variable (S G B : ℕ)

-- Given conditions
axiom h1 : (1 / 3 : ℚ) * G = (1 / 4 : ℚ) * S
axiom h2 : S = B + G

-- Definitions based on conditions
def boys_in_GradeA (B : ℕ) := (2 / 5 : ℚ) * B
def girls_in_GradeB (G : ℕ) := (3 / 5 : ℚ) * G

-- The proof goal
theorem ratio_boysGradeA_girlsGradeB (S G B : ℕ) (h1 : (1 / 3 : ℚ) * G = (1 / 4 : ℚ) * S) (h2 : S = B + G) :
    boys_in_GradeA B / girls_in_GradeB G = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_boysGradeA_girlsGradeB_l2239_223986


namespace NUMINAMATH_GPT_problem_l2239_223950

-- Definitions for the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def a (n : ℕ) : ℤ := sorry -- Define the arithmetic sequence a_n based on conditions

-- Problem statement
theorem problem : 
  (a 1 = 4) ∧
  (a 2 + a 4 = 4) →
  (∃ d : ℤ, arithmetic_sequence a d ∧ a 10 = -5) :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_l2239_223950


namespace NUMINAMATH_GPT_paul_baseball_cards_l2239_223934

-- Define the necessary variables and statements
variable {n : ℕ}

-- State the problem and the proof target
theorem paul_baseball_cards : ∃ k, k = 3 * n + 1 := sorry

end NUMINAMATH_GPT_paul_baseball_cards_l2239_223934


namespace NUMINAMATH_GPT_negation_of_sin_le_one_l2239_223917

theorem negation_of_sin_le_one : (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x > 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_sin_le_one_l2239_223917


namespace NUMINAMATH_GPT_millicent_fraction_books_l2239_223931

variable (M H : ℝ)
variable (F : ℝ)

-- Conditions
def harold_has_half_books (M H : ℝ) : Prop := H = (1 / 2) * M
def harold_brings_one_third_books (M H : ℝ) : Prop := (1 / 3) * H = (1 / 6) * M
def new_library_capacity (M F : ℝ) : Prop := (1 / 6) * M + F * M = (5 / 6) * M

-- Target Proof Statement
theorem millicent_fraction_books (M H F : ℝ) 
    (h1 : harold_has_half_books M H) 
    (h2 : harold_brings_one_third_books M H) 
    (h3 : new_library_capacity M F) : 
    F = 2 / 3 :=
sorry

end NUMINAMATH_GPT_millicent_fraction_books_l2239_223931


namespace NUMINAMATH_GPT_maddox_theo_equal_profit_l2239_223944

-- Definitions based on the problem conditions
def maddox_initial_cost := 10 * 35
def theo_initial_cost := 15 * 30
def maddox_revenue := 10 * 50
def theo_revenue := 15 * 40

-- Define profits based on the revenues and costs
def maddox_profit := maddox_revenue - maddox_initial_cost
def theo_profit := theo_revenue - theo_initial_cost

-- The theorem to be proved
theorem maddox_theo_equal_profit : maddox_profit = theo_profit :=
by
  -- Omitted proof steps
  sorry

end NUMINAMATH_GPT_maddox_theo_equal_profit_l2239_223944


namespace NUMINAMATH_GPT_fraction_of_white_surface_area_is_11_16_l2239_223990

theorem fraction_of_white_surface_area_is_11_16 :
  let cube_surface_area := 6 * 4^2
  let total_surface_faces := 96
  let corner_black_faces := 8 * 3
  let center_black_faces := 6 * 1
  let total_black_faces := corner_black_faces + center_black_faces
  let white_faces := total_surface_faces - total_black_faces
  (white_faces : ℚ) / total_surface_faces = 11 / 16 := 
by sorry

end NUMINAMATH_GPT_fraction_of_white_surface_area_is_11_16_l2239_223990


namespace NUMINAMATH_GPT_prime_power_minus_l2239_223947

theorem prime_power_minus (p : ℕ) (hp : Nat.Prime p) (hps : Nat.Prime (p + 3)) : p ^ 11 - 52 = 1996 := by
  -- this is where the proof would go
  sorry

end NUMINAMATH_GPT_prime_power_minus_l2239_223947


namespace NUMINAMATH_GPT_general_term_l2239_223951

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n / (2 + a n)

theorem general_term (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, n > 0 → a n = 2 / (n + 1) :=
by
sorry

end NUMINAMATH_GPT_general_term_l2239_223951


namespace NUMINAMATH_GPT_line_y2_does_not_pass_through_fourth_quadrant_l2239_223948

theorem line_y2_does_not_pass_through_fourth_quadrant (k b : ℝ) (h1 : k < 0) (h2 : b > 0) : 
  ¬(∃ x y : ℝ, (y = b * x - k ∧ x > 0 ∧ y < 0)) := 
by 
  sorry

end NUMINAMATH_GPT_line_y2_does_not_pass_through_fourth_quadrant_l2239_223948


namespace NUMINAMATH_GPT_correct_value_wrongly_copied_l2239_223928

theorem correct_value_wrongly_copied 
  (mean_initial : ℕ)
  (mean_correct : ℕ)
  (wrong_value : ℕ) 
  (n : ℕ) 
  (initial_mean : mean_initial = 250)
  (correct_mean : mean_correct = 251)
  (wrongly_copied : wrong_value = 135)
  (number_of_values : n = 30) : 
  ∃ x : ℕ, x = 165 := 
by
  use (wrong_value + (mean_correct - mean_initial) * n / n)
  sorry

end NUMINAMATH_GPT_correct_value_wrongly_copied_l2239_223928


namespace NUMINAMATH_GPT_sum_of_roots_l2239_223945

theorem sum_of_roots : 
  ( ∀ x : ℝ, x^2 - 7*x + 10 = 0 → x = 2 ∨ x = 5 ) → 
  ( 2 + 5 = 7 ) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l2239_223945


namespace NUMINAMATH_GPT_vertex_of_parabola_point_symmetry_on_parabola_range_of_m_l2239_223968

open Real

-- Problem 1: Prove the vertex of the parabola is at (1, -a)
theorem vertex_of_parabola (a : ℝ) (h : a ≠ 0) : 
  ∀ x : ℝ, y = a * x^2 - 2 * a * x → (1, -a) = ((1 : ℝ), - a) := 
sorry

-- Problem 2: Prove x_0 = 3 if m = n for given points on the parabola
theorem point_symmetry_on_parabola (a : ℝ) (h : a ≠ 0) (m n : ℝ) :
  m = n → ∀ (x0 : ℝ), y = a * x0 ^ 2 - 2 * a * x0 → x0 = 3 :=
sorry

-- Problem 3: Prove the conditions for y1 < y2 ≤ -a and the range of m
theorem range_of_m (a : ℝ) (h : a < 0) : 
  ∀ (m y1 y2 : ℝ), (y1 < y2) ∧ (y2 ≤ -a) → m < (1 / 2) := 
sorry

end NUMINAMATH_GPT_vertex_of_parabola_point_symmetry_on_parabola_range_of_m_l2239_223968


namespace NUMINAMATH_GPT_unique_solution_of_quadratic_l2239_223914

theorem unique_solution_of_quadratic (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a = 0 ∨ a = 9 / 8) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_of_quadratic_l2239_223914


namespace NUMINAMATH_GPT_reciprocal_of_5_is_1_div_5_l2239_223961

-- Define the concept of reciprocal
def is_reciprocal (a b : ℚ) : Prop := a * b = 1

-- The problem statement: Prove that the reciprocal of 5 is 1/5
theorem reciprocal_of_5_is_1_div_5 : is_reciprocal 5 (1 / 5) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_5_is_1_div_5_l2239_223961


namespace NUMINAMATH_GPT_fifth_friend_contribution_l2239_223965

variables (a b c d e : ℕ)

theorem fifth_friend_contribution:
  a + b + c + d + e = 120 ∧
  a = 2 * b ∧
  b = (c + d) / 3 ∧
  c = 2 * e →
  e = 12 :=
sorry

end NUMINAMATH_GPT_fifth_friend_contribution_l2239_223965


namespace NUMINAMATH_GPT_y_neither_directly_nor_inversely_proportional_l2239_223937

theorem y_neither_directly_nor_inversely_proportional (x y : ℝ) :
  ¬((∃ k : ℝ, x = k * y) ∨ (∃ k : ℝ, x * y = k)) ↔ 2 * x + 3 * y = 6 :=
by 
  sorry

end NUMINAMATH_GPT_y_neither_directly_nor_inversely_proportional_l2239_223937


namespace NUMINAMATH_GPT_opposite_of_neg_four_l2239_223992

-- Define the condition: the opposite of a number is the number that, when added to the original number, results in zero.
def is_opposite (a b : Int) : Prop := a + b = 0

-- The specific theorem we want to prove
theorem opposite_of_neg_four : is_opposite (-4) 4 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_opposite_of_neg_four_l2239_223992


namespace NUMINAMATH_GPT_find_e_l2239_223938

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_e (r s d e : ℝ) 
  (h1 : quadratic 2 (-4) (-6) r = 0)
  (h2 : quadratic 2 (-4) (-6) s = 0)
  (h3 : r + s = 2) 
  (h4 : r * s = -3)
  (h5 : d = -(r + s - 6))
  (h6 : e = (r - 3) * (s - 3)) : 
  e = 0 :=
sorry

end NUMINAMATH_GPT_find_e_l2239_223938


namespace NUMINAMATH_GPT_ratio_of_fractions_l2239_223932

-- Given conditions
variables {x y : ℚ}
variables (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0)

-- Assertion to be proved
theorem ratio_of_fractions (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) :
  (1 / 5 * x) / (1 / 6 * y) = 18 / 25 :=
sorry

end NUMINAMATH_GPT_ratio_of_fractions_l2239_223932


namespace NUMINAMATH_GPT_calculate_A_share_l2239_223904

variable (x : ℝ) (total_gain : ℝ)
variable (h_b_invests : 2 * x)  -- B invests double the amount after 6 months
variable (h_c_invests : 3 * x)  -- C invests thrice the amount after 8 months

/-- Calculate the share of A from the total annual gain -/
theorem calculate_A_share (h_total_gain : total_gain = 18600) :
  let a_investmentMonths := x * 12
  let b_investmentMonths := (2 * x) * 6
  let c_investmentMonths := (3 * x) * 4
  let total_investmentMonths := a_investmentMonths + b_investmentMonths + c_investmentMonths
  let a_share := (a_investmentMonths / total_investmentMonths) * total_gain
  a_share = 6200 :=
by
  sorry

end NUMINAMATH_GPT_calculate_A_share_l2239_223904


namespace NUMINAMATH_GPT_min_value_is_8_plus_4_sqrt_3_l2239_223993

noncomputable def min_value_of_expression (a b : ℝ) : ℝ :=
  2 / a + 1 / b

theorem min_value_is_8_plus_4_sqrt_3 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a + 2 * b = 1) :
  min_value_of_expression a b = 8 + 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_is_8_plus_4_sqrt_3_l2239_223993


namespace NUMINAMATH_GPT_bodhi_yacht_animals_l2239_223921

def total_animals (cows foxes zebras sheep : ℕ) : ℕ :=
  cows + foxes + zebras + sheep

theorem bodhi_yacht_animals :
  ∀ (cows foxes sheep : ℕ), foxes = 15 → cows = 20 → sheep = 20 → total_animals cows foxes (3 * foxes) sheep = 100 :=
by
  intros cows foxes sheep h1 h2 h3
  rw [h1, h2, h3]
  show total_animals 20 15 (3 * 15) 20 = 100
  sorry

end NUMINAMATH_GPT_bodhi_yacht_animals_l2239_223921


namespace NUMINAMATH_GPT_pqrs_product_l2239_223998

noncomputable def P : ℝ := Real.sqrt 2012 + Real.sqrt 2013
noncomputable def Q : ℝ := -Real.sqrt 2012 - Real.sqrt 2013
noncomputable def R : ℝ := Real.sqrt 2012 - Real.sqrt 2013
noncomputable def S : ℝ := Real.sqrt 2013 - Real.sqrt 2012

theorem pqrs_product : P * Q * R * S = 1 := 
by 
  sorry

end NUMINAMATH_GPT_pqrs_product_l2239_223998


namespace NUMINAMATH_GPT_factorize_expression_l2239_223978

theorem factorize_expression (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l2239_223978


namespace NUMINAMATH_GPT_students_who_like_yellow_l2239_223972

theorem students_who_like_yellow (total_students girls students_like_green girls_like_pink students_like_yellow : ℕ)
  (h1 : total_students = 30)
  (h2 : students_like_green = total_students / 2)
  (h3 : girls_like_pink = girls / 3)
  (h4 : girls = 18)
  (h5 : students_like_yellow = total_students - (students_like_green + girls_like_pink)) :
  students_like_yellow = 9 :=
by
  sorry

end NUMINAMATH_GPT_students_who_like_yellow_l2239_223972


namespace NUMINAMATH_GPT_probability_meeting_proof_l2239_223913

noncomputable def probability_meeting (arrival_time_paul arrival_time_caroline : ℝ) : Prop :=
  arrival_time_paul ≤ arrival_time_caroline + 1 / 4 ∧ arrival_time_paul ≥ arrival_time_caroline - 1 / 4

theorem probability_meeting_proof :
  ∀ (arrival_time_paul arrival_time_caroline : ℝ)
    (h_paul_range : 0 ≤ arrival_time_paul ∧ arrival_time_paul ≤ 1)
    (h_caroline_range: 0 ≤ arrival_time_caroline ∧ arrival_time_caroline ≤ 1),
  (probability_meeting arrival_time_paul arrival_time_caroline) → 
  ∃ p, p = 7/16 :=
by
  sorry

end NUMINAMATH_GPT_probability_meeting_proof_l2239_223913


namespace NUMINAMATH_GPT_log_sum_correct_l2239_223969

noncomputable def log_sum : ℝ := 
  Real.log 8 / Real.log 10 + 
  3 * Real.log 4 / Real.log 10 + 
  4 * Real.log 2 / Real.log 10 +
  2 * Real.log 5 / Real.log 10 +
  5 * Real.log 25 / Real.log 10

theorem log_sum_correct : abs (log_sum - 12.301) < 0.001 :=
by sorry

end NUMINAMATH_GPT_log_sum_correct_l2239_223969


namespace NUMINAMATH_GPT_power_difference_divisible_by_35_l2239_223958

theorem power_difference_divisible_by_35 (n : ℕ) : (3^(6*n) - 2^(6*n)) % 35 = 0 := 
by sorry

end NUMINAMATH_GPT_power_difference_divisible_by_35_l2239_223958


namespace NUMINAMATH_GPT_word_limit_correct_l2239_223991

-- Definition for the conditions
def saturday_words : ℕ := 450
def sunday_words : ℕ := 650
def exceeded_amount : ℕ := 100

-- The total words written
def total_words : ℕ := saturday_words + sunday_words

-- The word limit which we need to prove
def word_limit : ℕ := total_words - exceeded_amount

theorem word_limit_correct : word_limit = 1000 := by
  unfold word_limit total_words saturday_words sunday_words exceeded_amount
  sorry

end NUMINAMATH_GPT_word_limit_correct_l2239_223991


namespace NUMINAMATH_GPT_part_a_l2239_223929

theorem part_a (a x y : ℕ) (h_a_pos : a > 0) (h_x_pos : x > 0) (h_y_pos : y > 0) (h_neq : x ≠ y) :
  (a * x + Nat.gcd a x + Nat.lcm a x) ≠ (a * y + Nat.gcd a y + Nat.lcm a y) := sorry

end NUMINAMATH_GPT_part_a_l2239_223929


namespace NUMINAMATH_GPT_hydrogen_atomic_weight_is_correct_l2239_223907

-- Definitions and assumptions based on conditions
def molecular_weight : ℝ := 68
def number_of_hydrogen_atoms : ℕ := 1
def number_of_chlorine_atoms : ℕ := 1
def number_of_oxygen_atoms : ℕ := 2
def atomic_weight_chlorine : ℝ := 35.45
def atomic_weight_oxygen : ℝ := 16.00

-- Definition for the atomic weight of hydrogen to be proved
def atomic_weight_hydrogen (w : ℝ) : Prop :=
  w * number_of_hydrogen_atoms
  + atomic_weight_chlorine * number_of_chlorine_atoms
  + atomic_weight_oxygen * number_of_oxygen_atoms = molecular_weight

-- The theorem to prove the atomic weight of hydrogen
theorem hydrogen_atomic_weight_is_correct : atomic_weight_hydrogen 1.008 :=
by
  unfold atomic_weight_hydrogen
  simp
  sorry

end NUMINAMATH_GPT_hydrogen_atomic_weight_is_correct_l2239_223907


namespace NUMINAMATH_GPT_Kim_morning_routine_time_l2239_223900

theorem Kim_morning_routine_time :
  let senior_employees := 3
  let junior_employees := 3
  let interns := 3

  let senior_overtime := 2
  let junior_overtime := 3
  let intern_overtime := 1
  let senior_not_overtime := senior_employees - senior_overtime
  let junior_not_overtime := junior_employees - junior_overtime
  let intern_not_overtime := interns - intern_overtime

  let coffee_time := 5
  let email_time := 10
  let supplies_time := 8
  let meetings_time := 6
  let reports_time := 5

  let status_update_time := 3 * senior_employees + 2 * junior_employees + 1 * interns
  let payroll_update_time := 
    4 * senior_overtime + 2 * senior_not_overtime +
    3 * junior_overtime + 1 * junior_not_overtime +
    2 * intern_overtime + 0.5 * intern_not_overtime
  let daily_tasks_time :=
    4 * senior_employees + 3 * junior_employees + 2 * interns

  let total_time := coffee_time + status_update_time + payroll_update_time + daily_tasks_time + email_time + supplies_time + meetings_time + reports_time
  total_time = 101 := by
  sorry

end NUMINAMATH_GPT_Kim_morning_routine_time_l2239_223900


namespace NUMINAMATH_GPT_sin_double_angle_cos_condition_l2239_223943

theorem sin_double_angle_cos_condition (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) :
  Real.sin (2 * x) = -7 / 25 :=
sorry

end NUMINAMATH_GPT_sin_double_angle_cos_condition_l2239_223943


namespace NUMINAMATH_GPT_total_baseball_cards_l2239_223935
-- Import the broad Mathlib library

-- The conditions stating the number of cards each person has
def melanie_cards : ℕ := 3
def benny_cards : ℕ := 3
def sally_cards : ℕ := 3
def jessica_cards : ℕ := 3

-- The theorem to prove the total number of cards they have is 12
theorem total_baseball_cards : melanie_cards + benny_cards + sally_cards + jessica_cards = 12 := by
  sorry

end NUMINAMATH_GPT_total_baseball_cards_l2239_223935


namespace NUMINAMATH_GPT_diamond_4_3_l2239_223971

def diamond (a b : ℤ) : ℤ := 4 * a + 3 * b - 2 * a * b

theorem diamond_4_3 : diamond 4 3 = 1 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_diamond_4_3_l2239_223971


namespace NUMINAMATH_GPT_solve_equation_l2239_223982

theorem solve_equation (x : ℝ) : 2 * (x - 2)^2 = 6 - 3 * x ↔ (x = 2 ∨ x = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2239_223982


namespace NUMINAMATH_GPT_probability_correct_l2239_223962

-- Definition for the total number of ways to select topics
def total_ways : ℕ := 6 * 6

-- Definition for the number of ways two students select different topics
def different_topics_ways : ℕ := 6 * 5

-- Definition for the probability of selecting different topics
def probability_different_topics : ℚ := different_topics_ways / total_ways

-- The statement to be proved in Lean
theorem probability_correct :
  probability_different_topics = 5 / 6 := 
sorry

end NUMINAMATH_GPT_probability_correct_l2239_223962


namespace NUMINAMATH_GPT_min_area_triangle_ABC_l2239_223905

theorem min_area_triangle_ABC :
  let A := (0, 0) 
  let B := (42, 18)
  (∃ p q : ℤ, let C := (p, q) 
              ∃ area : ℝ, area = (1 / 2 : ℝ) * |42 * q - 18 * p| 
              ∧ area = 3) := 
sorry

end NUMINAMATH_GPT_min_area_triangle_ABC_l2239_223905


namespace NUMINAMATH_GPT_chess_games_total_l2239_223923

-- Conditions
def crowns_per_win : ℕ := 8
def uncle_wins : ℕ := 4
def draws : ℕ := 5
def father_net_gain : ℤ := 24

-- Let total_games be the total number of games played
def total_games : ℕ := sorry

-- Proof that under the given conditions, total_games equals 16
theorem chess_games_total :
  total_games = uncle_wins + (father_net_gain + uncle_wins * crowns_per_win) / crowns_per_win + draws := by
  sorry

end NUMINAMATH_GPT_chess_games_total_l2239_223923


namespace NUMINAMATH_GPT_natives_cannot_obtain_910_rupees_with_50_coins_l2239_223955

theorem natives_cannot_obtain_910_rupees_with_50_coins (x y z : ℤ) : 
  x + y + z = 50 → 
  10 * x + 34 * y + 62 * z = 910 → 
  false :=
by
  sorry

end NUMINAMATH_GPT_natives_cannot_obtain_910_rupees_with_50_coins_l2239_223955


namespace NUMINAMATH_GPT_initial_violet_balloons_l2239_223908

-- Define initial conditions and variables
def red_balloons := 4
def violet_balloons_lost := 3
def current_violet_balloons := 4

-- Define the theorem we want to prove
theorem initial_violet_balloons (red_balloons : ℕ) (violet_balloons_lost : ℕ) (current_violet_balloons : ℕ) : 
  red_balloons = 4 → violet_balloons_lost = 3 → current_violet_balloons = 4 → (current_violet_balloons + violet_balloons_lost) = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_violet_balloons_l2239_223908


namespace NUMINAMATH_GPT_integer_solutions_inequality_system_l2239_223959

theorem integer_solutions_inequality_system :
  {x : ℤ | (x + 2 > 0) ∧ (2 * x - 1 ≤ 0)} = {-1, 0} := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_integer_solutions_inequality_system_l2239_223959


namespace NUMINAMATH_GPT_find_g9_l2239_223985

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g3_value : g 3 = 4

theorem find_g9 : g 9 = 64 := sorry

end NUMINAMATH_GPT_find_g9_l2239_223985


namespace NUMINAMATH_GPT_find_cost_prices_l2239_223919

noncomputable def cost_price_per_meter
  (selling_price_per_meter : ℕ) (loss_per_meter : ℕ) : ℕ :=
  selling_price_per_meter + loss_per_meter

theorem find_cost_prices
  (selling_A : ℕ) (meters_A : ℕ) (loss_A : ℕ)
  (selling_B : ℕ) (meters_B : ℕ) (loss_B : ℕ)
  (selling_C : ℕ) (meters_C : ℕ) (loss_C : ℕ)
  (H_A : selling_A = 9000) (H_meters_A : meters_A = 300) (H_loss_A : loss_A = 6)
  (H_B : selling_B = 7000) (H_meters_B : meters_B = 250) (H_loss_B : loss_B = 4)
  (H_C : selling_C = 12000) (H_meters_C : meters_C = 400) (H_loss_C : loss_C = 8) :
  cost_price_per_meter (selling_A / meters_A) loss_A = 36 ∧
  cost_price_per_meter (selling_B / meters_B) loss_B = 32 ∧
  cost_price_per_meter (selling_C / meters_C) loss_C = 38 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_cost_prices_l2239_223919


namespace NUMINAMATH_GPT_inscribed_quadrilateral_inradius_l2239_223960

noncomputable def calculate_inradius (a b c d: ℝ) (A: ℝ) : ℝ := (A / ((a + c + b + d) / 2))

theorem inscribed_quadrilateral_inradius {a b c d: ℝ} (h1: a + c = 10) (h2: b + d = 10) (h3: a + b + c + d = 20) (hA: 12 = 12):
  calculate_inradius a b c d 12 = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_quadrilateral_inradius_l2239_223960


namespace NUMINAMATH_GPT_sum_center_radius_eq_neg2_l2239_223941

theorem sum_center_radius_eq_neg2 (c d s : ℝ) (h_eq : ∀ x y : ℝ, x^2 + 14 * x + y^2 - 8 * y = -64 ↔ (x + c)^2 + (y + d)^2 = s^2) :
  c + d + s = -2 :=
sorry

end NUMINAMATH_GPT_sum_center_radius_eq_neg2_l2239_223941


namespace NUMINAMATH_GPT_amgm_inequality_proof_l2239_223926

noncomputable def amgm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : Prop :=
  1 < (a / (Real.sqrt (a^2 + b^2))) + (b / (Real.sqrt (b^2 + c^2))) + (c / (Real.sqrt (c^2 + a^2))) 
  ∧ (a / (Real.sqrt (a^2 + b^2))) + (b / (Real.sqrt (b^2 + c^2))) + (c / (Real.sqrt (c^2 + a^2))) 
  ≤ (3 * Real.sqrt 2) / 2

theorem amgm_inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  amgm_inequality a b c ha hb hc := 
sorry

end NUMINAMATH_GPT_amgm_inequality_proof_l2239_223926


namespace NUMINAMATH_GPT_arun_brother_weight_upper_limit_l2239_223999

theorem arun_brother_weight_upper_limit (w : ℝ) (X : ℝ) 
  (h1 : 61 < w ∧ w < 72)
  (h2 : 60 < w ∧ w < X)
  (h3 : w ≤ 64)
  (h4 : ((62 + 63 + 64) / 3) = 63) :
  X = 64 :=
by
  sorry

end NUMINAMATH_GPT_arun_brother_weight_upper_limit_l2239_223999


namespace NUMINAMATH_GPT_inequality_solution_set_l2239_223956

theorem inequality_solution_set (a b : ℝ) (h1 : a = -2) (h2 : b = 1) :
  {x : ℝ | |2 * x + a| + |x - b| < 6} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l2239_223956


namespace NUMINAMATH_GPT_factorize_expression_l2239_223964

variable (a b : ℝ)

theorem factorize_expression : (a - b)^2 + 6 * (b - a) + 9 = (a - b - 3)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2239_223964


namespace NUMINAMATH_GPT_cone_base_radius_l2239_223933

variable (s : ℝ) (A : ℝ) (r : ℝ)

theorem cone_base_radius (h1 : s = 5) (h2 : A = 15 * Real.pi) : r = 3 :=
by
  sorry

end NUMINAMATH_GPT_cone_base_radius_l2239_223933


namespace NUMINAMATH_GPT_domain_of_function_l2239_223902

theorem domain_of_function :
  {x : ℝ | x + 3 ≥ 0 ∧ x + 2 ≠ 0} = {x : ℝ | x ≥ -3 ∧ x ≠ -2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l2239_223902


namespace NUMINAMATH_GPT_line_equation_intercept_twice_x_intercept_l2239_223953

theorem line_equation_intercept_twice_x_intercept 
  {x y : ℝ}
  (intersection_point : ∃ (x y : ℝ), 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0) 
  (y_intercept_is_twice_x_intercept : ∃ (a : ℝ), ∀ (x y : ℝ), y = 2 * a ∧ x = a) :
  (∃ (x y : ℝ), 2 * x - 3 * y = 0) ∨ (∃ (x y : ℝ), 2 * x + y - 8 = 0) :=
sorry

end NUMINAMATH_GPT_line_equation_intercept_twice_x_intercept_l2239_223953


namespace NUMINAMATH_GPT_part1_tangent_circles_part2_chords_l2239_223954

theorem part1_tangent_circles (t : ℝ) : 
  t = 1 → 
  ∃ (a b : ℝ), 
    (x + 1)^2 + y^2 = 1 ∨ 
    (x + (2/5))^2 + (y - (9/5))^2 = (1 : ℝ) :=
by
  sorry

theorem part2_chords (t : ℝ) : 
  (∀ (k1 k2 : ℝ), 
    k1 + k2 = -3 * t / 4 ∧ 
    k1 * k2 = (t^2 - 1) / 8 ∧ 
    |k1 - k2| = 3 / 4) → 
    t = 1 ∨ t = -1 :=
by
  sorry

end NUMINAMATH_GPT_part1_tangent_circles_part2_chords_l2239_223954


namespace NUMINAMATH_GPT_circle_equation_l2239_223979

-- Definitions of the conditions
def passes_through (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (r : ℝ) : Prop :=
  (c - a) ^ 2 + (d - b) ^ 2 = r ^ 2

def center_on_line (a : ℝ) (b : ℝ) : Prop :=
  a - b - 4 = 0

-- Statement of the problem to be proved
theorem circle_equation 
  (a b r : ℝ) 
  (h1 : passes_through a b (-1) (-4) r)
  (h2 : passes_through a b 6 3 r)
  (h3 : center_on_line a b) :
  -- Equation of the circle
  (a = 3 ∧ b = -1 ∧ r = 5) → ∀ x y : ℝ, 
    (x - 3)^2 + (y + 1)^2 = 25 :=
sorry

end NUMINAMATH_GPT_circle_equation_l2239_223979


namespace NUMINAMATH_GPT_fraction_equiv_subtract_l2239_223915

theorem fraction_equiv_subtract (n : ℚ) : (4 - n) / (7 - n) = 3 / 5 → n = 0.5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_fraction_equiv_subtract_l2239_223915
