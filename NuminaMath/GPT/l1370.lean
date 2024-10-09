import Mathlib

namespace main_theorem_l1370_137032

noncomputable def main_expr := (Real.pi - 2019) ^ 0 + |Real.sqrt 3 - 1| + (-1 / 2)⁻¹ - 2 * Real.tan (Real.pi / 6)

theorem main_theorem : main_expr = -2 + Real.sqrt 3 / 3 := by
  sorry

end main_theorem_l1370_137032


namespace evaluate_expression_l1370_137022

theorem evaluate_expression : (25 + 15)^2 - (25^2 + 15^2 + 150) = 600 := by
  sorry

end evaluate_expression_l1370_137022


namespace range_of_independent_variable_l1370_137047

theorem range_of_independent_variable (x : ℝ) (h : ∃ y, y = 2 / (Real.sqrt (x - 3))) : x > 3 :=
sorry

end range_of_independent_variable_l1370_137047


namespace value_of_z_l1370_137057

theorem value_of_z (x y z : ℤ) (h1 : x^2 = y - 4) (h2 : x = -6) (h3 : y = z + 2) : z = 38 := 
by
  -- Proof skipped
  sorry

end value_of_z_l1370_137057


namespace area_of_triangle_QCA_l1370_137010

noncomputable def triangle_area (x p : ℝ) (hx : x > 0) (hp : p < 12) : ℝ :=
  1 / 2 * x * (12 - p)

theorem area_of_triangle_QCA (x p : ℝ) (hx : x > 0) (hp : p < 12) :
  triangle_area x p hx hp = x * (12 - p) / 2 := by
  sorry

end area_of_triangle_QCA_l1370_137010


namespace opposites_of_each_other_l1370_137075

theorem opposites_of_each_other (a b : ℚ) (h : a + b = 0) : a = -b :=
  sorry

end opposites_of_each_other_l1370_137075


namespace arithmetic_progression_integers_l1370_137030

theorem arithmetic_progression_integers 
  (d : ℤ) (a : ℤ) (h_d_pos : d > 0)
  (h_progression : ∀ i j : ℤ, i ≠ j → ∃ k : ℤ, a * (a + i * d) = a + k * d)
  : ∀ n : ℤ, ∃ m : ℤ, a + n * d = m :=
by
  sorry

end arithmetic_progression_integers_l1370_137030


namespace add_congruence_l1370_137082

variable (a b c d m : ℤ)

theorem add_congruence (h₁ : a ≡ b [ZMOD m]) (h₂ : c ≡ d [ZMOD m]) : (a + c) ≡ (b + d) [ZMOD m] :=
sorry

end add_congruence_l1370_137082


namespace find_b_in_expression_l1370_137003

theorem find_b_in_expression
  (a b : ℚ)
  (h : (1 + Real.sqrt 3)^5 = a + b * Real.sqrt 3) :
  b = 44 :=
sorry

end find_b_in_expression_l1370_137003


namespace find_coordinates_A_l1370_137049

-- Define the point A
structure Point where
  x : ℝ
  y : ℝ

def PointA (a : ℝ) : Point :=
  { x := 3 * a + 2, y := 2 * a - 4 }

-- Define the conditions
def condition1 (a : ℝ) := (PointA a).y = 4

def condition2 (a : ℝ) := |(PointA a).x| = |(PointA a).y|

-- The coordinates solutions to be proven
def valid_coordinates (p : Point) : Prop :=
  p = { x := 14, y := 4 } ∨
  p = { x := -16, y := -16 } ∨
  p = { x := 3.2, y := -3.2 }

-- Main theorem to prove
theorem find_coordinates_A (a : ℝ) :
  (condition1 a ∨ condition2 a) → valid_coordinates (PointA a) :=
by
  sorry

end find_coordinates_A_l1370_137049


namespace amanda_weekly_earnings_l1370_137073

def amanda_rate_per_hour : ℝ := 20.00
def monday_appointments : ℕ := 5
def monday_hours_per_appointment : ℝ := 1.5
def tuesday_appointment_hours : ℝ := 3
def thursday_appointments : ℕ := 2
def thursday_hours_per_appointment : ℝ := 2
def saturday_appointment_hours : ℝ := 6

def total_hours_worked : ℝ :=
  monday_appointments * monday_hours_per_appointment +
  tuesday_appointment_hours +
  thursday_appointments * thursday_hours_per_appointment +
  saturday_appointment_hours

def total_earnings : ℝ := total_hours_worked * amanda_rate_per_hour

theorem amanda_weekly_earnings : total_earnings = 410.00 :=
  by
    unfold total_earnings total_hours_worked monday_appointments monday_hours_per_appointment tuesday_appointment_hours thursday_appointments thursday_hours_per_appointment saturday_appointment_hours amanda_rate_per_hour 
    -- The proof will involve basic arithmetic simplification, which is skipped here.
    -- Therefore, we simply state sorry.
    sorry

end amanda_weekly_earnings_l1370_137073


namespace sum_inequality_l1370_137072

variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {m n p k : ℕ}

-- Definitions for the conditions given in the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i j, a (i + 1) - a i = a (j + 1) - a j

def sum_of_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (a 1 + a (n - 1)) / 2

def non_negative_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≥ 0

-- The theorem to prove
theorem sum_inequality (arith_seq : is_arithmetic_sequence a)
  (S_eq : sum_of_arithmetic_sequence S a)
  (nn_seq : non_negative_sequence a)
  (h1 : m + n = 2 * p) (m_pos : 0 < m) (n_pos : 0 < n) (p_pos : 0 < p) :
  1 / (S m) ^ k + 1 / (S n) ^ k ≥ 2 / (S p) ^ k :=
by sorry

end sum_inequality_l1370_137072


namespace no_polyhedron_with_surface_area_2015_l1370_137011

theorem no_polyhedron_with_surface_area_2015 : 
  ¬ ∃ (n k : ℤ), 6 * n - 2 * k = 2015 :=
by
  sorry

end no_polyhedron_with_surface_area_2015_l1370_137011


namespace quadratic_has_one_positive_and_one_negative_root_l1370_137005

theorem quadratic_has_one_positive_and_one_negative_root
  (a : ℝ) (h₁ : a ≠ 0) (h₂ : a < -1) :
  ∃ x₁ x₂ : ℝ, (a * x₁^2 + 2 * x₁ + 1 = 0) ∧ (a * x₂^2 + 2 * x₂ + 1 = 0) ∧ (x₁ > 0) ∧ (x₂ < 0) :=
by
  sorry

end quadratic_has_one_positive_and_one_negative_root_l1370_137005


namespace jordan_trapezoid_height_l1370_137040

def rectangle_area (length width : ℕ) : ℕ :=
  length * width

def trapezoid_area (base1 base2 height : ℕ) : ℕ :=
  (base1 + base2) * height / 2

theorem jordan_trapezoid_height :
  ∀ (h : ℕ),
    rectangle_area 5 24 = trapezoid_area 2 6 h →
    h = 30 :=
by
  intro h
  intro h_eq
  sorry

end jordan_trapezoid_height_l1370_137040


namespace inning_is_31_l1370_137055

noncomputable def inning_number (s: ℕ) (i: ℕ) (a: ℕ) : ℕ := s - a + i

theorem inning_is_31
  (batsman_runs: ℕ)
  (increase_average: ℕ)
  (final_average: ℕ) 
  (n: ℕ) 
  (h1: batsman_runs = 92)
  (h2: increase_average = 3)
  (h3: final_average = 44)
  (h4: 44 * n - 92 = 41 * n): 
  inning_number 44 1 3 = 31 := 
by 
  sorry

end inning_is_31_l1370_137055


namespace smallest_number_in_set_l1370_137085

open Real

theorem smallest_number_in_set :
  ∀ (a b c d : ℝ), a = -3 → b = 3⁻¹ → c = -abs (-1 / 3) → d = 0 →
    a < b ∧ a < c ∧ a < d :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end smallest_number_in_set_l1370_137085


namespace find_all_triples_l1370_137090

def satisfying_triples (a b c : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 
  (a^2 + a*b = c) ∧ 
  (b^2 + b*c = a) ∧ 
  (c^2 + c*a = b)

theorem find_all_triples (a b c : ℝ) : satisfying_triples a b c ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = 1/2 ∧ b = 1/2 ∧ c = 1/2) :=
by sorry

end find_all_triples_l1370_137090


namespace score_ordering_l1370_137013

-- Definition of the problem conditions in Lean 4:
def condition1 (Q K : ℝ) : Prop := Q ≠ K
def condition2 (M Q S K : ℝ) : Prop := M < Q ∧ M < S ∧ M < K
def condition3 (S Q M K : ℝ) : Prop := S > Q ∧ S > M ∧ S > K

-- Theorem statement in Lean 4:
theorem score_ordering (M Q S K : ℝ) (h1 : condition1 Q K) (h2 : condition2 M Q S K) (h3 : condition3 S Q M K) : 
  M < Q ∧ Q < S :=
by
  sorry

end score_ordering_l1370_137013


namespace max_mondays_in_first_51_days_l1370_137059

theorem max_mondays_in_first_51_days (start_on_sunday_or_monday : Bool) :
  ∃ (n : ℕ), n = 8 ∧ (∀ weeks_days: ℕ, weeks_days = 51 → (∃ mondays: ℕ,
    mondays <= 8 ∧ mondays >= (weeks_days / 7 + if start_on_sunday_or_monday then 1 else 0))) :=
by {
  sorry -- the proof will go here
}

end max_mondays_in_first_51_days_l1370_137059


namespace train_avg_speed_without_stoppages_l1370_137091

/-- A train with stoppages has an average speed of 125 km/h. Given that the train stops for 30 minutes per hour,
the average speed of the train without stoppages is 250 km/h. -/
theorem train_avg_speed_without_stoppages (avg_speed_with_stoppages : ℝ) 
  (stoppage_time_per_hour : ℝ) (no_stoppage_speed : ℝ) 
  (h1 : avg_speed_with_stoppages = 125) (h2 : stoppage_time_per_hour = 0.5) : 
  no_stoppage_speed = 250 :=
sorry

end train_avg_speed_without_stoppages_l1370_137091


namespace inequality_holds_l1370_137048

theorem inequality_holds (x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℝ) 
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4)
  (h5 : 0 < x5) (h6 : 0 < x6) (h7 : 0 < x7) (h8 : 0 < x8) 
  (h9 : 0 < x9) :
  (x1 - x3) / (x1 * x3 + 2 * x2 * x3 + x2^2) +
  (x2 - x4) / (x2 * x4 + 2 * x3 * x4 + x3^2) +
  (x3 - x5) / (x3 * x5 + 2 * x4 * x5 + x4^2) +
  (x4 - x6) / (x4 * x6 + 2 * x5 * x6 + x5^2) +
  (x5 - x7) / (x5 * x7 + 2 * x6 * x7 + x6^2) +
  (x6 - x8) / (x6 * x8 + 2 * x7 * x8 + x7^2) +
  (x7 - x9) / (x7 * x9 + 2 * x8 * x9 + x8^2) +
  (x8 - x1) / (x8 * x1 + 2 * x9 * x1 + x9^2) +
  (x9 - x2) / (x9 * x2 + 2 * x1 * x2 + x1^2) ≥ 0 := 
sorry

end inequality_holds_l1370_137048


namespace farey_neighbors_of_half_l1370_137077

noncomputable def farey_neighbors (n : ℕ) : List (ℚ) :=
  if n % 2 = 1 then
    [ (n - 1 : ℚ) / (2 * n), (n + 1 : ℚ) / (2 * n) ]
  else
    [ (n - 2 : ℚ) / (2 * (n - 1)), n / (2 * (n - 1)) ]

theorem farey_neighbors_of_half (n : ℕ) (hn : 0 < n) : 
  ∃ a b : ℚ, a ∈ farey_neighbors n ∧ b ∈ farey_neighbors n ∧ 
    (n % 2 = 1 → a = (n - 1 : ℚ) / (2 * n) ∧ b = (n + 1 : ℚ) / (2 * n)) ∧
    (n % 2 = 0 → a = (n - 2 : ℚ) / (2 * (n - 1)) ∧ b = n / (2 * (n - 1))) :=
sorry

end farey_neighbors_of_half_l1370_137077


namespace percentage_solution_l1370_137066

noncomputable def percentage_of_difference (P : ℚ) (x y : ℚ) : Prop :=
  (P / 100) * (x - y) = (14 / 100) * (x + y)

theorem percentage_solution (x y : ℚ) (h1 : y = 0.17647058823529413 * x)
  (h2 : percentage_of_difference P x y) : 
  P = 20 := 
by
  sorry

end percentage_solution_l1370_137066


namespace number_of_divisors_8_factorial_l1370_137038

open Nat

theorem number_of_divisors_8_factorial :
  let n := 8!
  let factorization := [(2, 7), (3, 2), (5, 1), (7, 1)]
  let numberOfDivisors := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  n = 2^7 * 3^2 * 5^1 * 7^1 ->
  n.factors.count = 4 ->
  numberOfDivisors = 96 :=
by
  sorry

end number_of_divisors_8_factorial_l1370_137038


namespace number_of_extreme_points_zero_l1370_137095

def f (x a : ℝ) : ℝ := x^3 + 3*x^2 + 4*x - a

theorem number_of_extreme_points_zero (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ∀ x, f x1 a = f x a → x = x1 ∨ x = x2) → False := 
by
  sorry

end number_of_extreme_points_zero_l1370_137095


namespace earnings_per_visit_l1370_137027

-- Define the conditions of the problem
def website_visits_per_month : ℕ := 30000
def earning_per_day : Real := 10
def days_in_month : ℕ := 30

-- Prove that John gets $0.01 per visit
theorem earnings_per_visit :
  (earning_per_day * days_in_month) / website_visits_per_month = 0.01 :=
by
  sorry

end earnings_per_visit_l1370_137027


namespace probability_ge_first_second_l1370_137052

noncomputable def probability_ge_rolls : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1
  favorable_outcomes / total_outcomes

theorem probability_ge_first_second :
  probability_ge_rolls = 9 / 16 :=
by
  sorry

end probability_ge_first_second_l1370_137052


namespace watermelon_melon_weight_l1370_137012

variables {W M : ℝ}

theorem watermelon_melon_weight :
  (2 * W > 3 * M ∨ 3 * W > 4 * M) ∧ ¬ (2 * W > 3 * M ∧ 3 * W > 4 * M) → 12 * W ≤ 18 * M :=
by
  sorry

end watermelon_melon_weight_l1370_137012


namespace abs_diff_squares_eq_300_l1370_137019

theorem abs_diff_squares_eq_300 : 
  let a := (103 : ℚ) / 2 
  let b := (97 : ℚ) / 2
  |a^2 - b^2| = 300 := 
by
  let a := (103 : ℚ) / 2 
  let b := (97 : ℚ) / 2
  sorry

end abs_diff_squares_eq_300_l1370_137019


namespace find_union_A_B_r_find_range_m_l1370_137083

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def B (x m : ℝ) : Prop := (x - m) * (x - m - 1) ≥ 0

theorem find_union_A_B_r (x : ℝ) : A x ∨ B x 1 := by
  sorry

theorem find_range_m (m : ℝ) (x : ℝ) : (∀ x, A x ↔ B x m) ↔ (m ≥ 3 ∨ m ≤ -2) := by
  sorry

end find_union_A_B_r_find_range_m_l1370_137083


namespace no_positive_ints_m_n_m_square_plus_2_equals_n_square_plus_n_k_ge_3_positive_ints_m_n_exists_l1370_137009

-- Proof Problem 1:
theorem no_positive_ints_m_n_m_square_plus_2_equals_n_square_plus_n :
  ¬ ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m * (m + 2) = n * (n + 1) :=
by sorry

-- Proof Problem 2:
theorem k_ge_3_positive_ints_m_n_exists (k : ℕ) (hk : k ≥ 3) :
  (k = 3 → ¬ ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) ∧
  (k ≥ 4 → ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) :=
by sorry

end no_positive_ints_m_n_m_square_plus_2_equals_n_square_plus_n_k_ge_3_positive_ints_m_n_exists_l1370_137009


namespace frac_equality_l1370_137058

variables (a b : ℚ) -- Declare the variables as rational numbers

-- State the theorem with the given condition and the proof goal
theorem frac_equality (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry -- proof goes here

end frac_equality_l1370_137058


namespace sqrt_0_54_in_terms_of_a_b_l1370_137098

variable (a b : ℝ)

-- Conditions
def sqrt_two_eq_a : Prop := a = Real.sqrt 2
def sqrt_three_eq_b : Prop := b = Real.sqrt 3

-- The main statement to prove
theorem sqrt_0_54_in_terms_of_a_b (h1 : sqrt_two_eq_a a) (h2 : sqrt_three_eq_b b) :
  Real.sqrt 0.54 = 0.3 * a * b := sorry

end sqrt_0_54_in_terms_of_a_b_l1370_137098


namespace weeks_to_buy_bicycle_l1370_137094

-- Definitions based on problem conditions
def hourly_wage : Int := 5
def hours_monday : Int := 2
def hours_wednesday : Int := 1
def hours_friday : Int := 3
def weekly_hours : Int := hours_monday + hours_wednesday + hours_friday
def weekly_earnings : Int := weekly_hours * hourly_wage
def bicycle_cost : Int := 180

-- Statement of the theorem to prove
theorem weeks_to_buy_bicycle : ∃ w : Nat, w * weekly_earnings = bicycle_cost :=
by
  -- Since this is a statement only, the proof is omitted
  sorry

end weeks_to_buy_bicycle_l1370_137094


namespace f_at_11_l1370_137067

def f (n : ℕ) : ℕ := n^2 + n + 17

theorem f_at_11 : f 11 = 149 := sorry

end f_at_11_l1370_137067


namespace sum_of_eight_numbers_on_cards_l1370_137044

theorem sum_of_eight_numbers_on_cards :
  ∃ (a b c d e f g h : ℕ),
  (a + b) * (c + d) * (e + f) * (g + h) = 330 ∧
  (a + b + c + d + e + f + g + h) = 21 :=
by
  sorry

end sum_of_eight_numbers_on_cards_l1370_137044


namespace equivalent_prop_l1370_137092

theorem equivalent_prop (x : ℝ) : (x > 1 → (x - 1) * (x + 3) > 0) ↔ ((x - 1) * (x + 3) ≤ 0 → x ≤ 1) :=
sorry

end equivalent_prop_l1370_137092


namespace arithmetic_geometric_mean_inequality_l1370_137065

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end arithmetic_geometric_mean_inequality_l1370_137065


namespace sum_of_first_4n_integers_l1370_137054

theorem sum_of_first_4n_integers (n : ℕ) 
  (h : (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 150) : 
  (4 * n * (4 * n + 1)) / 2 = 300 :=
by
  sorry

end sum_of_first_4n_integers_l1370_137054


namespace inverse_proportion_symmetry_l1370_137014

theorem inverse_proportion_symmetry (a b : ℝ) :
  (b = - 6 / (-a)) → (-b = - 6 / a) :=
by
  intro h
  sorry

end inverse_proportion_symmetry_l1370_137014


namespace ceil_of_neg_sqrt_frac_64_over_9_l1370_137000

theorem ceil_of_neg_sqrt_frac_64_over_9 :
  ⌈-Real.sqrt (64 / 9)⌉ = -2 :=
by
  sorry

end ceil_of_neg_sqrt_frac_64_over_9_l1370_137000


namespace largest_four_digit_divisible_by_8_l1370_137070

/-- The largest four-digit number that is divisible by 8 is 9992. -/
theorem largest_four_digit_divisible_by_8 : ∃ x : ℕ, x = 9992 ∧ x < 10000 ∧ x % 8 = 0 ∧
  ∀ y : ℕ, y < 10000 ∧ y % 8 = 0 → y ≤ 9992 := 
by 
  sorry

end largest_four_digit_divisible_by_8_l1370_137070


namespace total_seeds_eaten_l1370_137071

def first_seeds := 78
def second_seeds := 53
def third_seeds := second_seeds + 30

theorem total_seeds_eaten : first_seeds + second_seeds + third_seeds = 214 := by
  -- Sorry, placeholder for proof
  sorry

end total_seeds_eaten_l1370_137071


namespace not_prime_5n_plus_3_l1370_137099

def isSquare (x : ℕ) : Prop := ∃ (k : ℕ), k * k = x

theorem not_prime_5n_plus_3 (n k m : ℕ) (h₁ : 2 * n + 1 = k * k) (h₂ : 3 * n + 1 = m * m) (n_pos : 0 < n) (k_pos : 0 < k) (m_pos : 0 < m) :
  ¬ Nat.Prime (5 * n + 3) :=
sorry -- Proof to be completed

end not_prime_5n_plus_3_l1370_137099


namespace profit_percentage_l1370_137029

theorem profit_percentage (SP : ℝ) (h : SP > 0) (CP : ℝ) (h1 : CP = 0.96 * SP) :
  (SP - CP) / CP * 100 = 4.17 :=
by
  sorry

end profit_percentage_l1370_137029


namespace pump_B_time_l1370_137096

theorem pump_B_time (T_B : ℝ) (h1 : ∀ (h1 : T_B > 0),
  (1 / 4 + 1 / T_B = 3 / 4)) :
  T_B = 2 := 
by
  sorry

end pump_B_time_l1370_137096


namespace maximum_value_of_f_l1370_137039

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * Real.sqrt x

theorem maximum_value_of_f :
  ∃ x_max : ℝ, x_max > 0 ∧ (∀ x : ℝ, x > 0 → f x ≤ f x_max) ∧ f x_max = -2 :=
by
  sorry

end maximum_value_of_f_l1370_137039


namespace race_ordering_l1370_137045

theorem race_ordering
  (Lotar Manfred Jan Victor Eddy : ℕ) 
  (h1 : Lotar < Manfred) 
  (h2 : Manfred < Jan) 
  (h3 : Jan < Victor) 
  (h4 : Eddy < Victor) : 
  ∀ x, x = Victor ↔ ∀ y, (y = Lotar ∨ y = Manfred ∨ y = Jan ∨ y = Eddy) → y < x :=
by
  sorry

end race_ordering_l1370_137045


namespace solve_inequality_l1370_137036

theorem solve_inequality (x : ℝ) : (2 * x - 3) / (x + 2) ≤ 1 ↔ (-2 < x ∧ x ≤ 5) :=
  sorry

end solve_inequality_l1370_137036


namespace correct_statements_l1370_137088

-- Define the regression condition
def regression_condition (b : ℝ) : Prop := b < 0

-- Conditon ③: Event A is the complement of event B implies mutual exclusivity
def mutually_exclusive_and_complementary (A B : Prop) : Prop := 
  (A → ¬B) → (¬A ↔ B)

-- Main theorem combining the conditions and questions
theorem correct_statements: 
  (∀ b, regression_condition b ↔ (b < 0)) ∧
  (∀ A B, mutually_exclusive_and_complementary A B → (¬A ≠ B)) :=
by
  sorry

end correct_statements_l1370_137088


namespace kat_boxing_training_hours_l1370_137061

theorem kat_boxing_training_hours :
  let strength_training_hours := 3
  let total_training_hours := 9
  let boxing_sessions := 4
  let boxing_training_hours := total_training_hours - strength_training_hours
  let hours_per_boxing_session := boxing_training_hours / boxing_sessions
  hours_per_boxing_session = 1.5 :=
sorry

end kat_boxing_training_hours_l1370_137061


namespace two_digit_integers_count_l1370_137041

def digits : Set ℕ := {3, 5, 7, 8, 9}

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem two_digit_integers_count : 
  ∃ (count : ℕ), count = 16 ∧
  (∀ (t : ℕ), t ∈ digits → 
  ∀ (u : ℕ), u ∈ digits → 
  t ≠ u ∧ is_odd u → 
  (∃ n : ℕ, 10 * t + u = n)) :=
by
  -- The total number of unique two-digit integers is 16
  use 16
  -- Proof skipped
  sorry

end two_digit_integers_count_l1370_137041


namespace max_sequence_term_value_l1370_137033

def a_n (n : ℕ) : ℤ := -2 * n^2 + 29 * n + 3

theorem max_sequence_term_value : ∃ n : ℕ, a_n n = 108 := 
sorry

end max_sequence_term_value_l1370_137033


namespace percentage_of_500_l1370_137020

theorem percentage_of_500 (P : ℝ) : 0.1 * (500 * P / 100) = 25 → P = 50 :=
by
  sorry

end percentage_of_500_l1370_137020


namespace constant_term_expansion_l1370_137016

noncomputable def sum_of_coefficients (a : ℕ) : ℕ := sorry

noncomputable def constant_term (a : ℕ) : ℕ := sorry

theorem constant_term_expansion (a : ℕ) (h : sum_of_coefficients a = 2) : constant_term 2 = 10 :=
sorry

end constant_term_expansion_l1370_137016


namespace certain_event_l1370_137081

-- Define the conditions for the problem
def EventA : Prop := ∃ (seat_number : ℕ), seat_number % 2 = 1
def EventB : Prop := ∃ (shooter_hits : Prop), shooter_hits
def EventC : Prop := ∃ (broadcast_news : Prop), broadcast_news
def EventD : Prop := 
  ∀ (red_ball_count white_ball_count : ℕ), (red_ball_count = 2) ∧ (white_ball_count = 1) → 
  ∀ (draw_count : ℕ), (draw_count = 2) → 
  (∃ (red_ball_drawn : Prop), red_ball_drawn)

-- Define the main statement to prove EventD is the certain event
theorem certain_event : EventA → EventB → EventC → EventD
:= 
sorry

end certain_event_l1370_137081


namespace minimal_benches_l1370_137079

theorem minimal_benches (x : ℕ) 
  (standard_adults : ℕ := x * 8) (standard_children : ℕ := x * 12)
  (extended_adults : ℕ := x * 8) (extended_children : ℕ := x * 16) 
  (hx : standard_adults + extended_adults = standard_children + extended_children) :
  x = 1 :=
by
  sorry

end minimal_benches_l1370_137079


namespace largest_possible_gcd_l1370_137051

theorem largest_possible_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 221) : ∃ d, Nat.gcd a b = d ∧ d = 17 :=
sorry

end largest_possible_gcd_l1370_137051


namespace isosceles_triangle_angle_measure_l1370_137053

theorem isosceles_triangle_angle_measure
  (isosceles : Triangle → Prop)
  (exterior_angles : Triangle → ℝ → ℝ → Prop)
  (ratio_1_to_4 : ∀ {T : Triangle} {a b : ℝ}, exterior_angles T a b → b = 4 * a)
  (interior_angles : Triangle → ℝ → ℝ → ℝ → Prop) :
  ∀ (T : Triangle), isosceles T → ∃ α β γ : ℝ, interior_angles T α β γ ∧ α = 140 ∧ β = 20 ∧ γ = 20 := 
by
  sorry

end isosceles_triangle_angle_measure_l1370_137053


namespace six_digit_number_division_l1370_137008

theorem six_digit_number_division :
  ∃ a b p : ℕ, 
    (111111 * a = 1111 * b * 233 + p) ∧ 
    (11111 * a = 111 * b * 233 + p - 1000) ∧
    (111111 * 7 = 777777) ∧
    (1111 * 3 = 3333) :=
by
  sorry

end six_digit_number_division_l1370_137008


namespace find_positive_integer_solutions_l1370_137006

theorem find_positive_integer_solutions :
  ∃ (x y z : ℕ), 
    2 * x * z = y^2 ∧ 
    x + z = 1987 ∧ 
    x = 1458 ∧ 
    y = 1242 ∧ 
    z = 529 :=
  by sorry

end find_positive_integer_solutions_l1370_137006


namespace find_values_l1370_137068

open Real

noncomputable def positive_numbers (x y : ℝ) := x > 0 ∧ y > 0

noncomputable def given_condition (x y : ℝ) := (sqrt (12 * x) * sqrt (20 * x) * sqrt (4 * y) * sqrt (25 * y) = 50)

theorem find_values (x y : ℝ) 
  (h1: positive_numbers x y) 
  (h2: given_condition x y) : 
  x * y = sqrt (25 / 24) := 
sorry

end find_values_l1370_137068


namespace total_boxes_sold_l1370_137064

-- Define the variables for each day's sales
def friday_sales : ℕ := 30
def saturday_sales : ℕ := 2 * friday_sales
def sunday_sales : ℕ := saturday_sales - 15
def total_sales : ℕ := friday_sales + saturday_sales + sunday_sales

-- State the theorem to prove the total sales over three days
theorem total_boxes_sold : total_sales = 135 :=
by 
  -- Here we would normally put the proof steps, but since we're asked only for the statement,
  -- we skip the proof with sorry
  sorry

end total_boxes_sold_l1370_137064


namespace sum_first_five_terms_arithmetic_seq_l1370_137080

theorem sum_first_five_terms_arithmetic_seq
  (a : ℕ → ℕ)
  (h_arith_seq : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (h_a2 : a 2 = 5)
  (h_a4 : a 4 = 9)
  : (Finset.range 5).sum a = 35 := by
  sorry

end sum_first_five_terms_arithmetic_seq_l1370_137080


namespace a_range_l1370_137017

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

theorem a_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) ↔ (0 < a ∧ a ≤ 1 / 4) :=
by
  sorry

end a_range_l1370_137017


namespace ratio_of_pete_to_susan_l1370_137021

noncomputable def Pete_backward_speed := 12 -- in miles per hour
noncomputable def Pete_handstand_speed := 2 -- in miles per hour
noncomputable def Tracy_cartwheel_speed := 4 * Pete_handstand_speed -- in miles per hour
noncomputable def Susan_forward_speed := Tracy_cartwheel_speed / 2 -- in miles per hour

theorem ratio_of_pete_to_susan :
  Pete_backward_speed / Susan_forward_speed = 3 := 
sorry

end ratio_of_pete_to_susan_l1370_137021


namespace num_possible_n_l1370_137089

theorem num_possible_n (n : ℕ) : (∃ a b c : ℕ, 9 * a + 99 * b + 999 * c = 5000 ∧ n = a + 2 * b + 3 * c) ↔ n ∈ {x | x = a + 2 * b + 3 * c ∧ 0 ≤ 9 * (b + 12 * c) ∧ 9 * (b + 12 * c) ≤ 555} :=
sorry

end num_possible_n_l1370_137089


namespace women_in_first_group_l1370_137056

-- Define the number of women in the first group as W
variable (W : ℕ)

-- Define the work parameters
def work_per_day := 75 / 8
def work_per_hour_first_group := work_per_day / 5

def work_per_day_second_group := 30 / 3
def work_per_hour_second_group := work_per_day_second_group / 8

-- The equation comes from work/hour equivalence
theorem women_in_first_group :
  (W : ℝ) * work_per_hour_first_group = 4 * work_per_hour_second_group → W = 5 :=
by 
  sorry

end women_in_first_group_l1370_137056


namespace percentage_increase_second_movie_l1370_137004

def length_first_movie : ℕ := 2
def total_length_marathon : ℕ := 9
def length_last_movie (F S : ℕ) := S + F - 1

theorem percentage_increase_second_movie :
  ∀ (S : ℕ), 
  length_first_movie + S + length_last_movie length_first_movie S = total_length_marathon →
  ((S - length_first_movie) * 100) / length_first_movie = 50 :=
by
  sorry

end percentage_increase_second_movie_l1370_137004


namespace december_fraction_of_yearly_sales_l1370_137097

theorem december_fraction_of_yearly_sales (A : ℝ) (h_sales : ∀ (x : ℝ), x = 6 * A) :
    let yearly_sales := 11 * A + 6 * A
    let december_sales := 6 * A
    december_sales / yearly_sales = 6 / 17 := by
  sorry

end december_fraction_of_yearly_sales_l1370_137097


namespace candidate_lost_by_l1370_137063

noncomputable def candidate_votes (total_votes : ℝ) := 0.35 * total_votes
noncomputable def rival_votes (total_votes : ℝ) := 0.65 * total_votes

theorem candidate_lost_by (total_votes : ℝ) (h : total_votes = 7899.999999999999) :
  rival_votes total_votes - candidate_votes total_votes = 2370 :=
by
  sorry

end candidate_lost_by_l1370_137063


namespace trig_expression_value_l1370_137078

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 2) : 
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := 
by
  sorry

end trig_expression_value_l1370_137078


namespace benny_total_hours_l1370_137024

-- Define the conditions
def hours_per_day : ℕ := 3
def days_worked : ℕ := 6

-- State the theorem (problem) to be proved
theorem benny_total_hours : hours_per_day * days_worked = 18 :=
by
  -- Sorry to skip the actual proof
  sorry

end benny_total_hours_l1370_137024


namespace fewest_candies_l1370_137034

-- Defining the conditions
def condition1 (x : ℕ) := x % 21 = 5
def condition2 (x : ℕ) := x % 22 = 3
def condition3 (x : ℕ) := x > 500

-- Stating the main theorem
theorem fewest_candies : ∃ x : ℕ, condition1 x ∧ condition2 x ∧ condition3 x ∧ x = 509 :=
  sorry

end fewest_candies_l1370_137034


namespace problem_l1370_137076

theorem problem (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, x^5 = a_0 + a_1 * (1 - x) + a_2 * (1 - x)^2 + a_3 * (1 - x)^3 + a_4 * (1 - x)^4 + a_5 * (1 - x)^5) →
  a_3 = -10 ∧ a_1 + a_3 + a_5 = -16 :=
by 
  sorry

end problem_l1370_137076


namespace sum_real_imag_parts_eq_l1370_137025

noncomputable def z (a b : ℂ) : ℂ := a / b

theorem sum_real_imag_parts_eq (z : ℂ) (h : z * (2 + I) = 2 * I - 1) : 
  (z.re + z.im) = 1 / 5 :=
sorry

end sum_real_imag_parts_eq_l1370_137025


namespace reciprocal_of_2023_l1370_137031

theorem reciprocal_of_2023 :
  1 / 2023 = 1 / (2023 : ℝ) :=
by
  sorry

end reciprocal_of_2023_l1370_137031


namespace circle_standard_equation_l1370_137086

theorem circle_standard_equation (x y : ℝ) (h : (x + 1)^2 + (y - 2)^2 = 4) : 
  (x + 1)^2 + (y - 2)^2 = 4 :=
sorry

end circle_standard_equation_l1370_137086


namespace calculate_y_l1370_137015

theorem calculate_y (x y : ℝ) (h1 : x = 101) (h2 : x^3 * y - 2 * x^2 * y + x * y = 101000) : y = 1 / 10 :=
by
  sorry

end calculate_y_l1370_137015


namespace area_ratio_eq_l1370_137043

-- Define the parameters used in the problem
variables (t t1 r ρ : ℝ)

-- Define the conditions given in the problem
def area_triangle_ABC : ℝ := t
def area_triangle_A1B1C1 : ℝ := t1
def circumradius_ABC : ℝ := r
def inradius_A1B1C1 : ℝ := ρ

-- Problem statement: Prove the given equation
theorem area_ratio_eq : t / t1 = 2 * ρ / r :=
sorry

end area_ratio_eq_l1370_137043


namespace quadratic_equal_roots_l1370_137001

theorem quadratic_equal_roots (k : ℝ) : (∃ r : ℝ, (r^2 - 2 * r + k = 0)) → k = 1 := 
by
  sorry

end quadratic_equal_roots_l1370_137001


namespace minimize_expr_l1370_137093

-- Define the function we need to minimize
noncomputable def expr (α β : ℝ) : ℝ := 
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2

-- State the theorem to prove the minimum value of this expression
theorem minimize_expr (α β : ℝ) : ∃ (α β : ℝ), expr α β = 100 := 
sorry

end minimize_expr_l1370_137093


namespace principal_amount_l1370_137050

/-
  Given:
  - Simple Interest (SI) = Rs. 4016.25
  - Rate (R) = 0.08 (8% per annum)
  - Time (T) = 5 years
  
  We want to prove:
  Principal = Rs. 10040.625
-/

def SI : ℝ := 4016.25
def R : ℝ := 0.08
def T : ℕ := 5

theorem principal_amount :
  ∃ P : ℝ, SI = (P * R * T) / 100 ∧ P = 10040.625 :=
by
  sorry

end principal_amount_l1370_137050


namespace sum_of_smallest_and_largest_is_correct_l1370_137037

-- Define the conditions
def digits : Set ℕ := {0, 3, 4, 8}

-- Define the smallest and largest valid four-digit number using the digits
def smallest_number : ℕ := 3048
def largest_number : ℕ := 8430

-- Define the sum of the smallest and largest numbers
def sum_of_numbers : ℕ := smallest_number + largest_number

-- The theorem to be proven
theorem sum_of_smallest_and_largest_is_correct : 
  sum_of_numbers = 11478 := 
by
  -- Proof omitted
  sorry

end sum_of_smallest_and_largest_is_correct_l1370_137037


namespace exists_good_placement_l1370_137007

-- Define a function that checks if a placement is "good" with respect to a symmetry axis
def is_good (f : Fin 1983 → ℕ) : Prop :=
  ∀ (i : Fin 1983), f i < f (i + 991) ∨ f (i + 991) < f i

-- Prove the existence of a "good" placement for the regular 1983-gon
theorem exists_good_placement : ∃ f : Fin 1983 → ℕ, is_good f :=
sorry

end exists_good_placement_l1370_137007


namespace number_of_teachers_l1370_137042

theorem number_of_teachers
  (T S : ℕ)
  (h1 : T + S = 2400)
  (h2 : 320 = 320) -- This condition is trivial and can be ignored
  (h3 : 280 = 280) -- This condition is trivial and can be ignored
  (h4 : S / 280 = T / 40) : T = 300 :=
by
  sorry

end number_of_teachers_l1370_137042


namespace fraction_addition_target_l1370_137028

open Rat

theorem fraction_addition_target (n : ℤ) : 
  (4 + n) / (7 + n) = 3 / 4 → 
  n = 5 := 
by
  intro h
  sorry

end fraction_addition_target_l1370_137028


namespace total_books_l1370_137060

def initial_books : ℝ := 41.0
def first_addition : ℝ := 33.0
def second_addition : ℝ := 2.0

theorem total_books (h1 : initial_books = 41.0) (h2 : first_addition = 33.0) (h3 : second_addition = 2.0) :
  initial_books + first_addition + second_addition = 76.0 := 
by
  -- placeholders for the proof steps, omitting the detailed steps as instructed
  sorry

end total_books_l1370_137060


namespace sell_decision_l1370_137035

noncomputable def profit_beginning (a : ℝ) : ℝ :=
(a + 100) * 1.024

noncomputable def profit_end (a : ℝ) : ℝ :=
a + 115

theorem sell_decision (a : ℝ) :
  (a > 525 → profit_beginning a > profit_end a) ∧
  (a < 525 → profit_beginning a < profit_end a) ∧
  (a = 525 → profit_beginning a = profit_end a) :=
by
  sorry

end sell_decision_l1370_137035


namespace quadratic_real_roots_iff_l1370_137069

theorem quadratic_real_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 + 4 * x + k = 0) ↔ k ≤ 4 :=
by
  -- Proof is omitted, we only need the statement
  sorry

end quadratic_real_roots_iff_l1370_137069


namespace power_of_two_contains_k_as_substring_l1370_137026

theorem power_of_two_contains_k_as_substring (k : ℕ) (h1 : 1000 ≤ k) (h2 : k < 10000) : 
  ∃ n < 20000, ∀ m, 10^m * k ≤ 2^n ∧ 2^n < 10^(m+4) * (k+1) :=
sorry

end power_of_two_contains_k_as_substring_l1370_137026


namespace price_of_uniform_l1370_137018

-- Definitions based on conditions
def total_salary : ℕ := 600
def months_worked : ℕ := 9
def months_in_year : ℕ := 12
def salary_received : ℕ := 400
def uniform_price (U : ℕ) : Prop := 
    (3/4 * total_salary) - salary_received = U

-- Theorem stating the price of the uniform
theorem price_of_uniform : ∃ U : ℕ, uniform_price U := by
  sorry

end price_of_uniform_l1370_137018


namespace symmetrical_line_equation_l1370_137084

-- Definitions for the conditions
def line_symmetrical (eq1 eq2 : String) : Prop :=
  eq1 = "x - 2y + 3 = 0" ∧ eq2 = "x + 2y + 3 = 0"

-- Prove the statement
theorem symmetrical_line_equation : line_symmetrical "x - 2y + 3 = 0" "x + 2y + 3 = 0" :=
  by
  -- This is just the proof skeleton; the actual proof is not required
  sorry

end symmetrical_line_equation_l1370_137084


namespace sixth_term_of_geometric_sequence_l1370_137074

noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ n

theorem sixth_term_of_geometric_sequence (a : ℝ) (r : ℝ)
  (h1 : a = 243) (h2 : geometric_sequence a r 7 = 32) :
  geometric_sequence a r 5 = 1 :=
by
  sorry

end sixth_term_of_geometric_sequence_l1370_137074


namespace total_money_found_l1370_137023

def value_of_quarters (n_quarters : ℕ) : ℝ := n_quarters * 0.25
def value_of_dimes (n_dimes : ℕ) : ℝ := n_dimes * 0.10
def value_of_nickels (n_nickels : ℕ) : ℝ := n_nickels * 0.05
def value_of_pennies (n_pennies : ℕ) : ℝ := n_pennies * 0.01

theorem total_money_found (n_quarters n_dimes n_nickels n_pennies : ℕ) :
  n_quarters = 10 →
  n_dimes = 3 →
  n_nickels = 4 →
  n_pennies = 200 →
  value_of_quarters n_quarters + value_of_dimes n_dimes + value_of_nickels n_nickels + value_of_pennies n_pennies = 5.00 := 
by
  intros h_quarters h_dimes h_nickels h_pennies
  sorry

end total_money_found_l1370_137023


namespace find_denominator_l1370_137002

theorem find_denominator (x : ℕ) (dec_form_of_frac_4128 : ℝ) (h1: 4128 / x = dec_form_of_frac_4128) 
    : x = 4387 :=
by
  have h: dec_form_of_frac_4128 = 0.9411764705882353 := sorry
  sorry

end find_denominator_l1370_137002


namespace average_marks_first_class_l1370_137087

theorem average_marks_first_class
  (n1 n2 : ℕ)
  (avg2 : ℝ)
  (combined_avg : ℝ)
  (h_n1 : n1 = 35)
  (h_n2 : n2 = 55)
  (h_avg2 : avg2 = 65)
  (h_combined_avg : combined_avg = 57.22222222222222) :
  (∃ avg1 : ℝ, avg1 = 45) :=
by
  sorry

end average_marks_first_class_l1370_137087


namespace count_satisfying_integers_l1370_137062

theorem count_satisfying_integers :
  (∃ S : Finset ℕ, (∀ n ∈ S, 9 < n ∧ n < 60) ∧ S.card = 50) :=
by
  sorry

end count_satisfying_integers_l1370_137062


namespace solve_equation_l1370_137046

theorem solve_equation (x : ℝ) (h : (4 * x ^ 2 + 6 * x + 2) / (x + 2) = 4 * x + 7) : x = -4 / 3 :=
by
  sorry

end solve_equation_l1370_137046
