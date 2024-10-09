import Mathlib

namespace two_pow_a_plus_two_pow_neg_a_l801_80136

theorem two_pow_a_plus_two_pow_neg_a (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 1) :
  2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end two_pow_a_plus_two_pow_neg_a_l801_80136


namespace total_profit_Q2_is_correct_l801_80188

-- Conditions as definitions
def profit_Q1_A := 1500
def profit_Q1_B := 2000
def profit_Q1_C := 1000

def profit_Q2_A := 2500
def profit_Q2_B := 3000
def profit_Q2_C := 1500

def profit_Q3_A := 3000
def profit_Q3_B := 2500
def profit_Q3_C := 3500

def profit_Q4_A := 2000
def profit_Q4_B := 3000
def profit_Q4_C := 2000

-- The total profit calculation for the second quarter
def total_profit_Q2 := profit_Q2_A + profit_Q2_B + profit_Q2_C

-- Proof statement
theorem total_profit_Q2_is_correct : total_profit_Q2 = 7000 := by
  sorry

end total_profit_Q2_is_correct_l801_80188


namespace fraction_sum_of_two_reciprocals_fraction_sum_of_equal_reciprocals_fraction_difference_of_two_reciprocals_l801_80135

theorem fraction_sum_of_two_reciprocals (n : ℕ) (hn : n > 0) : 
  ∃ a b : ℕ, (a ≠ b) ∧ (3 * 5 * n * (a + b) = a * b) :=
sorry

theorem fraction_sum_of_equal_reciprocals (n : ℕ) : 
  ∃ a : ℕ, 3 * 5 * n * 2 = a * a ↔ (∃ k : ℕ, n = 2 * k) :=
sorry

theorem fraction_difference_of_two_reciprocals (n : ℕ) (hn : n > 0) : 
  ∃ a b : ℕ, (a ≠ b) ∧ 3 * 5 * n * (a - b) = a * b :=
sorry

end fraction_sum_of_two_reciprocals_fraction_sum_of_equal_reciprocals_fraction_difference_of_two_reciprocals_l801_80135


namespace least_positive_integer_div_conditions_l801_80182

theorem least_positive_integer_div_conditions :
  ∃ n > 1, (n % 4 = 3) ∧ (n % 5 = 3) ∧ (n % 7 = 3) ∧ (n % 10 = 3) ∧ (n % 11 = 3) ∧ n = 1543 := 
by 
  sorry

end least_positive_integer_div_conditions_l801_80182


namespace smallest_possible_x_l801_80112

/-- Proof problem: When x is divided by 6, 7, and 8, remainders of 5, 6, and 7 (respectively) are obtained. 
We need to show that the smallest possible positive integer value of x is 167. -/
theorem smallest_possible_x (x : ℕ) (h1 : x % 6 = 5) (h2 : x % 7 = 6) (h3 : x % 8 = 7) : x = 167 :=
by 
  sorry

end smallest_possible_x_l801_80112


namespace irene_age_is_46_l801_80145

def eddie_age : ℕ := 92

def becky_age (e_age : ℕ) : ℕ := e_age / 4

def irene_age (b_age : ℕ) : ℕ := 2 * b_age

theorem irene_age_is_46 : irene_age (becky_age eddie_age) = 46 := 
  by
    sorry

end irene_age_is_46_l801_80145


namespace sequence_sum_l801_80179

-- Define the arithmetic sequence and conditions
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific values used in the problem
def specific_condition (a : ℕ → ℝ) : Prop :=
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450)

-- The proof goal that needs to be established
theorem sequence_sum (a : ℕ → ℝ) (h1 : arithmetic_seq a) (h2 : specific_condition a) : a 2 + a 8 = 180 :=
by
  sorry

end sequence_sum_l801_80179


namespace Terry_has_20_more_stickers_than_Steven_l801_80143

theorem Terry_has_20_more_stickers_than_Steven :
  let Ryan_stickers := 30
  let Steven_stickers := 3 * Ryan_stickers
  let Total_stickers := 230
  let Ryan_Steven_Total := Ryan_stickers + Steven_stickers
  let Terry_stickers := Total_stickers - Ryan_Steven_Total
  (Terry_stickers - Steven_stickers) = 20 := 
by 
  sorry

end Terry_has_20_more_stickers_than_Steven_l801_80143


namespace pizza_toppings_l801_80117

theorem pizza_toppings (toppings : Finset String) (h : toppings.card = 8) :
  (toppings.card.choose 1 + toppings.card.choose 2 + toppings.card.choose 3) = 92 := by
  have ht : toppings.card = 8 := h
  sorry

end pizza_toppings_l801_80117


namespace exponent_equality_l801_80164

theorem exponent_equality (M : ℕ) : 32^4 * 4^6 = 2^M → M = 32 :=
by
  sorry

end exponent_equality_l801_80164


namespace zoo_guides_children_total_l801_80154

theorem zoo_guides_children_total :
  let num_guides := 22
  let num_english_guides := 10
  let num_french_guides := 6
  let num_spanish_guides := num_guides - num_english_guides - num_french_guides
  let children_english_friday := 10 * 20
  let children_french_friday := 6 * 25
  let children_spanish_friday := num_spanish_guides * 30
  let children_english_saturday := 10 * 22
  let children_french_saturday := 6 * 24
  let children_spanish_saturday := num_spanish_guides * 32
  let children_english_sunday := 10 * 24
  let children_french_sunday := 6 * 23
  let children_spanish_sunday := num_spanish_guides * 35
  let total_children := children_english_friday + children_french_friday + children_spanish_friday + children_english_saturday + children_french_saturday + children_spanish_saturday + children_english_sunday + children_french_sunday + children_spanish_sunday
  total_children = 1674 :=
by
  let num_guides := 22
  let num_english_guides := 10
  let num_french_guides := 6
  let num_spanish_guides := num_guides - num_english_guides - num_french_guides
  let children_english_friday := 10 * 20
  let children_french_friday := 6 * 25
  let children_spanish_friday := num_spanish_guides * 30
  let children_english_saturday := 10 * 22
  let children_french_saturday := 6 * 24
  let children_spanish_saturday := num_spanish_guides * 32
  let children_english_sunday := 10 * 24
  let children_french_sunday := 6 * 23
  let children_spanish_sunday := num_spanish_guides * 35
  let total_children := children_english_friday + children_french_friday + children_spanish_friday + children_english_saturday + children_french_saturday + children_spanish_saturday + children_english_sunday + children_french_sunday + children_spanish_sunday
  sorry

end zoo_guides_children_total_l801_80154


namespace find_a2_given_conditions_l801_80173

variable (a : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) := ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

theorem find_a2_given_conditions
  {a : ℕ → ℤ}
  (h_seq : is_arithmetic_sequence a)
  (h1 : a 3 + a 5 = 24)
  (h2 : a 7 - a 3 = 24) :
  a 2 = 0 :=
by
  sorry

end find_a2_given_conditions_l801_80173


namespace min_value_l801_80127

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) : 
  ∃ xy : ℝ, (xy = 9 ∧ (forall (u v : ℝ), (u > 0) → (v > 0) → 2 * u + v = 1 → (2 / u) + (1 / v) ≥ xy)) :=
by
  use 9
  sorry

end min_value_l801_80127


namespace sum_of_roots_of_equation_l801_80131

theorem sum_of_roots_of_equation : 
  (∃ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 ∧ (x1 + x2 = 14)) :=
by
  sorry

end sum_of_roots_of_equation_l801_80131


namespace seunghye_saw_number_l801_80134

theorem seunghye_saw_number (x : ℝ) (h : 10 * x - x = 37.35) : x = 4.15 :=
by
  sorry

end seunghye_saw_number_l801_80134


namespace number_of_nephews_l801_80139

def total_jellybeans : ℕ := 70
def jellybeans_per_child : ℕ := 14
def number_of_nieces : ℕ := 2

theorem number_of_nephews : total_jellybeans / jellybeans_per_child - number_of_nieces = 3 := by
  sorry

end number_of_nephews_l801_80139


namespace total_ants_correct_l801_80133

def abe_ants : ℕ := 4
def beth_ants : ℕ := abe_ants + (abe_ants / 2)
def cece_ants : ℕ := 2 * abe_ants
def duke_ants : ℕ := abe_ants / 2
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

theorem total_ants_correct : total_ants = 20 :=
by
  sorry

end total_ants_correct_l801_80133


namespace arrange_natural_numbers_divisors_l801_80197

theorem arrange_natural_numbers_divisors :
  ∃ (seq : List ℕ), seq = [7, 1, 8, 4, 10, 6, 9, 3, 2, 5] ∧ 
  seq.length = 10 ∧
  ∀ n (h : n < seq.length), seq[n] ∣ (List.take n seq).sum := 
by
  sorry

end arrange_natural_numbers_divisors_l801_80197


namespace quadratic_other_root_l801_80181

theorem quadratic_other_root (m : ℝ) :
  (2 * 1^2 - m * 1 + 6 = 0) →
  ∃ y : ℝ, y ≠ 1 ∧ (2 * y^2 - m * y + 6 = 0) ∧ (1 * y = 3) :=
by
  intros h
  -- using sorry to skip the actual proof
  sorry

end quadratic_other_root_l801_80181


namespace ratio_gold_to_green_horses_l801_80132

theorem ratio_gold_to_green_horses (blue_horses purple_horses green_horses gold_horses : ℕ)
    (h1 : blue_horses = 3)
    (h2 : purple_horses = 3 * blue_horses)
    (h3 : green_horses = 2 * purple_horses)
    (h4 : blue_horses + purple_horses + green_horses + gold_horses = 33) :
  gold_horses / gcd gold_horses green_horses = 1 / 6 :=
by
  sorry

end ratio_gold_to_green_horses_l801_80132


namespace sum_of_squares_l801_80153

theorem sum_of_squares (a b c : ℝ) (h1 : ab + bc + ca = 4) (h2 : a + b + c = 17) : a^2 + b^2 + c^2 = 281 :=
by
  sorry

end sum_of_squares_l801_80153


namespace general_formula_for_sequence_l801_80102

theorem general_formula_for_sequence :
  ∀ (a : ℕ → ℕ), (a 0 = 1) → (a 1 = 1) →
  (∀ n, 2 ≤ n → a n = 2 * a (n - 1) - a (n - 2)) →
  ∀ n, a n = (2^n - 1)^2 :=
by
  sorry

end general_formula_for_sequence_l801_80102


namespace minimum_distance_from_mars_l801_80140

noncomputable def distance_function (a b c t : ℝ) : ℝ :=
  a * t^2 + b * t + c

theorem minimum_distance_from_mars :
  ∃ t₀ : ℝ, distance_function (11/54) (-1/18) 4 t₀ = (9:ℝ) :=
  sorry

end minimum_distance_from_mars_l801_80140


namespace abs_sum_less_than_two_l801_80138

theorem abs_sum_less_than_two (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) : |a + b| + |a - b| < 2 := 
sorry

end abs_sum_less_than_two_l801_80138


namespace transformed_stats_l801_80130

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def std_dev (l : List ℝ) : ℝ :=
  Real.sqrt ((l.map (λ x => (x - mean l)^2)).sum / l.length)

theorem transformed_stats (l : List ℝ) 
  (hmean : mean l = 10)
  (hstddev : std_dev l = 2) :
  mean (l.map (λ x => 2 * x - 1)) = 19 ∧ std_dev (l.map (λ x => 2 * x - 1)) = 4 := by
  sorry

end transformed_stats_l801_80130


namespace total_fruits_l801_80113

theorem total_fruits (a b c : ℕ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 4) : a + b + c = 15 := by
  sorry

end total_fruits_l801_80113


namespace age_of_son_l801_80176

theorem age_of_son (S F : ℕ) (h1 : F = S + 28) (h2 : F + 2 = 2 * (S + 2)) : S = 26 := 
by
  -- skip the proof
  sorry

end age_of_son_l801_80176


namespace loss_percentage_l801_80151

/--
A man sells a car to his friend at a certain loss percentage. The friend then sells it 
for Rs. 54000 and gains 20%. The original cost price of the car was Rs. 52941.17647058824.
Prove that the loss percentage when the man sold the car to his friend was 15%.
-/
theorem loss_percentage (CP SP_2 : ℝ) (gain_percent : ℝ) (h_CP : CP = 52941.17647058824) 
(h_SP2 : SP_2 = 54000) (h_gain : gain_percent = 20) : (CP - SP_2 / (1 + gain_percent / 100)) / CP * 100 = 15 := by
  sorry

end loss_percentage_l801_80151


namespace sum_nonpositive_inequality_l801_80110

theorem sum_nonpositive_inequality (x : ℝ) : x + 5 ≤ 0 ↔ x + 5 ≤ 0 :=
by
  sorry

end sum_nonpositive_inequality_l801_80110


namespace other_denominations_l801_80178

theorem other_denominations :
  ∀ (total_checks : ℕ) (total_value : ℝ) (fifty_denomination_checks : ℕ) (remaining_avg : ℝ),
    total_checks = 30 →
    total_value = 1800 →
    fifty_denomination_checks = 15 →
    remaining_avg = 70 →
    ∃ (other_denomination : ℝ), other_denomination = 70 :=
by
  intros total_checks total_value fifty_denomination_checks remaining_avg
  intros h1 h2 h3 h4
  let other_denomination := 70
  use other_denomination
  sorry

end other_denominations_l801_80178


namespace fraction_of_work_left_l801_80156

theorem fraction_of_work_left 
  (A_days : ℕ) (B_days : ℕ) (work_days : ℕ) 
  (A_rate : ℚ := 1 / A_days) (B_rate : ℚ := 1 / B_days) (combined_rate : ℚ := 1 / A_days + 1 / B_days) 
  (work_completed : ℚ := combined_rate * work_days) (fraction_left : ℚ := 1 - work_completed)
  (hA : A_days = 15) (hB : B_days = 20) (hW : work_days = 4) 
  : fraction_left = 8 / 15 :=
sorry

end fraction_of_work_left_l801_80156


namespace af_b_lt_bf_a_l801_80129

variable {f : ℝ → ℝ}
variable {a b : ℝ}

theorem af_b_lt_bf_a (h1 : ∀ x y, 0 < x → 0 < y → x < y → f x > f y)
                    (h2 : ∀ x, 0 < x → f x > 0)
                    (h3 : 0 < a)
                    (h4 : 0 < b)
                    (h5 : a < b) :
  a * f b < b * f a :=
sorry

end af_b_lt_bf_a_l801_80129


namespace distance_from_focus_l801_80170

theorem distance_from_focus (x : ℝ) (A : ℝ × ℝ) (hA_on_parabola : A.1^2 = 4 * A.2) (hA_coord : A.2 = 4) : 
  dist A (0, 1) = 5 := 
by
  sorry

end distance_from_focus_l801_80170


namespace remainder_is_three_l801_80147

def dividend : ℕ := 15
def divisor : ℕ := 3
def quotient : ℕ := 4

theorem remainder_is_three : dividend = (divisor * quotient) + Nat.mod dividend divisor := by
  sorry

end remainder_is_three_l801_80147


namespace find_x_for_equation_l801_80163

theorem find_x_for_equation :
  ∃ x : ℝ, 9 - 3 / (1 / x) + 3 = 3 ↔ x = 3 := 
by 
  sorry

end find_x_for_equation_l801_80163


namespace square_of_ratio_is_specified_value_l801_80100

theorem square_of_ratio_is_specified_value (a b c : ℝ) (h1 : c = Real.sqrt (a^2 + b^2)) (h2 : a / b = b / c) :
  (a / b)^2 = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end square_of_ratio_is_specified_value_l801_80100


namespace hyperbola_parabola_foci_l801_80193

-- Definition of the hyperbola
def hyperbola (k : ℝ) (x y : ℝ) : Prop := y^2 / 5 - x^2 / k = 1

-- Definition of the parabola
def parabola (x y : ℝ) : Prop := x^2 = 12 * y

-- Condition that both curves have the same foci
def same_foci (focus : ℝ) (x y : ℝ) : Prop := focus = 3 ∧ (parabola x y → ((0, focus) : ℝ×ℝ) = (0, 3)) ∧ (∃ k : ℝ, hyperbola k x y ∧ ((0, focus) : ℝ×ℝ) = (0, 3))

theorem hyperbola_parabola_foci (k : ℝ) (x y : ℝ) : same_foci 3 x y → k = -4 := 
by {
  sorry
}

end hyperbola_parabola_foci_l801_80193


namespace xiao_wang_programming_methods_l801_80150

theorem xiao_wang_programming_methods :
  ∃ (n : ℕ), n = 20 :=
by sorry

end xiao_wang_programming_methods_l801_80150


namespace find_directrix_l801_80122

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop := x^2 = 8 * y

-- State the problem to find the directrix of the given parabola
theorem find_directrix (x y : ℝ) (h : parabola_eq x y) : y = -2 :=
sorry

end find_directrix_l801_80122


namespace favorite_movies_hours_l801_80183

theorem favorite_movies_hours (J M N R : ℕ) (h1 : J = M + 2) (h2 : N = 3 * M) (h3 : R = (4 * N) / 5) (h4 : N = 30) : 
  J + M + N + R = 76 :=
by
  sorry

end favorite_movies_hours_l801_80183


namespace minimum_value_expression_l801_80128

theorem minimum_value_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  4 * a ^ 3 + 8 * b ^ 3 + 27 * c ^ 3 + 64 * d ^ 3 + 2 / (a * b * c * d) ≥ 16 * Real.sqrt 3 :=
by
  sorry

end minimum_value_expression_l801_80128


namespace num_zeros_in_interval_l801_80177

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 6 * x ^ 2 + 7

theorem num_zeros_in_interval : 
    (∃ (a b : ℝ), a < b ∧ a = 0 ∧ b = 2 ∧
     (∀ x, f x = 0 → (0 < x ∧ x < 2)) ∧
     (∃! x, (0 < x ∧ x < 2) ∧ f x = 0)) :=
by
    sorry

end num_zeros_in_interval_l801_80177


namespace range_of_m_l801_80192

-- Conditions:
def is_opposite_sides_of_line (p1 p2 : ℝ × ℝ) (a b m : ℝ) : Prop :=
  let l1 := a * p1.1 + b * p1.2 + m
  let l2 := a * p2.1 + b * p2.2 + m
  l1 * l2 < 0

-- Point definitions:
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (-4, -2)

-- Line definition with coefficients
def a : ℝ := 2
def b : ℝ := 1

-- Proof Goal:
theorem range_of_m (m : ℝ) : is_opposite_sides_of_line point1 point2 a b m ↔ -5 < m ∧ m < 10 :=
by sorry

end range_of_m_l801_80192


namespace Yihana_uphill_walking_time_l801_80104

theorem Yihana_uphill_walking_time :
  let t1 := 3
  let t2 := 2
  let t_total := t1 + t2
  t_total = 5 :=
by
  let t1 := 3
  let t2 := 2
  let t_total := t1 + t2
  show t_total = 5
  sorry

end Yihana_uphill_walking_time_l801_80104


namespace find_px_l801_80189

theorem find_px (p : ℕ → ℚ) (h1 : p 1 = 1) (h2 : p 2 = 1 / 4) (h3 : p 3 = 1 / 9) 
  (h4 : p 4 = 1 / 16) (h5 : p 5 = 1 / 25) : p 6 = 1 / 18 :=
sorry

end find_px_l801_80189


namespace find_number_of_lines_l801_80142

theorem find_number_of_lines (n : ℕ) (h : (n * (n - 1) / 2) * 8 = 280) : n = 10 :=
by
  sorry

end find_number_of_lines_l801_80142


namespace work_days_l801_80169

theorem work_days (hp : ℝ) (hq : ℝ) (fraction_left : ℝ) (d : ℝ) :
  hp = 1 / 20 → hq = 1 / 10 → fraction_left = 0.7 → (3 / 20) * d = (1 - fraction_left) → d = 2 :=
  by
  intros hp_def hq_def fraction_def work_eq
  sorry

end work_days_l801_80169


namespace day_crew_fraction_correct_l801_80187

variable (D Wd : ℕ) -- D = number of boxes loaded by each worker on the day crew, Wd = number of workers on the day crew

-- fraction of all boxes loaded by day crew
def fraction_loaded_by_day_crew (D Wd : ℕ) : ℚ :=
  (D * Wd) / (D * Wd + (3 / 4 * D) * (2 / 3 * Wd))

theorem day_crew_fraction_correct (h1 : D > 0) (h2 : Wd > 0) :
  fraction_loaded_by_day_crew D Wd = 2 / 3 := by
  sorry

end day_crew_fraction_correct_l801_80187


namespace find_angle_D_l801_80167

variable (A B C D : ℝ)
variable (h1 : A + B = 180)
variable (h2 : C = D)
variable (h3 : C + 50 + 60 = 180)

theorem find_angle_D : D = 70 := by
  sorry

end find_angle_D_l801_80167


namespace probability_sum_even_is_five_over_eleven_l801_80106

noncomputable def probability_even_sum : ℚ :=
  let totalBalls := 12
  let totalWays := totalBalls * (totalBalls - 1)
  let evenBalls := 6
  let oddBalls := 6
  let evenWays := evenBalls * (evenBalls - 1)
  let oddWays := oddBalls * (oddBalls - 1)
  let totalEvenWays := evenWays + oddWays
  totalEvenWays / totalWays

theorem probability_sum_even_is_five_over_eleven : probability_even_sum = 5 / 11 := sorry

end probability_sum_even_is_five_over_eleven_l801_80106


namespace darren_and_fergie_same_amount_in_days_l801_80107

theorem darren_and_fergie_same_amount_in_days : 
  ∀ (t : ℕ), (200 + 16 * t = 300 + 12 * t) → t = 25 := 
by sorry

end darren_and_fergie_same_amount_in_days_l801_80107


namespace determine_k_l801_80114

def f(x : ℝ) : ℝ := 5 * x^2 - 3 * x + 8
def g(x k : ℝ) : ℝ := x^3 - k * x - 10

theorem determine_k : 
  (f (-5) - g (-5) k = -24) → k = 61 := 
by 
-- Begin the proof script here
sorry

end determine_k_l801_80114


namespace find_ratio_l801_80152

def celsius_to_fahrenheit_ratio (ratio : ℝ) (c f : ℝ) : Prop :=
  f = ratio * c + 32

theorem find_ratio (ratio : ℝ) :
  (∀ c f, celsius_to_fahrenheit_ratio ratio c f ∧ ((f = 58) → (c = 14.444444444444445)) → f = 1.8 * c + 32) ∧ 
  (f - 32 = ratio * (c - 0)) ∧
  (c = 14.444444444444445 → f = 32 + 26) ∧
  (f = 58 → c = 14.444444444444445) ∧ 
  (ratio = 1.8)
  → ratio = 1.8 := 
sorry 


end find_ratio_l801_80152


namespace geometric_number_difference_l801_80165

-- Definitions
def is_geometric_sequence (a b c d : ℕ) : Prop := ∃ r : ℚ, b = a * r ∧ c = a * r^2 ∧ d = a * r^3

def is_valid_geometric_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧  -- 4-digit number
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ -- distinct digits
    is_geometric_sequence a b c d ∧ -- geometric sequence
    n = a * 1000 + b * 100 + c * 10 + d -- digits form the number

-- Theorem statement
theorem geometric_number_difference : 
  ∃ (m M : ℕ), is_valid_geometric_number m ∧ is_valid_geometric_number M ∧ (M - m = 7173) :=
sorry

end geometric_number_difference_l801_80165


namespace arithmetic_sequence_a20_l801_80144

theorem arithmetic_sequence_a20 (a : Nat → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + a 3 + a 5 = 18)
  (h3 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 :=
sorry

end arithmetic_sequence_a20_l801_80144


namespace math_problem_l801_80141

theorem math_problem
  (x y : ℚ)
  (h1 : x + y = 11 / 17)
  (h2 : x - y = 1 / 143) :
  x^2 - y^2 = 11 / 2431 :=
by
  sorry

end math_problem_l801_80141


namespace box_volume_l801_80121

theorem box_volume
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12)
  (h4 : l = h + 1) :
  l * w * h = 120 := 
sorry

end box_volume_l801_80121


namespace hair_cut_length_l801_80190

theorem hair_cut_length (original_length after_haircut : ℕ) (h1 : original_length = 18) (h2 : after_haircut = 9) :
  original_length - after_haircut = 9 :=
by
  sorry

end hair_cut_length_l801_80190


namespace linear_function_quadrants_l801_80126

theorem linear_function_quadrants : 
  ∀ (x y : ℝ), y = -5 * x + 3 
  → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by 
  intro x y h
  sorry

end linear_function_quadrants_l801_80126


namespace volume_of_wedge_l801_80159

theorem volume_of_wedge (h : 2 * Real.pi * r = 18 * Real.pi) :
  let V := (4 / 3) * Real.pi * (r ^ 3)
  let V_wedge := V / 6
  V_wedge = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l801_80159


namespace max_g_value_l801_80108

def g (n : ℕ) : ℕ :=
if h : n < 10 then 2 * n + 3 else g (n - 7)

theorem max_g_value : ∃ n, g n = 21 ∧ ∀ m, g m ≤ 21 :=
sorry

end max_g_value_l801_80108


namespace four_cubic_feet_to_cubic_inches_l801_80119

theorem four_cubic_feet_to_cubic_inches (h : 1 = 12) : 4 * (12^3) = 6912 :=
by
  sorry

end four_cubic_feet_to_cubic_inches_l801_80119


namespace min_value_frac_l801_80175

theorem min_value_frac (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : a + b = 1) : 
  (1 / a) + (4 / b) ≥ 9 :=
by sorry

end min_value_frac_l801_80175


namespace arithmetic_sequence_lemma_l801_80162

theorem arithmetic_sequence_lemma (a : ℕ → ℝ) (h_arith_seq : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0)
  (h_condition : a 3 + a 11 = 22) : a 7 = 11 :=
sorry

end arithmetic_sequence_lemma_l801_80162


namespace sum_of_x_coordinates_mod13_intersection_l801_80161

theorem sum_of_x_coordinates_mod13_intersection :
  (∀ x y : ℕ, y ≡ 3 * x + 5 [MOD 13] → y ≡ 7 * x + 4 [MOD 13]) → (x ≡ 10 [MOD 13]) :=
by
  sorry

end sum_of_x_coordinates_mod13_intersection_l801_80161


namespace cost_of_plastering_l801_80180

def length := 25
def width := 12
def depth := 6
def cost_per_sq_meter_paise := 75

def surface_area_of_two_walls_one := 2 * (length * depth)
def surface_area_of_two_walls_two := 2 * (width * depth)
def surface_area_of_bottom := length * width

def total_surface_area := surface_area_of_two_walls_one + surface_area_of_two_walls_two + surface_area_of_bottom

def cost_per_sq_meter_rupees := cost_per_sq_meter_paise / 100
def total_cost := total_surface_area * cost_per_sq_meter_rupees

theorem cost_of_plastering : total_cost = 558 := by
  sorry

end cost_of_plastering_l801_80180


namespace y_value_when_x_is_20_l801_80157

theorem y_value_when_x_is_20 :
  ∀ (x : ℝ), (∀ m c : ℝ, m = 2.5 → c = 3 → (y = m * x + c) → x = 20 → y = 53) :=
by
  sorry

end y_value_when_x_is_20_l801_80157


namespace triangle_areas_l801_80155

theorem triangle_areas (r s : ℝ) (h1 : s = (1/2) * r + 6)
                       (h2 : (12 + r) * ((1/2) * r + 6) = 18) :
  r + s = -3 :=
by
  sorry

end triangle_areas_l801_80155


namespace simultaneous_equations_solution_l801_80172

theorem simultaneous_equations_solution (x y : ℚ) (h1 : 3 * x - 4 * y = 11) (h2 : 9 * x + 6 * y = 33) : 
  x = 11 / 3 ∧ y = 0 :=
by {
  sorry
}

end simultaneous_equations_solution_l801_80172


namespace digits_difference_l801_80137

-- Definitions based on conditions
variables (X Y : ℕ)

-- Condition: The difference between the original number and the interchanged number is 27
def difference_condition : Prop :=
  (10 * X + Y) - (10 * Y + X) = 27

-- Problem to prove: The difference between the two digits is 3
theorem digits_difference (h : difference_condition X Y) : X - Y = 3 :=
by sorry

end digits_difference_l801_80137


namespace find_initial_girls_l801_80171

variable (b g : ℕ)

theorem find_initial_girls 
  (h1 : 3 * (g - 18) = b)
  (h2 : 4 * (b - 36) = g - 18) :
  g = 31 := 
by
  sorry

end find_initial_girls_l801_80171


namespace find_x_l801_80146

theorem find_x (x m n : ℤ) 
  (h₁ : 15 + x = m^2) 
  (h₂ : x - 74 = n^2) :
  x = 2010 :=
by
  sorry

end find_x_l801_80146


namespace sum_of_four_primes_is_prime_l801_80186

theorem sum_of_four_primes_is_prime
    (A B : ℕ)
    (hA_prime : Prime A)
    (hB_prime : Prime B)
    (hA_minus_B_prime : Prime (A - B))
    (hA_plus_B_prime : Prime (A + B)) :
    Prime (A + B + (A - B) + A) :=
by
  sorry

end sum_of_four_primes_is_prime_l801_80186


namespace bob_password_probability_l801_80105

def num_non_negative_single_digits : ℕ := 10
def num_odd_single_digits : ℕ := 5
def num_even_positive_single_digits : ℕ := 4
def probability_first_digit_odd : ℚ := num_odd_single_digits / num_non_negative_single_digits
def probability_middle_letter : ℚ := 1
def probability_last_digit_even_positive : ℚ := num_even_positive_single_digits / num_non_negative_single_digits

theorem bob_password_probability :
  probability_first_digit_odd * probability_middle_letter * probability_last_digit_even_positive = 1 / 5 :=
by
  sorry

end bob_password_probability_l801_80105


namespace alpha_sufficient_but_not_necessary_condition_of_beta_l801_80185
open Classical

variable (x : ℝ)
def α := x = -1
def β := x ≤ 0

theorem alpha_sufficient_but_not_necessary_condition_of_beta :
  (α x → β x) ∧ ¬(β x → α x) :=
by
  sorry

end alpha_sufficient_but_not_necessary_condition_of_beta_l801_80185


namespace sin_sum_given_cos_tan_conditions_l801_80124

open Real

theorem sin_sum_given_cos_tan_conditions 
  (α β : ℝ)
  (h1 : cos α + cos β = 1 / 3)
  (h2 : tan (α + β) = 24 / 7)
  : sin α + sin β = 1 / 4 ∨ sin α + sin β = -4 / 9 := 
  sorry

end sin_sum_given_cos_tan_conditions_l801_80124


namespace fraction_increase_l801_80168

theorem fraction_increase (m n a : ℕ) (h1 : m > n) (h2 : a > 0) : 
  (n : ℚ) / m < (n + a : ℚ) / (m + a) :=
by
  sorry

end fraction_increase_l801_80168


namespace price_of_each_tomato_l801_80166

theorem price_of_each_tomato
  (customers_per_month : ℕ)
  (lettuce_per_customer : ℕ)
  (lettuce_price : ℕ)
  (tomatoes_per_customer : ℕ)
  (total_monthly_sales : ℕ)
  (total_lettuce_sales : ℕ)
  (total_tomato_sales : ℕ)
  (price_per_tomato : ℝ)
  (h1 : customers_per_month = 500)
  (h2 : lettuce_per_customer = 2)
  (h3 : lettuce_price = 1)
  (h4 : tomatoes_per_customer = 4)
  (h5 : total_monthly_sales = 2000)
  (h6 : total_lettuce_sales = customers_per_month * lettuce_per_customer * lettuce_price)
  (h7 : total_tomato_sales = total_monthly_sales - total_lettuce_sales)
  (h8 : total_lettuce_sales = 1000)
  (h9 : total_tomato_sales = 1000)
  (total_tomatoes_sold : ℕ := customers_per_month * tomatoes_per_customer)
  (h10 : total_tomatoes_sold = 2000) :
  price_per_tomato = 0.50 :=
by
  sorry

end price_of_each_tomato_l801_80166


namespace minimum_tan_theta_is_sqrt7_l801_80118

noncomputable def min_tan_theta (z : ℂ) : ℝ := (Complex.abs (Complex.im z) / Complex.abs (Complex.re z))

theorem minimum_tan_theta_is_sqrt7 {z : ℂ} 
  (hz_real : 0 ≤ Complex.re z)
  (hz_imag : 0 ≤ Complex.im z)
  (hz_condition : Complex.abs (z^2 + 2) ≤ Complex.abs z) :
  min_tan_theta z = Real.sqrt 7 := sorry

end minimum_tan_theta_is_sqrt7_l801_80118


namespace remainder_of_470521_div_5_l801_80196

theorem remainder_of_470521_div_5 : 470521 % 5 = 1 := 
by sorry

end remainder_of_470521_div_5_l801_80196


namespace machine_A_production_is_4_l801_80149

noncomputable def machine_production (A : ℝ) (B : ℝ) (T_A : ℝ) (T_B : ℝ) := 
  (440 / A = T_A) ∧
  (440 / B = T_B) ∧
  (T_A = T_B + 10) ∧
  (B = 1.10 * A)

theorem machine_A_production_is_4 {A B T_A T_B : ℝ}
  (h : machine_production A B T_A T_B) : 
  A = 4 :=
by
  sorry

end machine_A_production_is_4_l801_80149


namespace garden_area_l801_80101

variable (L W A : ℕ)
variable (H1 : 3000 = 50 * L)
variable (H2 : 3000 = 15 * (2*L + 2*W))

theorem garden_area : A = 2400 :=
by
  sorry

end garden_area_l801_80101


namespace intersection_P_Q_eq_Q_l801_80111

def P : Set ℝ := { x | x < 2 }
def Q : Set ℝ := { x | x^2 ≤ 1 }

theorem intersection_P_Q_eq_Q : P ∩ Q = Q := 
sorry

end intersection_P_Q_eq_Q_l801_80111


namespace find_f_of_monotonic_and_condition_l801_80195

noncomputable def monotonic (f : ℝ → ℝ) :=
  ∀ {a b : ℝ}, a < b → f a ≤ f b

theorem find_f_of_monotonic_and_condition (f : ℝ → ℝ) (h_mono : monotonic f) (h_cond : ∀ x : ℝ, 0 < x → f (f x - x^2) = 6) : f 2 = 6 :=
by
  sorry

end find_f_of_monotonic_and_condition_l801_80195


namespace find_constant_b_l801_80148

theorem find_constant_b 
  (a b c : ℝ)
  (h1 : 3 * a = 9) 
  (h2 : (-2 * a + 3 * b) = -5) 
  : b = 1 / 3 :=
by 
  have h_a : a = 3 := by linarith
  
  have h_b : -2 * 3 + 3 * b = -5 := by linarith [h2]
  
  linarith

end find_constant_b_l801_80148


namespace residue_calculation_l801_80115

theorem residue_calculation 
  (h1 : 182 ≡ 0 [MOD 14])
  (h2 : 182 * 12 ≡ 0 [MOD 14])
  (h3 : 15 * 7 ≡ 7 [MOD 14])
  (h4 : 3 ≡ 3 [MOD 14]) :
  (182 * 12 - 15 * 7 + 3) ≡ 10 [MOD 14] :=
sorry

end residue_calculation_l801_80115


namespace proof_problem_l801_80199

theorem proof_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b ∣ c * (c ^ 2 - c + 1))
  (h5 : (c ^ 2 + 1) ∣ (a + b)) :
  (a = c ∧ b = c ^ 2 - c + 1) ∨ (a = c ^ 2 - c + 1 ∧ b = c) :=
sorry

end proof_problem_l801_80199


namespace arithmetic_sequence_general_term_and_sum_l801_80109

theorem arithmetic_sequence_general_term_and_sum :
  (∃ (a₁ d : ℤ), a₁ + d = 14 ∧ a₁ + 4 * d = 5 ∧ ∀ n : ℤ, a_n = a₁ + (n - 1) * d ∧ (∀ N : ℤ, N ≥ 1 → S_N = N * ((2 * a₁ + (N - 1) * d) / 2) ∧ N = 10 → S_N = 35)) :=
sorry

end arithmetic_sequence_general_term_and_sum_l801_80109


namespace cos_alpha_plus_pi_six_l801_80123

theorem cos_alpha_plus_pi_six (α : ℝ) (hα_in_interval : 0 < α ∧ α < π / 2) (h_cos : Real.cos α = Real.sqrt 3 / 3) :
  Real.cos (α + π / 6) = (3 - Real.sqrt 6) / 6 := 
by
  sorry

end cos_alpha_plus_pi_six_l801_80123


namespace fraction_q_over_p_l801_80125

noncomputable def proof_problem (p q : ℝ) : Prop :=
  ∃ k : ℝ, p = 9^k ∧ q = 12^k ∧ p + q = 16^k

theorem fraction_q_over_p (p q : ℝ) (h : proof_problem p q) : q / p = (1 + Real.sqrt 5) / 2 :=
sorry

end fraction_q_over_p_l801_80125


namespace value_sq_dist_OP_OQ_l801_80174

-- Definitions from problem conditions
def origin : ℝ × ℝ := (0, 0)
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1
def perpendicular (p q : ℝ × ℝ) : Prop := p.1 * q.1 + p.2 * q.2 = 0

-- The proof statement
theorem value_sq_dist_OP_OQ 
  (P Q : ℝ × ℝ) 
  (hP : ellipse P.1 P.2) 
  (hQ : ellipse Q.1 Q.2) 
  (h_perp : perpendicular P Q)
  : (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) = 48 / 7 := 
sorry

end value_sq_dist_OP_OQ_l801_80174


namespace possible_values_f_zero_l801_80191

theorem possible_values_f_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = 2 * f x * f y) :
    f 0 = 0 ∨ f 0 = 1 / 2 := 
sorry

end possible_values_f_zero_l801_80191


namespace ratio_a_b_is_zero_l801_80158

-- Setting up the conditions
variables (a y b : ℝ)
variable (d : ℝ)
-- Condition for arithmetic sequence
axiom h1 : a + d = y
axiom h2 : y + d = b
axiom h3 : b + d = 3 * y

-- The Lean statement to prove
theorem ratio_a_b_is_zero (h1 : a + d = y) (h2 : y + d = b) (h3 : b + d = 3 * y) : a / b = 0 :=
sorry

end ratio_a_b_is_zero_l801_80158


namespace transformation_correctness_l801_80194

theorem transformation_correctness :
  (∀ x : ℝ, 3 * x = -4 → x = -4 / 3) ∧
  (∀ x : ℝ, 5 = 2 - x → x = -3) ∧
  (∀ x : ℝ, (x - 1) / 6 - (2 * x + 3) / 8 = 1 → 4 * (x - 1) - 3 * (2 * x + 3) = 24) ∧
  (∀ x : ℝ, 3 * x - (2 - 4 * x) = 5 → 3 * x + 4 * x - 2 = 5) :=
by
  -- Prove the given conditions
  sorry

end transformation_correctness_l801_80194


namespace percentage_not_red_roses_l801_80184

-- Definitions for the conditions
def roses : Nat := 25
def tulips : Nat := 40
def daisies : Nat := 60
def lilies : Nat := 15
def sunflowers : Nat := 10
def totalFlowers : Nat := roses + tulips + daisies + lilies + sunflowers -- 150
def redRoses : Nat := roses / 2 -- 12 (considering integer division)

-- Statement to prove
theorem percentage_not_red_roses : 
  ((totalFlowers - redRoses) * 100 / totalFlowers) = 92 := by
  sorry

end percentage_not_red_roses_l801_80184


namespace train_speed_l801_80160

theorem train_speed (L1 L2 : ℕ) (V2 : ℕ) (t : ℝ) (V1 : ℝ) : 
  L1 = 200 → 
  L2 = 280 → 
  V2 = 30 → 
  t = 23.998 → 
  (0.001 * (L1 + L2)) / (t / 3600) = V1 + V2 → 
  V1 = 42 :=
by 
  intros
  sorry

end train_speed_l801_80160


namespace inequality_proof_l801_80116

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  (1 / Real.sqrt (x + y)) + (1 / Real.sqrt (y + z)) + (1 / Real.sqrt (z + x)) ≤ 1 / Real.sqrt (2 * x * y * z) :=
by
  sorry

end inequality_proof_l801_80116


namespace mean_properties_l801_80120

theorem mean_properties (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (arith_mean : (x + y + z) / 3 = 10)
  (geom_mean : (x * y * z) ^ (1 / 3) = 6)
  (harm_mean : 3 / (1/x + 1/y + 1/z) = 2.5) :
  x^2 + y^2 + z^2 = 540 := 
sorry

end mean_properties_l801_80120


namespace Aiyanna_cookies_l801_80103

-- Define the conditions
def Alyssa_cookies : ℕ := 129
variable (x : ℕ)
def difference_condition : Prop := (Alyssa_cookies - x) = 11

-- The theorem to prove
theorem Aiyanna_cookies (x : ℕ) (h : difference_condition x) : x = 118 :=
by sorry

end Aiyanna_cookies_l801_80103


namespace james_marbles_left_l801_80198

def total_initial_marbles : Nat := 28
def marbles_in_bag_A : Nat := 4
def marbles_in_bag_B : Nat := 6
def marbles_in_bag_C : Nat := 2
def marbles_in_bag_D : Nat := 8
def marbles_in_bag_E : Nat := 4
def marbles_in_bag_F : Nat := 4

theorem james_marbles_left : total_initial_marbles - marbles_in_bag_D = 20 := by
  -- James has 28 marbles initially.
  -- He gives away Bag D which has 8 marbles.
  -- 28 - 8 = 20
  sorry

end james_marbles_left_l801_80198
