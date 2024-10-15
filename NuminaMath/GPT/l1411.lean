import Mathlib

namespace NUMINAMATH_GPT_geometric_means_insertion_l1411_141183

noncomputable def is_geometric_progression (s : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (r_pos : r > 0), ∀ n, s (n + 1) = s n * r

theorem geometric_means_insertion (s : ℕ → ℝ) (n : ℕ)
  (h : is_geometric_progression s)
  (h_pos : ∀ i, s i > 0) :
  ∃ t : ℕ → ℝ, is_geometric_progression t :=
sorry

end NUMINAMATH_GPT_geometric_means_insertion_l1411_141183


namespace NUMINAMATH_GPT_shortest_path_correct_l1411_141112

noncomputable def shortest_path_length (length width height : ℕ) : ℝ :=
  let diagonal := Real.sqrt ((length + height)^2 + width^2)
  Real.sqrt 145

theorem shortest_path_correct :
  ∀ (length width height : ℕ),
    length = 4 → width = 5 → height = 4 →
    shortest_path_length length width height = Real.sqrt 145 :=
by
  intros length width height h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_shortest_path_correct_l1411_141112


namespace NUMINAMATH_GPT_parabola_intersection_sum_l1411_141179

theorem parabola_intersection_sum : 
  ∃ x_0 y_0 : ℝ, (y_0 = x_0^2 + 15 * x_0 + 32) ∧ (x_0 = y_0^2 + 49 * y_0 + 593) ∧ (x_0 + y_0 = -33) :=
by
  sorry

end NUMINAMATH_GPT_parabola_intersection_sum_l1411_141179


namespace NUMINAMATH_GPT_second_number_is_correct_l1411_141144

theorem second_number_is_correct (A B C : ℝ) 
  (h1 : A + B + C = 157.5)
  (h2 : A / B = 14 / 17)
  (h3 : B / C = 2 / 3)
  (h4 : A - C = 12.75) : 
  B = 18.75 := 
sorry

end NUMINAMATH_GPT_second_number_is_correct_l1411_141144


namespace NUMINAMATH_GPT_base_four_to_base_ten_of_20314_eq_568_l1411_141142

-- Define what it means to convert a base-four number to base-ten
def base_four_to_base_ten (digits : List ℕ) : ℕ :=
  digits.reverse.enum.foldr (λ ⟨index, digit⟩ acc => acc + digit * 4^index) 0

-- Define the specific base-four number 20314_4 as a list of its digits
def num_20314_base_four : List ℕ := [2, 0, 3, 1, 4]

-- Theorem stating that the base-ten equivalent of 20314_4 is 568
theorem base_four_to_base_ten_of_20314_eq_568 : base_four_to_base_ten num_20314_base_four = 568 := sorry

end NUMINAMATH_GPT_base_four_to_base_ten_of_20314_eq_568_l1411_141142


namespace NUMINAMATH_GPT_find_x_l1411_141174

theorem find_x (x : ℝ) (h : 0.25 * x = 0.10 * 500 - 5) : x = 180 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1411_141174


namespace NUMINAMATH_GPT_area_of_triangle_HFG_l1411_141139

noncomputable def calculate_area_of_triangle (A B C : (ℝ × ℝ)) :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_HFG :
  let A := (0, 0)
  let B := (2, 0)
  let C := (2, 4)
  let D := (0, 4)
  let E := (2, 2)
  let F := (1, 4)
  let G := (0, 2)
  let H := ((2 + 1 + 0) / 3, (2 + 4 + 2) / 3)
  calculate_area_of_triangle H F G = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_HFG_l1411_141139


namespace NUMINAMATH_GPT_remainder_when_added_then_divided_l1411_141136

def num1 : ℕ := 2058167
def num2 : ℕ := 934
def divisor : ℕ := 8

theorem remainder_when_added_then_divided :
  (num1 + num2) % divisor = 5 := 
sorry

end NUMINAMATH_GPT_remainder_when_added_then_divided_l1411_141136


namespace NUMINAMATH_GPT_age_difference_l1411_141177

variable (Patrick_age Michael_age Monica_age : ℕ)

theorem age_difference 
  (h1 : ∃ x : ℕ, Patrick_age = 3 * x ∧ Michael_age = 5 * x)
  (h2 : ∃ y : ℕ, Michael_age = 3 * y ∧ Monica_age = 5 * y)
  (h3 : Patrick_age + Michael_age + Monica_age = 245) :
  Monica_age - Patrick_age = 80 := by 
sorry

end NUMINAMATH_GPT_age_difference_l1411_141177


namespace NUMINAMATH_GPT_all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l1411_141122

-- Conditions
def investment := 13500  -- in yuan
def total_yield := 19000 -- in kg
def price_orchard := 4   -- in yuan/kg
def price_market (x : ℝ) := x -- in yuan/kg
def market_daily_sale := 1000 -- in kg/day

-- Part 1: Days to sell all fruits in the market
theorem all_fruits_sold_in_market (x : ℝ) (h : x > 4) : total_yield / market_daily_sale = 19 :=
by
  sorry

-- Part 2: Income difference between market and orchard sales
theorem market_vs_orchard_income_diff (x : ℝ) (h : x > 4) : total_yield * price_market x - total_yield * price_orchard = 19000 * x - 76000 :=
by
  sorry

-- Part 3: Total profit from selling partly in the orchard and partly in the market
theorem total_profit (x : ℝ) (h : x > 4) : 6000 * price_orchard + (total_yield - 6000) * price_market x - investment = 13000 * x + 10500 :=
by
  sorry

end NUMINAMATH_GPT_all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l1411_141122


namespace NUMINAMATH_GPT_find_p_q_l1411_141153

variable (p q : ℝ)
def f (x : ℝ) : ℝ := x^2 + p * x + q

theorem find_p_q:
  (p, q) = (-6, 7) →
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 5) → |f p q x| ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_l1411_141153


namespace NUMINAMATH_GPT_employee_n_salary_l1411_141180

variable (m n : ℝ)

theorem employee_n_salary 
  (h1 : m + n = 605) 
  (h2 : m = 1.20 * n) : 
  n = 275 :=
by
  sorry

end NUMINAMATH_GPT_employee_n_salary_l1411_141180


namespace NUMINAMATH_GPT_range_of_m_l1411_141133

theorem range_of_m (m : ℝ) (h : 1 < m) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → -m ≤ x ∧ x ≤ m - 1) → (3 ≤ m) :=
by
  sorry  -- The proof will be constructed here.

end NUMINAMATH_GPT_range_of_m_l1411_141133


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1411_141137

noncomputable def f (a : ℝ) (x : ℝ) := 2 * Real.log x + a / x
noncomputable def g (a : ℝ) (x : ℝ) := (x / 2) * f a x - a * x^2 - x

theorem problem_part1 (a : ℝ) :
  (∀ x : ℝ, f a x = 0 → x > 0) ↔ 0 < a ∧ a < 2/Real.exp 1 := sorry

theorem problem_part2 (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : g a x₁ = 0) (h₃ : g a x₂ = 0) :
  0 < a ∧ a < 2/Real.exp 1 → Real.log x₁ + 2 * Real.log x₂ > 3 := sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1411_141137


namespace NUMINAMATH_GPT_ken_got_1750_l1411_141167

theorem ken_got_1750 (K : ℝ) (h : K + 2 * K = 5250) : K = 1750 :=
sorry

end NUMINAMATH_GPT_ken_got_1750_l1411_141167


namespace NUMINAMATH_GPT_john_loses_probability_eq_3_over_5_l1411_141108

-- Definitions used directly from the conditions in a)
def probability_win := 2 / 5
def probability_lose := 1 - probability_win

-- The theorem statement
theorem john_loses_probability_eq_3_over_5 : 
  probability_lose = 3 / 5 := 
by
  sorry -- proof is to be filled in later

end NUMINAMATH_GPT_john_loses_probability_eq_3_over_5_l1411_141108


namespace NUMINAMATH_GPT_length_of_single_row_l1411_141157

-- Define smaller cube properties and larger cube properties
def side_length_smaller_cube : ℕ := 5  -- in cm
def side_length_larger_cube : ℕ := 100  -- converted from 1 meter to cm

-- Prove that the row of smaller cubes is 400 meters long
theorem length_of_single_row :
  let num_smaller_cubes := (side_length_larger_cube / side_length_smaller_cube) ^ 3
  let length_in_cm := num_smaller_cubes * side_length_smaller_cube
  let length_in_m := length_in_cm / 100
  length_in_m = 400 :=
by
  sorry

end NUMINAMATH_GPT_length_of_single_row_l1411_141157


namespace NUMINAMATH_GPT_sarah_min_days_l1411_141165

theorem sarah_min_days (r P B : ℝ) (x : ℕ) (h_r : r = 0.1) (h_P : P = 20) (h_B : B = 60) :
  (P + r * P * x ≥ B) → (x ≥ 20) :=
by
  sorry

end NUMINAMATH_GPT_sarah_min_days_l1411_141165


namespace NUMINAMATH_GPT_shelves_of_picture_books_l1411_141131

-- Define the conditions
def n_mystery : ℕ := 5
def b_per_shelf : ℕ := 4
def b_total : ℕ := 32

-- State the main theorem to be proven
theorem shelves_of_picture_books :
  (b_total - n_mystery * b_per_shelf) / b_per_shelf = 3 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_shelves_of_picture_books_l1411_141131


namespace NUMINAMATH_GPT_prob_xi_eq_12_l1411_141143

noncomputable def prob_of_draws (total_draws red_draws : ℕ) (prob_red prob_white : ℚ) : ℚ :=
    (Nat.choose (total_draws - 1) (red_draws - 1)) * (prob_red ^ (red_draws - 1)) * (prob_white ^ (total_draws - red_draws)) * prob_red

theorem prob_xi_eq_12 :
    prob_of_draws 12 10 (3 / 8) (5 / 8) = 
    (Nat.choose 11 9) * (3 / 8)^9 * (5 / 8)^2 * (3 / 8) :=
by sorry

end NUMINAMATH_GPT_prob_xi_eq_12_l1411_141143


namespace NUMINAMATH_GPT_jenny_investment_l1411_141120

theorem jenny_investment :
  ∃ (m r : ℝ), m + r = 240000 ∧ r = 6 * m ∧ r = 205714.29 :=
by
  sorry

end NUMINAMATH_GPT_jenny_investment_l1411_141120


namespace NUMINAMATH_GPT_fraction_students_say_like_actually_dislike_l1411_141176

theorem fraction_students_say_like_actually_dislike :
  let n := 200
  let p_l := 0.70
  let p_d := 0.30
  let p_ll := 0.85
  let p_ld := 0.15
  let p_dd := 0.80
  let p_dl := 0.20
  let num_like := p_l * n
  let num_dislike := p_d * n
  let num_ll := p_ll * num_like
  let num_ld := p_ld * num_like
  let num_dd := p_dd * num_dislike
  let num_dl := p_dl * num_dislike
  let total_say_like := num_ll + num_dl
  (num_dl / total_say_like) = 12 / 131 := 
by
  sorry

end NUMINAMATH_GPT_fraction_students_say_like_actually_dislike_l1411_141176


namespace NUMINAMATH_GPT_sum_of_n_values_l1411_141161

theorem sum_of_n_values (n1 n2 : ℚ) (h1 : 3 * n1 - 8 = 5) (h2 : 3 * n2 - 8 = -5) :
  n1 + n2 = 16 / 3 := 
sorry

end NUMINAMATH_GPT_sum_of_n_values_l1411_141161


namespace NUMINAMATH_GPT_graph_location_l1411_141101

theorem graph_location (k : ℝ) (H : k > 0) :
    (∀ x : ℝ, (0 < x → 0 < y) → (y = 2/x) → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by
    sorry

end NUMINAMATH_GPT_graph_location_l1411_141101


namespace NUMINAMATH_GPT_grabbed_books_l1411_141145

-- Definitions from conditions
def initial_books : ℕ := 99
def boxed_books : ℕ := 3 * 15
def room_books : ℕ := 21
def table_books : ℕ := 4
def kitchen_books : ℕ := 18
def current_books : ℕ := 23

-- Proof statement
theorem grabbed_books : (boxed_books + room_books + table_books + kitchen_books = initial_books - (23 - current_books)) → true := sorry

end NUMINAMATH_GPT_grabbed_books_l1411_141145


namespace NUMINAMATH_GPT_support_percentage_l1411_141158

theorem support_percentage (men women : ℕ) (support_men_percentage support_women_percentage : ℝ) 
(men_support women_support total_support : ℕ)
(hmen : men = 150) 
(hwomen : women = 850) 
(hsupport_men_percentage : support_men_percentage = 0.55) 
(hsupport_women_percentage : support_women_percentage = 0.70) 
(hmen_support : men_support = 83) 
(hwomen_support : women_support = 595)
(htotal_support : total_support = men_support + women_support) :
  ((total_support : ℝ) / (men + women) * 100) = 68 :=
by
  -- Insert the proof here to verify each step of the calculation and rounding
  sorry

end NUMINAMATH_GPT_support_percentage_l1411_141158


namespace NUMINAMATH_GPT_train_speed_is_30_kmh_l1411_141194

noncomputable def speed_of_train (train_length : ℝ) (cross_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / cross_time
  let train_speed_ms := relative_speed + man_speed_ms
  train_speed_ms * (3600 / 1000)

theorem train_speed_is_30_kmh :
  speed_of_train 400 59.99520038396929 6 = 30 :=
by
  -- Using the approximation mentioned in the solution, hence no computation proof required.
  sorry

end NUMINAMATH_GPT_train_speed_is_30_kmh_l1411_141194


namespace NUMINAMATH_GPT_cone_cube_volume_ratio_l1411_141154

theorem cone_cube_volume_ratio (s : ℝ) (h : ℝ) (r : ℝ) (π : ℝ) 
  (cone_inscribed_in_cube : r = s / 2 ∧ h = s ∧ π > 0) :
  ((1/3) * π * r^2 * h) / (s^3) = π / 12 :=
by
  sorry

end NUMINAMATH_GPT_cone_cube_volume_ratio_l1411_141154


namespace NUMINAMATH_GPT_number_of_answer_choices_l1411_141166

theorem number_of_answer_choices (n : ℕ) (H1 : (n + 1)^4 = 625) : n = 4 :=
sorry

end NUMINAMATH_GPT_number_of_answer_choices_l1411_141166


namespace NUMINAMATH_GPT_range_of_m_l1411_141117

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define A as the set of real numbers satisfying 2x^2 - x = 0
def A : Set ℝ := {x | 2 * x^2 - x = 0}

-- Define B based on the parameter m as the set of real numbers satisfying mx^2 - mx - 1 = 0
def B (m : ℝ) : Set ℝ := {x | m * x^2 - m * x - 1 = 0}

-- Define the condition (¬U A) ∩ B = ∅
def condition (m : ℝ) : Prop := (U \ A) ∩ B m = ∅

theorem range_of_m : ∀ m : ℝ, condition m → -4 ≤ m ∧ m ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1411_141117


namespace NUMINAMATH_GPT_math_problem_l1411_141125

theorem math_problem :
  3 ^ (2 + 4 + 6) - (3 ^ 2 + 3 ^ 4 + 3 ^ 6) + (3 ^ 2 * 3 ^ 4 * 3 ^ 6) = 1062242 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1411_141125


namespace NUMINAMATH_GPT_ab_sum_l1411_141141

theorem ab_sum (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 7) (h3 : |a - b| = b - a) : a + b = 10 ∨ a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_ab_sum_l1411_141141


namespace NUMINAMATH_GPT_find_ratio_l1411_141151

-- Definitions and conditions
def sides_form_right_triangle (x d : ℝ) : Prop :=
  x > d ∧ d > 0 ∧ (x^2 + (x^2 - d)^2 = (x^2 + d)^2)

-- The theorem stating the required ratio
theorem find_ratio (x d : ℝ) (h : sides_form_right_triangle x d) : 
  x / d = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_ratio_l1411_141151


namespace NUMINAMATH_GPT_range_of_expression_l1411_141197

variable (a b c : ℝ)

theorem range_of_expression (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 :=
sorry

end NUMINAMATH_GPT_range_of_expression_l1411_141197


namespace NUMINAMATH_GPT_polar_to_rectangular_l1411_141170

theorem polar_to_rectangular (r θ : ℝ) (h₁ : r = 6) (h₂ : θ = Real.pi / 2) :
  (r * Real.cos θ, r * Real.sin θ) = (0, 6) :=
by
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_l1411_141170


namespace NUMINAMATH_GPT_speed_in_still_water_l1411_141149

/-- A man can row upstream at 37 km/h and downstream at 53 km/h, 
    prove that the speed of the man in still water is 45 km/h. --/
theorem speed_in_still_water 
  (upstream_speed : ℕ) 
  (downstream_speed : ℕ)
  (h1 : upstream_speed = 37)
  (h2 : downstream_speed = 53) : 
  (upstream_speed + downstream_speed) / 2 = 45 := 
by 
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l1411_141149


namespace NUMINAMATH_GPT_find_k_l1411_141134

variable (c : ℝ) (k : ℝ)
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a (n + 1) = c * a n

def sum_sequence (S : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n, S n = 3^n + k

theorem find_k (c_ne_zero : c ≠ 0)
  (h_geo : geometric_sequence a c)
  (h_sum : sum_sequence S k)
  (h_a1 : a 1 = 3 + k)
  (h_a2 : a 2 = S 2 - S 1)
  (h_a3 : a 3 = S 3 - S 2) :
  k = -1 :=
sorry

end NUMINAMATH_GPT_find_k_l1411_141134


namespace NUMINAMATH_GPT_correct_conclusion_l1411_141130

theorem correct_conclusion (x : ℝ) (hx : x > 1/2) : -2 * x + 1 < 0 :=
by
  -- sorry placeholder
  sorry

end NUMINAMATH_GPT_correct_conclusion_l1411_141130


namespace NUMINAMATH_GPT_production_time_l1411_141110

-- Define the conditions
def machineProductionRate (machines: ℕ) (units: ℕ) (hours: ℕ): ℕ := units / machines / hours

-- The question we need to answer: How long will it take for 10 machines to produce 100 units?
theorem production_time (h1 : machineProductionRate 5 20 10 = 4 / 10)
  : 10 * 0.4 * 25 = 100 :=
by sorry

end NUMINAMATH_GPT_production_time_l1411_141110


namespace NUMINAMATH_GPT_sum_of_absolute_values_of_coefficients_l1411_141113

theorem sum_of_absolute_values_of_coefficients :
  ∀ (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ),
  (∀ x : ℝ, (1 - 3 * x) ^ 9 = a + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5 + a_6 * x ^ 6 + a_7 * x ^ 7 + a_8 * x ^ 8 + a_9 * x ^ 9) →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| + |a_9| = 4 ^ 9 :=
by
  intro a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 h
  sorry

end NUMINAMATH_GPT_sum_of_absolute_values_of_coefficients_l1411_141113


namespace NUMINAMATH_GPT_concentric_circles_ratio_l1411_141156

theorem concentric_circles_ratio (R r k : ℝ) (hr : r > 0) (hRr : R > r) (hk : k > 0)
  (area_condition : π * (R^2 - r^2) = k * π * r^2) :
  R / r = Real.sqrt (k + 1) :=
by
  sorry

end NUMINAMATH_GPT_concentric_circles_ratio_l1411_141156


namespace NUMINAMATH_GPT_distribute_positions_l1411_141152

theorem distribute_positions :
  let positions := 11
  let classes := 6
  ∃ total_ways : ℕ, total_ways = Nat.choose (positions - 1) (classes - 1) ∧ total_ways = 252 :=
by
  let positions := 11
  let classes := 6
  have : Nat.choose (positions - 1) (classes - 1) = 252 := by sorry
  exact ⟨Nat.choose (positions - 1) (classes - 1), this, this⟩

end NUMINAMATH_GPT_distribute_positions_l1411_141152


namespace NUMINAMATH_GPT_interest_rate_is_4_l1411_141189

-- Define the conditions based on the problem statement
def principal : ℕ := 500
def time : ℕ := 8
def simple_interest : ℕ := 160

-- Assuming the formula for simple interest
def simple_interest_formula (P R T : ℕ) : ℕ := P * R * T / 100

-- The interest rate we aim to prove
def interest_rate : ℕ := 4

-- The statement we want to prove: Given the conditions, the interest rate is 4%
theorem interest_rate_is_4 : simple_interest_formula principal interest_rate time = simple_interest := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_interest_rate_is_4_l1411_141189


namespace NUMINAMATH_GPT_LCM_of_18_and_27_l1411_141121

theorem LCM_of_18_and_27 : Nat.lcm 18 27 = 54 := by
  sorry

end NUMINAMATH_GPT_LCM_of_18_and_27_l1411_141121


namespace NUMINAMATH_GPT_find_coefficient_b_l1411_141105

variable (a b c p : ℝ)

def parabola (x : ℝ) := a * x^2 + b * x + c

theorem find_coefficient_b (h_vertex : ∀ x, parabola a b c x = a * (x - p)^2 + p)
                           (h_y_intercept : parabola a b c 0 = -3 * p)
                           (hp_nonzero : p ≠ 0) :
  b = 8 / p :=
by
  sorry

end NUMINAMATH_GPT_find_coefficient_b_l1411_141105


namespace NUMINAMATH_GPT_decreasing_function_positive_l1411_141148

variable {f : ℝ → ℝ}

axiom decreasing (h : ℝ → ℝ) : ∀ x1 x2, x1 < x2 → h x1 > h x2

theorem decreasing_function_positive (h_decreasing: ∀ x1 x2: ℝ, x1 < x2 → f x1 > f x2)
    (h_condition: ∀ x: ℝ, f x / (deriv^[2] f x) + x < 1) :
  ∀ x : ℝ, f x > 0 := 
by
  sorry

end NUMINAMATH_GPT_decreasing_function_positive_l1411_141148


namespace NUMINAMATH_GPT_intersection_A_B_l1411_141109

def A : Set ℤ := {-2, 0, 1, 2}
def B : Set ℤ := { x | -2 ≤ x ∧ x ≤ 1 }

theorem intersection_A_B : A ∩ B = {-2, 0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1411_141109


namespace NUMINAMATH_GPT_problem_1_problem_2_l1411_141162

noncomputable def f (x : ℝ) (a : ℝ) := Real.sqrt (a - x^2)

-- First proof problem statement: 
theorem problem_1 (a : ℝ) (x : ℝ) (A B : Set ℝ) (h1 : a = 4) (h2 : A = {x : ℝ | -2 ≤ x ∧ x ≤ 2}) (h3 : B = {y : ℝ | 0 ≤ y ∧ y ≤ 2}) : 
  (A ∩ B = {y : ℝ | 0 ≤ y ∧ y ≤ 2}) :=
sorry

-- Second proof problem statement:
theorem problem_2 (a : ℝ) (h : 1 ∈ {y : ℝ | 0 ≤ y ∧ y ≤ Real.sqrt a}) : a ≥ 1 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1411_141162


namespace NUMINAMATH_GPT_arithmetic_sequence_a_eq_zero_l1411_141191

theorem arithmetic_sequence_a_eq_zero (a : ℝ) :
  (∀ n : ℕ, n > 0 → ∃ S : ℕ → ℝ, S n = (n^2 : ℝ) + 2 * n + a) →
  a = 0 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a_eq_zero_l1411_141191


namespace NUMINAMATH_GPT_find_divisor_l1411_141160

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h_dividend : dividend = 125) 
  (h_quotient : quotient = 8) 
  (h_remainder : remainder = 5) 
  (h_equation : dividend = (divisor * quotient) + remainder) : 
  divisor = 15 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l1411_141160


namespace NUMINAMATH_GPT_g_sum_even_function_l1411_141115

def g (a b c d x : ℝ) : ℝ := a * x ^ 8 + b * x ^ 6 - c * x ^ 4 + d * x ^ 2 + 5

theorem g_sum_even_function 
  (a b c d : ℝ) 
  (h : g a b c d 2 = 4)
  : g a b c d 2 + g a b c d (-2) = 8 :=
by
  sorry

end NUMINAMATH_GPT_g_sum_even_function_l1411_141115


namespace NUMINAMATH_GPT_problem_statement_l1411_141128

noncomputable def myFunction (f : ℝ → ℝ) := 
  (∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)) 

theorem problem_statement (f : ℝ → ℝ) 
  (h : myFunction f) : 
  ∀ x : ℝ, f (2005 * x) = 2005 * f x :=
sorry

end NUMINAMATH_GPT_problem_statement_l1411_141128


namespace NUMINAMATH_GPT_productivity_increase_correct_l1411_141155

def productivity_increase (that: ℝ) :=
  ∃ x : ℝ, (x + 1) * (x + 1) * 2500 = 2809

theorem productivity_increase_correct :
  productivity_increase (0.06) :=
by
  sorry

end NUMINAMATH_GPT_productivity_increase_correct_l1411_141155


namespace NUMINAMATH_GPT_denise_spent_l1411_141126

theorem denise_spent (price_simple : ℕ) (price_meat : ℕ) (price_fish : ℕ)
  (price_milk_smoothie : ℕ) (price_fruit_smoothie : ℕ) (price_special_smoothie : ℕ)
  (julio_spent_more : ℕ) :
  price_simple = 7 →
  price_meat = 11 →
  price_fish = 14 →
  price_milk_smoothie = 6 →
  price_fruit_smoothie = 7 →
  price_special_smoothie = 9 →
  julio_spent_more = 6 →
  ∃ (d_price : ℕ), (d_price = 14 ∨ d_price = 17) :=
by
  sorry

end NUMINAMATH_GPT_denise_spent_l1411_141126


namespace NUMINAMATH_GPT_first_place_beat_joe_l1411_141175

theorem first_place_beat_joe (joe_won joe_draw first_place_won first_place_draw points_win points_draw : ℕ) 
    (h1 : joe_won = 1) (h2 : joe_draw = 3) (h3 : first_place_won = 2) (h4 : first_place_draw = 2)
    (h5 : points_win = 3) (h6 : points_draw = 1) : 
    (first_place_won * points_win + first_place_draw * points_draw) - (joe_won * points_win + joe_draw * points_draw) = 2 :=
by
   sorry

end NUMINAMATH_GPT_first_place_beat_joe_l1411_141175


namespace NUMINAMATH_GPT_george_elaine_ratio_l1411_141132

-- Define the conditions
def time_jerry := 3
def time_elaine := 2 * time_jerry
def time_kramer := 0
def total_time := 11

-- Define George's time based on the given total time condition
def time_george := total_time - (time_jerry + time_elaine + time_kramer)

-- Prove the ratio of George's time to Elaine's time is 1:3
theorem george_elaine_ratio : time_george / time_elaine = 1 / 3 :=
by
  -- Lean proof would go here
  sorry

end NUMINAMATH_GPT_george_elaine_ratio_l1411_141132


namespace NUMINAMATH_GPT_find_k_value_l1411_141163

def line (k : ℝ) (x y : ℝ) : Prop := 3 - 2 * k * x = -4 * y

def on_line (k : ℝ) : Prop := line k 5 (-2)

theorem find_k_value (k : ℝ) : on_line k → k = -0.5 :=
by
  sorry

end NUMINAMATH_GPT_find_k_value_l1411_141163


namespace NUMINAMATH_GPT_circle_equation_center_at_1_2_passing_through_origin_l1411_141193

theorem circle_equation_center_at_1_2_passing_through_origin :
  ∃ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 5 ∧
                (0 - 1)^2 + (0 - 2)^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_center_at_1_2_passing_through_origin_l1411_141193


namespace NUMINAMATH_GPT_trapezoid_bases_12_and_16_l1411_141168

theorem trapezoid_bases_12_and_16 :
  ∀ (h R : ℝ) (a b : ℝ),
    (R = 10) →
    (h = (a + b) / 2) →
    (∀ k m, ((k = 3/7 * h) ∧ (m = 4/7 * h) ∧ (R^2 = k^2 + (a/2)^2) ∧ (R^2 = m^2 + (b/2)^2))) →
    (a = 12) ∧ (b = 16) :=
by
  intros h R a b hR hMid eqns
  sorry

end NUMINAMATH_GPT_trapezoid_bases_12_and_16_l1411_141168


namespace NUMINAMATH_GPT_tom_cost_cheaper_than_jane_l1411_141199

def store_A_full_price : ℝ := 125
def store_A_discount_single : ℝ := 0.08
def store_A_discount_bulk : ℝ := 0.12
def store_A_tax_rate : ℝ := 0.07
def store_A_shipping_fee : ℝ := 10
def store_A_club_discount : ℝ := 0.05

def store_B_full_price : ℝ := 130
def store_B_discount_single : ℝ := 0.10
def store_B_discount_bulk : ℝ := 0.15
def store_B_tax_rate : ℝ := 0.05
def store_B_free_shipping_threshold : ℝ := 250
def store_B_club_discount : ℝ := 0.03

def tom_smartphones_qty : ℕ := 2
def jane_smartphones_qty : ℕ := 3

theorem tom_cost_cheaper_than_jane :
  let tom_cost := 
    let total := store_A_full_price * tom_smartphones_qty
    let discount := if tom_smartphones_qty ≥ 2 then store_A_discount_bulk else store_A_discount_single
    let price_after_discount := total * (1 - discount)
    let price_after_club_discount := price_after_discount * (1 - store_A_club_discount)
    let price_after_tax := price_after_club_discount * (1 + store_A_tax_rate) 
    price_after_tax + store_A_shipping_fee

  let jane_cost := 
    let total := store_B_full_price * jane_smartphones_qty
    let discount := if jane_smartphones_qty ≥ 3 then store_B_discount_bulk else store_B_discount_single
    let price_after_discount := total * (1 - discount)
    let price_after_club_discount := price_after_discount * (1 - store_B_club_discount)
    let price_after_tax := price_after_club_discount * (1 + store_B_tax_rate)
    let shipping_fee := if total > store_B_free_shipping_threshold then 0 else 0
    price_after_tax + shipping_fee
  
  jane_cost - tom_cost = 104.01 := 
by 
  sorry

end NUMINAMATH_GPT_tom_cost_cheaper_than_jane_l1411_141199


namespace NUMINAMATH_GPT_inequality_solution_set_l1411_141185

theorem inequality_solution_set (a : ℝ) (x : ℝ) (h : (a - 1) * x > 2) : x < 2 / (a - 1) ↔ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1411_141185


namespace NUMINAMATH_GPT_intersection_M_N_l1411_141106

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 + 2 * x - 3 ≤ 0}
def intersection : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1411_141106


namespace NUMINAMATH_GPT_compare_abc_l1411_141150

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 0.3)

theorem compare_abc : a < c ∧ c < b :=
by
  -- The proof will be provided here.
  sorry

end NUMINAMATH_GPT_compare_abc_l1411_141150


namespace NUMINAMATH_GPT_initial_earning_members_l1411_141147

theorem initial_earning_members (average_income_before: ℝ) (average_income_after: ℝ) (income_deceased: ℝ) (n: ℝ)
    (H1: average_income_before = 735)
    (H2: average_income_after = 650)
    (H3: income_deceased = 990)
    (H4: n * average_income_before - (n - 1) * average_income_after = income_deceased)
    : n = 4 := 
by 
  rw [H1, H2, H3] at H4
  linarith


end NUMINAMATH_GPT_initial_earning_members_l1411_141147


namespace NUMINAMATH_GPT_max_magnitude_value_is_4_l1411_141146

noncomputable def max_value_vector_magnitude (θ : ℝ) : ℝ :=
  let a := (Real.cos θ, Real.sin θ)
  let b := (Real.sqrt 3, -1)
  let vector := (2 * a.1 - b.1, 2 * a.2 + 1)
  Real.sqrt (vector.1 ^ 2 + vector.2 ^ 2)

theorem max_magnitude_value_is_4 (θ : ℝ) : 
  ∃ θ : ℝ, max_value_vector_magnitude θ = 4 :=
sorry

end NUMINAMATH_GPT_max_magnitude_value_is_4_l1411_141146


namespace NUMINAMATH_GPT_part1_a2_part1_a3_part2_general_formula_l1411_141111

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| n + 1 => (n + 1) * n / 2

noncomputable def S (n : ℕ) : ℚ := (n + 2) * a n / 3

theorem part1_a2 : a 2 = 3 := sorry

theorem part1_a3 : a 3 = 6 := sorry

theorem part2_general_formula (n : ℕ) (h : n > 0) : a n = n * (n + 1) / 2 := sorry

end NUMINAMATH_GPT_part1_a2_part1_a3_part2_general_formula_l1411_141111


namespace NUMINAMATH_GPT_max_value_f1_solve_inequality_f2_l1411_141181

def f_1 (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem max_value_f1 : ∃ x, f_1 x = 2 :=
sorry

def f_2 (x : ℝ) : ℝ := |2 * x - 1| - |x - 1|

theorem solve_inequality_f2 (x : ℝ) : f_2 x ≥ 1 ↔ x ≤ -1 ∨ x ≥ 1 :=
sorry

end NUMINAMATH_GPT_max_value_f1_solve_inequality_f2_l1411_141181


namespace NUMINAMATH_GPT_nonnegative_integer_solutions_l1411_141171

theorem nonnegative_integer_solutions :
  {ab : ℕ × ℕ | 3 * 2^ab.1 + 1 = ab.2^2} = {(0, 2), (3, 5), (4, 7)} :=
by
  sorry

end NUMINAMATH_GPT_nonnegative_integer_solutions_l1411_141171


namespace NUMINAMATH_GPT_exists_indices_l1411_141195

theorem exists_indices (a : ℕ → ℕ) 
  (h_seq_perm : ∀ n, ∃ m, a m = n) : 
  ∃ ℓ m, 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ :=
by
  sorry

end NUMINAMATH_GPT_exists_indices_l1411_141195


namespace NUMINAMATH_GPT_total_oranges_in_buckets_l1411_141188

theorem total_oranges_in_buckets (a b c : ℕ) 
  (h1 : a = 22) 
  (h2 : b = a + 17) 
  (h3 : c = b - 11) : 
  a + b + c = 89 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_oranges_in_buckets_l1411_141188


namespace NUMINAMATH_GPT_diff_of_squares_not_2018_l1411_141140

theorem diff_of_squares_not_2018 (a b : ℕ) (h : a > b) : ¬(a^2 - b^2 = 2018) :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_diff_of_squares_not_2018_l1411_141140


namespace NUMINAMATH_GPT_find_k_value_l1411_141123

theorem find_k_value (k : ℝ) : 
  (∃ x1 x2 x3 x4 : ℝ,
    x1 ≠ 0 ∧ x2 ≠ 0 ∧ x3 ≠ 0 ∧ x4 ≠ 0 ∧
    (x1^2 - 1) * (x1^2 - 4) = k ∧
    (x2^2 - 1) * (x2^2 - 4) = k ∧
    (x3^2 - 1) * (x3^2 - 4) = k ∧
    (x4^2 - 1) * (x4^2 - 4) = k ∧
    x1 ≠ x2 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
    x4 - x3 = x3 - x2 ∧ x2 - x1 = x4 - x3) → 
  k = 7/4 := 
by
  sorry

end NUMINAMATH_GPT_find_k_value_l1411_141123


namespace NUMINAMATH_GPT_expression_evaluation_l1411_141164

theorem expression_evaluation :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
sorry

end NUMINAMATH_GPT_expression_evaluation_l1411_141164


namespace NUMINAMATH_GPT_find_stickers_before_birthday_l1411_141192

variable (stickers_received : ℕ) (total_stickers : ℕ)

def stickers_before_birthday (stickers_received total_stickers : ℕ) : ℕ :=
  total_stickers - stickers_received

theorem find_stickers_before_birthday (h1 : stickers_received = 22) (h2 : total_stickers = 61) : 
  stickers_before_birthday stickers_received total_stickers = 39 :=
by 
  have h1 : stickers_received = 22 := h1
  have h2 : total_stickers = 61 := h2
  rw [h1, h2]
  rfl

end NUMINAMATH_GPT_find_stickers_before_birthday_l1411_141192


namespace NUMINAMATH_GPT_spending_on_clothes_transport_per_month_l1411_141102

noncomputable def monthly_spending_on_clothes_transport (S : ℝ) : ℝ :=
  0.2 * S

theorem spending_on_clothes_transport_per_month :
  ∃ (S : ℝ), (monthly_spending_on_clothes_transport S = 1584) ∧
             (12 * S - (12 * 0.6 * S + 12 * monthly_spending_on_clothes_transport S) = 19008) :=
by
  sorry

end NUMINAMATH_GPT_spending_on_clothes_transport_per_month_l1411_141102


namespace NUMINAMATH_GPT_bob_weekly_income_increase_l1411_141196

theorem bob_weekly_income_increase
  (raise_per_hour : ℝ)
  (hours_per_week : ℝ)
  (benefit_reduction_per_month : ℝ)
  (weeks_per_month : ℝ)
  (h_raise : raise_per_hour = 0.50)
  (h_hours : hours_per_week = 40)
  (h_reduction : benefit_reduction_per_month = 60)
  (h_weeks : weeks_per_month = 4.33) :
  (raise_per_hour * hours_per_week - benefit_reduction_per_month / weeks_per_month) = 6.14 :=
by
  simp [h_raise, h_hours, h_reduction, h_weeks]
  norm_num
  sorry

end NUMINAMATH_GPT_bob_weekly_income_increase_l1411_141196


namespace NUMINAMATH_GPT_total_birds_distance_l1411_141124

def birds_flew_collectively : Prop := 
  let distance_eagle := 15 * 2.5
  let distance_falcon := 46 * 2.5
  let distance_pelican := 33 * 2.5
  let distance_hummingbird := 30 * 2.5
  let distance_hawk := 45 * 3
  let distance_swallow := 25 * 1.5
  let total_distance := distance_eagle + distance_falcon + distance_pelican + distance_hummingbird + distance_hawk + distance_swallow
  total_distance = 482.5

theorem total_birds_distance : birds_flew_collectively := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_birds_distance_l1411_141124


namespace NUMINAMATH_GPT_find_pairs_l1411_141118

theorem find_pairs (x y : ℕ) (h1 : 0 < x ∧ 0 < y)
  (h2 : ∃ p : ℕ, Prime p ∧ (x + y = 2 * p))
  (h3 : (x! + y!) % (x + y) = 0) : ∃ p : ℕ, Prime p ∧ x = p ∧ y = p :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l1411_141118


namespace NUMINAMATH_GPT_constant_term_in_expansion_l1411_141173

theorem constant_term_in_expansion : 
  let a := (x : ℝ)
  let b := - (2 / Real.sqrt x)
  let n := 6
  let general_term (r : Nat) : ℝ := Nat.choose n r * a * (b ^ (n - r))
  (∀ x : ℝ, ∃ (r : Nat), r = 4 ∧ (1 - (n - r) / 2 = 0) →
  general_term 4 = 60) :=
by
  sorry

end NUMINAMATH_GPT_constant_term_in_expansion_l1411_141173


namespace NUMINAMATH_GPT_find_m_parallel_l1411_141178

def vector_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k • v) ∨ v = (k • u)

theorem find_m_parallel (m : ℝ) (a b : ℝ × ℝ) (h_a : a = (-1, 1)) (h_b : b = (3, m)) 
  (h_parallel : vector_parallel a (a.1 + b.1, a.2 + b.2)) : m = -3 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_parallel_l1411_141178


namespace NUMINAMATH_GPT_find_multiple_l1411_141114

theorem find_multiple (x m : ℕ) (h₁ : x = 69) (h₂ : x - 18 = m * (86 - x)) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l1411_141114


namespace NUMINAMATH_GPT_geometric_series_sum_l1411_141187

theorem geometric_series_sum :
  ∀ (a r : ℤ) (n : ℕ),
  a = 3 → r = -2 → n = 10 →
  (a * ((r ^ n - 1) / (r - 1))) = -1024 :=
by
  intros a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1411_141187


namespace NUMINAMATH_GPT_max_value_f_l1411_141129

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x - Real.tan x

theorem max_value_f : 
  ∃ x ∈ Set.Ioo 0 (Real.pi / 2), ∀ y ∈ Set.Ioo 0 (Real.pi / 2), f y ≤ f x ∧ f x = 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_max_value_f_l1411_141129


namespace NUMINAMATH_GPT_grazing_time_for_36_cows_l1411_141103

-- Defining the problem conditions and the question in Lean 4
theorem grazing_time_for_36_cows :
  ∀ (g r b : ℕ), 
    (24 * 6 * b = g + 6 * r) →
    (21 * 8 * b = g + 8 * r) →
    36 * 3 * b = g + 3 * r :=
by
  intros
  sorry

end NUMINAMATH_GPT_grazing_time_for_36_cows_l1411_141103


namespace NUMINAMATH_GPT_find_first_hour_speed_l1411_141138

variable (x : ℝ)

-- Conditions
def speed_second_hour : ℝ := 60
def average_speed_two_hours : ℝ := 102.5

theorem find_first_hour_speed (h1 : average_speed_two_hours = (x + speed_second_hour) / 2) : 
  x = 145 := 
by
  sorry

end NUMINAMATH_GPT_find_first_hour_speed_l1411_141138


namespace NUMINAMATH_GPT_david_recreation_l1411_141119

theorem david_recreation (W : ℝ) (P : ℝ) 
  (h1 : 0.95 * W = this_week_wages) 
  (h2 : 0.5 * this_week_wages = recreation_this_week)
  (h3 : 1.1875 * (P / 100) * W = recreation_this_week) : P = 40 :=
sorry

end NUMINAMATH_GPT_david_recreation_l1411_141119


namespace NUMINAMATH_GPT_total_distance_is_1095_l1411_141198

noncomputable def totalDistanceCovered : ℕ :=
  let running_first_3_months := 3 * 3 * 10
  let running_next_3_months := 3 * 3 * 20
  let running_last_6_months := 3 * 6 * 30
  let total_running := running_first_3_months + running_next_3_months + running_last_6_months

  let swimming_first_6_months := 3 * 6 * 5
  let total_swimming := swimming_first_6_months

  let total_hiking := 13 * 15

  total_running + total_swimming + total_hiking

theorem total_distance_is_1095 : totalDistanceCovered = 1095 := by
  sorry

end NUMINAMATH_GPT_total_distance_is_1095_l1411_141198


namespace NUMINAMATH_GPT_number_of_matching_pages_l1411_141107

theorem number_of_matching_pages : 
  ∃ (n : Nat), n = 13 ∧ ∀ x, 1 ≤ x ∧ x ≤ 63 → (x % 10 = (64 - x) % 10) ↔ x % 10 = 2 ∨ x % 10 = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_matching_pages_l1411_141107


namespace NUMINAMATH_GPT_compare_neg_two_and_neg_one_l1411_141169

theorem compare_neg_two_and_neg_one : -2 < -1 :=
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_compare_neg_two_and_neg_one_l1411_141169


namespace NUMINAMATH_GPT_find_x_l1411_141182

theorem find_x (x : ℝ) (hx : x > 0) (h : Real.sqrt (12*x) * Real.sqrt (5*x) * Real.sqrt (7*x) * Real.sqrt (21*x) = 21) : 
  x = 21 / 97 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1411_141182


namespace NUMINAMATH_GPT_false_proposition_l1411_141127

open Classical

variables (a b : ℝ) (x : ℝ)

def P := ∃ (a b : ℝ), (0 < a) ∧ (0 < b) ∧ (a + b = 1) ∧ ((1 / a) + (1 / b) = 3)
def Q := ∀ (x : ℝ), x^2 - x + 1 ≥ 0

theorem false_proposition :
  (¬ P ∧ ¬ Q) = false → (¬ P ∨ ¬ Q) = true → (¬ P ∨ Q) = true → (¬ P ∧ Q) = true :=
sorry

end NUMINAMATH_GPT_false_proposition_l1411_141127


namespace NUMINAMATH_GPT_find_m_l1411_141186

theorem find_m (m : ℝ) (h : |m| = |m + 2|) : m = -1 :=
sorry

end NUMINAMATH_GPT_find_m_l1411_141186


namespace NUMINAMATH_GPT_solve_for_x_l1411_141116

-- Lean 4 statement for the problem
theorem solve_for_x (x : ℝ) (h : (x + 1)^3 = -27) : x = -4 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1411_141116


namespace NUMINAMATH_GPT_players_quit_game_l1411_141184

variable (total_players initial num_lives players_left players_quit : Nat)
variable (each_player_lives : Nat)

theorem players_quit_game :
  (initial = 8) →
  (each_player_lives = 3) →
  (num_lives = 15) →
  players_left = num_lives / each_player_lives →
  players_quit = initial - players_left →
  players_quit = 3 :=
by
  intros h_initial h_each_player_lives h_num_lives h_players_left h_players_quit
  sorry

end NUMINAMATH_GPT_players_quit_game_l1411_141184


namespace NUMINAMATH_GPT_tamia_bell_pepper_pieces_l1411_141104

theorem tamia_bell_pepper_pieces :
  let bell_peppers := 5
  let slices_per_pepper := 20
  let initial_slices := bell_peppers * slices_per_pepper
  let half_slices_cut := initial_slices / 2
  let small_pieces := half_slices_cut * 3
  let total_pieces := (initial_slices - half_slices_cut) + small_pieces
  total_pieces = 200 :=
by
  sorry

end NUMINAMATH_GPT_tamia_bell_pepper_pieces_l1411_141104


namespace NUMINAMATH_GPT_complement_intersection_l1411_141100

open Set

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5, 6})
variable (M_def : M = {2, 3})
variable (N_def : N = {1, 4})

theorem complement_intersection (U M N : Set ℕ) (U_def : U = {1, 2, 3, 4, 5, 6}) (M_def : M = {2, 3}) (N_def : N = {1, 4}) :
  (U \ M) ∩ (U \ N) = {5, 6} := by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1411_141100


namespace NUMINAMATH_GPT_num_integers_contains_3_and_4_l1411_141159

theorem num_integers_contains_3_and_4 
  (n : ℕ) (h1 : 500 ≤ n) (h2 : n < 1000) :
  (∀ a b c : ℕ, n = 100 * a + 10 * b + c → (b = 3 ∧ c = 4) ∨ (b = 4 ∧ c = 3)) → 
  n = 10 :=
sorry

end NUMINAMATH_GPT_num_integers_contains_3_and_4_l1411_141159


namespace NUMINAMATH_GPT_n_squared_sum_of_squares_l1411_141190

theorem n_squared_sum_of_squares (n a b c : ℕ) (h : n = a^2 + b^2 + c^2) : 
  ∃ x y z : ℕ, n^2 = x^2 + y^2 + z^2 :=
by 
  sorry

end NUMINAMATH_GPT_n_squared_sum_of_squares_l1411_141190


namespace NUMINAMATH_GPT_zero_is_a_root_of_polynomial_l1411_141135

theorem zero_is_a_root_of_polynomial :
  (12 * (0 : ℝ)^4 + 38 * (0)^3 - 51 * (0)^2 + 40 * (0) = 0) :=
by simp

end NUMINAMATH_GPT_zero_is_a_root_of_polynomial_l1411_141135


namespace NUMINAMATH_GPT_numLinesTangentToCircles_eq_2_l1411_141172

noncomputable def lineTangents (A B : Point) (dAB rA rB : ℝ) : ℕ :=
  if dAB < rA + rB then 2 else 0

theorem numLinesTangentToCircles_eq_2
  (A B : Point) (dAB rA rB : ℝ)
  (hAB : dAB = 4) (hA : rA = 3) (hB : rB = 2) :
  lineTangents A B dAB rA rB = 2 := by
  sorry

end NUMINAMATH_GPT_numLinesTangentToCircles_eq_2_l1411_141172
