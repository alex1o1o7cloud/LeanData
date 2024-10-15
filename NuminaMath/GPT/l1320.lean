import Mathlib

namespace NUMINAMATH_GPT_lambda_inequality_l1320_132094

-- Define the problem hypothesis and conclusion
theorem lambda_inequality (n : ℕ) (hn : n ≥ 4) (lambda_n : ℝ) :
  lambda_n ≥ 2 * Real.sin ((n-2) * Real.pi / (2 * n)) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_lambda_inequality_l1320_132094


namespace NUMINAMATH_GPT_boat_travel_distance_downstream_l1320_132024

-- Definitions of the given conditions
def boatSpeedStillWater : ℕ := 10 -- km/hr
def streamSpeed : ℕ := 8 -- km/hr
def timeDownstream : ℕ := 3 -- hours

-- Effective speed downstream
def effectiveSpeedDownstream : ℕ := boatSpeedStillWater + streamSpeed

-- Goal: Distance traveled downstream equals 54 km
theorem boat_travel_distance_downstream :
  effectiveSpeedDownstream * timeDownstream = 54 := 
by
  -- Since only the statement is needed, we use sorry to indicate the proof is skipped
  sorry

end NUMINAMATH_GPT_boat_travel_distance_downstream_l1320_132024


namespace NUMINAMATH_GPT_imaginary_part_z1z2_l1320_132066

open Complex

-- Define the complex numbers z1 and z2
def z1 : ℂ := (1 : ℂ) - I
def z2 : ℂ := (2 : ℂ) + 4 * I

-- Define the product of z1 and z2
def z1z2 : ℂ := z1 * z2

-- State the theorem that the imaginary part of z1z2 is 2
theorem imaginary_part_z1z2 : z1z2.im = 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_imaginary_part_z1z2_l1320_132066


namespace NUMINAMATH_GPT_at_least_240_students_l1320_132020

-- Define the total number of students
def total_students : ℕ := 1200

-- Define the 80th percentile score
def percentile_80_score : ℕ := 103

-- Define the number of students below the 80th percentile
def students_below_80th_percentile : ℕ := total_students * 80 / 100

-- Define the number of students with at least the 80th percentile score
def students_at_least_80th_percentile : ℕ := total_students - students_below_80th_percentile

-- The theorem to prove
theorem at_least_240_students : students_at_least_80th_percentile ≥ 240 :=
by
  -- Placeholder proof, to be filled in as the actual proof
  sorry

end NUMINAMATH_GPT_at_least_240_students_l1320_132020


namespace NUMINAMATH_GPT_ellipse_product_l1320_132017

noncomputable def a (b : ℝ) := b + 4
noncomputable def AB (a: ℝ) := 2 * a
noncomputable def CD (b: ℝ) := 2 * b

theorem ellipse_product:
  (∀ (a b : ℝ), a = b + 4 → a^2 - b^2 = 64) →
  (∃ (a b : ℝ), (AB a) * (CD b) = 240) :=
by
  intros h
  use 10, 6
  simp [AB, CD]
  sorry

end NUMINAMATH_GPT_ellipse_product_l1320_132017


namespace NUMINAMATH_GPT_coupon_percentage_l1320_132023

theorem coupon_percentage (P i d final_price total_price discount_amount percentage: ℝ)
  (h1 : P = 54) (h2 : i = 20) (h3 : d = 0.20 * i) 
  (h4 : total_price = P - d) (h5 : final_price = 45) 
  (h6 : discount_amount = total_price - final_price) 
  (h7 : percentage = (discount_amount / total_price) * 100) : 
  percentage = 10 := 
by
  sorry

end NUMINAMATH_GPT_coupon_percentage_l1320_132023


namespace NUMINAMATH_GPT_gcd_2024_2048_l1320_132064

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := 
by
  sorry

end NUMINAMATH_GPT_gcd_2024_2048_l1320_132064


namespace NUMINAMATH_GPT_option_d_not_equal_four_thirds_l1320_132073

theorem option_d_not_equal_four_thirds :
  1 + (2 / 7) ≠ 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_option_d_not_equal_four_thirds_l1320_132073


namespace NUMINAMATH_GPT_root_expression_l1320_132065

theorem root_expression {p q x1 x2 : ℝ}
  (h1 : x1^2 + p * x1 + q = 0)
  (h2 : x2^2 + p * x2 + q = 0) :
  (x1 / x2 + x2 / x1) = (p^2 - 2 * q) / q :=
by {
  sorry
}

end NUMINAMATH_GPT_root_expression_l1320_132065


namespace NUMINAMATH_GPT_evaluate_expression_l1320_132038

theorem evaluate_expression : (↑7 ^ (1/4) / ↑7 ^ (1/6)) = (↑7 ^ (1/12)) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1320_132038


namespace NUMINAMATH_GPT_problem_statement_l1320_132054

theorem problem_statement (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + 2 * c) + b / (c + 2 * a) + c / (a + 2 * b) > 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1320_132054


namespace NUMINAMATH_GPT_hyperbola_vertex_distance_l1320_132063

theorem hyperbola_vertex_distance (a b : ℝ) (h_eq : a^2 = 16) (hyperbola_eq : ∀ x y : ℝ, 
  (x^2 / 16) - (y^2 / 9) = 1) : 
  (2 * a) = 8 :=
by
  have h_a : a = 4 := by sorry
  rw [h_a]
  norm_num

end NUMINAMATH_GPT_hyperbola_vertex_distance_l1320_132063


namespace NUMINAMATH_GPT_min_seats_to_occupy_l1320_132036

theorem min_seats_to_occupy (n : ℕ) (h_n : n = 150) : 
  ∃ (k : ℕ), k = 90 ∧ ∀ m : ℕ, m ≥ k → ∀ i : ℕ, i < n → ∃ j : ℕ, (j < n) ∧ ((j = i + 1) ∨ (j = i - 1)) :=
sorry

end NUMINAMATH_GPT_min_seats_to_occupy_l1320_132036


namespace NUMINAMATH_GPT_suitable_for_lottery_method_B_l1320_132030

def total_items_A : Nat := 3000
def samples_A : Nat := 600

def total_items_B (n: Nat) : Nat := 2 * 15
def samples_B : Nat := 6

def total_items_C : Nat := 2 * 15
def samples_C : Nat := 6

def total_items_D : Nat := 3000
def samples_D : Nat := 10

def is_lottery_suitable (total_items : Nat) (samples : Nat) (different_factories : Bool) : Bool :=
  total_items <= 30 && samples <= total_items && !different_factories

theorem suitable_for_lottery_method_B : 
  is_lottery_suitable (total_items_B 2) samples_B false = true :=
  sorry

end NUMINAMATH_GPT_suitable_for_lottery_method_B_l1320_132030


namespace NUMINAMATH_GPT_average_visitors_per_day_l1320_132053

theorem average_visitors_per_day (average_sunday : ℕ) (average_other : ℕ) (days_in_month : ℕ) (begins_with_sunday : Bool) :
  average_sunday = 600 → average_other = 240 → days_in_month = 30 → begins_with_sunday = true → (8640 / 30 = 288) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_average_visitors_per_day_l1320_132053


namespace NUMINAMATH_GPT_func1_max_min_func2_max_min_l1320_132034

noncomputable def func1 (x : ℝ) : ℝ := 2 * Real.sin x - 3
noncomputable def func2 (x : ℝ) : ℝ := (7/4 : ℝ) + Real.sin x - (Real.sin x) ^ 2

theorem func1_max_min : (∀ x : ℝ, func1 x ≤ -1) ∧ (∃ x : ℝ, func1 x = -1) ∧ (∀ x : ℝ, func1 x ≥ -5) ∧ (∃ x : ℝ, func1 x = -5)  :=
by
  sorry

theorem func2_max_min : (∀ x : ℝ, func2 x ≤ 2) ∧ (∃ x : ℝ, func2 x = 2) ∧ (∀ x : ℝ, func2 x ≥ 7 / 4) ∧ (∃ x : ℝ, func2 x = 7 / 4) :=
by
  sorry

end NUMINAMATH_GPT_func1_max_min_func2_max_min_l1320_132034


namespace NUMINAMATH_GPT_inequality_of_positive_reals_l1320_132069

variable {a b c : ℝ}

theorem inequality_of_positive_reals (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_of_positive_reals_l1320_132069


namespace NUMINAMATH_GPT_bond_yield_correct_l1320_132000

-- Definitions of the conditions
def number_of_bonds : ℕ := 1000
def holding_period : ℕ := 2
def bond_income : ℚ := 980 - 980 + 1000 * 0.07 * 2
def initial_investment : ℚ := 980000

-- Yield for 2 years
def yield_2_years : ℚ := (number_of_bonds * bond_income) / initial_investment * 100

-- Average annual yield
def avg_annual_yield : ℚ := yield_2_years / holding_period

-- The main theorem to prove
theorem bond_yield_correct :
  yield_2_years = 15.31 ∧ avg_annual_yield = 7.65 :=
by
  sorry

end NUMINAMATH_GPT_bond_yield_correct_l1320_132000


namespace NUMINAMATH_GPT_find_linear_function_l1320_132026

theorem find_linear_function (α : ℝ) (hα : α > 0)
  (f : ℕ+ → ℝ)
  (h : ∀ (k m : ℕ+), α * (m : ℝ) ≤ (k : ℝ) ∧ (k : ℝ) < (α + 1) * (m : ℝ) → f (k + m) = f k + f m)
: ∃ (b : ℝ), ∀ (n : ℕ+), f n = b * (n : ℝ) :=
sorry

end NUMINAMATH_GPT_find_linear_function_l1320_132026


namespace NUMINAMATH_GPT_exists_line_l_l1320_132062

-- Define the parabola and line l1
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 8 * P.1
def line_l1 (P : ℝ × ℝ) : Prop := P.1 + 5 * P.2 - 5 = 0

-- Define the problem statement
theorem exists_line_l :
  ∃ l : ℝ × ℝ → Prop, 
    ((∃ A B : ℝ × ℝ, parabola A ∧ parabola B ∧ A ≠ B ∧ l A ∧ l B) ∧
    (∃ M : ℝ × ℝ, M = (1, 4/5) ∧ line_l1 M) ∧
    (∀ A B : ℝ × ℝ, l A ∧ l B → (A.2 - B.2) / (A.1 - B.1) = 5)) ∧
    (∀ P : ℝ × ℝ, l P ↔ 25 * P.1 - 5 * P.2 - 21 = 0) :=
sorry

end NUMINAMATH_GPT_exists_line_l_l1320_132062


namespace NUMINAMATH_GPT_sea_creatures_lost_l1320_132058

theorem sea_creatures_lost (sea_stars seashells snails items_left : ℕ) 
  (h1 : sea_stars = 34) 
  (h2 : seashells = 21) 
  (h3 : snails = 29) 
  (h4 : items_left = 59) : 
  sea_stars + seashells + snails - items_left = 25 :=
by
  sorry

end NUMINAMATH_GPT_sea_creatures_lost_l1320_132058


namespace NUMINAMATH_GPT_coeff_x6_in_expansion_l1320_132068

theorem coeff_x6_in_expansion : 
  (Polynomial.coeff ((1 - 3 * Polynomial.X ^ 3) ^ 7 : Polynomial ℤ) 6) = 189 :=
by
  sorry

end NUMINAMATH_GPT_coeff_x6_in_expansion_l1320_132068


namespace NUMINAMATH_GPT_initial_population_first_village_equals_l1320_132096

-- Definitions of the conditions
def initial_population_second_village : ℕ := 42000
def decrease_first_village_per_year : ℕ := 1200
def increase_second_village_per_year : ℕ := 800
def years : ℕ := 13

-- Proposition we want to prove
/-- The initial population of the first village such that both villages have the same population after 13 years. -/
theorem initial_population_first_village_equals :
  ∃ (P : ℕ), (P - decrease_first_village_per_year * years) = (initial_population_second_village + increase_second_village_per_year * years) 
  := sorry

end NUMINAMATH_GPT_initial_population_first_village_equals_l1320_132096


namespace NUMINAMATH_GPT_marathon_yards_l1320_132040

theorem marathon_yards (miles_per_marathon : ℕ) (extra_yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ)
  (total_miles : ℕ) (total_yards : ℕ) 
  (H1 : miles_per_marathon = 26) 
  (H2 : extra_yards_per_marathon = 395) 
  (H3 : yards_per_mile = 1760) 
  (H4 : num_marathons = 15) 
  (H5 : total_miles = num_marathons * miles_per_marathon + (num_marathons * extra_yards_per_marathon) / yards_per_mile)
  (H6 : total_yards = (num_marathons * extra_yards_per_marathon) % yards_per_mile)
  (H7 : 0 ≤ total_yards ∧ total_yards < yards_per_mile) 
  : total_yards = 645 :=
sorry

end NUMINAMATH_GPT_marathon_yards_l1320_132040


namespace NUMINAMATH_GPT_range_of_a_l1320_132091

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2 * x - 3 > 0 → x > a) ↔ a ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1320_132091


namespace NUMINAMATH_GPT_skirt_more_than_pants_l1320_132097

def amount_cut_off_skirt : ℝ := 0.75
def amount_cut_off_pants : ℝ := 0.5

theorem skirt_more_than_pants : 
  amount_cut_off_skirt - amount_cut_off_pants = 0.25 := 
by
  sorry

end NUMINAMATH_GPT_skirt_more_than_pants_l1320_132097


namespace NUMINAMATH_GPT_cookies_per_box_correct_l1320_132081

variable (cookies_per_box : ℕ)

-- Define the conditions
def morning_cookie : ℕ := 1 / 2
def bed_cookie : ℕ := 1 / 2
def day_cookies : ℕ := 2
def daily_cookies := morning_cookie + bed_cookie + day_cookies

def days : ℕ := 30
def total_cookies := days * daily_cookies

def boxes : ℕ := 2
def total_cookies_in_boxes : ℕ := cookies_per_box * boxes

-- Theorem we want to prove
theorem cookies_per_box_correct :
  total_cookies_in_boxes = 90 → cookies_per_box = 45 :=
by
  sorry

end NUMINAMATH_GPT_cookies_per_box_correct_l1320_132081


namespace NUMINAMATH_GPT_polynomial_factorization_l1320_132060

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end NUMINAMATH_GPT_polynomial_factorization_l1320_132060


namespace NUMINAMATH_GPT_problem1_problem2_l1320_132089

noncomputable def f (x : Real) : Real := 
  let a := (2 * Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
  let b := (Real.cos x, 1)
  a.1 * b.1 + a.2 * b.2

theorem problem1 (x : Real) : 
  ∃ k : Int, - Real.pi / 3 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + k * Real.pi :=
  sorry

theorem problem2 (A B C a b c : Real)
  (h1 : a = Real.sqrt 7)
  (h2 : Real.sin B = 2 * Real.sin C)
  (h3 : f A = 2)
  : (∃ area : Real, area = (7 * Real.sqrt 3) / 6) :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1320_132089


namespace NUMINAMATH_GPT_factor_in_range_l1320_132072

-- Define the given constants
def a : ℕ := 201212200619
def lower_bound : ℕ := 6000000000
def upper_bound : ℕ := 6500000000
def m : ℕ := 6490716149

-- The Lean proof statement
theorem factor_in_range :
  m ∣ a ∧ lower_bound < m ∧ m < upper_bound :=
by
  exact ⟨sorry, sorry, sorry⟩

end NUMINAMATH_GPT_factor_in_range_l1320_132072


namespace NUMINAMATH_GPT_maximize_exponential_sum_l1320_132084

theorem maximize_exponential_sum (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 ≤ 4) : 
  e^a + e^b + e^c + e^d ≤ 4 * Real.exp 1 := 
sorry

end NUMINAMATH_GPT_maximize_exponential_sum_l1320_132084


namespace NUMINAMATH_GPT_vectors_perpendicular_l1320_132074

def vec (a b : ℝ) := (a, b)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

@[simp]
def a := vec (-1) 2
@[simp]
def b := vec 1 3

theorem vectors_perpendicular :
  dot_product a (vector_sub a b) = 0 := by
  sorry

end NUMINAMATH_GPT_vectors_perpendicular_l1320_132074


namespace NUMINAMATH_GPT_max_product_of_functions_l1320_132032

theorem max_product_of_functions (f h : ℝ → ℝ) (hf : ∀ x, -5 ≤ f x ∧ f x ≤ 3) (hh : ∀ x, -3 ≤ h x ∧ h x ≤ 4) :
  ∃ x, f x * h x = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_product_of_functions_l1320_132032


namespace NUMINAMATH_GPT_two_a7_minus_a8_l1320_132004

variable (a : ℕ → ℝ) -- Assuming the arithmetic sequence {a_n} is a sequence of real numbers

-- Definitions and conditions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

axiom a1_plus_3a6_plus_a11 : a 1 + 3 * (a 6) + a 11 = 120

-- The theorem to be proved
theorem two_a7_minus_a8 (h : is_arithmetic_sequence a) : 2 * a 7 - a 8 = 24 := 
sorry

end NUMINAMATH_GPT_two_a7_minus_a8_l1320_132004


namespace NUMINAMATH_GPT_relative_value_ex1_max_value_of_m_plus_n_l1320_132092

-- Definition of relative relationship value
def relative_relationship_value (a b n : ℚ) : ℚ := abs (a - n) + abs (b - n)

-- First problem statement
theorem relative_value_ex1 : relative_relationship_value 2 (-5) 2 = 7 := by
  sorry

-- Second problem statement: maximum value of m + n given the relative relationship value is 2
theorem max_value_of_m_plus_n (m n : ℚ) (h : relative_relationship_value m n 2 = 2) : m + n ≤ 6 := by
  sorry

end NUMINAMATH_GPT_relative_value_ex1_max_value_of_m_plus_n_l1320_132092


namespace NUMINAMATH_GPT_min_value_expression_l1320_132087

theorem min_value_expression (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1320_132087


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l1320_132016

theorem number_of_ordered_pairs :
  ∃ n : ℕ, n = 89 ∧ (∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x < y ∧ 2 * x * y = 8 ^ 30 * (x + y)) := sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l1320_132016


namespace NUMINAMATH_GPT_max_cables_cut_l1320_132056

/-- 
Prove that given 200 computers connected by 345 cables initially forming a single cluster, after 
cutting cables to form 8 clusters, the maximum possible number of cables that could have been 
cut is 153.
--/
theorem max_cables_cut (computers : ℕ) (initial_cables : ℕ) (final_clusters : ℕ) (initial_clusters : ℕ) 
  (minimal_cables : ℕ) (cuts : ℕ) : 
  computers = 200 ∧ initial_cables = 345 ∧ final_clusters = 8 ∧ initial_clusters = 1 ∧ 
  minimal_cables = computers - final_clusters ∧ 
  cuts = initial_cables - minimal_cables →
  cuts = 153 := 
sorry

end NUMINAMATH_GPT_max_cables_cut_l1320_132056


namespace NUMINAMATH_GPT_greatest_divisor_consistent_remainder_l1320_132018

noncomputable def gcd_of_differences : ℕ :=
  Nat.gcd (Nat.gcd 1050 28770) 71670

theorem greatest_divisor_consistent_remainder :
  gcd_of_differences = 30 :=
by
  -- The proof can be filled in here...
  sorry

end NUMINAMATH_GPT_greatest_divisor_consistent_remainder_l1320_132018


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l1320_132076

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x - y = 9) : y = 3 * x - 9 := 
by
  sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_l1320_132076


namespace NUMINAMATH_GPT_gold_coins_count_l1320_132059

theorem gold_coins_count (n c : ℕ) (h1 : n = 8 * (c - 3))
                                     (h2 : n = 5 * c + 4)
                                     (h3 : c ≥ 10) : n = 54 :=
by
  sorry

end NUMINAMATH_GPT_gold_coins_count_l1320_132059


namespace NUMINAMATH_GPT_largest_number_Ahn_can_get_l1320_132031

theorem largest_number_Ahn_can_get :
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (100 ≤ m ∧ m ≤ 999) → 3 * (500 - m) ≤ 1200) := sorry

end NUMINAMATH_GPT_largest_number_Ahn_can_get_l1320_132031


namespace NUMINAMATH_GPT_evening_temperature_is_correct_l1320_132067

-- Define the temperatures at noon and in the evening
def T_noon : ℤ := 3
def T_evening : ℤ := -2

-- State the theorem to prove
theorem evening_temperature_is_correct : T_evening = -2 := by
  sorry

end NUMINAMATH_GPT_evening_temperature_is_correct_l1320_132067


namespace NUMINAMATH_GPT_angelina_speed_from_library_to_gym_l1320_132090

theorem angelina_speed_from_library_to_gym :
  ∃ (v : ℝ), 
    (840 / v - 510 / (1.5 * v) = 40) ∧
    (510 / (1.5 * v) - 480 / (2 * v) = 20) ∧
    (2 * v = 25) :=
by
  sorry

end NUMINAMATH_GPT_angelina_speed_from_library_to_gym_l1320_132090


namespace NUMINAMATH_GPT_max_a_plus_2b_plus_c_l1320_132043

open Real

theorem max_a_plus_2b_plus_c
  (A : Set ℝ := {x | |x + 1| ≤ 4})
  (T : ℝ := 3)
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_T : a^2 + b^2 + c^2 = T) :
  a + 2 * b + c ≤ 3 * sqrt 2 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_max_a_plus_2b_plus_c_l1320_132043


namespace NUMINAMATH_GPT_find_function_expression_l1320_132008

noncomputable def f (x : ℝ) : ℝ := x^2 - 5*x + 7

theorem find_function_expression (x : ℝ) :
  (∀ x : ℝ, f (x + 2) = x^2 - x + 1) →
  f x = x^2 - 5*x + 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_function_expression_l1320_132008


namespace NUMINAMATH_GPT_purely_imaginary_solution_l1320_132025

noncomputable def complex_number_is_purely_imaginary (m : ℝ) : Prop :=
  (m^2 - 2 * m - 3 = 0) ∧ (m + 1 ≠ 0)

theorem purely_imaginary_solution (m : ℝ) (h : complex_number_is_purely_imaginary m) : m = 3 := by
  sorry

end NUMINAMATH_GPT_purely_imaginary_solution_l1320_132025


namespace NUMINAMATH_GPT_gcd_1722_966_l1320_132035

theorem gcd_1722_966 : Nat.gcd 1722 966 = 42 :=
  sorry

end NUMINAMATH_GPT_gcd_1722_966_l1320_132035


namespace NUMINAMATH_GPT_average_first_15_nat_l1320_132075

-- Define the sequence and necessary conditions
def sum_first_n_nat (n : ℕ) : ℕ := n * (n + 1) / 2

theorem average_first_15_nat : (sum_first_n_nat 15) / 15 = 8 := 
by 
  -- Here we shall place the proof to show the above statement holds true
  sorry

end NUMINAMATH_GPT_average_first_15_nat_l1320_132075


namespace NUMINAMATH_GPT_mila_social_media_hours_l1320_132046

/-- 
Mila spends 6 hours on his phone every day. 
Half of this time is spent on social media. 
Prove that Mila spends 21 hours on social media in a week.
-/
theorem mila_social_media_hours 
  (hours_per_day : ℕ)
  (phone_time_per_day : hours_per_day = 6)
  (daily_social_media_fraction : ℕ)
  (fractional_time : daily_social_media_fraction = hours_per_day / 2)
  (days_per_week : ℕ)
  (days_in_week : days_per_week = 7) :
  (daily_social_media_fraction * days_per_week = 21) :=
sorry

end NUMINAMATH_GPT_mila_social_media_hours_l1320_132046


namespace NUMINAMATH_GPT_base_problem_l1320_132005

theorem base_problem (c d : Nat) (pos_c : c > 0) (pos_d : d > 0) (h : 5 * c + 8 = 8 * d + 5) : c + d = 15 :=
sorry

end NUMINAMATH_GPT_base_problem_l1320_132005


namespace NUMINAMATH_GPT_pizza_slices_left_l1320_132044

theorem pizza_slices_left (total_slices john_ate : ℕ) 
  (initial_slices : total_slices = 12) 
  (john_slices : john_ate = 3) 
  (sam_ate : ¬¬(2 * john_ate = 6)) : 
  ∃ slices_left, slices_left = 3 :=
by
  sorry

end NUMINAMATH_GPT_pizza_slices_left_l1320_132044


namespace NUMINAMATH_GPT_family_of_four_children_has_at_least_one_boy_and_one_girl_l1320_132011

noncomputable section

def probability_at_least_one_boy_one_girl : ℚ :=
  1 - (1 / 16 + 1 / 16)

theorem family_of_four_children_has_at_least_one_boy_and_one_girl :
  probability_at_least_one_boy_one_girl = 7 / 8 := by
  sorry

end NUMINAMATH_GPT_family_of_four_children_has_at_least_one_boy_and_one_girl_l1320_132011


namespace NUMINAMATH_GPT_solve_problem_l1320_132052

theorem solve_problem :
  ∃ a b c d e f : ℤ,
  (208208 = 8^5 * a + 8^4 * b + 8^3 * c + 8^2 * d + 8 * e + f) ∧
  (0 ≤ a ∧ a ≤ 7) ∧ (0 ≤ b ∧ b ≤ 7) ∧ (0 ≤ c ∧ c ≤ 7) ∧
  (0 ≤ d ∧ d ≤ 7) ∧ (0 ≤ e ∧ e ≤ 7) ∧ (0 ≤ f ∧ f ≤ 7) ∧
  (a * b * c + d * e * f = 72) :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l1320_132052


namespace NUMINAMATH_GPT_dark_squares_exceed_light_squares_by_one_l1320_132009

theorem dark_squares_exceed_light_squares_by_one 
  (m n : ℕ) (h_m : m = 9) (h_n : n = 9) (h_total_squares : m * n = 81) :
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 5 * 4 + 4 * 5
  dark_squares - light_squares = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_dark_squares_exceed_light_squares_by_one_l1320_132009


namespace NUMINAMATH_GPT_diagonals_in_nine_sided_polygon_l1320_132006

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end NUMINAMATH_GPT_diagonals_in_nine_sided_polygon_l1320_132006


namespace NUMINAMATH_GPT_total_votes_election_l1320_132002

theorem total_votes_election
  (pct_candidate1 pct_candidate2 pct_candidate3 pct_candidate4 : ℝ)
  (votes_candidate4 total_votes : ℝ)
  (h1 : pct_candidate1 = 0.42)
  (h2 : pct_candidate2 = 0.30)
  (h3 : pct_candidate3 = 0.20)
  (h4 : pct_candidate4 = 0.08)
  (h5 : votes_candidate4 = 720)
  (h6 : votes_candidate4 = pct_candidate4 * total_votes) :
  total_votes = 9000 :=
sorry

end NUMINAMATH_GPT_total_votes_election_l1320_132002


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_l1320_132093

theorem equation1_solution (x : ℝ) (h : 5 / (x + 1) = 1 / (x - 3)) : x = 4 :=
sorry

theorem equation2_solution (x : ℝ) (h : (2 - x) / (x - 3) + 2 = 1 / (3 - x)) : x = 7 / 3 :=
sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_l1320_132093


namespace NUMINAMATH_GPT_problem1_l1320_132045

theorem problem1 (a : ℝ) (h : Real.sqrt a + 1 / Real.sqrt a = 3) :
  (a ^ 2 + 1 / a ^ 2 + 3) / (4 * a + 1 / (4 * a)) = 10 * Real.sqrt 5 := sorry

end NUMINAMATH_GPT_problem1_l1320_132045


namespace NUMINAMATH_GPT_quadratic_example_correct_l1320_132071

-- Define the quadratic function
def quad_func (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

-- Conditions defined
def condition1 := quad_func 1 = 0
def condition2 := quad_func 5 = 0
def condition3 := quad_func 3 = 8

-- Theorem statement combining the conditions
theorem quadratic_example_correct :
  condition1 ∧ condition2 ∧ condition3 :=
by
  -- Proof omitted as per instructions
  sorry

end NUMINAMATH_GPT_quadratic_example_correct_l1320_132071


namespace NUMINAMATH_GPT_regular_21_gon_symmetry_calculation_l1320_132057

theorem regular_21_gon_symmetry_calculation:
  let L := 21
  let R := 360 / 21
  L + R = 38 :=
by
  sorry

end NUMINAMATH_GPT_regular_21_gon_symmetry_calculation_l1320_132057


namespace NUMINAMATH_GPT_total_spokes_in_garage_l1320_132001

-- Definitions based on the problem conditions
def num_bicycles : ℕ := 4
def spokes_per_wheel : ℕ := 10
def wheels_per_bicycle : ℕ := 2

-- The goal is to prove the total number of spokes
theorem total_spokes_in_garage : (num_bicycles * wheels_per_bicycle * spokes_per_wheel) = 80 :=
by
    sorry

end NUMINAMATH_GPT_total_spokes_in_garage_l1320_132001


namespace NUMINAMATH_GPT_evaluate_g_of_neg_one_l1320_132095

def g (x : ℤ) : ℤ :=
  x^2 - 2*x + 1

theorem evaluate_g_of_neg_one :
  g (g (g (g (g (g (-1 : ℤ)))))) = 15738504 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_of_neg_one_l1320_132095


namespace NUMINAMATH_GPT_problem_statement_l1320_132037

def U := Set ℝ
def M := { x : ℝ | x^2 - 4 * x - 5 < 0 }
def N := { x : ℝ | 1 ≤ x }
def comp_U_N := { x : ℝ | x < 1 }
def intersection := { x : ℝ | -1 < x ∧ x < 1 }

theorem problem_statement : M ∩ comp_U_N = intersection := sorry

end NUMINAMATH_GPT_problem_statement_l1320_132037


namespace NUMINAMATH_GPT_expression_not_computable_by_square_difference_l1320_132039

theorem expression_not_computable_by_square_difference (x : ℝ) :
  ¬ ((x + 1) * (1 + x) = (x + 1) * (x - 1) ∨
     (x + 1) * (1 + x) = (-x + 1) * (-x - 1) ∨
     (x + 1) * (1 + x) = (x + 1) * (-x + 1)) :=
by
  sorry

end NUMINAMATH_GPT_expression_not_computable_by_square_difference_l1320_132039


namespace NUMINAMATH_GPT_length_of_plot_l1320_132041

open Real

variable (breadth : ℝ) (length : ℝ)
variable (b : ℝ)

axiom H1 : length = b + 40
axiom H2 : 26.5 * (4 * b + 80) = 5300

theorem length_of_plot : length = 70 :=
by
  -- To prove: The length of the plot is 70 meters.
  exact sorry

end NUMINAMATH_GPT_length_of_plot_l1320_132041


namespace NUMINAMATH_GPT_find_last_number_l1320_132014

theorem find_last_number (A B C D : ℝ) (h1 : A + B + C = 18) (h2 : B + C + D = 9) (h3 : A + D = 13) : D = 2 :=
by
sorry

end NUMINAMATH_GPT_find_last_number_l1320_132014


namespace NUMINAMATH_GPT_tangerine_count_l1320_132061

def initial_tangerines : ℕ := 10
def added_tangerines : ℕ := 6

theorem tangerine_count : initial_tangerines + added_tangerines = 16 :=
by
  sorry

end NUMINAMATH_GPT_tangerine_count_l1320_132061


namespace NUMINAMATH_GPT_parallel_lines_m_eq_neg2_l1320_132022

def l1_equation (m : ℝ) (x y: ℝ) : Prop :=
  (m+1) * x + y - 1 = 0

def l2_equation (m : ℝ) (x y: ℝ) : Prop :=
  2 * x + m * y - 1 = 0

theorem parallel_lines_m_eq_neg2 (m : ℝ) :
  (∀ x y : ℝ, l1_equation m x y) →
  (∀ x y : ℝ, l2_equation m x y) →
  (m ≠ 1) →
  (m = -2) :=
sorry

end NUMINAMATH_GPT_parallel_lines_m_eq_neg2_l1320_132022


namespace NUMINAMATH_GPT_gcd_of_powers_of_three_l1320_132021

theorem gcd_of_powers_of_three :
  let a := 3^1001 - 1
  let b := 3^1012 - 1
  gcd a b = 177146 := by
  sorry

end NUMINAMATH_GPT_gcd_of_powers_of_three_l1320_132021


namespace NUMINAMATH_GPT_fish_caught_l1320_132007

theorem fish_caught (x y : ℕ) 
  (h1 : y - 2 = 4 * (x + 2))
  (h2 : y - 6 = 2 * (x + 6)) :
  x = 4 ∧ y = 26 :=
by
  sorry

end NUMINAMATH_GPT_fish_caught_l1320_132007


namespace NUMINAMATH_GPT_factorize_expression_l1320_132015

theorem factorize_expression (x : ℝ) :
  (x + 1)^4 + (x + 3)^4 - 272 = 2 * (x^2 + 4*x + 19) * (x + 5) * (x - 1) :=
  sorry

end NUMINAMATH_GPT_factorize_expression_l1320_132015


namespace NUMINAMATH_GPT_crazy_silly_school_books_movies_correct_l1320_132079

noncomputable def crazy_silly_school_books_movies (B M : ℕ) : Prop :=
  M = 61 ∧ M = B + 2 ∧ M = 10 ∧ B = 8

theorem crazy_silly_school_books_movies_correct {B M : ℕ} :
  crazy_silly_school_books_movies B M → B = 8 :=
by
  intro h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end NUMINAMATH_GPT_crazy_silly_school_books_movies_correct_l1320_132079


namespace NUMINAMATH_GPT_probability_two_of_three_survive_l1320_132086

-- Let's define the necessary components
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of exactly 2 out of 3 seedlings surviving
theorem probability_two_of_three_survive (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) :
  binomial_coefficient 3 2 * p^2 * (1 - p) = 3 * p^2 * (1 - p) :=
by
  sorry

end NUMINAMATH_GPT_probability_two_of_three_survive_l1320_132086


namespace NUMINAMATH_GPT_rectangle_width_is_3_l1320_132003

-- Define the given conditions
def length_square : ℝ := 9
def length_rectangle : ℝ := 27

-- Calculate the area based on the given conditions
def area_square : ℝ := length_square * length_square

-- Define the area equality condition
def area_equality (width_rectangle : ℝ) : Prop :=
  area_square = length_rectangle * width_rectangle

-- The theorem stating the width of the rectangle
theorem rectangle_width_is_3 (width_rectangle: ℝ) :
  area_equality width_rectangle → width_rectangle = 3 :=
by
  -- Skipping the proof itself as instructed
  intro h
  sorry

end NUMINAMATH_GPT_rectangle_width_is_3_l1320_132003


namespace NUMINAMATH_GPT_find_abc_squares_l1320_132082

variable (a b c x : ℕ)

theorem find_abc_squares (h1 : 1 ≤ a) (h2 : a + b + c = 9) (h3 : 99 * (c - a) = 65 * x) (h4 : 495 = 65 * x) : a^2 + b^2 + c^2 = 53 :=
  sorry

end NUMINAMATH_GPT_find_abc_squares_l1320_132082


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_shifted_roots_l1320_132050

theorem sum_of_reciprocals_of_shifted_roots (p q r : ℝ)
  (h1 : p^3 - 2 * p^2 - p + 3 = 0)
  (h2 : q^3 - 2 * q^2 - q + 3 = 0)
  (h3 : r^3 - 2 * r^2 - r + 3 = 0) :
  (1 / (p - 2)) + (1 / (q - 2)) + (1 / (r - 2)) = -3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_shifted_roots_l1320_132050


namespace NUMINAMATH_GPT_complex_mul_eq_l1320_132029

/-- Proof that the product of two complex numbers (1 + i) and (2 + i) is equal to (1 + 3i) -/
theorem complex_mul_eq (i : ℂ) (h_i_squared : i^2 = -1) : (1 + i) * (2 + i) = 1 + 3 * i :=
by
  -- The actual proof logic goes here.
  sorry

end NUMINAMATH_GPT_complex_mul_eq_l1320_132029


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1320_132088

theorem simplify_and_evaluate (a : ℤ) (h : a = -4) :
  (4 * a ^ 2 - 3 * a) - (2 * a ^ 2 + a - 1) + (2 - a ^ 2 + 4 * a) = 19 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1320_132088


namespace NUMINAMATH_GPT_lisa_interest_after_10_years_l1320_132077

noncomputable def compounded_amount (P : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P * (1 + r) ^ n

theorem lisa_interest_after_10_years :
  let P := 2000
  let r := (2 : ℚ) / 100
  let n := 10
  let A := compounded_amount P r n
  A - P = 438 := by
    let P := 2000
    let r := (2 : ℚ) / 100
    let n := 10
    let A := compounded_amount P r n
    have : A - P = 438 := sorry
    exact this

end NUMINAMATH_GPT_lisa_interest_after_10_years_l1320_132077


namespace NUMINAMATH_GPT_daily_coffee_machine_cost_l1320_132019

def coffee_machine_cost := 200 -- $200
def discount := 20 -- $20
def daily_coffee_cost := 2 * 4 -- $8/day
def days_to_pay_off := 36 -- 36 days

theorem daily_coffee_machine_cost :
  (days_to_pay_off * daily_coffee_cost - (coffee_machine_cost - discount)) / days_to_pay_off = 3 := 
by
  -- Using the given conditions: 
  -- coffee_machine_cost = 200
  -- discount = 20
  -- daily_coffee_cost = 8
  -- days_to_pay_off = 36
  sorry

end NUMINAMATH_GPT_daily_coffee_machine_cost_l1320_132019


namespace NUMINAMATH_GPT_radio_price_position_l1320_132042

def price_positions (n : ℕ) (total_items : ℕ) (rank_lowest : ℕ) : Prop :=
  rank_lowest = total_items - n + 1

theorem radio_price_position :
  ∀ (n total_items rank_lowest : ℕ),
    total_items = 34 →
    rank_lowest = 21 →
    price_positions n total_items rank_lowest →
    n = 14 :=
by
  intros n total_items rank_lowest h_total h_rank h_pos
  rw [h_total, h_rank] at h_pos
  sorry

end NUMINAMATH_GPT_radio_price_position_l1320_132042


namespace NUMINAMATH_GPT_union_complement_subset_range_l1320_132055

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def B : Set ℝ := {x | 2 * x ^ 2 - 3 * x - 2 < 0}

-- Define the complement of B
def complement_R (s : Set ℝ) : Set ℝ := {x | x ∉ s}

-- 1. The proof problem for A ∪ (complement of B) when a = 1
theorem union_complement (a : ℝ) (h : a = 1) :
  { x : ℝ | (-1/2 < x ∧ x ≤ 1) ∨ (x ≥ 2 ∨ x ≤ -1/2) } = 
  { x : ℝ | x ≤ 1 ∨ x ≥ 2 } :=
by
  sorry

-- 2. The proof problem for A ⊆ B to find the range of a
theorem subset_range (a : ℝ) :
  (∀ x, A a x → B x) ↔ -1 < a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_union_complement_subset_range_l1320_132055


namespace NUMINAMATH_GPT_page_added_twice_is_33_l1320_132013

noncomputable def sum_first_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem page_added_twice_is_33 :
  ∃ n : ℕ, ∃ m : ℕ, sum_first_n n + m = 1986 ∧ 1 ≤ m ∧ m ≤ n → m = 33 := 
by {
  sorry
}

end NUMINAMATH_GPT_page_added_twice_is_33_l1320_132013


namespace NUMINAMATH_GPT_calls_on_friday_l1320_132048

noncomputable def total_calls_monday := 35
noncomputable def total_calls_tuesday := 46
noncomputable def total_calls_wednesday := 27
noncomputable def total_calls_thursday := 61
noncomputable def average_calls_per_day := 40
noncomputable def number_of_days := 5
noncomputable def total_calls_week := average_calls_per_day * number_of_days

theorem calls_on_friday : 
  total_calls_week - (total_calls_monday + total_calls_tuesday + total_calls_wednesday + total_calls_thursday) = 31 :=
by
  sorry

end NUMINAMATH_GPT_calls_on_friday_l1320_132048


namespace NUMINAMATH_GPT_edward_earnings_l1320_132049

theorem edward_earnings
    (total_lawns : ℕ := 17)
    (forgotten_lawns : ℕ := 9)
    (total_earnings : ℕ := 32) :
    (total_earnings / (total_lawns - forgotten_lawns) = 4) :=
by
  sorry

end NUMINAMATH_GPT_edward_earnings_l1320_132049


namespace NUMINAMATH_GPT_perfect_cube_factors_count_l1320_132027

-- Define the given prime factorization
def prime_factorization_8820 : Prop :=
  ∃ a b c d : ℕ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 2 ∧
  (2 ^ a) * (3 ^ b) * (5 ^ c) * (7 ^ d) = 8820

-- Prove the statement about positive integer factors that are perfect cubes
theorem perfect_cube_factors_count : prime_factorization_8820 → (∃ n : ℕ, n = 1) :=
by
  sorry

end NUMINAMATH_GPT_perfect_cube_factors_count_l1320_132027


namespace NUMINAMATH_GPT_central_angle_radian_measure_l1320_132098

namespace SectorProof

variables (R l : ℝ)
variables (α : ℝ)

-- Given conditions
def condition1 : Prop := 2 * R + l = 20
def condition2 : Prop := 1 / 2 * l * R = 9
def α_definition : Prop := α = l / R

-- Central angle result
theorem central_angle_radian_measure (h1 : condition1 R l) (h2 : condition2 R l) :
  α_definition α l R → α = 2 / 9 :=
by
  intro h_α
  -- proof steps would be here, but we skip them with sorry
  sorry

end SectorProof

end NUMINAMATH_GPT_central_angle_radian_measure_l1320_132098


namespace NUMINAMATH_GPT_intersection_eq_l1320_132010

def A : Set ℤ := {-2, -1, 3, 4}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_eq : A ∩ B = {-1, 3} := 
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1320_132010


namespace NUMINAMATH_GPT_find_a_plus_2b_l1320_132051

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6 * a * x + b

noncomputable def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 6 * a

theorem find_a_plus_2b (a b : ℝ) 
  (h1 : f' a b 2 = 0)
  (h2 : f a b 2 = 9) : a + 2 * b = -24 := 
by sorry

end NUMINAMATH_GPT_find_a_plus_2b_l1320_132051


namespace NUMINAMATH_GPT_elegant_interval_solution_l1320_132080

noncomputable def elegant_interval : ℝ → ℝ × ℝ := sorry

theorem elegant_interval_solution (m : ℝ) (a b : ℕ) (s : ℝ) (p : ℕ) :
  a < m ∧ m < b ∧ a + 1 = b ∧ 3 < s + b ∧ s + b ≤ 13 ∧ s = Real.sqrt a ∧ b * b + a * s = p → p = 33 ∨ p = 127 := 
by sorry

end NUMINAMATH_GPT_elegant_interval_solution_l1320_132080


namespace NUMINAMATH_GPT_eval_expression_l1320_132083

theorem eval_expression : 3 ^ 4 - 4 * 3 ^ 3 + 6 * 3 ^ 2 - 4 * 3 + 1 = 16 := 
by 
  sorry

end NUMINAMATH_GPT_eval_expression_l1320_132083


namespace NUMINAMATH_GPT_Union_A_B_eq_l1320_132033

noncomputable def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
noncomputable def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem Union_A_B_eq : A ∪ B = {x | -2 < x ∧ x ≤ 4} :=
by
  sorry

end NUMINAMATH_GPT_Union_A_B_eq_l1320_132033


namespace NUMINAMATH_GPT_lucky_lucy_l1320_132028

theorem lucky_lucy (a b c d e : ℤ)
  (ha : a = 2)
  (hb : b = 4)
  (hc : c = 6)
  (hd : d = 8)
  (he : a + b - c + d - e = a + (b - (c + (d - e)))) :
  e = 8 :=
by
  rw [ha, hb, hc, hd] at he
  exact eq_of_sub_eq_zero (by linarith)

end NUMINAMATH_GPT_lucky_lucy_l1320_132028


namespace NUMINAMATH_GPT_hyperbola_vertex_distance_l1320_132085

theorem hyperbola_vertex_distance : 
  ∀ x y: ℝ, (x^2 / 144 - y^2 / 49 = 1) → (∃ a: ℝ, a = 12 ∧ 2 * a = 24) :=
by 
  sorry

end NUMINAMATH_GPT_hyperbola_vertex_distance_l1320_132085


namespace NUMINAMATH_GPT_minimize_tangent_triangle_area_l1320_132070

open Real

theorem minimize_tangent_triangle_area {a b x y : ℝ} 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) :
  (∃ x y : ℝ, (x = a / sqrt 2 ∨ x = -a / sqrt 2) ∧ (y = b / sqrt 2 ∨ y = -b / sqrt 2)) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_minimize_tangent_triangle_area_l1320_132070


namespace NUMINAMATH_GPT_cans_purchased_l1320_132047

theorem cans_purchased (S Q E : ℝ) (h1 : Q ≠ 0) (h2 : S > 0) :
  (10 * E * S) / Q = (10 * (E : ℝ) * (S : ℝ)) / (Q : ℝ) := by 
  sorry

end NUMINAMATH_GPT_cans_purchased_l1320_132047


namespace NUMINAMATH_GPT_angle_measure_triple_complement_l1320_132099

theorem angle_measure_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
by { sorry }

end NUMINAMATH_GPT_angle_measure_triple_complement_l1320_132099


namespace NUMINAMATH_GPT_barium_atoms_in_compound_l1320_132078

noncomputable def barium_atoms (total_molecular_weight : ℝ) (weight_ba_per_atom : ℝ) (weight_br_per_atom : ℝ) (num_br_atoms : ℕ) : ℝ :=
  (total_molecular_weight - (num_br_atoms * weight_br_per_atom)) / weight_ba_per_atom

theorem barium_atoms_in_compound :
  barium_atoms 297 137.33 79.90 2 = 1 :=
by
  unfold barium_atoms
  norm_num
  sorry

end NUMINAMATH_GPT_barium_atoms_in_compound_l1320_132078


namespace NUMINAMATH_GPT_shallow_depth_of_pool_l1320_132012

theorem shallow_depth_of_pool (w l D V : ℝ) (h₀ : w = 9) (h₁ : l = 12) (h₂ : D = 4) (h₃ : V = 270) :
  (0.5 * (d + D) * w * l = V) → d = 1 :=
by
  intros h_equiv
  sorry

end NUMINAMATH_GPT_shallow_depth_of_pool_l1320_132012
