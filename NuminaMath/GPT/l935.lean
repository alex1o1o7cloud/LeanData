import Mathlib

namespace NUMINAMATH_GPT_unique_solution_of_quadratic_l935_93531

theorem unique_solution_of_quadratic :
  ∀ (b : ℝ), b ≠ 0 → (∃ x : ℝ, 3 * x^2 + b * x + 12 = 0 ∧ ∀ y : ℝ, 3 * y^2 + b * y + 12 = 0 → y = x) → 
  (b = 12 ∧ ∃ x : ℝ, x = -2 ∧ 3 * x^2 + 12 * x + 12 = 0 ∧ (∀ y : ℝ, 3 * y^2 + 12 * y + 12 = 0 → y = x)) ∨ 
  (b = -12 ∧ ∃ x : ℝ, x = 2 ∧ 3 * x^2 - 12 * x + 12 = 0 ∧ (∀ y : ℝ, 3 * y^2 - 12 * y + 12 = 0 → y = x)) :=
by 
  sorry

end NUMINAMATH_GPT_unique_solution_of_quadratic_l935_93531


namespace NUMINAMATH_GPT_solution_proof_l935_93587

noncomputable def proof_problem : Prop :=
  ∀ (x : ℝ), x ≠ 1 → (1 - 1 / (x - 1) = 2 * x / (1 - x)) → x = 2 / 3

theorem solution_proof : proof_problem := 
by
  sorry

end NUMINAMATH_GPT_solution_proof_l935_93587


namespace NUMINAMATH_GPT_polynomial_coefficients_sum_and_difference_l935_93545

theorem polynomial_coefficients_sum_and_difference :
  ∀ (a_0 a_1 a_2 a_3 a_4 : ℤ),
  (∀ (x : ℤ), (2 * x - 3)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  (a_1 + a_2 + a_3 + a_4 = -80) ∧ ((a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 625) :=
by
  intros a_0 a_1 a_2 a_3 a_4 h
  sorry

end NUMINAMATH_GPT_polynomial_coefficients_sum_and_difference_l935_93545


namespace NUMINAMATH_GPT_monotonic_intervals_find_f_max_l935_93506

noncomputable def f (x : ℝ) : ℝ := Real.log x / x

theorem monotonic_intervals :
  (∀ x, 0 < x → x < Real.exp 1 → 0 < (1 - Real.log x) / x^2) ∧
  (∀ x, x > Real.exp 1 → (1 - Real.log x) / x^2 < 0) :=
sorry

theorem find_f_max (m : ℝ) (h : m > 0) :
  if 0 < 2 * m ∧ 2 * m ≤ Real.exp 1 then f (2 * m) = Real.log (2 * m) / (2 * m)
  else if m ≥ Real.exp 1 then f m = Real.log m / m
  else f (Real.exp 1) = 1 / Real.exp 1 :=
sorry

end NUMINAMATH_GPT_monotonic_intervals_find_f_max_l935_93506


namespace NUMINAMATH_GPT_trip_time_40mph_l935_93595

noncomputable def trip_time_80mph : ℝ := 6.75
noncomputable def speed_80mph : ℝ := 80
noncomputable def speed_40mph : ℝ := 40

noncomputable def distance : ℝ := speed_80mph * trip_time_80mph

theorem trip_time_40mph : distance / speed_40mph = 13.50 :=
by
  sorry

end NUMINAMATH_GPT_trip_time_40mph_l935_93595


namespace NUMINAMATH_GPT_prime_roots_quadratic_l935_93542

theorem prime_roots_quadratic (p q : ℕ) (x1 x2 : ℕ) 
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (h_prime_x1 : Nat.Prime x1)
  (h_prime_x2 : Nat.Prime x2)
  (h_eq : p * x1 * x1 + p * x2 * x2 - q * x1 * x2 + 1985 = 0) :
  12 * p * p + q = 414 :=
sorry

end NUMINAMATH_GPT_prime_roots_quadratic_l935_93542


namespace NUMINAMATH_GPT_sum_of_number_and_reverse_l935_93580

theorem sum_of_number_and_reverse :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → ((10 * a + b) - (10 * b + a) = 7 * (a + b)) → a = 8 * b → 
  (10 * a + b) + (10 * b + a) = 99 :=
by
  intros a b conditions eq diff
  sorry

end NUMINAMATH_GPT_sum_of_number_and_reverse_l935_93580


namespace NUMINAMATH_GPT_mass_percentage_O_in_CaO_l935_93555

theorem mass_percentage_O_in_CaO :
  let molar_mass_Ca := 40.08
  let molar_mass_O := 16.00
  let molar_mass_CaO := molar_mass_Ca + molar_mass_O
  let mass_percentage_O := (molar_mass_O / molar_mass_CaO) * 100
  mass_percentage_O = 28.53 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_O_in_CaO_l935_93555


namespace NUMINAMATH_GPT_converse_proposition_l935_93533

-- Define the condition: The equation x^2 + x - m = 0 has real roots
def has_real_roots (a b c : ℝ) : Prop :=
  let Δ := b * b - 4 * a * c
  Δ ≥ 0

theorem converse_proposition (m : ℝ) :
  has_real_roots 1 1 (-m) → m > 0 :=
by
  sorry

end NUMINAMATH_GPT_converse_proposition_l935_93533


namespace NUMINAMATH_GPT_number_of_ordered_tuples_l935_93544

noncomputable def count_tuples 
  (a1 a2 a3 a4 : ℕ) 
  (H_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (H_range : 1 ≤ a1 ∧ a1 ≤ 100 ∧ 1 ≤ a2 ∧ a2 ≤ 100 ∧ 1 ≤ a3 ∧ a3 ≤ 100 ∧ 1 ≤ a4 ∧ a4 ≤ 100)
  (H_eqn : (a1^2 + a2^2 + a3^2) * (a2^2 + a3^2 + a4^2) = (a1 * a2 + a2 * a3 + a3 * a4)^2): ℕ :=
40

theorem number_of_ordered_tuples 
  (a1 a2 a3 a4 : ℕ)
  (H_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (H_range : 1 ≤ a1 ∧ a1 ≤ 100 ∧ 1 ≤ a2 ∧ a2 ≤ 100 ∧ 1 ≤ a3 ∧ a3 ≤ 100 ∧ 1 ≤ a4 ∧ a4 ≤ 100)
  (H_eqn : (a1^2 + a2^2 + a3^2) * (a2^2 + a3^2 + a4^2) = (a1 * a2 + a2 * a3 + a3 * a4)^2) : 
  count_tuples a1 a2 a3 a4 H_distinct H_range H_eqn = 40 :=
sorry

end NUMINAMATH_GPT_number_of_ordered_tuples_l935_93544


namespace NUMINAMATH_GPT_proof_expression_equals_60_times_10_power_1501_l935_93504

noncomputable def expression_equals_60_times_10_power_1501 : Prop :=
  (2^1501 + 5^1502)^3 - (2^1501 - 5^1502)^3 = 60 * 10^1501

theorem proof_expression_equals_60_times_10_power_1501 :
  expression_equals_60_times_10_power_1501 :=
by 
  sorry

end NUMINAMATH_GPT_proof_expression_equals_60_times_10_power_1501_l935_93504


namespace NUMINAMATH_GPT_min_value_expression_l935_93548

theorem min_value_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 215 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l935_93548


namespace NUMINAMATH_GPT_weeks_per_mouse_correct_l935_93530

def years_in_decade : ℕ := 10
def weeks_per_year : ℕ := 52
def total_mice : ℕ := 130

def total_weeks_in_decade : ℕ := years_in_decade * weeks_per_year
def weeks_per_mouse : ℕ := total_weeks_in_decade / total_mice

theorem weeks_per_mouse_correct : weeks_per_mouse = 4 := 
sorry

end NUMINAMATH_GPT_weeks_per_mouse_correct_l935_93530


namespace NUMINAMATH_GPT_min_value_fraction_l935_93543

theorem min_value_fraction (x : ℝ) (h : x > 9) : 
  (∃ y, y > 9 ∧ (∀ z, z > 9 → y ≤ (z^3 / (z - 9)))) ∧ (∀ z, z > 9 → (∃ w, w > 9 ∧ z^3 / (z - 9) = 325)) := 
  sorry

end NUMINAMATH_GPT_min_value_fraction_l935_93543


namespace NUMINAMATH_GPT_no_intersection_points_l935_93556

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 5

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ := -x^2 + 6 * x - 8

-- The statement asserting that the parabolas do not intersect
theorem no_intersection_points :
  ∀ (x y : ℝ), parabola1 x = y → parabola2 x = y → false :=
by
  -- Introducing x and y as elements of the real numbers
  intros x y h1 h2
  
  -- Since this is only the statement, we use sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_no_intersection_points_l935_93556


namespace NUMINAMATH_GPT_division_correct_l935_93549

-- Definitions based on conditions
def expr1 : ℕ := 12 + 15 * 3
def expr2 : ℚ := 180 / expr1

-- Theorem statement using the question and correct answer
theorem division_correct : expr2 = 180 / 57 := by
  sorry

end NUMINAMATH_GPT_division_correct_l935_93549


namespace NUMINAMATH_GPT_arccos_one_eq_zero_l935_93599

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end NUMINAMATH_GPT_arccos_one_eq_zero_l935_93599


namespace NUMINAMATH_GPT_evaluate_expression_l935_93563

theorem evaluate_expression : 3 ^ 123 + 9 ^ 5 / 9 ^ 3 = 3 ^ 123 + 81 :=
by
  -- we add sorry as the proof is not required
  sorry

end NUMINAMATH_GPT_evaluate_expression_l935_93563


namespace NUMINAMATH_GPT_positive_diff_two_largest_prime_factors_l935_93592

theorem positive_diff_two_largest_prime_factors (a b c d : ℕ) (h : 178469 = a * b * c * d) 
  (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d) 
  (hle1 : a ≤ b) (hle2 : b ≤ c) (hle3 : c ≤ d):
  d - c = 2 := by sorry

end NUMINAMATH_GPT_positive_diff_two_largest_prime_factors_l935_93592


namespace NUMINAMATH_GPT_max_ab_value_l935_93528

theorem max_ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_perpendicular : (2 * a - 1) * b = -1) : ab <= 1 / 8 := by
  sorry

end NUMINAMATH_GPT_max_ab_value_l935_93528


namespace NUMINAMATH_GPT_m_value_l935_93596

theorem m_value (A : Set ℝ) (B : Set ℝ) (m : ℝ) 
                (hA : A = {0, 1, 2}) 
                (hB : B = {1, m}) 
                (h_subset : B ⊆ A) : 
                m = 0 ∨ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_m_value_l935_93596


namespace NUMINAMATH_GPT_probability_page_multiple_of_7_l935_93522

theorem probability_page_multiple_of_7 (total_pages : ℕ) (probability : ℚ)
  (h_total_pages : total_pages = 500) 
  (h_probability : probability = 71 / 500) :
  probability = 0.142 := 
sorry

end NUMINAMATH_GPT_probability_page_multiple_of_7_l935_93522


namespace NUMINAMATH_GPT_valid_fractions_l935_93509

theorem valid_fractions :
  ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ (1 ≤ z ∧ z ≤ 9) ∧
  (10 * x + y) % (10 * y + z) = 0 ∧ (10 * x + y) / (10 * y + z) = x / z :=
sorry

end NUMINAMATH_GPT_valid_fractions_l935_93509


namespace NUMINAMATH_GPT_multiply_powers_l935_93569

theorem multiply_powers (a : ℝ) : (a^3) * (a^3) = a^6 := by
  sorry

end NUMINAMATH_GPT_multiply_powers_l935_93569


namespace NUMINAMATH_GPT_loan_principal_and_repayment_amount_l935_93561

theorem loan_principal_and_repayment_amount (P R : ℝ) (r : ℝ) (years : ℕ) (total_interest : ℝ)
    (h1: r = 0.12)
    (h2: years = 3)
    (h3: total_interest = 5400)
    (h4: total_interest / years = R)
    (h5: R = P * r) :
    P = 15000 ∧ R = 1800 :=
sorry

end NUMINAMATH_GPT_loan_principal_and_repayment_amount_l935_93561


namespace NUMINAMATH_GPT_payment_to_C_l935_93516

def work_rate (days : ℕ) : ℚ := 1 / days

def total_payment : ℚ := 3360

def work_done (rate : ℚ) (days : ℕ) : ℚ := rate * days

-- Conditions
def person_A_work_rate := work_rate 6
def person_B_work_rate := work_rate 8
def combined_work_rate := person_A_work_rate + person_B_work_rate
def work_by_A_and_B_in_3_days := work_done combined_work_rate 3
def total_work : ℚ := 1
def work_done_by_C := total_work - work_by_A_and_B_in_3_days

-- Proof problem statement
theorem payment_to_C :
  (work_done_by_C / total_work) * total_payment = 420 := 
sorry

end NUMINAMATH_GPT_payment_to_C_l935_93516


namespace NUMINAMATH_GPT_correct_operation_l935_93527
variable (a x y: ℝ)

theorem correct_operation : 
  ¬ (5 * a - 2 * a = 3) ∧
  ¬ ((x + 2 * y)^2 = x^2 + 4 * y^2) ∧
  ¬ (x^8 / x^4 = x^2) ∧
  ((2 * a)^3 = 8 * a^3) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l935_93527


namespace NUMINAMATH_GPT_slopes_of_intersecting_line_l935_93567

theorem slopes_of_intersecting_line {m : ℝ} :
  (∃ x y : ℝ, y = m * x + 4 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ Set.Iic (-Real.sqrt 0.48) ∪ Set.Ici (Real.sqrt 0.48) :=
by
  sorry

end NUMINAMATH_GPT_slopes_of_intersecting_line_l935_93567


namespace NUMINAMATH_GPT_cody_money_final_l935_93578

theorem cody_money_final (initial_money : ℕ) (birthday_money : ℕ) (money_spent : ℕ) (final_money : ℕ) 
  (h1 : initial_money = 45) (h2 : birthday_money = 9) (h3 : money_spent = 19) :
  final_money = initial_money + birthday_money - money_spent :=
by {
  sorry  -- The proof is not required here.
}

end NUMINAMATH_GPT_cody_money_final_l935_93578


namespace NUMINAMATH_GPT_candy_bars_per_bag_l935_93562

theorem candy_bars_per_bag (total_candy_bars : ℕ) (number_of_bags : ℕ) (h1 : total_candy_bars = 15) (h2 : number_of_bags = 5) : total_candy_bars / number_of_bags = 3 :=
by
  sorry

end NUMINAMATH_GPT_candy_bars_per_bag_l935_93562


namespace NUMINAMATH_GPT_g_of_1986_l935_93576

-- Define the function g and its properties
noncomputable def g : ℕ → ℤ :=
sorry  -- Placeholder for the actual definition according to the conditions

axiom g_is_defined (x : ℕ) : x ≥ 0 → ∃ y : ℤ, g x = y
axiom g_at_1 : g 1 = 1
axiom g_add (a b : ℕ) (h_a : a ≥ 0) (h_b : b ≥ 0) : g (a + b) = g a + g b - 3 * g (a * b) + 1

-- Lean statement for the proof problem
theorem g_of_1986 : g 1986 = 0 :=
sorry

end NUMINAMATH_GPT_g_of_1986_l935_93576


namespace NUMINAMATH_GPT_cows_and_goats_sum_l935_93550

theorem cows_and_goats_sum (x y z : ℕ) 
  (h1 : x + y + z = 12) 
  (h2 : 4 * x + 2 * y + 4 * z = 18 + 2 * (x + y + z)) 
  : x + z = 9 := by 
  sorry

end NUMINAMATH_GPT_cows_and_goats_sum_l935_93550


namespace NUMINAMATH_GPT_four_digit_unique_count_l935_93517

theorem four_digit_unique_count : 
  (∃ k : ℕ, k = 14 ∧ ∃ lst : List ℕ, lst.length = 4 ∧ 
    (∀ d ∈ lst, d = 2 ∨ d = 3) ∧ (2 ∈ lst) ∧ (3 ∈ lst)) :=
by
  sorry

end NUMINAMATH_GPT_four_digit_unique_count_l935_93517


namespace NUMINAMATH_GPT_similarity_ratio_of_polygons_l935_93511

theorem similarity_ratio_of_polygons (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) : a / (b : ℚ) = 3 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_similarity_ratio_of_polygons_l935_93511


namespace NUMINAMATH_GPT_find_largest_integer_solution_l935_93540

theorem find_largest_integer_solution:
  ∃ x: ℤ, (1/4 : ℝ) < (x / 6 : ℝ) ∧ (x / 6 : ℝ) < (7/9 : ℝ) ∧ (x = 4) := by
  sorry

end NUMINAMATH_GPT_find_largest_integer_solution_l935_93540


namespace NUMINAMATH_GPT_algebraic_expression_value_l935_93535

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a + 2 * b = 2) : 
  (-a * (-2) ^ 2 + b * (-2) + 1) = -1 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l935_93535


namespace NUMINAMATH_GPT_consistent_price_per_kg_l935_93507

theorem consistent_price_per_kg (m₁ m₂ : ℝ) (p₁ p₂ : ℝ)
  (h₁ : p₁ = 6) (h₂ : m₁ = 2)
  (h₃ : p₂ = 36) (h₄ : m₂ = 12) :
  (p₁ / m₁ = p₂ / m₂) := 
by 
  sorry

end NUMINAMATH_GPT_consistent_price_per_kg_l935_93507


namespace NUMINAMATH_GPT_hari_digs_well_alone_in_48_days_l935_93591

theorem hari_digs_well_alone_in_48_days :
  (1 / 16 + 1 / 24 + 1 / (Hari_days)) = 1 / 8 → Hari_days = 48 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_hari_digs_well_alone_in_48_days_l935_93591


namespace NUMINAMATH_GPT_length_of_adult_bed_is_20_decimeters_l935_93520

-- Define the length of an adult bed as per question context
def length_of_adult_bed := 20

-- Prove that the length of an adult bed in decimeters equals 20
theorem length_of_adult_bed_is_20_decimeters : length_of_adult_bed = 20 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_length_of_adult_bed_is_20_decimeters_l935_93520


namespace NUMINAMATH_GPT_corrected_average_l935_93502

theorem corrected_average (incorrect_avg : ℕ) (correct_val incorrect_val number_of_values : ℕ) (avg := 17) (n := 10) (inc := 26) (cor := 56) :
  incorrect_avg = 17 →
  number_of_values = 10 →
  correct_val = 56 →
  incorrect_val = 26 →
  correct_avg = (incorrect_avg * number_of_values + (correct_val - incorrect_val)) / number_of_values →
  correct_avg = 20 := by
  sorry

end NUMINAMATH_GPT_corrected_average_l935_93502


namespace NUMINAMATH_GPT_find_vanilla_cookies_l935_93526

variable (V : ℕ)

def num_vanilla_cookies_sold (choc_cookies: ℕ) (vanilla_cookies: ℕ) (total_revenue: ℕ) : Prop :=
  choc_cookies * 1 + vanilla_cookies * 2 = total_revenue

theorem find_vanilla_cookies (h : num_vanilla_cookies_sold 220 V 360) : V = 70 :=
by
  sorry

end NUMINAMATH_GPT_find_vanilla_cookies_l935_93526


namespace NUMINAMATH_GPT_initial_girls_count_l935_93546

variable (p : ℝ) (g : ℝ) (b : ℝ) (initial_girls : ℝ)

-- Conditions
def initial_percentage_of_girls (p g : ℝ) : Prop := g / p = 0.6
def final_percentage_of_girls (g : ℝ) (p : ℝ) : Prop := (g - 3) / p = 0.5

-- Statement only (no proof)
theorem initial_girls_count (p : ℝ) (h1 : initial_percentage_of_girls p (0.6 * p)) (h2 : final_percentage_of_girls (0.6 * p) p) :
  initial_girls = 18 :=
by
  sorry

end NUMINAMATH_GPT_initial_girls_count_l935_93546


namespace NUMINAMATH_GPT_smallest_circle_tangent_to_line_and_circle_l935_93501

-- Define the line equation as a condition
def line_eq (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the original circle equation as a condition
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 2 * y = 0

-- Define the smallest circle equation as a condition
def smallest_circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- The main lemma to prove that the smallest circle's equation matches the expected result
theorem smallest_circle_tangent_to_line_and_circle :
  (∀ x y, line_eq x y → smallest_circle_eq x y) ∧ (∀ x y, circle_eq x y → smallest_circle_eq x y) :=
by
  sorry -- Proof is omitted, as instructed

end NUMINAMATH_GPT_smallest_circle_tangent_to_line_and_circle_l935_93501


namespace NUMINAMATH_GPT_inequality_proof_l935_93553

variable (x1 x2 y1 y2 z1 z2 : ℝ)
variable (h0 : 0 < x1)
variable (h1 : 0 < x2)
variable (h2 : x1 * y1 > z1^2)
variable (h3 : x2 * y2 > z2^2)

theorem inequality_proof :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l935_93553


namespace NUMINAMATH_GPT_min_ab_l935_93598

theorem min_ab (a b : ℝ) (h_cond1 : a > 0) (h_cond2 : b > 0)
  (h_eq : a * b = a + b + 3) : a * b = 9 :=
sorry

end NUMINAMATH_GPT_min_ab_l935_93598


namespace NUMINAMATH_GPT_gamma_start_time_correct_l935_93508

noncomputable def trisection_points (AB : ℕ) : Prop := AB ≥ 3

structure Walkers :=
  (d : ℕ) -- Total distance AB
  (Vα : ℕ) -- Speed of person α
  (Vβ : ℕ) -- Speed of person β
  (Vγ : ℕ) -- Speed of person γ

def meeting_times (w : Walkers) := 
  w.Vα = w.d / 72 ∧ 
  w.Vβ = w.d / 36 ∧ 
  w.Vγ = w.Vβ

def start_times_correct (startA timeA_meetC : ℕ) (startB timeB_reachesA: ℕ) (startC_latest: ℕ): Prop :=
  startA = 0 ∧ 
  startB = 12 ∧
  timeA_meetC = 24 ∧ 
  timeB_reachesA = 30 ∧
  startC_latest = 16

theorem gamma_start_time_correct (AB : ℕ) (w : Walkers) (t : Walkers → Prop) : 
  trisection_points AB → 
  meeting_times w →
  start_times_correct 0 24 12 30 16 → 
  ∃ tγ_start, tγ_start = 16 :=
sorry

end NUMINAMATH_GPT_gamma_start_time_correct_l935_93508


namespace NUMINAMATH_GPT_sector_area_l935_93570

theorem sector_area (α : ℝ) (l : ℝ) (r : ℝ) (S : ℝ) : 
  α = 1 ∧ l = 6 ∧ l = α * r → S = (1/2) * α * r ^ 2 → S = 18 :=
by
  intros h h' 
  sorry

end NUMINAMATH_GPT_sector_area_l935_93570


namespace NUMINAMATH_GPT_average_of_second_set_l935_93559

open Real

theorem average_of_second_set 
  (avg6 : ℝ)
  (n1 n2 n3 n4 n5 n6 : ℝ)
  (avg1_set : ℝ)
  (avg3_set : ℝ)
  (h1 : avg6 = 3.95)
  (h2 : (n1 + n2 + n3 + n4 + n5 + n6) / 6 = avg6)
  (h3 : (n1 + n2) / 2 = 3.6)
  (h4 : (n5 + n6) / 2 = 4.400000000000001) :
  (n3 + n4) / 2 = 3.85 :=
by
  sorry

end NUMINAMATH_GPT_average_of_second_set_l935_93559


namespace NUMINAMATH_GPT_cake_volume_l935_93512

theorem cake_volume :
  let thickness := 1 / 2
  let diameter := 16
  let radius := diameter / 2
  let total_volume := Real.pi * radius^2 * thickness
  total_volume / 16 = 2 * Real.pi := by
    sorry

end NUMINAMATH_GPT_cake_volume_l935_93512


namespace NUMINAMATH_GPT_problem_statement_l935_93594

-- Define proposition p
def prop_p : Prop := ∃ x : ℝ, Real.exp x ≥ x + 1

-- Define proposition q
def prop_q : Prop := ∀ (a b : ℝ), a^2 < b^2 → a < b

-- The final statement we want to prove
theorem problem_statement : (prop_p ∧ ¬prop_q) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l935_93594


namespace NUMINAMATH_GPT_can_weigh_1kg_with_300g_and_650g_weights_l935_93523

-- Definitions based on conditions
def balance_scale (a b : ℕ) (w₁ w₂ : ℕ) : Prop :=
  a * w₁ + b * w₂ = 1000

-- Statement to prove based on the problem and solution
theorem can_weigh_1kg_with_300g_and_650g_weights (w₁ : ℕ) (w₂ : ℕ) (a b : ℕ)
  (h_w1 : w₁ = 300) (h_w2 : w₂ = 650) (h_a : a = 1) (h_b : b = 1) :
  balance_scale a b w₁ w₂ :=
by 
  -- We are given:
  -- - w1 = 300 g
  -- - w2 = 650 g
  -- - we want to measure 1000 g using these weights
  -- - a = 1
  -- - b = 1
  -- Prove that:
  --   a * w1 + b * w2 = 1000
  -- Which is:
  --   1 * 300 + 1 * 650 = 1000
  sorry

end NUMINAMATH_GPT_can_weigh_1kg_with_300g_and_650g_weights_l935_93523


namespace NUMINAMATH_GPT_product_of_two_numbers_l935_93577

theorem product_of_two_numbers :
  ∃ (a b : ℚ), (∀ k : ℚ, a = k + b) ∧ (∀ k : ℚ, a + b = 8 * k) ∧ (∀ k : ℚ, a * b = 40 * k) ∧ (a * b = 6400 / 63) :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_two_numbers_l935_93577


namespace NUMINAMATH_GPT_students_like_basketball_or_cricket_or_both_l935_93566

theorem students_like_basketball_or_cricket_or_both {A B C : ℕ} (hA : A = 12) (hB : B = 8) (hC : C = 3) :
    A + B - C = 17 :=
by
  sorry

end NUMINAMATH_GPT_students_like_basketball_or_cricket_or_both_l935_93566


namespace NUMINAMATH_GPT_math_problem_l935_93564

theorem math_problem (x : ℝ) : 
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1/2 ∧ (x^2 + x^3 - 2 * x^4) / (x + x^2 - 2 * x^3) ≥ -1 ↔ 
  x ∈ Set.Icc (-1 : ℝ) (-1/2) ∪ Set.Ioc (-1/2 : ℝ) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioi 1 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_l935_93564


namespace NUMINAMATH_GPT_quadratic_function_incorrect_statement_l935_93557

theorem quadratic_function_incorrect_statement (x : ℝ) : 
  ∀ y : ℝ, y = -(x + 2)^2 - 1 → ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ y = 0 ∧ -(x1 + 2)^2 - 1 = 0 ∧ -(x2 + 2)^2 - 1 = 0) :=
by 
sorry

end NUMINAMATH_GPT_quadratic_function_incorrect_statement_l935_93557


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l935_93505

-- Define an isosceles triangle structure
structure IsoscelesTriangle where
  (a b c : ℝ) 
  (isosceles : a = b ∨ a = c ∨ b = c)
  (side_lengths : (a = 2 ∨ a = 3) ∧ (b = 2 ∨ b = 3) ∧ (c = 2 ∨ c = 3))
  (valid_triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a)

-- Define the theorem to prove the perimeter
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  t.a + t.b + t.c = 7 ∨ t.a + t.b + t.c = 8 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l935_93505


namespace NUMINAMATH_GPT_one_cubic_foot_is_1728_cubic_inches_l935_93573

-- Define the basic equivalence of feet to inches.
def foot_to_inch : ℝ := 12

-- Define the conversion from cubic feet to cubic inches.
def cubic_foot_to_cubic_inch (cubic_feet : ℝ) : ℝ :=
  (foot_to_inch * cubic_feet) ^ 3

-- State the theorem to prove the equivalence in cubic measurement.
theorem one_cubic_foot_is_1728_cubic_inches : cubic_foot_to_cubic_inch 1 = 1728 :=
  sorry -- Proof skipped.

end NUMINAMATH_GPT_one_cubic_foot_is_1728_cubic_inches_l935_93573


namespace NUMINAMATH_GPT_rental_days_l935_93565

-- Definitions based on conditions
def daily_rate := 30
def weekly_rate := 190
def total_payment := 310

-- Prove that Jennie rented the car for 11 days
theorem rental_days : ∃ d : ℕ, d = 11 ∧ (total_payment = weekly_rate + (d - 7) * daily_rate) ∨ (d < 7 ∧ total_payment = d * daily_rate) :=
by
  sorry

end NUMINAMATH_GPT_rental_days_l935_93565


namespace NUMINAMATH_GPT_inflation_over_two_years_real_yield_deposit_second_year_l935_93503

-- Inflation problem setup and proof
theorem inflation_over_two_years :
  ((1 + 0.015) ^ 2 - 1) * 100 = 3.0225 :=
by sorry

-- Real yield problem setup and proof
theorem real_yield_deposit_second_year :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by sorry

end NUMINAMATH_GPT_inflation_over_two_years_real_yield_deposit_second_year_l935_93503


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l935_93547

theorem hyperbola_eccentricity
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b = -4 * a / 3)
  (hc : c = (Real.sqrt (a ^ 2 + b ^ 2)))
  (point_on_asymptote : ∃ x y : ℝ, x = 3 ∧ y = -4 ∧ (y = b / a * x ∨ y = -b / a * x)) :
  (c / a) = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l935_93547


namespace NUMINAMATH_GPT_smallest_n_divisible_l935_93583

theorem smallest_n_divisible {n : ℕ} : 
  (∃ n : ℕ, n > 0 ∧ 18 ∣ n^2 ∧ 1152 ∣ n^3 ∧ 
    (∀ m : ℕ, m > 0 → 18 ∣ m^2 → 1152 ∣ m^3 → n ≤ m)) :=
  sorry

end NUMINAMATH_GPT_smallest_n_divisible_l935_93583


namespace NUMINAMATH_GPT_find_police_stations_in_pittsburgh_l935_93500

-- Conditions
def stores_in_pittsburgh : ℕ := 2000
def hospitals_in_pittsburgh : ℕ := 500
def schools_in_pittsburgh : ℕ := 200
def total_buildings_in_new_city : ℕ := 2175

-- Define the problem statement and the target proof
theorem find_police_stations_in_pittsburgh (P : ℕ) :
  1000 + 1000 + 150 + (P + 5) = total_buildings_in_new_city → P = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_police_stations_in_pittsburgh_l935_93500


namespace NUMINAMATH_GPT_sports_club_membership_l935_93525

theorem sports_club_membership (B T Both Neither : ℕ) (hB : B = 17) (hT : T = 19) (hBoth : Both = 11) (hNeither : Neither = 2) :
  B + T - Both + Neither = 27 := by
  sorry

end NUMINAMATH_GPT_sports_club_membership_l935_93525


namespace NUMINAMATH_GPT_tan_B_eq_one_third_l935_93521

theorem tan_B_eq_one_third
  (A B : ℝ)
  (h1 : Real.cos A = 4 / 5)
  (h2 : Real.tan (A - B) = 1 / 3) :
  Real.tan B = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_tan_B_eq_one_third_l935_93521


namespace NUMINAMATH_GPT_minuend_is_12_point_5_l935_93536

theorem minuend_is_12_point_5 (x y : ℝ) (h : x + y + (x - y) = 25) : x = 12.5 := by
  sorry

end NUMINAMATH_GPT_minuend_is_12_point_5_l935_93536


namespace NUMINAMATH_GPT_number_of_correct_statements_l935_93589

def input_statement (s : String) : Prop :=
  s = "INPUT a; b; c"

def output_statement (s : String) : Prop :=
  s = "A=4"

def assignment_statement1 (s : String) : Prop :=
  s = "3=B"

def assignment_statement2 (s : String) : Prop :=
  s = "A=B=-2"

theorem number_of_correct_statements :
    input_statement "INPUT a; b; c" = false ∧
    output_statement "A=4" = false ∧
    assignment_statement1 "3=B" = false ∧
    assignment_statement2 "A=B=-2" = false :=
sorry

end NUMINAMATH_GPT_number_of_correct_statements_l935_93589


namespace NUMINAMATH_GPT_find_initial_number_l935_93593

theorem find_initial_number (N : ℝ) (h : ∃ k : ℝ, 330 * k = N + 69.00000000008731) : 
  ∃ m : ℝ, N = 330 * m - 69.00000000008731 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_number_l935_93593


namespace NUMINAMATH_GPT_find_f_neg2_l935_93532

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) := a * x^4 + b * x^2 - x + 1

-- Define the conditions and statement to be proved
theorem find_f_neg2 (a b : ℝ) (h1 : f a b 2 = 9) : f a b (-2) = 13 :=
by
  -- Conditions lead to the conclusion to be proved
  sorry

end NUMINAMATH_GPT_find_f_neg2_l935_93532


namespace NUMINAMATH_GPT_bars_not_sold_l935_93558

-- Definitions for the conditions
def cost_per_bar : ℕ := 3
def total_bars : ℕ := 9
def money_made : ℕ := 18

-- The theorem we need to prove
theorem bars_not_sold : total_bars - (money_made / cost_per_bar) = 3 := sorry

end NUMINAMATH_GPT_bars_not_sold_l935_93558


namespace NUMINAMATH_GPT_lcm_18_45_l935_93539

theorem lcm_18_45 : Nat.lcm 18 45 = 90 := by
  sorry

end NUMINAMATH_GPT_lcm_18_45_l935_93539


namespace NUMINAMATH_GPT_system_has_real_solution_l935_93519

theorem system_has_real_solution (k : ℝ) : 
  (∃ x y : ℝ, y = k * x + 4 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_system_has_real_solution_l935_93519


namespace NUMINAMATH_GPT_sin_of_cos_in_third_quadrant_ratio_of_trig_functions_l935_93584

-- Proof for Problem 1
theorem sin_of_cos_in_third_quadrant (α : ℝ) 
  (hcos : Real.cos α = -4 / 5)
  (hquad : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -3 / 5 :=
by
  sorry

-- Proof for Problem 2
theorem ratio_of_trig_functions (α : ℝ) 
  (htan : Real.tan α = -3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_of_cos_in_third_quadrant_ratio_of_trig_functions_l935_93584


namespace NUMINAMATH_GPT_parallel_tangents_a3_plus_b2_plus_d_eq_seven_l935_93597

theorem parallel_tangents_a3_plus_b2_plus_d_eq_seven:
  ∃ (a b d : ℝ),
  (1, 1).snd = a * (1:ℝ)^3 + b * (1:ℝ)^2 + d ∧
  (-1, -3).snd = a * (-1:ℝ)^3 + b * (-1:ℝ)^2 + d ∧
  (3 * a * (1:ℝ)^2 + 2 * b * 1 = 3 * a * (-1:ℝ)^2 + 2 * b * -1) ∧
  a^3 + b^2 + d = 7 := 
sorry

end NUMINAMATH_GPT_parallel_tangents_a3_plus_b2_plus_d_eq_seven_l935_93597


namespace NUMINAMATH_GPT_total_boxes_l935_93537
namespace AppleBoxes

theorem total_boxes (initial_boxes : ℕ) (apples_per_box : ℕ) (rotten_apples : ℕ)
  (apples_per_bag : ℕ) (bags_per_box : ℕ) (good_apples : ℕ) (final_boxes : ℕ) :
  initial_boxes = 14 →
  apples_per_box = 105 →
  rotten_apples = 84 →
  apples_per_bag = 6 →
  bags_per_box = 7 →
  final_boxes = (initial_boxes * apples_per_box - rotten_apples) / (apples_per_bag * bags_per_box) →
  final_boxes = 33 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  simp at h6
  exact h6

end AppleBoxes

end NUMINAMATH_GPT_total_boxes_l935_93537


namespace NUMINAMATH_GPT_tangent_line_hyperbola_l935_93538

variable {a b x x₀ y y₀ : ℝ}
variable (ha : a > 0) (hb : b > 0) (h_ab : a > b)
variable (he : x₀^2 / a^2 + y₀^2 / b^2 = 1)
variable (hh : x₀^2 / a^2 - y₀^2 / b^2 = 1)

theorem tangent_line_hyperbola
  (h_tangent_ellipse : (x₀ * x / a^2 + y₀ * y / b^2 = 1)) :
  (x₀ * x / a^2 - y₀ * y / b^2 = 1) :=
sorry

end NUMINAMATH_GPT_tangent_line_hyperbola_l935_93538


namespace NUMINAMATH_GPT_sum_of_ages_l935_93560

/-- Given a woman's age is three years more than twice her son's age, 
and the son is 27 years old, prove that the sum of their ages is 84 years. -/
theorem sum_of_ages (son_age : ℕ) (woman_age : ℕ)
  (h1 : son_age = 27)
  (h2 : woman_age = 3 + 2 * son_age) :
  son_age + woman_age = 84 := 
sorry

end NUMINAMATH_GPT_sum_of_ages_l935_93560


namespace NUMINAMATH_GPT_apples_remain_correct_l935_93541

def total_apples : ℕ := 15
def apples_eaten : ℕ := 7
def apples_remaining : ℕ := total_apples - apples_eaten

theorem apples_remain_correct : apples_remaining = 8 :=
by
  -- Initial number of apples
  let total := total_apples
  -- Number of apples eaten
  let eaten := apples_eaten
  -- Remaining apples
  let remain := total - eaten
  -- Assertion
  have h : remain = 8 := by
      sorry
  exact h

end NUMINAMATH_GPT_apples_remain_correct_l935_93541


namespace NUMINAMATH_GPT_books_checked_out_on_Thursday_l935_93534

theorem books_checked_out_on_Thursday (initial_books : ℕ) (wednesday_checked_out : ℕ) 
                                      (thursday_returned : ℕ) (friday_returned : ℕ) (final_books : ℕ) 
                                      (thursday_checked_out : ℕ) : 
  (initial_books = 98) → 
  (wednesday_checked_out = 43) → 
  (thursday_returned = 23) → 
  (friday_returned = 7) → 
  (final_books = 80) → 
  (initial_books - wednesday_checked_out + thursday_returned - thursday_checked_out + friday_returned = final_books) → 
  (thursday_checked_out = 5) :=
by
  intros
  sorry

end NUMINAMATH_GPT_books_checked_out_on_Thursday_l935_93534


namespace NUMINAMATH_GPT_west_movement_is_negative_seven_l935_93571

-- Define a function to represent the movement notation
def movement_notation (direction: String) (distance: Int) : Int :=
  if direction = "east" then distance else -distance

-- Define the movement in the east direction
def east_movement := movement_notation "east" 3

-- Define the movement in the west direction
def west_movement := movement_notation "west" 7

-- Theorem statement
theorem west_movement_is_negative_seven : west_movement = -7 := by
  sorry

end NUMINAMATH_GPT_west_movement_is_negative_seven_l935_93571


namespace NUMINAMATH_GPT_find_k_l935_93552

theorem find_k (x y k : ℝ) (h1 : x + 2 * y = k + 1) (h2 : 2 * x + y = 1) (h3 : x + y = 3) : k = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l935_93552


namespace NUMINAMATH_GPT_coefficient_of_x3y7_in_expansion_l935_93588

-- Definitions based on the conditions in the problem
def a : ℚ := (2 / 3)
def b : ℚ := - (3 / 4)
def n : ℕ := 10
def k1 : ℕ := 3
def k2 : ℕ := 7

-- Statement of the math proof problem
theorem coefficient_of_x3y7_in_expansion :
  (a * x ^ k1 + b * y ^ k2) ^ n = x3y7_coeff * x ^ k1 * y ^ k2  :=
sorry

end NUMINAMATH_GPT_coefficient_of_x3y7_in_expansion_l935_93588


namespace NUMINAMATH_GPT_becky_to_aliyah_ratio_l935_93518

def total_school_days : ℕ := 180
def days_aliyah_packs_lunch : ℕ := total_school_days / 2
def days_becky_packs_lunch : ℕ := 45

theorem becky_to_aliyah_ratio :
  (days_becky_packs_lunch : ℚ) / days_aliyah_packs_lunch = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_becky_to_aliyah_ratio_l935_93518


namespace NUMINAMATH_GPT_radian_measure_of_200_degrees_l935_93524

theorem radian_measure_of_200_degrees :
  (200 : ℝ) * (Real.pi / 180) = (10 / 9) * Real.pi :=
sorry

end NUMINAMATH_GPT_radian_measure_of_200_degrees_l935_93524


namespace NUMINAMATH_GPT_coordinate_plane_points_l935_93551

theorem coordinate_plane_points (x y : ℝ) :
    4 * x^2 * y^2 = 4 * x * y + 3 ↔ (x * y = 3 / 2 ∨ x * y = -1 / 2) :=
by 
  sorry

end NUMINAMATH_GPT_coordinate_plane_points_l935_93551


namespace NUMINAMATH_GPT_rotational_transform_preserves_expression_l935_93515

theorem rotational_transform_preserves_expression
  (a b c : ℝ)
  (ϕ : ℝ)
  (a1 b1 c1 : ℝ)
  (x' y' x'' y'' : ℝ)
  (h1 : x'' = x' * Real.cos ϕ + y' * Real.sin ϕ)
  (h2 : y'' = -x' * Real.sin ϕ + y' * Real.cos ϕ)
  (def_a1 : a1 = a * (Real.cos ϕ)^2 - 2 * b * (Real.cos ϕ) * (Real.sin ϕ) + c * (Real.sin ϕ)^2)
  (def_b1 : b1 = a * (Real.cos ϕ) * (Real.sin ϕ) + b * ((Real.cos ϕ)^2 - (Real.sin ϕ)^2) - c * (Real.cos ϕ) * (Real.sin ϕ))
  (def_c1 : c1 = a * (Real.sin ϕ)^2 + 2 * b * (Real.cos ϕ) * (Real.sin ϕ) + c * (Real.cos ϕ)^2) :
  a1 * c1 - b1^2 = a * c - b^2 := sorry

end NUMINAMATH_GPT_rotational_transform_preserves_expression_l935_93515


namespace NUMINAMATH_GPT_minimum_radius_of_third_sphere_l935_93585

noncomputable def cone_height : ℝ := 4
noncomputable def cone_base_radius : ℝ := 3

noncomputable def radius_identical_spheres : ℝ := 4 / 3  -- derived from the conditions

theorem minimum_radius_of_third_sphere
    (h r1 r2 : ℝ) -- heights and radii one and two
    (R1 R2 Rb : ℝ) -- radii of the common base
    (cond_h : h = 4)
    (cond_Rb : Rb = 3)
    (cond_radii_eq : r1 = r2) 
  : r2 = 27 / 35 :=
by
  sorry

end NUMINAMATH_GPT_minimum_radius_of_third_sphere_l935_93585


namespace NUMINAMATH_GPT_three_topping_pizzas_l935_93514

theorem three_topping_pizzas : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_GPT_three_topping_pizzas_l935_93514


namespace NUMINAMATH_GPT_blocks_from_gallery_to_work_l935_93574

theorem blocks_from_gallery_to_work (b_store b_gallery b_already_walked b_more_to_work total_blocks blocks_to_work_from_gallery : ℕ) 
  (h1 : b_store = 11)
  (h2 : b_gallery = 6)
  (h3 : b_already_walked = 5)
  (h4 : b_more_to_work = 20)
  (h5 : total_blocks = b_store + b_gallery + b_more_to_work)
  (h6 : blocks_to_work_from_gallery = total_blocks - b_already_walked - b_store - b_gallery) :
  blocks_to_work_from_gallery = 15 :=
by
  sorry

end NUMINAMATH_GPT_blocks_from_gallery_to_work_l935_93574


namespace NUMINAMATH_GPT_target_hit_probability_l935_93554

/-- 
The probabilities for two shooters to hit a target are 1/2 and 1/3, respectively.
If both shooters fire at the target simultaneously, the probability that the target 
will be hit is 2/3.
-/
theorem target_hit_probability (P₁ P₂ : ℚ) (h₁ : P₁ = 1/2) (h₂ : P₂ = 1/3) :
  1 - ((1 - P₁) * (1 - P₂)) = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_target_hit_probability_l935_93554


namespace NUMINAMATH_GPT_lean_proof_problem_l935_93513

section

variable {R : Type*} [AddCommGroup R]

def is_odd_function (f : ℝ → R) : Prop :=
  ∀ x, f (-x) = -f x

theorem lean_proof_problem (f: ℝ → ℝ) (h_odd: is_odd_function f)
    (h_cond: f 3 + f (-2) = 2) : f 2 - f 3 = -2 :=
by
  sorry

end

end NUMINAMATH_GPT_lean_proof_problem_l935_93513


namespace NUMINAMATH_GPT_intersection_correct_l935_93590

noncomputable def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def intersection_M_N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_correct : M ∩ N = intersection_M_N :=
by
  sorry

end NUMINAMATH_GPT_intersection_correct_l935_93590


namespace NUMINAMATH_GPT_angle_between_vectors_with_offset_l935_93529

noncomputable def vector_angle_with_offset : ℝ :=
  let v1 := (4, -1)
  let v2 := (6, 8)
  let dot_product := 4 * 6 + (-1) * 8
  let magnitude_v1 := Real.sqrt (4 ^ 2 + (-1) ^ 2)
  let magnitude_v2 := Real.sqrt (6 ^ 2 + 8 ^ 2)
  let cos_theta := dot_product / (magnitude_v1 * magnitude_v2)
  Real.arccos cos_theta + 30

theorem angle_between_vectors_with_offset :
  vector_angle_with_offset = Real.arccos (8 / (5 * Real.sqrt 17)) + 30 := 
sorry

end NUMINAMATH_GPT_angle_between_vectors_with_offset_l935_93529


namespace NUMINAMATH_GPT_probability_defective_first_lathe_overall_probability_defective_conditional_probability_second_lathe_conditional_probability_third_lathe_l935_93581

noncomputable def defect_rate_first_lathe : ℝ := 0.06
noncomputable def defect_rate_second_lathe : ℝ := 0.05
noncomputable def defect_rate_third_lathe : ℝ := 0.05
noncomputable def proportion_first_lathe : ℝ := 0.25
noncomputable def proportion_second_lathe : ℝ := 0.30
noncomputable def proportion_third_lathe : ℝ := 0.45

theorem probability_defective_first_lathe :
  defect_rate_first_lathe * proportion_first_lathe = 0.015 :=
by sorry

theorem overall_probability_defective :
  defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe = 0.0525 :=
by sorry

theorem conditional_probability_second_lathe :
  (defect_rate_second_lathe * proportion_second_lathe) /
  (defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe) = 2 / 7 :=
by sorry

theorem conditional_probability_third_lathe :
  (defect_rate_third_lathe * proportion_third_lathe) /
  (defect_rate_first_lathe * proportion_first_lathe +
  defect_rate_second_lathe * proportion_second_lathe +
  defect_rate_third_lathe * proportion_third_lathe) = 3 / 7 :=
by sorry

end NUMINAMATH_GPT_probability_defective_first_lathe_overall_probability_defective_conditional_probability_second_lathe_conditional_probability_third_lathe_l935_93581


namespace NUMINAMATH_GPT_find_side_a_l935_93586

noncomputable def maximum_area (A b c : ℝ) : Prop :=
  A = 2 * Real.pi / 3 ∧ (b + 2 * c = 8) ∧ 
  ((1 / 2) * b * c * Real.sin (2 * Real.pi / 3) = (Real.sqrt 3 / 2) * c * (4 - c) ∧ 
   (∀ (c' : ℝ), (Real.sqrt 3 / 2) * c' * (4 - c') ≤ 2 * Real.sqrt 3) ∧ 
   c = 2)

theorem find_side_a (A b c a : ℝ) (h : maximum_area A b c) :
  a = 2 * Real.sqrt 7 := 
by
  sorry

end NUMINAMATH_GPT_find_side_a_l935_93586


namespace NUMINAMATH_GPT_cos_double_angle_l935_93575

theorem cos_double_angle (α : ℝ) (h : Real.cos (α + Real.pi / 2) = 3 / 5) : Real.cos (2 * α) = 7 / 25 :=
by 
  sorry

end NUMINAMATH_GPT_cos_double_angle_l935_93575


namespace NUMINAMATH_GPT_solution_set_of_inequality_l935_93572

theorem solution_set_of_inequality :
  {x : ℝ | -6 * x ^ 2 - x + 2 < 0} = {x : ℝ | x < -(2 / 3)} ∪ {x | x > 1 / 2} := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l935_93572


namespace NUMINAMATH_GPT_hexagon_exists_equal_sides_four_equal_angles_hexagon_exists_equal_angles_four_equal_sides_l935_93568

theorem hexagon_exists_equal_sides_four_equal_angles : 
  ∃ (A B C D E F : Type) (AB BC CD DE EF FA : ℝ) (angle_A angle_B angle_C angle_D angle_E angle_F : ℝ), 
  (AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF ∧ EF = FA ∧ FA = AB) ∧ 
  (angle_A = angle_B ∧ angle_B = angle_E ∧ angle_E = angle_F) ∧ 
  4 * angle_A + angle_C + angle_D = 720 :=
sorry

theorem hexagon_exists_equal_angles_four_equal_sides :
  ∃ (A B C D E F : Type) (AB BC CD DA : ℝ) (angle : ℝ), 
  (angle_A = angle_B ∧ angle_B = angle_C ∧ angle_C = angle_D ∧ angle_D = angle_E ∧ angle_E = angle_F ∧ angle_F = angle_A) ∧ 
  (AB = BC ∧ BC = CD ∧ CD = DA) :=
sorry

end NUMINAMATH_GPT_hexagon_exists_equal_sides_four_equal_angles_hexagon_exists_equal_angles_four_equal_sides_l935_93568


namespace NUMINAMATH_GPT_ant_travel_finite_path_exists_l935_93582

theorem ant_travel_finite_path_exists :
  ∃ (x y z t : ℝ), |x| < |y - z + t| ∧ |y| < |x - z + t| ∧ 
                   |z| < |x - y + t| ∧ |t| < |x - y + z| :=
by
  sorry

end NUMINAMATH_GPT_ant_travel_finite_path_exists_l935_93582


namespace NUMINAMATH_GPT_solution_set_l935_93579

-- Define the function and the conditions
variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

-- Problem statement
theorem solution_set (hf_even : is_even f)
                     (hf_increasing : increasing_on f (Set.Ioi 0))
                     (hf_value : f (-2013) = 0) :
  {x | x * f x < 0} = {x | x < -2013 ∨ (0 < x ∧ x < 2013)} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l935_93579


namespace NUMINAMATH_GPT_polygon_area_l935_93510

-- Definitions and conditions
def side_length (n : ℕ) (p : ℕ) := p / n
def rectangle_area (s : ℕ) := 2 * s * s
def total_area (r : ℕ) (area : ℕ) := r * area

-- Theorem statement with conditions and conclusion
theorem polygon_area (n r p : ℕ) (h1 : n = 24) (h2 : r = 4) (h3 : p = 48) :
  total_area r (rectangle_area (side_length n p)) = 32 := by
  sorry

end NUMINAMATH_GPT_polygon_area_l935_93510
