import Mathlib

namespace police_coverage_l3_320

-- Define the intersections and streets
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

open Intersection

-- Define each street as a set of intersections
def horizontal_streets : List (List Intersection) :=
  [[A, B, C, D], [E, F, G], [H, I, J, K]]

def vertical_streets : List (List Intersection) :=
  [[A, E, H], [B, F, I], [D, G, J]]

def diagonal_streets : List (List Intersection) :=
  [[H, F, C], [C, G, K]]

def all_streets : List (List Intersection) :=
  horizontal_streets ++ vertical_streets ++ diagonal_streets

-- Define the set of police officers' placements
def police_officers : List Intersection := [B, G, H]

-- Check if each street is covered by at least one police officer
def is_covered (street : List Intersection) (officers : List Intersection) : Prop :=
  ∃ i, i ∈ street ∧ i ∈ officers

-- Define the proof problem statement
theorem police_coverage :
  ∀ street ∈ all_streets, is_covered street police_officers :=
by sorry

end police_coverage_l3_320


namespace part_I_part_II_l3_355

def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem part_I (x : ℝ) : (f x > 4) ↔ (x < -1.5 ∨ x > 2.5) :=
by
  sorry

theorem part_II (x : ℝ) : ∀ x : ℝ, f x ≥ 3 :=
by
  sorry

end part_I_part_II_l3_355


namespace area_of_square_l3_375

-- Defining the points A and B as given in the conditions.
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (4, 6)

-- Theorem statement: proving that the area of the square given the endpoints A and B is 12.5.
theorem area_of_square : 
  ∀ (A B : ℝ × ℝ),
  A = (1, 2) → B = (4, 6) → 
  ∃ (area : ℝ), area = 12.5 := 
by
  intros A B hA hB
  sorry

end area_of_square_l3_375


namespace probability_no_physics_and_chemistry_l3_305

-- Define the probabilities for the conditions
def P_physics : ℚ := 5 / 8
def P_no_physics : ℚ := 1 - P_physics
def P_chemistry_given_no_physics : ℚ := 2 / 3

-- Define the theorem we want to prove
theorem probability_no_physics_and_chemistry :
  P_no_physics * P_chemistry_given_no_physics = 1 / 4 :=
by sorry

end probability_no_physics_and_chemistry_l3_305


namespace positive_integers_satisfy_l3_322

theorem positive_integers_satisfy (n : ℕ) (h1 : 25 - 5 * n > 15) : n = 1 :=
by sorry

end positive_integers_satisfy_l3_322


namespace max_marks_l3_388

theorem max_marks (M S : ℕ) :
  (267 + 45 = 312) ∧ (312 = (45 * M) / 100) ∧ (292 + 38 = 330) ∧ (330 = (50 * S) / 100) →
  (M + S = 1354) :=
by
  sorry

end max_marks_l3_388


namespace geometric_sequence_sum_l3_352

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 1) (h3 : a 4 * a 5 * a 6 = 8) :
  a 2 + a 5 + a 8 + a 11 = 15 :=
by
  sorry

end geometric_sequence_sum_l3_352


namespace max_value_of_expression_l3_325

-- We have three nonnegative real numbers a, b, and c,
-- such that a + b + c = 3.
def nonnegative (x : ℝ) := x ≥ 0

theorem max_value_of_expression (a b c : ℝ) (h1 : nonnegative a) (h2 : nonnegative b) (h3 : nonnegative c) (h4 : a + b + c = 3) :
  a + b^2 + c^4 ≤ 3 :=
  sorry

end max_value_of_expression_l3_325


namespace count_m_in_A_l3_396

def A : Set ℕ := { 
  x | ∃ (a0 a1 a2 a3 : ℕ), a0 ∈ Finset.range 8 ∧ 
                           a1 ∈ Finset.range 8 ∧ 
                           a2 ∈ Finset.range 8 ∧ 
                           a3 ∈ Finset.range 8 ∧ 
                           a3 ≠ 0 ∧ 
                           x = a0 + a1 * 8 + a2 * 8^2 + a3 * 8^3 }

theorem count_m_in_A (m n : ℕ) (hA_m : m ∈ A) (hA_n : n ∈ A) (h_sum : m + n = 2018) (h_m_gt_n : m > n) :
  ∃! (count : ℕ), count = 497 := 
sorry

end count_m_in_A_l3_396


namespace correct_calculation_l3_359

variable (a b : ℝ)

theorem correct_calculation :
  -(a - b) = -a + b := by
  sorry

end correct_calculation_l3_359


namespace parallel_vectors_m_eq_neg3_l3_340

-- Define vectors
def vec (x y : ℝ) : ℝ × ℝ := (x, y)
def OA : ℝ × ℝ := vec 3 (-4)
def OB : ℝ × ℝ := vec 6 (-3)
def OC (m : ℝ) : ℝ × ℝ := vec (2 * m) (m + 1)

-- Define the condition that AB is parallel to OC
def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

-- Calculate AB
def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

-- The theorem to prove
theorem parallel_vectors_m_eq_neg3 (m : ℝ) :
  is_parallel AB (OC m) → m = -3 := by
  sorry

end parallel_vectors_m_eq_neg3_l3_340


namespace roots_quadratic_l3_308

theorem roots_quadratic (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0)
    (h3 : m * n = -5) : m^2 + m * n + 2 * m = 0 := by
  sorry

end roots_quadratic_l3_308


namespace Vanya_number_thought_of_l3_399

theorem Vanya_number_thought_of :
  ∃ m n : ℕ, m < 10 ∧ n < 10 ∧ (10 * m + n = 81 ∧ (10 * n + m)^2 = 4 * (10 * m + n)) :=
sorry

end Vanya_number_thought_of_l3_399


namespace find_possible_y_values_l3_341

noncomputable def validYValues (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 90) : Set ℝ :=
  { y | y = (x - 3)^2 * (x + 4) / (2 * x - 4) }

theorem find_possible_y_values (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 90) :
  validYValues x hx = {39, 6} :=
sorry

end find_possible_y_values_l3_341


namespace matrix_determinant_zero_implies_sum_of_squares_l3_358

theorem matrix_determinant_zero_implies_sum_of_squares (a b : ℝ)
  (h : (Matrix.det ![![a - Complex.I, b - 2 * Complex.I],
                       ![1, 1 + Complex.I]]) = 0) :
  a^2 + b^2 = 1 :=
sorry

end matrix_determinant_zero_implies_sum_of_squares_l3_358


namespace find_age_of_B_l3_368

-- Define A and B as natural numbers (assuming ages are non-negative integers)
variables (A B : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := A + 10 = 2 * (B - 10)
def condition2 : Prop := A = B + 6

-- The goal is to prove that B = 36 given the conditions
theorem find_age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 36 :=
sorry

end find_age_of_B_l3_368


namespace train_speed_in_kmh_l3_339

/-- Definition of length of the train in meters. -/
def train_length : ℕ := 200

/-- Definition of time taken to cross the electric pole in seconds. -/
def time_to_cross : ℕ := 20

/-- The speed of the train in km/h is 36 given the length of the train and time to cross. -/
theorem train_speed_in_kmh (length : ℕ) (time : ℕ) (h_len : length = train_length) (h_time: time = time_to_cross) : 
  (length / time : ℚ) * 3.6 = 36 := 
by
  sorry

end train_speed_in_kmh_l3_339


namespace roots_equality_l3_348

variable {α β p q : ℝ}

theorem roots_equality (h1 : α ≠ β)
    (h2 : α * α + p * α + q = 0 ∧ β * β + p * β + q = 0)
    (h3 : α^3 - α^2 * β - α * β^2 + β^3 = 0) : 
  p = 0 ∧ q < 0 :=
by 
  sorry

end roots_equality_l3_348


namespace remainder_is_37_l3_353

theorem remainder_is_37
    (d q v r : ℕ)
    (h1 : d = 15968)
    (h2 : q = 89)
    (h3 : v = 179)
    (h4 : d = q * v + r) :
  r = 37 :=
sorry

end remainder_is_37_l3_353


namespace five_digit_numbers_with_4_or_5_l3_332

theorem five_digit_numbers_with_4_or_5 : 
  let total_five_digit := 99999 - 10000 + 1
  let without_4_or_5 := 7 * 8^4
  total_five_digit - without_4_or_5 = 61328 :=
by
  let total_five_digit := 99999 - 10000 + 1
  let without_4_or_5 := 7 * 8^4
  have h : total_five_digit - without_4_or_5 = 61328 := by sorry
  exact h

end five_digit_numbers_with_4_or_5_l3_332


namespace airplane_rows_l3_379

theorem airplane_rows (R : ℕ) 
  (h1 : ∀ n, n = 5) 
  (h2 : ∀ s, s = 7) 
  (h3 : ∀ f, f = 2) 
  (h4 : ∀ p, p = 1400):
  (2 * 5 * 7 * R = 1400) → R = 20 :=
by
  -- Assuming the given equation 2 * 5 * 7 * R = 1400
  sorry

end airplane_rows_l3_379


namespace apples_sold_by_noon_l3_316

theorem apples_sold_by_noon 
  (k g c l : ℕ) 
  (hk : k = 23) 
  (hg : g = 37) 
  (hc : c = 14) 
  (hl : l = 38) :
  k + g + c - l = 36 := 
by
  -- k = 23
  -- g = 37
  -- c = 14
  -- l = 38
  -- k + g + c - l = 36

  sorry

end apples_sold_by_noon_l3_316


namespace total_price_correct_l3_304

-- Definitions based on given conditions
def basic_computer_price : ℝ := 2125
def enhanced_computer_price : ℝ := 2125 + 500
def printer_price (P : ℝ) := P = 1/8 * (enhanced_computer_price + P)

-- Statement to prove the total price of the basic computer and printer
theorem total_price_correct (P : ℝ) (h : printer_price P) : 
  basic_computer_price + P = 2500 :=
by
  sorry

end total_price_correct_l3_304


namespace prob_even_heads_40_l3_317

noncomputable def probability_even_heads (n : ℕ) : ℚ :=
  if n = 0 then 1 else
  (1/2) * (1 + (2/5) ^ n)

theorem prob_even_heads_40 :
  probability_even_heads 40 = 1/2 * (1 + (2/5) ^ 40) :=
by {
  sorry
}

end prob_even_heads_40_l3_317


namespace problem_solution_l3_394

-- Definitions of odd function and given conditions.
variables {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x) (h_eq : f 3 - f 2 = 1)

-- Proof statement of the math problem.
theorem problem_solution : f (-2) - f (-3) = 1 :=
by
  sorry

end problem_solution_l3_394


namespace sin_add_pi_over_4_eq_l3_331

variable (α : Real)
variables (hα1 : 0 < α ∧ α < Real.pi) (hα2 : Real.tan (α - Real.pi / 4) = 1 / 3)

theorem sin_add_pi_over_4_eq : Real.sin (Real.pi / 4 + α) = 3 * Real.sqrt 10 / 10 :=
by
  sorry

end sin_add_pi_over_4_eq_l3_331


namespace inequality_proof_l3_313

theorem inequality_proof (x a : ℝ) (hx : 0 < x) (ha : 0 < a) :
  (1 / Real.sqrt (x + 1)) + (1 / Real.sqrt (a + 1)) + Real.sqrt ( (a * x) / (a * x + 8) ) ≤ 2 := 
by {
  sorry
}

end inequality_proof_l3_313


namespace work_rate_problem_l3_351

theorem work_rate_problem (A B : ℚ) (h1 : A + B = 1/8) (h2 : A = 1/12) : B = 1/24 :=
sorry

end work_rate_problem_l3_351


namespace range_of_y_l3_326

noncomputable def y (x : ℝ) : ℝ := (Real.log x / Real.log 2 + 2) * (2 * (Real.log x / (2 * Real.log 2)) - 4)

theorem range_of_y :
  (1 ≤ x ∧ x ≤ 8) →
  (∀ t : ℝ, t = Real.log x / Real.log 2 → y x = t^2 - 2 * t - 8 ∧ 0 ≤ t ∧ t ≤ 3) →
  ∃ ymin ymax, (ymin ≤ y x ∧ y x ≤ ymax) ∧ ymin = -9 ∧ ymax = -5 :=
by
  sorry

end range_of_y_l3_326


namespace income_increase_by_parental_support_l3_330

variables (a b c S : ℝ)

theorem income_increase_by_parental_support 
  (h1 : S = a + b + c)
  (h2 : 2 * a + b + c = 1.05 * S)
  (h3 : a + 2 * b + c = 1.15 * S) :
  (a + b + 2 * c) = 1.8 * S :=
sorry

end income_increase_by_parental_support_l3_330


namespace simplify_and_evaluate_expr_l3_334

noncomputable def a : ℝ := 3 + Real.sqrt 5
noncomputable def b : ℝ := 3 - Real.sqrt 5

theorem simplify_and_evaluate_expr : 
  (a^2 - 2 * a * b + b^2) / (a^2 - b^2) * (a * b) / (a - b) = 2 / 3 := by
  sorry

end simplify_and_evaluate_expr_l3_334


namespace choir_average_age_l3_300

theorem choir_average_age :
  let num_females := 10
  let avg_age_females := 32
  let num_males := 18
  let avg_age_males := 35
  let num_people := num_females + num_males
  let sum_ages_females := avg_age_females * num_females
  let sum_ages_males := avg_age_males * num_males
  let total_sum_ages := sum_ages_females + sum_ages_males
  let avg_age := (total_sum_ages : ℚ) / num_people
  avg_age = 33.92857 := by
  sorry

end choir_average_age_l3_300


namespace total_water_heaters_l3_321

-- Define the conditions
variables (W C : ℕ) -- W: capacity of Wallace's water heater, C: capacity of Catherine's water heater
variable (wallace_3over4_full : W = 40 ∧ W * 3 / 4 ∧ C = W / 2 ∧ C * 3 / 4)

-- The proof problem
theorem total_water_heaters (wallace_3over4_full : W = 40 ∧ (W * 3 / 4 = 30) ∧ C = W / 2 ∧ (C * 3 / 4 = 15)) : W * 3 / 4 + C * 3 / 4 = 45 :=
sorry

end total_water_heaters_l3_321


namespace regina_has_20_cows_l3_329

theorem regina_has_20_cows (C P : ℕ)
  (h1 : P = 4 * C)
  (h2 : 400 * P + 800 * C = 48000) :
  C = 20 :=
by
  sorry

end regina_has_20_cows_l3_329


namespace ab_value_l3_398

theorem ab_value (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 29) : a * b = 2 :=
by
  -- proof will be provided here
  sorry

end ab_value_l3_398


namespace total_bees_in_colony_l3_342

def num_bees_in_hive_after_changes (initial_bees : ℕ) (bees_in : ℕ) (bees_out : ℕ) : ℕ :=
  initial_bees + bees_in - bees_out

theorem total_bees_in_colony :
  let hive1 := num_bees_in_hive_after_changes 45 12 8
  let hive2 := num_bees_in_hive_after_changes 60 15 20
  let hive3 := num_bees_in_hive_after_changes 75 10 5
  hive1 + hive2 + hive3 = 184 :=
by
  sorry

end total_bees_in_colony_l3_342


namespace leo_third_part_time_l3_376

theorem leo_third_part_time :
  ∃ (T3 : ℕ), 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 3 → T = 25 * k) →
  T1 = 25 →
  T2 = 50 →
  Break1 = 10 →
  Break2 = 15 →
  TotalTime = 2 * 60 + 30 →
  (TotalTime - (T1 + Break1 + T2 + Break2) = T3) →
  T3 = 50 := 
sorry

end leo_third_part_time_l3_376


namespace inequality_range_of_k_l3_393

theorem inequality_range_of_k 
  (a b k : ℝ)
  (h : ∀ a b : ℝ, a^2 + b^2 ≥ 2 * k * a * b) : k ∈ Set.Icc (-1 : ℝ) (1 : ℝ) :=
by
  sorry

end inequality_range_of_k_l3_393


namespace base_of_1987_with_digit_sum_25_l3_364

theorem base_of_1987_with_digit_sum_25 (b a c : ℕ) (h₀ : a * b^2 + b * b + c = 1987)
(h₁ : a + b + c = 25) (h₂ : 1 ≤ b ∧ b ≤ 45) : b = 19 :=
sorry

end base_of_1987_with_digit_sum_25_l3_364


namespace no_positive_real_roots_l3_397

theorem no_positive_real_roots (x : ℝ) : (x^3 + 6 * x^2 + 11 * x + 6 = 0) → x < 0 :=
sorry

end no_positive_real_roots_l3_397


namespace bus_ride_cost_l3_372

theorem bus_ride_cost (B T : ℝ) 
  (h1 : T = B + 6.85)
  (h2 : T + B = 9.65)
  (h3 : ∃ n : ℤ, B = 0.35 * n ∧ ∃ m : ℤ, T = 0.35 * m) : 
  B = 1.40 := 
by
  sorry

end bus_ride_cost_l3_372


namespace log2_ratio_squared_l3_306

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log2_ratio_squared :
  ∀ (x y : ℝ), x ≠ 1 → y ≠ 1 → log_base 2 x = log_base y 25 → x * y = 81
  → (log_base 2 (x / y))^2 = 5.11 :=
by
  intros x y hx hy hlog hxy
  sorry

end log2_ratio_squared_l3_306


namespace customers_who_didnt_tip_l3_366

theorem customers_who_didnt_tip:
  ∀ (total_customers tips_per_customer total_tips : ℕ),
  total_customers = 10 →
  tips_per_customer = 3 →
  total_tips = 15 →
  (total_customers - total_tips / tips_per_customer) = 5 :=
by
  intros
  sorry

end customers_who_didnt_tip_l3_366


namespace seashells_total_l3_335

theorem seashells_total (x y z T : ℕ) (m k : ℝ) 
  (h₁ : x = 2) 
  (h₂ : y = 5) 
  (h₃ : z = 9) 
  (h₄ : x + y = T) 
  (h₅ : m * x + k * y = z) : 
  T = 7 :=
by
  -- This is where the proof would go.
  sorry

end seashells_total_l3_335


namespace sequence_expression_l3_382

theorem sequence_expression (n : ℕ) (h : n ≥ 2) (T : ℕ → ℕ) (a : ℕ → ℕ)
  (hT : ∀ k : ℕ, T k = 2 * k^2)
  (ha : ∀ k : ℕ, k ≥ 2 → a k = T k / T (k - 1)) :
  a n = (n / (n - 1))^2 := 
sorry

end sequence_expression_l3_382


namespace equilateral_triangle_bound_l3_338

theorem equilateral_triangle_bound (n k : ℕ) (h_n_gt_3 : n > 3) 
  (h_k_triangles : ∃ T : Finset (Finset (ℝ × ℝ)), T.card = k ∧ ∀ t ∈ T, 
  ∃ a b c : (ℝ × ℝ), t = {a, b, c} ∧ dist a b = 1 ∧ dist b c = 1 ∧ dist c a = 1) :
  k < (2 * n) / 3 :=
by
  sorry

end equilateral_triangle_bound_l3_338


namespace susie_investment_l3_307

theorem susie_investment :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2000 ∧
  (x * 1.04 + (2000 - x) * 1.06 = 2120) → (x = 0) :=
by
  sorry

end susie_investment_l3_307


namespace bread_cost_l3_392

theorem bread_cost (H C B : ℕ) (h₁ : H = 150) (h₂ : C = 200) (h₃ : H + B = C) : B = 50 :=
by
  sorry

end bread_cost_l3_392


namespace increasing_iff_positive_difference_l3_357

variable (a : ℕ → ℝ) (d : ℝ)

def arithmetic_sequence (aₙ : ℕ → ℝ) (d : ℝ) := ∃ (a₁ : ℝ), ∀ n : ℕ, aₙ n = a₁ + n * d

theorem increasing_iff_positive_difference (a : ℕ → ℝ) (d : ℝ) (h : arithmetic_sequence a d) :
  (∀ n, a (n+1) > a n) ↔ d > 0 :=
by
  sorry

end increasing_iff_positive_difference_l3_357


namespace cube_side_length_l3_333

def cube_volume (side : ℝ) : ℝ := side ^ 3

theorem cube_side_length (volume : ℝ) (h : volume = 729) : ∃ (side : ℝ), side = 9 ∧ cube_volume side = volume :=
by
  sorry

end cube_side_length_l3_333


namespace total_fish_count_l3_337

-- Define the number of fish for each person
def Billy := 10
def Tony := 3 * Billy
def Sarah := Tony + 5
def Bobby := 2 * Sarah

-- Define the total number of fish
def TotalFish := Billy + Tony + Sarah + Bobby

-- Prove that the total number of fish all 4 people have put together is 145
theorem total_fish_count : TotalFish = 145 := 
by
  -- provide the proof steps here
  sorry

end total_fish_count_l3_337


namespace find_constants_to_satisfy_equation_l3_369

-- Define the condition
def equation_condition (x : ℝ) (A B C : ℝ) :=
  -2 * x^2 + 5 * x - 6 = A * (x^2 + 1) + (B * x + C) * x

-- Define the proof problem as a Lean 4 statement
theorem find_constants_to_satisfy_equation (A B C : ℝ) :
  A = -6 ∧ B = 4 ∧ C = 5 ↔ ∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 → equation_condition x A B C := 
by
  sorry

end find_constants_to_satisfy_equation_l3_369


namespace real_number_a_value_l3_319

open Set

variable {a : ℝ}

theorem real_number_a_value (A B : Set ℝ) (hA : A = {-1, 1, 3}) (hB : B = {a + 2, a^2 + 4}) (hAB : A ∩ B = {3}) : a = 1 := 
by 
-- Step proof will be here
sorry

end real_number_a_value_l3_319


namespace odd_function_condition_l3_350

noncomputable def f (x a : ℝ) : ℝ := (Real.sin x) / ((x - a) * (x + 1))

theorem odd_function_condition (a : ℝ) (h : ∀ x : ℝ, f x a = - f (-x) a) : a = 1 := 
sorry

end odd_function_condition_l3_350


namespace polynomial_divisibility_l3_301

theorem polynomial_divisibility (A B : ℝ)
  (h: ∀ (x : ℂ), x^2 + x + 1 = 0 → x^104 + A * x^3 + B * x = 0) :
  A + B = 0 :=
by
  sorry

end polynomial_divisibility_l3_301


namespace isosceles_triangle_perimeter_l3_386

theorem isosceles_triangle_perimeter (a b : ℕ) (h₀ : a = 3 ∨ a = 4) (h₁ : b = 3 ∨ b = 4) (h₂ : a ≠ b) :
  (a = 3 ∧ b = 4 ∧ 4 ∈ [b]) ∨ (a = 4 ∧ b = 3 ∧ 4 ∈ [a]) → 
  (a + a + b = 10) ∨ (a + b + b = 11) :=
by
  sorry

end isosceles_triangle_perimeter_l3_386


namespace soccer_league_teams_l3_345

theorem soccer_league_teams (n : ℕ) (h : n * (n - 1) / 2 = 105) : n = 15 :=
by
  -- Proof will go here
  sorry

end soccer_league_teams_l3_345


namespace sin_cos_solution_count_l3_390

-- Statement of the problem
theorem sin_cos_solution_count : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin (3 * x) = Real.cos (x / 2)) ∧ s.card = 6 := by
  sorry

end sin_cos_solution_count_l3_390


namespace sum_consecutive_powers_of_2_divisible_by_6_l3_362

theorem sum_consecutive_powers_of_2_divisible_by_6 (n : ℕ) :
  ∃ k : ℕ, 2^n + 2^(n+1) = 6 * k :=
sorry

end sum_consecutive_powers_of_2_divisible_by_6_l3_362


namespace sum_equality_l3_324

-- Define the conditions and hypothesis
variables (x y z : ℝ)
axiom condition : (x - 6)^2 + (y - 7)^2 + (z - 8)^2 = 0

-- State the theorem
theorem sum_equality : x + y + z = 21 :=
by sorry

end sum_equality_l3_324


namespace partial_fraction_series_sum_l3_349

theorem partial_fraction_series_sum : 
  (∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 3))) = 1 / 2 := by
  sorry

end partial_fraction_series_sum_l3_349


namespace min_x1_x2_squared_l3_378

theorem min_x1_x2_squared (x1 x2 m : ℝ) (hm : (m + 3)^2 ≥ 0) 
  (h_sum : x1 + x2 = -(m + 1)) 
  (h_prod : x1 * x2 = 2 * m - 2) : 
  (x1^2 + x2^2 = (m - 1)^2 + 4) ∧ ∃ m, m = 1 → x1^2 + x2^2 = 4 :=
by {
  sorry
}

end min_x1_x2_squared_l3_378


namespace central_angle_of_sector_l3_328

-- Define the given conditions
def radius : ℝ := 10
def area : ℝ := 100

-- The statement to be proved
theorem central_angle_of_sector (α : ℝ) (h : area = (1 / 2) * α * radius ^ 2) : α = 2 :=
by
  sorry

end central_angle_of_sector_l3_328


namespace calculate_difference_l3_309

def f (x : ℝ) : ℝ := x + 2
def g (x : ℝ) : ℝ := 2 * x + 4

theorem calculate_difference :
  f (g 5) - g (f 5) = -2 := by
  sorry

end calculate_difference_l3_309


namespace margarita_vs_ricciana_l3_323

-- Ricciana's distances
def ricciana_run : ℕ := 20
def ricciana_jump : ℕ := 4
def ricciana_total : ℕ := ricciana_run + ricciana_jump

-- Margarita's distances
def margarita_run : ℕ := 18
def margarita_jump : ℕ := (2 * ricciana_jump) - 1
def margarita_total : ℕ := margarita_run + margarita_jump

-- Statement to prove Margarita ran and jumped 1 more foot than Ricciana
theorem margarita_vs_ricciana : margarita_total = ricciana_total + 1 := by
  sorry

end margarita_vs_ricciana_l3_323


namespace highest_monthly_profit_max_average_profit_l3_387

noncomputable def profit (x : ℕ) : ℤ :=
if 1 ≤ x ∧ x ≤ 5 then 26 * x - 56
else if 5 < x ∧ x ≤ 12 then 210 - 20 * x
else 0

noncomputable def average_profit (x : ℕ) : ℝ :=
if 1 ≤ x ∧ x ≤ 5 then (13 * ↑x - 43 : ℤ) / ↑x
else if 5 < x ∧ x ≤ 12 then (-10 * ↑x + 200 - 640 / ↑x : ℝ)
else 0

theorem highest_monthly_profit :
  ∃ m p, m = 6 ∧ p = 90 ∧ profit m = p :=
by sorry

theorem max_average_profit (x : ℕ) :
  1 ≤ x ∧ x ≤ 12 →
  average_profit x ≤ 40 ∧ (average_profit 8 = 40 → x = 8) :=
by sorry

end highest_monthly_profit_max_average_profit_l3_387


namespace total_trophies_correct_l3_391

-- Define the current number of Michael's trophies
def michael_current_trophies : ℕ := 30

-- Define the number of trophies Michael will have in three years
def michael_trophies_in_three_years : ℕ := michael_current_trophies + 100

-- Define the number of trophies Jack will have in three years
def jack_trophies_in_three_years : ℕ := 10 * michael_current_trophies

-- Define the total number of trophies Jack and Michael will have after three years
def total_trophies_in_three_years : ℕ := michael_trophies_in_three_years + jack_trophies_in_three_years

-- Prove that the total number of trophies after three years is 430
theorem total_trophies_correct : total_trophies_in_three_years = 430 :=
by
  sorry -- proof is omitted

end total_trophies_correct_l3_391


namespace find_total_amount_l3_315

noncomputable def total_amount (a b c : ℕ) : Prop :=
  a = 3 * b ∧ b = c + 25 ∧ b = 134 ∧ a + b + c = 645

theorem find_total_amount : ∃ a b c, total_amount a b c :=
by
  sorry

end find_total_amount_l3_315


namespace base8_to_decimal_l3_303

theorem base8_to_decimal (n : ℕ) (h : n = 54321) : 
  (5 * 8^4 + 4 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1 * 8^0) = 22737 := 
by
  sorry

end base8_to_decimal_l3_303


namespace original_number_is_142857_l3_373

-- Definitions based on conditions
def six_digit_number (x : ℕ) : ℕ := 100000 + x
def moved_digit_number (x : ℕ) : ℕ := 10 * x + 1

-- Lean statement of the equivalent problem
theorem original_number_is_142857 : ∃ x, six_digit_number x = 142857 ∧ moved_digit_number x = 3 * six_digit_number x :=
  sorry

end original_number_is_142857_l3_373


namespace number_of_people_speaking_both_languages_l3_389

theorem number_of_people_speaking_both_languages
  (total : ℕ) (L : ℕ) (F : ℕ) (N : ℕ) (B : ℕ) :
  total = 25 → L = 13 → F = 15 → N = 6 → total = L + F - B + N → B = 9 :=
by
  intros h_total h_L h_F h_N h_inclusion_exclusion
  sorry

end number_of_people_speaking_both_languages_l3_389


namespace min_positive_d_l3_318

theorem min_positive_d (a b t d : ℤ) (h1 : 3 * t = 2 * a + 2 * b + 2016)
                                       (h2 : t - a = d)
                                       (h3 : t - b = 2 * d)
                                       (h4 : 2 * a + 2 * b > 0) :
    ∃ d : ℤ, d > 0 ∧ (505 ≤ d ∧ ∀ e : ℤ, e > 0 → 3 * (a + d) = 2 * (b + 2 * e) + 2016 → 505 ≤ e) := 
sorry

end min_positive_d_l3_318


namespace quadratic_has_two_distinct_real_roots_l3_383

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2 * x1 + m = 0) ∧ (x2^2 - 2 * x2 + m = 0)) ↔ (m < 1) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l3_383


namespace yanni_paintings_l3_311

theorem yanni_paintings
  (total_area : ℤ)
  (painting1 : ℕ → ℤ × ℤ)
  (painting2 : ℤ × ℤ)
  (painting3 : ℤ × ℤ)
  (num_paintings : ℕ) :
  total_area = 200
  → painting1 1 = (5, 5)
  → painting1 2 = (5, 5)
  → painting1 3 = (5, 5)
  → painting2 = (10, 8)
  → painting3 = (5, 9)
  → num_paintings = 5 := 
by
  sorry

end yanni_paintings_l3_311


namespace expand_expression_l3_344

variable (x y : ℝ)

theorem expand_expression :
  12 * (3 * x + 4 * y - 2) = 36 * x + 48 * y - 24 :=
by
  sorry

end expand_expression_l3_344


namespace find_largest_x_and_compute_ratio_l3_377

theorem find_largest_x_and_compute_ratio (a b c d : ℤ) (h : x = (a + b * Real.sqrt c) / d)
   (cond : (5 * x / 7) + 1 = 3 / x) : a * c * d / b = -70 :=
by
  sorry

end find_largest_x_and_compute_ratio_l3_377


namespace min_value_reciprocal_l3_365

theorem min_value_reciprocal (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_eq : 2 * a + b = 4) : 
  (∀ (x : ℝ), (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = 4 -> x ≥ 1 / (2 * a * b)) -> x ≥ 1 / 2) := 
by
  sorry

end min_value_reciprocal_l3_365


namespace linda_savings_l3_347

theorem linda_savings (S : ℝ) (h : (1 / 2) * S = 300) : S = 600 :=
sorry

end linda_savings_l3_347


namespace initial_wine_volume_l3_395

theorem initial_wine_volume (x : ℝ) 
  (h₁ : ∀ k : ℝ, k = x → ∀ n : ℕ, n = 3 → 
    (∀ y : ℝ, y = k - 4 * (1 - ((k - 4) / k) ^ n) + 2.5)) :
  x = 16 := by
  sorry

end initial_wine_volume_l3_395


namespace sum_lent_is_3000_l3_346

noncomputable def principal_sum (P : ℕ) : Prop :=
  let R := 5
  let T := 5
  let SI := (P * R * T) / 100
  SI = P - 2250

theorem sum_lent_is_3000 : ∃ (P : ℕ), principal_sum P ∧ P = 3000 :=
by
  use 3000
  unfold principal_sum
  -- The following are the essential parts
  sorry

end sum_lent_is_3000_l3_346


namespace greatest_k_inequality_l3_343

theorem greatest_k_inequality :
  ∃ k : ℕ, k = 13 ∧ ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → a * b * c = 1 → 
  (1 / a + 1 / b + 1 / c + k / (a + b + c + 1) ≥ 3 + k / 4) :=
sorry

end greatest_k_inequality_l3_343


namespace lucy_total_fish_l3_360

variable (current_fish additional_fish : ℕ)

def total_fish (current_fish additional_fish : ℕ) : ℕ :=
  current_fish + additional_fish

theorem lucy_total_fish (h1 : current_fish = 212) (h2 : additional_fish = 68) : total_fish current_fish additional_fish = 280 :=
by
  sorry

end lucy_total_fish_l3_360


namespace solve_inequality_l3_371

theorem solve_inequality : { x : ℝ // (x < -1) ∨ (-2/3 < x) } :=
sorry

end solve_inequality_l3_371


namespace new_rectangle_area_eq_a_squared_l3_370

theorem new_rectangle_area_eq_a_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let d := Real.sqrt (a^2 + b^2)
  let base := 2 * (d + b)
  let height := (d - b) / 2
  base * height = a^2 := by
  sorry

end new_rectangle_area_eq_a_squared_l3_370


namespace num_male_students_selected_l3_312

def total_students := 220
def male_students := 60
def selected_female_students := 32

def selected_male_students (total_students male_students selected_female_students : Nat) : Nat :=
  (selected_female_students * male_students) / (total_students - male_students)

theorem num_male_students_selected : selected_male_students total_students male_students selected_female_students = 12 := by
  unfold selected_male_students
  sorry

end num_male_students_selected_l3_312


namespace compute_expression_l3_310

theorem compute_expression : 42 * 52 + 48 * 42 = 4200 :=
by sorry

end compute_expression_l3_310


namespace correct_equation_l3_302

theorem correct_equation (x : ℝ) (h1 : 2000 > 0) (h2 : x > 0) (h3 : x + 40 > 0) :
  (2000 / x) - (2000 / (x + 40)) = 3 :=
by
  sorry

end correct_equation_l3_302


namespace bakery_made_muffins_l3_327

-- Definitions based on conditions
def muffins_per_box : ℕ := 5
def available_boxes : ℕ := 10
def additional_boxes_needed : ℕ := 9

-- Theorem statement
theorem bakery_made_muffins :
  (available_boxes * muffins_per_box) + (additional_boxes_needed * muffins_per_box) = 95 := 
by
  sorry

end bakery_made_muffins_l3_327


namespace union_of_A_and_B_l3_314

variables (A B : Set ℤ)
variable (a : ℤ)
theorem union_of_A_and_B : (A = {4, a^2}) → (B = {a-6, 1+a, 9}) → (A ∩ B = {9}) → (A ∪ B = {-9, -2, 4, 9}) :=
by
  intros hA hB hInt
  sorry

end union_of_A_and_B_l3_314


namespace max_side_of_triangle_l3_367

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l3_367


namespace min_vitamins_sold_l3_381

theorem min_vitamins_sold (n : ℕ) (h1 : n % 11 = 0) (h2 : n % 23 = 0) (h3 : n % 37 = 0) : n = 9361 :=
by
  sorry

end min_vitamins_sold_l3_381


namespace symmetry_center_l3_354

theorem symmetry_center {φ : ℝ} (hφ : |φ| < Real.pi / 2) (h : 2 * Real.sin φ = Real.sqrt 3) : 
  ∃ x : ℝ, 2 * Real.sin (2 * x + φ) = 2 * Real.sin (- (2 * x + φ)) ∧ x = -Real.pi / 6 :=
by
  sorry

end symmetry_center_l3_354


namespace divisibility_equivalence_l3_356

theorem divisibility_equivalence (a b c d : ℤ) (h : a ≠ c) :
  (a - c) ∣ (a * b + c * d) ↔ (a - c) ∣ (a * d + b * c) :=
by
  sorry

end divisibility_equivalence_l3_356


namespace attendees_not_from_A_B_C_D_l3_361

theorem attendees_not_from_A_B_C_D
  (num_A : ℕ) (num_B : ℕ) (num_C : ℕ) (num_D : ℕ) (total_attendees : ℕ)
  (hA : num_A = 30)
  (hB : num_B = 2 * num_A)
  (hC : num_C = num_A + 10)
  (hD : num_D = num_C - 5)
  (hTotal : total_attendees = 185)
  : total_attendees - (num_A + num_B + num_C + num_D) = 20 := by
  sorry

end attendees_not_from_A_B_C_D_l3_361


namespace negation_of_no_vegetarian_students_eat_at_cafeteria_l3_380

variable (Student : Type) 
variable (isVegetarian : Student → Prop)
variable (eatsAtCafeteria : Student → Prop)

theorem negation_of_no_vegetarian_students_eat_at_cafeteria :
  (∀ x, isVegetarian x → ¬ eatsAtCafeteria x) →
  (∃ x, isVegetarian x ∧ eatsAtCafeteria x) :=
by
  sorry

end negation_of_no_vegetarian_students_eat_at_cafeteria_l3_380


namespace log_expression_value_l3_384

theorem log_expression_value (x : ℝ) (hx : x < 1) (h : (Real.log x / Real.log 10)^3 - 2 * (Real.log (x^3) / Real.log 10) = 150) :
  (Real.log x / Real.log 10)^4 - (Real.log (x^4) / Real.log 10) = 645 := 
sorry

end log_expression_value_l3_384


namespace range_of_m_l3_374

theorem range_of_m (m : ℝ) :
  ( ∀ x : ℝ, |x + m| ≤ 4 → -2 ≤ x ∧ x ≤ 8) ↔ -4 ≤ m ∧ m ≤ -2 := 
by
  sorry

end range_of_m_l3_374


namespace sin_X_value_l3_385

variables (a b X : ℝ)

-- Conditions
def conditions :=
  (1/2 * a * b * Real.sin X = 100) ∧ (Real.sqrt (a * b) = 15)

theorem sin_X_value (h : conditions a b X) : Real.sin X = 8 / 9 := by
  sorry

end sin_X_value_l3_385


namespace right_triangle_hypotenuse_l3_363

theorem right_triangle_hypotenuse (A : ℝ) (h height : ℝ) :
  A = 320 ∧ height = 16 →
  ∃ c : ℝ, c = 4 * Real.sqrt 116 :=
by
  intro h
  sorry

end right_triangle_hypotenuse_l3_363


namespace amoeba_count_14_l3_336

noncomputable def amoeba_count (day : ℕ) : ℕ :=
  if day = 1 then 1
  else if day = 2 then 2
  else 2^(day - 3) * 5

theorem amoeba_count_14 : amoeba_count 14 = 10240 := by
  sorry

end amoeba_count_14_l3_336
