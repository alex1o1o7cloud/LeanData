import Mathlib

namespace find_x_when_parallel_l2422_242290

-- Given vectors
def a : ℝ × ℝ := (-2, 4)
def b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Conditional statement: parallel vectors
def parallel_vectors (u v : ℝ × ℝ) : Prop := 
  u.1 * v.2 = u.2 * v.1

-- Proof statement
theorem find_x_when_parallel (x : ℝ) (h : parallel_vectors a (b x)) : x = 1 := 
  sorry

end find_x_when_parallel_l2422_242290


namespace jason_initial_cards_l2422_242285

theorem jason_initial_cards (a : ℕ) (b : ℕ) (x : ℕ) : 
  a = 224 → 
  b = 452 → 
  x = a + b → 
  x = 676 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jason_initial_cards_l2422_242285


namespace percentage_increase_second_year_l2422_242239

theorem percentage_increase_second_year 
  (initial_population : ℝ)
  (first_year_increase : ℝ) 
  (population_after_2_years : ℝ) 
  (final_population : ℝ)
  (H_initial_population : initial_population = 800)
  (H_first_year_increase : first_year_increase = 0.22)
  (H_population_after_2_years : final_population = 1220) :
  ∃ P : ℝ, P = 25 := 
by
  -- Define the population after the first year
  let population_after_first_year := initial_population * (1 + first_year_increase)
  -- Define the equation relating populations and solve for P
  let second_year_increase := (final_population / population_after_first_year - 1) * 100
  -- Show P equals 25
  use second_year_increase
  sorry

end percentage_increase_second_year_l2422_242239


namespace sum_of_squares_of_coefficients_l2422_242224

theorem sum_of_squares_of_coefficients :
  let p := 3 * (X^5 + 4 * X^3 + 2 * X + 1)
  let coeffs := [3, 12, 6, 3, 0, 0]
  let sum_squares := coeffs.map (λ c => c * c) |>.sum
  sum_squares = 198 := by
  sorry

end sum_of_squares_of_coefficients_l2422_242224


namespace smallest_positive_integer_div_conditions_l2422_242256

theorem smallest_positive_integer_div_conditions :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 2 ∧ x % 7 = 3 ∧ ∀ y : ℕ, (y > 0 ∧ y % 5 = 2 ∧ y % 7 = 3) → x ≤ y :=
  sorry

end smallest_positive_integer_div_conditions_l2422_242256


namespace area_of_regular_octagon_l2422_242286

/-- The perimeters of a square and a regular octagon are equal.
    The area of the square is 16.
    Prove that the area of the regular octagon is 8 + 8 * sqrt 2. -/
theorem area_of_regular_octagon (a b : ℝ) (h1 : 4 * a = 8 * b) (h2 : a^2 = 16) :
  2 * (1 + Real.sqrt 2) * b^2 = 8 + 8 * Real.sqrt 2 :=
by
  sorry

end area_of_regular_octagon_l2422_242286


namespace smallest_positive_e_l2422_242248

-- Define the polynomial and roots condition
def polynomial (a b c d e : ℤ) (x : ℝ) : ℝ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

def has_integer_roots (p : ℝ → ℝ) (roots : List ℝ) : Prop :=
  ∀ r ∈ roots, p r = 0

def polynomial_with_given_roots (a b c d e : ℤ) : Prop :=
  has_integer_roots (polynomial a b c d e) [-3, 4, 11, -(1/4)]

-- Main theorem to prove the smallest positive integer e
theorem smallest_positive_e (a b c d : ℤ) :
  ∃ e : ℤ, e > 0 ∧ polynomial_with_given_roots a b c d e ∧
            (∀ e' : ℤ, e' > 0 ∧ polynomial_with_given_roots a b c d e' → e ≤ e') :=
  sorry

end smallest_positive_e_l2422_242248


namespace minimum_value_nine_l2422_242264

noncomputable def min_value (a b c k : ℝ) : ℝ :=
  (a + b + k) / c + (a + c + k) / b + (b + c + k) / a

theorem minimum_value_nine (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  min_value a b c k ≥ 9 :=
sorry

end minimum_value_nine_l2422_242264


namespace arithmetic_mean_is_12_l2422_242294

/-- The arithmetic mean of the numbers 3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14, and 7 is equal to 12 -/
theorem arithmetic_mean_is_12 : 
  let numbers := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14, 7]
  let sum := numbers.foldl (· + ·) 0
  let count := numbers.length
  (sum / count) = 12 :=
by
  sorry

end arithmetic_mean_is_12_l2422_242294


namespace kids_outside_l2422_242258

theorem kids_outside (s t n c : ℕ)
  (h1 : s = 644997)
  (h2 : t = 893835)
  (h3 : n = 1538832)
  (h4 : (n - s) = t) : c = 0 :=
by {
  sorry
}

end kids_outside_l2422_242258


namespace find_a_l2422_242263

theorem find_a (a : ℝ) 
  (h1 : ∀ x y : ℝ, 2*x + y - 2 = 0)
  (h2 : ∀ x y : ℝ, a*x + 4*y + 1 = 0)
  (perpendicular : ∀ (m1 m2 : ℝ), m1 = -2 → m2 = -a/4 → m1 * m2 = -1) :
  a = -2 :=
sorry

end find_a_l2422_242263


namespace algebraic_expression_transformation_l2422_242291

theorem algebraic_expression_transformation (a b : ℝ) (h : ∀ x : ℝ, x^2 - 6*x + b = (x - a)^2 - 1) : b - a = 5 :=
by
  sorry

end algebraic_expression_transformation_l2422_242291


namespace total_time_spent_l2422_242203

-- Definitions based on the conditions
def number_of_chairs := 2
def number_of_tables := 2
def minutes_per_piece := 8
def total_pieces := number_of_chairs + number_of_tables

-- The statement we want to prove
theorem total_time_spent : total_pieces * minutes_per_piece = 32 :=
by
  sorry

end total_time_spent_l2422_242203


namespace mostSuitableSampleSurvey_l2422_242271

-- Conditions
def conditionA := "Security check for passengers before boarding a plane"
def conditionB := "Understanding the amount of physical exercise each classmate does per week"
def conditionC := "Interviewing job applicants for a company's recruitment process"
def conditionD := "Understanding the lifespan of a batch of light bulbs"

-- Define a predicate to determine the most suitable for a sample survey
def isMostSuitableForSampleSurvey (s : String) : Prop :=
  s = conditionD

-- Theorem statement
theorem mostSuitableSampleSurvey :
  isMostSuitableForSampleSurvey conditionD :=
by
  -- Skipping the proof for now
  sorry

end mostSuitableSampleSurvey_l2422_242271


namespace find_x_l2422_242233

theorem find_x (x : ℕ) (h : 220030 = (x + 445) * (2 * (x - 445)) + 30) : x = 555 := 
sorry

end find_x_l2422_242233


namespace sequence_a_2011_l2422_242219

noncomputable def sequence_a : ℕ → ℕ
| 0       => 2
| 1       => 3
| (n+2)   => (sequence_a (n+1) * sequence_a n) % 10

theorem sequence_a_2011 : sequence_a 2010 = 2 :=
by
  sorry

end sequence_a_2011_l2422_242219


namespace total_students_l2422_242296

-- Define n as total number of students
variable (n : ℕ)

-- Define conditions
variable (h1 : 550 ≤ n)
variable (h2 : (n / 10) + 10 ≤ n)

-- Define the proof statement
theorem total_students (h : (550 * 10 + 5) = n ∧ 
                        550 * 10 / n + 10 = 45 + n) : 
                        n = 1000 := by
  sorry

end total_students_l2422_242296


namespace min_value_x3y3z2_is_1_over_27_l2422_242252

noncomputable def min_value_x3y3z2 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) (h' : 1 / x + 1 / y + 1 / z = 9) : ℝ :=
  x^3 * y^3 * z^2

theorem min_value_x3y3z2_is_1_over_27 (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z)
  (h' : 1 / x + 1 / y + 1 / z = 9) : min_value_x3y3z2 x y z h h' = 1 / 27 :=
sorry

end min_value_x3y3z2_is_1_over_27_l2422_242252


namespace total_hats_purchased_l2422_242238

theorem total_hats_purchased (B G : ℕ) (h1 : G = 38) (h2 : 6 * B + 7 * G = 548) : B + G = 85 := 
by 
  sorry

end total_hats_purchased_l2422_242238


namespace additional_tickets_won_l2422_242278

-- Definitions from the problem
def initial_tickets : ℕ := 49
def spent_tickets : ℕ := 25
def final_tickets : ℕ := 30

-- The main statement we need to prove
theorem additional_tickets_won (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : 
  final_tickets - (initial_tickets - spent_tickets) = 6 :=
by
  sorry

end additional_tickets_won_l2422_242278


namespace average_marks_l2422_242262

/--
Given:
1. The average marks in physics (P) and mathematics (M) is 90.
2. The average marks in physics (P) and chemistry (C) is 70.
3. The student scored 110 marks in physics (P).

Prove that the average marks the student scored in the 3 subjects (P, C, M) is 70.
-/
theorem average_marks (P C M : ℝ) 
  (h1 : (P + M) / 2 = 90)
  (h2 : (P + C) / 2 = 70)
  (h3 : P = 110) : 
  (P + C + M) / 3 = 70 :=
sorry

end average_marks_l2422_242262


namespace probability_same_color_plates_l2422_242206

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end probability_same_color_plates_l2422_242206


namespace sum_ages_l2422_242266

theorem sum_ages (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 := 
by 
  sorry

end sum_ages_l2422_242266


namespace shift_sin_to_cos_l2422_242235

open Real

theorem shift_sin_to_cos:
  ∀ x: ℝ, 3 * cos (2 * x) = 3 * sin (2 * (x + π / 6) - π / 6) :=
by 
  sorry

end shift_sin_to_cos_l2422_242235


namespace find_a4_l2422_242280

variable {a_n : ℕ → ℕ}
variable {S : ℕ → ℕ}

def is_arithmetic_sequence (a_n : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def sum_first_n_terms (S : ℕ → ℕ) (a_n : ℕ → ℕ) :=
  ∀ n : ℕ, S n = (n * (a_n 1 + a_n n)) / 2

theorem find_a4 (h : S 7 = 35) (hs : sum_first_n_terms S a_n) (ha : is_arithmetic_sequence a_n) : a_n 4 = 5 := 
  by sorry

end find_a4_l2422_242280


namespace Kiera_envelopes_l2422_242276

theorem Kiera_envelopes (blue yellow green : ℕ) (total_envelopes : ℕ) 
  (cond1 : blue = 14) 
  (cond2 : total_envelopes = 46) 
  (cond3 : green = 3 * yellow) 
  (cond4 : total_envelopes = blue + yellow + green) : yellow = 6 - 8 := 
by sorry

end Kiera_envelopes_l2422_242276


namespace speech_competition_sequences_l2422_242267

theorem speech_competition_sequences
    (contestants : Fin 5 → Prop)
    (girls boys : Fin 5 → Prop)
    (girl_A : Fin 5)
    (not_girl_A_first : ¬contestants 0)
    (no_consecutive_boys : ∀ i, boys i → ¬boys (i + 1))
    (count_girls : ∀ x, girls x → x = girl_A ∨ (contestants x ∧ ¬boys x))
    (count_boys : ∀ x, (boys x) → contestants x)
    (total_count : Fin 5 → Fin 5 → ℕ)
    (correct_answer : total_count = 276) : 
    ∃ seq_count, seq_count = 276 := 
sorry

end speech_competition_sequences_l2422_242267


namespace john_spent_on_candy_l2422_242205

theorem john_spent_on_candy (M : ℝ) 
  (h1 : M = 29.999999999999996)
  (h2 : 1/5 + 1/3 + 1/10 = 19/30) :
  (11 / 30) * M = 11 :=
by {
  sorry
}

end john_spent_on_candy_l2422_242205


namespace sqrt_expression_equals_l2422_242243

theorem sqrt_expression_equals : Real.sqrt (5^2 * 7^4) = 245 :=
by
  sorry

end sqrt_expression_equals_l2422_242243


namespace largest_determinable_1986_l2422_242283

-- Define main problem with conditions
def largest_determinable_cards (total : ℕ) (select : ℕ) : ℕ :=
  total - 27

-- Statement we need to prove
theorem largest_determinable_1986 :
  largest_determinable_cards 2013 10 = 1986 :=
by
  sorry

end largest_determinable_1986_l2422_242283


namespace horse_distribution_l2422_242232

variable (b₁ b₂ b₃ : ℕ) 
variable (a : Matrix (Fin 3) (Fin 3) ℝ)
variable (h1 : a 0 0 > a 0 1 ∧ a 0 0 > a 0 2)
variable (h2 : a 1 1 > a 1 0 ∧ a 1 1 > a 1 2)
variable (h3 : a 2 2 > a 2 0 ∧ a 2 2 > a 2 1)

theorem horse_distribution :
  ∃ n : ℕ, ∀ (b₁ b₂ b₃ : ℕ), min b₁ (min b₂ b₃) > n → 
  ∃ (x1 y1 x2 y2 x3 y3 : ℕ), 3*x1 + y1 = b₁ ∧ 3*x2 + y2 = b₂ ∧ 3*x3 + y3 = b₃ ∧
  y1*a 0 0 > y2*a 0 1 ∧ y1*a 0 0 > y3*a 0 2 ∧
  y2*a 1 1 > y1*a 1 0 ∧ y2*a 1 1 > y3*a 1 2 ∧
  y3*a 2 2 > y1*a 2 0 ∧ y3*a 2 2 > y2*a 2 1 :=
sorry

end horse_distribution_l2422_242232


namespace decrease_in_area_of_equilateral_triangle_l2422_242268

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

theorem decrease_in_area_of_equilateral_triangle :
  (equilateral_triangle_area 20 - equilateral_triangle_area 14) = 51 * Real.sqrt 3 := by
  sorry

end decrease_in_area_of_equilateral_triangle_l2422_242268


namespace calc_problem_l2422_242221

def odot (a b : ℕ) : ℕ := a * b - (a + b)

theorem calc_problem : odot 6 (odot 5 4) = 49 :=
by
  sorry

end calc_problem_l2422_242221


namespace find_a5_l2422_242201

-- Sequence definition
def a : ℕ → ℤ
| 0     => 1
| (n+1) => 2 * a n + 3

-- Theorem to prove
theorem find_a5 : a 4 = 61 := sorry

end find_a5_l2422_242201


namespace find_f_expression_l2422_242260

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_expression (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) : 
  f (x) = (1 / (x - 1)) :=
by sorry

example (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) (hx: f (1 / x) = x / (1 - x)) :
  f x = 1 / (x - 1) :=
find_f_expression x h₀ h₁

end find_f_expression_l2422_242260


namespace sum_of_consecutive_odds_eq_power_l2422_242214

theorem sum_of_consecutive_odds_eq_power (n : ℕ) (k : ℕ) (hn : n > 0) (hk : k ≥ 2) :
  ∃ a : ℤ, n * (2 * a + n) = n^k ∧
            (∀ i : ℕ, i < n → 2 * a + 2 * (i : ℤ) + 1 = 2 * a + 1 + 2 * i) :=
by
  sorry

end sum_of_consecutive_odds_eq_power_l2422_242214


namespace find_number_l2422_242269

theorem find_number (x : ℝ) (h : (2 * x - 37 + 25) / 8 = 5) : x = 26 :=
sorry

end find_number_l2422_242269


namespace inverse_function_of_13_l2422_242211

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4
noncomputable def f_inv (y : ℝ) : ℝ := (y - 4) / 3

theorem inverse_function_of_13 : f_inv (f_inv 13) = -1 / 3 := by
  sorry

end inverse_function_of_13_l2422_242211


namespace measure_of_angle_F_l2422_242220

theorem measure_of_angle_F {D E F : ℝ}
  (isosceles : D = E)
  (angle_F_condition : F = D + 40)
  (sum_of_angles : D + E + F = 180) :
  F = 260 / 3 :=
by
  sorry

end measure_of_angle_F_l2422_242220


namespace car_bus_initial_speed_l2422_242259

theorem car_bus_initial_speed {d : ℝ} {t : ℝ} {s_c : ℝ} {s_b : ℝ}
    (h1 : t = 4) 
    (h2 : s_c = s_b + 8) 
    (h3 : d = 384)
    (h4 : ∀ t, 0 ≤ t → t ≤ 2 → d = s_c * t + s_b * t) 
    (h5 : ∀ t, 2 < t → t ≤ 4 → d = (s_c - 10) * (t - 2) + s_b * (t - 2)) 
    : s_b = 46.5 ∧ s_c = 54.5 := 
by 
    sorry

end car_bus_initial_speed_l2422_242259


namespace time_to_traverse_nth_mile_l2422_242270

theorem time_to_traverse_nth_mile (n : ℕ) (h : n ≥ 2) :
  (∃ k : ℝ, (∀ d : ℝ, d = n - 1 → (s_n = k / d)) ∧ (s_2 = 1 / 2)) → 
  t_n = 2 * (n - 1) :=
by 
  sorry

end time_to_traverse_nth_mile_l2422_242270


namespace vacation_cost_in_usd_l2422_242204

theorem vacation_cost_in_usd :
  let n := 7
  let rent_per_person_eur := 65
  let transport_per_person_usd := 25
  let food_per_person_gbp := 50
  let activities_per_person_jpy := 2750
  let eur_to_usd := 1.20
  let gbp_to_usd := 1.40
  let jpy_to_usd := 0.009
  let total_rent_usd := n * rent_per_person_eur * eur_to_usd
  let total_transport_usd := n * transport_per_person_usd
  let total_food_usd := n * food_per_person_gbp * gbp_to_usd
  let total_activities_usd := n * activities_per_person_jpy * jpy_to_usd
  let total_cost_usd := total_rent_usd + total_transport_usd + total_food_usd + total_activities_usd
  total_cost_usd = 1384.25 := by
    sorry

end vacation_cost_in_usd_l2422_242204


namespace find_f_4_l2422_242284

noncomputable def f (x : ℕ) (a b c : ℕ) : ℕ := 2 * a * x + b * x + c

theorem find_f_4
  (a b c : ℕ)
  (f1 : f 1 a b c = 10)
  (f2 : f 2 a b c = 20) :
  f 4 a b c = 40 :=
sorry

end find_f_4_l2422_242284


namespace KeatonAnnualEarnings_l2422_242245

-- Keaton's conditions for oranges
def orangeHarvestInterval : ℕ := 2
def orangeSalePrice : ℕ := 50

-- Keaton's conditions for apples
def appleHarvestInterval : ℕ := 3
def appleSalePrice : ℕ := 30

-- Annual earnings calculation
def annualEarnings (monthsInYear : ℕ) : ℕ :=
  let orangeEarnings := (monthsInYear / orangeHarvestInterval) * orangeSalePrice
  let appleEarnings := (monthsInYear / appleHarvestInterval) * appleSalePrice
  orangeEarnings + appleEarnings

-- Prove the total annual earnings is 420
theorem KeatonAnnualEarnings : annualEarnings 12 = 420 :=
  by 
    -- We skip the proof details here.
    sorry

end KeatonAnnualEarnings_l2422_242245


namespace injective_of_comp_injective_surjective_of_comp_surjective_l2422_242272

section FunctionProperties

variables {X Y V : Type} (f : X → Y) (g : Y → V)

-- Proof for part (i) if g ∘ f is injective, then f is injective
theorem injective_of_comp_injective (h : Function.Injective (g ∘ f)) : Function.Injective f :=
  sorry

-- Proof for part (ii) if g ∘ f is surjective, then g is surjective
theorem surjective_of_comp_surjective (h : Function.Surjective (g ∘ f)) : Function.Surjective g :=
  sorry

end FunctionProperties

end injective_of_comp_injective_surjective_of_comp_surjective_l2422_242272


namespace smallest_int_k_for_64_pow_k_l2422_242241

theorem smallest_int_k_for_64_pow_k (k : ℕ) (base : ℕ) (h₁ : k = 7) : 
  64^k > base^20 → base = 4 := by
  sorry

end smallest_int_k_for_64_pow_k_l2422_242241


namespace bc_ad_divisible_by_u_l2422_242292

theorem bc_ad_divisible_by_u 
  (a b c d u : ℤ) 
  (h1 : u ∣ a * c)
  (h2 : u ∣ b * c + a * d)
  (h3 : u ∣ b * d) : 
  u ∣ b * c ∧ u ∣ a * d :=
by
  sorry

end bc_ad_divisible_by_u_l2422_242292


namespace uncle_ben_parking_probability_l2422_242261

theorem uncle_ben_parking_probability :
  let total_spaces := 20
  let cars := 15
  let rv_spaces := 3
  let total_combinations := Nat.choose total_spaces cars
  let non_adjacent_empty_combinations := Nat.choose (total_spaces - rv_spaces) cars
  (1 - (non_adjacent_empty_combinations / total_combinations)) = (232 / 323) := by
  sorry

end uncle_ben_parking_probability_l2422_242261


namespace find_angle_D_l2422_242209

theorem find_angle_D (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 40) (h4 : B + C = 130) : D = 40 := by
  sorry

end find_angle_D_l2422_242209


namespace quadrilateral_is_parallelogram_l2422_242236

theorem quadrilateral_is_parallelogram (a b c d : ℝ) (h : (a - c) ^ 2 + (b - d) ^ 2 = 0) : 
  -- The theorem states that if lengths a, b, c, d of a quadrilateral satisfy the given equation,
  -- then the quadrilateral must be a parallelogram.
  a = c ∧ b = d :=
by {
  sorry
}

end quadrilateral_is_parallelogram_l2422_242236


namespace smallest_six_digit_divisible_by_111_l2422_242277

theorem smallest_six_digit_divisible_by_111 : ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 ∧ n = 100011 :=
by {
  sorry
}

end smallest_six_digit_divisible_by_111_l2422_242277


namespace problem_l2422_242287

noncomputable def M (x y z : ℝ) : ℝ :=
  (Real.sqrt (x^2 + x * y + y^2) * Real.sqrt (y^2 + y * z + z^2)) +
  (Real.sqrt (y^2 + y * z + z^2) * Real.sqrt (z^2 + z * x + x^2)) +
  (Real.sqrt (z^2 + z * x + x^2) * Real.sqrt (x^2 + x * y + y^2))

theorem problem (x y z : ℝ) (α β : ℝ) 
  (h1 : ∀ x y z, α * (x * y + y * z + z * x) ≤ M x y z)
  (h2 : ∀ x y z, M x y z ≤ β * (x^2 + y^2 + z^2)) :
  (∀ α, α ≤ 3) ∧ (∀ β, β ≥ 3) :=
sorry

end problem_l2422_242287


namespace balancing_point_is_vertex_l2422_242255

-- Define a convex polygon and its properties
structure ConvexPolygon (n : ℕ) :=
(vertices : Fin n → Point)

-- Define a balancing point for a convex polygon
def is_balancing_point (P : ConvexPolygon n) (Q : Point) : Prop :=
  -- Placeholder for the actual definition that the areas formed by drawing lines from Q to vertices of P are equal
  sorry

-- Define the uniqueness of the balancing point
def unique_balancing_point (P : ConvexPolygon n) (Q : Point) : Prop :=
  ∀ R : Point, is_balancing_point P R → R = Q

-- Main theorem statement
theorem balancing_point_is_vertex (P : ConvexPolygon n) (Q : Point) 
  (h_balance : is_balancing_point P Q) (h_unique : unique_balancing_point P Q) : 
  ∃ i : Fin n, Q = P.vertices i :=
sorry

end balancing_point_is_vertex_l2422_242255


namespace find_f_7_l2422_242244

noncomputable def f (a b c d x : ℝ) : ℝ :=
  a * x^8 + b * x^7 + c * x^3 + d * x - 6

theorem find_f_7 (a b c d : ℝ) (h : f a b c d (-7) = 10) :
  f a b c d 7 = 11529580 * a - 22 :=
sorry

end find_f_7_l2422_242244


namespace area_ratio_l2422_242237

variables {A B C D: Type} [LinearOrderedField A]
variables {AB AD AR AE : A}

-- Conditions
axiom cond1 : AR = (2 / 3) * AB
axiom cond2 : AE = (1 / 3) * AD

theorem area_ratio (h : A) (h1 : A) (S_ABCD : A) (S_ARE : A)
  (h_eq : S_ABCD = AD * h)
  (h1_eq : S_ARE = (1 / 2) * AE * h1)
  (ratio_heights : h / h1 = 3 / 2) :
  S_ABCD / S_ARE = 9 :=
by {
  sorry
}

end area_ratio_l2422_242237


namespace min_abc_sum_l2422_242218

theorem min_abc_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 8) : a + b + c ≥ 6 :=
by {
  sorry
}

end min_abc_sum_l2422_242218


namespace Basel_series_l2422_242257

theorem Basel_series :
  (∑' (n : ℕ+), 1 / (n : ℝ)^2) = π^2 / 6 := by sorry

end Basel_series_l2422_242257


namespace trajectory_is_line_segment_l2422_242274

theorem trajectory_is_line_segment : 
  ∃ (P : ℝ × ℝ) (F1 F2: ℝ × ℝ), 
    F1 = (-3, 0) ∧ F2 = (3, 0) ∧ (|F1.1 - P.1|^2 + |F1.2 - P.2|^2).sqrt + (|F2.1 - P.1|^2 + |F2.2 - P.2|^2).sqrt = 6
  → (P.1 = F1.1 ∨ P.1 = F2.1) ∧ (P.2 = F1.2 ∨ P.2 = F2.2) :=
by sorry

end trajectory_is_line_segment_l2422_242274


namespace gumball_probability_l2422_242225

theorem gumball_probability :
  let total_gumballs : ℕ := 25
  let orange_gumballs : ℕ := 10
  let green_gumballs : ℕ := 6
  let yellow_gumballs : ℕ := 9
  let total_gumballs_after_first : ℕ := total_gumballs - 1
  let total_gumballs_after_second : ℕ := total_gumballs - 2
  let orange_probability_first : ℚ := orange_gumballs / total_gumballs
  let green_or_yellow_probability_second : ℚ := (green_gumballs + yellow_gumballs) / total_gumballs_after_first
  let orange_probability_third : ℚ := (orange_gumballs - 1) / total_gumballs_after_second
  orange_probability_first * green_or_yellow_probability_second * orange_probability_third = 9 / 92 :=
by
  sorry

end gumball_probability_l2422_242225


namespace triangle_side_m_l2422_242246

theorem triangle_side_m (a b m : ℝ) (ha : a = 2) (hb : b = 3) (h1 : a + b > m) (h2 : a + m > b) (h3 : b + m > a) :
  (1 < m ∧ m < 5) → m = 3 :=
by
  sorry

end triangle_side_m_l2422_242246


namespace product_B_original_price_l2422_242250

variable (a b : ℝ)

theorem product_B_original_price (h1 : a = 1.2 * b) (h2 : 0.9 * a = 198) : b = 183.33 :=
by
  sorry

end product_B_original_price_l2422_242250


namespace red_shoes_drawn_l2422_242200

-- Define the main conditions
def total_shoes : ℕ := 8
def red_shoes : ℕ := 4
def green_shoes : ℕ := 4
def probability_red : ℝ := 0.21428571428571427

-- Problem statement in Lean
theorem red_shoes_drawn (x : ℕ) (hx : ↑x / total_shoes = probability_red) : x = 2 := by
  sorry

end red_shoes_drawn_l2422_242200


namespace geometric_sequence_ratio_l2422_242215

/-
Given a geometric sequence {a_n} with common ratio q ≠ -1 and q ≠ 1,
and S_n is the sum of the first n terms of the geometric sequence.
Given S_{12} = 7 S_{4}, prove:
S_{8}/S_{4} = 3
-/

theorem geometric_sequence_ratio {a_n : ℕ → ℝ} (q : ℝ) (h₁ : q ≠ -1) (h₂ : q ≠ 1)
  (S : ℕ → ℝ) (hSn : ∀ n, S n = a_n 0 * (1 - q ^ n) / (1 - q)) (h : S 12 = 7 * S 4) :
  S 8 / S 4 = 3 :=
by
  sorry

end geometric_sequence_ratio_l2422_242215


namespace integer_mod_105_l2422_242240

theorem integer_mod_105 (x : ℤ) :
  (4 + x ≡ 2 * 2 [ZMOD 3^3]) →
  (6 + x ≡ 3 * 3 [ZMOD 5^3]) →
  (8 + x ≡ 5 * 5 [ZMOD 7^3]) →
  x % 105 = 3 :=
by
  sorry

end integer_mod_105_l2422_242240


namespace mailman_total_pieces_l2422_242227

def piecesOfMailFirstHouse := 6 + 5 + 3 + 4 + 2
def piecesOfMailSecondHouse := 4 + 7 + 2 + 5 + 3
def piecesOfMailThirdHouse := 8 + 3 + 4 + 6 + 1

def totalPiecesOfMail := piecesOfMailFirstHouse + piecesOfMailSecondHouse + piecesOfMailThirdHouse

theorem mailman_total_pieces : totalPiecesOfMail = 63 := by
  sorry

end mailman_total_pieces_l2422_242227


namespace problem_acute_angles_l2422_242222

theorem problem_acute_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (h1 : 3 * (Real.sin α) ^ 2 + 2 * (Real.sin β) ^ 2 = 1)
  (h2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π / 2 := 
by 
  sorry

end problem_acute_angles_l2422_242222


namespace isosceles_triangle_l2422_242210

theorem isosceles_triangle
  (α β γ : ℝ)
  (triangle_sum : α + β + γ = Real.pi)
  (second_triangle_angle1 : α + β < Real.pi)
  (second_triangle_angle2 : α + γ < Real.pi) :
  β = γ := 
sorry

end isosceles_triangle_l2422_242210


namespace eccentricity_of_hyperbola_l2422_242281

variables (a b c e : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c = Real.sqrt (a^2 + b^2))
variable (h4 : 3 * -(a^2 / c) + c = a^2 * c / (b^2 - a^2) + c)
variable (h5 : e = c / a)

theorem eccentricity_of_hyperbola : e = Real.sqrt 3 :=
by {
  sorry
}

end eccentricity_of_hyperbola_l2422_242281


namespace pascal_triangle_ratio_l2422_242288

theorem pascal_triangle_ratio (n r : ℕ) 
  (h1 : (3 * r + 3 = 2 * n - 2 * r))
  (h2 : (4 * r + 8 = 3 * n - 3 * r - 3)) : 
  n = 34 :=
sorry

end pascal_triangle_ratio_l2422_242288


namespace obtuse_dihedral_angles_l2422_242282

theorem obtuse_dihedral_angles (AOB BOC COA : ℝ) (h1 : AOB > 90) (h2 : BOC > 90) (h3 : COA > 90) :
  ∃ α β γ : ℝ, α > 90 ∧ β > 90 ∧ γ > 90 :=
sorry

end obtuse_dihedral_angles_l2422_242282


namespace volume_of_box_l2422_242228

theorem volume_of_box (l w h : ℝ) (hlw : l * w = 36) (hwh : w * h = 18) (hlh : l * h = 8) : 
    l * w * h = 72 := 
by 
    sorry

end volume_of_box_l2422_242228


namespace two_primes_equal_l2422_242279

theorem two_primes_equal
  (a b c : ℕ)
  (p q r : ℕ)
  (hp : p = b^c + a ∧ Nat.Prime p)
  (hq : q = a^b + c ∧ Nat.Prime q)
  (hr : r = c^a + b ∧ Nat.Prime r) :
  p = q ∨ q = r ∨ r = p := 
sorry

end two_primes_equal_l2422_242279


namespace z_value_l2422_242254

theorem z_value (z : ℝ) (h : |z + 2| = |z - 3|) : z = 1 / 2 := 
sorry

end z_value_l2422_242254


namespace umbrella_cost_l2422_242229

theorem umbrella_cost (house_umbrellas car_umbrellas cost_per_umbrella : ℕ) (h1 : house_umbrellas = 2) (h2 : car_umbrellas = 1) (h3 : cost_per_umbrella = 8) : 
  (house_umbrellas + car_umbrellas) * cost_per_umbrella = 24 := 
by
  sorry

end umbrella_cost_l2422_242229


namespace find_polynomial_l2422_242208

-- Define the polynomial function and the constant
variables {F : Type*} [Field F]

-- The main condition of the problem
def satisfies_condition (p : F → F) (c : F) :=
  ∀ x : F, p (p x) = x * p x + c * x^2

-- Prove the correct answers
theorem find_polynomial (p : F → F) (c : F) : 
  (c = 0 → ∀ x, p x = x) ∧ (c = -2 → ∀ x, p x = -x) :=
by
  sorry

end find_polynomial_l2422_242208


namespace partition_displacement_l2422_242234

variables (l : ℝ) (R T : ℝ) (initial_V1 initial_V2 final_Vleft final_Vright initial_P1 initial_P2 : ℝ)

-- Conditions
def initial_conditions (initial_V1 initial_V2 : ℝ) : Prop :=
  initial_V1 + initial_V2 = l ∧
  initial_V2 = 2 * initial_V1 ∧
  initial_P1 * initial_V1 = R * T ∧
  initial_P2 * initial_V2 = 2 * R * T ∧
  initial_P1 = initial_P2

-- Final volumes
def final_volumes (final_Vleft final_Vright : ℝ) : Prop :=
  final_Vleft = l / 2 ∧ final_Vright = l / 2 

-- Displacement of the partition
def displacement (initial_position final_position : ℝ) : ℝ :=
  initial_position - final_position

-- Theorem statement: the displacement of the partition is l / 6
theorem partition_displacement (l R T initial_V1 initial_V2 final_Vleft final_Vright initial_P1 initial_P2 : ℝ)
  (h_initial_cond : initial_conditions l R T initial_V1 initial_V2 initial_P1 initial_P2)
  (h_final_vol : final_volumes l final_Vleft final_Vright) 
  (initial_position final_position : ℝ)
  (initial_position_def : initial_position = 2 * l / 3)
  (final_position_def : final_position = l / 2) :
  displacement initial_position final_position = l / 6 := 
by sorry

end partition_displacement_l2422_242234


namespace find_slant_height_l2422_242297

-- Definitions of the given conditions
variable (r1 r2 L A1 A2 : ℝ)
variable (π : ℝ := Real.pi)

-- The conditions as given in the problem
def conditions : Prop := 
  r1 = 3 ∧ r2 = 4 ∧ 
  (π * L * (r1 + r2) = A1 + A2) ∧ 
  (A1 = π * r1^2) ∧ 
  (A2 = π * r2^2)

-- The theorem stating the question and the correct answer
theorem find_slant_height (h : conditions r1 r2 L A1 A2) : 
  L = 5 := 
sorry

end find_slant_height_l2422_242297


namespace fish_market_customers_l2422_242207

theorem fish_market_customers :
  let num_tuna := 10
  let weight_per_tuna := 200
  let weight_per_customer := 25
  let num_customers_no_fish := 20
  let total_tuna_weight := num_tuna * weight_per_tuna
  let num_customers_served := total_tuna_weight / weight_per_customer
  num_customers_served + num_customers_no_fish = 100 := 
by
  sorry

end fish_market_customers_l2422_242207


namespace sharp_triple_72_l2422_242298

-- Definition of the transformation function
def sharp (N : ℝ) : ℝ := 0.4 * N + 3

-- Theorem statement
theorem sharp_triple_72 : sharp (sharp (sharp 72)) = 9.288 := by
  sorry

end sharp_triple_72_l2422_242298


namespace friday_vs_tuesday_l2422_242249

def tuesday_amount : ℝ := 8.5
def wednesday_amount : ℝ := 5.5 * tuesday_amount
def thursday_amount : ℝ := wednesday_amount + 0.10 * wednesday_amount
def friday_amount : ℝ := 0.75 * thursday_amount

theorem friday_vs_tuesday :
  friday_amount - tuesday_amount = 30.06875 :=
sorry

end friday_vs_tuesday_l2422_242249


namespace sum_first_5n_eq_630_l2422_242242

theorem sum_first_5n_eq_630 (n : ℕ)
  (h : (4 * n * (4 * n + 1)) / 2 = (2 * n * (2 * n + 1)) / 2 + 300) :
  (5 * n * (5 * n + 1)) / 2 = 630 :=
sorry

end sum_first_5n_eq_630_l2422_242242


namespace daniel_practices_each_school_day_l2422_242253

-- Define the conditions
def total_minutes : ℕ := 135
def school_days : ℕ := 5
def weekend_days : ℕ := 2

-- Define the variables
def x : ℕ := 15

-- Define the practice time equations
def school_week_practice_time (x : ℕ) := school_days * x
def weekend_practice_time (x : ℕ) := weekend_days * 2 * x
def total_practice_time (x : ℕ) := school_week_practice_time x + weekend_practice_time x

-- The proof goal
theorem daniel_practices_each_school_day :
  total_practice_time x = total_minutes := by
  sorry

end daniel_practices_each_school_day_l2422_242253


namespace basis_service_B_l2422_242202

def vector := ℤ × ℤ

def not_collinear (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 ≠ v1.2 * v2.1

def A : vector × vector := ((0, 0), (2, 3))
def B : vector × vector := ((-1, 3), (5, -2))
def C : vector × vector := ((3, 4), (6, 8))
def D : vector × vector := ((2, -3), (-2, 3))

theorem basis_service_B : not_collinear B.1 B.2 := by
  sorry

end basis_service_B_l2422_242202


namespace neither_probability_l2422_242226

-- Definitions of the probabilities P(A), P(B), and P(A ∩ B)
def P_A : ℝ := 0.63
def P_B : ℝ := 0.49
def P_A_and_B : ℝ := 0.32

-- Definition stating the probability of neither event
theorem neither_probability :
  (1 - (P_A + P_B - P_A_and_B)) = 0.20 := 
sorry

end neither_probability_l2422_242226


namespace smallest_integer_x_l2422_242289

theorem smallest_integer_x (x : ℤ) (h : x < 3 * x - 12) : x ≥ 7 :=
sorry

end smallest_integer_x_l2422_242289


namespace find_new_bottle_caps_l2422_242275

theorem find_new_bottle_caps (initial caps_thrown current : ℕ) (h_initial : initial = 69)
  (h_thrown : caps_thrown = 60) (h_current : current = 67) :
  ∃ n, initial - caps_thrown + n = current ∧ n = 58 := by
sorry

end find_new_bottle_caps_l2422_242275


namespace lost_card_number_l2422_242293

theorem lost_card_number (n x : ℕ) (sum_n : ℕ) (h_sum : sum_n = n * (n + 1) / 2) (h_remaining_sum : sum_n - x = 101) : x = 4 :=
sorry

end lost_card_number_l2422_242293


namespace solve_fraction_equation_l2422_242230

theorem solve_fraction_equation (x : ℝ) (h : x ≠ 3) : (x^2 - 9) / (x - 3) = 0 ↔ x = -3 :=
sorry

end solve_fraction_equation_l2422_242230


namespace value_of_x_when_y_equals_8_l2422_242213

noncomputable def inverse_variation(cube_root : ℝ → ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y * (cube_root x) = k

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem value_of_x_when_y_equals_8 : 
  ∃ k : ℝ, (inverse_variation cube_root k 8 2) → 
  (inverse_variation cube_root k (1 / 8) 8) := 
sorry

end value_of_x_when_y_equals_8_l2422_242213


namespace number_of_rowers_l2422_242295

theorem number_of_rowers (total_coaches : ℕ) (votes_per_coach : ℕ) (votes_per_rower : ℕ) 
  (htotal_coaches : total_coaches = 36) (hvotes_per_coach : votes_per_coach = 5) 
  (hvotes_per_rower : votes_per_rower = 3) : 
  (total_coaches * votes_per_coach) / votes_per_rower = 60 :=
by 
  sorry

end number_of_rowers_l2422_242295


namespace tennis_tournament_matches_l2422_242231

theorem tennis_tournament_matches (n : ℕ) (h₁ : n = 128) (h₂ : ∃ m : ℕ, m = 32) (h₃ : ∃ k : ℕ, k = 96) (h₄ : ∀ i : ℕ, i > 1 → i ≤ n → ∃ j : ℕ, j = 1 + (i - 1)) :
  ∃ total_matches : ℕ, total_matches = 127 := 
by 
  sorry

end tennis_tournament_matches_l2422_242231


namespace tan_zero_l2422_242223

theorem tan_zero : Real.tan 0 = 0 := 
by
  sorry

end tan_zero_l2422_242223


namespace product_of_three_consecutive_integers_is_square_l2422_242247

theorem product_of_three_consecutive_integers_is_square (x : ℤ) : 
  ∃ n : ℤ, x * (x + 1) * (x + 2) = n^2 → x = 0 ∨ x = -1 ∨ x = -2 :=
by
  sorry

end product_of_three_consecutive_integers_is_square_l2422_242247


namespace correct_equation_l2422_242273

-- Define the conditions
variables {x : ℝ}

-- Condition 1: The unit price of a notebook is 2 yuan less than that of a water-based pen.
def notebook_price (water_pen_price : ℝ) : ℝ := water_pen_price - 2

-- Condition 2: Xiaogang bought 5 notebooks and 3 water-based pens for exactly 14 yuan.
def total_cost (notebook_price water_pen_price : ℝ) : ℝ :=
  5 * notebook_price + 3 * water_pen_price

-- Question restated as a theorem: Verify the given equation is correct
theorem correct_equation (water_pen_price : ℝ) (h : total_cost (notebook_price water_pen_price) water_pen_price = 14) :
  5 * (water_pen_price - 2) + 3 * water_pen_price = 14 :=
  by
    -- Introduce the assumption
    intros
    -- Sorry to skip the proof
    sorry

end correct_equation_l2422_242273


namespace largest_possible_red_socks_l2422_242299

theorem largest_possible_red_socks (t r g : ℕ) (h1 : t = r + g) (h2 : t ≤ 3000)
    (h3 : (r * (r - 1) + g * (g - 1)) * 5 = 3 * t * (t - 1)) :
    r ≤ 1199 :=
sorry

end largest_possible_red_socks_l2422_242299


namespace liquid_X_percentage_in_B_l2422_242265

noncomputable def percentage_of_solution_B (X_A : ℝ) (w_A w_B total_X : ℝ) : ℝ :=
  let X_B := (total_X - (w_A * (X_A / 100))) / w_B 
  X_B * 100

theorem liquid_X_percentage_in_B :
  percentage_of_solution_B 0.8 500 700 19.92 = 2.274 := by
  sorry

end liquid_X_percentage_in_B_l2422_242265


namespace max_monthly_profit_l2422_242251

theorem max_monthly_profit (x : ℝ) (h : 0 < x ∧ x ≤ 15) :
  let C := 100 + 4 * x
  let p := 76 + 15 * x - x^2
  let L := p * x - C
  L = -x^3 + 15 * x^2 + 72 * x - 100 ∧
  (∀ x, 0 < x ∧ x ≤ 15 → L ≤ -12^3 + 15 * 12^2 + 72 * 12 - 100) :=
by
  sorry

end max_monthly_profit_l2422_242251


namespace polynomial_degree_l2422_242217

noncomputable def polynomial1 : Polynomial ℤ := 3 * Polynomial.monomial 5 1 + 2 * Polynomial.monomial 4 1 - Polynomial.monomial 1 1 + Polynomial.C 5
noncomputable def polynomial2 : Polynomial ℤ := 4 * Polynomial.monomial 11 1 - 2 * Polynomial.monomial 8 1 + 5 * Polynomial.monomial 5 1 - Polynomial.C 9
noncomputable def polynomial3 : Polynomial ℤ := (Polynomial.monomial 2 1 - Polynomial.C 3) ^ 9

theorem polynomial_degree :
  (polynomial1 * polynomial2 - polynomial3).degree = 18 := by
  sorry

end polynomial_degree_l2422_242217


namespace linear_function_y1_greater_y2_l2422_242216

theorem linear_function_y1_greater_y2 :
  ∀ (y_1 y_2 : ℝ), 
    (y_1 = -(-1) + 6) → (y_2 = -(2) + 6) → y_1 > y_2 :=
by
  intros y_1 y_2 h1 h2
  sorry

end linear_function_y1_greater_y2_l2422_242216


namespace rect_area_perimeter_l2422_242212

def rect_perimeter (Length Width : ℕ) : ℕ :=
  2 * (Length + Width)

theorem rect_area_perimeter (Area Length : ℕ) (hArea : Area = 192) (hLength : Length = 24) :
  ∃ (Width Perimeter : ℕ), Width = Area / Length ∧ Perimeter = rect_perimeter Length Width ∧ Perimeter = 64 :=
by
  sorry

end rect_area_perimeter_l2422_242212
