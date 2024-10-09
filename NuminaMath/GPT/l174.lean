import Mathlib

namespace obtuse_angled_triangles_in_polygon_l174_17426

/-- The number of obtuse-angled triangles formed by the vertices of a regular polygon with 2n+1 sides -/
theorem obtuse_angled_triangles_in_polygon (n : ℕ) : 
  (2 * n + 1) * (n * (n - 1)) / 2 = (2 * n + 1) * (n * (n - 1)) / 2 :=
by
  sorry

end obtuse_angled_triangles_in_polygon_l174_17426


namespace challenge_Jane_l174_17489

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def card_pairs : List (Char ⊕ ℕ) :=
  [Sum.inl 'A', Sum.inl 'T', Sum.inl 'U', Sum.inr 5, Sum.inr 8, Sum.inr 10, Sum.inr 14]

def Jane_claim (c : Char ⊕ ℕ) : Prop :=
  match c with
  | Sum.inl v => is_vowel v → ∃ n, Sum.inr n ∈ card_pairs ∧ is_even n
  | Sum.inr n => false

theorem challenge_Jane (cards : List (Char ⊕ ℕ)) (h : card_pairs = cards) :
  ∃ c ∈ cards, c = Sum.inr 5 ∧ ¬Jane_claim (Sum.inr 5) :=
sorry

end challenge_Jane_l174_17489


namespace infinitely_many_primes_l174_17468

theorem infinitely_many_primes : ∀ (p : ℕ) (h_prime : Nat.Prime p), ∃ (q : ℕ), Nat.Prime q ∧ q > p :=
by
  sorry

end infinitely_many_primes_l174_17468


namespace standard_equation_of_ellipse_l174_17488

theorem standard_equation_of_ellipse :
  ∀ (m n : ℝ), 
    (m > 0 ∧ n > 0) →
    (∃ (c : ℝ), c^2 = m^2 - n^2 ∧ c = 2) →
    (∃ (e : ℝ), e = c / m ∧ e = 1 / 2) →
    (m = 4 ∧ n = 2 * Real.sqrt 3) →
    (∀ x y : ℝ, (x^2 / 16 + y^2 / 12 = 1)) :=
by
  intros m n hmn hc he hm_eq hn_eq
  sorry

end standard_equation_of_ellipse_l174_17488


namespace coefficient_6th_term_expansion_l174_17492

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n, k => if k > n then 0 else Nat.choose n k

-- Define the coefficient of the general term of binomial expansion
def binomial_coeff (n r : ℕ) : ℤ := (-1)^r * binom n r

-- Define the theorem to show the coefficient of the 6th term in the expansion of (x-1)^10
theorem coefficient_6th_term_expansion :
  binomial_coeff 10 5 = -binom 10 5 :=
by sorry

end coefficient_6th_term_expansion_l174_17492


namespace finite_odd_divisors_condition_l174_17462

theorem finite_odd_divisors_condition (k : ℕ) (hk : 0 < k) :
  (∃ N : ℕ, ∀ n : ℕ, n > N → ¬ (n % 2 = 1 ∧ n ∣ k^n + 1)) ↔ (∃ c : ℕ, k + 1 = 2^c) :=
by sorry

end finite_odd_divisors_condition_l174_17462


namespace binom_18_7_l174_17438

theorem binom_18_7 : Nat.choose 18 7 = 31824 := by sorry

end binom_18_7_l174_17438


namespace sixty_five_percent_of_40_minus_four_fifths_of_25_l174_17487

theorem sixty_five_percent_of_40_minus_four_fifths_of_25 : 
  (0.65 * 40) - (0.8 * 25) = 6 := 
by
  sorry

end sixty_five_percent_of_40_minus_four_fifths_of_25_l174_17487


namespace diagonal_length_of_rectangular_prism_l174_17421

-- Define the dimensions of the rectangular prism
variables (a b c : ℕ) (a_pos : a = 12) (b_pos : b = 15) (c_pos : c = 8)

-- Define the theorem statement
theorem diagonal_length_of_rectangular_prism : 
  ∃ d : ℝ, d = Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2) ∧ d = Real.sqrt 433 := 
by
  -- Note that the proof is intentionally omitted
  sorry

end diagonal_length_of_rectangular_prism_l174_17421


namespace inequality_abc_l174_17453

variable {a b c : ℝ}

theorem inequality_abc (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 8) :
  (a - 2) / (a + 1) + (b - 2) / (b + 1) + (c - 2) / (c + 1) ≤ 0 := by
  sorry

end inequality_abc_l174_17453


namespace simpleInterest_500_l174_17490

def simpleInterest (P R T : ℝ) : ℝ := P * R * T

theorem simpleInterest_500 :
  simpleInterest 10000 0.05 1 = 500 :=
by
  sorry

end simpleInterest_500_l174_17490


namespace periodic_decimal_to_fraction_l174_17423

theorem periodic_decimal_to_fraction : (0.7 + 0.32 : ℝ) == (1013 / 990 : ℝ) := by
  sorry

end periodic_decimal_to_fraction_l174_17423


namespace greatest_divisor_with_sum_of_digits_four_l174_17446

/-- Define the given numbers -/
def a := 4665
def b := 6905

/-- Define the sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Define the greatest number n that divides both a and b, leaving the same remainder and having a sum of digits equal to 4 -/
theorem greatest_divisor_with_sum_of_digits_four :
  ∃ (n : ℕ), (∀ (d : ℕ), (d ∣ a - b ∧ sum_of_digits d = 4) → d ≤ n) ∧ (n ∣ a - b) ∧ (sum_of_digits n = 4) ∧ n = 40 := sorry

end greatest_divisor_with_sum_of_digits_four_l174_17446


namespace international_news_duration_l174_17432

theorem international_news_duration
  (total_duration : ℕ := 30)
  (national_news : ℕ := 12)
  (sports : ℕ := 5)
  (weather_forecasts : ℕ := 2)
  (advertising : ℕ := 6) :
  total_duration - national_news - sports - weather_forecasts - advertising = 5 :=
by
  sorry

end international_news_duration_l174_17432


namespace num_teachers_l174_17413

variable (num_students : ℕ) (ticket_cost : ℕ) (total_cost : ℕ)

theorem num_teachers (h1 : num_students = 20) (h2 : ticket_cost = 5) (h3 : total_cost = 115) :
  (total_cost / ticket_cost - num_students = 3) :=
by
  sorry

end num_teachers_l174_17413


namespace train_speed_in_km_per_hr_l174_17493

variables (L : ℕ) (t : ℕ) (train_speed : ℕ)

-- Conditions
def length_of_train : ℕ := 1050
def length_of_platform : ℕ := 1050
def crossing_time : ℕ := 1

-- Given calculation of speed in meters per minute
def speed_in_m_per_min : ℕ := (length_of_train + length_of_platform) / crossing_time

-- Conversion units
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000
def minutes_to_hours (min : ℕ) : ℕ := min / 60

-- Speed in km/hr
def speed_in_km_per_hr : ℕ := speed_in_m_per_min * (meters_to_kilometers 1000) * (minutes_to_hours 60)

theorem train_speed_in_km_per_hr : speed_in_km_per_hr = 35 :=
by {
  -- We will include the proof steps here, but for now, we just assert with sorry.
  sorry
}

end train_speed_in_km_per_hr_l174_17493


namespace P_cubed_plus_7_is_composite_l174_17451

theorem P_cubed_plus_7_is_composite (P : ℕ) (h_prime_P : Nat.Prime P) (h_prime_P3_plus_5 : Nat.Prime (P^3 + 5)) : ¬ Nat.Prime (P^3 + 7) ∧ (P^3 + 7).factors.length > 1 :=
by
  sorry

end P_cubed_plus_7_is_composite_l174_17451


namespace liquid_X_percentage_36_l174_17463

noncomputable def liquid_X_percentage (m : ℕ) (pX : ℕ) (m_evaporate : ℕ) (m_add : ℕ) (p_add : ℕ) : ℕ :=
  let m_X_initial := (pX * m / 100)
  let m_water_initial := ((100 - pX) * m / 100)
  let m_X_after_evaporation := m_X_initial
  let m_water_after_evaporation := m_water_initial - m_evaporate
  let m_X_additional := (p_add * m_add / 100)
  let m_water_additional := ((100 - p_add) * m_add / 100)
  let m_X_new := m_X_after_evaporation + m_X_additional
  let m_water_new := m_water_after_evaporation + m_water_additional
  let m_total_new := m_X_new + m_water_new
  (m_X_new * 100 / m_total_new)

theorem liquid_X_percentage_36 :
  liquid_X_percentage 10 30 2 2 30 = 36 := by
  sorry

end liquid_X_percentage_36_l174_17463


namespace find_d_l174_17422

open Real

-- Define the given conditions
variable (a b c d e : ℝ)

axiom cond1 : 3 * (a^2 + b^2 + c^2) + 4 = 2 * d + sqrt (a + b + c - d + e)
axiom cond2 : e = 1

-- Define the theorem stating that d = 7/4 under the given conditions
theorem find_d : d = 7/4 := by
  sorry

end find_d_l174_17422


namespace smallest_rel_prime_210_l174_17467

theorem smallest_rel_prime_210 : ∃ x : ℕ, x > 1 ∧ gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 210 = 1 → x ≤ y := 
by
  sorry

end smallest_rel_prime_210_l174_17467


namespace selection_ways_l174_17424

/-- There are a total of 70 ways to select 3 people from 4 teachers and 5 students,
with the condition that there must be at least one teacher and one student among the selected. -/
theorem selection_ways (teachers students : ℕ) (T : 4 = teachers) (S : 5 = students) :
  ∃ (ways : ℕ), ways = 70 := by
  sorry

end selection_ways_l174_17424


namespace total_interest_l174_17450

variable (P R : ℝ)

-- Given condition: Simple interest on sum of money is Rs. 700 after 10 years
def interest_10_years (P R : ℝ) : Prop := (P * R * 10) / 100 = 700

-- Principal is trebled after 5 years
def interest_5_years_treble (P R : ℝ) : Prop := (15 * P * R) / 100 = 105

-- The final interest is the sum of interest for the first 10 years and next 5 years post trebling the principal
theorem total_interest (P R : ℝ) (h1: interest_10_years P R) (h2: interest_5_years_treble P R) : 
  (700 + 105 = 805) := 
  by 
  sorry

end total_interest_l174_17450


namespace area_of_square_with_perimeter_40_l174_17472

theorem area_of_square_with_perimeter_40 (P : ℝ) (s : ℝ) (A : ℝ) 
  (hP : P = 40) (hs : s = P / 4) (hA : A = s^2) : A = 100 :=
by
  sorry

end area_of_square_with_perimeter_40_l174_17472


namespace largest_integer_satisfying_inequality_l174_17499

theorem largest_integer_satisfying_inequality : ∃ (x : ℤ), (5 * x - 4 < 3 - 2 * x) ∧ (∀ (y : ℤ), (5 * y - 4 < 3 - 2 * y) → y ≤ x) ∧ x = 0 :=
by
  sorry

end largest_integer_satisfying_inequality_l174_17499


namespace production_movie_count_l174_17486

theorem production_movie_count
  (LJ_annual : ℕ)
  (H1 : LJ_annual = 220)
  (H2 : ∀ n, n = 275 → n = LJ_annual + (LJ_annual * 25 / 100))
  (years : ℕ)
  (H3 : years = 5) :
  (LJ_annual + 275) * years = 2475 :=
by {
  sorry
}

end production_movie_count_l174_17486


namespace sum_of_x_y_l174_17475

theorem sum_of_x_y (m x y : ℝ) (h₁ : x + m = 4) (h₂ : y - 3 = m) : x + y = 7 :=
sorry

end sum_of_x_y_l174_17475


namespace find_a5_l174_17400

variable {α : Type*} [Field α]

def geometric_seq (a : α) (q : α) (n : ℕ) : α := a * q ^ (n - 1)

theorem find_a5 (a q : α) 
  (h1 : geometric_seq a q 2 = 4)
  (h2 : geometric_seq a q 6 * geometric_seq a q 7 = 16 * geometric_seq a q 9) :
  geometric_seq a q 5 = 32 ∨ geometric_seq a q 5 = -32 :=
by
  -- Proof is omitted as per instructions
  sorry

end find_a5_l174_17400


namespace domain_of_function_y_eq_sqrt_2x_3_div_x_2_l174_17496

def domain (x : ℝ) : Prop :=
  (2 * x - 3 ≥ 0) ∧ (x ≠ 2)

theorem domain_of_function_y_eq_sqrt_2x_3_div_x_2 :
  ∀ x : ℝ, domain x ↔ ((x ≥ 3 / 2) ∧ (x ≠ 2)) :=
by
  sorry

end domain_of_function_y_eq_sqrt_2x_3_div_x_2_l174_17496


namespace pascal_sum_difference_l174_17404

open BigOperators

noncomputable def a_i (i : ℕ) := Nat.choose 3005 i
noncomputable def b_i (i : ℕ) := Nat.choose 3006 i
noncomputable def c_i (i : ℕ) := Nat.choose 3007 i

theorem pascal_sum_difference :
  (∑ i in Finset.range 3007, (b_i i) / (c_i i)) - (∑ i in Finset.range 3006, (a_i i) / (b_i i)) = 1 / 2 := by
  sorry

end pascal_sum_difference_l174_17404


namespace range_of_f_l174_17448

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 2*x)

theorem range_of_f : Set.Ioo 0 3 ∪ {3} = { y | ∃ x, f x = y } :=
by
  sorry

end range_of_f_l174_17448


namespace rectangle_probability_l174_17465

theorem rectangle_probability (m n : ℕ) (h_m : m = 1003^2) (h_n : n = 1003 * 2005) :
  (1 - (m / n)) = 1002 / 2005 :=
by
  sorry

end rectangle_probability_l174_17465


namespace count_solutions_eq_4_l174_17403

theorem count_solutions_eq_4 :
  ∀ x : ℝ, (x^2 - 5)^2 = 16 → x = 3 ∨ x = -3 ∨ x = 1 ∨ x = -1  := sorry

end count_solutions_eq_4_l174_17403


namespace sum_of_roots_l174_17478

theorem sum_of_roots (y1 y2 k m : ℝ) (h1 : y1 ≠ y2) (h2 : 5 * y1^2 - k * y1 = m) (h3 : 5 * y2^2 - k * y2 = m) : 
  y1 + y2 = k / 5 := 
by
  sorry

end sum_of_roots_l174_17478


namespace jori_water_left_l174_17466

theorem jori_water_left (initial_gallons used_gallons : ℚ) (h1 : initial_gallons = 3) (h2 : used_gallons = 11 / 4) :
  initial_gallons - used_gallons = 1 / 4 :=
by
  sorry

end jori_water_left_l174_17466


namespace y_intercept_of_line_l174_17412

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 6 * y = 24) : y = 4 := by
  sorry

end y_intercept_of_line_l174_17412


namespace total_tickets_sold_l174_17433

theorem total_tickets_sold (n : ℕ) 
  (h1 : n * n = 1681) : 
  2 * n = 82 :=
by
  sorry

end total_tickets_sold_l174_17433


namespace least_number_subtracted_l174_17449

theorem least_number_subtracted (n m k : ℕ) (h1 : n = 3830) (h2 : k = 15) (h3 : n % k = m) (h4 : m = 5) : 
  (n - m) % k = 0 :=
by
  sorry

end least_number_subtracted_l174_17449


namespace minimum_value_g_l174_17474

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  if a > 1 then 
    a * (-1/a) + 1 
  else 
    if 0 < a then 
      a^2 + 1 
    else 
      0  -- adding a default value to make it computable

theorem minimum_value_g (a : ℝ) (m : ℝ) : 0 < a ∧ a < 2 ∧ ∃ x₀, f x₀ a = m → m ≥ 5 / 2 :=
by
  sorry

end minimum_value_g_l174_17474


namespace intersection_with_y_axis_is_correct_l174_17456

theorem intersection_with_y_axis_is_correct (x y : ℝ) (h : y = 5 * x + 1) (hx : x = 0) : y = 1 :=
by
  sorry

end intersection_with_y_axis_is_correct_l174_17456


namespace reduced_price_tickets_first_week_l174_17406

theorem reduced_price_tickets_first_week (total_tickets sold_at_full_price : ℕ) 
  (condition1 : total_tickets = 25200) 
  (condition2 : sold_at_full_price = 16500)
  (condition3 : ∃ R, total_tickets = R + 5 * R) : 
  ∃ R : ℕ, R = 3300 := 
by sorry

end reduced_price_tickets_first_week_l174_17406


namespace find_b_of_sin_l174_17482

theorem find_b_of_sin (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
                       (h_period : (2 * Real.pi) / b = Real.pi / 2) : b = 4 := by
  sorry

end find_b_of_sin_l174_17482


namespace evening_customers_l174_17439

-- Define the conditions
def matinee_price : ℕ := 5
def evening_price : ℕ := 7
def opening_night_price : ℕ := 10
def popcorn_price : ℕ := 10
def num_matinee_customers : ℕ := 32
def num_opening_night_customers : ℕ := 58
def total_revenue : ℕ := 1670

-- Define the number of evening customers as a variable
variable (E : ℕ)

-- Prove that the number of evening customers E equals 40 given the conditions
theorem evening_customers :
  5 * num_matinee_customers +
  7 * E +
  10 * num_opening_night_customers +
  10 * (num_matinee_customers + E + num_opening_night_customers) / 2 = total_revenue
  → E = 40 :=
by
  intro h
  sorry

end evening_customers_l174_17439


namespace sacks_harvested_per_section_l174_17485

theorem sacks_harvested_per_section (total_sacks : ℕ) (sections : ℕ) (sacks_per_section : ℕ) 
  (h1 : total_sacks = 360) 
  (h2 : sections = 8) 
  (h3 : total_sacks = sections * sacks_per_section) :
  sacks_per_section = 45 :=
by sorry

end sacks_harvested_per_section_l174_17485


namespace x_minus_q_eq_three_l174_17498

theorem x_minus_q_eq_three (x q : ℝ) (h1 : |x - 3| = q) (h2 : x > 3) : x - q = 3 :=
by 
  sorry

end x_minus_q_eq_three_l174_17498


namespace simplify_fraction_l174_17414

theorem simplify_fraction :
  ((1 / 4) + (1 / 6)) / ((3 / 8) - (1 / 3)) = 10 := by
  sorry

end simplify_fraction_l174_17414


namespace max_grain_mass_l174_17483

def platform_length : ℝ := 10
def platform_width : ℝ := 5
def grain_density : ℝ := 1200
def angle_of_repose : ℝ := 45
def max_mass : ℝ := 175000

theorem max_grain_mass :
  let height_of_pile := platform_width / 2
  let volume_of_prism := platform_length * platform_width * height_of_pile
  let volume_of_pyramid := (1 / 3) * (platform_width * height_of_pile) * height_of_pile
  let total_volume := volume_of_prism + 2 * volume_of_pyramid
  let calculated_mass := total_volume * grain_density
  calculated_mass = max_mass :=
by {
  sorry
}

end max_grain_mass_l174_17483


namespace combined_mpg_l174_17445

theorem combined_mpg (ray_mpg tom_mpg ray_miles tom_miles : ℕ) 
  (h1 : ray_mpg = 50) (h2 : tom_mpg = 8) 
  (h3 : ray_miles = 100) (h4 : tom_miles = 200) : 
  (ray_miles + tom_miles) / ((ray_miles / ray_mpg) + (tom_miles / tom_mpg)) = 100 / 9 :=
by
  sorry

end combined_mpg_l174_17445


namespace contradiction_method_conditions_l174_17410

theorem contradiction_method_conditions :
  (using_judgments_contrary_to_conclusion ∧ using_conditions_of_original_proposition ∧ using_axioms_theorems_definitions) =
  (needed_conditions_method_of_contradiction) :=
sorry

end contradiction_method_conditions_l174_17410


namespace find_A_l174_17491

noncomputable def A_value (A B C : ℝ) := (A = 1/4) 

theorem find_A : 
  ∀ (A B C : ℝ),
  (∀ x : ℝ, x ≠ 1 → x ≠ 3 → (1 / (x^3 - 3*x^2 - 13*x + 15) = A / (x - 1) + B / (x - 3) + C / (x - 3)^2)) →
  A_value A B C :=
by 
  sorry

end find_A_l174_17491


namespace measure_45_minutes_l174_17419

-- Definitions of the conditions
structure Conditions where
  lighter : Prop
  strings : ℕ
  burn_time : ℕ → ℕ
  non_uniform_burn : Prop

-- We can now state the problem in Lean
theorem measure_45_minutes (c : Conditions) (h1 : c.lighter) (h2 : c.strings = 2)
  (h3 : ∀ s, s < 2 → c.burn_time s = 60) (h4 : c.non_uniform_burn) :
  ∃ t, t = 45 := 
sorry

end measure_45_minutes_l174_17419


namespace M_subset_N_l174_17436

def M : Set ℝ := {x | ∃ k : ℤ, x = (k / 2) * 180 + 45}
def N : Set ℝ := {x | ∃ k : ℤ, x = (k / 4) * 180 + 45}

theorem M_subset_N : M ⊆ N :=
sorry

end M_subset_N_l174_17436


namespace bart_total_pages_l174_17444

theorem bart_total_pages (total_spent : ℝ) (cost_per_notepad : ℝ) (pages_per_notepad : ℕ)
  (h1 : total_spent = 10) (h2 : cost_per_notepad = 1.25) (h3 : pages_per_notepad = 60) :
  total_spent / cost_per_notepad * pages_per_notepad = 480 :=
by
  sorry

end bart_total_pages_l174_17444


namespace gcd_of_72_120_168_l174_17441

theorem gcd_of_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  sorry

end gcd_of_72_120_168_l174_17441


namespace neg_p_sufficient_for_neg_q_l174_17430

def p (x : ℝ) : Prop := |2 * x - 3| > 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem neg_p_sufficient_for_neg_q :
  (∀ x, ¬ p x → ¬ q x) ∧ ¬ (∀ x, ¬ q x → ¬ p x) :=
by
  -- Placeholder to indicate skipping the proof
  sorry

end neg_p_sufficient_for_neg_q_l174_17430


namespace max_value_of_quadratic_on_interval_l174_17401

theorem max_value_of_quadratic_on_interval : 
  ∃ (x : ℝ), -2 ≤ x ∧ x ≤ 2 ∧ (∀ y, (∃ x, -2 ≤ x ∧ x ≤ 2 ∧ y = (x + 1)^2 - 4) → y ≤ 5) :=
sorry

end max_value_of_quadratic_on_interval_l174_17401


namespace triangle_at_most_one_obtuse_l174_17452

theorem triangle_at_most_one_obtuse (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 90 → B + C < 90) (h3 : B > 90 → A + C < 90) (h4 : C > 90 → A + B < 90) :
  ¬ (A > 90 ∧ B > 90 ∨ B > 90 ∧ C > 90 ∨ A > 90 ∧ C > 90) :=
by
  sorry

end triangle_at_most_one_obtuse_l174_17452


namespace find_sister_candy_l174_17437

/-- Define Katie's initial amount of candy -/
def Katie_candy : ℕ := 10

/-- Define the amount of candy eaten the first night -/
def eaten_candy : ℕ := 9

/-- Define the amount of candy left after the first night -/
def remaining_candy : ℕ := 7

/-- Define the number of candies Katie's sister had -/
def sister_candy (S : ℕ) : Prop :=
  Katie_candy + S - eaten_candy = remaining_candy

/-- Theorem stating that Katie's sister had 6 pieces of candy -/
theorem find_sister_candy : ∃ S, sister_candy S ∧ S = 6 :=
by
  sorry

end find_sister_candy_l174_17437


namespace largest_possible_length_d_l174_17418

theorem largest_possible_length_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 2) 
  (h2 : a ≤ b)
  (h3 : b ≤ c)
  (h4 : c ≤ d) 
  (h5 : d < a + b + c) : 
  d < 1 :=
sorry

end largest_possible_length_d_l174_17418


namespace gavin_shirts_l174_17471

theorem gavin_shirts (t g b : ℕ) (h_total : t = 23) (h_green : g = 17) (h_blue : b = t - g) : b = 6 :=
by sorry

end gavin_shirts_l174_17471


namespace students_wanted_fruit_l174_17402

theorem students_wanted_fruit (red_apples green_apples extra_fruit : ℕ)
  (h_red : red_apples = 42)
  (h_green : green_apples = 7)
  (h_extra : extra_fruit = 40) :
  red_apples + green_apples + extra_fruit - (red_apples + green_apples) = 40 :=
by
  sorry

end students_wanted_fruit_l174_17402


namespace gondor_laptops_wednesday_l174_17443

/-- Gondor's phone repair earnings per unit -/
def phone_earning : ℕ := 10

/-- Gondor's laptop repair earnings per unit -/
def laptop_earning : ℕ := 20

/-- Number of phones repaired on Monday -/
def phones_monday : ℕ := 3

/-- Number of phones repaired on Tuesday -/
def phones_tuesday : ℕ := 5

/-- Number of laptops repaired on Thursday -/
def laptops_thursday : ℕ := 4

/-- Total earnings of Gondor -/
def total_earnings : ℕ := 200

/-- Number of laptops repaired on Wednesday, which we need to prove equals 2 -/
def laptops_wednesday : ℕ := 2

theorem gondor_laptops_wednesday : 
    (phones_monday * phone_earning + phones_tuesday * phone_earning + 
    laptops_thursday * laptop_earning + laptops_wednesday * laptop_earning = total_earnings) :=
by
    sorry

end gondor_laptops_wednesday_l174_17443


namespace cost_of_old_car_l174_17429

theorem cost_of_old_car (C_old C_new : ℝ): 
  C_new = 2 * C_old → 
  1800 + 2000 = C_new → 
  C_old = 1900 :=
by
  intros H1 H2
  sorry

end cost_of_old_car_l174_17429


namespace rainfall_difference_correct_l174_17425

def rainfall_difference (monday_rain : ℝ) (tuesday_rain : ℝ) : ℝ :=
  monday_rain - tuesday_rain

theorem rainfall_difference_correct : rainfall_difference 0.9 0.2 = 0.7 :=
by
  simp [rainfall_difference]
  sorry

end rainfall_difference_correct_l174_17425


namespace simplify_expression_correct_l174_17407

variable {R : Type} [CommRing R]

def simplify_expression (x : R) : R :=
  2 * x^2 * (4 * x^3 - 3 * x + 1) - 7 * (x^3 - 3 * x^2 + 2 * x - 8)

theorem simplify_expression_correct (x : R) : 
  simplify_expression x = 8 * x^5 + 0 * x^4 - 13 * x^3 + 23 * x^2 - 14 * x + 56 :=
by
  sorry

end simplify_expression_correct_l174_17407


namespace find_ab_l174_17409

theorem find_ab (a b : ℝ) (h1 : a - b = 26) (h2 : a + b = 15) :
  a = 41 / 2 ∧ b = 11 / 2 :=
sorry

end find_ab_l174_17409


namespace benny_cards_left_l174_17427

theorem benny_cards_left (n : ℕ) : ℕ :=
  (n + 4) / 2

end benny_cards_left_l174_17427


namespace gasVolume_at_20_l174_17458

variable (V : ℕ → ℕ)

/-- Given conditions:
 1. The gas volume expands by 3 cubic centimeters for every 5 degree rise in temperature.
 2. The volume is 30 cubic centimeters when the temperature is 30 degrees.
  -/
def gasVolume : Prop :=
  (∀ T ΔT, ΔT = 5 → V (T + ΔT) = V T + 3) ∧ V 30 = 30

theorem gasVolume_at_20 :
  gasVolume V → V 20 = 24 :=
by
  intro h
  -- Proof steps would go here.
  sorry

end gasVolume_at_20_l174_17458


namespace one_third_percent_of_150_l174_17434

theorem one_third_percent_of_150 : (1/3) * (150 / 100) = 0.5 := by
  sorry

end one_third_percent_of_150_l174_17434


namespace simplify_expression_l174_17497

variable (a : ℝ)

theorem simplify_expression : 2 * a * (2 * a ^ 2 + a) - a ^ 2 = 4 * a ^ 3 + a ^ 2 := 
  sorry

end simplify_expression_l174_17497


namespace problem1_problem2_l174_17408

-- (Problem 1)
def A : Set ℝ := {x | x^2 + 2 * x < 0}
def B : Set ℝ := {x | x ≥ -1}
def complement_A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 0}
def intersection_complement_A_B : Set ℝ := {x | x ≥ 0}

theorem problem1 : (complement_A ∩ B) = intersection_complement_A_B :=
by
  sorry

-- (Problem 2)
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 1}

theorem problem2 {a : ℝ} : (C a ⊆ A) ↔ (a ≤ -1 / 2) :=
by
  sorry

end problem1_problem2_l174_17408


namespace bike_route_length_l174_17479

theorem bike_route_length (u1 u2 u3 l1 l2 : ℕ) (h1 : u1 = 4) (h2 : u2 = 7) (h3 : u3 = 2) (h4 : l1 = 6) (h5 : l2 = 7) :
  u1 + u2 + u3 + u1 + u2 + u3 + l1 + l2 + l1 + l2 = 52 := 
by
  sorry

end bike_route_length_l174_17479


namespace smallest_not_prime_nor_square_no_prime_factor_lt_60_correct_l174_17481

noncomputable def smallest_not_prime_nor_square_no_prime_factor_lt_60 : ℕ :=
  4087

theorem smallest_not_prime_nor_square_no_prime_factor_lt_60_correct :
  ∀ n : ℕ, 
    (n > 0) → 
    (¬ Prime n) →
    (¬ ∃ k : ℕ, k * k = n) →
    (∀ p : ℕ, Prime p → p ∣ n → p ≥ 60) →
    n ≥ 4087 :=
sorry

end smallest_not_prime_nor_square_no_prime_factor_lt_60_correct_l174_17481


namespace orchard_yield_correct_l174_17454

-- Definitions for conditions
def gala3YrTreesYield : ℕ := 10 * 120
def gala2YrTreesYield : ℕ := 10 * 150
def galaTotalYield : ℕ := gala3YrTreesYield + gala2YrTreesYield

def fuji4YrTreesYield : ℕ := 5 * 180
def fuji5YrTreesYield : ℕ := 5 * 200
def fujiTotalYield : ℕ := fuji4YrTreesYield + fuji5YrTreesYield

def redhaven6YrTreesYield : ℕ := 15 * 50
def redhaven4YrTreesYield : ℕ := 15 * 60
def redhavenTotalYield : ℕ := redhaven6YrTreesYield + redhaven4YrTreesYield

def elberta2YrTreesYield : ℕ := 5 * 70
def elberta3YrTreesYield : ℕ := 5 * 75
def elberta5YrTreesYield : ℕ := 5 * 80
def elbertaTotalYield : ℕ := elberta2YrTreesYield + elberta3YrTreesYield + elberta5YrTreesYield

def appleTotalYield : ℕ := galaTotalYield + fujiTotalYield
def peachTotalYield : ℕ := redhavenTotalYield + elbertaTotalYield
def orchardTotalYield : ℕ := appleTotalYield + peachTotalYield

-- Theorem to prove
theorem orchard_yield_correct : orchardTotalYield = 7375 := 
by sorry

end orchard_yield_correct_l174_17454


namespace percent_sparrows_not_pigeons_l174_17459

-- Definitions of percentages
def crows_percent : ℝ := 0.20
def sparrows_percent : ℝ := 0.40
def pigeons_percent : ℝ := 0.15
def doves_percent : ℝ := 0.25

-- The statement to prove
theorem percent_sparrows_not_pigeons :
  (sparrows_percent / (1 - pigeons_percent)) = 0.47 :=
by
  sorry

end percent_sparrows_not_pigeons_l174_17459


namespace intersection_S_T_eq_T_l174_17435

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l174_17435


namespace cos_seven_pi_over_six_l174_17447

theorem cos_seven_pi_over_six :
  Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 :=
sorry

end cos_seven_pi_over_six_l174_17447


namespace books_about_outer_space_l174_17464

variable (x : ℕ)

theorem books_about_outer_space :
  160 + 48 + 16 * x = 224 → x = 1 :=
by
  intro h
  sorry

end books_about_outer_space_l174_17464


namespace find_extra_factor_l174_17460

theorem find_extra_factor (w : ℕ) (h1 : w > 0) (h2 : w = 156) (h3 : ∃ (k : ℕ), (2^5 * 13^2) ∣ (936 * w))
  : 3 ∣ w := sorry

end find_extra_factor_l174_17460


namespace part1_part2_l174_17415

theorem part1 (m n : ℕ) (h1 : m > n) (h2 : Nat.gcd m n + Nat.lcm m n = m + n) : n ∣ m := 
sorry

theorem part2 (m n : ℕ) (h1 : m > n) (h2 : Nat.gcd m n + Nat.lcm m n = m + n)
(h3 : m - n = 10) : (m, n) = (11, 1) ∨ (m, n) = (12, 2) ∨ (m, n) = (15, 5) ∨ (m, n) = (20, 10) := 
sorry

end part1_part2_l174_17415


namespace email_count_first_day_l174_17417

theorem email_count_first_day (E : ℕ) 
  (h1 : ∃ E, E + E / 2 + E / 4 + E / 8 = 30) : E = 16 :=
by
  sorry

end email_count_first_day_l174_17417


namespace prove_interest_rates_equal_l174_17476

noncomputable def interest_rates_equal : Prop :=
  let initial_savings := 1000
  let savings_simple := initial_savings / 2
  let savings_compound := initial_savings / 2
  let simple_interest_earned := 100
  let compound_interest_earned := 105
  let time := 2
  let r_s := simple_interest_earned / (savings_simple * time)
  let r_c := (compound_interest_earned / savings_compound + 1) ^ (1 / time) - 1
  r_s = r_c

theorem prove_interest_rates_equal : interest_rates_equal :=
  sorry

end prove_interest_rates_equal_l174_17476


namespace find_number_l174_17473

theorem find_number (x : ℝ) (h : 160 = 3.2 * x) : x = 50 :=
by 
  sorry

end find_number_l174_17473


namespace order_of_exponents_l174_17480

theorem order_of_exponents (p q r : ℕ) (hp : p = 2^3009) (hq : q = 3^2006) (hr : r = 5^1003) : r < p ∧ p < q :=
by {
  sorry -- Proof will go here
}

end order_of_exponents_l174_17480


namespace total_tomatoes_l174_17455

def tomatoes_first_plant : Nat := 2 * 12
def tomatoes_second_plant : Nat := (tomatoes_first_plant / 2) + 5
def tomatoes_third_plant : Nat := tomatoes_second_plant + 2

theorem total_tomatoes :
  (tomatoes_first_plant + tomatoes_second_plant + tomatoes_third_plant) = 60 := by
  sorry

end total_tomatoes_l174_17455


namespace number_of_blue_marbles_l174_17461

-- Definitions based on the conditions
def total_marbles : ℕ := 20
def red_marbles : ℕ := 9
def probability_red_or_white : ℚ := 0.7

-- The question to prove: the number of blue marbles (B)
theorem number_of_blue_marbles (B W : ℕ) (h1 : B + W + red_marbles = total_marbles)
  (h2: (red_marbles + W : ℚ) / total_marbles = probability_red_or_white) : 
  B = 6 := 
by
  sorry

end number_of_blue_marbles_l174_17461


namespace analytical_expression_smallest_positive_period_min_value_max_value_l174_17431

noncomputable def P (x : ℝ) : ℝ × ℝ :=
  (Real.cos (2 * x) + 1, 1)

noncomputable def Q (x : ℝ) : ℝ × ℝ :=
  (1, Real.sqrt 3 * Real.sin (2 * x) + 1)

noncomputable def f (x : ℝ) : ℝ :=
  (P x).1 * (Q x).1 + (P x).2 * (Q x).2

theorem analytical_expression (x : ℝ) : 
  f x = 2 * Real.sin (2 * x + Real.pi / 6) + 2 :=
sorry

theorem smallest_positive_period : 
  ∀ x : ℝ, f (x + Real.pi) = f x :=
sorry

theorem min_value : 
  ∃ x : ℝ, f x = 0 :=
sorry

theorem max_value : 
  ∃ y : ℝ, f y = 4 :=
sorry

end analytical_expression_smallest_positive_period_min_value_max_value_l174_17431


namespace tangent_line_at_P_l174_17495

noncomputable def tangent_line (x : ℝ) (y : ℝ) := (8 * x - y - 12 = 0)

def curve (x : ℝ) := x^3 - x^2

def derivative (f : ℝ → ℝ) (x : ℝ) := 3 * x^2 - 2 * x

theorem tangent_line_at_P :
    tangent_line 2 4 :=
by
  sorry

end tangent_line_at_P_l174_17495


namespace weight_of_replaced_person_l174_17428

theorem weight_of_replaced_person 
  (avg_increase : ℝ)
  (num_persons : ℕ)
  (new_person_weight : ℝ)
  (weight_increase : ℝ)
  (new_person_might_be_90_kg : new_person_weight = 90)
  (average_increase_by_3_5_kg : avg_increase = 3.5)
  (group_of_8_persons : num_persons = 8)
  (total_weight_increase_formula : weight_increase = num_persons * avg_increase)
  (weight_of_replaced_person : ℝ)
  (weight_difference_formula : weight_of_replaced_person = new_person_weight - weight_increase) :
  weight_of_replaced_person = 62 :=
sorry

end weight_of_replaced_person_l174_17428


namespace range_of_m_l174_17405

def f (m x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

def f_derivative_nonnegative_on_interval (m : ℝ) : Prop :=
  ∀ x : ℝ, 1 < x → 6 * x^2 - 6 * m * x + 6 ≥ 0

theorem range_of_m (m : ℝ) : f_derivative_nonnegative_on_interval m ↔ m ≤ 2 :=
by
  sorry

end range_of_m_l174_17405


namespace mark_speed_l174_17420

theorem mark_speed
  (chris_speed : ℕ)
  (distance_to_school : ℕ)
  (mark_total_distance : ℕ)
  (mark_time_longer : ℕ)
  (chris_speed_eq : chris_speed = 3)
  (distance_to_school_eq : distance_to_school = 9)
  (mark_total_distance_eq : mark_total_distance = 15)
  (mark_time_longer_eq : mark_time_longer = 2) :
  mark_total_distance / (distance_to_school / chris_speed + mark_time_longer) = 3 := 
by
  sorry 

end mark_speed_l174_17420


namespace fraction_of_ponies_with_horseshoes_l174_17442

theorem fraction_of_ponies_with_horseshoes 
  (P H : ℕ) 
  (h1 : H = P + 4) 
  (h2 : H + P ≥ 164) 
  (x : ℚ)
  (h3 : ∃ (n : ℕ), n = (5 / 8) * (x * P)) :
  x = 1 / 10 := by
  sorry

end fraction_of_ponies_with_horseshoes_l174_17442


namespace find_intercept_l174_17457

theorem find_intercept (avg_height : ℝ) (avg_shoe_size : ℝ) (a : ℝ)
  (h1 : avg_height = 170)
  (h2 : avg_shoe_size = 40) 
  (h3 : 3 * avg_shoe_size + a = avg_height) : a = 50 := 
by
  sorry

end find_intercept_l174_17457


namespace raja_monthly_income_l174_17469

theorem raja_monthly_income (X : ℝ) 
  (h1 : 0.1 * X = 5000) : X = 50000 :=
sorry

end raja_monthly_income_l174_17469


namespace find_n_l174_17484

theorem find_n (n : ℤ) (h : (1 : ℤ)^2 + 3 * 1 + n = 0) : n = -4 :=
sorry

end find_n_l174_17484


namespace rational_inequality_solution_l174_17477

theorem rational_inequality_solution (x : ℝ) (h : x ≠ 4) :
  (4 < x ∧ x ≤ 5) ↔ (x - 2) / (x - 4) ≤ 3 :=
sorry

end rational_inequality_solution_l174_17477


namespace jasmine_percentage_is_approx_l174_17494

noncomputable def initial_solution_volume : ℝ := 80
noncomputable def initial_jasmine_percent : ℝ := 0.10
noncomputable def initial_lemon_percent : ℝ := 0.05
noncomputable def initial_orange_percent : ℝ := 0.03
noncomputable def added_jasmine_volume : ℝ := 8
noncomputable def added_water_volume : ℝ := 12
noncomputable def added_lemon_volume : ℝ := 6
noncomputable def added_orange_volume : ℝ := 7

noncomputable def initial_jasmine_volume := initial_solution_volume * initial_jasmine_percent
noncomputable def initial_lemon_volume := initial_solution_volume * initial_lemon_percent
noncomputable def initial_orange_volume := initial_solution_volume * initial_orange_percent
noncomputable def initial_water_volume := initial_solution_volume - (initial_jasmine_volume + initial_lemon_volume + initial_orange_volume)

noncomputable def new_jasmine_volume := initial_jasmine_volume + added_jasmine_volume
noncomputable def new_water_volume := initial_water_volume + added_water_volume
noncomputable def new_lemon_volume := initial_lemon_volume + added_lemon_volume
noncomputable def new_orange_volume := initial_orange_volume + added_orange_volume
noncomputable def new_total_volume := new_jasmine_volume + new_water_volume + new_lemon_volume + new_orange_volume

noncomputable def new_jasmine_percent := (new_jasmine_volume / new_total_volume) * 100

theorem jasmine_percentage_is_approx :
  abs (new_jasmine_percent - 14.16) < 0.01 := sorry

end jasmine_percentage_is_approx_l174_17494


namespace total_fish_caught_l174_17411

-- Definitions based on conditions
def brenden_morning_fish := 8
def brenden_fish_thrown_back := 3
def brenden_afternoon_fish := 5
def dad_fish := 13

-- Theorem representing the main question and its answer
theorem total_fish_caught : 
  (brenden_morning_fish + brenden_afternoon_fish - brenden_fish_thrown_back) + dad_fish = 23 :=
by
  sorry -- Proof goes here

end total_fish_caught_l174_17411


namespace product_lcm_gcd_l174_17470

def a : ℕ := 6
def b : ℕ := 8

theorem product_lcm_gcd : Nat.lcm a b * Nat.gcd a b = 48 := by
  sorry

end product_lcm_gcd_l174_17470


namespace first_group_men_l174_17416

theorem first_group_men (x : ℕ) (days1 days2 : ℝ) (men2 : ℕ) (h1 : days1 = 25) (h2 : days2 = 17.5) (h3 : men2 = 20) (h4 : x * days1 = men2 * days2) : x = 14 := 
by
  sorry

end first_group_men_l174_17416


namespace distance_is_18_l174_17440

noncomputable def distance_walked (x t d : ℝ) : Prop :=
  let faster := (x + 1) * (3 * t / 4) = d
  let slower := (x - 1) * (t + 3) = d
  let normal := x * t = d
  faster ∧ slower ∧ normal

theorem distance_is_18 : 
  ∃ (x t : ℝ), distance_walked x t 18 :=
by
  sorry

end distance_is_18_l174_17440
