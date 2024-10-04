import Mathlib

namespace intercepts_sum_eq_seven_l144_144800

theorem intercepts_sum_eq_seven :
    (∃ a b c, (∀ y, (3 * y^2 - 9 * y + 4 = a) → y = 0) ∧ 
              (∀ y, (3 * y^2 - 9 * y + 4 = 0) → (y = b ∨ y = c)) ∧ 
              (a + b + c = 7)) := 
sorry

end intercepts_sum_eq_seven_l144_144800


namespace limit_of_sequence_l144_144196

open Real

noncomputable def sequence (n : ℕ) : ℝ :=
  (2 * n - real.sin n) / (sqrt n - (n ^ 3 - 7) ^ (1 / 3))

theorem limit_of_sequence : 
  tendsto (λ n : ℕ, (2 * (n : ℝ) - sin (n : ℝ)) / (sqrt n - cbrt (↑n^3 - 7))) atTop (𝓝 (-2)) :=
by
  sorry

end limit_of_sequence_l144_144196


namespace rooks_impossible_to_reposition_l144_144139

-- Variables representing the initial positions of the rooks
variables (A B C : Type) [field A] [field B] [field C]

-- Main theorem: Proving that it is impossible to move the rooks while maintaining the protection condition
theorem rooks_impossible_to_reposition (initial_positions : (A × B × C))
  (protected_condition : ∀ (P Q R : A × B × C), (P.1 = A ∧ Q.1 = B ∧ R.1 = C) → 
    (∃ S : A × B × C, S.1 = A ∧ S.2 = B ∧ S.3 = C)) :
  ¬ ∃ moves : (A × B × C) → (A × B × C), 
    (∀ positions, protected_condition (moves positions, initial_positions)) ∧ 
    (moves initial_positions = (C, B, A)) :=
begin
  -- Proof content will go here
  sorry
end

end rooks_impossible_to_reposition_l144_144139


namespace least_blue_eyes_and_backpack_l144_144618

theorem least_blue_eyes_and_backpack (total_students blue_eyes backpack : ℕ)
  (total_eq : total_students = 25)
  (blue_eyes_eq : blue_eyes = 15)
  (backpack_eq : backpack = 18) :
  ∃ (blue_eyes_and_backpack : ℕ), blue_eyes_and_backpack = 7 :=
by {
  have hw := total_eq,
  have hb := blue_eyes_eq,
  have hbp := backpack_eq,
  sorry
}

end least_blue_eyes_and_backpack_l144_144618


namespace nth_term_formula_sixth_term_l144_144915

def sequence (n : ℕ) : ℝ := (Real.sqrt (((n + 1) ^ 2) - 1)) / n

theorem nth_term_formula (n : ℕ) (h : n > 0) : sequence n = (Real.sqrt (((n + 1) ^ 2) - 1)) / n :=
  by sorry

theorem sixth_term : sequence 6 = (2 * Real.sqrt 3) / 3 :=
  by sorry

end nth_term_formula_sixth_term_l144_144915


namespace order_of_numbers_l144_144982

open Real

def a := sqrt 3
def b := (1 / 2) ^ 3
def c := log (1 / 2) 3

theorem order_of_numbers : a > b ∧ b > c :=
by
  have h1 : sqrt 3 > 1 := Real.sqrt_lt (by norm_num: 3 > 1) (by norm_num: 1 > 0)
  have h2 : (1 / 2) ^ 3 = 1 / 8 := by norm_num
  have h3 : 0 < 1 / 8 := by norm_num
  have h4 : (1 / 2) ^ 3 < 1 := by
    rw h2
    norm_num
  have h5 : log (1 / 2) 3 < 0 :=
    by apply Real.log_of_base_lt1; norm_num; norm_num
  exact ⟨h1, h4, h5⟩

end order_of_numbers_l144_144982


namespace time_to_complete_job_is_2_hours_l144_144553

-- Define the conditions
def type_R_work_rate := 1 / 36
def type_S_work_rate := 1 / 9
def num_machines_R := 3.6
def num_machines_S := 3.6

-- Define the combined rate of work and the time to complete the job
def combined_work_rate (n_R n_S : ℝ) := n_R * type_R_work_rate + n_S * type_S_work_rate
def time_to_complete_job (n_R n_S : ℝ) := 1 / (combined_work_rate n_R n_S)

-- The main theorem to be proved
theorem time_to_complete_job_is_2_hours : time_to_complete_job num_machines_R num_machines_S = 2 := by
  sorry

end time_to_complete_job_is_2_hours_l144_144553


namespace original_perimeter_not_necessarily_multiple_of_four_l144_144566

/-
Define the conditions given in the problem:
1. A rectangle is divided into several smaller rectangles.
2. The perimeter of each of these smaller rectangles is a multiple of 4.
-/
structure Rectangle where
  length : ℕ
  width : ℕ

def perimeter (r : Rectangle) : ℕ :=
  2 * (r.length + r.width)

def is_multiple_of_four (n : ℕ) : Prop :=
  n % 4 = 0

def smaller_rectangles (rs : List Rectangle) : Prop :=
  ∀ r ∈ rs, is_multiple_of_four (perimeter r)

-- Define the main statement to be proved
theorem original_perimeter_not_necessarily_multiple_of_four (original : Rectangle) (rs : List Rectangle)
  (h1 : smaller_rectangles rs) (h2 : ∀ r ∈ rs, r.length * r.width = original.length * original.width) :
  ¬ is_multiple_of_four (perimeter original) :=
by
  sorry

end original_perimeter_not_necessarily_multiple_of_four_l144_144566


namespace expected_value_of_winnings_is_0_18_l144_144013
noncomputable def expected_value_winnings : ℝ :=
  let prime_numbers := {2, 3, 5, 7}
  let composite_numbers := {4, 6, 8, 9, 10}
  let unitary_number := 1
  let prob_prime := 0.4
  let prob_composite := 0.5
  let prob_unitary := 0.1
  let winnings_prime := (2 + 3 + 5 + 7 : ℝ) / 10
  let loss_unitary := -5 * 0.1
  prob_prime * winnings_prime + prob_composite * 0 + loss_unitary

theorem expected_value_of_winnings_is_0_18 :
  expected_value_winnings = 0.18 :=
by
  sorry

end expected_value_of_winnings_is_0_18_l144_144013


namespace number_of_valid_pairings_l144_144821

-- Define people and the knowledge relationship around the circle
def person := Fin 12
def knows (p q : person) : Prop := 
  q = (p + 1) % 12 ∨ q = (p + 11) % 12 ∨ q = (p + 3) % 12

-- Define what it means to be a valid pairing
def is_valid_pairing (pairs : Finset (person × person)) : Prop :=
  pairs.card = 6 ∧
  -- Every person appears exactly once
  ∀ p : person, ∃! q : person, (p, q) ∈ pairs ∨ (q, p) ∈ pairs ∧ knows p q

-- The theorem to prove
theorem number_of_valid_pairings : 
  ∃ (pairs : Finset (Finset (person × person))), (pairs.card = 14 ∧ ∀ p ∈ pairs, is_valid_pairing p) := 
sorry

end number_of_valid_pairings_l144_144821


namespace sahil_selling_price_l144_144027

def purchase_price : ℕ := 12000
def repair_costs : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_margin : ℚ := 0.50

def total_cost : ℕ := purchase_price + repair_costs + transportation_charges

def profit : ℚ := profit_margin * total_cost

def selling_price : ℕ := total_cost + profit.to_nat -- We need to convert profit to a natural number since it's in ℚ

theorem sahil_selling_price : selling_price = 27000 := by
  sorry

end sahil_selling_price_l144_144027


namespace largest_divisor_360_450_l144_144492

theorem largest_divisor_360_450 : ∃ d, (d ∣ 360 ∧ d ∣ 450) ∧ (∀ e, (e ∣ 360 ∧ e ∣ 450) → e ≤ d) ∧ d = 90 :=
by
  sorry

end largest_divisor_360_450_l144_144492


namespace constant_term_in_expansion_l144_144058

theorem constant_term_in_expansion (n : ℕ) (h1 : (n = 10)) : 
  let T_r := λ (r : ℕ), (nat.choose 10 r) * ((2 : ℝ)^r) * (x : ℝ)^(5 - (5 : ℝ)*r / 2)
  in T_r 2 = 180 :=
by 
  sorry

end constant_term_in_expansion_l144_144058


namespace largest_mult_seven_below_minus_95_l144_144112

theorem largest_mult_seven_below_minus_95 : ∃ n : ℤ, 7 * n = -98 ∧ 7 * n < -95 := 
by {
  let n := -14,
  use n,
  split,
  { norm_num, },  -- Proving that 7 * (-14) = -98 
  { norm_num, }   -- Proving that -98 < -95
}

end largest_mult_seven_below_minus_95_l144_144112


namespace dunkers_lineup_count_l144_144781

theorem dunkers_lineup_count (players : Finset ℕ) (h_players : players.card = 15) (alice zen : ℕ) 
  (h_alice : alice ∈ players) (h_zen : zen ∈ players) (h_distinct : alice ≠ zen) :
  (∃ (S : Finset (Finset ℕ)), S.card = 2717 ∧ ∀ s ∈ S, s.card = 5 ∧ ¬ (alice ∈ s ∧ zen ∈ s)) :=
by
  sorry

end dunkers_lineup_count_l144_144781


namespace other_x_intercept_l144_144950

noncomputable def ellipse_x_intercepts (f1 f2 : ℝ × ℝ) (x_intercept1 : ℝ × ℝ) : ℝ × ℝ :=
  let d := dist f1 x_intercept1 + dist f2 x_intercept1
  let x := (d^2 - 2 * d * sqrt (3^2 + (d / 2 - 4)^2)) / (2 * d - 8)
  (x, 0)

theorem other_x_intercept :
  ellipse_x_intercepts (0, 3) (4, 0) (0, 0) = (56 / 11, 0) :=
by
  sorry

end other_x_intercept_l144_144950


namespace maximum_value_of_expression_l144_144493

noncomputable def f (t : ℝ) : ℝ := (3 ^ t - 2 * t ^ 2) * t / (9 ^ t)

theorem maximum_value_of_expression : 
  ∃ t : ℝ, ∀ x : ℝ, f x ≤ f t ∧ f t = 1 / 8 :=
begin
  sorry
end

end maximum_value_of_expression_l144_144493


namespace exists_circle_through_points_intersecting_line_with_chord_length_l144_144479

open Classical
noncomputable theory

structure Point (α : Type) := 
(x : α) 
(y : α)

structure Circle (α : Type) := 
(center : Point α) 
(radius : α)

structure Line (α : Type) := 
(a : α) 
(b : α) 
(c : α)

def distance {α : Type} [LinearOrderedField α] (p1 p2 : Point α) : α :=
((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2).sqrt

def intersects {α : Type} [LinearOrderedField α] (c : Circle α) (l : Line α) : Prop := 
sorry -- Define intersection property

def chord_length_on_line {α : Type} [LinearOrderedField α] 
  (c : Circle α) (l : Line α) (p : α) : Prop :=
∃ M N : Point α, intersects c l ∧ distance M N = p

theorem exists_circle_through_points_intersecting_line_with_chord_length 
  {α : Type} [LinearOrderedField α]
  (A B : Point α) (l : Line α) (p : α) : 
  ∃ c : Circle α, (distance A c.center = c.radius ∧ distance B c.center = c.radius) ∧ chord_length_on_line c l p :=
sorry

end exists_circle_through_points_intersecting_line_with_chord_length_l144_144479


namespace cara_pairs_between_l144_144204

theorem cara_pairs_between (friends : Fin 8) (emma : Fin 8) (cara : Fin 8) :
  (emma ≠ cara ∧ ∀ f : Fin 8, f ≠ emma ∧ f ≠ cara → true) →
  ∃ (n : Nat), n = 6 :=
by
  sorry

end cara_pairs_between_l144_144204


namespace four_digit_numbers_count_l144_144316

theorem four_digit_numbers_count :
  ∃ n : ℕ, n = 4140 ∧
  (∀ d1 d2 d3 d4 : ℕ,
    (4 ≤ d1 ∧ d1 ≤ 9) ∧
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d2 * d3 > 8) →
    (∃ m : ℕ, m = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ m > 3999) →
    n = 4140) :=
sorry

end four_digit_numbers_count_l144_144316


namespace number_of_geese_more_than_ducks_l144_144095

theorem number_of_geese_more_than_ducks (geese ducks : ℝ) (h1 : geese = 58.0) (h2 : ducks = 37.0) :
  geese - ducks = 21.0 :=
by
  sorry

end number_of_geese_more_than_ducks_l144_144095


namespace johns_train_speed_l144_144374

noncomputable def average_speed_of_train (D : ℝ) (V_t : ℝ) : ℝ := D / (0.8 * D / V_t + 0.2 * D / 20)

theorem johns_train_speed (D : ℝ) (V_t : ℝ) (h1 : average_speed_of_train D V_t = 50) : V_t = 64 :=
by
  sorry

end johns_train_speed_l144_144374


namespace green_chips_count_l144_144471

variable (total_chips : ℕ) (blue_chips : ℕ) (white_percentage : ℕ) (blue_percentage : ℕ)
variable (green_percentage : ℕ) (green_chips : ℕ)

def chips_condition1 : Prop := blue_chips = (blue_percentage * total_chips) / 100
def chips_condition2 : Prop := blue_percentage = 10
def chips_condition3 : Prop := white_percentage = 50
def green_percentage_calculation : Prop := green_percentage = 100 - (blue_percentage + white_percentage)
def green_chips_calculation : Prop := green_chips = (green_percentage * total_chips) / 100

theorem green_chips_count :
  (chips_condition1) →
  (chips_condition2) →
  (chips_condition3) →
  (green_percentage_calculation) →
  (green_chips_calculation) →
  green_chips = 12 :=
by
  intros
  sorry

end green_chips_count_l144_144471


namespace average_consecutive_pairs_in_subset_l144_144491

open Finset

/-- Prove that the average number of pairs of consecutive integers in a randomly selected subset of
    6 distinct integers chosen from the set {1, 2, ..., 40} is 2/3. -/
theorem average_consecutive_pairs_in_subset :
  (let s := univ.filter (λ (x : ℕ), x ≥ 1 ∧ x ≤ 40) in
   ∑ x in (powerset_len 6 s), 
     (∑ i in range 5, if x.to_list.nth i + 1 = x.to_list.nth (i + 1) then 1 else 0 : ℝ) /
   (powerset_len 6 s).card) = (2 / 3 : ℝ) :=
sorry

end average_consecutive_pairs_in_subset_l144_144491


namespace parabola_intercepts_l144_144798

noncomputable def question (y : ℝ) := 3 * y ^ 2 - 9 * y + 4

theorem parabola_intercepts (a b c : ℝ) (h_a : a = question 0) (h_b : 3 * b ^ 2 - 9 * b + 4 = 0) (h_c : 3 * c ^ 2 - 9 * c + 4 = 0) :
  a + b + c = 7 :=
by
  sorry

end parabola_intercepts_l144_144798


namespace area_of_parallelogram_l144_144135

-- Define the base and height of the parallelogram
def base : ℝ := 12
def height : ℝ := 6

-- Define the area of the parallelogram
def area (base height : ℝ) : ℝ := base * height

-- The statement to be proved
theorem area_of_parallelogram : area base height = 72 := by
  sorry

end area_of_parallelogram_l144_144135


namespace cylinder_lateral_surface_area_l144_144081

theorem cylinder_lateral_surface_area (r l : ℝ) (A : ℝ) (h_r : r = 1) (h_l : l = 2) : A = 4 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l144_144081


namespace ted_alex_age_ratio_l144_144100

theorem ted_alex_age_ratio (t a : ℕ) 
  (h1 : t - 3 = 4 * (a - 3))
  (h2 : t - 5 = 5 * (a - 5)) : 
  ∃ x : ℕ, (t + x) / (a + x) = 3 ∧ x = 1 :=
by
  sorry

end ted_alex_age_ratio_l144_144100


namespace find_repeating_digits_l144_144172

-- Specify given conditions
def incorrect_result (a : ℚ) (b : ℚ) : ℚ := 54 * b - 1.8
noncomputable def correct_multiplication_value (d: ℚ) := 2 + d
noncomputable def repeating_decimal_value : ℚ := 2 + 35 / 99

-- Define what needs to be proved
theorem find_repeating_digits : ∃ (x : ℕ), x * 100 = 35 := by
  sorry

end find_repeating_digits_l144_144172


namespace certain_number_l144_144335

theorem certain_number (x y : ℕ) (h₁ : x = 14) (h₂ : 2^x - 2^(x - 2) = 3 * 2^y) : y = 12 :=
  by
  sorry

end certain_number_l144_144335


namespace extra_time_walk_vs_bike_l144_144524

-- Given conditions as definitions
def blocks : ℕ := 9
def walk_time_per_block : ℝ := 1 -- time in minutes
def bike_time_per_block : ℝ := 20 / 60 -- time in minutes

-- Define the total time calculation functions
noncomputable def total_walk_time (blocks : ℕ) : ℝ := blocks * walk_time_per_block
noncomputable def total_bike_time (blocks : ℕ) : ℝ := blocks * bike_time_per_block

-- The mathematically equivalent proof problem statement
theorem extra_time_walk_vs_bike : total_walk_time blocks - total_bike_time blocks = 6 := by
  sorry

end extra_time_walk_vs_bike_l144_144524


namespace pure_imag_condition_l144_144292

theorem pure_imag_condition (a b : ℝ) : 
  (z : ℂ) = complex.I * (a + b * complex.I) 
  → (z.re = 0 ∧ z.im ≠ 0 ↔ a ≠ 0 ∧ b = 0) := 
by
  sorry

end pure_imag_condition_l144_144292


namespace total_roses_planted_l144_144833

def roses_planted_two_days_ago := 50
def roses_planted_yesterday := roses_planted_two_days_ago + 20
def roses_planted_today := 2 * roses_planted_two_days_ago

theorem total_roses_planted :
  roses_planted_two_days_ago + roses_planted_yesterday + roses_planted_today = 220 := by
  sorry

end total_roses_planted_l144_144833


namespace comparison_f_values_l144_144286

variables {R : Type} [OrderedRing R]

-- f is defined on ℝ and is an even function
def even_function (f : R → R) : Prop := ∀ x, f x = f (-x)

-- f is an increasing function when x ∈ [0, +∞)
def increasing_on_nonneg (f : R → R) : Prop := ∀ ⦃x y⦄, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem comparison_f_values
  {f : ℝ → ℝ}
  (hf_even : even_function f)
  (hf_increasing : increasing_on_nonneg f) :
  f real.pi > f (-3 : ℝ) ∧ f (-3 : ℝ) > f (-2 : ℝ) := 
by
  sorry

end comparison_f_values_l144_144286


namespace trip_duration_l144_144877

theorem trip_duration 
    (h_avg1 : ∀ h1 : ℕ, h1 = 4 → avg_speed1 : ℕ, avg_speed1 = 75)
    (h_avg2 : ∀ h2 : ℕ, h2 > 0 → avg_speed2 : ℕ, avg_speed2 = 60)
    (h_avg_total : avg_speed_total : ℕ, avg_speed_total = 70):
    total_hours : ℕ := 
begin
  let d1 := 75 * 4,
  let x := 2,
  let d2 := 60 * x,
  let total_distance := d1 + d2,
  let total_time := 4 + x,
  have h_avg := total_distance / total_time,
  show total_time = 6,
  sorry 
end

end trip_duration_l144_144877


namespace remainder_when_divided_by_7_l144_144118

theorem remainder_when_divided_by_7
  (x : ℤ) (k : ℤ) (h : x = 52 * k + 19) : x % 7 = 5 :=
sorry

end remainder_when_divided_by_7_l144_144118


namespace campers_last_week_l144_144548

theorem campers_last_week :
  ∀ (total_campers : ℕ) (campers_two_weeks_ago : ℕ) (campers_diff : ℕ),
    (total_campers = 150) →
    (campers_two_weeks_ago = 40) →
    (campers_two_weeks_ago - campers_diff = campers_diff + 10) →
    ∃ (campers_last_week : ℕ), campers_last_week = 80 :=
by
  intros total_campers campers_two_weeks_ago campers_diff
  assume h1 h2 h3
  have h4 : campers_diff = 30 := by linarith
  have h5 : campers_last_week = total_campers - (campers_diff + campers_two_weeks_ago) := by linarith
  use (80 : ℕ)
  exact h5
  sorry

end campers_last_week_l144_144548


namespace compare_abc_l144_144259

noncomputable def a : ℝ := (2 / 5) ^ (3 / 5)
noncomputable def b : ℝ := (2 / 5) ^ (2 / 5)
noncomputable def c : ℝ := (3 / 5) ^ (2 / 5)

theorem compare_abc : a < b ∧ b < c := sorry

end compare_abc_l144_144259


namespace binomial_12_0_eq_one_l144_144964

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_12_0_eq_one : binomial 12 0 = 1 :=
by
  -- This is where the mathematical steps would be placed
  sorry

end binomial_12_0_eq_one_l144_144964


namespace minor_axis_length_l144_144070

theorem minor_axis_length {x y : ℝ} (h : x^2 / 16 + y^2 / 9 = 1) : 6 = 6 :=
by
  sorry

end minor_axis_length_l144_144070


namespace solve_eq_64_16_pow_x_minus_1_l144_144777

theorem solve_eq_64_16_pow_x_minus_1 (x : ℝ) (h : 64 = 4 * (16 : ℝ) ^ (x - 1)) : x = 2 :=
sorry

end solve_eq_64_16_pow_x_minus_1_l144_144777


namespace max_m_value_l144_144981

theorem max_m_value (a : ℚ) (m : ℚ) : (∀ x : ℤ, 0 < x ∧ x ≤ 50 → ¬ ∃ y : ℤ, y = m * x + 3) ∧ (1 / 2 < m) ∧ (m < a) → a = 26 / 51 :=
by sorry

end max_m_value_l144_144981


namespace range_of_m_l144_144397

def A := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
def B (m : ℝ) := { x : ℝ | x^2 - (2 * m + 1) * x + 2 * m < 0 }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → (-1 / 2 ≤ m ∧ m ≤ 1) :=
by
  sorry

end range_of_m_l144_144397


namespace roses_per_bush_l144_144733

theorem roses_per_bush
  (petals_per_ounce : ℕ)
  (petals_per_rose : ℕ)
  (num_bushes : ℕ)
  (num_bottles : ℕ)
  (ounces_per_bottle : ℕ)
  (perfume_to_petals : petals_per_ounce = 320)
  (rose_to_petals : petals_per_rose = 8)
  (bushes_to_bottles : num_bushes = 800)
  (bottles_to_ounces : num_bottles = 20)
  (ounces_per_bottle_eq : ounces_per_bottle = 12) :
  (num_bottles * ounces_per_bottle * petals_per_ounce) / petals_per_rose / num_bushes = 12 := 
by
  rw [perfume_to_petals, rose_to_petals, bushes_to_bottles, bottles_to_ounces, ounces_per_bottle_eq]
  norm_num
  sorry

end roses_per_bush_l144_144733


namespace geometric_series_statements_l144_144206

-- Define the infinite geometric series
def geometric_series (n : ℕ) : ℝ := 3 / (2 ^ n)

-- Prove the statements
theorem geometric_series_statements :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, abs (geometric_series n) < ε) ∧
  (∀ ε > 0, abs (6 - ∑' n, geometric_series n) < ε) ∧
  (∑' n, geometric_series n = 6) :=
by
  sorry

end geometric_series_statements_l144_144206


namespace area_triangle_MPQ_l144_144808

namespace Geometry

-- Define lengths of the rectangle
def lengthLM := 8
def widthLO := 6

-- Define the area of the triangle MPQ
def areaMPQ : ℝ :=
  let lengthLN := Real.sqrt (lengthLM^2 + widthLO^2)
  let segment := lengthLN / 4
  (1/2) * segment * widthLO

-- Statement to be proved
theorem area_triangle_MPQ :
  areaMPQ = 7.5 :=
sorry

end Geometry

end area_triangle_MPQ_l144_144808


namespace basketball_scoring_l144_144543

-- Definitions corresponding to the conditions
def num_baskets : ℕ := 8
def points (x y z : ℕ) : ℕ := x + 2 * y + 3 * z

-- Main theorem statement
theorem basketball_scoring :
  {S : ℕ | ∃ (x y z : ℕ), x + y + z = num_baskets ∧ S = points x y z}.card = 17 :=
sorry

end basketball_scoring_l144_144543


namespace total_cost_l144_144079

theorem total_cost :
  ∀ (cost_caramel cost_candy cost_cotton : ℕ),
  (cost_candy = 2 * cost_caramel) →
  (cost_cotton = (1 / 2) * (4 * cost_candy)) →
  (cost_caramel = 3) →
  6 * cost_candy + 3 * cost_caramel + cost_cotton = 57 :=
begin
  intro cost_caramel, intro cost_candy, intro cost_cotton,
  assume h1 : cost_candy = 2 * cost_caramel,
  assume h2 : cost_cotton = (1 / 2) * (4 * cost_candy),
  assume h3 : cost_caramel = 3,
  sorry
end

end total_cost_l144_144079


namespace smallest_geometric_sequence_number_l144_144507

theorem smallest_geometric_sequence_number :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
    (∀ d ∈ [((n / 100) % 10), ((n / 10) % 10), (n % 10)], d ∈ [1,2,3,4,5,6,7,8,9]) ∧
    (let digits := [((n / 100) % 10), ((n / 10) % 10), (n % 10)] in
       digits.nodup ∧
       ∃ r : ℕ, r > 1 ∧ digits = [digits.head!, digits.head! * r, digits.head! * r * r]) ∧
    n = 124 :=
begin
  sorry
end

end smallest_geometric_sequence_number_l144_144507


namespace number_of_odd_sum_pairs_l144_144462

theorem number_of_odd_sum_pairs :
  let S := {1, 2, 3, 4} in
  (S.choose 2).count (λ x, (x.sum % 2 = 1)) = 4 :=
by {
  sorry
}

end number_of_odd_sum_pairs_l144_144462


namespace calculation_problem_l144_144195

theorem calculation_problem : (0.5 ^ 4) / (0.05 ^ 3) = 500 := 
by sorry

end calculation_problem_l144_144195


namespace green_chips_l144_144475

def total_chips (T : ℕ) := 3 = 0.10 * T

def white_chips (T : ℕ) := 0.50 * T

theorem green_chips (T : ℕ) (h1 : total_chips T) (h2 : white_chips T) : (T - (3 + h2) = 12) :=
by sorry

end green_chips_l144_144475


namespace monotonic_decreasing_a_1_max_value_a_le_1_max_value_a_gt_1_max_a_for_f_le_0_l144_144677

-- Statement 1
theorem monotonic_decreasing_a_1 (f : ℝ → ℝ) : 
  (∀ x, f x = 2 * (1:ℝ) * Real.log x - x^2 + 1) → 
  ∀ x, x > 1 → (f' x < 0) :=
sorry

-- Statement 2a
theorem max_value_a_le_1 (f : ℝ → ℝ) (a : ℝ) : 
  (0 < a ∧ a ≤ 1) → 
  (∀ x, x ≥ 1 → f x = 2 * a * Real.log x - x^2 + 1) → 
  (f 1 = 0) ∧ (∀ x, x ≥ 1 → f x ≤ f 1) :=
sorry

-- Statement 2b
theorem max_value_a_gt_1 (f : ℝ → ℝ) (a : ℝ) : 
  (a > 1) → 
  (∀ x, x ≥ 1 → f x = 2 * a * Real.log x - x^2 + 1) → 
  let x_max := Real.sqrt a in 
  (f x_max = a * Real.log a - a + 1) ∧ (∀ x, x ≥ 1 → f x ≤ f x_max) :=
sorry

-- Statement 3
theorem max_a_for_f_le_0 (f : ℝ → ℝ) : 
  (∀ x, x ≥ 1 → f x = 2 * a * Real.log x - x^2 + 1) → 
  (∀ x, x ≥ 1 → f x ≤ 0) → 
  (a ≤ 1) :=
sorry

end monotonic_decreasing_a_1_max_value_a_le_1_max_value_a_gt_1_max_a_for_f_le_0_l144_144677


namespace solve_for_x_l144_144703

theorem solve_for_x (x : ℝ) (h : 2 * x - 5 = 15) : x = 10 :=
sorry

end solve_for_x_l144_144703


namespace arithmetic_sequence_range_of_Sn_l144_144732

-- Problem conditions
def a (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 8
  else if n = 3 then 24
  else sorry  -- The rest of the sequence definition could be derived from the given information, but is not explicitly provided.

-- Sum of the first n terms of the sequence
def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, a i.succ)

-- Problem statements
theorem arithmetic_sequence (n : ℕ) : 
  is_arithmetic_sequence (λ n, a n / 2^n) :=
sorry

theorem range_of_Sn (n : ℕ) : 
  0 < 1 / S n ∧ 1 / S n ≤ 1/2 :=
sorry

end arithmetic_sequence_range_of_Sn_l144_144732


namespace example_gauss_lemma_l144_144007

open Nat LegendreSymbol

def is_odd_prime (p : ℕ) := Prime p ∧ p % 2 = 1

def gauss_lemma (p q : ℕ) (r : ℕ → ℕ) (mu : ℕ) : Prop :=
  is_odd_prime p ∧ (q < p ∧ q > 0) ∧ (∀ l, 1 ≤ l ∧ l ≤ (p - 1) / 2 → (1 ≤ r l ∧ r l ≤ (p - 1) / 2) ∧ (l * q ≡ r l [MOD p] ∨ l * q ≡ -r l [MOD p])) →
  legendreSym q p = (-1) ^ mu

-- Example usage:
theorem example_gauss_lemma (p q mu : ℕ) (r : ℕ → ℕ)
  (hp : is_odd_prime p)
  (hq : q < p ∧ q > 0)
  (hr : ∀ l, 1 ≤ l ∧ l ≤ (p - 1) / 2 → (1 ≤ r l ∧ r l ≤ (p - 1) / 2) ∧ (l * q ≡ r l [MOD p] ∨ l * q ≡ -r l [MOD p])) :
  legendreSym q p = (-1) ^ mu :=
  by sorry

end example_gauss_lemma_l144_144007


namespace other_x_intercept_l144_144949

noncomputable def ellipse_x_intercepts (f1 f2 : ℝ × ℝ) (x_intercept1 : ℝ × ℝ) : ℝ × ℝ :=
  let d := dist f1 x_intercept1 + dist f2 x_intercept1
  let x := (d^2 - 2 * d * sqrt (3^2 + (d / 2 - 4)^2)) / (2 * d - 8)
  (x, 0)

theorem other_x_intercept :
  ellipse_x_intercepts (0, 3) (4, 0) (0, 0) = (56 / 11, 0) :=
by
  sorry

end other_x_intercept_l144_144949


namespace train_journey_duration_l144_144532

def battery_lifespan (talk_time standby_time : ℝ) :=
  talk_time <= 6 ∧ standby_time <= 210

def full_battery_usage (total_time : ℝ) :=
  (total_time / 2) / 6 + (total_time / 2) / 210 = 1

theorem train_journey_duration (t : ℝ) (h1 : battery_lifespan (t / 2) (t / 2)) (h2 : full_battery_usage t) :
  t = 35 / 3 :=
sorry

end train_journey_duration_l144_144532


namespace at_least_three_consecutive_color_cards_l144_144417

-- Definitions
def standard_deck := list(card) -- Representing a deck as a list of cards
-- Assume a card can be red or black
inductive color | red | black
structure card := (color : color)

def num_red (deck : list card) (n : ℕ) : ℕ :=
  (deck.take n).countP (λ c, c.color = color.red)

def num_black (deck : list card) (n : ℕ) : ℕ :=
  (deck.drop (deck.length - n)).countP (λ c, c.color = color.black)

def consecutive_color_cards (deck : list card) (n : ℕ) : Prop :=
  ∃ (i : ℕ), i + n ≤ deck.length ∧ 
    (deck.drop i).take n = list.replicate n ⟨color.red⟩ ∨ 
    (deck.drop i).take n = list.replicate n ⟨color.black⟩

-- Problem statement
theorem at_least_three_consecutive_color_cards 
  (deck : list card)
  (hsize : deck.length = 52)
  (h_shuffled : ∀ i j (h1 : i ≠ j), deck.nth i ≠ none ∧ deck.nth j ≠ none → deck.nth i ≠ deck.nth j) -- condition of being shuffled
  (h_condition : num_red deck 26 > num_black deck 26) :
  consecutive_color_cards deck 3 :=
sorry

end at_least_three_consecutive_color_cards_l144_144417


namespace solve_for_a_l144_144289

-- Defining the equation and given solution
theorem solve_for_a (x a : ℝ) (h : 2 * x - 5 * a = 3 * a + 22) (hx : x = 3) : a = -2 := by
  sorry

end solve_for_a_l144_144289


namespace total_students_l144_144822

def numGirls : ℕ := 25
def diffGirlsBoys : ℕ := 3
def numBoys : ℕ := numGirls - diffGirlsBoys

theorem total_students (numGirls : ℕ) (numBoys : ℕ) : numGirls = 25 → numBoys = 22 → numGirls + numBoys = 47 :=
by
  intro h1 h2
  rw [h1, h2]
  exact rfl

#eval total_students 25 22 rfl rfl

end total_students_l144_144822


namespace difference_of_squares_65_35_l144_144595

theorem difference_of_squares_65_35 :
  let a := 65
  let b := 35
  a^2 - b^2 = (a + b) * (a - b) :=
by
  sorry

end difference_of_squares_65_35_l144_144595


namespace train_speeds_l144_144838

-- Define the parameters
variable (lengthA lengthB : ℝ) -- lengths of the trains in meters
variable (speedA : ℝ) -- speed of the slower train in kmph
variable (time_seconds : ℝ) -- time taken to cross each other in seconds

-- Convert time from seconds to hours and distance from meters to kilometers
def time_hours := time_seconds / 3600
def total_distance_km := (lengthA + lengthB) / 1000

-- Define the proposition to be proved
theorem train_speeds (lengthA lengthB : ℝ) (speedA : ℝ) (time_seconds : ℝ) : 
    lengthA = 200 → 
    lengthB = 150 → 
    speedA = 40 → 
    time_seconds = 210 →
    ∀ (V_f : ℝ), V_f - speedA = total_distance_km / time_hours → V_f = 45.95 :=
by
  intros h1 h2 h3 h4 V_f hyp
  sorry

end train_speeds_l144_144838


namespace largest_n_factors_l144_144245

theorem largest_n_factors (n : ℤ) :
  (∃ A B : ℤ, 3 * B + A = n ∧ A * B = 72) → n ≤ 217 :=
by {
  sorry
}

end largest_n_factors_l144_144245


namespace find_k_l144_144306

variable (k : ℝ)
noncomputable def vec_a : ℝ × ℝ × ℝ := (-1, 0, 2)
noncomputable def vec_b : ℝ × ℝ × ℝ := (1, 1, 0)
noncomputable def vec_sum := (vec_a.1 + k * vec_b.1, vec_a.2 + k * vec_b.2, vec_a.3 + k * vec_b.3)
noncomputable def vec_diff := (2 * vec_b.1 - vec_a.1, 2 * vec_b.2 - vec_a.2, 2 * vec_b.3 - vec_a.3)
noncomputable def dot_product : ℝ :=
  vec_sum.1 * vec_diff.1 + vec_sum.2 * vec_diff.2 + vec_sum.3 * vec_diff.3

theorem find_k (h : dot_product k = 0) : k = 7 / 5 := by
  sorry

end find_k_l144_144306


namespace sequence_limit_l144_144137

open Real

-- Define the sequence as a function of n
def sequence (n : ℕ) : ℝ :=
  (sqrt ((n^4 + 1)*(n^2 - 1)) - sqrt (n^6 - 1)) / n

-- State the limit problem
theorem sequence_limit :
  tendsto (λ n : ℕ, sequence n) atTop (𝓝 (-1/2)) :=
sorry

end sequence_limit_l144_144137


namespace log_base_4_frac_l144_144225

theorem log_base_4_frac :
  logb 4 (1/64) = -3 :=
sorry

end log_base_4_frac_l144_144225


namespace base_conversion_l144_144365

theorem base_conversion (x : ℕ) (h : 4 * x + 7 = 71) : x = 16 := 
by {
  sorry
}

end base_conversion_l144_144365


namespace domain_and_range_of_f_shifted_l144_144706

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder for f, since detailed definition isn't given.

-- Conditions
def domain_f := Icc (0 : ℝ) 1
def range_f := Icc (1 : ℝ) 2

-- Question Restatement
def domain_f_shifted := Icc (-2 : ℝ) (-1)
def range_f_shifted := Icc (1 : ℝ) 2

-- Proof problem: state that the domain of f(x+2) is [-2, -1] and its range is [1, 2]
theorem domain_and_range_of_f_shifted :
  (∀ x, x ∈ domain_f_shifted → (x+2) ∈ domain_f) ∧ 
  (∀ y ∈ range_f, y ∈ range_f_shifted) :=
by 
  sorry

end domain_and_range_of_f_shifted_l144_144706


namespace ratio_of_areas_l144_144020

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {A B C M : V}

-- Define the collinearity condition point M in the triangle plane with respect to vectors AB and AC
def point_condition (A B C M : V) : Prop :=
  5 • (M - A) = (B - A) + 3 • (C - A)

-- Define an area ratio function
def area_ratio_triangles (A B C M : V) [AddCommGroup V] [Module ℝ V] : ℝ :=
  sorry  -- Implementation of area ratio comparison, abstracted out for the given problem statement

-- The theorem to prove
theorem ratio_of_areas (hM : point_condition A B C M) : area_ratio_triangles A B C M = 3 / 5 :=
sorry

end ratio_of_areas_l144_144020


namespace statement_A_statement_B_statement_C_statement_D_l144_144730

variables (A B C a b c : ℝ)

-- Given conditions
def law_of_sines : Prop := a / Real.sin A = b / Real.sin B ∧ a / Real.sin A = c / Real.sin C

def cos_condition : Prop := a * Real.cos B = b * Real.cos A

def acute_triangle (A B C : ℝ) : Prop := A < π / 2 ∧ B < π / 2 ∧ C < π / 2

-- Statements to be proved
theorem statement_A (h1 : law_of_sines A B C a b c) : a / Real.sin A = (b + c) / (Real.sin B + Real.sin C) := 
sorry

theorem statement_B (h2 : cos_condition a b A B) : a = b := 
sorry

theorem statement_C (h3 : Real.sin (2 * A) = Real.sin (2 * B)) : ¬ ((A = B) ∨ (A + B = π / 2)) := 
sorry

theorem statement_D (h4 : acute_triangle A B C) : Real.sin B > Real.cos C := 
sorry

end statement_A_statement_B_statement_C_statement_D_l144_144730


namespace ned_did_not_wash_1_l144_144015

theorem ned_did_not_wash_1 (short_sleeve_shirts : ℕ) (long_sleeve_shirts : ℕ) (washed_shirts : ℕ) :
  short_sleeve_shirts = 9 →
  long_sleeve_shirts = 21 →
  washed_shirts = 29 →
  let total_shirts := short_sleeve_shirts + long_sleeve_shirts in
  total_shirts - washed_shirts = 1 :=
by
  intros h1 h2 h3
  let total_shirts := short_sleeve_shirts + long_sleeve_shirts
  sorry

end ned_did_not_wash_1_l144_144015


namespace KC_eq_LC_l144_144753

noncomputable def point : Type := sorry -- Placeholder for the type of points

variables (A B C D E K L : point) -- Variables for the points

-- Definitions for the sides and conditions
def on_side_BC (D : point) : Prop := sorry
def on_side_AC (E : point) : Prop := sorry
def equal_segments (BD AE : point) : Prop := sorry
def circumcenter (T : Type) : point := sorry -- Placeholder for circumcenter of a triangle
def intersection_ACBC_at_KL (O1 O2 : point) (AC BC : point) : point := sorry -- Placeholder for such intersections

-- Assume required conditions
axiom D_on_BC : on_side_BC D
axiom E_on_AC : on_side_AC E
axiom BD_eq_AE : equal_segments D E

-- Circumcenters of triangles ADC and BEC
def O1 := circumcenter (A, D, C)
def O2 := circumcenter (B, E, C)

-- Definitions for \( K \) and \( L \)
def K := intersection_ACBC_at_KL O1 O2 A C -- Intersection at AC
def L := intersection_ACBC_at_KL O1 O2 B C -- Intersection at BC

-- The theorem to be proven
theorem KC_eq_LC : distance K C = distance L C := sorry

end KC_eq_LC_l144_144753


namespace locus_of_point_C_l144_144209

variable {S A B C : Point}
-- Given angle ASB
variable (angle_ASB : ℝ)

-- Circle tangent conditions
variable (circle_tangent_SA_at_A : Circle)
variable (circle_tangent_SB_at_B : Circle)
variable (circles_tangent_at_C : circle_tangent_SA_at_A.isTangentAt (circle_tangent_SB_at_B, C))

-- Plane geometry context
open Geometry

-- Define constant angle condition
noncomputable def angle_ACB : ℝ :=
  180 - (1 / 2) * angle_ASB

-- Define the locus of C with the provided condition
theorem locus_of_point_C :
  ∀ {C : Point}, circle_tangent_SA_at_A.isTangentAt (circle_tangent_SB_at_B, C) → 
  ∠(A, C, B) = angle_ACB angle_ASB :=
begin
  intros C hC,
  -- Statements go here to prove the theorem
  sorry
end

end locus_of_point_C_l144_144209


namespace four_digit_numbers_l144_144324

theorem four_digit_numbers (n : ℕ) :
    (∃ a b c d : ℕ, 
        n = a * 1000 + b * 100 + c * 10 + d 
        ∧ 4 ≤ a ∧ a ≤ 9 
        ∧ 1 ≤ b ∧ b ≤ 9 
        ∧ 1 ≤ c ∧ c ≤ 9 
        ∧ 0 ≤ d ∧ d ≤ 9 
        ∧ b * c > 8) → n ∈ {n | 4000 ≤ n ∧ n < 10000}
           → n ∈ {n | 4000 ≤ n ∧ n < 10000 ∧ b * c > 8} := sorry

end four_digit_numbers_l144_144324


namespace range_of_independent_variable_l144_144723

def domain_of_function := { x : ℝ // x > 2 }

def function_definition (x : domain_of_function) : ℝ := 2 / Real.sqrt (x.val - 2)

theorem range_of_independent_variable (x : ℝ) (h : y = 2 / Real.sqrt (x - 2)) : x > 2 :=
begin
  sorry
end

end range_of_independent_variable_l144_144723


namespace number_of_integers_count_integers_l144_144693

theorem number_of_integers(n : ℕ) : 100 < n^2 ∧ n^2 < 1400 → 11 ≤ n ∧ n ≤ 37 := sorry

theorem count_integers : { n : ℕ // 100 < n^2 ∧ n^2 < 1400 }.card = 27 := sorry

end number_of_integers_count_integers_l144_144693


namespace corn_plant_growth_l144_144954

theorem corn_plant_growth :
  let week1 := 2 in
  let week2 := 2 * week1 in
  let week3 := 4 * week2 in
  let week4 := week1 + week2 + week3 in
  let week5 := (week4 / 2) - 3 in
  let week6 := 2 * week5 in
  week1 + week2 + week3 + week4 + week5 + week6 = 68 :=
by
  let week1 := 2
  let week2 := 2 * week1
  let week3 := 4 * week2
  let week4 := week1 + week2 + week3
  let week5 := (week4 / 2) - 3
  let week6 := 2 * week5
  have sum := week1 + week2 + week3 + week4 + week5 + week6
  show sum = 68
  sorry

end corn_plant_growth_l144_144954


namespace inverse_of_5_mod_34_l144_144236

theorem inverse_of_5_mod_34 : ∃ x : ℕ, (5 * x) % 34 = 1 ∧ 0 ≤ x ∧ x < 34 :=
by
  use 7
  have h : (5 * 7) % 34 = 1 := by sorry
  exact ⟨h, by norm_num, by norm_num⟩

end inverse_of_5_mod_34_l144_144236


namespace percentage_reduction_in_price_l144_144159

variable (R P : ℝ) (R_eq : R = 30) (H : 600 / R - 600 / P = 4)

theorem percentage_reduction_in_price (R_eq : R = 30) (H : 600 / R - 600 / P = 4) :
  ((P - R) / P) * 100 = 20 := sorry

end percentage_reduction_in_price_l144_144159


namespace molecular_weight_CaO_l144_144113

theorem molecular_weight_CaO (m : ℕ -> ℝ) (h : m 7 = 392) : m 1 = 56 :=
sorry

end molecular_weight_CaO_l144_144113


namespace sqrt_inequality_sum_inverse_ge_9_l144_144141

-- (1) Prove that \(\sqrt{3} + \sqrt{8} < 2 + \sqrt{7}\)
theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 8 < 2 + Real.sqrt 7 := sorry

-- (2) Prove that given \(a > 0, b > 0, c > 0\) and \(a + b + c = 1\), \(\frac{1}{a} + \frac{1}{b} + \frac{1}{c} \geq 9\)
theorem sum_inverse_ge_9 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) : 
    1 / a + 1 / b + 1 / c ≥ 9 := sorry

end sqrt_inequality_sum_inverse_ge_9_l144_144141


namespace find_divisor_l144_144988

theorem find_divisor (n x y z a b c : ℕ) (h1 : 63 = n * x + a) (h2 : 91 = n * y + b) (h3 : 130 = n * z + c) (h4 : a + b + c = 26) : n = 43 :=
sorry

end find_divisor_l144_144988


namespace intercepts_sum_eq_seven_l144_144801

theorem intercepts_sum_eq_seven :
    (∃ a b c, (∀ y, (3 * y^2 - 9 * y + 4 = a) → y = 0) ∧ 
              (∀ y, (3 * y^2 - 9 * y + 4 = 0) → (y = b ∨ y = c)) ∧ 
              (a + b + c = 7)) := 
sorry

end intercepts_sum_eq_seven_l144_144801


namespace cost_of_ice_making_l144_144414

theorem cost_of_ice_making :
  let num_cubes := 10 * 16
  let hours := num_cubes / 10
  let ice_maker_cost := hours * 1.5
  let water_needed := num_cubes * 2
  let water_cost := water_needed * 0.1
  let total_cost := ice_maker_cost + water_cost
  total_cost = 56 :=
by
  let num_cubes := 10 * 16
  let hours := num_cubes / 10
  let ice_maker_cost := hours * 1.5
  let water_needed := num_cubes * 2
  let water_cost := water_needed * 0.1
  let total_cost := ice_maker_cost + water_cost
  sorry

end cost_of_ice_making_l144_144414


namespace radian_of_15_degrees_l144_144454

-- Define the degree to radian conversion factor
def degree_to_radian (deg : ℝ) : ℝ :=
  deg * (Real.pi / 180)

-- Prove that the radian measure of 15 degrees is π/12
theorem radian_of_15_degrees : 
  degree_to_radian 15 = Real.pi / 12 :=
by 
  sorry

end radian_of_15_degrees_l144_144454


namespace edge_equivalence_l144_144255

universe u

variables {V : Type u} [DecidableEq V]

structure Graph (V : Type u) :=
(edges : set (V × V)) (adj : V → V → Prop)
(adj_def : ∀ u v, adj u v ↔ (u, v) ∈ edges ∨ (v, u) ∈ edges)
(simple : ∀ u v, adj u v → adj v u)

variable {G : Graph V}

-- Define blocks, cycles, and cut-vertices in terms of edge membership
def same_block (G : Graph V) (e f : V × V) : Prop :=
∃ B : set (V × V), B ∈ blocks G ∧ e ∈ B ∧ f ∈ B

def same_cycle (G : Graph V) (e f : V × V) : Prop :=
∃ C : set (V × V), C ∈ cycles G ∧ e ∈ C ∧ f ∈ C

def same_cut_vertex (G : Graph V) (e f : V × V) : Prop :=
∃ v : V, cut_vertex G v ∧ (∃ C : set (V × V), e ∈ C ∧ f ∈ C ∧ (∃ u w, path G u v ∧ path G v w))

-- Statement of equivalence
theorem edge_equivalence (G : Graph V) (e f : V × V) (h_diff : e ≠ f) :
  (same_block G e f ↔ same_cycle G e f) ∧ 
  (same_cycle G e f ↔ same_cut_vertex G e f) ∧ 
  (same_cut_vertex G e f ↔ same_block G e f) :=
sorry

end edge_equivalence_l144_144255


namespace ellipse_equation_min_area_of_triangle_l144_144274

-- Define the conditions given in the problem
variables (a b x y e : ℝ)
variable (h1 : a > b)
variable (h2 : b > 0)
variable (h3 : e = sqrt 3 / 2)
variable (h4 : a^2 = b^2 + 3 / 4)

-- Define the equation of the ellipse using the given conditions
def ellipse_eq := x^2 / a^2 + y^2 / b^2 = 1

-- Prove that the equation of the ellipse is as specified
theorem ellipse_equation : a = 2 ∧ b = 1 → ellipse_eq x y (2 : ℝ) (1 : ℝ) :=
by {
  intros h,
  exact sorry
}

-- Variables to define the points A, B, C on the ellipse and other conditions
variables (A B C : ℝ × ℝ)
variable (symmetric_AB : A = (-B.1, -B.2))
variable (length_AC_equals_CB : dist A C = dist C B)

-- Derive the line equation when area of triangle ABC is minimized
theorem min_area_of_triangle (h_eq_min : isosceles_triangle A B C) 
  (h_min_area : min_area ∆ABC) : 
  (equation_of_line AB = y = x) ∨ (equation_of_line AB = y = -x) :=
by {
  intros h,
  exact sorry
}

end ellipse_equation_min_area_of_triangle_l144_144274


namespace parabola_intercepts_l144_144796

noncomputable def question (y : ℝ) := 3 * y ^ 2 - 9 * y + 4

theorem parabola_intercepts (a b c : ℝ) (h_a : a = question 0) (h_b : 3 * b ^ 2 - 9 * b + 4 = 0) (h_c : 3 * c ^ 2 - 9 * c + 4 = 0) :
  a + b + c = 7 :=
by
  sorry

end parabola_intercepts_l144_144796


namespace remainder_6_pow_305_div_11_l144_144115

theorem remainder_6_pow_305_div_11 :
  (6 ^ 305) % 11 = 10 :=
by {
  -- Given conditions
  have h1 : 6 % 11 = 6 := by norm_num,
  have h2 : (6 ^ 2) % 11 = 3 := by norm_num,
  have h3 : (6 ^ 3) % 11 = 7 := by norm_num,
  have h4 : (6 ^ 4) % 11 = 9 := by norm_num,
  have h5 : (6 ^ 5) % 11 = 10 := by norm_num,
  have h6 : (6 ^ 6) % 11 = 1 := by norm_num,
  have h7 : 305 % 6 = 5 := by norm_num,
  -- Result
  calc (6 ^ 305) % 11 = ((6 ^ 6) ^ 50 * 6 ^ 5) % 11 : by rw [pow_add, pow_mul]
                 ... = (1 ^ 50 * 6 ^ 5) % 11 : by rw h6
                 ... = 6 ^ 5 % 11 : by norm_num
                 ... = 10 : by exact h5
}

end remainder_6_pow_305_div_11_l144_144115


namespace magician_initial_decks_l144_144908

theorem magician_initial_decks
  (price_per_deck : ℕ)
  (decks_left : ℕ)
  (total_earned : ℕ)
  (price_per_deck_eq : price_per_deck = 7)
  (decks_left_eq : decks_left = 8)
  (total_earned_eq : total_earned = 56)
  : ∃ initial_decks : ℕ, initial_decks = 16 :=
by
  have decks_sold : ℕ := total_earned / price_per_deck,
  have initial_decks := decks_sold + decks_left,
  use initial_decks,
  rw [price_per_deck_eq, decks_left_eq, total_earned_eq],
  show 8 = total_earned / price_per_deck,
  show initial_decks = decks_sold + decks_left,
  sorry

end magician_initial_decks_l144_144908


namespace ellipse_x_intercept_l144_144939

theorem ellipse_x_intercept
  (foci1 foci2 : ℝ × ℝ)
  (x_intercept : ℝ × ℝ)
  (d : ℝ)
  (h_foci1 : foci1 = (0, 3))
  (h_foci2 : foci2 = (4, 0))
  (h_x_intercept : x_intercept = (0, 0))
  (h_d : d = 7)
  : ∃ x : ℝ, (x, 0) ≠ x_intercept ∧ (abs (x - 4) + real.sqrt (x^2 + 9) = 7) ∧ x = 56 / 11 := by
  sorry

end ellipse_x_intercept_l144_144939


namespace cos_alpha_beta_l144_144705

def sin_cos_eq (x : ℝ) : Prop := sin (2 * x) + 2 * cos (2 * x) = -2

theorem cos_alpha_beta {α β : ℝ} (hα : 0 ≤ α ∧ α < π) (hβ : 0 ≤ β ∧ β < π)
  (h_eq : ∃ (α β : ℝ), α ≠ β ∧ sin_cos_eq α ∧ sin_cos_eq β) :
  cos (α - β) = (2 * Real.sqrt 5) / 5 :=
sorry

end cos_alpha_beta_l144_144705


namespace simplify_expression_l144_144775

theorem simplify_expression :
  (let a := 3 in
  let b := (√34) / 2 in
  (a - b)^2 = (35 - 6 * √34) / 2) := sorry

end simplify_expression_l144_144775


namespace tree_height_at_2_years_l144_144918

-- Define the conditions
def triples_height (height : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, height (n + 1) = 3 * height n

def height_at_5_years (height : ℕ → ℕ) : Prop :=
  height 5 = 243

-- Set up the problem statement
theorem tree_height_at_2_years (height : ℕ → ℕ) 
  (H1 : triples_height height) 
  (H2 : height_at_5_years height) : 
  height 2 = 9 :=
sorry

end tree_height_at_2_years_l144_144918


namespace contradiction_example_l144_144487

theorem contradiction_example (x y : ℝ) (h : x < y) : x ^ (1 / 5) > y ^ (1 / 5) :=
by
  have contra : x ^ (1 / 5) ≤ y ^ (1 / 5) → false := sorry
  exact contra

end contradiction_example_l144_144487


namespace complement_set_A_l144_144304

open Set

theorem complement_set_A :
  let A : Set ℝ := {x | x < 0 ∨ x > 1}
  in Aᶜ = {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end complement_set_A_l144_144304


namespace smallest_three_digit_geometric_sequence_l144_144505

theorem smallest_three_digit_geometric_sequence : ∃ n : ℕ, n = 124 ∧ (∀ (a r : ℕ), a = 1 ∧ r * a < 10 ∧ r^2 * a < 100 → digits n = [1, r, r^2] ∧ (digits n).nodup) := by
  sorry

end smallest_three_digit_geometric_sequence_l144_144505


namespace minimal_polynomial_with_given_roots_l144_144627
-- Import the entire necessary library

variable (x : ℝ)

theorem minimal_polynomial_with_given_roots (x : ℝ) :
  (minimal_polynomial_with_given_roots (x : ℝ)).coeffs (x : ℝ) = [1, -4, 1, 8, -2] := 
  by
  -- Define the conditions for the polynomial
  let p := polynomial of minimal degree with rational coefficients, leading coefficient 1, and roots [sqrt(2), -sqrt(2), 2+sqrt(3), 2-sqrt(3)]
  have hp1 : p(x) = ((x - sqrt(2)) * (x + sqrt(2))),
  have hp2 : p(x) = (x - 2 - sqrt(3)) * (x - 2 + sqrt(3)),
  have hp3 : p = (x^2 - 2) * ((x^2 - 4x + 1)),
  exact hp1 + hp2 + hp3,
  sorry
end

end minimal_polynomial_with_given_roots_l144_144627


namespace find_N_l144_144878

theorem find_N : 
  let Sum := 555 + 445 in
  let Difference := 555 - 445 in
  let Quotient := 2 * Difference in
  let Remainder := 40 in
  let N := (Sum * Quotient) + Remainder in
  N = 220040 :=
by
  -- Definitions
  let Sum := 555 + 445
  let Difference := 555 - 445
  let Quotient := 2 * Difference
  let Remainder := 40
  let N := (Sum * Quotient) + Remainder
  
  -- Start proof (will be skipped)
  sorry

end find_N_l144_144878


namespace campers_last_week_correct_l144_144549

variable (C : ℝ)
variable (campers_last_week : ℝ)

theorem campers_last_week_correct 
  (h1 : C + (C + 10) + campers_last_week = 150)
  (h2 : C + 10 = 40) : 
  campers_last_week = 80 :=
by
  have hC : C = 30 := by linarith
  rw [hC] at h1
  rw [hC] at h2
  linarith

end campers_last_week_correct_l144_144549


namespace sin_shift_eq_l144_144826

theorem sin_shift_eq (x : ℝ) :
  sin (x - π / 4 + π / 2) = sin (x + π / 4) :=
by sorry

end sin_shift_eq_l144_144826


namespace tetrahedron_height_relation_l144_144728

-- Definitions of heights and lengths
variables (A B C D : Type) [Metric_Space A] [Metric_Space B] [Metric_Space C] [Metric_Space D] 

-- Variables to represent the lengths of the edges and the height
variables {a b c h : ℝ}

-- Tetrahedron with given conditions
variables (h : height := h)
variables (a b c : length := a b c)

-- The proof statement
theorem tetrahedron_height_relation 
  (h : ℝ) (a b c : ℝ)
  (right_angles_at_D : ∀ (A B C : Type), planes_angles_at_vertex_D_right ∧ length DA = a ∧ length DB = b ∧ length DC = c ∧ height_from_D = h) : 
  h^(-2) = a^(-2) + b^(-2) + c^(-2) := 
by sorry

end tetrahedron_height_relation_l144_144728


namespace change_in_money_supply_l144_144145

/-- Constants for initial and post-sanction exchange rates. --/
def E0 : ℝ := 90
def E1 : ℝ := 100

/-- Condition: Percentage change in exchange rate given initial and post-sanction exchange rates. --/
def percentage_change (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

/-- Condition: Function that models change in money supply and its effect on exchange rate. --/
def money_supply_effect (money_change : ℝ) : ℝ := money_change * 5

/-- Theorem: The required change in money supply to return the exchange rate to its initial level. --/
theorem change_in_money_supply :
  percentage_change E0 E1 = 11.11 → 2 = 11.11 / 5 := by
  sorry

end change_in_money_supply_l144_144145


namespace tree_height_at_end_of_2_years_l144_144920

-- Conditions:
-- 1. The tree tripled its height every year.
-- 2. The tree reached a height of 243 feet at the end of 5 years.
theorem tree_height_at_end_of_2_years (h5 : ℕ) (H5 : h5 = 243) : 
  ∃ h2, h2 = 9 := 
by sorry

end tree_height_at_end_of_2_years_l144_144920


namespace constant_term_in_expansion_l144_144789

theorem constant_term_in_expansion : 
      constant_term ((x^2 - (2 / x^3))^5) = 40 :=
sorry

end constant_term_in_expansion_l144_144789


namespace interest_rate_of_first_investment_l144_144895

theorem interest_rate_of_first_investment (x y : ℝ) (h1 : x + y = 2000) (h2 : y = 650) (h3 : 0.10 * x - 0.08 * y = 83) : (0.10 * x) / x = 0.10 := by
  sorry

end interest_rate_of_first_investment_l144_144895


namespace number_of_triangles_with_fixed_vertex_l144_144611

theorem number_of_triangles_with_fixed_vertex (circumference_points : Finset Point) (A : Point) (h_distinct: CircumferenceDistinctPoints circumference_points) (h_size: circumference_points.card = 8) (h_A : A ∈ circumference_points) :
  ∃ n : ℕ, n = 21 ∧ ∀ Δ : Triangle, A ∈ Δ.vertices → Δ ∈ triangles_with_point(A, circumference_points) → Δ ∈ triangles count.

end number_of_triangles_with_fixed_vertex_l144_144611


namespace compare_negative_positive_l144_144960

theorem compare_negative_positive : -897 < 0.01 := sorry

end compare_negative_positive_l144_144960


namespace range_of_a_l144_144443

def f (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ {x y : ℝ}, a < x ∧ x < y ∧ y ≤ b → f x ≥ f y

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 4 → ∀ y : ℝ, x < y ∧ y ≤ 4 → f a x ≥ f a y) → a ≤ -3 :=
begin
  sorry
end

end range_of_a_l144_144443


namespace eccentricity_of_hyperbola_l144_144683

variable (a b c e : ℝ)

noncomputable def hyperbola : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ 
  (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → 
  ((x, y) = (a, 0) ∨ (x, y) = (-c, b^2 / a) ∨ (x, y) = (-c, -b^2 / a)))

theorem eccentricity_of_hyperbola (a b : ℝ) (h: a > 0 ∧ b > 0)
  (right_vertex : (a, 0)) (left_focus : (-2*a, 0))
  (intersecs : ((-2*a, b^2 / a) ∧ (-2*a, -b^2 / a)))
  (angle_condition : ∠((-2*a, b^2 / a), (a, 0), (-2*a, -b^2 / a)) = 90) :
  ∃ e : ℝ, e = 2 := by
  sorry

end eccentricity_of_hyperbola_l144_144683


namespace pipe_B_fills_6_times_faster_l144_144415

theorem pipe_B_fills_6_times_faster :
  let R_A := 1 / 32
  let combined_rate := 7 / 32
  let R_B := combined_rate - R_A
  (R_B / R_A = 6) :=
by
  let R_A := 1 / 32
  let combined_rate := 7 / 32
  let R_B := combined_rate - R_A
  sorry

end pipe_B_fills_6_times_faster_l144_144415


namespace problem_proof_l144_144280

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- The conditions
def even_function_on_ℝ := ∀ x : ℝ, f x = f (-x)
def f_at_0_is_2 := f 0 = 2
def odd_after_translation := ∀ x : ℝ, f (x - 1) = -f (-x - 1)

-- Prove the required condition
theorem problem_proof (h1 : even_function_on_ℝ f) (h2 : f_at_0_is_2 f) (h3 : odd_after_translation f) :
    f 1 + f 3 + f 5 + f 7 + f 9 = 0 :=
by
  sorry

end problem_proof_l144_144280


namespace valid_four_digit_numbers_count_l144_144311

noncomputable def num_valid_four_digit_numbers : ℕ := 6 * 65 * 10

theorem valid_four_digit_numbers_count :
  num_valid_four_digit_numbers = 3900 :=
by
  -- We provide the steps in the proof to guide the automation
  let total_valid_numbers := 6 * 65 * 10
  have h1 : total_valid_numbers = 3900 := rfl
  exact h1

end valid_four_digit_numbers_count_l144_144311


namespace range_of_k_l144_144971

variable {a k x m n : ℝ}
variable {D : Set ℝ}

noncomputable def f (x : ℝ) : ℝ := log a (a^x + k)

theorem range_of_k (h₁ : a > 0) (h₂ : a ≠ 1) 
  (h₃ : ∃ m n, [m, n] ⊆ D ∧ (Set.range (f '' [m, n])) = [0.5 * m, 0.5 * n]) :
  0 < k ∧ k < 0.25 :=
sorry

end range_of_k_l144_144971


namespace loss_percentage_l144_144136

theorem loss_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 1500) (h_sell : selling_price = 1260) : 
  (cost_price - selling_price) / cost_price * 100 = 16 := 
by
  sorry

end loss_percentage_l144_144136


namespace uncle_welly_roses_l144_144832

theorem uncle_welly_roses :
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  roses_two_days_ago + roses_yesterday + roses_today = 220 :=
by
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  show roses_two_days_ago + roses_yesterday + roses_today = 220
  sorry

end uncle_welly_roses_l144_144832


namespace library_wall_leftover_space_l144_144528

theorem library_wall_leftover_space
  (D : ℕ) -- number of desks
  (B : ℕ) -- number of bookcases
  (h_eq : D = B) -- condition: equal number of desks and bookcases
  (wall_length : ℕ) -- length of the wall
  (desk_length : ℕ) -- length of each desk
  (bookcase_length : ℕ) -- length of each bookcase
  (h_wall : wall_length = 15) -- the wall length is 15 meters
  (h_desk : desk_length = 2) -- each desk is 2 meters long
  (h_bookcase : bookcase_length = 1.5) -- each bookcase is 1.5 meters long
  (h_max : ∀ (D B : ℕ), D = B → 2 * D + 1.5 * B ≤ 15) -- total length constraint
  : 15 - (2 * D + 1.5 * B) = 1 := -- the leftover space is 1 meter
begin
  sorry
end

end library_wall_leftover_space_l144_144528


namespace angle_EFD_eq_sum_angle_ACD_ECB_l144_144738

noncomputable def semicircle (AB : ℝ) : Set ℝ := sorry -- Defining the semicircle (abstract)
noncomputable def point_on_diameter (C : ℝ) (AB : ℝ) : Prop := sorry -- Definition of point on diameter (abstract)
noncomputable def point_on_semicircle (D E : ℝ) (AB : ℝ) : Prop := sorry -- Definition of points on semicircle (abstract)
noncomputable def angle_eq (∠ACD ∠ECB : ℝ) : Prop := sorry -- Angle equality condition (abstract)
noncomputable def tangents_intersect (D E : ℝ) (Γ : Set ℝ) : ℝ := sorry -- Intersection of tangents (abstract)

theorem angle_EFD_eq_sum_angle_ACD_ECB 
    (Γ : Set ℝ)
    (A B C D E F : ℝ)
    (AB_diameter : AB = B - A)
    (on_diameter : point_on_diameter C AB)
    (on_semicircle : point_on_semicircle D E AB)
    (angle_cond : angle_eq (∠ACD) (∠ECB))
    (tangent_intersect_F : F = tangents_intersect D E Γ) :
    ∠EFD = ∠ACD + ∠ECB :=
sorry

end angle_EFD_eq_sum_angle_ACD_ECB_l144_144738


namespace minimum_a_translation_l144_144680

-- Define function f
noncomputable def f (ω x : ℝ) := sin (ω * x) ^ 2 - 1 / 2

-- Define the conditions
variables {a ω : ℝ} (hω : ω > 0) (ha : a > 0)

-- Define the theorem statement
theorem minimum_a_translation :
  ∃ (a : ℝ), (a > 0 ∧
    (∀ x : ℝ, f ω (x - a) = f ω (-x + a)) → a = π / 4) :=
begin
  sorry,
end

end minimum_a_translation_l144_144680


namespace total_eggs_collected_l144_144585

-- Define the variables given in the conditions
def Benjamin_eggs := 6
def Carla_eggs := 3 * Benjamin_eggs
def Trisha_eggs := Benjamin_eggs - 4

-- State the theorem using the conditions and correct answer in the equivalent proof problem
theorem total_eggs_collected :
  Benjamin_eggs + Carla_eggs + Trisha_eggs = 26 := by
  -- Proof goes here.
  sorry

end total_eggs_collected_l144_144585


namespace assign_roles_ways_l144_144911

theorem assign_roles_ways :
  let men : Fin 6 := 6
  let women : Fin 7 := 7
  let male_roles := 3
  let female_roles := 3
  let either_roles := 3
  let male_combinations := Nat.factorial men / Nat.factorial (men - male_roles)
  let female_combinations := Nat.factorial women / Nat.factorial (women - female_roles)
  let remaining_actors := men + women - male_roles - female_roles
  let either_combinations := Nat.factorial remaining_actors / Nat.factorial (remaining_actors - either_roles)
  male_combinations * female_combinations * either_combinations = 5292000 :=
by
  trivial -- "trivial" here used to fill the hole of proof, you can replace it with actual proof if needed

end assign_roles_ways_l144_144911


namespace ellipse_x_intercept_l144_144938

theorem ellipse_x_intercept
  (foci1 foci2 : ℝ × ℝ)
  (x_intercept : ℝ × ℝ)
  (d : ℝ)
  (h_foci1 : foci1 = (0, 3))
  (h_foci2 : foci2 = (4, 0))
  (h_x_intercept : x_intercept = (0, 0))
  (h_d : d = 7)
  : ∃ x : ℝ, (x, 0) ≠ x_intercept ∧ (abs (x - 4) + real.sqrt (x^2 + 9) = 7) ∧ x = 56 / 11 := by
  sorry

end ellipse_x_intercept_l144_144938


namespace count_valid_numbers_is_56_l144_144325

-- Definition of a 3-digit number with three different digits
def is_three_digit_number (n : ℕ) : Prop :=
  (200 ≤ n ∧ n ≤ 999) ∧
  let d1 := n / 100 in
  let d2 := (n / 10) % 10 in
  let d3 := n % 10 in
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

-- Check if the digits of a number are in strictly increasing order
def is_strictly_increasing (n : ℕ) : Prop :=
  let d1 := n / 100 in
  let d2 := (n / 10) % 10 in
  let d3 := n % 10 in
  d1 < d2 ∧ d2 < d3

-- Check if the digits of a number are in strictly decreasing order
def is_strictly_decreasing (n : ℕ) : Prop :=
  let d1 := n / 100 in
  let d2 := (n / 10) % 10 in
  let d3 := n % 10 in
  d1 > d2 ∧ d2 > d3

-- Check if the sum of the digits of a number is even
def is_sum_even (n : ℕ) : Prop :=
  let d1 := n / 100 in
  let d2 := (n / 10) % 10 in
  let d3 := n % 10 in
  (d1 + d2 + d3) % 2 = 0

-- Definition of valid numbers based on given conditions
def valid_numbers (n : ℕ) : Prop :=
  is_three_digit_number n ∧ (is_strictly_increasing n ∨ is_strictly_decreasing n) ∧ is_sum_even n

-- The theorem to prove
theorem count_valid_numbers_is_56 : ∃ (count : ℕ), count = 56 ∧
  count = (Finset.filter valid_numbers (Finset.range 1000)).card :=
by
  sorry

end count_valid_numbers_is_56_l144_144325


namespace sequence_count_length_10_l144_144969

def valid_sequence_count (n : ℕ) : ℕ :=
  if n = 1 then 1 else if n = 2 then 1 else valid_sequence_count (n - 2) + valid_sequence_count (n - 1)

theorem sequence_count_length_10 : valid_sequence_count 10 = 37 :=
by
  sorry

end sequence_count_length_10_l144_144969


namespace solve_equation_a_solve_equation_b_l144_144432

-- Problem a
theorem solve_equation_a (a b x : ℝ) (h₀ : x ≠ a) (h₁ : x ≠ b) (h₂ : a + b ≠ 0) (h₃ : a ≠ 0) (h₄ : b ≠ 0) (h₅ : a ≠ b):
  (x + a) / (x - a) + (x + b) / (x - b) = 2 ↔ x = (2 * a * b) / (a + b) :=
by
  sorry

-- Problem b
theorem solve_equation_b (a b c d x : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : x ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) (h₅ : ab + c ≠ 0):
  c * (d / (a * b) - (a * b) / x) + d = c^2 / x ↔ x = (a * b * c) / d :=
by
  sorry

end solve_equation_a_solve_equation_b_l144_144432


namespace family_b_initial_members_l144_144525

variable (x : ℕ)

theorem family_b_initial_members (h : 6 + (x - 1) + 9 + 12 + 5 + 9 = 48) : x = 8 :=
by
  sorry

end family_b_initial_members_l144_144525


namespace sum_sin_ge_one_l144_144436

theorem sum_sin_ge_one (n : ℕ) (α : Fin n → ℝ)
  (h₁ : ∀ i, 0 ≤ α i ∧ α i ≤ π)
  (h₂ : Odd (∑ i, 1 + cos (α i))) :
  (∑ i, sin (α i)) ≥ 1 :=
by
  sorry

end sum_sin_ge_one_l144_144436


namespace variance_Z_l144_144082

-- Given Conditions:
variables {Ω : Type*} [ProbabilitySpace Ω]
variables (X Y : Ω → ℝ)

-- Define independence and known variances
axiom independent_X_Y : Independent X Y
axiom var_X : variance X = 5
axiom var_Y : variance Y = 9

-- Define the new random variable Z
def Z : Ω → ℝ := λ ω, 2 * X ω - Y ω + 5

-- The goal statement to be proved
theorem variance_Z : variance Z = 29 :=
by sorry

end variance_Z_l144_144082


namespace production_equation_l144_144555

-- Definitions based on the problem conditions
def original_production_rate (x : ℕ) := x
def additional_parts_per_day := 4
def original_days := 20
def actual_days := 15
def extra_parts := 10

-- Prove the equation
theorem production_equation (x : ℕ) :
  original_days * original_production_rate x = actual_days * (original_production_rate x + additional_parts_per_day) - extra_parts :=
by
  simp [original_production_rate, additional_parts_per_day, original_days, actual_days, extra_parts]
  sorry

end production_equation_l144_144555


namespace average_visitors_per_day_in_november_l144_144161
-- Import the entire Mathlib library for necessary definitions and operations.

-- Define the average visitors per different days of the week.
def sunday_visitors := 510
def monday_visitors := 240
def tuesday_visitors := 240
def wednesday_visitors := 300
def thursday_visitors := 300
def friday_visitors := 200
def saturday_visitors := 200

-- Define the counts of each type of day in November.
def sundays := 5
def mondays := 4
def tuesdays := 4
def wednesdays := 4
def thursdays := 4
def fridays := 4
def saturdays := 4

-- Define the number of days in November.
def days_in_november := 30

-- State the theorem to prove the average number of visitors per day.
theorem average_visitors_per_day_in_november : 
  (5 * sunday_visitors + 
   4 * monday_visitors + 
   4 * tuesday_visitors + 
   4 * wednesday_visitors + 
   4 * thursday_visitors + 
   4 * friday_visitors + 
   4 * saturday_visitors) / days_in_november = 282 :=
by
  sorry

end average_visitors_per_day_in_november_l144_144161


namespace isogonal_conjugate_ceva_theorem_l144_144258

open Real EuclideanGeometry

theorem isogonal_conjugate_ceva_theorem
  {A B C P Q Q₁ Q₂ Q₃ D E F : Point}
  (h1: isogonal_conjugate P Q A B C)
  (h2: is_reflection Q Q₁ BC)
  (h3: is_reflection Q Q₂ CA)
  (h4: is_reflection Q Q₃ AB)
  (h5: line_intersect PQ₁ D BC)
  (h6: line_intersect PQ₂ E CA)
  (h7: line_intersect PQ₃ F AB):
  concurrent (line A D) (line B E) (line C F) :=
sorry

end isogonal_conjugate_ceva_theorem_l144_144258


namespace prob_divisor_of_8_l144_144930

theorem prob_divisor_of_8 (n : ℕ) (h : n ∈ {1, 2, 4, 8}) : (∃ p, p = 1 / 2) :=
sorry

end prob_divisor_of_8_l144_144930


namespace sasha_or_maxim_must_be_mistaken_l144_144031

theorem sasha_or_maxim_must_be_mistaken
    (a : ℕ → ℕ → ℕ)
    (non_zero : ∀ i j, a i j ≠ 0)
    (sasha : ∀ i, ∑ j in finset.range 100, a i j % 9 = 0)
    (maxim : ∃! j, ∑ i in finset.range 100, a i j % 9 ≠ 0 ∧ ∀ k ≠ j, ∑ i in finset.range 100, a i k % 9 = 0) :
    false :=
begin
  -- Proof omitted
  sorry
end

end sasha_or_maxim_must_be_mistaken_l144_144031


namespace sphere_diameter_from_cylinder_is_6_84_l144_144904

noncomputable def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

noncomputable def sphere_radius_from_volume (V : ℝ) : ℝ := (3 * V / (4 * π))^(1 / 3)

noncomputable def sphere_diameter_from_cylinder (d_cylinder h_cylinder : ℝ) : ℝ :=
  let r_cylinder := d_cylinder / 2
  let V_cylinder := cylinder_volume r_cylinder h_cylinder
  let r_sphere := sphere_radius_from_volume V_cylinder
  2 * r_sphere

theorem sphere_diameter_from_cylinder_is_6_84 :
  sphere_diameter_from_cylinder 6 6 ≈ 6.84 :=
by
  sorry

end sphere_diameter_from_cylinder_is_6_84_l144_144904


namespace total_boxes_correct_l144_144046

noncomputable def friday_boxes : ℕ := 40

noncomputable def saturday_boxes : ℕ := 2 * friday_boxes - 10

noncomputable def sunday_boxes : ℕ := saturday_boxes / 2

noncomputable def monday_boxes : ℕ := 
  let extra_boxes := (25 * sunday_boxes + 99) / 100 -- (25/100) * sunday_boxes rounded to nearest integer
  sunday_boxes + extra_boxes

noncomputable def total_boxes : ℕ := 
  friday_boxes + saturday_boxes + sunday_boxes + monday_boxes

theorem total_boxes_correct : total_boxes = 189 := by
  sorry

end total_boxes_correct_l144_144046


namespace combined_weight_l144_144337

-- Definitions based on conditions
def jake_weight : ℕ := 156
def sister_weight : ℕ := sorry -- to be defined within the proof 

-- Condition: If Jake loses 20 pounds, he will weigh twice as much as his sister.
def condition (S : ℕ) : Prop := (jake_weight - 20) = 2 * S

-- Theorem to prove the combined weight
theorem combined_weight : (∃ (S : ℕ), condition S) → jake_weight + sorry = 224 :=
by
  intro h
  cases h with S hS
  have h_sister_weight : sister_weight = S := by sorry
  have h_combined : 156 + S = 224 := by sorry
  rw [h_combined]
  sorry

end combined_weight_l144_144337


namespace cost_of_ice_making_l144_144413

theorem cost_of_ice_making :
  let num_cubes := 10 * 16
  let hours := num_cubes / 10
  let ice_maker_cost := hours * 1.5
  let water_needed := num_cubes * 2
  let water_cost := water_needed * 0.1
  let total_cost := ice_maker_cost + water_cost
  total_cost = 56 :=
by
  let num_cubes := 10 * 16
  let hours := num_cubes / 10
  let ice_maker_cost := hours * 1.5
  let water_needed := num_cubes * 2
  let water_cost := water_needed * 0.1
  let total_cost := ice_maker_cost + water_cost
  sorry

end cost_of_ice_making_l144_144413


namespace sum_of_distances_constant_l144_144418

variables {P : Type} [point : Point P]
variables {f : ℕ} -- number of faces
variables {Q V : ℝ} -- area of each face and total volume
variables {h : Fin f → ℝ} -- distances from point P to each face

theorem sum_of_distances_constant (h : Fin f → ℝ) (Q V : ℝ) :
  (∑ i in Finset.range f, h i) = 3 * V / Q :=
sorry

end sum_of_distances_constant_l144_144418


namespace find_n_cubes_l144_144621

theorem find_n_cubes (n : ℕ) (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h1 : 837 + n = y^3) (h2 : 837 - n = x^3) : n = 494 :=
by {
  sorry
}

end find_n_cubes_l144_144621


namespace income_increase_correct_l144_144910

noncomputable def income_increase_percentage (I1 : ℝ) (S1 : ℝ) (E1 : ℝ) (I2 : ℝ) (S2 : ℝ) (E2 : ℝ) (P : ℝ) :=
  S1 = 0.5 * I1 ∧
  S2 = 2 * S1 ∧
  E1 = 0.5 * I1 ∧
  E2 = I2 - S2 ∧
  I2 = I1 * (1 + P / 100) ∧
  E1 + E2 = 2 * E1

theorem income_increase_correct (I1 : ℝ) (S1 : ℝ) (E1 : ℝ) (I2 : ℝ) (S2 : ℝ) (E2 : ℝ) (P : ℝ)
  (h1 : income_increase_percentage I1 S1 E1 I2 S2 E2 P) : P = 50 :=
sorry

end income_increase_correct_l144_144910


namespace complement_of_A_in_I_is_246_l144_144398

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def complement_A_in_I : Set ℕ := {2, 4, 6}

theorem complement_of_A_in_I_is_246 :
  (universal_set \ set_A) = complement_A_in_I :=
  by sorry

end complement_of_A_in_I_is_246_l144_144398


namespace area_of_garden_l144_144194

def length_cm := 50
def width_cm := 70

def length_m := (length_cm : ℝ) / 100
def width_m := (width_cm : ℝ) / 100

theorem area_of_garden :
  length_m * width_m = 0.35 :=
by {
  -- Let's convert the length and width to meters and calculate the area.
  have len_conv: length_m = 0.5 := by norm_num [length_m],
  have wid_conv: width_m = 0.7 := by norm_num [width_m],
  rw [len_conv, wid_conv],
  norm_num,
}

end area_of_garden_l144_144194


namespace smallest_multiple_of_2_3_5_eq_30_l144_144563

theorem smallest_multiple_of_2_3_5_eq_30 : 
  ∃ n : ℕ, (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ n = 30 :=
by
  have ex_smallest := nat.find_spec (nat.exists_least (λ n, (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∘ (nat.find) ));
  use nat.find (nat.exists_least (λ n, n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0));
  split;
  exact ex_smallest
  sorry

end smallest_multiple_of_2_3_5_eq_30_l144_144563


namespace average_marks_passed_l144_144785

noncomputable def total_candidates := 120
noncomputable def total_average_marks := 35
noncomputable def passed_candidates := 100
noncomputable def failed_candidates := total_candidates - passed_candidates
noncomputable def average_marks_failed := 15
noncomputable def total_marks := total_average_marks * total_candidates
noncomputable def total_marks_failed := average_marks_failed * failed_candidates

theorem average_marks_passed :
  ∃ P, P * passed_candidates + total_marks_failed = total_marks ∧ P = 39 := by
  sorry

end average_marks_passed_l144_144785


namespace consecutive_sum_525_l144_144716

theorem consecutive_sum_525 : 
  (∃ (S : Finset ℕ), S.card = 5 ∧ 
    ∀ (k ∈ S), 
      (2 ⊕ Odd k) ∧ 
      (k.divisors = {d ∈ 525.divisors | d ∈ S} ∧ 
      (∃ n : ℕ, 525 = k * n + k * (k-1) / 2))) :=
sorry

end consecutive_sum_525_l144_144716


namespace greatest_common_factor_of_two_digit_palindromes_is_11_l144_144111

-- Define a two-digit palindrome
def is_two_digit_palindrome (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (n / 10 = n % 10)

-- Define the GCD of the set of all such numbers
def GCF_two_digit_palindromes : ℕ :=
  gcd (11 * 1) (gcd (11 * 2) (gcd (11 * 3) (gcd (11 * 4)
  (gcd (11 * 5) (gcd (11 * 6) (gcd (11 * 7) (gcd (11 * 8) (11 * 9))))))))

-- The statement to prove
theorem greatest_common_factor_of_two_digit_palindromes_is_11 :
  GCF_two_digit_palindromes = 11 :=
by
  sorry

end greatest_common_factor_of_two_digit_palindromes_is_11_l144_144111


namespace distance_ridden_l144_144574

theorem distance_ridden (speed : ℝ) (time : ℝ) (distance : ℝ) (h1 : speed = 50) (h2 : time = 18) : distance = speed * time := 
by
  rw [h1, h2]
  show 50 * 18 = 900
  exact sorry

end distance_ridden_l144_144574


namespace conjugate_z2_l144_144254

-- Given definitions according to the problem conditions
def z1 : Complex := 3 + 4 * Complex.i
  
-- Definition of rotating a complex number 90 degrees counterclockwise
def rotate_90 (z : Complex) : Complex :=
  Complex.i * z

-- Definition: z2 is the result of rotating z1
def z2 : Complex := rotate_90 z1

-- The required proof in Lean
theorem conjugate_z2 : Complex.conj z2 = -4 - 3 * Complex.i := by
  sorry

end conjugate_z2_l144_144254


namespace eleven_C_equals_308_l144_144074

def is_second_largest_factor (n k : ℕ) : Prop :=
  ∃ (factors : List ℕ), 
    factors = List.filter (λ m, n % m = 0) (List.range (n+1)) ∧ 
    (factors.length > 2 ∧ factors.get? (factors.length - 2) = some k)

def S : Set ℕ :=
  { n | is_second_largest_factor n (n - 6) }

theorem eleven_C_equals_308 : 
  11 * (S.to_finset.sum id) = 308 :=
by
  sorry

end eleven_C_equals_308_l144_144074


namespace simplify_expression_l144_144033

variable (a b : ℝ)

theorem simplify_expression : 
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a := 
  sorry

end simplify_expression_l144_144033


namespace enclosed_triangle_area_l144_144924

theorem enclosed_triangle_area (ABC : Triangle) (area_ABC : ABC.area = 1)
  (A' B' C' : Point)
  (hA' : on_line A' ABC.BC ∧ distance ABC.B A' = 2 * distance A' ABC.C)
  (hB' : on_line B' ABC.CA ∧ distance ABC.C B' = 2 * distance B' ABC.A)
  (hC' : on_line C' ABC.AB ∧ distance ABC.A C' = 2 * distance C' ABC.B) :
  area (enclosed_triangle (line_through ABC.A A') (line_through ABC.B B') (line_through ABC.C C')) = 1 / 7 :=
sorry

end enclosed_triangle_area_l144_144924


namespace make_all_ones_l144_144883

def transform_step (numbers : List ℕ) (a : ℕ) : List ℕ :=
  let N := numbers.foldl lcm 1
  numbers.map (λ x => if x = a then N / a else x)

theorem make_all_ones (n : ℕ) (numbers : List ℕ) (h1 : n ≥ 2) (h2 : ∀ x ∈ numbers, x > 0) :
  ∃ steps : ℕ, (∀ (i : ℕ) (l : List ℕ), i ≤ steps → 
    (l = (List.range i).foldl (λ acc _, transform_step acc (acc.headD 1)) numbers → l = List.repeat 1 n)) :=
sorry

end make_all_ones_l144_144883


namespace modulus_of_z_l144_144340

-- Define the complex number z
def z : ℂ := (3 + 4 * Complex.i) / (1 - Complex.i)

-- State the theorem for the modulus of z
theorem modulus_of_z : Complex.abs z = 5 * Real.sqrt 2 / 2 := by
  sorry

end modulus_of_z_l144_144340


namespace sara_change_l144_144128

-- Define the costs of individual items
def cost_book_1 : ℝ := 5.5
def cost_book_2 : ℝ := 6.5
def cost_notebook : ℝ := 3
def cost_bookmarks : ℝ := 2

-- Define the discounts and taxes
def discount_books : ℝ := 0.10
def sales_tax : ℝ := 0.05

-- Define the payment amount
def amount_given : ℝ := 20

-- Calculate the total cost, discount, and final amount
def discounted_book_cost := (cost_book_1 + cost_book_2) * (1 - discount_books)
def subtotal := discounted_book_cost + cost_notebook + cost_bookmarks
def total_with_tax := subtotal * (1 + sales_tax)
def change := amount_given - total_with_tax

-- State the theorem
theorem sara_change : change = 3.41 := by
  sorry

end sara_change_l144_144128


namespace equal_angles_MKO_MLO_l144_144841

-- Definitions and conditions:
variables {O K L A M P Q : Point}
variable {Γ : Circle}
variable (on_circle_K : on_circle K Γ)
variable (on_circle_L : on_circle L Γ)
variable (center_O : center O Γ)
variable (line_KL : collinear K L A)
variable (A_tangent_P : tangent_from A Γ P)
variable (A_tangent_Q : tangent_from A Γ Q)
variable (midpoint_M : midpoint M P Q)

-- Statement to prove:
theorem equal_angles_MKO_MLO :
  ∠(M, K, O) = ∠(M, L, O) :=
sorry

end equal_angles_MKO_MLO_l144_144841


namespace parallel_KL_AC_l144_144719

noncomputable theory
open_locale classical

variables {A B C A1 B1 C1 A2 C2 K L : Type*}
variables [triangle A B C] [height A1 ht1] [height B1 ht2] [height C1 ht3]

/-- In the acute-angled triangle ABC, given the conditions:
- The heights from A, B, and C meet at A1, B1, and C1 respectively.
- Points A2 and C2 are on segment A1C1.
- B1A2 is bisected by height CC1, intersects height AA1 at K.
- B1C2 is bisected by height AA1, intersects height CC1 at L.
To prove that KL is parallel to AC. -/
theorem parallel_KL_AC : 
  ∀ (A B C A1 B1 C1 A2 C2 K L : Type*)
  (ht1 : height A1) (ht2 : height B1) (ht3 : height C1)
  (h1: is_on_height A A1) (h2: is_on_height B B1) (h3: is_on_height C C1) 
  (h4: is_on_segment A2 A1 C1) (h5: is_on_segment C2 A1 C1)
  (h6: bisects_segment CC1 B1 A2) (h7: intersects_at_height AA1 K B1 A2)
  (h8: bisects_segment AA1 B1 C2) (h9: intersects_at_height CC1 L B1 C2),
  parallel KL AC :=
sorry

end parallel_KL_AC_l144_144719


namespace parabola_slope_l144_144793

-- Definitions according to the problem statement
def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0) -- Coordinates of focus for y^2 = 4x

def first_quadrant (Q : ℝ × ℝ) : Prop := 0 < Q.1 ∧ 0 < Q.2

def vector_relation (P Q F : ℝ × ℝ) : Prop := 
  (3 * (P.1 - F.1), 3 * (P.2 - F.2)) = (Q.1 - F.1, Q.2 - F.2)

def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- The theorem statement
theorem parabola_slope (P Q F : ℝ × ℝ) (hF : focus F) 
    (hP : parabola_eq P.1 P.2)
    (hQ : parabola_eq Q.1 Q.2)
    (hQ1 : first_quadrant Q)
    (hVR : vector_relation P Q F)
    (hPQ : Q.1 ≠ P.1) : 
    slope P Q = sqrt 3 := 
sorry

end parabola_slope_l144_144793


namespace cricket_average_increase_l144_144903

theorem cricket_average_increase :
  ∀ T : ℕ, (T + 134) / 22 = 60.5 →
  ∃ I : ℝ, I = 3.5 :=
by
  sorry

end cricket_average_increase_l144_144903


namespace circle_with_diameter_AB_equation_line_MN_fixed_point_l144_144294

noncomputable def ellipse_Equation (x y : ℝ) : Prop := 
  (y^2 / 12) + (x^2 / 4) = 1

noncomputable def point_A : CLLocationCoordinate2D :=
  ⟨2, 0⟩

noncomputable def point_P : CLLocationCoordinate2D :=
  ⟨0, -2⟩

theorem circle_with_diameter_AB_equation :
  ∃ x y, (point_P (0, -2)) ∧ (point_A (2,0)) ∧  
  (x - 2) * (x + 1) + (y - 0) * (y + 3) = 0 →
  (x^2 + y^2 - x + 3y - 2 = 0) := 
sorry

theorem line_MN_fixed_point :
  ∀ k (x_C y_C x_D y_D : ℝ),
  ellipse_Equation x_C y_C ∧ ellipse_Equation x_D y_D ∧
  (x - (4 * k)/(3 + k^2)) - (y) + 10 = 0 ∨ (x - 3 * y - 10 = 0) →
  (∃ m n, (m = 0) ∧ (n = -10/3)) → 
sorry

end circle_with_diameter_AB_equation_line_MN_fixed_point_l144_144294


namespace value_of_n_l144_144433

variable (s P k : Real)

noncomputable def n : Real := 2 * Real.log(s / P) / Real.log(1 + k)

theorem value_of_n (h : P = s / Real.sqrt((1 + k) ^ n s P k)) :
  n s P k = 2 * Real.log(s / P) / Real.log(1 + k) := by
  sorry

end value_of_n_l144_144433


namespace smallest_number_l144_144866

/--
  The smallest number which when increased by 3 is divisible by 18, 70, 100, and 21 is 6297.
-/
theorem smallest_number (n : ℕ) :
  (∃ m : ℕ, n = m * 6300 - 3) ∧
  ∀ k : ℕ, (∃ l : ℕ, k = l * 6300 - 3) → k ≥ n :=
begin
  sorry,
end

end smallest_number_l144_144866


namespace smallest_non_palindrome_power_of_13_is_13_l144_144631

def is_palindrome (n : ℕ) : Prop :=
  n.toString = n.toString.reverse

def is_power_of_13 (x : ℕ) : Prop :=
  ∃ n : ℕ, x = 13 ^ n

theorem smallest_non_palindrome_power_of_13_is_13 :
  ∀ x : ℕ, is_power_of_13 x ∧ ¬ is_palindrome x → x ≥ 13 :=
by
  sorry

end smallest_non_palindrome_power_of_13_is_13_l144_144631


namespace correct_area_ratio_l144_144912

noncomputable def area_ratio (P : ℝ) : ℝ :=
  let x := P / 6 
  let length := P / 3
  let diagonal := (P * Real.sqrt 5) / 6
  let r := diagonal / 2
  let A := (5 * (P^2) * Real.pi) / 144
  let s := P / 5
  let R := P / (10 * Real.sin (36 * Real.pi / 180))
  let B := (P^2 * Real.pi) / (100 * (Real.sin (36 * Real.pi / 180))^2)
  A / B

theorem correct_area_ratio (P : ℝ) : area_ratio P = 500 * (Real.sin (36 * Real.pi / 180))^2 / 144 := 
  sorry

end correct_area_ratio_l144_144912


namespace parallel_planes_of_nonintersecting_lines_l144_144656

variables (m n : Set Point) (α β : Set Point)
-- Assuming m and n are lines
-- and α and β are planes
variables (h: m ∩ n = ∅)
variables (h' : ∀ p ∈ α, ∀ q ∈ β, p ≠ q)
variables (hmn : ∀ P Q ∈ m, P ≠ Q → Line_through P Q)
variables (hm : ∀ P Q ∈ α, P ≠ Q → Plane_through P Q)
variables (hn : ∀ P Q ∈ β, P ≠ Q → Plane_through P Q)

-- Given conditions
variables (h1 : Parallel m n)
variables (h2 : Perpendicular m α)
variables (h3 : Perpendicular n β)

-- We need to prove α ∥ β.
theorem parallel_planes_of_nonintersecting_lines (m n α β : Set Point)
  (h : m ∩ n = ∅) (h' : ∀ p ∈ α, ∀ q ∈ β, p ≠ q)
  (hmn : ∀ P Q ∈ m, P ≠ Q → Line_through P Q)
  (hm : ∀ P Q ∈ α, P ≠ Q → Plane_through P Q)
  (hn : ∀ P Q ∈ β, P ≠ Q → Plane_through P Q)
  (h1 : Parallel m n) (h2 : Perpendicular m α) (h3 : Perpendicular n β)
  : Parallel α β := sorry

end parallel_planes_of_nonintersecting_lines_l144_144656


namespace solution_is_D_l144_144119

-- Definitions of the equations
def eqA (x : ℝ) := 3 * x + 6 = 0
def eqB (x : ℝ) := 2 * x + 4 = 0
def eqC (x : ℝ) := (1 / 2) * x = -4
def eqD (x : ℝ) := 2 * x - 4 = 0

-- Theorem stating that only eqD has a solution x = 2
theorem solution_is_D : 
  ¬ eqA 2 ∧ ¬ eqB 2 ∧ ¬ eqC 2 ∧ eqD 2 := 
by
  sorry

end solution_is_D_l144_144119


namespace locus_of_point_inside_trapezoid_l144_144624

variable (A B C D X : Point)
variable (ABCD : Trapezoid)
variable (H : parallel (BC ABCD) (AD ABCD))
variable (P Q : Point)
variable (P_is_midpoint : midpoint P B C)
variable (Q_is_midpoint : midpoint Q A D)

theorem locus_of_point_inside_trapezoid :
  (area X A B = area X C D) → lies_on_segment X P Q :=
sorry

end locus_of_point_inside_trapezoid_l144_144624


namespace smallest_three_digit_geometric_sequence_l144_144502

theorem smallest_three_digit_geometric_sequence : ∃ n : ℕ, n = 124 ∧ (∀ (a r : ℕ), a = 1 ∧ r * a < 10 ∧ r^2 * a < 100 → digits n = [1, r, r^2] ∧ (digits n).nodup) := by
  sorry

end smallest_three_digit_geometric_sequence_l144_144502


namespace find_quotient_of_sum_of_squares_mod_17_l144_144780

theorem find_quotient_of_sum_of_squares_mod_17 :
  let squares_mod := {1, 2, 4, 8, 9, 13, 15, 16},
      m := (∑ x in squares_mod, x) in
  m / 17 = 4 :=
by
  let squares_mod := {1, 2, 4, 8, 9, 13, 15, 16}
  let m := (∑ x in squares_mod, x)
  have h : m = 68 := sorry
  rw [h, Nat.div_eq_of_lt]
  exact nat.lt_succ_self 68
  norm_num
  exact nat.le_add_right (∑ x in squares_mod, x) 0 68

end find_quotient_of_sum_of_squares_mod_17_l144_144780


namespace rex_lesson_schedule_l144_144026

-- Define the total lessons and weeks
def total_lessons : ℕ := 40
def weeks_completed : ℕ := 6
def weeks_remaining : ℕ := 4

-- Define the proof statement
theorem rex_lesson_schedule : (weeks_completed + weeks_remaining) * 4 = total_lessons := by
  -- Proof placeholder, to be filled in 
  sorry

end rex_lesson_schedule_l144_144026


namespace ellipse_x_intercept_l144_144940

theorem ellipse_x_intercept
  (foci1 foci2 : ℝ × ℝ)
  (x_intercept : ℝ × ℝ)
  (d : ℝ)
  (h_foci1 : foci1 = (0, 3))
  (h_foci2 : foci2 = (4, 0))
  (h_x_intercept : x_intercept = (0, 0))
  (h_d : d = 7)
  : ∃ x : ℝ, (x, 0) ≠ x_intercept ∧ (abs (x - 4) + real.sqrt (x^2 + 9) = 7) ∧ x = 56 / 11 := by
  sorry

end ellipse_x_intercept_l144_144940


namespace sum_of_reciprocal_squares_leq_reciprocal_product_square_l144_144820

theorem sum_of_reciprocal_squares_leq_reciprocal_product_square (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 3) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a^2 * b^2 * c^2 * d^2) :=
sorry

end sum_of_reciprocal_squares_leq_reciprocal_product_square_l144_144820


namespace speed_of_man_in_still_water_l144_144134

def upstream_speed : ℝ := 32
def downstream_speed : ℝ := 48

theorem speed_of_man_in_still_water :
  (upstream_speed + (downstream_speed - upstream_speed) / 2) = 40 :=
by
  sorry

end speed_of_man_in_still_water_l144_144134


namespace find_x2_l144_144837

-- Define the points A and B on the parabola y = x^2
def A : Point := ⟨1, 1⟩
def B (x2 : ℝ) : Point := ⟨x2, x2^2⟩

-- Define the equations of the tangents at A and B
def tangent_at_A (x : ℝ) : ℝ := 2 * x - 1
def tangent_at_B (x2 x : ℝ) : ℝ := 2 * x2 * x - x2^2

-- Function to find the intersection point of two tangents
def intersection_point (x2 : ℝ) : Point :=
  let xc := (x2^2 - 1) / (2 - 2 * x2)
  let yc := 2 * xc - 1
  ⟨xc, yc⟩

-- Dot product condition
def dot_product_condition (x2 : ℝ) : Prop :=
  let C := intersection_point x2
  let AC := (C.x - 1, C.y - 1)
  let BC := (C.x - x2, C.y - x2^2)
  AC.1 * BC.1 + AC.2 * BC.2 = 0

-- Main theorem stating the value of x2
theorem find_x2 : dot_product_condition (-1 / 4) := 
sorry

end find_x2_l144_144837


namespace subsets_bound_l144_144006

variable {n : ℕ} (S : Finset (Fin n)) (m : ℕ) (A : ℕ → Finset (Fin n))

theorem subsets_bound {n : ℕ} (hn : n ≥ 2) (hA : ∀ i, 1 ≤ i ∧ i ≤ m → (A i).card ≥ 2)
  (h_inter : ∀ i j k, 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → 1 ≤ k ∧ k ≤ m →
    (A i) ∩ (A j) ≠ ∅ ∧ (A i) ∩ (A k) ≠ ∅ ∧ (A j) ∩ (A k) ≠ ∅ → (A i) ∩ (A j) ∩ (A k) ≠ ∅) :
  m ≤ 2 ^ (n - 1) - 1 := 
sorry

end subsets_bound_l144_144006


namespace copper_tin_ratio_l144_144364

theorem copper_tin_ratio 
    (w1 w2 w_new : ℝ) 
    (r1_copper r1_tin r2_copper r2_tin : ℝ) 
    (r_new_copper r_new_tin : ℝ)
    (pure_copper : ℝ)
    (h1 : w1 = 10)
    (h2 : w2 = 16)
    (h3 : r1_copper = 4 / 5 * w1)
    (h4 : r1_tin = 1 / 5 * w1)
    (h5 : r2_copper = 1 / 4 * w2)
    (h6 : r2_tin = 3 / 4 * w2)
    (h7 : r_new_copper = r1_copper + r2_copper + pure_copper)
    (h8 : r_new_tin = r1_tin + r2_tin)
    (h9 : w_new = 35)
    (h10 : r_new_copper + r_new_tin + pure_copper = w_new)
    (h11 : pure_copper = 9) :
    r_new_copper / r_new_tin = 3 / 2 :=
by
  sorry

end copper_tin_ratio_l144_144364


namespace AT_parallel_EF_l144_144068

-- Define points A, B, and C.
variables {A B C D E F M T : Type*}

-- Define that we have a triangle ABC with an incircle touching sides BC, AC, and AB at D, E, and F respectively.
variables [IncircleTouchesSides ABC BC AC AB D E F]

-- Define M as the midpoint of EF.
variable [Midpoint EF M]

-- Define T as the intersection of ray DE and ray BM.
variable [Intersection (ray DE) (ray BM) T]

-- State that AT is parallel to EF.
theorem AT_parallel_EF : Parallel AT EF :=
sorry

end AT_parallel_EF_l144_144068


namespace savings_for_23_students_is_30_yuan_l144_144993

-- Define the number of students
def number_of_students : ℕ := 23

-- Define the price per ticket in yuan
def price_per_ticket : ℕ := 10

-- Define the discount rate for the group ticket
def discount_rate : ℝ := 0.8

-- Define the group size that is eligible for the discount
def group_size_discount : ℕ := 25

-- Define the cost without ticket discount
def cost_without_discount : ℕ := number_of_students * price_per_ticket

-- Define the cost with the group ticket discount
def cost_with_discount : ℝ := price_per_ticket * discount_rate * group_size_discount

-- Define the expected amount saved by using the group discount
def expected_savings : ℝ := cost_without_discount - cost_with_discount

-- Theorem statement that the expected_savings is 30 yuan
theorem savings_for_23_students_is_30_yuan :
  expected_savings = 30 := 
sorry

end savings_for_23_students_is_30_yuan_l144_144993


namespace abs_eq_abs_implies_l144_144870

theorem abs_eq_abs_implies (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 := 
sorry

end abs_eq_abs_implies_l144_144870


namespace proof_BZ_YZ_AY_l144_144535

-- Definition of the isosceles right triangle
variables (A B C X Y Z : Type) [InnerProductSpace ℝ : Type]
variable [HasAngle A B C : Type]
variable [HasAngle C A B : Type]
variables (angle_90 : Angle C A B = 90)
variables (isosceles : (A - C) = (B - C))
variable (on_line_seg : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (1 - t) • A + t • C)
variable (foot_Y : Perpendicular (Line A X) (Line B Y))
variable (foot_Z : Perpendicular (Line C X) (Line B Z))

theorem proof_BZ_YZ_AY :
  ∃ (Y Z : Point), IsFoot Y A (Line B X) ∧ IsFoot Z C (Line B X) ∧ BZ = YZ + AY :=
by
  sorry

end proof_BZ_YZ_AY_l144_144535


namespace chess_tournament_num_players_l144_144348

theorem chess_tournament_num_players (n : ℕ) :
  (∀ k, k ≠ n → exists m, m ≠ n ∧ (k = m)) ∧ 
  ((1 / 2 * (n - 1)) + (1 / 4 * (n - 1))) = (1 / 13 * ((1 / 2 * n * (n - 1)) - ((1 / 2 * (n - 1)) + (1 / 4 * (n - 1))))) →
  n = 21 :=
by
  sorry

end chess_tournament_num_players_l144_144348


namespace average_math_test_score_l144_144959

theorem average_math_test_score :
    let june_score := 97
    let patty_score := 85
    let josh_score := 100
    let henry_score := 94
    let num_children := 4
    let total_score := june_score + patty_score + josh_score + henry_score
    total_score / num_children = 94 := by
  sorry

end average_math_test_score_l144_144959


namespace river_width_l144_144150

theorem river_width (boat_max_speed : ℝ) (river_current_speed : ℝ) (time_to_cross : ℝ) (width : ℝ) :
  boat_max_speed = 4 ∧ river_current_speed = 3 ∧ time_to_cross = 2 ∧ width = 8 → 
  width = boat_max_speed * time_to_cross := by
  intros h
  cases h
  sorry

end river_width_l144_144150


namespace gcd_factorials_l144_144845

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  -- Proof steps would go here
  sorry

end gcd_factorials_l144_144845


namespace maximum_n_l144_144634

noncomputable def a1 : ℝ := sorry -- define a1 solving a_5 equations
noncomputable def q : ℝ := sorry -- define q solving a_5 and a_6 + a_7 equations
noncomputable def sn (n : ℕ) : ℝ := (a1 * (1 - q^n)) / (1 - q)  -- S_n of geometric series with a1 and q
noncomputable def pin (n : ℕ) : ℝ := (a1 * (q^((1 + n) * n / 2 - (11 * n) / 2 + 19 / 2)))  -- Pi solely in terms of n, a1, and q

theorem maximum_n (n : ℕ) (h1 : (a1 : ℝ) > 0) (h2 : q > 0) (h3 : q ≠ 1)
(h4 : a1 * q^4 = 1 / 4) (h5 : a1 * q^5 + a1 * q^6 = 3 / 2) :
  ∃ n : ℕ, sn n > pin n ∧ ∀ m : ℕ, m > 13 → sn m ≤ pin m := sorry

end maximum_n_l144_144634


namespace find_20th_integer_l144_144293

theorem find_20th_integer : 
  let digits := [5, 6, 7, 8],
      perms := List.permutations digits,
      reorder_last_two (l : List Nat) : List Nat := 
        if l.head! = 7 ∨ l.head! = 8 then l.take 2 ++ l.drop 2.reverse else l,
      all_perms := perms.map reorder_last_two,
      sorted_perms := all_perms.sort (≤) in
  sorted_perms.ilast! = [7, 8, 6, 5] := 
by
  sorry

end find_20th_integer_l144_144293


namespace a_eq_0_necessary_not_sufficient_purely_imaginary_l144_144750

open Complex

-- Define the statement
theorem a_eq_0_necessary_not_sufficient_purely_imaginary (a b : ℝ) : 
  (a = 0) → (a + b * Complex.I).im = b ∧ (a + b * Complex.I).re = 0 := 
begin
  -- Proof goes here  
  sorry  
end

end a_eq_0_necessary_not_sufficient_purely_imaginary_l144_144750


namespace probability_digit_three_l144_144408

theorem probability_digit_three (h : ∃ d1 d2 d3 : ℕ, d1 = 3 ∧ d2 = 7 ∧ d3 = 5 ∧ 0.375 = d1 / 10^1 + d2 / 10^2 + d3 / 10^3): 
  (Nat.succ 0 : ℚ) / 3 = 1 / 3 :=
by
  sorry

end probability_digit_three_l144_144408


namespace combined_time_l144_144422

def time_pulsar : ℕ := 10
def time_polly : ℕ := 3 * time_pulsar
def time_petra : ℕ := time_polly / 6

theorem combined_time : time_pulsar + time_polly + time_petra = 45 := 
by 
  -- proof steps will go here
  sorry

end combined_time_l144_144422


namespace triangle_area_l144_144438

theorem triangle_area :
  ∀ (a b c : ℝ), 
  (∃ k : ℝ, a = 2 * k ∧ b = 3 * k ∧ c = k * √7 ∧ (a + b + c = 10 + 2 * √7)) →
  (S = √((1 / 4) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2))) →
  S = 6 * √ 3 :=
begin
  intros a b c harea h,
  obtain ⟨k, ha, hb, hc, hperimeter⟩ := harea,
  sorry
end

end triangle_area_l144_144438


namespace general_term_of_sequence_l144_144271

theorem general_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hSn : ∀ n, S n = 3 * n^2 - n + 1) :
  (∀ n, a n = if n = 1 then 3 else 6 * n - 4) :=
by
  sorry

end general_term_of_sequence_l144_144271


namespace dual_colored_numbers_l144_144345

theorem dual_colored_numbers (table : Matrix (Fin 10) (Fin 20) ℕ)
  (distinct_numbers : ∀ (i j k l : Fin 10) (m n : Fin 20), 
    (i ≠ k ∨ m ≠ n) → table i m ≠ table k n)
  (row_red : ∀ (i : Fin 10), ∃ r₁ r₂ : Fin 20, r₁ ≠ r₂ ∧ 
    (∀ (j : Fin 20), table i j ≤ table i r₁ ∨ table i j ≤ table i r₂))
  (col_blue : ∀ (j : Fin 20), ∃ b₁ b₂ : Fin 10, b₁ ≠ b₂ ∧ 
    (∀ (i : Fin 10), table i j ≤ table b₁ j ∨ table i j ≤ table b₂ j)) : 
  ∃ i₁ i₂ i₃ : Fin 10, ∃ j₁ j₂ j₃ : Fin 20, 
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₂ ≠ i₃ ∧ j₁ ≠ j₂ ∧ j₁ ≠ j₃ ∧ j₂ ≠ j₃ ∧ 
    ((table i₁ j₁ ≤ table i₁ j₂ ∨ table i₁ j₁ ≤ table i₃ j₂) ∧ 
     (table i₂ j₂ ≤ table i₂ j₁ ∨ table i₂ j₂ ≤ table i₃ j₁) ∧ 
     (table i₃ j₃ ≤ table i₃ j₁ ∨ table i₃ j₃ ≤ table i₂ j₁)) := 
  sorry

end dual_colored_numbers_l144_144345


namespace complex_pow_identity_l144_144962

noncomputable def cos_30 : ℝ := real.cos (30 * real.pi / 180)
noncomputable def sin_30 : ℝ := real.sin (30 * real.pi / 180)

theorem complex_pow_identity :
  (2 * (cos_30 + complex.i * sin_30))^4 = -8 + 8 * complex.i * real.sqrt 3 :=
by {
  have h_cos_30 : cos_30 = real.sqrt 3 / 2 := by sorry,
  have h_sin_30 : sin_30 = 1 / 2 := by sorry,
  sorry
}

end complex_pow_identity_l144_144962


namespace total_distance_run_l144_144053

theorem total_distance_run (length : ℕ) (width : ℕ) (laps : ℕ) (h_length : length = 100) (h_width : width = 50) (h_laps : laps = 6) : 
  let perimeter := 2 * length + 2 * width in
  let distance := laps * perimeter in
  distance = 1800 :=
by
  sorry

end total_distance_run_l144_144053


namespace find_number_l144_144564

theorem find_number (x : ℝ) (h : x / 100 = 31.76 + 0.28) : x = 3204 := 
  sorry

end find_number_l144_144564


namespace no_2021_numbers_exist_l144_144987

theorem no_2021_numbers_exist :
  ¬ ∃ (a : Fin 2021 → ℝ), (∀ i, |a i| < 1) ∧ (Finset.univ.sum (λ i, |a i|) - |Finset.univ.sum (λ i, a i)| = 2020) :=
by
  sorry

end no_2021_numbers_exist_l144_144987


namespace area_of_trapezoid_l144_144048

theorem area_of_trapezoid (S1 S2 : ℝ) (h1 : 0 ≤ S1) (h2 : 0 ≤ S2) :
  let area := (Real.sqrt S1 + Real.sqrt S2) ^ 2
  in area = S1 + S2 + 2 * Real.sqrt (S1 * S2) :=
by
  sorry

end area_of_trapezoid_l144_144048


namespace log_base4_of_1_div_64_l144_144219

theorem log_base4_of_1_div_64 : log 4 (1 / 64) = -3 := by
  sorry

end log_base4_of_1_div_64_l144_144219


namespace XAXAXA_divisible_by_seven_l144_144024

theorem XAXAXA_divisible_by_seven (X A : ℕ) (hX : X < 10) (hA : A < 10) : 
  (101010 * X + 10101 * A) % 7 = 0 := 
by 
  sorry

end XAXAXA_divisible_by_seven_l144_144024


namespace total_height_of_sandcastles_l144_144952

structure Sandcastle :=
  (feet : Nat)
  (fraction_num : Nat)
  (fraction_den : Nat)

def janet : Sandcastle := ⟨3, 5, 6⟩
def sister : Sandcastle := ⟨2, 7, 12⟩
def tom : Sandcastle := ⟨1, 11, 20⟩
def lucy : Sandcastle := ⟨2, 13, 24⟩

-- a function to convert a Sandcastle to a common denominator
def convert_to_common_denominator (s : Sandcastle) : Sandcastle :=
  let common_den := 120 -- LCM of 6, 12, 20, 24
  ⟨s.feet, (s.fraction_num * (common_den / s.fraction_den)), common_den⟩

-- Definition of heights after conversion to common denominator
def janet_converted : Sandcastle := convert_to_common_denominator janet
def sister_converted : Sandcastle := convert_to_common_denominator sister
def tom_converted : Sandcastle := convert_to_common_denominator tom
def lucy_converted : Sandcastle := convert_to_common_denominator lucy

-- Proof problem
def total_height_proof_statement : Sandcastle :=
  let total_feet := janet.feet + sister.feet + tom.feet + lucy.feet
  let total_numerator := janet_converted.fraction_num + sister_converted.fraction_num + tom_converted.fraction_num + lucy_converted.fraction_num
  let total_denominator := 120
  ⟨total_feet + (total_numerator / total_denominator), total_numerator % total_denominator, total_denominator⟩

theorem total_height_of_sandcastles :
  total_height_proof_statement = ⟨10, 61, 120⟩ :=
by
  sorry

end total_height_of_sandcastles_l144_144952


namespace weng_hourly_rate_l144_144842

theorem weng_hourly_rate (minutes_worked : ℝ) (earnings : ℝ) (fraction_of_hour : ℝ) 
  (conversion_rate : ℝ) (hourly_rate : ℝ) : 
  minutes_worked = 50 → earnings = 10 → 
  fraction_of_hour = minutes_worked / conversion_rate → 
  conversion_rate = 60 → 
  hourly_rate = earnings / fraction_of_hour → 
  hourly_rate = 12 := by
    sorry

end weng_hourly_rate_l144_144842


namespace base6_addition_problem_l144_144997

theorem base6_addition_problem (X Y : ℕ) (h1 : 3 * 6^2 + X * 6 + Y + 24 = 6 * 6^2 + 1 * 6 + X) :
  X = 5 ∧ Y = 1 ∧ X + Y = 6 := by
  sorry

end base6_addition_problem_l144_144997


namespace stephanie_total_spent_l144_144036

def price_per_orange : ℕ → ℝ
| 1 := 0.50
| 2 := 0.60
| 3 := 0.55
| 4 := 0.65
| 5 := 0.70
| 6 := 0.55
| 7 := 0.50
| 8 := 0.60
| _ := 0 -- default value for safety, although this should never be used

def oranges_bought_each_visit : ℕ := 2

def total_cost : ℝ :=
  let costs := List.map (λ n => oranges_bought_each_visit * price_per_orange n) [1, 2, 3, 4, 5, 6, 7, 8]
  costs.sum

theorem stephanie_total_spent : total_cost = 9.30 :=
  sorry

end stephanie_total_spent_l144_144036


namespace Riley_fewer_pairs_l144_144216

-- Define the conditions
def Ellie_pairs : ℕ := 8
def Total_pairs : ℕ := 13

-- Prove the statement
theorem Riley_fewer_pairs : (Total_pairs - Ellie_pairs) - Ellie_pairs = 3 :=
by
  -- Skip the proof
  sorry

end Riley_fewer_pairs_l144_144216


namespace determine_r_l144_144005

-- Non-zero real numbers a_i
variables (n : ℕ) (a : Fin n → ℝ)
-- Assume a_i are non-zero
variables (h_a : ∀ i, a i ≠ 0)
-- Variables r_i
variables (r : Fin n → ℝ)

-- Define the condition of the inequality holding for all x_i
def inequality_holds := ∀ (x : Fin n → ℝ), 
  (∑ i, r i * (x i - a i)) ≤ (Real.sqrt (∑ i, (x i)^2) - Real.sqrt (∑ i, (a i)^2))

-- The theorem to prove that r_i = a_i / sqrt(a_1^2 + a_2^2 + ... + a_n^2)
theorem determine_r (h_ineq : inequality_holds n a h_a r) : ∀ i, r i = a i / Real.sqrt (∑ i, (a i)^2) := 
sorry

end determine_r_l144_144005


namespace number_of_proper_subsets_l144_144810

theorem number_of_proper_subsets (M : Finset ℕ) (hM : M.card = 3) : (2 ^ M.card - 1) = 7 :=
by
  sorry

end number_of_proper_subsets_l144_144810


namespace ellipse_x_intercept_other_l144_144943

noncomputable def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 3), (4, 0))
noncomputable def sum_of_distances : ℝ := 7
noncomputable def first_intercept : (ℝ × ℝ) := (0, 0)

theorem ellipse_x_intercept_other 
  (foci : (ℝ × ℝ) × (ℝ × ℝ))
  (sum_of_distances : ℝ)
  (first_intercept : (ℝ × ℝ))
  (hx : foci = ((0, 3), (4, 0)))
  (d_sum : sum_of_distances = 7)
  (intercept : first_intercept = (0, 0)) :
  ∃ (x : ℝ), x > 0 ∧ ((x, 0) = (56 / 11, 0)) := 
sorry

end ellipse_x_intercept_other_l144_144943


namespace parabola_vertex_l144_144094

def parabola (x : ℝ) : ℝ := -(x - 1)^2 - 2

theorem parabola_vertex :
  ∃ v : ℝ × ℝ, v = (1, -2) ∧ (parabola v.1 = v.2) :=
by
  use (1, -2)
  split
  { refl }
  { sorry }

end parabola_vertex_l144_144094


namespace volume_of_rectangular_solid_l144_144722

theorem volume_of_rectangular_solid : 
  let l := 100 -- length in cm
  let w := 20  -- width in cm
  let h := 50  -- height in cm
  let V := l * w * h
  V = 100000 :=
by
  rfl

end volume_of_rectangular_solid_l144_144722


namespace major_premise_wrong_l144_144466

-- Given conditions:
def major_premise : Prop := ∃ n : ℤ, n ∈ ℕ
def minor_premise : -2 ∈ ℤ

-- Problem statement:
theorem major_premise_wrong : major_premise → False :=
by sorry

end major_premise_wrong_l144_144466


namespace number_of_four_digit_numbers_l144_144319

/-- The number of four-digit numbers greater than 3999 where the product of the middle two digits exceeds 8 is 2400. -/
def four_digit_numbers_with_product_exceeding_8 : Nat :=
  let first_digit_options := 6  -- Choices are 4,5,6,7,8,9
  let last_digit_options := 10  -- Choices are 0,1,...,9
  let middle_digit_options := 40  -- Valid pairs counted from the solution
  first_digit_options * middle_digit_options * last_digit_options

theorem number_of_four_digit_numbers : four_digit_numbers_with_product_exceeding_8 = 2400 :=
by
  sorry

end number_of_four_digit_numbers_l144_144319


namespace solve_for_integers_l144_144034

theorem solve_for_integers (a b c : ℕ) (hb : nat.prime b) (hc : nat.prime c) : 
  (1 / a + 1 / b = 1 / c) → (a = 6 ∧ b = 3 ∧ c = 2) :=
by 
  sorry

end solve_for_integers_l144_144034


namespace ratio_of_ages_l144_144773

theorem ratio_of_ages (Sandy_age : ℕ) (Molly_age : ℕ)
  (h1 : Sandy_age = 56)
  (h2 : Molly_age = Sandy_age + 16) :
  (Sandy_age : ℚ) / Molly_age = 7 / 9 :=
by
  -- Proof goes here
  sorry

end ratio_of_ages_l144_144773


namespace remainder_of_power_sums_modulo_seven_l144_144499

theorem remainder_of_power_sums_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := 
by 
  sorry

end remainder_of_power_sums_modulo_seven_l144_144499


namespace similarity_of_triangles_l144_144882

variables {P A B C A1 B1 C1 A2 B2 C2 A3 B3 C3 : Type}
variable {triangle : Type} [isTriangle triangle]

-- Assuming that the points are inside the triangle and the perpendiculars create new triangles
axiom triangle_acute (ABC : triangle) (acute : ∀ (A B C : triangle), angle A < π / 2 ∧ angle B < π / 2 ∧ angle C < π / 2)
axiom point_inside (P : { x | x ∈ interior ABC })

axiom perpendiculars (PA1 PB1 PC1 : line) 
  (PA1 : isPerpendicular PA1 BC)
  (PB1 : isPerpendicular PB1 CA)
  (PC1 : isPerpendicular PC1 AB)

-- Formation of pedal triangles from the perpendicular 
def pedal_triangle (P : point) (triangle : triangle) : triangle :=
  ((PA1, PB1, PC1))

def repeat_pedal (P : point) (triangle : triangle) : triangle :=
  pedal_triangle(P, pedal_triangle(P, pedal_triangle(P, triangle)))

open_locale classical

theorem similarity_of_triangles (ABC : triangle) 
  (acute : triangle_acute ABC)
  (P : point_inside P)
  (A1 B1 C1 : triangle) 
  (H₁ : pedal_triangle P ABC = (A1, B1, C1))
  (A2 B2 C2 : triangle)
  (H₂ : pedal_triangle P (A1, B1, C1) = (A2, B2, C2))
  (A3 B3 C3 : triangle)
  (H₃ : pedal_triangle P (A2, B2, C2) = (A3, B3, C3)) :
  (∃ (A B C : triangle), isSimilar ABC (A3, B3, C3)) :=
sorry

end similarity_of_triangles_l144_144882


namespace four_digit_numbers_count_l144_144315

theorem four_digit_numbers_count :
  ∃ n : ℕ, n = 4140 ∧
  (∀ d1 d2 d3 d4 : ℕ,
    (4 ≤ d1 ∧ d1 ≤ 9) ∧
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d2 * d3 > 8) →
    (∃ m : ℕ, m = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ m > 3999) →
    n = 4140) :=
sorry

end four_digit_numbers_count_l144_144315


namespace smallest_three_digit_geometric_seq_l144_144513

/-!

# The Smallest Three-Digit Integer with Distinct Digits in a Geometric Sequence

## Problem Statement
Prove that the smallest three-digit integer whose digits are distinct and form a geometric sequence is 248.

## Conditions
1. The integer must be three digits long.
2. The integer's digits must be distinct.
3. The digits must form a geometric sequence with a common ratio \( r > 1 \).

-/

def hundred_digit := 2
def tens_digit := 2 * 2
def units_digit := 2 * 2 * 2

theorem smallest_three_digit_geometric_seq :
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧
  let a := ((n / 100) % 10) in
  let b := ((n / 10) % 10) in
  let c := (n % 10) in
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  b = a * 2 ∧ c = a * 2 * 2 ∧
  n = 248 :=
sorry

end smallest_three_digit_geometric_seq_l144_144513


namespace prob_divisible_by_5_l144_144088

theorem prob_divisible_by_5 (M: ℕ) (h1: 100 ≤ M ∧ M < 1000) (h2: M % 10 = 5): 
  (∃ (k: ℕ), M = 5 * k) :=
by
  sorry

end prob_divisible_by_5_l144_144088


namespace parabola_range_l144_144815

theorem parabola_range (x : ℝ) (h : 0 < x ∧ x < 3) : 
  1 ≤ (x^2 - 4*x + 5) ∧ (x^2 - 4*x + 5) < 5 :=
sorry

end parabola_range_l144_144815


namespace daniel_subtracts_number_l144_144482

theorem daniel_subtracts_number:
  let a := 40
  let b := 1 in
  (a - b) ^ 2 = a ^ 2 - 79 := 
by
  sorry

end daniel_subtracts_number_l144_144482


namespace ellipse_x_intercept_l144_144934

theorem ellipse_x_intercept (x : ℝ) :
  let f1 := (0, 3)
  let f2 := (4, 0)
  let origin := (0, 0)
  let d := sqrt ((fst f1)^2 + (snd f1)^2) + sqrt ((fst f2)^2 + (snd f2)^2)
  d = 7 → -- sum of distances from origin to the foci is 7
  (d_1 : ℝ := abs x - 4 + sqrt (x^2 + 9))
  d_1 = 7 → -- sum of distances from (x, 0) to the foci is 7
  x ≠ 0 → -- x is not 0 because the other x-intercept is not (0, 0)
  x = 56 / 11 → -- x > 4
  (x, 0) = ((56 : ℝ) / 11, 0) :=
by
  sorry

end ellipse_x_intercept_l144_144934


namespace fewest_four_dollar_frisbees_l144_144568

theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : x + y = 60) (h2 : 3 * x + 4 * y = 200) : y = 20 :=
by 
  sorry  

end fewest_four_dollar_frisbees_l144_144568


namespace count_sequences_l144_144327

theorem count_sequences (n : ℕ) (a : fin n → ℕ) (h1 : ∃ i : fin n, a i > 0) 
(h2 : (finset.univ.sum (λ i, a i)) = 10) 
(h3 : ∀ i, i < n - 1 → a i + a (i + 1) > 0) : 
∑ k in (finset.range 10).filter (λ k, k ≥ 1 ∧ k ≤ 10), 
((nat.choose 9 (k-1)) * (2^(k-1))) = 19683 :=
by {
  sorry
}

end count_sequences_l144_144327


namespace some_seniors_not_club_members_l144_144972

variables {People : Type} (Senior ClubMember : People → Prop) (Punctual : People → Prop)

-- Conditions:
def some_seniors_not_punctual := ∃ x, Senior x ∧ ¬Punctual x
def all_club_members_punctual := ∀ x, ClubMember x → Punctual x

-- Theorem statement to be proven:
theorem some_seniors_not_club_members (h1 : some_seniors_not_punctual Senior Punctual) (h2 : all_club_members_punctual ClubMember Punctual) : 
  ∃ x, Senior x ∧ ¬ ClubMember x :=
sorry

end some_seniors_not_club_members_l144_144972


namespace parabola_through_point_l144_144819

theorem parabola_through_point (x y : ℝ) (hx : x = 2) (hy : y = 4) : 
  (∃ a : ℝ, y^2 = a * x ∧ a = 8) ∨ (∃ b : ℝ, x^2 = b * y ∧ b = 1) :=
sorry

end parabola_through_point_l144_144819


namespace distance_each_player_runs_l144_144055

-- Definitions based on conditions
def length : ℝ := 100
def width : ℝ := 50
def laps : ℝ := 6

def perimeter (l w : ℝ) : ℝ := 2 * (l + w)

def total_distance (l w laps : ℝ) : ℝ := laps * perimeter l w

-- Theorem statement
theorem distance_each_player_runs :
  total_distance length width laps = 1800 := 
by 
  sorry

end distance_each_player_runs_l144_144055


namespace largest_n_for_factorable_polynomial_l144_144243

theorem largest_n_for_factorable_polynomial :
  (∃ (A B : ℤ), A * B = 72 ∧ ∀ (n : ℤ), n = 3 * B + A → n ≤ 217) ∧
  (∃ (A B : ℤ), A * B = 72 ∧ 3 * B + A = 217) :=
by
    sorry

end largest_n_for_factorable_polynomial_l144_144243


namespace log_base4_of_1_div_64_l144_144221

theorem log_base4_of_1_div_64 : log 4 (1 / 64) = -3 := by
  sorry

end log_base4_of_1_div_64_l144_144221


namespace quotient_is_8_l144_144628

def dividend : ℕ := 64
def divisor : ℕ := 8
def quotient := dividend / divisor

theorem quotient_is_8 : quotient = 8 := 
by 
  show quotient = 8 
  sorry

end quotient_is_8_l144_144628


namespace petya_can_obtain_1001_l144_144764

noncomputable def can_obtain_1001 (n : ℕ) : Prop :=
  ∃ (seq : ℕ → ℚ), (seq 0 = n) ∧ (∀ k, seq (k + 1) = seq k * (seq k * 3).toNat) ∧ (seq ? = 1001)

theorem petya_can_obtain_1001 (n : ℕ) (h : ∃ r : ℚ, r ≥ 1/3 ∧ r ≤ 2) : can_obtain_1001 n :=
by sorry

end petya_can_obtain_1001_l144_144764


namespace ganesh_speed_y_to_x_l144_144531

variable (D : ℝ) (S : ℝ)
variable (T1 T2 T : ℝ)

def average_speed_x_to_y : ℝ := 60
def average_speed_journey : ℝ := 45

def time_from_x_to_y := D / average_speed_x_to_y
def time_from_y_to_x := D / S

def total_time := time_from_x_to_y + time_from_y_to_x

def total_distance := 2 * D

def avg_speed_formula : ℝ := total_distance / total_time

theorem ganesh_speed_y_to_x 
  (D_pos : 0 < D)
  (h_avg_speed_journey : average_speed_journey = avg_speed_formula) 
  : S = 36 :=
by
  sorry

end ganesh_speed_y_to_x_l144_144531


namespace fraction_halfway_between_l144_144065

theorem fraction_halfway_between : 
  ∃ (x : ℚ), (x = (1 / 6 + 1 / 4) / 2) ∧ x = 5 / 24 :=
by
  sorry

end fraction_halfway_between_l144_144065


namespace find_f_2015_l144_144279

noncomputable def f : ℝ → ℝ :=
sorry

lemma f_period : ∀ x : ℝ, f (x + 8) = f x :=
sorry

axiom f_func_eq : ∀ x : ℝ, f (x + 2) = (1 + f x) / (1 - f x)

axiom f_initial : f 1 = 1 / 4

theorem find_f_2015 : f 2015 = -3 / 5 :=
sorry

end find_f_2015_l144_144279


namespace range_f_interval_l144_144514

def f (x : ℝ) : ℝ := x^2 + 2 * x - 2

theorem range_f_interval :
  set.range f ∩ set.Icc (-2 : ℝ) 1 = set.Icc (-3 : ℝ) 1 := 
sorry

end range_f_interval_l144_144514


namespace populationTwoYearsAgo_l144_144453

-- Definitions for the conditions
def initialPopulation (P : ℝ) := P
def firstYearIncreaseRate := 0.10
def secondYearIncreaseRate := 0.20
def presentPopulation := 1320.0

-- Define the math problem
theorem populationTwoYearsAgo (P : ℝ) 
  (H1 : P * (1 + firstYearIncreaseRate) * (1 + secondYearIncreaseRate) = presentPopulation) : 
  P = 1000 := 
sorry -- Proof to be provided

end populationTwoYearsAgo_l144_144453


namespace expression_evaluation_l144_144700

theorem expression_evaluation (x : ℤ) (hx : x = 4) : 5 * x + 3 - x^2 = 7 :=
by
  sorry

end expression_evaluation_l144_144700


namespace roundness_of_8_million_l144_144592

theorem roundness_of_8_million : 
  let n := 8000000
  let prime_factorization := 2^9 * 5^6
  (∃ a b : ℕ, n = 2^a * 5^b ∧ a + b = 15) :=
by { 
  let a := 9, 
  let b := 6, 
  have factorization : 8000000 = 2^a * 5^b := 
    by norm_num,
  exact ⟨a, b, factorization, by norm_num⟩,
  sorry 
}

end roundness_of_8_million_l144_144592


namespace solve_for_x_l144_144430

theorem solve_for_x (x : ℝ) : (2 ^ (8 ^ (2 * x)) = 8 ^ (2 ^ (2 * x))) → x = (Real.log 3 / Real.log 2) / 4 :=
by
  intro h
  sorry

end solve_for_x_l144_144430


namespace project_hours_ratio_l144_144435

theorem project_hours_ratio (x y z : ℕ) (h : x + y + z = 90)
  (h1 : y = x + 20) (h2 : z = 70 - 2 * x): 
  x : y : z = 5 : 9 : 4 :=
by {
  have h_y : y = x + 20 := h1,
  have h_z : z = 70 - 2 * x := h2,
  rw [h_y, h_z] at h,
  sorry
}

end project_hours_ratio_l144_144435


namespace number_of_dots_on_A_number_of_dots_on_B_number_of_dots_on_C_number_of_dots_on_D_l144_144823

variable (A B C D : ℕ)
variable (dice : ℕ → ℕ)

-- Conditions
axiom dice_faces : ∀ {i : ℕ}, 1 ≤ i ∧ i ≤ 6 → ∃ j, dice i = j
axiom opposite_faces_sum : ∀ {i j : ℕ}, dice i + dice j = 7
axiom configuration : True -- Placeholder for the specific arrangement configuration

-- Questions and Proof Statements
theorem number_of_dots_on_A :
  A = 3 := sorry

theorem number_of_dots_on_B :
  B = 5 := sorry

theorem number_of_dots_on_C :
  C = 6 := sorry

theorem number_of_dots_on_D :
  D = 5 := sorry

end number_of_dots_on_A_number_of_dots_on_B_number_of_dots_on_C_number_of_dots_on_D_l144_144823


namespace smallest_geometric_sequence_number_l144_144508

theorem smallest_geometric_sequence_number :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
    (∀ d ∈ [((n / 100) % 10), ((n / 10) % 10), (n % 10)], d ∈ [1,2,3,4,5,6,7,8,9]) ∧
    (let digits := [((n / 100) % 10), ((n / 10) % 10), (n % 10)] in
       digits.nodup ∧
       ∃ r : ℕ, r > 1 ∧ digits = [digits.head!, digits.head! * r, digits.head! * r * r]) ∧
    n = 124 :=
begin
  sorry
end

end smallest_geometric_sequence_number_l144_144508


namespace complete_square_transform_l144_144516

theorem complete_square_transform (x : ℝ) :
  x^2 - 8 * x + 2 = 0 → (x - 4)^2 = 14 :=
by
  intro h
  sorry

end complete_square_transform_l144_144516


namespace smallest_three_digit_geometric_seq_l144_144510

/-!

# The Smallest Three-Digit Integer with Distinct Digits in a Geometric Sequence

## Problem Statement
Prove that the smallest three-digit integer whose digits are distinct and form a geometric sequence is 248.

## Conditions
1. The integer must be three digits long.
2. The integer's digits must be distinct.
3. The digits must form a geometric sequence with a common ratio \( r > 1 \).

-/

def hundred_digit := 2
def tens_digit := 2 * 2
def units_digit := 2 * 2 * 2

theorem smallest_three_digit_geometric_seq :
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧
  let a := ((n / 100) % 10) in
  let b := ((n / 10) % 10) in
  let c := (n % 10) in
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  b = a * 2 ∧ c = a * 2 * 2 ∧
  n = 248 :=
sorry

end smallest_three_digit_geometric_seq_l144_144510


namespace condition1_condition2_condition3_l144_144636

noncomputable def Z (m : ℝ) : ℂ := (m^2 - 4 * m) + (m^2 - m - 6) * Complex.I

-- Condition 1: Point Z is in the third quadrant
theorem condition1 (m : ℝ) (h_quad3 : (m^2 - 4 * m) < 0 ∧ (m^2 - m - 6) < 0) : 0 < m ∧ m < 3 :=
sorry

-- Condition 2: Point Z is on the imaginary axis
theorem condition2 (m : ℝ) (h_imaginary : (m^2 - 4 * m) = 0 ∧ (m^2 - m - 6) ≠ 0) : m = 0 ∨ m = 4 :=
sorry

-- Condition 3: Point Z is on the line x - y + 3 = 0
theorem condition3 (m : ℝ) (h_line : (m^2 - 4 * m) - (m^2 - m - 6) + 3 = 0) : m = 3 :=
sorry

end condition1_condition2_condition3_l144_144636


namespace log_base_4_frac_l144_144224

theorem log_base_4_frac :
  logb 4 (1/64) = -3 :=
sorry

end log_base_4_frac_l144_144224


namespace possible_values_n_l144_144359

theorem possible_values_n (n : ℕ) (h_pos : 0 < n) (h1 : n > 9 / 4) (h2 : n < 14) :
  ∃ S : Finset ℕ, S.card = 11 ∧ ∀ k ∈ S, k = n :=
by
  -- proof to be filled in
  sorry

end possible_values_n_l144_144359


namespace scientific_notation_of_300670_l144_144776

theorem scientific_notation_of_300670 : ∃ a : ℝ, ∃ n : ℤ, (1 ≤ |a| ∧ |a| < 10) ∧ 300670 = a * 10^n ∧ a = 3.0067 ∧ n = 5 :=
  by
    sorry

end scientific_notation_of_300670_l144_144776


namespace probability_three_suits_in_five_draws_l144_144701

open ProbabilityTheory

theorem probability_three_suits_in_five_draws :
  (∑ (suit_missing : Fin 4), ((3 / 4 : ℚ) ^ 5)) = 243 / 256 := 
sorry

end probability_three_suits_in_five_draws_l144_144701


namespace hyperbola_eccentricity_10_l144_144662

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (line : ∃ x y : ℝ, y = x + 1 ∧ (x^2 / a^2 - y^2 / b^2 = 1))
  (condition : |⟨a / (b - a), b / (b - a)⟩| = 2 * |⟨-a / (b + a), b / (b + a)⟩|) : ℝ :=
  sqrt ((a^2 + b^2) / a^2)

theorem hyperbola_eccentricity_10 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (line : ∃ x y : ℝ, y = x + 1 ∧ (x^2 / a^2 - y^2 / b^2 = 1))
  (condition : |⟨a / (b - a), b / (b - a)⟩| = 2 * |⟨-a / (b + a), b / (b + a)⟩|)
  : hyperbola_eccentricity a b ha hb line condition = sqrt 10 :=
sorry

end hyperbola_eccentricity_10_l144_144662


namespace quadrilateral_projections_l144_144687

structure Quadrilateral (A B C D : Type) :=
(angle_at_A_is_right : ∠A = 90)
(angle_at_C_is_right : ∠C = 90)

noncomputable def projections_equal 
  {A B C D : Type} 
  (Q : Quadrilateral A B C D)
  (proj_AB_AC : ℝ)
  (proj_CD_AC : ℝ) : Prop :=
  proj_AB_AC = proj_CD_AC

theorem quadrilateral_projections 
  {A B C D : Type} 
  (Q : Quadrilateral A B C D)
  (proj_AB_AC : ℝ)
  (proj_CD_AC : ℝ)
  (h : projections_equal Q proj_AB_AC proj_CD_AC) :
  proj_AB_AC = proj_CD_AC := 
by 
  sorry

end quadrilateral_projections_l144_144687


namespace cube_skew_lines_l144_144263

/-- Given a cube, the number of pairs of skew lines that can be formed by one of its diagonals
    and the edges of the cube is equal to 6. -/
theorem cube_skew_lines : ∀ (c : Cube), (number_of_skew_pairs c) = 6 :=
by
  sorry

end cube_skew_lines_l144_144263


namespace remainder_of_powers_l144_144496

theorem remainder_of_powers (n1 n2 n3 : ℕ) : (9^6 + 8^8 + 7^9) % 7 = 2 :=
by
  sorry

end remainder_of_powers_l144_144496


namespace concurrency_of_AN_BQ_CP_l144_144726

variables {A B C D M N P Q : Type}

-- Assuming the properties of points
variables [non_empty A] [non_empty B] [non_empty C]
variables (D : Point) [altitude_from A_to BC : Altitude A BC D]
variables (M : Point) [midpoint M_of_BC : Midpoint M B C]
variables (N : Point) [reflection M_across_D : Reflection M D N]
variables [circumcircle_AMN : Circumcircle A M N]

-- Points P and Q on the circumcircle
variables (P : Point) [on_AB_and_circumcircle_AMN P : OnLineAndCircle P AB (Circumcircle A M N)]
variables (Q : Point) [on_AC_and_circumcircle_AMN Q : OnLineAndCircle Q AC (Circumcircle A M N)]

theorem concurrency_of_AN_BQ_CP :
  Concurrent (LineThrough A N) (LineThrough B Q) (LineThrough C P) := 
  sorry

end concurrency_of_AN_BQ_CP_l144_144726


namespace zebras_total_games_l144_144190

theorem zebras_total_games 
  (x y : ℝ)
  (h1 : x = 0.40 * y)
  (h2 : (x + 8) / (y + 11) = 0.55) 
  : y + 11 = 24 :=
sorry

end zebras_total_games_l144_144190


namespace trigonometric_identity_proof_l144_144416

theorem trigonometric_identity_proof :
  3.438 * (Real.sin (84 * Real.pi / 180)) * (Real.sin (24 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * (Real.sin (12 * Real.pi / 180)) = 1 / 16 :=
  sorry

end trigonometric_identity_proof_l144_144416


namespace slope_of_l4_l144_144010

open Real

def line1 (x y : ℝ) : Prop := 4 * x - 3 * y = 6
def pointD : ℝ × ℝ := (0, -2)
def line2 (y : ℝ) : Prop := y = -1
def area_triangle_DEF := 4

theorem slope_of_l4 
  (l4_slope : ℝ)
  (H1 : ∃ x, line1 x (-1))
  (H2 : ∀ x y, 
         x ≠ 0 ∧
         y ≠ -2 ∧
         y ≠ -1 →
         line2 y →
         l4_slope = (y - (-2)) / (x - 0) →
         (1/2) * |(y + 1)| * (sqrt ((x-0) * (x-0) + (y-(-2)) * (y-(-2)))) = area_triangle_DEF ) :
  l4_slope = 1 / 8 :=
sorry

end slope_of_l4_l144_144010


namespace conjugate_complex_number_l144_144056

theorem conjugate_complex_number :
  let z := (2 - (complex.I)) / (2 + (complex.I))
  complex.conj z = (3 / 5) + (4 / 5) * (complex.I) :=
by
  sorry

end conjugate_complex_number_l144_144056


namespace interest_problem_l144_144457

theorem interest_problem
  (P : ℝ)
  (h : P * 0.04 * 5 = P * 0.05 * 4) : 
  (P * 0.04 * 5) = 20 := 
by 
  sorry

end interest_problem_l144_144457


namespace increasing_on_interval_min_value_on_interval_l144_144681

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

theorem increasing_on_interval (a : ℝ) : 
  (∀ x, 1 < x → (deriv (f a) x) ≥ 0) ↔ (a ≥ -2) := sorry

theorem min_value_on_interval (a : ℝ) :
  let f_min := 
    if a ≥ -2 then 1
    else if -2 * Real.exp 2 < a ∧ a < -2 then (a / 2) * Real.log (-a / 2) - a / 2
    else a + Real.exp 2 in
  let x_min := 
    if a ≥ -2 then 1
    else if -2 * Real.exp 2 < a ∧ a < -2 then Real.sqrt (-a / 2)
    else Real.exp 1 in
  ∃ (x : ℝ), x ∈ Set.Icc 1 (Real.exp 1) ∧ ∀ y ∈ Set.Icc 1 (Real.exp 1), f a y ≥ f a x ∧ f a x = f_min ∧ x = x_min := sorry

end increasing_on_interval_min_value_on_interval_l144_144681


namespace rational_expression_iff_rational_square_l144_144083

theorem rational_expression_iff_rational_square (x : ℝ) :
  (∃ r : ℚ, x^2 + (Real.sqrt (x^4 + 1)) - 1 / (x^2 + (Real.sqrt (x^4 + 1))) = r) ↔
  (∃ q : ℚ, x^2 = q) := by
  sorry

end rational_expression_iff_rational_square_l144_144083


namespace winning_candidate_votes_l144_144468

noncomputable def percentage_votes_winner : ℝ := 55.371428571428574 / 100
def votes_candidate_2 : ℕ := 8236
def votes_candidate_3 : ℕ := 11628
def total_votes : ℕ := votes_candidate_2 + votes_candidate_3

theorem winning_candidate_votes (total_votes : ℝ) : 
  total_votes = votes_candidate_2 + votes_candidate_3 + W →
  W = percentage_votes_winner * total_votes →
  W ≈ 24648 :=
by sorry

end winning_candidate_votes_l144_144468


namespace gcd_factorial_l144_144857

theorem gcd_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  sorry

end gcd_factorial_l144_144857


namespace max_maple_trees_l144_144558

theorem max_maple_trees (n : ℕ) : 
  ( ∀ i j (h : i < j ∧ j < 20), (j - i > 4) ↔ (i ∉ {0, 4, 8, 12, 16} → (j ≠ 4 + i ∧ j ≠ 8 + i ∧ j ≠ 12 + i ∧ j ≤ 20 ∧ j ∉ {4, 8, 12, 16, 20}))) → 
  ∃ k, k = 12 := 
by 
  sorry

end max_maple_trees_l144_144558


namespace find_angle_A_max_area_triangle_l144_144655

def f (x : ℝ) : ℝ := 3 + 2 * real.sqrt(3) * real.sin x * real.cos x + 2 * (real.cos x)^2

theorem find_angle_A :
  ∀ (a b c : ℝ) (A B C : ℝ),
  (a > 0) → (b > 0) → (c > 0) →
  A ∈ (0 : ℝ, real.pi) →
  f A = 5 →
  A = real.pi / 3 := sorry

theorem max_area_triangle :
  ∀ (a b c : ℝ) (A B C : ℝ) (area : ℝ),
  (a = 2) →
  (a > 0) → (b > 0) → (c > 0) →
  A = real.pi / 3 →
  area = 0.5 * b * c * real.sin A →
  area ≤ real.sqrt 3 := sorry

end find_angle_A_max_area_triangle_l144_144655


namespace interval_of_monotonic_increase_l144_144448

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp(x)

theorem interval_of_monotonic_increase :
  {x : ℝ | f' x > 0} = {x | x > 2} :=
by
  sorry

end interval_of_monotonic_increase_l144_144448


namespace wilson_buys_3_bottles_of_cola_l144_144127

theorem wilson_buys_3_bottles_of_cola
    (num_hamburgers : ℕ := 2) 
    (cost_per_hamburger : ℕ := 5) 
    (cost_per_cola : ℕ := 2) 
    (discount : ℕ := 4) 
    (total_paid : ℕ := 12) :
    num_hamburgers * cost_per_hamburger - discount + x * cost_per_cola = total_paid → x = 3 :=
by
  sorry

end wilson_buys_3_bottles_of_cola_l144_144127


namespace probability_two_red_balls_l144_144876

open Nat

theorem probability_two_red_balls (total_balls red_balls blue_balls green_balls balls_picked : Nat) 
  (total_eq : total_balls = red_balls + blue_balls + green_balls) 
  (red_eq : red_balls = 7) 
  (blue_eq : blue_balls = 5) 
  (green_eq : green_balls = 4) 
  (picked_eq : balls_picked = 2) :
  (choose red_balls balls_picked) / (choose total_balls balls_picked) = 7 / 40 :=
by
  sorry

end probability_two_red_balls_l144_144876


namespace remainder_of_power_sums_modulo_seven_l144_144500

theorem remainder_of_power_sums_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := 
by 
  sorry

end remainder_of_power_sums_modulo_seven_l144_144500


namespace probability_no_more_than_10_seconds_l144_144189

noncomputable def total_cycle_time : ℕ := 80
noncomputable def green_time : ℕ := 30
noncomputable def yellow_time : ℕ := 10
noncomputable def red_time : ℕ := 40
noncomputable def can_proceed : ℕ := green_time + yellow_time + yellow_time

theorem probability_no_more_than_10_seconds : 
  can_proceed / total_cycle_time = 5 / 8 := 
  sorry

end probability_no_more_than_10_seconds_l144_144189


namespace first_proof_l144_144167

def triangular (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def covers_all_columns (k : ℕ) : Prop :=
  ∀ c : ℕ, (c < 10) → (∃ m : ℕ, m ≤ k ∧ (triangular m) % 10 = c)

theorem first_proof (k : ℕ) (h : covers_all_columns 28) : 
  triangular k = 435 :=
sorry

end first_proof_l144_144167


namespace sarah_storage_l144_144426

variables (rye_flour whole_wheat_bread_flour chickpea_flour_g almond_flour_g coconut_flour_oz whole_wheat_pastry_flour all_purpose_flour_g : ℝ)
variables (grams_per_pound ounces_per_pound storage_limit : ℝ)
variables h_storage : storage_limit = 45
variables h_pounds_to_grams : grams_per_pound = 454
variables h_pounds_to_ounces : ounces_per_pound = 16

def total_flour_weight (rye_flour whole_wheat_bread_flour chickpea_flour_g almond_flour_g coconut_flour_oz whole_wheat_pastry_flour all_purpose_flour_g : ℝ) (grams_per_pound ounces_per_pound : ℝ) : ℝ :=
  rye_flour + 
  whole_wheat_bread_flour + 
  (chickpea_flour_g / grams_per_pound) + 
  (almond_flour_g / grams_per_pound) + 
  (coconut_flour_oz / ounces_per_pound) + 
  whole_wheat_pastry_flour + 
  (all_purpose_flour_g / grams_per_pound) 

theorem sarah_storage (rye_flour whole_wheat_bread_flour chickpea_flour_g almond_flour_g coconut_flour_oz whole_wheat_pastry_flour all_purpose_flour_g grams_per_pound ounces_per_pound storage_limit : ℝ)
  (h_storage : storage_limit = 45)
  (h_pounds_to_grams : grams_per_pound = 454)
  (h_pounds_to_ounces : ounces_per_pound = 16)
  (h_rye_flour : rye_flour = 5)
  (h_whole_wheat_bread_flour : whole_wheat_bread_flour = 10)
  (h_chickpea_flour_g : chickpea_flour_g = 1800)
  (h_almond_flour_g : almond_flour_g = 3000)
  (h_coconut_flour_oz : coconut_flour_oz = 10)
  (h_whole_wheat_pastry_flour : whole_wheat_pastry_flour = 2)
  (h_all_purpose_flour_g : all_purpose_flour_g = 500) :
  total_flour_weight rye_flour whole_wheat_bread_flour chickpea_flour_g almond_flour_g coconut_flour_oz whole_wheat_pastry_flour all_purpose_flour_g grams_per_pound ounces_per_pound ≤ storage_limit :=
by sorry

end sarah_storage_l144_144426


namespace find_a_l144_144261

-- Assume: f(x) = ax - log2(4^x + 1) is an even function
def f (a x : ℝ) : ℝ := a * x - Real.log2 (4^x + 1)

-- f is even
def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

theorem find_a (a : ℝ) (h : is_even (f a)) : a = 1 := by
  sorry

end find_a_l144_144261


namespace max_difference_bound_l144_144748

theorem max_difference_bound (n : ℕ) (a : Fin (2 * n) → ℝ) 
  (h_condition : ∑ i in Finset.range (2 * n - 1), (a i.succ - a i) ^ 2 = 1) :
  (∑ i in Finset.range n, a (Fin.ofNat (i + n))) - (∑ i in Finset.range n, a (Fin.ofNat i)) 
  ≤ real.sqrt (n * (2 * n^2 + 1) / 3) := 
sorry

end max_difference_bound_l144_144748


namespace other_x_intercept_l144_144948

noncomputable def ellipse_x_intercepts (f1 f2 : ℝ × ℝ) (x_intercept1 : ℝ × ℝ) : ℝ × ℝ :=
  let d := dist f1 x_intercept1 + dist f2 x_intercept1
  let x := (d^2 - 2 * d * sqrt (3^2 + (d / 2 - 4)^2)) / (2 * d - 8)
  (x, 0)

theorem other_x_intercept :
  ellipse_x_intercepts (0, 3) (4, 0) (0, 0) = (56 / 11, 0) :=
by
  sorry

end other_x_intercept_l144_144948


namespace sum_of_cubes_mod_6_l144_144591

theorem sum_of_cubes_mod_6 :
  (∑ k in Finset.range 150, (k + 1) ^ 3) % 6 = 5 :=
by
  have h : ∀ n : ℕ, n % 6 = (n ^ 3) % 6 := sorry
  sorry

end sum_of_cubes_mod_6_l144_144591


namespace village_population_percentage_l144_144925

theorem village_population_percentage (P0 P2 P1 : ℝ) (x : ℝ)
  (hP0 : P0 = 7800)
  (hP2 : P2 = 5265)
  (hP1 : P1 = P0 * (1 - x / 100))
  (hP2_eq : P2 = P1 * 0.75) :
  x = 10 :=
by
  sorry

end village_population_percentage_l144_144925


namespace probability_of_multiples_of_2_3_5_7_l144_144926

def number_of_cards := 120
def multiples_of (n : ℕ) := { x : ℕ // x ≤ number_of_cards ∧ x % n = 0 }

def A := multiples_of 2
def B := multiples_of 3
def C := multiples_of 5
def D := multiples_of 7

noncomputable def card_A : ℕ := (finset.map finset.range (λ x, x / 2)).card
noncomputable def card_B : ℕ := (finset.map finset.range (λ x, x / 3)).card
noncomputable def card_C : ℕ := (finset.map finset.range (λ x, x / 5)).card
noncomputable def card_D : ℕ := (finset.map finset.range (λ x, x / 7)).card

noncomputable def card_A_inter_B : ℕ := (finset.map finset.range (λ x, x / 6)).card
noncomputable def card_A_inter_C : ℕ := (finset.map finset.range (λ x, x / 10)).card
noncomputable def card_A_inter_D : ℕ := (finset.map finset.range (λ x, x / 14)).card
noncomputable def card_B_inter_C : ℕ := (finset.map finset.range (λ x, x / 15)).card
noncomputable def card_B_inter_D : ℕ := (finset.map finset.range (λ x, x / 21)).card
noncomputable def card_C_inter_D : ℕ := (finset.map finset.range (λ x, x / 35)).card

noncomputable def card_A_inter_B_inter_C : ℕ := (finset.map finset.range (λ x, x / 30)).card
noncomputable def card_A_inter_B_inter_D : ℕ := (finset.map finset.range (λ x, x / 42)).card
noncomputable def card_A_inter_C_inter_D : ℕ := (finset.map finset.range (λ x, x / 70)).card
noncomputable def card_B_inter_C_inter_D : ℕ := (finset.map finset.range (λ x, x / 105)).card
noncomputable def card_all_inter : ℕ := (finset.map finset.range (λ x, x / 210)).card

noncomputable def total_intersections : ℕ :=
  card_A_inter_B + card_A_inter_C + card_A_inter_D + card_B_inter_C + card_B_inter_D + card_C_inter_D
  - card_A_inter_B_inter_C - card_A_inter_B_inter_D - card_A_inter_C_inter_D - card_B_inter_C_inter_D + card_all_inter

noncomputable def union_card : ℕ :=
  card_A + card_B + card_C + card_D - total_intersections

theorem probability_of_multiples_of_2_3_5_7 : 
  union_card * 15 = 104 * number_of_cards := 
by sorry

end probability_of_multiples_of_2_3_5_7_l144_144926


namespace swap_outer_digits_increases_by_198_l144_144897

-- Given definitions and conditions
variables {c d u : ℕ}

-- Condition 1: When rightmost two digits are swapped, the number increases by 45
def condition1 : Prop := 100 * c + 10 * d + u = 100 * c + 10 * u + d - 45

-- Condition 2: When leftmost two digits are swapped, the number decreases by 270
def condition2 : Prop := 100 * c + 10 * d + u = 100 * d + 10 * c + u + 270

-- Proof statement
theorem swap_outer_digits_increases_by_198 (h1 : condition1) (h2 : condition2) :
  (100 * u + 10 * d + c) - (100 * c + 10 * d + u) = 198 :=
sorry

end swap_outer_digits_increases_by_198_l144_144897


namespace intersect_of_given_circles_l144_144689

noncomputable def circle_center (a b c : ℝ) : ℝ × ℝ :=
  let x := -a / 2
  let y := -b / 2
  (x, y)

noncomputable def radius_squared (a b c : ℝ) : ℝ :=
  (a / 2) ^ 2 + (b / 2) ^ 2 - c

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def circles_intersect (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  let center1 := circle_center a1 b1 c1
  let center2 := circle_center a2 b2 c2
  let r1 := Real.sqrt (radius_squared a1 b1 c1)
  let r2 := Real.sqrt (radius_squared a2 b2 c2)
  let d := distance center1 center2
  r1 - r2 < d ∧ d < r1 + r2

theorem intersect_of_given_circles :
  circles_intersect 4 3 2 2 3 1 :=
sorry

end intersect_of_given_circles_l144_144689


namespace ellipse_x_intercept_other_l144_144941

noncomputable def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 3), (4, 0))
noncomputable def sum_of_distances : ℝ := 7
noncomputable def first_intercept : (ℝ × ℝ) := (0, 0)

theorem ellipse_x_intercept_other 
  (foci : (ℝ × ℝ) × (ℝ × ℝ))
  (sum_of_distances : ℝ)
  (first_intercept : (ℝ × ℝ))
  (hx : foci = ((0, 3), (4, 0)))
  (d_sum : sum_of_distances = 7)
  (intercept : first_intercept = (0, 0)) :
  ∃ (x : ℝ), x > 0 ∧ ((x, 0) = (56 / 11, 0)) := 
sorry

end ellipse_x_intercept_other_l144_144941


namespace find_a_passing_point_increasing_function_values_function_value_zero_l144_144301

namespace MathProofs

variable (a x y : ℝ)

-- Problem 1
theorem find_a_passing_point (h : ∀ x y, y = a * (x - 1)^2 - 4) 
                             (pass_through : h 0 (-3)):
  a = 1 :=
sorry

-- Problem 2
theorem increasing_function_values (h : ∀ x y, y = (x - 1)^2 - 4) :
  ∀ x, x > 1 → ∀ x' > x, y' = (x' - 1)^2 - 4 → y' > y :=
sorry

-- Problem 3
theorem function_value_zero (h : ∀ x y, y = (x - 1)^2 - 4) :
  ∀ x, y = 0 → (x = -1 ∨ x = 3) :=
sorry

end MathProofs

end find_a_passing_point_increasing_function_values_function_value_zero_l144_144301


namespace hyperbola_inequality_l144_144449

theorem hyperbola_inequality
  (a b : ℝ)
  (h_hyperbola : ∃ (P : ℝ × ℝ), (2 * a + 2 * b, a - b) = P ∧ (P.1 ^ 2) / 4 - (P.2 ^ 2) = 1)
  (h_points : (A B : ℝ × ℝ), A = (2, 1) ∧ B = (2, -1)) :
  |a + b| ≥ 1 :=
sorry

end hyperbola_inequality_l144_144449


namespace functional_equation_solution_l144_144237

theorem functional_equation_solution (f : ℤ → ℤ) (c : ℤ) :
  (∀ m n : ℤ, f (f m + n) + f m = f n + f (3 * m) + 2014) →
  ∀ m : ℤ, f m = 2 * m + c :=
begin
  intro h,
  sorry
end

end functional_equation_solution_l144_144237


namespace average_speed_last_segment_l144_144424

variable (total_distance : ℕ := 120)
variable (total_minutes : ℕ := 120)
variable (first_segment_minutes : ℕ := 40)
variable (first_segment_speed : ℕ := 50)
variable (second_segment_minutes : ℕ := 40)
variable (second_segment_speed : ℕ := 55)
variable (third_segment_speed : ℕ := 75)

theorem average_speed_last_segment :
  let total_hours := total_minutes / 60
  let average_speed := total_distance / total_hours
  let speed_first_segment := first_segment_speed * (first_segment_minutes / 60)
  let speed_second_segment := second_segment_speed * (second_segment_minutes / 60)
  let speed_third_segment := third_segment_speed * (third_segment_minutes / 60)
  average_speed = (speed_first_segment + speed_second_segment + speed_third_segment) / 3 →
  third_segment_speed = 75 :=
by
  sorry

end average_speed_last_segment_l144_144424


namespace total_cost_of_items_l144_144078

-- Definitions based on conditions in a)
def price_of_caramel : ℕ := 3
def price_of_candy_bar : ℕ := 2 * price_of_caramel
def price_of_cotton_candy : ℕ := (4 * price_of_candy_bar) / 2
def cost_of_6_candy_bars : ℕ := 6 * price_of_candy_bar
def cost_of_3_caramels : ℕ := 3 * price_of_caramel

-- Problem statement to be proved
theorem total_cost_of_items : cost_of_6_candy_bars + cost_of_3_caramels + price_of_cotton_candy = 57 :=
by
  sorry

end total_cost_of_items_l144_144078


namespace hyperbola_eccentricity_l144_144160

theorem hyperbola_eccentricity 
  (focus_on_x_axis : True)  -- Placeholder condition indicating the hyperbola has its focus on the x-axis
  (asymptotes : ∀ x : ℝ, ∀ y : ℝ, y = (3/4) * x ∨ y = -(3/4) * x) : 
  ∃ e : ℝ, e = 5/4 :=
begin
  use 5/4,
  sorry

end hyperbola_eccentricity_l144_144160


namespace time_expression_l144_144148

theorem time_expression (h V₀ g S V t : ℝ) :
  (V = g * t + V₀) →
  (S = h + (1 / 2) * g * t^2 + V₀ * t) →
  t = (2 * (S - h)) / (V + V₀) :=
by
  intro h_eq v_eq
  sorry

end time_expression_l144_144148


namespace tent_relationship_and_profit_l144_144329

-- Definitions from the conditions
def purchase_price_regular : ℕ := 150
def purchase_price_sunshade : ℕ := 300
def total_budget : ℕ := 9000
def selling_price_regular : ℕ := 180
def selling_price_sunshade : ℕ := 380

-- Conditions
variables (x y : ℕ)

-- Lean statement for proof problem
theorem tent_relationship_and_profit :
  (150 * y + 300 * x = 9000) ∧
  (y = -2 * x + 60) ∧
  (y ≥ 12) ∧
  (y ≤ x) →
  (20 * x + 1800 ≤ 2280) → 
  (20 ≤ x ∧ x ≤ 24) ∧ 
  (y = 12) ∧
  (20 * 24 + 1800 = 2280) :=
sorry

end tent_relationship_and_profit_l144_144329


namespace triangle_identity_l144_144968

theorem triangle_identity
    {A B C D : Type}
    [MetricSpace A]
    [MetricSpace B]
    [MetricSpace C]
    [MetricSpace D]
    (BAC ACB ABC : Real)
    (D_perpendicular : Perpendicular A C D) -- assuming Perpendicular is already defined
    (h1 : ∠ACB = 2 * ∠CAB)
    (h2 : ∠ABC > π/2) : 
    1 / (dist B C) - 2 / (dist D C) = 1 / (dist C A) := 
sorry

end triangle_identity_l144_144968


namespace sasha_or_maxim_mistaken_l144_144029

-- Defining the 100 x 100 grid with non-zero digits
def digit_grid : Type := Matrix (Fin 100) (Fin 100) (Fin 9)

-- Sasha's claim
def all_rows_divisible_by_9 (grid : digit_grid) : Prop :=
  ∀ i : Fin 100, (∑ j, grid i j : ℕ) % 9 = 0

-- Maxim's claim
def all_but_one_column_divisible_by_9 (grid : digit_grid) : Prop :=
  ∃! j : Fin 100, (∑ i, grid i j : ℕ) % 9 ≠ 0

theorem sasha_or_maxim_mistaken (grid : digit_grid) :
  all_rows_divisible_by_9 grid → all_but_one_column_divisible_by_9 grid → False :=
by
  sorry

end sasha_or_maxim_mistaken_l144_144029


namespace second_hand_angle_after_2_minutes_l144_144126

theorem second_hand_angle_after_2_minutes :
  ∀ angle_in_radians, (∀ rotations:ℝ, rotations = 2 → one_full_circle = 2 * Real.pi → angle_in_radians = - (rotations * one_full_circle)) →
  angle_in_radians = -4 * Real.pi :=
by
  intros
  sorry

end second_hand_angle_after_2_minutes_l144_144126


namespace circle_x_intercept_l144_144898

theorem circle_x_intercept (x1 y1 x2 y2 : ℝ) (h1 : x1 = 3) (k1 : y1 = 2) (h2 : x2 = 11) (k2 : y2 = 8) :
  ∃ x : ℝ, (x ≠ 3) ∧ ((x - 7) ^ 2 + (0 - 5) ^ 2 = 25) ∧ (x = 7) :=
by
  sorry

end circle_x_intercept_l144_144898


namespace greatest_prime_factor_of_factorial_sum_l144_144863

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ p, Prime p ∧ p > 11 ∧ (∀ q, Prime q ∧ q > 11 → q ≤ 61) ∧ p = 61 :=
by
  sorry

end greatest_prime_factor_of_factorial_sum_l144_144863


namespace tv_show_cost_l144_144373

theorem tv_show_cost :
  let first_half_cost_per_ep := 1000
  let second_half_increase_percent := 120
  let total_episodes := 22
  let first_half_episodes := total_episodes / 2
  let second_half_episodes := total_episodes / 2
  let first_half_total_cost := first_half_cost_per_ep * first_half_episodes
  let second_half_cost_per_ep := first_half_cost_per_ep + (second_half_increase_percent / 100) * first_half_cost_per_ep
  let second_half_total_cost := second_half_cost_per_ep * second_half_episodes
  let total_cost := first_half_total_cost + second_half_total_cost
  in total_cost = 35200 := by
  -- Skip proof using sorry
  sorry

end tv_show_cost_l144_144373


namespace arrangement_ways_l144_144328

theorem arrangement_ways (math_books english_books : ℕ) (h_math: math_books = 3) (h_english: english_books = 5) : 
  (2! * 3! * 5!) = 1440 :=   
by
  -- Given that we can arrange the 2 groups in 2! ways,
  -- and 3 math books in 3! ways,
  -- and 5 English books in 5! ways,
  -- we expect the total arrangement to be 2! * 3! * 5!
  sorry

end arrangement_ways_l144_144328


namespace solution1_solution2_solution3_l144_144641

noncomputable def problem1 : Nat :=
  (1) * (2 - 1) * (2 + 1)

theorem solution1 : problem1 = 3 := by
  sorry

noncomputable def problem2 : Nat :=
  (2) * (2 + 1) * (2^2 + 1)

theorem solution2 : problem2 = 15 := by
  sorry

noncomputable def problem3 : Nat :=
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)

theorem solution3 : problem3 = 2^64 - 1 := by
  sorry

end solution1_solution2_solution3_l144_144641


namespace ci_values_l144_144039

noncomputable def determine_ci (a : ℝ) (b : Finₓ n → ℝ) (c : Finₓ n → ℝ) : Prop :=
∀ x : ℝ, (x^(2*n) + a * ∑ i in Finset.range (2*n-1), x^i + 1 = ∏ i in Finset.range n, (x^2 + b ⟨i, sorry⟩ * x + c ⟨i, sorry⟩))

theorem ci_values (a : ℝ) (b : Finₓ n → ℝ) (c : Finₓ n → ℝ) (h : determine_ci a b c) : 
(∀ i, c i = 1) :=
sorry

end ci_values_l144_144039


namespace ivan_nails_purchase_l144_144734

variable (x : ℝ)
variable (amount_ivan_had first_store_price_per_100g second_store_price_per_100g : ℝ)
variable (shortfall1 change2 : ℝ)

-- Conditions from the problem
def c1 : first_store_price_per_100g = 180 := rfl
def c2 : shortfall1 = 1430 := rfl
def c3 : second_store_price_per_100g = 120 := rfl
def c4 : change2 = 490 := rfl

-- Theorem that we want to prove
theorem ivan_nails_purchase 
  (h1 : first_store_price_per_100g = 180)
  (h2 : shortfall1 = 1430)
  (h3 : second_store_price_per_100g = 120)
  (h4 : change2 = 490) 
  (amount_ivan_had : ℝ) : 
  x = 3.2 :=
by
  sorry -- The proof elided

end ivan_nails_purchase_l144_144734


namespace geo_progression_bn_l144_144643

variable {ℕ : Type}
noncomputable def sequence_a (n : ℕ) : ℕ 
noncomputable def sum_sequence (n : ℕ) : ℕ :=
sorry

axiom a₁ : sequence_a 1 = 1
axiom Sₙ₊₁ (n : ℕ) : sum_sequence (n + 1) = 4 * sequence_a n + 1
def sequence_b (n : ℕ) : ℕ := sequence_a (n + 1) - 2 * sequence_a n

theorem geo_progression_bn (n : ℕ) : sequence_b n = 2 ^ n := 
sorry

end geo_progression_bn_l144_144643


namespace direction_cosines_squared_sum_one_l144_144906

theorem direction_cosines_squared_sum_one
  (α β γ : ℝ)
  (l : ℝ) -- Representing the line; although we don't use it directly
  (h : ∀ {n}, n ∈ {1, 2, 3} → 
    ((n = 1 → ∃x, x = l ∧ ∀ θ, θ = α ∧ cos θ = l) ∧
    (n = 2 → ∃y, y = l ∧ ∀ θ, θ = β ∧ cos θ = l) ∧
    (n = 3 → ∃z, z = l ∧ ∀ θ, θ = γ ∧ cos θ = l))) :
  cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 = 1 := sorry

end direction_cosines_squared_sum_one_l144_144906


namespace propositions_validity_l144_144182

variable {ℝ : Type*} [LinearOrderedField ℝ]

def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f x = - f (-x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x : ℝ, f (x + p) = f x

theorem propositions_validity
  (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x + f (-x) = f x + f (-x))
  (h2 : is_odd f)
  (h3 : ∀ x : ℝ, f (x + 2) = - f (-(x + 2)))
  : (∀ x : ℝ, f x + f (-x) = f (-x) + f x) ∧
    (∃ p : ℝ, is_periodic f 4) :=
by
  sorry

end propositions_validity_l144_144182


namespace trigonometric_values_of_theta_l144_144290

theorem trigonometric_values_of_theta 
  (θ : ℝ) 
  (origin : θ = 0) 
  (x_axis : ∀ x, θ = x → y = 0) 
  (terminal_side : ∀ (x y : ℝ), y = 2 * x → ∃ (P : ℝ × ℝ), P = (x, y) ∧ θ = atan2 y x) : 
  ∃ (cos_val sin_val tan_val : ℝ), 
  (cos_val = ±(sqrt 5 / 5) ∧ sin_val = ±(2 * sqrt 5 / 5) ∧ tan_val = 2) :=
begin
  sorry
end

end trigonometric_values_of_theta_l144_144290


namespace circumference_is_720_l144_144885

-- Given conditions
def uniform_speed (A_speed B_speed : ℕ) : Prop := A_speed > 0 ∧ B_speed > 0
def diametrically_opposite_start (A_pos B_pos : ℕ) (circumference : ℕ) : Prop := A_pos = 0 ∧ B_pos = circumference / 2
def meets_first_after_B_travel (A_distance B_distance : ℕ) : Prop := B_distance = 150
def meets_second_90_yards_before_A_lap (A_distance_lap B_distance_lap A_distance B_distance : ℕ) : Prop := 
  A_distance_lap = A_distance + 2 * (A_distance - B_distance) - 90 ∧ B_distance_lap = A_distance - B_distance_lap + (B_distance + 90)

theorem circumference_is_720 (circumference A_speed B_speed A_pos B_pos
                     A_distance B_distance
                     A_distance_lap B_distance_lap : ℕ) :
  uniform_speed A_speed B_speed →
  diametrically_opposite_start A_pos B_pos circumference →
  meets_first_after_B_travel A_distance B_distance →
  meets_second_90_yards_before_A_lap A_distance_lap B_distance_lap A_distance B_distance →
  circumference = 720 :=
sorry

end circumference_is_720_l144_144885


namespace fruit_counts_l144_144905

variable (A O B P G : ℤ)

theorem fruit_counts :
  (A - (0.5 * A) + 20 = 370) ∧
  (O - (0.35 * O) = 195) ∧
  (B - (0.6 * B) + 15 = 95) ∧
  (P - (0.45 * P) = 50) ∧
  (G - (0.3 * G) = 140) ↔
  (A = 700) ∧ (O = 300) ∧ (B = 200) ∧ (P = 91) ∧ (G = 200) :=
by sorry

end fruit_counts_l144_144905


namespace find_next_numbers_in_sequence_l144_144626

theorem find_next_numbers_in_sequence :
  ∃ (a b : ℕ), 
  (∀ n, (n % 2 = 0 ∧ n < 8 → nth_sequence_element n = nth_sequence_element (n-1)^2) 
  ∧ nth_sequence_element 7 = 5 
  ∧ a = 25 ∧ nth_sequence_element 8 = a
  ∧ b = 6 ∧ nth_sequence_element 9 = b) :=
sorry

end find_next_numbers_in_sequence_l144_144626


namespace orange_profit_44_percent_l144_144162

theorem orange_profit_44_percent :
  (∀ CP SP : ℚ, 0.99 * CP = 1 ∧ SP = CP / 16 → 1 / 11 = CP * (1 + 44 / 100)) :=
by
  sorry

end orange_profit_44_percent_l144_144162


namespace product_of_g_on_roots_l144_144391

-- Define the given polynomials f and g
def f (x : ℝ) : ℝ := x^5 + 3 * x^2 + 1
def g (x : ℝ) : ℝ := x^2 - 5

-- Define the roots of the polynomial f
axiom roots : ∃ (x1 x2 x3 x4 x5 : ℝ), 
  f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0 ∧ f x5 = 0

theorem product_of_g_on_roots : 
  (∃ x1 x2 x3 x4 x5: ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0 ∧ f x5 = 0) 
  → g x1 * g x2 * g x3 * g x4 * g x5 = 131 := 
by
  sorry

end product_of_g_on_roots_l144_144391


namespace solve_equation_l144_144431

theorem solve_equation (x : ℝ) (h : x * (x - 2) = 2 * (x + 1)) : 
  x = 2 + real.sqrt 6 ∨ x = 2 - real.sqrt 6 :=
by
  sorry

end solve_equation_l144_144431


namespace conjugate_complex_number_l144_144057

theorem conjugate_complex_number :
  let z := (2 - (complex.I)) / (2 + (complex.I))
  complex.conj z = (3 / 5) + (4 / 5) * (complex.I) :=
by
  sorry

end conjugate_complex_number_l144_144057


namespace number_of_members_is_20_l144_144989

noncomputable def number_of_members (candy_bar_cost : ℝ) (avg_candy_bars_per_member : ℝ) (total_earnings : ℝ) : ℕ :=
  let total_candy_bars := total_earnings / candy_bar_cost
  let num_members := total_candy_bars / avg_candy_bars_per_member
  ⌊num_members⌋.to_nat  -- converting to natural number by flooring and casting

theorem number_of_members_is_20 :
  number_of_members 0.50 8 80 = 20 :=
by
  sorry

end number_of_members_is_20_l144_144989


namespace parabola_intercepts_l144_144797

noncomputable def question (y : ℝ) := 3 * y ^ 2 - 9 * y + 4

theorem parabola_intercepts (a b c : ℝ) (h_a : a = question 0) (h_b : 3 * b ^ 2 - 9 * b + 4 = 0) (h_c : 3 * c ^ 2 - 9 * c + 4 = 0) :
  a + b + c = 7 :=
by
  sorry

end parabola_intercepts_l144_144797


namespace smallest_N_and_digit_sum_l144_144953

def Bernardo_wins (N : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 499 ∧ 3 * N < 500 ∧ 9 * N + 75 ≥ 475 ∧ 9 * N + 75 ≤ 499

theorem smallest_N_and_digit_sum :
  ∃ (N : ℕ), (Bernardo_wins N) ∧ 
  (N = 45) ∧ 
  (Nat.digits 10 45).sum = 9 :=
by
  sorry

end smallest_N_and_digit_sum_l144_144953


namespace total_roses_planted_l144_144834

def roses_planted_two_days_ago := 50
def roses_planted_yesterday := roses_planted_two_days_ago + 20
def roses_planted_today := 2 * roses_planted_two_days_ago

theorem total_roses_planted :
  roses_planted_two_days_ago + roses_planted_yesterday + roses_planted_today = 220 := by
  sorry

end total_roses_planted_l144_144834


namespace probability_beautiful_equation_l144_144761

def tetrahedron_faces : Set ℕ := {1, 2, 3, 4}

def is_beautiful_equation (a b : ℕ) : Prop :=
    ∃ m ∈ tetrahedron_faces, a = m + 1 ∨ a = m + 2 ∨ a = m + 3 ∨ a = m + 4 ∧ b = m * (a - m)

theorem probability_beautiful_equation : 
  (∃ a b1 b2, is_beautiful_equation a b1 ∧ is_beautiful_equation a b2) ∧
  (∃ a b1 b2, tetrahedron_faces ⊆ {a} ∧ tetrahedron_faces ⊆ {b1} ∧ tetrahedron_faces ⊆ {b2}) :=
  sorry

end probability_beautiful_equation_l144_144761


namespace smallest_positive_n_l144_144607

theorem smallest_positive_n (x y z : ℕ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (xyz ∣ (x + y + z)^13) :=
by
  sorry

end smallest_positive_n_l144_144607


namespace count_D_zero_two_digit_l144_144253

def rem_sum (n : ℕ) : ℕ :=
  (List.range' 3 (12 - 3 + 1)).sum (λ k => n % k)

def D (n : ℕ) : ℕ :=
  rem_sum n - rem_sum (n - 1)

def count_two_digit_integral_solutions (f : ℕ → ℕ) : ℕ :=
  (List.range' 10 90).count (λ n => f n = 0)

theorem count_D_zero_two_digit : count_two_digit_integral_solutions D = 2 :=
  sorry

end count_D_zero_two_digit_l144_144253


namespace neeleys_sandwich_slices_l144_144757

theorem neeleys_sandwich_slices :
  ∀ (total_slices : ℕ)
    (family_ate_fraction : ℚ)
    (remaining_slices : ℕ),
  (total_slices = 12) →
  (family_ate_fraction = 1 / 3) →
  (remaining_slices = 6) →
  (total_slices - total_slices * family_ate_fraction - remaining_slices = 2) :=
by
  intros total_slices family_ate_fraction remaining_slices 
  assume h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact zero_add 2

end neeleys_sandwich_slices_l144_144757


namespace Payton_score_l144_144402

-- Conditions turned into definitions
def num_students := 15
def first_14_avg := 80
def full_class_avg := 81

-- Desired fact to prove
theorem Payton_score :
  ∃ score : ℕ, score = 95 ∧
  (let total_14 := 14 * first_14_avg in
   let total_15 := num_students * full_class_avg in
   score = total_15 - total_14) :=
begin
  have H14 : total_14 = 1120, from rfl,
  have H15 : total_15 = 1215, from rfl,
  use 95,
  split,
  {
    refl,  -- score = 95
  },
  {
    dsimp [total_14, total_15],
    rw [H14, H15],
    refl,
  },
end

end Payton_score_l144_144402


namespace find_first_hour_l144_144177

noncomputable def plankton_consumed_in_first_hour (P : ℕ) : Prop :=
  let P2 := P + 3 in
  let P3 := P + 6 in
  let P4 := P + 9 in
  let P5 := P + 12 in
  P3 = 93 ∧ P + P2 + P3 + P4 + P5 = 450

theorem find_first_hour : ∃ P : ℕ, plankton_consumed_in_first_hour P ∧ P = 87 :=
by
  sorry

end find_first_hour_l144_144177


namespace cubic_as_diff_of_squares_l144_144260

theorem cubic_as_diff_of_squares (n : ℕ) (h : n > 1) :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n^3 = a^2 - b^2 := 
sorry

end cubic_as_diff_of_squares_l144_144260


namespace total_legos_156_l144_144586

def pyramid_bottom_legos (side_length : Nat) : Nat := side_length * side_length
def pyramid_second_level_legos (length : Nat) (width : Nat) : Nat := length * width
def pyramid_third_level_legos (side_length : Nat) : Nat :=
  let total_legos := (side_length * (side_length + 1)) / 2
  total_legos - 3  -- Subtracting 3 Legos for the corners

def pyramid_fourth_level_legos : Nat := 1

def total_pyramid_legos : Nat :=
  pyramid_bottom_legos 10 +
  pyramid_second_level_legos 8 6 +
  pyramid_third_level_legos 4 +
  pyramid_fourth_level_legos

theorem total_legos_156 : total_pyramid_legos = 156 := by
  sorry

end total_legos_156_l144_144586


namespace sasha_or_maxim_mistaken_l144_144030

-- Defining the 100 x 100 grid with non-zero digits
def digit_grid : Type := Matrix (Fin 100) (Fin 100) (Fin 9)

-- Sasha's claim
def all_rows_divisible_by_9 (grid : digit_grid) : Prop :=
  ∀ i : Fin 100, (∑ j, grid i j : ℕ) % 9 = 0

-- Maxim's claim
def all_but_one_column_divisible_by_9 (grid : digit_grid) : Prop :=
  ∃! j : Fin 100, (∑ i, grid i j : ℕ) % 9 ≠ 0

theorem sasha_or_maxim_mistaken (grid : digit_grid) :
  all_rows_divisible_by_9 grid → all_but_one_column_divisible_by_9 grid → False :=
by
  sorry

end sasha_or_maxim_mistaken_l144_144030


namespace ball_pit_total_l144_144464

-- Define the conditions given in the problem
variables (B : ℕ)
def red_balls := (1 / 4 : ℚ) * B 
def remaining_after_red := B - red_balls
def blue_balls := (1 / 5 : ℚ) * remaining_after_red
def neither_red_nor_blue := B - red_balls - blue_balls

-- The statement to prove
theorem ball_pit_total : neither_red_nor_blue = 216 -> B = 360 := 
by sorry

end ball_pit_total_l144_144464


namespace greatest_k_for_100k_dividing_50_factorial_l144_144133

theorem greatest_k_for_100k_dividing_50_factorial :
  ∃ k : ℕ, (100^k ∣ (∏ i in Finset.range 51, i)) ∧ (∀ l : ℕ, 100^l ∣ (∏ i in Finset.range 51, i) → l ≤ 12) :=
by
  sorry

end greatest_k_for_100k_dividing_50_factorial_l144_144133


namespace gcd_of_factorials_l144_144852

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definitions for the two factorial terms involved
def term1 : ℕ := factorial 7
def term2 : ℕ := factorial 10 / factorial 4

-- Definition for computing the GCD
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- The statement to prove
theorem gcd_of_factorials : gcd term1 term2 = 2520 :=
by {
  -- Here, the proof would go.
  sorry
}

end gcd_of_factorials_l144_144852


namespace height_of_fourth_person_l144_144880

theorem height_of_fourth_person 
  (H : ℕ) 
  (h_avg : ((H) + (H + 2) + (H + 4) + (H + 10)) / 4 = 79) :
  (H + 10 = 85) :=
by
  sorry

end height_of_fourth_person_l144_144880


namespace appears_every_number_smallest_triplicate_number_l144_144867

open Nat

/-- Pascal's triangle is constructed such that each number 
    is the sum of the two numbers directly above it in the 
    previous row -/
def pascal (r k : ℕ) : ℕ :=
  if k > r then 0 else Nat.choose r k

/-- Every positive integer does appear at least once, but not 
    necessarily more than once for smaller numbers -/
theorem appears_every_number (n : ℕ) : ∃ r k : ℕ, pascal r k = n := sorry

/-- The smallest three-digit number in Pascal's triangle 
    that appears more than once is 102 -/
theorem smallest_triplicate_number : ∃ r1 k1 r2 k2 : ℕ, 
  100 ≤ pascal r1 k1 ∧ pascal r1 k1 < 1000 ∧ 
  pascal r1 k1 = 102 ∧ 
  r1 ≠ r2 ∧ k1 ≠ k2 ∧ 
  pascal r1 k1 = pascal r2 k2 := sorry

end appears_every_number_smallest_triplicate_number_l144_144867


namespace non_factorial_tail_up_to_1500_l144_144976

def is_factorial_tail (n m : ℕ) : Prop :=
  n = (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) + (m / 78125) + (m / 390625) + (m / 1953125) + (m / 9765625) + (m / 48828125) + (m / 244140625) + (m / 1220703125) + (m / 6103515625)

def factorial_tail_count (n m : ℕ) : Prop :=
  ∃ m : ℕ, is_factorial_tail n m

def non_factorial_tail_count (N : ℕ) : Prop :=
  (finset.range N).filter (λ n, ¬ factorial_tail_count n N).card = 300

theorem non_factorial_tail_up_to_1500 : non_factorial_tail_count 1500 :=
by sorry

end non_factorial_tail_up_to_1500_l144_144976


namespace second_child_can_ensure_divisible_by_9_k_10_second_child_cannot_ensure_divisible_by_9_k_13_l144_144829

theorem second_child_can_ensure_divisible_by_9_k_10 :
  (∀ (d : Fin 10 → Fin 10),
    (∃ (d2 : Fin 5 → Fin 10), 
       (∑ i in Finset.range 5, d 2 * i + ∑ i in Finset.range 5, d2 i) % 9 = 0)) :=
sorry

theorem second_child_cannot_ensure_divisible_by_9_k_13 :
  ¬(∀ (d : Fin 13 → Fin 10),
    (∃ (d2 : Fin 6 → Fin 10), 
       (∑ i in Finset.range 6, d (2 * i + 1) + ∑ i in Finset.range 7, d2 i) % 9 = 0)) :=
sorry

end second_child_can_ensure_divisible_by_9_k_10_second_child_cannot_ensure_divisible_by_9_k_13_l144_144829


namespace sufficient_but_not_necessary_condition_l144_144439

theorem sufficient_but_not_necessary_condition (a : ℝ) : 
  (∃ x ∈ Icc (-1 : ℝ) 2, x^2 - 2 * x + 4 - a ≤ 0) → (4 ≤ a) := sorry

end sufficient_but_not_necessary_condition_l144_144439


namespace second_reduction_percentage_approx_25_l144_144814

theorem second_reduction_percentage_approx_25 
  (original_price reduced_once_price reduced_twice_price : ℝ)
  (x y z : ℝ)
  (h1 : reduced_once_price = original_price * (1 - 0.25)) 
  (h2 : reduced_twice_price = reduced_once_price * (1 - x))
  (h3 : reduced_twice_price * (1 + 0.7778) = original_price)
  : x ≈ 0.25 := 
sorry

end second_reduction_percentage_approx_25_l144_144814


namespace area_triangle_formula1_area_triangle_formula2_l144_144421

theorem area_triangle_formula1 (α β γ : ℝ) (a : ℝ) (S : ℝ):
  (S = a^2 * sin β * sin γ / (2 * sin α)) :=
sorry

theorem area_triangle_formula2 (α β γ : ℝ) (R : ℝ) (S : ℝ):
  (S = 2 * R^2 * sin α * sin β * sin γ) :=
sorry

end area_triangle_formula1_area_triangle_formula2_l144_144421


namespace find_f_minus_one_l144_144632

variable {ℝ : Type*} [field ℝ]

def f : ℝ → ℝ

axiom functional_eq (x y : ℝ) : f (x^2 + y) = f x + f (y^2)

theorem find_f_minus_one : f (-1) = 0 :=
by
  sorry

end find_f_minus_one_l144_144632


namespace ice_making_cost_l144_144411

-- Pauly's ice making parameters
def pounds_needed : ℝ := 10
def oz_per_cube : ℝ := 2
def weight_per_cube_lbs : ℝ := 1/16
def time_per_10_cubes_hours : ℝ := 1
def cost_per_hour : ℝ := 1.5
def cost_per_oz : ℝ := 0.1

-- The proof statement
theorem ice_making_cost : 
  let num_cubes := pounds_needed / weight_per_cube_lbs in
  let time_required := num_cubes / 10 * time_per_10_cubes_hours in
  let cost_running_ice_maker := time_required * cost_per_hour in
  let total_water_needed := num_cubes * oz_per_cube in
  let cost_water := total_water_needed * cost_per_oz in
  cost_running_ice_maker + cost_water = 56 :=
by
  -- Proof goes here
  sorry

end ice_making_cost_l144_144411


namespace log_base4_one_over_sixty_four_l144_144229

theorem log_base4_one_over_sixty_four : log 4 (1 / 64) = -3 := by
  sorry

end log_base4_one_over_sixty_four_l144_144229


namespace multiply_polynomials_l144_144198

variables {R : Type*} [CommRing R] -- Define R as a commutative ring
variable (x : R) -- Define variable x in R

theorem multiply_polynomials : (2 * x) * (5 * x^2) = 10 * x^3 := 
sorry -- Placeholder for the proof

end multiply_polynomials_l144_144198


namespace number_of_four_digit_numbers_l144_144317

/-- The number of four-digit numbers greater than 3999 where the product of the middle two digits exceeds 8 is 2400. -/
def four_digit_numbers_with_product_exceeding_8 : Nat :=
  let first_digit_options := 6  -- Choices are 4,5,6,7,8,9
  let last_digit_options := 10  -- Choices are 0,1,...,9
  let middle_digit_options := 40  -- Valid pairs counted from the solution
  first_digit_options * middle_digit_options * last_digit_options

theorem number_of_four_digit_numbers : four_digit_numbers_with_product_exceeding_8 = 2400 :=
by
  sorry

end number_of_four_digit_numbers_l144_144317


namespace value_of_a2016_l144_144456

def sequence (a : ℕ → ℚ) : Prop :=
∀ n, a (n + 1) = 1 / (1 - a n)

theorem value_of_a2016 (a : ℕ → ℚ) (h_seq : sequence a) (h_a1 : a 1 = 2) : a 2016 = 1 / 2 :=
sorry

end value_of_a2016_l144_144456


namespace sin_alpha_plus_7pi_div_6_sequence_formula_range_of_a_determine_a_eq_b_l144_144887

-- Problem 1
theorem sin_alpha_plus_7pi_div_6 (α : ℝ) (h : cos(α - π / 3) = 3 / 4) : sin (α + 7 * π / 6) = - 3 / 4 := 
sorry

-- Problem 2
theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : a 2 = 3 / 4) (h₃ : a 3 = 5 / 9) (h₄ : a 4 = 7 / 16) : 
(∀ n : ℕ, a n = (2 * n - 1) / (n * n) ) := 
sorry

-- Problem 3
theorem range_of_a (a b : ℝ) (hA : ∠A = 30) (hb : b = 2) (unique_triangle : ∃ ! (△ABC : Triangle), a) :
a = 1 ∨ a ≥ 2 := 
sorry

-- Problem 4
theorem determine_a_eq_b (∠A ∠B : ℝ) (h₁ : sin ∠A = sin ∠B) (h₂ : cos ∠A = cos ∠B) (h₃ : sin (2 * ∠A) = sin (2 * ∠B)) (h₄ : cos (2 * ∠A) = cos (2 * ∠B)) :
(a = b := 
sorry

end sin_alpha_plus_7pi_div_6_sequence_formula_range_of_a_determine_a_eq_b_l144_144887


namespace igor_min_score_needed_l144_144342

theorem igor_min_score_needed
  (scores : List ℕ)
  (goal : ℚ)
  (next_test_score : ℕ)
  (h_scores : scores = [88, 92, 75, 83, 90])
  (h_goal : goal = 87)
  (h_solution : next_test_score = 94)
  : 
  let current_sum := scores.sum
  let current_tests := scores.length
  let required_total := (goal * (current_tests + 1))
  let next_test_needed := required_total - current_sum
  next_test_needed ≤ next_test_score := 
by 
  sorry

end igor_min_score_needed_l144_144342


namespace find_complex_z_l144_144684

-- Define the determinant operation
def det2x2 (a b c d : ℂ) : ℂ := a * d - b * c

-- Given condition
axiom determinant_condition : ∀ (z : ℂ), det2x2 2 (-1) z (z * complex.I) = 1 + complex.I

-- Theorem stating the result
theorem find_complex_z : ∃ z : ℂ, z = (3 / 5) - ((1 / 5) * complex.I) := by
  -- Using the given axiom to prove the theorem
  sorry

end find_complex_z_l144_144684


namespace find_angle_APQ_l144_144363

variable {A B C D P Q : Type} [euclidean_geometry : EuclideanGeometry A B C D P Q]

-- Definitions of angles
def angle_ABC := 50° -- ∠ABC = 50°
def angle_ACB := 70° -- ∠ACB = 70°

-- Definitions of points and segments
variable (D_mid : Midpoint D B C)
variable (tangent_circle_B : TangentCircle B (Segment D A))
variable (tangent_circle_C : TangentCircle C (Segment D A))

-- Intersection points P and Q
variable (P_on_AB : Intersection tangent_circle_B (Segment A B))
variable (Q_on_AC : Intersection tangent_circle_C (Segment A C))

-- Main theorem statement
theorem find_angle_APQ :
  ∠APQ = 70° :=
by
  -- proof
  sorry

end find_angle_APQ_l144_144363


namespace binary_mult_div_to_decimal_l144_144996

theorem binary_mult_div_to_decimal:
  let n1 := 2 ^ 5 + 2 ^ 4 + 2 ^ 2 + 2 ^ 1 -- This represents 101110_2
  let n2 := 2 ^ 6 + 2 ^ 4 + 2 ^ 2         -- This represents 1010100_2
  let d := 2 ^ 2                          -- This represents 100_2
  n1 * n2 / d = 2995 := 
by
  sorry

end binary_mult_div_to_decimal_l144_144996


namespace ellipse_x_intercept_l144_144932

theorem ellipse_x_intercept (x : ℝ) :
  let f1 := (0, 3)
  let f2 := (4, 0)
  let origin := (0, 0)
  let d := sqrt ((fst f1)^2 + (snd f1)^2) + sqrt ((fst f2)^2 + (snd f2)^2)
  d = 7 → -- sum of distances from origin to the foci is 7
  (d_1 : ℝ := abs x - 4 + sqrt (x^2 + 9))
  d_1 = 7 → -- sum of distances from (x, 0) to the foci is 7
  x ≠ 0 → -- x is not 0 because the other x-intercept is not (0, 0)
  x = 56 / 11 → -- x > 4
  (x, 0) = ((56 : ℝ) / 11, 0) :=
by
  sorry

end ellipse_x_intercept_l144_144932


namespace average_matches_rounded_l144_144713

def total_matches : ℕ := 6 * 1 + 3 * 2 + 3 * 3 + 2 * 4 + 6 * 5

def total_players : ℕ := 6 + 3 + 3 + 2 + 6

noncomputable def average_matches : ℚ := total_matches / total_players

theorem average_matches_rounded : Int.floor (average_matches + 0.5) = 3 :=
by
  unfold average_matches total_matches total_players
  norm_num
  sorry

end average_matches_rounded_l144_144713


namespace solve_system_l144_144778

theorem solve_system :
  ∃ (x1 y1 x2 y2 x3 y3 : ℚ), 
    (x1 = 0 ∧ y1 = 0) ∧ 
    (x2 = -14 ∧ y2 = 6) ∧ 
    (x3 = -85/6 ∧ y3 = 35/6) ∧ 
    ((x1 + 2*y1)*(x1 + 3*y1) = x1 + y1 ∧ (2*x1 + y1)*(3*x1 + y1) = -99*(x1 + y1)) ∧ 
    ((x2 + 2*y2)*(x2 + 3*y2) = x2 + y2 ∧ (2*x2 + y2)*(3*x2 + y2) = -99*(x2 + y2)) ∧ 
    ((x3 + 2*y3)*(x3 + 3*y3) = x3 + y3 ∧ (2*x3 + y3)*(3*x3 + y3) = -99*(x3 + y3)) :=
by
  -- skips the actual proof
  sorry

end solve_system_l144_144778


namespace arithmetic_seq_G_minus_L_l144_144578

noncomputable def arithmetic_seq_75th_term_difference (a d : ℝ) (n : ℕ) :=
  if n % 2 = 1 then
    2 * (a + (n - 1) / 2 * d) / 2
  else
    (a + (n/2 - 1) * d) - (a - (n/2) * d)

theorem arithmetic_seq_G_minus_L :
  ∃ (a d : ℝ) (n : ℕ) (terms : ℕ → ℝ),
    (∀ k, 0 ≤ k → k < 250 → 20 ≤ terms k ∧ terms k ≤ 90) ∧
    (∑ k in finset.range 250, terms k = 15000) ∧
    G - L = 2 * 175 * d = (10500 / 249) :=
begin
  sorry
end

end arithmetic_seq_G_minus_L_l144_144578


namespace determine_operation_l144_144210

theorem determine_operation (a b c d : Int) : ((a - b) + c - (3 * 1) = d) → ((a - b) + 2 = 6) → (a - b = 4) :=
by
  sorry

end determine_operation_l144_144210


namespace total_students_l144_144171

theorem total_students (rank_right rank_left : ℕ) (h1 : rank_right = 6) (h2 : rank_left = 5) : (rank_right + rank_left - 1) = 10 :=
by {
  rw [h1, h2],
  norm_num,
  -- Resulting in showing 6 + 5 - 1 = 10
  sorry
}

end total_students_l144_144171


namespace tangent_sum_eq_neg_2021_l144_144587

theorem tangent_sum_eq_neg_2021 :
  ∑ k in Finset.range 2021, (Real.tan (k.succ * Real.pi / 47) * Real.tan ((k + 2) * Real.pi / 47)) = -2021 :=
sorry

end tangent_sum_eq_neg_2021_l144_144587


namespace sqrt_x_plus_one_over_sqrt_x_l144_144395

theorem sqrt_x_plus_one_over_sqrt_x (x : ℝ) (hx : 0 < x) (h : x + x⁻¹ = 50) : 
  sqrt x + sqrt ((1 : ℝ)/x) = sqrt 52 :=
by
  sorry

end sqrt_x_plus_one_over_sqrt_x_l144_144395


namespace change_in_money_supply_change_in_discount_rate_l144_144143

/-- Part (c) Assumptions and Proof -/
theorem change_in_money_supply (E₀ E₁ : ℝ) (fixed_rate : E₀ = 90) (new_rate : E₁ = 100) (k : ℝ) (k_eq : k = 5) :
  ∃ ΔM : ℝ, ΔM = 2 :=
by {
  have ΔE := ((E₁ - E₀) / E₀) * 100,
  have ΔE_val : ΔE = 11.11 := by sorry,
  have ΔE_rounded : ΔE ≈ 10 := by sorry,
  have ΔM := ΔE / k,
  use ΔM,
  linarith,
}

/-- Part (d) Assumptions and Proof -/
theorem change_in_discount_rate (ΔM : ℝ) (ΔE : ℝ) (per_p_p : ℕ) (per_pp_change : per_p_p = 1) (per_pp_rate : ℝ) (per_pp_rate_change : per_pp_rate = 4) :
  ∃ Δr : ℝ, Δr = 0.5 :=
by {
  have rate_change := (ΔM / per_pp_rate),
  use rate_change,
  linarith,
}

end change_in_money_supply_change_in_discount_rate_l144_144143


namespace omega_value_decreasing_interval_max_value_g_l144_144298

noncomputable def t (ω : ℝ) (x : ℝ) : ℝ :=
  2 * sin (ω * x) * cos (ω * x) + 2 * sqrt 3 * (cos (ω * x))^2

def is_min_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x ∧ (∀ T' > 0, T' < T → ∃ x', f (x' + T') ≠ f x')

theorem omega_value (ω : ℝ) (h : ω > 0)
  (h_min_period : is_min_period (λ x, t ω x) π) : ω = 1 := 
sorry

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3) + sqrt 3

def monotonic_decreasing_interval (f : ℝ → ℝ) (l : ℝ) (u : ℝ) : Prop :=
  ∀ x y, l ≤ x → x ≤ y → y ≤ u → f x ≥ f y

theorem decreasing_interval : ∃ k : ℤ, monotonic_decreasing_interval f (π / 12 + k * π) (7 * π / 12 + k * π) :=
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x) + sqrt 3

theorem max_value_g : ∀ x ∈ Icc 0 (π / 2), g x ≤ 2 + sqrt 3 := 
sorry

end omega_value_decreasing_interval_max_value_g_l144_144298


namespace binom_sum_alternating_50_l144_144995

theorem binom_sum_alternating_50 :
  ∑ k in Finset.range 51, (Nat.choose 50 k) * (-1)^k = 0 :=
by
  sorry

end binom_sum_alternating_50_l144_144995


namespace elementary_events_die_l144_144515

theorem elementary_events_die : 
  let outcomes := {1, 2, 3, 4, 5, 6} in 
  outcomes.card = 6 := 
by 
  -- List the elementary events
  let outcomes := {1, 2, 3, 4, 5, 6}
  -- Calculate the number of elementary events
  sorry

end elementary_events_die_l144_144515


namespace Nina_has_16dollars65_l144_144406

-- Definitions based on given conditions
variables (W M : ℝ)

-- Condition 1: Nina has exactly enough money to purchase 5 widgets
def condition1 : Prop := 5 * W = M

-- Condition 2: If the cost of each widget were reduced by $1.25, Nina would have exactly enough money to purchase 8 widgets
def condition2 : Prop := 8 * (W - 1.25) = M

-- Statement: Proving the amount of money Nina has is $16.65
theorem Nina_has_16dollars65 (h1 : condition1 W M) (h2 : condition2 W M) : M = 16.65 :=
sorry

end Nina_has_16dollars65_l144_144406


namespace length_PF_is_8_l144_144907

noncomputable def parabola : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y^2 = 8 * (x + 2)}

def focus : ℝ × ℝ := (-2, 0)

def line_through_focus_angle (θ : ℝ) :  ℝ × ℝ → ℝ × ℝ → Prop := 
  λ p q, (p.2 - q.2) = tan θ * (p.1 - q.1)

def chord_midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

def perpendicular_bisector (p q : ℝ × ℝ) :  ℝ × ℝ → ℝ × ℝ → Prop :=
  let m := chord_midpoint p q in
  λ p q, (p.2 - q.2) = - (m.1 / m.2) * (p.1 - q.1)

theorem length_PF_is_8 :
  ∀ P : ℝ × ℝ, 
  ∃ A B : ℝ × ℝ,
  A ∈ parabola ∧ B ∈ parabola ∧
  line_through_focus_angle (π/3) focus A ∧
  line_through_focus_angle (π/3) focus B ∧
  perpendicular_bisector A B P ∧
  P.2 = 0 → (dist focus P = 8) :=
begin
  sorry
end

end length_PF_is_8_l144_144907


namespace some_number_is_l144_144336

theorem some_number_is (x some_number : ℤ) (h1 : x = 4) (h2 : 5 * x + 3 = 10 * x - some_number) : some_number = 17 := by
  sorry

end some_number_is_l144_144336


namespace non_factorial_tail_up_to_1500_l144_144977

def is_factorial_tail (n m : ℕ) : Prop :=
  n = (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) + (m / 78125) + (m / 390625) + (m / 1953125) + (m / 9765625) + (m / 48828125) + (m / 244140625) + (m / 1220703125) + (m / 6103515625)

def factorial_tail_count (n m : ℕ) : Prop :=
  ∃ m : ℕ, is_factorial_tail n m

def non_factorial_tail_count (N : ℕ) : Prop :=
  (finset.range N).filter (λ n, ¬ factorial_tail_count n N).card = 300

theorem non_factorial_tail_up_to_1500 : non_factorial_tail_count 1500 :=
by sorry

end non_factorial_tail_up_to_1500_l144_144977


namespace elise_hospital_distance_l144_144215

noncomputable def distance_to_hospital (total_fare: ℝ) (base_price: ℝ) (toll_price: ℝ) 
(tip_percent: ℝ) (cost_per_mile: ℝ) (increase_percent: ℝ) (toll_count: ℕ) : ℝ :=
let base_and_tolls := base_price + (toll_price * toll_count)
let fare_before_tip := total_fare / (1 + tip_percent)
let distance_fare := fare_before_tip - base_and_tolls
let original_travel_fare := distance_fare / (1 + increase_percent)
original_travel_fare / cost_per_mile

theorem elise_hospital_distance : distance_to_hospital 34.34 3 2 0.15 4 0.20 3 = 5 := 
sorry

end elise_hospital_distance_l144_144215


namespace average_increase_l144_144894

-- Define the conditions as Lean definitions
def runs_in_17th_inning : ℕ := 50
def average_after_17th_inning : ℕ := 18

-- The condition about the average increase can be written as follows
theorem average_increase 
  (initial_average: ℕ) -- The batsman's average after the 16th inning
  (h1: runs_in_17th_inning = 50)
  (h2: average_after_17th_inning = 18)
  (h3: 16 * initial_average + runs_in_17th_inning = 17 * average_after_17th_inning) :
  average_after_17th_inning - initial_average = 2 := 
sorry

end average_increase_l144_144894


namespace sum_a_is_10000_l144_144002

-- Define the sequence a_n with the given recurrence relation and initial condition
noncomputable def a : ℕ → ℝ
| 1 := 1
| n := if n = 100 then 199 else a (n - 1) * (n - 1) / (n - 2) - 1 / (n - 2) 

-- Prove that the sum of the sequence from 1 to 100 is 10000 under the given conditions
theorem sum_a_is_10000 (h : ∀ n, 2 ≤ n ∧ n ≤ 100 → (n - 2) * a n - (n - 1) * a (n - 1) + 1 = 0)
: (∑ i in finset.range 100, a (i + 1)) = 10000 := 
begin
  sorry
end

end sum_a_is_10000_l144_144002


namespace find_fake_coin_l144_144109

theorem find_fake_coin (k : ℕ) :
  ∃ (weighings : ℕ), (weighings ≤ 3 * k + 1) :=
sorry

end find_fake_coin_l144_144109


namespace octagon_quadrilateral_area_l144_144696

open Real -- Use real number operations

-- Define the problem
theorem octagon_quadrilateral_area (a : ℝ) (h : a = 2) : 
     let s := 4 * (1 + sqrt 2) / 3, 
         q := (3 / 4) * s in
     q * q = 3 + 2 * sqrt 2 :=
by
   sorry

end octagon_quadrilateral_area_l144_144696


namespace imaginary_roots_of_quadratic_l144_144605

noncomputable def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

theorem imaginary_roots_of_quadratic (k : ℚ) :
  (let a := 3
       b := -5 * k
       c := 4 * k^2 - 2 in
   (4 * k^2 - 2 = 27 / 3) → discriminant a b c < 0) → 
   - (27 * (29 / 4) - 24) / 4 < 0 := 
by 
  intros h
  have k_value : 4 * k^2 - 2 = 27 / 3 := sorry
  have h_discriminant : discriminant 3 (-5 * k) (4 * k^2 - 2) < 0 := sorry
  exact h_discriminant

end imaginary_roots_of_quadratic_l144_144605


namespace inequality_solution_l144_144238

theorem inequality_solution (x : ℝ) :
  (3 / 16) + abs (x - 17 / 64) < 7 / 32 ↔ (15 / 64) < x ∧ x < (19 / 64) :=
by
  sorry

end inequality_solution_l144_144238


namespace area_enclosed_by_curve_l144_144786

theorem area_enclosed_by_curve :
  let arc_length := (3 * Real.pi) / 4
  let side_length := 3
  let radius := arc_length / ((3 * Real.pi) / 4)
  let sector_area := (radius ^ 2 * Real.pi * (3 * Real.pi) / (4 * 2 * Real.pi))
  let total_sector_area := 8 * sector_area
  let octagon_area := 2 * (1 + Real.sqrt 2) * (side_length ^ 2)
  total_sector_area + octagon_area = 54 + 54 * Real.sqrt 2 + 3 * Real.pi
:= sorry

end area_enclosed_by_curve_l144_144786


namespace smallest_geometric_sequence_number_l144_144509

theorem smallest_geometric_sequence_number :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
    (∀ d ∈ [((n / 100) % 10), ((n / 10) % 10), (n % 10)], d ∈ [1,2,3,4,5,6,7,8,9]) ∧
    (let digits := [((n / 100) % 10), ((n / 10) % 10), (n % 10)] in
       digits.nodup ∧
       ∃ r : ℕ, r > 1 ∧ digits = [digits.head!, digits.head! * r, digits.head! * r * r]) ∧
    n = 124 :=
begin
  sorry
end

end smallest_geometric_sequence_number_l144_144509


namespace total_distance_run_l144_144052

theorem total_distance_run (length : ℕ) (width : ℕ) (laps : ℕ) (h_length : length = 100) (h_width : width = 50) (h_laps : laps = 6) : 
  let perimeter := 2 * length + 2 * width in
  let distance := laps * perimeter in
  distance = 1800 :=
by
  sorry

end total_distance_run_l144_144052


namespace exercise_l144_144779

noncomputable def problem : Prop :=
  ∃ (s x y : ℝ), 
    0 < s ∧ 0 < x ∧ 0 < y ∧
    0.25 * s * s = 0.4 * x * y ∧ 
    y = s ∧ 
    (LM LP : ℝ) → LM = x ∧ LP = y → LM / LP = 5 / 8

theorem exercise : problem := 
sorry

end exercise_l144_144779


namespace box_volume_l144_144061

theorem box_volume (x y z : ℕ) 
  (h1 : 2 * x + 2 * y = 26)
  (h2 : x + z = 10)
  (h3 : y + z = 7) :
  x * y * z = 80 :=
by
  sorry

end box_volume_l144_144061


namespace jana_walking_distance_l144_144367

theorem jana_walking_distance (t_walk_mile : ℝ) (speed : ℝ) (time : ℝ) (distance : ℝ) :
  t_walk_mile = 24 → speed = 1 / t_walk_mile → time = 36 → distance = speed * time → distance = 1.5 :=
by
  intros h1 h2 h3 h4
  sorry

end jana_walking_distance_l144_144367


namespace n_must_be_even_l144_144062

open Nat

-- Define the system of equations:
def equation (n : ℕ) (x : ℕ → ℤ) : Prop :=
  (∀ i, 2 ≤ i ∧ i ≤ n - 1 → (-x (i-1) + 2 * x i - x (i+1) = 1)) ∧
  (2 * x 1 - x 2 = 1) ∧
  (∀ i, 1 ≤ i ∧ i ≤ n → x i > 0)

-- Define the last equation separately due to its unique form:
def last_equation (n : ℕ) (x : ℕ → ℤ) : Prop :=
  (n ≥ 1 → -x (n-1) + 2 * x n = 1)

-- The theorem to prove that n must be even:
theorem n_must_be_even (n : ℕ) (x : ℕ → ℤ) : 
  equation n x → last_equation n x → Even n :=
by
  intros h₁ h₂
  sorry

end n_must_be_even_l144_144062


namespace product_of_min_max_values_l144_144745

variable (x y : ℝ)

theorem product_of_min_max_values :
  (9 * x^2 + 12 * x * y + 8 * y^2 = 1) →
  let k := 3 * x^2 + 4 * x * y + 3 * y^2 in
  let m := inf {k : ℝ | ∃ (x y : ℝ), 9 * x^2 + 12 * x * y + 8 * y^2 = 1 ∧ k = 3 * x^2 + 4 * x * y + 3 * y^2} in
  let M := sup {k : ℝ | ∃ (x y : ℝ), 9 * x^2 + 12 * x * y + 8 * y^2 = 1 ∧ k = 3 * x^2 + 4 * x * y + 3 * y^2} in
  m * M = 11555.9025 / 82944 := by sorry

end product_of_min_max_values_l144_144745


namespace junior_high_ten_total_games_l144_144434

theorem junior_high_ten_total_games :
  let teams := 10
  let conference_games_per_team := 3
  let non_conference_games_per_team := 5
  let pairs_of_teams := Nat.choose teams 2
  let total_conference_games := pairs_of_teams * conference_games_per_team
  let total_non_conference_games := teams * non_conference_games_per_team
  let total_games := total_conference_games + total_non_conference_games
  total_games = 185 :=
by
  sorry

end junior_high_ten_total_games_l144_144434


namespace mutually_exclusive_event_l144_144165

def hit_event_A := ∀ (shots : ℕ), shots <= 1
def hit_event_B := ∀ (shots : ℕ), shots ≥ 1 ∧ shots ≤ 2
def hit_event_C := ∀ (shots : ℕ), shots ≥ 1
def hit_event_D := ∀ (shots : ℕ), shots = 0
def hit_event_E := ∀ (shots : ℕ), shots = 1

theorem mutually_exclusive_event : 
  (∀ shots, hit_event_D shots → ¬hit_event_E shots) ∧
  (∀ shots, hit_event_E shots → ¬hit_event_D shots) ∧
  ∃ (shots : ℕ), hit_event_E shots ∧ ¬hit_event_D shots := by
  sorry

end mutually_exclusive_event_l144_144165


namespace sum_of_groups_for_eight_candies_l144_144874

noncomputable def p : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 5
| 4 := 15
| 5 := 52
| 6 := 203
| 7 := 877
| 8 := 4140
| 9 := 21147
| 10 := 115975
| _ := 0 -- for the purpose of this problem, we only care about p(1) to p(10)

-- Proof statement
theorem sum_of_groups_for_eight_candies :
  ∑ k in finset.range 9, k * (q(8, k)) = 17007 :=
by 
  -- q(8, k) is not explicitly defined here since it's indirectly used through p(n)
  have h1: ∑ k in finset.range 9, q(8, k) = p(8), { sorry },
  have h2: p(9) - p(8) = 17007, { sorry },
  have h3: ∑ k in finset.range 9, k * q(8, k) = p(9) - p(8), { sorry },
  exact h3.trans h2

end sum_of_groups_for_eight_candies_l144_144874


namespace quadratic_one_solution_interval_l144_144707

theorem quadratic_one_solution_interval (a : ℝ) :
  2ax^2 - x - 1 = 0 ↔ (∀ x : ℝ, 0 < x ∧ x < 1 → (2a - 2 > 0)) := by
  sorry

end quadratic_one_solution_interval_l144_144707


namespace tangent_line_at_1_intervals_of_monotonicity_and_extrema_l144_144678

open Real

noncomputable def f (x : ℝ) := 6 * log x + (1 / 2) * x^2 - 5 * x

theorem tangent_line_at_1 :
  let f' (x : ℝ) := (6 / x) + x - 5
  (f 1 = -9 / 2) →
  (f' 1 = 2) →
  (∀ x y : ℝ, y + 9 / 2 = 2 * (x - 1) → 4 * x - 2 * y - 13 = 0) := 
by
  sorry

theorem intervals_of_monotonicity_and_extrema :
  let f' (x : ℝ) := (x^2 - 5 * x + 6) / x
  (∀ x, 0 < x ∧ x < 2 → f' x > 0) → 
  (∀ x, 3 < x → f' x > 0) →
  (∀ x, 2 < x ∧ x < 3 → f' x < 0) →
  (f 2 = -8 + 6 * log 2) →
  (f 3 = -21 / 2 + 6 * log 3) :=
by
  sorry

end tangent_line_at_1_intervals_of_monotonicity_and_extrema_l144_144678


namespace cost_price_USD_l144_144567

-- Assume the conditions in Lean as given:
variable {C_USD : ℝ}

def condition1 (C_USD : ℝ) : Prop := 0.9 * C_USD + 200 = 1.04 * C_USD

theorem cost_price_USD (h : condition1 C_USD) : C_USD = 200 / 0.14 :=
by
  sorry

end cost_price_USD_l144_144567


namespace infinite_coprime_exists_l144_144380

def polynomial (P : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, P(x + y) = P(x) + P(y) ∧ P(x * y) = P(x) * P(y)

noncomputable def sequence (P : ℤ → ℤ) : ℕ → ℤ
| 0       := 0
| (n + 1) := P (sequence n)

theorem infinite_coprime_exists (P : ℤ → ℤ) (c : ℤ) (hP : polynomial P) (hP0 : P 0 = 1) (hc : c > 1) :
  ∃^∞ n : ℕ, Int.gcd (sequence P n) (n + c) = 1 := 
sorry

end infinite_coprime_exists_l144_144380


namespace clock_hands_angle_120_l144_144984

theorem clock_hands_angle_120 (t : ℝ) :
  (7 <= t) ∧ (t < 8) ∧ ((t = 7 + 16/60) ∨ (t = 7 + 27/60)) →
  (let hour_angle := 210 + 30 * (t - 7),
       minute_angle := 360 * (t - 7),
       angle_diff := |hour_angle - minute_angle| % 360 in
   angle_diff = 120 ∨ angle_diff = 240) :=
begin
  sorry
end

end clock_hands_angle_120_l144_144984


namespace equation_satisfies_solution_l144_144517

theorem equation_satisfies_solution (x y : ℤ) (h₁ : x = 2) (h₂ : y = -1) : 3 * x - 4 * y = 10 := by
  rw [h₁, h₂]
  norm_num
  sorry

end equation_satisfies_solution_l144_144517


namespace sum_of_sequence_l144_144724

theorem sum_of_sequence {n : ℕ} (h1 : a₁ = 1)
  (h2 : ∀ k : ℕ, 1 ≤ k → S (k + 1) = ((k + 1) * a k / k) + S k) :
  S n = n * (n + 1) / 2 :=
by
  sorry

end sum_of_sequence_l144_144724


namespace three_lines_through_one_point_l144_144610

theorem three_lines_through_one_point (lines : Fin 9 → Line) (square : Square)
  (H : ∀ i, divides_in_ratio lines[i] square (2/5) (3/5)) : 
  ∃ P : Point, ∃ I J K : Fin 9, I ≠ J ∧ J ≠ K ∧ K ≠ I ∧ 
  (lines[I].passes_through P ∧ lines[J].passes_through P ∧ lines[K].passes_through P) :=
sorry

end three_lines_through_one_point_l144_144610


namespace green_chips_count_l144_144474

-- Definitions of given conditions
def total_chips (total : ℕ) : Prop :=
  ∃ blue_chips white_percentage, blue_chips = 3 ∧ (blue_chips : ℚ) / total = 10 / 100 ∧ white_percentage = 50 / 100 ∧
  let green_percentage := 1 - (10 / 100 + white_percentage) in
  green_percentage * total = 12

-- Proposition to prove the number of green chips equals 12
theorem green_chips_count (total : ℕ) (h : total_chips total) : ∃ green_chips, green_chips = 12 := 
by 
  sorry

end green_chips_count_l144_144474


namespace problem1_l144_144596

theorem problem1 :
  0.064 ^ (-1 / 3) - (-1 / 8 : ℝ) ^ 0 + 16 ^ (3 / 4) + 0.25 ^ (1 / 2) = 10 :=
by
  sorry

end problem1_l144_144596


namespace probability_four_distinct_numbers_l144_144480

theorem probability_four_distinct_numbers (prob : ℚ) (h : prob = 325 / 648) :
  let total_outcomes := 6 ^ 6 in
  let case1_outcomes := 6 * 20 * 10 * 6 in
  let case2_outcomes := 15 * 15 * 6 * 6 * 2 in
  prob = (case1_outcomes + case2_outcomes) / total_outcomes := by
  sorry

end probability_four_distinct_numbers_l144_144480


namespace num_distinct_ordered_pairs_l144_144668

theorem num_distinct_ordered_pairs (a b c : ℕ) (h₀ : a + b + c = 50) (h₁ : c = 10) (h₂ : 0 < a ∧ 0 < b) :
  ∃ n : ℕ, n = 39 := 
sorry

end num_distinct_ordered_pairs_l144_144668


namespace tangent_line_eq_l144_144388

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem tangent_line_eq
  (a b : ℝ)
  (h1 : 3 + 2*a + b = 2*a)
  (h2 : 12 + 4*a + b = -b)
  : ∀ x y : ℝ , (f a b 1 = -5/2 ∧
  y - (f a b 1) = -3 * (x - 1))
  → (6*x + 2*y - 1 = 0) :=
by
  sorry

end tangent_line_eq_l144_144388


namespace min_max_S_l144_144582

open Finset
open BigOperators

noncomputable def S (x : fin 10 → ℕ) : ℕ :=
  x 1 + x 2 + x 3 + x 4 + x 6 + x 7 + x 8

theorem min_max_S:
  ∀ (x : fin 10 → ℕ),
    Multiset.sort (Multiset.map x (Finset.univ : Finset (fin 10))) = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] →
    21 ≤ S x ∧ S x ≤ 25 :=
by
  sorry

end min_max_S_l144_144582


namespace seat_adjustment_schemes_l144_144349

theorem seat_adjustment_schemes {n k : ℕ} (h1 : n = 7) (h2 : k = 3) :
  (2 * Nat.choose n k) = 70 :=
by
  -- n is the number of people, k is the number chosen
  rw [h1, h2]
  -- the rest is skipped for the statement only
  sorry

end seat_adjustment_schemes_l144_144349


namespace three_digit_numbers_with_distinct_digits_avg_condition_l144_144694

theorem three_digit_numbers_with_distinct_digits_avg_condition : 
  ∃ (S : Finset (Fin 1000)), 
  (∀ n ∈ S, (n / 100 ≠ (n / 10 % 10) ∧ (n / 100 ≠ n % 10) ∧ (n / 10 % 10 ≠ n % 10))) ∧
  (∀ n ∈ S, ((n / 100 + n % 10) / 2 = n / 10 % 10)) ∧
  (∀ n ∈ S, abs ((n / 100) - (n / 10 % 10)) ≤ 5 ∧ abs ((n / 10 % 10) - (n % 10)) ≤ 5) ∧
  S.card = 120 :=
sorry

end three_digit_numbers_with_distinct_digits_avg_condition_l144_144694


namespace measure_angle_ZYV_l144_144886

theorem measure_angle_ZYV (X Y Z V : Type)
  (XZ_eq_YZ : XZ = YZ)
  (angle_Z_eq_50 : m∠ Z = 50) :
  m∠ ZYV = 115 := 
begin
  sorry
end

end measure_angle_ZYV_l144_144886


namespace initial_position_is_correct_l144_144579

noncomputable def initial_position : ℤ :=
  let a := -20 in
  let K₀ := a in
  let K₁ := K₀ - 1 in
  let K₂ := K₁ + 2 in
  let K₃ := K₂ - 3 in
  let K₄ := K₃ + 4 in
  -- This pattern continues up to K₁₀₀, we need to confirm K₁₀₀ = 30
  let K₁₀₀ := K₀ + (List.sum (List.map (λ i, if i % 2 = 0 then i else -i) (List.range 101)) : ℤ) in
  K₁₀₀

theorem initial_position_is_correct : initial_position = 30 :=
by 
  have sum_alternating_steps : (List.sum (List.map (λ i, if i % 2 = 0 then i else -i) (List.range 101)) : ℤ) = 50 :=
    sorry -- The sum of alternating series calculation should be here
  show initial_position = 30 from
  by
    rw [initial_position, sum_alternating_steps]
    simp
    sorry -- Remaining part of the proof

end initial_position_is_correct_l144_144579


namespace correct_statement_l144_144521

-- Define each condition as a proposition
def statement_A : Prop :=
  ∀ (event : String), ¬ (event = "weather forecast")

def statement_B : Prop :=
  ∀ (students : Type) (is_class : students → Prop), ∃ sample, ∀ s, is_class s → sample s

def statement_C : Prop :=
  ∀ (population : Type) (samples : ℕ → population → Prop), 
    (∀ n1 n2, n1 < n2 → ∀ (sample : population → Prop), samples n2 sample → samples n1 sample)

def statement_D : Prop :=
  ∃ (A B : Type), ∃ (S_A² S_B² : ℕ), S_A² = 2 ∧ S_B² = 1 ∧ stability B S_B² > stability A S_A²

-- Define the theorem to prove that statement C is the correct one
theorem correct_statement : statement_A → statement_B → statement_C → statement_D → statement_C := by
  intro A B C D
  exact C

end correct_statement_l144_144521


namespace square_complex_number_l144_144334
open Complex

theorem square_complex_number :
  let z := 5 + 2 * Complex.i
  z^2 = 21 + 20 * Complex.i := by
  sorry

end square_complex_number_l144_144334


namespace other_x_intercept_l144_144946

noncomputable def ellipse_x_intercepts (f1 f2 : ℝ × ℝ) (x_intercept1 : ℝ × ℝ) : ℝ × ℝ :=
  let d := dist f1 x_intercept1 + dist f2 x_intercept1
  let x := (d^2 - 2 * d * sqrt (3^2 + (d / 2 - 4)^2)) / (2 * d - 8)
  (x, 0)

theorem other_x_intercept :
  ellipse_x_intercepts (0, 3) (4, 0) (0, 0) = (56 / 11, 0) :=
by
  sorry

end other_x_intercept_l144_144946


namespace remainder_of_powers_l144_144498

theorem remainder_of_powers (n1 n2 n3 : ℕ) : (9^6 + 8^8 + 7^9) % 7 = 2 :=
by
  sorry

end remainder_of_powers_l144_144498


namespace cost_price_of_article_l144_144455

theorem cost_price_of_article 
  (final_sale_price : ℝ)
  (sales_tax_rate : ℝ)
  (discount_rate : ℝ)
  (processing_fee_rate : ℝ)
  (profit_rate : ℝ)
  (cost_price : ℝ)
  (h1 : final_sale_price = 650)
  (h2 : sales_tax_rate = 0.10)
  (h3 : discount_rate = 0.02)
  (h4 : processing_fee_rate = 0.05)
  (h5 : profit_rate = 0.14)
  (h6 : final_sale_price = (1 + processing_fee_rate) * (1 + sales_tax_rate - discount_rate) * (1 + profit_rate) * cost_price) :
  cost_price ≈ 503.12 := 
by 
  sorry

end cost_price_of_article_l144_144455


namespace chromatic_number_graph_P_l144_144003

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, 1 < m → m < n → n % m ≠ 0

def has_edge (a b : ℕ) : Prop :=
  a < b ∧ is_prime (b / a)

def chromatic_number (P : Type) [Graph P] : ℕ :=
  Inf {n | ∃ f : P → Fin n, ∀ v₁ v₂, Graph.adj v₁ v₂ → f v₁ ≠ f v₂ }

noncomputable def graph_P : Type := ℕ

instance graph_P_graph : Graph graph_P :=
  { adj := λ a b, a < b ∧ is_prime (b / a),
    symm := λ a b ⟨hab, h⟩, and.intro (lt_trans hab (nat.le_of_div_mul hab)) (is_prime_of_prime_div h) }

theorem chromatic_number_graph_P : chromatic_number graph_P = 2 := sorry

end chromatic_number_graph_P_l144_144003


namespace tangent_parabola_line_l144_144795

theorem tangent_parabola_line (a : ℝ) :
  (∃ x : ℝ, ax^2 + 1 = x ∧ ∀ y : ℝ, (y = ax^2 + 1 → y = x)) ↔ a = 1/4 :=
by
  sorry

end tangent_parabola_line_l144_144795


namespace find_y_value_find_y_at_8_l144_144296

theorem find_y_value (k : ℝ) (y : ℝ) (x : ℝ) (h1 : y = k * x^(1 / 3)) (h2 : y = 5 * real.sqrt 2) (h3 : x = 64) :
  y = 2.5 * real.sqrt 2 :=
by
  sorry

theorem find_y_at_8 (y : ℝ) (k : ℝ) (h1 : k = (5 * real.sqrt 2) / 4) (h2 : y = k * 8^(1 / 3)) :
  y = 2.5 * real.sqrt 2 :=
by
  sorry

end find_y_value_find_y_at_8_l144_144296


namespace correct_statements_l144_144266

open Classical

variables {α l m n p : Type*}
variables (is_perpendicular_to : α → α → Prop) (is_parallel_to : α → α → Prop)
variables (is_in_plane : α → α → Prop)

noncomputable def problem_statement (l : α) (α : α) : Prop :=
  (∀ m, is_perpendicular_to m l → is_parallel_to m α) ∧
  (∀ m, is_perpendicular_to m α → is_parallel_to m l) ∧
  (∀ m, is_parallel_to m α → is_perpendicular_to m l) ∧
  (∀ m, is_parallel_to m l → is_perpendicular_to m α)

theorem correct_statements (l : α) (α : α) (h_l_α : is_perpendicular_to l α) :
  (∀ m, is_perpendicular_to m α → is_parallel_to m l) ∧
  (∀ m, is_parallel_to m α → is_perpendicular_to m l) ∧
  (∀ m, is_parallel_to m l → is_perpendicular_to m α) :=
sorry

end correct_statements_l144_144266


namespace triangle_JKL_area_l144_144999

open Real

-- Definitions used in the problem
def JKL : Type := {J : Point ℝ // True}
def K : Point ℝ := ⟨sqrt 3, 0⟩
def L : Point ℝ := ⟨0, 1⟩

-- Main theorem statement
theorem triangle_JKL_area (h₁ : angle J K L = 90) 
  (h₂ : angle L J K = 60) 
  (KL : dist K L = 20) : 
  area (triangle J K L) = 50 * sqrt 3 := 
sorry

end triangle_JKL_area_l144_144999


namespace value_of_expression_l144_144664

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  (a + b + c + d).sqrt + (a^2 - 2*a + 3 - b).sqrt - (b - c^2 + 4*c - 8).sqrt = 3

theorem value_of_expression (a b c d : ℝ) (h : proof_problem a b c d) : a - b + c - d = -7 :=
sorry

end value_of_expression_l144_144664


namespace integer_solution_x_l144_144208

theorem integer_solution_x (x : ℤ) (h₁ : x + 8 > 10) (h₂ : -3 * x < -9) : x ≥ 4 ↔ x > 3 := by
  sorry

end integer_solution_x_l144_144208


namespace value_of_expression_l144_144250

theorem value_of_expression (x y z : ℝ) (h : x * y * z = 1) :
  1 / (1 + x + x * y) + 1 / (1 + y + y * z) + 1 / (1 + z + z * x) = 1 :=
sorry

end value_of_expression_l144_144250


namespace medicine_supply_duration_l144_144736

-- Define the conditions
def rate_of_consumption : ℕ := 3  -- days per pill
def total_pills : ℕ := 90
def days_per_month : ℕ := 30

-- Define the duration the supply will last in days
def total_days : ℕ := total_pills * rate_of_consumption

-- Define the expected duration in months, and prove it will last 9 months
theorem medicine_supply_duration : total_days // days_per_month = 9 :=
by
  -- sorry is used to skip the proof
  sorry

end medicine_supply_duration_l144_144736


namespace log_base4_one_over_sixty_four_l144_144230

theorem log_base4_one_over_sixty_four : log 4 (1 / 64) = -3 := by
  sorry

end log_base4_one_over_sixty_four_l144_144230


namespace max_red_rows_and_cols_l144_144347

-- Definition of the problem conditions:
def is_red (arr : matrix (fin 7) (fin 7) bool) (r : ℕ) (is_row : bool) : Prop :=
  if is_row then (finset.card (finset.filter (λ c, arr ⟨r, c⟩) finset.univ) ≥ 4)
  else (finset.card (finset.filter (λ c, arr ⟨c, r⟩) finset.univ) ≥ 4)

-- Definition of the main problem theorem:
theorem max_red_rows_and_cols (arr : matrix (fin 7) (fin 7) bool) (h : finset.card (finset.filter (λ rc, arr rc) finset.univ) = 19) :
  (finset.card (finset.filter (λ r, is_red arr r tt) finset.univ) +
   finset.card (finset.filter (λ c, is_red arr c ff) finset.univ) ≤ 8) :=
sorry

end max_red_rows_and_cols_l144_144347


namespace sum_of_ceil_sqrt_l144_144616

/-- 
  We are given that for the range [10, 50], the sum of ceil(sqrt(n)) is to be found.
  We want to prove that this sum is equal to 238. 
-/
theorem sum_of_ceil_sqrt (sqrt: ℝ → ℝ) (ceil: ℝ → ℤ): 
  (∀ x: ℝ, ceil x = ⌈x⌉) →
  ∑ n in finset.range (51), if 10 ≤ n ∧ n ≤ 50 then ceil (sqrt (↑n)) else 0 = 238 := 
by
  sorry

end sum_of_ceil_sqrt_l144_144616


namespace value_of_EA_plus_EB_l144_144714

noncomputable def polarToRectangularEquation : Real :=
  let C := (x - 1)^2 + (y - 1)^2 = 2
  have rho := 2 * (cos theta + sin theta)
  have polarEq : ∀ (theta : Real), rho^2 = 2 * rho * cos theta + 2 * rho * sin theta :=
    by sorry
  have rectangularEq : ∀ (x y : Real), x^2 + y^2 = 2 * x + 2 * y :=
    by sorry
  (x - 1)^2 + (y - 1)^2 = 2

noncomputable def lineParametricEq(t : Real) : (Real × Real) :=
  (1/2 * t, 1 + √3/2 * t)

theorem value_of_EA_plus_EB : Real :=
  let t1 := (1 + √5) / 2
  let t2 := (1 - √5) / 2
  let A := lineParametricEq t1
  let B := lineParametricEq t2
  let E : (Real × Real) := (0, 1)
  let dist (P Q : Real × Real) := Math.sqrt ((P.fst - Q.fst)^2 + (P.snd - Q.snd)^2)
  dist E A + dist E B = √5 :=
  by sorry

end value_of_EA_plus_EB_l144_144714


namespace luke_weed_eating_income_l144_144011

theorem luke_weed_eating_income :
  ∀ (W : ℕ), (9 + W = 27) → W = 18 :=
by
  intro W h
  rw [← Nat.add_right_cancel_iff] at h
  exact h
  sorry

end luke_weed_eating_income_l144_144011


namespace Ap_perp_BC_l144_144393

open EuclideanGeometry
open_locale EuclideanGeometry

theorem Ap_perp_BC
  (A B C I M N D E P : Point)
  (h1 : incenter I (Triangle A B C))
  (h2 : midpoint M A B)
  (h3 : midpoint N A C)
  (h4 : line_perpendicular D (line_through I M))
  (h5 : line_perpendicular E (line_through I N))
  (h6 : line_intersection (line_through D ⟂ (line_through I M)) (line_through E ⟂ (line_through I N)) P)
  (h7 : dist B D = dist C E = dist B C):
  perpendicular (line_through A P) (line_through B C) :=
sorry

end Ap_perp_BC_l144_144393


namespace middle_number_is_nine_l144_144540

theorem middle_number_is_nine (x : ℝ) (h : (2 * x)^2 + (4 * x)^2 = 180) : 3 * x = 9 :=
by
  sorry

end middle_number_is_nine_l144_144540


namespace mrs_late_on_time_l144_144405

variables (d t : ℝ)

/-- The problem conditions:
  - Mrs. Late drives at 30 mph and is 4 minutes (1/15 hour) late.
  - Mrs. Late drives at 50 mph and is 2 minutes (1/30 hour) early.
  - Determine the required speed to arrive on time.
-/
theorem mrs_late_on_time (d_ideally: d = 30 * (t + 1 / 15)) (d_quickly: d = 50 * (t - 1 / 30)) :
  let r := d / t in
  r = 41 :=
by
  sorry

end mrs_late_on_time_l144_144405


namespace gcd_of_factorials_l144_144853

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definitions for the two factorial terms involved
def term1 : ℕ := factorial 7
def term2 : ℕ := factorial 10 / factorial 4

-- Definition for computing the GCD
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- The statement to prove
theorem gcd_of_factorials : gcd term1 term2 = 2520 :=
by {
  -- Here, the proof would go.
  sorry
}

end gcd_of_factorials_l144_144853


namespace riding_hours_l144_144191

theorem riding_hours (rides_per_week : ℕ) :
  (rides_per_week / 2) * 4 + 2 * 2 + rides_per_week = 12 → (rides_per_week / 2) * 3 = 6 :=
begin
  sorry
end

end riding_hours_l144_144191


namespace equality_AB_BZ_l144_144731

noncomputable section

variables {A B C D I X Y Z : Type}
variables (h_triangle : ∀ {A B C : Type}, Type)
variables (h_incircle : ∀ {I : Type} [incircle I], I.touches B C at D)
variables (h_line_DI : ∀ {D I : Type} [line_DI D I], (D I).intersects A C at X)
variables (h_tangent : ∀ {X : Type} [tangent X I], tangent.to_incircle X.intersects A B at Y)
variables (h_intersection_YI : ∀ {Y I : Type} [Y_line_intersect I], YI.intersects B C at Z)

theorem equality_AB_BZ (h_triangle : ∀ ⦃A B C : Type⦄, Type)
  (h_incircle : ∀ ⦃I : Type⦄ [incircle I], I.touches B C at D)
  (h_line_DI : ∀ ⦃D I : Type⦄ [line_DI D I], (D I).intersects A C at X)
  (h_tangent : ∀ ⦃X : Type⦄ [tangent X I], tangent.to_incircle X.intersects A B at Y)
  (h_intersection_YI : ∀ ⦃Y I : Type⦄ [Y_line_intersect I], YI.intersects B C at Z) :
  AB = BZ := sorry

end equality_AB_BZ_l144_144731


namespace polygon_partition_numbering_l144_144534

structure ConvexPolygon (n : ℕ) :=
  (vertices : Fin (n + 1) → Point)
  (is_convex : convex (set.range vertices))

-- Definition of non-intersecting diagonals
def non_intersecting_diagonals (P : ConvexPolygon n) (d : set (Fin (n+1) × Fin (n+1))) : Prop :=
  2*|d| = n-2 ∧ ∀ (a b : Fin (n+1) × Fin (n+1)), (a ∈ d ∧ b ∈ d) → non_intersecting a b

-- Definition of partition into triangles
def partition_into_triangles (P: ConvexPolygon n) (triangles : Fin (n-1) → triangle (Fin (n+1))) : Prop :=
  (∀ i, P.vertices i ∈ triangles)

theorem polygon_partition_numbering (n : ℕ) (P : ConvexPolygon n) (d : set (Fin (n+1) × Fin (n+1))) (ht : non_intersecting_diagonals P d) :
  ∃ triangles : Fin (n-1) → triangle (Fin (n+1)),
    partition_into_triangles P triangles ∧
    (∀ i, P.vertices (i+1) ∈ triangles i) :=
begin
  sorry
end

end polygon_partition_numbering_l144_144534


namespace chord_length_circle_line_l144_144050

theorem chord_length_circle_line 
  (t : ℝ )
  (x y : ℝ)
  (h1 : x = -2 + t)
  (h2 : y = 1 - t)
  (h3 : (x - 3)^2 + (y + 1)^2 = 25) :
  (chord_length : ℝ), chord_length = sqrt 82 :=
sorry

end chord_length_circle_line_l144_144050


namespace correct_option_C_l144_144690

variables {a b : Line} {α : Plane}

theorem correct_option_C (h1 : a ⊥ α) (h2 : b ∥ α) : a ⊥ b :=
sorry

end correct_option_C_l144_144690


namespace gcd_of_factorials_l144_144851

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definitions for the two factorial terms involved
def term1 : ℕ := factorial 7
def term2 : ℕ := factorial 10 / factorial 4

-- Definition for computing the GCD
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- The statement to prove
theorem gcd_of_factorials : gcd term1 term2 = 2520 :=
by {
  -- Here, the proof would go.
  sorry
}

end gcd_of_factorials_l144_144851


namespace square_area_from_adjacent_points_l144_144016

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def is_adjacent_square_points (p1 p2 : ℝ × ℝ) (d : ℝ) : Prop :=
  distance p1 p2 = d

theorem square_area_from_adjacent_points :
  ∀ (p1 p2 : ℝ × ℝ) (d : ℝ),
  is_adjacent_square_points p1 p2 d →
  p1 = (1, 2) →
  p2 = (4, 6) →
  d = 5 →
  d ^ 2 = 25 :=
by
  intros p1 p2 d h_adj h_p1 h_p2 h_d
  rw [h_d]
  rfl

end square_area_from_adjacent_points_l144_144016


namespace trisha_cookies_count_l144_144256

noncomputable def number_of_trisha_cookies : ℕ :=
  let pi := Real.pi
  let sqrt := Real.sqrt
  let art_cookie_area : ℝ := pi * 4^2
  let trisha_cookie_area : ℝ := (sqrt 3 / 4) * 6^2
  let total_dough : ℝ := 10 * art_cookie_area
  (total_dough / trisha_cookie_area).to_nat

theorem trisha_cookies_count : number_of_trisha_cookies = 30 := 
by 
  sorry

end trisha_cookies_count_l144_144256


namespace find_inverse_of_512_l144_144702

-- Define the function f with the given properties
def f : ℕ → ℕ := sorry

axiom f_initial : f 5 = 2
axiom f_property : ∀ x, f (2 * x) = 2 * f x

-- State the problem as a theorem
theorem find_inverse_of_512 : ∃ x, f x = 512 ∧ x = 1280 :=
by 
  -- Sorry to skip the proof
  sorry

end find_inverse_of_512_l144_144702


namespace floor_T_sq_eq_l144_144396

def sum_sqrt_series (n : ℕ) : ℝ :=
  ∑ j in Finset.range (n - 1), sqrt (1 + (1 : ℝ) / (j + 2)^2 + 1 / (j + 3)^2)

theorem floor_T_sq_eq : 
  let T := sum_sqrt_series 2008 in
  (⌊T^2⌋ : ℤ) = 4032062 :=
  sorry

end floor_T_sq_eq_l144_144396


namespace perpendicular_planes_dot_product_zero_l144_144440

variables {α β : Type} [AddCommGroup α] [Module ℝ α] [AddCommGroup β] [Module ℝ β]

variables (m n : α) (l : β)

-- Define perpendicular planes using their normal vectors
def planes_perpendicular (m n : α) : Prop :=
  m ⋅ n = 0

-- Given the planes α and β are perpendicular, prove that their normal vectors have zero dot product
theorem perpendicular_planes_dot_product_zero (h : planes_perpendicular m n) : m ⋅ n = 0 :=
by
  exact h

end perpendicular_planes_dot_product_zero_l144_144440


namespace sum_of_five_primes_is_145_l144_144098

-- Condition: common difference is 12
def common_difference : ℕ := 12

-- Five prime numbers forming an arithmetic sequence with the given common difference
def a1 : ℕ := 5
def a2 : ℕ := a1 + common_difference
def a3 : ℕ := a2 + common_difference
def a4 : ℕ := a3 + common_difference
def a5 : ℕ := a4 + common_difference

-- The sum of the arithmetic sequence
def sum_of_primes : ℕ := a1 + a2 + a3 + a4 + a5

-- Prove that the sum of these five prime numbers is 145
theorem sum_of_five_primes_is_145 : sum_of_primes = 145 :=
by
  -- Proof goes here
  sorry

end sum_of_five_primes_is_145_l144_144098


namespace find_c_l144_144725

theorem find_c (c : ℝ) (h : ∀ x y : ℝ, 5 * x + 8 * y + c = 0 ∧ x + y = 26) : c = -80 :=
sorry

end find_c_l144_144725


namespace symmetric_line_eq_l144_144442

theorem symmetric_line_eq : ∀ (x y : ℝ), (x - 2*y - 1 = 0) ↔ (2*x - y + 1 = 0) :=
by sorry

end symmetric_line_eq_l144_144442


namespace change_in_money_supply_l144_144144

/-- Constants for initial and post-sanction exchange rates. --/
def E0 : ℝ := 90
def E1 : ℝ := 100

/-- Condition: Percentage change in exchange rate given initial and post-sanction exchange rates. --/
def percentage_change (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

/-- Condition: Function that models change in money supply and its effect on exchange rate. --/
def money_supply_effect (money_change : ℝ) : ℝ := money_change * 5

/-- Theorem: The required change in money supply to return the exchange rate to its initial level. --/
theorem change_in_money_supply :
  percentage_change E0 E1 = 11.11 → 2 = 11.11 / 5 := by
  sorry

end change_in_money_supply_l144_144144


namespace num_subsets_of_abc_eq_eight_l144_144086

theorem num_subsets_of_abc_eq_eight : 
  (∃ (s : Finset ℕ), s = {1, 2, 3} ∧ s.powerset.card = 8) :=
sorry

end num_subsets_of_abc_eq_eight_l144_144086


namespace gcd_of_factorials_l144_144854

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definitions for the two factorial terms involved
def term1 : ℕ := factorial 7
def term2 : ℕ := factorial 10 / factorial 4

-- Definition for computing the GCD
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- The statement to prove
theorem gcd_of_factorials : gcd term1 term2 = 2520 :=
by {
  -- Here, the proof would go.
  sorry
}

end gcd_of_factorials_l144_144854


namespace linear_function_k_range_l144_144339

theorem linear_function_k_range (k b : ℝ) (h1 : k ≠ 0) (h2 : ∃ x : ℝ, (x = 2) ∧ (-3 = k * x + b)) (h3 : 0 < b ∧ b < 1) : -2 < k ∧ k < -3 / 2 :=
by
  sorry

end linear_function_k_range_l144_144339


namespace pentagon_area_ratio_is_correct_correct_m_plus_n_l144_144382

noncomputable def pentagon_area_ratio : ℚ :=
  let AB := 3
  let BC := 5
  let DE := 15
  let angle_ABC := 120
  let AC := Real.sqrt (AB^2 + BC^2 - 2 * AB * BC * Real.cos (angle_ABC * Real.pi / 180))
  let area_ratio := (AC / DE) * (AC / DE * AB * (BC - AB * Real.cos (angle_ABC * Real.pi / 180)) / (2 * (AB + BC - AB * Real.cos(120 * Real.pi / 180) + 15 / AC)))
  area_ratio

theorem pentagon_area_ratio_is_correct :
  let AB := 3
  let BC := 5
  let DE := 15
  let angle_ABC := 120
  let AC := Real.sqrt (AB^2 + BC^2 - 2 * AB * BC * Real.cos (angle_ABC * Real.pi / 180))
  let area_ratio := (AC / DE) * (AC / DE * AB * (BC - AB * Real.cos (angle_ABC * Real.pi / 180)) / (2 * (AB + BC - AB * Real.cos(120 * Real.pi / 180) + 15 / AC)))
  area_ratio = 49 / 435 :=
by
  -- Placeholder proof
  sorry

theorem correct_m_plus_n :
  let m := 49
  let n := 435
  let sum_mn := m + n
  sum_mn = 484 :=
by
  -- Placeholder proof
  sorry

end pentagon_area_ratio_is_correct_correct_m_plus_n_l144_144382


namespace boat_speed_in_still_water_l144_144163

theorem boat_speed_in_still_water (V_s : ℝ) (V_b : ℝ) : 
  (V_s = 6) ∧ (1 / (V_b - V_s) = 2 / (V_b + V_s)) → V_b = 18 :=
by
  intros h
  cases h with vs_eq stream_relation
  simp [vs_eq] at stream_relation
  sorry

end boat_speed_in_still_water_l144_144163


namespace can_form_triangle_l144_144125

theorem can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

example : can_form_triangle 8 6 3 := by
  sorry

end can_form_triangle_l144_144125


namespace regular_polygon_sides_l144_144991

theorem regular_polygon_sides (n : ℕ) (h : ∀ (polygon : ℕ), (polygon = 160) → 2 < polygon ∧ (180 * (polygon - 2) / polygon) = 160) : n = 18 := 
sorry

end regular_polygon_sides_l144_144991


namespace intersection_distance_zero_l144_144071

noncomputable def A : Type := ℝ × ℝ

def P : A := (2, 0)

def line_intersects_parabola (x y : ℝ) : Prop :=
  y - 2 * x + 5 = 0 ∧ y^2 = 3 * x + 4

def distance (p1 p2 : A) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem intersection_distance_zero :
  ∀ (A1 A2 : A),
  line_intersects_parabola A1.1 A1.2 ∧ line_intersects_parabola A2.1 A2.2 →
  (abs (distance A1 P - distance A2 P) = 0) :=
sorry

end intersection_distance_zero_l144_144071


namespace area_EFGH_is_correct_l144_144836

-- The given radius
def radius : ℝ := 15

-- The given angle each sector subtends at the center, which is 90 degrees or π/2 radians
def angle_radians : ℝ := Real.pi / 2

-- The area of a full circle
noncomputable def area_circle : ℝ := Real.pi * radius ^ 2

-- The area of one sector
noncomputable def area_sector : ℝ := (angle_radians / (2 * Real.pi)) * area_circle

-- The total area of two sectors
noncomputable def area_EFGH : ℝ := 2 * area_sector

-- The theorem stating the problem
theorem area_EFGH_is_correct : area_EFGH = 112.5 * Real.pi := 
  by sorry

end area_EFGH_is_correct_l144_144836


namespace matrix_C_pow_50_eq_l144_144377

-- Define matrices C and I
def C : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 1], ![-4, -1]]
def I : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, 1]]

-- Given conditions on C and I
def D := C - I
lemma C_squared_eq : C⬝C = 2•C - I := sorry
lemma D_squared_eq_zero : D⬝D = 0 := sorry

-- Final proof goal
theorem matrix_C_pow_50_eq : C^50 = ![![101, 50], ![-200, -99]] :=
by
  sorry

end matrix_C_pow_50_eq_l144_144377


namespace paytons_score_l144_144404

theorem paytons_score (total_score_14_students : ℕ)
    (average_14_students : total_score_14_students / 14 = 80)
    (total_score_15_students : ℕ)
    (average_15_students : total_score_15_students / 15 = 81) :
  total_score_15_students - total_score_14_students = 95 :=
by
  sorry

end paytons_score_l144_144404


namespace correct_propositions_l144_144598

structure Proposition :=
  (statement : String)
  (is_correct : Prop)

def prop1 : Proposition := {
  statement := "All sufficiently small positive numbers form a set.",
  is_correct := False -- From step b
}

def prop2 : Proposition := {
  statement := "The set containing 1, 2, 3, 1, 9 is represented by enumeration as {1, 2, 3, 1, 9}.",
  is_correct := False -- From step b
}

def prop3 : Proposition := {
  statement := "{1, 3, 5, 7} and {7, 5, 3, 1} denote the same set.",
  is_correct := True -- From step b
}

def prop4 : Proposition := {
  statement := "{y = -x} represents the collection of all points on the graph of the function y = -x.",
  is_correct := False -- From step b
}

theorem correct_propositions :
  prop3.is_correct ∧ ¬prop1.is_correct ∧ ¬prop2.is_correct ∧ ¬prop4.is_correct :=
by
  -- Here we put the proof steps, but for the exercise's purpose, we use sorry.
  sorry

end correct_propositions_l144_144598


namespace triangle_angle_contradiction_l144_144101

theorem triangle_angle_contradiction (A B C : ℝ) (h_sum : A + B + C = 180) (h_lt_60 : A < 60 ∧ B < 60 ∧ C < 60) : false := 
sorry

end triangle_angle_contradiction_l144_144101


namespace curve_is_upper_semicircle_l144_144790

theorem curve_is_upper_semicircle (x y : ℝ) : 
  (y - 1 = sqrt (1 - x^2)) ↔ (x^2 + (y - 1)^2 = 1 ∧ y ≥ 1) :=
by
  sorry

end curve_is_upper_semicircle_l144_144790


namespace f_prime_at_1_is_2_max_g_on_interval_f_g_cos_inequality_l144_144645

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 2) / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * Real.log x - 1 / x

theorem f_prime_at_1_is_2 : f' 1 = 2 :=
by
  sorry

theorem max_g_on_interval (a : ℝ) : 
  (a ≤ -1 → ∃ x ∈ interval 1 2, ∀ y ∈ interval 1 2, g x a ≥ g y a ∧ g x a = -1) ∧
  (-1 < a ∧ a < -1/2 → ∃ x ∈ interval 1 2, ∀ y ∈ interval 1 2, g x a ≥ g y a ∧ g x a = a * Real.log (-1 / a) + a) ∧
  (-1/2 ≤ a → ∃ x ∈ interval 1 2, ∀ y ∈ interval 1 2, g x a ≥ g y a ∧ g x a = a * Real.log 2 - 1 / 2) :=
by
  sorry

theorem f_g_cos_inequality (x : ℝ) (h : 0 < x) : f x > g x 1 - Real.cos x / x :=
by
  sorry

end f_prime_at_1_is_2_max_g_on_interval_f_g_cos_inequality_l144_144645


namespace contradiction_in_triangle_l144_144420

theorem contradiction_in_triangle :
  (∀ (A B C : ℝ), A + B + C = 180 ∧ A < 60 ∧ B < 60 ∧ C < 60 → false) :=
by
  sorry

end contradiction_in_triangle_l144_144420


namespace max_S_sum_min_S_sum_l144_144234

/-- 
  Show that the sum of the products 
  S = ∑ 1 ≤ i < j ≤ 5 xi xj 
  is maximized for 
  x1 = 402, 
  x2 = x3 = x4 = x5 = 401, 
  given x1 + x2 + x3 + x4 + x5 = 2006.
-/
theorem max_S_sum (x1 x2 x3 x4 x5 : ℕ) :
  (x1 + x2 + x3 + x4 + x5 = 2006) →
  (S = ∑ (1 ≤ i < j ≤ 5) (xi * xj)) →
  S ≤ ∑ (1 ≤ i < j ≤ 5) (xi * xj) := 
begin
  -- Placeholder proof
  sorry
end

/-- 
  Show that the sum of the products 
  S = ∑ 1 ≤ i < j ≤ 5 xi xj 
  is minimized for 
  x1 = x2 = x3 = 402, 
  x4 = x5 = 400, 
  given x1 + x2 + x3 + x4 + x5 = 2006 
  and | xi - xj | ≤ 2 for 1 ≤ i, j ≤ 5.
-/
theorem min_S_sum (x1 x2 x3 x4 x5 : ℕ) :
  (x1 + x2 + x3 + x4 + x5 = 2006) →
  (∀ i j, 1 ≤ i ∧ i ≤ 5 → 1 ≤ j ∧ j ≤ 5 → |xi - xj| ≤ 2) →
  (S = ∑ (1 ≤ i < j ≤ 5) (xi * xj)) →
  S ≥ ∑ (1 ≤ i < j ≤ 5) (xi * xj) := 
begin
  -- Placeholder proof
  sorry
end

end max_S_sum_min_S_sum_l144_144234


namespace matrix_multiplication_l144_144966

noncomputable def A : Matrix (Fin 3) (Fin 3) ℤ := ![![ 2, -1,  3],
                                                    ![ 0,  3, -2],
                                                    ![ 1,  3,  2]]

noncomputable def B : Matrix (Fin 3) (Fin 3) ℤ := ![![ 1,  3,  0],
                                                    ![-2,  0,  4],
                                                    ![ 5,  0,  1]]

-- The expected product matrix C
noncomputable def C : Matrix (Fin 3) (Fin 3) ℤ := ![![ 19,  6, -1],
                                                    ![-16,  0, 10],
                                                    ![  5,  3, 14]]

theorem matrix_multiplication :
  Matrix.mul A B = C :=
by
  -- Provide proof here
  sorry


end matrix_multiplication_l144_144966


namespace four_digit_numbers_count_l144_144314

theorem four_digit_numbers_count :
  ∃ n : ℕ, n = 4140 ∧
  (∀ d1 d2 d3 d4 : ℕ,
    (4 ≤ d1 ∧ d1 ≤ 9) ∧
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d2 * d3 > 8) →
    (∃ m : ℕ, m = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ m > 3999) →
    n = 4140) :=
sorry

end four_digit_numbers_count_l144_144314


namespace Petya_can_obtain_1001_through_operations_l144_144762

theorem Petya_can_obtain_1001_through_operations :
  ∃ n : ℕ, ∃ f : ℕ → ℚ, (∀ k : ℕ, (f k) ∈ set.Icc (1/3 : ℚ) 2) ∧ (foldl (λ x, λ c, x * f c) n (range k) = 1001) :=
sorry

end Petya_can_obtain_1001_through_operations_l144_144762


namespace number_of_outfits_l144_144771

theorem number_of_outfits (trousers shirts jackets ties : ℕ) 
  (h_trousers : trousers = 5) 
  (h_shirts : shirts = 6) 
  (h_jackets : jackets = 4) 
  (h_ties : ties = 2) :
  trousers * shirts * jackets * ties = 240 :=
by 
  have h1 : trousers * shirts = 30, from 
    by rw [h_trousers, h_shirts]; exact mul_self_div_two 5 6,
  have h2 : trousers * shirts * jackets = 120, from 
    by rw [h1, h_jackets]; exact mul_self_div_two 30 4,
  show trousers * shirts * jackets * ties = 240, from 
    by rw [h2, h_ties]; exact mul_self_div_two 120 2

end number_of_outfits_l144_144771


namespace round_trip_and_car_percent_single_trip_and_motorcycle_percent_l144_144759

noncomputable def totalPassengers := 100
noncomputable def roundTripPercent := 35
noncomputable def singleTripPercent := 100 - roundTripPercent

noncomputable def roundTripCarPercent := 40
noncomputable def roundTripMotorcyclePercent := 15
noncomputable def roundTripNoVehiclePercent := 60

noncomputable def singleTripCarPercent := 25
noncomputable def singleTripMotorcyclePercent := 10
noncomputable def singleTripNoVehiclePercent := 45

theorem round_trip_and_car_percent : 
  ((roundTripCarPercent / 100) * (roundTripPercent / 100) * totalPassengers) = 14 :=
by
  sorry

theorem single_trip_and_motorcycle_percent :
  ((singleTripMotorcyclePercent / 100) * (singleTripPercent / 100) * totalPassengers) = 6 :=
by
  sorry

end round_trip_and_car_percent_single_trip_and_motorcycle_percent_l144_144759


namespace positive_solution_count_l144_144625

theorem positive_solution_count :
  ∃! (x : ℝ), 0 < x ∧ (cos (arcsin (tan (arccos x))) = x) :=
sorry

end positive_solution_count_l144_144625


namespace count_satisfying_values_l144_144742

def d1 (a : ℤ) : ℤ := a^2 + 3^a + a * 3^((a+1)/2)
def d2 (a : ℤ) : ℤ := a^2 + 3^a - a * 3^((a+1)/2)

def is_multiple_of_seven (n : ℤ) : Prop := n % 7 = 0

def satisfies_condition (a : ℤ) : Prop :=
  1 ≤ a ∧ a ≤ 50 ∧ is_multiple_of_seven (d1 a * d2 a)

theorem count_satisfying_values : fintype (finset {a : ℤ | satisfies_condition a}).card = 28 := by
  sorry

end count_satisfying_values_l144_144742


namespace no_closed_broken_line_with_odd_vertices_l144_144023

theorem no_closed_broken_line_with_odd_vertices 
    (vertices : List (ℚ × ℚ)) 
    (h1 : ∀ i < vertices.length - 1, (vertices.get i).1 ≠ (vertices.get (i+1)).1 ∨ (vertices.get i).2 ≠ (vertices.get (i+1)).2)
    (h2 : ∀ (i : Fin (vertices.length - 1)), (vertices.get (i.nat.succ)).1 - (vertices.get i.cast_val.succ)).1)^2 + 
          ((vertices.get (i.nat.succ)).2 - (vertices.get i.cast_val.succ)).2)^2 = 1
    (h3 : vertices.length % 2 = 1) :
  False := sorry

end no_closed_broken_line_with_odd_vertices_l144_144023


namespace triangle_DEF_area_correct_l144_144721

-- Definitions and given conditions
def square_PQRS_area : ℝ := 16

def side_length_of_PQRS (x : ℝ) := x * x = square_PQRS_area

def side_length_of_smaller_squares : ℝ := 2

structure Triangle where
  E : ℝ × ℝ
  D : ℝ × ℝ
  F : ℝ × ℝ

def isosceles_triangle (T : Triangle) := 
  let dE := (T.E.1 - T.D.1) ^ 2 + (T.E.2 - T.D.2) ^ 2
  let dF := (T.F.1 - T.D.1) ^ 2 + (T.F.2 - T.D.2) ^ 2
  dE = dF

def center_of_square (x : ℝ) := (x / 2, x / 2)

-- Problem statement: Prove that the area of triangle DEF is as follows
def area_of_triangle_DEF (T : Triangle) (x : ℝ) := 
  let dE := (T.E.1 - T.D.1) ^ 2 + (T.E.2 - T.D.2) ^ 2
  let d := real.sqrt (2 * x^2)
  let DT := (d / 2) - side_length_of_smaller_squares
  1/2 * dE * DT

theorem triangle_DEF_area_correct (T : Triangle) (x : ℝ)
  (hx : side_length_of_PQRS x)
  (h_iso : isosceles_triangle T)
  (h_align : T.D = center_of_square x) :
  area_of_triangle_DEF T x = 2 * real.sqrt 2 - 2 := 
sorry

end triangle_DEF_area_correct_l144_144721


namespace largest_n_factors_l144_144244

theorem largest_n_factors (n : ℤ) :
  (∃ A B : ℤ, 3 * B + A = n ∧ A * B = 72) → n ≤ 217 :=
by {
  sorry
}

end largest_n_factors_l144_144244


namespace angle_BAC_eq_7_l144_144812

-- Given triangles
variable {A B C D E : Point}

-- Conditions
variable (h1 : Triangle A B C ≅ Triangle E B D)
variable (h2 : ∠ D A E = 37 ∧ ∠ D E A = 37)

theorem angle_BAC_eq_7 :
  ∠ B A C = 7 :=
sorry

end angle_BAC_eq_7_l144_144812


namespace marys_age_l144_144025

variable (M R : ℕ) -- Define M (Mary's current age) and R (Rahul's current age) as natural numbers

theorem marys_age
  (h1 : R = M + 40)       -- Rahul is 40 years older than Mary
  (h2 : R + 30 = 3 * (M + 30))  -- In 30 years, Rahul will be three times as old as Mary
  : M = 20 := 
sorry  -- The proof goes here

end marys_age_l144_144025


namespace problem_solution_l144_144390

variables (x y : ℝ)

def p : Prop := (x - 2) * (y - 5) ≠ 0
def q : Prop := x ≠ 2 ∨ y ≠ 5

/-- \( p \) is a sufficient but not necessary condition for \( q \). -/
theorem problem_solution : (p → q) ∧ ¬(q → p) := 
by
  split
  sorry

end problem_solution_l144_144390


namespace smallest_t_l144_144037

theorem smallest_t (p q r : ℕ) (h₁ : 0 < p) (h₂ : 0 < q) (h₃ : 0 < r) (h₄ : p + q + r = 2510) 
                   (k : ℕ) (t : ℕ) (h₅ : p! * q! * r! = k * 10^t) (h₆ : ¬(10 ∣ k)) : t = 626 := 
by sorry

end smallest_t_l144_144037


namespace sum_of_numbers_Carolyn_removes_l144_144594

noncomputable def game_carolyn_paul_sum : ℕ :=
  let initial_list := [1, 2, 3, 4, 5]
  let removed_by_paul := [3, 4]
  let removed_by_carolyn := [1, 2, 5]
  removed_by_carolyn.sum

theorem sum_of_numbers_Carolyn_removes :
  game_carolyn_paul_sum = 8 :=
by
  sorry

end sum_of_numbers_Carolyn_removes_l144_144594


namespace proof_expression_simplified_l144_144597

noncomputable def expression_simplified : Real :=
0.027 ^ (-1 / 3) - ((-1 / 7) ^ (-2)) + 256 ^ (3 / 4) - 3 ^ (-1) + (Real.sqrt 2 - 1) ^ 0

theorem proof_expression_simplified : expression_simplified = 19 :=
by
  sorry

end proof_expression_simplified_l144_144597


namespace AB_plus_BC_eq_AD_plus_DC_l144_144423

variables {A B C D P Q R S : Point}
variables (AB BC CD DA : Line)
variables (cir : Circle)
variables (inscribed : InscribedInAngle cir ∠BAD)

-- Assume the tangent points
axiom AP_eq_AS : (length (tangent_from A to cir) = length (tangent_from S to cir)) -- (1)
axiom BP_eq_BQ : (length (tangent_from B to cir) = length (tangent_from Q to cir)) -- (1)
axiom CQ_eq_CR : (length (tangent_from C to cir) = length (tangent_from R to cir)) -- (1)
axiom DR_eq_DS : (length (tangent_from D to cir) = length (tangent_from S to cir)) -- (1)

theorem AB_plus_BC_eq_AD_plus_DC :
  (length AB + length BC) = (length AD + length DC) :=
by
  -- Assertions here to satisfy the compiler's requirements
  sorry

end AB_plus_BC_eq_AD_plus_DC_l144_144423


namespace least_degree_polynomial_l144_144875

-- Definitions
structure Point where
  x : ℝ
  y : ℝ
  color : Bool  -- true for red, false for blue

def isAcceptable (pts : List Point) : Prop :=
  (pts.map (λ p => p.x)).nodup ∧ pts.length ≥ 3

def divides (P : ℝ → ℝ) (pts : List Point) : Prop :=
  ∀ pt ∈ pts, 
    (pt.color → P pt.x ≤ pt.y) ∧ (¬pt.color → pt.y ≤ P pt.x)

-- Theorem
theorem least_degree_polynomial (N : ℕ) (pts : List Point) (h : isAcceptable pts) : 
  ∃ (k : ℕ), ∀ (P : (ℝ → ℝ)), (∃ (k' : ℕ), k' ≤ N - 2 ∧ divides P pts) → k = N - 2 :=
by
  sorry

end least_degree_polynomial_l144_144875


namespace photos_approximation_l144_144913

-- Define the given conditions
constant photos_per_roll : ℕ := 36
constant num_rolls : ℕ := 6

-- Define the approximate number of photos
constant approximate_photos : ℕ := 240

-- State the theorem to be proven
-- Prove the total number of photos taken by 6 rolls is approximately 240
theorem photos_approximation :
  (photos_per_roll * num_rolls ≈ approximate_photos) :=
sorry

end photos_approximation_l144_144913


namespace limit_sum_over_n_squared_plus_one_l144_144193

noncomputable def sum_first_n (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem limit_sum_over_n_squared_plus_one :
  (Real.normed_space.normed_monoid_algebra.normed_algebra.normed_add_comm_algebra.norm_space.complete_space_R.some_space.complete_space.See.topological_ring_forall_finset.) (λ n : ℕ, (sum_first_n n) / (n^2 + 1) : ℝ) 0 :
    ∃ a : ℝ, (λ n : ℕ, (sum_first_n n : ℝ) / (n^2 + 1 : ℝ)) ⟶ a :=
begin
  use 1 / 2,
  sorry
end

end limit_sum_over_n_squared_plus_one_l144_144193


namespace plane_equation_l144_144241

-- We will create a structure for 3D points to use in our problem
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the problem conditions and the equation we want to prove
def containsPoint (p: Point3D) : Prop := p.x = 1 ∧ p.y = 4 ∧ p.z = -8

def onLine (p: Point3D) : Prop := 
  ∃ t : ℝ, 
    (p.x = 4 * t + 2) ∧ 
    (p.y = - t - 1) ∧ 
    (p.z = 5 * t + 3)

def planeEq (p: Point3D) : Prop := 
  -4 * p.x + 2 * p.y - 5 * p.z + 3 = 0

-- Now state the theorem
theorem plane_equation (p: Point3D) : 
  containsPoint p ∨ onLine p → planeEq p := 
  sorry

end plane_equation_l144_144241


namespace larger_pie_crust_flour_l144_144370

theorem larger_pie_crust_flour
  (p1 p2 : ℕ)
  (f1 f2 c : ℚ)
  (h1 : p1 = 36)
  (h2 : p2 = 24)
  (h3 : f1 = 1 / 8)
  (h4 : p1 * f1 = c)
  (h5 : p2 * f2 = c)
  : f2 = 3 / 16 :=
sorry

end larger_pie_crust_flour_l144_144370


namespace min_value_expression_l144_144394

theorem min_value_expression (a1 a2 a3 : ℂ) 
  (h1: a1 ≠ 0) (h2: a2 ≠ 0) (h3: a3 ≠ 0) 
  (h4: 0 ≤ a1.re ∧ 0 ≤ a1.im) (h5: 0 ≤ a2.re ∧ 0 ≤ a2.im) (h6: 0 ≤ a3.re ∧ 0 ≤ a3.im) :
  (complex.abs (a1 + a2 + a3) / complex.abs (a1 * a2 * a3) ^ (1/3)) ≥ (real.sqrt 3 * (real.sqrt 2) ^ (1/3)) := 
sorry

end min_value_expression_l144_144394


namespace ellipse_focal_length_l144_144064

theorem ellipse_focal_length (a : ℝ) (h : a > 1) (h_eq : ellipse {x // \frac{x^2}{a} + y^2 = 1}) (c : ℝ) (h_c : c = 2) : 
a = 5 :=
by
  sorry

end ellipse_focal_length_l144_144064


namespace max_area_height_l144_144712

-- Define the radius of the circle
def radius : ℝ := 6

-- Define the height to the base of the triangle
def height_to_base_of_triangle (h : ℝ) : Prop :=
  ∃ (T : Triangle ℝ), T.inscribed_in_circle radius ∧ T.is_isosceles ∧ T.height_to_base = h

-- The theorem statement we want to prove
theorem max_area_height : height_to_base_of_triangle 9 :=
sorry

end max_area_height_l144_144712


namespace combined_length_of_trains_is_correct_l144_144106

noncomputable def combined_length_of_trains : ℕ :=
  let speed_A := 120 * 1000 / 3600 -- speed of train A in m/s
  let speed_B := 100 * 1000 / 3600 -- speed of train B in m/s
  let speed_motorbike := 64 * 1000 / 3600 -- speed of motorbike in m/s
  let relative_speed_A := (120 - 64) * 1000 / 3600 -- relative speed of train A with respect to motorbike in m/s
  let relative_speed_B := (100 - 64) * 1000 / 3600 -- relative speed of train B with respect to motorbike in m/s
  let length_A := relative_speed_A * 75 -- length of train A in meters
  let length_B := relative_speed_B * 90 -- length of train B in meters
  length_A + length_B

theorem combined_length_of_trains_is_correct :
  combined_length_of_trains = 2067 :=
  by
  sorry

end combined_length_of_trains_is_correct_l144_144106


namespace concurrency_of_lines_l144_144379

   variables {A B C U V W P : Type} [EuclideanGeometry A B C U V W P]

   -- Defining the triangle and the properties
   structure Triangle (A B C : Type) :=
   (side1 : segment A B)
   (side2 : segment B C)
   (side3 : segment C A)

   -- Defining the ex-circles and touch points
   structure ExcircleTouches (T : Triangle A B C) :=
   (U_on_BC : TePoint U (line_through (T.side2)))
   (V_on_CA : TePoint V (line_through (T.side3)))
   (W_on_AB : TePoint W (line_through (T.side1)))

   -- Defining the perpendicular lines passing through the touch points
   def perpendicular_line_through_point {A B : Type} (p : TePoint A (line_through B))
   : Line (TePlane B) := sorry

   def ru := perpendicular_line_through_point U
   def rv := perpendicular_line_through_point V
   def rw := perpendicular_line_through_point W

   -- Statement of concurrency problem
   theorem concurrency_of_lines (T : Triangle A B C) (E : ExcircleTouches T) :
     (exists P, TePoint P (intersection_line ru rv) ∧ TePoint P (intersection_line rv rw) ∧ TePoint P (intersection_line rw ru)) :=
   sorry
   
end concurrency_of_lines_l144_144379


namespace evaluate_composite_function_l144_144299

def f (x : ℝ) : ℝ := 1 / (x + 1)
def g (x : ℝ) : ℝ := x^2 + 1

theorem evaluate_composite_function :
  f (g 0) = 1 / 2 :=
by
  sorry

end evaluate_composite_function_l144_144299


namespace segments_comparison_l144_144729

variables {A B C A' B' : Type} [triangle : triangle A B C]
variables {a b c : ℝ} (h_triangle : triangle.abc A B C a b c)
variables {AA' : bisector A A'} {BB' : bisector B B'}
variable (h_angle_bisectors : angle_bisectors AA' BB')

theorem segments_comparison (h_a_gt_b : a > b) :
  segment_length AA' C > segment_length AA' B' ∧ segment_length BB' A > segment_length BB' C :=
sorry

end segments_comparison_l144_144729


namespace shift_right_by_pi_over_4_l144_144483

def f (x : ℝ) : ℝ := Real.cos (2 * x)
def g (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 2)

theorem shift_right_by_pi_over_4 (x : ℝ) : g x = f (x - Real.pi / 4) :=
by sorry

end shift_right_by_pi_over_4_l144_144483


namespace solve_for_y_in_equation_l144_144429

theorem solve_for_y_in_equation : ∃ y : ℝ, 7 * (2 * y - 3) + 5 = -3 * (4 - 5 * y) ∧ y = -4 :=
by
  use -4
  sorry

end solve_for_y_in_equation_l144_144429


namespace slope_of_line_tan_45_l144_144084

theorem slope_of_line_tan_45 (h : Real.tan (Real.pi / 4) = 1) : ∀ x : ℝ, ∀ y : ℝ, y = 1 → 0 = 0 :=
  by
  intro x y hy
  rw hy
  sorry

end slope_of_line_tan_45_l144_144084


namespace power_24_eq_one_l144_144333

theorem power_24_eq_one (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^24 = 1 :=
by
  sorry

end power_24_eq_one_l144_144333


namespace grocery_cost_l144_144828

def rent : ℕ := 1100
def utilities : ℕ := 114
def roommate_payment : ℕ := 757

theorem grocery_cost (total_payment : ℕ) (half_rent_utilities : ℕ) (half_groceries : ℕ) (total_groceries : ℕ) :
  total_payment = 757 →
  half_rent_utilities = (rent + utilities) / 2 →
  half_groceries = total_payment - half_rent_utilities →
  total_groceries = half_groceries * 2 →
  total_groceries = 300 :=
by
  intros
  sorry

end grocery_cost_l144_144828


namespace ratio_of_segments_l144_144021

theorem ratio_of_segments (A B C A1 B1 : Type) (p q : ℝ) 
  (h1 : BA1_ratio_eq : (length B A1)/(length A1 C) = 1/p) 
  (h2 : AB1_ratio_eq : (length A B1)/(length B1 C) = 1/q) :
  (segment_ratio AA1 BB1) = (p+1)/q :=
sorry

end ratio_of_segments_l144_144021


namespace sum_BC_values_l144_144269

-- Definitions
variable (AB BC CD DA: ℝ)
-- Condition Definitions
def isArithmeticProgression : Prop := AB = 20 ∧ AB = CD ∧ AB = CD ∧  (BC = AB - d) ∧ (DA = AB-2*d)
def angleRight : Prop := ∠ A = 90°

theorem sum_BC_values {AB BC CD DA: ℝ} (h: isArithmeticProgression AB BC CD DA) (angleRight) : 
∑ BC_values = 6 :=
sorry

end sum_BC_values_l144_144269


namespace number_of_girls_l144_144350

theorem number_of_girls
  (B G : ℕ)
  (h1 : B = (8 * G) / 5)
  (h2 : B + G = 351) :
  G = 135 :=
sorry

end number_of_girls_l144_144350


namespace sequence_sum_l144_144040

theorem sequence_sum (n : ℕ) (h : 0 < n):
  let x : ℕ → ℕ := λ k, 3 + (k * (k - 1)) / 2
  in (∑ k in finset.range n, x (k + 1)) = 3 * n + (n * (n + 1) * (2 * n - 1)) / 12 :=
by sorry

end sequence_sum_l144_144040


namespace curve_theta_pi_div_4_is_line_l144_144240

theorem curve_theta_pi_div_4_is_line :
  ∀ (r : ℝ), ∃ (x y : ℝ), x = r * cos(π / 4) ∧ y = r * sin(π / 4) → 
  ∃ θ : ℝ, θ = π / 4 ∧ (x / y) = 1 :=
by 
  sorry

end curve_theta_pi_div_4_is_line_l144_144240


namespace books_sold_l144_144371

theorem books_sold (initial_books remaining_books sold_books : ℕ):
  initial_books = 33 → 
  remaining_books = 7 → 
  sold_books = initial_books - remaining_books → 
  sold_books = 26 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end books_sold_l144_144371


namespace intersection_complement_l144_144889

def U : Set ℝ := Set.univ
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {x | x > 3}

theorem intersection_complement :
  A ∩ (U \ B) = {x | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_complement_l144_144889


namespace parabola_intersects_directrix_length_l144_144461

theorem parabola_intersects_directrix_length :
  let e := ellipse 25 16
  let parabola := parabola_vertex_directrix 0 (-25/3)
  let x_right_directrix := 25 / 3
  let points_A_B := parabola_intersect_directrix parabola x_right_directrix
  let A := points_A_B.1
  let B := points_A_B.2
  segment_length A B = 100 / 3 :=
by
  sorry

end parabola_intersects_directrix_length_l144_144461


namespace ceil_minus_val_eq_one_minus_frac_l144_144698

variable (x : ℝ)

theorem ceil_minus_val_eq_one_minus_frac (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ f : ℝ, 0 ≤ f ∧ f < 1 ∧ ⌈x⌉ - x = 1 - f := 
sorry

end ceil_minus_val_eq_one_minus_frac_l144_144698


namespace range_of_m_l144_144672

def f (x : ℝ) : ℝ :=
  if x ≤ 10 then (1 / 10) ^ x else -real.log (x + 2)

theorem range_of_m (m : ℝ) (h : f (8 - m^2) < f (2 * m)) : -4 < m ∧ m < 2 := 
  sorry

end range_of_m_l144_144672


namespace black_balls_probability_both_black_l144_144539

theorem black_balls_probability_both_black (balls_total balls_black balls_gold : ℕ) (prob : ℚ) 
  (h1 : balls_total = 11)
  (h2 : balls_black = 7)
  (h3 : balls_gold = 4)
  (h4 : balls_total = balls_black + balls_gold)
  (h5 : prob = (21 : ℚ) / 55) :
  balls_total.choose 2 * prob = balls_black.choose 2 :=
sorry

end black_balls_probability_both_black_l144_144539


namespace max_min_distance_A_B_l144_144276

-- Define the ellipse condition
def on_ellipse (A : ℝ × ℝ) : Prop :=
  let (x, y) := A in x^2 + 4 * y^2 = 4

-- Define the circle condition
def on_circle (B : ℝ × ℝ) : Prop :=
  let (x, y) := B in x^2 + (y - 2)^2 = 1 / 3

-- Define the distance function between two points
def distance (A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the maximum and minimum distance between points A and B
def max_distance (A B : ℝ × ℝ) : ℝ :=
  (2 * Real.sqrt 21) / 3 + Real.sqrt 3 / 3

def min_distance (A B : ℝ × ℝ) : ℝ :=
  1 - Real.sqrt 3 / 3

-- The proof statement for maximum and minimum distance
theorem max_min_distance_A_B:
  ∀ A B : ℝ × ℝ, 
  on_ellipse A → 
  on_circle B → 
  (distance A B = max_distance A B) ∨ 
  (distance A B = min_distance A B) :=
  sorry

end max_min_distance_A_B_l144_144276


namespace sum_of_roots_cubic_l144_144970

theorem sum_of_roots_cubic :
  let polynomial := λ x: ℝ, 3 * x^3 - 9 * x^2 - 45 * x - 6
  let roots := {r : ℝ // polynomial r = 0}
  let sum_of_roots := (λ s, ∑ r in s, r)
  sum_of_roots roots = 9 :=
sorry

end sum_of_roots_cubic_l144_144970


namespace error_in_simplification_l144_144917

theorem error_in_simplification (x : ℚ) (h₁ : x^2 - 4 ≠ 0) (h₂ : x + 2 ≠ 0) (h₃ : x - 2 ≠ 0):
  let expr_initial := (x + 1) / (x^2 - 4) - 1 / (x + 2) in
  let expr_div := expr_initial / (3 / (x - 2)) in
  let expr_step1 := (x + 1) / ((x + 2) * (x - 2)) - 1 / (x + 2) in
  let expr_step2 := ((x + 1) / ((x + 2) * (x - 2))) - ((x - 2) / ((x + 2) * (x - 2))) in
  let expr_step3 := (x + 1 - (x - 2)) / ((x + 2) * (x - 2)) in
  let expr_step4 := (3) / ((x + 2) * (x - 2)) in
  let expr_simplified := expr_step4 * ((x - 2) / 3) in
  expr_simplified = 1 / (x + 2) ∧
  let incorrect_step := expr_step2 in
  incorrect_step ≠ (x + 1) / ((x + 2) * (x - 2)) - 1 / (x + 2) := 
sorry

end error_in_simplification_l144_144917


namespace multiply_polynomials_l144_144201

theorem multiply_polynomials (x : ℝ) : 2 * x * (5 * x ^ 2) = 10 * x ^ 3 := by
  sorry

end multiply_polynomials_l144_144201


namespace a_circ_b_eq_3_2_l144_144633

variables {α β a b : Vector3} (cos θ : Real) (n m : Int)
-- Conditions
variables (h1 : ∀ (α β : Vector3), α ≠ 0 ∧ β ≠ 0 → α.⊙ β = (α.⋅ β) / (β.⋅ β))
variables (h2 : |a| ≥ |b| ∧ |b| > 0)
variables (h3 : θ ∈ (0, π / 4))
variables (h4 : ∃ n m ∈ ℤ, a.⊙ b = n/2 ∧ b.⊙ a = m/2 ∧ n ≥ m)
variables (h5 : ∃ θ, cos θ = cos (θ))
-- Definition of ⊙ and the goal statement
def Vector3.⊙ (α β : Vector3) : Real := (α.⋅ β) / (β.⋅ β)

theorem a_circ_b_eq_3_2 : 
  a.⊙ b = 3 / 2 := sorry

end a_circ_b_eq_3_2_l144_144633


namespace option_c_opp_numbers_l144_144928

theorem option_c_opp_numbers : (- (2 ^ 2)) = - ((-2) ^ 2) :=
by
  sorry

end option_c_opp_numbers_l144_144928


namespace triathlon_minimum_speeds_l144_144755

theorem triathlon_minimum_speeds (x : ℝ) (T : ℝ := 80) (total_time : ℝ := (800 / x + 20000 / (7.5 * x) + 4000 / (3 * x))) :
  total_time ≤ T → x ≥ 60 ∧ 3 * x = 180 ∧ 7.5 * x = 450 :=
by
  sorry

end triathlon_minimum_speeds_l144_144755


namespace find_base_l144_144049

noncomputable def base_of_power_property (x k : ℝ) : Prop :=
  ∀ y : ℝ, y ≠ 0 → (kx)^(y/k) = x^y

theorem find_base (k : ℝ) (hk : k > 0) : ∃ x : ℝ, base_of_power_property x k ∧ x = k ^ (1 / (k - 1)) :=
sorry

end find_base_l144_144049


namespace exist_disjoint_translations_l144_144392

open Set

variable (S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 1000000})
variable (A : Set ℕ)

noncomputable def has_101_elements (A : Set ℕ) : Prop :=
  ∃ l, A = {x | x ∈ S ∧ ∃ i, i < 101 ∧ l i = x}

theorem exist_disjoint_translations (hA : has_101_elements A) :
  ∃ (t : Fin 100 → ℕ), (∀ i j, i ≠ j → Disjoint (A + {t i}) (A + {t j})) :=
sorry

end exist_disjoint_translations_l144_144392


namespace irreducible_poly_no_double_root_l144_144428

theorem irreducible_poly_no_double_root {P : Polynomial ℚ} (hP_irreducible : Irreducible P) :
  ¬ ∃ α : ℂ, (P.eval α = 0 ∧ (P.derivative.eval α = 0)) := 
sorry

end irreducible_poly_no_double_root_l144_144428


namespace g_zeros_g_inequality_l144_144673

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * (x - π / 6))

theorem g_zeros : ∃ (count : ℕ), count = 6 ∧ ∀ (x : ℝ), 
  0 < x ∧ x < 3 * π → g x = 0 → x = π / 6 + k * π / 2 for some k ∈ ℤ := by
  sorry

theorem g_inequality (a : ℝ) : 
  (∀ x : ℝ, -π / 6 ≤ x ∧ x ≤ π / 2 → g x - a ≥ f (5 * π / 12)) → a ≤ -3 / 2 := by
  sorry

end g_zeros_g_inequality_l144_144673


namespace solution_set_of_inequality_l144_144459

theorem solution_set_of_inequality (x : ℝ) : |x^2 - 2| < 2 ↔ ((-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2)) :=
by sorry

end solution_set_of_inequality_l144_144459


namespace variance_of_constant_variance_of_scaled_variable_variance_of_sum_independent_variance_of_difference_l144_144419

noncomputable section

variable (ProbabilitySpace : Type)
variable [ProbabilityTheory ProbabilitySpace]

namespace VarianceProof

variable {X Y : ProbabilitySpace → ℝ}
variable {C : ℝ}

-- Definitions needed for variance and expectation
def variance (X : ProbabilitySpace → ℝ) : ℝ := sorry -- Placeholder for the variance definition
def expectation (X : ProbabilitySpace → ℝ) : ℝ := sorry -- Placeholder for the expectation definition
def independent (X Y : ProbabilitySpace → ℝ) : Prop := sorry -- Placeholder for the independence definition

-- Statements to prove
theorem variance_of_constant (C : ℝ) : variance (λ _, C) = 0 := 
sorry

theorem variance_of_scaled_variable (C : ℝ) (X : ProbabilitySpace → ℝ) : 
    variance (λ ω, C * X ω) = C^2 * variance X := 
sorry

theorem variance_of_sum_independent (X Y : ProbabilitySpace → ℝ) (h : independent X Y) : 
    variance (λ ω, X ω + Y ω) = variance X + variance Y := 
sorry

theorem variance_of_difference (X Y : ProbabilitySpace → ℝ) : 
    variance (λ ω, X ω - Y ω) = variance X + variance Y := 
sorry

end VarianceProof

end variance_of_constant_variance_of_scaled_variable_variance_of_sum_independent_variance_of_difference_l144_144419


namespace find_cos_A_and_a_l144_144708

theorem find_cos_A_and_a (A B C : ℝ) (a b c : ℝ) 
  (hb : b = 3) 
  (hc : c = 1) 
  (area : ℝ) 
  (h_area : area = sqrt 2) 
  (h_sinA : (1 / 2) * b * c * Real.sin A = sqrt 2) 
  : (Real.cos A = 1 / 3 ∨ Real.cos A = -1 / 3) ∧ 
    (a = 2 * sqrt 2 ∨ a = 2 * sqrt 3) :=
by
  sorry

end find_cos_A_and_a_l144_144708


namespace fraction_division_l144_144699

theorem fraction_division (a b : ℚ) (ha : a = 3) (hb : b = 4) :
  (1 / b) / (1 / a) = 3 / 4 :=
by 
  -- Solve the proof
  sorry

end fraction_division_l144_144699


namespace log_base4_one_over_sixty_four_l144_144228

theorem log_base4_one_over_sixty_four : log 4 (1 / 64) = -3 := by
  sorry

end log_base4_one_over_sixty_four_l144_144228


namespace sum_of_angles_is_180_l144_144749

variables {A B C D E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
variables {α β γ δ ε : ℝ}
variables (D A C E B : Type) 

-- Conditions
def is_convex (ABCDE : A → B → C → D → E → Prop) : Prop :=
sorry -- A function that defines the convexity condition

def angle_DAC (α : ℝ) : Prop :=
sorry -- Angle α between line segments DA and AC

def angle_EBD (β : ℝ) : Prop :=
sorry -- Angle β between line segments EB and BD

def angle_ACE (γ : ℝ) : Prop :=
sorry -- Angle γ between line segments AC and CE

def angle_BDA (δ : ℝ) : Prop :=
sorry -- Angle δ between line segments BD and DA

def angle_BEC (ε : ℝ) : Prop :=
sorry -- Angle ε between line segments BE and EC

-- Proof statement
theorem sum_of_angles_is_180 :
  ∀ (ABCDE : A → B → C → D → E → Prop),
    is_convex ABCDE →
    angle_DAC α →
    angle_EBD β →
    angle_ACE γ →
    angle_BDA δ →
    angle_BEC ε →
    α + β + γ + δ + ε = 180 :=
sorry

end sum_of_angles_is_180_l144_144749


namespace volleyball_team_starters_l144_144410

open Function

noncomputable def binom : ℕ → ℕ → ℕ :=
  λ n k, Nat.choose n k

theorem volleyball_team_starters :
  ∃ (n k : ℕ), n = 18 ∧ k = 7 ∧
  ∃ quad : Finset ℕ → ℕ,
  quad (Finset.filter (λ i, i < 4) (Finset.range n)).card = 4 ∧
  let quad_combinations := binom 4 2,
      remaining_players := n - 4,
      remaining_combinations := binom remaining_players 5 in
  quad_combinations * remaining_combinations = 12012 := 
by
  let n := 18
  let k := 7
  let quadruplets := Finset.filter (λ i, i < 4) (Finset.range n)
  let qc := binom 4 2
  let rp := n - 4
  let rc := binom rp 5
  have hq_quadruplets : quadruplets.card = 4 := by sorry
  use n, k
  exact ⟨rfl, ⟨rfl, ⟨quadruplets, ⟨hq_quadruplets, qc, rp, rc, (nat.cast_mul qc rc).symm ▸ rfl⟩⟩⟩⟩

end volleyball_team_starters_l144_144410


namespace slices_ratio_l144_144975

theorem slices_ratio (total_slices : ℕ) (hawaiian_slices : ℕ) (cheese_slices : ℕ) 
  (dean_hawaiian_eaten : ℕ) (frank_hawaiian_eaten : ℕ) (sammy_cheese_eaten : ℕ)
  (total_leftover : ℕ) (hawaiian_leftover : ℕ) (cheese_leftover : ℕ)
  (H1 : total_slices = 12)
  (H2 : hawaiian_slices = 12)
  (H3 : cheese_slices = 12)
  (H4 : dean_hawaiian_eaten = 6)
  (H5 : frank_hawaiian_eaten = 3)
  (H6 : total_leftover = 11)
  (H7 : hawaiian_leftover = hawaiian_slices - dean_hawaiian_eaten - frank_hawaiian_eaten)
  (H8 : cheese_leftover = total_leftover - hawaiian_leftover)
  (H9 : sammy_cheese_eaten = cheese_slices - cheese_leftover)
  : sammy_cheese_eaten / cheese_slices = 1 / 3 :=
by sorry

end slices_ratio_l144_144975


namespace log_base4_of_1_div_64_l144_144218

theorem log_base4_of_1_div_64 : log 4 (1 / 64) = -3 := by
  sorry

end log_base4_of_1_div_64_l144_144218


namespace find_functions_l144_144041

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.sin x

theorem find_functions : 
  (∀ x : ℝ, ∃ C : ℝ, f(x) = 1/2 - ∫ t in 0..x, (deriv f t + g t)) ∧ 
  (∀ x : ℝ, ∃ C : ℝ, g(x) = Real.sin x - ∫ t in 0..π, (f t - deriv g t)) ∧ 
  (Continuous (deriv f)) ∧ 
  (Continuous (deriv g)) :=
by
  sorry

end find_functions_l144_144041


namespace hyperbola_center_l144_144239

noncomputable def hyperbola : ℝ → ℝ → Prop :=
  λ x y, (4 * y - 8)^2 / 7^2 - (2 * x + 6)^2 / 9^2 = 1

theorem hyperbola_center : ∃ x y, hyperbola x y ∧ x = -3 ∧ y = 2 := by
  sorry

end hyperbola_center_l144_144239


namespace number_of_four_digit_numbers_l144_144320

/-- The number of four-digit numbers greater than 3999 where the product of the middle two digits exceeds 8 is 2400. -/
def four_digit_numbers_with_product_exceeding_8 : Nat :=
  let first_digit_options := 6  -- Choices are 4,5,6,7,8,9
  let last_digit_options := 10  -- Choices are 0,1,...,9
  let middle_digit_options := 40  -- Valid pairs counted from the solution
  first_digit_options * middle_digit_options * last_digit_options

theorem number_of_four_digit_numbers : four_digit_numbers_with_product_exceeding_8 = 2400 :=
by
  sorry

end number_of_four_digit_numbers_l144_144320


namespace tangent_normal_lines_l144_144637

theorem tangent_normal_lines :
  ∃ m_t b_t m_n b_n,
    (∀ x y, y = 1 / (1 + x^2) → y = m_t * x + b_t → 4 * x + 25 * y - 13 = 0) ∧
    (∀ x y, y = 1 / (1 + x^2) → y = m_n * x + b_n → 125 * x - 20 * y - 246 = 0) :=
by
  sorry

end tangent_normal_lines_l144_144637


namespace smallest_abundant_not_multiple_of_5_is_12_l144_144183

-- Definition: Proper divisors of a number
def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d, d > 0 ∧ n % d = 0)

-- Definition: A number is abundant if the sum of its proper divisors is greater than the number itself
def is_abundant (n : ℕ) : Prop :=
  (List.sum (proper_divisors n)) > n

-- The main theorem to state that the smallest abundant number that is not a multiple of 5 is 12
theorem smallest_abundant_not_multiple_of_5_is_12 : 
  ∃ n : ℕ, is_abundant n ∧ ¬ (5 ∣ n) ∧ (∀ m : ℕ, is_abundant m ∧ ¬ (5 ∣ m) → n ≤ m) ∧ n = 12 := 
by 
  sorry

end smallest_abundant_not_multiple_of_5_is_12_l144_144183


namespace unique_solution_of_diophantine_l144_144998

theorem unique_solution_of_diophantine (m n : ℕ) (hm_pos : m > 0) (hn_pos: n > 0) :
  m^2 = Int.sqrt n + Int.sqrt (2 * n + 1) → (m = 13 ∧ n = 4900) :=
by
  sorry

end unique_solution_of_diophantine_l144_144998


namespace area_S_div_area_T_l144_144385

def is_in_plane (x y z : ℝ) := x + y + z = 2

def supports (x y z : ℝ) := 
  (x >= 1 ∧ y >= (2/3) ∧ z < (1/3)) ∨
  (x >= 1 ∧ y < (2/3) ∧ z >= (1/3)) ∨
  (x < 1 ∧ y >= (2/3) ∧ z >= (1/3))

def in_set_S (x y z : ℝ) := 
  is_in_plane x y z ∧ supports x y z

-- Main statement to prove
theorem area_S_div_area_T : 
  (S : set (ℝ × ℝ × ℝ)) :=
  { p | in_set_S p.1 p.2 p.3 } 
  (T : set (ℝ × ℝ × ℝ)) :=
  { p | is_in_plane p.1 p.2 p.3 } 
  -- The required ratio of areas
  sorry

end area_S_div_area_T_l144_144385


namespace evaluate_expression_l144_144955

theorem evaluate_expression : 4 * 6 * 8 + 24 / 4 - 2 = 196 := by
  sorry

end evaluate_expression_l144_144955


namespace Aren_listening_time_l144_144581

/--
Aren’s flight from New York to Hawaii will take 11 hours 20 minutes. He spends 2 hours reading, 
4 hours watching two movies, 30 minutes eating his dinner, some time listening to the radio, 
and 1 hour 10 minutes playing games. He has 3 hours left to take a nap. 
Prove that he spends 40 minutes listening to the radio.
-/
theorem Aren_listening_time 
  (total_flight_time : ℝ := 11 * 60 + 20)
  (reading_time : ℝ := 2 * 60)
  (watching_movies_time : ℝ := 4 * 60)
  (eating_dinner_time : ℝ := 30)
  (playing_games_time : ℝ := 1 * 60 + 10)
  (nap_time : ℝ := 3 * 60) :
  total_flight_time - (reading_time + watching_movies_time + eating_dinner_time + playing_games_time + nap_time) = 40 :=
by sorry

end Aren_listening_time_l144_144581


namespace minimum_1x1_tiles_needed_l144_144146

theorem minimum_1x1_tiles_needed :
  ∀ (n : ℕ), exists m : ℕ, m > 0 ∧ (∀ (f : fin 23 × fin 23 → ℕ), 
  (∀ x y, f (x, y) = 1 ∨ f (x, y) = 2 ∨ f (x, y) = 3) →
  (∀ st : fin 23 × fin 23, if f st = 1 then tile1 st else if f st = 2 then tile2 st else tile3 (x, y) st) →
  (∃ (p : fin 23 × fin 23 → Prop), (p = 1) →
  ∀ a b. a * b = 23 * 23 ∧ (a, fin 23) ∈ p → b = m)) sorry

end minimum_1x1_tiles_needed_l144_144146


namespace kate_has_7_dollars_80_cents_left_l144_144375

-- Define the given conditions.
def savings_march := 27
def savings_april := 13
def savings_may := 28
def savings_june := 35
def rate_euro_to_dollar := 1.20
def rate_pound_to_dollar := 1.40

-- Define the expenses in euros and pounds
def keyboard_expense_euro := 42
def mouse_expense_euro := 4
def headset_expense_euro := 16
def videogame_expense_euro := 25
def book_expense_pound := 12

-- Define the total savings calculation
def total_savings_january_to_june := savings_march + savings_april + savings_may + savings_june
def savings_july := 2 * savings_april
def total_savings := total_savings_january_to_june + savings_july

-- Define the conversion from euros/pounds to dollars for each expense
def total_expense_euros := (keyboard_expense_euro * rate_euro_to_dollar) + 
                            (mouse_expense_euro * rate_euro_to_dollar) + 
                            (headset_expense_euro * rate_euro_to_dollar) + 
                            (videogame_expense_euro * rate_euro_to_dollar)

def book_expense_dollars := book_expense_pound * rate_pound_to_dollar
def total_expenses := total_expense_euros + book_expense_dollars

-- Define the resulting money left calculation
def money_left := total_savings - total_expenses

-- The theorem statement to prove that Kate has $7.80 left
theorem kate_has_7_dollars_80_cents_left : money_left = 7.80 := 
by 
  sorry

end kate_has_7_dollars_80_cents_left_l144_144375


namespace infinite_geometric_series_common_ratio_l144_144185

theorem infinite_geometric_series_common_ratio :
  ∀ (a r S : ℝ), a = 500 ∧ S = 4000 ∧ (a / (1 - r) = S) → r = 7 / 8 :=
by
  intros a r S h
  cases h with h_a h_S_eq
  cases h_S_eq with h_S h_sum_eq
  -- Now we have: a = 500, S = 4000, and a / (1 - r) = S
  sorry

end infinite_geometric_series_common_ratio_l144_144185


namespace employed_males_percentage_l144_144360

variables {p : ℕ} -- total population
variables {employed_p : ℕ} {employed_females_p : ℕ}

-- 60 percent of the population is employed
def employed_population (p : ℕ) : ℕ := 60 * p / 100

-- 20 percent of the employed people are females
def employed_females (employed : ℕ) : ℕ := 20 * employed / 100

-- The question we're solving:
theorem employed_males_percentage (h1 : employed_p = employed_population p)
  (h2 : employed_females_p = employed_females employed_p)
  : (employed_p - employed_females_p) * 100 / p = 48 :=
by
  sorry

end employed_males_percentage_l144_144360


namespace campers_last_week_correct_l144_144550

variable (C : ℝ)
variable (campers_last_week : ℝ)

theorem campers_last_week_correct 
  (h1 : C + (C + 10) + campers_last_week = 150)
  (h2 : C + 10 = 40) : 
  campers_last_week = 80 :=
by
  have hC : C = 30 := by linarith
  rw [hC] at h1
  rw [hC] at h2
  linarith

end campers_last_week_correct_l144_144550


namespace no_two_perfect_cubes_l144_144747

theorem no_two_perfect_cubes (n : ℕ) : ¬ (∃ a b : ℕ, a^3 = n + 2 ∧ b^3 = n^2 + n + 1) := by
  sorry

end no_two_perfect_cubes_l144_144747


namespace pipe_B_filling_time_l144_144102

theorem pipe_B_filling_time (T_B : ℝ) 
  (A_filling_time : ℝ := 10) 
  (combined_filling_time: ℝ := 20/3)
  (A_rate : ℝ := 1 / A_filling_time)
  (combined_rate : ℝ := 1 / combined_filling_time) : 
  1 / T_B = combined_rate - A_rate → T_B = 20 := by 
  sorry

end pipe_B_filling_time_l144_144102


namespace min_avg_cost_at_200_max_annual_profit_at_230_l144_144356

-- Define problem conditions
def production_volume (x : ℝ) : Prop := 150 ≤ x ∧ x ≤ 250
def total_annual_cost (x : ℝ) : ℝ := (x^2)/10 - 30 * x + 4000

-- Define the problem questions
theorem min_avg_cost_at_200 : 
  ∀ (x : ℝ), production_volume x → 
    let W := total_annual_cost x / x in
    W ≥ 10 ∧ (∀ y, production_volume y → total_annual_cost y / y ≥ W)

theorem max_annual_profit_at_230 :
  ∀ (x : ℝ), production_volume x → 
    let profit := 16 * x - total_annual_cost x in
    profit ≤ 1290 ∧ (∀ y, production_volume y → 16 * y - total_annual_cost y ≤ profit)

end min_avg_cost_at_200_max_annual_profit_at_230_l144_144356


namespace f_monotonically_increasing_l144_144806

def f (x : ℝ) : ℝ := Real.exp (-x^2 + 4 * x - 9)

theorem f_monotonically_increasing : ∀ x1 x2 : ℝ, x1 < x2 → x2 < 2 → f x1 ≤ f x2 := by
  sorry

end f_monotonically_increasing_l144_144806


namespace bounded_variation_iff_diff_non_decreasing_l144_144767

noncomputable theory

open Set

-- Definitions of bounded variation and non-decreasing functions
def bounded_variation (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ M : ℝ, 0 ≤ M ∧ ∀ (P : List (ℝ × ℝ)), (∀ {x y : ℝ}, (x, y) ∈ P → x < y) → 
  (∑ (x_i, x_i1) in P.zip (P.tail ++ [(a, b)]), |f x_i1 - f x_i|) ≤ M

def non_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

-- The theorem we need to prove
theorem bounded_variation_iff_diff_non_decreasing (f : ℝ → ℝ) (a b : ℝ) :
  bounded_variation f a b ↔ 
  ∃ (g h : ℝ → ℝ), non_decreasing g a b ∧ non_decreasing h a b ∧ ∀ x : ℝ, a ≤ x → x ≤ b → f x = g x - h x :=
by sorry

end bounded_variation_iff_diff_non_decreasing_l144_144767


namespace valid_four_digit_numbers_count_l144_144309

noncomputable def num_valid_four_digit_numbers : ℕ := 6 * 65 * 10

theorem valid_four_digit_numbers_count :
  num_valid_four_digit_numbers = 3900 :=
by
  -- We provide the steps in the proof to guide the automation
  let total_valid_numbers := 6 * 65 * 10
  have h1 : total_valid_numbers = 3900 := rfl
  exact h1

end valid_four_digit_numbers_count_l144_144309


namespace relation_between_A_and_B_l144_144740

variables {R : Type*} [linear_ordered_field R]

-- Definitions for subsets A and B
variable (A B : set R)

-- Define m and n based on the conditions
def m (x : R) : ℕ := if x ∈ A then 1 else 0
def n (x : R) : ℕ := if x ∈ B then 1 else 0

-- Statement of the theorem
theorem relation_between_A_and_B (h : ∀ x : R, m A x + n B x = 1) : B = Aᶜ :=
by {
  sorry
}

end relation_between_A_and_B_l144_144740


namespace collinear_feet_of_perpendiculars_l144_144378

theorem collinear_feet_of_perpendiculars
  (A B C D E F P Q R S : Type)
  [h1 : altitude A D (line B C)]
  [h2 : altitude B E (line A C)]
  [h3 : altitude C F (line A B)]
  [footP : foot_from_point P D (line B A)]
  [footQ : foot_from_point Q D (line B E)]
  [footR : foot_from_point R D (line C F)]
  [footS : foot_from_point S D (line C A)] :
  collinear P Q R S :=
sorry

end collinear_feet_of_perpendiculars_l144_144378


namespace sum_series_eq_l144_144205

-- Define the general term of the series
def a (n : ℕ) : ℝ := (4 * n + 3) / ((4 * n + 1) ^ 2 * (4 * n + 5) ^ 2)

-- State the theorem we want to prove
theorem sum_series_eq : 
  (∑' n : ℕ, a n) = 1 / 200 :=
sorry

end sum_series_eq_l144_144205


namespace can_form_triangle_l144_144124

theorem can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

example : can_form_triangle 8 6 3 := by
  sorry

end can_form_triangle_l144_144124


namespace percentage_return_is_25_l144_144153

noncomputable def percentage_return_on_investment
  (dividend_rate : ℝ)
  (face_value : ℝ)
  (purchase_price : ℝ) : ℝ :=
  (dividend_rate / 100 * face_value / purchase_price) * 100

theorem percentage_return_is_25 :
  percentage_return_on_investment 18.5 50 37 = 25 := 
by
  sorry

end percentage_return_is_25_l144_144153


namespace root_set_conclusion_check_l144_144751

variables {a b c : ℝ}

def f (x : ℝ) : ℝ := a^x + b^x - c^x

def M : set (ℝ × ℝ × ℝ) := 
  { t | let (a, b, c) := t in
        a = b ∧ ¬(a + b > c ∧ a + c > b ∧ b + c > a) }

theorem root_set (a b c : ℝ) (h1 : c > a) (h2 : c > b) 
  (h3 : (a, b, c) ∈ M) : 
  ∃ x ∈ (0, 1), f(x) = 0 := 
sorry

theorem conclusion_check (h1 : c > a) (h2 : c > b) 
  (a b c : ℝ) :
  ¬ (∀ x ∈ set.Iio 1, f(x) > 0) ∧ 
  (∃ x : ℝ, ¬ (a^x + b^x > c^x ∧ a^x + c^x > b^x ∧ b^x + c^x > a^x)) ∧ 
  (∃ x ∈ set.Icc 1 2, f(x) = 0 → (c^2 > a^2 + b^2)) :=
sorry

end root_set_conclusion_check_l144_144751


namespace seq_example_proof_l144_144727

-- Define the sequence a_n with the given initial conditions and recurrence relation
def seq (a : ℕ → ℝ) : Prop :=
  (a 0 = 1) ∧
  (a 1 = 2) ∧
  (∀ n ≥ 1, n * (n + 1) * a (n + 1) = n * (n - 1) * a n - (n - 2) * a (n - 1))

-- Example property we're interested in proving
def example_property (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ n ≥ 2, a n = 1 / (n.factorial)

-- Pure definition of the conjecture
theorem seq_example_proof :
  ∃ a : ℕ → ℝ, seq a ∧ example_property a :=
sorry

end seq_example_proof_l144_144727


namespace savings_is_zero_l144_144916

/-- Define the cost per window, quantity of windows needed by Dave and Doug -/
def window_price : ℕ := 100
def dave_windows_needed : ℕ := 11
def doug_windows_needed : ℕ := 9
def free_per_three_purchased : ℕ := 1
def total_windows_needed : ℕ := dave_windows_needed + doug_windows_needed

/-- Calculation for windows paid when a certain number is needed with the given discount offer -/
def windows_to_pay_for (needed : ℕ) : ℕ :=
  needed - (needed / 3 * free_per_three_purchased)

/-- Cost calculation for windows needed -/
def cost (needed : ℕ) : ℕ :=
  windows_to_pay_for needed * window_price

/-- Calculate savings when purchasing together vs separately -/
def savings : ℕ :=
  (cost dave_windows_needed + cost doug_windows_needed) - cost total_windows_needed

/-- The theorem stating that the savings by purchasing together is zero -/
theorem savings_is_zero : savings = 0 := by
  sorry

end savings_is_zero_l144_144916


namespace ellipse_x_intercept_l144_144931

theorem ellipse_x_intercept (x : ℝ) :
  let f1 := (0, 3)
  let f2 := (4, 0)
  let origin := (0, 0)
  let d := sqrt ((fst f1)^2 + (snd f1)^2) + sqrt ((fst f2)^2 + (snd f2)^2)
  d = 7 → -- sum of distances from origin to the foci is 7
  (d_1 : ℝ := abs x - 4 + sqrt (x^2 + 9))
  d_1 = 7 → -- sum of distances from (x, 0) to the foci is 7
  x ≠ 0 → -- x is not 0 because the other x-intercept is not (0, 0)
  x = 56 / 11 → -- x > 4
  (x, 0) = ((56 : ℝ) / 11, 0) :=
by
  sorry

end ellipse_x_intercept_l144_144931


namespace work_completion_time_l144_144132

theorem work_completion_time :
  let a_rate := (1 : ℚ) / 11
      b_rate := (1 : ℚ) / 45
      c_rate := (1 : ℚ) / 55
      ab_rate := a_rate + b_rate
      ac_rate := a_rate + c_rate
      two_day_work := ab_rate + ac_rate
      work_done_per_two_days := two_day_work
      total_two_day_cycles := (1 : ℚ) / work_done_per_two_days
  in total_two_day_cycles ≤ 5 ∧ 2 * 5 = 10 :=
by
  let a_rate := (1 : ℚ) / 11
  let b_rate := (1 : ℚ) / 45
  let c_rate := (1 : ℚ) / 55
  let ab_rate := a_rate + b_rate
  let ac_rate := a_rate + c_rate
  let two_day_work := ab_rate + ac_rate
  let work_done_per_two_days := two_day_work
  let total_two_day_cycles := (1 : ℚ) / work_done_per_two_days
  show total_two_day_cycles ≤ 5 ∧ 2 * 5 = 10
  sorry

end work_completion_time_l144_144132


namespace curve_length_l144_144246

noncomputable def parametric_curve_length : Real :=
  let x t := 3 * Real.sin t
  let y t := 3 * Real.cos t
  Real.sqrt ((Real.deriv x)^2 + (Real.deriv y)^2)

theorem curve_length : parametric_curve_length (t = 0) (t = 2π) = 6 * Real.pi := sorry

end curve_length_l144_144246


namespace solve_system_l144_144688

theorem solve_system :
  ∀ (a1 a2 c1 c2 x y : ℝ),
  (a1 * 5 + 10 = c1) →
  (a2 * 5 + 10 = c2) →
  (a1 * x + 2 * y = a1 - c1) →
  (a2 * x + 2 * y = a2 - c2) →
  (x = -4) ∧ (y = -5) := by
  intros a1 a2 c1 c2 x y h1 h2 h3 h4
  sorry

end solve_system_l144_144688


namespace no_sequence_a_no_sequence_b_l144_144527

-- Definitions for part (a)
def sequence_condition_a (a : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), a n ≤ n ^ 10

def sum_condition (a : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), ∀ (S : finset ℕ), (S.card > 0 ∧ S.card < n) ∧ (∀ (i : ℕ), i ∈ S → i < n) → a n ≠ S.sum a

-- Definitions for part (b)
def sequence_condition_b (a : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), a n ≤ n * nat.sqrt n

-- Theorems to prove non-existence of such sequences
theorem no_sequence_a : ¬ (∃ (a : ℕ → ℕ), sequence_condition_a a ∧ sum_condition a) := 
sorry

theorem no_sequence_b : ¬ (∃ (a : ℕ → ℕ), sequence_condition_b a ∧ sum_condition a) := 
sorry

end no_sequence_a_no_sequence_b_l144_144527


namespace meaningless_expression_l144_144518

-- Let's define the meaning of meaningfulness for real numbers.
def is_meaningful (x : ℝ) : Prop :=
  x >= 0

def is_meaningful_cubert (x : ℝ) : Prop :=
  true

-- Define the expressions 
def expression_a := real.sqrt 2
def expression_b := real.sqrt (-2)
def expression_c := real.cbrt (-2)
def expression_d := -real.sqrt 2

-- Main statement to prove
theorem meaningless_expression : 
  (¬ is_meaningful (expression_b)) ∧ 
  (is_meaningful (expression_a)) ∧ 
  (is_meaningful_cubert (expression_c)) ∧ 
  (is_meaningful (expression_d)) :=
by 
  sorry

end meaningless_expression_l144_144518


namespace find_smallest_angle_l144_144087

theorem find_smallest_angle 
  (x y : ℝ)
  (hx : x + y = 45)
  (hy : y = x - 5)
  (hz : x > 0 ∧ y > 0 ∧ x + y < 180) :
  min x y = 20 := 
sorry

end find_smallest_angle_l144_144087


namespace eval_log_expression_correct_l144_144615

noncomputable def eval_log_expression : ℚ :=
  let log_6 (x : ℕ) := real.log x / real.log 6
  let log_7 (x : ℕ) := real.log x / real.log 7
  let expr := (2 : ℚ) / (7 * log_6 3000) + (3 : ℚ) / (7 * log_7 3000)
  let approx_log_3000 (x : ℕ) := 1.5
  let result := (1 : ℚ) / 7 * approx_log_3000 12348
  result

theorem eval_log_expression_correct : eval_log_expression = 3 / 14 := by
  sorry

end eval_log_expression_correct_l144_144615


namespace percentage_of_men_and_women_l144_144180

theorem percentage_of_men_and_women
  (W M : ℕ)
  (h1 : W + M = 120)
  (h2 : 0.65 * W = 39) :
  (W / 120 : ℝ) * 100 = 50 ∧ (M / 120 : ℝ) * 100 = 50 :=
by
  sorry

end percentage_of_men_and_women_l144_144180


namespace machine_returns_to_station_1_machine_max_stops_before_halt_l144_144169

-- We define a function for the machine's movement behavior based on the given rules.
def move (n : ℕ) : ℕ :=
  if n % 5 = 0 then n - 1 else n + 4

-- Main theorem to check the machine eventually stops at station 1
theorem machine_returns_to_station_1 : ∀ (start : ℕ), 1 ≤ start ∧ start ≤ 2009 → (∃ (k : ℕ), iterate move k start = 1) :=
by
  sorry

-- Main theorem to determine the maximum number of stops
theorem machine_max_stops_before_halt : ∃ (m : ℕ), m = 812 :=
by
  sorry

end machine_returns_to_station_1_machine_max_stops_before_halt_l144_144169


namespace inequality_solution_l144_144818

theorem inequality_solution (x : ℝ) : 
  (x + 1) * (2 - x) < 0 ↔ x < -1 ∨ x > 2 := 
sorry

end inequality_solution_l144_144818


namespace line_symmetric_to_itself_l144_144623

theorem line_symmetric_to_itself :
  ∀ x y : ℝ, y = 3 * x + 3 ↔ ∃ (m b : ℝ), y = m * x + b ∧ m = 3 ∧ b = 3 :=
by
  sorry

end line_symmetric_to_itself_l144_144623


namespace pqrs_product_l144_144277

theorem pqrs_product :
  let P := (Real.sqrt 2010 + Real.sqrt 2009 + Real.sqrt 2008)
  let Q := (-Real.sqrt 2010 - Real.sqrt 2009 + Real.sqrt 2008)
  let R := (Real.sqrt 2010 - Real.sqrt 2009 - Real.sqrt 2008)
  let S := (-Real.sqrt 2010 + Real.sqrt 2009 - Real.sqrt 2008)
  P * Q * R * S = 1 := by
{
  sorry -- Proof is omitted as per the provided instructions.
}

end pqrs_product_l144_144277


namespace max_elements_A_l144_144042

noncomputable def A (N : ℕ) : Finset ℕ :=
  (Finset.range N).filter (λ x, x % 32 = 1 ∨ x % 32 = 5 ∨ x % 32 = 9 ∨ 
                           x % 32 = 10 ∨ x % 32 = 13 ∨ x % 32 = 14 ∨ 
                           x % 32 = 17 ∨ x % 32 = 21 ∨ x % 32 = 25 ∨ 
                           x % 32 = 29 ∨ x % 32 = 30)

theorem max_elements_A (N : ℕ) : 
  (A N).card ≥ (11 * N) / 32 :=
sorry

end max_elements_A_l144_144042


namespace minimum_dot_product_of_tangents_l144_144670

noncomputable def circleO := { P : ℝ × ℝ // P.1^2 + P.2^2 = 4 }

noncomputable def circleM (θ : ℝ) := 
  { P : ℝ × ℝ // (P.1 - 5 * Real.cos θ)^2 + (P.2 - 5 * Real.sin θ)^2 = 1 }

theorem minimum_dot_product_of_tangents (θ : ℝ) (P : ℝ × ℝ) 
  (hP : (P ∈ circleM θ)) :
  ∃ E F : ℝ × ℝ, 
  (E ∈ circleO ∧ F ∈ circleO) ∧
  collinear (insert (submodule.span ℝ {E, F}) 0) 
  ∧  ( ∀ (Q: ℝ × ℝ), collinear (insert (submodule.span ℝ {E, F, Q}) 0) →    
  ((P - E) • (P - F)) ≥ 6 )  :=
sorry

end minimum_dot_product_of_tangents_l144_144670


namespace find_p_l144_144285

noncomputable def A_x : ℝ := (3 * real.sqrt 3) / 2
noncomputable def A_y : ℝ := 9 / 2
noncomputable def A := (A_x, A_y)
noncomputable def O := (0, 0)
noncomputable def M := (0, 9)

def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

theorem find_p (p : ℝ) (h1 : parabola p A.1 A.2) (h2 : p > 0)
  (h3 : dist A M = dist A O) (h4 : dist A O = dist B O) (h5 : ∃ B, dist A B = dist B O) : p = 3 / 4 := 
begin 
  sorry 
end

end find_p_l144_144285


namespace increasing_intervals_lambda_range_l144_144679

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x^2 - (2 * a + 2) * x + (2 * a + 1) * Real.log x

theorem increasing_intervals (a : ℝ) (ha : a ∈ Set.Icc (3/2 : ℝ) (5/2 : ℝ)) :
  (f' x a > 0 → x ∈ Set.union (Set.Ioo 0 1) (Set.Ioi (2 * a + 1))) ∧ 
  (f' x a < 0 → x ∈ Set.Ioo 1 (2 * a + 1)) :=
sorry

theorem lambda_range (λ : ℝ) (a : ℝ) (ha : a ∈ Set.Icc (3/2 : ℝ) (5/2 : ℝ))
  (x₁ x₂ : ℝ) (hx₁ : x₁ ∈ Set.Icc 0 2) (hx₂ : x₂ ∈ Set.Icc 0 2) (x₁_ne_x₂ : x₁ ≠ x₂)
  (h : |f x a x₁ - f x a x₂| < λ * |1 / x₁ - 1 / x₂|) :
  λ ≥ 8 :=
sorry

end increasing_intervals_lambda_range_l144_144679


namespace xyz_sum_fraction_l144_144619

theorem xyz_sum_fraction (a1 a2 a3 b1 b2 b3 c1 c2 c3 a b c : ℤ) 
  (h1 : a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1) = 9)
  (h2 : a * (b2 * c3 - b3 * c2) - a2 * (b * c3 - b3 * c) + a3 * (b * c2 - b2 * c) = 17)
  (h3 : a1 * (b * c3 - b3 * c) - a * (b1 * c3 - b3 * c1) + a3 * (b1 * c - b * c1) = -8)
  (h4 : a1 * (b2 * c - b * c2) - a2 * (b1 * c - b * c1) + a * (b1 * c2 - b2 * c1) = 7)
  (eq1 : a1 * x + a2 * y + a3 * z = a)
  (eq2 : b1 * x + b2 * y + b3 * z = b)
  (eq3 : c1 * x + c2 * y + c3 * z = c)
  : x + y + z = 16 / 9 := 
sorry

end xyz_sum_fraction_l144_144619


namespace planes_perpendicular_iff_line_perpendicular_l144_144770

variables {α β : Type} [Plane α] [Plane β] (c : Line (α ∩ β))

-- Define what it means for two planes to be perpendicular
def planes_perpendicular (α β : Type) [Plane α] [Plane β] : Prop :=
  dihedral_angle α β = 90

-- Define what it means for a line to be perpendicular to a plane
def line_perpendicular_plane (a : Line α) (β : Type) [Plane β] : Prop :=
  ∃ (M : α), a.contains M ∧ orthogonal (a.direction_at M : vector) (plane_normal β)

-- The main statement to be proved
theorem planes_perpendicular_iff_line_perpendicular (α β : Type) [Plane α] [Plane β] (c : Line (α ∩ β)) :
  planes_perpendicular α β ↔ ∃ (a : Line α), line_perpendicular_plane a β :=
sorry

end planes_perpendicular_iff_line_perpendicular_l144_144770


namespace g_5_l144_144794

variable (g : ℝ → ℝ)

axiom additivity_condition : ∀ (x y : ℝ), g (x + y) = g x + g y
axiom g_1_nonzero : g 1 ≠ 0

theorem g_5 : g 5 = 5 * g 1 :=
by
  sorry

end g_5_l144_144794


namespace min_value_x2y2z2_l144_144001

open Real

noncomputable def condition (x y z : ℝ) : Prop := (1 / x + 1 / y + 1 / z = 3)

theorem min_value_x2y2z2 (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : condition x y z) :
  x^2 * y^2 * z^2 ≥ 1 / 64 :=
by
  sorry

end min_value_x2y2z2_l144_144001


namespace sasha_or_maxim_must_be_mistaken_l144_144032

theorem sasha_or_maxim_must_be_mistaken
    (a : ℕ → ℕ → ℕ)
    (non_zero : ∀ i j, a i j ≠ 0)
    (sasha : ∀ i, ∑ j in finset.range 100, a i j % 9 = 0)
    (maxim : ∃! j, ∑ i in finset.range 100, a i j % 9 ≠ 0 ∧ ∀ k ≠ j, ∑ i in finset.range 100, a i k % 9 = 0) :
    false :=
begin
  -- Proof omitted
  sorry
end

end sasha_or_maxim_must_be_mistaken_l144_144032


namespace find_y_value_l144_144697

theorem find_y_value (y : ℕ) : (1/8 * 2^36 = 2^33) ∧ (8^y = 2^(3 * y)) → y = 11 :=
by
  intros h
  -- additional elaboration to verify each step using Lean, skipped for simplicity
  sorry

end find_y_value_l144_144697


namespace ball_bounce_height_l144_144147

theorem ball_bounce_height (h : ℝ) (r : ℝ) (k : ℕ) (hk : h * r^k < 6) :
  h = 2000 ∧ r = 1/3 → k = 6 :=
by
  intros h_cond r_cond
  rw [h_cond, r_cond] at hk
  sorry

end ball_bounce_height_l144_144147


namespace law_of_cosines_l144_144715

theorem law_of_cosines {α β γ : ℝ} (a b c : ℝ) (A B C : ℝ)
  (hA : A + B + C = π) (ha : a = sqrt (b^2 + c^2 - 2 * b * c * cos A)) :
  a^2 = b^2 + c^2 - 2 * b * c * cos A :=
by sorry

end law_of_cosines_l144_144715


namespace oblique_projection_properties_l144_144488

/--
Under the oblique projection method, the following properties hold:
1. A triangle transforms to a triangle.
2. A parallelogram transforms to a parallelogram.
3. A square does not transform to a square.
4. A rhombus does not transform to a rhombus.
--/
theorem oblique_projection_properties:
  (∀ (T : Type) [is_triangle T], is_triangle (oblique_projection T)) ∧
  (∀ (P : Type) [is_parallelogram P], is_parallelogram (oblique_projection P)) ∧
  (∀ (S : Type) [is_square S], ¬ is_square (oblique_projection S)) ∧
  (∀ (R : Type) [is_rhombus R], ¬ is_rhombus (oblique_projection R)) := 
sorry

end oblique_projection_properties_l144_144488


namespace valid_four_digit_numbers_count_l144_144310

noncomputable def num_valid_four_digit_numbers : ℕ := 6 * 65 * 10

theorem valid_four_digit_numbers_count :
  num_valid_four_digit_numbers = 3900 :=
by
  -- We provide the steps in the proof to guide the automation
  let total_valid_numbers := 6 * 65 * 10
  have h1 : total_valid_numbers = 3900 := rfl
  exact h1

end valid_four_digit_numbers_count_l144_144310


namespace equal_parallelogram_areas_locus_l144_144892

theorem equal_parallelogram_areas_locus 
  {a b c : ℝ} {A B C P : Point} 
  (hA : A = (0, a)) (hB : B = (-b, 0)) (hC : C = (c, 0)) 
  (hP : P ∈ interior (triangle ABC)) :
  (locus_of_equal_areas GPDC FPEB P) ↔
  2 * a * P.x + (c - b) * P.y + a * (b - c) = 0 := sorry

end equal_parallelogram_areas_locus_l144_144892


namespace intercept_sum_l144_144802

noncomputable def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

def x_intercept := 4

def y_intercepts : ℝ × ℝ :=
  let delta := (9 : ℝ)^2 - 4 * 3 * 4
  ((9 - Real.sqrt delta)/6, (9 + Real.sqrt delta)/6)

def a : ℝ := x_intercept
def b : ℝ := (y_intercepts.1)
def c : ℝ := (y_intercepts.2)

theorem intercept_sum : a + b + c = 7 := by
  have h_delta : (9 : ℝ)^2 - 4 * 3 * 4 = 33 := by
    sorry
  have h_b : b = (9 - Real.sqrt 33) / 6 := by
    simp only [b, y_intercepts]
    sorry
  have h_c : c = (9 + Real.sqrt 33) / 6 := by
    simp only [c, y_intercepts]
    sorry
  simp [a, b, c, h_b, h_c]
  field_simp
  ring
  sorry

end intercept_sum_l144_144802


namespace log_base4_one_over_sixty_four_l144_144231

theorem log_base4_one_over_sixty_four : log 4 (1 / 64) = -3 := by
  sorry

end log_base4_one_over_sixty_four_l144_144231


namespace determine_k_l144_144212

-- Assumptions and Definitions
variable {x k : ℝ}
variable h₁ : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 12)
variable h₂ : k ≠ 0

-- Main theorem to be proven
theorem determine_k (h₁ : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 12)) (h₂ : k ≠ 0) : k = 12 :=
sorry -- placeholder for the proof

end determine_k_l144_144212


namespace largest_n_for_factorable_polynomial_l144_144242

theorem largest_n_for_factorable_polynomial :
  (∃ (A B : ℤ), A * B = 72 ∧ ∀ (n : ℤ), n = 3 * B + A → n ≤ 217) ∧
  (∃ (A B : ℤ), A * B = 72 ∧ 3 * B + A = 217) :=
by
    sorry

end largest_n_for_factorable_polynomial_l144_144242


namespace find_b_l144_144295

variable (x y b : ℝ)
variable (F1 F2 A B l : Set ℝ)
variable (ellipse : Set ℝ)

axiom ellipse_def :
  ellipse = {p | let xp := p.1 in let yp := p.2 in 
    xp^2 / 4 + yp^2 / b^2 = 1}

axiom b_range : 0 < b ∧ b < 2

axiom foci_prop :
  (∃ F1 F2 : ℝ × ℝ, ∀ p ∈ ellipse, dist p F1 + dist p F2 = 4 * sqrt 2)

axiom line_intersection :
  ∃ A B : ℝ × ℝ, l = λ(p : ℝ × ℝ), p = F1 ∧ p ∈ ellipse ∧ p = A ∧ p = B

axiom max_distance :
  (| dist B F2 + dist A F2 | = 5)

theorem find_b (hp : p ∈ ellipse) (hF1 : F1 ∈ F1) (hF2 : F2 ∈ F2)
  (hA : A ∈ A) (hB : B ∈ B):
  b = sqrt 3 :=
by sorry

end find_b_l144_144295


namespace quadratic_root_diff_one_l144_144669

theorem quadratic_root_diff_one {p q : ℝ} (h_pos_p : p > 0) (h_pos_q : q > 0) 
  (h_eqn : ∀ x, x^2 + p * x + q = 0) (h_diff : ∃ x1 x2, (x1 - x2 = 1) ∧ (x^2 + p * x + q = 0)) : 
  p = sqrt (4 * q + 1) := by
  sorry

end quadratic_root_diff_one_l144_144669


namespace gcd_factorial_l144_144860

theorem gcd_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  sorry

end gcd_factorial_l144_144860


namespace pointA_is_in_QuadrantIII_l144_144717

-- Definition of the quadrants
def isQuadrantI (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def isQuadrantII (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

def isQuadrantIII (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

def isQuadrantIV (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Coordinates of point A
def pointA : ℝ × ℝ := (-1, -3)

-- Proof statement
theorem pointA_is_in_QuadrantIII : isQuadrantIII (fst pointA) (snd pointA) :=
by
  sorry

end pointA_is_in_QuadrantIII_l144_144717


namespace intercept_sum_l144_144803

noncomputable def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

def x_intercept := 4

def y_intercepts : ℝ × ℝ :=
  let delta := (9 : ℝ)^2 - 4 * 3 * 4
  ((9 - Real.sqrt delta)/6, (9 + Real.sqrt delta)/6)

def a : ℝ := x_intercept
def b : ℝ := (y_intercepts.1)
def c : ℝ := (y_intercepts.2)

theorem intercept_sum : a + b + c = 7 := by
  have h_delta : (9 : ℝ)^2 - 4 * 3 * 4 = 33 := by
    sorry
  have h_b : b = (9 - Real.sqrt 33) / 6 := by
    simp only [b, y_intercepts]
    sorry
  have h_c : c = (9 + Real.sqrt 33) / 6 := by
    simp only [c, y_intercepts]
    sorry
  simp [a, b, c, h_b, h_c]
  field_simp
  ring
  sorry

end intercept_sum_l144_144803


namespace fraction_of_square_above_line_l144_144809

theorem fraction_of_square_above_line :
  let A := (2, 1)
  let B := (5, 1)
  let C := (5, 4)
  let D := (2, 4)
  let P := (2, 3)
  let Q := (5, 1)
  ∃ f : ℚ, f = 2 / 3 := 
by
  -- Placeholder for the proof
  sorry

end fraction_of_square_above_line_l144_144809


namespace prove_problem_statement_l144_144387

noncomputable def problem_statement (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : ab + ac + bc ≠ 0) : Prop :=
  (a^7 + b^7 + c^7) / (abc * (ab + ac + bc)) = -7

theorem prove_problem_statement (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : ab + ac + bc ≠ 0) : 
  problem_statement a b c h1 h2 h3 h4 h5 :=
by 
  sorry

end prove_problem_statement_l144_144387


namespace ounces_in_pound_l144_144409

theorem ounces_in_pound : 
  (ton_to_pounds : ℕ) 
  (packet_count : ℕ) 
  (weight_pounds : ℕ) 
  (weight_ounces : ℕ) 
  (bag_capacity_tons : ℕ) 
  (conversion_result : ℕ)
  (total_weight_pounds := packet_count * (weight_pounds + weight_ounces / conversion_result)) 
  (bag_capacity_pounds := bag_capacity_tons * ton_to_pounds) :
  ton_to_pounds = 2500 ∧ packet_count = 2000 ∧ weight_pounds = 16 ∧ weight_ounces = 4 ∧ bag_capacity_tons = 13 → 
  total_weight_pounds = bag_capacity_pounds → 
  conversion_result = 16 :=
by
  sorry

end ounces_in_pound_l144_144409


namespace general_formulas_sum_elements_in_matrix_l144_144665

-- Given conditions
def S (n : ℕ) : ℕ := 2^(n + 1) - 2
def b (n : ℕ) : ℕ := Int.log2 (a n)

-- Problem: Prove general formulas for sequences {a_n} and {b_n}
theorem general_formulas (n : ℕ) : 
  ∃ a : ℕ → ℕ, (∀ n, S n = ∑ i in Finset.range (n+1), a i)
  ∧ (∀ n, a n = 2^n)
  ∧ (∀ n, b n = n) := 
sorry

-- Problem: Prove the sum of all elements in the matrix T_n
theorem sum_elements_in_matrix (n : ℕ) :
  let a_seq := λ i : ℕ, 2^i
  let b_seq := λ j : ℕ, j
  let T_n := ∑ i in Finset.range n, ∑ j in Finset.range n, a_seq i * b_seq j
  T_n = n * (n + 1) * (2^n - 1) :=
sorry

end general_formulas_sum_elements_in_matrix_l144_144665


namespace minimum_value_of_xyz_l144_144650

variables {x y z : ℝ}

def equation1 : Prop := x * y + 2 * z = 1
def equation2 : Prop := x^2 + y^2 + z^2 = 5

theorem minimum_value_of_xyz :
  (∃ (x y z : ℝ), equation1 ∧ equation2) →
  (∀ (x y z : ℝ), equation1 ∧ equation2 → (xyz : ℝ) = x * y * z) →
  ∃ (x y z : ℝ), equation1 ∧ equation2 ∧ (xyz = 9 * sqrt 11 - 32) :=
by
  sorry

end minimum_value_of_xyz_l144_144650


namespace fixed_point_exists_l144_144262

noncomputable theory

open Real

def circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

def line (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y = 7 * m + 4

def fixed_point (P : ℝ × ℝ) : Prop :=
  ∀ m : ℝ, line m P.1 P.2

theorem fixed_point_exists :
  fixed_point (3, 1) ∧ (∀ x y : ℝ, circle x y → (x + y = 4 → (∃ a b : ℝ, circle a b ∧ a ≠ b ∧ (a + b = 4) ∧ (a - b).abs = (7 * sqrt 2)))) :=
sorry

end fixed_point_exists_l144_144262


namespace intercept_sum_l144_144804

noncomputable def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

def x_intercept := 4

def y_intercepts : ℝ × ℝ :=
  let delta := (9 : ℝ)^2 - 4 * 3 * 4
  ((9 - Real.sqrt delta)/6, (9 + Real.sqrt delta)/6)

def a : ℝ := x_intercept
def b : ℝ := (y_intercepts.1)
def c : ℝ := (y_intercepts.2)

theorem intercept_sum : a + b + c = 7 := by
  have h_delta : (9 : ℝ)^2 - 4 * 3 * 4 = 33 := by
    sorry
  have h_b : b = (9 - Real.sqrt 33) / 6 := by
    simp only [b, y_intercepts]
    sorry
  have h_c : c = (9 + Real.sqrt 33) / 6 := by
    simp only [c, y_intercepts]
    sorry
  simp [a, b, c, h_b, h_c]
  field_simp
  ring
  sorry

end intercept_sum_l144_144804


namespace jellybean_count_l144_144609

noncomputable def original_jar_count (x : ℕ) : Prop :=
  0.75^3 * x = 27 ∧ x = 64

theorem jellybean_count : ∃ x : ℕ, original_jar_count x :=
by
  sorry

end jellybean_count_l144_144609


namespace distance_each_player_runs_l144_144054

-- Definitions based on conditions
def length : ℝ := 100
def width : ℝ := 50
def laps : ℝ := 6

def perimeter (l w : ℝ) : ℝ := 2 * (l + w)

def total_distance (l w laps : ℝ) : ℝ := laps * perimeter l w

-- Theorem statement
theorem distance_each_player_runs :
  total_distance length width laps = 1800 := 
by 
  sorry

end distance_each_player_runs_l144_144054


namespace correct_answer_is_B_l144_144123

-- Definitions for each set of line segments
def setA := (2, 2, 4)
def setB := (8, 6, 3)
def setC := (2, 6, 3)
def setD := (11, 4, 6)

-- Triangle inequality theorem checking function
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statements to verify each set
lemma check_setA : ¬ is_triangle 2 2 4 := by sorry
lemma check_setB : is_triangle 8 6 3 := by sorry
lemma check_setC : ¬ is_triangle 2 6 3 := by sorry
lemma check_setD : ¬ is_triangle 11 4 6 := by sorry

-- Final theorem combining all checks to match the given problem
theorem correct_answer_is_B : 
  ¬ is_triangle 2 2 4 ∧ is_triangle 8 6 3 ∧ ¬ is_triangle 2 6 3 ∧ ¬ is_triangle 11 4 6 :=
by sorry

end correct_answer_is_B_l144_144123


namespace isosceles_triangle_max_area_l144_144022

theorem isosceles_triangle_max_area {α p : ℝ} (hα : 0 < α ∧ α < π)
  (hABC : exists (A B C : Type), has_fixed_angle A B C α ∧ semiperimeter A B C = p) :
  ∀ (A B C : Type), is_isosceles A B C → (area A B C ≥ area_of_any_other_triangle_with_same_conditions A B C α p) :=
by
  sorry

end isosceles_triangle_max_area_l144_144022


namespace parallel_lines_condition_l144_144741

theorem parallel_lines_condition (a : ℝ) :
  (a = 1) → (a ≠ -2) → (a * (a + 1) = 2) → (line_1_parallel_to_line_2) :=
sorry

/--  Definition that captures the condition for lines being parallel./
def line_1_parallel_to_line_2 {a : ℝ} : Prop :=
  ∀ x y : ℝ, (ax + 2y = 0) → (x + (a+1)y + 4 = 0) → False :=
sorry

end parallel_lines_condition_l144_144741


namespace find_radius_of_stationary_tank_l144_144557

theorem find_radius_of_stationary_tank
  (h_stationary : Real) (r_truck : Real) (h_truck : Real) (drop : Real) (V_truck : Real)
  (ht1 : h_stationary = 25)
  (ht2 : r_truck = 4)
  (ht3 : h_truck = 10)
  (ht4 : drop = 0.016)
  (ht5 : V_truck = π * r_truck ^ 2 * h_truck) :
  ∃ R : Real, π * R ^ 2 * drop = V_truck ∧ R = 100 :=
by
  sorry

end find_radius_of_stationary_tank_l144_144557


namespace log_base_4_frac_l144_144222

theorem log_base_4_frac :
  logb 4 (1/64) = -3 :=
sorry

end log_base_4_frac_l144_144222


namespace factor_expression_l144_144994

variable (a b : ℤ)

theorem factor_expression : 2 * a^2 * b - 4 * a * b^2 + 2 * b^3 = 2 * b * (a - b)^2 := 
sorry

end factor_expression_l144_144994


namespace range_of_m_in_second_quadrant_l144_144720

theorem range_of_m_in_second_quadrant (m : ℝ) :
  let z := m^2 * (1 + Complex.i) - m * (4 + Complex.i) - 6 * Complex.i in
  (0 < m ∧ m < 4) ∧ (m > 3 ∨ m < -2) → 3 < m ∧ m < 4 :=
sorry

end range_of_m_in_second_quadrant_l144_144720


namespace pedestrians_speed_ratio_l144_144835

-- Definitions based on conditions
variable (v v1 v2 : ℝ)

-- Conditions
def first_meeting (v1 v : ℝ) := (1 / 3) * v1 = (1 / 4) * v
def second_meeting (v2 v : ℝ) := (5 / 12) * v2 = (1 / 6) * v

-- Theorem Statement
theorem pedestrians_speed_ratio (h1 : first_meeting v1 v) (h2 : second_meeting v2 v) : v1 / v2 = 15 / 8 :=
by
  -- Proof will go here
  sorry

end pedestrians_speed_ratio_l144_144835


namespace base5_division_l144_144992

theorem base5_division :
  ∀ (a b : ℕ), a = 1121 ∧ b = 12 → 
   ∃ (q r : ℕ), (a = b * q + r) ∧ (r < b) ∧ (q = 43) :=
by sorry

end base5_division_l144_144992


namespace highest_vs_lowest_temp_difference_l144_144805

theorem highest_vs_lowest_temp_difference 
  (highest_temp lowest_temp : ℤ) 
  (h_highest : highest_temp = 26) 
  (h_lowest : lowest_temp = 14) : 
  highest_temp - lowest_temp = 12 := 
by 
  sorry

end highest_vs_lowest_temp_difference_l144_144805


namespace train_length_l144_144174

-- Definitions based on conditions
def train_speed_kmh := 54 -- speed of the train in km/h
def time_to_cross_sec := 16 -- time to cross the telegraph post in seconds
def kmh_to_ms (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 5 / 18 -- conversion factor from km/h to m/s

-- Prove that the length of the train is 240 meters
theorem train_length (h1 : train_speed_kmh = 54) (h2 : time_to_cross_sec = 16) : 
  (kmh_to_ms train_speed_kmh * time_to_cross_sec) = 240 := by
  sorry

end train_length_l144_144174


namespace product_difference_of_squares_ninety_nine_times_one_hundred_and_one_l144_144486

theorem product_difference_of_squares (a b : ℕ) (h1: a = 100) (h2: b = 1) :
  (a - b) * (a + b) = a^2 - b^2 := by
  rw [h1, h2]
  exact eq.refl (100 - 1) * (100 + 1) = 100^2 - 1^2

theorem ninety_nine_times_one_hundred_and_one (a b : ℕ) (h1: a = 100) (h2: b = 1) :
  (a - b) * (a + b) = 9999 := by
  rw [h1, h2]
  exact calc
    (100 - 1) * (100 + 1) = 100^2 - 1^2   : by rw mul_self_sub_mul_self
    ...                    = 10000 - 1    : by rw [pow_two, pow_two]
    ...                    = 9999         : by norm_num


end product_difference_of_squares_ninety_nine_times_one_hundred_and_one_l144_144486


namespace average_rounded_to_4_l144_144811

-- Definitions for the given conditions
def rounds_played : List (ℕ × ℕ) :=
  [(4, 1), (3, 2), (5, 3), (6, 4), (2, 5), (7, 6)]

-- Function to compute total rounds
def total_rounds (rp : List (ℕ × ℕ)) : ℕ :=
  rp.map (λ ⟨members, rounds⟩ => members * rounds).sum

-- Function to compute total members
def total_members (rp : List (ℕ × ℕ)) : ℕ :=
  rp.map Prod.fst.sum

-- Function to compute the average rounds
def average_rounds (rp : List (ℕ × ℕ)) : ℚ :=
  (total_rounds rp : ℚ) / (total_members rp)

-- Rounded average
noncomputable def rounded_average (rp : List (ℕ × ℕ)) : ℕ :=
  Int.toNat (Rat.round (average_rounds rp))

-- Proof statement
theorem average_rounded_to_4 :
  rounded_average rounds_played = 4 :=
by sorry

end average_rounded_to_4_l144_144811


namespace find_length_of_second_train_l144_144893

noncomputable def length_of_second_train 
  (l1 : ℕ) (v1 : ℕ) (v2 : ℕ) (t : ℕ) : ℕ :=
l2

theorem find_length_of_second_train
  (l1 : ℕ) (v1 : ℕ) (v2 : ℕ) (t : ℕ) 
  (hv1 : v1 = 120) (hv2 : v2 = 80) (ht : t = 9) 
  (hl1 : l1 = 90) : 
  length_of_second_train l1 v1 v2 t = 410 :=
by 
  -- Perfect;
  sorry

end find_length_of_second_train_l144_144893


namespace they_will_meet_probability_l144_144129

noncomputable theory

def xiaoQiangArrival := 40
def xiaoHuaArrival := {x : ℝ // 0 < x ∧ x < 60}
def waitingTime := 10

def probabilityMeeting : ℝ :=
  let totalTime := 60
  let meetingTime := (xiaoQiangArrival, xiaoQiangArrival + waitingTime)
  let intervalLength := min meetingTime.2 totalTime - meetingTime.1 -- Ensuring not to exceed total interval
  intervalLength / totalTime

theorem they_will_meet_probability :
  probabilityMeeting = 1 / 3 :=
by
  sorry

end they_will_meet_probability_l144_144129


namespace domain_of_f_l144_144844

noncomputable def f (x : ℝ) : ℝ := Real.log 2 (Real.log 3 (Real.log 5 (Real.log 7 x)))

theorem domain_of_f {x : ℝ} : x > 16807 ↔ ∀ f (x : ℝ), ∃ y : ℝ, f x = y := 
begin
  sorry
end

end domain_of_f_l144_144844


namespace gcd_of_factorials_l144_144855

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definitions for the two factorial terms involved
def term1 : ℕ := factorial 7
def term2 : ℕ := factorial 10 / factorial 4

-- Definition for computing the GCD
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- The statement to prove
theorem gcd_of_factorials : gcd term1 term2 = 2520 :=
by {
  -- Here, the proof would go.
  sorry
}

end gcd_of_factorials_l144_144855


namespace playground_area_l144_144807

noncomputable def length (w : ℝ) := 2 * w + 30
noncomputable def perimeter (l w : ℝ) := 2 * (l + w)
noncomputable def area (l w : ℝ) := l * w

theorem playground_area :
  ∃ (w l : ℝ), length w = l ∧ perimeter l w = 700 ∧ area l w = 25955.56 :=
by {
  sorry
}

end playground_area_l144_144807


namespace single_elimination_23_teams_games_needed_l144_144569

theorem single_elimination_23_teams_games_needed : ∀ {teams : ℕ}, teams = 23 → ∃ games : ℕ, games = 22 ∧
  (∀ t : ℕ, (∀ n : ℕ, n < t → t - n - 1) = games) :=
by
  sorry

end single_elimination_23_teams_games_needed_l144_144569


namespace symmetry_about_neg_pi_div_3_l144_144179

def f (x : ℝ) : ℝ := 
  cos (2 * (x + (π / 6)) + (π / 3))

theorem symmetry_about_neg_pi_div_3 : 
  ∀ x : ℝ, f(- π / 3 - x) = f(- π / 3 + x) :=
by sorry

end symmetry_about_neg_pi_div_3_l144_144179


namespace tangent_line_eq_at_1_max_value_on_interval_unique_solution_exists_l144_144646

noncomputable def f (x : ℝ) : ℝ := x^3 - x
noncomputable def g (x : ℝ) : ℝ := 2 * x - 3

theorem tangent_line_eq_at_1 : 
  ∃ c : ℝ, ∀ x y : ℝ, y = f x → (x = 1 → y = 0) → y = 2 * (x - 1) → 2 * x - y - 2 = 0 := 
by sorry

theorem max_value_on_interval :
  ∃ xₘ : ℝ, (0 ≤ xₘ ∧ xₘ ≤ 2) ∧ ∀ x : ℝ, (0 ≤ x ∧ x ≤ 2) → f x ≤ 6 :=
by sorry

theorem unique_solution_exists :
  ∃! x₀ : ℝ, f x₀ = g x₀ :=
by sorry

end tangent_line_eq_at_1_max_value_on_interval_unique_solution_exists_l144_144646


namespace highest_lowest_difference_l144_144251

variable (x1 x2 x3 x4 x5 x_max x_min : ℝ)

theorem highest_lowest_difference (h1 : x1 + x2 + x3 + x4 + x5 - x_max = 37.84)
                                  (h2 : x1 + x2 + x3 + x4 + x5 - x_min = 38.64):
                                  x_max - x_min = 0.8 := 
by
  sorry

end highest_lowest_difference_l144_144251


namespace closest_point_on_plane_l144_144606

theorem closest_point_on_plane :
  let Q := (30 / 7, 10 / 7, -15 / 7) in
  Q.1 * 2 + Q.2 * 3 - Q.3 = 15 ∧ 
  ∀ P : ℝ × ℝ × ℝ, 
  (P.1 * 2 + P.2 * 3 - P.3 = 15) → 
  ∥(Q.1 - 4, Q.2 - 1, Q.3 + 2)∥ ≤ ∥(P.1 - 4, P.2 - 1, P.3 + 2)∥
:= 
by
  -- proof omitted
  sorry

end closest_point_on_plane_l144_144606


namespace range_of_y_l144_144816

-- Define the function
def quadratic_function (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the interval condition
def in_interval (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2

-- Define the range condition as a proposition to be proved
def range_of_quadratic_function : Set ℝ := {y | ∃ x : ℝ, in_interval x ∧ quadratic_function x = y}

-- The main statement to be proved
theorem range_of_y : range_of_quadratic_function = Set.Icc 2 6 :=
by {
  -- Proof steps will go here
  sorry,
}

end range_of_y_l144_144816


namespace arithmetic_sequence_problem_l144_144063

theorem arithmetic_sequence_problem :
  ∃ (n : ℕ), 
    let x := 32 / 7 in
    let a := 3 * x - 5 in
    let b := 7 * x - 17 in
    let c := 4 * x + 3 in
    let d := b - a in
    let an := a + (n - 1) * d in
    an = 4021 →
    n = 639 :=
begin
  -- Placeholder for the proof.
  sorry
end

end arithmetic_sequence_problem_l144_144063


namespace find_f_of_f_neg10_l144_144676

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then 2^(x - 2) else log x

theorem find_f_of_f_neg10 : f (f (-10)) = 1 / 2 :=
by
  have h1 : f (-10) = 1, from
    if_neg (show -10 > 0, by norm_cast)
    have hf : ∀ x < 0, f x = log x := λ x hx, if_neg hx.sqrt_le_zero

  let h2 : f 1 = 1 / 2, from
    if_pos (nat.le_of_lt_one h1)

  sorry

end find_f_of_f_neg10_l144_144676


namespace IntermediateValueTheorem_l144_144445

variables {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

theorem IntermediateValueTheorem (f : α → ℝ) {a b : α} 
  (h_cont : ContinuousOn f (Set.Icc a b)) 
  (h_sign : f a * f b ≤ 0) :
  ∃ x₀ ∈ Set.Icc a b, f x₀ = 0 :=
begin
  sorry
end

end IntermediateValueTheorem_l144_144445


namespace constant_term_binomial_expansion_eq_l144_144341

theorem constant_term_binomial_expansion_eq (a : ℝ) (h : a > 0) :
  let term := (choose 6 2) * a^2 in
  term = 15/4 → a = 1/2 :=
by
  sorry

end constant_term_binomial_expansion_eq_l144_144341


namespace paytons_score_l144_144403

theorem paytons_score (total_score_14_students : ℕ)
    (average_14_students : total_score_14_students / 14 = 80)
    (total_score_15_students : ℕ)
    (average_15_students : total_score_15_students / 15 = 81) :
  total_score_15_students - total_score_14_students = 95 :=
by
  sorry

end paytons_score_l144_144403


namespace determine_B_l144_144575

noncomputable def polynomial : Polynomial ℤ :=
  Polynomial.C 32 + Polynomial.X + Polynomial.C D * Polynomial.X +
  Polynomial.C C * Polynomial.X ^ 2 + Polynomial.C B * Polynomial.X ^ 3 +
  Polynomial.C A * Polynomial.X ^ 5 + Polynomial.X ^ 7 - Polynomial.C 15 * Polynomial.X ^ 6

theorem determine_B (roots : Multiset ℕ) (h_sum : roots.sum = 15) (h_pos : ∀ r ∈ roots, r > 0) :
  let B := -(roots.elems.combinations 3).sum (λ xs, xs.prod)
  B = -306 := by
  sorry

end determine_B_l144_144575


namespace number_of_chords_of_ten_points_on_circle_number_of_regions_by_chords_in_circle_l144_144047

theorem number_of_chords_of_ten_points_on_circle : 
  (nat.choose 10 2) = 45 :=
by
  sorry

theorem number_of_regions_by_chords_in_circle :
  let R (n : ℕ) := 1 + nat.choose n 2 + nat.choose n 4 in
  R 10 = 256 :=
by
  sorry

end number_of_chords_of_ten_points_on_circle_number_of_regions_by_chords_in_circle_l144_144047


namespace curve_no_lattice_points_l144_144603

theorem curve_no_lattice_points :
  ∀ (a b : ℤ), b ≠ (a^2 - a + 1) / 5 :=
begin
  intros a b,
  unfold division,
  sorry
end

end curve_no_lattice_points_l144_144603


namespace log_base4_one_over_sixty_four_l144_144227

theorem log_base4_one_over_sixty_four : log 4 (1 / 64) = -3 := by
  sorry

end log_base4_one_over_sixty_four_l144_144227


namespace decagon_perimeter_l144_144983

theorem decagon_perimeter (n : ℕ) (h : n = 10) (a b : ℕ) (ha : a = 3) (hb : b = 4) :
  (5 * a) + (5 * b) = 35 := by
  rw [ha, hb]
  sorry

end decagon_perimeter_l144_144983


namespace customer_payment_eq_3000_l144_144076

theorem customer_payment_eq_3000 (cost_price : ℕ) (markup_percentage : ℕ) (payment : ℕ)
  (h1 : cost_price = 2500)
  (h2 : markup_percentage = 20)
  (h3 : payment = cost_price + (markup_percentage * cost_price / 100)) :
  payment = 3000 :=
by
  sorry

end customer_payment_eq_3000_l144_144076


namespace common_ratio_of_series_l144_144187

-- Define the terms and conditions for the infinite geometric series problem.
def first_term : ℝ := 500
def series_sum : ℝ := 4000

-- State the theorem that needs to be proven: the common ratio of the series is 7/8.
theorem common_ratio_of_series (a S r : ℝ) (h_a : a = 500) (h_S : S = 4000) (h_eq : S = a / (1 - r)) :
  r = 7 / 8 :=
by
  sorry

end common_ratio_of_series_l144_144187


namespace initial_milk_water_ratio_l144_144709

theorem initial_milk_water_ratio
  (M W : ℕ)
  (h1 : M + W = 40000)
  (h2 : (M : ℚ) / (W + 1600) = 3 / 1) :
  (M : ℚ) / W = 3.55 :=
by
  sorry

end initial_milk_water_ratio_l144_144709


namespace isosceles_triangles_possible_l144_144351

theorem isosceles_triangles_possible :
  ∃ (sticks : List ℕ), (sticks = [1, 1, 2, 2, 3, 3] ∧ 
    ∀ (a b c : ℕ), a ∈ sticks → b ∈ sticks → c ∈ sticks → 
    ((a + b > c ∧ b + c > a ∧ c + a > b) → a = b ∨ b = c ∨ c = a)) :=
sorry

end isosceles_triangles_possible_l144_144351


namespace hyperbola_asymptote_tangent_to_circle_l144_144642

noncomputable def circle_eq : ℝ → ℝ → ℝ := λ x y, x^2 + y^2 - 6*x + 8

noncomputable def asymptote_hyperbola_eq (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y, y = x / m ∨ y = -x / m

noncomputable def distance_point_line (px py : ℝ) (a b c : ℝ) : ℝ :=
  abs (a * px + b * py + c) / real.sqrt (a^2 + b^2)

theorem hyperbola_asymptote_tangent_to_circle :
  ∃ m > 0, ∀ x y, circle_eq x y = 0 → (∃ x y, asymptote_hyperbola_eq m x y ∧ distance_point_line 3 0 (1 / m) (-1) 0 = 1) → m = 2 * real.sqrt 2 :=
sorry  -- Proof is omitted

end hyperbola_asymptote_tangent_to_circle_l144_144642


namespace rectangle_perimeter_l144_144069

theorem rectangle_perimeter (b l : ℕ) (h1 : l = 3 * b) (A : ℕ) (h2 : A = l * b) (h3 : A = 363) : 
  let P := 2 * (l + b) in P = 88 :=
by
  sorry

end rectangle_perimeter_l144_144069


namespace remainder_of_powers_l144_144497

theorem remainder_of_powers (n1 n2 n3 : ℕ) : (9^6 + 8^8 + 7^9) % 7 = 2 :=
by
  sorry

end remainder_of_powers_l144_144497


namespace min_n_for_triangle_pattern_l144_144096

/-- 
There are two types of isosceles triangles with a waist length of 1:
-  Type 1: An acute isosceles triangle with a vertex angle of 30 degrees.
-  Type 2: A right isosceles triangle with a vertex angle of 90 degrees.
They are placed around a point in a clockwise direction in a sequence such that:
- The 1st and 2nd are acute isosceles triangles (30 degrees),
- The 3rd is a right isosceles triangle (90 degrees),
- The 4th and 5th are acute isosceles triangles (30 degrees),
- The 6th is a right isosceles triangle (90 degrees), and so on.

Prove that the minimum value of n such that the nth triangle coincides exactly with
the 1st triangle is 23.
-/
theorem min_n_for_triangle_pattern : ∃ n : ℕ, n = 23 ∧ (∀ m < 23, m ≠ 23) :=
sorry

end min_n_for_triangle_pattern_l144_144096


namespace green_chips_count_l144_144470

variable (total_chips : ℕ) (blue_chips : ℕ) (white_percentage : ℕ) (blue_percentage : ℕ)
variable (green_percentage : ℕ) (green_chips : ℕ)

def chips_condition1 : Prop := blue_chips = (blue_percentage * total_chips) / 100
def chips_condition2 : Prop := blue_percentage = 10
def chips_condition3 : Prop := white_percentage = 50
def green_percentage_calculation : Prop := green_percentage = 100 - (blue_percentage + white_percentage)
def green_chips_calculation : Prop := green_chips = (green_percentage * total_chips) / 100

theorem green_chips_count :
  (chips_condition1) →
  (chips_condition2) →
  (chips_condition3) →
  (green_percentage_calculation) →
  (green_chips_calculation) →
  green_chips = 12 :=
by
  intros
  sorry

end green_chips_count_l144_144470


namespace rectangle_area_l144_144754

theorem rectangle_area (w L : ℝ) (h1 : L = w^2) (h2 : L + w = 25) : 
  L * w = (Real.sqrt 101 - 1)^3 / 8 := 
sorry

end rectangle_area_l144_144754


namespace cosine_of_angle_between_tangents_l144_144638

-- Define the circle's equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the point P
def P : ℝ × ℝ := (3, 2)

-- Define the center of the circle M
def M : ℝ × ℝ := (1, 1)

-- Prove the cosine of the angle between the two tangents drawn from point P to the circle is 3/5
theorem cosine_of_angle_between_tangents : 
  let distance_PM := (P.1 - M.1)^2 + (P.2 - M.2)^2 in
  let r := 1 in
  let tangent_theta := r / real.sqrt distance_PM in
  let tangent_2theta := 2 * tangent_theta / (1 - tangent_theta^2) in
  real.cos (real.atan tangent_2theta) = 3 / 5 :=
by
  sorry

end cosine_of_angle_between_tangents_l144_144638


namespace volume_of_water_needed_l144_144554

noncomputable def volume_of_water (container_radius ball_radius : ℝ) (ball_count : ℕ) : ℝ :=
  let ball_volume := (4/3) * Real.pi * (ball_radius ^ 3)
  let total_ball_volume := ball_count * ball_volume
  let water_height := container_radius + ball_radius
  let total_volume := Real.pi * (container_radius ^ 2) * water_height
  total_volume - total_ball_volume

theorem volume_of_water_needed : 
  volume_of_water 1 0.5 4 = (2/3) * Real.pi := by
  sorry

end volume_of_water_needed_l144_144554


namespace find_alpha_l144_144283

theorem find_alpha
  (α : Real)
  (h1 : α > 0)
  (h2 : α < π)
  (h3 : 1 / Real.sin α + 1 / Real.cos α = 2) :
  α = π + 1 / 2 * Real.arcsin ((1 - Real.sqrt 5) / 2) :=
sorry

end find_alpha_l144_144283


namespace smallest_integer_solution_l144_144116

theorem smallest_integer_solution (x : ℤ) : 
  (10 * x * x - 40 * x + 36 = 0) → x = 2 :=
sorry

end smallest_integer_solution_l144_144116


namespace sum_of_prime_no_integer_solution_in_congruence_l144_144608

theorem sum_of_prime_no_integer_solution_in_congruence :
  (∑ p in {2, 3}, p) = 5 :=
by sorry

end sum_of_prime_no_integer_solution_in_congruence_l144_144608


namespace vector_norm_sum_l144_144384

open Real

variables (a b : ℝ × ℝ) (m : ℝ × ℝ)
def is_midpoint (m : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def norm_squared (a : ℝ × ℝ) : ℝ :=
  a.1 ^ 2 + a.2 ^ 2

theorem vector_norm_sum :
  let a := (a.1, a.2)
  let b := (b.1, b.2)
  let m := (4 : ℝ, 10 : ℝ)
  (is_midpoint m a b) → 
  (dot_product a b = 12) →
  (norm_squared a + norm_squared b = 440) :=
by
  intros hmid hdot
  sorry

end vector_norm_sum_l144_144384


namespace max_value_of_f_l144_144072

noncomputable def f (x : ℝ) : ℝ := (1/5) * Real.sin (x + Real.pi/3) + Real.cos (x - Real.pi/6)

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 6/5 := by
  sorry

end max_value_of_f_l144_144072


namespace radius_of_circle_eq_l144_144629

-- Define the given quadratic equation representing the circle
noncomputable def circle_eq (x y : ℝ) : ℝ :=
  16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 68

-- State that the radius of the circle given by the equation is 1
theorem radius_of_circle_eq : ∃ r, (∀ x y, circle_eq x y = 0 ↔ (x - 1)^2 + (y - 1.5)^2 = r^2) ∧ r = 1 :=
by 
  use 1
  sorry

end radius_of_circle_eq_l144_144629


namespace modulo_17_residue_l144_144865

theorem modulo_17_residue :
  let a := 512
  let b := 6 * 104
  let c := 8 * 289
  let d := 5 * 68
  (a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3) % 17 = 9 :=
by
  let a := 512
  let b := 6 * 104
  let c := 8 * 289
  let d := 5 * 68
  have ha : a % 17 = 2 := by sorry
  have hb : (6 * (104 % 17)) % 17 = 9 := by sorry
  have hc : (8 * (289 % 17)) % 17 = 0 := by sorry
  have hd : (5 * (68 % 17)) % 17 = 0 := by sorry
  calc
    (a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3) % 17
        = (2 ^ 3 + 9 ^ 3 + 0 ^ 3 + 0 ^ 3) % 17 := by sorry
    ... = (8 + 729) % 17 := by sorry
    ... = (8 + 729) % 17
    ... = 737 % 17
    ... = 9 := by sorry

end modulo_17_residue_l144_144865


namespace ArithmeticSquareRoot4_SquareRoot5_CubeRootNeg27_l144_144622

theorem ArithmeticSquareRoot4 : sqrt 4 = 2 := by
  sorry

theorem SquareRoot5 : ∃ (x : ℝ), x^2 = 5 := by
  sorry

theorem CubeRootNeg27 : ∃ (x : ℝ), x^3 = -27 := by
  sorry

end ArithmeticSquareRoot4_SquareRoot5_CubeRootNeg27_l144_144622


namespace octagon_perimeter_l144_144114

theorem octagon_perimeter (side_length : ℝ) (h : side_length = 12) : 
  ∑ i in finset.range 8, side_length = 96 :=
by
  rw [finset.sum_const, finset.card_range, h]
  simp only [mul_assoc, mul_one, mul_comm]
  sorry

end octagon_perimeter_l144_144114


namespace distribution_series_l144_144155

variable (X : Type) [Probability.IsDiscrete X]
variable (x1 x2 x3 : ℝ) (P : X -> ℝ) (E : X -> ℝ) 
variable (V : X -> ℝ)
variable [h1 : Probability.random_variable X [1, x2, x3]]
variable [hx1 : Probability.object P X 1 = 0.3]
variable [hx2 : Probability.object P X x2 = 0.2]
variable [H : ((x1 * 0.3) + (x2 * 0.2) + (x3 * 0.5)) = 2.2]
variable [V.A := (5.6 - X * X [x1^2] 0.3 + x2^2 0.2  + x3^2 0.5) = 0.76]

theorem distribution_series ( x1 < x2 < x3 ) :
  inf.T 0.5 * ( sum(b1: 3). (0.3 + 0.2) = 1) :=  sorry
 
end distribution_series_l144_144155


namespace fraction_subtraction_l144_144490

theorem fraction_subtraction :
  (3 + 6 + 9) = 18 →
  (2 + 5 + 8) = 15 →
  (2 + 5 + 8) = 15 →
  (3 + 6 + 9) = 18 →
  (18 / 15 - 15 / 18) = 11 / 30 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end fraction_subtraction_l144_144490


namespace laptop_sticker_price_l144_144307

theorem laptop_sticker_price (x : ℝ) (h1 : 0.8 * x - 120 = y) (h2 : 0.7 * x = z) (h3 : y + 25 = z) : x = 950 :=
sorry

end laptop_sticker_price_l144_144307


namespace max_value_f_interval_2_to_5_l144_144451

noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x - 1)

theorem max_value_f_interval_2_to_5 : 
  (∀ x ∈ set.Ico 2 5, (x ≠ 1) ∧ (f x ≤ 4)) ∧ (f 2 = 4) ∧ (¬ ∃ x ∈ set.Ico 2 5, x ≠ 1 ∧ f x < 4) :=
by
  sorry

end max_value_f_interval_2_to_5_l144_144451


namespace volume_of_piece_containing_W_l144_144176

theorem volume_of_piece_containing_W :
  let unit_cube : ℝ^3 := {p | (0 ≤ p.1 ∧ p.1 ≤ 1) ∧ (0 ≤ p.2 ∧ p.2 ≤ 1) ∧ (0 ≤ p.3 ∧ p.3 ≤ 1)},
      cut_cube : ℝ^3 := unit_cube,
      pieces := (ℝ^3 → ℝ) := λ cut_cube, (λ x ⟨h1, h2, h3⟩, ⟨f1, f2, f3⟩),
      piece_W : ℝ^3 := λ pieces, (piece_W) -- piece containing vertex W
  in volume piece_W = 1 / 12 :=
sorry

end volume_of_piece_containing_W_l144_144176


namespace problem1_problem2_problem3_problem4_l144_144590

-- Defining each problem as a theorem statement
theorem problem1 : 20 + 3 - (-27) + (-5) = 45 :=
by sorry

theorem problem2 : (-7) - (-6 + 5 / 6) + abs (-3) + 1 + 1 / 6 = 4 :=
by sorry

theorem problem3 : (1 / 4 + 3 / 8 - 7 / 12) / (1 / 24) = 1 :=
by sorry

theorem problem4 : -1 ^ 4 - (1 - 0.4) + 1 / 3 * ((-2) ^ 2 - 6) = -2 - 4 / 15 :=
by sorry

end problem1_problem2_problem3_problem4_l144_144590


namespace estimate_expression_l144_144614

theorem estimate_expression :
  6 < (3 * Real.sqrt 6 + Real.sqrt 12) * Real.sqrt (1 / 3) ∧ 
  (3 * Real.sqrt 6 + Real.sqrt 12) * Real.sqrt (1 / 3) < 7 :=
by
  have h_sqrt2 : 1.4 < Real.sqrt 2 ∧ Real.sqrt 2 < 1.5 := sorry,
  sorry

end estimate_expression_l144_144614


namespace ellipse_x_intercept_l144_144936

theorem ellipse_x_intercept
  (foci1 foci2 : ℝ × ℝ)
  (x_intercept : ℝ × ℝ)
  (d : ℝ)
  (h_foci1 : foci1 = (0, 3))
  (h_foci2 : foci2 = (4, 0))
  (h_x_intercept : x_intercept = (0, 0))
  (h_d : d = 7)
  : ∃ x : ℝ, (x, 0) ≠ x_intercept ∧ (abs (x - 4) + real.sqrt (x^2 + 9) = 7) ∧ x = 56 / 11 := by
  sorry

end ellipse_x_intercept_l144_144936


namespace ellipse_x_intercept_l144_144935

theorem ellipse_x_intercept (x : ℝ) :
  let f1 := (0, 3)
  let f2 := (4, 0)
  let origin := (0, 0)
  let d := sqrt ((fst f1)^2 + (snd f1)^2) + sqrt ((fst f2)^2 + (snd f2)^2)
  d = 7 → -- sum of distances from origin to the foci is 7
  (d_1 : ℝ := abs x - 4 + sqrt (x^2 + 9))
  d_1 = 7 → -- sum of distances from (x, 0) to the foci is 7
  x ≠ 0 → -- x is not 0 because the other x-intercept is not (0, 0)
  x = 56 / 11 → -- x > 4
  (x, 0) = ((56 : ℝ) / 11, 0) :=
by
  sorry

end ellipse_x_intercept_l144_144935


namespace number_of_moles_of_NaNO3_l144_144247

theorem number_of_moles_of_NaNO3 :
  ∀ (x y : ℕ), (NH4NO3 + NaOH = NaNO3 + NH3 + H2O) ∧ (NH4NO3 moles) = x ∧ (NaOH moles) = y ∧ (x = 3 ∧ y = 3) 
  → (NaNO3 moles) = 3 := 
by
  sorry

end number_of_moles_of_NaNO3_l144_144247


namespace reconstruct_trihedral_angle_l144_144291

theorem reconstruct_trihedral_angle (S A B C A' B' C' : Point)
  (h1 : is_bisector S A' B SC)
  (h2 : is_bisector S B' C SA)
  (h3 : is_bisector S C' B ASB) :
  is_fixed_rotation_axis S B := 
sorry

end reconstruct_trihedral_angle_l144_144291


namespace green_chips_l144_144477

def total_chips (T : ℕ) := 3 = 0.10 * T

def white_chips (T : ℕ) := 0.50 * T

theorem green_chips (T : ℕ) (h1 : total_chips T) (h2 : white_chips T) : (T - (3 + h2) = 12) :=
by sorry

end green_chips_l144_144477


namespace perfect_family_subset_bound_l144_144157

-- Define what it means for a family of sets to be perfect
def perfect (U : Type _) (F : set (set U)) : Prop :=
  ∀ X1 X2 X3 ∈ F, 
  (X1 \ X2) ∩ X3 = ∅ ∨ (X2 \ X1) ∩ X3 = ∅

-- Define the finite type and necessary concept of cardinality
variables {U : Type _} [Fintype U]

-- The theorem definition
theorem perfect_family_subset_bound (F : set (set U)) (hF : perfect U F) : F.to_finset.card ≤ Fintype.card U + 1 :=
sorry

end perfect_family_subset_bound_l144_144157


namespace car_arrangement_valid_l144_144097

def total_parking_spaces : ℕ := 50

def total_ways_to_place_two_cars : ℕ := total_parking_spaces * (total_parking_spaces - 1)

def adjacent_car_placements : ℕ :=
  (total_parking_spaces - 1) * 2 -- for each position except the last

def valid_ways_to_place_two_cars : ℕ :=
  total_ways_to_place_two_cars - adjacent_car_placements

theorem car_arrangement_valid : valid_ways_to_place_two_cars = 2352 :=
  by
  have h1: total_ways_to_place_two_cars = total_parking_spaces * (total_parking_spaces - 1),
  sorry,
  have h2 : adjacent_car_placements = (total_parking_spaces - 1) * 2,
  sorry,
  have h3 : valid_ways_to_place_two_cars = total_ways_to_place_two_cars - adjacent_car_placements,
  sorry,
  show valid_ways_to_place_two_cars = 2352,
  sorry

end car_arrangement_valid_l144_144097


namespace triangle_problem_l144_144008

-- Definitions for the given conditions
structure Triangle (A B C D : Type) :=
(AB AC : ℝ)
(BC : ℝ)
(AD : ℝ)
(angle_sum : ℝ)

-- Defining the given problem and proof goal
theorem triangle_problem :
  let A B C D : Type := ℝ
  let triangle := Triangle.mk 22 22 11 19 90
  let BD_CD_squared := (BD^2 + CD^2)
  let frac_expr := BD_CD_squared = 361 / 4
  let a := 361
  let b := 4
  (100 * a + b) = 36104 := sorry

end triangle_problem_l144_144008


namespace three_digit_integer_divisible_by_5_l144_144093

theorem three_digit_integer_divisible_by_5 (M : ℕ) (h1 : 100 ≤ M ∧ M < 1000) (h2 : M % 10 = 5) : M % 5 = 0 := 
sorry

end three_digit_integer_divisible_by_5_l144_144093


namespace remaining_count_is_42_l144_144817

def S := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

def not_multiple_of (a b : ℕ) : Prop :=
¬ ∃ k, b = k * a

def remaining_elements_count : ℕ :=
(set.count (λ n, not_multiple_of 2 n ∧ not_multiple_of 3 n ∧ not_multiple_of 5 n) S)

theorem remaining_count_is_42 : remaining_elements_count = 42 :=
sorry

end remaining_count_is_42_l144_144817


namespace minimum_value_of_a_l144_144813

-- Define conditions and relevant properties
def has_three_positive_integer_roots (p : Polynomial ℝ) : Prop :=
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ p = Polynomial.X^3 - Polynomial.C (↑(x + y + z)) * Polynomial.X^2 + Polynomial.C (↑((x * y + y * z + z * x))) * Polynomial.X - Polynomial.C (3003)

def polynomial_with_minimum_a (p : Polynomial ℝ) (a_min : ℕ) : Prop :=
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ p = Polynomial.X^3 - Polynomial.C (↑a_min) * Polynomial.X^2 + Polynomial.C (↑((x * y + y * z + z * x))) * Polynomial.X - Polynomial.C (3003) ∧ x * y * z = 3003

-- Problem statement
theorem minimum_value_of_a (p : Polynomial ℝ) (a_min : ℕ) :
  has_three_positive_integer_roots p →
  polynomial_with_minimum_a p a_min →
  a_min = 45 :=
by
  intros hpos hmin
  sorry

end minimum_value_of_a_l144_144813


namespace monotonic_f_no_zeros_g_f_g_inequality_l144_144743
-- 1. Monotonicity of f(x)
theorem monotonic_f (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (a ≤ 0 → f(x) < 0)) ∧ 
  (a > 0 → (∀ x : ℝ, x > 0 → 
  (0 < x ∧ x < (sqrt (2 * a)) / (2 * a) → f'(x) < 0) ∧ 
  ((sqrt (2 * a)) / (2 * a) < x → f'(x) > 0))) :=
sorry

-- 2. g(x) has no zeros when x > 1
theorem no_zeros_g (x : ℝ) : 
  x > 1 → (g(x) > 0) :=
sorry

-- 3. Values of a such that f(x) > g(x) for all x ∈ (1, +∞)
theorem f_g_inequality (a : ℝ) : 
  (a ≥ 1/2) → (∀ x : ℝ, x > 1 → (f(x) > g(x))) :=
sorry

-- Definitions of f(x) and g(x)
def f (x : ℝ) (a : ℝ) : ℝ := a * x^2 - a - real.log x
def g (x : ℝ) : ℝ := 1 / x - real.exp (1 - x)

end monotonic_f_no_zeros_g_f_g_inequality_l144_144743


namespace distinct_three_digit_even_count_monotonic_increasing_sequences_count_l144_144667

open Nat

/-
 Given the digits {0, 1, 2, 3, 4, 5}:
 Prove:
 1. The number of distinct three-digit even numbers formed without repetition is 52.
 2. The number of monotonic increasing sequences formed by choosing any three different digits is 20.
-/

theorem distinct_three_digit_even_count (s : Finset ℕ) (h : s = {0, 1, 2, 3, 4, 5}) :
  ∃ n : ℕ, n = 52 ∧ (s.filter even).card * (s.erase 0).card * (s.erase 0).card = n := 
by sorry

theorem monotonic_increasing_sequences_count (s : Finset ℕ) (h : s = {0, 1, 2, 3, 4, 5}) :
  ∃ n : ℕ, n = 20 ∧ (s.card.choose 3) = n :=
by sorry

end distinct_three_digit_even_count_monotonic_increasing_sequences_count_l144_144667


namespace no_odd_n_distinct_primes_perfect_squares_l144_144213

theorem no_odd_n_distinct_primes_perfect_squares (n : ℕ) (h_n : n ≥ 3 ∧ n % 2 = 1) :
  ¬ ∃ (p : Fin n → ℕ), (∀ i, Prime (p i)) ∧ (∀ i, ∃ a : ℕ, (p i + p ((i + 1) % n)) = a^2) := by
  sorry

end no_odd_n_distinct_primes_perfect_squares_l144_144213


namespace max_min_values_l144_144671

def f (x a : ℝ) : ℝ := -x^2 + 2*x + a

theorem max_min_values (a : ℝ) (h : a ≠ 0) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 1 + a) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f x a = 1 + a) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 3 → -3 + a ≤ f x a) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f x a = -3 + a) := 
sorry

end max_min_values_l144_144671


namespace smallest_three_digit_geometric_sequence_l144_144504

theorem smallest_three_digit_geometric_sequence : ∃ n : ℕ, n = 124 ∧ (∀ (a r : ℕ), a = 1 ∧ r * a < 10 ∧ r^2 * a < 100 → digits n = [1, r, r^2] ∧ (digits n).nodup) := by
  sorry

end smallest_three_digit_geometric_sequence_l144_144504


namespace solve_problem_expr_l144_144067

noncomputable def problem_expr : ℝ :=
  (Real.log 2)^2 + (Real.log 5) * ((Real.log 20) + (2016 : ℝ)^0) + (0.027 : ℝ)^(-2/3) * (1/3 : ℝ)^(-2)

theorem solve_problem_expr : problem_expr = 102 := by
  sorry

end solve_problem_expr_l144_144067


namespace dog_age_64_human_years_l144_144099

def dog_years (human_years : ℕ) : ℕ :=
if human_years = 0 then
  0
else if human_years = 1 then
  1
else if human_years = 2 then
  2
else
  2 + (human_years - 2) / 5

theorem dog_age_64_human_years : dog_years 64 = 10 :=
by 
    sorry

end dog_age_64_human_years_l144_144099


namespace boy_late_first_day_l144_144545

open Real

theorem boy_late_first_day {d : ℝ} (h_d : d = 2.5) {v1 v2 : ℝ} (h_v1 : v1 = 5) (h_v2 : v2 = 10)
  {early : ℝ} (h_early : early = 8) : 
  let t1 := d / v1 * 60,
      t2 := d / v2 * 60 in
  t1 - (t2 + early) = 7 := by
  sorry

end boy_late_first_day_l144_144545


namespace object_speed_mph_l144_144951

-- Define the conditions given in the problem
def distance_ft : ℝ := 90
def time_sec : ℝ := 3
def feet_per_mile : ℝ := 5280
def seconds_per_hour : ℝ := 3600

-- Calculate speed in miles per hour based on the given conditions
noncomputable def calculate_speed_mph (d_ft : ℝ) (t_sec : ℝ) (ft_per_mile : ℝ) (sec_per_hr : ℝ) : ℝ :=
  (d_ft / t_sec) * (sec_per_hr / ft_per_mile)

-- State the theorem
theorem object_speed_mph : calculate_speed_mph distance_ft time_sec feet_per_mile seconds_per_hour ≈ 20.45 :=
  sorry

end object_speed_mph_l144_144951


namespace chocolate_mixture_l144_144326

theorem chocolate_mixture (x : ℝ) (h_initial : 110 / 220 = 0.5)
  (h_equation : (110 + x) / (220 + x) = 0.75) : x = 220 := by
  sorry

end chocolate_mixture_l144_144326


namespace num_chips_initially_count_l144_144427

-- Definitions based on conditions
def hexagonal_cells := 37
def neighbors (cell : ℕ) := {3, 4, 6}  -- Represent neighbors count in a cell

-- Chip placement representation
def chips (cell : ℕ) (initially : bool) := initially

-- Number representation in empty cells after chip removal
def cell_number (cell : ℕ) (neighbors_with_chips_count : ℕ) := neighbors_with_chips_count

-- The main theorem to prove the number of chips originally placed
theorem num_chips_initially_count :
  ∃ n, n = 8 :=
by
  -- Proof will go here
  sorry

end num_chips_initially_count_l144_144427


namespace tangent_line_to_circle_l144_144792

theorem tangent_line_to_circle (x y : ℝ) (h : x^2 + y^2 = 5) (h_point : (2, -1)) :
  (∀ x y, x^2 + y^2 = 5 → y = (-1 :ℝ) ∨ (∃ k, y = k * (x - 2) - 1)) -> 
  (y = 2 * (x - 2) - 1) -> 2 * x - y - 5 = 0 :=
by
  sorry

end tangent_line_to_circle_l144_144792


namespace factorization_l144_144235

theorem factorization (x : ℝ) : x^2 - 2 * x = x * (x - 2) := 
sorry

end factorization_l144_144235


namespace cos_120_l144_144110

open Real

theorem cos_120 : cos (120 * real.pi / 180) = - 1 / 2 :=
by sorry

end cos_120_l144_144110


namespace prob_shoots_3_times_correct_expected_value_X_correct_l144_144710

-- Define the conditions
def shots : ℕ := 3
def points_A : ℕ := 3
def points_B : ℕ := 2
def accuracy_A : ℚ := 0.25
def accuracy_B : ℚ := 0.8
def first_shot_A_then_B : list ℕ := [A, B, B]

-- Define the total score stopping condition
def stops_shooting (shot_1 shot_2 : ℕ) := shot_1 + shot_2 > 3

-- Define the probability to shoot 3 times according to given conditions
def probability_shoots_3_times : ℚ :=
  let prob_cond1 := accuracy_A * (1 - accuracy_B)
  let prob_cond2 := 1 - accuracy_A
  prob_cond1 + prob_cond2

-- Prove that the probability is 4/5
theorem prob_shoots_3_times_correct : probability_shoots_3_times = 4 / 5 := sorry

-- Define the expected value calculation according to given conditions
def expected_value_X : ℚ :=
  let P2 := 0.75 * 0.8 * (1 - 0.8) * 2
  let P3 := 0.25 * (1 - 0.8) ^ 2
  let P4 := 0.75 * 0.8 ^ 2
  let P5 := 0.25 * 0.8 * (1 - 0.8) + 0.25 * 0.8
  0 * 0.03 + 2 * P2 + 3 * P3 + 4 * P4 + 5 * P5

-- Prove that the expected value E(X) is 3.63
theorem expected_value_X_correct : expected_value_X = 3.63 := sorry

end prob_shoots_3_times_correct_expected_value_X_correct_l144_144710


namespace no_finite_open_interval_l144_144746

def S : Set ℝ := {x | ∃ m n : ℕ, 0 < m ∧ 0 < n ∧ x = sqrt m - sqrt n}

theorem no_finite_open_interval (a b : ℝ) (h : a < b) : 
  ¬ ∃ (s : Finset ℝ), (∀ (x ∈ s), x ∈ S) ∧ (∀ x ∈ S, (a < x ∧ x < b) → x ∈ s) :=
sorry

end no_finite_open_interval_l144_144746


namespace sum_of_distances_l144_144152

theorem sum_of_distances (d_1 d_2 : ℝ) (h1 : d_1 = 1 / 9 * d_2) (h2 : d_1 + d_2 = 6) : d_1 + d_2 + 6 = 20 :=
by
  sorry

end sum_of_distances_l144_144152


namespace iPhone_savings_l144_144399

theorem iPhone_savings
  (costX costY : ℕ)
  (discount_same_model discount_mixed : ℝ)
  (h1 : costX = 600)
  (h2 : costY = 800)
  (h3 : discount_same_model = 0.05)
  (h4 : discount_mixed = 0.03) :
  (costX + costX + costY) - ((costX * (1 - discount_same_model)) * 2 + costY * (1 - discount_mixed)) = 84 :=
by
  sorry

end iPhone_savings_l144_144399


namespace jackson_spiral_shells_l144_144368

-- Define the conditions
variable (hermit_crabs : ℕ) (souvenirs : ℕ) (spiral_shells_per_hermit_crab : ℕ) (starfish_per_spiral_shell : ℕ)

-- Assumptions based on the conditions
def conditions := 
  hermit_crabs = 45 ∧
  starfish_per_spiral_shell = 2 ∧
  souvenirs = 450

-- Define the total number of spiral shells and starfish
def total_spiral_shells := hermit_crabs * spiral_shells_per_hermit_crab
def total_starfish := total_spiral_shells * starfish_per_spiral_shell

-- Equation based on the total number of souvenirs
def total_souvenirs := hermit_crabs + total_spiral_shells + total_starfish

-- The theorem to be proved
theorem jackson_spiral_shells :
  ∀ (s : ℕ),
  (hermit_crabs = 45) →
  (starfish_per_spiral_shell = 2) →
  (souvenirs = 450) →
  (hermit_crabs + hermit_crabs * s + hermit_crabs * s * starfish_per_spiral_shell = souvenirs) →
  s = 3 := by
  intros s hc ssf tot eq
  sorry

end jackson_spiral_shells_l144_144368


namespace IMO1988Problem_l144_144630

theorem IMO1988Problem:
  ∀ n : ℕ, (∀ A B : Finset ℕ, A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅ → 
  ∃ a b c ∈ A, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b = c)
  → (96 ≤ n) :=
begin
  sorry
end

end IMO1988Problem_l144_144630


namespace disproves_proposition_b_l144_144522

-- Definition and condition of complementary angles
def angles_complementary (angle1 angle2: ℝ) : Prop := angle1 + angle2 = 180

-- Proposition to disprove
def disprove (angle1 angle2: ℝ) : Prop := ¬ ((angle1 < 90 ∧ angle2 > 90 ∧ angle2 < 180) ∨ (angle2 < 90 ∧ angle1 > 90 ∧ angle1 < 180))

-- Definition of angles in sets
def set_a := (120, 60)
def set_b := (95.1, 84.9)
def set_c := (30, 60)
def set_d := (90, 90)

-- Statement to prove
theorem disproves_proposition_b : 
  (angles_complementary 95.1 84.9) ∧ (disprove 95.1 84.9) :=
by
  sorry

end disproves_proposition_b_l144_144522


namespace four_digit_numbers_l144_144323

theorem four_digit_numbers (n : ℕ) :
    (∃ a b c d : ℕ, 
        n = a * 1000 + b * 100 + c * 10 + d 
        ∧ 4 ≤ a ∧ a ≤ 9 
        ∧ 1 ≤ b ∧ b ≤ 9 
        ∧ 1 ≤ c ∧ c ≤ 9 
        ∧ 0 ≤ d ∧ d ≤ 9 
        ∧ b * c > 8) → n ∈ {n | 4000 ≤ n ∧ n < 10000}
           → n ∈ {n | 4000 ≤ n ∧ n < 10000 ∧ b * c > 8} := sorry

end four_digit_numbers_l144_144323


namespace green_chips_count_l144_144472

-- Definitions of given conditions
def total_chips (total : ℕ) : Prop :=
  ∃ blue_chips white_percentage, blue_chips = 3 ∧ (blue_chips : ℚ) / total = 10 / 100 ∧ white_percentage = 50 / 100 ∧
  let green_percentage := 1 - (10 / 100 + white_percentage) in
  green_percentage * total = 12

-- Proposition to prove the number of green chips equals 12
theorem green_chips_count (total : ℕ) (h : total_chips total) : ∃ green_chips, green_chips = 12 := 
by 
  sorry

end green_chips_count_l144_144472


namespace point_reflection_across_x_axis_l144_144059

theorem point_reflection_across_x_axis (x y : ℝ) : (x, y) = (-3, -4) → (x, -y) = (-3, 4) :=
by
  intros h
  rcases h with ⟨hx, hy⟩
  simp [hx, hy]
  sorry

end point_reflection_across_x_axis_l144_144059


namespace Payton_score_l144_144401

-- Conditions turned into definitions
def num_students := 15
def first_14_avg := 80
def full_class_avg := 81

-- Desired fact to prove
theorem Payton_score :
  ∃ score : ℕ, score = 95 ∧
  (let total_14 := 14 * first_14_avg in
   let total_15 := num_students * full_class_avg in
   score = total_15 - total_14) :=
begin
  have H14 : total_14 = 1120, from rfl,
  have H15 : total_15 = 1215, from rfl,
  use 95,
  split,
  {
    refl,  -- score = 95
  },
  {
    dsimp [total_14, total_15],
    rw [H14, H15],
    refl,
  },
end

end Payton_score_l144_144401


namespace gcd_factorial_l144_144858

theorem gcd_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  sorry

end gcd_factorial_l144_144858


namespace digit_sum_225th_element_is_specific_number_l144_144760

-- Define the sequence of natural numbers with a digit sum of 2018
def digit_sum (n : ℕ) : ℕ := 
  n.digits.sum

def sequence_nat_with_digit_sum_2018 : LazyList ℕ :=
  LazyList.filter (λ n, digit_sum n = 2018) LazyList.nat

-- Define the 225th element in this sequence
noncomputable def nth_element (n : ℕ) := sequence_nat_with_digit_sum_2018.nth n

-- Define the specific number 3 followed by 223 nines and 8 at the end
def specific_number : ℕ := 
  3 * 10^224 + 10^0 - 2

-- Theorem statement claiming the 225th element in the sequence is the specific number
theorem digit_sum_225th_element_is_specific_number : 
  nth_element 225 = some specific_number :=
sorry

end digit_sum_225th_element_is_specific_number_l144_144760


namespace log_base4_of_1_div_64_l144_144220

theorem log_base4_of_1_div_64 : log 4 (1 / 64) = -3 := by
  sorry

end log_base4_of_1_div_64_l144_144220


namespace correct_answer_is_B_l144_144122

-- Definitions for each set of line segments
def setA := (2, 2, 4)
def setB := (8, 6, 3)
def setC := (2, 6, 3)
def setD := (11, 4, 6)

-- Triangle inequality theorem checking function
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statements to verify each set
lemma check_setA : ¬ is_triangle 2 2 4 := by sorry
lemma check_setB : is_triangle 8 6 3 := by sorry
lemma check_setC : ¬ is_triangle 2 6 3 := by sorry
lemma check_setD : ¬ is_triangle 11 4 6 := by sorry

-- Final theorem combining all checks to match the given problem
theorem correct_answer_is_B : 
  ¬ is_triangle 2 2 4 ∧ is_triangle 8 6 3 ∧ ¬ is_triangle 2 6 3 ∧ ¬ is_triangle 11 4 6 :=
by sorry

end correct_answer_is_B_l144_144122


namespace odd_function_expression_l144_144281

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 2 * x + 8 else if x < 0 then -x^2 + 2 * x - 8 else 0

theorem odd_function_expression :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, 0 < x → f x = x^2 + 2 * x + 8)
  → f = λ x, if x > 0 then x^2 + 2 * x + 8 else if x < 0 then -x^2 + 2 * x - 8 else 0 :=
by
  sorry

end odd_function_expression_l144_144281


namespace average_length_correct_l144_144914

-- Given lengths of the two pieces
def length1 : ℕ := 2
def length2 : ℕ := 6

-- Define the average length
def average_length (l1 l2 : ℕ) : ℕ := (l1 + l2) / 2

-- State the theorem to prove
theorem average_length_correct : average_length length1 length2 = 4 := 
by 
  sorry

end average_length_correct_l144_144914


namespace ice_making_cost_l144_144412

-- Pauly's ice making parameters
def pounds_needed : ℝ := 10
def oz_per_cube : ℝ := 2
def weight_per_cube_lbs : ℝ := 1/16
def time_per_10_cubes_hours : ℝ := 1
def cost_per_hour : ℝ := 1.5
def cost_per_oz : ℝ := 0.1

-- The proof statement
theorem ice_making_cost : 
  let num_cubes := pounds_needed / weight_per_cube_lbs in
  let time_required := num_cubes / 10 * time_per_10_cubes_hours in
  let cost_running_ice_maker := time_required * cost_per_hour in
  let total_water_needed := num_cubes * oz_per_cube in
  let cost_water := total_water_needed * cost_per_oz in
  cost_running_ice_maker + cost_water = 56 :=
by
  -- Proof goes here
  sorry

end ice_making_cost_l144_144412


namespace bee_distance_to_P7_l144_144149

noncomputable def P0 : complex := 0
noncomputable def P1 : complex := 1

noncomputable def ω : complex := complex.exp (real.pi * complex.I / 4)

noncomputable def P (j : ℕ) : complex :=
if j = 0 then P0
else if j = 1 then P1
else (∑ k in finset.range j, (k+1:ℕ) * ω^k : ℂ)

noncomputable def distance (n : ℕ) : ℝ :=
complex.abs (P n)

theorem bee_distance_to_P7 : distance 7 = real.sqrt (25 - 7 * real.sqrt 2 / 2) :=
by
  sorry

end bee_distance_to_P7_l144_144149


namespace ellipse_x_intercept_other_l144_144945

noncomputable def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 3), (4, 0))
noncomputable def sum_of_distances : ℝ := 7
noncomputable def first_intercept : (ℝ × ℝ) := (0, 0)

theorem ellipse_x_intercept_other 
  (foci : (ℝ × ℝ) × (ℝ × ℝ))
  (sum_of_distances : ℝ)
  (first_intercept : (ℝ × ℝ))
  (hx : foci = ((0, 3), (4, 0)))
  (d_sum : sum_of_distances = 7)
  (intercept : first_intercept = (0, 0)) :
  ∃ (x : ℝ), x > 0 ∧ ((x, 0) = (56 / 11, 0)) := 
sorry

end ellipse_x_intercept_other_l144_144945


namespace tangent_circle_angle_l144_144105

theorem tangent_circle_angle
  (O A B C : Point) -- Points O, A, B, and C
  (hO : is_center O) -- O is the center of the circle
  (hT1 : is_tangent A B O) -- AB is a tangent to the circle at B
  (hT2 : is_tangent A C O) -- AC is a tangent to the circle at C
  (hR : arc_ratio B C 1 4) -- The arc BC and arc CB' have a ratio of 1:4
  : angle A B C = 36 :=
sorry

end tangent_circle_angle_l144_144105


namespace expression_for_M_value_of_M_when_x_eq_negative_2_and_y_eq_1_l144_144692

noncomputable def A (x y : ℝ) := x^2 - 3 * x * y - y^2
noncomputable def B (x y : ℝ) := x^2 - 3 * x * y - 3 * y^2
noncomputable def M (x y : ℝ) := 2 * A x y - B x y

theorem expression_for_M (x y : ℝ) : M x y = x^2 - 3 * x * y + y^2 := by
  sorry

theorem value_of_M_when_x_eq_negative_2_and_y_eq_1 :
  M (-2) 1 = 11 := by
  sorry

end expression_for_M_value_of_M_when_x_eq_negative_2_and_y_eq_1_l144_144692


namespace bridge_toll_fees_for_annie_are_5_l144_144400

-- Conditions
def start_fee : ℝ := 2.50
def cost_per_mile : ℝ := 0.25
def mike_miles : ℕ := 36
def annie_miles : ℕ := 16
def total_cost_mike : ℝ := start_fee + cost_per_mile * mike_miles

-- Hypothesis from conditions
axiom both_charged_same : ∀ (bridge_fees : ℝ), total_cost_mike = start_fee + cost_per_mile * annie_miles + bridge_fees

-- Proof problem
theorem bridge_toll_fees_for_annie_are_5 : ∃ (bridge_fees : ℝ), bridge_fees = 5 :=
by
  existsi 5
  sorry

end bridge_toll_fees_for_annie_are_5_l144_144400


namespace sophia_collection_value_l144_144035

-- Define the conditions
def stamps_count : ℕ := 24
def partial_stamps_count : ℕ := 8
def partial_value : ℤ := 40
def stamp_value_per_each : ℤ := partial_value / partial_stamps_count
def total_value : ℤ := stamps_count * stamp_value_per_each

-- Statement of the conclusion that needs proving
theorem sophia_collection_value :
  total_value = 120 := by
  sorry

end sophia_collection_value_l144_144035


namespace intercepts_sum_eq_seven_l144_144799

theorem intercepts_sum_eq_seven :
    (∃ a b c, (∀ y, (3 * y^2 - 9 * y + 4 = a) → y = 0) ∧ 
              (∀ y, (3 * y^2 - 9 * y + 4 = 0) → (y = b ∨ y = c)) ∧ 
              (a + b + c = 7)) := 
sorry

end intercepts_sum_eq_seven_l144_144799


namespace jar_ratios_l144_144369

theorem jar_ratios (C_X C_Y : ℝ) 
  (h1 : 0 < C_X) 
  (h2 : 0 < C_Y)
  (h3 : (1/2) * C_X + (1/2) * C_Y = (3/4) * C_X) : 
  C_Y = (1/2) * C_X := 
sorry

end jar_ratios_l144_144369


namespace like_terms_correct_l144_144520

theorem like_terms_correct : 
  (¬(∀ x y z w : ℝ, (x * y^2 = z ∧ x^2 * y = w)) ∧ 
   ¬(∀ x y : ℝ, (x * y = -2 * y)) ∧ 
    (2^3 = 8 ∧ 3^2 = 9) ∧ 
   ¬(∀ x y z w : ℝ, (5 * x * y = z ∧ 6 * x * y^2 = w))) :=
by
  sorry

end like_terms_correct_l144_144520


namespace range_of_k_l144_144635

theorem range_of_k (x y k : ℝ) (h1 : x - y = k - 1) (h2 : 3 * x + 2 * y = 4 * k + 5) (hk : 2 * x + 3 * y > 7) : k > 1 / 3 := 
sorry

end range_of_k_l144_144635


namespace base9_addition_l144_144573

-- Define the numbers in base 9
def num1 : ℕ := 1 * 9^2 + 7 * 9^1 + 5 * 9^0
def num2 : ℕ := 7 * 9^2 + 1 * 9^1 + 4 * 9^0
def num3 : ℕ := 6 * 9^1 + 1 * 9^0
def result : ℕ := 1 * 9^3 + 0 * 9^2 + 6 * 9^1 + 1 * 9^0

-- State the theorem
theorem base9_addition : num1 + num2 + num3 = result := by
  sorry

end base9_addition_l144_144573


namespace chord_length_l144_144899

theorem chord_length (r d : ℝ) (h_r : r = 5) (h_d : d = 4) : 
  ∃ EF : ℝ, EF = 6 :=
by
  sorry

end chord_length_l144_144899


namespace green_chips_count_l144_144469

variable (total_chips : ℕ) (blue_chips : ℕ) (white_percentage : ℕ) (blue_percentage : ℕ)
variable (green_percentage : ℕ) (green_chips : ℕ)

def chips_condition1 : Prop := blue_chips = (blue_percentage * total_chips) / 100
def chips_condition2 : Prop := blue_percentage = 10
def chips_condition3 : Prop := white_percentage = 50
def green_percentage_calculation : Prop := green_percentage = 100 - (blue_percentage + white_percentage)
def green_chips_calculation : Prop := green_chips = (green_percentage * total_chips) / 100

theorem green_chips_count :
  (chips_condition1) →
  (chips_condition2) →
  (chips_condition3) →
  (green_percentage_calculation) →
  (green_chips_calculation) →
  green_chips = 12 :=
by
  intros
  sorry

end green_chips_count_l144_144469


namespace range_of_A_l144_144389

theorem range_of_A (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
    (h_mono : ∀ x y, 0 < x → x < y → f x < f y) 
    (h_half : f (1 / 2) = 0) :
    ∀ A, (A ∈ {A : ℝ | 0 < A ∧ A < π} → f (Real.cos A) < 0) → 
    (A ∈ Ioo (π / 3) (π / 2) ∨ A ∈ Ioo (2 * π / 3) π) :=
by 
    intros
    sorry

end range_of_A_l144_144389


namespace find_values_l144_144275

-- Helper definitions to state the conditions
def distinct_pairwise_sums (n : ℕ) (a b c : ℝ) : Prop :=
  let sums := [1 + a, 1 + b, 1 + c, a + b, a + c, b + c]
  list.pairwise (≠) sums ∧ list.sorted (≤) sums

noncomputable def arithmetic_sequence (sums : list ℝ) : Prop :=
  ∃ (d : ℝ), list.sorted (≤) sums ∧
  ∀ i j, i < j ∧ j < sums.length → sums.nth i + d = sums.nth j

theorem find_values (a b c : ℝ) :
  1 < a ∧ a < b ∧ b < c ∧ 
  distinct_pairwise_sums 4 a b c ∧ 
  list.sum [1 + a, 1 + b, 1 + c, a + b, a + c, b + c] = 201 ∧ 
  arithmetic_sequence [1 + a, 1 + b, 1 + c, a + b, a + c, b + c] →
  (a = 10 ∧ b = 19 ∧ c = 37) ∨ (a = 15 ∧ b = 22 ∧ c = 29) :=
by
  sorry

end find_values_l144_144275


namespace has_one_real_solution_l144_144075

noncomputable def eq_sin_add_const (c : ℝ) (x : ℝ) : Prop :=
  x = sin x + c

theorem has_one_real_solution :
  ∀ c : ℝ, 1992 ≤ c ∧ c ≤ 1994 → ∃! x : ℝ, eq_sin_add_const c x :=
by {
  sorry
}

end has_one_real_solution_l144_144075


namespace factorize_difference_of_squares_l144_144617

theorem factorize_difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) :=
sorry

end factorize_difference_of_squares_l144_144617


namespace angle_sum_around_point_l144_144117

theorem angle_sum_around_point (y : ℕ) (h1 : 210 + 3 * y = 360) : y = 50 := 
by 
  sorry

end angle_sum_around_point_l144_144117


namespace bob_can_determine_S_l144_144927

theorem bob_can_determine_S (m n : ℤ) (hm : 0 < m) (hn : m ≤ n) :
  ∀ S : set (ℤ × ℤ), 
    (S = {p : ℤ × ℤ | let (x, y) := p in m ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ n}) →
    (∃ line_counts : (ℤ × ℤ → Prop) → ℕ, ∀ l, line_counts l = (S ∩ {z | l z}).card) →
  True :=
by sorry

end bob_can_determine_S_l144_144927


namespace four_digit_numbers_count_l144_144313

theorem four_digit_numbers_count :
  ∃ n : ℕ, n = 4140 ∧
  (∀ d1 d2 d3 d4 : ℕ,
    (4 ≤ d1 ∧ d1 ≤ 9) ∧
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d2 * d3 > 8) →
    (∃ m : ℕ, m = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ m > 3999) →
    n = 4140) :=
sorry

end four_digit_numbers_count_l144_144313


namespace prime_exponent_sum_is_9_l144_144211

def roundness := (n : ℕ) → Finset ℕ

theorem prime_exponent_sum_is_9 :
  ∃ f : roundness, (f 1234560).sum = 9 :=
by
  sorry

end prime_exponent_sum_is_9_l144_144211


namespace reference_city_stores_2000_l144_144735

variable (S : ℕ)

-- Conditions from the problem
def reference_city_conditions :=
  let hospitals := 500
  let schools := 200
  let police_stations := 20
  hospitals = 500 ∧ schools = 200 ∧ police_stations = 20

def new_city_conditions (S : ℕ) :=
  let new_stores := S / 2
  let new_hospitals := 500 * 2
  let new_schools := 200 - 50
  let new_police_stations := 20 + 5
  (new_stores + new_hospitals + new_schools + new_police_stations = 2175)

-- Prove that the number of stores in the reference city is 2000
theorem reference_city_stores_2000 
  (h_ref : reference_city_conditions S) 
  (h_new : new_city_conditions S) :
  S = 2000 :=
sorry

end reference_city_stores_2000_l144_144735


namespace polynomial_root_transformation_l144_144604

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem polynomial_root_transformation :
  let P (a b c d e : ℝ) (x : ℂ) := (x^6 : ℂ) + (a : ℂ) * x^5 + (b : ℂ) * x^4 + (c : ℂ) * x^3 + (d : ℂ) * x^2 + (e : ℂ) * x + 4096
  (∀ r : ℂ, P 0 0 0 0 0 r = 0 → P 0 0 0 0 0 (ω * r) = 0) →
  ∃ a b c d e : ℝ, ∃ p : ℕ, p = 3 := sorry

end polynomial_root_transformation_l144_144604


namespace quadrilateral_AD_eq_DC_l144_144178

open EuclideanGeometry

variables {A B C D : Point}

theorem quadrilateral_AD_eq_DC (h₁ : dist A B = dist B C) 
  (h₂ : ∠CB D = 2 * ∠AD B) 
  (h₃ : 2 * ∠AB D = ∠CD B) : dist A D = dist D C := 
sorry

end quadrilateral_AD_eq_DC_l144_144178


namespace common_ratio_of_series_l144_144186

-- Define the terms and conditions for the infinite geometric series problem.
def first_term : ℝ := 500
def series_sum : ℝ := 4000

-- State the theorem that needs to be proven: the common ratio of the series is 7/8.
theorem common_ratio_of_series (a S r : ℝ) (h_a : a = 500) (h_S : S = 4000) (h_eq : S = a / (1 - r)) :
  r = 7 / 8 :=
by
  sorry

end common_ratio_of_series_l144_144186


namespace smallest_n_coprime_partition_l144_144004

open Nat

theorem smallest_n_coprime_partition : 
  ∃ n : ℕ, (∀ T : Finset ℕ, T.card = n → T ⊆ Finset.range 99 → 
  ∃ A B : Finset ℕ, A.card = 5 ∧ B.card = 5 ∧ A ∪ B ⊆ T ∧ 
  (∃ a ∈ A, ∀ x ∈ A \ {a}, coprime a x) ∧ 
  (∃ b ∈ B, ¬ ∀ y ∈ B \ {b}, coprime b y)) ∧ 
  (∀ m : ℕ, m < n → 
  ∃ T : Finset ℕ, T.card = m ∧ T ⊆ Finset.range 99 ∧ 
  ∀ A B : Finset ℕ, A ∪ B ⊆ T → ¬((∃ a ∈ A, ∀ x ∈ A \ {a}, coprime a x) ∧ 
  (∃ b ∈ B, ¬ ∀ y ∈ B \ {b}, coprime b y))) :=
sorry

end smallest_n_coprime_partition_l144_144004


namespace number_of_positive_numbers_l144_144649

theorem number_of_positive_numbers (a b c : ℚ) (h1 : a + b + c = 0) (h2 : a * b * c = 1) : 
  (if (0 < a).toBool + (0 < b).toBool + (0 < c).toBool = 1 then true else false) := 
  sorry

end number_of_positive_numbers_l144_144649


namespace square_pyramid_slant_height_lateral_area_total_area_l144_144270

theorem square_pyramid_slant_height_lateral_area_total_area :
  ∃ s h l : ℝ, 
    s = 4 ∧ 
    h = 35 ∧ 
    l = 3.49 ∧
    (l ^ 2 = h^2 + (s/2)^2) ∧ -- slant height related to height and half of the side
    (2 * s * l / 2) = 27.92 ∧ -- lateral area for all 4 triangular faces
    (s ^ 2 + 2 * s * l / 2) = 43.92 := -- total surface area including base and lateral area
exists_intro 4 (exists_intro 35 (exists_intro 3.49
(And.intro rfl (And.intro rfl (And.intro rfl (And.intro sorry (And.intro sorry sorry))))))) -- skip proof steps

end square_pyramid_slant_height_lateral_area_total_area_l144_144270


namespace overlapping_area_of_thirty_sixty_ninety_triangles_l144_144830

-- Definitions for 30-60-90 triangle and the overlapping region
def thirty_sixty_ninety_triangle (hypotenuse : ℝ) := 
  (hypotenuse > 0) ∧ 
  (exists (short_leg long_leg : ℝ), short_leg = hypotenuse / 2 ∧ long_leg = short_leg * (Real.sqrt 3))

-- Area of a parallelogram given base and height
def parallelogram_area (base height : ℝ) : ℝ :=
  base * height

theorem overlapping_area_of_thirty_sixty_ninety_triangles :
  ∀ (hypotenuse : ℝ), thirty_sixty_ninety_triangle hypotenuse →
  hypotenuse = 10 →
  (∃ (base height : ℝ), base = height ∧ base * height = parallelogram_area (5 * Real.sqrt 3) (5 * Real.sqrt 3)) →
  parallelogram_area (5 * Real.sqrt 3) (5 * Real.sqrt 3) = 75 :=
by
  sorry

end overlapping_area_of_thirty_sixty_ninety_triangles_l144_144830


namespace gcd_factorials_l144_144846

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  -- Proof steps would go here
  sorry

end gcd_factorials_l144_144846


namespace number_of_particular_propositions_is_two_l144_144576

def proposition1 : Prop := ∃ x : ℝ, ¬(x ≠ 0 → ∃ y : ℝ, y * x = 1)
def proposition2 : Prop := ∀ x : Type, x = {p : Prop // true} → x.is_polyhedron
def proposition3 : Prop := ∀ x : ℝ, ∀ a b : ℝ, (∃ y : ℝ, a * y + b = x) 
def proposition4 : Prop := ∃ t : Type, t.is_triangle ∧ t.is_acute

theorem number_of_particular_propositions_is_two :
  (∃ x : ℕ, x = 2) := by
  sorry

end number_of_particular_propositions_is_two_l144_144576


namespace blocks_differing_in_two_ways_l144_144900

noncomputable def num_blocks_differing_in_two_ways : ℕ :=
  let materials := ["plastic", "wood", "metal"]
  let sizes := ["small", "medium", "large"]
  let colors := ["blue", "green", "red", "yellow"]
  let shapes := ["circle", "hexagon", "square", "triangle"]
  let wood_large_blue_hexagon := ("wood", "large", "blue", "hexagon")

  let num_materials := 2
  let num_sizes := 2
  let num_colors := 3
  let num_shapes := 3

  let comb1 := num_materials * num_sizes
  let comb2 := num_materials * num_colors
  let comb3 := num_materials * num_shapes
  let comb4 := num_sizes * num_colors
  let comb5 := num_sizes * num_shapes
  let comb6 := num_colors * num_shapes

  comb1 + comb2 + comb3 + comb4 + comb5 + comb6

theorem blocks_differing_in_two_ways (num_blocks_differing_in_two_ways = 37) : Prop :=
  num_blocks_differing_in_two_ways = 37

end blocks_differing_in_two_ways_l144_144900


namespace problem1_problem2_l144_144640

-- Definitions and assumptions for the first proof problem
def z : ℂ := 1 + complex.i
def ω : ℂ := z^2 + 3 * conj z - 4

-- Statement for the first proof problem
theorem problem1 : complex.abs ω = real.sqrt 2 :=
by { sorry }

-- Definitions and assumptions for the second proof problem
def w : ℂ := 1 - complex.i
def a : ℂ := -1
def b : ℂ := 2

-- Statement for the second proof problem
theorem problem2 (h : (z^2 + a * z + b) / (z^2 - z + 1) = w) : 
  a = -1 ∧ b = 2 :=
by { sorry }

end problem1_problem2_l144_144640


namespace infinite_geometric_series_common_ratio_l144_144184

theorem infinite_geometric_series_common_ratio :
  ∀ (a r S : ℝ), a = 500 ∧ S = 4000 ∧ (a / (1 - r) = S) → r = 7 / 8 :=
by
  intros a r S h
  cases h with h_a h_S_eq
  cases h_S_eq with h_S h_sum_eq
  -- Now we have: a = 500, S = 4000, and a / (1 - r) = S
  sorry

end infinite_geometric_series_common_ratio_l144_144184


namespace train_avg_speed_l144_144526

variable (x : ℝ)

def avg_speed_of_train (x : ℝ) : ℝ := 3

theorem train_avg_speed (h : x > 0) : avg_speed_of_train x / (x / 7.5) = 22.5 :=
  sorry

end train_avg_speed_l144_144526


namespace ellipse_x_intercept_other_l144_144942

noncomputable def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 3), (4, 0))
noncomputable def sum_of_distances : ℝ := 7
noncomputable def first_intercept : (ℝ × ℝ) := (0, 0)

theorem ellipse_x_intercept_other 
  (foci : (ℝ × ℝ) × (ℝ × ℝ))
  (sum_of_distances : ℝ)
  (first_intercept : (ℝ × ℝ))
  (hx : foci = ((0, 3), (4, 0)))
  (d_sum : sum_of_distances = 7)
  (intercept : first_intercept = (0, 0)) :
  ∃ (x : ℝ), x > 0 ∧ ((x, 0) = (56 / 11, 0)) := 
sorry

end ellipse_x_intercept_other_l144_144942


namespace period_of_sin_div_x_l144_144494

theorem period_of_sin_div_x (x : ℝ) : 
  (∀ x, has_period (λ x, sin (x / 3)) x) → 
  has_period (λ x, sin (x / 3)) (6 * real.pi) :=
by
  sorry

end period_of_sin_div_x_l144_144494


namespace arithmetic_mean_location_l144_144103

theorem arithmetic_mean_location (a b : ℝ) : 
    abs ((a + b) / 2 - a) = abs (b - (a + b) / 2) := 
by 
    sorry

end arithmetic_mean_location_l144_144103


namespace train_length_calculation_l144_144572

noncomputable def train_speed_kmph : ℝ := 126
noncomputable def time_to_cross_seconds : ℝ := 2.856914303998537
noncomputable def speed_conversion_factor : ℝ := 5 / 18

theorem train_length_calculation : 
  let speed_mps := train_speed_kmph * speed_conversion_factor in
  let train_length := speed_mps * time_to_cross_seconds in
  train_length = 99.992 :=
by
  sorry

end train_length_calculation_l144_144572


namespace distance_D_D_l144_144009

def point := (ℝ × ℝ)

def D : point := (2, -4)
def D' : point := (-2, -4)

def distance (p1 p2 : point) : ℝ := 
  ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2).sqrt

theorem distance_D_D' : distance D D' = 4 :=
by sorry

end distance_D_D_l144_144009


namespace gcd_factorial_l144_144862

theorem gcd_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  sorry

end gcd_factorial_l144_144862


namespace log_base4_of_1_div_64_l144_144217

theorem log_base4_of_1_div_64 : log 4 (1 / 64) = -3 := by
  sorry

end log_base4_of_1_div_64_l144_144217


namespace area_of_triangle_ABC_l144_144344

theorem area_of_triangle_ABC
  {A B C : Type*} 
  (AC BC : ℝ)
  (B : ℝ)
  (h1 : AC = Real.sqrt (13))
  (h2 : BC = 1)
  (h3 : B = Real.sqrt 3 / 2): 
  ∃ area : ℝ, area = Real.sqrt 3 := 
sorry

end area_of_triangle_ABC_l144_144344


namespace correct_operation_l144_144519

variable (a : ℝ)

theorem correct_operation :
  (2 * a^2 * a = 2 * a^3) ∧
  ((a + 1)^2 ≠ a^2 + 1) ∧
  ((a^2 / (2 * a)) ≠ 2 * a) ∧
  ((2 * a^2)^3 ≠ 6 * a^6) :=
by
  { sorry }

end correct_operation_l144_144519


namespace base4_calculation_correct_l144_144588

noncomputable def base4_to_base10 (n : ℕ) : ℕ := 
  let digits := List.reverse ((n.toDigits 4).toList)
  digits.enumFrom 0 |>.foldl (λ acc (e : ℕ × ℕ) => acc + e.snd * 4^e.fst) 0

def multiply_and_divide_base4 (n m d : ℕ) : ℕ :=
  let product := base4_to_base10 n * base4_to_base10 m
  let quotient := product / base4_to_base10 d
  quotient.toDigits 4 |>.foldl (λ acc x => acc * 10 + x) 0

theorem base4_calculation_correct : 
  multiply_and_divide_base4 132 21 3 = 1122 := 
by
  sorry

end base4_calculation_correct_l144_144588


namespace dot_product_expression_l144_144282

variables (a b : ℝ^3)
variables (abs_a : ℝ) (abs_b : ℝ)
variables (theta : ℝ)
hypothesis h_abs_a : abs_a = 3
hypothesis h_abs_b : abs_b = 4
hypothesis h_theta : theta = 3 * Real.pi / 4

theorem dot_product_expression :
  let a_dot_b := abs_a * abs_b * Real.cos (theta)
  a_dot_b = -6 * Real.sqrt 2 →
  (2 * a - b) • (a + b) = 2 - 6 * Real.sqrt 2 :=
by
  sorry

end dot_product_expression_l144_144282


namespace tangent_product_l144_144338

theorem tangent_product (A B C : ℝ) (hA : A = 15) (hB : B = 30) (hC : A + B + C = 90) :
    (1 + Real.tan (A * Real.pi / 180 % Real.pi))(1 + Real.tan (B * Real.pi / 180 % Real.pi))(1 + Real.tan (C * Real.pi / 180 % Real.pi)) = (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end tangent_product_l144_144338


namespace saucepan_capacity_l144_144060

-- Define the conditions
variable (x : ℝ)
variable (h : 0.28 * x = 35)

-- State the theorem
theorem saucepan_capacity : x = 125 :=
by
  sorry

end saucepan_capacity_l144_144060


namespace find_a8_l144_144303

theorem find_a8 :
  ∀ (a : ℕ → ℝ) (d₁ d₂ : ℝ),
  (∀ n : ℕ, (n % 2 = 1 → a(n+2)= a(n) + 2 * d₁) ∧ (n % 2 = 0 → a(n+2)= a(n) + 2 * d₂))
  ∧ (∀ n : ℕ, a(n) < a(n + 1))
  ∧ a 1 = 1
  ∧ a 2 = 2
  ∧ (∑ i in finset.range 10, a (i + 1)) = 75
  → a 8 = 20 := 
sorry

end find_a8_l144_144303


namespace min_value_difference_l144_144639

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - 2 * x + 1

-- Define conditions on a
def condition_a (a : ℝ) : Prop := 1/3 ≤ a ∧ a ≤ 1

-- Define the minimum value of f(x) in its domain based on given conditions
def min_f (a : ℝ) : ℝ := 1 - (1 / a)

-- Define the function for M(a)
def M (a : ℝ) : ℝ :=
if 1/2 ≤ a ∧ a ≤ 1 then
  f a 3
else
  f a 1

-- Define the function for N(a)
def N (a : ℝ) : ℝ := f a (1 / a)

-- Define the difference between M(a) and N(a)
def M_minus_N (a : ℝ) : ℝ :=
if 1/2 ≤ a ∧ a ≤ 1 then
  9 * a + (1 / a) - 6
else
  a + (1 / a) - 2

-- Prove that the minimum value of M(a) - N(a) is 1/2
theorem min_value_difference : ∀ (a : ℝ), condition_a a → M_minus_N a = 1/2 :=
begin
  -- proof placeholder
  sorry
end

end min_value_difference_l144_144639


namespace harkamal_total_payment_l144_144529

def grapes_quantity : ℕ := 10
def grapes_rate : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_rate : ℕ := 55

def cost_of_grapes : ℕ := grapes_quantity * grapes_rate
def cost_of_mangoes : ℕ := mangoes_quantity * mangoes_rate

def total_amount_paid : ℕ := cost_of_grapes + cost_of_mangoes

theorem harkamal_total_payment : total_amount_paid = 1195 := by
  sorry

end harkamal_total_payment_l144_144529


namespace max_trains_final_count_l144_144756

-- Define the conditions
def trains_per_birthdays : Nat := 1
def trains_per_christmas : Nat := 2
def trains_per_easter : Nat := 3
def years : Nat := 7

-- Function to calculate total trains after 7 years
def total_trains_after_years (trains_per_years : Nat) (num_years : Nat) : Nat :=
  trains_per_years * num_years

-- Calculate inputs
def trains_per_year : Nat := trains_per_birthdays + trains_per_christmas + trains_per_easter
def total_initial_trains : Nat := total_trains_after_years trains_per_year years

-- Bonus and final steps
def bonus_trains_from_cousins (initial_trains : Nat) : Nat := initial_trains / 2
def final_total_trains (initial_trains : Nat) (bonus_trains : Nat) : Nat :=
  let after_bonus := initial_trains + bonus_trains
  let additional_from_parents := after_bonus * 3
  after_bonus + additional_from_parents

-- Main theorem
theorem max_trains_final_count : final_total_trains total_initial_trains (bonus_trains_from_cousins total_initial_trains) = 252 := by
  sorry

end max_trains_final_count_l144_144756


namespace three_digit_integer_divisible_by_5_l144_144092

theorem three_digit_integer_divisible_by_5 (M : ℕ) (h1 : 100 ≤ M ∧ M < 1000) (h2 : M % 10 = 5) : M % 5 = 0 := 
sorry

end three_digit_integer_divisible_by_5_l144_144092


namespace common_terms_count_l144_144305

def first_sequence (n : ℕ) : ℕ := 2 * n - 1

def second_sequence (n : ℕ) : ℕ := 5 * n - 4

theorem common_terms_count : 
  (set_of (λ x, ∃ m n : ℕ, first_sequence m = x ∧ second_sequence n = x ∧ x ≤ 1991)).card = 200 := 
sorry

end common_terms_count_l144_144305


namespace sum_of_squares_of_cosines_l144_144248

theorem sum_of_squares_of_cosines :
  (∑ k in Finset.range 89, (Real.cos ((k + 1 : ℕ) * (Real.pi / 180)))^2) = 44.5 :=
by
  sorry

end sum_of_squares_of_cosines_l144_144248


namespace regular_pentagon_of_congruent_triangles_l144_144252

-- Definitions of geometric entities and congruency
structure Triangle (A B C : Type) :=
  (angles : ℝ × ℝ × ℝ)
  (sides : ℝ × ℝ × ℝ)
  (is_triangle : sides.fst + sides.snd > sides.snd.snd ∧ sides.fst + sides.snd.snd > sides.snd ∧ sides.snd + sides.snd.snd > sides.fst)

-- Defining congruence of triangles
def congruent (Δ₁ Δ₂ : Triangle) : Prop :=
  Δ₁.angles = Δ₂.angles ∧ Δ₁.sides = Δ₂.sides

-- Pentagon and regularity
structure Pentagon (P Q R S T : Type) :=
  (angles : ℝ × ℝ × ℝ × ℝ × ℝ)
  (sides : ℝ × ℝ × ℝ × ℝ × ℝ)
  (is_pentagon : sides.fst + sides.snd + sides.snd.snd > 2 * sides.snd.mpr.to_nnd)

def regular_pentagon (P : Pentagon) : Prop :=
  P.angles.fst = P.angles.snd ∧ P.angles.snd = P.angles.snd.snd ∧ P.angles.snd.snd = P.angles.snd.mpr ∧ P.sides.fst = P.sides.snd ∧ P.sides.snd = P.sides.snd.snd

-- Main theorem: Prove that the pentagon formed inside the star is regular
theorem regular_pentagon_of_congruent_triangles
  (A B C D E F : Type)
  (Δ₁ : Triangle A B C)
  (Δ₂ : Triangle C D E)
  (Δ₃ : Triangle E F A)
  (Δ₄ : Triangle F A B)
  (Δ₅ : Triangle B C D)
  (cong₁ : congruent Δ₁ Δ₂)
  (cong₂ : congruent Δ₂ Δ₃)
  (cong₃ : congruent Δ₃ Δ₄)
  (cong₄ : congruent Δ₄ Δ₅) :
  ∃ P : Pentagon, regular_pentagon P :=
by
  apply exists.intro, { sorry }

end regular_pentagon_of_congruent_triangles_l144_144252


namespace tree_height_at_end_of_2_years_l144_144921

-- Conditions:
-- 1. The tree tripled its height every year.
-- 2. The tree reached a height of 243 feet at the end of 5 years.
theorem tree_height_at_end_of_2_years (h5 : ℕ) (H5 : h5 = 243) : 
  ∃ h2, h2 = 9 := 
by sorry

end tree_height_at_end_of_2_years_l144_144921


namespace campers_last_week_l144_144547

theorem campers_last_week :
  ∀ (total_campers : ℕ) (campers_two_weeks_ago : ℕ) (campers_diff : ℕ),
    (total_campers = 150) →
    (campers_two_weeks_ago = 40) →
    (campers_two_weeks_ago - campers_diff = campers_diff + 10) →
    ∃ (campers_last_week : ℕ), campers_last_week = 80 :=
by
  intros total_campers campers_two_weeks_ago campers_diff
  assume h1 h2 h3
  have h4 : campers_diff = 30 := by linarith
  have h5 : campers_last_week = total_campers - (campers_diff + campers_two_weeks_ago) := by linarith
  use (80 : ℕ)
  exact h5
  sorry

end campers_last_week_l144_144547


namespace four_digit_numbers_l144_144321

theorem four_digit_numbers (n : ℕ) :
    (∃ a b c d : ℕ, 
        n = a * 1000 + b * 100 + c * 10 + d 
        ∧ 4 ≤ a ∧ a ≤ 9 
        ∧ 1 ≤ b ∧ b ≤ 9 
        ∧ 1 ≤ c ∧ c ≤ 9 
        ∧ 0 ≤ d ∧ d ≤ 9 
        ∧ b * c > 8) → n ∈ {n | 4000 ≤ n ∧ n < 10000}
           → n ∈ {n | 4000 ≤ n ∧ n < 10000 ∧ b * c > 8} := sorry

end four_digit_numbers_l144_144321


namespace directrix_equation_of_parabola_l144_144685

-- Define the parameters of the problem
variables (p : ℝ) (x y : ℝ) 
variable h_p_pos : p > 0
variable h_point_A : (1, 1 / 2) ∈ {q : ℝ × ℝ | q.2^2 = 2 * p * q.1}

-- State the theorem
theorem directrix_equation_of_parabola (h_parabola : y^2 = 2 * p * x)
  (hA_x : 1) (hA_y : 1 / 2)
  : x = -1 / 16 :=
sorry

end directrix_equation_of_parabola_l144_144685


namespace find_number_l144_144249

noncomputable def solve_N (x : ℝ) (N : ℝ) : Prop :=
  ((N / x) / (3.6 * 0.2) = 2)

theorem find_number (x : ℝ) (N : ℝ) (h1 : x = 12) (h2 : solve_N x N) : N = 17.28 :=
  by
  sorry

end find_number_l144_144249


namespace negation_of_proposition_l144_144073

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x < 0 → x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x < 0 ∧ x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_of_proposition_l144_144073


namespace sum_of_numbers_gt_0_point_4_l144_144868

def numbers : List ℝ := [0.8, 1/2, 0.3, 1/3]

def condition (x : ℝ) : Prop := x > 0.4

def filtered_numbers : List ℝ := numbers.filter condition

def result : ℝ := 0.8 + 1/2

theorem sum_of_numbers_gt_0_point_4 :
  (filtered_numbers.sum = 1.3) :=
by sorry

end sum_of_numbers_gt_0_point_4_l144_144868


namespace solution_l144_144444

section
  variable {g : ℝ → ℝ}

  def equation_1 (x : ℝ) (hx : x ≠ 0) : Prop :=
    g(x) - 3 * g(1/x) = 4 * x^3

  def equation_2 (x : ℝ) (hx : x ≠ 0) : Prop :=
    g(1/x) - 3 * g(x) = 4 * (1/x)^3

  theorem solution (x : ℝ) (hx : x ≠ 0) (h : ∀ y : ℝ, y ≠ 0 → g(y) - 3 * g(1/y) = 4 * y^3) :
    g(x) = g(-x) :=
  sorry
end

end solution_l144_144444


namespace probability_calculation_l144_144562

noncomputable def probability_of_intersection : ℝ :=
  let center := (0 : ℝ, 0 : ℝ)
  let radius := (2 : ℝ)
  let line (k : ℝ) := fun x => k * (x - 4)
  let circle := fun (x y : ℝ) => x^2 + y^2 = radius^2
  let distance_to_line (k : ℝ) := |4 * k| / real.sqrt (1 + k^2)
  let k_in_range := \[k : ℝ | k ≥ -1 ∧ k ≤ 2\]
  let intersection_range := real.Ioo (-real.sqrt(3) / 3) (real.sqrt(3) / 3)
  let probability := (real.volume intersection_range) / (real.volume k_in_range)
  probability

theorem probability_calculation :
  probability_of_intersection = (2 * real.sqrt 3 / 9) := sorry

end probability_calculation_l144_144562


namespace angle_BDC_measure_l144_144140

open EuclideanGeometry  -- We assume the necessary geometric definitions are within some geometry library

-- Definitions for right triangle and angle bisector
def right_triangle (A B C : Point) := ∠ C = 90° ∧ (triangle A B C)

def angle_bisector (B D C : Point) (α : Angle) :=
  ∃ β γ, β = α / 2 ∧ γ = α / 2 ∧ β + γ = α ∧ betweenness B D C

theorem angle_BDC_measure {A B C D : Point} :
  right_triangle A B C ∧ ∠ A = 30° ∧ angle_bisector B D C (∠ B) ∧ D ∈ (segment A C)
  → ∠ BDC = 60° :=
by
  sorry

end angle_BDC_measure_l144_144140


namespace median_of_trapezoid_is_12_l144_144446

-- Define conditions
variables (A B C D O : Point)
variables (h : ℝ) (α : ℝ) -- height of the trapezoid and angle AOD
variables [IsoscelesTrapezoid A B C D]

-- Given conditions
hypothesis height_condition : h = 4 * Real.sqrt 3
hypothesis angle_condition : α = 120

-- Define the function to calculate the median
def calculate_median (A B C D : Point) (h : ℝ) (α : ℝ) : ℝ := 
  let cotangent := Real.cot (Real.pi / 6) -- cotangent of 30 degrees
  h * cotangent

-- Statement to prove
theorem median_of_trapezoid_is_12 (A B C D O : Point)
  [IsoscelesTrapezoid A B C D]
  (height_condition : 4 * Real.sqrt 3 = 4 * Real.sqrt 3)
  (angle_condition : 120 = 120) :
  calculate_median A B C D (4 * Real.sqrt 3) 120 = 12 := by
  sorry

end median_of_trapezoid_is_12_l144_144446


namespace sufficient_but_not_necessary_condition_l144_144648

variable (a : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : a = 1) (h2 : |a| = 1) : 
  (a = 1 → |a| = 1) ∧ ¬(|a| = 1 → a = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l144_144648


namespace termite_ridden_fraction_l144_144758

theorem termite_ridden_fraction:
  ∀ T: ℝ, (3 / 4) * T = 1 / 4 → T = 1 / 3 :=
by
  intro T
  intro h
  sorry

end termite_ridden_fraction_l144_144758


namespace length_of_ae_l144_144131

-- Definitions for lengths of segments
variable {ab bc cd de ac ae : ℝ}

-- Given conditions as assumptions
axiom h1 : bc = 3 * cd
axiom h2 : de = 8
axiom h3 : ab = 5
axiom h4 : ac = 11

-- The main theorem to prove
theorem length_of_ae : ae = ab + bc + cd + de → bc = ac - ab → bc = 6 → cd = bc / 3 → ae = 21 :=
by sorry

end length_of_ae_l144_144131


namespace value_of_t_l144_144407

theorem value_of_t (t : ℚ) (my_hours my_rate mike_hours mike_rate my_earnings mike_earnings : ℚ)
  (h1 : my_hours = t + 2)
  (h2 : my_rate = 4t - 4)
  (h3 : mike_hours = 4t - 2)
  (h4 : mike_rate = t + 3)
  (h5 : my_earnings = (t + 2) * (4t - 4))
  (h6 : mike_earnings = (4t - 2) * (t + 3))
  (h7 : my_earnings = mike_earnings + 3) :
  t = -14/9 :=
sorry

end value_of_t_l144_144407


namespace cuboid_edge_length_l144_144441

theorem cuboid_edge_length (x : ℝ) (h1 : (2 * (x * 5 + x * 6 + 5 * 6)) = 148) : x = 4 :=
by 
  sorry

end cuboid_edge_length_l144_144441


namespace totalPeaches_l144_144463

-- Definitions based on the given conditions
def redPeaches : Nat := 13
def greenPeaches : Nat := 3

-- Problem statement
theorem totalPeaches : redPeaches + greenPeaches = 16 := by
  sorry

end totalPeaches_l144_144463


namespace geometric_seq_neither_necess_nor_suff_l144_144045

theorem geometric_seq_neither_necess_nor_suff (a_1 q : ℝ) (h₁ : a_1 ≠ 0) (h₂ : q ≠ 0) :
  ¬ (∀ n : ℕ, (a_1 * q > 0 → a_1 * q ^ n < a_1 * q ^ (n + 1)) ∧ (∀ n : ℕ, (a_1 * q ^ n < a_1 * q ^ (n + 1)) → a_1 * q > 0)) :=
by
  sorry

end geometric_seq_neither_necess_nor_suff_l144_144045


namespace unique_plane_through_P_with_conditions_l144_144467

noncomputable theory

open_locale real

-- Definitions and conditions
def point_in_first_bisector_plane (P : ℝ × ℝ × ℝ) : Prop :=
  P.1 = P.2 ∧ P.2 = P.3

def plane_satisfying_conditions (s : set (ℝ × ℝ × ℝ)) (P : ℝ × ℝ × ℝ) (alpha beta : ℝ) : Prop :=
  (P ∈ s) ∧ 
  -- assume there's a way to express the first angle of projection and trace angles mathematically
  (first_angle_of_projection s = alpha) ∧ 
  (angle_between_first_and_second_traces s = beta)

-- The main theorem statement
theorem unique_plane_through_P_with_conditions (P : ℝ × ℝ × ℝ) (alpha beta : ℝ) 
  (h1 : point_in_first_bisector_plane P) :
  ∃! s : set (ℝ × ℝ × ℝ), plane_satisfying_conditions s P alpha beta :=
sorry

end unique_plane_through_P_with_conditions_l144_144467


namespace polynomial_root_reciprocal_l144_144739

theorem polynomial_root_reciprocal (a b : ℝ)
    (h : ∀ (λ : ℝ), (λ ∈ (roots (λ x : ℝ, x^4 - 4*x^3 + 2*x^2 + a*x + b))) →
                     (λ ≠ 0) → (1/λ ∈ (roots (λ x : ℝ, x^4 - 4*x^3 + 2*x^2 + a*x + b)))) :
    a + b = -3 :=
sorry

end polynomial_root_reciprocal_l144_144739


namespace polygon_sides_18_degree_exterior_angle_l144_144168

theorem polygon_sides_18_degree_exterior_angle (n : ℕ) (h1 : ∑ (i : ℕ) in finset.range n, 18 = 360) :
  n = 20 :=
sorry

end polygon_sides_18_degree_exterior_angle_l144_144168


namespace amin_probability_four_attempts_before_three_hits_amin_probability_not_qualified_stops_after_two_consecutive_misses_l144_144881

/-- Prove that the probability Amin makes 4 attempts before hitting 3 times (given the probability of each hit is 1/2) is 3/16. -/
theorem amin_probability_four_attempts_before_three_hits (p_hit : ℚ := 1 / 2) : 
  (∃ (P : ℚ), P = 3/16) :=
sorry

/-- Prove that the probability Amin stops shooting after missing two consecutive shots and not qualifying as level B or A player is 25/32, given the probability of each hit is 1/2. -/
theorem amin_probability_not_qualified_stops_after_two_consecutive_misses (p_hit : ℚ := 1 / 2) : 
  (∃ (P : ℚ), P = 25/32) :=
sorry

end amin_probability_four_attempts_before_three_hits_amin_probability_not_qualified_stops_after_two_consecutive_misses_l144_144881


namespace equilateral_triangle_tangent_area_l144_144353

theorem equilateral_triangle_tangent_area (a b : ℝ) (h : 0 < b ∧ b < a / 2) :
    let p := 3 * a / 2,
        r := (Math.sqrt 3) / 6 * a,
        p' := a / 2,
        r' := (a / 2 - b) / (Math.sqrt 3)
    in (p * r) / 3 = (a * Math.sqrt 3 * (a - 2 * b)) / 12 :=
by sorry

end equilateral_triangle_tangent_area_l144_144353


namespace cone_volume_proof_l144_144458

noncomputable def cone_volume (l h : ℕ) : ℝ :=
  let r := Real.sqrt (l^2 - h^2)
  1 / 3 * Real.pi * r^2 * h

theorem cone_volume_proof :
  cone_volume 13 12 = 100 * Real.pi :=
by
  sorry

end cone_volume_proof_l144_144458


namespace three_digit_integer_divisible_by_5_l144_144091

theorem three_digit_integer_divisible_by_5 (M : ℕ) (h1 : 100 ≤ M ∧ M < 1000) (h2 : M % 10 = 5) : M % 5 = 0 := 
sorry

end three_digit_integer_divisible_by_5_l144_144091


namespace hypotenuse_of_right_triangle_l144_144864

theorem hypotenuse_of_right_triangle (a b : ℕ) (h1 : a = 12) (h2 : b = 16) : 
  ∃ c, c = Real.sqrt ((a:ℝ)^2 + (b:ℝ)^2) ∧ c = 20 := 
by
  have h3 : a^2 + b^2 = 12^2 + 16^2, from by rw [h1, h2]
  have h4 : (12^2 + 16^2 : ℝ) = 400, from sorry
  have h5 : Real.sqrt (12^2 + 16^2) = 20, from sorry
  use 20
  split
  · exact h5
  · rfl

end hypotenuse_of_right_triangle_l144_144864


namespace temperature_reaches_100_at_5_hours_past_noon_l144_144343

theorem temperature_reaches_100_at_5_hours_past_noon :
  ∃ t : ℝ, (-2 * t^2 + 16 * t + 40 = 100) ∧ ∀ t' : ℝ, (-2 * t'^2 + 16 * t' + 40 = 100) → 5 ≤ t' :=
by
  -- We skip the proof and assume the theorem is true.
  sorry

end temperature_reaches_100_at_5_hours_past_noon_l144_144343


namespace trains_clear_time_l144_144530

-- Define the given lengths of the trains
def length_train1 : ℝ := 180
def length_train2 : ℝ := 280

-- Define the speeds of the trains in km/h and convert to m/s
def speed_train1_kmph : ℝ := 42
def speed_train2_kmph : ℝ := 30
def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)  -- Conversion factor from km/h to m/s

def speed_train1_mps : ℝ := kmph_to_mps speed_train1_kmph
def speed_train2_mps : ℝ := kmph_to_mps speed_train2_kmph

-- Calculate the relative speed when moving towards each other
def relative_speed : ℝ := speed_train1_mps + speed_train2_mps

-- Calculate the total distance to be covered (sum of the lengths of the trains)
def total_distance : ℝ := length_train1 + length_train2

-- Formulate the proof problem
theorem trains_clear_time : total_distance / relative_speed = 23 := by
  sorry

end trains_clear_time_l144_144530


namespace find_the_median_l144_144644

-- Define the function that finds the median
def find_median (m: Array Int) (N: Nat) : Int :=
  let half := N / 2
  let candidate := m.filter (fun x => m.count (fun y => y < x) <= half).get!
  candidate

-- Add the conditions as necessary lemmas
lemma median_condition (N : ℕ) (m : Array ℤ) :
  1 ≤ N ∧ N % 2 = 1 ∧ N ≤ 10001 ∧
  (∀ i < N, -10000 ≤ m[i] ∧ m[i] ≤ 10000) →
  ∃ r, r = find_median m N :=
by
  intros h
  sorry

-- The main theorem stating the existence of the median
theorem find_the_median (N : ℕ) (m : Array ℤ) (h : 1 ≤ N ∧ N % 2 = 1 ∧ N ≤ 10001 ∧
  (∀ i < N, -10000 ≤ m[i] ∧ m[i] ≤ 10000)) :
  ∃ r, r = find_median m N :=
median_condition N m h

end find_the_median_l144_144644


namespace pencil_distribution_l144_144824

/-- There are 10 ways to distribute a total of 9 identical pencils to three friends 
    where each friend gets at least two pencils. -/
theorem pencil_distribution (a b c : ℕ) (h₁ : a + b + c = 9) (h₂ : 2 ≤ a) (h₃ : 2 ≤ b) (h₄ : 2 ≤ c) : 
  ∃ n : ℕ, n = 10 :=
by
  use 10
  sorry

end pencil_distribution_l144_144824


namespace simplify_expression_l144_144232

theorem simplify_expression : 
  (2 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7) = 
  Real.sqrt 6 + 2 * Real.sqrt 2 - Real.sqrt 14 :=
sorry

end simplify_expression_l144_144232


namespace smallest_three_digit_geometric_sequence_l144_144503

theorem smallest_three_digit_geometric_sequence : ∃ n : ℕ, n = 124 ∧ (∀ (a r : ℕ), a = 1 ∧ r * a < 10 ∧ r^2 * a < 100 → digits n = [1, r, r^2] ∧ (digits n).nodup) := by
  sorry

end smallest_three_digit_geometric_sequence_l144_144503


namespace total_cost_of_items_l144_144077

-- Definitions based on conditions in a)
def price_of_caramel : ℕ := 3
def price_of_candy_bar : ℕ := 2 * price_of_caramel
def price_of_cotton_candy : ℕ := (4 * price_of_candy_bar) / 2
def cost_of_6_candy_bars : ℕ := 6 * price_of_candy_bar
def cost_of_3_caramels : ℕ := 3 * price_of_caramel

-- Problem statement to be proved
theorem total_cost_of_items : cost_of_6_candy_bars + cost_of_3_caramels + price_of_cotton_candy = 57 :=
by
  sorry

end total_cost_of_items_l144_144077


namespace find_angle_A_find_side_a_l144_144647

variable {A B C a b c : Real}
variable {area : Real}
variable (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A)
variable (h2 : b = 2)
variable (h3 : area = Real.sqrt 3)
variable (h4 : area = 1 / 2 * b * c * Real.sin A)

theorem find_angle_A (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A) : A = Real.pi / 3 :=
  sorry

theorem find_side_a (h4 : area = 1 / 2 * b * c * Real.sin A) (h2 : b = 2) (h3 : area = Real.sqrt 3) : a = 2 :=
  sorry

end find_angle_A_find_side_a_l144_144647


namespace moving_even_adj_red_l144_144346

-- Definition of the chessboard
def Chessboard := Fin 7 × Fin 12

-- Definition of red squares
def RedSquares := Finset Chessboard

-- Condition: there are exactly 25 red squares
axiom red_square_cardinality (R : RedSquares) : R.card = 25

-- Condition: adjacency definition
def adjacent (a b : Chessboard) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 = b.2 - 1)) ∨ 
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 = b.1 - 1))

-- Condition: each red square should have an even number of red neighbors
def even_adj_red (R : RedSquares) : Prop :=
  ∀ r ∈ R, ∃ n : ℕ, n % 2 = 0 ∧ (R.filter (λ s, adjacent r s)).card = n

-- Main theorem: proving the question
theorem moving_even_adj_red (R : RedSquares) 
  (h_red_square_cardinality : red_square_cardinality R) : even_adj_red R := 
  sorry

end moving_even_adj_red_l144_144346


namespace different_fractions_count_l144_144358

theorem different_fractions_count (n : ℕ) (hn : n ≥ 2) : 
  (number_of_different_fractions n) = 2^(n-2) :=
sorry

end different_fractions_count_l144_144358


namespace train_length_calculation_l144_144571

noncomputable def train_speed_kmph : ℝ := 126
noncomputable def time_to_cross_seconds : ℝ := 2.856914303998537
noncomputable def speed_conversion_factor : ℝ := 5 / 18

theorem train_length_calculation : 
  let speed_mps := train_speed_kmph * speed_conversion_factor in
  let train_length := speed_mps * time_to_cross_seconds in
  train_length = 99.992 :=
by
  sorry

end train_length_calculation_l144_144571


namespace additional_toothpicks_needed_l144_144372

theorem additional_toothpicks_needed 
  (t : ℕ → ℕ)
  (h1 : t 1 = 4)
  (h2 : t 2 = 10)
  (h3 : t 3 = 18)
  (h4 : t 4 = 28)
  (h5 : t 5 = 40)
  (h6 : t 6 = 54) :
  t 6 - t 4 = 26 :=
by
  sorry

end additional_toothpicks_needed_l144_144372


namespace cos_neg_300_eq_half_l144_144985

theorem cos_neg_300_eq_half :
  ∃ (θ : ℝ), θ = -300 ∧ cos θ = 1/2 :=
by
  use -300
  split
  { refl }
  { sorry }

end cos_neg_300_eq_half_l144_144985


namespace domain_of_f_l144_144791

def f (x : ℝ) : ℝ := (1 / x) + Real.log (1 - 2 * x)

def domain (x : ℝ) : Prop :=
  x < 1 / 2 ∧ x ≠ 0

theorem domain_of_f :
  ∀ x : ℝ, (x < 1 / 2 ∧ x ≠ 0) ↔ domain x := by
  sorry

end domain_of_f_l144_144791


namespace salesman_bonus_l144_144879

theorem salesman_bonus (S B : ℝ) 
  (h1 : S > 10000) 
  (h2 : 0.09 * S + 0.03 * (S - 10000) = 1380) 
  : B = 0.03 * (S - 10000) :=
sorry

end salesman_bonus_l144_144879


namespace log_base_4_frac_l144_144223

theorem log_base_4_frac :
  logb 4 (1/64) = -3 :=
sorry

end log_base_4_frac_l144_144223


namespace q_correct_l144_144038

noncomputable def q (x : ℝ) : ℝ := 10 * x^4 + 36 * x^3 + 37 * x^2 + 5 - (2 * x^6 + 4 * x^4 + 12 * x^2)

theorem q_correct (x : ℝ) :
  q(x) = -2 * x^6 + 6 * x^4 + 36 * x^3 + 25 * x^2 + 5 :=
by
  sorry

end q_correct_l144_144038


namespace correctness_l144_144452

def valid_expressions : Nat :=
  let a : Real := 1 -- arbitrary non-zero value to satisfy a ≠ 0
  let x : Real := 1
  let ex1 := a^4 / a^4 = 1
  let ex2 := (1/4 - 0.25)^0 = 1 
  let ex3 := (-0.1)^3 = -1/1000
  let ex4 := a^2 + a^2 = 2 * a^2
  let ex5 := a^2 * a^3 = a^5
  let ex6 := (-3 * x)^3 / (3 * x) = -9 * x^2
  [ex1, ex2, ex3, ex4, ex5, ex6].count (λ b, b = true)

theorem correctness : valid_expressions = 1 :=
by
  have h1 : a^4 / a^4 = 1 := by sorry
  have h2 : (1/4 - 0.25)^0 = 1 := by sorry
  have h3 : (-0.1)^3 = -1/1000 := by sorry
  have h4 : a^2 + a^2 = 2 * a^2 := by sorry
  have h5 : a^2 * a^3 = a^5 := by sorry
  have h6 : (-3 * x)^3 / (3 * x) = -9 * x^2 := by sorry
  have exs := [h1, h2, h3, h4, h5, h6]
  exact congrArg List.count (by finish)

end correctness_l144_144452


namespace cookies_per_pack_l144_144956

theorem cookies_per_pack
  (trays : ℕ) (cookies_per_tray : ℕ) (packs : ℕ)
  (h1 : trays = 8) (h2 : cookies_per_tray = 36) (h3 : packs = 12) :
  (trays * cookies_per_tray) / packs = 24 :=
by
  sorry

end cookies_per_pack_l144_144956


namespace distinct_three_digit_numbers_distinct_three_digit_odd_numbers_l144_144840

-- Definitions based on conditions
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}
def odd_digits : Finset ℕ := {1, 3, 5}

-- Problem 1: Number of distinct three-digit numbers
theorem distinct_three_digit_numbers : (digits.erase 0).card * (digits.erase 0).card.pred * (digits.erase 0).card.pred.pred = 100 := by
  sorry

-- Problem 2: Number of distinct three-digit odd numbers
theorem distinct_three_digit_odd_numbers : (odd_digits.card) * (digits.erase 0).card.pred * (digits.erase 0).card.pred.pred = 48 := by
  sorry

end distinct_three_digit_numbers_distinct_three_digit_odd_numbers_l144_144840


namespace abs_condition_sufficient_not_necessary_l144_144884

theorem abs_condition_sufficient_not_necessary:
  (∀ x : ℝ, (-2 < x ∧ x < 3) → (-1 < x ∧ x < 3)) :=
by
  sorry

end abs_condition_sufficient_not_necessary_l144_144884


namespace variance_binomial_l144_144302

-- Define the random variable and its properties
def binomial_distribution (n : ℕ) (p : ℚ) := sorry

-- Define X as a binomial random variable with given parameters
def X : Type := binomial_distribution 8 (3/4)

-- Define the variance of a binomial distribution
def variance (X : Type) : ℚ := sorry

-- The theorem we need to prove
theorem variance_binomial (X : Type) (h : X = binomial_distribution 8 (3/4)) : variance X = 3/2 := 
by sorry

end variance_binomial_l144_144302


namespace number_of_four_digit_numbers_l144_144318

/-- The number of four-digit numbers greater than 3999 where the product of the middle two digits exceeds 8 is 2400. -/
def four_digit_numbers_with_product_exceeding_8 : Nat :=
  let first_digit_options := 6  -- Choices are 4,5,6,7,8,9
  let last_digit_options := 10  -- Choices are 0,1,...,9
  let middle_digit_options := 40  -- Valid pairs counted from the solution
  first_digit_options * middle_digit_options * last_digit_options

theorem number_of_four_digit_numbers : four_digit_numbers_with_product_exceeding_8 = 2400 :=
by
  sorry

end number_of_four_digit_numbers_l144_144318


namespace probability_two_consecutive_wins_l144_144019

variable (P_A1 P_A2 P_A3 : ℚ)
variable (P_A1_win P_A2_win P_A3_win : ℚ)

-- Define the probabilities
def P_A1 := 1/4
def P_A2 := 1/3
def P_A3 := 1/3

-- Define the scenarios
def P_scenario1 := P_A1 * P_A2 * (1 - P_A3)
def P_scenario2 := (1 - P_A1) * P_A2 * P_A3

-- Calculate the total probability
def P_total := P_scenario1 + P_scenario2

-- The theorem stating that the total probability is 5/36
theorem probability_two_consecutive_wins :
  P_total = 5/36 := by
  sorry

end probability_two_consecutive_wins_l144_144019


namespace frequency_polygon_exists_l144_144600

def dataPoints : List (ℕ × ℕ) := [(1, 20), (4, 10), (5, 14), (7, 6)]

theorem frequency_polygon_exists : 
  ∃ p1 p2 p3 p4 : ℕ × ℕ, 
  p1 = (1, 20) ∧ 
  p2 = (4, 10) ∧ 
  p3 = (5, 14) ∧ 
  p4 = (7, 6) ∧ 
  (lineseg p1 p2) ∧ 
  (lineseg p2 p3) ∧ 
  (lineseg p3 p4) :=
by
  let p1 : ℕ × ℕ := (1, 20)
  let p2 : ℕ × ℕ := (4, 10)
  let p3 : ℕ × ℕ := (5, 14)
  let p4 : ℕ × ℕ := (7, 6)
  exact ⟨p1, p2, p3, p4, rfl, rfl, rfl, rfl, sorry, sorry, sorry⟩

end frequency_polygon_exists_l144_144600


namespace slope_of_line_AB_l144_144288

open Real

/-- Prove the slope of the line passing through points A(3 - √3, 6 - √3) and B(3 + 2√3, 3 - √3) is -√3/3. -/
theorem slope_of_line_AB : 
  let A := (3 - sqrt 3, 6 - sqrt 3)
  let B := (3 + 2 * sqrt 3, 3 - sqrt 3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  in slope = - sqrt 3 / 3 :=
by
  let A := (3 - sqrt 3, 6 - sqrt 3)
  let B := (3 + 2 * sqrt 3, 3 - sqrt 3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  show slope = - sqrt 3 / 3
  sorry

end slope_of_line_AB_l144_144288


namespace cube_path_count_l144_144154

noncomputable def numberOfWaysToMoveOnCube : Nat :=
  20

theorem cube_path_count :
  ∀ (cube : Type) (top bottom side1 side2 side3 side4 : cube),
    (∀ (p : cube → cube → Prop), 
      (p top side1 ∨ p top side2 ∨ p top side3 ∨ p top side4) ∧ 
      (p side1 bottom ∨ p side2 bottom ∨ p side3 bottom ∨ p side4 bottom)) →
    numberOfWaysToMoveOnCube = 20 :=
by
  intros
  sorry

end cube_path_count_l144_144154


namespace find_x_l144_144357

theorem find_x (x : ℝ) 
  (h: 3 * x + 6 * x + 2 * x + x = 360) : 
  x = 30 := 
sorry

end find_x_l144_144357


namespace smallest_value_of_n_l144_144104

theorem smallest_value_of_n 
  (n : ℕ) 
  (h1 : ∀ θ : ℝ, θ = (n - 2) * 180 / n) 
  (h2 : ∀ α : ℝ, α = 360 / n) 
  (h3 : 28 = 180 / n) :
  n = 45 :=
sorry

end smallest_value_of_n_l144_144104


namespace values_of_abc_range_of_fx_l144_144284

-- Problem setup for Question 1
def poly_eqn_condition (x a b c : ℝ) : Prop :=
  x^3 - 2 * x^2 - x + 2 = (x + a) * (x + b) * (x + c)

-- Problem setup for Question 2
def f (a b c x : ℝ) : ℝ :=
  a * x^2 + 2 * b * x + c

theorem values_of_abc (a b c : ℝ) (h₁ : ∀ x : ℝ, poly_eqn_condition x a b c) (h₂ : a > b) (h₃ : b > c) :
  a = 1 ∧ b = -1 ∧ c = -2 :=
sorry

theorem range_of_fx (f : ℝ → ℝ) (h₁ : f = λ x, (1 : ℝ) * x^2 + 2 * (-1 : ℝ) * x + (-2 : ℝ)) :
  set.range (λ x : ℝ, f x) = set.Icc (-3 : ℝ) 1 :=
sorry

end values_of_abc_range_of_fx_l144_144284


namespace tan_product_sqrt_seven_l144_144965

theorem tan_product_sqrt_seven :
  tan (Real.pi / 7) * tan (2 * Real.pi / 7) * tan (3 * Real.pi / 7) = Real.sqrt 7 := 
sorry

end tan_product_sqrt_seven_l144_144965


namespace part1_cos_alpha_minus_beta_part2_tan_C_l144_144888

-- Definitions for part (1)
def sin_alpha_minus_sin_beta (α β : ℝ) := sin α - sin β = -1 / 3
def cos_alpha_minus_cos_beta (α β : ℝ) := cos α - cos β = 1 / 2

-- Proof statement for part (1)
theorem part1_cos_alpha_minus_beta (α β : ℝ) (H1 : sin_alpha_minus_sin_beta α β) (H2 : cos_alpha_minus_cos_beta α β) :
  cos (α - β) = 59 / 72 := by 
  -- We skip the proof steps, directly stating the conclusion.
  sorry

-- Definitions for part (2)
def roots_of_quadratic_tan (A B : ℝ) := 
  (3 * A^2 - 7 * A + 2 = 0) ∧ (3 * B^2 - 7 * B + 2 = 0)

-- Proof statement for part (2)
theorem part2_tan_C (A B C : ℝ) (H3 : roots_of_quadratic_tan A B) :
  tan C = -7 := by
  -- We skip the proof steps, directly stating the conclusion.
  sorry

end part1_cos_alpha_minus_beta_part2_tan_C_l144_144888


namespace quadratic_other_root_l144_144686

noncomputable def other_root (k l m: ℝ) : ℝ :=
  (m - k) / (k - l)

theorem quadratic_other_root (k l m: ℝ)
  (h: 2*(k-l)*(2:ℝ)^2 + 3*(l-m)*(2:ℝ) + 4*(m-k) = 0) :
  ∃ r: ℝ, r = other_root k l m :=
begin
  use other_root k l m,
  sorry
end

end quadratic_other_root_l144_144686


namespace second_smallest_is_4_l144_144957

theorem second_smallest_is_4 : ∀ (a b c d e : ℕ), {a, b, c, d, e} = {5, 8, 4, 3, 7} → second_smallest_number a b c d e = 4 :=
by
  intros a b c d e h,
  -- We would define the second_smallest_number function here
  sorry

end second_smallest_is_4_l144_144957


namespace pentagon_area_ratio_is_correct_correct_m_plus_n_l144_144383

noncomputable def pentagon_area_ratio : ℚ :=
  let AB := 3
  let BC := 5
  let DE := 15
  let angle_ABC := 120
  let AC := Real.sqrt (AB^2 + BC^2 - 2 * AB * BC * Real.cos (angle_ABC * Real.pi / 180))
  let area_ratio := (AC / DE) * (AC / DE * AB * (BC - AB * Real.cos (angle_ABC * Real.pi / 180)) / (2 * (AB + BC - AB * Real.cos(120 * Real.pi / 180) + 15 / AC)))
  area_ratio

theorem pentagon_area_ratio_is_correct :
  let AB := 3
  let BC := 5
  let DE := 15
  let angle_ABC := 120
  let AC := Real.sqrt (AB^2 + BC^2 - 2 * AB * BC * Real.cos (angle_ABC * Real.pi / 180))
  let area_ratio := (AC / DE) * (AC / DE * AB * (BC - AB * Real.cos (angle_ABC * Real.pi / 180)) / (2 * (AB + BC - AB * Real.cos(120 * Real.pi / 180) + 15 / AC)))
  area_ratio = 49 / 435 :=
by
  -- Placeholder proof
  sorry

theorem correct_m_plus_n :
  let m := 49
  let n := 435
  let sum_mn := m + n
  sum_mn = 484 :=
by
  -- Placeholder proof
  sorry

end pentagon_area_ratio_is_correct_correct_m_plus_n_l144_144383


namespace next_leap_year_visible_after_2017_l144_144782

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0) ∧ ((y % 100 ≠ 0) ∨ (y % 400 = 0))

def stromquist_visible (start_year interval next_leap : ℕ) : Prop :=
  ∃ k : ℕ, next_leap = start_year + k * interval ∧ is_leap_year next_leap

theorem next_leap_year_visible_after_2017 :
  stromquist_visible 2017 61 2444 :=
  sorry

end next_leap_year_visible_after_2017_l144_144782


namespace valid_four_digit_numbers_count_l144_144312

noncomputable def num_valid_four_digit_numbers : ℕ := 6 * 65 * 10

theorem valid_four_digit_numbers_count :
  num_valid_four_digit_numbers = 3900 :=
by
  -- We provide the steps in the proof to guide the automation
  let total_valid_numbers := 6 * 65 * 10
  have h1 : total_valid_numbers = 3900 := rfl
  exact h1

end valid_four_digit_numbers_count_l144_144312


namespace pretty_numbers_sum_2019_div_20_l144_144203

def is_pretty (k n : ℕ) : Prop :=
  ∃ d : ℕ, (nat.totient n = k) ∧ (n % k = 0)

def sum_pretty_numbers_below (k bound : ℕ) : ℕ :=
  (finset.range bound).filter (is_pretty k) |>.sum id

theorem pretty_numbers_sum_2019_div_20 :
  (sum_pretty_numbers_below 20 2019 / 20 = 472) :=
sorry

end pretty_numbers_sum_2019_div_20_l144_144203


namespace find_possible_q_l144_144752

noncomputable def possible_geometric_q (a : ℕ) (q : ℕ) (n : ℕ) : Prop :=
  ∃ m t p : ℕ, m > 0 ∧ t > 0 ∧ p > 0 ∧ 
  a = 2^81 ∧ q ∈ {2, 8, 512, 134217728, 2^81} ∧
  2^81 * q^(m-1) * 2^81 * q^(t-1) = 2^81 * q^(p-1)

theorem find_possible_q :
  ∀ (q : ℕ), possible_geometric_q 2^81 q 81 →
  q ∈ {2, 8, 512, 134217728, 2^81} := 
by
  sorry

end find_possible_q_l144_144752


namespace correct_exponentiation_rule_l144_144121

theorem correct_exponentiation_rule (x y : ℝ) : ((x^2)^3 = x^6) :=
  by sorry

end correct_exponentiation_rule_l144_144121


namespace expression_evaluation_l144_144202

theorem expression_evaluation : -20 + 8 * (5 ^ 2 - 3) = 156 := by
  sorry

end expression_evaluation_l144_144202


namespace four_digit_numbers_l144_144322

theorem four_digit_numbers (n : ℕ) :
    (∃ a b c d : ℕ, 
        n = a * 1000 + b * 100 + c * 10 + d 
        ∧ 4 ≤ a ∧ a ≤ 9 
        ∧ 1 ≤ b ∧ b ≤ 9 
        ∧ 1 ≤ c ∧ c ≤ 9 
        ∧ 0 ≤ d ∧ d ≤ 9 
        ∧ b * c > 8) → n ∈ {n | 4000 ≤ n ∧ n < 10000}
           → n ∈ {n | 4000 ≤ n ∧ n < 10000 ∧ b * c > 8} := sorry

end four_digit_numbers_l144_144322


namespace identical_numbers_in_grid_l144_144355

theorem identical_numbers_in_grid (s : ℕ) (grid : ℕ → ℕ → ℤ) 
    (h_const_sum : ∀ (r1 r2 r3 r4 : ℕ), 
      r1 + r3 ≤ 1000 ∧ r2 + r4 ≤ 1000 → (r3 - r1 + 1) * (r4 - r2 + 1) = s → 
      (∑ i in finset.range (r3 - r1 + 1), ∑ j in finset.range (r4 - r2 + 1), grid (r1 + i) (r2 + j)) = C) : 
    (∀ i j : ℕ, i < 1000 → j < 1000 → grid i j = grid 0 0) ↔ s = 1 :=
begin
  sorry
end

end identical_numbers_in_grid_l144_144355


namespace triangle_angle_and_area_l144_144362

section Geometry

variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def triangle_sides_opposite_angles (a b c : ℝ) (A B C : ℝ) : Prop := 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

def vectors_parallel (a b : ℝ) (A B : ℝ) : Prop := 
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A

-- Problem statement
theorem triangle_angle_and_area (A B C a b c : ℝ) : 
  triangle_sides_opposite_angles a b c A B C ∧ vectors_parallel a b A B ∧ a = Real.sqrt 7 ∧ b = 2 ∧ A = Real.pi / 3
  → A = Real.pi / 3 ∧ (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
sorry

end Geometry

end triangle_angle_and_area_l144_144362


namespace ellipse_x_intercept_l144_144933

theorem ellipse_x_intercept (x : ℝ) :
  let f1 := (0, 3)
  let f2 := (4, 0)
  let origin := (0, 0)
  let d := sqrt ((fst f1)^2 + (snd f1)^2) + sqrt ((fst f2)^2 + (snd f2)^2)
  d = 7 → -- sum of distances from origin to the foci is 7
  (d_1 : ℝ := abs x - 4 + sqrt (x^2 + 9))
  d_1 = 7 → -- sum of distances from (x, 0) to the foci is 7
  x ≠ 0 → -- x is not 0 because the other x-intercept is not (0, 0)
  x = 56 / 11 → -- x > 4
  (x, 0) = ((56 : ℝ) / 11, 0) :=
by
  sorry

end ellipse_x_intercept_l144_144933


namespace parallel_lines_a_value_l144_144663

theorem parallel_lines_a_value (a : ℝ) 
  (h1 : ∀ (x y : ℝ), x + a * y - 1 = 0) 
  (h2 : ∀ (x y : ℝ), a * x + 4 * y + 2 = 0)
  (h_parallel : ∀ m1 m2 : ℝ, (m1 = (-1 / a)) ∧ (m2 = (-a / 4)) ∧ (m1 = m2)) : 
  a = 2 :=
begin
  sorry
end

end parallel_lines_a_value_l144_144663


namespace spherical_coordinates_of_point_l144_144973

theorem spherical_coordinates_of_point
  (x y z : ℝ) (h_point : (x, y, z) = (4 * real.sqrt 3, -2, 5))
  (ρ θ φ : ℝ)
  (h_conditions : ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * real.pi ∧ 0 ≤ φ ∧ φ ≤ real.pi) :
  (ρ = real.sqrt (x^2 + y^2 + z^2) ∧
   φ = real.arccos (z / real.sqrt (x^2 + y^2 + z^2)) ∧
   (θ = 2 * real.pi - real.arctan (Real.abs y / Real.abs x) ∨ θ = real.arctan2 y x)) →
  (ρ, θ, φ) = (real.sqrt 77, 11 * real.pi / 6, real.arccos (5 / real.sqrt 77)) :=
sorry

end spherical_coordinates_of_point_l144_144973


namespace smallest_three_digit_geometric_seq_l144_144512

/-!

# The Smallest Three-Digit Integer with Distinct Digits in a Geometric Sequence

## Problem Statement
Prove that the smallest three-digit integer whose digits are distinct and form a geometric sequence is 248.

## Conditions
1. The integer must be three digits long.
2. The integer's digits must be distinct.
3. The digits must form a geometric sequence with a common ratio \( r > 1 \).

-/

def hundred_digit := 2
def tens_digit := 2 * 2
def units_digit := 2 * 2 * 2

theorem smallest_three_digit_geometric_seq :
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧
  let a := ((n / 100) % 10) in
  let b := ((n / 10) % 10) in
  let c := (n % 10) in
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  b = a * 2 ∧ c = a * 2 * 2 ∧
  n = 248 :=
sorry

end smallest_three_digit_geometric_seq_l144_144512


namespace induction_step_factor_l144_144839

theorem induction_step_factor (k : ℕ) (h₀ : k > 0)
  (h₁ : ∀ n, (∏ i in (finset.range n).image (λ i, i + 1 + n), (i : ℕ)) = 2^n * ∏ i in (finset.range n).filter (λ i, odd i), (i+1) )
  : (∏ i in (finset.range (k+1)).image (λ i, i + 1 + k), (i : ℕ)) = (∏ i in (finset.range k).image (λ i, i + 1 + k), (i : ℕ)) * (2 * (2*k + 1)) :=
sorry

end induction_step_factor_l144_144839


namespace gcd_factorial_l144_144859

theorem gcd_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  sorry

end gcd_factorial_l144_144859


namespace soccer_tournament_probability_l144_144612

theorem soccer_tournament_probability :
  let num_teams := 8
  let games_each := (num_teams * (num_teams - 1)) / 2
  let wins_a_b := 1
  let games_left := 6
  let prob_a_ties_b := (Nat.choose games_left (games_left / 2)) / (2 ^ games_left)
  let prob_a_gt_b := 1 - (prob_a_ties_b + prob_a_ties_b / 2)
  let prob_fract : ℚ := prob_a_gt_b
  let m := 19
  let n := 32
  let gcd := Nat.gcd m n
  let p := m / gcd
  let q := n / gcd
  in p + q = 51 :=
by
  sorry

end soccer_tournament_probability_l144_144612


namespace tree_height_at_2_years_l144_144919

-- Define the conditions
def triples_height (height : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, height (n + 1) = 3 * height n

def height_at_5_years (height : ℕ → ℕ) : Prop :=
  height 5 = 243

-- Set up the problem statement
theorem tree_height_at_2_years (height : ℕ → ℕ) 
  (H1 : triples_height height) 
  (H2 : height_at_5_years height) : 
  height 2 = 9 :=
sorry

end tree_height_at_2_years_l144_144919


namespace polygon_area_l144_144478

-- Defining the side length of each square sheet
def side_length : ℝ := 8

-- Defining the rotation angles
def rotation1 : ℝ := 45
def rotation2 : ℝ := 90

-- Prove that the area of the polygon formed by the given conditions equals 64√2
theorem polygon_area (s : ℝ) (r1 r2 : ℝ) (h_s : s = 8) (h_r1 : r1 = 45) (h_r2 : r2 = 90) :
  let R := s * Real.sqrt 2 / 2
      sin22_5 := Real.sin (22.5 * Real.pi / 180)
      side_length := 2 * R * sin22_5
      area := 2 * (1 + Real.sqrt 2) * side_length ^ 2
  in area = 64 * Real.sqrt 2 :=
by
  sorry

end polygon_area_l144_144478


namespace current_short_trees_l144_144465

-- Definitions of conditions in a)
def tall_trees : ℕ := 44
def short_trees_planted : ℕ := 57
def total_short_trees_after_planting : ℕ := 98

-- Statement to prove the question == answer given conditions
theorem current_short_trees (S : ℕ) (h : S + short_trees_planted = total_short_trees_after_planting) : S = 41 :=
by
  -- Proof would go here
  sorry

end current_short_trees_l144_144465


namespace second_train_length_is_120_l144_144107

noncomputable def length_of_second_train
  (speed_train1_kmph : ℝ) 
  (speed_train2_kmph : ℝ) 
  (crossing_time : ℝ) 
  (length_train1_m : ℝ) : ℝ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600
  let relative_speed := speed_train1_mps + speed_train2_mps
  let distance := relative_speed * crossing_time
  distance - length_train1_m

theorem second_train_length_is_120 :
  length_of_second_train 60 40 6.119510439164867 50 = 120 :=
by
  -- Here's where the proof would go
  sorry

end second_train_length_is_120_l144_144107


namespace min_moves_to_monochrome_l144_144541

-- Define the initial state of the 7x7 chessboard with alternating colors
def initial_chessboard : ℕ → ℕ → bool
| r, c => (r + c) % 2 = 0

-- Define the operation of inverting colors within an m x n rectangle
def invert_rectangle (board : ℕ → ℕ → bool) (r_start c_start m n : ℕ) : ℕ → ℕ → bool :=
  λ r c => if r_start ≤ r ∧ r < r_start + m ∧ c_start ≤ c ∧ c < c_start + n then
             ¬ board r c
           else
             board r c

-- Define the proposition: the minimum number of moves to make the entire board a single color is 6
theorem min_moves_to_monochrome : ∀ (invert_rectangle : (ℕ → ℕ → bool) → ℕ → ℕ → ℕ → ℕ → (ℕ → ℕ → bool)), 
                                 (initial_chessboard : ℕ → ℕ → bool) → 
                                 (∀ (r c : ℕ), ¬initial_chessboard r c) :=
  sorry

end min_moves_to_monochrome_l144_144541


namespace max_value_of_expression_l144_144704

theorem max_value_of_expression (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_product : a * b * c = 16) : 
  a^b - b^c + c^a ≤ 263 :=
sorry

end max_value_of_expression_l144_144704


namespace rectangle_perimeter_l144_144537

theorem rectangle_perimeter (ABCD : Type) (A B C D : Point) (E : Point) (F : Point)
  (hABCD_rect : Rectangle ABCD A B C D)
  (hB_to_B' : ∃ (B' : Point), matched_to B B' AD)
  (hE_on_AB : is_on E AB)
  (hF_on_CD : is_on F CD)
  (hAE : dist A E = 5)
  (hBE : dist B E = 25)
  (hCF : dist C F = 5) :
  perimeter ABCD = 10 * (1 + sqrt 26) :=
begin
  sorry
end

end rectangle_perimeter_l144_144537


namespace num_adults_on_field_trip_l144_144602

-- Definitions of the conditions
def num_vans : Nat := 6
def people_per_van : Nat := 9
def num_students : Nat := 40

-- The theorem to prove
theorem num_adults_on_field_trip : (num_vans * people_per_van) - num_students = 14 := by
  sorry

end num_adults_on_field_trip_l144_144602


namespace compute_expression_l144_144961

theorem compute_expression :
  let num := (1 + 22) * (1 + 22/2) * (1 + 22/3) * (1 + 22/4) * (1 + 22/5) * (1 + 22/6) * (1 + 22/7) * (1 + 22/8) * (1 + 22/9) * (1 + 22/10) * (1 + 22/11) * (1 + 22/12) * (1 + 22/13) * (1 + 22/14) * (1 + 22/15) * (1 + 22/16) * (1 + 22/17) * (1 + 22/18) * (1 + 22/19) * (1 + 22/20) * (1 + 22/21) * (1 + 22/22) * (1 + 22/23) * (1 + 22/24) * (1 + 22/25)
  let denom := (1 + 25) * (1 + 25/2) * (1 + 25/3) * (1 + 25/4) * (1 + 25/5) * (1 + 25/6) * (1 + 25/7) * (1 + 25/8) * (1 + 25/9) * (1 + 25/10) * (1 + 25/11) * (1 + 25/12) * (1 + 25/13) * (1 + 25/14) * (1 + 25/15) * (1 + 25/16) * (1 + 25/17) * (1 + 25/18) * (1 + 25/19) * (1 + 25/20) * (1 + 25/21) * (1 + 25/22)
  in (num / denom) = 1 := by
  sorry

end compute_expression_l144_144961


namespace area_formula_min_area_q_eq_p_plus_1_min_area_pq_eq_neg_1_l144_144267

def parabola (x : ℝ) := 1 - x^2

def point_P (p : ℝ) : ℝ × ℝ := (p, parabola p)

def point_Q (q : ℝ) : ℝ × ℝ := (q, parabola q)

def area_under_curve (p q : ℝ) : ℝ :=
  (q - q^3 / 3) - (p - p^3 / 3)

def triangle_area (p q : ℝ) : ℝ :=
  1 / 2 * abs (p + q * p^2 - q - p * q^2)

def enclosed_area (p q : ℝ) : ℝ :=
  area_under_curve p q - triangle_area p q

theorem area_formula (p q : ℝ) (h : p < q) :
  enclosed_area p q = (q - q^3 / 3) - (p - p^3 / 3) - 1 / 2 * abs (p + q * p^2 - q - p * q^2) :=
sorry

theorem min_area_q_eq_p_plus_1 (p : ℝ) :
  enclosed_area p (p + 1) = 1 / 2 :=
sorry

theorem min_area_pq_eq_neg_1 (p : ℝ) :
  enclosed_area p (-1 / p) = 7 / 3 :=
sorry

end area_formula_min_area_q_eq_p_plus_1_min_area_pq_eq_neg_1_l144_144267


namespace cone_volume_l144_144666

theorem cone_volume (S : ℝ) (h_S : S = 12 * Real.pi) (h_lateral : ∃ r : ℝ, S = 3 * Real.pi * r^2) :
    ∃ V : ℝ, V = (8 * Real.sqrt 3 * Real.pi / 3) :=
by
  sorry

end cone_volume_l144_144666


namespace sphere_radius_same_volume_as_cone_and_cylinder_l144_144901

theorem sphere_radius_same_volume_as_cone_and_cylinder :
  ∀ (h : ℝ) (r_cone : ℝ), h = 6 ∧ r_cone = 1.5 →
  ∃ r_sphere : ℝ, (∃ V : ℝ, V = (1/3) * real.pi * r_cone^2 * h ∧ V = (4/3) * real.pi * r_sphere^3) ∧ r_sphere = 1.5 :=
by
  intros h r_cone h_cond
  rcases h_cond with ⟨h_eq, r_cone_eq⟩
  use 1.5
  use (1/3) * real.pi * (1.5)^2 * 6
  split
  · split
    sorry
  · refl

end sphere_radius_same_volume_as_cone_and_cylinder_l144_144901


namespace total_time_spent_l144_144613

-- Define the conditions
def number_of_chairs : ℕ := 4
def number_of_tables : ℕ := 2
def time_per_piece : ℕ := 8

-- Prove that the total time spent is 48 minutes
theorem total_time_spent : (number_of_chairs + number_of_tables) * time_per_piece = 48 :=
by
  sorry

end total_time_spent_l144_144613


namespace common_factor_l144_144787

theorem common_factor (m n : ℕ) : common_factor 2mn (2 * m^2 * n + 6 * m * n - 4 * m^3 * n) := 
sorry

end common_factor_l144_144787


namespace interest_group_prob_and_expectation_l144_144151

variables (N M n : ℕ) (pX3 EX : ℚ) (C : ℕ → ℕ → ℚ)
noncomputable def hypergeometric_prob (N M n k : ℕ) : ℚ :=
  (C M k * C (N - M) (n - k)) / C N n

noncomputable def expected_value (n M N : ℕ) : ℚ := n * M / N

theorem interest_group_prob_and_expectation :
  N = 10 → M = 5 → n = 4 →
  pX3 = hypergeometric_prob N M n 3 →
  EX = expected_value n M N →
  pX3 = 5 / 21 ∧ EX = 2 :=
by {
  intros hN hM hn hpX3 hEX,
  rw [hN, hM, hn, hpX3, hEX],
  -- the calculations based on the given conditions would ensue here
  sorry
}

end interest_group_prob_and_expectation_l144_144151


namespace distance_from_focus_to_asymptote_area_of_triangle_l144_144682

open Real

variables a b c d: ℝ
variables x y F1 F2 P: ℝ

def hyperbola_equation := ∀ x y: ℝ, x^2 / 9 - y^2 / 16 = 1
def distance_to_asymptote := ∀ F2: ℝ, F2.distance 0 (4/3 * x) = 4
def area_triangle : ∀ F1 F2 P: ℝ, ∠F1 P F2 = 30°

theorem distance_from_focus_to_asymptote: 
  hyperbola_equation → distance_to_asymptote :=
sorry

theorem area_of_triangle: 
  hyperbola_equation → ∠F1 P F2 = 30° → area_triangle = 16 * (2 + sqrt 3) :=
sorry

end distance_from_focus_to_asymptote_area_of_triangle_l144_144682


namespace sin_alpha_sub_5o6_pi_sum_alpha_beta_l144_144652

noncomputable def problem_sin (α: ℝ) (π: ℝ) : ℝ :=
  if (π / 2 < α ∧ α < π ∧ cos α = - (3 / 10 ) * real.sqrt 10) then sin (α - (5/6) * π) else 0

theorem sin_alpha_sub_5o6_pi (α: ℝ) (π: ℝ) :
  (π / 2 < α ∧ α < π ∧ cos α = - (3 / 10 ) * real.sqrt 10) →
  problem_sin α π = ((3 * real.sqrt 10) - real.sqrt 30) / 20 :=
by
  sorry

noncomputable def problem_sum (α β π: ℝ) : ℝ :=
  if (π / 2 < α ∧ α < π ∧ π / 2 < β ∧ β < π ∧ tan β = - (1 / 2)) then α + β else 0

theorem sum_alpha_beta (α β π: ℝ) :
  (π / 2 < α ∧ α < π ∧ π / 2 < β ∧ β < π ∧ tan β = - (1 / 2)) →
  problem_sum α β π = 7 * π / 4 :=
by
  sorry

end sin_alpha_sub_5o6_pi_sum_alpha_beta_l144_144652


namespace stockholm_to_uppsala_distance_l144_144560

-- Definition of conditions
def map_distance : ℝ := 45 -- in cm
def scale1 : ℝ := 10 -- first scale 1 cm : 10 km
def scale2 : ℝ := 5 -- second scale 1 cm : 5 km
def boundary : ℝ := 15 -- first 15 cm at scale 2

-- Calculation of the two parts
def part1_distance (boundary : ℝ) (scale2 : ℝ) := boundary * scale2
def remaining_distance (map_distance boundary : ℝ) := map_distance - boundary
def part2_distance (remaining_distance : ℝ) (scale1 : ℝ) := remaining_distance * scale1

-- Total distance
def total_distance (part1 part2: ℝ) := part1 + part2

theorem stockholm_to_uppsala_distance : 
  total_distance (part1_distance boundary scale2) 
                 (part2_distance (remaining_distance map_distance boundary) scale1) 
  = 375 := 
by
  -- Proof to be provided
  sorry

end stockholm_to_uppsala_distance_l144_144560


namespace square_area_measurement_error_percentage_l144_144580

theorem square_area_measurement_error_percentage 
  (x : ℝ) (h1 : ∃ (x : ℝ), x > 0) (h2 : ∀ (y : ℝ), y = 1.38 * x) :
  let actual_area := x^2,
      measured_area := (1.38 * x)^2,
      area_error := measured_area - actual_area,
      error_percentage := (area_error / actual_area) * 100 in
  error_percentage = 90.44 := by
  sorry

end square_area_measurement_error_percentage_l144_144580


namespace other_x_intercept_l144_144947

noncomputable def ellipse_x_intercepts (f1 f2 : ℝ × ℝ) (x_intercept1 : ℝ × ℝ) : ℝ × ℝ :=
  let d := dist f1 x_intercept1 + dist f2 x_intercept1
  let x := (d^2 - 2 * d * sqrt (3^2 + (d / 2 - 4)^2)) / (2 * d - 8)
  (x, 0)

theorem other_x_intercept :
  ellipse_x_intercepts (0, 3) (4, 0) (0, 0) = (56 / 11, 0) :=
by
  sorry

end other_x_intercept_l144_144947


namespace domain_of_g_l144_144843

noncomputable def g (t : ℝ) : ℝ := 1 / ((t - 2)^2 + (t + 2)^2 + 1)

theorem domain_of_g : ∀ t : ℝ, ∃ y : ℝ, y = g t :=
begin
  intro t,
  use g t,
  exact rfl,
end

end domain_of_g_l144_144843


namespace quadratic_root_discriminant_l144_144207

theorem quadratic_root_discriminant:
  ∃ m p : ℕ, p > 0 ∧ m > 0 ∧ gcd m p = 1 ∧ (∃ n : ℕ, 3*x^2 - 7*x + 1 = 0 → x = (m + √n) / p ∨ x = (m - √n) / p ∧ n = 37) := 
begin
  sorry
end

end quadratic_root_discriminant_l144_144207


namespace trailing_zeros_in_product_of_multiples_of_5_l144_144929

theorem trailing_zeros_in_product_of_multiples_of_5 : 
  let count_multiples_of (n: ℕ) (k: ℕ) := (n / k) in
  let multiples_of_5 := count_multiples_of 2020 5 in
  let multiples_of_25 := count_multiples_of 2020 25 in
  let multiples_of_125 := count_multiples_of 2020 125 in
  let multiples_of_625 := count_multiples_of 2020 625 in
  (multiples_of_5 + multiples_of_25 + multiples_of_125 + multiples_of_625) = 503 :=
by
  sorry

end trailing_zeros_in_product_of_multiples_of_5_l144_144929


namespace original_angle_measure_l144_144788

theorem original_angle_measure : 
  ∃ x : ℝ, (90 - x) = 3 * x - 2 ∧ x = 23 :=
by
  sorry

end original_angle_measure_l144_144788


namespace ellipse_x_intercept_other_l144_144944

noncomputable def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 3), (4, 0))
noncomputable def sum_of_distances : ℝ := 7
noncomputable def first_intercept : (ℝ × ℝ) := (0, 0)

theorem ellipse_x_intercept_other 
  (foci : (ℝ × ℝ) × (ℝ × ℝ))
  (sum_of_distances : ℝ)
  (first_intercept : (ℝ × ℝ))
  (hx : foci = ((0, 3), (4, 0)))
  (d_sum : sum_of_distances = 7)
  (intercept : first_intercept = (0, 0)) :
  ∃ (x : ℝ), x > 0 ∧ ((x, 0) = (56 / 11, 0)) := 
sorry

end ellipse_x_intercept_other_l144_144944


namespace no_consecutive_odd_sum_is_prime_l144_144769

theorem no_consecutive_odd_sum_is_prime :
  ∀ (n : ℕ) (seq : ℕ → ℕ),
    (n ≥ 2) →
    (∀ i : ℕ, i < n → seq i % 2 = 1) →
    (∀ i : ℕ, i < n - 1 → seq (i + 1) = seq i + 2) →
    ¬ prime (∑ i in Finset.range n, seq i) :=
by
  sorry

end no_consecutive_odd_sum_is_prime_l144_144769


namespace gcd_factorials_l144_144850

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  -- Proof steps would go here
  sorry

end gcd_factorials_l144_144850


namespace angle_PAQ_eq_angle_PEQ_l144_144583

theorem angle_PAQ_eq_angle_PEQ (Ω : Type) [circle Γ Ω] 
  (A B C D E P Q : Ω) 
  (h_consec : consecutive A B C D E)
  (h_angles : angle_subtended_by_arc Γ A B C = angle_subtended_by_arc Γ B C D ∧ 
              angle_subtended_by_arc Γ B C D = angle_subtended_by_arc Γ C D E)
  (h_P_on_AD : segment_contains A D P)
  (h_Q_on_BE : segment_contains B E Q)
  (h_P_on_CQ : segment_contains C Q P) :
  ∠ P A Q = ∠ P E Q := 
sorry

end angle_PAQ_eq_angle_PEQ_l144_144583


namespace Petya_can_obtain_1001_through_operations_l144_144763

theorem Petya_can_obtain_1001_through_operations :
  ∃ n : ℕ, ∃ f : ℕ → ℚ, (∀ k : ℕ, (f k) ∈ set.Icc (1/3 : ℚ) 2) ∧ (foldl (λ x, λ c, x * f c) n (range k) = 1001) :=
sorry

end Petya_can_obtain_1001_through_operations_l144_144763


namespace Tim_cookie_packages_l144_144481

theorem Tim_cookie_packages 
    (cookies_in_package : ℕ)
    (packets_in_package : ℕ)
    (min_packet_count : ℕ)
    (h1 : cookies_in_package = 5)
    (h2 : packets_in_package = 7)
    (h3 : min_packet_count = 30) :
  ∃ (cookie_packages : ℕ) (packet_packages : ℕ),
    cookie_packages = 7 ∧ packet_packages = 5 ∧
    cookie_packages * cookies_in_package = packet_packages * packets_in_package ∧
    packet_packages * packets_in_package ≥ min_packet_count :=
by
  sorry

end Tim_cookie_packages_l144_144481


namespace smallest_positive_period_of_f_cos_2alpha_given_alpha_l144_144674

noncomputable def f (x : ℝ) : ℝ :=
  sin (2*x - π/3) + cos (2*x - π/6) + 2 * cos x ^ 2 - 1

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ ε > 0, ∃ x, 0 < x < ε ∧ (f (x + T) ≠ f x)) ∧ T = π :=
sorry

theorem cos_2alpha_given_alpha :
  ∀ α, (α ∈ set.Icc (π/4) (π/2)) → f α = (3 * Real.sqrt 2) / 5 → cos (2 * α) = -Real.sqrt 2 / 10 :=
sorry

end smallest_positive_period_of_f_cos_2alpha_given_alpha_l144_144674


namespace solve_abs_eq_l144_144871

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 :=
  sorry

end solve_abs_eq_l144_144871


namespace minimal_visible_sum_l144_144542

noncomputable def smallest_sum_of_visible_faces : ℕ :=
  let corner_sum := 8 * 6
  let edge_sum := 24 * 3
  let face_center_sum := 16 * 1
  in corner_sum + edge_sum + face_center_sum

theorem minimal_visible_sum :
  smallest_sum_of_visible_faces = 136 :=
by
  -- Calculation of sums based on conditions
  unfold smallest_sum_of_visible_faces
  sorry

end minimal_visible_sum_l144_144542


namespace math_problem_equiv_l144_144265

noncomputable def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 = 6

def point_on_hyperbola (x y : ℝ) (hx : hyperbola_eq x y) : Prop := hx

def vectors_orthogonal (m : ℝ) (hm : m^2 = 3) : Prop :=
  let MF1 := (-3 - 2 * real.sqrt 3, -m)
  let MF2 := (2 * real.sqrt 3 - 3, -m)
  (MF1.1 * MF2.1 + MF1.2 * MF2.2) = 0

def triangle_area (m : ℝ) (hm : m^2 = 3) : ℝ :=
  1/2 * 4 * real.sqrt 3 * abs m

theorem math_problem_equiv (x y m : ℝ) (hx : hyperbola_eq x y)
  (hm : m^2 = 3) :
  hyperbola_eq 4 (-(real.sqrt 10)) ∧ vectors_orthogonal m hm ∧ triangle_area m hm = 6 :=
by
  sorry

end math_problem_equiv_l144_144265


namespace cos_20_cos_10_minus_sin_160_sin_10_l144_144536

theorem cos_20_cos_10_minus_sin_160_sin_10 : 
  (Real.cos (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) - 
   Real.sin (160 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 
   Real.cos (30 * Real.pi / 180) :=
by
  sorry

end cos_20_cos_10_minus_sin_160_sin_10_l144_144536


namespace sum_of_numbers_l144_144546

variables {x y : ℝ}

theorem sum_of_numbers (h : (x - 1) * (y - 1) = x * y) : x + y = 1 := 
begin
  sorry
end

end sum_of_numbers_l144_144546


namespace distance_from_focus_l144_144268

theorem distance_from_focus (x : ℝ) (A : ℝ × ℝ) (hA_on_parabola : A.1^2 = 4 * A.2) (hA_coord : A.2 = 4) : 
  dist A (0, 1) = 5 := 
by
  sorry

end distance_from_focus_l144_144268


namespace smallest_three_digit_geometric_seq_l144_144511

/-!

# The Smallest Three-Digit Integer with Distinct Digits in a Geometric Sequence

## Problem Statement
Prove that the smallest three-digit integer whose digits are distinct and form a geometric sequence is 248.

## Conditions
1. The integer must be three digits long.
2. The integer's digits must be distinct.
3. The digits must form a geometric sequence with a common ratio \( r > 1 \).

-/

def hundred_digit := 2
def tens_digit := 2 * 2
def units_digit := 2 * 2 * 2

theorem smallest_three_digit_geometric_seq :
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧
  let a := ((n / 100) % 10) in
  let b := ((n / 10) % 10) in
  let c := (n % 10) in
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  b = a * 2 ∧ c = a * 2 * 2 ∧
  n = 248 :=
sorry

end smallest_three_digit_geometric_seq_l144_144511


namespace other_root_l144_144018

-- Definition of the condition: one root is 3.
def quadratic_equation (k : ℝ) := 3 * x^2 + k * x + 5 = 0

-- Lean 4 statement to prove the equivalent proof problem.
theorem other_root (k : ℝ) (h : quadratic_equation k 3) : 
  exists (β : ℝ), β = 5 / 9 :=
sorry

end other_root_l144_144018


namespace Sara_quarters_after_borrowing_l144_144425

theorem Sara_quarters_after_borrowing (initial_quarters borrowed_quarters : ℕ) (h1 : initial_quarters = 783) (h2 : borrowed_quarters = 271) :
  initial_quarters - borrowed_quarters = 512 := by
  sorry

end Sara_quarters_after_borrowing_l144_144425


namespace number_of_correct_statements_l144_144577

-- Defining statements based on the conditions
def statement1 : Prop :=
  ∀ (residuals: List ℝ), ¬(all_evenly_distributed_in_horizontal_band residuals → model_is_appropriate residuals)

def statement2 : Prop :=
  ∀ (R2: ℝ), (R2 ≥ 0 ∧ R2 ≤ 1) → (better_fitting_model R2 R2)

def statement3 : Prop :=
  ∀ (ssr1 ssr2: ℝ), (ssr1 < ssr2 → model_is_better_fitting ssr1 ssr2)

-- Main theorem stating the number of correct statements
theorem number_of_correct_statements : 
  statement1 → statement2 → statement3 → 
  (0 + 1 + 1 = 2) :=
by
  sorry

end number_of_correct_statements_l144_144577


namespace largest_possible_x_plus_y_l144_144827

theorem largest_possible_x_plus_y :
  ∃ (x y : ℚ),
  let F := (x, y)
  let D := (10, 17 : ℚ × ℚ)
  let E := (25, 22 : ℚ × ℚ)
  ((F = (x, y)) ∧ (D = (10, 17) : ℚ × ℚ) ∧ (E = (25, 22) : ℚ × ℚ) ∧
  (1/2 * |(x - 10) * (22 - 17) - (x - 25) * (17 - y) + (10 - 25) * (y - 22)| = 84) ∧
  (y - 39/2 = -3 * (x - 35/2))) →
  x + y = 943 / 21 := by
  sorry

end largest_possible_x_plus_y_l144_144827


namespace abs_eq_abs_implies_l144_144869

theorem abs_eq_abs_implies (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 := 
sorry

end abs_eq_abs_implies_l144_144869


namespace prob_select_math_books_l144_144017

theorem prob_select_math_books :
  let total_books := 5
  let math_books := 3
  let total_ways_select_2 := Nat.choose total_books 2
  let ways_select_2_math := Nat.choose math_books 2
  let probability := (ways_select_2_math : ℚ) / total_ways_select_2
  probability = 3 / 10 :=
by
  sorry

end prob_select_math_books_l144_144017


namespace line_length_after_erasing_l144_144028

-- Definition of the initial length in meters and the erased length in centimeters
def initial_length_meters : ℝ := 1.5
def erased_length_centimeters : ℝ := 15.25

-- Conversion factor from meters to centimeters
def meters_to_centimeters (m : ℝ) : ℝ := m * 100

-- Definition of the initial length in centimeters
def initial_length_centimeters : ℝ := meters_to_centimeters initial_length_meters

-- Statement of the theorem
theorem line_length_after_erasing :
  initial_length_centimeters - erased_length_centimeters = 134.75 :=
by
  -- The proof would go here
  sorry

end line_length_after_erasing_l144_144028


namespace remainder_of_power_sums_modulo_seven_l144_144501

theorem remainder_of_power_sums_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := 
by 
  sorry

end remainder_of_power_sums_modulo_seven_l144_144501


namespace yuan_to_scientific_notation_l144_144990

/-- Express 2.175 billion yuan in scientific notation,
preserving three significant figures. --/
theorem yuan_to_scientific_notation (a : ℝ) (h : a = 2.175 * 10^9) : a = 2.18 * 10^9 :=
sorry

end yuan_to_scientific_notation_l144_144990


namespace square_cut_total_length_l144_144170

noncomputable
def total_cut_length (side : ℝ) (num_rectangles : ℕ) (area_per_rectangle : ℝ) (cut_length1 cut_length2 : ℝ) : ℝ :=
  cut_length1 + cut_length2

theorem square_cut_total_length :
  let side := 36
  let num_rectangles := 3
  let area := side * side
  let area_per_rectangle := area / num_rectangles
  let cut_length1 := side -- The length of one cut (can be vertically or horizontally determined)
  let cut_length2 := side -- The length of the other cut sharing edges properly with the first cut
  area_per_rectangle = 432 → total_cut_length side num_rectangles area_per_rectangle cut_length1 cut_length2 = 60 :=
by
  intros
  sorry

end square_cut_total_length_l144_144170


namespace green_chips_l144_144476

def total_chips (T : ℕ) := 3 = 0.10 * T

def white_chips (T : ℕ) := 0.50 * T

theorem green_chips (T : ℕ) (h1 : total_chips T) (h2 : white_chips T) : (T - (3 + h2) = 12) :=
by sorry

end green_chips_l144_144476


namespace gcd_factorial_l144_144861

theorem gcd_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  sorry

end gcd_factorial_l144_144861


namespace change_in_money_supply_change_in_discount_rate_l144_144142

/-- Part (c) Assumptions and Proof -/
theorem change_in_money_supply (E₀ E₁ : ℝ) (fixed_rate : E₀ = 90) (new_rate : E₁ = 100) (k : ℝ) (k_eq : k = 5) :
  ∃ ΔM : ℝ, ΔM = 2 :=
by {
  have ΔE := ((E₁ - E₀) / E₀) * 100,
  have ΔE_val : ΔE = 11.11 := by sorry,
  have ΔE_rounded : ΔE ≈ 10 := by sorry,
  have ΔM := ΔE / k,
  use ΔM,
  linarith,
}

/-- Part (d) Assumptions and Proof -/
theorem change_in_discount_rate (ΔM : ℝ) (ΔE : ℝ) (per_p_p : ℕ) (per_pp_change : per_p_p = 1) (per_pp_rate : ℝ) (per_pp_rate_change : per_pp_rate = 4) :
  ∃ Δr : ℝ, Δr = 0.5 :=
by {
  have rate_change := (ΔM / per_pp_rate),
  use rate_change,
  linarith,
}

end change_in_money_supply_change_in_discount_rate_l144_144142


namespace abs_sin2x_minus_cosx_eq_abs_abs_sin2x_minus_abs_cosx_l144_144533

open Real

theorem abs_sin2x_minus_cosx_eq_abs_abs_sin2x_minus_abs_cosx (x : ℝ) :
  x ∈ Set.Ioc (-2 * π) 2 * π → 
  (|sin (2 * x) - cos x| = | |sin (2 * x)| - |cos x| | ↔
  x ∈ (Set.Ioc (-2 * π) (-π) ∪ Set.Icc 0 π ∪ { -π / 2, 3 * π / 2, 2 * π }) :=
by
  intros hx_interval
  sorry

end abs_sin2x_minus_cosx_eq_abs_abs_sin2x_minus_abs_cosx_l144_144533


namespace sequence_a_l144_144651

-- Define the sum of first n terms condition
def S (a : ℕ → ℕ) : ℕ → ℕ
| 0       := 0
| (n + 1) := S n + a (n + 1)

theorem sequence_a (a : ℕ → ℕ) (h : ∀ n, log 2 (S a n + 1) = n + 1) : 
  (a 1 = 3) ∧ (∀ n, n ≥ 2 → a n = 2^n) := 
by 
  sorry

end sequence_a_l144_144651


namespace field_area_l144_144737

theorem field_area
  (speed : ℝ)
  (time : ℝ)
  (distance : ℝ)
  (side : ℝ)
  (area : ℝ)
  (h1 : speed = 8) 
  (h2 : time = 0.5) 
  (h3 : distance = speed * time)
  (h4 : distance = side * real.sqrt 2)
  (h5 : side = 2 * real.sqrt 2)
  (h6 : area = side * side) :
  area = 8 :=
by
  sorry

end field_area_l144_144737


namespace find_f2_l144_144675

noncomputable theory

variables {R : Type*} [field R]

def f (a b : R) (x : R) : R := a * x^2 + b * real.cos x
def g (c : R) (x : R) : R := c * real.sin x

-- The conditions
variables (a b c : R) (h_abc_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
variable (h1 : f a b 2 + g c 2 = 3)
variable (h2 : f a b (-2) + g c (-2) = 1)

theorem find_f2 : f a b 2 = 2 :=
by
  sorry

end find_f2_l144_144675


namespace gcd_factorials_l144_144848

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  -- Proof steps would go here
  sorry

end gcd_factorials_l144_144848


namespace find_k_l144_144657

variables {x1 x2 k : ℝ}

theorem find_k
  (h1 : x1 ^ 2 - k * x1 - 4 = 0)
  (h2 : x2 ^ 2 - k * x2 - 4 = 0)
  (h3 : x1^2 + x2^2 + x1 * x2 = 6) :
  k = real.sqrt 2 ∨ k = -real.sqrt 2 :=
sorry

end find_k_l144_144657


namespace available_wire_length_l144_144551

   noncomputable def pi_approx : ℝ := 3.14159

   theorem available_wire_length (r : ℝ) (h : r = 2.5) : 2 * π * r ≈ 15.71 :=
   by
     have h1 : 2 * π * r = 2 * π * 2.5,
     from congr_arg (λ x, 2 * π * x) h,
     rw [←mul_assoc] at h1,
     have h2 : 2 * π * 2.5 = 5 * π,
     from by norm_num,
     rw h2 at h1,
     have h3 : 5 * pi ≈ 15.71,
     from by norm_num,
     rw h3 at h1,
     exact h1
   
end available_wire_length_l144_144551


namespace virginia_ends_up_with_93_eggs_l144_144108

-- Define the initial and subtracted number of eggs as conditions
def initial_eggs : ℕ := 96
def taken_eggs : ℕ := 3

-- The theorem we want to prove
theorem virginia_ends_up_with_93_eggs : (initial_eggs - taken_eggs) = 93 :=
by
  sorry

end virginia_ends_up_with_93_eggs_l144_144108


namespace find_h_l144_144744

-- Define the bowtie operation
noncomputable def bowtie (x y : ℝ) : ℝ :=
x + Real.sqrt (y + Real.sqrt (y + Real.sqrt (y + ...)))

-- Hypothesis for the problem
axiom hyp : bowtie 3 h = 5

-- Goal: Prove the value of h
theorem find_h (h : ℝ) : h = 2 :=
by
  -- We state the goal
  sorry

end find_h_l144_144744


namespace b_10_eq_64_l144_144691

noncomputable def a (n : ℕ) : ℕ := -- Definition of the sequence a_n
  sorry

noncomputable def b (n : ℕ) : ℕ := -- Definition of the sequence b_n
  a n + a (n + 1)

theorem b_10_eq_64 (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a n * a (n + 1) = 2^n) :
  b 10 = 64 :=
sorry

end b_10_eq_64_l144_144691


namespace basis_v_l144_144278

variable {V : Type*} [AddCommGroup V] [Module ℝ V]  -- specifying V as a real vector space
variables (a b c : V)

-- Assume a, b, and c are linearly independent, forming a basis
axiom linear_independent_a_b_c : LinearIndependent ℝ ![a, b, c]

-- The main theorem which we need to prove
theorem basis_v (h : LinearIndependent ℝ ![a, b, c]) :
  LinearIndependent ℝ ![c, a + b, a - b] :=
sorry

end basis_v_l144_144278


namespace acute_triangle_proof_problem_l144_144718

variables {A B C E F M N P R L D : Type*}

-- Given parameters in the problem
axiom acute_triangle (h_triangle: Triangle A B C) (h_acute: is_acutriangle A B C): Prop 

axiom trisectors (AE AF: Set (LineSegment A)) (h_trisectors: ∀ (tri: Triangle), tri = (A, B, C) → 
  LineSegment_trisector_of_angle AE A tri ∧ LineSegment_trisector_of_angle AF A tri): Prop

axiom circumcircle_intersections (E M N: Point) (h_E: Circumcircle_intersection E (Triangle A B C) AE M):
  ∃ E M, AE.cross_circle E (Triangle A B C) = M 

axiom angle_conditions (P R : Set (Point)) (h_PE: ∀ (angle PEA B : Angle), Angle_Condition_angle_angle PEA B)
  (h_AR: ∀ (angle C AER: Angle), Angle_Condition_angle_angle AER C): Prop 

axiom intersection1 (L: Point) (h_PR: Set (LineSegment P R)) (h_Intersection_point PR AE L: Point_intersection PR AE L):
  ∃ L, (PR ∩ AE) = L 

axiom intersection2 (D: Point) (h_LN: Set (LineSegment L N)) (h_Intersection_point LN BC D: Point_intersection LN BC D):
  ∃ D, (LN ∩ BC) = D 

theorem acute_triangle_proof_problem : 
  acute_triangle h_triangle h_acute ∧ trisectors AE AF h_trisectors ∧ circumcircle_intersections E M N h_E ∧
  angle_conditions P R h_PE h_AR ∧ intersection1 L PR h_Intersection_point PR AE ∧
  intersection2 D LN h_Intersection_point LN BC
  → (1 / distance M N) + (1 / distance E F) = 1 / distance E D := 
by sorry

end acute_triangle_proof_problem_l144_144718


namespace impossible_rearrangement_l144_144593

open Nat

theorem impossible_rearrangement : ¬ (∃ (seq : Fin 3972 → ℕ), 
  (∀ n, seq n ∈ (Finset.range 1986).image (λ i, i + 1) ∧ 
  ∀ i, ∃ (m_i n_i : Fin 3972), m_i < n_i ∧ seq m_i = i ∧ seq n_i = i ∧ n_i = m_i + i + 1)) := 
sorry

end impossible_rearrangement_l144_144593


namespace probability_of_prime_number_on_spinner_l144_144495

-- Definitions of conditions
def spinner_sections : List ℕ := [2, 3, 4, 5, 7, 9, 10, 11]
def total_sectors : ℕ := 8
def prime_count : ℕ := List.filter Nat.Prime spinner_sections |>.length

-- Statement of the theorem we want to prove
theorem probability_of_prime_number_on_spinner :
  (prime_count : ℚ) / total_sectors = 5 / 8 := by
  sorry

end probability_of_prime_number_on_spinner_l144_144495


namespace matrix_power_eight_l144_144963

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![ (Real.sqrt 2 / 2), -(Real.sqrt 2 / 2)
   ; (Real.sqrt 2 / 2),  (Real.sqrt 2 / 2)]

theorem matrix_power_eight :
  A ^ 8 = !![ 16, 0
            ; 0, 16] := 
by
  sorry

end matrix_power_eight_l144_144963


namespace largest_integer_crates_same_oranges_l144_144922

theorem largest_integer_crates_same_oranges (crates : ℕ) (minOranges maxOranges : ℕ) 
    (h1 : crates = 150) 
    (h2 : minOranges = 130) 
    (h3 : maxOranges = 150) 
    (h4 : ∀ c, c ∈ set.Icc minOranges maxOranges) : 
  ∃ n : ℕ, n = 8 ∧ ∀ t ∈ set.Icc minOranges maxOranges, ∃ S : finset ℕ, S.card ≥ n ∧ ∀ x ∈ S, x = t :=
begin
  sorry
end

end largest_integer_crates_same_oranges_l144_144922


namespace count_expansion_terms_l144_144197

theorem count_expansion_terms : 
  ∀ (a b : ℝ), 
  let expression := (a^2 - 9*b^2)^6 in 
  ∃ (terms : Finset (ℝ × ℝ)), 
  (∑ x in terms, (λ (p : ℝ × ℝ), p.1 * p.2) x = expression) ∧ 
  (terms.card = 7) := 
by
  intros a b
  let expression := (a^2 - 9*b^2)^6
  sorry

end count_expansion_terms_l144_144197


namespace tall_cupboard_glasses_l144_144601

-- Define the number of glasses held by the tall cupboard (T)
variable (T : ℕ)

-- Condition: Wide cupboard holds twice as many glasses as the tall cupboard
def wide_cupboard_holds_twice_as_many (T : ℕ) : Prop :=
  ∃ W : ℕ, W = 2 * T

-- Condition: Narrow cupboard holds 15 glasses initially, 5 glasses per shelf, one shelf broken
def narrow_cupboard_holds_after_break : Prop :=
  ∃ N : ℕ, N = 10

-- Final statement to prove: Number of glasses in the tall cupboard is 5
theorem tall_cupboard_glasses (T : ℕ) (h1 : wide_cupboard_holds_twice_as_many T) (h2 : narrow_cupboard_holds_after_break) : T = 5 :=
sorry

end tall_cupboard_glasses_l144_144601


namespace multiply_polynomials_l144_144199

variables {R : Type*} [CommRing R] -- Define R as a commutative ring
variable (x : R) -- Define variable x in R

theorem multiply_polynomials : (2 * x) * (5 * x^2) = 10 * x^3 := 
sorry -- Placeholder for the proof

end multiply_polynomials_l144_144199


namespace problem_like_terms_l144_144120

def like_terms (expr1 expr2 : Expr) : Prop :=
  match expr1, expr2 with
  | Expr.Mul x1 y1, Expr.Mul x2 y2 => (x1 = x2 ∧ y1 = y2) ∨ (x1 = y2 ∧ y1 = x2)
  | _, _ => false

theorem problem_like_terms :
  like_terms (Expr.Mul (Expr.Var "x") (Expr.Var "y")) (Expr.Mul (Expr.Var "y") (Expr.Var "x")) ∧
  ¬ like_terms (Expr.Var "x") (Expr.Var "a") ∧
  ¬ like_terms (Expr.Mul (Expr.Var "2") (Expr.Var "a")) (Expr.Mul (Expr.Var "2") (Expr.Pow (Expr.Var "a") 2)) ∧
  ¬ like_terms (Expr.Mul (Expr.Mul (Expr.Var "3") (Expr.Pow (Expr.Var "x") 3)) (Expr.Pow (Expr.Var "y") 4)) (Expr.Mul (Expr.Mul (Expr.Mul (Expr.Var "3") (Expr.Pow (Expr.Var "x") 3)) (Expr.Pow (Expr.Var "y") 4)) (Expr.Pow (Expr.Var "z") 1)) :=
by
  sorry

end problem_like_terms_l144_144120


namespace log_calculation_l144_144589

-- Definitions of logarithmic properties

def log_rule1 (a x y : ℝ) : ℝ :=
  log a (x * y) = log a x + log a y

def log_rule2 (a x y : ℝ) : ℝ :=
  log a (x / y) = log a x - log a y

-- Main theorem to be proved
theorem log_calculation : 
  log 10 60 + log 10 40 - log 10 15 = 2.204 :=
by 
  sorry

end log_calculation_l144_144589


namespace plane_through_A_perpendicular_to_BC_l144_144523

-- Definition of points
def A : ℝ × ℝ × ℝ := (-3, 5, -2)
def B : ℝ × ℝ × ℝ := (-4, 0, 3)
def C : ℝ × ℝ × ℝ := (-3, 2, 5)

-- Calculating the vector BC
def vectorBC : ℝ × ℝ × ℝ := (C.1 - B.1, C.2 - B.2, C.3 - B.3)

-- The plane equation that we wish to prove
def plane_eq (x y z : ℝ) : ℝ := x + 2 * y + 2 * z - 3

-- Statement of the proof problem
theorem plane_through_A_perpendicular_to_BC :
  ∀ (x y z : ℝ),
    ∃ k : ℝ,
    plane_eq x y z = k ↔ (x, y, z) = A :=
  sorry

end plane_through_A_perpendicular_to_BC_l144_144523


namespace construct_triangle_l144_144272

theorem construct_triangle
  (A B C A0 : Point)
  (s_a : ℝ)
  (α1 α2 : ℝ)
  (h_midpoint : midpoint A0 B C)
  (h_median : dist A A0 = s_a)
  (h_angle1 : angle A0 A B = α1)
  (h_angle2 : angle A0 A C = α2) :
  ∃ (Δ : Triangle), same_triangle Δ (Triangle.mk A B C) := by
  sorry

end construct_triangle_l144_144272


namespace milk_tea_sales_ratio_l144_144561

theorem milk_tea_sales_ratio 
(Total_sales Winter_melon_sales Chocolate_sales Okinawa_sales : ℕ)
(h1 : Total_sales = 50)
(h2 : Winter_melon_sales = (2/5) * Total_sales)
(h3 : Chocolate_sales = 15)
(h4 : Okinawa_sales = Total_sales - Winter_melon_sales - Chocolate_sales):
Okinawa_sales / Total_sales = 3 / 10 :=
begin
  sorry
end

end milk_tea_sales_ratio_l144_144561


namespace Mark_final_position_l144_144012

theorem Mark_final_position :
  ∀ (initial_position total_distance total_steps forward_steps backward_steps : ℕ),
  initial_position = 0 →
  total_distance = 40 →
  total_steps = 10 →
  forward_steps = 8 →
  backward_steps = 1 →
  let step_length := total_distance / total_steps,
      net_steps := forward_steps - backward_steps,
      final_position := net_steps * step_length in
  final_position = 28 :=
by
  intros initial_position total_distance total_steps forward_steps backward_steps hire eq_initial eq_distance eq_steps eq_forward eq_backward
  let step_length := total_distance / total_steps
  let net_steps := forward_steps - backward_steps
  let final_position := net_steps * step_length
  show final_position = 28 from
  sorry

end Mark_final_position_l144_144012


namespace log_graph_cuts_y_axis_l144_144599

theorem log_graph_cuts_y_axis (x : ℝ) (h1 : x > -1) (h2 : x = 0) : log 2 (x + 1) = 0 :=
by
  sorry

end log_graph_cuts_y_axis_l144_144599


namespace ellipse_x_intercept_l144_144937

theorem ellipse_x_intercept
  (foci1 foci2 : ℝ × ℝ)
  (x_intercept : ℝ × ℝ)
  (d : ℝ)
  (h_foci1 : foci1 = (0, 3))
  (h_foci2 : foci2 = (4, 0))
  (h_x_intercept : x_intercept = (0, 0))
  (h_d : d = 7)
  : ∃ x : ℝ, (x, 0) ≠ x_intercept ∧ (abs (x - 4) + real.sqrt (x^2 + 9) = 7) ∧ x = 56 / 11 := by
  sorry

end ellipse_x_intercept_l144_144937


namespace sum_distances_parabola_l144_144658

theorem sum_distances_parabola (P A B F : ℝ × ℝ) :
  let parabola := λ x y, y^2 = 12 * x,
      midpoint := λ P A B, P = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)),
      distance := λ P F, (Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)),
      focus := (3, 0) in
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ midpoint P A B ∧ F = focus →
  distance A F + distance B F = 10 :=
  by sorry

end sum_distances_parabola_l144_144658


namespace expression_value_l144_144873

theorem expression_value : ((40 + 15) ^ 2 - 15 ^ 2) = 2800 := 
by
  sorry

end expression_value_l144_144873


namespace gcd_factorials_l144_144849

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  -- Proof steps would go here
  sorry

end gcd_factorials_l144_144849


namespace quasi_limit_acc_point_l144_144980

variable (S : ℕ → ℝ)
variable (T : ℕ → ℝ)
variable (f : (ℕ → ℝ) → ℝ)
variable (H : set (ℕ → ℝ))
variable (epsilon : ℝ)

-- Condition (i)
axiom H_seq' : ∀ S ∈ H, (λ n, S (n+1)) ∈ H

-- Condition (ii)
axiom H_add_mul : ∀ S T, S ∈ H → (λ n, S n + T n) ∈ H ∧ (λ n, S n * T n) ∈ H 

-- Condition (iii)
axiom H_minus_one_seq : (λ n, -1) ∈ H

-- Quasi-limit properties
axiom quasi_limit_const : ∀ (c : ℝ), f (λ n, c) = c
axiom quasi_limit_nonneg : ∀ S, (∀ n, 0 ≤ S n) → 0 ≤ f S
axiom quasi_limit_add : ∀ S T, f (λ n, S n + T n) = f S + f T
axiom quasi_limit_mul : ∀ S T, f (λ n, S n * T n) = f S * f T
axiom quasi_limit_seq' : ∀ S, f (λ n, S (n+1)) = f S

-- Proof statement
theorem quasi_limit_acc_point : ∀ S ∈ H, 
  ∃ ε > 0, ∀ n, |S n - f S| < ε :=
sorry

end quasi_limit_acc_point_l144_144980


namespace explicit_formula_correct_amplitude_condition_period_condition_monotonic_intervals_range_condition_l144_144783

def amplitude (A : ℝ) (ω : ℝ) : ℝ := A

def period (A : ℝ) (ω : ℝ) : ℝ := 2 * π / ω

def f (x : ℝ) : ℝ := 2 * sin (2*x + π / 3)

theorem explicit_formula_correct : ∀ x, f x = 2 * sin (2*x + π / 3) :=
by
  sorry

theorem amplitude_condition : ∀ (A ω : ℝ), A > 0 → ω > 0 → 
  (amplitude 2 2 = 2) :=
by
  intro A ω hA hω
  sorry

theorem period_condition : ∀ (A ω : ℝ), A > 0 → ω > 0 → 
  (period 2 2 = π) :=
by
  intro A ω hA hω
  sorry

theorem monotonic_intervals : ∀ (k : ℤ), 
  -5 * π / 12 + k * π ≤ x ∧ x ≤ π / 12 + k * π :=
by
  intro k
  sorry

theorem range_condition : ∀ x ∈ Icc (-π/2) 0, 
  -2 ≤ f x ∧ f x ≤ sqrt 3 :=
by
  intro x hx
  sorry

end explicit_formula_correct_amplitude_condition_period_condition_monotonic_intervals_range_condition_l144_144783


namespace log_base_4_frac_l144_144226

theorem log_base_4_frac :
  logb 4 (1/64) = -3 :=
sorry

end log_base_4_frac_l144_144226


namespace factorial_problem_l144_144695

theorem factorial_problem (m : ℕ) (h : 4! * 5 = m!) : m = 5 := by
  sorry

end factorial_problem_l144_144695


namespace petya_can_obtain_1001_l144_144765

noncomputable def can_obtain_1001 (n : ℕ) : Prop :=
  ∃ (seq : ℕ → ℚ), (seq 0 = n) ∧ (∀ k, seq (k + 1) = seq k * (seq k * 3).toNat) ∧ (seq ? = 1001)

theorem petya_can_obtain_1001 (n : ℕ) (h : ∃ r : ℚ, r ≥ 1/3 ∧ r ≤ 2) : can_obtain_1001 n :=
by sorry

end petya_can_obtain_1001_l144_144765


namespace DE_EF_ratio_l144_144361

variables {α : Type*} [AddCommGroup α] [Module ℚ α]
variables (A B C D E F : α)
variables (k l : ℚ)

def AD_DB_ratio (A D B : α) (k : ℚ) : Prop := D = (1 / (k + 1)) • A + (k / (k + 1)) • B
def BE_EC_ratio (B E C : α) (l : ℚ) : Prop := E = (l / (l + 1)) • B + (1 / (l + 1)) • C

theorem DE_EF_ratio
  (h1 : AD_DB_ratio A D B 2)
  (h2 : BE_EC_ratio B E C (3 / 2))
  (h3 : ∃ t : ℚ, F = (1 - t) • D + t • E)
  : (∃ z : ℚ, DE_EF_ratio D E F z) := sorry

end DE_EF_ratio_l144_144361


namespace sum_of_distances_l144_144660

open Real

-- Define the points and parabola properties
structure Point where
  x : ℝ
  y : ℝ

-- Focus of the parabola y^2 = 12x
def F : Point := { x := 3, y := 0 }

-- Midpoint P(2, 1)
def P : Point := { x := 2, y := 1 }

-- Definition of parabola:y^2 = 12x
def parabola (p : Point) : Prop := p.y ^ 2 = 12 * p.x

-- Proposition of the problem
theorem sum_of_distances (A B : Point) :
  (parabola A ∧ parabola B ∧ P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2) →
  dist A F + dist B F = 10 :=
sorry

end sum_of_distances_l144_144660


namespace initial_number_is_nine_l144_144164

theorem initial_number_is_nine (x : ℕ) (h : 3 * (2 * x + 9) = 81) : x = 9 :=
by
  sorry

end initial_number_is_nine_l144_144164


namespace part_a_part_b_l144_144711

-- Part (a): For \( n \geq 5 \), prove that there exist five players such that the pairs \( ab, ac, bc, ad, ae, de \) have played each other.
theorem part_a (n : ℕ) (h : n ≥ 5) : 
  ∃ a b c d e : fin n, 
    let games_played := ⌊ n^2 / 4 ⌋ + 2 in
    (∃ (played : fin n → fin n → Prop), 
      (((played a b ∧ played a c ∧ played b c) ∧ 
       (played a d ∧ played a e ∧ played d e)) ∧ 
       (foldl Add 0 (list.map (λ p : fin n × fin n, if played p.1 p.2 then 1 else 0) 
          (fin.cartesian_prod (fin n) (fin n)))) = games_played)) :=
sorry

-- Part (b): Show that the statement is not valid for \( \left\lfloor \frac{n^2}{4} \right\rfloor + 1 \) games played.
theorem part_b (n : ℕ) (h : n ≥ 5) : 
  ¬ (∃ a b c d e : fin n, 
    let games_played := ⌊ n^2 / 4 ⌋ + 1 in
    (∃ (played : fin n → fin n → Prop), 
      ((played a b ∧ played a c ∧ played b c) ∧ 
       (played a d ∧ played a e ∧ played d e))
       ∧ (foldl Add 0 (list.map (λ p : fin n × fin n, if played p.1 p.2 then 1 else 0) 
          (fin.cartesian_prod (fin n) (fin n)))) = games_played)) :=
sorry

end part_a_part_b_l144_144711


namespace assignment_of_tasks_l144_144825

theorem assignment_of_tasks :
  ∃ (f : Fin 3 → Finset (Fin 5)),
    (∀ i : Fin 3, (f i).Nonempty) ∧
    (Finset.univ.card : ℕ) = 150 := by
  sorry

end assignment_of_tasks_l144_144825


namespace tan_angle_between_median_angle_bisector_l144_144460

-- Define the isosceles triangle and the given condition
def isosceles_triangle (ABC : Triangle) : Prop :=
  (ABC.AB = ABC.CB) ∧ (tan(ABC.base_angle) = 3/4)

-- Define the question as a theorem statement to prove
theorem tan_angle_between_median_angle_bisector
  (ABC : Triangle) (h_iso : isosceles_triangle ABC) :
  ∃ γ : ℝ, tan(γ) = 1/13 :=
sorry

end tan_angle_between_median_angle_bisector_l144_144460


namespace verify_hyperbola_equation_l144_144447

noncomputable def hyperbola_equation_proof : Prop :=
  let center := (0, 0)
  let foci_on_y_axis := true
  let eccentricity := Real.sqrt 2
  let tangent_vertex := (0, -1)
  (vertex : (0, -1) ∧ (eccentricity = Real.sqrt 2) ∧ ((tangent_vertex * tangent_vertex) - 
                   ((1,0)* (1,0))) = 1 = 
                    hyperbola (y^2 - x^2 = 1).

theorem verify_hyperbola_equation : hyperbola_equation_proof :=
sorry

end verify_hyperbola_equation_l144_144447


namespace point_not_on_graph_l144_144300

noncomputable def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  (m-2) * x^2 + (m+3) * x + (m+2)

def point_on_graph (m p : ℝ) : Prop :=
  quadratic_function m p = -p^2

theorem point_not_on_graph (p : ℝ) : ¬point_on_graph 3 p :=
by
  simp only [quadratic_function, point_on_graph]
  intro h
  have h' : 2 * p^2 + 6 * p + 5 = 0 := by
    rw [quadratic_function]
    simp at h
    linarith
  let Δ := 6^2 - 4 * 2 * 5
  have : Δ < 0 := by
    norm_num
  exact absurd h' this

end point_not_on_graph_l144_144300


namespace shorter_silo_radius_l144_144484

theorem shorter_silo_radius
  (h : ℝ) -- height of the shorter silo
  (r_taller : ℝ := 15) -- radius of the taller silo
  (h_taller : ℝ := 4 * h) -- height of the taller silo
  (V_shorter : ℝ := π * (r := sqrt(900)) ^ 2 * h) -- volume of the shorter silo
  (V_taller : ℝ := π * (r_taller) ^ 2 * h_taller) -- volume of the taller silo
  (h_ne_zero : h ≠ 0) -- h is not zero
  (V_eq : V_shorter = V_taller) -- volumes are equal
  : r_shorter = 30 := 
begin
  sorry
end

end shorter_silo_radius_l144_144484


namespace disk_stack_count_l144_144381

theorem disk_stack_count (n : ℕ) (hn : 0 < n) :
  ∑ S in (finset.powerset (finset.range n.succ \ {0})), 
  ∏ x in S, x⁻¹ * n!.succ = (n + 1)! :=
sorry

end disk_stack_count_l144_144381


namespace triangle_sides_range_x_l144_144654

noncomputable def range_of_x (x : ℝ) : Prop :=
  √2 < x ∧ x < 2

theorem triangle_sides_range_x 
  (a b c : ℝ) (A B C : ℝ)
  (hC : C = π/4) -- C is 45 degrees
  (hc : c = √2) -- given c = √2
  (hA : (π/4) < A ∧ A < (2*π/3)) -- angle A is between 45 and 120 degrees
  : range_of_x a := 
sorry

end triangle_sides_range_x_l144_144654


namespace man_double_son_age_in_two_years_l144_144909

theorem man_double_son_age_in_two_years (S M Y : ℕ) (h1 : S = 14) (h2 : M = S + 16) (h3 : Y = 2) : 
  M + Y = 2 * (S + Y) :=
by
  sorry

-- Explanation:
-- h1 establishes the son's current age.
-- h2 establishes the man's current age in relation to the son's age.
-- h3 gives the solution Y = 2 years.
-- We need to prove that M + Y = 2 * (S + Y).

end man_double_son_age_in_two_years_l144_144909


namespace rohan_age_proof_l144_144890

theorem rohan_age_proof : ∃ R : ℤ, R + 15 = 4 * (R - 15) ∧ R = 25 :=
by {
  use 25,
  split,
  { -- Condition: Rohan's age equation
    linarith },
  { -- The answer
    refl }
}

end rohan_age_proof_l144_144890


namespace odd_cycle_through_vertex_l144_144130

variables (G : Type) [graph G] [simple G] [undirected G] [biconnected G] [¬ bipartite G]

theorem odd_cycle_through_vertex (v : vertex G) : ∃ (C : cycle G), simple C ∧ length(C) % 2 = 1 ∧ v ∈ C :=
sorry

end odd_cycle_through_vertex_l144_144130


namespace books_more_than_students_l144_144214

theorem books_more_than_students :
  ∀ (students_per_classroom books_per_student number_of_classrooms : ℕ),
    students_per_classroom = 18 →
    books_per_student = 3 →
    number_of_classrooms = 5 →
    (books_per_student * (students_per_classroom * number_of_classrooms)) -
    (students_per_classroom * number_of_classrooms) = 180 :=
by
  intros students_per_classroom books_per_student number_of_classrooms
  intros h1 h2 h3
  rw [h1, h2, h3]
  have total_students := 18 * 5
  have total_books := 3 * total_students
  have difference := total_books - total_students
  show difference = 180
  sorry

end books_more_than_students_l144_144214


namespace PQRS_is_parallelogram_l144_144051

variable {A B C D P Q R S I O : Type*}

-- Definitions of points and cyclic quadrilateral with given properties
variable [Circumcenter O ABCD]
variable [Inside O ABCD]
variable [NotOnDiagonal O AC]
variable [Intersection I (Diagonals ABCD)]
variable [Intersection P Q (Circumcircle AOI) AD AB]
variable [Intersection R S (Circumcircle COI) CB CD]

theorem PQRS_is_parallelogram :
  Parallelogram P Q R S := 
sorry

end PQRS_is_parallelogram_l144_144051


namespace g_neither_even_nor_odd_l144_144986

def g (x : ℝ) : ℝ := 5^(x^2 - 5*x + 6) - Real.sin x

theorem g_neither_even_nor_odd : ¬ (∀ x, g (-x) = g x) ∧ ¬ (∀ x, g (-x) = -g x) :=
by
  have h1 : ∀ x, g (-x) = 5^(x^2 + 5*x + 6) + Real.sin x := 
    fun x => by simp [g, pow_add, pow_bit0, pow_one, Real.sin_neg]
  have h2 : ∀ x, g x = 5^(x^2 - 5*x + 6) - Real.sin x := 
    fun x => rfl
  exact ⟨
    fun h => by
      specialize h 1
      simp [h1 1, h2 1]
    , fun h => by
      specialize h 1
      simp [h1 1, h2 1]⟩

end g_neither_even_nor_odd_l144_144986


namespace arrangement_of_programs_l144_144489

theorem arrangement_of_programs : 
  let total_singing := 4
  let total_skit := 2
  let required_distance := 3
  total_singing >= required_distance + 1 ∧
  (total_singing + total_skit = 6) ∧
  (1 <= required_distance) ∧ 
  (required_distance <= total_singing - 1)
  ∃ arrangements : ℕ, arrangements = 96  := 
by
  sorry

end arrangement_of_programs_l144_144489


namespace select_team_of_5_l144_144014

def boys : ℕ := 7
def girls : ℕ := 9
def total_students : ℕ := boys + girls

theorem select_team_of_5 (n : ℕ := total_students) (k : ℕ := 5) :
  (Nat.choose n k) = 4368 :=
by
  sorry

end select_team_of_5_l144_144014


namespace train_pass_time_l144_144173

theorem train_pass_time (length : ℝ) (speed_kmh : ℝ) (conversion_factor : ℝ) (speed_ms : ℝ) :
  length = 630 → 
  speed_kmh = 63 → 
  conversion_factor = 1000 / 3600 → 
  speed_ms = speed_kmh * conversion_factor → 
  length / speed_ms = 36 :=
by
  intros h_length h_speed h_conversion h_speed_ms
  rw [h_length, h_speed, h_conversion, h_speed_ms]
  sorry

end train_pass_time_l144_144173


namespace gcd_of_factorials_l144_144856

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definitions for the two factorial terms involved
def term1 : ℕ := factorial 7
def term2 : ℕ := factorial 10 / factorial 4

-- Definition for computing the GCD
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- The statement to prove
theorem gcd_of_factorials : gcd term1 term2 = 2520 :=
by {
  -- Here, the proof would go.
  sorry
}

end gcd_of_factorials_l144_144856


namespace non_factorial_tails_below_1500_l144_144978

open Nat

def trailing_zeroes (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125)

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ (m : ℕ), trailing_zeroes m = n

theorem non_factorial_tails_below_1500 : 
  {n : ℕ | ¬is_factorial_tail n ∧ n < 1500}.card = 286 := 
sorry

end non_factorial_tails_below_1500_l144_144978


namespace nine_digit_numbers_divisible_by_eleven_l144_144308

theorem nine_digit_numbers_divisible_by_eleven :
  ∃ (n : ℕ), n = 31680 ∧
    ∃ (num : ℕ), num < 10^9 ∧ num ≥ 10^8 ∧
      (∀ d : ℕ, 1 ≤ d ∧ d ≤ 9 → ∃ i : ℕ, i ≤ 8 ∧ (num / 10^i) % 10 = d) ∧
      (num % 11 = 0) := 
sorry

end nine_digit_numbers_divisible_by_eleven_l144_144308


namespace find_unknown_rate_l144_144923

-- Definitions for the costs and total price
def cost3 (rate : ℕ) (count : ℕ) : ℕ := rate * count
def cost2 (rate : ℕ) (count : ℕ) : ℕ := rate * count
def total_cost (cost1 cost2 cost3 : ℕ) : ℕ := cost1 + cost2 + cost3
def total_towels (qty1 qty2 qty3 : ℕ) : ℕ := qty1 + qty2 + qty3

-- Problem conditions
def rate1 := 100
def qty1 := 3
def rate2 := 150
def qty2 := 5
def unknown_rate := sorry  -- This is the unknown rate we need to prove.
def qty3 := 2
def avg := 145
def total_price := avg * total_towels qty1 qty2 qty3

-- The Lean statement to prove the unknown rate
theorem find_unknown_rate : unknown_rate = 200 := by
  -- Express total costs and solve
  let cost1 := cost3 rate1 qty1
  let cost2 := cost3 rate2 qty2
  let total_cost := cost1 + cost2 + (cost2 unknown_rate qty3)
  let calculated_total_cost := 1450
  have eq1 := total_cost = calculated_total_cost
  sorry

end find_unknown_rate_l144_144923


namespace Morse_code_distinct_symbols_l144_144354

theorem Morse_code_distinct_symbols (n : ℕ) (h : n ∈ {1, 2, 3, 4, 5}) : 
  (∑ i in {1, 2, 3, 4, 5}, 2^i) = 62 :=
by sorry

end Morse_code_distinct_symbols_l144_144354


namespace green_chips_count_l144_144473

-- Definitions of given conditions
def total_chips (total : ℕ) : Prop :=
  ∃ blue_chips white_percentage, blue_chips = 3 ∧ (blue_chips : ℚ) / total = 10 / 100 ∧ white_percentage = 50 / 100 ∧
  let green_percentage := 1 - (10 / 100 + white_percentage) in
  green_percentage * total = 12

-- Proposition to prove the number of green chips equals 12
theorem green_chips_count (total : ℕ) (h : total_chips total) : ∃ green_chips, green_chips = 12 := 
by 
  sorry

end green_chips_count_l144_144473


namespace colonization_combinations_l144_144330

noncomputable def number_of_combinations : ℕ :=
  let earth_like_bound := 7
  let mars_like_bound := 8
  let total_units := 21
  Finset.card ((Finset.range (earth_like_bound + 1)).filter (λ a, 
    (3 * a ≤ total_units ∧ (total_units - 3 * a) ≤ mars_like_bound)))

theorem colonization_combinations : number_of_combinations = 981 := by
  sorry

end colonization_combinations_l144_144330


namespace average_speed_including_stoppages_l144_144233

theorem average_speed_including_stoppages : 
  ∀ (speed_without_stops : ℕ) (stop_duration_per_hour_min : ℕ)
  (stop_duration_per_hour_hr : ℝ) (effective_travel_time_per_hour_hr : ℝ) (average_speed_with_stops : ℝ), 
  speed_without_stops = 75 → 
  stop_duration_per_hour_min = 28 →
  stop_duration_per_hour_hr = stop_duration_per_hour_min / 60 →
  effective_travel_time_per_hour_hr = 1 - stop_duration_per_hour_hr →
  average_speed_with_stops = (speed_without_stops * effective_travel_time_per_hour_hr) →
  average_speed_with_stops = 40 := 
by 
  intros speed_without_stops stop_duration_per_hour_min stop_duration_per_hour_hr effective_travel_time_per_hour_hr average_speed_with_stops
  intro h_speed_without_stops 
  intro h_stop_duration_per_hour_min 
  intro h_conv_stop_duration_per_hour_hr
  intro h_effective_travel_time_per_hour
  intro h_avg_speed_calculation
  rw [h_speed_without_stops, h_stop_duration_per_hour_min, h_conv_stop_duration_per_hour_hr, h_effective_travel_time_per_hour, h_avg_speed_calculation]
  sorry

end average_speed_including_stoppages_l144_144233


namespace min_value_f_x_sum_of_squares_l144_144653

variable (a b c : ℝ)

theorem min_value_f_x (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) 
  (h_min : ∀ x, |x - a| + |x + b| + c ≥ 1 := by sorry) : 
  a + b + c = 1 :=
sorry

theorem sum_of_squares (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) 
  (h_sum : a + b + c = 1) : 
  a^2 + b^2 + c^2 ≥ 1 / 3 :=
begin
  sorry
end

end min_value_f_x_sum_of_squares_l144_144653


namespace num_possible_values_a_l144_144044

theorem num_possible_values_a (a : ℕ) :
  (9 ∣ a) ∧ (a ∣ 18) ∧ (0 < a) → ∃ n : ℕ, n = 2 :=
by
  sorry

end num_possible_values_a_l144_144044


namespace cosine_sine_inequality_l144_144352

-- Definitions based on conditions
variable {A : ℝ} {B : ℝ} {C : ℝ}
variable {c : ℝ} {R : ℝ} {AC : ℝ} {AB : ℝ}
variable (cos : ℝ → ℝ)
variable (sin : ℝ → ℝ)

-- Conditions
variables (triangle : ∀ (A B C : ℝ), Type)
variables (acute_angled_triangle : triangle A B C → Prop)
variables (circumradius : triangle A B C → ℝ)

noncomputable def AC := 1
variable (AC_eq_1 : AC = 1)
variable (AB_eq_c : AB = c)
variable (R_le_one : R ≤ 1)

-- The proof statement
theorem cosine_sine_inequality
  (h_triangle : triangle A B C)
  (h_acute : acute_angled_triangle h_triangle)
  (h_AC : AC_eq_1)
  (h_AB : AB_eq_c)
  (h_R : R_le_one)
  : cos A < c ∧ c ≤ cos A + sqrt 3 * sin A := by
  sorry

end cosine_sine_inequality_l144_144352


namespace infinite_process_implies_regular_l144_144138

-- Define our scenario: a convex semiregular octagon where each interior angle is equal.
structure SemiregularOctagon (AB BC CD DE EF FG GH HA : ℝ) : Prop :=
  (convex : True)
  (equal_interior_angles : True)
  (every_other_equal : AB = CD ∧ CD = EF ∧ EF = GH ∧ BC = DE ∧ DE = FG ∧ FG = HA ∧ b < a)

noncomputable def is_regular (AB BC : ℝ) : Prop :=
  AB = BC

theorem infinite_process_implies_regular (a b : ℝ) (h1: b < a):
  (∀ n : ℕ, is_regular a b) → is_regular a b :=
sorry

end infinite_process_implies_regular_l144_144138


namespace students_prefer_windows_to_mac_l144_144891

-- Define the conditions
def total_students : ℕ := 210
def students_prefer_mac : ℕ := 60
def students_equally_prefer_both : ℕ := 20
def students_no_preference : ℕ := 90

-- The proof problem
theorem students_prefer_windows_to_mac :
  total_students - students_prefer_mac - students_equally_prefer_both - students_no_preference = 40 :=
by sorry

end students_prefer_windows_to_mac_l144_144891


namespace smaller_angle_at_8_pm_l144_144188

-- Definitions of conditions
def degrees_per_hour : ℝ := 360 / 12
def minute_hand_position (h : ℕ) (m : ℕ) : ℝ := 0  -- At the top of the clock
def hour_hand_position (h : ℕ) (m : ℕ) : ℝ := h * degrees_per_hour
def smaller_angle (a b : ℝ) : ℝ := min (abs (a - b)) (360 - abs (a - b))

-- The main statement to prove
theorem smaller_angle_at_8_pm : smaller_angle (minute_hand_position 20 0) (hour_hand_position 20 0) = 120 :=
by
  sorry

end smaller_angle_at_8_pm_l144_144188


namespace triangle_angle_ratio_l144_144766

theorem triangle_angle_ratio 
  (A B C P : Type*)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  [MetricSpace P]
  (is_equilateral : is_equilateral_triangle A B C)
  (angle_ratio : angle APB / angle BPC / angle CPA = 5 / 6 / 7) 
  : (triangle_angle_ratio A P B C = (2 / 3 / 4)) :=
sorry

end triangle_angle_ratio_l144_144766


namespace max_quadratic_value_l144_144366

def quadratic_function (f : ℕ → ℤ) : Prop :=
  ∃ (a b c : ℤ), ∀ x : ℕ, f x = a * x^2 + b * x + c

theorem max_quadratic_value (f : ℕ → ℤ) (n : ℕ)
  (h1 : f(n) = 6) 
  (h2 : f(n + 1) = 14) 
  (h3 : f(n + 2) = 14) 
  (hf : quadratic_function f) : 
  ∃ m : ℤ, m = 15 ∧ ∀ x : ℕ, f(x) ≤ m :=
begin
  sorry
end

end max_quadratic_value_l144_144366


namespace multiply_polynomials_l144_144200

theorem multiply_polynomials (x : ℝ) : 2 * x * (5 * x ^ 2) = 10 * x ^ 3 := by
  sorry

end multiply_polynomials_l144_144200


namespace prob_divisible_by_5_l144_144089

theorem prob_divisible_by_5 (M: ℕ) (h1: 100 ≤ M ∧ M < 1000) (h2: M % 10 = 5): 
  (∃ (k: ℕ), M = 5 * k) :=
by
  sorry

end prob_divisible_by_5_l144_144089


namespace mixed_tea_sale_price_l144_144570

noncomputable def sale_price_of_mixed_tea (weight1 weight2 weight3 price1 price2 price3 profit1 profit2 profit3 : ℝ) : ℝ :=
  let total_cost1 := weight1 * price1
  let total_cost2 := weight2 * price2
  let total_cost3 := weight3 * price3
  let total_profit1 := profit1 * total_cost1
  let total_profit2 := profit2 * total_cost2
  let total_profit3 := profit3 * total_cost3
  let selling_price1 := total_cost1 + total_profit1
  let selling_price2 := total_cost2 + total_profit2
  let selling_price3 := total_cost3 + total_profit3
  let total_selling_price := selling_price1 + selling_price2 + selling_price3
  let total_weight := weight1 + weight2 + weight3
  total_selling_price / total_weight

theorem mixed_tea_sale_price :
  sale_price_of_mixed_tea 120 45 35 30 40 60 0.50 0.30 0.25 = 51.825 :=
by
  sorry

end mixed_tea_sale_price_l144_144570


namespace largest_square_test_plots_l144_144158

/-- 
  A fenced, rectangular field measures 30 meters by 45 meters. 
  An agricultural researcher has 1500 meters of fence that can be used for internal fencing to partition 
  the field into congruent, square test plots. 
  The entire field must be partitioned, and the sides of the squares must be parallel to the edges of the field. 
  What is the largest number of square test plots into which the field can be partitioned using all or some of the 1500 meters of fence?
 -/
theorem largest_square_test_plots
  (field_length : ℕ := 30)
  (field_width : ℕ := 45)
  (total_fence_length : ℕ := 1500):
  ∃ (n : ℕ), n = 576 := 
sorry

end largest_square_test_plots_l144_144158


namespace uncle_welly_roses_l144_144831

theorem uncle_welly_roses :
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  roses_two_days_ago + roses_yesterday + roses_today = 220 :=
by
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  show roses_two_days_ago + roses_yesterday + roses_today = 220
  sorry

end uncle_welly_roses_l144_144831


namespace perimeter_of_convex_quad_l144_144902

theorem perimeter_of_convex_quad (ABCD : ConvexQuadrilateral) (P : Point)
  (h_area : area ABCD = 2500)
  (h_PA : distance P A = 30)
  (h_PB : distance P B = 40)
  (h_PC : distance P C = 35)
  (h_PD : distance P D = 50) :
  perimeter ABCD = 169 + 5 * real.sqrt 113 := 
sorry

end perimeter_of_convex_quad_l144_144902


namespace height_of_triangle_l144_144565

-- Define the dimensions of the rectangle
variable (l w : ℝ)

-- Assume the base of the triangle is equal to the length of the rectangle
-- We need to prove that the height of the triangle h = 2w

theorem height_of_triangle (h : ℝ) (hl_eq_length : l > 0) (hw_eq_width : w > 0) :
  (l * w) = (1 / 2) * l * h → h = 2 * w :=
by
  sorry

end height_of_triangle_l144_144565


namespace sum_of_distances_l144_144661

open Real

-- Define the points and parabola properties
structure Point where
  x : ℝ
  y : ℝ

-- Focus of the parabola y^2 = 12x
def F : Point := { x := 3, y := 0 }

-- Midpoint P(2, 1)
def P : Point := { x := 2, y := 1 }

-- Definition of parabola:y^2 = 12x
def parabola (p : Point) : Prop := p.y ^ 2 = 12 * p.x

-- Proposition of the problem
theorem sum_of_distances (A B : Point) :
  (parabola A ∧ parabola B ∧ P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2) →
  dist A F + dist B F = 10 :=
sorry

end sum_of_distances_l144_144661


namespace alice_password_probability_l144_144181

section AlicePassword

def is_even_digit (n : ℕ) : Prop := n ∈ {0, 2, 4, 6, 8}
def is_non_zero_digit (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
def is_vowel (c : Char) : Prop := c ∈ {'A', 'E', 'I', 'O', 'U'}
def is_letter (c : Char) : Prop := c.isAlpha

theorem alice_password_probability :
  let total_digits := 10 in
  let total_letters := 26 in
  let even_digits := 5 in
  let non_zero_digits := 9 in
  let vowel_letters := 5 in
  (even_digits/total_digits : ℚ) * (1 : ℚ) * (non_zero_digits/total_digits : ℚ) * (vowel_letters/total_letters : ℚ) = 9/104 :=
sorry

end AlicePassword

end alice_password_probability_l144_144181


namespace trains_clear_time_approx_l144_144485

-- Define the lengths of the trains
def length_train1 : ℕ := 235 -- in meters
def length_train2 : ℕ := 275 -- in meters

-- Define the speeds of the trains
def speed_train1_kmh : ℕ := 120 -- in km/h
def speed_train2_kmh : ℕ := 95 -- in km/h

-- Conversion factor from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℝ := speed_kmh * 1000 / 3600

-- Converted speeds of the trains in m/s
def speed_train1_ms : ℝ := kmh_to_ms speed_train1_kmh
def speed_train2_ms : ℝ := kmh_to_ms speed_train2_kmh

-- Total distance the trains need to cover to be clear of each other
def total_distance : ℕ := length_train1 + length_train2

-- Relative speed of the trains when moving in opposite directions
def relative_speed : ℝ := speed_train1_ms + speed_train2_ms

-- Time required for the trains to be completely clear of each other
noncomputable def time_clear : ℝ := total_distance / relative_speed

-- Lean 4 statement for the proof problem
theorem trains_clear_time_approx :
  time_clear ≈ 8.54 :=
by
  sorry

end trains_clear_time_approx_l144_144485


namespace number_of_male_students_l144_144559

variables (total_students sample_size female_sampled female_students male_students : ℕ)
variables (h_total : total_students = 1600)
variables (h_sample : sample_size = 200)
variables (h_female_sampled : female_sampled = 95)
variables (h_prob : (sample_size : ℚ) / total_students = (female_sampled : ℚ) / female_students)
variables (h_female_students : female_students = 760)

theorem number_of_male_students : male_students = total_students - female_students := by
  sorry

end number_of_male_students_l144_144559


namespace non_factorial_tails_below_1500_l144_144979

open Nat

def trailing_zeroes (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125)

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ (m : ℕ), trailing_zeroes m = n

theorem non_factorial_tails_below_1500 : 
  {n : ℕ | ¬is_factorial_tail n ∧ n < 1500}.card = 286 := 
sorry

end non_factorial_tails_below_1500_l144_144979


namespace prob_divisible_by_5_l144_144090

theorem prob_divisible_by_5 (M: ℕ) (h1: 100 ≤ M ∧ M < 1000) (h2: M % 10 = 5): 
  (∃ (k: ℕ), M = 5 * k) :=
by
  sorry

end prob_divisible_by_5_l144_144090


namespace smallest_geometric_sequence_number_l144_144506

theorem smallest_geometric_sequence_number :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧
    (∀ d ∈ [((n / 100) % 10), ((n / 10) % 10), (n % 10)], d ∈ [1,2,3,4,5,6,7,8,9]) ∧
    (let digits := [((n / 100) % 10), ((n / 10) % 10), (n % 10)] in
       digits.nodup ∧
       ∃ r : ℕ, r > 1 ∧ digits = [digits.head!, digits.head! * r, digits.head! * r * r]) ∧
    n = 124 :=
begin
  sorry
end

end smallest_geometric_sequence_number_l144_144506


namespace gcd_factorials_l144_144847

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  -- Proof steps would go here
  sorry

end gcd_factorials_l144_144847


namespace solve_abs_eq_l144_144872

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 :=
  sorry

end solve_abs_eq_l144_144872


namespace max_revenue_l144_144538

variable (x y : ℝ)

-- Conditions
def ads_time_constraint := x + y ≤ 300
def ads_cost_constraint := 500 * x + 200 * y ≤ 90000
def revenue := 0.3 * x + 0.2 * y

-- Question: Prove that the maximum revenue is 70 million yuan
theorem max_revenue (h_time : ads_time_constraint (x := 100) (y := 200))
                    (h_cost : ads_cost_constraint (x := 100) (y := 200)) :
  revenue (x := 100) (y := 200) = 70 := 
sorry

end max_revenue_l144_144538


namespace inscribed_circle_area_ratio_l144_144958

theorem inscribed_circle_area_ratio
  (R : ℝ) -- Radius of the original circle
  (r : ℝ) -- Radius of the inscribed circle
  (h : R = 3 * r) -- Relationship between the radii based on geometry problem
  :
  (π * R^2) / (π * r^2) = 9 :=
by sorry

end inscribed_circle_area_ratio_l144_144958


namespace number_of_ways_to_sign_up_l144_144331

theorem number_of_ways_to_sign_up (S I : ℕ) (choices : I) (students : S) (condition1 : choices = 4) (condition2 : students = 3) : 
  S * I = 4 ^ 3 :=
by
  sorry

end number_of_ways_to_sign_up_l144_144331


namespace georgie_initial_avocados_l144_144257

-- Define the conditions
def avocados_needed_per_serving := 3
def servings_made := 3
def avocados_bought_by_sister := 4
def total_avocados_needed := avocados_needed_per_serving * servings_made

-- The statement to prove
theorem georgie_initial_avocados : (total_avocados_needed - avocados_bought_by_sister) = 5 :=
sorry

end georgie_initial_avocados_l144_144257


namespace stratified_sampling_number_of_products_drawn_l144_144156

theorem stratified_sampling_number_of_products_drawn (T S W X : ℕ) 
  (h1 : T = 1024) (h2 : S = 64) (h3 : W = 128) :
  X = S * (W / T) → X = 8 :=
by
  sorry

end stratified_sampling_number_of_products_drawn_l144_144156


namespace sum_distances_parabola_l144_144659

theorem sum_distances_parabola (P A B F : ℝ × ℝ) :
  let parabola := λ x y, y^2 = 12 * x,
      midpoint := λ P A B, P = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)),
      distance := λ P F, (Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)),
      focus := (3, 0) in
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ midpoint P A B ∧ F = focus →
  distance A F + distance B F = 10 :=
  by sorry

end sum_distances_parabola_l144_144659


namespace total_cost_l144_144080

theorem total_cost :
  ∀ (cost_caramel cost_candy cost_cotton : ℕ),
  (cost_candy = 2 * cost_caramel) →
  (cost_cotton = (1 / 2) * (4 * cost_candy)) →
  (cost_caramel = 3) →
  6 * cost_candy + 3 * cost_caramel + cost_cotton = 57 :=
begin
  intro cost_caramel, intro cost_candy, intro cost_cotton,
  assume h1 : cost_candy = 2 * cost_caramel,
  assume h2 : cost_cotton = (1 / 2) * (4 * cost_candy),
  assume h3 : cost_caramel = 3,
  sorry
end

end total_cost_l144_144080


namespace triangle_angles_l144_144273

theorem triangle_angles (A B C : ℝ) 
  (h1 : A + B + C = 180)
  (h2 : B = 120)
  (h3 : (∃D, A = D ∧ (A + A + C = 180 ∨ A + C + C = 180)) ∨ (∃E, C = E ∧ (B + 15 + 45 = 180 ∨ B + 15 + 15 = 180))) :
  (A = 40 ∧ C = 20) ∨ (A = 45 ∧ C = 15) :=
sorry

end triangle_angles_l144_144273


namespace handshake_problem_l144_144584

theorem handshake_problem (x : ℕ) (hx : (x * (x - 1)) / 2 = 55) : x = 11 := 
sorry

end handshake_problem_l144_144584


namespace mutually_exclusive_event_3_l144_144774

-- Definitions based on the conditions.
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Events based on problem conditions
def event_1 (a b : ℕ) : Prop := is_even a ∧ is_odd b ∨ is_odd a ∧ is_even b
def event_2 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ is_odd a ∧ is_odd b
def event_3 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ is_even a ∧ is_even b
def event_4 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ (is_even a ∨ is_even b)

-- Problem: Proving that event_3 is mutually exclusive with other events.
theorem mutually_exclusive_event_3 :
  ∀ (a b : ℕ), (event_3 a b) → ¬ (event_1 a b ∨ event_2 a b ∨ event_4 a b) :=
by
  sorry

end mutually_exclusive_event_3_l144_144774


namespace total_weight_puffy_muffy_l144_144192

def scruffy_weight : ℕ := 12
def muffy_weight : ℕ := scruffy_weight - 3
def puffy_weight : ℕ := muffy_weight + 5

theorem total_weight_puffy_muffy : puffy_weight + muffy_weight = 23 := 
by
  sorry

end total_weight_puffy_muffy_l144_144192


namespace danny_chemistry_marks_l144_144974

theorem danny_chemistry_marks 
  (eng marks_physics marks_biology math : ℕ)
  (average: ℕ) 
  (total_marks: ℕ) 
  (chemistry: ℕ) 
  (h_eng : eng = 76) 
  (h_math : math = 65) 
  (h_phys : marks_physics = 82) 
  (h_bio : marks_biology = 75) 
  (h_avg : average = 73) 
  (h_total : total_marks = average * 5) : 
  chemistry = total_marks - (eng + math + marks_physics + marks_biology) :=
by
  sorry

end danny_chemistry_marks_l144_144974


namespace prime_pairs_divisibility_l144_144620

theorem prime_pairs_divisibility (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (p * q) ∣ (p ^ p + q ^ q + 1) ↔ (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2) :=
by
  sorry

end prime_pairs_divisibility_l144_144620


namespace total_pawns_left_l144_144376

  -- Definitions of initial conditions
  def initial_pawns_in_chess : Nat := 8
  def kennedy_pawns_lost : Nat := 4
  def riley_pawns_lost : Nat := 1

  -- Theorem statement to prove the total number of pawns left
  theorem total_pawns_left : (initial_pawns_in_chess - kennedy_pawns_lost) + (initial_pawns_in_chess - riley_pawns_lost) = 11 := by
    sorry
  
end total_pawns_left_l144_144376


namespace arithmetic_mean_of_remaining_numbers_l144_144784

theorem arithmetic_mean_of_remaining_numbers
  (original_mean : ℕ → ℝ)
  (original_set_length : ℕ)
  (removed_numbers : List ℕ)
  (discarded_sum : ℕ)
  (remaining_set_length : ℕ)
  (new_mean : ℝ) :
  original_mean = 42 → 
  original_set_length = 60 → 
  removed_numbers = [40, 50, 60] → 
  discarded_sum = 40 + 50 + 60 → 
  remaining_set_length = original_set_length - removed_numbers.length → 
  new_mean = (original_mean * original_set_length - discarded_sum) / remaining_set_length → 
  new_mean = 41.6 := 
by 
  sorry

end arithmetic_mean_of_remaining_numbers_l144_144784


namespace axis_of_symmetry_is_angle_bisector_l144_144437

-- Define the angle and its bisector
structure Angle (V : Type) [OrderedField V] :=
  (ray1 : V → V → Prop)  -- defining a ray as a binary relation
  (ray2 : V → V → Prop)
  (vertex : V)

-- Define the angle bisector as a line
def is_angle_bisector {V : Type} [OrderedField V] (a : Angle V) (l : V → V → Prop) : Prop :=
  ∀ x y, l x y → a.ray1 x y → a.ray2 x y

-- Statement: Prove that the axis of symmetry of an angle is the line on which the angle bisector lies
theorem axis_of_symmetry_is_angle_bisector {V : Type} [OrderedField V] (a : Angle V) (l : V → V → Prop) : 
  is_angle_bisector a l → (∀ x y, l x y ↔ (a.ray1 x y ∧ a.ray2 x y)) :=
by
  sorry

end axis_of_symmetry_is_angle_bisector_l144_144437


namespace sequence_u5_eq_27_l144_144043

theorem sequence_u5_eq_27 (u : ℕ → ℝ) 
  (h_recurrence : ∀ n, u (n + 2) = 3 * u (n + 1) - 2 * u n)
  (h_u3 : u 3 = 15)
  (h_u6 : u 6 = 43) :
  u 5 = 27 :=
  sorry

end sequence_u5_eq_27_l144_144043


namespace problem_statement_l144_144297

def f (x : ℝ) (a : ℝ) := 2 * Real.log x - a * x + a

theorem problem_statement (x1 x2 : ℝ) (h₀ : 0 < x1) (h₁ : x1 < x2) :
  (∀ x, f x 2 ≤ 0) →
  (f x2 2 - f x1 2) / (x2 - x1) < 2 * (1 / x1 - 1) := by
  sorry

end problem_statement_l144_144297


namespace arithmetic_sequence_sum_l144_144386

/-!
    Let \( \{a_n\} \) be an arithmetic sequence with the sum of the first \( n \) terms denoted as \( S_n \).
    If \( S_{17} = \frac{17}{2} \), then \( a_3 + a_{15} = 1 \).
-/

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n m : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n / 2) * (a 0 + a (n - 1))

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (h_arith : arithmetic_sequence a) 
  (h_sum : sum_of_first_n_terms a S) (h_S17 : S 17 = 17 / 2) : a 2 + a 14 = 1 :=
sorry

end arithmetic_sequence_sum_l144_144386


namespace slope_of_tangent_at_point_l144_144085

theorem slope_of_tangent_at_point : 
  let y := λ x : ℝ, -2 * x^2 + 1 in
  let point := (0 : ℝ, 1 : ℝ) in
  let y' := λ x : ℝ, -4 * x in
  y' (point.fst) = 0 :=
sorry

end slope_of_tangent_at_point_l144_144085


namespace largest_triangle_perimeter_l144_144175

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem largest_triangle_perimeter : 
  ∃ (x : ℕ), x ≤ 14 ∧ 2 ≤ x ∧ is_valid_triangle 7 8 x ∧ (7 + 8 + x = 29) :=
sorry

end largest_triangle_perimeter_l144_144175


namespace picture_area_l144_144166

theorem picture_area (total_width total_length margin : ℝ) (h1 : total_width = 8.5) (h2 : total_length = 10) (h3 : margin = 1.5) :
  (total_length - 2 * margin) * (total_width - 2 * margin) = 38.5 :=
by
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end picture_area_l144_144166


namespace ellipse_eq_l144_144287

theorem ellipse_eq (h : ∃ a b c : ℝ, a = 3 ∧ c = real.sqrt 5 ∧ b = real.sqrt (a^2 - c^2)) : 
  ∀ x y: ℝ, (x^2)/9 + (y^2)/4 = 1 :=
begin
  intro x,
  intro y,
  sorry
end

end ellipse_eq_l144_144287


namespace thirty_divides_p_squared_minus_one_iff_p_eq_five_l144_144332

theorem thirty_divides_p_squared_minus_one_iff_p_eq_five (p : ℕ) (hp : Nat.Prime p) (h_ge : p ≥ 5) : 30 ∣ (p^2 - 1) ↔ p = 5 :=
by
  sorry

end thirty_divides_p_squared_minus_one_iff_p_eq_five_l144_144332


namespace frigate_catches_smuggler_at_five_l144_144556

noncomputable def time_to_catch : ℝ :=
  2 + (12 / 4) -- Initial leading distance / Relative speed before storm
  
theorem frigate_catches_smuggler_at_five 
  (initial_distance : ℝ)
  (frigate_speed_before_storm : ℝ)
  (smuggler_speed_before_storm : ℝ)
  (time_before_storm : ℝ)
  (frigate_speed_after_storm : ℝ)
  (smuggler_speed_after_storm : ℝ) :
  initial_distance = 12 →
  frigate_speed_before_storm = 14 →
  smuggler_speed_before_storm = 10 →
  time_before_storm = 3 →
  frigate_speed_after_storm = 12 →
  smuggler_speed_after_storm = 9 →
  time_to_catch = 5 :=
by
{
  sorry
}

end frigate_catches_smuggler_at_five_l144_144556


namespace string_length_l144_144552

-- Definitions
def circumference := 6 -- in feet
def num_loops := 6
def height := 18 -- in feet

-- Pythagorean theorem applied in Lean
theorem string_length : 
  let vertical_distance_per_loop := height / num_loops
  let base := circumference
  let height_per_loop := vertical_distance_per_loop
  let hypotenuse := Real.sqrt (base^2 + height_per_loop^2)
  (num_loops * hypotenuse) = 18 * Real.sqrt 5 :=
by
  let vertical_distance_per_loop := height / num_loops
  let base := circumference
  let height_per_loop := vertical_distance_per_loop
  let hypotenuse := Real.sqrt (base^2 + height_per_loop^2)
  have hypotenuse_eq : hypotenuse = 3 * Real.sqrt 5 := 
    by
      calc
        hypotenuse = Real.sqrt (6^2 + 3^2) : by sorry
                    ... = 3 * Real.sqrt 5 : by sorry
  calc
    num_loops * hypotenuse = 6 * (3 * Real.sqrt 5) : by rw ←hypotenuse_eq
                        ... = 18 * Real.sqrt 5 : by linarith
  sorry -- placeholder for the detailed proof

end string_length_l144_144552


namespace find_f_f_f_1_l144_144066

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then log 2 x else 3 ^ x

theorem find_f_f_f_1 : f (f (f 1)) = 0 := 
by
  -- f(1) = log(2, 1) = 0
  have h1 : f 1 = 0 := by
    simp [f]
    rw [if_pos (show 1 > 0, by norm_num)]
    exact Real.logb_eq_zero_of_one 2

  -- f(f(1)) = f(0) = 3^0 = 1
  have h2 : f (f 1) = 1 := by 
    rw [h1]
    simp [f]
    rw [if_neg (show 0 > 0 → false, by norm_num)]

  -- f(f(f(1))) = f(1) = 0
  rw [h2]
  exact h1

end find_f_f_f_1_l144_144066


namespace geometric_to_arithmetic_sequence_l144_144264

theorem geometric_to_arithmetic_sequence {a : ℕ → ℝ} (q : ℝ) 
    (h_gt0 : 0 < q) (h_pos : ∀ n, 0 < a n)
    (h_geom_seq : ∀ n, a (n + 1) = a n * q)
    (h_arith_seq : 2 * (1 / 2 * a 3) = a 1 + 2 * a 2) :
    a 10 / a 8 = 3 + 2 * Real.sqrt 2 := 
by
  sorry

end geometric_to_arithmetic_sequence_l144_144264


namespace max_profit_at_85_l144_144896

noncomputable def fixed_cost : ℝ := 1
noncomputable def selling_price : ℝ := 6

def variable_cost (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 80 then
    (1/30) * x^2 + 2 * x
  else if x ≥ 80 then
    (103/17) * x + 425/x - 135
  else 0

def profit (x : ℝ) : ℝ :=
  selling_price * x - fixed_cost - variable_cost x

theorem max_profit_at_85 : (profit 85 = 25) ∧ (∀ x : ℝ, 0 < x → 
  x ≠ 85 → profit x < 25) :=
by
  sorry

end max_profit_at_85_l144_144896


namespace bear_small_animal_weight_l144_144544

theorem bear_small_animal_weight :
  let total_weight_needed := 1200
  let berries_weight := 1/5 * total_weight_needed
  let insects_weight := 1/10 * total_weight_needed
  let acorns_weight := 2 * berries_weight
  let honey_weight := 3 * insects_weight
  let total_weight_gained := berries_weight + insects_weight + acorns_weight + honey_weight
  let remaining_weight := total_weight_needed - total_weight_gained
  remaining_weight = 0 -> 0 = 0 := by
  intros total_weight_needed berries_weight insects_weight acorns_weight honey_weight
         total_weight_gained remaining_weight h
  exact Eq.refl 0

end bear_small_animal_weight_l144_144544


namespace calc_expression_l144_144000

theorem calc_expression (x y z : ℚ) (h1 : x = 1 / 3) (h2 : y = 2 / 3) (h3 : z = x * y) :
  3 * x^2 * y^5 * z^3 = 768 / 1594323 :=
by
  sorry

end calc_expression_l144_144000


namespace pyramid_volume_relation_l144_144768

-- Assume setup conditions for the geometric objects.
variables {P A B C D Q : Type*}
variables {AD : ℝ} {PQ : ℝ}
variables [affine_space P A B C D Q] [metric_space P A B C D Q]
variables (P A B C D Q : affine_space P A B C D Q)

-- Conditions for quadrilateral pyramid with a parallelogram base
def is_parallelogram_base (ABCD : affine_space P A B C D) (AD: ℝ) : Prop :=
parallel AD PQ ∧ dist PQ AD = AD.length

def volume_triangular_pyramid (QPCD : affine_space P A B C D Q) : ℝ := sorry
def volume_quadrilateral_pyramid (PABCD : affine_space P A B C D) : ℝ := sorry

-- Final theorem statement in Lean 4
theorem pyramid_volume_relation (h1 : is_parallelogram_base ABCD AD) : 
  volume_triangular_pyramid QPCD = (1/2) * volume_quadrilateral_pyramid PABCD :=
sorry

end pyramid_volume_relation_l144_144768


namespace total_packs_sold_is_160_l144_144772

noncomputable def total_cookie_packs : ℚ :=
  let robyn_neighborhood_1 := 15
  let lucy_neighborhood_1 := 12.5
  let robyn_neighborhood_2 := 23
  let lucy_neighborhood_2 := 15.25
  let robyn_neighborhood_3 := 17.75
  let lucy_neighborhood_3 := 16.5
  let combined_first_park := 25
  let lucy_first_park := (25 : ℚ) * 3 / 7
  let robyn_first_park := (lucy_first_park * 4 / 3)
  let combined_second_park := 35
  let robyn_second_park := (35 : ℚ) * 4 / 9
  let lucy_second_park := (robyn_second_park * 5 / 4)
  let robyn_total := robyn_neighborhood_1 + robyn_neighborhood_2 + robyn_neighborhood_3 + robyn_first_park + robyn_second_park
  let lucy_total := lucy_neighborhood_1 + lucy_neighborhood_2 + lucy_neighborhood_3 + lucy_first_park + lucy_second_park
  robyn_total + lucy_total

theorem total_packs_sold_is_160 : total_cookie_packs ≈ 160 := by
  sorry

end total_packs_sold_is_160_l144_144772


namespace T1_T2_final_theorem_l144_144967

-- Define the postulates
def P1 (pib : Type) (maa : Type) :=
  ∀ (p : pib), ∃ (m : set maa), m.nonempty ∧ p ∈ m

def P2 (pib : Type) (maa : Type) :=
  ∀ (p1 p2 p3 : pib), p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
  ∃ (m : maa), m ∈ p1 ∧ m ∈ p2 ∧ m ∈ p3 ∧
  (∀ (m' : maa), (m' ∈ p1 ∧ m' ∈ p2 ∧ m' ∈ p3) → m' = m)

def P3 (pib : Type) (maa : Type) :=
  ∀ (m : maa), ∃ (p1 p2 p3 : pib), m ∈ p1 ∧ m ∈ p2 ∧ m ∈ p3 ∧
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

def P4 (pib : Type) := 
  fintype.card pib = 7

-- Define the theorems
theorem T1 (pib : Type) [fintype pib] (maa : Type) [fintype maa] 
  (hP1 : P1 pib maa) (hP2 : P2 pib maa) (hP3 : P3 pib maa) (hP4 : P4 pib) :
  fintype.card maa = 35 := 
sorry

theorem T2 (pib : Type) [fintype pib] (maa : Type) [fintype maa]
  (hP1 : P1 pib maa) (hP2 : P2 pib maa) (hP3 : P3 pib maa) (hP4 : P4 pib) :
  ∀ (p : pib), fintype.card {m : maa // m ∈ p} = 15 := 
sorry

-- Prove the summation of given theorems based on P1 to P4.
theorem final_theorem (pib : Type) [fintype pib] (maa : Type) [fintype maa]
  (hP1 : P1 pib maa) (hP2 : P2 pib maa) (hP3 : P3 pib maa) (hP4 : P4 pib) :
  (T1 pib maa hP1 hP2 hP3 hP4 ∧ T2 pib maa hP1 hP2 hP3 hP4) ∧ ¬T3 pib maa hP1 hP2 hP3 hP4 := 
sorry

end T1_T2_final_theorem_l144_144967


namespace find_a_l144_144450

theorem find_a 
(a : ℝ) : 
(ax + y + 1 = 0 ∧ 3x + (a-2)y + a^2 - 4 = 0) → a = 3 :=
by
  sorry

end find_a_l144_144450
