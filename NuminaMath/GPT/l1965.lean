import Mathlib

namespace expected_rice_yield_l1965_196595

theorem expected_rice_yield (x : ℝ) (y : ℝ) (h : y = 5 * x + 250) (hx : x = 80) : y = 650 :=
by
  sorry

end expected_rice_yield_l1965_196595


namespace height_of_shorter_tree_l1965_196543

theorem height_of_shorter_tree (H h : ℝ) (h_difference : H = h + 20) (ratio : h / H = 5 / 7) : h = 50 := 
by
  sorry

end height_of_shorter_tree_l1965_196543


namespace option_c_correct_l1965_196550

theorem option_c_correct (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by
  sorry

end option_c_correct_l1965_196550


namespace leak_empty_time_l1965_196506

theorem leak_empty_time :
  let A := (1:ℝ)/6
  let AL := A - L
  ∀ L: ℝ, (A - L = (1:ℝ)/8) → (1 / L = 24) :=
by
  intros A AL L h
  sorry

end leak_empty_time_l1965_196506


namespace four_digit_cubes_divisible_by_16_l1965_196563

theorem four_digit_cubes_divisible_by_16 (n : ℕ) : 
  1000 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 9999 ∧ (4 * n)^3 % 16 = 0 ↔ n = 4 ∨ n = 5 := 
sorry

end four_digit_cubes_divisible_by_16_l1965_196563


namespace largest_side_of_enclosure_l1965_196589

-- Definitions for the conditions
def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w
def area (l w : ℝ) : ℝ := l * w

theorem largest_side_of_enclosure (l w : ℝ) (h_fencing : perimeter l w = 240) (h_area : area l w = 12 * 240) : l = 86.83 ∨ w = 86.83 :=
by {
  sorry
}

end largest_side_of_enclosure_l1965_196589


namespace extremum_condition_l1965_196522

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

def has_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, 3 * a * x^2 + 1 = 0

theorem extremum_condition (a : ℝ) : has_extremum a ↔ a < 0 := 
  sorry

end extremum_condition_l1965_196522


namespace find_a_minus_b_l1965_196561

theorem find_a_minus_b (a b c d : ℤ) 
  (h1 : (a - b) + c - d = 19) 
  (h2 : a - b - c - d = 9) : 
  a - b = 14 :=
sorry

end find_a_minus_b_l1965_196561


namespace general_term_correct_l1965_196575

-- Define the sequence a_n
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 2^n

-- Define the general term formula for the sequence a_n
def general_term (a : ℕ → ℕ) : Prop :=
  ∀ n, a n = n * 2^(n - 1)

-- Theorem statement: the general term formula holds for the sequence a_n
theorem general_term_correct (a : ℕ → ℕ) (h_seq : seq a) : general_term a :=
by
  sorry

end general_term_correct_l1965_196575


namespace fraction_check_l1965_196577

variable (a b x y : ℝ)
noncomputable def is_fraction (expr : ℝ) : Prop :=
∃ n d : ℝ, d ≠ 0 ∧ expr = n / d ∧ ∃ var : ℝ, d = var

theorem fraction_check :
  is_fraction ((x + 3) / x) :=
sorry

end fraction_check_l1965_196577


namespace avg_b_c_weight_l1965_196571

theorem avg_b_c_weight (a b c : ℝ) (H1 : (a + b + c) / 3 = 45) (H2 : (a + b) / 2 = 40) (H3 : b = 39) : (b + c) / 2 = 47 :=
by
  sorry

end avg_b_c_weight_l1965_196571


namespace triangle_area_l1965_196507

theorem triangle_area :
  ∃ (A : ℝ),
  let a := 65
  let b := 60
  let c := 25
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  a = 65 ∧ b = 60 ∧ c = 25 ∧ s = 75 ∧  area = 750 :=
by
  let a := 65
  let b := 60
  let c := 25
  let s := (a + b + c) / 2
  use Real.sqrt (s * (s - a) * (s - b) * (s - c))
  -- We would prove the conditions and calculations here, but we skip the proof parts
  sorry

end triangle_area_l1965_196507


namespace simplify_expression_l1965_196572

theorem simplify_expression :
  8 * (15 / 4) * (-45 / 50) = - (12 / 25) :=
by
  sorry

end simplify_expression_l1965_196572


namespace complex_fraction_value_l1965_196504

theorem complex_fraction_value :
  (Complex.mk 1 2) * (Complex.mk 1 2) / Complex.mk 3 (-4) = -1 :=
by
  -- Here we would provide the proof, but as per instructions,
  -- we will insert sorry to skip it.
  sorry

end complex_fraction_value_l1965_196504


namespace license_plates_count_l1965_196570

theorem license_plates_count :
  let letters := 26
  let digits := 10
  let odd_digits := 5
  let even_digits := 5
  (letters^3) * digits * (odd_digits + even_digits) = 878800 := by
  sorry

end license_plates_count_l1965_196570


namespace sequence_value_l1965_196533

theorem sequence_value (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) : a 5 = 17 :=
by
  -- The proof is not required, so we add sorry to indicate that
  sorry

end sequence_value_l1965_196533


namespace problem1_problem2_l1965_196539

theorem problem1 (x : ℝ) (h1 : x * (x + 4) = -5 * (x + 4)) : x = -4 ∨ x = -5 := 
by 
  sorry

theorem problem2 (x : ℝ) (h2 : (x + 2) ^ 2 = (2 * x - 1) ^ 2) : x = 3 ∨ x = -1 / 3 := 
by 
  sorry

end problem1_problem2_l1965_196539


namespace length_of_field_l1965_196566

def width : ℝ := 13.5

def length (w : ℝ) : ℝ := 2 * w - 3

theorem length_of_field : length width = 24 :=
by
  -- full proof goes here
  sorry

end length_of_field_l1965_196566


namespace simplify_expression_l1965_196554

theorem simplify_expression (x y : ℝ) (h : y = x / (1 - 2 * x)) :
    (2 * x - 3 * x * y - 2 * y) / (y + x * y - x) = -7 / 3 := 
by {
  sorry
}

end simplify_expression_l1965_196554


namespace exists_positive_integer_m_l1965_196592

theorem exists_positive_integer_m (a b c d : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (hpos_d : d > 0) (h_cd : c * d = 1) : 
  ∃ m : ℕ, (a * b ≤ ↑m * ↑m) ∧ (↑m * ↑m ≤ (a + c) * (b + d)) :=
by
  sorry

end exists_positive_integer_m_l1965_196592


namespace unique_triple_l1965_196581

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def find_triples (x y z : ℕ) : Prop :=
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  is_prime x ∧ is_prime y ∧ is_prime z ∧
  is_prime (x - y) ∧ is_prime (y - z) ∧ is_prime (x - z)

theorem unique_triple :
  ∀ (x y z : ℕ), find_triples x y z → (x, y, z) = (7, 5, 2) :=
by
  sorry

end unique_triple_l1965_196581


namespace liters_pepsi_144_l1965_196520

/-- A drink vendor has 50 liters of Maaza, some liters of Pepsi, and 368 liters of Sprite. -/
def liters_maaza : ℕ := 50
def liters_sprite : ℕ := 368
def num_cans : ℕ := 281

/-- The total number of liters of drinks the vendor has -/
def total_liters (lit_pepsi: ℕ) : ℕ := liters_maaza + lit_pepsi + liters_sprite

/-- Given that the least number of cans required is 281, prove that the liters of Pepsi is 144. -/
theorem liters_pepsi_144 (P : ℕ) (h: total_liters P % num_cans = 0) : P = 144 :=
by
  sorry

end liters_pepsi_144_l1965_196520


namespace packets_of_sugar_per_week_l1965_196585

theorem packets_of_sugar_per_week (total_grams : ℕ) (packet_weight : ℕ) (total_packets : ℕ) :
  total_grams = 2000 →
  packet_weight = 100 →
  total_packets = total_grams / packet_weight →
  total_packets = 20 := 
  by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3 

end packets_of_sugar_per_week_l1965_196585


namespace diameter_circle_C_inscribed_within_D_l1965_196546

noncomputable def circle_diameter_C (d_D : ℝ) (ratio : ℝ) : ℝ :=
  let R := d_D / 2
  let r := (R : ℝ) / (Real.sqrt 5)
  2 * r

theorem diameter_circle_C_inscribed_within_D 
  (d_D : ℝ) (ratio : ℝ) (h_dD_pos : 0 < d_D) (h_ratio : ratio = 4)
  (h_dD : d_D = 24) : 
  circle_diameter_C d_D ratio = 24 * Real.sqrt 5 / 5 :=
by
  sorry

end diameter_circle_C_inscribed_within_D_l1965_196546


namespace football_team_progress_l1965_196598

theorem football_team_progress : 
  ∀ {loss gain : ℤ}, loss = 5 → gain = 11 → gain - loss = 6 :=
by
  intros loss gain h_loss h_gain
  rw [h_loss, h_gain]
  sorry

end football_team_progress_l1965_196598


namespace number_of_girls_l1965_196542

theorem number_of_girls (B G : ℕ) (h1 : B + G = 400) 
  (h2 : 0.60 * B = (6 / 10 : ℝ) * B) 
  (h3 : 0.80 * G = (8 / 10 : ℝ) * G) 
  (h4 : (6 / 10 : ℝ) * B + (8 / 10 : ℝ) * G = (65 / 100 : ℝ) * 400) : G = 100 := by
sorry

end number_of_girls_l1965_196542


namespace cody_spent_19_dollars_l1965_196556

-- Given conditions
def initial_money : ℕ := 45
def birthday_gift : ℕ := 9
def remaining_money : ℕ := 35

-- Problem: Prove that the amount of money spent on the game is $19.
theorem cody_spent_19_dollars :
  (initial_money + birthday_gift - remaining_money) = 19 :=
by sorry

end cody_spent_19_dollars_l1965_196556


namespace correct_average_and_variance_l1965_196537

theorem correct_average_and_variance
  (n : ℕ) (avg incorrect_variance correct_variance : ℝ)
  (incorrect_score1 actual_score1 incorrect_score2 actual_score2 : ℝ)
  (H1 : n = 48)
  (H2 : avg = 70)
  (H3 : incorrect_variance = 75)
  (H4 : incorrect_score1 = 50)
  (H5 : actual_score1 = 80)
  (H6 : incorrect_score2 = 100)
  (H7 : actual_score2 = 70)
  (Havg : avg = (n * avg - incorrect_score1 - incorrect_score2 + actual_score1 + actual_score2) / n)
  (Hvar : correct_variance = incorrect_variance + (actual_score1 - avg) ^ 2 + (actual_score2 - avg) ^ 2
                     - (incorrect_score1 - avg) ^ 2 - (incorrect_score2 - avg) ^ 2 / n) :
  avg = 70 ∧ correct_variance = 50 :=
by {
  sorry
}

end correct_average_and_variance_l1965_196537


namespace segment_AB_length_l1965_196512

-- Define the problem conditions
variables (AB CD h : ℝ)
variables (x : ℝ)
variables (AreaRatio : ℝ)
variable (k : ℝ := 5 / 2)

-- The given conditions
def condition1 : Prop := AB = 5 * x ∧ CD = 2 * x
def condition2 : Prop := AB + CD = 280
def condition3 : Prop := h = AB - 20
def condition4 : Prop := AreaRatio = k

-- The statement to prove
theorem segment_AB_length (h k : ℝ) (x : ℝ) :
  (AB = 5 * x ∧ CD = 2 * x) ∧ (AB + CD = 280) ∧ (h = AB - 20) ∧ (AreaRatio = k) → AB = 200 :=
by 
  sorry

end segment_AB_length_l1965_196512


namespace diagonal_of_square_l1965_196540

theorem diagonal_of_square (d : ℝ) (s : ℝ) (h : d = 2) (h_eq : s * Real.sqrt 2 = d) : s = Real.sqrt 2 :=
by sorry

end diagonal_of_square_l1965_196540


namespace increasing_interval_when_a_neg_increasing_and_decreasing_intervals_when_a_pos_l1965_196593

noncomputable def f (a x : ℝ) : ℝ := x - a / x

theorem increasing_interval_when_a_neg {a : ℝ} (h : a < 0) :
  ∀ x : ℝ, x > 0 → f a x > 0 :=
sorry

theorem increasing_and_decreasing_intervals_when_a_pos {a : ℝ} (h : a > 0) :
  (∀ x : ℝ, 0 < x → x < Real.sqrt a → f a x < 0) ∧
  (∀ x : ℝ, x > Real.sqrt a → f a x > 0) :=
sorry

end increasing_interval_when_a_neg_increasing_and_decreasing_intervals_when_a_pos_l1965_196593


namespace marta_sold_on_saturday_l1965_196584

-- Definitions of conditions
def initial_shipment : ℕ := 1000
def rotten_tomatoes : ℕ := 200
def second_shipment : ℕ := 2000
def tomatoes_on_tuesday : ℕ := 2500
def x := 300

-- Total tomatoes on Monday after the second shipment
def tomatoes_on_monday (sold_tomatoes : ℕ) : ℕ :=
  initial_shipment - sold_tomatoes - rotten_tomatoes + second_shipment

-- Theorem statement to prove
theorem marta_sold_on_saturday : (tomatoes_on_monday x = tomatoes_on_tuesday) -> (x = 300) :=
by 
  intro h
  sorry

end marta_sold_on_saturday_l1965_196584


namespace inheritance_amount_l1965_196574

theorem inheritance_amount (x : ℝ) 
  (federal_tax : ℝ := 0.25 * x) 
  (state_tax : ℝ := 0.15 * (x - federal_tax)) 
  (city_tax : ℝ := 0.05 * (x - federal_tax - state_tax)) 
  (total_tax : ℝ := 20000) :
  (federal_tax + state_tax + city_tax = total_tax) → 
  x = 50704 :=
by
  intros h
  sorry

end inheritance_amount_l1965_196574


namespace avg_variance_stability_excellent_performance_probability_l1965_196573

-- Define the scores of players A and B in seven games
def scores_A : List ℕ := [26, 28, 32, 22, 37, 29, 36]
def scores_B : List ℕ := [26, 29, 32, 28, 39, 29, 27]

-- Define the mean and variance calculations
def mean (scores : List ℕ) : ℚ := (scores.sum : ℚ) / scores.length
def variance (scores : List ℕ) : ℚ := 
  (scores.map (λ x => (x - mean scores) ^ 2)).sum / scores.length

theorem avg_variance_stability :
  mean scores_A = 30 ∧ mean scores_B = 30 ∧
  variance scores_A = 174 / 7 ∧ variance scores_B = 116 / 7 ∧
  variance scores_A > variance scores_B := 
by
  sorry

-- Define the probabilities of scoring higher than 30
def probability_excellent (scores : List ℕ) : ℚ := 
  (scores.filter (λ x => x > 30)).length / scores.length

theorem excellent_performance_probability :
  probability_excellent scores_A = 3 / 7 ∧ probability_excellent scores_B = 2 / 7 ∧
  (probability_excellent scores_A * probability_excellent scores_B = 6 / 49) :=
by
  sorry

end avg_variance_stability_excellent_performance_probability_l1965_196573


namespace total_hoodies_l1965_196535

def Fiona_hoodies : ℕ := 3
def Casey_hoodies : ℕ := Fiona_hoodies + 2

theorem total_hoodies : (Fiona_hoodies + Casey_hoodies) = 8 := by
  sorry

end total_hoodies_l1965_196535


namespace cost_calculation_l1965_196558

variables (H M F : ℝ)

theorem cost_calculation 
  (h1 : 3 * H + 5 * M + F = 23.50) 
  (h2 : 5 * H + 9 * M + F = 39.50) : 
  2 * H + 2 * M + 2 * F = 15.00 :=
sorry

end cost_calculation_l1965_196558


namespace set_complement_union_l1965_196514

namespace ProblemOne

def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem set_complement_union :
  (Aᶜ ∪ B) = {x : ℝ | -3 < x ∧ x < 5} := sorry

end ProblemOne

end set_complement_union_l1965_196514


namespace find_a_value_l1965_196509

theorem find_a_value (a x : ℝ) (h1 : 6 * (x + 8) = 18 * x) (h2 : 6 * x - 2 * (a - x) = 2 * a + x) : a = 7 :=
by
  sorry

end find_a_value_l1965_196509


namespace percentage_of_ore_contains_alloy_l1965_196502

def ore_contains_alloy_iron (weight_ore weight_iron : ℝ) (P : ℝ) : Prop :=
  (P / 100 * weight_ore) * 0.9 = weight_iron

theorem percentage_of_ore_contains_alloy (w_ore : ℝ) (w_iron : ℝ) (P : ℝ) 
    (h_w_ore : w_ore = 266.6666666666667) (h_w_iron : w_iron = 60) 
    (h_ore_contains : ore_contains_alloy_iron w_ore w_iron P) 
    : P = 25 :=
by
  rw [h_w_ore, h_w_iron] at h_ore_contains
  sorry

end percentage_of_ore_contains_alloy_l1965_196502


namespace customs_days_l1965_196591

-- Definitions from the problem conditions
def navigation_days : ℕ := 21
def transport_days : ℕ := 7
def total_days : ℕ := 30

-- Proposition we need to prove
theorem customs_days (expected_days: ℕ) (ship_departure_days : ℕ) : expected_days = 2 → ship_departure_days = 30 → (navigation_days + expected_days + transport_days = total_days) → expected_days = 2 :=
by
  intros h_expected h_departure h_eq
  sorry

end customs_days_l1965_196591


namespace max_value_of_a_l1965_196562

theorem max_value_of_a (a b c : ℕ) (h : a + b + c = Nat.gcd a b + Nat.gcd b c + Nat.gcd c a + 120) : a ≤ 240 :=
by
  sorry

end max_value_of_a_l1965_196562


namespace joe_height_l1965_196580

theorem joe_height (S J : ℕ) (h1 : S + J = 120) (h2 : J = 2 * S + 6) : J = 82 :=
by
  sorry

end joe_height_l1965_196580


namespace abc_divides_sum_exp21_l1965_196516

theorem abc_divides_sum_exp21
  (a b c : ℕ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hab : a ∣ b^4)
  (hbc : b ∣ c^4)
  (hca : c ∣ a^4)
  : abc ∣ (a + b + c)^21 :=
by
sorry

end abc_divides_sum_exp21_l1965_196516


namespace penthouse_units_l1965_196586

theorem penthouse_units (total_floors : ℕ) (regular_units_per_floor : ℕ) (penthouse_floors : ℕ) (total_units : ℕ) :
  total_floors = 23 →
  regular_units_per_floor = 12 →
  penthouse_floors = 2 →
  total_units = 256 →
  (total_units - (total_floors - penthouse_floors) * regular_units_per_floor) / penthouse_floors = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end penthouse_units_l1965_196586


namespace max_k_l1965_196588

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

theorem max_k (k : ℤ) : (∀ x : ℝ, 1 < x → f x - k * x + k > 0) → k ≤ 3 :=
by
  sorry

end max_k_l1965_196588


namespace hyperbola_eccentricity_condition_l1965_196564

theorem hyperbola_eccentricity_condition (m : ℝ) (h : m > 0) : 
  (∃ e : ℝ, e = Real.sqrt (1 + m) ∧ e > Real.sqrt 2) → m > 1 :=
by
  sorry

end hyperbola_eccentricity_condition_l1965_196564


namespace final_percentage_is_46_l1965_196551

def initial_volume : ℚ := 50
def initial_concentration : ℚ := 0.60
def drained_volume : ℚ := 35
def replacement_concentration : ℚ := 0.40

def initial_chemical_amount : ℚ := initial_volume * initial_concentration
def drained_chemical_amount : ℚ := drained_volume * initial_concentration
def remaining_chemical_amount : ℚ := initial_chemical_amount - drained_chemical_amount
def added_chemical_amount : ℚ := drained_volume * replacement_concentration
def final_chemical_amount : ℚ := remaining_chemical_amount + added_chemical_amount
def final_volume : ℚ := initial_volume

def final_percentage : ℚ := (final_chemical_amount / final_volume) * 100

theorem final_percentage_is_46 :
  final_percentage = 46 := by
  sorry

end final_percentage_is_46_l1965_196551


namespace probability_abc_plus_ab_plus_a_divisible_by_4_l1965_196531

noncomputable def count_multiples_of (n m : ℕ) : ℕ := (m / n)

noncomputable def probability_divisible_by_4 : ℚ := 
  let total_numbers := 2008
  let multiples_of_4 := count_multiples_of 4 total_numbers
  -- Probability that 'a' is divisible by 4
  let p_a := (multiples_of_4 : ℚ) / total_numbers
  -- Probability that 'a' is not divisible by 4
  let p_not_a := 1 - p_a
  -- Considering specific cases for b and c modulo 4
  let p_bc_cases := (2 * ((1 / 4) * (1 / 4)))  -- Probabilities for specific cases noted as 2 * (1/16)
  -- Adjusting probabilities for non-divisible 'a'
  let p_not_a_cases := p_bc_cases * p_not_a
  -- Total Probability
  p_a + p_not_a_cases

theorem probability_abc_plus_ab_plus_a_divisible_by_4 :
  probability_divisible_by_4 = 11 / 32 :=
sorry

end probability_abc_plus_ab_plus_a_divisible_by_4_l1965_196531


namespace savannah_rolls_l1965_196552

-- Definitions and conditions
def total_gifts := 12
def gifts_per_roll_1 := 3
def gifts_per_roll_2 := 5
def gifts_per_roll_3 := 4

-- Prove the number of rolls
theorem savannah_rolls :
  gifts_per_roll_1 + gifts_per_roll_2 + gifts_per_roll_3 = total_gifts →
  3 + 5 + 4 = 12 →
  3 = total_gifts / (gifts_per_roll_1 + gifts_per_roll_2 + gifts_per_roll_3) :=
by
  intros h1 h2
  sorry

end savannah_rolls_l1965_196552


namespace solve_system_l1965_196538

section system_equations

variable (x y : ℤ)

def equation1 := 2 * x - y = 5
def equation2 := 5 * x + 2 * y = 8
def solution := x = 2 ∧ y = -1

theorem solve_system : (equation1 x y) ∧ (equation2 x y) ↔ solution x y := by
  sorry

end system_equations

end solve_system_l1965_196538


namespace parallel_conditions_l1965_196524

-- Definitions of the lines
def l1 (m : ℝ) (x y : ℝ) : Prop := m * x + 3 * y - 6 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y + 2 = 0

-- Definition of parallel lines
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, l1 x y → l2 x y

-- Proof statement
theorem parallel_conditions (m : ℝ) :
  parallel (l1 m) (l2 m) ↔ (m = 1 ∨ m = -6) :=
by
  intros
  sorry

end parallel_conditions_l1965_196524


namespace part1_part2_l1965_196511

theorem part1 (a b c C : ℝ) (h : b - 1/2 * c = a * Real.cos C) (h1 : ∃ (A B : ℝ), Real.sin B - 1/2 * Real.sin C = Real.sin A * Real.cos C) :
  ∃ A : ℝ, A = 60 :=
sorry

theorem part2 (a b c : ℝ) (h1 : 4 * (b + c) = 3 * b * c) (h2 : a = 2 * Real.sqrt 3) (h3 : b - 1/2 * c = a * Real.cos 60)
  (h4 : ∀ (A : ℝ), A = 60) : ∃ S : ℝ, S = 2 * Real.sqrt 3 :=
sorry

end part1_part2_l1965_196511


namespace arithmetic_seq_problem_l1965_196519

noncomputable def a_n (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_seq_problem :
  ∃ d : ℕ, a_n 1 2 d = 2 ∧ a_n 2 2 d + a_n 3 2 d = 13 ∧ (a_n 4 2 d + a_n 5 2 d + a_n 6 2 d = 42) :=
by
  sorry

end arithmetic_seq_problem_l1965_196519


namespace sum_2004_impossible_sum_2005_possible_l1965_196560

-- Condition Definitions
def is_valid_square (s : ℕ × ℕ × ℕ × ℕ) : Prop :=
  s = (1, 2, 3, 4) ∨ s = (1, 2, 4, 3) ∨ s = (1, 3, 2, 4) ∨ s = (1, 3, 4, 2) ∨ 
  s = (1, 4, 2, 3) ∨ s = (1, 4, 3, 2) ∨ s = (2, 1, 3, 4) ∨ s = (2, 1, 4, 3) ∨ 
  s = (2, 3, 1, 4) ∨ s = (2, 3, 4, 1) ∨ s = (2, 4, 1, 3) ∨ s = (2, 4, 3, 1) ∨ 
  s = (3, 1, 2, 4) ∨ s = (3, 1, 4, 2) ∨ s = (3, 2, 1, 4) ∨ s = (3, 2, 4, 1) ∨ 
  s = (3, 4, 1, 2) ∨ s = (3, 4, 2, 1) ∨ s = (4, 1, 2, 3) ∨ s = (4, 1, 3, 2) ∨ 
  s = (4, 2, 1, 3) ∨ s = (4, 2, 3, 1) ∨ s = (4, 3, 1, 2) ∨ s = (4, 3, 2, 1)

-- Proof Problems
theorem sum_2004_impossible (n : ℕ) (corners : ℕ → ℕ × ℕ × ℕ × ℕ) (h : ∀ i, is_valid_square (corners i)) :
  4 * 2004 ≠ n * 10 := 
sorry

theorem sum_2005_possible (h : ∃ n, ∃ corners : ℕ → ℕ × ℕ × ℕ × ℕ, (∀ i, is_valid_square (corners i)) ∧ 4 * 2005 = n * 10 + 2005) :
  true := 
sorry

end sum_2004_impossible_sum_2005_possible_l1965_196560


namespace mod_37_5_l1965_196500

theorem mod_37_5 : 37 % 5 = 2 := 
by 
  sorry

end mod_37_5_l1965_196500


namespace unit_circle_arc_length_l1965_196508

theorem unit_circle_arc_length (r : ℝ) (A : ℝ) (θ : ℝ) : r = 1 ∧ A = 1 ∧ A = (1 / 2) * r^2 * θ → r * θ = 2 :=
by
  -- Given r = 1 (radius of unit circle) and area A = 1
  -- A = (1 / 2) * r^2 * θ is the formula for the area of the sector
  sorry

end unit_circle_arc_length_l1965_196508


namespace ferris_wheel_seat_capacity_l1965_196532

theorem ferris_wheel_seat_capacity
  (total_seats : ℕ)
  (broken_seats : ℕ)
  (total_people : ℕ)
  (seats_available : ℕ)
  (people_per_seat : ℕ)
  (h1 : total_seats = 18)
  (h2 : broken_seats = 10)
  (h3 : total_people = 120)
  (h4 : seats_available = total_seats - broken_seats)
  (h5 : people_per_seat = total_people / seats_available) :
  people_per_seat = 15 := 
by sorry

end ferris_wheel_seat_capacity_l1965_196532


namespace Martha_improvement_in_lap_time_l1965_196527

theorem Martha_improvement_in_lap_time 
  (initial_laps : ℕ) (initial_time : ℕ) 
  (first_month_laps : ℕ) (first_month_time : ℕ) 
  (second_month_laps : ℕ) (second_month_time : ℕ)
  (sec_per_min : ℕ)
  (conds : initial_laps = 15 ∧ initial_time = 30 ∧ first_month_laps = 18 ∧ first_month_time = 27 ∧ 
           second_month_laps = 20 ∧ second_month_time = 27 ∧ sec_per_min = 60)
  : ((initial_time / initial_laps : ℚ) - (second_month_time / second_month_laps)) * sec_per_min = 39 :=
by
  sorry

end Martha_improvement_in_lap_time_l1965_196527


namespace paint_cost_of_cube_l1965_196576

def cube_side_length : ℝ := 10
def paint_cost_per_quart : ℝ := 3.20
def coverage_per_quart : ℝ := 1200
def number_of_faces : ℕ := 6

theorem paint_cost_of_cube : 
  (number_of_faces * (cube_side_length^2) / coverage_per_quart) * paint_cost_per_quart = 3.20 :=
by 
  sorry

end paint_cost_of_cube_l1965_196576


namespace lcm_of_two_numbers_l1965_196536

theorem lcm_of_two_numbers (x y : ℕ) (h1 : Nat.gcd x y = 12) (h2 : x * y = 2460) : Nat.lcm x y = 205 :=
by
  -- Proof omitted
  sorry

end lcm_of_two_numbers_l1965_196536


namespace intersection_x_axis_l1965_196545

theorem intersection_x_axis (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (7, 3)) (h2 : (x2, y2) = (3, -1)) :
  ∃ x : ℝ, (x, 0) = (4, 0) :=
by sorry

end intersection_x_axis_l1965_196545


namespace natural_numbers_divisors_l1965_196553

theorem natural_numbers_divisors (n : ℕ) : 
  n + 1 ∣ n^2 + 1 → n = 0 ∨ n = 1 :=
by
  intro h
  sorry

end natural_numbers_divisors_l1965_196553


namespace selecting_female_probability_l1965_196529

theorem selecting_female_probability (female male : ℕ) (total : ℕ)
  (h_female : female = 4)
  (h_male : male = 6)
  (h_total : total = female + male) :
  (female / total : ℚ) = 2 / 5 := 
by
  -- Insert proof steps here
  sorry

end selecting_female_probability_l1965_196529


namespace find_other_number_l1965_196559

theorem find_other_number (a b : ℕ) (h1 : a + b = 62) (h2 : b - a = 12) (h3 : a = 25) : b = 37 :=
sorry

end find_other_number_l1965_196559


namespace minimum_value_ineq_l1965_196501

theorem minimum_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) : 
  (1 : ℝ) ≤ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by {
  sorry
}

end minimum_value_ineq_l1965_196501


namespace semicircle_perimeter_l1965_196594

-- Assuming π as 3.14 for approximation
def π_approx : ℝ := 3.14

-- Radius of the semicircle
def radius : ℝ := 2.1

-- Half of the circumference
def half_circumference (r : ℝ) : ℝ := π_approx * r

-- Diameter of the semicircle
def diameter (r : ℝ) : ℝ := 2 * r

-- Perimeter of the semicircle
def perimeter (r : ℝ) : ℝ := half_circumference r + diameter r

-- Theorem stating the perimeter of the semicircle with given radius
theorem semicircle_perimeter : perimeter radius = 10.794 := by
  sorry

end semicircle_perimeter_l1965_196594


namespace no_fixed_point_implies_no_double_fixed_point_l1965_196518

theorem no_fixed_point_implies_no_double_fixed_point (f : ℝ → ℝ) 
  (hf : Continuous f)
  (h : ∀ x : ℝ, f x ≠ x) :
  ∀ x : ℝ, f (f x) ≠ x :=
sorry

end no_fixed_point_implies_no_double_fixed_point_l1965_196518


namespace max_sum_of_factors_of_48_l1965_196523

theorem max_sum_of_factors_of_48 (d Δ : ℕ) (h : d * Δ = 48) : d + Δ ≤ 49 :=
sorry

end max_sum_of_factors_of_48_l1965_196523


namespace equalize_foma_ierema_l1965_196568

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l1965_196568


namespace stratified_sampling_second_year_students_l1965_196599

theorem stratified_sampling_second_year_students 
  (total_athletes : ℕ) 
  (first_year_students : ℕ) 
  (sample_size : ℕ) 
  (second_year_students_in_sample : ℕ)
  (h1 : total_athletes = 98) 
  (h2 : first_year_students = 56) 
  (h3 : sample_size = 28)
  (h4 : second_year_students_in_sample = (42 * sample_size) / total_athletes) :
  second_year_students_in_sample = 4 := 
sorry

end stratified_sampling_second_year_students_l1965_196599


namespace angle_measure_l1965_196596

theorem angle_measure (x : ℝ) 
  (h : x = 2 * (90 - x) - 60) : 
  x = 40 := 
  sorry

end angle_measure_l1965_196596


namespace scientific_notation_correct_l1965_196526

def num_people : ℝ := 2580000
def scientific_notation_form : ℝ := 2.58 * 10^6

theorem scientific_notation_correct : num_people = scientific_notation_form :=
by
  sorry

end scientific_notation_correct_l1965_196526


namespace distance_covered_is_9_17_miles_l1965_196569

noncomputable def totalDistanceCovered 
  (walkingTimeInMinutes : ℕ) (walkingRate : ℝ)
  (runningTimeInMinutes : ℕ) (runningRate : ℝ)
  (cyclingTimeInMinutes : ℕ) (cyclingRate : ℝ) : ℝ :=
  (walkingRate * (walkingTimeInMinutes / 60.0)) + 
  (runningRate * (runningTimeInMinutes / 60.0)) + 
  (cyclingRate * (cyclingTimeInMinutes / 60.0))

theorem distance_covered_is_9_17_miles :
  totalDistanceCovered 30 3 20 8 25 12 = 9.17 := 
by 
  sorry

end distance_covered_is_9_17_miles_l1965_196569


namespace inequality_may_not_hold_l1965_196578

theorem inequality_may_not_hold (a b : ℝ) (h : 0 < b ∧ b < a) :
  ¬(∀ x y : ℝ,  x = 1 / (a - b) → y = 1 / b → x > y) :=
sorry

end inequality_may_not_hold_l1965_196578


namespace equilateral_triangle_side_length_l1965_196534

theorem equilateral_triangle_side_length (c : ℕ) (h : c = 4 * 21) : c / 3 = 28 := by
  sorry

end equilateral_triangle_side_length_l1965_196534


namespace correct_propositions_l1965_196503

variable (P1 P2 P3 P4 : Prop)

-- Proposition 1: The negation of ∀ x ∈ ℝ, cos(x) > 0 is ∃ x ∈ ℝ such that cos(x) ≤ 0. 
def prop1 : Prop := 
  (¬ (∀ x : ℝ, Real.cos x > 0)) ↔ (∃ x : ℝ, Real.cos x ≤ 0)

-- Proposition 2: If 0 < a < 1, then the equation x^2 + a^x - 3 = 0 has only one real root.
def prop2 : Prop := 
  ∀ a : ℝ, (0 < a ∧ a < 1) → (∃! x : ℝ, x^2 + a^x - 3 = 0)

-- Proposition 3: For any real number x, if f(-x) = f(x) and f'(x) > 0 when x > 0, then f'(x) < 0 when x < 0.
def prop3 (f : ℝ → ℝ) : Prop := 
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, x > 0 → deriv f x > 0) →
  (∀ x : ℝ, x < 0 → deriv f x < 0)

-- Proposition 4: For a rectangle with area S and perimeter l, the pair of real numbers (6, 8) is a valid (S, l) pair.
def prop4 : Prop :=
  ∃ (a b : ℝ), (a * b = 6) ∧ (2 * (a + b) = 8)

theorem correct_propositions (P1_def : prop1)
                            (P3_def : ∀ f : ℝ → ℝ, prop3 f) :
                          P1 ∧ P3 :=
by
  sorry

end correct_propositions_l1965_196503


namespace price_of_10_pound_bag_l1965_196528

variables (P : ℝ) -- price of the 10-pound bag
def cost (n5 n10 n25 : ℕ) := n5 * 13.85 + n10 * P + n25 * 32.25

theorem price_of_10_pound_bag (h : ∃ (n5 n10 n25 : ℕ), n5 * 5 + n10 * 10 + n25 * 25 ≥ 65
  ∧ n5 * 5 + n10 * 10 + n25 * 25 ≤ 80 
  ∧ cost P n5 n10 n25 = 98.77) : 
  P = 20.42 :=
by
  -- Proof skipped
  sorry

end price_of_10_pound_bag_l1965_196528


namespace total_amount_divided_l1965_196582

-- Define the conditions
variables (A B C : ℕ)
axiom h1 : 4 * A = 5 * B
axiom h2 : 4 * A = 10 * C
axiom h3 : C = 160

-- Define the theorem to prove the total amount
theorem total_amount_divided (h1 : 4 * A = 5 * B) (h2 : 4 * A = 10 * C) (h3 : C = 160) : 
  A + B + C = 880 :=
sorry

end total_amount_divided_l1965_196582


namespace candies_count_l1965_196525

theorem candies_count :
  ∃ n, (n = 35 ∧ ∃ x, x ≥ 11 ∧ n = 3 * (x - 1) + 2) ∧ ∃ y, y ≤ 9 ∧ n = 4 * (y - 1) + 3 :=
  by {
    sorry
  }

end candies_count_l1965_196525


namespace maximum_overtakes_l1965_196513

-- Definitions based on problem conditions
structure Team where
  members : List ℕ
  speed_const : ℕ → ℝ -- Speed of each member is constant but different
  run_segment : ℕ → ℕ -- Each member runs exactly one segment
  
def relay_race_condition (team1 team2 : Team) : Prop :=
  team1.members.length = 20 ∧
  team2.members.length = 20 ∧
  ∀ i, (team1.speed_const i ≠ team2.speed_const i)

def transitions (team : Team) : ℕ :=
  team.members.length - 1

-- The theorem to be proved
theorem maximum_overtakes (team1 team2 : Team) (hcond : relay_race_condition team1 team2) : 
  ∃ n, n = 38 :=
by
  sorry

end maximum_overtakes_l1965_196513


namespace fraction_to_decimal_representation_l1965_196549

/-- Determine the decimal representation of a given fraction. -/
theorem fraction_to_decimal_representation : (45 / (2 ^ 3 * 5 ^ 4) = 0.0090) :=
sorry

end fraction_to_decimal_representation_l1965_196549


namespace inequality_non_empty_solution_set_l1965_196567

theorem inequality_non_empty_solution_set (a : ℝ) : ∃ x : ℝ, ax^2 - (a-2)*x - 2 ≤ 0 :=
sorry

end inequality_non_empty_solution_set_l1965_196567


namespace susan_correct_question_percentage_l1965_196521

theorem susan_correct_question_percentage (y : ℕ) : 
  (75 * (2 * y - 1) / y) = 
  ((6 * y - 3) / (8 * y) * 100)  :=
sorry

end susan_correct_question_percentage_l1965_196521


namespace product_percent_x_l1965_196587

variables {x y z w : ℝ}
variables (h1 : 0.45 * z = 1.2 * y) 
variables (h2 : y = 0.75 * x) 
variables (h3 : z = 0.8 * w)

theorem product_percent_x :
  (w * y) / x = 1.875 :=
by 
  sorry

end product_percent_x_l1965_196587


namespace quotient_ab_solution_l1965_196510

noncomputable def a : Real := sorry
noncomputable def b : Real := sorry

def condition1 (a b : Real) : Prop :=
  (1/(3 * a) + 1/b = 2011)

def condition2 (a b : Real) : Prop :=
  (1/a + 1/(3 * b) = 1)

theorem quotient_ab_solution (a b : Real) 
  (h1 : condition1 a b) 
  (h2 : condition2 a b) : 
  (a + b) / (a * b) = 1509 :=
sorry

end quotient_ab_solution_l1965_196510


namespace consecutive_even_integer_bases_l1965_196555

/-- Given \(X\) and \(Y\) are consecutive even positive integers and the equation
\[ 241_X + 36_Y = 94_{X+Y} \]
this theorem proves that \(X + Y = 22\). -/
theorem consecutive_even_integer_bases (X Y : ℕ) (h1 : X > 0) (h2 : Y = X + 2)
    (h3 : 2 * X^2 + 4 * X + 1 + 3 * Y + 6 = 9 * (X + Y) + 4) : 
    X + Y = 22 :=
by sorry

end consecutive_even_integer_bases_l1965_196555


namespace ellipse_with_foci_on_x_axis_l1965_196590

theorem ellipse_with_foci_on_x_axis (a : ℝ) :
  (∀ x y : ℝ, (x^2) / (a - 5) + (y^2) / 2 = 1 →  
   (∃ cx cy : ℝ, ∀ x', cx - x' = a - 5 ∧ cy = 2)) → 
  a > 7 :=
by sorry

end ellipse_with_foci_on_x_axis_l1965_196590


namespace range_of_n_l1965_196548

theorem range_of_n (m n : ℝ) (h₁ : n = m^2 + 2 * m + 2) (h₂ : |m| < 2) : -1 ≤ n ∧ n < 10 :=
sorry

end range_of_n_l1965_196548


namespace two_pow_2023_add_three_pow_2023_mod_seven_not_zero_l1965_196557

theorem two_pow_2023_add_three_pow_2023_mod_seven_not_zero : (2^2023 + 3^2023) % 7 ≠ 0 := 
by sorry

end two_pow_2023_add_three_pow_2023_mod_seven_not_zero_l1965_196557


namespace unique_p_value_l1965_196541

theorem unique_p_value (p : Nat) (h₁ : Nat.Prime (p+10)) (h₂ : Nat.Prime (p+14)) : p = 3 := by
  sorry

end unique_p_value_l1965_196541


namespace product_of_digits_in_base7_7891_is_zero_l1965_196544

/-- The function to compute the base 7 representation. -/
def to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else 
    let rest := to_base7 (n / 7)
    rest ++ [n % 7]

/-- The function to compute the product of the digits of a list. -/
def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * d) 1

theorem product_of_digits_in_base7_7891_is_zero :
  product_of_digits (to_base7 7891) = 0 := by
  sorry

end product_of_digits_in_base7_7891_is_zero_l1965_196544


namespace perfect_square_of_polynomial_l1965_196547

theorem perfect_square_of_polynomial (k : ℝ) (h : ∃ (p : ℝ), ∀ x : ℝ, x^2 + 6*x + k^2 = (x + p)^2) : k = 3 ∨ k = -3 := 
sorry

end perfect_square_of_polynomial_l1965_196547


namespace sequence_1234_to_500_not_divisible_by_9_l1965_196505

-- Definition for the sum of the digits of concatenated sequence
def sum_of_digits (n : ℕ) : ℕ :=
  -- This is a placeholder for the actual function calculating the sum of digits
  -- of all numbers from 1 to n concatenated together.
  sorry 

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem sequence_1234_to_500_not_divisible_by_9 : ¬ is_divisible_by_9 (sum_of_digits 500) :=
by
  -- Placeholder indicating the solution facts and methods should go here.
  sorry

end sequence_1234_to_500_not_divisible_by_9_l1965_196505


namespace find_expression_l1965_196565

theorem find_expression : 1^567 + 3^5 / 3^3 - 2 = 8 :=
by
  sorry

end find_expression_l1965_196565


namespace smallest_base_10_integer_l1965_196530

theorem smallest_base_10_integer :
  ∃ (c d : ℕ), 3 < c ∧ 3 < d ∧ (3 * c + 4 = 4 * d + 3) ∧ (3 * c + 4 = 19) :=
by {
 sorry
}

end smallest_base_10_integer_l1965_196530


namespace sum_proper_divisors_243_l1965_196597

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 :=
by
  sorry

end sum_proper_divisors_243_l1965_196597


namespace principal_trebled_after_5_years_l1965_196517

-- Definitions of the conditions
def original_simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100
def total_simple_interest (P R n T : ℕ) : ℕ := (P * R * n) / 100 + (3 * P * R * (T - n)) / 100

-- The theorem statement
theorem principal_trebled_after_5_years :
  ∀ (P R : ℕ), original_simple_interest P R 10 = 800 →
              total_simple_interest P R 5 10 = 1600 →
              5 = 5 :=
by
  intros P R h1 h2
  sorry

end principal_trebled_after_5_years_l1965_196517


namespace mass_percentage_of_Cl_in_NaClO_l1965_196583

noncomputable def molarMassNa : ℝ := 22.99
noncomputable def molarMassCl : ℝ := 35.45
noncomputable def molarMassO : ℝ := 16.00

noncomputable def molarMassNaClO : ℝ := molarMassNa + molarMassCl + molarMassO

theorem mass_percentage_of_Cl_in_NaClO : 
  (molarMassCl / molarMassNaClO) * 100 = 47.61 :=
by 
  sorry

end mass_percentage_of_Cl_in_NaClO_l1965_196583


namespace num_triangles_correct_num_lines_correct_l1965_196515

-- Definition for the first proof problem: Number of triangles
def num_triangles (n : ℕ) : ℕ := Nat.choose n 3

theorem num_triangles_correct :
  num_triangles 9 = 84 :=
by
  sorry

-- Definition for the second proof problem: Number of lines
def num_lines (n : ℕ) : ℕ := Nat.choose n 2

theorem num_lines_correct :
  num_lines 9 = 36 :=
by
  sorry

end num_triangles_correct_num_lines_correct_l1965_196515


namespace Namjoon_gave_Yoongi_9_pencils_l1965_196579

theorem Namjoon_gave_Yoongi_9_pencils
  (stroke_pencils : ℕ)
  (strokes : ℕ)
  (pencils_left : ℕ)
  (total_pencils : ℕ := stroke_pencils * strokes)
  (given_pencils : ℕ := total_pencils - pencils_left) :
  stroke_pencils = 12 →
  strokes = 2 →
  pencils_left = 15 →
  given_pencils = 9 := by
  sorry

end Namjoon_gave_Yoongi_9_pencils_l1965_196579
