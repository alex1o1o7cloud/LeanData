import Mathlib

namespace NUMINAMATH_GPT_fifth_pyTriple_is_correct_l1775_177569

-- Definitions based on conditions from part (a)
def pyTriple (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := 2 * n + 1
  let b := 2 * n * (n + 1)
  let c := b + 1
  (a, b, c)

-- Question: Prove that the 5th Pythagorean triple is (11, 60, 61)
theorem fifth_pyTriple_is_correct : pyTriple 5 = (11, 60, 61) :=
  by
    -- Skip the proof
    sorry

end NUMINAMATH_GPT_fifth_pyTriple_is_correct_l1775_177569


namespace NUMINAMATH_GPT_rectangle_area_l1775_177549

theorem rectangle_area (w : ℝ) (h : ℝ) (area : ℝ) 
  (h1 : w = 5)
  (h2 : h = 2 * w) :
  area = h * w := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1775_177549


namespace NUMINAMATH_GPT_age_ratio_in_4_years_l1775_177536

-- Definitions based on conditions
def Age6YearsAgoVimal := 12
def Age6YearsAgoSaroj := 10
def CurrentAgeSaroj := 16
def CurrentAgeVimal := Age6YearsAgoVimal + 6

-- Lean statement to prove the problem
theorem age_ratio_in_4_years (x : ℕ) 
  (h_ratio : (CurrentAgeVimal + x) / (CurrentAgeSaroj + x) = 11 / 10) :
  x = 4 := 
sorry

end NUMINAMATH_GPT_age_ratio_in_4_years_l1775_177536


namespace NUMINAMATH_GPT_janice_initial_sentences_l1775_177548

theorem janice_initial_sentences :
  ∀ (initial_sentences total_sentences erased_sentences: ℕ)
    (typed_rate before_break_minutes additional_minutes after_meeting_minutes: ℕ),
  typed_rate = 6 →
  before_break_minutes = 20 →
  additional_minutes = 15 →
  after_meeting_minutes = 18 →
  erased_sentences = 40 →
  total_sentences = 536 →
  (total_sentences - (before_break_minutes * typed_rate + (before_break_minutes + additional_minutes) * typed_rate + after_meeting_minutes * typed_rate - erased_sentences)) = initial_sentences →
  initial_sentences = 138 :=
by
  intros initial_sentences total_sentences erased_sentences typed_rate before_break_minutes additional_minutes after_meeting_minutes
  intros h_rate h_before h_additional h_after_meeting h_erased h_total h_eqn
  rw [h_rate, h_before, h_additional, h_after_meeting, h_erased, h_total] at h_eqn
  linarith

end NUMINAMATH_GPT_janice_initial_sentences_l1775_177548


namespace NUMINAMATH_GPT_find_fahrenheit_l1775_177558

variable (F : ℝ)
variable (C : ℝ)

theorem find_fahrenheit (h : C = 40) (h' : C = 5 / 9 * (F - 32)) : F = 104 := by
  sorry

end NUMINAMATH_GPT_find_fahrenheit_l1775_177558


namespace NUMINAMATH_GPT_alice_walks_miles_each_morning_l1775_177507

theorem alice_walks_miles_each_morning (x : ℕ) :
  (5 * x + 5 * 12 = 110) → x = 10 :=
by
  intro h
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_alice_walks_miles_each_morning_l1775_177507


namespace NUMINAMATH_GPT_increasing_interval_l1775_177508

noncomputable def f (x k : ℝ) : ℝ := (x^2 / 2) - k * (Real.log x)

theorem increasing_interval (k : ℝ) (h₀ : 0 < k) : 
  ∃ (a : ℝ), (a = Real.sqrt k) ∧ 
  ∀ (x : ℝ), (x > a) → (∃ ε > 0, ∀ y, (x < y) → (f y k > f x k)) :=
sorry

end NUMINAMATH_GPT_increasing_interval_l1775_177508


namespace NUMINAMATH_GPT_gcd_problem_l1775_177514

theorem gcd_problem : Nat.gcd 12740 220 - 10 = 10 :=
by
  sorry

end NUMINAMATH_GPT_gcd_problem_l1775_177514


namespace NUMINAMATH_GPT_correct_transformation_l1775_177582

theorem correct_transformation (m : ℤ) (h : 2 * m - 1 = 3) : 2 * m = 4 :=
by
  sorry

end NUMINAMATH_GPT_correct_transformation_l1775_177582


namespace NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l1775_177539

-- Define the necessary conditions
def a := 2
def d := 5
def l := 1007  -- last term

-- Prove the number of terms in the sequence
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 202 :=
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l1775_177539


namespace NUMINAMATH_GPT_parabola_directrix_l1775_177523

theorem parabola_directrix (x : ℝ) : 
  (6 * x^2 + 5 = y) → (y = 6 * x^2 + 5) → (y = 6 * 0^2 + 5) → (y = (119 : ℝ) / 24) := 
sorry

end NUMINAMATH_GPT_parabola_directrix_l1775_177523


namespace NUMINAMATH_GPT_steve_total_time_on_roads_l1775_177504

variables (d : ℝ) (v_back : ℝ) (v_to_work : ℝ)

-- Constants from the problem statement
def distance := 10 -- The distance from Steve's house to work is 10 km
def speed_back := 5 -- Steve's speed on the way back from work is 5 km/h

-- Given conditions
def speed_to_work := speed_back / 2 -- On the way back, Steve drives twice as fast as he did on the way to work

-- Define the time to get to work and back
def time_to_work := distance / speed_to_work
def time_back_home := distance / speed_back

-- Total time on roads
def total_time := time_to_work + time_back_home

-- The theorem to prove
theorem steve_total_time_on_roads : total_time = 6 := by
  -- Proof here
  sorry

end NUMINAMATH_GPT_steve_total_time_on_roads_l1775_177504


namespace NUMINAMATH_GPT_probability_neither_red_nor_purple_l1775_177522

theorem probability_neither_red_nor_purple 
    (total_balls : ℕ)
    (white_balls : ℕ) 
    (green_balls : ℕ) 
    (yellow_balls : ℕ) 
    (red_balls : ℕ) 
    (purple_balls : ℕ) 
    (h_total : total_balls = white_balls + green_balls + yellow_balls + red_balls + purple_balls)
    (h_counts : white_balls = 50 ∧ green_balls = 30 ∧ yellow_balls = 8 ∧ red_balls = 9 ∧ purple_balls = 3):
    (88 : ℚ) / 100 = 0.88 :=
by
  sorry

end NUMINAMATH_GPT_probability_neither_red_nor_purple_l1775_177522


namespace NUMINAMATH_GPT_logan_usual_cartons_l1775_177562

theorem logan_usual_cartons 
  (C : ℕ)
  (h1 : ∀ cartons, (∀ jars : ℕ, jars = 20 * cartons) → jars = 20 * C)
  (h2 : ∀ cartons, cartons = C - 20)
  (h3 : ∀ damaged_jars, (∀ cartons : ℕ, cartons = 5) → damaged_jars = 3 * 5)
  (h4 : ∀ completely_damaged_jars, completely_damaged_jars = 20)
  (h5 : ∀ good_jars, good_jars = 565) :
  C = 50 :=
by
  sorry

end NUMINAMATH_GPT_logan_usual_cartons_l1775_177562


namespace NUMINAMATH_GPT_imaginary_part_of_z_l1775_177527

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : z = 2 / (-1 + I)) : z.im = -1 := 
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l1775_177527


namespace NUMINAMATH_GPT_train_speed_is_60_kmph_l1775_177570

noncomputable def speed_of_train_in_kmph (length_meters time_seconds : ℝ) : ℝ :=
  (length_meters / time_seconds) * 3.6

theorem train_speed_is_60_kmph (length_meters time_seconds : ℝ) :
  length_meters = 50 → time_seconds = 3 → speed_of_train_in_kmph length_meters time_seconds = 60 :=
by
  intros h_length h_time
  simp [speed_of_train_in_kmph, h_length, h_time]
  norm_num
  sorry

end NUMINAMATH_GPT_train_speed_is_60_kmph_l1775_177570


namespace NUMINAMATH_GPT_initial_distance_is_18_l1775_177530

-- Step a) Conditions and Definitions
def distance_covered (v t d : ℝ) : Prop := 
  d = v * t

def increased_speed_time (v t d : ℝ) : Prop := 
  d = (v + 1) * (3 / 4 * t)

def decreased_speed_time (v t d : ℝ) : Prop := 
  d = (v - 1) * (t + 3)

-- Step c) Mathematically Equivalent Proof Problem
theorem initial_distance_is_18 (v t d : ℝ) 
  (h1 : distance_covered v t d) 
  (h2 : increased_speed_time v t d) 
  (h3 : decreased_speed_time v t d) : 
  d = 18 :=
sorry

end NUMINAMATH_GPT_initial_distance_is_18_l1775_177530


namespace NUMINAMATH_GPT_base_height_calculation_l1775_177587

noncomputable def height_of_sculpture : ℚ := 2 + 5/6 -- 2 feet 10 inches in feet
noncomputable def total_height : ℚ := 3.5
noncomputable def height_of_base : ℚ := 2/3

theorem base_height_calculation (h1 : height_of_sculpture = 17/6) (h2 : total_height = 21/6):
  height_of_base = total_height - height_of_sculpture := by
  sorry

end NUMINAMATH_GPT_base_height_calculation_l1775_177587


namespace NUMINAMATH_GPT_cost_to_plant_flowers_l1775_177544

theorem cost_to_plant_flowers :
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  cost_flowers + cost_clay_pot + cost_soil = 45 := 
by
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  show cost_flowers + cost_clay_pot + cost_soil = 45
  sorry

end NUMINAMATH_GPT_cost_to_plant_flowers_l1775_177544


namespace NUMINAMATH_GPT_find_value_of_abc_cubed_l1775_177596

-- Variables and conditions
variables {a b c : ℝ}
variables (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = a^4 + b^4 + c^4)

-- The statement
theorem find_value_of_abc_cubed (ha : a ≠ 0) (hb: b ≠ 0) (hc: c ≠ 0) :
  a^3 + b^3 + c^3 = -3 * a * b * (a + b) :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_abc_cubed_l1775_177596


namespace NUMINAMATH_GPT_number_of_ways_l1775_177585

theorem number_of_ways (h_walk : ℕ) (h_drive : ℕ) (h_eq1 : h_walk = 3) (h_eq2 : h_drive = 4) : h_walk + h_drive = 7 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_ways_l1775_177585


namespace NUMINAMATH_GPT_complete_square_l1775_177571

theorem complete_square (x : ℝ) : x^2 + 4*x + 1 = 0 -> (x + 2)^2 = 3 :=
by sorry

end NUMINAMATH_GPT_complete_square_l1775_177571


namespace NUMINAMATH_GPT_smallest_value_of_n_l1775_177547

theorem smallest_value_of_n 
  (n : ℕ) 
  (h1 : ∀ θ : ℝ, θ = (n - 2) * 180 / n) 
  (h2 : ∀ α : ℝ, α = 360 / n) 
  (h3 : 28 = 180 / n) :
  n = 45 :=
sorry

end NUMINAMATH_GPT_smallest_value_of_n_l1775_177547


namespace NUMINAMATH_GPT_quoted_price_correct_l1775_177542

noncomputable def after_tax_yield (yield : ℝ) (tax_rate : ℝ) : ℝ :=
  yield * (1 - tax_rate)

noncomputable def real_yield (after_tax_yield : ℝ) (inflation_rate : ℝ) : ℝ :=
  after_tax_yield - inflation_rate

noncomputable def quoted_price (dividend_rate : ℝ) (real_yield : ℝ) (commission_rate : ℝ) : ℝ :=
  real_yield / (dividend_rate / (1 + commission_rate))

theorem quoted_price_correct :
  quoted_price 0.16 (real_yield (after_tax_yield 0.08 0.15) 0.03) 0.02 = 24.23 :=
by
  -- This is the proof statement. Since the task does not require us to prove it, we use 'sorry'.
  sorry

end NUMINAMATH_GPT_quoted_price_correct_l1775_177542


namespace NUMINAMATH_GPT_correct_option_l1775_177520

-- Conditions as definitions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^6
def optionB (a : ℝ) : Prop := 3 * a - 2 * a = 1
def optionC (a : ℝ) : Prop := (-2 * a^2)^3 = -8 * a^6
def optionD (a : ℝ) : Prop := a^6 / a^2 = a^3

-- The statement to prove
theorem correct_option (a : ℝ) : optionC a :=
by 
  unfold optionC
  sorry

end NUMINAMATH_GPT_correct_option_l1775_177520


namespace NUMINAMATH_GPT_chord_on_ellipse_midpoint_l1775_177543

theorem chord_on_ellipse_midpoint :
  ∀ (A B : ℝ × ℝ)
    (hx1 : (A.1^2) / 2 + A.2^2 = 1)
    (hx2 : (B.1^2) / 2 + B.2^2 = 1)
    (mid : (A.1 + B.1) / 2 = 1/2 ∧ (A.2 + B.2) / 2 = 1/2),
  ∃ (k : ℝ), ∀ (x y : ℝ), y - 1/2 = k * (x - 1/2) ↔ 2 * x + 4 * y = 3 := 
sorry

end NUMINAMATH_GPT_chord_on_ellipse_midpoint_l1775_177543


namespace NUMINAMATH_GPT_find_train_parameters_l1775_177572

-- Definitions based on the problem statement
def bridge_length : ℕ := 1000
def time_total : ℕ := 60
def time_on_bridge : ℕ := 40
def speed_train (x : ℕ) := (40 * x = bridge_length)
def length_train (x y : ℕ) := (60 * x = bridge_length + y)

-- Stating the problem to be proved
theorem find_train_parameters (x y : ℕ) (h₁ : speed_train x) (h₂ : length_train x y) :
  x = 20 ∧ y = 200 :=
sorry

end NUMINAMATH_GPT_find_train_parameters_l1775_177572


namespace NUMINAMATH_GPT_find_x_l1775_177505

theorem find_x (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1)
  (geom_seq : (x - ⌊x⌋) * x = ⌊x⌋^2) : x = 1.618 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1775_177505


namespace NUMINAMATH_GPT_expression_value_l1775_177581

theorem expression_value (x : ℤ) (h : x = 2) : (2 * x + 5)^3 = 729 := by
  sorry

end NUMINAMATH_GPT_expression_value_l1775_177581


namespace NUMINAMATH_GPT_decorations_left_to_put_up_l1775_177506

variable (S B W P C T : Nat)
variable (h₁ : S = 12)
variable (h₂ : B = 4)
variable (h₃ : W = 12)
variable (h₄ : P = 2 * W)
variable (h₅ : C = 1)
variable (h₆ : T = 83)

theorem decorations_left_to_put_up (h₁ : S = 12) (h₂ : B = 4) (h₃ : W = 12) (h₄ : P = 2 * W) (h₅ : C = 1) (h₆ : T = 83) :
  T - (S + B + W + P + C) = 30 := sorry

end NUMINAMATH_GPT_decorations_left_to_put_up_l1775_177506


namespace NUMINAMATH_GPT_sum_of_roots_l1775_177551

theorem sum_of_roots (a b : ℝ) (h1 : a^2 - 4*a - 2023 = 0) (h2 : b^2 - 4*b - 2023 = 0) : a + b = 4 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l1775_177551


namespace NUMINAMATH_GPT_prime_in_choices_l1775_177580

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def twenty := 20
def twenty_one := 21
def twenty_three := 23
def twenty_five := 25
def twenty_seven := 27

theorem prime_in_choices :
  is_prime twenty_three ∧ ¬ is_prime twenty ∧ ¬ is_prime twenty_one ∧ ¬ is_prime twenty_five ∧ ¬ is_prime twenty_seven :=
by
  sorry

end NUMINAMATH_GPT_prime_in_choices_l1775_177580


namespace NUMINAMATH_GPT_unique_A_value_l1775_177554

theorem unique_A_value (A : ℝ) (x1 x2 : ℂ) (hx1_ne : x1 ≠ x2) :
  (x1 * (x1 + 1) = A) ∧ (x2 * (x2 + 1) = A) ∧ (A * x1^4 + 3 * x1^3 + 5 * x1 = x2^4 + 3 * x2^3 + 5 * x2) 
  → A = -7 := by
  sorry

end NUMINAMATH_GPT_unique_A_value_l1775_177554


namespace NUMINAMATH_GPT_sum_max_min_expr_l1775_177534

theorem sum_max_min_expr (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) : 
    let expr := (x / |x|) + (|y| / y) - (|x * y| / (x * y))
    max (max expr (expr)) (min expr expr) = -2 :=
sorry

end NUMINAMATH_GPT_sum_max_min_expr_l1775_177534


namespace NUMINAMATH_GPT_x_is_36_percent_of_z_l1775_177591

variable (x y z : ℝ)

theorem x_is_36_percent_of_z (h1 : x = 1.20 * y) (h2 : y = 0.30 * z) : x = 0.36 * z :=
by
  sorry

end NUMINAMATH_GPT_x_is_36_percent_of_z_l1775_177591


namespace NUMINAMATH_GPT_tangent_line_at_2_number_of_zeros_l1775_177593

noncomputable def f (x : ℝ) := 3 * Real.log x + (1/2) * x^2 - 4 * x + 1

theorem tangent_line_at_2 :
  let x := 2
  ∃ k b : ℝ, (∀ y : ℝ, y = k * x + b) ∧ (k = -1/2) ∧ (b = 3 * Real.log 2 - 5) ∧ (∀ x y : ℝ, (y - (3 * Real.log 2 - 5) = -1/2 * (x - 2)) ↔ (x + 2 * y - 6 * Real.log 2 + 8 = 0)) :=
by
  sorry

noncomputable def g (x : ℝ) (m : ℝ) := f x - m

theorem number_of_zeros (m : ℝ) :
  let g := g
  (m > -5/2 ∨ m < 3 * Real.log 3 - 13/2 → ∃ x : ℝ, g x = 0) ∧ 
  (m = -5/2 ∨ m = 3 * Real.log 3 - 13/2 → ∃ x y : ℝ, g x = 0 ∧ g y = 0 ∧ x ≠ y) ∧
  (3 * Real.log 3 - 13/2 < m ∧ m < -5/2 → ∃ x y z : ℝ, g x = 0 ∧ g y = 0 ∧ g z = 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_2_number_of_zeros_l1775_177593


namespace NUMINAMATH_GPT_total_marbles_l1775_177537

def mary_marbles := 9
def joan_marbles := 3
def john_marbles := 7

theorem total_marbles :
  mary_marbles + joan_marbles + john_marbles = 19 :=
by
  sorry

end NUMINAMATH_GPT_total_marbles_l1775_177537


namespace NUMINAMATH_GPT_range_m_l1775_177566

noncomputable def circle_c (x y : ℝ) : Prop := (x - 4) ^ 2 + (y - 3) ^ 2 = 4

def point_A (m : ℝ) : ℝ × ℝ := (-m, 0)
def point_B (m : ℝ) : ℝ × ℝ := (m, 0)

theorem range_m (m : ℝ) (P : ℝ × ℝ) :
  circle_c P.1 P.2 ∧ m > 0 ∧ (∃ (a b : ℝ), P = (a, b) ∧ (a + m) * (a - m) + b ^ 2 = 0) → m ∈ Set.Icc 3 7 :=
sorry

end NUMINAMATH_GPT_range_m_l1775_177566


namespace NUMINAMATH_GPT_find_a6_l1775_177550

open Nat

noncomputable def arith_seq (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

def a2 := 4
def a4 := 2

theorem find_a6 (a1 d : ℤ) (h_a2 : arith_seq a1 d 2 = a2) (h_a4 : arith_seq a1 d 4 = a4) : 
  arith_seq a1 d 6 = 0 := by
  sorry

end NUMINAMATH_GPT_find_a6_l1775_177550


namespace NUMINAMATH_GPT_mary_total_spent_l1775_177599

-- The conditions given in the problem
def cost_berries : ℝ := 11.08
def cost_apples : ℝ := 14.33
def cost_peaches : ℝ := 9.31

-- The theorem to prove the total cost
theorem mary_total_spent : cost_berries + cost_apples + cost_peaches = 34.72 := 
by
  sorry

end NUMINAMATH_GPT_mary_total_spent_l1775_177599


namespace NUMINAMATH_GPT_distance_from_plate_to_bottom_edge_l1775_177564

theorem distance_from_plate_to_bottom_edge :
    ∀ (d : ℕ), 10 + 63 = 20 + d → d = 53 :=
by
  intros d h
  sorry

end NUMINAMATH_GPT_distance_from_plate_to_bottom_edge_l1775_177564


namespace NUMINAMATH_GPT_problem1_problem2_l1775_177574

-- Problem 1:
theorem problem1 (P : ℝ × ℝ) (hP : P = (-1, 0)) :
  (∃ m b, (∀ x y, y = m * x + b ↔ (3 * x - y + 3 = 0)) ∧ ∀ x y, (x = -1 → y = 0 → y = m * x + b)) → 
  ∃ m b, ∀ x y, y = m * x + b ↔ (3 * x - y + 3 = 0) :=
sorry

-- Problem 2:
theorem problem2 (L1 : ℝ → ℝ → Prop) (hL1 : ∀ x y, L1 x y ↔ 3 * x + 4 * y - 12 = 0) (d : ℝ) (hd : d = 7) :
  (∃ c, ∀ x y, (3 * x + 4 * y + c = 0 ∨ 3 * x + 4 * y - 47 = 0) ↔ L1 x y ∧ d = 7) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1775_177574


namespace NUMINAMATH_GPT_AM_GM_inequality_example_l1775_177529

theorem AM_GM_inequality_example (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) : 
  (a + b) * (a + c) * (b + c) ≥ 8 * a * b * c :=
by
  sorry

end NUMINAMATH_GPT_AM_GM_inequality_example_l1775_177529


namespace NUMINAMATH_GPT_solve_for_m_l1775_177561

def f (x : ℝ) (m : ℝ) := x^3 - m * x + 3

def f_prime (x : ℝ) (m : ℝ) := 3 * x^2 - m

theorem solve_for_m (m : ℝ) : f_prime 1 m = 0 → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_m_l1775_177561


namespace NUMINAMATH_GPT_combined_pre_tax_and_pre_tip_cost_l1775_177528

theorem combined_pre_tax_and_pre_tip_cost (x y : ℝ) 
  (hx : 1.28 * x = 35.20) 
  (hy : 1.19 * y = 22.00) : 
  x + y = 46 := 
by
  sorry

end NUMINAMATH_GPT_combined_pre_tax_and_pre_tip_cost_l1775_177528


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1775_177586

theorem quadratic_inequality_solution_set (x : ℝ) : (x + 3) * (2 - x) < 0 ↔ x < -3 ∨ x > 2 := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1775_177586


namespace NUMINAMATH_GPT_complex_abs_of_sqrt_l1775_177584

variable (z : ℂ)

theorem complex_abs_of_sqrt (h : z^2 = 16 - 30 * Complex.I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end NUMINAMATH_GPT_complex_abs_of_sqrt_l1775_177584


namespace NUMINAMATH_GPT_solve_a_range_m_l1775_177575

def f (x : ℝ) (a : ℝ) : ℝ := |x - a|

theorem solve_a :
  (∀ x : ℝ, f x 2 ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) ↔ (2 = 2) :=
by {
  sorry
}

theorem range_m :
  (∀ x : ℝ, f (3 * x) 2 + f (x + 3) 2 ≥ m) ↔ (m ≤ 5 / 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_a_range_m_l1775_177575


namespace NUMINAMATH_GPT_puppy_food_total_correct_l1775_177552

def daily_food_first_two_weeks : ℚ := 3 / 4
def weekly_food_first_two_weeks : ℚ := 7 * daily_food_first_two_weeks
def total_food_first_two_weeks : ℚ := 2 * weekly_food_first_two_weeks

def daily_food_following_two_weeks : ℚ := 1
def weekly_food_following_two_weeks : ℚ := 7 * daily_food_following_two_weeks
def total_food_following_two_weeks : ℚ := 2 * weekly_food_following_two_weeks

def today_food : ℚ := 1 / 2

def total_food_over_4_weeks : ℚ :=
  total_food_first_two_weeks + total_food_following_two_weeks + today_food

theorem puppy_food_total_correct :
  total_food_over_4_weeks = 25 := by
  sorry

end NUMINAMATH_GPT_puppy_food_total_correct_l1775_177552


namespace NUMINAMATH_GPT_patricia_candies_final_l1775_177503

def initial_candies : ℕ := 764
def taken_candies : ℕ := 53
def back_candies_per_7_taken : ℕ := 19

theorem patricia_candies_final :
  let given_back_times := taken_candies / 7
  let total_given_back := given_back_times * back_candies_per_7_taken
  let final_candies := initial_candies - taken_candies + total_given_back
  final_candies = 844 :=
by
  sorry

end NUMINAMATH_GPT_patricia_candies_final_l1775_177503


namespace NUMINAMATH_GPT_solve_system_l1775_177524

theorem solve_system :
  ∀ (a1 a2 c1 c2 x y : ℝ),
  (a1 * 5 + 10 = c1) →
  (a2 * 5 + 10 = c2) →
  (a1 * x + 2 * y = a1 - c1) →
  (a2 * x + 2 * y = a2 - c2) →
  (x = -4) ∧ (y = -5) := by
  intros a1 a2 c1 c2 x y h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_solve_system_l1775_177524


namespace NUMINAMATH_GPT_repeating_decimal_427_diff_l1775_177597

theorem repeating_decimal_427_diff :
  let G := 0.427427427427
  let num := 427
  let denom := 999
  num.gcd denom = 1 →
  denom - num = 572 :=
by
  intros G num denom gcd_condition
  sorry

end NUMINAMATH_GPT_repeating_decimal_427_diff_l1775_177597


namespace NUMINAMATH_GPT_tan_of_geometric_sequence_is_negative_sqrt_3_l1775_177595

variable {a : ℕ → ℝ} 

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q, m + n = p + q → a m * a n = a p * a q

theorem tan_of_geometric_sequence_is_negative_sqrt_3 
  (hgeo : is_geometric_sequence a)
  (hcond : a 2 * a 3 * a 4 = - a 7 ^ 2 ∧ a 7 ^ 2 = 64) :
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = - Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_of_geometric_sequence_is_negative_sqrt_3_l1775_177595


namespace NUMINAMATH_GPT_isosceles_triangle_base_angles_l1775_177526

theorem isosceles_triangle_base_angles 
  (α β : ℝ) -- α and β are the base angles
  (h : α = β)
  (height leg : ℝ)
  (h_height_leg : height = leg / 2) : 
  α = 75 ∨ α = 15 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angles_l1775_177526


namespace NUMINAMATH_GPT_integer_points_on_segment_l1775_177573

open Int

def is_integer_point (x y : ℝ) : Prop := ∃ (a b : ℤ), x = a ∧ y = b

def f (n : ℕ) : ℕ := 
  if 3 ∣ n then 2
  else 0

theorem integer_points_on_segment (n : ℕ) (hn : 0 < n) :
  (f n) = if 3 ∣ n then 2 else 0 := 
  sorry

end NUMINAMATH_GPT_integer_points_on_segment_l1775_177573


namespace NUMINAMATH_GPT_find_f_prime_at_2_l1775_177525

noncomputable def f (f' : ℝ → ℝ) (x : ℝ) : ℝ :=
  x^2 + 2 * x * f' 2 - Real.log x

theorem find_f_prime_at_2 (f' : ℝ → ℝ) (h : ∀ x, deriv (f f') x = f' x) :
  f' 2 = -7 / 2 :=
by
  have H := h 2
  sorry

end NUMINAMATH_GPT_find_f_prime_at_2_l1775_177525


namespace NUMINAMATH_GPT_jackson_weeks_of_school_l1775_177579

def jackson_sandwich_per_week : ℕ := 2

def missed_wednesdays : ℕ := 1
def missed_fridays : ℕ := 2
def total_missed_sandwiches : ℕ := missed_wednesdays + missed_fridays

def total_sandwiches_eaten : ℕ := 69

def total_sandwiches_without_missing : ℕ := total_sandwiches_eaten + total_missed_sandwiches

def calculate_weeks_of_school (total_sandwiches : ℕ) (sandwiches_per_week : ℕ) : ℕ :=
total_sandwiches / sandwiches_per_week

theorem jackson_weeks_of_school : calculate_weeks_of_school total_sandwiches_without_missing jackson_sandwich_per_week = 36 :=
by
  sorry

end NUMINAMATH_GPT_jackson_weeks_of_school_l1775_177579


namespace NUMINAMATH_GPT_solve_car_production_l1775_177560

def car_production_problem : Prop :=
  ∃ (NorthAmericaCars : ℕ) (TotalCars : ℕ) (EuropeCars : ℕ),
    NorthAmericaCars = 3884 ∧
    TotalCars = 6755 ∧
    EuropeCars = TotalCars - NorthAmericaCars ∧
    EuropeCars = 2871

theorem solve_car_production : car_production_problem := by
  sorry

end NUMINAMATH_GPT_solve_car_production_l1775_177560


namespace NUMINAMATH_GPT_angle_DEF_EDF_proof_l1775_177538

theorem angle_DEF_EDF_proof (angle_DOE : ℝ) (angle_EOD : ℝ) 
  (h1 : angle_DOE = 130) (h2 : angle_EOD = 90) :
  let angle_DEF := 45
  let angle_EDF := 45
  angle_DEF = 45 ∧ angle_EDF = 45 :=
by
  sorry

end NUMINAMATH_GPT_angle_DEF_EDF_proof_l1775_177538


namespace NUMINAMATH_GPT_roots_not_in_interval_l1775_177511

theorem roots_not_in_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, (a^x + a^(-x) = 2 * a) → (x < -1 ∨ x > 1) :=
by
  sorry

end NUMINAMATH_GPT_roots_not_in_interval_l1775_177511


namespace NUMINAMATH_GPT_sum_of_ages_l1775_177533

theorem sum_of_ages (a b c : ℕ) (twin : a = b) (product : a * b * c = 256) : a + b + c = 20 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l1775_177533


namespace NUMINAMATH_GPT_dvaneft_percentage_bounds_l1775_177594

theorem dvaneft_percentage_bounds (x y z : ℝ) (n m : ℕ) 
  (h1 : x * n + y * m = z * (m + n))
  (h2 : 3 * x * n = y * m)
  (h3_1 : 10 ≤ y - x)
  (h3_2 : y - x ≤ 18)
  (h4_1 : 18 ≤ z)
  (h4_2 : z ≤ 42)
  : (15 ≤ (n:ℝ) / (2 * (n + m)) * 100) ∧ ((n:ℝ) / (2 * (n + m)) * 100 ≤ 25) :=
by
  sorry

end NUMINAMATH_GPT_dvaneft_percentage_bounds_l1775_177594


namespace NUMINAMATH_GPT_double_counted_toddlers_l1775_177576

def number_of_toddlers := 21
def missed_toddlers := 3
def billed_count := 26

theorem double_counted_toddlers : 
  ∃ (D : ℕ), (number_of_toddlers + D - missed_toddlers = billed_count) ∧ D = 8 :=
by
  sorry

end NUMINAMATH_GPT_double_counted_toddlers_l1775_177576


namespace NUMINAMATH_GPT_inequality_subtraction_l1775_177559

variable (a b : ℝ)

-- Given conditions
axiom nonzero_a : a ≠ 0 
axiom nonzero_b : b ≠ 0 
axiom a_lt_b : a < b 

-- Proof statement
theorem inequality_subtraction : a - 3 < b - 3 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_subtraction_l1775_177559


namespace NUMINAMATH_GPT_reggie_free_throws_l1775_177510

namespace BasketballShootingContest

-- Define the number of points for different shots
def points (layups free_throws long_shots : ℕ) : ℕ :=
  1 * layups + 2 * free_throws + 3 * long_shots

-- Conditions given in the problem
def Reggie_points (F: ℕ) : ℕ := 
  points 3 F 1

def Brother_points : ℕ := 
  points 0 0 4

-- The given condition that Reggie loses by 2 points
theorem reggie_free_throws:
  ∃ F : ℕ, Reggie_points F + 2 = Brother_points :=
sorry

end BasketballShootingContest

end NUMINAMATH_GPT_reggie_free_throws_l1775_177510


namespace NUMINAMATH_GPT_smallest_x_y_sum_l1775_177589

theorem smallest_x_y_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hne : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 24) :
  x + y = 100 :=
sorry

end NUMINAMATH_GPT_smallest_x_y_sum_l1775_177589


namespace NUMINAMATH_GPT_fish_buckets_last_l1775_177598

theorem fish_buckets_last (buckets_sharks : ℕ) (buckets_total : ℕ) 
  (h1 : buckets_sharks = 4)
  (h2 : ∀ (buckets_dolphins : ℕ), buckets_dolphins = buckets_sharks / 2)
  (h3 : ∀ (buckets_other : ℕ), buckets_other = 5 * buckets_sharks)
  (h4 : buckets_total = 546)
  : 546 / ((buckets_sharks + (buckets_sharks / 2) + (5 * buckets_sharks)) * 7) = 3 :=
by
  -- Calculation steps skipped for brevity
  sorry

end NUMINAMATH_GPT_fish_buckets_last_l1775_177598


namespace NUMINAMATH_GPT_distance_D_to_plane_l1775_177521

-- Given conditions about the distances from points A, B, and C to plane M
variables (a b c : ℝ)

-- Formalizing the distance from vertex D to plane M
theorem distance_D_to_plane (a b c : ℝ) : 
  ∃ d : ℝ, d = |a + b + c| ∨ d = |a + b - c| ∨ d = |a - b + c| ∨ d = |-a + b + c| ∨ 
                    d = |a - b - c| ∨ d = |-a - b + c| ∨ d = |-a + b - c| ∨ d = |-a - b - c| := sorry

end NUMINAMATH_GPT_distance_D_to_plane_l1775_177521


namespace NUMINAMATH_GPT_savings_in_july_l1775_177546

-- Definitions based on the conditions
def savings_june : ℕ := 27
def savings_august : ℕ := 21
def expenses_books : ℕ := 5
def expenses_shoes : ℕ := 17
def final_amount_left : ℕ := 40

-- Main theorem stating the problem
theorem savings_in_july (J : ℕ) : 
  savings_june + J + savings_august - (expenses_books + expenses_shoes) = final_amount_left → 
  J = 14 :=
by
  sorry

end NUMINAMATH_GPT_savings_in_july_l1775_177546


namespace NUMINAMATH_GPT_problem1_problem2_l1775_177583

theorem problem1 (a b : ℝ) : ((a * b) ^ 6 / (a * b) ^ 2 * (a * b) ^ 4) = a^8 * b^8 := 
by sorry

theorem problem2 (x : ℝ) : ((3 * x^3)^2 * x^5 - (-x^2)^6 / x) = 8 * x^11 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1775_177583


namespace NUMINAMATH_GPT_total_chickens_and_ducks_l1775_177517

-- Definitions based on conditions
def num_chickens : Nat := 45
def more_chickens_than_ducks : Nat := 8
def num_ducks : Nat := num_chickens - more_chickens_than_ducks

-- The proof statement
theorem total_chickens_and_ducks : num_chickens + num_ducks = 82 := by
  -- The actual proof is omitted, only the statement is required
  sorry

end NUMINAMATH_GPT_total_chickens_and_ducks_l1775_177517


namespace NUMINAMATH_GPT_find_integer_solutions_l1775_177512

theorem find_integer_solutions (n : ℕ) (h1 : ∃ b : ℤ, 8 * n - 7 = b^2) (h2 : ∃ a : ℤ, 18 * n - 35 = a^2) : 
  n = 2 ∨ n = 22 := 
sorry

end NUMINAMATH_GPT_find_integer_solutions_l1775_177512


namespace NUMINAMATH_GPT_unique_solution_of_system_l1775_177555

theorem unique_solution_of_system :
  ∀ (a : ℝ), (∃ (x y : ℝ), x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) →
  ((a = 1 ∧ ∃ x y : ℝ, x = -1 ∧ y = 0 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) ∨
  (a = -3 ∧ ∃ x y : ℝ, x = 1 ∧ y = 2 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0)) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_of_system_l1775_177555


namespace NUMINAMATH_GPT_polynomial_remainder_l1775_177500

theorem polynomial_remainder (x : ℂ) : 
  (3 * x ^ 1010 + x ^ 1000) % (x ^ 2 + 1) * (x - 1) = 3 * x ^ 2 + 1 := 
sorry

end NUMINAMATH_GPT_polynomial_remainder_l1775_177500


namespace NUMINAMATH_GPT_find_num_oranges_l1775_177535

def num_oranges (O : ℝ) (x : ℕ) : Prop :=
  6 * 0.21 + O * (x : ℝ) = 1.77 ∧ 2 * 0.21 + 5 * O = 1.27
  ∧ 0.21 = 0.21

theorem find_num_oranges (O : ℝ) (x : ℕ) (h : num_oranges O x) : x = 3 :=
  sorry

end NUMINAMATH_GPT_find_num_oranges_l1775_177535


namespace NUMINAMATH_GPT_percent_greater_than_l1775_177513

theorem percent_greater_than (M N : ℝ) (hN : N ≠ 0) : (M - N) / N * 100 = 100 * (M - N) / N :=
by sorry

end NUMINAMATH_GPT_percent_greater_than_l1775_177513


namespace NUMINAMATH_GPT_last_digit_322_power_111569_l1775_177588

theorem last_digit_322_power_111569 : (322 ^ 111569) % 10 = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_last_digit_322_power_111569_l1775_177588


namespace NUMINAMATH_GPT_most_cost_effective_80_oranges_l1775_177556

noncomputable def cost_of_oranges (p1 p2 p3 : ℕ) (q1 q2 q3 : ℕ) : ℕ :=
  let cost_per_orange_p1 := p1 / q1
  let cost_per_orange_p2 := p2 / q2
  let cost_per_orange_p3 := p3 / q3
  if cost_per_orange_p3 ≤ cost_per_orange_p2 ∧ cost_per_orange_p3 ≤ cost_per_orange_p1 then
    (80 / q3) * p3
  else if cost_per_orange_p2 ≤ cost_per_orange_p1 then
    (80 / q2) * p2
  else
    (80 / q1) * p1

theorem most_cost_effective_80_oranges :
  cost_of_oranges 35 45 95 6 9 20 = 380 :=
by sorry

end NUMINAMATH_GPT_most_cost_effective_80_oranges_l1775_177556


namespace NUMINAMATH_GPT_ratio_of_intercepts_l1775_177515

variable {c : ℝ} (non_zero_c : c ≠ 0) (u v : ℝ)
-- Condition: The first line, slope 8, y-intercept c, x-intercept (u, 0)
variable (h_u : u = -c / 8)
-- Condition: The second line, slope 4, y-intercept c, x-intercept (v, 0)
variable (h_v : v = -c / 4)

theorem ratio_of_intercepts (non_zero_c : c ≠ 0)
    (h_u : u = -c / 8) (h_v : v = -c / 4) : u / v = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_intercepts_l1775_177515


namespace NUMINAMATH_GPT_direct_variation_y_value_l1775_177545

theorem direct_variation_y_value (x y k : ℝ) (h1 : y = k * x) (h2 : ∀ x, x = 5 → y = 10) 
                                 (h3 : ∀ x, x < 0 → k = 4) (hx : x = -6) : y = -24 :=
sorry

end NUMINAMATH_GPT_direct_variation_y_value_l1775_177545


namespace NUMINAMATH_GPT_derivative_f_at_1_l1775_177578

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * Real.sin x

theorem derivative_f_at_1 : (deriv f 1) = 2 + 2 * Real.cos 1 := 
sorry

end NUMINAMATH_GPT_derivative_f_at_1_l1775_177578


namespace NUMINAMATH_GPT_find_a_l1775_177563

noncomputable def f (a x : ℝ) : ℝ := Real.log (Real.sqrt (1 + a * x ^ 2) - x)

theorem find_a (a : ℝ) :
  (∀ (x : ℝ), f a (-x) = -f a x) ↔ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1775_177563


namespace NUMINAMATH_GPT_find_a_and_b_l1775_177557

-- Given conditions
def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_and_b (a b : ℝ) :
  (∀ x, ∀ y, tangent_line x y → y = b ∧ x = 0) ∧
  (∀ x, ∀ y, y = curve x a b) →
  a = 1 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l1775_177557


namespace NUMINAMATH_GPT_find_XY_XZ_l1775_177519

open Set

variable (P Q R X Y Z : Type) [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited X] [Inhabited Y] [Inhabited Z]
variable (length : (P → P → Real) → (Q → Q → Real) → (R → R → Real) → (X → X → Real) → (Y → Y → Real) → (Z → Z → Real) )


-- Definitions based on the conditions
def similar_triangles (PQ QR PR XY XZ YZ : Real) : Prop :=
  QR / YZ = PQ / XY ∧ QR / YZ = PR / XZ

def PQ : Real := 8
def QR : Real := 16
def YZ : Real := 32

-- We need to prove (XY = 16 ∧ XZ = 32) given the conditions of similarity
theorem find_XY_XZ (XY XZ : Real) (h_sim : similar_triangles PQ QR PQ XY XZ YZ) : XY = 16 ∧ XZ = 32 :=
by
  sorry

end NUMINAMATH_GPT_find_XY_XZ_l1775_177519


namespace NUMINAMATH_GPT_group_selection_l1775_177518

theorem group_selection (m k n : ℕ) (h_m : m = 6) (h_k : k = 7) 
  (groups : ℕ → ℕ) (h_groups : groups k = n) : 
  n % 10 = (m + k) % 10 :=
by
  sorry

end NUMINAMATH_GPT_group_selection_l1775_177518


namespace NUMINAMATH_GPT_max_b_c_plus_four_over_a_l1775_177501

theorem max_b_c_plus_four_over_a (a b c : ℝ) (ha : a < 0)
  (h_quad : ∀ x : ℝ, -1 < x ∧ x < 2 → (a * x^2 + b * x + c) > 0) : 
  b - c + 4 / a ≤ -4 :=
sorry

end NUMINAMATH_GPT_max_b_c_plus_four_over_a_l1775_177501


namespace NUMINAMATH_GPT_pedoe_inequality_l1775_177532

variables {a b c a' b' c' Δ Δ' : ℝ} {A A' : ℝ}

theorem pedoe_inequality :
  a' ^ 2 * (-a ^ 2 + b ^ 2 + c ^ 2) +
  b' ^ 2 * (a ^ 2 - b ^ 2 + c ^ 2) +
  c' ^ 2 * (a ^ 2 + b ^ 2 - c ^ 2) -
  16 * Δ * Δ' =
  2 * (b * c' - b' * c) ^ 2 +
  8 * b * b' * c * c' * (Real.sin ((A - A') / 2)) ^ 2 := sorry

end NUMINAMATH_GPT_pedoe_inequality_l1775_177532


namespace NUMINAMATH_GPT_max_extra_packages_l1775_177577

/-- Max's delivery performance --/
def max_daily_packages : Nat := 35

/-- (1) Max delivered the maximum number of packages on two days --/
def max_2_days : Nat := 2 * max_daily_packages

/-- (2) On two other days, Max unloaded a total of 50 packages --/
def two_days_50 : Nat := 50

/-- (3) On one day, Max delivered one-seventh of the maximum possible daily performance --/
def one_seventh_day : Nat := max_daily_packages / 7

/-- (4) On the last two days, the sum of packages was four-fifths of the maximum daily performance --/
def last_2_days : Nat := 2 * (4 * max_daily_packages / 5)

/-- (5) Total packages delivered in the week --/
def total_delivered : Nat := max_2_days + two_days_50 + one_seventh_day + last_2_days

/-- (6) Total possible packages in a week if worked at maximum performance --/
def total_possible : Nat := 7 * max_daily_packages

/-- (7) Difference between total possible and total delivered packages --/
def difference : Nat := total_possible - total_delivered

/-- Proof problem: Prove the difference is 64 --/
theorem max_extra_packages : difference = 64 := by
  sorry

end NUMINAMATH_GPT_max_extra_packages_l1775_177577


namespace NUMINAMATH_GPT_alley_width_l1775_177590

theorem alley_width (L w : ℝ) (k h : ℝ)
    (h1 : k = L / 2)
    (h2 : h = L * (Real.sqrt 3) / 2)
    (h3 : w^2 + (L / 2)^2 = L^2)
    (h4 : w^2 + (L * (Real.sqrt 3) / 2)^2 = L^2):
    w = (Real.sqrt 3) * L / 2 := 
sorry

end NUMINAMATH_GPT_alley_width_l1775_177590


namespace NUMINAMATH_GPT_blueberry_pancakes_count_l1775_177516

-- Definitions of the conditions
def total_pancakes : ℕ := 67
def banana_pancakes : ℕ := 24
def plain_pancakes : ℕ := 23

-- Statement of the problem
theorem blueberry_pancakes_count :
  total_pancakes - banana_pancakes - plain_pancakes = 20 := by
  sorry

end NUMINAMATH_GPT_blueberry_pancakes_count_l1775_177516


namespace NUMINAMATH_GPT_james_lifting_heavy_after_39_days_l1775_177568

noncomputable def JamesInjuryHealingTime : Nat := 3
noncomputable def HealingTimeFactor : Nat := 5
noncomputable def WaitingTimeAfterHealing : Nat := 3
noncomputable def AdditionalWaitingTimeWeeks : Nat := 3

theorem james_lifting_heavy_after_39_days :
  let healing_time := JamesInjuryHealingTime * HealingTimeFactor
  let total_time_before_workout := healing_time + WaitingTimeAfterHealing
  let additional_waiting_time_days := AdditionalWaitingTimeWeeks * 7
  let total_time_before_lifting_heavy := total_time_before_workout + additional_waiting_time_days
  total_time_before_lifting_heavy = 39 := by
  sorry

end NUMINAMATH_GPT_james_lifting_heavy_after_39_days_l1775_177568


namespace NUMINAMATH_GPT_expression_for_f_l1775_177502

noncomputable def f (x : ℝ) : ℝ := sorry

theorem expression_for_f (x : ℝ) :
  (∀ x, f (x - 1) = x^2) → f x = x^2 + 2 * x + 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_expression_for_f_l1775_177502


namespace NUMINAMATH_GPT_longest_side_is_48_l1775_177567

noncomputable def longest_side_of_triangle (a b c : ℝ) (ha : a/b = 1/2) (hb : b/c = 1/3) (hc : c/a = 1/4) (hp : a + b + c = 104) : ℝ :=
  a

theorem longest_side_is_48 {a b c : ℝ} (ha : a/b = 1/2) (hb : b/c = 1/3) (hc : c/a = 1/4) (hp : a + b + c = 104) : 
  longest_side_of_triangle a b c ha hb hc hp = 48 :=
sorry

end NUMINAMATH_GPT_longest_side_is_48_l1775_177567


namespace NUMINAMATH_GPT_sum_of_integers_eq_28_24_23_l1775_177565

theorem sum_of_integers_eq_28_24_23 
  (a b : ℕ) 
  (h1 : a * b + a + b = 143)
  (h2 : Nat.gcd a b = 1)
  (h3 : a < 30)
  (h4 : b < 30) 
  : a + b = 28 ∨ a + b = 24 ∨ a + b = 23 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_eq_28_24_23_l1775_177565


namespace NUMINAMATH_GPT_cone_volume_l1775_177541

theorem cone_volume (p q : ℕ) (a α : ℝ) :
  V = (2 * π * a^3) / (3 * (Real.sin (2 * α)) * (Real.cos (180 * q / (p + q)))^2 * (Real.cos α)) :=
sorry

end NUMINAMATH_GPT_cone_volume_l1775_177541


namespace NUMINAMATH_GPT_jacket_purchase_price_l1775_177553

theorem jacket_purchase_price (S D P : ℝ) 
  (h1 : S = P + 0.30 * S)
  (h2 : D = 0.80 * S)
  (h3 : 6.000000000000007 = D - P) :
  P = 42 :=
by
  sorry

end NUMINAMATH_GPT_jacket_purchase_price_l1775_177553


namespace NUMINAMATH_GPT_ratio_of_sums_l1775_177592

noncomputable def first_sum : Nat := 
  let sequence := (List.range' 1 15)
  let differences := (List.range' 2 30).map (fun x => 2 * x)
  let sequence_sum := sequence.zip differences |>.map (λ ⟨a, d⟩ => 10 / 2 * (2 * a + 9 * d))
  5 * (20 * (sequence_sum.sum))

noncomputable def second_sum : Nat :=
  let sequence := (List.range' 1 15)
  let differences := (List.range' 1 29).filterMap (fun x => if x % 2 = 1 then some x else none)
  let sequence_sum := sequence.zip differences |>.map (λ ⟨a, d⟩ => 10 / 2 * (2 * a + 9 * d))
  5 * (20 * (sequence_sum.sum) - 135)

theorem ratio_of_sums : (first_sum / second_sum : Rat) = (160 / 151 : Rat) :=
  sorry

end NUMINAMATH_GPT_ratio_of_sums_l1775_177592


namespace NUMINAMATH_GPT_origin_moves_distance_l1775_177540

noncomputable def origin_distance_moved : ℝ :=
  let B := (3, 1)
  let B' := (7, 9)
  let k := 1.5
  let center_of_dilation := (-1, -3)
  let d0 := Real.sqrt ((-1)^2 + (-3)^2)
  let d1 := k * d0
  d1 - d0

theorem origin_moves_distance :
  origin_distance_moved = 0.5 * Real.sqrt 10 :=
by 
  sorry

end NUMINAMATH_GPT_origin_moves_distance_l1775_177540


namespace NUMINAMATH_GPT_units_digit_fraction_l1775_177531

theorem units_digit_fraction :
  (30 * 31 * 32 * 33 * 34 * 35) % 10000 % 10 = 4 :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_units_digit_fraction_l1775_177531


namespace NUMINAMATH_GPT_p_necessary_not_sufficient_for_q_l1775_177509

def condition_p (x : ℝ) : Prop := x > 2
def condition_q (x : ℝ) : Prop := x > 3

theorem p_necessary_not_sufficient_for_q (x : ℝ) :
  (∀ (x : ℝ), condition_q x → condition_p x) ∧ ¬(∀ (x : ℝ), condition_p x → condition_q x) :=
by
  sorry

end NUMINAMATH_GPT_p_necessary_not_sufficient_for_q_l1775_177509
