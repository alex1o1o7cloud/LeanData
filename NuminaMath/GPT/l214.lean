import Mathlib

namespace NUMINAMATH_GPT_sum_mobile_phone_keypad_l214_21433

/-- The numbers on a standard mobile phone keypad are 0 through 9. -/
def mobile_phone_keypad : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- The sum of all the numbers on a standard mobile phone keypad is 45. -/
theorem sum_mobile_phone_keypad : mobile_phone_keypad.sum = 45 := by
  sorry

end NUMINAMATH_GPT_sum_mobile_phone_keypad_l214_21433


namespace NUMINAMATH_GPT_roof_ratio_l214_21476

theorem roof_ratio (L W : ℝ) 
  (h1 : L * W = 784) 
  (h2 : L - W = 42) : 
  L / W = 4 := by 
  sorry

end NUMINAMATH_GPT_roof_ratio_l214_21476


namespace NUMINAMATH_GPT_least_possible_value_of_one_integer_l214_21414

theorem least_possible_value_of_one_integer (
  A B C D E F : ℤ
) (h1 : (A + B + C + D + E + F) / 6 = 63)
  (h2 : A ≤ 100 ∧ B ≤ 100 ∧ C ≤ 100 ∧ D ≤ 100 ∧ E ≤ 100 ∧ F ≤ 100)
  (h3 : (A + B + C) / 3 = 65) : 
  ∃ D E F, (D + E + F) = 183 ∧ min D (min E F) = 83 := sorry

end NUMINAMATH_GPT_least_possible_value_of_one_integer_l214_21414


namespace NUMINAMATH_GPT_find_m_l214_21430

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := -x^3 + 6*x^2 - m

theorem find_m (m : ℝ) (h : ∃ x : ℝ, f x m = 12) : m = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l214_21430


namespace NUMINAMATH_GPT_ratio_of_packets_to_tent_stakes_l214_21479

-- Definitions based on the conditions provided
def total_items (D T W : ℕ) : Prop := D + T + W = 22
def tent_stakes (T : ℕ) : Prop := T = 4
def bottles_of_water (W T : ℕ) : Prop := W = T + 2

-- The goal is to prove the ratio of packets of drink mix to tent stakes
theorem ratio_of_packets_to_tent_stakes (D T W : ℕ) :
  total_items D T W →
  tent_stakes T →
  bottles_of_water W T →
  D = 3 * T :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_packets_to_tent_stakes_l214_21479


namespace NUMINAMATH_GPT_part_I_part_II_l214_21410

noncomputable def f (x a : ℝ) := 2 * |x - 1| - a
noncomputable def g (x m : ℝ) := - |x + m|

theorem part_I (a : ℝ) : 
  (∀ x : ℝ, g x 3 > -1 ↔ x = -3) :=
by
  sorry

theorem part_II (a : ℝ) (m : ℝ) :
  (∀ x : ℝ, f x a ≥ g x m) ↔ (a < 4) :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l214_21410


namespace NUMINAMATH_GPT_compound_interest_double_l214_21465

theorem compound_interest_double (t : ℕ) (r : ℝ) (n : ℕ) (P : ℝ) :
  r = 0.15 → n = 1 → (2 : ℝ) < (1 + r)^t → t ≥ 5 :=
by
  intros hr hn h
  sorry

end NUMINAMATH_GPT_compound_interest_double_l214_21465


namespace NUMINAMATH_GPT_find_a2_l214_21477

def arithmetic_sequence (a : ℕ → ℚ) := 
  (a 1 = 1) ∧ ∀ n, a (n + 2) - a n = 3

theorem find_a2 (a : ℕ → ℚ) (h : arithmetic_sequence a) : 
  a 2 = 5 / 2 := 
by
  -- Conditions
  have a1 : a 1 = 1 := h.1
  have h_diff : ∀ n, a (n + 2) - a n = 3 := h.2
  -- Proof steps can be written here
  sorry

end NUMINAMATH_GPT_find_a2_l214_21477


namespace NUMINAMATH_GPT_number_of_friends_l214_21496

-- Conditions/Definitions
def total_cost : ℤ := 13500
def cost_per_person : ℤ := 900

-- Prove that Dawson is going with 14 friends.
theorem number_of_friends (h1 : total_cost = 13500) (h2 : cost_per_person = 900) :
  (total_cost / cost_per_person) - 1 = 14 :=
by
  sorry

end NUMINAMATH_GPT_number_of_friends_l214_21496


namespace NUMINAMATH_GPT_simplify_expression_l214_21409

theorem simplify_expression (x y z : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : z ≠ 0) 
  (h : x^2 + y^2 + z^2 = xy + yz + zx) : 
  (1 / (y^2 + z^2 - x^2)) + (1 / (x^2 + z^2 - y^2)) + (1 / (x^2 + y^2 - z^2)) = 3 / x^2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l214_21409


namespace NUMINAMATH_GPT_inscribed_sphere_radius_l214_21426

-- Define the distances from points X and Y to the faces of the tetrahedron
variable (X_AB X_AD X_AC X_BC : ℝ)
variable (Y_AB Y_AD Y_AC Y_BC : ℝ)

-- Setting the given distances in the problem
axiom dist_X_AB : X_AB = 14
axiom dist_X_AD : X_AD = 11
axiom dist_X_AC : X_AC = 29
axiom dist_X_BC : X_BC = 8

axiom dist_Y_AB : Y_AB = 15
axiom dist_Y_AD : Y_AD = 13
axiom dist_Y_AC : Y_AC = 25
axiom dist_Y_BC : Y_BC = 11

-- The theorem to prove that the radius of the inscribed sphere of the tetrahedron is 17
theorem inscribed_sphere_radius : 
  ∃ r : ℝ, r = 17 ∧ 
  (∀ (d_X_AB d_X_AD d_X_AC d_X_BC d_Y_AB d_Y_AD d_Y_AC d_Y_BC: ℝ),
    d_X_AB = 14 ∧ d_X_AD = 11 ∧ d_X_AC = 29 ∧ d_X_BC = 8 ∧
    d_Y_AB = 15 ∧ d_Y_AD = 13 ∧ d_Y_AC = 25 ∧ d_Y_BC = 11 → 
    r = 17) :=
sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_l214_21426


namespace NUMINAMATH_GPT_frosting_cans_needed_l214_21440

theorem frosting_cans_needed :
  let daily_cakes := 10
  let days := 5
  let total_cakes := daily_cakes * days
  let eaten_cakes := 12
  let remaining_cakes := total_cakes - eaten_cakes
  let cans_per_cake := 2
  let total_cans := remaining_cakes * cans_per_cake
  total_cans = 76 := 
by
  sorry

end NUMINAMATH_GPT_frosting_cans_needed_l214_21440


namespace NUMINAMATH_GPT_percentage_increase_decrease_l214_21407

theorem percentage_increase_decrease (p q M : ℝ) (hp : 0 < p) (hq : 0 < q) (hM : 0 < M) (hq100 : q < 100) :
  (M * (1 + p / 100) * (1 - q / 100) = 1.1 * M) ↔ (p = (10 + 100 * q) / (100 - q)) :=
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_decrease_l214_21407


namespace NUMINAMATH_GPT_joan_balloons_l214_21491

def sally_balloons : ℕ := 5
def jessica_balloons : ℕ := 2
def total_balloons : ℕ := 16

theorem joan_balloons : sally_balloons + jessica_balloons = 7 ∧ total_balloons = 16 → total_balloons - (sally_balloons + jessica_balloons) = 9 :=
by
  sorry

end NUMINAMATH_GPT_joan_balloons_l214_21491


namespace NUMINAMATH_GPT_unique_prime_sum_diff_l214_21418

theorem unique_prime_sum_diff (p : ℕ) (primeP : Prime p)
  (hx : ∃ (x y : ℕ), Prime x ∧ Prime y ∧ p = x + y)
  (hz : ∃ (z w : ℕ), Prime z ∧ Prime w ∧ p = z - w) : p = 5 :=
sorry

end NUMINAMATH_GPT_unique_prime_sum_diff_l214_21418


namespace NUMINAMATH_GPT_simplify_expression_l214_21413

theorem simplify_expression (x : ℝ) : 
  x^2 * (4 * x^3 - 3 * x + 1) - 6 * (x^3 - 3 * x^2 + 4 * x - 5) = 
  4 * x^5 - 9 * x^3 + 19 * x^2 - 24 * x + 30 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l214_21413


namespace NUMINAMATH_GPT_jills_present_age_l214_21401

-- Define the problem parameters and conditions
variables (H J : ℕ)
axiom cond1 : H + J = 43
axiom cond2 : H - 5 = 2 * (J - 5)

-- State the goal
theorem jills_present_age : J = 16 :=
sorry

end NUMINAMATH_GPT_jills_present_age_l214_21401


namespace NUMINAMATH_GPT_area_difference_triangles_l214_21429

theorem area_difference_triangles
  (A B C F D : Type)
  (angle_FAB_right : true) 
  (angle_ABC_right : true) 
  (AB : Real) (hAB : AB = 5)
  (BC : Real) (hBC : BC = 3)
  (AF : Real) (hAF : AF = 7)
  (area_triangle : A -> B -> C -> Real)
  (angle_bet : A -> D -> F) 
  (angle_bet : B -> D -> C)
  (area_ADF : Real)
  (area_BDC : Real) : (area_ADF - area_BDC = 10) :=
sorry

end NUMINAMATH_GPT_area_difference_triangles_l214_21429


namespace NUMINAMATH_GPT_general_formula_for_sequences_c_seq_is_arithmetic_fn_integer_roots_l214_21457

noncomputable def a_seq (n : ℕ) : ℕ :=
  if h : n > 0 then n else 1

noncomputable def b_seq (n : ℕ) : ℚ :=
  if h : n > 0 then n * (n - 1) / 4 else 0

noncomputable def c_seq (n : ℕ) : ℚ :=
  a_seq n ^ 2 - 4 * b_seq n

theorem general_formula_for_sequences (n : ℕ) (h : n > 0) :
  a_seq n = n ∧ b_seq n = (n * (n - 1)) / 4 :=
sorry

theorem c_seq_is_arithmetic (n : ℕ) (h : n > 0) : 
  ∀ m : ℕ, (h2 : m > 0) -> c_seq (m+1) - c_seq m = 1 :=
sorry

theorem fn_integer_roots (n : ℕ) : 
  ∃ k : ℤ, n = k ^ 2 ∧ k ≠ 0 :=
sorry

end NUMINAMATH_GPT_general_formula_for_sequences_c_seq_is_arithmetic_fn_integer_roots_l214_21457


namespace NUMINAMATH_GPT_maximum_value_of_sum_l214_21499

variables (x y : ℝ)

def s : ℝ := x + y

theorem maximum_value_of_sum (h : s ≤ 9) : s = 9 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_sum_l214_21499


namespace NUMINAMATH_GPT_number_of_pieces_of_tape_l214_21452

variable (length_of_tape : ℝ := 8.8)
variable (overlap : ℝ := 0.5)
variable (total_length : ℝ := 282.7)

theorem number_of_pieces_of_tape : 
  ∃ (N : ℕ), total_length = length_of_tape + (N - 1) * (length_of_tape - overlap) ∧ N = 34 :=
sorry

end NUMINAMATH_GPT_number_of_pieces_of_tape_l214_21452


namespace NUMINAMATH_GPT_eq_or_neg_eq_of_eq_frac_l214_21493

theorem eq_or_neg_eq_of_eq_frac (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h : a^2 + b^3 / a = b^2 + a^3 / b) :
  a = b ∨ a = -b :=
by
  sorry

end NUMINAMATH_GPT_eq_or_neg_eq_of_eq_frac_l214_21493


namespace NUMINAMATH_GPT_length_of_AB_l214_21415

theorem length_of_AB {L : ℝ} (h : 9 * Real.pi * L + 36 * Real.pi = 216 * Real.pi) : L = 20 :=
sorry

end NUMINAMATH_GPT_length_of_AB_l214_21415


namespace NUMINAMATH_GPT_find_positive_number_l214_21488
-- Prove the positive number x that satisfies the condition is 8
theorem find_positive_number (x : ℝ) (hx : 0 < x) :
    x + 8 = 128 * (1 / x) → x = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_positive_number_l214_21488


namespace NUMINAMATH_GPT_movie_length_l214_21461

theorem movie_length (paused_midway : ∃ t : ℝ, t = t ∧ t / 2 = 30) : 
  ∃ total_length : ℝ, total_length = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_movie_length_l214_21461


namespace NUMINAMATH_GPT_larger_number_is_37_l214_21439

-- Defining the conditions
def sum_of_two_numbers (a b : ℕ) : Prop := a + b = 62
def one_is_12_more (a b : ℕ) : Prop := a = b + 12

-- Proof statement
theorem larger_number_is_37 (a b : ℕ) (h₁ : sum_of_two_numbers a b) (h₂ : one_is_12_more a b) : a = 37 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_37_l214_21439


namespace NUMINAMATH_GPT_pirate_treasure_l214_21451

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end NUMINAMATH_GPT_pirate_treasure_l214_21451


namespace NUMINAMATH_GPT_floor_eq_48_iff_l214_21400

-- Define the real number set I to be [8, 49/6)
def I : Set ℝ := { x | 8 ≤ x ∧ x < 49/6 }

-- The main statement to be proven
theorem floor_eq_48_iff (x : ℝ) : (Int.floor (x * Int.floor x) = 48) ↔ x ∈ I := 
by
  sorry

end NUMINAMATH_GPT_floor_eq_48_iff_l214_21400


namespace NUMINAMATH_GPT_number_of_students_in_chemistry_class_l214_21411

variables (students : Finset ℕ) (n : ℕ)
  (x y z cb cp bp c b : ℕ)
  (students_in_total : students.card = 120)
  (chem_bio : cb = 35)
  (bio_phys : bp = 15)
  (chem_phys : cp = 10)
  (total_equation : 120 = x + y + z + cb + bp + cp)
  (chem_equation : c = y + cb + cp)
  (bio_equation : b = x + cb + bp)
  (chem_bio_relation : 4 * b = c)
  (no_all_three_classes : true)

theorem number_of_students_in_chemistry_class : c = 153 :=
  sorry

end NUMINAMATH_GPT_number_of_students_in_chemistry_class_l214_21411


namespace NUMINAMATH_GPT_computer_price_after_six_years_l214_21447

def price_decrease (p_0 : ℕ) (rate : ℚ) (t : ℕ) : ℚ :=
  p_0 * rate ^ (t / 2)

theorem computer_price_after_six_years :
  price_decrease 8100 (2 / 3) 6 = 2400 := by
  sorry

end NUMINAMATH_GPT_computer_price_after_six_years_l214_21447


namespace NUMINAMATH_GPT_scale_model_height_l214_21474

theorem scale_model_height 
  (scale_ratio : ℚ) (actual_height : ℚ)
  (h_ratio : scale_ratio = 1/30)
  (h_actual_height : actual_height = 305) 
  : Int.ceil (actual_height * scale_ratio) = 10 := by
  -- Define variables and the necessary conditions
  let height_of_model: ℚ := actual_height * scale_ratio
  -- Skip the proof steps
  sorry

end NUMINAMATH_GPT_scale_model_height_l214_21474


namespace NUMINAMATH_GPT_pardee_road_length_l214_21419

theorem pardee_road_length (t p : ℕ) (h1 : t = 162 * 1000) (h2 : t = p + 150 * 1000) : p = 12 * 1000 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_pardee_road_length_l214_21419


namespace NUMINAMATH_GPT_cows_eat_husk_l214_21497

theorem cows_eat_husk :
  ∀ (cows : ℕ) (days : ℕ) (husk_per_cow : ℕ),
    cows = 45 →
    days = 45 →
    husk_per_cow = 1 →
    (cows * husk_per_cow = 45) :=
by
  intros cows days husk_per_cow h_cows h_days h_husk_per_cow
  sorry

end NUMINAMATH_GPT_cows_eat_husk_l214_21497


namespace NUMINAMATH_GPT_range_of_x_l214_21486

noncomputable 
def proposition_p (x : ℝ) : Prop := 6 - 3 * x ≥ 0

noncomputable 
def proposition_q (x : ℝ) : Prop := 1 / (x + 1) < 0

theorem range_of_x (x : ℝ) : proposition_p x ∧ ¬proposition_q x → x ∈ Set.Icc (-1 : ℝ) (2 : ℝ) := by
  sorry

end NUMINAMATH_GPT_range_of_x_l214_21486


namespace NUMINAMATH_GPT_find_real_numbers_l214_21471

theorem find_real_numbers :
  ∀ (x y z : ℝ), x^2 - y*z = |y - z| + 1 ∧ y^2 - z*x = |z - x| + 1 ∧ z^2 - x*y = |x - y| + 1 ↔
  (x = 4/3 ∧ y = 4/3 ∧ z = -5/3) ∨
  (x = 4/3 ∧ y = -5/3 ∧ z = 4/3) ∨
  (x = -5/3 ∧ y = 4/3 ∧ z = 4/3) ∨
  (x = -4/3 ∧ y = -4/3 ∧ z = 5/3) ∨
  (x = -4/3 ∧ y = 5/3 ∧ z = -4/3) ∨
  (x = 5/3 ∧ y = -4/3 ∧ z = -4/3) :=
by
  sorry

end NUMINAMATH_GPT_find_real_numbers_l214_21471


namespace NUMINAMATH_GPT_hotdog_eating_ratio_l214_21458

variable (rate_first rate_second rate_third total_hotdogs time_minutes : ℕ)
variable (rate_ratio : ℕ)

def rate_first_eq : rate_first = 10 := by sorry
def rate_second_eq : rate_second = 3 * rate_first := by sorry
def total_hotdogs_eq : total_hotdogs = 300 := by sorry
def time_minutes_eq : time_minutes = 5 := by sorry
def rate_third_eq : rate_third = total_hotdogs / time_minutes := by sorry

theorem hotdog_eating_ratio :
  rate_ratio = rate_third / rate_second :=
  by sorry

end NUMINAMATH_GPT_hotdog_eating_ratio_l214_21458


namespace NUMINAMATH_GPT_avg_last_three_numbers_l214_21444

-- Definitions of conditions
def avg_seven_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.sum / 7 = 60)

def avg_first_four_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.take 4).sum / 4 = 55

-- Proof statement
theorem avg_last_three_numbers (numbers : List ℝ) (h_len : numbers.length = 7)
  (h1 : avg_seven_numbers numbers h_len)
  (h2 : avg_first_four_numbers numbers h_len) :
  (numbers.drop 4).sum / 3 = 200 / 3 :=
sorry

end NUMINAMATH_GPT_avg_last_three_numbers_l214_21444


namespace NUMINAMATH_GPT_kittens_total_number_l214_21428

theorem kittens_total_number (W L H R : ℕ) (k : ℕ) 
  (h1 : W = 500) 
  (h2 : L = 80) 
  (h3 : H = 200) 
  (h4 : L + H + R = W) 
  (h5 : 40 * k ≤ R) 
  (h6 : R ≤ 50 * k) 
  (h7 : ∀ m, m ≠ 4 → m ≠ 6 → m ≠ k →
        40 * m ≤ R → R ≤ 50 * m → False) : 
  k = 5 ∧ 2 + 4 + k = 11 := 
by {
  -- The proof would go here
  sorry 
}

end NUMINAMATH_GPT_kittens_total_number_l214_21428


namespace NUMINAMATH_GPT_sum_three_numbers_l214_21459

theorem sum_three_numbers 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : ab + bc + ca = 72) : 
  a + b + c = 14 := 
by 
  sorry

end NUMINAMATH_GPT_sum_three_numbers_l214_21459


namespace NUMINAMATH_GPT_fraction_sum_neg_one_l214_21405

variable (a : ℚ)

theorem fraction_sum_neg_one (h : a ≠ 1/2) : (a / (1 - 2 * a)) + ((a - 1) / (1 - 2 * a)) = -1 := 
sorry

end NUMINAMATH_GPT_fraction_sum_neg_one_l214_21405


namespace NUMINAMATH_GPT_trapezium_area_l214_21463

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem trapezium_area :
  (1/2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
  sorry

end NUMINAMATH_GPT_trapezium_area_l214_21463


namespace NUMINAMATH_GPT_sine_addition_l214_21404

noncomputable def sin_inv_45 := Real.arcsin (4 / 5)
noncomputable def tan_inv_12 := Real.arctan (1 / 2)

theorem sine_addition :
  Real.sin (sin_inv_45 + tan_inv_12) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end NUMINAMATH_GPT_sine_addition_l214_21404


namespace NUMINAMATH_GPT_kerosene_cost_l214_21448

theorem kerosene_cost (A B C : ℝ)
  (h1 : A = B)
  (h2 : C = A / 2)
  (h3 : C * 2 = 24 / 100) :
  24 = 24 := 
sorry

end NUMINAMATH_GPT_kerosene_cost_l214_21448


namespace NUMINAMATH_GPT_monkey_total_distance_l214_21438

theorem monkey_total_distance :
  let speedRunning := 15
  let timeRunning := 5
  let speedSwinging := 10
  let timeSwinging := 10
  let distanceRunning := speedRunning * timeRunning
  let distanceSwinging := speedSwinging * timeSwinging
  let totalDistance := distanceRunning + distanceSwinging
  totalDistance = 175 :=
by
  sorry

end NUMINAMATH_GPT_monkey_total_distance_l214_21438


namespace NUMINAMATH_GPT_incorrect_conclusion_l214_21432

variable {a b c : ℝ}

theorem incorrect_conclusion
  (h1 : a^2 + a * b = c)
  (h2 : a * b + b^2 = c + 5) :
  ¬(2 * c + 5 < 0) ∧ ¬(∃ k, a^2 - b^2 ≠ k) ∧ ¬(a = b ∨ a = -b) ∧ ¬(b / a > 1) :=
by sorry

end NUMINAMATH_GPT_incorrect_conclusion_l214_21432


namespace NUMINAMATH_GPT_eval_expr_l214_21431

theorem eval_expr : 4 * (8 - 3 + 2) / 2 = 14 := 
by
  sorry

end NUMINAMATH_GPT_eval_expr_l214_21431


namespace NUMINAMATH_GPT_percentage_increase_in_area_l214_21489

-- Defining the lengths and widths in terms of real numbers
variables (L W : ℝ)

-- Defining the new lengths and widths
def new_length := 1.2 * L
def new_width := 1.2 * W

-- Original area of the rectangle
def original_area := L * W

-- New area of the rectangle
def new_area := new_length L * new_width W

-- Proof statement for the percentage increase
theorem percentage_increase_in_area : 
  ((new_area L W - original_area L W) / original_area L W) * 100 = 44 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_area_l214_21489


namespace NUMINAMATH_GPT_ceiling_example_l214_21420

/-- Lean 4 statement of the proof problem:
    Prove that ⌈4 (8 - 1/3)⌉ = 31.
-/
theorem ceiling_example : Int.ceil (4 * (8 - (1 / 3 : ℝ))) = 31 := 
by
  sorry

end NUMINAMATH_GPT_ceiling_example_l214_21420


namespace NUMINAMATH_GPT_amount_distribution_l214_21484

theorem amount_distribution :
  ∃ (P Q R S T : ℝ), 
    (P + Q + R + S + T = 24000) ∧ 
    (R = (3 / 5) * (P + Q)) ∧ 
    (S = 0.45 * 24000) ∧ 
    (T = (1 / 2) * R) ∧ 
    (P + Q = 7000) ∧ 
    (R = 4200) ∧ 
    (S = 10800) ∧ 
    (T = 2100) :=
by
  sorry

end NUMINAMATH_GPT_amount_distribution_l214_21484


namespace NUMINAMATH_GPT_total_balloons_l214_21425

theorem total_balloons (allan_balloons : ℕ) (jake_balloons : ℕ)
  (h_allan : allan_balloons = 2)
  (h_jake : jake_balloons = 1) :
  allan_balloons + jake_balloons = 3 :=
by 
  -- Provide proof here
  sorry

end NUMINAMATH_GPT_total_balloons_l214_21425


namespace NUMINAMATH_GPT_g_at_10_l214_21495

noncomputable def g (n : ℕ) : ℝ := sorry

axiom g_definition : g 2 = 4
axiom g_recursive : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = (3 * g (2 * m) + g (2 * n)) / 4

theorem g_at_10 : g 10 = 64 := sorry

end NUMINAMATH_GPT_g_at_10_l214_21495


namespace NUMINAMATH_GPT_largest_common_element_l214_21449

theorem largest_common_element (S1 S2 : ℕ → ℕ) (a_max : ℕ) :
  (∀ n, S1 n = 2 + 5 * n → ∃ k, S2 k = 3 + 8 * k ∧ S1 n = S2 k) →
  (147 < a_max) →
  ∀ m, (m < a_max → (∀ n, S1 n = 2 + 5 * n → ∃ k, S2 k = 3 + 8 * k ∧ S1 n = S2 k) → 147 = 27 + 40 * 3) :=
sorry

end NUMINAMATH_GPT_largest_common_element_l214_21449


namespace NUMINAMATH_GPT_expression_meaningful_l214_21434

theorem expression_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_meaningful_l214_21434


namespace NUMINAMATH_GPT_carlson_fraction_l214_21469

-- Define variables
variables (n m k p T : ℝ)

theorem carlson_fraction (h1 : k = 0.6 * n)
                         (h2 : p = 2.5 * m)
                         (h3 : T = n * m + k * p) :
                         k * p / T = 3 / 5 := by
  -- Omitted proof
  sorry

end NUMINAMATH_GPT_carlson_fraction_l214_21469


namespace NUMINAMATH_GPT_temperature_celsius_range_l214_21470

theorem temperature_celsius_range (C : ℝ) :
  (∀ C : ℝ, let F_approx := 2 * C + 30;
             let F_exact := (9 / 5) * C + 32;
             abs ((2 * C + 30 - ((9 / 5) * C + 32)) / ((9 / 5) * C + 32)) ≤ 0.05) →
  (40 / 29) ≤ C ∧ C ≤ (360 / 11) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_temperature_celsius_range_l214_21470


namespace NUMINAMATH_GPT_four_digit_3_or_6_l214_21460

theorem four_digit_3_or_6 : 
  ∃ n : ℕ, n = 16 ∧ 
    (∀ (x : ℕ), 
      (x >= 1000 ∧ x < 10000) → 
      (∀ d ∈ [3, 6], ∃ (a b c e : ℕ), 
        (a = 3 ∨ a = 6) ∧
        (b = 3 ∨ b = 6) ∧
        (c = 3 ∨ c = 6) ∧
        (e = 3 ∨ e = 6) ∧ 
        x = a * 1000 + b * 100 + c * 10 + e)
    )
:= 
by
  sorry

end NUMINAMATH_GPT_four_digit_3_or_6_l214_21460


namespace NUMINAMATH_GPT_investor_difference_l214_21412

def investment_A : ℝ := 300
def investment_B : ℝ := 200
def rate_A : ℝ := 0.30
def rate_B : ℝ := 0.50

theorem investor_difference :
  ((investment_A * (1 + rate_A)) - (investment_B * (1 + rate_B))) = 90 := 
by
  sorry

end NUMINAMATH_GPT_investor_difference_l214_21412


namespace NUMINAMATH_GPT_ryan_flyers_l214_21482

theorem ryan_flyers (total_flyers : ℕ) (alyssa_flyers : ℕ) (scott_flyers : ℕ) (belinda_percentage : ℚ) (belinda_flyers : ℕ) (ryan_flyers : ℕ)
  (htotal : total_flyers = 200)
  (halyssa : alyssa_flyers = 67)
  (hscott : scott_flyers = 51)
  (hbelinda_percentage : belinda_percentage = 0.20)
  (hbelinda : belinda_flyers = belinda_percentage * total_flyers)
  (hryan : ryan_flyers = total_flyers - (alyssa_flyers + scott_flyers + belinda_flyers)) :
  ryan_flyers = 42 := by
    sorry

end NUMINAMATH_GPT_ryan_flyers_l214_21482


namespace NUMINAMATH_GPT_sequence_an_properties_l214_21487

theorem sequence_an_properties
(S : ℕ → ℝ) (a : ℕ → ℝ)
(h_mean : ∀ n, 2 * a n = S n + 2) :
a 1 = 2 ∧ a 2 = 4 ∧ ∀ n, a n = 2 ^ n :=
by
  sorry

end NUMINAMATH_GPT_sequence_an_properties_l214_21487


namespace NUMINAMATH_GPT_bonifac_distance_l214_21408

/-- Given the conditions provided regarding the paths of Pankrác, Servác, and Bonifác,
prove that the total distance Bonifác walked is 625 meters. -/
theorem bonifac_distance
  (path_Pankrac : ℕ)  -- distance of Pankráč's path in segments
  (meters_Pankrac : ℕ)  -- distance Pankráč walked in meters
  (path_Bonifac : ℕ)  -- distance of Bonifác's path in segments
  (meters_per_segment : ℚ)  -- meters per segment walked
  (Hp : path_Pankrac = 40)  -- Pankráč's path in segments
  (Hm : meters_Pankrac = 500)  -- Pankráč walked 500 meters
  (Hms : meters_per_segment = 500 / 40)  -- meters per segment
  (Hb : path_Bonifac = 50)  -- Bonifác's path in segments
  : path_Bonifac * meters_per_segment = 625 := sorry

end NUMINAMATH_GPT_bonifac_distance_l214_21408


namespace NUMINAMATH_GPT_larger_square_side_length_l214_21492

theorem larger_square_side_length (s1 s2 : ℝ) (h1 : s1 = 5) (h2 : s2 = s1 * 3) (a1 a2 : ℝ) (h3 : a1 = s1^2) (h4 : a2 = s2^2) : s2 = 15 := 
by
  sorry

end NUMINAMATH_GPT_larger_square_side_length_l214_21492


namespace NUMINAMATH_GPT_smallest_palindrome_in_base3_and_base5_l214_21462

def is_palindrome_base (b n : ℕ) : Prop :=
  let digits := n.digits b
  digits = digits.reverse

theorem smallest_palindrome_in_base3_and_base5 :
  ∃ n : ℕ, n > 10 ∧ is_palindrome_base 3 n ∧ is_palindrome_base 5 n ∧ n = 20 :=
by
  sorry

end NUMINAMATH_GPT_smallest_palindrome_in_base3_and_base5_l214_21462


namespace NUMINAMATH_GPT_time_to_office_l214_21402

theorem time_to_office (S T : ℝ) (h1 : T > 0) (h2 : S > 0) 
    (h : S * (T + 15) = (4/5) * S * T) :
    T = 75 := by
  sorry

end NUMINAMATH_GPT_time_to_office_l214_21402


namespace NUMINAMATH_GPT_sandy_correct_sums_l214_21485

variables (x y : ℕ)

theorem sandy_correct_sums :
  (x + y = 30) →
  (3 * x - 2 * y = 50) →
  x = 22 :=
by
  intro h1 h2
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_sandy_correct_sums_l214_21485


namespace NUMINAMATH_GPT_evaluate_expression_at_x_eq_2_l214_21475

theorem evaluate_expression_at_x_eq_2 :
  (3 * 2 + 4)^2 = 100 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_x_eq_2_l214_21475


namespace NUMINAMATH_GPT_transformed_polynomial_l214_21441

noncomputable def P : Polynomial ℝ := Polynomial.C 9 + Polynomial.X ^ 3 - 4 * Polynomial.X ^ 2 

noncomputable def Q : Polynomial ℝ := Polynomial.C 243 + Polynomial.X ^ 3 - 12 * Polynomial.X ^ 2 

theorem transformed_polynomial :
  ∀ (r : ℝ), Polynomial.aeval r P = 0 → Polynomial.aeval (3 * r) Q = 0 := 
by
  sorry

end NUMINAMATH_GPT_transformed_polynomial_l214_21441


namespace NUMINAMATH_GPT_Sasha_can_paint_8x9_Sasha_cannot_paint_8x10_l214_21442

-- Definition of the problem conditions
def initially_painted (m n : ℕ) : Prop :=
  ∃ i j : ℕ, i < m ∧ j < n
  
def odd_painted_neighbors (m n : ℕ) : Prop :=
  ∀ i j : ℕ, i < m ∧ j < n →
  (∃ k l : ℕ, (k = i+1 ∨ k = i-1 ∨ l = j+1 ∨ l = j-1) ∧ k < m ∧ l < n → true)

-- Part (a): 8x9 rectangle
theorem Sasha_can_paint_8x9 : (initially_painted 8 9 ∧ odd_painted_neighbors 8 9) → ∀ (i j : ℕ), i < 8 ∧ j < 9 :=
by
  -- Proof here
  sorry

-- Part (b): 8x10 rectangle
theorem Sasha_cannot_paint_8x10 : (initially_painted 8 10 ∧ odd_painted_neighbors 8 10) → ¬ (∀ (i j : ℕ), i < 8 ∧ j < 10) :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_Sasha_can_paint_8x9_Sasha_cannot_paint_8x10_l214_21442


namespace NUMINAMATH_GPT_find_intersection_l214_21436

noncomputable def f (n : ℕ) : ℕ := 2 * n + 1

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {3, 4, 5, 6, 7}

def f_set (s : Set ℕ) : Set ℕ := {n | f n ∈ s}

theorem find_intersection : f_set A ∩ f_set B = {1, 2} := 
by {
  sorry
}

end NUMINAMATH_GPT_find_intersection_l214_21436


namespace NUMINAMATH_GPT_problem_l214_21466

variable (x y : ℝ)

theorem problem
  (h : (3 * x + 1) ^ 2 + |y - 3| = 0) :
  (x + 2 * y) * (x - 2 * y) + (x + 2 * y) ^ 2 - x * (2 * x + 3 * y) = -1 :=
sorry

end NUMINAMATH_GPT_problem_l214_21466


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l214_21456

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x
noncomputable def g (a x : ℝ) : ℝ := f a x + 2 * x

theorem problem1 (a : ℝ) : a = 1 → ∀ x : ℝ, f 1 x = x^2 - 3 * x + Real.log x → 
  (∀ x : ℝ, f 1 1 = -2) :=
by sorry

theorem problem2 (a : ℝ) (h : 0 < a) : (∀ x : ℝ, 1 ≤ x → x ≤ Real.exp 1 → f a x ≥ -2) → a ≥ 1 :=
by sorry

theorem problem3 (a : ℝ) : (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f a x1 + 2 * x1 < f a x2 + 2 * x2) → 0 ≤ a ∧ a ≤ 8 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l214_21456


namespace NUMINAMATH_GPT_distance_from_center_to_chord_l214_21467

theorem distance_from_center_to_chord (a b : ℝ) : 
  ∃ d : ℝ, d = (1/4) * |a - b| := 
sorry

end NUMINAMATH_GPT_distance_from_center_to_chord_l214_21467


namespace NUMINAMATH_GPT_complement_union_correct_l214_21421

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {2, 3, 4})
variable (hB : B = {1, 4})

theorem complement_union_correct :
  (compl A ∪ B) = {1, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_correct_l214_21421


namespace NUMINAMATH_GPT_circleEquation_and_pointOnCircle_l214_21422

-- Definition of the Cartesian coordinate system and the circle conditions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def inSecondQuadrant (p : ℝ × ℝ) := p.1 < 0 ∧ p.2 > 0

def tangentToLine (C : Circle) (line : ℝ → ℝ) (tangentPoint : ℝ × ℝ) :=
  let centerToLineDistance := (abs (C.center.1 - C.center.2)) / Real.sqrt 2
  C.radius = centerToLineDistance ∧ tangentPoint = (0, 0)

-- Main statements to prove
theorem circleEquation_and_pointOnCircle :
  ∃ C : Circle, ∃ Q : ℝ × ℝ,
    inSecondQuadrant C.center ∧
    C.radius = 2 * Real.sqrt 2 ∧
    tangentToLine C (fun x => x) (0, 0) ∧
    ((∃ p : ℝ × ℝ, p = (-2, 2) ∧ C = Circle.mk p (2 * Real.sqrt 2) ∧
      (∀ x y : ℝ, ((x + 2)^2 + (y - 2)^2 = 8))) ∧
    (Q = (4/5, 12/5) ∧
      ((Q.1 + 2)^2 + (Q.2 - 2)^2 = 8) ∧
      Real.sqrt ((Q.1 - 4)^2 + Q.2^2) = 4))
    := sorry

end NUMINAMATH_GPT_circleEquation_and_pointOnCircle_l214_21422


namespace NUMINAMATH_GPT_first_divisor_is_13_l214_21468

theorem first_divisor_is_13 (x : ℤ) (h : (377 / x) / 29 * (1/4 : ℚ) / 2 = (1/8 : ℚ)) : x = 13 := by
  sorry

end NUMINAMATH_GPT_first_divisor_is_13_l214_21468


namespace NUMINAMATH_GPT_complete_square_add_term_l214_21424

theorem complete_square_add_term (x : ℝ) :
  ∃ (c : ℝ), (c = 4 * x ^ 4 ∨ c = 4 * x ∨ c = -4 * x ∨ c = -1 ∨ c = -4 * x ^2) ∧
  (4 * x ^ 2 + 1 + c) * (4 * x ^ 2 + 1 + c) = (2 * x + 1) * (2 * x + 1) :=
sorry

end NUMINAMATH_GPT_complete_square_add_term_l214_21424


namespace NUMINAMATH_GPT_geometric_series_second_term_l214_21437

theorem geometric_series_second_term 
  (r : ℚ) (S : ℚ) (a : ℚ) (second_term : ℚ)
  (h1 : r = 1 / 4)
  (h2 : S = 16)
  (h3 : S = a / (1 - r))
  : second_term = a * r := 
sorry

end NUMINAMATH_GPT_geometric_series_second_term_l214_21437


namespace NUMINAMATH_GPT_players_either_left_handed_or_throwers_l214_21435

theorem players_either_left_handed_or_throwers (total_players throwers : ℕ) (h1 : total_players = 70) (h2 : throwers = 34) (h3 : ∀ n, n = total_players - throwers → 1 / 3 * n = n / 3) :
  ∃ n, n = 46 := 
sorry

end NUMINAMATH_GPT_players_either_left_handed_or_throwers_l214_21435


namespace NUMINAMATH_GPT_price_of_baseball_cards_l214_21472

theorem price_of_baseball_cards 
    (packs_Digimon : ℕ)
    (price_per_pack : ℝ)
    (total_spent : ℝ)
    (total_cost_Digimon : ℝ) 
    (price_baseball_deck : ℝ) 
    (h1 : packs_Digimon = 4) 
    (h2 : price_per_pack = 4.45) 
    (h3 : total_spent = 23.86) 
    (h4 : total_cost_Digimon = packs_Digimon * price_per_pack) 
    (h5 : price_baseball_deck = total_spent - total_cost_Digimon) : 
    price_baseball_deck = 6.06 :=
sorry

end NUMINAMATH_GPT_price_of_baseball_cards_l214_21472


namespace NUMINAMATH_GPT_divides_power_of_odd_l214_21403

theorem divides_power_of_odd (k : ℕ) (hk : k % 2 = 1) (n : ℕ) (hn : n ≥ 1) : 2^(n + 2) ∣ (k^(2^n) - 1) :=
by
  sorry

end NUMINAMATH_GPT_divides_power_of_odd_l214_21403


namespace NUMINAMATH_GPT_solve_for_x_l214_21498

theorem solve_for_x :
  (∀ y : ℝ, 10 * x * y - 15 * y + 4 * x - 6 = 0) ↔ x = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l214_21498


namespace NUMINAMATH_GPT_avg_weight_l214_21478

theorem avg_weight (A B C : ℝ)
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 43)
  (h3 : B = 31) :
  (A + B + C) / 3 = 45 :=
by sorry

end NUMINAMATH_GPT_avg_weight_l214_21478


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l214_21443

theorem quadratic_has_distinct_real_roots {m : ℝ} (hm : m > 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + x1 - 2 = m) ∧ (x2^2 + x2 - 2 = m) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l214_21443


namespace NUMINAMATH_GPT_remaining_amount_after_purchase_l214_21481

def initial_amount : ℕ := 78
def kite_cost : ℕ := 8
def frisbee_cost : ℕ := 9

theorem remaining_amount_after_purchase : initial_amount - kite_cost - frisbee_cost = 61 := by
  sorry

end NUMINAMATH_GPT_remaining_amount_after_purchase_l214_21481


namespace NUMINAMATH_GPT_abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one_l214_21464

theorem abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one 
  (a b c : ℝ) 
  (h1 : a * b * c = 1) 
  (h2 : a + b + c = 1 / a + 1 / b + 1 / c) : 
  (a = 1) ∨ (b = 1) ∨ (c = 1) :=
by
  sorry

end NUMINAMATH_GPT_abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one_l214_21464


namespace NUMINAMATH_GPT_pie_price_l214_21453

theorem pie_price (cakes_sold : ℕ) (cake_price : ℕ) (cakes_total_earnings : ℕ)
                  (pies_sold : ℕ) (total_earnings : ℕ) (price_per_pie : ℕ)
                  (H1 : cakes_sold = 453)
                  (H2 : cake_price = 12)
                  (H3 : pies_sold = 126)
                  (H4 : total_earnings = 6318)
                  (H5 : cakes_total_earnings = cakes_sold * cake_price)
                  (H6 : price_per_pie * pies_sold = total_earnings - cakes_total_earnings) :
    price_per_pie = 7 := by
    sorry

end NUMINAMATH_GPT_pie_price_l214_21453


namespace NUMINAMATH_GPT_car_speed_second_hour_l214_21427

variable (x : ℝ)
variable (s1 : ℝ := 100)
variable (avg_speed : ℝ := 90)
variable (total_time : ℝ := 2)

-- The Lean statement equivalent to the problem
theorem car_speed_second_hour : (100 + x) / 2 = 90 → x = 80 := by 
  intro h
  have h₁ : 2 * 90 = 100 + x := by 
    linarith [h]
  linarith [h₁]

end NUMINAMATH_GPT_car_speed_second_hour_l214_21427


namespace NUMINAMATH_GPT_max_n_value_l214_21445

noncomputable def max_n_avoid_repetition : ℕ :=
sorry

theorem max_n_value : max_n_avoid_repetition = 155 :=
by
  -- Assume factorial reciprocals range from 80 to 99
  -- We show no n-digit segments are repeated in such range while n <= 155
  sorry

end NUMINAMATH_GPT_max_n_value_l214_21445


namespace NUMINAMATH_GPT_time_relationship_l214_21480

variable (T x : ℝ)
variable (h : T = x + (2/6) * x)

theorem time_relationship : T = (4/3) * x := by 
sorry

end NUMINAMATH_GPT_time_relationship_l214_21480


namespace NUMINAMATH_GPT_max_ab_ac_bc_l214_21473

theorem max_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 6) : 
    ab + ac + bc <= 8 :=
sorry

end NUMINAMATH_GPT_max_ab_ac_bc_l214_21473


namespace NUMINAMATH_GPT_cupcake_cost_l214_21490

def initialMoney : ℝ := 20
def moneyFromMother : ℝ := 2 * initialMoney
def totalMoney : ℝ := initialMoney + moneyFromMother
def costPerBoxOfCookies : ℝ := 3
def numberOfBoxesOfCookies : ℝ := 5
def costOfCookies : ℝ := costPerBoxOfCookies * numberOfBoxesOfCookies
def moneyAfterCookies : ℝ := totalMoney - costOfCookies
def moneyLeftAfterCupcakes : ℝ := 30
def numberOfCupcakes : ℝ := 10

noncomputable def costPerCupcake : ℝ := 
  (moneyAfterCookies - moneyLeftAfterCupcakes) / numberOfCupcakes

theorem cupcake_cost :
  costPerCupcake = 1.50 :=
by 
  sorry

end NUMINAMATH_GPT_cupcake_cost_l214_21490


namespace NUMINAMATH_GPT_domain_of_f_l214_21406

noncomputable def f (x : ℝ) : ℝ := (1 / (x - 5)) + (1 / (x^2 - 4)) + (1 / (x^3 - 27))

theorem domain_of_f :
  ∀ x : ℝ, x ≠ 5 ∧ x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 3 ↔
          ∃ y : ℝ, f y = f x :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l214_21406


namespace NUMINAMATH_GPT_each_wolf_needs_to_kill_one_deer_l214_21416

-- Conditions
def wolves_out_hunting : ℕ := 4
def additional_wolves : ℕ := 16
def wolves_total : ℕ := wolves_out_hunting + additional_wolves
def meat_per_wolf_per_day : ℕ := 8
def days_no_hunt : ℕ := 5
def meat_per_deer : ℕ := 200

-- Calculate total meat needed for all wolves over five days.
def total_meat_needed : ℕ := wolves_total * meat_per_wolf_per_day * days_no_hunt
-- Calculate total number of deer needed to meet the meat requirement.
def deer_needed : ℕ := total_meat_needed / meat_per_deer
-- Calculate number of deer each hunting wolf needs to kill.
def deer_per_wolf : ℕ := deer_needed / wolves_out_hunting

-- The proof statement
theorem each_wolf_needs_to_kill_one_deer : deer_per_wolf = 1 := 
by { sorry }

end NUMINAMATH_GPT_each_wolf_needs_to_kill_one_deer_l214_21416


namespace NUMINAMATH_GPT_exists_positive_x_for_inequality_l214_21483

-- Define the problem conditions and the final proof goal.
theorem exists_positive_x_for_inequality (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^2 + |x + a| < 2) ↔ a ∈ Set.Ico (-9/4 : ℝ) (2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_exists_positive_x_for_inequality_l214_21483


namespace NUMINAMATH_GPT_functional_equation_solution_l214_21417

-- Define the function
def f : ℝ → ℝ := sorry

-- The main theorem to prove
theorem functional_equation_solution :
  (∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)) → (∀ x : ℝ, f x = x - 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l214_21417


namespace NUMINAMATH_GPT_series_sum_proof_l214_21494

noncomputable def infinite_series_sum : ℝ :=
  ∑' n : ℕ, if n % 3 = 0 then 1 / (27 ^ (n / 3)) * (5 / 9) else 0

theorem series_sum_proof : infinite_series_sum = 15 / 26 :=
  sorry

end NUMINAMATH_GPT_series_sum_proof_l214_21494


namespace NUMINAMATH_GPT_elevation_above_sea_level_mauna_kea_correct_total_height_mauna_kea_correct_elevation_mount_everest_correct_l214_21450

-- Define the initial conditions
def sea_level_drop : ℝ := 397
def submerged_depth_initial : ℝ := 5000
def height_diff_mauna_kea_everest : ℝ := 358

-- Define intermediate calculations based on conditions
def submerged_depth_adjusted : ℝ := submerged_depth_initial - sea_level_drop
def total_height_mauna_kea : ℝ := 2 * submerged_depth_adjusted
def elevation_above_sea_level_mauna_kea : ℝ := total_height_mauna_kea - submerged_depth_initial
def elevation_mount_everest : ℝ := total_height_mauna_kea - height_diff_mauna_kea_everest

-- Define the proof statements
theorem elevation_above_sea_level_mauna_kea_correct :
  elevation_above_sea_level_mauna_kea = 4206 := by
  sorry

theorem total_height_mauna_kea_correct :
  total_height_mauna_kea = 9206 := by
  sorry

theorem elevation_mount_everest_correct :
  elevation_mount_everest = 8848 := by
  sorry

end NUMINAMATH_GPT_elevation_above_sea_level_mauna_kea_correct_total_height_mauna_kea_correct_elevation_mount_everest_correct_l214_21450


namespace NUMINAMATH_GPT_simplify_product_of_fractions_l214_21423

theorem simplify_product_of_fractions :
  (25 / 24) * (18 / 35) * (56 / 45) = (50 / 3) :=
by sorry

end NUMINAMATH_GPT_simplify_product_of_fractions_l214_21423


namespace NUMINAMATH_GPT_sin_three_pi_four_minus_alpha_l214_21446

theorem sin_three_pi_four_minus_alpha 
  (α : ℝ) 
  (h₁ : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (3 * π / 4 - α) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_three_pi_four_minus_alpha_l214_21446


namespace NUMINAMATH_GPT_number_of_people_in_tour_l214_21455

theorem number_of_people_in_tour (x : ℕ) : 
  (x ≤ 25 ∧ 100 * x = 2700 ∨ 
  (x > 25 ∧ 
   (100 - 2 * (x - 25)) * x = 2700 ∧ 
   70 ≤ 100 - 2 * (x - 25))) → 
  x = 30 := 
by
  sorry

end NUMINAMATH_GPT_number_of_people_in_tour_l214_21455


namespace NUMINAMATH_GPT_find_slope_l214_21454

noncomputable def slope_of_first_line
    (m : ℝ)
    (intersect_point : ℝ × ℝ)
    (slope_second_line : ℝ)
    (x_intercept_distance : ℝ) 
    : Prop :=
  let (x₀, y₀) := intersect_point
  let x_intercept_first := (40 * m - 30) / m
  let x_intercept_second := 35
  abs (x_intercept_first - x_intercept_second) = x_intercept_distance

theorem find_slope : ∃ m : ℝ, slope_of_first_line m (40, 30) 6 10 :=
by
  use 2
  sorry

end NUMINAMATH_GPT_find_slope_l214_21454
