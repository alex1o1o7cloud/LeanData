import Mathlib

namespace log_product_l2180_218033

theorem log_product : (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 2 := by
  sorry

end log_product_l2180_218033


namespace students_taking_neither_l2180_218056

theorem students_taking_neither (total_students music art science music_and_art music_and_science art_and_science three_subjects : ℕ)
  (h1 : total_students = 800)
  (h2 : music = 80)
  (h3 : art = 60)
  (h4 : science = 50)
  (h5 : music_and_art = 30)
  (h6 : music_and_science = 25)
  (h7 : art_and_science = 20)
  (h8 : three_subjects = 15) :
  total_students - (music + art + science - music_and_art - music_and_science - art_and_science + three_subjects) = 670 :=
by sorry

end students_taking_neither_l2180_218056


namespace non_rain_hours_correct_l2180_218021

def total_hours : ℕ := 9
def rain_hours : ℕ := 4

theorem non_rain_hours_correct : (total_hours - rain_hours) = 5 := 
by
  sorry

end non_rain_hours_correct_l2180_218021


namespace fewest_cookies_by_ben_l2180_218099

noncomputable def cookie_problem : Prop :=
  let ana_area := 4 * Real.pi
  let ben_area := 9
  let carol_area := Real.sqrt (5 * (5 + 2 * Real.sqrt 5))
  let dave_area := 3.375 * Real.sqrt 3
  let dough := ana_area * 10
  let ana_cookies := dough / ana_area
  let ben_cookies := dough / ben_area
  let carol_cookies := dough / carol_area
  let dave_cookies := dough / dave_area
  ben_cookies < ana_cookies ∧ ben_cookies < carol_cookies ∧ ben_cookies < dave_cookies

theorem fewest_cookies_by_ben : cookie_problem := by
  sorry

end fewest_cookies_by_ben_l2180_218099


namespace algebraic_expression_simplification_l2180_218060

theorem algebraic_expression_simplification :
  0.25 * (-1 / 2) ^ (-4 : ℝ) - 4 / (Real.sqrt 5 - 1) ^ (0 : ℝ) - (1 / 16) ^ (-1 / 2 : ℝ) = -4 :=
by
  sorry

end algebraic_expression_simplification_l2180_218060


namespace Delaney_missed_bus_by_l2180_218009

def time_in_minutes (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

def Delaney_start_time : ℕ := time_in_minutes 7 50
def bus_departure_time : ℕ := time_in_minutes 8 0
def travel_duration : ℕ := 30

theorem Delaney_missed_bus_by :
  Delaney_start_time + travel_duration - bus_departure_time = 20 :=
by
  sorry

end Delaney_missed_bus_by_l2180_218009


namespace probability_bypass_kth_intersection_l2180_218063

variable (n k : ℕ)

def P (n k : ℕ) : ℚ := (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2

theorem probability_bypass_kth_intersection :
  P n k = (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 :=
by
  sorry

end probability_bypass_kth_intersection_l2180_218063


namespace range_of_a_l2180_218005

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 4 * x + a ≥ 0) → a ≥ 4 :=
by
  sorry

end range_of_a_l2180_218005


namespace sequence_periodicity_l2180_218068

theorem sequence_periodicity (a : ℕ → ℚ) (h1 : a 1 = 6 / 7)
  (h_rec : ∀ n, 0 ≤ a n ∧ a n < 1 → a (n+1) = if a n ≤ 1/2 then 2 * a n else 2 * a n - 1) :
  a 2017 = 6 / 7 :=
  sorry

end sequence_periodicity_l2180_218068


namespace abs_five_minus_e_l2180_218088

noncomputable def e : ℝ := 2.718

theorem abs_five_minus_e : |5 - e| = 2.282 := 
by 
    -- Proof is omitted 
    sorry

end abs_five_minus_e_l2180_218088


namespace intersection_of_A_and_B_l2180_218074

-- Define sets A and B
def A : Set ℝ := { x | x > -1 }
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- The proof statement
theorem intersection_of_A_and_B : A ∩ B = { x | -1 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l2180_218074


namespace sum_of_c_n_l2180_218073

-- Define the sequence {b_n}
def b : ℕ → ℕ
| 0       => 1
| (n + 1) => 2 * b n + 3

-- Define the sequence {a_n}
def a (n : ℕ) : ℕ := 2 * n + 1

-- Define the sequence {c_n}
def c (n : ℕ) : ℚ := (a n) / (b n + 3)

-- Define the sum of the first n terms of {c_n}
def T (n : ℕ) : ℚ := (Finset.range n).sum (λ i => c i)

-- Theorem to prove
theorem sum_of_c_n : ∀ (n : ℕ), T n = (3 / 2 : ℚ) - ((2 * n + 3) / 2^(n + 1)) :=
by
  sorry

end sum_of_c_n_l2180_218073


namespace find_f8_l2180_218057

theorem find_f8 (f : ℕ → ℕ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) : f 8 = 26 :=
by
  sorry

end find_f8_l2180_218057


namespace gcd_lcm_product_eq_l2180_218097

theorem gcd_lcm_product_eq (a b : ℕ) : gcd a b * lcm a b = a * b := by
  sorry

example : ∃ (a b : ℕ), a = 30 ∧ b = 75 ∧ gcd a b * lcm a b = a * b :=
  ⟨30, 75, rfl, rfl, gcd_lcm_product_eq 30 75⟩

end gcd_lcm_product_eq_l2180_218097


namespace perimeter_difference_l2180_218064

-- Definitions for the conditions
def num_stakes_sheep : ℕ := 96
def interval_sheep : ℕ := 10
def num_stakes_horse : ℕ := 82
def interval_horse : ℕ := 20

-- Definition for the perimeters
def perimeter_sheep : ℕ := num_stakes_sheep * interval_sheep
def perimeter_horse : ℕ := num_stakes_horse * interval_horse

-- Definition for the target difference
def target_difference : ℕ := 680

-- The theorem stating the proof problem
theorem perimeter_difference : perimeter_horse - perimeter_sheep = target_difference := by
  sorry

end perimeter_difference_l2180_218064


namespace five_segments_acute_angle_l2180_218028

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_obtuse (a b c : ℝ) : Prop :=
  c^2 > a^2 + b^2

def is_acute (a b c : ℝ) : Prop :=
  c^2 < a^2 + b^2

theorem five_segments_acute_angle (a b c d e : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (T1 : is_triangle a b c) (T2 : is_triangle a b d) (T3 : is_triangle a b e)
  (T4 : is_triangle a c d) (T5 : is_triangle a c e) (T6 : is_triangle a d e)
  (T7 : is_triangle b c d) (T8 : is_triangle b c e) (T9 : is_triangle b d e)
  (T10 : is_triangle c d e) : 
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧ 
           (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧ 
           (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
           is_triangle x y z ∧ is_acute x y z :=
by
  sorry

end five_segments_acute_angle_l2180_218028


namespace triangle_non_existent_l2180_218032

theorem triangle_non_existent (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (tangent_condition : (c^2) = 2 * (a^2) + 2 * (b^2)) : False := by
  sorry

end triangle_non_existent_l2180_218032


namespace tangent_line_at_1_l2180_218011

def f (x : ℝ) : ℝ := sorry

theorem tangent_line_at_1 (f' : ℝ → ℝ) (h1 : ∀ x, deriv f x = f' x) (h2 : ∀ y, 2 * 1 + y - 3 = 0) :
  f' 1 + f 1 = -1 :=
by
  sorry

end tangent_line_at_1_l2180_218011


namespace smallest_repeating_block_length_l2180_218015

-- Define the decimal expansion of 3/11
noncomputable def decimalExpansion : Rational → List Nat :=
  sorry

-- Define the repeating block determination of a given decimal expansion
noncomputable def repeatingBlockLength : List Nat → Nat :=
  sorry

-- Define the fraction 3/11
def frac := (3 : Rat) / 11

-- State the theorem
theorem smallest_repeating_block_length :
  repeatingBlockLength (decimalExpansion frac) = 2 :=
  sorry

end smallest_repeating_block_length_l2180_218015


namespace series_sum_is_correct_l2180_218031

noncomputable def series_sum : ℝ := ∑' k, 5^((2 : ℕ)^k) / (25^((2 : ℕ)^k) - 1)

theorem series_sum_is_correct : series_sum = 1 / (Real.sqrt 5 - 1) := 
by
  sorry

end series_sum_is_correct_l2180_218031


namespace light_travel_distance_in_km_l2180_218042

-- Define the conditions
def speed_of_light_miles_per_sec : ℝ := 186282
def conversion_factor_mile_to_km : ℝ := 1.609
def time_seconds : ℕ := 500
def expected_distance_km : ℝ := 1.498 * 10^8

-- The theorem we need to prove
theorem light_travel_distance_in_km :
  (speed_of_light_miles_per_sec * time_seconds * conversion_factor_mile_to_km) = expected_distance_km :=
  sorry

end light_travel_distance_in_km_l2180_218042


namespace option_B_option_D_l2180_218090

noncomputable def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- The maximum value of (y + 1) / (x + 1) is 2 + sqrt(6)
theorem option_B (x y : ℝ) (h : curve_C x y) :
  ∃ k, (y + 1) / (x + 1) = k ∧ k = 2 + Real.sqrt 6 :=
sorry

-- A tangent line through the point (0, √2) on curve C has the equation x - √2 * y + 2 = 0
theorem option_D (h : curve_C 0 (Real.sqrt 2)) :
  ∃ a b c, a * 0 + b * Real.sqrt 2 + c = 0 ∧ c = 2 ∧ a = 1 ∧ b = - Real.sqrt 2 :=
sorry

end option_B_option_D_l2180_218090


namespace find_inheritance_amount_l2180_218037

noncomputable def totalInheritance (tax_amount : ℕ) : ℕ :=
  let federal_rate := 0.20
  let state_rate := 0.10
  let combined_rate := federal_rate + (state_rate * (1 - federal_rate))
  sorry

theorem find_inheritance_amount : totalInheritance 10500 = 37500 := 
  sorry

end find_inheritance_amount_l2180_218037


namespace mean_volume_of_cubes_l2180_218080

theorem mean_volume_of_cubes (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 6) :
  ((a^3 + b^3 + c^3) / 3) = 135 :=
by
  -- known cube volumes and given edge lengths conditions
  sorry

end mean_volume_of_cubes_l2180_218080


namespace soda_cost_l2180_218062

theorem soda_cost (x : ℝ) : 
    (1.5 * 35 + x * (87 - 35) = 78.5) → 
    x = 0.5 := 
by 
  intros h
  sorry

end soda_cost_l2180_218062


namespace range_quadratic_function_l2180_218093

theorem range_quadratic_function : 
  ∀ y : ℝ, ∃ x : ℝ, y = x^2 - 2 * x + 5 ↔ y ∈ Set.Ici 4 :=
by 
  sorry

end range_quadratic_function_l2180_218093


namespace find_starting_point_of_a_l2180_218067

def point := ℝ × ℝ
def vector := ℝ × ℝ

def B : point := (1, 0)

def b : vector := (-3, -4)
def c : vector := (1, 1)

def a : vector := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)

theorem find_starting_point_of_a (hb : b = (-3, -4)) (hc : c = (1, 1)) (hB : B = (1, 0)) :
    let a := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)
    let start_A := (B.1 - a.1, B.2 - a.2)
    start_A = (12, 14) :=
by
  rw [hb, hc, hB]
  let a := (3 * (-3) - 2 * (1), 3 * (-4) - 2 * (1))
  let start_A := (1 - a.1, 0 - a.2)
  simp [a]
  sorry

end find_starting_point_of_a_l2180_218067


namespace cds_per_rack_l2180_218023

theorem cds_per_rack (total_cds : ℕ) (racks_per_shelf : ℕ) (cds_per_rack : ℕ) 
  (h1 : total_cds = 32) 
  (h2 : racks_per_shelf = 4) : 
  cds_per_rack = total_cds / racks_per_shelf :=
by 
  sorry

end cds_per_rack_l2180_218023


namespace isabel_money_left_l2180_218082

theorem isabel_money_left (initial_amount : ℕ) (half_toy_expense half_book_expense money_left : ℕ) :
  initial_amount = 204 →
  half_toy_expense = initial_amount / 2 →
  half_book_expense = (initial_amount - half_toy_expense) / 2 →
  money_left = initial_amount - half_toy_expense - half_book_expense →
  money_left = 51 :=
by
  intros h1 h2 h3 h4
  sorry

end isabel_money_left_l2180_218082


namespace units_digit_37_pow_37_l2180_218066

theorem units_digit_37_pow_37 : (37 ^ 37) % 10 = 7 := by
  -- The proof is omitted as per instructions.
  sorry

end units_digit_37_pow_37_l2180_218066


namespace gain_per_year_is_200_l2180_218022

noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem gain_per_year_is_200 :
  let borrowed_principal := 5000
  let borrowing_rate := 4
  let borrowing_time := 2
  let lent_principal := 5000
  let lending_rate := 8
  let lending_time := 2

  let interest_paid := simple_interest borrowed_principal borrowing_rate borrowing_time
  let interest_earned := simple_interest lent_principal lending_rate lending_time

  let total_gain := interest_earned - interest_paid
  let gain_per_year := total_gain / 2

  gain_per_year = 200 := by
  sorry

end gain_per_year_is_200_l2180_218022


namespace degrees_to_radians_l2180_218096

theorem degrees_to_radians (deg : ℝ) (rad : ℝ) (h1 : 1 = π / 180) (h2 : deg = 60) : rad = deg * (π / 180) :=
by
  sorry

end degrees_to_radians_l2180_218096


namespace min_degree_g_l2180_218084

open Polynomial

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := sorry

-- Conditions
axiom cond1 : 5 • f + 7 • g = h
axiom cond2 : natDegree f = 10
axiom cond3 : natDegree h = 12

-- Question: Minimum degree of g
theorem min_degree_g : natDegree g = 12 :=
sorry

end min_degree_g_l2180_218084


namespace residue_of_neg_1237_mod_37_l2180_218070

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by
  sorry

end residue_of_neg_1237_mod_37_l2180_218070


namespace sum_eq_sqrt_122_l2180_218087

theorem sum_eq_sqrt_122 
  (a b c : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h1 : a^2 + b^2 + c^2 = 58) 
  (h2 : a * b + b * c + c * a = 32) :
  a + b + c = Real.sqrt 122 := 
by
  sorry

end sum_eq_sqrt_122_l2180_218087


namespace general_term_of_geometric_sequence_l2180_218036

theorem general_term_of_geometric_sequence 
  (positive_terms : ∀ n : ℕ, 0 < a_n) 
  (h1 : a_1 = 1) 
  (h2 : ∃ a : ℕ, a_2 = a + 1 ∧ a_3 = 2 * a + 5) : 
  ∃ q : ℕ, ∀ n : ℕ, a_n = q^(n-1) :=
by
  sorry

end general_term_of_geometric_sequence_l2180_218036


namespace isabella_hair_length_l2180_218053

-- Define conditions: original length and doubled length
variable (original_length : ℕ)
variable (doubled_length : ℕ := 36)

-- Theorem: Prove that if the original length doubled equals 36, then the original length is 18.
theorem isabella_hair_length (h : 2 * original_length = doubled_length) : original_length = 18 := by
  sorry

end isabella_hair_length_l2180_218053


namespace POTOP_correct_l2180_218061

def POTOP : Nat := 51715

theorem POTOP_correct :
  (99999 * POTOP) % 1000 = 285 := by
  sorry

end POTOP_correct_l2180_218061


namespace axis_of_symmetry_of_shifted_function_l2180_218017

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem axis_of_symmetry_of_shifted_function :
  (∃ x : ℝ, g x = 1 ∧ x = Real.pi / 12) :=
by
  sorry

end axis_of_symmetry_of_shifted_function_l2180_218017


namespace length_of_de_equals_eight_l2180_218027

theorem length_of_de_equals_eight
  (a b c d e : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (bc : c - b = 3 * (d - c))
  (ab : b - a = 5)
  (ac : c - a = 11)
  (ae : e - a = 21) :
  e - d = 8 := by
  sorry

end length_of_de_equals_eight_l2180_218027


namespace no_stromino_covering_of_5x5_board_l2180_218002

-- Define the conditions
def isStromino (r : ℕ) (c : ℕ) : Prop := 
  (r = 3 ∧ c = 1) ∨ (r = 1 ∧ c = 3)

def is5x5Board (r c : ℕ) : Prop := 
  r = 5 ∧ c = 5

-- The main goal is to show this proposition
theorem no_stromino_covering_of_5x5_board : 
  ∀ (board_size : ℕ × ℕ),
    is5x5Board board_size.1 board_size.2 →
    ∀ (stromino_count : ℕ),
      stromino_count = 16 →
      (∀ (stromino : ℕ × ℕ), 
        isStromino stromino.1 stromino.2 →
        ∀ (cover : ℕ), 
          3 = cover) →
      ¬(∃ (cover_fn : ℕ × ℕ → ℕ), 
          (∀ (pos : ℕ × ℕ), pos.fst < 5 ∧ pos.snd < 5 →
            cover_fn pos = 1 ∨ cover_fn pos = 2) ∧
          (∀ (i : ℕ), i < 25 → 
            ∃ (stromino_pos : ℕ × ℕ), 
              stromino_pos.fst < 5 ∧ stromino_pos.snd < 5 ∧ 
              -- Each stromino must cover exactly 3 squares, 
              -- which implies that the covering function must work appropriately.
              (cover_fn (stromino_pos.fst, stromino_pos.snd) +
               cover_fn (stromino_pos.fst + 1, stromino_pos.snd) +
               cover_fn (stromino_pos.fst + 2, stromino_pos.snd) = 3 ∨
               cover_fn (stromino_pos.fst, stromino_pos.snd + 1) +
               cover_fn (stromino_pos.fst, stromino_pos.snd + 2) = 3))) :=
by sorry

end no_stromino_covering_of_5x5_board_l2180_218002


namespace find_a_value_l2180_218085

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 := 
by 
  sorry -- Placeholder for the proof

end find_a_value_l2180_218085


namespace smallest_even_number_l2180_218016

theorem smallest_even_number (x : ℤ) (h : (x + (x + 2) + (x + 4) + (x + 6)) = 140) : x = 32 :=
by
  sorry

end smallest_even_number_l2180_218016


namespace number_of_adults_in_family_l2180_218055

-- Conditions as definitions
def total_apples : ℕ := 1200
def number_of_children : ℕ := 45
def apples_per_child : ℕ := 15
def apples_per_adult : ℕ := 5

-- Calculations based on conditions
def apples_eaten_by_children : ℕ := number_of_children * apples_per_child
def remaining_apples : ℕ := total_apples - apples_eaten_by_children
def number_of_adults : ℕ := remaining_apples / apples_per_adult

-- Proof target: number of adults in Bob's family equals 105
theorem number_of_adults_in_family : number_of_adults = 105 := by
  sorry

end number_of_adults_in_family_l2180_218055


namespace smallest_d_factors_l2180_218075

theorem smallest_d_factors (d : ℕ) (h₁ : ∃ p q : ℤ, p * q = 2050 ∧ p + q = d ∧ p > 0 ∧ q > 0) :
    d = 107 :=
by
  sorry

end smallest_d_factors_l2180_218075


namespace imaginary_unit_sum_l2180_218077

theorem imaginary_unit_sum (i : ℂ) (H : i^4 = 1) : i^1234 + i^1235 + i^1236 + i^1237 = 0 :=
by
  sorry

end imaginary_unit_sum_l2180_218077


namespace quadratic_roots_properties_l2180_218058

-- Given the quadratic equation x^2 - 7x + 12 = 0
-- Prove that the absolute value of the difference of the roots is 1
-- Prove that the maximum value of the roots is 4

theorem quadratic_roots_properties :
  (∀ r1 r2 : ℝ, (r1 + r2 = 7) → (r1 * r2 = 12) → abs (r1 - r2) = 1) ∧ 
  (∀ r1 r2 : ℝ, (r1 + r2 = 7) → (r1 * r2 = 12) → max r1 r2 = 4) :=
by sorry

end quadratic_roots_properties_l2180_218058


namespace bird_counts_l2180_218040

theorem bird_counts :
  ∀ (num_cages_1 num_cages_2 num_cages_empty parrot_per_cage parakeet_per_cage canary_per_cage cockatiel_per_cage lovebird_per_cage finch_per_cage total_cages : ℕ),
    num_cages_1 = 7 →
    num_cages_2 = 6 →
    num_cages_empty = 2 →
    parrot_per_cage = 3 →
    parakeet_per_cage = 5 →
    canary_per_cage = 4 →
    cockatiel_per_cage = 2 →
    lovebird_per_cage = 3 →
    finch_per_cage = 1 →
    total_cages = 15 →
    (num_cages_1 * parrot_per_cage = 21) ∧
    (num_cages_1 * parakeet_per_cage = 35) ∧
    (num_cages_1 * canary_per_cage = 28) ∧
    (num_cages_2 * cockatiel_per_cage = 12) ∧
    (num_cages_2 * lovebird_per_cage = 18) ∧
    (num_cages_2 * finch_per_cage = 6) :=
by
  intros
  sorry

end bird_counts_l2180_218040


namespace circle_table_acquaintance_impossible_l2180_218050

theorem circle_table_acquaintance_impossible (P : Finset ℕ) (hP : P.card = 40) :
  ¬ (∀ (a b : ℕ), (a ∈ P) → (b ∈ P) → (∃ k, 2 * k ≠ 0) → (∃ c, c ∈ P) ∧ (a ≠ b) ∧ (c = a ∨ c = b)
       ↔ ¬(∃ k, 2 * k + 1 ≠ 0)) :=
by
  sorry

end circle_table_acquaintance_impossible_l2180_218050


namespace inequality_proof_l2180_218020

open Real

-- Given conditions
variables (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1)

-- Goal to prove
theorem inequality_proof : 
  (1 + x) * (1 + y) * (1 + z) ≥ 2 * (1 + (y / x)^(1/3) + (z / y)^(1/3) + (x / z)^(1/3)) :=
sorry

end inequality_proof_l2180_218020


namespace pow_add_div_eq_l2180_218081

   theorem pow_add_div_eq (a b c d e : ℕ) (h1 : b = 2) (h2 : c = 345) (h3 : d = 9) (h4 : e = 8 - 5) : 
     a = b^c + d^e -> a = 2^345 + 729 := 
   by 
     intros 
     sorry
   
end pow_add_div_eq_l2180_218081


namespace graph_symmetric_about_x_2_l2180_218025

variables {D : Set ℝ} {f : ℝ → ℝ}

theorem graph_symmetric_about_x_2 (h : ∀ x ∈ D, f (x + 1) = f (-x + 3)) : 
  ∀ x ∈ D, f (x) = f (4 - x) :=
by
  sorry

end graph_symmetric_about_x_2_l2180_218025


namespace eldorado_license_plates_count_l2180_218083

theorem eldorado_license_plates_count:
  let letters := 26
  let digits := 10
  let total := (letters ^ 3) * (digits ^ 4)
  total = 175760000 :=
by
  sorry

end eldorado_license_plates_count_l2180_218083


namespace cats_kittentotal_l2180_218046

def kittens_given_away : ℕ := 2
def kittens_now : ℕ := 6
def kittens_original : ℕ := 8

theorem cats_kittentotal : kittens_now + kittens_given_away = kittens_original := 
by 
  sorry

end cats_kittentotal_l2180_218046


namespace number_of_parents_who_volunteered_to_bring_refreshments_l2180_218014

theorem number_of_parents_who_volunteered_to_bring_refreshments 
  (total : ℕ) (supervise : ℕ) (supervise_and_refreshments : ℕ) (N : ℕ) (R : ℕ)
  (h_total : total = 84)
  (h_supervise : supervise = 25)
  (h_supervise_and_refreshments : supervise_and_refreshments = 11)
  (h_R_eq_1_5N : R = 3 * N / 2)
  (h_eq : total = (supervise - supervise_and_refreshments) + (R - supervise_and_refreshments) + supervise_and_refreshments + N) :
  R = 42 :=
by
  sorry

end number_of_parents_who_volunteered_to_bring_refreshments_l2180_218014


namespace find_y_l2180_218078

variables (x y : ℝ)

theorem find_y (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 :=
by
  sorry

end find_y_l2180_218078


namespace cards_remaining_l2180_218035

theorem cards_remaining (initial_cards : ℕ) (cards_given : ℕ) (remaining_cards : ℕ) :
  initial_cards = 242 → cards_given = 136 → remaining_cards = initial_cards - cards_given → remaining_cards = 106 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end cards_remaining_l2180_218035


namespace tony_squat_weight_l2180_218034

-- Definitions from conditions
def curl_weight := 90
def military_press_weight := 2 * curl_weight
def squat_weight := 5 * military_press_weight

-- Theorem statement
theorem tony_squat_weight : squat_weight = 900 := by
  sorry

end tony_squat_weight_l2180_218034


namespace emily_selects_green_apples_l2180_218044

theorem emily_selects_green_apples :
  let total_apples := 10
  let red_apples := 6
  let green_apples := 4
  let selected_apples := 3
  let total_combinations := Nat.choose total_apples selected_apples
  let green_combinations := Nat.choose green_apples selected_apples
  (green_combinations / total_combinations : ℚ) = 1 / 30 :=
by
  sorry

end emily_selects_green_apples_l2180_218044


namespace trips_Jean_l2180_218069

theorem trips_Jean (x : ℕ) (h1 : x + (x + 6) = 40) : x + 6 = 23 := by
  sorry

end trips_Jean_l2180_218069


namespace cos_of_angle_in_third_quadrant_l2180_218098

theorem cos_of_angle_in_third_quadrant (B : ℝ) (h1 : π < B ∧ B < 3 * π / 2) (h2 : Real.sin B = -5 / 13) : Real.cos B = -12 / 13 := 
by 
  sorry

end cos_of_angle_in_third_quadrant_l2180_218098


namespace curve_representation_l2180_218010

   theorem curve_representation :
     ∀ (x y : ℝ), x^4 - y^4 - 4*x^2 + 4*y^2 = 0 ↔ (x + y = 0 ∨ x - y = 0 ∨ x^2 + y^2 = 4) :=
   by
     sorry
   
end curve_representation_l2180_218010


namespace second_part_of_ratio_l2180_218095

theorem second_part_of_ratio (first_part : ℝ) (whole second_part : ℝ) (h1 : first_part = 5) (h2 : first_part / whole = 25 / 100) : second_part = 15 :=
by
  sorry

end second_part_of_ratio_l2180_218095


namespace complex_number_value_l2180_218072

open Complex

theorem complex_number_value (a : ℝ) 
  (h1 : z = (2 + a * I) / (1 + I)) 
  (h2 : (z.re, z.im) ∈ { p : ℝ × ℝ | p.2 = -p.1 }) : 
  a = 0 :=
by
  sorry

end complex_number_value_l2180_218072


namespace ratio_Binkie_Frankie_eq_4_l2180_218039

-- Definitions based on given conditions
def SpaatzGems : ℕ := 1
def BinkieGems : ℕ := 24

-- Assume the number of gemstones on Frankie's collar
variable (FrankieGems : ℕ)

-- Given condition about the gemstones on Spaatz's collar
axiom SpaatzCondition : SpaatzGems = (FrankieGems / 2) - 2

-- The theorem to be proved
theorem ratio_Binkie_Frankie_eq_4 
    (FrankieGems : ℕ) 
    (SpaatzCondition : SpaatzGems = (FrankieGems / 2) - 2) 
    (BinkieGems_eq : BinkieGems = 24) 
    (SpaatzGems_eq : SpaatzGems = 1) 
    (f_nonzero : FrankieGems ≠ 0) :
    BinkieGems / FrankieGems = 4 :=
by
  sorry  -- We're only writing the statement, not the proof.

end ratio_Binkie_Frankie_eq_4_l2180_218039


namespace integer_solutions_to_abs_equation_l2180_218047

theorem integer_solutions_to_abs_equation :
  {p : ℤ × ℤ | abs (p.1 - 2) + abs (p.2 - 1) = 1} =
  {(3, 1), (1, 1), (2, 2), (2, 0)} :=
by
  sorry

end integer_solutions_to_abs_equation_l2180_218047


namespace best_fitting_model_l2180_218089

/-- Four models with different coefficients of determination -/
def model1_R2 : ℝ := 0.98
def model2_R2 : ℝ := 0.80
def model3_R2 : ℝ := 0.50
def model4_R2 : ℝ := 0.25

/-- Prove that Model 1 has the best fitting effect among the given models -/
theorem best_fitting_model :
  model1_R2 > model2_R2 ∧ model1_R2 > model3_R2 ∧ model1_R2 > model4_R2 :=
by {sorry}

end best_fitting_model_l2180_218089


namespace find_salary_month_l2180_218079

variable (J F M A May : ℝ)

def condition_1 : Prop := (J + F + M + A) / 4 = 8000
def condition_2 : Prop := (F + M + A + May) / 4 = 8450
def condition_3 : Prop := J = 4700
def condition_4 (X : ℝ) : Prop := X = 6500

theorem find_salary_month (J F M A May : ℝ) 
  (h1 : condition_1 J F M A) 
  (h2 : condition_2 F M A May) 
  (h3 : condition_3 J) 
  : ∃ M : ℝ, condition_4 May :=
by sorry

end find_salary_month_l2180_218079


namespace circle_area_solution_l2180_218076

def circle_area_problem : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + 6 * x - 8 * y - 12 = 0 -> ∃ (A : ℝ), A = 37 * Real.pi

theorem circle_area_solution : circle_area_problem :=
by
  sorry

end circle_area_solution_l2180_218076


namespace laptop_sticker_price_l2180_218026

theorem laptop_sticker_price (x : ℝ) (h₁ : 0.70 * x = 0.80 * x - 50 - 30) : x = 800 := 
  sorry

end laptop_sticker_price_l2180_218026


namespace sum_of_octal_numbers_l2180_218013

theorem sum_of_octal_numbers :
  (176 : ℕ) + 725 + 63 = 1066 := by
sorry

end sum_of_octal_numbers_l2180_218013


namespace modulus_of_complex_raised_to_eight_l2180_218001

-- Define the complex number 2 + i in Lean
def z : Complex := Complex.mk 2 1

-- State the proof problem with conditions
theorem modulus_of_complex_raised_to_eight : Complex.abs (z ^ 8) = 625 := by
  sorry

end modulus_of_complex_raised_to_eight_l2180_218001


namespace length_of_AP_in_right_triangle_l2180_218041

theorem length_of_AP_in_right_triangle 
  (A B C : ℝ × ℝ)
  (hA : A = (0, 2))
  (hB : B = (0, 0))
  (hC : C = (2, 0))
  (M : ℝ × ℝ)
  (hM : M.1 = 0 ∧ M.2 = 0)
  (inc : ℝ × ℝ)
  (hinc : inc = (1, 1)) :
  ∃ P : ℝ × ℝ, (P.1 = 0 ∧ P.2 = 1) ∧ dist A P = 1 := by
  sorry

end length_of_AP_in_right_triangle_l2180_218041


namespace Sandwiches_count_l2180_218049

-- Define the number of toppings and the number of choices for the patty
def num_toppings : Nat := 10
def num_choices_per_topping : Nat := 2
def num_patties : Nat := 3

-- Define the theorem to prove the total number of sandwiches
theorem Sandwiches_count : (num_choices_per_topping ^ num_toppings) * num_patties = 3072 :=
by
  sorry

end Sandwiches_count_l2180_218049


namespace find_k_and_prove_geometric_sequence_l2180_218019

/-
Given conditions:
1. Sequence sa : ℕ → ℝ with sum sequence S : ℕ → ℝ satisfying the recurrence relation S (n + 1) = (k + 1) * S n + 2
2. Initial terms a_1 = 2 and a_2 = 1
-/

def sequence_sum_relation (S : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n : ℕ, S (n + 1) = (k + 1) * S n + 2

def init_sequence_terms (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ a 2 = 1

/-
Proof goal:
1. Prove k = -1/2 given the conditions.
2. Prove sequence a is a geometric sequence with common ratio 1/2 given the conditions.
-/

theorem find_k_and_prove_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ) :
  sequence_sum_relation S k →
  init_sequence_terms a →
  (k = (-1:ℝ)/2) ∧ (∀ n: ℕ, n ≥ 1 → a (n+1) = (1/2) * a n) :=
by
  sorry

end find_k_and_prove_geometric_sequence_l2180_218019


namespace mary_screws_sections_l2180_218071

def number_of_sections (initial_screws : Nat) (multiplier : Nat) (screws_per_section : Nat) : Nat :=
  let additional_screws := initial_screws * multiplier
  let total_screws := initial_screws + additional_screws
  total_screws / screws_per_section

theorem mary_screws_sections :
  number_of_sections 8 2 6 = 4 := by
  sorry

end mary_screws_sections_l2180_218071


namespace form_eleven_form_twelve_form_thirteen_form_fourteen_form_fifteen_form_sixteen_form_seventeen_form_eighteen_form_nineteen_form_twenty_l2180_218045

theorem form_eleven : 22 - (2 + (2 / 2)) = 11 := by
  sorry

theorem form_twelve : (2 * 2 * 2) - 2 / 2 = 12 := by
  sorry

theorem form_thirteen : (22 + 2 + 2) / 2 = 13 := by
  sorry

theorem form_fourteen : 2 * 2 * 2 * 2 - 2 = 14 := by
  sorry

theorem form_fifteen : (2 * 2)^2 - 2 / 2 = 15 := by
  sorry

theorem form_sixteen : (2 * 2)^2 * (2 / 2) = 16 := by
  sorry

theorem form_seventeen : (2 * 2)^2 + 2 / 2 = 17 := by
  sorry

theorem form_eighteen : 2 * 2 * 2 * 2 + 2 = 18 := by
  sorry

theorem form_nineteen : 22 - 2 - 2 / 2 = 19 := by
  sorry

theorem form_twenty : (22 - 2) * (2 / 2) = 20 := by
  sorry

end form_eleven_form_twelve_form_thirteen_form_fourteen_form_fifteen_form_sixteen_form_seventeen_form_eighteen_form_nineteen_form_twenty_l2180_218045


namespace adam_bought_26_books_l2180_218004

-- Conditions
def initial_books : ℕ := 56
def shelves : ℕ := 4
def avg_books_per_shelf : ℕ := 20
def leftover_books : ℕ := 2

-- Definitions based on conditions
def capacity_books : ℕ := shelves * avg_books_per_shelf
def total_books_after_trip : ℕ := capacity_books + leftover_books

-- Question: How many books did Adam buy on his shopping trip?
def books_bought : ℕ := total_books_after_trip - initial_books

theorem adam_bought_26_books :
  books_bought = 26 :=
by
  sorry

end adam_bought_26_books_l2180_218004


namespace systematic_sampling_sequence_l2180_218029

theorem systematic_sampling_sequence :
  ∃ (s : Set ℕ), s = {3, 13, 23, 33, 43} ∧
  (∀ n, n ∈ s → n ≤ 50 ∧ ∃ k, k < 5 ∧ n = 3 + k * 10) :=
by
  sorry

end systematic_sampling_sequence_l2180_218029


namespace ratio_of_volumes_l2180_218059

noncomputable def volumeSphere (p : ℝ) : ℝ := (4/3) * Real.pi * (p^3)

noncomputable def volumeHemisphere (p : ℝ) : ℝ := (1/2) * (4/3) * Real.pi * (3*p)^3

theorem ratio_of_volumes (p : ℝ) (hp : p > 0) : volumeSphere p / volumeHemisphere p = 2 / 27 :=
by
  sorry

end ratio_of_volumes_l2180_218059


namespace problem1_l2180_218092

theorem problem1 {a m n : ℝ} (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + n) = 6 := by
  sorry

end problem1_l2180_218092


namespace f_1992_eq_1992_l2180_218054

def f (x : ℕ) : ℤ := sorry

theorem f_1992_eq_1992 (f : ℕ → ℤ) 
  (h1 : ∀ x : ℕ, 0 < x -> f x = f (x - 1) + f (x + 1))
  (h2 : f 0 = 1992) :
  f 1992 = 1992 := 
sorry

end f_1992_eq_1992_l2180_218054


namespace jasmine_cookies_l2180_218018

theorem jasmine_cookies (J : ℕ) (h1 : 20 + J + (J + 10) = 60) : J = 15 :=
sorry

end jasmine_cookies_l2180_218018


namespace geometric_sequence_ratio_l2180_218065

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ)
  (hq_pos : 0 < q)
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_arith : 2 * (1/2) * a 2 = 3 * a 0 + 2 * a 1) :
  (a 10 + a 12) / (a 7 + a 9) = 27 :=
sorry

end geometric_sequence_ratio_l2180_218065


namespace number_of_possible_values_of_k_l2180_218051

-- Define the primary conditions and question
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def quadratic_roots_prime (p q k : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p + q = 72 ∧ p * q = k

theorem number_of_possible_values_of_k :
  ¬ ∃ k : ℕ, ∃ p q : ℕ, quadratic_roots_prime p q k :=
by
  sorry

end number_of_possible_values_of_k_l2180_218051


namespace f_is_constant_l2180_218048

noncomputable def f (x θ : ℝ) : ℝ :=
  (Real.cos (x - θ))^2 + (Real.cos x)^2 - 2 * Real.cos θ * Real.cos (x - θ) * Real.cos x

theorem f_is_constant (θ : ℝ) : ∀ x, f x θ = (Real.sin θ)^2 :=
by
  intro x
  sorry

end f_is_constant_l2180_218048


namespace arithmetic_seq_a7_a8_l2180_218003

theorem arithmetic_seq_a7_a8 (a : ℕ → ℤ) (d : ℤ) (h₁ : a 1 + a 2 = 4) (h₂ : d = 2) :
  a 7 + a 8 = 28 := by
  sorry

end arithmetic_seq_a7_a8_l2180_218003


namespace min_sum_of_factors_240_l2180_218091

theorem min_sum_of_factors_240 :
  ∃ a b : ℕ, a * b = 240 ∧ (∀ a' b' : ℕ, a' * b' = 240 → a + b ≤ a' + b') ∧ a + b = 31 :=
sorry

end min_sum_of_factors_240_l2180_218091


namespace cylinder_inscribed_in_sphere_l2180_218052

noncomputable def sphere_volume (r : ℝ) : ℝ := 
  (4 / 3) * Real.pi * r^3

theorem cylinder_inscribed_in_sphere 
  (r_cylinder : ℝ)
  (h₁ : r_cylinder > 0)
  (height_cylinder : ℝ)
  (radius_sphere : ℝ)
  (h₂ : radius_sphere = r_cylinder + 2)
  (h₃ : height_cylinder = r_cylinder + 1)
  (h₄ : 2 * radius_sphere = Real.sqrt ((2 * r_cylinder)^2 + (height_cylinder)^2))
  : sphere_volume 17 = 6550 * 2 / 3 * Real.pi :=
by
  -- solution steps and proof go here
  sorry

end cylinder_inscribed_in_sphere_l2180_218052


namespace ratio_of_boys_to_total_l2180_218094

theorem ratio_of_boys_to_total (b : ℝ) (h1 : b = 3 / 4 * (1 - b)) : b = 3 / 7 :=
by
  {
    -- The given condition (we use it to prove the target statement)
    sorry
  }

end ratio_of_boys_to_total_l2180_218094


namespace grinder_price_l2180_218038

variable (G : ℝ) (PurchasedMobile : ℝ) (SoldMobile : ℝ) (overallProfit : ℝ)

theorem grinder_price (h1 : PurchasedMobile = 10000)
                      (h2 : SoldMobile = 11000)
                      (h3 : overallProfit = 400)
                      (h4 : 0.96 * G + SoldMobile = G + PurchasedMobile + overallProfit) :
                      G = 15000 := by
  sorry

end grinder_price_l2180_218038


namespace find_R_when_S_7_l2180_218000

-- Define the variables and equations in Lean
variables (R S g : ℕ)

-- The theorem statement based on the given conditions and desired conclusion
theorem find_R_when_S_7 (h1 : R = 2 * g * S + 3) (h2: R = 23) (h3 : S = 5) : (∃ g : ℕ, R = 2 * g * 7 + 3) :=
by {
  -- This part enforces the proof will be handled later
  sorry
}

end find_R_when_S_7_l2180_218000


namespace base8_9257_digits_product_sum_l2180_218012

theorem base8_9257_digits_product_sum :
  let base10 := 9257
  let base8_digits := [2, 2, 0, 5, 1] -- base 8 representation of 9257
  let product_of_digits := 2 * 2 * 0 * 5 * 1
  let sum_of_digits := 2 + 2 + 0 + 5 + 1
  product_of_digits = 0 ∧ sum_of_digits = 10 := 
by
  sorry

end base8_9257_digits_product_sum_l2180_218012


namespace intersecting_chords_l2180_218086

theorem intersecting_chords (n : ℕ) (h1 : 0 < n) :
  ∃ intersecting_points : ℕ, intersecting_points ≥ n :=
  sorry

end intersecting_chords_l2180_218086


namespace part1_part2_l2180_218030

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) :=
by
  sorry

end part1_part2_l2180_218030


namespace exists_solution_in_interval_l2180_218043

noncomputable def f (x : ℝ) : ℝ := x^3 - 2^x

theorem exists_solution_in_interval : ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ f x = 0 :=
by {
  -- Use the Intermediate Value Theorem, given f is continuous on [1, 2]
  sorry
}

end exists_solution_in_interval_l2180_218043


namespace darla_total_payment_l2180_218006

-- Define the cost per watt, total watts used, and late fee
def cost_per_watt : ℝ := 4
def total_watts : ℝ := 300
def late_fee : ℝ := 150

-- Define the total cost of electricity
def electricity_cost : ℝ := cost_per_watt * total_watts

-- Define the total amount Darla needs to pay
def total_amount : ℝ := electricity_cost + late_fee

-- The theorem to prove the total amount equals $1350
theorem darla_total_payment : total_amount = 1350 := by
  sorry

end darla_total_payment_l2180_218006


namespace div_gcd_iff_div_ab_gcd_mul_l2180_218024

variable (a b n c : ℕ)
variables (h₀ : a ≠ 0) (d : ℕ)
variable (hd : d = Nat.gcd a b)

theorem div_gcd_iff_div_ab : (n ∣ a ∧ n ∣ b) ↔ n ∣ d :=
by
  sorry

theorem gcd_mul (h₁ : c > 0) : Nat.gcd (a * c) (b * c) = c * Nat.gcd a b :=
by
  sorry

end div_gcd_iff_div_ab_gcd_mul_l2180_218024


namespace mark_owes_joanna_l2180_218008

def dollars_per_room : ℚ := 12 / 3
def rooms_cleaned : ℚ := 9 / 4
def total_amount_owed : ℚ := 9

theorem mark_owes_joanna :
  dollars_per_room * rooms_cleaned = total_amount_owed :=
by
  sorry

end mark_owes_joanna_l2180_218008


namespace abc_sum_zero_l2180_218007

variable (a b c : ℝ)

-- Conditions given in the original problem
axiom h1 : a + b / c = 1
axiom h2 : b + c / a = 1
axiom h3 : c + a / b = 1

theorem abc_sum_zero : a * b + b * c + c * a = 0 :=
by
  sorry

end abc_sum_zero_l2180_218007
