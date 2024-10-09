import Mathlib

namespace intersection_point_l1055_105517

theorem intersection_point (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ d) :
  let x := (d - c) / (2 * b)
  let y := (a * (d - c)^2) / (4 * b^2) + (d + c) / 2
  (ax^2 + bx + c = y) ∧ (ax^2 - bx + d = y) :=
by
  sorry

end intersection_point_l1055_105517


namespace divisor_of_p_l1055_105596

theorem divisor_of_p (p q r s : ℕ) (h₁ : Nat.gcd p q = 30) (h₂ : Nat.gcd q r = 45) (h₃ : Nat.gcd r s = 75) (h₄ : 120 < Nat.gcd s p) (h₅ : Nat.gcd s p < 180) : 5 ∣ p := 
sorry

end divisor_of_p_l1055_105596


namespace simplify_fraction_1_210_plus_17_35_l1055_105571

theorem simplify_fraction_1_210_plus_17_35 :
  1 / 210 + 17 / 35 = 103 / 210 :=
by sorry

end simplify_fraction_1_210_plus_17_35_l1055_105571


namespace pet_center_final_count_l1055_105595

def initial_dogs : Nat := 36
def initial_cats : Nat := 29
def adopted_dogs : Nat := 20
def collected_cats : Nat := 12
def final_pets : Nat := 57

theorem pet_center_final_count :
  (initial_dogs - adopted_dogs) + (initial_cats + collected_cats) = final_pets := 
by
  sorry

end pet_center_final_count_l1055_105595


namespace angle_measure_l1055_105570

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l1055_105570


namespace projective_iff_fractional_linear_l1055_105562

def projective_transformation (P : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a * d - b * c ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d))

theorem projective_iff_fractional_linear (P : ℝ → ℝ) : 
  projective_transformation P ↔ ∃ (a b c d : ℝ), (a * d - b * c ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d)) :=
by 
  sorry

end projective_iff_fractional_linear_l1055_105562


namespace calculation_proof_l1055_105534

theorem calculation_proof
    (a : ℝ) (b : ℝ) (c : ℝ)
    (h1 : a = 3.6)
    (h2 : b = 0.25)
    (h3 : c = 0.5) :
    (a * b) / c = 1.8 := 
by
  sorry

end calculation_proof_l1055_105534


namespace triangular_weight_60_grams_l1055_105537

-- Define the weights as variables
variables {R T : ℝ} -- round weights and triangular weights are real numbers

-- Define the conditions as hypotheses
theorem triangular_weight_60_grams
  (h1 : R + T = 3 * R)
  (h2 : 4 * R + T = T + R + 90) :
  T = 60 :=
by
  -- indicate that the actual proof is omitted
  sorry

end triangular_weight_60_grams_l1055_105537


namespace toad_difference_l1055_105574

variables (Tim_toads Jim_toads Sarah_toads : ℕ)

theorem toad_difference (h1 : Tim_toads = 30) 
                        (h2 : Jim_toads > Tim_toads) 
                        (h3 : Sarah_toads = 2 * Jim_toads) 
                        (h4 : Sarah_toads = 100) :
  Jim_toads - Tim_toads = 20 :=
by
  -- The next lines are placeholders for the logical steps which need to be proven
  sorry

end toad_difference_l1055_105574


namespace cost_per_dvd_l1055_105500

theorem cost_per_dvd (total_cost : ℝ) (num_dvds : ℕ) 
  (h1 : total_cost = 4.8) (h2 : num_dvds = 4) : (total_cost / num_dvds) = 1.2 :=
by
  sorry

end cost_per_dvd_l1055_105500


namespace problem_l1055_105519

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if h : (-1 : ℝ) ≤ x ∧ x < 0 then a*x + 1
else if h : (0 : ℝ) ≤ x ∧ x ≤ 1 then (b*x + 2) / (x + 1)
else 0 -- This should not matter as we only care about the given ranges

theorem problem (a b : ℝ) (h₁ : f 0.5 a b = f 1.5 a b) : a + 3 * b = -10 :=
by
  -- We'll derive equations from given conditions and prove the result.
  sorry

end problem_l1055_105519


namespace distribute_positions_l1055_105559

structure DistributionProblem :=
  (volunteer_positions : ℕ)
  (schools : ℕ)
  (min_positions : ℕ)
  (distinct_allocations : ∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c)

noncomputable def count_ways (p : DistributionProblem) : ℕ :=
  if p.volunteer_positions = 7 ∧ p.schools = 3 ∧ p.min_positions = 1 then 6 else 0

theorem distribute_positions (p : DistributionProblem) :
  count_ways p = 6 :=
by
  sorry

end distribute_positions_l1055_105559


namespace cube_painting_l1055_105554

-- Let's start with importing Mathlib for natural number operations

theorem cube_painting (n : ℕ) (h : 2 < n)
  (num_one_black_face : ℕ := 3 * (n - 2)^2)
  (num_unpainted : ℕ := (n - 2)^3) :
  num_one_black_face = num_unpainted → n = 5 :=
by
  sorry

end cube_painting_l1055_105554


namespace cannot_form_triangle_l1055_105525

theorem cannot_form_triangle (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  ¬ ∃ a b c : ℕ, (a, b, c) = (1, 2, 3) := 
  sorry

end cannot_form_triangle_l1055_105525


namespace soccer_season_length_l1055_105524

def total_games : ℕ := 27
def games_per_month : ℕ := 9
def months_in_season : ℕ := total_games / games_per_month

theorem soccer_season_length : months_in_season = 3 := by
  unfold months_in_season
  unfold total_games
  unfold games_per_month
  sorry

end soccer_season_length_l1055_105524


namespace central_angle_of_sector_l1055_105550

theorem central_angle_of_sector (r l : ℝ) (h1 : r = 1) (h2 : l = 4 - 2*r) : 
    ∃ α : ℝ, α = 2 :=
by
  use l / r
  have hr : r = 1 := h1
  have hl : l = 4 - 2*r := h2
  sorry

end central_angle_of_sector_l1055_105550


namespace initial_inventory_correct_l1055_105531

-- Define the conditions as given in the problem
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_wed_to_sun : ℕ := 50
def days_wed_to_sun : ℕ := 5
def bottles_delivered_saturday : ℕ := 650
def final_inventory : ℕ := 1555

-- Define the total number of bottles sold during the week
def total_bottles_sold : ℕ :=
  bottles_sold_monday + bottles_sold_tuesday + (bottles_sold_per_day_wed_to_sun * days_wed_to_sun)

-- Define the initial inventory calculation
def initial_inventory : ℕ :=
  final_inventory + total_bottles_sold - bottles_delivered_saturday

-- The theorem we want to prove
theorem initial_inventory_correct :
  initial_inventory = 4500 :=
by
  sorry

end initial_inventory_correct_l1055_105531


namespace find_remaining_rectangle_area_l1055_105506

-- Definitions of given areas
def S_DEIH : ℝ := 20
def S_HILK : ℝ := 40
def S_ABHG : ℝ := 126
def S_GHKJ : ℝ := 63
def S_DFMK : ℝ := 161

-- Definition of areas of the remaining rectangle
def S_EFML : ℝ := 101

-- Theorem statement to prove the area of the remaining rectangle
theorem find_remaining_rectangle_area :
  S_DFMK - S_DEIH - S_HILK = S_EFML :=
by
  -- This is where the proof would go
  sorry

end find_remaining_rectangle_area_l1055_105506


namespace abs_inequalities_imply_linear_relationship_l1055_105572

theorem abs_inequalities_imply_linear_relationship (a b c : ℝ)
(h1 : |a - b| ≥ |c|)
(h2 : |b - c| ≥ |a|)
(h3 : |c - a| ≥ |b|) :
a = b + c ∨ b = c + a ∨ c = a + b :=
sorry

end abs_inequalities_imply_linear_relationship_l1055_105572


namespace a1_a9_sum_l1055_105503

noncomputable def arithmetic_sequence (a: ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem a1_a9_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a3_a7_roots : (a 3 = 3 ∧ a 7 = -1) ∨ (a 3 = -1 ∧ a 7 = 3)) :
  a 1 + a 9 = 2 :=
by
  sorry

end a1_a9_sum_l1055_105503


namespace compare_negative_positive_l1055_105545

theorem compare_negative_positive : -897 < 0.01 := sorry

end compare_negative_positive_l1055_105545


namespace min_tablets_to_ensure_three_each_l1055_105556

theorem min_tablets_to_ensure_three_each (A B C : ℕ) (hA : A = 20) (hB : B = 25) (hC : C = 15) : 
  ∃ n, n = 48 ∧ (∀ x y z, x + y + z = n → x ≥ 3 ∧ y ≥ 3 ∧ z ≥ 3) :=
by
  -- proof goes here
  sorry

end min_tablets_to_ensure_three_each_l1055_105556


namespace Andy_is_late_l1055_105557

def school_start_time : Nat := 8 * 60 -- in minutes (8:00 AM)
def normal_travel_time : Nat := 30 -- in minutes
def delay_red_lights : Nat := 4 * 3 -- in minutes (4 red lights * 3 minutes each)
def delay_construction : Nat := 10 -- in minutes
def delay_detour_accident : Nat := 7 -- in minutes
def delay_store_stop : Nat := 5 -- in minutes
def delay_searching_store : Nat := 2 -- in minutes
def delay_traffic : Nat := 15 -- in minutes
def delay_neighbor_help : Nat := 6 -- in minutes
def delay_closed_road : Nat := 8 -- in minutes
def all_delays : Nat := delay_red_lights + delay_construction + delay_detour_accident + delay_store_stop + delay_searching_store + delay_traffic + delay_neighbor_help + delay_closed_road
def departure_time : Nat := 7 * 60 + 15 -- in minutes (7:15 AM)

def arrival_time : Nat := departure_time + normal_travel_time + all_delays
def late_minutes : Nat := arrival_time - school_start_time

theorem Andy_is_late : late_minutes = 50 := by
  sorry

end Andy_is_late_l1055_105557


namespace product_of_midpoint_coordinates_l1055_105568

theorem product_of_midpoint_coordinates
  (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 4) (h2 : y1 = -3) (h3 : x2 = -8) (h4 : y2 = 7) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx * my = -4) :=
by
  -- Here we would carry out the proof.
  sorry

end product_of_midpoint_coordinates_l1055_105568


namespace hyperbola_ellipse_b_value_l1055_105504

theorem hyperbola_ellipse_b_value (a c b : ℝ) (h1 : c = 5 * a / 4) (h2 : c^2 - a^2 = (9 * a^2) / 16) (h3 : 4 * (b^2 - 4) = 16 * b^2 / 25) :
  b = 6 / 5 ∨ b = 10 / 3 :=
by
  sorry

end hyperbola_ellipse_b_value_l1055_105504


namespace eval_sequence_l1055_105553

noncomputable def b : ℕ → ℤ
| 1 => 1
| 2 => 4
| 3 => 9
| n => if h : n > 3 then b (n - 1) * (b (n - 1) - 1) + 1 else 0

theorem eval_sequence :
  b 1 * b 2 * b 3 * b 4 * b 5 * b 6 - (b 1 ^ 2 + b 2 ^ 2 + b 3 ^ 2 + b 4 ^ 2 + b 5 ^ 2 + b 6 ^ 2)
  = -3166598256 :=
by
  /- The proof steps are omitted. -/
  sorry

end eval_sequence_l1055_105553


namespace area_of_absolute_value_sum_l1055_105518

theorem area_of_absolute_value_sum :
  ∃ area : ℝ, (area = 80) ∧ (∀ x y : ℝ, |2 * x| + |5 * y| = 20 → area = 80) :=
by
  sorry

end area_of_absolute_value_sum_l1055_105518


namespace prime_719_exists_l1055_105538

theorem prime_719_exists (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) :
  (a^4 + b^4 + c^4 - 3 = 719) → Nat.Prime (a^4 + b^4 + c^4 - 3) := sorry

end prime_719_exists_l1055_105538


namespace cos_decreasing_intervals_l1055_105583

open Real

def is_cos_decreasing_interval (k : ℤ) : Prop := 
  let f (x : ℝ) := cos (π / 4 - 2 * x)
  ∀ x y : ℝ, (k * π + π / 8 ≤ x) → (x ≤ k * π + 5 * π / 8) → 
             (k * π + π / 8 ≤ y) → (y ≤ k * π + 5 * π / 8) → 
             x < y → f x > f y

theorem cos_decreasing_intervals : ∀ k : ℤ, is_cos_decreasing_interval k :=
by
  sorry

end cos_decreasing_intervals_l1055_105583


namespace inequality_proof_l1055_105597

theorem inequality_proof (a b c : ℝ) 
    (ha : a > 1) (hb : b > 1) (hc : c > 1) :
    (a^2 / (b - 1)) + (b^2 / (c - 1)) + (c^2 / (a - 1)) ≥ 12 :=
by {
    sorry
}

end inequality_proof_l1055_105597


namespace remainder_17_plus_x_mod_31_l1055_105513

theorem remainder_17_plus_x_mod_31 {x : ℕ} (h : 13 * x ≡ 3 [MOD 31]) : (17 + x) % 31 = 22 := 
sorry

end remainder_17_plus_x_mod_31_l1055_105513


namespace parabola_inequality_l1055_105521

theorem parabola_inequality (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 2 * k * x + (k^2 + 2 * k + 2) > x^2 + 2 * k * x - 2 * k^2 - 1) ↔ (-1 < k ∧ k < 3) := 
sorry

end parabola_inequality_l1055_105521


namespace bobby_pancakes_left_l1055_105540

theorem bobby_pancakes_left (initial_pancakes : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) :
  initial_pancakes = 21 → bobby_ate = 5 → dog_ate = 7 → initial_pancakes - (bobby_ate + dog_ate) = 9 :=
by
  intros h1 h2 h3
  sorry

end bobby_pancakes_left_l1055_105540


namespace ellipse_area_l1055_105586

def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 9 * y^2 - 36 * y + 36 = 0

theorem ellipse_area :
  (∀ x y : ℝ, ellipse_equation x y → true) →
  (π * 1 * (4/3) = 4 * π / 3) :=
by
  intro h
  norm_num
  sorry

end ellipse_area_l1055_105586


namespace logarithmic_inequality_and_integral_l1055_105526

theorem logarithmic_inequality_and_integral :
  let a := Real.log 3 / Real.log 2
  let b := Real.log 2 / Real.log 3
  let c := 2 / Real.pi^2
  a > b ∧ b > c :=
by
  let a := Real.log 3 / Real.log 2
  let b := Real.log 2 / Real.log 3
  let c := 2 / Real.pi^2
  sorry

end logarithmic_inequality_and_integral_l1055_105526


namespace tino_more_jellybeans_than_lee_l1055_105566

-- Declare the conditions
variables (arnold_jellybeans lee_jellybeans tino_jellybeans : ℕ)
variables (arnold_jellybeans_half_lee : arnold_jellybeans = lee_jellybeans / 2)
variables (arnold_jellybean_count : arnold_jellybeans = 5)
variables (tino_jellybean_count : tino_jellybeans = 34)

-- The goal is to prove how many more jellybeans Tino has than Lee
theorem tino_more_jellybeans_than_lee : tino_jellybeans - lee_jellybeans = 24 :=
by
  sorry -- proof skipped

end tino_more_jellybeans_than_lee_l1055_105566


namespace foreign_students_next_sem_eq_740_l1055_105587

def total_students : ℕ := 1800
def percentage_foreign : ℕ := 30
def new_foreign_students : ℕ := 200

def initial_foreign_students : ℕ := total_students * percentage_foreign / 100
def total_foreign_students_next_semester : ℕ :=
  initial_foreign_students + new_foreign_students

theorem foreign_students_next_sem_eq_740 :
  total_foreign_students_next_semester = 740 :=
by
  sorry

end foreign_students_next_sem_eq_740_l1055_105587


namespace seq_a_seq_b_l1055_105529

theorem seq_a (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  a 1 = 1 ∧ (∀ n, S (n + 1) = 3 * S n + 2) →
  (∀ n, a n = if n = 1 then 1 else 4 * 3 ^ (n - 2)) :=
by
  sorry

theorem seq_b (b : ℕ → ℕ) (a : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ) :
  (b n = 8 * n / (a (n + 1) - a n)) →
  (T n = 77 / 12 - (n / 2 + 3 / 4) * (1 / 3) ^ (n - 2)) :=
by
  sorry

end seq_a_seq_b_l1055_105529


namespace find_three_digit_number_l1055_105547

theorem find_three_digit_number (A B C D : ℕ) 
  (h1 : A + C = 5) 
  (h2 : B = 3)
  (h3 : A * 100 + B * 10 + C + 124 = D * 111) 
  (h4 : A ≠ B ∧ A ≠ C ∧ B ≠ C) : 
  A * 100 + B * 10 + C = 431 := 
by 
  sorry

end find_three_digit_number_l1055_105547


namespace sum_expression_l1055_105508

theorem sum_expression : 3 * 501 + 2 * 501 + 4 * 501 + 500 = 5009 := by
  sorry

end sum_expression_l1055_105508


namespace problem_solution_l1055_105592

theorem problem_solution : (275^2 - 245^2) / 30 = 520 := by
  sorry

end problem_solution_l1055_105592


namespace fraction_reach_impossible_l1055_105579

theorem fraction_reach_impossible :
  ¬ ∃ (a b : ℕ), (2 + 2013 * a) / (3 + 2014 * b) = 3 / 5 := by
  sorry

end fraction_reach_impossible_l1055_105579


namespace circular_sequence_zero_if_equidistant_l1055_105515

noncomputable def circular_sequence_property (x y z : ℤ): Prop :=
  (x = 0 ∧ y = 0 ∧ dist x y = dist y z) → z = 0

theorem circular_sequence_zero_if_equidistant {x y z : ℤ} :
  (x = 0 ∧ y = 0 ∧ dist x y = dist y z) → z = 0 :=
by sorry

end circular_sequence_zero_if_equidistant_l1055_105515


namespace five_letter_words_start_end_same_l1055_105532

def num_five_letter_words_start_end_same : ℕ :=
  26 ^ 4

theorem five_letter_words_start_end_same :
  num_five_letter_words_start_end_same = 456976 :=
by
  -- Sorry is used as a placeholder for the proof.
  sorry

end five_letter_words_start_end_same_l1055_105532


namespace mask_digits_l1055_105577

theorem mask_digits : 
  ∃ (elephant mouse pig panda : ℕ), 
  (elephant ≠ mouse ∧ elephant ≠ pig ∧ elephant ≠ panda ∧ 
   mouse ≠ pig ∧ mouse ≠ panda ∧ pig ≠ panda) ∧
  (4 * 4 = 16) ∧ (7 * 7 = 49) ∧ (8 * 8 = 64) ∧ (9 * 9 = 81) ∧
  (elephant = 6) ∧ (mouse = 4) ∧ (pig = 8) ∧ (panda = 1) :=
by
  sorry

end mask_digits_l1055_105577


namespace min_disks_to_store_files_l1055_105584

open Nat

theorem min_disks_to_store_files :
  ∃ minimum_disks : ℕ,
    (minimum_disks = 24) ∧
    ∀ (files : ℕ) (disk_capacity : ℕ) (file_sizes : List ℕ),
      files = 36 →
      disk_capacity = 144 →
      (∃ (size_85 : ℕ) (size_75 : ℕ) (size_45 : ℕ),
         size_85 = 5 ∧
         size_75 = 15 ∧
         size_45 = 16 ∧
         (∀ (disks : ℕ), disks >= minimum_disks →
            ∃ (used_disks_85 : ℕ) (remaining_files_45 : ℕ) (used_disks_45 : ℕ) (used_disks_75 : ℕ),
              remaining_files_45 = size_45 - used_disks_85 ∧
              used_disks_85 = size_85 ∧
              (remaining_files_45 % 3 = 0 → used_disks_45 = remaining_files_45 / 3) ∧
              (remaining_files_45 % 3 ≠ 0 → used_disks_45 = remaining_files_45 / 3 + 1) ∧
              used_disks_75 = size_75 ∧
              disks = used_disks_85 + used_disks_45 + used_disks_75)) :=
by
  sorry

end min_disks_to_store_files_l1055_105584


namespace find_a1_plus_a9_l1055_105511

variable (a : ℕ → ℝ) (d : ℝ)

-- condition: arithmetic sequence
def is_arithmetic_seq : Prop := ∀ n, a (n + 1) = a n + d

-- condition: sum of specific terms
def sum_specific_terms : Prop := a 3 + a 4 + a 5 + a 6 + a 7 = 450

-- theorem: prove the desired sum
theorem find_a1_plus_a9 (h1 : is_arithmetic_seq a d) (h2 : sum_specific_terms a) : 
  a 1 + a 9 = 180 :=
  sorry

end find_a1_plus_a9_l1055_105511


namespace n_plus_d_is_155_l1055_105512

noncomputable def n_and_d_sum : Nat :=
sorry

theorem n_plus_d_is_155 (n d : Nat) (hn : 0 < n) (hd : d < 10) 
  (h1 : 4 * n^2 + 2 * n + d = 305) 
  (h2 : 4 * n^3 + 2 * n^2 + d * n + 1 = 577 + 8 * d) : n + d = 155 := 
sorry

end n_plus_d_is_155_l1055_105512


namespace simplify_expression_l1055_105535

-- Define the initial expression
def expr (q : ℚ) := (5 * q^4 - 4 * q^3 + 7 * q - 8) + (3 - 5 * q^2 + q^3 - 2 * q)

-- Define the simplified expression
def simplified_expr (q : ℚ) := 5 * q^4 - 3 * q^3 - 5 * q^2 + 5 * q - 5

-- The theorem stating that the two expressions are equal
theorem simplify_expression (q : ℚ) : expr q = simplified_expr q :=
by
  sorry

end simplify_expression_l1055_105535


namespace abigail_lost_money_l1055_105564

theorem abigail_lost_money (initial_amount spent_first_store spent_second_store remaining_amount_lost: ℝ) 
  (h_initial : initial_amount = 50) 
  (h_spent_first : spent_first_store = 15.25) 
  (h_spent_second : spent_second_store = 8.75) 
  (h_remaining : remaining_amount_lost = 16) : (initial_amount - spent_first_store - spent_second_store - remaining_amount_lost = 10) :=
by
  sorry

end abigail_lost_money_l1055_105564


namespace S_ploughing_time_l1055_105501

theorem S_ploughing_time (R S : ℝ) (hR_rate : R = 1 / 15) (h_combined_rate : R + S = 1 / 10) : S = 1 / 30 := sorry

end S_ploughing_time_l1055_105501


namespace p_squared_plus_41_composite_for_all_primes_l1055_105555

theorem p_squared_plus_41_composite_for_all_primes (p : ℕ) (hp : Prime p) : 
  ∃ d : ℕ, d > 1 ∧ d < p^2 + 41 ∧ d ∣ (p^2 + 41) :=
by
  sorry

end p_squared_plus_41_composite_for_all_primes_l1055_105555


namespace lesser_solution_of_quadratic_eq_l1055_105536

theorem lesser_solution_of_quadratic_eq : ∃ x ∈ {x | x^2 + 10*x - 24 = 0}, x = -12 :=
by 
  sorry

end lesser_solution_of_quadratic_eq_l1055_105536


namespace polynomial_roots_sum_l1055_105591

theorem polynomial_roots_sum (p q : ℂ) (hp : p + q = 5) (hq : p * q = 7) : 
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 559 := 
by 
  sorry

end polynomial_roots_sum_l1055_105591


namespace sum_smallest_largest_eq_2z_l1055_105581

theorem sum_smallest_largest_eq_2z (m b z : ℤ) (h1 : m > 0) (h2 : z = (b + (b + 2 * (m - 1))) / 2) :
  b + (b + 2 * (m - 1)) = 2 * z :=
sorry

end sum_smallest_largest_eq_2z_l1055_105581


namespace determine_pairs_l1055_105528

theorem determine_pairs (p : ℕ) (hp: Nat.Prime p) :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ p^x - y^3 = 1 ∧ ((x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2)) := 
sorry

end determine_pairs_l1055_105528


namespace rectangle_breadth_l1055_105589

theorem rectangle_breadth (sq_area : ℝ) (rect_area : ℝ) (radius_rect_relation : ℝ → ℝ) 
  (rect_length_relation : ℝ → ℝ) (breadth_correct: ℝ) : 
  (sq_area = 3600) →
  (rect_area = 240) →
  (forall r, radius_rect_relation r = r) →
  (forall r, rect_length_relation r = (2/5) * r) →
  breadth_correct = 10 :=
by
  intros h_sq_area h_rect_area h_radius_rect h_rect_length
  sorry

end rectangle_breadth_l1055_105589


namespace tessa_owes_30_l1055_105598

-- Definitions based on given conditions
def initial_debt : ℕ := 40
def paid_back : ℕ := initial_debt / 2
def remaining_debt_after_payment : ℕ := initial_debt - paid_back
def additional_borrowing : ℕ := 10
def total_debt : ℕ := remaining_debt_after_payment + additional_borrowing

-- Theorem to be proved
theorem tessa_owes_30 : total_debt = 30 :=
by
  sorry

end tessa_owes_30_l1055_105598


namespace cube_vertices_probability_l1055_105567

theorem cube_vertices_probability (totalVertices : ℕ) (selectedVertices : ℕ) 
   (totalCombinations : ℕ) (favorableOutcomes : ℕ) : 
   totalVertices = 8 ∧ selectedVertices = 4 ∧ totalCombinations = 70 ∧ favorableOutcomes = 12 → 
   (favorableOutcomes : ℚ) / totalCombinations = 6 / 35 := by
   sorry

end cube_vertices_probability_l1055_105567


namespace addition_example_l1055_105542

theorem addition_example : 0.4 + 56.7 = 57.1 := by
  -- Here we need to prove the main statement
  sorry

end addition_example_l1055_105542


namespace expand_expression_l1055_105505

theorem expand_expression (x : ℝ) : (x - 3) * (x + 6) = x^2 + 3 * x - 18 :=
by
  sorry

end expand_expression_l1055_105505


namespace remaining_tickets_equation_l1055_105594

-- Define the constants and variables
variables (x y : ℕ)

-- Conditions from the problem
def tickets_whack_a_mole := 32
def tickets_skee_ball := 25
def tickets_space_invaders : ℕ := x

def spent_hat := 7
def spent_keychain := 10
def spent_toy := 15

-- Define the condition for the total number of tickets spent
def total_tickets_spent := spent_hat + spent_keychain + spent_toy
-- Prove the remaining tickets equation
theorem remaining_tickets_equation : y = (tickets_whack_a_mole + tickets_skee_ball + tickets_space_invaders) - total_tickets_spent ->
                                      y = 25 + x :=
by
  sorry

end remaining_tickets_equation_l1055_105594


namespace Rebecca_tent_stakes_l1055_105563

theorem Rebecca_tent_stakes : 
  ∃ T D W : ℕ, 
    D = 3 * T ∧ 
    W = T + 2 ∧ 
    T + D + W = 22 ∧ 
    T = 4 := 
by
  sorry

end Rebecca_tent_stakes_l1055_105563


namespace eden_stuffed_bears_l1055_105585

theorem eden_stuffed_bears
  (initial_bears : ℕ)
  (favorite_bears : ℕ)
  (sisters : ℕ)
  (eden_initial_bears : ℕ)
  (remaining_bears := initial_bears - favorite_bears)
  (bears_per_sister := remaining_bears / sisters)
  (eden_bears_now := eden_initial_bears + bears_per_sister)
  (h1 : initial_bears = 20)
  (h2 : favorite_bears = 8)
  (h3 : sisters = 3)
  (h4 : eden_initial_bears = 10) :
  eden_bears_now = 14 := by
{
  sorry
}

end eden_stuffed_bears_l1055_105585


namespace distinct_factors_1320_l1055_105599

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l1055_105599


namespace fraction_defined_range_l1055_105514

theorem fraction_defined_range (x : ℝ) : 
  (∃ y, y = 3 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_defined_range_l1055_105514


namespace find_dividend_l1055_105527

theorem find_dividend (divisor quotient remainder : ℕ) (h_divisor : divisor = 38) (h_quotient : quotient = 19) (h_remainder : remainder = 7) :
  divisor * quotient + remainder = 729 := by
  sorry

end find_dividend_l1055_105527


namespace number_of_players_l1055_105551

-- Definitions of the conditions
def initial_bottles : ℕ := 4 * 12
def bottles_remaining : ℕ := 15
def bottles_taken_per_player : ℕ := 2 + 1

-- Total number of bottles taken
def bottles_taken := initial_bottles - bottles_remaining

-- The main theorem stating that the number of players is 11.
theorem number_of_players : (bottles_taken / bottles_taken_per_player) = 11 :=
by
  sorry

end number_of_players_l1055_105551


namespace john_paid_correct_amount_l1055_105507

def cost_bw : ℝ := 160
def markup_percentage : ℝ := 0.5

def cost_color : ℝ := cost_bw * (1 + markup_percentage)

theorem john_paid_correct_amount : 
  cost_color = 240 := 
by
  -- proof required here
  sorry

end john_paid_correct_amount_l1055_105507


namespace senior_citizen_ticket_cost_l1055_105530

theorem senior_citizen_ticket_cost 
  (total_tickets : ℕ)
  (regular_ticket_cost : ℕ)
  (total_sales : ℕ)
  (sold_regular_tickets : ℕ)
  (x : ℕ)
  (h1 : total_tickets = 65)
  (h2 : regular_ticket_cost = 15)
  (h3 : total_sales = 855)
  (h4 : sold_regular_tickets = 41)
  (h5 : total_sales = (sold_regular_tickets * regular_ticket_cost) + ((total_tickets - sold_regular_tickets) * x)) :
  x = 10 :=
by
  sorry

end senior_citizen_ticket_cost_l1055_105530


namespace arithmetic_mean_of_16_23_38_and_11_point_5_is_22_point_125_l1055_105561

theorem arithmetic_mean_of_16_23_38_and_11_point_5_is_22_point_125 :
  (16 + 23 + 38 + 11.5) / 4 = 22.125 :=
by
  sorry

end arithmetic_mean_of_16_23_38_and_11_point_5_is_22_point_125_l1055_105561


namespace randy_piggy_bank_l1055_105552

theorem randy_piggy_bank : 
  ∀ (initial_amount trips_per_month cost_per_trip months_per_year total_spent_left : ℕ),
  initial_amount = 200 →
  cost_per_trip = 2 →
  trips_per_month = 4 →
  months_per_year = 12 →
  total_spent_left = initial_amount - (cost_per_trip * trips_per_month * months_per_year) →
  total_spent_left = 104 :=
by
  intros initial_amount trips_per_month cost_per_trip months_per_year total_spent_left
  sorry

end randy_piggy_bank_l1055_105552


namespace man_l1055_105522

-- Given conditions
def V_m := 15 - 3.2
def V_c := 3.2
def man's_speed_with_current : Real := 15

-- Required to prove
def man's_speed_against_current := V_m - V_c

theorem man's_speed_against_current_is_correct : man's_speed_against_current = 8.6 := by
  sorry

end man_l1055_105522


namespace Rogers_age_more_than_twice_Jills_age_l1055_105502

/--
Jill is 20 years old.
Finley is 40 years old.
Roger's age is more than twice Jill's age.
In 15 years, the age difference between Roger and Jill will be 30 years less than Finley's age.
Prove that Roger's age is 5 years more than twice Jill's age.
-/
theorem Rogers_age_more_than_twice_Jills_age 
  (J F : ℕ) (hJ : J = 20) (hF : F = 40) (R x : ℕ)
  (hR : R = 2 * J + x) 
  (age_diff_condition : (R + 15) - (J + 15) = (F + 15) - 30) :
  x = 5 := 
sorry

end Rogers_age_more_than_twice_Jills_age_l1055_105502


namespace distance_traveled_by_car_l1055_105520

theorem distance_traveled_by_car (total_distance : ℕ) (fraction_foot : ℚ) (fraction_bus : ℚ)
  (h_total : total_distance = 40) (h_fraction_foot : fraction_foot = 1/4)
  (h_fraction_bus : fraction_bus = 1/2) :
  (total_distance * (1 - fraction_foot - fraction_bus)) = 10 :=
by
  sorry

end distance_traveled_by_car_l1055_105520


namespace oranges_count_l1055_105544

theorem oranges_count (N : ℕ) (k : ℕ) (m : ℕ) (j : ℕ) :
  (N ≡ 2 [MOD 10]) ∧ (N ≡ 0 [MOD 12]) → N = 72 :=
by
  sorry

end oranges_count_l1055_105544


namespace matrix_power_is_correct_l1055_105509

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]
def A_cubed : Matrix (Fin 2) (Fin 2) ℤ := !![3, -6; 6, -3]

theorem matrix_power_is_correct : A ^ 3 = A_cubed := by 
  sorry

end matrix_power_is_correct_l1055_105509


namespace annual_fixed_costs_l1055_105549

theorem annual_fixed_costs
  (profit : ℝ := 30500000)
  (selling_price : ℝ := 9035)
  (variable_cost : ℝ := 5000)
  (units_sold : ℕ := 20000) :
  ∃ (fixed_costs : ℝ), profit = (selling_price * units_sold) - (variable_cost * units_sold) - fixed_costs :=
sorry

end annual_fixed_costs_l1055_105549


namespace black_car_speed_l1055_105576

theorem black_car_speed
  (red_speed black_speed : ℝ)
  (initial_distance time : ℝ)
  (red_speed_eq : red_speed = 10)
  (initial_distance_eq : initial_distance = 20)
  (time_eq : time = 0.5)
  (distance_eq : black_speed * time = initial_distance + red_speed * time) :
  black_speed = 50 := by
  rw [red_speed_eq, initial_distance_eq, time_eq] at distance_eq
  sorry

end black_car_speed_l1055_105576


namespace fraction_students_walk_home_l1055_105560

theorem fraction_students_walk_home :
  let bus := 1/3
  let auto := 1/5
  let bicycle := 1/8
  let total_students := 1
  let other_transportation := bus + auto + bicycle
  let walk_home := total_students - other_transportation
  walk_home = 41/120 :=
by 
  let bus := 1/3
  let auto := 1/5
  let bicycle := 1/8
  let total_students := 1
  let other_transportation := bus + auto + bicycle
  let walk_home := total_students - other_transportation
  have h_bus : bus = 40 / 120 := by sorry
  have h_auto : auto = 24 / 120 := by sorry
  have h_bicycle : bicycle = 15 / 120 := by sorry
  have h_total_transportation : other_transportation = 40 / 120 + 24 / 120 + 15 / 120 := by sorry
  have h_other_transportation_sum : other_transportation = 79 / 120 := by sorry
  have h_walk_home : walk_home = 1 - 79 / 120 := by sorry
  have h_walk_home_simplified : walk_home = 41 / 120 := by sorry
  exact h_walk_home_simplified

end fraction_students_walk_home_l1055_105560


namespace worst_player_is_nephew_l1055_105575

-- Define the family members
inductive Player
| father : Player
| sister : Player
| son : Player
| nephew : Player

open Player

-- Define a twin relationship
def is_twin (p1 p2 : Player) : Prop :=
  (p1 = son ∧ p2 = nephew) ∨ (p1 = nephew ∧ p2 = son)

-- Define that two players are of opposite sex
def opposite_sex (p1 p2 : Player) : Prop :=
  (p1 = sister ∧ (p2 = father ∨ p2 = son ∨ p2 = nephew)) ∨
  (p2 = sister ∧ (p1 = father ∨ p1 = son ∨ p1 = nephew))

-- Predicate for the worst player
structure WorstPlayer (p : Player) : Prop :=
  (twin_exists : ∃ twin : Player, is_twin p twin)
  (opposite_sex_best : ∀ twin best, is_twin p twin → best ≠ twin → opposite_sex twin best)

-- The goal is to show that the worst player is the nephew
theorem worst_player_is_nephew : WorstPlayer nephew := sorry

end worst_player_is_nephew_l1055_105575


namespace hotel_flat_fee_l1055_105516

theorem hotel_flat_fee (f n : ℝ) (h1 : f + n = 120) (h2 : f + 6 * n = 330) : f = 78 :=
by
  sorry

end hotel_flat_fee_l1055_105516


namespace journey_time_proof_l1055_105569

noncomputable def journey_time_on_wednesday (d s x : ℝ) : ℝ :=
  d / s

theorem journey_time_proof (d s x : ℝ) (usual_speed_nonzero : s ≠ 0) :
  (journey_time_on_wednesday d s x) = 11 * x :=
by
  have thursday_speed : ℝ := 1.1 * s
  have thursday_time : ℝ := d / thursday_speed
  have time_diff : ℝ := (d / s) - thursday_time
  have reduced_time_eq_x : time_diff = x := by sorry
  have journey_time_eq : (d / s) = 11 * x := by sorry
  exact journey_time_eq

end journey_time_proof_l1055_105569


namespace find_length_of_AB_l1055_105510

theorem find_length_of_AB (x y : ℝ) (AP PB AQ QB PQ AB : ℝ) 
  (h1 : AP = 3 * x) 
  (h2 : PB = 4 * x) 
  (h3 : AQ = 4 * y) 
  (h4 : QB = 5 * y)
  (h5 : PQ = 5) 
  (h6 : AP + PB = AB)
  (h7 : AQ + QB = AB)
  (h8 : PQ = AQ - AP)
  (h9 : 7 * x = 9 * y) : 
  AB = 315 := 
by
  sorry

end find_length_of_AB_l1055_105510


namespace smallest_integer_in_consecutive_set_l1055_105565

theorem smallest_integer_in_consecutive_set :
  ∃ (n : ℤ), 2 < n ∧ ∀ m : ℤ, m < n → ¬ (m + 6 < 2 * (m + 3) - 2) :=
sorry

end smallest_integer_in_consecutive_set_l1055_105565


namespace combined_cost_price_l1055_105539

theorem combined_cost_price :
  let stock1_price := 100
  let stock1_discount := 5 / 100
  let stock1_brokerage := 1.5 / 100
  let stock2_price := 200
  let stock2_discount := 7 / 100
  let stock2_brokerage := 0.75 / 100
  let stock3_price := 300
  let stock3_discount := 3 / 100
  let stock3_brokerage := 1 / 100

  -- Calculated values
  let stock1_discounted_price := stock1_price * (1 - stock1_discount)
  let stock1_total_price := stock1_discounted_price * (1 + stock1_brokerage)
  
  let stock2_discounted_price := stock2_price * (1 - stock2_discount)
  let stock2_total_price := stock2_discounted_price * (1 + stock2_brokerage)
  
  let stock3_discounted_price := stock3_price * (1 - stock3_discount)
  let stock3_total_price := stock3_discounted_price * (1 + stock3_brokerage)
  
  let combined_cost := stock1_total_price + stock2_total_price + stock3_total_price
  combined_cost = 577.73 := sorry

end combined_cost_price_l1055_105539


namespace average_not_1380_l1055_105582

-- Define the set of numbers
def numbers := [1200, 1400, 1510, 1520, 1530, 1200]

-- Define the claimed average
def claimed_avg := 1380

-- The sum of the numbers
def sumNumbers := numbers.sum

-- The number of items in the set
def countNumbers := numbers.length

-- The correct average calculation
def correct_avg : ℚ := sumNumbers / countNumbers

-- The proof problem: proving that the correct average is not equal to the claimed average
theorem average_not_1380 : correct_avg ≠ claimed_avg := by
  sorry

end average_not_1380_l1055_105582


namespace lcm_12_18_l1055_105580

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end lcm_12_18_l1055_105580


namespace concert_attendance_difference_l1055_105588

noncomputable def first_concert : ℕ := 65899
noncomputable def second_concert : ℕ := 66018

theorem concert_attendance_difference :
  (second_concert - first_concert) = 119 :=
by
  sorry

end concert_attendance_difference_l1055_105588


namespace alligators_not_hiding_l1055_105590

theorem alligators_not_hiding (total_alligators hiding_alligators : ℕ) 
  (h1 : total_alligators = 75) 
  (h2 : hiding_alligators = 19) : 
  total_alligators - hiding_alligators = 56 :=
by
  -- The proof will go here, which is currently a placeholder.
  sorry

end alligators_not_hiding_l1055_105590


namespace sam_walked_distance_l1055_105578

theorem sam_walked_distance
  (distance_apart : ℝ) (fred_speed : ℝ) (sam_speed : ℝ) (t : ℝ)
  (H1 : distance_apart = 35) (H2 : fred_speed = 2) (H3 : sam_speed = 5)
  (H4 : 2 * t + 5 * t = distance_apart) :
  5 * t = 25 :=
by
  -- Lean proof goes here
  sorry

end sam_walked_distance_l1055_105578


namespace work_last_duration_l1055_105543

theorem work_last_duration
  (work_rate_x : ℚ := 1 / 20)
  (work_rate_y : ℚ := 1 / 12)
  (days_x_worked_alone : ℚ := 4)
  (combined_work_rate : ℚ := work_rate_x + work_rate_y)
  (remaining_work : ℚ := 1 - days_x_worked_alone * work_rate_x) :
  (remaining_work / combined_work_rate + days_x_worked_alone = 10) :=
by
  sorry

end work_last_duration_l1055_105543


namespace g_84_value_l1055_105533

-- Define the function g with the given conditions
def g (x : ℝ) : ℝ := sorry

-- Conditions given in the problem
axiom g_property1 : ∀ x y : ℝ, g (x * y) = y * g x
axiom g_property2 : g 2 = 48

-- Statement to prove
theorem g_84_value : g 84 = 2016 :=
by
  sorry

end g_84_value_l1055_105533


namespace selected_numbers_in_range_l1055_105593

noncomputable def systematic_sampling (n_students selected_students interval_num start_num n : ℕ) : ℕ :=
  start_num + interval_num * (n - 1)

theorem selected_numbers_in_range (x : ℕ) :
  (500 = 500) ∧ (50 = 50) ∧ (10 = 500 / 50) ∧ (6 ∈ {y : ℕ | 1 ≤ y ∧ y ≤ 10}) ∧ (125 ≤ x ∧ x ≤ 140) → 
  (x = systematic_sampling 500 50 10 6 13 ∨ x = systematic_sampling 500 50 10 6 14) :=
by
  sorry

end selected_numbers_in_range_l1055_105593


namespace final_amount_after_5_years_l1055_105573

-- Define conditions as hypotheses
def principal := 200
def final_amount_after_2_years := 260
def time_2_years := 2

-- Define our final question and answer as a Lean theorem
theorem final_amount_after_5_years : 
  (final_amount_after_2_years - principal) = principal * (rate * time_2_years) →
  (rate * 3) = 90 →
  final_amount_after_2_years + (principal * rate * 3) = 350 :=
by
  intros h1 h2
  -- Proof skipped using sorry
  sorry

end final_amount_after_5_years_l1055_105573


namespace probability_divisor_of_60_l1055_105541

theorem probability_divisor_of_60 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 60 ∧ (∃ a b c : ℕ, n = 2 ^ a * 3 ^ b * 5 ^ c ∧ a ≤ 2 ∧ b ≤ 1 ∧ c ≤ 1)) → 
  ∃ p : ℚ, p = 1 / 5 :=
by
  sorry

end probability_divisor_of_60_l1055_105541


namespace price_returns_to_initial_l1055_105558

theorem price_returns_to_initial (x : ℝ) (h : 0.918 * (100 + x) = 100) : x = 9 := 
by
  sorry

end price_returns_to_initial_l1055_105558


namespace equidistant_point_x_coord_l1055_105546

theorem equidistant_point_x_coord :
  ∃ x y : ℝ, y = x ∧ dist (x, y) (x, 0) = dist (x, y) (0, y) ∧ dist (x, y) (0, y) = dist (x, y) (x, 5 - x)
    → x = 5 / 2 :=
by sorry

end equidistant_point_x_coord_l1055_105546


namespace max_marks_l1055_105548

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 59 + 40) : M = 300 :=
by
  sorry

end max_marks_l1055_105548


namespace intersection_A_B_l1055_105523

def A : Set ℤ := {-2, -1, 1, 2}

def B : Set ℤ := {x | x^2 - x - 2 ≥ 0}

theorem intersection_A_B : (A ∩ B) = {-2, -1, 2} := by
  sorry

end intersection_A_B_l1055_105523
